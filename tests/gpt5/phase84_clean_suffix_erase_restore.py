from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import torch


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))

from hf_probe_env import get_layers, release_loaded  # noqa: E402
from phase68_object_attribute_natural_exchange import load_model, parse_csv  # noqa: E402
from phase70_object_relation_value_closure import parse_layer_pairs  # noqa: E402
from phase72_object_relation_value_fullseq_closure import capture_state, stats_from_scores  # noqa: E402
from phase75_relation_frame_token_intervention import get_frame_positions  # noqa: E402
from phase76_object_frame_joint_closure import uniq  # noqa: E402
from phase77_balanced_cross_relation_joint_closure import build_expanded_items  # noqa: E402
from phase79_rank_sweep_remainder_audit import fullseq_logprob_rank_patch  # noqa: E402
from phase83_suffix_token_decomposition import build_suffix_token_bases  # noqa: E402


def log(msg: str = "") -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def avg(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def zero_source(dim: int) -> torch.Tensor:
    return torch.zeros(dim, dtype=torch.float32)


def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [v for v in vals if v["base_clean_rank"] == 1]
    return {
        "n": len(vals),
        "eligible_n": len(eligible),
        "clean_drop": avg([float(v["clean_drop"]) for v in vals]),
        "restore_gain": avg([float(v["restore_gain"]) for v in vals]),
        "restore_gap": avg([float(v["restore_gap"]) for v in vals]),
        "erase_top1": avg([1.0 if v["erase_rank"] == 1 else 0.0 for v in vals]),
        "restore_top1": avg([1.0 if v["restore_rank"] == 1 else 0.0 for v in vals]),
        "eligible_clean_drop": avg([float(v["clean_drop"]) for v in eligible]),
        "eligible_restore_gain": avg([float(v["restore_gain"]) for v in eligible]),
        "eligible_restore_gap": avg([float(v["restore_gap"]) for v in eligible]),
        "eligible_erase_top1": avg([1.0 if v["erase_rank"] == 1 else 0.0 for v in eligible]),
        "eligible_restore_top1": avg([1.0 if v["restore_rank"] == 1 else 0.0 for v in eligible]),
        "eligible_base_margin": avg([float(v["base_clean_margin"]) for v in eligible]),
        "eligible_erase_margin": avg([float(v["erase_margin"]) for v in eligible]),
        "eligible_restore_margin": avg([float(v["restore_margin"]) for v in eligible]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_condition_path: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    by_condition_relation: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        c = str(row["condition"])
        rel = str(row["relation"])
        dl, rl = int(row["destroy_layer"]), int(row["restore_layer"])
        by_condition[c].append(row)
        by_condition_path[(c, dl, rl)].append(row)
        by_condition_relation[(c, rel)].append(row)
    return {
        "by_condition": {k: group_summary(v) for k, v in by_condition.items()},
        "by_condition_path": {f"{c}:L{dl}->L{rl}": group_summary(v) for (c, dl, rl), v in by_condition_path.items()},
        "by_condition_relation": {f"{c}:{rel}": group_summary(v) for (c, rel), v in by_condition_relation.items()},
    }


def add_row(
    results: dict[str, Any],
    item: dict[str, Any],
    idx: int,
    destroy_layer: int,
    restore_layer: int,
    cond: str,
    base_stats: dict[str, Any],
    erase_scores: dict[str, float],
    restore_scores: dict[str, float],
    candidates: list[str],
) -> None:
    es = stats_from_scores(erase_scores, item["target"], [v for v in candidates if v != item["target"]])
    rs = stats_from_scores(restore_scores, item["target"], [v for v in candidates if v != item["target"]])
    results["rows"].append({
        "item_idx": idx,
        "destroy_layer": destroy_layer,
        "restore_layer": restore_layer,
        "condition": cond,
        "relation": item["relation"],
        "frame_key": item["frame_key"],
        "base_clean_margin": base_stats["margin"],
        "erase_margin": es["margin"],
        "restore_margin": rs["margin"],
        "clean_drop": base_stats["margin"] - es["margin"],
        "restore_gain": rs["margin"] - es["margin"],
        "restore_gap": base_stats["margin"] - rs["margin"],
        "base_clean_rank": base_stats["rank"],
        "erase_rank": es["rank"],
        "restore_rank": rs["rank"],
    })


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE84_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    layer_pairs = parse_layer_pairs(args.layer_pairs)
    items = build_expanded_items(args.max_items, parse_csv(args.relations), parse_csv(args.frames))
    log(f"Phase84 model={args.model} items={len(items)} layer_pairs={layer_pairs} contrast_rank={args.contrast_rank} component_rank={args.component_rank}")

    results: dict[str, Any] = {
        "phase": 84,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "clean_suffix_erase_restore",
        "layer_pairs": layer_pairs,
        "module": args.module,
        "contrast_rank": args.contrast_rank,
        "component_rank": args.component_rank,
        "max_basis_items": args.max_basis_items,
        "relations": sorted({x["relation"] for x in items}),
        "num_items": len(items),
        "rows": [],
        "summary": {},
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_phase84_clean_suffix_erase_restore.json"
    partial_path = out_dir / f"{args.model}_phase84_clean_suffix_erase_restore.partial.json"
    if args.resume:
        resume_path = final_path if final_path.exists() else partial_path
        if resume_path.exists():
            loaded = json.loads(resume_path.read_text(encoding="utf-8"))
            if loaded.get("model") == args.model and loaded.get("phase") == 84:
                results = loaded
                results.setdefault("rows", [])
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")

    completed: set[tuple[int, int, int]] = set()
    counts: dict[tuple[int, int, int], int] = defaultdict(int)
    for row in results["rows"]:
        counts[(int(row["destroy_layer"]), int(row["restore_layer"]), int(row["item_idx"]))] += 1
    for k, v in counts.items():
        if v >= 12:
            completed.add(k)

    t0 = time.time()
    for destroy_layer, restore_layer in layer_pairs:
        log(f"building suffix bases for L{destroy_layer} and L{restore_layer}")
        bases_d = build_suffix_token_bases(model, tokenizer, device, layers, items, destroy_layer, args.module, args.max_length, args.contrast_rank, args.component_rank, args.max_basis_items)
        bases_r = build_suffix_token_bases(model, tokenizer, device, layers, items, restore_layer, args.module, args.max_length, args.contrast_rank, args.component_rank, args.max_basis_items)
        log(f"bases ready for {destroy_layer}->{restore_layer}")
        for idx, item in enumerate(items):
            if (destroy_layer, restore_layer, idx) in completed:
                continue
            clean_pos = get_frame_positions(tokenizer, item["clean_prompt"], item["object"])
            if any(x is None for x in (clean_pos.get("object_last"), clean_pos.get("frame_last"))):
                continue
            clean_distractors = [x["target"] for x in items if x["target"] != item["target"] and x["relation"] == item["relation"]]
            candidates = uniq([item["target"]] + clean_distractors[: args.max_distractors])
            base_scores = {
                v: fullseq_logprob_rank_patch(model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length, args.module)
                for v in candidates
            }
            base_stats = stats_from_scores(base_scores, item["target"], [v for v in candidates if v != item["target"]])
            h_clean_r = capture_state(model, tokenizer, device, layers[restore_layer], args.module, item["clean_prompt"], args.max_length)
            op, fp = int(clean_pos["object_last"]), int(clean_pos["frame_last"])
            dim = int(h_clean_r.shape[-1])
            z = zero_source(dim)
            restore_obj = h_clean_r[op]
            restore_frame = h_clean_r[fp]
            cond_bases = {
                "suffix_all": "suffix_all_basis",
                "suffix_final": "suffix_final_basis",
                "suffix_nonfinal": "suffix_nonfinal_basis",
                "suffix_function": "suffix_function_basis",
                "suffix_lexical": "suffix_lexical_basis",
                "all_suffix_tokens": "all_suffix_token_basis",
            }
            for label, bkey in cond_bases.items():
                basis_d = bases_d[bkey]
                basis_r = bases_r[bkey]
                conditions = {
                    f"erase_frame_{label}": [(fp, z, basis_d, "subspace")],
                    f"erase_object_{label}": [(op, z, basis_d, "subspace")],
                    f"erase_both_{label}": [(op, z, basis_d, "subspace"), (fp, z, basis_d, "subspace")],
                }
                for cond, destroy_patches in conditions.items():
                    restore_patches = []
                    if "object" in cond or "both" in cond:
                        restore_patches.append((op, restore_obj, basis_r, "subspace"))
                    if "frame" in cond or "both" in cond:
                        restore_patches.append((fp, restore_frame, basis_r, "subspace"))
                    erase_scores = {
                        v: fullseq_logprob_rank_patch(model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length, args.module, destroy_layer, destroy_patches)
                        for v in candidates
                    }
                    restore_scores = {
                        v: fullseq_logprob_rank_patch(model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length, args.module, destroy_layer, destroy_patches, restore_layer, restore_patches)
                        for v in candidates
                    }
                    add_row(results, item, idx, destroy_layer, restore_layer, cond, base_stats, erase_scores, restore_scores, candidates)
            if (idx + 1) % args.progress_every == 0:
                log(f"pair={destroy_layer}->{restore_layer} item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s")
                partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                cleanup_cuda()
        partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    results["summary"] = summarize(results["rows"])
    final_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {final_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--layer-pairs", required=True)
    parser.add_argument("--module", default="resid_out")
    parser.add_argument("--relations", default="")
    parser.add_argument("--frames", default="")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--contrast-rank", type=int, default=64)
    parser.add_argument("--component-rank", type=int, default=24)
    parser.add_argument("--max-basis-items", type=int, default=224)
    parser.add_argument("--max-distractors", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=84)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    try:
        run_model(args)
    finally:
        release_loaded(None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    if args.hard_exit_after_model:
        log("Hard exit after model requested.")
        os._exit(0)


if __name__ == "__main__":
    main()
