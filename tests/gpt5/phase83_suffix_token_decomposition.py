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
from phase75_relation_frame_token_intervention import get_frame_positions, token_ids  # noqa: E402
from phase76_object_frame_joint_closure import uniq  # noqa: E402
from phase77_balanced_cross_relation_joint_closure import build_expanded_items, find_matched_source, find_mismatch_frame_source  # noqa: E402
from phase78_factor_subspace_audit import make_basis  # noqa: E402
from phase79_rank_sweep_remainder_audit import fullseq_logprob_rank_patch  # noqa: E402
from phase80_orthogonal_factor_audit import orthonormalize, remove_nuisance  # noqa: E402


FUNCTION_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "for", "to", "in", "on", "at", "from", "with",
    "can", "usually", "often", "able", "be", "used", "use", "people", "main", "common", "place",
}


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


def safe_basis(diffs: list[torch.Tensor], rank: int, dim: int) -> torch.Tensor:
    if not diffs:
        return torch.zeros((dim, 0), dtype=torch.float32)
    return make_basis(diffs, rank)


def concat_bases(bases: list[torch.Tensor], rank: int) -> torch.Tensor:
    xs = [b.float() for b in bases if b.numel() and b.shape[1] > 0]
    if not xs:
        raise ValueError("empty suffix-token bases")
    return orthonormalize(torch.cat(xs, dim=1), rank)


def object_only_prompt(obj: str) -> str:
    return f"{obj}"


def normalize_token_piece(text: str) -> str:
    return text.strip().lower().strip(".,:;!?\"'` ")


def suffix_positions(tokenizer: Any, prompt: str, obj: str) -> tuple[list[int], list[str]]:
    ids = token_ids(tokenizer, prompt)
    pos = get_frame_positions(tokenizer, prompt, obj)
    if pos.get("object_last") is None or pos.get("frame_last") is None:
        return [], []
    start = int(pos["object_last"]) + 1
    end = int(pos["frame_last"])
    if start > end:
        return [], []
    positions = list(range(start, end + 1))
    pieces = [normalize_token_piece(tokenizer.decode([ids[p]], clean_up_tokenization_spaces=False)) for p in positions]
    return positions, pieces


def build_suffix_token_bases(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    items: list[dict[str, Any]],
    layer_idx: int,
    module: str,
    max_length: int,
    contrast_rank: int,
    component_rank: int,
    max_basis_items: int,
) -> dict[str, torch.Tensor]:
    object_diffs: list[torch.Tensor] = []
    frame_diffs: list[torch.Tensor] = []
    suffix_all_diffs: list[torch.Tensor] = []
    suffix_nonfinal_diffs: list[torch.Tensor] = []
    suffix_final_diffs: list[torch.Tensor] = []
    suffix_penultimate_diffs: list[torch.Tensor] = []
    suffix_first_diffs: list[torch.Tensor] = []
    suffix_second_diffs: list[torch.Tensor] = []
    suffix_function_diffs: list[torch.Tensor] = []
    suffix_lexical_diffs: list[torch.Tensor] = []

    limit = min(max_basis_items, len(items))
    for idx in range(limit):
        item = items[idx]
        matched = find_matched_source(items, idx)
        clean_pos = get_frame_positions(tokenizer, item["clean_prompt"], item["object"])
        if clean_pos.get("object_last") is None or clean_pos.get("frame_last") is None:
            continue
        h_clean = capture_state(model, tokenizer, device, layers[layer_idx], module, item["clean_prompt"], max_length)
        if matched is not None:
            matched_pos = get_frame_positions(tokenizer, matched["clean_prompt"], matched["object"])
            if matched_pos.get("object_last") is not None and matched_pos.get("frame_last") is not None:
                h_matched = capture_state(model, tokenizer, device, layers[layer_idx], module, matched["clean_prompt"], max_length)
                object_diffs.append(h_matched[int(matched_pos["object_last"])] - h_clean[int(clean_pos["object_last"])])
                frame_diffs.append(h_matched[int(matched_pos["frame_last"])] - h_clean[int(clean_pos["frame_last"])])

        obj_prompt = object_only_prompt(item["object"])
        obj_pos = get_frame_positions(tokenizer, obj_prompt, item["object"])
        if obj_pos.get("object_last") is None:
            continue
        h_obj = capture_state(model, tokenizer, device, layers[layer_idx], module, obj_prompt, max_length)
        obj_state = h_obj[int(obj_pos["object_last"])]
        positions, pieces = suffix_positions(tokenizer, item["clean_prompt"], item["object"])
        if not positions:
            continue
        for j, p in enumerate(positions):
            diff = h_clean[p] - obj_state
            suffix_all_diffs.append(diff)
            if j < len(positions) - 1:
                suffix_nonfinal_diffs.append(diff)
            if pieces[j] in FUNCTION_WORDS:
                suffix_function_diffs.append(diff)
            else:
                suffix_lexical_diffs.append(diff)
        suffix_first_diffs.append(h_clean[positions[0]] - obj_state)
        if len(positions) > 1:
            suffix_second_diffs.append(h_clean[positions[1]] - obj_state)
            suffix_penultimate_diffs.append(h_clean[positions[-2]] - obj_state)
        suffix_final_diffs.append(h_clean[positions[-1]] - obj_state)

    if not object_diffs:
        raise ValueError("no object diffs for suffix token basis")
    dim = int(object_diffs[0].numel())
    object_basis = safe_basis(object_diffs, contrast_rank, dim)
    frame_basis = safe_basis(frame_diffs, contrast_rank, dim)
    suffix_all_basis = safe_basis(suffix_all_diffs, component_rank, dim)
    suffix_nonfinal_basis = safe_basis(suffix_nonfinal_diffs, component_rank, dim)
    suffix_final_basis = safe_basis(suffix_final_diffs, component_rank, dim)
    suffix_penultimate_basis = safe_basis(suffix_penultimate_diffs, component_rank, dim)
    suffix_first_basis = safe_basis(suffix_first_diffs, component_rank, dim)
    suffix_second_basis = safe_basis(suffix_second_diffs, component_rank, dim)
    suffix_function_basis = safe_basis(suffix_function_diffs, component_rank, dim)
    suffix_lexical_basis = safe_basis(suffix_lexical_diffs, component_rank, dim)
    all_components = concat_bases(
        [suffix_first_basis, suffix_second_basis, suffix_nonfinal_basis, suffix_final_basis, suffix_function_basis, suffix_lexical_basis],
        component_rank * 6,
    )
    return {
        "object_basis": object_basis,
        "frame_basis": frame_basis,
        "suffix_all_basis": suffix_all_basis,
        "suffix_nonfinal_basis": suffix_nonfinal_basis,
        "suffix_final_basis": suffix_final_basis,
        "suffix_penultimate_basis": suffix_penultimate_basis,
        "suffix_first_basis": suffix_first_basis,
        "suffix_second_basis": suffix_second_basis,
        "suffix_function_basis": suffix_function_basis,
        "suffix_lexical_basis": suffix_lexical_basis,
        "all_suffix_token_basis": all_components,
        "object_orth_suffix_all": remove_nuisance(object_basis, suffix_all_basis, contrast_rank),
        "frame_orth_suffix_all": remove_nuisance(frame_basis, suffix_all_basis, contrast_rank),
        "object_orth_suffix_nonfinal": remove_nuisance(object_basis, suffix_nonfinal_basis, contrast_rank),
        "frame_orth_suffix_nonfinal": remove_nuisance(frame_basis, suffix_nonfinal_basis, contrast_rank),
        "object_orth_suffix_final": remove_nuisance(object_basis, suffix_final_basis, contrast_rank),
        "frame_orth_suffix_final": remove_nuisance(frame_basis, suffix_final_basis, contrast_rank),
        "object_orth_suffix_penultimate": remove_nuisance(object_basis, suffix_penultimate_basis, contrast_rank),
        "frame_orth_suffix_penultimate": remove_nuisance(frame_basis, suffix_penultimate_basis, contrast_rank),
        "object_orth_suffix_first": remove_nuisance(object_basis, suffix_first_basis, contrast_rank),
        "frame_orth_suffix_first": remove_nuisance(frame_basis, suffix_first_basis, contrast_rank),
        "object_orth_suffix_second": remove_nuisance(object_basis, suffix_second_basis, contrast_rank),
        "frame_orth_suffix_second": remove_nuisance(frame_basis, suffix_second_basis, contrast_rank),
        "object_orth_suffix_function": remove_nuisance(object_basis, suffix_function_basis, contrast_rank),
        "frame_orth_suffix_function": remove_nuisance(frame_basis, suffix_function_basis, contrast_rank),
        "object_orth_suffix_lexical": remove_nuisance(object_basis, suffix_lexical_basis, contrast_rank),
        "frame_orth_suffix_lexical": remove_nuisance(frame_basis, suffix_lexical_basis, contrast_rank),
        "object_orth_all_suffix_tokens": remove_nuisance(object_basis, all_components, contrast_rank),
        "frame_orth_all_suffix_tokens": remove_nuisance(frame_basis, all_components, contrast_rank),
    }


def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [v for v in vals if v["base_clean_rank"] == 1]
    return {
        "n": len(vals),
        "eligible_n": len(eligible),
        "clean_drop": avg([float(v["clean_drop"]) for v in vals]),
        "matched_gain": avg([float(v["matched_gain"]) for v in vals]),
        "patched_clean_top1": avg([1.0 if v["patched_clean_rank"] == 1 else 0.0 for v in vals]),
        "patched_matched_top1": avg([1.0 if v["patched_matched_rank"] == 1 else 0.0 for v in vals]),
        "eligible_clean_drop": avg([float(v["clean_drop"]) for v in eligible]),
        "eligible_matched_gain": avg([float(v["matched_gain"]) for v in eligible]),
        "eligible_patched_clean_top1": avg([1.0 if v["patched_clean_rank"] == 1 else 0.0 for v in eligible]),
        "eligible_patched_matched_top1": avg([1.0 if v["patched_matched_rank"] == 1 else 0.0 for v in eligible]),
        "eligible_clean_margin_after": avg([float(v["patched_clean_margin"]) for v in eligible]),
        "eligible_matched_margin_after": avg([float(v["patched_matched_margin"]) for v in eligible]),
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
    base_clean_stats: dict[str, Any],
    base_matched_stats: dict[str, Any],
    patched_scores: dict[str, float],
    candidates: list[str],
    matched_target: str,
) -> None:
    pcs = stats_from_scores(patched_scores, item["target"], [v for v in candidates if v != item["target"]])
    pms = stats_from_scores(patched_scores, matched_target, [v for v in candidates if v != matched_target])
    results["rows"].append({
        "item_idx": idx,
        "destroy_layer": destroy_layer,
        "restore_layer": restore_layer,
        "condition": cond,
        "relation": item["relation"],
        "frame_key": item["frame_key"],
        "base_clean_margin": base_clean_stats["margin"],
        "base_matched_margin": base_matched_stats["margin"],
        "patched_clean_margin": pcs["margin"],
        "patched_matched_margin": pms["margin"],
        "clean_drop": base_clean_stats["margin"] - pcs["margin"],
        "matched_gain": pms["margin"] - base_matched_stats["margin"],
        "base_clean_rank": base_clean_stats["rank"],
        "base_matched_rank": base_matched_stats["rank"],
        "patched_clean_rank": pcs["rank"],
        "patched_matched_rank": pms["rank"],
    })


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE83_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    layer_pairs = parse_layer_pairs(args.layer_pairs)
    items = build_expanded_items(args.max_items, parse_csv(args.relations), parse_csv(args.frames))
    log(f"Phase83 model={args.model} items={len(items)} layer_pairs={layer_pairs} contrast_rank={args.contrast_rank} component_rank={args.component_rank}")
    results: dict[str, Any] = {
        "phase": 83,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "suffix_token_decomposition",
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
    final_path = out_dir / f"{args.model}_phase83_suffix_token_decomposition.json"
    partial_path = out_dir / f"{args.model}_phase83_suffix_token_decomposition.partial.json"
    if args.resume:
        resume_path = final_path if final_path.exists() else partial_path
        if resume_path.exists():
            loaded = json.loads(resume_path.read_text(encoding="utf-8"))
            if loaded.get("model") == args.model and loaded.get("phase") == 83:
                results = loaded
                results.setdefault("rows", [])
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")
    completed_item_keys: set[tuple[int, int, int]] = set()
    by_item_count: dict[tuple[int, int, int], int] = defaultdict(int)
    for row in results["rows"]:
        by_item_count[(int(row["destroy_layer"]), int(row["restore_layer"]), int(row["item_idx"]))] += 1
    for key, count in by_item_count.items():
        if count >= 17:
            completed_item_keys.add(key)
    t0 = time.time()
    for destroy_layer, restore_layer in layer_pairs:
        log(f"building suffix token bases for L{destroy_layer} and L{restore_layer}")
        bases_d = build_suffix_token_bases(model, tokenizer, device, layers, items, destroy_layer, args.module, args.max_length, args.contrast_rank, args.component_rank, args.max_basis_items)
        bases_r = build_suffix_token_bases(model, tokenizer, device, layers, items, restore_layer, args.module, args.max_length, args.contrast_rank, args.component_rank, args.max_basis_items)
        log(f"bases ready for {destroy_layer}->{restore_layer}")
        for idx, item in enumerate(items):
            if (destroy_layer, restore_layer, idx) in completed_item_keys:
                continue
            matched = find_matched_source(items, idx)
            mismatch = find_mismatch_frame_source(items, idx, matched) if matched is not None else None
            if matched is None or mismatch is None:
                continue
            clean_pos = get_frame_positions(tokenizer, item["clean_prompt"], item["object"])
            matched_pos = get_frame_positions(tokenizer, matched["clean_prompt"], matched["object"])
            mismatch_pos = get_frame_positions(tokenizer, mismatch["clean_prompt"], mismatch["object"])
            if any(x is None for x in (clean_pos.get("object_last"), clean_pos.get("frame_last"), matched_pos.get("object_last"), matched_pos.get("frame_last"), mismatch_pos.get("frame_last"))):
                continue
            clean_distractors = [x["target"] for x in items if x["target"] != item["target"] and x["relation"] == item["relation"]]
            candidates = uniq([item["target"], matched["target"], mismatch["target"]] + clean_distractors[: args.max_distractors])
            base_scores = {
                v: fullseq_logprob_rank_patch(model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length, args.module)
                for v in candidates
            }
            base_clean_stats = stats_from_scores(base_scores, item["target"], [v for v in candidates if v != item["target"]])
            base_matched_stats = stats_from_scores(base_scores, matched["target"], [v for v in candidates if v != matched["target"]])
            h_clean_r = capture_state(model, tokenizer, device, layers[restore_layer], args.module, item["clean_prompt"], args.max_length)
            h_matched_d = capture_state(model, tokenizer, device, layers[destroy_layer], args.module, matched["clean_prompt"], args.max_length)
            h_mismatch_d = capture_state(model, tokenizer, device, layers[destroy_layer], args.module, mismatch["clean_prompt"], args.max_length)
            op, fp = int(clean_pos["object_last"]), int(clean_pos["frame_last"])
            mop, mfp = int(matched_pos["object_last"]), int(matched_pos["frame_last"])
            xfp = int(mismatch_pos["frame_last"])
            matched_obj_destroy = h_matched_d[mop]
            matched_frame_destroy = h_matched_d[mfp]
            mismatch_frame_destroy = h_mismatch_d[xfp]
            clean_obj_restore = h_clean_r[op]
            clean_frame_restore = h_clean_r[fp]
            conditions = {
                "joint_raw": ("object_basis", "frame_basis", matched_frame_destroy, matched["target"]),
                "joint_orth_suffix_all": ("object_orth_suffix_all", "frame_orth_suffix_all", matched_frame_destroy, matched["target"]),
                "joint_orth_suffix_nonfinal": ("object_orth_suffix_nonfinal", "frame_orth_suffix_nonfinal", matched_frame_destroy, matched["target"]),
                "joint_orth_suffix_final": ("object_orth_suffix_final", "frame_orth_suffix_final", matched_frame_destroy, matched["target"]),
                "joint_orth_suffix_penultimate": ("object_orth_suffix_penultimate", "frame_orth_suffix_penultimate", matched_frame_destroy, matched["target"]),
                "joint_orth_suffix_first": ("object_orth_suffix_first", "frame_orth_suffix_first", matched_frame_destroy, matched["target"]),
                "joint_orth_suffix_second": ("object_orth_suffix_second", "frame_orth_suffix_second", matched_frame_destroy, matched["target"]),
                "joint_orth_suffix_function": ("object_orth_suffix_function", "frame_orth_suffix_function", matched_frame_destroy, matched["target"]),
                "joint_orth_suffix_lexical": ("object_orth_suffix_lexical", "frame_orth_suffix_lexical", matched_frame_destroy, matched["target"]),
                "joint_orth_all_suffix_tokens": ("object_orth_all_suffix_tokens", "frame_orth_all_suffix_tokens", matched_frame_destroy, matched["target"]),
                "joint_suffix_all_basis_only": ("suffix_all_basis", "suffix_all_basis", matched_frame_destroy, matched["target"]),
                "joint_suffix_nonfinal_basis_only": ("suffix_nonfinal_basis", "suffix_nonfinal_basis", matched_frame_destroy, matched["target"]),
                "joint_suffix_final_basis_only": ("suffix_final_basis", "suffix_final_basis", matched_frame_destroy, matched["target"]),
                "joint_suffix_function_basis_only": ("suffix_function_basis", "suffix_function_basis", matched_frame_destroy, matched["target"]),
                "joint_suffix_lexical_basis_only": ("suffix_lexical_basis", "suffix_lexical_basis", matched_frame_destroy, matched["target"]),
                "joint_mismatched_frame_raw": ("object_basis", "frame_basis", mismatch_frame_destroy, mismatch["target"]),
            }
            for cond, (ob_key, fb_key, frame_source, matched_target) in conditions.items():
                destroy_patches = [
                    (op, matched_obj_destroy, bases_d[ob_key], "subspace"),
                    (fp, frame_source, bases_d[fb_key], "subspace"),
                ]
                if cond == "joint_raw":
                    restore_patches = [
                        (op, clean_obj_restore, bases_r[ob_key], "subspace"),
                        (fp, clean_frame_restore, bases_r[fb_key], "subspace"),
                    ]
                    patched_scores = {
                        v: fullseq_logprob_rank_patch(model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length, args.module, destroy_layer, destroy_patches, restore_layer, restore_patches)
                        for v in candidates
                    }
                    add_row(results, item, idx, destroy_layer, restore_layer, "joint_raw_restore_both", base_clean_stats, base_matched_stats, patched_scores, candidates, matched["target"])
                patched_scores = {
                    v: fullseq_logprob_rank_patch(model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length, args.module, destroy_layer, destroy_patches)
                    for v in candidates
                }
                add_row(results, item, idx, destroy_layer, restore_layer, cond, base_clean_stats, base_matched_stats, patched_scores, candidates, matched_target)
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
