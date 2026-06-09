from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import ctypes
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))

from hf_probe_env import get_layers, release_loaded  # noqa: E402
from phase68_object_attribute_natural_exchange import get_module, load_model, parse_csv  # noqa: E402
from phase70_object_relation_value_closure import parse_layer_pairs  # noqa: E402
from phase72_object_relation_value_fullseq_closure import capture_state, stats_from_scores  # noqa: E402
from phase75_relation_frame_token_intervention import get_frame_positions  # noqa: E402
from phase76_object_frame_joint_closure import uniq  # noqa: E402
from phase77_balanced_cross_relation_joint_closure import build_expanded_items, find_matched_source, find_mismatch_frame_source  # noqa: E402
from phase78_factor_subspace_audit import make_basis, summarize_rows  # noqa: E402
from phase79_rank_sweep_remainder_audit import fullseq_logprob_rank_patch  # noqa: E402


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


def orthonormalize(x: torch.Tensor, rank: int) -> torch.Tensor:
    if x.numel() == 0 or x.shape[1] == 0:
        return x[:, :0].contiguous()
    q, _r = torch.linalg.qr(x.float(), mode="reduced")
    return q[:, : min(rank, q.shape[1])].contiguous()


def safe_basis(diffs: list[torch.Tensor], rank: int, dim: int) -> torch.Tensor:
    if not diffs:
        return torch.zeros((dim, 0), dtype=torch.float32)
    return make_basis(diffs, rank)


def concat_bases(bases: list[torch.Tensor], rank: int) -> torch.Tensor:
    xs = [b.float() for b in bases if b.numel() and b.shape[1] > 0]
    if not xs:
        raise ValueError("empty nuisance bases")
    return orthonormalize(torch.cat(xs, dim=1), rank)


def remove_nuisance(base: torch.Tensor, nuisance: torch.Tensor, rank: int) -> torch.Tensor:
    b = base.float()
    n = nuisance.float()
    if n.numel() and n.shape[1] > 0:
        b = b - n @ (n.T @ b)
    return orthonormalize(b, rank)


def value_last_pos(tokenizer: Any, prompt: str, value: str) -> int | None:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    value_ids = tokenizer(" " + value, add_special_tokens=False)["input_ids"]
    if not value_ids:
        return None
    return len(prompt_ids) + len(value_ids) - 1


def find_same_item_other_frame(items: list[dict[str, Any]], idx: int) -> dict[str, Any] | None:
    item = items[idx]
    for j, other in enumerate(items):
        if j == idx:
            continue
        if (
            other["object"] == item["object"]
            and other["relation"] == item["relation"]
            and other["target"] == item["target"]
            and other["frame_key"] != item["frame_key"]
        ):
            return other
    return None


def build_factor_bases(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    items: list[dict[str, Any]],
    layer_idx: int,
    module: str,
    max_length: int,
    contrast_rank: int,
    nuisance_rank: int,
    max_basis_items: int,
) -> dict[str, torch.Tensor]:
    object_diffs: list[torch.Tensor] = []
    frame_diffs: list[torch.Tensor] = []
    value_diffs: list[torch.Tensor] = []
    template_diffs: list[torch.Tensor] = []
    position_diffs: list[torch.Tensor] = []
    limit = min(max_basis_items, len(items))
    for idx in range(limit):
        item = items[idx]
        matched = find_matched_source(items, idx)
        if matched is None:
            continue
        clean_pos = get_frame_positions(tokenizer, item["clean_prompt"], item["object"])
        matched_pos = get_frame_positions(tokenizer, matched["clean_prompt"], matched["object"])
        if any(x is None for x in (clean_pos.get("object_last"), clean_pos.get("frame_last"), matched_pos.get("object_last"), matched_pos.get("frame_last"))):
            continue
        h_clean = capture_state(model, tokenizer, device, layers[layer_idx], module, item["clean_prompt"], max_length)
        h_matched = capture_state(model, tokenizer, device, layers[layer_idx], module, matched["clean_prompt"], max_length)
        object_diffs.append(h_matched[int(matched_pos["object_last"])] - h_clean[int(clean_pos["object_last"])])
        frame_diffs.append(h_matched[int(matched_pos["frame_last"])] - h_clean[int(clean_pos["frame_last"])])

        clean_full = item["clean_prompt"] + " " + item["target"]
        matched_full = matched["clean_prompt"] + " " + matched["target"]
        cvp = value_last_pos(tokenizer, item["clean_prompt"], item["target"])
        mvp = value_last_pos(tokenizer, matched["clean_prompt"], matched["target"])
        if cvp is not None and mvp is not None:
            h_clean_val = capture_state(model, tokenizer, device, layers[layer_idx], module, clean_full, max_length)
            h_matched_val = capture_state(model, tokenizer, device, layers[layer_idx], module, matched_full, max_length)
            if cvp < h_clean_val.shape[0] and mvp < h_matched_val.shape[0]:
                value_diffs.append(h_matched_val[mvp] - h_clean_val[cvp])

        other_frame = find_same_item_other_frame(items, idx)
        if other_frame is not None:
            other_pos = get_frame_positions(tokenizer, other_frame["clean_prompt"], other_frame["object"])
            if other_pos.get("frame_last") is not None:
                h_other = capture_state(model, tokenizer, device, layers[layer_idx], module, other_frame["clean_prompt"], max_length)
                template_diffs.append(h_other[int(other_pos["frame_last"])] - h_clean[int(clean_pos["frame_last"])])

        short_prompt = f"{item['object']}."
        long_prompt = f"In this item, {item['object']}."
        short_pos = get_frame_positions(tokenizer, short_prompt, item["object"])
        long_pos = get_frame_positions(tokenizer, long_prompt, item["object"])
        if short_pos.get("object_last") is not None and long_pos.get("object_last") is not None:
            h_short = capture_state(model, tokenizer, device, layers[layer_idx], module, short_prompt, max_length)
            h_long = capture_state(model, tokenizer, device, layers[layer_idx], module, long_prompt, max_length)
            sp, lp = int(short_pos["object_last"]), int(long_pos["object_last"])
            if sp < h_short.shape[0] and lp < h_long.shape[0]:
                position_diffs.append(h_long[lp] - h_short[sp])

    if not object_diffs:
        raise ValueError("no object diffs for contrast basis")
    dim = int(object_diffs[0].numel())
    object_basis = safe_basis(object_diffs, contrast_rank, dim)
    frame_basis = safe_basis(frame_diffs, contrast_rank, dim)
    value_basis = safe_basis(value_diffs, nuisance_rank, dim)
    template_basis = safe_basis(template_diffs, nuisance_rank, dim)
    position_basis = safe_basis(position_diffs, nuisance_rank, dim)
    all_nuisance = concat_bases([value_basis, template_basis, position_basis], nuisance_rank * 3)
    return {
        "object_basis": object_basis,
        "frame_basis": frame_basis,
        "value_basis": value_basis,
        "template_basis": template_basis,
        "position_basis": position_basis,
        "all_nuisance_basis": all_nuisance,
        "object_orth_value": remove_nuisance(object_basis, value_basis, contrast_rank),
        "frame_orth_value": remove_nuisance(frame_basis, value_basis, contrast_rank),
        "object_orth_template": remove_nuisance(object_basis, template_basis, contrast_rank),
        "frame_orth_template": remove_nuisance(frame_basis, template_basis, contrast_rank),
        "object_orth_position": remove_nuisance(object_basis, position_basis, contrast_rank),
        "frame_orth_position": remove_nuisance(frame_basis, position_basis, contrast_rank),
        "object_orth_all": remove_nuisance(object_basis, all_nuisance, contrast_rank),
        "frame_orth_all": remove_nuisance(frame_basis, all_nuisance, contrast_rank),
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


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE80_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    layer_pairs = parse_layer_pairs(args.layer_pairs)
    items = build_expanded_items(args.max_items, parse_csv(args.relations), parse_csv(args.frames))
    log(f"Phase80 model={args.model} items={len(items)} layer_pairs={layer_pairs} contrast_rank={args.contrast_rank} nuisance_rank={args.nuisance_rank}")
    results: dict[str, Any] = {
        "phase": 80,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "orthogonal_factor_audit",
        "layer_pairs": layer_pairs,
        "module": args.module,
        "contrast_rank": args.contrast_rank,
        "nuisance_rank": args.nuisance_rank,
        "max_basis_items": args.max_basis_items,
        "relations": sorted({x["relation"] for x in items}),
        "num_items": len(items),
        "rows": [],
        "summary": {},
    }
    t0 = time.time()

    for destroy_layer, restore_layer in layer_pairs:
        log(f"building factor bases for L{destroy_layer} and L{restore_layer}")
        bases_d = build_factor_bases(model, tokenizer, device, layers, items, destroy_layer, args.module, args.max_length, args.contrast_rank, args.nuisance_rank, args.max_basis_items)
        bases_r = build_factor_bases(model, tokenizer, device, layers, items, restore_layer, args.module, args.max_length, args.contrast_rank, args.nuisance_rank, args.max_basis_items)
        log(f"bases ready for {destroy_layer}->{restore_layer}")

        for idx, item in enumerate(items):
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

            condition_bases = {
                "joint_raw": ("object_basis", "frame_basis", matched_frame_destroy),
                "joint_orth_value": ("object_orth_value", "frame_orth_value", matched_frame_destroy),
                "joint_orth_template": ("object_orth_template", "frame_orth_template", matched_frame_destroy),
                "joint_orth_position": ("object_orth_position", "frame_orth_position", matched_frame_destroy),
                "joint_orth_all": ("object_orth_all", "frame_orth_all", matched_frame_destroy),
                "joint_mismatched_frame_raw": ("object_basis", "frame_basis", mismatch_frame_destroy),
                "joint_value_basis_only": ("value_basis", "value_basis", matched_frame_destroy),
                "joint_template_basis_only": ("template_basis", "template_basis", matched_frame_destroy),
                "joint_position_basis_only": ("position_basis", "position_basis", matched_frame_destroy),
            }

            for cond, (ob_key, fb_key, frame_source) in condition_bases.items():
                destroy_patches = [
                    (op, matched_obj_destroy, bases_d[ob_key], "subspace"),
                    (fp, frame_source, bases_d[fb_key], "subspace"),
                ]
                restore_patches = []
                if cond == "joint_raw":
                    restore_cond = f"{cond}_restore_both"
                    restore_patches = [
                        (op, clean_obj_restore, bases_r[ob_key], "subspace"),
                        (fp, clean_frame_restore, bases_r[fb_key], "subspace"),
                    ]
                    patched_scores = {
                        v: fullseq_logprob_rank_patch(model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length, args.module, destroy_layer, destroy_patches, restore_layer, restore_patches)
                        for v in candidates
                    }
                    pcs = stats_from_scores(patched_scores, item["target"], [v for v in candidates if v != item["target"]])
                    pms = stats_from_scores(patched_scores, matched["target"], [v for v in candidates if v != matched["target"]])
                    results["rows"].append({
                        "item_idx": idx,
                        "destroy_layer": destroy_layer,
                        "restore_layer": restore_layer,
                        "condition": restore_cond,
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
                patched_scores = {
                    v: fullseq_logprob_rank_patch(model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length, args.module, destroy_layer, destroy_patches)
                    for v in candidates
                }
                pcs = stats_from_scores(patched_scores, item["target"], [v for v in candidates if v != item["target"]])
                pms = stats_from_scores(patched_scores, matched["target"], [v for v in candidates if v != matched["target"]])
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

            if (idx + 1) % args.progress_every == 0:
                log(f"pair={destroy_layer}->{restore_layer} item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s")
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{args.model}_phase80_orthogonal_factor_audit.partial.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                cleanup_cuda()

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{args.model}_phase80_orthogonal_factor_audit.partial.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    results["summary"] = summarize(results["rows"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_phase80_orthogonal_factor_audit.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
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
    parser.add_argument("--nuisance-rank", type=int, default=24)
    parser.add_argument("--max-basis-items", type=int, default=224)
    parser.add_argument("--max-distractors", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=84)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
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
