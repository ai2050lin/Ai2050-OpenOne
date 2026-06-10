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
from phase77_balanced_cross_relation_joint_closure import build_expanded_items, find_matched_source, find_mismatch_frame_source  # noqa: E402
from phase78_factor_subspace_audit import make_basis  # noqa: E402
from phase79_rank_sweep_remainder_audit import fullseq_logprob_rank_patch  # noqa: E402
from phase80_orthogonal_factor_audit import orthonormalize, remove_nuisance  # noqa: E402


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
        raise ValueError("empty component bases")
    return orthonormalize(torch.cat(xs, dim=1), rank)


def find_same_item_other_frame(items: list[dict[str, Any]], idx: int) -> dict[str, Any] | None:
    item = items[idx]
    pool = [
        x for x in items
        if x is not item
        and x["object"] == item["object"]
        and x["relation"] == item["relation"]
        and x["target"] == item["target"]
        and x["frame_key"] != item["frame_key"]
    ]
    if not pool:
        return None
    return pool[(idx * 11 + 5) % len(pool)]


def object_only_prompt(obj: str) -> str:
    return f"{obj}"


def boundary_prompt(prompt: str) -> str:
    return f"{prompt} Answer:"


def relation_label_prompt(obj: str, relation: str) -> str:
    return f"{obj} {relation.replace('_', ' ')}"


def build_component_bases(
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
    full_frame_diffs: list[torch.Tensor] = []
    pre_object_diffs: list[torch.Tensor] = []
    suffix_diffs: list[torch.Tensor] = []
    boundary_diffs: list[torch.Tensor] = []
    relation_label_diffs: list[torch.Tensor] = []
    limit = min(max_basis_items, len(items))
    for idx in range(limit):
        item = items[idx]
        matched = find_matched_source(items, idx)
        other_frame = find_same_item_other_frame(items, idx)
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

        if other_frame is not None:
            other_pos = get_frame_positions(tokenizer, other_frame["clean_prompt"], other_frame["object"])
            if other_pos.get("object_last") is not None and other_pos.get("frame_last") is not None:
                h_other = capture_state(model, tokenizer, device, layers[layer_idx], module, other_frame["clean_prompt"], max_length)
                pre_object_diffs.append(h_other[int(other_pos["object_last"])] - h_clean[int(clean_pos["object_last"])])
                full_frame_diffs.append(h_other[int(other_pos["frame_last"])] - h_clean[int(clean_pos["frame_last"])])

        obj_prompt = object_only_prompt(item["object"])
        obj_pos = get_frame_positions(tokenizer, obj_prompt, item["object"])
        if obj_pos.get("object_last") is not None:
            h_obj = capture_state(model, tokenizer, device, layers[layer_idx], module, obj_prompt, max_length)
            suffix_diffs.append(h_clean[int(clean_pos["frame_last"])] - h_obj[int(obj_pos["object_last"])])

        b_prompt = boundary_prompt(item["clean_prompt"])
        b_pos = get_frame_positions(tokenizer, b_prompt, item["object"])
        if b_pos.get("frame_last") is not None:
            h_b = capture_state(model, tokenizer, device, layers[layer_idx], module, b_prompt, max_length)
            boundary_diffs.append(h_b[int(b_pos["frame_last"])] - h_clean[int(clean_pos["frame_last"])])

        rel_prompt = relation_label_prompt(item["object"], item["relation"])
        rel_pos = get_frame_positions(tokenizer, rel_prompt, item["object"])
        if rel_pos.get("frame_last") is not None:
            h_rel = capture_state(model, tokenizer, device, layers[layer_idx], module, rel_prompt, max_length)
            relation_label_diffs.append(h_rel[int(rel_pos["frame_last"])] - h_clean[int(clean_pos["frame_last"])])

    if not object_diffs:
        raise ValueError("no object diffs for natural frame component basis")
    dim = int(object_diffs[0].numel())
    object_basis = safe_basis(object_diffs, contrast_rank, dim)
    frame_basis = safe_basis(frame_diffs, contrast_rank, dim)
    full_frame_basis = safe_basis(full_frame_diffs, component_rank, dim)
    pre_object_basis = safe_basis(pre_object_diffs, component_rank, dim)
    suffix_basis = safe_basis(suffix_diffs, component_rank, dim)
    boundary_basis = safe_basis(boundary_diffs, component_rank, dim)
    relation_label_basis = safe_basis(relation_label_diffs, component_rank, dim)
    all_components = concat_bases([full_frame_basis, pre_object_basis, suffix_basis, boundary_basis, relation_label_basis], component_rank * 5)
    return {
        "object_basis": object_basis,
        "frame_basis": frame_basis,
        "full_frame_basis": full_frame_basis,
        "pre_object_basis": pre_object_basis,
        "suffix_basis": suffix_basis,
        "boundary_basis": boundary_basis,
        "relation_label_basis": relation_label_basis,
        "all_component_basis": all_components,
        "object_orth_full_frame": remove_nuisance(object_basis, full_frame_basis, contrast_rank),
        "frame_orth_full_frame": remove_nuisance(frame_basis, full_frame_basis, contrast_rank),
        "object_orth_pre_object": remove_nuisance(object_basis, pre_object_basis, contrast_rank),
        "frame_orth_pre_object": remove_nuisance(frame_basis, pre_object_basis, contrast_rank),
        "object_orth_suffix": remove_nuisance(object_basis, suffix_basis, contrast_rank),
        "frame_orth_suffix": remove_nuisance(frame_basis, suffix_basis, contrast_rank),
        "object_orth_boundary": remove_nuisance(object_basis, boundary_basis, contrast_rank),
        "frame_orth_boundary": remove_nuisance(frame_basis, boundary_basis, contrast_rank),
        "object_orth_relation_label": remove_nuisance(object_basis, relation_label_basis, contrast_rank),
        "frame_orth_relation_label": remove_nuisance(frame_basis, relation_label_basis, contrast_rank),
        "object_orth_all_components": remove_nuisance(object_basis, all_components, contrast_rank),
        "frame_orth_all_components": remove_nuisance(frame_basis, all_components, contrast_rank),
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
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE82_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    layer_pairs = parse_layer_pairs(args.layer_pairs)
    items = build_expanded_items(args.max_items, parse_csv(args.relations), parse_csv(args.frames))
    log(f"Phase82 model={args.model} items={len(items)} layer_pairs={layer_pairs} contrast_rank={args.contrast_rank} component_rank={args.component_rank}")
    results: dict[str, Any] = {
        "phase": 82,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "natural_frame_component_ablation",
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
    t0 = time.time()
    for destroy_layer, restore_layer in layer_pairs:
        log(f"building natural component bases for L{destroy_layer} and L{restore_layer}")
        bases_d = build_component_bases(model, tokenizer, device, layers, items, destroy_layer, args.module, args.max_length, args.contrast_rank, args.component_rank, args.max_basis_items)
        bases_r = build_component_bases(model, tokenizer, device, layers, items, restore_layer, args.module, args.max_length, args.contrast_rank, args.component_rank, args.max_basis_items)
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

            conditions = {
                "joint_raw": ("object_basis", "frame_basis", matched_frame_destroy, matched["target"]),
                "joint_orth_full_frame": ("object_orth_full_frame", "frame_orth_full_frame", matched_frame_destroy, matched["target"]),
                "joint_orth_pre_object": ("object_orth_pre_object", "frame_orth_pre_object", matched_frame_destroy, matched["target"]),
                "joint_orth_suffix": ("object_orth_suffix", "frame_orth_suffix", matched_frame_destroy, matched["target"]),
                "joint_orth_boundary": ("object_orth_boundary", "frame_orth_boundary", matched_frame_destroy, matched["target"]),
                "joint_orth_relation_label": ("object_orth_relation_label", "frame_orth_relation_label", matched_frame_destroy, matched["target"]),
                "joint_orth_all_components": ("object_orth_all_components", "frame_orth_all_components", matched_frame_destroy, matched["target"]),
                "joint_full_frame_basis_only": ("full_frame_basis", "full_frame_basis", matched_frame_destroy, matched["target"]),
                "joint_pre_object_basis_only": ("pre_object_basis", "pre_object_basis", matched_frame_destroy, matched["target"]),
                "joint_suffix_basis_only": ("suffix_basis", "suffix_basis", matched_frame_destroy, matched["target"]),
                "joint_boundary_basis_only": ("boundary_basis", "boundary_basis", matched_frame_destroy, matched["target"]),
                "joint_relation_label_basis_only": ("relation_label_basis", "relation_label_basis", matched_frame_destroy, matched["target"]),
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
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{args.model}_phase82_natural_frame_component_ablation.partial.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                cleanup_cuda()

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{args.model}_phase82_natural_frame_component_ablation.partial.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    results["summary"] = summarize(results["rows"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_phase82_natural_frame_component_ablation.json"
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
    parser.add_argument("--component-rank", type=int, default=24)
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
