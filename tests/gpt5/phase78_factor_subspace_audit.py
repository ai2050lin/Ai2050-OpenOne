from __future__ import annotations

import argparse
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
import torch.nn.functional as F


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))

from hf_probe_env import get_layers, release_loaded  # noqa: E402
from phase68_object_attribute_natural_exchange import encode, get_module, load_model, parse_csv  # noqa: E402
from phase70_object_relation_value_closure import parse_layer_pairs  # noqa: E402
from phase72_object_relation_value_fullseq_closure import capture_state, stats_from_scores  # noqa: E402
from phase75_relation_frame_token_intervention import get_frame_positions  # noqa: E402
from phase76_object_frame_joint_closure import fullseq_logprob_multi, uniq  # noqa: E402
from phase77_balanced_cross_relation_joint_closure import (  # noqa: E402
    build_expanded_items,
    find_matched_source,
    find_mismatch_frame_source,
)


def log(msg: str = "") -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def avg(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def make_basis(diffs: list[torch.Tensor], rank: int) -> torch.Tensor:
    if not diffs:
        raise ValueError("no diffs for basis")
    x = torch.stack([d.float() for d in diffs], dim=0)
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[0] < 2:
        q, _r = torch.linalg.qr(x.T, mode="reduced")
        return q[:, : min(rank, q.shape[1])].contiguous()
    _u, _s, vh = torch.linalg.svd(x, full_matrices=False)
    k = min(rank, vh.shape[0])
    return vh[:k].T.contiguous()


def project_replace_state(current: torch.Tensor, source: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    b = basis.to(device=current.device, dtype=current.dtype)
    delta = source.to(device=current.device, dtype=current.dtype) - current
    return current + b @ (b.T @ delta)


def fullseq_logprob_subspace(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    value: str,
    max_length: int,
    module_name: str,
    destroy_layer: int | None = None,
    destroy_patches: list[tuple[int, torch.Tensor, torch.Tensor]] | None = None,
    restore_layer: int | None = None,
    restore_patches: list[tuple[int, torch.Tensor, torch.Tensor]] | None = None,
) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    value_ids = tokenizer(" " + value, add_special_tokens=False)["input_ids"]
    if not value_ids:
        return float("-inf")
    full_ids = prompt_ids + value_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles = []

    def make_hook(patches: list[tuple[int, torch.Tensor, torch.Tensor]]):
        def hook_fn(_module: Any, _inputs: Any, output: Any):
            hs = output[0].clone() if isinstance(output, tuple) else output.clone()
            for pos_raw, source_cpu, basis_cpu in patches:
                pos = int(pos_raw) if pos_raw >= 0 else hs.shape[1] + int(pos_raw)
                if 0 <= pos < hs.shape[1]:
                    hs[0, pos, :] = project_replace_state(hs[0, pos, :], source_cpu, basis_cpu)
            return (hs,) + output[1:] if isinstance(output, tuple) else hs

        return hook_fn

    try:
        if destroy_layer is not None and destroy_patches:
            handles.append(get_module(layers[destroy_layer], module_name).register_forward_hook(make_hook(destroy_patches)))
        if restore_layer is not None and restore_patches:
            handles.append(get_module(layers[restore_layer], module_name).register_forward_hook(make_hook(restore_patches)))
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
            log_probs = F.log_softmax(logits.float(), dim=-1)
    finally:
        for h in handles:
            h.remove()

    start = len(prompt_ids)
    total = 0.0
    for i, tok in enumerate(value_ids):
        logit_pos = start + i - 1
        if logit_pos < 0 or logit_pos >= log_probs.shape[0]:
            return float("-inf")
        total += float(log_probs[logit_pos, tok].detach().cpu())
    return total


def build_bases(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    items: list[dict[str, Any]],
    layer_idx: int,
    module: str,
    max_length: int,
    rank: int,
    max_basis_items: int,
) -> dict[str, torch.Tensor]:
    object_diffs: list[torch.Tensor] = []
    frame_diffs: list[torch.Tensor] = []
    object_same_relation_diffs: list[torch.Tensor] = []
    frame_same_relation_diffs: list[torch.Tensor] = []
    limit = min(max_basis_items, len(items))
    for idx in range(limit):
        item = items[idx]
        matched = find_matched_source(items, idx)
        mismatch = find_mismatch_frame_source(items, idx, matched) if matched is not None else None
        if matched is None or mismatch is None:
            continue
        clean_pos = get_frame_positions(tokenizer, item["clean_prompt"], item["object"])
        matched_pos = get_frame_positions(tokenizer, matched["clean_prompt"], matched["object"])
        mismatch_pos = get_frame_positions(tokenizer, mismatch["clean_prompt"], mismatch["object"])
        if any(x is None for x in (clean_pos.get("object_last"), clean_pos.get("frame_last"), matched_pos.get("object_last"), matched_pos.get("frame_last"), mismatch_pos.get("frame_last"))):
            continue
        h_clean = capture_state(model, tokenizer, device, layers[layer_idx], module, item["clean_prompt"], max_length)
        h_matched = capture_state(model, tokenizer, device, layers[layer_idx], module, matched["clean_prompt"], max_length)
        h_mismatch = capture_state(model, tokenizer, device, layers[layer_idx], module, mismatch["clean_prompt"], max_length)
        object_diffs.append(h_matched[int(matched_pos["object_last"])] - h_clean[int(clean_pos["object_last"])])
        frame_diffs.append(h_matched[int(matched_pos["frame_last"])] - h_clean[int(clean_pos["frame_last"])])
        frame_same_relation_diffs.append(h_mismatch[int(mismatch_pos["frame_last"])] - h_clean[int(clean_pos["frame_last"])])
        object_same_relation_diffs.append(h_matched[int(matched_pos["object_last"])] - h_clean[int(clean_pos["object_last"])])
    return {
        "object_basis": make_basis(object_diffs, rank),
        "frame_basis": make_basis(frame_diffs, rank),
        "frame_mismatch_basis": make_basis(frame_same_relation_diffs, rank),
        "object_aux_basis": make_basis(object_same_relation_diffs, rank),
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


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_condition_path: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    by_condition_relation: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cond = str(row["condition"])
        rel = str(row["relation"])
        dl, rl = int(row["destroy_layer"]), int(row["restore_layer"])
        by_condition[cond].append(row)
        by_condition_path[(cond, dl, rl)].append(row)
        by_condition_relation[(cond, rel)].append(row)
    return {
        "by_condition": {k: group_summary(v) for k, v in by_condition.items()},
        "by_condition_path": {f"{c}:L{dl}->L{rl}": group_summary(v) for (c, dl, rl), v in by_condition_path.items()},
        "by_condition_relation": {f"{c}:{r}": group_summary(v) for (c, r), v in by_condition_relation.items()},
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE78_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    layer_pairs = parse_layer_pairs(args.layer_pairs)
    items = build_expanded_items(args.max_items, parse_csv(args.relations), parse_csv(args.frames))
    log(f"Phase78 model={args.model} items={len(items)} layer_pairs={layer_pairs} rank={args.basis_rank}")

    results: dict[str, Any] = {
        "phase": 78,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "factor_subspace_audit",
        "layer_pairs": layer_pairs,
        "module": args.module,
        "basis_rank": args.basis_rank,
        "max_basis_items": args.max_basis_items,
        "relations": sorted({x["relation"] for x in items}),
        "num_items": len(items),
        "rows": [],
        "summary": {},
    }
    t0 = time.time()

    for destroy_layer, restore_layer in layer_pairs:
        log(f"building bases for L{destroy_layer} and L{restore_layer}")
        bases_d = build_bases(model, tokenizer, device, layers, items, destroy_layer, args.module, args.max_length, args.basis_rank, args.max_basis_items)
        bases_r = build_bases(model, tokenizer, device, layers, items, restore_layer, args.module, args.max_length, args.basis_rank, args.max_basis_items)
        log(f"bases ready for {destroy_layer}->{restore_layer}")

        for idx, item in enumerate(items):
            matched = find_matched_source(items, idx)
            mismatch = find_mismatch_frame_source(items, idx, matched) if matched is not None else None
            if matched is None or mismatch is None:
                continue

            clean_pos = get_frame_positions(tokenizer, item["clean_prompt"], item["object"])
            matched_pos = get_frame_positions(tokenizer, matched["clean_prompt"], matched["object"])
            mismatch_pos = get_frame_positions(tokenizer, mismatch["clean_prompt"], mismatch["object"])
            needed = [
                clean_pos.get("object_last"), clean_pos.get("frame_last"),
                matched_pos.get("object_last"), matched_pos.get("frame_last"),
                mismatch_pos.get("frame_last"),
            ]
            if any(x is None for x in needed):
                continue

            h_clean_r = capture_state(model, tokenizer, device, layers[restore_layer], args.module, item["clean_prompt"], args.max_length)
            h_matched_d = capture_state(model, tokenizer, device, layers[destroy_layer], args.module, matched["clean_prompt"], args.max_length)
            h_mismatch_d = capture_state(model, tokenizer, device, layers[destroy_layer], args.module, mismatch["clean_prompt"], args.max_length)

            candidates = uniq([item["target"], *item["distractors"], matched["target"], mismatch["target"]])
            base_scores = {
                v: fullseq_logprob_multi(model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length, args.module)
                for v in candidates
            }
            base_clean_stats = stats_from_scores(base_scores, item["target"], [v for v in candidates if v != item["target"]])
            base_matched_stats = stats_from_scores(base_scores, matched["target"], [v for v in candidates if v != matched["target"]])

            op = int(clean_pos["object_last"])
            fp = int(clean_pos["frame_last"])
            mop = int(matched_pos["object_last"])
            mfp = int(matched_pos["frame_last"])
            xfp = int(mismatch_pos["frame_last"])

            clean_obj_restore = h_clean_r[op]
            clean_frame_restore = h_clean_r[fp]
            matched_obj_destroy = h_matched_d[mop]
            matched_frame_destroy = h_matched_d[mfp]
            mismatch_frame_destroy = h_mismatch_d[xfp]

            ob_d, fb_d = bases_d["object_basis"], bases_d["frame_basis"]
            ob_r, fb_r = bases_r["object_basis"], bases_r["frame_basis"]

            conditions: dict[str, tuple[list[tuple[int, torch.Tensor, torch.Tensor]], list[tuple[int, torch.Tensor, torch.Tensor]]]] = {
                "object_subspace_matched": ([(op, matched_obj_destroy, ob_d)], []),
                "frame_subspace_matched": ([(fp, matched_frame_destroy, fb_d)], []),
                "joint_subspace_matched": ([(op, matched_obj_destroy, ob_d), (fp, matched_frame_destroy, fb_d)], []),
                "joint_subspace_mismatched_frame": ([(op, matched_obj_destroy, ob_d), (fp, mismatch_frame_destroy, fb_d)], []),
                "joint_subspace_restore_object_only": ([(op, matched_obj_destroy, ob_d), (fp, matched_frame_destroy, fb_d)], [(op, clean_obj_restore, ob_r)]),
                "joint_subspace_restore_frame_only": ([(op, matched_obj_destroy, ob_d), (fp, matched_frame_destroy, fb_d)], [(fp, clean_frame_restore, fb_r)]),
                "joint_subspace_restore_both": ([(op, matched_obj_destroy, ob_d), (fp, matched_frame_destroy, fb_d)], [(op, clean_obj_restore, ob_r), (fp, clean_frame_restore, fb_r)]),
            }

            for cond, (destroy_patches, restore_patches) in conditions.items():
                patched_scores = {
                    v: fullseq_logprob_subspace(
                        model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length, args.module,
                        destroy_layer, destroy_patches, restore_layer if restore_patches else None, restore_patches
                    )
                    for v in candidates
                }
                patched_clean_stats = stats_from_scores(patched_scores, item["target"], [v for v in candidates if v != item["target"]])
                patched_matched_stats = stats_from_scores(patched_scores, matched["target"], [v for v in candidates if v != matched["target"]])
                results["rows"].append(
                    {
                        "destroy_layer": destroy_layer,
                        "restore_layer": restore_layer,
                        "module": args.module,
                        "condition": cond,
                        "relation": item["relation"],
                        "frame_key": item["frame_key"],
                        "object": item["object"],
                        "target": item["target"],
                        "matched_object": matched["object"],
                        "matched_relation": matched["relation"],
                        "matched_target": matched["target"],
                        "mismatch_relation": mismatch["relation"],
                        "mismatch_target": mismatch["target"],
                        "base_clean_margin": base_clean_stats["margin"],
                        "base_matched_margin": base_matched_stats["margin"],
                        "patched_clean_margin": patched_clean_stats["margin"],
                        "patched_matched_margin": patched_matched_stats["margin"],
                        "clean_drop": base_clean_stats["margin"] - patched_clean_stats["margin"],
                        "matched_gain": patched_matched_stats["margin"] - base_matched_stats["margin"],
                        "base_clean_rank": base_clean_stats["rank"],
                        "base_matched_rank": base_matched_stats["rank"],
                        "patched_clean_rank": patched_clean_stats["rank"],
                        "patched_matched_rank": patched_matched_stats["rank"],
                    }
                )

            if (idx + 1) % args.progress_every == 0:
                log(f"pair={destroy_layer}->{restore_layer} item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        partial = out_dir / f"{args.model}_phase78_factor_subspace_audit.partial.json"
        partial.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    results["summary"] = summarize_rows(results["rows"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_phase78_factor_subspace_audit.json"
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
    parser.add_argument("--max-length", type=int, default=112)
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--max-basis-items", type=int, default=168)
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
