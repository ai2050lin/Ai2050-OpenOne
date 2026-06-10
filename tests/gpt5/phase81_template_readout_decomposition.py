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
from phase68_object_attribute_natural_exchange import get_module, load_model, parse_csv  # noqa: E402
from phase70_object_relation_value_closure import parse_layer_pairs  # noqa: E402
from phase72_object_relation_value_fullseq_closure import capture_state, stats_from_scores  # noqa: E402
from phase75_relation_frame_token_intervention import get_frame_positions  # noqa: E402
from phase76_object_frame_joint_closure import uniq  # noqa: E402
from phase77_balanced_cross_relation_joint_closure import build_expanded_items  # noqa: E402
from phase78_factor_subspace_audit import make_basis  # noqa: E402
from phase79_rank_sweep_remainder_audit import fullseq_logprob_rank_patch  # noqa: E402
from phase80_orthogonal_factor_audit import orthonormalize, remove_nuisance  # noqa: E402


RELATION_PHRASES: dict[str, list[str]] = {
    "is_a": ["category", "kind", "type", "class"],
    "used_for": ["use", "purpose", "function", "used for"],
    "can_do": ["ability", "can do", "typical action", "action"],
    "location": ["location", "place", "where found", "usual place"],
    "material": ["material", "made of", "substance", "composition"],
    "property": ["property", "trait", "quality", "feature"],
    "part_of": ["larger whole", "part of", "belongs to", "component relation"],
}

SLOT_STYLES: list[tuple[str, str]] = [
    ("answer", " Answer:"),
    ("value", " Value:"),
    ("arrow", " ->"),
    ("equals", " ="),
]


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


def build_relation_targets(relations: list[str]) -> list[dict[str, Any]]:
    rows = build_expanded_items(None, relations, [])
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (row["object"], row["relation"])
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "object": row["object"],
            "relation": row["relation"],
            "target": row["target"],
            "distractors": row["distractors"],
        })
    return out


def make_prompt(obj: str, phrase: str, slot_text: str) -> str:
    return f"Object: {obj}. Relation: {phrase}.{slot_text}"


def build_controlled_items(max_items: int | None, relations: list[str], phrase_ids: list[int], slot_ids: list[str]) -> list[dict[str, Any]]:
    targets = build_relation_targets(relations)
    wanted_phrase_ids = set(phrase_ids)
    wanted_slot_ids = set(slot_ids)
    rows: list[dict[str, Any]] = []
    for base in targets:
        rel = base["relation"]
        for phrase_idx, phrase in enumerate(RELATION_PHRASES[rel]):
            if wanted_phrase_ids and phrase_idx not in wanted_phrase_ids:
                continue
            for slot_id, slot_text in SLOT_STYLES:
                if wanted_slot_ids and slot_id not in wanted_slot_ids:
                    continue
                rows.append({
                    **base,
                    "phrase_id": phrase_idx,
                    "phrase": phrase,
                    "slot_id": slot_id,
                    "slot_text": slot_text,
                    "frame_key": f"p{phrase_idx}_{slot_id}",
                    "clean_prompt": make_prompt(base["object"], phrase, slot_text),
                })
    if not max_items or max_items >= len(rows):
        return rows
    idxs = sorted({round(i * (len(rows) - 1) / max(max_items - 1, 1)) for i in range(max_items)})
    return [rows[i] for i in idxs]


def select_from_pool(pool: list[dict[str, Any]], idx: int, salt: int) -> dict[str, Any] | None:
    if not pool:
        return None
    return pool[(idx * salt + 7) % len(pool)]


def find_matched_object_source(items: list[dict[str, Any]], idx: int) -> dict[str, Any] | None:
    item = items[idx]
    clean_values = {item["target"], *item["distractors"]}
    pool = [
        x for x in items
        if x is not item
        and x["object"] != item["object"]
        and x["relation"] != item["relation"]
        and x["phrase_id"] == item["phrase_id"]
        and x["slot_id"] == item["slot_id"]
        and x["target"] not in clean_values
    ]
    return select_from_pool(pool, idx, 19)


def find_other_phrase_source(items: list[dict[str, Any]], idx: int) -> dict[str, Any] | None:
    item = items[idx]
    pool = [
        x for x in items
        if x is not item
        and x["object"] == item["object"]
        and x["relation"] == item["relation"]
        and x["target"] == item["target"]
        and x["slot_id"] == item["slot_id"]
        and x["phrase_id"] != item["phrase_id"]
    ]
    return select_from_pool(pool, idx, 11)


def find_other_slot_source(items: list[dict[str, Any]], idx: int) -> dict[str, Any] | None:
    item = items[idx]
    pool = [
        x for x in items
        if x is not item
        and x["object"] == item["object"]
        and x["relation"] == item["relation"]
        and x["target"] == item["target"]
        and x["phrase_id"] == item["phrase_id"]
        and x["slot_id"] != item["slot_id"]
    ]
    return select_from_pool(pool, idx, 13)


def find_other_relation_source(items: list[dict[str, Any]], idx: int) -> dict[str, Any] | None:
    item = items[idx]
    pool = [
        x for x in items
        if x is not item
        and x["object"] == item["object"]
        and x["relation"] != item["relation"]
        and x["phrase_id"] == item["phrase_id"]
        and x["slot_id"] == item["slot_id"]
    ]
    return select_from_pool(pool, idx, 17)


def safe_basis(diffs: list[torch.Tensor], rank: int, dim: int) -> torch.Tensor:
    if not diffs:
        return torch.zeros((dim, 0), dtype=torch.float32)
    return make_basis(diffs, rank)


def concat_bases(bases: list[torch.Tensor], rank: int) -> torch.Tensor:
    xs = [b.float() for b in bases if b.numel() and b.shape[1] > 0]
    if not xs:
        raise ValueError("empty nuisance bases")
    return orthonormalize(torch.cat(xs, dim=1), rank)


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
    phrase_diffs: list[torch.Tensor] = []
    slot_diffs: list[torch.Tensor] = []
    relation_diffs: list[torch.Tensor] = []
    limit = min(max_basis_items, len(items))
    for idx in range(limit):
        item = items[idx]
        matched = find_matched_object_source(items, idx)
        other_phrase = find_other_phrase_source(items, idx)
        other_slot = find_other_slot_source(items, idx)
        other_relation = find_other_relation_source(items, idx)
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

        if other_phrase is not None:
            pos = get_frame_positions(tokenizer, other_phrase["clean_prompt"], other_phrase["object"])
            if pos.get("frame_last") is not None:
                h = capture_state(model, tokenizer, device, layers[layer_idx], module, other_phrase["clean_prompt"], max_length)
                phrase_diffs.append(h[int(pos["frame_last"])] - h_clean[int(clean_pos["frame_last"])])

        if other_slot is not None:
            pos = get_frame_positions(tokenizer, other_slot["clean_prompt"], other_slot["object"])
            if pos.get("frame_last") is not None:
                h = capture_state(model, tokenizer, device, layers[layer_idx], module, other_slot["clean_prompt"], max_length)
                slot_diffs.append(h[int(pos["frame_last"])] - h_clean[int(clean_pos["frame_last"])])

        if other_relation is not None:
            pos = get_frame_positions(tokenizer, other_relation["clean_prompt"], other_relation["object"])
            if pos.get("frame_last") is not None:
                h = capture_state(model, tokenizer, device, layers[layer_idx], module, other_relation["clean_prompt"], max_length)
                relation_diffs.append(h[int(pos["frame_last"])] - h_clean[int(clean_pos["frame_last"])])

    if not object_diffs:
        raise ValueError("no object diffs for controlled template/readout basis")
    dim = int(object_diffs[0].numel())
    object_basis = safe_basis(object_diffs, contrast_rank, dim)
    frame_basis = safe_basis(frame_diffs, contrast_rank, dim)
    phrase_basis = safe_basis(phrase_diffs, nuisance_rank, dim)
    slot_basis = safe_basis(slot_diffs, nuisance_rank, dim)
    relation_basis = safe_basis(relation_diffs, nuisance_rank, dim)
    all_nuisance = concat_bases([phrase_basis, slot_basis, relation_basis], nuisance_rank * 3)
    phrase_slot = concat_bases([phrase_basis, slot_basis], nuisance_rank * 2)
    return {
        "object_basis": object_basis,
        "frame_basis": frame_basis,
        "phrase_basis": phrase_basis,
        "slot_basis": slot_basis,
        "relation_basis": relation_basis,
        "phrase_slot_basis": phrase_slot,
        "all_nuisance_basis": all_nuisance,
        "object_orth_phrase": remove_nuisance(object_basis, phrase_basis, contrast_rank),
        "frame_orth_phrase": remove_nuisance(frame_basis, phrase_basis, contrast_rank),
        "object_orth_slot": remove_nuisance(object_basis, slot_basis, contrast_rank),
        "frame_orth_slot": remove_nuisance(frame_basis, slot_basis, contrast_rank),
        "object_orth_relation": remove_nuisance(object_basis, relation_basis, contrast_rank),
        "frame_orth_relation": remove_nuisance(frame_basis, relation_basis, contrast_rank),
        "object_orth_phrase_slot": remove_nuisance(object_basis, phrase_slot, contrast_rank),
        "frame_orth_phrase_slot": remove_nuisance(frame_basis, phrase_slot, contrast_rank),
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
    by_condition_slot: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        c = str(row["condition"])
        rel = str(row["relation"])
        slot = str(row["slot_id"])
        dl, rl = int(row["destroy_layer"]), int(row["restore_layer"])
        by_condition[c].append(row)
        by_condition_path[(c, dl, rl)].append(row)
        by_condition_relation[(c, rel)].append(row)
        by_condition_slot[(c, slot)].append(row)
    return {
        "by_condition": {k: group_summary(v) for k, v in by_condition.items()},
        "by_condition_path": {f"{c}:L{dl}->L{rl}": group_summary(v) for (c, dl, rl), v in by_condition_path.items()},
        "by_condition_relation": {f"{c}:{rel}": group_summary(v) for (c, rel), v in by_condition_relation.items()},
        "by_condition_slot": {f"{c}:{slot}": group_summary(v) for (c, slot), v in by_condition_slot.items()},
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
        "phrase_id": item["phrase_id"],
        "slot_id": item["slot_id"],
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
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE81_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    layer_pairs = parse_layer_pairs(args.layer_pairs)
    phrase_ids = [int(x) for x in parse_csv(args.phrase_ids)]
    slot_ids = parse_csv(args.slot_ids)
    items = build_controlled_items(args.max_items, parse_csv(args.relations), phrase_ids, slot_ids)
    log(f"Phase81 model={args.model} items={len(items)} layer_pairs={layer_pairs} contrast_rank={args.contrast_rank} nuisance_rank={args.nuisance_rank}")
    results: dict[str, Any] = {
        "phase": 81,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "template_readout_decomposition",
        "layer_pairs": layer_pairs,
        "module": args.module,
        "contrast_rank": args.contrast_rank,
        "nuisance_rank": args.nuisance_rank,
        "max_basis_items": args.max_basis_items,
        "relations": sorted({x["relation"] for x in items}),
        "phrase_ids": sorted({int(x["phrase_id"]) for x in items}),
        "slot_ids": sorted({str(x["slot_id"]) for x in items}),
        "num_items": len(items),
        "rows": [],
        "summary": {},
    }
    t0 = time.time()

    for destroy_layer, restore_layer in layer_pairs:
        log(f"building controlled factor bases for L{destroy_layer} and L{restore_layer}")
        bases_d = build_factor_bases(model, tokenizer, device, layers, items, destroy_layer, args.module, args.max_length, args.contrast_rank, args.nuisance_rank, args.max_basis_items)
        bases_r = build_factor_bases(model, tokenizer, device, layers, items, restore_layer, args.module, args.max_length, args.contrast_rank, args.nuisance_rank, args.max_basis_items)
        log(f"bases ready for {destroy_layer}->{restore_layer}")

        for idx, item in enumerate(items):
            matched = find_matched_object_source(items, idx)
            other_phrase = find_other_phrase_source(items, idx)
            other_slot = find_other_slot_source(items, idx)
            other_relation = find_other_relation_source(items, idx)
            if matched is None or other_phrase is None or other_slot is None or other_relation is None:
                continue
            clean_pos = get_frame_positions(tokenizer, item["clean_prompt"], item["object"])
            matched_pos = get_frame_positions(tokenizer, matched["clean_prompt"], matched["object"])
            phrase_pos = get_frame_positions(tokenizer, other_phrase["clean_prompt"], other_phrase["object"])
            slot_pos = get_frame_positions(tokenizer, other_slot["clean_prompt"], other_slot["object"])
            rel_pos = get_frame_positions(tokenizer, other_relation["clean_prompt"], other_relation["object"])
            need = (
                clean_pos.get("object_last"),
                clean_pos.get("frame_last"),
                matched_pos.get("object_last"),
                matched_pos.get("frame_last"),
                phrase_pos.get("frame_last"),
                slot_pos.get("frame_last"),
                rel_pos.get("frame_last"),
            )
            if any(x is None for x in need):
                continue

            clean_distractors = [x["target"] for x in items if x["target"] != item["target"] and x["relation"] == item["relation"]]
            candidates = uniq([item["target"], matched["target"], other_relation["target"]] + clean_distractors[: args.max_distractors])
            base_scores = {
                v: fullseq_logprob_rank_patch(model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length, args.module)
                for v in candidates
            }
            base_clean_stats = stats_from_scores(base_scores, item["target"], [v for v in candidates if v != item["target"]])
            base_matched_stats = stats_from_scores(base_scores, matched["target"], [v for v in candidates if v != matched["target"]])

            h_clean_r = capture_state(model, tokenizer, device, layers[restore_layer], args.module, item["clean_prompt"], args.max_length)
            h_matched_d = capture_state(model, tokenizer, device, layers[destroy_layer], args.module, matched["clean_prompt"], args.max_length)
            h_phrase_d = capture_state(model, tokenizer, device, layers[destroy_layer], args.module, other_phrase["clean_prompt"], args.max_length)
            h_slot_d = capture_state(model, tokenizer, device, layers[destroy_layer], args.module, other_slot["clean_prompt"], args.max_length)
            h_rel_d = capture_state(model, tokenizer, device, layers[destroy_layer], args.module, other_relation["clean_prompt"], args.max_length)

            op, fp = int(clean_pos["object_last"]), int(clean_pos["frame_last"])
            mop, mfp = int(matched_pos["object_last"]), int(matched_pos["frame_last"])
            pfp, sfp, rfp = int(phrase_pos["frame_last"]), int(slot_pos["frame_last"]), int(rel_pos["frame_last"])
            matched_obj_destroy = h_matched_d[mop]
            matched_frame_destroy = h_matched_d[mfp]
            phrase_frame_destroy = h_phrase_d[pfp]
            slot_frame_destroy = h_slot_d[sfp]
            relation_frame_destroy = h_rel_d[rfp]
            clean_obj_restore = h_clean_r[op]
            clean_frame_restore = h_clean_r[fp]

            conditions = {
                "joint_raw": ("object_basis", "frame_basis", matched_frame_destroy, matched["target"]),
                "joint_orth_phrase": ("object_orth_phrase", "frame_orth_phrase", matched_frame_destroy, matched["target"]),
                "joint_orth_slot": ("object_orth_slot", "frame_orth_slot", matched_frame_destroy, matched["target"]),
                "joint_orth_phrase_slot": ("object_orth_phrase_slot", "frame_orth_phrase_slot", matched_frame_destroy, matched["target"]),
                "joint_orth_relation": ("object_orth_relation", "frame_orth_relation", matched_frame_destroy, matched["target"]),
                "joint_orth_all": ("object_orth_all", "frame_orth_all", matched_frame_destroy, matched["target"]),
                "joint_phrase_basis_only": ("phrase_basis", "phrase_basis", matched_frame_destroy, matched["target"]),
                "joint_slot_basis_only": ("slot_basis", "slot_basis", matched_frame_destroy, matched["target"]),
                "joint_relation_basis_only": ("relation_basis", "relation_basis", matched_frame_destroy, matched["target"]),
                "joint_same_relation_other_phrase_frame": ("object_basis", "frame_basis", phrase_frame_destroy, item["target"]),
                "joint_same_relation_other_slot_frame": ("object_basis", "frame_basis", slot_frame_destroy, item["target"]),
                "joint_same_object_other_relation_frame": ("object_basis", "frame_basis", relation_frame_destroy, other_relation["target"]),
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
                (out_dir / f"{args.model}_phase81_template_readout_decomposition.partial.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                cleanup_cuda()

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{args.model}_phase81_template_readout_decomposition.partial.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    results["summary"] = summarize(results["rows"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_phase81_template_readout_decomposition.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--layer-pairs", required=True)
    parser.add_argument("--module", default="resid_out")
    parser.add_argument("--relations", default="")
    parser.add_argument("--phrase-ids", default="")
    parser.add_argument("--slot-ids", default="")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--contrast-rank", type=int, default=64)
    parser.add_argument("--nuisance-rank", type=int, default=24)
    parser.add_argument("--max-basis-items", type=int, default=448)
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
