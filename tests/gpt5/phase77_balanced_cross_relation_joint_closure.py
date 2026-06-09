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


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))

from hf_probe_env import get_layers, release_loaded  # noqa: E402
from phase68_object_attribute_natural_exchange import load_model, parse_csv  # noqa: E402
from phase70_object_relation_value_closure import parse_layer_pairs  # noqa: E402
from phase72_object_relation_value_fullseq_closure import capture_state, stats_from_scores  # noqa: E402
from phase75_relation_frame_token_intervention import get_frame_positions  # noqa: E402
from phase76_object_frame_joint_closure import fullseq_logprob_multi, uniq  # noqa: E402


def log(msg: str = "") -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


RELATION_FRAMES: dict[str, list[tuple[str, str]]] = {
    "is_a": [
        ("kind", "A {obj} is a kind of"),
        ("simple", "A {obj} is a"),
        ("usually", "The {obj} is usually a"),
        ("category", "The category of a {obj} is"),
    ],
    "used_for": [
        ("used_for", "A {obj} is used for"),
        ("people_use", "People use a {obj} for"),
        ("helps_with", "A {obj} helps with"),
        ("main_use", "The main use of a {obj} is"),
    ],
    "can_do": [
        ("can", "A {obj} can"),
        ("usually_can", "A {obj} usually can"),
        ("often_can", "The {obj} often can"),
        ("able_to", "A {obj} is able to"),
    ],
    "location": [
        ("found_in", "A {obj} is usually found in"),
        ("often_in", "A {obj} is often in"),
        ("belongs_in", "The {obj} belongs in"),
        ("common_place", "A common place for a {obj} is"),
    ],
    "material": [
        ("made_of", "A {obj} is usually made of"),
        ("the_made_of", "The {obj} is made of"),
        ("built_from", "A {obj} can be built from"),
        ("material_is", "The material of a {obj} is"),
    ],
    "property": [
        ("looks", "A {obj} usually looks"),
        ("feels", "A {obj} usually feels"),
        ("is_often", "A {obj} is often"),
        ("has_quality", "A common quality of a {obj} is"),
    ],
    "part_of": [
        ("part", "A {obj} is part of"),
        ("belongs_to", "A {obj} belongs to"),
        ("found_on", "A {obj} can be found on"),
        ("component_of", "A {obj} is a component of"),
    ],
}


RELATION_DISTRACTOR_POOLS: dict[str, list[str]] = {
    "is_a": [
        "metal tool", "small container", "sweet fruit", "small animal", "water animal", "electronic device",
        "printed object", "green plant", "home animal", "road vehicle", "flowering plant", "writing tool",
        "household item", "medical worker", "wooden furniture", "glass object", "cloth garment", "round coin",
    ],
    "used_for": [
        "cutting food", "drinking water", "eating food", "making nests", "cooking food", "making calls",
        "reading stories", "giving shade", "guarding homes", "driving roads", "giving flowers", "hitting nails",
        "writing notes", "sleeping comfortably", "taking pictures", "cleaning floors", "checking patients",
        "storing money", "covering windows", "sitting down", "wearing clothes", "letting light", "lighting rooms",
    ],
    "can_do": [
        "cut food", "hold water", "grow seeds", "fly fast", "swim well", "send messages", "hold words",
        "grow leaves", "bark loudly", "carry people", "bloom brightly", "break rocks", "write notes",
        "support weight", "reflect light", "cover skin", "heal people", "roll around", "open locks",
        "clean floors", "shine brightly",
    ],
    "location": [
        "home kitchen", "grocery store", "blue sky", "fresh water", "coat pocket", "school building",
        "deep forest", "family home", "city road", "garden bed", "tool box", "office desk", "bedroom room",
        "doctor office", "dining room", "window frame", "clothes closet", "wallet pocket", "garage shelf",
        "library room",
    ],
    "material": [
        "shiny metal", "clear glass", "soft flesh", "soft feathers", "white paper", "hard wood",
        "soft fur", "black rubber", "soft petals", "yellow gold", "soft cloth", "red brick",
        "hard plastic", "smooth leather", "fresh water", "green leaf", "rough stone", "bright ceramic",
    ],
    "property": [
        "sharp edge", "round shape", "sweet taste", "light body", "wet skin", "bright screen",
        "many pages", "rough bark", "loud bark", "loud engine", "red flower", "hard head",
        "soft cloth", "deep blue", "pure white", "heavy weight", "clear surface", "warm light",
        "smooth surface", "clean smell",
    ],
    "part_of": [
        "a kitchen set", "a dining set", "a fruit tree", "a bird flock", "a river habitat", "a phone system",
        "a book shelf", "a forest ecosystem", "a family group", "a transport system", "a flower garden",
        "a tool kit", "a writing kit", "a bedroom set", "a camera system", "a cleaning kit",
        "a hospital team", "a money purse", "a window assembly", "a furniture set", "a clothing set",
    ],
}


EXPANDED_OBJECTS: list[dict[str, str]] = [
    {"object": "knife", "is_a": "metal tool", "used_for": "cutting food", "can_do": "cut food", "location": "home kitchen", "material": "shiny metal", "property": "sharp edge", "part_of": "a kitchen set"},
    {"object": "cup", "is_a": "small container", "used_for": "drinking water", "can_do": "hold water", "location": "home kitchen", "material": "clear glass", "property": "round shape", "part_of": "a dining set"},
    {"object": "apple", "is_a": "sweet fruit", "used_for": "eating food", "can_do": "grow seeds", "location": "grocery store", "material": "soft flesh", "property": "sweet taste", "part_of": "a fruit tree"},
    {"object": "bird", "is_a": "small animal", "used_for": "making nests", "can_do": "fly fast", "location": "blue sky", "material": "soft feathers", "property": "light body", "part_of": "a bird flock"},
    {"object": "fish", "is_a": "water animal", "used_for": "eating food", "can_do": "swim well", "location": "fresh water", "material": "soft flesh", "property": "wet skin", "part_of": "a river habitat"},
    {"object": "phone", "is_a": "electronic device", "used_for": "making calls", "can_do": "send messages", "location": "coat pocket", "material": "shiny metal", "property": "bright screen", "part_of": "a phone system"},
    {"object": "book", "is_a": "printed object", "used_for": "reading stories", "can_do": "hold words", "location": "school building", "material": "white paper", "property": "many pages", "part_of": "a book shelf"},
    {"object": "tree", "is_a": "green plant", "used_for": "giving shade", "can_do": "grow leaves", "location": "deep forest", "material": "hard wood", "property": "rough bark", "part_of": "a forest ecosystem"},
    {"object": "dog", "is_a": "home animal", "used_for": "guarding homes", "can_do": "bark loudly", "location": "family home", "material": "soft fur", "property": "loud bark", "part_of": "a family group"},
    {"object": "car", "is_a": "road vehicle", "used_for": "driving roads", "can_do": "carry people", "location": "city road", "material": "shiny metal", "property": "loud engine", "part_of": "a transport system"},
    {"object": "rose", "is_a": "flowering plant", "used_for": "giving flowers", "can_do": "bloom brightly", "location": "garden bed", "material": "soft petals", "property": "red flower", "part_of": "a flower garden"},
    {"object": "hammer", "is_a": "metal tool", "used_for": "hitting nails", "can_do": "break rocks", "location": "tool box", "material": "shiny metal", "property": "hard head", "part_of": "a tool kit"},
    {"object": "pen", "is_a": "writing tool", "used_for": "writing notes", "can_do": "write notes", "location": "office desk", "material": "hard plastic", "property": "smooth surface", "part_of": "a writing kit"},
    {"object": "bed", "is_a": "household item", "used_for": "sleeping comfortably", "can_do": "support weight", "location": "bedroom room", "material": "soft cloth", "property": "soft cloth", "part_of": "a bedroom set"},
    {"object": "camera", "is_a": "electronic device", "used_for": "taking pictures", "can_do": "reflect light", "location": "coat pocket", "material": "hard plastic", "property": "clear surface", "part_of": "a camera system"},
    {"object": "broom", "is_a": "household item", "used_for": "cleaning floors", "can_do": "clean floors", "location": "home kitchen", "material": "hard wood", "property": "clean smell", "part_of": "a cleaning kit"},
    {"object": "doctor", "is_a": "medical worker", "used_for": "checking patients", "can_do": "heal people", "location": "doctor office", "material": "soft flesh", "property": "warm light", "part_of": "a hospital team"},
    {"object": "coin", "is_a": "round coin", "used_for": "storing money", "can_do": "roll around", "location": "wallet pocket", "material": "shiny metal", "property": "round shape", "part_of": "a money purse"},
    {"object": "curtain", "is_a": "cloth garment", "used_for": "covering windows", "can_do": "cover skin", "location": "window frame", "material": "soft cloth", "property": "deep blue", "part_of": "a window assembly"},
    {"object": "chair", "is_a": "wooden furniture", "used_for": "sitting down", "can_do": "support weight", "location": "dining room", "material": "hard wood", "property": "heavy weight", "part_of": "a furniture set"},
    {"object": "spoon", "is_a": "metal tool", "used_for": "eating food", "can_do": "hold water", "location": "home kitchen", "material": "shiny metal", "property": "smooth surface", "part_of": "a kitchen set"},
    {"object": "shirt", "is_a": "cloth garment", "used_for": "wearing clothes", "can_do": "cover skin", "location": "clothes closet", "material": "soft cloth", "property": "soft cloth", "part_of": "a clothing set"},
    {"object": "window", "is_a": "glass object", "used_for": "letting light", "can_do": "reflect light", "location": "window frame", "material": "clear glass", "property": "clear surface", "part_of": "a window assembly"},
    {"object": "lamp", "is_a": "household item", "used_for": "lighting rooms", "can_do": "shine brightly", "location": "bedroom room", "material": "hard plastic", "property": "warm light", "part_of": "a bedroom set"},
]


def build_expanded_items(max_items: int | None, relations: list[str], frames: list[str]) -> list[dict[str, Any]]:
    wanted_relations = set(relations)
    wanted_frames = set(frames)
    rows: list[dict[str, Any]] = []
    for entry in EXPANDED_OBJECTS:
        obj = entry["object"]
        for relation, rel_frames in RELATION_FRAMES.items():
            if wanted_relations and relation not in wanted_relations:
                continue
            target = entry[relation]
            pool = RELATION_DISTRACTOR_POOLS[relation]
            distractors = [x for x in pool if x != target][:4]
            for frame_key, frame in rel_frames:
                if wanted_frames and frame_key not in wanted_frames:
                    continue
                rows.append(
                    {
                        "object": obj,
                        "relation": relation,
                        "target": target,
                        "distractors": distractors,
                        "frame_key": frame_key,
                        "clean_prompt": frame.format(obj=obj),
                    }
                )
    if not max_items or max_items >= len(rows):
        return rows
    idxs = sorted({round(i * (len(rows) - 1) / max(max_items - 1, 1)) for i in range(max_items)})
    return [rows[i] for i in idxs]


def avg(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def find_matched_source(items: list[dict[str, Any]], idx: int) -> dict[str, Any] | None:
    item = items[idx]
    clean_values = {item["target"], *item["distractors"]}
    pool = [
        x for x in items
        if x is not item
        and x["object"] != item["object"]
        and x["relation"] != item["relation"]
        and x["target"] not in clean_values
    ]
    if not pool:
        return None
    return pool[(idx * 19 + 3) % len(pool)]


def find_mismatch_frame_source(items: list[dict[str, Any]], idx: int, matched: dict[str, Any]) -> dict[str, Any] | None:
    item = items[idx]
    pool = [
        x for x in items
        if x is not item
        and x is not matched
        and x["relation"] != item["relation"]
        and x["relation"] != matched["relation"]
        and x["object"] != item["object"]
    ]
    if not pool:
        return None
    return pool[(idx * 23 + 11) % len(pool)]


def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [v for v in vals if v["base_clean_rank"] == 1]
    return {
        "n": len(vals),
        "eligible_n": len(eligible),
        "clean_drop": avg([float(v["clean_drop"]) for v in vals]),
        "matched_gain": avg([float(v["matched_gain"]) for v in vals]),
        "base_clean_top1": avg([1.0 if v["base_clean_rank"] == 1 else 0.0 for v in vals]),
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
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE77_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    layer_pairs = parse_layer_pairs(args.layer_pairs)
    items = build_expanded_items(args.max_items, parse_csv(args.relations), parse_csv(args.frames))
    log(f"Phase77 model={args.model} items={len(items)} layer_pairs={layer_pairs}")

    results: dict[str, Any] = {
        "phase": 77,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "balanced_cross_relation_joint_closure",
        "layer_pairs": layer_pairs,
        "module": args.module,
        "relations": sorted({x["relation"] for x in items}),
        "num_items": len(items),
        "num_objects": len(EXPANDED_OBJECTS),
        "rows": [],
        "summary": {},
    }
    t0 = time.time()

    for destroy_layer, restore_layer in layer_pairs:
        for idx, item in enumerate(items):
            matched = find_matched_source(items, idx)
            if matched is None:
                continue
            mismatch = find_mismatch_frame_source(items, idx, matched)
            if mismatch is None:
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

            conditions: dict[str, tuple[list[tuple[int, torch.Tensor]], list[tuple[int, torch.Tensor]]]] = {
                "object_only_matched": ([(op, matched_obj_destroy)], []),
                "frame_only_matched": ([(fp, matched_frame_destroy)], []),
                "joint_matched": ([(op, matched_obj_destroy), (fp, matched_frame_destroy)], []),
                "joint_mismatched_frame": ([(op, matched_obj_destroy), (fp, mismatch_frame_destroy)], []),
                "joint_restore_object_only": ([(op, matched_obj_destroy), (fp, matched_frame_destroy)], [(op, clean_obj_restore)]),
                "joint_restore_frame_only": ([(op, matched_obj_destroy), (fp, matched_frame_destroy)], [(fp, clean_frame_restore)]),
                "joint_restore_both": ([(op, matched_obj_destroy), (fp, matched_frame_destroy)], [(op, clean_obj_restore), (fp, clean_frame_restore)]),
            }

            for cond, (destroy_patches, restore_patches) in conditions.items():
                patched_scores = {
                    v: fullseq_logprob_multi(
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
                        "matched_frame_key": matched["frame_key"],
                        "matched_target": matched["target"],
                        "mismatch_object": mismatch["object"],
                        "mismatch_relation": mismatch["relation"],
                        "mismatch_frame_key": mismatch["frame_key"],
                        "mismatch_target": mismatch["target"],
                        "candidate_count": len(candidates),
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
                        "base_top": base_clean_stats["top"],
                        "patched_top": patched_clean_stats["top"],
                    }
                )

            if (idx + 1) % args.progress_every == 0:
                log(f"pair={destroy_layer}->{restore_layer} item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        partial = out_dir / f"{args.model}_phase77_balanced_cross_relation_joint_closure.partial.json"
        partial.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    results["summary"] = summarize_rows(results["rows"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_phase77_balanced_cross_relation_joint_closure.json"
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=56)
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
