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
from phase68_object_attribute_natural_exchange import (  # noqa: E402
    find_subseq,
    load_model,
    parse_csv,
    token_ids,
)
from phase70_object_relation_value_closure import parse_layer_pairs  # noqa: E402
from phase72_object_relation_value_fullseq_closure import (  # noqa: E402
    capture_state,
    fullseq_logprob,
    stats_from_scores,
)


def log(msg: str = "") -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


BALANCED_OBJECTS: list[dict[str, Any]] = [
    {
        "object": "knife",
        "is_a": ("metal tool", ["small bird", "sweet fruit", "home animal", "green plant"]),
        "used_for": ("cutting food", ["drinking water", "flying fast", "sleeping well", "growing leaves"]),
        "can_do": ("cut food", ["drink water", "fly fast", "grow leaves", "bark loudly"]),
        "location": ("home kitchen", ["blue sky", "deep forest", "fresh water", "school building"]),
        "material": ("shiny metal", ["soft cloth", "hard wood", "clear water", "green leaf"]),
        "property": ("sharp edge", ["soft fur", "sweet taste", "deep blue", "fresh smell"]),
    },
    {
        "object": "cup",
        "is_a": ("small container", ["large bird", "green plant", "road vehicle", "wild animal"]),
        "used_for": ("drinking water", ["cutting food", "flying fast", "writing notes", "growing leaves"]),
        "can_do": ("hold water", ["fly fast", "bark loudly", "grow leaves", "cut food"]),
        "location": ("home kitchen", ["blue sky", "deep forest", "hot desert", "school building"]),
        "material": ("clear glass", ["soft cloth", "hard wood", "fresh water", "green leaf"]),
        "property": ("round shape", ["sharp edge", "sweet taste", "loud sound", "deep blue"]),
    },
    {
        "object": "apple",
        "is_a": ("sweet fruit", ["metal tool", "small bird", "road vehicle", "home animal"]),
        "used_for": ("eating food", ["cutting wood", "drinking water", "flying fast", "writing notes"]),
        "can_do": ("grow seeds", ["cut food", "fly fast", "hold water", "bark loudly"]),
        "location": ("grocery store", ["blue sky", "deep forest", "fresh water", "school building"]),
        "material": ("soft flesh", ["shiny metal", "hard wood", "clear glass", "black rubber"]),
        "property": ("sweet taste", ["sharp edge", "deep blue", "loud sound", "rough bark"]),
    },
    {
        "object": "bird",
        "is_a": ("small animal", ["metal tool", "sweet fruit", "road vehicle", "wooden object"]),
        "used_for": ("making nests", ["cutting food", "drinking water", "writing notes", "holding water"]),
        "can_do": ("fly fast", ["cut food", "hold water", "rust slowly", "write notes"]),
        "location": ("blue sky", ["home kitchen", "fresh water", "school building", "coat pocket"]),
        "material": ("soft feathers", ["shiny metal", "hard wood", "clear glass", "white paper"]),
        "property": ("light body", ["sharp edge", "sweet taste", "hard surface", "deep black"]),
    },
    {
        "object": "fish",
        "is_a": ("water animal", ["metal tool", "sweet fruit", "road vehicle", "wooden object"]),
        "used_for": ("eating food", ["cutting wood", "writing notes", "drinking water", "holding water"]),
        "can_do": ("swim well", ["fly fast", "write notes", "rust slowly", "cut food"]),
        "location": ("fresh water", ["blue sky", "home kitchen", "hot desert", "school building"]),
        "material": ("soft flesh", ["shiny metal", "hard wood", "clear glass", "white paper"]),
        "property": ("wet skin", ["sharp edge", "dry sand", "deep black", "sweet taste"]),
    },
    {
        "object": "phone",
        "is_a": ("electronic device", ["small bird", "sweet fruit", "green plant", "wild animal"]),
        "used_for": ("making calls", ["cutting food", "drinking water", "flying fast", "growing leaves"]),
        "can_do": ("send messages", ["fly fast", "grow leaves", "bark loudly", "cut food"]),
        "location": ("coat pocket", ["blue sky", "fresh water", "deep forest", "hot desert"]),
        "material": ("shiny metal", ["soft cloth", "fresh water", "green leaf", "sweet fruit"]),
        "property": ("bright screen", ["sweet taste", "soft fur", "rough bark", "sharp edge"]),
    },
    {
        "object": "book",
        "is_a": ("printed object", ["small bird", "sweet fruit", "metal tool", "wild animal"]),
        "used_for": ("reading stories", ["cutting food", "drinking water", "flying fast", "growing leaves"]),
        "can_do": ("hold words", ["fly fast", "bark loudly", "grow leaves", "cut food"]),
        "location": ("school building", ["blue sky", "fresh water", "hot desert", "coat pocket"]),
        "material": ("white paper", ["shiny metal", "soft cloth", "fresh water", "green leaf"]),
        "property": ("many pages", ["sweet taste", "soft fur", "sharp edge", "deep blue"]),
    },
    {
        "object": "tree",
        "is_a": ("green plant", ["metal tool", "sweet fruit", "road vehicle", "water animal"]),
        "used_for": ("giving shade", ["cutting food", "drinking water", "writing notes", "flying fast"]),
        "can_do": ("grow leaves", ["fly fast", "write notes", "hold water", "cut food"]),
        "location": ("deep forest", ["home kitchen", "blue sky", "fresh water", "coat pocket"]),
        "material": ("hard wood", ["shiny metal", "soft cloth", "clear glass", "white paper"]),
        "property": ("rough bark", ["sweet taste", "sharp edge", "deep blue", "soft fur"]),
    },
    {
        "object": "dog",
        "is_a": ("home animal", ["metal tool", "sweet fruit", "green plant", "road vehicle"]),
        "used_for": ("guarding homes", ["cutting food", "drinking water", "writing notes", "growing leaves"]),
        "can_do": ("bark loudly", ["fly fast", "grow leaves", "hold water", "cut food"]),
        "location": ("family home", ["blue sky", "fresh water", "hot desert", "coat pocket"]),
        "material": ("soft fur", ["shiny metal", "hard wood", "clear glass", "white paper"]),
        "property": ("loud bark", ["sweet taste", "sharp edge", "deep blue", "rough bark"]),
    },
    {
        "object": "car",
        "is_a": ("road vehicle", ["small bird", "sweet fruit", "green plant", "home animal"]),
        "used_for": ("driving roads", ["cutting food", "drinking water", "writing notes", "growing leaves"]),
        "can_do": ("carry people", ["fly fast", "grow leaves", "bark loudly", "hold water"]),
        "location": ("city road", ["blue sky", "fresh water", "deep forest", "coat pocket"]),
        "material": ("shiny metal", ["soft cloth", "fresh water", "green leaf", "white paper"]),
        "property": ("loud engine", ["sweet taste", "soft fur", "rough bark", "deep blue"]),
    },
    {
        "object": "rose",
        "is_a": ("green plant", ["metal tool", "road vehicle", "water animal", "wooden object"]),
        "used_for": ("giving flowers", ["cutting food", "drinking water", "writing notes", "flying fast"]),
        "can_do": ("bloom brightly", ["fly fast", "write notes", "hold water", "cut food"]),
        "location": ("garden bed", ["blue sky", "fresh water", "school building", "coat pocket"]),
        "material": ("soft petals", ["shiny metal", "hard wood", "clear glass", "white paper"]),
        "property": ("red flower", ["sharp edge", "loud bark", "deep blue", "hard surface"]),
    },
    {
        "object": "hammer",
        "is_a": ("metal tool", ["small bird", "sweet fruit", "green plant", "water animal"]),
        "used_for": ("hitting nails", ["drinking water", "flying fast", "reading stories", "growing leaves"]),
        "can_do": ("break rocks", ["fly fast", "grow leaves", "bark loudly", "hold water"]),
        "location": ("tool box", ["blue sky", "fresh water", "deep forest", "school building"]),
        "material": ("shiny metal", ["soft cloth", "fresh water", "green leaf", "sweet fruit"]),
        "property": ("hard head", ["sweet taste", "soft fur", "deep blue", "wet skin"]),
    },
]


RELATION_FRAMES: dict[str, list[tuple[str, str]]] = {
    "is_a": [
        ("kind", "A {obj} is a kind of"),
        ("simple", "A {obj} is a"),
        ("usually", "The {obj} is usually a"),
    ],
    "used_for": [
        ("used_for", "A {obj} is used for"),
        ("people_use", "People use a {obj} for"),
        ("helps_with", "A {obj} helps with"),
    ],
    "can_do": [
        ("can", "A {obj} can"),
        ("usually_can", "A {obj} usually can"),
        ("often_can", "The {obj} often can"),
    ],
    "location": [
        ("found_in", "A {obj} is usually found in"),
        ("often_in", "A {obj} is often in"),
        ("belongs_in", "The {obj} belongs in"),
    ],
    "material": [
        ("made_of", "A {obj} is usually made of"),
        ("the_made_of", "The {obj} is made of"),
        ("built_from", "A {obj} can be built from"),
    ],
    "property": [
        ("looks", "A {obj} usually looks"),
        ("feels", "A {obj} usually feels"),
        ("is_often", "A {obj} is often"),
    ],
}


def build_items(max_items: int | None, relations: list[str], frames: list[str]) -> list[dict[str, Any]]:
    wanted_relations = set(relations)
    wanted_frames = set(frames)
    rows: list[dict[str, Any]] = []
    for entry in BALANCED_OBJECTS:
        obj = entry["object"]
        for relation, rel_frames in RELATION_FRAMES.items():
            if wanted_relations and relation not in wanted_relations:
                continue
            target, distractors = entry[relation]
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


def get_frame_positions(tokenizer: Any, prompt: str, obj: str) -> dict[str, int | None]:
    ids = token_ids(tokenizer, prompt)
    match = None
    for variant in (obj, " " + obj):
        match = find_subseq(ids, token_ids(tokenizer, variant))
        if match is not None:
            break
    object_last = match[1] if match else None
    frame_first = object_last + 1 if object_last is not None and object_last + 1 < len(ids) else None
    return {
        "object_last": object_last,
        "frame_first": frame_first,
        "frame_last": len(ids) - 1 if ids else None,
    }


def find_control(items: list[dict[str, Any]], idx: int, control_type: str) -> dict[str, Any] | None:
    item = items[idx]
    if control_type == "wrong_relation_same_object":
        pool = [x for x in items if x is not item and x["object"] == item["object"] and x["relation"] != item["relation"]]
    elif control_type == "same_relation_other_frame":
        pool = [
            x for x in items
            if x is not item
            and x["object"] == item["object"]
            and x["relation"] == item["relation"]
            and x["frame_key"] != item["frame_key"]
        ]
    elif control_type == "same_relation_frame_other_object":
        pool = [
            x for x in items
            if x is not item
            and x["object"] != item["object"]
            and x["relation"] == item["relation"]
            and x["frame_key"] == item["frame_key"]
        ]
    else:
        raise ValueError(f"unknown control_type={control_type}")
    if not pool:
        return None
    return pool[(idx * 11 + len(control_type)) % len(pool)]


def avg(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [v for v in vals if v["clean_target_rank"] == 1]
    return {
        "n": len(vals),
        "eligible_n": len(eligible),
        "destroy_drop": avg([float(v["destroy_drop"]) for v in vals]),
        "restore_gain": avg([float(v["restore_gain"]) for v in vals]),
        "restore_to_clean_gap": avg([float(v["restore_to_clean_gap"]) for v in vals]),
        "eligible_destroy_drop": avg([float(v["destroy_drop"]) for v in eligible]),
        "eligible_restore_gain": avg([float(v["restore_gain"]) for v in eligible]),
        "eligible_restore_to_clean_gap": avg([float(v["restore_to_clean_gap"]) for v in eligible]),
        "clean_top1": avg([1.0 if v["clean_target_rank"] == 1 else 0.0 for v in vals]),
        "destroy_top1": avg([1.0 if v["destroy_target_rank"] == 1 else 0.0 for v in vals]),
        "restore_top1": avg([1.0 if v["restore_target_rank"] == 1 else 0.0 for v in vals]),
        "eligible_destroy_top1": avg([1.0 if v["destroy_target_rank"] == 1 else 0.0 for v in eligible]),
        "eligible_restore_top1": avg([1.0 if v["restore_target_rank"] == 1 else 0.0 for v in eligible]),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_control: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_control_path: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    by_control_relation: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ct = str(row["control_type"])
        dl, rl, pos, rel = int(row["destroy_layer"]), int(row["restore_layer"]), str(row["position"]), str(row["relation"])
        by_control[ct].append(row)
        by_control_path[(ct, dl, rl, pos)].append(row)
        by_control_relation[(ct, rel)].append(row)
    return {
        "by_control": {k: group_summary(v) for k, v in by_control.items()},
        "by_control_path": {f"{ct}:L{dl}->L{rl}:{pos}": group_summary(v) for (ct, dl, rl, pos), v in by_control_path.items()},
        "by_control_relation": {f"{ct}:{rel}": group_summary(v) for (ct, rel), v in by_control_relation.items()},
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE75_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    layer_pairs = parse_layer_pairs(args.layer_pairs)
    positions = parse_csv(args.positions)
    control_types = parse_csv(args.control_types)
    items = build_items(args.max_items, parse_csv(args.relations), parse_csv(args.frames))
    log(f"Phase75 model={args.model} items={len(items)} layer_pairs={layer_pairs} positions={positions} controls={control_types}")

    results: dict[str, Any] = {
        "phase": 75,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "relation_frame_token_intervention",
        "layer_pairs": layer_pairs,
        "module": args.module,
        "positions": positions,
        "control_types": control_types,
        "relations": sorted({x["relation"] for x in items}),
        "num_items": len(items),
        "rows": [],
        "summary": {},
    }
    t0 = time.time()

    for destroy_layer, restore_layer in layer_pairs:
        for idx, item in enumerate(items):
            clean_pos = get_frame_positions(tokenizer, item["clean_prompt"], item["object"])
            h_clean_r = capture_state(model, tokenizer, device, layers[restore_layer], args.module, item["clean_prompt"], args.max_length)
            values = [item["target"]] + item["distractors"]
            clean_scores = {v: fullseq_logprob(model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length) for v in values}
            clean_stats = stats_from_scores(clean_scores, item["target"], item["distractors"])

            for control_type in control_types:
                control = find_control(items, idx, control_type)
                if control is None:
                    continue
                control_pos = get_frame_positions(tokenizer, control["clean_prompt"], control["object"])
                h_control_d = capture_state(model, tokenizer, device, layers[destroy_layer], args.module, control["clean_prompt"], args.max_length)
                for pos_name in positions:
                    sp = clean_pos.get(pos_name)
                    cp = control_pos.get(pos_name)
                    if sp is None or cp is None:
                        continue
                    destroy_scores = {
                        v: fullseq_logprob(
                            model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length,
                            args.module, destroy_layer, None, int(sp), h_control_d[int(cp)], None
                        )
                        for v in values
                    }
                    restore_scores = {
                        v: fullseq_logprob(
                            model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length,
                            args.module, destroy_layer, restore_layer, int(sp), h_control_d[int(cp)], h_clean_r[int(sp)]
                        )
                        for v in values
                    }
                    destroy_stats = stats_from_scores(destroy_scores, item["target"], item["distractors"])
                    restore_stats = stats_from_scores(restore_scores, item["target"], item["distractors"])
                    results["rows"].append(
                        {
                            "destroy_layer": destroy_layer,
                            "restore_layer": restore_layer,
                            "module": args.module,
                            "position": pos_name,
                            "control_type": control_type,
                            "relation": item["relation"],
                            "frame_key": item["frame_key"],
                            "object": item["object"],
                            "target": item["target"],
                            "control_relation": control["relation"],
                            "control_frame_key": control["frame_key"],
                            "control_object": control["object"],
                            "control_target": control["target"],
                            "clean_margin": clean_stats["margin"],
                            "destroy_margin": destroy_stats["margin"],
                            "restore_margin": restore_stats["margin"],
                            "destroy_drop": clean_stats["margin"] - destroy_stats["margin"],
                            "restore_gain": restore_stats["margin"] - destroy_stats["margin"],
                            "restore_to_clean_gap": clean_stats["margin"] - restore_stats["margin"],
                            "clean_target_rank": clean_stats["rank"],
                            "destroy_target_rank": destroy_stats["rank"],
                            "restore_target_rank": restore_stats["rank"],
                            "clean_top": clean_stats["top"],
                            "destroy_top": destroy_stats["top"],
                            "restore_top": restore_stats["top"],
                        }
                    )
            if (idx + 1) % args.progress_every == 0:
                log(f"pair={destroy_layer}->{restore_layer} item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        partial = out_dir / f"{args.model}_phase75_relation_frame_token_intervention.partial.json"
        partial.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    results["summary"] = summarize_rows(results["rows"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_phase75_relation_frame_token_intervention.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--layer-pairs", required=True)
    parser.add_argument("--module", default="resid_out")
    parser.add_argument("--positions", default="frame_first,frame_last")
    parser.add_argument("--control-types", default="wrong_relation_same_object,same_relation_other_frame,same_relation_frame_other_object")
    parser.add_argument("--relations", default="")
    parser.add_argument("--frames", default="")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=112)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=24)
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
