from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))

from hf_probe_env import get_layers, release_loaded  # noqa: E402
from phase68_object_attribute_natural_exchange import get_positions, load_model, parse_csv  # noqa: E402
from phase70_object_relation_value_closure import parse_layer_pairs, pick_control  # noqa: E402
from phase72_object_relation_value_fullseq_closure import (  # noqa: E402
    capture_state,
    candidate_ids,
    fullseq_logprob,
    stats_from_scores,
    summarize_rows,
)


def log(msg: str = "") -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


MULTITOKEN_FRAMES: dict[str, list[tuple[str, str]]] = {
    "is_a": [
        ("kind", "A {obj} is a kind of"),
        ("simple", "A {obj} is a"),
        ("usually", "The {obj} is usually a"),
    ],
    "part_of": [
        ("part", "A {obj} is part of"),
        ("belongs", "A {obj} belongs to"),
        ("found_on", "A {obj} can be found on"),
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
        ("lives_in", "A {obj} is often in"),
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


MULTITOKEN_VALUE_DATA: dict[str, list[tuple[str, str, list[str]]]] = {
    "is_a": [
        ("robin", "small bird", ["freshwater fish", "metal tool", "sweet fruit", "road vehicle"]),
        ("sparrow", "small bird", ["freshwater fish", "metal tool", "green plant", "road vehicle"]),
        ("eagle", "large bird", ["freshwater fish", "metal tool", "sweet fruit", "road vehicle"]),
        ("salmon", "freshwater fish", ["small bird", "metal tool", "sweet fruit", "road vehicle"]),
        ("trout", "freshwater fish", ["small bird", "metal tool", "green plant", "road vehicle"]),
        ("shark", "large fish", ["small bird", "metal tool", "sweet fruit", "road vehicle"]),
        ("hammer", "metal tool", ["small bird", "freshwater fish", "sweet fruit", "green plant"]),
        ("knife", "metal tool", ["small bird", "freshwater fish", "sweet fruit", "green plant"]),
        ("apple", "sweet fruit", ["small bird", "freshwater fish", "metal tool", "road vehicle"]),
        ("banana", "sweet fruit", ["small bird", "freshwater fish", "metal tool", "green plant"]),
        ("rose", "green plant", ["small bird", "freshwater fish", "metal tool", "road vehicle"]),
        ("oak", "large plant", ["small bird", "freshwater fish", "metal tool", "sweet fruit"]),
        ("car", "road vehicle", ["small bird", "freshwater fish", "metal tool", "sweet fruit"]),
        ("bus", "road vehicle", ["small bird", "freshwater fish", "metal tool", "green plant"]),
        ("dog", "domestic animal", ["small bird", "freshwater fish", "metal tool", "sweet fruit"]),
        ("cat", "domestic animal", ["small bird", "freshwater fish", "metal tool", "green plant"]),
    ],
    "part_of": [
        ("wheel", "a car", ["a tree", "a shirt", "a book", "a cup"]),
        ("engine", "a car", ["a tree", "a shirt", "a book", "a cup"]),
        ("door", "a car", ["a tree", "a shirt", "a book", "a fish"]),
        ("leaf", "a tree", ["a car", "a shirt", "a book", "a cup"]),
        ("branch", "a tree", ["a car", "a shirt", "a book", "a fish"]),
        ("sleeve", "a shirt", ["a car", "a tree", "a book", "a cup"]),
        ("button", "a shirt", ["a car", "a tree", "a fish", "a cup"]),
        ("page", "a book", ["a car", "a tree", "a shirt", "a cup"]),
        ("cover", "a book", ["a car", "a tree", "a shirt", "a fish"]),
        ("handle", "a cup", ["a car", "a tree", "a shirt", "a book"]),
        ("screen", "a phone", ["a tree", "a shirt", "a book", "a cup"]),
        ("keyboard", "a computer", ["a tree", "a shirt", "a fish", "a cup"]),
        ("roof", "a house", ["a tree", "a shirt", "a book", "a fish"]),
        ("wing", "a bird", ["a car", "a shirt", "a book", "a cup"]),
        ("fin", "a fish", ["a car", "a tree", "a shirt", "a book"]),
        ("tail", "an animal", ["a car", "a tree", "a shirt", "a book"]),
    ],
    "used_for": [
        ("knife", "cutting food", ["drinking water", "writing notes", "sleeping comfortably", "taking pictures"]),
        ("scissors", "cutting paper", ["drinking water", "writing notes", "sleeping comfortably", "taking pictures"]),
        ("saw", "cutting wood", ["drinking water", "writing notes", "sleeping comfortably", "taking pictures"]),
        ("cup", "drinking water", ["cutting food", "writing notes", "sleeping comfortably", "taking pictures"]),
        ("glass", "drinking water", ["cutting food", "writing notes", "sleeping comfortably", "taking pictures"]),
        ("pen", "writing notes", ["cutting food", "drinking water", "sleeping comfortably", "taking pictures"]),
        ("pencil", "writing notes", ["cutting food", "drinking water", "sleeping comfortably", "taking pictures"]),
        ("bed", "sleeping comfortably", ["cutting food", "drinking water", "writing notes", "taking pictures"]),
        ("pillow", "sleeping comfortably", ["cutting food", "drinking water", "writing notes", "taking pictures"]),
        ("camera", "taking pictures", ["cutting food", "drinking water", "writing notes", "sleeping comfortably"]),
        ("broom", "cleaning floors", ["cutting food", "drinking water", "writing notes", "taking pictures"]),
        ("stove", "cooking food", ["cutting food", "drinking water", "writing notes", "sleeping comfortably"]),
        ("phone", "making calls", ["cutting food", "drinking water", "writing notes", "sleeping comfortably"]),
        ("needle", "sewing cloth", ["cutting food", "drinking water", "writing notes", "sleeping comfortably"]),
        ("lamp", "lighting rooms", ["cutting food", "drinking water", "writing notes", "sleeping comfortably"]),
        ("map", "finding places", ["cutting food", "drinking water", "writing notes", "sleeping comfortably"]),
    ],
    "can_do": [
        ("bird", "fly fast", ["swim well", "write words", "rust slowly", "melt quickly"]),
        ("eagle", "fly fast", ["swim well", "write words", "rust slowly", "melt quickly"]),
        ("fish", "swim well", ["fly fast", "write words", "rust slowly", "melt quickly"]),
        ("salmon", "swim well", ["fly fast", "write words", "rust slowly", "melt quickly"]),
        ("student", "write words", ["fly fast", "swim well", "rust slowly", "melt quickly"]),
        ("writer", "write words", ["fly fast", "swim well", "rust slowly", "melt quickly"]),
        ("iron", "rust slowly", ["fly fast", "swim well", "write words", "bloom brightly"]),
        ("steel", "rust slowly", ["fly fast", "swim well", "write words", "melt quickly"]),
        ("ice", "melt quickly", ["fly fast", "swim well", "write words", "rust slowly"]),
        ("flower", "bloom brightly", ["fly fast", "swim well", "write words", "rust slowly"]),
        ("dog", "bark loudly", ["fly fast", "write words", "rust slowly", "melt quickly"]),
        ("cat", "climb trees", ["fly fast", "write words", "rust slowly", "melt quickly"]),
        ("snake", "crawl quietly", ["fly fast", "write words", "rust slowly", "melt quickly"]),
        ("horse", "run fast", ["fly fast", "write words", "rust slowly", "melt quickly"]),
        ("bee", "sting painfully", ["swim well", "write words", "rust slowly", "melt quickly"]),
        ("duck", "swim well", ["write words", "rust slowly", "melt quickly", "bloom brightly"]),
    ],
    "location": [
        ("fish", "fresh water", ["blue sky", "hot desert", "school building", "coat pocket"]),
        ("boat", "fresh water", ["blue sky", "hot desert", "school building", "coat pocket"]),
        ("shark", "deep water", ["blue sky", "hot desert", "school building", "coat pocket"]),
        ("bird", "blue sky", ["fresh water", "hot desert", "school building", "coat pocket"]),
        ("cloud", "blue sky", ["fresh water", "hot desert", "school building", "coat pocket"]),
        ("plane", "blue sky", ["fresh water", "hot desert", "school building", "coat pocket"]),
        ("cactus", "hot desert", ["fresh water", "blue sky", "school building", "coat pocket"]),
        ("camel", "hot desert", ["fresh water", "blue sky", "school building", "coat pocket"]),
        ("teacher", "school building", ["fresh water", "blue sky", "hot desert", "coat pocket"]),
        ("doctor", "hospital room", ["fresh water", "blue sky", "hot desert", "coat pocket"]),
        ("plate", "home kitchen", ["fresh water", "blue sky", "hot desert", "coat pocket"]),
        ("bed", "bedroom room", ["fresh water", "blue sky", "hot desert", "coat pocket"]),
        ("tree", "deep forest", ["fresh water", "blue sky", "hot desert", "coat pocket"]),
        ("bear", "deep forest", ["fresh water", "blue sky", "hot desert", "coat pocket"]),
        ("coin", "coat pocket", ["fresh water", "blue sky", "hot desert", "school building"]),
        ("wallet", "coat pocket", ["fresh water", "blue sky", "hot desert", "school building"]),
    ],
    "material": [
        ("spoon", "shiny metal", ["hard wood", "soft cloth", "clear glass", "white paper"]),
        ("coin", "shiny metal", ["hard wood", "soft cloth", "clear glass", "white paper"]),
        ("key", "shiny metal", ["hard wood", "soft cloth", "clear glass", "white paper"]),
        ("table", "hard wood", ["shiny metal", "soft cloth", "clear glass", "white paper"]),
        ("chair", "hard wood", ["shiny metal", "soft cloth", "clear glass", "white paper"]),
        ("door", "hard wood", ["shiny metal", "soft cloth", "clear glass", "white paper"]),
        ("shirt", "soft cloth", ["shiny metal", "hard wood", "clear glass", "white paper"]),
        ("sock", "soft cloth", ["shiny metal", "hard wood", "clear glass", "white paper"]),
        ("curtain", "soft cloth", ["shiny metal", "hard wood", "clear glass", "white paper"]),
        ("window", "clear glass", ["shiny metal", "hard wood", "soft cloth", "white paper"]),
        ("bottle", "clear glass", ["shiny metal", "hard wood", "soft cloth", "white paper"]),
        ("book", "white paper", ["shiny metal", "hard wood", "soft cloth", "clear glass"]),
        ("letter", "white paper", ["shiny metal", "hard wood", "soft cloth", "clear glass"]),
        ("tire", "black rubber", ["shiny metal", "hard wood", "soft cloth", "white paper"]),
        ("wall", "red brick", ["shiny metal", "hard wood", "soft cloth", "white paper"]),
        ("ring", "yellow gold", ["hard wood", "soft cloth", "clear glass", "white paper"]),
    ],
    "property": [
        ("apple", "bright red", ["deep blue", "rough stone", "shiny metal", "soft cloth"]),
        ("cherry", "bright red", ["deep blue", "rough stone", "shiny metal", "soft cloth"]),
        ("sky", "deep blue", ["bright red", "rough stone", "shiny metal", "soft cloth"]),
        ("ocean", "deep blue", ["bright red", "rough stone", "shiny metal", "soft cloth"]),
        ("silk", "soft cloth", ["rough stone", "shiny metal", "bright red", "deep blue"]),
        ("pillow", "soft cloth", ["rough stone", "shiny metal", "bright red", "deep blue"]),
        ("rock", "rough stone", ["soft cloth", "shiny metal", "bright red", "deep blue"]),
        ("sandpaper", "rough stone", ["soft cloth", "shiny metal", "bright red", "deep blue"]),
        ("spoon", "shiny metal", ["rough stone", "soft cloth", "bright red", "deep blue"]),
        ("knife", "shiny metal", ["rough stone", "soft cloth", "bright red", "deep blue"]),
        ("snow", "pure white", ["bright red", "deep blue", "rough stone", "shiny metal"]),
        ("coal", "deep black", ["pure white", "deep blue", "rough stone", "shiny metal"]),
        ("fire", "very hot", ["deep cold", "deep blue", "rough stone", "soft cloth"]),
        ("ice", "deep cold", ["very hot", "deep blue", "rough stone", "soft cloth"]),
        ("sugar", "very sweet", ["very sour", "deep blue", "rough stone", "shiny metal"]),
        ("lemon", "very sour", ["very sweet", "deep blue", "rough stone", "shiny metal"]),
    ],
}


def build_multitoken_items(max_items: int | None, relations: list[str], frames: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wanted_relations = set(relations)
    wanted_frames = set(frames)
    for relation, examples in MULTITOKEN_VALUE_DATA.items():
        if wanted_relations and relation not in wanted_relations:
            continue
        for obj, target, distractors in examples:
            for frame_key, frame in MULTITOKEN_FRAMES[relation]:
                if wanted_frames and frame_key not in wanted_frames:
                    continue
                rows.append(
                    {
                        "relation": relation,
                        "object": obj,
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


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE73_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    layer_pairs = parse_layer_pairs(args.layer_pairs)
    positions = parse_csv(args.positions)
    items = build_multitoken_items(args.max_items, parse_csv(args.relations), parse_csv(args.frames))
    log(f"Phase73 model={args.model} items={len(items)} layer_pairs={layer_pairs} positions={positions}")

    results: dict[str, Any] = {
        "phase": 73,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "multitoken_object_relation_value_fullseq_closure",
        "layer_pairs": layer_pairs,
        "module": args.module,
        "positions": positions,
        "relations": sorted({x["relation"] for x in items}),
        "num_items": len(items),
        "rows": [],
        "summary": {},
    }
    t0 = time.time()

    for destroy_layer, restore_layer in layer_pairs:
        for idx, item in enumerate(items):
            control = pick_control(items, idx)
            clean_pos = get_positions(tokenizer, item["clean_prompt"], item["object"])
            control_pos = get_positions(tokenizer, control["clean_prompt"], control["object"])
            h_control_d = capture_state(model, tokenizer, device, layers[destroy_layer], args.module, control["clean_prompt"], args.max_length)
            h_clean_r = capture_state(model, tokenizer, device, layers[restore_layer], args.module, item["clean_prompt"], args.max_length)
            values = [item["target"]] + item["distractors"]
            clean_scores = {v: fullseq_logprob(model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length) for v in values}
            clean_stats = stats_from_scores(clean_scores, item["target"], item["distractors"])
            target_token_len = len(candidate_ids(tokenizer, item["target"]))
            distractor_token_lens = [len(candidate_ids(tokenizer, v)) for v in item["distractors"]]

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
                        "relation": item["relation"],
                        "frame_key": item["frame_key"],
                        "object": item["object"],
                        "target": item["target"],
                        "target_token_len": target_token_len,
                        "distractor_token_lens": distractor_token_lens,
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
        partial = out_dir / f"{args.model}_phase73_multitoken_value_closure.partial.json"
        partial.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    results["summary"] = summarize_rows(results["rows"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_phase73_multitoken_value_closure.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--layer-pairs", required=True)
    parser.add_argument("--module", default="resid_out")
    parser.add_argument("--positions", default="object_first,object_last")
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
