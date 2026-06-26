#!/usr/bin/env python3
"""
Phase 652: Intent Gate Layer Localization and Component Narrowing Audit.

Phase 651 showed that task intent can open or suppress value-token support, but
the L14-L22 interval scan was broad and slow. This phase narrows the audit to
single layers and component-level rank deltas, without per-patch generation.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase630_distributed_format_route_multisource import collect_positions_components  # noqa: E402
from phase635_final_readout_projection_bridge_audit import final_state_probe  # noqa: E402
from phase647_protocol_writer_graph_audit import COMPONENTS, make_multi_patch  # noqa: E402
from phase651_task_intent_gate_protocol_boundary_audit import (  # noqa: E402
    TASKS,
    ladder_row,
    make_prompt,
    position_units,
    select_cases,
)


OUT_ROOT = Path("results/glm5_phase652_intent_gate_layer_localization_audit")
TASK_ORDER = ["explanation_required", "yes_no_required"]
POSITION_UNITS = ["intent_word", "instruction_span", "label_aligned", "separator", "relation_tail"]
SCAN_LAYERS = list(range(14, 23))
DIRECTIONS = ["value_to_task", "task_to_value"]
CONTROLS = ["restore"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def single_layer_patch(
    target_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    source_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    target_pos: List[int],
    layer: int,
    component: str,
    control: str,
    seed: int,
) -> List[Tuple[int, str, List[int], List[torch.Tensor]]]:
    return make_multi_patch(target_cache, source_cache, target_pos, [layer], component, control, seed)


def summarize(rows: List[Dict]) -> Dict:
    by_key: Dict[Tuple, Dict] = {}
    for row in rows:
        key = (
            row["pair_task"],
            row["eval_task"],
            row["kind"],
            row.get("position_unit"),
            row.get("direction"),
            row.get("layer"),
            row.get("component"),
            row.get("control"),
        )
        item = by_key.setdefault(key, {
            "pair_task": row["pair_task"],
            "eval_task": row["eval_task"],
            "mode": row["mode"],
            "kind": row["kind"],
            "position_unit": row.get("position_unit"),
            "direction": row.get("direction"),
            "layer": row.get("layer"),
            "component": row.get("component"),
            "control": row.get("control"),
            "n": 0,
            "tok0_hit": 0,
            "newline_top0": 0,
            "sum_rank": 0.0,
            "sum_prefix_minus_newline": 0.0,
            "top0_category": {},
            "top0_text": {},
        })
        item["n"] += 1
        item["tok0_hit"] += int(row["top0_id"] == row["prefix_id"])
        item["newline_top0"] += int(row["top0_category"] == "newline")
        item["sum_rank"] += row["prefix_rank"]
        item["sum_prefix_minus_newline"] += row["prefix_minus_newline"]
        item["top0_category"].setdefault(row["top0_category"], 0)
        item["top0_category"][row["top0_category"]] += 1
        item["top0_text"].setdefault(row["top0_text_clean"], 0)
        item["top0_text"][row["top0_text_clean"]] += 1

    out = []
    for item in by_key.values():
        n = max(1, item["n"])
        row = dict(item)
        row["tok0_rate"] = item["tok0_hit"] / n
        row["newline_top0_rate"] = item["newline_top0"] / n
        row["mean_prefix_rank"] = item["sum_rank"] / n
        row["mean_prefix_minus_newline"] = item["sum_prefix_minus_newline"] / n
        row["top0_category"] = dict(sorted(row["top0_category"].items(), key=lambda kv: kv[1], reverse=True))
        row["top0_text"] = dict(sorted(row["top0_text"].items(), key=lambda kv: kv[1], reverse=True)[:8])
        out.append(row)

    baseline = {
        (r["pair_task"], r["eval_task"]): r
        for r in out
        if r["kind"] == "baseline"
    }
    for row in out:
        base = baseline.get((row["pair_task"], row["eval_task"]))
        if base:
            row["baseline_rank"] = base["mean_prefix_rank"]
            row["baseline_tok0"] = base["tok0_hit"]
            row["rank_improvement"] = base["mean_prefix_rank"] - row["mean_prefix_rank"]
            row["tok0_delta"] = row["tok0_hit"] - base["tok0_hit"]
        else:
            row["baseline_rank"] = None
            row["baseline_tok0"] = None
            row["rank_improvement"] = None
            row["tok0_delta"] = None

    absorption = [
        r for r in out
        if r["kind"] == "patch" and r["direction"] == "value_to_task"
    ]
    suppression = [
        r for r in out
        if r["kind"] == "patch" and r["direction"] == "task_to_value"
    ]
    absorption.sort(key=lambda r: (-(r["rank_improvement"] or 0), -r["tok0_delta"], r["mean_prefix_rank"]))
    suppression.sort(key=lambda r: ((r["rank_improvement"] or 0), r["tok0_delta"], r["mean_prefix_rank"]))

    by_layer_component = {}
    for row in out:
        if row["kind"] != "patch":
            continue
        key = f"{row['direction']}::{row['pair_task']}::{row['position_unit']}::{row['component']}"
        by_layer_component.setdefault(key, []).append(row)
    for rows0 in by_layer_component.values():
        rows0.sort(key=lambda r: r["layer"])

    out.sort(key=lambda r: (
        r["pair_task"],
        r["eval_task"],
        0 if r["kind"] == "baseline" else 1,
        DIRECTIONS.index(r["direction"]) if r.get("direction") in DIRECTIONS else -1,
        POSITION_UNITS.index(r["position_unit"]) if r.get("position_unit") in POSITION_UNITS else -1,
        r.get("layer") if r.get("layer") is not None else -1,
        COMPONENTS.index(r.get("component")) if r.get("component") in COMPONENTS else -1,
    ))
    return {
        "by_mode": out,
        "strongest_absorption": absorption[:120],
        "strongest_suppression": suppression[:120],
        "by_layer_component": by_layer_component,
    }


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = [li for li in SCAN_LAYERS if 0 <= li < info.n_layers]
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        relation_pool = sorted({c["relation"] for c in raw_cases})
        selected, selection_stats = select_cases(
            model, tokenizer, device, raw_cases, values, args.max_cases, relation_pool
        )
        value_prefix_ids = {answer_ids(tokenizer, v)[0] for v in values}
        rows = []
        examples = []
        filtered = {"position_missing": 0, "position_len_mismatch": 0, "empty_patch": 0}
        log(
            f"{args.model}: raw_cases={len(raw_cases)}, selected={len(selected)}, "
            f"tasks={TASK_ORDER}, positions={POSITION_UNITS}, layers={layers}, components={COMPONENTS}"
        )

        for item_i, item in enumerate(selected):
            case = item["case"]
            correct_ids = answer_ids(tokenizer, case["correct"])
            old_wrong = item["base_top_wrong"] or item["repair_top_wrong"] or item["mode_v_top_wrong"] or values[0]
            old_wrong_ids = answer_ids(tokenizer, old_wrong)
            prefix_id = correct_ids[0]
            old_wrong_prefix_id = old_wrong_ids[0]
            value_prompt, value_relation, value_intent = make_prompt(
                case, "short_value_allowed", relation_pool, tokenizer, item["sample_idx"]
            )
            value_units = position_units(tokenizer, value_prompt, case, value_relation, value_intent)
            value_base = final_state_probe(model, tokenizer, device, value_prompt)

            for task_i, task in enumerate(TASK_ORDER):
                task_prompt, task_relation, task_intent = make_prompt(
                    case, task, relation_pool, tokenizer, item["sample_idx"] + task_i * 17
                )
                task_units = position_units(tokenizer, task_prompt, case, task_relation, task_intent)
                task_base = final_state_probe(model, tokenizer, device, task_prompt)
                baseline_specs = [
                    ("value_baseline", "short_value_allowed", value_prompt, value_base),
                    ("task_baseline", task, task_prompt, task_base),
                ]
                for mode, eval_task, prompt, probe in baseline_specs:
                    row = {
                        "sample_idx": item["sample_idx"],
                        "item_idx": item_i,
                        "pair_task": task,
                        "eval_task": eval_task,
                        "mode": mode,
                        "kind": "baseline",
                        "position_unit": None,
                        "direction": None,
                        "layer": None,
                        "component": None,
                        "control": None,
                        "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                        "task_relation": task_relation,
                        "task_intent": task_intent,
                        "prefix_id": prefix_id,
                        "prefix_text": tokenizer.decode([prefix_id]),
                        "old_wrong_prefix_id": old_wrong_prefix_id,
                        "old_wrong_prefix_text": tokenizer.decode([old_wrong_prefix_id]),
                        **ladder_row(tokenizer, probe["logits"], prefix_id, old_wrong_prefix_id, value_prefix_ids, args.top_k),
                    }
                    rows.append(row)
                    if len(examples) < args.example_limit:
                        examples.append(row)

                for pos_unit in POSITION_UNITS:
                    value_pos = value_units.get(pos_unit, [])
                    task_pos = task_units.get(pos_unit, [])
                    if not value_pos or not task_pos:
                        filtered["position_missing"] += 1
                        continue
                    if len(value_pos) != len(task_pos):
                        filtered["position_len_mismatch"] += 1
                        continue
                    value_cache = collect_positions_components(
                        model, tokenizer, device, value_prompt, value_pos, layers, COMPONENTS
                    )
                    task_cache = collect_positions_components(
                        model, tokenizer, device, task_prompt, task_pos, layers, COMPONENTS
                    )
                    for direction in DIRECTIONS:
                        target_cache, source_cache, target_pos, prompt, eval_task = (
                            (task_cache, value_cache, task_pos, task_prompt, task)
                            if direction == "value_to_task"
                            else (value_cache, task_cache, value_pos, value_prompt, "short_value_allowed")
                        )
                        for li in layers:
                            for component in COMPONENTS:
                                for control in CONTROLS:
                                    patches = single_layer_patch(
                                        target_cache,
                                        source_cache,
                                        target_pos,
                                        li,
                                        component,
                                        control,
                                        item["sample_idx"] * 1009 + item_i * 101 + task_i * 17 + li,
                                    )
                                    if not patches:
                                        filtered["empty_patch"] += 1
                                        continue
                                    probe = final_state_probe(model, tokenizer, device, prompt, source_patches=patches)
                                    mode = f"{pos_unit}_{direction}_L{li:02d}_{component}_{control}"
                                    row = {
                                        "sample_idx": item["sample_idx"],
                                        "item_idx": item_i,
                                        "pair_task": task,
                                        "eval_task": eval_task,
                                        "mode": mode,
                                        "kind": "patch",
                                        "position_unit": pos_unit,
                                        "direction": direction,
                                        "layer": li,
                                        "component": component,
                                        "control": control,
                                        "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                                        "task_relation": task_relation,
                                        "task_intent": task_intent,
                                        "prefix_id": prefix_id,
                                        "prefix_text": tokenizer.decode([prefix_id]),
                                        "old_wrong_prefix_id": old_wrong_prefix_id,
                                        "old_wrong_prefix_text": tokenizer.decode([old_wrong_prefix_id]),
                                        **ladder_row(tokenizer, probe["logits"], prefix_id, old_wrong_prefix_id, value_prefix_ids, args.top_k),
                                    }
                                    rows.append(row)
                                    if len(examples) < args.example_limit:
                                        examples.append(row)

        summary = summarize(rows)
        log("Top localized absorption:")
        for row in summary["strongest_absorption"][:18]:
            log(
                f"  {row['pair_task']} {row['position_unit']} L{row['layer']:02d} {row['component']}: "
                f"rank {row['baseline_rank']:.1f}->{row['mean_prefix_rank']:.1f} "
                f"dR={row['rank_improvement']:.1f} tok0={row['baseline_tok0']}->{row['tok0_hit']}"
            )
        log("Top localized suppression:")
        for row in summary["strongest_suppression"][:18]:
            log(
                f"  {row['pair_task']} {row['position_unit']} L{row['layer']:02d} {row['component']}: "
                f"rank {row['baseline_rank']:.1f}->{row['mean_prefix_rank']:.1f} "
                f"dR={row['rank_improvement']:.1f} tok0={row['baseline_tok0']}->{row['tok0_hit']}"
            )

        return {
            "phase": 652,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "layers": layers,
            "components": COMPONENTS,
            "tasks": TASK_ORDER,
            "position_units": POSITION_UNITS,
            "directions": DIRECTIONS,
            "controls": CONTROLS,
            "top_k": args.top_k,
            "n_raw_cases": len(raw_cases),
            "n_selected_items": len(selected),
            "n_mode_rows": len(rows),
            "max_cases": args.max_cases,
            "selection_stats": selection_stats,
            "filtered": filtered,
            "summary": summary,
            "examples": examples,
            "rows": rows if args.save_rows else examples,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--save-rows", action="store_true")
    parser.add_argument("--example-limit", type=int, default=240)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 12
        args.max_cases = 1
        args.top_k = min(args.top_k, 12)
        args.example_limit = 120
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 40)
        args.max_samples = max(args.max_samples, 320)
        args.max_cases = max(args.max_cases, 20)
        args.top_k = max(args.top_k, 20)
        args.example_limit = max(args.example_limit, 320)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase652_{args.model}_intent_gate_layer_localization_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
