#!/usr/bin/env python3
"""
Phase 653: Localized Intent-Gate Control and Generation Closure Audit.

Phase 652 localized task-intent carriers with rank-only restore patches. This
phase keeps only the strongest localized sites, adds random/reverse controls,
and checks whether those sites also change short natural generation.
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
from phase628_prefix_format_semantic_integration import generation_eval  # noqa: E402
from phase630_distributed_format_route_multisource import collect_positions_components  # noqa: E402
from phase635_final_readout_projection_bridge_audit import final_state_probe, greedy_generate_bridge  # noqa: E402
from phase647_protocol_writer_graph_audit import make_multi_patch  # noqa: E402
from phase651_task_intent_gate_protocol_boundary_audit import (  # noqa: E402
    ladder_row,
    make_prompt,
    position_units,
    select_cases,
)


OUT_ROOT = Path("results/glm5_phase653_localized_intent_gate_generation_closure")
TASK_ORDER = ["explanation_required", "yes_no_required"]
CONTROLS = ["restore", "random", "reverse"]
DIRECTIONS = ["value_to_task", "task_to_value"]

SITE_SPECS = {
    "qwen3": [
        {
            "name": "early_peak_layer_out",
            "positions": ["label_aligned", "separator", "relation_tail"],
            "layers": [14, 15, 16, 17],
            "components": ["layer_out"],
        },
        {
            "name": "separator_input_edge",
            "positions": ["separator"],
            "layers": [14],
            "components": ["layer_input"],
        },
        {
            "name": "mid_suppression_attn_mlp",
            "positions": ["label_aligned", "separator"],
            "layers": [16, 17, 18],
            "components": ["attn_out", "mlp_out"],
        },
    ],
    "glm4": [
        {
            "name": "late_peak_layer_out",
            "positions": ["label_aligned", "separator", "relation_tail"],
            "layers": [21, 22],
            "components": ["layer_out"],
        },
        {
            "name": "l22_peak_layer_out",
            "positions": ["label_aligned", "separator", "relation_tail"],
            "layers": [22],
            "components": ["layer_out"],
        },
        {
            "name": "relation_separator_l21_l22",
            "positions": ["separator", "relation_tail"],
            "layers": [21, 22],
            "components": ["layer_out"],
        },
    ],
    "deepseek7b": [
        {
            "name": "late_peak_layer_out",
            "positions": ["label_aligned", "separator", "relation_tail"],
            "layers": [20, 21, 22],
            "components": ["layer_out"],
        },
        {
            "name": "l22_peak_layer_out",
            "positions": ["label_aligned", "separator", "relation_tail"],
            "layers": [22],
            "components": ["layer_out"],
        },
        {
            "name": "relation_separator_l22",
            "positions": ["separator", "relation_tail"],
            "layers": [22],
            "components": ["layer_out"],
        },
    ],
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def classify_text(text: str) -> Dict[str, int]:
    stripped = text.strip()
    lower = stripped.lower()
    word_count = len(stripped.replace("\n", " ").split())
    return {
        "empty": int(not stripped),
        "short": int(word_count <= 1),
        "starts_yes_no": int(lower.startswith("yes") or lower.startswith("no")),
        "explanation_signal": int(any(s in lower for s in ["because", "therefore", "means", " is ", " are "])),
        "full_sentence_like": int(word_count >= 4),
        "has_newline": int("\n" in text),
    }


def site_layers_for_model(args_model: str, info_layers: int, spec: Dict) -> List[int]:
    return [li for li in spec["layers"] if 0 <= li < info_layers]


def collect_site_caches(model, tokenizer, device, prompt: str, units: Dict[str, List[int]], layers: List[int], components: List[str]):
    caches = {}
    missing = []
    for pos_name, pos in units.items():
        if not pos:
            missing.append(pos_name)
            continue
        caches[pos_name] = collect_positions_components(model, tokenizer, device, prompt, pos, layers, components)
    return caches, missing


def build_site_patch(
    target_caches: Dict[str, Dict],
    source_caches: Dict[str, Dict],
    target_units: Dict[str, List[int]],
    source_units: Dict[str, List[int]],
    spec: Dict,
    layers: List[int],
    control: str,
    seed: int,
) -> Tuple[List[Tuple[int, str, List[int], List[torch.Tensor]]], Dict[str, int]]:
    patches = []
    stats = {"position_missing": 0, "position_len_mismatch": 0, "empty_patch": 0}
    for pi, pos_name in enumerate(spec["positions"]):
        target_pos = target_units.get(pos_name, [])
        source_pos = source_units.get(pos_name, [])
        if not target_pos or not source_pos or pos_name not in target_caches or pos_name not in source_caches:
            stats["position_missing"] += 1
            continue
        if len(target_pos) != len(source_pos):
            stats["position_len_mismatch"] += 1
            continue
        for ci, component in enumerate(spec["components"]):
            part = make_multi_patch(
                target_caches[pos_name],
                source_caches[pos_name],
                target_pos,
                layers,
                component,
                control,
                seed + pi * 1009 + ci * 131 + len(control) * 17,
            )
            if not part:
                stats["empty_patch"] += 1
            patches.extend(part)
    if not patches:
        stats["empty_patch"] += 1
    return patches, stats


def summarize(rows: List[Dict]) -> Dict:
    by_key: Dict[Tuple, Dict] = {}
    for row in rows:
        key = (
            row["pair_task"],
            row["eval_task"],
            row["mode"],
            row.get("direction"),
            row.get("site"),
            row.get("control"),
        )
        item = by_key.setdefault(key, {
            "pair_task": row["pair_task"],
            "eval_task": row["eval_task"],
            "mode": row["mode"],
            "kind": row["kind"],
            "direction": row.get("direction"),
            "site": row.get("site"),
            "control": row.get("control"),
            "positions": row.get("positions", []),
            "layers": row.get("layers", []),
            "components": row.get("components", []),
            "n": 0,
            "exact": 0,
            "wrong_exact": 0,
            "tok0_hit": 0,
            "newline_top0": 0,
            "gen_short": 0,
            "gen_yes_no": 0,
            "gen_explanation": 0,
            "gen_full_sentence": 0,
            "sum_rank": 0.0,
            "sum_prefix_minus_newline": 0.0,
            "top0_category": {},
            "generation_text": {},
        })
        item["n"] += 1
        item["exact"] += int(row["eval"]["exact_correct"])
        item["wrong_exact"] += int(row["eval"]["exact_wrong"])
        item["tok0_hit"] += int(row["top0_id"] == row["prefix_id"])
        item["newline_top0"] += int(row["top0_category"] == "newline")
        item["gen_short"] += int(row["text_flags"]["short"])
        item["gen_yes_no"] += int(row["text_flags"]["starts_yes_no"])
        item["gen_explanation"] += int(row["text_flags"]["explanation_signal"])
        item["gen_full_sentence"] += int(row["text_flags"]["full_sentence_like"])
        item["sum_rank"] += row["prefix_rank"]
        item["sum_prefix_minus_newline"] += row["prefix_minus_newline"]
        item["top0_category"].setdefault(row["top0_category"], 0)
        item["top0_category"][row["top0_category"]] += 1
        gen_text = row["generation_text"].replace("\n", "\\n")
        item["generation_text"].setdefault(gen_text, 0)
        item["generation_text"][gen_text] += 1

    out = []
    for item in by_key.values():
        n = max(1, item["n"])
        row = dict(item)
        row["exact_rate"] = item["exact"] / n
        row["wrong_exact_rate"] = item["wrong_exact"] / n
        row["tok0_rate"] = item["tok0_hit"] / n
        row["newline_top0_rate"] = item["newline_top0"] / n
        row["gen_short_rate"] = item["gen_short"] / n
        row["gen_yes_no_rate"] = item["gen_yes_no"] / n
        row["gen_explanation_rate"] = item["gen_explanation"] / n
        row["gen_full_sentence_rate"] = item["gen_full_sentence"] / n
        row["mean_prefix_rank"] = item["sum_rank"] / n
        row["mean_prefix_minus_newline"] = item["sum_prefix_minus_newline"] / n
        row["top0_category"] = dict(sorted(row["top0_category"].items(), key=lambda kv: kv[1], reverse=True))
        row["generation_text"] = dict(sorted(row["generation_text"].items(), key=lambda kv: kv[1], reverse=True)[:10])
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
            row["rank_improvement"] = base["mean_prefix_rank"] - row["mean_prefix_rank"]
            row["delta_exact"] = row["exact"] - base["exact"]
            row["delta_tok0"] = row["tok0_hit"] - base["tok0_hit"]
            row["delta_short"] = row["gen_short"] - base["gen_short"]
            row["baseline_exact"] = base["exact"]
            row["baseline_tok0"] = base["tok0_hit"]
        else:
            row["baseline_rank"] = None
            row["rank_improvement"] = None
            row["delta_exact"] = None
            row["delta_tok0"] = None
            row["delta_short"] = None
            row["baseline_exact"] = None
            row["baseline_tok0"] = None

    restore_rows = [r for r in out if r.get("control") == "restore"]
    controls = [r for r in out if r.get("control") in {"random", "reverse"}]
    absorption = [r for r in restore_rows if r.get("direction") == "value_to_task"]
    suppression = [r for r in restore_rows if r.get("direction") == "task_to_value"]
    absorption.sort(key=lambda r: (-(r["rank_improvement"] or 0), -(r["delta_exact"] or 0), r["mean_prefix_rank"]))
    suppression.sort(key=lambda r: ((r["rank_improvement"] or 0), (r["delta_exact"] or 0), r["mean_prefix_rank"]))
    controls.sort(key=lambda r: (r["pair_task"], r["eval_task"], r.get("site") or "", r.get("control") or ""))
    out.sort(key=lambda r: (
        r["pair_task"],
        r["eval_task"],
        0 if r["kind"] == "baseline" else 1,
        DIRECTIONS.index(r["direction"]) if r.get("direction") in DIRECTIONS else -1,
        r.get("site") or "",
        CONTROLS.index(r["control"]) if r.get("control") in CONTROLS else 999,
    ))
    return {
        "by_mode": out,
        "restore_absorption": absorption,
        "restore_suppression": suppression,
        "controls": controls,
    }


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        site_specs = SITE_SPECS[args.model]
        all_layers = sorted({li for s in site_specs for li in s["layers"] if 0 <= li < info.n_layers})
        all_components = sorted({c for s in site_specs for c in s["components"]})
        all_positions = sorted({p for s in site_specs for p in s["positions"]})
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
            f"sites={[s['name'] for s in site_specs]}, layers={all_layers}, components={all_components}"
        )

        for item_i, item in enumerate(selected):
            case = item["case"]
            correct_ids = answer_ids(tokenizer, case["correct"])
            old_wrong = item["base_top_wrong"] or item["repair_top_wrong"] or item["mode_v_top_wrong"] or values[0]
            old_wrong_ids = answer_ids(tokenizer, old_wrong)
            prefix_id = correct_ids[0]
            old_wrong_prefix_id = old_wrong_ids[0]
            max_new_tokens = args.max_new_tokens
            value_prompt, value_relation, value_intent = make_prompt(
                case, "short_value_allowed", relation_pool, tokenizer, item["sample_idx"]
            )
            value_units_all = position_units(tokenizer, value_prompt, case, value_relation, value_intent)
            value_units = {p: value_units_all.get(p, []) for p in all_positions}
            value_caches, _ = collect_site_caches(
                model, tokenizer, device, value_prompt, value_units, all_layers, all_components
            )

            for task_i, task in enumerate(TASK_ORDER):
                task_prompt, task_relation, task_intent = make_prompt(
                    case, task, relation_pool, tokenizer, item["sample_idx"] + task_i * 17
                )
                task_units_all = position_units(tokenizer, task_prompt, case, task_relation, task_intent)
                task_units = {p: task_units_all.get(p, []) for p in all_positions}
                task_caches, _ = collect_site_caches(
                    model, tokenizer, device, task_prompt, task_units, all_layers, all_components
                )

                specs = {
                    "value_baseline": {
                        "kind": "baseline",
                        "prompt": value_prompt,
                        "patches": [],
                        "eval_task": "short_value_allowed",
                        "direction": None,
                        "site": None,
                        "control": None,
                        "positions": [],
                        "layers": [],
                        "components": [],
                    },
                    "task_baseline": {
                        "kind": "baseline",
                        "prompt": task_prompt,
                        "patches": [],
                        "eval_task": task,
                        "direction": None,
                        "site": None,
                        "control": None,
                        "positions": [],
                        "layers": [],
                        "components": [],
                    },
                }

                for site_i, site in enumerate(site_specs):
                    site_layers = site_layers_for_model(args.model, info.n_layers, site)
                    if not site_layers:
                        continue
                    for direction in DIRECTIONS:
                        for control in CONTROLS:
                            if direction == "value_to_task":
                                patches, stats = build_site_patch(
                                    task_caches,
                                    value_caches,
                                    task_units,
                                    value_units,
                                    site,
                                    site_layers,
                                    control,
                                    item["sample_idx"] * 1009 + task_i * 199 + site_i * 37,
                                )
                                prompt = task_prompt
                                eval_task = task
                            else:
                                patches, stats = build_site_patch(
                                    value_caches,
                                    task_caches,
                                    value_units,
                                    task_units,
                                    site,
                                    site_layers,
                                    control,
                                    item["sample_idx"] * 1009 + task_i * 199 + site_i * 37 + 50021,
                                )
                                prompt = value_prompt
                                eval_task = "short_value_allowed"
                            for k, v in stats.items():
                                filtered[k] += v
                            if not patches:
                                continue
                            mode = f"{direction}_{site['name']}_{control}"
                            specs[mode] = {
                                "kind": "patch",
                                "prompt": prompt,
                                "patches": patches,
                                "eval_task": eval_task,
                                "direction": direction,
                                "site": site["name"],
                                "control": control,
                                "positions": site["positions"],
                                "layers": site_layers,
                                "components": site["components"],
                            }

                for mode, spec in specs.items():
                    probe = final_state_probe(model, tokenizer, device, spec["prompt"], source_patches=spec["patches"])
                    gen = greedy_generate_bridge(
                        model,
                        tokenizer,
                        device,
                        spec["prompt"],
                        max_new_tokens,
                        source_patches=spec["patches"],
                        answer_patches=[],
                        final_patch=None,
                    )
                    ev = generation_eval(gen, correct_ids, old_wrong_ids)
                    row = {
                        "sample_idx": item["sample_idx"],
                        "item_idx": item_i,
                        "pair_task": task,
                        "eval_task": spec["eval_task"],
                        "mode": mode,
                        "kind": spec["kind"],
                        "direction": spec["direction"],
                        "site": spec["site"],
                        "control": spec["control"],
                        "positions": spec["positions"],
                        "layers": spec["layers"],
                        "components": spec["components"],
                        "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                        "task_relation": task_relation,
                        "task_intent": task_intent,
                        "prefix_id": prefix_id,
                        "prefix_text": tokenizer.decode([prefix_id]),
                        "old_wrong_prefix_id": old_wrong_prefix_id,
                        "old_wrong_prefix_text": tokenizer.decode([old_wrong_prefix_id]),
                        "eval": ev,
                        "generation_text": gen["text"],
                        "generation_tokens": gen["tokens"],
                        "text_flags": classify_text(gen["text"]),
                        **ladder_row(tokenizer, probe["logits"], prefix_id, old_wrong_prefix_id, value_prefix_ids, args.top_k),
                    }
                    rows.append(row)
                    if len(examples) < args.example_limit:
                        examples.append(row)

        summary = summarize(rows)
        log("Restore absorption:")
        for row in summary["restore_absorption"][:12]:
            log(
                f"  {row['pair_task']} {row['site']}: n={row['n']} "
                f"rank {row['baseline_rank']:.1f}->{row['mean_prefix_rank']:.1f} "
                f"dR={row['rank_improvement']:.1f} exact {row['baseline_exact']}->{row['exact']} "
                f"tok0 {row['baseline_tok0']}->{row['tok0_hit']}"
            )
        log("Restore suppression:")
        for row in summary["restore_suppression"][:12]:
            log(
                f"  {row['pair_task']} {row['site']}: n={row['n']} "
                f"rank {row['baseline_rank']:.1f}->{row['mean_prefix_rank']:.1f} "
                f"dR={row['rank_improvement']:.1f} exact {row['baseline_exact']}->{row['exact']} "
                f"tok0 {row['baseline_tok0']}->{row['tok0_hit']}"
            )

        return {
            "phase": 653,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "site_specs": site_specs,
            "tasks": TASK_ORDER,
            "controls": CONTROLS,
            "directions": DIRECTIONS,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
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
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--save-rows", action="store_true")
    parser.add_argument("--example-limit", type=int, default=260)
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
        args.max_new_tokens = min(args.max_new_tokens, 4)
        args.example_limit = 120
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 32)
        args.max_samples = max(args.max_samples, 260)
        args.max_cases = max(args.max_cases, 12)
        args.top_k = max(args.top_k, 20)
        args.max_new_tokens = min(max(args.max_new_tokens, 6), 6)
        args.example_limit = max(args.example_limit, 300)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase653_{args.model}_localized_intent_gate_generation_closure_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
