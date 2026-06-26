#!/usr/bin/env python3
"""
Phase 651: Task Intent Gate and Protocol Field Boundary Audit.

Phase 650 showed that the protocol field can force short value answers even on
explanation / non-value prompts. This phase adds a compact explicit mode signal
and tests whether task intent positions can open or suppress the short-answer
protocol field.
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
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase628_prefix_format_semantic_integration import generation_eval  # noqa: E402
from phase630_distributed_format_route_multisource import collect_positions_components, token_span  # noqa: E402
from phase635_final_readout_projection_bridge_audit import final_state_probe, greedy_generate_bridge  # noqa: E402
from phase636_prefix_competitor_ladder_audit import clean_token, ladder_for_logits  # noqa: E402
from phase637_newline_prior_suppression_source_audit import prompt_common  # noqa: E402
from phase647_protocol_writer_graph_audit import COMPONENTS, make_multi_patch  # noqa: E402


OUT_ROOT = Path("results/glm5_phase651_task_intent_gate_protocol_boundary_audit")
TASKS = {
    "explanation_required": "reason",
    "yes_no_required": "yesno",
    "full_sentence_required": "sentence",
    "relation_changed": "value",
}
POSITION_UNITS = [
    "intent_word",
    "instruction_span",
    "instruction_prefix",
    "question_span",
    "relation_text",
    "label_aligned",
    "separator",
    "relation_tail",
]
INTERVAL_SPECS = [
    ("L14_22", list(range(14, 23)), "layer_out"),
    ("L17_20", [17, 18, 19, 20], "layer_out"),
    ("L17_20", [17, 18, 19, 20], "attn_out"),
    ("L17_20", [17, 18, 19, 20], "mlp_out"),
]
CONTROLS = ["restore", "random", "reverse"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ordered_unique(xs: List[int]) -> List[int]:
    out = []
    seen = set()
    for x in xs:
        x = int(x)
        if x >= 0 and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def relation_alternative(tokenizer, relation_pool: List[str], relation: str, idx: int) -> str:
    rel_len = len(tokenizer.encode(relation, add_special_tokens=False))
    fallback = None
    for off in range(1, len(relation_pool) + 1):
        alt = relation_pool[(idx + off) % len(relation_pool)]
        if alt == relation:
            continue
        fallback = fallback or alt
        if len(tokenizer.encode(alt, add_special_tokens=False)) == rel_len:
            return alt
    return fallback or "is related to"


def make_prompt(case: Dict, task: str, relation_pool: List[str], tokenizer, idx: int) -> Tuple[str, str, str]:
    common = prompt_common(case["base_prompt"])
    intent_word = "value" if task == "short_value_allowed" else TASKS[task]
    relation = case["relation"]
    if task == "relation_changed":
        relation = relation_alternative(tokenizer, relation_pool, relation, idx)
    prompt = (
        common
        + f"Instruction: Answer with {intent_word}.\n"
        + f"Question: {case['category']} {relation} ?\nAnswer:"
    )
    return prompt, relation, intent_word


def position_units(tokenizer, prompt: str, case: Dict, relation: str, intent_word: str) -> Dict[str, List[int]]:
    colon = ordered_unique(token_span(tokenizer, prompt, ":", "last"))
    label_word = ordered_unique([colon[0] - 1] if colon else [])
    label_aligned = ordered_unique(label_word + colon)
    question_text = f"Question: {case['category']} {relation} ?"
    instruction_text = f"Instruction: Answer with {intent_word}."
    sep_text = " ?\nAnswer:"
    return {
        "intent_word": ordered_unique(token_span(tokenizer, prompt, intent_word, "last")),
        "instruction_span": ordered_unique(token_span(tokenizer, prompt, instruction_text, "last")),
        "instruction_prefix": ordered_unique(token_span(tokenizer, prompt, "Instruction:", "last")),
        "question_span": ordered_unique(token_span(tokenizer, prompt, question_text, "last")),
        "relation_text": ordered_unique(token_span(tokenizer, prompt, relation, "last")),
        "label_aligned": label_aligned,
        "separator": ordered_unique(token_span(tokenizer, prompt, sep_text, "last")),
        "relation_tail": ordered_unique(token_span(tokenizer, prompt, f"{relation}{sep_text}", "last")),
    }


def make_pair_modes(
    value_prompt: str,
    task_prompt: str,
    value_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    task_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    value_pos: List[int],
    task_pos: List[int],
    layers: List[int],
    position_unit: str,
    seed: int,
) -> Dict[str, Dict]:
    specs = {}
    for interval, interval_layers0, component in INTERVAL_SPECS:
        interval_layers = [li for li in interval_layers0 if li in layers]
        if not interval_layers:
            continue
        for direction in ["value_to_task", "task_to_value"]:
            target_cache, source_cache, target_pos, prompt = (
                (task_cache, value_cache, task_pos, task_prompt)
                if direction == "value_to_task"
                else (value_cache, task_cache, value_pos, value_prompt)
            )
            for control in CONTROLS:
                patches = make_multi_patch(
                    target_cache,
                    source_cache,
                    target_pos,
                    interval_layers,
                    component,
                    control,
                    seed + len(position_unit) * 1009 + len(interval) * 101 + len(component) * 29 + len(control) * 7,
                )
                name = f"{position_unit}_{direction}_{interval}_{component}_{control}"
                specs[name] = {
                    "kind": "patch",
                    "prompt": prompt,
                    "patches": patches,
                    "position_unit": position_unit,
                    "direction": direction,
                    "interval": interval,
                    "component": component,
                    "control": control,
                    "layers": interval_layers,
                    "eval_prompt_task": task_prompt if direction == "value_to_task" else value_prompt,
                }
    return specs


def ladder_row(tokenizer, logits, prefix_id, old_wrong_prefix_id, value_prefix_ids, top_k) -> Dict:
    ladder = ladder_for_logits(tokenizer, logits, prefix_id, old_wrong_prefix_id, value_prefix_ids, top_k)
    newline_group = ladder["groups"].get("newline")
    prefix_minus_newline = newline_group["prefix_minus_group_max"] if newline_group else 99.0
    return {
        "prefix_rank": ladder["prefix_rank"],
        "top0_id": ladder["top0_id"],
        "top0_text": ladder["top0_text"],
        "top0_text_clean": clean_token(ladder["top0_text"]),
        "top0_category": ladder["top0_category"],
        "prefix_logit": ladder["prefix_logit"],
        "prefix_margin_vs_top": ladder["prefix_logit"] - float(logits[ladder["top0_id"]].item()),
        "prefix_minus_newline": prefix_minus_newline,
        "top": ladder["top"][:8],
        "groups": ladder["groups"],
    }


def classify_generation_text(text: str) -> Dict[str, int]:
    stripped = text.strip()
    lower = stripped.lower()
    word_count = len(stripped.replace("\n", " ").split())
    return {
        "has_newline": int("\n" in text),
        "is_short": int(word_count <= 1),
        "has_explanation_signal": int(any(s in lower for s in ["because", "therefore", "means", " is ", " are "])),
        "starts_yes_no": int(lower.startswith("yes") or lower.startswith("no")),
        "full_sentence_like": int(word_count >= 4),
    }


def select_cases(
    model,
    tokenizer,
    device,
    raw_cases: List[Dict],
    values: List[str],
    max_cases: int,
    relation_pool: List[str],
) -> Tuple[List[Dict], Dict]:
    selected = []
    fallback = []
    stats = {
        "mode_v_correct_seen": 0,
        "repair_correct_seen": 0,
        "target_failure_seen": 0,
        "fallback_used": 0,
        "scanned": 0,
    }
    for si, case0 in enumerate(raw_cases):
        case = dict(case0)
        stats["scanned"] += 1
        base = winner_stats(score_map(model, tokenizer, device, case["base_prompt"], values), case["correct"])
        repair = winner_stats(score_map(model, tokenizer, device, case["repair_prompt"], values), case["correct"])
        value_prompt, _, _ = make_prompt(case, "short_value_allowed", relation_pool, tokenizer, si)
        mode_v = winner_stats(score_map(model, tokenizer, device, value_prompt, values), case["correct"])
        if repair["correct"]:
            stats["repair_correct_seen"] += 1
            stats["target_failure_seen"] += int(not base["correct"])
            item = {
                "sample_idx": si,
                "case": case,
                "base_correct": base["correct"],
                "repair_correct": repair["correct"],
                "mode_v_correct": mode_v["correct"],
                "base_top_wrong": base["top_wrong"],
                "repair_top_wrong": repair["top_wrong"],
                "mode_v_top_wrong": mode_v["top_wrong"],
            }
            if mode_v["correct"]:
                stats["mode_v_correct_seen"] += 1
                selected.append(item)
            else:
                fallback.append(item)
        if len(selected) >= max_cases:
            break
    if len(selected) < max_cases:
        need = max_cases - len(selected)
        selected.extend(fallback[:need])
        stats["fallback_used"] = min(need, len(fallback))
    return selected, stats


def summarize(rows: List[Dict]) -> Dict:
    by_key: Dict[Tuple, Dict] = {}
    for row in rows:
        key = (
            row["pair_task"],
            row["eval_task"],
            row["mode"],
            row.get("position_unit"),
            row.get("direction"),
            row.get("interval"),
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
            "interval": row.get("interval"),
            "component": row.get("component"),
            "control": row.get("control"),
            "layers": row.get("layers", []),
            "n": 0,
            "exact": 0,
            "wrong_exact": 0,
            "tok0_hit": 0,
            "newline_top0": 0,
            "gen_short": 0,
            "gen_explanation": 0,
            "gen_yes_no": 0,
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
        item["gen_short"] += row["text_flags"]["is_short"]
        item["gen_explanation"] += row["text_flags"]["has_explanation_signal"]
        item["gen_yes_no"] += row["text_flags"]["starts_yes_no"]
        item["gen_full_sentence"] += row["text_flags"]["full_sentence_like"]
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
        row["tok0_rate"] = item["tok0_hit"] / n
        row["newline_top0_rate"] = item["newline_top0"] / n
        row["gen_short_rate"] = item["gen_short"] / n
        row["gen_explanation_rate"] = item["gen_explanation"] / n
        row["gen_yes_no_rate"] = item["gen_yes_no"] / n
        row["gen_full_sentence_rate"] = item["gen_full_sentence"] / n
        row["mean_prefix_rank"] = item["sum_rank"] / n
        row["mean_prefix_minus_newline"] = item["sum_prefix_minus_newline"] / n
        row["top0_category"] = dict(sorted(row["top0_category"].items(), key=lambda kv: kv[1], reverse=True))
        row["generation_text"] = dict(sorted(row["generation_text"].items(), key=lambda kv: kv[1], reverse=True)[:8])
        out.append(row)

    baseline = {
        (r["pair_task"], r["eval_task"]): r
        for r in out
        if r["kind"] == "baseline"
    }
    for row in out:
        base = baseline.get((row["pair_task"], row["eval_task"]))
        if base:
            row["delta_exact_vs_eval_baseline"] = row["exact"] - base["exact"]
            row["delta_short_vs_eval_baseline"] = row["gen_short"] - base["gen_short"]
            row["delta_rank_improvement_vs_eval_baseline"] = base["mean_prefix_rank"] - row["mean_prefix_rank"]
            row["baseline_exact"] = base["exact"]
            row["baseline_short"] = base["gen_short"]
            row["baseline_rank"] = base["mean_prefix_rank"]
        else:
            row["delta_exact_vs_eval_baseline"] = None
            row["delta_short_vs_eval_baseline"] = None
            row["delta_rank_improvement_vs_eval_baseline"] = None
            row["baseline_exact"] = None
            row["baseline_short"] = None
            row["baseline_rank"] = None

    absorption = [
        r for r in out
        if r.get("control") == "restore"
        and r.get("direction") == "value_to_task"
        and r["eval_task"] != "short_value_allowed"
    ]
    suppression = [
        r for r in out
        if r.get("control") == "restore"
        and r.get("direction") == "task_to_value"
        and r["eval_task"] == "short_value_allowed"
    ]
    absorption.sort(key=lambda r: (
        -(r["delta_exact_vs_eval_baseline"] or 0),
        -(r["delta_rank_improvement_vs_eval_baseline"] or 0),
        -r["exact"],
        r["mean_prefix_rank"],
    ))
    suppression.sort(key=lambda r: (
        (r["delta_exact_vs_eval_baseline"] or 0),
        (r["delta_rank_improvement_vs_eval_baseline"] or 0),
        r["exact"],
        r["mean_prefix_rank"],
    ))
    out.sort(key=lambda r: (
        r["pair_task"],
        r["eval_task"],
        0 if r["kind"] == "baseline" else 1,
        POSITION_UNITS.index(r["position_unit"]) if r.get("position_unit") in POSITION_UNITS else -1,
        r.get("direction") or "",
        r.get("component") or "",
        CONTROLS.index(r.get("control")) if r.get("control") in CONTROLS else 999,
    ))
    return {
        "by_mode": out,
        "strongest_absorption": absorption[:120],
        "strongest_suppression": suppression[:120],
    }


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = [li for li in range(14, 23) if 0 <= li < info.n_layers]
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
            f"tasks={list(TASKS)}, layers={layers}"
        )

        for item_i, item in enumerate(selected):
            case = item["case"]
            correct_ids = answer_ids(tokenizer, case["correct"])
            old_wrong = item["base_top_wrong"] or item["repair_top_wrong"] or values[0]
            old_wrong_ids = answer_ids(tokenizer, old_wrong)
            prefix_id = correct_ids[0]
            old_wrong_prefix_id = old_wrong_ids[0]
            value_prompt, value_relation, value_intent = make_prompt(
                case, "short_value_allowed", relation_pool, tokenizer, item["sample_idx"]
            )
            value_units = position_units(tokenizer, value_prompt, case, value_relation, value_intent)

            for task_i, task in enumerate(TASKS):
                task_prompt, task_relation, task_intent = make_prompt(
                    case, task, relation_pool, tokenizer, item["sample_idx"] + task_i * 17
                )
                task_units = position_units(tokenizer, task_prompt, case, task_relation, task_intent)
                specs = {
                    "value_baseline": {
                        "kind": "baseline",
                        "prompt": value_prompt,
                        "patches": [],
                        "position_unit": None,
                        "direction": None,
                        "interval": None,
                        "component": None,
                        "control": None,
                        "layers": [],
                        "eval_task": "short_value_allowed",
                    },
                    "task_baseline": {
                        "kind": "baseline",
                        "prompt": task_prompt,
                        "patches": [],
                        "position_unit": None,
                        "direction": None,
                        "interval": None,
                        "component": None,
                        "control": None,
                        "layers": [],
                        "eval_task": task,
                    },
                }

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
                    specs.update(make_pair_modes(
                        value_prompt,
                        task_prompt,
                        value_cache,
                        task_cache,
                        value_pos,
                        task_pos,
                        layers,
                        pos_unit,
                        item["sample_idx"] * 1009 + item_i * 101 + task_i * 17,
                    ))

                for mode, spec in specs.items():
                    if spec["kind"] != "baseline" and not spec["patches"]:
                        filtered["empty_patch"] += 1
                        continue
                    probe = final_state_probe(model, tokenizer, device, spec["prompt"], source_patches=spec["patches"])
                    gen = greedy_generate_bridge(
                        model,
                        tokenizer,
                        device,
                        spec["prompt"],
                        args.max_new_tokens,
                        source_patches=spec["patches"],
                        answer_patches=[],
                        final_patch=None,
                    )
                    ev = generation_eval(gen, correct_ids, old_wrong_ids)
                    row = {
                        "sample_idx": item["sample_idx"],
                        "item_idx": item_i,
                        "pair_task": task,
                        "eval_task": spec.get("eval_task") or (task if spec["direction"] == "value_to_task" else "short_value_allowed"),
                        "mode": mode,
                        "kind": spec["kind"],
                        "position_unit": spec["position_unit"],
                        "direction": spec["direction"],
                        "interval": spec["interval"],
                        "component": spec["component"],
                        "control": spec["control"],
                        "layers": spec["layers"],
                        "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                        "task_relation": task_relation,
                        "task_intent": task_intent,
                        "base_correct": item["base_correct"],
                        "repair_correct": item["repair_correct"],
                        "old_top_wrong": old_wrong,
                        "prefix_id": prefix_id,
                        "prefix_text": tokenizer.decode([prefix_id]),
                        "old_wrong_prefix_id": old_wrong_prefix_id,
                        "old_wrong_prefix_text": tokenizer.decode([old_wrong_prefix_id]),
                        "eval": ev,
                        "generation_text": gen["text"],
                        "generation_tokens": gen["tokens"],
                        "text_flags": classify_generation_text(gen["text"]),
                        **ladder_row(tokenizer, probe["logits"], prefix_id, old_wrong_prefix_id, value_prefix_ids, args.top_k),
                    }
                    rows.append(row)
                    if len(examples) < args.example_limit:
                        examples.append(row)

        summary = summarize(rows)
        log("Strongest value-to-task absorption:")
        for row in summary["strongest_absorption"][:18]:
            log(
                f"  {row['pair_task']} {row['mode']}: exact {row['baseline_exact']}->{row['exact']} "
                f"delta={row['delta_exact_vs_eval_baseline']} short={row['gen_short']}/{row['n']}"
            )
        log("Strongest task-to-value suppression:")
        for row in summary["strongest_suppression"][:18]:
            log(
                f"  {row['pair_task']} {row['mode']}: exact {row['baseline_exact']}->{row['exact']} "
                f"delta={row['delta_exact_vs_eval_baseline']} short={row['gen_short']}/{row['n']}"
            )

        return {
            "phase": 651,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "layers": layers,
            "components": COMPONENTS,
            "tasks": TASKS,
            "position_units": POSITION_UNITS,
            "interval_specs": [
                {"interval": name, "layers": [li for li in lis if li in layers], "component": comp}
                for name, lis, comp in INTERVAL_SPECS
            ],
            "controls": CONTROLS,
            "max_new_tokens": args.max_new_tokens,
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
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=12)
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
        args.max_new_tokens = min(args.max_new_tokens, 8)
        args.top_k = min(args.top_k, 12)
        args.example_limit = 120
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 40)
        args.max_samples = max(args.max_samples, 320)
        args.max_cases = max(args.max_cases, 12)
        args.max_new_tokens = max(args.max_new_tokens, 12)
        args.top_k = max(args.top_k, 20)
        args.example_limit = max(args.example_limit, 320)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase651_{args.model}_task_intent_gate_protocol_boundary_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
