#!/usr/bin/env python3
"""
Phase 650: Protocol Field Template and Side-Effect Audit.

Phase 649 found that answer_label_aligned / answer_colon / separator are strong
protocol-field positions in DS7B. This phase asks whether the same field
generalizes across answer-label templates and whether it causes side effects on
already-correct, relation-changed, explanation-needed, and non-value prompts.
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
from phase639_protocol_tail_minimal_causal_unit_audit import make_repair_prompt  # noqa: E402
from phase647_protocol_writer_graph_audit import COMPONENTS, make_multi_patch  # noqa: E402


OUT_ROOT = Path("results/glm5_phase650_protocol_field_template_side_effect_audit")
TEMPLATE_LABELS = ["Answer", "Response", "Value"]
SPLIT_ORDER = [
    "target_failure",
    "original_correct",
    "relation_changed",
    "explanation_needed",
    "non_value",
]
POSITION_UNITS = ["label_aligned", "label_colon", "separator", "relation_tail"]
INTERVAL_SPECS = [
    ("L17_20", [17, 18, 19, 20], "layer_out"),
    ("L17_20", [17, 18, 19, 20], "attn_out"),
    ("L17_20", [17, 18, 19, 20], "mlp_out"),
]
CONTROLS = ["restore", "random", "reverse"]
MODE_PREFIX = [
    "original",
    "inline",
]


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


def relation_alternative(case: Dict, relation_pool: List[str], idx: int) -> str:
    for off in range(1, len(relation_pool) + 1):
        rel = relation_pool[(idx + off) % len(relation_pool)]
        if rel != case["relation"]:
            return rel
    return "is not the same relation as"


def make_prompt_pair(case: Dict, split: str, relation_pool: List[str], idx: int, label: str) -> Tuple[str, str, str]:
    common = prompt_common(case["base_prompt"])
    question_relation = case["relation"]
    if split in {"target_failure", "original_correct"} and label == "Answer":
        return make_repair_prompt(case, inline=False), make_repair_prompt(case, inline=True), question_relation
    if split == "relation_changed":
        question_relation = relation_alternative(case, relation_pool, idx)

    question = f"{case['category']} {question_relation}"
    prefix = ""
    if split == "explanation_needed":
        prefix = "Instruction: Answer with a brief explanation.\n"
    elif split == "non_value":
        prefix = "Instruction: Answer yes or no, not a category value.\n"
    return (
        common + f"{prefix}Question: {question} ?\n{label}:",
        common + f"{prefix}Question: {question} ? {label}:",
        question_relation,
    )


def position_units(tokenizer, prompt: str, inline: bool, label: str, question_relation: str) -> Dict[str, List[int]]:
    sep_text = f" ? {label}:" if inline else f" ?\n{label}:"
    colon = ordered_unique(token_span(tokenizer, prompt, ":", "last"))
    label_word = ordered_unique([colon[0] - 1] if colon else [])
    label_colon = ordered_unique(label_word + colon)
    return {
        "label_aligned": label_colon,
        "label_colon": label_colon,
        "separator": ordered_unique(token_span(tokenizer, prompt, sep_text, "last")),
        "relation_tail": ordered_unique(token_span(tokenizer, prompt, f"{question_relation}{sep_text}", "last")),
    }


def make_modes(
    original_prompt: str,
    inline_prompt: str,
    original_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    inline_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    original_pos: List[int],
    inline_pos: List[int],
    layers: List[int],
    position_unit: str,
    seed: int,
) -> Dict[str, Dict]:
    specs = {
        "original": {
            "kind": "baseline",
            "prompt": original_prompt,
            "patches": [],
            "position_unit": None,
            "direction": None,
            "interval": None,
            "component": None,
            "control": None,
            "layers": [],
        },
        "inline": {
            "kind": "baseline",
            "prompt": inline_prompt,
            "patches": [],
            "position_unit": None,
            "direction": None,
            "interval": None,
            "component": None,
            "control": None,
            "layers": [],
        },
    }
    for interval, interval_layers0, component in INTERVAL_SPECS:
        interval_layers = [li for li in interval_layers0 if li in layers]
        if not interval_layers:
            continue
        for direction in ["to_original", "remove_from_inline"]:
            target_cache, source_cache, target_pos, prompt = (
                (original_cache, inline_cache, original_pos, original_prompt)
                if direction == "to_original"
                else (inline_cache, original_cache, inline_pos, inline_prompt)
            )
            for control in CONTROLS:
                patches = make_multi_patch(
                    target_cache,
                    source_cache,
                    target_pos,
                    interval_layers,
                    component,
                    control,
                    seed + len(position_unit) * 1009 + len(component) * 97 + len(control) * 31,
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
    word_count = len(stripped.replace("\n", " ").split())
    has_explanation_signal = any(s in stripped.lower() for s in ["because", "therefore", "means", " is ", " are "])
    return {
        "has_newline": int("\n" in text),
        "has_space_prefix": int(text.startswith(" ")),
        "is_short": int(word_count <= 1),
        "has_explanation_signal": int(has_explanation_signal),
    }


def select_items(model, tokenizer, device, raw_cases: List[Dict], values: List[str], max_per_split: int) -> Tuple[List[Dict], Dict]:
    selected: List[Dict] = []
    counts = {split: 0 for split in SPLIT_ORDER}
    stats = {"target_failure_seen": 0, "original_correct_seen": 0}

    for si, case0 in enumerate(raw_cases):
        case = dict(case0)
        base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
        repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
        base = winner_stats(base_scores, case["correct"])
        repair = winner_stats(repair_scores, case["correct"])
        target_failure = (not base["correct"]) and repair["correct"]
        original_correct = bool(base["correct"])
        stats["target_failure_seen"] += int(target_failure)
        stats["original_correct_seen"] += int(original_correct)

        meta = {
            "sample_idx": si,
            "case": case,
            "base_correct": base["correct"],
            "repair_correct": repair["correct"],
            "base_top_wrong": base["top_wrong"],
            "repair_top_wrong": repair["top_wrong"],
        }
        if target_failure and counts["target_failure"] < max_per_split:
            item = dict(meta)
            item["split"] = "target_failure"
            selected.append(item)
            counts["target_failure"] += 1
        if original_correct and counts["original_correct"] < max_per_split:
            item = dict(meta)
            item["split"] = "original_correct"
            selected.append(item)
            counts["original_correct"] += 1

        for split in ["relation_changed", "explanation_needed", "non_value"]:
            if counts[split] < max_per_split:
                item = dict(meta)
                item["split"] = split
                selected.append(item)
                counts[split] += 1

        if all(counts[s] >= max_per_split for s in SPLIT_ORDER):
            break

    return selected, {**stats, "counts": counts}


def summarize(rows: List[Dict]) -> Dict:
    by_key: Dict[Tuple, Dict] = {}
    for row in rows:
        key = (
            row["split"],
            row["template_label"],
            row["mode"],
            row.get("position_unit"),
            row.get("direction"),
            row.get("interval"),
            row.get("component"),
            row.get("control"),
        )
        item = by_key.setdefault(key, {
            "split": row["split"],
            "template_label": row["template_label"],
            "mode": row["mode"],
            "kind": row["kind"],
            "position_unit": row.get("position_unit"),
            "direction": row.get("direction"),
            "interval": row.get("interval"),
            "component": row.get("component"),
            "control": row.get("control"),
            "layers": row.get("layers", []),
            "n": 0,
            "tok0_hit": 0,
            "exact": 0,
            "wrong_exact": 0,
            "newline_top0": 0,
            "gen_newline": 0,
            "gen_short": 0,
            "gen_explanation_signal": 0,
            "sum_rank": 0.0,
            "sum_prefix_minus_newline": 0.0,
            "sum_prefix_margin_vs_top": 0.0,
            "top0_category": {},
            "generation_text": {},
        })
        item["n"] += 1
        item["tok0_hit"] += int(row["top0_id"] == row["prefix_id"])
        item["exact"] += int(row["eval"]["exact_correct"])
        item["wrong_exact"] += int(row["eval"]["exact_wrong"])
        item["newline_top0"] += int(row["top0_category"] == "newline")
        item["gen_newline"] += row["text_flags"]["has_newline"]
        item["gen_short"] += row["text_flags"]["is_short"]
        item["gen_explanation_signal"] += row["text_flags"]["has_explanation_signal"]
        item["sum_rank"] += row["prefix_rank"]
        item["sum_prefix_minus_newline"] += row["prefix_minus_newline"]
        item["sum_prefix_margin_vs_top"] += row["prefix_margin_vs_top"]
        item["top0_category"].setdefault(row["top0_category"], 0)
        item["top0_category"][row["top0_category"]] += 1
        gen_text = row["generation_text"].replace("\n", "\\n")
        item["generation_text"].setdefault(gen_text, 0)
        item["generation_text"][gen_text] += 1

    out = []
    for item in by_key.values():
        n = max(1, item["n"])
        row = dict(item)
        row["tok0_rate"] = item["tok0_hit"] / n
        row["exact_rate"] = item["exact"] / n
        row["wrong_exact_rate"] = item["wrong_exact"] / n
        row["newline_top0_rate"] = item["newline_top0"] / n
        row["gen_newline_rate"] = item["gen_newline"] / n
        row["gen_short_rate"] = item["gen_short"] / n
        row["gen_explanation_signal_rate"] = item["gen_explanation_signal"] / n
        row["mean_prefix_rank"] = item["sum_rank"] / n
        row["mean_prefix_minus_newline"] = item["sum_prefix_minus_newline"] / n
        row["mean_prefix_margin_vs_top"] = item["sum_prefix_margin_vs_top"] / n
        row["top0_category"] = dict(sorted(row["top0_category"].items(), key=lambda kv: kv[1], reverse=True))
        row["generation_text"] = dict(sorted(row["generation_text"].items(), key=lambda kv: kv[1], reverse=True)[:8])
        out.append(row)

    def sort_key(r):
        return (
            SPLIT_ORDER.index(r["split"]) if r["split"] in SPLIT_ORDER else 999,
            TEMPLATE_LABELS.index(r["template_label"]) if r["template_label"] in TEMPLATE_LABELS else 999,
            0 if r["kind"] == "baseline" else 1,
            POSITION_UNITS.index(r["position_unit"]) if r.get("position_unit") in POSITION_UNITS else -1,
            r.get("direction") or "",
            r.get("component") or "",
            CONTROLS.index(r.get("control")) if r.get("control") in CONTROLS else 999,
        )

    out.sort(key=sort_key)
    by_split_template: Dict[str, List[Dict]] = {}
    by_position: Dict[str, List[Dict]] = {}
    for row in out:
        by_split_template.setdefault(f"{row['split']}::{row['template_label']}", []).append(row)
        if row.get("position_unit"):
            by_position.setdefault(row["position_unit"], []).append(row)

    restore = [r for r in out if r.get("control") == "restore"]
    target_suff = [
        r for r in restore
        if r["split"] == "target_failure" and r.get("direction") == "to_original"
    ]
    side_effect = [
        r for r in restore
        if r["split"] in {"original_correct", "relation_changed", "explanation_needed", "non_value"}
    ]
    target_suff.sort(key=lambda r: (-r["exact_rate"], r["newline_top0_rate"], r["mean_prefix_rank"]))
    side_effect.sort(key=lambda r: (-r["exact_rate"], -r["gen_short_rate"], r["mean_prefix_rank"]))
    return {
        "by_mode": out,
        "by_split_template": by_split_template,
        "by_position": by_position,
        "best_target_sufficiency": target_suff[:80],
        "largest_side_effect_old_value": side_effect[:120],
    }


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = [li for li in [17, 18, 19, 20] if 0 <= li < info.n_layers]
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        relation_pool = sorted({c["relation"] for c in raw_cases})
        selected, selection_stats = select_items(model, tokenizer, device, raw_cases, values, args.max_per_split)
        value_prefix_ids = {answer_ids(tokenizer, v)[0] for v in values}
        max_new_tokens = max(len(answer_ids(tokenizer, v)) for v in values)
        rows = []
        examples = []
        filtered = {"position_missing": 0, "position_len_mismatch": 0, "empty_patch": 0}
        log(
            f"{args.model}: raw_cases={len(raw_cases)}, selected={len(selected)}, "
            f"templates={TEMPLATE_LABELS}, layers={layers}, selection={selection_stats['counts']}"
        )

        for item_i, item in enumerate(selected):
            case = item["case"]
            correct_ids = answer_ids(tokenizer, case["correct"])
            old_wrong = item["base_top_wrong"] or item["repair_top_wrong"] or values[0]
            old_wrong_ids = answer_ids(tokenizer, old_wrong)
            prefix_id = correct_ids[0]
            old_wrong_prefix_id = old_wrong_ids[0]

            for label_i, label in enumerate(TEMPLATE_LABELS):
                original_prompt, inline_prompt, question_relation = make_prompt_pair(
                    case, item["split"], relation_pool, item["sample_idx"], label
                )
                original_units = position_units(tokenizer, original_prompt, False, label, question_relation)
                inline_units = position_units(tokenizer, inline_prompt, True, label, question_relation)

                baseline_specs = {
                    "original": {
                        "kind": "baseline",
                        "prompt": original_prompt,
                        "patches": [],
                        "position_unit": None,
                        "direction": None,
                        "interval": None,
                        "component": None,
                        "control": None,
                        "layers": [],
                    },
                    "inline": {
                        "kind": "baseline",
                        "prompt": inline_prompt,
                        "patches": [],
                        "position_unit": None,
                        "direction": None,
                        "interval": None,
                        "component": None,
                        "control": None,
                        "layers": [],
                    },
                }
                specs = dict(baseline_specs)
                for pos_unit in POSITION_UNITS:
                    original_pos = original_units.get(pos_unit, [])
                    inline_pos = inline_units.get(pos_unit, [])
                    if not original_pos or not inline_pos:
                        filtered["position_missing"] += 1
                        continue
                    if len(original_pos) != len(inline_pos):
                        filtered["position_len_mismatch"] += 1
                        continue
                    original_cache = collect_positions_components(
                        model, tokenizer, device, original_prompt, original_pos, layers, COMPONENTS
                    )
                    inline_cache = collect_positions_components(
                        model, tokenizer, device, inline_prompt, inline_pos, layers, COMPONENTS
                    )
                    specs.update(make_modes(
                        original_prompt,
                        inline_prompt,
                        original_cache,
                        inline_cache,
                        original_pos,
                        inline_pos,
                        layers,
                        pos_unit,
                        item["sample_idx"] * 1009 + item_i * 101 + label_i * 17,
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
                        max_new_tokens,
                        source_patches=spec["patches"],
                        answer_patches=[],
                        final_patch=None,
                    )
                    ev = generation_eval(gen, correct_ids, old_wrong_ids)
                    row = {
                        "sample_idx": item["sample_idx"],
                        "item_idx": item_i,
                        "split": item["split"],
                        "template_label": label,
                        "mode": mode,
                        "kind": spec["kind"],
                        "position_unit": spec["position_unit"],
                        "direction": spec["direction"],
                        "interval": spec["interval"],
                        "component": spec["component"],
                        "control": spec["control"],
                        "layers": spec["layers"],
                        "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                        "question_relation": question_relation,
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
        log("Top target sufficiency rows:")
        for row in summary["best_target_sufficiency"][:20]:
            log(
                f"  {row['split']}/{row['template_label']} {row['mode']}: n={row['n']} "
                f"exact={row['exact']}/{row['n']} nl={row['newline_top0']}/{row['n']} "
                f"rank={row['mean_prefix_rank']:.1f}"
            )
        log("Largest old-value side-effect rows:")
        for row in summary["largest_side_effect_old_value"][:20]:
            log(
                f"  {row['split']}/{row['template_label']} {row['mode']}: n={row['n']} "
                f"exact={row['exact']}/{row['n']} short={row['gen_short']}/{row['n']} "
                f"nl={row['newline_top0']}/{row['n']}"
            )

        return {
            "phase": 650,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "top_k": args.top_k,
            "layers": layers,
            "components": COMPONENTS,
            "template_labels": TEMPLATE_LABELS,
            "position_units": POSITION_UNITS,
            "interval_specs": [
                {"interval": name, "layers": [li for li in lis if li in layers], "component": comp}
                for name, lis, comp in INTERVAL_SPECS
            ],
            "controls": CONTROLS,
            "max_new_tokens": max_new_tokens,
            "n_raw_cases": len(raw_cases),
            "n_selected_items": len(selected),
            "n_mode_rows": len(rows),
            "max_per_split": args.max_per_split,
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
    parser.add_argument("--max-per-split", type=int, default=6)
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
        args.max_per_split = 1
        args.top_k = min(args.top_k, 12)
        args.example_limit = 160
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 40)
        args.max_samples = max(args.max_samples, 320)
        args.max_per_split = max(args.max_per_split, 8)
        args.top_k = max(args.top_k, 20)
        args.example_limit = max(args.example_limit, 320)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase650_{args.model}_protocol_field_template_side_effect_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
