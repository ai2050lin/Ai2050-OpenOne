#!/usr/bin/env python3
"""
Phase 645: Protocol Trajectory Side-Effect and Boundary Atlas.

Phase 643 closed the DS7B separator protocol trajectory on target failures.
This phase asks whether the same L17-L20 middle trajectory is narrowly useful
or whether it causes side effects on already-correct, relation-shifted,
explanation-needed, and non-value prompts.
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
from phase630_distributed_format_route_multisource import collect_positions_components  # noqa: E402
from phase634_multi_position_format_source_field_closure import make_group_patch, merge_patches  # noqa: E402
from phase635_final_readout_projection_bridge_audit import final_state_probe, greedy_generate_bridge  # noqa: E402
from phase636_prefix_competitor_ladder_audit import clean_token, ladder_for_logits  # noqa: E402
from phase637_newline_prior_suppression_source_audit import prompt_common  # noqa: E402
from phase639_protocol_tail_minimal_causal_unit_audit import make_repair_prompt, tail_units  # noqa: E402


OUT_ROOT = Path("results/glm5_phase645_protocol_trajectory_side_effect_boundary_atlas")
COMPONENT = "layer_out"
FULL_LAYERS = [17, 18, 19, 20]
MIDDLE_LAYERS = [18, 19]
MODE_ORDER = [
    "original",
    "inline",
    "to_original_middle_restore",
    "to_original_middle_random",
    "to_original_middle_reverse",
    "remove_from_inline_middle_restore",
    "remove_from_inline_middle_random",
    "remove_from_inline_middle_reverse",
]
SPLIT_ORDER = [
    "target_failure",
    "original_correct",
    "inline_bad",
    "relation_changed",
    "explanation_needed",
    "non_value",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def separator_positions(tokenizer, prompt: str, inline: bool) -> List[int]:
    return tail_units(tokenizer, prompt, inline=inline)["separator"]


def make_interval_patch(
    target_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    source_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    target_pos: List[int],
    layers: List[int],
    control: str,
    seed: int,
) -> List[Tuple[int, str, List[int], List[torch.Tensor]]]:
    patches = []
    for li in layers:
        patches.extend(make_group_patch(
            target_cache,
            source_cache,
            target_pos,
            li,
            COMPONENT,
            control,
            seed + li * 997,
        ))
    return merge_patches(patches)


def relation_alternative(case: Dict, relation_pool: List[str], idx: int) -> str:
    for off in range(1, len(relation_pool) + 1):
        rel = relation_pool[(idx + off) % len(relation_pool)]
        if rel != case["relation"]:
            return rel
    return "is not the same relation as"


def make_prompt_pair(case: Dict, split: str, relation_pool: List[str], idx: int) -> Tuple[str, str]:
    common = prompt_common(case["base_prompt"])
    question = f"{case['category']} {case['relation']}"
    if split in {"target_failure", "original_correct", "inline_bad"}:
        return make_repair_prompt(case, inline=False), make_repair_prompt(case, inline=True)
    if split == "relation_changed":
        alt = relation_alternative(case, relation_pool, idx)
        question = f"{case['category']} {alt}"
        return (
            common + f"Question: {question} ?\nAnswer:",
            common + f"Question: {question} ? Answer:",
        )
    if split == "explanation_needed":
        return (
            common + f"Instruction: Answer with a brief explanation.\nQuestion: {question} ?\nAnswer:",
            common + f"Instruction: Answer with a brief explanation.\nQuestion: {question} ? Answer:",
        )
    if split == "non_value":
        return (
            common + f"Instruction: Answer yes or no, not a category value.\nQuestion: {question} ?\nAnswer:",
            common + f"Instruction: Answer yes or no, not a category value.\nQuestion: {question} ? Answer:",
        )
    raise ValueError(split)


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
    has_newline = "\n" in text
    has_space = text.startswith(" ")
    word_count = len(stripped.replace("\n", " ").split())
    has_explanation_signal = any(s in stripped.lower() for s in ["because", "therefore", "means", " is ", " are "])
    return {
        "has_newline": int(has_newline),
        "has_space_prefix": int(has_space),
        "is_short": int(word_count <= 1),
        "has_explanation_signal": int(has_explanation_signal),
    }


def mode_specs(
    original_prompt: str,
    inline_prompt: str,
    original_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    inline_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    original_pos: List[int],
    inline_pos: List[int],
    layers: List[int],
    seed: int,
) -> Dict:
    specs = {
        "original": {
            "kind": "baseline",
            "prompt": original_prompt,
            "source": [],
            "direction": None,
            "control": None,
            "layers": [],
        },
        "inline": {
            "kind": "baseline",
            "prompt": inline_prompt,
            "source": [],
            "direction": None,
            "control": None,
            "layers": [],
        },
    }
    for control in ["restore", "random", "reverse"]:
        patch = make_interval_patch(
            original_cache,
            inline_cache,
            original_pos,
            layers,
            control,
            seed + len(control) * 37,
        )
        specs[f"to_original_middle_{control}"] = {
            "kind": "trajectory_patch",
            "prompt": original_prompt,
            "source": patch,
            "direction": "to_original",
            "control": control,
            "layers": layers,
        }
        patch = make_interval_patch(
            inline_cache,
            original_cache,
            inline_pos,
            layers,
            control,
            seed + len(control) * 37 + 10007,
        )
        specs[f"remove_from_inline_middle_{control}"] = {
            "kind": "trajectory_patch",
            "prompt": inline_prompt,
            "source": patch,
            "direction": "remove_from_inline",
            "control": control,
            "layers": layers,
        }
    return specs


def summarize(rows: List[Dict]) -> Dict:
    by_key = {}
    for row in rows:
        key = (row["split"], row["mode"])
        item = by_key.setdefault(key, {
            "split": row["split"],
            "prompt_kind": row["prompt_kind"],
            "mode": row["mode"],
            "kind": row["kind"],
            "direction": row.get("direction"),
            "control": row.get("control"),
            "layers": row.get("layers", []),
            "n": 0,
            "tok0_hit": 0,
            "exact": 0,
            "wrong_exact": 0,
            "newline_top0": 0,
            "sum_rank": 0.0,
            "sum_prefix_minus_newline": 0.0,
            "sum_prefix_margin_vs_top": 0.0,
            "gen_newline": 0,
            "gen_short": 0,
            "gen_explanation_signal": 0,
            "top0_category": {},
            "top0_text": {},
            "generation_text": {},
        })
        item["n"] += 1
        item["tok0_hit"] += int(row["top0_id"] == row["prefix_id"])
        item["exact"] += int(row["eval"]["exact_correct"])
        item["wrong_exact"] += int(row["eval"]["exact_wrong"])
        item["newline_top0"] += int(row["top0_category"] == "newline")
        item["sum_rank"] += row["prefix_rank"]
        item["sum_prefix_minus_newline"] += row["prefix_minus_newline"]
        item["sum_prefix_margin_vs_top"] += row["prefix_margin_vs_top"]
        item["gen_newline"] += row["text_flags"]["has_newline"]
        item["gen_short"] += row["text_flags"]["is_short"]
        item["gen_explanation_signal"] += row["text_flags"]["has_explanation_signal"]
        item["top0_category"].setdefault(row["top0_category"], 0)
        item["top0_category"][row["top0_category"]] += 1
        item["top0_text"].setdefault(row["top0_text_clean"], 0)
        item["top0_text"][row["top0_text_clean"]] += 1
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
        row["top0_text"] = dict(sorted(row["top0_text"].items(), key=lambda kv: kv[1], reverse=True)[:8])
        row["generation_text"] = dict(sorted(row["generation_text"].items(), key=lambda kv: kv[1], reverse=True)[:8])
        out.append(row)
    out.sort(key=lambda r: (
        SPLIT_ORDER.index(r["split"]) if r["split"] in SPLIT_ORDER else 999,
        MODE_ORDER.index(r["mode"]) if r["mode"] in MODE_ORDER else 999,
    ))

    by_split = {}
    for row in out:
        by_split.setdefault(row["split"], []).append(row)
    return {"by_split_mode": out, "by_split": by_split}


def select_items(
    model,
    tokenizer,
    device,
    raw_cases: List[Dict],
    values: List[str],
    max_per_split: int,
    relation_pool: List[str],
) -> Tuple[List[Dict], Dict]:
    selected: List[Dict] = []
    counts = {split: 0 for split in SPLIT_ORDER}
    stats = {"target_failure_seen": 0, "original_correct_seen": 0, "inline_bad_seen": 0}

    for si, case0 in enumerate(raw_cases):
        case = dict(case0)
        base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
        repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
        base = winner_stats(base_scores, case["correct"])
        repair = winner_stats(repair_scores, case["correct"])
        target_failure = (not base["correct"]) and repair["correct"]
        original_correct = bool(base["correct"])
        inline_bad = bool(base["correct"] and not repair["correct"])
        stats["target_failure_seen"] += int(target_failure)
        stats["original_correct_seen"] += int(original_correct)
        stats["inline_bad_seen"] += int(inline_bad)

        base_meta = {
            "sample_idx": si,
            "case": case,
            "base_correct": base["correct"],
            "repair_correct": repair["correct"],
            "base_top_wrong": base["top_wrong"],
            "repair_top_wrong": repair["top_wrong"],
        }
        for split, ok in [
            ("target_failure", target_failure),
            ("original_correct", original_correct),
            ("inline_bad", inline_bad),
        ]:
            if ok and counts[split] < max_per_split:
                item = dict(base_meta)
                item["split"] = split
                item["prompt_kind"] = "normal"
                selected.append(item)
                counts[split] += 1

        for split in ["relation_changed", "explanation_needed", "non_value"]:
            if counts[split] < max_per_split:
                item = dict(base_meta)
                item["split"] = split
                item["prompt_kind"] = split
                selected.append(item)
                counts[split] += 1

        if all(counts[s] >= max_per_split for s in SPLIT_ORDER if s != "inline_bad"):
            if counts["inline_bad"] >= min(max_per_split, max(1, stats["inline_bad_seen"])):
                break

    return selected, {"counts": counts, **stats, "relation_pool_size": len(relation_pool)}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = [li for li in MIDDLE_LAYERS if 0 <= li < info.n_layers]
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        relation_pool = sorted({c["relation"] for c in raw_cases})
        value_prefix_ids = {answer_ids(tokenizer, v)[0] for v in values}
        max_new_tokens = max(len(answer_ids(tokenizer, v)) for v in values)
        selected, selection_stats = select_items(
            model, tokenizer, device, raw_cases, values, args.max_per_split, relation_pool
        )
        log(
            f"{args.model}: raw_cases={len(raw_cases)}, selected={len(selected)}, "
            f"layers={layers}, selection={selection_stats['counts']}"
        )

        rows = []
        examples = []
        filtered = {"separator_len_mismatch": 0, "empty_patch": 0}
        for item_i, item in enumerate(selected):
            case = item["case"]
            correct_ids = answer_ids(tokenizer, case["correct"])
            old_wrong = item["base_top_wrong"] or item["repair_top_wrong"] or values[0]
            old_wrong_ids = answer_ids(tokenizer, old_wrong)
            prefix_id = correct_ids[0]
            old_wrong_prefix_id = old_wrong_ids[0]
            original_prompt, inline_prompt = make_prompt_pair(case, item["split"], relation_pool, item["sample_idx"])
            original_pos = separator_positions(tokenizer, original_prompt, inline=False)
            inline_pos = separator_positions(tokenizer, inline_prompt, inline=True)
            if not original_pos or len(original_pos) != len(inline_pos):
                filtered["separator_len_mismatch"] += 1
                continue
            original_cache = collect_positions_components(
                model, tokenizer, device, original_prompt, original_pos, layers, [COMPONENT]
            )
            inline_cache = collect_positions_components(
                model, tokenizer, device, inline_prompt, inline_pos, layers, [COMPONENT]
            )
            specs = mode_specs(
                original_prompt,
                inline_prompt,
                original_cache,
                inline_cache,
                original_pos,
                inline_pos,
                layers,
                item["sample_idx"] * 1009 + item_i * 17,
            )
            for mode in MODE_ORDER:
                spec = specs[mode]
                if spec["kind"] != "baseline" and not spec["source"]:
                    filtered["empty_patch"] += 1
                    continue
                probe = final_state_probe(model, tokenizer, device, spec["prompt"], source_patches=spec["source"])
                gen = greedy_generate_bridge(
                    model,
                    tokenizer,
                    device,
                    spec["prompt"],
                    max_new_tokens,
                    source_patches=spec["source"],
                    answer_patches=[],
                    final_patch=None,
                )
                ev = generation_eval(gen, correct_ids, old_wrong_ids)
                row = {
                    "sample_idx": item["sample_idx"],
                    "item_idx": item_i,
                    "split": item["split"],
                    "prompt_kind": item["prompt_kind"],
                    "mode": mode,
                    "kind": spec["kind"],
                    "direction": spec["direction"],
                    "control": spec["control"],
                    "layers": spec["layers"],
                    "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
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
        for split, split_rows in summary["by_split"].items():
            log(f"{split}:")
            for row in split_rows:
                log(
                    f"  {row['mode']}: n={row['n']} tok0={row['tok0_hit']}/{row['n']} "
                    f"exact={row['exact']}/{row['n']} wrong={row['wrong_exact']}/{row['n']} "
                    f"nl0={row['newline_top0']}/{row['n']} gen_short={row['gen_short']}/{row['n']} "
                    f"rank={row['mean_prefix_rank']:.1f}"
                )

        return {
            "phase": 645,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "top_k": args.top_k,
            "component": COMPONENT,
            "layers": layers,
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
    parser.add_argument("--max-per-split", type=int, default=16)
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
        args.max_per_split = 3
        args.top_k = min(args.top_k, 12)
        args.example_limit = 120
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 40)
        args.max_samples = max(args.max_samples, 320)
        args.max_per_split = max(args.max_per_split, 48)
        args.top_k = max(args.top_k, 20)
        args.example_limit = max(args.example_limit, 240)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase645_{args.model}_protocol_trajectory_side_effect_boundary_atlas_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
