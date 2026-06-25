#!/usr/bin/env python3
"""
Phase 639: Protocol Tail Minimal Causal Unit Audit.

Phase 638 showed that DS7B's inline protocol state can be restored from broad
question-tail groups. This phase shrinks the causal unit to token-level spans:
question mark, separator, Answer, colon, and prompt_last.
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
from phase634_multi_position_format_source_field_closure import COMPONENT, group_layer_defaults, make_group_patch, merge_patches  # noqa: E402
from phase635_final_readout_projection_bridge_audit import final_state_probe, greedy_generate_bridge  # noqa: E402
from phase636_prefix_competitor_ladder_audit import clean_token, ladder_for_logits  # noqa: E402
from phase637_newline_prior_suppression_source_audit import prompt_common  # noqa: E402


OUT_ROOT = Path("results/glm5_phase639_protocol_tail_minimal_causal_unit_audit")
UNIT_ORDER = [
    "qmark",
    "separator",
    "answer_word",
    "colon",
    "prompt_last",
    "qmark_separator",
    "separator_answer",
    "answer_colon",
    "tail_all",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_repair_prompt(case: Dict, inline: bool) -> str:
    common = prompt_common(case["base_prompt"])
    question = f"{case['category']} {case['relation']}"
    if inline:
        return common + f"Question: {question} ? Answer:"
    return common + f"Question: {question} ?\nAnswer:"


def unique_sorted(xs: List[int]) -> List[int]:
    return sorted({int(x) for x in xs if x >= 0})


def tail_units(tokenizer, prompt: str, inline: bool) -> Dict[str, List[int]]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    qmark = token_span(tokenizer, prompt, " ?", "last")
    separator = token_span(tokenizer, prompt, " Answer:" if inline else "\nAnswer:", "last")
    answer_word = token_span(tokenizer, prompt, "Answer", "last")
    colon = token_span(tokenizer, prompt, ":", "last")
    prompt_last = [len(ids) - 1]
    return {
        "qmark": unique_sorted(qmark),
        "separator": unique_sorted(separator),
        "answer_word": unique_sorted(answer_word),
        "colon": unique_sorted(colon),
        "prompt_last": unique_sorted(prompt_last),
        "qmark_separator": unique_sorted(qmark + separator),
        "separator_answer": unique_sorted(separator + answer_word),
        "answer_colon": unique_sorted(answer_word + colon),
        "tail_all": unique_sorted(qmark + separator + answer_word + colon + prompt_last),
    }


def unit_layer(model_name: str, unit: str) -> int:
    layer_map = group_layer_defaults(model_name)
    if unit in {"prompt_last"}:
        return layer_map["prompt_last"]
    if unit in {"qmark", "separator", "qmark_separator", "separator_answer", "answer_word", "colon", "answer_colon"}:
        return layer_map["question_mark_answer"]
    return layer_map["question_all"]


def make_unit_patch(
    model,
    tokenizer,
    device,
    original_prompt: str,
    inline_prompt: str,
    model_name: str,
    unit: str,
    filtered: Dict[str, int],
    seed: int,
) -> List[Tuple[int, str, List[int], List[torch.Tensor]]]:
    original_units = tail_units(tokenizer, original_prompt, inline=False)
    inline_units = tail_units(tokenizer, inline_prompt, inline=True)
    original_pos = original_units.get(unit, [])
    inline_pos = inline_units.get(unit, [])
    if not original_pos or not inline_pos:
        filtered["unit_missing"] += 1
        return []
    if len(original_pos) != len(inline_pos):
        filtered["unit_len_mismatch"] += 1
        return []
    li = unit_layer(model_name, unit)
    original_cache = collect_positions_components(
        model, tokenizer, device, original_prompt, original_pos, [li], [COMPONENT]
    )
    inline_cache = collect_positions_components(
        model, tokenizer, device, inline_prompt, inline_pos, [li], [COMPONENT]
    )
    return merge_patches(make_group_patch(
        original_cache,
        inline_cache,
        original_pos,
        li,
        COMPONENT,
        "restore",
        seed,
    ))


def summarize(rows: List[Dict]) -> Dict:
    by_mode_split = {}
    for row in rows:
        key = (row["mode"], row["split"])
        item = by_mode_split.setdefault(key, {
            "mode": row["mode"],
            "split": row["split"],
            "n": 0,
            "tok0_hit": 0,
            "exact": 0,
            "wrong_exact": 0,
            "newline_top0": 0,
            "space_top0": 0,
            "sum_rank": 0.0,
            "sum_prefix_minus_newline": 0.0,
            "sum_prefix_margin_vs_top": 0.0,
            "top0_category": {},
            "top0_text": {},
        })
        item["n"] += 1
        item["tok0_hit"] += int(row["top0_id"] == row["prefix_id"])
        item["exact"] += int(row["eval"]["exact_correct"])
        item["wrong_exact"] += int(row["eval"]["exact_wrong"])
        item["newline_top0"] += int(row["top0_category"] == "newline")
        item["space_top0"] += int(row["top0_category"] == "space")
        item["sum_rank"] += row["prefix_rank"]
        item["sum_prefix_minus_newline"] += row["prefix_minus_newline"]
        item["sum_prefix_margin_vs_top"] += row["prefix_margin_vs_top"]
        item["top0_category"].setdefault(row["top0_category"], 0)
        item["top0_category"][row["top0_category"]] += 1
        item["top0_text"].setdefault(row["top0_text_clean"], 0)
        item["top0_text"][row["top0_text_clean"]] += 1
    out = []
    for item in by_mode_split.values():
        n = max(1, item["n"])
        row = dict(item)
        row["tok0_rate"] = item["tok0_hit"] / n
        row["exact_rate"] = item["exact"] / n
        row["wrong_exact_rate"] = item["wrong_exact"] / n
        row["newline_top0_rate"] = item["newline_top0"] / n
        row["mean_prefix_rank"] = item["sum_rank"] / n
        row["mean_prefix_minus_newline"] = item["sum_prefix_minus_newline"] / n
        row["mean_prefix_margin_vs_top"] = item["sum_prefix_margin_vs_top"] / n
        row["top0_category"] = dict(sorted(row["top0_category"].items(), key=lambda kv: kv[1], reverse=True))
        row["top0_text"] = dict(sorted(row["top0_text"].items(), key=lambda kv: kv[1], reverse=True)[:8])
        out.append(row)
    out.sort(key=lambda x: (x["split"], x["newline_top0_rate"], -x["tok0_rate"], x["mean_prefix_rank"], x["mode"]))
    return {"by_mode_split": out}


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


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        value_prefix_ids = {answer_ids(tokenizer, v)[0] for v in values}
        max_new_tokens = max(len(answer_ids(tokenizer, v)) for v in values)
        modes = ["original", "inline", "final_output_inline_to_original"] + [f"patch_{u}" for u in UNIT_ORDER]
        filtered = {"not_target": 0, "unit_missing": 0, "unit_len_mismatch": 0, "empty_patch": 0}
        rows = []
        examples = []
        target_seen = 0
        unit_token_lens: Dict[str, Dict[str, Dict[str, int]]] = {}
        log(f"{args.model}: raw_cases={len(raw_cases)}, units={UNIT_ORDER}")

        for si, case0 in enumerate(raw_cases):
            case = dict(case0)
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
            base = winner_stats(base_scores, case["correct"])
            repair = winner_stats(repair_scores, case["correct"])
            target_case = (not base["correct"]) and repair["correct"]
            split = "target" if target_case else "non_target"
            target_seen += int(target_case)
            if args.target_only and not target_case:
                filtered["not_target"] += 1
                continue

            correct_ids = answer_ids(tokenizer, case["correct"])
            old_wrong_ids = answer_ids(tokenizer, base["top_wrong"])
            prefix_id = correct_ids[0]
            old_wrong_prefix_id = old_wrong_ids[0]
            original_prompt = make_repair_prompt(case, inline=False)
            inline_prompt = make_repair_prompt(case, inline=True)
            if si < 8:
                orig_units = tail_units(tokenizer, original_prompt, inline=False)
                inl_units = tail_units(tokenizer, inline_prompt, inline=True)
                for unit in UNIT_ORDER:
                    slot = unit_token_lens.setdefault(unit, {"original": {}, "inline": {}})
                    slot["original"].setdefault(str(len(orig_units.get(unit, []))), 0)
                    slot["inline"].setdefault(str(len(inl_units.get(unit, []))), 0)
                    slot["original"][str(len(orig_units.get(unit, [])))] += 1
                    slot["inline"][str(len(inl_units.get(unit, [])))] += 1

            inline_state = final_state_probe(model, tokenizer, device, inline_prompt)
            specs = {
                "original": {"prompt": original_prompt, "source": [], "final": None},
                "inline": {"prompt": inline_prompt, "source": [], "final": None},
            }
            if inline_state["final_norm_output"] is not None:
                specs["final_output_inline_to_original"] = {
                    "prompt": original_prompt,
                    "source": [],
                    "final": {"kind": "output", "target": inline_state["final_norm_output"]},
                }
            for unit in UNIT_ORDER:
                patch = make_unit_patch(
                    model,
                    tokenizer,
                    device,
                    original_prompt,
                    inline_prompt,
                    args.model,
                    unit,
                    filtered,
                    si * 1009 + len(unit),
                )
                if not patch:
                    filtered["empty_patch"] += 1
                    continue
                specs[f"patch_{unit}"] = {"prompt": original_prompt, "source": patch, "final": None}

            for mode in modes:
                if mode not in specs:
                    continue
                spec = specs[mode]
                probe = final_state_probe(
                    model,
                    tokenizer,
                    device,
                    spec["prompt"],
                    source_patches=spec["source"],
                    final_patch=spec["final"],
                )
                gen = greedy_generate_bridge(
                    model,
                    tokenizer,
                    device,
                    spec["prompt"],
                    max_new_tokens,
                    source_patches=spec["source"],
                    answer_patches=[],
                    final_patch=spec["final"],
                )
                ev = generation_eval(gen, correct_ids, old_wrong_ids)
                row = {
                    "sample_idx": si,
                    "split": split,
                    "mode": mode,
                    "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                    "old_top_wrong": base["top_wrong"],
                    "prefix_id": prefix_id,
                    "prefix_text": tokenizer.decode([prefix_id]),
                    "old_wrong_prefix_id": old_wrong_prefix_id,
                    "old_wrong_prefix_text": tokenizer.decode([old_wrong_prefix_id]),
                    "eval": ev,
                    "generation_text": gen["text"],
                    **ladder_row(tokenizer, probe["logits"], prefix_id, old_wrong_prefix_id, value_prefix_ids, args.top_k),
                }
                rows.append(row)
                if len(examples) < args.example_limit:
                    examples.append(row)

        summary = summarize(rows)
        log("Minimal unit modes:")
        for item in [r for r in summary["by_mode_split"] if r["split"] == "target"][:24]:
            log(
                f"  {item['mode']}: n={item['n']} tok0={item['tok0_hit']}/{item['n']} "
                f"exact={item['exact']}/{item['n']} newline={item['newline_top0']}/{item['n']} "
                f"rank={item['mean_prefix_rank']:.1f} p-newline={item['mean_prefix_minus_newline']:.3f}"
            )
        return {
            "phase": 639,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "top_k": args.top_k,
            "modes": modes,
            "unit_order": UNIT_ORDER,
            "unit_token_lens_sample": unit_token_lens,
            "n_raw_cases": len(raw_cases),
            "n_rows": len({r["sample_idx"] for r in rows}),
            "n_mode_rows": len(rows),
            "n_target_cases_seen": target_seen,
            "target_only": args.target_only,
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
    parser.add_argument("--n-tables", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
    parser.add_argument("--save-rows", action="store_true")
    parser.add_argument("--example-limit", type=int, default=96)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 4
        args.top_k = min(args.top_k, 12)
        args.example_limit = 24
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 32)
        args.max_samples = max(args.max_samples, 256)
        args.top_k = max(args.top_k, 20)
        args.example_limit = max(args.example_limit, 120)
        args.target_only = False
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase639_{args.model}_protocol_tail_minimal_causal_unit_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
