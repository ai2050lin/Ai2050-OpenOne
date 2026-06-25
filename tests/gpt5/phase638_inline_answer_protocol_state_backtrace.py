#!/usr/bin/env python3
"""
Phase 638: Inline Answer Protocol State Backtrace.

Phase 637 showed that changing only the answer layout from:
    "?\\nAnswer:"
to:
    "? Answer:"
strongly suppresses DS7B's newline prior. This phase asks where that protocol
state is carried by restoring inline-prompt states into the original prompt.
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
from phase628_prefix_format_semantic_integration import generation_eval, token_strings  # noqa: E402
from phase630_distributed_format_route_multisource import collect_positions_components, token_span  # noqa: E402
from phase631_token0_prefix_readout_competition import token0_logits  # noqa: E402
from phase634_multi_position_format_source_field_closure import (  # noqa: E402
    COMPONENT,
    group_layer_defaults,
    make_group_patch,
    merge_patches,
)
from phase635_final_readout_projection_bridge_audit import final_state_probe, greedy_generate_bridge  # noqa: E402
from phase636_prefix_competitor_ladder_audit import clean_token, ladder_for_logits  # noqa: E402
from phase637_newline_prior_suppression_source_audit import prompt_common  # noqa: E402


OUT_ROOT = Path("results/glm5_phase638_inline_answer_protocol_state_backtrace")
GROUPS = ["prompt_last", "answer_label", "question_mark_answer", "relation_tail", "question_all"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_repair_prompt(case: Dict, inline: bool) -> str:
    common = prompt_common(case["base_prompt"])
    question = f"{case['category']} {case['relation']}"
    if inline:
        return common + f"Question: {question} ? Answer:"
    return common + f"Question: {question} ?\nAnswer:"


def protocol_groups(tokenizer, prompt: str, case: Dict, inline: bool) -> Dict[str, List[int]]:
    subject = case["category"]
    rel = case["relation"]
    tail = " ? Answer:" if inline else " ?\nAnswer:"
    question = f"Question: {subject} {rel}{tail}"
    groups = {
        "prompt_last": [len(tokenizer.encode(prompt, add_special_tokens=False)) - 1],
        "answer_label": token_span(tokenizer, prompt, "Answer:", "last"),
        "question_mark_answer": token_span(tokenizer, prompt, tail, "last"),
        "relation_tail": token_span(tokenizer, prompt, f"{rel}{tail}", "last"),
        "question_all": token_span(tokenizer, prompt, question, "last"),
    }
    return {k: [p for p in v if p >= 0] for k, v in groups.items() if v}


def make_inline_to_original_patch(
    model,
    tokenizer,
    device,
    original_prompt: str,
    inline_prompt: str,
    case: Dict,
    layer_map: Dict[str, int],
    groups: List[str],
    filtered: Dict[str, int],
    seed: int,
) -> List[Tuple[int, str, List[int], List[torch.Tensor]]]:
    original_groups = protocol_groups(tokenizer, original_prompt, case, inline=False)
    inline_groups = protocol_groups(tokenizer, inline_prompt, case, inline=True)
    patches = []
    for gi, group in enumerate(groups):
        original_pos = original_groups.get(group, [])
        inline_pos = inline_groups.get(group, [])
        if not original_pos or not inline_pos:
            filtered["group_missing"] += 1
            continue
        if len(original_pos) != len(inline_pos):
            filtered["group_len_mismatch"] += 1
            continue
        li = layer_map[group]
        original_cache = collect_positions_components(
            model, tokenizer, device, original_prompt, original_pos, [li], [COMPONENT]
        )
        inline_cache = collect_positions_components(
            model, tokenizer, device, inline_prompt, inline_pos, [li], [COMPONENT]
        )
        patches.extend(make_group_patch(
            original_cache,
            inline_cache,
            original_pos,
            li,
            COMPONENT,
            "restore",
            seed + gi * 173,
        ))
    return merge_patches(patches)


def greedy_generate(model, tokenizer, device, prompt: str, max_new_tokens: int, source_patches=None, final_patch=None) -> Dict:
    return greedy_generate_bridge(
        model,
        tokenizer,
        device,
        prompt,
        max_new_tokens,
        source_patches=source_patches or [],
        answer_patches=[],
        final_patch=final_patch,
    )


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
            "word_top0": 0,
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
        item["word_top0"] += int(row["top0_category"] == "word")
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


def row_for_probe(tokenizer, logits: torch.Tensor, prefix_id: int, old_wrong_prefix_id: int, value_prefix_ids: set[int], top_k: int) -> Dict:
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
        layer_map = group_layer_defaults(args.model)
        allowed_groups = [g for g in GROUPS if g in layer_map]
        modes = [
            "original",
            "inline",
            "final_output_inline_to_original",
            "patch_prompt_last",
            "patch_answer_label",
            "patch_question_mark_answer",
            "patch_relation_tail",
            "patch_question_all",
            "patch_all5",
        ]
        rows = []
        examples = []
        filtered = {"not_target": 0, "group_missing": 0, "group_len_mismatch": 0, "empty_patch": 0}
        target_seen = 0
        log(f"{args.model}: raw_cases={len(raw_cases)}, groups={allowed_groups}, layer_map={layer_map}")

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
            inline_state = None

            mode_specs = {
                "original": {"prompt": original_prompt, "source": [], "final": None, "state": None},
                "inline": {"prompt": inline_prompt, "source": [], "final": None, "state": None},
            }
            inline_state = final_state_probe(model, tokenizer, device, inline_prompt)
            if inline_state["final_norm_output"] is not None:
                mode_specs["final_output_inline_to_original"] = {
                    "prompt": original_prompt,
                    "source": [],
                    "final": {"kind": "output", "target": inline_state["final_norm_output"]},
                    "state": None,
                }

            for group in allowed_groups:
                patch = make_inline_to_original_patch(
                    model,
                    tokenizer,
                    device,
                    original_prompt,
                    inline_prompt,
                    case,
                    layer_map,
                    [group],
                    filtered,
                    si * 1009 + len(group),
                )
                if not patch:
                    filtered["empty_patch"] += 1
                    continue
                mode_specs[f"patch_{group}"] = {"prompt": original_prompt, "source": patch, "final": None, "state": None}

            all_patch = make_inline_to_original_patch(
                model,
                tokenizer,
                device,
                original_prompt,
                inline_prompt,
                case,
                layer_map,
                allowed_groups,
                filtered,
                si * 1009 + 638,
            )
            if all_patch:
                mode_specs["patch_all5"] = {"prompt": original_prompt, "source": all_patch, "final": None, "state": None}
            else:
                filtered["empty_patch"] += 1

            for mode in modes:
                if mode not in mode_specs:
                    continue
                spec = mode_specs[mode]
                if spec.get("state") is not None:
                    probe = spec["state"]
                else:
                    probe = final_state_probe(
                        model,
                        tokenizer,
                        device,
                        spec["prompt"],
                        source_patches=spec["source"],
                        final_patch=spec["final"],
                    )
                gen = greedy_generate(
                    model,
                    tokenizer,
                    device,
                    spec["prompt"],
                    max_new_tokens,
                    source_patches=spec["source"],
                    final_patch=spec["final"],
                )
                ev = generation_eval(gen, correct_ids, old_wrong_ids)
                probe_row = row_for_probe(
                    tokenizer,
                    probe["logits"],
                    prefix_id,
                    old_wrong_prefix_id,
                    value_prefix_ids,
                    args.top_k,
                )
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
                    **probe_row,
                }
                rows.append(row)
                if len(examples) < args.example_limit:
                    examples.append(row)

        summary = summarize(rows)
        log("Protocol backtrace modes:")
        for item in [r for r in summary["by_mode_split"] if r["split"] == "target"][:20]:
            log(
                f"  {item['mode']}: n={item['n']} tok0={item['tok0_hit']}/{item['n']} "
                f"exact={item['exact']}/{item['n']} newline={item['newline_top0']}/{item['n']} "
                f"rank={item['mean_prefix_rank']:.1f} p-newline={item['mean_prefix_minus_newline']:.3f}"
            )
        return {
            "phase": 638,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "top_k": args.top_k,
            "layer_map": layer_map,
            "groups": allowed_groups,
            "modes": modes,
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
    out_path = out_dir / f"phase638_{args.model}_inline_answer_protocol_state_backtrace_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
