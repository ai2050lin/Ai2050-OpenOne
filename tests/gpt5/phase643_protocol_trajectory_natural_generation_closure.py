#!/usr/bin/env python3
"""
Phase 643: Protocol Trajectory Natural Generation Closure.

Phase 642 showed that DS7B's L17-L20 separator protocol trajectory changes the
token0 prefix-vs-newline competition. This phase tests whether the same small
set of trajectory patches changes greedy natural generation.
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
from phase639_protocol_tail_minimal_causal_unit_audit import make_repair_prompt, tail_units  # noqa: E402


OUT_ROOT = Path("results/glm5_phase643_protocol_trajectory_natural_generation_closure")
COMPONENT = "layer_out"
INTERVAL_NAME = "L17_20"
FULL_LAYERS = [17, 18, 19, 20]
MIDDLE_LAYERS = [18, 19]


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


def summarize(rows: List[Dict]) -> Dict:
    by_mode = {}
    for row in rows:
        item = by_mode.setdefault(row["mode"], {
            "mode": row["mode"],
            "kind": row["kind"],
            "direction": row.get("direction"),
            "variant": row.get("variant"),
            "control": row.get("control"),
            "layers": row.get("layers", []),
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
            "generation_text": {},
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
        gen_text = row["generation_text"].replace("\n", "\\n")
        item["generation_text"].setdefault(gen_text, 0)
        item["generation_text"][gen_text] += 1

    out = []
    mode_order = [
        "original",
        "inline",
        "to_original_full_restore",
        "to_original_middle_restore",
        "to_original_full_random",
        "to_original_full_reverse",
        "remove_from_inline_full_restore",
        "remove_from_inline_middle_restore",
        "remove_from_inline_full_random",
        "remove_from_inline_full_reverse",
    ]
    for item in by_mode.values():
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
        row["generation_text"] = dict(sorted(row["generation_text"].items(), key=lambda kv: kv[1], reverse=True)[:10])
        out.append(row)
    out.sort(key=lambda r: mode_order.index(r["mode"]) if r["mode"] in mode_order else 999)
    return {"by_mode": out}


def mode_specs(original_prompt, inline_prompt, original_cache, inline_cache, original_pos, inline_pos, seed: int):
    specs = {
        "original": {
            "kind": "baseline",
            "prompt": original_prompt,
            "source": [],
            "direction": None,
            "variant": None,
            "control": None,
            "layers": [],
        },
        "inline": {
            "kind": "baseline",
            "prompt": inline_prompt,
            "source": [],
            "direction": None,
            "variant": None,
            "control": None,
            "layers": [],
        },
    }
    variants = {
        "full": FULL_LAYERS,
        "middle": MIDDLE_LAYERS,
    }
    for variant, layers in variants.items():
        for control in ["restore", "random", "reverse"] if variant == "full" else ["restore"]:
            patch = make_interval_patch(
                original_cache,
                inline_cache,
                original_pos,
                layers,
                control,
                seed + len(variant) * 101 + len(control) * 17,
            )
            specs[f"to_original_{variant}_{control}"] = {
                "kind": "trajectory_patch",
                "prompt": original_prompt,
                "source": patch,
                "direction": "to_original",
                "variant": variant,
                "control": control,
                "layers": layers,
            }
            patch = make_interval_patch(
                inline_cache,
                original_cache,
                inline_pos,
                layers,
                control,
                seed + len(variant) * 101 + len(control) * 17 + 313,
            )
            specs[f"remove_from_inline_{variant}_{control}"] = {
                "kind": "trajectory_patch",
                "prompt": inline_prompt,
                "source": patch,
                "direction": "remove_from_inline",
                "variant": variant,
                "control": control,
                "layers": layers,
            }
    return specs


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        needed_layers = [li for li in FULL_LAYERS if 0 <= li < info.n_layers]
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        value_prefix_ids = {answer_ids(tokenizer, v)[0] for v in values}
        max_new_tokens = max(len(answer_ids(tokenizer, v)) for v in values)
        rows = []
        examples = []
        filtered = {"not_target": 0, "separator_len_mismatch": 0, "empty_patch": 0}
        target_seen = 0
        log(
            f"{args.model}: raw_cases={len(raw_cases)}, layers={needed_layers}, "
            f"max_new_tokens={max_new_tokens}"
        )

        for si, case0 in enumerate(raw_cases):
            case = dict(case0)
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
            base = winner_stats(base_scores, case["correct"])
            repair = winner_stats(repair_scores, case["correct"])
            target_case = (not base["correct"]) and repair["correct"]
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
            original_pos = separator_positions(tokenizer, original_prompt, inline=False)
            inline_pos = separator_positions(tokenizer, inline_prompt, inline=True)
            if not original_pos or len(original_pos) != len(inline_pos):
                filtered["separator_len_mismatch"] += 1
                continue

            original_cache = collect_positions_components(
                model, tokenizer, device, original_prompt, original_pos, needed_layers, [COMPONENT]
            )
            inline_cache = collect_positions_components(
                model, tokenizer, device, inline_prompt, inline_pos, needed_layers, [COMPONENT]
            )
            specs = mode_specs(
                original_prompt,
                inline_prompt,
                original_cache,
                inline_cache,
                original_pos,
                inline_pos,
                si * 1009,
            )

            for mode in [
                "original",
                "inline",
                "to_original_full_restore",
                "to_original_middle_restore",
                "to_original_full_random",
                "to_original_full_reverse",
                "remove_from_inline_full_restore",
                "remove_from_inline_middle_restore",
                "remove_from_inline_full_random",
                "remove_from_inline_full_reverse",
            ]:
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
                    "sample_idx": si,
                    "mode": mode,
                    "kind": spec["kind"],
                    "direction": spec["direction"],
                    "variant": spec["variant"],
                    "control": spec["control"],
                    "layers": spec["layers"],
                    "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                    "old_top_wrong": base["top_wrong"],
                    "prefix_id": prefix_id,
                    "prefix_text": tokenizer.decode([prefix_id]),
                    "old_wrong_prefix_id": old_wrong_prefix_id,
                    "old_wrong_prefix_text": tokenizer.decode([old_wrong_prefix_id]),
                    "eval": ev,
                    "generation_text": gen["text"],
                    "generation_tokens": gen["tokens"],
                    **ladder_row(tokenizer, probe["logits"], prefix_id, old_wrong_prefix_id, value_prefix_ids, args.top_k),
                }
                rows.append(row)
                if len(examples) < args.example_limit:
                    examples.append(row)

        summary = summarize(rows)
        log("Natural generation closure:")
        for item in summary["by_mode"]:
            log(
                f"  {item['mode']}: n={item['n']} tok0={item['tok0_hit']}/{item['n']} "
                f"exact={item['exact']}/{item['n']} wrong={item['wrong_exact']}/{item['n']} "
                f"newline={item['newline_top0']}/{item['n']} rank={item['mean_prefix_rank']:.1f} "
                f"p-newline={item['mean_prefix_minus_newline']:.3f}"
            )
        return {
            "phase": 643,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "top_k": args.top_k,
            "component": COMPONENT,
            "interval": INTERVAL_NAME,
            "full_layers": needed_layers,
            "middle_layers": [li for li in MIDDLE_LAYERS if 0 <= li < info.n_layers],
            "max_new_tokens": max_new_tokens,
            "n_raw_cases": len(raw_cases),
            "n_cases_written": len({r["sample_idx"] for r in rows}),
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
    parser.add_argument("--example-limit", type=int, default=160)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 8
        args.top_k = min(args.top_k, 12)
        args.example_limit = 80
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 32)
        args.max_samples = max(args.max_samples, 256)
        args.top_k = max(args.top_k, 20)
        args.example_limit = max(args.example_limit, 200)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase643_{args.model}_protocol_trajectory_natural_generation_closure_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
