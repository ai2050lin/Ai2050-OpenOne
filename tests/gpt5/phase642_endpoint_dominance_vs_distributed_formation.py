#!/usr/bin/env python3
"""
Phase 642: Endpoint Dominance vs Distributed Formation Audit.

Phase 641 showed that DS7B's separator protocol trajectory is already strong
across L10-L20 intervals. This phase splits each strong interval into endpoint
and leave-one-end variants, and tests both sufficiency (inline -> original)
and necessity (original -> inline) at the separator boundary.
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
from phase630_distributed_format_route_multisource import collect_positions_components  # noqa: E402
from phase634_multi_position_format_source_field_closure import make_group_patch, merge_patches  # noqa: E402
from phase635_final_readout_projection_bridge_audit import final_state_probe  # noqa: E402
from phase636_prefix_competitor_ladder_audit import clean_token, ladder_for_logits  # noqa: E402
from phase639_protocol_tail_minimal_causal_unit_audit import make_repair_prompt, tail_units  # noqa: E402


OUT_ROOT = Path("results/glm5_phase642_endpoint_dominance_vs_distributed_formation")
COMPONENT = "layer_out"
ALL_VARIANTS = ["full", "first", "last", "without_first", "without_last", "middle"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def model_intervals(model_name: str, n_layers: int) -> Dict[str, List[int]]:
    if model_name == "deepseek7b":
        specs = [
            ("L10_14", 10, 14),
            ("L14_17", 14, 17),
            ("L17_20", 17, 20),
            ("L10_20", 10, 20),
        ]
    elif model_name == "glm4":
        specs = [
            ("L00_08", 0, 8),
            ("L08_16", 8, 16),
            ("L16_24", 16, 24),
            ("L24_32", 24, 32),
        ]
    else:
        specs = [
            ("L00_08", 0, 8),
            ("L08_16", 8, 16),
            ("L16_24", 16, 24),
            ("L24_32", 24, 32),
        ]
    out = {}
    for name, a, b in specs:
        layers = [li for li in range(a, b + 1) if 0 <= li < n_layers]
        if layers:
            out[name] = layers
    return out


def parse_interval_spec(spec: str, n_layers: int) -> Dict[str, List[int]]:
    if not spec:
        return {}
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, rng = part.split(":", 1)
        else:
            rng = part
            name = "L" + rng.replace("-", "_")
        a, b = rng.split("-", 1)
        layers = [li for li in range(int(a), int(b) + 1) if 0 <= li < n_layers]
        if layers:
            out[name] = layers
    return out


def separator_positions(tokenizer, prompt: str, inline: bool) -> List[int]:
    return tail_units(tokenizer, prompt, inline=inline)["separator"]


def variant_layers(layers: List[int]) -> Dict[str, List[int]]:
    if not layers:
        return {}
    out = {
        "full": list(layers),
        "first": [layers[0]],
        "last": [layers[-1]],
    }
    if len(layers) > 1:
        out["without_first"] = layers[1:]
        out["without_last"] = layers[:-1]
    if len(layers) > 2:
        out["middle"] = layers[1:-1]
    return {k: v for k, v in out.items() if v}


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
            "interval": row.get("interval"),
            "variant": row.get("variant"),
            "control": row.get("control"),
            "layers": row.get("layers", []),
            "n": 0,
            "tok0_hit": 0,
            "newline_top0": 0,
            "sum_rank": 0.0,
            "sum_prefix_minus_newline": 0.0,
            "sum_prefix_margin_vs_top": 0.0,
            "top0_category": {},
            "top0_text": {},
        })
        item["n"] += 1
        item["tok0_hit"] += int(row["top0_id"] == row["prefix_id"])
        item["newline_top0"] += int(row["top0_category"] == "newline")
        item["sum_rank"] += row["prefix_rank"]
        item["sum_prefix_minus_newline"] += row["prefix_minus_newline"]
        item["sum_prefix_margin_vs_top"] += row["prefix_margin_vs_top"]
        item["top0_category"].setdefault(row["top0_category"], 0)
        item["top0_category"][row["top0_category"]] += 1
        item["top0_text"].setdefault(row["top0_text_clean"], 0)
        item["top0_text"][row["top0_text_clean"]] += 1

    mode_rows = []
    for item in by_mode.values():
        n = max(1, item["n"])
        row = dict(item)
        row["tok0_rate"] = item["tok0_hit"] / n
        row["newline_top0_rate"] = item["newline_top0"] / n
        row["mean_prefix_rank"] = item["sum_rank"] / n
        row["mean_prefix_minus_newline"] = item["sum_prefix_minus_newline"] / n
        row["mean_prefix_margin_vs_top"] = item["sum_prefix_margin_vs_top"] / n
        row["top0_category"] = dict(sorted(row["top0_category"].items(), key=lambda kv: kv[1], reverse=True))
        row["top0_text"] = dict(sorted(row["top0_text"].items(), key=lambda kv: kv[1], reverse=True)[:8])
        mode_rows.append(row)
    mode_rows.sort(key=lambda r: (
        r["kind"] != "baseline",
        r.get("direction") or "",
        r.get("interval") or "",
        ALL_VARIANTS.index(r["variant"]) if r.get("variant") in ALL_VARIANTS else 99,
        r.get("control") or "",
    ))
    return {
        "by_mode": mode_rows,
        "baselines": [r for r in mode_rows if r["kind"] == "baseline"],
        "restore_to_original": [
            r for r in mode_rows
            if r.get("direction") == "to_original" and r.get("control") == "restore"
        ],
        "remove_from_inline": [
            r for r in mode_rows
            if r.get("direction") == "remove_from_inline" and r.get("control") == "restore"
        ],
        "controls": [r for r in mode_rows if r.get("control") in {"random", "reverse"}],
    }


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        intervals = parse_interval_spec(args.intervals, info.n_layers) or model_intervals(args.model, info.n_layers)
        variants_wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
        control_variants = {v.strip() for v in args.control_variants.split(",") if v.strip()}
        controls = [c.strip() for c in args.controls.split(",") if c.strip()]
        directions = [d.strip() for d in args.directions.split(",") if d.strip()]
        layers_needed = sorted({li for layers in intervals.values() for li in layers})
        value_prefix_ids = {answer_ids(tokenizer, v)[0] for v in values}
        rows = []
        examples = []
        filtered = {
            "not_target": 0,
            "separator_len_mismatch": 0,
            "empty_patch": 0,
            "skipped_control_variant": 0,
        }
        target_seen = 0
        log(
            f"{args.model}: raw_cases={len(raw_cases)}, intervals={intervals}, variants={variants_wanted}, "
            f"controls={controls}, control_variants={sorted(control_variants)}, directions={directions}"
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

            prompt_probe = {
                "original": final_state_probe(model, tokenizer, device, original_prompt),
                "inline": final_state_probe(model, tokenizer, device, inline_prompt),
            }
            for mode, probe in prompt_probe.items():
                row = {
                    "sample_idx": si,
                    "mode": mode,
                    "kind": "baseline",
                    "direction": None,
                    "interval": None,
                    "variant": None,
                    "control": None,
                    "layers": [],
                    "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                    "old_top_wrong": base["top_wrong"],
                    "prefix_id": prefix_id,
                    "prefix_text": tokenizer.decode([prefix_id]),
                    **ladder_row(tokenizer, probe["logits"], prefix_id, old_wrong_prefix_id, value_prefix_ids, args.top_k),
                }
                rows.append(row)
                if len(examples) < args.example_limit:
                    examples.append(row)

            original_cache = collect_positions_components(
                model, tokenizer, device, original_prompt, original_pos, layers_needed, [COMPONENT]
            )
            inline_cache = collect_positions_components(
                model, tokenizer, device, inline_prompt, inline_pos, layers_needed, [COMPONENT]
            )

            for interval_name, interval_layers in intervals.items():
                vmap = variant_layers(interval_layers)
                for variant in variants_wanted:
                    layers = vmap.get(variant)
                    if not layers:
                        continue
                    for direction in directions:
                        if direction == "to_original":
                            target_prompt = original_prompt
                            target_cache = original_cache
                            source_cache = inline_cache
                            target_pos = original_pos
                        elif direction == "remove_from_inline":
                            target_prompt = inline_prompt
                            target_cache = inline_cache
                            source_cache = original_cache
                            target_pos = inline_pos
                        else:
                            raise ValueError(direction)

                        for control in controls:
                            if control != "restore" and variant not in control_variants:
                                filtered["skipped_control_variant"] += 1
                                continue
                            patch = make_interval_patch(
                                target_cache,
                                source_cache,
                                target_pos,
                                layers,
                                control,
                                si * 1009 + len(interval_name) * 17 + len(variant) * 31,
                            )
                            if not patch:
                                filtered["empty_patch"] += 1
                                continue
                            probe = final_state_probe(
                                model,
                                tokenizer,
                                device,
                                target_prompt,
                                source_patches=patch,
                            )
                            mode = f"{direction}_{interval_name}_{variant}_{control}"
                            row = {
                                "sample_idx": si,
                                "mode": mode,
                                "kind": "endpoint_variant",
                                "direction": direction,
                                "interval": interval_name,
                                "variant": variant,
                                "control": control,
                                "layers": layers,
                                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                                "old_top_wrong": base["top_wrong"],
                                "prefix_id": prefix_id,
                                "prefix_text": tokenizer.decode([prefix_id]),
                                **ladder_row(tokenizer, probe["logits"], prefix_id, old_wrong_prefix_id, value_prefix_ids, args.top_k),
                            }
                            rows.append(row)
                            if len(examples) < args.example_limit:
                                examples.append(row)

        summary = summarize(rows)
        log("Restore-to-original results:")
        for item in summary["restore_to_original"][:24]:
            log(
                f"  {item['interval']} {item['variant']}: n={item['n']} tok0={item['tok0_hit']}/{item['n']} "
                f"newline={item['newline_top0']}/{item['n']} rank={item['mean_prefix_rank']:.1f} "
                f"p-newline={item['mean_prefix_minus_newline']:.3f}"
            )
        log("Remove-from-inline results:")
        for item in summary["remove_from_inline"][:24]:
            log(
                f"  {item['interval']} {item['variant']}: n={item['n']} tok0={item['tok0_hit']}/{item['n']} "
                f"newline={item['newline_top0']}/{item['n']} rank={item['mean_prefix_rank']:.1f} "
                f"p-newline={item['mean_prefix_minus_newline']:.3f}"
            )
        return {
            "phase": 642,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "top_k": args.top_k,
            "component": COMPONENT,
            "intervals": intervals,
            "variants": variants_wanted,
            "controls": controls,
            "control_variants": sorted(control_variants),
            "directions": directions,
            "layers_needed": layers_needed,
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
    parser.add_argument("--intervals", default="")
    parser.add_argument("--variants", default="full,first,last,without_first,without_last,middle")
    parser.add_argument("--controls", default="restore,random,reverse")
    parser.add_argument("--control-variants", default="full,last")
    parser.add_argument("--directions", default="to_original,remove_from_inline")
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
        args.intervals = args.intervals or "L0_2:0-2"
        args.variants = "full,first,last,without_first,without_last,middle"
        args.controls = "restore,random,reverse"
        args.control_variants = "full,last"
        args.top_k = min(args.top_k, 12)
        args.example_limit = 60
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
    out_path = out_dir / f"phase642_{args.model}_endpoint_dominance_vs_distributed_formation_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
