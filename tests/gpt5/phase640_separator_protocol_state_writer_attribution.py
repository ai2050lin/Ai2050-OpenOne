#!/usr/bin/env python3
"""
Phase 640: Separator Protocol State Writer Attribution.

Phase 639 localized DS7B's protocol switch to the separator boundary. This
phase scans which layer/component at the separator can carry the inline
protocol state into the original prompt.
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


OUT_ROOT = Path("results/glm5_phase640_separator_protocol_state_writer_attribution")
COMPONENTS = ["layer_input", "attn_out", "mlp_out", "layer_out"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def default_anchor_layer(model_name: str, n_layers: int) -> int:
    if model_name == "qwen3":
        return min(27, n_layers - 1)
    if model_name == "glm4":
        return min(32, n_layers - 1)
    if model_name == "deepseek7b":
        return min(21, n_layers - 1)
    return max(0, n_layers - 6)


def default_scan_layers(model_name: str, n_layers: int) -> List[int]:
    anchor = default_anchor_layer(model_name, n_layers)
    coarse = set(range(0, n_layers, 2))
    local = set(range(max(0, anchor - 5), min(n_layers, anchor + 6)))
    final = set(range(max(0, n_layers - 4), n_layers))
    return sorted(coarse | local | final)


def default_control_layers(model_name: str, n_layers: int) -> List[int]:
    anchor = default_anchor_layer(model_name, n_layers)
    return sorted(set(range(max(0, anchor - 2), min(n_layers, anchor + 3))))


def parse_layers(spec: str, n_layers: int, fallback: List[int]) -> List[int]:
    if not spec:
        return fallback
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(li for li in out if 0 <= li < n_layers)


def separator_positions(tokenizer, prompt: str, inline: bool) -> List[int]:
    return tail_units(tokenizer, prompt, inline=inline)["separator"]


def patch_from_caches(
    original_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    inline_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    original_pos: List[int],
    layer_idx: int,
    component: str,
    mode: str,
    seed: int,
) -> List[Tuple[int, str, List[int], List[torch.Tensor]]]:
    return merge_patches(make_group_patch(
        original_cache,
        inline_cache,
        original_pos,
        layer_idx,
        component,
        mode,
        seed,
    ))


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
    by_layer_component = {}
    for row in rows:
        item = by_mode.setdefault(row["mode"], {
            "mode": row["mode"],
            "kind": row["kind"],
            "layer": row.get("layer"),
            "component": row.get("component"),
            "control": row.get("control"),
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
        if row["kind"] == "patch":
            key = (row["layer"], row["component"], row["control"])
            lc = by_layer_component.setdefault(key, {
                "layer": row["layer"],
                "component": row["component"],
                "control": row["control"],
                "n": 0,
                "tok0_hit": 0,
                "newline_top0": 0,
                "sum_rank": 0.0,
                "sum_prefix_minus_newline": 0.0,
            })
            lc["n"] += 1
            lc["tok0_hit"] += int(row["top0_id"] == row["prefix_id"])
            lc["newline_top0"] += int(row["top0_category"] == "newline")
            lc["sum_rank"] += row["prefix_rank"]
            lc["sum_prefix_minus_newline"] += row["prefix_minus_newline"]

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
    mode_rows.sort(key=lambda x: (x["newline_top0_rate"], -x["tok0_rate"], x["mean_prefix_rank"], x["mode"]))

    lc_rows = []
    for item in by_layer_component.values():
        n = max(1, item["n"])
        row = dict(item)
        row["tok0_rate"] = item["tok0_hit"] / n
        row["newline_top0_rate"] = item["newline_top0"] / n
        row["mean_prefix_rank"] = item["sum_rank"] / n
        row["mean_prefix_minus_newline"] = item["sum_prefix_minus_newline"] / n
        lc_rows.append(row)
    lc_rows.sort(key=lambda x: (x["control"], x["newline_top0_rate"], -x["tok0_rate"], x["mean_prefix_rank"], x["layer"], x["component"]))
    return {
        "by_mode": mode_rows,
        "by_layer_component": lc_rows,
        "best_restore": [r for r in lc_rows if r["control"] == "restore"][:40],
        "best_controls": [r for r in lc_rows if r["control"] != "restore"][:40],
    }


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        scan_layers = parse_layers(args.layers, info.n_layers, default_scan_layers(args.model, info.n_layers))
        control_layers = parse_layers(args.control_layers, info.n_layers, default_control_layers(args.model, info.n_layers))
        value_prefix_ids = {answer_ids(tokenizer, v)[0] for v in values}
        rows = []
        examples = []
        filtered = {"not_target": 0, "separator_len_mismatch": 0, "empty_patch": 0}
        target_seen = 0
        log(
            f"{args.model}: raw_cases={len(raw_cases)}, scan_layers={scan_layers}, "
            f"control_layers={control_layers}, components={COMPONENTS}"
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

            baselines = {
                "original": final_state_probe(model, tokenizer, device, original_prompt),
                "inline": final_state_probe(model, tokenizer, device, inline_prompt),
            }
            for mode, probe in baselines.items():
                row = {
                    "sample_idx": si,
                    "mode": mode,
                    "kind": "baseline",
                    "layer": None,
                    "component": None,
                    "control": None,
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
                model, tokenizer, device, original_prompt, original_pos, scan_layers, COMPONENTS
            )
            inline_cache = collect_positions_components(
                model, tokenizer, device, inline_prompt, inline_pos, scan_layers, COMPONENTS
            )

            for li in scan_layers:
                for comp in COMPONENTS:
                    controls = ["restore"]
                    if li in control_layers:
                        controls.extend(["random", "reverse"])
                    for control in controls:
                        patch = patch_from_caches(
                            original_cache,
                            inline_cache,
                            original_pos,
                            li,
                            comp,
                            control,
                            si * 1009 + li * 101 + len(comp),
                        )
                        if not patch:
                            filtered["empty_patch"] += 1
                            continue
                        probe = final_state_probe(
                            model,
                            tokenizer,
                            device,
                            original_prompt,
                            source_patches=patch,
                        )
                        mode = f"L{li:02d}_{comp}_{control}"
                        row = {
                            "sample_idx": si,
                            "mode": mode,
                            "kind": "patch",
                            "layer": li,
                            "component": comp,
                            "control": control,
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
        log("Best restore writer candidates:")
        for item in summary["best_restore"][:16]:
            log(
                f"  L{item['layer']:02d} {item['component']}: n={item['n']} "
                f"tok0={item['tok0_hit']}/{item['n']} newline={item['newline_top0']}/{item['n']} "
                f"rank={item['mean_prefix_rank']:.1f} p-newline={item['mean_prefix_minus_newline']:.3f}"
            )
        return {
            "phase": 640,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "top_k": args.top_k,
            "scan_layers": scan_layers,
            "control_layers": control_layers,
            "components": COMPONENTS,
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
    parser.add_argument("--layers", default="")
    parser.add_argument("--control-layers", default="")
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
    parser.add_argument("--save-rows", action="store_true")
    parser.add_argument("--example-limit", type=int, default=120)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 8
        args.layers = args.layers or "0,1,2"
        args.control_layers = args.control_layers or "1"
        args.top_k = min(args.top_k, 12)
        args.example_limit = 40
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 32)
        args.max_samples = max(args.max_samples, 256)
        args.top_k = max(args.top_k, 20)
        args.example_limit = max(args.example_limit, 160)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase640_{args.model}_separator_protocol_state_writer_attribution_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
