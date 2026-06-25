#!/usr/bin/env python3
"""
Phase 647: Protocol Writer Graph Audit.

Phase 646 created the first protocol atlas node. This phase tests which
writer-like components inside the L17-L20 separator trajectory can reproduce or
remove the value short-answer protocol.
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


OUT_ROOT = Path("results/glm5_phase647_protocol_writer_graph_audit")
COMPONENTS = ["layer_input", "attn_out", "mlp_out", "layer_out"]
SCAN_LAYERS = [17, 18, 19, 20]
INTERVAL_SPECS = {
    "L17_20": [17, 18, 19, 20],
    "L18_19": [18, 19],
}
CONTROL_COMPONENTS = {"attn_out", "mlp_out", "layer_out"}
MODE_PREFIX = [
    "original",
    "inline",
    "to_original_interval_L17_20_layer_out_restore",
    "remove_from_inline_interval_L17_20_layer_out_restore",
    "to_original_interval_L18_19_layer_out_restore",
    "remove_from_inline_interval_L18_19_layer_out_restore",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def separator_positions(tokenizer, prompt: str, inline: bool) -> List[int]:
    return tail_units(tokenizer, prompt, inline=inline)["separator"]


def make_patch(
    target_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    source_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    target_pos: List[int],
    layer: int,
    component: str,
    control: str,
    seed: int,
) -> List[Tuple[int, str, List[int], List[torch.Tensor]]]:
    return merge_patches(make_group_patch(
        target_cache,
        source_cache,
        target_pos,
        layer,
        component,
        control,
        seed,
    ))


def make_multi_patch(
    target_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    source_cache: Dict[int, Dict[str, List[torch.Tensor]]],
    target_pos: List[int],
    layers: List[int],
    component: str,
    control: str,
    seed: int,
) -> List[Tuple[int, str, List[int], List[torch.Tensor]]]:
    patches = []
    for li in layers:
        patches.extend(make_patch(
            target_cache,
            source_cache,
            target_pos,
            li,
            component,
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
    stats = {}
    for row in rows:
        key = row["mode"]
        item = stats.setdefault(key, {
            "mode": row["mode"],
            "kind": row["kind"],
            "direction": row.get("direction"),
            "scope": row.get("scope"),
            "layer": row.get("layer"),
            "interval": row.get("interval"),
            "component": row.get("component"),
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
            "top0_category": {},
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
        item["top0_category"].setdefault(row["top0_category"], 0)
        item["top0_category"][row["top0_category"]] += 1
        gen_text = row["generation_text"].replace("\n", "\\n")
        item["generation_text"].setdefault(gen_text, 0)
        item["generation_text"][gen_text] += 1

    out = []
    for item in stats.values():
        n = max(1, item["n"])
        r = dict(item)
        r["tok0_rate"] = item["tok0_hit"] / n
        r["exact_rate"] = item["exact"] / n
        r["wrong_exact_rate"] = item["wrong_exact"] / n
        r["newline_top0_rate"] = item["newline_top0"] / n
        r["mean_prefix_rank"] = item["sum_rank"] / n
        r["mean_prefix_minus_newline"] = item["sum_prefix_minus_newline"] / n
        r["mean_prefix_margin_vs_top"] = item["sum_prefix_margin_vs_top"] / n
        r["top0_category"] = dict(sorted(r["top0_category"].items(), key=lambda kv: kv[1], reverse=True))
        r["generation_text"] = dict(sorted(r["generation_text"].items(), key=lambda kv: kv[1], reverse=True)[:8])
        out.append(r)

    def sort_key(r):
        if r["mode"] in MODE_PREFIX:
            return (0, MODE_PREFIX.index(r["mode"]))
        return (
            1,
            r.get("direction") or "",
            r.get("scope") or "",
            r.get("interval") or "",
            r.get("layer") if r.get("layer") is not None else 999,
            COMPONENTS.index(r.get("component")) if r.get("component") in COMPONENTS else 999,
            r.get("control") or "",
        )

    out.sort(key=sort_key)
    restore = [r for r in out if r.get("control") == "restore" and r["kind"] != "baseline"]
    suff = [r for r in restore if r.get("direction") == "to_original"]
    nec = [r for r in restore if r.get("direction") == "remove_from_inline"]
    suff.sort(key=lambda r: (-r["exact_rate"], r["newline_top0_rate"], r["mean_prefix_rank"]))
    nec.sort(key=lambda r: (r["exact_rate"], -r["newline_top0_rate"], r["mean_prefix_rank"]))
    return {
        "by_mode": out,
        "best_sufficiency_restore": suff[:40],
        "best_necessity_remove": nec[:40],
        "controls": [r for r in out if r.get("control") in {"random", "reverse"}],
    }


def mode_specs(original_prompt, inline_prompt, original_cache, inline_cache, original_pos, inline_pos, layers, seed):
    specs = {
        "original": {
            "kind": "baseline",
            "prompt": original_prompt,
            "patches": [],
            "direction": None,
            "scope": "baseline",
            "layer": None,
            "interval": None,
            "component": None,
            "control": None,
            "layers": [],
        },
        "inline": {
            "kind": "baseline",
            "prompt": inline_prompt,
            "patches": [],
            "direction": None,
            "scope": "baseline",
            "layer": None,
            "interval": None,
            "component": None,
            "control": None,
            "layers": [],
        },
    }
    for interval, interval_layers in INTERVAL_SPECS.items():
        interval_layers = [li for li in interval_layers if li in layers]
        for component in ["attn_out", "mlp_out", "layer_out"]:
            for direction in ["to_original", "remove_from_inline"]:
                target_cache, source_cache, target_pos, prompt = (
                    (original_cache, inline_cache, original_pos, original_prompt)
                    if direction == "to_original"
                    else (inline_cache, original_cache, inline_pos, inline_prompt)
                )
                patches = make_multi_patch(
                    target_cache,
                    source_cache,
                    target_pos,
                    interval_layers,
                    component,
                    "restore",
                    seed + len(interval) * 101 + len(component) * 17,
                )
                specs[f"{direction}_interval_{interval}_{component}_restore"] = {
                    "kind": "patch",
                    "prompt": prompt,
                    "patches": patches,
                    "direction": direction,
                    "scope": "interval",
                    "layer": None,
                    "interval": interval,
                    "component": component,
                    "control": "restore",
                    "layers": interval_layers,
                }
    for li in layers:
        for component in COMPONENTS:
            controls = ["restore"]
            if component in CONTROL_COMPONENTS:
                controls += ["random", "reverse"]
            for control in controls:
                for direction in ["to_original", "remove_from_inline"]:
                    target_cache, source_cache, target_pos, prompt = (
                        (original_cache, inline_cache, original_pos, original_prompt)
                        if direction == "to_original"
                        else (inline_cache, original_cache, inline_pos, inline_prompt)
                    )
                    patches = make_patch(
                        target_cache,
                        source_cache,
                        target_pos,
                        li,
                        component,
                        control,
                        seed + li * 997 + len(component) * 31 + len(control) * 7,
                    )
                    specs[f"{direction}_L{li:02d}_{component}_{control}"] = {
                        "kind": "patch",
                        "prompt": prompt,
                        "patches": patches,
                        "direction": direction,
                        "scope": "single_layer",
                        "layer": li,
                        "interval": None,
                        "component": component,
                        "control": control,
                        "layers": [li],
                    }
    return specs


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = [li for li in SCAN_LAYERS if 0 <= li < info.n_layers]
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        value_prefix_ids = {answer_ids(tokenizer, v)[0] for v in values}
        max_new_tokens = max(len(answer_ids(tokenizer, v)) for v in values)
        rows = []
        examples = []
        filtered = {"not_target": 0, "separator_len_mismatch": 0, "empty_patch": 0, "case_cap": 0}
        target_seen = 0
        cases_written = 0
        log(f"{args.model}: raw_cases={len(raw_cases)}, layers={layers}, components={COMPONENTS}")

        for si, case0 in enumerate(raw_cases):
            if cases_written >= args.max_cases:
                filtered["case_cap"] += 1
                break
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
                model, tokenizer, device, original_prompt, original_pos, layers, COMPONENTS
            )
            inline_cache = collect_positions_components(
                model, tokenizer, device, inline_prompt, inline_pos, layers, COMPONENTS
            )
            specs = mode_specs(
                original_prompt,
                inline_prompt,
                original_cache,
                inline_cache,
                original_pos,
                inline_pos,
                layers,
                si * 1009,
            )
            cases_written += 1

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
                    "sample_idx": si,
                    "mode": mode,
                    "kind": spec["kind"],
                    "direction": spec["direction"],
                    "scope": spec["scope"],
                    "layer": spec["layer"],
                    "interval": spec["interval"],
                    "component": spec["component"],
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
        log("Writer graph top sufficiency restore:")
        for item in summary["best_sufficiency_restore"][:12]:
            log(
                f"  {item['mode']}: n={item['n']} exact={item['exact']}/{item['n']} "
                f"tok0={item['tok0_hit']}/{item['n']} nl={item['newline_top0']}/{item['n']} "
                f"rank={item['mean_prefix_rank']:.1f}"
            )
        log("Writer graph top necessity/remove restore:")
        for item in summary["best_necessity_remove"][:12]:
            log(
                f"  {item['mode']}: n={item['n']} exact={item['exact']}/{item['n']} "
                f"tok0={item['tok0_hit']}/{item['n']} nl={item['newline_top0']}/{item['n']} "
                f"rank={item['mean_prefix_rank']:.1f}"
            )

        return {
            "phase": 647,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "top_k": args.top_k,
            "layers": layers,
            "components": COMPONENTS,
            "interval_specs": {k: [li for li in v if li in layers] for k, v in INTERVAL_SPECS.items()},
            "max_new_tokens": max_new_tokens,
            "n_raw_cases": len(raw_cases),
            "n_cases_written": cases_written,
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
    parser.add_argument("--n-tables", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--max-cases", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
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
        args.max_cases = 2
        args.top_k = min(args.top_k, 12)
        args.example_limit = 120
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 40)
        args.max_samples = max(args.max_samples, 320)
        args.max_cases = max(args.max_cases, 48)
        args.top_k = max(args.top_k, 20)
        args.example_limit = max(args.example_limit, 240)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase647_{args.model}_protocol_writer_graph_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
