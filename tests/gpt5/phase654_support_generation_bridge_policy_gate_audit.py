#!/usr/bin/env python3
"""
Phase 654: Support-to-Generation Bridge and Final Policy Gate Audit.

Phase 653 showed that localized intent-gate patches can strongly change value
support, but generation closure remains partial. This phase narrows to restore
patches at the strongest sites and audits the final readout / generation bridge:
rank, top0, top-k competition, final norm movement, and generated first token.
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
import torch.nn.functional as F

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


OUT_ROOT = Path("results/glm5_phase654_support_generation_bridge_policy_gate_audit")
TASK_ORDER = ["explanation_required", "yes_no_required"]
DIRECTIONS = ["value_to_task", "task_to_value"]

SITE_SPECS = {
    "qwen3": [
        {
            "name": "separator_input_edge",
            "positions": ["separator"],
            "layers": [14],
            "components": ["layer_input"],
        },
        {
            "name": "early_peak_layer_out",
            "positions": ["label_aligned", "separator", "relation_tail"],
            "layers": [14, 15, 16, 17],
            "components": ["layer_out"],
        },
    ],
    "glm4": [
        {
            "name": "l22_peak_layer_out",
            "positions": ["label_aligned", "separator", "relation_tail"],
            "layers": [22],
            "components": ["layer_out"],
        },
        {
            "name": "late_peak_layer_out",
            "positions": ["label_aligned", "separator", "relation_tail"],
            "layers": [21, 22],
            "components": ["layer_out"],
        },
    ],
    "deepseek7b": [
        {
            "name": "l22_peak_layer_out",
            "positions": ["label_aligned", "separator", "relation_tail"],
            "layers": [22],
            "components": ["layer_out"],
        },
        {
            "name": "late_peak_layer_out",
            "positions": ["label_aligned", "separator", "relation_tail"],
            "layers": [20, 21, 22],
            "components": ["layer_out"],
        },
    ],
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tensor_metrics(base: torch.Tensor | None, patched: torch.Tensor | None) -> Dict[str, float | None]:
    if base is None or patched is None:
        return {"final_cos": None, "final_l2": None, "final_norm_delta": None}
    b = base.float()
    p = patched.float()
    return {
        "final_cos": float(F.cosine_similarity(b, p, dim=0).item()),
        "final_l2": float(torch.norm(p - b).item()),
        "final_norm_delta": float(torch.norm(p).item() - torch.norm(b).item()),
    }


def top5_text(tokenizer, gen: Dict) -> List[Dict]:
    if not gen.get("top5"):
        return []
    return [
        {"id": x["id"], "text": x["text"], "logprob": x["logprob"]}
        for x in gen["top5"][0]
    ]


def build_patch(
    target_caches: Dict[str, Dict],
    source_caches: Dict[str, Dict],
    target_units: Dict[str, List[int]],
    source_units: Dict[str, List[int]],
    spec: Dict,
    layers: List[int],
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
                "restore",
                seed + pi * 1009 + ci * 131,
            )
            if not part:
                stats["empty_patch"] += 1
            patches.extend(part)
    if not patches:
        stats["empty_patch"] += 1
    return patches, stats


def collect_caches(model, tokenizer, device, prompt: str, units: Dict[str, List[int]], layers: List[int], components: List[str]):
    out = {}
    for pos_name, pos in units.items():
        if pos:
            out[pos_name] = collect_positions_components(model, tokenizer, device, prompt, pos, layers, components)
    return out


def summarize(rows: List[Dict]) -> Dict:
    by_key: Dict[Tuple, Dict] = {}
    bridge_failures = []
    for row in rows:
        key = (
            row["pair_task"],
            row["eval_task"],
            row["mode"],
            row.get("direction"),
            row.get("site"),
        )
        item = by_key.setdefault(key, {
            "pair_task": row["pair_task"],
            "eval_task": row["eval_task"],
            "mode": row["mode"],
            "kind": row["kind"],
            "direction": row.get("direction"),
            "site": row.get("site"),
            "positions": row.get("positions", []),
            "layers": row.get("layers", []),
            "components": row.get("components", []),
            "n": 0,
            "exact": 0,
            "tok0_hit": 0,
            "support_without_generation": 0,
            "sum_rank": 0.0,
            "sum_prefix_margin_vs_top": 0.0,
            "sum_final_l2": 0.0,
            "final_l2_n": 0,
            "top0_category": {},
            "gen_first_text": {},
        })
        item["n"] += 1
        exact = int(row["eval"]["exact_correct"])
        tok0 = int(row["top0_id"] == row["prefix_id"])
        item["exact"] += exact
        item["tok0_hit"] += tok0
        item["sum_rank"] += row["prefix_rank"]
        item["sum_prefix_margin_vs_top"] += row["prefix_margin_vs_top"]
        if row.get("final_l2") is not None:
            item["sum_final_l2"] += row["final_l2"]
            item["final_l2_n"] += 1
        item["top0_category"].setdefault(row["top0_category"], 0)
        item["top0_category"][row["top0_category"]] += 1
        first = row.get("gen_first_text", "")
        item["gen_first_text"].setdefault(first, 0)
        item["gen_first_text"][first] += 1
        if row["kind"] == "patch" and row["prefix_rank"] <= 15 and not exact:
            item["support_without_generation"] += 1
            bridge_failures.append(row)

    out = []
    for item in by_key.values():
        n = max(1, item["n"])
        row = dict(item)
        row["exact_rate"] = item["exact"] / n
        row["tok0_rate"] = item["tok0_hit"] / n
        row["support_without_generation_rate"] = item["support_without_generation"] / n
        row["mean_prefix_rank"] = item["sum_rank"] / n
        row["mean_prefix_margin_vs_top"] = item["sum_prefix_margin_vs_top"] / n
        row["mean_final_l2"] = item["sum_final_l2"] / max(1, item["final_l2_n"])
        row["top0_category"] = dict(sorted(row["top0_category"].items(), key=lambda kv: kv[1], reverse=True))
        row["gen_first_text"] = dict(sorted(row["gen_first_text"].items(), key=lambda kv: kv[1], reverse=True)[:10])
        out.append(row)
    out.sort(key=lambda r: (
        r["pair_task"],
        r["eval_task"],
        0 if r["kind"] == "baseline" else 1,
        DIRECTIONS.index(r["direction"]) if r.get("direction") in DIRECTIONS else -1,
        r.get("site") or "",
    ))
    bridge_failures.sort(key=lambda r: (r["prefix_rank"], -r["prefix_margin_vs_top"]))
    return {
        "by_mode": out,
        "bridge_failures": bridge_failures[:160],
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
            value_prompt, value_relation, value_intent = make_prompt(
                case, "short_value_allowed", relation_pool, tokenizer, item["sample_idx"]
            )
            value_units_all = position_units(tokenizer, value_prompt, case, value_relation, value_intent)
            value_units = {p: value_units_all.get(p, []) for p in all_positions}
            value_caches = collect_caches(model, tokenizer, device, value_prompt, value_units, all_layers, all_components)
            value_probe_base = final_state_probe(model, tokenizer, device, value_prompt)

            for task_i, task in enumerate(TASK_ORDER):
                task_prompt, task_relation, task_intent = make_prompt(
                    case, task, relation_pool, tokenizer, item["sample_idx"] + task_i * 17
                )
                task_units_all = position_units(tokenizer, task_prompt, case, task_relation, task_intent)
                task_units = {p: task_units_all.get(p, []) for p in all_positions}
                task_caches = collect_caches(model, tokenizer, device, task_prompt, task_units, all_layers, all_components)
                task_probe_base = final_state_probe(model, tokenizer, device, task_prompt)
                specs = {
                    "value_baseline": {
                        "kind": "baseline",
                        "prompt": value_prompt,
                        "patches": [],
                        "eval_task": "short_value_allowed",
                        "direction": None,
                        "site": None,
                        "positions": [],
                        "layers": [],
                        "components": [],
                        "base_probe": value_probe_base,
                    },
                    "task_baseline": {
                        "kind": "baseline",
                        "prompt": task_prompt,
                        "patches": [],
                        "eval_task": task,
                        "direction": None,
                        "site": None,
                        "positions": [],
                        "layers": [],
                        "components": [],
                        "base_probe": task_probe_base,
                    },
                }
                for site_i, site in enumerate(site_specs):
                    site_layers = [li for li in site["layers"] if 0 <= li < info.n_layers]
                    for direction in DIRECTIONS:
                        if direction == "value_to_task":
                            patches, stats = build_patch(
                                task_caches,
                                value_caches,
                                task_units,
                                value_units,
                                site,
                                site_layers,
                                item["sample_idx"] * 1009 + task_i * 199 + site_i * 37,
                            )
                            prompt = task_prompt
                            eval_task = task
                            base_probe = task_probe_base
                        else:
                            patches, stats = build_patch(
                                value_caches,
                                task_caches,
                                value_units,
                                task_units,
                                site,
                                site_layers,
                                item["sample_idx"] * 1009 + task_i * 199 + site_i * 37 + 50021,
                            )
                            prompt = value_prompt
                            eval_task = "short_value_allowed"
                            base_probe = value_probe_base
                        for k, v in stats.items():
                            filtered[k] += v
                        if not patches:
                            continue
                        mode = f"{direction}_{site['name']}_restore"
                        specs[mode] = {
                            "kind": "patch",
                            "prompt": prompt,
                            "patches": patches,
                            "eval_task": eval_task,
                            "direction": direction,
                            "site": site["name"],
                            "positions": site["positions"],
                            "layers": site_layers,
                            "components": site["components"],
                            "base_probe": base_probe,
                        }

                for mode, spec in specs.items():
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
                    final_metrics = tensor_metrics(
                        spec["base_probe"].get("final_norm_output"),
                        probe.get("final_norm_output"),
                    )
                    row = {
                        "sample_idx": item["sample_idx"],
                        "item_idx": item_i,
                        "pair_task": task,
                        "eval_task": spec["eval_task"],
                        "mode": mode,
                        "kind": spec["kind"],
                        "direction": spec["direction"],
                        "site": spec["site"],
                        "positions": spec["positions"],
                        "layers": spec["layers"],
                        "components": spec["components"],
                        "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                        "prefix_id": prefix_id,
                        "prefix_text": tokenizer.decode([prefix_id]),
                        "old_wrong_prefix_id": old_wrong_prefix_id,
                        "old_wrong_prefix_text": tokenizer.decode([old_wrong_prefix_id]),
                        "eval": ev,
                        "generation_text": gen["text"],
                        "generation_tokens": gen["tokens"],
                        "gen_first_id": gen["ids"][0] if gen["ids"] else None,
                        "gen_first_text": tokenizer.decode([gen["ids"][0]]) if gen["ids"] else "",
                        "gen_first_top5": top5_text(tokenizer, gen),
                        **final_metrics,
                        **ladder_row(tokenizer, probe["logits"], prefix_id, old_wrong_prefix_id, value_prefix_ids, args.top_k),
                    }
                    rows.append(row)
                    if len(examples) < args.example_limit:
                        examples.append(row)

        summary = summarize(rows)
        log("Support-generation bridge:")
        for row in summary["by_mode"]:
            if row["kind"] != "patch":
                continue
            log(
                f"  {row['pair_task']} {row['direction']} {row['site']}: "
                f"n={row['n']} rank={row['mean_prefix_rank']:.1f} "
                f"exact={row['exact']}/{row['n']} tok0={row['tok0_hit']}/{row['n']} "
                f"support_no_gen={row['support_without_generation']}/{row['n']} "
                f"top0={row['top0_category']}"
            )
        return {
            "phase": 654,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "site_specs": site_specs,
            "tasks": TASK_ORDER,
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
    parser.add_argument("--max-cases", type=int, default=10)
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
        args.n_tables = max(args.n_tables, 40)
        args.max_samples = max(args.max_samples, 320)
        args.max_cases = max(args.max_cases, 20)
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
    out_path = out_dir / f"phase654_{args.model}_support_generation_bridge_policy_gate_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
