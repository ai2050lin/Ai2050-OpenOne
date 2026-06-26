#!/usr/bin/env python3
"""
Phase 656: Space-Newline-Explanation Prior Writer Localization Audit.

Phase 655 showed that correct-prefix support often loses to final-token format
priors. This phase fixes the strongest localized intent-gate restore patches
and ablates attn_out / mlp_out at the final readout position, looking for
components whose removal increases correct_prefix-vs-format margins.
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

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import get_mlp  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase630_distributed_format_route_multisource import (  # noqa: E402
    collect_positions_components,
    install_source_patch_hooks,
)
from phase635_final_readout_projection_bridge_audit import final_state_probe  # noqa: E402
from phase647_protocol_writer_graph_audit import make_multi_patch  # noqa: E402
from phase651_task_intent_gate_protocol_boundary_audit import (  # noqa: E402
    ladder_row,
    make_prompt,
    position_units,
    select_cases,
)


OUT_ROOT = Path("results/glm5_phase656_format_prior_writer_localization_audit")
TASK_ORDER = ["explanation_required", "yes_no_required"]
SCAN_COMPONENTS = ["attn_out", "mlp_out"]
POLICY_GROUPS = ["space", "newline", "explanation", "word", "punctuation", "symbol"]

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


def default_scan_layers(model_key: str, n_layers: int) -> List[int]:
    if model_key == "qwen3":
        raw = range(12, 25)
    else:
        raw = range(18, 28)
    return [li for li in raw if 0 <= li < n_layers]


def group_margin(row: Dict, group: str) -> float | None:
    item = row.get("groups", {}).get(group)
    if not item:
        return None
    return item.get("prefix_minus_group_max")


def final_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False)) - 1


def install_component_ablation_hook(model, layer_idx: int, component: str, pos: int):
    layers = get_layers(model)
    layer = layers[layer_idx]
    module = get_attn(layer) if component == "attn_out" else get_mlp(layer)
    if module is None:
        return None

    def hook(_module, _inputs, output):
        y = extract_tensor(output)
        y_new = y.clone()
        if 0 <= pos < y_new.shape[1]:
            y_new[0, pos, :] = 0
        if isinstance(output, tuple):
            return (y_new,) + output[1:]
        return y_new

    return module.register_forward_hook(hook)


def probe_with_ablation(model, tokenizer, device, prompt: str, patches, layer_idx: int | None, component: str | None):
    handles = []
    try:
        if layer_idx is not None and component is not None:
            h = install_component_ablation_hook(model, layer_idx, component, final_pos(tokenizer, prompt))
            if h is not None:
                handles.append(h)
        return final_state_probe(model, tokenizer, device, prompt, source_patches=patches)
    finally:
        for h in handles:
            h.remove()


def build_site_patch(
    target_caches: Dict[str, Dict],
    source_caches: Dict[str, Dict],
    target_units: Dict[str, List[int]],
    source_units: Dict[str, List[int]],
    site: Dict,
    layers: List[int],
    seed: int,
) -> Tuple[List[Tuple[int, str, List[int], List[torch.Tensor]]], Dict[str, int]]:
    patches = []
    stats = {"position_missing": 0, "position_len_mismatch": 0, "empty_patch": 0}
    for pi, pos_name in enumerate(site["positions"]):
        target_pos = target_units.get(pos_name, [])
        source_pos = source_units.get(pos_name, [])
        if not target_pos or not source_pos or pos_name not in target_caches or pos_name not in source_caches:
            stats["position_missing"] += 1
            continue
        if len(target_pos) != len(source_pos):
            stats["position_len_mismatch"] += 1
            continue
        for ci, component in enumerate(site["components"]):
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


def make_metric_row(tokenizer, probe: Dict, prefix_id: int, old_wrong_prefix_id: int, value_prefix_ids: set[int], top_k: int) -> Dict:
    row = ladder_row(tokenizer, probe["logits"], prefix_id, old_wrong_prefix_id, value_prefix_ids, top_k)
    row["policy_margins"] = {g: group_margin(row, g) for g in POLICY_GROUPS}
    row["format_margin"] = min(
        [m for g, m in row["policy_margins"].items() if m is not None],
        default=None,
    )
    return row


def summarize(rows: List[Dict]) -> Dict:
    baseline = {}
    for row in rows:
        if row["kind"] == "site_restore_baseline":
            key = (row["sample_idx"], row["pair_task"], row["site"])
            baseline[key] = row

    by_key: Dict[Tuple, Dict] = {}
    for row in rows:
        if row["kind"] != "component_ablation":
            continue
        base = baseline.get((row["sample_idx"], row["pair_task"], row["site"]))
        if not base:
            continue
        key = (row["pair_task"], row["site"], row["layer"], row["component"], base["top0_category"])
        item = by_key.setdefault(key, {
            "pair_task": row["pair_task"],
            "site": row["site"],
            "layer": row["layer"],
            "component": row["component"],
            "baseline_top0_category": base["top0_category"],
            "n": 0,
            "flipped_to_correct": 0,
            "sum_rank_delta": 0.0,
            "sum_top_margin_delta": 0.0,
            "sum_format_margin_delta": 0.0,
            "group_margin_delta": {g: 0.0 for g in POLICY_GROUPS},
            "group_margin_count": {g: 0 for g in POLICY_GROUPS},
            "baseline_top0": {},
            "ablated_top0": {},
        })
        item["n"] += 1
        item["flipped_to_correct"] += int(row["top0_id"] == row["prefix_id"] and base["top0_id"] != base["prefix_id"])
        item["sum_rank_delta"] += base["prefix_rank"] - row["prefix_rank"]
        item["sum_top_margin_delta"] += row["prefix_margin_vs_top"] - base["prefix_margin_vs_top"]
        if row.get("format_margin") is not None and base.get("format_margin") is not None:
            item["sum_format_margin_delta"] += row["format_margin"] - base["format_margin"]
        for group in POLICY_GROUPS:
            m0 = base.get("policy_margins", {}).get(group)
            m1 = row.get("policy_margins", {}).get(group)
            if m0 is not None and m1 is not None:
                item["group_margin_delta"][group] += m1 - m0
                item["group_margin_count"][group] += 1
        item["baseline_top0"][base["top0_category"]] = item["baseline_top0"].get(base["top0_category"], 0) + 1
        item["ablated_top0"][row["top0_category"]] = item["ablated_top0"].get(row["top0_category"], 0) + 1

    out = []
    for item in by_key.values():
        n = max(1, item["n"])
        r = dict(item)
        r["mean_rank_improvement"] = item["sum_rank_delta"] / n
        r["mean_top_margin_delta"] = item["sum_top_margin_delta"] / n
        r["mean_format_margin_delta"] = item["sum_format_margin_delta"] / n
        r["mean_group_margin_delta"] = {
            g: item["group_margin_delta"][g] / item["group_margin_count"][g]
            for g in POLICY_GROUPS
            if item["group_margin_count"][g]
        }
        r["baseline_top0"] = dict(sorted(item["baseline_top0"].items(), key=lambda kv: kv[1], reverse=True))
        r["ablated_top0"] = dict(sorted(item["ablated_top0"].items(), key=lambda kv: kv[1], reverse=True))
        out.append(r)
    # Positive top margin delta means removing this component helped correct_prefix
    # against the previous winner. This is a format-prior writer candidate.
    candidates = sorted(
        out,
        key=lambda r: (-(r["mean_top_margin_delta"]), -r["flipped_to_correct"], -(r["mean_rank_improvement"])),
    )
    blockers = sorted(
        out,
        key=lambda r: (r["mean_top_margin_delta"], r["flipped_to_correct"], r["mean_rank_improvement"]),
    )
    by_mode = []
    for row in rows:
        if row["kind"] == "site_restore_baseline":
            by_mode.append(row)
    return {
        "by_component": out,
        "format_prior_writer_candidates": candidates[:120],
        "value_support_writer_candidates": blockers[:120],
        "site_restore_baselines": by_mode,
    }


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        scan_layers = [li for li in (args.layers or default_scan_layers(args.model, info.n_layers)) if 0 <= li < info.n_layers]
        site_specs = SITE_SPECS[args.model]
        site_layers = sorted({li for s in site_specs for li in s["layers"] if 0 <= li < info.n_layers})
        site_components = sorted({c for s in site_specs for c in s["components"]})
        site_positions = sorted({p for s in site_specs for p in s["positions"]})
        values = CANDIDATE_VALUES[:4]
        value_prefix_ids = {answer_ids(tokenizer, v)[0] for v in values}
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        relation_pool = sorted({c["relation"] for c in raw_cases})
        selected, selection_stats = select_cases(
            model, tokenizer, device, raw_cases, values, args.max_cases, relation_pool
        )
        rows = []
        examples = []
        filtered = {"position_missing": 0, "position_len_mismatch": 0, "empty_patch": 0}
        log(
            f"{args.model}: selected={len(selected)}, scan_layers={scan_layers}, "
            f"site_specs={[s['name'] for s in site_specs]}"
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
            value_units = {p: value_units_all.get(p, []) for p in site_positions}
            value_caches = collect_caches(model, tokenizer, device, value_prompt, value_units, site_layers, site_components)

            for task_i, task in enumerate(TASK_ORDER):
                task_prompt, task_relation, task_intent = make_prompt(
                    case, task, relation_pool, tokenizer, item["sample_idx"] + task_i * 17
                )
                task_units_all = position_units(tokenizer, task_prompt, case, task_relation, task_intent)
                task_units = {p: task_units_all.get(p, []) for p in site_positions}
                task_caches = collect_caches(model, tokenizer, device, task_prompt, task_units, site_layers, site_components)

                for site_i, site in enumerate(site_specs):
                    layers0 = [li for li in site["layers"] if 0 <= li < info.n_layers]
                    patches, stats = build_site_patch(
                        task_caches,
                        value_caches,
                        task_units,
                        value_units,
                        site,
                        layers0,
                        item["sample_idx"] * 1009 + task_i * 199 + site_i * 37,
                    )
                    for k, v in stats.items():
                        filtered[k] += v
                    if not patches:
                        continue

                    base_probe = probe_with_ablation(model, tokenizer, device, task_prompt, patches, None, None)
                    base_metric = make_metric_row(
                        tokenizer, base_probe, prefix_id, old_wrong_prefix_id, value_prefix_ids, args.top_k
                    )
                    base_row = {
                        "sample_idx": item["sample_idx"],
                        "item_idx": item_i,
                        "pair_task": task,
                        "site": site["name"],
                        "kind": "site_restore_baseline",
                        "layer": None,
                        "component": None,
                        "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                        "prefix_id": prefix_id,
                        "prefix_text": tokenizer.decode([prefix_id]),
                        "old_wrong_prefix_id": old_wrong_prefix_id,
                        "old_wrong_prefix_text": tokenizer.decode([old_wrong_prefix_id]),
                        **base_metric,
                    }
                    rows.append(base_row)
                    if len(examples) < args.example_limit:
                        examples.append(base_row)

                    for li in scan_layers:
                        for component in SCAN_COMPONENTS:
                            probe = probe_with_ablation(model, tokenizer, device, task_prompt, patches, li, component)
                            metric = make_metric_row(
                                tokenizer, probe, prefix_id, old_wrong_prefix_id, value_prefix_ids, args.top_k
                            )
                            row = {
                                "sample_idx": item["sample_idx"],
                                "item_idx": item_i,
                                "pair_task": task,
                                "site": site["name"],
                                "kind": "component_ablation",
                                "layer": li,
                                "component": component,
                                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                                "prefix_id": prefix_id,
                                "prefix_text": tokenizer.decode([prefix_id]),
                                "old_wrong_prefix_id": old_wrong_prefix_id,
                                "old_wrong_prefix_text": tokenizer.decode([old_wrong_prefix_id]),
                                **metric,
                            }
                            rows.append(row)
                            if len(examples) < args.example_limit:
                                examples.append(row)

        summary = summarize(rows)
        log("Top format-prior writer candidates:")
        for r in summary["format_prior_writer_candidates"][:18]:
            log(
                f"  {r['pair_task']} {r['site']} L{r['layer']:02d} {r['component']} "
                f"base_top={r['baseline_top0_category']} n={r['n']} "
                f"dTop={r['mean_top_margin_delta']:.3f} dRank={r['mean_rank_improvement']:.2f} "
                f"flip={r['flipped_to_correct']}"
            )
        return {
            "phase": 656,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "scan_layers": scan_layers,
            "scan_components": SCAN_COMPONENTS,
            "site_specs": site_specs,
            "tasks": TASK_ORDER,
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


def parse_layers(text: str) -> List[int]:
    if not text:
        return []
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--layers", type=parse_layers, default=[])
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--save-rows", action="store_true")
    parser.add_argument("--example-limit", type=int, default=320)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 12
        args.max_cases = 1
        args.top_k = min(args.top_k, 20)
        args.layers = args.layers or ([14, 15] if args.model == "qwen3" else [21, 22])
        args.example_limit = 120
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 40)
        args.max_samples = max(args.max_samples, 320)
        args.max_cases = max(args.max_cases, 20)
        args.top_k = max(args.top_k, 30)
        args.example_limit = max(args.example_limit, 360)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase656_{args.model}_format_prior_writer_localization_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
