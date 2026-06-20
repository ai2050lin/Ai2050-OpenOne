#!/usr/bin/env python3
"""
Phase 551: minimal layer-combination circuit audit for the clean paraphrase gate.

This phase keeps the Phase548 scoring surface, but replaces the fixed 3-layer
window with single-layer, pair-layer, and all-layer interventions.  The goal is
to determine whether the clean paraphrase effect is carried by one layer or by
a cumulative multi-layer field.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_W_U, get_layers, get_model_info, release_model  # noqa: E402
from phase530_state_pair_decomposition import hidden_at_layer, load_model_bf16_flash, mean_dir  # noqa: E402
from phase536_pair_quality_selectivity import TEMPLATES  # noqa: E402
from phase539_interface_cluster_mechanism import PAIR_SPECS, layer_windows  # noqa: E402
import phase544_natural_decode_policy_gate_audit as p544  # noqa: E402
import phase545_sampling_stability_cross_category as p545  # noqa: E402
import phase548_paraphrase_candidate_robustness as p548  # noqa: E402


OUT_ROOT = Path("results/glm5_phase551_paraphrase_minimal_layer_circuit")
DEFAULT_PAIR = "vehicle_tool"
DEFAULT_SCAFFOLDS = ["forbidden_sentence_completion", "forbidden_natural_qa", "forbidden_definition"]
DEFAULT_MODES = ["temperature", "top_p"]
DEFAULT_COMPONENTS = ["baseline", "residual_perp", "residual_full", "residual_parallel", "random_perp", "random_full"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def combo_layers(window: list[int], mode: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    modes = set(parse_csv(mode))
    if "single" in modes:
        for layer_id in window:
            out[f"L{layer_id}"] = [layer_id]
    if "pairs" in modes:
        for a, b in itertools.combinations(window, 2):
            out[f"L{a}+L{b}"] = [a, b]
    if "all" in modes:
        out["all"] = list(window)
    return out


def build_components_by_layer(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    pair: str,
    layers_to_collect: list[int],
    train_n: int,
    batch_size: int,
    max_length: int,
    W_U: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    candidates = p548.build_candidates(pair, train_n)
    components_by_layer: dict[str, dict[str, np.ndarray]] = {}
    for layer_id in layers_to_collect:
        log(f"  collect L{layer_id}")
        dirs = {}
        for name, meta in candidates.items():
            pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, batch_size, max_length)
            neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, batch_size, max_length)
            dirs[name] = mean_dir(pos_h, neg_h)
        components_by_layer[str(layer_id)] = p548.build_components(pair, dirs, W_U, tokenizer, layer_id)
    return components_by_layer


def interventions_for_subset(
    components_by_layer: dict[str, dict[str, np.ndarray]],
    layers: list[int],
    component: str,
    alpha: float,
) -> dict[int, tuple[np.ndarray, float]] | None:
    if component == "baseline":
        return None
    return {layer_id: (components_by_layer[str(layer_id)][component], alpha) for layer_id in layers}


def compact_row(agg: dict[str, Any], base: dict[str, Any], random_best: dict[str, Any]) -> dict[str, float | str]:
    clean_gain = agg["clean_non_object_rate"] - base["clean_non_object_rate"]
    label_gain = agg["any_label_violation_rate"] - base["any_label_violation_rate"]
    score_gain = agg["clean_non_object_score"] - base["clean_non_object_score"]
    random_gain = random_best["clean_non_object_rate"] - base["clean_non_object_rate"]
    above_random = clean_gain - random_gain
    if clean_gain >= 0.15 and score_gain >= 0.10 and label_gain <= 0.05 and above_random >= 0.08:
        cls = "robust_clean_positive"
    elif clean_gain >= 0.10 and score_gain >= 0.05 and label_gain <= 0.05 and above_random >= 0.04:
        cls = "partial_clean_positive"
    elif label_gain >= 0.12:
        cls = "label_leak"
    elif clean_gain >= 0.06:
        cls = "weak_clean"
    elif score_gain <= -0.08:
        cls = "negative"
    else:
        cls = "flat"
    return {
        "clean_gain": float(clean_gain),
        "label_gain": float(label_gain),
        "score_gain": float(score_gain),
        "random_gain": float(random_gain),
        "above_random": float(above_random),
        "class": cls,
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pair = args.pair
    scaffolds = parse_csv(args.scaffolds)
    modes = parse_csv(args.decode_modes)
    components = parse_csv(args.components)
    sample_seeds = parse_int_csv(args.sample_seeds)
    alpha = float(args.alpha)

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        if len(windows) != 1:
            raise ValueError(f"Phase551 expects one window, got {windows}")
        window_name, window = next(iter(windows.items()))
        layer_combos = combo_layers(window, args.layer_combo_modes)
        all_layers = sorted(set(x for vals in layer_combos.values() for x in vals))
        W_U = get_W_U(model, args.model).astype(np.float32)
        groups = p544.token_groups(tokenizer, pair)
        prompt_sets = p548.build_prompts(pair, args.test_n, scaffolds)
        components_by_layer = build_components_by_layer(
            model, tokenizer, device, pair, all_layers, args.train_n, args.batch_size, args.max_length, W_U
        )
        log(f"{args.model}: phase551 pair={pair}, window={window}, combos={list(layer_combos)}")

        audit: dict[str, Any] = {}
        saved_samples: list[dict[str, Any]] = []
        all_tsv: list[dict[str, Any]] = []
        for combo_name, combo in layer_combos.items():
            audit[combo_name] = {"layers": combo, "scaffolds": {}}
            for scaffold, prompt_rows in prompt_sets.items():
                audit[combo_name]["scaffolds"][scaffold] = {}
                for mode in modes:
                    audit[combo_name]["scaffolds"][scaffold][mode] = {}
                    for component in components:
                        all_records = []
                        seed_rows = []
                        for seed in sample_seeds:
                            agg, records = p548.decode_and_classify(
                                model, tokenizer, device, layers, prompt_rows,
                                interventions_for_subset(components_by_layer, combo, component, alpha),
                                groups, pair, mode, args.max_new_tokens, args.batch_size,
                                args.max_length, seed, args.temperature, args.top_p,
                            )
                            seed_rows.append({"seed": seed, **agg})
                            for rec in records:
                                rec2 = {
                                    "combo": combo_name,
                                    "layers": combo,
                                    "pair": pair,
                                    "scaffold": scaffold,
                                    "mode": mode,
                                    "component": component,
                                    "seed": seed,
                                    **rec,
                                }
                                all_records.append(rec2)
                        row = p548.aggregate(all_records)
                        row["seed_aggregates"] = seed_rows
                        audit[combo_name]["scaffolds"][scaffold][mode][component] = row
                        saved_samples.extend(all_records[: args.samples_per_row])
                        all_tsv.extend(all_records)
                    base = audit[combo_name]["scaffolds"][scaffold][mode]["baseline"]
                    rp = audit[combo_name]["scaffolds"][scaffold][mode].get("residual_perp", base)
                    rf = audit[combo_name]["scaffolds"][scaffold][mode].get("residual_full", base)
                    rnd = audit[combo_name]["scaffolds"][scaffold][mode].get("random_perp", base)
                    log(
                        f"    {combo_name} {scaffold} {mode}: "
                        f"base={base['clean_non_object_rate']:.2f}; "
                        f"perp={rp['clean_non_object_rate']:.2f} label={rp['any_label_violation_rate']:.2f}; "
                        f"full={rf['clean_non_object_rate']:.2f} label={rf['any_label_violation_rate']:.2f}; "
                        f"rand={rnd['clean_non_object_rate']:.2f}"
                    )

        compact = []
        for combo_name, combo_data in audit.items():
            for scaffold in scaffolds:
                for mode in modes:
                    rows = combo_data["scaffolds"][scaffold][mode]
                    base = rows["baseline"]
                    random_rows = [rows[x] for x in ("random_perp", "random_full") if x in rows]
                    random_best = max(random_rows, key=lambda r: r["clean_non_object_rate"]) if random_rows else base
                    for component, row in rows.items():
                        if component == "baseline":
                            continue
                        compact.append({
                            "combo": combo_name,
                            "layers": combo_data["layers"],
                            "scaffold": scaffold,
                            "mode": mode,
                            "component": component,
                            "base_clean_non_object_rate": base["clean_non_object_rate"],
                            "clean_non_object_rate": row["clean_non_object_rate"],
                            "base_label_violation_rate": base["any_label_violation_rate"],
                            "label_violation_rate": row["any_label_violation_rate"],
                            "object_echo_rate": row["object_echo_rate"],
                            "prompt_echo_rate": row["prompt_echo_rate"],
                            "clean_non_object_score": row["clean_non_object_score"],
                            **compact_row(row, base, random_best),
                        })

        return {
            "phase": 551,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "pair": pair,
            "window_name": window_name,
            "window": window,
            "layer_combos": layer_combos,
            "components": components,
            "scaffolds": scaffolds,
            "decode_modes": modes,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "sample_seeds": sample_seeds,
            "alpha": alpha,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "pair_spec": PAIR_SPECS[pair],
            "audit": audit,
            "compact_rows": compact,
            "sample_records": saved_samples[: args.max_saved_samples],
            "all_records_for_tsv": all_tsv[: args.max_tsv_records],
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_tsv(result: dict[str, Any], out_dir: Path, model_name: str) -> None:
    path = out_dir / f"phase551_{model_name}_readable_samples.tsv"
    fields = [
        "combo", "layers", "pair", "scaffold", "mode", "component", "seed", "object", "quality",
        "clean_non_object", "any_label_violation", "object_echo", "prompt_echo",
        "target_non_object_matches", "target_label_matches", "competitor_synonym_matches",
        "generated_suffix",
    ]
    lines = ["\t".join(fields)]
    for rec in result.get("all_records_for_tsv", []):
        vals = []
        for field in fields:
            val = rec.get(field, "")
            if isinstance(val, list):
                val = ",".join(str(x) for x in val)
            vals.append(str(val).replace("\t", " ").replace("\n", " "))
        lines.append("\t".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--windows", default=None)
    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--test-n", type=int, default=12)
    parser.add_argument("--sample-seeds", default="101,103,107,109,113,127,131,137")
    parser.add_argument("--scaffolds", default=",".join(DEFAULT_SCAFFOLDS))
    parser.add_argument("--decode-modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--components", default=",".join(DEFAULT_COMPONENTS))
    parser.add_argument("--layer-combo-modes", default="single,pairs,all")
    parser.add_argument("--alpha", type=float, default=6.0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--samples-per-row", type=int, default=2)
    parser.add_argument("--max-saved-samples", type=int, default=1200)
    parser.add_argument("--max-tsv-records", type=int, default=8000)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase551_{args.model}_paraphrase_minimal_layer_circuit.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_tsv(result, out_dir, args.model)
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
