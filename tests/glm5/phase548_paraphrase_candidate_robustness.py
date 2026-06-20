#!/usr/bin/env python3
"""
Phase 548: paraphrase candidate robustness and readable sample audit.

This phase stress-tests the Phase547 vehicle_tool clean paraphrase candidate
with heldout objects, random same-norm controls, matched-term logging, and
object/prompt echo checks.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_W_U, get_layers, get_model_info, release_model  # noqa: E402
from phase530_state_pair_decomposition import hidden_at_layer, load_model_bf16_flash, mean_dir  # noqa: E402
from phase532_multi_seed_controls import normalize  # noqa: E402
from phase536_pair_quality_selectivity import CATEGORY_BANK, TEMPLATES, cat_prompt  # noqa: E402
from phase539_interface_cluster_mechanism import PAIR_SPECS, layer_windows, readout_direction  # noqa: E402
import phase544_natural_decode_policy_gate_audit as p544  # noqa: E402
import phase545_sampling_stability_cross_category as p545  # noqa: E402
from phase546_semantic_quality_decomposition import GENERIC_TERMS, contains_term, exact_terms  # noqa: E402
from phase547_label_forbidden_paraphrase_gate import SYNONYM_TERMS, forbidden_prompt  # noqa: E402


OUT_ROOT = Path("results/glm5_phase548_paraphrase_candidate_robustness")
DEFAULT_PAIR = "vehicle_tool"
DEFAULT_SCAFFOLDS = ["forbidden_definition", "forbidden_sentence_completion", "forbidden_natural_qa"]
DEFAULT_MODES = ["top_p", "temperature"]
DEFAULT_CONDITIONS = [
    "baseline",
    "residual_parallel",
    "residual_full",
    "residual_perp",
    "readout",
    "random_full",
    "random_perp",
]
PROMPT_ECHO_TERMS = ["do not use", "without using", "these words", "answer naturally", "complete the sentence"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def has_any(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(contains_term(low, term) for term in terms)


def matched_terms(text: str, terms: list[str]) -> list[str]:
    low = text.lower()
    return [term for term in terms if contains_term(low, term)]


def all_family_terms(label: str) -> list[str]:
    base = list(p544.FAMILY_TERMS.get(label, [label]))
    extra = SYNONYM_TERMS.get(label, [])
    seen, out = set(), []
    for term in base + extra:
        term = term.strip().lower()
        if term and term not in seen:
            seen.add(term)
            out.append(term)
    return out


def synonym_terms(label: str) -> list[str]:
    exact = set(exact_terms(label))
    return [x for x in all_family_terms(label) if x not in exact]


def classify_suffix(suffix: str, obj: str, target_label: str, competitor_label: str) -> dict[str, Any]:
    low = suffix.lower()
    target_label_terms = exact_terms(target_label)
    competitor_label_terms = exact_terms(competitor_label)
    target_terms = synonym_terms(target_label)
    competitor_terms = synonym_terms(competitor_label)
    obj_terms = [obj.lower()]
    non_object_target_terms = [x for x in target_terms if x != obj.lower()]

    target_label_matches = matched_terms(low, target_label_terms)
    competitor_label_matches = matched_terms(low, competitor_label_terms)
    target_matches = matched_terms(low, target_terms)
    target_non_object_matches = matched_terms(low, non_object_target_terms)
    competitor_matches = matched_terms(low, competitor_terms)
    generic_matches = matched_terms(low, GENERIC_TERMS)
    prompt_echo_matches = matched_terms(low, PROMPT_ECHO_TERMS)
    object_echo = has_any(low, obj_terms)
    any_label = bool(target_label_matches or competitor_label_matches)
    target_syn = bool(target_matches)
    target_non_object = bool(target_non_object_matches)
    competitor_syn = bool(competitor_matches)
    prompt_echo = bool(prompt_echo_matches)
    degenerate = sum(ch.isalpha() for ch in low) < 2

    if any_label and target_syn:
        quality = "synonym_with_label_violation"
    elif target_syn and competitor_syn:
        quality = "mixed_synonym"
    elif target_syn:
        quality = "clean_synonym"
    elif any_label:
        quality = "label_violation"
    elif competitor_syn:
        quality = "wrong_synonym"
    elif generic_matches:
        quality = "generic_only"
    elif degenerate:
        quality = "degenerate"
    else:
        quality = "other"

    clean_non_object = quality == "clean_synonym" and target_non_object and not prompt_echo
    return {
        "quality": quality,
        "target_label_matches": target_label_matches,
        "competitor_label_matches": competitor_label_matches,
        "target_synonym_matches": target_matches,
        "target_non_object_matches": target_non_object_matches,
        "competitor_synonym_matches": competitor_matches,
        "generic_matches": generic_matches,
        "prompt_echo_matches": prompt_echo_matches,
        "target_label_violation": bool(target_label_matches),
        "competitor_label_violation": bool(competitor_label_matches),
        "any_label_violation": any_label,
        "target_synonym": target_syn,
        "target_non_object_synonym": target_non_object,
        "competitor_synonym": competitor_syn,
        "object_echo": object_echo,
        "prompt_echo": prompt_echo,
        "clean_non_object": clean_non_object,
        "degenerate": degenerate,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(records))
    counts = Counter(r["quality"] for r in records)
    clean = counts["clean_synonym"] / n
    clean_non_object = sum(1 for r in records if r["clean_non_object"]) / n
    label_violation = sum(1 for r in records if r["any_label_violation"]) / n
    wrong = sum(1 for r in records if r["competitor_synonym"] and not r["target_synonym"]) / n
    object_echo = sum(1 for r in records if r["object_echo"]) / n
    prompt_echo = sum(1 for r in records if r["prompt_echo"]) / n
    generic = counts["generic_only"] / n
    return {
        "n": len(records),
        "counts": dict(sorted(counts.items())),
        "rates": {k: float(v / n) for k, v in sorted(counts.items())},
        "clean_synonym_rate": float(clean),
        "clean_non_object_rate": float(clean_non_object),
        "target_synonym_any_rate": float(sum(1 for r in records if r["target_synonym"]) / n),
        "any_label_violation_rate": float(label_violation),
        "wrong_synonym_rate": float(wrong),
        "object_echo_rate": float(object_echo),
        "prompt_echo_rate": float(prompt_echo),
        "generic_only_rate": float(generic),
        "degenerate_rate": float(counts["degenerate"] / n),
        "clean_non_object_score": float(clean_non_object - label_violation - wrong - prompt_echo),
        "sample_records": records[:10],
    }


def build_candidates(pair: str, train_n: int) -> dict[str, dict[str, Any]]:
    pos_label, neg_label = PAIR_SPECS[pair]
    out = {}
    for template in TEMPLATES:
        name = f"{pair}_{template}"
        out[name] = {
            "pair": pair,
            "template": template,
            "pos": [cat_prompt(template, x) for x in CATEGORY_BANK[pos_label][:train_n]],
            "neg": [cat_prompt(template, x) for x in CATEGORY_BANK[neg_label][:train_n]],
        }
    return out


def stable_seed(*parts: str) -> int:
    acc = 17
    for ch in "|".join(parts):
        acc = (acc * 131 + ord(ch)) % 1000003
    return acc


def random_same_norm(dim: int, norm: float, *parts: str) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(*parts))
    vec = rng.normal(size=dim).astype(np.float32)
    return (normalize(vec) * float(norm)).astype(np.float32)


def build_components(
    pair: str,
    dirs: dict[str, np.ndarray],
    W_U: np.ndarray,
    tokenizer: Any,
    layer_id: int,
) -> dict[str, np.ndarray]:
    by_template = {template: dirs[f"{pair}_{template}"] for template in TEMPLATES}
    common_unit = normalize(np.mean([normalize(by_template[t]) for t in TEMPLATES], axis=0).astype(np.float32))
    common_norm = float(np.mean([np.linalg.norm(by_template[t]) for t in TEMPLATES]))
    common_full = (common_unit * common_norm).astype(np.float32)
    readout = readout_direction(W_U, tokenizer, pair)
    dec = p545.decompose(common_full, readout)
    residual_perp = dec["perp"].astype(np.float32)
    residual_parallel = dec["parallel"].astype(np.float32)
    return {
        "residual_full": common_full,
        "residual_perp": residual_perp,
        "residual_parallel": residual_parallel,
        "readout": readout.astype(np.float32),
        "random_full": random_same_norm(common_full.shape[0], np.linalg.norm(common_full), pair, str(layer_id), "random_full"),
        "random_perp": random_same_norm(residual_perp.shape[0], np.linalg.norm(residual_perp), pair, str(layer_id), "random_perp"),
    }


def build_prompts(pair: str, test_n: int, scaffolds: list[str]) -> dict[str, list[dict[str, str]]]:
    pos_label, neg_label = PAIR_SPECS[pair]
    objects = CATEGORY_BANK[pos_label][-test_n:]
    out: dict[str, list[dict[str, str]]] = {}
    for scaffold in scaffolds:
        rows = []
        for obj in objects:
            rows.append({
                "object": obj,
                "prompt": forbidden_prompt(scaffold, obj, pos_label, neg_label),
            })
        out[scaffold] = rows
    return out


def interventions_for(
    components_by_layer: dict[str, dict[str, np.ndarray]],
    pair: str,
    window: list[int],
    condition: str,
    alpha: float,
) -> dict[int, tuple[np.ndarray, float]] | None:
    if condition == "baseline":
        return None
    return {layer_id: (components_by_layer[str(layer_id)][condition], alpha) for layer_id in window}


def decode_and_classify(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt_rows: list[dict[str, str]],
    interventions: dict[int, tuple[np.ndarray, float]] | None,
    groups: dict[str, list[int]],
    pair: str,
    mode: str,
    max_new_tokens: int,
    batch_size: int,
    max_length: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompts = [r["prompt"] for r in prompt_rows]
    prepared = p544.prepare_interventions(interventions)
    rng = np.random.default_rng(seed)
    generated, suffixes, first_types, target_ranks, competitor_ranks = p544.run_linear_decode(
        model, tokenizer, device, layers, prompts, prepared, groups, mode,
        max_new_tokens, batch_size, max_length, rng, temperature, top_p
    )
    pos_label, neg_label = PAIR_SPECS[pair]
    records = []
    for i, (row, suffix, ids, first_type, target_rank, competitor_rank) in enumerate(
        zip(prompt_rows, suffixes, generated, first_types, target_ranks, competitor_ranks)
    ):
        cls = classify_suffix(suffix, row["object"], pos_label, neg_label)
        records.append({
            "prompt_index": i,
            "object": row["object"],
            "prompt": row["prompt"],
            "generated_suffix": suffix,
            "generated_ids": ids,
            "first_type": first_type,
            "first_target_rank": float(target_rank),
            "first_competitor_rank": float(competitor_rank),
            **cls,
        })
    return aggregate(records), records


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pair = args.pair
    scaffolds = parse_csv(args.scaffolds)
    modes = parse_csv(args.decode_modes)
    conditions = parse_csv(args.conditions)
    sample_seeds = parse_int_csv(args.sample_seeds)

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        all_layers = sorted(set(x for vals in windows.values() for x in vals))
        alpha = max(float(x) for x in args.alphas.split(",") if x.strip())
        W_U = get_W_U(model, args.model).astype(np.float32)
        log(f"{args.model}: phase548 pair={pair}, windows={windows}, seeds={sample_seeds}")

        candidates = build_candidates(pair, args.train_n)
        source_prompts = build_prompts(pair, args.test_n, scaffolds)

        components_by_layer = {}
        for layer_id in all_layers:
            log(f"  collect L{layer_id}")
            dirs = {}
            for name, meta in candidates.items():
                pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, args.batch_size, args.max_length)
                neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, args.batch_size, args.max_length)
                dirs[name] = mean_dir(pos_h, neg_h)
            components_by_layer[str(layer_id)] = build_components(pair, dirs, W_U, tokenizer, layer_id)

        groups = p544.token_groups(tokenizer, pair)
        audit = {}
        saved_samples = []
        all_records_for_tsv = []
        for win_name, window in windows.items():
            audit[win_name] = {"window": window, "sources": {pair: {}}}
            for scaffold, prompt_rows in source_prompts.items():
                audit[win_name]["sources"][pair][scaffold] = {}
                for mode in modes:
                    audit[win_name]["sources"][pair][scaffold][mode] = {}
                    for condition in conditions:
                        seed_aggs, all_records = [], []
                        for seed in sample_seeds:
                            agg, records = decode_and_classify(
                                model, tokenizer, device, layers, prompt_rows,
                                interventions_for(components_by_layer, pair, window, condition, alpha),
                                groups, pair, mode, args.max_new_tokens, args.batch_size,
                                args.max_length, seed, args.temperature, args.top_p,
                            )
                            seed_aggs.append({"seed": seed, **agg})
                            for rec in records:
                                rec2 = {
                                    "window": win_name,
                                    "pair": pair,
                                    "scaffold": scaffold,
                                    "mode": mode,
                                    "condition": condition,
                                    "seed": seed,
                                    **rec,
                                }
                                all_records.append(rec2)
                        row = aggregate(all_records)
                        row["seed_aggregates"] = seed_aggs
                        audit[win_name]["sources"][pair][scaffold][mode][condition] = row
                        saved_samples.extend(all_records[: args.samples_per_row])
                        all_records_for_tsv.extend(all_records)
                    base = audit[win_name]["sources"][pair][scaffold][mode]["baseline"]
                    rp = audit[win_name]["sources"][pair][scaffold][mode].get("residual_perp", base)
                    rf = audit[win_name]["sources"][pair][scaffold][mode].get("residual_full", base)
                    rnd = audit[win_name]["sources"][pair][scaffold][mode].get("random_perp", base)
                    log(
                        f"    {win_name} {pair} {scaffold} {mode}: "
                        f"base clean_no={base['clean_non_object_rate']:.2f}; "
                        f"perp={rp['clean_non_object_rate']:.2f} label={rp['any_label_violation_rate']:.2f}; "
                        f"full={rf['clean_non_object_rate']:.2f} label={rf['any_label_violation_rate']:.2f}; "
                        f"rand={rnd['clean_non_object_rate']:.2f}"
                    )

        return {
            "phase": 548,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "pair": pair,
            "conditions": conditions,
            "scaffolds": scaffolds,
            "decode_modes": modes,
            "windows": windows,
            "all_layers": all_layers,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "max_new_tokens": args.max_new_tokens,
            "alpha": alpha,
            "sample_seeds": sample_seeds,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "family_terms": p544.FAMILY_TERMS,
            "synonym_terms": SYNONYM_TERMS,
            "generic_terms": GENERIC_TERMS,
            "prompt_echo_terms": PROMPT_ECHO_TERMS,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "audit": audit,
            "sample_records": saved_samples[: args.max_saved_samples],
            "all_records_for_tsv": all_records_for_tsv[: args.max_tsv_records],
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_tsv(result: dict[str, Any], out_dir: Path, model_name: str) -> None:
    path = out_dir / f"phase548_{model_name}_readable_samples.tsv"
    fields = [
        "window", "pair", "scaffold", "mode", "condition", "seed", "object", "quality",
        "clean_non_object", "any_label_violation", "object_echo", "prompt_echo",
        "target_non_object_matches", "target_label_matches", "competitor_synonym_matches",
        "prompt", "generated_suffix",
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
    parser.add_argument("--alphas", default="6")
    parser.add_argument("--sample-seeds", default="101,103,107,109,113,127,131,137")
    parser.add_argument("--scaffolds", default=",".join(DEFAULT_SCAFFOLDS))
    parser.add_argument("--decode-modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--samples-per-row", type=int, default=4)
    parser.add_argument("--max-saved-samples", type=int, default=1200)
    parser.add_argument("--max-tsv-records", type=int, default=5000)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase548_{args.model}_paraphrase_candidate_robustness.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_tsv(result, out_dir, args.model)
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
