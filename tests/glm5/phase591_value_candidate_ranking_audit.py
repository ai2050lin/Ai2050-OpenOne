#!/usr/bin/env python3
"""
Phase 591: Value Candidate Internal Ranking Audit
值候选内部排序审计

Phase 590 showed that stronger hidden patches mostly activate the value candidate
set and do not reliably change the internal winner. This phase records the full
candidate delta matrix and audits why the old top-wrong candidate stays ahead.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, compute_full_string_logprob_batch, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions, random_same_norm, selected_layers  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase589_component_value_path_attribution import collect_component_outputs  # noqa: E402
from phase590_value_winner_multilayer_patch import (  # noqa: E402
    build_patch_vectors,
    patched_score_map_multi,
)


OUT_ROOT = Path("results/glm5_phase591_value_candidate_ranking_audit")
RULE_RE = re.compile(r"^(c\d+)\s+(r\d+)\s+(v\d+)\.$")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_map(model, tokenizer, device, prompt: str, candidates: List[str]) -> Dict[str, float]:
    return {k: v[0] for k, v in compute_full_string_logprob_batch(model, tokenizer, device, prompt, candidates).items()}


def parse_crv_rules(prompt: str) -> List[Dict]:
    rules = []
    for line in prompt.splitlines():
        m = RULE_RE.match(line.strip())
        if m:
            cat, rel, val = m.groups()
            rules.append({"category": cat, "relation": rel, "value": val, "index": len(rules)})
    return rules


def candidate_rule_features(case: Dict, candidate: str, base_prompt: str, wrong_rel: str) -> Dict:
    rules = parse_crv_rules(base_prompt)
    cat = case["category"]
    rel = case["relation"]
    hits = [r for r in rules if r["value"] == candidate]
    true_pair = any(r["category"] == cat and r["relation"] == rel for r in hits)
    same_cat_wrong_rel = any(r["category"] == cat and r["relation"] != rel for r in hits)
    wrong_rel_same_cat = any(r["category"] == cat and r["relation"] == wrong_rel for r in hits)
    same_rel_other_cat = any(r["category"] != cat and r["relation"] == rel for r in hits)
    wrong_rel_any = any(r["relation"] == wrong_rel for r in hits)
    first_index = min((r["index"] for r in hits), default=None)
    last_index = max((r["index"] for r in hits), default=None)
    return {
        "rule_occurrences": len(hits),
        "first_rule_index": first_index,
        "last_rule_index": last_index,
        "true_pair": true_pair,
        "same_category_wrong_relation": same_cat_wrong_rel,
        "wrong_relation_same_category": wrong_rel_same_cat,
        "same_relation_other_category": same_rel_other_cat,
        "wrong_relation_any_category": wrong_rel_any,
    }


def embedding_cosines(model, tokenizer, correct: str, candidates: List[str]) -> Dict[str, float]:
    emb = model.get_output_embeddings().weight.detach().float().cpu()

    def answer_vec(answer: str) -> torch.Tensor:
        ids = tokenizer.encode(" " + answer, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(answer, add_special_tokens=False)
        return emb[ids].mean(dim=0)

    c = answer_vec(correct)
    result = {}
    for cand in candidates:
        v = answer_vec(cand)
        result[cand] = float(F.cosine_similarity(c[None, :], v[None, :]).item())
    return result


def get_patch_combo(probe_layers: List[int]) -> List[Tuple[str, int]]:
    mid, late = probe_layers[-2], probe_layers[-1]
    return [("residual", mid), ("residual", late), ("attn", mid), ("attn", late)]


def candidate_delta_records(base_scores: Dict[str, float], mode_scores: Dict[str, float],
                            correct: str, old_top_wrong: str, features: Dict[str, Dict],
                            prior_scores: Dict[str, float], cosines: Dict[str, float]) -> Tuple[List[Dict], Dict]:
    values = list(base_scores)
    deltas = {v: mode_scores[v] - base_scores[v] for v in values}
    common = sum(deltas.values()) / max(1, len(deltas))
    records = []
    for val in values:
        records.append({
            "value": val,
            "is_correct": val == correct,
            "is_old_top_wrong": val == old_top_wrong,
            "base_score": base_scores[val],
            "mode_score": mode_scores[val],
            "delta": deltas[val],
            "specific_delta": deltas[val] - common,
            "prior_score": prior_scores[val],
            "embedding_cosine_to_correct": cosines[val],
            "features": features[val],
        })
    metric = {
        "common_delta": common,
        "correct_delta": deltas[correct],
        "old_top_wrong_delta": deltas[old_top_wrong],
        "correct_specific": deltas[correct] - common,
        "old_top_wrong_specific": deltas[old_top_wrong] - common,
        "margin_gain_vs_old_top_wrong": deltas[correct] - deltas[old_top_wrong],
    }
    return records, metric


def source_labels(feature: Dict, top_wrong_prior_gt_correct: bool) -> List[str]:
    labels = []
    if feature["same_category_wrong_relation"]:
        labels.append("same_category_wrong_relation")
    if feature["same_relation_other_category"]:
        labels.append("same_relation_other_category")
    if feature["wrong_relation_any_category"]:
        labels.append("wrong_relation_any_category")
    if feature["rule_occurrences"] > 1:
        labels.append("repeated_value")
    if top_wrong_prior_gt_correct:
        labels.append("value_prior_higher_than_correct")
    if not labels:
        labels.append("unclassified")
    return labels


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        all_layers = selected_layers(info.n_layers)
        probe_layers = [all_layers[-2], all_layers[-1]]
        combo = get_patch_combo(probe_layers)
        components = ["residual", "attn"]
        cases = list(build_cases(args.n_tables, args.max_samples))
        values = CANDIDATE_VALUES[:4]
        prior_scores = score_map(model, tokenizer, device, "Answer:", values)
        log(f"{args.model}: n_layers={info.n_layers}, cases={len(cases)}, layers={probe_layers}")

        rows = []
        for si, case in enumerate(cases):
            correct = case["correct"]
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
            wrong_scores = score_map(model, tokenizer, device, case["wrong_prompt"], values)
            base = winner_stats(base_scores, correct)
            repair = winner_stats(repair_scores, correct)
            wrong = winner_stats(wrong_scores, correct)
            target_case = (not base["correct"]) and repair["correct"]

            base_pos = case_positions(tokenizer, case, case["base_prompt"], case["relation"])
            repair_pos = case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"])
            base_out = collect_component_outputs(model, tokenizer, device, case["base_prompt"], probe_layers, components)
            repair_out = collect_component_outputs(model, tokenizer, device, case["repair_prompt"], probe_layers, components)
            patch_scores = {}
            for pos_name in ["prompt_last", "query_relation"]:
                patches = build_patch_vectors(
                    combo, pos_name, base_pos, repair_pos, base_out, repair_out, "repair", si * 101
                )
                if patches:
                    patch_scores[f"patch_{pos_name}_residual_attn"] = patched_score_map_multi(
                        model, tokenizer, device, case["base_prompt"], values, patches, args.alpha
                    )
                    random_patches = []
                    for comp, li, bp, delta in patches:
                        random_patches.append((comp, li, bp, random_same_norm(delta, seed=si * 1009 + li)))
                    patch_scores[f"random_{pos_name}_residual_attn"] = patched_score_map_multi(
                        model, tokenizer, device, case["base_prompt"], values, random_patches, args.alpha
                    )

            features = {
                val: candidate_rule_features(case, val, case["base_prompt"], case["wrong_rel"])
                for val in values
            }
            cosines = embedding_cosines(model, tokenizer, correct, values)
            mode_scores = {
                "repair_prompt": repair_scores,
                "wrong_relation_prompt": wrong_scores,
                **patch_scores,
            }
            modes = {}
            for mode, scores in mode_scores.items():
                records, metric = candidate_delta_records(
                    base_scores, scores, correct, base["top_wrong"], features, prior_scores, cosines
                )
                modes[mode] = {
                    "winner": winner_stats(scores, correct),
                    "metric": metric,
                    "candidate_records": records,
                }

            top_wrong = base["top_wrong"]
            top_wrong_feature = features[top_wrong]
            labels = source_labels(top_wrong_feature, prior_scores[top_wrong] > prior_scores[correct])
            rows.append({
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "wrong_prompt": wrong,
                "top_wrong_attribution": {
                    "top_wrong": top_wrong,
                    "labels": labels,
                    "features": top_wrong_feature,
                    "prior_score": prior_scores[top_wrong],
                    "correct_prior_score": prior_scores[correct],
                    "embedding_cosine_to_correct": cosines[top_wrong],
                },
                "modes": modes,
            })

        summary = summarize(rows)
        return {
            "phase": 591,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "probe_layers": probe_layers,
            "n_cases": len(cases),
            "alpha": args.alpha,
            "prior_scores": prior_scores,
            "summary": summary,
            "rows": rows,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summarize(rows: List[Dict]) -> Dict:
    target_rows = [r for r in rows if r["target_case"]]
    modes = sorted({m for r in rows for m in r["modes"]})
    by_mode = {}
    for mode in modes:
        items = [r for r in target_rows if mode in r["modes"]]
        entry = {
            "target_n": len(items),
            "switch": 0,
            "mean_common_delta": 0.0,
            "mean_correct_delta": 0.0,
            "mean_old_top_wrong_delta": 0.0,
            "mean_correct_specific": 0.0,
            "mean_old_top_wrong_specific": 0.0,
            "mean_margin_gain_vs_old_top_wrong": 0.0,
            "correct_specific_gt_old_top_wrong": 0,
        }
        for r in items:
            m = r["modes"][mode]
            metric = m["metric"]
            entry["switch"] += int(m["winner"]["correct"])
            for k in [
                "common_delta",
                "correct_delta",
                "old_top_wrong_delta",
                "correct_specific",
                "old_top_wrong_specific",
                "margin_gain_vs_old_top_wrong",
            ]:
                entry[f"mean_{k}"] += metric[k]
            entry["correct_specific_gt_old_top_wrong"] += int(
                metric["correct_specific"] > metric["old_top_wrong_specific"]
            )
        denom = max(1, len(items))
        for k in list(entry):
            if k.startswith("mean_"):
                entry[k] /= denom
        by_mode[mode] = entry

    label_counts = Counter()
    target_label_counts = Counter()
    top_wrong_values = Counter()
    target_top_wrong_values = Counter()
    cosine_sum = 0.0
    cosine_n = 0
    for r in rows:
        labels = r["top_wrong_attribution"]["labels"]
        for label in labels:
            label_counts[label] += 1
            if r["target_case"]:
                target_label_counts[label] += 1
        top_wrong_values[r["top_wrong_attribution"]["top_wrong"]] += 1
        if r["target_case"]:
            target_top_wrong_values[r["top_wrong_attribution"]["top_wrong"]] += 1
            cosine_sum += r["top_wrong_attribution"]["embedding_cosine_to_correct"]
            cosine_n += 1

    best = sorted(
        by_mode.items(),
        key=lambda kv: (
            kv[1]["target_n"],
            kv[1]["switch"],
            kv[1]["mean_margin_gain_vs_old_top_wrong"],
            kv[1]["mean_correct_specific"],
        ),
        reverse=True,
    )
    for mode, item in best[:10]:
        log(
            f"  {mode}: target={item['target_n']}, switch={item['switch']}/{item['target_n']}, "
            f"common={item['mean_common_delta']:.3f}, "
            f"cd={item['mean_correct_delta']:.3f}, wd={item['mean_old_top_wrong_delta']:.3f}, "
            f"cspec={item['mean_correct_specific']:.3f}, wspec={item['mean_old_top_wrong_specific']:.3f}, "
            f"mgain={item['mean_margin_gain_vs_old_top_wrong']:.3f}"
        )

    return {
        "n": len(rows),
        "target_n": len(target_rows),
        "by_mode": by_mode,
        "top_wrong_label_counts": dict(label_counts),
        "target_top_wrong_label_counts": dict(target_label_counts),
        "top_wrong_value_counts": dict(top_wrong_values),
        "target_top_wrong_value_counts": dict(target_top_wrong_values),
        "target_mean_top_wrong_embedding_cosine_to_correct": cosine_sum / max(1, cosine_n),
        "best_modes": [{"mode": mode, **item} for mode, item in best[:10]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 3
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 8)
        args.max_samples = max(args.max_samples, 64)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase591_{args.model}_value_candidate_ranking_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    import os

    main()
