#!/usr/bin/env python3
"""
Phase 592: Relation-Specific Ranking Factor Atlas
关系特异排序因子图谱

This phase stops blind patch search and builds atlas-ready measurements:
where does prompt-level repair first create candidate-specific ranking?
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, compute_full_string_logprob_batch, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402


OUT_ROOT = Path("results/glm5_phase592_relation_specific_ranking_atlas")
POSITIONS = ["query_relation", "query_category", "rule_relation", "rule_value", "prompt_last"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_map(model, tokenizer, device, prompt: str, candidates: List[str]) -> Dict[str, float]:
    return {k: v[0] for k, v in compute_full_string_logprob_batch(model, tokenizer, device, prompt, candidates).items()}


def answer_vectors(model, tokenizer, candidates: List[str]) -> Dict[str, torch.Tensor]:
    emb = model.get_output_embeddings().weight.detach().float().cpu()
    vecs = {}
    for answer in candidates:
        ids = tokenizer.encode(" " + answer, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(answer, add_special_tokens=False)
        vecs[answer] = emb[ids].mean(dim=0)
    return vecs


def get_hidden_all(model, tokenizer, device, prompt: str) -> List[torch.Tensor]:
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
    return [h[0].detach().float().cpu() for h in out.hidden_states]


def projection_delta(delta: torch.Tensor, vecs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(torch.dot(delta, v)) for k, v in vecs.items()}


def specific_metric(scores: Dict[str, float], correct: str, old_top_wrong: str) -> Dict:
    common = sum(scores.values()) / max(1, len(scores))
    correct_specific = scores[correct] - common
    old_top_wrong_specific = scores[old_top_wrong] - common
    return {
        "common": common,
        "correct_delta": scores[correct],
        "old_top_wrong_delta": scores[old_top_wrong],
        "correct_specific": correct_specific,
        "old_top_wrong_specific": old_top_wrong_specific,
        "specific_margin": correct_specific - old_top_wrong_specific,
        "scores": scores,
    }


def layer_bucket(layer: int, n_layers: int) -> str:
    frac = layer / max(1, n_layers - 1)
    if frac < 0.25:
        return "early"
    if frac < 0.5:
        return "mid"
    if frac < 0.75:
        return "late_mid"
    return "late"


def first_crossing(layer_items: List[Dict], threshold: float) -> Optional[Dict]:
    for item in sorted(layer_items, key=lambda x: x["layer"]):
        if item["mean_specific_margin"] >= threshold and item["mean_correct_specific"] > 0:
            return item
    return None


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        vecs = answer_vectors(model, tokenizer, values)
        cases = list(build_cases(args.n_tables, args.max_samples))
        log(f"{args.model}: n_layers={info.n_layers}, cases={len(cases)}, positions={POSITIONS}")

        rows = []
        accum = defaultdict(lambda: {
            "n": 0,
            "common": 0.0,
            "correct_delta": 0.0,
            "old_top_wrong_delta": 0.0,
            "correct_specific": 0.0,
            "old_top_wrong_specific": 0.0,
            "specific_margin": 0.0,
            "positive_specific": 0,
        })

        target_n = 0
        for si, case in enumerate(cases):
            correct = case["correct"]
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
            base = winner_stats(base_scores, correct)
            repair = winner_stats(repair_scores, correct)
            target_case = (not base["correct"]) and repair["correct"]
            if not target_case and args.target_only:
                continue
            if target_case:
                target_n += 1

            base_pos = case_positions(tokenizer, case, case["base_prompt"], case["relation"])
            repair_pos = case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"])
            base_h = get_hidden_all(model, tokenizer, device, case["base_prompt"])
            repair_h = get_hidden_all(model, tokenizer, device, case["repair_prompt"])
            old_top_wrong = base["top_wrong"]

            case_row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "positions": {},
            }
            for pos_name in POSITIONS:
                bp = base_pos.get(pos_name)
                rp = repair_pos.get(pos_name)
                if bp is None or rp is None:
                    continue
                if bp < 0 or rp < 0:
                    continue
                pos_layers = []
                for li in range(info.n_layers):
                    hs_idx = li + 1
                    if hs_idx >= len(base_h) or bp >= base_h[hs_idx].shape[0] or rp >= repair_h[hs_idx].shape[0]:
                        continue
                    delta = repair_h[hs_idx][rp] - base_h[hs_idx][bp]
                    metric = specific_metric(projection_delta(delta, vecs), correct, old_top_wrong)
                    metric.update({"layer": li, "bucket": layer_bucket(li, info.n_layers)})
                    pos_layers.append(metric)
                    key = (pos_name, li)
                    acc = accum[key]
                    acc["n"] += 1
                    for m in [
                        "common",
                        "correct_delta",
                        "old_top_wrong_delta",
                        "correct_specific",
                        "old_top_wrong_specific",
                        "specific_margin",
                    ]:
                        acc[m] += metric[m]
                    acc["positive_specific"] += int(metric["correct_specific"] > metric["old_top_wrong_specific"])
                case_row["positions"][pos_name] = pos_layers
            rows.append(case_row)

        summary_rows = []
        for (pos_name, li), acc in accum.items():
            n = max(1, acc["n"])
            summary_rows.append({
                "position": pos_name,
                "layer": li,
                "bucket": layer_bucket(li, info.n_layers),
                "n": acc["n"],
                "mean_common": acc["common"] / n,
                "mean_correct_delta": acc["correct_delta"] / n,
                "mean_old_top_wrong_delta": acc["old_top_wrong_delta"] / n,
                "mean_correct_specific": acc["correct_specific"] / n,
                "mean_old_top_wrong_specific": acc["old_top_wrong_specific"] / n,
                "mean_specific_margin": acc["specific_margin"] / n,
                "positive_specific_rate": acc["positive_specific"] / n,
            })
        summary_rows.sort(key=lambda r: (r["mean_specific_margin"], r["mean_correct_specific"]), reverse=True)
        by_pos = defaultdict(list)
        for r in summary_rows:
            by_pos[r["position"]].append(r)
        threshold = args.cross_threshold
        first_by_pos = {
            pos: first_crossing(items, threshold)
            for pos, items in by_pos.items()
        }
        best = summary_rows[:20]
        for item in best[:12]:
            log(
                f"  {item['position']} L{item['layer']}: "
                f"spec_margin={item['mean_specific_margin']:.3f}, "
                f"cspec={item['mean_correct_specific']:.3f}, "
                f"wspec={item['mean_old_top_wrong_specific']:.3f}, "
                f"common={item['mean_common']:.3f}, pos_rate={item['positive_specific_rate']:.2f}"
            )

        atlas = build_atlas(args.model, info.n_layers, len(cases), target_n, summary_rows, first_by_pos)
        return {
            "phase": 592,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "n_cases": len(cases),
            "target_n": target_n,
            "positions": POSITIONS,
            "cross_threshold": threshold,
            "summary": {
                "best": best,
                "first_crossing_by_position": first_by_pos,
                "bucket_counts_top20": dict(Counter(r["bucket"] for r in best)),
            },
            "atlas": atlas,
            "rows": rows,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_atlas(model: str, n_layers: int, n_cases: int, target_n: int,
                summary_rows: List[Dict], first_by_pos: Dict[str, Optional[Dict]]) -> Dict:
    nodes = []
    edges = []
    for item in summary_rows[:60]:
        role = "candidate_specific_ranking" if item["mean_specific_margin"] > 0 else "candidate_common_or_noise"
        node_id = f"{model}_L{item['layer']}_{item['position']}_ranking_projection"
        nodes.append({
            "node_id": node_id,
            "node_type": "component_state_projection",
            "model": model,
            "layer": item["layer"],
            "layer_bucket": item["bucket"],
            "position": item["position"],
            "component": "residual_hidden_state",
            "role": role,
            "causal_level": 2,
            "metrics": item,
        })
        edges.append({
            "source": "relation_filter_prompt",
            "target": node_id,
            "edge_type": "induces_projection_delta",
            "causal_level": 2,
            "effect": {
                "mean_common": item["mean_common"],
                "mean_correct_specific": item["mean_correct_specific"],
                "mean_old_top_wrong_specific": item["mean_old_top_wrong_specific"],
                "mean_specific_margin": item["mean_specific_margin"],
            },
        })
    for pos, item in first_by_pos.items():
        if item is None:
            continue
        edges.append({
            "source": f"{model}_{pos}_first_specific_crossing",
            "target": f"{model}_candidate_specific_ranking_factor",
            "edge_type": "candidate_first_detected_at",
            "causal_level": 2,
            "effect": item,
        })
    return {
        "graph_type": "mechanism_atlas_slice",
        "phase": 592,
        "model": model,
        "n_layers": n_layers,
        "n_cases": n_cases,
        "target_n": target_n,
        "nodes": nodes,
        "edges": edges,
        "schema": {
            "causal_level": {
                "2": "decodable projection, not causal repair",
            },
            "roles": ["candidate_specific_ranking", "candidate_common_or_noise"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--cross-threshold", type=float, default=0.25)
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 4
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
    out_path = out_dir / f"phase592_{args.model}_relation_specific_ranking_atlas_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")

    if args.hard_exit_after_model:
        import os

        os._exit(0)


if __name__ == "__main__":
    main()
