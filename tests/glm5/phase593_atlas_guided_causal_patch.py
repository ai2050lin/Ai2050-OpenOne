#!/usr/bin/env python3
"""
Phase 593: Atlas-Guided Causal Patch Validation
图谱引导因果修补验证

Phase 592 found Level-2 projection nodes for candidate-specific ranking. This
phase patches selected nodes with raw/common/specific/random vectors and tests
whether projection nodes upgrade toward causal repair evidence.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, compute_full_string_logprob_batch, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions, get_hidden, patch_full_logprob, random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402


OUT_ROOT = Path("results/glm5_phase593_atlas_guided_causal_patch")
PHASE592_ROOT = Path("results/glm5_phase592_relation_specific_ranking_atlas")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_map(model, tokenizer, device, prompt: str, candidates: List[str]) -> Dict[str, float]:
    return {k: v[0] for k, v in compute_full_string_logprob_batch(model, tokenizer, device, prompt, candidates).items()}


def patched_score_map(model, tokenizer, device, prompt: str, candidates: List[str],
                      layer_idx: int, patch_pos: int, vec: torch.Tensor, alpha: float) -> Dict[str, float]:
    return {
        ans: patch_full_logprob(model, tokenizer, device, prompt, ans, layer_idx, patch_pos, vec, "add", alpha)[0]
        for ans in candidates
    }


def answer_vectors(model, tokenizer, candidates: List[str]) -> torch.Tensor:
    emb = model.get_output_embeddings().weight.detach().float().cpu()
    vecs = []
    for answer in candidates:
        ids = tokenizer.encode(" " + answer, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(answer, add_special_tokens=False)
        vecs.append(emb[ids].mean(dim=0))
    return torch.stack(vecs, dim=0)


def min_norm_vector_for_scores(E: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
    gram = E @ E.T
    coeff = torch.linalg.pinv(gram.float(), rtol=1e-4) @ target_scores.float()
    return E.T @ coeff


def decompose_by_candidate_scores(delta: torch.Tensor, E: torch.Tensor) -> Dict[str, torch.Tensor]:
    d = delta.float()
    scores = E @ d
    common_scores = torch.full_like(scores, float(scores.mean()))
    specific_scores = scores - scores.mean()
    common_vec = min_norm_vector_for_scores(E, common_scores)
    specific_vec = min_norm_vector_for_scores(E, specific_scores)
    raw_norm = torch.linalg.vector_norm(d).clamp_min(1e-8)
    spec_norm = torch.linalg.vector_norm(specific_vec).clamp_min(1e-8)
    common_norm = torch.linalg.vector_norm(common_vec).clamp_min(1e-8)
    return {
        "raw": d,
        "common_only": common_vec,
        "specific_only": specific_vec,
        "specific_norm_raw": specific_vec / spec_norm * raw_norm,
        "common_norm_raw": common_vec / common_norm * raw_norm,
        "random_same_norm": random_same_norm(d, seed=int(raw_norm.item() * 1000) % 1000003),
    }


def candidate_delta_metric(base_scores: Dict[str, float], patched_scores: Dict[str, float],
                           correct: str, old_top_wrong: str) -> Dict:
    deltas = {k: patched_scores[k] - base_scores[k] for k in base_scores}
    common = sum(deltas.values()) / max(1, len(deltas))
    cs = deltas[correct] - common
    ws = deltas[old_top_wrong] - common
    return {
        "common_delta": common,
        "correct_delta": deltas[correct],
        "old_top_wrong_delta": deltas[old_top_wrong],
        "correct_specific": cs,
        "old_top_wrong_specific": ws,
        "specific_margin_gain": cs - ws,
        "margin_gain": deltas[correct] - deltas[old_top_wrong],
    }


def load_phase592_nodes(model: str, top_k: int) -> List[Dict]:
    path = PHASE592_ROOT / f"phase592_{model}_relation_specific_ranking_atlas_confirm.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if model == "deepseek7b":
        priority = [
            ("rule_value", 26),
            ("prompt_last", 26),
            ("rule_relation", 18),
            ("rule_relation", 20),
            ("query_relation", 16),
            ("query_relation", 19),
        ]
        by_key = {(r["position"], r["layer"]): r for r in data["summary"]["best"]}
        nodes = []
        for pos, layer in priority:
            item = by_key.get((pos, layer))
            if item and item not in nodes:
                nodes.append(item)
        for item in data["summary"]["best"]:
            if item not in nodes:
                nodes.append(item)
        return nodes[:top_k]
    nodes = data["summary"]["best"][: max(top_k, 1)]
    return nodes[:top_k]


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        E = answer_vectors(model, tokenizer, values)
        cases = list(build_cases(args.n_tables, args.max_samples))
        nodes = load_phase592_nodes(args.model, args.top_nodes)
        if not nodes:
            raise RuntimeError(f"No phase592 nodes found for {args.model}")
        node_layers = sorted({n["layer"] for n in nodes})
        log(f"{args.model}: n_layers={info.n_layers}, cases={len(cases)}, nodes={[(n['position'], n['layer']) for n in nodes]}")

        rows = []
        for si, case in enumerate(cases):
            correct = case["correct"]
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
            base = winner_stats(base_scores, correct)
            repair = winner_stats(repair_scores, correct)
            target_case = (not base["correct"]) and repair["correct"]
            if args.target_only and not target_case:
                continue
            base_pos = case_positions(tokenizer, case, case["base_prompt"], case["relation"])
            repair_pos = case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"])
            base_h = get_hidden(model, tokenizer, device, case["base_prompt"], node_layers)
            repair_h = get_hidden(model, tokenizer, device, case["repair_prompt"], node_layers)
            old_top_wrong = base["top_wrong"]

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "patches": {},
            }
            for node in nodes:
                pos_name = node["position"]
                li = node["layer"]
                bp = base_pos.get(pos_name)
                rp = repair_pos.get(pos_name)
                if bp is None or rp is None:
                    continue
                if li not in base_h or li not in repair_h:
                    continue
                if bp >= base_h[li].shape[0] or rp >= repair_h[li].shape[0]:
                    continue
                delta = repair_h[li][rp] - base_h[li][bp]
                vecs = decompose_by_candidate_scores(delta, E)
                for mode, vec in vecs.items():
                    key = f"{pos_name}|L{li}|{mode}"
                    scores = patched_score_map(model, tokenizer, device, case["base_prompt"], values, li, bp, vec, args.alpha)
                    patched = winner_stats(scores, correct)
                    row["patches"][key] = {
                        "node": {"position": pos_name, "layer": li},
                        "mode": mode,
                        "winner": patched,
                        "metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                        "vec_norm": float(torch.linalg.vector_norm(vec.float())),
                        "raw_norm": float(torch.linalg.vector_norm(delta.float())),
                    }
            rows.append(row)

        summary = summarize(rows)
        return {
            "phase": 593,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "n_cases": len(cases),
            "n_target_rows": len(rows),
            "alpha": args.alpha,
            "nodes": nodes,
            "summary": summary,
            "rows": rows,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summarize(rows: List[Dict]) -> Dict:
    keys = sorted({k for r in rows for k in r["patches"]})
    by_key = {}
    for key in keys:
        items = [r["patches"][key] for r in rows if key in r["patches"]]
        if not items:
            continue
        entry = {
            "key": key,
            "position": items[0]["node"]["position"],
            "layer": items[0]["node"]["layer"],
            "mode": items[0]["mode"],
            "n": len(items),
            "switch": 0,
            "mean_common_delta": 0.0,
            "mean_correct_delta": 0.0,
            "mean_old_top_wrong_delta": 0.0,
            "mean_correct_specific": 0.0,
            "mean_old_top_wrong_specific": 0.0,
            "mean_specific_margin_gain": 0.0,
            "mean_margin_gain": 0.0,
            "positive_margin": 0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            m = item["metric"]
            for k in [
                "common_delta",
                "correct_delta",
                "old_top_wrong_delta",
                "correct_specific",
                "old_top_wrong_specific",
                "specific_margin_gain",
                "margin_gain",
            ]:
                entry[f"mean_{k}"] += m[k]
            entry["positive_margin"] += int(m["margin_gain"] > 0)
        n = max(1, len(items))
        for k in list(entry):
            if k.startswith("mean_"):
                entry[k] /= n
        entry["switch_rate"] = entry["switch"] / n
        entry["positive_margin_rate"] = entry["positive_margin"] / n
        by_key[key] = entry
    best = sorted(
        by_key.values(),
        key=lambda x: (x["switch"], x["mean_margin_gain"], x["mean_correct_specific"]),
        reverse=True,
    )[:24]
    for item in best[:12]:
        log(
            f"  {item['key']}: switch={item['switch']}/{item['n']}, "
            f"mgain={item['mean_margin_gain']:.3f}, spec_gain={item['mean_specific_margin_gain']:.3f}, "
            f"common={item['mean_common_delta']:.3f}, cspec={item['mean_correct_specific']:.3f}"
        )
    return {"by_key": by_key, "best": best}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--top-nodes", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=1.0)
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
        args.top_nodes = min(args.top_nodes, 2)
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 8)
        args.max_samples = max(args.max_samples, 64)
        args.top_nodes = max(args.top_nodes, 6)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase593_{args.model}_atlas_guided_causal_patch_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
