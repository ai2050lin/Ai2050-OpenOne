#!/usr/bin/env python3
"""
Phase 595: MLP Update Causal Validation
MLP 更新因果验证

Phase 594 found strong candidate-specific projection in DS7B rule_value L26
MLP update. This phase patches the MLP module output itself, rather than the
residual hidden state, and checks whether that update causally repairs candidate
ranking.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, compute_full_string_logprob_batch, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase589_component_value_path_attribution import collect_component_outputs, patched_score_map as component_patched_score_map  # noqa: E402
from phase593_atlas_guided_causal_patch import answer_vectors, candidate_delta_metric, decompose_by_candidate_scores  # noqa: E402


OUT_ROOT = Path("results/glm5_phase595_mlp_update_causal_validation")
PHASE594_ROOT = Path("results/glm5_phase594_conditional_transformation_atlas")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_map(model, tokenizer, device, prompt: str, candidates: List[str]) -> Dict[str, float]:
    return {k: v[0] for k, v in compute_full_string_logprob_batch(model, tokenizer, device, prompt, candidates).items()}


def load_phase594_mlp_nodes(model: str, top_k: int) -> List[Dict]:
    path = PHASE594_ROOT / f"phase594_{model}_conditional_transformation_atlas_confirm.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase594 confirm result: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    by_key = data.get("summary", {}).get("by_key", {})

    if model == "deepseek7b":
        priority = [
            ("rule_value", 26),
            ("query_relation", 19),
            ("prompt_last", 26),
            ("rule_relation", 20),
            ("rule_relation", 18),
            ("query_relation", 16),
        ]
        nodes = []
        for pos, layer in priority:
            item = by_key.get(f"{pos}|L{layer}|mlp_update")
            if item:
                nodes.append({"position": pos, "layer": layer, "phase594": item})
        for item in sorted(by_key.values(), key=lambda x: x.get("mean_specific_margin", -999.0), reverse=True):
            if item.get("source") != "mlp_update":
                continue
            node = {"position": item["position"], "layer": item["layer"], "phase594": item}
            if node not in nodes:
                nodes.append(node)
        return nodes[:top_k]

    mlp_items = [
        item for item in by_key.values()
        if item.get("source") == "mlp_update" and item.get("n", 0) > 0
    ]
    mlp_items.sort(key=lambda x: (x.get("mean_specific_margin", -999.0), x.get("positive_rate", 0.0)), reverse=True)
    return [
        {"position": item["position"], "layer": item["layer"], "phase594": item}
        for item in mlp_items[:top_k]
    ]


def select_vecs(raw_delta: torch.Tensor, E: torch.Tensor, modes: List[str]) -> Dict[str, torch.Tensor]:
    vecs = decompose_by_candidate_scores(raw_delta, E)
    return {m: vecs[m] for m in modes if m in vecs}


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
            "component": items[0]["component"],
            "mode": items[0]["mode"],
            "n": len(items),
            "switch": 0,
            "base_correct": 0,
            "repair_correct": 0,
            "mean_common_delta": 0.0,
            "mean_correct_delta": 0.0,
            "mean_old_top_wrong_delta": 0.0,
            "mean_correct_specific": 0.0,
            "mean_old_top_wrong_specific": 0.0,
            "mean_specific_margin_gain": 0.0,
            "mean_margin_gain": 0.0,
            "positive_margin": 0,
            "mean_vec_norm": 0.0,
            "mean_raw_norm": 0.0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            entry["base_correct"] += int(item["base_correct"])
            entry["repair_correct"] += int(item["repair_correct"])
            entry["mean_vec_norm"] += item["vec_norm"]
            entry["mean_raw_norm"] += item["raw_norm"]
            m = item["metric"]
            for name in [
                "common_delta",
                "correct_delta",
                "old_top_wrong_delta",
                "correct_specific",
                "old_top_wrong_specific",
                "specific_margin_gain",
                "margin_gain",
            ]:
                entry[f"mean_{name}"] += m[name]
            entry["positive_margin"] += int(m["margin_gain"] > 0)
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["switch_rate"] = entry["switch"] / n
        entry["positive_margin_rate"] = entry["positive_margin"] / n
        by_key[key] = entry

    best = sorted(
        by_key.values(),
        key=lambda x: (x["switch"], x["mean_margin_gain"], x["mean_specific_margin_gain"]),
        reverse=True,
    )[:36]
    for item in best[:14]:
        log(
            f"  {item['key']}: switch={item['switch']}/{item['n']}, "
            f"mgain={item['mean_margin_gain']:.3f}, spec_gain={item['mean_specific_margin_gain']:.3f}, "
            f"common={item['mean_common_delta']:.3f}, cspec={item['mean_correct_specific']:.3f}"
        )
    return {"by_key": by_key, "best": best}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        E = answer_vectors(model, tokenizer, values)
        cases = list(build_cases(args.n_tables, args.max_samples))
        nodes = load_phase594_mlp_nodes(args.model, args.top_nodes)
        if not nodes:
            raise RuntimeError(f"No Phase594 MLP nodes found for {args.model}")
        node_layers = sorted({n["layer"] for n in nodes})
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        log(
            f"{args.model}: n_layers={info.n_layers}, cases={len(cases)}, "
            f"nodes={[(n['position'], n['layer']) for n in nodes]}, modes={modes}"
        )

        rows = []
        target_seen = 0
        for si, case in enumerate(cases):
            correct = case["correct"]
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
            base = winner_stats(base_scores, correct)
            repair = winner_stats(repair_scores, correct)
            target_case = (not base["correct"]) and repair["correct"]
            if args.target_only and not target_case:
                continue
            target_seen += int(target_case)

            base_pos = case_positions(tokenizer, case, case["base_prompt"], case["relation"])
            repair_pos = case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"])
            base_out = collect_component_outputs(model, tokenizer, device, case["base_prompt"], node_layers, ["mlp"])
            repair_out = collect_component_outputs(model, tokenizer, device, case["repair_prompt"], node_layers, ["mlp"])
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
                if li not in base_out.get("mlp", {}) or li not in repair_out.get("mlp", {}):
                    continue
                if bp >= base_out["mlp"][li].shape[1] or rp >= repair_out["mlp"][li].shape[1]:
                    continue
                raw_delta = repair_out["mlp"][li][0, rp, :] - base_out["mlp"][li][0, bp, :]
                for mode, vec in select_vecs(raw_delta, E, modes).items():
                    key = f"{pos_name}|L{li}|mlp|{mode}"
                    patched_scores = component_patched_score_map(
                        model, tokenizer, device, case["base_prompt"], values,
                        li, "mlp", bp, vec, args.alpha
                    )
                    patched = winner_stats(patched_scores, correct)
                    row["patches"][key] = {
                        "node": {"position": pos_name, "layer": li},
                        "component": "mlp",
                        "mode": mode,
                        "winner": patched,
                        "metric": candidate_delta_metric(base_scores, patched_scores, correct, old_top_wrong),
                        "vec_norm": float(torch.linalg.vector_norm(vec.float())),
                        "raw_norm": float(torch.linalg.vector_norm(raw_delta.float())),
                        "base_correct": base["correct"],
                        "repair_correct": repair["correct"],
                    }
            rows.append(row)

        return {
            "phase": 595,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "n_cases": len(cases),
            "n_target_cases_seen": target_seen,
            "n_rows": len(rows),
            "target_only": args.target_only,
            "alpha": args.alpha,
            "modes": modes,
            "nodes": nodes,
            "summary": summarize(rows),
            "rows": rows,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--top-nodes", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--modes",
        default="raw,specific_only,common_only,specific_norm_raw,common_norm_raw,random_same_norm",
    )
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
        args.modes = "raw,specific_only,random_same_norm"
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
    out_path = out_dir / f"phase595_{args.model}_mlp_update_causal_validation_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
