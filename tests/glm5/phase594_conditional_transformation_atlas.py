#!/usr/bin/env python3
"""
Phase 594: Conditional Transformation Atlas
条件化状态变换图谱

Phase 593 showed that projection nodes are not directly portable additive
vectors. This phase measures layer-to-layer transitions around atlas nodes:
incoming hidden state, outgoing hidden state, residual update, attention update,
and MLP update.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
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
from phase589_component_value_path_attribution import collect_component_outputs  # noqa: E402


OUT_ROOT = Path("results/glm5_phase594_conditional_transformation_atlas")
PHASE592_ROOT = Path("results/glm5_phase592_relation_specific_ranking_atlas")


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


def projection_scores(vec: torch.Tensor, answer_vecs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    v = vec.float()
    return {k: float(torch.dot(v, av.float())) for k, av in answer_vecs.items()}


def specific_metric(vec: torch.Tensor, answer_vecs: Dict[str, torch.Tensor],
                    correct: str, old_top_wrong: str) -> Dict:
    scores = projection_scores(vec, answer_vecs)
    common = sum(scores.values()) / max(1, len(scores))
    cs = scores[correct] - common
    ws = scores[old_top_wrong] - common
    return {
        "common": common,
        "correct_delta": scores[correct],
        "old_top_wrong_delta": scores[old_top_wrong],
        "correct_specific": cs,
        "old_top_wrong_specific": ws,
        "specific_margin": cs - ws,
    }


def load_phase592_nodes(model: str, top_k: int) -> List[Dict]:
    path = PHASE592_ROOT / f"phase592_{model}_relation_specific_ranking_atlas_confirm.json"
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
        nodes = [by_key[k] for k in priority if k in by_key]
        for item in data["summary"]["best"]:
            if item not in nodes:
                nodes.append(item)
        return nodes[:top_k]
    return data["summary"]["best"][:top_k]


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        answer_vecs = answer_vectors(model, tokenizer, values)
        cases = list(build_cases(args.n_tables, args.max_samples))
        nodes = load_phase592_nodes(args.model, args.top_nodes)
        node_layers = sorted({n["layer"] for n in nodes})
        components = ["attn", "mlp"]
        log(f"{args.model}: n_layers={info.n_layers}, cases={len(cases)}, nodes={[(n['position'], n['layer']) for n in nodes]}")

        rows = []
        accum = defaultdict(lambda: defaultdict(float))
        counts = defaultdict(int)
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
            base_h = get_hidden_all(model, tokenizer, device, case["base_prompt"])
            repair_h = get_hidden_all(model, tokenizer, device, case["repair_prompt"])
            base_comp = collect_component_outputs(model, tokenizer, device, case["base_prompt"], node_layers, components)
            repair_comp = collect_component_outputs(model, tokenizer, device, case["repair_prompt"], node_layers, components)
            old_top_wrong = base["top_wrong"]
            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "nodes": {},
            }
            for node in nodes:
                pos_name = node["position"]
                li = node["layer"]
                bp = base_pos.get(pos_name)
                rp = repair_pos.get(pos_name)
                if bp is None or rp is None or li + 1 >= len(base_h):
                    continue
                if bp >= base_h[li].shape[0] or rp >= repair_h[li].shape[0]:
                    continue
                if bp >= base_h[li + 1].shape[0] or rp >= repair_h[li + 1].shape[0]:
                    continue
                incoming = repair_h[li][rp] - base_h[li][bp]
                outgoing = repair_h[li + 1][rp] - base_h[li + 1][bp]
                residual_update = outgoing - incoming
                metrics = {
                    "incoming": specific_metric(incoming, answer_vecs, correct, old_top_wrong),
                    "outgoing": specific_metric(outgoing, answer_vecs, correct, old_top_wrong),
                    "residual_update": specific_metric(residual_update, answer_vecs, correct, old_top_wrong),
                }
                for comp in components:
                    if li in base_comp.get(comp, {}) and li in repair_comp.get(comp, {}):
                        if bp < base_comp[comp][li].shape[1] and rp < repair_comp[comp][li].shape[1]:
                            comp_delta = repair_comp[comp][li][0, rp, :] - base_comp[comp][li][0, bp, :]
                            metrics[f"{comp}_update"] = specific_metric(comp_delta, answer_vecs, correct, old_top_wrong)
                trans = {
                    "out_minus_in_specific_margin": metrics["outgoing"]["specific_margin"] - metrics["incoming"]["specific_margin"],
                    "out_minus_in_correct_specific": metrics["outgoing"]["correct_specific"] - metrics["incoming"]["correct_specific"],
                    "out_minus_in_old_top_wrong_specific": metrics["outgoing"]["old_top_wrong_specific"] - metrics["incoming"]["old_top_wrong_specific"],
                }
                metrics["transition_gain"] = trans
                key_prefix = f"{pos_name}|L{li}"
                row["nodes"][key_prefix] = metrics
                for source, metric in metrics.items():
                    if source == "transition_gain":
                        continue
                    key = f"{key_prefix}|{source}"
                    counts[key] += 1
                    for m, val in metric.items():
                        accum[key][m] += val
                    accum[key]["positive_specific_margin"] += int(metric["specific_margin"] > 0)
                key = f"{key_prefix}|transition_gain"
                counts[key] += 1
                for m, val in trans.items():
                    accum[key][m] += val
                accum[key]["positive_specific_margin"] += int(trans["out_minus_in_specific_margin"] > 0)
            rows.append(row)

        summary_rows = []
        for key, vals in accum.items():
            n = max(1, counts[key])
            position, layer_text, source = key.split("|", 2)
            item = {"key": key, "position": position, "layer": int(layer_text[1:]), "source": source, "n": counts[key]}
            for m, val in vals.items():
                if m == "positive_specific_margin":
                    item["positive_rate"] = val / n
                else:
                    item[f"mean_{m}"] = val / n
            summary_rows.append(item)
        summary_rows.sort(
            key=lambda r: (
                r.get("mean_specific_margin", r.get("mean_out_minus_in_specific_margin", -1e9)),
                r.get("mean_correct_specific", r.get("mean_out_minus_in_correct_specific", -1e9)),
            ),
            reverse=True,
        )
        for item in summary_rows[:12]:
            val = item.get("mean_specific_margin", item.get("mean_out_minus_in_specific_margin", 0.0))
            cs = item.get("mean_correct_specific", item.get("mean_out_minus_in_correct_specific", 0.0))
            log(f"  {item['key']}: val={val:.3f}, cs={cs:.3f}, pos={item.get('positive_rate',0):.2f}")
        atlas = build_atlas(args.model, info.n_layers, len(cases), len(rows), summary_rows)
        return {
            "phase": 594,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "n_cases": len(cases),
            "n_target_rows": len(rows),
            "nodes": nodes,
            "summary": {
                "best": summary_rows[:30],
                "by_key": {r["key"]: r for r in summary_rows},
                "source_counts_top30": dict(Counter(r["source"] for r in summary_rows[:30])),
            },
            "atlas": atlas,
            "rows": rows,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_atlas(model: str, n_layers: int, n_cases: int, n_target_rows: int, summary_rows: List[Dict]) -> Dict:
    nodes = []
    edges = []
    for item in summary_rows[:80]:
        node_id = f"{model}_{item['position']}_L{item['layer']}_{item['source']}"
        nodes.append({
            "node_id": node_id,
            "node_type": "transition_or_component_update",
            "model": model,
            "layer": item["layer"],
            "position": item["position"],
            "source": item["source"],
            "causal_level": 2,
            "metrics": item,
        })
        edges.append({
            "source": f"{model}_{item['position']}_L{item['layer']}_incoming",
            "target": node_id,
            "edge_type": "conditional_update_projection",
            "causal_level": 2,
            "effect": item,
        })
    return {
        "graph_type": "conditional_transformation_atlas_slice",
        "phase": 594,
        "model": model,
        "n_layers": n_layers,
        "n_cases": n_cases,
        "n_target_rows": n_target_rows,
        "nodes": nodes,
        "edges": edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--top-nodes", type=int, default=6)
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
    out_path = out_dir / f"phase594_{args.model}_conditional_transformation_atlas_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
