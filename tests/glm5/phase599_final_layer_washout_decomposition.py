#!/usr/bin/env python3
"""
Phase 599: Final Layer Washout Decomposition
最终层冲洗分解

Phase 598 showed that strong L26 generated projection survives into H27 but is
mostly gone by H28/final hidden. This phase decomposes the final block and final
norm/readout to locate the washout point.
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
from typing import Dict, List, Optional

import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, compute_full_string_logprob_batch, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions, random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import answer_vectors, candidate_delta_metric  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import (  # noqa: E402
    collect_mlp_input_output,
    get_mlp,
    mlp_parts_from_input,
    patched_score_map as input_patched_score_map,
    projection_metric,
    replace_input,
    score_map,
)
from phase598_downstream_trajectory_acceptance_audit import select_nodes  # noqa: E402


OUT_ROOT = Path("results/glm5_phase599_final_layer_washout_decomposition")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def get_attn(layer):
    for name in ["self_attn", "attention", "attn"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def get_final_norm(model):
    for path in [
        ("model", "norm"),
        ("transformer", "norm"),
        ("model", "final_layernorm"),
        ("transformer", "ln_f"),
    ]:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj
    return None


def candidate_first_ids(tokenizer, candidates: List[str]) -> Dict[str, int]:
    out = {}
    for ans in candidates:
        ids = tokenizer.encode(" " + ans, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(ans, add_special_tokens=False)
        out[ans] = ids[0]
    return out


def first_token_logit_metric(base_logits: Dict[str, float], patched_logits: Dict[str, float],
                             correct: str, old_top_wrong: str) -> Dict:
    deltas = {k: patched_logits[k] - base_logits[k] for k in base_logits}
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


def hidden_projection(effect: torch.Tensor, E: torch.Tensor, correct: str, old_top_wrong: str,
                      values: List[str]) -> Dict:
    return projection_metric(effect.float().cpu(), E, correct, old_top_wrong, values)


def build_targets(base_x: torch.Tensor, repair_x: torch.Tensor, wrong_x: Optional[torch.Tensor],
                  alpha: float, seed: int) -> List[Dict]:
    d_repair = repair_x.float().cpu() - base_x.float().cpu()
    specs = [
        {"name": f"repair_alpha{alpha:g}", "kind": "repair", "alpha": alpha, "target": base_x.float().cpu() + alpha * d_repair},
        {"name": f"random_alpha{alpha:g}", "kind": "random", "alpha": alpha, "target": base_x.float().cpu() + alpha * random_same_norm(d_repair, seed=seed)},
    ]
    if wrong_x is not None:
        specs.append({
            "name": f"wrong_alpha{alpha:g}",
            "kind": "wrong",
            "alpha": alpha,
            "target": base_x.float().cpu() + alpha * (wrong_x.float().cpu() - base_x.float().cpu()),
        })
    return specs


def collect_decomp(model, tokenizer, device, prompt: str, probe_layer: int,
                   source_layer: Optional[int] = None,
                   patch_pos: Optional[int] = None,
                   target_input: Optional[torch.Tensor] = None) -> Dict:
    layers = get_layers(model)
    layer = layers[probe_layer]
    attn = get_attn(layer)
    mlp = get_mlp(layer)
    final_norm = get_final_norm(model)
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    if source_layer is not None and patch_pos is not None and target_input is not None:
        source_mlp = get_mlp(layers[source_layer])
        target = target_input.to(device=device)

        def source_pre(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            x_new[0, patch_pos, :] = target.to(dtype=x_new.dtype)
            return replace_input(inputs, x_new)

        handles.append(source_mlp.register_forward_pre_hook(source_pre))

    def layer_pre(_module, inputs):
        captured["layer_input"] = inputs[0].detach().float().cpu()

    def attn_out(_module, _inputs, output):
        captured["attn_out"] = extract_tensor(output).detach().float().cpu()

    def mlp_pre(_module, inputs):
        captured["mlp_input"] = inputs[0].detach().float().cpu()

    def mlp_out(_module, _inputs, output):
        captured["mlp_out"] = extract_tensor(output).detach().float().cpu()

    def layer_out(_module, _inputs, output):
        captured["layer_out"] = extract_tensor(output).detach().float().cpu()

    handles.append(layer.register_forward_pre_hook(layer_pre))
    handles.append(layer.register_forward_hook(layer_out))
    if attn is not None:
        handles.append(attn.register_forward_hook(attn_out))
    if mlp is not None:
        handles.append(mlp.register_forward_pre_hook(mlp_pre))
        handles.append(mlp.register_forward_hook(mlp_out))
    if final_norm is not None:
        def norm_pre(_module, inputs):
            captured["final_norm_input"] = inputs[0].detach().float().cpu()

        def norm_out(_module, _inputs, output):
            captured["final_norm_output"] = extract_tensor(output).detach().float().cpu()

        handles.append(final_norm.register_forward_pre_hook(norm_pre))
        handles.append(final_norm.register_forward_hook(norm_out))

    try:
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, return_dict=True)
        captured["logits"] = out.logits.detach().float().cpu()
    finally:
        for h in handles:
            h.remove()
    return captured


def tensor_at(captured: Dict[str, torch.Tensor], name: str, pos: int) -> Optional[torch.Tensor]:
    t = captured.get(name)
    if t is None or pos >= t.shape[1]:
        return None
    return t[0, pos]


def logits_at(captured: Dict, pos: int, first_ids: Dict[str, int]) -> Dict[str, float]:
    logits = captured["logits"][0, pos].float()
    return {ans: float(logits[tid]) for ans, tid in first_ids.items()}


def summarize(rows: List[Dict]) -> Dict:
    comp_keys = sorted({k for r in rows for k in r["components"]})
    by_component = {}
    for key in comp_keys:
        items = [r["components"][key] for r in rows if key in r["components"]]
        entry = {
            "key": key,
            "component": items[0]["component"],
            "position": items[0]["node"]["position"],
            "source_layer": items[0]["node"]["source_layer"],
            "probe_layer": items[0]["node"]["probe_layer"],
            "kind": items[0]["kind"],
            "n": len(items),
            "mean_projection_specific_margin": 0.0,
            "mean_projection_correct_specific": 0.0,
            "mean_projection_old_top_wrong_specific": 0.0,
            "positive_projection_margin": 0,
        }
        for item in items:
            m = item["metric"]
            entry["mean_projection_specific_margin"] += m["projection_specific_margin"]
            entry["mean_projection_correct_specific"] += m["projection_correct_specific"]
            entry["mean_projection_old_top_wrong_specific"] += m["projection_old_top_wrong_specific"]
            entry["positive_projection_margin"] += int(m["projection_specific_margin"] > 0)
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["positive_projection_rate"] = entry["positive_projection_margin"] / n
        by_component[key] = entry

    patch_keys = sorted({k for r in rows for k in r["patches"]})
    by_patch = {}
    for key in patch_keys:
        items = [r["patches"][key] for r in rows if key in r["patches"]]
        entry = {
            "key": key,
            "position": items[0]["node"]["position"],
            "source_layer": items[0]["node"]["source_layer"],
            "kind": items[0]["kind"],
            "n": len(items),
            "switch": 0,
            "mean_generated_down_projection": 0.0,
            "mean_full_margin_gain": 0.0,
            "mean_first_token_logit_margin_gain": 0.0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            entry["mean_generated_down_projection"] += item["generated_down_metric"]["projection_specific_margin"]
            entry["mean_full_margin_gain"] += item["full_metric"]["margin_gain"]
            entry["mean_first_token_logit_margin_gain"] += item["first_token_logit_metric"]["margin_gain"]
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["switch_rate"] = entry["switch"] / n
        by_patch[key] = entry

    best_components = sorted(
        by_component.values(),
        key=lambda x: (x["mean_projection_specific_margin"], x["positive_projection_rate"]),
        reverse=True,
    )[:60]
    best_patches = sorted(
        by_patch.values(),
        key=lambda x: (x["switch"], x["mean_full_margin_gain"], x["mean_generated_down_projection"]),
        reverse=True,
    )[:36]
    log("Best component projections:")
    for item in best_components[:12]:
        log(
            f"  {item['key']}: comp={item['component']}, "
            f"proj={item['mean_projection_specific_margin']:.3f}, rate={item['positive_projection_rate']:.3f}"
        )
    log("Best final effects:")
    for item in best_patches[:8]:
        log(
            f"  {item['key']}: switch={item['switch']}/{item['n']}, "
            f"full={item['mean_full_margin_gain']:.3f}, first={item['mean_first_token_logit_margin_gain']:.3f}"
        )
    return {"by_component": by_component, "by_patch": by_patch, "best_components": best_components, "best_patches": best_patches}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        first_ids = candidate_first_ids(tokenizer, values)
        E = answer_vectors(model, tokenizer, values)
        cases = list(build_cases(args.n_tables, args.max_samples))
        nodes = select_nodes(args.model, args.top_nodes)
        source_layers = sorted({n["layer"] for n in nodes})
        probe_layer = info.n_layers - 1
        log(f"{args.model}: layers={info.n_layers}, cases={len(cases)}, nodes={[(n['position'], n['layer']) for n in nodes]}, probe=L{probe_layer}")

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
            old_top_wrong = base["top_wrong"]
            base_pos = case_positions(tokenizer, case, case["base_prompt"], case["relation"])
            repair_pos = case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"])
            wrong_pos = case_positions(tokenizer, case, case["wrong_prompt"], case["wrong_rel"]) if args.include_wrong_controls else {}
            base_cap = collect_mlp_input_output(model, tokenizer, device, case["base_prompt"], source_layers)
            repair_cap = collect_mlp_input_output(model, tokenizer, device, case["repair_prompt"], source_layers)
            wrong_cap = collect_mlp_input_output(model, tokenizer, device, case["wrong_prompt"], source_layers) if args.include_wrong_controls else None
            base_decomp = collect_decomp(model, tokenizer, device, case["base_prompt"], probe_layer)
            base_first_logits = logits_at(base_decomp, len(tokenizer.encode(case["base_prompt"], add_special_tokens=False)) - 1, first_ids)

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "components": {},
                "patches": {},
            }
            for node in nodes:
                pos_name = node["position"]
                source_layer = node["layer"]
                bp = base_pos.get(pos_name)
                rp = repair_pos.get(pos_name)
                wp = wrong_pos.get(pos_name) if args.include_wrong_controls else None
                if bp is None or rp is None:
                    continue
                if source_layer not in base_cap["mlp_input"] or source_layer not in repair_cap["mlp_input"]:
                    continue
                if bp >= base_cap["mlp_input"][source_layer].shape[1] or rp >= repair_cap["mlp_input"][source_layer].shape[1]:
                    continue
                base_x = base_cap["mlp_input"][source_layer][0, bp]
                repair_x = repair_cap["mlp_input"][source_layer][0, rp]
                wrong_x = None
                if wrong_cap is not None and wp is not None and source_layer in wrong_cap["mlp_input"] and wp < wrong_cap["mlp_input"][source_layer].shape[1]:
                    wrong_x = wrong_cap["mlp_input"][source_layer][0, wp]
                d_repair = repair_x.float().cpu() - base_x.float().cpu()
                specs = [
                    {"name": f"repair_alpha{args.alpha:g}", "kind": "repair", "target": base_x.float().cpu() + args.alpha * d_repair},
                    {"name": f"random_alpha{args.alpha:g}", "kind": "random", "target": base_x.float().cpu() + args.alpha * random_same_norm(d_repair, seed=si * 1009 + source_layer)},
                ]
                if wrong_x is not None:
                    specs.append({"name": f"wrong_alpha{args.alpha:g}", "kind": "wrong", "target": base_x.float().cpu() + args.alpha * (wrong_x.float().cpu() - base_x.float().cpu())})
                source_mlp = get_mlp(get_layers(model)[source_layer])
                base_down = mlp_parts_from_input(source_mlp, base_x)["down"]

                for spec in specs:
                    pkey = f"{pos_name}|L{source_layer}|{spec['name']}"
                    generated_down = mlp_parts_from_input(source_mlp, spec["target"])["down"] - base_down
                    gen_metric = projection_metric(generated_down, E, correct, old_top_wrong, values)
                    patched_scores = input_patched_score_map(model, tokenizer, device, case["base_prompt"], values, source_layer, bp, spec["target"])
                    patched = winner_stats(patched_scores, correct)
                    patched_decomp = collect_decomp(
                        model, tokenizer, device, case["base_prompt"], probe_layer,
                        source_layer, bp, spec["target"]
                    )
                    patched_first_logits = logits_at(patched_decomp, len(tokenizer.encode(case["base_prompt"], add_special_tokens=False)) - 1, first_ids)
                    row["patches"][pkey] = {
                        "node": {"position": pos_name, "source_layer": source_layer, "probe_layer": probe_layer},
                        "kind": spec["kind"],
                        "winner": patched,
                        "generated_down_metric": gen_metric,
                        "full_metric": candidate_delta_metric(base_scores, patched_scores, correct, old_top_wrong),
                        "first_token_logit_metric": first_token_logit_metric(base_first_logits, patched_first_logits, correct, old_top_wrong),
                    }

                    for component in [
                        "layer_input",
                        "attn_out",
                        "mlp_input",
                        "mlp_out",
                        "layer_out",
                        "final_norm_input",
                        "final_norm_output",
                    ]:
                        base_vec = tensor_at(base_decomp, component, bp)
                        patched_vec = tensor_at(patched_decomp, component, bp)
                        if base_vec is None or patched_vec is None:
                            continue
                        effect = patched_vec - base_vec
                        ckey = f"{pos_name}|L{source_layer}|{spec['name']}|{component}"
                        row["components"][ckey] = {
                            "node": {"position": pos_name, "source_layer": source_layer, "probe_layer": probe_layer},
                            "kind": spec["kind"],
                            "component": component,
                            "metric": hidden_projection(effect, E, correct, old_top_wrong, values),
                        }
            rows.append(row)

        return {
            "phase": 599,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "probe_layer": probe_layer,
            "n_cases": len(cases),
            "n_target_cases_seen": target_seen,
            "n_rows": len(rows),
            "target_only": args.target_only,
            "alpha": args.alpha,
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
    parser.add_argument("--top-nodes", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
    parser.add_argument("--include-wrong-controls", action="store_true", default=True)
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
        args.top_nodes = max(args.top_nodes, 3)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase599_{args.model}_final_layer_washout_decomposition_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
