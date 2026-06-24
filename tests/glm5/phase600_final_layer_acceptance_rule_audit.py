#!/usr/bin/env python3
"""
Phase 600: Final-Layer Acceptance Rule Audit
最后层轨迹接受规则审计

Phase 599 located the washout inside the final block. This phase compares
natural correct, natural wrong, artificial repair, and artificial random
trajectories at the final block to identify what the final layer accepts.
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
import torch.nn.functional as F

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
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn, get_final_norm  # noqa: E402


OUT_ROOT = Path("results/glm5_phase600_final_layer_acceptance_rule_audit")
COMPONENTS = [
    "layer_input",
    "attn_out",
    "mlp_input",
    "mlp_out",
    "layer_out",
    "final_norm_input",
    "final_norm_output",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.float()).cpu())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().cpu()
    b = b.float().cpu()
    if torch.linalg.vector_norm(a) < 1e-8 or torch.linalg.vector_norm(b) < 1e-8:
        return 0.0
    return float(F.cosine_similarity(a.view(1, -1), b.view(1, -1)).item())


def get_position(captured: Dict, name: str, pos: int) -> Optional[torch.Tensor]:
    t = captured.get(name)
    if t is None or pos is None or pos < 0 or pos >= t.shape[1]:
        return None
    return t[0, pos]


def get_attn_slice(attn: Optional[torch.Tensor], pos: int) -> Optional[torch.Tensor]:
    if attn is None or pos is None or pos < 0:
        return None
    if attn.dim() != 4 or pos >= attn.shape[2]:
        return None
    return attn[0, :, pos, :].float().cpu()


def attn_stats(base_attn: Optional[torch.Tensor], target_attn: Optional[torch.Tensor]) -> Dict:
    if base_attn is None or target_attn is None:
        return {}
    min_len = min(base_attn.shape[-1], target_attn.shape[-1])
    if min_len <= 0:
        return {}
    b = base_attn[..., :min_len]
    t = target_attn[..., :min_len]
    eps = 1e-8
    b = b / b.sum(dim=-1, keepdim=True).clamp_min(eps)
    t = t / t.sum(dim=-1, keepdim=True).clamp_min(eps)
    l1 = torch.abs(t - b).sum(dim=-1).mean()
    entropy = -(t * torch.log(t.clamp_min(eps))).sum(dim=-1).mean()
    top_mass = t.max(dim=-1).values.mean()
    return {
        "attn_l1_to_base": float(l1.cpu()),
        "attn_entropy": float(entropy.cpu()),
        "attn_top_mass": float(top_mass.cpu()),
    }


def collect_final_block(model, tokenizer, device, prompt: str, probe_layer: int,
                        source_layer: Optional[int] = None,
                        patch_pos: Optional[int] = None,
                        target_input: Optional[torch.Tensor] = None,
                        capture_attn: bool = True) -> Dict:
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

    def layer_out(_module, _inputs, output):
        captured["layer_out"] = extract_tensor(output).detach().float().cpu()

    def attn_out(_module, _inputs, output):
        captured["attn_out"] = extract_tensor(output).detach().float().cpu()

    def mlp_pre(_module, inputs):
        captured["mlp_input"] = inputs[0].detach().float().cpu()

    def mlp_out(_module, _inputs, output):
        captured["mlp_out"] = extract_tensor(output).detach().float().cpu()

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
            out = model(
                input_ids=input_ids,
                output_attentions=capture_attn,
                return_dict=True,
            )
        captured["logits"] = out.logits.detach().float().cpu()
        if capture_attn and getattr(out, "attentions", None) is not None:
            if probe_layer < len(out.attentions) and out.attentions[probe_layer] is not None:
                captured["attention_pattern"] = out.attentions[probe_layer].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()
    return captured


def effect_metric(effect: torch.Tensor, natural_effect: Optional[torch.Tensor],
                  E: torch.Tensor, correct: str, old_top_wrong: str, values: List[str]) -> Dict:
    proj = projection_metric(effect, E, correct, old_top_wrong, values)
    out = {
        **proj,
        "effect_norm": safe_norm(effect),
        "cos_to_natural_correct": cosine(effect, natural_effect) if natural_effect is not None else 0.0,
        "norm_ratio_to_natural_correct": 0.0,
    }
    if natural_effect is not None:
        n = safe_norm(natural_effect)
        out["norm_ratio_to_natural_correct"] = safe_norm(effect) / max(n, 1e-8)
    return out


def make_targets(base_x: torch.Tensor, repair_x: torch.Tensor, wrong_x: Optional[torch.Tensor],
                 alpha: float, seed: int) -> List[Dict]:
    d_repair = repair_x.float().cpu() - base_x.float().cpu()
    specs = [
        {"name": "artificial_repair", "kind": "artificial_repair", "target": base_x.float().cpu() + alpha * d_repair},
        {"name": "artificial_random", "kind": "artificial_random", "target": base_x.float().cpu() + alpha * random_same_norm(d_repair, seed=seed)},
    ]
    if wrong_x is not None:
        specs.append({"name": "artificial_wrong", "kind": "artificial_wrong", "target": base_x.float().cpu() + alpha * (wrong_x.float().cpu() - base_x.float().cpu())})
    return specs


def summarize(rows: List[Dict]) -> Dict:
    keys = sorted({k for r in rows for k in r["trajectory_components"]})
    by_component = {}
    for key in keys:
        items = [r["trajectory_components"][key] for r in rows if key in r["trajectory_components"]]
        entry = {
            "key": key,
            "trajectory": items[0]["trajectory"],
            "component": items[0]["component"],
            "position": items[0]["node"]["position"],
            "source_layer": items[0]["node"]["source_layer"],
            "probe_layer": items[0]["node"]["probe_layer"],
            "n": len(items),
            "mean_projection_specific_margin": 0.0,
            "mean_correct_specific": 0.0,
            "mean_old_top_wrong_specific": 0.0,
            "mean_effect_norm": 0.0,
            "mean_cos_to_natural_correct": 0.0,
            "mean_norm_ratio_to_natural_correct": 0.0,
            "mean_attn_l1_to_base": 0.0,
            "mean_attn_entropy": 0.0,
            "mean_attn_top_mass": 0.0,
            "positive_projection": 0,
        }
        attn_count = 0
        for item in items:
            m = item["metric"]
            entry["mean_projection_specific_margin"] += m["projection_specific_margin"]
            entry["mean_correct_specific"] += m["projection_correct_specific"]
            entry["mean_old_top_wrong_specific"] += m["projection_old_top_wrong_specific"]
            entry["mean_effect_norm"] += m["effect_norm"]
            entry["mean_cos_to_natural_correct"] += m["cos_to_natural_correct"]
            entry["mean_norm_ratio_to_natural_correct"] += m["norm_ratio_to_natural_correct"]
            entry["positive_projection"] += int(m["projection_specific_margin"] > 0)
            a = item.get("attention", {})
            if a:
                attn_count += 1
                entry["mean_attn_l1_to_base"] += a.get("attn_l1_to_base", 0.0)
                entry["mean_attn_entropy"] += a.get("attn_entropy", 0.0)
                entry["mean_attn_top_mass"] += a.get("attn_top_mass", 0.0)
        n = max(1, len(items))
        for name in [
            "mean_projection_specific_margin",
            "mean_correct_specific",
            "mean_old_top_wrong_specific",
            "mean_effect_norm",
            "mean_cos_to_natural_correct",
            "mean_norm_ratio_to_natural_correct",
        ]:
            entry[name] /= n
        if attn_count:
            entry["mean_attn_l1_to_base"] /= attn_count
            entry["mean_attn_entropy"] /= attn_count
            entry["mean_attn_top_mass"] /= attn_count
        entry["positive_projection_rate"] = entry["positive_projection"] / n
        by_component[key] = entry

    pkeys = sorted({k for r in rows for k in r["patches"]})
    by_patch = {}
    for key in pkeys:
        items = [r["patches"][key] for r in rows if key in r["patches"]]
        entry = {
            "key": key,
            "trajectory": items[0]["trajectory"],
            "position": items[0]["node"]["position"],
            "source_layer": items[0]["node"]["source_layer"],
            "n": len(items),
            "switch": 0,
            "mean_full_margin_gain": 0.0,
            "mean_generated_down_projection": 0.0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            entry["mean_full_margin_gain"] += item["full_metric"]["margin_gain"]
            entry["mean_generated_down_projection"] += item["generated_down_metric"]["projection_specific_margin"]
        n = max(1, len(items))
        entry["mean_full_margin_gain"] /= n
        entry["mean_generated_down_projection"] /= n
        entry["switch_rate"] = entry["switch"] / n
        by_patch[key] = entry

    best_projection = sorted(
        by_component.values(),
        key=lambda x: (x["mean_projection_specific_margin"], x["mean_cos_to_natural_correct"]),
        reverse=True,
    )[:80]
    natural_alignment = sorted(
        by_component.values(),
        key=lambda x: (x["mean_cos_to_natural_correct"], x["mean_projection_specific_margin"]),
        reverse=True,
    )[:80]
    final_effects = sorted(
        by_patch.values(),
        key=lambda x: (x["switch"], x["mean_full_margin_gain"], x["mean_generated_down_projection"]),
        reverse=True,
    )[:40]
    log("Best projection components:")
    for item in best_projection[:10]:
        log(
            f"  {item['key']}: proj={item['mean_projection_specific_margin']:.3f}, "
            f"cos_nat={item['mean_cos_to_natural_correct']:.3f}"
        )
    log("Best artificial final effects:")
    for item in final_effects[:8]:
        log(
            f"  {item['key']}: switch={item['switch']}/{item['n']}, "
            f"full={item['mean_full_margin_gain']:.3f}, gen={item['mean_generated_down_projection']:.3f}"
        )
    return {
        "by_component": by_component,
        "by_patch": by_patch,
        "best_projection": best_projection,
        "natural_alignment": natural_alignment,
        "final_effects": final_effects,
    }


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
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
            wrong_scores = score_map(model, tokenizer, device, case["wrong_prompt"], values)
            base = winner_stats(base_scores, correct)
            repair = winner_stats(repair_scores, correct)
            wrong = winner_stats(wrong_scores, correct)
            target_case = (not base["correct"]) and repair["correct"]
            if args.target_only and not target_case:
                continue
            target_seen += int(target_case)
            old_top_wrong = base["top_wrong"]

            base_pos = case_positions(tokenizer, case, case["base_prompt"], case["relation"])
            repair_pos = case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"])
            wrong_pos = case_positions(tokenizer, case, case["wrong_prompt"], case["wrong_rel"])

            base_cap = collect_mlp_input_output(model, tokenizer, device, case["base_prompt"], source_layers)
            repair_cap = collect_mlp_input_output(model, tokenizer, device, case["repair_prompt"], source_layers)
            wrong_cap = collect_mlp_input_output(model, tokenizer, device, case["wrong_prompt"], source_layers)

            base_decomp = collect_final_block(model, tokenizer, device, case["base_prompt"], probe_layer, capture_attn=args.capture_attn)
            repair_decomp = collect_final_block(model, tokenizer, device, case["repair_prompt"], probe_layer, capture_attn=args.capture_attn)
            wrong_decomp = collect_final_block(model, tokenizer, device, case["wrong_prompt"], probe_layer, capture_attn=args.capture_attn)

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "wrong_prompt": wrong,
                "trajectory_components": {},
                "patches": {},
            }

            for node in nodes:
                pos_name = node["position"]
                source_layer = node["layer"]
                bp = base_pos.get(pos_name)
                rp = repair_pos.get(pos_name)
                wp = wrong_pos.get(pos_name)
                if bp is None or rp is None or wp is None:
                    continue
                if source_layer not in base_cap["mlp_input"] or source_layer not in repair_cap["mlp_input"]:
                    continue
                if bp >= base_cap["mlp_input"][source_layer].shape[1] or rp >= repair_cap["mlp_input"][source_layer].shape[1]:
                    continue
                base_x = base_cap["mlp_input"][source_layer][0, bp]
                repair_x = repair_cap["mlp_input"][source_layer][0, rp]
                wrong_x = None
                if source_layer in wrong_cap["mlp_input"] and wp < wrong_cap["mlp_input"][source_layer].shape[1]:
                    wrong_x = wrong_cap["mlp_input"][source_layer][0, wp]

                source_mlp = get_mlp(get_layers(model)[source_layer])
                base_down = mlp_parts_from_input(source_mlp, base_x)["down"]
                specs = make_targets(base_x, repair_x, wrong_x, args.alpha, seed=si * 1009 + source_layer)

                natural_effects: Dict[str, Optional[torch.Tensor]] = {}
                for comp in COMPONENTS:
                    b = get_position(base_decomp, comp, bp)
                    r = get_position(repair_decomp, comp, rp)
                    w = get_position(wrong_decomp, comp, wp)
                    if b is not None and r is not None:
                        natural_effects[comp] = r - b
                    else:
                        natural_effects[comp] = None
                    for tname, tdecomp, tpos, trajectory in [
                        ("natural_correct", repair_decomp, rp, "natural_correct"),
                        ("natural_wrong", wrong_decomp, wp, "natural_wrong"),
                    ]:
                        tv = get_position(tdecomp, comp, tpos)
                        if b is None or tv is None:
                            continue
                        effect = tv - b
                        key = f"{pos_name}|L{source_layer}|{trajectory}|{comp}"
                        row["trajectory_components"][key] = {
                            "node": {"position": pos_name, "source_layer": source_layer, "probe_layer": probe_layer},
                            "trajectory": trajectory,
                            "component": comp,
                            "metric": effect_metric(effect, natural_effects[comp], E, correct, old_top_wrong, values),
                            "attention": attn_stats(
                                get_attn_slice(base_decomp.get("attention_pattern"), bp),
                                get_attn_slice(tdecomp.get("attention_pattern"), tpos),
                            ) if comp == "attn_out" else {},
                        }

                for spec in specs:
                    pkey = f"{pos_name}|L{source_layer}|{spec['name']}"
                    generated_down = mlp_parts_from_input(source_mlp, spec["target"])["down"] - base_down
                    gen_metric = projection_metric(generated_down, E, correct, old_top_wrong, values)
                    patched_scores = input_patched_score_map(model, tokenizer, device, case["base_prompt"], values, source_layer, bp, spec["target"])
                    patched = winner_stats(patched_scores, correct)
                    patched_decomp = collect_final_block(
                        model, tokenizer, device, case["base_prompt"], probe_layer,
                        source_layer, bp, spec["target"], capture_attn=args.capture_attn,
                    )
                    row["patches"][pkey] = {
                        "node": {"position": pos_name, "source_layer": source_layer, "probe_layer": probe_layer},
                        "trajectory": spec["kind"],
                        "winner": patched,
                        "generated_down_metric": gen_metric,
                        "full_metric": candidate_delta_metric(base_scores, patched_scores, correct, old_top_wrong),
                    }

                    for comp in COMPONENTS:
                        b = get_position(base_decomp, comp, bp)
                        p = get_position(patched_decomp, comp, bp)
                        if b is None or p is None:
                            continue
                        effect = p - b
                        key = f"{pos_name}|L{source_layer}|{spec['name']}|{comp}"
                        row["trajectory_components"][key] = {
                            "node": {"position": pos_name, "source_layer": source_layer, "probe_layer": probe_layer},
                            "trajectory": spec["kind"],
                            "component": comp,
                            "metric": effect_metric(effect, natural_effects.get(comp), E, correct, old_top_wrong, values),
                            "attention": attn_stats(
                                get_attn_slice(base_decomp.get("attention_pattern"), bp),
                                get_attn_slice(patched_decomp.get("attention_pattern"), bp),
                            ) if comp == "attn_out" else {},
                        }
            rows.append(row)

        return {
            "phase": 600,
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
            "capture_attn": args.capture_attn,
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
    parser.add_argument("--max-samples", type=int, default=48)
    parser.add_argument("--top-nodes", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
    parser.add_argument("--capture-attn", action="store_true", default=True)
    parser.add_argument("--no-capture-attn", dest="capture_attn", action="store_false")
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
        args.n_tables = max(args.n_tables, 12)
        args.max_samples = max(args.max_samples, 96)
        args.top_nodes = max(args.top_nodes, 3)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase600_{args.model}_final_layer_acceptance_rule_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
