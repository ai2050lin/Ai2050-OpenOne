#!/usr/bin/env python3
"""
Phase 603: Post-Attention MLP Compensation Audit
注意力后 MLP 补偿审计

Phase 602 showed that natural attention effect can improve final hidden
trajectory similarity but not full candidate margin. This phase audits final
MLP gate/up/z/down compensation and tests one planned MLP-output compensation.
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
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions, random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import answer_vectors, candidate_delta_metric  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import (  # noqa: E402
    collect_mlp_input_output,
    get_mlp,
    mlp_parts_from_input,
    projection_metric,
    replace_input,
    score_map,
)
from phase598_downstream_trajectory_acceptance_audit import select_nodes  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn  # noqa: E402
from phase600_final_layer_acceptance_rule_audit import collect_final_block, get_position  # noqa: E402
from phase602_attention_source_factor_causal_patch import patched_scores  # noqa: E402


OUT_ROOT = Path("results/glm5_phase603_post_attention_mlp_compensation_audit")
MLP_PARTS = ["mlp_input", "gate", "up", "z", "down", "mlp_out", "layer_out", "final_norm_output"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.float()).cpu())


def cosine(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> float:
    if a is None or b is None:
        return 0.0
    a = a.float().cpu()
    b = b.float().cpu()
    if torch.linalg.vector_norm(a) < 1e-8 or torch.linalg.vector_norm(b) < 1e-8:
        return 0.0
    return float(F.cosine_similarity(a.view(1, -1), b.view(1, -1)).item())


def part_metric(effect: Optional[torch.Tensor], natural: Optional[torch.Tensor],
                E: torch.Tensor, correct: str, old_top_wrong: str, values: List[str]) -> Dict:
    if effect is None:
        return {
            "effect_norm": 0.0,
            "cos_to_natural": 0.0,
            "norm_ratio_to_natural": 0.0,
            "projection_specific_margin": 0.0,
        }
    out = {
        "effect_norm": safe_norm(effect),
        "cos_to_natural": cosine(effect, natural),
        "norm_ratio_to_natural": 0.0,
        "projection_specific_margin": 0.0,
    }
    if natural is not None:
        out["norm_ratio_to_natural"] = safe_norm(effect) / max(safe_norm(natural), 1e-8)
    if effect.numel() == E.shape[1]:
        out["projection_specific_margin"] = projection_metric(effect, E, correct, old_top_wrong, values)["projection_specific_margin"]
    return out


def final_mlp_parts(model, mlp_input: Optional[torch.Tensor], mlp_out: Optional[torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
    if mlp_input is None:
        return {k: None for k in ["mlp_input", "gate", "up", "z", "down", "mlp_out"]}
    mlp = get_mlp(get_layers(model)[-1])
    parts = mlp_parts_from_input(mlp, mlp_input)
    parts["mlp_input"] = mlp_input.float().cpu()
    parts["mlp_out"] = mlp_out.float().cpu() if mlp_out is not None else parts["down"]
    return parts


def patch_answer_with_mlpout(model, tokenizer, device, prompt: str, answer: str,
                             source_layer: Optional[int], source_pos: Optional[int],
                             target_input: Optional[torch.Tensor],
                             probe_layer: int,
                             attn_pos: Optional[int],
                             attn_delta: Optional[torch.Tensor],
                             attn_scale: float,
                             mlpout_pos: Optional[int],
                             mlpout_delta: Optional[torch.Tensor],
                             mlpout_scale: float) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids:
        return -100.0
    all_ids = prompt_ids + answer_ids
    layers = get_layers(model)
    handles = []

    if source_layer is not None and source_pos is not None and target_input is not None:
        if source_pos < 0 or source_pos >= len(prompt_ids):
            return -100.0
        source_mlp = get_mlp(layers[source_layer])
        target = target_input.to(device=device)

        def source_pre(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            x_new[0, source_pos, :] = target.to(dtype=x_new.dtype)
            return replace_input(inputs, x_new)

        handles.append(source_mlp.register_forward_pre_hook(source_pre))

    if attn_delta is not None and attn_pos is not None:
        attn = get_attn(layers[probe_layer])
        delta = (attn_scale * attn_delta.float().cpu()).to(device=device)

        def attn_hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            y_new[0, attn_pos, :] = y_new[0, attn_pos, :] + delta.to(dtype=y_new.dtype)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new

        handles.append(attn.register_forward_hook(attn_hook))

    if mlpout_delta is not None and mlpout_pos is not None:
        final_mlp = get_mlp(layers[probe_layer])
        delta = (mlpout_scale * mlpout_delta.float().cpu()).to(device=device)

        def mlp_out_hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            y_new[0, mlpout_pos, :] = y_new[0, mlpout_pos, :] + delta.to(dtype=y_new.dtype)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new

        handles.append(final_mlp.register_forward_hook(mlp_out_hook))

    try:
        total = 0.0
        with torch.inference_mode():
            input_ids = torch.tensor([all_ids], device=device)
            out = model(input_ids=input_ids, return_dict=True)
            logits = out.logits[0].float()
            start = len(prompt_ids) - 1
            for i, tid in enumerate(answer_ids):
                pos = start + i
                if pos >= logits.shape[0]:
                    break
                total += float(torch.log_softmax(logits[pos], dim=-1)[tid].cpu())
        return total
    finally:
        for h in handles:
            h.remove()


def patched_scores_with_mlpout(model, tokenizer, device, prompt: str, values: List[str],
                               source_layer: Optional[int], source_pos: Optional[int],
                               target_input: Optional[torch.Tensor],
                               probe_layer: int,
                               attn_pos: Optional[int],
                               attn_delta: Optional[torch.Tensor],
                               attn_scale: float,
                               mlpout_pos: Optional[int],
                               mlpout_delta: Optional[torch.Tensor],
                               mlpout_scale: float) -> Dict[str, float]:
    return {
        ans: patch_answer_with_mlpout(
            model, tokenizer, device, prompt, ans,
            source_layer, source_pos, target_input,
            probe_layer, attn_pos, attn_delta, attn_scale,
            mlpout_pos, mlpout_delta, mlpout_scale,
        )
        for ans in values
    }


def summarize(rows: List[Dict]) -> Dict:
    dkeys = sorted({k for r in rows for k in r["diagnostics"]})
    by_diag = {}
    for key in dkeys:
        items = [r["diagnostics"][key] for r in rows if key in r["diagnostics"]]
        entry = {
            "key": key,
            "mode": items[0]["mode"],
            "part": items[0]["part"],
            "position": items[0]["node"]["position"],
            "source_layer": items[0]["node"]["source_layer"],
            "n": len(items),
            "mean_cos_to_natural": 0.0,
            "mean_norm_ratio": 0.0,
            "mean_projection_specific_margin": 0.0,
        }
        for item in items:
            m = item["metric"]
            entry["mean_cos_to_natural"] += m["cos_to_natural"]
            entry["mean_norm_ratio"] += m["norm_ratio_to_natural"]
            entry["mean_projection_specific_margin"] += m["projection_specific_margin"]
        n = max(1, len(items))
        for name in ["mean_cos_to_natural", "mean_norm_ratio", "mean_projection_specific_margin"]:
            entry[name] /= n
        by_diag[key] = entry

    pkeys = sorted({k for r in rows for k in r["patches"]})
    by_patch = {}
    for key in pkeys:
        items = [r["patches"][key] for r in rows if key in r["patches"]]
        entry = {
            "key": key,
            "mode": items[0]["mode"],
            "position": items[0]["node"]["position"],
            "source_layer": items[0]["node"]["source_layer"],
            "n": len(items),
            "switch": 0,
            "mean_full_margin_gain": 0.0,
            "positive_margin": 0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            entry["mean_full_margin_gain"] += item["full_metric"]["margin_gain"]
            entry["positive_margin"] += int(item["full_metric"]["margin_gain"] > 0)
        n = max(1, len(items))
        entry["mean_full_margin_gain"] /= n
        entry["switch_rate"] = entry["switch"] / n
        entry["positive_margin_rate"] = entry["positive_margin"] / n
        by_patch[key] = entry

    best_diag = sorted(
        by_diag.values(),
        key=lambda x: (x["mean_cos_to_natural"], x["mean_projection_specific_margin"]),
        reverse=True,
    )[:80]
    best_patch = sorted(
        by_patch.values(),
        key=lambda x: (x["switch"], x["mean_full_margin_gain"]),
        reverse=True,
    )[:40]
    log("Best MLP compensation diagnostics:")
    for item in best_diag[:10]:
        log(f"  {item['key']}: cos={item['mean_cos_to_natural']:.3f}, proj={item['mean_projection_specific_margin']:.3f}")
    log("Best compensation patch effects:")
    for item in best_patch[:8]:
        log(f"  {item['key']}: switch={item['switch']}/{item['n']}, full={item['mean_full_margin_gain']:.3f}")
    return {"by_diag": by_diag, "by_patch": by_patch, "best_diag": best_diag, "best_patch": best_patch}


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
            base = winner_stats(base_scores, correct)
            repair = winner_stats(repair_scores, correct)
            target_case = (not base["correct"]) and repair["correct"]
            if args.target_only and not target_case:
                continue
            target_seen += int(target_case)
            old_top_wrong = base["top_wrong"]

            base_pos = case_positions(tokenizer, case, case["base_prompt"], case["relation"])
            repair_pos = case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"])
            base_cap = collect_mlp_input_output(model, tokenizer, device, case["base_prompt"], source_layers)
            repair_cap = collect_mlp_input_output(model, tokenizer, device, case["repair_prompt"], source_layers)
            base_decomp = collect_final_block(model, tokenizer, device, case["base_prompt"], probe_layer, capture_attn=False)
            repair_decomp = collect_final_block(model, tokenizer, device, case["repair_prompt"], probe_layer, capture_attn=False)

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "diagnostics": {},
                "patches": {},
            }

            for node in nodes:
                pos_name = node["position"]
                source_layer = node["layer"]
                bp = base_pos.get(pos_name)
                rp = repair_pos.get(pos_name)
                if bp is None or rp is None:
                    continue
                if source_layer not in base_cap["mlp_input"] or source_layer not in repair_cap["mlp_input"]:
                    continue
                if bp >= base_cap["mlp_input"][source_layer].shape[1] or rp >= repair_cap["mlp_input"][source_layer].shape[1]:
                    continue

                base_x = base_cap["mlp_input"][source_layer][0, bp]
                repair_x = repair_cap["mlp_input"][source_layer][0, rp]
                d_repair = repair_x.float().cpu() - base_x.float().cpu()
                repair_target = base_x.float().cpu() + args.alpha * d_repair
                random_target = base_x.float().cpu() + args.alpha * random_same_norm(d_repair, seed=si * 1009 + source_layer)

                b_attn = get_position(base_decomp, "attn_out", bp)
                r_attn = get_position(repair_decomp, "attn_out", rp)
                if b_attn is None or r_attn is None:
                    continue
                attn_delta = (r_attn - b_attn).float().cpu()

                base_parts = final_mlp_parts(
                    model,
                    get_position(base_decomp, "mlp_input", bp),
                    get_position(base_decomp, "mlp_out", bp),
                )
                nat_parts = final_mlp_parts(
                    model,
                    get_position(repair_decomp, "mlp_input", rp),
                    get_position(repair_decomp, "mlp_out", rp),
                )
                natural_effects = {
                    "layer_out": get_position(repair_decomp, "layer_out", rp) - get_position(base_decomp, "layer_out", bp),
                    "final_norm_output": get_position(repair_decomp, "final_norm_output", rp) - get_position(base_decomp, "final_norm_output", bp),
                }
                for part in ["mlp_input", "gate", "up", "z", "down", "mlp_out"]:
                    if base_parts.get(part) is not None and nat_parts.get(part) is not None:
                        natural_effects[part] = nat_parts[part] - base_parts[part]

                modes = [
                    ("mlp_repair_only", repair_target, None),
                    ("mlp_plus_attn_effect", repair_target, attn_delta),
                    ("mlp_plus_attn_random", repair_target, random_same_norm(attn_delta, seed=si * 917 + source_layer)),
                    ("mlp_random_plus_attn_effect", random_target, attn_delta),
                ]
                patched_decomps = {}
                for mode, target, adelta in modes:
                    pdecomp = collect_final_block(
                        model, tokenizer, device, case["base_prompt"], probe_layer,
                        source_layer=source_layer,
                        patch_pos=bp,
                        target_input=target,
                        capture_attn=False,
                    )
                    if adelta is not None:
                        from phase602_attention_source_factor_causal_patch import collect_patched_final  # local import avoids cycle at module load
                        pdecomp = collect_patched_final(
                            model, tokenizer, device, case["base_prompt"], probe_layer,
                            source_layer, bp, target, bp, adelta, args.attn_scale,
                        )
                    patched_decomps[mode] = pdecomp

                    p_parts = final_mlp_parts(
                        model,
                        get_position(pdecomp, "mlp_input", bp),
                        get_position(pdecomp, "mlp_out", bp),
                    )
                    for part in ["mlp_input", "gate", "up", "z", "down", "mlp_out"]:
                        if base_parts.get(part) is None or p_parts.get(part) is None:
                            continue
                        effect = p_parts[part] - base_parts[part]
                        key = f"{pos_name}|L{source_layer}|{mode}|{part}"
                        row["diagnostics"][key] = {
                            "node": {"position": pos_name, "source_layer": source_layer, "probe_layer": probe_layer},
                            "mode": mode,
                            "part": part,
                            "metric": part_metric(effect, natural_effects.get(part), E, correct, old_top_wrong, values),
                        }
                    for part in ["layer_out", "final_norm_output"]:
                        b = get_position(base_decomp, part, bp)
                        p = get_position(pdecomp, part, bp)
                        if b is None or p is None:
                            continue
                        key = f"{pos_name}|L{source_layer}|{mode}|{part}"
                        row["diagnostics"][key] = {
                            "node": {"position": pos_name, "source_layer": source_layer, "probe_layer": probe_layer},
                            "mode": mode,
                            "part": part,
                            "metric": part_metric(p - b, natural_effects.get(part), E, correct, old_top_wrong, values),
                        }

                mlpout_delta = natural_effects.get("mlp_out")
                patch_modes = [
                    ("mlp_repair_only", repair_target, None, None),
                    ("mlp_plus_attn_effect", repair_target, attn_delta, None),
                    ("mlpout_effect_only", None, None, mlpout_delta),
                    ("mlp_plus_mlpout_effect", repair_target, None, mlpout_delta),
                    ("mlp_plus_attn_plus_mlpout_effect", repair_target, attn_delta, mlpout_delta),
                ]
                for mode, target, adelta, mdelta in patch_modes:
                    scores = patched_scores_with_mlpout(
                        model, tokenizer, device, case["base_prompt"], values,
                        source_layer if target is not None else None,
                        bp if target is not None else None,
                        target,
                        probe_layer,
                        bp,
                        adelta,
                        args.attn_scale,
                        bp,
                        mdelta,
                        args.mlpout_scale,
                    )
                    patched = winner_stats(scores, correct)
                    key = f"{pos_name}|L{source_layer}|{mode}"
                    row["patches"][key] = {
                        "node": {"position": pos_name, "source_layer": source_layer, "probe_layer": probe_layer},
                        "mode": mode,
                        "winner": patched,
                        "full_metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                    }
            rows.append(row)

        return {
            "phase": 603,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "probe_layer": probe_layer,
            "n_cases": len(cases),
            "n_target_cases_seen": target_seen,
            "n_rows": len(rows),
            "target_only": args.target_only,
            "alpha": args.alpha,
            "attn_scale": args.attn_scale,
            "mlpout_scale": args.mlpout_scale,
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
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--top-nodes", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--attn-scale", type=float, default=1.0)
    parser.add_argument("--mlpout-scale", type=float, default=1.0)
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
    out_path = out_dir / f"phase603_{args.model}_post_attention_mlp_compensation_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
