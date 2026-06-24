#!/usr/bin/env python3
"""
Phase 596: MLP Internal Gate Path Audit
MLP 内部门控路径审计

Phase 595 showed that final MLP output additive patch does not causally repair
candidate ranking. This phase moves one level inward and tests gate/up/gated
activation/down paths.
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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, compute_full_string_logprob_batch, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions, random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import answer_vectors, candidate_delta_metric  # noqa: E402
from phase595_mlp_update_causal_validation import load_phase594_mlp_nodes  # noqa: E402


OUT_ROOT = Path("results/glm5_phase596_mlp_internal_gate_path_audit")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_map(model, tokenizer, device, prompt: str, candidates: List[str]) -> Dict[str, float]:
    return {k: v[0] for k, v in compute_full_string_logprob_batch(model, tokenizer, device, prompt, candidates).items()}


def extract_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_tensor(output, tensor):
    if isinstance(output, tuple):
        return (tensor,) + output[1:]
    return tensor


def mlp_act(mlp, x: torch.Tensor) -> torch.Tensor:
    fn = getattr(mlp, "act_fn", None) or getattr(mlp, "activation_func", None)
    if fn is None:
        return F.silu(x)
    return fn(x)


def down_project(mlp, z: torch.Tensor) -> torch.Tensor:
    down = mlp.down_proj
    weight = down.weight.detach().float().cpu()
    bias = down.bias.detach().float().cpu() if getattr(down, "bias", None) is not None else None
    return F.linear(z.float().cpu(), weight, bias)


def split_gate_up(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    half = t.shape[-1] // 2
    return t[..., :half], t[..., half:]


def get_mlp(layer):
    return getattr(layer, "mlp", None)


def collect_mlp_internal(model, tokenizer, device, prompt: str, layer_indices: List[int]) -> Dict[str, Dict[int, torch.Tensor]]:
    layers = get_layers(model)
    captured: Dict[str, Dict[int, torch.Tensor]] = {
        "gate": {},
        "up": {},
        "gate_up": {},
        "z": {},
        "down_out": {},
    }
    hooks = []
    for li in layer_indices:
        mlp = get_mlp(layers[li])
        if mlp is None:
            continue

        if hasattr(mlp, "gate_proj"):
            hooks.append(mlp.gate_proj.register_forward_hook(
                lambda _m, _i, out, layer_idx=li: captured["gate"].__setitem__(
                    layer_idx, extract_tensor(out).detach().float().cpu()
                )
            ))
        if hasattr(mlp, "up_proj"):
            hooks.append(mlp.up_proj.register_forward_hook(
                lambda _m, _i, out, layer_idx=li: captured["up"].__setitem__(
                    layer_idx, extract_tensor(out).detach().float().cpu()
                )
            ))
        if hasattr(mlp, "gate_up_proj"):
            hooks.append(mlp.gate_up_proj.register_forward_hook(
                lambda _m, _i, out, layer_idx=li: captured["gate_up"].__setitem__(
                    layer_idx, extract_tensor(out).detach().float().cpu()
                )
            ))

        def make_down_pre(layer_idx):
            def pre_hook(_module, inputs):
                captured["z"][layer_idx] = inputs[0].detach().float().cpu()
            return pre_hook

        def make_down_out(layer_idx):
            def hook(_module, _inputs, output):
                captured["down_out"][layer_idx] = extract_tensor(output).detach().float().cpu()
            return hook

        hooks.append(mlp.down_proj.register_forward_pre_hook(make_down_pre(li)))
        hooks.append(mlp.down_proj.register_forward_hook(make_down_out(li)))
    try:
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
        with torch.inference_mode():
            model(input_ids=input_ids, return_dict=True)
    finally:
        for h in hooks:
            h.remove()
    return captured


def get_gate_up(captured: Dict[str, Dict[int, torch.Tensor]], li: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    gate = captured.get("gate", {}).get(li)
    up = captured.get("up", {}).get(li)
    fused = captured.get("gate_up", {}).get(li)
    if gate is not None and up is not None:
        return gate, up
    if fused is not None:
        return split_gate_up(fused)
    return None, None


def equivalent_effects(mlp, base_cap, repair_cap, li: int, bp: int, rp: int) -> Dict[str, torch.Tensor]:
    bg, bu = get_gate_up(base_cap, li)
    rg, ru = get_gate_up(repair_cap, li)
    bz = base_cap["z"].get(li)
    rz = repair_cap["z"].get(li)
    bd = base_cap["down_out"].get(li)
    rd = repair_cap["down_out"].get(li)
    effects = {}
    if bg is not None and bu is not None and rg is not None and ru is not None:
        bgv, buv = bg[0, bp], bu[0, bp]
        rgv, ruv = rg[0, rp], ru[0, rp]
        base_z = mlp_act(mlp, bgv) * buv
        effects["gate_only"] = down_project(mlp, mlp_act(mlp, rgv) * buv - base_z)
        effects["up_only"] = down_project(mlp, mlp_act(mlp, bgv) * ruv - base_z)
        effects["gate_up_pair"] = down_project(mlp, mlp_act(mlp, rgv) * ruv - base_z)
    if bz is not None and rz is not None:
        effects["z_pair"] = down_project(mlp, rz[0, rp] - bz[0, bp])
    if bd is not None and rd is not None:
        effects["down_out"] = rd[0, rp] - bd[0, bp]
    return effects


def specific_projection_metric(effect: torch.Tensor, E: torch.Tensor, correct: str, old_top_wrong: str,
                               values: List[str]) -> Dict:
    scores = E @ effect.float().cpu()
    common = float(scores.mean())
    ci = values.index(correct)
    wi = values.index(old_top_wrong)
    cs = float(scores[ci] - common)
    ws = float(scores[wi] - common)
    return {
        "projection_common": common,
        "projection_correct_specific": cs,
        "projection_old_top_wrong_specific": ws,
        "projection_specific_margin": cs - ws,
    }


def channel_topk(delta_z: torch.Tensor, mlp, E: torch.Tensor, correct: str, old_top_wrong: str,
                 values: List[str], k: int) -> torch.Tensor:
    weight = mlp.down_proj.weight.detach().float().cpu()
    ci = values.index(correct)
    wi = values.index(old_top_wrong)
    readout = E[ci] - E[wi]
    per_channel = readout.float().cpu() @ weight
    contrib = per_channel * delta_z.float().cpu()
    k = min(k, delta_z.numel())
    if k <= 0:
        return torch.zeros_like(delta_z)
    idx = torch.topk(contrib, k=k).indices
    out = torch.zeros_like(delta_z.float().cpu())
    out[idx] = delta_z.float().cpu()[idx]
    return out


def make_patch_specs(base_cap, repair_cap, wrong_cap, li: int, bp: int, rp: int, wp: Optional[int],
                     mlp, E: torch.Tensor, correct: str, old_top_wrong: str, values: List[str],
                     topks: List[int]) -> List[Dict]:
    specs = []
    bg, bu = get_gate_up(base_cap, li)
    rg, ru = get_gate_up(repair_cap, li)
    wg, wu = get_gate_up(wrong_cap, li) if wrong_cap is not None else (None, None)
    bz = base_cap["z"].get(li)
    rz = repair_cap["z"].get(li)
    wz = wrong_cap["z"].get(li) if wrong_cap is not None else None
    bd = base_cap["down_out"].get(li)
    rd = repair_cap["down_out"].get(li)

    if bg is not None and bu is not None and rg is not None and ru is not None:
        d_gate = rg[0, rp] - bg[0, bp]
        d_up = ru[0, rp] - bu[0, bp]
        if hasattr(mlp, "gate_proj"):
            specs.append({"name": "gate_raw", "kind": "gate", "vec": d_gate})
        if hasattr(mlp, "up_proj"):
            specs.append({"name": "up_raw", "kind": "up", "vec": d_up})
        if (hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj")) or hasattr(mlp, "gate_up_proj"):
            specs.extend([
                {"name": "gate_up_pair_raw", "kind": "gate_up_pair", "gate_vec": d_gate, "up_vec": d_up},
                {"name": "gate_up_pair_random", "kind": "gate_up_pair",
                 "gate_vec": random_same_norm(d_gate, seed=li * 97 + 1),
                 "up_vec": random_same_norm(d_up, seed=li * 97 + 2)},
            ])
        if wg is not None and wu is not None and wp is not None and wp < wg.shape[1] and wp < wu.shape[1]:
            specs.append({
                "name": "wrong_gate_up_pair_raw",
                "kind": "gate_up_pair",
                "gate_vec": wg[0, wp] - bg[0, bp],
                "up_vec": wu[0, wp] - bu[0, bp],
            })

    if bz is not None and rz is not None:
        dz = rz[0, rp] - bz[0, bp]
        specs.extend([
            {"name": "z_pair_raw", "kind": "z", "vec": dz},
            {"name": "z_pair_random", "kind": "z", "vec": random_same_norm(dz, seed=li * 193 + 3)},
        ])
        for k in topks:
            specs.append({"name": f"z_pair_top{k}", "kind": "z", "vec": channel_topk(dz, mlp, E, correct, old_top_wrong, values, k)})
        if wz is not None and wp is not None and wp < wz.shape[1]:
            specs.append({"name": "wrong_z_pair_raw", "kind": "z", "vec": wz[0, wp] - bz[0, bp]})

    if bd is not None and rd is not None:
        dd = rd[0, rp] - bd[0, bp]
        specs.extend([
            {"name": "down_out_raw", "kind": "down_out", "vec": dd},
            {"name": "down_out_random", "kind": "down_out", "vec": random_same_norm(dd, seed=li * 389 + 4)},
        ])
    return specs


def patch_full_logprob_internal(model, tokenizer, device, prompt: str, answer: str, layer_idx: int,
                                patch_pos: int, spec: Dict, alpha: float) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids or patch_pos < 0 or patch_pos >= len(prompt_ids):
        return -100.0
    all_ids = prompt_ids + answer_ids
    mlp = get_mlp(get_layers(model)[layer_idx])
    handles = []

    def output_add_hook(vec):
        v = vec.to(device=device)

        def hook(_module, _inputs, output):
            h = extract_tensor(output)
            h_new = h.clone()
            h_new[0, patch_pos, :] = h_new[0, patch_pos, :] + alpha * v.to(dtype=h_new.dtype)
            return replace_tensor(output, h_new)
        return hook

    def down_pre_add_hook(vec):
        v = vec.to(device=device)

        def hook(_module, inputs):
            z = inputs[0]
            z_new = z.clone()
            z_new[0, patch_pos, :] = z_new[0, patch_pos, :] + alpha * v.to(dtype=z_new.dtype)
            return (z_new,) + tuple(inputs[1:])
        return hook

    kind = spec["kind"]
    if kind == "gate" and hasattr(mlp, "gate_proj"):
        handles.append(mlp.gate_proj.register_forward_hook(output_add_hook(spec["vec"])))
    elif kind == "up" and hasattr(mlp, "up_proj"):
        handles.append(mlp.up_proj.register_forward_hook(output_add_hook(spec["vec"])))
    elif kind == "gate_up_pair":
        if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
            handles.append(mlp.gate_proj.register_forward_hook(output_add_hook(spec["gate_vec"])))
            handles.append(mlp.up_proj.register_forward_hook(output_add_hook(spec["up_vec"])))
        elif hasattr(mlp, "gate_up_proj"):
            fused = torch.cat([spec["gate_vec"].float().cpu(), spec["up_vec"].float().cpu()], dim=-1)
            handles.append(mlp.gate_up_proj.register_forward_hook(output_add_hook(fused)))
    elif kind == "z":
        handles.append(mlp.down_proj.register_forward_pre_hook(down_pre_add_hook(spec["vec"])))
    elif kind == "down_out":
        handles.append(mlp.down_proj.register_forward_hook(output_add_hook(spec["vec"])))
    else:
        return -100.0

    try:
        total = 0.0
        with torch.inference_mode():
            full_input = torch.tensor([all_ids], device=device)
            out = model(input_ids=full_input, return_dict=True)
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


def patched_score_map_internal(model, tokenizer, device, prompt: str, candidates: List[str],
                               layer_idx: int, patch_pos: int, spec: Dict, alpha: float) -> Dict[str, float]:
    return {
        ans: patch_full_logprob_internal(model, tokenizer, device, prompt, ans, layer_idx, patch_pos, spec, alpha)
        for ans in candidates
    }


def summarize(rows: List[Dict]) -> Dict:
    patch_keys = sorted({k for r in rows for k in r["patches"]})
    patch_by_key = {}
    for key in patch_keys:
        items = [r["patches"][key] for r in rows if key in r["patches"]]
        entry = {
            "key": key,
            "position": items[0]["node"]["position"],
            "layer": items[0]["node"]["layer"],
            "patch": items[0]["patch"],
            "n": len(items),
            "switch": 0,
            "mean_margin_gain": 0.0,
            "mean_specific_margin_gain": 0.0,
            "mean_common_delta": 0.0,
            "mean_correct_specific": 0.0,
            "mean_old_top_wrong_specific": 0.0,
            "positive_margin": 0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            m = item["metric"]
            entry["mean_margin_gain"] += m["margin_gain"]
            entry["mean_specific_margin_gain"] += m["specific_margin_gain"]
            entry["mean_common_delta"] += m["common_delta"]
            entry["mean_correct_specific"] += m["correct_specific"]
            entry["mean_old_top_wrong_specific"] += m["old_top_wrong_specific"]
            entry["positive_margin"] += int(m["margin_gain"] > 0)
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["switch_rate"] = entry["switch"] / n
        entry["positive_margin_rate"] = entry["positive_margin"] / n
        patch_by_key[key] = entry

    proj_keys = sorted({k for r in rows for k in r["projections"]})
    proj_by_key = {}
    for key in proj_keys:
        items = [r["projections"][key] for r in rows if key in r["projections"]]
        entry = {
            "key": key,
            "position": items[0]["node"]["position"],
            "layer": items[0]["node"]["layer"],
            "source": items[0]["source"],
            "n": len(items),
            "mean_projection_common": 0.0,
            "mean_projection_correct_specific": 0.0,
            "mean_projection_old_top_wrong_specific": 0.0,
            "mean_projection_specific_margin": 0.0,
            "positive_projection_margin": 0,
        }
        for item in items:
            m = item["metric"]
            for name in [
                "projection_common",
                "projection_correct_specific",
                "projection_old_top_wrong_specific",
                "projection_specific_margin",
            ]:
                entry[f"mean_{name}"] += m[name]
            entry["positive_projection_margin"] += int(m["projection_specific_margin"] > 0)
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["positive_projection_rate"] = entry["positive_projection_margin"] / n
        proj_by_key[key] = entry

    best_patches = sorted(
        patch_by_key.values(),
        key=lambda x: (x["switch"], x["mean_margin_gain"], x["mean_specific_margin_gain"]),
        reverse=True,
    )[:36]
    best_proj = sorted(
        proj_by_key.values(),
        key=lambda x: (x["mean_projection_specific_margin"], x["positive_projection_rate"]),
        reverse=True,
    )[:36]
    log("Best causal patches:")
    for item in best_patches[:12]:
        log(
            f"  {item['key']}: switch={item['switch']}/{item['n']}, "
            f"mgain={item['mean_margin_gain']:.3f}, spec={item['mean_specific_margin_gain']:.3f}, "
            f"common={item['mean_common_delta']:.3f}"
        )
    log("Best internal projections:")
    for item in best_proj[:8]:
        log(
            f"  {item['key']}: proj_spec={item['mean_projection_specific_margin']:.3f}, "
            f"rate={item['positive_projection_rate']:.3f}"
        )
    return {
        "patch_by_key": patch_by_key,
        "projection_by_key": proj_by_key,
        "best_patches": best_patches,
        "best_projections": best_proj,
    }


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
        topks = [int(x) for x in args.topks.split(",") if x.strip()]
        log(f"{args.model}: layers={info.n_layers}, cases={len(cases)}, nodes={[(n['position'], n['layer']) for n in nodes]}, topks={topks}")

        rows = []
        target_seen = 0
        layers = get_layers(model)
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
            wrong_scores = score_map(model, tokenizer, device, case["wrong_prompt"], values) if args.include_wrong_controls else None
            old_top_wrong = base["top_wrong"]
            base_pos = case_positions(tokenizer, case, case["base_prompt"], case["relation"])
            repair_pos = case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"])
            wrong_pos = case_positions(tokenizer, case, case["wrong_prompt"], case["wrong_rel"]) if args.include_wrong_controls else {}

            base_cap = collect_mlp_internal(model, tokenizer, device, case["base_prompt"], node_layers)
            repair_cap = collect_mlp_internal(model, tokenizer, device, case["repair_prompt"], node_layers)
            wrong_cap = collect_mlp_internal(model, tokenizer, device, case["wrong_prompt"], node_layers) if args.include_wrong_controls else None

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "wrong_prompt_scores": wrong_scores,
                "projections": {},
                "patches": {},
            }

            for node in nodes:
                pos_name = node["position"]
                li = node["layer"]
                mlp = get_mlp(layers[li])
                bp = base_pos.get(pos_name)
                rp = repair_pos.get(pos_name)
                wp = wrong_pos.get(pos_name) if args.include_wrong_controls else None
                if bp is None or rp is None:
                    continue
                ok = True
                for cap, p in [(base_cap, bp), (repair_cap, rp)]:
                    if li not in cap["z"] or p >= cap["z"][li].shape[1]:
                        ok = False
                if not ok:
                    continue

                effects = equivalent_effects(mlp, base_cap, repair_cap, li, bp, rp)
                for source, effect in effects.items():
                    key = f"{pos_name}|L{li}|{source}"
                    row["projections"][key] = {
                        "node": {"position": pos_name, "layer": li},
                        "source": source,
                        "metric": specific_projection_metric(effect, E, correct, old_top_wrong, values),
                    }

                patch_specs = make_patch_specs(
                    base_cap, repair_cap, wrong_cap, li, bp, rp, wp, mlp, E,
                    correct, old_top_wrong, values, topks
                )
                for spec in patch_specs:
                    if args.skip_down_out and spec["kind"] == "down_out":
                        continue
                    key = f"{pos_name}|L{li}|{spec['name']}"
                    patched_scores = patched_score_map_internal(
                        model, tokenizer, device, case["base_prompt"], values, li, bp, spec, args.alpha
                    )
                    patched = winner_stats(patched_scores, correct)
                    row["patches"][key] = {
                        "node": {"position": pos_name, "layer": li},
                        "patch": spec["name"],
                        "winner": patched,
                        "metric": candidate_delta_metric(base_scores, patched_scores, correct, old_top_wrong),
                    }
            rows.append(row)

        return {
            "phase": 596,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "n_cases": len(cases),
            "n_target_cases_seen": target_seen,
            "n_rows": len(rows),
            "target_only": args.target_only,
            "alpha": args.alpha,
            "topks": topks,
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
    parser.add_argument("--topks", default="32,128")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
    parser.add_argument("--include-wrong-controls", action="store_true", default=True)
    parser.add_argument("--skip-down-out", action="store_true")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 4
        args.top_nodes = min(args.top_nodes, 2)
        args.topks = "32"
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
    out_path = out_dir / f"phase596_{args.model}_mlp_internal_gate_path_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
