#!/usr/bin/env python3
"""
Phase 606: Digit1 Upstream Source Decomposition
第一位数字上游来源分解

Phase 605 localized the value gate to the first discriminative digit token. This
phase patches final-layer components at that digit1 prediction position and
audits source attention groups.
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
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions, random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import candidate_delta_metric  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import get_mlp, replace_input, score_map  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn, get_final_norm  # noqa: E402


OUT_ROOT = Path("results/glm5_phase606_digit1_upstream_source_decomposition")
COMPONENTS = ["layer_input", "attn_out", "mlp_out", "final_norm_input", "final_norm_output"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_ids(tokenizer, answer: str) -> List[int]:
    ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(answer, add_special_tokens=False)
    return ids


def digit1_pos(tokenizer, prompt: str) -> Optional[int]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    return len(ids) if len(ids) > 0 else None


def full_ids(tokenizer, prompt: str, answer: str) -> List[int]:
    return tokenizer.encode(prompt, add_special_tokens=False) + answer_ids(tokenizer, answer)


def group_positions(tokenizer, case: Dict, prompt: str, relation: str, answer: str) -> Dict[str, List[int]]:
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    pos = case_positions(tokenizer, case, prompt, relation)
    ans_len = len(answer_ids(tokenizer, answer))
    groups = {
        "prompt_last": [prompt_len - 1],
        "answer_prefix": [prompt_len] if ans_len >= 1 else [],
        "digit1_position": [prompt_len + 1] if ans_len >= 2 else [],
        "rule_value": [],
        "rule_relation": [],
        "query_relation": [],
        "object": [],
        "other": [],
    }
    for name in ["rule_value", "rule_relation", "query_relation", "object"]:
        p = pos.get(name)
        if p is not None:
            groups[name].append(p)
    known = {p for xs in groups.values() for p in xs if p is not None and p >= 0}
    total = prompt_len + ans_len
    groups["other"] = [i for i in range(total) if i not in known]
    return groups


def collect_digit1_components(model, tokenizer, device, prompt: str, answer: str,
                              probe_layer: int, capture_attn: bool = True) -> Dict:
    ids = full_ids(tokenizer, prompt, answer)
    dpos = digit1_pos(tokenizer, prompt)
    layers = get_layers(model)
    layer = layers[probe_layer]
    attn = get_attn(layer)
    mlp = get_mlp(layer)
    final_norm = get_final_norm(model)
    captured: Dict[str, torch.Tensor] = {}
    handles = []

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

    def norm_pre(_module, inputs):
        captured["final_norm_input"] = inputs[0].detach().float().cpu()

    def norm_out(_module, _inputs, output):
        captured["final_norm_output"] = extract_tensor(output).detach().float().cpu()

    handles.append(layer.register_forward_pre_hook(layer_pre))
    handles.append(layer.register_forward_hook(layer_out))
    if attn is not None:
        handles.append(attn.register_forward_hook(attn_out))
    if mlp is not None:
        handles.append(mlp.register_forward_pre_hook(mlp_pre))
        handles.append(mlp.register_forward_hook(mlp_out))
    handles.append(final_norm.register_forward_pre_hook(norm_pre))
    handles.append(final_norm.register_forward_hook(norm_out))
    try:
        with torch.inference_mode():
            input_ids = torch.tensor([ids], device=device)
            out = model(input_ids=input_ids, output_attentions=capture_attn, return_dict=True)
            captured["logits"] = out.logits.detach().float().cpu()
            if capture_attn and getattr(out, "attentions", None) is not None:
                if probe_layer < len(out.attentions) and out.attentions[probe_layer] is not None:
                    captured["attention_pattern"] = out.attentions[probe_layer].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()

    out = {"pos": dpos}
    for comp in COMPONENTS + ["layer_out", "mlp_input"]:
        t = captured.get(comp)
        if t is not None and dpos is not None and 0 <= dpos < t.shape[1]:
            out[comp] = t[0, dpos].float().cpu()
    attn_pat = captured.get("attention_pattern")
    if attn_pat is not None and dpos is not None and dpos < attn_pat.shape[2]:
        out["attention_slice"] = attn_pat[0, :, dpos, :].float().cpu()
    return out


def attention_group_mass(attn_slice: Optional[torch.Tensor], groups: Dict[str, List[int]]) -> Dict[str, float]:
    if attn_slice is None:
        return {}
    max_len = attn_slice.shape[-1]
    out = {}
    for name, positions in groups.items():
        valid = [p for p in positions if 0 <= p < max_len]
        if not valid:
            out[name] = 0.0
            continue
        out[name] = float(attn_slice[:, valid].sum(dim=-1).mean().cpu())
    return out


def patched_answer(model, tokenizer, device, prompt: str, answer: str, probe_layer: int,
                   component: str, pos: int, target: torch.Tensor) -> float:
    ids = full_ids(tokenizer, prompt, answer)
    ans_ids = answer_ids(tokenizer, answer)
    layers = get_layers(model)
    layer = layers[probe_layer]
    attn = get_attn(layer)
    mlp = get_mlp(layer)
    final_norm = get_final_norm(model)
    target = target.to(device=device)
    handle = None

    if component == "layer_input":
        def hook(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            x_new[0, pos, :] = target.to(dtype=x_new.dtype)
            return replace_input(inputs, x_new)
        handle = layer.register_forward_pre_hook(hook)
    elif component == "attn_out":
        def hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            y_new[0, pos, :] = target.to(dtype=y_new.dtype)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new
        handle = attn.register_forward_hook(hook)
    elif component == "mlp_out":
        def hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            y_new[0, pos, :] = target.to(dtype=y_new.dtype)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new
        handle = mlp.register_forward_hook(hook)
    elif component == "final_norm_input":
        def hook(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            x_new[0, pos, :] = target.to(dtype=x_new.dtype)
            return replace_input(inputs, x_new)
        handle = final_norm.register_forward_pre_hook(hook)
    elif component == "final_norm_output":
        def hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            y_new[0, pos, :] = target.to(dtype=y_new.dtype)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new
        handle = final_norm.register_forward_hook(hook)
    else:
        raise ValueError(component)

    try:
        total = 0.0
        with torch.inference_mode():
            input_ids = torch.tensor([ids], device=device)
            logits = model(input_ids=input_ids, return_dict=True).logits[0].float()
            start = len(tokenizer.encode(prompt, add_special_tokens=False)) - 1
            for i, tid in enumerate(ans_ids):
                p = start + i
                if p >= logits.shape[0]:
                    break
                total += float(torch.log_softmax(logits[p], dim=-1)[tid].cpu())
        return total
    finally:
        if handle is not None:
            handle.remove()


def patched_scores(model, tokenizer, device, prompt: str, values: List[str], probe_layer: int,
                   component: str, comp_cache: Dict[str, Dict], random_mode: bool, seed: int) -> Dict[str, float]:
    out = {}
    for ai, ans in enumerate(values):
        base_comp = comp_cache[ans]["base"].get(component)
        repair_comp = comp_cache[ans]["repair"].get(component)
        pos = comp_cache[ans]["base"]["pos"]
        if base_comp is None or repair_comp is None or pos is None:
            out[ans] = -100.0
            continue
        delta = repair_comp.float().cpu() - base_comp.float().cpu()
        if random_mode:
            delta = random_same_norm(delta, seed=seed + ai * 101)
        target = base_comp.float().cpu() + delta
        out[ans] = patched_answer(model, tokenizer, device, prompt, ans, probe_layer, component, pos, target)
    return out


def summarize(rows: List[Dict]) -> Dict:
    keys = sorted({k for r in rows for k in r["patches"]})
    by_patch = {}
    for key in keys:
        items = [r["patches"][key] for r in rows if key in r["patches"]]
        entry = {
            "key": key,
            "component": items[0]["component"],
            "random": items[0]["random"],
            "n": len(items),
            "switch": 0,
            "mean_margin_gain": 0.0,
            "mean_correct_delta": 0.0,
            "mean_wrong_delta": 0.0,
            "positive_margin": 0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            m = item["metric"]
            entry["mean_margin_gain"] += m["margin_gain"]
            entry["mean_correct_delta"] += m["correct_delta"]
            entry["mean_wrong_delta"] += m["old_top_wrong_delta"]
            entry["positive_margin"] += int(m["margin_gain"] > 0)
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["switch_rate"] = entry["switch"] / n
        entry["positive_margin_rate"] = entry["positive_margin"] / n
        by_patch[key] = entry

    source_keys = sorted({k for r in rows for k in r["attention_mass_delta"]})
    by_source = {}
    for key in source_keys:
        vals = [r["attention_mass_delta"][key] for r in rows if key in r["attention_mass_delta"]]
        by_source[key] = {"source": key, "n": len(vals), "mean_delta": sum(vals) / max(1, len(vals))}

    best = sorted(by_patch.values(), key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)[:80]
    source_best = sorted(by_source.values(), key=lambda x: abs(x["mean_delta"]), reverse=True)
    log("Best digit1 upstream patches:")
    for item in best[:12]:
        log(f"  {item['key']}: switch={item['switch']}/{item['n']}, margin={item['mean_margin_gain']:.3f}")
    log("Largest attention source mass deltas:")
    for item in source_best[:8]:
        log(f"  {item['source']}: delta={item['mean_delta']:.4f}")
    return {"by_patch": by_patch, "best_patch": best, "by_source": by_source, "source_best": source_best}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        probe_layer = info.n_layers - 1
        values = CANDIDATE_VALUES[:4]
        cases = list(build_cases(args.n_tables, args.max_samples))
        rows = []
        target_seen = 0
        log(f"{args.model}: layers={info.n_layers}, cases={len(cases)}, probe=L{probe_layer}, values={values}")

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

            comp_cache = {}
            for ans in values:
                comp_cache[ans] = {
                    "base": collect_digit1_components(model, tokenizer, device, case["base_prompt"], ans, probe_layer, capture_attn=True),
                    "repair": collect_digit1_components(model, tokenizer, device, case["repair_prompt"], ans, probe_layer, capture_attn=True),
                }

            groups_base = group_positions(tokenizer, case, case["base_prompt"], case["relation"], correct)
            groups_repair = group_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"], correct)
            b_mass = attention_group_mass(comp_cache[correct]["base"].get("attention_slice"), groups_base)
            r_mass = attention_group_mass(comp_cache[correct]["repair"].get("attention_slice"), groups_repair)
            mass_delta = {k: r_mass.get(k, 0.0) - b_mass.get(k, 0.0) for k in sorted(set(b_mass) | set(r_mass))}

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "repair_metric": candidate_delta_metric(base_scores, repair_scores, correct, old_top_wrong),
                "attention_mass_base": b_mass,
                "attention_mass_repair": r_mass,
                "attention_mass_delta": mass_delta,
                "patches": {},
            }
            for comp in COMPONENTS:
                for random_mode in [False, True]:
                    if random_mode and comp not in ("layer_input", "attn_out", "mlp_out", "final_norm_input"):
                        continue
                    key = f"{comp}{'_random' if random_mode else ''}"
                    scores = patched_scores(
                        model, tokenizer, device, case["base_prompt"], values, probe_layer,
                        comp, comp_cache, random_mode=random_mode, seed=si * 1009 + len(comp),
                    )
                    patched = winner_stats(scores, correct)
                    row["patches"][key] = {
                        "component": comp,
                        "random": random_mode,
                        "winner": patched,
                        "metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                    }
            rows.append(row)

        return {
            "phase": 606,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "probe_layer": probe_layer,
            "n_cases": len(cases),
            "n_rows": len(rows),
            "n_target_cases_seen": target_seen,
            "target_only": args.target_only,
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
        args.n_tables = max(args.n_tables, 12)
        args.max_samples = max(args.max_samples, 96)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase606_{args.model}_digit1_upstream_source_decomposition_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
