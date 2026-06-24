#!/usr/bin/env python3
"""
Phase 604: Final Norm and Readout Acceptance Audit
最终归一化与读出接受审计

Phase 603 showed that attention-effect and final-MLP-output compensation can
raise trajectory similarity without opening full candidate margin. This phase
tests whether the remaining bottleneck is at final norm / lm_head readout.
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

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions, random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import answer_vectors, candidate_delta_metric  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import projection_metric, score_map  # noqa: E402
from phase598_downstream_trajectory_acceptance_audit import select_nodes  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_final_norm  # noqa: E402
from phase600_final_layer_acceptance_rule_audit import collect_final_block, get_position  # noqa: E402


OUT_ROOT = Path("results/glm5_phase604_final_norm_readout_acceptance_audit")
BETAS = [0.25, 0.5, 1.0, 1.5, 2.0]
LOCAL_COMPONENTS = ["final_norm_input", "final_norm_output"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prompt_last_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False)) - 1


def safe_norm(x: Optional[torch.Tensor]) -> float:
    if x is None:
        return 0.0
    return float(torch.linalg.vector_norm(x.float()).cpu())


def rms(x: Optional[torch.Tensor]) -> float:
    if x is None or x.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(x.float().cpu() ** 2)).item())


def cosine(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> float:
    if a is None or b is None:
        return 0.0
    a = a.float().cpu()
    b = b.float().cpu()
    if torch.linalg.vector_norm(a) < 1e-8 or torch.linalg.vector_norm(b) < 1e-8:
        return 0.0
    return float(F.cosine_similarity(a.view(1, -1), b.view(1, -1)).item())


def answer_ids(tokenizer, answer: str) -> List[int]:
    ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(answer, add_special_tokens=False)
    return ids


def score_first_token_map(model, tokenizer, device, prompt: str, candidates: List[str]) -> Dict[str, float]:
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
    pos = input_ids.shape[1] - 1
    with torch.inference_mode():
        out = model(input_ids=input_ids, return_dict=True)
        lp = torch.log_softmax(out.logits[0, pos].float(), dim=-1)
    scores = {}
    for ans in candidates:
        ids = answer_ids(tokenizer, ans)
        scores[ans] = float(lp[ids[0]].cpu()) if ids else -100.0
    return scores


def final_norm_sequence(model, tokenizer, device, prompt: str, answer: str) -> Dict[str, List[torch.Tensor]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    all_ids = prompt_ids + ans_ids
    start = len(prompt_ids) - 1
    final_norm = get_final_norm(model)
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def pre_hook(_module, inputs):
        captured["input"] = inputs[0].detach().float().cpu()

    def out_hook(_module, _inputs, output):
        captured["output"] = extract_tensor(output).detach().float().cpu()

    handles.append(final_norm.register_forward_pre_hook(pre_hook))
    handles.append(final_norm.register_forward_hook(out_hook))
    try:
        with torch.inference_mode():
            input_ids = torch.tensor([all_ids], device=device)
            model(input_ids=input_ids, return_dict=True)
    finally:
        for h in handles:
            h.remove()

    seq = {"input": [], "output": [], "positions": []}
    for i in range(len(ans_ids)):
        pos = start + i
        if pos >= captured["input"].shape[1]:
            break
        seq["input"].append(captured["input"][0, pos].float().cpu())
        seq["output"].append(captured["output"][0, pos].float().cpu())
        seq["positions"].append(pos)
    return seq


def patched_answer_logprobs(model, tokenizer, device, prompt: str, answer: str,
                            patch_kind: Optional[str],
                            patch_pos: Optional[int],
                            target_vec: Optional[torch.Tensor],
                            target_seq: Optional[List[torch.Tensor]] = None,
                            target_positions: Optional[List[int]] = None) -> Dict[str, float]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    if not ans_ids:
        return {"full": -100.0, "first": -100.0}
    if patch_kind is not None and target_seq is None and (patch_pos is None or patch_pos < 0 or patch_pos >= len(prompt_ids)):
        return {"full": -100.0, "first": -100.0}

    all_ids = prompt_ids + ans_ids
    final_norm = get_final_norm(model)
    handle = None
    if patch_kind is not None and (target_vec is not None or target_seq is not None):
        targets = None
        positions = None
        if target_seq is not None:
            targets = [t.to(device=device) for t in target_seq]
            positions = target_positions or [len(prompt_ids) - 1 + i for i in range(len(targets))]
        else:
            target = target_vec.to(device=device)

        if patch_kind == "input":
            def pre_hook(_module, inputs):
                x = inputs[0]
                x_new = x.clone()
                if targets is not None:
                    for p, t in zip(positions, targets):
                        if 0 <= p < x_new.shape[1]:
                            x_new[0, p, :] = t.to(dtype=x_new.dtype)
                else:
                    x_new[0, patch_pos, :] = target.to(dtype=x_new.dtype)
                return (x_new,) + tuple(inputs[1:])

            handle = final_norm.register_forward_pre_hook(pre_hook)
        elif patch_kind == "output":
            def out_hook(_module, _inputs, output):
                y = extract_tensor(output)
                y_new = y.clone()
                if targets is not None:
                    for p, t in zip(positions, targets):
                        if 0 <= p < y_new.shape[1]:
                            y_new[0, p, :] = t.to(dtype=y_new.dtype)
                else:
                    y_new[0, patch_pos, :] = target.to(dtype=y_new.dtype)
                if isinstance(output, tuple):
                    return (y_new,) + output[1:]
                return y_new

            handle = final_norm.register_forward_hook(out_hook)
        else:
            raise ValueError(f"unknown patch_kind={patch_kind}")

    try:
        total = 0.0
        first = None
        with torch.inference_mode():
            input_ids = torch.tensor([all_ids], device=device)
            out = model(input_ids=input_ids, return_dict=True)
            logits = out.logits[0].float()
            start = len(prompt_ids) - 1
            for i, tid in enumerate(ans_ids):
                pos = start + i
                if pos >= logits.shape[0]:
                    break
                lp = torch.log_softmax(logits[pos], dim=-1)
                val = float(lp[tid].cpu())
                if i == 0:
                    first = val
                total += val
        return {"full": total, "first": first if first is not None else -100.0}
    finally:
        if handle is not None:
            handle.remove()


def patched_candidate_scores(model, tokenizer, device, prompt: str, candidates: List[str],
                             patch_kind: Optional[str],
                             patch_pos: Optional[int],
                             target_vec: Optional[torch.Tensor]) -> Dict[str, Dict[str, float]]:
    full = {}
    first = {}
    for ans in candidates:
        scores = patched_answer_logprobs(model, tokenizer, device, prompt, ans, patch_kind, patch_pos, target_vec)
        full[ans] = scores["full"]
        first[ans] = scores["first"]
    return {"full": full, "first": first}


def patched_candidate_scores_sequence(model, tokenizer, device, prompt: str, candidates: List[str],
                                      patch_kind: str, target_by_answer: Dict[str, List[torch.Tensor]],
                                      pos_by_answer: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    full = {}
    first = {}
    for ans in candidates:
        scores = patched_answer_logprobs(
            model, tokenizer, device, prompt, ans,
            patch_kind=patch_kind,
            patch_pos=None,
            target_vec=None,
            target_seq=target_by_answer.get(ans),
            target_positions=pos_by_answer.get(ans),
        )
        full[ans] = scores["full"]
        first[ans] = scores["first"]
    return {"full": full, "first": first}


def local_component_metric(base_vec: torch.Tensor, repair_vec: torch.Tensor, E: torch.Tensor,
                           correct: str, old_top_wrong: str, values: List[str]) -> Dict:
    effect = repair_vec.float().cpu() - base_vec.float().cpu()
    return {
        "base_norm": safe_norm(base_vec),
        "repair_norm": safe_norm(repair_vec),
        "effect_norm": safe_norm(effect),
        "base_rms": rms(base_vec),
        "repair_rms": rms(repair_vec),
        "cos_base_repair": cosine(base_vec, repair_vec),
        **projection_metric(effect, E, correct, old_top_wrong, values),
    }


def summarize(rows: List[Dict]) -> Dict:
    ikeys = sorted({k for r in rows for k in r["interpolations"]})
    by_interp = {}
    for key in ikeys:
        items = [r["interpolations"][key] for r in rows if key in r["interpolations"]]
        entry = {
            "key": key,
            "kind": items[0]["kind"],
            "beta": items[0]["beta"],
            "n": len(items),
            "first_switch": 0,
            "full_switch": 0,
            "mean_first_margin_gain": 0.0,
            "mean_full_margin_gain": 0.0,
            "mean_correct_full_delta": 0.0,
            "mean_wrong_full_delta": 0.0,
            "mean_correct_first_delta": 0.0,
            "mean_wrong_first_delta": 0.0,
            "positive_full_margin": 0,
            "positive_first_margin": 0,
        }
        for item in items:
            entry["first_switch"] += int(item["first_winner"]["correct"])
            entry["full_switch"] += int(item["full_winner"]["correct"])
            fm = item["full_metric"]
            ft = item["first_metric"]
            entry["mean_full_margin_gain"] += fm["margin_gain"]
            entry["mean_first_margin_gain"] += ft["margin_gain"]
            entry["mean_correct_full_delta"] += fm["correct_delta"]
            entry["mean_wrong_full_delta"] += fm["old_top_wrong_delta"]
            entry["mean_correct_first_delta"] += ft["correct_delta"]
            entry["mean_wrong_first_delta"] += ft["old_top_wrong_delta"]
            entry["positive_full_margin"] += int(fm["margin_gain"] > 0)
            entry["positive_first_margin"] += int(ft["margin_gain"] > 0)
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["first_switch_rate"] = entry["first_switch"] / n
        entry["full_switch_rate"] = entry["full_switch"] / n
        entry["positive_full_margin_rate"] = entry["positive_full_margin"] / n
        entry["positive_first_margin_rate"] = entry["positive_first_margin"] / n
        by_interp[key] = entry

    lkeys = sorted({k for r in rows for k in r["local_readout"]})
    by_local = {}
    for key in lkeys:
        items = [r["local_readout"][key] for r in rows if key in r["local_readout"]]
        entry = {
            "key": key,
            "position": items[0]["position"],
            "component": items[0]["component"],
            "n": len(items),
            "mean_projection_margin": 0.0,
            "mean_effect_norm": 0.0,
            "mean_base_norm": 0.0,
            "mean_repair_norm": 0.0,
            "mean_cos_base_repair": 0.0,
        }
        for item in items:
            m = item["metric"]
            entry["mean_projection_margin"] += m["projection_specific_margin"]
            entry["mean_effect_norm"] += m["effect_norm"]
            entry["mean_base_norm"] += m["base_norm"]
            entry["mean_repair_norm"] += m["repair_norm"]
            entry["mean_cos_base_repair"] += m["cos_base_repair"]
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        by_local[key] = entry

    best_interp = sorted(
        by_interp.values(),
        key=lambda x: (x["full_switch"], x["mean_full_margin_gain"], x["mean_first_margin_gain"]),
        reverse=True,
    )[:80]
    best_local = sorted(
        by_local.values(),
        key=lambda x: (x["mean_projection_margin"], x["mean_effect_norm"]),
        reverse=True,
    )[:80]

    log("Best final-norm interpolation effects:")
    for item in best_interp[:12]:
        log(
            f"  {item['key']}: full={item['full_switch']}/{item['n']} "
            f"gain={item['mean_full_margin_gain']:.3f}, first={item['first_switch']}/{item['n']} "
            f"gain={item['mean_first_margin_gain']:.3f}"
        )
    log("Best local readout deltas:")
    for item in best_local[:8]:
        log(f"  {item['key']}: proj={item['mean_projection_margin']:.3f}, norm={item['mean_effect_norm']:.3f}")
    return {
        "by_interpolation": by_interp,
        "best_interpolation": best_interp,
        "by_local": by_local,
        "best_local": best_local,
    }


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        E = answer_vectors(model, tokenizer, values)
        cases = list(build_cases(args.n_tables, args.max_samples))
        probe_layer = info.n_layers - 1
        nodes = select_nodes(args.model, args.top_nodes)
        if not any(n["position"] == "prompt_last" for n in nodes):
            nodes.append({"position": "prompt_last", "layer": max(0, probe_layer - 1)})
        log(f"{args.model}: layers={info.n_layers}, cases={len(cases)}, probe=L{probe_layer}, nodes={[(n['position'], n['layer']) for n in nodes]}")

        rows = []
        target_seen = 0
        for si, case in enumerate(cases):
            correct = case["correct"]
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
            base_first = score_first_token_map(model, tokenizer, device, case["base_prompt"], values)
            repair_first = score_first_token_map(model, tokenizer, device, case["repair_prompt"], values)
            base = winner_stats(base_scores, correct)
            repair = winner_stats(repair_scores, correct)
            target_case = (not base["correct"]) and repair["correct"]
            if args.target_only and not target_case:
                continue
            target_seen += int(target_case)
            old_top_wrong = base["top_wrong"]

            base_pos = case_positions(tokenizer, case, case["base_prompt"], case["relation"])
            repair_pos = case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"])
            base_decomp = collect_final_block(model, tokenizer, device, case["base_prompt"], probe_layer, capture_attn=False)
            repair_decomp = collect_final_block(model, tokenizer, device, case["repair_prompt"], probe_layer, capture_attn=False)

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "repair_full_metric": candidate_delta_metric(base_scores, repair_scores, correct, old_top_wrong),
                "repair_first_metric": candidate_delta_metric(base_first, repair_first, correct, old_top_wrong),
                "local_readout": {},
                "interpolations": {},
            }

            for node in nodes:
                pos_name = node["position"]
                bp = base_pos.get(pos_name)
                rp = repair_pos.get(pos_name)
                if pos_name == "prompt_last":
                    bp = prompt_last_pos(tokenizer, case["base_prompt"])
                    rp = prompt_last_pos(tokenizer, case["repair_prompt"])
                if bp is None or rp is None:
                    continue
                for comp in LOCAL_COMPONENTS:
                    bv = get_position(base_decomp, comp, bp)
                    rv = get_position(repair_decomp, comp, rp)
                    if bv is None or rv is None:
                        continue
                    key = f"{pos_name}|{comp}"
                    row["local_readout"][key] = {
                        "position": pos_name,
                        "component": comp,
                        "metric": local_component_metric(bv, rv, E, correct, old_top_wrong, values),
                    }

            bp_last = prompt_last_pos(tokenizer, case["base_prompt"])
            rp_last = prompt_last_pos(tokenizer, case["repair_prompt"])
            base_fni = get_position(base_decomp, "final_norm_input", bp_last)
            repair_fni = get_position(repair_decomp, "final_norm_input", rp_last)
            base_fno = get_position(base_decomp, "final_norm_output", bp_last)
            repair_fno = get_position(repair_decomp, "final_norm_output", rp_last)
            if base_fni is None or repair_fni is None or base_fno is None or repair_fno is None:
                rows.append(row)
                continue

            input_delta = repair_fni.float().cpu() - base_fni.float().cpu()
            output_delta = repair_fno.float().cpu() - base_fno.float().cpu()
            random_input = random_same_norm(input_delta, seed=si * 1009 + 17)
            random_output = random_same_norm(output_delta, seed=si * 1009 + 19)

            specs = []
            for beta in args.betas:
                specs.extend([
                    ("input_interp", beta, "input", base_fni.float().cpu() + beta * input_delta),
                    ("output_interp", beta, "output", base_fno.float().cpu() + beta * output_delta),
                    ("input_random", beta, "input", base_fni.float().cpu() + beta * random_input),
                    ("output_random", beta, "output", base_fno.float().cpu() + beta * random_output),
                ])

            for kind, beta, patch_kind, target_vec in specs:
                scores = patched_candidate_scores(
                    model, tokenizer, device, case["base_prompt"], values,
                    patch_kind, bp_last, target_vec,
                )
                full_scores = scores["full"]
                first_scores = scores["first"]
                full_winner = winner_stats(full_scores, correct)
                first_winner = winner_stats(first_scores, correct)
                key = f"{kind}|beta{beta:g}"
                row["interpolations"][key] = {
                    "kind": kind,
                    "beta": beta,
                    "full_winner": full_winner,
                    "first_winner": first_winner,
                    "full_metric": candidate_delta_metric(base_scores, full_scores, correct, old_top_wrong),
                    "first_metric": candidate_delta_metric(base_first, first_scores, correct, old_top_wrong),
                }

            seq_cache = {}
            for ans in values:
                base_seq = final_norm_sequence(model, tokenizer, device, case["base_prompt"], ans)
                repair_seq = final_norm_sequence(model, tokenizer, device, case["repair_prompt"], ans)
                seq_cache[ans] = {
                    "base": base_seq,
                    "repair": repair_seq,
                    "input_delta": [
                        r.float().cpu() - b.float().cpu()
                        for b, r in zip(base_seq["input"], repair_seq["input"])
                    ],
                    "output_delta": [
                        r.float().cpu() - b.float().cpu()
                        for b, r in zip(base_seq["output"], repair_seq["output"])
                    ],
                    "random_input": [
                        random_same_norm(r.float().cpu() - b.float().cpu(), seed=si * 1009 + ai * 31 + 101)
                        for ai, (b, r) in enumerate(zip(base_seq["input"], repair_seq["input"]))
                    ],
                    "random_output": [
                        random_same_norm(r.float().cpu() - b.float().cpu(), seed=si * 1009 + ai * 31 + 103)
                        for ai, (b, r) in enumerate(zip(base_seq["output"], repair_seq["output"]))
                    ],
                }

            for beta in args.betas:
                seq_specs = [
                    ("seq_input_interp", "input", "input_delta"),
                    ("seq_output_interp", "output", "output_delta"),
                    ("seq_input_random", "input", "random_input"),
                    ("seq_output_random", "output", "random_output"),
                ]
                for kind, patch_kind, delta_name in seq_specs:
                    target_by_answer = {}
                    pos_by_answer = {}
                    for ans in values:
                        base_seq = seq_cache[ans]["base"]
                        base_part = base_seq[patch_kind]
                        deltas = seq_cache[ans][delta_name]
                        target_by_answer[ans] = [
                            b.float().cpu() + beta * d.float().cpu()
                            for b, d in zip(base_part, deltas)
                        ]
                        pos_by_answer[ans] = base_seq["positions"]
                    scores = patched_candidate_scores_sequence(
                        model, tokenizer, device, case["base_prompt"], values,
                        patch_kind, target_by_answer, pos_by_answer,
                    )
                    full_scores = scores["full"]
                    first_scores = scores["first"]
                    full_winner = winner_stats(full_scores, correct)
                    first_winner = winner_stats(first_scores, correct)
                    key = f"{kind}|beta{beta:g}"
                    row["interpolations"][key] = {
                        "kind": kind,
                        "beta": beta,
                        "full_winner": full_winner,
                        "first_winner": first_winner,
                        "full_metric": candidate_delta_metric(base_scores, full_scores, correct, old_top_wrong),
                        "first_metric": candidate_delta_metric(base_first, first_scores, correct, old_top_wrong),
                    }
            rows.append(row)

        return {
            "phase": 604,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "probe_layer": probe_layer,
            "n_cases": len(cases),
            "n_rows": len(rows),
            "n_target_cases_seen": target_seen,
            "target_only": args.target_only,
            "betas": args.betas,
            "nodes": nodes,
            "summary": summarize(rows),
            "rows": rows,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_betas(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--top-nodes", type=int, default=3)
    parser.add_argument("--betas", type=parse_betas, default=BETAS)
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
        args.betas = [1.0]
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
    out_path = out_dir / f"phase604_{args.model}_final_norm_readout_acceptance_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
