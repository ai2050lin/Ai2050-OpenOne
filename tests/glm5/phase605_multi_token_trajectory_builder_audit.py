#!/usr/bin/env python3
"""
Phase 605: Multi-Token Trajectory Builder Audit
多词元轨迹构造器审计

Phase 604 showed that full value recovery requires sequence-level final-norm
trajectory patching. This phase localizes which answer-token steps carry the
candidate competition: common prefix, first digit, second digit, or digit span.
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

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import candidate_delta_metric  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_final_norm  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402


OUT_ROOT = Path("results/glm5_phase605_multi_token_trajectory_builder_audit")
TOKEN_GROUPS = {
    "prefix0": [0],
    "digit1": [1],
    "digit2": [2],
    "digits": [1, 2],
    "all": [0, 1, 2],
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_ids(tokenizer, answer: str) -> List[int]:
    ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(answer, add_special_tokens=False)
    return ids


def token_labels(tokenizer, answer: str) -> List[str]:
    return [tokenizer.decode([i]) for i in answer_ids(tokenizer, answer)]


def capture_final_norm_sequence(model, tokenizer, device, prompt: str, answer: str) -> Dict:
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
            out = model(input_ids=input_ids, return_dict=True)
            logits = out.logits[0].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()

    seq = {"input": [], "output": [], "positions": [], "token_logprobs": [], "token_ids": ans_ids}
    for i, tid in enumerate(ans_ids):
        pos = start + i
        if pos >= captured["input"].shape[1] or pos >= logits.shape[0]:
            break
        seq["input"].append(captured["input"][0, pos].float().cpu())
        seq["output"].append(captured["output"][0, pos].float().cpu())
        seq["positions"].append(pos)
        lp = torch.log_softmax(logits[pos].float(), dim=-1)
        seq["token_logprobs"].append(float(lp[tid]))
    return seq


def patched_answer(model, tokenizer, device, prompt: str, answer: str, patch_kind: str,
                   target_seq: List[torch.Tensor], target_positions: List[int]) -> Dict:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    all_ids = prompt_ids + ans_ids
    final_norm = get_final_norm(model)
    targets = [t.to(device=device) for t in target_seq]
    positions = list(target_positions)

    if patch_kind == "input":
        def pre_hook(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            for p, t in zip(positions, targets):
                if 0 <= p < x_new.shape[1]:
                    x_new[0, p, :] = t.to(dtype=x_new.dtype)
            return (x_new,) + tuple(inputs[1:])

        handle = final_norm.register_forward_pre_hook(pre_hook)
    elif patch_kind == "output":
        def out_hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            for p, t in zip(positions, targets):
                if 0 <= p < y_new.shape[1]:
                    y_new[0, p, :] = t.to(dtype=y_new.dtype)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new

        handle = final_norm.register_forward_hook(out_hook)
    else:
        raise ValueError(patch_kind)

    try:
        total = 0.0
        token_lps = []
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
                token_lps.append(val)
                total += val
        return {"full": total, "token_logprobs": token_lps}
    finally:
        handle.remove()


def build_targets(base_seq: Dict, repair_seq: Dict, group: List[int], kind: str,
                  random_mode: bool, seed: int) -> Dict:
    targets = []
    positions = []
    source = kind
    max_len = min(len(base_seq[source]), len(repair_seq[source]))
    for idx in group:
        if idx >= max_len:
            continue
        base_v = base_seq[source][idx].float().cpu()
        delta = repair_seq[source][idx].float().cpu() - base_v
        if random_mode:
            delta = random_same_norm(delta, seed=seed + idx * 31)
        targets.append(base_v + delta)
        positions.append(base_seq["positions"][idx])
    return {"targets": targets, "positions": positions}


def score_patch_mode(model, tokenizer, device, prompt: str, values: List[str], patch_kind: str,
                     seq_cache: Dict[str, Dict], group: List[int], random_mode: bool, seed: int) -> Dict:
    full = {}
    token_lps = {}
    for ai, ans in enumerate(values):
        t = build_targets(
            seq_cache[ans]["base"],
            seq_cache[ans]["repair"],
            group,
            patch_kind,
            random_mode=random_mode,
            seed=seed + ai * 101,
        )
        if not t["targets"]:
            full[ans] = -100.0
            token_lps[ans] = []
            continue
        out = patched_answer(model, tokenizer, device, prompt, ans, patch_kind, t["targets"], t["positions"])
        full[ans] = out["full"]
        token_lps[ans] = out["token_logprobs"]
    return {"full": full, "token_logprobs": token_lps}


def summarize(rows: List[Dict]) -> Dict:
    keys = sorted({k for r in rows for k in r["patches"]})
    by_patch = {}
    for key in keys:
        items = [r["patches"][key] for r in rows if key in r["patches"]]
        entry = {
            "key": key,
            "kind": items[0]["kind"],
            "group": items[0]["group"],
            "random": items[0]["random"],
            "n": len(items),
            "switch": 0,
            "mean_margin_gain": 0.0,
            "mean_correct_delta": 0.0,
            "mean_wrong_delta": 0.0,
            "positive_margin": 0,
        }
        max_steps = max((len(x["token_delta_correct"]) for x in items), default=0)
        for step in range(max_steps):
            entry[f"mean_token{step}_correct_delta"] = 0.0
            entry[f"mean_token{step}_wrong_delta"] = 0.0
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            m = item["metric"]
            entry["mean_margin_gain"] += m["margin_gain"]
            entry["mean_correct_delta"] += m["correct_delta"]
            entry["mean_wrong_delta"] += m["old_top_wrong_delta"]
            entry["positive_margin"] += int(m["margin_gain"] > 0)
            for step in range(max_steps):
                if step < len(item["token_delta_correct"]):
                    entry[f"mean_token{step}_correct_delta"] += item["token_delta_correct"][step]
                if step < len(item["token_delta_wrong"]):
                    entry[f"mean_token{step}_wrong_delta"] += item["token_delta_wrong"][step]
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["switch_rate"] = entry["switch"] / n
        entry["positive_margin_rate"] = entry["positive_margin"] / n
        by_patch[key] = entry

    best = sorted(by_patch.values(), key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)[:80]
    log("Best token-step trajectory patches:")
    for item in best[:14]:
        log(f"  {item['key']}: switch={item['switch']}/{item['n']}, margin={item['mean_margin_gain']:.3f}")
    return {"by_patch": by_patch, "best": best}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        cases = list(build_cases(args.n_tables, args.max_samples))
        rows = []
        target_seen = 0
        log(f"{args.model}: layers={info.n_layers}, cases={len(cases)}, values={values}")

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

            seq_cache = {}
            for ans in values:
                b = capture_final_norm_sequence(model, tokenizer, device, case["base_prompt"], ans)
                r = capture_final_norm_sequence(model, tokenizer, device, case["repair_prompt"], ans)
                seq_cache[ans] = {"base": b, "repair": r}

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "tokenization": {ans: token_labels(tokenizer, ans) for ans in values},
                "base": base,
                "repair_prompt": repair,
                "repair_metric": candidate_delta_metric(base_scores, repair_scores, correct, old_top_wrong),
                "patches": {},
            }

            base_correct_lps = seq_cache[correct]["base"]["token_logprobs"]
            base_wrong_lps = seq_cache[old_top_wrong]["base"]["token_logprobs"]
            for kind in ["input", "output"]:
                for group_name, group in TOKEN_GROUPS.items():
                    for random_mode in [False, True]:
                        if random_mode and group_name not in ("digits", "all"):
                            continue
                        mode = f"{kind}_{group_name}{'_random' if random_mode else ''}"
                        scores = score_patch_mode(
                            model, tokenizer, device, case["base_prompt"], values,
                            kind, seq_cache, group, random_mode=random_mode, seed=si * 1009 + len(mode),
                        )
                        full_scores = scores["full"]
                        winner = winner_stats(full_scores, correct)
                        correct_lps = scores["token_logprobs"].get(correct, [])
                        wrong_lps = scores["token_logprobs"].get(old_top_wrong, [])
                        token_delta_correct = [
                            correct_lps[i] - base_correct_lps[i]
                            for i in range(min(len(correct_lps), len(base_correct_lps)))
                        ]
                        token_delta_wrong = [
                            wrong_lps[i] - base_wrong_lps[i]
                            for i in range(min(len(wrong_lps), len(base_wrong_lps)))
                        ]
                        row["patches"][mode] = {
                            "kind": kind,
                            "group": group_name,
                            "random": random_mode,
                            "winner": winner,
                            "metric": candidate_delta_metric(base_scores, full_scores, correct, old_top_wrong),
                            "token_delta_correct": token_delta_correct,
                            "token_delta_wrong": token_delta_wrong,
                        }
            rows.append(row)

        return {
            "phase": 605,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
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
    out_path = out_dir / f"phase605_{args.model}_multi_token_trajectory_builder_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
