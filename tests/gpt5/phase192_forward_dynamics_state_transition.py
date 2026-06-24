#!/usr/bin/env python3
"""Phase192: forward dynamics state-transition trajectory audit.

After Phase190/191 downgraded static z-patch routes, this script measures how
candidate margins evolve across layers for base/repair/wrong trajectories.
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
from statistics import mean
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "glm5"))

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, compute_full_string_logprob_batch, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import answer_vectors  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase192_forward_dynamics_state_transition")
DEFAULT_POSITIONS = "prompt_last,query_relation,rule_value,query_category,rule_relation"


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_map(model, tokenizer, device, prompt: str, candidates: list[str]) -> dict[str, float]:
    return {k: v[0] for k, v in compute_full_string_logprob_batch(model, tokenizer, device, prompt, candidates).items()}


def final_norm(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    return None


def route_positions(tokenizer, case: dict[str, Any]) -> dict[str, dict[str, int | None]]:
    return {
        "base": case_positions(tokenizer, case, case["base_prompt"], case["relation"]),
        "repair": case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"]),
        "wrong": case_positions(tokenizer, case, case["wrong_prompt"], case["wrong_rel"]),
    }


def collect_route_margins(model, tokenizer, device, prompt: str, positions: dict[str, int | None],
                          pos_names: list[str], E: torch.Tensor, correct: str,
                          old_top_wrong: str, values: list[str]) -> dict[str, Any]:
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
    ci = values.index(correct)
    wi = values.index(old_top_wrong)
    norm = final_norm(model)
    E_dev = E.to(device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        hidden_states = out.hidden_states[1:]
    n_layers = len(hidden_states)
    result: dict[str, Any] = {}
    for pos_name in pos_names:
        pos = positions.get(pos_name)
        if pos is None or pos < 0 or pos >= input_ids.shape[1]:
            continue
        hs = torch.stack([hidden_states[li][0, pos, :] for li in range(n_layers)], dim=0)
        if norm is not None:
            hs = norm(hs)
        scores = hs.float() @ E_dev.T.float()
        margin = scores[:, ci] - scores[:, wi]
        common = scores.mean(dim=1)
        correct_specific = scores[:, ci] - common
        old_specific = scores[:, wi] - common
        result[pos_name] = {
            "pos": int(pos),
            "margin": [float(x) for x in margin.detach().cpu()],
            "correct_specific": [float(x) for x in correct_specific.detach().cpu()],
            "old_top_wrong_specific": [float(x) for x in old_specific.detach().cpu()],
            "common": [float(x) for x in common.detach().cpu()],
        }
    del out, hidden_states
    return result


def first_idx(xs: list[float], pred) -> int | None:
    for i, x in enumerate(xs):
        if pred(x):
            return i
    return None


def trajectory_metrics(base: list[float], repair: list[float], wrong: list[float],
                       delta: float) -> dict[str, Any]:
    n = min(len(base), len(repair), len(wrong))
    b = base[:n]
    r = repair[:n]
    w = wrong[:n]
    repair_minus_base = [r[i] - b[i] for i in range(n)]
    wrong_minus_base = [w[i] - b[i] for i in range(n)]
    specificity = [
        repair_minus_base[i] / (abs(wrong_minus_base[i]) + 1e-6)
        for i in range(n)
    ]
    repair_transition = [r[i + 1] - r[i] for i in range(n - 1)]
    base_transition = [b[i + 1] - b[i] for i in range(n - 1)]
    wrong_transition = [w[i + 1] - w[i] for i in range(n - 1)]
    transition_adv = [
        repair_transition[i] - max(base_transition[i], wrong_transition[i])
        for i in range(n - 1)
    ]
    repair_over_base_layer = first_idx(repair_minus_base, lambda x: x > delta)
    strict_flip_layer = None
    for i in range(n):
        if r[i] > 0 and b[i] < 0:
            strict_flip_layer = i
            break
    best_rb_layer = max(range(n), key=lambda i: repair_minus_base[i]) if n else None
    best_spec_layer = max(range(n), key=lambda i: specificity[i]) if n else None
    best_transition_layer = max(range(n - 1), key=lambda i: transition_adv[i]) if n > 1 else None
    return {
        "n_layers": n,
        "repair_over_base_layer": repair_over_base_layer,
        "strict_flip_layer": strict_flip_layer,
        "best_repair_minus_base_layer": best_rb_layer,
        "best_repair_minus_base": repair_minus_base[best_rb_layer] if best_rb_layer is not None else None,
        "best_specificity_layer": best_spec_layer,
        "best_specificity": specificity[best_spec_layer] if best_spec_layer is not None else None,
        "best_transition_layer": best_transition_layer,
        "best_transition_advantage": transition_adv[best_transition_layer] if best_transition_layer is not None else None,
        "final_base_margin": b[-1] if n else None,
        "final_repair_margin": r[-1] if n else None,
        "final_wrong_margin": w[-1] if n else None,
        "final_repair_minus_base": repair_minus_base[-1] if n else None,
        "final_wrong_minus_base": wrong_minus_base[-1] if n else None,
        "repair_minus_base": repair_minus_base,
        "wrong_minus_base": wrong_minus_base,
        "specificity": specificity,
        "transition_advantage": transition_adv,
    }


def avg_curve(curves: list[list[float]]) -> list[float]:
    if not curves:
        return []
    n = min(len(c) for c in curves)
    return [mean(c[i] for c in curves) for i in range(n)]


def summarize(rows: list[dict[str, Any]], pos_names: list[str]) -> dict[str, Any]:
    by_pos: dict[str, Any] = {}
    for pos_name in pos_names:
        items = [row["positions"][pos_name] for row in rows if pos_name in row["positions"]]
        if not items:
            continue
        rb_layers = [x["metrics"]["repair_over_base_layer"] for x in items if x["metrics"]["repair_over_base_layer"] is not None]
        strict_layers = [x["metrics"]["strict_flip_layer"] for x in items if x["metrics"]["strict_flip_layer"] is not None]
        trans_layers = [x["metrics"]["best_transition_layer"] for x in items if x["metrics"]["best_transition_layer"] is not None]
        best_rb = [x["metrics"]["best_repair_minus_base"] for x in items if x["metrics"]["best_repair_minus_base"] is not None]
        best_spec = [x["metrics"]["best_specificity"] for x in items if x["metrics"]["best_specificity"] is not None]
        best_trans = [x["metrics"]["best_transition_advantage"] for x in items if x["metrics"]["best_transition_advantage"] is not None]
        final_rb = [x["metrics"]["final_repair_minus_base"] for x in items if x["metrics"]["final_repair_minus_base"] is not None]
        final_wb = [abs(x["metrics"]["final_wrong_minus_base"]) for x in items if x["metrics"]["final_wrong_minus_base"] is not None]
        by_pos[pos_name] = {
            "n": len(items),
            "repair_over_base_count": len(rb_layers),
            "strict_flip_count": len(strict_layers),
            "mean_repair_over_base_layer": mean(rb_layers) if rb_layers else None,
            "mean_strict_flip_layer": mean(strict_layers) if strict_layers else None,
            "mean_best_transition_layer": mean(trans_layers) if trans_layers else None,
            "mean_best_repair_minus_base": mean(best_rb) if best_rb else None,
            "mean_best_specificity": mean(best_spec) if best_spec else None,
            "mean_best_transition_advantage": mean(best_trans) if best_trans else None,
            "mean_final_repair_minus_base": mean(final_rb) if final_rb else None,
            "mean_abs_final_wrong_minus_base": mean(final_wb) if final_wb else None,
            "final_control_leak": (mean(final_wb) / (abs(mean(final_rb)) + 1e-6)) if final_wb and final_rb else None,
            "mean_base_margin_curve": avg_curve([x["routes"]["base"]["margin"] for x in items]),
            "mean_repair_margin_curve": avg_curve([x["routes"]["repair"]["margin"] for x in items]),
            "mean_wrong_margin_curve": avg_curve([x["routes"]["wrong"]["margin"] for x in items]),
            "mean_repair_minus_base_curve": avg_curve([x["metrics"]["repair_minus_base"] for x in items]),
            "mean_wrong_minus_base_curve": avg_curve([x["metrics"]["wrong_minus_base"] for x in items]),
            "mean_transition_advantage_curve": avg_curve([x["metrics"]["transition_advantage"] for x in items]),
        }
    best_positions = sorted(
        by_pos.values(),
        key=lambda x: (
            x["strict_flip_count"],
            x["repair_over_base_count"],
            x["mean_best_repair_minus_base"] or -999.0,
            -(x["final_control_leak"] or 999.0),
        ),
        reverse=True,
    )
    return {"by_position": by_pos, "best_positions": best_positions}


def run_model(args) -> dict[str, Any]:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        E = answer_vectors(model, tokenizer, values)
        cases = list(build_cases(args.n_tables, args.max_samples))
        pos_names = [x.strip() for x in args.positions.split(",") if x.strip()]
        log(f"{args.model}: layers={info.n_layers}, cases={len(cases)}, positions={pos_names}")

        rows: list[dict[str, Any]] = []
        target_seen = 0
        for si, case in enumerate(cases):
            correct = case["correct"]
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
            wrong_scores = score_map(model, tokenizer, device, case["wrong_prompt"], values)
            base = winner_stats(base_scores, correct)
            repair = winner_stats(repair_scores, correct)
            target_case = (not base["correct"]) and repair["correct"]
            if args.target_only and not target_case:
                continue
            target_seen += int(target_case)
            old_top_wrong = base["top_wrong"]
            positions = route_positions(tokenizer, case)
            route_data = {
                "base": collect_route_margins(model, tokenizer, device, case["base_prompt"], positions["base"], pos_names, E, correct, old_top_wrong, values),
                "repair": collect_route_margins(model, tokenizer, device, case["repair_prompt"], positions["repair"], pos_names, E, correct, old_top_wrong, values),
                "wrong": collect_route_margins(model, tokenizer, device, case["wrong_prompt"], positions["wrong"], pos_names, E, correct, old_top_wrong, values),
            }

            pos_records: dict[str, Any] = {}
            for pos_name in pos_names:
                if not all(pos_name in route_data[r] for r in ("base", "repair", "wrong")):
                    continue
                metrics = trajectory_metrics(
                    route_data["base"][pos_name]["margin"],
                    route_data["repair"][pos_name]["margin"],
                    route_data["wrong"][pos_name]["margin"],
                    args.delta,
                )
                pos_records[pos_name] = {
                    "positions": {
                        "base": route_data["base"][pos_name]["pos"],
                        "repair": route_data["repair"][pos_name]["pos"],
                        "wrong": route_data["wrong"][pos_name]["pos"],
                    },
                    "routes": {
                        "base": route_data["base"][pos_name],
                        "repair": route_data["repair"][pos_name],
                        "wrong": route_data["wrong"][pos_name],
                    },
                    "metrics": metrics,
                }
            rows.append({
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "old_top_wrong": old_top_wrong,
                "base": base,
                "repair_prompt": repair,
                "wrong_prompt": winner_stats(wrong_scores, correct),
                "positions": pos_records,
            })

        summary = summarize(rows, pos_names)
        for pos, item in summary["by_position"].items():
            log(
                f"{pos}: n={item['n']} strict={item['strict_flip_count']} "
                f"over_base={item['repair_over_base_count']} "
                f"best_rb={item['mean_best_repair_minus_base'] or 0:.3f} "
                f"trans={item['mean_best_transition_advantage'] or 0:.3f} "
                f"leak={item['final_control_leak'] if item['final_control_leak'] is not None else -1:.3f}"
            )
        return {
            "phase": 192,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "atlas_target": {
                "edge": "candidate-specific ranking repair",
                "old_level": "weak Level4 candidate after Phase191",
                "target": "trajectory atlas: locate layers where repair trajectory separates from base and wrong",
                "stop_condition": "if repair trajectory does not separate from wrong trajectory, move to prompt-program/interface graph level",
            },
            "n_layers": info.n_layers,
            "n_cases": len(cases),
            "n_target_cases_seen": target_seen,
            "n_rows": len(rows),
            "target_only": args.target_only,
            "positions": pos_names,
            "delta": args.delta,
            "summary": summary,
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
    parser.add_argument("--n-tables", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--positions", default=DEFAULT_POSITIONS)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 8
        args.positions = "prompt_last,query_relation"
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 12)
        args.max_samples = max(args.max_samples, 256)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase192_{args.model}_forward_dynamics_state_transition_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
