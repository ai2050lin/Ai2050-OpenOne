#!/usr/bin/env python3
"""Phase193: trajectory-localized causal transition test.

This phase tests whether the layer transition identified in Phase192 can
causally move candidate ranking, rather than merely being observable via
logit-lens trajectory separation.
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
from statistics import mean
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, compute_full_string_logprob_batch, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import answer_vectors, candidate_delta_metric  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase193_trajectory_localized_causal_transition")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def default_positions(model_name: str) -> list[str]:
    if model_name == "qwen3":
        return ["prompt_last", "query_category"]
    if model_name == "glm4":
        return ["prompt_last", "query_relation"]
    return ["prompt_last", "rule_value", "query_relation"]


def score_map(model, tokenizer, device, prompt: str, candidates: list[str]) -> dict[str, float]:
    return {k: v[0] for k, v in compute_full_string_logprob_batch(model, tokenizer, device, prompt, candidates).items()}


def margin(scores: dict[str, float], correct: str, old_top_wrong: str) -> float:
    return scores[correct] - scores[old_top_wrong]


def final_norm(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    return None


def collect_hidden(model, tokenizer, device, prompt: str) -> list[torch.Tensor]:
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
    return [h[0].detach().float().cpu() for h in out.hidden_states]


def lens_margin_curve(model, hidden: list[torch.Tensor], pos: int, E: torch.Tensor,
                      correct: str, old_top_wrong: str, values: list[str]) -> list[float]:
    norm = final_norm(model)
    ci = values.index(correct)
    wi = values.index(old_top_wrong)
    hs = torch.stack([hidden[i][pos] for i in range(1, len(hidden))], dim=0).to(next(model.parameters()).device)
    if norm is not None:
        hs = norm(hs)
    scores = hs.float().cpu() @ E.T.float()
    return [float(x) for x in (scores[:, ci] - scores[:, wi])]


def best_transition_layer(base_curve: list[float], repair_curve: list[float],
                          wrong_curve: list[float]) -> int | None:
    n = min(len(base_curve), len(repair_curve), len(wrong_curve))
    if n < 2:
        return None
    base_t = [base_curve[i + 1] - base_curve[i] for i in range(n - 1)]
    repair_t = [repair_curve[i + 1] - repair_curve[i] for i in range(n - 1)]
    wrong_t = [wrong_curve[i + 1] - wrong_curve[i] for i in range(n - 1)]
    adv = [repair_t[i] - max(base_t[i], wrong_t[i]) for i in range(n - 1)]
    return max(range(n - 1), key=lambda i: adv[i])


def patch_full_logprob_layer_output(model, tokenizer, device, prompt: str, answer: str,
                                    layer_idx: int, patch_pos: int,
                                    patch_vec: torch.Tensor) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids or patch_pos < 0 or patch_pos >= len(prompt_ids):
        return -100.0
    all_ids = prompt_ids + answer_ids
    target = get_layers(model)[layer_idx]
    vec = patch_vec.to(device=device)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h_new = h.clone()
            if patch_pos < h_new.shape[1]:
                h_new[0, patch_pos, :] = vec.to(dtype=h_new.dtype)
            return (h_new,) + output[1:]
        h_new = output.clone()
        if patch_pos < h_new.shape[1]:
            h_new[0, patch_pos, :] = vec.to(dtype=h_new.dtype)
        return h_new

    handle = target.register_forward_hook(hook)
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
        handle.remove()


def patched_score_map(model, tokenizer, device, prompt: str, candidates: list[str],
                      layer_idx: int, patch_pos: int, patch_vec: torch.Tensor) -> dict[str, float]:
    return {
        ans: patch_full_logprob_layer_output(model, tokenizer, device, prompt, ans, layer_idx, patch_pos, patch_vec)
        for ans in candidates
    }


def nearby_layers(layer: int, n_layers: int, radius: int) -> list[int]:
    return [l for l in range(max(0, layer - radius), min(n_layers - 1, layer + radius) + 1)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for item in row["interventions"]:
            groups[(item["position"], item["kind"])].append(item)

    by_key: dict[str, Any] = {}
    for (position, kind), items in groups.items():
        entry = {
            "position": position,
            "kind": kind,
            "n": len(items),
            "switch": sum(int(x["winner"]["correct"]) for x in items),
            "mean_margin_gain": mean(x["margin_gain"] for x in items),
            "positive_margin": sum(int(x["margin_gain"] > 0) for x in items),
            "mean_layer": mean(x["layer"] for x in items),
            "mean_base_margin": mean(x["base_margin"] for x in items),
            "mean_patched_margin": mean(x["patched_margin"] for x in items),
        }
        entry["switch_rate"] = entry["switch"] / max(1, entry["n"])
        entry["positive_margin_rate"] = entry["positive_margin"] / max(1, entry["n"])
        by_key[f"{position}|{kind}"] = entry

    by_position: dict[str, Any] = {}
    for pos in sorted({k[0] for k in groups}):
        repair = by_key.get(f"{pos}|base_repair_transition")
        wrong = by_key.get(f"{pos}|base_wrong_transition")
        ablate = by_key.get(f"{pos}|repair_base_transition")
        repair_gain = repair["mean_margin_gain"] if repair else 0.0
        wrong_gain = wrong["mean_margin_gain"] if wrong else 0.0
        ablation_loss = -(ablate["mean_margin_gain"]) if ablate else 0.0
        by_position[pos] = {
            "position": pos,
            "repair_gain": repair_gain,
            "wrong_gain": wrong_gain,
            "ablation_loss": ablation_loss,
            "transition_specificity": repair_gain / (abs(wrong_gain) + 1e-6),
            "repair_positive_rate": repair["positive_margin_rate"] if repair else 0.0,
            "wrong_positive_rate": wrong["positive_margin_rate"] if wrong else 0.0,
            "ablation_loss_positive_rate": sum(
                int(x["margin_gain"] < 0) for x in groups.get((pos, "repair_base_transition"), [])
            ) / max(1, len(groups.get((pos, "repair_base_transition"), []))),
            "repair_switch": repair["switch"] if repair else 0,
            "wrong_switch": wrong["switch"] if wrong else 0,
            "repair_n": repair["n"] if repair else 0,
        }
    best_positions = sorted(
        by_position.values(),
        key=lambda x: (
            x["repair_gain"],
            x["transition_specificity"],
            x["ablation_loss"],
            -abs(x["wrong_gain"]),
        ),
        reverse=True,
    )
    return {"by_key": by_key, "by_position": by_position, "best_positions": best_positions}


def run_model(args) -> dict[str, Any]:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        E = answer_vectors(model, tokenizer, values)
        cases = list(build_cases(args.n_tables, args.max_samples))
        pos_names = [x.strip() for x in (args.positions or ",".join(default_positions(args.model))).split(",") if x.strip()]
        log(f"{args.model}: layers={info.n_layers}, cases={len(cases)}, positions={pos_names}, radius={args.radius}")

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
            base_margin = margin(base_scores, correct, old_top_wrong)
            repair_margin = margin(repair_scores, correct, old_top_wrong)

            positions = {
                "base": case_positions(tokenizer, case, case["base_prompt"], case["relation"]),
                "repair": case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"]),
                "wrong": case_positions(tokenizer, case, case["wrong_prompt"], case["wrong_rel"]),
            }
            base_h = collect_hidden(model, tokenizer, device, case["base_prompt"])
            repair_h = collect_hidden(model, tokenizer, device, case["repair_prompt"])
            wrong_h = collect_hidden(model, tokenizer, device, case["wrong_prompt"])

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "old_top_wrong": old_top_wrong,
                "base": base,
                "repair_prompt": repair,
                "wrong_prompt": winner_stats(wrong_scores, correct),
                "interventions": [],
            }

            for pos_name in pos_names:
                bp = positions["base"].get(pos_name)
                rp = positions["repair"].get(pos_name)
                wp = positions["wrong"].get(pos_name)
                if bp is None or rp is None or wp is None:
                    continue
                if bp >= base_h[0].shape[0] or rp >= repair_h[0].shape[0] or wp >= wrong_h[0].shape[0]:
                    continue
                base_curve = lens_margin_curve(model, base_h, bp, E, correct, old_top_wrong, values)
                repair_curve = lens_margin_curve(model, repair_h, rp, E, correct, old_top_wrong, values)
                wrong_curve = lens_margin_curve(model, wrong_h, wp, E, correct, old_top_wrong, values)
                best_l = best_transition_layer(base_curve, repair_curve, wrong_curve)
                if best_l is None:
                    continue
                for li in nearby_layers(best_l, info.n_layers, args.radius):
                    if li + 1 >= len(base_h) or li + 1 >= len(repair_h) or li + 1 >= len(wrong_h):
                        continue
                    repair_transition = repair_h[li + 1][rp] - repair_h[li][rp]
                    wrong_transition = wrong_h[li + 1][wp] - wrong_h[li][wp]
                    base_transition = base_h[li + 1][bp] - base_h[li][bp]
                    specs = [
                        {
                            "kind": "base_repair_transition",
                            "prompt": case["base_prompt"],
                            "baseline_scores": base_scores,
                            "baseline_margin": base_margin,
                            "patch_pos": bp,
                            "patch_vec": base_h[li][bp] + repair_transition,
                        },
                        {
                            "kind": "base_wrong_transition",
                            "prompt": case["base_prompt"],
                            "baseline_scores": base_scores,
                            "baseline_margin": base_margin,
                            "patch_pos": bp,
                            "patch_vec": base_h[li][bp] + wrong_transition,
                        },
                        {
                            "kind": "repair_base_transition",
                            "prompt": case["repair_prompt"],
                            "baseline_scores": repair_scores,
                            "baseline_margin": repair_margin,
                            "patch_pos": rp,
                            "patch_vec": repair_h[li][rp] + base_transition,
                        },
                    ]
                    for spec in specs:
                        scores = patched_score_map(
                            model, tokenizer, device, spec["prompt"], values, li, spec["patch_pos"], spec["patch_vec"]
                        )
                        patched = winner_stats(scores, correct)
                        patched_margin = margin(scores, correct, old_top_wrong)
                        metric = candidate_delta_metric(spec["baseline_scores"], scores, correct, old_top_wrong)
                        row["interventions"].append({
                            "position": pos_name,
                            "kind": spec["kind"],
                            "layer": li,
                            "best_transition_layer": best_l,
                            "winner": patched,
                            "base_margin": spec["baseline_margin"],
                            "patched_margin": patched_margin,
                            "margin_gain": patched_margin - spec["baseline_margin"],
                            "metric": metric,
                        })
            rows.append(row)

        summary = summarize(rows)
        for pos, item in summary["by_position"].items():
            log(
                f"{pos}: repair_gain={item['repair_gain']:.3f} wrong_gain={item['wrong_gain']:.3f} "
                f"ablation_loss={item['ablation_loss']:.3f} spec={item['transition_specificity']:.3f} "
                f"switch={item['repair_switch']}/{item['repair_n']}"
            )
        return {
            "phase": 193,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "atlas_target": {
                "edge": "candidate-specific ranking repair",
                "old_level": "trajectory-Level4 candidate",
                "target": "localized causal transition evidence",
                "success": "base<-repair transition improves margin, base<-wrong does not, repair<-base ablation lowers margin",
            },
            "n_layers": info.n_layers,
            "n_cases": len(cases),
            "n_target_cases_seen": target_seen,
            "n_rows": len(rows),
            "target_only": args.target_only,
            "positions": pos_names,
            "radius": args.radius,
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
    parser.add_argument("--positions", default="")
    parser.add_argument("--radius", type=int, default=1)
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
        args.positions = "prompt_last"
        args.radius = 0
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
    out_path = out_dir / f"phase193_{args.model}_trajectory_localized_causal_transition_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
