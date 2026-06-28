#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import load_model, release_model  # noqa: E402
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402
from phase721_global_functional_head_atlas_expansion import prompt_for, select_cases  # noqa: E402


OUT_ROOT = Path("results/glm5_phase722_functional_head_atlas_causal_ablation")
PHASE721_ROOT = Path("results/glm5_phase721_global_functional_head_atlas_expansion")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_top_heads(model_name: str, top_k_per_family: int) -> dict[str, list[dict[str, Any]]]:
    path = PHASE721_ROOT / f"phase721_{model_name}_head_scores.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, Any]]] = {}
    for family, rec in data["by_family"].items():
        rows = rec["top_source_focus_heads"][:top_k_per_family]
        out[family] = [
            {
                "family": family,
                "layer": int(r["layer"]),
                "head": int(r["head"]),
                "head_key": r["head_key"],
                "source_focus_score": float(r["source_focus_score"]),
                "mean_mass_target_value": float(r.get("mean_mass_target_value", 0.0)),
                "mean_mass_object_name": float(r.get("mean_mass_object_name", 0.0)),
                "mean_mass_relation_name": float(r.get("mean_mass_relation_name", 0.0)),
            }
            for r in rows
        ]
    return out


def target_token_ids(tokenizer, answer: str) -> list[int]:
    ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(answer, add_special_tokens=False)
    if not ids:
        raise ValueError(f"empty tokenization for answer={answer!r}")
    return ids


def logit_diag(logits: torch.Tensor, target_id: int) -> dict[str, Any]:
    vals = logits.detach().float()
    lp = torch.log_softmax(vals, dim=-1)
    target_logit = float(vals[target_id].item())
    target_logprob = float(lp[target_id].item())
    rank = int((vals > vals[target_id]).sum().item()) + 1
    top_id = int(torch.argmax(vals).item())
    top_logit = float(vals[top_id].item())
    masked = vals.clone()
    masked[target_id] = -torch.inf
    best_other_id = int(torch.argmax(masked).item())
    return {
        "target_id": int(target_id),
        "target_logit": target_logit,
        "target_logprob": target_logprob,
        "target_rank": rank,
        "target_top1": top_id == target_id,
        "top_id": top_id,
        "top_logit": top_logit,
        "best_other_id": best_other_id,
        "margin_vs_best_other": target_logit - float(masked[best_other_id].item()),
    }


def install_zero_head_ablation(model, specs: list[dict[str, int]]):
    by_layer: dict[int, set[int]] = defaultdict(set)
    for spec in specs:
        by_layer[int(spec["layer"])].add(int(spec["head"]))
    handles = []
    for layer_idx, heads_set in by_layer.items():
        o_proj, n_heads, head_dim = head_meta(model, layer_idx)
        heads = [h for h in sorted(heads_set) if 0 <= h < n_heads]

        def pre_hook(_module, inputs, heads=heads, n_heads=n_heads, head_dim=head_dim):
            x = inputs[0]
            y = x.clone()
            yv = y.view(y.shape[0], y.shape[1], n_heads, head_dim)
            for h in heads:
                yv[0, -1, h, :] = 0
            return (y,) + tuple(inputs[1:])

        handles.append(o_proj.register_forward_pre_hook(pre_hook))
    return handles


def run_logits(model, tokenizer, device, prompt: str, ablations: list[dict[str, int]] | None = None) -> torch.Tensor:
    handles = install_zero_head_ablation(model, ablations or []) if ablations else []
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return out.logits[0, -1].detach().cpu()
    finally:
        for h in handles:
            h.remove()


def random_control_head(model, layer: int, avoid: set[int], rng: random.Random) -> int:
    _o_proj, n_heads, _head_dim = head_meta(model, layer)
    candidates = [h for h in range(n_heads) if h not in avoid]
    if not candidates:
        candidates = list(range(n_heads))
    return int(rng.choice(candidates))


def summarize(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["function_family"], row["condition_kind"], row["head_key"])].append(row)

    head_summaries: list[dict[str, Any]] = []
    for (family, kind, head_key), vals in groups.items():
        n = len(vals)
        head_summaries.append(
            {
                "model": model_name,
                "function_family": family,
                "condition_kind": kind,
                "head_key": head_key,
                "layer": vals[0]["layer"],
                "head": vals[0]["head"],
                "n": n,
                "mean_logprob_delta": sum(v["target_logprob_delta"] for v in vals) / n,
                "mean_rank_delta": sum(v["target_rank_delta"] for v in vals) / n,
                "mean_margin_delta": sum(v["margin_delta"] for v in vals) / n,
                "top1_drop_rate": sum(1 for v in vals if v["baseline_top1"] and not v["patched_top1"]) / n,
                "rank_worse_rate": sum(1 for v in vals if v["target_rank_delta"] > 0) / n,
                "logprob_worse_rate": sum(1 for v in vals if v["target_logprob_delta"] < 0) / n,
                "source_focus_score": vals[0].get("source_focus_score"),
            }
        )

    by_family: dict[str, dict[str, Any]] = {}
    for family in sorted({r["function_family"] for r in rows}):
        cand = [r for r in head_summaries if r["function_family"] == family and r["condition_kind"] == "candidate"]
        ctrl = [r for r in head_summaries if r["function_family"] == family and r["condition_kind"] == "same_layer_random"]
        by_family[family] = {
            "candidate_heads": sorted(cand, key=lambda r: r["mean_logprob_delta"])[:12],
            "random_control_heads": sorted(ctrl, key=lambda r: r["mean_logprob_delta"])[:12],
            "mean_candidate_logprob_delta": sum(r["mean_logprob_delta"] for r in cand) / len(cand) if cand else None,
            "mean_random_logprob_delta": sum(r["mean_logprob_delta"] for r in ctrl) / len(ctrl) if ctrl else None,
            "mean_candidate_rank_delta": sum(r["mean_rank_delta"] for r in cand) / len(cand) if cand else None,
            "mean_random_rank_delta": sum(r["mean_rank_delta"] for r in ctrl) / len(ctrl) if ctrl else None,
        }

    return {
        "phase": 722,
        "title": "Functional Head Atlas Causal Ablation Validation",
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": len(rows),
        "by_family": by_family,
        "most_harmful_candidate_heads": sorted(
            [r for r in head_summaries if r["condition_kind"] == "candidate"],
            key=lambda r: r["mean_logprob_delta"],
        )[:24],
        "most_harmful_random_controls": sorted(
            [r for r in head_summaries if r["condition_kind"] == "same_layer_random"],
            key=lambda r: r["mean_logprob_delta"],
        )[:24],
        "interpretation": "negative mean_logprob_delta or positive mean_rank_delta means zeroing this head locally hurt the target first token",
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n", encoding="utf-8")


def run_model(args) -> dict[str, Any]:
    rng = random.Random(args.seed)
    cases = select_cases(args.max_cases_per_family)
    cases_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        cases_by_family[case["function_family"]].append(case)
    top_heads = load_top_heads(args.model, args.top_heads_per_family)
    log(f"{args.model}: cases={len(cases)}, top_heads_per_family={args.top_heads_per_family}")

    model, tokenizer, device = load_model(args.model)
    rows: list[dict[str, Any]] = []
    try:
        for family, family_heads in top_heads.items():
            family_cases = cases_by_family.get(family, [])[:args.max_cases_per_family]
            if not family_cases:
                continue
            avoid_by_layer: dict[int, set[int]] = defaultdict(set)
            for h in family_heads:
                avoid_by_layer[int(h["layer"])].add(int(h["head"]))
            random_heads = [
                {
                    "family": family,
                    "layer": h["layer"],
                    "head": random_control_head(model, h["layer"], avoid_by_layer[h["layer"]], rng),
                    "head_key": None,
                    "source_focus_score": None,
                }
                for h in family_heads
            ]
            for h in random_heads:
                h["head_key"] = f"L{h['layer']}H{h['head']}"
            tests = [("candidate", h) for h in family_heads] + [("same_layer_random", h) for h in random_heads]

            for idx, case in enumerate(family_cases, 1):
                prompt = prompt_for(case)
                tgt_ids = target_token_ids(tokenizer, case["answer"])
                target_id = int(tgt_ids[0])
                baseline = logit_diag(run_logits(model, tokenizer, device, prompt), target_id)
                for kind, h in tests:
                    patched = logit_diag(
                        run_logits(model, tokenizer, device, prompt, [{"layer": h["layer"], "head": h["head"]}]),
                        target_id,
                    )
                    rows.append(
                        {
                            "model": args.model,
                            "function_family": family,
                            "case_id": case["case_id"],
                            "answer": case["answer"],
                            "target_id": target_id,
                            "target_token_text": tokenizer.decode([target_id]),
                            "condition_kind": kind,
                            "layer": h["layer"],
                            "head": h["head"],
                            "head_key": h["head_key"],
                            "source_focus_score": h.get("source_focus_score"),
                            "baseline_logprob": baseline["target_logprob"],
                            "patched_logprob": patched["target_logprob"],
                            "target_logprob_delta": patched["target_logprob"] - baseline["target_logprob"],
                            "baseline_rank": baseline["target_rank"],
                            "patched_rank": patched["target_rank"],
                            "target_rank_delta": patched["target_rank"] - baseline["target_rank"],
                            "baseline_margin": baseline["margin_vs_best_other"],
                            "patched_margin": patched["margin_vs_best_other"],
                            "margin_delta": patched["margin_vs_best_other"] - baseline["margin_vs_best_other"],
                            "baseline_top1": baseline["target_top1"],
                            "patched_top1": patched["target_top1"],
                            "baseline_top_id": baseline["top_id"],
                            "patched_top_id": patched["top_id"],
                        }
                    )
                if idx % args.log_every == 0 or idx == len(family_cases):
                    log(f"{args.model}: {family} {idx}/{len(family_cases)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(rows, args.model)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_ROOT / f"phase722_{args.model}_causal_ablation_rows.jsonl", rows)
    write_json(OUT_ROOT / f"phase722_{args.model}_causal_ablation_summary.json", summary)
    compact = {
        "model": args.model,
        "n_rows": summary["n_rows"],
        "most_harmful_candidate_heads": summary["most_harmful_candidate_heads"][:8],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return summary


def write_cross_summary() -> dict[str, Any]:
    models = []
    for model in MODELS:
        path = OUT_ROOT / f"phase722_{model}_causal_ablation_summary.json"
        if path.exists():
            models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 722,
        "title": "Functional Head Atlas Causal Ablation Validation",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [m["model"] for m in models],
        "status": "complete" if len(models) == len(MODELS) else "partial",
        "evidence_type": "local zero ablation at answer_last o_proj input",
        "small_model_caution": "zero ablation is off-manifold and first-token only; it validates necessity hints, not full mechanism closure",
        "by_model": {
            m["model"]: {
                "n_rows": m["n_rows"],
                "most_harmful_candidate_heads": m["most_harmful_candidate_heads"][:12],
                "most_harmful_random_controls": m["most_harmful_random_controls"][:12],
                "by_family": m["by_family"],
            }
            for m in models
        },
    }
    write_json(OUT_ROOT / "phase722_cross_model_summary.json", payload)
    lines = [
        "# Phase 722 Functional Head Atlas Causal Ablation Validation",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: local zero ablation at answer_last o_proj input.",
        "",
        "## Most Harmful Candidate Heads",
        "",
    ]
    for model, item in payload["by_model"].items():
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| family | head | mean_logprob_delta | mean_rank_delta | top1_drop | source_focus |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in item["most_harmful_candidate_heads"][:12]:
            sf = r.get("source_focus_score")
            sf_text = "" if sf is None else f"{sf:.4f}"
            lines.append(
                f"| {r['function_family']} | {r['head_key']} | {r['mean_logprob_delta']:.4f} | "
                f"{r['mean_rank_delta']:.2f} | {r['top1_drop_rate']:.3f} | {sf_text} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Strict Interpretation",
            "",
            "- Negative logprob delta means zeroing the head hurt the target first token.",
            "- This is a necessity hint, not a sufficiency proof.",
            "- Full phrase likelihood and natural generation closure still need validation.",
            "",
        ]
    )
    (OUT_ROOT / "phase722_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "models": payload["models"]}, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--top-heads-per-family", type=int, default=3)
    parser.add_argument("--max-cases-per-family", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--seed", type=int, default=722)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        write_cross_summary()
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
