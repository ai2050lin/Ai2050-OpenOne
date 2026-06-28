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
from phase722_functional_head_atlas_causal_ablation import (  # noqa: E402
    install_zero_head_ablation,
    logit_diag,
    random_control_head,
    target_token_ids,
    write_json,
    write_jsonl,
)


OUT_ROOT = Path("results/glm5_phase723_apple_fruit_attribute_micro_atlas")
PHASE722_ROOT = Path("results/glm5_phase722_functional_head_atlas_causal_ablation")
MODELS = ["qwen3", "glm4", "deepseek7b"]
FAMILY_KEY = "fruit_identity_reuse_difference"

FALLBACK_HEADS = {
    "qwen3": [(28, 0), (24, 29), (26, 26)],
    "glm4": [(24, 19), (29, 28), (29, 18)],
    "deepseek7b": [(20, 17), (27, 23), (23, 0)],
}

BASE_OBJECTS = [
    {"object": "apple", "group": "apple", "category": "fruit", "color": "red", "taste": "sweet", "shape": "round", "edible": "yes", "tree": "yes"},
    {"object": "banana", "group": "other_fruit", "category": "fruit", "color": "yellow", "taste": "sweet", "shape": "long", "edible": "yes", "tree": "no"},
    {"object": "pear", "group": "other_fruit", "category": "fruit", "color": "green", "taste": "sweet", "shape": "oval", "edible": "yes", "tree": "yes"},
    {"object": "grape", "group": "other_fruit", "category": "fruit", "color": "purple", "taste": "sweet", "shape": "round", "edible": "yes", "tree": "no"},
    {"object": "orange", "group": "other_fruit", "category": "fruit", "color": "orange", "taste": "sweet", "shape": "round", "edible": "yes", "tree": "yes"},
    {"object": "lemon", "group": "other_fruit", "category": "fruit", "color": "yellow", "taste": "sour", "shape": "oval", "edible": "yes", "tree": "yes"},
    {"object": "carrot", "group": "nonfruit", "category": "vegetable", "color": "orange", "taste": "earthy", "shape": "long", "edible": "yes", "tree": "no"},
    {"object": "potato", "group": "nonfruit", "category": "vegetable", "color": "brown", "taste": "starchy", "shape": "round", "edible": "yes", "tree": "no"},
    {"object": "stone", "group": "nonfruit", "category": "object", "color": "gray", "taste": "none", "shape": "irregular", "edible": "no", "tree": "no"},
    {"object": "chair", "group": "nonfruit", "category": "furniture", "color": "brown", "taste": "none", "shape": "rectangular", "edible": "no", "tree": "no"},
    {"object": "car", "group": "nonfruit", "category": "vehicle", "color": "red", "taste": "none", "shape": "rectangular", "edible": "no", "tree": "no"},
    {"object": "spoon", "group": "nonfruit", "category": "tool", "color": "silver", "taste": "none", "shape": "long", "edible": "no", "tree": "no"},
]

CONFLICT_OBJECTS = [
    {"object": "apple", "group": "apple", "category": "tool", "color": "blue", "taste": "bitter", "shape": "square", "edible": "no", "tree": "no"},
    {"object": "banana", "group": "other_fruit", "category": "tool", "color": "blue", "taste": "bitter", "shape": "square", "edible": "no", "tree": "yes"},
    {"object": "carrot", "group": "nonfruit", "category": "fruit", "color": "purple", "taste": "sweet", "shape": "round", "edible": "yes", "tree": "yes"},
    {"object": "stone", "group": "nonfruit", "category": "fruit", "color": "yellow", "taste": "sweet", "shape": "oval", "edible": "yes", "tree": "yes"},
]

RELATIONS = [
    ("category", "category"),
    ("color", "color"),
    ("taste", "taste"),
    ("shape", "shape"),
    ("edible", "edible"),
    ("grows_on_tree", "tree"),
]
COMMONSENSE_RELATIONS = [("category", "category"), ("color", "color"), ("taste", "taste")]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def records_for(obj: dict[str, str]) -> str:
    name = obj["object"]
    return "\n".join(
        [
            f"{name}.category = {obj['category']}",
            f"{name}.color = {obj['color']}",
            f"{name}.taste = {obj['taste']}",
            f"{name}.shape = {obj['shape']}",
            f"{name}.edible = {obj['edible']}",
            f"{name}.grows_on_tree = {obj['tree']}",
        ]
    )


def question_for(name: str, relation: str) -> str:
    if relation == "category":
        return f"What is the category of {name}?"
    if relation == "color":
        return f"What is the color of {name}?"
    if relation == "taste":
        return f"What is the taste of {name}?"
    if relation == "shape":
        return f"What is the shape of {name}?"
    if relation == "edible":
        return f"Is {name} edible?"
    if relation == "grows_on_tree":
        return f"Does {name} grow on a tree?"
    raise ValueError(relation)


def prompt_for(case: dict[str, Any]) -> str:
    if case["prompt_type"] == "commonsense":
        return (
            "Answer using common everyday knowledge.\n"
            "Use exactly one short value.\n"
            f"Question: {question_for(case['object'], case['relation'])}\n"
            "Answer:"
        )
    scope = "Facts:" if case["prompt_type"] == "explicit_profile" else "Temporary world facts:"
    return (
        f"{scope}\n"
        f"{case['records']}\n"
        "Use the facts above. Answer with exactly one short value.\n"
        f"Question: {question_for(case['object'], case['relation'])}\n"
        "Answer:"
    )


def build_cases(max_cases: int | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    cid = 0
    for obj in BASE_OBJECTS:
        for relation, key in RELATIONS:
            cid += 1
            cases.append(
                {
                    "case_id": f"p723_explicit_{cid:04d}",
                    "prompt_type": "explicit_profile",
                    "object": obj["object"],
                    "object_group": obj["group"],
                    "relation": relation,
                    "answer": obj[key],
                    "records": records_for(obj),
                }
            )
    for obj in CONFLICT_OBJECTS:
        for relation, key in RELATIONS:
            cid += 1
            cases.append(
                {
                    "case_id": f"p723_conflict_{cid:04d}",
                    "prompt_type": "conflict_profile",
                    "object": obj["object"],
                    "object_group": obj["group"],
                    "relation": relation,
                    "answer": obj[key],
                    "records": records_for(obj),
                }
            )
    for obj in BASE_OBJECTS[:6]:
        for relation, key in COMMONSENSE_RELATIONS:
            cid += 1
            cases.append(
                {
                    "case_id": f"p723_commonsense_{cid:04d}",
                    "prompt_type": "commonsense",
                    "object": obj["object"],
                    "object_group": obj["group"],
                    "relation": relation,
                    "answer": obj[key],
                    "records": "",
                }
            )
    return cases[:max_cases] if max_cases else cases


def load_candidate_heads(model_name: str, top_k: int) -> list[dict[str, Any]]:
    path = PHASE722_ROOT / f"phase722_{model_name}_causal_ablation_summary.json"
    rows: list[dict[str, Any]] = []
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data["by_family"].get(FAMILY_KEY, {}).get("candidate_heads", [])[:top_k]:
            rows.append(
                {
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "head_key": row["head_key"],
                    "phase722_mean_logprob_delta": float(row["mean_logprob_delta"]),
                    "phase722_source_focus_score": row.get("source_focus_score"),
                }
            )
    if len(rows) < top_k:
        seen = {(r["layer"], r["head"]) for r in rows}
        for layer, head in FALLBACK_HEADS[model_name]:
            if (layer, head) not in seen:
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "head_key": f"L{layer}H{head}",
                        "phase722_mean_logprob_delta": None,
                        "phase722_source_focus_score": None,
                    }
                )
            if len(rows) >= top_k:
                break
    return rows[:top_k]


def run_logits_ids(model, device, ids: list[int], ablations: list[dict[str, int]] | None = None) -> torch.Tensor:
    handles = install_zero_head_ablation(model, ablations or []) if ablations else []
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return out.logits[0, -1].detach().cpu()
    finally:
        for h in handles:
            h.remove()


def phrase_diag(model, tokenizer, device, prompt: str, answer: str, ablations: list[dict[str, int]] | None = None) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = target_token_ids(tokenizer, answer)
    cur = list(prompt_ids)
    token_diags = []
    for target_id in ans_ids:
        diag = logit_diag(run_logits_ids(model, device, cur, ablations), int(target_id))
        token_diags.append(diag)
        cur.append(int(target_id))
    return {
        "answer_token_ids": [int(x) for x in ans_ids],
        "answer_token_texts": [tokenizer.decode([int(x)]) for x in ans_ids],
        "sum_logprob": sum(d["target_logprob"] for d in token_diags),
        "mean_logprob": sum(d["target_logprob"] for d in token_diags) / len(token_diags),
        "first_logprob": token_diags[0]["target_logprob"],
        "first_rank": token_diags[0]["target_rank"],
        "first_top1": token_diags[0]["target_top1"],
        "first_margin": token_diags[0]["margin_vs_best_other"],
        "n_answer_tokens": len(ans_ids),
    }


def mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def summarize(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition_kind"], row["head_key"])].append(row)

    head_summaries = []
    for (kind, head_key), vals in groups.items():
        by_object_group = {
            g: {
                "n": len(gvals),
                "mean_mean_logprob_delta": mean([v["mean_logprob_delta"] for v in gvals]),
                "mean_necessity": mean([-v["mean_logprob_delta"] for v in gvals]),
            }
            for g, gvals in group_by(vals, "object_group").items()
        }
        by_prompt_type = {
            g: {
                "n": len(gvals),
                "mean_mean_logprob_delta": mean([v["mean_logprob_delta"] for v in gvals]),
                "mean_necessity": mean([-v["mean_logprob_delta"] for v in gvals]),
            }
            for g, gvals in group_by(vals, "prompt_type").items()
        }
        by_relation = {
            g: {
                "n": len(gvals),
                "mean_mean_logprob_delta": mean([v["mean_logprob_delta"] for v in gvals]),
                "mean_necessity": mean([-v["mean_logprob_delta"] for v in gvals]),
            }
            for g, gvals in group_by(vals, "relation").items()
        }
        explicit = [v for v in vals if v["prompt_type"] == "explicit_profile"]
        explicit_groups = group_by(explicit, "object_group")
        apple_need = mean([-v["mean_logprob_delta"] for v in explicit_groups.get("apple", [])])
        fruit_need = mean([-v["mean_logprob_delta"] for v in explicit_groups.get("other_fruit", [])])
        nonfruit_need = mean([-v["mean_logprob_delta"] for v in explicit_groups.get("nonfruit", [])])
        head_summaries.append(
            {
                "model": model_name,
                "condition_kind": kind,
                "head_key": head_key,
                "layer": vals[0]["layer"],
                "head": vals[0]["head"],
                "n": len(vals),
                "mean_mean_logprob_delta": mean([v["mean_logprob_delta"] for v in vals]),
                "mean_sum_logprob_delta": mean([v["sum_logprob_delta"] for v in vals]),
                "mean_first_logprob_delta": mean([v["first_logprob_delta"] for v in vals]),
                "mean_first_rank_delta": mean([v["first_rank_delta"] for v in vals]),
                "first_top1_drop_rate": sum(1 for v in vals if v["baseline_first_top1"] and not v["patched_first_top1"]) / len(vals),
                "logprob_worse_rate": sum(1 for v in vals if v["mean_logprob_delta"] < 0) / len(vals),
                "phase722_mean_logprob_delta": vals[0].get("phase722_mean_logprob_delta"),
                "phase722_source_focus_score": vals[0].get("phase722_source_focus_score"),
                "by_object_group": by_object_group,
                "by_prompt_type": by_prompt_type,
                "by_relation": by_relation,
                "reuse_difference": {
                    "apple_explicit_necessity": apple_need,
                    "other_fruit_explicit_necessity": fruit_need,
                    "nonfruit_explicit_necessity": nonfruit_need,
                    "apple_minus_other_fruit": None if apple_need is None or fruit_need is None else apple_need - fruit_need,
                    "other_fruit_minus_nonfruit": None if fruit_need is None or nonfruit_need is None else fruit_need - nonfruit_need,
                },
            }
        )

    cand = [r for r in head_summaries if r["condition_kind"] == "candidate"]
    ctrl = [r for r in head_summaries if r["condition_kind"] == "same_layer_random"]
    return {
        "phase": 723,
        "title": "Apple-Fruit-Attribute Reuse-Difference Micro-Atlas",
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "evidence_type": "teacher-forced answer phrase likelihood under local zero head ablation",
        "candidate_heads": cand,
        "random_control_heads": ctrl,
        "most_harmful_candidate_heads": sorted(cand, key=lambda r: r["mean_mean_logprob_delta"] or 0)[:8],
        "most_fruit_shared_candidate_heads": sorted(
            cand,
            key=lambda r: r["reuse_difference"]["other_fruit_minus_nonfruit"]
            if r["reuse_difference"]["other_fruit_minus_nonfruit"] is not None
            else -999,
            reverse=True,
        )[:8],
        "most_apple_specific_candidate_heads": sorted(
            cand,
            key=lambda r: r["reuse_difference"]["apple_minus_other_fruit"]
            if r["reuse_difference"]["apple_minus_other_fruit"] is not None
            else -999,
            reverse=True,
        )[:8],
        "strict_interpretation": "positive necessity means zeroing the head reduced answer phrase likelihood; it is a necessity hint, not a complete coding mechanism",
    }


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row[key])].append(row)
    return out


def run_model(args) -> dict[str, Any]:
    rng = random.Random(args.seed)
    cases = build_cases(args.max_cases)
    candidate_heads = load_candidate_heads(args.model, args.top_heads)
    log(f"{args.model}: cases={len(cases)}, candidate_heads={[h['head_key'] for h in candidate_heads]}")

    model, tokenizer, device = load_model(args.model)
    rows: list[dict[str, Any]] = []
    try:
        avoid_by_layer: dict[int, set[int]] = defaultdict(set)
        for h in candidate_heads:
            avoid_by_layer[int(h["layer"])].add(int(h["head"]))
        random_heads = []
        for h in candidate_heads:
            rh = random_control_head(model, int(h["layer"]), avoid_by_layer[int(h["layer"])], rng)
            random_heads.append(
                {
                    "layer": int(h["layer"]),
                    "head": rh,
                    "head_key": f"L{h['layer']}H{rh}",
                    "phase722_mean_logprob_delta": None,
                    "phase722_source_focus_score": None,
                }
            )
        tests = [("candidate", h) for h in candidate_heads] + [("same_layer_random", h) for h in random_heads]

        for idx, case in enumerate(cases, 1):
            prompt = prompt_for(case)
            baseline = phrase_diag(model, tokenizer, device, prompt, case["answer"])
            for kind, h in tests:
                patched = phrase_diag(
                    model,
                    tokenizer,
                    device,
                    prompt,
                    case["answer"],
                    [{"layer": h["layer"], "head": h["head"]}],
                )
                rows.append(
                    {
                        "model": args.model,
                        "case_id": case["case_id"],
                        "prompt_type": case["prompt_type"],
                        "object": case["object"],
                        "object_group": case["object_group"],
                        "relation": case["relation"],
                        "answer": case["answer"],
                        "answer_token_ids": baseline["answer_token_ids"],
                        "answer_token_texts": baseline["answer_token_texts"],
                        "n_answer_tokens": baseline["n_answer_tokens"],
                        "condition_kind": kind,
                        "layer": int(h["layer"]),
                        "head": int(h["head"]),
                        "head_key": h["head_key"],
                        "phase722_mean_logprob_delta": h.get("phase722_mean_logprob_delta"),
                        "phase722_source_focus_score": h.get("phase722_source_focus_score"),
                        "baseline_mean_logprob": baseline["mean_logprob"],
                        "patched_mean_logprob": patched["mean_logprob"],
                        "mean_logprob_delta": patched["mean_logprob"] - baseline["mean_logprob"],
                        "baseline_sum_logprob": baseline["sum_logprob"],
                        "patched_sum_logprob": patched["sum_logprob"],
                        "sum_logprob_delta": patched["sum_logprob"] - baseline["sum_logprob"],
                        "baseline_first_logprob": baseline["first_logprob"],
                        "patched_first_logprob": patched["first_logprob"],
                        "first_logprob_delta": patched["first_logprob"] - baseline["first_logprob"],
                        "baseline_first_rank": baseline["first_rank"],
                        "patched_first_rank": patched["first_rank"],
                        "first_rank_delta": patched["first_rank"] - baseline["first_rank"],
                        "baseline_first_top1": baseline["first_top1"],
                        "patched_first_top1": patched["first_top1"],
                        "baseline_first_margin": baseline["first_margin"],
                        "patched_first_margin": patched["first_margin"],
                        "first_margin_delta": patched["first_margin"] - baseline["first_margin"],
                    }
                )
            if idx % args.log_every == 0 or idx == len(cases):
                log(f"{args.model}: {idx}/{len(cases)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(rows, args.model)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_ROOT / f"phase723_{args.model}_micro_atlas_rows.jsonl", rows)
    write_json(OUT_ROOT / f"phase723_{args.model}_micro_atlas_summary.json", summary)
    compact = {
        "model": args.model,
        "n_cases": summary["n_cases"],
        "n_rows": summary["n_rows"],
        "most_harmful_candidate_heads": summary["most_harmful_candidate_heads"][:5],
        "most_fruit_shared_candidate_heads": summary["most_fruit_shared_candidate_heads"][:5],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return summary


def write_cross_summary() -> dict[str, Any]:
    summaries = []
    for model in MODELS:
        path = OUT_ROOT / f"phase723_{model}_micro_atlas_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 723,
        "title": "Apple-Fruit-Attribute Reuse-Difference Micro-Atlas",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "teacher-forced answer phrase likelihood under local zero head ablation",
        "small_model_caution": "small models can implement different or distorted internal routes; cross-model agreement is evidence, disagreement is not direct falsification",
        "by_model": {
            s["model"]: {
                "n_cases": s["n_cases"],
                "n_rows": s["n_rows"],
                "most_harmful_candidate_heads": s["most_harmful_candidate_heads"],
                "most_fruit_shared_candidate_heads": s["most_fruit_shared_candidate_heads"],
                "most_apple_specific_candidate_heads": s["most_apple_specific_candidate_heads"],
                "random_control_heads": s["random_control_heads"],
            }
            for s in summaries
        },
    }
    write_json(OUT_ROOT / "phase723_cross_model_summary.json", payload)

    lines = [
        "# Phase 723 Apple-Fruit-Attribute Reuse-Difference Micro-Atlas",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: teacher-forced answer phrase likelihood under local zero head ablation.",
        "- Interpretation: positive necessity means zeroing the head reduced answer phrase likelihood.",
        "",
        "## Most Harmful Candidate Heads",
        "",
    ]
    for model, item in payload["by_model"].items():
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| head | mean_logprob_delta | first_rank_delta | top1_drop | apple_need | fruit_need | nonfruit_need | fruit-nonfruit | apple-fruit |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in item["most_harmful_candidate_heads"][:8]:
            rd = r["reuse_difference"]
            lines.append(
                f"| {r['head_key']} | {r['mean_mean_logprob_delta']:.4f} | {r['mean_first_rank_delta']:.2f} | "
                f"{r['first_top1_drop_rate']:.3f} | {fmt(rd['apple_explicit_necessity'])} | "
                f"{fmt(rd['other_fruit_explicit_necessity'])} | {fmt(rd['nonfruit_explicit_necessity'])} | "
                f"{fmt(rd['other_fruit_minus_nonfruit'])} | {fmt(rd['apple_minus_other_fruit'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Strict Interpretation",
            "",
            "- This is a micro-world causal screen, not a global neuron atlas.",
            "- Strong shared fruit necessity suggests a reusable category route.",
            "- Strong apple-minus-fruit suggests object-specific differential routing.",
            "- Weak qwen3/GLM4 effects may mean redundancy or different implementation, not absence of coding.",
            "",
        ]
    )
    (OUT_ROOT / "phase723_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "models": payload["models"]}, ensure_ascii=False), flush=True)
    return payload


def fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--top-heads", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=12)
    parser.add_argument("--seed", type=int, default=723)
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
