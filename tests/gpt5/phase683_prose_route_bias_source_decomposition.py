#!/usr/bin/env python3
"""
Phase 683: Prose-Route Bias Source Decomposition.

Phase 682 found that DS7B failures are dominated by prose/explanation route.
This phase tests where that bias comes from by holding the record/query fixed
and changing only the output instruction/protocol.

No learned classifier is used. The main observables are route scores:
  S_route = max lm_head score over route token set
and especially:
  prose_minus_value = S_prose - S_value
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_final_norm  # noqa: E402


CONTROL_PATH = Path(
    "results/glm5_phase670_graph_atlas_counterfactual_control_set/phase670_counterfactual_control_set.json"
)
OUT_ROOT = Path("results/glm5_phase683_prose_route_bias_source_decomposition")

SOURCE_FAMILIES = {
    "same_format_random_value": 72,
    "different_value_same_format": 48,
    "same_prefix_different_continuation": 24,
}

VARIANTS = [
    {
        "name": "short_only",
        "instruction": "Answer with only the value.",
        "target_route": "value",
        "expected_mode": "value",
    },
    {
        "name": "terse_no_explain",
        "instruction": "Return exactly the value. Do not explain.",
        "target_route": "value",
        "expected_mode": "value",
    },
    {
        "name": "bare_answer",
        "instruction": None,
        "target_route": "value",
        "expected_mode": "value",
    },
    {
        "name": "sentence",
        "instruction": "Answer in one complete sentence.",
        "target_route": "prose",
        "expected_mode": "sentence",
    },
    {
        "name": "explanation",
        "instruction": "Answer with a short explanation.",
        "target_route": "prose",
        "expected_mode": "explanation",
    },
    {
        "name": "json",
        "instruction": "Return JSON with keys object, relation, value.",
        "target_route": "json",
        "expected_mode": "json",
    },
    {
        "name": "label",
        "instruction": "Write the answer after 'Value:'.",
        "target_route": "label",
        "expected_mode": "label",
    },
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_base_cases() -> list[dict]:
    data = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
    by_family: dict[str, list[dict]] = defaultdict(list)
    for case in data["cases"]:
        if case["family"] in SOURCE_FAMILIES and case.get("format_name") == "short":
            by_family[case["family"]].append(case)
    selected = []
    for family, limit in SOURCE_FAMILIES.items():
        cases = by_family.get(family, [])
        selected.extend(cases[:limit])
    return selected


def split_prompt_base(prompt: str) -> str:
    if "\nInstruction:" in prompt:
        return prompt.split("\nInstruction:", 1)[0]
    if "\nAnswer:" in prompt:
        return prompt.split("\nAnswer:", 1)[0]
    return prompt.rstrip()


def value_phrase(case: dict) -> str:
    return str(case.get("value") or case["expected_output"]).strip()


def expected_for(case: dict, variant: dict) -> str:
    value = value_phrase(case)
    obj = case.get("object_name", "object")
    rel = case.get("relation", "relation")
    mode = variant["expected_mode"]
    if mode == "value":
        return value
    if mode == "sentence":
        return f"The {rel} of {obj} is {value}."
    if mode == "explanation":
        return f"The record states that {obj} has {rel} {value}."
    if mode == "json":
        return f'{{"object":"{obj}","relation":"{rel}","value":"{value}"}}'
    if mode == "label":
        return f"Value: {value}"
    raise ValueError(mode)


def prompt_for(case: dict, variant: dict) -> str:
    base = split_prompt_base(case["prompt"])
    if variant["instruction"] is None:
        return base + "\nAnswer:"
    return base + f"\nInstruction: {variant['instruction']}\nAnswer:"


def encode_first_ids(tokenizer: Any, phrases: list[str]) -> set[int]:
    ids: set[int] = set()
    for phrase in phrases:
        for variant in [phrase, " " + phrase, "\n" + phrase]:
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if toks:
                ids.add(int(toks[0]))
    return ids


def route_id_sets(tokenizer: Any, case: dict, expected_text: str) -> dict[str, set[int]]:
    value = value_phrase(case)
    return {
        "value": encode_first_ids(tokenizer, [value]),
        "prose": encode_first_ids(tokenizer, ["The", "It", "This", "The record", "Record", "Because"]),
        "json": encode_first_ids(tokenizer, ["{", '"', "["]),
        "label": encode_first_ids(tokenizer, ["Value", "Value:"]),
        "list": encode_first_ids(tokenizer, ["-", "- "]),
        "yesno": encode_first_ids(tokenizer, ["yes", "Yes", "no", "No"]),
        "continuation": encode_first_ids(tokenizer, ["\n", " ", ".", ":", ","]),
        "expected": encode_first_ids(tokenizer, [expected_text]),
    }


def expected_first_ids(tokenizer: Any, expected_text: str) -> set[int]:
    return encode_first_ids(tokenizer, [expected_text])


def selected_layers(n_layers: int) -> list[int]:
    idxs = set()
    for li in range(17, 23):
        if 0 <= li < n_layers:
            idxs.add(li)
    for r in [0.45, 0.50, 0.55, 0.60, 0.65]:
        idxs.add(max(0, min(n_layers - 1, round((n_layers - 1) * r))))
    return sorted(idxs)


def capture_states(model, tokenizer, device, prompt: str, layer_indices: list[int]) -> dict:
    layers = get_layers(model)
    final_norm = get_final_norm(model)
    captured: dict[str, Any] = {"layer_out": {}}
    handles = []
    for li in layer_indices:
        layer = layers[li]

        def layer_out(_module, _inputs, output, layer_idx=li):
            y = extract_tensor(output)
            captured["layer_out"][layer_idx] = y[0, -1].detach().float().cpu()

        handles.append(layer.register_forward_hook(layer_out))

    if final_norm is not None:
        def norm_pre(_module, inputs):
            captured["final_norm_input"] = inputs[0][0, -1].detach().float().cpu()

        def norm_out(_module, _inputs, output):
            y = extract_tensor(output)
            captured["final_norm_output"] = y[0, -1].detach().float().cpu()

        handles.append(final_norm.register_forward_pre_hook(norm_pre))
        handles.append(final_norm.register_forward_hook(norm_out))

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True)
        captured["logits"] = out.logits[0, -1].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()
    return captured


def logits_from_state(model, state: torch.Tensor | None) -> torch.Tensor | None:
    if state is None:
        return None
    emb = model.get_output_embeddings()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        return emb(state.to(device=device, dtype=dtype).unsqueeze(0)).squeeze(0).float().detach().cpu()


def route_scores(logits: torch.Tensor, routes: dict[str, set[int]]) -> dict[str, float]:
    scores = {}
    for route, ids in routes.items():
        valid = [tid for tid in ids if 0 <= tid < logits.numel()]
        scores[route] = max((float(logits[tid].item()) for tid in valid), default=float("-inf"))
    return scores


def route_diag(scores: dict[str, float], target_route: str) -> dict:
    target_score = scores[target_route]
    non_target = {k: v for k, v in scores.items() if k not in {target_route, "expected"}}
    best_other, best_other_score = max(non_target.items(), key=lambda kv: kv[1])
    sorted_routes = sorted({k: v for k, v in scores.items() if k != "expected"}.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "target_route": target_route,
        "target_score": target_score,
        "best_other_route": best_other,
        "best_other_score": best_other_score,
        "target_margin": target_score - best_other_score,
        "target_rank": 1 + [k for k, _ in sorted_routes].index(target_route),
        "prose_minus_value": scores["prose"] - scores["value"],
        "scores": scores,
    }


def best_expected(logits: torch.Tensor, ids: set[int]) -> tuple[int, int]:
    best_id = max(ids, key=lambda tid: float(logits[tid].item()))
    rank = int((logits > logits[best_id]).sum().item()) + 1
    return int(best_id), rank


def run_model(args) -> dict:
    base_cases = select_base_cases()
    model, tokenizer, device = load_model_flash(args.model)
    rows = []
    try:
        n_layers = len(get_layers(model))
        layer_indices = selected_layers(n_layers)
        total = len(base_cases) * len(VARIANTS)
        done = 0
        for case in base_cases:
            for variant in VARIANTS:
                prompt = prompt_for(case, variant)
                expected_text = expected_for(case, variant)
                exp_ids = expected_first_ids(tokenizer, expected_text)
                routes = route_id_sets(tokenizer, case, expected_text)
                captured = capture_states(model, tokenizer, device, prompt, layer_indices)
                logits = captured["logits"]
                expected_id, expected_rank = best_expected(logits, exp_ids)
                top1_id = int(torch.argmax(logits).item())
                final_diag = route_diag(route_scores(logits, routes), variant["target_route"])
                layer_diags = {}
                for li in layer_indices:
                    layer_logits = logits_from_state(model, captured["layer_out"].get(li))
                    if layer_logits is not None:
                        layer_diags[str(li)] = route_diag(route_scores(layer_logits, routes), variant["target_route"])
                fn_in_logits = logits_from_state(model, captured.get("final_norm_input"))
                fn_out_logits = logits_from_state(model, captured.get("final_norm_output"))
                fn_in_diag = route_diag(route_scores(fn_in_logits, routes), variant["target_route"]) if fn_in_logits is not None else None
                fn_out_diag = route_diag(route_scores(fn_out_logits, routes), variant["target_route"]) if fn_out_logits is not None else None
                protocol_margins = [d["target_margin"] for d in layer_diags.values()]
                protocol_pmv = [d["prose_minus_value"] for d in layer_diags.values()]
                row = {
                    "case_id": case["case_id"],
                    "family": case["family"],
                    "object_name": case.get("object_name"),
                    "relation": case.get("relation"),
                    "value": value_phrase(case),
                    "variant": variant["name"],
                    "target_route": variant["target_route"],
                    "expected_text": expected_text,
                    "expected_id": expected_id,
                    "expected_rank": expected_rank,
                    "expected_top1": expected_rank == 1,
                    "top1_id": top1_id,
                    "top1_text": tokenizer.decode([top1_id]),
                    "layer_indices": layer_indices,
                    "layer_route_diags": layer_diags,
                    "final_norm_input_diag": fn_in_diag,
                    "final_norm_output_diag": fn_out_diag,
                    "final_diag": final_diag,
                    "features": {
                        "protocol_mean_target_margin": sum(protocol_margins) / max(1, len(protocol_margins)),
                        "protocol_min_target_margin": min(protocol_margins) if protocol_margins else None,
                        "protocol_mean_prose_minus_value": sum(protocol_pmv) / max(1, len(protocol_pmv)),
                        "protocol_max_prose_minus_value": max(protocol_pmv) if protocol_pmv else None,
                        "final_norm_input_prose_minus_value": fn_in_diag["prose_minus_value"] if fn_in_diag else None,
                        "final_prose_minus_value": final_diag["prose_minus_value"],
                        "final_target_margin": final_diag["target_margin"],
                    },
                }
                rows.append(row)
                done += 1
                if done % args.log_every == 0 or done == total:
                    log(f"{args.model}: {done}/{total} variant cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase683_{args.model}_prose_bias_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 683,
        "title": "Prose-Route Bias Source Decomposition",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "n_cases": len(rows),
        "base_cases": len(base_cases),
        "variants": [v["name"] for v in VARIANTS],
        "summary": summary,
    }
    (OUT_ROOT / f"phase683_{args.model}_prose_bias_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def summarize_subset(rows: list[dict]) -> dict:
    n = len(rows)
    cats = Counter(r["final_diag"]["best_other_route"] for r in rows if not r["expected_top1"])
    target_routes = Counter(r["target_route"] for r in rows)
    return {
        "n": n,
        "expected_top1_rate": sum(1 for r in rows if r["expected_top1"]) / max(1, n),
        "mean_expected_rank": sum(r["expected_rank"] for r in rows) / max(1, n),
        "mean_protocol_prose_minus_value": sum(r["features"]["protocol_mean_prose_minus_value"] for r in rows) / max(1, n),
        "mean_final_norm_input_prose_minus_value": sum(r["features"]["final_norm_input_prose_minus_value"] for r in rows) / max(1, n),
        "mean_final_prose_minus_value": sum(r["features"]["final_prose_minus_value"] for r in rows) / max(1, n),
        "mean_final_target_margin": sum(r["features"]["final_target_margin"] for r in rows) / max(1, n),
        "prose_best_route_rate": sum(1 for r in rows if r["final_diag"]["best_other_route"] == "prose") / max(1, n),
        "failure_best_other_route": dict(cats.most_common()),
        "target_routes": dict(target_routes.most_common()),
    }


def summarize_model(model: str, rows: list[dict]) -> dict:
    variants = sorted({r["variant"] for r in rows})
    families = sorted({r["family"] for r in rows})
    by_variant = {v: summarize_subset([r for r in rows if r["variant"] == v]) for v in variants}
    by_family_variant = {}
    for fam in families:
        by_family_variant[fam] = {
            v: summarize_subset([r for r in rows if r["family"] == fam and r["variant"] == v])
            for v in variants
        }
    value_variants = {"short_only", "terse_no_explain", "bare_answer"}
    value_rows = [r for r in rows if r["variant"] in value_variants]
    return {
        "model": model,
        "overall": summarize_subset(rows),
        "value_target_variants": summarize_subset(value_rows),
        "by_variant": by_variant,
        "by_family_variant": by_family_variant,
        "key_contrasts": build_contrasts(by_variant),
    }


def build_contrasts(by_variant: dict[str, dict]) -> dict:
    def get(v: str, key: str) -> float:
        return by_variant.get(v, {}).get(key, 0.0)
    return {
        "terse_minus_short_final_pmv": get("terse_no_explain", "mean_final_prose_minus_value") - get("short_only", "mean_final_prose_minus_value"),
        "bare_minus_short_final_pmv": get("bare_answer", "mean_final_prose_minus_value") - get("short_only", "mean_final_prose_minus_value"),
        "sentence_minus_short_final_pmv": get("sentence", "mean_final_prose_minus_value") - get("short_only", "mean_final_prose_minus_value"),
        "explanation_minus_short_final_pmv": get("explanation", "mean_final_prose_minus_value") - get("short_only", "mean_final_prose_minus_value"),
        "json_minus_short_final_pmv": get("json", "mean_final_prose_minus_value") - get("short_only", "mean_final_prose_minus_value"),
        "label_minus_short_final_pmv": get("label", "mean_final_prose_minus_value") - get("short_only", "mean_final_prose_minus_value"),
    }


def write_cross_summary() -> dict:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase683_*_prose_bias_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 683,
        "title": "Prose-Route Bias Source Decomposition Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase683_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 683 Prose-Route Bias Source Decomposition",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | rows | value_top1 | value_final_pmv | short_top1 | short_final_pmv | terse_final_pmv | bare_final_pmv | sentence_final_pmv | explanation_final_pmv | json_final_pmv | label_final_pmv |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in models:
        s = item["summary"]
        by = s["by_variant"]

        def val(variant: str, key: str) -> float:
            return by.get(variant, {}).get(key, 0.0)

        lines.append(
            f"| {item['model']} | {item['n_cases']} | "
            f"{s['value_target_variants']['expected_top1_rate']:.3f} | "
            f"{s['value_target_variants']['mean_final_prose_minus_value']:.3f} | "
            f"{val('short_only', 'expected_top1_rate'):.3f} | {val('short_only', 'mean_final_prose_minus_value'):.3f} | "
            f"{val('terse_no_explain', 'mean_final_prose_minus_value'):.3f} | "
            f"{val('bare_answer', 'mean_final_prose_minus_value'):.3f} | "
            f"{val('sentence', 'mean_final_prose_minus_value'):.3f} | "
            f"{val('explanation', 'mean_final_prose_minus_value'):.3f} | "
            f"{val('json', 'mean_final_prose_minus_value'):.3f} | "
            f"{val('label', 'mean_final_prose_minus_value'):.3f} |"
        )
    lines.extend(["", "## Variant Details", ""])
    for item in models:
        lines.append(f"### {item['model']}")
        lines.append("")
        lines.append("| variant | n | top1 | mean_rank | protocol_pmv | final_norm_input_pmv | final_pmv | final_target_margin | failure_best_other |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for variant, row in item["summary"]["by_variant"].items():
            lines.append(
                f"| {variant} | {row['n']} | {row['expected_top1_rate']:.3f} | "
                f"{row['mean_expected_rank']:.2f} | {row['mean_protocol_prose_minus_value']:.3f} | "
                f"{row['mean_final_norm_input_prose_minus_value']:.3f} | {row['mean_final_prose_minus_value']:.3f} | "
                f"{row['mean_final_target_margin']:.3f} | {row['failure_best_other_route']} |"
            )
        lines.append("")
    (OUT_ROOT / "phase683_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--log-every", type=int, default=72)
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
