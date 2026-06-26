#!/usr/bin/env python3
"""
Phase 680: Pre-Readout Natural Gate and Cross-Family Generalization Audit.

This phase asks whether the Phase 679 near-readout failure gate can be moved
earlier, before final logits/top1. It intentionally avoids learned classifiers:
only simple threshold gates over diagnostic pre-readout features are enumerated.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
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
OUT_ROOT = Path("results/glm5_phase680_pre_readout_natural_gate_cross_family_audit")

DEFAULT_FAMILY_LIMITS = {
    "same_format_random_value": 72,
    "same_value_different_format": 144,
    "different_value_same_format": 48,
    "same_prefix_different_continuation": 24,
    "factor_isolation": 54,
}

PRE_FEATURE_KINDS = {
    "final_norm_input_gap": "pre_final_norm",
    "final_norm_input_rank": "pre_final_norm",
    "last_layer_gap": "late_residual",
    "late_gap_shift": "late_residual",
    "mid_to_late_shift": "late_residual",
    "max_layer_gap": "trajectory",
    "min_layer_gap": "trajectory",
    "positive_layer_count": "trajectory",
    "first_positive_layer_frac": "trajectory",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def expected_variants(text: str) -> list[str]:
    return [text, " " + text, "\n" + text]


def encode_expected(tokenizer: Any, text: str) -> list[list[int]]:
    out = []
    seen = set()
    for variant in expected_variants(text):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids and tuple(ids) not in seen:
            seen.add(tuple(ids))
            out.append(ids)
    return out


def token_category(text: str, expected_ids: set[int], tid: int) -> str:
    s = text.strip()
    if tid in expected_ids:
        return "expected"
    if text == " " or (text.isspace() and "\n" not in text):
        return "space"
    if "\n" in text:
        return "newline"
    if s.startswith(("{", "[", '"')):
        return "json_or_quote"
    if s.startswith(("The", "the", "Record", "You", "I", "It", "This", "A", "An")):
        return "word_or_explanation"
    if s in {":", ".", ",", ";"}:
        return "punctuation"
    if not s:
        return "blank"
    return "other"


def parse_family_limits(raw: str | None) -> dict[str, int]:
    if not raw:
        return dict(DEFAULT_FAMILY_LIMITS)
    limits = dict(DEFAULT_FAMILY_LIMITS)
    for item in raw.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        limits[key.strip()] = int(value)
    return limits


def select_cases(family_limits: dict[str, int]) -> list[dict]:
    data = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
    by_family: dict[str, list[dict]] = defaultdict(list)
    for case in data["cases"]:
        by_family[case["family"]].append(case)
    selected = []
    for family, limit in family_limits.items():
        cases = by_family.get(family, [])
        selected.extend(cases if limit <= 0 else cases[:limit])
    return selected


def selected_layers(n_layers: int) -> list[int]:
    rel = [0.45, 0.55, 0.65, 0.75, 0.85, 0.92]
    idxs = {max(0, min(n_layers - 1, round((n_layers - 1) * r))) for r in rel}
    idxs.update(range(max(0, n_layers - 6), n_layers))
    # Include DS7B-inspired absolute protocol/bridge layers when they exist.
    for li in [17, 18, 19, 20, 22, 25, 26, 27]:
        if 0 <= li < n_layers:
            idxs.add(li)
    return sorted(idxs)


def capture_states(model, tokenizer, device, prompt: str, layer_indices: list[int]) -> dict:
    layers = get_layers(model)
    final_norm = get_final_norm(model)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    captured: dict[str, Any] = {"layer_out": {}, "layer_in": {}}
    handles = []

    for li in layer_indices:
        layer = layers[li]

        def layer_pre(_module, inputs, layer_idx=li):
            captured["layer_in"][layer_idx] = inputs[0][0, -1].detach().float().cpu()

        def layer_out(_module, _inputs, output, layer_idx=li):
            y = extract_tensor(output)
            captured["layer_out"][layer_idx] = y[0, -1].detach().float().cpu()

        handles.append(layer.register_forward_pre_hook(layer_pre))
        handles.append(layer.register_forward_hook(layer_out))

    if final_norm is not None:
        def norm_pre(_module, inputs):
            captured["final_norm_input"] = inputs[0][0, -1].detach().float().cpu()

        def norm_out(_module, _inputs, output):
            y = extract_tensor(output)
            captured["final_norm_output"] = y[0, -1].detach().float().cpu()

        handles.append(final_norm.register_forward_pre_hook(norm_pre))
        handles.append(final_norm.register_forward_hook(norm_out))

    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True)
        captured["logits"] = out.logits[0, -1].detach().float().cpu()
        captured["n_tokens"] = len(ids)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def logits_from_state(model, state: torch.Tensor | None) -> torch.Tensor | None:
    if state is None:
        return None
    emb = model.get_output_embeddings()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        return emb(state.to(device=device, dtype=dtype).unsqueeze(0)).squeeze(0).float().detach().cpu()


def best_expected(logits: torch.Tensor, expected_ids: set[int]) -> tuple[int, float, int]:
    best_id = max(expected_ids, key=lambda tid: float(logits[tid].item()))
    best_score = float(logits[best_id].item())
    rank = int((logits > logits[best_id]).sum().item()) + 1
    return int(best_id), best_score, rank


def top_competitor(tokenizer, logits: torch.Tensor, expected_ids: set[int], top_k: int) -> dict:
    vals, ids = torch.topk(logits, k=min(top_k, logits.numel()))
    top = []
    comp = None
    for rank, (score, tid_t) in enumerate(zip(vals.tolist(), ids.tolist()), start=1):
        tid = int(tid_t)
        text = tokenizer.decode([tid])
        cat = token_category(text, expected_ids, tid)
        item = {"rank": rank, "id": tid, "text": text, "score": float(score), "category": cat}
        top.append(item)
        if cat != "expected" and comp is None:
            comp = item
    return comp or top[0]


def gap_and_rank(logits: torch.Tensor | None, expected_id: int, competitor_id: int) -> tuple[float | None, int | None]:
    if logits is None:
        return None, None
    gap = float(logits[competitor_id].item() - logits[expected_id].item())
    rank = int((logits > logits[expected_id]).sum().item()) + 1
    return gap, rank


def finite(value: Any) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


def run_model(args) -> dict:
    family_limits = parse_family_limits(args.family_limits)
    cases = select_cases(family_limits)
    model, tokenizer, device = load_model_flash(args.model)
    rows = []
    try:
        n_layers = len(get_layers(model))
        layer_indices = selected_layers(n_layers)
        for i, case in enumerate(cases):
            expected_seqs = encode_expected(tokenizer, case["expected_output"])
            expected_ids = {seq[0] for seq in expected_seqs if seq}
            if not expected_ids:
                continue
            captured = capture_states(model, tokenizer, device, case["prompt"], layer_indices)
            logits = captured["logits"]
            expected_id, expected_score, expected_rank = best_expected(logits, expected_ids)
            competitor = top_competitor(tokenizer, logits, expected_ids, args.top_k)
            competitor_id = int(competitor["id"])
            top1_id = int(torch.argmax(logits).item())
            top1_text = tokenizer.decode([top1_id])
            top1_category = token_category(top1_text, expected_ids, top1_id)
            final_gap = float(logits[competitor_id].item() - logits[expected_id].item())

            fn_input_logits = logits_from_state(model, captured.get("final_norm_input"))
            fn_output_logits = logits_from_state(model, captured.get("final_norm_output"))
            final_norm_input_gap, final_norm_input_rank = gap_and_rank(fn_input_logits, expected_id, competitor_id)
            final_norm_output_gap, _ = gap_and_rank(fn_output_logits, expected_id, competitor_id)

            layer_gaps = {}
            layer_ranks = {}
            for li in layer_indices:
                layer_logits = logits_from_state(model, captured["layer_out"].get(li))
                gap, rank = gap_and_rank(layer_logits, expected_id, competitor_id)
                layer_gaps[str(li)] = gap
                layer_ranks[str(li)] = rank

            valid_gaps = [(int(k), v) for k, v in layer_gaps.items() if finite(v)]
            valid_gaps.sort()
            first_positive = None
            for li, gap in valid_gaps:
                if gap > 0:
                    first_positive = li / max(1, n_layers - 1)
                    break
            mid_idx = layer_indices[len(layer_indices) // 2]
            last_idx = layer_indices[-1]
            mid_gap = layer_gaps.get(str(mid_idx))
            last_gap = layer_gaps.get(str(last_idx))
            features = {
                "final_norm_input_gap": final_norm_input_gap,
                "final_norm_input_rank": final_norm_input_rank,
                "final_norm_shift": (
                    final_gap - final_norm_input_gap
                    if finite(final_gap) and finite(final_norm_input_gap)
                    else None
                ),
                "final_norm_output_proxy_gap": final_norm_output_gap,
                "last_layer_gap": last_gap,
                "late_gap_shift": (
                    final_norm_input_gap - last_gap
                    if finite(final_norm_input_gap) and finite(last_gap)
                    else None
                ),
                "mid_to_late_shift": (
                    last_gap - mid_gap
                    if finite(last_gap) and finite(mid_gap)
                    else None
                ),
                "max_layer_gap": max((g for _, g in valid_gaps), default=None),
                "min_layer_gap": min((g for _, g in valid_gaps), default=None),
                "positive_layer_count": sum(1 for _, g in valid_gaps if g > 0),
                "first_positive_layer_frac": first_positive if first_positive is not None else 2.0,
            }

            row = {
                "case_id": case["case_id"],
                "family": case["family"],
                "axis": case.get("axis"),
                "format_name": case.get("format_name"),
                "relation": case.get("relation"),
                "expected_output": case["expected_output"],
                "expected_id": expected_id,
                "expected_text": tokenizer.decode([expected_id]),
                "expected_rank": expected_rank,
                "expected_top1": expected_rank == 1,
                "top1_id": top1_id,
                "top1_text": top1_text,
                "top1_category": top1_category,
                "competitor": competitor,
                "final_gap": final_gap,
                "n_layers": n_layers,
                "layer_indices": layer_indices,
                "layer_gaps": layer_gaps,
                "layer_ranks": layer_ranks,
                "pre_features": features,
            }
            rows.append(row)
            if (i + 1) % args.log_every == 0 or i + 1 == len(cases):
                log(f"{args.model}: {i + 1}/{len(cases)} cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(args.model, rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_ROOT / f"phase680_{args.model}_pre_readout_rows.jsonl"
    rows_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 680,
        "title": "Pre-Readout Natural Gate and Cross-Family Generalization Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "family_limits": family_limits,
        "n_cases": len(rows),
        "summary": summary,
    }
    summary_path = OUT_ROOT / f"phase680_{args.model}_pre_readout_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def thresholds(values: list[float]) -> list[float]:
    vals = sorted({float(v) for v in values if math.isfinite(float(v))})
    if not vals:
        return []
    idxs = {0, len(vals) // 4, len(vals) // 2, (3 * len(vals)) // 4, len(vals) - 1}
    out = {0.0}
    for idx in idxs:
        out.add(vals[idx])
    for i in range(len(vals) - 1):
        if vals[i] <= 0 <= vals[i + 1]:
            out.add((vals[i] + vals[i + 1]) / 2.0)
            break
    return sorted(out)


def make_gates(rows: list[dict], include_reference: bool) -> list[dict]:
    gates = []
    for feature, kind in PRE_FEATURE_KINDS.items():
        vals = [r["pre_features"].get(feature) for r in rows if finite(r["pre_features"].get(feature))]
        for t in thresholds(vals):
            gates.append({"name": f"{feature}_gt_{t:.4g}", "kind": kind, "feature": feature, "op": ">", "threshold": t})
            gates.append({"name": f"{feature}_lt_{t:.4g}", "kind": kind, "feature": feature, "op": "<", "threshold": t})
    if include_reference:
        vals = [r["final_gap"] for r in rows if finite(r.get("final_gap"))]
        for t in thresholds(vals):
            gates.append({"name": f"REF_final_gap_gt_{t:.4g}", "kind": "near_readout_reference", "feature": "final_gap", "op": ">", "threshold": t})
        gates.append({"name": "REF_top1_category_not_expected", "kind": "near_readout_reference", "feature": "top1_category", "op": "!=", "threshold": "expected"})
    return gates


def gate_fire(row: dict, gate: dict) -> bool:
    if gate["feature"] == "top1_category":
        return row.get("top1_category") != gate["threshold"]
    if gate["feature"] == "final_gap":
        value = row.get("final_gap")
    else:
        value = row["pre_features"].get(gate["feature"])
    if not finite(value):
        return False
    if gate["op"] == ">":
        return float(value) > float(gate["threshold"])
    return float(value) < float(gate["threshold"])


def eval_gate(rows: list[dict], gate: dict) -> dict:
    n = len(rows)
    failures = sum(1 for r in rows if not r["expected_top1"])
    successes = n - failures
    predicted = [r for r in rows if gate_fire(r, gate)]
    predicted_failures = sum(1 for r in predicted if not r["expected_top1"])
    predicted_successes = len(predicted) - predicted_failures
    precision = predicted_failures / max(1, len(predicted))
    capture = predicted_failures / max(1, failures)
    false_pos = predicted_successes / max(1, successes)
    return {
        "gate": gate["name"],
        "kind": gate["kind"],
        "feature": gate["feature"],
        "op": gate["op"],
        "threshold": gate["threshold"],
        "n": n,
        "failures": failures,
        "successes": successes,
        "predicted_count": len(predicted),
        "predicted_rate": len(predicted) / max(1, n),
        "failure_capture_rate": capture,
        "success_false_positive_rate": false_pos,
        "failure_precision": precision,
        "gate_score": capture - false_pos,
    }


def rank_evals(evals: list[dict]) -> list[dict]:
    return sorted(
        evals,
        key=lambda r: (
            -r["gate_score"],
            -r["failure_capture_rate"],
            r["success_false_positive_rate"],
            -r["failure_precision"],
            r["predicted_rate"],
        ),
    )


def summarize_rows(model: str, rows: list[dict]) -> dict:
    families = sorted({r["family"] for r in rows})
    all_gates = make_gates(rows, include_reference=True)
    pre_gates = [g for g in all_gates if g["kind"] != "near_readout_reference"]
    ref_gates = [g for g in all_gates if g["kind"] == "near_readout_reference"]

    def compact(rows_subset: list[dict]) -> dict:
        n = len(rows_subset)
        failures = sum(1 for r in rows_subset if not r["expected_top1"])
        cats = Counter(r["top1_category"] for r in rows_subset)
        return {
            "n": n,
            "expected_top1_rate": (n - failures) / max(1, n),
            "failure_rate": failures / max(1, n),
            "mean_expected_rank": sum(r["expected_rank"] for r in rows_subset) / max(1, n),
            "mean_final_gap": sum(r["final_gap"] for r in rows_subset) / max(1, n),
            "top1_category": dict(cats.most_common()),
        }

    baseline = {"overall": compact(rows)}
    for fam in families:
        baseline[fam] = compact([r for r in rows if r["family"] == fam])

    overall_pre = rank_evals([eval_gate(rows, g) for g in pre_gates])
    overall_ref = rank_evals([eval_gate(rows, g) for g in ref_gates])
    by_family = {}
    for fam in families:
        subset = [r for r in rows if r["family"] == fam]
        by_family[fam] = {
            "pre_readout": rank_evals([eval_gate(subset, g) for g in pre_gates])[:15],
            "near_readout_reference": rank_evals([eval_gate(subset, g) for g in ref_gates])[:8],
        }

    cross_family = []
    for source in families:
        src_rows = [r for r in rows if r["family"] == source]
        src_failures = sum(1 for r in src_rows if not r["expected_top1"])
        src_successes = len(src_rows) - src_failures
        if src_failures < 3 or src_successes < 3:
            continue
        src_ranked = rank_evals([eval_gate(src_rows, g) for g in pre_gates])
        source_candidates = [
            e for e in src_ranked
            if e["failure_capture_rate"] >= 0.70
            and e["success_false_positive_rate"] <= 0.30
            and e["predicted_count"] > 0
        ][:5]
        gate_by_name = {g["name"]: g for g in pre_gates}
        for e in source_candidates:
            gate = gate_by_name[e["gate"]]
            for target in families:
                if target == source:
                    continue
                target_rows = [r for r in rows if r["family"] == target]
                te = eval_gate(target_rows, gate)
                cross_family.append({
                    "source_family": source,
                    "target_family": target,
                    "gate": gate["name"],
                    "kind": gate["kind"],
                    "source_failure_capture": e["failure_capture_rate"],
                    "source_false_pos": e["success_false_positive_rate"],
                    "target_failure_capture": te["failure_capture_rate"],
                    "target_false_pos": te["success_false_positive_rate"],
                    "target_failures": te["failures"],
                    "target_successes": te["successes"],
                    "target_score": te["gate_score"],
                })

    return {
        "model": model,
        "baseline": baseline,
        "top_pre_readout_gates": overall_pre[:25],
        "top_near_readout_reference_gates": overall_ref[:10],
        "by_family": by_family,
        "cross_family_generalization": sorted(
            cross_family,
            key=lambda r: (-r["target_score"], -r["target_failure_capture"], r["target_false_pos"]),
        )[:40],
    }


def write_cross_summary() -> dict:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase680_*_pre_readout_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 680,
        "title": "Pre-Readout Natural Gate and Cross-Family Generalization Audit Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase680_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# Phase 680 Pre-Readout Natural Gate and Cross-Family Generalization Audit",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | cases | top1_rate | failures | best pre-readout gate | pre score | pre capture | pre false_pos | best near-readout ref | ref score |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    for item in models:
        model = item["model"]
        overall = item["summary"]["baseline"]["overall"]
        pre = item["summary"]["top_pre_readout_gates"][0] if item["summary"]["top_pre_readout_gates"] else {}
        ref = item["summary"]["top_near_readout_reference_gates"][0] if item["summary"]["top_near_readout_reference_gates"] else {}
        lines.append(
            f"| {model} | {item['n_cases']} | {overall['expected_top1_rate']:.3f} | "
            f"{int(overall['failure_rate'] * item['n_cases'])} | "
            f"{pre.get('gate', 'NA')} | {pre.get('gate_score', 0.0):.3f} | "
            f"{pre.get('failure_capture_rate', 0.0):.3f} | {pre.get('success_false_positive_rate', 0.0):.3f} | "
            f"{ref.get('gate', 'NA')} | {ref.get('gate_score', 0.0):.3f} |"
        )
    lines.extend(["", "## Family Baseline", ""])
    for item in models:
        lines.append(f"### {item['model']}")
        lines.append("")
        lines.append("| family | n | top1_rate | failure_rate | mean_rank | mean_final_gap |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for fam, base in item["summary"]["baseline"].items():
            if fam == "overall":
                continue
            lines.append(
                f"| {fam} | {base['n']} | {base['expected_top1_rate']:.3f} | "
                f"{base['failure_rate']:.3f} | {base['mean_expected_rank']:.2f} | {base['mean_final_gap']:.3f} |"
            )
        lines.append("")
    (OUT_ROOT / "phase680_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--family-limits", default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=24)
    parser.add_argument("--summarize-only", action="store_true")
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
