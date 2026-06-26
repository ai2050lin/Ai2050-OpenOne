#!/usr/bin/env python3
"""
Phase 682: Protocol-Level Failure Monitor Localization.

This phase moves beyond expected-vs-competitor token gap. It asks whether an
earlier protocol-route margin can predict first-token failures:

    target route score - max(non-target route scores)

Routes are simple interpretable sets: value, prose, json, label, list, yesno,
and continuation. No learned classifier is used.
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
OUT_ROOT = Path("results/glm5_phase682_protocol_route_monitor_localization")

DEFAULT_FAMILY_LIMITS = {
    "same_format_random_value": 72,
    "same_value_different_format": 144,
    "different_value_same_format": 48,
    "same_prefix_different_continuation": 24,
    "factor_isolation": 54,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_family_limits(raw: str | None) -> dict[str, int]:
    if not raw:
        return dict(DEFAULT_FAMILY_LIMITS)
    limits = dict(DEFAULT_FAMILY_LIMITS)
    for item in raw.split(","):
        if item.strip():
            key, value = item.split("=", 1)
            limits[key.strip()] = int(value)
    return limits


def select_cases(family_limits: dict[str, int]) -> list[dict]:
    data = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
    by_family: dict[str, list[dict]] = defaultdict(list)
    for case in data["cases"]:
        by_family[case["family"]].append(case)
    out = []
    for fam, limit in family_limits.items():
        cases = by_family.get(fam, [])
        out.extend(cases if limit <= 0 else cases[:limit])
    return out


def encode_first_ids(tokenizer: Any, phrases: list[str]) -> set[int]:
    ids: set[int] = set()
    for phrase in phrases:
        for variant in [phrase, " " + phrase, "\n" + phrase]:
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if toks:
                ids.add(int(toks[0]))
    return ids


def infer_target_route(case: dict) -> str:
    expected = case["expected_output"].lstrip()
    fmt = case.get("format_name")
    axis = case.get("axis")
    if expected.startswith("{") or fmt in {"json", "protocol_only_json"}:
        return "json"
    if expected.startswith("-") or fmt == "list":
        return "list"
    if expected.startswith("Value:") or fmt == "label":
        return "label"
    first_word = expected.lower().split()[0].strip(".,:;!?") if expected.split() else ""
    if first_word in {"yes", "no"} or axis == "intent_only_existence":
        return "yesno"
    if expected.startswith("The") or fmt in {"sentence", "explanation"}:
        return "prose"
    return "value"


def value_phrase(case: dict) -> str:
    if case.get("value"):
        return str(case["value"])
    expected = case["expected_output"].strip()
    if expected.startswith("Value:"):
        return expected.split(":", 1)[1].strip()
    if expected.startswith("-"):
        return expected[1:].strip()
    return expected


def route_id_sets(tokenizer: Any, case: dict) -> dict[str, set[int]]:
    val = value_phrase(case)
    return {
        "value": encode_first_ids(tokenizer, [val]),
        "prose": encode_first_ids(tokenizer, ["The", "It", "This", "The record", "Record"]),
        "json": encode_first_ids(tokenizer, ["{", '"', "["]),
        "label": encode_first_ids(tokenizer, ["Value", "Value:"]),
        "list": encode_first_ids(tokenizer, ["-", "- "]),
        "yesno": encode_first_ids(tokenizer, ["yes", "Yes", "no", "No"]),
        "continuation": encode_first_ids(tokenizer, ["\n", " ", ".", ":", ","]),
    }


def expected_first_ids(tokenizer: Any, text: str) -> set[int]:
    return encode_first_ids(tokenizer, [text])


def selected_protocol_layers(n_layers: int) -> list[int]:
    idxs = set()
    for li in range(17, 23):
        if 0 <= li < n_layers:
            idxs.add(li)
    # Fallback relative layers for smaller or architecture-shifted models.
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


def route_scores(logits: torch.Tensor, routes: dict[str, set[int]]) -> dict[str, float]:
    scores = {}
    for route, ids in routes.items():
        valid = [tid for tid in ids if 0 <= tid < logits.numel()]
        if valid:
            scores[route] = float(torch.tensor([logits[tid].item() for tid in valid]).max().item())
        else:
            scores[route] = float("-inf")
    return scores


def route_diag(scores: dict[str, float], target_route: str) -> dict:
    target_score = scores.get(target_route, float("-inf"))
    competitors = {k: v for k, v in scores.items() if k != target_route}
    best_other_route, best_other_score = max(competitors.items(), key=lambda kv: kv[1])
    sorted_routes = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    target_rank = 1 + [k for k, _ in sorted_routes].index(target_route)
    return {
        "target_route": target_route,
        "target_score": target_score,
        "best_other_route": best_other_route,
        "best_other_score": best_other_score,
        "route_margin": target_score - best_other_score,
        "target_route_rank": target_rank,
        "route_scores": scores,
    }


def best_expected_rank(logits: torch.Tensor, expected_ids: set[int]) -> tuple[int, int]:
    best_id = max(expected_ids, key=lambda tid: float(logits[tid].item()))
    rank = int((logits > logits[best_id]).sum().item()) + 1
    return int(best_id), rank


def run_model(args) -> dict:
    family_limits = parse_family_limits(args.family_limits)
    cases = select_cases(family_limits)
    model, tokenizer, device = load_model_flash(args.model)
    rows = []
    try:
        n_layers = len(get_layers(model))
        layer_indices = selected_protocol_layers(n_layers)
        for i, case in enumerate(cases):
            target_route = infer_target_route(case)
            routes = route_id_sets(tokenizer, case)
            exp_ids = expected_first_ids(tokenizer, case["expected_output"])
            if not exp_ids or not routes.get(target_route):
                continue
            captured = capture_states(model, tokenizer, device, case["prompt"], layer_indices)
            final_logits = captured["logits"]
            expected_id, expected_rank = best_expected_rank(final_logits, exp_ids)
            top1_id = int(torch.argmax(final_logits).item())
            final_diag = route_diag(route_scores(final_logits, routes), target_route)

            layer_diags = {}
            margins = []
            for li in layer_indices:
                logits = logits_from_state(model, captured["layer_out"].get(li))
                if logits is None:
                    continue
                diag = route_diag(route_scores(logits, routes), target_route)
                layer_diags[str(li)] = diag
                margins.append((li, diag["route_margin"]))

            fn_input_logits = logits_from_state(model, captured.get("final_norm_input"))
            fn_input_diag = route_diag(route_scores(fn_input_logits, routes), target_route) if fn_input_logits is not None else None
            min_margin = min((m for _, m in margins), default=None)
            max_margin = max((m for _, m in margins), default=None)
            last_margin = margins[-1][1] if margins else None
            first_negative_frac = 2.0
            for li, margin in margins:
                if margin < 0:
                    first_negative_frac = li / max(1, n_layers - 1)
                    break
            features = {
                "protocol_min_margin": min_margin,
                "protocol_max_margin": max_margin,
                "protocol_last_margin": last_margin,
                "protocol_negative_count": sum(1 for _, m in margins if m < 0),
                "protocol_first_negative_frac": first_negative_frac,
                "protocol_final_norm_input_margin": fn_input_diag["route_margin"] if fn_input_diag else None,
                "protocol_final_norm_input_rank": fn_input_diag["target_route_rank"] if fn_input_diag else None,
                "protocol_late_shift": (
                    fn_input_diag["route_margin"] - last_margin
                    if fn_input_diag is not None and last_margin is not None
                    else None
                ),
            }
            rows.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "axis": case.get("axis"),
                "format_name": case.get("format_name"),
                "expected_output": case["expected_output"],
                "target_route": target_route,
                "expected_id": expected_id,
                "expected_rank": expected_rank,
                "expected_top1": expected_rank == 1,
                "top1_id": top1_id,
                "layer_indices": layer_indices,
                "layer_route_diags": layer_diags,
                "final_route_diag": final_diag,
                "final_norm_input_route_diag": fn_input_diag,
                "protocol_features": features,
            })
            if (i + 1) % args.log_every == 0 or i + 1 == len(cases):
                log(f"{args.model}: {i + 1}/{len(cases)} cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase682_{args.model}_protocol_route_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 682,
        "title": "Protocol-Level Failure Monitor Localization",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "family_limits": family_limits,
        "n_cases": len(rows),
        "summary": summary,
    }
    (OUT_ROOT / f"phase682_{args.model}_protocol_route_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


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
            out.add((vals[i] + vals[i + 1]) / 2)
            break
    return sorted(out)


FEATURE_KINDS = {
    "protocol_min_margin": "protocol_layers",
    "protocol_max_margin": "protocol_layers",
    "protocol_last_margin": "protocol_layers",
    "protocol_negative_count": "protocol_layers",
    "protocol_first_negative_frac": "protocol_layers",
    "protocol_final_norm_input_margin": "pre_final_norm_route",
    "protocol_final_norm_input_rank": "pre_final_norm_route",
    "protocol_late_shift": "protocol_to_final_norm",
}


def make_gates(rows: list[dict], include_reference: bool) -> list[dict]:
    gates = []
    for feature, kind in FEATURE_KINDS.items():
        vals = [r["protocol_features"].get(feature) for r in rows if finite(r["protocol_features"].get(feature))]
        for t in thresholds(vals):
            gates.append({"name": f"{feature}_gt_{t:.4g}", "kind": kind, "feature": feature, "op": ">", "threshold": t})
            gates.append({"name": f"{feature}_lt_{t:.4g}", "kind": kind, "feature": feature, "op": "<", "threshold": t})
    if include_reference:
        vals = [r["final_route_diag"]["route_margin"] for r in rows if finite(r["final_route_diag"]["route_margin"])]
        for t in thresholds(vals):
            gates.append({"name": f"REF_final_route_margin_lt_{t:.4g}", "kind": "near_readout_reference", "feature": "final_route_margin", "op": "<", "threshold": t})
        gates.append({"name": "REF_final_route_rank_gt_1", "kind": "near_readout_reference", "feature": "final_route_rank", "op": ">", "threshold": 1})
    return gates


def gate_fire(row: dict, gate: dict) -> bool:
    if gate["feature"] == "final_route_margin":
        value = row["final_route_diag"]["route_margin"]
    elif gate["feature"] == "final_route_rank":
        value = row["final_route_diag"]["target_route_rank"]
    else:
        value = row["protocol_features"].get(gate["feature"])
    if not finite(value):
        return False
    return float(value) > float(gate["threshold"]) if gate["op"] == ">" else float(value) < float(gate["threshold"])


def eval_gate(rows: list[dict], gate: dict) -> dict:
    n = len(rows)
    failures = sum(1 for r in rows if not r["expected_top1"])
    successes = n - failures
    pred = [r for r in rows if gate_fire(r, gate)]
    pred_fail = sum(1 for r in pred if not r["expected_top1"])
    pred_success = len(pred) - pred_fail
    capture = pred_fail / max(1, failures)
    false_pos = pred_success / max(1, successes)
    precision = pred_fail / max(1, len(pred))
    return {
        "gate": gate["name"],
        "kind": gate["kind"],
        "feature": gate["feature"],
        "op": gate["op"],
        "threshold": gate["threshold"],
        "n": n,
        "failures": failures,
        "successes": successes,
        "predicted_count": len(pred),
        "predicted_rate": len(pred) / max(1, n),
        "failure_capture_rate": capture,
        "success_false_positive_rate": false_pos,
        "failure_precision": precision,
        "gate_score": capture - false_pos,
    }


def rank_evals(items: list[dict]) -> list[dict]:
    return sorted(
        items,
        key=lambda r: (
            -r["gate_score"],
            -r["failure_capture_rate"],
            r["success_false_positive_rate"],
            -r["failure_precision"],
            r["predicted_rate"],
        ),
    )


def split_alternate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    rows = sorted(rows, key=lambda r: r["case_id"])
    return [r for i, r in enumerate(rows) if i % 2 == 0], [r for i, r in enumerate(rows) if i % 2 == 1]


def best_train_test(train: list[dict], test: list[dict], include_reference: bool) -> dict | None:
    gates = make_gates(train, include_reference=include_reference)
    gates = [g for g in gates if (g["kind"] == "near_readout_reference") == include_reference]
    if not gates:
        return None
    ranked = rank_evals([eval_gate(train, g) for g in gates])
    best = ranked[0]
    gate = next(g for g in gates if g["name"] == best["gate"])
    return {"gate": gate["name"], "kind": gate["kind"], "train": best, "test": eval_gate(test, gate)}


def summarize_subset(rows: list[dict]) -> dict:
    n = len(rows)
    failures = sum(1 for r in rows if not r["expected_top1"])
    route_counts = Counter(r["target_route"] for r in rows)
    wrong_route_counts = Counter(
        r["final_route_diag"]["best_other_route"]
        for r in rows
        if not r["expected_top1"]
    )
    return {
        "n": n,
        "expected_top1_rate": (n - failures) / max(1, n),
        "failure_rate": failures / max(1, n),
        "mean_expected_rank": sum(r["expected_rank"] for r in rows) / max(1, n),
        "mean_final_route_margin": sum(r["final_route_diag"]["route_margin"] for r in rows) / max(1, n),
        "target_route": dict(route_counts.most_common()),
        "failure_best_other_route": dict(wrong_route_counts.most_common()),
    }


def summarize_model(model: str, rows: list[dict]) -> dict:
    families = sorted({r["family"] for r in rows})
    baseline = {"overall": summarize_subset(rows)}
    for fam in families:
        baseline[fam] = summarize_subset([r for r in rows if r["family"] == fam])

    proto_gates = [g for g in make_gates(rows, include_reference=False) if g["kind"] != "near_readout_reference"]
    ref_gates = [g for g in make_gates(rows, include_reference=True) if g["kind"] == "near_readout_reference"]
    overall_protocol = rank_evals([eval_gate(rows, g) for g in proto_gates])[:25]
    overall_reference = rank_evals([eval_gate(rows, g) for g in ref_gates])[:10]

    holdout = {}
    for group, group_rows in {"overall": rows, **{f: [r for r in rows if r["family"] == f] for f in families}}.items():
        train, test = split_alternate(group_rows)
        if sum(1 for r in train if not r["expected_top1"]) >= 3 and sum(1 for r in train if r["expected_top1"]) >= 3:
            holdout[group] = {
                "n_train": len(train),
                "n_test": len(test),
                "train_failures": sum(1 for r in train if not r["expected_top1"]),
                "test_failures": sum(1 for r in test if not r["expected_top1"]),
                "protocol": best_train_test(train, test, include_reference=False),
                "near_readout_reference": best_train_test(train, test, include_reference=True),
            }

    by_family = {}
    for fam in families:
        subset = [r for r in rows if r["family"] == fam]
        by_family[fam] = {
            "protocol": rank_evals([eval_gate(subset, g) for g in proto_gates])[:10],
            "near_readout_reference": rank_evals([eval_gate(subset, g) for g in ref_gates])[:6],
        }

    return {
        "model": model,
        "baseline": baseline,
        "top_protocol_gates": overall_protocol,
        "top_near_readout_reference_gates": overall_reference,
        "by_family": by_family,
        "holdout": holdout,
    }


def write_cross_summary() -> dict:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase682_*_protocol_route_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 682,
        "title": "Protocol-Level Failure Monitor Localization Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase682_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 682 Protocol-Level Failure Monitor Localization",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | cases | top1_rate | failures | best protocol gate | score | capture | false_pos | holdout gate | holdout score | holdout capture | holdout false_pos | ref holdout score |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for item in models:
        model = item["model"]
        s = item["summary"]
        base = s["baseline"]["overall"]
        best = s["top_protocol_gates"][0] if s["top_protocol_gates"] else {}
        hold = s["holdout"].get("overall", {})
        hp = (hold.get("protocol") or {"gate": "NA", "test": {}})
        hr = (hold.get("near_readout_reference") or {"test": {}})
        lines.append(
            f"| {model} | {item['n_cases']} | {base['expected_top1_rate']:.3f} | "
            f"{int(base['failure_rate'] * item['n_cases'])} | {best.get('gate', 'NA')} | "
            f"{best.get('gate_score', 0.0):.3f} | {best.get('failure_capture_rate', 0.0):.3f} | "
            f"{best.get('success_false_positive_rate', 0.0):.3f} | {hp.get('gate', 'NA')} | "
            f"{hp.get('test', {}).get('gate_score', 0.0):.3f} | "
            f"{hp.get('test', {}).get('failure_capture_rate', 0.0):.3f} | "
            f"{hp.get('test', {}).get('success_false_positive_rate', 0.0):.3f} | "
            f"{hr.get('test', {}).get('gate_score', 0.0):.3f} |"
        )
    lines.extend(["", "## Family Baseline", ""])
    for item in models:
        lines.append(f"### {item['model']}")
        lines.append("")
        lines.append("| family | n | top1_rate | failure_rate | mean_rank | target_routes | failure_best_other_route |")
        lines.append("|---|---:|---:|---:|---:|---|---|")
        for fam, base in item["summary"]["baseline"].items():
            if fam == "overall":
                continue
            lines.append(
                f"| {fam} | {base['n']} | {base['expected_top1_rate']:.3f} | "
                f"{base['failure_rate']:.3f} | {base['mean_expected_rank']:.2f} | "
                f"{base['target_route']} | {base['failure_best_other_route']} |"
            )
        lines.append("")
    (OUT_ROOT / "phase682_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--family-limits", default=None)
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
