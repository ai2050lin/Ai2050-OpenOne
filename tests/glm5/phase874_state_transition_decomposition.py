#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402


PHASE = 874
RESULT_ROOT = Path("tests/result/phase874_state_transition_decomposition")
DEFAULT_PHASE872_ROOT = Path("tests/result/phase872_output_gate_readout_transition_audit")


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path) if path.exists() else []


def binary_stats(rows: list[dict[str, Any]], pred_key: str, target_key: str) -> dict[str, Any]:
    tp = sum(1 for row in rows if row.get(pred_key) and row.get(target_key))
    fp = sum(1 for row in rows if row.get(pred_key) and not row.get(target_key))
    fn = sum(1 for row in rows if not row.get(pred_key) and row.get(target_key))
    tn = sum(1 for row in rows if not row.get(pred_key) and not row.get(target_key))
    n = len(rows)
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "accuracy": (tp + tn) / n if n else 0.0,
    }


def load_rows(round_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_dir in round_dirs:
        path = round_dir / "phase872_output_gate_rows.jsonl"
        for row in read_jsonl(path):
            copied = dict(row)
            copied["phase872_round"] = round_dir.name
            rows.append(copied)
    return rows


def parse_round_dirs(text: str, root: Path) -> list[Path]:
    if text:
        return [Path(part.strip()) for part in text.split(",") if part.strip()]
    preferred = ["holdout_phase867", "validation_phase871", "replication_phase873"]
    return [root / name for name in preferred if (root / name / "phase872_output_gate_rows.jsonl").exists()]


def transition_failure_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not row.get("field_base_admissible"):
        reasons.append("field_not_base_admissible")
    if not row.get("field_strict_admissible"):
        reasons.append("field_not_strict_admissible")
    if not row.get("phase866_pair_rule"):
        reasons.append("not_phase866_pair_rule")
    if finite(row.get("original_blocker_delta")) >= 0:
        reasons.append("original_blocker_not_reduced")
    if finite(row.get("blocker_reduction")) <= 0:
        reasons.append("blocker_not_reduced")
    if bool(row.get("object_echo_induced")):
        reasons.append("object_echo_induced")
    if bool(row.get("format_or_other_induced")):
        reasons.append("format_or_other_induced")
    for tag in row.get("field_tags") or []:
        if tag != "field_low_pressure":
            reasons.append(f"field_tag:{tag}")
    return reasons or ["clean_route_conditions_met"]


def enrich_row(row: dict[str, Any], margin_threshold: float) -> dict[str, Any]:
    out = dict(row)
    out["base_output_state_open"] = bool(
        out.get("base_top1_target") and finite(out.get("base_clear_margin_vs_non_target"), -999.0) > margin_threshold
    )
    out["intervened_output_state_open"] = bool(
        out.get("intervened_top1_target")
        and finite(out.get("intervened_clear_margin_vs_non_target"), -999.0) > margin_threshold
    )
    out["base_strict_state_open"] = bool(
        out.get("base_top1_role") == "strict_target"
        and finite(out.get("base_strict_margin_vs_non_target"), -999.0) > margin_threshold
    )
    out["intervened_strict_state_open"] = bool(
        out.get("intervened_top1_role") == "strict_target"
        and finite(out.get("intervened_strict_margin_vs_non_target"), -999.0) > margin_threshold
    )
    out["observed_clear_transition_rule"] = bool(
        (not out.get("base_rollout_clear_answer_class")) and out.get("intervened_output_state_open")
    )
    out["observed_strict_transition_rule"] = bool(
        (not out.get("base_rollout_strict_canonical")) and out.get("intervened_strict_state_open")
    )
    out["latent_output_gate_transition_rule"] = bool(
        (not out.get("base_output_state_open")) and out.get("intervened_output_state_open")
    )
    out["latent_strict_gate_transition_rule"] = bool(
        (not out.get("base_strict_state_open")) and out.get("intervened_strict_state_open")
    )
    out["clean_causal_edge_rule"] = bool(
        out.get("field_strict_plus_effect_rule")
        and out.get("observed_clear_transition_rule")
        and not out.get("object_echo_induced")
        and not out.get("format_or_other_induced")
    )
    if out.get("target_clean_transition"):
        transition_class = "clean_causal_transition"
    elif out.get("target_output_clean_transition"):
        transition_class = "nonclean_output_transition"
    elif out.get("base_rollout_clear_answer_class") and not out.get("intervened_rollout_clear_answer_class"):
        transition_class = "answer_class_loss"
    elif out.get("base_rollout_clear_answer_class") and out.get("intervened_rollout_clear_answer_class"):
        transition_class = "answer_class_stable_open"
    elif (not out.get("base_rollout_clear_answer_class")) and (not out.get("intervened_rollout_clear_answer_class")):
        transition_class = "answer_class_stable_closed"
    else:
        transition_class = "other_transition"
    out["transition_class"] = transition_class
    out["nonclean_transition_reasons"] = (
        transition_failure_reasons(out)
        if out.get("target_output_clean_transition") and not out.get("target_clean_transition")
        else []
    )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_round = defaultdict(Counter)
    by_model_domain = defaultdict(Counter)
    transition_classes = Counter(str(row.get("transition_class")) for row in rows)
    reasons = Counter(reason for row in rows for reason in row.get("nonclean_transition_reasons") or [])
    source_labels = Counter(
        (str(row.get("base_rollout_label")), str(row.get("intervened_rollout_label")))
        for row in rows
        if row.get("target_output_clean_transition") or row.get("target_clean_transition")
    )
    for row in rows:
        by_round[str(row.get("phase872_round"))][str(row.get("transition_class"))] += 1
        by_model_domain[f"{row.get('model')}:{row.get('domain')}"][str(row.get("transition_class"))] += 1
    return {
        "n_rows": len(rows),
        "transition_class_counts": dict(transition_classes),
        "nonclean_transition_reason_counts": dict(reasons),
        "observed_transition_label_counts": {f"{a}->{b}": count for (a, b), count in source_labels.items()},
        "round_transition_class_counts": {key: dict(value) for key, value in sorted(by_round.items())},
        "model_domain_transition_class_counts": {key: dict(value) for key, value in sorted(by_model_domain.items())},
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 874 State Transition Decomposition",
        "",
        "- Boundary: offline decomposition from Phase 872 rows; no new model run.",
        "- Goal: separate output state, observed state transition, and clean causal edge.",
        "",
        "## Rule Results",
        "",
        "| rule | target | n | TP | FP | FN | TN | precision | recall | accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("rule_results") or []:
        lines.append(
            f"| `{row['rule']}` | `{row['target']}` | {row['n']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['accuracy']:.3f} |"
        )
    lines += [
        "",
        "## Summary",
        "",
        f"- Transition class counts: `{payload['summary']['transition_class_counts']}`",
        f"- Nonclean transition reasons: `{payload['summary']['nonclean_transition_reason_counts']}`",
        f"- Observed transition labels: `{payload['summary']['observed_transition_label_counts']}`",
        "",
        "## Output Transitions",
        "",
        "| round | model | domain | object | prompt | mode | class | labels | clear-rule | strict-rule | clean-edge | field tags | reasons |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload.get("rows") or []:
        if not (
            row.get("target_output_clean_transition")
            or row.get("target_clean_transition")
            or row.get("observed_clear_transition_rule")
            or row.get("latent_output_gate_transition_rule")
        ):
            continue
        lines.append(
            f"| `{row.get('phase872_round')}` | {row.get('model')} | {row.get('domain')} | {row.get('object')} | "
            f"`{row.get('prompt_variant')}` | `{row.get('edit_mode')}` | `{row.get('transition_class')}` | "
            f"`{row.get('base_rollout_label')}->{row.get('intervened_rollout_label')}` | "
            f"{row.get('observed_clear_transition_rule')} | {row.get('observed_strict_transition_rule')} | "
            f"{row.get('clean_causal_edge_rule')} | `{row.get('field_tags')}` | `{row.get('nonclean_transition_reasons')}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase872-root", default=str(DEFAULT_PHASE872_ROOT))
    parser.add_argument("--round-dirs", default="")
    parser.add_argument("--output-round", default="combined")
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    args = parser.parse_args()

    round_dirs = parse_round_dirs(str(args.round_dirs), Path(args.phase872_root))
    rows = [enrich_row(row, float(args.margin_threshold)) for row in load_rows(round_dirs)]
    rule_results = []
    for rule in (
        "intervened_output_state_open",
        "latent_output_gate_transition_rule",
        "observed_clear_transition_rule",
        "observed_strict_transition_rule",
        "clean_causal_edge_rule",
    ):
        for target in (
            "intervened_rollout_clear_answer_class",
            "target_output_clean_transition",
            "target_clean_transition",
        ):
            rule_results.append({"rule": rule, "target": target, **binary_stats(rows, rule, target)})
    payload = {
        "phase": PHASE,
        "title": "State Transition Decomposition",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase872_round_dirs": [str(path) for path in round_dirs],
        "margin_threshold": float(args.margin_threshold),
        "summary": summarize(rows),
        "rule_results": rule_results,
        "rows": rows,
        "boundary": "Offline state transition decomposition; no model run and no closure claim.",
    }
    out_dir = RESULT_ROOT / str(args.output_round)
    p846.write_json(out_dir / "phase874_summary.json", payload)
    p846.write_jsonl(out_dir / "phase874_transition_rows.jsonl", rows)
    write_markdown(out_dir / "phase874_summary.md", payload)
    print(json.dumps({"summary": payload["summary"], "rule_results": rule_results}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
