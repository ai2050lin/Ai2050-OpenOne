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
import phase870_blocker_field_admissibility_rule as p870  # noqa: E402


PHASE = 872
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase872_output_gate_readout_transition_audit")
DEFAULT_SOURCE_ROOT = Path("tests/result/phase871_field_admissibility_external_validation")


TARGET_ROLES = {"strict_target", "answer_class"}


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path) if path.exists() else []


def full_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("model")),
        str(row.get("domain")),
        str(row.get("case_id")),
        str(row.get("prompt_variant")),
        str(row.get("candidate_key")),
        str(row.get("edit_mode")),
    )


def context_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("model")),
        str(row.get("domain")),
        str(row.get("case_id")),
        str(row.get("prompt_variant")),
    )


def load_raw_rows(source_root: Path, source_round: str, file_prefix: str) -> tuple[dict[tuple[str, str, str, str, str, str], dict[str, Any]], dict[tuple[str, str, str, str], dict[str, Any]]]:
    by_full: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    by_context: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for model_name in MODELS:
        for row in read_jsonl(source_root / source_round / f"{file_prefix}_{model_name}_rows.jsonl"):
            by_full[full_key(row)] = row
            if row.get("condition_type") == "original":
                by_context[context_key(row)] = row
    return by_full, by_context


def best_non_target(top_tokens: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [tok for tok in top_tokens if str(tok.get("role")) not in TARGET_ROLES]
    if not candidates:
        return {"role": "none", "token": None, "logit": -999.0, "token_id": None}
    return max(candidates, key=lambda tok: finite(tok.get("logit"), -999.0))


def output_features(row: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    if row is None:
        return {
            f"{prefix}_missing": True,
            f"{prefix}_top1_role": "missing",
            f"{prefix}_top1_target": False,
            f"{prefix}_clear_margin_vs_non_target": -999.0,
            f"{prefix}_strict_margin_vs_non_target": -999.0,
        }
    top_tokens = row.get("top_tokens") or []
    top1 = top_tokens[0] if top_tokens else {}
    non_target = best_non_target(top_tokens)
    clear_logit = finite(row.get("clear_class_best_logit"), finite(row.get("class_best_logit")))
    strict_logit = finite(row.get("strict_best_logit"), clear_logit)
    top1_role = str(top1.get("role") or "none")
    non_target_logit = finite(non_target.get("logit"), -999.0)
    return {
        f"{prefix}_missing": False,
        f"{prefix}_rollout_label": row.get("rollout_label"),
        f"{prefix}_generated_clean": row.get("generated_clean"),
        f"{prefix}_first_token_strict": bool(row.get("first_token_strict")),
        f"{prefix}_first_token_answer_class": bool(row.get("first_token_answer_class")),
        f"{prefix}_first_token_clear_answer_class": bool(row.get("first_token_clear_answer_class")),
        f"{prefix}_rollout_strict_canonical": bool(row.get("rollout_strict_canonical")),
        f"{prefix}_rollout_clear_answer_class": bool(row.get("rollout_clear_answer_class")),
        f"{prefix}_top1_role": top1_role,
        f"{prefix}_top1_token": top1.get("token"),
        f"{prefix}_top1_logit": finite(top1.get("logit"), -999.0),
        f"{prefix}_top1_target": top1_role in TARGET_ROLES,
        f"{prefix}_best_non_target_role": str(non_target.get("role") or "none"),
        f"{prefix}_best_non_target_token": non_target.get("token"),
        f"{prefix}_best_non_target_logit": non_target_logit,
        f"{prefix}_clear_rank": finite(row.get("clear_class_best_rank"), 999.0),
        f"{prefix}_strict_rank": finite(row.get("strict_best_rank"), 999.0),
        f"{prefix}_clear_logit": clear_logit,
        f"{prefix}_strict_logit": strict_logit,
        f"{prefix}_clear_blocker_count": finite(row.get("clear_class_blocker_count"), 999.0),
        f"{prefix}_strict_blocker_count": finite(row.get("strict_blocker_count"), 999.0),
        f"{prefix}_class_minus_object_logit": finite(row.get("class_minus_object_logit")),
        f"{prefix}_clear_margin_vs_non_target": clear_logit - non_target_logit,
        f"{prefix}_strict_margin_vs_non_target": strict_logit - non_target_logit,
        f"{prefix}_blocker_answer_class_closure": bool(row.get("blocker_answer_class_closure")),
        f"{prefix}_blocker_strict_closure": bool(row.get("blocker_strict_closure")),
    }


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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    field_rows = [row for row in rows if row.get("field_strict_plus_effect_rule")]
    by_status = Counter(str(row.get("transfer_status")) for row in rows)
    by_field_status = Counter(str(row.get("transfer_status")) for row in field_rows)
    by_top1_role = Counter(str(row.get("intervened_top1_role")) for row in field_rows)
    by_non_target = Counter(str(row.get("intervened_best_non_target_role")) for row in field_rows)
    by_model_domain = defaultdict(Counter)
    for row in rows:
        key = f"{row.get('model')}:{row.get('domain')}"
        by_model_domain[key][str(row.get("transfer_status"))] += 1
    return {
        "n_rows": len(rows),
        "field_strict_plus_effect_rows": len(field_rows),
        "transfer_status_counts": dict(by_status),
        "field_strict_transfer_status_counts": dict(by_field_status),
        "field_strict_top1_role_counts": dict(by_top1_role),
        "field_strict_best_non_target_role_counts": dict(by_non_target),
        "model_domain_transfer_status_counts": {key: dict(value) for key, value in sorted(by_model_domain.items())},
    }


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_root = Path(args.source_root)
    pair_rows = p870.load_pair_rows(
        str(args.source_round),
        source_root,
        str(args.file_prefix),
        float(args.blocker_threshold),
        float(args.format_threshold),
        float(args.object_delta_threshold),
    )
    raw_by_full, raw_by_context = load_raw_rows(source_root, str(args.source_round), str(args.file_prefix))
    out: list[dict[str, Any]] = []
    for pair in pair_rows:
        raw = raw_by_full.get(
            (
                str(pair.get("model")),
                str(pair.get("domain")),
                str(pair.get("case_id")),
                str(pair.get("prompt_variant")),
                str(pair.get("candidate_key")),
                str(pair.get("edit_mode")),
            )
        )
        base = raw_by_context.get(
            (
                str(pair.get("model")),
                str(pair.get("domain")),
                str(pair.get("case_id")),
                str(pair.get("prompt_variant")),
            )
        )
        row = dict(pair)
        row.update(output_features(base, "base"))
        row.update(output_features(raw, "intervened"))
        row["readout_rank_closure_rule"] = bool(
            row.get("field_strict_plus_effect_rule")
            and finite(row.get("intervened_clear_rank"), 999.0) <= float(args.rank_threshold)
            and finite(row.get("intervened_clear_blocker_count"), 999.0) <= 0.0
        )
        row["output_gate_raw_rule"] = bool(
            row.get("intervened_top1_target")
            and finite(row.get("intervened_clear_margin_vs_non_target"), -999.0) > float(args.margin_threshold)
        )
        row["output_gate_field_rule"] = bool(row.get("field_base_admissible") and row.get("output_gate_raw_rule"))
        row["output_gate_top1_rule"] = bool(row.get("field_strict_plus_effect_rule") and row.get("intervened_top1_target"))
        row["output_gate_margin_rule"] = bool(
            row.get("field_strict_plus_effect_rule")
            and row.get("intervened_top1_target")
            and finite(row.get("intervened_clear_margin_vs_non_target"), -999.0) > float(args.margin_threshold)
        )
        row["output_gate_strict_margin_rule"] = bool(
            row.get("field_strict_plus_effect_rule")
            and row.get("intervened_top1_target")
            and finite(row.get("intervened_strict_margin_vs_non_target"), -999.0) > float(args.margin_threshold)
        )
        row["gate_failure_reason"] = "none"
        if row.get("field_strict_plus_effect_rule") and not row.get("output_gate_margin_rule"):
            if not row.get("intervened_top1_target"):
                row["gate_failure_reason"] = f"top1_non_target:{row.get('intervened_top1_role')}"
            elif finite(row.get("intervened_clear_margin_vs_non_target"), -999.0) <= float(args.margin_threshold):
                row["gate_failure_reason"] = "target_non_target_margin_not_positive"
            else:
                row["gate_failure_reason"] = "unknown"
        out.append(row)
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 872 Output Gate / Readout Transition Audit ({payload['source_round']})",
        "",
        "- Boundary: offline audit from existing row-level logits and rollouts; no new model run.",
        "- Goal: test whether output/readout dominance explains failures left by FieldAdmissible + GearEffect.",
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
        f"- Transfer status counts: `{payload['summary']['transfer_status_counts']}`",
        f"- Field-strict transfer status counts: `{payload['summary']['field_strict_transfer_status_counts']}`",
        f"- Field-strict top1 role counts: `{payload['summary']['field_strict_top1_role_counts']}`",
        f"- Field-strict best non-target role counts: `{payload['summary']['field_strict_best_non_target_role_counts']}`",
        "",
        "## Gate-Candidate Rows",
        "",
        "| model | domain | object | prompt | mode | target | field+effect | gate | top1 | margin | non-target | failure | rollout |",
        "|---|---|---|---|---|---|---|---|---|---:|---|---|---|",
    ]
    for row in payload.get("rows") or []:
        if not (row.get("field_strict_plus_effect_rule") or row.get("target_clean_transition")):
            continue
        lines.append(
            f"| {row.get('model')} | {row.get('domain')} | {row.get('object')} | `{row.get('prompt_variant')}` | `{row.get('edit_mode')}` | "
            f"{row.get('target_clean_transition')} | {row.get('field_strict_plus_effect_rule')} | {row.get('output_gate_margin_rule')} | "
            f"`{row.get('intervened_top1_role')}:{row.get('intervened_top1_token')}` | "
            f"{finite(row.get('intervened_clear_margin_vs_non_target')):.3f} | "
            f"`{row.get('intervened_best_non_target_role')}:{row.get('intervened_best_non_target_token')}` | "
            f"`{row.get('gate_failure_reason')}` | `{row.get('base_rollout_label')} -> {row.get('intervened_rollout_label')}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-round", default="validation")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--file-prefix", default="phase871")
    parser.add_argument("--output-round")
    parser.add_argument("--blocker-threshold", type=float, default=20.0)
    parser.add_argument("--format-threshold", type=float, default=3.0)
    parser.add_argument("--object-delta-threshold", type=float, default=0.25)
    parser.add_argument("--rank-threshold", type=float, default=1.0)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    args = parser.parse_args()

    rows = build_rows(args)
    rule_results = []
    for rule in (
        "output_gate_raw_rule",
        "output_gate_field_rule",
        "field_strict_plus_effect_rule",
        "readout_rank_closure_rule",
        "output_gate_top1_rule",
        "output_gate_margin_rule",
        "output_gate_strict_margin_rule",
    ):
        for target in (
            "intervened_rollout_clear_answer_class",
            "intervened_rollout_strict_canonical",
            "target_output_clean_transition",
            "target_clean_transition",
        ):
            rule_results.append({"rule": rule, "target": target, **binary_stats(rows, rule, target)})
    payload = {
        "phase": PHASE,
        "title": "Output Gate / Readout Transition Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_round": args.source_round,
        "source": str(Path(args.source_root) / args.source_round),
        "file_prefix": str(args.file_prefix),
        "rank_threshold": float(args.rank_threshold),
        "margin_threshold": float(args.margin_threshold),
        "summary": summarize(rows),
        "rule_results": rule_results,
        "rows": rows,
        "boundary": "Offline readout gate audit; it is a diagnostic equation, not closure.",
    }
    out_dir = RESULT_ROOT / (args.output_round or args.source_round)
    p846.write_json(out_dir / "phase872_summary.json", payload)
    p846.write_jsonl(out_dir / "phase872_output_gate_rows.jsonl", rows)
    write_markdown(out_dir / "phase872_summary.md", payload)
    print(json.dumps({"summary": payload["summary"], "rule_results": rule_results}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
