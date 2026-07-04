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


PHASE = 889
MODELS = ["qwen3", "glm4", "deepseek7b"]
PHASE888_ROOT = Path("tests/result/phase888_direction_set_internal_subspace_probe")
RESULT_ROOT = Path("tests/result/phase889_restore_failure_direction_profile")


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return p846.mean(values)


def counter_values(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_rows(round_name: str, model: str) -> list[dict[str, Any]]:
    path = PHASE888_ROOT / round_name / f"phase888_{model}_rows.jsonl"
    return p846.read_jsonl(path) if path.exists() else []


def mechanism_label(row: dict[str, Any]) -> str:
    if not row.get("mode_closure_from_open"):
        return "no_internal_closure"
    if row.get("restored_reopens_boundary"):
        return "cut_token_coupled_internal_signal"
    if finite(row.get("mode_class_logit_delta")) > 0 and row.get("base_mask_boundary_closed"):
        return "target_lift_or_distributed_boundary"
    if finite(row.get("cut_token_delta_mean")) < 0:
        return "cut_suppression_not_necessary"
    return "distributed_boundary_no_restore"


def analyze_rows(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        row["phase889_mechanism_label"] = mechanism_label(row)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("parent_candidate_key"))].append(row)
    candidate_groups = []
    for key, vals in groups.items():
        closure_rows = [row for row in vals if row.get("mode_closure_from_open")]
        restore_rows = [row for row in vals if row.get("restored_reopens_boundary")]
        no_restore_closure = [row for row in closure_rows if not row.get("restored_reopens_boundary")]
        label_counts = Counter(str(row.get("phase889_mechanism_label")) for row in vals)
        closure_modes = Counter(str(row.get("edit_mode")) for row in closure_rows)
        if restore_rows:
            evidence = "internal_cut_token_coupling"
        elif closure_rows and label_counts.get("target_lift_or_distributed_boundary", 0) >= max(1, len(closure_rows) // 2):
            evidence = "distributed_target_lift_direction_set"
        elif closure_rows:
            evidence = "direction_set_closure_without_cut_necessity"
        else:
            evidence = "negative_no_direction_profile"
        candidate_groups.append(
            {
                "model": model,
                "parent_candidate_key": key,
                "evidence_label": evidence,
                "n_rows": len(vals),
                "n_source_cases": len(set((str(row.get("case_id")), str(row.get("prompt_variant"))) for row in vals)),
                "mode_closure_from_open": len(closure_rows),
                "restored_reopens_boundary": len(restore_rows),
                "no_restore_closure": len(no_restore_closure),
                "base_mask_boundary_closed": sum(1 for row in vals if row.get("base_mask_boundary_closed")),
                "mechanism_label_counts": counter_values(label_counts),
                "closure_modes": counter_values(closure_modes),
                "mean_closure_class_logit_delta": mean([finite(row.get("mode_class_logit_delta")) for row in closure_rows]) or 0.0,
                "mean_closure_cut_token_delta": mean([finite(row.get("cut_token_delta_mean")) for row in closure_rows if row.get("cut_token_delta_mean") is not None]) or 0.0,
                "mean_no_restore_class_logit_delta": mean([finite(row.get("mode_class_logit_delta")) for row in no_restore_closure]) or 0.0,
                "mean_no_restore_cut_token_delta": mean([finite(row.get("cut_token_delta_mean")) for row in no_restore_closure if row.get("cut_token_delta_mean") is not None]) or 0.0,
                "objects": sorted(set(str(row.get("object")) for row in vals)),
                "prompt_variants": sorted(set(str(row.get("prompt_variant")) for row in vals)),
            }
        )
    candidate_groups.sort(
        key=lambda row: (
            row.get("restored_reopens_boundary") or 0,
            row.get("mode_closure_from_open") or 0,
            row.get("base_mask_boundary_closed") or 0,
        ),
        reverse=True,
    )
    return {
        "phase": PHASE,
        "model": model,
        "source_phase": 888,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": len(rows),
        "overall": {
            "mode_closure_from_open": sum(1 for row in rows if row.get("mode_closure_from_open")),
            "restored_reopens_boundary": sum(1 for row in rows if row.get("restored_reopens_boundary")),
            "no_restore_closure": sum(1 for row in rows if row.get("mode_closure_from_open") and not row.get("restored_reopens_boundary")),
            "base_mask_boundary_closed": sum(1 for row in rows if row.get("base_mask_boundary_closed")),
            "mechanism_label_counts": counter_values(Counter(str(row.get("phase889_mechanism_label")) for row in rows)),
            "closure_modes": counter_values(Counter(str(row.get("edit_mode")) for row in rows if row.get("mode_closure_from_open"))),
        },
        "candidate_groups": candidate_groups,
        "evidence_label_counts": counter_values(Counter(str(row.get("evidence_label")) for row in candidate_groups)),
    }


def write_model(round_name: str, model: str) -> dict[str, Any]:
    rows = load_rows(round_name, model)
    payload = analyze_rows(model, rows)
    out_dir = RESULT_ROOT / round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    p846.write_jsonl(out_dir / f"phase889_{model}_rows.jsonl", rows)
    p846.write_json(out_dir / f"phase889_{model}_summary.json", payload)
    print(json.dumps({"phase": PHASE, "model": model, "overall": payload["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 889 restore failure and direction profile",
        "",
        "## Overall",
        "",
        f"- source_rows: {payload.get('source_rows')}",
        f"- mode_closure_from_open: {payload.get('overall', {}).get('mode_closure_from_open')}",
        f"- restored_reopens_boundary: {payload.get('overall', {}).get('restored_reopens_boundary')}",
        f"- no_restore_closure: {payload.get('overall', {}).get('no_restore_closure')}",
        "",
        "## Candidate groups",
        "",
        "| model | candidate | label | closures | restore | no restore | mean class delta | mean cut delta | modes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload.get("candidate_groups", [])[:30]:
        lines.append(
            "| {model} | {key} | {label} | {closure} | {restore} | {no_restore} | {class_delta:.3f} | {cut_delta:.3f} | {modes} |".format(
                model=row.get("model"),
                key=row.get("parent_candidate_key"),
                label=row.get("evidence_label"),
                closure=row.get("mode_closure_from_open"),
                restore=row.get("restored_reopens_boundary"),
                no_restore=row.get("no_restore_closure"),
                class_delta=finite(row.get("mean_closure_class_logit_delta")),
                cut_delta=finite(row.get("mean_closure_cut_token_delta")),
                modes=json.dumps(row.get("closure_modes") or {}, ensure_ascii=False),
            )
        )
    return "\n".join(lines) + "\n"


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase889_{model}_summary.json") for model in MODELS]
    summaries = [item for item in summaries if item and item.get("status") == "complete"]
    groups: list[dict[str, Any]] = []
    overall = Counter()
    source_rows = 0
    for summary in summaries:
        source_rows += int(summary.get("n_rows") or 0)
        groups.extend(summary.get("candidate_groups") or [])
        ov = summary.get("overall") or {}
        for key, value in ov.items():
            if isinstance(value, int):
                overall[key] += value
    groups.sort(
        key=lambda row: (
            row.get("restored_reopens_boundary") or 0,
            row.get("mode_closure_from_open") or 0,
        ),
        reverse=True,
    )
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "source_rows": source_rows,
        "overall": {key: int(value) for key, value in sorted(overall.items())},
        "candidate_groups": groups,
        "evidence_label_counts": counter_values(Counter(str(row.get("evidence_label")) for row in groups)),
    }
    p846.write_json(out_dir / "phase889_cross_model_summary.json", payload)
    (out_dir / "phase889_cross_model_summary.md").write_text(markdown_summary(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="direction_set_probe")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "overall": payload["overall"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    write_model(args.round_name, args.model)


if __name__ == "__main__":
    main()
