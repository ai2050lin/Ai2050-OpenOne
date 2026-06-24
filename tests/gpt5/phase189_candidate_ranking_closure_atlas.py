#!/usr/bin/env python3
"""Phase189: Candidate-ranking closure atlas update.

This is an atlas analysis/update phase.  It consumes existing Phase591-597
results and creates a focused candidate-specific ranking evidence table.  It
does not run CUDA models.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODELS = ("qwen3", "glm4", "deepseek7b")


@dataclass(frozen=True)
class SourceSpec:
    phase: int
    root: Path
    pattern: str
    evidence_family: str


SOURCES = (
    SourceSpec(591, Path("results/glm5_phase591_value_candidate_ranking_audit"), "phase591_{model}_value_candidate_ranking_audit_confirm.json", "prompt_vs_hidden_ranking"),
    SourceSpec(592, Path("results/glm5_phase592_relation_specific_ranking_atlas"), "phase592_{model}_relation_specific_ranking_atlas_confirm.json", "projection_atlas"),
    SourceSpec(593, Path("results/glm5_phase593_atlas_guided_causal_patch"), "phase593_{model}_atlas_guided_causal_patch_confirm.json", "atlas_guided_patch"),
    SourceSpec(594, Path("results/glm5_phase594_conditional_transformation_atlas"), "phase594_{model}_conditional_transformation_atlas_confirm.json", "transition_atlas"),
    SourceSpec(595, Path("results/glm5_phase595_mlp_update_causal_validation"), "phase595_{model}_mlp_update_causal_validation_confirm.json", "mlp_output_patch"),
    SourceSpec(596, Path("results/glm5_phase596_mlp_internal_gate_path_audit"), "phase596_{model}_mlp_internal_gate_path_audit_confirm.json", "mlp_internal_patch"),
    SourceSpec(597, Path("results/glm5_phase597_state_conditioned_mlp_generation_audit"), "phase597_{model}_state_conditioned_mlp_generation_audit_confirm.json", "state_conditioned_mlp"),
)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def as_float(x: Any, default: float = 0.0) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    return default


def as_int(x: Any, default: int = 0) -> int:
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    return default


def normalized_key(row: dict[str, Any]) -> str:
    if row.get("key"):
        return str(row["key"])
    pos = row.get("position", "unknown_pos")
    layer = row.get("layer", row.get("best_layer", "unknown_layer"))
    source = row.get("source", row.get("component", row.get("mode", row.get("bucket", "unknown_source"))))
    return f"{pos}|L{layer}|{source}"


def classify_projection(row: dict[str, Any], phase: int) -> tuple[int, str]:
    spec_margin = as_float(
        row.get("mean_specific_margin", row.get("projection_specific_margin", row.get("mean_projection_specific_margin")))
    )
    positive_rate = as_float(row.get("positive_specific_rate", row.get("positive_rate", row.get("positive_projection_margin_rate"))))
    if spec_margin >= 0.5 and positive_rate >= 0.5:
        return 2, "strong_projection_not_causal"
    if spec_margin >= 0.1:
        return 2, "weak_projection_not_causal"
    return 1, "low_projection"


def classify_transition(row: dict[str, Any]) -> tuple[int, str]:
    spec_margin = as_float(row.get("mean_specific_margin", row.get("mean_projection_specific_margin")))
    positive_rate = as_float(row.get("positive_rate", row.get("positive_projection_margin_rate")))
    source = str(row.get("source", ""))
    if "mlp_update" in source and spec_margin >= 0.5:
        return 3, "strong_mlp_transition"
    if spec_margin >= 0.25:
        return 3, "transition_candidate"
    return 2, "projection_or_weak_transition"


def classify_patch(row: dict[str, Any]) -> tuple[int, str]:
    n = as_int(row.get("n"))
    switch = as_int(row.get("switch"))
    margin = as_float(row.get("mean_margin_gain", row.get("margin_gain")))
    spec = as_float(row.get("mean_specific_margin_gain", row.get("specific_margin_gain")))
    common = as_float(row.get("mean_common_delta", row.get("common_delta")))
    if n > 0 and switch > 0 and margin > 0.05 and spec > 0.05:
        # Weak component contribution, not necessarily specific enough.
        return 4, "weak_component_margin_or_switch"
    if margin > 0.05 and abs(common) < max(0.25, abs(spec) * 4.0):
        return 4, "weak_component_margin_no_switch"
    if abs(common) > 1.0 and abs(spec) < 0.1:
        return 3, "common_activation_only"
    if margin <= 0.01 and spec <= 0.01:
        return 2, "patch_no_specific_effect"
    return 3, "patch_inconclusive"


def classify_row(row: dict[str, Any], phase: int, family: str) -> tuple[int, str]:
    if phase == 592:
        return classify_projection(row, phase)
    if phase == 594:
        return classify_transition(row)
    if phase in (593, 595, 596, 597):
        return classify_patch(row)
    if phase == 591:
        key = str(row.get("key", ""))
        if "repair_prompt" == key:
            return 6, "prompt_level_specific_repair"
        return classify_patch(row)
    return 1, "unclassified"


def rows_from_summary(data: dict[str, Any], phase: int) -> list[dict[str, Any]]:
    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        return []
    rows: list[dict[str, Any]] = []

    # Most phases expose summary.best.
    best = summary.get("best")
    if isinstance(best, list):
        rows.extend([r for r in best if isinstance(r, dict)])

    # Phase591/595/596/597 have by_key dictionaries with richer entries.
    by_key = summary.get("by_key")
    if isinstance(by_key, dict):
        for key, value in by_key.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("key", key)
                rows.append(row)

    # Phase597 also has generated projection entries.
    generated = summary.get("generated_by_key")
    if isinstance(generated, dict):
        for key, value in generated.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("key", key)
                row.setdefault("source", "generated_down")
                rows.append(row)
    return rows


def build_closure(out_dir: Path) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    seen_evidence: set[str] = set()
    by_model_phase: Counter[tuple[str, int]] = Counter()
    closure_counts: Counter[str] = Counter()
    level_counts: Counter[int] = Counter()
    gap_counts: Counter[str] = Counter()

    for spec in SOURCES:
        for model in MODELS:
            path = spec.root / spec.pattern.format(model=model)
            data = load_json(path)
            if data is None:
                continue
            by_model_phase[(model, spec.phase)] += 1
            n_cases = data.get("n_cases")
            target_n = data.get("target_n", data.get("n_target_rows", data.get("n_target_cases_seen")))
            for row in rows_from_summary(data, spec.phase):
                level, closure_status = classify_row(row, spec.phase, spec.evidence_family)
                key = normalized_key(row)
                evidence_id = f"phase{spec.phase}:{model}:{spec.evidence_family}:{key}"
                if evidence_id in seen_evidence:
                    continue
                seen_evidence.add(evidence_id)
                gap = infer_gap(closure_status, row)
                evidence.append(
                    {
                        "evidence_id": evidence_id,
                        "phase": spec.phase,
                        "model": model,
                        "evidence_family": spec.evidence_family,
                        "key": key,
                        "position": row.get("position"),
                        "layer": row.get("layer"),
                        "component": row.get("component", row.get("source", row.get("mode"))),
                        "n": row.get("n", target_n),
                        "n_cases": n_cases,
                        "target_n": target_n,
                        "causal_level": level,
                        "closure_status": closure_status,
                        "gap_type": gap,
                        "metrics": compact_metrics(row),
                        "source_path": str(path),
                    }
                )
                closure_counts[closure_status] += 1
                level_counts[level] += 1
                gap_counts[gap] += 1

    evidence.sort(key=lambda r: (r["phase"], r["model"], r["evidence_family"], r["key"]))
    write_jsonl(out_dir / "candidate_ranking_evidence.jsonl", evidence)

    atlas_edges = []
    for row in evidence:
        atlas_edges.append(
            {
                "source": f"phase:{row['phase']}",
                "target": f"candidate_ranking_evidence:{row['evidence_id']}",
                "edge_type": "updates_candidate_ranking_evidence",
                "model": row["model"],
                "causal_level": row["causal_level"],
                "closure_status": row["closure_status"],
                "gap_type": row["gap_type"],
            }
        )
        atlas_edges.append(
            {
                "source": f"candidate_ranking_evidence:{row['evidence_id']}",
                "target": f"gap:{row['gap_type']}",
                "edge_type": "supports_gap_diagnosis",
                "model": row["model"],
                "phase": row["phase"],
            }
        )
    write_jsonl(out_dir / "atlas_candidate_ranking_edges.jsonl", atlas_edges)

    report = write_report(out_dir, evidence, closure_counts, level_counts, gap_counts)
    index = {
        "phase": 189,
        "name": "Candidate Ranking Closure Atlas Update",
        "counts": {
            "evidence_rows": len(evidence),
            "atlas_edges": len(atlas_edges),
            "closure_statuses": dict(closure_counts),
            "causal_levels": {str(k): v for k, v in sorted(level_counts.items())},
            "gap_types": dict(gap_counts),
        },
        "files": {
            "evidence": "candidate_ranking_evidence.jsonl",
            "edges": "atlas_candidate_ranking_edges.jsonl",
            "report": "candidate_ranking_closure_report.md",
        },
        "main_report": report,
    }
    (out_dir / "candidate_ranking_closure_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index


def compact_metrics(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "switch",
        "switch_rate",
        "mean_margin_gain",
        "margin_gain",
        "mean_specific_margin_gain",
        "specific_margin_gain",
        "mean_common_delta",
        "common_delta",
        "mean_correct_specific",
        "correct_specific",
        "mean_old_top_wrong_specific",
        "old_top_wrong_specific",
        "mean_specific_margin",
        "projection_specific_margin",
        "mean_projection_specific_margin",
        "positive_specific_rate",
        "positive_rate",
        "positive_margin_rate",
        "mean_projection_correct_specific",
        "mean_projection_old_top_wrong_specific",
    ]
    return {k: row[k] for k in keys if k in row}


def infer_gap(status: str, row: dict[str, Any]) -> str:
    if "common_activation" in status:
        return "candidate_common_only"
    if "projection" in status and "causal" in status:
        return "projection_not_causal"
    if "patch_no_specific_effect" in status:
        return "component_patch_no_specific_ranking"
    if "weak_component" in status:
        return "weak_component_candidate"
    if "prompt_level_specific_repair" in status:
        return "prompt_has_missing_state"
    if "transition" in status:
        return "transition_not_component_closure"
    return "ranking_unresolved"


def write_report(
    out_dir: Path,
    evidence: list[dict[str, Any]],
    closure_counts: Counter[str],
    level_counts: Counter[int],
    gap_counts: Counter[str],
) -> str:
    lines = [
        "# Phase189 Candidate Ranking Closure Atlas Report",
        "",
        "This report consumes existing Phase591-597 results. It does not rerun CUDA models.",
        "",
        "## Evidence Counts",
        "",
        f"- evidence rows: {len(evidence)}",
        "",
        "## Causal Levels",
        "",
        "| level | count |",
        "|---:|---:|",
    ]
    for level, count in sorted(level_counts.items()):
        lines.append(f"| {level} | {count} |")

    lines.extend(["", "## Closure Statuses", "", "| status | count |", "|---|---:|"])
    for status, count in closure_counts.most_common():
        lines.append(f"| {status} | {count} |")

    lines.extend(["", "## Gap Types", "", "| gap | count |", "|---|---:|"])
    for gap, count in gap_counts.most_common():
        lines.append(f"| {gap} | {count} |")

    lines.extend(["", "## Best Level 4 Component Candidates", ""])
    level4 = [r for r in evidence if r["causal_level"] >= 4 and r["phase"] != 591]
    level4.sort(key=lambda r: metric_score(r), reverse=True)
    if not level4:
        lines.append("- none")
    else:
        lines.append("| phase | model | key | status | score | source |")
        lines.append("|---:|---|---|---|---:|---|")
        for row in level4[:20]:
            lines.append(
                f"| {row['phase']} | {row['model']} | `{row['key']}` | {row['closure_status']} | {metric_score(row):.3f} | {row['evidence_family']} |"
            )

    lines.extend(["", "## DS7B Critical Path", ""])
    ds7b = [r for r in evidence if r["model"] == "deepseek7b" and any(t in r["key"] for t in ("rule_value|L26", "query_relation|L19"))]
    ds7b.sort(key=lambda r: (r["phase"], r["key"]))
    if not ds7b:
        lines.append("- no DS7B critical-path rows found")
    else:
        lines.append("| phase | key | level | status | gap | metrics |")
        lines.append("|---:|---|---:|---|---|---|")
        for row in ds7b[:40]:
            lines.append(
                f"| {row['phase']} | `{row['key']}` | {row['causal_level']} | {row['closure_status']} | {row['gap_type']} | `{json.dumps(row['metrics'], ensure_ascii=False)[:180]}` |"
            )

    lines.extend(
        [
            "",
            "## Main Conclusion",
            "",
            "- Prompt-level repair can produce strong candidate-specific ranking, but static hidden/component patches mostly fail to transfer it.",
            "- DS7B rule_value L26 and query_relation L19 remain the best mechanistic handles, but current evidence is still not Level 5 repair.",
            "- The next useful experiment should isolate component/channel state conditions rather than adding another broad residual or MLP-output delta.",
            "",
            "## Recommended Next Move",
            "",
            "Run a small but targeted channel-state intervention on DS7B first, then only scale to Qwen3/GLM4 if DS7B shows a Level4 margin effect above controls.",
        ]
    )
    path = out_dir / "candidate_ranking_closure_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def metric_score(row: dict[str, Any]) -> float:
    m = row.get("metrics", {})
    if not isinstance(m, dict):
        return 0.0
    vals = [
        as_float(m.get("mean_margin_gain")),
        as_float(m.get("margin_gain")),
        as_float(m.get("mean_specific_margin_gain")),
        as_float(m.get("specific_margin_gain")),
        as_float(m.get("mean_specific_margin")),
        as_float(m.get("projection_specific_margin")),
        as_float(m.get("mean_projection_specific_margin")),
    ]
    return max(vals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/gpt5_phase189_candidate_ranking_closure_atlas")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = build_closure(out_dir)
    print(json.dumps(index["counts"], ensure_ascii=False, sort_keys=True))
    print(f"[ok] wrote {out_dir}")


if __name__ == "__main__":
    main()
