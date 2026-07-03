#!/usr/bin/env python3
from __future__ import annotations

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


PHASE = 883
RESULT_ROOT = Path("tests/result/phase883_atlas_score_table_and_evidence_calibration")
ROUND_NAME = "phase875_882_calibration"

PHASE_CONTEXT_FILES = [
    Path("tests/result/phase875_nonclean_output_transition_route_audit/combined_phase876/phase875_summary.json"),
    Path("tests/result/phase876_nonclean_route_causal_validation/validation/phase876_cross_model_summary.json"),
    Path("tests/result/phase877_effective_transition_source_gate_audit/source_gate_phase876/phase877_summary.json"),
    Path("tests/result/phase878_full_vocab_blocker_displacement_audit/source_gate_phase876/phase878_summary.json"),
    Path("tests/result/phase879_blocker_min_cut_proxy_audit/source_gate_phase876/phase879_summary.json"),
]
PHASE880_ROWS = Path("tests/result/phase880_counterfactual_gear_min_cut_validation/gear_subset_phase879/phase880_deepseek7b_rows.jsonl")
PHASE881_DIR = Path("tests/result/phase881_dominant_gear_robustness_and_repair/dominant_l27c16651_repair")
PHASE882_DIR = Path("tests/result/phase882_domain_conditioned_dominant_gear_discovery/material_color_domain_discovery")


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path) if path.exists() else []


def clipped(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return min(hi, max(lo, float(value)))


def evidence_label(row: dict[str, Any]) -> str:
    if row.get("phase_source") == "phase880" and row.get("gear_set_minimal_candidate"):
        return "gear_set_minimal_cut_candidate"
    if row.get("phase_source") == "phase880" and row.get("proper_subset_also_boundary_closed"):
        return "observed_pair_not_minimal"
    if row.get("domain_specific_rate", 0.0) > 0:
        return "domain_repair"
    if row.get("cross_domain_rate", 0.0) > 0:
        return "cross_domain_side_effect"
    if row.get("closure_rate", 0.0) > 0:
        return "repair_candidate"
    if row.get("mean_blocker_reduction", 0.0) and row.get("mean_blocker_reduction", 0.0) > 1.0:
        return "weak_modulator"
    if row.get("candidate_source_established"):
        return "candidate_source_no_repair"
    return "no_repair_or_context"


def atlas_score(row: dict[str, Any]) -> float:
    closure = finite(row.get("closure_rate"))
    answer = finite(row.get("answer_rate"))
    domain = finite(row.get("domain_specific_rate"))
    cross = finite(row.get("cross_domain_rate"))
    false_closed = finite(row.get("false_closed_rate"))
    blocker = clipped(finite(row.get("mean_blocker_reduction")) / 10.0)
    control = 1.0 if row.get("control_type") == "same_layer_random" else 0.0
    minimal_bonus = 1.0 if row.get("gear_set_minimal_candidate") else 0.0
    minimal_fail = 1.0 if row.get("proper_subset_also_boundary_closed") else 0.0
    minimal_fail_penalty = 4.0 if row.get("phase_source") == "phase880" else 1.5
    return (
        5.0 * domain
        + 3.0 * closure
        + 2.0 * answer
        + 1.0 * blocker
        + 2.0 * minimal_bonus
        - 3.0 * cross
        - 1.0 * false_closed
        - 1.5 * control
        - minimal_fail_penalty * minimal_fail
    )


def add_rates(row: dict[str, Any]) -> dict[str, Any]:
    n = max(1, int(row.get("n") or 0))
    row["closure_rate"] = finite(row.get("closure_from_open")) / n
    row["answer_rate"] = finite(row.get("answer_gain")) / n
    row["domain_specific_rate"] = finite(row.get("domain_specific_closure")) / n
    row["cross_domain_rate"] = finite(row.get("cross_domain_closure")) / n
    intervened_closed = finite(row.get("intervened_boundary_closed"))
    closure = finite(row.get("closure_from_open"))
    row["false_closed_count"] = max(0.0, intervened_closed - closure)
    row["false_closed_rate"] = row["false_closed_count"] / n
    row["evidence_label"] = evidence_label(row)
    row["atlas_score"] = atlas_score(row)
    return row


def group_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row.get(key, "")) for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_values, vals in buckets.items():
        base = {key: value for key, value in zip(keys, key_values, strict=False)}
        n = len(vals)
        item = {
            **base,
            "n": n,
            "candidate_source_established": True,
            "closure_from_open": sum(1 for row in vals if row.get("closure_from_open")),
            "answer_gain": sum(1 for row in vals if row.get("answer_gain")),
            "domain_specific_closure": sum(1 for row in vals if row.get("domain_specific_closure")),
            "cross_domain_closure": sum(1 for row in vals if row.get("cross_domain_closure")),
            "intervened_boundary_closed": sum(1 for row in vals if row.get("intervened_class_boundary_closed")),
            "mean_blocker_reduction": mean([finite(row.get("blocker_reduction")) for row in vals if row.get("blocker_reduction") is not None]) or 0.0,
            "mean_class_logit_delta": mean([finite(row.get("class_logit_delta")) for row in vals if row.get("class_logit_delta") is not None]) or 0.0,
        }
        out.append(add_rates(item))
    out.sort(key=lambda row: finite(row.get("atlas_score")), reverse=True)
    return out


def phase880_edges() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in read_jsonl(PHASE880_ROWS):
        n = 1
        item = {
            "phase_source": "phase880",
            "model": row.get("model"),
            "candidate_key": row.get("candidate_key"),
            "gear_key": "+".join(row.get("gear_keys") or []),
            "domain": row.get("domain"),
            "eval_domain": row.get("domain"),
            "discovery_domain": row.get("domain"),
            "prompt_variant": row.get("prompt_variant"),
            "control_type": "counterfactual_subset",
            "n": n,
            "candidate_source_established": True,
            "closure_from_open": 1 if row.get("full_class_boundary_closed") and not row.get("base_class_boundary_closed") else 0,
            "answer_gain": 1 if row.get("full_answer_like") and row.get("base_rollout_label") not in {"strict_canonical", "answer_alias"} else 0,
            "domain_specific_closure": 1 if row.get("full_class_boundary_closed") and not row.get("base_class_boundary_closed") else 0,
            "cross_domain_closure": 0,
            "intervened_boundary_closed": 1 if row.get("full_class_boundary_closed") else 0,
            "mean_blocker_reduction": finite(row.get("base_class_blocker_count")) - finite(row.get("full_class_blocker_count")),
            "mean_class_logit_delta": 0.0,
            "gear_set_minimal_candidate": bool(row.get("gear_set_boundary_minimal_candidate")),
            "proper_subset_also_boundary_closed": bool(row.get("proper_subset_also_boundary_closed_count"))
            or "proper_subset" in str(row.get("minimality_class") or ""),
            "minimality_class": row.get("minimality_class"),
            "transition_class": row.get("transition_class"),
        }
        out.append(add_rates(item))
    return out


def phase881_edges() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ("qwen3", "glm4", "deepseek7b"):
        rows.extend(read_jsonl(PHASE881_DIR / f"phase881_{model}_rows.jsonl"))
    grouped = group_rows(rows, ["model", "candidate_source", "gear_key", "candidate_key", "domain"])
    for row in grouped:
        row["phase_source"] = "phase881"
        row["control_type"] = "none"
        row["discovery_domain"] = row.get("domain")
        row["eval_domain"] = row.get("domain")
        row["gear_set_minimal_candidate"] = False
        row["proper_subset_also_boundary_closed"] = False
        row["evidence_label"] = evidence_label(row)
        row["atlas_score"] = atlas_score(row)
    return grouped


def phase882_edges() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ("qwen3", "glm4", "deepseek7b"):
        rows.extend(read_jsonl(PHASE882_DIR / f"phase882_{model}_rows.jsonl"))
    grouped = group_rows(rows, ["model", "candidate_source", "control_type", "discovery_domain", "eval_domain", "gear_key", "candidate_key"])
    for row in grouped:
        row["phase_source"] = "phase882"
        row["gear_set_minimal_candidate"] = False
        row["proper_subset_also_boundary_closed"] = False
        row["evidence_label"] = evidence_label(row)
        row["atlas_score"] = atlas_score(row)
    return grouped


def phase_context() -> list[dict[str, Any]]:
    out = []
    for path in PHASE_CONTEXT_FILES:
        payload = read_json(path)
        out.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "phase": payload.get("phase"),
                "title": payload.get("title"),
                "status": payload.get("status"),
                "summary_keys": sorted(payload.keys())[:40] if payload else [],
            }
        )
    return out


def summarize(edges: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(str(row.get("evidence_label")) for row in edges)
    phase_counts = Counter(str(row.get("phase_source")) for row in edges)
    model_counts = Counter(str(row.get("model")) for row in edges)
    control_counts = Counter(str(row.get("control_type")) for row in edges)
    return {
        "n_edges": len(edges),
        "phase_counts": dict(phase_counts),
        "model_counts": dict(model_counts),
        "control_counts": dict(control_counts),
        "evidence_label_counts": dict(label_counts),
        "positive_score_edges": sum(1 for row in edges if finite(row.get("atlas_score")) > 0),
        "negative_score_edges": sum(1 for row in edges if finite(row.get("atlas_score")) < 0),
        "mean_atlas_score": mean([finite(row.get("atlas_score")) for row in edges]),
        "top_score_edges": edges[:20],
        "bottom_score_edges": sorted(edges, key=lambda row: finite(row.get("atlas_score")))[:20],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 883 Atlas Score Table ({payload['round']})",
        "",
        "- Boundary: offline evidence calibration; no new model forward pass.",
        "- Inputs: Phase875-879 context, Phase880 counterfactual subset rows, Phase881 robustness rows, Phase882 discovery/control rows.",
        "",
        "## Summary",
        "",
        f"- Overall: `{payload['summary']}`",
        "",
        "## Score Formula",
        "",
        "```text",
        "S_edge =",
        "  5 * domain_specific_rate",
        "+ 3 * closure_rate",
        "+ 2 * answer_rate",
        "+ 1 * clipped(mean_blocker_reduction / 10)",
        "+ 2 * minimal_bonus",
        "- 3 * cross_domain_rate",
        "- 1 * false_closed_rate",
        "- 1.5 * same_layer_random_control",
        "- 1.5 * proper_subset_minimality_failure",
        "```",
        "",
        "## Top Edges",
        "",
        "| score | label | phase | model | candidate | domain | eval | n | closure | answer | false closed |",
        "|---:|---|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["edges"][:40]:
        lines.append(
            f"| {finite(row.get('atlas_score')):.3f} | {row.get('evidence_label')} | {row.get('phase_source')} | "
            f"{row.get('model')} | `{row.get('candidate_key')}` | {row.get('discovery_domain') or row.get('domain')} | "
            f"{row.get('eval_domain') or row.get('domain')} | {row.get('n')} | {row.get('closure_from_open')} | "
            f"{row.get('answer_gain')} | {row.get('false_closed_count')} |"
        )
    lines += [
        "",
        "## Bottom Edges",
        "",
        "| score | label | phase | model | candidate | domain | eval | n | cross | false closed | control |",
        "|---:|---|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in payload["summary"]["bottom_score_edges"][:30]:
        lines.append(
            f"| {finite(row.get('atlas_score')):.3f} | {row.get('evidence_label')} | {row.get('phase_source')} | "
            f"{row.get('model')} | `{row.get('candidate_key')}` | {row.get('discovery_domain') or row.get('domain')} | "
            f"{row.get('eval_domain') or row.get('domain')} | {row.get('n')} | {row.get('cross_domain_closure')} | "
            f"{row.get('false_closed_count')} | {row.get('control_type')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = RESULT_ROOT / ROUND_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = phase880_edges() + phase881_edges() + phase882_edges()
    edges.sort(key=lambda row: finite(row.get("atlas_score")), reverse=True)
    payload = {
        "phase": PHASE,
        "title": "Atlas Score Table and Evidence-Level Calibration",
        "round": ROUND_NAME,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase_context": phase_context(),
        "score_formula": {
            "domain_specific_rate": 5.0,
            "closure_rate": 3.0,
            "answer_rate": 2.0,
            "mean_blocker_reduction_clipped": 1.0,
            "minimal_bonus": 2.0,
            "cross_domain_rate_penalty": -3.0,
            "false_closed_rate_penalty": -1.0,
            "same_layer_random_control_penalty": -1.5,
            "proper_subset_minimality_failure_penalty": -1.5,
        },
        "summary": summarize(edges),
        "edges": edges,
        "boundary": "Offline atlas scoring, not closure. Scores are heuristic calibration weights for prioritizing future audits.",
    }
    p846.write_json(out_dir / "phase883_atlas_score_summary.json", payload)
    p846.write_jsonl(out_dir / "phase883_atlas_score_edges.jsonl", edges)
    write_markdown(out_dir / "phase883_atlas_score_summary.md", payload)
    print(json.dumps({k: payload[k] for k in ("phase", "round", "summary")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
