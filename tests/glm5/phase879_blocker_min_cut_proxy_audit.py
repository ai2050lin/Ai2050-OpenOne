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


PHASE = 879
RESULT_ROOT = Path("tests/result/phase879_blocker_min_cut_proxy_audit")
DEFAULT_ROWS = Path(
    "tests/result/phase878_full_vocab_blocker_displacement_audit/source_gate_phase876/phase878_displacement_rows.jsonl"
)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path) if path.exists() else []


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def token_key(item: dict[str, Any]) -> str:
    token_id = item.get("token_id")
    if token_id is not None:
        return f"id:{token_id}"
    return f"tok:{item.get('token')}"


def role_counts(tokens: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(item.get("role")) for item in tokens)


def token_set(tokens: list[dict[str, Any]]) -> set[str]:
    return {token_key(item) for item in tokens}


def nonzero_role_delta(row: dict[str, Any]) -> bool:
    return any(finite(value) != 0 for value in (row.get("top_role_delta") or {}).values())


def blocker_cut_roles(row: dict[str, Any]) -> dict[str, int]:
    base = list(row.get("base_blocker_tokens") or [])
    intervened_keys = token_set(list(row.get("intervened_blocker_tokens") or []))
    removed = [item for item in base if token_key(item) not in intervened_keys]
    return dict(role_counts(removed))


def cut_tokens(row: dict[str, Any]) -> list[dict[str, Any]]:
    base = list(row.get("base_blocker_tokens") or [])
    intervened_keys = token_set(list(row.get("intervened_blocker_tokens") or []))
    return [
        {
            "token": item.get("token"),
            "token_id": item.get("token_id"),
            "role": item.get("role"),
            "logit": item.get("logit"),
            "gap_vs_threshold": item.get("gap_vs_threshold"),
        }
        for item in base
        if token_key(item) not in intervened_keys
    ]


def displacement_subtype(row: dict[str, Any]) -> str:
    classification_cut = finite(row.get("blocker_count_reduction_raw")) > 0 and finite(row.get("removed_blocker_token_count")) > 0
    if not classification_cut:
        return "no_observed_blocker_cut"
    top_changed = bool(row.get("top_set_changed"))
    role_changed = nonzero_role_delta(row)
    if top_changed and role_changed:
        return "top_membership_and_role_displacement"
    if top_changed:
        return "top_membership_displacement"
    if role_changed:
        return "role_reweight_displacement"
    return "rank_threshold_reclassification"


def minimal_proxy_status(row: dict[str, Any]) -> str:
    base_count = finite(row.get("base_blocker_count"))
    int_count = finite(row.get("intervened_blocker_count"))
    removed = finite(row.get("removed_blocker_token_count"))
    target_top1 = bool(row.get("target_rank_reached_top1"))
    if base_count > 0 and int_count == 0 and removed >= base_count and target_top1:
        return "observed_blocker_boundary_closed"
    if base_count > 0 and int_count < base_count and target_top1:
        return "partial_observed_cut_with_target_takeover"
    if target_top1:
        return "target_takeover_without_observed_cut"
    return "not_closed_in_observed_topk"


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    cut = cut_tokens(row)
    cut_size = len(cut)
    base_count = finite(row.get("base_blocker_count"))
    int_count = finite(row.get("intervened_blocker_count"))
    rank_improvement = finite(row.get("target_rank_improvement"))
    status = minimal_proxy_status(row)
    subtype = displacement_subtype(row)
    observed_proxy_closed = status == "observed_blocker_boundary_closed"
    original_weakening = finite(row.get("original_blocker_delta")) < 0
    return {
        "phase": PHASE,
        "model": row.get("model"),
        "domain": row.get("domain"),
        "case_id": row.get("case_id"),
        "object": row.get("object"),
        "prompt_variant": row.get("prompt_variant"),
        "candidate_key": row.get("candidate_key"),
        "edit_mode": row.get("edit_mode"),
        "transition_class": row.get("transition_class"),
        "primary_route": row.get("primary_route"),
        "source_gate": row.get("source_gate"),
        "prompt_gate": row.get("prompt_gate"),
        "field_gates": row.get("field_gates") or [],
        "transition_label": row.get("transition_label"),
        "base_generated_clean": row.get("base_generated_clean"),
        "intervened_generated_clean": row.get("intervened_generated_clean"),
        "base_blocker_count": base_count,
        "intervened_blocker_count": int_count,
        "blocker_count_reduction_raw": finite(row.get("blocker_count_reduction_raw")),
        "base_target_rank": finite(row.get("base_target_rank")),
        "intervened_target_rank": finite(row.get("intervened_target_rank")),
        "target_rank_improvement": rank_improvement,
        "target_rank_reached_top1": bool(row.get("target_rank_reached_top1")),
        "target_logit_delta_raw": finite(row.get("target_logit_delta_raw")),
        "original_blocker_delta": finite(row.get("original_blocker_delta")),
        "original_blocker_weakening": original_weakening,
        "count_reduced_without_original_blocker_reduction": bool(row.get("count_reduced_without_original_blocker_reduction")),
        "blocker_set_changed": bool(row.get("blocker_set_changed")),
        "top_set_changed": bool(row.get("top_set_changed")),
        "top_role_changed": nonzero_role_delta(row),
        "top_token_overlap_ratio": row.get("top_token_overlap_ratio"),
        "base_top1_role_raw": row.get("base_top1_role_raw"),
        "intervened_top1_role_raw": row.get("intervened_top1_role_raw"),
        "base_top1_token_raw": row.get("base_top1_token_raw"),
        "intervened_top1_token_raw": row.get("intervened_top1_token_raw"),
        "top_role_delta": row.get("top_role_delta") or {},
        "cut_tokens": cut,
        "cut_token_count": cut_size,
        "cut_role_counts": blocker_cut_roles(row),
        "observed_proxy_status": status,
        "observed_proxy_closed": observed_proxy_closed,
        "displacement_subtype": subtype,
        "rank_only_cut": subtype == "rank_threshold_reclassification",
        "membership_cut": subtype in {"top_membership_displacement", "top_membership_and_role_displacement"},
        "role_cut": subtype in {"role_reweight_displacement", "top_membership_and_role_displacement"},
        "proxy_cut_size": cut_size if observed_proxy_closed else None,
        "proxy_cut_equals_base_blocker_count": cut_size == int(base_count),
        "proxy_cut_equals_rank_improvement": cut_size == int(rank_improvement),
    }


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    return {name: summarize_group(items) for name, items in sorted(groups.items())}


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "transition_class_counts": dict(Counter(str(row.get("transition_class")) for row in rows)),
        "route_counts": dict(Counter(str(row.get("primary_route")) for row in rows)),
        "candidate_counts": dict(Counter(str(row.get("candidate_key")) for row in rows)),
        "objects": sorted({str(row.get("object")) for row in rows}),
        "prompts": sorted({str(row.get("prompt_variant")) for row in rows}),
        "observed_proxy_closed": sum(1 for row in rows if row.get("observed_proxy_closed")),
        "observed_proxy_status_counts": dict(Counter(str(row.get("observed_proxy_status")) for row in rows)),
        "displacement_subtype_counts": dict(Counter(str(row.get("displacement_subtype")) for row in rows)),
        "rank_only_cut": sum(1 for row in rows if row.get("rank_only_cut")),
        "membership_cut": sum(1 for row in rows if row.get("membership_cut")),
        "role_cut": sum(1 for row in rows if row.get("role_cut")),
        "cut_role_counts": dict(Counter(role for row in rows for role, count in (row.get("cut_role_counts") or {}).items() for _ in range(count))),
        "mean_proxy_cut_size": mean([finite(row.get("proxy_cut_size")) for row in rows if row.get("proxy_cut_size") is not None]),
        "mean_target_rank_improvement": mean([finite(row.get("target_rank_improvement")) for row in rows]),
        "mean_target_logit_delta_raw": mean([finite(row.get("target_logit_delta_raw")) for row in rows]),
        "mean_original_blocker_delta": mean([finite(row.get("original_blocker_delta")) for row in rows]),
        "mean_top_token_overlap_ratio": mean(
            [finite(row.get("top_token_overlap_ratio")) for row in rows if row.get("top_token_overlap_ratio") is not None]
        ),
    }


def object_prompt_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"{row.get('object')}::{row.get('prompt_variant')}"].append(row)
    return {name: summarize_group(items) for name, items in sorted(groups.items())}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nonclean = [row for row in rows if row.get("transition_class") == "nonclean_output_transition"]
    clean = [row for row in rows if row.get("transition_class") == "clean_causal_transition"]
    return {
        "n_rows": len(rows),
        "transition_class_counts": dict(Counter(str(row.get("transition_class")) for row in rows)),
        "route_counts": dict(Counter(str(row.get("primary_route")) for row in rows)),
        "candidate_counts": dict(Counter(str(row.get("candidate_key")) for row in rows)),
        "model_domain_counts": dict(Counter(f"{row.get('model')}:{row.get('domain')}" for row in rows)),
        "all_rows": summarize_group(rows),
        "nonclean": summarize_group(nonclean),
        "clean": summarize_group(clean),
        "by_route": grouped(rows, "primary_route"),
        "by_candidate": grouped(rows, "candidate_key"),
        "by_object_prompt": object_prompt_summary(rows),
        "rows": rows,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Phase 879 Blocker Min-Cut Proxy Audit",
        "",
        "- Boundary: offline proxy audit from saved Phase878 top-k blocker displacement rows; no new model run.",
        "- Goal: separate observed blocker-boundary cuts from true causal minimal cuts.",
        "",
        "## Summary",
        "",
        f"- Rows: `{s['n_rows']}`",
        f"- Transition classes: `{s['transition_class_counts']}`",
        f"- Routes: `{s['route_counts']}`",
        f"- Nonclean: `{s['nonclean']}`",
        f"- Clean: `{s['clean']}`",
        "",
        "## By Route",
        "",
        "| route | n | observed closed | subtype counts | rank-only | membership | role | mean cut | mean rank improve | mean target logit delta | mean original blocker | cut roles |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for route, info in s["by_route"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    route,
                    str(info["n"]),
                    str(info["observed_proxy_closed"]),
                    json.dumps(info["displacement_subtype_counts"], ensure_ascii=False, sort_keys=True),
                    str(info["rank_only_cut"]),
                    str(info["membership_cut"]),
                    str(info["role_cut"]),
                    f"{info['mean_proxy_cut_size']:.3f}" if info["mean_proxy_cut_size"] is not None else "null",
                    f"{info['mean_target_rank_improvement']:.3f}" if info["mean_target_rank_improvement"] is not None else "null",
                    f"{info['mean_target_logit_delta_raw']:.3f}" if info["mean_target_logit_delta_raw"] is not None else "null",
                    f"{info['mean_original_blocker_delta']:.4f}" if info["mean_original_blocker_delta"] is not None else "null",
                    json.dumps(info["cut_role_counts"], ensure_ascii=False, sort_keys=True),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## By Object Prompt",
            "",
            "| object prompt | n | routes | candidates | observed closed | subtype counts | cut roles |",
            "|---|---:|---|---|---:|---|---|",
        ]
    )
    for name, info in s["by_object_prompt"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(info["n"]),
                    json.dumps(info["route_counts"], ensure_ascii=False, sort_keys=True),
                    json.dumps(info["candidate_counts"], ensure_ascii=False, sort_keys=True),
                    str(info["observed_proxy_closed"]),
                    json.dumps(info["displacement_subtype_counts"], ensure_ascii=False, sort_keys=True),
                    json.dumps(info["cut_role_counts"], ensure_ascii=False, sort_keys=True),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- `observed_blocker_boundary_closed` means saved top-k blocker rows show all observed blockers removed and target rank moved to 1.",
            "- It is not a true causal minimal cut, because each blocker token/edge was not counterfactually ablated and logits were not recomputed.",
            "- `rank_threshold_reclassification` means top-k membership can stay stable while blocker labels disappear because target crosses the rank boundary.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [compact_row(row) for row in read_jsonl(args.input)]
    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": PHASE,
        "title": "Blocker Field Minimal Cut Proxy Audit",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": str(args.input),
        "summary": summarize(rows),
        "boundary": "Observed top-k blocker-boundary cut proxy only; no counterfactual minimal-cut model run.",
    }
    write_jsonl(out_dir / "phase879_min_cut_proxy_rows.jsonl", rows)
    (out_dir / "phase879_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(out_dir / "phase879_summary.md", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 879 blocker field minimal-cut proxy audit.")
    parser.add_argument("--input", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--run-name", default="source_gate_phase876")
    args = parser.parse_args()
    payload = run(args)
    s = payload["summary"]
    print(json.dumps({
        "phase": payload["phase"],
        "n_rows": s["n_rows"],
        "transition_class_counts": s["transition_class_counts"],
        "nonclean": s["nonclean"],
        "clean": s["clean"],
        "by_route": s["by_route"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
