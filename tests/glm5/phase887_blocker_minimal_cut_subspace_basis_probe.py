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


PHASE = 887
MODELS = ["qwen3", "glm4", "deepseek7b"]
PHASE885_ROOT = Path(
    "tests/result/phase885_stable_boundary_minimality_cross_model_audit/holdout_minimality_cross_model"
)
PHASE886_ROOT = Path(
    "tests/result/phase886_local_subspace_boundary_decomposition/local_subspace_decomposition"
)
RESULT_ROOT = Path("tests/result/phase887_blocker_minimal_cut_subspace_basis_probe")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return p846.mean(values)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def counter_values(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def row_pair_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("parent_candidate_key") or ""),
        str(row.get("case_id") or ""),
        str(row.get("prompt_variant") or ""),
        str(row.get("case_split") or ""),
        str(row.get("eval_domain") or ""),
    )


def phase886_pair_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("parent_candidate_key") or ""),
        str(row.get("case_id") or ""),
        str(row.get("prompt_variant") or ""),
        str(row.get("case_split") or ""),
        str(row.get("eval_domain") or ""),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path) if path.exists() else []


def load_phase885_candidate_details(model: str) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    rows = load_jsonl(PHASE885_ROOT / f"phase885_{model}_rows.jsonl")
    details: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("condition_type") != "candidate" or row.get("control_type") != "none":
            continue
        blockers = row.get("base_top_blockers") or []
        token_table: dict[int, dict[str, Any]] = {}
        for item in blockers:
            if item.get("token_id") is None:
                continue
            try:
                token_id = int(item["token_id"])
            except (TypeError, ValueError):
                continue
            token_table[token_id] = {
                "token_id": token_id,
                "token": item.get("token"),
                "role": item.get("role"),
                "logit": item.get("logit"),
                "gap_vs_threshold": item.get("gap_vs_threshold"),
            }
        details[row_pair_key(row)] = {
            "base_top_blockers": blockers,
            "token_table": token_table,
            "base_class_blocker_count": row.get("base_class_blocker_count"),
            "base_class_logit": row.get("base_class_logit"),
            "base_class_rank": row.get("base_class_rank"),
        }
    return details


def token_records(token_ids: set[int], token_table: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for token_id in sorted(token_ids):
        item = dict(token_table.get(token_id) or {"token_id": token_id})
        item.setdefault("token_id", token_id)
        out.append(item)
    return out


def cut_role_counter(records: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(item.get("role") or "unknown") for item in records)


def classify_cut(pair: dict[str, Any]) -> dict[str, Any]:
    base_ids = set(int(x) for x in pair.get("base_blocker_ids") or [])
    shared_removed = set(int(x) for x in pair.get("shared_removed_ids") or [])
    candidate_removed = set(int(x) for x in pair.get("candidate_removed_ids") or [])
    opposite_removed = set(int(x) for x in pair.get("opposite_removed_ids") or [])
    candidate_residual = set(int(x) for x in pair.get("candidate_residual_ids") or [])
    opposite_residual = set(int(x) for x in pair.get("opposite_residual_ids") or [])
    base_count = int(pair.get("base_blocker_count") or 0)
    base_topk = int(pair.get("base_blocker_topk") or len(base_ids))
    topk_complete_available = bool(base_ids and base_count <= base_topk)
    candidate_complete_topk_cut = bool(
        pair.get("candidate_closure") and topk_complete_available and base_ids <= candidate_removed
    )
    opposite_complete_topk_cut = bool(
        pair.get("opposite_closure") and topk_complete_available and base_ids <= opposite_removed
    )
    shared_complete_topk_cut = bool(
        pair.get("same_boundary_closure") and topk_complete_available and base_ids <= shared_removed
    )
    exact_single_blocker_cut = bool(
        pair.get("same_boundary_closure") and base_count == 1 and len(shared_removed) == 1 and base_ids <= shared_removed
    )
    candidate_observed_sufficient = bool(pair.get("candidate_closure") and not candidate_residual)
    opposite_observed_sufficient = bool(pair.get("opposite_closure") and not opposite_residual)
    shared_observed_sufficient = bool(pair.get("same_boundary_closure") and base_ids <= shared_removed)
    return {
        "topk_complete_available": topk_complete_available,
        "candidate_complete_topk_cut": candidate_complete_topk_cut,
        "opposite_complete_topk_cut": opposite_complete_topk_cut,
        "shared_complete_topk_cut": shared_complete_topk_cut,
        "exact_single_blocker_cut": exact_single_blocker_cut,
        "candidate_observed_sufficient": candidate_observed_sufficient,
        "opposite_observed_sufficient": opposite_observed_sufficient,
        "shared_observed_sufficient": shared_observed_sufficient,
        "observed_min_cut_size": len(base_ids) if shared_complete_topk_cut else None,
        "shared_cut_size": len(shared_removed),
        "candidate_cut_size": len(candidate_removed),
        "opposite_cut_size": len(opposite_removed),
    }


def make_cut_rows(model: str) -> list[dict[str, Any]]:
    pairs = load_jsonl(PHASE886_ROOT / f"phase886_{model}_pairs.jsonl")
    details = load_phase885_candidate_details(model)
    out = []
    for pair in pairs:
        key = phase886_pair_key(pair)
        detail = details.get(key, {})
        token_table = detail.get("token_table") or {}
        base_ids = set(int(x) for x in pair.get("base_blocker_ids") or [])
        shared_removed = set(int(x) for x in pair.get("shared_removed_ids") or [])
        candidate_removed = set(int(x) for x in pair.get("candidate_removed_ids") or [])
        opposite_removed = set(int(x) for x in pair.get("opposite_removed_ids") or [])
        cut_flags = classify_cut(pair)
        exact_cut_ids = base_ids if cut_flags["shared_complete_topk_cut"] else set()
        exact_single_ids = shared_removed if cut_flags["exact_single_blocker_cut"] else set()
        exact_records = token_records(exact_cut_ids, token_table)
        shared_records = token_records(shared_removed, token_table)
        out.append(
            {
                "phase": PHASE,
                "row_kind": "phase887_blocker_minimal_cut_probe",
                "model": model,
                "parent_candidate_key": pair.get("parent_candidate_key"),
                "case_id": pair.get("case_id"),
                "prompt_variant": pair.get("prompt_variant"),
                "case_split": pair.get("case_split"),
                "eval_domain": pair.get("eval_domain"),
                "object": pair.get("object"),
                "canonical_answer": pair.get("canonical_answer"),
                "candidate_mode": pair.get("candidate_mode"),
                "opposite_mode": pair.get("opposite_mode"),
                "candidate_closure": bool(pair.get("candidate_closure")),
                "opposite_closure": bool(pair.get("opposite_closure")),
                "same_boundary_closure": bool(pair.get("same_boundary_closure")),
                "candidate_clean_like": bool(pair.get("candidate_clean_like")),
                "candidate_nonclean_like": bool(pair.get("candidate_nonclean_like")),
                "opposite_clean_like": bool(pair.get("opposite_clean_like")),
                "opposite_nonclean_like": bool(pair.get("opposite_nonclean_like")),
                "removed_jaccard": pair.get("removed_jaccard"),
                "base_blocker_count": pair.get("base_blocker_count"),
                "base_blocker_topk": pair.get("base_blocker_topk"),
                "base_blocker_ids": sorted(base_ids),
                "candidate_removed_ids": sorted(candidate_removed),
                "opposite_removed_ids": sorted(opposite_removed),
                "shared_removed_ids": sorted(shared_removed),
                "exact_cut_ids": sorted(exact_cut_ids),
                "exact_single_cut_ids": sorted(exact_single_ids),
                "shared_removed_records": shared_records,
                "exact_cut_records": exact_records,
                "shared_removed_role_counts": counter_values(cut_role_counter(shared_records)),
                "exact_cut_role_counts": counter_values(cut_role_counter(exact_records)),
                **cut_flags,
            }
        )
    return out


def token_counter_from_rows(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    counts: Counter[int] = Counter()
    roles: dict[int, Counter[str]] = defaultdict(Counter)
    tokens: dict[int, Counter[str]] = defaultdict(Counter)
    domains: dict[int, Counter[str]] = defaultdict(Counter)
    objects: dict[int, Counter[str]] = defaultdict(Counter)
    prompts: dict[int, Counter[str]] = defaultdict(Counter)
    clean: Counter[int] = Counter()
    nonclean: Counter[int] = Counter()
    for row in rows:
        records = row.get(field) or []
        for item in records:
            token_id = int(item.get("token_id"))
            counts[token_id] += 1
            roles[token_id][str(item.get("role") or "unknown")] += 1
            tokens[token_id][str(item.get("token") or token_id)] += 1
            domains[token_id][str(row.get("eval_domain"))] += 1
            objects[token_id][str(row.get("object"))] += 1
            prompts[token_id][str(row.get("prompt_variant"))] += 1
            if row.get("candidate_clean_like") or row.get("opposite_clean_like"):
                clean[token_id] += 1
            if row.get("candidate_nonclean_like") or row.get("opposite_nonclean_like"):
                nonclean[token_id] += 1
    out = []
    for token_id, n in counts.most_common():
        out.append(
            {
                "token_id": token_id,
                "n": int(n),
                "token": tokens[token_id].most_common(1)[0][0],
                "roles": counter_values(roles[token_id]),
                "domains": counter_values(domains[token_id]),
                "objects": counter_values(objects[token_id]),
                "prompts": counter_values(prompts[token_id]),
                "clean_like_count": int(clean[token_id]),
                "nonclean_like_count": int(nonclean[token_id]),
            }
        )
    return out


def evidence_label(group: dict[str, Any]) -> str:
    if int(group.get("same_boundary_closure") or 0) == 0:
        return "negative_no_min_cut"
    if int(group.get("exact_single_blocker_cut") or 0) >= 3:
        return "single_token_minimal_cut_signal"
    if int(group.get("shared_complete_topk_cut") or 0) >= 3:
        return "topk_complete_minimal_cut_signal"
    if int(group.get("candidate_complete_topk_cut") or 0) or int(group.get("opposite_complete_topk_cut") or 0):
        return "directional_topk_cut_signal"
    return "partial_cut_signal"


def summarize_rows(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("parent_candidate_key"))].append(row)
    groups = []
    for key, vals in buckets.items():
        n = len(vals)
        exact_rows = [row for row in vals if row.get("shared_complete_topk_cut")]
        exact_single_rows = [row for row in vals if row.get("exact_single_blocker_cut")]
        clean_exact = sum(1 for row in exact_rows if row.get("candidate_clean_like") or row.get("opposite_clean_like"))
        nonclean_exact = sum(1 for row in exact_rows if row.get("candidate_nonclean_like") or row.get("opposite_nonclean_like"))
        exact_cut_sizes = [int(row.get("observed_min_cut_size") or 0) for row in exact_rows]
        shared_cut_sizes = [int(row.get("shared_cut_size") or 0) for row in vals if int(row.get("shared_cut_size") or 0) > 0]
        role_counts: Counter[str] = Counter()
        for row in exact_rows:
            role_counts.update(row.get("exact_cut_role_counts") or {})
        group = {
            "model": model,
            "parent_candidate_key": key,
            "n_pairs": n,
            "candidate_closure": sum(1 for row in vals if row.get("candidate_closure")),
            "opposite_closure": sum(1 for row in vals if row.get("opposite_closure")),
            "same_boundary_closure": sum(1 for row in vals if row.get("same_boundary_closure")),
            "candidate_complete_topk_cut": sum(1 for row in vals if row.get("candidate_complete_topk_cut")),
            "opposite_complete_topk_cut": sum(1 for row in vals if row.get("opposite_complete_topk_cut")),
            "shared_complete_topk_cut": len(exact_rows),
            "exact_single_blocker_cut": len(exact_single_rows),
            "topk_complete_available": sum(1 for row in vals if row.get("topk_complete_available")),
            "mean_removed_jaccard": mean(
                [finite(row.get("removed_jaccard")) for row in vals if row.get("removed_jaccard") is not None]
            )
            or 0.0,
            "mean_exact_cut_size": mean(exact_cut_sizes) or 0.0,
            "min_exact_cut_size": min(exact_cut_sizes) if exact_cut_sizes else None,
            "max_exact_cut_size": max(exact_cut_sizes) if exact_cut_sizes else None,
            "mean_shared_cut_size": mean(shared_cut_sizes) or 0.0,
            "exact_clean_like": clean_exact,
            "exact_nonclean_like": nonclean_exact,
            "exact_cut_role_counts": counter_values(role_counts),
            "top_exact_cut_tokens": token_counter_from_rows(exact_rows, "exact_cut_records")[:12],
            "top_shared_removed_tokens": token_counter_from_rows(vals, "shared_removed_records")[:12],
            "objects_with_exact_cut": sorted(set(str(row.get("object")) for row in exact_rows)),
            "prompts_with_exact_cut": sorted(set(str(row.get("prompt_variant")) for row in exact_rows)),
        }
        group["evidence_label"] = evidence_label(group)
        group["phase887_score"] = (
            3.0 * (group["exact_single_blocker_cut"] / max(1, n))
            + 2.0 * (group["shared_complete_topk_cut"] / max(1, n))
            + 1.0 * (group["same_boundary_closure"] / max(1, n))
            + min(1.0, finite(group["mean_removed_jaccard"]))
            - 0.5 * (group["exact_nonclean_like"] / max(1, n))
        )
        groups.append(group)
    groups.sort(key=lambda row: finite(row.get("phase887_score")), reverse=True)
    exact_rows = [row for row in rows if row.get("shared_complete_topk_cut")]
    exact_single_rows = [row for row in rows if row.get("exact_single_blocker_cut")]
    return {
        "phase": PHASE,
        "model": model,
        "source_phase": 886,
        "paired_rows": len(rows),
        "overall": {
            "candidate_closure": sum(1 for row in rows if row.get("candidate_closure")),
            "opposite_closure": sum(1 for row in rows if row.get("opposite_closure")),
            "same_boundary_closure": sum(1 for row in rows if row.get("same_boundary_closure")),
            "candidate_complete_topk_cut": sum(1 for row in rows if row.get("candidate_complete_topk_cut")),
            "opposite_complete_topk_cut": sum(1 for row in rows if row.get("opposite_complete_topk_cut")),
            "shared_complete_topk_cut": len(exact_rows),
            "exact_single_blocker_cut": len(exact_single_rows),
            "mean_removed_jaccard": mean(
                [finite(row.get("removed_jaccard")) for row in rows if row.get("removed_jaccard") is not None]
            )
            or 0.0,
            "top_exact_cut_tokens": token_counter_from_rows(exact_rows, "exact_cut_records")[:20],
            "top_shared_removed_tokens": token_counter_from_rows(rows, "shared_removed_records")[:20],
        },
        "candidate_groups": groups,
        "evidence_label_counts": counter_values(Counter(str(row.get("evidence_label")) for row in groups)),
    }


def write_model_outputs(model: str) -> dict[str, Any]:
    rows = make_cut_rows(model)
    summary = summarize_rows(model, rows)
    out_dir = RESULT_ROOT / "blocker_minimal_cut_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    p846.write_jsonl(out_dir / f"phase887_{model}_cut_rows.jsonl", rows)
    p846.write_json(out_dir / f"phase887_{model}_summary.json", summary)
    log(
        "{model}: pairs={pairs} shared_complete={shared} exact_single={single}".format(
            model=model,
            pairs=len(rows),
            shared=summary["overall"]["shared_complete_topk_cut"],
            single=summary["overall"]["exact_single_blocker_cut"],
        )
    )
    return summary


def markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 887 blocker-token minimal cut and subspace basis probe",
        "",
        "## Overall",
        "",
        f"- paired_rows: {payload.get('paired_rows')}",
        f"- same_boundary_closure: {payload.get('overall', {}).get('same_boundary_closure')}",
        f"- shared_complete_topk_cut: {payload.get('overall', {}).get('shared_complete_topk_cut')}",
        f"- exact_single_blocker_cut: {payload.get('overall', {}).get('exact_single_blocker_cut')}",
        f"- candidate_complete_topk_cut: {payload.get('overall', {}).get('candidate_complete_topk_cut')}",
        f"- opposite_complete_topk_cut: {payload.get('overall', {}).get('opposite_complete_topk_cut')}",
        "",
        "## Candidate groups",
        "",
        "| model | candidate | label | pairs | both | shared cut | single cut | exact clean | exact nonclean | mean cut |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("candidate_groups", [])[:30]:
        lines.append(
            "| {model} | {key} | {label} | {n} | {both} | {shared} | {single} | {clean} | {nonclean} | {mean_cut:.2f} |".format(
                model=row.get("model"),
                key=row.get("parent_candidate_key"),
                label=row.get("evidence_label"),
                n=row.get("n_pairs"),
                both=row.get("same_boundary_closure"),
                shared=row.get("shared_complete_topk_cut"),
                single=row.get("exact_single_blocker_cut"),
                clean=row.get("exact_clean_like"),
                nonclean=row.get("exact_nonclean_like"),
                mean_cut=finite(row.get("mean_exact_cut_size")),
            )
        )
    return "\n".join(lines) + "\n"


def summarize_cross_model() -> dict[str, Any]:
    out_dir = RESULT_ROOT / "blocker_minimal_cut_probe"
    summaries = [read_json(out_dir / f"phase887_{model}_summary.json") for model in MODELS]
    summaries = [item for item in summaries if item]
    all_groups: list[dict[str, Any]] = []
    overall = Counter()
    paired_rows = 0
    exact_token_rows: list[dict[str, Any]] = []
    shared_token_rows: list[dict[str, Any]] = []
    for summary in summaries:
        paired_rows += int(summary.get("paired_rows") or 0)
        all_groups.extend(summary.get("candidate_groups") or [])
        for key, value in (summary.get("overall") or {}).items():
            if key.startswith("top_") or key.startswith("mean_"):
                continue
            overall[key] += int(value or 0)
        exact_token_rows.extend(summary.get("overall", {}).get("top_exact_cut_tokens") or [])
        shared_token_rows.extend(summary.get("overall", {}).get("top_shared_removed_tokens") or [])
    all_groups.sort(key=lambda row: finite(row.get("phase887_score")), reverse=True)
    payload = {
        "phase": PHASE,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_phase": 886,
        "paired_rows": paired_rows,
        "models": [item.get("model") for item in summaries],
        "overall": {key: int(value) for key, value in sorted(overall.items())},
        "evidence_label_counts": counter_values(Counter(str(row.get("evidence_label")) for row in all_groups)),
        "candidate_groups": all_groups,
        "top_groups": all_groups[:20],
        "top_exact_cut_tokens_by_model": exact_token_rows,
        "top_shared_removed_tokens_by_model": shared_token_rows,
    }
    p846.write_json(out_dir / "phase887_cross_model_summary.json", payload)
    (out_dir / "phase887_cross_model_summary.md").write_text(markdown_summary(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 887: estimate blocker-token minimal cuts from Phase886 local subspace pairs."
    )
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize:
        payload = summarize_cross_model()
        log(
            "cross_model: pairs={pairs} shared_complete={shared} exact_single={single}".format(
                pairs=payload.get("paired_rows"),
                shared=payload.get("overall", {}).get("shared_complete_topk_cut"),
                single=payload.get("overall", {}).get("exact_single_blocker_cut"),
            )
        )
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize is set")
    write_model_outputs(args.model)


if __name__ == "__main__":
    main()
