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


PHASE = 886
MODELS = ["qwen3", "glm4", "deepseek7b"]
PHASE885_ROOT = Path(
    "tests/result/phase885_stable_boundary_minimality_cross_model_audit/holdout_minimality_cross_model"
)
RESULT_ROOT = Path("tests/result/phase886_local_subspace_boundary_decomposition")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return p846.mean(values)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def sign(value: Any, eps: float = 1e-9) -> int:
    val = finite(value)
    if val > eps:
        return 1
    if val < -eps:
        return -1
    return 0


def token_ids(items: list[dict[str, Any]] | None) -> set[int]:
    out: set[int] = set()
    for item in items or []:
        if item.get("token_id") is None:
            continue
        try:
            out.add(int(item["token_id"]))
        except (TypeError, ValueError):
            continue
    return out


def roles_by_id(items: list[dict[str, Any]] | None) -> dict[int, str]:
    out: dict[int, str] = {}
    for item in items or []:
        if item.get("token_id") is None:
            continue
        try:
            out[int(item["token_id"])] = str(item.get("role") or "")
        except (TypeError, ValueError):
            continue
    return out


def jaccard(left: set[int], right: set[int]) -> float | None:
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


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


def row_role(row: dict[str, Any]) -> str:
    if row.get("condition_type") == "candidate" and str(row.get("control_type")) == "none":
        return "candidate"
    control = str(row.get("control_type") or "")
    if control:
        return control
    return str(row.get("condition_type") or "unknown")


def load_phase885_rows(model: str) -> list[dict[str, Any]]:
    path = PHASE885_ROOT / f"phase885_{model}_rows.jsonl"
    if not path.exists():
        return []
    return p846.read_jsonl(path)


def pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        buckets[row_pair_key(row)][row_role(row)] = row
    out = []
    for key, by_role in buckets.items():
        if "candidate" not in by_role or "opposite_mode" not in by_role:
            continue
        candidate = by_role["candidate"]
        opposite = by_role["opposite_mode"]
        base_ids = token_ids(candidate.get("base_top_blockers"))
        candidate_ids = token_ids(candidate.get("intervened_top_blockers"))
        opposite_ids = token_ids(opposite.get("intervened_top_blockers"))
        candidate_removed = base_ids - candidate_ids
        opposite_removed = base_ids - opposite_ids
        candidate_residual = base_ids & candidate_ids
        opposite_residual = base_ids & opposite_ids
        base_roles = roles_by_id(candidate.get("base_top_blockers"))
        removed_overlap = candidate_removed & opposite_removed
        shared_removed_roles = Counter(base_roles.get(token_id, "unknown") for token_id in removed_overlap)
        scalar_sign_agreement = (
            sign(candidate.get("blocker_reduction")) == sign(opposite.get("blocker_reduction"))
            and sign(candidate.get("rank_improvement")) == sign(opposite.get("rank_improvement"))
        )
        class_logit_sign_agreement = sign(candidate.get("class_logit_delta")) == sign(opposite.get("class_logit_delta"))
        same_boundary_closure = bool(candidate.get("closure_from_open") and opposite.get("closure_from_open"))
        same_answer_gain = bool(candidate.get("answer_gain") and opposite.get("answer_gain"))
        removed_jaccard = jaccard(candidate_removed, opposite_removed)
        residual_jaccard = jaccard(candidate_residual, opposite_residual)
        random = by_role.get("same_layer_random")
        neighbor = by_role.get("neighbor_channel")
        out.append(
            {
                "phase": PHASE,
                "row_kind": "phase886_local_subspace_pair",
                "model": str(candidate.get("model") or ""),
                "parent_candidate_key": key[0],
                "case_id": key[1],
                "prompt_variant": key[2],
                "case_split": key[3],
                "eval_domain": key[4],
                "object": candidate.get("object"),
                "canonical_answer": candidate.get("canonical_answer"),
                "discovery_domain": candidate.get("discovery_domain"),
                "is_same_domain": bool(candidate.get("is_same_domain")),
                "candidate_key": candidate.get("candidate_key"),
                "candidate_mode": candidate.get("edit_mode"),
                "opposite_key": opposite.get("candidate_key"),
                "opposite_mode": opposite.get("edit_mode"),
                "base_blocker_count": candidate.get("base_class_blocker_count"),
                "base_blocker_ids": sorted(base_ids),
                "candidate_removed_ids": sorted(candidate_removed),
                "opposite_removed_ids": sorted(opposite_removed),
                "shared_removed_ids": sorted(removed_overlap),
                "candidate_residual_ids": sorted(candidate_residual),
                "opposite_residual_ids": sorted(opposite_residual),
                "base_blocker_topk": len(base_ids),
                "candidate_removed_count": len(candidate_removed),
                "opposite_removed_count": len(opposite_removed),
                "shared_removed_count": len(removed_overlap),
                "removed_jaccard": removed_jaccard,
                "residual_jaccard": residual_jaccard,
                "shared_removed_roles": counter_values(shared_removed_roles),
                "candidate_closure": bool(candidate.get("closure_from_open")),
                "opposite_closure": bool(opposite.get("closure_from_open")),
                "candidate_answer_gain": bool(candidate.get("answer_gain")),
                "opposite_answer_gain": bool(opposite.get("answer_gain")),
                "candidate_clean_like": bool(candidate.get("clean_like_closure")),
                "opposite_clean_like": bool(opposite.get("clean_like_closure")),
                "candidate_nonclean_like": bool(candidate.get("nonclean_like_closure")),
                "opposite_nonclean_like": bool(opposite.get("nonclean_like_closure")),
                "candidate_blocker_reduction": candidate.get("blocker_reduction"),
                "opposite_blocker_reduction": opposite.get("blocker_reduction"),
                "candidate_rank_improvement": candidate.get("rank_improvement"),
                "opposite_rank_improvement": opposite.get("rank_improvement"),
                "candidate_class_logit_delta": candidate.get("class_logit_delta"),
                "opposite_class_logit_delta": opposite.get("class_logit_delta"),
                "candidate_original_blocker_delta_mean": candidate.get("original_blocker_delta_mean"),
                "opposite_original_blocker_delta_mean": opposite.get("original_blocker_delta_mean"),
                "blocker_reduction_sign_agreement": scalar_sign_agreement,
                "class_logit_sign_agreement": class_logit_sign_agreement,
                "same_boundary_closure": same_boundary_closure,
                "same_answer_gain": same_answer_gain,
                "same_blocker_direction": bool(
                    removed_jaccard is not None
                    and removed_jaccard >= 0.5
                    and scalar_sign_agreement
                    and candidate_removed
                    and opposite_removed
                ),
                "candidate_only_closure": bool(candidate.get("closure_from_open") and not opposite.get("closure_from_open")),
                "opposite_only_closure": bool(opposite.get("closure_from_open") and not candidate.get("closure_from_open")),
                "random_closure": bool(random and random.get("closure_from_open")),
                "neighbor_closure": bool(neighbor and neighbor.get("closure_from_open")),
                "random_answer_gain": bool(random and random.get("answer_gain")),
                "neighbor_answer_gain": bool(neighbor and neighbor.get("answer_gain")),
            }
        )
    return out


def summarize_pairs(model: str, rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[str(pair.get("parent_candidate_key"))].append(pair)
    candidate_groups = []
    for key, vals in groups.items():
        n = len(vals)
        removed_vals = [finite(row.get("removed_jaccard")) for row in vals if row.get("removed_jaccard") is not None]
        residual_vals = [finite(row.get("residual_jaccard")) for row in vals if row.get("residual_jaccard") is not None]
        candidate_closure = sum(1 for row in vals if row.get("candidate_closure"))
        opposite_closure = sum(1 for row in vals if row.get("opposite_closure"))
        candidate_only = sum(1 for row in vals if row.get("candidate_only_closure"))
        opposite_only = sum(1 for row in vals if row.get("opposite_only_closure"))
        both = sum(1 for row in vals if row.get("same_boundary_closure"))
        random_closure = sum(1 for row in vals if row.get("random_closure"))
        neighbor_closure = sum(1 for row in vals if row.get("neighbor_closure"))
        same_blocker_direction = sum(1 for row in vals if row.get("same_blocker_direction"))
        sign_agree = sum(1 for row in vals if row.get("blocker_reduction_sign_agreement"))
        class_sign_agree = sum(1 for row in vals if row.get("class_logit_sign_agreement"))
        mean_removed_jaccard = mean(removed_vals) or 0.0
        mean_residual_jaccard = mean(residual_vals) or 0.0
        clean_split = (
            sum(1 for row in vals if row.get("candidate_clean_like")),
            sum(1 for row in vals if row.get("candidate_nonclean_like")),
            sum(1 for row in vals if row.get("opposite_clean_like")),
            sum(1 for row in vals if row.get("opposite_nonclean_like")),
        )
        shared_removed_roles_total: Counter[str] = Counter()
        for row in vals:
            shared_removed_roles_total.update(row.get("shared_removed_roles") or {})
        if candidate_closure == 0 and opposite_closure == 0:
            label = "negative_no_local_boundary"
        elif random_closure or neighbor_closure:
            label = "control_contaminated_local_boundary"
        elif both >= 2 and mean_removed_jaccard >= 0.5:
            label = "same_blocker_local_subspace_boundary"
        elif same_blocker_direction >= max(2, n // 4):
            label = "same_direction_subspace_candidate"
        elif candidate_only > opposite_only:
            label = "candidate_direction_specific_boundary"
        elif opposite_only > candidate_only:
            label = "opposite_direction_specific_boundary"
        else:
            label = "mixed_local_subspace_boundary"
        subspace_consistency = (
            (same_blocker_direction / max(1, n))
            + mean_removed_jaccard
            + (both / max(1, n))
            + (sign_agree / max(1, n))
        ) / 4.0
        opposite_mode_penalty = opposite_closure / max(1, n)
        direction_separation = abs(candidate_closure - opposite_closure) / max(1, n)
        clean_nonclean_balance = (clean_split[0] + clean_split[2] - clean_split[1] - clean_split[3]) / max(1, n)
        score = (
            2.0 * subspace_consistency
            + direction_separation
            + 0.5 * clean_nonclean_balance
            - 1.5 * ((random_closure + neighbor_closure) / max(1, n))
            - 0.5 * opposite_mode_penalty
        )
        candidate_groups.append(
            {
                "model": model,
                "parent_candidate_key": key,
                "n_pairs": n,
                "candidate_closure": candidate_closure,
                "opposite_closure": opposite_closure,
                "both_closure": both,
                "candidate_only_closure": candidate_only,
                "opposite_only_closure": opposite_only,
                "candidate_answer_gain": sum(1 for row in vals if row.get("candidate_answer_gain")),
                "opposite_answer_gain": sum(1 for row in vals if row.get("opposite_answer_gain")),
                "same_blocker_direction": same_blocker_direction,
                "blocker_reduction_sign_agreement": sign_agree,
                "class_logit_sign_agreement": class_sign_agree,
                "random_closure": random_closure,
                "neighbor_closure": neighbor_closure,
                "candidate_clean_like": clean_split[0],
                "candidate_nonclean_like": clean_split[1],
                "opposite_clean_like": clean_split[2],
                "opposite_nonclean_like": clean_split[3],
                "mean_removed_jaccard": mean_removed_jaccard,
                "mean_residual_jaccard": mean_residual_jaccard,
                "mean_candidate_blocker_reduction": mean(
                    [finite(row.get("candidate_blocker_reduction")) for row in vals]
                )
                or 0.0,
                "mean_opposite_blocker_reduction": mean(
                    [finite(row.get("opposite_blocker_reduction")) for row in vals]
                )
                or 0.0,
                "mean_candidate_rank_improvement": mean(
                    [finite(row.get("candidate_rank_improvement")) for row in vals]
                )
                or 0.0,
                "mean_opposite_rank_improvement": mean(
                    [finite(row.get("opposite_rank_improvement")) for row in vals]
                )
                or 0.0,
                "mean_candidate_class_logit_delta": mean(
                    [finite(row.get("candidate_class_logit_delta")) for row in vals]
                )
                or 0.0,
                "mean_opposite_class_logit_delta": mean(
                    [finite(row.get("opposite_class_logit_delta")) for row in vals]
                )
                or 0.0,
                "subspace_consistency": subspace_consistency,
                "opposite_mode_penalty": opposite_mode_penalty,
                "direction_separation": direction_separation,
                "clean_nonclean_balance": clean_nonclean_balance,
                "phase886_score": score,
                "evidence_label": label,
                "case_splits": counter_values(Counter(str(row.get("case_split")) for row in vals)),
                "eval_domains": counter_values(Counter(str(row.get("eval_domain")) for row in vals)),
                "shared_removed_role_counts": counter_values(shared_removed_roles_total),
                "objects_with_candidate_closure": sorted(
                    set(str(row.get("object")) for row in vals if row.get("candidate_closure"))
                ),
                "objects_with_opposite_closure": sorted(
                    set(str(row.get("object")) for row in vals if row.get("opposite_closure"))
                ),
                "prompts_with_candidate_closure": sorted(
                    set(str(row.get("prompt_variant")) for row in vals if row.get("candidate_closure"))
                ),
                "prompts_with_opposite_closure": sorted(
                    set(str(row.get("prompt_variant")) for row in vals if row.get("opposite_closure"))
                ),
            }
        )
    candidate_groups.sort(key=lambda row: finite(row.get("phase886_score")), reverse=True)
    return {
        "phase": PHASE,
        "model": model,
        "source_phase": 885,
        "source_rows": len(rows),
        "paired_rows": len(pairs),
        "candidate_groups": candidate_groups,
        "evidence_label_counts": counter_values(Counter(str(row.get("evidence_label")) for row in candidate_groups)),
        "overall": {
            "candidate_closure": sum(1 for row in pairs if row.get("candidate_closure")),
            "opposite_closure": sum(1 for row in pairs if row.get("opposite_closure")),
            "both_closure": sum(1 for row in pairs if row.get("same_boundary_closure")),
            "candidate_only_closure": sum(1 for row in pairs if row.get("candidate_only_closure")),
            "opposite_only_closure": sum(1 for row in pairs if row.get("opposite_only_closure")),
            "same_blocker_direction": sum(1 for row in pairs if row.get("same_blocker_direction")),
            "random_closure": sum(1 for row in pairs if row.get("random_closure")),
            "neighbor_closure": sum(1 for row in pairs if row.get("neighbor_closure")),
            "mean_removed_jaccard": mean(
                [finite(row.get("removed_jaccard")) for row in pairs if row.get("removed_jaccard") is not None]
            )
            or 0.0,
            "mean_residual_jaccard": mean(
                [finite(row.get("residual_jaccard")) for row in pairs if row.get("residual_jaccard") is not None]
            )
            or 0.0,
        },
    }


def write_model_outputs(model: str) -> dict[str, Any]:
    rows = load_phase885_rows(model)
    pairs = pair_rows(rows)
    summary = summarize_pairs(model, rows, pairs)
    out_dir = RESULT_ROOT / "local_subspace_decomposition"
    out_dir.mkdir(parents=True, exist_ok=True)
    p846.write_jsonl(out_dir / f"phase886_{model}_pairs.jsonl", pairs)
    p846.write_json(out_dir / f"phase886_{model}_summary.json", summary)
    log(f"{model}: source_rows={len(rows)} pairs={len(pairs)} groups={len(summary['candidate_groups'])}")
    return summary


def markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 886 local subspace boundary decomposition",
        "",
        "## Overall",
        "",
        f"- source_rows: {payload.get('source_rows')}",
        f"- paired_rows: {payload.get('paired_rows')}",
        f"- candidate_closure: {payload.get('overall', {}).get('candidate_closure')}",
        f"- opposite_closure: {payload.get('overall', {}).get('opposite_closure')}",
        f"- both_closure: {payload.get('overall', {}).get('both_closure')}",
        f"- same_blocker_direction: {payload.get('overall', {}).get('same_blocker_direction')}",
        f"- mean_removed_jaccard: {payload.get('overall', {}).get('mean_removed_jaccard')}",
        f"- random_closure: {payload.get('overall', {}).get('random_closure')}",
        f"- neighbor_closure: {payload.get('overall', {}).get('neighbor_closure')}",
        "",
        "## Candidate groups",
        "",
        "| model | candidate | label | pairs | cand | opp | both | same blocker | removed J | random | neighbor |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("candidate_groups", [])[:30]:
        lines.append(
            "| {model} | {key} | {label} | {n} | {cand} | {opp} | {both} | {same} | {jac:.3f} | {rnd} | {nei} |".format(
                model=row.get("model"),
                key=row.get("parent_candidate_key"),
                label=row.get("evidence_label"),
                n=row.get("n_pairs"),
                cand=row.get("candidate_closure"),
                opp=row.get("opposite_closure"),
                both=row.get("both_closure"),
                same=row.get("same_blocker_direction"),
                jac=finite(row.get("mean_removed_jaccard")),
                rnd=row.get("random_closure"),
                nei=row.get("neighbor_closure"),
            )
        )
    return "\n".join(lines) + "\n"


def summarize_cross_model() -> dict[str, Any]:
    out_dir = RESULT_ROOT / "local_subspace_decomposition"
    summaries = [read_json(out_dir / f"phase886_{model}_summary.json") for model in MODELS]
    summaries = [item for item in summaries if item]
    all_groups: list[dict[str, Any]] = []
    overall = Counter()
    shared_removed_roles_total: Counter[str] = Counter()
    source_rows = 0
    paired_rows = 0
    removed_jaccards: list[float] = []
    residual_jaccards: list[float] = []
    for summary in summaries:
        source_rows += int(summary.get("source_rows") or 0)
        paired_rows += int(summary.get("paired_rows") or 0)
        all_groups.extend(summary.get("candidate_groups") or [])
        for key, value in (summary.get("overall") or {}).items():
            if key.startswith("mean_"):
                continue
            overall[key] += int(value or 0)
        for group in summary.get("candidate_groups") or []:
            shared_removed_roles_total.update(group.get("shared_removed_role_counts") or {})
        ov = summary.get("overall") or {}
        if summary.get("paired_rows"):
            removed_jaccards.append(finite(ov.get("mean_removed_jaccard")))
            residual_jaccards.append(finite(ov.get("mean_residual_jaccard")))
    all_groups.sort(key=lambda row: finite(row.get("phase886_score")), reverse=True)
    payload = {
        "phase": PHASE,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_phase": 885,
        "source_rows": source_rows,
        "paired_rows": paired_rows,
        "models": [item.get("model") for item in summaries],
        "overall": {
            **{key: int(value) for key, value in sorted(overall.items())},
            "mean_removed_jaccard": mean(removed_jaccards) or 0.0,
            "mean_residual_jaccard": mean(residual_jaccards) or 0.0,
            "shared_removed_role_counts": counter_values(shared_removed_roles_total),
        },
        "evidence_label_counts": counter_values(Counter(str(row.get("evidence_label")) for row in all_groups)),
        "candidate_groups": all_groups,
        "top_groups": all_groups[:20],
    }
    p846.write_json(out_dir / "phase886_cross_model_summary.json", payload)
    (out_dir / "phase886_cross_model_summary.md").write_text(markdown_summary(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 886: decompose Phase885 stable candidates as local subspace boundary gears."
    )
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize:
        payload = summarize_cross_model()
        log(
            "cross_model: source_rows={source_rows} pairs={paired_rows} groups={groups}".format(
                source_rows=payload.get("source_rows"),
                paired_rows=payload.get("paired_rows"),
                groups=len(payload.get("candidate_groups") or []),
            )
        )
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize is set")
    write_model_outputs(args.model)


if __name__ == "__main__":
    main()
