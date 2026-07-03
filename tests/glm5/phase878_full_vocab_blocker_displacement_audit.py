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


PHASE = 878
RESULT_ROOT = Path("tests/result/phase878_full_vocab_blocker_displacement_audit")
DEFAULT_PHASE877_ROWS = Path(
    "tests/result/phase877_effective_transition_source_gate_audit/source_gate_phase876/phase877_source_gate_rows.jsonl"
)
DEFAULT_PHASE876_ROOT = Path("tests/result/phase876_nonclean_route_causal_validation/validation")


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path) if path.exists() else []


def token_key(item: dict[str, Any]) -> str:
    token_id = item.get("token_id")
    if token_id is not None:
        return f"id:{token_id}"
    return f"tok:{item.get('token')}"


def role_counts(tokens: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(item.get("role")) for item in tokens)


def token_set(tokens: list[dict[str, Any]]) -> set[str]:
    return {token_key(item) for item in tokens}


def blocker_tokens(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(row.get("blocker_class_top_blockers") or [])


def top_tokens(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(row.get("blocker_top_tokens") or row.get("top_tokens") or [])


def load_phase876_rows(root: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[tuple[str, str, str, str], dict[str, Any]]]:
    originals: dict[tuple[str, str, str], dict[str, Any]] = {}
    full_sets: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for path in sorted(root.glob("phase876_*_rows.jsonl")):
        for row in read_jsonl(path):
            model = str(row.get("model"))
            case_id = str(row.get("case_id"))
            prompt = str(row.get("prompt_variant"))
            if row.get("condition_type") == "original":
                originals[(model, case_id, prompt)] = row
            elif row.get("condition_type") == "full_set":
                full_sets[(model, case_id, prompt, str(row.get("candidate_key")))] = row
    return originals, full_sets


def compare_row(audit_row: dict[str, Any], base: dict[str, Any] | None, intervened: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(audit_row)
    out["raw_pair_found"] = base is not None and intervened is not None
    if base is None or intervened is None:
        return out

    base_blockers = blocker_tokens(base)
    int_blockers = blocker_tokens(intervened)
    base_top = top_tokens(base)
    int_top = top_tokens(intervened)
    base_blocker_keys = token_set(base_blockers)
    int_blocker_keys = token_set(int_blockers)
    base_top_keys = token_set(base_top)
    int_top_keys = token_set(int_top)
    removed = base_blocker_keys - int_blocker_keys
    added = int_blocker_keys - base_blocker_keys
    persistent = base_blocker_keys & int_blocker_keys
    top_removed = base_top_keys - int_top_keys
    top_added = int_top_keys - base_top_keys
    top_persistent = base_top_keys & int_top_keys
    base_roles = role_counts(base_top)
    int_roles = role_counts(int_top)
    role_delta = {role: int_roles.get(role, 0) - base_roles.get(role, 0) for role in sorted(set(base_roles) | set(int_roles))}

    base_rank = finite(base.get("blocker_class_best_target_rank"), default=999999.0)
    int_rank = finite(intervened.get("blocker_class_best_target_rank"), default=999999.0)
    base_logit = finite(base.get("blocker_class_best_target_logit"))
    int_logit = finite(intervened.get("blocker_class_best_target_logit"))
    base_count = finite(base.get("blocker_class_blocker_count"))
    int_count = finite(intervened.get("blocker_class_blocker_count"))

    out.update(
        {
            "base_blocker_count": base_count,
            "intervened_blocker_count": int_count,
            "blocker_count_reduction_raw": base_count - int_count,
            "base_target_rank": base_rank,
            "intervened_target_rank": int_rank,
            "target_rank_improvement": base_rank - int_rank,
            "base_target_logit": base_logit,
            "intervened_target_logit": int_logit,
            "target_logit_delta_raw": int_logit - base_logit,
            "base_top1_role_raw": (base_top[0].get("role") if base_top else None),
            "intervened_top1_role_raw": (int_top[0].get("role") if int_top else None),
            "base_top1_token_raw": (base_top[0].get("token") if base_top else None),
            "intervened_top1_token_raw": (int_top[0].get("token") if int_top else None),
            "base_blocker_tokens": base_blockers,
            "intervened_blocker_tokens": int_blockers,
            "removed_blocker_token_count": len(removed),
            "added_blocker_token_count": len(added),
            "persistent_blocker_token_count": len(persistent),
            "base_top_token_count": len(base_top),
            "intervened_top_token_count": len(int_top),
            "removed_top_token_count": len(top_removed),
            "added_top_token_count": len(top_added),
            "persistent_top_token_count": len(top_persistent),
            "top_token_overlap_ratio": len(top_persistent) / len(base_top_keys | int_top_keys) if (base_top_keys | int_top_keys) else None,
            "base_top_role_counts": dict(base_roles),
            "intervened_top_role_counts": dict(int_roles),
            "top_role_delta": role_delta,
            "blocker_set_changed": bool(removed or added),
            "top_set_changed": bool(top_removed or top_added),
            "target_rank_reached_top1": int_rank == 1,
            "count_reduced_without_original_blocker_reduction": bool(
                (base_count - int_count) > 0 and finite(out.get("original_blocker_delta")) >= 0
            ),
        }
    )
    return out


def grouped_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    out: dict[str, Any] = {}
    for name, items in sorted(groups.items()):
        out[name] = {
            "n": len(items),
            "transition_class_counts": dict(Counter(str(row.get("transition_class")) for row in items)),
            "route_counts": dict(Counter(str(row.get("primary_route")) for row in items)),
            "candidate_counts": dict(Counter(str(row.get("candidate_key")) for row in items)),
            "objects": sorted({str(row.get("object")) for row in items}),
            "prompts": sorted({str(row.get("prompt_variant")) for row in items}),
            "mean_blocker_count_reduction_raw": mean([finite(row.get("blocker_count_reduction_raw")) for row in items]),
            "mean_target_rank_improvement": mean([finite(row.get("target_rank_improvement")) for row in items]),
            "mean_target_logit_delta_raw": mean([finite(row.get("target_logit_delta_raw")) for row in items]),
            "mean_original_blocker_delta": mean([finite(row.get("original_blocker_delta")) for row in items]),
            "mean_top_token_overlap_ratio": mean(
                [finite(row.get("top_token_overlap_ratio")) for row in items if row.get("top_token_overlap_ratio") is not None]
            ),
            "blocker_set_changed": sum(1 for row in items if row.get("blocker_set_changed")),
            "top_set_changed": sum(1 for row in items if row.get("top_set_changed")),
            "count_reduced_without_original_blocker_reduction": sum(
                1 for row in items if row.get("count_reduced_without_original_blocker_reduction")
            ),
            "target_rank_reached_top1": sum(1 for row in items if row.get("target_rank_reached_top1")),
        }
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nonclean = [row for row in rows if row.get("transition_class") == "nonclean_output_transition"]
    clean = [row for row in rows if row.get("transition_class") == "clean_causal_transition"]
    return {
        "n_rows": len(rows),
        "n_pair_found": sum(1 for row in rows if row.get("raw_pair_found")),
        "transition_class_counts": dict(Counter(str(row.get("transition_class")) for row in rows)),
        "route_counts": dict(Counter(str(row.get("primary_route")) for row in rows)),
        "model_domain_counts": dict(Counter(f"{row.get('model')}:{row.get('domain')}" for row in rows)),
        "candidate_counts": dict(Counter(str(row.get("candidate_key")) for row in rows)),
        "nonclean_displacement": {
            "n": len(nonclean),
            "blocker_set_changed": sum(1 for row in nonclean if row.get("blocker_set_changed")),
            "top_set_changed": sum(1 for row in nonclean if row.get("top_set_changed")),
            "count_reduced_without_original_blocker_reduction": sum(
                1 for row in nonclean if row.get("count_reduced_without_original_blocker_reduction")
            ),
            "target_rank_reached_top1": sum(1 for row in nonclean if row.get("target_rank_reached_top1")),
            "mean_blocker_count_reduction_raw": mean([finite(row.get("blocker_count_reduction_raw")) for row in nonclean]),
            "mean_target_rank_improvement": mean([finite(row.get("target_rank_improvement")) for row in nonclean]),
            "mean_target_logit_delta_raw": mean([finite(row.get("target_logit_delta_raw")) for row in nonclean]),
            "mean_original_blocker_delta": mean([finite(row.get("original_blocker_delta")) for row in nonclean]),
        },
        "clean_reference": {
            "n": len(clean),
            "blocker_set_changed": sum(1 for row in clean if row.get("blocker_set_changed")),
            "top_set_changed": sum(1 for row in clean if row.get("top_set_changed")),
            "target_rank_reached_top1": sum(1 for row in clean if row.get("target_rank_reached_top1")),
            "mean_blocker_count_reduction_raw": mean([finite(row.get("blocker_count_reduction_raw")) for row in clean]),
            "mean_target_rank_improvement": mean([finite(row.get("target_rank_improvement")) for row in clean]),
            "mean_target_logit_delta_raw": mean([finite(row.get("target_logit_delta_raw")) for row in clean]),
            "mean_original_blocker_delta": mean([finite(row.get("original_blocker_delta")) for row in clean]),
        },
        "by_route": grouped_summary(rows, "primary_route"),
        "by_candidate": grouped_summary(rows, "candidate_key"),
        "rows": rows,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Phase 878 Full-Vocabulary Blocker Displacement Audit",
        "",
        "- Boundary: offline audit using saved Phase876 top-token/top-blocker rows; no new model run.",
        "- Goal: check whether original_blocker_not_reduced hides top blocker-field displacement.",
        "",
        "## Summary",
        "",
        f"- Rows: `{s['n_rows']}`",
        f"- Pair found: `{s['n_pair_found']}`",
        f"- Transition classes: `{s['transition_class_counts']}`",
        f"- Routes: `{s['route_counts']}`",
        f"- Nonclean displacement: `{s['nonclean_displacement']}`",
        f"- Clean reference: `{s['clean_reference']}`",
        "",
        "## By Route",
        "",
        "| route | n | candidates | objects | prompts | mean blocker red. | mean rank improve | mean target logit delta | mean orig blocker | blocker set changed | top set changed | target top1 |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for route, info in s["by_route"].items():
        lines.append(
            f"| `{route}` | {info['n']} | `{info['candidate_counts']}` | `{info['objects']}` | `{info['prompts']}` | "
            f"{finite(info['mean_blocker_count_reduction_raw']):.3f} | {finite(info['mean_target_rank_improvement']):.3f} | "
            f"{finite(info['mean_target_logit_delta_raw']):.3f} | {finite(info['mean_original_blocker_delta']):.3f} | "
            f"{info['blocker_set_changed']} | {info['top_set_changed']} | {info['target_rank_reached_top1']} |"
        )
    lines += [
        "",
        "## Rows",
        "",
        "| class | route | object | prompt | candidate | label | count red. | rank improve | logit delta | orig blocker | top1 raw | blocker changed | top changed | target top1 |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in s.get("rows") or []:
        lines.append(
            f"| `{row.get('transition_class')}` | `{row.get('primary_route')}` | {row.get('object')} | `{row.get('prompt_variant')}` | "
            f"`{row.get('candidate_key')}` | `{row.get('transition_label')}` | "
            f"{finite(row.get('blocker_count_reduction_raw')):.3f} | {finite(row.get('target_rank_improvement')):.3f} | "
            f"{finite(row.get('target_logit_delta_raw')):.3f} | {finite(row.get('original_blocker_delta')):.3f} | "
            f"`{row.get('base_top1_role_raw')}->{row.get('intervened_top1_role_raw')}` | "
            f"{row.get('blocker_set_changed')} | {row.get('top_set_changed')} | {row.get('target_rank_reached_top1')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase877-rows", default=str(DEFAULT_PHASE877_ROWS))
    parser.add_argument("--phase876-root", default=str(DEFAULT_PHASE876_ROOT))
    parser.add_argument("--output-round", default="source_gate_phase876")
    args = parser.parse_args()

    audit_rows = read_jsonl(Path(args.phase877_rows))
    originals, full_sets = load_phase876_rows(Path(args.phase876_root))
    compared: list[dict[str, Any]] = []
    for row in audit_rows:
        base = originals.get((str(row.get("model")), str(row.get("case_id")), str(row.get("prompt_variant"))))
        intervened = full_sets.get(
            (str(row.get("model")), str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("candidate_key")))
        )
        compared.append(compare_row(row, base, intervened))

    payload = {
        "phase": PHASE,
        "title": "Full-Vocabulary Blocker Displacement Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase877_rows": str(args.phase877_rows),
        "phase876_root": str(args.phase876_root),
        "summary": summarize(compared),
        "boundary": "Saved top-token/top-blocker displacement audit; no new model run and no closure claim.",
    }
    out_dir = RESULT_ROOT / str(args.output_round)
    p846.write_json(out_dir / "phase878_summary.json", payload)
    p846.write_jsonl(out_dir / "phase878_displacement_rows.jsonl", compared)
    write_markdown(out_dir / "phase878_summary.md", payload)
    printable = dict(payload["summary"])
    printable.pop("rows", None)
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
