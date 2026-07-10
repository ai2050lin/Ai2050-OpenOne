#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
PHASE = "Phase303"
SCHEMA_VERSION = "2.30.0"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return round(len(a & b) / len(a | b), 6) if (a | b) else 0.0


def clamp(x: float) -> float:
    return round(max(0.0, min(1.0, x)), 6)


def main() -> None:
    objects = read_jsonl(V2 / "phase301_semantic_object_rows.jsonl")
    behavior = read_jsonl(V2 / "phase302_semantic_behavior_rows.jsonl")
    readout = read_jsonl(V2 / "phase302_semantic_readout_rows.jsonl")
    object_map = {str(o["object_id"]): o for o in objects}
    profile = build_profiles(behavior, readout)
    object_rows = build_object_summary(objects, profile)
    reuse_rows, delta_rows = build_pair_rows(objects, profile)
    cluster_rows = build_clusters(object_rows, reuse_rows)
    path_rows = build_attribute_path_rows(behavior, readout)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete",
        "object_rows": len(objects),
        "object_summary_rows": len(object_rows),
        "reuse_matrix_rows": len(reuse_rows),
        "delta_matrix_rows": len(delta_rows),
        "attribute_path_rows": len(path_rows),
        "cluster_rows": len(cluster_rows),
        "behavior_rows": len(behavior),
        "readout_rows": len(readout),
        "mean_attribute_success_rate": mean_safe([safe_float(r.get("attribute_success_rate")) for r in object_rows]),
        "mean_measured_reuse_score": mean_safe([safe_float(r.get("measured_reuse_score")) for r in reuse_rows]),
        "mean_theoretical_reuse_score": mean_safe([safe_float(r.get("theoretical_reuse_score")) for r in reuse_rows]),
        "mean_delta_score": mean_safe([safe_float(r.get("delta_score")) for r in delta_rows]),
        "high_reuse_pair_count": sum(1 for r in reuse_rows if safe_float(r.get("combined_reuse_score")) >= 0.55),
        "high_delta_pair_count": sum(1 for r in delta_rows if safe_float(r.get("delta_score")) >= 0.65),
        "cluster_counts": dict(Counter(str(r.get("cluster_id")) for r in cluster_rows)),
        "progress": {
            "language_pattern_family_atlas": 0.81,
            "semantic_reuse_delta_subatlas": 0.34,
            "sample_type_coverage": 0.72,
            "large_data_feature_mining": 0.70,
            "physical_distribution_puzzle": 0.75,
            "mechanism_causal_audit": 0.52,
            "closure": 0.21,
        },
        "hard_limits": [
            "Phase303 is semantic behavior/readout graphing, not internal component localization.",
            "Reuse score is partly theory-guided by object feature tables.",
            "Small models may answer common fruit attributes with memorized stereotypes.",
            "Semantic object subatlas must later be connected to component paths.",
        ],
    }
    write_jsonl(V2 / "phase303_semantic_object_summary_rows.jsonl", object_rows)
    write_jsonl(V2 / "phase303_semantic_reuse_matrix_rows.jsonl", reuse_rows)
    write_jsonl(V2 / "phase303_semantic_delta_matrix_rows.jsonl", delta_rows)
    write_jsonl(V2 / "phase303_semantic_attribute_path_rows.jsonl", path_rows)
    write_jsonl(V2 / "phase303_semantic_family_cluster_rows.jsonl", cluster_rows)
    write_json(V2 / "phase303_semantic_reuse_delta_atlas_summary.json", summary)
    write_json(V2 / "progress.json", {**read_json(V2 / "progress.json"), **summary["progress"], "last_phase": PHASE, "updated_at": now()})
    update_manifest(summary)
    write_report(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_profiles(behavior: list[dict[str, Any]], readout: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    r_by_key = {(str(r.get("model")), str(r.get("case_id"))): r for r in readout}
    profile: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"attrs": {}, "success": [], "margins": [], "winners": []})
    for b in behavior:
        obj = str(b.get("object_id"))
        model = str(b.get("model"))
        key = (model, str(b.get("case_id")))
        attr = str(b.get("attribute_type"))
        r = r_by_key.get(key, {})
        bucket = profile[(model, obj)]
        bucket["attrs"][attr] = 1.0 if b.get("answer_correct_proxy") else 0.0
        bucket["success"].append(1.0 if b.get("answer_correct_proxy") else 0.0)
        bucket["margins"].append(safe_float(r.get("target_margin_vs_winner")))
        bucket["winners"].append(str(r.get("competition_winner")))
    return profile


def build_object_summary(objects: list[dict[str, Any]], profile: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for obj in objects:
        object_id = str(obj["object_id"])
        vals = [v for (model, oid), v in profile.items() if oid == object_id]
        successes = [x for v in vals for x in v["success"]]
        margins = [x for v in vals for x in v["margins"]]
        winners = [x for v in vals for x in v["winners"]]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "object_id": object_id,
                "object_label": obj.get("label"),
                "category_id": obj.get("category"),
                "subclass_id": obj.get("subclass"),
                "feature_count": len(obj.get("features") or []),
                "attribute_success_rate": mean_safe(successes),
                "mean_target_margin_vs_winner": mean_safe(margins),
                "competition_winner_counts": dict(Counter(winners)),
                "semantic_status": "measured_behavior_readout_only",
            }
        )
    return rows


def build_pair_rows(objects: list[dict[str, Any]], profile: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reuse_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for left, right in combinations(objects, 2):
        l_id, r_id = str(left["object_id"]), str(right["object_id"])
        theoretical_reuse = jaccard(set(left.get("features") or []), set(right.get("features") or []))
        measured_scores = []
        for model in sorted({m for (m, _oid) in profile}):
            lp = profile.get((model, l_id), {}).get("attrs", {})
            rp = profile.get((model, r_id), {}).get("attrs", {})
            attrs = sorted(set(lp) | set(rp))
            if attrs:
                same = sum(1 for a in attrs if safe_float(lp.get(a)) == safe_float(rp.get(a)))
                measured_scores.append(same / len(attrs))
        measured_reuse = mean_safe(measured_scores)
        combined = clamp(0.55 * theoretical_reuse + 0.45 * measured_reuse)
        delta = clamp(1.0 - combined)
        shared = sorted(set(left.get("features") or []) & set(right.get("features") or []))
        left_only = sorted(set(left.get("features") or []) - set(right.get("features") or []))
        right_only = sorted(set(right.get("features") or []) - set(left.get("features") or []))
        base = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "left_object_id": l_id,
            "right_object_id": r_id,
            "left_category_id": left.get("category"),
            "right_category_id": right.get("category"),
            "left_subclass_id": left.get("subclass"),
            "right_subclass_id": right.get("subclass"),
            "shared_feature_count": len(shared),
            "left_only_feature_count": len(left_only),
            "right_only_feature_count": len(right_only),
            "shared_features": shared[:20],
            "left_delta_features": left_only[:20],
            "right_delta_features": right_only[:20],
        }
        reuse_rows.append(
            {
                **base,
                "reuse_id": f"phase303:reuse:{l_id}:{r_id}",
                "theoretical_reuse_score": theoretical_reuse,
                "measured_reuse_score": measured_reuse,
                "combined_reuse_score": combined,
                "reuse_band": "high" if combined >= 0.55 else "medium" if combined >= 0.35 else "low",
            }
        )
        delta_rows.append(
            {
                **base,
                "delta_id": f"phase303:delta:{l_id}:{r_id}",
                "delta_score": delta,
                "delta_band": "high" if delta >= 0.65 else "medium" if delta >= 0.45 else "low",
            }
        )
    return reuse_rows, delta_rows


def build_clusters(object_rows: list[dict[str, Any]], reuse_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_category = defaultdict(list)
    for row in object_rows:
        by_category[str(row.get("category_id"))].append(row)
    for category, vals in sorted(by_category.items()):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "cluster_id": f"category:{category}",
                "cluster_type": "category",
                "member_count": len(vals),
                "members": [v["object_id"] for v in vals],
                "mean_attribute_success_rate": mean_safe([safe_float(v.get("attribute_success_rate")) for v in vals]),
            }
        )
    for subclass in sorted({str(r.get("left_subclass_id")) for r in reuse_rows} | {str(r.get("right_subclass_id")) for r in reuse_rows}):
        members = sorted(
            {
                str(r.get("left_object_id"))
                for r in reuse_rows
                if str(r.get("left_subclass_id")) == subclass
            }
            | {
                str(r.get("right_object_id"))
                for r in reuse_rows
                if str(r.get("right_subclass_id")) == subclass
            }
        )
        if len(members) >= 2:
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "cluster_id": f"subclass:{subclass}",
                    "cluster_type": "subclass",
                    "member_count": len(members),
                    "members": members,
                }
            )
    return rows


def build_attribute_path_rows(behavior: list[dict[str, Any]], readout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    r_by_key = {(str(r.get("model")), str(r.get("case_id"))): r for r in readout}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for b in behavior:
        groups[(str(b.get("model")), str(b.get("attribute_type")))].append(b)
    rows = []
    for (model, attr), vals in sorted(groups.items()):
        rvals = [r_by_key.get((model, str(v.get("case_id"))), {}) for v in vals]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "attribute_path_id": f"phase303:attribute_path:{model}:{attr}",
                "model": model,
                "attribute_type": attr,
                "rows": len(vals),
                "answer_correct_proxy_rate": mean_safe([1.0 if v.get("answer_correct_proxy") else 0.0 for v in vals]),
                "pattern_matched_proxy_rate": mean_safe([1.0 if v.get("pattern_matched_proxy") else 0.0 for v in vals]),
                "mean_target_margin_vs_winner": mean_safe([safe_float(r.get("target_margin_vs_winner")) for r in rvals]),
                "competition_winner_counts": dict(Counter(str(r.get("competition_winner")) for r in rvals)),
                "top_continue_channel_counts": dict(Counter(str(r.get("top_continue_channel")) for r in rvals)),
            }
        )
    return rows


def update_manifest(summary: dict[str, Any]) -> None:
    path = V2 / "manifest.json"
    manifest = read_json(path)
    manifest.setdefault("generated_files", [])
    for name in [
        "phase303_semantic_object_summary_rows.jsonl",
        "phase303_semantic_reuse_matrix_rows.jsonl",
        "phase303_semantic_delta_matrix_rows.jsonl",
        "phase303_semantic_attribute_path_rows.jsonl",
        "phase303_semantic_family_cluster_rows.jsonl",
        "phase303_semantic_reuse_delta_atlas_summary.json",
    ]:
        if name not in manifest["generated_files"]:
            manifest["generated_files"].append(name)
    manifest["last_phase"] = PHASE
    manifest["updated_at"] = now()
    manifest["phase303_summary"] = {
        "reuse_matrix_rows": summary["reuse_matrix_rows"],
        "delta_matrix_rows": summary["delta_matrix_rows"],
        "mean_measured_reuse_score": summary["mean_measured_reuse_score"],
    }
    write_json(path, manifest)


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase303 Semantic Reuse-Delta Atlas",
        "",
        f"- object_summary_rows: {summary['object_summary_rows']}",
        f"- reuse_matrix_rows: {summary['reuse_matrix_rows']}",
        f"- delta_matrix_rows: {summary['delta_matrix_rows']}",
        f"- attribute_path_rows: {summary['attribute_path_rows']}",
        f"- mean_attribute_success_rate: {summary['mean_attribute_success_rate']}",
        f"- mean_measured_reuse_score: {summary['mean_measured_reuse_score']}",
        f"- mean_theoretical_reuse_score: {summary['mean_theoretical_reuse_score']}",
        f"- mean_delta_score: {summary['mean_delta_score']}",
        f"- high_reuse_pair_count: {summary['high_reuse_pair_count']}",
        f"- high_delta_pair_count: {summary['high_delta_pair_count']}",
        "",
        "This is a semantic behavior/readout subatlas; it does not yet localize component paths.",
    ]
    (V2 / "phase303_semantic_reuse_delta_atlas_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
