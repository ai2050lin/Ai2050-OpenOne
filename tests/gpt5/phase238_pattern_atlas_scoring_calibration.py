#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE = 238
SOURCE_PHASE = 236
SCHEMA_VERSION = "1.0.0"
INPUT_ROOT = Path("tests/result/phase236_pattern_family_behavior_benchmark/pattern_family_behavior_benchmark")
RESULT_ROOT = Path("tests/result/phase238_pattern_atlas_scoring_calibration")
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
MODELS = ["qwen3", "glm4", "deepseek7b"]


ALIASES_BY_TARGET = {
    "red": ["red", "reddish"],
    "yellow": ["yellow", "golden"],
    "green": ["green"],
    "white": ["white"],
    "black": ["black"],
    "sour": ["sour", "tart", "acidic", "citrusy"],
    "hit": ["hit", "strike", "striking", "drive", "driving", "pound", "pounding", "hammering"],
    "car": ["car", "vehicle", "automobile", "auto"],
    "no": ["no", "not", "false"],
    "blue": ["blue"],
    "红": ["红", "红色", "red"],
    "雪": ["雪", "snow"],
}

RELATION_SCHEMAS = {
    "color": {
        "relation_type": "attribute_color",
        "answer_policy": "alias_match",
        "ambiguity_risk": "low",
    },
    "taste": {
        "relation_type": "sensory_attribute",
        "answer_policy": "alias_match",
        "ambiguity_risk": "medium",
    },
    "function": {
        "relation_type": "affordance_or_action",
        "answer_policy": "alias_or_phrase_match",
        "ambiguity_risk": "high",
    },
    "part_of": {
        "relation_type": "part_whole_relation",
        "answer_policy": "alias_or_hypernym_match",
        "ambiguity_risk": "high",
    },
    "special": {
        "relation_type": "task_specific",
        "answer_policy": "alias_match",
        "ambiguity_risk": "medium",
    },
}

AMBIGUOUS_RELATIONS = {"function", "part_of", "taste"}


def utc_now() -> str:
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


def append_unique_jsonl(path: Path, new_rows: list[dict[str, Any]], id_key: str) -> None:
    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows + new_rows:
        key = str(row.get(id_key) or row.get("metric_id") or row.get("edge_id") or row.get("observation_id") or len(by_id))
        by_id[key] = row
    write_jsonl(path, list(by_id.values()))


def words(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


def contains_alias(output: str, aliases: list[str]) -> tuple[bool, str]:
    low = output.lower()
    word_set = words(output)
    for alias in aliases:
        a = alias.lower()
        if re.search(r"[\u4e00-\u9fff]", a):
            if a in output:
                return True, alias
        elif " " in a:
            if a in low:
                return True, alias
        elif a in word_set or a in low:
            return True, alias
    return False, ""


def infer_relation(row: dict[str, Any]) -> str:
    if row.get("relation"):
        return str(row["relation"])
    prompt = str(row.get("prompt") or "").lower()
    if "color" in prompt:
        return "color"
    if "taste" in prompt:
        return "taste"
    if "function" in prompt:
        return "function"
    if "part_of" in prompt or "part of" in prompt:
        return "part_of"
    return "special"


def case_alias_row(row: dict[str, Any]) -> dict[str, Any]:
    target = str(row.get("target") or "")
    relation = infer_relation(row)
    schema = RELATION_SCHEMAS.get(relation, RELATION_SCHEMAS["special"])
    aliases = ALIASES_BY_TARGET.get(target, [target])
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase238",
        "created_at": utc_now(),
        "case_id": row["case_id"],
        "family_id": row["family_id"],
        "mode_id": row["mode_id"],
        "target": target,
        "target_aliases": aliases,
        "acceptable_answers": aliases,
        "relation": relation,
        "relation_schema": schema["relation_type"],
        "answer_policy": schema["answer_policy"],
        "ambiguity_risk": schema["ambiguity_risk"],
    }


def protocol_match(row: dict[str, Any], output: str) -> bool:
    expected = str(row.get("expected_pattern") or "")
    token_count = int(row.get("output_token_count") or len(output.split()))
    low = output.lower()
    if expected in {"short", "short_answer"}:
        return token_count <= 3
    if "explain" in expected or "because" in expected:
        return "because" in low or "因为" in output
    if "repeat" in expected:
        return "," in output or "，" in output
    if "list" in expected:
        return "\n" in output or "," in output or "1." in output or "-" in output
    return bool(row.get("pattern_match"))


def closure_signal(row: dict[str, Any], output: str) -> bool:
    expected = str(row.get("expected_pattern") or "")
    token_count = int(row.get("output_token_count") or len(output.split()))
    if expected in {"short", "short_answer"}:
        return token_count <= 3
    if token_count <= 24 and "Answer:" not in output[1:]:
        return True
    return False


def calibrated_score(row: dict[str, Any], alias_row: dict[str, Any]) -> dict[str, Any]:
    output = str(row.get("output_text") or "")
    target = str(row.get("target") or "")
    aliases = list(alias_row["target_aliases"])
    target_hit, _ = contains_alias(output, [target])
    alias_hit, matched_alias = contains_alias(output, aliases)
    protocol_ok = protocol_match(row, output)
    closure_ok = closure_signal(row, output)
    semantic_equivalent = alias_hit
    semantic_correct_but_target_mismatch = semantic_equivalent and not target_hit
    ambiguous = alias_row["ambiguity_risk"] in {"medium", "high"} and semantic_correct_but_target_mismatch
    score = 0.35 * float(alias_hit) + 0.25 * float(protocol_ok) + 0.25 * float(semantic_equivalent) + 0.15 * float(closure_ok)
    if alias_hit and not protocol_ok:
        drift_type = "protocol_or_over_generation"
    elif semantic_correct_but_target_mismatch:
        drift_type = "semantic_correct_but_target_mismatch"
    elif not alias_hit:
        drift_type = "semantic_or_target_failure"
    elif not closure_ok:
        drift_type = "closure_or_rollout_failure"
    else:
        drift_type = "none"
    return {
        "calibrated_behavior_score": round(score, 4),
        "answer_hit": alias_hit,
        "target_literal_hit": target_hit,
        "matched_alias": matched_alias,
        "protocol_match_calibrated": protocol_ok,
        "semantic_equivalent": semantic_equivalent,
        "semantic_correct_but_target_mismatch": semantic_correct_but_target_mismatch,
        "ambiguous_target_or_relation": ambiguous,
        "closure_signal": closure_ok,
        "calibrated_drift_type": drift_type,
    }


def observation_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase238",
        "created_at": utc_now(),
        "run_id": f"phase238:{row['model']}:scoring_calibration",
        "case_id": row["case_id"],
        "model": row["model"],
        "family_id": row["family_id"],
        "mode_id": row["mode_id"],
    }
    fields = {
        "calibrated_behavior_score": row["calibrated_behavior_score"],
        "semantic_equivalent": float(row["semantic_equivalent"]),
        "semantic_correct_but_target_mismatch": float(row["semantic_correct_but_target_mismatch"]),
        "ambiguous_target_or_relation": float(row["ambiguous_target_or_relation"]),
        "protocol_match_calibrated": float(row["protocol_match_calibrated"]),
        "closure_signal": float(row["closure_signal"]),
    }
    return [
        {
            **common,
            "observation_id": f"phase238:{row['model']}:{row['case_id']}:{name}",
            "level": "behavior_calibration",
            "metric_name": name,
            "metric_value": value,
            "metric_unit": "score",
            "calibrated_drift_type": row["calibrated_drift_type"],
            "matched_alias": row["matched_alias"],
        }
        for name, value in fields.items()
    ]


def summarize_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    now = utc_now()
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["model"], row["family_id"], row["mode_id"])].append(row)
        buckets[(row["model"], row["family_id"], "*")].append(row)
        buckets[("cross_model", row["family_id"], "*")].append(row)
    for (model, family_id, mode_id), items in sorted(buckets.items()):
        scope = "family" if mode_id == "*" else "mode"
        metric_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase238",
                "created_at": now,
                "metric_id": f"phase238:{model}:{scope}:{family_id}:{mode_id}:calibrated_behavior",
                "scope": scope,
                "model": model,
                "family_id": family_id,
                "mode_id": "" if mode_id == "*" else mode_id,
                "metric_name": "mean_calibrated_behavior_score",
                "metric_value": round(sum(float(x["calibrated_behavior_score"]) for x in items) / len(items), 4),
                "case_count": len(items),
                "semantic_equivalent_rate": round(sum(1 for x in items if x["semantic_equivalent"]) / len(items), 4),
                "ambiguous_rate": round(sum(1 for x in items if x["ambiguous_target_or_relation"]) / len(items), 4),
                "semantic_mismatch_rate": round(sum(1 for x in items if x["semantic_correct_but_target_mismatch"]) / len(items), 4),
                "calibrated_drift_types": dict(Counter(str(x["calibrated_drift_type"]) for x in items).most_common()),
            }
        )
    return metric_rows


def stable_failure_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    out = []
    for case_id, items in sorted(grouped.items()):
        models = sorted(set(str(x["model"]) for x in items))
        low = [x for x in items if float(x["calibrated_behavior_score"]) < 0.5 and not x["ambiguous_target_or_relation"]]
        protocol_fail = [x for x in items if x["calibrated_drift_type"] == "protocol_or_over_generation"]
        if len(low) >= 2 or len(protocol_fail) >= 2:
            first = items[0]
            out.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase238",
                    "created_at": utc_now(),
                    "candidate_id": f"phase238:stable_failure:{case_id}",
                    "case_id": case_id,
                    "family_id": first["family_id"],
                    "mode_id": first["mode_id"],
                    "models": models,
                    "low_score_models": [x["model"] for x in low],
                    "protocol_fail_models": [x["model"] for x in protocol_fail],
                    "mean_calibrated_behavior_score": round(sum(float(x["calibrated_behavior_score"]) for x in items) / len(items), 4),
                    "failure_type": "stable_protocol_failure" if len(protocol_fail) >= 2 else "stable_semantic_or_target_failure",
                    "next_phase_recommendation": "prompt_trigger_anchor_ablation",
                }
            )
    return out


def update_atlas(summary: dict[str, Any], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], aliases: list[dict[str, Any]], flags: list[dict[str, Any]], stable_failures: list[dict[str, Any]]) -> None:
    manifest = read_json(ATLAS_ROOT / "manifest.json")
    if manifest:
        files = manifest.setdefault("files", {})
        files["case_aliases"] = "case_aliases.jsonl"
        files["semantic_equivalence_flags"] = "semantic_equivalence_flags.jsonl"
        files["stable_failure_candidates"] = "stable_failure_candidates.jsonl"
        files["ambiguous_case_report"] = "ambiguous_case_report.md"
        manifest["phase"] = "Phase238"
        manifest["created_at"] = utc_now()
        write_json(ATLAS_ROOT / "manifest.json", manifest)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    write_jsonl(ATLAS_ROOT / "case_aliases.jsonl", aliases)
    write_jsonl(ATLAS_ROOT / "semantic_equivalence_flags.jsonl", flags)
    write_jsonl(ATLAS_ROOT / "stable_failure_candidates.jsonl", stable_failures)
    progress = read_json(ATLAS_ROOT / "progress.json")
    if progress:
        progress["phase_id"] = "Phase238"
        progress["created_at"] = utc_now()
        progress.setdefault("global_progress", {})["pattern_family_atlas"] = 0.48
        progress.setdefault("global_progress", {})["general_language_mechanism_confidence"] = 0.46
        progress.setdefault("levels", {})["behavior"] = 0.54
        progress["next_phase"] = "Phase239_prompt_trigger_anchor_ablation"
        progress["latest_phase"] = {
            "phase_id": "Phase238",
            "title": "行为评分校准与歧义样例标记",
            "case_rows": summary["case_rows"],
            "calibrated_observation_rows": summary["calibrated_observation_rows"],
            "mean_calibrated_behavior_score": summary["mean_calibrated_behavior_score"],
            "ambiguous_rows": summary["ambiguous_rows"],
            "stable_failure_candidates": summary["stable_failure_candidates"],
        }
        write_json(ATLAS_ROOT / "progress.json", progress)
    summary_path = ATLAS_ROOT / "summary.md"
    old = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    marker = "## Phase238 Scoring Calibration Update"
    if marker in old:
        old = old.split(marker, 1)[0].rstrip()
    addition = (
        f"\n{marker}\n\n"
        f"- case_rows: {summary['case_rows']}\n"
        f"- calibrated_observation_rows: {summary['calibrated_observation_rows']}\n"
        f"- mean_original_behavior_score: {summary['mean_original_behavior_score']}\n"
        f"- mean_calibrated_behavior_score: {summary['mean_calibrated_behavior_score']}\n"
        f"- ambiguous_rows: {summary['ambiguous_rows']}\n"
        f"- semantic_mismatch_rows: {summary['semantic_mismatch_rows']}\n"
        f"- stable_failure_candidates: {summary['stable_failure_candidates']}\n"
    )
    summary_path.write_text(old.rstrip() + "\n" + addition, encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], stable_failures: list[dict[str, Any]]) -> None:
    ambiguous = [r for r in rows if r["ambiguous_target_or_relation"] or r["semantic_correct_but_target_mismatch"]]
    lines = ["# Phase238 Ambiguous Case Report", ""]
    lines.append(f"case_rows: {summary['case_rows']}")
    lines.append(f"mean_original_behavior_score: {summary['mean_original_behavior_score']}")
    lines.append(f"mean_calibrated_behavior_score: {summary['mean_calibrated_behavior_score']}")
    lines.append(f"ambiguous_rows: {summary['ambiguous_rows']}")
    lines.append(f"semantic_mismatch_rows: {summary['semantic_mismatch_rows']}")
    lines.append(f"stable_failure_candidates: {summary['stable_failure_candidates']}")
    lines.extend(["", "## Ambiguous / Semantic Mismatch Rows", "", "| model | case | family | target | alias | score | calibrated | drift | output |", "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |"])
    for row in ambiguous[:80]:
        output = str(row.get("output_text") or "").replace("\n", " ")[:100]
        lines.append(
            f"| {row['model']} | {row['case_id']} | {row['family_id']} | {row['target']} | {row['matched_alias']} | "
            f"{float(row.get('behavior_score') or 0):.2f} | {float(row['calibrated_behavior_score']):.2f} | {row['calibrated_drift_type']} | {output} |"
        )
    lines.extend(["", "## Stable Failure Candidates", "", "| case | family | mode | type | models | mean score |", "| --- | --- | --- | --- | --- | ---: |"])
    for row in stable_failures[:80]:
        lines.append(
            f"| {row['case_id']} | {row['family_id']} | {row['mode_id']} | {row['failure_type']} | "
            f"{','.join(row['models'])} | {row['mean_calibrated_behavior_score']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    source_rows = read_jsonl(INPUT_ROOT / "phase236_cross_model_case_rows.jsonl")
    if not source_rows:
        raise FileNotFoundError("Missing phase236_cross_model_case_rows.jsonl")
    alias_by_case: dict[str, dict[str, Any]] = {}
    calibrated_rows: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for row in source_rows:
        cid = str(row["case_id"])
        alias = alias_by_case.get(cid)
        if alias is None:
            alias = case_alias_row(row)
            alias_by_case[cid] = alias
        cal = calibrated_score(row, alias)
        new_row = {
            **row,
            "phase": PHASE,
            "source_phase": SOURCE_PHASE,
            "phase236_behavior_score": row.get("behavior_score"),
            **alias,
            **cal,
        }
        calibrated_rows.append(new_row)
        flags.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase238",
                "created_at": utc_now(),
                "flag_id": f"phase238:{row['model']}:{cid}:semantic_equivalence",
                "case_id": cid,
                "model": row["model"],
                "family_id": row["family_id"],
                "mode_id": row["mode_id"],
                "semantic_equivalent": cal["semantic_equivalent"],
                "semantic_correct_but_target_mismatch": cal["semantic_correct_but_target_mismatch"],
                "ambiguous_target_or_relation": cal["ambiguous_target_or_relation"],
                "matched_alias": cal["matched_alias"],
                "calibrated_drift_type": cal["calibrated_drift_type"],
            }
        )
        observations.extend(observation_rows(new_row))
    aliases = list(alias_by_case.values())
    metrics = summarize_metrics(calibrated_rows)
    stable_failures = stable_failure_candidates(calibrated_rows)
    summary = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Pattern atlas scoring calibration",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "case_rows": len(calibrated_rows),
        "unique_cases": len(aliases),
        "calibrated_observation_rows": len(observations),
        "metric_rows": len(metrics),
        "mean_original_behavior_score": round(sum(float(x.get("behavior_score") or 0) for x in source_rows) / len(source_rows), 4),
        "mean_calibrated_behavior_score": round(sum(float(x["calibrated_behavior_score"]) for x in calibrated_rows) / len(calibrated_rows), 4),
        "ambiguous_rows": sum(1 for x in calibrated_rows if x["ambiguous_target_or_relation"]),
        "semantic_mismatch_rows": sum(1 for x in calibrated_rows if x["semantic_correct_but_target_mismatch"]),
        "stable_failure_candidates": len(stable_failures),
        "calibrated_drift_types": dict(Counter(str(x["calibrated_drift_type"]) for x in calibrated_rows).most_common()),
        "family_counts": dict(Counter(str(x["family_id"]) for x in calibrated_rows).most_common()),
    }
    write_json(out_dir / "phase238_scoring_calibration_summary.json", summary)
    write_jsonl(out_dir / "phase238_case_aliases.jsonl", aliases)
    write_jsonl(out_dir / "phase238_calibrated_case_rows.jsonl", calibrated_rows)
    write_jsonl(out_dir / "phase238_semantic_equivalence_flags.jsonl", flags)
    write_jsonl(out_dir / "phase238_calibrated_observations.jsonl", observations)
    write_jsonl(out_dir / "phase238_calibrated_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase238_stable_failure_candidates.jsonl", stable_failures)
    write_report(out_dir / "phase238_ambiguous_case_report.md", summary, calibrated_rows, stable_failures)
    update_atlas(summary, observations, metrics, aliases, flags, stable_failures)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase238 pattern atlas scoring calibration")
    parser.add_argument("--round-name", default="pattern_atlas_scoring_calibration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(args.round_name)


if __name__ == "__main__":
    main()
