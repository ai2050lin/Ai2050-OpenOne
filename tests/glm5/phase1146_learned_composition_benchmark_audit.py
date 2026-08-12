from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import phase1146_learned_composition_benchmark as primary


PHASE = 1146
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1146_learned_composition_benchmark"


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance)


def recompute_prediction_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for split in ["seen", "holdout", "quartet"]:
        selected = [row for row in rows if row["split"] == split]
        correct = np.asarray([bool(row["correct"]) for row in selected], dtype=np.float64)
        confidences = np.asarray([float(row["confidence"]) for row in selected], dtype=np.float64)
        per_field: dict[str, float] = {}
        for field in sorted({int(row["query_field"]) for row in selected}):
            values = [bool(row["correct"]) for row in selected if int(row["query_field"]) == field]
            per_field[str(field)] = float(np.mean(values))
        per_entity: dict[str, float] = {}
        for entity in sorted({int(row["query_entity"]) for row in selected}):
            values = [bool(row["correct"]) for row in selected if int(row["query_entity"]) == entity]
            per_entity[str(entity)] = float(np.mean(values))
        result[split] = {
            "case_count": len(selected),
            "accuracy": float(np.mean(correct)),
            "minimum_field_accuracy": float(min(per_field.values())),
            "minimum_entity_accuracy": float(min(per_entity.values())),
            "mean_confidence": float(np.mean(confidences)),
            "per_field_accuracy": per_field,
            "per_entity_accuracy": per_entity,
        }
    return result


def audit_run(model_name: str, prereg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = prereg["models"][model_name]
    run_dir = OUT_ROOT / "runs" / spec["split"] / model_name
    summary = read_json(run_dir / "summary.json")
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool) -> None:
        checks.append({"model": model_name, "name": name, "passed": bool(passed)})

    body = dict(summary)
    stored_summary_digest = body.pop("summary_digest")
    add("summary_digest", canonical_digest(body) == stored_summary_digest)
    add("protocol_digest", summary["protocol_digest"] == prereg["protocol_digest"])
    add("model_hash", file_sha256(ROOT / summary["model_path"]) == summary["model_sha256"])
    add(
        "predictions_hash",
        file_sha256(ROOT / summary["predictions_path"]) == summary["predictions_sha256"],
    )
    add("split", summary["split"] == spec["split"])
    add("lexicon", summary["lexicon"] == spec["lexicon"])
    add("no_nonfinite_steps", int(summary["nonfinite_steps"]) == 0)
    add("prediction_count", len(predictions) == 4096 + 4096 + 96)

    recomputed = recompute_prediction_metrics(predictions)
    metric_match = True
    for split, metrics in recomputed.items():
        stored = summary["evaluation"][split]
        metric_match &= int(stored["case_count"]) == int(metrics["case_count"])
        for name in ["accuracy", "minimum_field_accuracy", "minimum_entity_accuracy", "mean_confidence"]:
            metric_match &= close(stored[name], metrics[name], tolerance=2e-7 if name == "mean_confidence" else 1e-12)
        metric_match &= stored["per_field_accuracy"] == metrics["per_field_accuracy"]
        metric_match &= stored["per_entity_accuracy"] == metrics["per_entity_accuracy"]
    add("metrics_recomputed", metric_match)

    pair_sets = prereg["data"]["pairs"]
    lexicon = spec["lexicon"]
    seen = primary.make_dataset(
        4096, pair_sets["train"], int(spec["data_seeds"]["seen_evaluation"]), lexicon
    )
    holdout = primary.make_dataset(
        4096,
        pair_sets[spec["split"]],
        int(spec["data_seeds"]["holdout_evaluation"]),
        lexicon,
    )
    quartet, metadata = primary.make_quartets(
        pair_sets[spec["split"]], int(spec["data_seeds"]["quartet"]), lexicon
    )
    regenerated = {
        "seen": primary.array_digest(
            seen["inputs"], seen["targets"], seen["entities"], seen["fields"]
        ),
        "holdout": primary.array_digest(
            holdout["inputs"], holdout["targets"], holdout["entities"], holdout["fields"]
        ),
        "quartet": primary.array_digest(
            quartet["inputs"], quartet["targets"], quartet["entities"], quartet["fields"]
        ),
    }
    add("datasets_regenerated", regenerated == summary["dataset_digests"])
    add("quartet_metadata_count", len(metadata) == 96)
    recomputed_gates = primary.gate_metrics(summary["evaluation"], prereg["thresholds"])
    add("gates_recomputed", recomputed_gates == summary["gate_checks"])
    required = int(spec["training"]["required_consecutive_passes"])
    expected_qualified = all(recomputed_gates.values()) and int(summary["training_logs"][-1]["consecutive_passes"]) >= required
    add("qualification_recomputed", bool(summary["qualified"]) == bool(expected_qualified))
    detail = {
        "model": model_name,
        "summary_digest": stored_summary_digest,
        "qualified": summary["qualified"],
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
    }
    return checks, detail


def main() -> None:
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool) -> None:
        checks.append({"name": name, "passed": bool(passed)})

    body = dict(prereg)
    protocol_digest = body.pop("protocol_digest")
    add("protocol_digest", canonical_digest(body) == protocol_digest)
    add(
        "primary_hash",
        file_sha256(Path(primary.__file__).resolve()) == prereg["source_hashes"]["primary_script"],
    )
    protocol_audit = read_json(OUT_ROOT / "protocol" / "audit.json")
    add("protocol_audit", bool(protocol_audit["all_checks_passed"]))

    final_path = OUT_ROOT / "analysis" / "final.json"
    final = read_json(final_path) if final_path.exists() else None
    expected_models = list(prereg["selection"]["discovery_models"])
    discovery_selection = read_json(OUT_ROOT / "analysis" / "discovery_selection.json")
    if discovery_selection["confirmation_authorized"]:
        expected_models.extend(prereg["selection"]["confirmation_models"])
    details: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for model_name in expected_models:
        model_checks, detail = audit_run(model_name, prereg)
        checks.extend(model_checks)
        details.append(detail)
        spec = prereg["models"][model_name]
        summaries[model_name] = read_json(
            OUT_ROOT / "runs" / spec["split"] / model_name / "summary.json"
        )

    discovery_expected = all(
        summaries[name]["qualified"] for name in prereg["selection"]["discovery_models"]
    )
    add("discovery_selection", discovery_selection["all_qualified"] == discovery_expected)
    confirmation_path = OUT_ROOT / "analysis" / "confirmation_selection.json"
    if discovery_expected:
        confirmation = read_json(confirmation_path)
        confirmation_expected = all(
            summaries[name]["qualified"] for name in prereg["selection"]["confirmation_models"]
        )
        add("confirmation_selection", confirmation["all_qualified"] == confirmation_expected)
    else:
        confirmation_expected = False
        add("confirmation_not_run", not confirmation_path.exists())
    if final is not None:
        expected_mechanism = bool(discovery_expected and confirmation_expected)
        add("final_mechanism_scope", final["mechanism_phase_authorized"] == expected_mechanism)
        add("components_denied", not final["component_search_authorized"])
        add("natural_claim_denied", not final["natural_llm_claim_authorized"])

    report = {
        "phase": PHASE,
        "protocol_digest": protocol_digest,
        "check_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "all_checks_passed": all(check["passed"] for check in checks),
        "checks": checks,
        "details": details,
        "final_digest": final["final_digest"] if final else None,
    }
    report["audit_digest"] = canonical_digest(report)
    write_json(OUT_ROOT / "audit" / "independent_result_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
