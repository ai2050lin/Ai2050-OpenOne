from __future__ import annotations

import gzip
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE991 = ROOT / "tests/glm5/result/phase991_delayed_binding_gpu_admission"
PHASE992 = ROOT / "tests/glm5/result/phase992_delayed_binding_behavior_execution"
PHASE993 = ROOT / "tests/glm5/result/phase993_delayed_binding_emission_topology"
RESULT_PATH = PHASE993 / "phase993_emission_topology.json"
AUDIT_PATH = PHASE993 / "phase993_emission_topology_audit.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("discovery", "confirmation", "adversarial")
VALUES = ("red", "blue", "green", "black")
MARKER_RE = re.compile(r"(?<![A-Za-z])(red|blue|green|black)(?![A-Za-z])", re.I)


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_truth() -> dict[str, dict]:
    result: dict[str, dict] = {}
    for split in SPLITS:
        with (PHASE991 / f"scoring_truth/private/{split}.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["record_id"] in result:
                    raise RuntimeError("duplicate truth identity")
                result[row["record_id"]] = row
    return result


def verify_self_hash(document: dict, field: str) -> bool:
    expected = document[field]
    unsigned = dict(document)
    unsigned.pop(field)
    return expected == sha256_bytes(canonical_bytes(unsigned))


def main() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    score_path = PHASE992 / "scores/public_score.json"
    score = json.loads(score_path.read_text(encoding="utf-8"))
    truth = load_truth()
    checks: dict[str, bool] = {
        "result_self_hash": verify_self_hash(result, "artifact_sha256"),
        "result_phase_and_role": result.get("phase") == 993
        and result.get("role") == "posthoc_external_emission_topology_after_verified_public_failure",
        "source_score_hash": result["source_seals"]["phase992_score_file_sha256"] == sha256_file(score_path),
        "source_score_terminal_failure": score.get("passed") is False,
        "truth_count": len(truth) == 8192,
        "holdout_raw_absent": not (PHASE992 / "raw/holdout").exists(),
        "holdout_score_absent": not (PHASE992 / "scores/holdout_score.json").exists(),
        "scope_claims_conservative": result["scope_guards"]["internal_trace_read"] is False
        and result["scope_guards"]["causal_intervention_performed"] is False,
    }
    per_model: dict[str, dict] = {}
    for model in MODELS:
        raw_path = PHASE992 / f"raw/primary/{model}.jsonl.gz"
        counts = {
            "rows": 0,
            "parsed": 0,
            "correct": 0,
            "budget_exhausted": 0,
            "eos_seen": 0,
            "tokens_24": 0,
            "tf_strict_correct": 0,
        }
        identities: set[str] = set()
        with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                gold = truth[row["record_id"]]
                identities.add(row["record_id"])
                matches = [match.casefold() for match in MARKER_RE.findall(row["generated_text"])]
                first = matches[0] if matches else None
                logits = {name: float(row["teacher_forced_candidates"][name]["logit"]) for name in VALUES}
                counts["rows"] += 1
                counts["parsed"] += int(first is not None)
                counts["correct"] += int(first == gold["gold_value"])
                counts["budget_exhausted"] += int(bool(row["budget_exhausted"]))
                counts["eos_seen"] += int(bool(row["eos_seen"]))
                counts["tokens_24"] += int(len(row["generated_suffix_token_ids"]) == 24)
                counts["tf_strict_correct"] += int(
                    logits[gold["gold_value"]] > max(value for key, value in logits.items() if key != gold["gold_value"])
                )
        reported = result["models"][model]
        model_checks = {
            "raw_hash": reported["raw_sha256"] == sha256_file(raw_path),
            "identity_set": identities == set(truth),
            "row_count": reported["row_count"] == counts["rows"] == 8192,
            "parsed": reported["natural_marker"]["parsed"] == counts["parsed"],
            "correct": reported["natural_marker"]["correct"] == counts["correct"],
            "budget": reported["termination"]["budget_exhausted"] == counts["budget_exhausted"] == 8192,
            "eos": reported["termination"]["eos_seen"] == counts["eos_seen"] == 0,
            "token_length": counts["tokens_24"] == 8192,
            "teacher_forced": reported["teacher_forced_vs_natural"]["teacher_forced_strict_correct"]
            == counts["tf_strict_correct"],
            "published_natural": score["models"][model]["natural_generation"]["overall"]["correct"]
            == counts["correct"],
            "published_teacher": score["models"][model]["teacher_forced_diagnostic"]["positive"]
            == counts["tf_strict_correct"],
        }
        per_model[model] = {"checks": model_checks, "passed": all(model_checks.values()), "counts": counts}
    checks["all_models"] = all(item["passed"] for item in per_model.values())

    audit = {
        "schema_version": "phase993_delayed_binding_emission_topology_audit.v1",
        "phase": 993,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "independent_minimal_recomputation_of_external_emission_topology",
        "result_file_sha256": sha256_file(RESULT_PATH),
        "audit_source_sha256": sha256_file(Path(__file__)),
        "checks": checks,
        "models": per_model,
        "passed": all(checks.values()),
        "scientific_scope": {
            "posthoc_external_behavior_only": True,
            "internal_structure_evidence": False,
            "causal_mechanism_evidence": False,
            "holdout_authorized": False,
        },
    }
    audit["audit_sha256"] = sha256_bytes(canonical_bytes(audit))
    AUDIT_PATH.write_bytes(canonical_bytes(audit))
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
