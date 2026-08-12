#!/usr/bin/env python3
"""Strictly audit Phase1065-1067 protocols and serialized results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "tests" / "glm5" / "result"
PHASE_ROOTS = {
    1065: RESULT_ROOT / "phase1065_multimode_response_atlas",
    1066: RESULT_ROOT / "phase1066_reasoning_role_causal",
    1067: RESULT_ROOT / "phase1067_reasoning_necessity_coalition",
}
MODELS = ("qwen3", "glm4", "deepseek7b")
OUTPUT = (
    PHASE_ROOTS[1067] / "analysis" / "integrity_audit.json"
)


def reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_nonfinite,
    )


def read_jsonl(path: Path) -> list[Any]:
    return [
        json.loads(line, parse_constant=reject_nonfinite)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_digest(value: Any) -> str:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_precision(summary: dict[str, Any]) -> None:
    precision = summary["precision"]
    if (
        not precision["has_fp16_parameters"]
        or precision["has_bf16_parameters"]
        or precision["has_quantized_modules"]
    ):
        raise RuntimeError(
            f"FP16/no-quantization audit failed: {summary['model']}"
        )


def audit_phase(phase: int, root: Path) -> dict[str, Any]:
    json_files = sorted(
        path for path in root.rglob("*.json") if path != OUTPUT
    )
    jsonl_files = sorted(root.rglob("*.jsonl"))
    for path in json_files:
        read_json(path)
    jsonl_rows = sum(len(read_jsonl(path)) for path in jsonl_files)

    preregistration = read_json(
        root / "protocol" / "preregistration.json"
    )
    claimed_digest = preregistration["protocol_digest"]
    digest_material = dict(preregistration)
    digest_material.pop("protocol_digest")
    recomputed_digest = canonical_digest(digest_material)
    if recomputed_digest != claimed_digest:
        raise RuntimeError(f"Phase{phase} protocol digest mismatch")

    aggregate = read_json(root / "aggregate.json")
    if aggregate["protocol_digest"] != claimed_digest:
        raise RuntimeError(f"Phase{phase} aggregate digest mismatch")

    summaries = {
        model: read_json(root / "atlas" / model / "summary.json")
        for model in MODELS
    }
    for model, summary in summaries.items():
        if summary["protocol_digest"] != claimed_digest:
            raise RuntimeError(
                f"Phase{phase} {model} protocol digest mismatch"
            )
        assert_precision(summary)

    integrity: dict[str, Any]
    if phase == 1065:
        if any(summary["case_count"] != 1440 for summary in summaries.values()):
            raise RuntimeError("Phase1065 case count mismatch")
        if any(summary["identity_maximum"] != 0.0 for summary in summaries.values()):
            raise RuntimeError("Phase1065 identity control failed")
        integrity = {
            "case_counts": {
                model: summary["case_count"]
                for model, summary in summaries.items()
            },
            "identity_maximum": {
                model: summary["identity_maximum"]
                for model, summary in summaries.items()
            },
            "nonfinite_candidate_counts": {
                model: summary["nonfinite_candidate_count"]
                for model, summary in summaries.items()
            },
        }
    elif phase == 1066:
        if any(summary["pair_count"] != 120 for summary in summaries.values()):
            raise RuntimeError("Phase1066 pair count mismatch")
        if any(
            summary["clean_candidate_replay_rate"] != 1.0
            for summary in summaries.values()
        ):
            raise RuntimeError("Phase1066 clean replay failed")
        integrity = {
            "pair_counts": {
                model: summary["pair_count"]
                for model, summary in summaries.items()
            },
            "clean_candidate_replay_rates": {
                model: summary["clean_candidate_replay_rate"]
                for model, summary in summaries.items()
            },
            "condition_counts": {
                model: len(summary["condition_results"])
                for model, summary in summaries.items()
            },
        }
    else:
        if any(
            set(summary["clean_candidate_replay_rates"].values()) != {1.0}
            for summary in summaries.values()
        ):
            raise RuntimeError("Phase1067 clean replay failed")
        integrity = {
            "clean_candidate_replay_rates": {
                model: summary["clean_candidate_replay_rates"]
                for model, summary in summaries.items()
            },
            "condition_counts": {
                model: len(summary["condition_results"])
                for model, summary in summaries.items()
            },
        }

    return {
        "phase": phase,
        "protocol_digest": claimed_digest,
        "protocol_digest_recomputed": recomputed_digest,
        "strict_json_passed": True,
        "json_file_count": len(json_files),
        "jsonl_file_count": len(jsonl_files),
        "jsonl_row_count": jsonl_rows,
        "fp16_no_quantization_passed": True,
        "integrity": integrity,
    }


def main() -> None:
    result = {
        "schema_version": "phase1065_1067_integrity_audit.v1",
        "phases": {
            str(phase): audit_phase(phase, root)
            for phase, root in PHASE_ROOTS.items()
        },
        "all_checks_passed": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(OUTPUT)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
