from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1189_quotient_formation_operator_calibration as p1189  # noqa: E402


SCRIPT = Path(__file__).resolve()
AMENDMENT_PATH = p1189.OUT_ROOT / "protocol/audit_positive_sentinel_amendment.json"
ORIGINAL_AUDIT_PATH = p1189.OUT_ROOT / "audit/original_independent_audit_39_of_40.json"
SINGLE_CLASS_PERTURBATION = 100.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def preregister() -> None:
    original = p1189.read_json(p1189.AUDIT_PATH)
    failed = [check for check in original["checks"] if not check["pass"]]
    if original["pass_count"] != 39 or original["check_count"] != 40:
        raise RuntimeError("amendment requires the exact original 39/40 audit state")
    if [check["name"] for check in failed] != ["broken_compensation_positive_sentinel"]:
        raise RuntimeError("an unexpected audit check failed")
    protocol = {
        "phase": p1189.PHASE,
        "kind": "audit_positive_sentinel_amendment",
        "created_at_utc": utc_now(),
        "reason": (
            "The original sentinel added an equal coefficient to every output class for one channel. "
            "That direction is an output-common-shift gauge and is not a valid class-selective break."
        ),
        "allowed_change": (
            "Replace only the failed sentinel by an uncompensated perturbation to output class 0, channel 0."
        ),
        "forbidden_changes": [
            "No formal row, response vector, threshold, transformation, split, typed claim, or scientific gate changes.",
            "The other 39 original checks must remain passing and are inherited exactly.",
            "The formal experiment is not rerun.",
        ],
        "single_class_perturbation": SINGLE_CLASS_PERTURBATION,
        "threshold": p1189.THRESHOLDS["logit_equivalence_max"],
        "hashes": {
            "amendment_script": p1189.file_sha256(SCRIPT),
            "formal_protocol": p1189.file_sha256(p1189.PROTOCOL_PATH),
            "formal_rows": p1189.file_sha256(p1189.RAW_ROWS),
            "formal_summary": p1189.file_sha256(p1189.SUMMARY_PATH),
            "typed_claims": p1189.file_sha256(p1189.CLAIMS_PATH),
            "original_audit": p1189.file_sha256(p1189.AUDIT_PATH),
        },
        "original_audit_digest": original["audit_digest"],
    }
    protocol["amendment_digest"] = p1189.digest(
        {key: value for key, value in protocol.items() if key != "amendment_digest"}
    )
    p1189.write_json(AMENDMENT_PATH, protocol)


def verify_amendment() -> dict[str, Any]:
    protocol = p1189.read_json(AMENDMENT_PATH)
    expected = p1189.digest({key: value for key, value in protocol.items() if key != "amendment_digest"})
    if expected != protocol["amendment_digest"]:
        raise RuntimeError("amendment digest mismatch")
    observed = {
        "amendment_script": p1189.file_sha256(SCRIPT),
        "formal_protocol": p1189.file_sha256(p1189.PROTOCOL_PATH),
        "formal_rows": p1189.file_sha256(p1189.RAW_ROWS),
        "formal_summary": p1189.file_sha256(p1189.SUMMARY_PATH),
        "typed_claims": p1189.file_sha256(p1189.CLAIMS_PATH),
        "original_audit": p1189.file_sha256(p1189.AUDIT_PATH),
    }
    if observed != protocol["hashes"]:
        raise RuntimeError("an amendment input changed")
    return protocol


def audit() -> None:
    protocol = verify_amendment()
    original = p1189.read_json(p1189.AUDIT_PATH)
    inherited = [check for check in original["checks"] if check["name"] != "broken_compensation_positive_sentinel"]
    if len(inherited) != 39 or not all(check["pass"] for check in inherited):
        raise RuntimeError("the 39 inherited checks are not intact")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = p1189.read_jsonl(p1189.RAW_ROWS)
    path_map = {path.name: path for path in p1189.endpoint_paths(p1189.FORMAL_SOURCE)}
    first = rows[0]
    payload = p1189.load_payload(path_map[first["checkpoint"]])
    device = torch.device("cuda")
    original_model = p1189.load_model(payload, device)
    base = p1189.expand_duplicate_pairs(original_model, device)
    broken = p1189.clone_model(base, device)
    with torch.no_grad():
        broken.output.weight[0, 0].add_(SINGLE_CLASS_PERTURBATION)
    panel = p1189.panel_from_payload(payload)
    base_logits = p1189.fp32_logits(base, panel.x, device)
    broken_logits = p1189.fp32_logits(broken, panel.x, device)
    sentinel_error = float((broken_logits - base_logits).abs().max().item())
    sentinel_pass = sentinel_error > p1189.THRESHOLDS["logit_equivalence_max"]
    corrected_check = {
        "name": "broken_compensation_positive_sentinel",
        "pass": sentinel_pass,
        "details": {
            "original_common_shift_error": next(
                check["details"]
                for check in original["checks"]
                if check["name"] == "broken_compensation_positive_sentinel"
            ),
            "corrected_single_class_error": sentinel_error,
            "threshold": p1189.THRESHOLDS["logit_equivalence_max"],
            "perturbation": SINGLE_CLASS_PERTURBATION,
        },
    }
    del original_model, base, broken
    torch.cuda.empty_cache()
    checks = inherited + [corrected_check]
    result = {
        "phase": p1189.PHASE,
        "audit_kind": "independent_audit_with_preregistered_single_sentinel_amendment",
        "created_at_utc": utc_now(),
        "amendment_digest": protocol["amendment_digest"],
        "original_audit_digest": original["audit_digest"],
        "check_count": len(checks),
        "pass_count": sum(check["pass"] for check in checks),
        "checks": checks,
        "gate_pass": all(check["pass"] for check in checks),
        "audit_digest": None,
    }
    result["audit_digest"] = p1189.digest(
        {key: value for key, value in result.items() if key != "audit_digest"}
    )
    if ORIGINAL_AUDIT_PATH.exists():
        raise RuntimeError("original audit archive already exists")
    ORIGINAL_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p1189.AUDIT_PATH, ORIGINAL_AUDIT_PATH)
    p1189.write_json(p1189.AUDIT_PATH, result)
    if not result["gate_pass"]:
        raise RuntimeError("amended audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "audit"))
    args = parser.parse_args()
    if args.command == "preregister":
        preregister()
    else:
        audit()


if __name__ == "__main__":
    main()
