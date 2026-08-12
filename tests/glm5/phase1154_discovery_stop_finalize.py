#!/usr/bin/env python3
"""Deterministically close Phase1154 after the discovery candidate failed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1154_learned_morphology_external_validity"
MAIN_SCRIPT = ROOT / "tests/glm5/phase1154_learned_morphology_external_validity.py"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    discovery = read_json(OUT_ROOT / "runs/discovery/summary.json")
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    if sha256_file(MAIN_SCRIPT) != protocol["script_sha256"]:
        raise RuntimeError("frozen Phase1154 script mismatch")
    if not discovery["behavior_gate_passed"]:
        reason = "discovery_behavior_gate_failed"
    elif fit["candidate_qualified"]:
        raise RuntimeError("early stop is illegal because the candidate qualified")
    else:
        reason = "discovery_identification_gate_failed"
    if (OUT_ROOT / "runs/confirmation").exists() or (OUT_ROOT / "predictions").exists():
        raise RuntimeError("confirmation artifacts exist despite discovery stop")
    final = {
        "phase": 1154,
        "protocol_digest": protocol["protocol_digest"],
        "discovery_summary_digest": discovery["summary_digest"],
        "fit_digest": fit["fit_digest"],
        "discovery_behavior_passed": bool(discovery["behavior_gate_passed"]),
        "discovery_candidate_qualified": bool(fit["candidate_qualified"]),
        "confirmation_trained": False,
        "confirmation_predicted": False,
        "learned_morphology_external_validity_confirmed": False,
        "phase1155_free_network_tomography_authorized": False,
        "pretrained_model_mechanism_claim_authorized": False,
        "stop_reason": reason,
        "outcome": "learned_morphology_discovery_transfer_failed",
        "claim_boundary": "The frozen functional-tomography candidate failed discovery transfer to learned factorized-role systems. No confirmation or free-network inference is allowed.",
        "auto_continue": False,
    }
    final["final_digest"] = digest(final)
    path = OUT_ROOT / "analysis/final.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite final")
    path.write_text(json.dumps(final, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical(final))


if __name__ == "__main__":
    main()
