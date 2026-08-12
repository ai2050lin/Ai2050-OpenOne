from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1191_prefix_future_formation_identity as p1191  # noqa: E402


SCRIPT = Path(__file__).resolve()
PREREG_PATH = p1191.OUT_ROOT / "protocol/finalize_type_mapping_amendment_preregistration.json"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def frozen_inputs() -> dict[str, Path]:
    return {
        "phase1191_source": p1191.SCRIPT,
        "phase1191_protocol": p1191.PROTOCOL_PATH,
        "phase1191_summary": p1191.SUMMARY_PATH,
        "phase1191_claims": p1191.CLAIMS_PATH,
        "phase1191_audit": p1191.AUDIT_PATH,
        "amendment_source": SCRIPT,
    }


def preregister() -> None:
    p1191.verify_protocol()
    summary = read_json(p1191.SUMMARY_PATH)
    claims = read_json(p1191.CLAIMS_PATH)
    audit = read_json(p1191.AUDIT_PATH)
    if summary.get("decision") != "negative_boundary":
        raise RuntimeError("amendment is only valid for the frozen negative_boundary decision")
    if not claims.get("negative", {}).get("gate_pass"):
        raise RuntimeError("the frozen negative typed gate did not pass")
    if not audit.get("gate_pass"):
        raise RuntimeError("the frozen independent audit did not pass")
    payload = {
        "phase": 1191,
        "created_at_utc": utc_now(),
        "amendment_kind": "finalize_decision_to_typed_claim_key_mapping_only",
        "reason": (
            "The frozen finalizer indexed typed_claims with the decision label negative_boundary, "
            "while the frozen typed ledger uses the key negative. This amendment maps only that "
            "label and changes no vector, null, threshold, split, statistic, audit, or decision."
        ),
        "frozen_mapping": {"positive": "positive", "negative_boundary": "negative"},
        "frozen_decision": summary["decision"],
        "frozen_summary_digest": summary["summary_digest"],
        "frozen_audit_digest": audit["audit_digest"],
        "input_hashes": {name: file_sha256(path) for name, path in frozen_inputs().items()},
        "amendment_digest": None,
    }
    payload["amendment_digest"] = digest(
        {key: value for key, value in payload.items() if key != "amendment_digest"}
    )
    write_json(PREREG_PATH, payload)


def verify_amendment() -> dict[str, Any]:
    amendment = read_json(PREREG_PATH)
    expected = digest({key: value for key, value in amendment.items() if key != "amendment_digest"})
    if amendment.get("amendment_digest") != expected:
        raise RuntimeError("finalization amendment digest mismatch")
    for name, path in frozen_inputs().items():
        if file_sha256(path) != amendment["input_hashes"][name]:
            raise RuntimeError(f"frozen finalization input changed: {name}")
    p1191.verify_protocol()
    return amendment


def finalize() -> None:
    amendment = verify_amendment()
    protocol = read_json(p1191.PROTOCOL_PATH)
    summary = read_json(p1191.SUMMARY_PATH)
    claims = read_json(p1191.CLAIMS_PATH)
    audit = read_json(p1191.AUDIT_PATH)
    claim_key = amendment["frozen_mapping"].get(summary["decision"])
    typed_match = bool(claim_key and claims[claim_key]["gate_pass"])
    completed = bool(summary["decision"] != "ambiguous" and typed_match and audit.get("gate_pass"))
    final = {
        "phase": 1191,
        "created_at_utc": utc_now(),
        "status": "local_event_only_negative_boundary" if completed else "ambiguous_or_failed",
        "protocol_digest": protocol["protocol_digest"],
        "summary_digest": summary["summary_digest"],
        "claims_sha256": file_sha256(p1191.CLAIMS_PATH),
        "audit_digest": audit.get("audit_digest"),
        "finalization_amendment_digest": amendment["amendment_digest"],
        "decision": summary["decision"],
        "typed_claim_key": claim_key,
        "independent_audit_pass": bool(audit.get("gate_pass")),
        "main_gate_complete": completed,
        "evidence_grade": "E3_KT_free_network" if completed else "no_upgrade",
        "authorized_next": {
            "static_prefix_identity_search": False,
            "formation_causal_branch_preregistration": False,
            "transformer_or_language_model_transfer": False,
            "theory_closure": False,
        },
        "claim_scope": (
            "The result separates a locally repeatable SGD response event from a persistent trajectory identity. "
            "The frozen negative boundary denies this early-to-future identity rule on the tested RoleSquare "
            "task family; it does not deny the local formation events confirmed in Phase1190."
        ),
        "final_digest": None,
    }
    final["final_digest"] = digest({key: value for key, value in final.items() if key != "final_digest"})
    write_json(p1191.FINAL_PATH, final)
    print(json.dumps(final, ensure_ascii=False, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "finalize"))
    args = parser.parse_args()
    {"preregister": preregister, "finalize": finalize}[args.command]()


if __name__ == "__main__":
    main()
