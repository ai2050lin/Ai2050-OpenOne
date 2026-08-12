from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1191_prefix_future_formation_identity as p1191  # noqa: E402
import phase1191_prefix_future_formation_identity_audit as base_audit  # noqa: E402


SCRIPT = Path(__file__).resolve()
AMENDMENT_PATH = p1191.OUT_ROOT / "protocol/audit_replay_closure_amendment.json"
AMENDMENT_RESULT = p1191.OUT_ROOT / "audit/replay_closure_amendment_result.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hashes() -> dict[str, str]:
    return {
        "amendment_script": p1191.file_sha256(SCRIPT),
        "formal_protocol": p1191.file_sha256(p1191.PROTOCOL_PATH),
        "formal_rows": p1191.file_sha256(p1191.RAW_ROWS),
        "formal_summary": p1191.file_sha256(p1191.SUMMARY_PATH),
        "typed_claims": p1191.file_sha256(p1191.CLAIMS_PATH),
        "frozen_audit_script": p1191.file_sha256(p1191.AUDIT_SCRIPT),
    }


def preregister() -> None:
    if p1191.AUDIT_PATH.exists():
        raise RuntimeError("an audit result already exists")
    protocol = {
        "phase": p1191.PHASE,
        "kind": "audit_replay_input_closure_amendment",
        "created_at_utc": utc_now(),
        "reason": (
            "The frozen replay passed one trajectory into a constructor that deterministically requires all "
            "eight same-task replicates to close the cyclic replicate-null mapping."
        ),
        "allowed_change": (
            "For each replicate-0 CUDA replay, provide all eight endpoints from that task, then compare the "
            "replicate-0 reconstruction exactly as originally specified."
        ),
        "forbidden_changes": [
            "No formal vector, metric, threshold, split, null mapping, typed claim, or decision changes.",
            "No scientific analysis is rerun or filtered.",
            "The frozen audit source remains byte-identical.",
        ],
        "hashes": hashes(),
    }
    protocol["amendment_digest"] = p1191.digest(
        {key: value for key, value in protocol.items() if key != "amendment_digest"}
    )
    p1191.write_json(AMENDMENT_PATH, protocol)


def verify() -> dict[str, Any]:
    protocol = p1191.read_json(AMENDMENT_PATH)
    expected = p1191.digest({key: value for key, value in protocol.items() if key != "amendment_digest"})
    if expected != protocol["amendment_digest"] or protocol["hashes"] != hashes():
        raise RuntimeError("amendment changed")
    return protocol


def audit() -> None:
    protocol = verify()
    original_builder = p1191.build_rows
    endpoint_paths = p1191.endpoints(p1191.FORMAL_SOURCE)
    payload_by_path = {path: p1191.p1189.load_payload(path) for path in endpoint_paths}

    def closed_builder(paths: list[Path], corpus: str, device: torch.device) -> list[dict[str, Any]]:
        if len(paths) != 1:
            return original_builder(paths, corpus, device)
        payload = payload_by_path[paths[0]]
        task_paths = sorted(
            path
            for path, candidate in payload_by_path.items()
            if candidate["task_name"] == payload["task_name"]
        )
        rows = original_builder(task_paths, corpus, device)
        return [row for row in rows if row["replicate"] == payload["replicate"]]

    p1191.build_rows = closed_builder
    try:
        base_audit.audit()
    finally:
        p1191.build_rows = original_builder
    result = p1191.read_json(p1191.AUDIT_PATH)
    amendment_result = {
        "phase": p1191.PHASE,
        "created_at_utc": utc_now(),
        "amendment_digest": protocol["amendment_digest"],
        "audit_digest": result["audit_digest"],
        "audit_gate_pass": result["gate_pass"],
        "result_digest": None,
    }
    amendment_result["result_digest"] = p1191.digest(
        {key: value for key, value in amendment_result.items() if key != "result_digest"}
    )
    p1191.write_json(AMENDMENT_RESULT, amendment_result)


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
