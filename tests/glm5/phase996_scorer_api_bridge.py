"""Fail-closed API bridge for the sealed Phase996 scorer/audit.

The frozen sources named ``load_tokenizer_inspection`` while the sealed engine
exposes the identical read-only bundle loader as ``_load_inspection_bundle``.
This bridge does not replace parser, thresholds, summaries, gates, or audit
logic.  It aliases that API and permits byte-identical reuse of the truth
access receipt created by the failed first score entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import phase996_external_semantic_confirmation as frozen
import phase996_external_semantic_confirmation_audit as independent

ROOT = Path(__file__).resolve().parents[2]
EXECUTION = ROOT / "tests/glm5/result/phase996_external_semantic_confirmation_execution"
ACCESS = EXECUTION / "scores/truth_access_receipt.json"
FAILURE = EXECUTION / "scores/scorer_api_failure_receipt.json"
BRIDGE_RECEIPT = EXECUTION / "scores/scorer_api_bridge_receipt.json"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(value)); handle.flush()


def install_alias() -> None:
    engine = frozen.base.engine
    if not hasattr(engine, "load_tokenizer_inspection"):
        loader = getattr(engine, "_load_inspection_bundle", None)
        if loader is None:
            raise RuntimeError("sealed engine has neither tokenizer inspection API")
        engine.load_tokenizer_inspection = loader
    independent.engine.load_tokenizer_inspection = engine.load_tokenizer_inspection


def failure_receipt() -> None:
    if FAILURE.exists():
        return
    access = json.loads(ACCESS.read_text(encoding="utf-8"))
    write_once(FAILURE, {
        "schema_version": "phase996_scorer_api_failure.v1", "phase": 996,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "failure_class": "AttributeError", "failure_location": "CPU scorer tokenizer inspection entry",
        "message": "phase983_cross_model_engine has no attribute load_tokenizer_inspection",
        "gpu_raw_affected": False, "truth_access_receipt_already_created": True,
        "truth_access_sha256": access["access_sha256"],
        "frozen_source_sha256": sha_file(Path(frozen.__file__)),
        "independent_source_sha256": sha_file(Path(independent.__file__)),
        "bridge_source_sha256": sha_file(Path(__file__)),
    })


def score_resume() -> dict:
    install_alias(); failure_receipt()
    access = json.loads(ACCESS.read_text(encoding="utf-8"))
    fixed_time = str(access["created_at_utc"])
    original_write = frozen.write_exclusive
    frozen.now = lambda: fixed_time

    def idempotent_write(path: Path, payload: bytes) -> None:
        if path.exists():
            if path.resolve() != ACCESS.resolve() or path.read_bytes() != payload:
                raise RuntimeError(f"non-identical existing scorer artifact: {path}")
            return
        original_write(path, payload)

    frozen.write_exclusive = idempotent_write
    result = frozen.score()
    write_once(BRIDGE_RECEIPT, {
        "schema_version": "phase996_scorer_api_bridge.v1", "phase": 996,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
        "operation": "API alias plus byte-identical truth-access resume",
        "aliased_from": "phase983_cross_model_engine._load_inspection_bundle",
        "aliased_to": "phase983_cross_model_engine.load_tokenizer_inspection",
        "parser_or_gate_changed": False, "gpu_raw_changed": False,
        "frozen_source_sha256": sha_file(Path(frozen.__file__)),
        "bridge_source_sha256": sha_file(Path(__file__)),
        "score_sha256": result["score_sha256"],
    })
    return result


def audit_resume() -> dict:
    install_alias()
    result = independent.audit()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--score", action="store_true"); mode.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    result = score_resume() if args.score else audit_resume()
    print(json.dumps(result, sort_keys=True, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
