#!/usr/bin/env python3
"""Freeze the Phase516 model-specific GLM4 relation observation protocol."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
PHASE509_DIR = ROOT / "tests/gpt5/result/phase509_dual_contract_protocol"
PHASE509_CONTRACT = PHASE509_DIR / "phase509_frozen_contract.json"
AUTH_PATH = ROOT / "tests/gpt5/result/phase515_physical_authorization/phase515_physical_authorization.json"
OUT_DIR = ROOT / "tests/gpt5/result/phase516_relation_physical_protocol"

FIT_PATH = PHASE509_DIR / "phase509_physical_fit_relation.jsonl"
PREDICTION_PATH = PHASE509_DIR / "phase509_physical_prediction_relation.jsonl"
POSITION_ROLES = (
    "target_evidence_end",
    "distractor_evidence_end",
    "claim_entity_end",
    "claim_relation_end",
    "claim_end",
    "prompt_end",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    phase509 = read_json(PHASE509_CONTRACT)
    authorization = read_json(AUTH_PATH)
    if authorization["physical_contracts_by_model"].get("glm4") != ["R"]:
        raise RuntimeError("Phase516 requires GLM4 model-specific R-only authorization")
    expected_fit = phase509["split_files"]["physical_fit_relation"]["sha256"]
    expected_prediction = phase509["split_files"]["physical_prediction_relation"]["sha256"]
    if sha256_file(FIT_PATH) != expected_fit or sha256_file(PREDICTION_PATH) != expected_prediction:
        raise RuntimeError("Phase509 relation physical split hash drift")
    contract = {
        "schema_version": "phase516_relation_physical_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_physical_collection",
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "authorization_path": str(AUTH_PATH.relative_to(ROOT)),
        "model": "glm4",
        "contract": "R",
        "scope": "model-specific natural-true-false direct-directed relation observation",
        "fit_split": {
            "path": str(FIT_PATH.relative_to(ROOT)),
            "sha256": expected_fit,
            "observer_train_pair_parity": 0,
            "window_selection_pair_parity": 1,
        },
        "prediction_split": {
            "path": str(PREDICTION_PATH.relative_to(ROOT)),
            "sha256": expected_prediction,
            "read_only_after_observer_ledger_written": True,
        },
        "projection": {"dimension": 64, "seed": 516037},
        "random_label_controls": {"count": 4, "seeds": [516101, 516103, 516107, 516109]},
        "position_roles": list(POSITION_ROLES),
        "observer": {
            "kind": "unit-normalized center difference with midpoint threshold",
            "selection_rank": [
                "minimum surface accuracy",
                "four-way paired-world accuracy",
                "overall accuracy",
                "earlier layer",
                "frozen role order",
            ],
            "selection_uses_only_fit_odd_pairs": True,
            "prediction_window_frozen_before_prediction_read": True,
        },
        "prediction_gate": {
            "identity_lcb95_min": 0.80,
            "native_plain_lcb95_min": 0.80,
            "overall_lcb95_min": 0.80,
            "four_way_pair_lcb95_min": 0.75,
        },
        "evidence_boundaries": {
            "shared_cross_model_claim": False,
            "observation_is_compute_transport": False,
            "causal_intervention": False,
            "head_channel_neuron_scan": False,
            "sealed_read": False,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "phase516_frozen_relation_physical_contract.json"
    path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    static = {
        "schema_version": "phase516_relation_physical_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run",
        "contract_sha256": sha256_file(path),
        "fit_file_exists": FIT_PATH.exists(),
        "prediction_file_exists": PREDICTION_PATH.exists(),
        "cuda_used": False,
        "model_loaded": False,
        "prediction_read": False,
        "sealed_read": False,
    }
    audit_path = OUT_DIR / "phase516_relation_physical_static_audit.json"
    audit_path.write_text(
        json.dumps(static, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(path)
    print(audit_path)


if __name__ == "__main__":
    main()
