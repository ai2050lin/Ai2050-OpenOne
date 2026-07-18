#!/usr/bin/env python3
"""Freeze the Phase507 physical-collection decision without loading a model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUTH_PATH = (
    ROOT
    / "tests/gpt5/result/phase506_staged_behavior_authorization"
    / "phase506_confirmation_authorization.json"
)
OUT_DIR = ROOT / "tests/gpt5/result/phase507_conditional_physical_gate"
OUT_PATH = OUT_DIR / "phase507_conditional_physical_gate.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    authorization = read_json(AUTH_PATH)
    authorized_models = authorization["physical_models_in_required_order"]
    if authorized_models:
        status = "authorized_requires_separate_physical_runner"
        reason = "At least two models confirmed an identical frozen native contract."
    else:
        status = "gate_stopped_no_physical_collection"
        reason = (
            "No model passed the complete vocabulary-observer contract, so independent "
            "confirmation and conditional physical collection have empty denominators."
        )
    payload = {
        "schema_version": "phase507_conditional_physical_gate.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason": reason,
        "authorization_source": str(AUTH_PATH.relative_to(ROOT)),
        "authorized_models": authorized_models,
        "shared_confirmed_contracts": authorization["shared_confirmed_contracts"],
        "cuda_used": False,
        "model_weights_loaded": False,
        "physical_rows_collected": 0,
        "sealed_split_read": False,
        "causal_intervention": False,
        "head_channel_neuron_scan": False,
        "gate_compliance": not authorized_models,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUT_PATH)


if __name__ == "__main__":
    main()
