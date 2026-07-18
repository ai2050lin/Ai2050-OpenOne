#!/usr/bin/env python3
"""Freeze Phase509 behavior and physical authorizations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("qwen3", "glm4", "deepseek7b")

STAGES = {
    "calibration": {
        "phase": 511,
        "input_phase": 510,
        "input": ROOT / "tests/gpt5/result/phase510_relation_binding_calibration",
        "out": ROOT / "tests/gpt5/result/phase511_calibration_authorization",
        "filename": "phase511_calibration_authorization.json",
    },
    "confirmation": {
        "phase": 513,
        "input_phase": 512,
        "input": ROOT / "tests/gpt5/result/phase512_relation_binding_confirmation",
        "out": ROOT / "tests/gpt5/result/phase513_confirmation_authorization",
        "filename": "phase513_confirmation_authorization.json",
    },
    "physical": {
        "phase": 515,
        "input_phase": 514,
        "input": ROOT / "tests/gpt5/result/phase514_joint_confirmation",
        "out": ROOT / "tests/gpt5/result/phase515_physical_authorization",
        "filename": "phase515_physical_authorization.json",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_path(directory: Path, phase: int, model: str) -> Path:
    return directory / f"phase{phase}_{model}_summary.json"


def load_summaries(stage: str) -> dict[str, dict[str, Any]]:
    spec = STAGES[stage]
    return {
        model: read_json(summary_path(spec["input"], spec["input_phase"], model))
        for model in MODELS
    }


def calibration_payload(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    relation_models = [
        model for model in MODELS
        if summaries[model]["contract_summaries"].get("R", {}).get("gate_pass", False)
    ]
    binding_models = [
        model for model in MODELS
        if summaries[model]["contract_summaries"].get("B", {}).get("gate_pass", False)
    ]
    return {
        "relation_models": relation_models,
        "binding_models": binding_models,
        "joint_models": [],
        "authorization": {
            "relation_confirmation": bool(relation_models),
            "binding_confirmation": bool(binding_models),
            "joint_confirmation": False,
            "physical": False,
        },
    }


def confirmation_payload(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    relation_models = [
        model for model in MODELS
        if summaries[model]["contract_summaries"].get("R", {}).get("gate_pass", False)
    ]
    binding_models = [
        model for model in MODELS
        if summaries[model]["contract_summaries"].get("B", {}).get("gate_pass", False)
    ]
    joint_models = [model for model in MODELS if model in relation_models and model in binding_models]
    return {
        "relation_models": relation_models,
        "binding_models": binding_models,
        "joint_models": joint_models,
        "authorization": {
            "relation_physical_candidate": bool(relation_models),
            "binding_physical_candidate": bool(binding_models),
            "joint_confirmation": bool(joint_models),
            "physical": False,
        },
    }


def physical_payload(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    confirmation_auth = read_json(
        ROOT
        / "tests/gpt5/result/phase513_confirmation_authorization"
        / "phase513_confirmation_authorization.json"
    )
    relation_models = confirmation_auth["relation_models"]
    binding_models = confirmation_auth["binding_models"]
    joint_models = [
        model for model in MODELS
        if summaries[model]["contract_summaries"].get("J", {}).get("gate_pass", False)
    ]
    by_model = {
        model: [
            contract
            for contract, selected in (
                ("R", model in relation_models),
                ("B", model in binding_models),
                ("J", model in joint_models),
            )
            if selected
        ]
        for model in MODELS
    }
    return {
        "relation_models": relation_models,
        "binding_models": binding_models,
        "joint_models": joint_models,
        "physical_contracts_by_model": by_model,
        "physical_models_in_required_order": [model for model in MODELS if by_model[model]],
        "shared_subcontracts": {
            "R": len(relation_models) >= 2,
            "B": len(binding_models) >= 2,
            "J": len(joint_models) >= 2,
        },
        "authorization": {
            "model_specific_physical": any(by_model.values()),
            "shared_relation_physical": len(relation_models) >= 2,
            "shared_binding_physical": len(binding_models) >= 2,
            "shared_joint_physical": len(joint_models) >= 2,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    args = parser.parse_args()
    summaries = load_summaries(args.stage)
    if args.stage == "calibration":
        selection = calibration_payload(summaries)
    elif args.stage == "confirmation":
        selection = confirmation_payload(summaries)
    else:
        selection = physical_payload(summaries)
    payload = {
        "schema_version": f"phase{STAGES[args.stage]['phase']}_{args.stage}_authorization.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "authorization_frozen",
        "stage": args.stage,
        "models_in_required_order": list(MODELS),
        "model_summaries": summaries,
        **selection,
        "sealed_split_read": False,
        "causal_intervention": False,
        "head_channel_neuron_scan": False,
    }
    out_dir = STAGES[args.stage]["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / STAGES[args.stage]["filename"]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
