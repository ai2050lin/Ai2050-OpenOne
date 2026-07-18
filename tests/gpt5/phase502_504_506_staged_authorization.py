#!/usr/bin/env python3
"""Freeze authorizations between Phase500 staged behavior reads."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("qwen3", "glm4", "deepseek7b")
STAGES = {
    "calibration": {
        "input_phase": 501,
        "input_dir": ROOT / "tests" / "gpt5" / "result" / "phase501_function_polarity_calibration",
        "selection_key": "passed_function_polarity_cells",
        "output_dir": ROOT / "tests" / "gpt5" / "result" / "phase502_staged_behavior_authorization",
        "output": "phase502_calibration_authorization.json",
    },
    "contract": {
        "input_phase": 503,
        "input_dir": ROOT / "tests" / "gpt5" / "result" / "phase503_vocab_observer_calibration",
        "selection_key": "passed_native_contracts",
        "output_dir": ROOT / "tests" / "gpt5" / "result" / "phase504_staged_behavior_authorization",
        "output": "phase504_contract_authorization.json",
    },
    "confirmation": {
        "input_phase": 505,
        "input_dir": ROOT / "tests" / "gpt5" / "result" / "phase505_independent_confirmation",
        "selection_key": "confirmed_native_contracts",
        "output_dir": ROOT / "tests" / "gpt5" / "result" / "phase506_staged_behavior_authorization",
        "output": "phase506_confirmation_authorization.json",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def key(item: dict[str, str]) -> tuple[str, ...]:
    return tuple(item[field] for field in ("function_class", "polarity", "vocab_system") if field in item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    args = parser.parse_args()
    spec = STAGES[args.stage]
    summaries = {}
    selections = {}
    shared: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for model in MODELS:
        path = spec["input_dir"] / f"phase{spec['input_phase']}_{model}_summary.json"
        if not path.exists():
            raise RuntimeError(f"Missing required summary: {path}")
        summary = load_json(path)
        if summary["model"] != model or summary["stage"] != args.stage:
            raise RuntimeError(f"Stage/model mismatch in {path}")
        summaries[model] = summary
        selected = list(summary.get(spec["selection_key"], []))
        selections[model] = selected
        for item in selected:
            shared[key(item)].append(model)
    shared_items = [
        {"contract_key": list(contract_key), "models": models}
        for contract_key, models in sorted(shared.items()) if len(models) >= 2
    ]
    payload: dict[str, Any] = {
        "schema_version": f"phase{spec['input_phase'] + 1}_{args.stage}_authorization.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "authorization_frozen",
        "stage": args.stage,
        "models_in_required_order": list(MODELS),
        "model_summaries": summaries,
        "shared_cells_or_contracts": shared_items,
        "sealed_read": False,
        "causal_intervention": False,
        "head_channel_neuron_scan": False,
    }
    if args.stage == "calibration":
        payload["stage_b_cells_by_model"] = selections
        payload["authorization"] = {"vocab_observer_calibration": any(selections.values())}
    elif args.stage == "contract":
        payload["stage_c_contracts_by_model"] = selections
        payload["authorization"] = {"independent_confirmation": any(selections.values())}
    else:
        payload["confirmed_contracts_by_model"] = selections
        payload["shared_confirmed_contracts"] = shared_items
        models = []
        for model in MODELS:
            if any(model in item["models"] for item in shared_items):
                models.append(model)
        payload["physical_models_in_required_order"] = models
        payload["authorization"] = {
            "open_conditional_physical": bool(shared_items),
            "cross_model_physical": len(models) >= 2,
        }
    spec["output_dir"].mkdir(parents=True, exist_ok=True)
    path = spec["output_dir"] / spec["output"]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
