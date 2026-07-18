#!/usr/bin/env python3
"""Phase491 read-only geometry contamination audit and prediction freeze.

The Phase490 claim-end embedding result is a lexical identity artifact: the
counterfactual claim changes an attribute token while identity/plain share the
same claim token. This audit excludes positions that cannot yet contain the
full relation event and freezes one late prompt-end window per authorized model
before the independent physical-prediction split is read.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "tests" / "gpt5" / "result" / "phase490_open_native_relation_geometry"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase491_geometry_contamination_audit"
AUDIT_PATH = OUT_DIR / "phase491_geometry_contamination_audit.json"
FREEZE_PATH = OUT_DIR / "phase491_physical_prediction_freeze.json"
MODELS = ("qwen3", "glm4")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_window(summary: dict[str, Any]) -> dict[str, Any]:
    reports = summary["geometry"]["reports"]
    eligible = [
        row
        for row in reports
        if row["position_role"] == "prompt_end"
        and row["normalized_depth"] >= 0.5
        and row["q_native"] > 0
        and row["relation_direction_coherence"] >= 0.5
        and all(payload["q_native"] > 0 for payload in row["by_family"].values())
    ]
    if not eligible:
        raise RuntimeError(f"No deconfounded prompt-end window for {summary['model']}")
    best = max(
        eligible,
        key=lambda row: (
            min(payload["q_native"] for payload in row["by_family"].values()),
            row["q_native"],
            row["relation_direction_coherence"],
            -row["layer_with_embedding"],
        ),
    )
    return {
        "model": summary["model"],
        "position_role": "prompt_end",
        "layer_with_embedding": best["layer_with_embedding"],
        "normalized_depth": best["normalized_depth"],
        "geometry_window_q_native": best["q_native"],
        "geometry_window_family_q": {
            family: payload["q_native"] for family, payload in best["by_family"].items()
        },
        "geometry_window_direction_coherence": best["relation_direction_coherence"],
        "selection_rule": (
            "prompt_end only; normalized depth >= 0.5; q>0 in both families; "
            "coherence>=0.5; maximize minimum family q, then overall q, coherence, lower layer"
        ),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {
        model: load_json(IN_DIR / f"phase490_{model}_summary.json")
        for model in MODELS
    }
    windows = [select_window(summaries[model]) for model in MODELS]
    lexical_artifacts = []
    for model, summary in summaries.items():
        embedding_claim = next(
            row
            for row in summary["geometry"]["reports"]
            if row["position_role"] == "claim_end" and row["layer_with_embedding"] == 0
        )
        lexical_artifacts.append({
            "model": model,
            "position_role": "claim_end",
            "layer_with_embedding": 0,
            "q_native": embedding_claim["q_native"],
            "relation_direction_coherence": embedding_claim["relation_direction_coherence"],
            "classification": "lexical_token_identity_artifact",
            "reason": (
                "identity/plain have the same claim token while entailed/counterfactual use "
                "different attribute tokens; no network computation has occurred at embedding layer"
            ),
        })
    audit = {
        "schema_version": "phase491_geometry_contamination_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "contamination_audit_complete",
        "sealed_split_read": False,
        "lexical_artifacts": lexical_artifacts,
        "position_eligibility": {
            "evidence_end": "ineligible because the queried claim has not yet been observed",
            "claim_end": "diagnostic only because counterfactual pairs change the claim attribute token",
            "prompt_end": "eligible candidate because evidence, claim, question, and output boundary are all present",
        },
        "selected_deconfounded_windows": windows,
        "allowed_claim": (
            "Qwen3 and GLM4 have late prompt-end observational candidates after removing the obvious embedding artifact."
        ),
        "forbidden_claim": (
            "The candidates are not yet independent physical predictions or causal relation mechanisms."
        ),
    }
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze = {
        "schema_version": "phase491_physical_prediction_freeze.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_physical_prediction_split_read",
        "models_in_required_order": list(MODELS),
        "windows": {row["model"]: row for row in windows},
        "projection": {
            "reuse_phase490_model_seed": True,
            "dimension": 64,
            "fit_direction_on_geometry_window_only": True,
            "fit_intercept_on_geometry_window_only": True,
        },
        "independent_gate": {
            "q_native_positive_overall": True,
            "q_native_positive_in_each_family": True,
            "direction_coherence_min": 0.5,
            "truth_prediction_lcb95_above": 0.5,
            "truth_prediction_required_for_each_native_track": True,
            "truth_prediction_required_for_each_family": True,
        },
        "authorization": {
            "physical_prediction_split_read": True,
            "sealed_split_read": False,
            "output_event_map": False,
            "head_channel_neuron_scan": False,
            "causal_intervention": False,
        },
    }
    FREEZE_PATH.write_text(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(AUDIT_PATH)
    print(FREEZE_PATH)


if __name__ == "__main__":
    main()
