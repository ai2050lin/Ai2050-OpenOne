#!/usr/bin/env python3
"""Post-hoc, no-model audit of the Phase1326 final-hidden readout semantics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import transformers

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
BEHAVIOR = T / "result/phase1325_c039_qwen3_behavior"
FIELD = T / "result/phase1326_c039_composition_field"
RAW = BEHAVIOR / "raw/candidate_scores.jsonl"
MANIFEST = FIELD / "protocol/frozen_field_manifest.jsonl"
ARRAYS = FIELD / "raw/full_state_composition_field.npz"
FINAL = FIELD / "analysis/final.json"
OUT = FIELD / "audit/posthoc_readout_semantics_erratum.json"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    model_source = Path(transformers.__file__).parent / "models/qwen3/modeling_qwen3.py"
    source_text = model_source.read_text(encoding="utf-8")
    source_final_norm = "hidden_states = self.norm(hidden_states)" in source_text
    source_direct_head = "logits = self.lm_head(hidden_states[:, slice_indices, :])" in source_text
    raw = {item["case_id"]: item for item in rows(RAW)}
    manifest = rows(MANIFEST)
    arrays = np.load(ARRAYS)
    field_margins = arrays["yes_no_margin"].astype(np.float64)
    direct_margins = np.empty_like(field_margins)
    for pair_index, pair in enumerate(manifest):
        yes_index, no_index = pair["candidates"].index("yes"), pair["candidates"].index("no")
        for state_index, state in enumerate(pair["states"]):
            logits = raw[state["case_id"]]["candidate_logits"]
            direct_margins[pair_index, state_index] = logits[yes_index] - logits[no_index]
    denominator = np.linalg.norm(field_margins) * np.linalg.norm(direct_margins)
    comparison = {
        "direct_behavior_accuracy": float(np.mean([item["correct"] for item in raw.values()])),
        "field_replay_accuracy": float(np.mean(arrays["behavior_correct"])),
        "margin_cosine": float(np.dot(field_margins.ravel(), direct_margins.ravel()) / denominator),
        "margin_mae": float(np.mean(np.abs(field_margins - direct_margins))),
        "same_margin_sign_fraction": float(np.mean(np.sign(field_margins) == np.sign(direct_margins))),
    }
    value = {
        "phase": 1326, "campaign": "C039", "audit_type": "posthoc_instrument_semantics_erratum",
        "model_weights_loaded": False, "formal_metrics_or_gates_changed": False,
        "source_evidence": {"path": str(model_source), "sha256": sha(model_source),
                            "final_hidden_is_normalized": source_final_norm,
                            "lm_head_consumes_returned_hidden_directly": source_direct_head},
        "artifact_hashes": {"behavior_raw": sha(RAW), "field_manifest": sha(MANIFEST),
                            "field_arrays": sha(ARRAYS), "field_final": sha(FINAL)},
        "comparison": comparison,
        "finding": "Phase1326 candidate_scores applied model.model.norm to a hidden_states[-1] that Qwen3Model had already normalized.",
        "scope": "Invalidates Phase1326 replay accuracy and numerical margin magnitudes as exact reproductions; it does not alter independently failed cross-surface panel-win and control gates.",
        "authorization": "none_c039_remains_closed_no_rerun",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical({"source_final_norm": source_final_norm, "source_direct_head": source_direct_head,
                     **comparison, "authorization": value["authorization"]}))
    if not source_final_norm or not source_direct_head:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
