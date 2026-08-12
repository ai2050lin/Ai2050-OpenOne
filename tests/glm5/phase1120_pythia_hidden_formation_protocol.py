#!/usr/bin/env python3
"""Freeze the Phase1120 Pythia residual-formation protocol."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PHASE = 1120
ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1117_pythia_training_dynamics_verified_safetensors_v4"
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1120_pythia_hidden_formation_map"
MODEL_ROOT = ROOT / "models" / "hf" / "pythia-1.4b-deduped"
MODEL_REPO = "EleutherAI/pythia-1.4b-deduped"
WEIGHT_FORMAT = "model.safetensors"
PRECISION = "fp16"

CHECKPOINTS = (
    "step0",
    "step1000",
    "step4000",
    "step16000",
    "step64000",
    "step143000",
)
SPLITS = ("discovery", "independent_confirmation", "heldout")
HIDDEN_SIZE = 2048
NUM_LAYERS = 24
HIDDEN_STATE_COUNT = 25
ELIGIBLE_LAYER_INDICES = tuple(range(1, 24))
PROJECTION_SEED = 1120
PROJECTION_DIM = 128

THRESHOLDS = {
    "minimum_finite_fraction": 0.99,
    "maximum_final_logit_reproduction_error": 0.02,
    "minimum_final_readout_advantage": 0.10,
    "minimum_step0_to_final_readout_gain": 0.10,
    "minimum_readout_onset_advantage": 0.10,
    "minimum_readout_onset_gain": 0.05,
    "minimum_final_geometry_advantage": 0.05,
    "minimum_step0_to_final_geometry_gain": 0.05,
    "minimum_geometry_onset_advantage": 0.05,
    "minimum_geometry_onset_gain": 0.025,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def make_projection() -> np.ndarray:
    rng = np.random.default_rng(PROJECTION_SEED)
    signs = rng.integers(0, 2, size=(HIDDEN_SIZE, PROJECTION_DIM), dtype=np.int8)
    return ((signs.astype(np.float32) * 2.0) - 1.0) / math.sqrt(PROJECTION_DIM)


def freeze() -> dict[str, Any]:
    source_prereg = read_json(SOURCE_ROOT / "protocol" / "preregistration.json")
    source_protocol_audit = read_json(SOURCE_ROOT / "protocol" / "audit.json")
    source_integrity = read_json(SOURCE_ROOT / "protocol" / "checkpoint_integrity.json")
    source_final = read_json(SOURCE_ROOT / "analysis" / "final_summary.json")
    source_result_audit = read_json(SOURCE_ROOT / "audit" / "result_audit.json")
    rows = list(read_jsonl(SOURCE_ROOT / "protocol" / "cases.jsonl"))

    if not source_protocol_audit["all_checks_passed"]:
        raise RuntimeError("Phase1117 source protocol audit failed")
    if not source_integrity["all_checks_passed"]:
        raise RuntimeError("Phase1117 checkpoint integrity failed")
    if not source_final["trajectory_authorized"] or not source_final["full_trajectory_complete"]:
        raise RuntimeError("Phase1117 trajectory is not a completed positive behavior object")
    if not source_result_audit["all_checks_passed"]:
        raise RuntimeError("Phase1117 result audit failed")
    if digest(rows) != source_prereg["case_digest"]:
        raise RuntimeError("Phase1117 source case digest mismatch")
    if len(rows) != 684:
        raise RuntimeError("expected 684 frozen source cases")

    split_counts = Counter(row["split"] for row in rows)
    pair_counts = Counter(row["pair_id"] for row in rows)
    concept_counts = Counter(row["split"] for row in {row["concept_id"]: row for row in rows}.values())
    if split_counts != Counter({split: 228 for split in SPLITS}):
        raise RuntimeError(f"unexpected case split counts: {split_counts}")
    if any(value != 2 for value in pair_counts.values()) or len(pair_counts) != 342:
        raise RuntimeError("source cases are not 342 complete sense pairs")
    if concept_counts != Counter({split: 19 for split in SPLITS}):
        raise RuntimeError(f"unexpected concept split counts: {concept_counts}")
    if any(name not in source_integrity["checkpoints"] for name in CHECKPOINTS):
        raise RuntimeError("a frozen Phase1120 checkpoint is absent from the Phase1117 integrity ledger")
    if any(not (MODEL_ROOT / name / WEIGHT_FORMAT).exists() for name in CHECKPOINTS):
        raise RuntimeError("a frozen local checkpoint is missing")

    projection = make_projection()
    projection_path = OUT_ROOT / "protocol" / "projection_matrix.npy"
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(projection_path, projection, allow_pickle=False)
    write_jsonl(OUT_ROOT / "protocol" / "cases.jsonl", rows)

    prereg_core = {
        "schema_version": "phase1120_pythia_hidden_formation_preregistration.v1",
        "phase": PHASE,
        "source_phase": 1117,
        "source_protocol_digest": source_prereg["protocol_digest"],
        "source_final_digest": source_final["final_digest"],
        "source_result_audit_digest": source_result_audit["audit_digest"],
        "model_repo": MODEL_REPO,
        "model_root": str(MODEL_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "weight_format": WEIGHT_FORMAT,
        "precision": PRECISION,
        "quantization": "none",
        "checkpoints": list(CHECKPOINTS),
        "splits": list(SPLITS),
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "hidden_state_count": HIDDEN_STATE_COUNT,
        "eligible_layer_indices": list(ELIGIBLE_LAYER_INDICES),
        "projection": {
            "seed": PROJECTION_SEED,
            "dimension": PROJECTION_DIM,
            "distribution": "Rademacher entries divided by sqrt(projection_dim)",
            "shape": list(projection.shape),
            "sha256": file_sha256(projection_path),
        },
        "case_count": len(rows),
        "pair_count": len(pair_counts),
        "concept_count": 57,
        "case_digest": digest(rows),
        "split_case_counts": dict(split_counts),
        "split_concept_counts": dict(concept_counts),
        "thresholds": THRESHOLDS,
        "layer_selection": {
            "readout": "discovery argmax of step143000-minus-step0 true-minus-control direction advantage",
            "geometry": "discovery argmax of step143000-minus-step0 within-concept-minus-deranged-control median cosine",
            "ties": "earliest eligible hidden-state index",
            "terminal_state_excluded": True,
            "confirmation": "both independent_confirmation and heldout must independently pass final level and formation gain",
        },
        "prospective_predictions": {
            "P1": "The inherited cases, physical checkpoint identities, fixed projection, counts, and digests pass before hidden outputs are read.",
            "P2": "At every checkpoint the terminal stored hidden state reproduces selected model logits within 0.02 and all recorded values are finite.",
            "P3": "At the discovery-selected intermediate layer, both confirmation splits show candidate-relative readout advantage at least 0.10 at step143000 and step0-to-final gain at least 0.10.",
            "P4": "At the independently discovery-selected intermediate layer, both confirmation splits show signed context-pair geometry advantage at least 0.05 at step143000 and step0-to-final gain at least 0.05.",
            "P5": "Only joint P3 and P4 success may authorize a separately preregistered component-stage proposal; this phase never selects heads, neurons, or interventions.",
            "P6": "A positive event is an intermediate residual readout or projected context-pair event under this WordNet/Pythia interface, not pure semantics, execution, causality, or cross-model conservation.",
        },
        "model_outputs_read_during_protocol": False,
        "forbidden_actions": [
            "change thresholds after any Phase1120 hidden output is read",
            "select a layer using confirmation or heldout data",
            "treat terminal logit reproduction as a mechanism discovery",
            "run attention-head, neuron, patch, ablation, or restoration scans in this phase",
            "reopen the closed exact-key registry",
        ],
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", prereg)

    checks = {
        "source_protocol_audit": source_protocol_audit["all_checks_passed"],
        "source_trajectory_complete": source_final["trajectory_authorized"] and source_final["full_trajectory_complete"],
        "source_result_audit": source_result_audit["all_checks_passed"],
        "source_case_digest": digest(rows) == source_prereg["case_digest"],
        "case_count_684": len(rows) == 684,
        "pair_count_342": len(pair_counts) == 342 and all(value == 2 for value in pair_counts.values()),
        "split_case_balance": split_counts == Counter({split: 228 for split in SPLITS}),
        "split_concept_balance": concept_counts == Counter({split: 19 for split in SPLITS}),
        "sense_pairing": all(sorted(row["sense"] for row in rows if row["pair_id"] == pair_id) == [0, 1] for pair_id in pair_counts),
        "checkpoint_subset_integrity": all(name in source_integrity["checkpoints"] for name in CHECKPOINTS),
        "checkpoint_files_present": all((MODEL_ROOT / name / WEIGHT_FORMAT).exists() for name in CHECKPOINTS),
        "projection_shape": projection.shape == (HIDDEN_SIZE, PROJECTION_DIM),
        "projection_finite": bool(np.isfinite(projection).all()),
        "projection_digest": file_sha256(projection_path) == prereg["projection"]["sha256"],
        "terminal_layer_excluded_from_selection": HIDDEN_STATE_COUNT - 1 not in ELIGIBLE_LAYER_INDICES,
        "three_disjoint_splits": set(SPLITS) == set(split_counts) and len(SPLITS) == 3,
        "protocol_digest": digest(prereg_core) == prereg["protocol_digest"],
        "outputs_unread": not prereg["model_outputs_read_during_protocol"],
    }
    audit_core = {
        "schema_version": "phase1120_pythia_hidden_formation_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    audit = dict(audit_core)
    audit["audit_digest"] = digest(audit_core)
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1120 protocol audit failed")
    return {"preregistration": prereg, "audit": audit}


if __name__ == "__main__":
    result = freeze()
    print(json.dumps(result, ensure_ascii=False, indent=2))
