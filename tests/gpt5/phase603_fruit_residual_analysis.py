#!/usr/bin/env python3
"""Find discovery-selected residual units that repeat across roles, surfaces and splits."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase603_fruit_residual_protocol as protocol  # noqa: E402
from phase603_fruit_residual_extract import output_paths  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_role_surface_effects(
    activations: np.ndarray,
    metadata: list[dict[str, Any]],
    track: str,
    split: str,
) -> np.ndarray:
    # [role, surface, layer, unit], with a shared per-layer/unit absolute-activation scale.
    selected = np.array([row["track"] == track and row["split"] == split for row in metadata])
    scale = np.mean(np.abs(activations[selected].astype(np.float32)), axis=0) + 1e-6
    effects = []
    for role in protocol.NONFRUIT_ROLES:
        role_effects = []
        for surface in protocol.SURFACES:
            fruit_mask = np.array([
                row["track"] == track and row["split"] == split and row["surface_id"] == surface
                and row["fruit_member"] for row in metadata
            ])
            role_mask = np.array([
                row["track"] == track and row["split"] == split and row["surface_id"] == surface
                and row["entity_role"] == role for row in metadata
            ])
            if not fruit_mask.any() or not role_mask.any():
                raise RuntimeError(f"Missing Phase603 group: {track}/{split}/{role}/{surface}")
            effect = (
                activations[fruit_mask].astype(np.float32).mean(axis=0)
                - activations[role_mask].astype(np.float32).mean(axis=0)
            ) / scale
            role_effects.append(effect)
        effects.append(np.stack(role_effects))
    return np.stack(effects)


def analyze_branch(activations: np.ndarray, metadata: list[dict[str, Any]], track: str) -> dict[str, Any]:
    effects = {
        split: normalized_role_surface_effects(activations, metadata, track, split)
        for split in protocol.SPLITS
    }
    discovery = effects["discovery"]
    sign_sum = np.sign(discovery).sum(axis=(0, 1))
    required = len(protocol.NONFRUIT_ROLES) * len(protocol.SURFACES)
    sign_consistent = np.abs(sign_sum) == required
    discovery_min = np.min(np.abs(discovery), axis=(0, 1))
    eligible = sign_consistent & (discovery_min >= protocol.DISCOVERY_MIN_NORMALIZED_EFFECT)
    candidates: list[dict[str, Any]] = []
    layer_count, hidden_size = eligible.shape
    for layer in range(layer_count):
        units = np.flatnonzero(eligible[layer])
        units = sorted(units, key=lambda unit: (-float(discovery_min[layer, unit]), int(unit)))
        for unit in units[: protocol.TOP_UNITS_PER_LAYER]:
            direction = 1 if sign_sum[layer, unit] > 0 else -1
            record: dict[str, Any] = {
                "layer": layer,
                "unit": int(unit),
                "direction": direction,
                "normalized_depth": layer / max(1, layer_count - 1),
                "discovery_min_normalized_effect": float(discovery_min[layer, unit]),
            }
            stable = True
            for split, minimum in (
                ("independent_confirmation", protocol.CONFIRMATION_MIN_NORMALIZED_EFFECT),
                ("heldout", protocol.HELDOUT_MIN_NORMALIZED_EFFECT),
            ):
                values = effects[split][:, :, layer, unit] * direction
                record[f"{split}_min_directional_effect"] = float(np.min(values))
                record[f"{split}_all_role_surface_directions"] = bool(np.all(values > 0))
                record[f"{split}_threshold_pass"] = bool(np.min(values) >= minimum)
                stable = stable and bool(np.min(values) >= minimum)
            record["strictly_repeated"] = stable
            candidates.append(record)
    repeated = [row for row in candidates if row["strictly_repeated"]]
    repeated.sort(key=lambda row: (-min(
        row["discovery_min_normalized_effect"],
        row["independent_confirmation_min_directional_effect"],
        row["heldout_min_directional_effect"],
    ), row["layer"], row["unit"]))
    return {
        "track": track,
        "layer_count_including_embedding_output": layer_count,
        "hidden_size": hidden_size,
        "discovery_eligible_unit_count": int(eligible.sum()),
        "discovery_selected_candidate_count": len(candidates),
        "strictly_repeated_candidate_count": len(repeated),
        "strict_repeat_rate": len(repeated) / max(1, len(candidates)),
        "repeated_count_by_depth_band": dict(Counter(
            "early" if row["normalized_depth"] < 1 / 3 else "middle" if row["normalized_depth"] < 2 / 3 else "late"
            for row in repeated
        )),
        "repeated_candidates": repeated,
        "all_discovery_selected_candidates": candidates,
    }


def analyze() -> dict[str, Any]:
    branches: dict[str, Any] = {}
    repeated_sets: dict[str, set[tuple[int, int]]] = {}
    source_hashes: dict[str, Any] = {}
    for model, tracks in protocol.QUALIFIED_BRANCHES.items():
        paths = output_paths(model)
        summary = json.loads(paths["summary"].read_text())
        if summary.get("status") != "complete":
            raise RuntimeError(f"Missing Phase603 extraction for {model}")
        if summary["arrays_sha256"] != sha256_file(paths["arrays"]):
            raise RuntimeError(f"Phase603 array drift for {model}")
        metadata = json.loads(paths["metadata"].read_text())
        with np.load(paths["arrays"]) as stored:
            activations = stored["activations"]
        source_hashes[model] = {
            "arrays_sha256": summary["arrays_sha256"],
            "metadata_sha256": summary["metadata_sha256"],
            "array_shape": summary["array_shape"],
        }
        for track in tracks:
            key = f"{model}/{track}"
            result = analyze_branch(activations, metadata, track)
            branches[key] = result
            repeated_sets[key] = {(row["layer"], row["unit"]) for row in result["repeated_candidates"]}
    qwen_keys = [f"qwen3/{track}" for track in protocol.QUALIFIED_BRANCHES["qwen3"]]
    overlap = set.intersection(*(repeated_sets[key] for key in qwen_keys)) if qwen_keys else set()
    pairwise = {}
    for index, left in enumerate(qwen_keys):
        for right in qwen_keys[index + 1 :]:
            union = repeated_sets[left] | repeated_sets[right]
            pairwise[f"{left}|{right}"] = {
                "intersection_count": len(repeated_sets[left] & repeated_sets[right]),
                "union_count": len(union),
                "jaccard": len(repeated_sets[left] & repeated_sets[right]) / max(1, len(union)),
            }
    payload = {
        "schema_version": "phase603_fruit_residual_analysis.v1",
        "phase_id": protocol.PHASE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "source_hashes": source_hashes,
        "branches": branches,
        "qwen_pairwise_exact_unit_overlap": pairwise,
        "qwen_all_three_track_exact_unit_overlap_count": len(overlap),
        "qwen_all_three_track_exact_units": [list(value) for value in sorted(overlap)],
        "cross_model_unit_identity_comparison_authorized": False,
        "causal_intervention_authorized": False,
        "mechanism_claim_authorized": False,
        "theory_or_formula_update_authorized": False,
        "evidence_boundary": (
            "Observer-only answer-boundary residual candidates. Strict repetition means direction "
            "survived every registered role and surface in discovery, confirmation and heldout; "
            "it does not establish necessity, sufficiency, category exclusivity, or a causal code."
        ),
    }
    protocol.ANALYSIS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2, sort_keys=True))
