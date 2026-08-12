from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


PHASE = 1125
PROTOCOL_REVISION = 2
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "hf" / "pythia-1.4b-deduped" / "step143000"
SOURCE_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1121_wordnet_adjective_double_orthogonal"
SOURCE_CASE_PATH = SOURCE_ROOT / "protocol" / "cases.pythia.jsonl"
SOURCE_FINAL_PATH = SOURCE_ROOT / "analysis" / "final_summary.json"
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1125_pythia_controlled_bridge_calibration"

MODEL_HIDDEN_SIZE = 2048
MODEL_LAYER_COUNT = 24
ADAPTER_LAYER_INDEX = 12
ADAPTER_RANK = 16
PROJECTION_DIMENSION = 256
PROJECTION_SEED = 1125003
PAD_TOKEN_ID = 0
TRAIN_SPLITS = ("discovery", "independent_confirmation")
TRAIN_TEMPLATES = (0, 1, 2, 3)
CALIBRATION_TEMPLATES = (4, 5)
TRANSFER_SPLIT = "heldout"

TRAINING = {
    "epochs": 6,
    "learning_rate": 0.002,
    "weight_decay": 0.0,
    "gradient_clip_norm": 1.0,
    "delta_l2_weight": 0.0001,
    "arms": {
        "behavior_only": {"seed": 1125001, "bridge_loss_weight": 0.0},
        "bridge_forced": {"seed": 1125002, "bridge_loss_weight": 2.0},
    },
}

THRESHOLDS = {
    "maximum_nonfinite_training_steps": 0,
    "minimum_behavior_only_calibration_accuracy": 0.75,
    "minimum_behavior_only_calibration_accuracy_gain": 0.20,
    "minimum_forced_calibration_accuracy": 0.75,
    "minimum_forced_calibration_projected_cd_cosine": 0.50,
    "minimum_forced_calibration_projected_cd_positive_rate": 0.80,
    "minimum_forced_calibration_cd_gain_over_base": 0.30,
    "minimum_forced_calibration_cd_gain_over_behavior_only": 0.20,
    "maximum_full_projection_median_gap": 0.15,
    "minimum_forced_transfer_accuracy": 0.60,
    "minimum_forced_transfer_projected_cd_cosine": 0.20,
    "minimum_forced_transfer_projected_cd_positive_rate": 0.65,
    "minimum_forced_transfer_cd_gain_over_base": 0.15,
}


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def role_index(tokenizer: Any, prompt: str, input_ids: list[int], fragment: str) -> tuple[int, str]:
    if prompt.count(fragment) != 1:
        raise RuntimeError(f"Role fragment is not unique: {fragment!r}")
    start = prompt.index(fragment)
    end = start + len(fragment)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != input_ids:
        raise RuntimeError("Tokenizer reproduction mismatch")
    overlapping = [
        index
        for index, (left, right) in enumerate(encoded["offset_mapping"])
        if right > start and left < end
    ]
    if not overlapping:
        raise RuntimeError(f"No token overlaps role fragment: {fragment!r}")
    index = max(overlapping)
    prefix_digest = hashlib.sha256(
        json.dumps(input_ids[: index + 1], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return index, prefix_digest


def projection_matrix() -> np.ndarray:
    generator = np.random.default_rng(PROJECTION_SEED)
    signs = generator.integers(0, 2, size=(MODEL_HIDDEN_SIZE, PROJECTION_DIMENSION), dtype=np.int8)
    return ((signs.astype(np.float32) * 2.0) - 1.0) / np.sqrt(PROJECTION_DIMENSION)


def interaction_partition(row: dict[str, Any]) -> str:
    if row["split"] in TRAIN_SPLITS and int(row["template"]) in TRAIN_TEMPLATES:
        return "train"
    if row["split"] in TRAIN_SPLITS and int(row["template"]) in CALIBRATION_TEMPLATES:
        return "calibration"
    if row["split"] == TRANSFER_SPLIT:
        return "transfer"
    raise RuntimeError(f"Unassigned case: {row['case_index']}")


def main() -> None:
    source_final = read_json(SOURCE_FINAL_PATH)
    if source_final["final_digest"] != "3e432fbae5129f62a299f1fccd713a85f9bd696cc3c63e2ac489e28a08dcc5ef":
        raise RuntimeError("Unexpected Phase1121 final digest")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    source_rows = read_jsonl(SOURCE_CASE_PATH)
    if len(source_rows) != 1152:
        raise RuntimeError("Unexpected Phase1121 Pythia case count")

    derived_rows: list[dict[str, Any]] = []
    role_checks: list[bool] = []
    for row in source_rows:
        context_index, context_digest = role_index(
            tokenizer, row["rendered_prompt"], row["input_ids"], row["sentence"]
        )
        definition_index, definition_digest = role_index(
            tokenizer, row["rendered_prompt"], row["input_ids"], row["definition"]
        )
        query_index = int(row["query_position"])
        partition = interaction_partition(row)
        role_checks.extend([
            0 <= context_index < definition_index < query_index,
            query_index == len(row["input_ids"]) - 1,
        ])
        derived_rows.append({
            **row,
            "phase1125_partition": partition,
            "role_indices": {
                "context_end": context_index,
                "definition_end": definition_index,
                "answer_boundary": query_index,
            },
            "role_prefix_digests": {
                "context_end": context_digest,
                "definition_end": definition_digest,
            },
        })

    interaction_groups: dict[str, list[dict[str, Any]]] = {}
    for row in derived_rows:
        interaction_groups.setdefault(row["interaction_id"], []).append(row)
    interaction_counts: dict[str, int] = {partition: 0 for partition in ("train", "calibration", "transfer")}
    case_counts: dict[str, int] = {partition: 0 for partition in interaction_counts}
    concept_sets: dict[str, set[str]] = {partition: set() for partition in interaction_counts}
    for interaction_id, rows in interaction_groups.items():
        if len(rows) != 4:
            raise RuntimeError(f"Malformed quartet: {interaction_id}")
        if {(int(row["context_sense"]), int(row["definition_sense"])) for row in rows} != {
            (0, 0), (0, 1), (1, 0), (1, 1)
        }:
            raise RuntimeError(f"Incomplete quartet: {interaction_id}")
        partitions = {row["phase1125_partition"] for row in rows}
        if len(partitions) != 1:
            raise RuntimeError(f"Quartet crosses partitions: {interaction_id}")
        partition = next(iter(partitions))
        interaction_counts[partition] += 1
        case_counts[partition] += 4
        concept_sets[partition].add(rows[0]["concept_id"])

    projection = projection_matrix()
    projection_path = OUT_ROOT / "protocol" / "projection.pythia.npy"
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(projection_path, projection, allow_pickle=False)
    case_path = OUT_ROOT / "protocol" / "cases.pythia.jsonl"
    write_jsonl(case_path, derived_rows)

    model_weight_path = MODEL_PATH / "model.safetensors"
    source_hashes = {
        "phase1121_cases": file_sha256(SOURCE_CASE_PATH),
        "phase1121_final": file_sha256(SOURCE_FINAL_PATH),
        "model_weights": file_sha256(model_weight_path),
        "derived_cases": file_sha256(case_path),
        "projection": file_sha256(projection_path),
    }
    audit_checks = {
        "source_final_digest_frozen": source_final["final_digest"]
        == "3e432fbae5129f62a299f1fccd713a85f9bd696cc3c63e2ac489e28a08dcc5ef",
        "source_case_count": len(source_rows) == 1152,
        "all_roles_ordered": all(role_checks),
        "all_quartets_complete": len(interaction_groups) == 288,
        "train_interactions_exact": interaction_counts["train"] == 128,
        "calibration_interactions_exact": interaction_counts["calibration"] == 64,
        "transfer_interactions_exact": interaction_counts["transfer"] == 96,
        "train_calibration_concepts_match": concept_sets["train"] == concept_sets["calibration"],
        "transfer_concepts_disjoint": not (concept_sets["train"] & concept_sets["transfer"]),
        "projection_shape": projection.shape == (MODEL_HIDDEN_SIZE, PROJECTION_DIMENSION),
        "projection_finite": bool(np.isfinite(projection).all()),
        "adapter_parameter_count": 2 * MODEL_HIDDEN_SIZE * ADAPTER_RANK == 65536,
    }
    if not all(audit_checks.values()):
        raise RuntimeError(f"Phase1125 protocol audit failed: {audit_checks}")

    preregistration: dict[str, Any] = {
        "schema_version": "phase1125_pythia_controlled_bridge_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase1121_final_digest": source_final["final_digest"],
        "source_hashes": source_hashes,
        "model": {
            "path": str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "architecture": "GPTNeoXForCausalLM",
            "parameter_scale": "1.4B",
            "hidden_size": MODEL_HIDDEN_SIZE,
            "hidden_layers": MODEL_LAYER_COUNT,
            "precision": "frozen base fp16; trainable adapter fp32; no quantization",
            "padding_token_id": PAD_TOKEN_ID,
            "padding_token": "<|endoftext|>",
        },
        "adapter": {
            "insertion": f"forward output of zero-based GPTNeoX block {ADAPTER_LAYER_INDEX}",
            "layer_index": ADAPTER_LAYER_INDEX,
            "rank": ADAPTER_RANK,
            "activation": "GELU",
            "parameter_count": 2 * MODEL_HIDDEN_SIZE * ADAPTER_RANK,
            "base_parameters_frozen": True,
            "up_projection_initialized_to_zero": True,
        },
        "training": TRAINING,
        "partitions": {
            "train": {
                "splits": list(TRAIN_SPLITS),
                "templates": list(TRAIN_TEMPLATES),
                "concept_count": len(concept_sets["train"]),
                "interaction_count": interaction_counts["train"],
                "case_count": case_counts["train"],
            },
            "calibration": {
                "splits": list(TRAIN_SPLITS),
                "templates": list(CALIBRATION_TEMPLATES),
                "concept_count": len(concept_sets["calibration"]),
                "interaction_count": interaction_counts["calibration"],
                "case_count": case_counts["calibration"],
            },
            "transfer": {
                "split": TRANSFER_SPLIT,
                "templates": [0, 1, 2, 3, 4, 5],
                "concept_count": len(concept_sets["transfer"]),
                "interaction_count": interaction_counts["transfer"],
                "case_count": case_counts["transfer"],
            },
        },
        "losses": {
            "behavior": "candidate cross entropy over the frozen single-token true/false IDs at answer_boundary",
            "bridge": (
                "1-cos(C_context_end,D_definition_end), computed in the full 2048-dimensional final normalized "
                "hidden state for each quartet"
            ),
            "delta_regularization": "mean squared adapter residual at the insertion layer",
        },
        "evaluation": {
            "primary_state": "final_layer_norm output",
            "full_dimension": MODEL_HIDDEN_SIZE,
            "independent_projection_dimension": PROJECTION_DIMENSION,
            "projection_seed": PROJECTION_SEED,
            "projection_path": str(projection_path.relative_to(ROOT)).replace("\\", "/"),
            "metrics": [
                "candidate accuracy",
                "truth-balanced behavior interaction",
                "full and projected C-D cosine per quartet",
                "projected C-D positive rate",
                "projected concept Gram relation geometry",
            ],
        },
        "thresholds": THRESHOLDS,
        "predictions": {
            "P1": "Both fixed adapter arms finish every step with finite loss and gradients.",
            "P2": "Behavior-only training improves calibration candidate accuracy above the frozen absolute and gain gates.",
            "P3": "The explicitly forced bridge is visible on unseen templates in the independent 256D projection, above base and behavior-only, with full/projection agreement.",
            "P4": "The forced bridge and behavior transfer to the eight completely unseen heldout concepts.",
            "P5": "Behavior-only learning is separately compared with bridge formation; neither arm is treated as a natural semantic mechanism.",
        },
        "evidence_level": (
            "Method/instrument calibration only. A forced positive can validate visibility of an engineered bridge; "
            "it cannot establish that natural semantic behavior uses that bridge."
        ),
        "scope_limit": (
            "This phase does not test Qwen3/GLM4/DS7B components, natural training formation, non-WordNet "
            "generalization, attention use, necessity, sufficiency, or biological optimality."
        ),
        "forbidden_actions": [
            "change epochs, loss weights, partitions, adapter layer/rank, seeds, thresholds, or projection after training begins",
            "select a checkpoint or epoch by calibration or transfer metrics",
            "train any base-model parameter",
            "drop an arm, template, surface, concept, or failed transfer result",
            "call forced-bridge success a natural mechanism",
            "run component or causal discovery from this calibration result",
        ],
        "engineering_smoke": {
            "performed_before_protocol": True,
            "scientific_metrics_read": False,
            "single_step_gradient_finite": True,
            "peak_allocated_memory_gb": 2.731,
            "purpose": "hook/backward/memory feasibility only",
        },
        "revision_note": (
            "Revision 2 freezes physical token ID 0 (<|endoftext|>) as right-padding after Transformers 5.x "
            "reported None for Pythia pad/eos metadata. Revision 1 stopped before base evaluation or any scientific "
            "metric was produced. Cases, partitions, losses, seeds, epochs, thresholds, and evaluation objects are unchanged."
        ),
    }
    preregistration["protocol_digest"] = canonical_digest(preregistration)

    audit = {
        "schema_version": "phase1125_pythia_controlled_bridge_protocol_audit.v1",
        "phase": PHASE,
        "checks": audit_checks,
        "interaction_counts": interaction_counts,
        "case_counts": case_counts,
        "concept_counts": {key: len(value) for key, value in concept_sets.items()},
        "protocol_digest": preregistration["protocol_digest"],
    }
    audit["audit_digest"] = canonical_digest(audit)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", preregistration)
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "audit_digest": audit["audit_digest"],
        "all_checks_passed": all(audit_checks.values()),
        "partitions": preregistration["partitions"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
