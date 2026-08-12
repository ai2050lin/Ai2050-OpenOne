"""Role decomposition of the Phase 1198 expanded rescue controller.

The experiment freezes a nested candidate family before new data: token
embedding, position embedding, their pair, the pair plus the first attention
or first full block, the four formerly omitted blocks, and the full expanded
coalition.  Sufficiency is paired with leave-role-out necessity and matched
negative, same-support random, and wrong-task embedding controls.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from phase1146_learned_composition_benchmark import TinyCausalTransformer  # noqa: E402
import phase1193_tiny_transformer_quotient_causal_bridge as p1193  # noqa: E402
import phase1194_natural_minibatch_tangent_and_minimal_rescue as p1194  # noqa: E402
import phase1195_continuous_sparse_coalition_rescue as p1195  # noqa: E402
import phase1197_rescue_failure_tomography as p1197  # noqa: E402
import phase1198_expanded_partition_sparse_rescue as p1198  # noqa: E402


PHASE = 1199
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1199_expanded_rescue_role_decomposition_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1199_expanded_rescue_role_decomposition"
DEVELOPMENT_ROWS = OUT_ROOT / "development/rows.jsonl"
DEVELOPMENT_SUMMARY = OUT_ROOT / "development/summary.json"
DEVELOPMENT_AUDIT = OUT_ROOT / "development/independent_audit.json"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
FORMAL_ROW_ROOT = OUT_ROOT / "runs/formal/rows"
REPLAY_ROOT = OUT_ROOT / "runs/formal/replay_capsules"
TRAINING_SEAL = OUT_ROOT / "runs/formal/seal.json"
RAW_ROWS = OUT_ROOT / "analysis/rows.jsonl"
SUMMARY_PATH = OUT_ROOT / "analysis/summary.json"
CLAIMS_PATH = OUT_ROOT / "analysis/typed_claims.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"

ARCHITECTURES = p1198.ARCHITECTURES
STAGE = p1198.RESCUE_STAGE
BATCH_SIZE = p1198.BATCH_SIZE
SUPPORT_EPSILON = p1198.SUPPORT_EPSILON
DEVELOPMENT_REPLICATES = 2
FORMAL_REPLICATES = 4

DEVELOPMENT_TASKS = (
    {"name": "role_dev_affine_00", "family": "affine", "task_seed": 1_199_011},
    {"name": "role_dev_affine_01", "family": "affine", "task_seed": 1_199_017},
    {"name": "role_dev_bitmix_00", "family": "bitmix", "task_seed": 1_199_023},
    {"name": "role_dev_bitmix_01", "family": "bitmix", "task_seed": 1_199_029},
    {"name": "role_dev_random_00", "family": "random", "task_seed": 1_199_037},
    {"name": "role_dev_random_01", "family": "random", "task_seed": 1_199_043},
)

FORMAL_TASKS = (
    {"name": "role_disc_affine_00", "split": "discovery", "family": "affine", "task_seed": 1_199_101},
    {"name": "role_disc_affine_01", "split": "discovery", "family": "affine", "task_seed": 1_199_107},
    {"name": "role_disc_bitmix_00", "split": "discovery", "family": "bitmix", "task_seed": 1_199_113},
    {"name": "role_disc_bitmix_01", "split": "discovery", "family": "bitmix", "task_seed": 1_199_119},
    {"name": "role_disc_random_00", "split": "discovery", "family": "random", "task_seed": 1_199_127},
    {"name": "role_disc_random_01", "split": "discovery", "family": "random", "task_seed": 1_199_133},
    {"name": "role_conf_affine_00", "split": "confirmation", "family": "affine", "task_seed": 1_199_203},
    {"name": "role_conf_affine_01", "split": "confirmation", "family": "affine", "task_seed": 1_199_209},
    {"name": "role_conf_bitmix_00", "split": "confirmation", "family": "bitmix", "task_seed": 1_199_217},
    {"name": "role_conf_bitmix_01", "split": "confirmation", "family": "bitmix", "task_seed": 1_199_223},
    {"name": "role_conf_random_00", "split": "confirmation", "family": "random", "task_seed": 1_199_231},
    {"name": "role_conf_random_01", "split": "confirmation", "family": "random", "task_seed": 1_199_239},
)

CANDIDATE_ORDER = (
    "token_embedding",
    "position_embedding",
    "embedding_pair",
    "embedding_front_attention",
    "embedding_front_block",
    "omitted_quartet",
)

THRESHOLDS = {
    "eligible_fraction_min": 0.95,
    "full_recovery_mean_min": 0.50,
    "embedding_pair_recovery_mean_min": 0.40,
    "embedding_selectivity_advantage_mean_min": 0.25,
    "embedding_selectivity_positive_fraction_min": 0.90,
    "embedding_necessity_mean_min": 0.25,
    "token_necessity_mean_min": 0.08,
    "position_necessity_mean_min": 0.08,
    "minimal_candidate_success_fraction_min": 0.75,
    "minimal_candidate_parameter_fraction_mean_max": 0.15,
    "candidate_recovery_min": 0.45,
    "candidate_full_gap_max": 0.15,
    "architecture_embedding_recovery_min": 0.30,
    "architecture_embedding_necessity_min": 0.15,
    "architecture_selectivity_advantage_min": 0.15,
    "architecture_selectivity_positive_fraction_min": 0.80,
    "family_embedding_recovery_min": 0.30,
    "family_selectivity_advantage_min": 0.10,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_seed(task_index: int, architecture: str, replicate: int, corpus: str) -> int:
    base = 1_199_900_000 if corpus == "development" else 1_199_000_000
    return base + task_index * 100_003 + list(ARCHITECTURES).index(architecture) * 10_007 + replicate * 1_009


def build_payload(
    task: dict[str, Any], task_index: int, architecture: str, replicate: int, corpus: str, device: torch.device
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = model_seed(task_index, architecture, replicate, corpus)
    set_seed(seed)
    inputs, targets, candidates, calibration, evaluation = p1194.make_data(
        int(task["task_seed"]), str(task["family"]), device
    )
    model = TinyCausalTransformer(ARCHITECTURES[architecture]).to(device)
    optimizer = p1193.optimizer_for(model)
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    batches = [
        torch.randint(0, len(inputs), (BATCH_SIZE,), generator=generator).to(device)
        for _ in range(STAGE + 1)
    ]
    for step in range(STAGE):
        p1193.training_step(model, optimizer, inputs[batches[step]], targets[batches[step]], candidates)
    payload = p1195.build_material(
        model,
        optimizer,
        inputs,
        targets,
        candidates,
        calibration,
        evaluation,
        batches[STAGE],
        seed + STAGE * 1009,
    )
    trajectory_id = f"{task['name']}::{architecture}::r{replicate}"
    payload.update(
        {
            "task": dict(task),
            "task_index": task_index,
            "architecture": architecture,
            "replicate": replicate,
            "trajectory_id": trajectory_id,
            "model_seed": seed,
        }
    )
    solution, _ = p1198.solve_payload(payload, device, seed + STAGE * 2003 + 43)
    payload.update(
        {
            "expanded_patch": solution["patch"].detach().cpu(),
            "expanded_alpha": solution["alpha"].tolist(),
            "expanded_group_names": solution["group_names"],
            "expanded_support_count": solution["support_count"],
            "expanded_support_parameter_fraction": solution["support_parameter_fraction"],
            "expanded_patch_update_fraction": solution["patch_update_fraction"],
            "expanded_partition": solution["partition"],
        }
    )
    row = {
        "trajectory_id": trajectory_id,
        "event_id": f"{trajectory_id}::s{STAGE}",
        "task_name": task["name"],
        "task_index": task_index,
        "task_seed": task["task_seed"],
        "family": task["family"],
        "split": task.get("split", "development"),
        "architecture": architecture,
        "replicate": replicate,
        "model_seed": seed,
        "stage": STAGE,
        "event_loss": payload["event_loss"],
        "real_child_accuracy": payload["real_child_accuracy"],
        "control_match": payload["control_metrics"],
        "expanded_alpha": payload["expanded_alpha"],
        "expanded_group_names": payload["expanded_group_names"],
        "expanded_support_count": payload["expanded_support_count"],
        "expanded_support_parameter_fraction": payload["expanded_support_parameter_fraction"],
        "expanded_patch_update_fraction": payload["expanded_patch_update_fraction"],
        "expanded_partition": payload["expanded_partition"],
    }
    del model, optimizer, inputs, targets, candidates, batches
    gc.collect()
    torch.cuda.empty_cache()
    return row, payload


def sum_patches(parts: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    reference = next(iter(parts.values()))
    result = torch.zeros_like(reference)
    for name in names:
        result += parts[name]
    return result


def role_material(
    payload: dict[str, Any], device: torch.device, seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    parent = TinyCausalTransformer(ARCHITECTURES[payload["architecture"]]).to(device)
    parent.load_state_dict(payload["parent_state"])
    difference = payload["difference"].to(device)
    groups = p1198.expanded_component_masks(parent)
    alpha = np.asarray(payload["expanded_alpha"], dtype=np.float64)
    weighted: dict[str, torch.Tensor] = {}
    selected_masks: dict[str, torch.Tensor] = {}
    for coefficient, (name, mask) in zip(alpha, groups):
        weighted[name] = float(coefficient) * torch.where(mask, difference, torch.zeros_like(difference))
        selected_masks[name] = mask if coefficient > SUPPORT_EPSILON else torch.zeros_like(mask)
    core_names = [name for name, _ in groups if name.startswith("layer")]
    omitted_names = ["token_embedding", "position_embedding", "final_norm", "lm_head"]
    embedding_names = ["token_embedding", "position_embedding"]
    front_attention_names = ["layer0.attn"]
    front_block_names = ["layer0.attn", "layer0.mlp"]
    full = sum_patches(weighted, list(weighted))
    embeddings = sum_patches(weighted, embedding_names)
    patches = {
        "full": full,
        "token_embedding": weighted["token_embedding"],
        "position_embedding": weighted["position_embedding"],
        "embedding_pair": embeddings,
        "core": sum_patches(weighted, core_names),
        "omitted_quartet": sum_patches(weighted, omitted_names),
        "front_attention": sum_patches(weighted, front_attention_names),
        "front_block": sum_patches(weighted, front_block_names),
        "embedding_front_attention": sum_patches(weighted, embedding_names + front_attention_names),
        "embedding_front_block": sum_patches(weighted, embedding_names + front_block_names),
        "full_without_token": full - weighted["token_embedding"],
        "full_without_position": full - weighted["position_embedding"],
        "full_without_embeddings": full - embeddings,
        "full_without_front_attention": full - weighted["layer0.attn"],
        "full_without_core": full - sum_patches(weighted, core_names),
        "embedding_negative": -embeddings,
    }
    embedding_support = selected_masks["token_embedding"] | selected_masks["position_embedding"]
    generator = torch.Generator(device=device).manual_seed(seed)
    random_patch = torch.zeros_like(difference)
    values = torch.randn(int(embedding_support.sum().item()), generator=generator, device=device)
    random_patch[embedding_support] = p1194.scaled_like(values, embeddings.norm())
    patches["embedding_random"] = random_patch

    candidate_names = {
        "token_embedding": ["token_embedding"],
        "position_embedding": ["position_embedding"],
        "embedding_pair": embedding_names,
        "embedding_front_attention": embedding_names + front_attention_names,
        "embedding_front_block": embedding_names + front_block_names,
        "omitted_quartet": omitted_names,
        "full": list(weighted),
    }
    parameter_fractions: dict[str, float] = {}
    for candidate, names in candidate_names.items():
        mask = torch.zeros_like(difference, dtype=torch.bool)
        for name in names:
            mask |= selected_masks[name]
        parameter_fractions[candidate] = float(mask.float().mean().item())
    del parent
    return {name: patch.detach().cpu() for name, patch in patches.items()}, parameter_fractions


@torch.inference_mode()
def evaluate_patches(
    payload: dict[str, Any], patches: dict[str, torch.Tensor], device: torch.device
) -> dict[str, dict[str, float]]:
    task = payload["task"]
    inputs, targets, candidates, _, evaluation = p1194.make_data(
        int(task["task_seed"]), str(task["family"]), device
    )
    parent = TinyCausalTransformer(ARCHITECTURES[payload["architecture"]]).to(device)
    parent.load_state_dict(payload["parent_state"])
    parent_vector = payload["parent_vector"].to(device)
    control_update = payload["control_update"].to(device)
    real = np.asarray(payload["real_child_q"], dtype=np.float64)

    def response(patch: torch.Tensor) -> np.ndarray:
        model = p1194.clone_model(parent)
        p1193.assign_parameters(model, parent_vector + control_update + patch.to(device))
        value = p1193.quotient_response(model, inputs[evaluation], targets[evaluation], candidates)
        del model
        return value

    control = response(torch.zeros_like(payload["expanded_patch"]))
    control_error = float(np.linalg.norm(control - real))
    measured = {
        "control": {"response_error": control_error, "response_recovery": 0.0}
    }
    for name, patch in patches.items():
        error = float(np.linalg.norm(response(patch) - real))
        measured[name] = {
            "response_error": error,
            "response_recovery": (control_error - error) / max(control_error, 1e-12),
        }
    del parent, inputs, targets, candidates
    return measured


def diagnose_payload(
    row: dict[str, Any], payload: dict[str, Any], device: torch.device
) -> None:
    patches, parameter_fractions = role_material(payload, device, int(payload["model_seed"]) + 911)
    measured = evaluate_patches(payload, patches, device)
    full_recovery = measured["full"]["response_recovery"]
    qualifying = [
        name
        for name in CANDIDATE_ORDER
        if measured[name]["response_recovery"] >= THRESHOLDS["candidate_recovery_min"]
        and full_recovery - measured[name]["response_recovery"] <= THRESHOLDS["candidate_full_gap_max"]
    ]
    minimal = min(
        qualifying,
        key=lambda name: (parameter_fractions[name], CANDIDATE_ORDER.index(name)),
    ) if qualifying else None
    payload["role_patches"] = patches
    payload["role_parameter_fractions"] = parameter_fractions
    row.update(
        {
            "role_variants": measured,
            "role_parameter_fractions": parameter_fractions,
            "full_recovery": full_recovery,
            "token_recovery": measured["token_embedding"]["response_recovery"],
            "position_recovery": measured["position_embedding"]["response_recovery"],
            "embedding_pair_recovery": measured["embedding_pair"]["response_recovery"],
            "core_recovery": measured["core"]["response_recovery"],
            "omitted_quartet_recovery": measured["omitted_quartet"]["response_recovery"],
            "embedding_front_attention_recovery": measured["embedding_front_attention"]["response_recovery"],
            "embedding_front_block_recovery": measured["embedding_front_block"]["response_recovery"],
            "front_attention_increment": measured["embedding_front_attention"]["response_recovery"] - measured["embedding_pair"]["response_recovery"],
            "embedding_necessity": full_recovery - measured["full_without_embeddings"]["response_recovery"],
            "token_necessity": full_recovery - measured["full_without_token"]["response_recovery"],
            "position_necessity": full_recovery - measured["full_without_position"]["response_recovery"],
            "front_attention_necessity": full_recovery - measured["full_without_front_attention"]["response_recovery"],
            "core_necessity": full_recovery - measured["full_without_core"]["response_recovery"],
            "minimal_candidate": minimal,
            "minimal_candidate_success": minimal is not None,
            "minimal_candidate_recovery": measured[minimal]["response_recovery"] if minimal else float("nan"),
            "minimal_candidate_parameter_fraction": parameter_fractions[minimal] if minimal else float("nan"),
            "eligible": bool(
                p1195.control_match_pass(row["control_match"])
                and row["expanded_partition"]["complete"]
                and row["expanded_support_parameter_fraction"] <= p1195.CONTROL_THRESHOLDS["support_parameter_fraction_max"]
                and row["expanded_patch_update_fraction"] <= p1195.CONTROL_THRESHOLDS["patch_update_fraction_max"]
            ),
        }
    )


def attach_wrong_task_controls(
    rows: list[dict[str, Any]], payloads: list[dict[str, Any]], device: torch.device
) -> None:
    by_trajectory = {payload["trajectory_id"]: payload for payload in payloads}
    by_cell = {
        (payload["task"].get("split", "development"), payload["architecture"], payload["replicate"], payload["task_index"]): payload
        for payload in payloads
    }
    split_indices: dict[str, list[int]] = {}
    for payload in payloads:
        split_indices.setdefault(payload["task"].get("split", "development"), []).append(payload["task_index"])
    split_indices = {key: sorted(set(value)) for key, value in split_indices.items()}
    for row in rows:
        payload = by_trajectory[row["trajectory_id"]]
        indices = split_indices[row["split"]]
        next_index = indices[(indices.index(row["task_index"]) + 1) % len(indices)]
        wrong_payload = by_cell[(row["split"], row["architecture"], row["replicate"], next_index)]
        own = payload["role_patches"]["embedding_pair"]
        wrong = p1194.scaled_like(wrong_payload["role_patches"]["embedding_pair"], own.norm())
        measured = evaluate_patches(payload, {"embedding_wrong_task": wrong}, device)["embedding_wrong_task"]
        row["role_variants"]["embedding_wrong_task"] = measured
        payload["role_patches"]["embedding_wrong_task"] = wrong
        payload["wrong_task_trajectory_id"] = wrong_payload["trajectory_id"]
        row["wrong_task_trajectory_id"] = wrong_payload["trajectory_id"]
        null_recovery = max(
            row["role_variants"][name]["response_recovery"]
            for name in ("embedding_negative", "embedding_random", "embedding_wrong_task")
        )
        row["embedding_selectivity_null_recovery"] = null_recovery
        row["embedding_selectivity_advantage"] = row["embedding_pair_recovery"] - null_recovery


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")


def role_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["eligible"]]
    successful = [row for row in eligible if row["minimal_candidate_success"]]
    counts = {name: sum(row["minimal_candidate"] == name for row in successful) for name in CANDIDATE_ORDER}
    return {
        "count": len(rows),
        "eligible_count": len(eligible),
        "full_recovery_mean": mean(eligible, "full_recovery"),
        "token_recovery_mean": mean(eligible, "token_recovery"),
        "position_recovery_mean": mean(eligible, "position_recovery"),
        "embedding_pair_recovery_mean": mean(eligible, "embedding_pair_recovery"),
        "core_recovery_mean": mean(eligible, "core_recovery"),
        "omitted_quartet_recovery_mean": mean(eligible, "omitted_quartet_recovery"),
        "embedding_front_attention_recovery_mean": mean(eligible, "embedding_front_attention_recovery"),
        "embedding_front_block_recovery_mean": mean(eligible, "embedding_front_block_recovery"),
        "front_attention_increment_mean": mean(eligible, "front_attention_increment"),
        "embedding_necessity_mean": mean(eligible, "embedding_necessity"),
        "token_necessity_mean": mean(eligible, "token_necessity"),
        "position_necessity_mean": mean(eligible, "position_necessity"),
        "front_attention_necessity_mean": mean(eligible, "front_attention_necessity"),
        "core_necessity_mean": mean(eligible, "core_necessity"),
        "embedding_selectivity_null_recovery_mean": mean(eligible, "embedding_selectivity_null_recovery"),
        "embedding_selectivity_advantage_mean": mean(eligible, "embedding_selectivity_advantage"),
        "embedding_selectivity_positive_fraction": float(np.mean([row["embedding_selectivity_advantage"] > 0 for row in eligible])) if eligible else 0.0,
        "minimal_candidate_success_fraction": len(successful) / max(len(eligible), 1),
        "minimal_candidate_parameter_fraction_mean": mean(successful, "minimal_candidate_parameter_fraction"),
        "minimal_candidate_counts": counts,
    }


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise RuntimeError(f"no rows for split {split}")
    overall = role_group(selected)
    overall["eligible_fraction"] = overall["eligible_count"] / len(selected)
    by_architecture = {name: role_group([row for row in selected if row["architecture"] == name]) for name in ARCHITECTURES}
    by_family = {name: role_group([row for row in selected if row["family"] == name]) for name in ("affine", "bitmix", "random")}
    gate = bool(
        overall["eligible_fraction"] >= THRESHOLDS["eligible_fraction_min"]
        and overall["full_recovery_mean"] >= THRESHOLDS["full_recovery_mean_min"]
        and overall["embedding_pair_recovery_mean"] >= THRESHOLDS["embedding_pair_recovery_mean_min"]
        and overall["embedding_selectivity_advantage_mean"] >= THRESHOLDS["embedding_selectivity_advantage_mean_min"]
        and overall["embedding_selectivity_positive_fraction"] >= THRESHOLDS["embedding_selectivity_positive_fraction_min"]
        and overall["embedding_necessity_mean"] >= THRESHOLDS["embedding_necessity_mean_min"]
        and overall["token_necessity_mean"] >= THRESHOLDS["token_necessity_mean_min"]
        and overall["position_necessity_mean"] >= THRESHOLDS["position_necessity_mean_min"]
        and overall["minimal_candidate_success_fraction"] >= THRESHOLDS["minimal_candidate_success_fraction_min"]
        and overall["minimal_candidate_parameter_fraction_mean"] <= THRESHOLDS["minimal_candidate_parameter_fraction_mean_max"]
        and all(
            group["embedding_pair_recovery_mean"] >= THRESHOLDS["architecture_embedding_recovery_min"]
            and group["embedding_necessity_mean"] >= THRESHOLDS["architecture_embedding_necessity_min"]
            and group["embedding_selectivity_advantage_mean"] >= THRESHOLDS["architecture_selectivity_advantage_min"]
            and group["embedding_selectivity_positive_fraction"] >= THRESHOLDS["architecture_selectivity_positive_fraction_min"]
            for group in by_architecture.values()
        )
        and all(
            group["embedding_pair_recovery_mean"] >= THRESHOLDS["family_embedding_recovery_min"]
            and group["embedding_selectivity_advantage_mean"] >= THRESHOLDS["family_selectivity_advantage_min"]
            for group in by_family.values()
        )
    )
    return {
        "split": split,
        "row_count": len(selected),
        "trajectory_count": len({row["trajectory_id"] for row in selected}),
        "roles": overall,
        "roles_by_architecture": by_architecture,
        "roles_by_family": by_family,
        "role_gate_pass": gate,
    }


def source_hashes() -> dict[str, str]:
    paths = {
        "phase1199": SCRIPT,
        "phase1199_audit": AUDIT_SCRIPT,
        "phase1198": p1198.SCRIPT,
        "phase1197": p1197.SCRIPT,
        "phase1195": p1195.SCRIPT,
        "phase1194": p1194.SCRIPT,
        "phase1193": p1193.SCRIPT,
        "phase1146_model": ROOT / "tests/glm5/phase1146_learned_composition_benchmark.py",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def run_corpus(tasks: tuple[dict[str, Any], ...], replicates: int, corpus: str, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        for architecture in ARCHITECTURES:
            for replicate in range(replicates):
                row, payload = build_payload(task, task_index, architecture, replicate, corpus, device)
                diagnose_payload(row, payload, device)
                rows.append(row)
                payloads.append(payload)
                print(canonical_json({"corpus": corpus, "task": task["name"], "architecture": architecture, "replicate": replicate, "rows": len(rows)}), flush=True)
    attach_wrong_task_controls(rows, payloads, device)
    if corpus == "formal":
        replay_ids = {
            "role_disc_affine_00::compact::r0",
            "role_disc_affine_00::deep::r0",
            "role_conf_affine_00::compact::r0",
            "role_conf_affine_00::deep::r0",
        }
        REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
        for payload in payloads:
            if payload["trajectory_id"] in replay_ids:
                torch.save(payload, REPLAY_ROOT / f"{payload['trajectory_id'].replace('::', '__')}.pt")
    del payloads
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def develop() -> None:
    if DEVELOPMENT_ROWS.exists() or DEVELOPMENT_SUMMARY.exists():
        raise RuntimeError("Phase1199 development outcomes already exist")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    upstream = read_json(p1198.FINAL_PATH)
    if not upstream["authorized_next"]["expanded_basis_minimality_role_decomposition_development"]:
        raise RuntimeError("Phase1198 did not authorize this development")
    rows = run_corpus(DEVELOPMENT_TASKS, DEVELOPMENT_REPLICATES, "development", torch.device("cuda"))
    summary = summarize(rows, "development")
    output = {
        "phase": PHASE,
        "kind": "authorized_development_only",
        "created_at": utc_now(),
        "development": summary,
        "development_gate_pass": summary["role_gate_pass"],
        "authorized_next": {"formal_preregistration": summary["role_gate_pass"]},
    }
    write_jsonl(DEVELOPMENT_ROWS, rows)
    write_json(DEVELOPMENT_SUMMARY, output)
    print(canonical_json({"development_gate_pass": output["development_gate_pass"], "roles": summary["roles"]}))


def preregister() -> None:
    if PROTOCOL_PATH.exists() or TRAINING_SEAL.exists() or RAW_ROWS.exists():
        raise RuntimeError("Phase1199 protocol or formal outcomes already exist")
    development = read_json(DEVELOPMENT_SUMMARY)
    audit = read_json(DEVELOPMENT_AUDIT)
    if not development["development_gate_pass"] or not audit.get("gate_pass", False):
        raise RuntimeError("development or its independent audit did not pass")
    upstream = read_json(p1198.FINAL_PATH)
    protocol = {
        "phase": PHASE,
        "created_at": utc_now(),
        "question": "Is the Phase1198 rescue carried by a small, role-stable embedding-dominant coalition that is both sufficient and necessary relative to matched direction controls?",
        "scope": "Role decomposition of immediate synthetic TinyTransformer quotient-response rescue; no future-learning or language claim.",
        "architectures": {name: asdict(config) for name, config in ARCHITECTURES.items()},
        "formal_tasks": list(FORMAL_TASKS),
        "formal_replicates": FORMAL_REPLICATES,
        "stage": STAGE,
        "candidate_order": list(CANDIDATE_ORDER),
        "candidate_definition": "fixed fitted coefficients inherited from the full expanded controller; no coefficient refit within candidates",
        "controls": ["sign_reversed_embedding_pair", "same_support_same_norm_random", "same_norm_wrong_task_embedding_pair"],
        "necessity_tests": ["remove_token", "remove_position", "remove_embedding_pair", "remove_front_attention", "remove_all_core"],
        "thresholds": THRESHOLDS,
        "continuation_rule": "No automatic future-learning experiment is authorized. A positive result licenses theory/measurement consolidation only.",
        "forbidden": [
            "refit coefficients separately for role candidates",
            "change candidate order, thresholds, or controls after formal outcomes",
            "select a different early layer after outcomes",
            "drop architectures, families, or negative cases",
            "call patch sufficiency an endogenous learned module",
            "claim behavior, future-learning, or natural-language recovery",
        ],
        "development": {
            "rows_sha256": file_sha256(DEVELOPMENT_ROWS),
            "summary_sha256": file_sha256(DEVELOPMENT_SUMMARY),
            "audit_sha256": file_sha256(DEVELOPMENT_AUDIT),
        },
        "upstream": {
            "phase1198_final_sha256": file_sha256(p1198.FINAL_PATH),
            "phase1198_final_digest": upstream["final_digest"],
        },
        "source_hashes": source_hashes(),
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"protocol_digest": protocol["protocol_digest"]}))


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    candidate = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    if digest(candidate) != protocol["protocol_digest"]:
        raise RuntimeError("protocol digest mismatch")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("source changed after preregistration")
    if file_sha256(p1198.FINAL_PATH) != protocol["upstream"]["phase1198_final_sha256"]:
        raise RuntimeError("Phase1198 final changed")
    for key, path in (("rows_sha256", DEVELOPMENT_ROWS), ("summary_sha256", DEVELOPMENT_SUMMARY), ("audit_sha256", DEVELOPMENT_AUDIT)):
        if file_sha256(path) != protocol["development"][key]:
            raise RuntimeError(f"development asset changed: {path}")
    return protocol


def run_formal() -> None:
    protocol = verify_protocol()
    if TRAINING_SEAL.exists() or RAW_ROWS.exists():
        raise RuntimeError("Phase1199 formal outcomes already exist")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = run_corpus(FORMAL_TASKS, FORMAL_REPLICATES, "formal", torch.device("cuda"))
    FORMAL_ROW_ROOT.mkdir(parents=True, exist_ok=True)
    for row in rows:
        write_json(FORMAL_ROW_ROOT / f"{row['event_id'].replace('::', '__')}.json", row)
    write_jsonl(RAW_ROWS, rows)
    seal = {
        "phase": PHASE,
        "created_at": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "row_count": len(rows),
        "trajectory_count": len({row["trajectory_id"] for row in rows}),
        "analysis_rows_sha256": file_sha256(RAW_ROWS),
        "row_manifest": {path.name: file_sha256(path) for path in sorted(FORMAL_ROW_ROOT.glob("*.json"))},
        "replay_manifest": {path.name: file_sha256(path) for path in sorted(REPLAY_ROOT.glob("*.pt"))},
    }
    seal["seal_digest"] = digest(seal)
    write_json(TRAINING_SEAL, seal)
    print(canonical_json({"row_count": len(rows), "seal_digest": seal["seal_digest"]}))


def analyze() -> None:
    verify_protocol()
    seal = read_json(TRAINING_SEAL)
    rows = read_jsonl(RAW_ROWS)
    if file_sha256(RAW_ROWS) != seal["analysis_rows_sha256"]:
        raise RuntimeError("formal rows hash mismatch")
    discovery = summarize(rows, "discovery")
    confirmation = summarize(rows, "confirmation")
    positive = discovery["role_gate_pass"] and confirmation["role_gate_pass"]
    summary = {
        "phase": PHASE,
        "created_at": utc_now(),
        "discovery": discovery,
        "confirmation": confirmation,
        "role_decision": "positive" if positive else "not_confirmed",
        "overall_status": "embedding_dominant_rescue_role_confirmed" if positive else "embedding_dominant_rescue_role_not_confirmed",
    }
    claims = {
        "expanded_rescue_role_decomposition": {
            "type": "E3-KT" if positive else "E3-KT-scope-boundary",
            "accepted": True,
            "claim": (
                "Across new tasks and both splits, the fitted token/position embedding pair is a small, selective, and necessary carrier of most immediate expanded-controller rescue, with a fixed nested candidate recovering near the full coalition."
                if positive
                else "The embedding-dominant role hypothesis did not satisfy all pre-registered sufficiency, necessity, selectivity, minimality, architecture, and family gates in both splits."
            ),
        }
    }
    write_json(SUMMARY_PATH, summary)
    write_json(CLAIMS_PATH, claims)
    print(canonical_json({"roles": summary["role_decision"], "status": summary["overall_status"]}))


def replay_capsule(path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    measured = evaluate_patches(payload, payload["role_patches"], device)
    return {"trajectory_id": payload["trajectory_id"], "measured": measured}


def finalize() -> None:
    protocol = verify_protocol()
    summary = read_json(SUMMARY_PATH)
    claims = read_json(CLAIMS_PATH)
    audit = read_json(AUDIT_PATH)
    if not audit.get("gate_pass", False):
        raise RuntimeError("independent audit did not pass")
    positive = summary["role_decision"] == "positive"
    final = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": summary["overall_status"],
        "evidence": claims,
        "protocol_digest": protocol["protocol_digest"],
        "audit_digest": audit["audit_digest"],
        "formal_summary": summary,
        "authorized_next": {
            "theory_and_measurement_consolidation": True,
            "future_learning_rescue": False,
            "self_consistent_optimizer_rescue": False,
            "natural_language_encoding_claim": False,
        },
        "scope": {
            "confirmed": "role decomposition of immediate quotient-response rescue only if both formal gates passed",
            "not_claimed": ["endogenous module identity", "behavior recovery", "future-learning recovery", "natural-language mechanism"],
        },
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": final["status"], "authorized_next": final["authorized_next"], "final_digest": final["final_digest"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("develop", "preregister", "run-formal", "analyze", "finalize"))
    command = parser.parse_args().command
    {"develop": develop, "preregister": preregister, "run-formal": run_formal, "analyze": analyze, "finalize": finalize}[command]()


if __name__ == "__main__":
    main()
