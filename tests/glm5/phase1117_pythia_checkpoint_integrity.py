#!/usr/bin/env python3
"""Preflight every Phase1117 revision-3 checkpoint before behavior is observed."""

from __future__ import annotations

import gc
import json
from typing import Any

import torch
from transformers import AutoModelForCausalLM

import phase1117_pythia_training_dynamics_behavior as behavior
import phase1117_pythia_training_dynamics_protocol as protocol


ALLOWED_COLLISION_GROUPS = ({"step0", "step1"},)


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    behavior_root = protocol.OUT_ROOT / "behavior"
    if behavior_root.exists() and any(behavior_root.rglob("candidate_detail.jsonl")):
        raise RuntimeError("behavior output exists before the all-checkpoint integrity preflight")

    checkpoints: dict[str, Any] = {}
    probe_groups: dict[str, list[str]] = {}
    for checkpoint in prereg["checkpoints"]:
        local_path, repo_commit = behavior.ensure_checkpoint(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            use_safetensors=protocol.WEIGHT_FORMAT.endswith(".safetensors"),
        )
        probe = behavior.parameter_probe(model)
        manifest = behavior.model_manifest(local_path)
        checkpoints[checkpoint] = {
            "repo_commit": repo_commit,
            "weight_format": protocol.WEIGHT_FORMAT,
            "parameter_probe": probe,
            "model_manifest": manifest,
            "model_manifest_digest": protocol.digest(manifest),
        }
        probe_groups.setdefault(probe["digest"], []).append(checkpoint)
        del model
        gc.collect()
        print(json.dumps({"phase": protocol.PHASE, "checkpoint_integrity_complete": checkpoint}), flush=True)

    collision_groups = [group for group in probe_groups.values() if len(group) > 1]
    unexpected_collisions = [
        group for group in collision_groups if set(group) not in ALLOWED_COLLISION_GROUPS
    ]
    checks = {
        "protocol_revision_4": prereg["protocol_revision"] == 4,
        "protocol_audit_passed": protocol_audit["all_checks_passed"],
        "checkpoint_set_complete": list(checkpoints) == prereg["checkpoints"],
        "declared_weight_carrier_only": all(
            item["weight_format"] == protocol.WEIGHT_FORMAT
            and any(entry["path"] == protocol.WEIGHT_FORMAT for entry in item["model_manifest"])
            and not any(entry["path"] == "pytorch_model.bin" for entry in item["model_manifest"])
            for item in checkpoints.values()
        ),
        "parameter_count_consistent": {item["parameter_probe"]["parameter_count"] for item in checkpoints.values()} == {292},
        "repo_commits_unique": len({item["repo_commit"] for item in checkpoints.values()}) == len(checkpoints),
        "known_corrupt_branches_excluded": "step16" not in checkpoints and "step32" not in checkpoints,
        "no_unexpected_parameter_probe_collision": not unexpected_collisions,
        "final_checkpoint_probe_unique": all(
            "step143000" not in group for group in collision_groups
        ),
        "no_behavior_outputs_read_or_present": True,
    }
    core = {
        "schema_version": "phase1117_pythia_checkpoint_integrity.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checkpoint_set": prereg["checkpoints"],
        "allowed_collision_groups": [sorted(group) for group in ALLOWED_COLLISION_GROUPS],
        "observed_collision_groups": collision_groups,
        "unexpected_collision_groups": unexpected_collisions,
        "checkpoints": checkpoints,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    result = dict(core)
    result["integrity_digest"] = protocol.digest(core)
    protocol.write_json(protocol.OUT_ROOT / "protocol" / "checkpoint_integrity.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
