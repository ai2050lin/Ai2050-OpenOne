#!/usr/bin/env python3
"""Phase 978 preregistration for legal-budget stabilization.

This module is intentionally CPU-only.  It defines the frozen decision gate
and can seal the complete Phase 978 protocol once the frozen legal core and
all five execution scripts exist.  The Phase 977 holdout dataset is an opaque commitment here:
``--freeze`` hashes its file bytes but never imports or parses the module.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PHASE = 978
PROTOCOL_SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]

CHECKPOINTS = (256, 512, 1024, 1536)
DECISION_CHECKPOINT = 1536
MODEL_NAME = "qwen3"
BASE_SEED = 977_000
MODEL_DIR = ROOT / "models" / "hf" / "qwen3-4b"
RESULT_DIR = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase978_legal_budget_stabilization"
)
PREREGISTRATION_PATH = RESULT_DIR / "protocol_preregistration.json"

# Literal snapshot: do not import this mapping from a mutable runner.
CONDITIONS: dict[str, dict[str, Any]] = {
    "hard_no_think": {
        "enable_thinking": False,
        "prompt_suffix": "",
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
    },
    "hard_thinking": {
        "enable_thinking": True,
        "prompt_suffix": "",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    },
    "soft_no_think": {
        "enable_thinking": True,
        "prompt_suffix": " /no_think",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    },
    "soft_thinking": {
        "enable_thinking": True,
        "prompt_suffix": " /think",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    },
}

TASKS = (
    "direct_fact",
    "classification",
    "arithmetic",
    "translation_format",
    "definition",
    "causal",
    "multistep_math",
    "logic",
)

EXPECTED_ITEMS = {"development": 64, "holdout": 128}
EXPECTED_ITEMS_PER_TASK = {"development": 8, "holdout": 16}

# Phase 977 inherited admission thresholds.  These are frozen before Phase 978
# development generation; the holdout thresholds are not selected post hoc.
MODE_VALID_THRESHOLDS = {
    "development": {
        "hard_no_think": 0.95,
        "hard_thinking": 0.80,
        "soft_no_think": 0.80,
        "soft_thinking": 0.80,
    },
    "holdout": {
        "hard_no_think": 0.95,
        "hard_thinking": 0.80,
        "soft_no_think": 0.80,
        "soft_thinking": 0.80,
    },
}
VALID_MODE_EOS_THRESHOLDS = {
    "development": {
        "hard_no_think": 0.75,
        "hard_thinking": 0.50,
        "soft_no_think": 0.65,
        "soft_thinking": 0.50,
    },
    "holdout": {
        "hard_no_think": 0.90,
        "hard_thinking": 0.75,
        "soft_no_think": 0.80,
        "soft_thinking": 0.75,
    },
}
TASK_MODE_VALID_THRESHOLD = 0.75
TASK_VALID_MODE_EOS_THRESHOLD = 0.25
REQUIRED_QUALIFIED_TASKS = 6
EXTENSION_REPLAY_EXACT_THRESHOLD = 1.0
BUDGET_OVERALL_HIT_CAP_MAX = 0.10
BUDGET_ANY_TASK_HIT_CAP_MAX = 0.25

GENERATED_MODE_PARSER_VERSION = "strict_final_region_v2"
ALIAS_POLICY = "phase977_frozen_alias_groups_and_exact_flags"
HIT_CAP_METRIC = "hit1536_rate"

# These are byte commitments.  In particular, the holdout module must only be
# passed to sha256_file; this module must never import, execute, or parse it.
PHASE977_DEV_DATASET_SHA256 = (
    "ac28e7d0b1a806653564f8f9e330c59ab3134062b45d5c578a2616e2d6997399"
)
PHASE977_HOLDOUT_DATASET_SHA256 = (
    "d4d630f00a7c0197f6e7ba83704fdcf13121d67b5b09d3a77d649cb3fdff4755"
)
PHASE977_SOURCE_FILES: dict[str, dict[str, str]] = {
    "runtime_model_registry": {
        "path": "tests/gpt5/model_registry.py",
        "sha256": (
            "84c30398a9effa47791635fd25662426164460e036e59bb83a38c855db370864"
        ),
    },
    "runtime_model_loader": {
        "path": "tests/glm5/model_utils.py",
        "sha256": (
            "204098dd67d64a333daaae4d2318bd9d33584764de877a7ebc5325eee85491ef"
        ),
    },
    "legacy_eos_helper_reference": {
        "path": "tests/glm5/phase973_conditional_trajectory.py",
        "sha256": (
            "b8fbee75e2e0c5729632399053681791717304dc9f0da7f3c2e90145c6fe8357"
        ),
    },
    "development_dataset_module": {
        "path": "tests/glm5/phase977_dev_dataset.py",
        "sha256": PHASE977_DEV_DATASET_SHA256,
    },
    "holdout_dataset_module_opaque": {
        "path": "tests/glm5/phase977_holdout_dataset.py",
        "sha256": PHASE977_HOLDOUT_DATASET_SHA256,
    },
    "legal_trajectory_runner": {
        "path": "tests/glm5/phase977_legal_mode_trajectories.py",
        "sha256": (
            "9b725cbac0cb5c975c4e588ee7f6924e60004154f0ac4cf2dbdcb9aa34a28481"
        ),
    },
    "legal_development_manifest": {
        "path": (
            "tests/glm5/result/phase977_legal_mode_trajectories/"
            "manifest_development.json"
        ),
        "sha256": (
            "de25a435eee181ebaa7219c4f5d8bb722cac948695cadd78c28243b5eb77bcb0"
        ),
    },
    "legal_development_rows": {
        "path": (
            "tests/glm5/result/phase977_legal_mode_trajectories/"
            "rows_development.jsonl"
        ),
        "sha256": (
            "8b7c9b4d2f8a1d6e8e5bf0c6a9575a8545169f6b86053a1b7fe4fa83be3fe426"
        ),
    },
    "legal_development_summary": {
        "path": (
            "tests/glm5/result/phase977_legal_mode_trajectories/"
            "summary_development.json"
        ),
        "sha256": (
            "48ae80112682e7f8b6dceab1202c2d7ade4b99d492deb9925432dabddc8d2968"
        ),
    },
    "legal_discovery_manifest": {
        "path": (
            "tests/glm5/result/phase977_legal_mode_trajectories/"
            "manifest_discovery.json"
        ),
        "sha256": (
            "496de5516cc3e03067e83f5ba80ba65caff15aee19fccd6c2dcebee9cd792f97"
        ),
    },
    "legal_discovery_rows": {
        "path": (
            "tests/glm5/result/phase977_legal_mode_trajectories/"
            "rows_discovery.jsonl"
        ),
        "sha256": (
            "fb031514b8b3cff3737c7b6a2151be2ec545437579c5ca61f8fb2a3ac877b349"
        ),
    },
    "budget_discovery_runner": {
        "path": "tests/glm5/phase977_thinking_budget_audit.py",
        "sha256": (
            "3491a1b0f5cc57afa4216796e730d9ed8502f44f8b4c565c5c9bf5517a47c39d"
        ),
    },
    "budget_discovery_manifest": {
        "path": "tests/glm5/result/phase977_thinking_budget_audit/manifest.json",
        "sha256": (
            "0b7314cbb3654d583ba94a80a91d2097710072524b842cb29027df9c8c197569"
        ),
    },
    "budget_discovery_rows": {
        "path": "tests/glm5/result/phase977_thinking_budget_audit/rows.jsonl",
        "sha256": (
            "1d57508f89f8c0ee7e644fad1eba64844d7959ffca0d778331b3f583a86dbd50"
        ),
    },
    "budget_discovery_summary": {
        "path": "tests/glm5/result/phase977_thinking_budget_audit/summary.json",
        "sha256": (
            "417240d25aede7c18cad4724c78508958262d7356e380cd48da8da052be05186"
        ),
    },
}

PHASE978_SCRIPT_PATHS = {
    "protocol": "tests/glm5/phase978_budget_protocol.py",
    "legal_core": "tests/glm5/phase978_legal_core.py",
    "development_runner": "tests/glm5/phase978_dev_budget_stabilization.py",
    "development_admission_auditor": "tests/glm5/phase978_dev_admission_audit.py",
    "holdout_runner": "tests/glm5/phase978_holdout_budget_confirmation.py",
    "wrong_answer_state_runner": "tests/glm5/phase978_wrong_answer_safety.py",
}

PROTOCOL_RULES = {
    "endpoint": (
        "The natural gate uses an actual generated EOS token. A length cap, "
        "synthetic terminator, parser-inferred ending, or forced close is not EOS."
    ),
    "diagnostic_boundary": (
        "g* is diagnostic only and cannot satisfy EOS, mode validity, semantic, "
        "task-coverage, replay, or budget-stability admission criteria."
    ),
    "decision_time": (
        "Only the complete 1536-token checkpoint is a decision point; 256, 512, "
        "and 1024 are preregistered trajectory observations only."
    ),
    "absorbing_completion": (
        "The first actual EOS is absorbing: a trajectory completed at an earlier "
        "checkpoint is never regenerated or reclassified at later checkpoints."
    ),
    "denominator": (
        "Every rate uses the full frozen split denominator, including malformed, "
        "non-EOS, wrong-answer, and cap-hit trajectories; no survivor denominator."
    ),
    "replay_denominator": (
        "extension_replay_exact_rate uses the full trajectory denominator; a "
        "trajectory needing no extension is vacuously exact. Any observed replay "
        "mismatch also fails closed before a decision artifact is produced."
    ),
    "split_specific_thresholds": (
        "Development intentionally retains the Phase977 development admission "
        "thresholds 0.75/0.50/0.65/0.50 for valid-mode-EOS; holdout retains the "
        "stricter Phase977 confirmation thresholds 0.90/0.75/0.80/0.75."
    ),
    "scoring": (
        "Phase 977 aliases/exact flags and strict_final_region_v2 are frozen; "
        "they cannot be edited after observing Phase 978 outputs."
    ),
    "wrong_answer_states": (
        "W/WP labels are secondary diagnostics and are excluded from every "
        "primary development or holdout admission gate."
    ),
    "holdout_firewall": (
        "Holdout may open exactly once only after the separately hashed independent "
        "development admission auditor reports a complete PASS; development runner "
        "output alone cannot authorize holdout access."
    ),
}


def canonical_json(value: Any) -> str:
    """Return the unique UTF-8 JSON representation used for commitments."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    """Hash a file as opaque bytes without importing or parsing it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_write_json(path: Path, value: Any) -> None:
    """Durably replace one JSON file in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _rate(summary: Mapping[str, Any], key: str, label: str) -> float:
    value = summary.get(key)
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label}.{key} must be a numeric rate",
    )
    rate = float(value)
    require(math.isfinite(rate) and 0.0 <= rate <= 1.0, f"invalid {label}.{key}")
    return rate


def _count(summary: Mapping[str, Any], expected: int, label: str) -> bool:
    value = summary.get("n")
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def evaluate_gate(
    *,
    split: str,
    checkpoint: int,
    complete: bool,
    condition_summaries: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure Phase 978 admission decision over already-computed summaries.

    The function performs no file, module, model, clock, random, or environment
    access.  Perfect interim metrics still return ``passed=False`` because 1536
    is the sole decision checkpoint.
    """
    require(split in EXPECTED_ITEMS, f"unsupported split: {split!r}")
    require(checkpoint in CHECKPOINTS, f"unsupported checkpoint: {checkpoint!r}")
    require(isinstance(complete, bool), "complete must be bool")
    require(isinstance(condition_summaries, Mapping), "condition_summaries missing")

    if checkpoint != DECISION_CHECKPOINT:
        return {
            "schema_version": 1,
            "split": split,
            "checkpoint": checkpoint,
            "decision_checkpoint": DECISION_CHECKPOINT,
            "decision_eligible": False,
            "complete": complete,
            "condition_checks": {},
            "passed": False,
            "reason": "interim checkpoint; observation only",
        }

    expected_conditions = set(CONDITIONS)
    require(
        set(condition_summaries) == expected_conditions,
        "condition_summaries must contain exactly the four official conditions",
    )

    condition_checks: dict[str, Any] = {}
    for condition in CONDITIONS:
        summary = condition_summaries[condition]
        require(isinstance(summary, Mapping), f"{condition} summary must be a mapping")
        overall = summary.get("overall")
        by_task = summary.get("by_task")
        require(isinstance(overall, Mapping), f"{condition}.overall missing")
        require(isinstance(by_task, Mapping), f"{condition}.by_task missing")
        require(
            set(by_task) == set(TASKS),
            f"{condition}.by_task must contain exactly the eight frozen tasks",
        )

        overall_mode = _rate(overall, "mode_valid_rate", f"{condition}.overall")
        overall_eos = _rate(
            overall, "valid_mode_eos_rate", f"{condition}.overall"
        )
        replay = _rate(
            overall, "extension_replay_exact_rate", f"{condition}.overall"
        )
        overall_hit_cap = _rate(overall, HIT_CAP_METRIC, f"{condition}.overall")
        denominator_passed = _count(
            overall, EXPECTED_ITEMS[split], f"{condition}.overall"
        )

        task_checks: dict[str, Any] = {}
        qualified_tasks: list[str] = []
        cap_violating_tasks: list[str] = []
        for task in TASKS:
            task_summary = by_task[task]
            require(
                isinstance(task_summary, Mapping),
                f"{condition}.by_task.{task} must be a mapping",
            )
            task_mode = _rate(
                task_summary, "mode_valid_rate", f"{condition}.by_task.{task}"
            )
            task_eos = _rate(
                task_summary,
                "valid_mode_eos_rate",
                f"{condition}.by_task.{task}",
            )
            task_hit_cap = _rate(
                task_summary, HIT_CAP_METRIC, f"{condition}.by_task.{task}"
            )
            task_denominator_passed = _count(
                task_summary,
                EXPECTED_ITEMS_PER_TASK[split],
                f"{condition}.by_task.{task}",
            )
            task_qualified = (
                task_denominator_passed
                and task_mode >= TASK_MODE_VALID_THRESHOLD
                and task_eos >= TASK_VALID_MODE_EOS_THRESHOLD
            )
            task_budget_passed = (
                task_denominator_passed
                and task_hit_cap <= BUDGET_ANY_TASK_HIT_CAP_MAX
            )
            if task_qualified:
                qualified_tasks.append(task)
            if not task_budget_passed:
                cap_violating_tasks.append(task)
            task_checks[task] = {
                "n": task_summary.get("n"),
                "denominator_passed": task_denominator_passed,
                "mode_valid_rate": task_mode,
                "mode_valid_passed": task_mode >= TASK_MODE_VALID_THRESHOLD,
                "valid_mode_eos_rate": task_eos,
                "valid_mode_eos_passed": (
                    task_eos >= TASK_VALID_MODE_EOS_THRESHOLD
                ),
                HIT_CAP_METRIC: task_hit_cap,
                "budget_stability_passed": task_budget_passed,
                "qualified": task_qualified,
            }

        mode_passed = overall_mode >= MODE_VALID_THRESHOLDS[split][condition]
        eos_passed = overall_eos >= VALID_MODE_EOS_THRESHOLDS[split][condition]
        replay_passed = replay == EXTENSION_REPLAY_EXACT_THRESHOLD
        coverage_passed = len(qualified_tasks) >= REQUIRED_QUALIFIED_TASKS
        overall_budget_passed = overall_hit_cap <= BUDGET_OVERALL_HIT_CAP_MAX
        task_budget_passed = not cap_violating_tasks
        condition_passed = all(
            (
                denominator_passed,
                mode_passed,
                eos_passed,
                replay_passed,
                coverage_passed,
                overall_budget_passed,
                task_budget_passed,
            )
        )
        condition_checks[condition] = {
            "n": overall.get("n"),
            "denominator_passed": denominator_passed,
            "mode_valid_rate": overall_mode,
            "mode_valid_threshold": MODE_VALID_THRESHOLDS[split][condition],
            "mode_valid_passed": mode_passed,
            "valid_mode_eos_rate": overall_eos,
            "valid_mode_eos_threshold": VALID_MODE_EOS_THRESHOLDS[split][condition],
            "valid_mode_eos_passed": eos_passed,
            "extension_replay_exact_rate": replay,
            "extension_replay_passed": replay_passed,
            "qualified_tasks": qualified_tasks,
            "qualified_task_count": len(qualified_tasks),
            "task_coverage_passed": coverage_passed,
            HIT_CAP_METRIC: overall_hit_cap,
            "overall_budget_stability_passed": overall_budget_passed,
            "cap_violating_tasks": cap_violating_tasks,
            "all_task_budget_stability_passed": task_budget_passed,
            "by_task": task_checks,
            "passed": condition_passed,
        }

    full_denominator_passed = all(
        check["denominator_passed"]
        and all(task["denominator_passed"] for task in check["by_task"].values())
        for check in condition_checks.values()
    )
    passed = (
        complete
        and full_denominator_passed
        and all(check["passed"] for check in condition_checks.values())
    )
    return {
        "schema_version": 1,
        "split": split,
        "checkpoint": checkpoint,
        "decision_checkpoint": DECISION_CHECKPOINT,
        "decision_eligible": True,
        "complete": complete,
        "full_denominator_passed": full_denominator_passed,
        "condition_checks": condition_checks,
        "passed": passed,
        "rule": (
            "all four official conditions; inherited overall thresholds; >=6/8 "
            "qualified tasks; exact extension replay; full denominator; at 1536 "
            "overall hit-cap <=0.10 and every task hit-cap <=0.25"
        ),
    }


def _verified_phase977_sources() -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for label, commitment in PHASE977_SOURCE_FILES.items():
        relative = commitment["path"]
        path = ROOT / Path(relative)
        require(path.is_file(), f"missing frozen Phase 977 source: {relative}")
        actual = sha256_file(path)
        require(
            actual == commitment["sha256"],
            f"frozen Phase 977 source changed: {relative}: {actual}",
        )
        verified[label] = {
            "path": relative,
            "sha256": actual,
            "verified": True,
            "access": (
                "opaque_file_hash_only"
                if label == "holdout_dataset_module_opaque"
                else "file_hash_only"
            ),
        }
    return verified


def _phase978_script_hashes() -> dict[str, Any]:
    hashes: dict[str, Any] = {}
    for label, relative in PHASE978_SCRIPT_PATHS.items():
        path = ROOT / Path(relative)
        require(path.is_file(), f"cannot freeze before script exists: {relative}")
        hashes[label] = {"path": relative, "sha256": sha256_file(path)}
    return hashes


def _local_model_artifact_identity() -> dict[str, Any]:
    require(MODEL_DIR.is_dir(), f"missing local model directory: {MODEL_DIR}")
    required_names = {
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    shard_names = {path.name for path in MODEL_DIR.glob("*.safetensors")}
    require(shard_names, f"no local model shards found: {MODEL_DIR}")
    names = sorted(required_names | shard_names)
    files: dict[str, Any] = {}
    for name in names:
        path = MODEL_DIR / name
        require(path.is_file(), f"missing required model artifact: {path}")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "logical_name": MODEL_NAME,
        "path": "models/hf/qwen3-4b",
        "files": files,
        "identity_sha256": sha256_json(files),
        "weights_loaded": False,
        "gpu_accessed": False,
    }


def _runtime_versions() -> dict[str, str]:
    try:
        torch_version = importlib.metadata.version("torch")
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required runtime package missing: {exc.name}") from exc
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_full": sys.version,
        "torch": torch_version,
        "transformers": transformers_version,
        "version_source": "installed_distribution_metadata_only",
    }


def _protocol_core() -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": "legal_budget_stabilization",
        "execution_contract": {
            "cpu_only": True,
            "generation_performed": False,
            "model_weights_loaded": False,
            "holdout_module_imported": False,
            "holdout_module_parsed": False,
        },
        "checkpoints": list(CHECKPOINTS),
        "decision_checkpoint": DECISION_CHECKPOINT,
        "generation_schedule": {
            "base_seed": BASE_SEED,
            "seed_rule": (
                "sha256(phase977|base_seed|split|item_id) modulo 2^31-1; "
                "the same item seed is reused across all four conditions"
            ),
            "extension_rule": (
                "rerun from the original official chat prefix with the same seed "
                "only when the preceding checkpoint exhausted its cap without EOS"
            ),
            "replay_requirement": (
                "all generated token IDs through the preceding checkpoint must "
                "match exactly; otherwise the protocol fails"
            ),
            "early_eos_absorbing": True,
        },
        "conditions": CONDITIONS,
        "tasks": list(TASKS),
        "expected_items": EXPECTED_ITEMS,
        "expected_items_per_task": EXPECTED_ITEMS_PER_TASK,
        "parser_and_scoring": {
            "generated_mode_parser_version": GENERATED_MODE_PARSER_VERSION,
            "alias_policy": ALIAS_POLICY,
        },
        "gate": {
            "mode_valid_thresholds": MODE_VALID_THRESHOLDS,
            "valid_mode_eos_thresholds": VALID_MODE_EOS_THRESHOLDS,
            "task_qualification": {
                "mode_valid_rate_min": TASK_MODE_VALID_THRESHOLD,
                "valid_mode_eos_rate_min": TASK_VALID_MODE_EOS_THRESHOLD,
                "qualified_tasks_required_per_condition": REQUIRED_QUALIFIED_TASKS,
                "task_count": len(TASKS),
            },
            "extension_replay_exact_rate_required": (
                EXTENSION_REPLAY_EXACT_THRESHOLD
            ),
            "budget_stability_at_1536": {
                "metric": HIT_CAP_METRIC,
                "overall_max_per_condition": BUDGET_OVERALL_HIT_CAP_MAX,
                "any_task_max": BUDGET_ANY_TASK_HIT_CAP_MAX,
            },
            "all_four_conditions_must_pass": True,
            "complete_full_denominator_required": True,
        },
        "protocol_rules": PROTOCOL_RULES,
        "phase977_frozen_sources": _verified_phase977_sources(),
        "phase978_script_hashes": _phase978_script_hashes(),
        "local_model_artifact_identity": _local_model_artifact_identity(),
        "runtime_versions": _runtime_versions(),
    }


def _without_envelope(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in document.items()
        if key not in {"protocol_sha256", "created_at_utc"}
    }


def freeze_protocol(path: Path = PREREGISTRATION_PATH) -> dict[str, Any]:
    """Seal the protocol once, or verify an identical existing seal."""
    core = _protocol_core()
    protocol_sha256 = sha256_json(core)
    document = {
        **core,
        "protocol_sha256": protocol_sha256,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if path.exists():
        require(path.is_file(), f"preregistration path is not a file: {path}")
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid existing preregistration: {path}: {exc}") from exc
        require(isinstance(existing, dict), "existing preregistration must be an object")
        claimed = existing.get("protocol_sha256")
        actual = sha256_json(_without_envelope(existing))
        require(
            isinstance(claimed, str) and claimed == actual,
            "existing preregistration self-hash mismatch; refusing overwrite",
        )
        require(
            actual == protocol_sha256 and _without_envelope(existing) == core,
            "existing preregistration differs from current frozen protocol; "
            "refusing overwrite",
        )
        return existing

    atomic_write_json(path, document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="verify all commitments and atomically create the preregistration",
    )
    args = parser.parse_args()
    if not args.freeze:
        parser.error("--freeze is required; importing this module is otherwise inert")
    document = freeze_protocol()
    print(
        canonical_json(
            {
                "path": str(PREREGISTRATION_PATH),
                "protocol_sha256": document["protocol_sha256"],
                "status": "frozen",
            }
        )
    )


if __name__ == "__main__":
    main()
