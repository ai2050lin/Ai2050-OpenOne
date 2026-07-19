#!/usr/bin/env python3
"""Phase 978 auxiliary W/WP wrong-answer termination audit on dev64.

This script is deliberately separate from the Phase 978 legal-trajectory gate.
It performs teacher-forced, next-token measurements under Qwen3's official
``hard_no_think`` prefix on the frozen Phase 977 development set only.

For every item it constructs four answer-end states:

* C:  the canonical correct answer without terminal sentence punctuation;
* P:  C followed by a period;
* W:  a wrong answer obtained by rotating canonical answers within the task;
* WP: W followed by a period.

The wrong-answer rotation is fixed before model execution.  Every W and WP is
checked against the target item's frozen alias groups with the same boundary-
safe matcher used by Phase 977.  Any collision fails closed.

Interpretation contract
-----------------------
This experiment tests formal completion/punctuation versus externally supplied
answer correctness at a teacher-forced answer boundary.  It is not a natural
trajectory experiment, is not one of the four official mode conditions, does
not affect the Phase 978 main gate, cannot authorize holdout access, and does
not by itself locate or prove an internal mechanism.

The script must never import the sealed holdout module.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
GLM5_DIR = ROOT / "tests" / "glm5"
if str(GLM5_DIR) not in sys.path:
    sys.path.insert(0, str(GLM5_DIR))

FORBIDDEN_HOLDOUT_MODULE = "phase977_holdout_dataset"


def assert_holdout_not_loaded() -> None:
    """Fail closed if any caller has imported the sealed holdout module."""
    loaded = sorted(
        name
        for name in sys.modules
        if name.rsplit(".", 1)[-1] == FORBIDDEN_HOLDOUT_MODULE
    )
    if loaded:
        raise RuntimeError(
            "sealed holdout module is loaded; refusing auxiliary dev-only run: "
            f"{loaded}"
        )


assert_holdout_not_loaded()

from model_utils import MODEL_CONFIGS, load_model, release_model  # noqa: E402
import phase977_dev_dataset as dev_dataset  # noqa: E402
from phase978_legal_core import get_eos_ids  # noqa: E402

assert_holdout_not_loaded()


PHASE = 978
MODEL_NAME = "qwen3"
BASE_SEED = 978_100
STATES = ("C", "P", "W", "WP")
OUT = ROOT / "tests" / "glm5" / "result" / "phase978_wrong_answer_safety"
MANIFEST_PATH = OUT / "manifest.json"
ROWS_PATH = OUT / "rows.jsonl"
SUMMARY_PATH = OUT / "summary.json"
SCRIPT_PATH = Path(__file__).resolve()
DEV_DATASET_PATH = GLM5_DIR / "phase977_dev_dataset.py"
PROTOCOL_PATH = GLM5_DIR / "phase978_budget_protocol.py"
PROTOCOL_PREREGISTRATION_PATH = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase978_legal_budget_stabilization"
    / "protocol_preregistration.json"
)
EXPECTED_DEV_DATASET_SHA256 = (
    "ac28e7d0b1a806653564f8f9e330c59ab3134062b45d5c578a2616e2d6997399"
)
SCHEMA_VERSION = 1
MATCHER_VERSION = "phase978_normalized_phase977_superset_v1"
EXPERIMENT_CONTRACT = {
    "role": "auxiliary_teacher_forced_dev_only",
    "tests": "formal completion and punctuation versus external answer correctness",
    "natural_trajectory_evidence": False,
    "official_four_mode_comparison": False,
    "internal_mechanism_evidence": False,
    "affects_main_legal_trajectory_gate": False,
    "authorizes_holdout": False,
    "imports_holdout": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def normalize_item(raw: dict[str, Any]) -> dict[str, Any]:
    required = {"id", "task", "prompt", "answer", "alias_groups", "exact"}
    missing = sorted(required - set(raw))
    if missing:
        raise RuntimeError(f"{raw.get('id', '<missing-id>')}: missing {missing}")
    groups = raw["alias_groups"]
    if not isinstance(groups, list) or not groups:
        raise RuntimeError(f"{raw['id']}: invalid alias_groups")
    normalized_groups: list[list[str]] = []
    for group in groups:
        if not isinstance(group, list) or not group:
            raise RuntimeError(f"{raw['id']}: invalid alias group")
        values = [str(value).strip() for value in group]
        if any(not value for value in values):
            raise RuntimeError(f"{raw['id']}: empty alias")
        normalized_groups.append(values)
    exact = bool(raw["exact"])
    if exact and len(normalized_groups) != 1:
        raise RuntimeError(f"{raw['id']}: exact item needs one alias group")
    return {
        "id": str(raw["id"]),
        "task": str(raw["task"]),
        "prompt": str(raw["prompt"]),
        "answer": str(raw["answer"]),
        "alias_groups": normalized_groups,
        "exact": exact,
    }


def load_dev_items() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert_holdout_not_loaded()
    current_dataset_sha256 = sha256_file(DEV_DATASET_PATH)
    if current_dataset_sha256 != EXPECTED_DEV_DATASET_SHA256:
        raise RuntimeError(
            "frozen dev dataset module hash changed: "
            f"expected={EXPECTED_DEV_DATASET_SHA256} "
            f"current={current_dataset_sha256}"
        )
    audit = dev_dataset.audit_dataset()
    if not audit.get("ok") or audit.get("errors"):
        raise RuntimeError(f"frozen dev dataset audit failed: {audit}")
    items = [normalize_item(row) for row in dev_dataset.build_dataset()]
    counts = dict(sorted(Counter(row["task"] for row in items).items()))
    if len(items) != 64 or counts != dev_dataset.EXPECTED_COUNTS:
        raise RuntimeError(
            f"frozen dev count mismatch: n={len(items)}, counts={counts}"
        )
    if len({row["id"] for row in items}) != len(items):
        raise RuntimeError("duplicate dev item id")
    assert_holdout_not_loaded()
    return items, audit


def answer_core(value: str) -> str:
    """Remove only terminal sentence punctuation used by C/P and W/WP."""
    core = re.sub(r"[\s.!?;:]+$", "", value).strip()
    if not core:
        raise RuntimeError(f"answer became empty after terminal trim: {value!r}")
    return core


def substring_normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", value).strip()


LEGACY_DISCOVERY_STEMS = {
    "condens", "refract", "dissolv", "reflect", "magnet"
}


def substring_alias_matches(alias: str, value: str) -> bool:
    alias_value = substring_normalize(alias)
    if not alias_value:
        return False
    escaped = re.escape(alias_value)
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\d+:\d+)", alias_value):
        pattern = r"(?<![\w.+-])" + escaped + r"(?!\w|\.\d)"
    elif alias_value in LEGACY_DISCOVERY_STEMS:
        pattern = r"(?<!\w)" + escaped + r"\w*"
    else:
        pattern = r"(?<!\w)" + escaped + r"(?!\w)"
    return re.search(pattern, value) is not None


def exact_candidates(text: str) -> set[str]:
    value = unicodedata.normalize("NFKC", text).strip().casefold()
    candidates = {value}
    if value and value[-1] in ".?!":
        candidates.add(value[:-1].rstrip())
    return candidates


def semantic_match(
    alias_groups: list[list[str]], text: str, exact: bool
) -> bool:
    if exact:
        if len(alias_groups) != 1:
            raise RuntimeError("exact semantic matching requires one alias group")
        candidates = exact_candidates(text)
        return any(
            unicodedata.normalize("NFKC", alias).strip().casefold()
            in candidates
            for alias in alias_groups[0]
        )
    value = substring_normalize(text)
    return all(
        any(substring_alias_matches(alias, value) for alias in group)
        for group in alias_groups
    )


def build_wrong_assignments(
    items: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Use the smallest collision-free cyclic shift inside each sorted task."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_task[item["task"]].append(item)

    assignments: dict[str, dict[str, str]] = {}
    for task in sorted(by_task):
        task_items = sorted(by_task[task], key=lambda row: row["id"])
        if len(task_items) < 2:
            raise RuntimeError(f"{task}: wrong-answer rotation needs >=2 items")

        selected_shift: int | None = None
        for shift in range(1, len(task_items)):
            collision = False
            for index, target in enumerate(task_items):
                donor = task_items[(index + shift) % len(task_items)]
                correct = answer_core(target["answer"])
                wrong = answer_core(donor["answer"])
                if substring_normalize(correct) == substring_normalize(wrong):
                    collision = True
                    break
                if semantic_match(
                    target["alias_groups"], wrong, target["exact"]
                ) or semantic_match(
                    target["alias_groups"], wrong + ".", target["exact"]
                ):
                    collision = True
                    break
            if not collision:
                selected_shift = shift
                break
        if selected_shift is None:
            raise RuntimeError(
                f"{task}: no global same-task rotation avoids target aliases; "
                "fail closed"
            )

        for index, target in enumerate(task_items):
            donor = task_items[(index + selected_shift) % len(task_items)]
            correct = answer_core(target["answer"])
            wrong = answer_core(donor["answer"])
            if substring_normalize(correct) == substring_normalize(wrong):
                raise RuntimeError(
                    f"{target['id']}: rotated W equals C from {donor['id']}"
                )
            if semantic_match(target["alias_groups"], wrong, target["exact"]):
                raise RuntimeError(
                    f"{target['id']}: rotated W matches target aliases; fail closed"
                )
            if semantic_match(
                target["alias_groups"], wrong + ".", target["exact"]
            ):
                raise RuntimeError(
                    f"{target['id']}: rotated WP matches target aliases; fail closed"
                )
            assignments[target["id"]] = {
                "correct": correct,
                "wrong": wrong,
                "wrong_donor_id": donor["id"],
                "wrong_donor_task": donor["task"],
                "rotation_shift": str(selected_shift),
                "rotation_rule": (
                    "smallest positive global collision-free cyclic shift in "
                    "sorted same-task items"
                ),
            }
    if set(assignments) != {item["id"] for item in items}:
        raise RuntimeError("wrong-answer assignment key mismatch")
    return assignments


def state_contents(assignment: dict[str, str]) -> dict[str, str]:
    return {
        "C": assignment["correct"],
        "P": assignment["correct"] + ".",
        "W": assignment["wrong"],
        "WP": assignment["wrong"] + ".",
    }


def stable_seed(item_id: str, state: str) -> int:
    raw = f"phase978|{BASE_SEED}|wrong_answer_safety|{item_id}|{state}"
    return int.from_bytes(
        hashlib.sha256(raw.encode("utf-8")).digest()[:8], "big"
    ) % (2**31 - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def local_model_artifact_identity() -> dict[str, Any]:
    """Hash the local config, tokenizer, and every weight shard."""
    model_dir = Path(MODEL_CONFIGS[MODEL_NAME]["path"]).resolve()
    required = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
    ]
    tokenizer_candidates = [
        "tokenizer.json",
        "tokenizer.model",
        "merges.txt",
        "vocab.json",
    ]
    names = list(required)
    names.extend(
        name for name in tokenizer_candidates if (model_dir / name).is_file()
    )
    weight_names = sorted(path.name for path in model_dir.glob("*.safetensors"))
    if not weight_names:
        raise RuntimeError(f"no safetensors files found in {model_dir}")
    names.extend(weight_names)
    if not any(name in names for name in tokenizer_candidates):
        raise RuntimeError(f"no tokenizer artifact found in {model_dir}")

    files: dict[str, Any] = {}
    for name in names:
        path = model_dir / name
        if not path.is_file():
            raise RuntimeError(f"required local model artifact missing: {path}")
        files[name] = {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    identity = {"model_dir": str(model_dir), "files": files}
    return {**identity, "combined_sha256": sha256_json(identity)}


def single_token_id(tokenizer, text: str) -> int:
    ids = list(
        tokenizer(
            text, add_special_tokens=False, return_attention_mask=False
        ).input_ids
    )
    if len(ids) != 1:
        raise RuntimeError(f"expected one token for {text!r}, got {ids}")
    return int(ids[0])


def render_hard_no_think_prefix(tokenizer, item: dict[str, Any]) -> tuple[str, list[int]]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": item["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = list(
        tokenizer(
            rendered, add_special_tokens=False, return_attention_mask=False
        ).input_ids
    )
    open_id = single_token_id(tokenizer, "<think>")
    close_id = single_token_id(tokenizer, "</think>")
    opens = [index for index, value in enumerate(ids) if int(value) == open_id]
    closes = [index for index, value in enumerate(ids) if int(value) == close_id]
    if not (
        len(opens) == 1
        and len(closes) == 1
        and opens[0] < closes[0]
    ):
        raise RuntimeError(
            f"{item['id']}: official hard_no_think prefix lacks one empty block"
        )
    return rendered, [int(value) for value in ids]


def prefix_identity(tokenizer, items: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for item in items:
        rendered, ids = render_hard_no_think_prefix(tokenizer, item)
        rows[item["id"]] = {
            "prompt_len": len(ids),
            "rendered_utf8_sha256": hashlib.sha256(
                rendered.encode("utf-8")
            ).hexdigest(),
            "token_ids_sha256": sha256_json(ids),
        }
    return {
        "condition": "hard_no_think",
        "apply_chat_template": {
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        "per_item": rows,
        "combined_sha256": sha256_json(rows),
    }


def frozen_dataset_identity(
    items: list[dict[str, Any]], assignments: dict[str, dict[str, str]]
) -> dict[str, Any]:
    stable_items = [
        {
            "id": item["id"],
            "task": item["task"],
            "prompt": item["prompt"],
            "answer": item["answer"],
            "alias_groups": item["alias_groups"],
            "exact": item["exact"],
        }
        for item in items
    ]
    stable_assignments = [
        {"id": item["id"], **assignments[item["id"]]} for item in items
    ]
    canonical_matcher_false_ids = [
        item["id"] for item in items
        if not semantic_match(item["alias_groups"], item["answer"], item["exact"])
    ]
    return {
        "module_path": str(DEV_DATASET_PATH),
        "module_sha256": sha256_file(DEV_DATASET_PATH),
        "n_items": len(items),
        "task_counts": dict(
            sorted(Counter(item["task"] for item in items).items())
        ),
        "exact_n": sum(bool(item["exact"]) for item in items),
        "items_sha256": sha256_json(stable_items),
        "wrong_assignments_sha256": sha256_json(stable_assignments),
        "wrong_rotation": (
            "smallest positive global collision-free cyclic shift in sorted "
            "same-task items; fail closed if no shift exists"
        ),
        "wrong_alias_collision_n": 0,
        "canonical_answer_matcher_false_n": len(canonical_matcher_false_ids),
        "canonical_answer_matcher_false_ids": canonical_matcher_false_ids,
        "canonical_answer_truth_source": (
            "external frozen dataset labels; matcher misses are reported but do "
            "not relabel C/P or change the Phase978 primary matcher"
        ),
    }


def load_frozen_protocol_identity() -> dict[str, Any]:
    """Verify the seal that commits to this exact auxiliary runner."""
    assert_holdout_not_loaded()
    if not PROTOCOL_PREREGISTRATION_PATH.is_file():
        raise RuntimeError(
            "formal model execution requires the frozen Phase 978 "
            f"preregistration: {PROTOCOL_PREREGISTRATION_PATH}"
        )
    try:
        document = json.loads(
            PROTOCOL_PREREGISTRATION_PATH.read_text(encoding="utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "invalid Phase 978 preregistration JSON"
        ) from exc
    if not isinstance(document, dict):
        raise RuntimeError("Phase 978 preregistration must be a JSON object")
    claimed = document.get("protocol_sha256")
    core = {
        key: value
        for key, value in document.items()
        if key not in {"protocol_sha256", "created_at_utc"}
    }
    actual = sha256_json(core)
    if not isinstance(claimed, str) or claimed != actual:
        raise RuntimeError("Phase 978 preregistration self-hash mismatch")
    if document.get("phase") != PHASE:
        raise RuntimeError("wrong phase in Phase 978 preregistration")
    if document.get("experiment") != "legal_budget_stabilization":
        raise RuntimeError("wrong experiment in Phase 978 preregistration")

    script_entries = document.get("phase978_script_hashes")
    if not isinstance(script_entries, dict):
        raise RuntimeError("preregistration lacks Phase 978 script hashes")
    for entry in script_entries.values():
        if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
            raise RuntimeError("invalid Phase 978 script commitment")
        committed_path = ROOT / str(entry["path"])
        if (not committed_path.is_file()
                or sha256_file(committed_path) != entry["sha256"]):
            raise RuntimeError(
                f"Phase 978 script changed after freeze: {entry.get('path')}"
            )
    expected_scripts = {
        "protocol": (
            "tests/glm5/phase978_budget_protocol.py",
            PROTOCOL_PATH,
        ),
        "wrong_answer_state_runner": (
            "tests/glm5/phase978_wrong_answer_safety.py",
            SCRIPT_PATH,
        ),
    }
    verified_scripts: dict[str, Any] = {}
    for label, (relative, path) in expected_scripts.items():
        entry = script_entries.get(label)
        if not isinstance(entry, dict):
            raise RuntimeError(f"preregistration lacks script entry {label}")
        current_sha256 = sha256_file(path)
        if entry.get("path") != relative or entry.get("sha256") != current_sha256:
            raise RuntimeError(
                f"preregistration does not commit to current {label}"
            )
        verified_scripts[label] = {
            "path": relative,
            "sha256": current_sha256,
        }

    frozen_sources = document.get("phase977_frozen_sources")
    if not isinstance(frozen_sources, dict):
        raise RuntimeError("preregistration lacks frozen Phase 977 sources")
    dev_entry = frozen_sources.get("development_dataset_module")
    if not isinstance(dev_entry, dict) or dev_entry.get("sha256") != (
        EXPECTED_DEV_DATASET_SHA256
    ):
        raise RuntimeError(
            "preregistration does not commit to the expected dev dataset"
        )
    for entry in frozen_sources.values():
        if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
            raise RuntimeError("invalid frozen source commitment")
        source_path = ROOT / str(entry["path"])
        if not source_path.is_file() or sha256_file(source_path) != entry["sha256"]:
            raise RuntimeError(
                f"frozen execution/source artifact changed: {entry.get('path')}"
            )

    expected_runtime = document.get("runtime_versions", {})
    current_runtime = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_full": sys.version,
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "version_source": "installed_distribution_metadata_only",
    }
    if current_runtime != expected_runtime:
        raise RuntimeError(
            "Python/torch/transformers runtime differs from frozen protocol"
        )

    frozen_model = document.get("local_model_artifact_identity", {})
    frozen_model_root = ROOT / str(frozen_model.get("path", ""))
    if Path(MODEL_CONFIGS[MODEL_NAME]["path"]).resolve() != frozen_model_root.resolve():
        raise RuntimeError("model loader registry path differs from frozen model path")
    current_model = local_model_artifact_identity()
    if current_model["files"] != frozen_model.get("files"):
        raise RuntimeError("local model artifacts differ from frozen protocol")
    assert_holdout_not_loaded()
    return {
        "path": str(PROTOCOL_PREREGISTRATION_PATH),
        "file_sha256": sha256_file(PROTOCOL_PREREGISTRATION_PATH),
        "protocol_sha256": claimed,
        "verified_scripts": verified_scripts,
        "development_dataset_sha256": EXPECTED_DEV_DATASET_SHA256,
        "runtime_versions": current_runtime,
        "verified_model_artifacts": current_model,
        "self_hash_verified": True,
        "holdout_module_imported": False,
    }


def make_manifest(
    model,
    tokenizer,
    device,
    eos_ids: list[int],
    items: list[dict[str, Any]],
    assignments: dict[str, dict[str, str]],
    data_audit: dict[str, Any],
    protocol_identity: dict[str, Any],
) -> dict[str, Any]:
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unavailable"

    core = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": "wrong_answer_safety_teacher_forced_dev64",
        "contract": EXPERIMENT_CONTRACT,
        "model": MODEL_NAME,
        "model_class": type(model).__name__,
        "model_name_or_path": str(getattr(model.config, "_name_or_path", "")),
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "model_artifacts": protocol_identity["verified_model_artifacts"],
        "tokenizer_class": type(tokenizer).__name__,
        "eos_token_ids": [int(value) for value in eos_ids],
        "official_prefix": prefix_identity(tokenizer, items),
        "dataset": frozen_dataset_identity(items, assignments),
        "dataset_audit": data_audit,
        "states": list(STATES),
        "state_definitions": {
            "C": "correct canonical answer with terminal sentence punctuation removed",
            "P": "C plus one ASCII period",
            "W": "same-task rotated canonical answer, alias-collision checked",
            "WP": "W plus one ASCII period",
        },
        "punctuation_tokenization_rule": (
            "P/WP are audited against their C/W token sequence; BPE retokenization "
            "is retained and reported rather than treated as a pure one-token suffix"
        ),
        "teacher_forced_measurement": (
            "append state text to the official hard_no_think rendered prefix; "
            "measure logits predicting the token immediately after the final "
            "state token"
        ),
        "gap_definition": "max_non_eos_logit - max_eos_logit",
        "matcher_version": MATCHER_VERSION,
        "base_seed": BASE_SEED,
        "seed_rule": (
            "sha256(phase978|base_seed|wrong_answer_safety|item_id|state)"
        ),
        "expected_rows": len(items) * len(STATES),
        "resume_key": ["id", "state"],
        "script_path": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "phase978_protocol_path": str(PROTOCOL_PATH),
        "phase978_protocol_sha256": (
            sha256_file(PROTOCOL_PATH) if PROTOCOL_PATH.is_file() else None
        ),
        "phase978_protocol_preregistration": protocol_identity,
        "holdout_module_forbidden": True,
        "holdout_loaded": False,
    }
    digest = sha256_json(core)
    return {**core, "manifest_sha256": digest, "created_at_utc": utc_now()}


def install_or_validate_manifest(manifest: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        prior = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        prior_core = {
            key: value
            for key, value in prior.items()
            if key not in {"manifest_sha256", "created_at_utc"}
        }
        if sha256_json(prior_core) != prior.get("manifest_sha256"):
            raise RuntimeError("stored manifest failed its own SHA256 commitment")
        if prior.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError(
                "manifest mismatch; refusing to mix old and current rows: "
                f"old={prior.get('manifest_sha256')} "
                f"new={manifest['manifest_sha256']}"
            )
        return
    atomic_write_json(MANIFEST_PATH, manifest)


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("id")), str(row.get("state"))


def verify_resume_row(
    row: dict[str, Any],
    manifest: dict[str, Any],
    tokenizer,
    item: dict[str, Any],
    assignment: dict[str, str],
) -> None:
    stored_row_sha256 = row.get("row_sha256")
    if not isinstance(stored_row_sha256, str):
        raise RuntimeError(f"{row_key(row)}: missing row_sha256")
    payload = {
        key: value for key, value in row.items() if key != "row_sha256"
    }
    if sha256_json(payload) != stored_row_sha256:
        raise RuntimeError(f"{row_key(row)}: row SHA256 mismatch")

    state = str(row.get("state"))
    if state not in STATES:
        raise RuntimeError(f"{row_key(row)}: unknown state")
    expected_content = state_contents(assignment)[state]
    expected_donor = (
        assignment["wrong_donor_id"] if state in ("W", "WP") else None
    )
    expected_values = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "manifest_sha256": manifest["manifest_sha256"],
        "id": item["id"],
        "task": item["task"],
        "state": state,
        "condition": "hard_no_think_teacher_forced",
        "seed": stable_seed(item["id"], state),
        "prompt": item["prompt"],
        "canonical_answer": item["answer"],
        "state_text": expected_content,
        "wrong_donor_id": expected_donor,
        "wrong_rotation_rule": assignment["rotation_rule"],
        "target_alias_groups": item["alias_groups"],
        "target_exact": bool(item["exact"]),
        "state_semantic_correct_by_construction": state in ("C", "P"),
        "wrong_state_alias_collision": (
            False if state in ("W", "WP") else None
        ),
        "contract": EXPERIMENT_CONTRACT,
    }
    for field, expected in expected_values.items():
        if row.get(field) != expected:
            raise RuntimeError(
                f"{row_key(row)}: frozen field {field} changed"
            )

    prefix_text, prefix_ids = render_hard_no_think_prefix(tokenizer, item)
    full_ids, content_ids = prepare_state_ids(
        tokenizer, prefix_text, prefix_ids, expected_content, item["id"]
    )
    token_fields = {
        "official_prefix_len": len(prefix_ids),
        "official_prefix_token_ids_sha256": sha256_json(prefix_ids),
        "full_input_len": len(full_ids),
        "content_token_count": len(content_ids),
        "content_token_ids": content_ids,
        "last_content_token_id": content_ids[-1],
        "last_content_token": tokenizer.convert_ids_to_tokens(content_ids[-1]),
        "answer_ends_at_last_input_position": True,
        "measured_next_token_position": len(full_ids),
    }
    token_fields.update(punctuation_token_fields(
        tokenizer, prefix_text, prefix_ids, state_contents(assignment),
        state, content_ids, item["id"]
    ))
    for field, expected in token_fields.items():
        if row.get(field) != expected:
            raise RuntimeError(
                f"{row_key(row)}: deterministic token field {field} changed"
            )

    required_metrics = {
        "max_eos_id", "max_eos_token", "max_eos_logit",
        "max_eos_probability", "total_eos_probability", "eos_rank",
        "max_non_eos_id", "max_non_eos_token", "max_non_eos_logit",
        "gap_max_non_eos_minus_max_eos", "gap_negative", "top1_id",
        "top1_token", "top1_text", "top1_logit", "top1_probability",
        "eos_top1",
    }
    missing_metrics = sorted(required_metrics - set(row))
    if missing_metrics:
        raise RuntimeError(
            f"{row_key(row)}: missing metrics {missing_metrics}"
        )
    eos_ids = {int(value) for value in manifest["eos_token_ids"]}
    gap = float(row["gap_max_non_eos_minus_max_eos"])
    reconstructed_gap = (
        float(row["max_non_eos_logit"]) - float(row["max_eos_logit"])
    )
    if abs(gap - reconstructed_gap) > 1e-5:
        raise RuntimeError(f"{row_key(row)}: gap/logit identity failed")
    if bool(row["gap_negative"]) != bool(gap < 0.0):
        raise RuntimeError(f"{row_key(row)}: gap_negative mismatch")
    if bool(row["eos_top1"]) != (int(row["top1_id"]) in eos_ids):
        raise RuntimeError(f"{row_key(row)}: eos_top1 mismatch")
    if int(row["max_eos_id"]) not in eos_ids:
        raise RuntimeError(f"{row_key(row)}: max_eos_id is not registered EOS")
    if int(row["max_non_eos_id"]) in eos_ids:
        raise RuntimeError(f"{row_key(row)}: max_non_eos_id is EOS")
    if int(row["eos_rank"]) < 1:
        raise RuntimeError(f"{row_key(row)}: invalid EOS rank")
    token_values = {
        "max_eos_token": tokenizer.convert_ids_to_tokens(
            int(row["max_eos_id"])
        ),
        "max_non_eos_token": tokenizer.convert_ids_to_tokens(
            int(row["max_non_eos_id"])
        ),
        "top1_token": tokenizer.convert_ids_to_tokens(int(row["top1_id"])),
        "top1_text": tokenizer.decode(
            [int(row["top1_id"])], skip_special_tokens=False
        ),
    }
    for field, expected in token_values.items():
        if row.get(field) != expected:
            raise RuntimeError(f"{row_key(row)}: {field} mismatch")
    finite_fields = (
        "max_eos_logit", "max_eos_probability", "total_eos_probability",
        "max_non_eos_logit", "gap_max_non_eos_minus_max_eos",
        "top1_logit", "top1_probability",
    )
    if any(not math.isfinite(float(row[field])) for field in finite_fields):
        raise RuntimeError(f"{row_key(row)}: non-finite metric")
    expected_top1_logit = max(
        float(row["max_eos_logit"]), float(row["max_non_eos_logit"])
    )
    if abs(float(row["top1_logit"]) - expected_top1_logit) > 1e-5:
        raise RuntimeError(f"{row_key(row)}: top1/logit identity failed")
    max_probability = float(row["max_eos_probability"])
    total_probability = float(row["total_eos_probability"])
    if not (0.0 <= max_probability <= total_probability <= 1.0 + 1e-6):
        raise RuntimeError(f"{row_key(row)}: invalid EOS probability fields")
    if not 0.0 <= float(row["top1_probability"]) <= 1.0:
        raise RuntimeError(f"{row_key(row)}: invalid top1 probability")


def load_rows_resume(
    manifest: dict[str, Any],
    tokenizer,
    items: list[dict[str, Any]],
    assignments: dict[str, dict[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    if not ROWS_PATH.exists():
        return records
    item_by_id = {item["id"]: item for item in items}

    with ROWS_PATH.open("rb+") as handle:
        while True:
            line_start = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            try:
                text = raw.decode("utf-8")
                row = json.loads(text)
            except (UnicodeDecodeError, json.JSONDecodeError):
                if handle.tell() == ROWS_PATH.stat().st_size and not raw.endswith(b"\n"):
                    handle.seek(line_start)
                    handle.truncate()
                    handle.flush()
                    os.fsync(handle.fileno())
                    break
                raise RuntimeError(f"invalid non-final JSONL record at byte {line_start}")
            if row.get("manifest_sha256") != manifest["manifest_sha256"]:
                raise RuntimeError(f"row manifest mismatch at byte {line_start}")
            key = row_key(row)
            if key in records:
                raise RuntimeError(f"duplicate resume key: {key}")
            item = item_by_id.get(key[0])
            if item is None or key[1] not in STATES:
                raise RuntimeError(f"unexpected resume key: {key}")
            verify_resume_row(
                row, manifest, tokenizer, item, assignments[item["id"]]
            )
            records[key] = row
    return records


def append_row(row: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(row) + "\n").encode("utf-8")
    if ROWS_PATH.exists() and ROWS_PATH.stat().st_size:
        with ROWS_PATH.open("rb+") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    with ROWS_PATH.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def prepare_state_ids(
    tokenizer, prefix_text: str, prefix_ids: list[int], content: str, item_id: str
) -> tuple[list[int], list[int]]:
    full_text = prefix_text + content
    full_ids = list(
        tokenizer(
            full_text,
            add_special_tokens=False,
            return_attention_mask=False,
        ).input_ids
    )
    full_ids = [int(value) for value in full_ids]
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise RuntimeError(f"{item_id}: state text changed official prefix tokenization")
    content_ids = full_ids[len(prefix_ids) :]
    if not content_ids:
        raise RuntimeError(f"{item_id}: state produced no content tokens")
    return full_ids, content_ids


def punctuation_token_fields(
    tokenizer, prefix_text: str, prefix_ids: list[int],
    contents: dict[str, str], state: str, content_ids: list[int], item_id: str,
) -> dict[str, Any]:
    base_state = {"P": "C", "WP": "W"}.get(state)
    if base_state is None:
        return {
            "punctuation_pair_base_state": None,
            "punctuation_base_token_ids_sha256": None,
            "punctuation_preserves_base_token_prefix": None,
            "punctuation_added_token_count": None,
            "pure_single_token_period_suffix": None,
        }
    _base_full, base_ids = prepare_state_ids(
        tokenizer, prefix_text, prefix_ids, contents[base_state], item_id)
    preserves = content_ids[:len(base_ids)] == base_ids
    added = len(content_ids) - len(base_ids)
    return {
        "punctuation_pair_base_state": base_state,
        "punctuation_base_token_ids_sha256": sha256_json(base_ids),
        "punctuation_preserves_base_token_prefix": preserves,
        "punctuation_added_token_count": added,
        "pure_single_token_period_suffix": bool(preserves and added == 1),
    }


def next_token_metrics(model, tokenizer, device, ids: list[int], eos_ids: list[int]) -> dict[str, Any]:
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    logits = output.logits[0, -1].float()
    eos_tensor = torch.tensor(eos_ids, dtype=torch.long, device=logits.device)
    eos_values = logits.index_select(0, eos_tensor)
    eos_offset = int(torch.argmax(eos_values).item())
    eos_id = int(eos_ids[eos_offset])
    eos_logit = eos_values[eos_offset]

    non_eos = logits.clone()
    non_eos[eos_tensor] = -torch.inf
    non_eos_logit, non_eos_id_tensor = torch.max(non_eos, dim=0)
    non_eos_id = int(non_eos_id_tensor.item())
    gap = float((non_eos_logit - eos_logit).item())

    probabilities = torch.softmax(logits, dim=-1)
    eos_max_probability = float(probabilities[eos_id].item())
    eos_total_probability = float(
        probabilities.index_select(0, eos_tensor).sum().item()
    )
    top1_id = int(torch.argmax(logits).item())
    eos_rank = 1 + int(torch.sum(logits > eos_logit).item())

    return {
        "max_eos_id": eos_id,
        "max_eos_token": tokenizer.convert_ids_to_tokens(eos_id),
        "max_eos_logit": float(eos_logit.item()),
        "max_eos_probability": eos_max_probability,
        "total_eos_probability": eos_total_probability,
        "eos_rank": eos_rank,
        "max_non_eos_id": non_eos_id,
        "max_non_eos_token": tokenizer.convert_ids_to_tokens(non_eos_id),
        "max_non_eos_logit": float(non_eos_logit.item()),
        "gap_max_non_eos_minus_max_eos": gap,
        "gap_negative": bool(gap < 0.0),
        "top1_id": top1_id,
        "top1_token": tokenizer.convert_ids_to_tokens(top1_id),
        "top1_text": tokenizer.decode([top1_id], skip_special_tokens=False),
        "top1_logit": float(logits[top1_id].item()),
        "top1_probability": float(probabilities[top1_id].item()),
        "eos_top1": bool(top1_id in set(eos_ids)),
    }


def build_row(
    manifest: dict[str, Any],
    model,
    tokenizer,
    device,
    eos_ids: list[int],
    item: dict[str, Any],
    assignment: dict[str, str],
    state: str,
) -> dict[str, Any]:
    assert_holdout_not_loaded()
    contents = state_contents(assignment)
    content = contents[state]
    prefix_text, prefix_ids = render_hard_no_think_prefix(tokenizer, item)
    full_ids, content_ids = prepare_state_ids(
        tokenizer, prefix_text, prefix_ids, content, item["id"]
    )
    seed = stable_seed(item["id"], state)
    seed_everything(seed)
    metrics = next_token_metrics(model, tokenizer, device, full_ids, eos_ids)
    punctuation_fields = punctuation_token_fields(
        tokenizer, prefix_text, prefix_ids, contents, state, content_ids, item["id"])
    row = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "manifest_sha256": manifest["manifest_sha256"],
        "id": item["id"],
        "task": item["task"],
        "state": state,
        "condition": "hard_no_think_teacher_forced",
        "seed": seed,
        "prompt": item["prompt"],
        "canonical_answer": item["answer"],
        "state_text": content,
        "wrong_donor_id": assignment["wrong_donor_id"] if state in ("W", "WP") else None,
        "wrong_rotation_rule": assignment["rotation_rule"],
        "target_alias_groups": item["alias_groups"],
        "target_exact": bool(item["exact"]),
        "state_semantic_correct_by_construction": state in ("C", "P"),
        "wrong_state_alias_collision": False if state in ("W", "WP") else None,
        "official_prefix_len": len(prefix_ids),
        "official_prefix_token_ids_sha256": sha256_json(prefix_ids),
        "full_input_len": len(full_ids),
        "content_token_count": len(content_ids),
        "content_token_ids": content_ids,
        "last_content_token_id": content_ids[-1],
        "last_content_token": tokenizer.convert_ids_to_tokens(content_ids[-1]),
        "answer_ends_at_last_input_position": True,
        "measured_next_token_position": len(full_ids),
        **punctuation_fields,
        **metrics,
        "contract": EXPERIMENT_CONTRACT,
        "recorded_at_utc": utc_now(),
    }
    row["row_sha256"] = sha256_json(row)
    return row


def mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def summarize_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "eos_top1_n": sum(bool(row["eos_top1"]) for row in rows),
        "gap_negative_n": sum(bool(row["gap_negative"]) for row in rows),
        "mean_gap": mean(
            [float(row["gap_max_non_eos_minus_max_eos"]) for row in rows]
        ),
        "mean_max_eos_logit": mean([float(row["max_eos_logit"]) for row in rows]),
        "mean_max_eos_probability": mean(
            [float(row["max_eos_probability"]) for row in rows]
        ),
        "mean_total_eos_probability": mean(
            [float(row["total_eos_probability"]) for row in rows]
        ),
        "mean_eos_rank": mean([float(row["eos_rank"]) for row in rows]),
    }


def paired_summary(records: dict[tuple[str, str], dict[str, Any]], item_ids: list[str]) -> dict[str, Any]:
    complete_ids = [
        item_id
        for item_id in item_ids
        if all((item_id, state) in records for state in STATES)
    ]
    p_wp_gap_direction = 0
    c_w_gap_direction = 0
    p_eos_wp_not = 0
    wp_eos_p_not = 0
    c_eos_w_not = 0
    w_eos_c_not = 0
    p_wp_gap_differences: list[float] = []
    c_w_gap_differences: list[float] = []
    punctuation_pair_n = 0
    punctuation_prefix_preserved_n = 0
    pure_single_token_period_n = 0
    retokenized_pairs: list[str] = []
    for item_id in complete_ids:
        c = records[(item_id, "C")]
        p = records[(item_id, "P")]
        w = records[(item_id, "W")]
        wp = records[(item_id, "WP")]
        p_gap = float(p["gap_max_non_eos_minus_max_eos"])
        wp_gap = float(wp["gap_max_non_eos_minus_max_eos"])
        c_gap = float(c["gap_max_non_eos_minus_max_eos"])
        w_gap = float(w["gap_max_non_eos_minus_max_eos"])
        p_wp_gap_differences.append(wp_gap - p_gap)
        c_w_gap_differences.append(w_gap - c_gap)
        p_wp_gap_direction += wp_gap > p_gap
        c_w_gap_direction += w_gap > c_gap
        p_eos_wp_not += bool(p["eos_top1"] and not wp["eos_top1"])
        wp_eos_p_not += bool(wp["eos_top1"] and not p["eos_top1"])
        c_eos_w_not += bool(c["eos_top1"] and not w["eos_top1"])
        w_eos_c_not += bool(w["eos_top1"] and not c["eos_top1"])
        for state, row in (("P", p), ("WP", wp)):
            punctuation_pair_n += 1
            preserved = bool(row["punctuation_preserves_base_token_prefix"])
            punctuation_prefix_preserved_n += preserved
            pure_single_token_period_n += bool(row["pure_single_token_period_suffix"])
            if not preserved:
                retokenized_pairs.append(f"{item_id}:{row['punctuation_pair_base_state']}->{state}")
    return {
        "n_complete_pairs": len(complete_ids),
        "WP_gap_greater_than_P_n": int(p_wp_gap_direction),
        "mean_WP_minus_P_gap": mean(p_wp_gap_differences),
        "P_eos_top1_and_WP_not_n": int(p_eos_wp_not),
        "WP_eos_top1_and_P_not_n": int(wp_eos_p_not),
        "W_gap_greater_than_C_n": int(c_w_gap_direction),
        "mean_W_minus_C_gap": mean(c_w_gap_differences),
        "C_eos_top1_and_W_not_n": int(c_eos_w_not),
        "W_eos_top1_and_C_not_n": int(w_eos_c_not),
        "punctuation_pairs_n": punctuation_pair_n,
        "punctuation_preserves_base_token_prefix_n": punctuation_prefix_preserved_n,
        "pure_single_token_period_suffix_n": pure_single_token_period_n,
        "punctuation_retokenized_pair_n": len(retokenized_pairs),
        "punctuation_retokenized_pairs": retokenized_pairs,
        "punctuation_interpretation_boundary": (
            "retokenized pairs are retained but are not pure one-token period suffix interventions"
        ),
        "interpretation": (
            "positive wrong-minus-correct gap means EOS is less competitive "
            "after the externally wrong answer; this remains an auxiliary "
            "teacher-forced association, not mechanism evidence"
        ),
    }


def build_summary(
    manifest: dict[str, Any],
    records: dict[tuple[str, str], dict[str, Any]],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_keys = {
        (item["id"], state) for item in items for state in STATES
    }
    unknown = sorted(set(records) - expected_keys)
    if unknown:
        raise RuntimeError(f"unknown row keys in result: {unknown}")
    rows = list(records.values())
    by_state = {
        state: summarize_state([row for row in rows if row["state"] == state])
        for state in STATES
    }
    by_task: dict[str, Any] = {}
    for task in sorted({item["task"] for item in items}):
        task_rows = [row for row in rows if row["task"] == task]
        by_task[task] = {
            state: summarize_state(
                [row for row in task_rows if row["state"] == state]
            )
            for state in STATES
        }
    complete = set(records) == expected_keys
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": "wrong_answer_safety_teacher_forced_dev64",
        "manifest_sha256": manifest["manifest_sha256"],
        "complete": complete,
        "expected_rows": len(expected_keys),
        "completed_rows": len(records),
        "remaining_rows": len(expected_keys - set(records)),
        "by_state": by_state,
        "by_task": by_task,
        "paired": paired_summary(
            records, [item["id"] for item in items]
        ),
        "decision_status": (
            "AUXILIARY_COMPLETE_NO_MAIN_GATE"
            if complete
            else "AUXILIARY_INCOMPLETE_RESUMABLE"
        ),
        "contract": EXPERIMENT_CONTRACT,
        "conclusion_boundary": (
            "These counts compare formal answer completion/punctuation with "
            "external gold correctness under teacher forcing. They are not "
            "natural trajectories, mode evidence, or internal mechanism evidence."
        ),
        "holdout_loaded": False,
        "updated_at_utc": utc_now(),
    }


def preflight() -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[str, dict[str, str]]
]:
    assert_holdout_not_loaded()
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError(
            f"frozen Phase 978 protocol is missing: {PROTOCOL_PATH}"
        )
    items, audit = load_dev_items()
    assignments = build_wrong_assignments(items)
    assert_holdout_not_loaded()
    return items, audit, assignments


def run(audit_only: bool = False) -> None:
    items, data_audit, assignments = preflight()
    if audit_only:
        identity = frozen_dataset_identity(items, assignments)
        report = {
            "ok": True,
            "n_items": len(items),
            "task_counts": dict(
                sorted(Counter(item["task"] for item in items).items())
            ),
            "dataset_module_sha256": sha256_file(DEV_DATASET_PATH),
            "wrong_assignments_sha256": identity["wrong_assignments_sha256"],
            "wrong_alias_collision_n": 0,
            "canonical_answer_matcher_false_n": (
                identity["canonical_answer_matcher_false_n"]),
            "canonical_answer_matcher_false_ids": (
                identity["canonical_answer_matcher_false_ids"]),
            "expected_rows": len(items) * len(STATES),
            "holdout_loaded": False,
            "contract": EXPERIMENT_CONTRACT,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    protocol_identity = load_frozen_protocol_identity()
    if not torch.cuda.is_available():
        raise RuntimeError("formal Phase978 W/WP run requires local CUDA")
    model = None
    try:
        assert_holdout_not_loaded()
        model, tokenizer, device = load_model(MODEL_NAME)
        if getattr(device, "type", str(device).split(":")[0]) != "cuda":
            raise RuntimeError(f"Qwen3 did not load on CUDA: {device}")
        assert_holdout_not_loaded()
        eos_ids = get_eos_ids(model, tokenizer)
        manifest = make_manifest(
            model,
            tokenizer,
            device,
            eos_ids,
            items,
            assignments,
            data_audit,
            protocol_identity,
        )
        install_or_validate_manifest(manifest)
        records = load_rows_resume(
            manifest, tokenizer, items, assignments
        )
        summary = build_summary(manifest, records, items)
        atomic_write_json(SUMMARY_PATH, summary)

        for item in items:
            for state in STATES:
                key = (item["id"], state)
                if key in records:
                    continue
                row = build_row(
                    manifest,
                    model,
                    tokenizer,
                    device,
                    eos_ids,
                    item,
                    assignments[item["id"]],
                    state,
                )
                verify_resume_row(
                    row,
                    manifest,
                    tokenizer,
                    item,
                    assignments[item["id"]],
                )
                append_row(row)
                records[key] = row
            atomic_write_json(
                SUMMARY_PATH, build_summary(manifest, records, items)
            )
            print(
                f"Phase978 W/WP: completed {len(records)}/{len(items) * len(STATES)} rows"
            )

        final_summary = build_summary(manifest, records, items)
        if not final_summary["complete"]:
            raise RuntimeError("run ended without all expected rows")
        atomic_write_json(SUMMARY_PATH, final_summary)
        assert_holdout_not_loaded()
        print(json.dumps({
            "status": final_summary["decision_status"],
            "rows": final_summary["completed_rows"],
            "manifest_sha256": manifest["manifest_sha256"],
            "summary": str(SUMMARY_PATH),
            "holdout_loaded": False,
        }, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_model(model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase978 dev-only teacher-forced C/P/W/WP safety audit"
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="audit dev64 and wrong-answer rotation without loading the model",
    )
    args = parser.parse_args()
    run(audit_only=bool(args.audit_only))


if __name__ == "__main__":
    main()
