#!/usr/bin/env python3
"""Phase 979 Qwen3 teacher-forced truth x punctuation runner.

The development and replication blocks share one preregistered source module,
but they are separate model-evaluation blocks.  Replication authentication is
completed before the dataset module is imported and before the literal
``build_pairs("replication")`` call can occur.  This is an execution firewall,
not a claim that the preregistered source text is analyst-blind.

The formal path is CUDA-only, deterministic, seed-free teacher forcing with
eight-way left-padded batches and an explicit attention mask.  Every row stores
the primitive next-token EOS measurements at the end of either ``A``/``B`` or
``A.``/``B.``; no generated token is sampled.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

import phase979_boundary_core as core  # noqa: E402


PHASE = 979
SCHEMA_VERSION = 1
PROTOCOL_EXPERIMENT = "three_boundary_factorial_and_truth_punctuation"
EXPERIMENT = "truth_punctuation_teacher_forcing"
MODEL_NAME = "qwen3"
SPLITS = ("development", "replication")
PROMPT_SIDES = ("qA", "qB")
CANDIDATES = ("A", "B")
PUNCTUATIONS = ("bare", "period")
BATCH_SIZE = 8
EXPECTED_PAIRS = 64
EXPECTED_ROWS = 512

OUT = GLM5 / "result" / "phase979_three_boundary_factorial"
PROTOCOL_PATH = OUT / "protocol_preregistration.json"
TRUTH_ADMISSION_PATH = OUT / "truth_admission_development.json"
SCRIPT_PATH = Path(__file__).resolve()
CORE_PATH = GLM5 / "phase979_boundary_core.py"
DATASET_PATH = GLM5 / "phase979_truth_punctuation_dataset.py"

PHASE978_DIR = GLM5 / "result" / "phase978_legal_budget_stabilization"
PHASE978_ADMISSION_PATH = PHASE978_DIR / "admission_development.json"
PHASE978_OPEN_RECEIPT_PATH = PHASE978_DIR / "holdout_open_receipt.json"
FORBIDDEN_HOLDOUT_MODULE = "phase977_holdout_dataset"
DATASET_MODULE = "phase979_truth_punctuation_dataset"

MANIFEST_HASH_EXCLUSIONS = {"manifest_sha256", "created_at_utc"}
STATUS_HASH_EXCLUSIONS = {"status_sha256", "updated_at_utc"}


def require(condition: bool, message: str) -> None:
    core.require(condition, message)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def manifest_path(split: str) -> Path:
    return OUT / f"manifest_truth_{split}.json"


def rows_path(split: str) -> Path:
    return OUT / f"rows_truth_{split}.jsonl"


def status_path(split: str) -> Path:
    return OUT / f"generator_status_truth_{split}.json"


def assert_phase978_holdout_closed() -> None:
    loaded = [
        name for name in sys.modules
        if name == FORBIDDEN_HOLDOUT_MODULE
        or name.endswith("." + FORBIDDEN_HOLDOUT_MODULE)
    ]
    require(not loaded, f"forbidden Phase977 holdout module imported: {loaded}")
    require(
        not PHASE978_OPEN_RECEIPT_PATH.exists(),
        "Phase978 holdout OPEN receipt exists; Phase979 truth run must stop",
    )


def verify_self_hash(
    document: dict[str, Any], hash_field: str, excluded: set[str], label: str,
) -> None:
    claimed = document.get(hash_field)
    require(
        isinstance(claimed, str) and len(claimed) == 64,
        f"{label} lacks a valid-looking {hash_field}",
    )
    payload = {key: value for key, value in document.items() if key not in excluded}
    require(claimed == core.sha256_json(payload), f"{label} self-hash invalid")


def iter_path_commitments(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for child in value.values():
            yield from iter_path_commitments(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_path_commitments(child)


def resolve_committed_path(path_text: str) -> Path:
    raw = Path(path_text)
    require(not raw.is_absolute(), f"sealed workspace path must be relative: {path_text}")
    resolved = (ROOT / raw).resolve()
    require(
        resolved == ROOT.resolve() or ROOT.resolve() in resolved.parents,
        f"sealed path escapes workspace: {path_text}",
    )
    return resolved


def verify_file_commitment(entry: dict[str, Any], label: str) -> Path:
    require(
        isinstance(entry.get("path"), str) and isinstance(entry.get("sha256"), str),
        f"invalid {label} path/SHA commitment",
    )
    path = resolve_committed_path(str(entry["path"]))
    require(path.is_file(), f"missing {label}: {entry['path']}")
    actual = core.sha256_file(path)
    require(actual == entry["sha256"], f"{label} changed after protocol seal")
    return path


def authenticate_phase978_no_go(commitments: dict[str, Any]) -> dict[str, Any]:
    """Authenticate the public Phase978 NO-GO without opening its holdout."""
    assert_phase978_holdout_closed()
    require(
        commitments.get("development_gate_passed") is False,
        "Phase979 protocol does not freeze the Phase978 development gate as false",
    )
    require(
        commitments.get("holdout_authorized") is False,
        "Phase979 protocol unexpectedly authorizes the Phase978 holdout",
    )
    require(
        commitments.get("holdout_loaded") is False,
        "Phase979 protocol reports Phase978 holdout access",
    )

    entries = list(iter_path_commitments(commitments))
    require(entries, "Phase979 protocol lacks Phase978 file commitments")
    admission_committed = False
    for entry in entries:
        normalized = str(entry["path"]).replace("\\", "/")
        require(
            not normalized.endswith("phase977_holdout_dataset.py"),
            "Phase979 must not commit to or inspect the old holdout source",
        )
        verified = verify_file_commitment(entry, "Phase978 lineage artifact")
        admission_committed |= verified == PHASE978_ADMISSION_PATH.resolve()
    require(
        admission_committed,
        "Phase979 protocol does not seal Phase978 admission_development.json",
    )

    admission = core.load_json(PHASE978_ADMISSION_PATH, "Phase978 development admission")
    verify_self_hash(
        admission,
        "admission_sha256",
        {"admission_sha256", "audited_at_utc"},
        "Phase978 development admission",
    )
    require(admission.get("phase") == 978, "Phase978 admission phase mismatch")
    require(
        admission.get("decision_gate", {}).get("passed") is False,
        "Phase978 development admission is not the frozen NO-GO",
    )
    require(
        admission.get("holdout_authorized") is False,
        "Phase978 development admission authorizes holdout access",
    )
    require(
        admission.get("holdout_loaded") is False,
        "Phase978 development admission reports holdout access",
    )
    assert_phase978_holdout_closed()
    return {
        "admission_sha256": admission["admission_sha256"],
        "admission_file_sha256": core.sha256_file(PHASE978_ADMISSION_PATH),
        "development_gate_passed": False,
        "holdout_authorized": False,
        "holdout_loaded": False,
        "open_receipt_exists": False,
    }


def authenticate_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the complete sealed Phase979 protocol before dataset/model access."""
    assert_phase978_holdout_closed()
    protocol = core.load_json(PROTOCOL_PATH, "Phase979 protocol preregistration")
    verify_self_hash(
        protocol,
        "protocol_sha256",
        {"protocol_sha256", "created_at_utc"},
        "Phase979 protocol preregistration",
    )
    require(protocol.get("phase") == PHASE, "Phase979 protocol phase mismatch")
    require(
        protocol.get("experiment") == PROTOCOL_EXPERIMENT,
        "Phase979 protocol experiment mismatch",
    )
    require(protocol.get("batch_size") == BATCH_SIZE, "truth batch size is not sealed at 8")
    require(
        protocol.get("holdout_loaded") is False,
        "Phase979 protocol reports old holdout access",
    )
    require(
        protocol.get("mechanism_authorized") is False,
        "Phase979 protocol unexpectedly authorizes old holdout mechanism work",
    )

    truth_identity = protocol.get("truth_dataset_identity")
    require(isinstance(truth_identity, dict), "protocol lacks truth_dataset_identity")
    require(
        isinstance(truth_identity.get("identity_sha256"), str),
        "truth_dataset_identity lacks identity_sha256",
    )
    identity_core = core.without_fields(truth_identity, "identity_sha256")
    require(
        truth_identity["identity_sha256"] == core.sha256_json(identity_core),
        "protocol truth dataset identity self-hash invalid",
    )
    require(truth_identity.get("n_pairs") == 128, "truth identity is not 128 pairs")
    require(
        truth_identity.get("expected_teacher_forced_rows") == 1024,
        "truth identity is not the sealed 1024-row two-block design",
    )
    require(
        protocol.get("expected_truth_rows") == {
            "development": EXPECTED_ROWS,
            "replication": EXPECTED_ROWS,
        },
        "protocol truth split denominators are not both 512",
    )
    truth_contract = protocol.get("truth_contract")
    require(isinstance(truth_contract, dict), "protocol lacks truth_contract")
    require(
        truth_contract.get("control_policy") == "hard_no_think"
        and truth_contract.get("teacher_forced") is True
        and truth_contract.get("sampling") is False
        and truth_contract.get("random_seed") is None,
        "protocol truth decoding contract mismatch",
    )
    require(
        truth_contract.get("gap_formula")
        == "g*=max_{j not in EOS} z_j - max_{e in EOS} z_e",
        "protocol truth EOS-gap formula mismatch",
    )
    require(
        truth_contract.get("development_precedes_replication") is True
        and truth_contract.get("replication_source_precommitted_and_preaudited") is True
        and truth_contract.get("replication_is_not_analyst_blind_holdout") is True
        and truth_contract.get(
            "replication_model_evaluation_requires_development_admission"
        ) is True,
        "protocol truth replication contract mismatch",
    )
    require(
        truth_contract.get("holdout_loaded") is False
        and truth_contract.get("mechanism_authorized") is False,
        "protocol truth contract crosses the old holdout/mechanism boundary",
    )

    scripts = protocol.get("phase979_script_hashes")
    require(isinstance(scripts, dict) and scripts, "protocol lacks Phase979 script hashes")
    verified: dict[Path, str] = {}
    for label, entry in scripts.items():
        require(isinstance(entry, dict), f"invalid Phase979 script entry: {label}")
        path = verify_file_commitment(entry, f"Phase979 script {label}")
        verified[path] = str(entry["sha256"])
    for required_path in (SCRIPT_PATH, CORE_PATH, DATASET_PATH):
        resolved = required_path.resolve()
        require(
            resolved in verified,
            f"protocol does not seal required truth runtime file: {required_path.name}",
        )
        require(
            verified[resolved] == core.sha256_file(required_path),
            f"sealed required runtime file changed: {required_path.name}",
        )

    commitments = protocol.get("phase978_commitments")
    require(isinstance(commitments, dict), "protocol lacks Phase978 commitments")
    phase978 = authenticate_phase978_no_go(commitments)
    assert_phase978_holdout_closed()
    return protocol, phase978


def authenticate_replication_admission(protocol: dict[str, Any]) -> dict[str, Any]:
    """Authenticate replication before importing/building its dataset block."""
    preimported = [
        name for name in sys.modules
        if name == DATASET_MODULE or name.endswith("." + DATASET_MODULE)
    ]
    require(
        not preimported,
        f"replication dataset module was imported before admission: {preimported}",
    )
    admission = core.load_json(TRUTH_ADMISSION_PATH, "truth development admission")
    verify_self_hash(
        admission,
        "admission_sha256",
        {"admission_sha256", "audited_at_utc"},
        "truth development admission",
    )
    require(
        admission.get("phase") == PHASE and admission.get("split") == "development",
        "truth admission phase/split mismatch",
    )
    require(
        admission.get("integrity_passed") is True
        and admission.get("truth_gate_passed") is True
        and admission.get("punctuation_gate_passed") is True,
        "truth development integrity/effect gates did not all pass",
    )
    require(
        admission.get("replication_authorized") is True,
        "truth development admission does not authorize replication",
    )
    require(
        admission.get("protocol_sha256") == protocol["protocol_sha256"],
        "truth admission/protocol SHA mismatch",
    )
    require(
        admission.get("dataset_identity") == protocol["truth_dataset_identity"],
        "truth admission/dataset identity mismatch",
    )
    require(
        admission.get("replication_model_evaluated") is False
        and admission.get("phase977_holdout_authorized") is False
        and admission.get("mechanism_authorized") is False
        and admission.get("holdout_loaded") is False,
        "truth admission crosses the replication/holdout/mechanism boundary",
    )
    require(
        admission.get("dataset_split_sha256")
        == protocol["truth_dataset_identity"]["development_pairs_sha256"],
        "truth admission development split hash mismatch",
    )

    development_manifest_path = manifest_path("development")
    development_rows_path = rows_path("development")
    development_manifest = core.load_json(
        development_manifest_path, "truth development manifest"
    )
    verify_self_hash(
        development_manifest,
        "manifest_sha256",
        MANIFEST_HASH_EXCLUSIONS,
        "truth development manifest",
    )
    require(
        admission.get("manifest_sha256") == development_manifest["manifest_sha256"]
        and admission.get("manifest_file_sha256")
        == core.sha256_file(development_manifest_path),
        "truth admission development manifest commitment mismatch",
    )
    require(
        development_manifest.get("protocol_sha256") == protocol["protocol_sha256"]
        and development_manifest.get("dataset_identity")
        == protocol["truth_dataset_identity"],
        "truth development manifest lineage mismatch",
    )
    require(development_rows_path.is_file(), "truth development rows are missing")
    require(
        admission.get("rows_file_sha256") == core.sha256_file(development_rows_path),
        "truth admission development rows commitment mismatch",
    )
    assert_phase978_holdout_closed()
    return admission


def load_dataset_block(
    split: str, protocol: dict[str, Any], replication_authenticated: bool,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any], str]:
    if split == "replication":
        require(replication_authenticated, "replication build attempted before admission")
    dataset = importlib.import_module(DATASET_MODULE)
    require(
        getattr(dataset, "STABLE_IDENTITY", None) == protocol["truth_dataset_identity"],
        "runtime truth dataset identity differs from protocol",
    )

    # Keep these literal split calls separate: development never executes the
    # replication builder branch, and replication can reach its branch only
    # after authenticate_replication_admission has returned successfully.
    if split == "development":
        pairs = dataset.build_pairs("development")
    elif split == "replication":
        require(replication_authenticated, "replication firewall is not open")
        pairs = dataset.build_pairs("replication")
    else:  # argparse already prevents this; keep the internal API fail-closed.
        raise RuntimeError(f"unknown truth split: {split}")

    require(isinstance(pairs, list), "truth build_pairs did not return a list")
    require(len(pairs) == EXPECTED_PAIRS, f"{split} is not exactly 64 pairs")
    require(all(pair.get("split") == split for pair in pairs), "truth split contamination")
    audit = dataset.audit_pairs(pairs)
    require(isinstance(audit, dict), "truth dataset audit did not return an object")
    require(audit.get("ok") is True and audit.get("passed") is True, "truth audit failed")
    require(not audit.get("errors"), f"truth audit errors: {audit.get('errors')}")
    require(audit.get("n_pairs") == EXPECTED_PAIRS, "truth audit pair count mismatch")
    require(
        audit.get("expected_teacher_forced_rows") == EXPECTED_ROWS,
        "truth audit row denominator mismatch",
    )
    pairs_sha256 = core.sha256_json(pairs)
    expected_sha = protocol["truth_dataset_identity"][f"{split}_pairs_sha256"]
    require(pairs_sha256 == expected_sha, f"{split} pair block differs from protocol")
    return dataset, pairs, audit, pairs_sha256


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_full": sys.version,
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "version_source": "installed_distribution_metadata_only",
    }


def local_model_artifact_identity(model_dir: Path) -> dict[str, Any]:
    """Hash local configuration, tokenizer artifacts, and every weight shard."""
    model_dir = model_dir.resolve()
    require(model_dir.is_dir(), f"local Qwen3 directory missing: {model_dir}")
    required = {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
    }
    candidates: set[str] = set()
    for pattern in ("*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"):
        candidates.update(path.name for path in model_dir.glob(pattern) if path.is_file())
    require(required.issubset(candidates), "local Qwen3 required artifacts are missing")
    require(any(name.endswith(".safetensors") for name in candidates), "no Qwen3 weights found")
    require(
        any(name in candidates for name in ("tokenizer.json", "tokenizer.model", "vocab.json")),
        "no Qwen3 tokenizer vocabulary artifact found",
    )
    files: dict[str, Any] = {}
    for name in sorted(candidates):
        path = model_dir / name
        files[name] = {"bytes": int(path.stat().st_size), "sha256": core.sha256_file(path)}
    identity_core = {"model_dir": str(model_dir), "files": files}
    return {**identity_core, "identity_sha256": core.sha256_json(identity_core)}


def authenticate_runtime_and_model(
    protocol: dict[str, Any], actual_runtime: dict[str, str],
    model_dir: Path, actual_identity: dict[str, Any],
) -> None:
    expected_runtime = protocol.get("runtime_versions")
    if expected_runtime is not None:
        require(expected_runtime == actual_runtime, "runtime differs from protocol seal")

    expected = protocol.get("local_model_artifact_identity")
    if expected is None:
        expected = protocol.get("model_identity")
    if expected is None:
        return
    require(isinstance(expected, dict), "protocol model identity is not an object")
    expected_path = expected.get("model_dir", expected.get("path"))
    if isinstance(expected_path, str):
        raw = Path(expected_path)
        resolved = raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()
        require(resolved == model_dir.resolve(), "Qwen3 registry path differs from protocol")
    files = expected.get("files")
    require(isinstance(files, dict) and files, "protocol model identity lacks files")
    actual_files = actual_identity["files"]
    for name, entry in files.items():
        require(isinstance(entry, dict), f"invalid protocol model artifact: {name}")
        require(name in actual_files, f"sealed Qwen3 artifact missing: {name}")
        if "bytes" in entry:
            require(int(entry["bytes"]) == actual_files[name]["bytes"], f"size changed: {name}")
        if "sha256" in entry:
            require(str(entry["sha256"]) == actual_files[name]["sha256"], f"hash changed: {name}")


def tokenize_ids(tok, text: str) -> list[int]:
    values = tok(text, add_special_tokens=False, return_attention_mask=False).input_ids
    return [int(value) for value in values]


def prepare_cases(
    tok, pairs: list[dict[str, Any]], split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_ids = {candidate: core.single_token_id(tok, candidate) for candidate in CANDIDATES}
    period_id = core.single_token_id(tok, ".")
    think_open_id = core.single_token_id(tok, "<think>")
    think_close_id = core.single_token_id(tok, "</think>")
    cases: list[dict[str, Any]] = []
    prefix_probes: list[dict[str, Any]] = []

    for pair in pairs:
        pair_id = str(pair["id"])
        for side in PROMPT_SIDES:
            prompt = str(pair["prompts"][side])
            user_prompt, rendered, input_ids = core.render_prefix(
                tok, {"prompt": prompt}, "hard_no_think"
            )
            require(user_prompt == prompt, f"{pair_id}/{side}: hard-no-think changed prompt")
            require(input_ids, f"{pair_id}/{side}: empty official rendered prefix")
            opens = [index for index, value in enumerate(input_ids) if value == think_open_id]
            closes = [index for index, value in enumerate(input_ids) if value == think_close_id]
            require(
                len(opens) == 1 and len(closes) == 1 and opens[0] < closes[0],
                f"{pair_id}/{side}: official hard-no-think prefix lacks one ordered think block",
            )
            hidden_text = tok.decode(
                input_ids[opens[0] + 1:closes[0]], skip_special_tokens=False
            ).strip()
            require(not hidden_text, f"{pair_id}/{side}: hard-no-think block is not empty")
            rendered_sha256 = core.sha256_json(rendered)
            prefix_probes.append({
                "pair_id": pair_id,
                "prompt_side": side,
                "prompt_id": str(pair["prompt_ids"][side]),
                "rendered_prefix_sha256": rendered_sha256,
                "input_ids_sha256": core.sha256_json(input_ids),
                "prompt_len": len(input_ids),
            })

            for candidate in CANDIDATES:
                bare_text = candidate
                period_text = candidate + "."
                bare_full = tokenize_ids(tok, rendered + bare_text)
                period_full = tokenize_ids(tok, rendered + period_text)
                require(
                    bare_full[:len(input_ids)] == input_ids,
                    f"{pair_id}/{side}/{candidate}: bare answer changed rendered prefix tokens",
                )
                require(
                    period_full[:len(input_ids)] == input_ids,
                    f"{pair_id}/{side}/{candidate}: period answer changed rendered prefix tokens",
                )
                bare_suffix = bare_full[len(input_ids):]
                period_suffix = period_full[len(input_ids):]
                require(
                    bare_suffix == [label_ids[candidate]],
                    f"{pair_id}/{side}/{candidate}: bare answer is not its one-token label",
                )
                require(
                    period_full[:len(bare_full)] == bare_full,
                    f"{pair_id}/{side}/{candidate}: punctuation does not preserve bare prefix",
                )
                require(
                    period_suffix == [label_ids[candidate], period_id],
                    f"{pair_id}/{side}/{candidate}: period is not one pure common suffix token",
                )

                common = {
                    "pair_id": pair_id,
                    "task": str(pair["task"]),
                    "split": split,
                    "prompt_side": side,
                    "prompt_id": str(pair["prompt_ids"][side]),
                    "candidate": candidate,
                    "is_correct": bool(pair["truth_table"][side][candidate]),
                    "input_ids": input_ids,
                    "rendered_prefix_sha256": rendered_sha256,
                    "prompt_len": len(input_ids),
                }
                cases.append({
                    **common,
                    "punctuation": "bare",
                    "answer_text": bare_text,
                    "answer_prefix_ids": bare_full,
                    "answer_suffix_ids": bare_suffix,
                })
                cases.append({
                    **common,
                    "punctuation": "period",
                    "answer_text": period_text,
                    "answer_prefix_ids": period_full,
                    "answer_suffix_ids": period_suffix,
                })

    require(len(cases) == EXPECTED_ROWS, "truth grid is not exactly 512 rows")
    keys = [core.truth_key(case) for case in cases]
    require(len(set(keys)) == EXPECTED_ROWS, "truth grid contains duplicate keys")
    correctness = sum(int(case["is_correct"]) for case in cases)
    require(correctness == EXPECTED_ROWS // 2, "truth grid correctness is not balanced")
    punctuation_counts = {
        state: sum(case["punctuation"] == state for case in cases)
        for state in PUNCTUATIONS
    }
    require(
        set(punctuation_counts.values()) == {EXPECTED_ROWS // 2},
        "truth punctuation cells are not balanced",
    )
    token_audit = {
        "official_control": "hard_no_think",
        "enable_thinking": False,
        "label_token_ids": label_ids,
        "period_token_id": period_id,
        "think_open_id": think_open_id,
        "think_close_id": think_close_id,
        "n_official_prefixes": len(prefix_probes),
        "n_teacher_forced_states": len(cases),
        "bare_labels_are_one_token": True,
        "period_is_same_pure_one_token_suffix_everywhere": True,
        "answer_states_preserve_official_prefix": True,
        "prefix_probes_sha256": core.sha256_json(prefix_probes),
        "case_token_identity_sha256": core.sha256_json([
            {
                "key": list(core.truth_key(case)),
                "input_ids": case["input_ids"],
                "answer_prefix_ids": case["answer_prefix_ids"],
                "answer_suffix_ids": case["answer_suffix_ids"],
            }
            for case in cases
        ]),
    }
    return cases, token_audit


def get_eos_ids(model, tok) -> list[int]:
    values = (
        getattr(tok, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
    )
    output: list[int] = []
    for value in values:
        if value is None:
            continue
        candidates = value if isinstance(value, (list, tuple, set)) else [value]
        for candidate in candidates:
            candidate = int(candidate)
            if candidate not in output:
                output.append(candidate)
    require(output, "Qwen3 tokenizer/model exposes no EOS token IDs")
    require(all(0 <= value < len(tok) for value in output), "EOS ID outside vocabulary")
    return output


def authenticate_tokenizer_audit(
    protocol: dict[str, Any], tok, token_audit: dict[str, Any],
) -> None:
    sealed = protocol.get("tokenizer_audit")
    require(isinstance(sealed, dict), "protocol lacks tokenizer_audit")
    require(sealed.get("tokenizer_class") == type(tok).__name__,
            "tokenizer class differs from protocol")
    require(sealed.get("tokenizer_length") == len(tok),
            "tokenizer length differs from protocol")
    require(
        sealed.get("chat_template_sha256")
        == core.sha256_json(str(getattr(tok, "chat_template", ""))),
        "tokenizer chat template differs from protocol",
    )
    require(sealed.get("eos_token_id") == int(tok.eos_token_id),
            "tokenizer EOS ID differs from protocol")
    require(sealed.get("pad_token_id") == int(tok.pad_token_id),
            "tokenizer pad ID differs from protocol")
    expected_special = sealed.get("special_token_ids")
    require(isinstance(expected_special, dict),
            "protocol tokenizer audit lacks special_token_ids")
    actual_special = {
        "A": token_audit["label_token_ids"]["A"],
        "B": token_audit["label_token_ids"]["B"],
        "period": token_audit["period_token_id"],
        "think_open": token_audit["think_open_id"],
        "think_close": token_audit["think_close_id"],
    }
    require(expected_special == actual_special,
            "truth special token IDs differ from protocol")
    require(
        sealed.get("truth_bare_period_context_pairs") == 512
        and sealed.get("truth_all_periods_are_same_pure_one_token_suffix") is True,
        "protocol tokenizer audit did not freeze the full truth suffix gate",
    )


def make_manifest(
    split: str, protocol: dict[str, Any], phase978: dict[str, Any],
    admission: dict[str, Any] | None, pairs: list[dict[str, Any]],
    pairs_audit: dict[str, Any], pairs_sha256: str, cases: list[dict[str, Any]],
    token_audit: dict[str, Any], model, tok, device, eos_ids: list[int],
    model_identity: dict[str, Any], actual_runtime: dict[str, str],
) -> dict[str, Any]:
    pad_id = tok.pad_token_id
    require(pad_id is not None, "Qwen3 tokenizer has no pad token ID")
    model_vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    require(
        isinstance(model_vocab_size, int) and not isinstance(model_vocab_size, bool)
        and model_vocab_size > 0,
        "Qwen3 model config lacks a valid vocabulary size",
    )
    special_token_ids = {
        "A": token_audit["label_token_ids"]["A"],
        "B": token_audit["label_token_ids"]["B"],
        "period": token_audit["period_token_id"],
        "think_open": token_audit["think_open_id"],
        "think_close": token_audit["think_close_id"],
    }
    committed_token_ids = [*eos_ids, *special_token_ids.values(), int(pad_id)]
    require(
        all(
            isinstance(value, int) and not isinstance(value, bool)
            and 0 <= value < model_vocab_size
            for value in committed_token_ids
        ),
        "sealed EOS/label/special token ID is outside the model vocabulary",
    )
    manifest_core = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "split": split,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": core.sha256_file(PROTOCOL_PATH),
        "runner_path": relative_path(SCRIPT_PATH),
        "runner_sha256": core.sha256_file(SCRIPT_PATH),
        "boundary_core_path": relative_path(CORE_PATH),
        "boundary_core_sha256": core.sha256_file(CORE_PATH),
        "dataset_module_path": relative_path(DATASET_PATH),
        "dataset_module_sha256": core.sha256_file(DATASET_PATH),
        "dataset_identity": protocol["truth_dataset_identity"],
        "dataset_split_sha256": pairs_sha256,
        "pairs_audit": pairs_audit,
        "pairs_audit_sha256": core.sha256_json(pairs_audit),
        "n_pairs": len(pairs),
        "expected_rows": EXPECTED_ROWS,
        "actual_case_count": len(cases),
        "replication_authorized": bool(admission is not None),
        "truth_admission_sha256": None if admission is None else admission["admission_sha256"],
        "truth_admission_file_sha256": (
            None if admission is None else core.sha256_file(TRUTH_ADMISSION_PATH)
        ),
        "phase978_no_go": phase978,
        "model_name": MODEL_NAME,
        "model_class": type(model).__name__,
        "model_dtype": str(getattr(model, "dtype", "unknown")),
        "model_vocab_size": int(model_vocab_size),
        "device": str(device),
        "device_type": getattr(device, "type", str(device).split(":")[0]),
        "model_artifact_identity": model_identity,
        "runtime_versions": actual_runtime,
        "tokenizer_class": type(tok).__name__,
        "tokenizer_length": len(tok),
        "tokenizer_padding_side": tok.padding_side,
        "pad_token_id": int(pad_id),
        "eos_token_ids": eos_ids,
        "special_token_ids": special_token_ids,
        "token_audit": token_audit,
        "batch_size": BATCH_SIZE,
        "control_policy": "hard_no_think",
        "teacher_forced": True,
        "teacher_forcing": True,
        "sampling": False,
        "sampling_performed": False,
        "rng_policy": "not_applicable_no_sampling",
        "model_mode": "eval_inference_mode",
        "tf32_enabled": False,
        "left_padding": True,
        "explicit_attention_mask": True,
        "explicit_position_ids": True,
        "eos_definition": "max_over_all_sealed_eos_ids",
        "gap_definition": "max_non_eos_logit_minus_max_eos_logit",
        "selected_eos_definition": "argmax_eos_logit",
        "eos_rank_definition": "one_plus_count_all_vocab_logits_strictly_greater",
        "phase978_holdout_loaded": False,
        "holdout_loaded": False,
        "mechanism_authorized": False,
        "replication_execution_contract": (
            "preregistered conditional block; no replication model forward or output "
            "before authenticated truth development admission"
        ),
    }
    document = {
        **manifest_core,
        "manifest_sha256": core.sha256_json(manifest_core),
        "created_at_utc": core.utc_now(),
    }
    return document


def install_or_validate_manifest(split: str, expected: dict[str, Any]) -> dict[str, Any]:
    path = manifest_path(split)
    if not path.exists():
        core.atomic_write_json(path, expected)
        return expected
    existing = core.load_json(path, f"truth {split} manifest")
    verify_self_hash(existing, "manifest_sha256", MANIFEST_HASH_EXCLUSIONS, f"truth {split} manifest")
    require(
        existing["manifest_sha256"] == expected["manifest_sha256"],
        f"existing truth {split} manifest differs from current sealed run",
    )
    return existing


def validate_row(
    row: dict[str, Any], case: dict[str, Any], manifest: dict[str, Any],
) -> None:
    key = core.truth_key(row)
    verify_self_hash(row, "row_sha256", {"row_sha256"}, f"truth row {key}")
    require(row.get("schema_version") == SCHEMA_VERSION, f"row schema mismatch: {key}")
    require(row.get("phase") == PHASE, f"row phase mismatch: {key}")
    require(row.get("manifest_sha256") == manifest["manifest_sha256"], f"row manifest mismatch: {key}")
    require(row.get("protocol_sha256") == manifest["protocol_sha256"], f"row protocol mismatch: {key}")
    for field in (
        "pair_id", "task", "split", "prompt_side", "prompt_id", "candidate",
        "punctuation", "is_correct", "answer_text", "input_ids",
        "answer_prefix_ids", "answer_suffix_ids", "rendered_prefix_sha256",
        "prompt_len",
    ):
        require(row.get(field) == case[field], f"row {field} mismatch: {key}")
    require(row.get("eos_ids") == manifest["eos_token_ids"], f"row EOS registry mismatch: {key}")
    require(row.get("holdout_loaded") is False, f"row reports holdout access: {key}")
    require(row.get("phase978_holdout_loaded") is False,
            f"row reports Phase978 holdout access: {key}")
    require(row.get("mechanism_authorized") is False, f"row crosses mechanism boundary: {key}")
    require(row.get("teacher_forcing") is True, f"row is not teacher forced: {key}")
    require(row.get("sampling_performed") is False, f"row reports sampling: {key}")

    model_vocab_size = manifest.get("model_vocab_size")
    require(
        isinstance(model_vocab_size, int) and not isinstance(model_vocab_size, bool)
        and model_vocab_size > 0,
        "truth manifest lacks a valid model_vocab_size",
    )
    eos_ids = {int(value) for value in manifest["eos_token_ids"]}
    for field in ("selected_eos_id", "max_non_eos_id", "top1_id", "eos_rank"):
        require(
            isinstance(row.get(field), int) and not isinstance(row.get(field), bool),
            f"row {field} is not a strict integer: {key}",
        )
    selected_eos_id = int(row["selected_eos_id"])
    max_non_eos_id = int(row["max_non_eos_id"])
    top1_id = int(row["top1_id"])
    require(
        all(
            0 <= value < model_vocab_size
            for value in (selected_eos_id, max_non_eos_id, top1_id)
        ),
        f"row argmax token ID outside model vocabulary: {key}",
    )
    require(selected_eos_id in eos_ids, f"selected EOS is not sealed: {key}")
    require(max_non_eos_id not in eos_ids, f"invalid non-EOS argmax: {key}")
    for field in ("eos_logit", "max_non_eos_logit", "gap", "eos_probability", "eos_probability_total"):
        core.finite_number(row.get(field), f"{key}/{field}")
    eos_logit = float(row["eos_logit"])
    max_non_eos_logit = float(row["max_non_eos_logit"])
    require(
        math.isclose(float(row["gap"]), max_non_eos_logit - eos_logit, rel_tol=0.0, abs_tol=1e-6),
        f"row gap primitive mismatch: {key}",
    )
    selected_probability = float(row["eos_probability"])
    total_probability = float(row["eos_probability_total"])
    require(0.0 <= selected_probability <= total_probability <= 1.0 + 1e-7, f"invalid EOS probabilities: {key}")
    require(1 <= int(row["eos_rank"]) <= model_vocab_size, f"invalid EOS rank: {key}")
    require(isinstance(row.get("eos_top1"), bool), f"EOS top1 is not Boolean: {key}")
    require(row["eos_top1"] == (top1_id in eos_ids), f"EOS top1 flag mismatch: {key}")
    require(row.get("left_padded_batch_with_explicit_attention_mask") is True, f"padding contract absent: {key}")


def load_existing_rows(
    split: str, manifest: dict[str, Any], cases_by_key: dict[tuple[str, str, str, str], dict[str, Any]],
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    path = rows_path(split)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    payload = path.read_bytes()
    require(payload.endswith(b"\n"), f"truth {split} JSONL lacks final newline")
    output: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for line_number, raw in enumerate(payload.splitlines(), 1):
        require(raw.strip(), f"blank truth JSONL line {line_number}")
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"malformed truth JSONL line {line_number}") from exc
        require(isinstance(row, dict), f"truth JSONL line {line_number} is not an object")
        key = core.truth_key(row)
        require(key in cases_by_key, f"truth JSONL key outside sealed grid: {key}")
        require(key not in output, f"duplicate truth JSONL key: {key}")
        validate_row(row, cases_by_key[key], manifest)
        output[key] = row
    require(len(output) <= EXPECTED_ROWS, "truth JSONL exceeds sealed denominator")
    return output


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    require(size == BATCH_SIZE, "truth batch size must remain eight")
    require(len(values) % size == 0, "truth grid must divide into full batches")
    for start in range(0, len(values), size):
        yield values[start:start + size]


def score_batch(
    model, tok, device, jobs: list[dict[str, Any]], eos_ids: list[int],
    model_vocab_size: int,
) -> list[dict[str, Any]]:
    import torch

    require(len(jobs) == BATCH_SIZE, "formal truth scoring requires exactly eight rows per batch")
    sequences = [case["answer_prefix_ids"] for case in jobs]
    require(all(sequence for sequence in sequences), "empty teacher-forced answer prefix")
    max_len = max(len(sequence) for sequence in sequences)
    pad_id = tok.pad_token_id
    require(pad_id is not None, "tokenizer pad ID vanished")
    input_tensor = torch.full(
        (BATCH_SIZE, max_len), int(pad_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (BATCH_SIZE, max_len), dtype=torch.long, device=device
    )
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        input_tensor[index, max_len - length:] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[index, max_len - length:] = 1
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    expected_lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    require(
        torch.equal(attention_mask.sum(dim=-1).cpu(), expected_lengths),
        "explicit truth attention mask does not match sequence lengths",
    )

    with torch.inference_mode():
        output = model(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
            logits_to_keep=1,
        )
    logits = output.logits[:, -1, :].float()
    require(
        logits.ndim == 2
        and logits.shape[0] == BATCH_SIZE
        and logits.shape[1] == model_vocab_size,
        f"unexpected Qwen3 logits shape {tuple(logits.shape)}; "
        f"expected ({BATCH_SIZE}, {model_vocab_size})",
    )
    require(bool(torch.isfinite(logits).all().item()), "Qwen3 produced non-finite next-token logits")
    eos_tensor = torch.tensor(eos_ids, dtype=torch.long, device=logits.device)
    eos_values = logits.index_select(1, eos_tensor)
    eos_offsets = torch.argmax(eos_values, dim=1)
    selected_eos_ids = eos_tensor.index_select(0, eos_offsets)
    eos_logits = eos_values.gather(1, eos_offsets.unsqueeze(1)).squeeze(1)

    non_eos = logits.clone()
    non_eos[:, eos_tensor] = -torch.inf
    max_non_eos_logits, max_non_eos_ids = torch.max(non_eos, dim=1)
    probabilities = torch.softmax(logits, dim=1)
    selected_eos_probabilities = probabilities.gather(
        1, selected_eos_ids.unsqueeze(1)
    ).squeeze(1)
    eos_probability_totals = probabilities.index_select(1, eos_tensor).sum(dim=1)
    top1_ids = torch.argmax(logits, dim=1)
    eos_ranks = 1 + torch.sum(logits > eos_logits.unsqueeze(1), dim=1)
    gaps = max_non_eos_logits - eos_logits
    eos_top1 = torch.isin(top1_ids, eos_tensor)

    primitives = zip(
        selected_eos_ids.detach().cpu().tolist(),
        eos_logits.detach().cpu().tolist(),
        selected_eos_probabilities.detach().cpu().tolist(),
        eos_probability_totals.detach().cpu().tolist(),
        max_non_eos_ids.detach().cpu().tolist(),
        max_non_eos_logits.detach().cpu().tolist(),
        gaps.detach().cpu().tolist(),
        eos_ranks.detach().cpu().tolist(),
        top1_ids.detach().cpu().tolist(),
        eos_top1.detach().cpu().tolist(),
    )
    results: list[dict[str, Any]] = []
    eos_set = set(eos_ids)
    for values in primitives:
        (
            selected_id, eos_logit, eos_probability, eos_probability_total,
            non_eos_id, non_eos_logit, gap, eos_rank, top1_id, is_eos_top1,
        ) = values
        result = {
            "selected_eos_id": int(selected_id),
            "selected_eos_token": tok.convert_ids_to_tokens(int(selected_id)),
            "eos_logit": core.finite_number(eos_logit, "eos_logit"),
            "eos_probability": core.finite_number(eos_probability, "eos_probability"),
            "eos_probability_total": core.finite_number(eos_probability_total, "eos_probability_total"),
            "max_non_eos_id": int(non_eos_id),
            "max_non_eos_token": tok.convert_ids_to_tokens(int(non_eos_id)),
            "max_non_eos_logit": core.finite_number(non_eos_logit, "max_non_eos_logit"),
            "gap": core.finite_number(gap, "gap"),
            "eos_rank": int(eos_rank),
            "top1_id": int(top1_id),
            "top1_token": tok.convert_ids_to_tokens(int(top1_id)),
            "eos_top1": bool(is_eos_top1),
        }
        require(result["selected_eos_id"] in eos_set, "batched selected EOS is invalid")
        require(result["max_non_eos_id"] not in eos_set, "batched non-EOS argmax is EOS")
        results.append(result)
    require(len(results) == BATCH_SIZE, "truth metric batch length mismatch")
    del output, logits, eos_values, non_eos, probabilities
    del input_tensor, attention_mask, position_ids
    return results


def build_row(
    case: dict[str, Any], metrics: dict[str, Any], manifest: dict[str, Any], batch_index: int,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "manifest_sha256": manifest["manifest_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        **case,
        "eos_ids": manifest["eos_token_ids"],
        **metrics,
        "batch_index": int(batch_index),
        "batch_size": BATCH_SIZE,
        "teacher_forcing": True,
        "sampling_performed": False,
        "left_padded_batch_with_explicit_attention_mask": True,
        "explicit_position_ids": True,
        "holdout_loaded": False,
        "phase978_holdout_loaded": False,
        "mechanism_authorized": False,
    }
    return {**payload, "row_sha256": core.sha256_json(payload)}


def write_status(
    split: str, context: dict[str, Any], state: str, complete: bool,
    error: BaseException | None = None,
) -> dict[str, Any]:
    require(not complete or state == "COMPLETE", "only COMPLETE status may be complete")
    require(not complete or int(context.get("completed_rows", 0)) == EXPECTED_ROWS,
            "complete status requires exactly 512 rows")
    status_core = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "split": split,
        "state": state,
        "complete": bool(complete),
        "expected_rows": EXPECTED_ROWS,
        "completed_rows": int(context.get("completed_rows", 0)),
        "protocol_sha256": context.get("protocol_sha256"),
        "manifest_sha256": context.get("manifest_sha256"),
        "truth_admission_sha256": context.get("truth_admission_sha256"),
        "protocol_authenticated": bool(context.get("protocol_authenticated", False)),
        "replication_authenticated": bool(context.get("replication_authenticated", False)),
        "dataset_block_built": bool(context.get("dataset_block_built", False)),
        "model_weights_loaded": bool(context.get("model_weights_loaded", False)),
        "model_forward_performed": bool(context.get("model_forward_performed", False)),
        "replication_model_evaluated": bool(
            split == "replication" and int(context.get("completed_rows", 0)) > 0
        ),
        "sampling_performed": False,
        "holdout_loaded": False,
        "phase978_open_receipt_exists": PHASE978_OPEN_RECEIPT_PATH.exists(),
        "mechanism_authorized": False,
        "elapsed_seconds_this_invocation": float(context.get("elapsed_seconds", 0.0)),
        "error_type": None if error is None else type(error).__name__,
        "error_message": None if error is None else str(error),
    }
    document = {
        **status_core,
        "status_sha256": core.sha256_json(status_core),
        "updated_at_utc": core.utc_now(),
    }
    core.atomic_write_json(status_path(split), document)
    return document


def run(split: str, context: dict[str, Any]) -> dict[str, Any]:
    require(split in SPLITS, f"unsupported truth split: {split}")
    require(core.PHASE == PHASE and core.SCHEMA_VERSION == SCHEMA_VERSION,
            "Phase979 shared core identity mismatch")
    require(core.BATCH_SIZE == BATCH_SIZE, "Phase979 shared batch size mismatch")
    assert_phase978_holdout_closed()
    started = time.monotonic()

    protocol, phase978 = authenticate_protocol()
    context.update({
        "protocol_authenticated": True,
        "protocol_sha256": protocol["protocol_sha256"],
    })
    admission: dict[str, Any] | None = None
    if split == "replication":
        admission = authenticate_replication_admission(protocol)
        context.update({
            "replication_authenticated": True,
            "truth_admission_sha256": admission["admission_sha256"],
        })

    # CUDA is required before the selected data block is constructed.  In
    # particular, a replication invocation without CUDA stops after admission
    # authentication and cannot create replication pairs, rows, or a manifest.
    import torch
    require(torch.cuda.is_available(), "formal Phase979 truth run requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    _dataset, pairs, pairs_audit, pairs_sha256 = load_dataset_block(
        split, protocol, replication_authenticated=bool(admission is not None)
    )
    context["dataset_block_built"] = True
    assert_phase978_holdout_closed()

    from model_utils import MODEL_CONFIGS, load_model, release_model

    model_dir = Path(MODEL_CONFIGS[MODEL_NAME]["path"]).resolve()
    actual_runtime = runtime_versions()
    model_identity = local_model_artifact_identity(model_dir)
    authenticate_runtime_and_model(
        protocol, actual_runtime, model_dir, model_identity
    )

    model = None
    try:
        model, tok, device = load_model(
            MODEL_NAME, dtype=torch.bfloat16, use_8bit=False
        )
        context["model_weights_loaded"] = True
        require(
            getattr(device, "type", str(device).split(":")[0]) == "cuda",
            f"Qwen3 did not load wholly on CUDA: {device}",
        )
        require(type(model).__name__ == MODEL_CONFIGS[MODEL_NAME]["arch"],
                f"unexpected local Qwen3 class: {type(model).__name__}")
        model.eval()
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            require(tok.eos_token_id is not None, "Qwen3 has neither pad nor EOS token")
            tok.pad_token = tok.eos_token
        require(tok.padding_side == "left", "Qwen3 tokenizer is not explicitly left padded")

        cases, token_audit = prepare_cases(tok, pairs, split)
        authenticate_tokenizer_audit(protocol, tok, token_audit)
        eos_ids = get_eos_ids(model, tok)
        manifest = install_or_validate_manifest(
            split,
            make_manifest(
                split, protocol, phase978, admission, pairs, pairs_audit,
                pairs_sha256, cases, token_audit, model, tok, device, eos_ids,
                model_identity, actual_runtime,
            ),
        )
        context["manifest_sha256"] = manifest["manifest_sha256"]
        cases_by_key = {core.truth_key(case): case for case in cases}
        rows = load_existing_rows(split, manifest, cases_by_key)
        context["completed_rows"] = len(rows)
        context["elapsed_seconds"] = time.monotonic() - started
        write_status(split, context, "RUNNING", complete=False)

        total_batches = EXPECTED_ROWS // BATCH_SIZE
        for batch_index, jobs in enumerate(chunks(cases, BATCH_SIZE), 1):
            assert_phase978_holdout_closed()
            batch_keys = [core.truth_key(case) for case in jobs]
            if all(key in rows for key in batch_keys):
                continue
            metrics = score_batch(
                model, tok, device, jobs, eos_ids,
                int(manifest["model_vocab_size"]),
            )
            context["model_forward_performed"] = True
            built = [
                build_row(case, metric, manifest, batch_index)
                for case, metric in zip(jobs, metrics)
            ]
            for row, case in zip(built, jobs):
                validate_row(row, case, manifest)
                key = core.truth_key(row)
                if key in rows:
                    require(
                        rows[key]["row_sha256"] == row["row_sha256"],
                        f"partial-batch replay changed truth row: {key}",
                    )
                else:
                    core.append_jsonl(rows_path(split), row)
                    rows[key] = row
            context["completed_rows"] = len(rows)
            context["elapsed_seconds"] = time.monotonic() - started
            write_status(split, context, "RUNNING", complete=False)
            log(
                f"truth {split}: batch {batch_index}/{total_batches}, "
                f"rows {len(rows)}/{EXPECTED_ROWS}"
            )

        require(set(rows) == set(cases_by_key), "truth output grid is incomplete or contaminated")
        require(len(rows) == EXPECTED_ROWS, "truth output does not contain exactly 512 rows")
        # Re-read and authenticate the durable JSONL before declaring completion.
        durable = load_existing_rows(split, manifest, cases_by_key)
        require(len(durable) == EXPECTED_ROWS, "durable truth JSONL is incomplete")
        context["completed_rows"] = len(durable)
        context["elapsed_seconds"] = time.monotonic() - started
        assert_phase978_holdout_closed()
        return write_status(split, context, "COMPLETE", complete=True)
    finally:
        if model is not None:
            try:
                release_model(model)
                model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the sealed Phase979 Qwen3 CUDA teacher-forced truth x "
            "punctuation diagnostic."
        )
    )
    parser.add_argument("--split", required=True, choices=SPLITS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    context: dict[str, Any] = {
        "completed_rows": 0,
        "elapsed_seconds": 0.0,
    }
    started = time.monotonic()
    try:
        status = run(args.split, context)
        print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        context["elapsed_seconds"] = time.monotonic() - started
        try:
            write_status(args.split, context, "FAILED", complete=False, error=exc)
        except Exception as status_exc:
            print(f"failed to write fail-closed status: {status_exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
