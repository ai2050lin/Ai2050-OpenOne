#!/usr/bin/env python3
"""Freeze the Phase578 scorer/runner bridge without loading models or CUDA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# This stage is intentionally CPU/read-only with respect to models.  Set the
# visibility guard before any local module can be imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests/glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

import phase578_gpt5_behavior_scorer as scorer  # noqa: E402


PHASE = "Phase578"
OUT_DIR = ROOT / "tests/glm5/result/phase578_gpt5_runner_scorer_protocol"
PHASE577_DIR = ROOT / "tests/glm5/result/phase577_gpt5_natural_behavior_protocol"
DEVELOPMENT_PATH = PHASE577_DIR / "phase577_development_cases.jsonl"
MANIFEST_NAME = "phase578_development_prompt_manifest.jsonl"
PROTOCOL_NAME = "phase578_preregistered_runner_protocol.json"
SELF_TEST_NAME = "phase578_scorer_self_test.json"
STAGE_COMMIT_NAME = "phase578_stage_commit.json"
AUDIT_NAME = "phase578_independent_audit.json"
FREEZE_NAME = "phase578_freeze_commit.json"
MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
FORMAL_PYTHON = Path(
    r"C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe"
)
FORMAL_PYTHON_SHA256 = (
    "0f11fb7422fa347b7609ba0964ceccef3c8fa9f15230c37b9ec27668e68e8a8a"
)

SOURCE_RELATIVES = (
    "tests/glm5/phase578_gpt5_behavior_scorer.py",
    "tests/glm5/phase578_gpt5_development_runner.py",
    "tests/glm5/phase578_gpt5_behavior_analysis.py",
    "tests/glm5/phase578_gpt5_runner_audit.py",
    "tests/glm5/phase578_gpt5_runner_protocol.py",
)

UPSTREAM_EXPECTED = {
    "phase577_development": (
        PHASE577_DIR / "phase577_development_cases.jsonl",
        "4c40ea882e1c0e2994441f64fe37dc531b326ab01edc8f24934866b127fd8a5c",
    ),
    "phase577_protocol": (
        PHASE577_DIR / "phase577_preregistered_protocol.json",
        "aaad6a29ae537255aa04df51c50c62a3f22c35943e05d30ab9897ea362d6df84",
    ),
    "phase577_stage_commit": (
        PHASE577_DIR / "phase577_stage_commit.json",
        "721a5fce40aee384f954e91570b6c7a52ed5e8fa50e2418fc00aa1858120204b",
    ),
    "phase577_tokenizer_precheck": (
        PHASE577_DIR / "phase577_tokenizer_precheck.json",
        "09975b0333ea59db6ba57e874596d2ac7ef5c1fd4d2ff959b389f9f4129bc5cf",
    ),
    "phase577_independent_audit": (
        PHASE577_DIR / "phase577_independent_audit.json",
        "a19a65296906cd32b4d359eba58adbb2024e9d52b1e9c114c7ecbed782139bc3",
    ),
    "phase577_final_freeze": (
        PHASE577_DIR / "phase577_freeze_commit.json",
        "4690654ac42259adcbb733028d147ab2ac8cd211d5e29eedecfa545c0c6b4533",
    ),
    "phase576r2_engineering_qualification": (
        ROOT / "tests/glm5/result/phase576r2_gpt5_fruit_structure/phase576_engineering_qualification.json",
        "d4f78cdbe665e04db345ca3e485d97172c8365c4441f72f7099ab79b207890b9",
    ),
    "phase576r2_engineering_receipt": (
        ROOT / "tests/glm5/result/phase576r2_gpt5_fruit_structure/engineering_qualification_execution/execution_receipt.json",
        "49e84995f6922dd4fc26fe79794741663fbc6acaf774390a5b68ea02ebb10417",
    ),
    "phase576r2_cleanup_qualification": (
        ROOT / "tests/glm5/result/phase576r2_gpt5_fruit_structure/phase576r2_cuda_cleanup_qualification.json",
        "efd5625cbe8c53f5f7400649ca33a33763a64de5087bec1543e9041ab9a59f1c",
    ),
    "cross_model_engine": (
        ROOT / "tests/glm5/phase983_cross_model_engine.py",
        "e345daf3c3eae289eb7a71b8a741eeaf3a11c6897d009a5f9d90a386b23eef6f",
    ),
    "phase576r2_runtime_reference": (
        ROOT / "tests/glm5/phase576r2_gpt5_fruit_runtime.py",
        "49ee8241ffb92912d66397ab2c1927ac732ec91a9bc73a5f404f1dec38a69013",
    ),
    "model_registry": (
        ROOT / "tests/gpt5/model_registry.py",
        "84c30398a9effa47791635fd25662426164460e036e59bb83a38c855db370864",
    ),
    "legacy_phase578_retrieval_closure": (
        ROOT / "tests/glm5/phase578_retrieval_closure.py",
        "9bfc7ee816ddee7443bbc7613de38e1268ab1f902ec98093aa595dbc0a910494",
    ),
}

INITIAL_FILES = {MANIFEST_NAME, PROTOCOL_NAME, SELF_TEST_NAME, STAGE_COMMIT_NAME}
FINAL_FILES = INITIAL_FILES | {AUDIT_NAME, FREEZE_NAME}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path, root: Path = ROOT) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"regular non-symlink file required: {path}")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root.resolve(strict=True))
        label = str(relative).replace("\\", "/")
    except ValueError:
        label = str(resolved).replace("\\", "/")
    stat = path.stat()
    return {
        "path": label,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
        "is_symlink": False,
        "hardlink_count": stat.st_nlink,
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_raw(path: Path) -> list[tuple[bytes, dict[str, Any]]]:
    output = []
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.rstrip(b"\r\n")
            if line:
                output.append((line, json.loads(line.decode("utf-8"))))
    return output


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (canonical_json(row) + "\n").encode("utf-8") for row in rows
    )


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or path.exists():
        raise RuntimeError(f"no-overwrite publication refused: {path}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def verify_upstream() -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for name, (path, expected_hash) in UPSTREAM_EXPECTED.items():
        observed = identity(path)
        if observed["sha256"] != expected_hash:
            raise RuntimeError(f"upstream identity drift: {name}")
        reports[name] = observed
    final = read_json(UPSTREAM_EXPECTED["phase577_final_freeze"][0])
    if not all((
        final.get("freeze_complete") is True,
        final.get("gpu_behavior_authorized") is False,
        final.get("gpu_behavior_run_count") == 0,
        final.get("internal_trace_run_count") == 0,
        final.get("next_required_stage")
        == "freeze_separate_gpu_runner_and_executable_scorer",
    )):
        raise RuntimeError("Phase577 final boundary drift")
    qualification = read_json(
        UPSTREAM_EXPECTED["phase576r2_engineering_qualification"][0]
    )
    receipt = read_json(UPSTREAM_EXPECTED["phase576r2_engineering_receipt"][0])
    cleanup = read_json(UPSTREAM_EXPECTED["phase576r2_cleanup_qualification"][0])
    if not all((
        qualification.get("passed") is True,
        qualification.get("models_in_execution_order") == list(MODEL_ORDER),
        qualification.get("activation_persisted") is False,
        receipt.get("execution_passed") is True,
        receipt.get("attempted_models_in_order") == list(MODEL_ORDER),
        receipt.get("final_cuda_cleanup_pass") is True,
        cleanup.get("passed") is True,
        cleanup.get("allocated_after_strict_cleanup") == 0,
        cleanup.get("reserved_after_strict_cleanup") == 0,
    )):
        raise RuntimeError("Phase576R2 engineering evidence drift")
    return reports


def verify_formal_interpreter() -> dict[str, Any]:
    observed = identity(FORMAL_PYTHON, FORMAL_PYTHON.parent)
    if observed["sha256"] != FORMAL_PYTHON_SHA256:
        raise RuntimeError("formal Python executable drift")
    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "bitsandbytes", "accelerate")
    }
    expected = {
        "torch": "2.11.0+cu128",
        "transformers": "5.12.0",
        "bitsandbytes": "0.49.2",
        "accelerate": "1.14.0",
    }
    if packages != expected:
        raise RuntimeError(f"formal package environment drift: {packages}")
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(
            "Phase578 freeze must run under the frozen CUDA-capable interpreter"
        )
    return {
        "python_executable": str(FORMAL_PYTHON),
        "python_executable_identity": observed,
        "python_version": sys.version.split()[0],
        "packages": packages,
        "side_environment_not_authorized": str(ROOT / ".venv/Scripts/python.exe"),
    }


def source_identities() -> dict[str, Any]:
    reports = {}
    for relative in SOURCE_RELATIVES:
        reports[relative] = identity(ROOT / relative)
    return reports


def build_manifest() -> tuple[list[dict[str, Any]], str]:
    source = read_jsonl_raw(DEVELOPMENT_PATH)
    if len(source) != 336:
        raise RuntimeError("Phase577 development denominator drift")
    rows = []
    seen: set[str] = set()
    for ordinal, (raw, row) in enumerate(source):
        case_id = row.get("case_id")
        if (
            row.get("phase_id") != "Phase577"
            or row.get("split") != "development"
            or row.get("sealed") is not False
            or not isinstance(case_id, str)
            or case_id in seen
            or not isinstance(row.get("raw_prompt"), str)
        ):
            raise RuntimeError(f"development row invalid at ordinal {ordinal}")
        seen.add(case_id)
        rows.append({
            "schema_version": "phase578_development_prompt_manifest_row.v1",
            "phase_id": PHASE,
            "source_phase_id": "Phase577",
            "split": "development",
            "ordinal": ordinal,
            "case_id": case_id,
            "raw_prompt": row["raw_prompt"],
            "normalized_prompt_sha256": row["normalized_prompt_sha256"],
            "source_case_record_sha256": sha256_bytes(raw),
        })
    return rows, sha256_bytes(jsonl_bytes(rows))


def development_cases() -> list[dict[str, Any]]:
    return [row for _raw, row in read_jsonl_raw(DEVELOPMENT_PATH)]


def build_protocol(created: str) -> dict[str, Any]:
    upstream = verify_upstream()
    runtime = verify_formal_interpreter()
    sources = source_identities()
    manifest, manifest_hash = build_manifest()
    self_test = scorer.self_test(development_cases())
    phase577_protocol = read_json(UPSTREAM_EXPECTED["phase577_protocol"][0])
    phase576_qualification = read_json(
        UPSTREAM_EXPECTED["phase576r2_engineering_qualification"][0]
    )
    return {
        "schema_version": "phase578_preregistered_runner_protocol.v1",
        "phase_id": PHASE,
        "created_at_utc": created,
        "research_role": (
            "model-free bridge that freezes the raw development runner, "
            "executable scorer, analysis, and independent audit"
        ),
        "source_behavior_protocol_phase": "Phase577",
        "source_identities": sources,
        "upstream_identities": upstream,
        "formal_runtime_identity": runtime,
        "frozen_tokenizer_input_identities": phase577_protocol[
            "tokenizer_input_identities"
        ],
        "frozen_model_artifact_identities": phase576_qualification[
            "model_artifact_identities"
        ],
        "development_prompt_manifest": {
            "filename": MANIFEST_NAME,
            "row_count": len(manifest),
            "sha256": manifest_hash,
            "truth_fields_present": False,
            "fields": list(manifest[0]),
            "source_development_sha256": upstream["phase577_development"]["sha256"],
        },
        "models_in_required_order": list(MODEL_ORDER),
        "generation_contract": {
            "batch_size": 8,
            "repeats": list(scorer.REPEATS),
            "max_new_tokens": 24,
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "use_cache": True,
            "pad_token_id": "adapter.pad_token_id",
            "eos_token_id": "adapter.effective_eos_token_ids",
            "tokenizer_padding_side": "left",
            "tokenizer_padding": True,
            "tokenizer_truncation": False,
            "tokenizer_add_special_tokens": False,
            "decode_skip_special_tokens": False,
            "decode_clean_up_tokenization_spaces": False,
            "qwen3_enable_thinking": False,
            "deepseek_empty_think_prefill_closed": True,
            "output_scores": False,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict_in_generate": False,
            "quantization": "bitsandbytes_int8",
            "nonquantized_dtype": "torch.bfloat16",
            "attention_implementation": "sdpa",
            "cpu_or_disk_offload": False,
            "automatic_fallback": False,
        },
        "scoring_contract": {
            "source_schema": scorer.SOURCE_SCHEMA,
            "primary": "both repeats target semantic prefix completed within 8 generated tokens",
            "complete_output_resolved_before_token_completion_index": True,
            "decode_skip_special_tokens": False,
            "decode_clean_up_tokenization_spaces": False,
            "gate": scorer.GATE,
            "eos_budget_exact_and_full_identity_are_diagnostics_only": True,
            "micro_floor_is_part_of_total_gate": True,
            "micro_floor_integer_test": "100 * stable_cases >= 85 * 336",
            "statistical_independence_claimed": False,
        },
        "execution_sequence_after_this_freeze": [
            "separate_synthetic_engineering_qualification",
            "development_raw_generation_if_and_only_if_qualification_passes",
            "immutable_raw_publication",
            "separate_cpu_scoring",
            "independent_execution_audit",
        ],
        "split_access_policy": {
            "gpu_runner_reads_prompt_only_manifest": True,
            "gpu_runner_reads_full_development_cases": False,
            "confirmation_access_authorized": False,
            "heldout_access_authorized": False,
            "sealed_access_authorized": False,
            "production_behavior_scorer_full_development_access_only_after_raw_publication": True,
            "prefreeze_gate_self_test_reads_full_development_without_model_outputs": True,
            "prefreeze_protocol_and_audit_non_model_development_access": True,
        },
        "instrumentation_policy": {
            "observer_only": True,
            "activation_collection": False,
            "hidden_states": False,
            "attentions": False,
            "scores_or_logits": False,
            "hooks": False,
            "causal_intervention": False,
            "candidate_coordinates": [],
            "candidate_mechanism_formulas": [],
        },
        "legacy_phase578_collision": {
            "identity": upstream["legacy_phase578_retrieval_closure"],
            "status": "excluded_not_imported_not_executed",
            "model_utils_dependency_forbidden": True,
            "reason": (
                "legacy source presupposes a retrieval circuit and requests "
                "attention/head ablation before current natural evidence"
            ),
        },
        "phase576r2_role": (
            "historical loader/forward/cleanup feasibility evidence only; "
            "not Phase578 execution authorization"
        ),
        "scorer_self_test": {
            "passed": self_test["passed"],
            "test_count": self_test["test_count"],
            "payload_sha256": scorer.sha256_json(self_test),
        },
        "gpu_behavior_authorized_by_this_protocol": False,
        "model_weights_loaded": False,
        "gpu_used": False,
        "candidate_coordinates": [],
        "candidate_mechanism_formulas": [],
        "scientific_limits": [
            "Phase578 freezes an executable observation contract; it discovers no internal structure",
            "Phase576R2 qualification cannot substitute for the new runner qualification",
            "prompt-only projection reduces label leakage but does not create third-party blinding",
            "Python path guards cannot prove the absence of preopened descriptors or every native/reparse alias",
            "the thresholds remain preregistered engineering admission rules rather than language laws",
        ],
    }


def self_test() -> dict[str, Any]:
    upstream = verify_upstream()
    runtime = verify_formal_interpreter()
    sources = source_identities()
    manifest, manifest_hash = build_manifest()
    score_test = scorer.self_test(development_cases())
    checks = {
        "upstream_count": len(upstream) == len(UPSTREAM_EXPECTED),
        "source_count": len(sources) == len(SOURCE_RELATIVES),
        "formal_interpreter": runtime["python_executable_identity"]["sha256"]
        == FORMAL_PYTHON_SHA256,
        "manifest_rows": len(manifest) == 336,
        "manifest_unique_cases": len({row["case_id"] for row in manifest}) == 336,
        "manifest_truth_free": not any(
            set(row) & {
                "target", "foil", "candidate_groups", "focus_object_class",
                "comparison_object_class", "target_truth_polarity",
            }
            for row in manifest
        ),
        "manifest_hash": len(manifest_hash) == 64,
        "scorer_self_test": score_test["passed"] is True
        and score_test.get("gate_self_test", {}).get("passed") is True,
        "torch_not_imported": "torch" not in sys.modules,
        "transformers_not_imported": "transformers" not in sys.modules,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase578 protocol self-test failed: {checks}")
    return {
        "schema_version": "phase578_protocol_self_test.v1",
        "phase_id": PHASE, "passed": True, "checks": checks,
        "gpu_used": False, "model_weights_loaded": False,
        "files_written": False,
    }


def write_stage() -> dict[str, Any]:
    if OUT_DIR.exists():
        raise RuntimeError("Phase578 protocol output already exists")
    created = now()
    protocol = build_protocol(created)
    manifest, manifest_hash = build_manifest()
    score_test = scorer.self_test(development_cases())
    pending = OUT_DIR.with_name(
        f".{OUT_DIR.name}.pending-{os.getpid()}-{uuid.uuid4().hex}"
    )
    pending.mkdir(parents=True, exist_ok=False)
    try:
        write_exclusive(pending / MANIFEST_NAME, jsonl_bytes(manifest))
        write_exclusive(pending / PROTOCOL_NAME, json_bytes(protocol))
        write_exclusive(pending / SELF_TEST_NAME, json_bytes(score_test))
        initial = {
            name: identity(pending / name, pending) for name in (
                MANIFEST_NAME, PROTOCOL_NAME, SELF_TEST_NAME
            )
        }
        stage_commit = {
            "schema_version": "phase578_stage_commit.v1",
            "phase_id": PHASE, "created_at_utc": created,
            "stage_complete": True,
            "artifact_identities": initial,
            "development_manifest_sha256": manifest_hash,
            "source_identities": protocol["source_identities"],
            "gpu_used": False, "model_weights_loaded": False,
            "gpu_behavior_authorized": False,
        }
        write_exclusive(pending / STAGE_COMMIT_NAME, json_bytes(stage_commit))
        pending.rename(OUT_DIR)
    except BaseException:
        if pending.exists():
            if pending.parent.resolve(strict=True) != OUT_DIR.parent.resolve(strict=True):
                raise RuntimeError("protocol pending quarantine escaped result root")
            pending.rename(pending.with_name(
                f".{OUT_DIR.name}.failed-{uuid.uuid4().hex}"
            ))
        raise
    return verify_stage(require_final=False)


def _exact_files(expected: set[str]) -> bool:
    if not OUT_DIR.is_dir() or OUT_DIR.is_symlink():
        return False
    files = {
        str(path.relative_to(OUT_DIR)).replace("\\", "/")
        for path in OUT_DIR.rglob("*") if path.is_file()
    }
    directories = [path for path in OUT_DIR.rglob("*") if path.is_dir()]
    return files == expected and not directories and not any(
        path.is_symlink() for path in OUT_DIR.rglob("*")
    )


def independent_audit_verification() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.run(
        [
            str(FORMAL_PYTHON),
            str(ROOT / "tests/glm5/phase578_gpt5_runner_audit.py"),
            "--verify-freeze-audit",
        ],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        env=environment, check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "Phase578 independent audit verifier failed: " + process.stderr
        )
    payload = json.loads(process.stdout)
    if payload.get("passed") is not True:
        raise RuntimeError("Phase578 independent audit verifier did not pass")
    return payload


def verify_stage(require_final: bool, allow_audit: bool = False) -> dict[str, Any]:
    expected = (
        FINAL_FILES if require_final
        else INITIAL_FILES | ({AUDIT_NAME} if allow_audit else set())
    )
    if not _exact_files(expected):
        raise RuntimeError(f"Phase578 exact artifact closure failed: {expected}")
    protocol = read_json(OUT_DIR / PROTOCOL_NAME)
    stage = read_json(OUT_DIR / STAGE_COMMIT_NAME)
    manifest, manifest_hash = build_manifest()
    checks = {
        "protocol_schema": protocol.get("schema_version")
        == "phase578_preregistered_runner_protocol.v1",
        "phase": protocol.get("phase_id") == PHASE,
        "sources": protocol.get("source_identities") == source_identities(),
        "upstream": protocol.get("upstream_identities") == verify_upstream(),
        "formal_runtime": protocol.get("formal_runtime_identity")
        == verify_formal_interpreter(),
        "manifest_bytes": (OUT_DIR / MANIFEST_NAME).read_bytes()
        == jsonl_bytes(manifest),
        "manifest_hash": protocol["development_prompt_manifest"]["sha256"]
        == manifest_hash,
        "scorer_report": read_json(OUT_DIR / SELF_TEST_NAME)
        == scorer.self_test(development_cases()),
        "stage_schema": stage.get("schema_version") == "phase578_stage_commit.v1",
        "stage_complete": stage.get("stage_complete") is True,
        "stage_artifact_identities": stage.get("artifact_identities") == {
            name: identity(OUT_DIR / name, OUT_DIR)
            for name in (MANIFEST_NAME, PROTOCOL_NAME, SELF_TEST_NAME)
        },
        "stage_manifest_hash": stage.get("development_manifest_sha256")
        == sha256_file(OUT_DIR / MANIFEST_NAME),
        "stage_sources": stage.get("source_identities") == source_identities(),
        "stage_gpu_false": stage.get("gpu_used") is False
        and stage.get("gpu_behavior_authorized") is False,
        "no_model_modules": "torch" not in sys.modules
        and "transformers" not in sys.modules,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase578 stage verification failed: {checks}")
    if require_final:
        audit = read_json(OUT_DIR / AUDIT_NAME)
        freeze = read_json(OUT_DIR / FREEZE_NAME)
        audit_verification = independent_audit_verification()
        final_checks = {
            "audit_passed": audit.get("passed") is True
            and all(audit.get("checks", {}).values()),
            "freeze_schema": freeze.get("schema_version")
            == "phase578_freeze_commit.v1",
            "freeze_complete": freeze.get("freeze_complete") is True,
            "freeze_protocol_hash": freeze.get("protocol_sha256")
            == sha256_file(OUT_DIR / PROTOCOL_NAME),
            "freeze_stage_hash": freeze.get("stage_commit_sha256")
            == sha256_file(OUT_DIR / STAGE_COMMIT_NAME),
            "freeze_audit_hash": freeze.get("independent_audit_sha256")
            == sha256_file(OUT_DIR / AUDIT_NAME),
            "audit_verification_payload": freeze.get(
                "independent_audit_verification_payload_sha256"
            ) == sha256_bytes(canonical_json(audit_verification).encode("utf-8")),
            "no_gpu_authority": freeze.get("gpu_behavior_authorized") is False,
            "no_runs": freeze.get("gpu_behavior_run_count") == 0
            and freeze.get("engineering_qualification_run_count") == 0,
            "next_stage": freeze.get("next_required_stage")
            == "phase578_separate_engineering_qualification",
            "freeze_sources": freeze.get("source_identities") == source_identities(),
            "freeze_model_order": freeze.get("models_in_required_future_order")
            == list(MODEL_ORDER),
            "no_future_or_internal_authority": all(
                freeze.get(name) is False for name in (
                    "confirmation_authorized", "heldout_authorized",
                    "sealed_authorized", "internal_trace_authorized",
                )
            ),
            "no_candidates": freeze.get("candidate_coordinates") == []
            and freeze.get("candidate_mechanism_formulas") == [],
        }
        checks.update(final_checks)
        if not all(final_checks.values()):
            raise RuntimeError(f"Phase578 final verification failed: {final_checks}")
    return {
        "schema_version": "phase578_protocol_verification.v1",
        "phase_id": PHASE, "passed": True, "checks": checks,
        "gpu_used": False, "model_weights_loaded": False,
        "files_written": False,
    }


def finalize() -> dict[str, Any]:
    verify_stage(require_final=False, allow_audit=True)
    audit_path = OUT_DIR / AUDIT_NAME
    if not audit_path.is_file():
        raise RuntimeError("Phase578 independent audit is required before finalize")
    audit = read_json(audit_path)
    if audit.get("passed") is not True or not all(audit.get("checks", {}).values()):
        raise RuntimeError("Phase578 independent audit did not pass")
    audit_verification = independent_audit_verification()
    if (OUT_DIR / FREEZE_NAME).exists():
        raise RuntimeError("Phase578 final freeze already exists")
    freeze = {
        "schema_version": "phase578_freeze_commit.v1",
        "phase_id": PHASE, "created_at_utc": now(),
        "freeze_complete": True,
        "protocol_sha256": sha256_file(OUT_DIR / PROTOCOL_NAME),
        "stage_commit_sha256": sha256_file(OUT_DIR / STAGE_COMMIT_NAME),
        "development_manifest_sha256": sha256_file(OUT_DIR / MANIFEST_NAME),
        "scorer_self_test_sha256": sha256_file(OUT_DIR / SELF_TEST_NAME),
        "independent_audit_sha256": sha256_file(audit_path),
        "independent_audit_verification_payload_sha256": sha256_bytes(
            canonical_json(audit_verification).encode("utf-8")
        ),
        "source_identities": source_identities(),
        "models_in_required_future_order": list(MODEL_ORDER),
        "engineering_qualification_run_count": 0,
        "gpu_behavior_run_count": 0,
        "gpu_behavior_authorized": False,
        "model_weights_loaded": False,
        "gpu_used": False,
        "confirmation_authorized": False,
        "heldout_authorized": False,
        "sealed_authorized": False,
        "internal_trace_authorized": False,
        "candidate_coordinates": [],
        "candidate_mechanism_formulas": [],
        "next_required_stage": "phase578_separate_engineering_qualification",
    }
    write_exclusive(OUT_DIR / FREEZE_NAME, json_bytes(freeze))
    return verify_stage(require_final=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify-stage", action="store_true")
    group.add_argument("--finalize", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.write:
        result = write_stage()
    elif args.verify_stage:
        result = verify_stage(require_final=False)
    elif args.finalize:
        result = finalize()
    else:
        result = verify_stage(require_final=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
