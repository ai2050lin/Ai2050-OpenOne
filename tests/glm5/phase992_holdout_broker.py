#!/usr/bin/env python3
"""Append-only Phase992 broker for the Phase991 sealed prompt manifest.

The formal model runner never names or opens the sealed source.  This broker
first publishes an exclusive hash-chain event, then makes one temporary,
hash-verified prompt-only copy.  It never reads scoring truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE991_ROOT = ROOT / "tests/glm5/result/phase991_delayed_binding_gpu_admission"
HOLDOUT_COMMITMENT = PHASE991_ROOT / "holdout_access_commitment.json"
# The private source is intentionally confined to this broker.  The runner's
# source is statically rejected if it contains this relative path.
SEALED_PROMPT_SOURCE = (
    PHASE991_ROOT / "runtime_prompts/private/sealed_holdout.jsonl"
)
DEFAULT_ACTIVATION = (
    ROOT / "tests/glm5/result/phase992_delayed_binding_behavior_protocol/activation.json"
)
BROKER_SOURCE = Path(__file__).resolve()
MODELS = ("qwen3", "glm4", "deepseek7b")
FORMAL_PYTHON = Path(
    r"C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe"
)
FORMAL_PYTHON_SHA256 = (
    "0f11fb7422fa347b7609ba0964ceccef3c8fa9f15230c37b9ec27668e68e8a8a"
)
EXPECTED_COMMITMENT_FILE_SHA256 = (
    "91393e63e0a8b4f7d4cacf5443a53de7bf4f31773cb763f43d57dcb5b8fde8a1"
)
EXPECTED_MANIFEST_SHA256 = (
    "297ba2e58697ccad1b43ed0f915828e5edbba07d7c71251667cc8e737462e1e5"
)
EXPECTED_MANIFEST_BYTES = 2_983_616


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing/plain JSON required: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def self_hash(document: dict[str, Any], field: str) -> str:
    payload = dict(document)
    payload.pop(field, None)
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def json_bytes(document: dict[str, Any]) -> bytes:
    return (json.dumps(
        document, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode("utf-8")


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _seal_for_path(activation: dict[str, Any], target: Path) -> dict[str, Any]:
    seals = activation.get("source_seals")
    require(isinstance(seals, dict), "activation source_seals missing")
    wanted = target.resolve(strict=True)
    matches: list[dict[str, Any]] = []
    for item in seals.values():
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        candidate = Path(item["path"])
        candidate = candidate if candidate.is_absolute() else ROOT / candidate
        if candidate.resolve(strict=True) == wanted:
            matches.append(item)
    require(len(matches) == 1, f"activation has no unique source seal for {target.name}")
    return matches[0]


def verify_activation(path: Path) -> dict[str, Any]:
    require(path.resolve(strict=True) == DEFAULT_ACTIVATION.resolve(strict=True),
            "activation must be the frozen default artifact")
    document = read_json(path)
    require(document.get("schema_version") == "phase992_gpu_behavior_activation.v1",
            "activation schema drift")
    require(document.get("phase") == 992, "activation phase drift")
    require(document.get("experiment") == "delayed_two_hop_gpu_behavior",
            "activation experiment drift")
    require(document.get("gpu_behavior_execution_authorized") is True,
            "GPU behavior execution is not authorized")
    for forbidden in (
        "internal_trace_authorized", "hidden_states_authorized",
        "attentions_authorized", "causal_intervention_authorized",
        "scoring_authorized",
    ):
        require(document.get(forbidden) is False,
                f"activation must keep {forbidden}=false")
    require(document.get("model_order") == list(MODELS), "activation model order drift")
    identity = document.get("formal_python")
    require(isinstance(identity, dict)
            and Path(identity.get("path", "")).resolve(strict=True)
            == FORMAL_PYTHON.resolve(strict=True)
            and identity.get("sha256") == FORMAL_PYTHON_SHA256
            and sha256_file(FORMAL_PYTHON) == FORMAL_PYTHON_SHA256,
            "formal Python identity drift")
    observed = document.get("activation_sha256")
    require(isinstance(observed, str) and observed == self_hash(document, "activation_sha256"),
            "activation self-hash mismatch")
    seal = _seal_for_path(document, BROKER_SOURCE)
    require(seal.get("sha256") == sha256_file(BROKER_SOURCE)
            and seal.get("bytes") == BROKER_SOURCE.stat().st_size,
            "broker source is not activation-sealed")
    return document


def verify_public_admission(
    path: Path, activation: dict[str, Any], run_id: str,
) -> dict[str, Any]:
    document = read_json(path)
    require(document.get("schema_version") == "phase992_public_behavior_admission.v1",
            "public behavior admission schema drift")
    require(document.get("phase") == 992 and document.get("run_id") == run_id,
            "public behavior admission scope drift")
    require(document.get("sealed_holdout_model_access_authorized") is True,
            "public behavior gate did not authorize holdout access")
    require(document.get("all_models_public_pass") is True,
            "all public model gates must pass before holdout access")
    require(document.get("model_order") == list(MODELS),
            "public behavior admission model order drift")
    require(document.get("activation_sha256") == activation["activation_sha256"],
            "public admission activation binding drift")
    require(document.get("admission_sha256") == self_hash(document, "admission_sha256"),
            "public behavior admission self-hash mismatch")
    return document


def verify_phase991_commitment() -> dict[str, Any]:
    require(sha256_file(HOLDOUT_COMMITMENT) == EXPECTED_COMMITMENT_FILE_SHA256,
            "Phase991 holdout commitment file drift")
    document = read_json(HOLDOUT_COMMITMENT)
    source = document.get("sealed_prompt_manifest")
    require(isinstance(source, dict)
            and source.get("sha256") == EXPECTED_MANIFEST_SHA256
            and source.get("bytes") == EXPECTED_MANIFEST_BYTES,
            "Phase991 sealed manifest commitment drift")
    require(document.get("first_model_evaluation_access_status") == "not_accessed",
            "Phase991 frozen commitment did not start unopened")
    contract = document.get("future_access_marker_contract")
    require(isinstance(contract, dict)
            and contract.get("create_before_open") is True
            and contract.get("create_exclusive_no_overwrite") is True
            and contract.get("per_model_markers") == list(MODELS),
            "Phase991 holdout access contract drift")
    return document


def _event_head(event: dict[str, Any]) -> str:
    payload = dict(event)
    payload.pop("new_head", None)
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def _verify_chain(events_dir: Path, genesis: str, run_id: str) -> tuple[int, str, list[dict[str, Any]]]:
    paths = sorted(events_dir.glob("*.json")) if events_dir.exists() else []
    head = genesis
    events: list[dict[str, Any]] = []
    for ordinal, path in enumerate(paths):
        require(path.name.startswith(f"{ordinal:04d}_"), "holdout event ordinal gap")
        event = read_json(path)
        require(event.get("schema_version") == "phase992_holdout_access_event.v1"
                and event.get("ordinal") == ordinal
                and event.get("run_id") == run_id
                and event.get("previous_head") == head
                and event.get("new_head") == _event_head(event),
                "holdout access hash chain invalid")
        head = event["new_head"]
        events.append(event)
    return len(events), head, events


def _acquire_lease(run_root: Path) -> Path:
    lease = run_root / "holdout_access/.broker.lock"
    lease.parent.mkdir(parents=True, exist_ok=True)
    write_exclusive(lease, canonical_json({"pid": os.getpid(), "created_at_utc": now()}).encode())
    return lease


def _release_lease(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _model_state(events: list[dict[str, Any]]) -> dict[str, str]:
    state = {model: "unopened" for model in MODELS}
    for event in events:
        model, action = event.get("model"), event.get("action")
        require(model in MODELS, "event model outside registry")
        if action == "grant":
            require(state[model] == "unopened", "duplicate/out-of-order holdout grant")
            state[model] = "granted"
        elif action == "seal_and_revoke":
            require(state[model] == "granted", "holdout seal without grant")
            state[model] = "sealed"
        elif action == "abort_and_revoke":
            require(state[model] == "granted", "holdout abort without grant")
            state[model] = "aborted"
        else:
            raise RuntimeError("unknown holdout event action")
    return state


def _write_event(
    events_dir: Path, ordinal: int, previous_head: str, run_id: str,
    model: str, action: str, input_sha: str, output_sha: str | None,
) -> tuple[Path, dict[str, Any]]:
    event = {
        "schema_version": "phase992_holdout_access_event.v1",
        "phase": 992,
        "ordinal": ordinal,
        "previous_head": previous_head,
        "run_id": run_id,
        "model": model,
        "model_order_index": MODELS.index(model),
        "action": action,
        "timestamp": now(),
        "input_manifest_sha256": input_sha,
        "output_receipt_sha256": output_sha,
    }
    event["new_head"] = _event_head(event)
    path = events_dir / f"{ordinal:04d}_{model}_{action}.json"
    write_exclusive(path, json_bytes(event))
    return path, event


def _validate_destination(run_root: Path, destination: Path) -> Path:
    root = (run_root / "temporary_holdout").resolve(strict=True)
    resolved = destination.resolve(strict=False)
    require(resolved.parent == root and resolved.suffix == ".jsonl",
            "temporary holdout destination escaped broker namespace")
    require(not resolved.exists(), "temporary holdout copy already exists")
    return resolved


def grant(
    activation_path: Path, admission_path: Path, run_root: Path, run_id: str,
    model: str, destination: Path, runner_sha256: str,
) -> dict[str, Any]:
    require(model in MODELS and run_id.strip(), "invalid broker grant scope")
    run_root = run_root.resolve(strict=True)
    activation = verify_activation(activation_path)
    verify_public_admission(admission_path, activation, run_id)
    commitment = verify_phase991_commitment()
    runner_seals = activation.get("source_seals", {})
    require(any(isinstance(value, dict)
                and value.get("sha256") == runner_sha256
                and str(value.get("path", "")).replace("\\", "/").endswith(
                    "/phase992_delayed_binding_runner.py")
                for value in runner_seals.values()),
            "runner hash is not activation-sealed")
    destination = _validate_destination(run_root, destination)
    lease = _acquire_lease(run_root)
    try:
        genesis = commitment["first_access_log_genesis_sha256"]
        events_dir = run_root / "holdout_access/events"
        ordinal, head, events = _verify_chain(events_dir, genesis, run_id)
        state = _model_state(events)
        expected_index = next((i for i, key in enumerate(MODELS)
                               if state[key] != "sealed"), len(MODELS))
        require(MODELS.index(model) == expected_index and state[model] == "unopened",
                "holdout model grant violated strict order")
        # This durable exclusive event is committed before the first source open.
        event_path, event = _write_event(
            events_dir, ordinal, head, run_id, model, "grant",
            EXPECTED_MANIFEST_SHA256, None,
        )
        try:
            require(SEALED_PROMPT_SOURCE.is_file() and not SEALED_PROMPT_SOURCE.is_symlink()
                    and SEALED_PROMPT_SOURCE.stat().st_size == EXPECTED_MANIFEST_BYTES
                    and sha256_file(SEALED_PROMPT_SOURCE) == EXPECTED_MANIFEST_SHA256,
                    "sealed prompt source drift after grant event")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with SEALED_PROMPT_SOURCE.open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            require(destination.stat().st_size == EXPECTED_MANIFEST_BYTES
                    and sha256_file(destination) == EXPECTED_MANIFEST_SHA256,
                    "temporary holdout copy identity drift")
        except BaseException:
            if destination.exists():
                destination.unlink()
            abort_path, abort_event = _write_event(
                events_dir, ordinal + 1, event["new_head"], run_id, model,
                "abort_and_revoke", EXPECTED_MANIFEST_SHA256, None,
            )
            failure = {
                "schema_version": "phase992_holdout_grant_failure_receipt.v1",
                "phase": 992, "run_id": run_id, "model": model,
                "model_order_index": MODELS.index(model), "status": "aborted",
                "created_at_utc": now(), "reason": "grant_copy_failure",
                "temporary_copy_revoked": not destination.exists(),
                "grant_event_sha256": sha256_file(event_path),
                "abort_event": {
                    "path": str(abort_path.relative_to(run_root)).replace("\\", "/"),
                    "sha256": sha256_file(abort_path), "new_head": abort_event["new_head"],
                },
                "scientific_status": "inconclusive_fail_closed",
            }
            failure["receipt_sha256"] = self_hash(failure, "receipt_sha256")
            write_exclusive(
                run_root / f"holdout_access/grant_failure_{MODELS.index(model):02d}_{model}.json",
                json_bytes(failure),
            )
            raise
        receipt = {
            "schema_version": "phase992_holdout_grant_receipt.v1",
            "phase": 992, "run_id": run_id, "model": model,
            "model_order_index": MODELS.index(model), "status": "granted",
            "created_at_utc": now(),
            "activation_sha256": activation["activation_sha256"],
            "public_admission_sha256": admission_path_sha256(admission_path),
            "runner_source_sha256": runner_sha256,
            "access_event": {"path": str(event_path.relative_to(run_root)).replace("\\", "/"),
                             "sha256": sha256_file(event_path), "new_head": event["new_head"]},
            "temporary_copy": {"path": str(destination), "bytes": destination.stat().st_size,
                               "sha256": sha256_file(destination)},
            "truth_read": False,
        }
        receipt["receipt_sha256"] = self_hash(receipt, "receipt_sha256")
        receipt_path = run_root / f"holdout_access/grant_{MODELS.index(model):02d}_{model}.json"
        write_exclusive(receipt_path, json_bytes(receipt))
        return {**receipt, "receipt_path": str(receipt_path)}
    finally:
        _release_lease(lease)


def admission_path_sha256(path: Path) -> str:
    return sha256_file(path)


def finalize(
    activation_path: Path, admission_path: Path, run_root: Path, run_id: str,
    model: str, grant_receipt_path: Path, output_receipt_path: Path,
    cleanup_receipt_path: Path,
) -> dict[str, Any]:
    require(model in MODELS and run_id.strip(), "invalid broker finalize scope")
    run_root = run_root.resolve(strict=True)
    activation = verify_activation(activation_path)
    verify_public_admission(admission_path, activation, run_id)
    commitment = verify_phase991_commitment()
    grant_receipt = read_json(grant_receipt_path)
    require(grant_receipt.get("receipt_sha256") == self_hash(grant_receipt, "receipt_sha256")
            and grant_receipt.get("run_id") == run_id
            and grant_receipt.get("model") == model
            and grant_receipt.get("status") == "granted",
            "grant receipt drift")
    output = read_json(output_receipt_path)
    cleanup = read_json(cleanup_receipt_path)
    require(output.get("receipt_sha256") == self_hash(output, "receipt_sha256")
            and output.get("scope") == "holdout" and output.get("model") == model
            and output.get("status") == "sealed"
            and output.get("execution_status") == "success",
            "holdout output receipt is not sealed")
    require(cleanup.get("receipt_sha256") == self_hash(cleanup, "receipt_sha256")
            and cleanup.get("scope") == "holdout" and cleanup.get("model") == model
            and cleanup.get("status") == "sealed"
            and cleanup.get("cleanup_pass") is True
            and cleanup.get("baseline_recovered") is True,
            "holdout cleanup receipt is not sealed")
    output_sha = sha256_file(output_receipt_path)
    temporary = Path(grant_receipt["temporary_copy"]["path"])
    require(temporary.is_file()
            and sha256_file(temporary) == EXPECTED_MANIFEST_SHA256,
            "temporary holdout copy missing before revoke")
    lease = _acquire_lease(run_root)
    try:
        genesis = commitment["first_access_log_genesis_sha256"]
        events_dir = run_root / "holdout_access/events"
        ordinal, head, events = _verify_chain(events_dir, genesis, run_id)
        state = _model_state(events)
        require(state[model] == "granted", "holdout model is not in granted state")
        temporary.unlink()
        require(not temporary.exists(), "temporary holdout copy revoke failed")
        event_path, event = _write_event(
            events_dir, ordinal, head, run_id, model, "seal_and_revoke",
            EXPECTED_MANIFEST_SHA256, output_sha,
        )
        receipt = {
            "schema_version": "phase992_holdout_seal_receipt.v1",
            "phase": 992, "run_id": run_id, "model": model,
            "model_order_index": MODELS.index(model), "status": "sealed",
            "created_at_utc": now(),
            "output_receipt_sha256": output_sha,
            "cleanup_receipt_sha256": sha256_file(cleanup_receipt_path),
            "temporary_copy_revoked": True,
            "access_event": {"path": str(event_path.relative_to(run_root)).replace("\\", "/"),
                             "sha256": sha256_file(event_path), "new_head": event["new_head"]},
        }
        receipt["receipt_sha256"] = self_hash(receipt, "receipt_sha256")
        receipt_path = run_root / f"holdout_access/seal_{MODELS.index(model):02d}_{model}.json"
        write_exclusive(receipt_path, json_bytes(receipt))
        return {**receipt, "receipt_path": str(receipt_path)}
    finally:
        _release_lease(lease)


def abort(
    activation_path: Path, admission_path: Path, run_root: Path, run_id: str,
    model: str, grant_receipt_path: Path, reason: str,
) -> dict[str, Any]:
    """Fail closed and revoke a granted temporary copy after any run failure."""
    require(model in MODELS and run_id.strip(), "invalid broker abort scope")
    require(reason in {"grant_copy_failure", "worker_failure", "cleanup_failure",
                       "finalize_failure", "parent_interruption"},
            "unregistered broker abort reason")
    run_root = run_root.resolve(strict=True)
    activation = verify_activation(activation_path)
    verify_public_admission(admission_path, activation, run_id)
    commitment = verify_phase991_commitment()
    grant_receipt = read_json(grant_receipt_path)
    require(grant_receipt.get("receipt_sha256") == self_hash(grant_receipt, "receipt_sha256")
            and grant_receipt.get("run_id") == run_id
            and grant_receipt.get("model") == model
            and grant_receipt.get("status") == "granted",
            "abort grant receipt drift")
    temporary = Path(str(grant_receipt["temporary_copy"]["path"]))
    lease = _acquire_lease(run_root)
    try:
        genesis = commitment["first_access_log_genesis_sha256"]
        events_dir = run_root / "holdout_access/events"
        ordinal, head, events = _verify_chain(events_dir, genesis, run_id)
        require(_model_state(events)[model] == "granted",
                "only a granted holdout copy can be aborted")
        if temporary.exists():
            require(temporary.is_file() and not temporary.is_symlink()
                    and sha256_file(temporary) == EXPECTED_MANIFEST_SHA256,
                    "abort temporary copy identity drift")
            temporary.unlink()
        require(not temporary.exists(), "abort failed to revoke temporary copy")
        event_path, event = _write_event(
            events_dir, ordinal, head, run_id, model, "abort_and_revoke",
            EXPECTED_MANIFEST_SHA256, None,
        )
        receipt = {
            "schema_version": "phase992_holdout_abort_receipt.v1",
            "phase": 992, "run_id": run_id, "model": model,
            "model_order_index": MODELS.index(model), "status": "aborted",
            "created_at_utc": now(), "reason": reason,
            "activation_sha256": activation["activation_sha256"],
            "grant_receipt_sha256": grant_receipt["receipt_sha256"],
            "temporary_copy_revoked": True,
            "access_event": {
                "path": str(event_path.relative_to(run_root)).replace("\\", "/"),
                "sha256": sha256_file(event_path), "new_head": event["new_head"],
            },
            "scientific_status": "inconclusive_fail_closed",
        }
        receipt["receipt_sha256"] = self_hash(receipt, "receipt_sha256")
        receipt_path = run_root / f"holdout_access/abort_{MODELS.index(model):02d}_{model}.json"
        write_exclusive(receipt_path, json_bytes(receipt))
        return {**receipt, "receipt_path": str(receipt_path)}
    finally:
        _release_lease(lease)


def publish_final_chain(run_root: Path, run_id: str) -> dict[str, Any]:
    run_root = run_root.resolve(strict=True)
    commitment = verify_phase991_commitment()
    lease = _acquire_lease(run_root)
    try:
        count, head, events = _verify_chain(
            run_root / "holdout_access/events",
            commitment["first_access_log_genesis_sha256"], run_id,
        )
        require(count == 2 * len(MODELS)
                and _model_state(events) == {model: "sealed" for model in MODELS},
                "holdout access chain is incomplete")
        document = {
            "schema_version": "phase992_holdout_access_chain_receipt.v1",
            "phase": 992, "run_id": run_id, "status": "complete",
            "model_order": list(MODELS), "event_count": count,
            "genesis_head": commitment["first_access_log_genesis_sha256"],
            "final_head": head, "created_at_utc": now(),
            "all_temporary_copies_revoked": not any(
                (run_root / "temporary_holdout").glob("*.jsonl")
            ),
        }
        require(document["all_temporary_copies_revoked"],
                "temporary holdout copies remain after chain completion")
        document["receipt_sha256"] = self_hash(document, "receipt_sha256")
        path = run_root / "holdout_access/final_chain_receipt.json"
        write_exclusive(path, json_bytes(document))
        return {**document, "receipt_path": str(path)}
    finally:
        _release_lease(lease)


def self_test() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    genesis = sha256_bytes(b"phase992-broker-self-test")
    with tempfile.TemporaryDirectory(prefix="phase992_broker_selftest_") as raw:
        root = Path(raw)
        events = root / "events"
        path0, event0 = _write_event(
            events, 0, genesis, "self-test", "qwen3", "grant",
            "a" * 64, None,
        )
        count, head, rows = _verify_chain(events, genesis, "self-test")
        checks["exclusive_event_and_chain"] = (
            path0.is_file() and count == 1 and head == event0["new_head"]
            and _model_state(rows)["qwen3"] == "granted"
        )
        path1, _ = _write_event(
            events, 1, head, "self-test", "qwen3", "seal_and_revoke",
            "a" * 64, "b" * 64,
        )
        count, _, rows = _verify_chain(events, genesis, "self-test")
        checks["grant_then_seal_state_machine"] = (
            path1.is_file() and count == 2 and _model_state(rows)["qwen3"] == "sealed"
        )
        _, event2 = _write_event(
            events, 2, _event_head(rows[-1]), "self-test", "glm4", "grant",
            "a" * 64, None,
        )
        _, _ = _write_event(
            events, 3, event2["new_head"], "self-test", "glm4", "abort_and_revoke",
            "a" * 64, None,
        )
        _, _, rows = _verify_chain(events, genesis, "self-test")
        checks["abort_is_terminal_fail_closed"] = _model_state(rows)["glm4"] == "aborted"
        try:
            write_exclusive(path0, b"tamper")
        except FileExistsError:
            checks["no_overwrite"] = True
        else:
            checks["no_overwrite"] = False
        tampered = read_json(path1)
        tampered["previous_head"] = "0" * 64
        path1.write_bytes(json_bytes(tampered))
        try:
            _verify_chain(events, genesis, "self-test")
        except RuntimeError:
            checks["chain_tamper_rejected"] = True
        else:
            checks["chain_tamper_rejected"] = False
    checks["runner_source_path_not_shared"] = SEALED_PROMPT_SOURCE.name not in (
        ROOT / "tests/glm5/phase992_delayed_binding_runner.py"
    ).read_text(encoding="utf-8") if (
        ROOT / "tests/glm5/phase992_delayed_binding_runner.py"
    ).is_file() else True
    require(all(checks.values()), f"broker self-test failed: {checks}")
    return {"schema_version": "phase992_holdout_broker_self_test.v1",
            "passed": True, "cuda_used": False, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--grant", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--abort", action="store_true")
    mode.add_argument("--finalize-chain", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--activation", type=Path, default=DEFAULT_ACTIVATION)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--runner-sha256")
    parser.add_argument("--grant-receipt", type=Path)
    parser.add_argument("--output-receipt", type=Path)
    parser.add_argument("--cleanup-receipt", type=Path)
    parser.add_argument("--reason")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.finalize_chain:
        require(args.run_root is not None and args.run_id, "chain arguments missing")
        result = publish_final_chain(args.run_root, args.run_id)
    elif args.grant:
        require(all((args.admission, args.run_root, args.run_id, args.model,
                     args.destination, args.runner_sha256)), "grant arguments missing")
        result = grant(args.activation, args.admission, args.run_root, args.run_id,
                       args.model, args.destination, args.runner_sha256)
    elif args.abort:
        require(all((args.admission, args.run_root, args.run_id, args.model,
                     args.grant_receipt, args.reason)), "abort arguments missing")
        result = abort(
            args.activation, args.admission, args.run_root, args.run_id, args.model,
            args.grant_receipt, args.reason,
        )
    else:
        require(all((args.admission, args.run_root, args.run_id, args.model,
                     args.grant_receipt, args.output_receipt,
                     args.cleanup_receipt)), "finalize arguments missing")
        result = finalize(
            args.activation, args.admission, args.run_root, args.run_id, args.model,
            args.grant_receipt, args.output_receipt, args.cleanup_receipt,
        )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
