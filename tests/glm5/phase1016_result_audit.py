#!/usr/bin/env python3
"""Independent integrity audit for all Phase1016 artifacts."""

from __future__ import annotations

import hashlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1016_query_factorial_protocol import (
    MODELS,
    OUT_ROOT,
    PHASE,
    PROTOCOL_REVISION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


AUDIT_ROOT = OUT_ROOT / "audit"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def gpu_python_processes() -> list[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [
        line.strip()
        for line in output.splitlines()
        if "python" in line.casefold()
    ]


def main() -> None:
    protocol_audit = read_json(OUT_ROOT / "protocol" / "audit.json")
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    analysis = read_json(OUT_ROOT / "analysis" / "summary.json")
    targeted = read_json(
        OUT_ROOT / "analysis" / "targeted_summary.json"
    )
    selection = read_json(
        OUT_ROOT / "targeted_behavior_scan" / "selection.json"
    )
    checks: dict[str, bool] = {
        "protocol_static_audit_valid": bool(protocol_audit["valid"]),
        "protocol_revision_matches": (
            int(prereg["protocol_revision"]) == PROTOCOL_REVISION
        ),
        "protocol_digest_matches_analysis": (
            prereg["protocol_digest"] == analysis["protocol_digest"]
        ),
        "registered_result_not_supported": (
            analysis["registered_result"]["status"] == "NOT_SUPPORTED"
        ),
        "registered_candidate_count_zero": (
            analysis["registered_result"]["panel_candidate_count"] == 0
        ),
        "target_selection_discovery_only": (
            not selection["confirmation_metrics_used"]
        ),
        "target_selection_behavior_blind": (
            not selection["behavior_labels_used"]
        ),
        "target_selection_digest_matches": (
            selection["selection_digest"]
            == targeted["selection_digest"]
        ),
        "targeted_neuron_gate_closed": (
            not targeted["automatic_continuation_decision"][
                "continue_to_neuron_localization"
            ]
        ),
        "targeted_causal_gate_closed": (
            not targeted["automatic_continuation_decision"][
                "continue_to_causal_closure"
            ]
        ),
    }
    model_rows = []
    required_panel_count = 20
    for model_name in MODELS:
        calibration = read_json(
            OUT_ROOT
            / "behavior_calibration"
            / model_name
            / "selection.json"
        )
        formal = read_json(
            OUT_ROOT / "formal_scan" / model_name / "summary.json"
        )
        target = read_json(
            OUT_ROOT
            / "targeted_behavior_scan"
            / model_name
            / "summary.json"
        )
        model_checks = {
            "calibration_digest": (
                calibration["protocol_digest"]
                == prereg["protocol_digest"]
            ),
            "formal_digest": (
                formal["protocol_digest"] == prereg["protocol_digest"]
            ),
            "formal_precision_bf16": formal["precision"] == "bf16",
            "formal_panel_count": (
                int(formal["panel_count"]) == required_panel_count
            ),
            "formal_forward_count": (
                int(formal["singleton_forward_count"]) == 3360
            ),
            "identity_exact": float(formal["identity_maximum"]) == 0.0,
            "causal_prefix_exact": (
                float(formal["semantic_causal_prefix_maximum"]) == 0.0
            ),
            "target_digest": (
                target["protocol_digest"] == prereg["protocol_digest"]
            ),
            "target_selection_digest": (
                target["selection_digest"]
                == selection["selection_digest"]
            ),
            "target_precision_bf16": target["precision"] == "bf16",
            "target_unit_count": int(target["unit_count"]) == 480,
            "target_factorial_case_count": (
                int(target["factorial_case_count"]) == 1920
            ),
        }
        checks.update({
            f"{model_name}:{key}": value
            for key, value in model_checks.items()
        })
        model_rows.append({
            "model": model_name,
            "selected_prompt_mode": calibration[
                "selected_prompt_mode"
            ],
            "formal_event_count": formal["event_count"],
            "formal_panel_count": formal["panel_count"],
            "formal_singleton_forward_count": formal[
                "singleton_forward_count"
            ],
            "formal_factorial_correct_count": formal[
                "factorial_candidate_all_hit_count"
            ],
            "target_selection_count": target["selection_count"],
            "target_factorial_correct_count": target[
                "factorial_correct_count"
            ],
            "checks": model_checks,
        })

    panel_file_errors = []
    for model_name in MODELS:
        for family_root in sorted(
            path for path in (OUT_ROOT / "formal_scan" / model_name).iterdir()
            if path.is_dir()
        ):
            for panel_root in sorted(
                path for path in family_root.iterdir()
                if path.is_dir() and path.name.startswith("template_")
            ):
                required = (
                    panel_root / "summary.json",
                    panel_root / "units.jsonl",
                    panel_root / "response_scalars.npz",
                    panel_root / "direction_metrics.npz",
                    panel_root / "key_direction_sums.npz",
                )
                if not all(path.exists() for path in required):
                    panel_file_errors.append(str(panel_root))
                    continue
                response = np.load(
                    panel_root / "response_scalars.npz"
                )
                if response["normalized_magnitude"].shape[0] != 24:
                    panel_file_errors.append(
                        str(panel_root) + ":unit_shape"
                    )
                if len(read_jsonl(panel_root / "units.jsonl")) != 24:
                    panel_file_errors.append(
                        str(panel_root) + ":unit_rows"
                    )
    checks["formal_panel_files_complete"] = not panel_file_errors

    script_paths = sorted(
        (ROOT / "tests" / "glm5").glob("phase1016*.py")
    ) + [ROOT / "tests" / "glm5_temp" / "phase1016_token_candidates.py"]
    compile_errors = []
    for path in script_paths:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as error:
            compile_errors.append(f"{path}:{error}")
    checks["all_phase1016_scripts_compile"] = not compile_errors

    forbidden_extensions = {".pt", ".pth", ".bin", ".safetensors", ".npy"}
    forbidden_names = ("raw_hidden", "hidden_tensor", "state_tensor")
    raw_tensor_leaks = [
        str(path.relative_to(OUT_ROOT))
        for path in OUT_ROOT.rglob("*")
        if path.is_file()
        and (
            path.suffix.casefold() in forbidden_extensions
            or any(name in path.name.casefold() for name in forbidden_names)
        )
    ]
    checks["no_raw_hidden_tensor_artifacts"] = not raw_tensor_leaks

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    files_to_hash = sorted(
        path for path in OUT_ROOT.rglob("*")
        if path.is_file() and AUDIT_ROOT not in path.parents
    )
    hash_rows = [{
        "path": str(path.relative_to(OUT_ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    } for path in files_to_hash]
    write_jsonl(AUDIT_ROOT / "hashes.jsonl", hash_rows)
    gpu_processes = gpu_python_processes()
    checks["no_gpu_python_model_processes"] = not gpu_processes
    summary: dict[str, Any] = {
        "schema_version": "phase1016_result_audit.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_digest": prereg["protocol_digest"],
        "selection_digest": selection["selection_digest"],
        "checks": checks,
        "model_rows": model_rows,
        "panel_file_errors": panel_file_errors,
        "compile_errors": compile_errors,
        "raw_tensor_leaks": raw_tensor_leaks,
        "gpu_python_processes": gpu_processes,
        "hashed_file_count": len(hash_rows),
        "hashed_byte_count": sum(row["bytes"] for row in hash_rows),
        "hash_manifest_sha256": sha256(AUDIT_ROOT / "hashes.jsonl"),
        "analysis_counts": {
            "registered_candidate_count": analysis[
                "registered_result"
            ]["panel_candidate_count"],
            "observed_heldout_core_count": analysis[
                "observation_led_result"
            ]["heldout_confirmed_core_count"],
            "targeted_heldout_correct_trace_count": targeted[
                "heldout_correct_trace_count"
            ],
            "targeted_behavior_specific_trace_count": targeted[
                "behavior_specific_trace_count"
            ],
            "targeted_trace_also_in_failed_count": targeted[
                "trace_also_present_in_failed_count"
            ],
        },
    }
    summary["valid"] = all(checks.values())
    write_json(AUDIT_ROOT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
