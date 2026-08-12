#!/usr/bin/env python3
"""Integrity and claim-boundary audit for Phase1017."""

from __future__ import annotations

import hashlib
import json
import py_compile
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1017_semantic_niche_protocol import (
    MODELS,
    OUT_ROOT,
    PHASE,
    PROTOCOL_REVISION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


SOURCE_PATHS = (
    Path(
        r"C:\Users\Admin\.codex\attachments"
        r"\1934a27e-c0d5-49e8-9aae-0e177d9c569d\pasted-text.txt"
    ),
    Path(
        r"C:\Users\Admin\.codex\attachments"
        r"\5a67efb2-ebfd-4f99-87e7-e695067c65aa\pasted-text.txt"
    ),
    ROOT
    / "research"
    / "MainAnalysis"
    / "20260727_01_###关键###deepseek每个词具有独特的生态位.md",
    ROOT
    / "research"
    / "MainAnalysis"
    / "20260727_01_###关键###gpt5.6每个词具有独特的生态位.md",
)
PHASE_SCRIPTS = (
    "phase1017_semantic_niche_protocol.py",
    "phase1017_semantic_niche_behavior.py",
    "phase1017_semantic_niche_scan.py",
    "phase1017_semantic_niche_finalize.py",
    "phase1017_semantic_niche_targeted_behavior.py",
    "phase1017_result_audit.py",
)
REQUIRED_PANEL_FILES = (
    "summary.json",
    "response_scalars.npz",
    "direction_metrics.npz",
    "key_direction_sums.npz",
    "units.jsonl",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def gpu_python_processes() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        if "python" in parts[1].lower():
            rows.append({
                "pid": parts[0],
                "process_name": parts[1],
                "used_memory_mib": parts[2],
            })
    return rows


def main() -> None:
    audit_root = OUT_ROOT / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    checks: dict[str, bool] = {}

    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    checks["protocol_revision"] = (
        int(prereg["protocol_revision"]) == PROTOCOL_REVISION
    )
    checks["protocol_word_count"] = len(prereg["words"]) == 12
    checks["protocol_model_count"] = len(MODELS) == 3
    protocol_digest = prereg["protocol_digest"]

    behavior_ok = True
    scan_ok = True
    panel_ok = True
    zero_checks_ok = True
    for model_name in MODELS:
        selection = read_json(
            OUT_ROOT / "behavior" / model_name / "selection.json"
        )
        formal_rows = read_jsonl(
            OUT_ROOT / "behavior" / model_name / "formal.jsonl"
        )
        behavior_ok &= selection["protocol_digest"] == protocol_digest
        behavior_ok &= len(formal_rows) == 768

        scan = read_json(OUT_ROOT / "formal_scan" / model_name / "summary.json")
        scan_ok &= scan["protocol_digest"] == protocol_digest
        scan_ok &= int(scan["panel_count"]) == 24
        scan_ok &= int(scan["unit_count"]) == 192
        scan_ok &= int(scan["singleton_forward_count"]) == 1728
        zero_checks_ok &= float(scan["identity_maximum"]) == 0.0
        zero_checks_ok &= float(scan["interaction_cue_maximum"]) == 0.0
        zero_checks_ok &= (
            float(scan["target_embedding_interaction_maximum"]) == 0.0
        )
        panels = list(
            (OUT_ROOT / "formal_scan" / model_name).glob("*/*/summary.json")
        )
        panel_ok &= len(panels) == 24
        for panel_summary in panels:
            panel_root = panel_summary.parent
            panel_ok &= all(
                (panel_root / filename).exists()
                for filename in REQUIRED_PANEL_FILES
            )

    checks["behavior_complete_and_digest_matched"] = behavior_ok
    checks["formal_scan_complete_and_digest_matched"] = scan_ok
    checks["formal_panel_files_complete"] = panel_ok
    checks["identity_and_causal_prefix_zero"] = zero_checks_ok

    analysis = read_json(OUT_ROOT / "analysis" / "summary.json")
    checks["analysis_counts"] = (
        int(analysis["panel_count"]) == 72
        and int(analysis["unit_count"]) == 576
        and int(analysis["singleton_forward_count"]) == 5184
        and int(analysis["heldout_confirmed_word_core_count"]) == 8009
        and int(analysis["shared_physical_core_count"]) == 1817
    )
    sensitivity = read_jsonl(
        OUT_ROOT / "analysis" / "threshold_sensitivity.jsonl"
    )
    checks["threshold_grid_complete"] = len(sensitivity) == 27

    selection = read_json(
        OUT_ROOT / "targeted_behavior_scan" / "selection.json"
    )
    selection_rows = read_jsonl(
        OUT_ROOT / "targeted_behavior_scan" / "selection.jsonl"
    )
    checks["target_selection_discovery_only_and_behavior_blind"] = (
        bool(selection["selection_used_discovery_only"])
        and not bool(selection["selection_used_behavior"])
        and not bool(selection["selection_used_confirmation"])
        and len(selection_rows) == 24
    )

    targeted = read_json(OUT_ROOT / "targeted_behavior_scan" / "summary.json")
    checks["targeted_followup_complete"] = (
        int(targeted["model_count"]) == 3
        and int(targeted["selection_count"]) == 24
        and int(targeted["batched_forward_count"]) == 576
        and int(targeted["confirmation_unit_scalar_count"]) == 2304
    )
    continuation = targeted["automatic_continuation_assessment"]
    checks["neuron_and_causal_gates_closed"] = (
        not continuation["continue_to_neuron_localization"]
        and not continuation["continue_to_causal_closure"]
    )

    compile_ok = True
    for filename in PHASE_SCRIPTS:
        try:
            py_compile.compile(
                str(ROOT / "tests" / "glm5" / filename),
                doraise=True,
            )
        except py_compile.PyCompileError:
            compile_ok = False
    checks["all_phase_scripts_compile"] = compile_ok

    forbidden = []
    for path in OUT_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".pt",
            ".pth",
            ".safetensors",
            ".npy",
        }:
            forbidden.append(path.relative_to(OUT_ROOT).as_posix())
    checks["no_raw_tensor_artifacts"] = not forbidden

    provenance = []
    for path in SOURCE_PATHS:
        provenance.append({
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
        })
    checks["all_requested_sources_present"] = all(
        row["exists"] for row in provenance
    )

    manifest_rows = []
    total_bytes = 0
    for path in sorted(OUT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(OUT_ROOT)
        if relative.parts and relative.parts[0] == "audit":
            continue
        size = path.stat().st_size
        total_bytes += size
        manifest_rows.append({
            "path": relative.as_posix(),
            "size_bytes": size,
            "sha256": sha256_file(path),
        })
    manifest_digest = canonical_digest(manifest_rows)
    write_jsonl(audit_root / "hash_manifest.jsonl", manifest_rows)

    gpu_rows = gpu_python_processes()
    checks["no_gpu_python_model_process"] = not gpu_rows
    summary = {
        "schema_version": "phase1017_result_audit.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "timestamp": datetime.now(
            ZoneInfo("America/Chicago")
        ).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "valid": all(checks.values()),
        "checks": checks,
        "protocol_digest": protocol_digest,
        "selection_digest": selection["selection_digest"],
        "pre_audit_file_count": len(manifest_rows),
        "pre_audit_total_bytes": total_bytes,
        "hash_manifest_sha256": manifest_digest,
        "source_provenance": provenance,
        "forbidden_raw_tensor_files": forbidden,
        "gpu_python_processes": gpu_rows,
        "claim_boundary": {
            "supported": (
                "held-out contextual target-conditioned interaction patterns "
                "and shared physical resources with word-conditioned directions"
            ),
            "not_supported": (
                "persistent lexical plasticity, unique fixed word slots, "
                "behavior-sufficient paths, neuron mechanisms, or causal closure"
            ),
        },
    }
    write_json(audit_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
