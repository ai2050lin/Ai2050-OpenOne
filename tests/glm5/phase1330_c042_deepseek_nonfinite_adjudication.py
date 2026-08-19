#!/usr/bin/env python3
"""Record the frozen DeepSeek non-finite failure and close Phase1330 without rerun."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
OUT = T / "result/phase1330_c042_sequential_behavior"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1330_c042_deepseek_nonfinite_adjudication_audit.py"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run() -> None:
    manifest = load(OUT / "protocol/execution_manifest.json")
    raw = OUT / "raw/deepseek7b_scores.jsonl"
    if not raw.exists() or raw.stat().st_size != 0:
        raise RuntimeError("expected the preserved empty DeepSeek serialization artifact")
    if (OUT / "analysis/deepseek7b_summary.json").exists() or (OUT / "runtime/deepseek7b.json").exists():
        raise RuntimeError("normal DeepSeek outputs unexpectedly exist")
    if not (OUT / "analysis/qwen3_summary.json").exists() or not (OUT / "analysis/glm4_summary.json").exists():
        raise RuntimeError("frozen model order evidence is incomplete")
    failure = {
        "phase": 1330, "campaign": "C042", "model": "deepseek7b",
        "run_status": "formal_run_failed_nonfinite_serialization",
        "observed_exception": "ValueError: Out of range float values are not JSON compliant: nan",
        "localization": "first raw record candidate_scores contained at least one non-finite value",
        "finite_fraction_upper_bound": 575 / 576,
        "frozen_gate_result": {"finite_fraction": False, "remaining_gates": "not_evaluated"},
        "qualified": False, "rerun_authorized": False,
        "raw_empty_artifact_sha256": sha(raw), "raw_empty_artifact_bytes": raw.stat().st_size,
        "execution_manifest_sha256": sha(OUT / "protocol/execution_manifest.json"),
        "adjudicator_sha256": sha(SCRIPT), "auditor_sha256": sha(AUDITOR),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save(OUT / "runtime/deepseek7b_failure.json", failure)
    qwen, glm = load(OUT / "analysis/qwen3_summary.json"), load(OUT / "analysis/glm4_summary.json")
    qualified = [name for name, value in (("qwen3", qwen), ("glm4", glm)) if value["qualified"]]
    final = {
        "phase": 1330, "campaign": "C042", "model_order": manifest["model_order"],
        "model_status": {"qwen3": "qualified" if qwen["qualified"] else "not_qualified",
                         "glm4": "qualified" if glm["qualified"] else "not_qualified",
                         "deepseek7b": "not_qualified_nonfinite"},
        "qualified_models": qualified, "qualified_model_count": len(qualified),
        "all_gates_passed": False, "authorization": "close_c042_before_hidden_states",
        "reason": "Only one of three frozen models qualified; at least two were required.",
        "summary_sha256": {"qwen3": sha(OUT / "analysis/qwen3_summary.json"),
                           "glm4": sha(OUT / "analysis/glm4_summary.json"),
                           "deepseek7b_failure": sha(OUT / "runtime/deepseek7b_failure.json")},
        "formal_reruns": 0, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    run()
