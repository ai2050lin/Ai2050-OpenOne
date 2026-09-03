from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "tests" / "glm5" / "result" / "auto_rnd"
TEMP_ROOT = PROJECT_ROOT / "tests" / "glm5_temp" / "auto_rnd"
KERNEL_ROOT = PROJECT_ROOT / "tests" / "glm5" / "result" / "research_kernel"
LEGACY_KERNEL_ROOT = PROJECT_ROOT / "tests" / "result" / "research_kernel"
PYTHON_EXE = Path(r"C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe")

MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")

ROLE_DEFINITIONS = {
    "principal_investigator": "提出可证伪问题并作最终裁决；不能用多模型意见替代实验数据。",
    "method_reviewer": "只审核样本、对照、判据、混杂因素和复现方案。",
    "code_agent": "只实现已批准的 ExperimentSpec，并输出规定格式的产物。",
    "data_auditor": "只依据完整原始产物复算基本量，不依据总结文本。",
    "adversarial_reviewer": "主动寻找反例、副作用、跨模型冲突和替代解释。",
    "theory_synthesizer": "只有裁决为 accepted 时才可提升理论主张等级。",
}

_GPU_LOCK = threading.Lock()
_ACTIVE_PROCESS_LOCK = threading.Lock()
_ACTIVE_PROCESS: subprocess.Popen[str] | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(value: str, limit: int = 42) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    slug = "-".join(part for part in slug.split("-") if part)
    return (slug or "research")[:limit]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResearchRun:
    run_id: str
    root: Path
    objective: str

    @property
    def script_path(self) -> Path:
        return self.root / "generated_script.py"

    @property
    def experiment_path(self) -> Path:
        return self.root / "experiment.json"

    @property
    def execution_path(self) -> Path:
        return self.root / "execution.json"


def create_research_run(objective: str, round_number: int) -> ResearchRun:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"airnd_r{round_number:04d}_{stamp}_{_safe_slug(objective)}"
    root = RUN_ROOT / run_id
    root.mkdir(parents=True, exist_ok=False)
    run = ResearchRun(run_id=run_id, root=root, objective=objective)
    experiment = {
        "schema_version": "experiment_spec.v1",
        "run_id": run_id,
        "objective": objective,
        "created_at": _utc_now(),
        "model_order": list(MODEL_ORDER),
        "required_stages": ["smoke", "full", "independent_replication"],
        "required_controls": [
            "matched_negative_control",
            "random_same_size_control",
            "unrelated_task_clean_control",
            "heldout_template_or_object",
        ],
        "required_outputs": [
            "model_snapshot.json",
            "cases.jsonl",
            "trace_events.jsonl",
            "unit_evidence.jsonl",
            "intervention_rows.jsonl",
            "manifest.json",
        ],
        "decision_rule": {
            "accepted": "all required artifacts validate and preregistered predictions pass on heldout data",
            "rejected": "a preregistered prediction is contradicted by valid data",
            "inconclusive": "artifacts, controls, scale, or cross-model evidence are incomplete",
        },
        "evidence_policy": "AI agreement is commentary; only validated artifacts can change a claim.",
    }
    _write_json(run.experiment_path, experiment)
    return run


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] in {"socket", "requests", "httpx", "urllib", "shutil"}:
                self.errors.append(f"blocked import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root in {"socket", "requests", "httpx", "urllib", "shutil"}:
            self.errors.append(f"blocked import: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts: list[str] = []
            cursor: ast.expr = node.func
            while isinstance(cursor, ast.Attribute):
                parts.append(cursor.attr)
                cursor = cursor.value
            if isinstance(cursor, ast.Name):
                parts.append(cursor.id)
            name = ".".join(reversed(parts))
        if name in {"eval", "exec", "compile", "__import__", "os.system", "subprocess.Popen", "subprocess.run", "subprocess.call"}:
            self.errors.append(f"blocked call: {name}")
        self.generic_visit(node)


def validate_generated_code(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"ok": False, "errors": [f"syntax error at line {exc.lineno}: {exc.msg}"]}
    visitor = _SafetyVisitor()
    visitor.visit(tree)
    return {"ok": not visitor.errors, "errors": visitor.errors}


def load_evidence_context() -> dict[str, Any]:
    kernel_root = KERNEL_ROOT if (KERNEL_ROOT / "manifest.json").exists() else LEGACY_KERNEL_ROOT
    manifest_path = kernel_root / "manifest.json"
    if not manifest_path.exists():
        return {"available": False, "reason": "research kernel manifest is missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claims_path = kernel_root / "claims.jsonl"
    gaps_path = kernel_root / "gaps.jsonl"

    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    claims = read_jsonl(claims_path)
    gaps = read_jsonl(gaps_path)
    return {
        "available": True,
        "schema_version": manifest.get("schema_version"),
        "run_count": len(manifest.get("runs", [])),
        "models": sorted({row.get("model") for row in manifest.get("runs", []) if row.get("model")}),
        "claim_counts": {
            state: sum(1 for claim in claims if claim.get("status") == state)
            for state in ("accepted", "rejected", "inconclusive", "open")
        },
        "open_gaps": gaps[:20],
        "source": str(manifest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }


def _artifact_rows(path: Path) -> int | None:
    if path.suffix != ".jsonl":
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def artifact_audit(run: ResearchRun) -> dict[str, Any]:
    experiment = json.loads(run.experiment_path.read_text(encoding="utf-8"))
    required = experiment.get("required_outputs", [])
    artifacts = []
    for path in sorted(run.root.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        artifacts.append({
            "path": str(path.relative_to(run.root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "rows": _artifact_rows(path),
        })
    names = {item["path"] for item in artifacts}
    missing = [name for name in required if name not in names]
    execution = {}
    if run.execution_path.exists():
        execution = json.loads(run.execution_path.read_text(encoding="utf-8"))
    if execution.get("status") != "success":
        decision = "inconclusive"
        reason = "execution did not complete successfully"
    elif missing:
        decision = "inconclusive"
        reason = "required evidence artifacts are incomplete"
    else:
        decision = "pending_review"
        reason = "artifact completeness passed; scientific predictions still require independent review"
    audit = {
        "schema_version": "artifact_audit.v1",
        "run_id": run.run_id,
        "created_at": _utc_now(),
        "decision": decision,
        "reason": reason,
        "missing_required_artifacts": missing,
        "artifacts": artifacts,
    }
    _write_json(run.root / "artifact_audit.json", audit)
    return audit


def execute_research_code(
    code: str,
    run: ResearchRun,
    timeout: int = 120,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    safety = validate_generated_code(code)
    run.script_path.write_text(code, encoding="utf-8")
    if not safety["ok"]:
        result = {"status": "rejected", "error": "; ".join(safety["errors"]), "safety": safety}
        _write_json(run.execution_path, result)
        return result

    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    temp_script = TEMP_ROOT / f"{run.run_id}.py"
    temp_script.write_text(code, encoding="utf-8")
    env = {
        **os.environ,
        "HF_HOME": r"D:\develop\model",
        "HF_ENDPOINT": "https://hf-mirror.com",
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "0",
        "AI_RND_RUN_ID": run.run_id,
        "AI_RND_RUN_DIR": str(run.root),
        "AI_RND_MODEL_ORDER": ",".join(MODEL_ORDER),
    }
    started = time.monotonic()
    with _GPU_LOCK:
        process = subprocess.Popen(
            [str(PYTHON_EXE), str(temp_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        global _ACTIVE_PROCESS
        with _ACTIVE_PROCESS_LOCK:
            _ACTIVE_PROCESS = process
        result: dict[str, Any] | None = None
        try:
            while process.poll() is None:
                if stop_requested and stop_requested():
                    process.terminate()
                    stdout, stderr = process.communicate(timeout=10)
                    result = {"status": "stopped", "output": stdout[-12000:], "error": stderr[-8000:]}
                    break
                if time.monotonic() - started > timeout:
                    process.terminate()
                    stdout, stderr = process.communicate(timeout=10)
                    result = {
                        "status": "error",
                        "output": stdout[-12000:],
                        "error": f"Execution timed out after {timeout}s\n{stderr[-8000:]}",
                    }
                    break
                time.sleep(0.2)
            if result is None:
                stdout, stderr = process.communicate()
                result = {
                    "status": "success" if process.returncode == 0 else "error",
                    "output": stdout[-12000:],
                    "error": stderr[-8000:] or None,
                    "return_code": process.returncode,
                }
        finally:
            with _ACTIVE_PROCESS_LOCK:
                _ACTIVE_PROCESS = None
    assert result is not None
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    result["run_id"] = run.run_id
    result["model_order"] = list(MODEL_ORDER)
    _write_json(run.execution_path, result)
    return result


def terminate_active_process() -> bool:
    with _ACTIVE_PROCESS_LOCK:
        process = _ACTIVE_PROCESS
    if process is None or process.poll() is not None:
        return False
    process.terminate()
    return True


def list_research_runs(limit: int = 50) -> list[dict[str, Any]]:
    if not RUN_ROOT.exists():
        return []
    rows = []
    for root in sorted((path for path in RUN_ROOT.iterdir() if path.is_dir()), reverse=True)[:limit]:
        experiment_path = root / "experiment.json"
        audit_path = root / "artifact_audit.json"
        experiment = json.loads(experiment_path.read_text(encoding="utf-8")) if experiment_path.exists() else {}
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
        rows.append({
            "run_id": root.name,
            "objective": experiment.get("objective", ""),
            "created_at": experiment.get("created_at"),
            "decision": audit.get("decision", "running_or_not_audited"),
            "missing_required_artifacts": audit.get("missing_required_artifacts", []),
        })
    return rows
