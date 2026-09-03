from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from server.research_asset_service import research_asset_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = Path(sys.executable)
TRACE_SCRIPT = PROJECT_ROOT / "tests" / "gpt5" / "phase287_real_component_trace.py"
TRACE_ROOT = PROJECT_ROOT / "tests" / "result" / "phase287_real_component_trace"
JOB_ROOT = PROJECT_ROOT / "tests" / "result" / "research_trace_jobs"
REGISTRY_PATH = JOB_ROOT / "registry.json"
FROZEN_MANIFEST = research_asset_root() / "real_component_trace" / "manifest.json"

MODEL_KEYS = {"qwen3", "glm4", "deepseek7b"}
COLOR_LABELS = {"red", "blue", "green", "yellow", "orange", "purple", "brown", "black", "white", "gray", "silver"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


class TraceRunRequest(BaseModel):
    model: str
    prompt: str = Field(min_length=1, max_length=2000)
    target_label: str = "red"
    top_k: int = Field(default=16, ge=4, le=64)
    capture_profile: str = "full_component"

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if value not in MODEL_KEYS:
            raise ValueError(f"unsupported model: {value}")
        return value

    @field_validator("target_label")
    @classmethod
    def validate_target_label(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in COLOR_LABELS:
            raise ValueError(f"unsupported target label: {normalized}")
        return normalized

    @field_validator("capture_profile")
    @classmethod
    def validate_profile(cls, value: str) -> str:
        if value != "full_component":
            raise ValueError("only full_component is currently supported")
        return value


class ResearchTraceManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._gpu_lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = read_json(REGISTRY_PATH, {}).get("jobs", {})
        self._active_process: subprocess.Popen[str] | None = None
        self._active_run_id: str | None = None
        interrupted = False
        for job in self._jobs.values():
            if job.get("status") in {"queued", "running"}:
                job["status"] = "interrupted"
                job["error"] = "server restarted before the trace completed"
                interrupted = True
        if interrupted:
            self._persist()

    def _persist(self) -> None:
        write_json_atomic(REGISTRY_PATH, {"schema_version": "research_trace_registry.v1", "jobs": self._jobs})

    @staticmethod
    def frozen_runs() -> list[dict[str, Any]]:
        manifest = read_json(FROZEN_MANIFEST, {"items": []})
        return [
            {
                **item,
                "source_mode": "replay",
                "status": "complete",
                "validated": True,
            }
            for item in manifest.get("items", [])
        ]

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            live = [dict(job) for job in self._jobs.values()]
        live.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        return live + self.frozen_runs()

    def get(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            if run_id in self._jobs:
                return dict(self._jobs[run_id])
        for item in self.frozen_runs():
            if item.get("run_id") == run_id:
                return item
        raise KeyError(run_id)

    def submit(self, request: TraceRunRequest) -> dict[str, Any]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"live_{request.model}_{request.target_label}_{stamp}_{uuid4().hex[:8]}"
        job = {
            "schema_version": "research_trace_job.v1",
            "run_id": run_id,
            "source_mode": "live",
            "status": "queued",
            "validated": False,
            "model": request.model,
            "prompt": request.prompt,
            "target_label": request.target_label,
            "top_k": request.top_k,
            "capture_profile": request.capture_profile,
            "created_at": utc_now(),
            "started_at": None,
            "completed_at": None,
            "trace_path": None,
            "progress_path": str((JOB_ROOT / run_id / "live_state.json").relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "error": None,
        }
        with self._lock:
            self._jobs[run_id] = job
            self._persist()
        threading.Thread(target=self._run_job, args=(run_id,), daemon=True, name=f"trace-{run_id}").start()
        return dict(job)

    def _set(self, run_id: str, **updates: Any) -> None:
        with self._lock:
            self._jobs[run_id].update(updates)
            self._persist()

    @staticmethod
    def command_for(job: dict[str, Any]) -> list[str]:
        progress_path = job.get("progress_path") or str(
            (JOB_ROOT / str(job["run_id"]) / "live_state.json").relative_to(PROJECT_ROOT)
        ).replace("\\", "/")
        return [
            str(PYTHON_EXE),
            str(TRACE_SCRIPT),
            "--model",
            str(job["model"]),
            "--prompt",
            str(job["prompt"]),
            "--target-label",
            str(job["target_label"]),
            "--top-k",
            str(job["top_k"]),
            "--round-name",
            str(job["run_id"]),
            "--progress-file",
            str(PROJECT_ROOT / str(progress_path)),
            "--skip-public-copy",
        ]

    @staticmethod
    def environment_for(model: str) -> dict[str, str]:
        env = dict(os.environ)
        env.update({
            "HF_HOME": str(PROJECT_ROOT / "models" / "hf"),
            "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "0",
        })
        if model == "glm4":
            env.update(PROBE_DEVICE_MAP_AUTO_MODELS="glm4", PROBE_MAX_GPU_MEMORY="11GiB")
        elif model == "deepseek7b":
            env.update(PROBE_DEVICE_MAP_AUTO_MODELS="deepseek7b", PROBE_MAX_GPU_MEMORY="12GiB")
        return env

    def _run_job(self, run_id: str) -> None:
        with self._gpu_lock:
            job = self.get(run_id)
            if job.get("status") == "cancelled":
                return
            self._set(run_id, status="running", started_at=utc_now())
            log_dir = JOB_ROOT / run_id
            log_dir.mkdir(parents=True, exist_ok=True)
            try:
                process = subprocess.Popen(
                    self.command_for(job),
                    cwd=str(PROJECT_ROOT),
                    env=self.environment_for(str(job["model"])),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                with self._lock:
                    if self._jobs[run_id].get("status") == "cancelled":
                        process.terminate()
                    else:
                        self._active_process = process
                        self._active_run_id = run_id
                stdout, stderr = process.communicate()
                (log_dir / "stdout.log").write_text(stdout, encoding="utf-8")
                (log_dir / "stderr.log").write_text(stderr, encoding="utf-8")
                with self._lock:
                    if self._jobs[run_id].get("status") == "cancelled":
                        return
                trace_path = TRACE_ROOT / run_id / "trace.json"
                if process.returncode != 0 or not trace_path.exists():
                    self._set(
                        run_id,
                        status="failed",
                        completed_at=utc_now(),
                        error=(stderr or stdout or f"exit code {process.returncode}")[-6000:],
                    )
                    return
                trace = read_json(trace_path, {})
                summary = trace.get("summary") or {}
                self._set(
                    run_id,
                    status="complete",
                    completed_at=utc_now(),
                    validated=True,
                    evidence_level="L2",
                    trace_path=str(trace_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    event_count=summary.get("event_count"),
                    vector_count=summary.get("vector_count"),
                    next_token=summary.get("next_token"),
                    evidence_boundary=trace.get("evidence_boundary"),
                )
            except Exception as exc:
                self._set(run_id, status="failed", completed_at=utc_now(), error=str(exc))
            finally:
                with self._lock:
                    self._active_process = None
                    self._active_run_id = None

    def trace(self, run_id: str) -> dict[str, Any]:
        job = self.get(run_id)
        if job.get("source_mode") == "replay":
            path = str(job.get("path") or "")
            filename = Path(path).name
            trace_path = research_asset_root() / "real_component_trace" / filename
        else:
            if job.get("status") != "complete":
                raise RuntimeError(f"trace is not complete: {job.get('status')}")
            trace_path = PROJECT_ROOT / str(job["trace_path"])
        if not trace_path.exists():
            raise FileNotFoundError(trace_path)
        return read_json(trace_path, {})

    def live_state(self, run_id: str) -> dict[str, Any]:
        job = self.get(run_id)
        if job.get("source_mode") != "live":
            raise RuntimeError("live state is only available for a live model run")
        progress_value = str(job.get("progress_path") or "")
        progress_path = PROJECT_ROOT / progress_value if progress_value else None
        payload = read_json(progress_path, {}) if progress_path else {}
        current_layer = payload.get("current_layer")
        hidden_state = payload.get("hidden_state") or {}
        if current_layer is not None and isinstance(hidden_state, dict):
            current_key = str(current_layer)
            payload = {
                **payload,
                "hidden_state": {current_key: hidden_state[current_key]} if current_key in hidden_state else {},
            }
        return {
            "schema_version": "live_state_heatmap.v1",
            "run_id": run_id,
            "model": job.get("model"),
            "prompt": job.get("prompt"),
            "target_label": job.get("target_label"),
            **payload,
            # Manager state wins so cancellation/failure is visible even if the
            # model process stopped before it could publish a final snapshot.
            "status": job.get("status"),
            "stage": "queued" if job.get("status") == "queued" else payload.get("stage", "loading_model"),
            "error": job.get("error"),
        }

    def cancel(self, run_id: str) -> dict[str, Any]:
        job = self.get(run_id)
        if job.get("source_mode") != "live" or job.get("status") not in {"queued", "running"}:
            return job
        with self._lock:
            process = self._active_process
            if self._active_run_id == run_id and process is not None and process.poll() is None:
                process.terminate()
        self._set(run_id, status="cancelled", completed_at=utc_now(), error="cancelled by user")
        return self.get(run_id)


manager = ResearchTraceManager()
router = APIRouter(prefix="/api/research-trace", tags=["research-trace"])


@router.get("/runs")
async def list_trace_runs():
    return {"runs": manager.list_runs()}


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
async def create_trace_run(request: TraceRunRequest):
    return manager.submit(request)


@router.get("/runs/{run_id}")
async def get_trace_run(run_id: str):
    try:
        return manager.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="trace run not found") from exc


@router.get("/runs/{run_id}/trace")
async def get_trace_payload(run_id: str):
    try:
        return manager.trace(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="trace run not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="trace artifact not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/runs/{run_id}/live-state")
async def get_live_state_payload(run_id: str):
    try:
        return manager.live_state(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="trace run not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/runs/{run_id}", status_code=status.HTTP_200_OK)
async def cancel_trace_run(run_id: str):
    try:
        return manager.cancel(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="trace run not found") from exc
