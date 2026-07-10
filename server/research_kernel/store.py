from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import validate_bundle_manifest, validate_trace_event, validate_unit_row


class ResearchEvidenceStore:
    def __init__(self, project_root: Path | str | None = None) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.root = self.project_root / "tests" / "result" / "research_kernel"

    @staticmethod
    def _read_json(path: Path, default: Any = None) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    def manifest(self) -> dict[str, Any]:
        return self._read_json(
            self.root / "manifest.json",
            {
                "schema_version": "research_kernel_manifest.v1",
                "generated_at": None,
                "runs": [],
                "claims": [],
                "gaps": [],
                "progress": {},
            },
        )

    def model_registry(self) -> dict[str, Any]:
        return self._read_json(self.root / "model_registry.json", {"models": []})

    def list_runs(self, model: str | None = None) -> list[dict[str, Any]]:
        runs = list(self.manifest().get("runs") or [])
        if model:
            runs = [run for run in runs if run.get("model") == model]
        return runs

    def run_dir(self, run_id: str) -> Path:
        if not run_id or any(part in run_id for part in ("/", "\\", "..")):
            raise ValueError("invalid run id")
        target = (self.root / "runs" / run_id).resolve()
        target.relative_to(self.root.resolve())
        return target

    def run_manifest(self, run_id: str) -> dict[str, Any]:
        path = self.run_dir(run_id) / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(run_id)
        return self._read_json(path, {})

    def run_artifact(self, run_id: str, artifact_name: str) -> Any:
        run_dir = self.run_dir(run_id)
        manifest = self.run_manifest(run_id)
        artifact = (manifest.get("artifacts") or {}).get(artifact_name)
        if not isinstance(artifact, dict) or not artifact.get("path"):
            raise FileNotFoundError(f"{run_id}:{artifact_name}")
        path = (run_dir / str(artifact["path"])).resolve()
        path.relative_to(run_dir)
        if path.suffix == ".jsonl":
            return self._read_jsonl(path)
        if path.suffix == ".json":
            return self._read_json(path, {})
        return path.read_text(encoding="utf-8")

    def claims(self, status: str | None = None) -> list[dict[str, Any]]:
        rows = list(self.manifest().get("claims") or [])
        return [row for row in rows if row.get("status") == status] if status else rows

    def gaps(self, status: str | None = None) -> list[dict[str, Any]]:
        rows = list(self.manifest().get("gaps") or [])
        return [row for row in rows if row.get("status") == status] if status else rows

    def validate_run(self, run_id: str) -> dict[str, Any]:
        run_dir = self.run_dir(run_id)
        manifest = self.run_manifest(run_id)
        issues = validate_bundle_manifest(manifest, run_dir)
        snapshot = self.run_artifact(run_id, "model_snapshot")
        unit_rows = self.run_artifact(run_id, "unit_evidence")
        trace_rows = self.run_artifact(run_id, "trace_events")
        for index, row in enumerate(unit_rows):
            issues.extend(validate_unit_row(row, snapshot))
            if len(issues) > 500:
                break
        for index, row in enumerate(trace_rows):
            issues.extend(validate_trace_event(row, snapshot))
            if len(issues) > 500:
                break
        return {
            "run_id": run_id,
            "valid": not issues,
            "issue_count": len(issues),
            "issues": [issue.to_dict() for issue in issues[:500]],
            "checked_unit_rows": len(unit_rows),
            "checked_trace_rows": len(trace_rows),
        }

    def status(self) -> dict[str, Any]:
        manifest = self.manifest()
        runs = manifest.get("runs") or []
        return {
            "available": (self.root / "manifest.json").exists(),
            "root": str(self.root.relative_to(self.project_root)).replace("\\", "/"),
            "generated_at": manifest.get("generated_at"),
            "run_count": len(runs),
            "claim_count": len(manifest.get("claims") or []),
            "open_gap_count": sum(1 for row in (manifest.get("gaps") or []) if row.get("status") == "open"),
            "progress": manifest.get("progress") or {},
        }
