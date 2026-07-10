from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .store import ResearchEvidenceStore


router = APIRouter(prefix="/api/research-kernel", tags=["research-kernel"])
store = ResearchEvidenceStore()


@router.get("/status")
async def get_status():
    return store.status()


@router.get("/models")
async def get_models():
    return store.model_registry()


@router.get("/runs")
async def get_runs(model: str | None = None):
    return {"runs": store.list_runs(model=model)}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    try:
        return store.run_manifest(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}/trace")
async def get_trace(run_id: str, limit: int = Query(default=2000, ge=1, le=20000)):
    try:
        rows = store.run_artifact(run_id, "trace_events")
        return {"run_id": run_id, "count": len(rows), "rows": rows[:limit]}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}/units")
async def get_units(
    run_id: str,
    layer: int | None = None,
    evidence_level: str | None = None,
    limit: int = Query(default=500, ge=1, le=20000),
):
    try:
        rows = store.run_artifact(run_id, "unit_evidence")
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if layer is not None:
        rows = [row for row in rows if row.get("layer") == layer]
    if evidence_level:
        rows = [row for row in rows if row.get("evidence_level") == evidence_level]
    return {"run_id": run_id, "count": len(rows), "rows": rows[:limit]}


@router.get("/runs/{run_id}/validate")
async def validate_run(run_id: str):
    try:
        return store.validate_run(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/claims")
async def get_claims(status: str | None = None):
    return {"claims": store.claims(status=status)}


@router.get("/gaps")
async def get_gaps(status: str | None = None):
    return {"gaps": store.gaps(status=status)}
