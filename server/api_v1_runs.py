from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from server.runtime.contracts import AnalysisSpec
from server.runtime.run_service import RunService


def create_runs_router(run_service: RunService) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["runtime-runs"])

    @router.post("/runs")
    async def create_run(spec: AnalysisSpec):
        try:
            record = run_service.create_run(spec)
            return {"status": "success", "run": record}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.get("/runs/{run_id}")
    async def get_run(run_id: str):
        try:
            record = run_service.get_run(run_id)
            return {"status": "success", "run": record}
        except KeyError:
            raise HTTPException(status_code=404, detail="run_id not found")

    @router.get("/runs/{run_id}/events")
    async def get_run_events(
        run_id: str,
        after_step: Optional[int] = Query(default=None, description="Filter by event step."),
        limit: int = Query(default=200, ge=1, le=2000),
    ):
        try:
            events = run_service.get_events(run_id, after_step=after_step, limit=limit)
            return {"status": "success", "run_id": run_id, "count": len(events), "events": events}
        except KeyError:
            raise HTTPException(status_code=404, detail="run_id not found")

    @router.get("/catalog/routes")
    async def list_routes():
        return {"status": "success", "routes": run_service.list_routes()}

    @router.get("/catalog/analyses")
    async def list_analyses():
        return {"status": "success", "analyses": run_service.list_analyses()}

    @router.get("/experiments/timeline")
    async def get_experiment_timeline(
        route: Optional[str] = Query(default=None, description="Filter by route name."),
        limit: int = Query(default=50, ge=1, le=500),
    ):
        payload = run_service.get_experiment_timeline(
            route=route, limit_per_route=limit
        )
        return {"status": "success", "timeline": payload}

    @router.get("/experiments/weekly-report")
    async def get_weekly_report(
        days: int = Query(default=7, ge=1, le=90),
        persist: bool = Query(default=False),
    ):
        payload = run_service.generate_weekly_report(days=days, persist=persist)
        return {"status": "success", **payload}

    return router
