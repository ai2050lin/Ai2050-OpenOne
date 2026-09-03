from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = PROJECT_ROOT / "tests" / "glm5" / "result" / "client_visualization_assets"


def research_asset_root() -> Path:
    configured = os.environ.get("AI2050_RESEARCH_ASSET_ROOT", "").strip()
    return Path(configured).expanduser().resolve() if configured else DEFAULT_ASSET_ROOT.resolve()


def resolve_research_asset(asset_path: str, *, root: Path | None = None) -> Path:
    asset_root = (root or research_asset_root()).resolve()
    normalized = (asset_path or "").strip().replace("\\", "/").lstrip("/")
    if normalized.startswith("vis_data/"):
        normalized = normalized.removeprefix("vis_data/")
    if not normalized:
        raise ValueError("asset path is required")

    candidate = (asset_root / normalized).resolve()
    if candidate != asset_root and asset_root not in candidate.parents:
        raise ValueError("asset path escapes the research asset root")
    return candidate


router = APIRouter(prefix="/api/research-assets", tags=["research-assets"])


@router.get("/health")
async def research_asset_health():
    root = research_asset_root()
    return {
        "status": "ok" if root.is_dir() else "unavailable",
        "mode": "backend-artifact-store",
        "asset_root": str(root),
        "available": root.is_dir(),
    }


@router.get("/file/{asset_path:path}")
async def get_research_asset(asset_path: str):
    try:
        path = resolve_research_asset(asset_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="research asset not found")
    return FileResponse(
        path,
        filename=None,
        headers={"Cache-Control": "no-store", "X-Research-Asset-Mode": "backend"},
    )
