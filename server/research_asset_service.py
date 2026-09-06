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


@router.get("/native-parameter")
def get_native_parameter(model: str = "qwen4", case: int = 0, j: int = 0, k: int = 0, checkpoint: int | None = None, token: int | None = None):
    from .native_parameter_query import query
    return query(model, case, j, k, checkpoint, token)


@router.get("/health")
async def research_asset_health():
    root = research_asset_root()
    return {
        "status": "ok" if root.is_dir() else "unavailable",
        "mode": "backend-artifact-store",
        "asset_root": str(root),
        "available": root.is_dir(),
    }


@router.get("/native-path-frames")
def get_native_path_frames():
    from .native_path_parameter_query import frame_options
    return frame_options()


@router.get("/native-path-parameter")
def get_native_path_parameter(frame: int = 0, layer: int = 35, module: str = "v_proj", j: int = 0, k: int = 0):
    from .native_path_parameter_query import query
    return query(frame, layer, module, j, k)


@router.get("/native-precision-frames")
def get_native_precision_frames():
    from .native_precision_parameter_query import options
    return options()


@router.get("/native-precision-parameter")
def get_native_precision_parameter(frame: int = 20, layer: int = 0, module: str = "v_proj", j: int = 0, k: int = 0, hj: int = 0, ak: int = 0):
    from .native_precision_parameter_query import query
    return query(frame, layer, module, j, k, hj, ak)


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


@router.get("/native-operation-cases")
def get_native_operation_cases():
    from .native_operation_parameter_query import options
    return options()


@router.get("/native-operation-parameter")
def get_native_operation_parameter(case: int = 0, layer: int = 0, j: int = 57, k: int = 32, hj: int = 0, ak: int = 0, checkpoint: int = 1, token: int | None = None):
    from .native_operation_parameter_query import query
    return query(case, layer, j, k, hj, ak, checkpoint, token)


@router.get("/native-output-cases")
def get_native_output_cases():
    from .native_output_parameter_query import options
    return options()


@router.get("/native-output-parameter")
def get_native_output_parameter(case: int = 0, layer: int = 0, j: int = 57, k: int = 32, hj: int = 0, ak: int = 0, checkpoint: int = 1, token: int | None = None):
    from .native_output_parameter_query import query
    return query(case, layer, j, k, hj, ak, checkpoint, token)


@router.get("/native-sequence-cases")
def get_native_sequence_cases():
    from .native_sequence_parameter_query import options
    return options()


@router.get("/native-sequence-parameter")
def get_native_sequence_parameter(case: int = 256, layer: int = 0, j: int = 57, k: int = 32, hj: int = 0, ak: int = 0, checkpoint: int = 1, token: int | None = None):
    from .native_sequence_parameter_query import query
    return query(case, layer, j, k, hj, ak, checkpoint, token)


@router.get("/native-multitoken-cases")
def get_native_multitoken_cases():
    from .native_multitoken_parameter_query import options
    return options()


@router.get("/native-multitoken-parameter")
def get_native_multitoken_parameter(case: int = 256, layer: int = 0, j: int = 57, k: int = 32, hj: int = 0, ak: int = 0, checkpoint: int = 1, token: int | None = None):
    from .native_multitoken_parameter_query import query
    return query(case, layer, j, k, hj, ak, checkpoint, token)


@router.get("/native-atlas-panels")
def get_native_atlas_panels(compact: bool = False):
    from .native_atlas_heatmap_query import options
    return options(include_rows=not compact)


@router.get("/native-mlp-cases")
def get_native_mlp_cases():
    from .native_mlp_parameter_query import options
    return options()


@router.get("/native-mlp-parameter")
def get_native_mlp_parameter(case: int = 256, layer: int = 23, unit: int = 6197, coordinate: int = 0, checkpoint: int = 24, token: int | None = None):
    from .native_mlp_parameter_query import query
    return query(case, layer, unit, coordinate, checkpoint, token)


@router.get("/native-atlas-rows")
def get_native_atlas_rows(panel: str, start: int = 0, count: int = 8):
    from .native_atlas_heatmap_query import rows
    return rows(panel, start, count)


@router.get("/native-source-cases")
def get_native_source_cases():
    from .native_source_parameter_query import options
    return options()


@router.get("/native-source-parameter")
def get_native_source_parameter(dataset: str = 'fresh', case: int = 128, layer: int = 23, unit: int = 6197,
                               coordinate: int = 0, checkpoint: int = 24, source_token: int = 0,
                               head: int = 0, head_coordinate: int = 0, query_position: int = 1, hidden_token: int | None = None):
    from .native_source_parameter_query import query
    return query(dataset, case, layer, unit, coordinate, checkpoint, source_token, head, head_coordinate, query_position, hidden_token)
