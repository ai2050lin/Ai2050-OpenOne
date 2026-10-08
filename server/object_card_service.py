"""对象卡服务（M3-P2，design/ui_decoupled_plan_v2.md §3）

对象 schema = 协议第五件：
    {id, label, layer, evidence, metrics[], activations[], links[], tm_ids, queue_refs}

数据源：server/object_registry.json（注册表 = 唯一内容源，新对象=追加一条，界面零改动）。
富化：related_results ← deploy/distributed_service（按 tm_ids 汇总结果行，deploy/ 缺失时跳过）。
未注册 fid → 404，前端降级 DEMO 卡；组件只认 schema 键，零内容耦合。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api", tags=["object-card"])

REG_PATH = Path(__file__).resolve().parent / "object_registry.json"


def _registry() -> dict:
    try:
        return json.loads(REG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"schema": "object_registry_v1", "objects": []}


def _distributed_rows(tm_ids: list) -> list:
    """尽力富化：deploy/distributed_service 在 :5001 进程内可直接调用（同 server.py 挂载方式）。"""
    out = []
    if not tm_ids:
        return out
    deploy_dir = str(Path(__file__).resolve().parents[1] / "deploy")
    try:
        if deploy_dir not in sys.path:
            sys.path.insert(1, deploy_dir)
        import distributed_service as ds  # type: ignore
        for tm in tm_ids:
            try:
                payload = ds.results_list(tm_id=tm, limit=5)
                for row in payload.get("results", []):
                    out.append({
                        "sha": row.get("sha"), "tm_id": row.get("tm_id"),
                        "model_id": row.get("model_id"), "node_id": row.get("node_id"),
                        "kind": row.get("kind"), "seed": row.get("seed"),
                        "summary_digest": row.get("summary_digest") or {},
                    })
            except Exception:
                continue
    except Exception:
        pass
    return out


@router.get("/objects")
def objects_list() -> dict:
    reg = _registry()
    items = [{"id": o.get("id"), "label": o.get("label"), "evidence": o.get("evidence"),
              "layer": o.get("layer")}
             for o in reg.get("objects", []) if isinstance(o, dict) and o.get("id")]
    return {"schema": reg.get("schema"), "count": len(items), "objects": items}


@router.get("/object/{fid:path}")
def object_detail(fid: str) -> dict:
    fid = (fid or "").strip().strip("/")
    reg = _registry()
    entry = next((o for o in reg.get("objects", [])
                  if isinstance(o, dict) and str(o.get("id", "")).strip("/") == fid), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="object not registered")
    obj = dict(entry)
    tm_ids = [t for t in obj.pop("tm_ids", []) or [] if isinstance(t, str)]
    return {
        "schema": "object_card.v1",
        "registered": True,
        "object": obj,
        "tm_ids": tm_ids,
        "related_results": _distributed_rows(tm_ids),
    }
