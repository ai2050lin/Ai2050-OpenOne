# -*- coding: utf-8 -*-
"""HTTP 层验证：TestClient（不进 lifespan，不加载模型）打 /api/neurons/* 与 /openapi.json。

用法（仓库根目录）：.venv/Scripts/python.exe tests/gpt5_temp/_verify_http_layer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from server.server import app  # noqa: E402  # 仅导入路由定义，lifespan 不执行

MK = "6be1034f0b55a6df-bf16"
client = TestClient(app)          # 不用 with：跳过 lifespan，模型不加载
out = {}


def check(name, cond, detail=""):
    out[name] = bool(cond)
    print(("[PASS] " if cond else "[FAIL] ") + name + (" | " + detail if detail else ""), flush=True)


r = client.get("/openapi.json")
check("openapi_200", r.status_code == 200)
check("title_fixed", r.json().get("info", {}).get("title") == "AI2050 LLM Mechanism Atlas API",
      str(r.json().get("info", {}).get("title")))

r = client.get(f"/api/neurons/layer", params={"model_key": MK, "block": 16})
j = r.json()
check("http_layer_200_pending_false", r.status_code == 200 and j.get("pending") is False,
      f"code={r.status_code} n_units={j.get('n_units')}")
check("http_layer_explained", len(j.get("explained") or {}) > 0, f"explained={len(j.get('explained') or {})}")

r = client.get("/api/neurons/tower", params={"model_key": MK})
j = r.json()
ready = sum(1 for b in j.get("blocks", []) if b.get("summary"))
check("http_tower_ready", r.status_code == 200 and ready == 36, f"ready={ready}")

r = client.get("/api/neurons/unit", params={"model_key": MK, "block": 16, "unit": 8056})
j = r.json()
check("http_unit_card", r.status_code == 200 and j.get("schema") == "unit_card.v1", str(j.get("id")))

r = client.get("/api/neurons/explained", params={"model_key": MK})
j = r.json()
check("http_explained", r.status_code == 200 and sum(j.get("counts", {}).values()) > 0,
      json.dumps(j.get("counts", {}), ensure_ascii=False))

n_fail = sum(1 for v in out.values() if not v)
print(f"\n== {len(out) - n_fail}/{len(out)} PASS ==", flush=True)
raise SystemExit(0 if n_fail == 0 else 1)
