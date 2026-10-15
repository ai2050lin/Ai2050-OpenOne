"""用 FastAPI 官方 API 取全量路由（server.py 部署面盘点）。"""
import os, sys, json

os.environ["AI2050_SKIP_MODEL_LOAD"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, r"D:\AI2050\Ai2050-OpenOne")
sys.path.insert(0, r"D:\AI2050\Ai2050-OpenOne\server")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "ai2050_r2", r"D:\AI2050\Ai2050-OpenOne\server\server.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
app = mod.app

# FastAPI 1.x 官方：遍历 app.routes 取 path；include_router 的路由就在其中
paths = sorted({getattr(r, "path", "") for r in app.routes if getattr(r, "path", "")})
out = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_server_routes2.txt"
with open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(paths))
print("TOTAL", len(paths))
for key in ("atlas", "templates", "news", "distributed", "rdc-feature", "rdc-prefix",
            "research-assets", "pattern_family_atlas", "api/v1/runs", "research/kernel",
            "research/workspace", "object"):
    n = sum(1 for p in paths if key in p.lower())
    print(f"  {key:22s} {n}")
