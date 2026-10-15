"""展开 FastAPI 0.141 的 _IncludedRouter，取真实全量路由。"""
import os, sys

os.environ["AI2050_SKIP_MODEL_LOAD"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, r"D:\AI2050\Ai2050-OpenOne")
sys.path.insert(0, r"D:\AI2050\Ai2050-OpenOne\server")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "ai2050_r3", r"D:\AI2050\Ai2050-OpenOne\server\server.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
app = mod.app

types = {}
for r in app.routes:
    types[type(r).__name__] = types.get(type(r).__name__, 0) + 1
print("ROUTE_TYPES:", types)

inc = [r for r in app.routes if type(r).__name__ == "_IncludedRouter"]
if inc:
    r0 = inc[0]
    print("ATTRS:", [a for a in dir(r0) if not a.startswith("__")][:40])
    for attr in ("routes", "router", "prefix", "app"):
        v = getattr(r0, attr, "MISSING")
        print(f"  {attr}: {type(v).__name__}", (len(v.routes) if hasattr(v, "routes") else ""))

paths = set()


def walk(routes, depth=0):
    if depth > 10:
        return
    for r in routes:
        p = getattr(r, "path", None)
        if isinstance(p, str) and p:
            paths.add(p)
        # FastAPI 0.141: _IncludedRouter 持有 original_router
        orig = getattr(r, "original_router", None)
        if orig is not None and hasattr(orig, "routes"):
            walk(orig.routes, depth + 1)
        for attr in ("routes", "router"):
            sub = getattr(r, attr, None)
            if sub is not None and hasattr(sub, "__iter__") and not isinstance(sub, str):
                walk(sub, depth + 1)


walk(app.routes)

out = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_server_routes3.txt"
with open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(sorted(paths)))
print("TOTAL_PATHS", len(paths))
for key in ("atlas", "templates", "news", "distributed", "rdc-feature", "rdc-prefix",
            "research-assets", "pattern_family_atlas", "runs", "research/kernel",
            "research/workspace", "object", "object-card"):
    print(f"  {key:22s}", sum(1 for p in paths if key in p.lower()))
