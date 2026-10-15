"""dump server.py 的真实路由清单（部署面盘点用）。"""
import os, sys

os.environ["AI2050_SKIP_MODEL_LOAD"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, r"D:\AI2050\Ai2050-OpenOne")
sys.path.insert(0, r"D:\AI2050\Ai2050-OpenOne\server")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "ai2050_server_routes", r"D:\AI2050\Ai2050-OpenOne\server\server.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
app = mod.app

paths = []
seen = set()


def walk(routes, depth=0):
    for r in routes:
        rid = id(r)
        if rid in seen or depth > 6:
            continue
        seen.add(rid)
        p = getattr(r, "path", None)
        if p:
            paths.append(p)
        for attr in ("routes", "app"):
            sub = getattr(r, attr, None)
            if sub is None:
                continue
            if attr == "app" and not hasattr(sub, "routes"):
                sub = getattr(sub, "router", None)
                if sub is not None and not hasattr(sub, "routes"):
                    sub = getattr(getattr(sub, "app", None), "routes", None)
            if hasattr(sub, "__iter__"):
                walk(sub, depth + 1)


walk(app.routes)
uniq = sorted(set(paths))
out = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_server_routes.txt"
with open(out, "w", encoding="utf-8", newline="\n") as f:
    for p in uniq:
        f.write(p + "\n")
print("TOTAL_ROUTES", len(uniq), "-> wrote", out)
for key in ("atlas", "templates", "news", "distributed", "rdc", "research-assets", "pattern"):
    n = sum(1 for p in uniq if key in p.lower())
    print(f"  key={key:16s} hits={n}")
