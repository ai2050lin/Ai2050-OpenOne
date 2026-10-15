"""无头探测：AI2050_SKIP_MODEL_LOAD=1 时 server.py 能否完成 import + app 构建。
不真正监听端口，只验证导入链与 FastAPI app 构造（CentOS 部署前置判定）。
"""
import os, sys, time, traceback


def paths_has(paths, probe):
    """路由匹配：支持前缀匹配（FastAPI 路由不含前缀扩展）。"""
    probe = probe.split("?")[0].rstrip("/")
    for p in paths:
        pp = p.rstrip("/")
        if pp == probe or pp.startswith(probe + "/") or probe.startswith(pp + "/"):
            return True
    return False


os.environ["AI2050_SKIP_MODEL_LOAD"] = "1"
os.environ["AI2050_PORT"] = "5099"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, r"D:\AI2050\Ai2050-OpenOne")
sys.path.insert(0, r"D:\AI2050\Ai2050-OpenOne\server")

t0 = time.time()
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ai2050_server_probe", r"D:\AI2050\Ai2050-OpenOne\server\server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    app = getattr(mod, "app", None)
    paths = set()
    if app is not None:
        def walk(routes):
            for r in routes:
                p = getattr(r, "path", None)
                if p:
                    paths.add(p)
                sub = getattr(r, "routes", None)
                if sub:
                    walk(sub)
                app_obj = getattr(r, "app", None)
                if app_obj is not None and hasattr(app_obj, "routes"):
                    walk(app_obj.routes)
        walk(app.routes)
    print("IMPORT_OK app=", type(app).__name__, "routes=", len(paths))
    print("ELAPSED_S %.1f" % (time.time() - t0))
    probes = ["/api/templates", "/api/news", "/atlas/tree", "/api/distributed/summary",
              "/api/research-assets/file/pattern_family_atlas/v1/manifest.json"]
    hit = [p for p in probes if any(paths_has(paths, p) for _ in [0])]
    print("PROBE_ROUTES_FOUND:", len(hit), "/", len(probes))
    for p in probes:
        print("   ", "OK " if paths_has(paths, p) else "MISS", p)
    print("HAS_ATLAS:", any("atlas" in p for p in paths))
    print("HAS_DIST:", any("distributed" in p for p in paths))
    print("HAS_RDC_FEATURE:", any("rdc-feature" in p or "rdc/feature" in p for p in paths))
except Exception:
    print("IMPORT_FAIL")
    traceback.print_exc()
    sys.exit(1)
