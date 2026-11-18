# -*- coding: utf-8 -*-
"""READ-ONLY: resolve which big files are client-served (manifest registered)."""
import os
import json
import ctypes
from datetime import datetime

PROJ = r"D:\AI2050\Ai2050-OpenOne"
RESULT = os.path.join(PROJ, "tests", "glm5", "result")
REPORT = os.path.join(PROJ, "tests", "gpt5", "result", "d_drive_serverdeps_20261117.txt")

REPARSE = 0x400
GFA = ctypes.windll.kernel32.GetFileAttributesW
GFA.restype = ctypes.c_uint32
GFA.argtypes = [ctypes.c_wchar_p]


def is_reparse(p):
    a = GFA(p)
    return a != 0xFFFFFFFF and bool(a & REPARSE)


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return "%.2f %s" % (n, u)
        n /= 1024.0


def ts(x):
    return datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M") if x else "-"


def dirfiles(p):
    out = []
    if not os.path.isdir(p):
        return out
    for dp, dns, fns in os.walk(p, topdown=True):
        dns[:] = [d for d in dns if not is_reparse(os.path.join(dp, d))]
        for fn in fns:
            q = os.path.join(dp, fn)
            try:
                st = os.lstat(q)
            except OSError:
                continue
            out.append((os.path.relpath(q, p), st.st_size, st.st_mtime))
    return out


L = []
w = L.append
w("SERVER RUNTIME DEPENDENCIES / READ-ONLY  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")


def inspect(label, p):
    w("--- %s ---" % label)
    w("    path = %s ; exists=%s ; junction=%s" %
      (p, os.path.exists(p), is_reparse(p) if os.path.exists(p) else "-"))
    mf = os.path.join(p, "manifest.json")
    if os.path.isfile(mf):
        try:
            d = json.loads(open(mf, encoding="utf-8").read())
            w("    manifest: schema=%s state=%s kind=%s run_id=%s" %
              (d.get("schema"), d.get("state"), d.get("kind"), d.get("run_id")))
            fl = d.get("files", [])
            w("    registered files = %d ; registered bytes = %s" %
              (len(fl), human(sum(int(x.get("bytes", 0)) for x in fl))))
            miss = []
            for x in fl:
                q = os.path.join(p, x.get("path", ""))
                if not os.path.isfile(q):
                    miss.append(x.get("path"))
            w("    MISSING registered = %d %s" % (len(miss), miss[:5]))
            reg = set(x.get("path") for x in fl)
        except Exception as e:
            w("    manifest parse ERR " + repr(e))
            reg = set()
    else:
        w("    manifest.json NOT FOUND")
        reg = set()
    fs = dirfiles(p)
    w("    dir total = %s / %d files / last=%s" %
      (human(sum(x[1] for x in fs)), len(fs), ts(max([x[2] for x in fs], default=0))))
    unreg = [x for x in fs if x[0].replace("\\", "/") not in reg and
             not x[0].endswith("manifest.json")]
    w("    UNREGISTERED files = %d / %s" %
      (len(unreg), human(sum(x[1] for x in unreg))))
    for rel, sz, mt in sorted(unreg, key=lambda x: -x[1])[:12]:
        w("        %10s  %-8s %s" %
          (human(sz), ts(mt)[:10], rel[:88]))
    w("")
    return reg


inspect("native_mlp_service qwen4",
        os.path.join(RESULT, "goodfire_middle_20261113_v2", "qwen4"))
inspect("native_mlp_service qwen14",
        os.path.join(RESULT, "goodfire_middle_20261113_v2", "qwen14"))
inspect("qwen14_mlp_service RUN_V2",
        os.path.join(RESULT, "qwen14_client_v2", "native"))
inspect("qwen14_mlp_service RUN (atlas junction)",
        os.path.join(RESULT, "qwen14_mlp_atlas_v1"))

w("[5] goodfire_middle_20261113_v2 ROOT-LEVEL FILES")
root = os.path.join(RESULT, "goodfire_middle_20261113_v2")
for rel, sz, mt in sorted(dirfiles(root), key=lambda x: -x[1])[:18]:
    if os.sep not in rel:
        w("    %10s  %s  %s" % (human(sz), ts(mt), rel))
w("")

w("[6] qwen14_client_v2 TOP-LEVEL")
root2 = os.path.join(RESULT, "qwen14_client_v2")
for n in sorted(os.listdir(root2)):
    p = os.path.join(root2, n)
    if os.path.isfile(p):
        w("    %10s  %s  %s" % (human(os.path.getsize(p)), ts(os.path.getmtime(p)), n))
    else:
        fs = dirfiles(p)
        w("    %10s  %5d files  %s/  %s" %
          (human(sum(x[1] for x in fs)), len(fs), n, ts(max([x[2] for x in fs], default=0))))
w("")

w("[7] ai2050_native SERVICE REFERENCES (grep-like path list)")
for label, rel in [("live_analysis", "_mainline"), ("neuron_census", "_census"),
                   ("state_formation", "visualization-research")]:
    p = os.path.join(r"D:\ai2050_native", rel)
    fs = dirfiles(p)
    w("    %-24s %-28s %10s / %d files / last=%s" %
      (label, rel, human(sum(x[1] for x in fs)), len(fs),
       ts(max([x[2] for x in fs], default=0))))
w("")

w("===== END SERVERDEPS =====")
with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("SERVERDEPS OK")
