# -*- coding: utf-8 -*-
"""Third pass: verify server/client runtime dependencies before any cleanup."""
import os
import json
import hashlib
from datetime import datetime

OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\d_drive_servedeps_20261110.txt"
PROJ = r"D:\AI2050\Ai2050-OpenOne"
RES = os.path.join(PROJ, "tests", "glm5", "result")
NATIVE = r"D:\ai2050_native"

L = []
def w(s=""):
    L.append(str(s))
def gb(n):
    return "%.2f GB" % (n / 1024.0 ** 3)
def mb(n):
    return "%.2f MB" % (n / 1024.0 ** 2)
def ts(t):
    try:
        return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"

def tree(path):
    rows = []
    for root, dirs, files in os.walk(path):
        for fn in files:
            try:
                st = os.stat(os.path.join(root, fn))
                rows.append((os.path.relpath(os.path.join(root, fn), path), st.st_size, st.st_mtime))
            except Exception:
                pass
    return rows

w("SERVER / CLIENT RUNTIME DEPENDENCY CHECK (READ-ONLY)  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("")

w("[1] ENVIRONMENT OVERRIDES")
for k in ("AI2050_QWEN14_MLP_DIR", "AI2050_DIST_DIR", "AI2050_NATIVE_DIR", "AI2050_RESEARCH_ASSET_ROOT"):
    w("    %-32s = %s" % (k, os.environ.get(k, "(unset)")))
w("")

w("[2] qwen14_mlp_service.py DEFAULT FOLDER")
q = os.path.join(RES, "qwen14_mlp_atlas_v1")
w("    " + q)
if os.path.isdir(q):
    rows = tree(q)
    w("    EXISTS  %s / %d files" % (gb(sum(r[1] for r in rows)), len(rows)))
    man = os.path.join(q, "manifest.json")
    if os.path.isfile(man):
        try:
            d = json.loads(open(man, encoding="utf-8").read())
            names = [f.get("path") for f in d.get("files", [])]
            w("    manifest: schema=%s kind=%s state=%s run_id=%s  files=%d" % (
                d.get("schema"), d.get("kind"), d.get("state"), d.get("run_id"), len(names)))
            missing = [n for n in names if not os.path.isfile(os.path.join(q, n))]
            w("    registered files missing on disk: %d" % len(missing))
            for n in missing[:8]:
                w("        MISSING " + str(n))
            need = {"report.json", "documents.json", "tokenizer.json", "prefix-extrema.npz",
                    "natural_exact-effects.json"}
            for b in range(28, 36):
                for f in ["G", "V", "A", "input", "W-gate", "W-up", "W-down"]:
                    need.add("B%d-%s.npy" % (b, f))
                for f in ["profile", "cues", "prediction", "readout"]:
                    need.add("B%d-%s.json" % (b, f))
            lack = need - set(names)
            w("    required-by-service but not registered: %d" % len(lack))
            for n in sorted(lack)[:8]:
                w("        LACK " + n)
        except Exception as e:
            w("    manifest parse error: " + repr(e))
    else:
        w("    manifest.json NOT PRESENT -> service would return 404")
    for r in sorted(rows, key=lambda x: -x[1])[:8]:
        w("        %10s  %s" % (gb(r[1]), r[0][:70]))
else:
    w("    MISSING -> /api/research/qwen14-mlp would 404 unless env override is set")
w("")

w("[3] native visualization-research/qwen14_mlp_atlas_v1 (candidate source of that pack)")
nv = os.path.join(NATIVE, "visualization-research", "qwen14_mlp_atlas_v1")
if os.path.isdir(nv):
    rows = tree(nv)
    w("    %s / %d files" % (gb(sum(r[1] for r in rows)), len(rows)))
    w("    top 12 by size:")
    for r in sorted(rows, key=lambda x: -x[1])[:12]:
        w("        %10s  %s  %s" % (gb(r[1]), ts(r[2]), r[0][:66]))
    nv_names = set(r[0] for r in rows)
    q_names = set()
    if os.path.isdir(q):
        q_names = set(r[0] for r in tree(q))
    w("    name overlap with result-side pack: %d / %d native files" % (len(nv_names & q_names), len(nv_names)))
else:
    w("    absent")
w("")

w("[4] OTHER SERVER-READ NATIVE DIRS")
for label, p in (("_mainline/MA-H2", os.path.join(NATIVE, "_mainline", "MA-H2")),
                 ("_census", os.path.join(NATIVE, "_census")),
                 ("_agent_home/node_service", os.path.join(NATIVE, "_agent_home", "node_service")),
                 ("_weights", os.path.join(NATIVE, "_weights"))):
    if not os.path.isdir(p):
        w("    %-32s MISSING" % label)
        continue
    rows = tree(p)
    w("    %-32s %10s / %5d files / last=%s" % (
        label, gb(sum(r[1] for r in rows)), len(rows), ts(max(r[2] for r in rows)) if rows else "?"))
w("")

w("[5] CLIENT ASSET ROOT + RELEASE PACKS")
for label, p in (("client_visualization_assets", os.path.join(RES, "client_visualization_assets")),
                 ("release/deployment-20261108", os.path.join(PROJ, "release", "deployment-20261108")),
                 ("distributed_data/datapack", os.path.join(PROJ, "distributed_data", "datapack")),
                 ("distributed_data/research_library", os.path.join(PROJ, "distributed_data", "research_library"))):
    if not os.path.isdir(p):
        w("    %-40s MISSING" % label)
        continue
    rows = tree(p)
    w("    %-40s %10s / %6d files / last=%s" % (
        label, gb(sum(r[1] for r in rows)), len(rows), ts(max(r[2] for r in rows)) if rows else "?"))
w("")

w("[6] UPLOAD / PUBLISH ARTIFACTS on disk (tar.gz / research bundles)")
hits = []
for base in (PROJ, NATIVE):
    for root, dirs, files in os.walk(base):
        dirs[:] = [x for x in dirs if x not in (".git", "node_modules", "objects", "lfs")]
        for fn in files:
            low = fn.lower()
            if low.endswith((".tar.gz", ".tgz", ".zip")) and (low.startswith("ai2050-") or "research" in low or "deployment" in low or "native" in low):
                p = os.path.join(root, fn)
                try:
                    st = os.stat(p)
                    if st.st_size >= 20 * 1024 ** 2:
                        hits.append((st.st_size, p, st.st_mtime))
                except Exception:
                    pass
w("    matched >=20MB: %d" % len(hits))
for sz, p, mt in sorted(hits, reverse=True)[:20]:
    w("        %10s  %s  %s" % (gb(sz), ts(mt), p[:110]))
w("")

w("[7] CAS BLOB REFERENCE CHECK")
blobdir = os.path.join(PROJ, "distributed_data", "research_native_blobs")
blobs = tree(blobdir)
w("    research_native_blobs: %s / %d files" % (gb(sum(r[1] for r in blobs)), len(blobs)))
w("    last write: %s" % (ts(max(r[2] for r in blobs)) if blobs else "?"))
w("    extension sample: " + ", ".join(sorted(set(os.path.splitext(r[0])[1] for r in blobs))[:12]))
# does any manifest/json in the project name one of these blob hashes?
sample = [os.path.splitext(os.path.basename(r[0]))[0] for r in blobs[:5]]
w("    sample blob ids: " + ", ".join(x[:24] for x in sample))
w("")

w("=== END (READ-ONLY) ===")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("done", len(L), "lines")
