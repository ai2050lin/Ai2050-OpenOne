# -*- coding: utf-8 -*-
"""Confirmed cleanup of 5 reproducible junk items. SHA256 registry -> delete -> verify."""
import os
import time
import json
import hashlib
import psutil

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = r"D:\AI2050\Ai2050-OpenOne\gpt5_temp\cleanup_exec_20260930.txt"
REG = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\cleanup_ledger_20260930.json"

# safety: no running phase script should be touching these campaigns
busy = []
for p in psutil.process_iter(["cmdline"]):
    try:
        cl = " ".join(p.info["cmdline"] or [])
        if "phase" in cl.lower() and ".py" in cl:
            busy.append(cl[:200])
    except Exception:
        pass

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# targets: (kind, path)  kind: file | dir
targets = [
    ("file", r"tests\glm5\result\rdc_trusted_rebuild_20260923\14B\lastblock_weights.npz"),
    ("file", r"tests\glm5\result\rdc_trusted_rebuild_20260923\14B_smoke\lastblock_weights.npz"),
    ("dir",  r"tests\codex_temp\stage460_high_dim_factors_20260401"),
    ("dir",  r"tests\glm5\result\_removed_dist_backup_old_version"),
    ("file", r"tests\glm5_temp\qwen3_14b_modelscope_probe_100m.bin"),
]

lines = []
w = lines.append
w("cleanup execution %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
w("running phase processes (advisory): %d" % len(busy))
for b in busy:
    w("  BUSY: %s" % b)

registry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": "confirmed-reproducible junk cleanup (user-approved 2026-09-30)",
            "items": []}

freed = 0
errors = []
for kind, rel in targets:
    full = os.path.join(ROOT, rel)
    if not os.path.exists(full) and not os.path.islink(full):
        w("MISSING (skip): %s" % rel)
        continue
    if kind == "file":
        st = os.stat(full)
        reg = {"path": rel, "size": st.st_size, "mtime": st.st_mtime}
        w("target file: %s (%.1f MB)" % (rel, st.st_size / 1024 / 1024))
        try:
            reg["sha256"] = sha256(full)
            w("  sha256=%s" % reg["sha256"][:16] + "...")
            os.remove(full)
            freed += st.st_size
            w("  DELETED")
            registry["items"].append(reg)
        except Exception as e:
            errors.append("%s: %s" % (rel, e))
            w("  ERROR: %s" % e)
    else:
        w("target dir: %s" % rel)
        for dp, dns, fns in os.walk(full):
            for fn in fns:
                fp = os.path.join(dp, fn)
                try:
                    st = os.stat(fp)
                    reg = {"path": os.path.relpath(fp, ROOT), "size": st.st_size, "mtime": st.st_mtime,
                           "sha256": sha256(fp)}
                    registry["items"].append(reg)
                    w("  file: %s (%.1f KB) sha=%s..." % (os.path.relpath(fp, ROOT), st.st_size / 1024.0, reg["sha256"][:16]))
                except Exception as e:
                    errors.append("%s: %s" % (fp, e))
        try:
            for dp, dns, fns in os.walk(full, topdown=False):
                for fn in fns:
                    os.remove(os.path.join(dp, fn))
                os.rmdir(dp)
            w("  DIR REMOVED")
        except Exception as e:
            errors.append("rmdir %s: %s" % (rel, e))
            w("  DIR ERROR: %s" % e)

# verify absence
w("")
w("---- post-delete verification ----")
for kind, rel in targets:
    full = os.path.join(ROOT, rel)
    gone = not (os.path.exists(full) or os.path.islink(full))
    w("%-70s %s" % (rel, "GONE (ok)" if gone else "STILL PRESENT (FAIL)"))
    if not gone:
        errors.append("still present: %s" % rel)

w("")
w("freed: %.3f GB" % (freed / 1024**3))
w("errors: %d" % len(errors))
for e in errors:
    w("  ERR: %s" % e)

with open(REG, "w", encoding="utf-8") as f:
    json.dump(registry, f, ensure_ascii=False, indent=1)
w("registry written: %s (%d items)" % (REG, len(registry["items"])))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("cleanup done, freed %.3f GB, errors %d" % (freed / 1024**3, len(errors)))
