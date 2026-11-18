# -*- coding: utf-8 -*-
"""READ-ONLY preflight before any cleanup. No deletion / move / rename."""
import os
import time
import shutil
import datetime
import subprocess

ROOT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"
BIN_D = r"D:\$RECYCLE.BIN"
CSV = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\glm5_result_delete_manifest_20261109.csv"
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\glm5_result_preflight_20261109.txt"

NOW = time.time()
REPARSE = 0x400


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return ("%.2f %s" % (n, u)) if u != "B" else ("%d B" % n)
        n /= 1024.0


L = []
w = L.append
w("CLEANUP PREFLIGHT (READ-ONLY)  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("TARGET = " + ROOT)
w("")
w("[1] DISK")
for d in ("C:\\", "D:\\"):
    u = shutil.disk_usage(d)
    w("   %s total=%s used=%s free=%s (%.1f%% used)" % (
        d, human(u.total), human(u.used), human(u.free), 100.0 * u.used / u.total))
w("")
w("[2] RECYCLE BIN D:")
bsz, bfiles = 0, 0
if os.path.isdir(BIN_D):
    for dp, dns, fns in os.walk(BIN_D):
        for fn in fns:
            try:
                bsz += os.lstat(os.path.join(dp, fn)).st_size
                bfiles += 1
            except OSError:
                pass
w("   size=%s files=%d" % (human(bsz), bfiles))
w("   NOTE: 2026-11-01 the recycle bin quota anomaly caused 40.9 GB to be")
w("         permanently deleted instead of moved to bin (irreversible).")
w("")
w("[3] BACKUP FEASIBILITY")
free_total = shutil.disk_usage("C:\\").free + shutil.disk_usage("D:\\").free
w("   free on C+D = %s ; DELETE payload = 297.38 GB" % human(free_total))
w("   => full backup IMPOSSIBLE with current free space.")
w("")
w("[4] LIVE WRITES INTO TARGET")
recent_5m, recent_1h, recent_24h, after_scan = 0, 0, 0, 0
newest = []
scan_ts = datetime.datetime(2026, 11, 9, 23, 19, 18).timestamp()
for dp, dns, fns in os.walk(ROOT, topdown=True, followlinks=False):
    kept = []
    for d in dns:
        q = os.path.join(dp, d)
        try:
            if getattr(os.lstat(q), "st_file_attributes", 0) & REPARSE:
                continue
        except OSError:
            continue
        kept.append(d)
    dns[:] = kept
    for fn in fns:
        q = os.path.join(dp, fn)
        try:
            st = os.lstat(q)
        except OSError:
            continue
        mt = st.st_mtime
        if mt >= NOW - 300:
            recent_5m += 1
        if mt >= NOW - 3600:
            recent_1h += 1
        if mt >= NOW - 86400:
            recent_24h += 1
        if mt > scan_ts:
            after_scan += 1
            newest.append((mt, os.path.relpath(q, ROOT)))
w("   files modified in last 5min=%d, 1h=%d, 24h=%d" % (recent_5m, recent_1h, recent_24h))
w("   files modified AFTER the 23:19 manifest scan = %d" % after_scan)
for mt, rel in sorted(newest, reverse=True)[:12]:
    w("      %s  %s" % (datetime.datetime.fromtimestamp(mt).strftime("%H:%M:%S"), rel))
w("")
w("[5] PROCESSES possibly using target (best effort)")
try:
    import psutil
    hits = []
    for pr in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cl = " ".join(pr.info.get("cmdline") or [])
            low = cl.lower()
            if any(k in low for k in ("glm5", "mlp_", "ai2050", "gpt5")):
                hits.append((pr.info["pid"], pr.info["name"], cl[:220]))
        except Exception:
            pass
    w("   psutil matches = %d" % len(hits))
    for pid, name, cl in hits[:20]:
        w("      pid=%s %s | %s" % (pid, name, cl))
except ImportError:
    w("   psutil unavailable; tasklist fallback:")
    try:
        raw = subprocess.check_output(["tasklist", "/FO", "CSV", "/NH"], stderr=subprocess.DEVNULL)
        txt = raw.decode("gbk", errors="ignore")
        for line in txt.splitlines():
            if any(k in line.lower() for k in ("python", "node", "code")):
                w("      " + line[:200])
    except Exception as e:
        w("      tasklist failed: " + repr(e))
w("")
w("[6] MANIFEST INTEGRITY (sample of 2000 listed paths)")
import csv as _csv
total_rows, checked, missing, size_mismatch = 0, 0, 0, 0
with open(CSV, "r", encoding="utf-8", newline="") as cf:
    rd = _csv.DictReader(cf)
    for i, row in enumerate(rd):
        total_rows += 1
        if i % 112 == 0 and checked < 2000:
            checked += 1
            p = row["path"]
            if not os.path.isfile(p):
                missing += 1
            else:
                try:
                    if os.lstat(p).st_size != int(row["size_bytes"]):
                        size_mismatch += 1
                except OSError:
                    missing += 1
w("   rows=%d sampled=%d missing=%d size_mismatch=%d" % (total_rows, checked, missing, size_mismatch))
w("")
w("=== END PREFLIGHT (nothing deleted) ===")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("OK " + OUT)
