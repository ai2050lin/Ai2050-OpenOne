# -*- coding: utf-8 -*-
"""
READ-ONLY: for every still-existing P2-cold file, count hard links.
nlink==1  -> deleting it really frees space
nlink>=2  -> deleting it only removes one directory entry; data lives on
Also survey the CAS blob store.
"""
import os
import csv
import time
import datetime
import collections

CSV = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\glm5_result_delete_manifest_20261109.csv"
BLOBS = r"D:\AI2050\Ai2050-OpenOne\distributed_data\research_native_blobs"
OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result\cleanup_hardlink_exposure_20261109.txt"


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return ("%.2f %s" % (n, u)) if u != "B" else ("%d B" % n)
        n /= 1024.0


t0 = time.time()
stat = collections.defaultdict(lambda: {"solo": [0, 0], "linked": [0, 0], "gone": [0, 0]})
ext_stat = collections.defaultdict(lambda: {"solo": [0, 0], "linked": [0, 0]})
nlink_hist = collections.Counter()
total_rows = 0

with open(CSV, "r", encoding="utf-8", newline="") as cf:
    for row in csv.DictReader(cf):
        if row["batch"] != "P2-cold":
            continue
        total_rows += 1
        p = row["path"]
        s = int(row["size_bytes"])
        d = row["top_dir"]
        e = row["ext"]
        try:
            st = os.stat(p)
        except OSError:
            stat[d]["gone"][0] += s
            stat[d]["gone"][1] += 1
            continue
        n = getattr(st, "st_nlink", 1)
        nlink_hist[min(n, 8)] += 1
        key = "solo" if n <= 1 else "linked"
        stat[d][key][0] += s
        stat[d][key][1] += 1
        ext_stat[e][key][0] += s
        ext_stat[e][key][1] += 1

# blob store survey
bsolo = [0, 0]
blinked = [0, 0]
bnlink = collections.Counter()
for dp, dns, fns in os.walk(BLOBS):
    for fn in fns:
        p = os.path.join(dp, fn)
        try:
            st = os.stat(p)
        except OSError:
            continue
        n = getattr(st, "st_nlink", 1)
        bnlink[min(n, 8)] += 1
        if n <= 1:
            bsolo[0] += st.st_size
            bsolo[1] += 1
        else:
            blinked[0] += st.st_size
            blinked[1] += 1

L = []
w = L.append
w("HARDLINK EXPOSURE OF P2 CLEANUP  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("scan_sec=%.1f  P2 rows=%d" % (time.time() - t0, total_rows))
w("")
w("[1] OVERALL (still-existing P2 files)")
tsolo = sum(v["solo"][0] for v in stat.values())
tlink = sum(v["linked"][0] for v in stat.values())
tgone = sum(v["gone"][0] for v in stat.values())
fsolo = sum(v["solo"][1] for v in stat.values())
flink = sum(v["linked"][1] for v in stat.values())
w("    nlink=1  (deletion FREES space) : %10s / %7d files" % (human(tsolo), fsolo))
w("    nlink>=2 (only unlinks a view)  : %10s / %7d files" % (human(tlink), flink))
w("    already deleted before scan     : %10s" % human(tgone))
w("    releasable share = %.2f%% of remaining bytes" % (100.0 * tsolo / (tsolo + tlink) if tsolo + tlink else 0))
w("")
w("[2] BY EXTENSION")
for e, v in sorted(ext_stat.items(), key=lambda kv: -(kv[1]["solo"][0] + kv[1]["linked"][0])):
    a, b = v["solo"][0], v["linked"][0]
    w("    %-12s solo=%10s/%6d  linked=%10s/%7d" % (
        e, human(a), v["solo"][1], human(b), v["linked"][1]))
w("")
w("[3] BY TOP DIR (solo bytes desc; top 30)")
w("    cols: SOLO(frees) | files | LINKED(no free) | files | already-deleted | name")
for d, v in sorted(stat.items(), key=lambda kv: -kv[1]["solo"][0])[:30]:
    if v["solo"][0] == 0 and v["linked"][0] == 0:
        continue
    w("    %10s | %6d | %10s | %7d | %10s | %s" % (
        human(v["solo"][0]), v["solo"][1], human(v["linked"][0]), v["linked"][1],
        human(v["gone"][0]), d))
w("")
w("[4] nlink histogram of remaining P2 files")
for k in sorted(nlink_hist):
    w("    nlink=%d%s : %d files" % (k, "+" if k == 8 else "", nlink_hist[k]))
w("")
w("[5] CAS BLOB STORE: " + BLOBS)
w("    nlink=1  : %10s / %7d files" % (human(bsolo[0]), bsolo[1]))
w("    nlink>=2 : %10s / %7d files" % (human(blinked[0]), blinked[1]))
w("    blob nlink histogram: " + str(dict(sorted(bnlink.items()))))
w("")
w("=== END (READ-ONLY) ===")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("OK " + OUT)
