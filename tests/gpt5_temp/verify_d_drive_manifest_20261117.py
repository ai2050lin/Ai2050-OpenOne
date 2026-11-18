# -*- coding: utf-8 -*-
"""READ-ONLY guard: ensure A-tier delete manifest never touches served files."""
import os
import csv
import json

OUT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5\result"
MANIFEST = os.path.join(OUT, "d_drive_delete_manifest_v2_20261117.csv")
REPORT = os.path.join(OUT, "d_drive_manifest_guard_20261117.txt")
RESULT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"

rows = list(csv.DictReader(open(MANIFEST, "r", encoding="utf-8", newline="")))
o = ["A-TIER DELETE MANIFEST GUARD / READ-ONLY  2026-11-17"]
o.append("total rows = %d" % len(rows))
o.append("")
o.append("[1] A3-PACKS per-file (should be only root-level archives)")
a3 = [r for r in rows if r["tier"] == "A3-PACKS"]
for r in a3:
    o.append("  %-26s %12s B  %s" % (r["label"], r["size_bytes"], r["path"]))
o.append("  A3 count = %d / %s bytes" % (len(a3), sum(int(r["size_bytes"]) for r in a3)))
o.append("")

o.append("[2] manifest-registered files in served dirs")
served = [
    os.path.join(RESULT, "goodfire_middle_20261113_v2", "qwen4"),
    os.path.join(RESULT, "goodfire_middle_20261113_v2", "qwen14"),
    os.path.join(RESULT, "qwen14_client_v2", "native"),
    os.path.join(RESULT, "qwen14_mlp_atlas_v1"),
]
protect = set()
for p in served:
    mf = os.path.join(p, "manifest.json")
    if not os.path.isfile(mf):
        o.append("  MISSING manifest: " + p)
        continue
    d = json.loads(open(mf, encoding="utf-8").read())
    fl = d.get("files", [])
    o.append("  %-58s registered=%d  %.2f GB" %
             (os.path.relpath(p, RESULT), len(fl),
              sum(int(x.get("bytes", 0)) for x in fl) / 1024 ** 3))
    for x in fl:
        protect.add(os.path.normcase(os.path.abspath(os.path.join(p, x.get("path", "")))))
    protect.add(os.path.normcase(os.path.abspath(mf)))
o.append("  protected absolute paths = %d" % len(protect))
o.append("")

o.append("[3] GUARD: delete-manifest rows that hit protected files")
hit = [r for r in rows if os.path.normcase(os.path.abspath(r["path"])) in protect]
o.append("  VIOLATIONS = %d  (MUST be 0)" % len(hit))
for r in hit[:10]:
    o.append("    !! " + r["path"])
o.append("")

o.append("[4] GUARD: rows under the 4 served dirs (exact absolute prefix)")
served_prefix = [os.path.normcase(os.path.abspath(p) + os.sep) for p in served]
sub = [r for r in rows
       if any(os.path.normcase(os.path.abspath(r["path"])).startswith(pre)
              for pre in served_prefix)]
o.append("  rows = %d  (MUST be 0)" % len(sub))
for r in sub[:10]:
    o.append("    !! " + r["tier"] + " " + r["path"])
o.append("")

o.append("[5] GUARD: rows under visualization-research (junction entity source)")
vr = [r for r in rows if "visualization-research" in r["path"]]
o.append("  rows = %d  (MUST be 0 - entity source must stay intact)" % len(vr))
for r in vr[:10]:
    o.append("    !! " + r["path"])
o.append("")

o.append("[6] GUARD: rows that still exist & are conclusion files")
keep_ext = (".json", ".jsonl", ".md", ".log", ".py", ".sha256", ".csv", ".tsv")
bad_ext = [r for r in rows if r["path"].lower().endswith(keep_ext)]
o.append("  conclusion-like rows = %d  (MUST be 0)" % len(bad_ext))
for r in bad_ext[:10]:
    o.append("    !! " + r["path"])
o.append("")

o.append("[7] tier totals")
from collections import defaultdict
agg = defaultdict(lambda: [0, 0])
for r in rows:
    agg[r["tier"]][0] += int(r["size_bytes"])
    agg[r["tier"]][1] += 1
for t in sorted(agg):
    o.append("  %-18s %12.2f GB  %6d files" % (t, agg[t][0] / 1024 ** 3, agg[t][1]))
o.append("")
o.append("===== END GUARD =====")
with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(o) + "\n")
print("GUARD violations=%d vr_rows=%d concl_rows=%d" % (len(hit), len(vr), len(bad_ext)))
