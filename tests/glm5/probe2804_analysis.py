# -*- coding: utf-8 -*-
import json
from pathlib import Path
BASE = Path(r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase2804\qwen4_polysemy_k3")
r = json.loads((BASE / "result.json").read_text(encoding="utf-8"))
lines = []
lines.append("== entity-sense check (country/company PR = word min?) ==")
for rec in r["records"]:
    ent = [s for s in rec["senses"] if s in ("country", "company")]
    if ent:
        prs = dict(zip(rec["senses"], rec["pr"]))
        for s in ent:
            lines.append("%-8s %s pr=%s min_sense=%s entity_is_min=%s"
                         % (rec["word"], s, prs[s],
                            min(prs, key=prs.get), prs[s] == min(prs.values())))
lines.append("")
lines.append("== named-natural tighter pairs breakdown ==")
tight = [p for p in r["named_natural_pairs"] if p["named_tighter"]]
loose = [p for p in r["named_natural_pairs"] if not p["named_tighter"]]
lines.append("TIGHTER (%d):" % len(tight))
for p in tight:
    lines.append("  %-8s %s(%s) < %s(%s)" % (p["word"], p["named"], p["pr_named"], p["natural"], p["pr_natural"]))
lines.append("LOOSER (%d):" % len(loose))
for p in loose:
    lines.append("  %-8s %s(%s) >= %s(%s)" % (p["word"], p["named"], p["pr_named"], p["natural"], p["pr_natural"]))
lines.append("")
lines.append("== sample profiles ==")
for rec in r["records"][:4]:
    for ph in rec["profiles"]:
        lines.append("%-8s %-8s hits=%s" % (rec["word"], ph["sense"], ph["hits"]))
lines.append("")
lines.append("== flatness range ==")
fl = [rec["flatness"] for rec in r["records"]]
lines.append("min=%.2f max=%.2f" % (min(fl), max(fl)))
p = Path(r"D:\AI2050\Ai2050-OpenOne\tests\glm5\probe2804_analysis.txt")
p.write_text("\n".join(lines), encoding="utf-8")
print("WROTE", p)
