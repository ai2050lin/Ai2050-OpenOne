import os, sys

BASE = r"D:\AI2050\Ai2050-OpenOne"
out = []

# 1. check phase result dirs
res_dir = os.path.join(BASE, "tests", "glm5", "result", "rdc_query_construction_20260913")
if os.path.isdir(res_dir):
    entries = sorted(os.listdir(res_dir))
    out.append("[rdc_query_construction_20260913] " + str(len(entries)) + " entries:")
    for e in entries:
        p = os.path.join(res_dir, e)
        if os.path.isdir(p):
            try:
                n = len(os.listdir(p))
            except Exception as ex:
                n = "ERR:" + str(ex)
            out.append("  [DIR] " + e + " (" + str(n) + " items)")
        else:
            out.append("  [FILE] " + e)
else:
    out.append("MISSING: " + res_dir)

# 2. check memory dir
mem_dir = os.path.join(BASE, ".workbuddy", "memory")
out.append("")
out.append("[.workbuddy/memory] " + str(sorted(os.listdir(mem_dir))))

# 3. check ledger candidates
for cand in [
    r"research\glm5\docs",
    r"research\gpt5\docs",
    r"tests\glm5\result\rdc_query_construction_20260913\phase3061",
    r"tests\glm5\result\rdc_query_construction_20260913\phase3062",
]:
    p = os.path.join(BASE, cand)
    if os.path.isdir(p):
        out.append("[DIR] " + cand + " -> " + str(sorted(os.listdir(p))[:40]))
    else:
        out.append("MISSING: " + cand)

# 4. MEMO files existence + size
for memo in [
    r"research\gpt5\docs\AGI_GPT5_MEMO.md",
    r"research\glm5\docs\AGI_GLM5_MEMO.md",
]:
    p = os.path.join(BASE, memo)
    if os.path.isfile(p):
        out.append("[FILE] " + memo + " size=" + str(os.path.getsize(p)))
    else:
        out.append("MISSING: " + memo)

# 5. recent phase scripts in tests/glm5
scr = os.path.join(BASE, "tests", "glm5")
scripts = sorted([f for f in os.listdir(scr) if f.startswith("phase30")], reverse=True)
out.append("")
out.append("[tests/glm5 phase30* scripts] " + str(scripts[:30]))

with open(r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\probe_3063_env.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("WROTE_OK")
