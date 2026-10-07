import os, json

WS = r"D:\AI2050\Ai2050-OpenOne"
out = []

def exists(rel):
    return os.path.isdir(os.path.join(WS, rel))

out.append("tests/glm5 exists: %s" % exists("tests/glm5"))
if exists("tests/glm5"):
    out.append("tests/glm5 entries: %s" % sorted(os.listdir(os.path.join(WS, "tests/glm5"))))
    res = os.path.join(WS, "tests/glm5/result")
    if os.path.isdir(res):
        entries = sorted(os.listdir(res))
        out.append("tests/glm5/result count=%d first20=%s" % (len(entries), entries[:20]))

hits = []
for base in ("tests", "research", "ai2050_research_os"):
    root = os.path.join(WS, base)
    if not os.path.isdir(root):
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            if d.startswith(("phase1246", "phase1247", "phase1248", "phase1249", "phase1250", "phase1251", "phase1252", "phase1253", "phase1254", "phase1255", "phase1256", "phase1257", "phase1258", "phase1259", "phase1260", "phase1261", "phase1262", "phase1263")):
                hits.append(os.path.relpath(os.path.join(dirpath, d), WS))
out.append("phase1246-1263 dir hits (%d): %s" % (len(hits), hits[:40]))

out.append("research top: %s" % sorted(os.listdir(os.path.join(WS, "research"))))
out.append("AGI_GLM5 at glm5/docs: %s" % os.path.isfile(os.path.join(WS, "research", "glm5", "docs", "AGI_GLM5_MEMO.md")))
out.append("AGI_GLM5 at gpt5/docs: %s" % os.path.isfile(os.path.join(WS, "research", "gpt5", "docs", "AGI_GLM5_MEMO.md")))
out.append("AGI_GPT5 at gpt5/docs: %s" % os.path.isfile(os.path.join(WS, "research", "gpt5", "docs", "AGI_GPT5_MEMO.md")))

# check one specific missing manifest target to see pattern
probe = os.path.join(WS, "tests", "glm5", "result", "phase1246_c001_wp01_typed_behavior_qualification")
out.append("probe dir exists: %s" % os.path.isdir(probe))
if os.path.isdir(probe):
    for dp, dn, fn in os.walk(probe):
        out.append("  %s -> %s" % (os.path.relpath(dp, probe), sorted(fn)))

res = os.path.join(WS, ".workbuddy", "tmp_probe_paths.txt")
with open(res, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("WROTE", res)
