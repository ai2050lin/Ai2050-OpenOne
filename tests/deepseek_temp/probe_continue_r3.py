import os, hashlib, json, re
root = r"D:\AI2050\Ai2050-OpenOne"
out = []

def fp(rel):
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        return [rel, "MISSING"]
    b = open(p, "rb").read()
    try:
        t = b.decode("utf-8-sig")
    except Exception:
        t = ""
    h2 = len([l for l in t.split("\n") if l.startswith("## ")])
    return [rel, len(b), b.count(b"\n") + 1, hashlib.sha256(b).hexdigest()[:8],
            bool(b[:3] == b"\xef\xbb\xbf"), h2]

out.append("=== [A] 文件指纹 ===")
for rel in [
    ".workbuddy/memory/MEMORY.md",
    ".workbuddy/memory/2026-10-02.md",
    ".workbuddy/memory/2026-10-03.md",
    "research/gpt5/docs/LOOP_DIAGNOSIS_AND_EXIT_v1.md",
    "tests/deepseek_temp/_review/loop_stats_r3.json",
    "tests/deepseek_temp/_review/loop_diagnosis_r3.html",
    "tests/deepseek/_review/gen_loop_diagnosis_r3.py",
    "tests/deepseek/_review/gen_loop_html_r3.py",
    "research/gpt5/atlas/atlas_ledger.json",
    "research/gpt5/atlas/phase_queue_v1.json",
    "research/gpt5/docs/RDC_RESEARCH_CONSTITUTION_v1.md",
]:
    out.append(str(fp(rel)))

out.append("")
out.append("=== [B] loop_stats_r3.json 顶层键 ===")
sp = os.path.join(root, "tests/deepseek_temp/_review/loop_stats_r3.json")
try:
    d = json.loads(open(sp, "rb").read().decode("utf-8-sig"))
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                out.append("  %-24s dict keys=%s" % (k, list(v.keys())[:14]))
            elif isinstance(v, list):
                out.append("  %-24s list len=%d  head=%s" % (k, len(v), str(v[:2])[:120]))
            else:
                out.append("  %-24s %s" % (k, str(v)[:120]))
    else:
        out.append("  type=" + str(type(d)))
except Exception as e:
    out.append("  ERR " + str(e))

out.append("")
out.append("=== [C] MEMORY.md 标题 ===")
mp = os.path.join(root, ".workbuddy/memory/MEMORY.md")
try:
    mt = open(mp, "rb").read().decode("utf-8")
    out.append("  chars=%d" % len(mt))
    for i, l in enumerate(mt.split("\n"), 1):
        if l.startswith("#"):
            out.append("  %4d %s" % (i, l[:110]))
except Exception as e:
    out.append("  ERR " + str(e))

out.append("")
out.append("=== [D] gpt5 MEMO 关键词（E_ar / 自回归 / 多步 / held-out）===")
gp = os.path.join(root, "research/gpt5/docs/AGI_GPT5_MEMO.md")
try:
    gt = open(gp, "rb").read().decode("utf-8-sig")
    for k in ["E_ar", "自回归", "多步", "held-out", "heldout", "外推", "E_read", "C_steer",
              "高秩散布", "未识别", "独立因子"]:
        out.append("  %-12s %d" % (k, gt.count(k)))
except Exception as e:
    out.append("  ERR " + str(e))

out.append("")
out.append("=== [E] _review 目录清单 ===")
for d2 in ["tests/deepseek/_review", "tests/deepseek_temp/_review"]:
    p = os.path.join(root, d2)
    if os.path.isdir(p):
        fs = sorted(os.listdir(p))
        out.append("  %s (%d):" % (d2, len(fs)))
        for f in fs:
            out.append("      " + f)
    else:
        out.append("  %s MISSING" % d2)

open(os.path.join(root, "tests/deepseek_temp/_review/probe_continue_r3.txt"), "w", encoding="utf-8").write("\n".join(out))
print("OK", len(out), "lines")
