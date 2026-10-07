# -*- coding: utf-8 -*-
"""R8 冻结前快照：备份将改动的 deepseek 自有文件 + 记录写前 sha8 + dump memo 尾部。"""
import os, shutil, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
ARCH = os.path.join(ROOT, r"tests\deepseek_temp\_archive_r8")
OUT  = os.path.join(ROOT, r"tests\deepseek_temp\_r8_freeze.txt")
os.makedirs(ARCH, exist_ok=True)

TARGETS = [
    (r"research\deepseek\docs\AGI_DEEPSEEK_MEMO.md", "AGI_DEEPSEEK_MEMO.md"),
    (r"research\deepseek\atlas\phase_queue_v1.json", "phase_queue_v1.json"),
    (r"tests\deepseek\result\meta_single_source_v4.json", "meta_single_source_v4.json"),
]
# 只读受保护（跨线）——仅记录指纹，不备份不改
PROTECTED = [
    (r"research\gpt5\docs\AGI_GPT5_MEMO.md", "AGI_GPT5_MEMO.md"),
    (r"research\gpt5\atlas\atlas_ledger.json", "atlas_ledger.json"),
    (r"tests\glm5\result\rdc_query_construction_20260913\phase3103\omega_p101_formula_audit\proposition_ledger.json", "proposition_ledger.json"),
]

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

r = []
def add(s): r.append(s)

add("=== 备份（写前） ===")
for rel, name in TARGETS:
    src = os.path.join(ROOT, rel)
    dst = os.path.join(ARCH, name)
    s = sha(src)
    shutil.copy2(src, dst)
    s2 = sha(dst)
    ok = (s == s2)
    add("  %-8s %s  %d B  %s -> backup %s" % ("OK" if ok else "FAIL", name, os.path.getsize(src), s[:8], s2[:8]))
    assert ok, "backup mismatch %s" % name

add("")
add("=== 受保护（跨线，只读）指纹 ===")
for rel, name in PROTECTED:
    p = os.path.join(ROOT, rel)
    add("  %-24s exists=%s  %d B  %s" % (name, os.path.exists(p),
        os.path.getsize(p) if os.path.exists(p) else -1, sha(p)[:8] if os.path.exists(p) else "MISSING"))

add("")
add("=== memo 尾部 8 行（判分隔格式） ===")
memo = os.path.join(ROOT, TARGETS[0][0])
raw = open(memo, "rb").read()
txt = raw.decode("utf-8-sig")
ls = txt.replace("\r\n", "\n").split("\n")
for i, l in enumerate(ls[-8:], len(ls) - 7):
    add("  %5d|%r" % (i, l[:150]))

open(OUT, "w", encoding="utf-8").write("\n".join(r))
print("\n".join(r))
