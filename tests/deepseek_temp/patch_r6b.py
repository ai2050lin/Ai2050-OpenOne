# -*- coding: utf-8 -*-
"""R6b: 登记「记录落点」规范变更 v3（用户 2026-10-03 指令）+ 技能补一条陷阱。逐处独立复核。"""
import os, hashlib

MEM = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md"
SKILL = r"C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md"
REP = r"D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_review\_patch_r6b_report.txt"

rep = []
def log(s): rep.append(s)
def sha8(b): return hashlib.sha256(b).hexdigest()[:8]

def load(p):
    raw = open(p, "rb").read()
    return raw, raw.startswith(b"\xef\xbb\xbf"), raw.decode("utf-8-sig"), (b"\r\n" in raw)

def save(p, txt, bom, crlf):
    b = (txt.replace("\n", "\r\n") if crlf else txt).encode("utf-8")
    if bom: b = b"\xef\xbb\xbf" + b
    open(p, "wb").write(b)
    return open(p, "rb").read()

# ---------------- 1) MEMORY ----------------
raw, bom, txt, crlf = load(MEM)
before = (len(raw), sha8(raw), len(txt))
REPS = [
 ("> 权威记录：`research\\deepseek\\docs\\AGI_DEEPSEEK_MEMO.md`（N 线）、`research\\gpt5\\docs\\AGI_GPT5_MEMO.md`（G 线）。本文件仅跨轮索引，细节一律回查 MEMO / 技能。",
  "> 权威记录：`research\\deepseek\\docs\\AGI_DEEPSEEK_MEMO.md`（**唯一研究日志**）。本文件仅跨轮索引，细节一律回查该 MEMO / 技能。"),
 ("- deepseek 线只写 deepseek 备忘录（append-only、UTF-8+**BOM**+**CRLF**、`bare_lf 0`）。",
  "- **所有研究日志只写** `AGI_DEEPSEEK_MEMO.md`（append-only、BOM+CRLF、`bare_lf 0`）；**不再新建其他 `.md`**（原 N/G 双备忘录制**废止**，用户 2026-10-03 指令）。该文件基线 `4ba5e22f`(4774 行) → R6 追加后 **`d25350d9`**(4832 行)。"),
]
for old, new in REPS:
    c = txt.count(old)
    assert c == 1, "MEMORY count=%d for %r" % (c, old[:40])
    txt = txt.replace(old, new)
assert len(txt) < 5300, "MEMORY 过长 %d" % len(txt)
raw2 = save(MEM, txt, bom, crlf)
back = open(MEM, "rb").read().decode("utf-8-sig")
log("[MEMORY] %d B/%s chars=%d -> %d B/%s chars=%d" % (before[0], before[1], before[2], len(raw2), sha8(raw2), len(back)))
for old, new in REPS:
    log("   readback(%s) = %s" % (new[:26].replace("\n", " "), new in back))

# ---------------- 2) SKILL：39 补 ⑤ ----------------
raw, bom, txt, crlf = load(SKILL)
before = (len(raw), sha8(raw))
ANCH = "    ④ 承教训 21(a)：幻影写入的根因是**匹配串凭记忆写**，一律先 dump 真实磁盘再构造。"
NEW = ANCH + "\n    ⑤ **写盘脚本自身要先审查编码逻辑**：`old.replace(\"\\r\\n\",\"\\n\") -> 拼接 -> .replace(\"\\n\",\"\\r\\n\")` 这种「先归一再转换」的顺序**必须显式归一**，否则对已是 CRLF 的正文再 `replace(\"\\n\",\"\\r\\n\")` 会产生 `\\r\\r\\n`（本轮实测在运行前审查拦下）。"
assert txt.count(ANCH) == 1, "SKILL anchor count=%d" % txt.count(ANCH)
txt = txt.replace(ANCH, NEW, 1)
raw2 = save(SKILL, txt, bom, crlf)
back = open(SKILL, "rb").read().decode("utf-8-sig")
log("[SKILL] %d B/%s -> %d B/%s" % (before[0], before[1], len(raw2), sha8(raw2)))
log("   readback_has_39_5 = %s" % ("先归一再转换" in back))

log("ALL_SITES_DONE")
open(REP, "w", encoding="utf-8").write("\n".join(rep) + "\n")
print("\n".join(rep))
