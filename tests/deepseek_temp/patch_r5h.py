# -*- coding: utf-8 -*-
"""R5h: 记录 seal 请求包 + 幻影写入纠错（wlog 追加 / MEMORY 计数），逐处独立复核。"""
import os, hashlib

WLOG = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-03.md"
MEM = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md"
REP = r"D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_review\_patch_r5h_report.txt"

rep = []
def log(s): rep.append(s)
def sha8(b): return hashlib.sha256(b).hexdigest()[:8]

def load(p):
    raw = open(p, "rb").read()
    return raw, raw.startswith(b"\xef\xbb\xbf"), raw.decode("utf-8-sig"), (b"\r\n" in raw)

def save(p, txt, bom, crlf):
    nl = "\r\n" if crlf else "\n"
    b = txt.replace("\n", nl).encode("utf-8")
    if bom: b = b"\xef\xbb\xbf" + b
    open(p, "wb").write(b)
    return open(p, "rb").read()

# ---- 1) wlog ----
raw, bom, txt, crlf = load(WLOG)
before = (len(raw), sha8(raw))
BLOCK = "\n".join([
    "## R5b：A 闸门 seal 请求包（只读汇总，零 GPU）",
    "- 零 GPU 部分（Q01/Q02/Q09/Q12）已全部完成；剩余全部是 **seal 决策**，不是待做实验。",
    "- 三件产物：`research/gpt5/atlas/seal_request_v1.json`（`d871a4b4`）、`research/gpt5/docs/SEAL_REQUEST_A_GATE.md`（`7678cbe0`）、`tests/deepseek_temp/_review/seal_request_a_gate.html`（`3976537e`）。数字全部现场渲染（K1 逐模型 + 聚合取自 `deadline_dual_track_v1.json`，C1–C6 逐条取自 `META_SINGLE_SOURCE_Q01.md` §5 表，指纹现场重算）。",
    "- 内容：Q08（K1 判定层；读法甲=改判·推荐 / 乙=重述并显式解冻升版 / 暂缓）+ C1–C6 更正表 + I1/I9 冻结确认 + 五枚受保护文件指纹 + seal 后执行顺序 + 一行回复模板 `seal: Q08=甲 | C=全接受 | I1=确认 | I9=确认`。",
    "- **幻影写入纠错（本轮回溯）**：上一轮 patch 自报的「wlog 追加成功」「技能教训 38 追加成功」实为**幻影**——独立探针发现 wlog（`d4540e4b`）与 `SKILL.md`（`afe6215c`）与写入前**逐字节相同**。本轮以 `patch_r5g.py` 补写并**逐处独立复核**：wlog `66e681ab`（2907→5694 B）、`SKILL.md` `66704c89`（65,880→68,234 B；含**教训 38** 禁入类规则可机械判定性、**教训 39** 一次脚本多处写入必须逐处独立复核）、`MEMORY.md` `ed3319bd`（5,606→5,130 字符，20/20 锚点）。",
    "- **未改动**：MEMO / TESTPLAN / Ledger / 队列 / 宪法 / 3151–3153 产物一律未触碰（指纹现场复核全 OK：`2a84776b` / `9a3c6ff4` / `71b85673` / `675836fd` / `01df6398`）。",
    "",
])
assert txt.count("## R5b：A 闸门 seal 请求包") == 0
txt = txt.rstrip("\n") + "\n" + BLOCK
raw2 = save(WLOG, txt, bom, crlf)
back = open(WLOG, "rb").read().decode("utf-8-sig")
log("[WLOG] %d B/%s -> %d B/%s" % (before[0], before[1], len(raw2), sha8(raw2)))
log("   has_R5b=%s has_phantom=%s" % ("## R5b：A 闸门 seal 请求包" in back, "幻影写入纠错" in back))

# ---- 2) MEMORY：指向 seal 汇总件 ----
raw, bom, txt, crlf = load(MEM)
before = (len(raw), sha8(raw), len(txt))
OLD = "- **⚠ 待用户 seal**：Q01 更正表 **C1–C6**；Q08（K1 改判，两种读法已备）；I1/I9。"
NEW = "- **⚠ 待用户 seal**（汇总件 `SEAL_REQUEST_A_GATE.md`）：Q01 更正表 **C1–C6**；Q08（K1 改判，两种读法已备）；I1/I9。"
assert txt.count(OLD) == 1, "MEMORY count=%d" % txt.count(OLD)
txt = txt.replace(OLD, NEW)
assert len(txt) < 5300, "MEMORY 过长 %d" % len(txt)
raw2 = save(MEM, txt, bom, crlf)
back = open(MEM, "rb").read().decode("utf-8-sig")
log("[MEMORY] %d B/%s chars=%d -> %d B/%s chars=%d" % (before[0], before[1], before[2], len(raw2), sha8(raw2), len(back)))
log("   has_SEAL_REQUEST_A_GATE=%s" % ("SEAL_REQUEST_A_GATE.md" in back))

log("ALL_SITES_DONE")
open(REP, "w", encoding="utf-8").write("\n".join(rep) + "\n")
print("\n".join(rep))
