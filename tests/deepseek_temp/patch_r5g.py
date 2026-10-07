# -*- coding: utf-8 -*-
"""R5g: 补写三处落盘（wlog / skill 教训 38+39 / MEMORY 计数），逐处独立复核。
承教训 39：一次脚本多处写入，必须对每一处做写前 count + 写后回读，并逐处打印指纹。
"""
import os, hashlib

MEM = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md"
WLOG = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-03.md"
SKILL = r"C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md"
REP = r"D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_review\_patch_r5g_report.txt"

report = []
def log(s):
    report.append(s)

def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]

def load(p):
    raw = open(p, "rb").read()
    bom = raw.startswith(b"\xef\xbb\xbf")
    txt = raw.decode("utf-8-sig")
    return raw, bom, txt, ("\r\n" in txt)

def save(p, txt, bom, crlf):
    nl = "\r\n" if crlf else "\n"
    payload = txt.replace("\n", nl).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    open(p, "wb").write(payload)
    return open(p, "rb").read()

# ---------------------------------------------------------------- 1) WLOG
raw, bom, txt, crlf = load(WLOG)
before = (len(raw), sha8(raw))
R5_BLOCK = "\n".join([
    "## R5：A 闸门 Q09（死线双轨重述）+ Q12（D/E 命题引用审计），零 GPU",
    "- **Q09 死线双轨（I3）**：三条死线（`RDC_TESTPLAN_v1.md` §8.2）触发条件**全为对模型/族数的全称量词合取**；**K2/K3 从未被测量**（`phase3154*` 目录 0 命中；全库无「top-50 覆盖率」量）；K2 在同一文件内还有**两种互斥操作化**（§8.2「响应 cos 降>50%」vs 3153 预注册「交互份额>50%」）。",
    "  - **K1 逐模型现场重算**（3151/3152 result）：qwen3-4b `above_add_gate=False`（3 对 margin 全负、MDE ~1.7e-3）、qwen3-14b `True`、glm4-9b `False`。",
    "  - **聚合**：池化 margin@k* **−0.025202**（MDE 0.005696，轨 A 通过）；池化 margin@readout **+0.940920**（MDE 0.193309）；E_read 池化 **0.373350**；模型级 bootstrap CI（n=3）20000 次 / seed 20261003。",
    "  - **判决**：k* 层 = **`model_specific`**（轨 A 通过、轨 B 由 qwen3-4b 否决）；读出层 = **`fired_all_models`**（3/3 否决 + 轨 A 触发）⇒ **与旧结论「未触发、算子代数线保住」相反**。层位选择属 Q08（待 seal）。",
    "  - 产物 `research/gpt5/atlas/deadline_dual_track_v1.json`（`4d1853d3`）+ `research/gpt5/docs/DEADLINE_DUAL_TRACK_Q09.md`（`0d8633c7`）。",
    "- **Q12 D/E 引用审计（I4）**：账本驱动 + 命名空间感知 + 文本指纹级。新推理链 = MEMO 自 Phase 3104（L13244）起 ∪ `docs/*.md`（26 个，排除归档快照与自身输出）。",
    "  - **F1 id 级**：命中**仅 1 处**（MEMO L13546 / Phase 3113 / R55，复合等级）；**F3 文本指纹**（claim 6-CJK-gram）**实质命中 0** ⇒ **无实质违规**。",
    "  - **制度缺陷**：**RULE-UNDECIDABLE**（D∪E 38/62=61.29%，其中复合 26、纯 E 仅 3 ⇒「E 级禁入」对 26/38 条不可机械判定）；**NS-COLLISION**（账本 `R\\d\\d` 与 `LOOP_DIAGNOSIS` 自带 §2 R1–R11 冲突 ⇒ 纯 id grep 必假阳性）。",
    "  - 工具 `tests/deepseek/_review/prop_citation_audit.py`；产物 `atlas/prop_citation_audit_v1.json`（`4c2ea9d2`）+ `docs/PROP_CITATION_AUDIT_Q12.md`（`32933855`）。",
    "- **独立复核**：`disk_verify_q09q12_r5.py` → **PASS 76 / FAIL 0 ALL_PASS**（从 result 独立重算池化量与 CI、从账本独立重算等级与命中、受保护文件指纹、LF-only）。",
    "- **汇报页** `tests/deepseek_temp/_review/q09q12_gate_r5.html`（`ee439e4c`）。",
    "- **未改动**：MEMO 原文、TESTPLAN、Ledger、队列、宪法、3151/3152/3153 产物一律未触碰；K1 改判为待 seal 的 Q08。",
    "- **技能** `rdc-phase-closeout` 新增**教训 38**（禁入类规则的可机械判定性）、**教训 39**（一次脚本多处写入必须逐处独立复核——本轮实测 wlog/skill 两处为幻影写入）；MEMORY.md 压缩重写（5606→5130）。",
    "",
])
assert txt.count("## R5：A 闸门 Q09") == 0, "WLOG 已存在 R5 块，勿重复追加"
new_txt = txt.rstrip("\n") + "\n" + R5_BLOCK
raw2 = save(WLOG, new_txt, bom, crlf)
back = open(WLOG, "rb").read().decode("utf-8-sig")
log("[WLOG] %s" % WLOG)
log("   before: %d B / %s" % (before[0], before[1]))
log("   after : %d B / %s" % (len(raw2), sha8(raw2)))
log("   readback_has_R5 = %s" % ("## R5：A 闸门 Q09" in back))
log("   readback_has_fired_all_models = %s" % ("fired_all_models" in back))
log("   readback_has_76/0 = %s" % ("76 / FAIL 0" in back))

# ---------------------------------------------------------------- 2) SKILL
raw, bom, txt, crlf = load(SKILL)
before = (len(raw), sha8(raw))
ANCHOR = "## 参照实现（Phase 3125，210/210 全绿；Phase 3128，19/19 全绿）"
assert txt.count(ANCHOR) == 1
assert "38. **" not in txt and "39. **" not in txt, "SKILL 已含 38/39"

L38 = "\n".join([
    "38. **元层「禁入」类规则必须先验证它是否可机械判定（Q12 实证，2026-10-03）**：",
    "",
    "```text",
    "(a) 分级口径先拆「分量 vs 条数」。账本 62 条的 grade 是复合标签（16 种，如 B+D / E+A）；",
    "    「D∪E 占 61%」按条数成立（38/62），而 MEMO 的 34%+32% 是分量和（和 94）——两者量纲",
    "    不同，且复合占 26/38 ⇒ 一条记录可同时携带「可引用」与「已撤回」分量。",
    "(b) 「E 级命题禁止进入新推理链」这类禁令，对复合等级**不可机械判定**：撤回的是 E 分量，",
    "    A/B/C 分量仍在。规则文本必须写明「按分量撤回」还是「整条撤回」，否则审计无解。",
    "(c) 账本 id 不是项目内唯一命名空间。同项目另有 LOOP_DIAGNOSIS 自定义 R1–R11（其 R10 =",
    "    「翻译轴不存在」），与账本 R10（紧凑性定律）毫无关系 ⇒ **纯 id grep 必假阳性**。",
    "    审计必须叠加「文本指纹」（claim 的 6-CJK-gram）或人工核定表。",
    "(d) 审计语料必须先排除「自指」：审计报告若写入被扫目录，第二次运行会把自身当语料",
    "    （实测 id 命中 24 -> 362 虚增）；月快照（AGI_GPT5_MEMO_2026*.md）会被双算，须显式排除。",
    "(e) 结论方向要为「阴性」留出口：新推理链 id 命中仅 1 处、文本指纹 0 命中 ⇒ 无实质违规；",
    "    真正的产出是「暴露了两项制度缺陷」，而不是找到违规。不得为凑结论编造命中。",
    "```",
])
L39 = "\n".join([
    "39. **一次脚本多处写入 ⇒ 必须逐处独立复核（R5 实证，2026-10-03）**：",
    "    同一 `patch_*.py` 内连续写 MEMORY.md / wlog / skill 三处，**只有第一处落了盘**；脚本自报的",
    "    「wlog 追加成功」「技能教训追加成功」是**幻影**——次日独立探针发现 wlog（`d4540e4b`）与",
    "    `SKILL.md`（`afe6215c`）**逐字节等于写入前**。**对策**：① 写盘脚本对**每一处**做「写前 count /",
    "    写后回读 =」双断言，报告逐处打印 `old_bytes -> new_bytes + sha8`；② 收尾复核**不以脚本日志为证**，",
    "    用**独立探针**分别 re-hash 每个目标文件；③ 承教训 34(c)：`Read` 视图可能陈旧 ⇒ 一律 Python/Grep 判定；",
    "    ④ 承教训 21(a)：幻影写入的根因是**匹配串凭记忆写**，一律先 dump 真实磁盘再构造。",
    "",
])
new_txt = txt.replace(ANCHOR, L38 + "\n\n" + L39 + "\n" + ANCHOR, 1)
raw2 = save(SKILL, new_txt, bom, crlf)
back = open(SKILL, "rb").read().decode("utf-8-sig")
log("")
log("[SKILL] %s" % SKILL)
log("   before: %d B / %s" % (before[0], before[1]))
log("   after : %d B / %s" % (len(raw2), sha8(raw2)))
log("   readback_has_38 = %s" % ("38. **元层「禁入」" in back))
log("   readback_has_39 = %s" % ("39. **一次脚本多处写入" in back))
log("   readback_has_RULE-UNDECIDABLE = %s" % ("不可机械判定" in back))
log("   readback_has_NS = %s" % ("纯 id grep 必假阳性" in back))

# ---------------------------------------------------------------- 3) MEMORY 计数
raw, bom, txt, crlf = load(MEM)
before = (len(raw), sha8(raw))
REPS = [
    ("见 MEMO 与技能教训 26–38。", "见 MEMO 与技能教训 26–39。"),
    ("在 skill `rdc-phase-closeout` 教训 26–38（P21：", "在 skill `rdc-phase-closeout` 教训 26–39（P21："),
    ("`rdc-phase-closeout`（**38 教训**/十四次链）", "`rdc-phase-closeout`（**39 教训**/十四次链）"),
]
for old, new in REPS:
    c = txt.count(old)
    assert c == 1, "MEMORY count=%d for %r" % (c, old[:40])
    txt = txt.replace(old, new)
assert len(txt) < 5300, "MEMORY 过长 %d" % len(txt)
raw2 = save(MEM, txt, bom, crlf)
back = open(MEM, "rb").read().decode("utf-8-sig")
log("")
log("[MEMORY] %s" % MEM)
log("   before: %d B / %s" % (before[0], before[1]))
log("   after : %d B / %s  chars=%d" % (len(raw2), sha8(raw2), len(back)))
for old, new in REPS:
    log("   readback %s = %s" % (new[:22], new in back))

log("")
log("ALL_SITES_DONE")
open(REP, "w", encoding="utf-8").write("\n".join(report) + "\n")
print("\n".join(report))
