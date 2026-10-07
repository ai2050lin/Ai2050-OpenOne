# -*- coding: utf-8 -*-
"""R8：更新 MEMORY.md / wlog / 技能（锚点 + assert count==1 + 回读）。"""
import os, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT  = os.path.join(ROOT, r"tests\deepseek_temp\_patch_r8_report.txt")
MEM  = os.path.join(ROOT, r".workbuddy\memory\MEMORY.md")
WLOG = os.path.join(ROOT, r".workbuddy\memory\2026-10-03.md")
SKILL= r"C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md"
MEMO = os.path.join(ROOT, r"research\deepseek\docs\AGI_DEEPSEEK_MEMO.md")

def load(p):
    b = open(p, "rb").read()
    return b, b.startswith(b"\xef\xbb\xbf"), b.decode("utf-8-sig")
def save(p, text, bom):
    open(p, "wb").write((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
def sha8(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

rep = []
def add(s): rep.append(s)

mb, mbom, mt = load(MEMO)
memo_lines = mt.count("\n") + 1
memo_b = len(mb)

# ---------------- MEMORY.md ----------------
b0, bom, t = load(MEM)
add("MEMORY 前: %d B  %s" % (len(b0), hashlib.sha256(b0).hexdigest()[:8]))
reps = [
 # L2
 ("> 权威记录 `research\\deepseek\\docs\\AGI_DEEPSEEK_MEMO.md`（**唯一研究日志**，Phase 1–34）。",
  "> 权威记录 `research\\deepseek\\docs\\AGI_DEEPSEEK_MEMO.md`（**唯一研究日志**，Phase 1–35）。"),
 # L7
 ("**P32–34 = 归属整理并入（N1/E1/MEMO 审计）**。现 `b257f54f`（676,086 B / 6902 行）。",
  "**P32–34 = 归属整理并入（N1/E1/MEMO 审计）**；**P35 = A 闸门 seal 执行/关闭（R8）**。现 `150241da`（%d B / %d 行）。" % (memo_b, memo_lines)),
 # L30-31
 ("- **⚠ 待 seal**（`tests\\deepseek\\result\\seal_request_v1.json`）：Q08（K1 改判：读法甲/乙）、Q01 更正表 **C1–C6**、I1/I9 ⇒ 之后才进 B 闸门 Q03（需 GPU）。\n- 零 GPU：**Q01/Q02/Q09/Q12 已完成**（复核 44/0、43/0、76/0）。挂账：下一实验 Phase（P8 `share_v` × P16/P17 `w_ℓ` 同精度对接）；N 线 P3–P7 补 Ledger；N2h1-α-1 权重级；N2h1-β 水果类；N3-β→ε；R1 补强；K4。",
  "- **✅ A 闸门已关闭（R8）**：seal `Q08=甲 | C=全接受 | I1=确认 | I9=确认`。K1 判定层 = **行为读出层** ⇒ **K1 触发**（读出层 3/3 否决、`E_read` 池化 **0.3734** = 门 **7.5×**）；k* 层 `model_specific`。「条件齿轮组 = 算子代数」**降级 descriptive**。C1–C6 全接受（**C4/C6 目标为跨线账本 ⇒ 只落补丁 `ledger_corrections_v1.json`，未施加**）。产物 `a_gate_closure_v1.json` `24c60160`；队列 5/30 sealed `37da9a8d`；复核 **38/0**。\n- **下一步 = B 闸门 Q03**（`E_read` 统一基线复算，**需 GPU**；本机已探明 RTX 5080 / 16 GB / torch 2.13+cu130 可用）→ Q04/Q05 `E_ar(k)` → Q06 `C_steer`。挂账：N 线 P3–P7 补 Ledger；跨线账本补丁施加确认；N2h1-α-1 权重级；水果类；K4。"),
 # §7
 ("`rdc-main-axis-probe`（15 臂 + 63 坑）、`rdc-phase-closeout`（**41 教训**）、`rdc-dual-arm-phase-template`。",
  "`rdc-main-axis-probe`（15 臂 + 63 坑）、`rdc-phase-closeout`（**42 教训**）、`rdc-dual-arm-phase-template`。"),
 # §6 追一条
 ("`atlas_ledger.json`（n=304）为**跨线共享**，不动。",
  "`atlas_ledger.json`（n=304）为**跨线共享**，不动。\n- **R8（seal 执行）**：A 闸门 **CLOSED**；Q08=甲 ⇒ K1 **fired@readout**（k* `model_specific`）；C1–C6 全接受（C4 `content_excluding_self` 独立复现 `0dc6e57a`）；memo **P35** 追加成功（前缀逐字节不变）。"),
]
for old, new in reps:
    c = t.count(old)
    assert c == 1, "MEMORY count=%d for %r" % (c, old[:70])
    t = t.replace(old, new)
add("MEMORY 后: 拟 %d chars" % len(t))
assert len(t) < 5200, "MEMORY 超阈值 %d" % len(t)

# ---------------- WLOG ----------------
wb, wbom, wt = load(WLOG)
add("WLOG 前: %d B  %s" % (len(wb), hashlib.sha256(wb).hexdigest()[:8]))
assert "## R8" not in wt
R8 = """## R8：A 闸门 seal 执行与关闭（Q08=甲 / C1–C6 全接受 / I1·I9 冻结）

- **seal 原文**：`seal: Q08=甲 | C=全接受 | I1=确认 | I9=确认`（用户 2026-10-03 05:00）。
- **Q08=甲**：K1 判定层 = **行为读出层**。逐模型读出层 margin `+0.7866 / +1.1296 / +0.9066`（MDE `0.1519 / 0.2168 / 0.2113`）⇒ **3/3 否决**；`E_read` = `0.3316 / 0.3986 / 0.3898`，池化 **0.3734 = 5% 门的 7.5×**。k* 层池化 margin `-0.0252`（2×MDE `0.0114`）⇒ 轨 A 不触发，但 `qwen3-4b` 单模型否决 ⇒ **`model_specific`**。判决：**K1 = fired_all_models** ⇒ 「条件齿轮组 = 算子代数」**降级 descriptive**。
- **C1–C6 全接受**：C1 加标 `count_mode=component`（62 条 / 94 分量）；C2 canonical **41.5%**(39/94)；C3 TESTPLAN T5 → `A=5/B=34/C=14/D=21/E=20`；C4 `ledger_sha256_8` `41d65a13` → **`0dc6e57a`**（`content_excluding_self`，本对话独立复现 ✓）；C5 `proposition_ledger.json` 真值 **`9a3c6ff4`**；C6 measurements schema 统一（`meas_id`+`phase`；`evidence_level` 三值枚举）。
- **⬛ 跨线冲突（诚实报告）**：取证 `atlas_ledger.json`（`bbda63df`, n=304）**同时**含 deepseek P8–21 与 G 线 P2902–3153 ⇒ 属**跨线共享**。依「避免与其他路线混合」纪律，**C4/C6 未就地施加**，只落补丁 spec `research\\deepseek\\atlas\\ledger_corrections_v1.json`（`beb775bd`，状态 `SEALED_BUT_NOT_APPLIED`，含 4 条 op + 目标前值 + 已复现正确值）。K1 改判登记同法（`phase: A-gate` 条目挂在补丁内）。**施加需一句确认**。
- **A 闸门 CLOSED**：Q01/Q02/Q08/Q09/Q12 = sealed；`phase_queue_v1.json` `675836fd` → **`37da9a8d`**（5/30 sealed，其余 25 仍 pending）。
- **产物**：`research\\deepseek\\atlas\\a_gate_closure_v1.json`（`24c60160`, 9496 B）+ memo **`## Phase 35`**（676,086 → **685,647 B**，`b257f54f` → **`150241da`**，前缀逐字节不变、bare_lf 0、35 个 Phase 标题）+ `tests\\deepseek\\result\\meta_single_source_v4.json` `f9d7ede6` → **`08f0dcf2`**（6 条更正全 accepted）+ 页 `a_gate_closure_r8.html`。
- **独立复核** `tests/deepseek/verify_r8.py` → **PASS 38 / FAIL 0 ALL_PASS**（memo 前缀=备份 + 独立重算 `E_read` 池化/C4 哈希 + 7 枚跨线受保护指纹未变 + 3 份备份可回滚）。
- **下一步**：B 闸门 Q03（`E_read` 统一基线复算，**需 GPU**；已探明 RTX 5080 / 16303 MiB / torch 2.13+cu130 可用）。
- **技能** `rdc-phase-closeout` 新增**教训 42**（seal 落盘先做目标归属判据；跨线目标只落补丁 + 挂账）。
- **一句话 ×3**：① A 闸门已关闭，主线存续定案 Q08=甲；② K1 触发 ⇒ 算子代数命题降级 descriptive；③ 全程未改跨线文件，C4/C6 以补丁挂账。
"""
wt = wt.rstrip("\n") + "\n\n" + R8
add("WLOG 后: 拟 %d chars" % len(wt))

# ---------------- SKILL ----------------
sb, sbom, st = load(SKILL)
add("SKILL 前: %d B  %s" % (len(sb), hashlib.sha256(sb).hexdigest()[:8]))
ANCH = "## 参照实现（Phase 3125，210/210 全绿；Phase 3128，19/19 全绿）"
assert st.count(ANCH) == 1, "SKILL anchor count=%d" % st.count(ANCH)
L42 = """42. **seal 落盘前必须做「目标归属判据」；跨线目标只落补丁 + 挂账（R8 实证，2026-10-03）**：
   用户 seal「C=全接受」时，**接受的是一条决定，不等于可以在任意文件上就地施加**——更正/登记的**目标文件**可能不属于本线。R8 实测：C1/C2/C5 的目标在**他线备忘录**、C4/C6 的目标是**跨线共享账本**（`atlas_ledger.json` 同时含 deepseek P8–21 与 G 线 P2902–3153）。
   (a) **先取证归属再动手**：数「本线备忘录引用次数 vs 他线备忘录引用次数 vs 文件自声明线」；只有本线独占的才可就地改。
   (b) 跨线/他线目标 ⇒ **不就地改**，落「补丁 spec」JSON（含 `target_sha256_before` + 逐条 `op` + `recorded_correct_value` + 状态 `SEALED_BUT_NOT_APPLIED`），并在备忘录里挂账。
   (c) **append-only 文档的「更正」= 追加 erratum，不改写归档正文**；归档全文（如并入的 TESTPLAN）逐字保留，更正只在新增节声明。
   (d) **自指哈希类更正先独立复现再落**：C4 的 `content_excluding_self` 必须自己按 `json.dumps(obj_去掉该字段, ensure_ascii=False, indent=1)`（**无尾换行**）重算并比对；带尾换行会得到另一个值（实测 `3aaec327` ≠ `0dc6e57a`）。
   (e) 关闭宣告要落**可机读**记录（`*_closure_v1.json`：`gate_closed` / 逐项 decision / KPI 现场值 / 受保护指纹），而非只写在正文里。
   (f) **队列状态更新属正常推进**（`pending` → `sealed`），但会改 `sha8`：新旧哈希都要在文本里写明，避免下游引用错位。
   (g) 收尾仍走五步 + 独立复核；本轮复核口径 = 「前缀逐字节不变 + 独立重算关键量 + 跨线指纹未变 + 备份可回滚」。

"""
st = st.replace(ANCH, L42 + ANCH)
add("SKILL 后: 拟 %d chars" % len(st))

# ---------------- 写入 + 回读 ----------------
save(MEM, t, bom); save(WLOG, wt, wbom); save(SKILL, st, sbom)
add("")
add("== 回读 ==")
for p, lbl, needle in [(MEM, "MEMORY", "A 闸门已关闭（R8）"), (WLOG, "WLOG", "## R8：A 闸门 seal 执行与关闭"),
                       (SKILL, "SKILL", "42. **seal 落盘前必须做")]:
    b = open(p, "rb").read(); tx = b.decode("utf-8-sig")
    ok = needle in tx
    add("  %-7s %s  %6d B  chars=%-5d  needle=%s" % (lbl, hashlib.sha256(b).hexdigest()[:8], len(b), len(tx), "OK" if ok else "MISS!"))
    if lbl == "MEMORY": assert len(tx) < 5200
    assert ok, "%s needle 缺失" % lbl

add("")
add("== 跨线受保护（须未变）==")
for rel, exp in [(r"research\gpt5\docs\AGI_GPT5_MEMO.md","2a84776b"),
                 (r"research\gpt5\atlas\atlas_ledger.json","bbda63df"),
                 (r"tests\glm5\result\rdc_query_construction_20260913\phase3103\omega_p101_formula_audit\proposition_ledger.json","9a3c6ff4")]:
    p = os.path.join(ROOT, rel); g = sha8(p)
    add("  %-8s exp=%s got=%s %s" % ("OK" if g==exp else "DRIFT!", exp, g, os.path.basename(p)))

txt = "\n".join(rep)
open(OUT, "w", encoding="utf-8").write(txt)
print(txt)
