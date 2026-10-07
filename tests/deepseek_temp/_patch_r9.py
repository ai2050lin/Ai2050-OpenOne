# -*- coding: utf-8 -*-
"""R9 收尾：MEMORY.md 全量重写（压缩+更新）+ wlog 追加 R9 + 技能补教训 43。每处独立 readback。"""
import os, hashlib

M = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md'
W = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-03.md'
S = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

NEW_MEMORY = """# RDC/LPF 项目纪律（工作区长期记忆）
> 权威记录 `research\\deepseek\\docs\\AGI_DEEPSEEK_MEMO.md`（**唯一研究日志**，Phase 1–37）。本文件仅跨轮索引，细节回查 MEMO / 技能。

## 0 约定（v4，2026-10-03，**仅本对话遵守**）
- 日志唯一落点 = `AGI_DEEPSEEK_MEMO.md`（append-only、BOM+CRLF、标题 `## Phase {N}: 短标题 [hh:mm]`）；**不新建其他 `.md`**。`AGI_GPT5_MEMO.md` 属他线，**不读不改不追加**。**另有并发写者**（P36 非本对话）⇒ 只 append 不碰前文；文件 free bare_lf = **45**（P36 引入）。
- 三桶：测试脚本 `tests\\deepseek\\`；临时脚本 `tests\\deepseek_temp\\`；测试结果 `tests\\deepseek\\result\\`。历史 `Phase1..21\\` 不动。
- MEMO：P1–21 N 线；P22 = A 闸门 R3–R5b；P23–30 = 8 件并入；P31 = R6 路由；P32–34 = 归属整理；**P35 = A 闸门 seal（R8）**；**P36 = E4/E4b（他线）**；**P37 = Q03 E_read 基线（R9）**。现 `df987b1f`（700044 B）。
- Ledger `atlas_ledger.json` n=**304** `bbda63df`（**N 线 P3–P7 待补**）。「好的，继续」= AI 主导续研；**关键发现重复 3 次**。

## 1 装置铁律（63 坑全文见 skill `rdc-main-axis-probe`）
(a) 份额只用**精确可加向量预算**；(b) SMOKE 必做**必看数字**；(o) 多 Edit 静默丢 ⇒ Python 补丁 + `assert count==1` + 回读；(ad) 实现与 seal **逐字一致**；(ae) 数字**一律 result 现场渲染**；(af) **冻结锚重验**。
其余：patch 剔末层；写入窗=相邻最大增量；`jump/max|effect|≥0.5`；双剂量 `x*·r_ℓ`；固定基报 `overlap`；paired `margin` 负=候选优。

## 2 N 线主线（P4→P21）
- P4–P7 主轴三段（嵌入=词典/层=开关/**权重绑定定 is-a 落点**）；单头 share_max **3.0%**；读位槽 **G−1=5 维**；跨族近正交 ⇒ 无通用类别算子。
- P8–P11 写入端**分布式**（向量预算 MLP 0.472）；L6 内**无阈值增益** ⇒ **栈=软门**；深端塌陷主因**方向失配**。
- P12–P16 `rho(xhalf,depth)=−0.783`；**P16 否证 ⇒ P12/13/14 深度表述撤回**；域 `REACH={ℓ:ρ≥0.10}`。
- P17–P21 `w_ℓ`+`com_V` ≈26 ≫ median(REACH)；`spearman(w,|b|)` 与 P17 **反号 ⇒ P17 P6 对象错配**；跨精度 7/9、A0_bf16 **逐位复现 P8 锚**。

## 3 挂账与限界
- **核心三条限界**：① 否定臂基线不成立；② 「未见类别」仍崩（水果 0.04/0.05）；③ **激活级干预 ≠ 权重级证明**（其余见 MEMO / 技能 26–43）。
- R1/R2：10 条成立；P1 纠错 1+降级 4+挂账 5。G 线索引：3151 k3_only / 3152 k1_not_triggered；判决符号 rev-3151b（负=优）。

## 4 本机缺陷（Windows）
- bash shim `ls/rm/tail/dirname/cd` 坏、内联 `python -c` stdout 常丢、**反引号被吃** ⇒ Python 文件化 + 写 `.txt` 再 Read。
- **`Edit` 幻影（R9 复现 2 次）**、脚本日志亦可能幻影 ⇒ 中文大段走 Python 补丁（**禁凭记忆写匹配串，先 dump 真实磁盘**）；**落盘证据只能靠独立进程 re-hash**。`Read` 大文件视图陈旧 ⇒ 改没改用 Grep/Python。
- **`%` 格式化陷阱**：`% (...)` 行里字面 `%`（如 `5% 门`）须写 `%%`；**无格式化**的行写 `%%` 会原样输出。两者都要查。
- **`AGI_DEEPSEEK_MEMO.md` 未被 git 跟踪** ⇒ 改动前落快照；**外部 +2 B 写入**已两次（R6→R7、R8→P36，非本线）。

## 5 下一步 / 死线
- **✅ A 闸门已关闭（R8）**：seal `Q08=甲 | C=全接受 | I1=确认 | I9=确认`。K1 判定层=**行为读出层** ⇒ **K1 触发**（3/3 否决、池化 **0.3734** = 门 7.5×）；k* `model_specific`。「条件齿轮组=算子代数」**降级 descriptive**。C1–C6 全接受（C4/C6 为跨线账本 ⇒ 只落补丁 `ledger_corrections_v1.json`，未施加）。产物 `24c60160`；复核 38/0。
- **✅ Q03 E_read 基线锁定（R9）**：三模型 0.331615/0.398601/0.389835，池化 **0.373350**，**5% 门 0/3**（最小 6.63×）；三份 `collect.npz` 锚**逐位复现 drift 0.00e+00**；三模型**同一 held-out 指纹**；队列 6/30 sealed `d0d2e208`；复核 42/0 + 28/0。
- **下一步 = Q04 `E_ar(k)` 装置**（gpu=mid，SMOKE 先通）→ Q05/Q06 `C_steer`。挂账：N 线 P3–P7 补 Ledger；跨线账本补丁施加确认；N2h1-α-1 权重级；水果类；K4。

## 6 元层诊断与 A 闸门（MEMO Phase 22–34）
- **诊断**：gpt5 MEMO 402 Phase/1.90 MB；判决 685 次但**唯一标签 48（复用率 1.10）⇒ 不可累积**；397 Phase 仅 26（6.55%）报告共享指标变化。
- **K1 双轨（Q09）**：k* `model_specific`；读出层 `fired_all_models`（margin +0.9409）。**K2/K3 从未被测量**。
- **Q01/Q02**：真源 62 条按**分量**复现 A5/B34/C14/D21/E20（和 94）；「55%/63%」= **量纲混淆**（41.5%=39/94）；`metric_dict` v2 `03887e51`。
- **Q12**：id 命中 1 + 指纹 0 ⇒ 无实质违规；缺陷 **RULE-UNDECIDABLE**、**NS-COLLISION**。
- **归档（R6+R7）**：11 件 `.md` → MEMO P23–34（备份 `_archive_r6|_r7\\gpt5_docs\\`）；`metric_dict/phase_queue` → `research\\deepseek\\atlas\\`；复核 44/0、44/0。
- **归属判据（R7）**：**谁的备忘录登记它，就是谁的线**。gpt5 余 20 件判为他线未动；`atlas_ledger.json` 跨线共享不动。

## 7 技能
`rdc-main-axis-probe`（15 臂 + 63 坑）、`rdc-phase-closeout`（**43 教训**）、`rdc-dual-arm-phase-template`。
"""

WLOG_BLOCK = """
## R9：B 闸门 Q03 E_read 统一基线复算（2026-10-03 07:23）
- 依据宪法 §1(I1) + metric_dict global_kpis.E_read；方法逐字复用 3152 的 split_s1/rows_of/phi_main/ridge_primal/b4_fit（零 GPU，recompute-only）。
- 三份冻结 collect.npz 独立重跑 B4@readout：qwen3-4b 4ef190b7 / qwen3-14b 733182c1 / glm4-9b c711946c。
- 结果：0.331615 / 0.398601 / 0.389835，池化 0.373350（sd 0.036408）；5% 门 0/3（最小 6.63×）；全部 6 组锚 drift = 0.00e+00（逐位复现）。
- 统一 held-out 指纹：PAIRS 41×6=246、seeds[7,8,9]、frac 0.2 ⇒ 每 seed 49 test pairs ×3 模板 = 147 test rows（3151/3152 预注册逐字相同，三模型共用）。
- bootstrap 95% CI：row-level n_boot=10000（rng 20261003），每 seed 147 行。
- 产物：q03_execution.json 5e2d4f0f（design_sha 9a26f72f）/ q03_result.json 57827730（res_sha8 cda99b05）/ q03_report.txt；verify_q03.txt 42/0；verify_q03_landing.txt 28/0；q03_baseline_r9.html 0c50df9a。
- memo：692727 → 700044 B（5b3bdcec → df987b1f），追加 Phase 37；前缀逐字节不变，Phase 37 区间纯 CRLF。队列：37da9a8d → d0d2e208，Q03 sealed（sealed_items 6 项）。
- ⚠ 并发写者：memo 在 R8 后、本轮前被外部写入（+2 B → 5be5bf12，随后 Phase 36 → 5b3bdcec）。P36 为 E4/E4b（非本对话）；其追加引入 45 个裸 LF（全在 P36 区间）。未改他线，仅报告。
- 本机缺陷复现：Edit 幻影 2 次（verify_q03.py 路径修正未落盘）；% 格式化陷阱 3 处；补技能教训 43。
"""

SKILL_LESSON = """43. **收尾前必须检测「并发写者」与「外部漂移」（R9 新增）**
   (a) 同一日志可能被**另一会话/进程**并行追加。若「上轮结束时的 sha8 ≠ 本轮开始时前缀 sha8」，先**逐字节定位**（`prefix(N)` re-hash 对照），不要默认是自己写坏。
   (b) 判据：找并发写者自己的**登记哈希**（其脚本里的 `before=...`）；若 `prefix(N)` 与之相符 ⇒ 追加未篡改前文，只是自己不是唯一写者。
   (c) **相位编号必须让位**：他人已占 `Phase 36` ⇒ 自己用 `Phase 37`；**绝不改写他人条目**，只在其后 append。
   (d) 他人质量缺陷（如换行不一致：本线 CRLF、他人引入裸 LF）**只报告、不擅自修**（属「改他线内容」）；自己的追加仍保持本线规范（BOM+CRLF、bare_lf 不变）。
   (e) **本机 `Edit` 幻影再次复现**（报成功未落盘）⇒ 跨文件补丁一律**文件化 Python 补丁 + `assert count==1` + 独立进程 re-hash**。
   (f) **`%` 格式化陷阱**：`% (...)` 行里字面 `%` 须写 `%%`；**无格式化**的行写 `%%` 会原样输出 —— 两种都要扫。
   (g) 收尾仍走五步 + 独立复核（本轮口径 = memo 前缀不变 + Phase 区间纯 CRLF + queue 状态 + 跨线指纹未变）。

"""

# ---- 1) MEMORY 全量重写 ----
chars = len(NEW_MEMORY)
assert chars < 4000, 'MEMORY 过长: %d' % chars
open(M, 'w', encoding='utf-8', newline='\n').write(NEW_MEMORY)
back = open(M, 'rb').read().decode('utf-8-sig')
ok_m = (back == NEW_MEMORY)
print('MEMORY   chars=%d  sha8=%s  readback_equal=%s' % (chars, sha8(M), ok_m))
for a in ['df987b1f', 'P37', 'Q03 E_read 基线锁定', '并发写者', '43 教训', 'd0d2e208', 'Edit` 幻影']:
    print('   %s %s' % ('OK ' if a in back else 'NO!', a))

# ---- 2) wlog append ----
wb = open(W, 'rb').read()
wt = wb.decode('utf-8-sig')
nl = '\r\n' if wt.count('\r\n') > (wt.count('\n') - wt.count('\r\n')) else '\n'
body = WLOG_BLOCK.replace('\n', nl)
if not wt.endswith(nl):
    body = nl + body
open(W, 'ab').write(body.encode('utf-8'))
wa = open(W, 'rb').read()
print('WLOG     %d -> %d B  sha8=%s  prefix_ok=%s  R9=%s' % (
    len(wb), len(wa), sha8(W), wa[:len(wb)] == wb, '## R9' in open(W, encoding='utf-8').read()))

# ---- 3) skill append lesson 43 ----
sb = open(S, 'rb').read()
st = sb.decode('utf-8-sig')
snl = '\r\n' if st.count('\r\n') > (st.count('\n') - st.count('\r\n')) else '\n'
anchor = '## 参照实现（Phase 3125'
assert st.count(anchor) == 1, 'anchor count=%d' % st.count(anchor)
lesson = SKILL_LESSON.replace('\n', snl)
new_st = st.replace(anchor, lesson + anchor)
open(S, 'w', encoding='utf-8', newline='').write(new_st)
sa = open(S, encoding='utf-8').read()
print('SKILL    %d -> %d chars  sha8=%s  lesson43=%s  a_g=%s' % (
    len(st), len(sa), sha8(S), '43. **收尾前必须检测' in sa,
    all(x in sa for x in ['(a) 同一日志可能被', '(g) 收尾仍走五步'])))
print('DONE')
