# -*- coding: utf-8 -*-
import os, time
ROOT = r'D:\AI2050\Ai2050-OpenOne'
rep = []

# ---------- 1) 工作区日志 append ----------
log = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-01.md')
add = """

## N1 探针：主轴三段分工（2026-10-01 03:45）
- 触发：用户「词嵌入必然带特征（否则所有嵌入参数应相同）；必须基于语言特性 + 嵌入-layer-反嵌入主轴破解编码机制」。
- 执行：**6 模型 × 4 臂**（E3/E3b 词嵌入审计；N1v2 主轴层扫描；N1b 反嵌入读出；N1c 任务依赖）。预注册 `tests/gpt5_temp/N1_design_seal.json`；GPU 全程空闲（3149 已于 01:58 结案），无冲突、无 OOM。
- 核心结果：
  - **嵌入带特征=成立且可量化**：类线性探针 LOO **97.9–100%**（9 类，chance 11.1%）；类内−类间 cos gap 全正（+0.080~+0.147）。
  - **嵌入只是一级词典**：`苹果-水果` 0.13–0.33，跳一级 `苹果-食物` 仅 **0.024–0.081（<随机 q95）**；`水果-食物` 0.11–0.19 → **层次不是距离梯度，是"直接上位对齐"**。
  - **居中修正 E1**：`cos(苹果,公司)` raw 在零带内，减公共模后 +0.082/+0.085 **越出零带** → 第二义确有微弱痕迹（修 E1「字面全无痕」表述）。
  - **★权重绑定决定 is-a 落点（6/6 干净二分）**：tied（qwen3-1.7b/4b、qwen2.5-3b）N1b L0 = **+18.9~+24.4**、比值 1.00×、B 放大 1.00–1.05×；untied（qwen2-7b、glm4-9b）L0 = **+0.068/+0.40**、比值 **6.0–33.8×**、B 放大 **1.68/2.41×**。→ 功能等价、实现路径不同，是账本"端口类"的**第二个独立实例**，落在最底层嵌入–输出对齐上。
  - **任务依赖**：`这是{W}` 下 is-a rank **105**（层主动抹掉范式信号）；`{W}是一种` 下 **rank 1 @L31–33**（层重建）。
  - **端口替换**：末位替换嵌入行 → top1 改变率 **6/6 = 100%**；有后缀时被覆盖（保持 0.31–0.88）。
  - dS 双峰形状（早窗 20–45% + 晚窗 25–40% + 静默中段）**5/5 适用模型**；gemma-3-4b 表示坍缩，本度量不适用（已排除）。
- 产物：`research/gpt5/docs/MAIN_AXIS_VERDICT_v1.md`（sha8 827fc48d）；脚本 `e3_embed_feature_audit.py`/`e3b_embed_followup.py`/`n1_v2_main_axis_scan.py`/`n1b_ontology_readout.py`/`n1c_ontology_cloze.py`；报告 13 份（n1v2×6 / n1b×5 / n1c×2 / e3 / e3b）。
- MEMO：追加 `## 探索性探针 N1`（L14804；**非 Phase 编号**，因 3150 已被 T4 线 Ω-P148 预注册占用）。1898629 → 1909262 bytes，前缀逐字节未变。
- **⚠️ 事故（需用户裁决）**：03:15:57 复查时 MEMO 含 `## 设计草案`(L14763) 与 `## 探索性探针 E1`(L14902)；03:43 复查两者 **count=0**，恰好 **250 行消失**，`## Phase 3149` 从 L15014 上移到 L14763。内容未丢（standalone 文档 `RDC_TESTPLAN_v1.md`/`EMBED_ANCHOR_VERDICT_v1.md` 完好），但 **append-only 纪律被破坏**。嫌疑：按 Phase 的 closeout 脚本含 `open(...,'w')` 模式（`p3149_closeout.py` 同时含 append 与 WRITE/TRUNC）+ 项目存在 memcompress 实践。当前 MEMO 的 `## 探索性探针 N1` 同样有被删风险。
"""
b0 = os.path.getsize(log)
with open(log, 'a', encoding='utf-8', newline='') as f:
    f.write(add)
rep.append('wlog %d -> %d (+%d)' % (b0, os.path.getsize(log), os.path.getsize(log) - b0))
T = open(log, encoding='utf-8').read()
rep.append('wlog has N1 section: %s' % ('## N1 探针：主轴三段分工' in T))

# ---------- 2) MEMORY.md 定点插入 ----------
mem = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
Tm = open(mem, encoding='utf-8').read()
anchor = "软门记录型优先于硬 assert（防长跑尾炸）。)"
ins = anchor + "\n4. **MEMO 非 Phase 节有被外部进程删除的风险**（2026-10-01 事故）：03:15 存在的 `## 设计草案`+`## 探索性探针 E1` 两节在 03:43 消失（恰好 250 行，Phase 3149 随之上移）。按 Phase 的 closeout 脚本含 `open(*,'w')` 模式且项目存在 memcompress 实践。对策：非 Phase 交付**必须同时留 standalone 文档**，并在 MEMO 追加后 5 分钟内二次 Grep 复核存在性。"
assert Tm.count(anchor) == 1, 'anchor count %d' % Tm.count(anchor)
Tm = Tm.replace(anchor, ins)

n1sec = """

## 主轴三段分工（N1，2026-10-01，6 模型）
- **嵌入=范式层**：身份/类簇（LOO 97.9–100%）/一级 is-a 对齐（tied 下即输出头对齐 +19~24 logit）/弱第二义（居中后越零带）。**只有 1 级**：`苹果-食物` <随机 q95。
- **层=范式→序列转换器 + 按需重建**：早窗（20–45% 深）dS 显著；中段压制范式信号（is-a rank 105→数千；B-margin 保留 0–53%）；晚窗（>68% 深）dS 重现 + 输出分叉。任务要求时 **L31–33（86–92% 深）把上位词顶到 rank 1**。
- **输出头=只读最近几页**：末位嵌入行替换 → top1 改变 **6/6 = 100%**；有后缀即被覆盖（保持 0.31–0.88）。
- **★绑定二分（6/6）**：`tie_word_embeddings=True` → is-a 在 L0 免费（+18.9~24.4），层压制（1.00–1.05×）；`=False` → L0≈0（+0.07/+0.40），层构建（1.68–33.8×，峰在末层）。**读任何"关系在哪些层"之前必须先看绑定。**
- 文档 `research/gpt5/docs/MAIN_AXIS_VERDICT_v1.md`；后续 N2（L29–33 重建源逐头归因，死线单头≤30%）/N3（untied 复现写头）/N4（held-out 概念集）/N5（早晚窗分离）/N6（绑定干预检验）。
- 端口替换仍是唯一合法输入干预；**方向去除/减法禁用于本系统**。
"""
b1 = len(Tm.encode('utf-8'))
Tm = Tm.rstrip('\n') + '\n' + n1sec
with open(mem, 'w', encoding='utf-8', newline='') as f:
    f.write(Tm)
T2 = open(mem, encoding='utf-8').read()
rep.append('MEMORY.md %d -> %d bytes' % (b1, len(T2.encode('utf-8'))))
rep.append('MEMORY has item4: %s' % ('MEMO 非 Phase 节有被外部进程删除的风险' in T2))
rep.append('MEMORY has N1 sec: %s' % ('## 主轴三段分工（N1，2026-10-01，6 模型）' in T2))

open(os.path.join(ROOT, 'gpt5_temp', 'verify_memory_n1.txt'), 'w', encoding='utf-8').write('\n'.join(rep))
print('done')
