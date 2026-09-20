# -*- coding: utf-8 -*-
"""Append Phase 2934 section to AGI_GPT5_MEMO.md."""
import hashlib

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2934_memo_append_report.txt')

sec = """## Phase 2934: 承重带功能解剖——双位置三条件 3x3 消融与 lin_r-CI 律普适性 [2026-09-19 16:44]

### 目的与设计（预注册，execution.json 先落盘）
- 问题：2933 的 lin_r-CI 律（准线性层承重、深非线性层静默）只在 pos 1 单位置 + func 单条件下测量——是普适律还是 pos1/func 伪象。
- 3x3 设计：消融配置 {pos0, pos1, pos01} × 读出条件 {same, func, null}（2927 prompt 构造 verbatim：same=[同语言最小 tid 词, w]、func=[the, w]、null=[rng(SEED) 非词表 tid, w]），全部 1120 门控格真实消融，读出 dirs_word[35] 投影。
- 判决映射（冻结）：三配置全稳定（r<0, p1b<=0.01, p1a<=0.05, diff>0）且三条件全稳定 => linr_ci_law_general；≥1 新配置或条件稳定 => linr_ci_law_partial；否则 linr_ci_law_pos1_only。

### run1 锚失败与修正（correction_note 入 PREREG）
- run1（单 batch171 拼接前向）a6 失败：pos1/func CI_rel vs 2933 max abs diff 3.69e-03 > 1e-4。a2 同 batch 组成确定性 0.00 证明非实现漂移，根因是 **bf16 跨 batch 组成数值噪声**（kernel 归约顺序随 batch 组成变化）。
- 修正：三条件独立 batch57 前向（与 2933 batch 组成 bit 级一致），a6 容差不变；run1 判决 anchor_fail_all_void 如实登记后删产物重跑。

### 锚（run2，7/7 全过）
- a1 dirs_word 重建 2.17e-08（**第十一次连续前向锚定**）；a2 确定性 0.0；a3 pos01 hook 有效性 1.4468；a4 掩码 469/382/295；a5 func 分离 185.70；**a6 pos1/func CI_rel vs 2933 全 1120 格 max abs diff = 0.00e+00（bit 级，修正后立即达成）**；a7 func 分离 > null 分离（185.7 > 77.3）。

### 结果
- **P1 位置维：三配置全稳定**——pos0 LOAD 0.004594 vs DEEP 0.001459（×3.15）rho(med_l, lin_r)=**−0.8443**；pos1 −0.6762；pos01 −0.6793（全部 p_band≤6e-4、p_linr≤1e-4）。**lin_r-CI 律是位置普适的**，pos0 下律更强（−0.84）。
- **P2 条件维：三条件全稳定（pos1）**——same −0.6854、func −0.6762、null −0.6574（p≤2e-4）。**lin_r-CI 律是条件普适的**。冻结判决 **linr_ci_law_general**。
- **P3 seal 取证（两项新发现）**：
  1. **强次可加性**：pos01 grand median 0.005183 vs pos1 0.005100（ratio 1.016）——ctx 位（pos0）的贡献几乎完全被词位消融覆盖（pos0/pos1 = 0.790 但联合只 +1.6%）；pos0 的 top 层是 L1-L5（ctx 位由早层主导）。
  2. **null 条件放大**：null 上下文 CI 全面高于 func（LOAD ×1.83、DEEP ×2.00）——随机 token 上下文使读出对消融**处处更敏感**（band 比值守恒，律不变但幅度放大）。
  - 幸存核 7/7 全部 shared 骨架成员、CI 与 2933 一致；骨架 vs 其余 ×1.183；top-5 pos1/func 格与 2933 bit 级相同（(15,34) 0.040383 全格 max）。

### 硬伤
- run1 修正属于锚口径修正（bf16 batch 组成噪声），非判据修正——但暴露"跨 batch 组成比较 CI"类锚必须组成一致，已写入 run 报告。
- null 放大的机制未解（上下文熵效应 vs 词覆盖效应）；单模型 n=1（a6 bit 级缓解）；CI 读出单一投影方向。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2934_loadband_anatomy.py: 8aa2c4ab
- execution.json: 3ebbc5a7（created 2026-09-19T16:44:39）
- result.json: 4769249b（final_verdict=linr_ci_law_general，runtime 510.9 s）
- loadband_anatomy.npz: 62da7e11（cells/ci_rel 9x1120/cfgs/conds/s_base/scale/sep/dirs_word）
- 源：2887 e4835a87；2927 84fec594；2929 57ed5651；2930 cb655825；2931 5307afe1；2933 1ff6df21
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2934/loadband_anatomy/
- Ledger：M2934_loadband_anatomy 入账，measurements 72->73，L14 connects 40->41，ledger sha256-8 = 1a41a0c2

### 接续（2935 候选）
- A（主选）：**null 放大机制解剖**——null tid 集合重采样（R 组）+ 上下文词频/熵分层，检验 CI 放大是上下文统计效应还是 token 身份效应（一次运行）。
- B：eps 扫描线性度——lin_r 在 eps ∈ {0.1, 0.3, 1.0} 缩放检验（偶阶 ~eps 预测；一次前向族）。
- C（零前向）：h4 L1<->L19 复用子空间主角度（roadmap 遗留项）。
- D：承重带跨模型复现（glm4 双条件消融子采样，一次前向；只主张 lin_r<0.9 层）。
"""

h = hashlib.sha256()
h.update(open(P, 'rb').read())
before = h.hexdigest()[:8]
with open(P, 'a', encoding='utf-8') as f:
    f.write('\n' + sec)
lines = open(P, encoding='utf-8').read().splitlines()
title_ok = any(l.startswith('## Phase 2934:') for l in lines)
h2 = hashlib.sha256()
h2.update(open(P, 'rb').read())
rep = ['before_sha=%s' % before,
       'after_sha=%s' % h2.hexdigest()[:8],
       'total_lines=%d' % len(lines),
       'title_ok=%s' % title_ok,
       'tail=%s' % lines[-1][:80]]
open(REP, 'w', encoding='utf-8').write('\n'.join(rep) + '\n')
print('memo ok')
