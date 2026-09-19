# -*- coding: utf-8 -*-
"""Append Phase 2861 section to AGI_GPT5_MEMO.md (append-only)."""
import io

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'

SEC = """
---

## Phase 2861 (2026-09-18) — 真残差工作点重测：no_amplification，窗口解剖战线正式收束

### 动机与协议（预注册冻结于任何观测前）
2860 定位硬伤④（工作点混合：2853 g_jac / 2855 g_comp 的扰动点用伪残差重构、基线用 hook 真值）。
2861 在**正确工作点**上重测窗口解剖三量——x_in 一律用 pre-hook 直接捕获（硬伤④修正协议首次执行）：
- ln2in_true = post_attention_layernorm 的 pre-hook（真 LN2 输入 = x + attn_out）；
- mlpin_true = mlp 的 pre-hook（真 mlp 输入 = LN2(ln2in_true)）；
- out_b_true = mlp_raw(mlpin_true)（显式重构，与扰动同点自洽）。
- g_ln_true = [LN2(ln2in+ε·cdir) − LN2(ln2in)]·cdir/ε；
  g_comp_true = [mlp(LN2(ln2in+ε·cdir)) − out_b_true]·cdir/ε；
  g_direct_true = [mlp(mlpin+ε·cdir) − out_b_true]·cdir/ε。
- 内建一致性校验（bf16 确定性，预期 ~0）：v1 = max|LN2 重算(ln2in_true) − mlpin_true|；
  v2 = max|mlp 重算(mlpin_true) − mlpout_true|。**实测 v1 = v2 = 0.0（逐位）**——
  工作点捕获正确性的铁证，同时证明 hook 重算与 hook 真值在本协议内完全互换。
- 其余协议逐句复刻 2860/2855（SEED=2855 词表 80 词、L26-35、EPS=1.0、L13H30 钳制臂 max_resid=0.01062 ≡ 前两轮）。
- 预注册：E1 = 2860 判据形式在真工作点重跑；E2 = 窗口大增益存在性
  （no_amplification iff max_L|g_comp_true|<1.2 且 max_L|g_direct_true|<1.2，2853 T1 阈值）；
  E3 描述性 r(g_direct_true, g_emp_2853)；E4 描述性 |g_comp_true − g_comp_2860|（工作点敏感性）。

### 结果（phase2861/g_true/；exec f23ea1d9… / result 040dce3b… / g_true.npz ac391ed0…）
| 量 | L26→L35 谱 | 判决 |
|---|---|---|
| E1 | n_close=0 / n_rewrite=0，ratio 2.49→8.16 | **mixed**（分母近零失义，与 2860 同构） |
| E2 | max\|g_comp_true\|=**0.116**，max\|g_direct_true\|=**0.947** | **no_amplification**（双双 <1.2） |
| E3 | r(g_direct_true, g_emp_2853) = **0.781** | clamp 实测增益与直接注入谱中强正相关 |
| E4 | max\|g_comp_true − g_comp_2860\| = 0.150 | 复合臂对工作点不敏感（两工作点自洽值都近零） |
| g_comp_true | −0.02, 0.03, 0.03, 0.07, 0.10, 0.06, −0.02, −0.07, −0.11, −0.12 | 复合响应近零 |
| g_direct_true | −0.05, 0.13, 0.12, 0.31, 0.48, 0.39, −0.12, −0.47, −0.86, **−0.95** | 峰值 L30 +0.48；L32-35 温和负响应 |
| g_ln_true | 0.29 → 0.12（单调递减） | 真工作点 LN2 改写幅度比伪工作点（0.71→0.44）小一半以上 |
| dln_cos | 0.94 → 0.76 | 方向保留趋势与 2860 一致 |
| v1 / v2 | **0.0 / 0.0** | 校验通过 |
| max_resid | 0.01062 | 钳制臂 ≡ 2855/2860 |

### 科学结论
1. **"L32-34 深层放大器 ~7"不存在**——真残差工作点上窗口内无任何 ≥1.2 的增益臂（E2）。
   2853 T1 的 active_amplification 判决正式作废；2853/2855 的 7.07 确认为纯工作点混合伪影（硬伤④）。
2. **深层 SwiGLU 对类方向的响应是温和负反馈**：g_direct_true 在 L32-35 单调走负（−0.12→−0.95），
   峰值正响应在 L30（+0.48）。mlp 臂无放大器角色——与 2846 双谱"放大器"头级分类无冲突
   （那是 attention OV 臂）。
3. **复合臂 g_comp ≈ 0 是真实小量而非伪影**：两个工作点独立自洽测量一致（E4 ≤0.15），
   LN2 改写（≤0.29）与直接注入（≤0.95）部分抵消（g_diff L35 = +0.83）。
4. **E3 r=0.78**：2853 clamp 态实测增益（干净量）的主要部分可由直接注入响应谱解释——
   clamp 位移与 cdir 方向的高度重叠所致；残余 0.22 方差来自 clamp 位移的非 cdir 分量。
5. g_ln_true（0.29→0.12）显著小于伪工作点 g_ln（0.71→0.44）：伪残差点（LN1 输出）的
   切向分量比真残差点大——LN1 本身就是强切向收缩器，"LN2 收缩切向扰动"的量级必须按工作点报告。

### 窗口解剖战线收束清单（2853→2861）
- 2853 T1 active_amplification：**作废**（伪影）；T1 正确判决 = no_amplification（2861 E2）。
- 2853 T2 r(g_jac, g_emp)：原值被 g_jac 伪影污染；替代读出 = 2861 E3 r(g_direct_true, g_emp) = 0.781。
- 2853 clamp 态 g_emp（1.68/1.28/0.90 @L29-31）：**存活**（双 hook 实测比，不受硬伤④污染）。
- 2855 W1 ln2_mediated：形式成立（g_ln 自洽）但机制解读作废——正确表述：
  LN2 改写与直接注入均为小量且部分抵消，复合响应近零，无放大臂。
- 2855 W2 neuron_diffuse / L1 linear_ok=false：伪残差工作点产物，降级为"未定"；
  如需复活须在真工作点重跑神经元分解（低优先级——放大臂不存在，神经元定位问题失去锚点）。
- 2854 gain_specificity：差分结构待复核（x_base/x_clamp 均为 sain+attn 伪残差，
  但若两点同为伪残差语义则自洽——其 clamp 增益结论与 2853 g_emp 同源，预计存活；列为附录待办）。

### 硬伤④修正协议的制度化确认
pre-hook 直接捕获工作点 + 显式重构差分 + v1/v2 逐位校验 = 本系列第一个端到端无瑕测量协议。
**凡涉及"在 X 点注入扰动"的实验，X 必须用目标模块的 pre-hook 直接捕获，禁止用上游 hook 值重构。**
（制度条文与 2860 节一致，本 Phase 为首次执行 + 校验机制落地。）

### 接续（MASTER_PLAN II1/II3 双谱规范主线）
窗口解剖战线（2853-2861，九个 Phase）完全关闭。下一阶段回归双谱规范推进：
对齐谱（share）⊕ 因果谱（drop）的机制解释主线（形成器/放大器头级分工的下游验证、
MA2 全头普查的 top-64 前缘机制解剖）。说"继续"即进入 2862。
（脚本 tests/glm5/phase2861_g_true.py；产物 phase2861/g_true/；SHA 见探针 _p2861_sha_report.txt）
"""

with io.open(MEMO, 'r', encoding='utf-8') as f:
    n_before = sum(1 for _ in f)

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(SEC)

with io.open(MEMO, 'r', encoding='utf-8') as f:
    n_after = sum(1 for _ in f)

with io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_p2861_memo_check.txt',
             'w', encoding='utf-8') as f:
    f.write('lines_before=%d lines_after=%d appended=%d\n'
            % (n_before, n_after, n_after - n_before))
print('OK %d -> %d' % (n_before, n_after))
