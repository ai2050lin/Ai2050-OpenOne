# -*- coding: utf-8 -*-
"""Append Phase 2860 section to AGI_GPT5_MEMO.md (append-only)."""
import io

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'

SEC = """
---

## Phase 2860 (2026-09-18) — g_direct 对照：R1 复现失败 → 挖出 2853/2855 混合工作点伪影（硬伤④：工作点混合）

### 动机与协议（预注册冻结于任何观测前）
2855 遗留缺口：其三谱分解中 g_mlp ≡ g_comp 数学恒等（硬伤①），缺失的独立对照是
**g_direct = [mlp(ln_x + ε·cdir) − mlp(ln_x)]·cdir/ε** —— 在 LN2 输出点直接注入 εcdir，
绕开 LN2 对输入的改写。对比语义：g_direct ≈ g_comp → SwiGLU 本征响应，LN2 改写无关；
g_direct << g_comp → LN2 改写承载响应（"ln2_mediated" 字面含义）；g_comp − g_direct = LN2 改写净效应。
- 协议逐句复刻 2855：SEED=2855 词表（80 词 × 10 类，dW_unit 类方向）、L26-35、EPS=1.0、
  x_in = sain2['same'][l][1] + attn_b2[l]、L13H30 钳制校验臂；每层每词一次 batched mlp
  调用（3 行：ln_x / ln_x+dln / ln_x+ε·cdir）；ratio 除法带 abs(gcm)>1e-6 负分母防护（第六次应用）。
- 预注册：R1 描述性复现校验（max_q |mean_w g_comp − g2855_L26_35| < 0.05，g2855 从 2855 result.json 读入）；
  D1 三分类（swiglu_intrinsic：≥6/10 层 |ratio−1|≤0.3；ln2_rewrites：≥6/10 层 |ratio|≤0.5；else mixed）；
  D2/D3 描述性。

### 结果（phase2860/g_direct/；exec b0e78d2d… / result c9254ae5… / g_direct.npz 8194a128…）
| 量 | L26→L35 谱 | 判决 |
|---|---|---|
| R1 max diff | **7.177**（@L34） | **R1 = false** |
| D1 | n_close=1 / n_rewrite=0 | **mixed** |
| g_comp 均值 | −0.08, 0.01, 0.08, 0.11, 0.22, 0.14, 0.01, −0.03, −0.11, −0.27 | 全层 \|g\|≤0.27 |
| g_direct 均值 | −0.10, 0.03, 0.15, 0.24, 0.49, 0.35, 0.02, −0.12, −0.41, −0.67 | 全层 \|g\|≤0.67 |
| ratio=g_d/g_c | 1.26, 5.16, 1.83, 2.26, 2.24, 2.55, 2.22, 3.75, 3.77, 2.53 | 分母近零，比值失义 |
| dln_norm 均值 | 0.73→0.33（L26→L33），尾部回升 0.60 | LN2 收缩 ~0.5-0.7 |
| dln_cos 均值 | 0.965 → 0.757 | 方向保留度高 |
| g_diff (g_c−g_d) | −0.02, −0.02, −0.07, −0.14, −0.27, −0.21, −0.01, +0.09, +0.30, +0.41 | LN2 改写净效应与直接注入同量级 |
| max_resid | 0.01062 | L13H30 钳制臂通过（≡2855） |

### R1 失败根因（本轮主要产出：word 级诊断 + 单前向探针 + 源码定位，三步闭环）
1. **npz 对比**（_p2860_diag.py）：2855 内部 g_mlp ≡ g_comp 逐位（max 0.000000，恒等缺陷实锤）；
   2855 g_ln 与 2860 dln_norm×dln_cos **逐元素 max diff 0.000000** → 词表/x_in/cdir/LN2 逐位复现；
   g_comp 差异全部落在基线项 out_b（L32-34 差异 75/80 词系统性偏正、单词最大 1056、均值 4.48±6.07）。
2. **单前向探针**（_p2860_probe.py，词 'apple'，同时捕获 LN2/mlp 的 pre-hook 真值）：
   \|x_in_f − ln2in_true\| = **65–420**、\|ln_x_f − mlpin_true\| = 7–53、\|mlp_f − mlpout_true\| = 11–700
   （cdir 投影差单词级 −2.1 ~ +24.5）。"sain+attn" 根本不是 LN2 的输入。
3. **源码根因**（transformers modeling_qwen3.py L315-327）：decoder layer 是 pre-norm——
   `residual = hidden_states` 先保存，attn 输入 = `input_layernorm(hidden_states)` = LN1(x)。
   **self_attn 的 pre-hook 捕获的是 LN1(x)，不是残差 x**。
   故 x_in = sain + attn = **LN1(x) + attn_out（伪残差）**；真实 LN2 输入 = x + attn_out。
4. **结论**：2855 g_comp = [mlp(LN2(伪残差+ε·cdir)) − mlp(LN2(真残差))]·cdir/ε ——
   **混合工作点测量**：扰动点用 hook 重构（伪残差），基线用 hook 真值（真残差），
   差值混入 (伪残差 − 真残差) 的深层 mlp 响应。2853 g_jac 同构（phase2853 L300 `out_b = mlp_b2[l]`
   + L306 `mlp_call(l, x_in+ε·cdir)`），两者数值逐位吻合
   （1.7291/4.5193/7.0677 ≡ 2855 勘误中的"base 窗口 L32-34 = 1.73/4.52/7.07"）。

### 勘误的勘误（对 2855 节第二处窗口勘误的修订）
- "base 窗口 L32-34 = 1.73/4.52/7.07" 实为 2853 g_jac（= 2855 g_comp）@L32-34，
  **是混合基线伪影，不是任何意义上的干净 SwiGLU 增益**。2853"深层放大器（L32-34 增益 ~7）"叙事作废。
- 2853 clamp 态 g_emp（1.68/1.28/0.90 @L29-31）= dout/din 双 hook 实测比值，
  分子分母同工作点语义，**不受混合基线污染，仍然成立**。
- 2853 T2 的 r(g_jac, g_emp) 因 g_jac 伪影被污染，待真残差重测后重算。
- 2855 W1（ln2_mediated）判据仅依赖 g_ln（自洽、逐位复现）→ 形式上成立；但 g_ln 的
  零特异性硬伤（LN2 对切向方向必然收缩）不变。W2 神经元分解与 L1 线性化检验均在
  伪残差工作点上，结论降级为"待真残差工作点重估"。

### 2860 判决语义
- D1=mixed 的机制含义：在自洽基线上 g_comp ≈ g_direct ≈ 0（|g|≤0.7），ratio 因分母近零而失义；
  有效读出是**绝对量级**——SwiGLU 对 ε·cdir 的响应（无论 LN2 改写与否）在 L26-35 均为小量，
  远小于伪影值 7.07。"ln2_mediated" 的原始机制解释（LN2 改写承载大响应）不成立；
  真实图像是直接响应与改写效应都是小量且部分抵消（D3）。
- **2855 final_verdict ln2_mediated/neuron_diffuse 的 g_comp 支柱坍塌**；W1 形式成立但机制解读作废。

### 硬伤④登记：工作点混合（workpoint mixing）
- 制度教训：**hook 捕获值只能与 hook 捕获值差分；显式重构值只能与显式重构值差分**
  （同工作点自洽原则）。任何 "[f(重构点+扰动) − f(hook基线)]" 形式的差分公式在 pre-norm
  架构中都混入 LN1(x) vs x 的偏移，深层可放大至 O(10²)。
- 修正协议（2861 起）：x_in 一律用 post_attention_layernorm 的 **pre-hook 直接捕获**
  （探针已验证捕获可行），禁止 sain+attn 重构。
- 冲击面排查（Grep phase28*.py 全扫）：伪残差重构仅出现在 2853/2854/2855/2860 四脚本
  （2854 为 x_base/x_clamp 双点同构）；2856-2859 双谱普查线全 hook 级差分，安全。
  2854 x_base/x_clamp 的具体差分结构待 2861 复核（其 clamp 增益若同为 hook-vs-重构差分则受染）。

### 接续（2861 候选，MASTER_PLAN II1/II3）
真残差工作点重测：x_in = ln2in pre-hook 真值，重测 g_comp/g_direct/dln 谱 + r(g_jac', g_emp)，
一并复核 2854 差分结构 —— 窗口解剖战线在正确工作点上的真正收束。
（脚本 tests/glm5/phase2860_g_direct.py；探针 tests/gpt5_temp/_p2860_probe.py、_p2860_diag.py）
"""

with io.open(MEMO, 'r', encoding='utf-8') as f:
    n_before = sum(1 for _ in f)

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(SEC)

with io.open(MEMO, 'r', encoding='utf-8') as f:
    n_after = sum(1 for _ in f)

with io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_p2860_memo_check.txt',
             'w', encoding='utf-8') as f:
    f.write('lines_before=%d lines_after=%d appended=%d\n'
            % (n_before, n_after, n_after - n_before))
print('OK %d -> %d' % (n_before, n_after))
