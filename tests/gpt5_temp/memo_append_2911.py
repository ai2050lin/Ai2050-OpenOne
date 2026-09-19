# -*- coding: utf-8 -*-
"""Append Phase 2911 section to AGI_GPT5_MEMO.md (append-only)."""
import hashlib

MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\memo_append_2911.txt')

SECTION = """
## Phase 2911: 交替结构形式检验——margin 层面真实且 qwen_attn 特有，载体定位到类间符号平衡 [2026-09-19 08:36]

### 原理
2910 发现 qwen_attn 逐层 d=1 margin 贡献奇偶交替（L00-L05 严格）。两个待解问题：(1) 交替是否统计真实（非噪声读法）；(2) 交替的载体在哪个层面——响应方向（B 列相关结构）、均值移位（delta 符号）、还是别的。本 Phase 三探针预注册检验：P1 margin score 符号翻转率（逐层 d=1 score 相邻符号翻转，精确二项尾）；P2 delta 符号翻转率（2905 delta_per_layer，零对排除）；P3 列相关振荡指数 osc = mean_corr(相邻列) − mean_corr(隔一列)（10000 次列置换 null，单侧 p = P(perm <= obs)）。

### 预注册（冻结于 execution.json，脚本 SHA256-8 7793c93a）
- 锚 a1：全层 margin/acc == stored（2e-5）；a2：Delta_B == 2905（1e-4）；**a3：重算逐层 d=1 score == 2910 score_true（1e-6，实测 maxabs 5.0e-7）**——跨 Phase 产物互锚。
- 校准审计（rng [2911,0]）：200 个 iid 57x10 正态矩阵 x 1000 置换——置换 p 的 [0.05,0.95] 覆盖率须在 [0.80,0.97] 且中位 p 在 [0.40,0.60]。实测 frac=0.88、median=0.505，pass。
- 判决（冻结）：qwen_attn 需 P1 p<=0.05 且 P3 p<=0.05；特异性由显著负 osc 集合 S 决定（S={qwen_attn} => specific；S ⊆ {两 attn} => channel_shared；|S|>=3 => generic；其他 partial）。

### 结果（零前向，6s）
- **判决：alternation_not_confirmed_margin_only**
- 探针表：
  | 组 | P1 margin flips | P2 delta flips | P3 osc | P3 perm p |
  |---|---|---|---|---|
  | glm4 mlp | 6/11 p=0.500 | 6/11 p=0.500 | +0.062 | 0.864 |
  | glm4 attn | 6/11 p=0.500 | **0/11 p=1.000（全同号）** | +0.054 | 0.849 |
  | qwen mlp | 6/9 p=0.254 | 3/9 p=0.910 | −0.030 | 0.342 |
  | **qwen attn** | **8/9 p=0.0195** | 5/9 p=0.500 | **+0.050** | **0.728** |
- P1 确认：margin 层面交替真实（8/9，p=0.0195）且 qwen_attn 特有（他组 p>=0.25）。P3 否定向列相关结构的延伸：osc 为正（相邻列相关反而更高），S 空集——原始响应跨层平滑。P2 delta 符号随机。
- 附带发现：glm4_attn delta 符号 0/11 翻转（全同号块，反向尾 p~0.0005）——其 delta_per_layer 全正。

### 载体定位（diag_2911c，post-hoc 描述性）
2910 的 d=1 margin 本质是**符号分离度**：(57,1) 矩阵行归一化后每元素 = 响应符号，margin_j = P(符号一致|同类) − P(符号一致|异类)。用符号直接重构 marg 与 2910 score_true 完全一致（校验通过）。逐层剖面：
- 类间符号平衡 gap_j = |pos_frac(类0) − pos_frac(类1)|：qwen_attn 序列 0.08/0.27/0.08/0.16/0.09/0.32/0.14/0.03/0.24/0.05——**锯齿 7/8**，与 margin_j 强对应（gap 大 → margin 正：0.27→0.070、0.32→0.064、0.24→0.072；gap 小 → margin 负：0.08→−0.029、0.03→−0.034、0.05→−0.040）。
- 驱动侧：pf0（类 0 正率）范围 0.32-0.86 宽幅波动，pf1（类 1）0.37-0.60 相对平稳——**交替主要由类 0 正响应率的层间波动驱动**。
- 跨组同律：glm4_attn L08/L09 gap 0.37/0.41（最大）→ margin 0.273/0.276（最大）——gap→margin 定律跨组成立。
- diag_2911b 教训（指标设计错误）：对 |delta| 幅值序列用符号翻转检验恒 0 flips（无信息）；且幅值保留归一化与 2910 的符号化 margin 是不同泛函——诊断指标必须与目标量的定义同构。以 diag_2911c 修正。

### 结论
1. **交替真实且特有，但不在信号层面**：margin 交替（P1 p=0.0195）不伴随响应方向（P3 osc 正）、均值移位（P2 随机）的振荡——**载体是每层的符号分离质量（类间符号平衡）的锯齿**：相邻层的类间符号混杂交错度一好一坏交替。
2. gap→margin 定律跨组成立，把 2910 的"聚合形态学"统一到符号平衡语言下：qwen_attn 深尾聚合 = 其 gap 锯齿在最全层组合下把低 gap 层（margin 负贡献）和高 gap 层（正贡献）按 2896 族 cosine 泛函非线性聚合的结果。
3. 机制连接修正：2910 假说的"层间方向振荡（候选竞争的层间表现）"不成立（P3）；替代假说——类 0 响应符号平衡的层间波动（pf0 宽幅 vs pf1 窄幅）指向**类 0（语言 0）在相邻层的响应符号翻转集合变化**，词级归因待 2912。

### 接续
- 2912 候选：A（主选）**符号平衡锯齿正式化 + 词级归因**——预注册 gap 锯齿检验（词级符号置换 null）+ lag-1 自相关符号检验 + 驱动词识别（逐词符号翻转剖面：哪些词的响应符号在相邻层翻转驱动 pf0 波动；词 x 层符号矩阵，零前向）；B（备选）头级 per-head W_VO 分解（L07/L09 层）；C 前向 SwiGLU（非零前向）。

### 文件
- 脚本 tests/glm5/phase2911_alternation_structure.py（7793c93a）
- 产物 phase2911/alternation_structure/：execution.json 0ab48eaf / result.json 61a08678 / alternation_structure.npz 8c791cbd
- 诊断 tests/gpt5_temp/diag_2911b.py（指标设计错误，教训入账）-> diag_2911c.py（修正后符号剖面）
- Ledger：M2911_alternation_structure + L14 再精化（2911 alternation formal tests；connects 18）（measurements 50 / errata 9 / negatives 12 / growth 26 / linkage 14，ledger c8a0d016）
"""

before = hashlib.sha256(open(MEMO, 'rb').read()).hexdigest()[:8]
with open(MEMO, 'a', encoding='utf-8') as f:
    f.write(SECTION)

tail = open(MEMO, encoding='utf-8').read()
ok_title = ('## Phase 2911: 交替结构形式检验——margin 层面真实且 qwen_attn 特有，载体定位到类间符号平衡 [2026-09-19 08:36]'
            in tail)
n2911 = tail.count('## Phase 2911')
after = hashlib.sha256(tail.encode('utf-8')).hexdigest()[:8]
rep = ('before %s / after %s / title_ok=%s / n2911=%d'
       % (before, after, ok_title, n2911))
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(rep + '\n')
print('OK', OUT)
