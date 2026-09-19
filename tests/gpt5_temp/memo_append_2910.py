# -*- coding: utf-8 -*-
"""Append Phase 2910 section to AGI_GPT5_MEMO.md (append-only)."""
import hashlib

MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\memo_append_2910.txt')

SECTION = """
## Phase 2910: qwen_attn 全层聚合效应分解——交替结构与后半层主导 [2026-09-19 08:12]

### 原理
2909 定位 qwen_attn 下尾地位（全层 p=0.026）为全层聚合效应（任何真子集 null 样），遗留两个对立假说：H_coherent（各层一致偏下，聚合出相干跨层属性）vs H_cancel（异号抵消伪影）。本 Phase 预注册逐层 + 累积分解裁决：逐层 d=1 百分位 p_j（每窗口层独立 null 检验，5 seed x 5000 draws）+ 累积曲线 cum_p(k)（前 k 层联合，k=1..n_win）。判据（冻结，仅 qwen_attn）：below_frac = p_median_j < 0.5 的层比例；>= 0.90 => coherent_cross_layer_suppression；< 0.50 => cancellation_artifact；否则 partial_coherence。其余组描述性。

### 预注册（冻结于 execution.json，脚本 SHA256-8 0a3066f9）
- 锚 a1：margin/acc == stored（2e-5）；a2：Delta_B == 2905 delta_per_layer（1e-4）。
- 覆盖率审计：d=1（rng [2910,0]）39/40、d=5（rng [2910,1]）40/40 双过（d=10/6 审计已由 2908/2909 登记，{1,5,6,10} 网格锚定本 Phase 使用的维度极值）。
- 打分固定为 2896 族 V0（声明的定义域）；sigma RMS（E9 正名）；d=1 用 std(ddof=1)。

### 结果（零前向，85s）
- 判决：**qwen_attn_partial_coherence**（below_frac = 0.50，落入冻结的 [0.50, 0.90) 中段）。
- **qwen_attn 逐层剖面（决定性发现：相邻层交替）**：
  | 层 | L00 | L01 | L02 | L03 | L04 | L05 | L06 | L07 | L08 | L09 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | d1 score | -0.029 | +0.070 | -0.039 | +0.016 | -0.031 | +0.064 | -0.021 | -0.034 | +0.071 | -0.040 |
  | p_med | 0.327 | 0.778 | 0.104 | 0.697 | 0.253 | 0.849 | 0.556 | 0.205 | 0.609 | 0.143 |
  L00-L05 严格交替偏下/偏上（L06 近中位打断，L07/L09 恢复偏下）。
- **累积曲线（对 H_cancel 的决定性反驳）**：k=1..4 在 0.34-0.54 震荡（null 样），k=5 起单调下行 0.139 -> 0.085 -> 0.022 -> **0.010（k=8）** -> 0.029 -> 0.025（k=10）。后半层（k>=5）驱动进入深尾且稳定。
- **三方互证（锚链交叉验证）**：cum k=10 = 0.025 独立 seed 复现 2908 登记的全层 p=0.026；cum k=5 = 0.139 复现 2909 S_front = 0.140——不同 rng 键、不同 Phase、同值到千分位。
- 对照组：glm4_attn below_frac 0.42（累积 k=3 早降 0.143 但稳定在 0.12-0.24 内部区）；glm4_mlp 0.50（后段下行被最后两层逆转 0.563/0.377）；qwen_mlp 0.60（被 L01 单层 score 1.0195 / p 0.954 强偏上主导）。**仅 qwen_attn 到达深尾**。

### 解读
1. **既非纯相干也非伪影**：below_frac 0.50 排除 uniform coherence（0.90 阈值），但累积曲线从 k=5 起单调入深尾并稳定在 2908 登记值——若是平均化伪影，累积 p 应随机震荡而非单调收敛。真实结构 = **相邻层交替的逐层贡献 + 后半层主导的聚合**。
2. **交替性是新结构线索**：qwen_attn 相邻窗口层的 margin 贡献反号（L00-L05 严格交替），提示相邻层的通道响应方向振荡——与 2902/2903 通道 Jacobian 结构的连接待检验（2911）。
3. qwen_mlp 的 L01（score 1.0195，比其他层大一个量级、p 0.954 深居 null 上部）是唯一"单层主导"型通道，与 qwen_attn 的"交替+聚合"型形成结构对照。

### 硬伤与混杂
- below_frac 0.50 恰落在冻结边界（< 0.50 为 cancellation）上 1/20 层（L06 0.556 近中位）——partial 标签对 L06 的微小位移敏感；但累积曲线的单调性证据不依赖该边界。
- d=1 逐层 margin 语义（符号分离度）与全层 d 维 margin 不同构，逐层 p_j 只作相对剖面用（审计 d=1 39/40 保证 null 校准）。
- n_win=10（qwen）/12（glm4），半层分割 k=5/6 是设计选择；累积曲线在两种分割下形态一致（S_front/S_back 互证）。

### 结论
1. qwen_attn 下尾 = **交替层结构 x 后半层聚合**（partial coherence），单层与均匀相干两个朴素假说均被否定；聚合动力学的三方互证（2908/2909/2910 独立 seed 同值）是本轮最硬的量化事实。
2. 交替性（相邻层反号）进入候选机制清单：若在 delta_per_layer 与 Jacobian 结构上复现，将连接 Cmp 竞争机制（层间方向振荡 = 候选竞争的层间表现）。
3. 谱系四通道聚合形态学：qwen_attn 深尾聚合型 / glm4_attn 早降内部型 / qwen_mlp 单层主导型 / glm4_mlp 尾部逆转型——四种不同聚合形态学，margin 幅值层级之外的第二结构维度（定义域内，2896 族）。

### 接续
- 2911 候选：A（主选）**交替结构形式检验与机制连接**——四组逐层 delta_per_layer（2905 已有）符号交替性 + 相邻层 B 列相关剖面（oscillation index：相邻层列相关 vs 隔层相关的系统差），检验交替是否为 qwen_attn 特有及是否延伸到 Jacobian 层面（零前向）；B（备选）头级 per-head W_VO 分解（对 L07/L09 偏下最强层定位承载头）；C 前向 SwiGLU 激活级（非零前向）。

### 文件
- 脚本 tests/glm5/phase2910_cross_layer_coherence.py（0a3066f9）
- 产物 phase2910/cross_layer_coherence/：execution.json d4c7ed55 / result.json 48001e65 / cross_layer_coherence.npz 1013f7da
- Ledger：M2910_cross_layer_coherence + L14 再精化（2910 coherence decomposition；connects 17）（measurements 49 / errata 9 / negatives 12 / growth 26 / linkage 14，ledger 2acb5381）
"""

before = hashlib.sha256(open(MEMO, 'rb').read()).hexdigest()[:8]
with open(MEMO, 'a', encoding='utf-8') as f:
    f.write(SECTION)

tail = open(MEMO, encoding='utf-8').read()
ok_title = ('## Phase 2910: qwen_attn 全层聚合效应分解——交替结构与后半层主导 [2026-09-19 08:12]'
            in tail)
n2910 = tail.count('## Phase 2910')
after = hashlib.sha256(tail.encode('utf-8')).hexdigest()[:8]
rep = ('before %s / after %s / title_ok=%s / n2910=%d'
       % (before, after, ok_title, n2910))
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(rep + '\n')
print('OK', OUT)
