# -*- coding: utf-8 -*-
"""Append Phase 2908 section to AGI_GPT5_MEMO.md (append-only)."""
import hashlib

MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\memo_append_2908.txt')

SECTION = """
## Phase 2908: qwen_attn M1 边界高精度判定（20000 抽样 x 5 seed） [2026-09-19 07:52]

### 原理
2907 遗留最脆弱一环：qwen_attn margin_full = -0.019459 落在 M1 区间 [-0.019820, 0.063102] 内但下界富余仅 0.0004，且 diag_2907b 显示其判定对 seed/流结构 MC 敏感（rms 下 inside 2/4、below 2/4）——400 抽样的区间端点 MC 误差 ~0.002-0.004 与富余同量级。本 Phase 把判定从"端点比较"升级为直接百分位检验 p = P(margin_synth <= margin_full)，抽样质量 400 -> 20000 x 5 独立 seed（单臂 50 倍、总量 250 倍于 2907 主阶梯），并配同协议覆盖率审计。p 的 SE = sqrt(0.025*0.975/20000) ~ 0.0011，把"below/inside/下缘"的分辨精度压到 3SE ~ 0.0033（概率尺度）。

### 预注册（冻结于 execution.json，脚本 SHA256-8 9f034700）
- 锚 a1：margin/acc 4 组复现（abs 2e-5）；锚 a2：Delta_B 重算 == 2905 delta_per_layer（abs 1e-4）；锚 a3：fast_margin（预计算 mask 批内）== margin_of_B（abs 1e-12）。
- 覆盖率审计（rng [2908,0]）：40 重复，真参数（d=10, n=57, split 22/35, mu0 ~ 0.3N, delta 0.5unit, sigma ~ U(0.3,1.0)），RMS sigma_hat 估计，20000-draw 95% 区间覆盖真 margin；pass [32,40]/40。
- 主判据（冻结，仅 qwen_attn）：5 独立 seed（rng [2908,10+k]）各 20000 抽样，p_median 跨 seed；p_median >= 0.028313（0.025+3SE）=> qwen_attn_inside_m1_confirmed；p_median <= 0.021687（0.025-3SE）=> qwen_attn_below_m1_confirmed；否则 => qwen_attn_at_m1_edge（真值恰在自身各向同性 null 的 2.5 分位附近，协议精度内不可分辨）。其余三组描述性。
- sigma 定义固定为 RMS（2906 prereg 文本 / 2907 实现，E9 已勘误正名）。

### 结果（零前向，38s）
- 守卫全过：锚 4/4（fast_margin 与 margin_of_B 差精确 0.0；dLB max 4.9e-5）；覆盖率审计 **40/40**。
- 判决：**qwen_attn_at_m1_edge**
  | 组 | margin_full | p_median（5 seed） | p 范围 | q2.5/q97.5（中位） | 判读 |
  |---|---|---|---|---|---|
  | glm4 mlp | 0.0198 | 0.3754 | [0.3707, 0.3808] | -0.0063 / 0.0705 | 深居区间内 |
  | glm4 attn | 0.1213 | 0.1676 | [0.1669, 0.1684] | 0.0933 / 0.2198 | 区间内（E9 修正后 firmly inside） |
  | qwen mlp | 0.1799 | 0.4741 | [0.4723, 0.4814] | 0.0980 / 0.2849 | 近居中 |
  | **qwen attn** | **-0.0195** | **0.0276** | **[0.0262, 0.0295]** | **-0.0199 / 0.0686** | **恰在 M1 下缘** |
- 关键读法（三条，等价陈述）：(1) **"below M1"被高精度否定**——5/5 seed 全部 p>0.025，真实 margin 在每个 seed 的名义 95% 区间下界之上；(2) 名义 95% 水平下 qwen_attn 5/5 inside，2907 both-M1 结论确认；(3) 按 3SE 保守带判定，p_median=0.0276 距 confirmed-inside 阈值 0.0283 仅 0.0007（<1 SE）——**真实位置恰在自身各向同性 null 的 ~2.8 百分位，即 M1 下缘，协议精度内不可分辨"恰在下缘"与"略高于下缘"**。
- seed 间极差 0.0034 ~ 3 SE_P，跨 seed 波动与二项统计吻合——判据无残余结构噪声。

### 新谱系维度：null 内百分位 p 独立于幅值排序
p（相对自身类内散布的标准化位置）排序：qwen mlp 0.474 > glm4 mlp 0.375 > glm4 attn 0.168 > **qwen attn 0.028**；margin 幅值排序：qwen mlp 0.180 > glm4 attn 0.121 > glm4 mlp 0.020 > qwen attn -0.019。两轴不同：glm4 attn 幅值居二但相对位置居三（其类内散布 sigma 0.0059/0.0078 为四组最小，margin 0.121 相对散布偏下缘方向）；qwen attn 幅值垫底且相对位置也在下缘——唯一"双轴皆末"通道。幅值谱系（L14 主轴）与新 p 轴互补：幅值 = 语言信息总量，p = 语言信号相对噪声的显著性。

### 硬伤与混杂
- 3SE 保守带判定使 qwen_attn 停留于 at_m1_edge 而非 confirmed_inside——这是刻意的诚实保守（避免把 0.6 SE 的差距当作确证）；名义 95% 水平的 5/5 inside 事实同时登记。
- p 值轴的语义解释（"显著性 vs 信息量"）目前是假说性框架，未预注册检验——若要升格为谱系第二主轴需专门 Phase 预注册（跨模型/跨通道稳定性检验）。
- 覆盖率审计参数域与 2906/2907 相同（d=10, n=57），未覆盖极端 SNR/类大小失衡域。

### 结论
1. **2906 通道分裂问题正式关闭**：E9 修正 sigma 定义后，glm4 attn firmly inside（p=0.168），qwen attn 5/5 名义 inside、精确定位 M1 下缘（p=0.0276）——四通道各向同性 summary 充分性成立（qwen attn 带下缘保留标记）。2906 的 amplitude_law_not_established 与 2907 的 shape_correction_mixed 两个冻结标签均已被高精度判定取代为可解释结论。
2. **谱系新轴**：null 内百分位 p 与 margin 幅值独立；qwen attn 是唯一双轴皆末通道。
3. 方法论：端点比较 -> 直接百分位检验 + 3SE 保守带 + 同协议覆盖率审计，是把 borderline 判定工程化的标准流程；fast_margin（预计算 mask）把 120 万次 margin 计算从预估 5-8 分钟压到 38 秒。

### 接续
- 2909 候选：A（主选）**p 轴谱系稳定性预注册检验**——把 null 内百分位升格为谱系第二主轴：跨 margin 家族变体（无对角 Sm/列 z-score 变体/acc 判据）、跨 seed 域、跨窗口层子集重算 p，检验四通道 p 排序的稳定性与 qwen attn 下缘地位的稳健性（零前向）；B（备选）头级 per-head W_VO 分解：定位 qwen attn 下缘/负 margin 的层 x 头来源（逐层 delta_per_layer 已有，2905）；C 前向 SwiGLU 激活级归因（非零前向，eps=1.0 协议）。

### 文件
- 脚本 tests/glm5/phase2908_qwen_attn_boundary_precision.py（9f034700）
- 产物 phase2908/qwen_attn_boundary_precision/：execution.json 4a3587e3 / result.json dff824f8 / qwen_attn_boundary_precision.npz 553ab55b
- Ledger：M2908_qwen_attn_boundary_precision + L14 再精化（2908 high-precision adjudication；connects 15）（measurements 47 / errata 9 / negatives 11 / growth 26 / linkage 14，ledger 0f413c96）
"""

before = hashlib.sha256(open(MEMO, 'rb').read()).hexdigest()[:8]
with open(MEMO, 'a', encoding='utf-8') as f:
    f.write(SECTION)

tail = open(MEMO, encoding='utf-8').read()
ok_title = ('## Phase 2908: qwen_attn M1 边界高精度判定（20000 抽样 x 5 seed） [2026-09-19 07:52]'
            in tail)
n2908 = tail.count('## Phase 2908')
after = hashlib.sha256(tail.encode('utf-8')).hexdigest()[:8]
rep = ('before %s / after %s / title_ok=%s / n2908=%d'
       % (before, after, ok_title, n2908))
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(rep + '\n')
print('OK', OUT)
