# -*- coding: utf-8 -*-
"""Append Phase 2909 section to AGI_GPT5_MEMO.md (append-only)."""
import hashlib

MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\memo_append_2909.txt')

SECTION = """
## Phase 2909: p 轴谱系稳定性预注册检验——升格被否决（N12） [2026-09-19 08:04]

### 原理
2908 发现 null 内百分位 p = P(score_synth <= score_true) 与 margin 幅值独立（新轴候选），qwen_attn 位于下缘。升格为谱系第二主轴前须检验其对打分定义与层子集的稳健性——按 E9 教训，任何"轴"若依赖任意实现选择即非不变量。本 Phase 把 p 轴放进预注册稳定性网格：4 通道 x [4 个 margin 变体 + 3 个层子集] = 28 配置，每配置 5 独立 seed x 10000 抽样，配 7 个同协议覆盖率审计。

### 预注册（冻结于 execution.json，脚本 SHA256-8 a9f4d1a4）
- 锚 a1：全层 baseline margin/acc == stored（2e-5）；a2：Delta_B == 2905 delta_per_layer（1e-4）；a3：fast margin == margin_of_B 且 acc_score == acc_of_B（1e-12）。
- 变体（全层）：V0_full（2896 族 baseline）；V1_diagin（same 项含对角）；V2_colz（列 z-score 后 V0——层等权化）；V3_acc（LOO 最近邻准确率族）。
- 层子集（V0 打分）：S_front（前半窗口层）、S_back（后半）、S_key（2905 argmax|delta| 单层）。
- 覆盖率审计：变体 rng [2909,100+v]（d=10）、子集 [2909,200+s]（d=6/6/1），40 reps x 4000 draws，pass [32,40]/40，任一 fail => audit_coverage_fail_all_void。
- 判决（冻结）：排序分量——四通道 p_med 序在 V1/V2/V3 与三个子集下精确复现 baseline 序；尾部分量——qwen_attn 全部 7 配置 p_med < 0.05；内部分量——glm4_attn 全部 7 配置 p_med ∈ [0.05,0.95]；order unstable => p_axis_order_unstable；stable & tail & interior => p_axis_second_dimension_confirmed；否则 p_axis_order_stable_edge_cases。

### 结果（零前向，92s）
- 守卫全过：锚 4/4；审计 7/7（V0 40/40、V1 40/40、V2 38/40、V3 39/40、S_front 40/40、S_back 40/40、S_key 40/40）。
- **判决：p_axis_order_unstable——p 轴升格被否决**
- 排序表（p_med）：
  | 配置 | glm4 mlp | glm4 attn | qwen mlp | qwen attn | 序同 baseline |
  |---|---|---|---|---|---|
  | V0_full | 0.380 | 0.168 | 0.473 | 0.026 | （基准 qm>gm>ga>qa） |
  | V1_diagin | 0.391 | 0.184 | 0.484 | 0.027 | **是** |
  | V2_colz | 0.061 | 0.052 | 0.535 | 0.097 | **否**（qm>qa>gm>ga） |
  | V3_acc | 0.874 | 0.378 | 0.855 | 0.735 | **否**（gm>qm>qa>ga） |
  | S_front | 0.296 | 0.117 | 0.760 | 0.140 | 否 |
  | S_back | 0.555 | 0.244 | 0.327 | 0.130 | 否 |
  | S_key | 0.599 | 0.302 | 0.953 | 0.607 | 否 |
- 三分量：order **unstable**（6/7 配置打破）；qwen_attn tail **fragile**（仅 V0/V1 < 0.05）；glm4_attn interior **robust**（7/7 ∈ [0.05,0.95]，最低 0.052@V2——唯一不变分量）。

### 结构发现：qwen_attn 下尾是全层聚合效应
qwen_attn 的 p=0.026（全层 V0）在所有真子集上消散：S_front 0.140、S_back 0.130、S_key（delta 最大层单独）0.607。单独任何层块都是 null 样——下尾地位由全层 cosine 平均把逐层小效应聚合而成。换言之 qwen_attn 的"margin 负值/下缘"不是某一层/层块的属性，而是跨层一致的微小偏移的聚合。V3_acc 下 qwen_attn p=0.735 进一步表明：几何分离度（margin）与分类可用性（acc）在 qwen_attn 通道不同源。

### 硬伤与混杂
- S_key 单层 d=1：margin 在 1 维上退化为符号分离度，sigma 用 std(ddof=1)——1 维审计 40/40 通过，实现可用但语义与高维 margin 不同，排序判定对此解释保守。
- 覆盖率审计参数域仍是标量各向同性合成（V2 列等权化在异方差真实数据上的性质由 38/40 覆盖近似保证）。
- 变体集（V1/V2/V3 + 3 子集）是设计选择，未穷尽打分族；"unstable" 结论只需一个反例，已充分。

### 结论
1. **N12 入账（负结果）**：p 值是定义相对统计量而非谱系不变量；第二轴升格否决；2908 的排序与 qwen_attn 下缘读法严格限定在其声明的 2896 族定义内。margin 幅值层级（L14 主轴）不受影响。
2. **幸存的结构发现**：qwen_attn 下尾 = 全层聚合效应（逐层 null 样、聚合显著）；glm4_attn 的内部地位是唯一打分族不变分量。
3. 方法论：把候选"轴"先过预注册稳定性网格再升格，是 E9 教训（定义相对性）的制度化延伸；本 Phase 三个实现 bug（n1 未定义 / d=1 std 转换 / SCORES 查表）各以清目录重跑处理，最终锚/审计链完整。

### 接续
- 2910 候选：A（主选）**全层聚合效应分解**——qwen_attn 逐层 delta_per_layer 与逐层 margin 响应的符号/幅值剖面（零前向），检验"逐层小效应同号"假说：若各层 margin 贡献一致偏负/偏下，则聚合效应是相干的跨层属性（与 Cmp 竞争机制连接）；若异号抵消则只是平均化伪影。B（备选）头级 per-head W_VO 分解（2909 遗留，从层级推进到头级）。C 前向 SwiGLU 激活级（非零前向）。

### 文件
- 脚本 tests/glm5/phase2909_p_axis_stability.py（a9f4d1a4）
- 产物 phase2909/p_axis_stability/：execution.json 8c6bf1a3 / result.json 2fb42fa1 / p_axis_stability.npz 6dafc809
- Ledger：M2909_p_axis_stability + N12_p_axis_not_score_family_invariant + L14 再精化（2909 p-axis promotion refuted；connects 16）（measurements 48 / errata 9 / negatives 12 / growth 26 / linkage 14，ledger a9c1963f）
"""

before = hashlib.sha256(open(MEMO, 'rb').read()).hexdigest()[:8]
with open(MEMO, 'a', encoding='utf-8') as f:
    f.write(SECTION)

tail = open(MEMO, encoding='utf-8').read()
ok_title = ('## Phase 2909: p 轴谱系稳定性预注册检验——升格被否决（N12） [2026-09-19 08:04]'
            in tail)
n2909 = tail.count('## Phase 2909')
after = hashlib.sha256(tail.encode('utf-8')).hexdigest()[:8]
rep = ('before %s / after %s / title_ok=%s / n2909=%d'
       % (before, after, ok_title, n2909))
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(rep + '\n')
print('OK', OUT)
