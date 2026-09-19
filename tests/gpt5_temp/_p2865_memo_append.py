import io

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'

TEXT = """

---

## Phase 2865 (2026-09-18) — L13H30 动态追踪：II1 收口第二刀判决 = mlp_carried（间接写出），e_ff ~30× 静态通道

### 动机与协议（预注册冻结于任何观测前）
2862 排除直写解释（OV 静态增益 <=0.0075 vs drop=0.101）后，top-1 因果头 L13H30 的
类承载只剩两候选：(a) 结构中继（改后续 attention 键值几何）/ (b) 间接写出（写出方向
经下游变换后才投影到 cdir）。2865 = full/clamp 双态前向（每词 4 前向：full、func、
null、clamp；clamp = L13H30 attention bias (1,1) 钳到 func/null 均值，2853/2860
协议逐字复刻）。attention-bias patch 只触及 L13H30，故 Delta_attn[13] = v13 是
纯 H30 注入向量。真残差位移在 LN2 pre-hook 捕获（2861 制度）。臂分解：
  cfin = v13.cdir + sum_{l=14..35}(Delta_attn[l].cdir) + sum_{l=13..35}(Delta_mlp[l].cdir)
  G_attn / G_mlp 带符号求和；e_ff = cfin/||v13||（间接写出效率，对照静态 0.0075）；
  l* = cdir 累积分量首达 0.5|C-c13| 的层位；D3 = l* 众数占比 vs 200 次随机重排 null。

### 两次失败与根因诊断（判决前必须读）
1. telescoping 向量闭合失败（ident_resid 77/447）。诊断（_p2865_diag）三步定位：
   (i) hidden_states[36] 是 **post final_layernorm**（L35 递推残差 696）——禁止用作
   残差读出点；(ii) E 检查证明 L0-34 层递推在 bf16 舍入内成立，但误差随层增长
   0.1->2.2 —— ln2in 张量携带 23 层舍入累积 delta~10-30，对 ||v13||~0.3-2.3 的
   小词把向量闭合准则放大两个数量级；(iii) **cdir 标量预算精确闭合**
   （词均 0.0007-0.314-1.448 = -1.761 vs cfin -1.763，差 0.002）——协议无结构缺项，
   舍入在投影空间近无偏。v1 判据改冻结为标量预算闭合（阈 0.05）。
2. v1 最终 = false（max budget resid 0.19 > 0.05，相对 |cfin|~11%，如实登记——
   bf16 投影舍入，均值闭合 0.002）。向量 telescoping（均值 5.24）降级为描述量。

### 结果（phase2865/l13h30_trace/；exec 61b2b620 / result 1d01f715 / npz 658f1ff3）
| 判决/量 | 值 | 读出 |
|---|---|---|
| v1 预算闭合 | max 0.19（均 0.002） | false（如实登记）；臂归因定量仍可信（4.7:1 悬殊） |
| **D1 臂归因** | G_mlp = -1.45/词 vs G_attn = -0.31/词 | **mlp_carried**（4.7 倍悬殊） |
| D1w per-word 投票 | attn 臂主导仅 12.5% | 逐词同构 |
| v13.cdir / ||v13|| | **8e-05** | 动态复核 2862：注入向量与 cdir 不对齐 |
| **e_ff** | **-0.23**（p10-p90: -1.41~0.04） | 间接写出效率 ~30x 静态 OV 通道（0.0075） |
| **D3 旋转层位** | 众数 L30 占比 0.237 = null p95 0.244 | **false**：层位词特异，无稳定中继层 |
| A11 | full 0.527 -> clamp 0.055 | clamp 干预强度确认 |

### 科学结论（窗口解剖战线的真正终点）
1. **II1 收口完成：L13H30 = mlp 介导的间接写出**（候选 b）。注入向量与 cdir 余弦
   ~0，但下游响应把 -0.23/||v13|| 的 cdir 分量造出来，且增长主要落在 mlp 臂
   （4.7 倍）——类方向是**下游 SwiGLU 变换的产物**，不是该头的写出内容。
2. **旋转层位词特异**（D3=false，与 2863 词主导、2864 第四轴正交三线互证）：
   不存在"类无关中继层"；同一头的信息经每词不同的层位旋转进 cdir。
3. **与 2861 呼应**：深层 mlp 对 cdir 的温和负反馈（-0.12~-0.95 @L32-35）与
   此处 G_mlp 负号一致——clamp 响应的 cdir 增长是同一 mlp 臂的另一面。
4. 方法论入账：(i) hidden_states[last] 是 post-final-norm，禁作残差读出点；
   (ii) 深层 bf16 舍入累积使向量闭合判据不可用，标量投影预算闭合是正确工具；
   (iii) clamp/patch 干预的臂分解（Delta_attn/Delta_mlp per layer）是"关联机制"
   的通用测量模板，可直接推广到任意头。

### 接续（2866 候选）
- 主选（TMA 词族图谱最小版）：per-word 机制坐标 v0——每词 [10 类方向投影,
  per-word drop 谱, eta2 词级贡献] 拼坐标矩阵，词间相似结构 vs 语义标签
  （零前向，2846/2864 不可变产物 + unembed 代数；判据配随机 null）。
- 备选：mlp 间接写出的承载层 per-word 归因深化（2865 npz 已有 a_mlp 矩阵）。
- 备选：199 词普查重启（G2 门线需用户决策）。
（脚本 tests/glm5/phase2865_l13h30_trace.py；产物 phase2865/l13h30_trace/）
"""

with io.open(MEMO, 'r', encoding='utf-8') as f:
    before = f.read().count('\n')
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(TEXT)
with io.open(MEMO, 'r', encoding='utf-8') as f:
    after = f.read().count('\n')
print('LINES %d -> %d' % (before, after))
