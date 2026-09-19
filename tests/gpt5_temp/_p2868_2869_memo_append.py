import io

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'

section = '''

## Phase 2868+2869（2026-09-18）：增长率曲线 v2/v3 —— 密度门控图像确立

### Phase 2868: growth_v2（零前向 0.2s，确定性复跑逐位一致）

**背景**：2867 的 T2 增长率曲线饱和于 B1=1.0 天花板（构造性循环），真正的非同源增量点 B23 未测。2868 补测：B2(1152 因果谱)/B3(10 mlp 谱)/B2w(η²² 加权)/B2s(显著头子集 p<0.01, 43 头)/B23(hstack)/B23z(z-score 块平衡)。

**产物**：`tests/glm5/phase2868_growth_v2.py`（sha 59b059d3 [execution.json] / 59300982 [result.json] / adf5cb98 [growth_v2.npz]），目录 `phase2868/growth_v2/`。数据源 2867 npz + 2864 npz（eta2/p_per_head）。

**判决**：

| 判据 | 观测 | 结果 |
|---|---|---|
| G1 配对 delta（B23 vs B3） | Δ = **−0.1375**（0.7375 vs 0.875），null-delta p95 0.0625 | **false** → causal_no_gain |
| G2 η² 加权 | acc(B2w) 0.35 > acc(B2) 0.325，> null p95 0.1381 | **true** → eta2_weighting_helps |
| G3 稀疏充分性 | acc(B2s) **0.425** > acc(B2) 0.325（43/1152 头，压缩 26.8×） | **true** → sparse_sufficient |
| growth_v2 | (0.7375−0.875)/0.125 = **−1.1** | 天真拼接 = 干扰主导（sublinear_reuse 标签在此失义） |
| B23z 对照 | 0.3375 | z-score 列平衡更糟（每列等权放大噪声维） |

**教训**：①execution.json 缺失 Gen1——2867 在 main() 开头自写 execution.json（含 cc.snapshot 源码快照），2868 初版漏此段，Gen2 补上并按纪律清理重跑，判决逐位复现（确定性验证）。②预注册 growth_label（<0.1 = sublinear_reuse）对负增长情形语义失义，登记为标签局限。

### Phase 2869: fusion_curve（零前向 0.3s）

**设计**：密度匹配融合 F(α) = unit([B3, α·B2s])，α ∈ {0.25,0.5,1,2,4}；**max-α null**（每个置换标签对所有 α 取自己最大值）校正选择偏差。

**产物**：`tests/glm5/phase2869_fusion_curve.py`（sha 433595e8 [execution.json] / 6125b38d [result.json] / b5c65356 [fusion_curve.npz]），目录 `phase2869/fusion_curve/`。

**判决**：

| 判据 | 观测 | 结果 |
|---|---|---|
| H1 fusion_gain | α* = **0.25**，acc = **0.8875** > acc(B3) 0.875，> max-α null p95 0.175 | **true** |
| H2 权重结构 | α* < 1；acc 随 α 单调降（0.25→0.8875, 0.5→0.8375, 1→0.7125, 2→0.5625, 4→0.5） | **density_matched_fusion** |
| H3 growth_v3 | (0.8875−0.875)/0.125 = **0.10**（恰在冻结阈） | **sublinear_reuse（边界）** |

### 核心发现（重复三遍）

**机制信息载体按信息密度分层：mlp 臂 10 维最密（0.875）＞ 因果谱 43 头稀疏子空间（0.425）＞ 全谱 1152 维（0.325）；融合只有在密度匹配权重（1:4）下才有正增益（0.8875），天真拼接是干扰主导（0.7375）。**

增长率曲线有效点序列：B2 0.325 → B2s 0.425 → B3 0.875 → B3+0.25·B2s **0.8875**（有效增量 +0.0125，过 max-α null）→ B23 0.7375（天真拼接干扰）。

**"有限参数→无限能力"的第一个定量支点：新信息块的有效接入方式 = 密度门控（低维高密块主导 + 稀疏块弱权重），而非全谱叠加。** 与门池 9%/骨干 91%（2837-2844）、mlp 间接写出 ~30× 静态通道（2865）三线互证——网络的信息集成几何是门控的、分层的，不是叠加的。

### 接续

- 增长率曲线现状：v2（天真拼接）失义已修正为 v3（密度匹配）= 0.1 边界。曲线需要更多"轴"才有形状——下一功能轴（属性轴/语法轴）接入后每轴一个点。
- 2870 候选：**A（主选）属性轴预研**（反义词对协议 + 400b 资产接入，类别轴×属性轴双轴图谱第一步）；B 语法轴探针（POS/位置）；C B2s 43 头的身份解剖（与 2864 η²-top、2846 top-64 的交集 = "显著头"三重定义统一）。
'''

with io.open(MEMO, 'r', encoding='utf-8') as f:
    before = f.read()

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(section)

n_before = before.count('\n')
n_after = (before + section).count('\n')
print('APPENDED: %d -> %d lines' % (n_before, n_after))
