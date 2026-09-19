
---

## Phase 2812: 属性子空间假说否证 + metal→fruit 法证清洗 [2026-09-17 02:27]

**锚点**：2811 遗留两问——(a) 91% 残差是否 = 属性流形（"齿轮形状"的主体）；(b) P-L1 cell metal→fruit 是否为"泛类几何伪影"。

### 1. 测试原理

零前向（2808 协议，tie 感知：2811 已证 lm_head 与 embed_tokens 绑定）。评测电池按 2811 精确重建，三重 gates 全过：dW_vs_2807 = 1.82e-09，eval_words 列表与 2811 逐词一致，Z_vs_2811 = 9.51e-07（< 1e-4 float32 容差）。

**属性轴构造**：15 个双极属性（size/weight/temperature/wetness/hardness/speed/brightness/loudness/cleanliness/sweetness/danger/value/age/strength/sharpness），每极 4 个形容词锚点词，单 token 过滤 + 与电池词表去重后须每极 ≥2 存活；sweetness 全灭（sugary/syrupy/honeyed multi-tok），14/15 存活，K=14。方向取 embedding 侧质心差：

```
dA_k = unit( mean_z(pos 锚点) − mean_z(neg 锚点) )
QA   = orthonormalize({dA_k})           # 2560×14
share(Z, Q) = mean_w |z_w·Q|² / |z_w|²
```

**P-S1/S2 法证**：dW_fruit 正投影占比、fruit 与 food/animal/nature 方向的 cos 对照非对角 q95、dW_fruit 的 top-10 unembed 词、以及"泛类清洗"——从 dW_fruit 中减去非水果评测词均值方向 ĝ 后重算 4 个 P-L1 cell：

```
ĝ = unit(mean_{w: 非fruit} z_w)
dW_fruit_rem = unit(dW_fruit − (dW_fruit·ĝ)ĝ)
```

预注册判据（冻结于任何 readout 之前，execution.json sha256=3f918b11…）：
- **P-L4a**：mean eval share > N1（pole-shuffle ×1000）q95 且 > 3× median
- **P-L4b**：≥60% 属性的 split-half |cos| > 其 N2（×300 pole-shuffle）q95
- **P-S1**：非水果词正投影占比 ≥0.60 且 trio cos > 非对角 |cos| q95
- **P-S2**：metal→fruit 与 food→fruit 清洗后 rate ≥0.50 且 > N3 逐 cell shuffle q95
- **verdict**：attribute_manifold_support = P-L4a AND P-L4b；P-S1/S2 仅诊断

### 2. 关键结果（result.json sha256=ee45b45cdf289e30…）

**判定：attribute_manifold_support = false（P-L4a = false，P-L4b = false）——经典双极形容词轴粒度上，91% 残差不是属性流形。**

| 臂 | 结果 | 数据 |
|---|---|---|
| P-L4a 属性子空间真实 | **false** | eval share 0.0219 vs N1 med 0.0194 / q95 0.0251（两项均未过）；随机 token share 0.0239 **反高于名词** |
| P-L4b 轴稳定性 | **false** | 0/14 属性通过；最高 speed 0.353；7/14 q95==cos_half（identity 置换污染，通过在构造上不可能），7/14 严格失败（错误配对比真配对更一致，如 weight swap 0.052 > true 0.033） |
| P-S1 fruit=泛类轴 | **false** | pos_frac 0.759 ✓ 但 trio_cos = **−0.058**（远低于 off_q95 0.178）；**top-10 unembed 全是纯水果词**（peach, apple, cherry, pear, berry, banana, grape, mango, lemon, orange）——方向语义干净 |
| P-S2 清洗后存活 | **true** | metal→fruit 0.643（q95 0.214）✓；food→fruit 0.500（q95 0.222）✓；对照 cell furniture/clothing→fruit = 0.0 ✓ |

三重关键读数：① pole-shuffle null median（0.0194）是各向同性理论值（K/2560 = 0.0055）的 3.5×——任何词池质心差方向都自带 ~2% 能量（词池共性几何）；属性轴只比这个本底高 **0.25pp**，特异属性内容近乎为零。② 名词电池对属性轴的投影（0.0219）不高于随机 token（0.0239）——名词根本不偏好形容词对方向。③ per-attr top 词跨属性大量复现（uniform/van/star/level/cap/boot/lead/mill/mat），提示锚点方向携带的是词频/长度类通用特征而非语义属性。

**P-S1/S2 联合裁定**：fruit dW 方向是语义干净的对比方向（unembed 读出纯水果），metal→fruit 不能被泛类分量解释（清洗后 0.786→0.643 仍强存活）——**2811 P-L1 的 metal/food→fruit 是真实的跨类内容，但为什么金属词携带 fruit 方向能量而家具/服装词完全没有（0.0），机制未解**。

### 3. 结论（核心发现 ×3）

**核心发现：名词嵌入残差的 91% 不组织在经典双极形容词轴上——形容词锚点方向的特异解释力 ≈ 0.25pp 能量，轴的 split-half 一致性 0/14，名词对它们的偏好不高于随机 token。**

重述一："属性流形"假说在单方向粒度被否证：若属性以 1D 方向编码于名词嵌入，share 应远超词池本底与随机 token，实测皆否。
重述二：属性编码（若存在）必须是分布式的/子空间式的，或与名词指称几何分离（形容词-名词跨词类几何不共享）——1D 探针给出的是下界，且这个下界已归零。
重述三：fruit 方向法证给出正面样板：干净的类对比方向 = top unembed 纯类词 + 对语义近邻低 cos（零和对比空间的自然性质）+ 跨类能量选择性（metal/food 有、furniture/clothing 无）——"齿轮"的真实形状比"泛类轴 + 噪声"精细得多。

### 4. 严格审视：问题、硬伤、瓶颈

1. **P-L4b null 设计缺陷**：小词池（≤4 词）时 identity 置换混入 null 分布顶端，q95==cos_half 使"通过"在构造上不可能（7/14 属性）。该子判据的 false 部分是设计伪影——但即使剔除这 7 个，另 7 个严格失败且全体 cos_half ≤ 0.353（真轴应 ≥0.5），实质否证不受影响。下轮须用 exclude-identity 或 leave-one-out 设计重注册。
2. **POS 错配**：形容词锚点 vs 名词电池。形容词嵌入携带谓词/语境几何，可能与名词指称几何分离——负结果可能是跨词类伪影而非名词内无属性结构。修正方案：用名词两极（mountain/ant 式）构造属性方向。
3. **1D 属性模型下界**：属性若以子空间编码（每属性 >1 维），单方向探针系统性低估；需 PCA 式多极方向。
4. **频率混杂**：per-attr top 词复现通用高频 token，锚点方向部分携带词频/长度特征；下轮应回归掉 log 频率再测。
5. K=14 轴对微小效应功效有限——但效应 ≈ 0 而非小效应，否证是干净的。

### 5. 智能理论视角的关键洞察（第一性原理）

2811 问"齿轮形状"，2812 回答了"齿轮不是什么"：**残差不是人类命名的属性轴张成的空间**。组合 2811+2812 的三层证据（类轴 9% + 形容词属性轴 ~0.25pp 特异 + 残差近满秩 282），名词嵌入的编码必定是以下之一或其组合：
- **分布式子空间码**：每属性占多个方向（如 5-20 维），单方向探针不可见；
- **指称几何与谓词几何分离**：名词嵌入编码的是"指称对象的知识网络坐标"（本体论位置），而非可名状的谓词轴——与用户总假设"知识网络"一致：嵌入可能直接编码知识图谱式的相对坐标，属性只是其投影；
- **非线性/流形码**：属性以弯曲流形而非线性方向存在（与 2790+ 的 Cmp 非线性发现呼应）。

第一性原理推论：**LLM 词汇编码的"高效"不来自正交轴打包，而来自把知识网络坐标（相对位置）作为主编码，属性和类别都是其低维投影。** 这把下一阶段的靶心从"找属性方向"转为"刻画名词嵌入的内在坐标系"——残差 PCA + unembed 读出是最低成本的下一步。

### 6. 接续（2813 候选，按优先级）

1. **残差 PCA 主任务**：对 283 词去类残差做 PCA，top 20 主成分逐一 unembed 投影读出 + 与 14 属性轴/dW 方向 cos——直接回答"91% 里是什么"。
2. **名词两极属性方向**（修 POS 错配）：mountain/ant、whale/mouse、fire/ice 等名词对构造属性轴，重测 share 与稳定性（预注册含 exclude-identity null）。
3. **频率回归控制**：share 测试前回归掉 log 词频与向量范数。
4. **metal→fruit 逐词表 + 实体假说**：metal 词逐词 fruit 投影排序；检验"元素/实体物"假说（metal 与 fruit 共享"自然实体"知识网络坐标，furniture/clothing 是人造功能物故无）。

### 7. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2812_attribute_subspace.py` sha256 = `f344c1fe9b8fea1a9396af5f3106a1013d0eb1781bc8f443498bb5821cc57449`
- `tests/glm5/result/rdc_query_construction_20260913/phase2812/attribute_subspace/execution.json` sha256 = `3f918b11cd4dd78ae41e294de877cc04bd52c16943d025963d03e9534bf37686`
- `…/result.json` sha256 = `ee45b45cdf289e307910aee9245bd8cd34ab79441a7b909262615a69933b5162`
- `…/axes.npz` sha256 = `a9926a6c5c8f513cf4309e34e85295b7ee3deb6627455477f137ca8066d54425`

**Gens 记录**：无崩溃。首写版本在编译前经静态审查修掉三处隐患（tie 感知 lm_head 回退防 KeyError；split-half 奇数切分重叠；null 循环 zrow 重复计算），随后首跑干净完成（8.0s）。判据全程未动。
