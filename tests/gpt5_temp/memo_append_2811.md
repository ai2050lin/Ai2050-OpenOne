
---

## Phase 2811: 名词嵌入叠加格式——"齿轮形状"的普查、跨词预测与效率审计 [2026-09-17 02:11]

**用户锚点（2026-09-17）**："当前破解的是是否有齿轮，但核心目标是破解齿轮的形状——苹果的词嵌入为什么是这样的，怎么表达出水果、食物、公司等含义。很可能有一种系统编码机制，让各种名词的词嵌入在当前参数下，既可以表达复杂的知识结构和属性，又同时极为高效。更本质的目标，需要的是破解所有名词的词嵌入的规则，以及背后的原因。"

### 0. 附件分析：九项"已破解"清单的裁定

用户同轮提供《九项"已破解"内容的详细原理讲解与"完全破解"评估》。裁定：

1. **清单总体成立**。九项（内容存储 / 写入渐进 / 层1换车 / W_U 竞争读出 / 带符号投票 / 对比编码 / 三种编码形态 / 层级剂量+10维效率 / 泛化边界）的机制描述与实验记录一致；其自我评估"框架已建立、内容未破解"（"知道有齿轮，不知道齿轮怎么转"）是对现状的准确概括，本 Phase 即针对"齿轮形状"。
2. **第 8 项须修正（2809 教训）**。其中"嵌套保持 1.000"不是经验发现而是代数恒等式：rank-9 零和空间对 rank-10 构造的保持是 dW_class 零和性的必然结果（2809 已证：1000 次 label shuffle 全部 1.000，null 与观测完全重合，判据不可区分）。"10 维效率"的经验部分只剩"9 维保留类别可分性（acc 不降）"；"嵌套"不得再作为独立证据引用。
3. **与 2810 的接口**。第 8 项的"剂量"叙事（语义调制存在）与 2810 的发现（通用上下文通道淹没语义通道：the 引发最大位移 1.400，same 类方向份额仅为 2.0-2.3× 对照的少数分量）合并后指向同一图景：**类别/语义信息在 hidden state 层面是少数派通道**。本 Phase 在 embedding 层面独立复测这一点（见 §2 效率审计：9%），两层测量相互印证。

### 1. 测试原理

零前向（2808 协议）：直接读 safetensors（embed_tokens + lm_head + model.norm + config rms_norm_eps），无 GPU forward，4.5s 完成。归一化嵌入：

```
z(w) = e(w) / sqrt(mean(e(w)^2) + eps) * g
```

类别方向（2806 atlas 100 词构造，模板-评测严格分离，2807 协议）：

```
Cm[i] = mean_{w in class i} lm_head_row(w)
dW_class = Cm − (sum(Cm) − Cm)/9      # 数值秩 9 的零和对比空间
unitD    = 10 个单位化 dW_class 行
```

每个名词的 10 维类别特征谱（带符号投票口径）：`p(w) = z(w) · unitD^T`。
主类 = argmax p；次级成员 = argmax_{c≠主类} p；普查矩阵 `M[c1][c2]` = 类 c1 词中次级为 c2 的比例。

三臂 + 双 label-shuffle null（2809 教训制度化）：
- **A 普查**：atlas / eval（2807 held 99 词 + NEW 84 词存活）分列算 M；eval label shuffle ×1000（保持类大小）给 per-cell q95 → P-L1
- **B 跨词预测**：仅用 atlas 的次级分布 R[c1][c2] 预测 eval 词的观测次级；atlas label shuffle ×1000 给 p 值 → P-L2
- **C 效率**：enlarged battery 最近质心准确率 acc_k（P-L3）；类子空间能量份额；去类残差有效秩；参与率 PR
- **E 探索性**：大小写词义对（apple/Apple, turkey/Turkey, china/China, japan/Japan），不进判定

预注册判据（冻结于任何 readout 之前，execution.json sha256=25cbe5ae…）：
- **P-L1** superposition_map_real：存在 cell (c1,c2), c2≠c1，eval rate ≥ 0.50 (n_c1_eval ≥ 8) 且 atlas rate ≥ 0.40 且 eval rate > 该 cell null q95
- **P-L2** secondary_systematic：跨词预测命中 ≥ 2/9 且 null-A2 p < 0.001
- **P-L3** battery_generalization：eval 最近质心 acc ≥ 0.75（chance 0.10）
- **verdict** gear_shape_superposition ⇔ P-L1 AND P-L2；P-L3 为电池有效性门

Gates 全过：dW_vs_2807 = 1.82e-09，Z_eval[:99]_vs_2807 = 9.51e-07，atlas/eval 词重叠 = 0，lm_head 与 embed_tokens 绑定（tie=True）。

### 2. 关键结果（result.json sha256=7a56665c0bbce032…）

**判定：gear_shape_superposition = false（P-L1 = true，P-L2 = false）**

P-L3（电池有效性）= **true**：183 词评测集最近质心 acc10 = 0.776（chance 0.10）；acc_k 曲线在 k=10 处从 0.399 跳到 0.776（真类方向不可替代，与 2807 的 k=10 跳变 0.869 同构）。

**P-L1（叠加地图真实）= true——4 个逐 cell 过 null 的真实次级成员 cell：**

| from | to | rate_eval | n_eval | rate_atlas | null q95 |
|---|---|---|---|---|---|
| metal | fruit | 0.786 | 14 | 0.90 | 0.286 |
| food | fruit | 0.611 | 18 | 0.90 | 0.278 |
| furniture | clothing | 0.846 | 13 | 0.90 | 0.308 |
| clothing | furniture | 0.692 | 13 | 0.90 | 0.308 |

**P-L2（全局系统性）= false**：跨词预测命中 0.377（chance 0.111，null median 0.148，q95 0.268），但 p = 0.003 > 预注册阈值 0.001。诚实执行协议判 false。备注：命中为 null median 的 3.4 倍且超过 q95，信号实在而弱，未达冻结阈值。

**效率审计（"极为高效"假设的检验）：**
- 类子空间能量份额 mean = **0.089**：10 维类别方向只承载嵌入能量的约 9%
- 去类残差有效秩 = **282 / 283 词**：残差几乎满秩，类方向之外另有广阔结构
- 参与率 PR：atlas 谱 **1.59**（近 one-hot 尖峰）< eval 2.92 < 随机 token 4.72
- 随机 token 次级强度 q95 = 0.587：名词的次级成员在随机本底之上仍是结构化的

**臂 E（探索性）——本 Phase 最直观的"齿轮形状"演示：**

| 对 | cos | lower 谱主峰 | upper 谱主峰 | 解读 |
|---|---|---|---|---|
| apple/Apple | 0.743 | fruit 0.997 | fruit 0.964 | 词义不变，双成员 fruit+food 微弱共存 |
| turkey/Turkey | 0.630 | **food 0.681 + animal 0.538 双峰** | country 0.975 | **10 维谱干净分解多义词：一 embedding 同时携带"食物+动物"双成员，大写后整体切换为国家** |
| china/China | 0.620 | country 0.947 | country 0.996 | 小写 china 的瓷器义未进入 10 类谱（谱外属性） |
| japan/Japan | 0.663 | country 0.965 | country 0.998 | 同上 |

### 3. 结论（核心发现 ×3）

**核心发现：名词嵌入的齿轮形状 = 近 one-hot 主类尖峰（PR 1.59）+ 微弱但结构化的次级混叠（仅特定类别对，4 cell 过 null）+ 巨大属性残差（91% 能量、有效秩 282/283）。**

重述一：类别信息以约 9% 的能量成本即可实现 0.776 的读出准确率——"极为高效"的第一重证据：类别不是嵌入的主体，而是稀疏索引。
重述二：叠加是局部的、非均匀的——不是所有类别共享子空间，只有语义近邻对（furniture↔clothing）系统性混叠；turkey 谱证明双成员可共存于同一 embedding，混叠机制服务于多义词。
重述三：回答"苹果的词嵌入为什么是这样"的第一近似——主类轴（fruit 尖峰 0.997）只是骨架，91% 的能量在属性/知识流形里，那部分才是"表达复杂知识结构"的载体，也是尚未破解的主体。

### 4. 严格审视：问题、硬伤、瓶颈

1. **P-L2 未达冻结阈值**（p=0.003 > 0.001）。协议必须尊重；但"4 cell 逐个过 per-cell null"与"聚合命中 3.4× null median"两证据合看，全局系统性是"弱而非零"，非纯噪声。
2. **metal→fruit 语义反直觉**：metal 词的次级 90% 是 fruit，大概率不是"语义"而是几何伪影——fruit 方向可能被"自然物/物体"泛类主导（2806 模板 cos 矩阵可查）。若属实，"次级成员"的语义解释须降级为"泛类几何近邻"。这是 2812 必须溯源的第一疑点。
3. **单 token 过滤偏倚**：171 候选仅 84 存活（Qwen tokenizer 将罕见词切开）；fruit 类 NEW 候选 12/12 全灭，fruit 相关 cell 的 eval 证据全部来自 2807 held 旧词。电池偏向常用词。
4. **embedding-only**：无上下文。2810 已证明进入 hidden state 后通用上下文通道主导；本 Phase 给出的是第 0 层（静态词位）的齿轮形状，两者必须拼合才是完整图景。
5. **次级 = argmax(9) 口径噪声**：谱弱时次级身份易翻转（这正是 P-L2 用分布 R 而非逐词 argmax 做预测的原因）。
6. **随机 token 基线含 junk token**：PR 4.72 的对照力有限，不能完全排除"atlas 谱尖峰 = 词频/训练剂量"混杂。

### 5. 智能理论视角的关键洞察（第一性原理）

语言能力 = 有限参数实现无限组合。破解名词嵌入规则 = 回答"一个 2560 维向量如何同时编码 (a) 离散类别身份 (b) 连续属性 (c) 多重成员"。2811 给出的第一性分工假说：

- **离散身份用"轴"编码**：log2(10) ≈ 3.3 bits 的信息，用 9-10 个零和对比方向、约 9% 能量实现，读出只需 W_U 竞争（带符号投票），无需前向计算——这是高效的部分，且天然支持"对比编码"（第 5 项）。
- **属性用"场"编码**：91% 能量、近满秩残差——属性是连续、可组合、开放集的，必须稠密表示。这与第 7 项"三种编码形态"中的场形态衔接。
- **多重成员用"混叠"编码（superposition）**：但按需分配——语义近邻类别共享子空间以省容量，无关类别近正交以避干扰。turkey 是直接标本：food+animal 双峰共存于小写形，大写形整体切换 country。叠加不是全局压缩技巧，而是容量分配策略。
- **两层测量的一致性**（embedding 9% vs hidden state 少数分量 2810）：类别是"稀疏索引"、属性流形是"主体"，这可能是 LLM 表示的普遍架构原则，而非某一层的巧合。

**瓶颈**：10 个类别只是齿轮的最粗分辨率。"苹果为什么是这样"的完整答案需要属性级方向（可食、圆形、红色、植物、公司品牌……）。类子空间 9% 意味着 91% 的形状完全未解释——下一步必须把 unitD 从"10 类方向"推广为"属性方向库"。

### 6. 接续（2812 候选，按优先级）

1. **P-L4 属性子空间假说（主任务）**：从 atlas 构造属性对方向（big/small、edible/inedible、alive/manmade、round/long……，≥12 对），测属性子空间能量份额与残差解释率。若属性方向解释率 >> 9%，则"残差 = 属性流形"成立，齿轮形状问题转化为"属性方向库的完备化"问题。
2. **metal→fruit 溯源（第一疑点）**：对 fruit dW 方向做 top-k unembed 投影 + 与 2806 全部模板方向 cos 分析，判定 fruit 方向是否被"自然物/物体"泛类污染。
3. **2810 通道分离**（Δ_specific = Δ_condition − mean(Δ_func, Δ_null)，预注册判据照旧）。
4. **多层谱演化**：把 10 维谱从 embedding 推广到逐层 hidden state（2807 载体头工具箱可复用），观察"主类尖峰 → 属性场"的逐层重构路线。

### 7. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2811_noun_superposition.py` sha256 = `02ac94d39db3d70c85024dea0ec21522844c2fdc5dacbf95575d52d9bbe8513d`（修复版）
- `tests/glm5/result/rdc_query_construction_20260913/phase2811/noun_superposition/execution.json` sha256 = `25cbe5aeef76e03f88aa7a420fb0d44f7787f88a138075041868ecc2575e50ac`
- `…/result.json` sha256 = `7a56665c0bbce0325cac1555d98b7b9c41799eab9ea0076f8c000c3dc5a5430a`
- `…/battery.npz` sha256 = `78a16913f472f84e187df8511e756c082fa6000a26ef174f2a76a5ef212b837a`

**Gens 记录**：首次运行崩溃于 line 427（投影公式维度写反：`Qb @ (Qb.T @ Zc)`，Qb 为 2560×10、Zc 为 283×2560；正确式 `(Zc @ Qb) @ Qb.T`）。崩溃发生在 fc.save 之前 → 零产物（陈旧 execution.json 已删）。判据未动，纯机械修正。重跑 4.5s 干净完成，elapsed 计时以第二次运行为准。
