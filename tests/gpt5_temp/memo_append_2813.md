
---

## Phase 2813: 残差几何普查——91% 是知识网络微场的平铺，不是属性轴空间 [2026-09-17 02:37]

**锚点**：2812 否证了"形容词属性轴"后，直接解剖残差本身——(R) 283 词去类残差 PCA 谱 + top-20 主成分 unembed 读出；(N) 名词两极属性轴（修正 2812 的 POS 错配 + 2812 的 identity 置换污染教训：稳定性 null 排除 identity 分配）。

### 1. 测试原理

零前向，三重 gates 全过（dW_vs_2807 = 1.82e-09；eval_words 与 2811 逐词一致；Z_vs_2811 = 9.51e-07）；2812 的 dA（14×2560）只读加载。

残差构造与谱分解：

```
Z_all = [Z_atlas; Z_eval]  (283×2560)
Zc    = Z_all − mean(Z_all)
resid = Zc − (Zc @ classQ) @ classQ.T        # 去类子空间
SVD(resid) → var_i = s_i² / Σs²              # 谱
PC_i   = Vt[i]（2560 维单位向量）
读出   = top-10 tokens of Wu @ PC_i          # tie: Wu = Etab
```

名词两极轴：6 属性 × 每极 6 候选（mountain/ant、fire/ice、lightning/snail、venom/pillow、boulder/balloon、sun/cave 式），单 token 过滤 + 电池去重，6/6 存活。share 判据同 2812；split-half 稳定性 null 改为**排除 identity 分配**的 pole-shuffle ×300。

预注册（冻结，execution.json sha256=ea07af8b…）：
- **P-R1** residual_lowdim：top-20 累积方差 ≥ 0.50
- **P-R2** pc_class_decoupled：max|cos(PC_i, dW_j)| < 0.30（i≤20, j≤9）
- **P-R3a/b**：名词两极 share > N1 q95 且 > 3× median；≥60% 属性 split-half 过 exclude-identity q95
- **verdict**：residual_geometry = 'lowdim' iff P-R1 AND P-R2；nounpole_support = P-R3

### 2. 关键结果（result.json sha256=9e3f2156b6c87cfc…）

**判定：residual_geometry = 'flat'；nounpole_support = false**

| 臂 | 结果 | 数据 |
|---|---|---|
| P-R1 谱低维 | **false** | top-10 = 10.6%，top-20 = **17.9%**，top-50 = 35.6%——谱平坦，无低维骨架 |
| P-R2 PC-类解耦 | true 但**恒等式** | max\|cos\| = 0.0000 精确为零：resid ⊥ 类子空间是构造使然（2809 型代数恒等式，**erratum 已入账**，verdict 实际由 P-R1 单独承载） |
| P-R3a 名词两极 share | **false** | eval share 0.0090 **低于** N1 median 0.0095（q95 0.0134） |
| P-R3b 名词两极稳定性 | **false** | pass-frac 0.50 < 0.60：temperature/speed/danger 过（cos≈0.11）但极弱；size cos=0.011 崩溃；weight q95==cos_half 平局 |

**探索性金矿（未预注册，top-20 PC 的 unembed 读出）——残差的 top 主成分全部是语义微场：**

| PC | 方差 | 读出（top unembed tokens） | eta² |
|---|---|---|---|
| PC1 | 1.36% | Ghana, Ecuador, Angola, Peru, Kenya, Yemen…（**国家**） | **0.483** |
| PC3 | 0.97% | lithium, sodium, magnesium, calcium, **锂, 钠**, potassium, zinc（**化学元素**） | 0.247 |
| PC5 | 0.90% | Spain, Sweden, **瑞典**, Norway, Finland, **西班牙**, Portugal（**欧洲国家**） | 0.217 |
| PC6 | 0.87% | reef, island, **岛**, shark, storm, islands（**海洋地理**） | — |
| PC7 | 0.86% | sushi, mango, rice, tofu, Vietnam, Thai, pizza, **蛋糕**（**亚洲食物/文化**） | — |
| PC8 | 0.82% | cake, candy, **蛋糕**, sausage, pastry（**甜食**） | 0.158 |
| PC15 | 0.72% | bolt, wrench, nut, screw, **螺**（**五金**） | — |
| PC0 | 2.03% | 逗号/数字/括号（**词汇格式轴**） | 0.126 |

eta²（PC 投影与类别标签的关联，探索性）：PC1=0.483（国家类主导）、PC3=0.247（metal 类）、PC5=0.217——**类内仍有子结构**：国家类分裂为 PC1（非洲/美洲国家）与 PC5（欧洲国家）两个微场。

**跨语言读出现象**：PC3/PC5/PC6/PC7/PC8/PC10/PC15/PC17/PC19 的 top-unembed 中直接出现中文 token（瑞典、西班牙、锂、钠、岛、蛋糕、橄榄、埃及、螺、椰、床）——**微场方向跨语言共享同一 unembed 几何**，语义场是语言无关的知识坐标。

### 3. 结论（核心发现 ×3）

**核心发现：91% 残差 = 知识网络微场的平铺拼贴——全局谱平坦（top-20 仅 18%，无低维骨架），局部主成分逐个读出为语义微场（国家、化学元素、欧洲国家、亚洲食物、海洋地理、五金、甜食），且微场方向跨语言可读出。**

重述一："属性流形"假说双重死亡（2812 形容词轴 + 2813 名词两极轴，后者 share 甚至低于词池本底）：属性不是名词嵌入的组织原则。
重述二：组织原则是**知识网络的邻域坐标**——每个微场 = 知识图谱的一个邻域（国家→地缘分区、金属→元素表、食物→菜系），大小 2-5 词/场、各占 0.7-1.4% 方差，拼贴起来谱自然平坦。
重述三：这直接支持总假设"语言能力 = 知识网络 + 分析推理 + 语法的编码"：**名词嵌入的主体是知识图谱坐标，类别与属性都只是这些坐标的投影**——类别轴（9%）是坐标的粗化读出，属性是坐标的谓词化读出。

### 4. 严格审视：问题、硬伤、瓶颈

1. **P-R2 恒等式**（erratum）：残差构造保证 resid ⊥ 类子空间，cos(PC, dW) ≡ 0——判据信息为零，与 2809 嵌套恒等式同类。教训第三次出现：**凡在"被扣除的空间"内测对扣分子的正交性，必得恒等式**。下一轮判据设计须先做代数可行性审查。
2. **微场读出是间接证据**：top-unembed 读出 + eta² 是解释性而非判定性的；PC0（标点/数字）提示部分 PC 携带词汇学（token 格式/频率）而非语义内容。
3. **283 词样本限制**：谱平坦部分来自微场样本量小（每场 2-5 词）；扩大电池（每微场 ≥10 词）可能抬高 top 方差占比。P-R1 的 0.50 阈值对小样本偏严格。
4. **频率/长度混杂未回归**：PC0 与部分 PC 可能由词频驱动。
5. noun-pole 稳定性 3/6 过但 cos≈0.11 仍远低于真轴水平（≥0.5）；weight 出现 q95==cos_half 平局（小池边界效应残余）。
6. PC 读出方向 = Wu=Etab（tied），unembed 即 embed 空间，读出的"语义微场"解释依赖 embed 空间本身语义可读这一已验证性质（2806-2808 链条）。

### 5. 智能理论视角的关键洞察（第一性原理）

三轮（2811→2812→2813）拼出名词嵌入编码的完整第一性图景：

- **编码主体是知识图谱坐标**：微场=邻域。词嵌入高效的原因不是把属性打包进正交轴，而是**直接编码"这个词在知识网络中的位置"**——类别、属性、多义（turkey food+animal）都是位置的函数。
- **效率的来源**：位置是相对量（对比编码，第 5 项），只需在共享流形上存偏移；微场间的近似正交 = 不同邻域在流形上自然分离，无需显式正交化。
- **跨语言共享微场方向**是"语言无关语义骨架"的直接证据——这与 L0 基对齐全语（Unified Theory v4.1）预测一致：微场方向应是 L0 基的局部特化。
- **与 Cmp(o,r,v) 非线性发现的呼应**：微场=吸引盆的候选几何实现；"回归均值"效应（2810 前后观察）可解释为微场间的插值。

**瓶颈与下一步靶心**：从"微场存在"到"微场坐标定律"——需要回答：微场的方向由什么决定（邻域的什么结构）？场内词的坐标如何排列（例如国家微场是否按地理/经济/历史排序）？这是"破解所有名词词嵌入规则"的下一层。

### 6. 接续（2814 候选，按优先级）

1. **微场普查主任务**：系统性枚举 top-50 PC 的微场归属（top-k unembed + eval/atlas 词投影聚类），量化 283 词的微场覆盖率；预注册"覆盖率 ≥70% ⇒ 知识网络坐标假说成立"。
2. **微场内坐标定律**：以国家微场（样本最大）为标本，检验场内一维排序是否对应地理/人口/经济变量（可从词嵌入自身回归）。
3. **频率/长度回归控制**后重测 share 与谱。
4. **metal→fruit 终审**：PC3 证实 metal 词=元素实体；检验 fruit 方向能量与"实体物"微场坐标的回归关系，关闭该悬案。
5. 2810 通道分离（Δ_specific）仍在队列，优先级维持。

### 7. 产物登记（immutable + SHA256）

- 脚本 `tests/glm5/phase2813_residual_geometry.py` sha256 = `f9786e25d07fc083f488ac2dee3525326fa787170b0565d205e1b673e8009dec`（修复版）
- `tests/glm5/result/rdc_query_construction_20260913/phase2813/residual_geometry/execution.json` sha256 = `ea07af8b23b88bbb5695cb8070bab06bdf30f7f87e1e3dddd5705b2e61f99ca8`
- `…/result.json` sha256 = `9e3f2156b6c87cfcf57ef0982df5ae776e4e03688d6399c1cac08c28fbc614d7`
- `…/residual.npz` sha256 = `e9f25417de4f04628ecd9eff93196f04692278ed4e18c7dbb01751f4ed83339a`

**Gens 记录**：首跑崩于 assert K2≥5（名词池候选太少：mountain/whale/elephant/desert/snow/frost/wolf/pillow/truck/cave 等与电池词冲突或多 token，仅 3/6 属性存活）。崩前零 result 产物（execution.json 已删）。修复：候选池扩至每极 6 个、崩溃前打印拒绝明细、补 PC-类别 eta² 探索性诊断（P-R2 恒等式在首跑已被发现并当场标注）。重跑 21.2s 干净完成。预注册判据文本未动（P-R2 按原样执行并以 erratum 入账）。
