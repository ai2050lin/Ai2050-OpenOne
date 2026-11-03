# 客户端框架：现状 vs v4 方案 vs 综合方案

日期：2026-11-02 · 依据：生产客户端源码逐文件实读 + 后端接口响应结构 + v4 效果稿

**结论先行：不是替换关系，也不该是「v4 补生产」。实读之后结论要收紧——v4 的两个核心卖点在源里已经存在且更严谨，真正的增量只剩两个。**

---

## 一、生产客户端的真实骨架

| 项 | 事实（文件:行） | 判断 |
|---|---|---|
| 路由壳 | `Workbench.jsx` **168 行**：四透镜 + 子导航 + URL 同步全在这一处 | 极干净，是资产 |
| 一级界面 | 进展(neuron/lang/rules) · 空间(live/neuron/lang/response/local) · 研发(probe/pipeline/catalog/census/autotest) · 配置(4 段) | 分组有语义：线之上=研究对象+实验生产，之下=形态相关 |
| 状态 | `workbenchUtil.js` 117 行 + `v8Util.js` 198 行，键名不重叠；**只写与缺省不同的键** | 比 v4 的全量写 URL 更克制 |
| 形态可见性 | `ADMIN_LIVE`（桌面/本机）· `admin` · `localTools`（本机中心） | v4 完全没有 |
| 空态纪律 | Workbench 头注释：「没有数据时显示空态，**不回退演示数字**」 | 硬纪律 |
| 3D 层塔 | `v8/Tower.jsx` **159 行**零依赖 canvas | 已有：旋转/缩放 0.5–4×/多边形命中/`edges` 跨层/`target` 红圈/`peak`/`marks`/`height`。**只有 1 处调用**（`SpaceLens.jsx:211`） |
| 神经元阵列 | `NeuronPane.jsx`：BlockDigest / PickedUnits(8 个起点) / BlockStructure / **UnitGrid 9,728** / CircuitPane / LanguageLinks，5 种着色 | 最成熟界面 |
| **单元卡** | `rdc_fusion/NeuronUnitCard.jsx` **240 行**，已挂载于 `NeuronPane.jsx:264` | **见下节，这是本次最大意外** |
| 注意力实测 | `lang_atlas_service.py`：hit / delta / mass，Bonferroni + 跨类型通用/特异判据**写死在 aggregate()** | 已有可操作判据 |
| 数据集清单 | `GET /api/lang/atlas/datasets` → `_datasets_index()` **按 source 分组**，含 by_category / missing / `testable` / 显式映射说明 | 已存在，但只在语言图谱内用 |
| 真实实测 | `LangBench`：41,240 题逐题前向，逐科目正确率 | 真跑过 |
| 实验台账 | `ResearchHeatmapRoute.jsx` **3,340 行**，C101–C244 共 20+ 条实验路线 | 真实资产，也是最大单体包袱 |

### 意外发现：单元卡已经比 v4 的特征卡强

`NeuronUnitCard` 的 8 个证据段（全部来自 `/api/neurons/unit`，前端不生成任何描述）：

1. 自然语料激活 —— n_tokens、sources、max/min/mean/std、fire_rate、**真实 symlog 直方图（按符号区 ++/+-/-+/-- 堆叠）**、regime_share、排除说明
2. 正极 / 负极例句 —— 激活值、符号区、上下文命中标记、语料来源
3. 直接输出读出 —— 推高 / 压低 token + 通路近似说明
4. **解释检验** —— EXPL-v1/v2，等级 + 置信度 + 文本测试 + 语料检索 + 漏报率 + **失败原因分类**
5. **模板选择性** —— run / 对照 / 读出 / **d(发现) 与 d(留出)** / 阈值 / 入选 / 来源
6. **干预记录** —— 预注册 sha / 段 / 方式 / 必要性 / 充分性 / 范围内因果使用
7. **因果归因** —— ΔL̂ 总/直接/经后续层/逐位置 + 真实均值消融 + **噪声底 z** + 触发位置消融 + BH 显著
8. caveats 三条固定边界提示

另有 `evidence_axes` 三轴（读出/控制/解释）把「有没有证据」压缩成三个词，正是 v4 想做而没做对的事。

**v4 特征卡画的手绘分布图在这里是真实直方图；v4 的「字典保真度」栏在这里是 d(发现)/d(留出) 两列。v4 的「三 tab」是把这条已经纵向排好的 240 行长卡切成三块——那是让步于面板宽度的退化，不是增益。**

### 生产客户端的真缺陷（收窄到 3 条）

**1. 数据集维度没有全局化。**
接口有、`LangAtlas` 在用，但 `picked` 是组件内 `useState`，不进 URL、不出语言图谱。于是「三类池 100% 来自 CMMLU，结论换个数据集还成立吗」在工作台上无法回答。这是当前研究最要紧的问题，也是唯一必须动后端的缺口。

**2. 单元卡缺 Goodfire S1–S9 段。**
`/api/neurons/unit` 的响应字段确认只有 census / template_selectivity / interventions / attribution / trigger / explanations——**没有 S1–S9**。S1–S9 在「研发·神经元测试」工位单独跑、单独落盘。同一个单元要看特征流程，必须换透镜、换语境。

**3. 没有「什么已经站住」的汇总。**
界面反复自陈「分型是描述性分类，不是证据等级」「本机检验结论，不是证据等级」——诚实，但 E1–E4 只活在 MEMO。研究者只能靠记忆拼全局。

历史负担：`App.jsx` 5,204 行 / 266 KB，其外并行着 `StructureAnalysisPanel`(2,659) / `blueprint/appleNeuron`(6 千行级) / `neural_vis` 13 个渲染器 / `LensSpatial`+`NeuronUnitPanel` 一套旧空间视图。v8 是第四代，但不是唯一在跑的。

---

## 二、v4 效果稿的重新评级

### 真增量（只有两个）

| 增量 | 为什么真 | 成本 |
|---|---|---|
| **数据集全局化** | 骨架里确实缺；接口 `atlas/datasets` 已有，缺的只是「提升为全局 + 进 URL」 | 低：1 个组件 + 1 个 URL 键 + bench 加 dataset 参数 |
| **层旁剖面盒** | Tower 有层板但无剖面，点层只能跳走，不能「就地展开」 | 中：Tower 加 `modules` prop，纯增量，1 处调用 |

### 三个幻觉（照搬即产生假数据，违纪）

1. **E0–E4 证据密度色块** —— 每格值无任何接口产出。
2. **三目标记分 2/6、2、NULL** —— 手填。接口能给出普查规模/特征卡数/判据通过数，给不出「6 族里 2 族成立」。
3. **SAE tab 全字段**（L0/重构/unembedding/反例）—— 本机 transcoder **只有 L30 一层**（m=16384 k=64），其余层与模型全空。且生产已有「解释检验」这一更贴项目实际的替代轨，没有理由再开一条。

### 两个工程问题

4. **无形态与权限模型**：v4 假设所有按钮都能点；生产里网页只读、桌面可执行、管理员才能开普查。
5. **会成为第四套 CSS**：现有 `v8.css`(643) + `workbench.css`(100) + `rdc_fusion.css`(953)，v4 自带 12.5–56px 阶梯。正解是**把阶梯注入 v8.css 作 token**，不新增文件。

---

## 三、综合方案

### 原则

1. 骨架不动（`Workbench.jsx` 168 行结构保持）。
2. 维度用现成接口，不新建服务。
3. 已有的不重做——`NeuronUnitCard` 一行不改地留着，只**加一段**。
4. 没有数据源的东西不上屏；空态沿用「不回退演示数字」。

### 改动收敛在 5 个文件

```
改动 1  wb-gctx 全局上下文条          新增组件 + workbenchUtil.js 加 ds 键
改动 2  Tower.jsx 加 modules prop      纯增量，唯一调用点无需改
改动 3  NeuronUnitCard 加一段 S1–S9    追加 section，不动前 8 段
改动 4  bench 接口加 dataset 过滤      唯一后端改动
改动 5  证据台账（替代证据矩阵）        新组件，只读聚合
```

**改动 1 · `<Gctx>`**
插在 `wb-top` 与 `wb-body` 之间，与 `SubNav` 同级。三选择器：数据集 / 模型 / 划分，右侧显示当前数据集证据摘要（进池题数 / 可用题数 / testable）。
数据集列表直接读 `/api/lang/atlas/datasets`；`LangAtlas` 的 `picked` 改为受控 prop（`value`/`onChange`），旧行为不变。
`workbenchUtil.readState/writeState` 加 `ds` 键，沿用「与缺省不同才写」。

**改动 2 · `Tower` 的 `modules` prop**
不传 = 现状零变化（唯一调用点 `SpaceLens:211` 不改）。传 `{attn:{heads}, mlp:{units}}` 时在选中层板右侧画两个透视盒 + 内部点阵，点盒回调 `onPickModule`。复用现有 `P()` 投影，不引入 three.js。

**改动 3 · `NeuronUnitCard` 追加一段**

在第 5 段（模板选择性）之后插 `<section>Goodfire S1–S9</section>`。这里有一个双形态陷阱，必须先说清：

`GoodfireProbePanel` **不走 HTTP**。它用 `nodeCall('GET','/probes')` / `nodeCall('GET','/probes/{id}')` 打桌面客户端本机的 node 服务（`contributeUtil.js:37`），结果落 `research/neurons/goodfire/GF-*.json`（现有 2 个：GF-L30-U80-local.json、GF-L30-U9517-local.json + spec-L30-v0.json）。所以浏览器里没有 bridge 就读不到。

**正解：server 加一个只读端点** `GET /api/neurons/goodfire?model_key=&block=&unit=`，扫该目录返回报告（判据随结果保存，直接透传）。理由：
- 网页与桌面都能用，一份客户端代码
- 桌面端新跑的结果落在同一目录，端点立刻能读到
- 不需要碰 node bridge，也不需要把 node 端能力搬到 server

前端部分：
- 表格结构**照搬 `GoodfireProbePanel` 的「步 / 量 / 本次值 / 判据 / 结论」**，判据来自 `goodfire_probe.CRITERIA`（随结果返回），不是前端常量
- 无结果时：「此单元未跑 S1–S9。去研发 · 神经元测试」+ 一键跳转（`onGo('probe')`），并注明该步需桌面客户端
- **不做 tab**。纵向排下去即可——240 行已证明纵向可读，切 tab 会让人以为证据被藏起来了。

**改动 4 · bench 加 `dataset` 参数**
`/api/lang/bench/run?dataset=CM`。当前是一次跑全量 41,240 题；按数据集过滤后单跑才合理（也顺带解决 71.5% CMExam 不进池却被计入正确率的口径问题）。

**改动 5 · 证据台账**
放弃 E0–E4 色块，改为**按层段 × 机制列出「已落盘的结果文件 + 判据是否随结果保存」**，数据来自 `/api/neurons/index` + `/api/rules/overview` + 结果目录清单。它回答「什么测过」，不回答「多强」——后者没数据源就不假装有。

### 不该做的四件事

1. **不接 Xiezhi**。未下载、无 HF 直链、非商业许可。先把「三类池单源」在界面标清楚，价值高于加第五个数据集。
2. **不删旧栈**。`StructureAnalysisPanel` / `appleNeuron` / `neural_vis` / `LensSpatial` 只做入口收敛，不物理删除——还有在跑的调用方。
3. **不给「优先级」造第四套定义**。沿用 `lang_atlas` 的 hit/delta/mass + Bonferroni（判据已写死），v4 的「干预翻转剂量倒数」留作后续。
4. **不把字阶做成第四套 CSS**。12.5/13/14.5/15/17/20/31/56 作为 token 写进 `v8.css`，旧规则引用变量。

---

## 四、分期与验收

| 期 | 内容 | 验收（可测，不靠目测） |
|---|---|---|
| **P0** | Gctx + `ds` 进 URL + bench 加 dataset | 切数据集后刷新不丢状态；`/api/lang/bench/run?dataset=CM` 的 n_questions == 该数据集 by_category 之和 |
| **P1** | Tower `modules` prop | 不传 prop 时与现状像素级一致（截图 diff）；点盒回调 hs 正确 |
| **P2** | UnitCard 补 S1–S9 段 | 同一单元在 UnitCard 与 GoodfireProbePanel 的 S1–S9 数值完全一致；无结果时显示跳转空态 |
| **P3** | 证据台账 | 台账行数 == 磁盘结果文件数；每条能点回原文件 |

P0 成本最低且直接回答当前最要紧的问题（单源结论的外部效度），建议先做。

---

## 五、一句话

v4 看起来是「新框架」，实读后是两个真增量（数据集全局化、层旁剖面盒）包着三个无数据源的幻觉；生产客户端看起来是「旧界面」，实读后是 168 行洁净路由壳 + 240 行比 v4 更严谨的单元卡 + 一套成熟的 159 行 3D 层塔。

**把两个真增量注进去，其余一律不动——这是唯一能保住两边优点的做法。**
