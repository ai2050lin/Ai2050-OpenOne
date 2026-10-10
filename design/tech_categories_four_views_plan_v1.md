# 技术四分类 × 四界面优化方案 v1（M7 提案）

> 目标重述：分布式研发平台 = 多种语言模板（提取的输入侧）× 多种分析技术（提取的分析侧）→ 特征结果空间。
> 分析技术收敛为四大类：**稀疏分解与字典学习 / 探测与因果干预 / 表示空间分析 / 可视化与自动化解释**。
> 本方案回答：总览、空间、研发、路线四个界面如何围绕这四类技术重组。

---

## 0. 核心思路：四类技术是注册表的一级分类，四个界面是结果空间的四种操作

```
结果空间 = 对象 × 语言模板 × 分析技术（四类）
总览   = 看覆盖度（哪些格子已做、哪些空着）
空间   = 看产物（技术结果如何投影到 3D / 表格）
研发   = 做缺口（空格 → 任务 → 领取 → 上传）
路线   = 看成熟度（每类技术在行业的演进 + 平台的支持进度）
```

纪律红线不变：技术信息全部登记在数据层（distributedData.js），JSX 零字面量；
协议键白名单渲染；LIVE/DEMO 双模；每个结果带 sha / 量纲登记。

---

## P0 数据层（distributedData.js，先行）

### 1. 新增 `TECH_CATEGORIES`（四类技术的一级注册表）

| id | 名称 | 代表方法 | 所需数据契约（input） | 平台状态（demo 初值） |
|----|------|----------|----------------------|----------------------|
| sparse | 稀疏分解与字典学习 | SAE、字典学习、NMF | result.activations（残差流/MLP 激活） | 未实现（装置待建） |
| causal | 探测与因果干预 | 线性探测、激活修补（patching）、定向消融 | result.activations + result.logits | 未实现 |
| repr | 表示空间分析 | RSA/RDM、CKA、PCA、有效秩 | result.cos / result.eta2（已有） | **已实现 3 项** |
| interp | 可视化与自动化解释 | 特征看板、LLM 自动解释、聚类标注 | 前序任一技术产物 + 元数据 | 部分（可视化已有，自动解释未实现） |

每条含：`id/name/methods[]/input[]/status`（status 取值：registered / device_built / measured / planned，与 Q04 的状态机同构）。

### 2. `ANALYSES` 扩容：每项加 `category` 字段

现有 3 项全部归入 `repr`（rsa-rdm / eta2-decompose / cka）。
新增条目（均带 input 契约与 status，未实现的自动置灰）：

- `sae-topk`（sparse）：top-k 特征编码，input=[result.activations]，status=planned
- `probe-linear`（causal）：线性探测得分，input=[result.activations]，status=planned
- `patching-swap`（causal）：激活修补 Δ 率，input=[result.activations_pairs]，status=planned
- `pca-proj`（repr）：主轴投影坐标，input=[result.activations]，status=planned
- `autointerp-notes`（interp）：LLM 自动解释草稿，input=[前序产物 sha]，status=planned

可用性判定沿用现有机制：TechPanel 按 input 契约对当前结果匹配，不硬编码。

### 3. 新增 `DEMO_COVERAGE`（覆盖度矩阵 demo 叙事）

结构：`[{obj, tpl, cat, status}]`，status ∈ done / pending / empty。
真源（P2）= 服务端 `GET /api/coverage`（由结果库按 sha→(对象,模板,技术) 聚合），前端先 DEMO 后 LIVE。

### 4. `RESEARCH_KITS` 扩容

现有 6 kit 全在 repr 类。补四类各 1 个 kit（status=pending_results）：
`sae-topk` / `probe-linear` / `pca-proj` / `autointerp-notes`（× is_a 模板起步）。

---

## 1. 总览（HomeView）：从"队列计数"升级为"技术版图"

1. **hero 数字**：sealed/总数、模板数、结果数之外，加第 4 格「技术覆盖 x/y 类」
   （y = TECH_CATEGORIES 中 status≠planned 的类数，x = 结果库实际出现过的类数）。
2. **新增「技术版图」四卡**：每类技术一张卡——
   名称 + 代表方法 + 已注册技术数 + 覆盖格数/总格数 + 缺口数；点击卡跳数据透镜并预选该类过滤。
3. **新增「覆盖度矩阵」缩略条**：对象 × 模板 × 技术四类的压缩热力条（每格一小方块，
   done 实心 / pending 半透明 / empty 虚线），点空格 → 跳研发透镜的缺口任务视图。
4. 数据源：`TECH_CATEGORIES` + `DEMO_COVERAGE`（P2 后 `/api/coverage`）。

## 2. 空间（LensSpatial）：3D 点云按技术产物切换「投影层」

数据分析模式（现有 3D 视图）加一层 **projection registry**（登记于数据层，组件零字面量）：

| category | 投影层 | 3D 呈现 |
|----------|--------|---------|
| sparse | 特征激活热图 | 点按 SAE 特征激活值着色 + top-k 特征清单 |
| causal | 判别轴 / 干预对比 | 沿 probe 法向投影；干预前后双点云 + 位移箭头 |
| repr | RDM-MDS 布局 / CKA 对齐 | 现有点云切换 MDS 坐标；跨模型对齐线 |
| interp | 自动标注叠加 | 聚类质心挂 autointerp 解释标签 |

交互规则：点云加载后按当前数据契约列出**可用投影**（ANALYSES input 匹配），
不可用的投影置灰并注明"缺 result.activations"之类契约缺口——与 TechPanel 同一套谓词。

实时分析模式（S9 连接本地模型）：协议位预留「实时探测 / 实时 SAE 编码」两个开关，
P2 runner 加载权重后生效；当前显示契约说明（不放假开关）。

## 3. 研发（LensProcess）：缺口 → 任务 → 领取的闭环

1. **任务队列 tab**：任务卡加技术类别徽标（四色 chip），tab 内加类别过滤器；
   现有 /api/tasks 真源不变，类别由任务登记的 analysis 反查 ANALYSES.category 得出。
2. **新增「缺口任务」区**（研发 tab 顶部）：
   - 输入 = 覆盖矩阵的 empty/pending 格；
   - 每条缺口生成任务建议卡：对象 × 模板 × 技术 + 所需 input 契约 + 预计算力（按技术类别查表）
     + 一键复制 `node_agent.py claim` 预填命令；
   - DEMO 叙事先行，P2 服务端 `POST /api/tasks/suggest` 转真。
3. **AGG 汇总按类别分节**：AGG-v1 结果条目按 category 分组渲染，跨类不混排（量纲纪律）。
4. 调度 tab 不动（节点租约层与技术分类正交，不混两层）。

## 4. 路线（LensProgress）：行业演进 + 平台成熟度双轨对齐四类技术

1. **行业进展**：MI_ERAS / RSA_ERAS 保持不变；每条时间轴节点 / 新闻条目加 `cat` 字段
   （归属四类之一），时间轴上方加四类筛选 chips（默认全部）。
   例：Circuit Tracing→sparse+causal；RSA 各阶段→repr；Neuronpedia→interp。
2. **平台进度 tab 新增「技术成熟度」卡组**：四类技术 × 平台状态机
   （registered → device_built → measured），状态与 ANALYSES.status 聚合联动，
   对齐 Q04/Q06 的「装置建成 / 正式测量」叙事；点击卡 → 数据透镜对应技术。
3. **新人教程补四条上手路径**：每类技术给一条「教程步骤 + 推荐首个 kit」
   （复用 TUTORIAL_STEPS 骨架，路径差异只在第 3/4 步的 kit 选择）。

---

## 实施分期

| 期 | 内容 | 改动面 |
|----|------|--------|
| P0 | 数据层：TECH_CATEGORIES / ANALYSES 扩容 / DEMO_COVERAGE / kit 扩容 | distributedData.js |
| P1 | 四界面接入：总览技术版图+覆盖条；空间投影层；研发缺口任务+类别徽标；路线成熟度+筛选 | HomeView 内嵌、LensSpatial、LensProcess、LensProgress、CSS |
| P2 | 服务端真源：/api/coverage、任务建议端点、runner 实跑 SAE/probe（激活级 forward）、autointerp LLM 调用 | distributed_service.py、node_agent、runner |

每期完成后：vite build + dev 200 + node import 冒烟 + 记日志。
