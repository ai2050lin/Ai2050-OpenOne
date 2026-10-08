# 界面解耦优化方案 v1 —— 协议驱动 UI（Protocol-Driven UI）

日期：2026-10-08 ｜ 状态：**M2 已落地**（D1-1/D1-3 + 五件协议组件 + 数据透镜 = SummaryBrowser + 平台进度 AGG 实时分桶卡）；M3 待排期
上游：`design/distributed_platform_plan_v1.md`（S2–S6）· `deploy/dist_templates_seed.py`（TM-01..06）

> M2 落地注记（2026-10-08）：组件实际协议以 `deploy/dist_runner_tm.py` 为准——
> `eta2_by_factor`/`per_layer` 键 = **因子名**（corpus.factors）+ `interaction`；`fingerprint` = 每类一组
> `{n_within, sep_out, spread_within}`。前端五件组件在 `rdc_fusion/protocolComponents.jsx`，
> 数据透镜 `LensData.jsx`。

## 0 核心原则

**协议（模板 + 产出 schema）是唯一稳定契约，界面是协议的通用渲染器。**

加一种句型/条件轴 = 注册一个新模板（纯数据，`POST /api/templates`），不改任何界面代码。
界面上不允许出现「TM-04 专属」「位置轴专属」的字段名或分支——组件只认识
`factors[]`、`eta2_by_factor{}`、`per_layer{}` 这类协议键。

判据（验收红线）：新模板注册后，不改动任何 JSX，平台进度的模板矩阵、
数据界面的结果表、空间界面的因子视图必须自动出现并正确渲染它。

## 1 数据层缺口（服务端补三件）

| 编号 | 接口 | 现状 | 需补 |
|---|---|---|---|
| D1-1 | `GET /api/templates/{tm_id}` | 缺 | 单模板详情：contract + corpus 元数据（analysis/factors/layers/items 计数），前端字段表直接渲染 |
| D1-2 | `GET /api/agg/{tm_id}` | AGG-v0 只回显分桶 | v1：桶内 per-factor η² 的 median+IQR、跨节点符号一致性、贡献节点数（仍是分桶回显，不做跨桶平均） |
| D1-3 | `GET /api/results` | 已有 tm_id/model 过滤 | 加 `summary 摘要字段平铺`（eta2_by_factor、status），供表格直接渲染 |

## 2 通用组件（前端五件，全部协议键驱动）

1. **TemplateCard** —— 渲染任意 tm_id：名称 / dim / 因子对 / 结果数 / 聚合状态。
   平台进度模板矩阵已通用化（liveTpls ← `summary.templates`），抽成组件复用。
2. **FactorEta2View** —— 任意 factorial 模板的 η² 因子条形图 + per_layer 深度折线。
   只认识 `factors` 数组，不认识 topic/frame 还是 entity/position。
3. **FingerprintBadge** —— `spread_within / sep_out` 徽标（指纹门可视化）。
4. **CellCosGrid** —— cell 方向 cos 矩阵热图（npz.cos），任意因子组合通用。
5. **SummaryBrowser** —— 结果表：行=一次上传（tm_id/model/kind/seed/sha/η² 摘要），点击展开 summary 原文。

## 3 各界面映射（当前 → 目标）

| 界面 | 现状 | 目标（解耦后） |
|---|---|---|
| 路线·平台进度 | ✅ 本轮：协议卡（目标/流程/格式/首轮结果）+ 模板矩阵通用 | 加 D1-2 聚合统计的 LIVE 渲染；「首轮结果」卡改为读 `/api/agg` 真值 |
| 路线·业界新闻 | 对比卡登记视角差 | 不变（与协议无关） |
| 空间 | 静态 3D 演示 | **特征目录**：按 维度×因子×层位 浏览全部模板聚合产物；入口 = TemplateCard 筛选器（M3） |
| 研发 | AI 自动研发流程 | **AI 模板生成器**：提示词 → LLM 产出 corpus JSON（factors+items+fingerprint）→ 本地校验（网格 balanced、target∈text、指纹词表存在）→ admin 提交注册；运行配置对句型无感知（M3） |
| 数据 | 静态卡 | **结果浏览器** = SummaryBrowser（M2，D1-3 到位即做） |
| 当前机器进度 | 本机横幅 | 渲染本机 Agent 当前 claim 的任意模板（TemplateCard 复用） |

## 4 分期

- **M1.5（已完成 2026-10-07）**：TM-04/05 因子模板 + runner v2（factorial/多层/指纹）+ 对齐式播种 + 协议卡。
- **M2**：D1-1/D1-3 + SummaryBrowser + FactorEta2View（数据界面先通用化）。
- **M3**：D1-2 聚合统计 + 研发界面模板生成器 + 空间特征目录 + 多模板对比视图。

## 5 首轮测量登记（qwen3-4b 单节点，2026-10-07，非结论）

- **TM-04 末层 η²**：句式 0.525 > 交互 0.304 > 知识 0.171；指纹类内等角 spread 0.010–0.029、类外 sep 0.046–0.057。
  末层句式效应偏大疑含「下一 token 句法规划」混杂 → TM-04 v2 候选改动：layers 加 mid。
- **TM-05**：位置 η² 随深度 0.152→0.524→0.515（增后平台），实体 0.766→0.279→0.292（浅层最强）。
  「末层位置≈0」预注册预测**被证伪**；浅层实体效应混杂 token embedding（小明≠小红词不同）。
  cell cos 矩阵：同位置跨实体 cos +0.17 均值、跨位置强负（−0.2~−0.66）→ 末层呈位置对比结构。
  TM-05 v2 候选改动：npz 保存全部层位的 cell 方向矩阵 + 显式「同实体跨位置对齐度」统计量；干预门（移位/换名）才是内容寻址的终审。
- 两轮 v2 改动都走「升 version → design_sha 变更 → 结果不跨版混比」的既有纪律。
