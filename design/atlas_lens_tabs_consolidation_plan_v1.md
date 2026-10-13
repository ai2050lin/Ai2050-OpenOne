# 路线-图谱界面 tab 梳理方案 v1（AtlasLens v1.5 → v1.6）

日期：2026-10-11 · 状态：待确认
范围：仅路线透镜 · 图谱 tab（AtlasLens）内部结构；研发/总览/空间不涉及。

## 0 梳理对象（6 tab 现状）

| tab | 现内容 | 数据源 |
|---|---|---|
| 图谱全景 | 指标条 + E 级分布条 + 三域泳道（图形/列表双模）+ 详情抽屉 + 最近闭环 8 行 | /atlas/tree + /atlas/ledger |
| 特征登记表 | 特征卡片网格（全量）+ ValueChips + 详情抽屉 + 版本链 select | /atlas/registry |
| 跨线账本 | measurements 表格（唯一 Phase 时间线） | /atlas/ledger |
| 结果库 | ResultsExplorer：结果浏览 + 行内分析（RSA/RDM · η²） | /api/results 等 |
| 注册表 | RegistryPanel：语言模板注册表 × 分析技术注册表 × 研究包下载 × 成熟度 | /api/kits + 词表 |
| 专项视图 | 10 个 /rdc-* 独立页纯链接目录（三组） | 无（静态目录） |

## 1 现状问题（4 处）

- **P1 特征有两家**：全景泳道节点 ≡ 特征登记表卡片（同一 registry 全量两呈现，各带选中入口）。
- **P2 时间线有两家**：全景「最近闭环」8 行 ≡ 跨线账本表格（同一 ledger 两渲染）。
- **P3 专项视图是过渡态**：纯链接目录零交互，单独占 tab，与信息量不匹配。
- **P4 注册表定位模糊**：内容实为"数据产线资产"（模板×技术×研究包×成熟度），tab 名与特征 registry 撞名；客户端特征版本链 v1.5–1.8 滞后研究侧 v1.28（既有挂账）。

## 2 梳理后结构：tab 6 → 4

### ① 图谱全景（默认 · 特征唯一家）
- 三模式切换：**图形**（三域泳道 AtlasPanorama）/ **列表**（域树 AtlasTree）/ **卡片**（原特征登记表网格，含 ValueChips）；三模共享同一 `sel` 选中态与 FeatureDetail 抽屉（三模一态）。
- registry 版本链 select 从工具栏移入全景头部（版本切换只影响特征数据）。
- **删**「最近闭环」列表：指标条"最新闭环 Phase / 账本 n 条"两卡可点击直达账本 tab。
- 底部新增「专项视图」折叠区（`<details>` 收起）：10 页按三组（特征与坐标 / 关系与更新 / 条件与规律），新窗口打开。

### ② 跨线账本（唯一时间线家）
- 不变。

### ③ 结果库
- 不变。

### ④ 模板与技术（原"注册表"）
- 改名 + RegistryPanel 头部加定位注记："数据产线资产（语言模板 × 分析技术 × 研究包 × 成熟度）；特征证据图谱见「图谱全景」"——消除与特征 registry 的撞名歧义。

### 兼容路由
- `?atlasTab=features` → overview（默认切到卡片模式）；`?atlasTab=special` → overview；既有 tree/map 归一保留。

## 3 不变项

- 数据源端点全部不动（tree/registry/ledger/results/kits）；LIVE/DEMO 双模；零实验字面量红线。
- FeatureDetail 抽屉为共享组件，无需改动；LensProgress 外壳不动。

## 4 收益

- tab 6→4；特征单一家（三模一态）；时间线单一家；专项页不占导航位；"注册表"撞名消除。

## 5 落地改动清单（确认后执行）

1. `AtlasLens.jsx`：panMode 三态 `map|list|cards`；卡片模式 JSX（原 features 网格迁入）；features/special tab 撤销；normTab 增加 features/special 映射；指标条两卡加 onClick；底部专项折叠区；版本 select 移位；头注 v1.6。
2. `RegistryPanel.jsx`：卡头定位注记。
3. `LensProgress.jsx`：图谱 tab 副标题微调。
4. `rdc_fusion.css`：清理 at-grid/at-card 系是否随卡片模式迁移保留（迁移则保留）、at-ledger 专项表样式删除、折叠区样式新增。

## 6 挂账（本方案外）

- 客户端特征版本链 v1.5–1.8 → v1.28：跨线施加确认（`_ATLAS_REGISTRY_CANDIDATES` 登记 v1.28 路径）。
- 专项 10 页逐页协议化并入图谱后删除独立路由（原 P2 计划不变）。
