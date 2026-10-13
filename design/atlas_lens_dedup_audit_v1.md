# 「路线」界面子界面重复度审计 v1（2026-10）

> 范围：`LensProgress`（路线透镜）→ 图谱 tab → `AtlasLens` 全部子界面。
> 性质：**分析报告，未执行任何删除**。删除需用户逐项确认。
> 先例依据：`LensProcess.jsx` 头注释记载的 v5 去重纪律——「技术运行 tab 撤销（ResultsExplorer 结果库已含技术运行能力）」；`AtlasSpace.jsx` 已退役留档。

## 1 现状盘点（AtlasLens 图谱全景 panMode 六模式 + 三 tab）

| 子界面 | 组件 | 行数 | 数据源 | 表达对象 |
|---|---|---|---|---|
| 模块视图（默认） | AtlasModuleView | 307 | atlas_tree_v2 | 三模块 × 模板↔机制双栏对照 + 关联边 |
| 模板空间 | AtlasTemplateSpace | 324 | atlas_tree_v2 | 14 模板 3D 关系空间 + 详情面板 |
| 图形（泳道） | AtlasPanorama | ~200+ | atlas_tree_v1 | 三域泳道 + 特征节点 |
| 3D（立体馆） | AtlasGallery | 355 | atlas_tree_v1 gallery3d | 三域塔×族楼层×特征节点 3D |
| 列表（域树） | AtlasTree | 123 | atlas_tree_v1 | 域→族→特征树 + MEC 过滤 |
| 卡片 | AtlasLens 内联 | — | atlas_tree_v1 | 特征卡网格 |
| 结果库 | ResultsExplorer(mode=results) | 317 | /api/results | 证据档案浏览 |
| 注册表 | RegistryPanel | 158 | distributedData | 模板×技术×研究包资产 |

## 2 重复项认定

### D1（高重复）模块视图 ↔ 域树 AtlasTree ↔ 泳道 AtlasPanorama
- 三者都是「域/模块 → 族 → 特征」的层级表达；AtlasTree 的「域间交叉」段与模块视图的跨模块桥 chip 完全同源（同一 cross_links 数据的两种渲染）。
- 泳道视图与模块视图同为「模块中心」呈现，模块视图上线后泳道的独立信息量仅剩布局样式。
- **建议**：退役 AtlasPanorama（map 模式）与 AtlasTree（list 模式）。若需保留纯文本快速扫描，保留二者之一即可（推荐保留域树做无障碍/导出兜底，删泳道）。

### D2（高重复）模板空间 ↔ 立体馆 AtlasGallery
- 两个 3D Canvas 视图，交互范式（拖拽旋转/缩放/hover/点击选中→共用 FeatureDetail）几乎逐段同构（Gallery 注释自认「与 AtlasSpace 同范式」）。
- 对象不同（模板 vs 特征塔），但项目核心目标是**模板→机制规律**，模板空间是主线；立体馆表达的特征几何已被模块视图右栏 + FeatureDetail 覆盖。
- **建议**：保留模板空间，退役立体馆（gallery 模式）。gallery3d 派生段可留档不删（数据层不动，符合 append-only 纪律）。

### D3（中重复）卡片模式 ↔ 域树/模块视图
- 特征卡网格 = 域树的平铺变体，无独有交互。
- **建议**：与 D1 二选一处理——若保留域树则删卡片，反之亦然。倾向「保留卡片、删域树」：卡片对 E 级着色与语句展示更完整。

### D4（低重复/保留）结果库 ↔ FeatureDetail 抽屉
- 粒度不同：结果库=一次上传（sha 寻址）的证据档案，FeatureDetail=特征级锚点汇总。非重复，保留。

### D5（低重复/保留）注册表 ↔ 模块视图左栏模板卡
- 模板列表出现两处，但注册表是「资产全景」（技术/研究包/成熟度），模块视图左栏是「归位导航」。保留，但模块视图模板卡应加「在注册表查看」跳转而非复制详情。

### D6（无重复）路线透镜「行业进展」tab
- EraTimeline（时间轴）与 PEERS 对比卡（同行横向）叙事轴不同，非重复。

## 3 删除清单（待确认）

| # | 对象 | 动作 | 影响 |
|---|---|---|---|
| 1 | AtlasPanorama.jsx + map 模式按钮 | 删除 | -200 行；泳道 CSS 段清理 |
| 2 | AtlasGallery.jsx + gallery 模式按钮 | 删除（组件留档 git 即可） | -355 行；gallery3d 派生段保留 |
| 3 | AtlasTree.jsx 或卡片模式 | 删除其一（建议删 AtlasTree） | -123 行 |
| 4 | 对应 CSS 段（at-tree-*、at-gal-* 等） | 随组件清理 | 瘦身 rdc_fusion.css |

**净效果**：六模式 → 三模式（模块 / 模板空间 / 卡片），约 -680 行，且每类表达对象只剩一个家（唯一家原则）。

## 4 保留理由摘要

- 模块视图：本轮新设计的核心（模板↔机制双栏 + 关联边一等公民）。
- 模板空间：3D 关系表达的家，服务「跨模板规律」主线。
- 卡片：最低门槛的扫读视图。
- 结果库 / 注册表：协议渲染器与资产面，与展示层正交。
