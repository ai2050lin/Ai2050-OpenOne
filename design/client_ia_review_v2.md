# 客户端界面系统梳理 · v2（目标对齐视角）

> 2026-10-19 · 评审基准：`RdcFusionWorkspace`（四透镜 + 总览）、`LensSpatial` v?（2 顶层 × 5 模式）、
> `LensProcess` v5.1（三栏 + 5 GATES）、`LensProgress` v8（2 tab）、`AtlasLens` v1.11（4 tab × 全景 4 模式）、
> 专项页 ×10、`/legacy` 归档页。证据全部实测自代码与注册表，不凭记忆。
>
> 与 `client_mission_ia_review_v1.md`（2026-10-11）的关系：v1 的 P1–P4 **已落地**
> （总览改图谱驾驶舱、通知 badge 接队列、本地进度 tab 撤销、atlas tab 收敛为 4）。
> 本文件只谈**剩余错位**，不重复已完成项。

---

## 0 判据：把项目目标翻译成 5 条可判定的尺子

项目目标（README / MEMO 口径）：**顶级 LLM 逆向工程——破解语言编码机制；方法论主产物 = 语言机制图谱**
（三域 KN/AR/GR → 语言族 → 特征 FTR-xx → E 级证据；证据门链依序判定；协议唯一契约；LIVE/DEMO 不伪造）。

| 判据 | 含义 | 判定方式 |
|---|---|---|
| **D1 主产物优先** | 图谱是核心产物，≤1 次点击可见 | 数点击深度 |
| **D2 对象唯一** | 同一特征/对象可跨界面流动，不换宇宙 | 数跨透镜跳转边 |
| **D3 证据可判** | E 级、门链、反证在界面上可读可下钻 | 数证据维度在 UI 的落点 |
| **D4 协议唯一** | 分类/词表从 API 派生，无第二事实源 | 查硬编码词表 |
| **D5 诚实** | 无写死假件；离线即空态 | 查常量与 placeholder |

---

## 1 现状结构树（实测）

```
/  · /rdc-fusion  →  RdcFusionWorkspace
├─ 全局件
│  ├─ 顶栏：logo · 模型选择器〔写死 qwen3-4b，title 自称「演示选择器」〕· 特征源选择器〔接 registry ✅〕
│  │        · 面包屑 F#3734 · ⌘K〔对象/任务/论文〕· 通知铃〔接 queue ✅〕· 头像
│  ├─ 底部状态条：GPU chip〔写死 "GPU 21.3/24G"〕· sealed chip〔LIVE ✅〕· ticker〔LIVE 优先 ✅〕
│  └─ 分享按钮〔window.alert 占位〕
├─ 总览 HomeView（= 图谱驾驶舱）
│  ├─ hero：使命一行 + 三统计（队列 sealed / 测试模板 / 结果条目）
│  ├─ 主体：AtlasPanorama 三域全景（只读）
│  └─ 侧栏：进行中（/ai-rnd/queue）· 最近闭环（/atlas/ledger）· 快速入口 · NewbieGuide
├─ 空间 LensSpatial
│  ├─ 顶层 2：数据分析（研究包） / 实时分析（本机 HF 目录，POST /api/live/connect）
│  └─ 模式 5：层平铺 StackMode / 神经元级 NeuronMode / 封存激活3D NeuronSpace / 特征点云 CloudMode / 参数热图 ParamMode
│      └─ 实时分析的「层平铺」= LiveStage「参数星河·光塔」（5 种显示视图）
├─ 研发 LensProcess v5.1（三栏）
│  ├─ 左：AI 配置（折叠） + 研究线队列（唯一队列）
│  ├─ 中：Composer + GATES〔证据缺口/冻结契约/执行实验/独立复核/证据回写〕 + 运行区（工件 + 实时事件）
│  └─ 右：PlatformOps（平台概览 / 节点与租约 / 模板矩阵 / AGG 分桶）
├─ 路线 LensProgress v8（2 tab）
│  ├─ 行业进展〔默认 tab〕：MI 时间轴 + NOW 实时论文流（/api/news）· RSA 时间轴 · 对比卡 ×5
│  └─ 图谱 = AtlasLens v1.11（4 tab）
│      ├─ 图谱全景（默认）：指标条 · E 级分布条 · 4 模式（模块视图 / 模块空间 / 模板空间 / 卡片）· FeatureDetail 抽屉
│      ├─ 跨线账本：measurements 表（唯一 Phase 时间线）
│      ├─ 结果库：ResultsExplorer（/api/results + 技术运行内核）
│      └─ 模板与技术：RegistryPanel（模板 × 技术 · 研究包 · 成熟度）
└─ 专项页 ×10（/rdc, /rdc-prefix, /rdc-relation, /rdc-joint, /rdc-operator,
               /rdc-law, /rdc-binding, /rdc-update, /rdc-query, /rdc-construction）
   /legacy  → App.jsx 旧版 3D 主界面（无入口链接）
```

**做得好的地方（不要在重构中破坏）**：LIVE/DEMO 徽标贯穿各透镜；账本/结果库/注册表三处同源；
跨线账本已成唯一 Phase 时间线；研发三栏分工清晰；总览已无首屏假数据。

---

## 2 诊断（按严重度）

### A 级 · 会误导结论（口径错，不是样式问题）

**A1 证据等级存在三套互不一致的口径，实测已造成 3/26 条特征在首屏"隐形"。**

实测 `atlas_registry_v1_32.json`（最新版，26 条特征）的证据等级分布：

| 等级 | 条数 | taxonomy 声明 | `AtlasLens.ECLASS` | `AtlasModuleSpace.E_COLOR` | 首屏分布条 |
|---|---|---|---|---|---|
| `E2_predictive` | 17 | ✅ | ✅ | ✅ | 显示 |
| `E1_repeatable` | 3 | ✅ | ✅ | ✅ | 显示 |
| `E3_causal_scoped` | 3 | ✅ | ✅ | ✅ | 显示 |
| `E2_held_out` | 1 | ✗（未声明） | **✗ 缺** | ✅ | **不显示** |
| `E2_behavioral` | 2 | ✗（未声明） | **✗ 缺** | **✗ 缺** | **不显示** |

- 分类法字段 `evidence_level_taxonomy` 是 **4 级**：`E0_candidate / E1_repeatable / E2_predictive / E3_causal_scoped`
  （实测自 `tests/glm5/phase3162_g5a1_atlas_foundation.py:83`，该字段经 `/api/ai-rnd/atlas/registry` 返回，但**前端从不读它**）。
- `AtlasLens.jsx:32` 的 `ECLASS` 硬编码 3 个键；`FeatureDetail` 对未知等级回落 `at-e1`
  → 上述 **3 条 E2 子类特征的徽标会显示成 E1 的颜色**（把"可预测/行为级"误读成"仅可重复"）。
- 首屏「E 级分布条」（`AtlasLens.jsx:193`）只遍历 3 个键 → **26 条里有 3 条不在分布条上**，
  且这 3 条恰好是 E2 的细分档（held-out / behavioral），是最该被看清的档位。
- 根因不是"少写两个键"，而是**分类法由协议给、界面另抄一份**；registry 实际已在用 5 个值（E2 带子档），
  协议自己也没跟上。

这是项目自己定义的"第二事实源腐化"的典型案例。

**A2 使命方法论的主视图（十道门）在现役界面里没有家。**
证据门链（行为资格 → 仪器 → 重复 → 选择性 → 充分性 → 必要性 → 负控制 → 未见预测 → 自然生成 → 跨模型）
是全项目最核心的判定工具，但：
- 现役 5 个界面里**没有任何一处**出现这些词（全仓仅 `blueprint/AtlasControlDashboard.jsx` 有另一套 L0–L8 旧词表）；
- `FeatureDetail` 只给 `anchors[]` 平铺列表（role + tags + sha8）与 `counter_evidence`，**没有"过到第几道门"**。

结果：看一个特征，你只能读到"它有哪些锚点"，读不到"它在门链上停在哪一步"——而这正是判断能不能采信的关键。

**A3 首屏仍有写死假件（与 D5 冲突）。**
| 位置 | 内容 | 性质 |
|---|---|---|
| 顶栏模型选择器 | 固定 `qwen3-4b`，title 自述"演示选择器" | 死件 |
| 底部状态条 | `qwen3-4b bf16 · GPU 21.3/24G` | 写死常量 |
| 顶栏「分享」 | `window.alert(...)` | 占位 |
| 行业进展对比卡 | 点击 `window.alert(note)` | 占位 |
| 特征点云 | 「以 X 为中心」「E_ar(k) 方向叠加」两个按钮 = alert | 占位 |

（另有 2 处 alert 是合法用途：复制领取命令的提示、研究包下载失败提示，不算占位。）

### B 级 · 结构摩擦（影响研究效率）

**B1 对象只单向流动，`v1-S1` 未闭环。** 实测跨透镜边只有 `FTR → 研发透镜` 一条；
缺 `FTR → 空间点云定位`、`FTR → 结果库该特征`，结果库也没有回流到特征/空间。
顶栏对象（`F#3734`，object registry）与图谱特征（`FTR-xx`，atlas registry）仍是两套宇宙，
只在顶栏"特征源选择器"处并排出现，**没有建立映射关系**。

**B2 路线透镜的默认落点与使命错位。** `LensProgress` 默认 `view='news'`（行业进展），
图谱要再点一次；而导航名"路线"暗示时间线，实际 80% 内容是图谱。
`route→图谱` 是使命主产物的第二跳，与 D1 有摩擦。

**B3 "URL 即状态"只覆盖 4 个键。** 实测 `syncURL` 只写 `lens / obj / atlasTab / atlasSel`；
空间透镜的「数据分析|实时分析」与 5 个模式、研发的三栏视图、图谱全景的 4 个模式**都不在 URL**
→ 即使把「分享」按钮做出来，也分享不了"我现在看的这一屏"。

**B4 空间透镜存在同名不同物与近义命名。**
- 「层平铺」在数据分析下 = `StackMode`（36 层玻璃盒横排），在实时分析下 = `LiveStage`（竖塔光塔）——**同一个标签，两种完全不同的视觉与数据口径**；
- 「神经元级」（NeuronMode：单层内部 16384 点，来自研究包）与「封存激活3D」（NeuronSpace：neuron3d 封存件）
  都是"神经元层"，区别在数据来源而非粒度，标签没表达出来。

### C 级 · 可用性细节

**C1 总览三统计是"生产计量"不是"使命计量"。** `sealed/总数`、`测试模板数`、`结果条目` 反映的是流水线跑了多少，
而不是**图谱长成什么样**：三域覆盖度、E 级分布、跨模型一致特征数、剩余缺口数——这些一个都不在首屏。

**C2 缺口没有一等视图。** `gaps` 只在 `AtlasModuleView.jsx:293-302` 一个角落渲染（无特征的模板 / 无模板的特征）；
研发透镜的 GAP 链接是单向的，缺口无法反向驱动任务创建。

**C3 实时分析的模型接入靠手输路径。** `LensSpatial` 的 live 模式给了一个文本输入框让用户敲
`D:\models\qwen3-4b`；而同屏 `StackMode` 已有 `ModelManager`（本地模型库，含已发现模型列表）被挂在
`modelSlot` 里（`LensSpatial.jsx:321`）——现成的能力没有复用。

**C4 术语首现无解释。** `FTR-xx` / `F#3734` / `KN·AR·GR` / `E0–E3` / `十道门` 在界面上都不带释义，
新手引导（NewbieGuide）讲的是操作路径，不是概念。顶栏"特征源"与面包屑"F#3734"并列，二者关系没有一句话说明。

**C5 专项页 ×10 与图谱 4 tab 并存。** 同一研究主题有两个入口（例：`/rdc-law` 与图谱·模板与技术），
`v1-S5` 的"逐页协议化后收敛"仍未开始；`/legacy` 旧版主界面仍在构建产物里。

---

## 3 改进建议（四批，含验收口径）

### P0 · 口径统一（0.5–1 天，零风险，先做）

**做法**：新建 `frontend/src/components/app/rdc_fusion/evidence.js`，作为证据等级**唯一真源**：

```js
// 由 API 的 evidence_level_taxonomy 派生，不在组件里硬编码
export const E_ORDER = ['E0_candidate','E1_repeatable','E2_predictive','E3_causal_scoped'];
export function eMeta(level, taxonomy){ ... }   // { cn, color, idx, countsToE2Plus, known }
```

- `AtlasLens` 的 `ECLASS`、`AtlasModuleSpace` 的 `E_COLOR/E_LABEL`、`FeatureDetail` 徽标、分布条**全部改为 import**；
- 分布条按 **父级** 聚合、按 **子档** 拆分：`E2_predictive 17 · E2_held_out 1 · E2_behavioral 2`
  显示为 `E2 20`，可展开看三档（合计必须等于注册表条数 26）；
- `E2_held_out / E2_behavioral` 是 **E2 的子档**，不另立一级——协议侧需在
  `evidence_level_taxonomy` 里补 `parent: E2_predictive`（append-only 追加，不改既有字节）；
- taxonomy 里没有的值 → 显示 `未知等级` + 灰色虚线徽标 + 控制台告警，**不允许回落到 E1 样式**。

**验收**（可对账，不靠目视）：拉 `/api/ai-rnd/atlas/registry` 最新版，
①分布条显示的总数 === `features.length`（当前 26）；
②26 条里每条都能在分布条找到归属（含 2 条 E2_behavioral、1 条 E2_held_out）；
③临时塞一条 `E0_candidate` 能正常显示且不影响其它档；④新增等级只改 taxonomy，不改组件。

### P1 · 首屏回正（3–5 天）

1. **总览统计换使命计量**：三域覆盖度（有特征/有模板的比例）· E 级分布（E0–E3，E2 带 held-out/behavioral 子档）·
   跨模型一致特征数 · 缺口数；生产计量（sealed/模板/结果）降为次级一行。
2. **路线透镜默认 tab = 图谱**；导航项"路线"改名为「图谱」（`?lens=progress` 保留兼容，`?lens=atlas` 已是别名）；
   行业进展降级为图谱内右侧「视野」抽屉或总览卡片——它服务"对标"，不服务"生产"。
3. **去写死**：模型选择器接 `/api/live/models` + `ModelManager`；状态条 GPU 接 `/api/live/status`
   或 `/api/distributed/summary`，接不到就显示"未接入"（删掉 `21.3/24G` 常量）；
   「分享」按钮在 P2 扩键后实现，之前显示 disabled + title 说明，而不是 alert。

**验收**：断网/离线启动时首屏不出现任何具体数值（只有"未接入"）；`grep -n "21.3/24G"` 为空。

### P2 · 对象闭环 + URL 全量状态（5–8 天）

1. **URL 键扩全**：`?lens=spatial&src=data|live&mode=neuron`、`?lens=process&panel=run|platform`、
   图谱全景 `&pan=modules|modspace|tplspace|cards`；所有 `setState` 统一走 `syncURL`
   （现状只有 `go()` 同步，透镜内部状态切不动 URL）。
2. **补三条跨透镜边**：`FeatureDetail → 空间点云定位`、`FeatureDetail → 结果库该特征`、`结果库行 → 特征/空间`。
3. 空间视图接受 `atlasSel` 高亮该特征族（CloudMode/NeuronSpace 已有点选能力，只缺入参）。

**验收**：任取一个 FTR，从图谱详情跳空间→跳结果库→回详情，全程 URL 可复制可复现；刷新后视图不变。

### P3 · 证据门链一等视图（8–12 天，收益最高）

1. **`FeatureDetail` 增「十道门」区块**：十格逐门三态（已过 / 未过 / 未测），未测必须是显式的"未测"，
   不能用灰色暗示已过；点击某门 → 展开对应 anchors 与反证。
2. **数据来源**：`anchors[].role + tags` → 门链映射表。这张映射表**属于协议**，建议在 registry 追加
   `gate_map` 段（v1.33，append-only，不改既有字节），由派生器产出，界面只渲染。
   协议未提供时整块显示"协议未提供门链映射"，**不猜**。
3. **总览加载"门链通过率热图"**（特征 × 十门），这是使命主产物最该被看见的形态，也直接暴露
   "哪些门全项目都还没通过"——比任何统计数字都更能指导下一步。

**验收**：热图上每个格子可下钻到特征详情对应门；未测格子数 = 注册表里 gate_map 缺失的特征数（可对账）。

---

## 4 工程一致性（顺手做，避免重复踩坑）

1. **3D 基座抽取**：`LiveStage` / `AtlasModuleSpace` / `StackMode` 各自实现相机与投影。
   最近的焦距缺陷（`cv.height` 已含 dpr 又乘一次 + `1.72` 像素经验常数，导致 28 层塔只有 22 CSS px）
   就是"每处各写一份"的代价。建议抽 `core/camera3d.js`：yaw/pitch/dist/pan + 投影 + 拾取 + 环命中，
   各视图只提供几何与着色。
2. **占位件清单化**：把 5 处占位 alert、3 处写死常量登记为 TODO 表，加 ESLint `no-alert`（放行注释里的合法用法）。
3. **术语单点**：`term.js` 收敛 `FTR/F#/KN/AR/GR/E0–E3/十门` 的中文名与释义，
   避免再次出现"过程透镜 / 研发透镜 / 路线 / 脉络"式命名漂移。
4. **专项页收敛路径**：按"协议化一页 → 图谱 tab 能承载 → 删路由"的节奏，
   先把 `/rdc-law`、`/rdc-joint` 并进图谱·模板与技术，再逐页推进；`/legacy` 从构建入口摘除（保分支即可）。

---

## 5 优先级与取舍

| 批次 | 解决 | 判据 | 量级 | 风险 | 收益 |
|---|---|---|---|---|---|
| P0 证据等级唯一派生 | A1 | D4 | 小 | 极低 | 首屏不再漏类/串色 |
| P1 首屏回正 | A3·B2·C1 | D1·D5 | 中 | 低 | 打开即看到图谱真实状态 |
| P2 对象闭环 + URL | B1·B3·C3 | D2 | 中 | 中 | 特征可跨界面流动、可分享可复现 |
| P3 门链视图 | A2·C2 | D3 | 中大 | 中（需协议追加） | 使命方法论首次进入界面 |
| 工程一致性 | B4·C4·C5 | — | 中 | 低 | 防止同类缺陷复发 |

**明确不做（保持 v1 结论）**：不再把图谱提为独立一级导航（总览已是驾驶舱）；
不再新增 3D 视图种类（现 5 种足够，边际收益递减）；不做"导出 PNG"（URL 全量状态优先）。

---

## 6 一句话结论

结构层（四透镜/一驾驶舱/图谱 4 tab）已经稳定，**剩下的问题从"界面结构"转到了"口径与证据"**：
证据等级抄了三份、门链没有视图、对象还在两套宇宙里。
所以下一步不该再加界面，而该**把协议里的东西如实搬到界面上**——P0 一小时级改动就能消除首屏口径错误，
P3 才是让这个客户端真正"像使命"的那一步。
