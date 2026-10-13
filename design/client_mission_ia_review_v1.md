# 前端界面系统梳理 · 机制图谱使命视角（v1）

> 2026-10-11。评审基准：RdcFusionWorkspace v5.1（四界面）+ AtlasLens v1.6（图谱 4 tab）
> + PlatformOps v3.3 + LensProcess v5.1 + 专项页 ×10。
> 使命锚点：**顶级 LLM 逆向工程——破解语言编码机制；方法论主产物 = 语言机制图谱**
> （三域树 KN/AR/GR → 语言族 → 特征 FTR-xx → 锚定样本，E 级证据，协议唯一契约）。

## 1. 现状结构树（实测自代码，非凭记忆）

```
RdcFusionWorkspace（/ 与 /rdc-fusion，URL 即状态 ?lens=&obj=&atlasTab=）
├─ 全局件
│  ├─ 顶栏：模型选择器〔演示死件〕· 特征源选择器〔演示死件〕· 对象面包屑 F#3734 · ⌘K · 通知铃(badge=2 写死)
│  ├─ 底部状态条：GPU chip · queue sealed chip · ticker（LIVE 优先回退 DEMO）
│  └─ 对象卡（右栏已隐藏，恢复点保留）
├─ 总览 HomeView
│  ├─ Hero：使命链 MISSION + 三统计（队列 sealed / 测试模板 / 结果条目）
│  ├─ 入口卡 ×4（文案仍写「过程透镜/脉络透镜/图谱透镜」旧称 ⚠）
│  ├─ 右栏：进行中（接 /ai-rnd/queue）· 事件流（纯 demoData ⚠）
│  └─ NewbieGuide 新手引导
├─ 空间 LensSpatial（3D 几何 + 实时权重）
│  ├─ 数据源切换：live（本地模型）/ data（研究包）
│  ├─ 层平铺 StackMode：模型总览 + ModelManager 模型库弹窗（下载/连接）+ 36 层列表
│  ├─ 神经元级 NeuronMode / 特征点云 CloudMode（1024 点·L6 族）/ 参数热图 ParamMode
│  └─ LiveAnalysis 实时分析（连接本地模型 → 权重数值扫描）
├─ 研发 LensProcess v5.1（图谱生产）
│  ├─ 左栏：AI 配置折叠 + 研究线队列（唯一归属）+ GAP 链接→图谱（单向）
│  ├─ 中栏：运行区（五段式：freeze→SMOKE→正式→verdict→closeout，工件查看）
│  └─ 右栏：PlatformOps v3.3 三卡（平台概览 / 节点与租约〔含本机行〕/ 模板与聚合）
├─ 路线 LensProgress（3 tab）
│  ├─ 行业进展 news：同行泳道 + RDC 主线（读 ledger）
│  ├─ 图谱 atlas（核心）：AtlasLens v1.6
│  │  ├─ 图谱全景（默认）：指标条（可点直达账本）· E 级分布 · 三模一态
│  │  │   （泳道 SVG / 域树 / 特征卡）· FeatureDetail 抽屉 · 专项页折叠区
│  │  ├─ 跨线账本：measurements 表（唯一 Phase 时间线）
│  │  ├─ 结果库：ResultsExplorer（浏览 + RSA/RDM · η² 行内分析）
│  │  └─ 模板与技术：RegistryPanel（语言模板 × 分析技术 · 研究包 · 成熟度）
│  └─ 本地进度 machine：NODE-A3F7 本机视图（与研发右栏本机行重叠 ⚠）
└─ 专项页 ×10（/rdc-*，RdcPageShell 壳；过渡态 ⚠）
   feature / prefix / relation / joint / operator / law / binding / update / query / construction
```

## 2. 使命对齐诊断（5 处错位 + 1 处战略级问题）

**S0（战略）图谱被埋两级**。使命主产物是机制图谱，但入口是 `路线 → 图谱` 两次点击；
总览首屏是"产品官网式" hero + 入口卡，与"图谱为核心"不对位。

**S1 两套对象宇宙互不相通**。顶栏/⌘K/对象卡用 object registry（F#3734），
图谱用 atlas registry（FTR-01…26）；特征详情只有「→ 研发透镜」，
没有「→ 空间点云定位」；CloudMode 1024 点与 26 特征无映射。逆向工程的对象（特征）
不能跨界面流动，是当前最大的结构缺口。

**S2 演示死件与自家纪律冲突**。「协议是唯一稳定契约」是融合主张第③条，
但事件流（EVENTS）、通知 badge、两个顶栏选择器均为写死演示件——第一屏就出现假数据。

**S3 命名漂移**。入口卡文案 vs 导航（空间/研发/路线）vs LENS_NAME 三套称呼并存。

**S4 本机状态两家**。路线·本地进度（NODE-A3F7）与 研发·平台运行·节点表（本机行）重叠。

**S5 专项页过渡态**（挂账不变）：10 页独立路由，逐页协议化并入图谱后收敛。

## 3. 建议（P1→P4 分四批）

| 批次 | 内容 | 量级 |
|---|---|---|
| **P1 卫生死件清理**（零风险） | 事件流改接 ledger tail（或删）；通知 badge 去写死；入口卡文案同步五轴称呼 | 小 |
| **P2 总览改「图谱驾驶舱」** | HomeView 主体 = 图谱全景（复用 AtlasPanorama，只读态）+ 侧栏（生产状态 / 最近闭环 / 覆盖度缺口）；hero 压缩为使命一行 | 中 |
| **P3 对象宇宙打通** | 特征源选择器接 atlas FTR；FTR 详情新增「→ 空间点云」「→ 结果库」；CloudMode 高亮选中特征族 | 中大 |
| **P4 本机状态归一** | 本地进度 tab 降为 PlatformOps 本机行展开（或删 tab）；专项页协议化按原计划推进 | 小/大 |

关键取舍说明：P2 不建议把图谱提为一级导航（路线-图谱已承载完整 4 tab 工作区），
而是让**总览直接成为图谱只读全景**——达到"打开即图谱"的效果且不动现有工作区。
