# AGI 技术方案与可视化解耦框架方案

## 1. 背景与目标

当前项目的优势是功能丰富（拓扑、曲率、RPT、AGI 统合等），但主要问题是：

1. 算法实现、服务 API、前端展示耦合较深。
2. 纤维丛路线在代码层面被“默认主路线”，难以并行验证其他 AGI 路线。
3. 前端面板与后端字段存在隐式绑定，替换算法时改动面大。

本方案目标是建立一个**路线无关（Route-Agnostic）**的框架：

1. 允许并行尝试多条 AGI 技术路线（纤维丛/世界模型/符号-神经混合等）。
2. 用统一观察协议采集内部结构数据，支持多种可视化工具复用。
3. 将“算法执行”和“可视化呈现”彻底分离，后续换路线不换前端壳。

## 2. 设计原则

1. 内核最小化：核心层不依赖具体路线与 UI。
2. 插件化：每种 AGI 路线以插件形式接入。
3. 契约优先：先定义数据契约，再写前后端代码。
4. 事件驱动：分析过程通过事件流输出，前端按事件订阅展示。
5. 可复现：每次运行都有 run_id、配置快照、指标与结论归档。

## 3. 分层架构（建议）

### L0. Core Kernel（统一抽象层）

职责：统一任务生命周期，不含具体算法。

核心对象：

1. `AnalysisSpec`：一次分析的声明（目标、输入、参数、路线路径）。
2. `AnalysisRun`：运行实例（run_id、状态、时间、错误）。
3. `Artifact`：中间或最终产物（张量、图、文本、3D mesh）。
4. `Metric`：量化指标（名称、数值、范围、解释）。
5. `Conclusion`：结构化结论（证据、置信度、风险）。

### L1. Route Plugins（AGI 路线插件层）

每条路线都实现同一接口，不直接依赖 UI：

1. `FiberBundleRoute`（现有路线）
2. `CausalCircuitRoute`
3. `WorldModelRoute`
4. `NeuroSymbolicRoute`
5. `HybridRoute`

统一接口（建议）：

```python
class RoutePlugin(Protocol):
    route_name: str
    version: str
    supported_analyses: list[str]
    def prepare(self, spec: AnalysisSpec) -> None: ...
    def run_step(self, ctx: RunContext) -> list[Event]: ...
    def finalize(self, ctx: RunContext) -> RunSummary: ...
```

### L2. Observability Layer（观测与探针层）

职责：把模型内部结构抽象成统一观察信号。

统一事件类型（最关键）：

1. `ActivationSnapshot`
2. `TopologySignal`
3. `AttentionFlow`
4. `CurvatureSignal`
5. `AlignmentSignal`
6. `ReasoningTrace`

事件协议统一为：

```json
{
  "event_type": "TopologySignal",
  "run_id": "run_20260218_xxx",
  "step": 12,
  "timestamp": "2026-02-18T10:20:30Z",
  "payload": {},
  "meta": { "route": "fiber_bundle", "model": "qwen3_4b" }
}
```

### L3. Orchestrator（调度与实验管理层）

职责：负责任务编排和多路线对比。

能力：

1. 单路线运行。
2. 多路线并行运行（同输入同评估）。
3. A/B 结果对比（指标、结论、稳定性）。
4. 失败恢复与重试。

### L4. Service API（服务网关层）

改为“通用运行 API”，避免前端绑死具体算法端点。

建议接口：

1. `POST /api/v1/runs` 创建分析任务（包含 `route`、`analysis_type`、参数）。
2. `GET /api/v1/runs/{run_id}` 获取状态与摘要。
3. `GET /api/v1/runs/{run_id}/events` 拉取事件流（或 SSE/WebSocket）。
4. `GET /api/v1/runs/{run_id}/artifacts/{artifact_id}` 获取产物。
5. `GET /api/v1/catalog/routes` 路线能力清单。
6. `GET /api/v1/catalog/analyses` 分析能力清单。

### L5. Visualization Workbench（前端工作台层）

前端不再调用算法专用端点，只消费通用 API 与事件流。

前端分为两类插件：

1. `ViewPlugin`：3D/2D 可视化组件（GlassMatrix、FlowTubes、Topology、TDA）。
2. `GuidePlugin`：算法指南组件（目标、思路、3D 原理、算法说明、指标范围、通俗说明、结论模板）。

## 4. 关键解耦点

## 4.1 路线与可视化解耦

可视化绑定“事件类型”，不绑定“算法名字”。

例如：

1. GlassMatrix 订阅 `ActivationSnapshot`。
2. FlowTubes 订阅 `AttentionFlow`。
3. 拓扑面板订阅 `TopologySignal` 与 `CurvatureSignal`。

只要新路线发出同类事件，原可视化可直接复用。

## 4.2 分析结论标准化

每个分析都输出统一 `ConclusionCard`：

1. `objective` 目标
2. `method` 思路
3. `evidence` 证据（指标 + 图）
4. `result` 结论
5. `confidence` 置信度
6. `limitations` 局限
7. `next_action` 下一步实验

这样“算法指南窗口”与“模型说明窗口”不依赖具体脚本字段。

## 4.3 配置驱动 UI

用一份注册表统一管理分析项：

1. 菜单名称
2. logo/icon
3. 对应事件类型
4. 对应 guide 模板
5. 对应默认指标范围

现有 `frontend/src/config/panels.js` 可升级为此注册表中心。

## 5. 对当前项目的落地映射

建议目录重构（渐进式，不一次性大改）：

1. `core/`：`spec.py`、`events.py`、`artifacts.py`、`conclusion.py`
2. `routes/`：`fiber_bundle/`、`causal/`、`world_model/`、`neuro_symbolic/`
3. `orchestrator/`：`runner.py`、`registry.py`、`comparator.py`
4. `api/`：`v1/runs.py`、`v1/catalog.py`、`schemas/`
5. `frontend/src/workbench/`：`view_plugins/`、`guide_plugins/`、`registry/`

现有模块迁移建议：

1. `scripts/*engine.py` 保留算法逻辑，逐步包裹为 `RoutePlugin`。
2. `server/server.py` 逐步把 `/nfb*` 端点收敛到 `/api/v1/runs*`。
3. `frontend/src/App.jsx` 从“直接请求多个端点”改为“run_id + event stream”模式。

## 6. 迭代里程碑（建议 4 阶段）

### Phase 1（1-2 周）：契约与最小骨架

1. 定义 `AnalysisSpec/Event/Metric/Conclusion` schema。
2. 做一个最小 `RoutePlugin`（先包 FiberBundle）。
3. 打通 `POST /runs` + `GET /runs/{id}` + `events`。

验收标准：前端至少一个面板可通过新协议显示数据。

### Phase 2（2-3 周）：前端工作台解耦

1. 引入 `analysis registry`（菜单、icon、guide、指标范围）。
2. 算法指南切换到配置驱动。
3. 3D 面板改为事件订阅模式。

验收标准：新增一个分析项无需改核心 App 逻辑。

### Phase 3（2-4 周）：多路线并行

1. 接入第二条路线（如 Causal/WorldModel）。
2. 增加同任务多路线 A/B 对比视图。
3. 输出统一比较报告（性能、稳定性、可解释性）。

验收标准：同一输入可生成两条路线的可视化和结论对比。

### Phase 4（持续）：评测闭环

1. 接入 `tests/agi` 作为路线评估统一门禁。
2. 建立“路线可行性看板”（完成度、风险、证据强度）。
3. 形成自动回归与实验归档流程。

验收标准：每次新路线提交都可自动评估并可视化对比。

## 7. 风险与控制

主要风险：

1. 历史端点多，过渡期间双协议并存会增加复杂度。
2. 事件定义过粗会导致前端表达能力不足，过细又导致维护成本高。
3. 多路线指标口径不统一，结论不可比。

控制策略：

1. 双轨过渡：保留旧端点，只在新功能使用 `/api/v1/*`。
2. 先定义最小事件集（6 类），按需扩展。
3. 指标分层：通用指标（必须）+ 路线专属指标（可选）。

## 8. 最小可执行清单（下一步）

1. 先冻结并发布 `v1` 数据契约（schema 文件）。
2. 把 `FiberBundle` 路线改成首个 `RoutePlugin`。
3. 新增 `/api/v1/runs` 三个基础端点。
4. 前端把一个面板（建议 GlassMatrix）切到事件流。
5. 算法指南改为注册表驱动，默认显示“结构分析大纲”。

---

结论：本框架的核心是把“纤维丛路线”降级为“可替换插件”，把“可视化”升级为“协议驱动观察层”。这样即使后续路线切换，前端与实验管理体系仍可复用，项目可持续推进到多路线 AGI 验证阶段。
