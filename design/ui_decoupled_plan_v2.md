# 界面解耦方案 v2 —— 研发透镜与全局元层的「流程/内容」二分裁决（M3 提案）

日期：2026-10-08 ｜ 状态：**M3 已落地**（2026-10-08：P1/P2/P3 全部实施，e2e 27/27 + playwright 18/18 零页面错误）
上游：`design/ui_decoupled_plan_v1.md`（M2 已落地：D1-1/D1-3 + 五件协议组件 + 数据透镜 + AGG 实时卡）

> M3 落地注记（2026-10-08）：实施与提案的差异——
> ① `GET /api/main/scan_files` 经核实为专用端点（只扫 tempdata/codex 的 JSON），不能当通用文件树，
>   改为在 ai_rnd_service 新增 **`GET /api/ai-rnd/workspace`**（白名单根 tests/research、防 `../` 逃逸——
>   首版按首段白名单校验存在逃逸漏洞，负路径测试揪出后改为解析后真实落点校验）与
>   **`GET /api/ai-rnd/workspace/file`**（文本 ≤512KB）。
> ② `GET /api/ai-rnd/queue` 读 `research/deepseek/atlas/phase_queue_v1.json`（30 条，sealed=8），缺文件返回空队列。
> ③ 对象卡数据源 = **`server/object_registry.json`**（object_registry_v1，播种 F#3734）+
>   `GET /api/object/{fid}`（富化：tm_ids → 分布式 results，F#3734 已 LIVE 富化 TM-04 两条结果）+
>   `GET /api/objects`（⌘K 数据源）。未注册 → 404 → 前端 DEMO_OBJECT 回退。
> ④ 前端：LensProcess v3（流程骨架保留，队列双 tab 研发线/分布式 + 工作区文件树 + 工件查看器 + 右栏选中概览/证据链全数据驱动）；
>   RdcFusionWorkspace v5.1（URL `?lens=&obj=` 即状态、对象卡纯渲染器、⌘K/ticker/HomeView 统计 live 优先、融合主张四透镜版）；
>   demo 内容唯一源 = demoData.js（DEMO_QUEUE/DEMO_WORKSPACE/DEMO_ARTIFACTS/DEMO_TERMINAL/DEMO_OBJECT）。
> ⑤ :5001 重启以 `AI2050_SKIP_MODEL_LOAD=1` 启动（融合界面不依赖本地 GPT-2；模型端点按需再开）。

## 0 判定准则：每个界面块二选一

承接 v1 核心原则（协议是唯一稳定契约，界面是协议的通用渲染器），v2 把判据细化为对
**研发透镜与全局框架** 的二分：

- **流程件**：驱动研发循环本身，与「测什么」无关——五证据门、AI 模型配置、目标 composer、
  自动/手动/暂停/停止控制、SSE 事件流。→ **结构保留，不动**。
- **内容件**：展示「测了什么、结果如何」——队列条目、文件树、代码/结果工件、结果卡、对象数据。
  → **必须只认协议键、数据来自 API，LIVE/DEMO 双模**（与 LensData 同一范式）。

**新红线**：JSX 中不得出现具体实验的字面量（`Q05`、`F#3734`、`collect_ar.py`、`tests/deepseek/` 等）。
它们只能存在于数据层——API 返回值，或 `demoData.js`/`distributedData.js` 这类集中 DEMO 回退文件。

## 1 逐项裁决（本轮五问）

| 项 | 现状（代码核实） | 裁决 | 动作 |
|---|---|---|---|
| **研发界面**（LensProcess） | 流程骨架已内容无关（门/模型配置/composer/SSE 事件 tab 均 LIVE）；但内容层 4 处硬编码：QUEUE、文件树、3 个 demo tab、右栏结果卡/证据链/终端 | **半解耦 → 分层调** | 骨架不动；内容层全部下沉为数据插槽（M3-P1） |
| **任务队列** | `QUEUE` 常量写死 5 条 Q 编号，点击 alert「详情接入 phase_queue.json 后开放」；phase_queue.json 无任何端点暴露 | **应调（A 级）** | 通用 QueueList 组件 + 双数据源（M3-P1） |
| **工作区** | `tests/deepseek/` 4 个文件名写死；`:5001` 已有 `GET /api/main/scan_files`（server.py:446）未被使用 | **应调（A 级）** | 通用 FileTree 接 scan_files，路径参数化（M3-P1） |
| **对象卡** | F#3734 全部写死（名称/kv/激活示例/三条跨透镜链接），与顶栏「对象路由即状态」自相矛盾；⌘K、crumb 同为 demo | **应调（B 级）** | 对象 schema（协议第五件）+ 纯渲染器 + URL 路由（M3-P2） |
| **融合主张** | ⓘ popover 是元层设计说明，不含数据渲染，无内容耦合；但叙述过时——仍讲「三透镜」，未含数据透镜与协议原则 | **小改（C 级）** | 文案更新为四透镜 + 两条新原则（M3-P3） |

要点：**研发界面 = 流程骨架（保留）+ 三个数据插槽（队列 / 工作区 / 工件）**。五证据门、模型配置、
composer、控制按钮就是「流程」，本身已经是内容无关的，不需要解耦；需要解耦的是插槽里的内容。

## 2 M3-P1：研发透镜内容层协议化

LensProcess 从「写死 Q05 的展示页」变成「研发流程渲染器」：

1. **任务队列 → QueueList 组件**（双数据源，行 = id / 名称 / 状态徽标 NEXT·SEALED·QUEUED·RUNNING）：
   - 研发队列：`GET /api/ai-rnd/orchestrator/status`（recent_runs、project_agent 计划任务）——端点已存在；
   - 分布式测试队列：`GET /api/templates`（注册=待测）+ `GET /api/results` 按 tm_id 聚合（done）；
   - 服务端补一个小端点 `GET /api/ai-rnd/queue`：读 `research/deepseek/atlas/phase_queue.json`，文件缺失返回空数组（不炸）。
2. **工作区 → FileTree 组件**：接 `GET /api/main/scan_files`（已有），根路径由顶栏「特征源」选择器决定；
   点文件 → 中栏工件查看器打开真实文件（可复用 `research_asset_service` 的 `/file/{asset_path}`）。
3. **中栏工件 tabs**：`collect_ar.py / result.json / review_report.txt` 三个 demo tab → 「运行工件查看器」：
   选中队列条目 → 拉 `orchestrator/runs`、`/history`、`/findings`（端点均已存在）；「实时事件」tab 已 LIVE，保留。
4. **右栏**：结果卡 / 证据链 / `TERM_LINES` 假终端 → 由选中项驱动，复用 protocolComponents 的
   StatChips / ResultDetail；终端行改为 SSE 事件原文流（事件流已是真实数据源）。

验收：切换任意 run / phase / 模板，队列、文件树、工件、右栏全部正确变化，不改一行 JSX。

## 3 M3-P2：对象卡协议化（对象路由闭环）

1. **对象 schema（协议第五件）**：`{ id, label, layer, evidence, metrics[], activations[], links[] }`。
2. 服务端聚合端点 `GET /api/object/{fid}`：汇聚特征注册表 + 指标（E_read / share_max）+ 最近相关 results
   （`/api/results` 按 tm_id 反查）；未命中返回 404，前端降级 DEMO 卡。
3. 对象卡 = 纯渲染器；`URL ?lens=&obj=F#3734`（URL 即状态落地）；顶栏 crumb、⌘K、三条跨透镜链接全部由对象数据驱动。
4. 现硬编码 F#3734 内容迁入 `demoData.js` 作 DEMO 回退。

验收：换 obj id，对象卡、面包屑、⌘K 搜索、跨透镜链接联动更新。

## 4 M3-P3：元层文案与全局 demo 清理

1. **融合主张 popover**：三透镜 → 四透镜（空间 · 过程 · 脉络 · **数据**）叙述；原则补两条——
   「协议是唯一稳定契约」与「对象路由即状态」。HomeView hero、入口卡（数据透镜入列）、LENS_NAME 同步。
2. 顶栏 crumb / 模型与特征源下拉 / ⌘K 分组 / 底部 ticker 与状态条 / HomeView 统计数字（306/40/7-30）：
   接真实数据源（templates+results、orchestrator、ledger 索引），接不通的显式挂 DEMO 徽标。
3. 全部 demo 内容集中到 `demoData.js`（唯一 demo 源），组件内不再出现内联常量数据。

## 5 分期、依赖与验收

- 顺序：**P1 → P2 → P3**。P1 无新依赖（三个端点已存在，仅补 `/queue`）；P2 需聚合端点；P3 纯前端。
- 与 v1 M3 待排期项合并：AI 模板生成器、空间特征目录、D1-2 聚合统计不变；本轮裁决（P1/P2）是它们的前置——
  队列与对象卡通用化后，模板生成器产物自动进队列、特征目录条目自动进对象卡。
- e2e 增补断言：队列 LIVE 渲染（含分布式 tm_id 行）、FileTree 打开真实文件、对象卡切 id 联动、四透镜文案。
- 纪律不变：LIVE/DEMO 双模 + 徽标；协议键白名单渲染；升级走 version / SHA 登记。
