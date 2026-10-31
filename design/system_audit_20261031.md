# 客户端与研发流程 · 系统审计

> 日期：2026-10-31　范围：`frontend/`、`server/`、`tools/`、`ai2050_research_os/`、`.github/workflows/`、`design/`
> 方法：全部结论基于本轮实测（文件计数、引用扫描、路由比对、CI 实跑），每条附可核查证据；不采信文档自述。
> 前序文档：`design/client_architecture_review_20261028.md`（客户端 P0/P1/P2 清单）。本文**不复述**它，只做状态复核 + 补齐它没覆盖的研发流程与对外对标。

---

## 0. 一句话结论

**这个项目现在有两个系统，而不是一个：一个以产品形态存在的研究平台（客户端 + 服务端），和一个以脚本形态存在的研究前线（`tools/` + `tempdata/` + 叙事 memo）。两者共享数据库和术语，但不共享版本控制、不共享事实源、不共享验收标准。** 客户端的问题主要是「产品面只占代码的 20%，遗留占 40%，且没有回归网」；研发流程的问题主要是「结论可读、不可复算」；对标顶级研究的差距不在聪明程度，而在**可复现性、统计功效、外部方法基线**这三件事上。

---

## 1. 体量与形态实测

| 区域 | 文件数 | 行数 | 说明 |
|---|---:|---:|---|
| `frontend/src` 合计 | 456 | 117,744 | 全部 JSX/JS（无 TypeScript） |
| ├ `blueprint/` | 170 | 46,974 | **40%**，只经 `App.jsx`（`/legacy`，dev-only）与已死组件可达 |
| ├ `components/app/`（当前产品面） | 110 | 23,129 | **20%**，工作台 + 配置 + 70 个 rdc 面板 |
| │└ `rdc_fusion/` | 63 | 14,239 | 工作台全部面板 |
| └ 其余（App.jsx 5,204 行、researchCenter、neural_vis、AIRnD 等） | 176 | 47,641 | 旧版 3D 界面与专项页 |
| `blueprint/data` 静态数据 | — | 4.4 MB | 编译进包的研究快照（.js 常量） |
| `server/` | 122 | 28,070 | FastAPI，`server.py` 单文件 2,506 行 / 68 路由 |
| `tools/` | 37 | 7,356 | 研究前线脚本（**git 跟踪 0 个**） |
| `design/` | 49 篇 | 8,088 | 设计文档 |
| `research/*/docs/*MEMO.md` | 8 篇 | 87,020 | 研究叙事（活跃：gpt5 17,787 行） |
| `ai2050_research_os/` | 29 contracts + 16 manifests | — | 治理系统，最后写入 2026-08-14 |

**第一个结构性判断**：产品面（23K 行）比遗留面（47K 行）小一半，但构建产物里两者都在。`dist` 3.9 MB，其中 `App-*.js` 1.8 MB 是 dev-only 的 `/legacy` 主包——**线上用户不加载它，但安装包里装着它**。

---

## 2. 客户端：三个层次的问题

### 2.1 代码基座与实际产品面错位（最贵的结构问题）

实测：

- **51 个模块无任何 import，合计 14,923 行**（占 `src` 12.7%）。最大的是 `blueprint/AtlasControlDashboard.jsx`(1,969)、`blueprint/GeminiTab.jsx`(1,823)、`HLAIBlueprint.jsx`(1,233)、`neural_vis/components/PuzzlePanel.jsx`(741)、`components/MultiLayer3DVisualization.jsx`(581)。
- 逐个抽查确认是真死码（只在注释里被提到，如 `ConceptGallery.jsx:30` 把 `HLAIBlueprint` 列为「已摘除」）。
- 源码树里还有 `HLAIBlueprint.jsx.bak_mojibake_fix`（114 KB 备份文件）参与目录扫描。

后果不是「不好看」，而是三条具体的：

1. **搜索与重构的信噪比**：改一个共享 hook，IDE 会给出几十个假引用，人会开始靠记忆判断「这个应该没人用」，而 10-30 那次误删 10 条规则的事故，本质就是"靠记忆判断引用关系"。
2. **新人/新 Agent 的定位成本**：`components/app/`、`components/v8/`、`blueprint/`、`researchCenter/`、`neural_vis/` 五套并存，没有一层说明谁是当前主线。
3. **包体**：dev-only 的 legacy 仍进构建，安装包白背 1.8 MB。

**改法（低风险、可增量）**：
- 建 `src/legacy/` 一层，把 `App.jsx` + `blueprint/` + `neural_vis/` + `AIRnD/` + `researchCenter/` 整层移入，并在 `vite.config.mjs` 给 legacy 单独一个 build target（`--mode legacy`），主包不再 import。
- 死码删除前先跑一次机器扫描（本轮脚本可直接复用：按 import 字符串反查引用，人工抽查 5 个，再批量删）。
- 目录名去掉版本号：`rdc_fusion/` 里 `api.js`/`dataStore.js`/`ui.jsx` 是**全站基础设施**，不是 rdc 专属，应移到 `src/infra/`；`v8/` 改 `workbench/`。

### 2.2 契约与回归网缺失（最危险的质量问题）

实测：

| 指标 | 现状 |
|---|---|
| 前端测试 | **3 个 vitest 文件 / 14 个用例**（`catalogUtil` 4、`api.timeout` 2、`diagnostics` 8），跑完 0.5 s |
| 产品 UI 测试 | **0**（工作台、70 个面板、图谱、3D 全部无测试） |
| 后端产品测试 | **0**（`tests/{unit,integration,acceptance}` 共 46 个文件 import `transformer_lens`，是上游 fork 的测试） |
| 服务端 `server/` 测试 | **0**（`server/test*.py` 是手工脚本，不是 pytest） |
| CI 6 个 job | 仅 `frontend-checks` 与本产品相关；其余 5 个（compatibility/code-checks/notebook-checks/build-docs/deploy-docs）是 **TransformerLens 上游的**，`release.yml:11` 甚至直接 `uses: TransformerLensOrg/TransformerLens/...@main` |
| ESLint | 非阻塞（`continue-on-error: true`），注释自称存量 732 个问题 |
| 裸 fetch | **208 处**（10-28 评审时是 166 处，**增加了**）；`rdc_fusion/*.jsx` 内直连 31 处，`import './api'` **0 处** |

最后一行是关键：那个统一了超时、错误码、取消、鉴权的 `api.js` 存在，但**面板几乎不用它**。所以「有封装」不等于「有约束」——只要不禁止裸 `fetch`，封装就会被绕过，而且已经绕过了。

**后端同样**：68 条路由塞在一个 2,506 行文件里，**0 个全局异常处理器，0 个 `logging` 调用，164 处 `print()`**。这意味着线上一次 500 只能靠翻 stdout，且错误不会带 trace id 回到前端。另有 **22 条路由前端完全没有字符串引用**（`/fiber/*`、`/nfb/*`、`/api/agi_chat/*` 一族），是历史实验接口。

**改法（按"能不能拦住错误"排序，而不是按"架构好不好看"）**：
1. **CI 三道硬门**：`vite build` 必须过；**新增文件** eslint 零 error（存量用 baselist 冻结，只增不减）；vitest 必须过。把 `continue-on-error` 从 lint 上摘掉（用增量门禁，不做一次性清零）。
2. **禁止裸 fetch**：eslint 自定义规则把 `src/**` 里的 `fetch(` 标 error，白名单只有 `api.js`/`dataStore.js`。然后按面板迁移，先 `LiveStage`、`ResultsExplorer`、`ModelManager` 三个高频面板。
3. **响应信封 + 校验**：顶层统一 `{ok, data, error, pending}`；`useApi` 已返回 `pending`（10-28 的 P0-2 已修一半），但**没有 schema 校验**，字段名错了仍表现为空白。给 5 个核心响应手写 `check(obj)`（不必上 zod），错了在界面上显式报「数据契约不匹配」，而不是静默空态。
4. **后端最小可观测**：一个 `@app.exception_handler(Exception)` + `logging` 结构化输出 + 每个响应带 `X-Request-Id`。这一步的收益比任何重构都高——现在一个 bug 从「用户说不行」到「定位到行」的成本是小时级。
5. **删 22 条无引用路由**（或迁到 `/api/legacy/*` 并加 deprecation 标记），让 API 面等于产品面。

### 2.3 信息架构：还差一次「按任务而不是按对象」的收敛

10-30 已经把 10 个研发工位收敛到 5 个、把 4 个工位合成带步骤条的流水线，这一步是对的。现在剩下的重复是**同一研究对象在两个界面各有一套视图**：

- 「进展」有 `neuron` / `lang`，「空间」也有 `neuron` / `lang`；
- 「进展」是统计视角（普查规模、分型、证据门链），「空间」是几何视角（层塔、点云、单元卡）。

这个切分本身站得住（一个是"我们知道了什么"，一个是"它在哪"），但**导航文案没有把这条区分说透**，用户看到两个同名 tab 的第一反应是"哪个才是真的"。建议：进展下的两个改名为「覆盖与证据」/「图谱与缺口」，让一级导航回答"我要做什么"，而不是"我要看哪个对象"。

其余 P2 级问题（无自动更新、Linux 目标不可用、`index.html lang="en"`、无 CSP、`vite host 0.0.0.0`、无 telemetry、`draft` 只在内存）10-28 已列，本轮复核**均未变**，不重复展开。

### 2.4 10-28 评审的状态复核（只报变化）

| 编号 | 10-28 的问题 | 现状 | 证据 |
|---|---|---|---|
| P0-1 | 前端零测试 + CI 不构建 | **部分修复**：CI 有 `frontend-checks`（vitest+build），但只有 14 个用例、lint 非阻塞 | `.github/workflows/checks.yml:47-62` |
| P0-2 | 数据契约无校验 | **部分修复**：`useApi` 现返回 `pending`；仍无 schema 校验 | `api.js:108-165` |
| P0-3 | 服务器地址可指向任意 HTTP | **部分修复**：`normalizeServer` 会把 `http` 升到 `https`，但仍接受 `http://` 字面量、无 `pinned_host`、无 CSP | `electron/main.js:56-66` |
| P0-4 | 渲染进程异常不可见 | **部分修复**：`diagnostics.js` 66 行 + `installGlobalErrorCapture()` 已挂；但 `Workbench.jsx` 内**无 ErrorBoundary**，一个面板抛错仍整页白 | `main.jsx:15`, `Workbench.jsx:0` |
| P0-5 | 权限判定靠缓存用户 | **未修**：仍是 `isAdmin` 散落判断（12 处），无 capability 对象 | `Workbench.jsx:95-104` |
| P1-1 | 无路由库 | **未修**：`main.jsx` 仍 `pathname` 字符串分发 | `main.jsx:36-54` |
| P1-3 | API 层形同虚设 | **恶化**：fetch 166 → 208 | 本轮 `grep -c` |
| P1-4 | 巨型单体 + 孤儿 | **恶化**：孤儿从"24+"到实测 51 个 / 14,923 行 | 本轮扫描 |

---

## 3. 研发流程：五个断点

这一节是本文的重点，因为 10-28 的评审完全没覆盖它，而它比客户端问题更影响"结论可不可信"。

### 3.1 实验代码不在版本控制里

```
$ git ls-files tools | wc -l
0
$ git status --short tools
?? tools/
```

37 个脚本、7,356 行——`mech_core.py`、`mech_carrier.py`、`mech_logit_effect.py`、`mech_prefix_scan.py`、`mech_head_tokens.py`、`rules_ingest.py`——**全部未被 git 跟踪**。同时 `.gitignore` 里有 `tempdata/*.json`，所以 48 个结果文件也不入库。

后果是可计算的：`data/rules/ledger.json` 里 46 条规则的 `evidence` 字段写着 `tools/mech_core.py` + `tempdata/core_L34.json`。**这两个路径指向的东西都没有版本**。任何人（包括三个月后的你）拿到这个仓库，都无法重建出那条结论。而这条线的全部价值就在于"结论可核查"。

### 3.2 结果文件没有 provenance

抽查三份核心结果：

| 文件 | 有 schema/model/block | 有 seed | 有 git commit | 有模型 revision |
|---|---|---|---|---|
| `core_L34.json` | ✓ | **✗** | **✗** | **✗** |
| `core_one_L34.json` | ✓ | **✗** | **✗** | **✗** |
| `logit_effect_L34.json` | ✓ | **✗** | **✗** | **✗** |

`model_name: "Qwen3-4B"` 不足以定位权重——同名的量化版本（fp16 / nf4）、不同 HF revision 会给出不同结果，而这条线已经踩过一次量化坑（nf4 上 G-C 判 FAIL → BLOCKED）。

**这是全项目性价比最高的一处修补**：在结果 json 里加 5 个字段（`git_sha`、`git_dirty`、`model_path`、`model_revision`、`seed`），一个 30 行的 `provenance()` 工具函数，之后所有脚本统一调用。成本接近零，收益是把"可读的结论"变成"可复算的结论"。

### 3.3 两套事实源：治理系统空转

`ai2050_research_os/README.md` 写得很硬：

> 任何 AI、Agent、Codex 实例、开发者或用户在基于本项目进行二次开发之前，都必须……确认修改没有建立新的平行事实源……`generated/`、客户端 JavaScript 常量、Memo 和页面文案都不得反向覆盖 `registry/` 中的正式事实。

而它自己的设计也确实到位：`schemas/` 有 6 份 JSON Schema，`contracts/` 有 29 份冻结实验合同，`registry/runs.json` 里每条 run 带 `gpu_hours`、`verdict`、`artifact_ids`（如 `RUN-EXP-C002-WP01-001-001`，`verdict: bounded_external_validity_failure`）——这正是顶级实验室的 run bundle 形态。

**但**：

- 最后一份 contract 是 **2026-08-14**，最后一条 registry 是 **2026-10-13**；
- **没有 `runs/` 目录**（README 里写的是"目标结构"）；
- 当前活跃的整条线（L34 单元归因、承载核、格式单元，H1–H22）在 `research_os` 里**一条都没有**；它跑在 `tools/` + `tempdata/` + `design/lang_code_joint_attribution_v3.md` + `data/rules/ledger.json` 上；
- `grep -l "L34\|767\|tailN" registry/*.json` 命中的是无关上下文。

所以现实是：**规范在 A 系统里，实践在 B 系统里，A 已经两个月没动。** 这比"没有规范"更坏——因为它让人以为有治理。

### 3.4 判据靠人肉，同类错误复发 5 次

本线账本里有 5 条同族规则，全部是"判据的计算口径与结论口径不同源"：

| 规则 | 事故 | 代价 |
|---|---|---|
| R-MET-06 | 解析失败计入分母，`P = rand` 是假象 | 一轮结论作废 |
| R-MET-07 | 脚本内置 verdict 把 None 记错，net(neg) −6→0 | 结论方向反转 |
| R-MET-08 | k80=None 时走 else 分支，把悬崖读成"接近线性" | 形状判断错 |
| R-MET-09 | 判据只写 `>`，0.093 vs 0.091 的噪声被判"更集中" | 一条假设误判 |
| R-MET-11 | 随机对照用"最大值 ≤ 阈值"，被一道通用脆性题击穿 | 特异性结论被误判为否 |

五次里没有一次是"模型行为难以理解"，全部是**判据实现错误**。而且这五条规则写下来之后，第六次仍可能发生——因为它们是自然语言，没有任何机制在执行它们。

同一节还有一次操作事故：10-30 改账本时用区间切片 `s[:A] + new + s[B:]`，A/B 之间还有别的条目，**误删 10 条规则**，其中 R-MET-06/07 原文已随会话压缩永久丢失，只能按工作日志重建并标 `restored: true`。

这两件事指向同一个结论：**这条线的纪律强度已经超过人（和 Agent）的可靠执行上限，必须把纪律编译成代码。**

### 3.5 统计功效与外部效度

最近一轮（H13）的实测数字：

- 题池 169 道，base 答对 110 道，`max_new_tokens=260`；
- core3 打坏 9 道（−4.67 pt），tailN 打坏 12 道（−6.67 pt），767 单点 11 道；
- p 值：core3 vs base **0.0654**、vs rand97 **0.0703**、vs core5 **0.3438**、vs tailN 0.25；唯一显著的是 tailN vs base **0.0129**。

这不是"做得不好"，而是**分辨率的硬上限**：效应只有 9–12 道题，n=110 时 McNemar 的检验力不足。文档里已经如实写了"只能读结构，不能读显著"——这个自觉是对的。但对标顶级实践，缺的是三件事：**功效分析（先算需要多少题）、多 seed 重复（现在单 seed）、跨模型验证（现在单模型单层）**。

---

## 4. 与行业顶级研究对标

### 4.1 八个维度

| 维度 | 本项目当前 | 顶级实践（可核查） | 差距 | 最小补救 |
|---|---|---|---|---|
| **可复现性** | 代码不入库、结果无 commit/seed、结果 json 被 gitignore | 每次 run 记录 git SHA + 配置 + 环境 + 数据版本；DVC / W&B / MLflow 管产物 | **大** | `provenance()` + `tools/` 入库 + 结果带 SHA，一周内可完成 |
| **统计推断** | 单 seed、n=169、无功效分析、无多重比较校正 | 预注册判据 + 功效分析 + 多 seed + 置信区间 | **大** | 每个 Phase 先写"需要多少题才有 80% 功效"，再决定题池 |
| **因果严格性** | 「删掉会不会坏」+ 剂量曲线 | causal scrubbing（Chan et al. 2022）：检验"假设解释了多少行为"，用反事实重采样算行为保留比例，区分「不重要」与「不忠实」 | **大** | H22 直接改用 scrubbing 口径，而不是再做一次消融 |
| **方法基线** | 只有自家 `\|Δ\|` 归因 + 消融，无外部方法对照 | 与 SAE、attribution patching（AtP*，Kramár et al. 2024）、ACDC（Conmy et al. 2023）、path patching（IOI）在同一任务上比 | **大** | 选 2 个方法在同一 169 题上跑，报"同一发现需要多少干预" |
| **公共基准** | 自建题库与自建判据 | MIB（Mueller et al. 2024）：circuit localization + causal variable localization 两任务标准化；SAEBench（Karvonen et al. 2025）：8 项 SAE 指标 | **中** | 至少把 MIB 的两个任务搬过来当外部校准，避免"自定判据自证" |
| **跨模型 / 跨层** | Qwen3-4B，L24 / L34 为主 | Gemma Scope 覆盖 Gemma 2 全部层与子层；Scaling Monosemanticity 在 Claude 3 Sonnet 上做 34M 特征并做 feature steering | **大** | 同协议跑第二个模型族（哪怕 0.5B），只验证"承载核现象是否复现" |
| **自动评估** | 人工读 memo、人工写账本 | auto-interp（Bills et al. 2023）：用 held-out 激活相关性给解释打分，自动排序 | **中** | 把账本 46 条转成可执行断言，跑 CI |
| **产物开放** | 内部文档 | feature dashboard、模型卡、代码与数据同步发布 | **小（但影响可信度）** | 至少给每条 confirmed 规则配一个可一键复跑的脚本 |

### 4.2 三个最该补的方法学

**(1) causal scrubbing 取代「删了坏不坏」**

现在整条线（H12→H21）的骨架是"删掉 X，看答案变不变"。这条路最终把结论从"知识单元"翻成"格式单元"，翻得对，但代价是 13 个实验。scrubbing 的问法不同：给定一个假设 H（"这 7 道题只依赖单元集合 S 的这 3 个方向"），用**保持 H 成立的重采样**替换掉 H 认为不重要的部分，看行为保留多少。它能直接区分两种失败：假设错了（行为掉了）vs 假设不忠实（行为在，但机制不是它说的那样）。这正好是当前最痛的那一刀——我们一直分不清"删掉会坏"和"它是原因"。

**(2) 把 46 条账本规则编译成断言**

账本是这个项目最好的资产（46 条，含 23 条 refuted，比大多数论文的 negative results 都全）。但它现在是 Markdown 式自然语言。可行路径：给每条规则加一个 `assert_fn` 字段，指向一个能在 30 秒内跑完的检查（"tailN 的 23 个单元 ⊂ 该层 9728"、"core3 的损伤题集 ⊆ tailN"、"随机 97 的改答率 ≤ 2×base"）。CI 每次跑，规则被推翻时立刻红——而不是等到某天有人重跑才发现。

**(3) 功效分析前置**

现在的问题是"跑完 88 分钟才发现效应只有 9 道题"。正确顺序：先估效应量（比如"预期能打坏 10% 的题"），算 n=110 时 McNemar 的检验力，若 <0.8 就先扩题池或改配对设计。这一条能省掉大量"跑了但读不出"的实验——本线已经有好几轮属于这类。

### 4.3 工程侧对标

| 维度 | 本项目 | 一流工程实践 | 补救 |
|---|---|---|---|
| 类型安全 | 全 JSX，无 TS；`@types/react` 装了但没用 | TS 或至少 JSDoc + `checkJs` | 新文件 `.tsx`，旧文件渐进 |
| 契约 | 手写 JSON，前后端人工对齐 | OpenAPI 生成客户端 / zod 运行时校验 | 后端加 `response_model`，前端生成类型 |
| 测试 | 14 例 / 3 文件 | 单元 + 组件 + e2e（Playwright）+ 视觉回归 | 先给 5 条核心用户路径上 e2e |
| CI | 6 个 job 里 5 个是上游的 | 构建/测试/lint/类型 全硬门 + 缓存 | 删上游 job，lint 转硬门 |
| 可观测 | 前端 localStorage 环形缓冲；后端 print | Sentry/OTel + trace id + 指标看板 | 后端 trace id + 前端错误上报 |
| 发布 | `build_center_release.py --bump patch` + 手工 scp | 制品仓库 + 签名 + 回滚 | 已有 sha256，补"上一版可回滚" |

---

## 5. 改进路线（三条并行，不互相阻塞）

### A. 客户端收敛（目标：包体减半、产品面可定位、错误可见）

| 步骤 | 动作 | 验收标准 |
|---|---|---|
| A1 | 死码清理：51 个孤儿 + `.bak` 文件 | `vite build` 通过；`src` 行数 −12%；无 import 报错 |
| A2 | `src/legacy/` 分层 + 独立 build target | 主包不再含 App.jsx/blueprint；`dist` 3.9 MB → <2 MB |
| A3 | 基础设施上移 `src/infra/`（api/dataStore/ui/config） | `rdc_fusion/` 只剩面板 |
| A4 | 禁裸 fetch + 统一信封 + 5 个核心响应校验 | eslint 规则生效；`grep -c "fetch(" src` 只剩 api.js |
| A5 | 后端：全局异常处理器 + logging + trace id + 删 22 条死路由 | 任一 500 能凭 trace id 定位到行 |
| A6 | 每 tab 一个 ErrorBoundary | 单面板崩溃只黑一块，不白屏 |
| A7 | 5 条核心路径 e2e（启动→连接→打开空间→跑一次测试→看结果） | CI 里跑，PR 拦截 |

### B. 研发流程硬化（目标：任何一条 confirmed 结论可一键复算）

| 步骤 | 动作 | 验收标准 |
|---|---|---|
| B1 | `tools/` 入库；`tempdata/*.json` 改为 `results/` 入库（大文件用 LFS 或只存摘要） | `git ls-files tools \| wc -l` > 0 |
| B2 | `provenance()`：git_sha / git_dirty / model_path / model_revision / seed，所有结果 json 必带 | 抽查 3 份结果，5 字段齐全 |
| B3 | 账本加 `assert_fn`，46 条里先落 10 条可执行的 | CI 跑断言，全绿 |
| B4 | 每个 Phase 预注册模板加"功效分析"一栏 | 新 Phase 文档里有 n 的依据 |
| B5 | 二选一：把当前线**并入** `research_os`（contracts+manifests+runs），或**正式废弃** `research_os` 并把它的 schema 思路移植到 B2/B3 | 不再存在两套事实源 |
| B6 | `AGENTS.md` 的 memo 指针改为当前活跃线（现指向 claude，实际 gpt5） | 文档指针与实际一致 |
| B7 | 改账本/注册表的脚本一律"插入锚点 + 唯一性断言"，禁区间切片 | 写进 `AGENTS.md` 与 `rules_ingest.py` docstring |

### C. 科学能力补齐（目标：结论从"可读"升级到"可被外部检验"）

| 步骤 | 动作 | 验收标准 |
|---|---|---|
| C1 | 用因果 scrubbing 重做一次核心结论（格式单元那条） | 报出"行为保留比例"，而不是"打坏几道" |
| C2 | 引入 1 个外部方法基线（AtP* 或 SAE）在同一 169 题上 | 两种方法的发现重叠度可量化 |
| C3 | 第二个模型族跑同协议（只验"承载核/格式单元是否复现"） | 复现或明确不复现，都算结论 |
| C4 | 多 seed（≥3）重复 H13 | 报 seed 间方差，而非单点 |
| C5 | 把 MIB 的两个任务作为外部校准（可选，但能极大提升可信度） | 在公共任务上给出本方法的定位 |

---

## 6. 如果只做五件事

1. **`tools/` 入库 + 结果带 provenance**（B1+B2）。成本一天，收益是把"叙事"变成"证据"。这是本审计里唯一"不做就一切白做"的一条。
2. **删 51 个孤儿 / 14,923 行，legacy 独立构建**（A1+A2）。包体减半，搜索信噪比恢复。
3. **CI 从「跑上游测试」改成「跑自己的门」**（A4+A5+A7）：禁裸 fetch、lint 硬门、5 条 e2e。现在 CI 绿不代表产品能用。
4. **后端加异常处理器 + trace id**（A5）。一个 bug 的定位成本从小时降到分钟。
5. **账本 10 条规则变成可执行断言**（B3）。这是把"人肉纪律"转成"机器纪律"的第一步，也是这个项目最独特的资产。

---

## 7. 证据索引

| 结论 | 证据位置 / 命令 |
|---|---|
| `tools/` 未被跟踪 | `git ls-files tools \| wc -l` → 0；`git status --short tools` → `?? tools/` |
| 结果无 provenance | `core_L34.json` 顶层键无 seed/git/model_revision |
| 孤儿 51 个 / 14,923 行 | 本轮 import 反查脚本（按 stem 匹配 `./x` 与 `/x`） |
| 产品面 20% / 遗留 40% | `frontend/src` 117,744 行；`blueprint` 46,974；`components/app` 23,129 |
| 前端 14 例测试 | `npx vitest run` → 3 files / 14 tests passed |
| 后端 0 产品测试 | `grep -rln "from server" tests` 仅命中 `tests/codex` 的临时脚本；`tests/{unit,integration}` 46 文件 import `transformer_lens` |
| CI 6 个 job 里 5 个属上游 | `.github/workflows/checks.yml`；`release.yml:11` `uses: TransformerLensOrg/...` |
| 裸 fetch 208 处 | `grep -rc "fetch(" frontend/src` 汇总 |
| 后端 68 路由 / 0 异常处理器 / 164 print | `grep -c "@app"`；`grep "errorhandler"` 空；`grep -c "print("` |
| 22 条路由前端无引用 | 路由表 × 前端字符串比对 |
| `research_os` 空转 | `contracts/` 最新 2026-08-14；无 `runs/` 目录 |
| 文档指针漂移 | `AGENTS.md:249`（并在 3、225 处重复）指向 `research/claude/...`（Phase 2760），实际活跃为 `research/gpt5/...`（Phase 3280） |
| 统计功效不足 | `core_L34.json` → p=0.0654 / 0.0703 / 0.3438，唯一显著 tailN vs base 0.0129 |
| 判据类错误 5 次 | `data/rules/ledger.json`：R-MET-06/07/08/09/11 |
