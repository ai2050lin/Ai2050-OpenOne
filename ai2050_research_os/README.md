# AI2050 研究操作系统与可视化研发中心方案规范

## 0. 文件定位与强制读取声明

**本文件是 `ai2050_research_os` 的系统治理与总体架构规范，也是所有方案文件的唯一权威入口。技术研究框架和科学主线由 [TECHNICAL_FRAMEWORK.md](TECHNICAL_FRAMEWORK.md) 专门维护。**

关于研究操作系统、证据中心、JSON 数据模型、可视化客户端、AI 编排、运行审计、单机闭环、分布式执行和协作体系的总体方案统一维护在本文件中。`TECHNICAL_FRAMEWORK.md` 保存研究技术路线、测量对象、主线阶段和科学停止条件。其他文件可以保存 Schema、合同、执行细则、历史基线和自动生成结果，但不得另行建立与这两份规范冲突的总体方案或技术主线。

任何 AI、Agent、Codex 实例、开发者或用户在基于本项目进行二次开发之前，都必须：

1. 完整读取本文件和 `TECHNICAL_FRAMEWORK.md`；
2. 检查 `schemas/`、`registry/` 和相关合同，不得只根据客户端界面推断数据结构；
3. 确认修改没有建立新的平行事实源；
4. 修改总体架构、数据流、核心 Schema 或开发阶段时，同步更新本文件；
5. 运行 `researchctl.py validate`，并在实现完成后验证所有引用、状态和 Schema；
6. 保留历史证据，不得通过覆盖旧记录制造“最新结论”；
7. 遵守仓库更高层级指令和用户当前明确指令；发生冲突时，以更高层级指令为准。

### 0.1 本文件与数据文件的关系

本文件定义“系统应该怎样工作”，但不保存动态研究状态。动态事实必须保存在 JSON 注册表、合同、Run Bundle 和规范快照中。

```text
README.md       = 系统治理与总体架构规范
TECHNICAL_FRAMEWORK.md = 技术方案框架与科学主线
schemas/        = 机器可校验的数据规范
registry/       = 规范研究事实源
contracts/      = 冻结实验合同
runs/           = 不可变运行包（目标结构）
snapshots/      = 面向客户端的规范快照（目标结构）
generated/      = 从事实源生成的人类可读视图
Memo            = 从已裁决证据生成或追加的人类叙事，不是数据库
```

`generated/`、客户端 JavaScript 常量、Memo 和页面文案都不得反向覆盖 `registry/` 中的正式事实。

---

## 客户端优化记录：Canonical Snapshot 最小闭环 [2026-08-19 03:22]

### 本轮目标与完成结果

本轮针对客户端事实漂移完成了第一段可运行闭环。修改前，Registry 的正式当前状态为 Phase 1263 / C013，但客户端核心总览仍从 `frontend/src/researchKernel/currentResearchState.js` 读取硬编码的 Phase 1236 / C001，已经违反“当前状态只来自 Canonical Snapshot”的约束。

本轮新增并完成：

1. `schemas/snapshot.schema.json`：定义最小 `snapshot.v1` 合同；
2. `researchctl.py build-snapshot`：从全部 Registry 记录确定性构建 `snapshots/current/snapshot.json`；
3. `researchctl.py validate-snapshot`：同时校验 Schema 和 Registry 重建一致性，拒绝漂移快照；
4. `researchctl.py export-client`：只有快照校验通过后才导出 `frontend/public/research_snapshot.json`；
5. `useResearchSnapshot()`：客户端统一读取接口，使用 `no-store` 并显式暴露加载和错误状态；
6. `ResearchProgressTab` 与 `ProjectRoadmapTab`：当前 Phase、Campaign、瓶颈和下一裁决已切换到 Canonical Snapshot；
7. 根目录发现不再依赖固定 `parents[n]`；历史 Manifest 的旧前缀通过只读路径映射解析，不修改历史记录及其哈希。

快照身份由 Registry 的规范 JSON 计算：

```text
source_sha256 = SHA256(canonical_json(all_registry_records))
snapshot_id = SNAPSHOT-{project.as_of}-{source_sha256[0:12]}
```

同一 Registry 输入必须生成逐字段完全相同的 Snapshot；任何 Registry 改动都会改变 `source_sha256`，旧客户端导出将被 `validate-snapshot` 判定为漂移。

### 验证结果

2026-08-19 03:22（America/Chicago）执行结果：

- `researchctl.py build-snapshot`：通过，生成 `SNAPSHOT-2026-08-13-66650dcad5e5`；
- `researchctl.py validate-snapshot`：通过；
- `researchctl.py export-client`：通过；
- `python -m unittest tests.glm5.test_research_snapshot -v`：4/4 通过；
- `researchctl.py validate`：通过，校验 13 Campaign、28 Phase、30 Evidence、16 Decision、16 Contract、16 Run 和 91 Artifact；
- `git diff --check`：通过；
- 前端生产构建：当前执行环境没有 Node/npm 可执行文件，未能运行，不得记为通过。

本轮没有运行模型实验，也没有增加语言机制证据。工程闭环只降低客户端状态漂移风险，不能提升任何 Claim 的证据等级。

### 严格审视：问题、硬伤与瓶颈

1. 当前只迁移了两个核心总览页；`SystemStatusTab`、`DeepAnalysisTab` 和部分历史看板仍读取硬编码状态，客户端硬编码当前 Phase 数量尚未降到 0；
2. `RESEARCH_EVIDENCE_GATES` 仍是 JavaScript 常量，虽然当前 Phase 已不依赖它，但门控卡片仍可能与 Registry 漂移；
3. Snapshot v1 是最小合同，只含项目、当前状态、计数和来源；尚未包含完整 Claim—Evidence—Decision 关系图和 `view_spec`；
4. 客户端导出目前需要显式执行命令，尚未接入构建前强制检查；
5. 前端缺少 Node/npm 的环境验证，JSX 生产编译结果仍待确认；
6. 历史 Manifest 路径映射已恢复校验，但尚未生成独立、机器可读的迁移报告。

### 理论边界与关键洞察

本轮只建立“研究状态如何被可靠读取”的工程结构。它的重要性在于：如果客户端展示的是旧 Phase，研究者会基于错误瓶颈设计实验，后续再严格的统计或因果测试也失去前提。单一事实源不是语言数学理论本身，但它是拼接大量可靠局部证据的必要条件。

当前关于智能理论可以保留的最小洞察是：语言机制研究必须把“事实身份、观测身份和解释身份”分开。Registry 决定事实身份，Snapshot 提供可验证的观测投影，客户端文本只负责解释；解释不得反向覆盖事实。只有长期保持这条单向数据流，未来从大量响应、干预和失败边界中浮现的结构才可被可信比较。

### 下一阶段大任务

下一阶段不应继续零散修改页面，而应完成“客户端当前状态零硬编码”任务包：

1. 把 Evidence Gate、Claim、Experiment、Run、Decision 和 Artifact 摘要纳入 Snapshot Schema；
2. 将剩余核心页统一迁移到 `useResearchSnapshot()`，删除 `CURRENT_RESEARCH_STATE` 的事实源职责；
3. 新增 `drift-audit`，扫描客户端中的当前 Phase、当前 Campaign、当前瓶颈和下一任务常量；
4. 生成只读路径迁移报告，记录旧路径、新路径、内容哈希和映射状态；
5. 将 `validate-snapshot` 和客户端导出接入前端构建前置步骤；
6. 恢复 Node/npm 环境后完成 lint、生产构建和浏览器加载验证，再把该阶段判为客户端最小闭环完成。

### 后续优化完成记录 [2026-08-19 03:50]

在上述最小闭环基础上，本轮继续完成了附件方案与系统规范的第一阶段收敛：

- 用户层继续保留 3D 空间、理论分析 Overlay、AI 研发 Sidebar 三个入口；代码层由 `ResearchCenter` 以 `theory` / `rnd` 两种模式复用同一实现；
- Snapshot 新增 Hypothesis、Puzzle、Evidence、Run、Decision 摘要及 R/M/P/E 信息架构元数据；
- 客户端导出路径统一为 `frontend/public/research_data/current/snapshot.json`，旧路径副本已移除；
- 理论路线、项目路线、研究进度、系统状态、AI 研发控制台、3D 证据链、证据 Cockpit 和机制来源面板均已迁移到 `useResearchSnapshot()`；
- 删除 `currentResearchState.js` 平行事实源；
- 新增 `researchctl drift-audit`，机器拒绝旧状态模块、硬编码 Evidence Gate 和旧快照路径重新进入客户端；
- 全量 Registry 校验、Snapshot 校验、漂移审计、5 项专项测试和 `git diff --check` 均通过。

当前“客户端硬编码当前 Phase = 0”已经在受审计的正式状态入口范围内达到。历史实验页面仍可显示其自身 Phase，因为那是 Run/Artifact 的历史身份，不属于“当前研究状态”硬编码。

仍未完成的部分是第二阶段通用研究工作台、完整 `view_spec.json` 驱动的 3D Overlay、AI 研发持久化闭环，以及缺少 Node/npm 导致的前端生产构建验证。这些事项不得被本轮的数据内核通过状态掩盖。

### 客户端极简化约束与实现 [2026-08-19 04:06]

客户端停止把“能够展示的数据”全部放入默认界面。默认界面固定只回答三个问题：

```text
当前状态是什么？
最近证据是什么？
下一步是什么？
```

3D 空间、理论分析和 AI 研发三个用户入口继续保留，但理论分析和 AI 研发默认使用同一个极简 `ResearchCenter`：

- 理论分析默认只显示 Phase、Campaign、状态、当前瓶颈、最近四条 Evidence 和下一项 Decision；
- AI 研发默认显示相同事实摘要，只保留一个“打开高级执行控制”按钮；
- 原多标签研发界面仅在用户明确进入高级控制时按需加载，不进入默认 JavaScript 加载路径；
- 旧路线、历史 Dashboard 和详细配置可以继续作为历史或高级能力存在，但不得重新进入默认导航；
- 不新增首页 KPI、装饰性图表、综合完成度或不能改变决策的信息卡。

复杂功能的准入规则固定为：只有当功能直接支持“选择证据、执行冻结合同或追溯 Run”之一时，才允许进入默认界面。其他功能按需加载或保留在历史代码中。`sites-building` 的现有站点能力路径促使本轮采用一个主表面、减少默认客户端状态并延迟加载高级控制；没有改变 Research OS 的证据规则。

本次精简没有删除历史实验、Registry、Run、Evidence 或高级研发能力，也没有增加任何科学证据。由于当前环境仍无 Node/npm，生产构建和浏览器视觉检查尚未完成。

### 研发框架主导航与真实状态热力图 [2026-08-19 04:17]

客户端的默认组织中心正式从 Phase 改为研发框架、Puzzle、Evidence 和重要结果类型。Phase 继续保存在 Registry、Run 和 provenance 中，但只承担历史坐标与来源追溯职责。

本轮新增 `registry/framework.json`，正式登记：

- R0–R8 研发阶段；
- 当前阶段 R1；
- 每个阶段的状态、实际进度和下一道资格门；
- 词嵌入状态、Hidden State、响应轨迹、因果路径、干预救援、复杂度曲面和跨模型映射等重要结果类型。

理论中心默认显示研发框架横向进度，不显示当前 Phase。3D 主空间的标题和来源面板改为显示结果类型与当前 R 阶段；Phase 只允许在用户进一步打开完整 Run 来源时出现。新增专项审计，防止默认研究中心、3D 证据层和热力图重新以 Phase 为标题。

热力图研究路线从视觉示例改为真实 Trace 驱动：

```text
Layer 左侧：当前 Token 的 embedding Top-K 向量分量
Layer 区域：各层 residual2（缺失时按冻结优先级回退）的 Hidden State Top-K 分量
暖色：正分量
蓝色：负分量
暗色：该维度未进入已采集 Top-K，属于未观测，不是数值零
```

维度选择只使用实际 `top_units`：embedding 与 Hidden State 分别选择自己的观测维度；Hidden State 优先选择跨层重复出现的维度，再按最大幅度和稳定编号排序。所有颜色使用同一 Trace 内冻结的最大绝对分量归一化。视图保存 Run ID 供来源追溯，但默认标题不展示带 Phase 的历史 Run 名称。

严格证据边界：该热力图只显示已经采集的稀疏 Top-K 摘要，不能恢复完整向量；暗色格不能解释为零激活；颜色相似不能自动解释为语义相同、机制相同或因果关系。当前只具有内部响应观察资格，因果结论仍需冻结干预、负控和救援实验。

验证结果：Research OS、Canonical Snapshot、客户端漂移审计和 7 项专项测试通过。当前环境仍缺少 Node/npm，因此生产构建和浏览器视觉检查未完成，不得记为前端发布通过。

---

## 一、系统定位

本系统的目标是建设一个统一的“AI 研发中心”，把目前分散的 Phase、Campaign、实验合同、脚本、结果、审计、理论拼图、可视化客户端、Codex、其他 AI 和计算节点整合为一个可追溯系统。

完整闭环为：

```text
研究规划
→ 合同冻结
→ 运行前审计
→ 受控执行
→ 独立复算
→ 证据聚合
→ Claim 裁决
→ 规范快照
→ 客户端与报告更新
```

系统角色划分：

- 可视化客户端是研发控制中心和证据浏览器；
- Codex、其他 AI、本地 GPU 和远程节点是受控执行者；
- JSON 证据中心是唯一事实来源；
- Memo 是人类可读研究记录，不承担查询和状态同步职责；
- Phase 是正式科学决策节点，不等于一次工程重试。

系统不以“产生更多 Phase”为目标，而以“改变核心科学命题的证据状态”为目标。

---

## 二、核心研究组织方式

### 2.1 命题中心，而不是 Phase 中心

正式研究对象按以下层级组织：

```text
Project
└── Research Direction
    └── Campaign
        └── Claim
            └── Experiment
                └── Run
                    ├── Metrics
                    ├── Artifacts
                    ├── Audit
                    ├── Evidence Delta
                    └── Decision
```

- `Claim` 是需要支持、限制或反证的科学命题；
- `Experiment` 是冻结的检验设计；
- `Run` 是一次具体执行；
- `Evidence` 只记录实际观测；
- `Decision` 记录为什么升级、停止、反证或拒答；
- `Phase` 只在形成正式科学裁决时增加；
- 启动失败、依赖修复、日志补采等工程动作不得自动建立新 Phase。

### 2.2 证据阶梯

机制主张必须沿固定证据阶梯升级：

```text
L0 问题登记
→ L1 行为资格
→ L2 内部响应可读
→ L3 未见响应可预测
→ L4 组件因果证据
→ L5 阻断、救援和功能状态闭环
→ L6 完整自回归生成闭环
→ L7 组合、迁移和跨模型复现
→ L8 训练形成与统一理论闭环
```

前一级没有通过时，后一级结果不得登记为正式机制证据。可视化上的热点、聚类、降维和轨迹只能继承源证据等级，不能自动升级证据。

### 2.3 探索与确认分离

- 探索数据用于发现候选、指标和可区分实验；
- 确认数据必须独立、冻结并一次性裁决；
- 封存确认集不得用于选层、选头、选神经元、调阈值或改指标；
- AI 提出的解释只能成为候选 Claim 或 Hypothesis；
- 重要结论必须具备负控、错误身份、泄漏审计、独立复算和明确停止条件。

---

## 三、当前工程基础与必须解决的问题

项目已经具备可复用基础，不应推倒重建：

1. `registry/` 已经登记 Campaign、Hypothesis、Puzzle、Evidence、Decision、Run 和 Artifact；
2. `contracts/` 与 `schemas/` 已具备实验冻结和机器校验基础；
3. 服务端 Research Kernel 已能读取部分 Run Bundle、Claim 和 Gap；
4. 前端已经具备热图、轨迹、因果链、Atlas、3D 图谱等渲染器；
5. 数据源注册表和适配器证明客户端可以由 JSON 驱动。

但二次开发必须首先解决以下结构性问题：

### 3.1 多事实源漂移

研究状态曾同时存在于客户端 JavaScript、Research OS、公共可视化副本和 Memo 中，导致“当前 Phase”“当前瓶颈”和“下一任务”可能不一致。

目标约束：

```text
正式研究事实源数量 = 1
客户端硬编码当前 Phase 数量 = 0
Memo 反向更新 Registry 的路径数量 = 0
```

### 3.2 页面与 Phase 过度耦合

不得继续为每个 Phase 新建专用 Dashboard。新增实验应优先输出统一 JSON 和 `view_spec.json`，由通用组件渲染。

### 3.3 Schema 碎片化

历史数据允许保留原 Schema，但进入规范快照前必须适配到有限的核心 Schema。不得为每个新 Phase 无限新增互不兼容的顶层 Manifest 类型。

### 3.4 服务端和客户端保存两份证据

正式证据只保存一次。客户端需要的静态 JSON 必须由事实源自动导出，并带源哈希和快照编号，不能人工复制后独立维护。

### 3.5 运行状态缺少持久化

运行队列、状态迁移、租约、审计和结果登记必须持久化。进程内对象只能作为缓存，不能作为正式 Run 状态源。

---

## 四、总体系统架构

```text
┌──────────────────────────────────────────────────────┐
│ 人类研究者 / 客户端 / Codex / Planner AI             │
└───────────────────────┬──────────────────────────────┘
                        │ Contract API
                        ▼
┌──────────────────────────────────────────────────────┐
│ 控制中心                                               │
│ 合同冻结 · 权限 · 预算 · 决策树 · 停止条件 · 调度       │
└───────────────────────┬──────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────┐
│ 执行中心                                               │
│ 本机脚本 · Codex · 本地GPU · 远程Worker · 云节点        │
└───────────────────────┬──────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────┐
│ 不可变 Run Bundle                                      │
│ Manifest · Metrics · Audit · Claim Delta · View Spec  │
└───────────────────────┬──────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────┐
│ JSON 证据中心                                          │
│ Registry · Schema · Provenance · Decision · Snapshot  │
└───────────────┬──────────────────────┬───────────────┘
                ▼                      ▼
       可视化客户端/API          Memo/看板/报告生成
```

采用“一套后端、两个操作入口”原则：

- 用户可以通过客户端操作；
- Codex 或脚本可以通过 CLI/API 操作；
- 两者必须调用同一 Contract API；
- 两者产生相同格式的 Run Bundle；
- 所有结果进入同一个证据中心。

---

## 五、JSON 唯一事实源规范

### 5.1 总原则

项目核心数据尽量采用 JSON，使客户端、服务端、CLI 和 AI 都能使用相同数据契约。

适合 JSON 的数据包括：

- Project、Direction、Campaign、Claim；
- Experiment Contract、Run Manifest；
- 聚合指标、门槛、门控结果；
- Evidence、Audit、Decision、Correction；
- Artifact 索引、来源和哈希；
- Term、Puzzle、Hypothesis；
- Current Snapshot 和 View Spec。

不适合强行放入普通 JSON 的大型数据包括：

- 大规模逐样本矩阵；
- Hidden state、Attention、梯度和激活张量；
- 超大事件流；
- 模型权重与二进制检查点。

这些数据可以使用 JSONL、Parquet、NPY、Safetensors 或对象文件，但必须由 JSON Manifest 描述，客户端首先读取 JSON，再按需请求大型 Artifact。

### 5.2 JSON 通用字段

所有正式 JSON 记录至少包含：

```json
{
  "schema_version": "record_type.v1",
  "id": "全局稳定主键",
  "created_at": "2026-08-14T12:00:00-05:00",
  "updated_at": "2026-08-14T12:00:00-05:00",
  "status": "合法状态枚举",
  "source_refs": [],
  "correction_of": null
}
```

强制约束：

1. 时间使用带时区的 ISO 8601；
2. 禁止使用 `NaN`、`Infinity` 和 `-Infinity`，缺失值使用 `null`；
3. 哈希使用小写十六进制 SHA-256；
4. ID 一旦公开引用就不能复用；
5. Schema 变更必须提升版本并提供迁移器；
6. 正式记录不允许静默覆盖，只能追加新版本或 Correction；
7. 所有比例必须同时保存分子和分母，不能只保存百分比；
8. 证据必须记录适用范围、允许推出什么、禁止推出什么。

### 5.3 规范核心实体

#### Claim

```json
{
  "schema_version": "claim.v1",
  "claim_id": "CLM-C001-001",
  "title": "命题标题",
  "plain_explanation": "通俗解释",
  "formal_definition": "可选数学定义",
  "status": "candidate",
  "closure_level": 0,
  "scope": {
    "models": [],
    "tasks": [],
    "languages": [],
    "protocols": []
  },
  "supporting_evidence": [],
  "opposing_evidence": [],
  "limitations": [],
  "next_decisive_experiment": null
}
```

建议 Claim 状态：

```text
unstarted
exploring
candidate
confirmed_within_scope
unconfirmed
bounded_rejected
closed_negative
retired
```

#### Experiment Contract

正式实验合同必须包含：

- 研究对象与命题；
- 数据、分区和泄漏规则；
- 模型、精度、种子和环境；
- 控制、零模型和错误身份；
- 指标、分母和阈值来源；
- 通过门、失败门和拒答门；
- 资源预算和右删失规则；
- 允许的后续分支；
- 独立审计要求；
- 合同哈希和冻结产物。

不同用途的合同必须通过 `contract_type` 区分，例如：

```text
full_experiment
continuation_checkpoint
recovery_contract
audit_only
client_export
```

不得让轻量恢复合同伪装成完整实验合同，也不得使用一个 Schema 强行解释所有合同类型。

#### Evidence

Evidence 只写实际观测，不写希望成立的理论：

```json
{
  "schema_version": "evidence.v1",
  "evidence_id": "E-C001-001",
  "polarity": "positive",
  "grade": "E2",
  "closure_level": 2,
  "claim": "实际观测内容",
  "scope": {},
  "authorizes": [],
  "forbids": [],
  "run_refs": [],
  "contract_sha256": "...",
  "result_sha256": "...",
  "audit_sha256": "..."
}
```

#### Decision

Decision 必须记录：

- 输入证据；
- 适用合同；
- 完整门控结果；
- Claim 状态变化；
- 授权或禁止的下一步；
- 是否允许自动继续；
- 人工审批人或自动裁决器版本。

---

## 六、标准 Run Bundle

每次正式执行必须生成统一运行包：

```text
runs/{run_id}/
├── manifest.json
├── contract.json
├── metrics.json
├── claims_delta.json
├── audit.json
├── decision.json
├── view_spec.json
├── provenance.json
└── artifacts/
    ├── raw_results.jsonl
    ├── sample_metrics.parquet
    ├── tensors.safetensors
    └── logs.jsonl
```

### 6.1 `manifest.json`

Manifest 至少登记：

```json
{
  "schema_version": "run_bundle.v1",
  "run_id": "RUN-EXP-C001-WP01-001-001",
  "contract_id": "EXP-C001-WP01-001",
  "status": "adjudicated",
  "model": {},
  "data": {},
  "environment": {},
  "timing": {},
  "artifacts": {},
  "counts": {},
  "hashes": {}
}
```

### 6.2 `metrics.json`

客户端默认读取聚合指标。每个指标保存：

- `metric_id`；
- 定义和单位；
- 分子、分母和值；
- 数据分区；
- 阈值和阈值来源；
- 是否通过；
- 非有限值计数；
- 可视化建议。

### 6.3 `audit.json`

独立审计至少重新检查：

- 样本数和分母；
- 指标重算；
- 门控逻辑；
- 数据泄漏；
- dtype 和非有限值；
- 文件哈希；
- 运行身份和环境；
- 最终授权是否符合合同。

执行器不能自行把自己的运行标为“正式通过”；正式状态必须由审计和裁决共同产生。

---

## 七、规范快照与客户端数据入口

### 7.1 Current Snapshot

客户端启动时只加载一个稳定入口：

```text
snapshots/current/snapshot.json
```

部署为静态客户端数据时可以导出为：

```text
frontend/public/research_data/current/snapshot.json
```

示例：

```json
{
  "schema_version": "research_snapshot.v1",
  "snapshot_id": "SNAP-20260814-0001",
  "generated_at": "2026-08-14T12:00:00-05:00",
  "source_registry_hash": "...",
  "project": {
    "project_id": "ai2050-language-mechanism",
    "latest_phase": 0,
    "active_campaign_id": null,
    "current_bottleneck": "",
    "next_decision": "",
    "auto_continue": false
  },
  "counts": {},
  "refs": {
    "directions": "directions.json",
    "claims": "claims/index.json",
    "campaigns": "campaigns/index.json",
    "experiments": "experiments/index.json",
    "runs": "runs/index.json",
    "terms": "terms.json"
  }
}
```

`snapshot.json` 必须是 Registry 和已裁决 Run 的确定性构建产物，禁止人工修改。

### 7.2 漂移拒绝规则

快照构建必须检查：

- Registry 最新 Phase；
- 最新已裁决 Decision；
- Run、Evidence、Claim 引用；
- 导出客户端快照；
- Memo 最新正式记录（只做漂移报警，不以 Memo 覆盖 Registry）。

出现无法解释的阶段或哈希不一致时，构建必须失败，而不是继续发布旧客户端数据。

### 7.3 静态与在线双模式

客户端支持两种读取方式，但数据结构相同：

1. 静态模式：读取 `frontend/public/research_data/` 导出快照；
2. 在线模式：通过 `/api/v2/research/...` 获取相同 JSON。

静态导出是可重建副本，不是第二事实源。每个导出文件必须带快照 ID 和源哈希。

---

## 八、通用可视化规范

### 8.1 `view_spec.json`

每个实验必须提供可视化描述或明确声明只适合结构化表格：

```json
{
  "schema_version": "view_spec.v1",
  "views": [
    {
      "view_id": "layer-token-heatmap",
      "view_type": "heatmap",
      "title": "Layer × Token 响应强度",
      "dataset_ref": "metrics.json#/layer_token_matrix",
      "encoding": {
        "x": "token_position",
        "y": "layer",
        "color": "response_delta"
      },
      "evidence_level": "L2",
      "warning": "观测差分不等于因果机制"
    }
  ]
}
```

### 8.2 结果类型与默认视图

| 结果类型 | 默认视图 |
|---|---|
| 门控、状态和标量 | 判决卡 |
| 模型 × 任务 × 种子 | 热图 |
| 层间变化 | 轨迹图 |
| Token × Layer | 二维热图 |
| 干预与零模型 | 配对分布 |
| 训练检查点 | 动态轨迹 |
| Claim 与证据关系 | 证据图 |
| 高维表征 | PCA/UMAP 导航图 |
| 物理单元与组件路径 | 2D/3D Atlas |
| 不适合画图 | 结构化表格 |

高维降维图只用于导航，不能自动成为科学证据。

### 8.3 禁止 Phase 专用页面扩散

新增实验的默认实现顺序是：

```text
输出标准 JSON
→ 选择已有 view_type
→ 配置 view_spec
→ 复用通用渲染器
```

只有当一种结果结构无法由现有视图表达，并且预计会被多个实验复用时，才允许新增渲染器。不得因为一个 Phase 有新字段就创建新的大型 Dashboard。

---

## 九、客户端信息架构

客户端采用逐层折叠和按需加载，默认不显示全部原始结果。

### 9.1 项目总览

只显示：

- 当前主任务；
- 最新正式裁决；
- 最大瓶颈；
- 活跃 Campaign；
- 正在运行的任务；
- 当前允许和禁止的后续操作。

### 9.2 研究地图

固定显示主要研究方向：

1. 行为与语义资格；
2. 静态表征与编码库存；
3. 状态转移与层间动力学；
4. Attention/MLP 组件因果；
5. 阻断、救援与组合；
6. 自回归自然生成；
7. 跨任务、种子和模型守恒；
8. 统一理论与数学闭合。

每个方向由 JSON Claim、Puzzle 和 Gap 聚合，不在 JSX 中写死进度。

### 9.3 Claim 页面

必须回答：

- 我们认为是什么；
- 已经观测到什么；
- 证据达到哪一级；
- 在什么范围成立；
- 有哪些反证和硬伤；
- 哪个实验最能改变当前状态。

### 9.4 Experiment 页面

必须显示：

- 测试原理和通俗例子；
- 材料、分区和控制；
- 公式和指标；
- 通过门、失败门和停止条件；
- 预算和权限；
- 所有 Run 与裁决。

### 9.5 Run 页面

显示运行状态、节点、环境、日志、聚合指标、原始 Artifact 引用、哈希和独立审计。

### 9.6 证据图谱

以以下关系为主图：

```text
Claim → Experiment → Run → Evidence → Decision
```

Phase 只是过滤和时间索引，不是主导航树。

### 9.7 Reader / Expert 双模式

- Reader Mode：通俗结论、核心图、证据边界和下一步；
- Expert Mode：公式、阈值、分母、控制、原始结果、哈希和审计。

两种模式读取同一 JSON，不得分别维护两套结论。

### 9.8 三入口与统一 Research Center

客户端在用户层保留三个稳定入口，不合并成单一大页面：

| 入口 | 职责 | 表现形式 |
|---|---|---|
| 3D 空间 | 查看真实实验轨迹、组件响应、干预差异和证据对象 | 现有全屏 Layer 3D 主空间 |
| 理论分析 | 组织框架、主线、拼图、Claim、规律和证据边界 | 全屏 Overlay |
| AI 研发 | 管理 Campaign、合同、运行前审计、执行、审计和裁决 | 侧边工作台 |

代码层将“理论分析”和“AI 研发”统一为 `ResearchCenter`，分别使用 `theory` 和 `rnd` 模式；这只是组件与数据流复用，不取消两个用户入口。3D、理论分析和 AI 研发必须读取同一 Canonical Snapshot。

三部分的单向闭环为：

```text
理论分析选择 Puzzle / Claim
→ AI 研发冻结并执行 Experiment
→ Run Bundle 生成 View Spec
→ 3D 空间展示可追溯证据
→ Evidence 与 Decision 更新 Snapshot
→ 理论分析读取新裁决
```

### 9.9 R/M/P/E 客户端命名规则

为避免阶段、模块、拼图和证据等级混淆，客户端采用以下稳定前缀：

- `R0–R8`：研究主线阶段；
- `M0–M9`：长期研究能力模块，不代表时间顺序；
- `Pxx`：研究拼图；
- `Ex`：证据等级；实验主键必须完整显示为 `experiment_id`，不得仅显示 `E001`。

`TECHNICAL_FRAMEWORK.md` 中历史使用的 `M0–M8` 阶段名称在客户端展示时只读映射为 `R0–R8`。映射不得修改历史 Registry、合同、Evidence 或 Decision 的身份，也不得提升证据等级。当前 Evidence grade、closure level 和 R 阶段是不同维度，禁止互相推导。

---

## 十、API 与客户端实现约束

### 10.1 API v2 目标

建议逐步收敛为：

```text
GET  /api/v2/research/snapshot/current
GET  /api/v2/research/directions
GET  /api/v2/research/claims
GET  /api/v2/research/claims/{claim_id}
GET  /api/v2/research/campaigns/{campaign_id}
GET  /api/v2/research/experiments/{experiment_id}
GET  /api/v2/research/runs/{run_id}
GET  /api/v2/research/runs/{run_id}/metrics
GET  /api/v2/research/runs/{run_id}/audit
GET  /api/v2/research/runs/{run_id}/view-spec
POST /api/v2/research/experiments/{experiment_id}/runs
```

### 10.2 前端数据访问

- API 根地址只能由统一配置模块提供；
- 组件中禁止新增硬编码 `localhost:5001` 或 `localhost:8000`；
- 所有请求经过统一客户端，处理超时、取消、缓存和错误；
- 组件不得直接拼接 Run Artifact 的物理路径；
- 当前研究状态通过 `useResearchSnapshot()` 一类统一 Hook 获取；
- 客户端不得直接解析 Memo 生成进度。

### 10.3 兼容旧数据

旧 API、旧 Manifest 和旧可视化数据通过 Adapter 进入规范模型。Adapter 只能做字段映射和证据等级继承，不能：

- 补造不存在的证据；
- 把观察边升级为因果边；
- 把聚合单元伪装成单神经元；
- 删除失败记录；
- 改写原始分母。

---

## 十一、自动研发工作流

### 11.1 建立 Campaign

用户配置主问题、预算、可用模型、最大自动分支和总停止条件。

### 11.2 AI 生成有限决策树

Planner 只能生成候选方案。用户确认后冻结：

- 行为合同；
- 零模型；
- 确认实验；
- 因果实验；
- 复现条件；
- 正负结果分支。

### 11.3 运行前审计

必须自动检查：

- Gold 是否唯一或允许拒答；
- 材料是否自然；
- 数据是否泄漏；
- Token 和接口条件是否一致；
- 分区是否独立；
- 合同是否完整；
- 环境和代码是否冻结；
- 预算和权限是否允许执行。

### 11.4 执行

调度器按节点能力分配任务。涉及本地模型测试时，Qwen3、GLM4、DS7B 必须依次运行，不得同时占用 GPU 导致显存溢出。

### 11.5 独立审计与证据更新

Auditor 从原始结果重算指标和门控，随后证据归并器生成：

- `claims_delta.json`；
- `audit.json`；
- `decision.json`；
- 新的规范 Snapshot；
- 客户端视图；
- 必要时生成 Memo 草稿。

### 11.6 自动续行边界

系统只能沿冻结分支自动继续。以下行为必须人工确认：

- 改变研究对象；
- 修改正式门槛；
- 启动新 Campaign；
- 扩大计算预算；
- 跨模型升级；
- 升级核心理论；
- 删除、公开或迁移敏感数据。

主门失败、预算耗尽、工程错误不可恢复、新对象不在冻结决策树或需要新权限时必须停止。

---

## 十二、AI 与 Codex 集成

AI 注册记录至少包含：

```text
提供方
模型名称与版本
角色
提示词版本
权限
上下文限制
费用与并发限制
工具权限
输出 Schema
```

AI 角色分离：

- Planner：设计有限研究方案；
- Executor：编写并运行已授权实验；
- Auditor：独立复算，不能复用 Executor 的裁决文本；
- Explainer：生成通俗说明；
- Adjudicator：依据冻结合同裁决授权。

AI 的正式输出必须符合 JSON Schema。自由文本可以作为解释，但不能直接改变 Claim、Run 或 Decision 状态。

Codex 标准流程：

```text
读取本 README
→ 读取当前 Snapshot 与合同
→ 检查权限和停止条件
→ 领取合同
→ 执行与保存检查点
→ 生成 Run Bundle
→ 请求独立审计
→ 同步证据中心
```

---

## 十三、分布式执行边界

分布式功能必须在 JSON 单机闭环稳定之后建设。否则只会把当前的数据漂移复制到更多节点。

### 13.1 中心节点

负责：

- 任务队列；
- 节点注册和能力目录；
- 运行租约与心跳；
- 超时恢复；
- 结果去重；
- 合同和结果哈希核验；
- 证据汇总。

### 13.2 Worker

登记 CPU、GPU、RAM、已有模型、dtype、数据位置、网络、存储和可信等级。

### 13.3 安全原则

- API 密钥加密；
- 节点最小权限；
- 通信身份认证；
- 结果哈希校验；
- 不可信节点抽样复算；
- 敏感原始数据允许只保存在本地；
- 中心只汇总必要统计和 Artifact 引用。

中心不得简单平均不同模型、任务、种子、节点和环境的结果。

---

## 十四、目录目标结构

当前目录继续兼容现有结构，目标逐步扩展为：

```text
ai2050_research_os/
├── README.md                 # 本方案规范，二次开发前强制阅读
├── TECHNICAL_FRAMEWORK.md    # 技术方案框架与科学主线，二次开发前强制阅读
├── schemas/                  # 所有核心 JSON Schema
├── registry/                 # 唯一事实源
├── contracts/                # 冻结合同
├── manifests/                # 冻结清单与哈希
├── runs/                     # 不可变 Run Bundle
├── snapshots/                # 确定性规范快照
│   └── current/
│       └── snapshot.json
├── adapters/                 # 历史 Schema → 规范 Schema
├── exporters/                # 客户端、Memo、报告导出器
├── templates/                # 新记录模板
├── scripts/
│   └── researchctl.py
├── docs/                     # 执行细则和历史基线，不另立总体方案
└── generated/                # 自动生成，禁止手工修改
```

---

## 十五、历史数据迁移

历史 Phase 和 Artifact 不删除，按以下步骤迁移：

1. 扫描 Phase、Campaign、合同、脚本和结果；
2. 处理 Phase 编号冲突，使用稳定 `record_id`；
3. 合并同义 Claim 和术语，但保留来源；
4. 建立 Claim—Experiment—Run—Evidence—Decision 关系；
5. 将旧结果标为支持、反对、限制、校准或退役；
6. 对历史 Schema 编写只读 Adapter；
7. 生成第一份 Canonical Snapshot；
8. 客户端默认只加载 Snapshot，历史证据按需展开；
9. AI 只接收当前问题和相关证据的最小上下文包，不再读取整个 Memo。

迁移不是“把所有历史结论都升级为正式证据”。缺少合同、原始结果、分母或独立审计的历史记录必须保留较低证据等级。

---

## 十六、实施路线

### 阶段一：JSON 证据内核统一

这是当前最高优先级。

任务：

1. 定义 Snapshot、Claim、Campaign、Experiment、Run、Evidence、Audit、Decision、View Spec Schema；
2. 为合同增加 `contract_type` 和分类型 Schema；
3. 扩展 `researchctl validate`，检查 Memo、Registry、Decision 和客户端快照漂移；
4. 实现 `researchctl export-client`；
5. 生成 `snapshots/current/snapshot.json`；
6. 客户端使用统一 `useResearchSnapshot()`；
7. 停止在 JavaScript 中维护当前 Phase、当前瓶颈和下一任务。

阶段验收：

```text
正式事实源 = 1
客户端硬编码当前 Phase = 0
Snapshot 可确定性重建 = true
所有引用有效 = true
状态漂移 = 0
```

### 阶段二：通用研究工作台

- 建立项目总览、研究地图、Claim、Experiment、Run 和证据图谱页面；
- 建立 Reader/Expert 双模式；
- 用 `view_spec.json` 驱动现有渲染器；
- 禁止继续增加一次性 Phase Dashboard。

### 阶段三：单机研发闭环

- Run 状态持久化；
- 打通合同冻结、执行、检查点、审计、证据更新和快照重建；
- 客户端和 Codex 使用相同 API；
- Memo 从已裁决 JSON 生成，不再承担数据库功能。

### 阶段四：分布式执行

- 中心调度器；
- Worker 节点；
- 租约、心跳、检查点和恢复；
- 结果去重和抽样复算；
- 保留节点、模型、任务、种子和环境差异。

### 阶段五：多人协作

- 用户和角色权限；
- 合同审查；
- 评论和复现请求；
- 贡献记录；
- 私有与公开项目；
- 可复现实验包导出。

### 阶段六：高级自动研究

- 自动 Gap 分析；
- 决定性实验排序；
- 跨模型复现编排；
- 信息增益建议；
- 总体封存测试；
- 严格受控的自动续行。

---

## 十七、开发禁止项

以下做法违反本方案：

1. 在 React/JSX 中手工写入“当前 Phase”并将其当作正式状态；
2. 让客户端、Memo 和 Registry 分别维护一套当前结论；
3. 为单个 Phase 新建不可复用的大型 Dashboard；
4. 没有 Schema 版本就新增 JSON；
5. 通过 Adapter 提升证据等级；
6. 在组件中新增硬编码服务地址；
7. 直接覆盖历史 Evidence、Run、Decision 或 Artifact；
8. 把大型 Hidden State 数组直接塞入项目总快照；
9. 在单机闭环未完成前优先开发分布式调度；
10. 未读取本文件就对 `ai2050_research_os` 进行架构性二次开发。

---

## 十八、验收标准

系统逐步完成后应满足：

1. 同一实验可从客户端或 Codex 启动，合同哈希完全一致；
2. 每个正式 Run 都有 Manifest、Metrics、Audit、Decision 和 View Spec；
3. 任一核心 Claim 可在三次点击内查看完整证据链；
4. 页面默认不加载海量原始数据；
5. 每项结果都有图形或结构化表格后备视图；
6. 客户端当前状态完全来自 Canonical Snapshot；
7. Snapshot、Registry 和裁决账不存在不可解释漂移；
8. 自动流程不能越过冻结停止门；
9. 历史 Phase 可追溯，但不再主导当前导航；
10. 新 AI、新实验和新图表可以通过 Schema、合同和插件扩展；
11. 多节点运行时不会重复汇总，也不会丢失模型和环境差异；
12. 系统能直接回答：

```text
有哪些研究方向？
哪些 Claim 已确认、受限或反证？
证据达到哪一级？
哪些实验已经完成？
当前最大瓶颈是什么？
下一项最能改变结论的实验是什么？
为什么允许或禁止自动继续？
```

---

## 十九、现有工具使用

### 19.1 根目录迁移状态

`ai2050_research_os` 已从 `research/ai2050_research_os` 移到仓库根目录。当前目录移动本身不等于迁移完成：历史合同索引、Manifest 文件路径和 `researchctl.py` 的 `WORKSPACE` 层级计算仍可能保留旧目录假设。

在完成路径迁移器并重新冻结受影响的路径记录之前：

- 不得把因旧路径产生的“文件不存在”误判为科学 Artifact 丢失；
- 不得直接修改历史 Manifest 中的路径和哈希来消除错误；
- 必须区分“文件内容未改变但位置迁移”与“文件内容真的改变”；
- `researchctl validate` 的路径类错误必须先经过迁移审计，才能作为有效校验结论；
- 修复脚本时应使用可靠的仓库根发现方式，不能继续依赖固定 `parents[n]` 层数。

目录迁移应生成独立迁移报告，记录旧路径、新路径、内容哈希、迁移状态和是否需要重新冻结。历史内容哈希不变时应保留原证据身份，并通过显式路径迁移记录建立新位置映射。

### 19.2 目标命令

从仓库根目录运行：

```powershell
.venv\Scripts\python.exe ai2050_research_os\scripts\researchctl.py validate
.venv\Scripts\python.exe ai2050_research_os\scripts\researchctl.py build
.venv\Scripts\python.exe ai2050_research_os\scripts\researchctl.py summary
.venv\Scripts\python.exe ai2050_research_os\scripts\researchctl.py verify-manifest ai2050_research_os\manifests\EXP-C001-WP01-001.manifest.json
```

从 `ai2050_research_os` 目录内部运行：

```powershell
..\.venv\Scripts\python.exe scripts\researchctl.py validate
..\.venv\Scripts\python.exe scripts\researchctl.py build
..\.venv\Scripts\python.exe scripts\researchctl.py summary
```

现有命令含义：

- `validate`：检查编号、引用、依赖、状态和值域；
- `build`：根据 `registry/` 重建自动看板；
- `summary`：输出当前战役、阻塞项和下一决策；
- `freeze`：为合同计算冻结摘要，不执行实验；
- `verify-manifest`：校验合同和冻结产物。

后续应增加：

```text
researchctl migrate
researchctl validate-snapshot
researchctl build-snapshot
researchctl export-client
researchctl export-memo
researchctl drift-audit
```

---

## 二十、当前首要大任务

当前不应先开发远程 Worker，也不应先重做所有客户端页面。首要大任务是：

> 建立可校验、可确定性重建、无状态漂移的 JSON Canonical Snapshot，并让客户端彻底停止从 JavaScript 常量、历史页面或 Memo 获取当前研究状态。

建议交付物：

1. 核心 JSON Schema v1；
2. Contract 分类型规范；
3. Canonical Snapshot 构建器；
4. 漂移审计器；
5. 客户端 JSON 导出器；
6. `useResearchSnapshot()`；
7. 一个由真实 Snapshot 驱动的项目总览页面；
8. 一份迁移报告，明确哪些旧数据已规范化、哪些仍只能保留为历史证据。

完成这一阶段后，系统才真正从“很多实验、Phase 和可视化页面的集合”升级为“以证据和命题为中心的 AI 研发操作系统”。
---

## 客户端精简优化方案（研发框架优先）

### 1) 目标

- 左侧面板不再以 Phase / Campaign / Run 作为主导航。
- 理论分析与 AI 研发都以“研发框架（R/M/P/E）”为核心进度。
- 3D 空间新增“热力图”路线：左侧层显示 Embedding 热力图，中层显示 Hidden State 热力图。
- 3D 默认不按 phase 展示测试结果，而只展示测试中的状态轨迹与变化趋势。

### 2) 左侧面板改造（关键动作）

1. `研发框架`：当前框架节点、完成率、阻塞项。
2. `命题摘要`：当前 Claim、核心 Evidence、最近决策。
3. `执行入口`：下一个可执行 Experiment / Run / Decision。

移除字段（面板主信息）:
- 当前 Phase
- Phase ID
- Run ID（保留在详情页用于追溯）

### 3) 3D 空间：热力图路线

- 路线名称：`热力图`
- Layer 左侧：`词嵌入 heatmap`
  - 目标：观察词嵌入随测试步进在 token/维度上的热度变化。
- Layer 中间：`Hidden State heatmap`
  - 目标：观察层间状态变化、是否稳定、是否出现异常层位。

视图规则：
- 同一 run 内统一色条归一化；不同 run 不直接复用同一色条。
- 默认不显示 phase 编号；默认标题使用：`当前路线 / 当前层 / 当前状态`。
- 点击展开后可显示 provenance（run/event）用于审计。

### 4) 实施节奏

- 第1步：左侧面板清理（本周）
- 第2步：理论中心换位为框架进度（本周）
- 第3步：3D 热力图接入与联动（下周）
- 第4步：回归验收（下周）

### 5) 验收标准（Done）

- 左侧面板主信息不含 phase 或 phase 数量。
- 新增“热力图”路线且可在左侧/中层切换 embedding 与 hidden state。
- 运行一个测试流程时可看到：测试状态由 embedding 与 hidden state 变化趋势驱动，而非 phase 清单。
- 历史回溯仍可打开，不影响主界面体验。

该方案与附件方向一致：**框架优先、phase 退场为追溯坐标、可视化按状态轨迹优化复杂度**。

## 继续简化实施清单（客户端当前轮）

### 目标
- 左侧默认面板只保留：当前框架、瓶颈一句话、最近证据、下一动作。
- 不在左侧默认入口作为主导航显示 Phase 列表。
- 3D 只展示“Embedding 热力图 / HiddenState 热力图”与同源 run 追溯，不显示 phase 数字作为主标题。

### 落地内容（本轮已改）
- `frontend/src/researchCenter/ResearchCenter.jsx`：统一左侧三入口风格，按框架阶段显示，去掉阶段主导语义。
- `frontend/src/blueprint/ResearchProgressTab.jsx`：改为研发框架与瓶颈主线。
- `frontend/src/blueprint/ProjectRoadmapTab.jsx`：保留路线图结构，移除 phase 导航噪音。
- `frontend/src/blueprint/SystemStatusTab.jsx`：模块状态以状态与任务驱动展示。
- `frontend/src/components/app/ResearchSpaceOverlay.jsx`：3D 侧边标注改为证据链与结果标签。
- `frontend/src/components/app/ResearchHeatmapRoute.jsx`：热力图标题改为 Embedding + HiddenState，默认不强调 phase。
- `frontend/src/researchKernel/heatmapResearchRoute.js`：热力图边界定义改为“可观测性与归一化声明”。

### 交付验收
- 默认左侧主视图无“当前 Phase / Phase ID / run phase”的长期列表。
- 左侧面板在 3 秒内可读取并回答：当前框架、瓶颈、下一动作。
- 3D 热力图 route 只需点击两层可直接读出 embedding 和 hidden-state。
- 历史追溯信息（run/event）保留在详情页，不占用默认入口。

### 继续优化实施方案（v2，可直接验收）

- 原则：左侧始终只展示三段式状态：框架、瓶颈、下一动作；证据与 run 信息只作为追溯附加。
- 第一优先级（本轮）：把 ResearchProgressTab、ProjectRoadmapTab、SystemStatusTab 再压缩为 3 个面板（不折叠、不分页），每行不超过 2 条信息。
- 第二优先级：把 ResearchHeatmapRoute 维持两层可视化，不显示任何阶段/phase 标识；标题为 Embedding + HiddenState。
- 第三优先级：在证据抽屉中只保留 Closure Lx、证据等级、Run/Artifact，去掉 phase 冗余字段。
- 最小验收：
  1. 新用户打开左侧 10 秒内能读到「当前框架 + 当前瓶颈 + 下一动作」。
  2. 点击一个 evidence 详情，仅在侧滑面板出现 closure、un、rtifact，不出现 phase 导航入口。
  3. 3D 中只可见 Embedding 与 HiddenState 两层图例，且默认说明为 run 级别追溯。

说明：以上改动仅更新 i2050_research_os/README.md 与前端对应文件，未写入 memo 文件。

## 继续简化实施清单（客户端本轮V3）

- 本次再压缩：默认侧栏仅保留三件事，左侧不出现 phase/阶段主导航入口。
  1. `当前框架`：显示框架标题与当前步骤（R*）。
  2. `当前瓶颈`：显示 `snapshot.current.bottleneck` 一句话。
  3. `下一动作`：显示 `snapshot.current.next_decision` 一句话。
- 左侧证据为追溯信息，不再承担导航：只显示最近 4 条 evidence。
- 3D 路线统一使用 `Heatmap` 研究路线，默认展示两部分：
  - 左侧：`Embedding` Heatmap
  - 右侧：`HiddenState` Heatmap（按 layer 显示）
- `Phase` 仅保留历史记录与后台元数据，不进入默认界面主内容。

### 交付标准（本轮验收）

1. 打开默认侧栏 10 秒内可读到：框架、瓶颈、下一动作。
2. 在 Heatmap 视图中只展示 Embedding + HiddenState，不出现 phase 结果列表。
3. 3D 场景不以 phase 为主标题，只展示 run 追溯（run id / model / token）和两层热力图。

对应文件更新（本轮）：
- `frontend/src/researchCenter/ResearchCenter.jsx`
- `frontend/src/components/app/ResearchHeatmapRoute.jsx`
- `frontend/src/researchKernel/heatmapResearchRoute.js`
- `frontend/src/components/app/ResearchSpaceOverlay.jsx`



## 鏈?缁堢増绠?鍖栨柟妗堬紙瀹㈡埛绔紝V4锛?

鐩爣锛氬乏渚у叆鍙ｅ敖閲忓彧鍥炵瓟涓変欢浜嬶紱3D 璺敱鍙己璋冩祴璇曠粨鏋滃彲杩芥函銆?

### 1锛夊乏渚ч潰鏉匡紙蹇呴』婊¤冻锛?

- 鍙睍绀猴細
  1. 褰撳墠妗嗘灦锛坒ramework锛?
  2. 褰撳墠鐡堕锛坆ottleneck锛?
  3. 涓嬩竴鍔ㄤ綔锛坣ext_decision锛?
- 鏈?杩? evidence 鍙繚鐣? 4 鏉★紝鐢ㄤ簬杩芥函锛屼笉鎵胯浇涓诲鑸??
- 涓嶆樉绀? `Phase`銆乣Phase ID`銆乣run phase` 浣滀负涓绘爣棰樻垨鍒楄〃鍏ュ彛銆?

瀵瑰簲鏂囦欢锛?

- `frontend/src/researchCenter/ResearchCenter.jsx`
- `frontend/src/blueprint/ResearchProgressTab.jsx`
- `frontend/src/blueprint/ProjectRoadmapTab.jsx`
- `frontend/src/blueprint/SystemStatusTab.jsx`

楠屾敹锛氭柊鐢ㄦ埛 8 绉掑唴鍙鍒扳?滃綋鍓嶆鏋? / 褰撳墠鐡堕 / 涓嬩竴鍔ㄤ綔鈥濄??

### 2锛?3D 鐑姏鍥捐矾绾匡紙蹇呴』婊¤冻锛?

- 3D 璺緞鏍囬鍥哄畾涓? `Embedding + HiddenState Heatmap`锛屼笉鍑虹幇 phase 鍒楄〃銆?
- 榛樿灞曠ず锛?
  - 宸︿晶锛欵mbedding 鐑姏鍥撅紙top-k 缁村害锛?
  - 鍙充晶锛欻iddenState 鐑姏鍥撅紙鎸? layer锛?
- 璺緞璇存槑浣跨敤 run 涓婁笅鏂囷紙model / token锛?+ provenance锛坮un id / artifact lineage锛夈??
- 鐐瑰嚮璇佹嵁鍙敤浜庤拷婧紝涓嶄綔涓洪粯璁や富瀵艰埅銆?

瀵瑰簲鏂囦欢锛?

- `frontend/src/components/app/ResearchHeatmapRoute.jsx`
- `frontend/src/researchKernel/heatmapResearchRoute.js`
- `frontend/src/components/app/ResearchSpaceOverlay.jsx`

楠屾敹锛?

1. 杩涘叆 heatmap 璺嚎鏃讹紝鍙湅鍒? embedding 涓? hidden-state 涓ゅ潡鍥俱??
2. 涓嶅嚭鐜? phase 缁撴灉鍒楄〃鏍囬銆?
3. 椤甸潰涓嶄細鎶娾?減hase鈥濅綔涓哄綋鍓嶄换鍔″叆鍙ｆ彁绀鸿瘝銆?

### 3锛夊疄鏂芥楠わ紙鏈?灏忓仠鏈虹増锛?

1. 鍏堢‘璁ゅ乏渚ч粯璁ら潰鏉挎枃妗堜笌瀛楁锛涘浠嶅嚭鐜? phase锛岀珛鍗虫崲涓? framework 鏍囩銆?
2. 灏? trace 瑙ｆ瀽缁熶竴鍒? run-level锛氬繀椤绘湁 `embedding + hidden-state top-k` 鎵嶅睍绀虹儹鍥俱??
3. 淇濈暀鍘嗗彶 phase 鏁版嵁鍦ㄥ揩鐓т笌 provenance 涓紝涓嶇Щ闄わ紝鍙檷鏉冿紝涓嶄綔涓? UI 涓诲眰銆?
4. 褰㈡垚涓?娆? smoke check锛氭墦寮? `Theory` 涓? `AI R&D` 涓ょ鍏ュ彛锛岀‘璁ゅ潎鏃? phase 瀵艰埅鍏ュ彛骞惰兘鐪嬪埌 run 绾х儹鍥俱??

娉ㄦ剰锛氭湰杞彧鏇存柊 `ai2050_research_os/README.md` 涓庡墠绔浉鍏虫枃浠讹紝鏈洿鏂? memo銆?
## 客户端精简收敛方案（V5：框架入口与状态热力图）

本轮只更新客户端与本 README；不更新 `research/glm5/docs/AGI_GLM5_MEMO.md`。

### 1. 入口与信息架构

- 左侧“研究驾驶舱”保留“全部研发框架”下拉框，并把三个高频入口固定为按钮：`语言机制`、`状态热力图`、`AI研究`。
- `状态热力图`不再是需要在下拉框中寻找的隐藏选项。点击该按钮即选中 `heatmap-analysis` 路线，开启热力图层并切换到它的简化路线面板。
- 默认页面仍可停留在语言机制路线；热力图不是默认弹窗，避免首次进入时增加噪音。

### 2. 热力图的数据语义

- 数据源为 `real_component_trace.v1` 的同一次 Run 原始 trace，而不是 memo 的文字说明。
- 左侧显示 `Embedding` 的稀疏 top-k 词嵌入状态；右侧显示按 `Layer` 排列的 `HiddenState` 稀疏 top-k 状态。
- `HiddenState` 采用 `residual2`、`residual_output`、`residual1` 或 `residual_input` 中同层优先级最高的记录。
- 深色格代表该 top-k 维度在当前稀疏记录中未观测，不能解释为数值为零。缺少任一类原始记录时，只显示等待提示，不能从 memo 或 Phase 文本推断数值。

### 3. 验收标准

1. 打开客户端左侧“研究驾驶舱”，可直接看到“状态热力图”按钮，无需展开下拉框。
2. 点击按钮后，左侧卡片显示模型、Token、Run ID 与数据可用状态；不显示 Phase 列表。
3. 3D 空间显示两块图：左侧“词嵌入（Embedding）”，右侧“HiddenState（按 Layer）”。
4. 原始 trace 缺失时，页面仍可启动并显示明确等待信息；不把真实 trace 标为 memo，也不发起重复的 fallback 加载。

### 4. Layer 就近显示规则（V6）

- Embedding 热力图固定在 Layer 展开模型左侧，使用独立坐标，不能与 Layer 模型重叠。
- HiddenState 不再汇总为一张“全部 Layer”大矩阵；仅在前向执行到 `Lk` 时，显示该 `Lk` 的 top-k 热力图。
- 当前 Layer 的 HiddenState 置于 Layer 模型正面内部区域，随 `fpCurrentLayer` 更新；尚未开始执行时，仅保留“运行到某个 Layer 后显示”的提示。
- 当前 Layer 的维度直接从该 Layer 的原始 top-k 记录选择，避免将其他 Layer 的维度混入当前状态图。

### 5. 热力图显示范围（V7）

- 左侧状态热力图卡片提供“显示范围”选项：`Top-4`、`Top-8`、`Top-12`、`Top-16` 或“全部已记录维度”。
- 此选项同时作用于左侧 Embedding 和当前 Layer 的 HiddenState，切换后即时重绘 3D 图。
- “全部”严格表示当前 trace 中实际写入的全部 `top_units` 维度，不表示未经采样的完整 HiddenState 向量；不得补零或推断未记录维度。

### 6. 完整参数热力图（V8，替代 V7 的“全部已记录”定义）

- 本轮发现每个 Run 已保存 `full_vectors.pt`：Embedding 和每层 `residual2` 都具有完整维度。客户端将其转换为 `full_state_vectors.v1` JSON，供浏览器直接读取。
- 选择“全部参数”时，Embedding 显示完整词向量：qwen3 为 2,560 维、GLM4 为 4,096 维、DeepSeek 为 3,584 维；当前 Layer 的 HiddenState 同样显示其完整 residual2 向量。
- 3D 使用 64 列网格压缩完整向量，避免把数千维绘制为一条超长横线；每个单元仍对应一个真实维度，未补零、未插值。
- 完整向量文件尚在加载时，界面明确显示加载状态；只有读取完成后才可标注“全部参数”。

## 实时模型热力图（V9）[2026-08-25 20:26]

热力图路线改为直接驱动本地模型，不再要求用户先生成并手动选择静态 JSON。左侧仅保留模型、输入文本、显示范围和“加载模型并运行/停止”控件；旧的生成/演示模式与逐步骤播放控件在热力图路线中隐藏。

实时数据流如下：

1. 客户端向 `/api/research-trace/runs` 提交模型与输入文本。
2. 后端按单 GPU 串行规则从本地硬盘加载 Qwen3、GLM4 或 DeepSeek-7B。
3. 模型开始推理时发布最后一个输入 Token 的完整 Embedding；每完成一个 Transformer Layer，立即原子更新该层完整 HiddenState。
4. 客户端每 350 ms 读取 `/api/research-trace/runs/{run_id}/live-state`，并让 3D Layer 与真实 `current_layer` 同步。
5. 推理完成后仍保存可追溯的最终 Trace；客户端无实时 Run 时可继续读取原有冻结数据作为降级显示。

显示规则：

- `Top-K` 从完整向量中按绝对激活值选取 K 个真实坐标，不补零、不插值。
- “全部参数”显示 Embedding 与当前 Layer HiddenState 的全部物理激活坐标，64 列排列。
- 模型尚在加载时，3D 空间显示“等待 GPU / 正在加载模型 / 正在准备推理”等真实状态，不显示旧数据冒充实时结果。
- HiddenState 只显示已经实际执行到的当前 Layer；停止或失败状态由运行管理器覆盖到客户端。

## Qwen3.8-27B 量化测试准备（V11）[2026-08-26 02:21]

Qwen3.8-27B 暂不加入客户端默认模型列表，先通过独立工程门禁验证，避免通用加载器误用 BF16 并造成显存溢出。

- 模型固定为 `Qwen/Qwen3.8-27B` 提交 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`；18 个权重分片和配置文件已完整下载到 `models/hf/Qwen3.8-27B`，官方文件清单共 55,586,114,863 bytes。
- 权重保持官方 BF16 原件，运行时才使用 bitsandbytes NF4 双重量化。受 RTX 5080 16 GB 显存限制，语言层 0–47 在 CUDA 上执行 NF4，语言层 48–63 保持 BF16 并由 CPU 卸载钩子按需执行；Embedding、LM Head、最终 Norm 和视觉模块也放在 CPU。
- Windows 下必须使用测试入口内置的逐分片加载器：每次只映射、转换并释放一个 safetensors 分片，避免 Transformers 5.12 同时映射全部 55.6 GB 权重而触发页面文件错误；该修复不会改动系统页面文件或全局 Python 包。
- 架构门禁固定为 64 个语言层、Hidden Size 5,120、48 个 Linear Attention 层和 16 个 Full Attention 层。
- 首轮只允许 batch size 1、最多 64 个输入 Token、`use_cache=False` 的短输入工程冒烟；通过前不得启动大样本研究。
- `hidden` 冒烟保存完整的 `token × 5120` Embedding 和全部 64 层 `token × 5120` HiddenState，不做 Top-K 截断。量化结果是独立测试条件，不能与 FP16/BF16 结果直接合并。

统一入口：

```powershell
tests\glm5\run_qwen38_27b_nf4_smoke.ps1 preflight
tests\glm5\run_qwen38_27b_nf4_smoke.ps1 load
tests\glm5\run_qwen38_27b_nf4_smoke.ps1 hidden --prompt "The capital of France is"
```

`preflight`、`load` 与单 Token `hidden` 工程冒烟均已通过：确认 27,356,728,560 个逻辑参数、372 个 NF4 线性模块、前 48 层 CUDA 驻留和后 16 层 CPU 卸载；完整装载耗时约 33 秒，CUDA 峰值分配 9,582,030,336 bytes（约 8.92 GiB）。`hidden` 以 `Hello` 的 1 个真实 Token 完成前向，保存 1 份完整 `1 × 5120` Embedding 与 64 份完整 `1 × 5120` Layer HiddenState，所有值有限，耗时约 46 秒，CUDA 峰值约 11.17 GiB。该单 Token 结果只证明量化前向和完整向量采集链路可用，不产生语言机制证据；正式研究仍须按独立方案扩大数据量并复测后，才可给出结论。

本轮没有更新 `research/glm5/docs/AGI_GLM5_MEMO.md`。

## 研究数据工作台与 Loop Engineering 精简方案（V12）[2026-08-27 02:06]

### 1. 总体目标

客户端不再以“展示更多实验页面”为目标，而要成为长期积累语言机制证据的研究工作台。系统只保留一条主线：

```text
语言程序 → 编译样本 → 模型运行 → Embedding/HiddenState 场 → 配对响应 → 证据裁决 → 下一实验
```

界面只设三个一级视图：

1. `研究图谱`：查看语言程序族、操作、查询、角色、行为资格和证据缺口；
2. `3D 观测`：查看同一研究对象的 Embedding、逐层 HiddenState、配对差分和输出身份；
3. `Loop Engineering`：围绕一个证据缺口生成、审核、执行和裁决下一批实验。

Phase 只保留在历史来源中，不作为导航、标题、进度条或新增组件的依据。

### 2. 当前实现需要先解决的问题

- `frontend/src/App.jsx` 已超过 5,000 行，`ResearchHeatmapRoute.jsx` 已超过 3,100 行，组件承担了运行控制、数据适配、历史 Campaign 和 3D 渲染等多种职责。
- 热力图组件仍包含大量 C101–C433 专用构造器和 Phase 标题；每增加一次 Campaign 都继续扩大组件，无法形成通用研究工具。
- 实时 Trace 服务仍固定调用颜色实验脚本，默认 `target_label=red`，请求中没有 `program_id / world_id / operation_id / query_id / role_map / pair_id`，所以单次热力图不能积累为语言机制证据。
- 当前实时状态主要发布最后一个 Token 的单向量；无法完整观察多 Token、语义角色、上下文位置和输出时钟。
- “全部参数”依赖大 JSON 或大量 3D Box；代表全场文件已经达到数百 MB，继续按 Campaign 复制会使浏览器和存储不可持续。
- AI 研发界面仍以 analyze / plan / generate / execute / summarize Phase 条展示；代码生成循环没有与语言程序图、配对响应和 E0–E6 证据账本形成强约束。
- 部分 AI 研发源码存在乱码；模型序列缺少 Qwen3-14B，临时目录和结果目录也未完全遵守当前 `tests/glm5_temp/`、`tests/glm5/result/` 约束。

因此，第一原则是先收敛数据协议和通用渲染器，再增加新的研究内容。

### 3. 六个稳定研究对象

客户端与后端只围绕六类对象工作：

| 对象 | 最小职责 |
| --- | --- |
| `program` | 版本化语言程序；保存世界、实体、关系、角色、作用域、合法操作、查询和答案身份 |
| `case` | 程序的一次表面编译；保存文本、字符跨度、物理 Token、程序角色和行为正确答案 |
| `run` | 一个模型对一个 case 的不可变运行；保存模型版本、精度、随机种子、输出和张量工件哈希 |
| `pair` | 基线与单操作变体的登记配对；保存 Token/角色对齐与响应定义 |
| `evidence` | 对某个命题的 E0–E6 证据记录、边界、反例和来源 |
| `loop` | 为填补一个证据缺口而冻结的任务、预算、门控、运行队列和最终裁决 |

建议新增并版本化以下 Schema：

```text
language_program.v1
compiled_case.v1
trace_manifest.v2
paired_response.v1
evidence_record.v1
loop_contract.v1
```

每条 Trace 至少保存：

```text
program_id / program_version / atlas_snapshot_id
case_id / world_id / operation_id / query_id / answer_identity
language / surface_id / role_char_spans / physical_token_map
model_id / model_revision / precision / seed
run_id / pair_id / artifact_sha256 / capture_schema
behavior_status / evidence_level / claim_boundary
```

原始张量使用 safetensors 或分片二进制保存；JSON 只保存 Manifest、索引、形状、量纲、哈希和小型摘要。工件采用内容哈希去重，所有旧版本只追加、不覆盖。

### 4. 3D 观测空间

3D 空间保留 LLM Layer 主模型，并固定成一条从输入到输出的观察轴：

```text
语言程序/角色映射 → Embedding 热力图 → LLM Layer 主体 → 输出身份时间线
```

- `Embedding` 继续放在 LLM 模型相对主 Layer 的另一侧，大小接近 LLM 主体；但数据扩展为全部 Token × 全部真实维度，而不是只显示最后一个 Token。
- `HiddenState` 固定显示在当前执行 Layer 上；切换 Layer 或实时执行到新 Layer 时，只替换同一块热力图，不生成 64 张同时可见的大墙。
- 输出侧增加很小的 `答案身份/生成时钟` 视图，区分候选答案、首个分叉 Token 和后续生成历史。
- 默认观察单位不是单 Run，而是登记 Pair。顶部只保留四种显示模式：`基线 A`、`变体 B`、`差分 B−A`、`干预后`。
- 角色—Token 对齐以连线或边框表示；主动/被动、翻译和多 Token 词可比较程序角色是否保持，而不是强制按物理位置对齐。
- 颜色在同一 Pair 内使用冻结的对称尺度；不得跨 Run 自动比较颜色。Top-K 只是定位工具，“全部坐标”才是完整观测；两者都不能自动升级为机制结论。
- 每个格子必须能追溯到 `run_id / pair_id / layer / token / coordinate / raw_value / artifact_sha256`。

完整张量不能一次性转成 DOM 或数万个独立 Mesh。后端按 `layer × token_range × dimension_range` 返回压缩 Tile，前端使用 WebGL Texture/InstancedMesh 绘制，只加载相机当前可见区域。这样既能显示全部参数，又不把数百 MB JSON 整体送入浏览器。

语言程序图默认采用普通 2D/轻量 3D 图；只有当需要观察程序节点、Token 角色与 Layer 状态之间的路由关系时，才叠加到主空间，避免所有图谱同时出现。

### 5. 研究数据积累流程

每次 Campaign 必须按同一流水线积累数据：

1. 从版本化 atlas 选择语言程序族，例如类型图、事件角色、否定作用域、更新顺序或输出身份；
2. 冻结 world、operation、query、surface 和答案身份，生成发现、确认、lockbox 与独立复现分区；
3. 编译字符跨度到物理 Token，并检查题面唯一性、角色覆盖和答案合同；
4. 先运行行为资格；行为不合格的切片允许保存观察数据，但不得形成正式内部机制裁决；
5. 对合格切片采集完整 Embedding、全部 Token 的逐层 HiddenState 和输出时间线；
6. 按登记 `pair_id` 计算

   $$
   R_{o,q}(x)=\Pi_o H_q(o(x))-H_q(x)
   $$

   其中 $\Pi_o$ 来自角色—Token 对齐，不能用长度补齐或坐标猜测代替；
7. 保存预测、负控、干预、副作用和独立复现结果；
8. 仅由验证器把证据登记为 E0–E6，AI 文本总结不能直接修改证据等级。

研究图谱中的每一条边只显示当前最高证据级别和失败边界：E0 外部程序有效、E1 行为合格、E2 内部响应可预测、E3 局部传动稳定、E4 未见组合成立、E5 输出充分/必要/特异救援、E6 跨语言或跨模型功能对应。

### 6. Loop Engineering

Loop Engineering 不再用大量 Phase 表示进度，只使用八个可审计状态：

```text
证据缺口 → 合同草案 → 等待审核 → 可运行 → 执行中 → 工件审计 → 裁决 → 下一缺口/停止
```

界面只显示四项：当前要裁决的问题、尚缺的证据、下一动作、GPU/数据预算。高级配置、AI 提示词、代码和完整日志放入二级抽屉。

一次 Loop 的固定职责：

1. 从 Evidence Ledger 选择一个具体缺口，不允许把“继续研究语言”作为目标；
2. 同时生成至少两个竞争解释、各自预测和死亡观察；
3. 先生成 `loop_contract.v1`，通过方法与对照审核后才允许生成代码；
4. 代码只能写入 `tests/glm5/`，临时文件只能写入 `tests/glm5_temp/`，结果只能写入 `tests/glm5/result/`；
5. GPU 队列严格串行：Qwen3-4B → Qwen3-14B → GLM4 → DeepSeek-7B，每个模型完成、保存和释放后才能加载下一个；不要求所有模型共享同一提示接口，先在各自行为合格接口内建立功能对象；
6. 先 Smoke，再扩大正式样本，再做独立复验；任何重要正结果必须自动产生更大规模复测任务；
7. 工件验证器检查 Schema、数量、哈希、模型快照、分区泄漏、负控和副作用；缺项只能裁决为 `inconclusive`；
8. AI 可以提出解释与下一任务，但只有冻结规则和验证数据可以提升 Evidence；理论文本更新必须经过人工确认。

### 7. 后端接口

建议将当前单 Run Trace 接口扩展为资源式接口：

```text
GET/POST /api/research/programs
POST     /api/research/programs/{id}/compile
GET/POST /api/research/campaigns
GET/POST /api/research/runs
GET      /api/research/traces/{run_id}/manifest
GET      /api/research/traces/{run_id}/tiles
GET      /api/research/pairs/{pair_id}
GET      /api/research/evidence
GET/POST /api/research/loops
POST     /api/research/loops/{id}/approve|start|pause|stop
```

单 GPU Scheduler 是所有 Trace 和 Loop 的共同入口，负责排队、显存预检、模型卸载、失败恢复和断点续跑；实时客户端只订阅事件，不自己推断 Layer 已执行。

### 8. 客户端代码收敛

- `App.jsx` 只保留路由、全局选择和布局；研究逻辑移入独立 workspace store。
- 把 `ResearchHeatmapRoute.jsx` 拆成通用 `TraceAdapter`、`HeatmapTexture3D`、`LayerStateView`、`PairDiffView` 和 `EvidenceInspector`。
- 删除 C101–C433 专用 JSX 分支及默认 Campaign 回退；历史工件统一通过 `view_spec.json` 适配到有限渲染器。
- AI 研发 Overlay 删除 Phase 进度条和 Round 中心叙事，替换为 Evidence Gap、Contract Gate、Run Queue、Artifact Audit、Decision 五个小组件。
- 先统一修复 UTF-8 乱码；乱码文件未清理前不得继续复制组件。
- 左侧默认只保留 `程序族/研究对象`、`模型/Pair`、`运行`三组控件；详细来源和高级参数全部折叠。

### 9. 分阶段实施

1. `P0 数据地基`：冻结六类 Schema、内容寻址目录、Pair 对齐规则、Tile API 和 E0–E6 账本；暂不改 3D 外观。
2. `P1 客户端减法`：修复乱码；拆分两个超大组件；删除 Campaign 专用默认分支和 Phase UI；建立三个一级视图。
3. `P2 真实研究闭环`：选择 program → 编译 case → 行为资格 → Pair Trace → 3D A/B/差分 → evidence 登记。
4. `P3 Loop Engineering`：让 AI 围绕 Evidence Gap 生成合同和受约束脚本，接入统一 GPU Scheduler 与工件审计。
5. `P4 扩展`：加入输出身份、干预路径、跨语言和跨模型功能对齐；只有被多个研究复用的新结构才增加渲染器。

### 10. 验收标准

- 默认界面只有三个一级入口，左侧不出现 Phase 列表；首次进入无需理解历史 Campaign。
- 任一热力格可追溯到原始工件与物理坐标；浏览器不再整体加载数百 MB JSON。
- 可以从一个语言程序生成并比较 A/B Pair，完整显示全部 Token 的 Embedding 与任一 Layer 的全部 HiddenState。
- 单次 Run、配对响应、证据等级和理论命题严格分账；颜色相似、Top-K 或 AI 共识不能提升结论。
- Loop 缺少行为资格、负控、独立分区或必要工件时不能进入 accepted。
- 所有本地模型严格单 GPU 串行，模型退出后显存释放可审计。
- 新语言族只增加 program/case/evidence 数据，不再新增 Phase 专用 Dashboard。

该方案的核心不是让 3D 场景更复杂，而是让每次观察都能成为可复用、可比较、可追溯、可证伪的一条研究数据。

## 可视化客户端完整改造蓝图（V13：研究积累、3D 观察、Loop Engineering）[2026-08-27 07:26]

### 1. 产品结构

V13 保留 V12 的 `program / case / run / pair / evidence / loop` 六类稳定对象，并把客户端固定为三个一级部分：

```text
研究积累中心 | 3D 机制观察台 | Loop Engineering
```

三部分不建立各自独立的数据副本：研究积累中心负责登记和裁决，3D 观察台读取同一批原始工件，Loop Engineering 生产新的受审计数据并更新同一证据账本。

全局顶部只保留当前研究对象、语言程序族、模型、数据版本和证据状态。左侧只显示当前视图必需的选择与操作；Phase、历史 Campaign 和原始日志都进入二级来源抽屉。

---

## 第一部分：研究积累中心

研究积累中心包含三个固定子页：`语言模式族`、`HiddenState 场`、`理论进展`。

### 1.1 语言模式族

语言研究对象必须区分四种身份：

| 身份 | 例子 | 作用 |
| --- | --- | --- |
| 语言单位 | 苹果、吃、非常、在、。 | 研究者定义的词汇、功能和标点对象 |
| 物理 Token | tokenizer ID、subtoken、位置 | 模型实际接收的离散输入 |
| 上下文实例 | “苹果很好吃”中的苹果 | 绑定具体文本、世界和上下文 |
| 程序角色 | 实体、受事、类型节点、查询边界 | 语言程序中的功能身份 |

客户端不能把这四种身份合并成“一个 token 一个定义”。同一语言单位可以映射多个物理 Token，同一物理 Token 在不同上下文中也可以承担不同程序角色。

#### 模式族目录

第一版目录按以下结构积累：

1. `词汇与符号`
   - 名词：实体、类别、抽象对象、专名；
   - 动词：动作、状态、变化、关系、态度；
   - 形容词与副词：属性、程度、方式、时间；
   - 介词、连词、助词、冠词、代词、数词；
   - 否定词、条件词、量词、疑问词；
   - 标点、分隔符、格式标记和输出控制符；
   - 临时词、伪词、反事实名称和跨语言等义词。
2. `构式与用法`
   - 主谓、动宾、双宾、主被动；
   - 系表、比较、数量、否定、条件、因果；
   - 事件角色、态度嵌套、作用域和指代；
   - 类型图、部分—整体、空间、时间与更新顺序；
   - 释义、翻译、同义改写和输出格式；
   - 多句话语写入、覆盖、修正和查询。
3. `可执行语言程序`
   - world、实体和关系；
   - operation、合法输入/输出类型和作用域；
   - query、答案身份和输出合同；
   - 基线、单操作变体、错误操作和反事实控制。

#### 每条记录保存什么

`language_unit` 最少保存：

```text
unit_id / language / lemma / surface_forms / category
senses / allowed_roles / construction_ids / example_case_ids
ambiguity_notes / human_review_status / version / provenance
```

`construction` 最少保存：

```text
construction_id / family_id / typed_slots / constraints
operations / queries / answer_contract / surfaces
positive_cases / negative_controls / counterexamples
human_review_status / version / provenance
```

#### 界面

- 左侧：模式族树，只显示词汇、构式和语言程序三层；
- 中间：覆盖矩阵，横轴是模型或语言，纵轴是研究对象，格子显示 `未采集 / 已采集 / 行为合格 / 有预测 / 有因果 / 已闭合`；
- 右侧：对象详情，包括定义、上下文实例、角色—Token 对齐、已有 Pair、证据和下一缺口；
- 搜索支持词面、lemma、物理 Token ID、构式、角色、操作、查询和 claim ID。

### 1.2 HiddenState 场积累

对每个行为合格的 case 保存完整观测：

$$
\mathcal F(x)=\left\{E_{t,d}(x),H_{q,t,d}(x)\right\}_{q,t,d}
$$

其中 $t$ 是全部输入 Token，$q$ 是 embedding、每个 Layer、final norm 和登记的输出时钟，$d$ 是全部真实物理坐标。

需要积累四类数据：

1. `原始场`：全部 Token 的 Embedding、逐层 HiddenState、final norm 和输出身份时间线；
2. `配对响应`：同一程序只改变一个登记操作后的 A/B 及角色对齐差分；
3. `重复与控制`：重复前向、错误操作、错误角色、错层、等长度和无操作控制；
4. `干预结果`：写入、删除、救援、剂量、输出变化和副作用。

词嵌入分析必须区分：

- tokenizer 的静态 embedding 权重行；
- 具体输入位置取出的 embedding；
- 位置、格式或模型预处理加入后的实际初始状态；
- 同一语言单位在不同 subtoken 切分和不同模型中的物理实现。

HiddenState 场目录以 `研究对象 × 模型快照 × 语言 × surface × role × checkpoint × evidence` 建索引。默认不生成“平均词义向量”；任何派生摘要都必须保存算法、参数、输入工件哈希和可回到原始全场的索引。

#### 存储

- 元数据、索引、关系和状态写入本地 SQLite；
- 大型张量写入内容寻址的 safetensors/二进制分片；
- 运行表只保存形状、dtype、capture point、文件哈希和路径，不把向量正文写入数据库；
- 旧工件只追加版本，不覆盖；Pair 和 Evidence 永远引用不可变 Run；
- 浏览器只通过 Tile API 加载当前 Layer、Token 范围和坐标范围。

建议的核心表：

```text
language_units / constructions / program_families / programs / cases
model_snapshots / runs / tensor_artifacts / pairs / interventions
evidence_records / claims / puzzles / closure_gates
loop_runs / agent_steps / prompt_profiles / artifact_audits
```

### 1.3 理论进展

理论中心不显示长篇连续 Memo 作为主界面，而把理论拆成可审计对象：

```text
claim_id / title / statement / formula / scope
evidence_for / evidence_against / counterexamples
status / evidence_level / closure_gates
open_variables / next_experiment / version / provenance
```

界面固定为四块：

1. `已确认拼图`：当前有资格保留的最窄结论；
2. `失败与反例`：哪些解释被淘汰、在哪些材料或模型上失败；
3. `最新理论`：理论名称、核心对象、公式、适用范围和最近一次变更；
4. `数学闭合`：每条理论还缺哪些门，不用笼统百分比代替。

数学闭合至少检查：

```text
对象定义闭合
语言程序与角色编译闭合
行为资格闭合
全场观测与重复性闭合
未见样本预测闭合
局部传动与负控闭合
未见组合闭合
语义到输出身份闭合
必要性、充分性和特异救援闭合
跨语言/跨模型功能边界闭合
```

闭合状态只有 `未测试 / 失败 / 窄范围通过 / 独立复现通过`，不能用 AI 置信度或热力图观感填写。例如“q0 目标语言指令 embedding 门控”应记录为一个窄拼图，并明确“不等于概念翻译检索机制已解释”。

理论图只连接有 Evidence 引用的边；观察性关联用虚线，干预支持用实线，失败或反例用红色截断。点击任何理论边必须能打开原始 Pair、控制、工件哈希和裁决规则。

---

## 第二部分：3D 机制观察台

### 2.1 三种观察模式

3D 工作区只保留三个模式切换：

1. `静态结构`
   - 从真实 model config 和模块枚举生成 Embedding、Layer、Attention、MLP、Norm、Residual、LM Head；
   - 显示层数、hidden size、head 数、精度和设备位置；
   - 不存在或未采集的内部模块不绘制，不用演示动画冒充真实结构。
2. `动态运行`
   - 由后端真实事件驱动 Tokenization、Embedding、当前 Layer、已完成 Layer 和输出 Token；
   - 当前执行 Layer 高亮，HiddenState 热力图只在数据写入后更新；
   - 如果采集了 Attention/MLP 子阶段才显示子阶段，否则只显示 block 输入/输出。
3. `研究观察`
   - 选择一个语言单位、构式、program、case、Pair 或 claim；
   - 查看研究覆盖、A/B/差分、重复、控制和干预结果；
   - 理论连线只有达到相应 Evidence 等级后才显示，不自动从颜色相似生成。

### 2.2 固定空间布局

```text
[语言程序与角色]  [Embedding 场]  [LLM Layer 主体]  [输出身份/生成时钟]
```

- Embedding 热力图位于 LLM 主模型相对展开 Layer 的另一侧，尺寸与 LLM 主体接近；
- 展开的当前 Layer 上显示 HiddenState 热力图，Layer 前进时复用同一个显示平面；
- 输入 Token 沿一个明确方向排列；纵轴是 Token/角色，横轴是物理坐标；
- 输出侧显示候选答案、首次分叉 Token、完整生成和答案身份，不与上游语义状态混为一张图；
- 角色连线连接程序槽位、物理 Token 和当前 Layer 中的对应行，用于检查对齐，不解释为因果路径。

### 2.3 研究操作

研究者在 3D 中只需要以下操作：

```text
选择研究对象
选择模型和模型版本
选择基线 A / 变体 B / B-A / 干预后
选择 Token、角色和 Layer
播放或拖动真实执行时间线
固定颜色尺度
加入对比篮
打开证据与原始数值
```

精确值、长表和来源在 2D Inspector 中读取；3D 只负责空间关系、层间演化、角色路由和异常定位。任何格子 Hover 均返回 raw value、Layer、Token、坐标、Pair、Run 和 artifact hash。

### 2.4 性能边界

- 全参数热力图使用 WebGL Texture 或 InstancedMesh，不为每个坐标创建 React 节点；
- 后端按 Tile 和可见范围传输，支持取消旧请求和 LRU 缓存；
- Top-K 是快速定位模式，全部坐标是正式观测模式；
- 同一 Pair 使用冻结对称颜色尺度，跨模型只比较功能索引和相对层深，不直接比较物理坐标颜色；
- 大型 3D 场景只允许同时固定少量 Pair，其余进入对比篮或2D矩阵。

### 2.5 研究进展叠加

选择名词、动词、标点或构式时，3D 不显示虚构的“已知机制”，而显示其真实研究状态：

```text
灰：没有数据
蓝：已有原始场
青：行为合格
黄：有未见预测
橙：有局部干预
绿：独立复现并达到登记闭合门
红框：存在正式反例或冲突
```

颜色只表示证据状态，不表示激活正负；激活热力图与研究进展必须使用不同图例。

---

## 第三部分：Loop Engineering

### 3.1 两种运行模式

1. `自动运行`
   - 从一个登记 Evidence Gap 开始；
   - 在预算、停止规则和权限范围内连续完成合同、代码、运行、审计和下一任务；
   - 遇到合同冻结、重要理论升级、权限扩大、连续失败或预算上限时必须暂停等待人工确认。
2. `手动执行`
   - 使用完全相同的状态机和数据协议；
   - 每完成一个 Gate 后停下，由用户查看方案、代码、数据或裁决并点击继续；
   - 允许退回合同草案，但已经运行的工件不可改写，只能创建新版本。

自动与手动只区别“是否自动跨过普通 Gate”，不能使用两套研究逻辑。

### 3.2 多 AI 模型机制

必须区分两类模型：

- `研发代理模型`：负责编程、审查、分析和规划；
- `被研究的本地模型`：Qwen3-4B、Qwen3-14B、GLM4、DeepSeek-7B，只负责接受实验，不参与裁决自身结果。

研发代理采用一个主模型和多个辅助模型：

```text
主模型：研究负责人 + 合同整合 + 编程 + 下一任务草案
辅助模型A：方法、样本、对照和泄漏审查
辅助模型B：原始结果复算与数据审计
辅助模型C：反例、副作用和替代解释
辅助模型D：理论边界与数学闭合检查
```

辅助模型必须独立读取冻结合同和原始工件摘要，不能先看到主模型结论后简单投票。主模型可以综合意见，但最终 Evidence 状态由确定性验证器和冻结规则写入。

### 3.3 模型和提示词配置

每个研发代理可配置：

```text
provider / api_base / model_id / enabled / role
timeout / token_budget / temperature / retry_policy
system_prompt / task_prompt / output_schema
prompt_version / prompt_sha256
```

API Key 只保存在后端安全配置，不写入浏览器 localStorage、运行结果或普通数据库导出。提示词按角色分为分析、方法审查、计划、代码生成、数据审计、对抗审查、理论综合和总结；每次运行记录实际使用的提示词版本与哈希，保证结果可追溯。

### 3.4 一轮完整循环

```text
1. 选择一个 Evidence Gap
2. 冻结 Evidence Snapshot
3. 主模型提出竞争假设、预测和死亡观察
4. 辅助模型独立审查
5. 主模型生成 loop_contract
6. 方法 Gate 与人工/自动权限检查
7. 主模型生成受约束测试脚本
8. AST、目录、Schema、预算和安全检查
9. Qwen3-4B Smoke
10. 扩大样本正式运行
11. Qwen3-14B → GLM4 → DeepSeek-7B 串行运行
12. 原始工件、哈希、模型快照和GPU释放审计
13. 辅助模型独立分析结果、反例和副作用
14. 主模型形成 accepted/rejected/inconclusive 草案
15. 确定性裁决器更新 Evidence Ledger
16. 生成理论更新草案或下一 Evidence Gap
```

本地被研究模型严格一次只加载一个。每个模型结束后必须记录显存释放、进程退出、工件完整性和模型快照，验证通过后才能进入下一个模型。某模型行为不合格时，该模型内部机制记为 `NA`，不应停止其他模型的独立资格测试，也不能写成跨模型机制失败。

### 3.5 保存结果和更新数据库

每个 Loop 自动建立不可变目录，并写入：

```text
loop_contract.json
evidence_snapshot.json
agent_steps.jsonl
prompt_manifest.json
generated_script.py
model_snapshot.json
cases.jsonl
trace_manifest.json
paired_responses.jsonl
interventions.jsonl
artifact_manifest.json
artifact_audit.json
decision.json
theory_update_draft.json
```

脚本写入 `tests/glm5/`，临时文件写入 `tests/glm5_temp/`，结果写入 `tests/glm5/result/`。数据库更新使用一个事务：先验证文件存在与哈希，再登记 Run/Pair/Evidence，最后推进 Loop 状态；中途失败时回滚元数据，但保留失败工件用于审计。

理论数据库不能由主模型直接更新。只有 `decision.json` 通过 Schema、控制、分区、工件和门槛验证后，系统才创建待审核 Theory Revision；重要结论仍需人工确认。

### 3.6 Loop 界面

一级界面只显示：

```text
运行模式：自动 / 手动
当前 Evidence Gap
主模型与辅助模型状态
被研究模型 GPU 队列
当前 Gate 与下一动作
预算、停止条件和错误
最新工件审计与裁决
```

二级页面为 `任务合同 / 模型与提示词 / 运行日志 / 工件 / 审查意见 / 历史版本`。不再显示 Phase 流程条和 Round 数量作为研究进展；循环进展由当前缺口是否取得新证据决定。

---

## 4. 共用后端与数据库

建议新增服务边界：

```text
CatalogService        语言单位、构式、程序和版本
CompileService        文本、字符跨度、Token和角色编译
TraceService          模型运行与完整状态采集
TensorTileService     大型张量分片读取
EvidenceService       Pair、证据、Claim、Puzzle和闭合门
ModelScheduler        单GPU串行队列、卸载和恢复
LoopEngine            自动/手动研发状态机
PromptRegistry        代理配置和提示词版本
ArtifactValidator     Schema、哈希、分区、控制和完整性
```

SQLite 保存关系和状态，张量工件保存在文件系统；所有写入同时生成 append-only 事件。客户端从 API 读取投影，不直接读取或修改数据库文件。

旧 Phase/Campaign 数据通过一次性 Adapter 导入为 legacy Run、Artifact 和 Evidence，并保留原始 Phase 作为 provenance。不得把旧数据自动提升到新证据等级，也不得继续在 React 中加入新的 Cxxx 专用分支。

## 5. 客户端代码结构

```text
frontend/src/researchWorkspace/
  ResearchWorkspace.jsx
  atlas/
  field/
  theory/
  three/
  loop/
  inspectors/
  store/
  adapters/
```

- `App.jsx` 只保留应用路由、全局模型和研究对象选择；
- `ResearchHeatmapRoute.jsx` 拆为数据适配、Tile 加载、通用热力图和证据 Inspector；
- `AIRnDOverlay` 替换为 Loop Workspace，不再读取 `RESEARCH_PHASES`；
- 通用渲染器由 `view_spec` 驱动，Campaign 只增加数据；
- 全部源码统一 UTF-8 后再迁移，避免把现有乱码复制到新模块。

## 6. 实施顺序

1. `P0 数据合同`：SQLite迁移、六类核心对象、语言单位/构式目录、Tensor Artifact、证据和理论Schema；
2. `P1 研究积累中心`：语言模式族、HiddenState覆盖矩阵、理论与数学闭合界面；
3. `P2 3D底座`：静态结构、真实动态事件、全Token Tile热力图和A/B/差分；
4. `P3 Loop底座`：自动/手动状态机、主辅模型配置、提示词版本、统一GPU Scheduler；
5. `P4 真闭环`：Loop结果事务写入数据库，并在研究中心和3D中自动出现；
6. `P5 旧数据迁移`：导入历史Campaign、移除专用渲染器和Phase入口；
7. `P6 扩展研究`：输出身份、干预、跨语言和跨模型功能对齐。

在 P0 完成前不继续增加新的专用 Dashboard；在 P1 数据可追溯通过前不重做复杂 3D 特效；在 P3 工件审计通过前不开放无人值守的无限自动循环。

## 7. 最终验收

- 可以登记一个名词、动词、标点或构式，并看到其 case、模型、HiddenState 和证据覆盖；
- 可以选择任意上下文 Token 实例，查看完整 Embedding、全部 Layer 的完整 HiddenState 和角色对齐；
- 可以选择 Pair 在 3D 中切换 A、B、B−A 和干预后，并追溯每个物理坐标；
- 理论中心能明确回答：完成了什么关联、结果是什么、有哪些关键拼图、最新理论是什么、哪些数学闭合门尚未通过；
- 3D 静态结构来自真实模型配置，动态运行来自真实后端事件；
- 自动和手动 Loop 使用同一状态机；主模型、多个辅助模型和提示词均可版本化配置；
- Loop结果、失败、代码、提示词、工件和裁决完整保存，验证后自动更新数据库；
- GPU模型严格串行，崩溃后可以恢复，Evidence不能被AI意见或视觉效果越级提升；
- 默认界面没有Phase列表，不再为单次Campaign增加专用页面。

V13 的最终产品定义是：客户端不是实验结果播放器，而是把语言模式、内部状态、理论拼图和自动研发连接成同一条可持续证据生产线。

---

## V13 第一批实施记录：研究积累与 Loop 工作区 [2026-08-27 10:01]

本批已完成 P0/P1/P3 的可运行骨架，范围刻意限定在非 3D 部分。现有 3D 空间中的 LLM 模型、Layer 布局、材质、相机和热力图渲染均未修改。

### 已落地内容

1. 新增研究积累 SQLite 服务 `server/research_workspace_service.py`：
   - `language_objects` 保存 token、词类、标点与构式；
   - `field_records` 保存 Embedding、HiddenState 完整参数的维度、模型、Case、Run 和不可变产物路径；
   - `theory_claims` 保存理论主张、证据等级、支持/冲突计数、关键拼图和下一验证；
   - `closure_gates` 保存语言覆盖、场覆盖、因果干预、跨模型复现和数学闭合状态；
   - `workspace_events` 以 append-only 方式记录数据库写入事件。
   - `loop_runs` 事务保存每次自动/手动 Loop 的目标、代理、裁决、摘要与工件审计。
2. 增加 `/api/research-workspace` API：
   - `GET /health`、`GET /snapshot`；
   - `POST /language-objects`、`POST /field-records`、`POST /claims`；
   - `PATCH /claims/{id}`、`PATCH /closure-gates/{id}`。
3. 理论研究中心精简为三个稳定入口：
   - `语言模式族`：查询、模式族汇总、样本覆盖与新增研究对象；
   - `HiddenState 场`：登记完整 Embedding 与逐层 HiddenState 产物，不把 Top-K 当成研究原始数据；
   - `理论进展`：集中展示理论主张、关键拼图、下一验证与数学闭合门。
4. AI 自动研发入口替换为简化的 Loop Engineering 工作区：
   - 运行模式只有自动与手动两种；
   - 主模型负责规划、编程与裁决，辅助模型可动态增删并独立配置；
   - 主模型的分析、规划、编程、裁决提示词和辅助模型分析提示词均可配置保存；
   - 用户界面只显示 `证据缺口 → 实验契约 → 串行执行 → 独立复核 → 证据回写` 五个研究门，不展示底层执行编号列表；
   - 证据页面展示单 GPU 串行模型顺序、角色边界、Evidence Kernel 与最近工件审计结果。
   - 每次 Loop 完成后自动把结果和 `artifact_audit` 回写研究数据库；回写失败会产生显式事件，同时保留已经生成的不可变运行工件。
5. 自动研发运行目录已统一为：
   - 正式结果：`tests/glm5/result/auto_rnd/`；
   - 临时脚本：`tests/glm5_temp/auto_rnd/`；
   - 旧 `tests/result/` Evidence Kernel 仅作为只读兼容回退，不再作为新写入位置。

### 数据边界

SQLite 只保存可搜索的关系、维度、状态和工件引用；完整大张量继续保存在 `tests/glm5/result/` 下的不可变文件中。数据库中的种子模式族和初始理论均明确标为 `E0`，不会因为写入目录或 AI 共识自动升级证据等级。运行数据库文件为本地状态，不进入 Git。

### 验证结果

- Python 编译检查通过：研究数据库、AI R&D 服务、执行器与服务器入口均可导入；
- `tests/glm5/test_research_workspace_service.py` 的 5 个回归测试通过，覆盖种子状态、完整场登记、理论/闭合门持久化、Loop 结果事务回写和 HTTP API 往返；
- 前端生产构建通过，共转换 2827 个模块；
- 新增研究中心文件的定向 ESLint 通过；
- 本地后端 HTTP 冒烟通过：研究工作区、AI R&D 会话与编排状态接口均正常返回；
- 当前会话没有可用的浏览器控制实例，因此尚未完成截图级视觉验收。

### 下一批边界

下一批应先完成真实测试产物到 `field_records` 的自动导入器，以及 Loop 裁决到 Claim/Closure Gate 的事务回写。3D 改造继续保持冻结，直到完整张量索引、Case/Run 来源和 Tile 读取合同稳定后，再单独实施 3D 观察层。

---

## AI 自动研发独立页面与纵向切换 [2026-08-29 06:08]

AI 自动研发不再作为 3D 空间右侧的浮层或抽屉。客户端改为一个纵向工作区：第一屏是原有 3D 研究空间，第二屏是独立的 Loop Engineering 页面。

### 交互方式

- 鼠标滚轮、触控板或触屏向下滚动：从 3D 空间进入 AI 自动研发；
- 3D 左上角 Bot 按钮：平滑滚动到 AI 自动研发页面；
- 3D 底部中央 `AI 自动研发` 提示：平滑滚动到第二屏；
- AI 自动研发页右上角 `返回 3D`：平滑回到第一屏；
- 浏览器原生滚动仍然有效，不需要先打开侧栏，也不会卸载 AI 研发状态。

### 页面结构

```text
app-scroll-shell（100vh 纵向滚动容器）
├── 3D 研究空间（第一屏，100vh，保持原有渲染）
└── AI 自动研发（第二屏，独立页面，可按内容继续向下滚动）
```

滚动容器使用轻量 `scroll-snap` 辅助对齐，但保留正常的长页面滚动。AI 页面改用 `mode="page"`，取消 `position: fixed`、侧栏宽度和浮层阴影；页面头部保持粘性，运行、代理与提示词、证据三个页签继续使用同一 Loop 状态。

进入 AI 页面前会关闭仍在 3D 屏幕上的理论中心、界面配置和帮助窗口，避免固定浮层遮挡第二屏。当前屏幕状态由滚动位置更新，因此顶部 Bot 按钮的激活颜色与实际可见页面一致。

### 修改边界与验证

- 只修改顶层页面容器、滚动导航和 AI 工作区页面模式；
- 未修改 3D LLM 模型、Layer 结构、Canvas 内容、热力图或相机；
- 未修改研究数据库和 Loop 后端协议；
- Vite 生产构建通过，共转换 2825 个模块；
- AI 工作区相关 JSX 定向 ESLint 通过；
- `App.jsx` 全文件 ESLint 仍报告既有未使用变量等历史问题，本次新增代码没有产生构建错误；
- 当前会话没有可用浏览器控制实例，未完成截图级滚动验收。

---

## 项目研发 Agent：有界自动推进当前研究项目 [2026-08-29 11:18]

AI 自动研发独立页面新增 `项目 Agent` 一级入口，并保留原有 `单目标 Loop`、`代理与提示词`、`证据`三个入口。项目 Agent 不是一个可以无限运行、任意改写项目的通用代码机器人；它是建立在现有证据门、工件审计、独立复核和单 GPU 串行约束之上的有界研究代理。

### 自动工作方式

项目 Agent 启动前会读取当前研究数据库和 Evidence Kernel，按以下顺序生成最多 1–12 个研发任务：

1. 用户明确填写的项目目标；
2. Evidence Kernel 中尚未解决的证据缺口；
3. 受挑战、假设级或仍开放的理论主张；
4. 阻塞中、进行中或尚未通过的数学闭合门；
5. 语言对象缺少完整 Embedding / Layer × Token × HiddenSize 场数据时的覆盖任务；
6. 没有显式缺口时，对最近有效结果做一次独立复核并定位下一项可证伪问题。

任务队列生成采用确定性规则，不依赖 AI 自由宣布优先级。每个任务都保存来源类型、来源 ID、目标、完成条件、状态、裁决和对应 Run ID。当前任务完成并回写研究数据库后，Agent 自动把下一任务设置为 Loop 目标。

### 运行与安全控制

- 启动条件：主研发模型必须配置 API Key，并且至少有一个配置完成的辅助模型负责独立复核；
- 运行模式：支持自动连续执行，也支持在每个证据门等待人工确认；
- 预算边界：最多运行 1–12 个 Loop，不允许无人值守无限循环；
- 停止条件：可配置在获得 `accepted`、出现 `rejected`、连续多次 `inconclusive` 或任务预算耗尽时停止；
- 人工控制：运行中可暂停、继续、单步确认和立即停止；停止时同步终止活动执行进程；
- 数据边界：每轮仍使用不可变 Run 目录、完整工件审计和数据库事务回写，项目 Agent 不绕过现有执行沙箱与 GPU 串行锁；
- 理论边界：`plan_completed` 只代表有界任务队列已经执行完毕，不能自动把理论、关键拼图或数学闭合门标为成立。

### 页面显示

`项目 Agent` 页面集中显示：

- 模型是否就绪、当前 Agent 状态和运行方式；
- 项目目标、最大 Loop 数、连续未决阈值及提前停止选项；
- 从真实研究数据库生成的任务列表与来源统计；
- 当前任务、完成进度、每个任务的裁决和 Run ID；
- 最终停止原因，以及“计划完成不等于理论成立”的固定安全提示。

### 后端协议

新增接口：

```text
GET  /api/ai-rnd/project-agent/status
POST /api/ai-rnd/project-agent/plan
POST /api/ai-rnd/project-agent/start
POST /api/ai-rnd/project-agent/stop
```

会话持久化协议升级为 `ai_rnd_session.v3`，编排状态升级为 `research_orchestrator_status.v3`，项目计划和运行状态分别使用 `project_research_plan.v1` 与 `project_research_agent.v1`。刷新页面后仍可恢复任务队列、当前索引、已完成 Loop、连续未决次数和停止原因。

### 验证结果与修改边界

- 新增 5 个项目 Agent 单元测试，覆盖计划优先级、完整场覆盖缺口、任务推进、计划边界和连续未决安全停止；
- 与研究数据库既有 5 个回归测试合并运行，共 10 个测试通过；
- Python 编译检查通过；
- 项目 Agent 状态、确定性计划和编排状态三个 HTTP 冒烟检查通过；
- AI 工作区 JSX 定向 ESLint 通过，Vite 生产构建通过，共转换 2825 个模块；
- 当前会话未连接可控制的浏览器实例，截图级视觉验收仍待完成；
- 未调用外部研发模型，也未运行本地 GPU 模型，本次验证不会产生实验结论；
- 未修改 3D LLM 模型、Layer、热力图、Canvas 或相机；
- 本次记录只追加到 `ai2050_research_os/README.md`，未修改 AGI Memo。

---

## 可视化客户端 404 启动修复 [2026-08-30 00:29]

### 根因

VS Code 的 `Frontend` 启动任务能够找到 Node.js 22.21.0，但原启动器随后通过 `npm.cmd` 间接调用 Vite。在当前 VS Code 集成终端环境中，`npm.cmd` 包装进程会停留在后台，却没有创建真正的 Vite Node 子进程。表面上前端任务仍在运行，实际上默认端口 5173 没有监听服务。与此同时，环境中还残留一个 5174 端口的旧静态服务器，访问后端 5001 根路径或错误的旧地址会得到 404。

### 修改

`scripts/start_visualization.ps1` 继续负责发现符合 Vite 7 要求的 Node.js，但启动链路改为：

```text
node.exe → frontend/node_modules/vite/bin/vite.js
```

开发、构建和预览模式均不再经过 `npm.cmd`。首次安装依赖时也改为使用选中 Node 直接执行 `npm-cli.js`；Lint 模式直接执行 ESLint 入口。这样可以保持 Node 版本检查和自动发现能力，同时消除 VS Code 集成终端中的 npm 包装层挂起问题。

### 验证

- PowerShell 脚本语法检查：0 个错误；
- 修正后的启动器成功建立 5173 监听，监听进程为 Node/Vite；
- `GET http://127.0.0.1:5173/` 返回 200；
- `GET http://127.0.0.1:5173/src/main.jsx` 返回 200 和 JavaScript 模块；
- `GET http://127.0.0.1:5001/api/ai-rnd/project-agent/status` 返回 200；
- 生产构建通过，共转换 2825 个模块；
- 已停止挂起的旧启动进程和 5174 静态服务器，保留后端 5001，并让修正后的客户端持续运行在 5173；
- 本次未修改 3D LLM、Layer、热力图和 AGI Memo。

---

## 可视化客户端完整修改方案 V14：语言计算图谱—HiddenState 条件算子闭合 [2026-08-30 23:20]

### 0. 新方案的核心判断

V14 保留“研究积累、3D 观察、Loop Engineering”三个一级部分，但修正 V13 中仍可能产生歧义的地方：

1. `语言模式族`不能只是 Token、词性和构式的分类表，而要升级为**类型化语言计算图谱**；
2. `HiddenState 场`不能只保存原始向量和热力图，还要积累在具体状态、角色、深度、位置和输出目标下主动测得的**跨坐标条件响应**；
3. 外部语言图谱与内部状态图谱之间必须经过未见预测、错族/错角色/错层控制、调用、删除、救援和自由生成，才能登记为候选机制；
4. AI 模型只能生成合同、代码、审查意见和理论修订草案，不能根据热力图观感或多模型投票自动提升证据等级；
5. 当前目标首先是闭合“语言智能与预测计算”的机制，不能把它直接表述为记忆、学习、规划和世界交互均已解释的完整 AGI 理论。

统一主线固定为：

```text
类型化外部语言计算图谱
        ↓ 编译为平衡 Case / Pair
完整 Embedding 与 HiddenState 轨迹
        ↓ 自然差分 + 主动探针
状态条件化跨坐标响应图谱
        ↓ 未见预测 + 调用/删除/救援
候选条件齿轮与输出功能
        ↓ 证据账本与人工裁决
理论拼图、反例和数学闭合
```

对应的闭合目标是：

$$
\Psi_\theta\!\left(\mathcal O_f^{\rm ext}(x)\right)
\approx
\mathcal G_f^{\rm int}
\left(\Psi_\theta(x);x,r,q,c\right)
$$

并进一步验证：

$$
\operatorname{Decode}
\left(\mathcal G_f^{\rm int}(\Psi_\theta(x))\right)
\approx
\operatorname{Decode}
\left(\Psi_\theta(\mathcal O_f^{\rm ext}(x))\right)
$$

这两个公式是研究目标，不是当前已经成立的实验结论。

### 1. 产品外壳：三个稳定工作区，一套研究上下文

客户端只保留三个一级工作区：

```text
研究积累中心 | 3D 机制观察台 | AI 自动研发
```

全局上下文条只显示：

```text
当前语言操作 / 构式或对象
当前 Case / Pair
当前模型快照
当前 Run
当前证据等级与冲突状态
```

- 原有 3D 与 AI 自动研发上下滚屏切换继续保留；三个一级入口也可直接跳转，不要求用户用滚动寻找功能；
- 理论研究中心改为独立研究工作区，不再与大量旧 Campaign 面板混排；
- Phase 只作为历史 Run 的 provenance 存在于来源抽屉，不作为首页导航、进度条、任务名称或 3D 标题；
- 三个工作区读取同一数据库、同一不可变工件和同一当前选择，不复制状态；
- 默认界面只回答“研究什么、数据是否完整、证据到哪一步、下一缺口是什么”，长日志和精确参数进入 Inspector。

---

## 第一部分：研究积累中心

研究积累中心固定为三个子页：

```text
语言计算图谱 | HiddenState 条件场 | 理论与闭合
```

现有独立 `Encoding Atlas` 页并入前两个子页：外部语言图谱放在“语言计算图谱”，Embedding/HiddenState Atlas 放在“HiddenState 条件场”，避免形成第四个概念重叠的入口。

### 1.1 语言计算图谱

#### 研究对象

外部图谱固定为：

$$
\mathcal G_{\rm language}
=
\left(
V_{\rm form},
V_{\rm concept},
V_{\rm role},
V_{\rm context},
E_{\rm relation},
E_{\rm transform},
E_{\rm compose}
\right)
$$

| 对象 | 保存内容 | 示例 |
| --- | --- | --- |
| 表面形式 | Token、多 Token 词、词形、标点、语言、Tokenizer 映射 | 苹果、apple、`.` |
| 概念 | 实体、类别、属性和多义项 | 苹果、水果、可食用、苹果公司 |
| 角色 | 施事、受事、查询、边界、指代目标、作用域 | “吃苹果”中的苹果是受事 |
| 上下文 | 肯定/否定、事实/假设、引用、语言、风格、输出接口 | 否定事实问答 |
| 关系 | is-a、part-of、has-property、before、causes、refers-to | 苹果 is-a 水果 |
| 变换 | 否定、被动化、提问、翻译、标点替换、风格转换 | 肯定句→否定句 |
| 组合 | 多个角色、关系和作用域共同形成的有类型超边 | 嵌套否定与态度构式 |

名词、动词、副词、介词、标点仍然需要完整积累，但它们是表面形式或功能入口，不是唯一的理论单位。例如：

- `苹果`必须同时关联词面、概念、多义项、taxonomy、属性、角色、上下文和输出任务；
- `in`必须关联空间/时间/状态关系、源角色、目标角色和方向，不能只登记为介词；
- 标点必须登记它承担的边界、引用、语气和分区功能；
- 翻译与风格是保持若干不变量、改变若干变量的操作，不是简单 Token 族。

#### 每个语言操作的最低数据合同

```text
operation_id / family_type / language / version
form_nodes / concept_nodes / semantic_roles / context_conditions
invariants / changed_factors / counterfactual_operations
composition_inputs / composition_order / expected_outputs
wrong_family_controls / wrong_role_controls / unseen_lexicon_split
unseen_construction_split / cross_language_links
behavior_qualification / human_semantic_review / provenance
```

每个操作必须明确“保持什么、改变什么”。例如翻译保持对象、关系、真值和角色，改变表面 Token、语序和输出语言身份。没有这组定义，就不能判断内部差异来自翻译机制、词汇、Tokenizer 还是输出编译。

#### 页面布局

```text
左：类型化图谱与对象搜索
中：图节点/边、构式槽位、A/B 变换和覆盖矩阵
右：定义、Case、角色—Token 对齐、已有 Run、证据与下一缺口
```

覆盖矩阵不显示 Phase，固定显示：

```text
未定义 → 已定义 → 行为合格 → 有完整场 → 有未见预测
→ 有调用 → 有删除 → 有救援 → 独立复现
```

### 1.2 HiddenState 条件场

#### 四类必须积累的数据

1. `完整原始场`
   - Tokenizer Embedding 权重行；
   - 当前输入位置实际取出的 Embedding；
   - 全部 Token、全部 Layer、全部物理坐标的 HiddenState；
   - final norm、next-token 分布和多步输出边界。
2. `自然配对场`
   - 同一语言程序只改变一个已登记操作的 A/B；
   - 保存 A、B、B−A、角色对齐、Token 对齐和冻结颜色尺度。
3. `主动响应场`
   - 来源节点为 $(q,t,j)$，目标节点为 $(q',t',k)$；
   - 保存方向、剂量、精度、奇偶响应、重复、错方向和输出影响；
   - 方向不足时明确标为投影 $J_xP_U$，不能命名为完整 Jacobian。
4. `因果结果场`
   - 调用、删除、救援、剂量曲线、自由生成变化和副作用；
   - 区分充分性、必要性、特异性和一般残差直通。

内部事件的索引至少包含：

```text
model_snapshot / precision / sample / language_operation
surface / context / token_position / semantic_role
source_checkpoint / target_checkpoint
source_coordinate / target_coordinate
base_activation / natural_difference / active_derivative
even_response / numerical_floor / output_distribution_effect
full_generation_effect / artifact_hash / evidence_id
```

#### 页面只保留四种观察方式

```text
原始场 | A/B 差分 | 主动响应 | 输出影响
```

- 顶部选择语言操作、Case/Pair、模型和 Run；
- 左侧选择 Token、角色、Layer 和来源/目标检查点；
- 中间显示完整场或响应矩阵，默认 Tile 加载，不一次传输整个大文件；
- 右侧显示数值、工件哈希、控制、重复性、输出分布变化和候选齿轮；
- Top-K 只用于快速定位，切换“全部参数”时必须读取真实完整张量，不能补零或用 Top-K 冒充；
- 不默认生成“语言族平均向量”；任何平均、低秩、稀疏或聚类结果都作为有参数、有来源哈希的派生视图。

#### 候选条件齿轮

候选齿轮不是一个颜色区域，而是一条带条件域的有向响应子图：

```text
gear_id / external_operation
source_nodes / target_nodes / condition_domain
sign_structure / amplitude_model / output_effect
fresh_generalization / wrong_family / wrong_role / wrong_layer
call_effect / delete_effect / rescue_effect
composition_status / cross_model_status / evidence_level
```

当前更合理的待检验形式是：

$$
J_{x,f}=J_{\rm shared}+\Delta J_{x,f}
$$

以及已测方向上的候选表达：

$$
D_{x,q\rightarrow t}(u)
=s_{q,t,u}\odot a_{x,q,t,u}+\varepsilon
$$

客户端必须把这些公式标记为“候选解释/当前证据范围”，不能显示为已完成的统一定律。

### 1.3 理论与闭合

理论页固定为五块：

1. `已确认的最窄拼图`：只显示通过当前证据门的有限结论；
2. `候选机制`：预测或局部干预支持、尚未完成删除/救援的条件齿轮；
3. `失败与反例`：固定向量、同坐标传动或跨材料失败等正式负结果；
4. `最新理论`：理论名称、公式、适用域、未解释变量和最近修订；
5. `数学闭合`：逐门显示证据和阻塞原因，不用总体百分比代替。

理论记录至少保存：

```text
claim_id / statement / formula / scope / status
supporting_evidence / conflicting_evidence / counterexamples
prediction_domain / intervention_domain / output_domain
open_variables / next_experiment / closure_gate_ids
revision / reviewer / provenance
```

闭合门固定检查：

```text
对象与操作定义
行为资格
完整场与数值重复性
未见词汇/表面/构式预测
错族/错角色/错层控制
主动调用
选择性删除
独立下游救援
自由生成与输出身份
组合规律
跨语言功能对应
跨模型或异架构复现
```

证据状态统一为：

```text
未测试 / 数据不完整 / 失败 / 观察支持 / 预测支持
/ 局部因果支持 / 窄范围闭合 / 独立复现
```

AI 只能提交 Theory Revision 草案；闭合门由工件验证器和人工审核更新。现有“点击后直接标记进行中”的简单操作要改为“创建待审核闭合申请”，避免 UI 操作本身改变科学结论。

---

## 第二部分：3D 机制观察台

### 2.1 只保留三种模式

1. `静态结构`
   - 从真实模型配置读取 Embedding、Layer、Attention、MLP、Norm、Residual 和 LM Head；
   - 只显示真实存在和已采集的检查点；
   - 结构图不承载研究结论。
2. `动态运行`
   - 由真实后端事件驱动 Tokenization、Embedding、当前 Layer、已完成 Layer 和输出 Token；
   - 没有数据时显示等待/缺失，不播放模拟激活；
   - 当前 Layer 只在原子写入完整状态后更新。
3. `机制观察`
   - 从语言操作、Case、Pair、候选齿轮或理论主张进入；
   - 查看原始场、A/B、差分、主动响应、删除和救援；
   - 只绘制真实测得的响应边，不根据相关热力图自动连线。

### 2.2 固定空间语法

```text
[外部语言图谱/角色]
          ↓
[Embedding 完整场] —— [LLM 主体与当前 Layer HiddenState] —— [输出身份/生成时钟]
                          ↕
                  [条件响应边与干预结果]
```

- Embedding 热力图继续位于 LLM 主体相对展开 Layer 的另一侧，尺寸接近 LLM 主体；
- 当前 Layer 内显示对应 HiddenState，执行到新 Layer 时复用同一显示面，不生成所有 Layer 同时可见的大墙；
- 纵轴是 Token/角色，横轴是全部真实物理坐标；
- 外部语言图谱只显示当前操作涉及的局部节点和有类型边，避免把完整知识图谱塞进 3D；
- 输出侧独立显示 next-token 分布、首次分叉位置、答案身份和完整生成，避免把上游状态变化直接解释成输出功能；
- 响应边 Hover 返回来源/目标坐标、条件域、响应值、输出影响、Run 和工件哈希。

### 2.3 最小研究交互

```text
选择语言操作或构式
选择模型快照与 Case/Pair
选择 A / B / B-A / 干预 / 删除 / 救援
选择 Token、角色、Layer、来源与目标检查点
播放真实运行或拖动时间线
固定颜色尺度
加入对比篮
打开数值与证据 Inspector
```

3D 只负责空间结构、层间演化、跨 Token 路由、响应传播和异常定位。精确长表、统计审计、合同、反例和理论说明仍在 2D Inspector 中读取。

### 2.4 视觉语义与性能边界

- 激活正负使用连续色标；证据等级使用独立边框/徽标，二者不能共用颜色图例；
- 灰/蓝/黄/橙/绿/红只表达无数据、原始场、预测、因果、闭合和冲突，不表达激活数值；
- 全参数场用 WebGL Texture、InstancedMesh 或 GPU Buffer，不为每个坐标建立 React DOM/Three 组件；
- 后端提供 Tensor Tile、可见范围、取消请求和 LRU 缓存；
- 同一 Pair 使用冻结对称色阶；跨模型只比较功能角色、相对深度和输出效应，不直接比较物理坐标编号；
- 默认同时固定一个主 Pair 和少量对照，其余进入 2D 对比篮；
- 历史 Campaign 通过通用 Adapter 转成 Run/Pair/Evidence，不再继续给 `ResearchHeatmapRoute.jsx` 增加 Cxxx 专用参数和渲染分支。

---

## 第三部分：AI 自动研发与 Loop Engineering

### 3.1 两种运行模式，共用同一状态机

1. `自动运行`
   - 项目 Agent 从最高优先级 Evidence Gap 生成有界任务队列；
   - 在冻结合同、权限、GPU、时间和 Loop 预算内连续执行；
   - 遇到重要理论升级、权限扩大、有效反证、连续未决或预算耗尽时停止。
2. `手动执行`
   - 使用完全相同的数据合同、工件格式、模型调度和证据门；
   - 每个研究门后等待人工确认，可退回并生成新合同版本；
   - 已经生成的工件不可覆盖。

默认界面不显示内部 Phase，只显示：

```text
当前证据缺口 → 冻结合同 → 串行执行 → 独立复核 → 证据回写
```

### 3.2 多模型职责必须隔离

| 角色 | 职责 | 禁止事项 |
| --- | --- | --- |
| 主研发模型 | 编写合同、生成代码、综合独立审查、提出下一任务 | 直接提升 Evidence 或闭合理论 |
| 方法审查模型 | 检查样本、分区、控制、泄漏和死亡观察 | 先读取主模型结论后附和 |
| 数据审计模型 | 读取原始工件、复算指标、检查缺失和哈希 | 只阅读摘要 |
| 对抗审查模型 | 寻找反例、副作用、替代解释和错层/错族失败 | 把共识当证据 |
| 理论边界模型 | 检查主张范围、数学闭合门和越级表述 | 自动写入正式理论 |

研发代理模型与被研究的本地模型必须严格分离。Qwen3-4B、Qwen3-14B、GLM4、DeepSeek-7B 只接受实验并输出数据；同一时刻只加载一个本地模型，完成卸载和显存审计后再测试下一个。

### 3.3 提示词和模型配置

每个代理保存：

```text
provider / api_base / model_id / role / enabled
timeout / token_budget / temperature / retry_policy
system_prompt / task_prompt / output_schema
prompt_version / prompt_sha256
```

- API Key 只保存在后端秘密配置，不进入浏览器存储、数据库导出和运行工件；
- 辅助模型默认独立接收冻结合同与原始工件索引，不接收主模型结论；
- 主模型在辅助报告全部完成后综合下一方案；
- 每次运行保存实际模型身份、提示词版本、输入摘要哈希和输出原文。

### 3.4 一轮 Loop 的正式合同

```text
1. 从数据库选择一个 Evidence Gap
2. 冻结语言操作、Case/Pair、Claim 与 Evidence Snapshot
3. 写明竞争解释、可区分预测和死亡观察
4. 辅助模型独立审查方法和控制
5. 主模型生成 loop_contract 与受约束测试脚本
6. AST、目录、权限、样本量、Schema 和预算验证
7. 行为资格与小规模 Smoke
8. 扩大样本正式测试
9. 合格本地模型依次串行运行
10. 保存完整场、响应、输出、模型快照和资源审计
11. 数据审计与对抗审查独立复核原始工件
12. 确定性规则形成 accepted/rejected/inconclusive
13. 事务写入 Run、Pair、Probe、Evidence 与 Artifact
14. 创建 Theory Revision 草案或下一 Evidence Gap
```

### 3.5 自动研发页面

一级只保留四个入口：

```text
项目 Agent | 单目标 Loop | 模型与提示词 | 证据与工件
```

`项目 Agent` 显示当前大目标、有界任务队列、当前任务、最大 Loop、连续未决阈值、停止原因和人工接管按钮。`单目标 Loop` 用于研究者明确指定一个操作或证据缺口。计划执行完毕只表示任务队列完成，不表示理论成立。

每个 Loop 的不可变目录至少保存：

```text
loop_contract.json / evidence_snapshot.json
agent_steps.jsonl / prompt_manifest.json / generated_script.py
model_snapshot.json / cases.jsonl / trace_manifest.json
tensor_manifest.json / paired_responses.jsonl / probe_responses.jsonl
interventions.jsonl / generation_outputs.jsonl
artifact_manifest.json / artifact_audit.json / decision.json
theory_update_draft.json
```

数据库更新必须使用事务：先检查文件、Schema、哈希、分区和控制，再登记 Run/Pair/Probe/Evidence，最后更新任务状态。验证失败时回滚索引但保留失败工件，用于定位系统问题。

---

## 4. 共用数据模型与服务

### 4.1 四组核心对象

| 数据域 | 核心对象 |
| --- | --- |
| 外部语言图谱 | `language_nodes / language_edges / operations / constructions / cases / role_alignments` |
| 内部状态与响应 | `runs / tensor_artifacts / pair_alignments / probe_responses / interventions / generation_outputs` |
| 机制与理论 | `gear_candidates / evidence_records / claims / puzzles / closure_gates / theory_revisions` |
| 自动研发 | `loop_runs / loop_contracts / agent_steps / prompt_profiles / artifact_audits / scheduler_jobs` |

现有 `language_objects`、`field_records`、`theory_claims`、`closure_gates` 和 `loop_runs` 不删除：先迁移为兼容视图，再逐步补齐图节点、操作、响应和证据引用。所有旧记录默认保持原 Evidence，不因迁移自动升级。

### 4.2 服务边界

```text
LanguageGraphService   类型化图谱、构式、操作和版本
CaseCompiler           图谱操作→平衡 Case/Pair 与角色—Token 对齐
TraceService           完整场、输出时钟和真实运行事件
TensorTileService      大张量分片、范围查询、取消和缓存
OperatorLab            探针、响应矩阵、候选齿轮与组合测试
EvidenceService        证据、反例、Claim、Puzzle、Closure Gate
ModelScheduler         单 GPU 串行调度、卸载、恢复和资源审计
LoopEngine             自动/手动状态机与项目 Agent
PromptRegistry         模型角色、提示词版本和哈希
ArtifactValidator      Schema、分区、控制、哈希和完整性
```

客户端只通过版本化 API 和 SSE/WebSocket 事件读取投影，不直接读取 SQLite 或扫描结果目录。大张量正文不写入 SQLite，只保存内容寻址路径、形状、dtype、检查点、哈希和 Tile 索引。

### 4.3 客户端代码目标结构

```text
frontend/src/researchWorkspace/
  shell/          三个工作区、全局上下文和导航
  language/       图谱、构式、Case 与覆盖矩阵
  field/          原始场、Pair、Probe、输出影响
  theory/         Claim、反例、拼图和闭合门
  three/          静态结构、动态运行、机制观察
  loop/           项目 Agent、单目标 Loop、代理配置
  inspectors/     数值、工件、来源和审查
  store/          统一选择、查询缓存和实时事件
  adapters/       旧 Phase/Campaign 到稳定对象的只读转换
```

- `App.jsx` 只保留 Shell、路由和全局选择；
- `ResearchHeatmapRoute.jsx` 拆成通用 Trace Adapter、Tile Loader、Heatmap Texture、Layer State、Pair Diff、Response Graph 和 Evidence Inspector；
- 新实验只增加数据和 `view_spec`，不能新增 Cxxx 专用 React 参数；
- 研究中心和 3D 使用同一 `selectedOperation / selectedCase / selectedPair / selectedRun / selectedClaim`；
- 所有源码统一 UTF-8 后再迁移，历史乱码不进入新模块。

---

## 5. 从当前实现迁移的实际差距

| 当前已有 | 当前不足 | V14 修改 |
| --- | --- | --- |
| SQLite `language_objects` | 只有 object_type/family，无法表达角色、上下文和有类型边 | 增加节点、边、操作、构式和角色对齐表 |
| `field_records` 保存完整场索引 | 还没有统一 Pair、Probe、响应矩阵和输出影响对象 | 增加 tensor manifest、pair alignment、probe response 和 intervention |
| 理论主张与闭合门页面 | 状态可以被简单 UI 操作推进，证据引用不够严格 | 改为闭合申请、确定性验证和 Theory Revision 草案 |
| Encoding Atlas + 三个研究页 | 入口概念重叠，默认仍有四个标签 | Atlas 并入语言图谱和 HiddenState 条件场 |
| 3D Embedding/HiddenState 热力图 | 仍有大型单文件和 Campaign 专用分支 | Tile 化、通用视图、条件响应边和输出身份面板 |
| 项目 Agent 和单目标 Loop | 已能生成有界任务并回写 Loop 结果，但任务来源仍较粗 | 直接以 Operation/Evidence Gap/Closure Gate 为任务合同 |
| 主模型与辅助模型配置 | 已有基本提示词字段 | 增加角色隔离、版本哈希、输出 Schema、独立审查输入边界 |
| GPU 串行锁和工件审计 | 已有执行底座 | 扩展到行为资格、完整场、Probe、卸载和跨模型资格审计 |

---

## 6. 实施里程碑

不再用大量 Phase 管理客户端开发，改为七个可验收里程碑：

1. `M0 冻结研究合同`
   - 固定语言图谱、完整场、Probe、Gear、Evidence 和 Theory Revision Schema；
   - 固定证据状态机、因果门、不可变工件和版本规则。
2. `M1 类型化语言图谱`
   - 完成数据库迁移、图谱编辑器、构式槽位、操作不变量/变量和 Case 编译预览；
   - 先覆盖 taxonomy、标点、介词、否定、翻译和风格六类代表操作。
3. `M2 条件场与响应资产`
   - 自动导入完整场；
   - 增加 Pair、主动探针、输出影响和 Tile API；
   - 明确数值精度、剂量和噪声底审计。
4. `M3 研究积累中心闭环`
   - 将 Atlas 合并为三个子页；
   - 完成覆盖矩阵、候选齿轮、反例、证据引用和闭合申请。
5. `M4 3D 通用观察台`
   - 在数据合同稳定后拆分旧热力图组件；
   - 接入真实静态结构、动态事件、Pair、Response 和输出身份；
   - 此前不继续增加 3D 特效或 Campaign 专用面板。
6. `M5 Loop Engineering 真闭环`
   - Agent 从图谱与 Evidence Gap 生成合同；
   - 自动保存 Case、完整场、Probe、干预、审计和裁决；
   - 事务更新数据库并创建理论修订草案。
7. `M6 因果与组合扩展`
   - 完成调用—删除—救援、未见构式、自然语言、组合次序、跨语言可交换性和跨模型功能复现；
   - 只有通过这些门后才讨论可组合语言机制。

### 第一批应立即实施的任务

```text
1. 把现有 language_objects 迁移为 LanguageNode + Operation + Construction
2. 为 Case 增加 invariants、changed_factors、roles 和 split
3. 为 field_records 增加 tensor manifest 与 capture point
4. 新建 PairAlignment、ProbeResponse、Intervention、GearCandidate
5. 把理论闭合按钮改为“提交闭合申请”
6. 将 Encoding Atlas 合并进三个研究子页
7. 让项目 Agent 优先读取 Operation、Evidence Gap 和 Closure Gate
8. 冻结上述 API 后，再修改 3D 数据适配层
```

---

## 7. 最终验收

### 研究积累

- 登记“苹果”时可以看到词面、Tokenizer、概念、多义项、taxonomy、属性、角色、上下文和输出任务，而不是只有 noun/family；
- 可以登记标点、介词、翻译和风格操作，并明确不变量、变化量、错族/错角色控制和未见分区；
- 任一 Case 都能追溯到语言操作、构式、角色—Token 对齐、模型快照和不可变工件；
- 任一完整场都能查看全部 Token、全部 Layer 和全部真实物理坐标；
- 任一候选齿轮都能查看主动响应、输出影响、负控、调用、删除、救援和证据等级。

### 3D 观察

- 静态结构来自真实模型配置，动态运行来自真实事件；
- Embedding 在 LLM 另一侧以完整场显示，当前 Layer 显示同一 Run 的完整 HiddenState；
- 可以切换 A、B、B−A、调用、删除和救援，并固定颜色尺度；
- 只有真实测得的条件响应才显示方向边；颜色相似不会自动生成“机制”；
- 所有热力图格子和响应边都能回到 raw value、Run、Case、Pair 和 artifact hash。

### Loop Engineering

- 自动与手动使用同一状态机；
- 主模型负责编程和综合，辅助模型独立审查原始工件；
- 模型、提示词、角色、版本、哈希、代码、失败和裁决均完整保存；
- 本地模型严格单 GPU 串行运行并完成卸载审计；
- 每轮结果事务写入数据库，失败工件保留但不会提升 Evidence；
- 项目 Agent 能从最高优先级证据缺口继续下一任务，并在反证、连续未决、预算或权限边界停止。

### 科学红线

- Token 分类不等于语言计算图谱；
- HiddenState 可分类不等于可预测；
- 热力图相关不等于内部算子；
- 固定方向有效不等于固定语言族向量；
- 局部响应不等于完整 Jacobian；
- 调用成功不等于必要机制；
- 删除失败不等于不存在分布式机制；
- 多模型 AI 共识不等于实验复现；
- 任务队列完成不等于理论闭合；
- 语言机制闭合不等于完整 AGI 理论完成。

V14 的产品定义是：

> 客户端不是 Token 分类器、热力图播放器或 AI 自动写代码面板，而是一套把外部语言操作、完整内部状态、主动因果响应、输出功能和理论证据连接起来的持续研究操作系统。

本节是根据附件思路形成的客户端设计与实施方案，不是新的模型测试结果。本轮只更新 `ai2050_research_os/README.md`，未修改客户端代码、3D 空间和 AGI Memo。

---

## V14 第一批实施记录：研究合同与三个研究入口 [2026-08-30 23:50]

本批已经从方案进入代码实施，范围严格限定为研究数据合同、研究积累中心和项目 Agent；没有修改 3D 空间，也没有写入 `research/glm5/docs/AGI_GLM5_MEMO.md`。

### 已完成

1. 研究数据库从 `research_workspace.v1` 兼容迁移到 `research_workspace.v2`：
   - 保留旧 `language_objects`、`field_records`、`theory_claims` 和已有 API；
   - 新增 `LanguageNode`、`LanguageEdge`、`LanguageOperation`、`Construction`、`ResearchCase`；
   - 新增 `PairAlignment`、`ProbeResponse`、`GearCandidate`、`Intervention`；
   - 新增 `ClosureApplication`，把理论闭合从直接改状态改为审核申请。
2. 初始化六类 E0 语言操作：类型关系、标点与边界、介词关系绑定、否定与作用域、跨语言运输、风格变换。所有种子均标记为 `untested`，不作为实验依据。
3. 研究中心精简为三个固定入口：
   - `语言计算图谱`：显示语言操作、类型化节点、图谱边、构式和冻结 Case；
   - `HiddenState 条件场`：显示完整原始场、Pair、Probe、候选齿轮和因果干预；
   - `理论与闭合`：显示理论主张、关键拼图、闭合门和审核队列。
4. 移除研究中心的独立 `Encoding Atlas` 导航入口；完整场索引和响应资产统一进入 `HiddenState 条件场`。
5. 移除前端“标记进行中”直接改闭合门的操作，改为“提交闭合申请”。提交申请后闭合门保持原状态，等待独立审核。
6. 项目 Agent 新增两个任务来源：
   - 优先读取语言操作的行为状态与 `next_evidence_gap`；
   - 读取待审核闭合申请，只生成审计任务，不自动批准申请或提升理论等级。

### 新增 API

```text
POST /api/research-workspace/language-nodes
POST /api/research-workspace/language-edges
POST /api/research-workspace/operations
POST /api/research-workspace/constructions
POST /api/research-workspace/cases
POST /api/research-workspace/pair-alignments
POST /api/research-workspace/probe-responses
POST /api/research-workspace/gear-candidates
POST /api/research-workspace/interventions
POST /api/research-workspace/closure-applications
```

`GET /api/research-workspace/snapshot` 同时返回 V1 兼容字段和全部 V2 资产，不破坏现有调用方。

### 验证结果

- Python 编译检查：通过；
- 数据库迁移、API、闭合申请与项目 Agent 回归：13 项全部通过；
- 前端目标文件 ESLint：0 error、0 warning；
- Vite 正式构建：通过，共转换 2822 个模块；
- 本地 HTTP 联通：前端 5173 返回 200，后端 5001 返回 `research_workspace.v2`，六类操作种子可见；
- 构建仍提示主包大于 500 kB，这是现有客户端的性能提醒，不影响本次功能正确性，后续应通过按工作区动态加载处理；
- 当前会话没有连接可控浏览器窗口，因此没有完成截图级视觉检查；构建、HTTP 和开发服务器模块检查均已通过。验证记录保存在 `tests/glm5/result/research_workspace_v14/validation_20260830_2349.json`。

### 下一批边界

1. 补齐图谱边、构式、Pair 和因果干预的精简编辑器；
2. 为完整场增加 tensor manifest、capture point、dtype、hash 和 Tile API；
3. 增加 Evidence 统一引用和闭合申请人工审核动作；
4. 数据合同冻结且真实工件可读取后，再开始修改 3D 适配层。

本批没有运行本地大模型，没有产生新的 HiddenState 科学结论；六类操作种子与空资产计数只能说明系统结构已准备好，不能说明语言机制已经得到支持。

---

## 可视化客户端修改方案 V15：Codex 式研究代理与 HiddenState 全场数字显微镜 [2026-09-01 02:27]

本方案综合两份附件，并建立在 V14 已完成的三入口、研究数据合同和项目 Agent 基础之上。本轮只形成客户端与接口修改方案，不修改 3D 代码，不修改 `research/glm5/docs/AGI_GLM5_MEMO.md`，也不把附件中的候选理论当作已经证实的结论。

### 1. 方案结论

客户端仍然只有三个一级工作区：

1. `研究积累`：语言计算图谱、HiddenState 条件场、理论与闭合；
2. `3D 机制显微镜`：观察真实模型结构、真实运行事件和实验场；
3. `AI 自动研发`：参考 Codex 的项目任务、对话线程、计划、终端、差异和审查交互，形成可持续执行的研究代理。

V15 不再以 Phase、Campaign 或大量专题面板组织界面。Phase 只作为历史来源元数据存在，不参与主导航。三个工作区共享同一个当前研究上下文：

```text
语言操作 / 构式 / Case 或 Pair / 模型快照 / Run / 数据分区 / 证据状态
```

附件中的两条主线被统一为一条可检验链路：

```text
外部语言操作
  → 可复查的语言图谱与对照样本
  → 完整 HiddenState 场
  → 条件相关的局部更新
  → 输出贡献与生成行为
  → 冻结预测
  → 未见数据与因果干预
  → 证据或反证
```

候选桥接公式保留为研究目标，而不是 UI 中的既定事实：

\[
\Psi_\theta\!\left(O_f^{ext}(x)\right)
\approx
G_f^{int}\!\left(\Psi_\theta(x);x,r,q,c\right)
\]

这里的内部算子依赖样本、角色、层深、上下文和输出目标，不能预设为固定语义方向或固定神经元集合。

### 2. 全局界面骨架

采用一个稳定的三入口侧栏和一个共享上下文条：

```text
┌──────────┬──────────────────────────────────────────────────────┐
│ 研究积累 │ 当前操作 · Case/Pair · Model · Run · Split · Evidence │
│ 3D 显微镜├──────────────────────────────────────────────────────┤
│ AI 研发  │                    当前工作区                         │
└──────────┴──────────────────────────────────────────────────────┘
```

- 侧栏只显示三个一级入口，不展开 Phase 列表；
- 顶部上下文条始终可见，切换工作区不丢失当前实验对象；
- 继续允许 3D 与 AI 研发上下滚屏切换，但不能把滚屏作为唯一入口；侧栏、快捷键和“在 3D 中打开”均可直接切换；
- URL 保存 `operation/case/pair/model/run/view`，刷新或分享后可恢复现场；
- 数据缺失时显示“尚未采集/等待运行”，禁止使用模拟激活填充真实研究视图。

---

## 3. AI 自动研发界面：改成 Codex 式研究任务工作台

“参考 Codex”指参考项目上下文、任务线程、计划、执行、终端、差异和审查的交互逻辑，不复制品牌外观。现有四个大 Tab 和多个表单面板应重组为一个任务中心：

```text
┌──────────────┬──────────────────────────────┬──────────────────┐
│ 项目与线程    │ 当前研究任务线程              │ 运行检查器        │
│              │                              │                  │
│ 进行中        │ 目标 / 约束 / 冻结合约        │ Plan             │
│ 等待确认      │ Agent 分析与操作记录          │ Contract         │
│ 已阻塞        │ 工具调用 / 产物 / 测试摘要    │ Artifacts        │
│ 已完成        │ 主模型综合 / 审核裁决          │ Diff / Tests     │
│              │                              │ Evidence         │
├──────────────┴──────────────────────────────┴──────────────────┤
│ 输入研究任务 · 附加 Case/文件/证据 · 自动/手动 · 开始/继续      │
├────────────────────────────────────────────────────────────────┤
│ 可收起底栏：终端 / 实时日志 / GPU / 测试输出 / 代码差异          │
└────────────────────────────────────────────────────────────────┘
```

### 3.1 左栏：项目与研究线程

- 用“一个目标对应一个持久线程”替代单纯的运行记录；
- 分组仅保留：进行中、等待人工确认、阻塞、已完成；
- 每条线程显示目标、当前步骤、主模型、最近更新时间和证据缺口；
- 支持搜索、固定、继续、复制为新实验和归档；
- 队列任务属于线程内部计划，不再单独占据一块主面板。

### 3.2 中栏：可审计任务时间线

一条线程按时间呈现：

1. 用户目标与研究边界；
2. Agent 读取到的语言操作、Case、证据缺口和闭合门；
3. 冻结后的执行计划与数据合同；
4. 主模型的实现动作；
5. 辅助模型彼此独立的分析和反证意见；
6. 命令、代码差异、测试、模型运行和产物摘要；
7. 主模型综合结论；
8. 确定性验证器的裁决；
9. 下一步任务、停止原因或人工确认请求。

冗长 stdout、完整张量和工具参数默认折叠，只在点击时展开；任何结论都必须能回到命令、代码版本、输入、模型快照和工件哈希。

### 3.3 底部任务输入框

输入框同时承担新任务与继续任务：

- 可附加 Operation、Construction、Case、Pair、Claim、Evidence、文件或当前 3D 视图；
- 可选 `自动运行` 或 `手动逐步`；
- 可选主模型、辅助模型组、提示词版本、时间/Token/GPU 预算；
- 提交前显示将要授予的目录、命令、数据库和模型权限；
- 快捷动作：`生成计划`、`开始`、`执行下一步`、`暂停`、`停止`、`从冻结合约重试`。

### 3.4 右栏：运行检查器

右栏只保留七个紧凑页签：

| 页签 | 内容 |
|---|---|
| Plan | 目标、步骤、完成条件、依赖和预算 |
| Contract | 样本、对照、不变量、变化量、capture point、split 和禁止泄漏规则 |
| Artifacts | 数据、图、日志、模型输出、哈希和生成者 |
| Diff | 代码与配置差异、影响文件、回滚点 |
| Tests | 编译、测试、模型实验、正负控制和失败原因 |
| Evidence | 支持、反证、未知、证据等级和闭合缺口 |
| Models | 主模型、辅助模型、提示词版本、上下文与资源占用 |

权限和危险操作只在需要时以确认卡出现，不再常驻占用主视野。

### 3.5 多模型研究流程

```text
主模型制定冻结计划
  → 主模型编程或组织实验
  → 辅助模型分别读取原始工件并独立分析
  → 主模型比较一致点、冲突点和反证
  → 确定性程序检查测试、哈希、分区和阈值
  → 保存全部结果
  → 只生成 Evidence 或 Closure 申请草稿
```

- 辅助模型意见不能在完成前互相覆盖，避免伪共识；
- AI 共识不是科学证据；只有可复现工件与预先冻结的判据可以改变证据状态；
- Agent 不得直接批准闭合、提升理论等级或修改 lockbox；
- 本地大模型按一个 GPU 任务一个模型串行执行，完成卸载审计后才切换下一个模型；
- 失败工件、否定结果和相互冲突的审查均永久保存。

### 3.6 Agent 事件合同

前端不再拼接多种状态对象，统一读取事件流：

```json
{
  "thread_id": "thread-...",
  "task_id": "task-...",
  "step_id": "step-...",
  "role": "primary|analyst|validator|human",
  "kind": "plan|message|command|artifact|diff|test|decision|approval",
  "status": "queued|running|passed|failed|blocked|cancelled",
  "summary": "...",
  "artifact_ids": [],
  "evidence_ids": [],
  "approval_required": false,
  "created_at": "..."
}
```

建议新增接口：

```text
GET    /api/research-agent/threads
POST   /api/research-agent/threads
GET    /api/research-agent/threads/{id}
GET    /api/research-agent/threads/{id}/events       # SSE
POST   /api/research-agent/threads/{id}/actions
GET    /api/research-agent/artifacts/{id}
GET    /api/research-agent/diffs/{id}
GET    /api/research-agent/runs/{id}/logs
```

浏览器不直接获得任意 shell。终端面板显示由后端权限策略执行的结构化命令、工作目录、退出码和输出；破坏性操作、越界目录和证据状态修改必须人工确认。

---

## 4. 3D 空间：从热力图展示器改成全场数字显微镜

3D 空间只保留三种顶层模式：

1. `静态结构`：来自真实模型配置的 Embedding、层、Attention、MLP、Norm 与输出头；
2. `实时运行`：来自真实采集事件的 token 流、当前层、执行状态和输出过程；
3. `机制显微镜`：查看 Case/Pair 的完整场、条件差分、局部更新、主动响应与输出贡献。

不再为每个 Cxxx 测试增加一个模式或一套组件属性。

### 4.1 固定空间布局

```text
外部语言局部图  →  Embedding 完整场  →  LLM 主体  →  响应/更新场  →  输出编译
                                      │
                                      └─ 当前 Layer：HiddenState 完整场

底部时间轴：语义事件 / Token 对齐 / Layer 进度 / 生成步骤
```

- `Embedding 完整场` 放在 LLM 主体另一侧，尺寸与 LLM 主体近似相同；它到 LLM 的距离，与展开 Layer 到 LLM 的距离一致；
- 当前 Layer 的 `HiddenState 完整场` 锚定在正在执行的 Layer 上；切换或实时执行到某层时只替换该层数据；
- Embedding 和 HiddenState 默认显示全部 token、全部物理坐标；`Top-K` 只是观察过滤器，绝不能被保存为“完整场”；
- 外部语言图只显示当前 Case 周围的局部操作、角色、构式和对照，不在 3D 中堆叠整张知识图谱；
- 响应边只显示真实测得的条件响应或干预响应，未测得的边不绘制；
- 输出侧显示当前 token 的输出贡献、候选概率和完整生成时间线。

### 4.2 八类场视图

3D 数据层统一支持以下基础场：

\[
H_{q,t,j}
\]

原始 HiddenState；

\[
\Delta_qH = H_{q+1}-H_q
\]

层更新；

\[
\Delta_fH = H(f_{on})-H(f_{off})
\]

条件差分；

\[
U_{f,u,q,t,j}=\Delta_qH(f_{on})-\Delta_qH(f_{off})
\]

局部更新差分，是优先观察对象；

\[
G_f=\mathbb{E}_u[U_{f,u}]
\]

模式族共享场；

\[
D_{f,u}=U_{f,u}-G_f
\]

样本差异场；

\[
I_{f,g}=U_{f+g}-U_f-U_g
\]

组合交互场；

\[
C_{j,v}=W_{U,vj}\widetilde H_j
\]

坐标到输出词项的编译贡献。

默认快捷入口只显示 `原始 H / 层更新 ΔH / 条件更新 U / 输出贡献 C` 四项，其他场放入“更多比较”，避免主界面复杂化。

### 4.3 观察控制

- 固定提供 `A / B / B-A`、调用、删除、救援切换；
- 颜色范围、零点、裁剪值和排序方式可冻结；A/B 必须共享同一色标；
- 坐标顺序默认使用真实物理顺序；允许保存“响应指纹冻结顺序”，但进入 lockbox 后不得重新排序；
- 相邻坐标不暗示语义相邻，聚团颜色不自动等于功能模块；
- 支持 Overview、Embedding、Current Layer、Response、Output 五个相机预设；
- 同一时刻只突出一个主场；需要比较时固定 A/B 两幅，不建立无限热力图墙；
- 鼠标点击任意格子或响应边，检查器必须显示 raw value、token/事件、role、layer、coordinate、run、dtype、scale、source、target、输出影响和 artifact hash。

### 4.4 与 3D 联动的二维研究视图

以下视图不强塞进 3D，而是在右侧检查器以二维小图联动：

1. Layer × Coordinate；
2. Semantic Event × Layer；
3. Token × Coordinate；
4. Family × Coordinate；
5. Sample × Coordinate；
6. Source Layer × Target Layer；
7. Coordinate-group 响应图；
8. 正负控制和数据完整性面板。

点击二维单元会定位 3D 的对应层、token 和物理坐标；点击 3D 对象也会反向筛选二维视图。

### 4.5 发现区、确认区与 Lockbox

3D 顶部始终显示当前数据分区：

```text
Discovery → Confirmation → Lockbox
```

- Discovery 可以探索排序、阈值和候选齿轮；
- 提交预测时冻结样本规则、坐标顺序、颜色尺度、层窗口、指标和失败条件；
- Confirmation 只验证已经冻结的预测；
- Lockbox 不能重新排序、重新筛选或调整阈值来追随结果；
- 随机方向、同范数方向、错模式族、错角色、错层和错位置控制长期常驻；
- 一个视觉纹理只能生成候选假设，不能直接升级为机制或理论闭合。

### 4.6 数据与渲染合同

完整张量不能作为一个巨大 JSON 进入浏览器。新增统一资产：

```text
TensorManifest      # shape、axis、dtype、scale、capture point、hash
TensorTile          # 按 token/layer/coordinate 分块的原始数据
FieldViewSpec       # 字段、比较、顺序、色标、过滤器和相机
SemanticEvent       # token 与语义事件、角色、构式的对齐
DerivedField        # ΔqH、ΔfH、U、G、D、I、C 的可复算定义
ResponseEdge        # 已测 source/target、方向、强度、控制和工件
FrozenHypothesis    # 预测、视图规格、split、阈值和失败条件
ComparisonSet       # A/B/调用/删除/救援的统一对齐
```

建议新增接口：

```text
GET  /api/research-fields/manifests/{id}
GET  /api/research-fields/manifests/{id}/tiles
POST /api/research-fields/view-specs
GET  /api/research-fields/view-specs/{id}
POST /api/research-fields/frozen-hypotheses
GET  /api/research-fields/comparisons/{id}
GET  /api/research-fields/response-edges
```

前端通过 WebGL texture/instancing、分块加载、LRU 缓存、请求取消和逐级细化显示全量参数。缩放前先显示低分辨率概览，停留后加载原始 tile；“全部”表示所有 tile 可访问，不要求一次把所有值放入内存。

### 4.7 关于流形、曲率与 Jacobian 的边界

附件提出的流形、曲率、切空间和张量场可以保留为候选派生视图，但不能成为默认解释或已证实的产品标签。首版只显示直接测得的场、差分、干预和输出贡献。

- 局部响应不等于完整 Jacobian；
- 可视化弯曲不等于已证明存在语义流形；
- 相似轨迹不等于测地线；
- 坐标群响应不等于已经发现稳定齿轮；
- 若以后增加候选数学视图，必须同时显示假设、计算定义、适用条件、负控制和反例。

优先测量有限且可复算的投影响应，例如 \(J_xP_U\)，不在客户端承诺构造不可承受的“全 Jacobian”。

---

## 5. 三个工作区的闭环联动

```text
研究积累选择 Operation / Case / Pair
  → AI 研发冻结合约并执行采集
  → 3D 显微镜打开真实工件
  → 研究者冻结一个可反驳的视觉预测
  → 一键创建新的 Agent 验证线程
  → Confirmation / Lockbox / 因果干预
  → 确定性验证器生成 Evidence
  → 理论中心提出 Closure 申请
```

关键交互只有三个：

- `在 3D 中打开`：Agent 任务或研究记录恢复精确的 run、view spec 和相机；
- `冻结为假设`：3D 保存视图、排序、色标、split、预测和失败条件；
- `创建验证任务`：把冻结假设送入 AI 研发线程，Agent 只能执行和提交证据，不能自己批准结论。

### 5.1 语言模式族的默认研究入口

首批模板覆盖：

- 水果—植物—食物—物体等类型关系；
- 标点、边界、介词、角色绑定与语法构式；
- 翻译、风格、否定、作用域和上下文切换；
- 多操作组合与顺序变化。

这里的“图谱”是生成实验与对照的类型化操作图，不是枚举所有 token。每个模板必须声明不变量、变化量、角色、自然语言实现、错误族/错误角色控制和未见分区。

---

## 6. 代码重构映射

| 现有文件 | 修改方向 |
|---|---|
| `frontend/src/researchCenter/LoopEngineeringWorkspace.jsx` | 拆成 `ResearchAgentShell`、`ThreadRail`、`TaskThread`、`TaskComposer`、`RunInspector`、`TerminalDrawer` |
| `frontend/src/components/app/ResearchHeatmapRoute.jsx` | 停止增加 Cxxx 专项 props；拆成 `TraceAdapter`、`TensorTileClient`、`FieldTexture3D`、`LayerFieldPlane`、`ResponseGraph3D`、`OutputCompileView`、`MicroscopeInspector` |
| `frontend/src/components/app/ResearchSpaceOverlay.jsx` | 只保留当前研究上下文、比较控制、图例和数据分区；长证据链移入检查器 |
| `frontend/src/App.jsx` | 只负责三个工作区路由、共享选择和懒加载，不继续承载研究业务分支 |
| `frontend/src/researchCenter/ResearchCenter.jsx` | 读取统一选择对象，与 Agent 和 3D 共享 Operation/Case/Pair/Run |

迁移期允许 `TraceAdapter` 把现有 Cxxx 数据转换为统一 `FieldViewSpec`，但新测试不得再给 3D 组件增加专项布尔值或专项属性。

---

## 7. 实施顺序

这里使用里程碑而不是 Phase，避免再次把客户端做成 Phase 浏览器。

### M1：共享研究上下文与轻量导航

- 建立三个一级入口、共享 selection store 和可恢复 URL；
- 保留现有页面能力，先完成壳层迁移；
- 验收：切换研究积累、3D、AI 研发时 Case/Pair/Run 不丢失。

### M2：Codex 式 Agent 线程界面

- 用现有 plan/start/pause/step/stop 接口驱动新任务线程；
- 先用适配器把现有轮询状态转换为统一事件；
- 完成任务输入、时间线、检查器和可收起日志底栏；
- 验收：一个项目研发目标可以从计划持续执行到测试，所有动作按时间可审计。

### M3：Agent 真实事件、产物与差异

- 增加 SSE、artifact、diff、test 和 approval 接口；
- 完成主模型与辅助模型独立审查卡；
- 验收：刷新页面后线程、日志、代码差异、测试和停止原因完全恢复。

### M4：统一完整场数据合同

- 实现 TensorManifest、Tile API、FieldViewSpec 和 FrozenHypothesis；
- 用适配器接通一个现有真实 Run，验证全 token、全 layer、全 coordinate 可寻址；
- 验收：`Top-K` 与 `全部` 只改变显示，不改变底层工件和证据身份。

### M5：3D 机制显微镜

- 先实现固定布局、Embedding 完整场、当前 Layer HiddenState 完整场和 A/B/Δ；
- 再实现响应边、输出贡献、语义事件时间轴和二维联动；
- 验收：任一可见格子可追溯到 raw value 和 artifact hash，切层显示同一 Run 的正确数据。

### M6：冻结预测与因果验证闭环

- 连接“冻结为假设”“创建验证任务”“提交 Evidence/Closure 申请”；
- 加入调用—删除—救援、未见组合、错族/错角色/错层控制；
- 验收：发现区产生的排序和阈值不能在 Lockbox 结果出现后改变。

---

## 8. 验收标准

### AI 自动研发

- 首页不再是四个互相割裂的大面板，而是一个项目线程工作台；
- 目标、计划、命令、差异、测试、原始工件、辅助审查和裁决在同一线程内可追踪；
- 自动与手动共用同一状态机，任何时刻可暂停并从冻结合约继续；
- AI 无权直接把结果升级为理论闭合；
- 本地模型严格串行，GPU 占用和卸载状态可见。

### 3D 空间

- 主界面只有静态结构、实时运行、机制显微镜三个模式；
- Embedding 位于 LLM 另一侧，大小接近 LLM，距离与展开 Layer 对称；
- 当前 Layer 上显示对应的完整 HiddenState，包含全部 token 与全部物理坐标；
- 支持 A/B/Δ、调用/删除/救援、固定色标和冻结顺序；
- 不为新实验增加 Cxxx 专项 3D 分支；
- 所有显示来自真实工件，缺失数据明确为空；
- 视觉模式只能生成候选假设，不能显示成已证明机制。

### 科学闭环

- 每个候选机制均能回到外部语言操作、内部场、输出功能和未见预测；
- 每项证据都记录模型、代码、数据分区、运行、视图规格、阈值、负控制和哈希；
- Token 分类不等于语言图谱闭合，热力图相似不等于内部算子，局部线性不等于全局数学结构；
- 多模型一致意见不替代复现实验、因果干预和确定性验证。

V15 的产品定义是：

> 客户端不是更大的研究仪表盘，而是一套由 Codex 式研究代理持续执行、由完整 HiddenState 场显微镜进行观察、由冻结预测和因果实验约束结论的语言机制研究操作系统。

---

## V15 第一批实施记录：Codex 式白色研究代理工作台 [2026-09-01 03:07]

本批完成 AI 自动研发界面的实际重构，未修改 3D LLM 模型、3D 热力图数据逻辑和 `research/glm5/docs/AGI_GLM5_MEMO.md`。

### 已完成

1. 删除原来以 `项目 Agent / 单目标 Loop / 代理与提示词 / 证据` 组织的四个主分页，把已有能力收敛到一个持续任务工作台；
2. 新界面采用三个固定区域：
   - 左侧：当前项目、任务队列、最近 Run 和搜索；
   - 中间：研究目标、Agent 计划、实时执行事件、停止原因和历史 Run；
   - 右侧：计划、契约、产物、测试、证据和模型配置检查器；
3. 底部增加类似研发客户端的任务输入区，可直接选择自动/手动模式、Loop 上限、生成计划、配置模型、开始、暂停、继续、逐门确认和停止；
4. 增加可收起的实时日志、生成代码和测试输出抽屉；
5. 接入已有 `/api/ai-rnd/session/events` SSE 事件流，同时保留五秒状态轮询作为恢复机制；
6. 继续复用现有 Project Agent、Session、Config 和 Orchestrator 接口，没有伪造尚未实现的线程、终端或 Diff 后端；
7. 模型与提示词配置移动到右侧检查器，主研究页面不再被大块配置表单占据；
8. AI 自动研发工作区整体改为白色主题，包括页面背景、顶栏、任务线程、输入区、检查器和日志抽屉；3D 工作区继续保持原有主题；
9. 增加 1120px、860px 和 640px 三档响应式布局，窄屏依次隐藏运行状态组、右侧检查器和左侧线程栏；
10. 所有科学边界继续保留：Agent 只能执行有界任务、保存工件和整理证据，不能自动批准 Closure 或把多模型共识当作实验结论。

### 修改文件

```text
frontend/src/researchCenter/LoopEngineeringWorkspace.jsx
frontend/src/researchCenter/ResearchCenter.css
frontend/src/index.css
```

### 验证结果

- 目标组件 ESLint：通过，0 error；
- Vite 正式构建：通过，共转换 2822 个模块；
- 构建产物仍有主包超过 500 kB 的既有性能提醒，不影响本批正确性；
- 本地前端 `5173` 与后端 `5001` 均返回 HTTP 200；
- Session 返回 `project_research_agent.v1`，Orchestrator 返回 `research_orchestrator_status.v3`，当前门与单进程串行 GPU 策略可正常读取；
- 当前运行环境没有可用的浏览器控制实例，因此未能完成截图级视觉检查；组件编译、CSS 打包和真实 API 联通已验证。

本批没有运行本地大模型，没有产生新的 HiddenState 数据或科学结论，也没有修改 Memo。

---

## 客户端重复启动报错修复记录 [2026-09-02 09:33]

### 根因

默认端口 `5173` 已经存在当前项目的 Vite 开发服务器时，`scripts/start_visualization.ps1` 仍使用 `--strictPort` 启动第二个进程，导致：

```text
Error: Port 5173 is already in use
```

前端进程本身正常，错误来自启动器缺少重复启动识别，不是 React、3D 或 AI 自动研发组件的编译错误。

### 修复

1. 启动前检查目标端口的监听进程；
2. 读取监听进程 PID、可执行文件和命令行；
3. 如果命令行指向当前仓库的 `frontend/node_modules/vite/bin/vite.js`，判定为已经运行的 AI2050 客户端，直接复用地址并以退出码 `0` 返回；
4. 如果端口由其他程序占用，不结束对方进程，明确显示 PID、命令行和 `-Port 5174` 换端口示例；
5. `build` 和 `lint` 模式不受端口检查影响。

### 验证

- 空闲端口 `5174`：Vite 正常启动，HTTP 200；
- 同一项目重复启动 `5174`：正确识别并复用，退出码 0；
- 已运行的默认端口 `5173`：正确识别 PID 并复用，退出码 0；
- 其他程序占用端口：返回占用者 PID 和命令，不误杀进程；
- `start_visualization.ps1 -Mode build`：通过，共转换 2822 个模块；
- 构建仍只有既有的大包体积提醒。

本次只修改客户端启动器和本记录，没有修改 3D 研究逻辑、模型测试数据或 Memo。

---

## AI 自动研发整屏滚动动画优化记录 [2026-09-03 21:44]

### 修改目标

3D 研究空间与 AI 自动研发工作区之间的滚动改为“一次输入、一次完整切换”，不再停留在两个工作区中间，也不因触控板连续事件重复启动动画。

### 已完成

1. 用固定 `640ms`、四次缓出的 `requestAnimationFrame` 动画替代容易被中断的原生 `scrollIntoView({ behavior: 'smooth' })`；
2. 动画始终以目标工作区真实 `offsetTop` 为终点，最后一帧再次校准位置；
3. 增加动画锁：动画进行期间拦截后续滚轮事件，避免连续叠加、反向打断或停在半屏；
4. 使用非被动的捕获阶段滚轮监听，触发整屏切换时阻止 3D Canvas 同时缩放；
5. 检测鼠标下方的内部滚动容器。任务时间线、线程列表、运行检查器、日志、代码、测试结果和文本框仍可独立滚动；只有内部容器到达边界后，才触发工作区切换；
6. 两个整屏工作区统一使用 `scroll-snap-type: y mandatory` 和 `scroll-snap-stop: always`，为触摸、滚动条和键盘产生的非动画滚动提供最终吸附；
7. 动画期间临时关闭 CSS scroll snap，防止吸附规则干扰逐帧动画；完成后恢复；
8. 尊重 `prefers-reduced-motion`，系统要求减少动画时直接定位到完整目标屏幕；
9. 组件卸载时取消未完成的动画帧并解除滚动锁。

### 验证

- Vite 正式构建通过，共转换 2822 个模块；
- 开发服务器上的 `src/App.jsx` 返回 HTTP 200，并包含动画时长、滚动锁和内部边界检测逻辑；
- `git diff --check` 未发现本次修改新增的空白错误；
- ESLint 对 `App.jsx` 的新增代码没有报告错误；该文件仍有修改前已存在的未使用状态等历史 lint 项；
- 构建仍只有既有的大包体积提醒。

本次只修改工作区滚动控制、对应 CSS 和本记录，没有修改 3D 模型数据、AI 研发后端、模型测试结果或 Memo。
