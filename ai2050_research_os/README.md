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
