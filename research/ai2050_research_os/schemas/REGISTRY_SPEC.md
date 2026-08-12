# 账本字段与状态规范

## 一、唯一事实源

| 文件 | 主键 | 作用 |
|---|---|---|
| project.json | project_id | 全局状态和闭合层级 |
| campaigns.json | id | 战役和工作包 |
| hypotheses.json | id | 候选机制生死合同 |
| puzzles.json | id | 关键拼图依赖与进度 |
| tests.json | id | 测试电池 |
| evidence.json | id | 不可变证据条目 |
| phases.json | phase | 执行批次索引 |
| decisions.json | id | 授权与停止裁决 |

## 二、合法状态

### 战役

`draft`、`active`、`blocked`、`completed`、`failed`、`archived`

### 工作包

`pending`、`active`、`blocked`、`completed`、`failed`、`cancelled`

### 候选机制

`candidate`、`discovery_supported`、`confirmation_supported`、`local_survivor`、`global_survivor`、`bounded_rejected`、`abstain`、`closed`

### 关键拼图

`not_started`、`instrument_only`、`partial`、`blocked`、`resolved_within_scope`、`closed_negative`、`gate_closed`

### 测试

`planned`、`preregistering`、`ready`、`running`、`auditing`、`passed`、`failed`、`blocked`、`cancelled`

### 检查项

`pending`、`partial`、`done`、`blocked`、`not_applicable`

### Phase

`draft`、`preregistered`、`running`、`auditing`、`adjudicated`、`invalid`、`censored`、`archived`

## 三、引用规则

- `campaign_id` 必须指向现有战役；历史条目可用保留值 `LEGACY`；
- `evidence_refs` 必须指向 `evidence.json`；
- `hypothesis_ids`、`puzzle_ids`、`test_battery_ids` 必须存在；
- 拼图依赖和测试前置依赖必须无环；
- `current_closure_level` 与 `target_closure_level` 必须在 0–8；
- 证据的闭合层级不能高于其实际完成链；
- 被限定否决的候选必须填写死亡条件和重开条件。

## 四、进度算法

拼图进度只用于工程看板：

\[
progress=
\frac{1.0\times done+0.5\times partial}
{done+partial+pending+blocked}.
\]

`not_applicable` 不进入分母。该比例不是“破解语言机制的百分比”，必须与闭合层级和状态同时展示。

## 五、生成文件

`generated/` 全部由 `researchctl.py build` 生成。任何手工修改都会在下一次生成中被覆盖。

