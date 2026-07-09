# Pattern Family Atlas v2 系统方案

记录时间：2026-07-08 18:25

## 0. 总判断

当前路线方向正确，但执行方式必须从“单 Phase 试探”升级为“图谱工程驱动”。

后续优先级：

```text
第一优先级：完成语言模式图谱的物理分布拼图；
第二优先级：在高质量候选上尝试闭合；
禁止把强干预直接解释成闭合。
```

Phase273 已把已有 v1 数据汇总为 v2：

```text
path_signature_rows: 972
atlas_scores: 36
case_details: 972
```

v2 的核心变化：

```text
用 path_signature_rows.jsonl 作为主表；
用 atlas_scores.jsonl 作为 family x model 矩阵；
用 case_details/ 做按需详情；
用 client_index.json 支持前端轻量加载。
```

## 1. 固定九大语言模式族

```text
content_knowledge
output_protocol
reasoning_constraint
syntax_structure
language_action
cross_lingual
readout_competition
state_drift
closure
```

每个模式族至少保留四类样本：

```text
base
protocol
boundary
perturbation / pressure
```

## 2. 全链路字段

每个 case-model 都应逐步补齐：

```text
behavior
readout
layer_path
component_path
causal
compensation
rollout
closure
span_alias_protocol
```

v2 当前主公式：

$$
Atlas(f,m)
=
[
Cases,
Behavior,
Readout,
LayerPath,
ComponentPath,
CausalAudit,
CompensationPath,
Rollout,
Closure
]
$$

每条路径签名：

$$
PathSignature(x,m)
=
[
State,
DominantLayers,
AttentionRoute,
MLPWrite,
Compensation,
ReadoutWinner,
ProtocolGate,
ClosureQuality
]
$$

## 3. 评分公式

v2 使用七项基础分：

$$
Score(x)
=
\frac{
B(x)
+R(x)
+L(x)
+C(x)
+I(x)
+G(x)
+K(x)
}{7}
$$

其中：

```text
B(x): behavior score
R(x): readout score
L(x): layer path score
C(x): component path score
I(x): intervention / causal score
G(x): rollout / protocol gate score
K(x): closure quality score
```

注意：这是图谱完成度评分，不是智能理论闭合公式。

## 4. 当前 v2 结果

```text
path_signatures: 972
status_counts:
  mapped_partial: 963
  path_candidate_not_closed: 7
  high_quality_candidate_not_closed: 2
```

当前最强候选：

```text
glm4 / output_protocol / explain_answer / structured_json
glm4 / closure / answer_correct / structured_json
```

这说明：

```text
行为与读出覆盖较多；
组件路径、因果审计和闭合质量字段覆盖仍不足；
真正高质量候选很少。
```

## 5. 后续 Batch 计划

### Batch A: Schema Freeze

目标：

```text
冻结 v2 schema；
冻结主表 path_signature_rows；
冻结 atlas_scores；
冻结 detail-on-demand 结构；
冻结前端 client_index。
```

当前 Phase273 已完成第一版。

### Batch B: Full Path Fill

目标：

```text
为 972 条 path signature 补齐 layer_path 和 component_path；
优先补九大族每模型 top 缺口；
不做闭合，只补物理分布。
```

### Batch C: Quality-Control Causal Audit

目标：

```text
只对 high_quality_candidate_not_closed 和 path_candidate_not_closed 做因果审计；
使用 half / mean_replace / window shrink；
强 zero/random 只作为定位工具。
```

### Batch D: Visualization Client v2

目标：

```text
Overview
Family Matrix
Path Explorer
Component View
Causal Audit
Case Detail
```

初始加载：

```text
manifest.json
client_index.json
atlas_scores.jsonl
families.jsonl
```

按需加载：

```text
case_details/{model}__{case_id}.json
```

## 6. 客户端显示方案

### Overview

显示：

```text
总 path signatures
高质量候选数量
各项平均分
最近 Phase
未完成字段比例
```

### Family Matrix

矩阵：

```text
family x model
```

每格显示：

```text
overall
physical path
component path
causal
closure
```

### Path Explorer

展示：

```text
prompt -> state -> dominant layers -> component route -> readout -> protocol gate -> closure quality
```

### Component View

展示：

```text
attention_route_score
mlp_write_score
dominant_layers
compensation_score
```

### Causal Audit

展示：

```text
zero
half
mean_replace
random_same_norm
window shrink
side effect
strict protocol clean
```

### Case Detail

展示：

```text
raw prompt
target
path signature
behavior row
readout row
layer summary
component summary
causal rows
closure fiber rows
span protocol rows
raw JSON
```

## 7. 关键改进

后续不应再使用：

```text
做一个小测试 -> 写一个 Phase -> 临时决定下一步
```

应改为：

```text
统一 schema -> 批量补字段 -> 自动评分 -> 前端查看缺口 -> 只对缺口开专项 Phase
```

Phase273 已完成系统骨架，接下来应进入：

```text
Phase274: v2 full-path gap fill batch
```

目标是补齐缺口最大的字段：

```text
component_path
causal
closure_quality
```

优先级：

```text
1. 先补 path_candidate_not_closed 和 high_quality_candidate_not_closed；
2. 再补每个 family-model 的 top 缺口；
3. 最后补低分族的物理分布。
```

## 8. 当前风险

1. v2 分数是工程评分，不是理论闭合。
2. 现有 component_path 覆盖集中在少量样本，导致大多数 case 仍是 mapped_partial。
3. case_details 文件数量多，但初始加载不应读取。
4. 小模型内部结构可能粗糙，跨模型差异不能直接外推真实语言机制。
5. 当前线性公式无法替代物理路径图谱。

## 9. 结论

Pattern Family Atlas v2 是当前研究路线的必要升级。

它把研究从实验日志推进到可查询数据库：

```text
每个样本有主表；
每个模式族有矩阵；
每个候选有详情；
每个缺口可以被前端定位；
每个后续 Phase 都服务于同一个系统。
```

下一步不应继续散点式实验，而应按 Batch 补全 v2 图谱。
