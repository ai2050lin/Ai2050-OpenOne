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

## 0.1 v2.1 修正判断

参考 `AGI_GPT5_MEMO_SUMMARY.md` 中 Phase195-278 的推进历史，原 v2 方案整体方向正确，但需要做四个关键修正：

```text
1. Score 不能再使用简单平均，否则会掩盖 causal / closure 的关键缺口；
2. closure 不能只作为一项软分数，必须加入硬性闭合判据；
3. family x model 矩阵必须显式处理跨模型矛盾，而不是简单汇总；
4. component / writer path 不能默认线性可加，必须加入非线性耦合审计。
```

因此 v2.1 的目标不是重做 v2，而是在 v2 主表之上增加：

```text
weighted_score
score_cap
closure_gate
cross_model_consistency
claim_registry
semantic_eval
nonlinear_coupling_audit
prediction_validation
```

原则：

```text
图谱评分是工程完成度；
闭合是后置硬判据；
因果结论必须登记为 claim；
跨模型矛盾必须保留为机制差异，而不是平均掉；
预测成功率必须成为理论是否成立的核心验证。
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

v2.1 增加模式族完备性检验：

```text
open_set_tasks -> auto_label_family -> unknown / mixed / known
```

判定规则：

```text
如果 unknown_family_rate > 5%，九大族不能视为冻结；
如果 mixed_family_rate > 15%，需要检查模式族是否过粗或存在强耦合；
新增 family 必须有独立触发条件、失败模式和路径签名差异。
```

当前九大族可以作为工作分类，但不能直接当作已证明的语言模式完备划分。

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
semantic_eval
cross_model_consistency
nonlinear_coupling
claim_id
evidence_level
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

v2.1 的机制主张记录：

$$
Claim(c)
=
[
ClaimID,
Scope,
EvidenceLevel,
PositiveEvidence,
NegativeEvidence,
CounterExamples,
NextTest
]
$$

其中：

```text
Scope: 适用的 family / mode / model / layer window；
EvidenceLevel: L0-L8 证据等级；
PositiveEvidence: 支持该机制主张的数据文件；
NegativeEvidence: 反例或跨模型冲突；
CounterExamples: 失败样本；
NextTest: 下一步验证脚本或 batch。
```

没有进入 claim registry 的结论，只能作为观察记录，不能作为机制结论。

## 3. 评分公式

原 v2 使用七项基础分：

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

但简单平均存在问题：

```text
behavior / readout 覆盖较高；
component_path / causal / closure 覆盖较低；
简单平均会让“行为正确但没有因果证据”的样本得到中等分；
这会误导后续 batch 选择。
```

v2.1 改为加权完成度评分：

$$
Score_{base}(x)
=
0.10B(x)
+0.10R(x)
+0.15L(x)
+0.20C(x)
+0.25I(x)
+0.10G(x)
+0.10K(x)
$$

其中：

```text
B: 行为正确性；
R: readout 竞争可解释性；
L: 层路径稳定性；
C: 组件路径归因；
I: 因果干预证据；
G: rollout / protocol gate 稳定性；
K: closure quality。
```

v2.1 增加证据封顶规则：

$$
Score_{atlas}(x)
=
\min(
Score_{base}(x),
Cap_I(x),
Cap_C(x),
Cap_K(x)
)
$$

封顶规则：

```text
如果 I(x) < 0.30，则 Score_atlas(x) <= 0.50；
如果 C(x) < 0.30，则 Score_atlas(x) <= 0.60；
如果 K(x) < 0.30，则 Score_atlas(x) <= 0.65；
如果 semantic_eval 未通过，则不能标记为 high_quality_candidate；
如果 cross_model_conflict 为 true，则不能汇总为 global mechanism，只能标记为 model_specific_mechanism。
```

跨模型一致性：

$$
X(f)
=
1
-
\frac{
N_{disagree}(f)
}{
N_{models}(f)
}
$$

其中：

```text
N_disagree(f): 在同一 family / mode 上出现方向相反或干预效果相反的模型数；
N_models(f): 参与该 family 测试的模型数。
```

如果出现 qwen3 / DS7B 支持、GLM4 反向这类现象，不能取平均，必须写入：

```text
cross_model_conflict: true
mechanism_scope: model_specific
next_test: compensation_path_audit
```

闭合硬判据：

$$
Closure(x)
=
SemanticDone(x)
\land
StopWins(x)
\land
ContinueSuppressed(x)
\land
RolloutStable(x)
$$

只有四项同时成立，才能标记为：

```text
closed
```

否则只能标记为：

```text
path_candidate_not_closed
high_quality_candidate_not_closed
mapped_partial
```

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

v2.1 解释：

```text
972 条 path signature 是图谱骨架，不是充分样本量；
9 族 x 3 模型后，平均每个 family-model 约 36 条，仍偏少；
mapped_partial 不能作为机制证据；
high_quality_candidate_not_closed 需要重新经过 semantic_eval 和 closure_gate；
跨模型矛盾样本优先进入 compensation_path_audit，而不是被 overall 平均掉。
```

## 5. 后续 Batch 计划

### Batch A: Schema Freeze

目标：

```text
冻结 v2 schema；
冻结主表 path_signature_rows；
冻结 atlas_scores；
冻结 detail-on-demand 结构；
冻结前端 client_index；
新增 claim_registry；
新增 evidence_level；
新增 cross_model_consistency；
新增 semantic_eval；
新增 nonlinear_coupling_audit；
新增 weighted_score / score_cap / closure_gate。
```

当前 Phase273 已完成第一版。

v2.1 追加目标：

```text
1. 模式族完备性检验；
2. 机制主张登记表；
3. L0-L8 证据等级；
4. 加权评分与封顶规则；
5. 闭合四条件硬判据。
```

claim registry 主表：

```text
mechanism_claim_rows.jsonl
```

每条记录至少包含：

```text
claim_id
claim_text
family_id
mode_id
model_scope
evidence_level
positive_files
negative_files
counterexamples
next_test
```

### Batch B: Full Path Fill

目标：

```text
为 972 条 path signature 补齐 layer_path 和 component_path；
优先补九大族每模型 top 缺口；
不做闭合，只补物理分布。
```

v2.1 修正：

```text
如果某个 family-model 的样本数 < 100，先扩充样本；
目标样本量为每个 family-model 至少 100 条；
成熟阶段目标为每个 family 至少 200-500 条高质量样本；
样本不足时，所有机制结论标记为 exploratory。
```

Full Path Fill 输出：

```text
layer_path_rows.jsonl
component_path_rows.jsonl
semantic_eval_rows.jsonl
cross_model_consistency_rows.jsonl
nonlinear_coupling_probe_rows.jsonl
```

### Batch C: Quality-Control Causal Audit

目标：

```text
只对 high_quality_candidate_not_closed 和 path_candidate_not_closed 做因果审计；
使用 half / mean_replace / window shrink；
强 zero/random 只作为定位工具。
```

v2.1 标准化因果干预协议：

```text
zero ablation
half scaling
mean replacement
random same norm
permutation control
cross-sample patch
multi-layer window patch
direction add
direction remove
attention+MLP combined patch
negative family control
```

方向增强：

$$
h' = h + \lambda \hat v
$$

方向移除：

$$
h' = h - (h \cdot \hat v)\hat v
$$

证据等级映射：

```text
readout probing -> L2；
layer path stable -> L3；
component attribution -> L4；
low-side-effect necessity -> L5；
controlled sufficiency -> L6；
rollout stable -> L7；
clean closure -> L8。
```

非线性叠加审计：

$$
\Delta_{linear}
=
\sum_i
\Delta M_i
$$

$$
\Delta_{actual}
=
M(h+\sum_i p_i)-M(h)
$$

如果：

$$
|\Delta_{actual}-\Delta_{linear}|>\epsilon
$$

则标记为：

```text
nonlinear_coupling: true
```

该样本不能用单 writer 线性贡献解释。

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

新增显示：

```text
Weighted Score / Score Cap
Closure Gate
Claim Registry
Evidence Ladder
Cross-model Conflict
Semantic Eval
Nonlinear Coupling
Prediction Validation
```

### Batch E: Prediction Validation

目标：

```text
从 heldout 样本中抽取未见过的 case；
用现有图谱预测 dominant_layers、component_path、readout_winner、closure_gate；
再运行真实测试比对；
统计 prediction_accuracy、calibration_error、failure_type。
```

预测公式：

$$
\hat P(x,m)
=
AtlasPredict(
family(x),
mode(x),
model(m),
prompt\_features(x)
)
$$

验证：

$$
Acc_{path}
=
\frac{
N(\hat P_{path}=P_{path})
}{
N_{heldout}
}
$$

只有前向预测显著高于随机基线，图谱才开始具备理论解释力。

### Batch F: Semantic Evaluator and Closure Recheck

目标：

```text
对 high_quality_candidate_not_closed 和 closed 候选做语义复核；
从 token-level target margin 升级为 phrase-level / span-level semantic eval；
对闭合四条件重新打标；
必要时加入人工校验子集。
```

输出：

```text
semantic_eval_rows.jsonl
closure_gate_rows.jsonl
human_check_sample_rows.jsonl
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
weighted_score
score_cap_reason
cross_model_conflict
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
nonlinear_coupling
writer_set_consistency
```

### Causal Audit

展示：

```text
zero
half
mean_replace
random_same_norm
permutation control
cross-sample patch
direction add
direction remove
window shrink
side effect
strict protocol clean
evidence_level
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
semantic eval row
claim registry row
cross-model conflict row
nonlinear coupling row
raw JSON
```

### Claim Registry

展示：

```text
claim_id
claim_text
scope
evidence_level
supported_models
failed_models
positive_files
negative_files
counterexamples
next_test
```

### Prediction Validation

展示：

```text
heldout_case
predicted_path
actual_path
prediction_accuracy
calibration_error
failure_type
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
Phase274+: v2.1 weighted gap fill + claim registry + causal audit
```

目标是补齐缺口最大的字段，同时避免把弱证据误判为强机制：

```text
component_path
causal
closure_quality
semantic_eval
cross_model_consistency
nonlinear_coupling
claim_registry
prediction_validation
```

优先级：

```text
1. 先修正 schema：加入 weighted_score、score_cap、closure_gate、claim_id；
2. 再补 high_signal_low_causal 和 high_quality_candidate_not_closed；
3. 对跨模型矛盾样本优先做 compensation_path_audit；
4. 对 component_path 高但 causal 低的样本做低副作用因果审计；
5. 对 closed / high_quality 候选做 semantic_eval 与 closure_gate 复核；
6. 最后用 heldout 样本做 prediction_validation。
```

v2.1 必须落地的改进：

```text
评分：从简单平均改为加权 + 封顶；
闭合：从软分数改为四条件 AND；
跨模型：从平均矩阵改为 conflict-aware matrix；
因果：从单点 patch 改为标准化干预协议；
非线性：从线性 writer 分解改为 coupling-aware 审计；
语义：从 token margin 代理改为 span / phrase / semantic eval；
理论：从 memo 结论改为 claim registry；
验证：从回顾性补表改为前向预测检验。
```

## 8. 当前风险

1. v2 分数是工程评分，不是理论闭合；必须显示 score_cap_reason。
2. 简单平均会掩盖 causal / closure 缺口；v2.1 必须使用加权分与封顶规则。
3. 现有 component_path 覆盖集中在少量样本，导致大多数 case 仍是 mapped_partial。
4. 972 条 path signature 分摊到 9 族 × 3 模型后，每格平均样本量仍偏低。
5. case_details 文件数量多，但初始加载不应读取。
6. 小模型内部结构可能粗糙，跨模型差异不能直接外推真实语言机制。
7. 跨模型矛盾不能取平均，必须标记为 model_specific_mechanism。
8. 当前线性公式无法替代物理路径图谱，多 writer 叠加必须做 nonlinear_coupling_audit。
9. behavior score 和 token margin 仍是代理指标，必须引入 semantic_eval。
10. 没有预测验证的图谱只能说明“历史数据被整理过”，不能说明理论已经成立。

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

v2.1 的结论是：

```text
Atlas v2 是正确骨架；
但必须加入 weighted score、closure gate、claim registry、cross-model conflict、nonlinear coupling 和 prediction validation；
否则图谱会积累大量“看似中等分、实际无因果闭合”的样本。
```

下一步不应继续散点式实验，而应按 v2.1 Batch 补全图谱：

```text
先冻结 schema 和 claim registry；
再扩充样本与补物理路径；
再做低副作用因果审计；
再做语义与闭合复核；
最后做 heldout 预测验证。
```
