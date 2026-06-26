# AGI Research Memo

> 本文档记录AGI研究的进展、问题分析和下一步行动

## Phase 601: Source-Resolved Final Attention Acceptance Atlas 源词元级最后层注意力接受图谱 [2026-06-24 11:17]

### 本阶段目标

根据附件对 Phase600 的分析，先判断其是否正确，再继续推进任务。

附件中正确部分：

```text
1. Phase600 的方向正确，已经从 projection strength 转向 trajectory acceptance。
2. DS7B 中人工修复虽然能制造局部候选投影，但没有形成 natural correct trajectory。
3. “最后层是 trajectory consistency filter”只能作为工作性描述，尚不是闭合理论。
4. 下一步必须把 attention pattern 粗差异拆成 source token 级图谱。
```

需要保持谨慎的部分：

```text
natural correct prompt 与 base prompt 不是纯净反事实；
source-token attention shift 只能说明最后层注意力源选择不同，
不能单独证明它就是完整 value gate。
```

本阶段因此进入：

```text
Source-Resolved Final Attention Acceptance Atlas
```

核心目标：

```text
把 Phase600 的 attention pattern 粗差异拆成语义源词元组，
判断 natural correct 与 artificial repair 的最后层注意力源选择到底差在哪里。
```

### 执行命令

```bash
python -m py_compile \
  tests/glm5/phase601_source_resolved_final_attention_atlas.py \
  tests/glm5/phase601_source_resolved_final_attention_atlas_summary.py

python tests/glm5/phase601_source_resolved_final_attention_atlas.py qwen3 \
  --smoke \
  --output-dir results/glm5_phase601_source_resolved_final_attention_atlas \
  --hard-exit-after-model

python tests/glm5/phase601_source_resolved_final_attention_atlas.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase601_source_resolved_final_attention_atlas \
  --hard-exit-after-model

python tests/glm5/phase601_source_resolved_final_attention_atlas.py glm4 \
  --confirm \
  --output-dir results/glm5_phase601_source_resolved_final_attention_atlas \
  --hard-exit-after-model

python tests/glm5/phase601_source_resolved_final_attention_atlas.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase601_source_resolved_final_attention_atlas \
  --hard-exit-after-model

python tests/glm5/phase601_source_resolved_final_attention_atlas_summary.py
```

### 脚本与结果文件

- 主脚本：`tests/glm5/phase601_source_resolved_final_attention_atlas.py`
- 汇总脚本：`tests/glm5/phase601_source_resolved_final_attention_atlas_summary.py`
- Qwen3 结果：`results/glm5_phase601_source_resolved_final_attention_atlas/phase601_qwen3_source_resolved_final_attention_atlas_confirm.json`
- GLM4 结果：`results/glm5_phase601_source_resolved_final_attention_atlas/phase601_glm4_source_resolved_final_attention_atlas_confirm.json`
- DS7B 结果：`results/glm5_phase601_source_resolved_final_attention_atlas/phase601_deepseek7b_source_resolved_final_attention_atlas_confirm.json`
- 跨模型汇总：`results/glm5_phase601_source_resolved_final_attention_atlas/phase601_cross_model_summary.md`

### 测试原理

Phase600 只知道：

```text
natural correct 的最后层 attention pattern 与 artificial repair 不同。
```

Phase601 把每个 prompt 的 source token 分成：

```text
rule_relation
rule_value
object
category_first
query_relation
query_category
prompt_last
punct_newline
other
```

对每个 target position 记录最后层 attention mass：

```text
natural_correct - base
natural_wrong - base
artificial_repair - base
artificial_random - base
artificial_wrong - base
```

并计算：

```text
natural_correct - artificial_repair
```

用来找：

```text
自然正确轨迹独有的 source attention shift。
```

### 测试范围

```text
models = qwen3, glm4, deepseek7b
confirm cases/model = 128
target_only = true
alpha = 2.0
Qwen3 target cases = 11
GLM4 target cases = 22
DS7B target cases = 49
```

DS7B watched nodes：

```text
rule_value L26
prompt_last L26
query_relation L19
```

### 客观结果

#### Qwen3

```text
target_cases_seen = 11
probe_layer = L35
```

最大 attention delta：

```text
query_category L32 natural_correct:
  punct_newline +0.0453
  query_category +0.0218
  object +0.0149
  other -0.0411

query_category L32 artificial_repair:
  query_category -0.0230
  other +0.0246
```

natural correct - artificial repair：

```text
query_category L32:
  other -0.0657
  punct_newline +0.0466
  query_category +0.0448
  object +0.0149
  L1 = 0.1780
```

Qwen3 结论：

```text
natural correct 与 artificial repair 的 source attention shift 不同；
natural correct 更偏向 punctuation/newline、query_category、object，
artificial repair 反而增加 other。
```

但 Qwen3 的 target cases 仍少，作为对照。

#### GLM4

```text
target_cases_seen = 22
probe_layer = L39
```

最大 attention delta：

```text
prompt_last L38 natural_correct:
  prompt_last +0.0373
  object +0.0109
  query_category +0.0096
  punct_newline +0.0065
  other -0.0731

prompt_last L38 artificial_repair:
  punct_newline -0.0389
  prompt_last -0.0175
  other +0.0069
```

natural correct - artificial repair：

```text
prompt_last L38:
  other -0.0800
  prompt_last +0.0547
  punct_newline +0.0455
  object +0.0092
  L1 = 0.2070
```

GLM4 结论：

```text
natural correct 会把注意力从 other 转向 prompt_last / punctuation / object 等结构源；
artificial repair 没有复现，甚至方向相反。
```

#### DS7B

```text
target_cases_seen = 49
probe_layer = L27
```

最大 attention delta：

```text
prompt_last L26 natural_correct:
  prompt_last +0.0269
  punct_newline +0.0481
  object +0.0174
  category_first +0.0150
  query_category +0.0125
  other -0.0865

prompt_last L26 artificial_repair:
  prompt_last -0.0503
  punct_newline -0.0450
  other +0.0497

rule_value L26 natural_correct:
  rule_relation +0.0277
  category_first +0.0465
  punct_newline +0.0113
  other -0.0724

rule_value L26 artificial_repair:
  rule_value +0.0385
  punct_newline -0.0109
  other -0.0390

query_relation L19 natural_correct:
  object +0.0248
  query_relation +0.0163
  query_category +0.0092
  punct_newline +0.0315
  other -0.0593

query_relation L19 artificial_repair:
  all semantic groups near 0
  other +0.0025
```

natural correct - artificial repair：

```text
prompt_last L26:
  prompt_last +0.0772
  punct_newline +0.0931
  other -0.1362
  object +0.0177
  category_first +0.0138
  query_category +0.0118
  L1 = 0.3654

rule_value L26:
  category_first +0.0470
  rule_relation +0.0267
  punct_newline +0.0222
  other -0.0334
  rule_value -0.0404
  L1 = 0.1697

query_relation L19:
  object +0.0255
  query_relation +0.0178
  query_category +0.0090
  punct_newline +0.0300
  other -0.0618
  L1 = 0.1536
```

DS7B 最可靠事实：

```text
1. natural correct 不只是改变 attention 总量，而是把注意力从 other 移向结构源。
2. prompt_last L26 中，natural correct 明显增加 prompt_last / punct_newline / object / category / query_category；
   artificial repair 则减少 prompt_last / punct_newline，并增加 other。
3. rule_value L26 中，natural correct 增加 rule_relation 与 category_first；
   artificial repair 增加 rule_value 自身，但缺少 rule_relation/category_first 组合。
4. query_relation L19 中，natural correct 增加 object/query_relation/query_category/punct_newline；
   artificial repair 几乎没有复制这些源选择。
```

### 当前最可靠客观事实

1. **Phase600 的附件分析基本正确。**

```text
artificial repair 没有进入 natural correct trajectory。
Phase601 进一步说明：其中一个具体差异是 source attention selection。
```

2. **自然正确轨迹有结构化注意力源选择。**

DS7B 中 natural correct 多次表现为：

```text
semantic / structural source mass 增加
other mass 减少
```

3. **人工 repair 与自然正确的 source shift 方向常常相反。**

DS7B prompt_last：

```text
natural_correct:
  prompt_last +0.0269
  punct_newline +0.0481
  other -0.0865

artificial_repair:
  prompt_last -0.0503
  punct_newline -0.0450
  other +0.0497
```

这非常关键：

```text
人工 repair 不是“弱自然正确轨迹”，而是在关键注意力源上走了相反方向。
```

4. **rule_value 的 missing factor 可能不是看 value 本身，而是看 category / relation 组合。**

DS7B rule_value：

```text
natural_correct:
  rule_relation +0.0277
  category_first +0.0465

artificial_repair:
  rule_value +0.0385
  category_first -0.0005
```

这提示：

```text
最后层接受值候选时，可能需要 relation/category context，而不是只增强 value token。
```

5. **source-token atlas 支持“图谱路线”。**

当前不能只找一个 repair vector；
必须记录：

```text
位置 -> 层 -> 组件 -> source attention group -> downstream projection -> readout
```

### 理论进展

Phase601 把机制链进一步细化为：

```text
candidate projection generation
-> residual entry
-> final attention source selection
-> final MLP/residual compensation
-> final norm/readout
```

新增的是：

```text
final attention source selection
```

更具体地说：

```text
natural correct trajectory 会在最后层重新选择信息源，
把注意力从 diffuse other 转向特定结构源；
artificial repair 只改 MLP input，不会自动触发这种源选择，
所以即使局部 projection 强，也无法被最后层当作合法轨迹。
```

这使 value gate 的定义更具体：

```text
value gate = candidate projection + source attention selection + component compensation + readout competition
```

### 硬伤与瓶颈

1. **source group 仍然粗糙。**

```text
other 仍然很大，说明分组还没有完全覆盖真实 source。
```

需要继续细分：

```text
rule line tokens
all value tokens
all category tokens
answer prefix
Question/Answer markers
newline clusters
period tokens
```

2. **自然 prompt 仍不是纯反事实。**

natural_correct 与 base/repair prompt 结构不同，因此 source attention shift 可能混入模板/长度因素。

3. **本轮只测 atlas，没有做双因素因果 patch。**

它证明了 missing factor 候选：

```text
source attention selection
```

但还没有证明：

```text
修补 source attention selection 能打开最终生成。
```

4. **attention patch 本身实现难度更高。**

需要谨慎选择：

```text
patch attention weights
patch attention output
patch q/k/v source contribution
patch post-attn residual
```

不能盲目扩大搜索。

### 下一步任务

Phase602 应进入：

```text
Attention-Source Factor Causal Patch
```

核心目标：

```text
测试 Phase601 找到的 source attention shift 是否是因果缺失项。
```

建议方案：

```text
1. 以 DS7B 为主，Qwen3/GLM4 做小确认。
2. 优先测试 DS7B prompt_last L26 与 rule_value L26。
3. 不直接 patch 全 attention matrix，先做更稳的 attention output source-component patch：
   - 记录 natural_correct 的 L27 attention output
   - 记录 artificial_repair 的 L27 attention output
   - 只替换/添加 attention output 中对应 source-shift 的 residual effect
4. 组合测试：
   A. MLP input repair only
   B. attention source effect only
   C. MLP input repair + attention source effect
   D. random same norm control
5. 指标：
   - full candidate logprob margin
   - switch
   - layer_out cos_to_natural_correct
   - final_norm_output cos_to_natural_correct
```

判据：

```text
如果 C 明显强于 A/B/D，
说明 source attention factor 与 candidate projection 需要共同出现。

如果 C 仍失败，
说明 missing factor 不是单层 source attention，
而可能在 MLP gate compensation / multi-token trajectory / final norm scale。
```

## Phase 602: Attention-Source Factor Causal Patch 注意力源因子因果修补 [2026-06-24 11:47]

### 本阶段目标

根据附件对 Phase601 的分析，先判断其是否正确，再继续完成任务。

附件中正确部分：

```text
1. Phase601 的方向正确：它把 trajectory acceptance factor 具体化为 source attention selection 候选机制。
2. artificial repair 不是 natural correct 的弱版本，而是在关键 source token 选择上走了不同甚至相反路径。
3. rule_value L26 的结果提示 value gate 不是只看 value token，而可能需要 category/relation binding。
4. Phase601 仍只是 atlas，不是 causal repair；下一步必须做注意力源因子的因果 patch。
```

本阶段因此进入：

```text
Attention-Source Factor Causal Patch
```

核心目标：

```text
测试 Phase601 找到的 source attention factor 是否是因果缺失项。
```

### 执行命令

```bash
python -m py_compile \
  tests/glm5/phase602_attention_source_factor_causal_patch.py \
  tests/glm5/phase602_attention_source_factor_causal_patch_summary.py

python tests/glm5/phase602_attention_source_factor_causal_patch.py qwen3 \
  --smoke \
  --output-dir results/glm5_phase602_attention_source_factor_causal_patch \
  --hard-exit-after-model

python tests/glm5/phase602_attention_source_factor_causal_patch.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase602_attention_source_factor_causal_patch \
  --hard-exit-after-model

python tests/glm5/phase602_attention_source_factor_causal_patch.py glm4 \
  --confirm \
  --output-dir results/glm5_phase602_attention_source_factor_causal_patch \
  --hard-exit-after-model

python tests/glm5/phase602_attention_source_factor_causal_patch.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase602_attention_source_factor_causal_patch \
  --hard-exit-after-model

python tests/glm5/phase602_attention_source_factor_causal_patch_summary.py
```

### 脚本与结果文件

- 主脚本：`tests/glm5/phase602_attention_source_factor_causal_patch.py`
- 汇总脚本：`tests/glm5/phase602_attention_source_factor_causal_patch_summary.py`
- Qwen3 结果：`results/glm5_phase602_attention_source_factor_causal_patch/phase602_qwen3_attention_source_factor_causal_patch_confirm.json`
- GLM4 结果：`results/glm5_phase602_attention_source_factor_causal_patch/phase602_glm4_attention_source_factor_causal_patch_confirm.json`
- DS7B 结果：`results/glm5_phase602_attention_source_factor_causal_patch/phase602_deepseek7b_attention_source_factor_causal_patch_confirm.json`
- 跨模型汇总：`results/glm5_phase602_attention_source_factor_causal_patch/phase602_cross_model_summary.md`

### 测试原理

Phase601 证明：

```text
natural correct 与 artificial repair 的最后层 source attention selection 不同。
```

Phase602 不直接 patch 全 attention matrix，而使用较稳的 attention output effect patch：

```text
natural_attention_effect = attn_out(natural_correct) - attn_out(base)
```

在 base prompt 上测试六类组合：

```text
1. mlp_repair_only
2. attn_effect_only
3. mlp_plus_attn_effect
4. attn_random
5. mlp_plus_attn_random
6. mlp_random_plus_attn_effect
```

指标：

```text
full candidate logprob margin
switch
generated_down_projection
attn_delta_projection
final_norm_projection
final_norm_cos_to_natural
positive_full_margin_rate
```

判据：

```text
如果 mlp_plus_attn_effect 明显强于 mlp_repair_only / attn_effect_only / random control，
说明 source attention factor 与 candidate projection 需要共同出现。

如果 final_norm_cos_to_natural 提升但 full_margin/switch 不提升，
说明 attention effect 是轨迹相似度因素之一，但不是充分读出因子。
```

### 测试范围

```text
models = qwen3, glm4, deepseek7b
confirm cases/model = 96
target_only = true
alpha = 2.0
attn_scale = 1.0
Qwen3 target cases = 7
GLM4 target cases = 13
DS7B target cases = 37
```

DS7B watched nodes：

```text
rule_value L26
prompt_last L26
query_relation L19
```

### 客观结果

#### Qwen3

```text
target_cases_seen = 7
probe_layer = L35
```

主要结果：

```text
query_category L32:
  mlp_repair_only:
    switch 2/7
    full_margin_gain +0.036
    final_norm_cos_to_natural 0.403

  mlp_plus_attn_effect:
    switch 2/7
    full_margin_gain +0.036
    final_norm_cos_to_natural 0.511

  mlp_plus_attn_random:
    switch 2/7
    full_margin_gain +0.036
    final_norm_cos_to_natural 0.392

prompt_last L32:
  mlp_repair_only:
    switch 1/7
    full_margin_gain +0.071
    final_norm_cos_to_natural 0.437

  mlp_plus_attn_effect:
    switch 1/7
    full_margin_gain +0.071
    final_norm_cos_to_natural 0.594
```

Qwen3 结论：

```text
attention effect 能提高 final_norm_cos_to_natural，
但没有提高 switch 或 full_margin_gain。
```

#### GLM4

```text
target_cases_seen = 13
probe_layer = L39
```

主要结果：

```text
prompt_last L38:
  mlp_repair_only:
    switch 0/13
    full_margin_gain +0.005
    final_norm_cos_to_natural 0.864

  mlp_plus_attn_effect:
    switch 0/13
    full_margin_gain +0.005
    final_norm_cos_to_natural 0.871

  mlp_plus_attn_random:
    switch 0/13
    full_margin_gain +0.005
    final_norm_cos_to_natural 0.862
```

GLM4 结论：

```text
attention effect 对 final_norm_cos_to_natural 有极小提升，
但没有任何 switch 改善；
GLM4 的最终生成闭合仍失败。
```

#### DS7B

```text
target_cases_seen = 37
probe_layer = L27
```

rule_value L26：

```text
mlp_repair_only:
  switch 0/37
  full_margin_gain -0.001
  mlp_down_projection +1.150
  final_norm_projection +0.222
  final_norm_cos_to_natural 0.420

attn_effect_only:
  switch 0/37
  full_margin_gain +0.000
  attn_delta_projection -3.019
  final_norm_projection -0.019
  final_norm_cos_to_natural 0.340

mlp_plus_attn_effect:
  switch 0/37
  full_margin_gain -0.001
  mlp_down_projection +1.150
  attn_delta_projection -3.019
  final_norm_projection +0.263
  final_norm_cos_to_natural 0.554

mlp_plus_attn_random:
  switch 0/37
  full_margin_gain -0.001
  final_norm_cos_to_natural 0.291
```

prompt_last L26：

```text
mlp_repair_only:
  switch 0/37
  full_margin_gain -0.018
  mlp_down_projection +3.084
  final_norm_projection +0.194
  final_norm_cos_to_natural 0.530

attn_effect_only:
  switch 0/37
  full_margin_gain +0.000
  attn_delta_projection +0.022
  final_norm_projection -0.001
  final_norm_cos_to_natural 0.180

mlp_plus_attn_effect:
  switch 0/37
  full_margin_gain -0.018
  final_norm_projection +0.181
  final_norm_cos_to_natural 0.541

mlp_plus_attn_random:
  switch 0/37
  full_margin_gain -0.018
  final_norm_cos_to_natural 0.524
```

query_relation L19：

```text
mlp_repair_only:
  switch 0/37
  full_margin_gain -0.013
  final_norm_cos_to_natural 0.225

attn_effect_only:
  switch 0/37
  full_margin_gain +0.000
  attn_delta_projection +0.923
  final_norm_cos_to_natural 0.167

mlp_plus_attn_effect:
  switch 0/37
  full_margin_gain -0.013
  final_norm_cos_to_natural 0.278
```

DS7B 最可靠事实：

```text
1. attention effect alone 不改变 full candidate margin，也没有 switch。
2. mlp_plus_attn_effect 能提高部分 final_norm_cos_to_natural。
   例如 rule_value L26: 0.420 -> 0.554。
3. 但 mlp_plus_attn_effect 没有提高 full_margin，也没有任何 switch。
4. random attention control 远弱于 natural attention effect 的 cos 改善，
   说明 attention effect 不是完全无效；
   但它仍不是充分因果因子。
```

### 当前最可靠客观事实

1. **Phase601 的附件分析基本正确。**

```text
source attention selection 是真实差异之一。
```

2. **source attention factor 可以提高轨迹相似度，但不能打开最终读出。**

DS7B rule_value：

```text
mlp_repair_only final_norm_cos_to_natural 0.420
mlp_plus_attn_effect final_norm_cos_to_natural 0.554
```

但：

```text
switch 仍为 0/37
full_margin_gain 仍为 -0.001
```

3. **attention effect 不是充分 value gate。**

```text
attn_effect_only:
  switch 0
  full_margin_gain 0
```

4. **MLP projection + attention source effect 仍不够。**

这说明：

```text
value gate 至少还缺少 MLP gate compensation / post-attn residual alignment / final norm scale / multi-token trajectory 中的某些因素。
```

5. **当前路线继续支持“轨迹图谱”而不是“单点 patch”。**

Phase602 不是失败回退，而是把一个候选缺失项降级：

```text
source attention selection = necessary-looking factor, not sufficient factor。
```

### 理论进展

当前 value gate 可进一步写成：

```text
candidate projection
+ source attention selection
+ component compensation
+ residual trajectory alignment
+ final norm/readout acceptance
```

Phase602 的修正是：

```text
source attention selection 不是完整 gate，
它只提高轨迹相似度的一部分。
```

也就是说，最终读出不是只需要：

```text
看对 source
```

还需要：

```text
看对 source 后，后续 MLP/residual/norm 形成正确补偿轨迹。
```

### 硬伤与瓶颈

1. **attention patch 是 full attn_out effect，不是精确 source-component patch。**

本轮为了稳健，没有直接改 attention matrix 或拆 v-source contribution。
因此它验证的是：

```text
natural attention output effect
```

而不是严格的：

```text
specific source-group causal effect
```

2. **full_margin 没有提升，说明当前补丁仍未进入 readout competition。**

即使 final_norm_cos_to_natural 提升，也没有改变最终候选排序。

3. **target cases 仍有限。**

```text
Qwen3 = 7
GLM4 = 13
DS7B = 37
```

主结论仍以 DS7B 为准。

4. **下一步不应继续盲目扩大 patch 空间。**

需要先定位：

```text
attention effect 后为什么 full_margin 不动？
```

### 下一步任务

Phase603 应进入：

```text
Post-Attention MLP Compensation Audit
```

核心目标：

```text
测试缺失项是否在 final layer 的 MLP compensation，而不是 attention source 本身。
```

建议方案：

```text
1. 以 DS7B 为主，Qwen3/GLM4 小确认。
2. 继续使用 rule_value L26 与 prompt_last L26。
3. 比较 natural_correct、mlp_repair_only、mlp_plus_attn_effect 的：
   - post-attn residual
   - MLP input
   - gate/up/z/down
   - MLP output
   - layer_out
   - final_norm_output
4. 不做大范围搜索，先做诊断：
   attention effect 已经提高 final_norm_cos，
   但为什么 MLP compensation 没有形成正确 full_margin。
5. 如果发现 MLP output 缺失自然补偿，再测试：
   mlp_repair + attn_effect + final-MLP-output compensation
```

判据：

```text
如果 natural_correct 的 final MLP output 与人工轨迹系统不同，
且补上 MLP compensation 后 full_margin 改善，
说明 value gate 的下一缺失项在 post-attn MLP compensation。

如果补上 MLP compensation 仍失败，
则瓶颈可能在 final norm / lm_head readout acceptance 或多 token 生成轨迹。
```

## Phase 603: Post-Attention MLP Compensation Audit 注意力后 MLP 补偿审计 [2026-06-24 12:54]

### 本阶段目标

根据附件对 Phase 602 的分析，先判断其正确性，再继续推进任务。

附件判断基本正确：

```text
Phase 602 不是 attention source factor 闭合。
Phase 602 的真实含义是：
attention effect 可以提高人工轨迹对 natural_correct 的相似度，
但不能打开 full candidate margin，也不能造成 switch。
```

因此下一步不能继续盲目扩大 patch 空间，而应直接审计：

```text
attention effect 已经提高 final_norm_cos 后，
为什么最终候选竞争仍不动？
```

本阶段测试假设：

```text
缺失项可能在 final layer 的 post-attention MLP compensation。
如果补上 natural_correct 的 final MLP output effect 后，
full_margin 或 switch 明显改善，
说明 value gate 的缺失项包含注意力后的 MLP 补偿。

如果仍然失败，
说明瓶颈继续后移到 final norm / lm_head readout acceptance，
或需要多 token 生成轨迹级别解释。
```

### 生成脚本

```text
tests/glm5/phase603_post_attention_mlp_compensation_audit.py
tests/glm5/phase603_post_attention_mlp_compensation_audit_summary.py
```

### 执行命令

```bash
python tests/glm5/phase603_post_attention_mlp_compensation_audit.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase603_post_attention_mlp_compensation_audit \
  --hard-exit-after-model

python tests/glm5/phase603_post_attention_mlp_compensation_audit.py glm4 \
  --confirm \
  --output-dir results/glm5_phase603_post_attention_mlp_compensation_audit \
  --hard-exit-after-model

python tests/glm5/phase603_post_attention_mlp_compensation_audit.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase603_post_attention_mlp_compensation_audit \
  --hard-exit-after-model

python tests/glm5/phase603_post_attention_mlp_compensation_audit_summary.py

python -m py_compile \
  tests/glm5/phase603_post_attention_mlp_compensation_audit.py \
  tests/glm5/phase603_post_attention_mlp_compensation_audit_summary.py
```

### 结果文件

```text
results/glm5_phase603_post_attention_mlp_compensation_audit/phase603_qwen3_post_attention_mlp_compensation_audit_confirm.json
results/glm5_phase603_post_attention_mlp_compensation_audit/phase603_glm4_post_attention_mlp_compensation_audit_confirm.json
results/glm5_phase603_post_attention_mlp_compensation_audit/phase603_deepseek7b_post_attention_mlp_compensation_audit_confirm.json
results/glm5_phase603_post_attention_mlp_compensation_audit/phase603_cross_model_summary.md
```

### 测试范围

```text
models = qwen3, glm4, deepseek7b
cases/model = 96
qwen3 target rows = 7
glm4 target rows = 13
deepseek7b target rows = 37
alpha = 2.0
attn_scale = 1.0
mlpout_scale = 1.0
```

审计部件：

```text
mlp_input
gate
up
z
down
mlp_out
layer_out
final_norm_output
```

对照轨迹：

```text
natural_correct
mlp_repair_only
mlp_plus_attn_effect
mlp_plus_attn_random
mlp_random_plus_attn_effect
```

补丁组合：

```text
mlp_repair_only
mlp_plus_attn_effect
mlpout_effect_only
mlp_plus_mlpout_effect
mlp_plus_attn_plus_mlpout_effect
```

### 客观结果

#### Qwen3

```text
target rows = 7
best diagnostic:
  prompt_last L32 mlp_plus_attn_effect mlp_out cos_to_natural = 0.691
  prompt_last L32 mlp_plus_attn_effect down cos_to_natural = 0.690
  query_category L32 mlp_plus_attn_effect down cos_to_natural = 0.667
```

但是补上 MLP output effect 没有新增因果收益：

```text
query_category L32:
  mlp_repair_only switch = 2/7, full_margin_gain = 0.036
  mlp_plus_attn_effect switch = 2/7, full_margin_gain = 0.036
  mlp_plus_mlpout_effect switch = 2/7, full_margin_gain = 0.036
  mlp_plus_attn_plus_mlpout_effect switch = 2/7, full_margin_gain = 0.036

prompt_last L32:
  mlp_repair_only switch = 1/7, full_margin_gain = 0.071
  mlp_plus_attn_effect switch = 1/7, full_margin_gain = 0.071
  mlp_plus_mlpout_effect switch = 1/7, full_margin_gain = 0.071
  mlp_plus_attn_plus_mlpout_effect switch = 1/7, full_margin_gain = 0.071
```

#### GLM4

```text
target rows = 13
best diagnostic:
  prompt_last L39 mlp_input cos_to_natural = 1.000
  prompt_last L39 gate/up cos_to_natural = 1.000
  prompt_last L39 mlp_out cos_to_natural = 0.962
```

但补丁没有造成 switch：

```text
prompt_last L38:
  mlp_repair_only switch = 0/13, full_margin_gain = 0.005
  mlp_plus_attn_effect switch = 0/13, full_margin_gain = 0.005
  mlp_plus_mlpout_effect switch = 0/13, full_margin_gain = 0.005
  mlp_plus_attn_plus_mlpout_effect switch = 0/13, full_margin_gain = 0.005

prompt_last L39:
  all repair combinations switch = 0/13, full_margin_gain = 0.000
```

#### DS7B

```text
target rows = 37
best diagnostic:
  rule_value L26 mlp_plus_attn_effect up cos_to_natural = 0.676
  rule_value L26 mlp_plus_attn_effect gate cos_to_natural = 0.644
  rule_value L26 mlp_plus_attn_effect z cos_to_natural = 0.590
  rule_value L26 mlp_plus_attn_effect mlp_out cos_to_natural = 0.586
  rule_value L26 mlp_plus_attn_effect final_norm_output cos_to_natural = 0.554
```

与 Phase 602 对齐的关键增量：

```text
rule_value L26:
  mlp_repair_only final_norm_output cos_to_natural = 0.420
  mlp_plus_attn_effect final_norm_output cos_to_natural = 0.554

attention effect 确实提高轨迹相似度。
```

但是候选竞争完全没有打开：

```text
rule_value L26:
  mlp_repair_only switch = 0/37, full_margin_gain = -0.001
  mlp_plus_attn_effect switch = 0/37, full_margin_gain = -0.001
  mlp_plus_mlpout_effect switch = 0/37, full_margin_gain = -0.001
  mlp_plus_attn_plus_mlpout_effect switch = 0/37, full_margin_gain = -0.001
  mlpout_effect_only switch = 0/37, full_margin_gain = 0.000

prompt_last L26:
  mlp_repair_only switch = 0/37, full_margin_gain = -0.018
  mlp_plus_attn_effect switch = 0/37, full_margin_gain = -0.018
  mlp_plus_mlpout_effect switch = 0/37, full_margin_gain = -0.018
  mlp_plus_attn_plus_mlpout_effect switch = 0/37, full_margin_gain = -0.018
  mlpout_effect_only switch = 0/37, full_margin_gain = 0.000
```

### 当前最可靠客观事实

1. Phase 602 的附件分析正确：

```text
attention effect 是真实轨迹因子，但不是充分 value gate 因子。
```

2. Phase 603 进一步排除了一个简单解释：

```text
不是只缺 final layer MLP output effect。
```

因为直接加入 natural_correct 的 MLP output effect 后：

```text
Qwen3 没有新增 switch 或 full_margin。
GLM4 没有新增 switch 或 full_margin。
DS7B 没有新增 switch 或 full_margin。
```

3. DS7B 的现象最关键：

```text
rule_value L26 的 attention effect 可以把 final_norm_cos 从 0.420 提高到 0.554，
也可以让 final MLP 内部部件更接近 natural_correct，
但 full_margin_gain 仍为 -0.001，switch 仍为 0/37。
```

这说明：

```text
局部部件相似度提高
不等于
最终候选竞争打开。
```

4. 继续单点补丁的边际收益已经很低。

当前失败链条已经从：

```text
candidate projection 不足
```

推进到：

```text
candidate projection + source attention effect + final MLP output effect
仍然不足
```

### 理论进展

value gate 的候选路径不能再理解成单个可加向量：

```text
x' = x + alpha * v
```

更接近当前数据的形式是：

```text
候选能否胜出取决于整条轨迹是否被最终读出系统接受。
```

更严格地说：

```text
局部补丁可以提高 hidden trajectory similarity，
但 readout competition 需要满足额外的 acceptance condition。
```

当前公式应从单因子加法推进为：

```text
S_c = R_c(N(F_L(...F_1(x, C)...)))
```

其中：

```text
C = context/state condition
F_l = 第 l 层的 attention + MLP + residual 更新
N = final norm
R_c = 候选 c 的读出函数
S_c = 候选 c 的最终分数
```

value gate 的真实判据不是某个局部差值是否存在，而是：

```text
S_target - max(S_competitors) > 0
```

Phase 603 说明：

```text
提高某些 F_l 内部部件与 natural_correct 的相似度，
不必然提高 S_target - max(S_competitors)。
```

### 硬伤和瓶颈

1. 目标样本仍不均衡。

```text
Qwen3 = 7
GLM4 = 13
DS7B = 37
```

主结论应以 DS7B 为主，Qwen3 和 GLM4 只作为弱确认。

2. final MLP output effect patch 仍是部件级补丁，不是完整动态轨迹重演。

它可能没有复现：

```text
gate 与 up 的同步状态
residual 中其它竞争方向
final norm 的缩放接受条件
lm_head 的候选子空间竞争
```

3. 不能把 cos_to_natural 当成充分指标。

本阶段再次确认：

```text
cos_to_natural 上升
不等于
full_margin 上升
```

4. 继续扩大单点 patch 搜索空间风险很高。

因为现在已经测试了：

```text
MLP repair
attention effect
MLP output effect
```

仍无法打开 DS7B 的 value gate。

### 下一步任务

Phase 604 应从单点部件补丁转向：

```text
Final Norm and Readout Acceptance Audit
```

目标：

```text
解释为什么 trajectory similarity 已经提高，
但 lm_head candidate margin 仍不接受。
```

建议测试：

```text
1. 比较 natural_correct、base、mlp_repair_only、mlp_plus_attn_effect、
   mlp_plus_attn_plus_mlpout_effect 在 final_norm 前后的：
   - norm scale
   - RMS
   - answer-vector projection
   - competitor-vector projection
   - target vs competitor margin decomposition

2. 对 final_norm_input 做直接插值：
   h = h_artificial + beta * (h_natural - h_artificial)
   beta = 0.25, 0.5, 1.0, 1.5, 2.0

3. 对 final_norm_output 做直接插值：
   y = y_artificial + beta * (y_natural - y_artificial)

4. 对 lm_head logit space 做只读分解：
   target gain 来自哪个方向？
   competitor gain/loss 来自哪个方向？

5. 判定瓶颈在：
   - final_norm scaling
   - lm_head readout vector alignment
   - competitor suppression
   - multi-token generation trajectory
```

判据：

```text
如果 final_norm_input 插值能提高 full_margin，
但 final_norm_output 之前的部件补丁不能，
说明 final norm 接受条件是关键瓶颈。

如果 final_norm_output 插值能提高 full_margin，
但 input 插值不能，
说明归一化映射本身改变了候选竞争结构。

如果 final_norm_output 插值仍不能打开 margin，
说明问题在 lm_head 候选读出空间或候选集合定义。
```

## Phase 604: Final Norm and Multi-Token Readout Acceptance Audit 最终归一化与多词元读出接受审计 [2026-06-24 15:07]

### 本阶段目标

根据附件对 Phase 603 的分析，继续检查：

```text
为什么 attention effect 和 final MLP output effect 都能提高 trajectory similarity，
但 full candidate margin 仍无法打开？
```

附件判断基本正确：

```text
Phase 603 说明局部轨迹更像 natural_correct，
不等于最终候选竞争被打开。
```

本阶段原始计划是测试：

```text
final norm / lm_head readout acceptance
```

实际执行中发现一个关键细节：

```text
候选值 v05 / v91 / v22 / v48 的第一个 token 相同，都是 " v"。
真正区分候选值的是后续数字 token。
```

tokenization 检查：

```text
Qwen3:
  v05 = [' v', '0', '5']
  v91 = [' v', '9', '1']
  v22 = [' v', '2', '2']
  v48 = [' v', '4', '8']

GLM4:
  v05 = [' v', '05']
  v91 = [' v', '91']
  v22 = [' v', '22']
  v48 = [' v', '48']

DS7B:
  v05 = [' v', '0', '5']
  v91 = [' v', '9', '1']
  v22 = [' v', '2', '2']
  v48 = [' v', '4', '8']
```

因此只 patch prompt_last 的 final_norm 只能影响第一个共同 token，不能改变完整候选值竞争。

本阶段最终改为两类测试：

```text
1. prompt_last 单点 final_norm_input / final_norm_output 插值。
2. sequence-level final_norm_input / final_norm_output 插值：
   对候选答案每一个被预测 token 位置，都用 natural_correct 轨迹插值。
```

### 生成脚本

```text
tests/glm5/phase604_final_norm_readout_acceptance_audit.py
tests/glm5/phase604_final_norm_readout_acceptance_audit_summary.py
```

### 执行命令

```bash
python tests/glm5/phase604_final_norm_readout_acceptance_audit.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase604_final_norm_readout_acceptance_audit \
  --hard-exit-after-model

python tests/glm5/phase604_final_norm_readout_acceptance_audit.py glm4 \
  --confirm \
  --output-dir results/glm5_phase604_final_norm_readout_acceptance_audit \
  --hard-exit-after-model

python tests/glm5/phase604_final_norm_readout_acceptance_audit.py deepseek7b \
  --confirm \
  --betas 1,2 \
  --output-dir results/glm5_phase604_final_norm_readout_acceptance_audit \
  --hard-exit-after-model

python tests/glm5/phase604_final_norm_readout_acceptance_audit_summary.py

python -m py_compile \
  tests/glm5/phase604_final_norm_readout_acceptance_audit.py \
  tests/glm5/phase604_final_norm_readout_acceptance_audit_summary.py
```

说明：

```text
DS7B 第一次 sequence-level 全 beta 运行被外部终止，未产生有效新版结果。
之后删除旧版单点结果，使用主判据 beta = 1,2 完成正式确认。
```

### 结果文件

```text
results/glm5_phase604_final_norm_readout_acceptance_audit/phase604_qwen3_final_norm_readout_acceptance_audit_confirm.json
results/glm5_phase604_final_norm_readout_acceptance_audit/phase604_glm4_final_norm_readout_acceptance_audit_confirm.json
results/glm5_phase604_final_norm_readout_acceptance_audit/phase604_deepseek7b_final_norm_readout_acceptance_audit_confirm.json
results/glm5_phase604_final_norm_readout_acceptance_audit/phase604_cross_model_summary.md
```

### 测试范围

```text
models = qwen3, glm4, deepseek7b
cases/model = 96
Qwen3 target rows = 7
GLM4 target rows = 13
DS7B target rows = 37
```

插值模式：

```text
input_interp:
  只在 prompt_last 位置替换 final_norm_input。

output_interp:
  只在 prompt_last 位置替换 final_norm_output。

seq_input_interp:
  对候选答案的每个预测位置替换 final_norm_input。

seq_output_interp:
  对候选答案的每个预测位置替换 final_norm_output。

random:
  same-norm 随机对照。
```

### 客观结果

#### Qwen3

```text
target rows = 7

prompt_last 单点插值：
  input_interp beta1 full_switch = 0/7, full_margin_gain = 0.000
  output_interp beta1 full_switch = 0/7, full_margin_gain = 0.000

sequence-level 插值：
  seq_input_interp beta1 full_switch = 7/7, full_margin_gain = 7.960
  seq_output_interp beta1 full_switch = 7/7, full_margin_gain = 7.960
  seq_input_interp beta2 full_switch = 7/7, full_margin_gain = 15.945
  seq_output_interp beta2 full_switch = 7/7, full_margin_gain = 18.246

sequence random 对照：
  seq_input_random beta1 full_switch = 1/7, full_margin_gain = 0.161
  seq_output_random beta1 full_switch = 2/7, full_margin_gain = 0.268
```

#### GLM4

```text
target rows = 13

prompt_last 单点插值：
  input_interp beta1 full_switch = 0/13, full_margin_gain = 0.000
  output_interp beta1 full_switch = 0/13, full_margin_gain = 0.000

sequence-level 插值：
  seq_input_interp beta1 full_switch = 13/13, full_margin_gain = 2.913
  seq_output_interp beta1 full_switch = 13/13, full_margin_gain = 2.913
  seq_input_interp beta2 full_switch = 13/13, full_margin_gain = 5.541
  seq_output_interp beta2 full_switch = 13/13, full_margin_gain = 5.829

sequence random 对照：
  seq_input_random beta1 full_switch = 3/13, full_margin_gain = 0.327
  seq_output_random beta1 full_switch = 3/13, full_margin_gain = 0.322
```

#### DS7B

```text
target rows = 37

prompt_last 单点插值：
  input_interp beta1 full_switch = 0/37, full_margin_gain = 0.000
  output_interp beta1 full_switch = 0/37, full_margin_gain = 0.000
  input_interp beta2 full_switch = 0/37, full_margin_gain = 0.000
  output_interp beta2 full_switch = 0/37, full_margin_gain = 0.000

sequence-level 插值：
  seq_input_interp beta1 full_switch = 37/37, full_margin_gain = 7.716
  seq_output_interp beta1 full_switch = 37/37, full_margin_gain = 7.716
  seq_input_interp beta2 full_switch = 37/37, full_margin_gain = 15.285
  seq_output_interp beta2 full_switch = 37/37, full_margin_gain = 17.438

sequence random 对照：
  seq_input_random beta1 full_switch = 2/37, full_margin_gain = -0.105
  seq_output_random beta1 full_switch = 4/37, full_margin_gain = -0.088
  seq_input_random beta2 full_switch = 5/37, full_margin_gain = 0.122
  seq_output_random beta2 full_switch = 7/37, full_margin_gain = -0.334
```

DS7B 的 margin 分解：

```text
seq_input_interp beta1:
  correct_full_delta = +7.443
  old_wrong_full_delta = -0.273
  full_margin_gain = +7.716

seq_output_interp beta1:
  correct_full_delta = +7.443
  old_wrong_full_delta = -0.273
  full_margin_gain = +7.716

seq_output_interp beta2:
  correct_full_delta = +8.519
  old_wrong_full_delta = -8.919
  full_margin_gain = +17.438
```

### 当前最可靠客观事实

1. Phase 603 的附件分析正确：

```text
局部部件相似度提高，不等于完整候选竞争打开。
```

2. prompt_last 的 final norm 单点读出不是 value gate 的完整位置。

跨模型均为：

```text
prompt_last final_norm_input/output 插值：
full_switch = 0
full_margin_gain = 0
```

原因不是 final norm 完全无效，而是：

```text
候选值第一个 token 相同，prompt_last 只影响共同前缀 token。
```

3. 真正的候选值竞争发生在多 token 序列的后续读出位置。

跨模型均为：

```text
sequence-level final_norm_input/output 插值：
Qwen3 = 7/7
GLM4 = 13/13
DS7B = 37/37
```

4. random same-norm 对照远弱于 natural sequence interpolation。

尤其 DS7B：

```text
natural seq beta1 = 37/37, margin +7.716
random seq beta1 = 2/37 或 4/37, margin 约 0 或负数
```

5. input interpolation 与 output interpolation 在 beta1 基本等价。

这说明：

```text
如果给定正确的多 token final_norm_input 轨迹，
final norm 本身可以正常把它映射成可读出的候选优势。
```

因此瓶颈不在 final norm 函数本身，而在：

```text
模型没有自然生成正确的多 token final_norm trajectory。
```

### 理论进展

Phase 604 把 value gate 的瓶颈从：

```text
单位置 readout acceptance
```

推进到：

```text
multi-token readout trajectory acceptance
```

旧的单步公式：

```text
S_c = R_c(N(h_t))
```

不足以解释 value gate，因为候选值不是单 token。

当前更准确的形式是：

```text
S_c = sum_k R_{c,k}(N(h_{t+k}))
```

其中：

```text
c = 候选值
k = 候选值的第 k 个 token
h_{t+k} = 生成到第 k 个预测位置时的隐藏状态
R_{c,k} = 对第 k 个 token 的读出函数
```

value gate 的完整 margin 应写成：

```text
M_c = sum_k log P(c_k | prompt, c_{<k})
      - max_{j != c} sum_k log P(j_k | prompt, j_{<k})
```

Phase 604 说明：

```text
只修补 h_t 不足以打开 M_c。
修补整条 h_{t:t+K} 的 final_norm trajectory 可以打开 M_c。
```

这是一条重要分界：

```text
value gate 不是单点候选选择，
而是候选字符串生成轨迹的多步读出闭合。
```

### 硬伤和限制

1. sequence-level 插值使用了候选答案自身的 natural trajectory。

这不是一个可直接用于生成的无监督修复，因为它相当于知道了候选答案路径。

它证明的是：

```text
如果多 token final_norm trajectory 正确，lm_head 可以读出正确候选。
```

但还没有证明：

```text
模型如何在不知道答案的情况下自然生成这条 trajectory。
```

2. 当前结论主要适用于 value gate 的候选值任务。

因为候选值是多 token 结构：

```text
v + digits
```

它不能直接推广到所有语言生成。

3. first-token margin 无变化不是失败，而是任务结构决定的。

所有候选第一个 token 都是：

```text
" v"
```

所以 first-token 无法区分候选值。

4. random 对照不是零影响。

部分 random sequence patch 能造成少量 switch，说明多 token final_norm 轨迹空间本身很敏感。

但它远弱于 natural sequence interpolation。

### 下一步任务

Phase 605 应转向：

```text
Multi-Token Trajectory Builder Audit
```

核心目标：

```text
找出自然正确轨迹如何从第一步共同 token " v" 过渡到后续数字 token。
```

建议测试：

```text
1. 分别审计 value token 第 1/2/3 个预测位置：
   - " v"
   - 第一位数字
   - 第二位数字

2. 对每个生成位置采集：
   - final_norm_input
   - final_norm_output
   - layer_out
   - final layer attn_out
   - final layer mlp_out
   - source attention pattern

3. 比较 base wrong 与 natural correct：
   - 哪一步开始出现正确数字优势？
   - 哪一步 suppress old_wrong？
   - target gain 和 competitor suppression 分别来自哪一步？

4. 做 token-step causal patch：
   - 只 patch 第一位数字位置
   - 只 patch 第二位数字位置
   - patch 两个数字位置
   - 与 random same-norm 对照比较

5. 如果数字位置 patch 可以局部闭合，
   再追踪这些数字位置的上游 source：
   - answer prefix token
   - rule_value token
   - prompt_last token
   - previous generated digit token
```

Phase 605 的关键判据：

```text
如果只 patch 数字 token 位置即可恢复 full_margin，
说明 value gate 的真正读出瓶颈在 answer-token trajectory。

如果必须 patch 生成前缀和数字位置共同恢复，
说明 value gate 是 autoregressive trajectory builder，
不是静态读出门。
```

## Phase 605: Multi-Token Trajectory Builder Audit 多词元轨迹构造器审计 [2026-06-24 15:37]

### 本阶段目标

根据附件对 Phase 604 的分析，继续推进任务。

附件判断基本正确：

```text
Phase 604 不是简单证明 final norm 有效，
而是纠正了读出位置：
候选值不是 single-token answer，
真正区分候选的是后续 digit token。
```

Phase 605 的目标：

```text
拆开候选答案 token 序列，
判断 value gate 的候选竞争到底发生在：
1. 共同前缀 " v"
2. 第一位数字
3. 第二位数字
4. 两个数字共同轨迹
```

### 生成脚本

```text
tests/glm5/phase605_multi_token_trajectory_builder_audit.py
tests/glm5/phase605_multi_token_trajectory_builder_audit_summary.py
```

### 执行命令

```bash
python tests/glm5/phase605_multi_token_trajectory_builder_audit.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase605_multi_token_trajectory_builder_audit \
  --hard-exit-after-model

python tests/glm5/phase605_multi_token_trajectory_builder_audit.py glm4 \
  --confirm \
  --output-dir results/glm5_phase605_multi_token_trajectory_builder_audit \
  --hard-exit-after-model

python tests/glm5/phase605_multi_token_trajectory_builder_audit.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase605_multi_token_trajectory_builder_audit \
  --hard-exit-after-model

python tests/glm5/phase605_multi_token_trajectory_builder_audit_summary.py

python -m py_compile \
  tests/glm5/phase605_multi_token_trajectory_builder_audit.py \
  tests/glm5/phase605_multi_token_trajectory_builder_audit_summary.py
```

### 结果文件

```text
results/glm5_phase605_multi_token_trajectory_builder_audit/phase605_qwen3_multi_token_trajectory_builder_audit_confirm.json
results/glm5_phase605_multi_token_trajectory_builder_audit/phase605_glm4_multi_token_trajectory_builder_audit_confirm.json
results/glm5_phase605_multi_token_trajectory_builder_audit/phase605_deepseek7b_multi_token_trajectory_builder_audit_confirm.json
results/glm5_phase605_multi_token_trajectory_builder_audit/phase605_cross_model_summary.md
```

### 测试范围

```text
models = qwen3, glm4, deepseek7b
cases/model = 96
Qwen3 target rows = 7
GLM4 target rows = 13
DS7B target rows = 37
```

token group：

```text
prefix0 = 第 0 个答案 token，共同前缀 " v"
digit1 = 第 1 个答案 token，第一位数字或 GLM4 的两位数字整体
digit2 = 第 2 个答案 token，第二位数字
digits = digit1 + digit2
all = prefix0 + digits
```

patch component：

```text
final_norm_input
final_norm_output
```

control：

```text
digits_random
all_random
```

### 客观结果

#### Qwen3

tokenization：

```text
v05 = [' v', '0', '5']
v91 = [' v', '9', '1']
v22 = [' v', '2', '2']
v48 = [' v', '4', '8']
```

关键结果：

```text
input_prefix0:
  switch = 0/7
  margin_gain = 0.000

input_digit1:
  switch = 7/7
  margin_gain = 7.821
  correct_delta = +1.761
  old_wrong_delta = -6.060

input_digit2:
  switch = 0/7
  margin_gain = 0.138

input_digits:
  switch = 7/7
  margin_gain = 7.960

input_digits_random:
  switch = 1/7
  margin_gain = 0.165
```

output 结果与 input 基本一致：

```text
output_digit1:
  switch = 7/7
  margin_gain = 7.821

output_digits:
  switch = 7/7
  margin_gain = 7.960
```

#### GLM4

tokenization：

```text
v05 = [' v', '05']
v91 = [' v', '91']
v22 = [' v', '22']
v48 = [' v', '48']
```

关键结果：

```text
input_prefix0:
  switch = 0/13
  margin_gain = 0.000

input_digit1:
  switch = 13/13
  margin_gain = 2.913
  correct_delta = +1.001
  old_wrong_delta = -1.913

input_digits:
  switch = 13/13
  margin_gain = 2.913

input_digits_random:
  switch = 2/13
  margin_gain = 0.217
```

说明：

```text
GLM4 的 digit1 实际是两位数字整体 token，
所以 digit1 就是主要候选区分位置。
```

#### DS7B

tokenization：

```text
v05 = [' v', '0', '5']
v91 = [' v', '9', '1']
v22 = [' v', '2', '2']
v48 = [' v', '4', '8']
```

关键结果：

```text
input_prefix0:
  switch = 0/37
  margin_gain = 0.000
  correct_delta = +5.370
  old_wrong_delta = +5.370

input_digit1:
  switch = 37/37
  margin_gain = 7.049
  correct_delta = +2.048
  old_wrong_delta = -5.001

input_digit2:
  switch = 7/37
  margin_gain = 0.667
  correct_delta = +0.024
  old_wrong_delta = -0.642

input_digits:
  switch = 37/37
  margin_gain = 7.716
  correct_delta = +2.073
  old_wrong_delta = -5.643

input_digits_random:
  switch = 4/37
  margin_gain = 0.131
```

output 结果与 input 基本一致：

```text
output_digit1:
  switch = 37/37
  margin_gain = 7.049

output_digits:
  switch = 37/37
  margin_gain = 7.716

output_digits_random:
  switch = 3/37
  margin_gain = -0.061
```

### 当前最可靠客观事实

1. Phase 604 的附件判断正确：

```text
value gate 不是 prompt_last 单点读出，
而是 multi-token readout trajectory。
```

2. 共同前缀不是候选区分位置。

跨模型：

```text
prefix0 switch = 0
margin_gain = 0
```

DS7B 说明最清楚：

```text
prefix0 同时提高 correct 和 old_wrong：
correct_delta = +5.370
old_wrong_delta = +5.370
margin_gain = 0
```

这说明 prefix0 只是在生成共同格式：

```text
" v"
```

不是选择具体 value。

3. 第一位数字是 value gate 的主读出位置。

跨模型：

```text
Qwen3 digit1 = 7/7
GLM4 digit1 = 13/13
DS7B digit1 = 37/37
```

4. 第二位数字是弱补偿位置，不是主决策位置。

```text
Qwen3 digit2 = 0/7
DS7B digit2 = 7/37
```

它主要提供少量后续确认或 competitor suppression。

5. input 与 output 结果一致。

说明：

```text
只要数字 token 位置的 final_norm_input 正确，
final_norm 和 lm_head 可以正常完成读出。
```

因此瓶颈不在 final norm 函数本身，也不在 lm_head 读不出，而在：

```text
模型如何构造第一位数字位置的正确 hidden trajectory。
```

### 理论进展

Phase 605 把 value gate 的结构从：

```text
multi-token readout trajectory
```

进一步拆成：

```text
format prefix generator + digit selector
```

当前最简结构：

```text
候选值生成 = prefix_format_step + value_digit_decision_step + weak_tail_confirmation_step
```

对应公式：

```text
Score(vab) =
log P(" v" | prompt)
+ log P(a | prompt, " v")
+ log P(b | prompt, " v", a)
```

其中：

```text
log P(" v" | prompt)
```

主要是格式前缀，不区分候选值。

候选选择主要发生在：

```text
log P(a | prompt, " v")
```

第二位数字：

```text
log P(b | prompt, " v", a)
```

更多是弱补偿和确认。

更完整的 value gate margin：

```text
M =
[log P(a_correct | prompt, " v") - log P(a_wrong | prompt, " v")]
+ [log P(b_correct | prompt, " v", a_correct)
   - log P(b_wrong | prompt, " v", a_wrong)]
```

Phase 605 的客观结果说明：

```text
第一项是主项，第二项是弱项。
```

### 问题和硬伤

1. 当前任务的候选值是人工编码：

```text
v05 / v91 / v22 / v48
```

结论对 value retrieval 很强，但不能直接等同于自然语言所有多词答案。

2. patch 是 oracle trajectory patch。

也就是说：

```text
我们知道 repair_prompt + candidate answer 的自然轨迹，
再把它补到 base answer 生成中。
```

这证明了读出位置和因果充分性，但还没有证明模型自然如何构造这条轨迹。

3. 第一位数字位置虽然被定位为主决策点，但其上游来源尚未拆解。

还不知道第一位数字位置的正确 final_norm_input 来自：

```text
rule_value token
previous prefix token
prompt_last
final attention source
final MLP
earlier residual trajectory
```

4. random 对照仍有弱效果。

说明数字位置的读出空间敏感，需要在 Phase 606 继续做 source-resolved audit，避免把纯读出敏感性误判成机制解释。

### 下一步任务

Phase 606 应进入：

```text
Digit1 Upstream Source Decomposition
```

核心目标：

```text
既然第一位数字是 value gate 的主决策位置，
下一步必须找出第一位数字位置的 hidden trajectory 是由哪些上游来源构造的。
```

建议测试：

```text
1. 固定第一位数字预测位置。

2. 分解该位置最后层：
   - layer_input
   - attn_out
   - mlp_input
   - mlp_out
   - layer_out
   - final_norm_input
   - final_norm_output

3. source attention 分组：
   - prompt_last
   - prefix token " v"
   - rule_value token
   - rule_relation token
   - object token
   - other tokens

4. causal patch：
   - 只 patch digit1 的 final layer attn_out
   - 只 patch digit1 的 final layer mlp_out
   - patch digit1 layer_input
   - patch digit1 final_norm_input
   - 加 random same-norm 对照

5. 判定：
   - digit1 主信号是 attention 读 rule_value 而来，
   - 还是 MLP 在 prefix-conditioned state 下生成，
   - 还是 residual 已经携带到 digit1。
```

关键判据：

```text
如果 digit1 attn_out patch 可以恢复大部分 margin，
说明 value retrieval 在生成第一位数字时主要是 attention source retrieval。

如果 digit1 mlp_out patch 才恢复，
说明第一位数字是 MLP 在上下文状态中完成映射。

如果 layer_input patch 恢复而 attn/mlp 单独不恢复，
说明第一位数字决策已经在 residual trajectory 中预构造。
```

## Phase 606: Digit1 Upstream Source Decomposition 第一位数字上游来源分解 [2026-06-24 15:51]

### 本阶段目标

根据附件对 Phase 605 的分析，继续完成任务。

附件判断基本正确：

```text
Phase 605 是读出位置纠偏阶段。
value gate 的主决策点不是 prompt_last，
而是 answer-token trajectory 中的 digit1 selector。
```

Phase 606 的目标：

```text
固定 digit1 预测位置，
判断该位置的正确 hidden trajectory 来自最后层的哪个部件：
layer_input
attn_out
mlp_out
final_norm_input
final_norm_output
```

同时记录最后层 digit1 位置的 source attention mass 差异。

### 生成脚本

```text
tests/glm5/phase606_digit1_upstream_source_decomposition.py
tests/glm5/phase606_digit1_upstream_source_decomposition_summary.py
```

### 执行命令

```bash
python tests/glm5/phase606_digit1_upstream_source_decomposition.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase606_digit1_upstream_source_decomposition \
  --hard-exit-after-model

python tests/glm5/phase606_digit1_upstream_source_decomposition.py glm4 \
  --confirm \
  --output-dir results/glm5_phase606_digit1_upstream_source_decomposition \
  --hard-exit-after-model

python tests/glm5/phase606_digit1_upstream_source_decomposition.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase606_digit1_upstream_source_decomposition \
  --hard-exit-after-model

python tests/glm5/phase606_digit1_upstream_source_decomposition_summary.py

python -m py_compile \
  tests/glm5/phase606_digit1_upstream_source_decomposition.py \
  tests/glm5/phase606_digit1_upstream_source_decomposition_summary.py
```

### 结果文件

```text
results/glm5_phase606_digit1_upstream_source_decomposition/phase606_qwen3_digit1_upstream_source_decomposition_confirm.json
results/glm5_phase606_digit1_upstream_source_decomposition/phase606_glm4_digit1_upstream_source_decomposition_confirm.json
results/glm5_phase606_digit1_upstream_source_decomposition/phase606_deepseek7b_digit1_upstream_source_decomposition_confirm.json
results/glm5_phase606_digit1_upstream_source_decomposition/phase606_cross_model_summary.md
```

### 测试范围

```text
models = qwen3, glm4, deepseek7b
cases/model = 96
Qwen3 target rows = 7
GLM4 target rows = 13
DS7B target rows = 37
```

component patch：

```text
layer_input
attn_out
mlp_out
final_norm_input
final_norm_output
```

random control：

```text
layer_input_random
attn_out_random
mlp_out_random
final_norm_input_random
```

source attention groups：

```text
prompt_last
answer_prefix
rule_value
rule_relation
query_relation
object
other
```

### 客观结果

#### Qwen3

```text
target rows = 7

layer_input:
  switch = 7/7
  margin_gain = 8.072
  correct_delta = +1.764
  old_wrong_delta = -6.308

final_norm_input:
  switch = 7/7
  margin_gain = 7.821
  correct_delta = +1.761
  old_wrong_delta = -6.060

final_norm_output:
  switch = 7/7
  margin_gain = 7.821

attn_out:
  switch = 0/7
  margin_gain = -1.214

mlp_out:
  switch = 1/7
  margin_gain = -0.786
```

attention source mass delta：

```text
prompt_last +0.014
rule_value +0.001
answer_prefix -0.005
other -0.008
```

#### GLM4

```text
target rows = 13

layer_input:
  switch = 13/13
  margin_gain = 2.923
  correct_delta = +1.004
  old_wrong_delta = -1.919

final_norm_input:
  switch = 13/13
  margin_gain = 2.913

final_norm_output:
  switch = 13/13
  margin_gain = 2.913

attn_out:
  switch = 0/13
  margin_gain = -0.038

mlp_out:
  switch = 3/13
  margin_gain = 0.120
```

attention source mass delta：

```text
prompt_last +0.030
rule_value +0.005
answer_prefix -0.018
other -0.017
```

#### DS7B

```text
target rows = 37

final_norm_input:
  switch = 37/37
  margin_gain = 7.049
  correct_delta = +2.048
  old_wrong_delta = -5.001

final_norm_output:
  switch = 37/37
  margin_gain = 7.049

layer_input:
  switch = 37/37
  margin_gain = 6.993
  correct_delta = +2.051
  old_wrong_delta = -4.942

attn_out:
  switch = 0/37
  margin_gain = -0.061

mlp_out:
  switch = 0/37
  margin_gain = -1.370
```

random control：

```text
layer_input_random:
  switch = 0/37
  margin_gain = -0.125

final_norm_input_random:
  switch = 3/37
  margin_gain = -0.058

attn_out_random:
  switch = 1/37
  margin_gain = -0.042

mlp_out_random:
  switch = 2/37
  margin_gain = 0.063
```

attention source mass delta：

```text
prompt_last +0.040
answer_prefix +0.011
rule_value +0.008
rule_relation +0.003
other -0.062
```

### 当前最可靠客观事实

1. Phase 605 的附件判断正确：

```text
digit1 是 value gate 的主读出位置。
```

2. digit1 的决策信号在最后层 layer_input 之前已经基本形成。

跨模型：

```text
layer_input patch:
  Qwen3 = 7/7
  GLM4 = 13/13
  DS7B = 37/37
```

并且接近 final_norm_input 的效果。

3. 最后一层 attn_out 不是 digit1 主决策信号的充分来源。

跨模型：

```text
attn_out patch:
  Qwen3 = 0/7
  GLM4 = 0/13
  DS7B = 0/37
```

4. 最后一层 mlp_out 也不是 digit1 主决策信号的充分来源。

尤其 DS7B：

```text
mlp_out patch:
  switch = 0/37
  margin_gain = -1.370
```

5. final_norm / lm_head 能正常读出已经形成的 digit1 residual state。

```text
final_norm_input patch:
  Qwen3 = 7/7
  GLM4 = 13/13
  DS7B = 37/37
```

6. attention source mass 只给出弱线索，不构成闭合解释。

自然正确轨迹中 digit1 最后一层注意力相对 base 主要表现为：

```text
prompt_last mass 上升
rule_value 小幅上升
other mass 下降
```

但由于 attn_out patch 本身无法恢复 margin，最后层 attention selection 更像伴随读出状态，而不是最后一步主生成器。

### 理论进展

Phase 606 把 value gate 的瓶颈继续前移：

```text
digit1 final readout position
```

不是由最后层 attn_out 或最后层 mlp_out 临时生成，而是：

```text
digit1 residual trajectory 在进入最后层之前已经构造完成。
```

因此当前最简链条变成：

```text
prompt / rule context
→ prefix token " v"
→ digit1 residual trajectory builder
→ final layer layer_input
→ final_norm / lm_head readout
→ value digit selection
```

Phase 606 的关键公式化：

```text
h_digit1^{L-1}
```

已经包含主候选选择方向。

最后层更新：

```text
h_digit1^L = h_digit1^{L-1} + Attn_L(h) + MLP_L(h)
```

不是主因果生成步骤，因为：

```text
Patch(h_digit1^{L-1}) 有效；
Patch(Attn_L) 无效；
Patch(MLP_L) 无效。
```

更准确地说：

```text
value gate 的主决策不是 final layer computation，
而是 pre-final residual trajectory computation。
```

### 问题和硬伤

1. 只定位到最后层入口，不等于找到真正生成层。

Phase 606 说明：

```text
信号已经在 final layer input 中。
```

但还不知道它是在：

```text
L24 / L25 / L26
```

还是更早层生成。

2. source attention mass 只是读出图谱，不是因果路径。

虽然 prompt_last / rule_value mass 有上升，但 attn_out patch 无法恢复，因此不能说：

```text
最后层 attention 从 rule_value 读取并生成 digit1。
```

3. mlp_out patch 失败不等于 MLP 无关。

可能是：

```text
上游层 MLP 已经生成；
最后层 MLP 只是归一化/竞争调整；
```

当前只排除了最后层 MLP 作为充分因果补丁。

4. 仍然是 oracle patch。

我们使用 natural correct digit1 trajectory 作为补丁来源，还没有解释模型自然如何构造它。

### 下一步任务

Phase 607 应进入：

```text
Pre-Final Residual Trajectory Layer Scan
```

核心目标：

```text
找到 digit1 主决策 residual trajectory 是在哪些上游层形成的。
```

建议测试：

```text
1. 固定 digit1 预测位置。

2. 对 probe layers 做扫描：
   Qwen3: L28-L35
   GLM4: L32-L39
   DS7B: L20-L27

3. 在每一层测试：
   - layer_input patch
   - layer_out patch
   - attn_out patch
   - mlp_out patch

4. 每个 patch 都测：
   - full switch
   - margin gain
   - correct_delta
   - old_wrong_delta
   - random same-norm control

5. 找到：
   - 最早可以恢复 digit1 margin 的 layer_input
   - 哪一层 layer_out 开始携带候选选择
   - attention / MLP 哪个模块首次写入该方向
```

判据：

```text
如果某层 layer_out 有效而该层 layer_input 无效，
说明该层是写入层。

如果某层 attn_out 有效，
说明 attention 是写入器。

如果某层 mlp_out 有效，
说明 MLP 是写入器。

如果 layer_input 从较早层开始一直有效，
说明 digit1 决策在更早 residual stream 中已经形成。
```

## Phase 607: Pre-Final Residual Trajectory Layer Scan 最终层前残差轨迹层扫描 [2026-06-24 16:41]

### 本阶段目标

根据附件对 Phase 606 的分析，继续完成任务。

附件判断基本正确：

```text
Phase 606 证明 digit1 的主决策信号在最后层 layer_input 之前已经基本形成。
```

Phase 607 的目标：

```text
固定 digit1 预测位置，
向上游层扫描 layer_input / layer_out / attn_out / mlp_out，
寻找 digit1 主决策 residual trajectory 首次形成的位置。
```

### 生成脚本

```text
tests/glm5/phase607_prefinal_residual_trajectory_layer_scan.py
tests/glm5/phase607_prefinal_residual_trajectory_layer_scan_summary.py
```

### 执行命令

```bash
python tests/glm5/phase607_prefinal_residual_trajectory_layer_scan.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase607_prefinal_residual_trajectory_layer_scan \
  --hard-exit-after-model

python tests/glm5/phase607_prefinal_residual_trajectory_layer_scan.py glm4 \
  --confirm \
  --output-dir results/glm5_phase607_prefinal_residual_trajectory_layer_scan \
  --hard-exit-after-model

python tests/glm5/phase607_prefinal_residual_trajectory_layer_scan.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase607_prefinal_residual_trajectory_layer_scan \
  --hard-exit-after-model

python tests/glm5/phase607_prefinal_residual_trajectory_layer_scan_summary.py

python -m py_compile \
  tests/glm5/phase607_prefinal_residual_trajectory_layer_scan.py \
  tests/glm5/phase607_prefinal_residual_trajectory_layer_scan_summary.py
```

### 结果文件

```text
results/glm5_phase607_prefinal_residual_trajectory_layer_scan/phase607_qwen3_prefinal_residual_trajectory_layer_scan_confirm.json
results/glm5_phase607_prefinal_residual_trajectory_layer_scan/phase607_glm4_prefinal_residual_trajectory_layer_scan_confirm.json
results/glm5_phase607_prefinal_residual_trajectory_layer_scan/phase607_deepseek7b_prefinal_residual_trajectory_layer_scan_confirm.json
results/glm5_phase607_prefinal_residual_trajectory_layer_scan/phase607_cross_model_summary.md
```

### 测试范围

```text
models = qwen3, glm4, deepseek7b
cases/model = 96
Qwen3 target rows = 7
GLM4 target rows = 13
DS7B target rows = 37
```

扫描层：

```text
Qwen3: L28-L35
GLM4: L32-L39
DS7B: L20-L27
```

扫描部件：

```text
layer_input
layer_out
attn_out
mlp_out
```

每个部件均加入 same-norm random control。

### 客观结果

#### Qwen3

```text
target rows = 7
scan layers = L28-L35
```

layer_input / layer_out 在扫描窗口第一层已经完全有效：

```text
L28 layer_input:
  switch = 7/7
  margin_gain = 8.663
  correct_delta = +1.764
  old_wrong_delta = -6.899

L28 layer_out:
  switch = 7/7
  margin_gain = 8.717
```

后续层继续保持强有效：

```text
L29 layer_input = 7/7, margin 8.717
L32 layer_input = 7/7, margin 8.448
L35 layer_input = 7/7, margin 8.072
L35 layer_out   = 7/7, margin 7.821
```

模块补丁较弱：

```text
L29 attn_out:
  switch = 6/7
  margin_gain = 2.037

L34 mlp_out:
  switch = 7/7
  margin_gain = 2.536
```

但这些都弱于 layer_input / layer_out。

#### GLM4

```text
target rows = 13
scan layers = L32-L39
```

扫描窗口第一层已经基本有效：

```text
L32 layer_input:
  switch = 12/13
  margin_gain = 2.356

L32 layer_out:
  switch = 13/13
  margin_gain = 2.413
```

中后层增强：

```text
L36 layer_out:
  switch = 13/13
  margin_gain = 2.942

L37 layer_input:
  switch = 13/13
  margin_gain = 2.942

L39 layer_out:
  switch = 13/13
  margin_gain = 2.913
```

模块补丁较弱：

```text
L34 attn_out:
  switch = 4/13
  margin_gain = 0.168

L39 mlp_out:
  switch = 3/13
  margin_gain = 0.120
```

#### DS7B

```text
target rows = 37
scan layers = L20-L27
```

DS7B 给出最清楚的形成曲线：

```text
L20 layer_input:
  switch = 3/37
  margin_gain = -0.389

L21 layer_out:
  switch = 6/37
  margin_gain = -0.183

L22 layer_out:
  switch = 33/37
  margin_gain = 4.726

L23 layer_input:
  switch = 33/37
  margin_gain = 4.726

L23 layer_out:
  switch = 37/37
  margin_gain = 6.287

L24 layer_input:
  switch = 37/37
  margin_gain = 6.287

L27 layer_out:
  switch = 37/37
  margin_gain = 7.049
```

DS7B 的关键模块：

```text
L22 attn_out:
  switch = 33/37
  margin_gain = 3.423
  correct_delta = +1.582
  old_wrong_delta = -1.841

L23 attn_out:
  switch = 24/37
  margin_gain = 1.665

L26 mlp_out:
  switch = 21/37
  margin_gain = 1.274

L25 mlp_out:
  switch = 14/37
  margin_gain = 0.881

L24 mlp_out:
  switch = 12/37
  margin_gain = 0.844
```

DS7B 最清楚地显示：

```text
L22 attention 是重要写入候选；
L22 layer_out / L23 layer_input 开始携带大部分 digit1 决策；
L23-L27 residual trajectory 持续巩固；
后续 MLP 有补偿但不是主写入。
```

### 当前最可靠客观事实

1. Phase 606 附件判断正确：

```text
digit1 决策不在最后层临时生成，
而是来自 pre-final residual trajectory。
```

2. DS7B 已经出现明确形成层：

```text
L22 是关键跃迁点。
```

证据：

```text
L21 layer_out:
  6/37, margin -0.183

L22 layer_out:
  33/37, margin +4.726

L22 attn_out:
  33/37, margin +3.423
```

因此 DS7B 的 digit1 主决策很可能在：

```text
L22 attention 写入，
L22 residual 输出开始形成，
L23 以后持续巩固。
```

3. Qwen3 和 GLM4 的扫描窗口仍然偏晚。

Qwen3：

```text
L28 layer_input 已经 7/7。
```

GLM4：

```text
L32 layer_input 已经 12/13，L32 layer_out 已经 13/13。
```

这说明二者的首次写入层可能早于当前扫描窗口。

4. layer_input / layer_out 是最稳定的因果补丁。

跨模型都明显强于单独 attn_out / mlp_out。

这说明：

```text
digit1 选择信号主要表现为 residual trajectory state，
而不是单个最后阶段模块输出。
```

5. 模块补丁的解释要谨慎。

Qwen3 的 L29 attn_out 和 L34 mlp_out 有一定效果，DS7B 的 L22 attn_out 很强。
但 Qwen3/GLM4 未找到完整写入层，不能简单说所有模型都由同一层同一模块写入。

### 理论进展

Phase 607 把 value gate 链条推进为：

```text
format prefix generator
→ digit1 residual trajectory builder
→ final readout
```

其中 DS7B 的可观测链条更具体：

```text
prefix " v"
→ L22 attention write
→ L22/L23 residual state
→ L24-L27 residual consolidation
→ final_norm / lm_head
→ digit1 selection
```

当前 value gate 不能再理解为：

```text
某层某向量直接被 lm_head 读出
```

而应理解为：

```text
一个在自回归答案位置上逐层形成的 residual trajectory。
```

更接近当前数据的公式：

```text
h_{digit1}^{l+1}
= h_{digit1}^{l}
+ A_l(h_{\le digit1})
+ M_l(h_{digit1}^{l} + A_l)
```

候选 margin 在某层的可恢复性：

```text
M_l =
Score_correct(Patch(h_{digit1}^{l}))
- Score_wrong(Patch(h_{digit1}^{l}))
```

DS7B 中：

```text
M_{L21} 低，
M_{L22 out} 高，
M_{L23 input} 高。
```

所以 L22 是候选主写入层。

### 问题和硬伤

1. Qwen3 / GLM4 没有找到首次写入层。

当前扫描窗口：

```text
Qwen3 L28-L35
GLM4 L32-L39
```

太靠后。两者在窗口起点已经强有效，需要向更早层扫描。

2. DS7B 的 first_effective 自动摘要不能直接当成主结论。

脚本里的 first_effective 是按：

```text
switch > 0
```

筛选，导致 DS7B L20 的弱偶然 switch 也被列为 first effective。

真正可靠判据应同时看：

```text
switch 数量
margin_gain
random control
```

因此 DS7B 的真实强跃迁应判定为：

```text
L22 layer_out / L22 attn_out。
```

3. 当前 patch 仍是 oracle repair trajectory patch。

它证明：

```text
某层状态足以恢复 digit1 margin。
```

但还没有解释模型如何在 base prompt 中自然生成该状态。

4. attention 写入还没做 source causal decomposition。

DS7B L22 attn_out 很强，但还不知道它主要读取：

```text
rule_value
answer_prefix
prompt_last
rule_relation
other
```

### 下一步任务

Phase 608 应进入两路并行中的第一路：

```text
DS7B L22 Attention Source Causal Decomposition
```

优先理由：

```text
DS7B 已经出现最清晰的 L22 attention 写入证据。
```

测试方案：

```text
1. 固定 DS7B digit1 位置与 L22。

2. 将 L22 attention output 按 source token group 分解：
   - rule_value
   - answer_prefix " v"
   - prompt_last
   - rule_relation
   - query_relation
   - object
   - other

3. 分别 patch 每个 source group 的 attention contribution。

4. 测：
   - switch
   - margin_gain
   - correct_delta
   - old_wrong_delta
   - random same-norm source contribution

5. 判定 L22 attention 写入到底读的是 value source，
   还是 prefix-conditioned state，
   或者分布式 source mixture。
```

同时 Phase 609 可作为后续：

```text
Qwen3 / GLM4 earlier-layer scan
```

把 Qwen3 扫描提前到 L16-L28，GLM4 提前到 L20-L32。

## Phase 608: Attention Source K/V Decomposition 注意力源词元K/V因果分解 [2026-06-24 17:22]

### 本阶段目标

根据 Phase 607 的结果，DS7B 在 L22 attention 出现最清晰的 digit1 选择轨迹跃迁：

```text
L22 layer_out: 33/37, margin +4.726
L22 attn_out:  33/37, margin +3.423
L23 layer_out: 37/37, margin +6.287
```

因此 Phase 608 测试一个更细的问题：

```text
L22 attention 的有效写入，是否来自某个明确 source token group 的 K/V 贡献？
```

如果某个 source group 的 K/V delta 能恢复 digit1 选择，就说明 value gate 的写入路径可以进一步定位到：

```text
source token group -> K/V -> attention write -> digit1 residual trajectory
```

### 附加分析判断

附件中对 Phase 607 的判断基本正确：

```text
1. Phase 607 正确把问题从 final readout 前移到 digit1 residual trajectory builder。
2. DS7B 的真实强跃迁不是脚本 first_effective 里的 L20 弱偶发项，而是 L22 layer_out / L22 attn_out。
3. L22 attention 是当前最强写入候选。
4. Qwen3 和 GLM4 的扫描窗口偏晚，只能说明信号已经存在，不能说明首次写入层。
5. 下一步应做 attention source causal decomposition。
```

需要修正和谨慎的地方：

```text
Phase 607 证明 L22 attention output 整体有效，
但还没有证明 source token K/V 层面的某个单点源有效。
```

Phase 608 正是对这一点做审计。

### 执行命令

```bash
python tests/glm5/phase608_attention_source_kv_decomposition.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase608_attention_source_kv_decomposition \
  --hard-exit-after-model

python tests/glm5/phase608_attention_source_kv_decomposition.py glm4 \
  --confirm \
  --output-dir results/glm5_phase608_attention_source_kv_decomposition \
  --hard-exit-after-model

python tests/glm5/phase608_attention_source_kv_decomposition.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase608_attention_source_kv_decomposition \
  --hard-exit-after-model

python tests/glm5/phase608_attention_source_kv_decomposition_summary.py

python -m py_compile \
  tests/glm5/phase608_attention_source_kv_decomposition.py \
  tests/glm5/phase608_attention_source_kv_decomposition_summary.py
```

### 脚本与结果

- 主脚本：`tests/glm5/phase608_attention_source_kv_decomposition.py`
- 汇总脚本：`tests/glm5/phase608_attention_source_kv_decomposition_summary.py`
- Qwen3 结果：`results/glm5_phase608_attention_source_kv_decomposition/phase608_qwen3_attention_source_kv_decomposition_confirm.json`
- GLM4 结果：`results/glm5_phase608_attention_source_kv_decomposition/phase608_glm4_attention_source_kv_decomposition_confirm.json`
- DS7B 结果：`results/glm5_phase608_attention_source_kv_decomposition/phase608_deepseek7b_attention_source_kv_decomposition_confirm.json`
- 跨模型汇总：`results/glm5_phase608_attention_source_kv_decomposition/phase608_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
cases = 96
rows = base wrong and repair correct target cases only
candidate values = v05, v91, v22, v48
source groups = rule_value, rule_relation, query_relation, query_category, query_object, prompt_last, answer_prefix, random_position
patch modes = v_delta, k_delta, kv_delta, kv_random
```

测试层位：

```text
Qwen3: L29
GLM4: L34
DS7B: L22
```

这些层位来自 Phase 607 中 attention component 相对最有信号的位置。

### 测试原理

对每个 target case，先计算：

```text
base prompt:   错误
repair prompt: 正确
```

然后在同一个候选答案评分过程中，捕获目标 attention 层的：

```text
k_proj output
v_proj output
```

对每个 source group，计算 repair 与 base 的 K/V 均值差：

```text
delta_K(g) = mean(K_repair[g]) - mean(K_base[g])
delta_V(g) = mean(V_repair[g]) - mean(V_base[g])
```

在 base prompt 真实 forward 中，对 source group 位置做四类 patch：

```text
v_delta:  只加 delta_V
k_delta:  只加 delta_K
kv_delta: 同时加 delta_K 和 delta_V
kv_random: 同范数随机 K/V 对照
```

然后重新计算四个候选 value 的完整答案分数，测：

```text
switch
margin_gain
correct_delta
old_wrong_delta
```

### 客观结果

#### Qwen3

```text
rows = 7
layer = L29
```

最强项：

```text
random_position k_delta: 2/7, margin -0.018
answer_prefix k_delta: 1/7, margin +0.054
query_object k_delta: 1/7, margin +0.054
rule_value k_delta: 1/7, margin +0.018
```

判断：

```text
没有可靠 source group。
最高 switch 来自 random_position k_delta，而且 margin 仍为负。
```

#### GLM4

```text
rows = 13
layer = L34
```

最强项：

```text
rule_value kv_delta: 1/13, margin +0.014
rule_value v_delta:  1/13, margin +0.005
rule_value k_delta:  1/13, margin +0.000
rule_value kv_random: 1/13, margin -0.030
```

判断：

```text
rule_value 有极弱信号，但强度接近随机对照，不能作为机制证据。
```

#### DS7B

```text
rows = 37
layer = L22
```

最强项：

```text
rule_value kv_random:      1/37, margin +0.115
answer_prefix v_delta:    1/37, margin +0.039
query_object kv_random:   1/37, margin +0.036
query_relation kv_random: 1/37, margin +0.004
rule_relation v_delta:    1/37, margin -0.005
```

关键对照：

```text
rule_value kv_delta:   0/37, margin -0.028
rule_value v_delta:    0/37, margin -0.017
rule_value k_delta:    0/37, margin -0.012
answer_prefix kv_delta: 0/37, margin +0.005
query_relation kv_delta: 0/37, margin -0.005
query_object kv_delta:   0/37, margin -0.018
```

判断：

```text
DS7B L22 attention output 整体在 Phase 607 中非常有效，
但单独 source-token K/V patch 几乎完全无效。
```

### 当前最可靠客观事实

1. Phase 608 是一个强负结果。

```text
source-token K/V delta patch 不能复现 Phase 607 的 L22 attn_out 强修复。
```

2. DS7B 的 L22 attention 仍然是写入候选，但写入机制不等于简单 source K/V 替换。

对比：

```text
Phase 607 DS7B L22 attn_out: 33/37, margin +3.423
Phase 608 DS7B best real source KV: answer_prefix v_delta 1/37, margin +0.039
Phase 608 DS7B rule_value kv_delta: 0/37, margin -0.028
```

3. rule_value 不是可直接替换的单点 value source。

如果 value gate 是简单读取正确 value rule 的 V 向量，那么：

```text
rule_value v_delta 或 rule_value kv_delta 应明显有效。
```

实际没有出现。

4. answer_prefix 有极弱迹象，但不足以支持结论。

```text
DS7B answer_prefix v_delta: 1/37, margin +0.039
Qwen3 answer_prefix k_delta: 1/7, margin +0.054
```

强度太低，且不能跨模型形成稳健模式。

5. L22 attention 的有效信息可能不在 source K/V 单点，而在更高阶组合。

候选解释包括：

```text
1. query state 控制了 attention 读出，而不是 source K/V 本身；
2. attention pattern / score matrix 发生了整体变化；
3. o_proj input 的多头组合才是有效单位；
4. repair-base 差异不是单 source delta，而是多 source 多头的联合构型；
5. K/V patch 只改源内容，没有改 query，因此无法重建选择轨迹。
```

### 理论进展

Phase 607 后的简化模型是：

```text
source token -> L22 attention -> digit1 residual trajectory
```

Phase 608 否定了其中过于简单的版本：

```text
single source K/V -> L22 attention -> digit1 residual trajectory
```

更准确的当前模型应改为：

```text
conditioned query state + distributed source field + attention pattern
-> o_proj input mixture
-> attention output
-> digit1 residual trajectory
```

也就是说，L22 attention 不是单纯读取某个 value token，而是在当前 answer_prefix query state 条件下，对多个源位置形成一种组合场。

当前公式可写为：

```text
A_l(t)
= W_O^l \sum_{h}\sum_{s\le t}
\alpha_{h,t,s}^l(Q_{h,t}^l, K_{h,s}^l)
V_{h,s}^l
```

Phase 607 证明：

```text
Patch(A_{22}(digit1)) 有强因果效果。
```

Phase 608 证明：

```text
Patch(K_s,V_s) for single source group 没有强因果效果。
```

因此下一步不应继续只扩大 source group，而应直接拆：

```text
Q_t
attention pattern alpha_{t,s}
o_proj input head mixture
```

### 问题和硬伤

1. 这不是机制闭合，而是排除了简单 source K/V 解释。

当前仍不知道 L22 attention output 强效果来自：

```text
query
attention weights
o_proj input
multi-head mixture
```

2. source group 仍是人工分组。

例如 rule_value 只取 correct value 的 token 末位，可能漏掉了：

```text
整条 rule line
category-token + relation-token + value-token 组合
多个 value candidates 的相对位置
```

3. patch 方式只做均值 delta。

如果真实机制依赖 head-specific / position-specific 结构，均值 delta 会破坏结构。

4. Qwen3 / GLM4 的 attention 层位不是首次写入层。

因此两者的负结果不能说明没有 source K/V 机制，只说明当前候选层位没有简单源补丁效果。

5. DS7B 的 L22 负结果更可靠。

因为 Phase 607 已经确认 L22 attn_out 强有效，而 Phase 608 在同一层测试 K/V source delta 却无效。

### 下一步任务

Phase 609 应继续围绕 DS7B L22 做更接近真实 attention 计算单位的拆解：

```text
Phase 609: DS7B L22 Query / Pattern / O-Proj Input Decomposition
```

测试目标：

```text
判断 L22 attention 强修复来自 query state、attention pattern，还是 o_proj input head mixture。
```

测试方案：

```text
1. 固定 DS7B、L22、digit1 answer_prefix 位置。

2. 捕获 base / repair 的：
   - q_proj output at answer_prefix
   - k_proj output by source positions
   - v_proj output by source positions
   - o_proj input at answer_prefix
   - attention output after o_proj

3. 做四类 patch：
   - query-only patch: 替换 answer_prefix 的 Q
   - o_proj_input patch: 替换 answer_prefix 的 o_proj input
   - head-slot patch: 按 head 替换 o_proj input slot
   - pattern-preserving V patch: 固定 base attention pattern，只替换 V mixture 或反过来

4. 对照：
   - random same-norm Q
   - random same-norm o_proj input
   - wrong prompt Q / o_proj input

5. 评价：
   - switch
   - margin_gain
   - correct_delta
   - old_wrong_delta
   - head-level cumulative curve
```

预期判据：

```text
如果 query-only 有效：瓶颈是 answer_prefix query condition。
如果 o_proj_input 有效但 Q/K/V source 无效：有效单位是多头混合场。
如果少数 head-slot 有效：可以继续做 head-level causal graph。
如果只有全 o_proj_input 有效：说明 value gate 是分布式多头组合，不宜继续追单头。
```

## Phase 609: Query O-Proj Head Decomposition 查询-输出投影输入-逐头槽位分解 [2026-06-24 17:50]

### 本阶段目标

Phase 608 得到强负结果：

```text
DS7B L22 attention output 整体有效，
但单个 source group 的 K/V delta 几乎无效。
```

关键对比：

```text
Phase 607 DS7B L22 attn_out: 33/37, margin +3.423
Phase 608 DS7B rule_value kv_delta: 0/37, margin -0.028
```

因此 Phase 609 不再继续扩大 source K/V 搜索，而是直接测试 attention 的更真实计算单位：

```text
1. query state
2. o_proj input mixture
3. per-head o_proj input slot
```

核心问题：

```text
L22 attention 的强因果效果到底来自 query，还是来自 o_proj input 的多头混合场？
```

### 附加分析判断

附件对 Phase 608 的判断基本正确：

```text
1. Phase 608 是强负结果，但没有推翻 Phase 607。
2. L22 attention output 整体仍然是有效写入器。
3. single source K/V 不是有效单位。
4. rule_value token -> V copy -> digit1 的简单模型被排除。
5. 下一步应直接测试 query / attention pattern / o_proj input / head mixture。
```

Phase 609 对这一路径进行了第一轮因果定位。

### 执行命令

```bash
python tests/glm5/phase609_query_oproj_head_decomposition.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase609_query_oproj_head_decomposition \
  --hard-exit-after-model

python tests/glm5/phase609_query_oproj_head_decomposition.py glm4 \
  --confirm \
  --output-dir results/glm5_phase609_query_oproj_head_decomposition \
  --hard-exit-after-model

python tests/glm5/phase609_query_oproj_head_decomposition.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase609_query_oproj_head_decomposition \
  --hard-exit-after-model

python tests/glm5/phase609_query_oproj_head_decomposition_summary.py

python -m py_compile \
  tests/glm5/phase609_query_oproj_head_decomposition.py \
  tests/glm5/phase609_query_oproj_head_decomposition_summary.py
```

### 脚本与结果

- 主脚本：`tests/glm5/phase609_query_oproj_head_decomposition.py`
- 汇总脚本：`tests/glm5/phase609_query_oproj_head_decomposition_summary.py`
- Qwen3 结果：`results/glm5_phase609_query_oproj_head_decomposition/phase609_qwen3_query_oproj_head_decomposition_confirm.json`
- GLM4 结果：`results/glm5_phase609_query_oproj_head_decomposition/phase609_glm4_query_oproj_head_decomposition_confirm.json`
- DS7B 结果：`results/glm5_phase609_query_oproj_head_decomposition/phase609_deepseek7b_query_oproj_head_decomposition_confirm.json`
- 跨模型汇总：`results/glm5_phase609_query_oproj_head_decomposition/phase609_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
cases = 96
rows = base wrong and repair correct target cases only
candidate values = v05, v91, v22, v48
```

测试层位：

```text
Qwen3: L29, 32 heads
GLM4: L34, 32 heads
DS7B: L22, 28 heads
```

patch 类型：

```text
q_delta
q_random
o_input_delta
o_input_random
head_delta for each head slot
head_random for each head slot
```

### 测试原理

对每个候选答案，在 base prompt 与 repair prompt 中捕获同一 attention 层的：

```text
q_proj output at answer_prefix position
o_proj input at answer_prefix position
attention output after o_proj
```

其中 answer_prefix position 是模型已经生成共同前缀：

```text
" v"
```

之后，用该位置隐藏状态预测下一位数字。

测试四类补丁：

```text
q_delta:
  在 base forward 中给 answer_prefix 的 q_proj output 加上 repair-base delta。

o_input_delta:
  在 base forward 中给 answer_prefix 的 o_proj input 加上 repair-base delta。

head_delta:
  只给某个 attention head 对应的 o_proj input slot 加 repair-base delta。

random:
  同范数随机对照。
```

然后重新计算四个候选 value 的完整答案分数，测：

```text
switch
margin_gain
correct_delta
old_wrong_delta
```

### 客观结果

#### Qwen3

```text
rows = 7
layer = L29
heads = 32
```

核心结果：

```text
o_input_delta: 6/7, margin +2.055
head_delta H11: 5/7, margin +1.894
q_delta: 3/7, margin +1.393
o_input_random: 0/7, margin -0.020
q_random: 1/7, margin -0.065
```

判断：

```text
Qwen3 L29 的有效单位主要在 o_proj input，且 H11 是非常强的单头槽位候选。
```

注意：Phase 607 显示 Qwen3 的扫描窗口偏晚，因此 L29 未必是首次写入层，但 L29 确实存在强 o_proj input / head-slot 可修复结构。

#### GLM4

```text
rows = 13
layer = L34
heads = 32
```

核心结果：

```text
o_input_delta: 3/13, margin +0.173
head_delta H12: 1/13, margin +0.125
head_delta H8:  1/13, margin +0.067
head_delta H4:  1/13, margin +0.063
q_delta: 1/13, margin -0.048
o_input_random: 0/13, margin +0.005
```

判断：

```text
GLM4 有弱 o_proj input 正效应，但没有形成强机制闭合。
```

这与前面 GLM4 value gate 效应整体偏弱一致。

#### DS7B

```text
rows = 37
layer = L22
heads = 28
```

核心结果：

```text
o_input_delta: 33/37, margin +3.428
head_delta H3: 16/37, margin +1.516
head_delta H1: 8/37, margin +0.759
head_delta H7: 5/37, margin +0.547
head_delta H24: 4/37, margin +0.229
q_delta: 2/37, margin +0.121
o_input_random: 0/37, margin +0.024
q_random: 1/37, margin -0.118
```

关键对比：

```text
Phase 607 DS7B L22 attn_out:    33/37, margin +3.423
Phase 609 DS7B L22 o_input_delta: 33/37, margin +3.428
```

判断：

```text
DS7B L22 attention 的强因果效果几乎完全定位到 o_proj input mixture。
```

这说明 Phase 607 的 attn_out 效果不是 W_O 之后才出现的，也不是单 source K/V，而是在进入 o_proj 前的多头混合槽位中已经形成。

### 当前最可靠客观事实

1. Phase 609 找回了 Phase 607 的强效果。

DS7B：

```text
attn_out patch ≈ o_proj_input patch
33/37 vs 33/37
+3.423 vs +3.428
```

2. DS7B 的瓶颈不在 query-only。

```text
q_delta: 2/37, margin +0.121
```

query delta 有弱效应，但远远不能解释完整 L22 attention 修复。

3. DS7B 的有效单位是 o_proj input 多头混合场。

```text
o_input_delta: 33/37, margin +3.428
```

这是目前 value gate 路线中最清晰的机制定位之一。

4. DS7B 中存在强单头候选，但单头不足以完全闭合。

```text
H3:  16/37, margin +1.516
H1:   8/37, margin +0.759
H7:   5/37, margin +0.547
H24:  4/37, margin +0.229
```

说明：

```text
H3 是主贡献头，但完整效应需要多头组合。
```

5. Qwen3 出现类似结构，但样本量较小。

```text
Qwen3 L29 o_input_delta: 6/7, margin +2.055
Qwen3 H11: 5/7, margin +1.894
```

说明 Qwen3 也可能存在强单头槽位，但需要更早层位和更大 target set 继续确认。

6. GLM4 效应弱。

```text
GLM4 o_input_delta: 3/13, margin +0.173
```

它不是完全无效，但不构成强闭合。

### 理论进展

Phase 608 后的模型是：

```text
conditioned query state + distributed source field + attention pattern
-> o_proj input mixture
-> attention output
-> digit1 residual trajectory
```

Phase 609 进一步收紧为：

```text
o_proj input mixture 是当前可因果修复的最小强单位。
```

更精确地说：

```text
M_l(t)
= concat_h z_{l,h,t}
```

其中：

```text
z_{l,h,t}
= sum_{s <= t} alpha_{l,h,t,s} V_{l,h,s}
```

attention output 是：

```text
A_l(t) = W_O^l M_l(t)
```

Phase 609 证明：

```text
Patch(M_{22}(digit1)) ≈ Patch(A_{22}(digit1))
```

而 Phase 608 证明：

```text
Patch(single source K/V) 无法恢复 M_{22}(digit1)
```

因此当前 value gate 的关键结构不是：

```text
single source value copy
```

而是：

```text
multi-head mixture state at answer_prefix / digit1 position
```

当前最可靠的局部链条：

```text
answer_prefix position
-> L22 multi-head o_proj input mixture
-> L22 attention output
-> L22/L23 residual trajectory
-> final digit1 readout
```

DS7B 的可观测强链条：

```text
L22 o_proj input: 33/37, margin +3.428
L22 attn_out:    33/37, margin +3.423
L22 layer_out:   33/37, margin +4.726
L23 layer_out:   37/37, margin +6.287
```

### 问题和硬伤

1. head_delta 是单头替换，不是多头累积曲线。

虽然 H3 很强：

```text
16/37, margin +1.516
```

但还不知道 H3+H1+H7+H24 是否能累积接近完整 o_input_delta。

2. 当前没有直接拆 attention pattern。

o_proj input 是：

```text
attention pattern × V
```

Phase 609 定位到 o_proj input，但还没有区分：

```text
attention weights 变化
V 内容变化
二者组合变化
```

3. q_delta 弱，不代表 query 不重要。

query 可能通过改变 attention pattern 产生非线性效果，单独加 q_proj delta 未必能稳定重建 repair pattern。

4. Qwen3 的 target rows 只有 7。

虽然效果强，但样本量小，需要更大任务或更早层扫描确认。

5. GLM4 仍然弱。

GLM4 的机制可能更分散，或者当前 target cases 不适合形成强 value gate 修复。

### 下一步任务

Phase 610 应继续推进 DS7B，因为 DS7B 的机制链条最清晰：

```text
Phase 610: DS7B L22 Head Cumulative Mixture and Pattern Split
```

测试目标：

```text
判断完整 o_proj input 效果是否由少数 head 累积构成，
以及 H3/H1/H7/H24 的贡献是否稳定可加。
```

测试方案：

```text
1. 固定 DS7B L22。

2. 使用 Phase 609 排名头：
   H3, H1, H7, H24, H25, H13。

3. 做 cumulative head-slot patch：
   - H3
   - H3+H1
   - H3+H1+H7
   - H3+H1+H7+H24
   - top6 heads
   - all heads

4. 加对照：
   - same number random heads
   - worst heads
   - random same-norm slots

5. 如果 top heads 累积接近 all heads，说明 value gate 是稀疏多头组合。

6. 如果 top heads 不能接近 all heads，说明完整效应依赖广泛分布式多头场。
```

后续 Phase 611 再做 pattern split：

```text
固定 V，替换 attention pattern；
固定 pattern，替换 V；
判断 o_proj input 的差异主要来自 routing 还是 content。
```

## Phase 610: Head Cumulative Mixture 逐头累积混合测试 [2026-06-24 18:35]

### 本阶段目标

Phase 609 已经把 DS7B L22 attention 的强因果效果定位到：

```text
o_proj input mixture
```

关键结果：

```text
DS7B L22 o_input_delta: 33/37, margin +3.428
DS7B L22 H3: 16/37, margin +1.516
```

Phase 610 的目标是判断：

```text
完整 o_proj input 效果是否由少数强 head 累积构成，
还是必须依赖广泛分布式的 all-head field。
```

### 附加分析判断

附件对 Phase 609 的判断基本正确：

```text
1. Phase 609 是关键正结果。
2. DS7B L22 attention output 的强因果效果已经定位到 o_proj input mixture。
3. query-only 不是主因。
4. H3 是强头，但单头不足以闭合全部效果。
5. 下一步应做 top-head cumulative patch。
```

Phase 610 正是对该判断进行累积验证。

### 执行命令

```bash
python tests/glm5/phase610_head_cumulative_mixture.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase610_head_cumulative_mixture \
  --hard-exit-after-model

python tests/glm5/phase610_head_cumulative_mixture.py glm4 \
  --confirm \
  --output-dir results/glm5_phase610_head_cumulative_mixture \
  --hard-exit-after-model

python tests/glm5/phase610_head_cumulative_mixture.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase610_head_cumulative_mixture \
  --hard-exit-after-model

python tests/glm5/phase610_head_cumulative_mixture_summary.py

python -m py_compile \
  tests/glm5/phase610_head_cumulative_mixture.py \
  tests/glm5/phase610_head_cumulative_mixture_summary.py
```

### 脚本与结果

- 主脚本：`tests/glm5/phase610_head_cumulative_mixture.py`
- 汇总脚本：`tests/glm5/phase610_head_cumulative_mixture_summary.py`
- Qwen3 结果：`results/glm5_phase610_head_cumulative_mixture/phase610_qwen3_head_cumulative_mixture_confirm.json`
- GLM4 结果：`results/glm5_phase610_head_cumulative_mixture/phase610_glm4_head_cumulative_mixture_confirm.json`
- DS7B 结果：`results/glm5_phase610_head_cumulative_mixture/phase610_deepseek7b_head_cumulative_mixture_confirm.json`
- 跨模型汇总：`results/glm5_phase610_head_cumulative_mixture/phase610_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
cases = 96
rows = base wrong and repair correct target cases only
candidate values = v05, v91, v22, v48
```

测试层位：

```text
Qwen3: L29, 32 heads
GLM4: L34, 32 heads
DS7B: L22, 28 heads
```

Phase 609 得到的 top heads：

```text
Qwen3: H11, H23, H6, H14, H5, H2
GLM4: H12, H8, H4, H28, H6, H7
DS7B: H3, H1, H7, H24, H25, H13
```

测试 patch：

```text
top1_delta
top2_delta
top3_delta
top4_delta
top6_delta
all_delta
```

对照：

```text
topN_random_slots
weakN_delta
randheadsN_delta
all_random_slots
```

### 测试原理

Phase 609 证明：

```text
Patch(all heads o_proj input) 强有效。
```

Phase 610 不再一次性 patch 全部 o_proj input，而是只 patch 某些 head slot：

```text
M_t^l = concat(z_{1,t}^l, z_{2,t}^l, ..., z_{H,t}^l)
```

对指定 head set S：

```text
Patch_S(M_t^l)
= M_t^l + sum_{h in S} Delta z_{h,t}^l
```

其中：

```text
Delta z_h = z_h(repair) - z_h(base)
```

如果 top heads 累积能接近 all_delta，说明机制是稀疏多头组合。
如果必须 all heads 才有效，说明机制是广泛分布式多头场。

### 客观结果

#### Qwen3

```text
rows = 7
layer = L29
heads = 32
```

累积曲线：

```text
top1 H11:                5/7, margin +1.894
top2 H11,H23:            5/7, margin +1.876
top3 H11,H23,H6:         7/7, margin +2.269
top4 H11,H23,H6,H14:     7/7, margin +2.251
top6 H11,H23,H6,H14,H5,H2: 7/7, margin +2.305
all_delta:               6/7, margin +2.055
```

对照：

```text
all_random_slots: 0/7, margin -0.098
weak6_delta:      0/7, margin -0.286
top6_random_slots: 2/7, margin +0.120
```

判断：

```text
Qwen3 L29 的有效信号高度集中在 top heads，top3/top6 已经达到甚至超过 all_delta。
```

注意：Qwen3 target rows 只有 7，结论仍需扩大数据确认。

#### GLM4

```text
rows = 13
layer = L34
heads = 32
```

累积曲线：

```text
top1 H12:            1/13, margin +0.125
top2 H12,H8:         2/13, margin +0.202
top3 H12,H8,H4:      3/13, margin +0.288
top4 H12,H8,H4,H28:  3/13, margin +0.279
top6 H12,H8,H4,H28,H6,H7: 3/13, margin +0.308
all_delta:           3/13, margin +0.173
```

对照：

```text
all_random_slots: 1/13, margin +0.018
weak6_delta:      0/13, margin -0.226
```

判断：

```text
GLM4 也有 top-head 累积模式，但整体效应弱，不能作为强机制闭合证据。
```

#### DS7B

```text
rows = 37
layer = L22
heads = 28
```

累积曲线：

```text
top1 H3:              16/37, margin +1.516
top2 H3,H1:           28/37, margin +2.223
top3 H3,H1,H7:        31/37, margin +2.670
top4 H3,H1,H7,H24:    32/37, margin +2.932
top6 H3,H1,H7,H24,H25,H13: 32/37, margin +3.085
all_delta:            33/37, margin +3.428
```

对照：

```text
top1_random_slots: 0/37, margin +0.049
top2_random_slots: 0/37, margin +0.086
top6_random_slots: 1/37, margin +0.074
weak6_delta:       0/37, margin -0.005
all_random_slots:  1/37, margin -0.154
```

判断：

```text
DS7B L22 value gate 是稀疏多头组合。
H3 是主头，但 H3+H1+H7 已经恢复 31/37，top4/top6 基本接近 all-head。
```

### 当前最可靠客观事实

1. DS7B 的完整 o_proj input 效果主要由少数 top heads 构成。

```text
top3: 31/37, margin +2.670
all:  33/37, margin +3.428
```

2. H3 是主贡献头，但不是唯一头。

```text
H3: 16/37
H3+H1: 28/37
H3+H1+H7: 31/37
```

3. 弱头和随机槽位基本无效。

```text
weak6: 0/37
all_random_slots: 1/37
```

说明这不是任意同范数扰动造成的。

4. Qwen3 出现更强的稀疏头现象。

```text
top3: 7/7
all:  6/7
```

但样本较小，不能和 DS7B 等强度看待。

5. GLM4 有同方向但弱得多的模式。

```text
top3: 3/13
all:  3/13
```

### 理论进展

Phase 609 的模型是：

```text
o_proj input mixture 是强因果单位。
```

Phase 610 进一步收紧：

```text
对 DS7B 来说，o_proj input mixture 不是均匀全头场，
而是少数 head slot 的稀疏组合。
```

当前 DS7B value gate 局部链条可写为：

```text
H3 + H1 + H7 + H24 + ...
-> L22 o_proj input mixture
-> L22 attention output
-> L22/L23 residual trajectory
-> digit1 selection
```

更形式化：

```text
M_{22,d1}
= concat_h z_{h}
```

有效子集：

```text
S* = {H3, H1, H7, H24, H25, H13}
```

并且：

```text
Patch(M_{S*}) ≈ Patch(M_all)
```

其中：

```text
Patch(M_{S*}): 32/37, margin +3.085
Patch(M_all):  33/37, margin +3.428
```

这意味着 value gate 的局部机制已经从：

```text
unknown attention output
```

推进到：

```text
specific sparse head-slot mixture at answer_prefix position
```

### 问题和硬伤

1. 仍然没有拆开 head 内部的 pattern 与 value content。

每个 head slot：

```text
z_h = sum_s alpha_{h,s} V_{h,s}
```

Phase 610 证明 top head slot 有效，但还不知道有效来自：

```text
attention pattern alpha
V content
pattern × V 的组合
```

2. top heads 来自 Phase 609 排名，存在选择偏差。

虽然有 random / weak controls，但仍需要后续做：

```text
heldout head ranking
跨 seed ranking
不同 case subset ranking
```

3. Qwen3 样本太少。

Qwen3 top3 结果很强，但 target rows 只有 7，不能过度泛化。

4. GLM4 仍弱。

可能是模型机制更分散，也可能是测试任务对 GLM4 不敏感。

5. 当前仍是 repair-base oracle patch。

它证明某些 head slot 足以因果恢复，但还没有解释 base prompt 中为什么这些 head slot 自然失败。

### 下一步任务

Phase 611 应在 DS7B L22 top heads 上做 pattern/content split：

```text
Phase 611: DS7B L22 Top-Head Pattern Content Split
```

测试目标：

```text
判断 H3/H1/H7/H24 的有效差异来自 attention pattern，还是 V content。
```

测试方案：

```text
1. 固定 DS7B L22 top heads: H3, H1, H7, H24。

2. 对每个 target case 捕获：
   - attention weights alpha_base / alpha_repair
   - V_base / V_repair
   - head output z_base / z_repair

3. 重构四类 head output：
   - alpha_base * V_base
   - alpha_repair * V_base
   - alpha_base * V_repair
   - alpha_repair * V_repair

4. patch 到对应 head slot，测试：
   - pattern-only effect
   - content-only effect
   - pattern+content full effect

5. 对照：
   - random attention pattern
   - shuffled source pattern
   - random same-norm head output
```

判据：

```text
如果 alpha_repair * V_base 有效：主要是 routing/pattern。
如果 alpha_base * V_repair 有效：主要是 value content。
如果只有 alpha_repair * V_repair 有效：pattern 与 content 必须耦合。
```

## Phase 611: Semantic Pattern Content Split 语义源组模式-内容拆分 [2026-06-24 19:02]

### 本阶段目标

Phase 610 已经证明 DS7B L22 value gate 是稀疏多头 o_proj input mixture：

```text
H3+H1+H7+H24: 32/37, margin +2.932
all heads:    33/37, margin +3.428
```

Phase 611 原计划拆分：

```text
attention pattern vs V content
```

但执行前发现一个重要硬伤：

```text
base prompt 和 repair prompt 不是 token-index aligned。
```

base prompt 是完整规则表，repair prompt 是 relation-filter 后的短规则表。直接做：

```text
alpha_repair * V_base
```

会把不同 prompt 中不同含义的 token index 硬配在一起，因此不严格。

本阶段采用更谨慎的替代测试：

```text
semantic source-group pattern/content split
```

也就是按语义源组对 attention mass 和 V content 做近似组合，而不是按原始 token index 强行对齐。

### 附加分析判断

附件对 Phase 610 的判断基本正确：

```text
1. Phase 610 是强正结果。
2. DS7B L22 value gate 是 sparse multi-head o_proj input mixture。
3. H3 是主头，H1/H7/H24 等提供增量。
4. weak heads 和 random slots 无效。
5. 下一步必须拆 pattern 和 content。
```

需要补充的关键限制：

```text
由于 base/repair prompt 源序列不对齐，严格的 pattern/content token-index split 目前不能直接做。
```

因此 Phase 611 是“语义源组近似拆分”，不是最终闭合。

### 执行命令

```bash
python tests/glm5/phase611_semantic_pattern_content_split.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase611_semantic_pattern_content_split \
  --hard-exit-after-model

python tests/glm5/phase611_semantic_pattern_content_split.py glm4 \
  --confirm \
  --output-dir results/glm5_phase611_semantic_pattern_content_split \
  --hard-exit-after-model

python tests/glm5/phase611_semantic_pattern_content_split.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase611_semantic_pattern_content_split \
  --hard-exit-after-model

python tests/glm5/phase611_semantic_pattern_content_split_summary.py

python -m py_compile \
  tests/glm5/phase611_semantic_pattern_content_split.py \
  tests/glm5/phase611_semantic_pattern_content_split_summary.py
```

### 脚本与结果

- 主脚本：`tests/glm5/phase611_semantic_pattern_content_split.py`
- 汇总脚本：`tests/glm5/phase611_semantic_pattern_content_split_summary.py`
- Qwen3 结果：`results/glm5_phase611_semantic_pattern_content_split/phase611_qwen3_semantic_pattern_content_split_confirm.json`
- GLM4 结果：`results/glm5_phase611_semantic_pattern_content_split/phase611_glm4_semantic_pattern_content_split_confirm.json`
- DS7B 结果：`results/glm5_phase611_semantic_pattern_content_split/phase611_deepseek7b_semantic_pattern_content_split_confirm.json`
- 跨模型汇总：`results/glm5_phase611_semantic_pattern_content_split/phase611_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
cases = 96
rows = base wrong and repair correct target cases only
candidate values = v05, v91, v22, v48
semantic groups = answer_prefix, prompt_last, rule_value, rule_relation, query_category, query_relation, query_object, other
top_k heads = 4
```

测试层位与头集合：

```text
Qwen3: L29, heads H11,H23,H6,H14
GLM4: L34, heads H12,H8,H4,H28
DS7B: L22, heads H3,H1,H7,H24
```

### 测试原理

真实 head output：

```text
z_h = sum_s alpha_{h,s} V_{h,s}
```

因为 base / repair 源序列不对齐，本阶段把 source token 划入语义组：

```text
rule_value
rule_relation
query_category
query_relation
query_object
prompt_last
answer_prefix
other
```

然后构造四类近似补丁：

```text
actual:
  直接使用 repair 的真实 top-head o_proj input slot。
  这是上限参照。

content:
  使用 base attention group mass，替换为 repair semantic group V content。

pattern:
  使用 repair attention group mass，保留 base group V content。

pattern_content:
  使用 repair group mass 和 repair group V content。

random:
  同范数随机 head slot 对照。
```

注意：

```text
content / pattern / pattern_content 是 semantic-group approximation，
不是严格 token-index reconstruction。
```

### 客观结果

#### Qwen3

```text
rows = 7
layer = L29
top heads = H11,H23,H6,H14
```

结果：

```text
actual:          7/7, margin +2.251
content:         1/7, margin +0.161
pattern:         0/7, margin +0.089
pattern_content: 1/7, margin +0.143
random:          1/7, margin +0.141
```

判断：

```text
Qwen3 的真实 top-head slot 很强，
但语义组 pattern/content 近似几乎不能解释该效果。
```

#### GLM4

```text
rows = 13
layer = L34
top heads = H12,H8,H4,H28
```

结果：

```text
actual:          3/13, margin +0.279
content:         10/13, margin +1.192
pattern:         1/13, margin +0.053
pattern_content: 10/13, margin +1.202
random:          0/13, margin -0.028
```

判断：

```text
GLM4 的 content / pattern_content 超过 actual，
说明语义组均值 V content 构造可能产生非自然增强。
这不能解释为真实机制闭合，只能说明该近似方法存在过强人工构造效应。
```

#### DS7B

```text
rows = 37
layer = L22
top heads = H3,H1,H7,H24
```

结果：

```text
actual:          32/37, margin +2.932
content:          9/37, margin +0.705
pattern:          5/37, margin +0.480
pattern_content: 17/37, margin +1.215
random:           0/37, margin -0.056
```

判断：

```text
DS7B 的语义组 pattern+content 能恢复一部分，但远低于真实 top-head actual。
content 强于 pattern，但 content 单独也不能闭合。
```

### 当前最可靠客观事实

1. DS7B top-head actual patch 仍然强，复现 Phase 610。

```text
DS7B top4 actual: 32/37, margin +2.932
```

2. DS7B 语义组近似只能解释部分效果。

```text
pattern_content: 17/37, margin +1.215
actual:          32/37, margin +2.932
```

说明：

```text
真实 head output 依赖更细粒度 token-level pattern/content 结构，
不是几个粗语义源组就能闭合。
```

3. DS7B 中 content 近似强于 pattern 近似。

```text
content: 9/37, margin +0.705
pattern: 5/37, margin +0.480
```

但二者都不足以闭合。

4. GLM4 的异常强 content 结果提示方法风险。

```text
GLM4 content: 10/13 > actual: 3/13
```

这说明语义组均值 V content patch 可能构造出模型自然 forward 中不存在的强信号。

5. Qwen3 的语义组近似失败。

```text
actual: 7/7
pattern_content: 1/7
```

说明 Qwen3 强头机制可能更依赖精确 token-level mixture，而不是粗语义组。

### 理论进展

Phase 611 没有完成严格 pattern/content 闭合，但它给出了一个重要边界：

```text
coarse semantic source groups are not enough to reconstruct the true top-head mixture。
```

当前理论应从：

```text
top heads read semantic source groups
```

修正为：

```text
top heads construct token-level weighted mixture under semantic constraints
```

也就是说，head slot 的有效信息不是简单的：

```text
看 rule_value
看 query_relation
看 answer_prefix
```

而是更细的：

```text
在具体 token 序列上形成 weighted mixture geometry。
```

当前链条：

```text
semantic task condition
-> token-level source field
-> top-head weighted mixture
-> o_proj input slot
-> attention output
-> residual trajectory
-> digit1 readout
```

### 问题和硬伤

1. 本阶段不是严格 pattern/content split。

因为 base/repair prompt 不对齐，无法直接做：

```text
alpha_repair * V_base
alpha_base * V_repair
```

2. semantic group approximation 会产生人工构造风险。

GLM4 的 content 超过 actual 就是警告。

3. source groups 太粗。

真实机制可能依赖：

```text
整条 relevant rule line
多个 distractor rule lines
行内 token 相对位置
标点和换行
候选值之间的相对排序
```

4. 仍然没有解释 base prompt 中 top heads 为什么失败。

Phase 611 只是说明粗语义组不足以重建 repair head mixture。

5. 下一步必须使用 source-aligned prompt pair。

要严格拆 pattern/content，必须构造同长度、同 token index 结构的 base/repair 对照。

### 下一步任务

Phase 612 应从数据设计上解决对齐问题：

```text
Phase 612: Source-Aligned Pattern Content Split
```

目标：

```text
构造 base / repair prompt 源序列 token-index 对齐的任务，
然后严格测试 alpha pattern 与 V content。
```

方案：

```text
1. 固定规则表结构、规则数量、行顺序、token 长度。

2. 构造 aligned base / repair：
   - 两者 token positions 完全对应；
   - 只改变 relation focus 或 category focus；
   - 保持 rule lines 数量一致；
   - 保持 answer prefix 一致。

3. 在 DS7B L22 top heads 上捕获：
   - alpha_base
   - alpha_repair
   - V_base
   - V_repair

4. 做严格重构：
   - alpha_base * V_base
   - alpha_repair * V_base
   - alpha_base * V_repair
   - alpha_repair * V_repair

5. patch 到 H3/H1/H7/H24，测试：
   - pattern-only
   - content-only
   - pattern+content
   - full actual
```

判据：

```text
如果 alpha_repair * V_base 有效：主要是 pattern/routing。
如果 alpha_base * V_repair 有效：主要是 V content。
如果只有 alpha_repair * V_repair 有效：pattern 和 content 必须耦合。
如果仍不能接近 actual：还缺少 position-level 或 normalization-level 因子。
```

## Phase 612: Source-Aligned Pattern Content Split 源序列对齐模式-内容拆分 [2026-06-24 20:09]

### 本阶段目标

根据用户要求，先分析 Phase611 附件判断是否正确，再综合当前进展继续完成任务。

附件对 Phase611 的判断基本正确：

```text
1. Phase611 方向正确，但必须保守解释。
2. Phase611 不是严格 pattern/content split，而是 semantic source group approximation。
3. DS7B actual 32/37 明显强于 pattern_content 17/37，说明粗语义组不足以重构真实 top-head mixture。
4. GLM4 content 强于 actual 是方法风险，不是机制闭合。
5. 下一步必须构造 source-aligned prompt pair，严格测试 alpha pattern 与 V content。
```

Phase612 因此不继续扩大语义组，而是从数据设计上解决 Phase611 的硬伤：

```text
base prompt 与 repair prompt 必须 token-index aligned。
```

### 执行命令

```bash
python -m py_compile \
  tests/glm5/phase612_source_aligned_pattern_content_split.py \
  tests/glm5/phase612_source_aligned_pattern_content_split_summary.py

python tests/glm5/phase612_source_aligned_pattern_content_split.py qwen3 \
  --smoke \
  --hard-exit-after-model

python tests/glm5/phase612_source_aligned_pattern_content_split.py qwen3 \
  --confirm \
  --include-nontarget \
  --output-dir results/glm5_phase612_source_aligned_pattern_content_split \
  --hard-exit-after-model

python tests/glm5/phase612_source_aligned_pattern_content_split.py glm4 \
  --confirm \
  --include-nontarget \
  --output-dir results/glm5_phase612_source_aligned_pattern_content_split \
  --hard-exit-after-model

python tests/glm5/phase612_source_aligned_pattern_content_split.py deepseek7b \
  --confirm \
  --include-nontarget \
  --output-dir results/glm5_phase612_source_aligned_pattern_content_split \
  --hard-exit-after-model

python tests/glm5/phase612_source_aligned_pattern_content_split_summary.py

python -m py_compile \
  tests/glm5/phase612_source_aligned_pattern_content_split.py \
  tests/glm5/phase612_source_aligned_pattern_content_split_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/glm5/phase612_source_aligned_pattern_content_split.py`
- 汇总脚本：`tests/glm5/phase612_source_aligned_pattern_content_split_summary.py`
- Qwen3 结果：`results/glm5_phase612_source_aligned_pattern_content_split/phase612_qwen3_source_aligned_pattern_content_split_confirm.json`
- GLM4 结果：`results/glm5_phase612_source_aligned_pattern_content_split/phase612_glm4_source_aligned_pattern_content_split_confirm.json`
- DS7B 结果：`results/glm5_phase612_source_aligned_pattern_content_split/phase612_deepseek7b_source_aligned_pattern_content_split_confirm.json`
- 跨模型汇总：`results/glm5_phase612_source_aligned_pattern_content_split/phase612_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
raw cases/model = 128
token length mismatch = 0 for all models
qwen3 target rows = 9
glm4 target rows = 12
deepseek7b target rows = 43
layers:
  qwen3 L29
  glm4 L34
  deepseek7b L22
top_k heads = 4
qwen3 heads = [11, 23, 6, 14]
glm4 heads = [12, 8, 4, 28]
deepseek7b heads = [3, 1, 7, 24]
```

本轮使用 `--include-nontarget`，因为 source-aligned prompt 的 target case 比例并不高。结果同时输出：

```text
summary: all rows
target_summary: only base wrong + repair correct rows
```

### 测试原理

Phase611 的问题是 base/repair prompt 不对齐，不能直接做：

```text
alpha_repair * V_base
```

Phase612 构造严格源对齐 prompt：

```text
1. base 和 repair 使用同一个 rule block。
2. OC rules 和 CRV rules 行顺序完全相同。
3. 显式 gold category line 完全相同。
4. Answer prefix 完全相同。
5. 只把 Question slot 从 object 改为 category。
6. 运行时过滤 token length mismatch。
```

然后在相同 source token index 上测试：

```text
bb = alpha_base   * V_base
rb = alpha_repair * V_base
br = alpha_base   * V_repair
rr = alpha_repair * V_repair
actual = repair top-head o_proj input
random_actual_norm = same-norm random control
```

如果 `rb_pattern` 接近 `actual`，说明主要是 attention routing / pattern。
如果 `br_content` 接近 `actual`，说明主要是 V content。
如果只有 `rr_pattern_content` 接近 `actual`，说明 pattern 和 content 必须耦合。

### 客观结果

#### Qwen3 target rows

```text
actual:             6/9,  margin +1.557
rr_pattern_content: 5/9,  margin +1.557
rb_pattern:         5/9,  margin +1.529
br_content:         1/9,  margin -0.014
bb:                 1/9,  margin +0.014
random_actual_norm: 1/9,  margin +0.105
```

Qwen3 在严格对齐后显示：

```text
repair pattern + base V 几乎复现 actual。
base pattern + repair V 无效。
```

#### GLM4 target rows

```text
actual:             1/12, margin +0.062
rr_pattern_content: 1/12, margin +0.089
rb_pattern:         1/12, margin +0.063
br_content:         0/12, margin -0.021
bb:                 0/12, margin +0.000
random_actual_norm: 1/12, margin +0.011
```

GLM4 效应仍然弱，不能作为强机制闭合证据。但 GLM4 也没有支持 content-only。

#### DS7B target rows

```text
actual:             31/43, margin +1.709
rr_pattern_content: 31/43, margin +1.708
rb_pattern:         32/43, margin +1.709
br_content:         1/43,  margin -0.013
bb:                 0/43,  margin -0.004
random_actual_norm: 1/43,  margin -0.038
```

DS7B 是本轮最关键结果：

```text
rb_pattern ≈ actual ≈ rr_pattern_content
br_content ≈ bb ≈ random control
```

这说明在严格 token-index aligned 条件下，DS7B L22 top-head value gate 的主因不是 V content 替换，而是 attention routing pattern 替换。

### 当前最可靠客观事实

1. **Phase611 的 content 线索被 Phase612 修正**

Phase611 的 DS7B 语义组结果是：

```text
content 9/37 > pattern 5/37
```

但 Phase612 严格对齐后是：

```text
rb_pattern 32/43, margin +1.709
br_content 1/43, margin -0.013
```

因此 Phase611 的 content 优势很可能来自 semantic-group mean V 人工构造，不是 token-level 真实机制。

2. **DS7B L22 top-head mixture 主要由 attention pattern / routing 控制**

在 DS7B：

```text
alpha_repair * V_base
```

几乎等于：

```text
alpha_repair * V_repair
repair actual o_proj input
```

说明 V_base 本身已经包含足够 value content，关键失败在于 base query/routing 没有把正确 source mixture 读出来。

3. **Qwen3 支持相同方向，但 target 样本较少**

Qwen3 target rows 只有 9 个，但结果方向与 DS7B 一致：

```text
rb_pattern 接近 actual
br_content 无效
```

4. **GLM4 仍然弱，不能过度解释**

GLM4 target rows 12 个，actual 只有 1/12。它只能作为弱支持：content-only 没有表现。

5. **source alignment 是 pattern/content split 的必要条件**

Phase612 证明，如果不做 token-index alignment，语义组拆分可能给出方向相反的假象。

### 理论进展

Phase610 的模型是：

```text
value gate = sparse top-head o_proj input mixture
```

Phase611 的边界是：

```text
coarse semantic source group cannot reconstruct top-head mixture
```

Phase612 进一步收紧为：

```text
在 source-aligned 条件下，top-head mixture 的关键变量主要是 alpha routing pattern，
不是 V content 本身。
```

更具体地说：

```text
base prompt 失败，不是因为 source V 中没有正确 value content；
而是因为 answer-position query 形成的 attention pattern 没有选择正确 token-level source mixture。
```

这把 value gate 的瓶颈从：

```text
value content 存在哪里？
```

移动到：

```text
query 如何产生正确 routing pattern？
```

### 硬伤与谨慎解释

1. **aligned prompt 改变了原任务分布**

Phase612 的 prompt 为了严格对齐，把 repair 从 filtered-relation prompt 改成 same-rule-block category-query prompt。它解决了对齐问题，但不是 Phase611 原 prompt 的完全同分布复刻。

2. **target rows 不均衡**

```text
Qwen3 target rows = 9
GLM4 target rows = 12
DS7B target rows = 43
```

DS7B 结论最强；Qwen3 是方向一致但样本偏少；GLM4 仍弱。

3. **rb_pattern 使用 repair alpha，是人为替换 pattern**

这证明 routing pattern 是充分修复因子，但还没有解释 repair alpha 在自然 forward 中如何生成。

4. **还没有分解 Q/K 来源**

如果 pattern 是关键，下一步必须拆：

```text
alpha = softmax(QK^T / sqrt(d))
```

到底是 Q_answer 改变，还是 K_source field 改变，还是二者耦合。

### 最新理论判断

当前 value gate 链条应更新为：

```text
prompt condition
  -> answer-position query state
  -> top-head attention routing pattern
  -> token-level source mixture
  -> o_proj input sparse head slots
  -> digit1 residual trajectory
  -> full candidate margin
```

Phase612 最关键的改进是把 `V content` 从主瓶颈中降级，把 `routing pattern` 提升为当前主瓶颈。

### 下一步任务

Phase613 应继续做：

```text
Q/K Factor Split for Routing Pattern
```

目标：

```text
既然 rb_pattern 能复现 actual，下一步必须解释 repair alpha 从哪里来。
```

测试方案：

```text
1. 继续使用 Phase612 的 source-aligned prompt pairs。
2. 在 DS7B L22 top heads 上捕获：
   - Q_base, Q_repair at answer position
   - K_base, K_repair at source positions
   - V_base, V_repair
3. 构造四种 attention score：
   - Q_base   * K_base
   - Q_repair * K_base
   - Q_base   * K_repair
   - Q_repair * K_repair
4. softmax 后与 V_base 组合，patch top-head o_proj input。
5. 对比 full candidate margin 和 target switch。
```

判据：

```text
如果 Q_repair * K_base 有效：routing 主要由 answer-position query state 控制。
如果 Q_base * K_repair 有效：routing 主要由 source key field 控制。
如果只有 Q_repair * K_repair 有效：Q/K 必须耦合。
如果都不接近 repair alpha：还缺少 mask、position bias、normalization 或 attention implementation 细节。
```

## Phase 613: QK Routing Factor Split Q/K路由因子拆分 [2026-06-24 21:41]

### 本阶段目标

根据用户要求，先分析 Phase612 附件判断是否正确，再综合当前进展继续完成任务。

附件对 Phase612 的判断基本正确：

```text
1. Phase612 是关键纠偏正结果。
2. Phase612 修复了 Phase611 的 token-index alignment 硬伤。
3. DS7B L22 top-head mixture 的主瓶颈不是 V content，而是 attention routing pattern。
4. Qwen3 方向一致但 target rows 少；GLM4 效应弱，不能强解释。
5. 下一步必须解释 repair alpha 从哪里来，即拆 Q/K 来源。
```

Phase613 因此不再继续做 V content 搜索，而是直接测试：

```text
repair routing pattern 主要来自 answer-position Q，还是 source K field，还是 Q/K 耦合。
```

### 执行命令

```bash
python -m py_compile \
  tests/glm5/phase613_qk_routing_factor_split.py \
  tests/glm5/phase613_qk_routing_factor_split_summary.py

python tests/glm5/phase613_qk_routing_factor_split.py qwen3 \
  --smoke \
  --include-nontarget \
  --hard-exit-after-model

python tests/glm5/phase613_qk_routing_factor_split.py qwen3 \
  --confirm \
  --include-nontarget \
  --output-dir results/glm5_phase613_qk_routing_factor_split \
  --hard-exit-after-model

python tests/glm5/phase613_qk_routing_factor_split.py glm4 \
  --confirm \
  --include-nontarget \
  --output-dir results/glm5_phase613_qk_routing_factor_split \
  --hard-exit-after-model

python tests/glm5/phase613_qk_routing_factor_split.py deepseek7b \
  --confirm \
  --include-nontarget \
  --output-dir results/glm5_phase613_qk_routing_factor_split \
  --hard-exit-after-model

python tests/glm5/phase613_qk_routing_factor_split_summary.py

python -m py_compile \
  tests/glm5/phase613_qk_routing_factor_split.py \
  tests/glm5/phase613_qk_routing_factor_split_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/glm5/phase613_qk_routing_factor_split.py`
- 汇总脚本：`tests/glm5/phase613_qk_routing_factor_split_summary.py`
- Qwen3 结果：`results/glm5_phase613_qk_routing_factor_split/phase613_qwen3_qk_routing_factor_split_confirm.json`
- GLM4 结果：`results/glm5_phase613_qk_routing_factor_split/phase613_glm4_qk_routing_factor_split_confirm.json`
- DS7B 结果：`results/glm5_phase613_qk_routing_factor_split/phase613_deepseek7b_qk_routing_factor_split_confirm.json`
- 跨模型汇总：`results/glm5_phase613_qk_routing_factor_split/phase613_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
raw cases/model = 128
token length mismatch = 0 for all models
qwen3 target rows = 9
glm4 target rows = 12
deepseek7b target rows = 43
layers:
  qwen3 L29
  glm4 L34
  deepseek7b L22
top_k heads = 4
qwen3 top heads = [11, 23, 6, 14]
glm4 top heads = [12, 8, 4, 28]
deepseek7b top heads = [3, 1, 7, 24]
kv heads:
  qwen3 L29 = 8
  glm4 L34 = 2
  deepseek7b L22 = 4
```

### 测试原理

Phase612 已证明：

```text
alpha_repair * V_base ≈ actual repair top-head slots
```

因此 Phase613 继续拆：

```text
alpha_repair 从哪里来？
```

本轮没有手写 QK attention，而是在模型自然 forward 内 patch q_proj/k_proj/v_proj 输出，让模型自己处理 rotary、position、mask、softmax 和 o_proj。

测试模式：

```text
q_only: repair Q at answer position + base K/V
k_only: base Q + repair K at source positions + base V
qk: repair Q + repair K + base V
v_only: base Q/K + repair V
qv: repair Q + base K + repair V
kv: base Q + repair K/V
qkv: repair Q/K/V
o_actual: direct repair o_proj-input top-head slots
random_o_actual_norm: same-norm random o_input control
```

判据：

```text
如果 q_only 接近 o_actual：routing 主要由 answer-position query state 控制。
如果 k_only 接近 o_actual：routing 主要由 source key field 控制。
如果只有 qk 接近 o_actual：Q/K 必须耦合。
如果 v_only 接近 o_actual：Phase612 的 V content 降级需要修正。
```

### 客观结果

#### Qwen3 target rows

```text
o_actual:             6/9, margin +1.557
q_only:               6/9, margin +1.571
k_only:               1/9, margin +0.028
qk:                   6/9, margin +1.571
v_only:               1/9, margin +0.042
qv:                   6/9, margin +1.585
kv:                   1/9, margin -0.000
qkv:                  5/9, margin +1.557
random_o_actual_norm: 0/9, margin -0.024
```

Qwen3 显示：

```text
q_only ≈ o_actual
k_only / v_only / kv ≈ weak control
```

#### GLM4 target rows

```text
o_actual:             1/12, margin +0.062
q_only:               1/12, margin +0.078
k_only:               0/12, margin -0.021
qk:                   1/12, margin +0.073
v_only:               0/12, margin -0.026
qv:                   1/12, margin +0.052
kv:                   0/12, margin -0.021
qkv:                  1/12, margin +0.073
random_o_actual_norm: 0/12, margin -0.017
```

GLM4 效应仍然很弱，但方向上仍是 q_only 最接近 o_actual。

#### DS7B target rows

```text
o_actual:             31/43, margin +1.709
q_only:               33/43, margin +1.718
k_only:               1/43,  margin -0.025
qk:                   32/43, margin +1.702
v_only:               0/43,  margin -0.043
qv:                   32/43, margin +1.718
kv:                   1/43,  margin -0.041
qkv:                  30/43, margin +1.702
random_o_actual_norm: 0/43,  margin -0.037
```

DS7B 结果非常明确：

```text
q_only >= o_actual
q_only ≈ qk ≈ qv ≈ qkv
k_only ≈ v_only ≈ kv ≈ random control
```

### 当前最可靠客观事实

1. **Phase612 的 routing pattern 主因被 Phase613 进一步定位到 Q**

Phase612 证明：

```text
repair alpha 是关键。
```

Phase613 证明：

```text
repair alpha 的关键来源主要是 answer-position Q。
```

2. **source K field 不是当前主瓶颈**

在 DS7B：

```text
k_only: 1/43, margin -0.025
```

说明 source key field 单独替换不能修复 value gate。

3. **V content 再次被降级**

在 DS7B：

```text
v_only: 0/43, margin -0.043
kv:     1/43, margin -0.041
```

这与 Phase612 的 `br_content` 失败一致。

4. **Qwen3 与 DS7B 方向一致**

Qwen3 target rows 少，但 q_only 也接近 o_actual：

```text
q_only 6/9, margin +1.571
o_actual 6/9, margin +1.557
```

5. **GLM4 仍然不能作为强证据**

GLM4 的数值太弱，只能说明没有反向支持 K/V 主导。

### 理论进展

Phase610：

```text
value gate = sparse top-head o_proj input mixture
```

Phase612：

```text
value gate = attention routing pattern 主导，而不是 V content 主导
```

Phase613：

```text
attention routing pattern 主要由 answer-position query state 控制
```

当前链条更新为：

```text
prompt condition
  -> answer-position query state
  -> top-head attention routing pattern
  -> token-level source mixture over existing V content
  -> sparse o_proj input head slots
  -> digit1 residual trajectory
  -> full candidate margin
```

这把 value gate 的关键瓶颈进一步从：

```text
source field / V content
```

移动到：

```text
answer-position Q state
```

### 硬伤与谨慎解释

1. **q_only 是人为 patch，不等于自然生成解释完成**

q_only 证明 repair Q 是充分修复因子，但还没有解释 repair Q 自然从哪些上游组件生成。

2. **patch 的是 q_proj output，不是 q_proj input**

本轮定位到 q_proj output 层级。下一步需要继续问：

```text
是 q_proj 线性映射本身关键，还是 q_proj 输入 hidden state 已经携带了正确 query state？
```

3. **target rows 仍不均衡**

```text
Qwen3 9
GLM4 12
DS7B 43
```

DS7B 是最可靠模型；Qwen3 是方向确认；GLM4 弱。

4. **aligned prompt 是实验构造，不是全部自然语言分布**

结论当前只覆盖 source-aligned rule task，不应直接泛化到所有语言任务。

### 最新理论判断

对于当前 value gate 任务，最简理论应写成：

```text
答案位置的 query state 是条件化路由控制器。
它决定 top heads 在 source field 中读取哪些 token-level V mixture。
V content 在 base 中已经足够存在；失败不是“内容不存在”，而是“查询状态没有正确路由”。
```

换句话说，当前真正要破解的是：

```text
prompt 如何把任务条件压缩成 answer-position query state。
```

### 下一步任务

Phase614 应继续做：

```text
Answer-Position Query State Builder Audit
```

目标：

```text
解释 repair Q 自然从哪里生成。
```

测试方案：

```text
1. 继续使用 Phase612/613 的 source-aligned prompt pairs。
2. 在 DS7B L22 捕获 q_proj input hidden state 与 q_proj output。
3. 先 patch q_proj input at answer position：
   - hidden_q_input_only
   - q_proj_output_only
   - random_same_norm
4. 如果 hidden_q_input_only 接近 q_proj_output_only，说明上游 hidden state 已经形成 query state。
5. 扫描 L18-L22 的 residual / attn / mlp 对 answer position hidden 的贡献。
6. 找出哪个上游模块写入了 repair query state。
```

判据：

```text
如果 L22 q_proj input patch 有效：瓶颈上移到 answer-position residual state builder。
如果只有 q_proj output patch 有效：q_proj 映射或 norm 后空间是关键转换点。
如果某个上游 attention/MLP patch 有效：进入 query-state builder 的模块级定位。
```

## Phase 614: Query State Builder Audit 查询状态生成器审计 [2026-06-24 22:15]

### 本阶段目标

根据用户要求，先分析 Phase613 附件判断是否正确，再综合当前进展继续完成任务。

附件对 Phase613 的判断基本正确：

```text
1. Phase613 是关键正结果。
2. Phase613 把 Phase612 的 routing pattern 主因继续推进到 answer-position Q。
3. DS7B q_only 33/43 接近并略强于 o_actual 31/43。
4. k_only、v_only、kv 接近随机，说明 K field 与 V content 不是当前主瓶颈。
5. 下一步必须解释 repair Q 自然从哪里生成。
```

Phase614 因此测试：

```text
repair Q 是只在 q_proj output 层面有效，
还是 q_proj input hidden state 已经携带了 repair query state，
甚至 decoder layer input residual state 已经携带更完整的修复状态。
```

### 执行命令

```bash
python -m py_compile \
  tests/glm5/phase614_query_state_builder_audit.py \
  tests/glm5/phase614_query_state_builder_audit_summary.py

python tests/glm5/phase614_query_state_builder_audit.py qwen3 \
  --smoke \
  --include-nontarget \
  --hard-exit-after-model

python tests/glm5/phase614_query_state_builder_audit.py qwen3 \
  --confirm \
  --include-nontarget \
  --output-dir results/glm5_phase614_query_state_builder_audit \
  --hard-exit-after-model

python tests/glm5/phase614_query_state_builder_audit.py glm4 \
  --confirm \
  --include-nontarget \
  --output-dir results/glm5_phase614_query_state_builder_audit \
  --hard-exit-after-model

python tests/glm5/phase614_query_state_builder_audit.py deepseek7b \
  --confirm \
  --include-nontarget \
  --output-dir results/glm5_phase614_query_state_builder_audit \
  --hard-exit-after-model

python tests/glm5/phase614_query_state_builder_audit_summary.py

python -m py_compile \
  tests/glm5/phase614_query_state_builder_audit.py \
  tests/glm5/phase614_query_state_builder_audit_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/glm5/phase614_query_state_builder_audit.py`
- 汇总脚本：`tests/glm5/phase614_query_state_builder_audit_summary.py`
- Qwen3 结果：`results/glm5_phase614_query_state_builder_audit/phase614_qwen3_query_state_builder_audit_confirm.json`
- GLM4 结果：`results/glm5_phase614_query_state_builder_audit/phase614_glm4_query_state_builder_audit_confirm.json`
- DS7B 结果：`results/glm5_phase614_query_state_builder_audit/phase614_deepseek7b_query_state_builder_audit_confirm.json`
- 跨模型汇总：`results/glm5_phase614_query_state_builder_audit/phase614_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
raw cases/model = 128
token length mismatch = 0 for all models
qwen3 target rows = 9
glm4 target rows = 12
deepseek7b target rows = 43
layers:
  qwen3 L29
  glm4 L34
  deepseek7b L22
top_k heads = 4
qwen3 top heads = [11, 23, 6, 14]
glm4 top heads = [12, 8, 4, 28]
deepseek7b top heads = [3, 1, 7, 24]
```

### 测试原理

Phase613 定位到：

```text
q_proj output at answer position 是充分修复因子。
```

Phase614 继续往上测试三个层级：

```text
o_actual:
  直接 patch repair o_proj-input top-head slots。

q_output_top:
  patch q_proj output selected heads，等价于 Phase613 q_only anchor。

q_input_full / q_input_delta:
  patch q_proj input hidden state at answer position。

layer_input_full / layer_input_delta:
  patch decoder layer input residual state at answer position。

q_input_random / layer_input_random / o_random:
  same-norm random controls。
```

判据：

```text
如果 q_input_full 接近 q_output_top：repair query state 已经在 q_proj input hidden 中形成。
如果 layer_input_full 接近或强于 q_input_full：目标层输入残差态已经携带更完整的修复状态。
如果只有 q_output_top 有效：q_proj 映射层才是关键转换点。
```

### 客观结果

#### Qwen3 target rows

```text
o_actual:           6/9, margin +1.557
q_output_top:       6/9, margin +1.571
q_input_full:       6/9, margin +1.585
q_input_delta:      6/9, margin +1.585
layer_input_full:   9/9, margin +4.392
layer_input_delta:  9/9, margin +4.392
q_input_random:     0/9, margin +0.043
layer_input_random: 0/9, margin +0.045
o_random:           0/9, margin -0.012
```

Qwen3 显示：

```text
q_input 已经足以复现 q_output/o_actual。
layer_input 更强，说明 L29 输入残差态携带超出 top-head Q 的额外修复信息。
```

#### GLM4 target rows

```text
o_actual:           1/12, margin +0.062
q_output_top:       1/12, margin +0.078
q_input_full:       0/12, margin -0.063
q_input_delta:      0/12, margin -0.063
layer_input_full:   11/12, margin +1.932
layer_input_delta:  11/12, margin +1.932
q_input_random:     1/12, margin +0.009
layer_input_random: 1/12, margin -0.044
o_random:           0/12, margin +0.000
```

GLM4 出现重要新信号：

```text
q pathway 仍弱，但 layer_input 强修复。
```

这说明 GLM4 的 value gate 可能不沿当前 top-head Q 路径闭合，而是更依赖整层 residual state。

#### DS7B target rows

```text
o_actual:           31/43, margin +1.709
q_output_top:       33/43, margin +1.718
q_input_full:       31/43, margin +1.748
q_input_delta:      31/43, margin +1.748
layer_input_full:   43/43, margin +3.370
layer_input_delta:  43/43, margin +3.370
q_input_random:     2/43, margin -0.102
layer_input_random: 5/43, margin -0.021
o_random:           1/43, margin -0.017
```

DS7B 最关键：

```text
q_input_full ≈ q_output_top ≈ o_actual
layer_input_full 明显强于 q_input/o_actual
random controls 无效
```

### 当前最可靠客观事实

1. **repair Q 不是只存在于 q_proj output**

DS7B：

```text
q_input_full 31/43, margin +1.748
q_output_top 33/43, margin +1.718
```

说明 q_proj input hidden state 已经携带 repair query state。

2. **瓶颈继续上移到 decoder layer input residual state**

DS7B：

```text
layer_input_full 43/43, margin +3.370
```

这明显强于：

```text
q_output_top 33/43, margin +1.718
o_actual 31/43, margin +1.709
```

说明 L22 layer input 不只携带 top-head Q，还携带其它对最终候选竞争有利的状态。

3. **Qwen3 与 DS7B 同方向**

Qwen3：

```text
q_input_full 6/9, margin +1.585
layer_input_full 9/9, margin +4.392
```

方向与 DS7B 一致。

4. **GLM4 出现整层状态路径，而不是当前 top-head Q 路径**

GLM4：

```text
q_input_full 0/12, margin -0.063
layer_input_full 11/12, margin +1.932
```

这说明 GLM4 在当前定位层的 top-head Q 不足以解释修复，但 layer input residual state 很强。

5. **random controls 排除同范数噪声解释**

DS7B：

```text
q_input_random 2/43
layer_input_random 5/43
o_random 1/43
```

都远低于真实修复。

### 理论进展

Phase613 的链条是：

```text
answer-position Q -> routing pattern -> top-head mixture -> digit1 margin
```

Phase614 更新为：

```text
answer-position layer-input residual state
  -> q_proj input hidden state
  -> answer-position Q
  -> routing pattern
  -> top-head mixture
  -> digit1 margin
```

但 Phase614 还发现：

```text
layer-input residual state 的效果明显强于 Q-only / top-head-only。
```

因此最新理论必须承认：

```text
Q path 是 value gate 的可解释主路径之一，
但 layer-input residual state 是更上游、更完整的状态包，
它可能同时影响 Q、其它 heads、MLP、后续 residual trajectory。
```

### 硬伤与谨慎解释

1. **layer_input patch 不是单机制 patch**

layer_input_full 会影响该层后续所有计算：

```text
attention Q/K/V at answer token
MLP input
residual trajectory
其它 heads
```

所以它强于 q_input/o_actual 并不意外，但也不能直接解释成单一路径闭合。

2. **q_input patch 是 full hidden patch，不是 selected-head patch**

q_input_full 替换整个 q_proj input hidden vector，因此可能影响所有 Q heads，不只 top heads。

3. **还没有定位 layer_input 修复状态由哪个上游层/组件写入**

Phase614 证明 L22 layer_input 是强修复点，但还没有回答：

```text
这个状态在 L18-L21 哪一层形成？
由 attention 写入，还是 MLP 写入？
```

4. **GLM4 路径可能与 DS7B/Qwen3 不同**

GLM4 top-head Q 弱，但 layer_input 强，说明 GLM4 可能需要不同层位或不同 head 集合重新定位。

### 最新理论判断

当前 value gate 的主链条应更新为：

```text
prompt condition
  -> answer-position residual state builder
  -> q_proj input hidden state
  -> answer-position Q state
  -> sparse top-head routing pattern
  -> token-level source mixture over existing V content
  -> o_proj input head slots
  -> digit1 residual trajectory
  -> full candidate margin
```

Phase614 最关键的进展是：

```text
Q 不是起点，Q 是 L22 layer-input residual state 的一个读出接口。
真正上游瓶颈是 answer-position residual state builder。
```

这与“固定网络在不同 prompt 下形成不同状态”的总体研究方向一致：

```text
同一参数网络之所以不混乱，是因为每个 token position 的 residual state 携带条件化计算状态。
```

### 下一步任务

Phase615 应继续做：

```text
Residual State Builder Layer/Component Scan
```

目标：

```text
定位 L22 layer_input repair state 是在前面哪一层、哪一类组件写入的。
```

测试方案：

```text
1. 继续使用 source-aligned prompt pairs。
2. 以 DS7B 为主，Qwen3/GLM4 做对照。
3. 扫描 DS7B L18-L22 answer position：
   - layer_input
   - attn_out
   - mlp_out
   - layer_out
4. 对每个组件 patch repair-base delta 到 base forward。
5. 测 full candidate margin 与 target switch。
6. 加 same-norm random controls。
```

判据：

```text
如果某层 attn_out 强：query state builder 主要由前层 attention 写入。
如果某层 mlp_out 强：query state builder 主要由 MLP 写入。
如果 layer_input 强但单组件弱：状态由多层 residual accumulation 形成。
如果只有 L22 layer_input 强：可能需要 multi-layer cumulative patch。
```

## Phase 615: Residual State Builder Layer Component Scan 残差状态生成器层位-组件扫描 [2026-06-24 22:49]

### 本阶段目标

根据 Phase614 的结果，answer-position query projection 本身不是机制起点，而更像是读取上游 residual state 的接口。本阶段继续追问：

```text
repair state 到底是在前面哪一层、哪一类组件写入的？
```

附件中对 Phase614 的判断基本正确：

```text
1. Q patch 有效，不等于 Q 是语义状态生成器。
2. layer_input_full 比 q_output_top 更强，说明上游 residual state 才是更完整的状态包。
3. 下一步不应继续只盯 Q/K/V，而应扫描 layer_input、attn_out、mlp_out、layer_out。
4. 如果 layer_input/layer_out 强但 attn_out/mlp_out 单项弱，就说明状态更可能由多层残差累积形成。
```

### 执行命令

```bash
python -m py_compile \
  tests/glm5/phase615_residual_state_builder_scan.py \
  tests/glm5/phase615_residual_state_builder_scan_summary.py

python tests/glm5/phase615_residual_state_builder_scan.py qwen3 \
  --smoke \
  --include-nontarget \
  --hard-exit-after-model

python tests/glm5/phase615_residual_state_builder_scan.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase615_residual_state_builder_scan \
  --hard-exit-after-model

python tests/glm5/phase615_residual_state_builder_scan.py glm4 \
  --confirm \
  --output-dir results/glm5_phase615_residual_state_builder_scan \
  --hard-exit-after-model

python tests/glm5/phase615_residual_state_builder_scan.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase615_residual_state_builder_scan \
  --hard-exit-after-model

python tests/glm5/phase615_residual_state_builder_scan_summary.py

python -m py_compile \
  tests/glm5/phase615_residual_state_builder_scan.py \
  tests/glm5/phase615_residual_state_builder_scan_summary.py
```

三模型按 qwen3、glm4、deepseek7b 顺序执行，每次模型结束后使用 `--hard-exit-after-model` 退出，避免 GPU 显存残留。

### 脚本与结果

- 主测试脚本：`tests/glm5/phase615_residual_state_builder_scan.py`
- 汇总脚本：`tests/glm5/phase615_residual_state_builder_scan_summary.py`
- 输出目录：`results/glm5_phase615_residual_state_builder_scan/`
- 跨模型汇总：`results/glm5_phase615_residual_state_builder_scan/phase615_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
prompt pairs = Phase612 source-aligned prompt pairs
candidate values = 前 4 个候选值
patch position = answer position
patch metric = full candidate logprob margin
controls = same-norm random delta
```

扫描层位：

```text
Qwen3: L25-L29
GLM4: L30-L34
DS7B: L18-L22
```

扫描组件：

```text
layer_input
attn_out
mlp_out
layer_out
```

样本过滤：

```text
只保留 base answer 错、repair answer 对的 target rows。
Qwen3: 9 rows
GLM4: 12 rows
DS7B: 43 rows
```

### 原理

对同一个 source-aligned prompt pair，记 base forward 中某层某组件状态为：

```text
h_base(l, c)
```

repair forward 中对应状态为：

```text
h_repair(l, c)
```

本阶段注入：

```text
delta(l, c) = h_repair(l, c) - h_base(l, c)
```

然后在 base forward 中只把 answer position 的对应组件改为：

```text
h_patch(l, c) = h_base(l, c) + delta(l, c)
```

如果某个组件 patch 能显著把错误 base answer 推向 repair answer，说明该组件携带了目标状态的因果信息。如果 layer_input/layer_out 强，而单独 attn_out/mlp_out 弱，说明状态不是单层单组件直接写入，而更可能是跨层 residual accumulation。

### 客观结果

#### Qwen3

```text
rows = 9
raw rows = 128
filtered = token_len_mismatch 0, not_target 119
layers = L25-L29
```

最强结果：

```text
L29 layer_out: 9/9, margin +4.392
L28 layer_out: 9/9, margin +4.392
L29 layer_input: 9/9, margin +4.392
L25 layer_out: 9/9, margin +4.336
L26 layer_input: 9/9, margin +4.336
L26 layer_out: 9/9, margin +4.336
L27 layer_input: 9/9, margin +4.336
L27 layer_out: 9/9, margin +4.322
L28 layer_input: 9/9, margin +4.322
L25 layer_input: 7/9, margin +4.156
L29 attn_out: 6/9, margin +1.599
```

分层事实：

```text
L25 layer_input 7/9 +4.156, attn_out 0/9 -1.391, mlp_out 2/9 +0.500, layer_out 9/9 +4.336
L26 layer_input 9/9 +4.336, attn_out 1/9 +0.334, mlp_out 2/9 +0.153, layer_out 9/9 +4.336
L27 layer_input 9/9 +4.336, attn_out 0/9 -0.139, mlp_out 2/9 +0.612, layer_out 9/9 +4.322
L28 layer_input 9/9 +4.322, attn_out 2/9 +0.181, mlp_out 3/9 +0.612, layer_out 9/9 +4.392
L29 layer_input 9/9 +4.392, attn_out 6/9 +1.599, mlp_out 1/9 +0.069, layer_out 9/9 +4.392
```

Qwen3 的客观现象：

```text
1. layer_input/layer_out 从 L26 开始几乎全恢复。
2. 单层 attn_out 只有 L29 较强。
3. mlp_out 整体弱。
4. 说明主要状态已经存在于 residual stream，不是当前层单组件直接生成。
```

#### GLM4

```text
rows = 12
raw rows = 128
filtered = token_len_mismatch 0, not_target 116
layers = L30-L34
```

最强结果：

```text
L33 layer_out: 11/12, margin +1.932
L34 layer_input: 11/12, margin +1.932
L34 layer_out: 11/12, margin +1.917
L32 layer_out: 11/12, margin +1.911
L33 layer_input: 11/12, margin +1.911
L31 layer_out: 10/12, margin +1.932
L32 layer_input: 10/12, margin +1.932
L30 layer_input: 10/12, margin +1.922
L30 layer_out: 10/12, margin +1.906
L31 layer_input: 10/12, margin +1.906
```

单组件结果：

```text
L34 mlp_out: 2/12, margin +0.156
L32 attn_out: 2/12, margin +0.089
```

GLM4 的客观现象：

```text
1. layer_input/layer_out 非常强。
2. attn_out/mlp_out 单项很弱。
3. 这解释了前面 GLM4 q path 偏弱但 layer_input 强的现象。
4. GLM4 的状态更像 whole-residual state，不像 DS7B 那样容易被单个 top attention path 抓住。
```

#### DS7B

```text
rows = 43
raw rows = 128
filtered = token_len_mismatch 0, not_target 85
layers = L18-L22
```

最强结果：

```text
L21 layer_out: 43/43, margin +3.370
L22 layer_input: 43/43, margin +3.370
L22 layer_out: 42/43, margin +3.337
L20 layer_out: 42/43, margin +3.111
L21 layer_input: 42/43, margin +3.111
L22 attn_out: 32/43, margin +1.738
L19 layer_out: 31/43, margin +1.953
L20 layer_input: 31/43, margin +1.953
L18 layer_out: 30/43, margin +1.955
L19 layer_input: 30/43, margin +1.955
```

分层事实：

```text
L18 layer_input 10/43 +0.394, attn_out 19/43 +1.071, mlp_out 3/43 +0.052, layer_out 30/43 +1.955
L19 layer_input 30/43 +1.955, attn_out 20/43 +0.874, mlp_out 1/43 -0.174, layer_out 31/43 +1.953
L20 layer_input 31/43 +1.953, attn_out 19/43 +0.935, mlp_out 11/43 +0.418, layer_out 42/43 +3.111
L21 layer_input 42/43 +3.111, attn_out 3/43 +0.118, mlp_out 6/43 +0.344, layer_out 43/43 +3.370
L22 layer_input 43/43 +3.370, attn_out 32/43 +1.738, mlp_out 2/43 +0.048, layer_out 42/43 +3.337
```

random control：

```text
same-norm random controls 明显弱于真实 delta。
例如 L22 layer_input random 只有 4/43, margin -0.145。
```

DS7B 的客观现象：

```text
1. L18-L20 已经形成部分 residual state。
2. L20 layer_out / L21 layer_input 从 31/43 跳到 42/43。
3. L21 layer_out / L22 layer_input 达到 43/43。
4. L22 attn_out 只有 32/43，弱于完整 residual state。
5. 单层 attn_out/mlp_out 都不能单独解释完整状态包。
```

### 当前最可靠客观事实

1. **Phase614 的判断被加强**

```text
Q 是接口，不是完整状态生成器。
```

证据：

```text
DS7B L22 attn_out: 32/43, margin +1.738
DS7B L22 layer_input: 43/43, margin +3.370
```

2. **answer-position residual state 是更完整的 causal state**

三模型都出现：

```text
layer_input/layer_out 明显强于单独 attn_out/mlp_out。
```

3. **DS7B 的 value gate repair state 有清楚的层级跃迁**

```text
L18-L19: partial state 约 30/43
L20-L21: near-full state 约 42/43
L21-L22: full state 43/43
```

4. **GLM4 的状态更分布式**

```text
GLM4 layer_input/layer_out 强，但 q path 和单组件弱。
```

这解释了之前 GLM4 经常出现：

```text
内部状态可见，但单点机制不明显。
```

5. **Qwen3 和 DS7B 有共同结构，但表面路径不同**

```text
Qwen3: residual state 已在 L26-L29 持续存在。
DS7B: residual state 在 L20-L22 出现明显跃迁。
```

### 理论进展

当前主链条从：

```text
value gate failure
```

推进为：

```text
answer-position residual state 形成不足或状态路径不稳定。
```

更精确地说，当前可暂时写成：

```text
prompt condition
  -> residual state package at answer position
  -> query interface / attention routing
  -> candidate-specific readout
  -> generation gate
```

其中 Phase615 把关键位置从：

```text
query projection
```

上移到：

```text
pre-query residual state package
```

### 问题和硬伤

1. **样本量仍由 target rows 限制**

```text
Qwen3 只有 9 rows
GLM4 只有 12 rows
DS7B 43 rows 较可靠
```

Qwen3/GLM4 的结论方向可信，但不能像 DS7B 一样下强结论。

2. **当前只做单层单组件 patch**

如果状态由多层小量累积形成，单组件 patch 会低估每个组件真实贡献。

3. **layer_input/layer_out 强不等于知道了生成算法**

它只证明状态包存在，并且有强因果作用；还没有拆出状态包内部的字段结构。

4. **没有直接分离 pattern state 与 content state**

虽然 Phase612-614 已经显示 routing pattern 很关键，但 Phase615 的 residual state package 可能同时包含：

```text
condition
value gate
candidate preference
format/readout context
```

5. **还不能说找到了统一语言编码公式**

当前是机制拼图推进，不是完整理论闭合。

### 下一步任务

Phase616 应继续做：

```text
Residual State Cumulative Patch and Boundary Localization
```

目标：

```text
判断完整 repair state 是由多层 residual accumulation 形成，还是由某个尚未拆分的隐藏组件形成。
```

优先测试 DS7B，再做 Qwen3/GLM4 对照：

```text
1. 对 L18-L22 做 cumulative layer_input/layer_out patch。
2. 对 L18/L19/L20/L22 attn_out 做 cumulative patch。
3. 对 L20/L21 mlp_out 做 cumulative patch。
4. 测 attn_out cumulative + mlp_out cumulative 是否接近 L22 layer_input。
5. 加 same-norm random cumulative control。
6. 记录 target switch、full candidate margin、rank change。
```

判据：

```text
如果 cumulative attn/mlp 接近 layer_input：
  residual state 是多组件累积形成。

如果 cumulative attn/mlp 仍远弱于 layer_input：
  需要进一步拆 attention head、MLP internal gate、norm/gating 或 residual mixing。

如果某个层段组合突然闭合：
  该层段就是 value gate state builder 的主要形成区间。
```

## Phase 616: Residual State Cumulative Patch and Boundary Localization 残差状态累积修补与形成边界定位 [2026-06-24 23:16]

### 本阶段目标

Phase615 证明：

```text
answer-position layer_input/layer_out 明显强于单独 attn_out/mlp_out。
```

附件中对 Phase615 的分析基本正确，尤其是：

```text
1. Q 不是机制起点，而是 residual state 的读出接口。
2. DS7B 的 repair state 在 L18-L22 之间逐步形成。
3. 单层单组件 patch 不足以解释完整状态包。
4. 下一步必须测试多层 residual accumulation，而不是继续单点搜索。
```

本阶段继续追问：

```text
多个弱组件的 repair delta 累积后，是否能接近完整 layer_input/layer_out repair state？
```

### 执行命令

```bash
python -m py_compile \
  tests/glm5/phase616_residual_state_cumulative_patch.py \
  tests/glm5/phase616_residual_state_cumulative_patch_summary.py

python tests/glm5/phase616_residual_state_cumulative_patch.py qwen3 \
  --smoke \
  --include-nontarget \
  --hard-exit-after-model

python tests/glm5/phase616_residual_state_cumulative_patch.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase616_residual_state_cumulative_patch \
  --hard-exit-after-model

python tests/glm5/phase616_residual_state_cumulative_patch.py glm4 \
  --confirm \
  --output-dir results/glm5_phase616_residual_state_cumulative_patch \
  --hard-exit-after-model

python tests/glm5/phase616_residual_state_cumulative_patch.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase616_residual_state_cumulative_patch \
  --hard-exit-after-model

python tests/glm5/phase616_residual_state_cumulative_patch_summary.py

python -m py_compile \
  tests/glm5/phase616_residual_state_cumulative_patch.py \
  tests/glm5/phase616_residual_state_cumulative_patch_summary.py
```

三模型仍然按 qwen3、glm4、deepseek7b 顺序执行，每个模型运行都带 `--hard-exit-after-model`，没有并行加载多个模型。

### 脚本与结果

- 主测试脚本：`tests/glm5/phase616_residual_state_cumulative_patch.py`
- 汇总脚本：`tests/glm5/phase616_residual_state_cumulative_patch_summary.py`
- 输出目录：`results/glm5_phase616_residual_state_cumulative_patch/`
- 跨模型汇总：`results/glm5_phase616_residual_state_cumulative_patch/phase616_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
prompt pairs = Phase612 source-aligned prompt pairs
candidate values = 前 4 个候选值
patch position = answer position
raw cases/model = 128
target rows = base answer 错、repair answer 对
controls = same-norm random cumulative controls
```

有效 target rows：

```text
Qwen3: 9/128
GLM4: 12/128
DS7B: 43/128
```

扫描层位：

```text
Qwen3: L25-L29
GLM4: L30-L34
DS7B: L18-L22
```

测试 patch 类型：

```text
1. replace last layer_input reference
2. replace previous layer_out reference
3. additive single layer_out
4. additive layer_out bridge
5. additive attn_out all layers
6. additive mlp_out all layers
7. additive attn_out + mlp_out all layers
8. additive early / late / midlate span
9. same-norm random controls
```

### 原理

Phase615 使用的是单点替换：

```text
h_patch(l,c) = h_base(l,c) + delta(l,c)
```

但一次只替换一个组件。本阶段改为多点加性累积：

```text
h_patch(l_i,c_i) = h_current(l_i,c_i) + delta(l_i,c_i)
```

其中：

```text
delta(l_i,c_i) = h_repair(l_i,c_i) - h_base(l_i,c_i)
```

如果多个 `attn_out` 或 `mlp_out` 的 delta 累积后能达到 `layer_input/layer_out` reference 的效果，说明完整 residual state 可以由多层组件累积形成。如果累积后仍然远弱于 reference，说明还有未拆分的隐藏路径。

### 客观结果

#### Qwen3

```text
rows = 9
layers = L25-L29
specs = 26
time = 1.17 min
```

最强结果：

```text
add_layer_out_L28_L29_bridge: 9/9, margin +7.282
replace_L28_layer_out_ref: 9/9, margin +4.392
replace_L29_layer_input_ref: 9/9, margin +4.392
add_L26_layer_out: 9/9, margin +4.364
add_L25_layer_out: 9/9, margin +4.364
add_L29_layer_out: 9/9, margin +4.350
add_L27_layer_out: 9/9, margin +4.336
add_layer_out_all: 8/9, margin +11.337
```

组件累积：

```text
add_attn_mlp_midlate_L27_L29: 6/9, margin +2.947
add_mlp_all: 6/9, margin +1.988
add_attn_late_L28_L29: 6/9, margin +1.849
add_L29_attn_out: 6/9, margin +1.599
add_attn_mlp_all: 5/9, margin +2.238
add_attn_all: 3/9, margin +0.543
```

random controls：

```text
add_mlp_all random: 2/9, margin +0.127
replace_L28_layer_out_ref random: 1/9, margin +0.037
add_layer_out_L28_L29_bridge random: 1/9, margin +0.030
add_attn_mlp_all random: 0/9, margin -0.081
```

Qwen3 的客观现象：

```text
1. layer_out bridge 强于单层 reference。
2. attn+mlp midlate 有部分恢复，但不到 layer_out bridge。
3. MLP 累积比 attention all 更强。
4. 样本只有 9 行，不能下过强结论。
```

#### GLM4

```text
rows = 12
layers = L30-L34
specs = 26
time = 2.21 min
```

最强结果：

```text
add_layer_out_L33_L34_bridge: 11/12, margin +4.310
add_L33_layer_out: 11/12, margin +1.943
replace_L33_layer_out_ref: 11/12, margin +1.932
replace_L34_layer_input_ref: 11/12, margin +1.932
add_L32_layer_out: 11/12, margin +1.917
add_L34_layer_out: 11/12, margin +1.906
add_layer_out_all: 10/12, margin +10.632
```

组件累积：

```text
add_attn_mlp_midlate_L32_L34: 5/12, margin +0.438
add_mlp_all: 4/12, margin +0.417
add_mlp_midlate_L32_L34: 3/12, margin +0.328
add_attn_mlp_all: 2/12, margin +0.479
add_attn_all: 1/12, margin -0.016
```

random controls：

```text
replace_L34_layer_input_ref random: 1/12, margin +0.091
add_layer_out_all random: 1/12, margin +0.014
add_attn_mlp_all random: 0/12, margin +0.017
add_attn_all random: 0/12, margin -0.073
```

GLM4 的客观现象：

```text
1. layer_out bridge 很强。
2. attn_out cumulative 几乎无效。
3. mlp_out cumulative 有弱正效应，但不能解释完整 residual state。
4. GLM4 仍然表现为 whole-residual / mixed-state path。
```

#### DS7B

```text
rows = 43
layers = L18-L22
specs = 26
time = 5.10 min
```

最强结果：

```text
add_layer_out_L21_L22_bridge: 43/43, margin +6.142
add_attn_mlp_all: 43/43, margin +5.580
add_attn_all: 43/43, margin +3.900
replace_L21_layer_out_ref: 43/43, margin +3.370
replace_L22_layer_input_ref: 43/43, margin +3.370
add_L22_layer_out: 43/43, margin +3.333
add_layer_out_all: 42/43, margin +9.748
add_L21_layer_out: 42/43, margin +3.350
add_L20_layer_out: 42/43, margin +3.113
```

组件累积：

```text
add_attn_mlp_midlate_L20_L22: 41/43, margin +3.925
add_attn_early_L18_L20: 37/43, margin +2.401
add_attn_late_L21_L22: 34/43, margin +1.902
add_L22_attn_out: 32/43, margin +1.736
add_mlp_midlate_L20_L22: 15/43, margin +0.914
add_mlp_all: 14/43, margin +0.853
```

random controls：

```text
replace_L22_layer_input_ref random: 6/43, margin +0.021
add_layer_out_L21_L22_bridge random: 4/43, margin +0.008
add_attn_mlp_all random: 4/43, margin -0.068
add_attn_all random: 3/43, margin +0.065
add_layer_out_all random: 3/43, margin -0.384
```

DS7B 的客观现象：

```text
1. attention cumulative path 已经达到 43/43。
2. attn+mlp cumulative 也达到 43/43，并且 margin 更高。
3. MLP cumulative 单独很弱，只有 14/43 到 15/43。
4. L22 单层 attn_out 是 32/43，但 L18-L22 attn_all 是 43/43。
5. 这说明 DS7B 的完整 value gate repair state 可以由多层 attention accumulation 解释大部分。
```

### 当前最可靠客观事实

1. **DS7B 的 residual state 不是单层生成，而是多层 attention 累积形成**

关键证据：

```text
L22 attn_out single: 32/43, margin +1.736
L18-L22 attn_all cumulative: 43/43, margin +3.900
L18-L22 attn+mlp cumulative: 43/43, margin +5.580
L22 layer_input reference: 43/43, margin +3.370
```

这说明 Phase615 的“单组件不足”并不是因为 attention 无关，而是因为 attention contribution 分布在多层。

2. **MLP 不是 DS7B value gate repair 的主路径**

```text
MLP all: 14/43, margin +0.853
MLP midlate: 15/43, margin +0.914
```

MLP 有辅助作用，但不能单独形成完整状态。

3. **layer_out bridge 是最强的低维边界定位**

三模型均出现：

```text
Qwen3 add_layer_out_L28_L29_bridge: 9/9, margin +7.282
GLM4 add_layer_out_L33_L34_bridge: 11/12, margin +4.310
DS7B add_layer_out_L21_L22_bridge: 43/43, margin +6.142
```

这说明最终 answer-state package 在倒数几层的 residual bridge 中非常稳定。

4. **GLM4 与 DS7B 的机制表面不同**

```text
DS7B: attention cumulative path 可闭合。
GLM4: attention cumulative path 仍弱，layer_out bridge 强。
```

因此不能把 DS7B 的 attention 累积机制直接泛化成所有模型统一路径。

5. **Qwen3 的方向与 DS7B 一致，但样本不足**

```text
Qwen3 layer_out bridge 强，attn+mlp midlate 有中等恢复。
```

但 target rows 只有 9，后续需要换任务或扩大 case pool。

### 理论进展

当前链条从：

```text
Q interface
```

继续推进到：

```text
multi-layer attention accumulation builds answer-position residual state
```

至少在 DS7B 上，当前可以写成更具体的路径：

```text
prompt condition
  -> multi-layer attention accumulation at answer position
  -> residual state package
  -> query interface
  -> routing pattern
  -> candidate readout
  -> generation gate
```

这对“同一个网络为什么不混乱”有一个更清楚的解释：

```text
同一参数网络不是靠固定单方向表达语义，而是在每个 answer position 形成条件化 residual state。
这个 residual state 由多层 attention 根据当前 prompt condition 逐步累积。
后续 Q/query 只是把这个状态读成路由模式。
```

### 问题和硬伤

1. **additive layer_out bridge 可能包含重复注入**

`layer_out` 本身已经包含前面 residual state，连续加性 patch 多层 layer_out 会放大状态差异。因此：

```text
layer_out bridge 很强
```

只能说明该区间是强状态边界，不能直接解释为独立组件贡献相加。

2. **attention cumulative 在 DS7B 闭合，但还没有拆到 head/source**

当前只知道：

```text
L18-L22 attention cumulative sufficient
```

还不知道：

```text
哪些 heads
哪些 source positions
哪些 attention pattern
哪些 content value
```

真正图谱还没完成。

3. **GLM4 仍然没有机制闭合**

GLM4 的 `layer_out bridge` 强，而 `attn_all` 弱，说明它可能需要：

```text
更细粒度 head scan
norm/gating path
MLP internal path
或更长层段 residual bridge
```

4. **样本量瓶颈仍然存在**

```text
Qwen3: 9 target rows
GLM4: 12 target rows
DS7B: 43 target rows
```

DS7B 结论最可靠，Qwen3/GLM4 仍需扩大任务池。

5. **仍未完成自然生成闭环**

当前主要看 candidate logprob/rank，不等于完整自然语言生成闭环。

### 下一步任务

Phase617 应继续做：

```text
Attention Cumulative Path Graph Decomposition
```

优先使用 DS7B，因为 DS7B 已经出现最清楚的闭合：

```text
attn_all cumulative = 43/43
```

目标：

```text
把 L18-L22 attention cumulative path 拆成 head、source position、pattern/content 三层图谱。
```

测试方案：

```text
1. 对 DS7B L18-L22 每层 attention 做 head-level cumulative patch。
2. 找出最小 head set，使恢复接近 attn_all。
3. 对关键 head 做 source position grouping：
   - answer/self
   - object line
   - category line
   - rule/value line
   - question line
   - punctuation/format
4. 对关键 head 做 pattern/content split：
   - repair pattern + base value
   - base pattern + repair value
   - repair pattern + repair value
5. 加 same-norm random 和 wrong-layer controls。
6. 输出 attention path graph：
   layer -> head -> source group -> pattern/content role -> target effect
```

判据：

```text
如果少数 heads + source groups 能接近 attn_all：
  DS7B value gate state builder 有可图谱化路径。

如果需要大量 heads 才能恢复：
  说明状态由 distributed attention field 形成，需要转向 field-level atlas。

如果 pattern 分量强于 content 分量：
  延续 Phase612 的 routing-pattern 主导结论。

如果 content 分量变强：
  说明 residual state builder 比最终 top-head routing 更依赖 value content。
```

## Phase 617: Attention Head Cumulative Graph 多层注意力 Head 累积图谱 [2026-06-24 23:51]

### 本阶段目标

Phase616 证明 DS7B 的：

```text
L18-L22 attn_all cumulative = 43/43
```

这说明 answer-position residual state 可以由多层 attention accumulation 形成。附件中对 Phase616 的判断基本正确：

```text
1. value gate 主链条已经从 Q interface 上移到 multi-layer attention accumulation。
2. 单层 attention 不足，但多层 attention 可以闭合。
3. DS7B 结果最可靠，Qwen3/GLM4 只能作为方向性对照。
4. 下一步应该把 attention accumulation 拆到 layer/head/source/pattern。
```

本阶段先完成其中第一步：

```text
把 multi-layer attention accumulation 拆到 layer -> head slot。
```

source position 与 pattern/content split 留到下一阶段，避免一次性测试过重导致结果不稳定。

### 执行命令

```bash
python -m py_compile \
  tests/glm5/phase617_attention_head_cumulative_graph.py \
  tests/glm5/phase617_attention_head_cumulative_graph_summary.py

python tests/glm5/phase617_attention_head_cumulative_graph.py qwen3 \
  --smoke \
  --include-nontarget \
  --hard-exit-after-model

python tests/glm5/phase617_attention_head_cumulative_graph.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase617_attention_head_cumulative_graph \
  --hard-exit-after-model

python tests/glm5/phase617_attention_head_cumulative_graph.py glm4 \
  --confirm \
  --output-dir results/glm5_phase617_attention_head_cumulative_graph \
  --hard-exit-after-model

python tests/glm5/phase617_attention_head_cumulative_graph.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase617_attention_head_cumulative_graph \
  --hard-exit-after-model
```

首次 DS7B 84-spec 全量 head 组合运行中进程以 139 退出，没有生成有效结果。判断是过重 hook/forward 组合触发底层不稳定，不作为实验结果使用。随后收紧 patch specs，不减少 raw cases 和 target rows，执行：

```bash
python tests/glm5/phase617_attention_head_cumulative_graph.py deepseek7b \
  --confirm \
  --compact \
  --output-dir results/glm5_phase617_attention_head_cumulative_graph \
  --hard-exit-after-model

python tests/glm5/phase617_attention_head_cumulative_graph_summary.py

python -m py_compile \
  tests/glm5/phase617_attention_head_cumulative_graph.py \
  tests/glm5/phase617_attention_head_cumulative_graph_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/glm5/phase617_attention_head_cumulative_graph.py`
- 汇总脚本：`tests/glm5/phase617_attention_head_cumulative_graph_summary.py`
- 输出目录：`results/glm5_phase617_attention_head_cumulative_graph/`
- 跨模型汇总：`results/glm5_phase617_attention_head_cumulative_graph/phase617_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
prompt pairs = Phase612 source-aligned prompt pairs
candidate values = 前 4 个候选值
patch position = answer position
patch site = attention o_proj input head slots
raw cases/model = 128
target rows = base answer 错、repair answer 对
controls = same-norm random slot controls
```

有效 target rows：

```text
Qwen3: 9/128
GLM4: 12/128
DS7B: 43/128
```

扫描层位与 head 数：

```text
Qwen3: L25-L29, 32 heads/layer
GLM4: L30-L34, 32 heads/layer
DS7B: L18-L22, 28 heads/layer
```

测试类型：

```text
1. all_heads_all_layers
2. all_heads_midlate
3. all_heads_single_layer
4. known_top{k}_all_layers
5. known_top{k}_midlate
6. single known heads
7. deterministic coverage heads
8. same-norm random controls
```

### 原理

Phase616 的 `attn_out` patch 是 attention module 输出层面的整体 patch。本阶段进一步在 `o_proj input` 的 head slot 上注入：

```text
delta_o(l,h) = o_repair(l,h) - o_base(l,h)
```

对多个 layer/head slot 做累积：

```text
o_patch(l,h) = o_current(l,h) + delta_o(l,h)
```

如果 `all_heads_all_layers` 可以接近 Phase616 的 `attn_all`，说明 Phase616 的 attention cumulative effect 可以在 head-slot 层面复现。如果少数 known top heads 接近 all heads，说明存在稀疏关键 head 集合。如果 all heads 强但 top heads 不够，说明是 broad/distributed attention field。

### 客观结果

#### Qwen3

```text
rows = 9
layers = L25-L29
heads = 32/layer
specs = 84
time = 2.90 min
```

最强结果：

```text
all_heads_midlate_L27_L29: 6/9, margin +1.821
all_heads_L29: 6/9, margin +1.640
known_top8_midlate_L27_L29: 4/9, margin +1.238
L29_H11: 4/9, margin +1.223
known_top1_midlate_L27_L29: 4/9, margin +1.209
known_top1_all_layers: 4/9, margin +1.182
all_heads_all_layers: 3/9, margin +0.529
```

random controls：

```text
all_heads_midlate random: 2/9, margin +0.175
all_heads_all_layers random: 1/9, margin -0.052
```

Qwen3 的客观现象：

```text
1. midlate attention head slots 有部分恢复。
2. L29 单层 all_heads 与 L27-L29 midlate 较强。
3. L29_H11 是最强单 head，达到 4/9。
4. all_layers 反而弱于 midlate，说明早层 head delta 可能有干扰。
5. 样本只有 9 行，不能强结论。
```

#### GLM4

```text
rows = 12
layers = L30-L34
heads = 32/layer
specs = 84
time = 5.73 min
```

最强结果：

```text
L32_coverage_H16: 2/12, margin +0.036
all_heads_L32: 1/12, margin +0.094
all_heads_midlate_L32_L34: 1/12, margin +0.089
all_heads_all_layers: 1/12, margin -0.021
known_top cumulative 基本 0/12 到 1/12
```

GLM4 的客观现象：

```text
1. o_proj input head-slot patch 基本不能恢复。
2. 这与 Phase616 的 GLM4 layer_out bridge 强形成对照。
3. GLM4 的 residual state 不是当前这种 attention head-slot path 可以解释的。
4. 需要继续考虑 whole-residual、norm/gating、MLP internal 或更复杂 mixed path。
```

#### DS7B

```text
rows = 43
layers = L18-L22
heads = 28/layer
specs = 44 compact
time = 8.78 min
```

最强结果：

```text
all_heads_all_layers: 43/43, margin +3.901
known_top6_all_layers: 42/43, margin +2.920
known_top6_midlate_L20_L22: 41/43, margin +2.621
all_heads_midlate_L20_L22: 39/43, margin +2.692
known_top4_all_layers: 36/43, margin +2.249
known_top4_midlate_L20_L22: 33/43, margin +1.948
all_heads_L22: 32/43, margin +1.727
known_top2_midlate_L20_L22: 30/43, margin +1.441
known_top2_all_layers: 30/43, margin +1.435
```

单层 all_heads：

```text
L22 all_heads: 32/43, margin +1.727
L18 all_heads: 21/43, margin +1.056
L19 all_heads: 21/43, margin +0.880
L20 all_heads: 18/43, margin +0.935
L21 all_heads: 3/43, margin +0.121
```

强单 head：

```text
L20_H25: 12/43, margin +0.724
L22_H1: 12/43, margin +0.714
L18_H24: 8/43, margin +0.512
L22_H3: 7/43, margin +0.381
L22_H24: 7/43, margin +0.355
L22_H7: 5/43, margin +0.370
L20_H1: 5/43, margin +0.223
```

random controls：

```text
all_heads_all_layers random: 6/43, margin -0.100
known_top6_all_layers random: 4/43, margin -0.159
all_heads_midlate random: 3/43, margin -0.018
all_heads_L22 random: 2/43, margin -0.039
```

DS7B 的客观现象：

```text
1. Phase616 的 attention cumulative effect 可以在 o_proj input head slots 复现。
2. all_heads_all_layers 达到 43/43，margin +3.901。
3. known_top6_all_layers 已经达到 42/43，说明存在较小 head 集合近似闭合。
4. known_top6_midlate L20-L22 达到 41/43，说明主要有效区间集中在 L20-L22，但 L18-L19 仍有补充作用。
5. 单 head 有效但不充分，最强单 head 只有 12/43。
6. L21 单层 all_heads 很弱，说明 L21 在 Phase616 的 layer_out bridge 强不等价于 L21 attention head slot 强。
```

### 当前最可靠客观事实

1. **DS7B attention cumulative path 已经被拆到 head-slot 层面**

```text
Phase616 attn_all: 43/43, margin +3.900
Phase617 all_heads_all_layers o_proj input slots: 43/43, margin +3.901
```

两者几乎一致，说明 Phase616 的 attention cumulative 不是 hook 假象，而可以由 o_proj input head slots 复现。

2. **DS7B 是 sparse + distributed hybrid**

```text
known_top6_all_layers: 42/43
known_top6_midlate: 41/43
all_heads_all_layers: 43/43
```

少数 top heads 已经接近完整恢复，但 all heads 仍更强，说明不是纯单头机制，也不是完全均匀场，而是：

```text
sparse dominant heads + distributed residual support
```

3. **L22 是最强单层，但不是完整路径**

```text
L22 all_heads: 32/43
L18-L22 all_heads: 43/43
```

所以单层 L22 只解释局部读出/末端路由，多层路径才解释完整 repair state。

4. **GLM4 明确不是这个 head-slot 路径**

```text
GLM4 all_heads_all_layers: 1/12, margin -0.021
```

这与 Phase616 的 layer_out bridge 强形成重要分歧。GLM4 需要另一条机制线。

5. **Qwen3 部分支持但不闭合**

```text
Qwen3 all_heads_midlate: 6/9
Qwen3 all_heads_all_layers: 3/9
```

这说明 Qwen3 可能有 late attention path，但早层注入会干扰；由于 target rows 少，暂不下强结论。

### 理论进展

当前 DS7B 局部链条可进一步细化为：

```text
prompt condition
  -> L18-L22 multi-layer attention head-slot accumulation
  -> answer-position residual state package
  -> q_proj input
  -> Q state
  -> routing pattern
  -> candidate readout
  -> generation gate
```

更具体地说：

```text
不是所有 attention heads 等价；
不是单 head 决定；
而是少数 dominant heads 与更宽的 distributed support 一起构成状态场。
```

这对“语言背后的编码机制”有一个谨慎推进：

```text
编码不是固定语义向量，而是条件化状态场；
状态场在 answer position 由多层 attention head slots 累积；
query 只是读取该状态场并转成路由。
```

### 问题和硬伤

1. **DS7B 首次 84-spec 运行崩溃**

这说明当前 hook-based exhaustive scan 工程上仍不稳定。有效结果来自 compact specs，不是完整 head exhaustive map。

2. **known_top heads 来自历史候选**

本阶段没有重新全量搜索所有 head 子集，因此：

```text
known_top6 近闭合
```

不能解释为真正最小 head set，只能解释为“已有 top head 集合足够强”。

3. **还没做 source position 分解**

当前知道哪些 head slot 有效，但不知道它们读的是：

```text
object line
category line
rule/value line
question line
format/punctuation
self position
```

4. **还没做 pattern/content split**

Phase612 显示最终 routing pattern 很关键，但 residual state builder 的 head-slot 里到底是 pattern 主导还是 content 主导，仍未验证。

5. **GLM4 路径仍未解释**

GLM4 的负结果很重要，但说明跨模型统一公式还不能只写成 attention head-slot accumulation。

### 下一步任务

Phase618 应继续做：

```text
DS7B Attention Source and Pattern/Content Decomposition
```

目标：

```text
把 Phase617 找到的 DS7B dominant head path 拆成 source group 与 pattern/content role。
```

优先对象：

```text
DS7B:
  L20_H25
  L22_H1
  L18_H24
  L22_H3
  L22_H24
  known_top6_midlate_L20_L22
```

测试方案：

```text
1. 只对 DS7B 先做，因为 DS7B head-slot path 已经闭合。
2. 使用 source-aligned target rows。
3. 对关键 heads 做 source group mask：
   - self / answer
   - object statement line
   - category statement line
   - rule/value line
   - question line
   - punctuation/format
4. 对关键 heads 做 pattern/content split：
   - repair pattern + base value
   - base pattern + repair value
   - repair pattern + repair value
5. 对 known_top6_midlate 做 cumulative source-group patch。
6. 加 random same-norm 与 wrong-source controls。
```

判据：

```text
如果少数 source groups 复现 known_top6：
  value gate state builder 可图谱化。

如果 source groups 分散：
  说明它是 distributed attention field。

如果 repair pattern + base value 强：
  延续 routing-pattern 主导结论。

如果 base pattern + repair value 强：
  说明 residual state builder 更依赖内容搬运。
```

## Phase 618: Attention Source and Pattern Content Decomposition 注意力 Source 与 Pattern/Content 分解 [2026-06-25 00:33]

### 本阶段目标

Phase617 已经把 DS7B 的 value gate repair path 拆到：

```text
L18-L22 multi-layer attention head slots
```

附件中对 Phase617 的判断总体正确：

```text
1. DS7B 的 Phase616 attention cumulative effect 可以在 head-slot 层面复现。
2. DS7B 是 sparse dominant heads + distributed support。
3. L22 是强 endpoint，但完整路径需要多层。
4. GLM4 明确不走当前 head-slot path。
5. 下一步必须拆 source group 和 pattern/content role。
```

本阶段目标：

```text
把 top attention head path 拆成 source group 与 pattern/content role。
```

### 执行命令

```bash
python -m py_compile \
  tests/glm5/phase618_attention_source_pattern_content.py \
  tests/glm5/phase618_attention_source_pattern_content_summary.py

python tests/glm5/phase618_attention_source_pattern_content.py qwen3 \
  --smoke \
  --include-nontarget \
  --compact \
  --hard-exit-after-model

python tests/glm5/phase618_attention_source_pattern_content.py qwen3 \
  --confirm \
  --compact \
  --output-dir results/glm5_phase618_attention_source_pattern_content \
  --hard-exit-after-model

python tests/glm5/phase618_attention_source_pattern_content.py glm4 \
  --confirm \
  --compact \
  --output-dir results/glm5_phase618_attention_source_pattern_content \
  --hard-exit-after-model

python tests/glm5/phase618_attention_source_pattern_content.py deepseek7b \
  --confirm \
  --compact \
  --output-dir results/glm5_phase618_attention_source_pattern_content \
  --hard-exit-after-model

python tests/glm5/phase618_attention_source_pattern_content_summary.py

python -m py_compile \
  tests/glm5/phase618_attention_source_pattern_content.py \
  tests/glm5/phase618_attention_source_pattern_content_summary.py
```

仍然按 qwen3、glm4、deepseek7b 顺序执行，每个模型带 `--hard-exit-after-model`。本阶段使用 compact specs，不减少 raw cases，只减少 source/spec 组合数，避免 Phase617 中过重 hook 组合造成底层不稳定。

### 脚本与结果

- 主测试脚本：`tests/glm5/phase618_attention_source_pattern_content.py`
- 汇总脚本：`tests/glm5/phase618_attention_source_pattern_content_summary.py`
- 输出目录：`results/glm5_phase618_attention_source_pattern_content/`
- 跨模型汇总：`results/glm5_phase618_attention_source_pattern_content/phase618_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
prompt pairs = Phase612 source-aligned prompt pairs
candidate values = 前 4 个候选值
raw cases/model = 128
target rows = base answer 错、repair answer 对
patch site = attention o_proj input head slots
```

有效 target rows：

```text
Qwen3: 9/128
GLM4: 12/128
DS7B: 43/128
```

层位：

```text
Qwen3: L27-L29
GLM4: L32-L34
DS7B: L20-L22
```

top heads：

```text
Qwen3: [11, 23, 6, 14, 5, 2]
GLM4: [12, 8, 4, 28, 6, 7]
DS7B: [3, 1, 7, 24, 25, 13]
```

source groups：

```text
self_answer
question_line
final_object_category_line
object_rule_lines
value_rule_lines
punct_format
other
all_source
```

pattern/content modes：

```text
rb_pattern = repair pattern + base value
br_content = base pattern + repair value
rr_pattern_content = repair pattern + repair value
```

### 原理

单个 attention head 的输出写成：

```text
z(l,h,t) = sum_s alpha(l,h,t,s) * V(l,h,s)
```

本阶段按 source group 切分 source token 集合：

```text
G = {s_1, s_2, ...}
```

然后构造三种差分：

```text
rb_pattern:
  sum_{s in G} alpha_repair(s) * V_base(s)
  -
  sum_{s in G} alpha_base(s) * V_base(s)

br_content:
  sum_{s in G} alpha_base(s) * V_repair(s)
  -
  sum_{s in G} alpha_base(s) * V_base(s)

rr_pattern_content:
  sum_{s in G} alpha_repair(s) * V_repair(s)
  -
  sum_{s in G} alpha_base(s) * V_base(s)
```

如果 `rb_pattern` 强，说明修复主要来自 attention pattern 改变。如果 `br_content` 强，说明修复主要来自 value content 改变。如果某个 source group 强，说明该组是关键读取来源。

### 客观结果

#### Qwen3

```text
rows = 9
layers = L27-L29
heads = 32/layer
top_heads = [11, 23, 6, 14, 5, 2]
specs = 48
time = 3.20 min
```

最强结果：

```text
L29_H11 all_source rr: 4/9, margin +1.237
L29_H11 value_rule_lines rr: 4/9, margin +1.210
top6_midlate value_rule_lines rb_pattern: 3/9, margin +0.890
top6_midlate value_rule_lines rr_pattern_content: 3/9, margin +0.890
top6_midlate all_source rb_pattern: 3/9, margin +0.862
top6_midlate all_source rr_pattern_content: 3/9, margin +0.848
```

Qwen3 的客观现象：

```text
1. 最强 source 是 value_rule_lines。
2. rb_pattern 与 rr_pattern_content 几乎相同。
3. br_content 很弱。
4. 说明 Qwen3 的部分路径也偏 pattern 主导，但样本只有 9 行。
```

#### GLM4

```text
rows = 12
layers = L32-L34
heads = 32/layer
top_heads = [12, 8, 4, 28, 6, 7]
specs = 48
time = 5.25 min
```

最强真实结果只有：

```text
L34_H8 all_source rr: 1/12, margin +0.026
L34_H4 all_source rr: 1/12, margin +0.010
L32_H12 question_line rr: 1/12, margin +0.005
top6_midlate value_rule_lines rb_pattern: 1/12, margin -0.016
top6_midlate all_source rr_pattern_content: 0/12, margin -0.036
```

GLM4 的客观现象：

```text
1. 当前 source/pattern-content head path 基本无效。
2. 这继续支持 Phase617 的负结果。
3. GLM4 的强 residual state 不是当前 DS7B-style head-slot source path。
```

#### DS7B

```text
rows = 43
layers = L20-L22
heads = 28/layer
top_heads = [3, 1, 7, 24, 25, 13]
specs = 48
time = 15.20 min
```

最强 top path：

```text
top6_midlate all_source rr_pattern_content:
  41/43, margin +2.624

top6_midlate value_rule_lines rb_pattern:
  33/43, margin +1.728

top6_midlate value_rule_lines rr_pattern_content:
  33/43, margin +1.728

top6_midlate all_source rb_pattern:
  32/43, margin +1.744

top6_midlate all_source br_content:
  14/43, margin +0.873

top6_midlate question_line rr_pattern_content:
  13/43, margin +0.731

top6_midlate question_line br_content:
  10/43, margin +0.618
```

强单 head：

```text
L22_H1 value_rule_lines rr: 13/43, margin +0.739
L22_H1 all_source rr: 13/43, margin +0.730
L22_H3 all_source rr: 7/43, margin +0.397
L22_H7 all_source rr: 7/43, margin +0.377
L22_H3 value_rule_lines rr: 6/43, margin +0.398
L22_H7 value_rule_lines rr: 6/43, margin +0.363
L20_H1 all_source rr: 5/43, margin +0.217
```

random controls：

```text
top6_midlate all_source rr random: 2/43, margin +0.014
top6_midlate value_rule_lines rb_pattern random: 0/43, margin -0.094
top6_midlate value_rule_lines rr random: 0/43, margin -0.046
top6_midlate all_source br_content random: 2/43, margin -0.114
```

DS7B 的客观现象：

```text
1. all_source rr_pattern_content 复现 Phase617 known_top6_midlate：41/43。
2. value_rule_lines alone 可恢复 33/43，是最强单 source group。
3. value_rule_lines rb_pattern 与 rr_pattern_content 完全相同：33/43, margin +1.728。
4. all_source rb_pattern 是 32/43，接近 value_rule_lines rb_pattern。
5. all_source br_content 只有 14/43，明显弱于 pattern。
6. question_line 有辅助，但弱于 value_rule_lines。
7. final_object_category_line 在当前分组下几乎为 0。
```

### 当前最可靠客观事实

1. **DS7B 的主 source 是 value_rule_lines**

```text
top6_midlate value_rule_lines rb_pattern: 33/43
top6_midlate all_source rb_pattern: 32/43
```

说明关键头主要从规则/值行读取，而不是从最后 object-category statement 或 question line 读取。

2. **DS7B 的 residual state builder 主要是 pattern 主导**

```text
rb_pattern: 32/43 to 33/43
br_content: 14/43
rr_pattern_content: 41/43 all_source, 33/43 value_rule_lines
```

这说明修复主要来自 attention pattern 的改变，value content 单独不足。

3. **pattern + content 的全 source 最强**

```text
all_source rr_pattern_content: 41/43
value_rule_lines rb_pattern: 33/43
```

说明 value_rule_lines 是主干，但完整效果还需要其它 source group 或 content 辅助。

4. **L22_H1 是最强单 head source path**

```text
L22_H1 value_rule_lines rr: 13/43
L22_H1 all_source rr: 13/43
```

单 head 仍远弱于 top6 cumulative，说明机制不是单头闭合。

5. **GLM4 继续负结果**

```text
GLM4 top6_midlate all_source rr: 0/12, margin -0.036
```

GLM4 不走当前 head/source/pattern path。

### 理论进展

DS7B 的局部链条现在可以写得更具体：

```text
prompt condition
  -> value_rule_lines attention-pattern shift
  -> L20-L22 top attention head slots
  -> answer-position residual state package
  -> q_proj input
  -> Q/routing pattern
  -> candidate readout
  -> generation gate
```

关键不是简单复制 value content，而是：

```text
模型改变了 answer position 对 value rule lines 的 attention pattern，
从而把正确规则路径写入 residual state。
```

这与 Phase612 的 pattern 主导结论形成一致链条：

```text
最终 routing pattern 重要；
上游 residual state builder 也主要由 pattern shift 驱动。
```

### 问题和硬伤

1. **source group 仍是粗分组**

当前 source group 基于文本行和 token offset，可能有边界误差，尤其 `punct_format` 与 line group 之间可能混入。

2. **final_object_category_line 为 0 需要谨慎解释**

这可能说明它确实不是当前 value gate path 的主 source，也可能是当前分组和 source-aligned prompt 设计让它没有有效 delta。

3. **只测 top6 midlate**

本阶段没有覆盖所有 heads/all layers，不能说已经完成全图谱。

4. **pattern/content split 是 head-slot 层面的近似**

它使用 `alpha * V` 重构 head slot，可能与模型内部实现存在细节差异，但 all_source rr 能复现 41/43，说明近似足够有效。

5. **跨模型仍未统一**

Qwen3 方向部分一致，GLM4 仍然负结果。当前理论不能直接泛化为所有模型都用 `value_rule_lines pattern shift`。

### 下一步任务

Phase619 应继续做：

```text
Source Position Micro-Atlas and Rule-Line Token Audit
```

目标：

```text
把 DS7B 的 value_rule_lines 继续拆成更细的 token-level source 图谱。
```

测试方案：

```text
1. 只对 DS7B 先做主线，因为 DS7B 已经闭合。
2. 固定 top6_midlate L20-L22。
3. 对 value_rule_lines 按 token/短 span 做 sliding source patch。
4. 区分：
   - category token
   - relation token
   - value token
   - punctuation token
   - line-level position
5. 对最强 token spans 做 rb_pattern / br_content / rr split。
6. 加 wrong value line、wrong relation line、random same-norm controls。
```

判据：

```text
如果 value token span 最强：
  说明 residual state builder 主要读取目标值位置。

如果 category/relation token span 最强：
  说明它先定位规则条件，再间接激活值。

如果整行才有效：
  说明 head path 使用的是 line-level relation pattern，不是局部 token。

如果 wrong value line 也强：
  说明存在格式/位置路径污染，需要重新控制。
```

## Phase 619: Rule-Line Token Micro Atlas 值规则行 Token 微图谱 [2026-06-25 07:31]

### 本阶段目标

根据用户提供的 Phase618 分析，先判断其正确性，再继续完成客观现象拼图。

附件分析中正确部分：

```text
1. Phase618 的关键进展不是简单证明 attention 有效，而是把 DS7B 的 value gate 路径拆到 source group 与 pattern/content 层面。
2. DS7B 的主 source group 是 value_rule_lines。
3. DS7B 的主因不是 value content copy，而是 attention pattern shift。
4. value_rule_lines 仍然是粗分组，必须继续拆成 category/relation/value/punctuation/wrong-line controls。
5. 下一步应做 token-level source micro-atlas。
```

本阶段目标：

```text
把 Phase618 的 value_rule_lines 继续拆成更细的 source token group：
  all_value_rule_lines
  correct_rule_line
  correct_category_token
  correct_relation_token
  correct_value_token
  correct_punct_token
  wrong_same_relation_lines
  wrong_same_category_lines
  other_value_rule_lines

继续使用 rb_pattern / br_content / rr_pattern_content 分解，
判断 DS7B 的 value gate residual state builder 到底依赖整行、值词元、类别词元、关系词元，还是错误规则行。
```

### 执行命令

```bash
python -m py_compile \
  tests/glm5/phase619_rule_line_token_micro_atlas.py \
  tests/glm5/phase619_rule_line_token_micro_atlas_summary.py

python tests/glm5/phase619_rule_line_token_micro_atlas.py qwen3 \
  --smoke \
  --compact \
  --include-nontarget \
  --output-dir results/glm5_phase619_rule_line_token_micro_atlas \
  --hard-exit-after-model

python tests/glm5/phase619_rule_line_token_micro_atlas.py qwen3 \
  --confirm \
  --compact \
  --output-dir results/glm5_phase619_rule_line_token_micro_atlas \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/glm5/phase619_rule_line_token_micro_atlas.py glm4 \
  --confirm \
  --compact \
  --output-dir results/glm5_phase619_rule_line_token_micro_atlas \
  --hard-exit-after-model

python tests/glm5/phase619_rule_line_token_micro_atlas.py deepseek7b \
  --confirm \
  --compact \
  --output-dir results/glm5_phase619_rule_line_token_micro_atlas \
  --hard-exit-after-model

python tests/glm5/phase619_rule_line_token_micro_atlas_summary.py
```

GLM4 第一次启动时出现一次 code 139，未进入脚本日志。随后做最小加载诊断：

```bash
PROBE_TORCH_DTYPE=bfloat16 python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('tests/glm5').resolve()))
from phase584_gate_repair import load_model_flash
from model_utils import release_model
m,t,d=load_model_flash('glm4')
print('loaded', type(m).__name__, d)
release_model(m)
PY
```

最小加载成功，重新执行 GLM4 confirm 后完成。判断为一次底层加载瞬时崩溃，不是 Phase619 逻辑错误。

### 脚本与结果

- 主脚本：`tests/glm5/phase619_rule_line_token_micro_atlas.py`
- 汇总脚本：`tests/glm5/phase619_rule_line_token_micro_atlas_summary.py`
- Qwen3 结果：`results/glm5_phase619_rule_line_token_micro_atlas/phase619_qwen3_rule_line_token_micro_atlas_confirm.json`
- GLM4 结果：`results/glm5_phase619_rule_line_token_micro_atlas/phase619_glm4_rule_line_token_micro_atlas_confirm.json`
- DS7B 结果：`results/glm5_phase619_rule_line_token_micro_atlas/phase619_deepseek7b_rule_line_token_micro_atlas_confirm.json`
- 跨模型汇总：`results/glm5_phase619_rule_line_token_micro_atlas/phase619_cross_model_summary.md`

### 测试范围

```text
raw cases = 128
target-only = true
Qwen3 target rows = 9
GLM4 target rows = 12
DS7B target rows = 43

Qwen3 layers = L27-L29
GLM4 layers = L32-L34
DS7B layers = L20-L22

top_k heads = 6
specs = 66
real/random controls = both
```

本阶段使用 compact specs，但不减少 raw cases。compact 只减少微观 source/spec 组合，保留目标样本筛选范围，避免 Phase617 中过重 hook 组合造成底层不稳定。

### 测试原理

Phase618 使用：

```text
z(l,h,t) = sum_s alpha(l,h,t,s) V(l,h,s)
```

并证明 DS7B 的主 source group 是 `value_rule_lines`，且主要是 `rb_pattern`。

Phase619 继续把 `value_rule_lines` 按文本位置和 token offset 拆为：

```text
correct_rule_line:
  正确 category-relation-value 规则行整体。

correct_category_token:
  正确规则行中的 category token。

correct_relation_token:
  正确规则行中的 relation token。

correct_value_token:
  正确规则行中的 value token。

wrong_same_relation_lines:
  relation 相同但 category 不同的错误规则行。

wrong_same_category_lines:
  category 相同但 relation 不同的错误规则行。

all_value_rule_lines:
  所有 category-relation-value 规则行。
```

对每个 group 继续构造：

```text
rb_pattern = repair alpha + base V - base alpha + base V
br_content = base alpha + repair V - base alpha + base V
rr_pattern_content = repair alpha + repair V - base alpha + base V
```

然后把对应 head-slot delta 加到 answer position 的 o_proj input，观察 candidate answer score 是否从旧错误值切换到正确值。

### 客观结果

#### Qwen3

```text
rows = 9
layers = L27-L29
time = 4.14 min

all_value_rule_lines rb_pattern:
  3/9, margin +0.904
  random 1/9, margin +0.055

correct_rule_line rb_pattern:
  2/9, margin +0.473
  random 1/9, margin +0.099

correct_value_token rb_pattern:
  2/9, margin +0.459
  random 0/9, margin -0.047

wrong_same_relation_lines rb_pattern:
  3/9, margin +0.389

correct_category_token rb_pattern:
  1/9, margin -0.028

correct_relation_token rb_pattern:
  1/9, margin -0.014
```

Qwen3 有弱正结果，但 wrong_same_relation_lines 也达到 3/9，说明 Qwen3 的当前路径不是干净的正确值词元定位机制。

#### GLM4

```text
rows = 12
layers = L32-L34
time = 6.88 min

all_value_rule_lines rb_pattern:
  1/12, margin -0.021
  random 0/12, margin -0.004

correct_rule_line rb_pattern:
  0/12, margin -0.036

correct_value_token rb_pattern:
  0/12, margin -0.036

wrong_same_relation_lines rb_pattern:
  2/12, margin +0.026
  random 1/12, margin +0.038
```

GLM4 仍是负结果。当前 DS7B-style value_rule_line token path 不适用于 GLM4。

#### DS7B

```text
rows = 43
layers = L20-L22
time = 20.49 min

all_value_rule_lines rb_pattern:
  32/43, margin +1.735
  correct_delta +1.092
  old_wrong_delta -0.643
  random 0/43, margin +0.029

all_value_rule_lines rr_pattern_content:
  32/43, margin +1.735

correct_rule_line rb_pattern:
  24/43, margin +1.227
  correct_delta +0.917
  old_wrong_delta -0.309
  random 1/43, margin -0.002

correct_rule_line rr_pattern_content:
  24/43, margin +1.227

correct_value_token rb_pattern:
  24/43, margin +1.194
  correct_delta +0.901
  old_wrong_delta -0.293
  random 0/43, margin -0.049

correct_value_token rr_pattern_content:
  24/43, margin +1.194

correct_category_token rb_pattern:
  1/43, margin -0.008

correct_relation_token rb_pattern:
  1/43, margin -0.010

wrong_same_relation_lines rb_pattern:
  5/43, margin +0.302

wrong_same_category_lines rb_pattern:
  2/43, margin +0.064
```

强单 head：

```text
L22_H1 correct_value_token rr:
  8/43, margin +0.515

L22_H1 correct_rule_line rr:
  8/43, margin +0.508

L22_H7 correct_rule_line rr:
  7/43, margin +0.302

L22_H7 correct_value_token rr:
  5/43, margin +0.290
```

### 当前最可靠客观事实

1. **DS7B 的 value_rule_lines 效果可以被 correct_value_token 单独解释大半**

```text
all_value_rule_lines rb_pattern = 32/43, margin +1.735
correct_rule_line rb_pattern = 24/43, margin +1.227
correct_value_token rb_pattern = 24/43, margin +1.194
```

正确值词元几乎等于正确规则行整体：

```text
correct_value_token ≈ correct_rule_line
```

这说明 Phase618 的 `value_rule_lines` 主效应不是均匀来自整行，也不是主要来自 category/relation token，而是集中在正确规则行的 value token 位置。

2. **category/relation token 几乎不能恢复**

```text
correct_category_token rb_pattern = 1/43, margin -0.008
correct_relation_token rb_pattern = 1/43, margin -0.010
```

这说明当前 head path 在 answer position 的恢复不是通过直接读 category 或 relation token 完成，而是通过已经被路由到的 value token 位置产生有效 residual delta。

3. **错误规则行远弱于正确值词元**

```text
wrong_same_relation_lines rb_pattern = 5/43, margin +0.302
wrong_same_category_lines rb_pattern = 2/43, margin +0.064
correct_value_token rb_pattern = 24/43, margin +1.194
```

错误规则行有少量作用，说明存在 relation/format/line-position 污染，但远弱于正确 value token，不能解释主效应。

4. **pattern 仍然主导**

```text
correct_value_token rb_pattern = 24/43, margin +1.194
correct_value_token rr_pattern_content = 24/43, margin +1.194
```

只换 pattern 与 pattern+content 完全一致，继续说明机制主因是 attention pattern shift，不是简单 V content copy。

5. **Qwen3 有弱同向迹象，GLM4 继续负结果**

```text
Qwen3 correct_value_token rb_pattern = 2/9, margin +0.459
GLM4 correct_value_token rb_pattern = 0/12, margin -0.036
```

Qwen3 有部分相似结构，但 wrong_same_relation_lines 也强，说明还不干净。GLM4 不走当前路径。

### 理论进展

DS7B 的局部链条现在可以进一步收紧为：

```text
prompt condition
  -> attention pattern selects correct value token in value rule line
  -> L20-L22 top attention head slots accumulate value-token-local delta
  -> answer-position residual state package
  -> Q/routing state
  -> candidate readout and generation gate
```

Phase618 说的是：

```text
value_rule_lines pattern shift
```

Phase619 进一步说明：

```text
correct value token pattern shift
```

这不是完整的 value gate 闭合，因为 Phase619 仍然没有解释：

```text
为什么 attention pattern 会指向 correct value token？
```

但它把后半段路径压实了：

```text
一旦 pattern 指向正确 value token，DS7B 的 top attention heads 就能在 answer position 写入足够强的 residual state。
```

### 问题和硬伤

1. **Phase619 解释的是 pattern 生效后的读取位置，不是 pattern 生成原因**

当前结果说明 correct value token 是主要被读取位置，但没有解释 Q/K 如何产生这个选择。

2. **all_value_rule_lines 仍强于 correct_value_token**

```text
32/43 vs 24/43
```

说明除正确值词元外，还有其它 token 或行级分布式 pattern 贡献。不能把机制简化成单 token copy。

3. **wrong_same_relation_lines 有弱正效应**

这说明 relation/格式/同列位置可能有污染，下一步必须把 value token 的位置结构与语义值结构分开。

4. **GLM4 继续负结果**

当前 DS7B 链条不能作为跨模型统一理论。GLM4 很可能使用不同的状态压缩或读出机制。

5. **token offset 分组仍是文本级近似**

本阶段使用 tokenizer offset mapping 拆 token group。对于特殊 tokenizer 或空白符 token，边界可能有少量误差。

### 下一步任务

Phase620 应继续做：

```text
Value Token Selection Cause Audit
```

核心目标：

```text
解释为什么 attention pattern 会选择 correct value token。
```

测试方案：

```text
1. 固定 DS7B L20-L22 top heads。
2. 对 correct value token 的 attention score 分解 QK 因子。
3. 对比：
   - correct value token
   - wrong same-relation value token
   - wrong same-category value token
   - random value token
4. 分别 patch：
   - Q from repair
   - K of value token from repair
   - QK score only
   - softmax pattern only
5. 观察是否能复现 Phase619 的 correct_value_token rb_pattern。
```

关键判据：

```text
如果 Q-only patch 可以使 correct value token attention 上升：
  说明 answer-position residual state 已经携带选择条件。

如果 K-only patch 有效：
  说明规则行 token 本身被上游写入了可匹配状态。

如果 Q/K 单独都弱，但 QK score patch 强：
  说明机制是二者耦合，不可分成单边状态。

如果 correct value token 与 wrong same-relation value token 差距来自 softmax competition：
  下一步必须做 value-token competition graph。
```

阶段性大任务：

```text
从 value token 被读取，推进到 value token 为什么被选中。
这会把当前后半段 residual state builder 图谱，连接回 Q/K routing 生成机制。
```

## Phase 620: Value Token Selection Cause Audit 正确值词元选择原因审计 [2026-06-25 07:50]

### 本阶段目标

根据用户提供的 Phase619 分析，先判断其正确性，再继续完成客观现象拼图。

附件分析中正确部分：

```text
1. Phase619 是关键正结果，它把 Phase618 的 value_rule_lines source group 推进到 token-level source atlas。
2. DS7B 的主 source span 是 correct_value_token，而不是 category/relation token。
3. correct_value_token 几乎等于 correct_rule_line 整体，说明主效应集中在正确 value token 位置。
4. wrong_same_relation_lines 有弱正效应，但远弱于 correct_value_token。
5. Phase619 解释的是 pattern 生效后读到了哪里，还没有解释 pattern 为什么会选中 correct value token。
```

本阶段目标：

```text
解释 correct value token 为什么被 attention pattern 选中。

具体拆分：
  q_only:
    repair Q at answer position + base K/V

  k_correct_value:
    base Q + repair K at correct value token + base V

  qk_correct_value:
    repair Q + repair K at correct value token + base V

  k_all_value_rule_lines:
    base Q + repair K at all value rule lines + base V

  qk_all_value_rule_lines:
    repair Q + repair K at all value rule lines + base V

  q_random_same_norm:
    random same-norm Q delta control
```

同时直接测 attention mass：

```text
base attention mass
repair attention mass
q_only patched attention mass
q_random_same_norm attention mass
```

重点观察这些 source groups：

```text
correct_value_token
correct_rule_line
all_value_rule_lines
wrong_same_relation_lines
wrong_same_category_lines
```

### 执行命令

```bash
python -m py_compile \
  tests/glm5/phase620_value_token_selection_cause_audit.py \
  tests/glm5/phase620_value_token_selection_cause_audit_summary.py

python tests/glm5/phase620_value_token_selection_cause_audit.py qwen3 \
  --smoke \
  --include-nontarget \
  --output-dir results/glm5_phase620_value_token_selection_cause_audit \
  --hard-exit-after-model

python tests/glm5/phase620_value_token_selection_cause_audit.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase620_value_token_selection_cause_audit \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/glm5/phase620_value_token_selection_cause_audit.py glm4 \
  --confirm \
  --output-dir results/glm5_phase620_value_token_selection_cause_audit \
  --hard-exit-after-model

python tests/glm5/phase620_value_token_selection_cause_audit.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase620_value_token_selection_cause_audit \
  --hard-exit-after-model

python tests/glm5/phase620_value_token_selection_cause_audit_summary.py
```

三模型按 qwen3、glm4、deepseek7b 顺序执行，每个模型带 `--hard-exit-after-model`。模型测试过程中没有并行加载其它模型。

### 脚本与结果

- 主脚本：`tests/glm5/phase620_value_token_selection_cause_audit.py`
- 汇总脚本：`tests/glm5/phase620_value_token_selection_cause_audit_summary.py`
- Qwen3 结果：`results/glm5_phase620_value_token_selection_cause_audit/phase620_qwen3_value_token_selection_cause_audit_confirm.json`
- GLM4 结果：`results/glm5_phase620_value_token_selection_cause_audit/phase620_glm4_value_token_selection_cause_audit_confirm.json`
- DS7B 结果：`results/glm5_phase620_value_token_selection_cause_audit/phase620_deepseek7b_value_token_selection_cause_audit_confirm.json`
- 跨模型汇总：`results/glm5_phase620_value_token_selection_cause_audit/phase620_cross_model_summary.md`

### 测试范围

```text
raw cases = 128
target-only = true

Qwen3 target rows = 9
GLM4 target rows = 12
DS7B target rows = 43

Qwen3 layers = L27-L29
GLM4 layers = L32-L34
DS7B layers = L20-L22

top_k heads = 6
patch modes = 6
alpha groups = 5
```

### 测试原理

Phase619 已经证明：

```text
DS7B correct_value_token rb_pattern:
  24/43, margin +1.194
```

这说明在 answer position 的 attention pattern 指向正确 value token 时，top heads 可以写入有效 residual state。

Phase620 继续问：

```text
这个 pattern shift 是从 Q 来，还是从 K 来？
```

注意一个关键因果事实：

```text
base prompt 与 repair prompt 的规则行完全位于 question 之前。
自回归模型中，前面的 rule tokens 不可能被后面的 question tokens 改变。
```

因此理论上：

```text
rule-line token 的 K/V 在 base 与 repair 之间应接近相同。
answer-position Q 才是最可能变化的选择因子。
```

本阶段用真实 forward hook 测试：

```text
q_only:
  在 base prompt 中，把 answer position 的 top-head Q 替换成 repair Q。

k_correct_value:
  在 base prompt 中，只替换 correct value token 的 K。

qk_correct_value:
  同时替换 answer Q 和 correct value token K。

q_random_same_norm:
  给 Q 加同范数随机 delta，作为方向对照。
```

如果：

```text
q_only ≈ qk_correct_value ≫ k_correct_value ≈ q_random
```

说明 correct value token selection 的主因是 answer-position Q，而不是 source-token K field。

同时直接测 attention mass：

```text
repair_mass - base_mass
q_only_mass - base_mass
q_random_mass - base_mass
```

如果：

```text
q_only_mass - base_mass ≈ repair_mass - base_mass
```

说明 q_only 不只是行为上修复，还真的复现了 attention pattern 对 correct value token 的转移。

### 客观结果

#### Qwen3

```text
rows = 9
time = 0.58 min

q_only:
  4/9, margin +1.182
  correct_delta +0.757
  old_wrong_delta -0.425

qk_correct_value:
  4/9, margin +1.182

qk_all_value_rule_lines:
  4/9, margin +1.182

k_correct_value:
  0/9, margin +0.000

k_all_value_rule_lines:
  0/9, margin +0.000

q_random_same_norm:
  0/9, margin +0.000
```

attention mass：

```text
correct_value_token:
  repair-base = +0.05108
  q-base      = +0.05123
  random-base = +0.00152

correct_rule_line:
  repair-base = +0.05314
  q-base      = +0.05329
  random-base = +0.00183

wrong_same_relation_lines:
  repair-base = -0.03511
  q-base      = -0.03556
```

Qwen3 有清楚的 Q-driven selection 迹象，但行为闭合只有 4/9。

#### GLM4

```text
rows = 12
time = 0.94 min

q_only:
  1/12, margin +0.016

qk_correct_value:
  1/12, margin +0.016

qk_all_value_rule_lines:
  1/12, margin +0.016

k_correct_value:
  0/12, margin +0.000

k_all_value_rule_lines:
  0/12, margin +0.000

q_random_same_norm:
  0/12, margin +0.000
```

attention mass：

```text
correct_value_token:
  repair-base = +0.02190
  q-base      = +0.02178
  random-base = +0.00105

correct_rule_line:
  repair-base = +0.02348
  q-base      = +0.02331
  random-base = +0.00103
```

GLM4 有弱 Q-driven attention shift，但行为上几乎不闭合。这继续说明 GLM4 不走当前 DS7B-style value gate path，或者后续 readout/generation gate 抑制更强。

#### DS7B

```text
rows = 43
time = 1.38 min

q_only:
  33/43, margin +1.769
  correct_delta +1.113
  old_wrong_delta -0.656

qk_correct_value:
  33/43, margin +1.769

qk_all_value_rule_lines:
  33/43, margin +1.769

k_correct_value:
  0/43, margin +0.000

k_all_value_rule_lines:
  0/43, margin +0.000

q_random_same_norm:
  0/43, margin +0.000
```

attention mass：

```text
correct_value_token:
  repair-base = +0.09897
  q-base      = +0.10177
  random-base = -0.00236

correct_rule_line:
  repair-base = +0.11126
  q-base      = +0.11486
  random-base = -0.00202

all_value_rule_lines:
  repair-base = +0.08633
  q-base      = +0.09741
  random-base = -0.00388

wrong_same_relation_lines:
  repair-base = -0.01620
  q-base      = -0.01321
  random-base = +0.00050
```

### 当前最可靠客观事实

1. **DS7B correct value token selection 的主因是 answer-position Q**

```text
q_only = 33/43, margin +1.769
qk_correct_value = 33/43, margin +1.769
k_correct_value = 0/43, margin +0.000
q_random_same_norm = 0/43, margin +0.000
```

这说明 K 不提供额外解释力。只替换 answer-position Q，就足以复现并略强于 Phase619 的 correct value token anchor。

2. **q_only 真实复现了 attention pattern shift**

```text
correct_value_token repair-base = +0.09897
correct_value_token q-base      = +0.10177
correct_value_token random-base = -0.00236
```

q_only 不只是分数变好，而是真的把 attention mass 推向 correct_value_token。

3. **K-only 完全无效符合因果结构**

规则行在 question 之前，因此 rule-token K/V 不应被后文改变。实验结果：

```text
k_correct_value = 0/43
k_all_value_rule_lines = 0/43
```

这与自回归因果结构一致。

4. **Qwen3 有同向机制，但不如 DS7B 闭合**

```text
Qwen3 q_only = 4/9, margin +1.182
Qwen3 correct_value_token q-base ≈ repair-base
```

Qwen3 的 Q-driven pattern shift 存在，但 target rows 少，且 Phase619 中 wrong_same_relation_lines 干扰更明显。

5. **GLM4 有 attention shift，但行为不闭合**

```text
GLM4 correct_value_token q-base ≈ repair-base ≈ +0.022
GLM4 q_only = 1/12
```

这说明 GLM4 的问题可能不在 attention selection 本身，而在 downstream residual/readout/generation gate。

### 理论进展

DS7B 当前链条可以进一步收紧：

```text
prompt condition
  -> answer-position Q state changes
  -> Q selects correct value token through attention pattern
  -> L20-L22 top attention heads read correct value token
  -> answer-position residual state package
  -> candidate readout and generation gate
```

Phase619 的问题是：

```text
为什么 pattern 选中 correct value token？
```

Phase620 给出的答案是：

```text
因为 answer-position Q 已经被上游 residual state 改写；
这个 Q 足以把 attention mass 从错误/无关规则行转到 correct value token。
```

这也把 Phase613 与 Phase619 连接起来：

```text
Phase613:
  q_only 是充分修复因子。

Phase619:
  correct_value_token 是主要读取位置。

Phase620:
  q_only 直接造成 correct_value_token attention mass 增加。
```

### 问题和硬伤

1. **Phase620 仍没有解释 Q state 从哪里来**

当前证明的是：

```text
Q state 是 value token selection 的充分因子。
```

但没有解释：

```text
哪个上游模块生成了这个 Q state？
```

这仍然要回到 Phase615/616 的 residual state builder。

2. **q_only 是人工替换，不是自然生成闭环**

虽然 q_only 行为和 attention mass 都高度复现 repair，但它仍然是 intervention，不是自然路径的完整生成解释。

3. **DS7B all_value_rule_lines q-base 稍强于 repair-base**

```text
all_value_rule_lines repair-base = +0.08633
all_value_rule_lines q-base = +0.09741
```

说明 Q patch 可能带来略强或略粗的规则行吸引，不是完全等同于自然 repair。

4. **GLM4 的 Q shift 不能转化成行为闭合**

这说明跨模型统一理论仍然没有完成。GLM4 可能卡在 readout/generation gate，不是 selection gate。

5. **attention mass 是 selected top heads 的均值**

本阶段没有对全部 heads 做完整图谱，只测 Phase617/619 已定位的 top heads。

### 下一步任务

Phase621 应继续做：

```text
Q State Builder Backtrace
```

目标：

```text
解释 answer-position repair Q state 是如何由上游 residual state 生成的。
```

测试方案：

```text
1. 固定 DS7B L20-L22 top heads 作为下游 selection gate。
2. 在 q_proj input 之前，逐层 patch residual stream：
   - layer input
   - attention output
   - MLP output
   - layer output
3. 测两个指标：
   - Q vector similarity / Q delta projection
   - downstream correct_value_token attention mass
   - candidate answer switch
4. 对比：
   - base residual
   - repair residual
   - random same-norm residual
   - wrong-condition residual
```

关键判据：

```text
如果某个上游 residual patch 能同时恢复：
  1. repair Q state
  2. correct_value_token attention mass
  3. final candidate switch

则该位置就是 Q state builder 的上游候选节点。
```

阶段性大任务：

```text
从“Q 是选择原因”继续回溯到“Q 由谁生成”。
把 value token selection gate 连接回 residual state builder 图谱。
```

## Phase 621: Q State Builder Backtrace Q 状态生成器回溯 [2026-06-25 08:53]

### 本阶段目标

根据用户提供的 Phase620 分析，先判断其正确性，再继续完成客观现象拼图。

附件分析中正确部分：

```text
1. Phase620 是关键正结果，把 Phase619 的“读到 correct value token”推进为“answer-position Q 足以选择 correct value token”。
2. DS7B 中 q_only = qk_correct_value = qk_all_value_rule_lines，而 K-only 完全无效。
3. q_only 不只是行为有效，也真实复现了 correct_value_token attention mass shift。
4. Qwen3 有同向机制但闭合较弱。
5. GLM4 有 attention shift 但行为不闭合，说明 GLM4 可能卡在 downstream readout/generation gate。
6. 当前瓶颈已经从“为什么读 correct value token”推进到“answer-position Q state 由谁生成”。
```

本阶段目标：

```text
回溯 answer-position repair Q state 的上游 residual builder。

测试一个上游 residual component patch 是否同时恢复：
  1. candidate answer switch
  2. selected-head Q delta projection
  3. correct_value_token attention mass
```

### 执行命令

```bash
python -m py_compile \
  tests/glm5/phase621_q_state_builder_backtrace.py \
  tests/glm5/phase621_q_state_builder_backtrace_summary.py

python tests/glm5/phase621_q_state_builder_backtrace.py qwen3 \
  --smoke \
  --include-nontarget \
  --output-dir results/glm5_phase621_q_state_builder_backtrace \
  --hard-exit-after-model

python tests/glm5/phase621_q_state_builder_backtrace.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase621_q_state_builder_backtrace \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/glm5/phase621_q_state_builder_backtrace.py glm4 \
  --confirm \
  --output-dir results/glm5_phase621_q_state_builder_backtrace \
  --hard-exit-after-model

python tests/glm5/phase621_q_state_builder_backtrace.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase621_q_state_builder_backtrace \
  --hard-exit-after-model

python tests/glm5/phase621_q_state_builder_backtrace_summary.py
```

### 脚本与结果

- 主脚本：`tests/glm5/phase621_q_state_builder_backtrace.py`
- 汇总脚本：`tests/glm5/phase621_q_state_builder_backtrace_summary.py`
- Qwen3 结果：`results/glm5_phase621_q_state_builder_backtrace/phase621_qwen3_q_state_builder_backtrace_confirm.json`
- GLM4 结果：`results/glm5_phase621_q_state_builder_backtrace/phase621_glm4_q_state_builder_backtrace_confirm.json`
- DS7B 结果：`results/glm5_phase621_q_state_builder_backtrace/phase621_deepseek7b_q_state_builder_backtrace_confirm.json`
- 跨模型汇总：`results/glm5_phase621_q_state_builder_backtrace/phase621_cross_model_summary.md`

### 测试范围

```text
raw cases = 128
target-only = true

Qwen3 target rows = 9
GLM4 target rows = 12
DS7B target rows = 43

patch components:
  layer_input
  attn_out
  mlp_out
  layer_out

Qwen3 patch layers = L25-L29
Qwen3 selection layers = L27-L29

GLM4 patch layers = L30-L34
GLM4 selection layers = L32-L34

DS7B patch layers = L18-L22
DS7B selection layers = L20-L22
```

### 测试原理

Phase620 已经证明：

```text
answer-position Q is sufficient for correct value token selection.
```

Phase621 继续问：

```text
哪个上游 residual component 能生成这个 Q？
```

本阶段对每个上游 layer/component 做 repair-base delta patch：

```text
patched_state = base_state + (repair_state - base_state)
```

并加 random same-norm control。

每个 patch 同时测三类指标：

```text
1. candidate switch:
   是否把候选答案从 old wrong value 切到 correct value。

2. Q delta projection:
   patched Q delta 在 repair Q delta 方向上的投影。

3. correct_value_token alpha delta:
   patched attention mass 相比 base 是否增加到 correct value token。
```

重要解释：

```text
如果 layer_out 有强 switch，但 qproj=0 且 alpha_cv=0，
它不是 Q builder，而是下游结果态/读出态。

如果 layer_input 或上一层 layer_out 同时有强 switch、qproj>0、alpha_cv>0，
它才是 Q state builder 路径上的候选节点。
```

### 客观结果

#### Qwen3

```text
rows = 9
time = 1.16 min

L29 layer_out:
  switch 9/9, margin +4.392
  qproj +0.000
  alpha_cv +0.00000

L28 layer_out:
  switch 9/9, margin +4.392
  qproj +0.333
  alpha_cv +0.03655

L29 layer_input:
  switch 9/9, margin +4.392
  qproj +0.333
  alpha_cv +0.03655

L25 layer_out:
  switch 9/9, margin +4.336
  qproj +0.976
  alpha_cv +0.05070

L26 layer_input:
  switch 9/9, margin +4.336
  qproj +0.976
  alpha_cv +0.05070

L26 layer_out:
  switch 9/9, margin +4.336
  qproj +0.988
  alpha_cv +0.05145

L27 layer_input:
  switch 9/9, margin +4.336
  qproj +0.988
  alpha_cv +0.05145
```

Qwen3 显示出连续 residual carrying：

```text
上一层 layer_out ≈ 下一层 layer_input
```

但 Qwen3 target rows 只有 9，且 Phase619 中 source micro-atlas 不如 DS7B 干净，所以仍需保守。

#### GLM4

```text
rows = 12
time = 2.18 min

L33 layer_out:
  switch 11/12, margin +1.932
  qproj +0.333
  alpha_cv +0.00355

L34 layer_input:
  switch 11/12, margin +1.932
  qproj +0.333
  alpha_cv +0.00355

L34 layer_out:
  switch 11/12, margin +1.917
  qproj +0.000
  alpha_cv +0.00000

L32 layer_out:
  switch 11/12, margin +1.911
  qproj +0.667
  alpha_cv +0.01098

L33 layer_input:
  switch 11/12, margin +1.911
  qproj +0.667
  alpha_cv +0.01098

L31 layer_out:
  switch 10/12, margin +1.932
  qproj +0.999
  alpha_cv +0.02127

L32 layer_input:
  switch 10/12, margin +1.932
  qproj +0.999
  alpha_cv +0.02127
```

GLM4 的上游 residual patch 能强烈改变候选分数，但 alpha_cv 很弱。这继续支持：

```text
GLM4 不是当前 DS7B-style correct_value_token selection gate 主瓶颈。
```

#### DS7B

```text
rows = 43
time = 5.00 min

L21 layer_out:
  switch 43/43, margin +3.370
  qproj +0.333
  alpha_cv +0.08571

L22 layer_input:
  switch 43/43, margin +3.370
  qproj +0.333
  alpha_cv +0.08571

L22 layer_out:
  switch 42/43, margin +3.337
  qproj +0.000
  alpha_cv +0.00000

L20 layer_out:
  switch 42/43, margin +3.111
  qproj +0.646
  alpha_cv +0.09005

L21 layer_input:
  switch 42/43, margin +3.111
  qproj +0.646
  alpha_cv +0.09005

L22 attn_out:
  switch 32/43, margin +1.738
  qproj +0.000
  alpha_cv +0.00000

L19 layer_out:
  switch 31/43, margin +1.953
  qproj +0.779
  alpha_cv +0.05724

L20 layer_input:
  switch 31/43, margin +1.953
  qproj +0.779
  alpha_cv +0.05724

L18 layer_out:
  switch 30/43, margin +1.955
  qproj +0.736
  alpha_cv +0.05701

L19 layer_input:
  switch 30/43, margin +1.955
  qproj +0.736
  alpha_cv +0.05701
```

### 当前最可靠客观事实

1. **DS7B 的 Q state builder 主要是 residual stream carried state**

最清楚链条：

```text
L20 layer_out -> L21 layer_input:
  42/43, qproj +0.646, alpha_cv +0.09005

L21 layer_out -> L22 layer_input:
  43/43, qproj +0.333, alpha_cv +0.08571
```

说明 Q state 不是某个孤立组件单点生成，而是在 residual stream 中逐层携带、逐层改写。

2. **L22 layer_out / L22 attn_out 是下游结果态，不是 Q builder**

```text
L22 layer_out:
  switch 42/43
  qproj 0
  alpha_cv 0

L22 attn_out:
  switch 32/43
  qproj 0
  alpha_cv 0
```

这些位置发生在 L22 Q 生成之后，因此能改最终分数，但不能解释 correct value token selection。

3. **早期 residual state 已经含有部分 Q builder 信息**

```text
L18 layer_out -> L19 layer_input:
  30/43, qproj +0.736, alpha_cv +0.05701

L19 layer_out -> L20 layer_input:
  31/43, qproj +0.779, alpha_cv +0.05724
```

说明 repair Q state 的形成不是只在 L21/L22 突然出现，而是在 L18-L21 连续积累。

4. **MLP 不是当前主 builder**

DS7B：

```text
L20 mlp_out:
  11/43, margin +0.418, qproj +0.112, alpha_cv +0.00170

L21 mlp_out:
  6/43, margin +0.344, qproj +0.048, alpha_cv +0.00821
```

MLP 有弱贡献，但远弱于 layer_input/layer_out carried state。

5. **GLM4 行为强但 selection 指标弱**

GLM4 的 layer patch 能强切换候选，但 alpha_cv 很弱，说明它的行为修复可能不是靠当前 correct_value_token attention path。

### 理论进展

DS7B 当前链条进一步收紧为：

```text
prompt condition
  -> L18-L21 residual stream gradually carries repair state
  -> L20/L21 layer_out -> next layer_input
  -> answer-position Q state at L20-L22 changes
  -> Q selects correct value token
  -> top attention heads read correct value token
  -> L22 attention/result state changes
  -> candidate readout and generation gate
```

Phase620 证明：

```text
Q 是 correct value token selection 的充分因子。
```

Phase621 进一步证明：

```text
这个 Q 不是凭空出现，而是由 answer-position residual stream carried state 生成。
```

### 问题和硬伤

1. **当前 patch 是整向量 residual patch**

它能说明哪个位置携带 repair state，但还不能说明该 state 内部哪些维度/子方向是关键。

2. **layer_out 与 layer_input 等价需要进一步分解**

上一层 layer_out 与下一层 layer_input 近似等价，这是残差流自然结构，但还没有说明这个 state 是由上一层 attention 还是更早 state 累积产生。

3. **attn_out 单独弱于 layer_out/layer_input**

这说明单个组件输出不足以解释全部 carried state，可能需要 cumulative residual patch 或多层共同作用。

4. **GLM4 机制仍未统一**

GLM4 的强行为 patch 与弱 alpha_cv 指标不匹配，说明 GLM4 路线可能需要单独图谱。

5. **Q projection 指标是 top-head 平均**

本阶段没有把每个 head 单独拆开，可能掩盖少数强 head 与弱 head 的差异。

### 下一步任务

Phase622 应继续做：

```text
Residual State Direction Decomposition
```

核心目标：

```text
把 L18-L22 residual carried state 从整向量 patch 拆成：
  Q-aligned component
  Q-orthogonal component
  correct-value alpha aligned component
  random same-norm controls
```

测试方案：

```text
1. 固定 DS7B 为主模型。
2. 对 L18-L22 layer_input/layer_out 的 repair-base residual delta 做方向分解：
   - project onto downstream repair Q delta
   - remove Q-aligned component
   - only Q-orthogonal component
3. 测三指标：
   - candidate switch
   - Q delta projection
   - correct_value_token alpha mass
4. 加 random same-norm 与 wrong-condition residual control。
```

关键判据：

```text
如果 Q-aligned component 足以恢复：
  residual carried state 的关键方向就是 Q builder direction。

如果 Q-orthogonal component 仍强：
  说明 residual state 中存在非 Q 但能影响后续 gate 的隐藏因子。

如果两者都弱、整向量强：
  说明当前 state 是多方向耦合，不可线性拆成单一方向。
```

阶段性大任务：

```text
从“哪一层携带 Q builder state”
推进到“这个 state 内部是什么方向结构”。
```

## Phase 622: Residual State Direction Decomposition 残差状态方向分解 [2026-06-25 09:12]

### 本阶段目标

根据用户提供的 Phase620/621 分析，先判断其正确性，再继续完成客观现象拼图。

附件分析中正确部分：

```text
1. Phase620 证明 answer-position Q 是 correct value token selection 的充分因子。
2. Phase621 证明 Q 不是起点，而是 residual stream 中已经形成的状态被 q_proj 读出的接口。
3. DS7B 的 Q state builder 主要表现为 L18-L21 residual stream carried state。
4. L22 layer_out / L22 attn_out 能改最终输出，但 qproj=0、alpha_cv=0，因此是下游结果态，不是 Q builder。
5. 当前硬伤是整向量 residual patch 还没拆出内部方向结构。
```

本阶段目标：

```text
把 residual carried state 从整向量 patch 拆成：
  full_delta
  q_backproj_aligned
  q_backproj_orthogonal
  random_same_norm

判断强效 residual state 是否主要由 Q-builder 方向解释，还是存在强 Q-orthogonal 因果分量。
```

### 执行命令

```bash
python -m py_compile \
  tests/glm5/phase622_residual_state_direction_decomposition.py \
  tests/glm5/phase622_residual_state_direction_decomposition_summary.py

python tests/glm5/phase622_residual_state_direction_decomposition.py qwen3 \
  --smoke \
  --include-nontarget \
  --output-dir results/glm5_phase622_residual_state_direction_decomposition \
  --hard-exit-after-model

python tests/glm5/phase622_residual_state_direction_decomposition.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase622_residual_state_direction_decomposition \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/glm5/phase622_residual_state_direction_decomposition.py glm4 \
  --confirm \
  --output-dir results/glm5_phase622_residual_state_direction_decomposition \
  --hard-exit-after-model

python tests/glm5/phase622_residual_state_direction_decomposition.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase622_residual_state_direction_decomposition \
  --hard-exit-after-model

python tests/glm5/phase622_residual_state_direction_decomposition_summary.py
```

### 脚本与结果

- 主脚本：`tests/glm5/phase622_residual_state_direction_decomposition.py`
- 汇总脚本：`tests/glm5/phase622_residual_state_direction_decomposition_summary.py`
- Qwen3 结果：`results/glm5_phase622_residual_state_direction_decomposition/phase622_qwen3_residual_state_direction_decomposition_confirm.json`
- GLM4 结果：`results/glm5_phase622_residual_state_direction_decomposition/phase622_glm4_residual_state_direction_decomposition_confirm.json`
- DS7B 结果：`results/glm5_phase622_residual_state_direction_decomposition/phase622_deepseek7b_residual_state_direction_decomposition_confirm.json`
- 跨模型汇总：`results/glm5_phase622_residual_state_direction_decomposition/phase622_cross_model_summary.md`

### 测试范围

```text
raw cases = 128
target-only = true

Qwen3 target rows = 9
GLM4 target rows = 12
DS7B target rows = 43

components = layer_input, layer_out
modes = full_delta, q_backproj_aligned, q_backproj_orthogonal, random_same_norm

Qwen3 patch layers = L25-L29
Qwen3 selection layers = L27-L29

GLM4 patch layers = L30-L34
GLM4 selection layers = L32-L34

DS7B patch layers = L18-L22
DS7B selection layers = L20-L22
```

### 测试原理

Phase621 使用整向量 residual patch：

```text
delta_h = h_repair - h_base
```

Phase622 构造一个 Q-backprojected direction：

```text
u_Q = sum_l W_Q(l)^T delta_Q_selected(l)
```

其中：

```text
delta_Q_selected(l)
```

只保留 Phase620/621 中的 selected top heads 的 repair-base Q delta。

然后把 residual delta 分解为：

```text
delta_aligned = Proj_{u_Q}(delta_h)
```

```text
delta_orthogonal = delta_h - delta_aligned
```

再分别 patch：

```text
full_delta:
  base + delta_h

q_backproj_aligned:
  base + delta_aligned

q_backproj_orthogonal:
  base + delta_orthogonal

random_same_norm:
  base + random_same_norm(delta_h)
```

每个 patch 继续测：

```text
1. candidate switch
2. q_delta_projection
3. correct_value_token alpha delta
4. wrong_same_relation alpha delta
```

判据：

```text
如果 q_backproj_aligned 恢复 qproj 和 alpha_cv，并产生候选切换，
说明该 residual state 的 selection-gate 因果方向主要是 Q-builder 方向。

如果 q_backproj_orthogonal 也强，
说明 residual state 还包含非 Q selection 的下游/读出因子。
```

### 客观结果

#### Qwen3

```text
rows = 9
time = 1.23 min

L29 layer_input full_delta:
  switch 9/9, margin +4.392
  qproj +0.333
  alpha_cv +0.03655

L29 layer_input q_backproj_aligned:
  switch 5/9, margin +1.085
  qproj +0.244
  alpha_cv +0.01818
  norm_ratio +0.462

L29 layer_input q_backproj_orthogonal:
  switch 7/9, margin +3.294
  qproj +0.093
  alpha_cv +0.01309
  norm_ratio +0.887
```

Qwen3 的 Q-aligned 分量能恢复部分 selection 指标，但 Q-orthogonal 分量仍有强行为效应。由于 target rows 只有 9，仍需谨慎。

#### GLM4

```text
rows = 12
time = 2.33 min

L34 layer_input full_delta:
  switch 11/12, margin +1.932
  qproj +0.333
  alpha_cv +0.00355

L34 layer_input q_backproj_aligned:
  switch 1/12, margin +0.089
  qproj +0.266
  alpha_cv +0.00511
  norm_ratio +0.272

L34 layer_input q_backproj_orthogonal:
  switch 10/12, margin +1.625
  qproj +0.069
  alpha_cv -0.00101
  norm_ratio +0.962
```

GLM4 的行为主要在 Q-orthogonal 分量，而不是 correct value token selection 分量。继续说明 GLM4 当前任务的行为修复不等于 DS7B-style selection gate 修复。

#### DS7B

```text
rows = 43
time = 5.43 min

L20 layer_out full_delta:
  switch 42/43, margin +3.111
  qproj +0.646
  alpha_cv +0.09005

L20 layer_out q_backproj_aligned:
  switch 36/43, margin +2.036
  qproj +0.547
  alpha_cv +0.07803
  norm_ratio +0.470

L20 layer_out q_backproj_orthogonal:
  switch 16/43, margin +1.015
  qproj +0.125
  alpha_cv +0.01674
  norm_ratio +0.882

L21 layer_out full_delta:
  switch 43/43, margin +3.370
  qproj +0.333
  alpha_cv +0.08571

L21 layer_out q_backproj_aligned:
  switch 37/43, margin +1.970
  qproj +0.243
  alpha_cv +0.06019
  norm_ratio +0.455

L21 layer_out q_backproj_orthogonal:
  switch 20/43, margin +1.206
  qproj +0.099
  alpha_cv +0.02099
  norm_ratio +0.890

L22 layer_out full_delta:
  switch 42/43, margin +3.337
  qproj +0.000
  alpha_cv +0.00000

L22 layer_out q_backproj_aligned:
  switch 10/43, margin +0.666
  qproj +0.000
  alpha_cv +0.00000
  norm_ratio +0.385

L22 layer_out q_backproj_orthogonal:
  switch 38/43, margin +2.753
  qproj +0.000
  alpha_cv +0.00000
  norm_ratio +0.922

random controls:
  L20 layer_out random_same_norm:
    2/43, margin -0.089
  L21 layer_out random_same_norm:
    1/43, margin -0.028
  L22 layer_out random_same_norm:
    4/43, margin -0.014
```

### 当前最可靠客观事实

1. **DS7B 上游 Q-builder 节点主要由 Q-backprojected aligned component 解释**

```text
L20 layer_out aligned:
  36/43, qproj +0.547, alpha_cv +0.07803

L21 layer_out aligned:
  37/43, qproj +0.243, alpha_cv +0.06019
```

虽然 norm_ratio 只有 0.45 到 0.47，但已经恢复大部分 correct value token selection 指标。

2. **Q-orthogonal component 仍有非零行为效应，但 selection 指标弱**

```text
L20 layer_out orthogonal:
  16/43, qproj +0.125, alpha_cv +0.01674

L21 layer_out orthogonal:
  20/43, qproj +0.099, alpha_cv +0.02099
```

这说明 residual state 中还有非 Q selection 的因果成分，但它不是主 selection gate。

3. **L22 layer_out 是下游结果态，主要在 Q-orthogonal 分量**

```text
L22 layer_out orthogonal:
  38/43, qproj 0, alpha_cv 0

L22 layer_out aligned:
  10/43, qproj 0, alpha_cv 0
```

这非常清楚地区分了：

```text
selection-state direction:
  L20/L21 layer_out aligned component

downstream result/readout state:
  L22 layer_out orthogonal component
```

4. **random controls 很弱**

```text
L20 random = 2/43
L21 random = 1/43
L22 random = 4/43
```

说明强效不是范数导致。

5. **GLM4 的行为效应主要是 Q-orthogonal**

GLM4：

```text
L34 layer_input aligned = 1/12
L34 layer_input orthogonal = 10/12
```

这进一步说明 GLM4 的当前强行为 patch 不属于 DS7B-style correct value token selection gate。

### 理论进展

DS7B 当前链条进一步分裂成两条不同状态：

```text
1. selection state:
   L20/L21 residual carried state
   -> Q-backprojected aligned component
   -> Q state
   -> correct value token attention mass
   -> value token read

2. result/readout state:
   L22 layer_out
   -> Q-orthogonal component
   -> candidate/readout/generation score change
```

这解释了 Phase621 中一个重要现象：

```text
为什么 L22 layer_out 很强，但 qproj=0、alpha_cv=0？
```

因为它不是 Q builder，而是 selection 之后的结果态。

### 问题和硬伤

1. **Q-backproject direction 是线性近似**

它使用：

```text
W_Q^T delta_Q
```

作为 residual space 中的 Q-builder 方向，但真实路径还有 layernorm 和非线性上下文影响。

2. **aligned + orthogonal 都不是完全闭合**

DS7B 中 full_delta 仍强于 aligned，说明 residual state 不是单方向机制。

3. **orthogonal component 仍有中等行为效应**

这说明还有 readout/result/generation 方向，需要单独建图谱。

4. **只做 top-head selected Q backprojection**

未覆盖所有 heads，可能漏掉非 top head 的辅助方向。

5. **跨模型仍不统一**

Qwen3 和 GLM4 都出现强 Q-orthogonal 行为分量，不能简单套用 DS7B 的 selection-state 解释。

### 下一步任务

Phase623 应继续做：

```text
Selection State vs Result State Separation
```

核心目标：

```text
把 DS7B 中两类状态明确分开：
  selection state:
    能恢复 Q/alpha/correct_value_token selection。

  result state:
    不恢复 Q/alpha，但能改变 candidate score。
```

测试方案：

```text
1. 固定 DS7B。
2. 选择两个代表位置：
   - L20/L21 layer_out q_backproj_aligned = selection state
   - L22 layer_out q_backproj_orthogonal = result state
3. 分别测：
   - candidate score delta vector
   - correct/wrong candidate logit movement
   - attention alpha movement
   - downstream final norm/readout projection
4. 做组合 patch：
   - selection only
   - result only
   - selection + result
   - random controls
5. 判断两类状态是否加和、竞争或相互覆盖。
```

关键判据：

```text
如果 selection + result > max(selection, result):
  两者是互补链条。

如果 selection + result ≈ result:
  result state 覆盖 selection state，说明后端读出更强。

如果 selection + result < selection:
  存在方向冲突或门控竞争。
```

阶段性大任务：

```text
从单一路径解释转向双状态图谱：
  selection state 负责找值；
  result state 负责把找到的值变成候选得分。
```

## Phase 623: Selection State vs Result State Separation 选择状态与结果状态分离 [2026-06-25 09:57]

### 本阶段目标

根据用户上传的 Phase622 附加分析，先判断其正确性，再继续完成任务。

附加分析的核心判断基本正确：

```text
1. Phase620 已经证明 correct value token selection 主要由 answer-position Q 驱动，而不是 K-only。
2. Phase621 把 Q state builder 回溯到 L20/L21 layer_out，L22 layer_out 更像 downstream result/readout state。
3. Phase622 的 Q-backprojection aligned / orthogonal 分解是合理下一步。
4. DS7B 中 L20/L21 aligned component 带来 Q/alpha 恢复，L22 orthogonal component 带来强行为恢复。
5. 下一步不能再只问单点 patch 是否成功，而要问 selection state 与 result state 是互补、冗余还是覆盖。
```

需要修正的地方：

```text
不能把 Q-aligned component 直接等同于完整 value selection mechanism。
它只是当前测量体系中最清楚的 selection-state 分量。

不能把 Q-orthogonal component 直接叫最终 readout。
它目前只能更谨慎地称为 result/readout-like state，因为还没有完成 logits head、unembedding、后续层洗出路径的完整闭环。
```

本阶段目标：

```text
把 selection state 和 result state 拆开测试：
  selection state:
    L20/L21 或对应模型近邻层的 q_backproj_aligned component。
    预期恢复 Q projection 和 correct_value_token attention alpha。

  result state:
    下游层 q_backproj_orthogonal component。
    预期不恢复 Q/alpha，但能直接改善 candidate score。

测试两者组合：
  selection only
  result only
  selection + result
  random controls
```

### 生成脚本

```text
tests/glm5/phase623_selection_result_state_separation.py
tests/glm5/phase623_selection_result_state_separation_summary.py
```

### 执行命令

静态检查：

```bash
python -m py_compile \
  tests/glm5/phase623_selection_result_state_separation.py \
  tests/glm5/phase623_selection_result_state_separation_summary.py
```

烟测：

```bash
python tests/glm5/phase623_selection_result_state_separation.py qwen3 \
  --smoke \
  --include-nontarget \
  --output-dir results/glm5_phase623_selection_result_state_separation \
  --hard-exit-after-model
```

正式加大确认测试：

```bash
python tests/glm5/phase623_selection_result_state_separation.py qwen3 \
  --confirm \
  --n-tables 32 \
  --max-samples 256 \
  --output-dir results/glm5_phase623_selection_result_state_separation \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/glm5/phase623_selection_result_state_separation.py glm4 \
  --confirm \
  --n-tables 32 \
  --max-samples 256 \
  --output-dir results/glm5_phase623_selection_result_state_separation \
  --hard-exit-after-model

python tests/glm5/phase623_selection_result_state_separation.py deepseek7b \
  --confirm \
  --n-tables 32 \
  --max-samples 256 \
  --output-dir results/glm5_phase623_selection_result_state_separation \
  --hard-exit-after-model

python tests/glm5/phase623_selection_result_state_separation_summary.py
```

### 输出文件

```text
results/glm5_phase623_selection_result_state_separation/phase623_qwen3_selection_result_state_separation_confirm.json
results/glm5_phase623_selection_result_state_separation/phase623_glm4_selection_result_state_separation_confirm.json
results/glm5_phase623_selection_result_state_separation/phase623_deepseek7b_selection_result_state_separation_confirm.json
results/glm5_phase623_selection_result_state_separation/phase623_cross_model_summary.md
```

### 测试范围

```text
models = qwen3, glm4, deepseek7b
raw cases/model = 256
candidate values = first 4 value candidates
target cases = base wrong and repair prompt correct

qwen3:
  patch layers = L26, L27, L29
  selection layers = L27, L28, L29

GLM4:
  patch layers = L31, L32, L34
  selection layers = L32, L33, L34

DS7B:
  patch layers = L20, L21, L22
  selection layers = L20, L21, L22
```

测试模式：

```text
selection_early:
  early q_backproj_aligned component only

selection_late:
  late q_backproj_aligned component only

selection_both:
  early + late q_backproj_aligned components

result_only:
  downstream q_backproj_orthogonal component only

selection_both_plus_result:
  selection_both + result_only

selection_late_plus_result:
  selection_late + result_only

selection_random_norm:
  selection layers random same-norm control

result_random_norm:
  result layer random same-norm control
```

### 客观结果

#### Qwen3

有效 target cases：

```text
17 / 256
```

主要结果：

```text
selection_both_plus_result:
  switch = 15/17
  margin gain = +4.407
  q projection = +0.847
  correct_value alpha = +0.03001

result_only:
  switch = 15/17
  margin gain = +4.407
  q projection = 0.000
  correct_value alpha = 0.00000

selection_late_plus_result:
  switch = 15/17
  margin gain = +4.407
  q projection = +0.534
  correct_value alpha = +0.02966

selection_early:
  switch = 9/17
  margin gain = +1.553
  q projection = +0.803
  correct_value alpha = +0.02526

selection_both:
  switch = 9/17
  margin gain = +1.384
  q projection = +0.847
  correct_value alpha = +0.03001

selection_random_norm:
  switch = 2/17
  margin gain = +0.131

result_random_norm:
  switch = 2/17
  margin gain = +0.088
```

#### GLM4 bf16

有效 target cases：

```text
31 / 256
```

主要结果：

```text
selection_both_plus_result:
  switch = 29/31
  margin gain = +2.131
  q projection = +0.886
  correct_value alpha = +0.03123

selection_late_plus_result:
  switch = 29/31
  margin gain = +2.131
  q projection = +0.588
  correct_value alpha = +0.01520

result_only:
  switch = 29/31
  margin gain = +2.131
  q projection = 0.000
  correct_value alpha = 0.00000

selection_early:
  switch = 9/31
  margin gain = +0.476
  q projection = +0.882
  correct_value alpha = +0.02395

selection_both:
  switch = 7/31
  margin gain = +0.347
  q projection = +0.886
  correct_value alpha = +0.03123

selection_random_norm:
  switch = 1/31
  margin gain = -0.082

result_random_norm:
  switch = 3/31
  margin gain = -0.069
```

#### DS7B

有效 target cases：

```text
82 / 256
```

主要结果：

```text
selection_both_plus_result:
  switch = 75/82
  margin gain = +2.892
  q projection = +0.538
  correct_value alpha = +0.07563

result_only:
  switch = 75/82
  margin gain = +2.890
  q projection = 0.000
  correct_value alpha = 0.00000

selection_late_plus_result:
  switch = 75/82
  margin gain = +2.889
  q projection = +0.242
  correct_value alpha = +0.05826

selection_early:
  switch = 64/82
  margin gain = +1.979
  q projection = +0.545
  correct_value alpha = +0.07503

selection_both:
  switch = 63/82
  margin gain = +1.907
  q projection = +0.538
  correct_value alpha = +0.07563

selection_late:
  switch = 62/82
  margin gain = +1.904
  q projection = +0.242
  correct_value alpha = +0.05826

selection_random_norm:
  switch = 8/82
  margin gain = -0.031

result_random_norm:
  switch = 2/82
  margin gain = -0.092
```

### 最可靠客观事实

1. **result_only 已经几乎达到 selection + result 的行为效果**

跨模型都是：

```text
Qwen3:
  result_only = 15/17
  selection_both_plus_result = 15/17

GLM4:
  result_only = 29/31
  selection_both_plus_result = 29/31

DS7B:
  result_only = 75/82
  selection_both_plus_result = 75/82
```

说明在当前 candidate-score 指标上，下游 result/readout-like state 是更接近直接行为恢复的瓶颈。

2. **selection state 确实恢复 Q/attention 指标**

DS7B：

```text
selection_early:
  q projection +0.545
  correct_value alpha +0.07503
  switch 64/82

selection_both:
  q projection +0.538
  correct_value alpha +0.07563
  switch 63/82
```

GLM4：

```text
selection_both:
  q projection +0.886
  correct_value alpha +0.03123
```

Qwen3：

```text
selection_both:
  q projection +0.847
  correct_value alpha +0.03001
```

这说明 Phase620-622 的 selection-state 路径不是噪声。

3. **selection + result 没有明显超过 result_only**

三模型中组合 patch 的行为得分几乎等于 result_only：

```text
selection_both_plus_result ≈ result_only
```

这说明两者不是简单加和关系。更像：

```text
selection state 是上游找值状态；
result state 是下游已读出的结果状态；
一旦直接补上 result state，selection state 对 candidate score 的额外贡献很小。
```

4. **random same-norm controls 基本无效**

DS7B：

```text
selection_random_norm = 8/82, margin -0.031
result_random_norm = 2/82, margin -0.092
```

GLM4：

```text
selection_random_norm = 1/31
result_random_norm = 3/31
```

Qwen3：

```text
selection_random_norm = 2/17
result_random_norm = 2/17
```

说明效果不是同范数扰动造成的。

5. **selection state 本身也有行为效果，但不是最强行为瓶颈**

DS7B：

```text
selection_early = 64/82
selection_both = 63/82
result_only = 75/82
```

这说明 selection state 不只是监测指标，而是真有因果行为效应；但它在当前任务中不如 result state 直接。

### 理论进展

当前链条可以谨慎更新为：

```text
prompt condition
  -> residual selection state
  -> answer-position Q
  -> correct value token attention selection
  -> downstream result/readout-like state
  -> candidate score / generation preference
```

Phase623 使两类状态的角色更清楚：

```text
selection state:
  保持“去哪里读”的条件。
  主要表现在 Q projection 和 correct value token attention alpha。

result state:
  保持“读出以后形成什么候选偏置”的结果。
  不需要恢复 Q/alpha，也能直接移动 candidate score。
```

这对语言编码机制图谱很关键：同一个 residual stream 中不是只有一个语义向量，而是至少存在不同功能态：

```text
1. 条件化选择态
2. 读出结果态
3. 候选竞争态
4. 生成门态
```

### 硬伤和问题

1. **Qwen3 有效 target cases 仍偏少**

```text
Qwen3 target cases = 17 / 256
```

虽然趋势和另外两个模型一致，但 Qwen3 的样本量仍然只适合作为支持证据，不适合作为强结论核心。

2. **result state 仍未证明是最终 readout**

当前只证明：

```text
q_backproj_orthogonal result component 可以强烈移动 candidate score。
```

还没有证明它如何经过后续层、MLP、attention、unembedding 变成最终 token。

3. **selection + result 不加和的原因尚未解释**

可能原因包括：

```text
1. result state 已经包含 selection 的下游投影。
2. candidate score 指标只对 result state 敏感。
3. selection state 的主要作用在更早时刻，直接补 result 会绕过它。
4. 两者存在饱和或门控覆盖。
```

4. **当前仍是 candidate-score 任务，不是完整自然生成闭环**

需要回到自然语言生成下测试：

```text
result_only 能否改变实际生成 token？
selection_only 能否改变 attention path 但不一定改变生成？
```

5. **没有做逐层 result-state 洗出图谱**

Phase623 只选了一个 result layer，没有系统追踪 result state 从 L22 之后如何保留、变形或被覆盖。

### 下一步任务

Phase624 应进入：

```text
Result State Downstream Propagation Atlas
结果态下游传播图谱
```

核心目标：

```text
不要继续盲目扩大 patch 搜索。
围绕 result state 建立下游传播图谱：
  L22 result state 被补上以后，
  哪些后续层保留它？
  哪些模块增强它？
  哪些模块洗掉它？
  哪个位置真正接近 final logits/readout？
```

测试方案：

```text
1. 以 DS7B 为主，Qwen3/GLM4 做验证。
2. 固定 Phase623 的 result_only component。
3. 在 patch 后逐层读取：
   - candidate score delta
   - correct minus old-top-wrong margin
   - residual projection onto result direction
   - MLP out projection
   - attention out projection
   - unembedding logit contribution proxy
4. 比较：
   - result_only
   - selection_only
   - selection + result
   - random same-norm
5. 目标不是寻找更强 patch，而是画出 result state 从内部状态到输出偏置的传播路径。
```

阶段性大任务：

```text
从“点状机制验证”升级为“状态传播图谱”：
  先把 value gate 这条链路画完整，
  再把同样方法扩展到 category、relation、format、punctuation 等结构。
```

## Phase 624: Result State Downstream Propagation Atlas 结果态下游传播图谱 [2026-06-25 10:24]

### 本阶段目标

根据用户上传的 Phase623 分析，先判断其正确性，再继续完成任务。

附加分析基本正确：

```text
1. Phase623 的关键意义不是“又找到一个 patch”，而是把 residual state 拆成 selection state 和 result state 两类功能态。
2. result_only 在 candidate-score 指标上几乎达到 selection + result 的行为效果，这是当前最重要客观事实。
3. selection state 不是噪声，它恢复 Q projection 和 correct_value_token attention alpha。
4. result state 不能直接叫最终 readout，只能称为 result/readout-like state。
5. 下一步必须追踪 result state 的下游传播，而不是继续盲目扩大 patch 搜索。
```

需要补充的谨慎点：

```text
Phase623 证明 result state 可以强烈移动 candidate score，
但没有证明它如何穿过后续层、MLP、attention、final norm、lm_head。

因此 Phase624 的目标不是寻找更强补丁，
而是画 result state 从 L22/L29/L34 一直到后续层的传播轨迹。
```

### 生成脚本

```text
tests/glm5/phase624_result_state_downstream_propagation_atlas.py
tests/glm5/phase624_result_state_downstream_propagation_summary.py
```

### 执行命令

静态检查：

```bash
python -m py_compile \
  tests/glm5/phase624_result_state_downstream_propagation_atlas.py \
  tests/glm5/phase624_result_state_downstream_propagation_summary.py
```

烟测：

```bash
python tests/glm5/phase624_result_state_downstream_propagation_atlas.py qwen3 \
  --smoke \
  --include-nontarget \
  --output-dir results/glm5_phase624_result_state_downstream_propagation_atlas \
  --hard-exit-after-model
```

正式确认测试：

```bash
python tests/glm5/phase624_result_state_downstream_propagation_atlas.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase624_result_state_downstream_propagation_atlas \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/glm5/phase624_result_state_downstream_propagation_atlas.py glm4 \
  --confirm \
  --output-dir results/glm5_phase624_result_state_downstream_propagation_atlas \
  --hard-exit-after-model

python tests/glm5/phase624_result_state_downstream_propagation_atlas.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase624_result_state_downstream_propagation_atlas \
  --hard-exit-after-model

python tests/glm5/phase624_result_state_downstream_propagation_summary.py
```

### 输出文件

```text
results/glm5_phase624_result_state_downstream_propagation_atlas/phase624_qwen3_result_state_downstream_propagation_confirm.json
results/glm5_phase624_result_state_downstream_propagation_atlas/phase624_glm4_result_state_downstream_propagation_confirm.json
results/glm5_phase624_result_state_downstream_propagation_atlas/phase624_deepseek7b_result_state_downstream_propagation_confirm.json
results/glm5_phase624_result_state_downstream_propagation_atlas/phase624_cross_model_summary.md
```

### 测试原理

本阶段固定 Phase623 的 result_only component，然后在 patch 后读取后续层组件：

```text
layer_input
attn_out
mlp_out
layer_out
```

每个节点计算：

```text
repair_projection:
  patch 后节点变化量在 repair-base 节点变化量上的投影。

repair_cos:
  patch 后节点变化量与 repair-base 节点变化量的余弦相似度。

seed_projection:
  patch 后节点变化量在原始 result seed 方向上的投影。
```

核心判据：

```text
如果 layer_out/layer_input 的 seed_projection 持续接近 1：
  result seed 在 residual stream 中被保留。

如果 mlp_out 的 repair_projection 高，但 seed_projection 低：
  MLP 输出不是简单复制 seed，而是在把状态转换到 repair trajectory。

如果后续层 repair_projection 快速下降：
  result state 被洗出。
```

### 测试范围

```text
raw cases/model = 256
candidate values = first 4 value candidates
target cases = base wrong and repair prompt correct

Qwen3:
  patch layers = L26, L27, L29
  downstream layers = L29-L35
  target cases = 17

GLM4:
  patch layers = L31, L32, L34
  downstream layers = L34-L39
  target cases = 31

DS7B:
  patch layers = L20, L21, L22
  downstream layers = L22-L27
  target cases = 82
```

### 客观结果

#### Qwen3

行为结果：

```text
result_only:
  switch = 15/17
  margin gain = +4.407
  correct delta = +1.814
  wrong delta = -2.593

selection_both:
  switch = 9/17
  margin gain = +1.384

selection_both_plus_result:
  switch = 15/17
  margin gain = +4.407

result_random_norm:
  switch = 2/17
  margin gain = +0.088
```

result_only 最高传播节点：

```text
L31 mlp_out:
  repair_proj = 0.886
  repair_cos = 0.925
  seed_proj = 0.031

L29 layer_out:
  repair_proj = 0.865
  repair_cos = 0.930
  seed_proj = 1.000

L30 layer_input:
  repair_proj = 0.865
  repair_cos = 0.930
  seed_proj = 1.000

L30 mlp_out:
  repair_proj = 0.856
  repair_cos = 0.844
  seed_proj = 0.002

L31 layer_out:
  repair_proj = 0.837
  repair_cos = 0.905
  seed_proj = 1.059
```

#### GLM4 bf16

行为结果：

```text
result_only:
  switch = 29/31
  margin gain = +2.131
  correct delta = +0.974
  wrong delta = -1.157

selection_both:
  switch = 7/31
  margin gain = +0.347

selection_both_plus_result:
  switch = 29/31
  margin gain = +2.131

result_random_norm:
  switch = 3/31
  margin gain = -0.069
```

result_only 最高传播节点：

```text
L34 layer_out:
  repair_proj = 0.939
  repair_cos = 0.969
  seed_proj = 1.000

L35 layer_input:
  repair_proj = 0.939
  repair_cos = 0.969
  seed_proj = 1.000

L38 layer_out:
  repair_proj = 0.925
  repair_cos = 0.967
  seed_proj = 1.310

L39 layer_input:
  repair_proj = 0.925
  repair_cos = 0.967
  seed_proj = 1.310

L36 layer_out:
  repair_proj = 0.922
  repair_cos = 0.963
  seed_proj = 1.000
```

#### DS7B

行为结果：

```text
result_only:
  switch = 75/82
  margin gain = +2.890
  correct delta = +1.537
  wrong delta = -1.352

selection_both:
  switch = 63/82
  margin gain = +1.907

selection_both_plus_result:
  switch = 75/82
  margin gain = +2.892

result_random_norm:
  switch = 2/82
  margin gain = -0.092
```

result_only 最高传播节点：

```text
L22 layer_out:
  repair_proj = 0.848
  repair_cos = 0.921
  seed_proj = 1.000

L23 layer_input:
  repair_proj = 0.848
  repair_cos = 0.921
  seed_proj = 1.000

L23 mlp_out:
  repair_proj = 0.834
  repair_cos = 0.918
  seed_proj = -0.009

L23 layer_out:
  repair_proj = 0.817
  repair_cos = 0.920
  seed_proj = 1.012

L24 layer_input:
  repair_proj = 0.817
  repair_cos = 0.920
  seed_proj = 1.012

L25 mlp_out:
  repair_proj = 0.800
  repair_cos = 0.913
  seed_proj = 0.027

L27 mlp_out:
  repair_proj = 0.797
  repair_cos = 0.938
  seed_proj = 0.016
```

### 最可靠客观事实

1. **result state 在 residual stream 中持续保留**

DS7B：

```text
L22 layer_out seed_proj = 1.000
L23 layer_input seed_proj = 1.000
L23 layer_out seed_proj = 1.012
L24 layer_input seed_proj = 1.012
L24 layer_out seed_proj = 1.033
L25 layer_input seed_proj = 1.033
```

GLM4：

```text
L34 layer_out seed_proj = 1.000
L35 layer_input seed_proj = 1.000
L38 layer_out seed_proj = 1.310
L39 layer_input seed_proj = 1.310
```

Qwen3：

```text
L29 layer_out seed_proj = 1.000
L30 layer_input seed_proj = 1.000
L31 layer_out seed_proj = 1.059
L32 layer_input seed_proj = 1.059
```

说明 result state 不是只在单层短暂有效，而是以 residual carrier 的形式穿过多个后续层。

2. **MLP 输出高度对齐 repair trajectory，但不是简单复制 seed**

DS7B：

```text
L23 mlp_out:
  repair_proj = 0.834
  seed_proj = -0.009

L25 mlp_out:
  repair_proj = 0.800
  seed_proj = 0.027

L27 mlp_out:
  repair_proj = 0.797
  seed_proj = 0.016
```

Qwen3：

```text
L31 mlp_out:
  repair_proj = 0.886
  seed_proj = 0.031
```

这说明 MLP 不是简单传递原始 result seed，而是在当前状态条件下生成与 repair trajectory 高度一致的新输出。

3. **selection + result 仍然约等于 result_only**

三模型行为结果仍稳定：

```text
Qwen3:
  result_only = 15/17
  selection_both_plus_result = 15/17

GLM4:
  result_only = 29/31
  selection_both_plus_result = 29/31

DS7B:
  result_only = 75/82
  selection_both_plus_result = 75/82
```

说明 Phase623 的机制分叉没有被传播图谱推翻。

4. **random control 仍不能解释现象**

DS7B：

```text
result_random_norm = 2/82
margin gain = -0.092
```

GLM4：

```text
result_random_norm = 3/31
margin gain = -0.069
```

Qwen3：

```text
result_random_norm = 2/17
margin gain = +0.088
```

### 理论进展

当前 value gate 路径可更新为：

```text
prompt condition
  -> residual selection state
  -> answer-position Q
  -> correct value token attention selection
  -> result seed state
  -> residual carrier propagation
  -> state-conditioned MLP transformation
  -> candidate score / generation preference
```

Phase624 的关键新增是：

```text
result state 不是孤立点，而是可沿 residual stream 传播的 carrier。
MLP 输出不是复制 carrier，而是把 carrier 和当前上下文状态结合，生成 repair-like downstream output。
```

这比“某个方向有因果效果”更接近内部编码机制，因为它显示了：

```text
同一网络在自回归计算中可以靠 residual state 的不同态来保持任务条件；
后续层并不是重新找值，而是在已有 result carrier 上继续变换；
MLP 可能承担“状态条件化转换器”的角色。
```

### 硬伤和问题

1. **Qwen3 target cases 仍偏少**

```text
Qwen3 = 17 / 256
```

趋势一致，但不能把 Qwen3 当成强统计核心。

2. **还没有接到 final norm / lm_head**

Phase624 扫到后续 transformer layers，但没有直接测：

```text
final norm input/output
lm_head logits contribution
first token vs full answer token sequence
```

3. **MLP 的作用还只是轨迹对齐，不是因果拆分**

当前证明：

```text
MLP out 与 repair trajectory 高度对齐。
```

但还没有证明：

```text
哪些 MLP 子方向负责增强 correct candidate；
哪些子方向负责压制 old top wrong；
MLP 是否可以单独恢复 generation。
```

4. **result state 的 seed_projection 后期可能超过 1**

例如 GLM4：

```text
L38 layer_out seed_proj = 1.310
```

这可能是真增强，也可能是 residual norm/方向混合造成，不能直接解释成“语义增强 31%”。

5. **仍然是 value gate testbed**

还没有证明 category、relation、format、punctuation 等机制都使用同样的 result carrier 结构。

### 下一步任务

Phase625 应进入：

```text
Final Readout Bridge and MLP Causal Split
最终读出桥接与 MLP 因果拆分
```

核心目标：

```text
把 Phase624 的 result carrier 接到 final norm / lm_head；
同时拆分 MLP 输出到底是在增强 correct candidate，还是压制 old top wrong。
```

测试方案：

```text
1. 固定 DS7B 为主，Qwen3/GLM4 做验证。
2. 使用 Phase624 的 result_only patch。
3. 读取并干预：
   - final norm input
   - final norm output
   - lm_head first-token logits
   - full-answer logprob
4. 对 MLP out 做方向分解：
   - correct candidate logit direction
   - old top wrong logit direction
   - orthogonal residual direction
5. 分别 patch：
   - correct-up component
   - wrong-down component
   - both
   - orthogonal
   - random same-norm
6. 判定 result carrier 到最终输出的桥是不是：
   residual carrier -> MLP conversion -> final norm acceptance -> lm_head candidate margin
```

阶段性大任务：

```text
从“状态传播图谱”推进到“输出桥接图谱”：
  不只知道状态在哪里传播，
  还要知道它如何变成最终 token 竞争。
```

## Phase 625: Final Readout Bridge and MLP Causal Split 最终读出桥接与 MLP 因果拆分 [2026-06-25 10:54]

### 本阶段目标

根据用户上传的 Phase624 分析，先判断其正确性，再继续完成任务。

附加分析基本正确：

```text
1. Phase624 的关键价值是把 result state 从单点有效状态推进为 residual carrier。
2. residual stream 中的 result seed 可以跨层保留。
3. MLP out 与 repair trajectory 高度对齐，但不是简单复制 seed。
4. Phase625 必须把 result carrier 接到 final norm / lm_head。
5. 同时要测试 MLP out 的作用到底是 correct-up、wrong-down、两者结合，还是更复杂的上下文状态转换。
```

需要收紧的地方：

```text
不能因为 MLP out repair_projection 高，就直接断言 MLP 是最终输出模块。
也不能因为 result_only 能改变 candidate score，就直接断言它已经完整接到 lm_head。
```

本阶段目标：

```text
1. 继续复现 result_only 的强行为效果。
2. 测 result_only 是否把 final norm input/output 推向 repair trajectory。
3. 用 output embedding candidate directions 拆分 MLP out：
   - full_delta
   - correct_up
   - wrong_down
   - correct_plus_wrong
   - margin_span
   - orthogonal
   - random_same_norm
4. 判断单层 MLP 子方向能否解释 result_only 的行为效果。
```

### 生成脚本

```text
tests/glm5/phase625_final_readout_bridge_mlp_causal_split.py
tests/glm5/phase625_final_readout_bridge_mlp_causal_split_summary.py
```

### 执行命令

静态检查：

```bash
python -m py_compile \
  tests/glm5/phase625_final_readout_bridge_mlp_causal_split.py \
  tests/glm5/phase625_final_readout_bridge_mlp_causal_split_summary.py
```

烟测：

```bash
python tests/glm5/phase625_final_readout_bridge_mlp_causal_split.py qwen3 \
  --smoke \
  --include-nontarget \
  --output-dir results/glm5_phase625_final_readout_bridge_mlp_causal_split \
  --hard-exit-after-model
```

正式确认测试：

```bash
python tests/glm5/phase625_final_readout_bridge_mlp_causal_split.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase625_final_readout_bridge_mlp_causal_split \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/glm5/phase625_final_readout_bridge_mlp_causal_split.py glm4 \
  --confirm \
  --output-dir results/glm5_phase625_final_readout_bridge_mlp_causal_split \
  --hard-exit-after-model

python tests/glm5/phase625_final_readout_bridge_mlp_causal_split.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase625_final_readout_bridge_mlp_causal_split \
  --hard-exit-after-model

python tests/glm5/phase625_final_readout_bridge_mlp_causal_split_summary.py
```

### 输出文件

```text
results/glm5_phase625_final_readout_bridge_mlp_causal_split/phase625_qwen3_final_readout_bridge_mlp_causal_split_confirm.json
results/glm5_phase625_final_readout_bridge_mlp_causal_split/phase625_glm4_final_readout_bridge_mlp_causal_split_confirm.json
results/glm5_phase625_final_readout_bridge_mlp_causal_split/phase625_deepseek7b_final_readout_bridge_mlp_causal_split_confirm.json
results/glm5_phase625_final_readout_bridge_mlp_causal_split/phase625_cross_model_summary.md
```

### 测试范围

```text
raw cases/model = 256
candidate values = v05, v91, v22, v48
target cases = base wrong and repair prompt correct

Qwen3:
  patch layers = L26, L27, L29
  MLP split layer = L31
  target cases = 17

GLM4:
  patch layers = L31, L32, L34
  MLP split layer = L39
  target cases = 31

DS7B:
  patch layers = L20, L21, L22
  MLP split layer = L23
  target cases = 82
```

### 测试原理

#### Final bridge

对 correct answer 的 final norm input/output 计算：

```text
patched-base 是否沿 repair-base 方向前进。
```

记录：

```text
input_repair_projection
output_repair_projection
output_repair_cos
output_projection_margin
```

其中 `output_projection_margin` 是 final norm output effect 在 candidate output embedding directions 上的 margin proxy。

#### MLP causal split

对 MLP out delta 做候选方向分解：

```text
correct_up:
  沿 correct candidate output embedding specific direction。

wrong_down:
  沿 old top wrong candidate 的负向 specific direction。

correct_plus_wrong:
  correct_up + wrong_down。

margin_span:
  correct_up 与 wrong_down 张成子空间中的投影。

orthogonal:
  full_delta 减去 margin_span。

random_same_norm:
  同范数随机对照。
```

### 客观结果

#### Qwen3

有效 target cases：

```text
17 / 256
```

行为结果：

```text
result_only:
  switch = 15/17
  margin gain = +4.407
  correct delta = +1.814
  wrong delta = -2.593

mlp_full_delta:
  switch = 4/17
  margin gain = +0.478

mlp_correct_up:
  switch = 2/17
  margin gain = +0.250

mlp_wrong_down:
  switch = 2/17
  margin gain = +0.169

mlp_correct_plus_wrong:
  switch = 2/17
  margin gain = +0.390

mlp_margin_span:
  switch = 2/17
  margin gain = +0.280

mlp_orthogonal:
  switch = 3/17
  margin gain = +0.162

mlp_random_same_norm:
  switch = 1/17
  margin gain = -0.006
```

final bridge：

```text
result_only:
  input repair projection = 0.363
  output repair projection = 0.355
  output repair cos = 0.667
  output margin proxy = +0.739
  correct proxy = +0.453
  wrong proxy = -0.285
```

#### GLM4 bf16

有效 target cases：

```text
31 / 256
```

行为结果：

```text
result_only:
  switch = 29/31
  margin gain = +2.131
  correct delta = +0.974
  wrong delta = -1.157

mlp_full_delta:
  switch = 4/31
  margin gain = -0.212

mlp_correct_up:
  switch = 3/31
  margin gain = -0.083

mlp_wrong_down:
  switch = 5/31
  margin gain = -0.117

mlp_correct_plus_wrong:
  switch = 6/31
  margin gain = -0.200

mlp_margin_span:
  switch = 4/31
  margin gain = -0.145

mlp_orthogonal:
  switch = 0/31
  margin gain = -0.081

mlp_random_same_norm:
  switch = 2/31
  margin gain = -0.010
```

final bridge：

```text
result_only:
  input repair projection = 0.320
  output repair projection = 0.311
  output repair cos = 0.565
  output margin proxy = +0.533
  correct proxy = +0.334
  wrong proxy = -0.199
```

#### DS7B

有效 target cases：

```text
82 / 256
```

行为结果：

```text
result_only:
  switch = 75/82
  margin gain = +2.890
  correct delta = +1.537
  wrong delta = -1.352

mlp_full_delta:
  switch = 14/82
  margin gain = +0.326

mlp_correct_up:
  switch = 4/82
  margin gain = +0.094

mlp_wrong_down:
  switch = 1/82
  margin gain = +0.037

mlp_correct_plus_wrong:
  switch = 4/82
  margin gain = +0.132

mlp_margin_span:
  switch = 4/82
  margin gain = +0.101

mlp_orthogonal:
  switch = 10/82
  margin gain = +0.229

mlp_random_same_norm:
  switch = 3/82
  margin gain = -0.016
```

final bridge：

```text
result_only:
  input repair projection = 0.340
  output repair projection = 0.372
  output repair cos = 0.667
  output margin proxy = +0.579
  correct proxy = +0.433
  wrong proxy = -0.145
```

### 最可靠客观事实

1. **result_only 确实桥接到 final norm / output embedding margin proxy**

三模型都有正向 final bridge：

```text
Qwen3:
  output repair projection = 0.355
  output margin proxy = +0.739

GLM4:
  output repair projection = 0.311
  output margin proxy = +0.533

DS7B:
  output repair projection = 0.372
  output margin proxy = +0.579
```

说明 Phase624 的 result carrier 不是停在中间层，它能把 final norm output 推向 repair-like candidate margin。

2. **final bridge 是部分桥接，不是完整闭合**

output repair projection 只有约：

```text
0.31 - 0.37
```

这说明 result_only 强行为效果并不等于 final norm state 完全变成 repair state。它只把最终输出状态推向 repair trajectory 的一部分。

3. **单层 MLP out 不能解释 result_only 行为效果**

DS7B：

```text
result_only = 75/82, margin +2.890
mlp_full_delta = 14/82, margin +0.326
mlp_correct_plus_wrong = 4/82, margin +0.132
mlp_orthogonal = 10/82, margin +0.229
```

GLM4：

```text
result_only = 29/31, margin +2.131
mlp_full_delta = 4/31, margin -0.212
```

Qwen3：

```text
result_only = 15/17, margin +4.407
mlp_full_delta = 4/17, margin +0.478
```

因此 Phase624 中 “MLP out repair_projection 高” 不等于 “单层 MLP out patch 可以恢复输出行为”。

4. **candidate embedding 简单分解不足**

correct_up、wrong_down、margin_span 都远弱于 result_only：

```text
DS7B:
  correct_up = 4/82
  wrong_down = 1/82
  margin_span = 4/82

Qwen3:
  correct_up = 2/17
  wrong_down = 2/17
  margin_span = 2/17

GLM4:
  所有 MLP split 基本无效或负向。
```

说明 MLP 输出的作用不是简单线性 candidate logit direction。

5. **orthogonal component 有时强于 candidate span**

DS7B：

```text
mlp_orthogonal = 10/82, margin +0.229
mlp_margin_span = 4/82, margin +0.101
```

Qwen3：

```text
mlp_orthogonal = 3/17
mlp_margin_span = 2/17
```

这说明当前 output embedding span 不是完整候选竞争空间，仍有隐藏的状态/门控/归一化接受因素。

### 理论进展

当前链条应更新为：

```text
prompt condition
  -> selection state
  -> Q selects correct value token
  -> result carrier state
  -> residual propagation
  -> distributed downstream transformation
  -> partial final norm bridge
  -> candidate margin / generation preference
```

Phase625 的关键修正是：

```text
MLP 是重要转换节点，但不是单层、单方向、候选 embedding span 可直接解释的简单读出器。
result carrier 到 final norm 的桥接是真实存在的，但只解释了最终 repair state 的一部分。
```

这意味着 value gate 的后端不是：

```text
某个 MLP out 直接写 correct token logit direction
```

而更可能是：

```text
result carrier 在多层 residual stream 中传播，
多层 attention/MLP/final norm 共同把它变成 candidate margin。
```

### 硬伤和问题

1. **MLP split 只测了一个默认层**

```text
Qwen3 L31
GLM4 L39
DS7B L23
```

但 Phase624 显示多个 MLP out 都有高 repair_projection。单层失败不能排除多层 MLP 累积机制。

2. **candidate embedding direction 太粗糙**

使用 output embeddings 分解 correct-up/wrong-down，可能不能代表真实 lm_head + final norm 后的非线性接受边界。

3. **final bridge 只看平均 answer positions**

value tokens 是多 token 字符串，例如 v05。first token 可能共享，后续 token 才是区分关键。后续需要按 token position 分开。

4. **没有做 final norm 直接 patch**

当前是观察 result_only 对 final norm 的影响，还没有直接 patch final_norm input/output 验证 acceptance gate。

5. **仍未完成自然生成闭环**

结果仍主要是 candidate logprob，不是开放式 generation。

### 下一步任务

Phase626 应进入：

```text
Multi-Layer Final Bridge and Token-Position Readout Audit
多层最终桥接与词元位置读出审计
```

核心目标：

```text
不要再假设单层 MLP 能解释 result carrier。
需要做两件事：
  1. 多层 MLP/attn/final norm 累积桥接。
  2. 按 answer token position 拆分 final readout。
```

测试方案：

```text
1. 固定 DS7B 为主，Qwen3/GLM4 验证。
2. 对 result carrier patch 后，分 token position 读取：
   - token0: 通常是 shared "v"
   - token1/token2: 真正区分 05/91/22/48 的位置
3. 分别计算 final_norm output projection：
   - correct token logit delta
   - old top wrong token logit delta
   - margin delta
4. 做多层累计 patch：
   - MLP out cumulative
   - attention out cumulative
   - layer_out carrier cumulative
   - final_norm input/output patch
5. 判断最终瓶颈是在：
   - 多层 MLP 累积
   - residual carrier 到 final norm 的接受
   - shared-prefix token position 掩盖
   - lm_head candidate competition
```

阶段性大任务：

```text
从“结果态传播”推进到“逐词元输出桥接”：
  语言输出不是一个答案整体，
  而是每个 token position 的竞争过程。
```

## Phase 626: Multi-Layer Final Bridge and Token-Position Readout Audit 多层最终桥接与词元位置读出审计 [2026-06-25 12:01]

### 本阶段目标

根据用户上传的 Phase625 分析，先判断其正确性，再继续完成任务。

附加分析基本正确：

```text
1. Phase625 是关键修正阶段，不是简单闭合阶段。
2. result_only 能部分桥接 final norm / output embedding margin proxy。
3. 单层 MLP out 和简单 candidate embedding direction 不能解释 result_only。
4. 后端更可能是多层 residual/attention/MLP/final norm 的分布式桥接。
5. 下一步必须按 token position 拆分，因为 value strings 存在共享前缀。
```

需要继续收紧的地方：

```text
final_output_all patch 是直接把 final norm 状态替换为 repair 状态，
它是上界测试，不等于自然机制已经完整实现。

cumulative layer_out patch 也可能包含多个模块的综合结果，
不能直接说某一个模块单独负责。
```

本阶段目标：

```text
1. 审计 v05/v91/v22/v48 的真实 tokenizer 分解。
2. 在 result_only 下按 answer token position 统计 logprob delta。
3. 直接 patch final norm input/output：
   - all answer tokens
   - token0 shared prefix
   - last token
   - random all control
4. 做多层累计 patch：
   - cumulative layer_out
   - cumulative attn_out
   - cumulative mlp_out
   - cumulative layer_out random
5. 判断最终瓶颈是在共享前缀、区分 token、final norm acceptance，还是多层累计桥接。
```

### 生成脚本

```text
tests/glm5/phase626_multilayer_final_bridge_token_position_audit.py
tests/glm5/phase626_multilayer_final_bridge_token_position_summary.py
```

### 执行命令

静态检查：

```bash
python -m py_compile \
  tests/glm5/phase626_multilayer_final_bridge_token_position_audit.py \
  tests/glm5/phase626_multilayer_final_bridge_token_position_summary.py
```

烟测：

```bash
python tests/glm5/phase626_multilayer_final_bridge_token_position_audit.py qwen3 \
  --smoke \
  --include-nontarget \
  --output-dir results/glm5_phase626_multilayer_final_bridge_token_position_audit \
  --hard-exit-after-model
```

正式确认测试：

```bash
python tests/glm5/phase626_multilayer_final_bridge_token_position_audit.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase626_multilayer_final_bridge_token_position_audit \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/glm5/phase626_multilayer_final_bridge_token_position_audit.py glm4 \
  --confirm \
  --output-dir results/glm5_phase626_multilayer_final_bridge_token_position_audit \
  --hard-exit-after-model

python tests/glm5/phase626_multilayer_final_bridge_token_position_audit.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase626_multilayer_final_bridge_token_position_audit \
  --hard-exit-after-model

python tests/glm5/phase626_multilayer_final_bridge_token_position_summary.py
```

### 输出文件

```text
results/glm5_phase626_multilayer_final_bridge_token_position_audit/phase626_qwen3_multilayer_final_bridge_token_position_audit_confirm.json
results/glm5_phase626_multilayer_final_bridge_token_position_audit/phase626_glm4_multilayer_final_bridge_token_position_audit_confirm.json
results/glm5_phase626_multilayer_final_bridge_token_position_audit/phase626_deepseek7b_multilayer_final_bridge_token_position_audit_confirm.json
results/glm5_phase626_multilayer_final_bridge_token_position_audit/phase626_cross_model_summary.md
```

### Tokenization 审计

Qwen3：

```text
v05 = [' v', '0', '5']
v91 = [' v', '9', '1']
v22 = [' v', '2', '2']
v48 = [' v', '4', '8']
```

DS7B：

```text
v05 = [' v', '0', '5']
v91 = [' v', '9', '1']
v22 = [' v', '2', '2']
v48 = [' v', '4', '8']
```

GLM4：

```text
v05 = [' v', '05']
v91 = [' v', '91']
v22 = [' v', '22']
v48 = [' v', '48']
```

这说明：

```text
token0 是共享前缀 ' v'；
真正区分类别值的位置是：
  Qwen3/DS7B: token1 为主要区分位，token2 贡献很小。
  GLM4: token1 是完整数字对，也是主要区分位。
```

### 测试范围

```text
raw cases/model = 256
candidate values = v05, v91, v22, v48
target cases = base wrong and repair prompt correct

Qwen3:
  target cases = 17
  result patch layer = L29
  downstream layers = L29-L35

GLM4:
  target cases = 31
  result patch layer = L34
  downstream layers = L34-L39

DS7B:
  target cases = 82
  result patch layer = L22
  downstream layers = L22-L27
```

### 客观结果

#### Qwen3

```text
result_only:
  switch = 15/17
  margin gain = +4.407
  correct delta = +1.814
  wrong delta = -2.593

final_input_all:
  switch = 17/17
  margin gain = +5.377

final_output_all:
  switch = 17/17
  margin gain = +5.377

final_output_token0:
  switch = 0/17
  margin gain = 0.000

final_output_last:
  switch = 2/17
  margin gain = +0.075

cumulative_layer_out:
  switch = 17/17
  margin gain = +5.305

cumulative_attn_out:
  switch = 12/17
  margin gain = +2.326

cumulative_mlp_out:
  switch = 12/17
  margin gain = +2.545

cumulative_layer_out_random:
  switch = 2/17
  margin gain = +0.020
```

result_only token-position delta：

```text
tok0:
  margin delta = 0.000

tok1:
  correct delta = +1.812
  wrong delta = -2.593
  margin delta = +4.404

tok2:
  correct delta = +0.002
  wrong delta = -0.001
  margin delta = +0.003
```

#### GLM4 bf16

```text
result_only:
  switch = 29/31
  margin gain = +2.131
  correct delta = +0.974
  wrong delta = -1.157

final_input_all:
  switch = 31/31
  margin gain = +2.300

final_output_all:
  switch = 31/31
  margin gain = +2.300

final_output_token0:
  switch = 0/31
  margin gain = 0.000

final_output_last:
  switch = 31/31
  margin gain = +2.300

cumulative_layer_out:
  switch = 31/31
  margin gain = +2.300

cumulative_attn_out:
  switch = 3/31
  margin gain = -0.121

cumulative_mlp_out:
  switch = 11/31
  margin gain = +0.518

cumulative_layer_out_random:
  switch = 4/31
  margin gain = +0.090
```

result_only token-position delta：

```text
tok0:
  margin delta = 0.000

tok1:
  correct delta = +0.974
  wrong delta = -1.157
  margin delta = +2.131
```

#### DS7B

```text
result_only:
  switch = 75/82
  margin gain = +2.890
  correct delta = +1.537
  wrong delta = -1.352

final_input_all:
  switch = 82/82
  margin gain = +3.602

final_output_all:
  switch = 82/82
  margin gain = +3.602

final_output_token0:
  switch = 0/82
  margin gain = 0.000

final_output_last:
  switch = 2/82
  margin gain = +0.053

cumulative_layer_out:
  switch = 81/82
  margin gain = +3.561

cumulative_attn_out:
  switch = 79/82
  margin gain = +3.114

cumulative_mlp_out:
  switch = 44/82
  margin gain = +1.414

cumulative_layer_out_random:
  switch = 9/82
  margin gain = +0.077
```

result_only token-position delta：

```text
tok0:
  margin delta = 0.000

tok1:
  correct delta = +1.525
  wrong delta = -1.354
  margin delta = +2.879

tok2:
  correct delta = +0.012
  wrong delta = +0.002
  margin delta = +0.011
```

### 最可靠客观事实

1. **共享前缀 token0 完全不是竞争位置**

三模型：

```text
final_output_token0:
  Qwen3 = 0/17, margin 0.000
  GLM4 = 0/31, margin 0.000
  DS7B = 0/82, margin 0.000
```

result_only 下 tok0：

```text
margin delta = 0.000
```

说明前面把 answer positions 平均在一起，会稀释甚至误导读出分析。

2. **真正竞争集中在第一个区分 token**

Qwen3：

```text
tok1 margin delta = +4.404
tok2 margin delta = +0.003
```

DS7B：

```text
tok1 margin delta = +2.879
tok2 margin delta = +0.011
```

GLM4：

```text
tok1 margin delta = +2.131
```

这说明 value gate 的输出竞争不是均匀分布在答案 token 序列上，而是集中在首个区分 token。

3. **final norm all patch 是强上界**

三模型：

```text
Qwen3:
  final_output_all = 17/17, margin +5.377

GLM4:
  final_output_all = 31/31, margin +2.300

DS7B:
  final_output_all = 82/82, margin +3.602
```

它证明 final norm 状态足够承载正确输出竞争，但这是直接 repair patch，不等于自然路径闭合。

4. **cumulative layer_out 几乎达到 final norm all 的上界**

Qwen3：

```text
cumulative_layer_out = 17/17, margin +5.305
final_output_all = 17/17, margin +5.377
```

GLM4：

```text
cumulative_layer_out = 31/31, margin +2.300
final_output_all = 31/31, margin +2.300
```

DS7B：

```text
cumulative_layer_out = 81/82, margin +3.561
final_output_all = 82/82, margin +3.602
```

这是本阶段最重要的新结果：多层 residual carrier 累计 patch 几乎可以接近 final norm 上界。

5. **attention/MLP 的作用跨模型不同**

DS7B：

```text
cumulative_attn_out = 79/82, margin +3.114
cumulative_mlp_out = 44/82, margin +1.414
```

Qwen3：

```text
cumulative_attn_out = 12/17, margin +2.326
cumulative_mlp_out = 12/17, margin +2.545
```

GLM4：

```text
cumulative_attn_out = 3/31, margin -0.121
cumulative_mlp_out = 11/31, margin +0.518
```

说明不能用单一“attention 或 MLP 是主因”的理论跨模型套用。更稳妥的是：

```text
layer_out carrier 是跨模型稳定主线；
attention/MLP 是模型特异的转换和补充路径。
```

6. **random controls 仍弱**

```text
Qwen3 cumulative_layer_out_random = 2/17, margin +0.020
GLM4 cumulative_layer_out_random = 4/31, margin +0.090
DS7B cumulative_layer_out_random = 9/82, margin +0.077
```

说明多层累计效果不是同范数随机扰动造成的。

### 理论进展

当前 value gate 路径应更新为：

```text
prompt condition
  -> selection state
  -> Q selects correct value token
  -> result carrier state
  -> multi-layer residual carrier accumulation
  -> final norm acceptance at discriminative token position
  -> candidate margin
```

Phase626 的关键修正是：

```text
语言输出竞争不是“整答案状态”的平均竞争，
而是由共享前缀之后的第一个区分 token 触发。
```

对破解语言编码机制的意义：

```text
1. residual stream 是跨层状态总线。
2. layer_out carrier 是当前 value gate 后端最稳定主线。
3. final norm 可以接受并表达这个 carrier。
4. 真正读出点必须按 token position 定位。
5. MLP/attention 不能孤立解释，必须放到多层状态图谱中。
```

### 硬伤和问题

1. **final_norm all patch 是上界，不是自然路径**

它说明 final norm 有能力表达正确 margin，但不能证明自然模型已经完全走这条路径。

2. **cumulative layer_out patch 可能过强**

同时 patch 多层 layer_out 可能绕过自然动态，属于机制上界或路径容量测试，不等于真实逐层自然生成。

3. **Qwen3 target cases 仍少**

```text
Qwen3 = 17 / 256
```

趋势一致，但不能作为强样本核心。

4. **没有测试开放式 generation**

仍然是 candidate logprob 测试。

5. **只测 value gate**

还未扩展到 category、relation、format、punctuation 的输出读出。

### 下一步任务

Phase627 应进入：

```text
Natural Generation Token-Position Closure
自然生成逐词元闭环
```

核心目标：

```text
把 Phase626 的逐 token 位置结论从 candidate logprob 推到实际 generation。
```

测试方案：

```text
1. 固定 DS7B 为主，Qwen3/GLM4 验证。
2. 使用相同 value gate cases。
3. 对比自然 greedy generation：
   - base prompt
   - repair prompt
   - result_only patch
   - cumulative_layer_out patch
   - final_output_all patch
   - random controls
4. 记录逐 token 生成：
   - token0 是否总是共享 ' v'
   - token1 是否被正确切换
   - token2 是否跟随 token1 或仍需独立修复
5. 不只看最终字符串是否正确，还要看每个 token position 的生成分布变化。
```

阶段性大任务：

```text
从 candidate logprob 图谱推进到真实自回归生成图谱。
如果 Phase627 成功，value gate 将第一次形成：
  selection -> result carrier -> final norm -> token-position generation
的完整闭环。
```

## Phase 627: Natural Generation Token-Position Closure Audit 自然生成词元位置闭环审计 [2026-06-25 12:23]

### 本阶段目标

根据 Phase626 的结论，继续检查一个更严格的问题：

```text
candidate logprob 中已经能恢复正确值词元，
但真实 autoregressive greedy generation 中是否也能生成正确答案。
```

Phase626 已经证明：

```text
1. value candidate 的真实竞争不在共享前缀 token0，而在第一个区分 token。
2. result_only 与 cumulative_layer_out 可以强烈修复 candidate margin。
3. final_output_all 在 teacher-forced candidate logprob 中接近上界。
```

Phase627 进一步测试：

```text
同一套 result / cumulative / final patch，
能否在自然 greedy generation 中完成 token-position 级别闭环。
```

### 脚本

```text
tests/glm5/phase627_natural_generation_token_position_closure.py
tests/glm5/phase627_natural_generation_token_position_summary.py
```

脚本原则：

```text
1. 不使用 transformers generate，手写 greedy loop，保证每一步可 hook。
2. 每个生成 step 都重新 forward，并在对应位置施加 patch。
3. 使用 teacher-forced correct answer cache 作为 donor。
4. 同时统计 exact string、wrong exact、prefix length、token-position hit。
5. 添加 random same-shape control。
```

### 执行命令

```bash
python -m py_compile \
  tests/glm5/phase627_natural_generation_token_position_closure.py \
  tests/glm5/phase627_natural_generation_token_position_summary.py

python tests/glm5/phase627_natural_generation_token_position_closure.py qwen3 \
  --smoke --include-nontarget \
  --output-dir results/glm5_phase627_natural_generation_token_position_closure \
  --hard-exit-after-model

python tests/glm5/phase627_natural_generation_token_position_closure.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase627_natural_generation_token_position_closure \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/glm5/phase627_natural_generation_token_position_closure.py glm4 \
  --confirm \
  --output-dir results/glm5_phase627_natural_generation_token_position_closure \
  --hard-exit-after-model

python tests/glm5/phase627_natural_generation_token_position_closure.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase627_natural_generation_token_position_closure \
  --hard-exit-after-model

python tests/glm5/phase627_natural_generation_token_position_summary.py
```

### 输出文件

```text
results/glm5_phase627_natural_generation_token_position_closure/phase627_qwen3_natural_generation_token_position_closure_confirm.json
results/glm5_phase627_natural_generation_token_position_closure/phase627_glm4_natural_generation_token_position_closure_confirm.json
results/glm5_phase627_natural_generation_token_position_closure/phase627_deepseek7b_natural_generation_token_position_closure_confirm.json
results/glm5_phase627_natural_generation_token_position_closure/phase627_cross_model_summary.md
```

### 测试范围

```text
models = qwen3, glm4, deepseek7b
confirm rows after target filtering:
  qwen3 = 17
  glm4 = 31
  deepseek7b = 82

modes:
  base
  repair_prompt
  result_only
  result_random
  cumulative_layer_out
  cumulative_layer_out_random
  final_output_all
  final_output_random_all
```

### 客观结果

#### Qwen3

```text
base:
  exact = 1/17
  wrong_exact = 9/17
  prefix_mean = 0.706
  token hit = tok0 0.588, tok1 0.059, tok2 0.059

repair_prompt:
  exact = 11/17
  wrong_exact = 3/17
  prefix_mean = 2.118
  token hit = tok0 0.824, tok1 0.824, tok2 0.824

result_only:
  exact = 8/17
  wrong_exact = 2/17
  prefix_mean = 1.529
  token hit = tok0 0.588, tok1 0.882, tok2 0.647

cumulative_layer_out:
  exact = 10/17
  wrong_exact = 0/17
  prefix_mean = 1.765
  token hit = tok0 0.588, tok1 1.000, tok2 0.824

result_random:
  exact = 0/17
  wrong_exact = 10/17
  token hit = tok0 0.588, tok1 0.059, tok2 0.059

cumulative_layer_out_random:
  exact = 2/17
  wrong_exact = 8/17
  token hit = tok0 0.588, tok1 0.235, tok2 0.176

final_output_all:
  exact = 0/17
  wrong_exact = 0/17
  token hit = tok0 0.588, tok1 0.000, tok2 0.235
```

Qwen3 的结果说明：

```text
result_only 与 cumulative_layer_out 在自然生成中真实推动了第一个区分词元。
但 exact closure 仍然没有完全闭合。
final_output_all 在生成中失败，常出现重复共享前缀或错位输出。
```

#### GLM4 bf16

```text
base:
  exact = 2/31
  wrong_exact = 9/31
  prefix_mean = 0.419
  token hit = tok0 0.355, tok1 0.065

repair_prompt:
  exact = 28/31
  wrong_exact = 1/31
  prefix_mean = 1.839
  token hit = tok0 0.935, tok1 0.903

result_only:
  exact = 10/31
  wrong_exact = 0/31
  prefix_mean = 0.677
  token hit = tok0 0.355, tok1 0.935

cumulative_layer_out:
  exact = 11/31
  wrong_exact = 0/31
  prefix_mean = 0.710
  token hit = tok0 0.355, tok1 1.000

result_random:
  exact = 0/31
  wrong_exact = 11/31
  token hit = tok0 0.355, tok1 0.065

cumulative_layer_out_random:
  exact = 2/31
  wrong_exact = 9/31
  token hit = tok0 0.355, tok1 0.226

final_output_all:
  exact = 0/31
  wrong_exact = 0/31
  token hit = tok0 0.355, tok1 0.000
```

GLM4 的结果说明：

```text
result_only 与 cumulative_layer_out 几乎完全修复第一个区分词元。
但 token0 共享前缀/格式位置仍然没有被修复，所以 exact 只到 10/31 和 11/31。
```

#### DS7B

```text
base:
  exact = 0/82
  wrong_exact = 0/82
  prefix_mean = 0.000
  token hit = tok0 0.000, tok1 0.000, tok2 0.000

repair_prompt:
  exact = 20/82
  wrong_exact = 0/82
  prefix_mean = 0.732
  token hit = tok0 0.244, tok1 0.256, tok2 0.256

result_only:
  exact = 0/82
  wrong_exact = 0/82
  prefix_mean = 0.000
  token hit = tok0 0.000, tok1 0.902, tok2 0.049

cumulative_layer_out:
  exact = 0/82
  wrong_exact = 0/82
  prefix_mean = 0.000
  token hit = tok0 0.000, tok1 0.988, tok2 0.024

result_random:
  exact = 0/82
  wrong_exact = 0/82
  token hit = tok0 0.000, tok1 0.110, tok2 0.000

cumulative_layer_out_random:
  exact = 0/82
  wrong_exact = 0/82
  token hit = tok0 0.000, tok1 0.098, tok2 0.000

final_output_all:
  exact = 0/82
  wrong_exact = 0/82
  token hit = tok0 0.000, tok1 0.000, tok2 0.293
```

DS7B 的结果说明：

```text
result_only 与 cumulative_layer_out 已经能强烈修复第一个区分词元：
  tok1: 0.000 -> 0.902 -> 0.988

但 token0 格式/前缀完全失败：
  tok0: 0.000

所以 exact generation 仍然是 0/82。
```

### 当前最可靠客观事实

1. **Phase626 的 candidate logprob 闭环不是假象。**

在自然 greedy generation 中，result_only 与 cumulative_layer_out 仍然显著推动正确的第一个区分词元：

```text
Qwen3 tok1:
  base 0.059
  result_only 0.882
  cumulative_layer_out 1.000

GLM4 tok1:
  base 0.065
  result_only 0.935
  cumulative_layer_out 1.000

DS7B tok1:
  base 0.000
  result_only 0.902
  cumulative_layer_out 0.988
```

2. **完整字符串生成没有闭合。**

```text
Qwen3:
  base exact 1/17
  result_only exact 8/17
  cumulative exact 10/17

GLM4:
  base exact 2/31
  result_only exact 10/31
  cumulative exact 11/31

DS7B:
  base exact 0/82
  result_only exact 0/82
  cumulative exact 0/82
```

3. **当前机制修复的是 semantic value discriminative token，不是完整 format/prefix generation。**

尤其 DS7B：

```text
result_only / cumulative_layer_out 可以让 tok1 接近正确，
但 tok0 完全不正确。
```

这说明：

```text
value semantic path 与 format/prefix path 是可分离的。
```

4. **final_output_all 在 candidate logprob 中是上界，但在自然生成中不是上界。**

Phase626 中：

```text
final_output_all 接近 full repair upper bound。
```

Phase627 中：

```text
final_output_all exact = 0
tok1 hit 反而很低。
```

常见现象是重复共享前缀、错位输出或自回归反馈污染。

因此：

```text
final norm patch 是 position-conditioned 和 prefix-conditioned 的，
不能直接当成自然生成闭环证明。
```

5. **random control 明显弱于真实 result/cumulative patch。**

```text
result_random 与 cumulative_layer_out_random 无法稳定修复 tok1。
```

这说明主效应不是随机范数注入。

### 对附件 Phase626 分析的判断

附件中认为 Phase626 是关键读出桥接阶段，这个判断正确。

正确部分：

```text
1. Phase626 确实把 result carrier 推进到 final norm / output margin。
2. 第一个区分词元是 value candidate 的真实竞争位置。
3. 需要继续测试自然生成，而不是停留在 candidate logprob。
4. result path 与 generation path 之间可能存在缺口。
```

需要修正的部分：

```text
Phase626 不能被解释成完整 generation closure。
Phase627 已经证明：
  result/cumulative 可以修复语义区分词元，
  但不能自动修复共享前缀、格式 token、自回归反馈。
```

### 理论进展

当前 value gate 的结构应从：

```text
selection -> result carrier -> final readout -> generation
```

修正为：

```text
selection state
  -> semantic result carrier
  -> discriminative value token readout

format/prefix state
  -> shared-prefix token generation
  -> autoregressive alignment

两者共同决定完整自然生成。
```

更具体地说：

```text
result carrier 主要控制“该选哪个值”；
format/prefix carrier 控制“答案以什么形式进入生成轨道”；
final output state 只有在 teacher-forced prefix 对齐时才像上界，
在自由生成中可能被自回归反馈打散。
```

### 硬伤和边界

1. **donor cache 仍然来自 teacher-forced correct answer。**

这证明的是 causal repair，而不是模型自发完成正确 reasoning。

2. **target rows 是筛选后的 value-gate 子集。**

本阶段是机制闭环审计，不是全任务准确率评估。

3. **exact generation 对 prompt 格式非常敏感。**

尤其 DS7B，semantic token 被修复但 prefix token 失败，导致 exact 仍为 0。

4. **final_output_all 的失败说明 candidate logprob 与 natural generation 不能混用。**

以后必须把：

```text
teacher-forced candidate score
natural greedy generation
sampled generation
```

分开记录。

### 下一步 Phase628

Phase628 应做：

```text
Prefix/Format Gate and Semantic Value Integration
```

核心问题：

```text
如果先修复或强制正确 token0 共享前缀，
result_only / cumulative_layer_out 是否能把 exact generation 闭合。
```

建议测试模式：

```text
1. base
2. prefix_forced_only
3. result_only
4. cumulative_layer_out
5. prefix_forced + result_only
6. prefix_forced + cumulative_layer_out
7. format_patch_only
8. semantic_patch_only
9. format_patch + semantic_patch
10. random controls
```

关键指标：

```text
exact generation
wrong exact
token0 shared-prefix hit
first discriminative token hit
full prefix length
self-feeding drift examples
```

如果 Phase628 成功，当前 value gate 图谱将变成：

```text
format/prefix gate + semantic value gate -> natural generation closure
```

这比继续扩大 hidden patch 搜索更接近语言编码机制的真实结构。

## Phase 628: Prefix/Format Gate and Semantic Value Integration 前缀格式门与语义值门整合 [2026-06-25 13:11]

### 本阶段目标

根据 Phase627 的关键反证继续推进：

```text
candidate logprob closure != natural greedy generation closure
```

Phase627 已经证明：

```text
result_only / cumulative_layer_out 可以强烈修复第一个区分 value token，
但如果 token0 的 format/prefix 入口错误，exact generation 仍然失败。
```

Phase628 的目标是直接检验：

```text
如果人为强制正确 token0 共享前缀，
semantic value patch 是否能把完整自然生成闭合。
```

### 对附件 Phase627 分析的判断

附件中把 Phase627 判断为“关键反证 + 关键分流阶段”，这个判断正确。

正确部分：

```text
1. candidate logprob 闭合不等于 natural greedy generation 闭合。
2. semantic value gate 与 format/prefix gate 必须分离。
3. DS7B 是最清晰证据：tok1 被修复，但 tok0 错误导致 exact=0。
4. 下一步必须测试 prefix/format gate 与 semantic gate 的组合，而不是继续盲目扩大 hidden patch。
```

需要补充的部分：

```text
Phase627 只证明 token0 是自然生成瓶颈之一，
还没有证明 token0 被修复后 exact generation 一定闭合。
Phase628 正是补上这个缺口。
```

### 脚本

```text
tests/glm5/phase628_prefix_format_semantic_integration.py
tests/glm5/phase628_prefix_format_semantic_integration_summary.py
```

脚本原则：

```text
1. 复用 Phase627 的手写 greedy loop。
2. 新增 forced_prefix_ids，只在自然生成 step0 人工强制正确 token0。
3. 比较 prefix_forced_only 与 prefix_forced + semantic patch。
4. 保留 result_random / cumulative_random / final_output_random controls。
5. 继续统计 exact、wrong_exact、prefix_len、token-position hit。
```

### 执行命令

```bash
python -m py_compile \
  tests/glm5/phase628_prefix_format_semantic_integration.py \
  tests/glm5/phase628_prefix_format_semantic_integration_summary.py

python tests/glm5/phase628_prefix_format_semantic_integration.py qwen3 \
  --smoke --include-nontarget \
  --output-dir results/glm5_phase628_prefix_format_semantic_integration \
  --hard-exit-after-model

python tests/glm5/phase628_prefix_format_semantic_integration.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase628_prefix_format_semantic_integration \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/glm5/phase628_prefix_format_semantic_integration.py glm4 \
  --confirm \
  --output-dir results/glm5_phase628_prefix_format_semantic_integration \
  --hard-exit-after-model

python tests/glm5/phase628_prefix_format_semantic_integration.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase628_prefix_format_semantic_integration \
  --hard-exit-after-model

python tests/glm5/phase628_prefix_format_semantic_integration_summary.py
```

### 输出文件

```text
results/glm5_phase628_prefix_format_semantic_integration/phase628_qwen3_prefix_format_semantic_integration_confirm.json
results/glm5_phase628_prefix_format_semantic_integration/phase628_glm4_prefix_format_semantic_integration_confirm.json
results/glm5_phase628_prefix_format_semantic_integration/phase628_deepseek7b_prefix_format_semantic_integration_confirm.json
results/glm5_phase628_prefix_format_semantic_integration/phase628_cross_model_summary.md
```

### 测试范围

```text
models = qwen3, glm4, deepseek7b
raw cases/model = 256
target filtered rows:
  qwen3 = 17
  glm4 = 31
  deepseek7b = 82

modes:
  base
  repair_prompt
  prefix_forced_only
  result_only
  result_random
  prefix_forced_result_only
  prefix_forced_result_random
  cumulative_layer_out
  cumulative_layer_out_random
  prefix_forced_cumulative_layer_out
  prefix_forced_cumulative_layer_out_random
  final_output_all
  prefix_forced_final_output_all
  final_output_random_all
```

### 客观结果

#### Qwen3

```text
base:
  exact = 1/17
  tok0 = 0.588
  tok1 = 0.059
  tok2 = 0.059

prefix_forced_only:
  exact = 3/17
  wrong_exact = 14/17
  tok0 = 1.000
  tok1 = 0.176
  tok2 = 0.176

result_only:
  exact = 8/17
  tok0 = 0.588
  tok1 = 0.882
  tok2 = 0.647

prefix_forced_result_only:
  exact = 15/17
  wrong_exact = 2/17
  tok0 = 1.000
  tok1 = 0.882
  tok2 = 0.882

cumulative_layer_out:
  exact = 10/17
  tok0 = 0.588
  tok1 = 1.000
  tok2 = 0.824

prefix_forced_cumulative_layer_out:
  exact = 17/17
  wrong_exact = 0/17
  tok0 = 1.000
  tok1 = 1.000
  tok2 = 1.000

prefix_forced_cumulative_layer_out_random:
  exact = 4/17
  wrong_exact = 13/17

prefix_forced_final_output_all:
  exact = 0/17
  tok0 = 1.000
  tok1 = 0.000
```

#### GLM4 bf16

```text
base:
  exact = 2/31
  tok0 = 0.355
  tok1 = 0.065

prefix_forced_only:
  exact = 5/31
  wrong_exact = 26/31
  tok0 = 1.000
  tok1 = 0.161

result_only:
  exact = 10/31
  tok0 = 0.355
  tok1 = 0.935

prefix_forced_result_only:
  exact = 29/31
  wrong_exact = 1/31
  tok0 = 1.000
  tok1 = 0.935

cumulative_layer_out:
  exact = 11/31
  tok0 = 0.355
  tok1 = 1.000

prefix_forced_cumulative_layer_out:
  exact = 31/31
  wrong_exact = 0/31
  tok0 = 1.000
  tok1 = 1.000

prefix_forced_cumulative_layer_out_random:
  exact = 7/31
  wrong_exact = 24/31

prefix_forced_final_output_all:
  exact = 0/31
  tok0 = 1.000
  tok1 = 0.000
```

#### DS7B

```text
base:
  exact = 0/82
  tok0 = 0.000
  tok1 = 0.000
  tok2 = 0.000

prefix_forced_only:
  exact = 3/82
  wrong_exact = 79/82
  tok0 = 1.000
  tok1 = 0.037
  tok2 = 0.037

result_only:
  exact = 0/82
  tok0 = 0.000
  tok1 = 0.902
  tok2 = 0.049

prefix_forced_result_only:
  exact = 74/82
  wrong_exact = 7/82
  tok0 = 1.000
  tok1 = 0.902
  tok2 = 0.902

cumulative_layer_out:
  exact = 0/82
  tok0 = 0.000
  tok1 = 0.988
  tok2 = 0.024

prefix_forced_cumulative_layer_out:
  exact = 81/82
  wrong_exact = 1/82
  tok0 = 1.000
  tok1 = 0.988
  tok2 = 0.988

prefix_forced_cumulative_layer_out_random:
  exact = 8/82
  wrong_exact = 62/82

prefix_forced_final_output_all:
  exact = 0/82
  tok0 = 1.000
  tok1 = 0.000
  tok2 = 0.293
```

### 当前最可靠客观事实

1. **Phase627 的 format/prefix bottleneck 判断被强力确认。**

DS7B 最清楚：

```text
cumulative_layer_out:
  exact = 0/82
  tok0 = 0.000
  tok1 = 0.988

prefix_forced_cumulative_layer_out:
  exact = 81/82
  tok0 = 1.000
  tok1 = 0.988
```

这说明：

```text
Phase627 exact 失败的主因不是 semantic value token 没修复，
而是 format/prefix token 没进入正确生成轨道。
```

2. **prefix_forced_only 不能解决语义选择。**

```text
Qwen3 prefix_forced_only:
  exact = 3/17
  wrong_exact = 14/17

GLM4 prefix_forced_only:
  exact = 5/31
  wrong_exact = 26/31

DS7B prefix_forced_only:
  exact = 3/82
  wrong_exact = 79/82
```

这说明：

```text
format/prefix gate 只负责进入答案格式，
不会自动修复 value semantic choice。
```

3. **semantic patch alone 不能解决格式入口。**

```text
DS7B result_only:
  exact = 0/82
  tok1 = 0.902

DS7B cumulative_layer_out:
  exact = 0/82
  tok1 = 0.988
```

这说明：

```text
semantic value gate 只负责选择正确值，
不会自动修复 token0 format/prefix。
```

4. **format/prefix gate + semantic value gate 可以接近完整自然生成闭合。**

```text
Qwen3:
  prefix_forced_cumulative_layer_out exact = 17/17

GLM4:
  prefix_forced_cumulative_layer_out exact = 31/31

DS7B:
  prefix_forced_cumulative_layer_out exact = 81/82
```

5. **random controls 明显弱，排除“只要强制 prefix 就会成功”的解释。**

```text
DS7B:
  prefix_forced_cumulative_layer_out = 81/82
  prefix_forced_cumulative_layer_out_random = 8/82

GLM4:
  prefix_forced_cumulative_layer_out = 31/31
  prefix_forced_cumulative_layer_out_random = 7/31

Qwen3:
  prefix_forced_cumulative_layer_out = 17/17
  prefix_forced_cumulative_layer_out_random = 4/17
```

6. **final_output_all 即使 prefix forced 也不能闭合。**

```text
Qwen3 prefix_forced_final_output_all exact = 0/17
GLM4 prefix_forced_final_output_all exact = 0/31
DS7B prefix_forced_final_output_all exact = 0/82
```

这进一步证明：

```text
final norm patch 不是自然生成中的稳定可组合状态。
真实可组合路径更像是 layer_out 累积语义 carrier，
而不是最后 logits 前的强行输出态。
```

### 理论进展

Phase628 把自然生成闭环从一个整体指标拆成了可验证的双门结构：

```text
format/prefix gate:
  决定生成是否进入答案格式轨道。

semantic value gate:
  决定进入格式轨道后选择哪个值。
```

当前更精确的链条是：

```text
prompt / rule condition
  -> residual state builder
  -> attention query/value selection
  -> semantic result carrier
  -> downstream layer_out cumulative carrier
  -> discriminative value token

prompt / format condition
  -> format/prefix gate
  -> shared prefix token
  -> generation alignment

semantic value gate + format/prefix gate
  -> exact natural generation
```

这说明语言输出至少不是单一语义向量直接读出，而是多条门控路径共同满足：

```text
正确内容 + 正确格式入口 + 正确自回归轨道
```

### 硬伤和边界

1. **prefix 是人工强制的，不是自然修复。**

Phase628 证明：

```text
如果 prefix gate 正确，semantic patch 足以闭合。
```

但还没有证明：

```text
模型内部哪里自然生成或修复 prefix gate。
```

2. **测试仍是 target filtered value-gate 子集。**

这不是全任务准确率实验，而是机制闭环实验。

3. **donor cache 仍来自 repair/correct condition。**

这说明 causal path 存在，不说明模型原始推理已经正确。

4. **final_output_all 的失败需要单独研究。**

它在 candidate score 中强，但在 natural generation 中不可组合，说明最后读出态可能不是可迁移机制载体。

### 下一步 Phase629

Phase629 应该做：

```text
Format/Prefix Gate Localization
```

核心问题：

```text
自然生成中的 token0 format/prefix gate 在哪里形成？
它是 prompt_last residual state、attention source、MLP format route，
还是早期 lexical/scaffold route。
```

建议测试：

```text
1. 不再人工 force token0，而是 patch token0 generation 前的 prompt_last state。
2. 对 prompt_last 做 residual / attn_out / mlp_out / layer_out 分层 patch。
3. 测 token0 hit、exact、wrong_exact。
4. 与 prefix_forced upper bound 对照。
5. 保留 semantic cumulative patch，测试 format patch + semantic patch 是否自然闭合。
```

Phase629 的目标不是再证明 semantic path，而是定位：

```text
format/prefix gate 的内部写入位置和可修复路径。
```

## Phase 629: Format/Prefix Gate Localization 格式前缀门定位 [2026-06-25 13:41]

### 本阶段目标

根据 Phase628 的强正结果继续推进：

```text
prefix_forced + semantic cumulative 可以让 exact generation 接近闭合。
```

但 Phase628 的最大硬伤是：

```text
token0 是人工强制的，不是模型内部自然修复的。
```

Phase629 的目标：

```text
不再 force token0，
而是 patch 生成 token0 之前的 prompt_last 状态，
定位 format/prefix gate 是否能由 prompt_last residual components 修复。
```

### 对附件 Phase628 分析的判断

附件认为 Phase628 是关键自然生成闭环阶段，这个判断基本正确，但需要严格修正：

```text
Phase628 是 conditional natural generation closure，
不是 full natural generation closure。
```

正确部分：

```text
1. Phase628 确认 semantic value gate 与 format/prefix gate 可分离。
2. prefix_forced_only 不能解决语义选择。
3. semantic patch alone 不能解决格式入口。
4. prefix_forced + cumulative_layer_out 几乎闭合三模型 exact generation。
```

必须补充的边界：

```text
1. token0 是人工强制，所以 format/prefix gate 本身还没有被定位。
2. final_output_all 即使 prefix forced 也失败，说明最终读出态不是稳定可组合 carrier。
3. 下一步不能继续证明 semantic path，而必须定位 format/prefix gate。
```

### 脚本

```text
tests/gpt5/phase629_format_prefix_gate_localization.py
tests/gpt5/phase629_format_prefix_gate_localization_summary.py
```

脚本原则：

```text
1. 收集 base_prompt 与 repair_prompt 在 prompt_last 位置的 residual components。
2. 组件包括 layer_input, attn_out, mlp_out, layer_out。
3. 只 patch prompt_last，不人工 force token0。
4. 同时保留 semantic_cumulative_only。
5. 测试 format_patch_only 与 format_patch + semantic_cumulative。
6. 加 random same-norm semantic control。
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase629_format_prefix_gate_localization.py \
  tests/gpt5/phase629_format_prefix_gate_localization_summary.py

python tests/gpt5/phase629_format_prefix_gate_localization.py qwen3 \
  --smoke --include-nontarget \
  --output-dir results/glm5_phase629_format_prefix_gate_localization \
  --hard-exit-after-model

python tests/gpt5/phase629_format_prefix_gate_localization.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase629_format_prefix_gate_localization \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase629_format_prefix_gate_localization.py glm4 \
  --confirm \
  --output-dir results/glm5_phase629_format_prefix_gate_localization \
  --hard-exit-after-model

python tests/gpt5/phase629_format_prefix_gate_localization.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase629_format_prefix_gate_localization \
  --hard-exit-after-model

python tests/gpt5/phase629_format_prefix_gate_localization_summary.py
```

### 输出文件

```text
results/glm5_phase629_format_prefix_gate_localization/phase629_qwen3_format_prefix_gate_localization_confirm.json
results/glm5_phase629_format_prefix_gate_localization/phase629_glm4_format_prefix_gate_localization_confirm.json
results/glm5_phase629_format_prefix_gate_localization/phase629_deepseek7b_format_prefix_gate_localization_confirm.json
results/glm5_phase629_format_prefix_gate_localization/phase629_cross_model_summary.md
```

### 测试范围

```text
raw cases/model = 256
target filtered rows:
  qwen3 = 17
  glm4 = 31
  deepseek7b = 82

format layers:
  qwen3 = L27-L32
  glm4 = L32-L37
  deepseek7b = L20-L25

components:
  layer_input
  attn_out
  mlp_out
  layer_out
```

### 客观结果

#### Qwen3

```text
base:
  exact = 1/17
  tok0 = 0.588
  tok1 = 0.059
  tok2 = 0.059

repair_prompt:
  exact = 11/17
  tok0 = 0.824
  tok1 = 0.824
  tok2 = 0.824

semantic_cumulative_only:
  exact = 10/17
  tok0 = 0.588
  tok1 = 1.000
  tok2 = 0.824

best prompt_last patch + semantic:
  format_L27_layer_out_semantic
  exact = 13/17
  wrong_exact = 0/17
  tok0 = 0.765
  tok1 = 1.000
  tok2 = 1.000

same format patch only:
  exact = 3/17
  wrong_exact = 10/17
  tok0 = 0.765
  tok1 = 0.235
  tok2 = 0.235

same random_semantic control:
  exact = 9/17
  tok0 = 0.529
  tok1 = 1.000
  tok2 = 0.824
```

Qwen3 结论：

```text
prompt_last L27 layer_out / L28 layer_input 可以部分修复 format/prefix gate。
但 random control 不低，Qwen3 的 format localization 不能过度解释。
```

#### GLM4 bf16

```text
base:
  exact = 2/31
  tok0 = 0.355
  tok1 = 0.065

repair_prompt:
  exact = 28/31
  tok0 = 0.935
  tok1 = 0.903

semantic_cumulative_only:
  exact = 11/31
  tok0 = 0.355
  tok1 = 1.000

best prompt_last patch + semantic:
  format_L32_layer_out_semantic
  exact = 30/31
  wrong_exact = 0/31
  tok0 = 0.968
  tok1 = 1.000

same format patch only:
  exact = 5/31
  wrong_exact = 25/31
  tok0 = 0.968
  tok1 = 0.161

same random_semantic control:
  exact = 11/31
  tok0 = 0.355
  tok1 = 1.000
```

GLM4 结论：

```text
GLM4 的 format/prefix gate 可以被 prompt_last layer_out / layer_input 高度修复。
format patch 单独只打开格式入口，仍大多选择旧错误值。
format patch + semantic cumulative 才接近 exact closure。
```

#### DS7B

```text
base:
  exact = 0/82
  tok0 = 0.000
  tok1 = 0.000
  tok2 = 0.000

repair_prompt:
  exact = 20/82
  tok0 = 0.244
  tok1 = 0.256
  tok2 = 0.256

semantic_cumulative_only:
  exact = 0/82
  tok0 = 0.000
  tok1 = 0.988
  tok2 = 0.024

best prompt_last patch + semantic:
  format_L25_layer_out_semantic
  exact = 21/82
  wrong_exact = 0/82
  tok0 = 0.256
  tok1 = 0.988
  tok2 = 0.305

same format patch only:
  exact = 3/82
  wrong_exact = 17/82
  tok0 = 0.256
  tok1 = 0.037
  tok2 = 0.049

same random_semantic control:
  exact = 0/82
  tok0 = 0.000
  tok1 = 0.988
  tok2 = 0.037
```

DS7B 结论：

```text
prompt_last patch 能轻微打开 format/prefix gate，
但远低于 Phase628 的 forced prefix upper bound。

Phase628:
  prefix_forced_cumulative_layer_out exact = 81/82

Phase629:
  best prompt_last format patch + semantic exact = 21/82
```

### 当前最可靠客观事实

1. **format/prefix gate 不是纯语义 cumulative carrier。**

```text
semantic_cumulative_only:
  Qwen3 exact = 10/17
  GLM4 exact = 11/31
  DS7B exact = 0/82
```

它强烈修复 tok1，但不能稳定修复 tok0。

2. **prompt_last state 确实携带一部分 format/prefix gate 信息。**

```text
Qwen3 best tok0:
  base 0.588 -> 0.765

GLM4 best tok0:
  base 0.355 -> 0.968

DS7B best tok0:
  base 0.000 -> 0.256
```

3. **GLM4 的 format/prefix gate 最容易由 prompt_last layer_out/layer_input 修复。**

```text
GLM4:
  format_L32_layer_out_semantic exact = 30/31
  format_L33_layer_input_semantic exact = 30/31
```

这接近 Phase628 forced-prefix 上界：

```text
prefix_forced_cumulative_layer_out exact = 31/31
```

4. **DS7B 的 format/prefix gate 仍未自然定位闭合。**

DS7B 最强 prompt_last patch：

```text
format_L25_layer_out_semantic exact = 21/82
tok0 = 0.256
```

远低于 Phase628 人工上界：

```text
prefix_forced_cumulative_layer_out exact = 81/82
```

这说明 DS7B 的 format/prefix gate 可能不在单点 prompt_last component，
或需要多层累计 / source token / scaffold route 共同作用。

5. **format patch only 与 format+semantic 的差异再次确认双门结构。**

GLM4：

```text
format_L32_layer_out only:
  exact = 5/31
  wrong_exact = 25/31
  tok0 = 0.968

format_L32_layer_out + semantic:
  exact = 30/31
  wrong_exact = 0/31
  tok0 = 0.968
  tok1 = 1.000
```

这说明：

```text
format gate 打开答案轨道，
semantic gate 决定轨道中写入哪个值。
```

### 理论进展

Phase629 后，自然生成图谱应写成三层：

```text
1. format/prefix gate
   负责进入答案生成轨道。

2. semantic value gate
   负责选择正确 value。

3. continuation/confirmation gate
   负责后续 token 跟随。
```

其中：

```text
GLM4:
  format/prefix gate 主要可由 prompt_last middle-late residual layer_out/input 修复。

Qwen3:
  prompt_last format signal 存在，但有随机对照偏高，定位不够干净。

DS7B:
  prompt_last 单点 patch 只恢复少部分 format gate。
  格式路径更可能是分布式/多层/跨 source token 的机制。
```

因此，Phase628 的完整公式需要修正为：

```text
exact generation =
  format/prefix gate
  AND semantic value gate
  AND continuation gate
```

但 format/prefix gate 本身不是单一位置的简单状态：

```text
format/prefix gate =
  prompt_last local state
  + source/scaffold route
  + multi-layer accumulation
  + model-specific decoding bias
```

### 硬伤和边界

1. **Phase629 只扫描 prompt_last 单点。**

如果 DS7B 的 format gate 来自多个 source token 或多层共同积累，本轮会低估它。

2. **Qwen3 random control 偏高。**

Qwen3 的 prompt_last 定位只能视为弱正结果，不能作为独立强机制闭合。

3. **repair_prompt 本身并不总是正确格式上界。**

例如 DS7B repair_prompt exact 只有 20/82，所以 donor prompt_last 并不是完美 format donor。

4. **format patch 与 semantic patch 使用不同 donor 条件。**

这证明组合因果有效，但还不是模型原始自然推理。

### 下一步 Phase630

Phase630 应做：

```text
Distributed Format Route Multi-Source Sweep
```

核心目标：

```text
解释 DS7B 中 Phase628 forced prefix 上界很高，
但 Phase629 prompt_last patch 只恢复 21/82 的差距。
```

建议测试：

```text
1. 扫描 source positions:
   prompt_last
   object token
   relation token
   rule/value line tokens
   punctuation/colon/newline tokens

2. 扫描 patch form:
   single position
   multi-position cumulative
   multi-layer layer_out cumulative

3. 固定 semantic_cumulative_layer_out，
   只观察 token0 format/prefix 是否自然恢复。

4. 将 Phase628 forced prefix 作为 upper bound。

5. 重点 DS7B，Qwen3/GLM4 做跨模型对照。
```

Phase630 的目标不是再证明双门结构，而是找到：

```text
format/prefix gate 的分布式 source route。
```

## Phase 630: Distributed Format Route Multi-Source Sweep 分布式格式路径多源扫描 [2026-06-25 14:19]

### 本阶段目标

根据 Phase629 的边界继续推进：

```text
DS7B:
  Phase628 forced prefix + semantic exact = 81/82
  Phase629 best prompt_last patch + semantic exact = 21/82
```

Phase630 的目标：

```text
检查 DS7B 缺失的 format/prefix gate 是否来自多个 post-query source spans，
而不是单点 prompt_last。
```

### 对附件 Phase629 分析的判断

附件中认为 Phase629 是关键定位阶段，这个判断正确。

正确部分：

```text
1. Phase629 不再证明 semantic value gate，而是追问 token0 format/prefix gate。
2. Phase628 只能叫 conditional natural generation closure。
3. GLM4 的 format gate 可由 prompt_last residual state 高度修复。
4. Qwen3 是弱正结果，random control 偏高。
5. DS7B 的 format gate 不是单点 prompt_last patch 可以闭合的。
```

需要继续推进的部分：

```text
Phase629 只扫描 prompt_last 单点。
Phase630 扩展到 answer_label、question_mark_answer、relation_tail、
question_subject、question_all 等多 source spans。
```

### 脚本

```text
tests/gpt5/phase630_distributed_format_route_multisource.py
tests/gpt5/phase630_distributed_format_route_multisource_summary.py
```

脚本原则：

```text
1. 不人工 force token0。
2. 固定 semantic_cumulative_layer_out，观察 token0 format/prefix 是否恢复。
3. 扫描 source groups:
   prompt_last
   answer_label
   question_mark_answer
   relation_tail
   question_subject
   question_all
4. 默认扫描 layer_out，避免搜索空间过大。
5. 每个 source patch 同时测:
   source_only
   source + semantic
   random_source + semantic
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase630_distributed_format_route_multisource.py \
  tests/gpt5/phase630_distributed_format_route_multisource_summary.py

python tests/gpt5/phase630_distributed_format_route_multisource.py qwen3 \
  --smoke --include-nontarget \
  --output-dir results/glm5_phase630_distributed_format_route_multisource \
  --hard-exit-after-model

python tests/gpt5/phase630_distributed_format_route_multisource.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase630_distributed_format_route_multisource \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase630_distributed_format_route_multisource.py glm4 \
  --confirm \
  --output-dir results/glm5_phase630_distributed_format_route_multisource \
  --hard-exit-after-model

python tests/gpt5/phase630_distributed_format_route_multisource.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase630_distributed_format_route_multisource \
  --hard-exit-after-model

python tests/gpt5/phase630_distributed_format_route_multisource_summary.py
```

### 输出文件

```text
results/glm5_phase630_distributed_format_route_multisource/phase630_qwen3_distributed_format_route_multisource_confirm.json
results/glm5_phase630_distributed_format_route_multisource/phase630_glm4_distributed_format_route_multisource_confirm.json
results/glm5_phase630_distributed_format_route_multisource/phase630_deepseek7b_distributed_format_route_multisource_confirm.json
results/glm5_phase630_distributed_format_route_multisource/phase630_cross_model_summary.md
```

### 测试范围

```text
raw cases/model = 256
target filtered rows:
  qwen3 = 17
  glm4 = 31
  deepseek7b = 82

format layers:
  qwen3 = L27-L32
  glm4 = L32-L37
  deepseek7b = L20-L25

source groups:
  prompt_last
  answer_label
  question_mark_answer
  relation_tail
  question_subject
  question_all

component:
  layer_out
```

### 客观结果

#### Qwen3

```text
base:
  exact = 1/17
  tok0 = 0.588
  tok1 = 0.059
  tok2 = 0.059

repair_prompt:
  exact = 11/17
  tok0 = 0.824
  tok1 = 0.824
  tok2 = 0.824

semantic_cumulative_only:
  exact = 10/17
  tok0 = 0.588
  tok1 = 1.000
  tok2 = 0.824

best source + semantic:
  question_all_L27_layer_out_semantic
  exact = 14/17
  wrong_exact = 0/17
  tok0 = 0.824
  tok1 = 1.000
  tok2 = 1.000

same source only:
  question_all_L27_layer_out
  exact = 4/17
  wrong_exact = 10/17
  tok0 = 0.824
  tok1 = 0.294
  tok2 = 0.294

same random_source + semantic:
  exact = 10/17
  tok0 = 0.588
  tok1 = 1.000
  tok2 = 0.824
```

Qwen3 结论：

```text
question_all 比 prompt_last 单点略强：
  Phase629 best = 13/17
  Phase630 best = 14/17

source_only 主要打开格式入口，但仍产生大量 wrong_exact。
source + semantic 才能减少 wrong_exact。
```

#### GLM4 bf16

```text
base:
  exact = 2/31
  tok0 = 0.355
  tok1 = 0.065

repair_prompt:
  exact = 28/31
  tok0 = 0.935
  tok1 = 0.903

semantic_cumulative_only:
  exact = 11/31
  tok0 = 0.355
  tok1 = 1.000

best source + semantic:
  answer_label_L32_layer_out_semantic
  exact = 30/31
  wrong_exact = 0/31
  tok0 = 0.968
  tok1 = 1.000

same source only:
  answer_label_L32_layer_out
  exact = 5/31
  wrong_exact = 25/31
  tok0 = 0.968
  tok1 = 0.161

same random_source + semantic:
  exact = 14/31
  tok0 = 0.452
  tok1 = 1.000
```

GLM4 结论：

```text
多 source 没有显著超过 Phase629 prompt_last 单点。
answer_label / prompt_last / question_mark_answer / relation_tail 都可达到 30/31。
GLM4 format gate 是局部后缀区间高度可修复的状态。
```

#### DS7B

```text
base:
  exact = 0/82
  tok0 = 0.000
  tok1 = 0.000
  tok2 = 0.000

repair_prompt:
  exact = 20/82
  tok0 = 0.244
  tok1 = 0.256
  tok2 = 0.256

semantic_cumulative_only:
  exact = 0/82
  tok0 = 0.000
  tok1 = 0.988
  tok2 = 0.024

best source + semantic:
  answer_label_L21_layer_out_semantic
  exact = 21/82
  wrong_exact = 0/82
  tok0 = 0.256
  tok1 = 0.988
  tok2 = 0.280

same source only:
  answer_label_L21_layer_out
  exact = 3/82
  wrong_exact = 18/82
  tok0 = 0.256
  tok1 = 0.037
  tok2 = 0.037

same random_source + semantic:
  exact = 0/82
  tok0 = 0.000
  tok1 = 0.988
  tok2 = 0.085
```

DS7B 结论：

```text
多 source layer_out 扫描没有突破 Phase629 的 21/82。

Phase628 upper bound:
  prefix_forced_cumulative_layer_out exact = 81/82

Phase629:
  prompt_last best exact = 21/82

Phase630:
  answer_label/question_mark_answer/relation_tail best exact = 21/82
```

### 当前最可靠客观事实

1. **format/prefix gate 与 semantic gate 的双门结构再次成立。**

source_only 往往打开 token0，但仍 wrong_exact 多。
source + semantic 才能把 wrong_exact 降下来。

2. **Qwen3 的 format route 有轻微分布式后缀增强。**

```text
prompt_last best from Phase629 = 13/17
question_all best from Phase630 = 14/17
```

3. **GLM4 的 format route 是后缀区间可替代状态。**

```text
answer_label / prompt_last / question_mark_answer / relation_tail
都能达到 30/31。
```

4. **DS7B 的 format route 不是当前扫描的单组 source layer_out 可以解释的。**

DS7B 在所有 source groups 中最高仍是：

```text
exact = 21/82
tok0 = 0.256
```

这没有接近 forced prefix 上界：

```text
81/82
```

5. **DS7B 的问题更像 final token0 readout/embedding/logit competition，而不是缺少某个简单 source carrier。**

因为：

```text
semantic tok1 已接近闭合；
source layer_out patch 可以略微改变 token0；
但任何单组 source 都不能把 token0 推到 “ v” 轨道。
```

### 理论进展

Phase630 后，当前自然生成机制应写成：

```text
semantic value gate:
  已经能在 DS7B 中强修复 tok1。

format/prefix gate:
  GLM4 可由后缀局部状态修复；
  Qwen3 有后缀多源弱增强；
  DS7B 不由当前单组 source layer_out 决定。

token0 readout barrier:
  DS7B 的主要剩余瓶颈。
```

这说明对 DS7B 来说，继续扩大 source position patch 的收益可能很低。
研究对象应从：

```text
where is the source state?
```

转向：

```text
why does token0 readout reject the desired prefix token?
```

也就是读出端竞争问题。

### 硬伤和边界

1. **Phase630 默认只测 layer_out。**

它没有穷尽 layer_input/attn_out/mlp_out 的所有 source group 组合。
但 Phase629 已经显示 DS7B 单点 components 也只能到 21/82 附近。

2. **没有做 multi-group simultaneous patch。**

本阶段扫描单组 source span；如果 DS7B 需要多个 source groups 同时修复，本轮仍会低估。

3. **source spans 都在 question/answer 后缀区间。**

没有扫描早期 rule/value lines 的多位置累计。
不过这些早期 token 在 base/repair prompt 分歧之前，hidden state 理论上差异很小。

4. **DS7B forced prefix 上界仍未解释。**

Phase630 缩小了搜索空间，但没有闭合 DS7B format gate。

### 下一步 Phase631

Phase631 应做：

```text
Token0 Prefix Readout Competition Audit
```

核心问题：

```text
DS7B 为什么即使 semantic carrier 正确，
仍然不选择 token0 = " v"？
```

建议测试：

```text
1. 直接审计 token0 logits:
   correct prefix token " v"
   competing tokens " ?\n\n", " c", " o", newline, space

2. 比较条件:
   base
   repair_prompt
   semantic_cumulative
   best format source patch
   best format source + semantic
   forced prefix upper-bound reference

3. 测 final_norm / unembedding 方向:
   prefix token logit delta
   competitor token logit delta
   margin correct_prefix - top_competitor

4. 做读出方向 patch:
   沿 unembedding(" v") - unembedding(top_competitor) 方向小 scale 注入，
   检查是否能把 DS7B token0 推入正确轨道。

5. 若读出方向有效，再回溯哪个内部组件自然产生该读出方向。
```

Phase631 的目标：

```text
把 DS7B format/prefix bottleneck 从 source-state 搜索，
推进到 token0 readout competition 的客观分解。
```

## Phase 631: Token0 Prefix Readout Competition Audit 词元0前缀读出竞争审计 [2026-06-25 14:45]

### 本阶段目标

根据用户附件对 Phase630 的分析，先判断其是否正确，再继续完成客观拼图。

附件中正确部分：

```text
1. Phase630 是重要负结果 + 关键收缩阶段。
2. DS7B 的 format/prefix gate 不是当前扫描的单组 post-query source layer_out 能解释。
3. GLM4 的格式门更像局部后缀 residual state 可修复。
4. Qwen3 有轻微分布式后缀增强，但不是强闭合证据。
5. DS7B 下一步应从 source-state 搜索转向 token0 readout competition。
```

Phase631 的直接目标：

```text
直接测量第一个生成词元 token0 的 correct prefix token 与 top competitor 的 logit margin。
在 final_norm 输出上沿 unembedding(correct_prefix) - unembedding(top_competitor) 方向注入。
观察该 readout-direction intervention 是否能替代缺失的 format/prefix gate。
```

### 生成脚本

```text
tests/gpt5/phase631_token0_prefix_readout_competition.py
tests/gpt5/phase631_token0_prefix_readout_competition_summary.py
```

输出目录：

```text
results/glm5_phase631_token0_prefix_readout_competition
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase631_token0_prefix_readout_competition.py \
  tests/gpt5/phase631_token0_prefix_readout_competition_summary.py

python tests/gpt5/phase631_token0_prefix_readout_competition.py qwen3 \
  --smoke \
  --include-nontarget \
  --output-dir results/glm5_phase631_token0_prefix_readout_competition \
  --hard-exit-after-model

python tests/gpt5/phase631_token0_prefix_readout_competition.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase631_token0_prefix_readout_competition \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase631_token0_prefix_readout_competition.py glm4 \
  --confirm \
  --output-dir results/glm5_phase631_token0_prefix_readout_competition \
  --hard-exit-after-model

python tests/gpt5/phase631_token0_prefix_readout_competition.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase631_token0_prefix_readout_competition \
  --hard-exit-after-model

python tests/gpt5/phase631_token0_prefix_readout_competition_summary.py
```

### 测试原理

此前阶段证明：

```text
semantic cumulative patch 可以修复答案值的后续语义词元。
但自然生成仍经常失败在 token0，例如没有先生成共享前缀 " v"。
```

本轮把第一个词元当作读出竞争问题：

```text
prefix token:
  正确答案共享前缀，例如 " v"

competitor token:
  base prompt 下 token0 的 top-1 竞争词元，例如 DS7B 常见 " ?\n\n"
```

直接计算：

```text
margin = logit(prefix_token) - logit(top_competitor)
```

然后在 final_norm 的 prompt_last 输出处注入：

```text
delta = scale * ||h_final|| * normalize(W[prefix_token] - W[top_competitor])
```

其中 W 是 lm_head/unembedding 权重。

如果注入后 token0 大幅变成正确前缀，并且加 semantic cumulative 后 exact generation 接近 forced-prefix 上界，则说明：

```text
剩余瓶颈主要在 token0 readout competition，而不是语义值门缺失。
```

### 关键结果

#### Qwen3

```text
target rows = 17 / raw cases = 256
source = question_all L27 layer_out
downstream = [29, 30, 31, 32, 33, 34, 35]

base:
  tok0 = 10/17
  exact = 1/17
  wrong_exact = 9/17
  mean_prefix_margin = 0.213

semantic_cumulative:
  tok0 = 10/17
  exact = 10/17
  wrong_exact = 0/17
  mean_prefix_margin = 0.213

best_source_semantic:
  tok0 = 14/17
  exact = 14/17
  wrong_exact = 0/17
  mean_prefix_margin = 1.110

readout_scale0.125_semantic:
  tok0 = 17/17
  exact = 17/17
  wrong_exact = 0/17
  mean_prefix_margin = 28.592

readout_scale0.25_semantic:
  tok0 = 17/17
  exact = 17/17
  wrong_exact = 0/17
  mean_prefix_margin = 56.974
```

Qwen3 现象：

```text
readout direction 可完全打开 token0 前缀门。
不加 semantic cumulative 时，readout_scale 只能保证前缀，exact 仍只有 3/17。
加 semantic cumulative 后，token0 + 后续值词元同时闭合。
```

#### GLM4

```text
target rows = 31 / raw cases = 256
source = answer_label L32 layer_out
downstream = [34, 35, 36, 37, 38, 39]

base:
  tok0 = 11/31
  exact = 2/31
  wrong_exact = 9/31
  mean_prefix_margin = -0.226

semantic_cumulative:
  tok0 = 11/31
  exact = 11/31
  wrong_exact = 0/31
  mean_prefix_margin = -0.226

best_source_semantic:
  tok0 = 30/31
  exact = 30/31
  wrong_exact = 0/31
  mean_prefix_margin = 1.442

readout_scale0.125_semantic:
  tok0 = 31/31
  exact = 31/31
  wrong_exact = 0/31
  mean_prefix_margin = 20.704

readout_scale0.25_semantic:
  tok0 = 31/31
  exact = 31/31
  wrong_exact = 0/31
  mean_prefix_margin = 41.673
```

GLM4 现象：

```text
GLM4 的自然 source patch 已经接近闭合。
readout direction 进一步把剩余 1/31 的前缀失败补上。
但 readout-only exact 仍只有 5/31，说明 token0 前缀门不能替代语义值门。
```

#### DS7B

```text
target rows = 82 / raw cases = 256
source = answer_label L21 layer_out
downstream = [22, 23, 24, 25, 26, 27]

base:
  tok0 = 0/82
  exact = 0/82
  wrong_exact = 0/82
  mean_prefix_margin = -6.356

repair_prompt:
  tok0 = 20/82
  exact = 20/82
  wrong_exact = 0/82
  mean_prefix_margin = -1.699

semantic_cumulative:
  tok0 = 0/82
  exact = 0/82
  wrong_exact = 0/82
  mean_prefix_margin = -6.356

best_source_semantic:
  tok0 = 21/82
  exact = 21/82
  wrong_exact = 0/82
  mean_prefix_margin = -2.158

readout_scale0.125_semantic:
  tok0 = 70/82
  exact = 70/82
  wrong_exact = 0/82
  mean_prefix_margin = 24.727

readout_scale0.25_semantic:
  tok0 = 82/82
  exact = 81/82
  wrong_exact = 1/82
  mean_prefix_margin = 55.759

readout_scale0.5_semantic:
  tok0 = 82/82
  exact = 81/82
  wrong_exact = 1/82
  mean_prefix_margin = 117.930
```

DS7B 现象：

```text
base 下 correct prefix " v" 对 top competitor 平均落后 6.356 logit。
semantic_cumulative 完全不能改变 token0，说明语义值门不负责第一个格式前缀词元。
best_source_semantic 只能到 21/82，与 Phase629/630 一致。
readout_scale0.25_semantic 直接达到 81/82，几乎复现 Phase628 forced-prefix 上界。
```

典型 DS7B 样例：

```text
base:
  token0 = " ?\n\n"
  generation = " ?\n\nTo solve"

semantic_cumulative:
  token0 = " ?\n\n"
  generation = " ?\n\n2\n"

best_source_semantic:
  token0 = " ?\n\n"
  generation = " ?\n\n2\n"

readout_scale1_semantic:
  token0 = " v"
  generation = " v22"
```

### 当前客观拼图

本轮把 Phase630 的负结果推进成了正定位：

```text
Phase630:
  DS7B 的 format/prefix gate 不在当前单组 source layer_out 中。

Phase631:
  DS7B 的 format/prefix gate 可以被 final_norm readout direction 直接打开。
```

这说明当前机制至少分成两层：

```text
1. semantic value gate:
   后续答案值词元的语义修复。
   对 DS7B 已经接近闭合。

2. token0 prefix readout gate:
   第一个格式/前缀词元的读出竞争。
   DS7B 自然状态下强烈偏向 " ?\n\n" 等竞争格式词元。
```

更具体地：

```text
readout-only:
  可以打开 " v" 前缀，但后续值仍错，wrong_exact 很多。

semantic-only:
  可以修复后续值，但 token0 仍被错误格式词元拦截。

readout + semantic:
  前缀轨道与语义值同时修复，生成接近闭合。
```

### 理论进展

当前自然生成可写成更客观的双门乘积：

```text
ExactGeneration
≈ PrefixReadoutGate(token0)
  AND SemanticValueGate(token1/token2)
  AND DownstreamStability
```

对 DS7B：

```text
SemanticValueGate:
  已经可由 cumulative layer_out patch 强修复。

PrefixReadoutGate:
  不是当前 source layer_out patch 能自然修复；
  但可以由 final_norm unembedding margin 方向直接人工打开。
```

这不是完整机制闭合，因为 readout direction 是人工构造，不是模型内部自然生成的来源。
但它把瓶颈位置从模糊的 source search 明确推进到：

```text
哪个内部组件自然产生 W[" v"] - W[competitor] 方向的 final readout bias？
```

### 硬伤和边界

1. **readout 注入是人工方向，不是自然路径**

本轮证明该方向充分有效，但还没有证明模型内部哪个头、MLP、残差路径自然写入该方向。

2. **scale 造成的 margin 很大**

0.25 以上已经把 margin 推到几十，属于强干预。
它证明读出竞争可以被方向性控制，但不能说明自然机制使用同等强度。

3. **competitor 只取 base top-1**

本轮主要比较 correct prefix 与 base top competitor。
如果干预后出现新的竞争词元，本轮没有完整建立多竞争者动力学图。

4. **target-only 样本**

主结果集中在 base 错、repair 对的目标样本。
这适合定位失败机制，但还需要对非目标样本做副作用审计，防止方向注入破坏本来正确的样本。

5. **仍未解释 1/82 剩余失败**

DS7B readout + semantic 达到 81/82，与 Phase628 上界一致，但没有解释最后一个样本失败原因。

### 下一步任务

Phase632 应从人工 readout direction 回溯到自然组件：

```text
Natural Prefix Readout Writer Backtrace
```

核心目标：

```text
寻找哪些 attention head / MLP / residual delta 在 prompt_last final_norm 前，
自然写入 W[" v"] - W[top_competitor] 方向。
```

测试要求：

```text
1. 对每层 layer_out、attn_out、mlp_out 计算 prefix_readout_margin_delta。
2. 分解 DS7B base vs repair_prompt 的 margin 变化。
3. 对候选 writer 做 causal remove / restore。
4. 同时保留 semantic cumulative patch，检查 writer restore 是否能把 exact 从 21/82 推向 81/82。
5. 加 random_same_norm、wrong_prefix_direction、non-target side-effect control。
```

判据：

```text
如果某个自然 writer 的 restore 能提高 token0 " v" 命中，
并在 semantic cumulative 条件下显著提高 exact，
则它是 prefix readout gate 的自然写入器候选。

如果所有单 writer 都失败，
则下一步进入 multi-writer cumulative readout field，而不是继续单点 patch。
```

## Phase 632: Natural Prefix Readout Writer Backtrace 自然前缀读出写入器回溯 [2026-06-25 15:26]

### 本阶段目标

根据用户附件，当前研究应同时推进两件事：

```text
1. Phase631 的 readout direction 是人工构造，不是自然机制闭合。
2. 语言智能更可能来自相对编码 + 复用差分机制，需要开始图谱化。
```

本阶段不做大理论总结，而是把这两个判断落到一个客观测试：

```text
从 Phase631 的人工 W[" v"] - W[top_competitor] readout direction 回溯，
寻找哪些自然层/组件的 base->repair 差分会写入该 prefix readout 方向。
```

核心问题：

```text
自然 forward 中，哪些 layer/component 在 prompt_last 位置增加
logit(" v") - logit(top_competitor) 的 margin？

这些自然 writer 单独 restore 后，能否在 semantic cumulative 条件下把 exact generation 推向 Phase631 的人工 readout 上界？
```

### 生成脚本

```text
tests/gpt5/phase632_natural_prefix_readout_writer_backtrace.py
tests/gpt5/phase632_natural_prefix_readout_writer_backtrace_summary.py
```

输出目录：

```text
results/glm5_phase632_natural_prefix_readout_writer_backtrace
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase632_natural_prefix_readout_writer_backtrace.py \
  tests/gpt5/phase632_natural_prefix_readout_writer_backtrace_summary.py

python tests/gpt5/phase632_natural_prefix_readout_writer_backtrace.py qwen3 \
  --smoke \
  --include-nontarget \
  --output-dir results/glm5_phase632_natural_prefix_readout_writer_backtrace \
  --hard-exit-after-model

python tests/gpt5/phase632_natural_prefix_readout_writer_backtrace.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase632_natural_prefix_readout_writer_backtrace \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase632_natural_prefix_readout_writer_backtrace.py glm4 \
  --confirm \
  --output-dir results/glm5_phase632_natural_prefix_readout_writer_backtrace \
  --hard-exit-after-model

python tests/gpt5/phase632_natural_prefix_readout_writer_backtrace.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase632_natural_prefix_readout_writer_backtrace \
  --hard-exit-after-model

python tests/gpt5/phase632_natural_prefix_readout_writer_backtrace_summary.py
```

### 测试原理

Phase631 证明人工方向：

```text
u_prefix = normalize(W[" v"] - W[top_competitor])
```

可以打开 token0 prefix gate。

Phase632 不再直接注入这个人工方向，而是对自然 base->repair 差分做图谱：

```text
delta_h(l, component) = h_repair(l, component, prompt_last) - h_base(l, component, prompt_last)
```

对每个层/组件计算它对前缀读出边际的自然贡献：

```text
natural_margin_delta =
  dot(W[" v"] - W[top_competitor], delta_h)
```

如果某个节点 natural_margin_delta 大且 positive_rate 高，说明它自然朝正确前缀读出方向移动。

随后对 top writer 做因果验证：

```text
restore:
  base prompt 中把该节点替换为 repair 对应节点。

restore_semantic:
  restore + semantic cumulative patch。

random_semantic:
  同范数随机 delta + semantic cumulative。

reverse_semantic:
  反向 delta + semantic cumulative。

remove_from_repair:
  repair prompt 中把该节点替换回 base 对应节点。
```

判据：

```text
如果 restore_semantic 显著超过 semantic_cumulative，
且 random/reverse 不能复现，则该节点是自然 prefix writer。

如果 scan 很强但 restore_semantic 仍不能接近 Phase631 上界，
说明 prefix readout gate 是多 writer / 累积场，而不是单 writer。
```

### 关键结果

#### Qwen3

目标样本：

```text
rows = 17 / raw_cases = 256
scan_layers = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
top_nodes = L34_layer_out, L35_layer_input, L33_layer_out, L34_layer_input, L32_layer_out, L33_layer_input
```

自然 margin writer 扫描：

```text
L34_layer_out:
  mean_margin_delta = 4.610
  positive_rate = 0.941
  mean_cos = 0.045

L35_layer_input:
  mean_margin_delta = 4.610
  positive_rate = 0.941
  mean_cos = 0.045

L33_layer_out:
  mean_margin_delta = 3.779
  positive_rate = 0.765
```

因果验证：

```text
baseline base:
  tok0 = 10/17
  exact = 1/17
  wrong_exact = 9/17

baseline semantic_cumulative:
  tok0 = 10/17
  exact = 10/17

baseline repair_prompt:
  tok0 = 14/17
  exact = 11/17

L34_layer_out restore_semantic:
  tok0 = 14/17
  exact = 14/17
  wrong_exact = 0/17
  mean_prefix_margin = 1.044

L35_layer_input restore_semantic:
  tok0 = 14/17
  exact = 14/17
  wrong_exact = 0/17
  mean_prefix_margin = 1.044
```

Qwen3 结论：

```text
后层 L34/L35 的自然差分确实携带 prefix readout 修复信号。
单节点 restore_semantic 可从 semantic_cumulative 的 10/17 提高到 14/17。
但它没有达到 Phase631 readout_scale_semantic 的 17/17，说明单 writer 不足。
```

#### GLM4

目标样本：

```text
rows = 31 / raw_cases = 256
scan_layers = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
top_nodes = L38_layer_out, L39_layer_input, L37_layer_out, L38_layer_input, L36_layer_out, L37_layer_input
```

自然 margin writer 扫描：

```text
L38_layer_out:
  mean_margin_delta = 3.842
  positive_rate = 1.000
  mean_cos = 0.085

L39_layer_input:
  mean_margin_delta = 3.842
  positive_rate = 1.000
  mean_cos = 0.085

L37_layer_out:
  mean_margin_delta = 3.068
  positive_rate = 1.000
```

因果验证：

```text
baseline base:
  tok0 = 11/31
  exact = 2/31
  wrong_exact = 9/31

baseline semantic_cumulative:
  tok0 = 11/31
  exact = 11/31

baseline repair_prompt:
  tok0 = 29/31
  exact = 28/31

L38_layer_out restore_semantic:
  tok0 = 29/31
  exact = 29/31
  wrong_exact = 0/31
  mean_prefix_margin = 1.712

L39_layer_input restore_semantic:
  tok0 = 29/31
  exact = 29/31
  wrong_exact = 0/31
  mean_prefix_margin = 1.712

remove_from_repair:
  多个 top writer 移除后 exact 降到 9/31 到 10/31。
```

GLM4 结论：

```text
GLM4 的自然 prefix writer 很清楚，后层 L36-L39 均稳定正向写入。
restore_semantic 从 11/31 提高到 29/31。
remove_from_repair 显著破坏 repair_prompt，说明这些后层 writer 对格式门是必要成分之一。
但仍低于 Phase631 人工 readout 的 31/31，也略低于 Phase630 best_source_semantic 的 30/31。
```

#### DS7B

目标样本：

```text
rows = 82 / raw_cases = 256
scan_layers = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
top_nodes = L26_layer_out, L27_layer_input, L27_layer_out, L25_layer_out, L26_layer_input, L24_layer_out
```

自然 margin writer 扫描：

```text
L26_layer_out:
  mean_margin_delta = 33.271
  positive_rate = 1.000
  mean_cos = 0.090

L27_layer_input:
  mean_margin_delta = 33.271
  positive_rate = 1.000
  mean_cos = 0.090

L27_layer_out:
  mean_margin_delta = 32.667
  positive_rate = 0.988
  mean_cos = 0.067

L25_layer_out:
  mean_margin_delta = 17.553
  positive_rate = 0.988
```

因果验证：

```text
baseline base:
  tok0 = 0/82
  exact = 0/82
  mean_prefix_margin = -6.356

baseline semantic_cumulative:
  tok0 = 0/82
  exact = 0/82
  mean_prefix_margin = -6.356

baseline repair_prompt:
  tok0 = 20/82
  exact = 20/82
  mean_prefix_margin = -1.699

L26_layer_out restore_semantic:
  tok0 = 21/82
  exact = 21/82
  mean_prefix_margin = -1.662

L27_layer_input restore_semantic:
  tok0 = 21/82
  exact = 21/82
  mean_prefix_margin = -1.662

L25_layer_out restore_semantic:
  tok0 = 21/82
  exact = 21/82
  mean_prefix_margin = -1.755

random_semantic:
  最高仅 1/82，多数 0/82。

remove_from_repair:
  多数降到 0/82 或 1/82。
```

DS7B 结论：

```text
DS7B 的自然 prefix margin 差分非常强，集中在 L24-L27 后层残差/层输出。
这些节点对 repair_prompt 是必要的：remove_from_repair 会几乎打回 base。
但是单节点 restore_semantic 最高只有 21/82，与 Phase629/630 持平。
它远低于 Phase631 人工 readout_scale0.25_semantic 的 81/82。
```

### 当前客观拼图

Phase632 得到一个关键的“正扫描 + 负闭合”结果：

```text
正结果：
  自然 prefix readout writer 确实存在，并且可被 margin_delta 指标稳定定位。

负结果：
  单个自然 writer restore 无法复现 Phase631 的人工 readout 闭合。
```

跨模型共同现象：

```text
1. prefix readout writer 集中在后层 layer_out/layer_input。
2. layer_out 与下一层 layer_input 成对出现，说明它们主要是残差流连续状态。
3. random_same_norm 不能复现 restore_semantic，说明不是范数扰动。
4. remove_from_repair 会破坏 repair_prompt，说明这些 writer 对自然修复有必要性。
5. 单 writer restore 仍不足，说明 prefix gate 更像多 writer cumulative field。
```

这正好支持“相对编码 + 复用差分”的方向：

```text
同一后层残差骨架被复用；
不同 prompt 条件通过 base->repair 的差分状态改变 prefix token 的相对读出边际；
但完整 gate 不是单点差分，而是跨层累积差分场。
```

### 理论进展

当前 prefix gate 应从单节点模型升级为累积读出场：

```text
M_prefix =
  dot(W[prefix] - W[competitor], h_final)
```

自然差分分解为：

```text
Delta M_prefix
≈ sum over writers dot(W[prefix] - W[competitor], Delta h_writer)
```

其中单个 writer 可以提供局部正贡献，但完整翻转需要多 writer 累积。

因此 Phase632 后，当前更合理的图谱节点不是：

```text
one writer causes prefix gate
```

而是：

```text
prefix readout field =
  residual-layer writer chain + attention/MLP local increments + final unembedding competition
```

### 硬伤和边界

1. **扫描强不等于因果闭合**

DS7B 的 L26/L27 margin_delta 很强，但 restore_semantic 仍只有 21/82。
这说明 margin_delta 能定位自然差分方向，但单节点 patch 不足以复现完整轨迹。

2. **layer_out / next layer_input 重复**

很多 top 节点成对出现，例如 L26_layer_out 和 L27_layer_input。
这不是两个独立机制，而是同一残差状态跨层传递的两个观测点。

3. **没有做多节点 cumulative restore**

本阶段故意先做单节点审计。
既然单节点不足，下一步必须做多节点累积，而不是继续扩大单节点搜索。

4. **只扫 prompt_last**

如果 prefix gate 还依赖 earlier source tokens 或 answer_label span 的多位置场，本轮没有覆盖。

5. **target-only 仍可能高估机制**

主样本仍是 base 错、repair 对目标子集。
下一步多节点 patch 必须增加 non-target side-effect control。

### 下一步任务

Phase633 应执行：

```text
Multi-Writer Prefix Readout Field Closure
```

目标：

```text
把 Phase632 的 top natural writer 从单节点 restore 升级为多节点 cumulative restore，
检查是否能从 21/82 接近 Phase631 人工 readout 的 81/82。
```

测试要求：

```text
1. 对 top writer chain 做 cumulative restore:
   top1, top2, top4, top8, top12。

2. 区分重复残差观测点:
   不同时使用 L26_layer_out 与 L27_layer_input 这类等价邻接点，
   构建去重后的 residual writer chain。

3. 每个 cumulative set 测：
   restore_only
   restore_semantic
   random_same_norm_semantic
   reverse_semantic
   remove_from_repair

4. 指标：
   tok0 hit
   exact generation
   wrong_exact
   mean_prefix_margin
   top competitor changes

5. 三模型都跑，但重点看 DS7B：
   如果 DS7B cumulative restore_semantic 接近 81/82，
   prefix readout field 基本闭合。

   如果仍停在 21/82，
   说明 prompt_last residual writer chain 仍不是完整路径，
   下一步必须加入多位置 source/format field。
```

阶段性判据：

```text
单 writer 不足已经成立。
下一步不应继续寻找“唯一自然 writer”，而应测试“多 writer 累积场”。
```

## Phase 633: Multi-Writer Prefix Readout Field Closure 多写入器前缀读出场闭合 [2026-06-25 15:53]

### 本阶段目标

根据用户附件对 Phase632 的分析，先判断其是否正确，再继续完成客观拼图。

附件中正确部分：

```text
1. Phase632 不是普通 patch，而是进入复用差分图谱方向。
2. Phase632 完成了从人工 readout direction 到自然 writer 的回溯。
3. 自然 prefix writer 存在，且集中在后层 residual layer_out / layer_input。
4. 单 writer restore 无法闭合，说明 prefix gate 更像多 writer cumulative readout field。
5. 下一步应测试多 writer 累积，而不是继续寻找唯一 writer。
```

Phase633 目标：

```text
把 Phase632 的 top natural writer 从单节点 restore 升级为多节点 cumulative restore。
检查 prompt_last residual writer field 是否能把 DS7B 从 21/82 推近 Phase631 人工 readout 的 81/82。
```

### 生成脚本

```text
tests/gpt5/phase633_multi_writer_prefix_readout_field_closure.py
tests/gpt5/phase633_multi_writer_prefix_readout_field_closure_summary.py
```

输出目录：

```text
results/glm5_phase633_multi_writer_prefix_readout_field_closure
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase633_multi_writer_prefix_readout_field_closure.py \
  tests/gpt5/phase633_multi_writer_prefix_readout_field_closure_summary.py

python tests/gpt5/phase633_multi_writer_prefix_readout_field_closure.py qwen3 \
  --smoke \
  --include-nontarget \
  --output-dir results/glm5_phase633_multi_writer_prefix_readout_field_closure \
  --hard-exit-after-model

python tests/gpt5/phase633_multi_writer_prefix_readout_field_closure.py qwen3 \
  --confirm \
  --output-dir results/glm5_phase633_multi_writer_prefix_readout_field_closure \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase633_multi_writer_prefix_readout_field_closure.py glm4 \
  --confirm \
  --output-dir results/glm5_phase633_multi_writer_prefix_readout_field_closure \
  --hard-exit-after-model

python tests/gpt5/phase633_multi_writer_prefix_readout_field_closure.py deepseek7b \
  --confirm \
  --output-dir results/glm5_phase633_multi_writer_prefix_readout_field_closure \
  --hard-exit-after-model

python tests/gpt5/phase633_multi_writer_prefix_readout_field_closure_summary.py
```

### 测试原理

本阶段使用 Phase632 已经得到的 natural prefix writer scan_rank。

为避免重复残差观测点，把下列节点视为同一 residual observation：

```text
Lx_layer_out ≈ L(x+1)_layer_input
```

然后从去重后的 writer rank 构建累积集合：

```text
top1
top2
top4
top8
top12
```

每个集合测试：

```text
restore:
  在 base prompt 的 prompt_last 多节点注入 repair 状态。

restore_semantic:
  restore + semantic cumulative patch。

random_semantic:
  同范数随机多节点 delta + semantic cumulative。

reverse_semantic:
  反向多节点 delta + semantic cumulative。

remove_from_repair:
  在 repair prompt 中把这些节点替换回 base 状态。
```

核心判据：

```text
如果 prompt_last 多 writer field 是完整 prefix gate，
则 topK_restore_semantic 应随 K 增加接近 Phase631 人工 readout 上界。

如果 topK_restore_semantic 仍停在 Phase632 单 writer 水平，
则 prompt_last residual writer field 不是完整路径。
```

### 关键结果

#### Qwen3

目标样本：

```text
rows = 17 / raw_cases = 256
candidate_nodes =
  L34_layer_out, L33_layer_out, L32_layer_out, L35_layer_out,
  L34_attn_out, L32_attn_out, L30_layer_out, L31_layer_out,
  L28_layer_out, L29_layer_out, L33_attn_out, L25_layer_out
```

结果：

```text
base:
  tok0 = 10/17
  exact = 1/17
  wrong_exact = 9/17
  mean_prefix_margin = 0.213

semantic_cumulative:
  tok0 = 10/17
  exact = 10/17
  mean_prefix_margin = 0.213

repair_prompt:
  tok0 = 14/17
  exact = 11/17
  wrong_exact = 3/17
  mean_prefix_margin = 1.110

top1_restore_semantic:
  tok0 = 14/17
  exact = 14/17
  mean_prefix_margin = 1.044

top4_restore_semantic:
  tok0 = 14/17
  exact = 14/17
  mean_prefix_margin = 1.110

top8_restore_semantic:
  tok0 = 14/17
  exact = 14/17
  mean_prefix_margin = 1.110

top12_restore_semantic:
  tok0 = 14/17
  exact = 14/17
  mean_prefix_margin = 1.110

top12_reverse_semantic:
  tok0 = 6/17
  exact = 6/17
  mean_prefix_margin = -0.537
```

Qwen3 结论：

```text
多 writer 累积没有超过 top1/top4 的 14/17。
prompt_last writer field 只能解释一部分格式门，不能达到 Phase631 人工 readout 的 17/17。
```

#### GLM4

目标样本：

```text
rows = 31 / raw_cases = 256
candidate_nodes =
  L38_layer_out, L37_layer_out, L36_layer_out, L39_layer_out,
  L35_layer_out, L34_layer_out, L33_layer_out, L32_layer_out,
  L32_attn_out, L38_mlp_out, L31_layer_out, L29_layer_out
```

结果：

```text
base:
  tok0 = 11/31
  exact = 2/31
  wrong_exact = 9/31
  mean_prefix_margin = -0.226

semantic_cumulative:
  tok0 = 11/31
  exact = 11/31
  mean_prefix_margin = -0.226

repair_prompt:
  tok0 = 29/31
  exact = 28/31
  wrong_exact = 1/31
  mean_prefix_margin = 1.710

top1_restore_semantic:
  tok0 = 29/31
  exact = 29/31
  mean_prefix_margin = 1.712

top4_restore_semantic:
  tok0 = 29/31
  exact = 29/31
  mean_prefix_margin = 1.710

top8_restore_semantic:
  tok0 = 29/31
  exact = 29/31
  mean_prefix_margin = 1.710

top12_restore_semantic:
  tok0 = 29/31
  exact = 29/31
  mean_prefix_margin = 1.710

top12_reverse_semantic:
  tok0 = 0/31
  exact = 0/31
  mean_prefix_margin = -2.064
```

GLM4 结论：

```text
GLM4 单 writer 已基本达到 prompt_last writer field 上限。
多 writer 累积没有超过 29/31。
remove_from_repair 会降回 11/31，说明这些 writer 对修复路径必要。
但剩余 2/31 缺口不在去重后的 prompt_last writer 累积中。
```

#### DS7B

目标样本：

```text
rows = 82 / raw_cases = 256
candidate_nodes =
  L26_layer_out, L27_layer_out, L25_layer_out, L24_layer_out,
  L26_attn_out, L23_layer_out, L26_mlp_out, L25_attn_out,
  L24_attn_out, L22_layer_out, L22_attn_out, L24_mlp_out
```

结果：

```text
base:
  tok0 = 0/82
  exact = 0/82
  mean_prefix_margin = -6.356

semantic_cumulative:
  tok0 = 0/82
  exact = 0/82
  mean_prefix_margin = -6.356

repair_prompt:
  tok0 = 20/82
  exact = 20/82
  mean_prefix_margin = -1.699

top1_restore_semantic:
  tok0 = 21/82
  exact = 21/82
  mean_prefix_margin = -1.662

top2_restore_semantic:
  tok0 = 20/82
  exact = 20/82
  mean_prefix_margin = -1.699

top4_restore_semantic:
  tok0 = 20/82
  exact = 20/82
  mean_prefix_margin = -1.699

top8_restore_semantic:
  tok0 = 20/82
  exact = 20/82
  mean_prefix_margin = -1.699

top12_restore_semantic:
  tok0 = 20/82
  exact = 20/82
  mean_prefix_margin = -1.699

top12_random_semantic:
  tok0 = 0/82
  exact = 0/82
  mean_prefix_margin = -5.989

top12_reverse_semantic:
  tok0 = 0/82
  exact = 0/82
  mean_prefix_margin = -10.592

top12_remove_from_repair:
  tok0 = 0/82
  exact = 0/82
  mean_prefix_margin = -6.356
```

DS7B 结论：

```text
多 writer cumulative restore 没有提升，反而 top2/top4/top8/top12 回到 repair_prompt 的 20/82。
top1 的 21/82 是最高值。
prompt_last residual writer field 不是 DS7B prefix gate 的完整路径。
```

### 当前客观拼图

Phase633 是一个关键负结果：

```text
Phase632:
  单 writer 不足。

Phase633:
  去重后的 prompt_last 多 writer 累积仍不足。
```

这排除了一个自然推测：

```text
只要把 prompt_last 后层 writer chain 累积起来，
就能复现 Phase631 人工 readout 的 81/82。
```

实际结果：

```text
Qwen3:
  prompt_last writer field 上限约 14/17。

GLM4:
  prompt_last writer field 上限约 29/31。

DS7B:
  prompt_last writer field 上限约 20/82 到 21/82。
```

因此，DS7B 的 Phase631 人工 readout 闭合并不是 prompt_last 多 writer restore 可以自然复现的。

### 理论进展

当前机制应进一步拆分：

```text
1. natural prompt_last writer field:
   可解释 repair_prompt 中一部分 prefix margin 改善。

2. final readout direction:
   人工足以打开 token0，但自然来源未闭合。

3. missing format/source field:
   可能位于多位置 source span、格式标签、answer_label、question span 或 decoder prior。
```

这对“相对编码 + 复用差分机制”的含义也更具体：

```text
差分不是只在一个位置累积；
复用骨架可能跨多个位置、多种状态接口共同形成输出竞争条件。
```

也就是说，真正图谱不能只画 prompt_last：

```text
完整 prefix gate 图谱 =
  prompt_last residual field
  + multi-source format field
  + token-level unembedding competition
  + decode prior / formatting prior
```

### 硬伤和边界

1. **Phase633 依赖 Phase632 的 scan_rank**

本轮没有重新扫描所有节点，而是用 Phase632 的 top writer rank 作为候选。
这提高效率，但如果 Phase632 漏掉多位置 source 节点，本轮不会发现。

2. **只测试 prompt_last 多 writer**

本轮没有测试 answer_label、question_all、relation_tail 等多位置 source span 的累积场。
DS7B 的失败很可能说明缺口在这些位置。

3. **累积 patch 可能互相覆盖轨迹**

多节点同时 restore 不一定等于自然动态生成，因为后层状态可能依赖前层状态。
这可能解释为什么 top12 没超过 top1。

4. **没有测试非目标副作用**

本阶段仍主要面向目标样本。
如果之后找到强多位置 field，必须做 non-target side-effect audit。

5. **仍未解释 DS7B 81/82 人工上界的自然来源**

Phase633 缩小了搜索空间，但没有闭合自然机制。

### 下一步任务

Phase634 应执行：

```text
Multi-Position Format Source Field Closure
```

核心目标：

```text
不要再只在 prompt_last 累积 writer。
改为测试多位置 source/format field 是否能提供 DS7B 缺失的 prefix gate。
```

测试设计：

```text
1. 使用 Phase630 的 source groups:
   prompt_last
   answer_label
   question_mark_answer
   relation_tail
   question_subject
   question_all

2. 不再单组扫描，而是做 multi-group cumulative patch:
   answer_label + prompt_last
   question_mark_answer + prompt_last
   relation_tail + answer_label + prompt_last
   question_all + answer_label + prompt_last

3. 每组测试:
   source_cumulative_only
   source_cumulative_semantic
   random_same_norm_semantic
   reverse/source_wrong_semantic

4. 仍使用自然生成指标:
   tok0 hit
   exact generation
   wrong_exact
   mean_prefix_margin
   top competitor distribution

5. DS7B 是重点:
   如果 multi-source field + semantic 接近 81/82，
   则 prefix gate 是多位置格式源场。

   如果仍停在 20/82，
   则瓶颈进一步后移到 final readout / decoding prior 的自然来源。
```

阶段性判断：

```text
prompt_last 单点不够；
prompt_last 多 writer 也不够；
下一步必须进入多位置 source/format field。
```

## Phase 634: Multi-Position Format Source Field Closure 多位置格式源场闭合 [2026-06-25 16:48]

### 触发背景

用户上传的 Phase633 分析判断基本正确。Phase633 的核心价值是一个关键负结果：

```text
prompt_last 单点 writer 不够；
prompt_last 多 writer 累积也不够；
因此必须测试 multi-position source / format field。
```

本阶段直接执行这个下一步，不再继续在 prompt_last 位置叠加 writer，而是把 Phase630 中已经定义过的 source groups 引入到同一轮跨模型确认测试。

### 生成脚本

```text
tests/gpt5/phase634_multi_position_format_source_field_closure.py
tests/gpt5/phase634_multi_position_format_source_field_closure_summary.py
```

输出目录：

```text
results/glm5_phase634_multi_position_format_source_field_closure/
```

核心输出：

```text
results/glm5_phase634_multi_position_format_source_field_closure/phase634_cross_model_summary.md
```

### 执行命令

语法检查：

```bash
python -m py_compile tests/gpt5/phase634_multi_position_format_source_field_closure.py tests/gpt5/phase634_multi_position_format_source_field_closure_summary.py
```

Qwen3 smoke：

```bash
python tests/gpt5/phase634_multi_position_format_source_field_closure.py qwen3 --smoke --include-nontarget --output-dir results/glm5_phase634_multi_position_format_source_field_closure --hard-exit-after-model
```

Qwen3 confirm：

```bash
python tests/gpt5/phase634_multi_position_format_source_field_closure.py qwen3 --confirm --output-dir results/glm5_phase634_multi_position_format_source_field_closure --hard-exit-after-model
```

GLM4 confirm：

```bash
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase634_multi_position_format_source_field_closure.py glm4 --confirm --output-dir results/glm5_phase634_multi_position_format_source_field_closure --hard-exit-after-model
```

DS7B confirm：

```bash
python tests/gpt5/phase634_multi_position_format_source_field_closure.py deepseek7b --confirm --output-dir results/glm5_phase634_multi_position_format_source_field_closure --hard-exit-after-model
```

汇总：

```bash
python tests/gpt5/phase634_multi_position_format_source_field_closure_summary.py
```

### 测试原理

Phase631 已经证明，人工 token0 readout direction 可以把 DS7B 从 0/82 推到 81/82。Phase632 和 Phase633 证明，自然 prompt_last writer 无论单点还是多 writer 累积，都只能到约 20/82 到 21/82。

本阶段测试的是：

```text
如果 DS7B 缺失的 prefix gate 不是 prompt_last 局部状态，
它是否来自多个 source / format 位置的共同作用？
```

测试的 source groups：

```text
prompt_last
answer_label
question_mark_answer
relation_tail
question_subject
question_all
```

测试的组合：

```text
single_prompt_last
single_answer_label
single_question_mark_answer
single_relation_tail
single_question_all
answer_prompt
qma_prompt
relation_answer_prompt
question_all_answer_prompt
answer_qma_relation_prompt
all6
```

每组都包含：

```text
restore
restore_semantic
random
random_semantic
reverse
reverse_semantic
remove_from_repair
```

核心判据仍然是自然输出层面的：

```text
tok0 hit
exact generation
wrong_exact
mean_prefix_margin
top0_text distribution
```

### 客观结果

#### Qwen3

样本：

```text
rows = 17
raw_cases = 256
```

关键结果：

```text
base = 10/17 tok0, 1/17 exact
semantic_cumulative = 10/17 tok0, 10/17 exact
repair_prompt = 14/17 tok0, 11/17 exact

single_question_all_restore_semantic = 14/17 tok0, 14/17 exact
question_all_answer_prompt_restore_semantic = 14/17 tok0, 14/17 exact
all6_restore_semantic = 14/17 tok0, 14/17 exact

single_prompt_last_restore_semantic = 13/17 tok0, 13/17 exact
single_answer_label_restore_semantic = 13/17 tok0, 13/17 exact
single_relation_tail_restore_semantic = 13/17 tok0, 13/17 exact
```

Qwen3 的多位置组合没有超过 single_question_all，也没有接近 Phase631 的人工 readout+semantic 17/17 上界。

#### GLM4

样本：

```text
rows = 31
raw_cases = 256
```

关键结果：

```text
base = 11/31 tok0, 2/31 exact
semantic_cumulative = 11/31 tok0, 11/31 exact
repair_prompt = 29/31 tok0, 28/31 exact

single_answer_label_restore_semantic = 30/31 tok0, 30/31 exact
single_question_mark_answer_restore_semantic = 30/31 tok0, 30/31 exact
single_relation_tail_restore_semantic = 30/31 tok0, 30/31 exact
single_prompt_last_restore_semantic = 30/31 tok0, 30/31 exact

answer_prompt_restore_semantic = 30/31 tok0, 30/31 exact
relation_answer_prompt_restore_semantic = 30/31 tok0, 30/31 exact
all6_restore_semantic = 29/31 tok0, 29/31 exact
```

GLM4 已经可以被多个单源位置修复到 30/31，但多位置 all6 没有超过单源，反而略低到 29/31。

#### DS7B

样本：

```text
rows = 82
raw_cases = 256
```

关键结果：

```text
base = 0/82 tok0, 0/82 exact
semantic_cumulative = 0/82 tok0, 0/82 exact
repair_prompt = 20/82 tok0, 20/82 exact

single_prompt_last_restore_semantic = 21/82 tok0, 21/82 exact
single_answer_label_restore_semantic = 21/82 tok0, 21/82 exact
single_question_mark_answer_restore_semantic = 21/82 tok0, 21/82 exact
single_relation_tail_restore_semantic = 21/82 tok0, 21/82 exact
single_question_all_restore_semantic = 20/82 tok0, 20/82 exact

answer_prompt_restore_semantic = 21/82 tok0, 21/82 exact
qma_prompt_restore_semantic = 21/82 tok0, 21/82 exact
relation_answer_prompt_restore_semantic = 21/82 tok0, 21/82 exact
question_all_answer_prompt_restore_semantic = 21/82 tok0, 21/82 exact
answer_qma_relation_prompt_restore_semantic = 21/82 tok0, 21/82 exact
all6_restore_semantic = 21/82 tok0, 21/82 exact

all6_random_semantic = 0/82
all6_reverse_semantic = 0/82
all6_remove_from_repair = 0/82
```

DS7B 是本阶段最关键的证据。所有多位置组合都没有超过 21/82，和单 prompt_last 基本相同，远低于 Phase631 的 readout_scale0.25+semantic 上界 81/82。

### 结论

本阶段是一个关键负结果。

它排除了一个强假设：

```text
DS7B 缺失的 prefix gate 来自 Phase630 source groups 的多位置 source / format field 累积。
```

实验证据不支持这个假设。

跨模型比较显示：

```text
Qwen3:
  多位置组合没有超过单个强源位置。

GLM4:
  多个单源位置都可以恢复到 30/31，
  但 all6 没有更强，说明多位置累积不是简单加法。

DS7B:
  all6_restore_semantic = 21/82，
  与单 prompt_last_restore_semantic = 21/82 相同。
```

因此，当前链条变成：

```text
Phase631:
  人工 token0 readout direction 可以几乎闭合 DS7B。

Phase632:
  自然 single writer 只能恢复 21/82。

Phase633:
  prompt_last multi-writer 仍不能超过 20/82 到 21/82。

Phase634:
  multi-position source / format field 仍不能超过 21/82。
```

这说明 DS7B 的关键缺口不在已测的自然 source state restore 空间内。

### 最严格的问题和硬伤

1. 本阶段没有证明 prefix gate 的真实来源，只是进一步排除了一个候选空间。

2. 多位置 patch 使用的是固定 layer_map，仍可能漏掉真正的跨层动态轨迹。

3. 多位置 restore 是静态替换，不等价于自然生成中的连续状态演化。

4. all6 不增益可能来自 patch 互相覆盖，也可能说明这些位置只是同一读出状态的重复投影。

5. 当前仍未解释 DS7B 为什么人工 readout direction 能达到 81/82，而自然 source restore 只能达到 21/82。

### 理论进展

本阶段进一步支持“复用差分机制”判断：

```text
语言框架不是由某个固定位置保存完整控制量；
多个位置可能共享部分状态，
但这些状态并不能直接累积成最终 token0 读出门。
```

也就是说，当前更像：

```text
source / format positions 提供局部可读差分；
final token0 readout gate 需要另一个后端机制把差分变成可输出词元竞争优势。
```

这将瓶颈从：

```text
source field 在哪里？
```

推进到：

```text
source field 如何被汇聚成 final readout / decoding prior？
```

### 下一步任务

Phase635 应执行：

```text
Final Readout Projection Bridge Audit
```

目标不是继续扩大 source 位置，而是直接审计：

```text
Phase631 的人工 readout direction 到底对应 final hidden / final norm / lm_head 读出链条中的哪一级。
```

建议测试：

```text
1. 对 DS7B 重点执行 final hidden pre_norm / post_norm / lm_head input 的分层替换。

2. 比较三类状态：
   semantic_cumulative
   source_field_restore_semantic
   readout_direction_semantic

3. 测量每类状态在 final norm 前后的变化：
   prefix margin
   correct token rank
   top competitor token
   norm scale
   cosine to readout direction

4. 测试是否存在 final norm projection collapse：
   即 hidden 中已经有语义，但经过 final norm / lm_head 后被格式竞争压制。

5. 对照 Qwen3、GLM4、DS7B：
   如果 Qwen3/GLM4 能在 final bridge 闭合而 DS7B 不能，
   则 DS7B 的瓶颈是 final readout bridge；
   如果三者一致，
   则应转向更高层的 decoding prior / output protocol 图谱。
```

阶段性判断：

```text
prompt_last 单点不够；
prompt_last 多 writer 不够；
multi-position source / format field 也不够；
下一步必须进入 final readout bridge。
```

## Phase 635: Final Readout Projection Bridge Audit 最终读出投影桥审计 [2026-06-25 17:10]

### 触发背景

用户上传的 Phase634 分析基本正确。Phase634 已经排除一个强假设：

```text
DS7B 缺失的 token0 prefix gate
不是由已扫描的 multi-position source / format field 简单累积形成的。
```

当前链条是：

```text
Phase631:
  artificial readout direction + semantic cumulative = DS7B 81/82

Phase632:
  natural single writer restore + semantic = DS7B 21/82

Phase633:
  prompt_last multi-writer restore + semantic = DS7B 20/82 到 21/82

Phase634:
  multi-position source / format field + semantic = DS7B 21/82
```

因此本阶段不再扩大 source 位置，而是直接审计：

```text
final_norm input -> final_norm output -> lm_head token competition
```

### 生成脚本

```text
tests/gpt5/phase635_final_readout_projection_bridge_audit.py
tests/gpt5/phase635_final_readout_projection_bridge_audit_summary.py
```

输出目录：

```text
results/glm5_phase635_final_readout_projection_bridge_audit/
```

核心输出：

```text
results/glm5_phase635_final_readout_projection_bridge_audit/phase635_cross_model_summary.md
```

### 执行命令

语法检查：

```bash
python -m py_compile tests/gpt5/phase635_final_readout_projection_bridge_audit.py tests/gpt5/phase635_final_readout_projection_bridge_audit_summary.py
```

Qwen3 smoke：

```bash
python tests/gpt5/phase635_final_readout_projection_bridge_audit.py qwen3 --smoke --include-nontarget --output-dir results/glm5_phase635_final_readout_projection_bridge_audit --hard-exit-after-model
```

Qwen3 confirm：

```bash
python tests/gpt5/phase635_final_readout_projection_bridge_audit.py qwen3 --confirm --output-dir results/glm5_phase635_final_readout_projection_bridge_audit --hard-exit-after-model
```

GLM4 confirm：

```bash
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase635_final_readout_projection_bridge_audit.py glm4 --confirm --output-dir results/glm5_phase635_final_readout_projection_bridge_audit --hard-exit-after-model
```

DS7B confirm：

```bash
python tests/gpt5/phase635_final_readout_projection_bridge_audit.py deepseek7b --confirm --output-dir results/glm5_phase635_final_readout_projection_bridge_audit --hard-exit-after-model
```

汇总：

```bash
python tests/gpt5/phase635_final_readout_projection_bridge_audit_summary.py
```

### 测试原理

本阶段比较三类路径：

```text
1. natural state:
   base
   repair_prompt
   source_all6

2. final bridge restore:
   final_input_repair
   final_output_repair
   final_output_source

3. artificial upper bound:
   readout_delta
   readout_delta_semantic
```

其中 final bridge restore 分别测试：

```text
final_input_repair:
  把 base 的 final_norm input 替换为 repair 的 final_norm input。

final_output_repair:
  把 base 的 final_norm output 直接替换为 repair 的 final_norm output。

final_output_source:
  把 base 的 final_norm output 替换为 source_all6 后得到的 final_norm output。
```

如果问题是 final norm 把已有正确状态冲掉，那么：

```text
final_input_repair 和 final_output_repair 应明显分离。
```

如果问题是 final output 已经正确但 lm_head 不接受，那么：

```text
final_output_repair 应接近 artificial readout_delta。
```

如果两者都不成立，则说明：

```text
自然 final state 本身没有形成足够的 readout-aligned direction。
```

### 客观结果

#### Qwen3

样本：

```text
rows = 17
raw_cases = 256
```

关键结果：

```text
base = 10/17 tok0, 1/17 exact, mean_margin = 0.213
semantic_cumulative = 10/17 tok0, 10/17 exact

repair_prompt = 14/17 tok0, 11/17 exact, out_proj = 0.581
source_all6_semantic = 14/17 tok0, 14/17 exact, out_proj = 0.581
final_input_repair_semantic = 14/17 tok0, 14/17 exact, out_proj = 0.581
final_output_repair_semantic = 14/17 tok0, 14/17 exact, out_proj = 0.581
final_output_source_semantic = 14/17 tok0, 14/17 exact, out_proj = 0.581

readout_delta_semantic = 17/17 tok0, 17/17 exact, out_proj = 39.638
```

Qwen3 的 final restore 路径没有超过 source_all6，人工 readout 仍然明显更强。

#### GLM4

样本：

```text
rows = 31
raw_cases = 256
```

关键结果：

```text
base = 11/31 tok0, 2/31 exact, mean_margin = -0.226
semantic_cumulative = 11/31 tok0, 11/31 exact

repair_prompt = 29/31 tok0, 28/31 exact, out_proj = 2.148
source_all6_semantic = 29/31 tok0, 29/31 exact, out_proj = 2.148
final_input_repair_semantic = 29/31 tok0, 29/31 exact, out_proj = 2.148
final_output_repair_semantic = 29/31 tok0, 29/31 exact, out_proj = 2.148
final_output_source_semantic = 29/31 tok0, 29/31 exact, out_proj = 2.148

readout_delta_semantic = 31/31 tok0, 31/31 exact, out_proj = 46.475
```

GLM4 的 natural final state 已经能闭合大部分样本，但仍低于人工 readout 上界。

#### DS7B

样本：

```text
rows = 82
raw_cases = 256
```

关键结果：

```text
base = 0/82 tok0, 0/82 exact, mean_rank = 92.8, mean_margin = -6.356
semantic_cumulative = 0/82 tok0, 0/82 exact

repair_prompt = 20/82 tok0, 20/82 exact, mean_rank = 9.4, mean_margin = -1.699, out_proj = 3.618
source_all6_semantic = 21/82 tok0, 21/82 exact, mean_rank = 9.9, mean_margin = -1.755, out_proj = 3.578

final_input_repair_semantic = 20/82 tok0, 20/82 exact, mean_rank = 9.4, mean_margin = -1.699, out_proj = 3.618
final_output_repair_semantic = 20/82 tok0, 20/82 exact, mean_rank = 9.4, mean_margin = -1.699, out_proj = 3.618
final_output_source_semantic = 21/82 tok0, 21/82 exact, mean_rank = 9.9, mean_margin = -1.755, out_proj = 3.578

readout_delta_semantic = 82/82 tok0, 81/82 exact, mean_rank = 1.0, mean_margin = 55.759, out_proj = 48.258
```

DS7B 是核心证据：

```text
final_input_repair_semantic = 20/82
final_output_repair_semantic = 20/82
final_output_source_semantic = 21/82
readout_delta_semantic = 81/82
```

final_norm input restore 和 final_norm output restore 完全没有拉近人工 readout 上界。

### 结论

本阶段是一个关键定位结果。

它说明 DS7B 的问题不是：

```text
repair/source 状态中已经含有足够正确方向，
但 final norm 把它冲掉。
```

因为：

```text
final_input_repair_semantic = final_output_repair_semantic = repair_prompt = 20/82
```

它也不是：

```text
source_all6 已经形成足够 final output，
只是自然路径没有把它送到 lm_head。
```

因为：

```text
final_output_source_semantic = source_all6_semantic = 21/82
```

更严格的结论是：

```text
自然 final state 只产生了很弱的 readout-aligned projection。
DS7B 人工 readout 上界来自直接沿 lm_head 竞争方向注入的大幅方向量。
自然 repair/source/final restore 的方向量远远不够。
```

用本阶段数值表示：

```text
DS7B:
  repair/source final output projection ≈ 3.6
  artificial readout projection ≈ 48.3
```

这不是 final norm 层面的崩溃，而是：

```text
自然轨迹没有产生足够强的 prefix readout vector。
```

### 理论进展

Phase635 把瓶颈进一步从：

```text
final readout bridge 是否冲掉已有状态？
```

推进为：

```text
自然轨迹为什么没有生成足够强的 readout-aligned vector？
```

当前更合适的机制表述是：

```text
source / format field 提供弱投影；
final norm 只是保留这个弱投影；
lm_head 只按词表方向读出；
真正缺失的是上游把格式意图放大成强 readout vector 的机制。
```

这意味着：

```text
读出桥不是一个简单位置；
它可能是 readout vector builder，
也就是一个把格式/任务状态转译成 lm_head 方向的生成器。
```

### 最严格的问题和硬伤

1. 本阶段没有找到 readout vector builder，只确认 final norm/input/output restore 不是闭合点。

2. artificial readout_delta 的尺度很大，可能不是自然可达状态，只能作为充分上界，不能直接当自然机制。

3. 当前只测了 token0 prefix gate，尚未把 token0、token1、token2 三门统一成完整读出图谱。

4. final_output projection 是相对 prefix-vs-competitor 的单方向指标，可能漏掉多竞争词元结构。

5. DS7B 中 prefix rank 从 92.8 改善到约 9.4，但仍不能到 1，说明存在一组强竞争词元没有被拆开。

### 下一步任务

Phase636 应执行：

```text
Prefix Competitor Ladder and Readout Vector Builder Audit
```

核心问题不再是：

```text
哪个位置 restore 能成功？
```

而是：

```text
DS7B token0 从 rank 92.8 到 rank 9.4 之后，
还剩哪些 competitor tokens 压住 correct prefix？
这些 competitor 是格式符号、解释性文本、换行、冒号，还是旧答案前缀？
```

建议测试：

```text
1. 对 DS7B 的 base / repair / source_all6 / final_output_repair / readout_delta
   记录 top20 competitor ladder。

2. 把 competitor tokens 分组：
   punctuation / newline / explanation / value_prefix / category_prefix / other。

3. 对每组 competitor 建立 group margin：
   correct_prefix_logit - max(group_logits)

4. 测试人工 readout_delta 究竟是全面抬高 correct prefix，
   还是主要压过某一类 competitor。

5. 对 Qwen3、GLM4、DS7B 比较：
   哪些模型的 competitor ladder 已经被自然 repair/source 消除，
   哪些模型仍停在 explanation / punctuation prior。
```

阶段性判断：

```text
source field 不够；
final norm restore 不够；
问题已经转向 output competitor ladder 与 readout vector builder。
```

## Phase 636: Prefix Competitor Ladder and Readout Vector Builder Audit 前缀竞争阶梯与读出向量生成器审计 [2026-06-25 17:41]

### 触发背景

用户上传的 Phase635 分析基本正确。Phase635 的关键定位是：

```text
DS7B 的问题不是 final norm 把已有正确方向冲掉；
也不是 source_all6 已经形成足够 final output 但没有送入 lm_head；
而是自然轨迹没有生成足够强的 prefix readout vector。
```

Phase635 中 DS7B 的核心数据是：

```text
repair_prompt out_proj ≈ 3.618
final_output_repair_semantic out_proj ≈ 3.618
readout_delta_semantic out_proj ≈ 48.258

repair_prompt tok0 = 20/82
final_output_repair_semantic tok0 = 20/82
readout_delta_semantic tok0 = 82/82
```

因此 Phase636 不再继续问：

```text
哪个位置 restore 能成功？
```

而是问：

```text
正确 prefix token 已经从 rank 92.8 拉到 rank 9.4 后，
到底还被哪些 competitor tokens 压住？
```

### 生成脚本

```text
tests/gpt5/phase636_prefix_competitor_ladder_audit.py
tests/gpt5/phase636_prefix_competitor_ladder_audit_summary.py
```

输出目录：

```text
results/glm5_phase636_prefix_competitor_ladder_audit/
```

核心输出：

```text
results/glm5_phase636_prefix_competitor_ladder_audit/phase636_cross_model_summary.md
```

### 执行命令

语法检查：

```bash
python -m py_compile tests/gpt5/phase636_prefix_competitor_ladder_audit.py tests/gpt5/phase636_prefix_competitor_ladder_audit_summary.py
```

Qwen3 smoke：

```bash
python tests/gpt5/phase636_prefix_competitor_ladder_audit.py qwen3 --smoke --include-nontarget --output-dir results/glm5_phase636_prefix_competitor_ladder_audit --hard-exit-after-model
```

Qwen3 confirm：

```bash
python tests/gpt5/phase636_prefix_competitor_ladder_audit.py qwen3 --confirm --output-dir results/glm5_phase636_prefix_competitor_ladder_audit --hard-exit-after-model
```

GLM4 confirm：

```bash
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase636_prefix_competitor_ladder_audit.py glm4 --confirm --output-dir results/glm5_phase636_prefix_competitor_ladder_audit --hard-exit-after-model
```

DS7B confirm：

```bash
python tests/gpt5/phase636_prefix_competitor_ladder_audit.py deepseek7b --confirm --output-dir results/glm5_phase636_prefix_competitor_ladder_audit --hard-exit-after-model
```

汇总：

```bash
python tests/gpt5/phase636_prefix_competitor_ladder_audit_summary.py
```

### 测试原理

本阶段不做长生成，只测 token0 logits，以减少额外变量。

测试模式：

```text
base
repair_prompt
source_all6
final_output_repair
final_output_source
readout_delta
```

每个样本记录 top20 competitor ladder，并把 token 分组为：

```text
correct_prefix
newline
punctuation
explanation
old_wrong_prefix
value_prefix
word
number
space
symbol
other
```

每组计算：

```text
prefix_rank
top0_category
top0_text
prefix_minus_group_max
winner_rate
mean_best_rank
```

核心判据：

```text
如果自然 repair/source 已经解决格式先验，
则 top0_category 应从 newline / explanation 转为 correct_prefix。

如果仍失败，
则剩余 winner category 会显示真实压制来源。
```

### 客观结果

#### Qwen3

样本：

```text
rows = 17
raw_cases = 256
top_k = 20
```

mode ladder：

```text
base:
  tok0 = 11/17
  mean_rank = 1.9
  top0_category = correct_prefix:11, newline:5, space:1

repair_prompt:
  tok0 = 14/17
  mean_rank = 1.2
  top0_category = correct_prefix:14, space:3

source_all6:
  tok0 = 14/17
  mean_rank = 1.2
  top0_category = correct_prefix:14, space:3

final_output_repair:
  tok0 = 14/17
  mean_rank = 1.2
  top0_category = correct_prefix:14, space:3

readout_delta:
  tok0 = 17/17
  mean_rank = 1.0
  top0_category = correct_prefix:17
```

Qwen3 的剩余竞争主要是 space 类，不是大规模 newline 类。

#### GLM4

样本：

```text
rows = 31
raw_cases = 256
top_k = 20
```

mode ladder：

```text
base:
  tok0 = 11/31
  mean_rank = 2.7
  top0_category = word:17, correct_prefix:11, explanation:3

repair_prompt:
  tok0 = 29/31
  mean_rank = 1.1
  top0_category = correct_prefix:29, word:2

source_all6:
  tok0 = 29/31
  mean_rank = 1.1
  top0_category = correct_prefix:29, word:2

final_output_repair:
  tok0 = 29/31
  mean_rank = 1.1
  top0_category = correct_prefix:29, word:2

readout_delta:
  tok0 = 31/31
  mean_rank = 1.0
  top0_category = correct_prefix:31
```

GLM4 的自然 repair/source/final restore 已经清除了大部分竞争，只剩 2/31 的 word 类竞争。

#### DS7B

样本：

```text
rows = 82
raw_cases = 256
top_k = 20
```

mode ladder：

```text
base:
  tok0 = 0/82
  mean_rank = 92.8
  top0_category = newline:81, word:1
  top0_text = " ?\n\n":81, " c":1

repair_prompt:
  tok0 = 20/82
  mean_rank = 9.4
  top0_category = newline:57, correct_prefix:20, word:3, space:1, explanation:1
  top0_text = " ?\n\n":57, " v":20

source_all6:
  tok0 = 20/82
  mean_rank = 9.9
  top0_category = newline:57, correct_prefix:20, word:2, space:2, explanation:1
  top0_text = " ?\n\n":57, " v":20

final_output_repair:
  tok0 = 20/82
  mean_rank = 9.4
  top0_category = newline:57, correct_prefix:20, word:3, space:1, explanation:1
  top0_text = " ?\n\n":57, " v":20

final_output_source:
  tok0 = 20/82
  mean_rank = 9.9
  top0_category = newline:57, correct_prefix:20, word:2, space:2, explanation:1

readout_delta:
  tok0 = 82/82
  mean_rank = 1.0
  top0_category = correct_prefix:82
```

DS7B 的 category margins 最关键：

```text
base newline:
  winner_rate = 0.99
  prefix_minus_group_max = -6.354

repair_prompt newline:
  winner_rate = 0.70
  prefix_minus_group_max = -1.704

source_all6 newline:
  winner_rate = 0.70
  prefix_minus_group_max = -1.765

final_output_repair newline:
  winner_rate = 0.70
  prefix_minus_group_max = -1.704

readout_delta newline:
  winner_rate = 0.00
  prefix_minus_group_max = 22.896
```

这说明 DS7B 不是被任意 competitor 压住，而是被非常明确的 newline prior 压住。

### 结论

Phase636 是一个关键定位结果。

它把 Phase635 中的模糊问题：

```text
自然 readout vector 不够强
```

进一步压缩为：

```text
自然 readout vector 没有足够压制 newline / format continuation prior。
```

最关键的是 DS7B：

```text
base:
  81/82 被 newline 类 token 占据 top0。

repair_prompt / final_output_repair:
  newline 仍然占据 57/82 top0。

readout_delta:
  correct_prefix 占据 82/82 top0。
```

因此，DS7B 的 token0 prefix gate 不是一般的语义值门问题，也不是 final norm 问题，而是：

```text
format continuation prior suppression 不足。
```

也就是说，模型自然轨迹虽然把正确 prefix 从 rank 92.8 拉到 9.4，但没有关闭“继续输出问题/换行/解释”的格式先验。

### 理论进展

当前三模型呈现清晰差异：

```text
Qwen3:
  剩余主要是 space 竞争。

GLM4:
  剩余主要是少量 word 竞争。

DS7B:
  剩余主要是 newline 竞争。
```

这说明不同模型的 token0 prefix gate 不是同构的单机制，而是同一个功能门下的不同竞争结构：

```text
GLM4:
  repair/source 可以基本压制 competitor ladder。

Qwen3:
  repair/source 可以压制 newline，但仍有 space。

DS7B:
  repair/source 只能削弱 newline，不能关闭 newline prior。
```

因此，“读出向量生成器”更具体地应分成两部分：

```text
1. correct prefix promotion
   抬高正确 prefix。

2. format continuation suppression
   压制继续提问、换行、解释、符号等格式续写先验。
```

Phase631 的人工 readout_delta 可能同时完成了两件事：

```text
大幅抬高 " v"；
相对压过 newline / punctuation / explanation / space。
```

但自然 repair/source 只完成了弱抬高，没有完成足够强的 newline suppression。

### 最严格的问题和硬伤

1. 本阶段只做 token0 logits，不做完整自然生成，因此它定位竞争结构，但不直接证明生成闭合。

2. category 分类是规则化的基础分类，可能把部分模型特有 token 分到 word / other，后续需要人工抽查 top tokens。

3. readout_delta 是人工强方向，它压过 newline prior，不代表自然网络一定存在同尺度机制。

4. 当前仍主要使用 target-filtered 样本。全局图谱阶段必须加入 non-target side-effect。

5. DS7B newline prior 的来源仍未定位。它可能来自 prompt 模板、训练中解释性回答习惯、chat 格式、tokenizer 组合，或更深的输出协议。

### 下一步任务

Phase637 应执行：

```text
Newline Prior Suppression Source Audit
```

核心目标：

```text
不要再泛泛找 prefix gate。
直接追踪 DS7B 的 newline prior 来自哪里，以及哪个自然机制能压制它。
```

建议测试：

```text
1. 对 DS7B 的 newline token " ?\n\n" / "\n\n" / " ?\n" 建立专门 logit group。

2. 对 base、repair、source_all6、final_output_repair、readout_delta 比较：
   prefix_logit
   newline_group_max_logit
   prefix_minus_newline

3. 做 prompt ablation:
   去掉问号
   改 answer label
   去掉解释空间
   加明确 short-answer instruction
   加 no-explanation instruction

4. 不要只测 target 样本，要加入 non-target side-effect。

5. 如果某个 prompt ablation 能自然压低 newline prior，
   再回溯它改变了哪些 residual / attention / final output 状态。
```

阶段性判断：

```text
DS7B 剩余瓶颈已经从 readout vector builder
进一步定位为 newline / format continuation prior suppression。
```

## Phase 637: Newline Prior Suppression Source Audit 换行先验压制来源审计 [2026-06-25 18:28]

### 触发背景

用户上传的 Phase636 分析基本正确。Phase636 的关键定位是：

```text
DS7B 的 token0 prefix gate 失败，
主要不是输给一般词表噪声，
而是输给 newline / format continuation prior。
```

Phase636 中 DS7B 的关键结果：

```text
base:
  newline top0 = 81/82
  correct_prefix top0 = 0/82

repair_prompt:
  newline top0 = 57/82
  correct_prefix top0 = 20/82

readout_delta:
  correct_prefix top0 = 82/82
```

因此 Phase637 不再泛泛寻找 prefix gate，而是做 prompt ablation，测试 newline prior 到底与哪些格式因素有关。

### 生成脚本

```text
tests/gpt5/phase637_newline_prior_suppression_source_audit.py
tests/gpt5/phase637_newline_prior_suppression_source_audit_summary.py
```

输出目录：

```text
results/glm5_phase637_newline_prior_suppression_source_audit/
```

核心输出：

```text
results/glm5_phase637_newline_prior_suppression_source_audit/phase637_cross_model_summary.md
```

### 执行命令

语法检查：

```bash
python -m py_compile tests/gpt5/phase637_newline_prior_suppression_source_audit.py tests/gpt5/phase637_newline_prior_suppression_source_audit_summary.py
```

Qwen3 smoke：

```bash
python tests/gpt5/phase637_newline_prior_suppression_source_audit.py qwen3 --smoke --output-dir results/glm5_phase637_newline_prior_suppression_source_audit --hard-exit-after-model
```

Qwen3 confirm：

```bash
python tests/gpt5/phase637_newline_prior_suppression_source_audit.py qwen3 --confirm --output-dir results/glm5_phase637_newline_prior_suppression_source_audit --hard-exit-after-model
```

GLM4 confirm：

```bash
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase637_newline_prior_suppression_source_audit.py glm4 --confirm --output-dir results/glm5_phase637_newline_prior_suppression_source_audit --hard-exit-after-model
```

DS7B confirm：

```bash
python tests/gpt5/phase637_newline_prior_suppression_source_audit.py deepseek7b --confirm --output-dir results/glm5_phase637_newline_prior_suppression_source_audit --hard-exit-after-model
```

汇总：

```bash
python tests/gpt5/phase637_newline_prior_suppression_source_audit_summary.py
```

### 测试设计

本阶段使用 256 raw cases，并且不只看 target subset，还把 non-target side-effect 单独统计。

subject 维度：

```text
base_subject
repair_subject
```

prompt variants：

```text
original:
  Question: X rel ?
  Answer:

no_qmark:
  Question: X rel
  Answer:

period:
  Question: X rel.
  Answer:

inline_answer:
  Question: X rel ? Answer:

short_only:
  Instruction: Answer with only the value.
  Question: X rel ?
  Answer:

no_explain:
  Instruction: Do not explain. Answer with only the value.
  Question: X rel ?
  Answer:

no_qmark_short:
  Instruction: Answer with only the value.
  Question: X rel
  Answer:

value_label:
  Question: X rel ?
  Value:

direct_value_label:
  Instruction: Return only the value.
  Question: X rel ?
  Value:
```

核心指标：

```text
tok0 hit
exact generation
wrong_exact
newline_top0
mean_prefix_rank
prefix_minus_newline
top0_category
top0_text
```

### 客观结果

#### Qwen3

样本：

```text
raw_cases = 256
target_seen = 17
rows = 4608
```

repair_subject / target：

```text
original:
  tok0 = 14/17
  exact = 11/17
  newline = 0/17

no_qmark:
  tok0 = 17/17
  exact = 16/17
  newline = 0/17

period:
  tok0 = 16/17
  exact = 15/17
  newline = 0/17

inline_answer:
  tok0 = 1/17
  exact = 0/17
  newline = 9/17

value_label:
  tok0 = 0/17
  exact = 0/17
  newline = 16/17
```

Qwen3 中去掉问号最好，inline_answer 反而破坏结果。

repair_subject / non_target：

```text
original:
  exact = 140/239
  newline = 2/239

no_qmark:
  exact = 208/239
  newline = 0/239

period:
  exact = 192/239
  newline = 0/239

inline_answer:
  exact = 27/239
  newline = 123/239
```

Qwen3 的 no_qmark 同时改善 target 和 non-target。

#### GLM4

样本：

```text
raw_cases = 256
target_seen = 31
rows = 4608
```

repair_subject / target：

```text
original:
  tok0 = 29/31
  exact = 28/31
  newline = 0/31

no_qmark:
  tok0 = 31/31
  exact = 31/31
  newline = 0/31

inline_answer:
  tok0 = 27/31
  exact = 26/31
  newline = 0/31

value_label:
  tok0 = 29/31
  exact = 24/31
  newline = 0/31
```

GLM4 中 no_qmark 也是最优。

repair_subject / non_target：

```text
original:
  exact = 183/225

no_qmark:
  exact = 199/225

inline_answer:
  exact = 177/225

value_label:
  exact = 185/225
```

GLM4 的 no_qmark 也有正向副作用，不破坏整体。

#### DS7B

样本：

```text
raw_cases = 256
target_seen = 82
rows = 4608
```

repair_subject / target：

```text
original:
  tok0 = 20/82
  exact = 20/82
  newline = 57/82
  rank = 9.4
  prefix_minus_newline = -1.704

no_qmark:
  tok0 = 39/82
  exact = 36/82
  newline = 27/82
  rank = 2.1
  prefix_minus_newline = 0.479

period:
  tok0 = 28/82
  exact = 27/82
  newline = 16/82
  rank = 2.8
  prefix_minus_newline = 0.393

inline_answer:
  tok0 = 75/82
  exact = 72/82
  newline = 0/82
  rank = 1.1
  prefix_minus_newline = 2.236

short_only:
  tok0 = 2/82
  exact = 2/82
  newline = 2/82
  top0 主要变成 space

no_explain:
  tok0 = 3/82
  exact = 3/82
  newline = 53/82

value_label:
  tok0 = 3/82
  exact = 3/82
  newline = 79/82

direct_value_label:
  tok0 = 0/82
  exact = 0/82
  newline = 82/82
```

DS7B 的关键结果非常清楚：

```text
inline_answer 是唯一强压制 newline prior 的 prompt 结构。
```

repair_subject / non_target：

```text
original:
  tok0 = 38/174
  exact = 36/174
  newline = 118/174

no_qmark:
  tok0 = 99/174
  exact = 95/174
  newline = 34/174

period:
  tok0 = 63/174
  exact = 62/174
  newline = 20/174

inline_answer:
  tok0 = 171/174
  exact = 159/174
  newline = 0/174

value_label:
  tok0 = 9/174
  exact = 10/174
  newline = 162/174
```

DS7B 的 inline_answer 不只是 target 有效，对 non-target 也显著改善：

```text
target exact:
  20/82 -> 72/82

non_target exact:
  36/174 -> 159/174

newline:
  target 57/82 -> 0/82
  non_target 118/174 -> 0/174
```

### 结论

Phase637 是一个关键正结果。

它把 DS7B newline prior 的来源从泛泛的“解释式回答先验”进一步定位为：

```text
多行问答模板中的换行 Answer 区域触发了强 format continuation prior。
```

最强证据是：

```text
原模板:
  Question: X rel ?
  Answer:

DS7B repair_subject target exact = 20/82
newline = 57/82

同一行模板:
  Question: X rel ? Answer:

DS7B repair_subject target exact = 72/82
newline = 0/82
```

这说明 DS7B 不是不能输出正确 value，而是原始模板把模型推入：

```text
继续解释 / 继续换行 / 继续问题格式
```

的输出协议；inline_answer 改变了输出协议，强行关闭 newline prior。

### 对 Phase636 判断的修正

Phase636 说 DS7B 缺少 newline suppression，这仍然正确。

Phase637 进一步说明：

```text
newline suppression 不是一定要靠内部 patch 完成；
prompt format 本身可以切换输出协议，
尤其是把 Answer 标签放在同一行，可以大幅改变 token0 prior。
```

也就是说：

```text
prefix gate 同时受内部语义状态和外部格式协议控制。
```

### 理论进展

本阶段对“复用差分机制”的意义很大。

同样的规则、同样的对象、同样的关系、同样的答案，只改一个格式结构：

```text
"\nAnswer:"
```

变为：

```text
" Answer:"
```

DS7B 的 exact 从 20/82 到 72/82。

这说明语言机制里存在强烈的：

```text
output protocol state
```

也就是说，模型不是只在“语义内容空间”里竞争，还在“输出协议空间”里竞争。

新的机制拆分应为：

```text
Generation =
SemanticSelection
+ FormatProtocolSelection
+ PrefixTokenCompetition
+ ValueTokenCompetition
+ ConfirmationContinuation
```

其中 DS7B 当前失败点主要是：

```text
FormatProtocolSelection 选择了 newline / explanation protocol，
而不是 inline value answer protocol。
```

### 最严格的问题和硬伤

1. inline_answer 是 prompt-level intervention，不是内部机制 patch。

2. 本阶段证明 prompt format 可以切换输出协议，但还没有定位这个协议状态在残差流中的具体位置。

3. Qwen3 与 GLM4 的最佳变体是 no_qmark，而 DS7B 的最佳变体是 inline_answer，说明跨模型协议触发条件不同。

4. short_only / no_explain 指令没有解决 DS7B，甚至常把 top0 推到 space，说明自然语言指令不等于输出协议切换。

5. Value label 反而强化 DS7B newline prior，说明标签语义不是关键，格式布局才更关键。

### 下一步任务

Phase638 应执行：

```text
Inline Answer Protocol State Backtrace
```

核心问题：

```text
inline_answer 为什么能关闭 DS7B newline prior？
它改变的是 prompt_last residual state，
answer_label state，
question_mark state，
还是 final readout protocol state？
```

建议测试：

```text
1. 构造 original vs inline_answer 对照。

2. 不改变 subject / relation / answer，只改变：
   "\nAnswer:" vs " Answer:"

3. 扫描 source groups:
   prompt_last
   answer_label
   question_mark_answer
   question_all

4. 测量每个位置对 newline margin 的贡献：
   prefix_logit - newline_group_max_logit

5. 做 restore:
   把 inline_answer 的某个位置状态 restore 到 original prompt，
   看是否能压低 newline prior。

6. 对 Qwen3、GLM4、DS7B 都跑，
   因为 Qwen3 / GLM4 的 protocol switch 条件不同。
```

阶段性判断：

```text
DS7B newline prior 来源已经从抽象输出竞争
进一步定位到 prompt format protocol；
下一步必须回溯 inline answer protocol state 的内部承载位置。
```

## Phase 638: Inline Answer Protocol State Backtrace [2026-06-25 18:57]

### 任务来源

用户要求分析 Phase636 附件内容是否正确，综合当前进展继续完成任务，并在无需确认时自动推进阶段性目标。

Phase637 已经证明：

```text
DS7B 的 newline / explanation prior 可以被 prompt format 强烈切换。
original:
Question: category relation ?
Answer:

inline_answer:
Question: category relation ? Answer:
```

其中 DS7B repair_subject target exact 从 20/82 提升到 72/82，newline_top0 从 57/82 降到 0/82。

Phase638 的核心问题是：

```text
inline_answer protocol state 到底由哪些内部位置承载？
如果把 inline prompt 的内部状态 restore 到 original prompt，
是否能压制 original prompt 的 newline prior？
```

### 生成脚本

新增脚本：

```text
tests/gpt5/phase638_inline_answer_protocol_state_backtrace.py
tests/gpt5/phase638_inline_answer_protocol_state_backtrace_summary.py
```

输出目录：

```text
results/glm5_phase638_inline_answer_protocol_state_backtrace/
```

跨模型摘要：

```text
results/glm5_phase638_inline_answer_protocol_state_backtrace/phase638_cross_model_summary.md
```

### 测试命令

smoke 检查：

```bash
python tests/gpt5/phase638_inline_answer_protocol_state_backtrace.py qwen3 --smoke --hard-exit-after-model
python tests/gpt5/phase638_inline_answer_protocol_state_backtrace.py qwen3 --smoke --include-nontarget --hard-exit-after-model
```

正式测试，按模型顺序执行，并使用 hard exit 防止显存残留：

```bash
python tests/gpt5/phase638_inline_answer_protocol_state_backtrace.py qwen3 --confirm --hard-exit-after-model
python tests/gpt5/phase638_inline_answer_protocol_state_backtrace.py glm4 --confirm --hard-exit-after-model
python tests/gpt5/phase638_inline_answer_protocol_state_backtrace.py deepseek7b --confirm --hard-exit-after-model
python tests/gpt5/phase638_inline_answer_protocol_state_backtrace_summary.py
```

运行时间：

```text
qwen3: 6.21 min
GLM4: 8.15 min
DS7B: 7.77 min
```

### 测试原理

Phase638 只改变一个格式差异：

```text
original: "\nAnswer:"
inline:   " Answer:"
```

然后把 inline prompt 的候选内部状态 restore 到 original prompt 的对应位置。

测试位置：

```text
prompt_last
answer_label
question_mark_answer
relation_tail
question_all
all5
```

测试模式：

```text
original
inline
final_output_inline_to_original
patch_prompt_last
patch_answer_label
patch_question_mark_answer
patch_relation_tail
patch_question_all
patch_all5
```

其中：

```text
final_output_inline_to_original
```

表示把 inline prompt 的 final_norm output 直接替换到 original prompt，用来测量最后读出状态是否已经完整携带协议切换。

```text
patch_question_mark_answer / patch_relation_tail / patch_question_all
```

表示把 inline prompt 的相应源位置 layer_out restore 到 original prompt 的对应位置，看是否可以关闭 newline prior。

### 重要过滤和硬约束

三模型均使用：

```text
raw_cases = 256
target + non_target 全部保留
```

每个模型均产生：

```text
mode_rows = 2048
```

过滤统计：

```text
qwen3: group_len_mismatch=512, empty_patch=256
GLM4: group_len_mismatch=512, empty_patch=256
DS7B: group_len_mismatch=512, empty_patch=256
```

原因是 original 与 inline 中部分 token span 不等长，尤其 answer_label 的 tokenizer 边界不完全一致，因此 patch_answer_label 没有形成稳定可比结果。

所以本阶段不能声称 answer_label 已被排除，只能说：

```text
在当前 token-aligned restore 设置下，
answer_label 没有可比 patch。
```

### 客观结果

#### qwen3

target：

```text
original:
tok0 = 14/17
exact = 11/17
newline_top0 = 0/17
mean_rank = 1.2

inline:
tok0 = 1/17
exact = 0/17
newline_top0 = 9/17
mean_rank = 4.8

final_output_inline_to_original:
tok0 = 1/17
exact = 0/17
newline_top0 = 9/17

patch_prompt_last:
tok0 = 2/17
exact = 1/17
newline_top0 = 7/17

patch_question_mark_answer / relation_tail / question_all / all5:
tok0 = 1/17
exact = 0/17
newline_top0 = 9/17
```

non_target：

```text
original:
tok0 = 144/239
exact = 140/239
newline_top0 = 2/239

inline:
tok0 = 26/239
exact = 27/239
newline_top0 = 123/239

final_output_inline_to_original:
tok0 = 26/239
exact = 27/239
newline_top0 = 123/239
```

qwen3 结论：

```text
inline_answer 对 qwen3 是坏协议。
把 inline state restore 到 original 会复制坏协议，
不是修复 original。
```

#### GLM4

target：

```text
original:
tok0 = 29/31
exact = 28/31
newline_top0 = 0/31

inline:
tok0 = 27/31
exact = 26/31
newline_top0 = 0/31

final_output_inline_to_original:
tok0 = 27/31
exact = 27/31
newline_top0 = 0/31

patch_question_mark_answer / relation_tail / question_all / all5:
tok0 = 27/31
exact = 27/31
newline_top0 = 0/31
```

non_target：

```text
original:
tok0 = 209/225
exact = 183/225
newline_top0 = 0/225

inline:
tok0 = 203/225
exact = 177/225
newline_top0 = 0/225
```

GLM4 结论：

```text
GLM4 不存在 DS7B 那种强 newline prior。
inline state 可以被复制，但没有产生关键修复，
因为 original 本来已经接近稳定。
```

#### DS7B

target：

```text
original:
tok0 = 20/82
exact = 20/82
newline_top0 = 57/82
mean_rank = 9.4
prefix_minus_newline = -1.704

inline:
tok0 = 75/82
exact = 72/82
newline_top0 = 0/82
mean_rank = 1.1
prefix_minus_newline = 2.236

final_output_inline_to_original:
tok0 = 75/82
exact = 72/82
newline_top0 = 0/82
mean_rank = 1.1
prefix_minus_newline = 2.236

patch_question_mark_answer:
tok0 = 75/82
exact = 71/82
newline_top0 = 0/82
mean_rank = 1.1
prefix_minus_newline = 2.236

patch_relation_tail:
tok0 = 75/82
exact = 71/82
newline_top0 = 0/82
mean_rank = 1.1
prefix_minus_newline = 2.236

patch_question_all:
tok0 = 75/82
exact = 72/82
newline_top0 = 0/82
mean_rank = 1.1
prefix_minus_newline = 2.236

patch_prompt_last:
tok0 = 67/82
exact = 63/82
newline_top0 = 0/82
mean_rank = 1.2
prefix_minus_newline = 2.217

patch_all5:
tok0 = 67/82
exact = 64/82
newline_top0 = 0/82
```

non_target：

```text
original:
tok0 = 38/174
exact = 36/174
newline_top0 = 118/174
mean_rank = 7.3
prefix_minus_newline = -1.282

inline:
tok0 = 171/174
exact = 159/174
newline_top0 = 0/174
mean_rank = 1.0
prefix_minus_newline = 2.362

final_output_inline_to_original:
tok0 = 171/174
exact = 154/174
newline_top0 = 0/174

patch_question_mark_answer:
tok0 = 171/174
exact = 154/174
newline_top0 = 0/174

patch_relation_tail:
tok0 = 171/174
exact = 154/174
newline_top0 = 0/174

patch_question_all:
tok0 = 171/174
exact = 158/174
newline_top0 = 0/174

patch_prompt_last:
tok0 = 165/174
exact = 148/174
newline_top0 = 0/174
```

DS7B 结论：

```text
DS7B 的 inline answer protocol state 可以被局部 restore 到 original prompt。

question_mark_answer / relation_tail / question_all 三个位置几乎复现 inline prompt 的 token0 效果。

prompt_last 也能关闭 newline prior，但恢复强度略弱。
```

这说明 DS7B 的 newline prior 不是只能在最后读出端处理，而是在 prompt 源位置已经形成了可迁移的协议状态。

### 对附件 Phase636 分析的评估

附件中关于 Phase636 的判断基本正确：

```text
DS7B 的 token0 failure 不是 value readout 完全缺失，
而是 newline / format continuation prior 压过了 correct prefix。
```

Phase638 对它做了关键补充：

```text
newline prior 的来源不是抽象的“模型偏好”，
而是具体 prompt protocol state。

这个 state 可以从 inline prompt 的 question_mark_answer / relation_tail / question_all 位置
restore 到 original prompt，并关闭 newline_top0。
```

所以 Phase636 到 Phase638 的链条是：

```text
Phase636:
定位竞争物是 newline prior。

Phase637:
证明 prompt format 可以自然关闭 newline prior。

Phase638:
证明 inline format 的内部状态可以迁移到 original，
并在 DS7B 上几乎复现 inline 的修复效果。
```

### 理论进展

本阶段把“格式协议”从外部 prompt 现象推进为内部状态对象。

新的机制拼图为：

```text
FormatProtocolState(prompt)
    -> SourcePositionState(question_mark_answer, relation_tail, question_all)
    -> FinalReadoutState
    -> PrefixTokenCompetition(correct_prefix vs newline)
    -> NaturalGeneration
```

对于 DS7B：

```text
original prompt:
FormatProtocolState = multiline_answer_protocol
newline prior dominates

inline prompt:
FormatProtocolState = inline_value_protocol
correct prefix dominates
```

而且：

```text
inline_value_protocol
```

不是只存在于 final output，它已经在 question tail 区域形成。

这对“相对编码和差分复用机制”的意义很大：

```text
同一套参数、同一语义、同一关系、同一答案，
仅仅由于 format protocol state 不同，
输出路径就从 explanation/newline 切到 value-prefix。
```

也就是说，神经网络复用同一计算结构时，很可能不是通过固定模块切换，而是通过：

```text
conditioned state field
```

改变后续读出竞争。

### 最严格的问题和硬伤

1. answer_label 没有稳定可比结果。

由于 original 与 inline 的 tokenization span 不完全等长，本阶段不能判断 answer_label 是否是协议状态核心位置。

2. patch_question_all 与 patch_question_mark_answer / relation_tail 的结果非常接近。

这说明当前方法还不能拆清楚到底是 question mark、Answer label、空格、还是整段尾部共同贡献。

3. all5 反而弱于 question_mark_answer / question_all。

这可能是多位置 patch 的干扰，也可能是 prompt_last 与上游局部状态有冲突；不能简单解释为“更多位置更好”。

4. qwen3 和 GLM4 的 inline protocol 不是好协议。

所以不能把 inline answer 当成跨模型通用规则，只能说它是 DS7B 的关键 protocol switch。

5. 本阶段仍然是 restore 实验，不是机制生成实验。

它证明 inline state 可迁移，但还没有解释 DS7B 为什么在 original 中自然生成 multiline state。

### 当前阶段性结论

最谨慎表述：

```text
DS7B 的 correct-prefix failure 很大程度来自 prompt protocol state。

original 的 "\nAnswer:" 触发 multiline / explanation protocol，
导致 newline token 进入 token0 竞争前列。

inline 的 " Answer:" 触发 inline value protocol，
并且这个协议状态在 question tail 相关源位置已经形成。

把 inline 的 question tail state restore 到 original，
可以几乎复制 inline 的 newline suppression 和 correct-prefix promotion。
```

这不是完整语言机制闭合，但已经把一个过去看似“最后读出失败”的问题，推进为：

```text
格式条件化状态 -> 源位置写入 -> 最终读出竞争
```

### 对破解语言编码机制的启发

Phase638 支持一个更核心的方向：

```text
语言能力不是由单个语义向量决定，
而是由多种条件化状态场共同决定。
```

至少包括：

```text
semantic state
format protocol state
relation state
value candidate state
prefix competition state
continuation policy state
```

这些状态不是完全独立模块，而是在同一残差流和同一参数上复用。

这与“相对编码 / 差分复用机制”的假设一致：

```text
模型不需要为每一种语言行为保存一套独立结构；
它通过 prompt condition 激活不同的相对差分，
让同一套参数在不同状态场下执行不同路径。
```

### 下一步任务

Phase639 应执行：

```text
Protocol Tail Minimal Causal Unit Audit
```

目标不是继续扩大 patch 空间，而是缩小最小因果单位。

核心问题：

```text
DS7B 的 inline protocol state 到底来自哪个最小 token / token pair？

是 question mark token？
是 question mark 后的 space/newline？
是 Answer token？
是 colon？
还是 relation tail 与 Answer label 的组合？
```

测试方案：

```text
1. 构造 original 与 inline 的 token 对齐图。

2. 不再只按 substring group patch，
   而是按 tail token index 做最小单位 restore。

3. 对每个 tail token 单独 restore：
   question_mark
   separator_space_or_newline
   Answer
   colon
   prompt_last

4. 测量：
   correct_prefix tok0
   newline_top0
   prefix_minus_newline
   exact generation

5. 对 DS7B 做主测试，
   qwen3 / GLM4 做边界对照。
```

如果 Phase639 能找到最小协议 token 单位，就可以进入：

```text
Protocol State Construction Path
```

即追踪这个最小状态由哪些 attention / MLP 写入，从而把 format protocol 从“可恢复状态”推进为“可生成机制图谱”。

## Phase 639: Protocol Tail Minimal Causal Unit Audit [2026-06-25 19:26]

### 任务来源

Phase638 已经证明：

```text
DS7B 的 inline protocol state 可以从 question_mark_answer / relation_tail / question_all 等较大区域 restore 到 original prompt，
并关闭 original 中的 newline prior。
```

但 Phase638 仍然存在关键硬伤：

```text
group 太大，不能判断真正因果单位。
```

Phase639 因此继续自动推进，目标是把 protocol tail 缩小到最小 token / token-pair 单位。

### 生成脚本

新增脚本：

```text
tests/gpt5/phase639_protocol_tail_minimal_causal_unit_audit.py
tests/gpt5/phase639_protocol_tail_minimal_causal_unit_audit_summary.py
```

输出目录：

```text
results/glm5_phase639_protocol_tail_minimal_causal_unit_audit/
```

跨模型摘要：

```text
results/glm5_phase639_protocol_tail_minimal_causal_unit_audit/phase639_cross_model_summary.md
```

### 测试命令

smoke：

```bash
python tests/gpt5/phase639_protocol_tail_minimal_causal_unit_audit.py qwen3 --smoke --include-nontarget --hard-exit-after-model
```

正式测试：

```bash
python tests/gpt5/phase639_protocol_tail_minimal_causal_unit_audit.py qwen3 --confirm --hard-exit-after-model
python tests/gpt5/phase639_protocol_tail_minimal_causal_unit_audit.py glm4 --confirm --hard-exit-after-model
python tests/gpt5/phase639_protocol_tail_minimal_causal_unit_audit.py deepseek7b --confirm --hard-exit-after-model
python tests/gpt5/phase639_protocol_tail_minimal_causal_unit_audit_summary.py
```

运行时间：

```text
qwen3: 6.94 min
GLM4: 8.91 min
DS7B: 8.68 min
```

### 测试原理

继续使用同一对 prompt：

```text
original: Question: category relation ?
          Answer:

inline:   Question: category relation ? Answer:
```

测试最小单位：

```text
qmark
separator
answer_word
colon
prompt_last
qmark_separator
separator_answer
answer_colon
tail_all
```

含义：

```text
qmark: 问号相关 token
separator: "\nAnswer:" 或 " Answer:" 的分隔区域
answer_word: Answer 文本
colon: 冒号
prompt_last: 最后 token
qmark_separator: 问号 + 分隔区域
separator_answer: 分隔区域 + Answer
answer_colon: Answer + 冒号
tail_all: 整个尾部可比区域
```

核心操作：

```text
把 inline prompt 的某个最小单位 layer_out
restore 到 original prompt 的对应单位，
然后观察 original prompt 的 token0 竞争是否从 newline 切换为 correct prefix。
```

### 重要过滤和边界

三模型均使用：

```text
raw_cases = 256
target + non_target 全部保留
```

mode_rows：

```text
qwen3: 2560
GLM4: 2560
DS7B: 2560
```

过滤统计：

```text
qwen3: unit_missing=256, unit_len_mismatch=256, empty_patch=512
GLM4: unit_missing=256, unit_len_mismatch=256, empty_patch=512
DS7B: unit_missing=256, unit_len_mismatch=256, empty_patch=512
```

token 长度样本显示：

```text
answer_word:
original length = 1
inline length = 0

answer_colon:
original length = 2
inline length = 1
```

所以：

```text
answer_word 和 answer_colon 在当前 tokenizer 对齐方式下不可比。
```

因此 Phase639 不能判断 Answer word 本身是否有独立作用，只能判断 qmark / separator / colon / prompt_last / tail pair 的作用。

### 客观结果

#### qwen3

target：

```text
original:
tok0 = 14/17
exact = 11/17
newline_top0 = 0/17

inline:
tok0 = 1/17
exact = 0/17
newline_top0 = 9/17

patch_qmark:
tok0 = 13/17
exact = 10/17
newline_top0 = 0/17

patch_separator:
tok0 = 0/17
exact = 0/17
newline_top0 = 16/17

patch_colon:
tok0 = 2/17
exact = 1/17
newline_top0 = 7/17

patch_prompt_last:
tok0 = 2/17
exact = 1/17
newline_top0 = 7/17

patch_qmark_separator / tail_all:
tok0 = 1/17
exact = 0/17
newline_top0 = 9/17
```

non_target：

```text
original:
tok0 = 144/239
exact = 140/239
newline_top0 = 2/239

inline:
tok0 = 26/239
exact = 27/239
newline_top0 = 123/239

patch_separator:
tok0 = 24/239
exact = 24/239
newline_top0 = 177/239
```

qwen3 结论：

```text
qwen3 中 inline separator 是强坏协议源。
separator restore 会显著制造 newline prior。
qmark 基本不破坏 original，separator 是主要破坏单位。
```

#### GLM4

target：

```text
original:
tok0 = 29/31
exact = 28/31
newline_top0 = 0/31

inline:
tok0 = 27/31
exact = 26/31
newline_top0 = 0/31

patch_qmark:
tok0 = 30/31
exact = 29/31
newline_top0 = 0/31

patch_separator:
tok0 = 27/31
exact = 25/31
newline_top0 = 0/31

patch_colon / prompt_last:
tok0 = 26/31
exact = 25/31
newline_top0 = 0/31
```

non_target：

```text
original:
tok0 = 209/225
exact = 183/225
newline_top0 = 0/225

patch_qmark:
tok0 = 216/225
exact = 189/225
newline_top0 = 0/225

patch_separator:
tok0 = 184/225
exact = 156/225
newline_top0 = 0/225
```

GLM4 结论：

```text
GLM4 没有 DS7B 的 newline prior。
qmark restore 甚至略微增强正确前缀。
separator 主要带来轻微退化，但不是 newline 问题。
```

#### DS7B

target：

```text
original:
tok0 = 20/82
exact = 20/82
newline_top0 = 57/82
mean_rank = 9.4
prefix_minus_newline = -1.704

inline:
tok0 = 75/82
exact = 72/82
newline_top0 = 0/82
mean_rank = 1.1
prefix_minus_newline = 2.236

final_output_inline_to_original:
tok0 = 75/82
exact = 72/82
newline_top0 = 0/82

patch_qmark:
tok0 = 20/82
exact = 20/82
newline_top0 = 60/82
mean_rank = 7.0
prefix_minus_newline = -1.572

patch_separator:
tok0 = 72/82
exact = 70/82
newline_top0 = 1/82
mean_rank = 1.1
prefix_minus_newline = 2.280

patch_colon:
tok0 = 69/82
exact = 68/82
newline_top0 = 1/82
mean_rank = 1.2
prefix_minus_newline = 1.968

patch_prompt_last:
tok0 = 67/82
exact = 63/82
newline_top0 = 0/82

patch_qmark_separator:
tok0 = 75/82
exact = 71/82
newline_top0 = 0/82

patch_separator_answer:
tok0 = 72/82
exact = 70/82
newline_top0 = 1/82

patch_tail_all:
tok0 = 75/82
exact = 72/82
newline_top0 = 0/82
```

non_target：

```text
original:
tok0 = 38/174
exact = 36/174
newline_top0 = 118/174

inline:
tok0 = 171/174
exact = 159/174
newline_top0 = 0/174

patch_qmark:
tok0 = 38/174
exact = 35/174
newline_top0 = 124/174

patch_separator:
tok0 = 168/174
exact = 153/174
newline_top0 = 0/174

patch_colon:
tok0 = 166/174
exact = 150/174
newline_top0 = 1/174

patch_prompt_last:
tok0 = 165/174
exact = 148/174
newline_top0 = 0/174

patch_qmark_separator:
tok0 = 171/174
exact = 154/174
newline_top0 = 0/174

patch_tail_all:
tok0 = 171/174
exact = 158/174
newline_top0 = 0/174
```

DS7B 结论：

```text
separator 是最小主因果单位。

只 patch_separator 就能把 target exact 从 20/82 提升到 70/82，
newline_top0 从 57/82 降到 1/82。

qmark 单独无效，甚至略微增加 newline_top0。

colon / prompt_last 有作用，但弱于 separator。

qmark_separator / tail_all 可接近完整 inline 效果。
```

### Phase638 到 Phase639 的关键推进

Phase638 得到：

```text
question_mark_answer / relation_tail / question_all 都能恢复 DS7B inline protocol。
```

Phase639 进一步拆解为：

```text
不是 question mark 本身。
主要是 separator。
```

这把协议状态定位从：

```text
question tail region
```

压缩为：

```text
space/newline + Answer label boundary state
```

更准确地说：

```text
DS7B 的 format protocol switch 主要发生在 separator boundary。
```

### 对附件 Phase636 分析的最终评估

附件对 Phase636 的分析是正确的，但现在可以更精确。

原判断：

```text
DS7B 的 token0 failure 来自 newline / format continuation prior。
```

现在修正为：

```text
DS7B 的 newline / format continuation prior 主要由 separator boundary state 触发。

"\nAnswer:" boundary 会让模型进入 multiline / explanation protocol。

" Answer:" boundary 会让模型进入 inline value protocol。
```

这比“newline prior”更具体，因为它不是输出端凭空偏好 newline，而是 prompt 尾部分隔结构写入了不同协议状态。

### 理论进展

本阶段支持一个非常重要的语言编码机制判断：

```text
语言格式不是表层符号。
格式边界本身会写入可迁移的内部状态。
```

同一个问题：

```text
Question: X relation ?
Answer:
```

与：

```text
Question: X relation ? Answer:
```

不是只差一个字符排版，而是差一个内部协议态：

```text
multiline_answer_protocol
vs
inline_value_protocol
```

这个协议态的最小强因果单位在 DS7B 上主要是：

```text
separator boundary
```

这说明“相对编码 / 差分复用机制”可以更具体化：

```text
同一参数网络复用同一语义计算路径，
但 separator boundary 改变 protocol state，
protocol state 再改变下一 token 的竞争场。
```

可以写成阶段性公式：

```text
H_t = F_\theta(Tokens_{\le t}, C_{\text{semantic}}, C_{\text{protocol}})
```

其中：

```text
C_{\text{protocol}} = \Delta_{\text{boundary}}("\nAnswer:" \rightarrow " Answer:")
```

而 token0 竞争可以写成：

```text
\text{logit}(y_0)
= W_U y_0 \cdot R_{\text{final}}
```

其中：

```text
R_{\text{final}}
= R_{\text{semantic}}
+ R_{\text{protocol}}
+ R_{\text{residual-noise}}
```

DS7B 的关键变化是：

```text
R_{\text{protocol}}("\nAnswer:")
    -> boosts newline / explanation continuation

R_{\text{protocol}}(" Answer:")
    -> boosts value prefix continuation
```

### 最严格的问题和硬伤

1. answer_word / answer_colon 不可比。

由于 tokenizer 边界问题，本阶段没有验证 Answer word 本体是否有独立贡献。

2. separator 的定义仍然包含边界复合。

当前 separator span 长度为 2，不能完全拆成“空格/换行本身”和“Answer 前缀交互”。

3. patch_separator 有轻微 residual side effect。

DS7B target patch_separator 达到 70/82，而完整 inline 是 72/82；说明 separator 是主因，但不是全部。

4. qwen3 上 separator 是坏协议，DS7B 上 separator 是好协议。

因此 separator 的含义依赖模型内部 learned protocol，不是跨模型同义。

5. 仍然是 restore 结果。

本阶段定位了可迁移协议状态，但还没解释这个状态由哪些 attention / MLP 生成。

### 当前阶段性结论

最谨慎结论：

```text
Phase639 基本完成了 DS7B inline protocol 的最小主因果单位定位。

DS7B 的 correct-prefix failure 不是 question mark 造成的，
也不是 final readout 单独造成的，
而主要是 "\nAnswer:" separator boundary 写入了 multiline / explanation protocol state。

把 inline 的 " Answer:" separator state restore 到 original，
即可大幅关闭 newline prior，并恢复 value-prefix generation。
```

这使当前机制链条变成：

```text
separator boundary
-> protocol state
-> final residual readout state
-> correct_prefix vs newline competition
-> natural generation
```

### 对破解语言背后编码机制的启发

这次结果非常重要，因为它说明：

```text
语言模型内部不仅编码“词义”和“关系”，
还编码“当前应该用哪种输出协议继续”的状态。
```

而这种协议状态可以由极小的边界差分触发。

这支持更一般的第一性原理：

```text
语言智能不是在固定语义空间中直接选答案，
而是在多个条件化状态场中动态复用同一网络参数。
```

当前至少有三类状态场已经被实证支持：

```text
semantic value state
format / protocol state
token competition state
```

下一步如果能追踪 separator protocol state 的生成路径，就可以把“状态场”从现象变成图谱。

### 下一步任务

Phase640 应执行：

```text
Separator Protocol State Writer Attribution
```

目标：

```text
找到 DS7B 中谁写入 separator boundary protocol state。
```

建议测试：

```text
1. 固定 DS7B 为主模型，qwen3 / GLM4 做对照。

2. 对 separator 位置扫描层和组件：
   layer_input
   attn_out
   mlp_out
   layer_out

3. 从 inline separator restore 到 original separator。

4. 找出最早能关闭 newline prior 的层。

5. 判断协议状态由 attention 写入还是 MLP 写入。

6. 对关键层做 remove / reverse / random control。
```

关键指标：

```text
newline_top0
correct_prefix tok0
prefix_minus_newline
exact generation
```

阶段性目标：

```text
从 separator boundary state
推进到 separator state writer graph。
```

## Phase 640: Separator Protocol State Writer Attribution [2026-06-25 19:56]

### 任务来源

用户上传的分析认为 Phase636 到 Phase639 的推进基本正确，并指出当前链条已经从：

```text
DS7B token0 prefix failure
```

推进为：

```text
separator boundary
-> format protocol state
-> newline / explanation prior
-> token0 prefix competition
-> natural generation
```

这个判断是正确的。

Phase639 已经证明 DS7B 的主因果单位不是 qmark，而是 separator boundary。Phase640 因此继续自动推进，目标是：

```text
找到 separator boundary protocol state 由哪一层、哪一类组件写入。
```

### 生成脚本

新增脚本：

```text
tests/gpt5/phase640_separator_protocol_state_writer_attribution.py
tests/gpt5/phase640_separator_protocol_state_writer_attribution_summary.py
```

输出目录：

```text
results/glm5_phase640_separator_protocol_state_writer_attribution/
```

跨模型摘要：

```text
results/glm5_phase640_separator_protocol_state_writer_attribution/phase640_cross_model_summary.md
```

### 测试命令

smoke：

```bash
python tests/gpt5/phase640_separator_protocol_state_writer_attribution.py qwen3 --smoke --hard-exit-after-model
python tests/gpt5/phase640_separator_protocol_state_writer_attribution.py qwen3 --smoke --include-nontarget --hard-exit-after-model
```

正式测试：

```bash
python tests/gpt5/phase640_separator_protocol_state_writer_attribution.py qwen3 --confirm --hard-exit-after-model
python tests/gpt5/phase640_separator_protocol_state_writer_attribution.py glm4 --confirm --hard-exit-after-model
python tests/gpt5/phase640_separator_protocol_state_writer_attribution.py deepseek7b --confirm --hard-exit-after-model
python tests/gpt5/phase640_separator_protocol_state_writer_attribution_summary.py
```

运行时间：

```text
qwen3: 2.34 min
GLM4: 5.34 min
DS7B: 8.45 min
```

### 测试原理

Phase640 固定 Phase639 的最小因果单位：

```text
separator boundary
```

对 original prompt 与 inline prompt 的 separator 位置进行状态差分：

```text
original: "\nAnswer:"
inline:   " Answer:"
```

然后在 original prompt 中按层和组件 restore inline separator state。

扫描组件：

```text
layer_input
attn_out
mlp_out
layer_out
```

核心指标：

```text
correct_prefix tok0
newline_top0
mean_prefix_rank
prefix_minus_newline
```

为了控制运行量，本阶段做的是 token0 attribution scan，不在每个 patch 上做完整 generation。完整 generation 已在 Phase639 对 separator patch 做过，证明 separator restore 已经能恢复 DS7B exact 到 70/82。

### 数据范围

三模型都使用：

```text
raw_cases = 256
target_only = True
```

target 样本数：

```text
qwen3: 17
GLM4: 31
DS7B: 82
```

mode rows：

```text
qwen3: 2414
GLM4: 4650
DS7B: 10004
```

过滤：

```text
qwen3: not_target=239, separator_len_mismatch=0, empty_patch=0
GLM4: not_target=225, separator_len_mismatch=0, empty_patch=0
DS7B: not_target=174, separator_len_mismatch=0, empty_patch=0
```

说明 separator span 在三模型上可比，没有 token 长度错配问题。

### 客观结果

#### qwen3

baseline：

```text
original:
tok0 = 14/17
newline_top0 = 0/17
mean_rank = 1.2
prefix_minus_newline = 1.272

inline:
tok0 = 1/17
newline_top0 = 9/17
mean_rank = 4.8
prefix_minus_newline = -1.471
```

最佳 restore：

```text
L26 mlp_out:
tok0 = 16/17
newline_top0 = 0/17
mean_rank = 1.1
prefix_minus_newline = 1.199

L04 attn_out:
tok0 = 15/17
newline_top0 = 0/17

L16 attn_out:
tok0 = 15/17
newline_top0 = 0/17

L20 attn_out:
tok0 = 15/17
newline_top0 = 0/17
```

qwen3 结论：

```text
qwen3 的 original 本来较稳，inline 是坏协议。
restore 结果不能解释为“修复”，只能作为跨模型对照。
```

#### GLM4

baseline：

```text
original:
tok0 = 29/31
newline_top0 = 0/31
mean_rank = 1.1

inline:
tok0 = 27/31
newline_top0 = 0/31
mean_rank = 1.2
```

restore 结果中大量层/组件都接近稳定：

```text
L06 mlp_out:
tok0 = 30/31
newline_top0 = 0/31

L39 mlp_out:
tok0 = 30/31
newline_top0 = 0/31

L00 attn_out:
tok0 = 30/31
newline_top0 = 0/31
```

GLM4 结论：

```text
GLM4 没有 DS7B 的 newline prior，所以 restore 扫描主要表现为轻微扰动，不是关键机制定位。
```

#### DS7B

baseline：

```text
original:
tok0 = 20/82
newline_top0 = 57/82
mean_rank = 9.4
prefix_minus_newline = -1.704

inline:
tok0 = 75/82
newline_top0 = 0/82
mean_rank = 1.1
prefix_minus_newline = 2.236
```

最佳 restore 候选：

```text
L20 layer_out:
tok0 = 77/82
newline_top0 = 0/82
mean_rank = 1.1
prefix_minus_newline = 2.419

L21 layer_input:
tok0 = 77/82
newline_top0 = 0/82
mean_rank = 1.1
prefix_minus_newline = 2.419

L27 layer_out:
tok0 = 75/82
newline_top0 = 0/82
prefix_minus_newline = 2.236

L26 layer_out:
tok0 = 75/82
newline_top0 = 0/82
prefix_minus_newline = 2.458

L27 layer_input:
tok0 = 75/82
newline_top0 = 0/82
prefix_minus_newline = 2.458

L23 layer_out:
tok0 = 72/82
newline_top0 = 0/82
prefix_minus_newline = 2.534

L24 layer_input:
tok0 = 72/82
newline_top0 = 0/82
prefix_minus_newline = 2.534
```

中层开始逐渐成形：

```text
L10 layer_out:
tok0 = 64/82
newline_top0 = 12/82

L14 layer_out:
tok0 = 76/82
newline_top0 = 2/82

L16 layer_out:
tok0 = 76/82
newline_top0 = 2/82

L17 layer_out:
tok0 = 76/82
newline_top0 = 1/82

L20 layer_out:
tok0 = 77/82
newline_top0 = 0/82
```

组件对比：

```text
layer_input / layer_out:
强恢复，且呈连续残差轨迹。

attn_out:
大多无法关闭 newline prior。

mlp_out:
大多无法关闭 newline prior。
```

DS7B control：

```text
random controls:
大多接近 original，newline_top0 仍然很高。

reverse controls:
显著破坏 correct_prefix，部分层 rank 变得极差。
```

例如：

```text
L21 layer_input random:
tok0 = 19/82
newline_top0 = 46/82
prefix_minus_newline = -1.688

L20 layer_out random:
tok0 = 17/82
newline_top0 = 46/82
prefix_minus_newline = -1.710

L20 layer_out reverse:
tok0 = 1/82
newline_top0 = 61/82
mean_rank = 367.0
prefix_minus_newline = -7.345
```

这说明 L20 layer_out / L21 layer_input 的效果不是随机同范数扰动造成的，而是方向性 separator protocol state。

### 对附件分析的评估

附件分析正确：

```text
Phase636 到 Phase639 的主链条已经成立：
separator boundary -> protocol state -> newline prior -> token0 competition。
```

Phase640 对附件分析作出进一步修正：

```text
separator protocol state 的可恢复承载位置主要在 residual stream，
尤其是 DS7B L14 之后逐步显现，
L20 layer_out / L21 layer_input 达到最强。
```

但 Phase640 没有证明：

```text
某个单独 attention out 或 MLP out 就是 writer。
```

更谨慎地说：

```text
separator protocol state 是一个中后层残差状态；
单层 isolated attn_out / mlp_out 不是充分解释。
```

### 理论进展

当前机制链进一步细化为：

```text
separator boundary
-> mid-layer residual protocol state
-> late residual readout-ready state
-> correct_prefix vs newline competition
-> natural generation
```

对于 DS7B，状态轨迹可以粗略写成：

```text
R_{\text{sep}}^{0..10}: weak / partial
R_{\text{sep}}^{14..17}: mostly formed
R_{\text{sep}}^{20..21}: strongest carrier
R_{\text{sep}}^{23..27}: readout-stable carrier
```

这支持“状态场”而不是“单点模块”的观点：

```text
separator protocol state 不是某个组件瞬间写出的独立向量，
而是在 residual stream 中逐层形成和携带。
```

### 最严格的问题和硬伤

1. Phase640 是 target-only。

为了覆盖全层组件，正式扫描只保留 target cases。DS7B target=82 已较充分，但 non-target 副作用没有在本阶段扫描。

2. 不是完整 writer graph。

本阶段定位到 L20 layer_out / L21 layer_input 是强承载点，但没有证明 L20 内部谁写入。

3. isolated attn_out / mlp_out 失败不等于 attention / MLP 无关。

因为 layer_out 是 residual 累积状态，真正写入可能分散在多个早期组件中。

4. L14 已经出现强恢复。

这说明 L20 是强承载点，不一定是最早形成点。真正形成区间可能在 L10-L14 或更早。

5. qwen3 / GLM4 是边界对照，不是同构机制。

qwen3 的 inline 是坏协议，GLM4 没有 newline prior，所以不能把 DS7B 的 writer 结论直接外推。

### 当前阶段性结论

最谨慎结论：

```text
DS7B 的 separator protocol state 不是最后一层才出现。

它在中层 residual stream 中逐渐形成，
到 L14-L17 已经能大幅关闭 newline prior，
到 L20 layer_out / L21 layer_input 达到最强承载，
并在 L23-L27 继续保持 readout-ready 状态。
```

这一结论把 Phase639 的：

```text
separator 是最小主因果单位
```

推进为：

```text
separator protocol state 的主承载轨迹在 DS7B 中后层 residual stream。
```

### 对破解语言编码机制的启发

这一步很关键，因为它说明语言的格式协议不是“末端输出规则”，而是逐层形成的内部状态。

可以用更清楚的状态公式表示：

```text
R_l = R_l^{semantic} + R_l^{protocol} + R_l^{syntax} + R_l^{noise}
```

其中：

```text
R_l^{protocol}
= f_l(separator_boundary, context)
```

DS7B 的 Phase640 结果显示：

```text
R_l^{protocol}
```

不是静态向量，而是层间轨迹：

```text
R_{10}^{protocol} < R_{14}^{protocol} < R_{20}^{protocol}
```

这进一步支持“相对编码 / 差分复用机制”：

```text
同一参数网络通过 separator boundary 这种极小差分，
激活不同的 residual protocol trajectory，
从而改变后续 token competition。
```

### 下一步任务

Phase641 应执行：

```text
Separator Protocol Formation Interval Audit
```

目标：

```text
把 Phase640 的 residual carrier 轨迹进一步拆成形成区间。
```

核心问题：

```text
separator protocol state 是在哪个层间区间从 weak 变 strong？

是 L10-L14 形成？
是 L14-L20 累积？
还是多个区间共同贡献？
```

建议测试：

```text
1. 固定 DS7B 为主，qwen3 / GLM4 做轻量对照。

2. 用 interval restore：
   L0-8
   L8-12
   L12-14
   L14-17
   L17-20
   L20-23
   L23-27

3. 对每个 interval patch separator position 的 layer_out trajectory。

4. 比较 restore / random / reverse。

5. 记录 correct_prefix tok0、newline_top0、prefix_minus_newline。
```

阶段目标：

```text
从“L20 是强承载点”
推进到“哪一段层间计算生成了 separator protocol state”。
```

## Phase 641: Separator Protocol Formation Interval Audit [2026-06-25 20:17]

### 本阶段目标

根据 Phase640 的结果，附件中对 Phase640 的判断基本正确：DS7B 的 separator protocol state 不是最后读出端的偶然现象，而是 residual stream 中可迁移、可修补、可追踪的 protocol trajectory。

但 Phase640 仍有一个关键硬伤：

```text
L20 layer_out 是强 carrier，
不等于 L20 是 formation point。
```

因此 Phase641 的目标是把单层 writer attribution 推进到 interval audit：

```text
把 inline separator 的 layer_out trajectory 按层区间 restore 到 original prompt，
观察哪个区间能关闭 newline prior，并把 correct prefix 推到 token0。
```

### 生成脚本

新增主测试脚本：

```text
tests/gpt5/phase641_separator_protocol_formation_interval_audit.py
```

新增汇总脚本：

```text
tests/gpt5/phase641_separator_protocol_formation_interval_audit_summary.py
```

输出目录：

```text
results/glm5_phase641_separator_protocol_formation_interval_audit/
```

### 执行命令

烟测命令：

```bash
python tests/gpt5/phase641_separator_protocol_formation_interval_audit.py qwen3 --smoke --include-nontarget --hard-exit-after-model
```

正式测试命令，三个模型严格顺序执行，避免 GPU 内存叠加：

```bash
python tests/gpt5/phase641_separator_protocol_formation_interval_audit.py qwen3 --confirm --hard-exit-after-model
python tests/gpt5/phase641_separator_protocol_formation_interval_audit.py glm4 --confirm --hard-exit-after-model
python tests/gpt5/phase641_separator_protocol_formation_interval_audit.py deepseek7b --confirm --hard-exit-after-model
python tests/gpt5/phase641_separator_protocol_formation_interval_audit_summary.py
```

汇总文件：

```text
results/glm5_phase641_separator_protocol_formation_interval_audit/phase641_cross_model_summary.md
```

### 测试原理

对每个样本构造两个 prompt：

```text
original:
Question: X relation ?
Answer:

inline:
Question: X relation ? Answer:
```

Phase639 已证明主因果单位是 separator boundary。Phase641 固定 separator position，只 patch：

```text
component = layer_out
```

对每个区间执行：

```text
original separator position 的 residual trajectory
<- inline separator position 的 layer_out trajectory
```

并比较三类控制：

```text
restore: 使用真实 inline state
random: 使用随机扰动控制
reverse: 使用反向差分控制
```

核心读出指标：

```text
correct_prefix token0 hit
newline_top0
mean_prefix_rank
prefix_minus_newline
```

其中：

$$
M_{\text{newline}}
=
\ell_{\text{correct prefix}}
-
\max_{r\in G_{\text{newline}}}\ell_r
$$

如果：

$$
M_{\text{newline}}>0
$$

说明 correct prefix 已经压过 newline prior。

### 客观结果

#### qwen3

样本：

```text
raw_cases = 256
target_seen = 17
cases_written = 17
```

baseline：

```text
original: tok0 = 14/17, newline = 0/17, rank = 1.2, prefix_minus_newline = 1.272
inline:   tok0 = 1/17,  newline = 9/17, rank = 4.8, prefix_minus_newline = -1.471
```

qwen3 的现象和 DS7B 相反：original 已经强，inline 反而引入 newline/space 竞争。因此 qwen3 不是本阶段 DS7B separator protocol failure 的同构样本。

restore 区间没有形成有效 inline protocol：

```text
L00_08: tok0 = 3/17, newline = 14/17
L08_16: tok0 = 1/17, newline = 16/17
L16_24: tok0 = 0/17, newline = 17/17
L24_32: tok0 = 1/17, newline = 13/17
L32_35: tok0 = 1/17, newline = 9/17
L24_35: tok0 = 1/17, newline = 9/17
```

解释：qwen3 上不应把 inline separator 当作“修复协议态”，因为它本身在目标子集上降低了结果。

#### GLM4

样本：

```text
raw_cases = 256
target_seen = 31
cases_written = 31
```

baseline：

```text
original: tok0 = 29/31, newline = 0/31, rank = 1.1, prefix_minus_newline = 80.722
inline:   tok0 = 27/31, newline = 0/31, rank = 1.2, prefix_minus_newline = 71.648
```

GLM4 本身几乎没有 DS7B 式 newline prior failure。所有 restore 区间都保持强 token0：

```text
L00_08: tok0 = 29/31, newline = 0/31
L08_16: tok0 = 29/31, newline = 0/31
L16_24: tok0 = 27/31, newline = 0/31
L24_32: tok0 = 27/31, newline = 0/31
L32_39: tok0 = 27/31, newline = 0/31
L24_39: tok0 = 27/31, newline = 0/31
```

解释：GLM4 可以作为“无明显换行先验问题”的对照模型，但不能用来定位 DS7B 的 separator protocol formation。

#### DS7B

样本：

```text
raw_cases = 256
target_seen = 82
cases_written = 82
```

baseline：

```text
original: tok0 = 20/82, newline = 57/82, rank = 9.4, prefix_minus_newline = -1.704
inline:   tok0 = 75/82, newline = 0/82,  rank = 1.1, prefix_minus_newline = 2.236
```

DS7B 的 restore 区间结果：

```text
L00_08: tok0 = 60/82, newline = 12/82, rank = 2.3, prefix_minus_newline = 1.042
L08_12: tok0 = 62/82, newline = 17/82, rank = 1.9, prefix_minus_newline = 0.910
L10_14: tok0 = 76/82, newline = 2/82,  rank = 1.2, prefix_minus_newline = 1.540
L12_14: tok0 = 76/82, newline = 2/82,  rank = 1.2, prefix_minus_newline = 1.540
L14_17: tok0 = 76/82, newline = 1/82,  rank = 1.2, prefix_minus_newline = 1.986
L17_20: tok0 = 77/82, newline = 0/82,  rank = 1.1, prefix_minus_newline = 2.419
L20_23: tok0 = 72/82, newline = 0/82,  rank = 1.1, prefix_minus_newline = 2.534
L23_27: tok0 = 75/82, newline = 0/82,  rank = 1.1, prefix_minus_newline = 2.236
L10_20: tok0 = 77/82, newline = 0/82,  rank = 1.1, prefix_minus_newline = 2.419
L14_20: tok0 = 77/82, newline = 0/82,  rank = 1.1, prefix_minus_newline = 2.419
L14_27: tok0 = 75/82, newline = 0/82,  rank = 1.1, prefix_minus_newline = 2.236
```

DS7B 控制结果：

```text
random:
tok0 大多 13/82 到 25/82，newline 大多 37/82 到 56/82，
接近 original 的失败结构，不能复制 restore。

reverse:
tok0 大多 0/82 到 3/82，
newline 或 explanation/word 竞争显著增强，
prefix rank 可恶化到数百位。
```

因此 DS7B 的 interval restore 不是普通扰动造成的，而是 inline separator trajectory 的方向性因果效应。

### 阶段性结论

Phase641 支持以下判断：

```text
DS7B 的 separator protocol state 在 L10-L14 已经明显成形，
在 L14-L17 继续增强，
在 L17-L20 达到最强可迁移闭合，
在 L20-L27 保持 readout-ready carrier state。
```

更谨慎地说：

```text
L20 不是唯一形成点；
L20 是强承载点和强读出就绪点。
真正的形成区间至少要提前到 L10-L14 / L14-L17，
完整闭合最强区间是 L17-L20。
```

这修正了 Phase640 中可能过强的解释：

```text
错误说法：
L20 生成 separator protocol state。

更准确说法：
L10-L20 之间形成并强化 separator protocol trajectory，
L20/L21 是最强 carrier/readout bridge。
```

### 对附件分析的评估

附件对 Phase640 的总体判断正确，尤其正确指出：

```text
Phase640 是从 separator boundary 到 residual protocol trajectory 的关键推进。
L20 是强 carrier，不一定是最早 formation point。
下一步应该做 interval audit。
```

Phase641 已经验证了这条建议的必要性，并给出更客观的区间证据。

### 对语言编码机制研究的进展

当前机制链条可以更新为：

```text
separator boundary
→ L10-L20 residual protocol trajectory formation
→ L20/L21 strong carrier and readout-ready bridge
→ prefix-vs-newline competition
→ natural generation token0 behavior
```

从“相对编码 / 差分复用机制”的角度看，这个结果非常重要：

```text
同一套参数没有固定输出一种格式；
极小 separator 差分会沿着 residual stream 激活不同 protocol trajectory；
这个 trajectory 改变最终 token competition。
```

也就是说，模型复用同一批层和参数，但通过 prompt boundary 的差分进入不同“状态轨道”：

$$
h_l^{\text{inline}}
-
h_l^{\text{original}}
=
\Delta h_l^{\text{protocol}}
$$

而这个差分不是只在输入端存在，而是沿层传播和强化：

$$
\Delta h_{10}^{\text{protocol}}
\rightarrow
\Delta h_{14}^{\text{protocol}}
\rightarrow
\Delta h_{20}^{\text{protocol}}
\rightarrow
\Delta \ell_{\text{prefix-newline}}
$$

### 问题和硬伤

1. Phase641 仍然是 target-only 测试。

它证明了对失败样本的修复有效，但没有完整检查 non-target side effect。后续需要确认是否会破坏原本正确样本，或把本应解释型回答的任务错误压成短答。

2. interval restore 仍可能被区间终点支配。

例如 L17-L20 的强结果可能主要来自 L20 layer_out，而不是整个 L17-L20 的逐层形成。因此 Phase641 不能单独证明“每一层都参与形成”，只能证明该区间含有足够的 protocol trajectory carrier。

3. 只 patch separator position。

Phase638/639 表明 separator 是主因果单位，但 answer label、question tail、final readout position 仍可能参与辅助闭合。

4. qwen3 / GLM4 不是 DS7B 的同构失败样本。

qwen3 的 target 子集上 inline 反而更差，GLM4 几乎没有 newline prior failure。因此跨模型结果不是“同一机制完全复现”，而是显示不同模型在同一 prompt protocol 差分上的状态轨道不同。

### 下一阶段任务

Phase642 应该从 interval restore 进入 endpoint dominance audit：

```text
目标：
区分“区间形成”与“区间终点携带”。
```

建议测试：

```text
1. 固定 DS7B 为主模型。
2. 对 L10-L14、L14-L17、L17-L20 三个关键区间做 leave-one-layer-out。
3. 对每个区间比较：
   full interval restore
   only first layer restore
   only last layer restore
   interval without last layer
   interval without first layer
4. 保持 random / reverse 控制。
5. 指标仍然用 tok0、newline_top0、prefix_minus_newline、prefix_rank。
```

阶段目标：

```text
把“哪个区间含有 protocol state”
推进到
“该区间是由终点 carrier 主导，还是由多层累积形成”。
```

## Phase 642: Endpoint Dominance vs Distributed Formation Audit [2026-06-25 20:49]

### 本阶段目标

根据 Phase641，附件对当前进展的判断基本正确：

```text
separator protocol state 不是 L20 单点瞬间生成，
而是在 L10-L20 之间形成、强化、并进入 readout-ready carrier state。
```

但 Phase641 的硬伤也很明确：

```text
interval restore 强，不等于区间内每一层都参与形成；
也可能只是区间终点 layer_out 已经携带完整 protocol state。
```

因此 Phase642 继续完成 endpoint dominance audit：

```text
区分 endpoint carrier 主导
和 distributed formation 多层分布式形成。
```

### 生成脚本

主脚本：

```text
tests/gpt5/phase642_endpoint_dominance_vs_distributed_formation.py
```

汇总脚本：

```text
tests/gpt5/phase642_endpoint_dominance_vs_distributed_formation_summary.py
```

结果目录：

```text
results/glm5_phase642_endpoint_dominance_vs_distributed_formation/
```

汇总文件：

```text
results/glm5_phase642_endpoint_dominance_vs_distributed_formation/phase642_cross_model_summary.md
```

### 执行命令

烟测：

```bash
python tests/gpt5/phase642_endpoint_dominance_vs_distributed_formation.py qwen3 --smoke --include-nontarget --hard-exit-after-model
```

正式测试，严格按模型顺序执行：

```bash
python tests/gpt5/phase642_endpoint_dominance_vs_distributed_formation.py qwen3 --confirm --hard-exit-after-model
python tests/gpt5/phase642_endpoint_dominance_vs_distributed_formation.py glm4 --confirm --hard-exit-after-model
python tests/gpt5/phase642_endpoint_dominance_vs_distributed_formation.py deepseek7b --confirm --hard-exit-after-model
python tests/gpt5/phase642_endpoint_dominance_vs_distributed_formation_summary.py
```

### 测试原理

Phase642 对 Phase641 的关键区间做六种拆分：

```text
full
first
last
without_first
without_last
middle
```

并测试两个方向：

```text
to_original:
  inline separator state -> original prompt
  测充分性。

remove_from_inline:
  original separator state -> inline prompt
  测必要性。
```

对每个 patch 读取：

```text
tok0 correct_prefix
newline_top0
mean_prefix_rank
prefix_minus_newline
```

核心判据：

```text
如果 last ≈ full：
  终点 carrier 很强。

如果 without_last / middle 仍强：
  不能解释成只有终点携带。

如果 remove_from_inline(full) 明显破坏 inline：
  该区间对 inline protocol trajectory 有必要性。
```

本阶段没有做全量 exact generation，因为双方向 × 多区间 × 多拆分 × 三模型的生成成本会显著放大。本阶段只回答 token0 prefix-vs-newline competition 的机制问题。

### 客观结果

#### qwen3

样本：

```text
raw_cases = 256
target_seen = 17
cases_written = 17
```

baseline：

```text
original: tok0 = 14/17, newline = 0/17, rank = 1.2, prefix_minus_newline = 1.272
inline:   tok0 = 1/17,  newline = 9/17, rank = 4.8, prefix_minus_newline = -1.471
```

qwen3 延续 Phase641 边界：inline separator 对目标子集是坏协议源，因此 to_original restore 多数会把 original 拉向坏协议。

代表结果：

```text
L00_08 full to_original: tok0 = 3/17, newline = 14/17
L00_08 first to_original: tok0 = 14/17, newline = 1/17
L16_24 full to_original: tok0 = 0/17, newline = 17/17
```

解释：

```text
qwen3 证明“同一 boundary 差分在不同模型中可能是坏协议源”，
但不是 DS7B separator failure 的同构证据。
```

#### GLM4

样本：

```text
raw_cases = 256
target_seen = 31
cases_written = 31
```

baseline：

```text
original: tok0 = 29/31, newline = 0/31, rank = 1.1, prefix_minus_newline = 80.722
inline:   tok0 = 27/31, newline = 0/31, rank = 1.2, prefix_minus_newline = 71.648
```

GLM4 的所有主要 to_original / remove_from_inline 拆分都保持强 token0，几乎没有 newline 竞争：

```text
L00_08 full to_original: tok0 = 29/31, newline = 0/31
L08_16 full to_original: tok0 = 29/31, newline = 0/31
L16_24 full to_original: tok0 = 27/31, newline = 0/31
L24_32 full to_original: tok0 = 27/31, newline = 0/31
```

解释：

```text
GLM4 是无明显 newline prior failure 的稳定对照。
```

#### DS7B

样本：

```text
raw_cases = 256
target_seen = 82
cases_written = 82
```

baseline：

```text
original: tok0 = 20/82, newline = 57/82, rank = 9.4, prefix_minus_newline = -1.704
inline:   tok0 = 75/82, newline = 0/82,  rank = 1.1, prefix_minus_newline = 2.236
```

##### 充分性方向：to_original

L10-L14：

```text
full:         tok0 = 76/82, newline = 2/82, prefix_minus_newline = 1.540
first L10:    tok0 = 64/82, newline = 12/82, prefix_minus_newline = 1.113
last L14:     tok0 = 76/82, newline = 2/82, prefix_minus_newline = 1.540
without_last: tok0 = 73/82, newline = 5/82, prefix_minus_newline = 1.457
middle:       tok0 = 73/82, newline = 5/82, prefix_minus_newline = 1.457
```

L14-L17：

```text
full:         tok0 = 76/82, newline = 1/82, prefix_minus_newline = 1.986
first L14:    tok0 = 76/82, newline = 2/82, prefix_minus_newline = 1.540
last L17:     tok0 = 76/82, newline = 1/82, prefix_minus_newline = 1.986
without_last: tok0 = 76/82, newline = 2/82, prefix_minus_newline = 1.664
middle:       tok0 = 76/82, newline = 2/82, prefix_minus_newline = 1.664
```

L17-L20：

```text
full:         tok0 = 77/82, newline = 0/82, prefix_minus_newline = 2.419
first L17:    tok0 = 76/82, newline = 1/82, prefix_minus_newline = 1.986
last L20:     tok0 = 77/82, newline = 0/82, prefix_minus_newline = 2.419
without_last: tok0 = 76/82, newline = 1/82, prefix_minus_newline = 2.281
middle:       tok0 = 76/82, newline = 1/82, prefix_minus_newline = 2.281
```

L10-L20：

```text
full:         tok0 = 77/82, newline = 0/82, prefix_minus_newline = 2.419
first L10:    tok0 = 64/82, newline = 12/82, prefix_minus_newline = 1.113
last L20:     tok0 = 77/82, newline = 0/82, prefix_minus_newline = 2.419
without_last: tok0 = 76/82, newline = 1/82, prefix_minus_newline = 2.281
middle:       tok0 = 76/82, newline = 1/82, prefix_minus_newline = 2.281
```

充分性结果说明：

```text
last layer 与 full interval 几乎相同，
说明 endpoint carrier 非常强。

但 without_last / middle 仍然强，
说明不能把机制解释成“只有终点层携带”。
```

##### 必要性方向：remove_from_inline

L10-L14：

```text
full:         tok0 = 31/82, newline = 15/82, prefix_minus_newline = 0.195
first L10:    tok0 = 48/82, newline = 3/82,  prefix_minus_newline = 1.010
last L14:     tok0 = 31/82, newline = 15/82, prefix_minus_newline = 0.195
without_last: tok0 = 31/82, newline = 12/82, prefix_minus_newline = 0.377
middle:       tok0 = 31/82, newline = 12/82, prefix_minus_newline = 0.377
```

L14-L17：

```text
full:         tok0 = 25/82, newline = 32/82, prefix_minus_newline = -0.408
first L14:    tok0 = 31/82, newline = 15/82, prefix_minus_newline = 0.195
last L17:     tok0 = 25/82, newline = 32/82, prefix_minus_newline = -0.408
without_last: tok0 = 25/82, newline = 23/82, prefix_minus_newline = -0.157
middle:       tok0 = 25/82, newline = 23/82, prefix_minus_newline = -0.157
```

L17-L20：

```text
full:         tok0 = 19/82, newline = 62/82, prefix_minus_newline = -1.503
first L17:    tok0 = 25/82, newline = 32/82, prefix_minus_newline = -0.408
last L20:     tok0 = 19/82, newline = 62/82, prefix_minus_newline = -1.503
without_last: tok0 = 24/82, newline = 53/82, prefix_minus_newline = -1.012
middle:       tok0 = 24/82, newline = 53/82, prefix_minus_newline = -1.012
```

L10-L20：

```text
full:         tok0 = 19/82, newline = 62/82, prefix_minus_newline = -1.503
first L10:    tok0 = 48/82, newline = 3/82,  prefix_minus_newline = 1.010
last L20:     tok0 = 19/82, newline = 62/82, prefix_minus_newline = -1.503
without_last: tok0 = 24/82, newline = 53/82, prefix_minus_newline = -1.012
middle:       tok0 = 24/82, newline = 53/82, prefix_minus_newline = -1.012
```

必要性结果说明：

```text
从 inline 中移除 L17-L20 或 L10-L20，
会把 inline 几乎拉回 original 的 newline failure：
inline baseline: tok0 = 75/82, newline = 0/82
remove L17-L20: tok0 = 19/82, newline = 62/82
original baseline: tok0 = 20/82, newline = 57/82
```

这是本阶段最强结果。

### 阶段性结论

Phase642 支持以下更精确结论：

```text
1. DS7B 的 separator protocol state 有强 endpoint carrier。
   L14、L17、L20 这些区间终点单层几乎可以复制 full interval 的充分性效果。

2. 但它不是“只有终点层”。
   without_last / middle 仍然能强恢复 original，
   说明中间层已经携带大量 protocol trajectory。

3. necessity 结果更强。
   从 inline 中移除 L17-L20 / L10-L20 会让模型重新出现 newline failure，
   说明这段轨迹对 inline protocol 不只是充分，而且接近必要。
```

更谨慎的机制表述：

```text
L10-L14:
  初始 protocol formation 已经足够强，但更像早期形成 + L14 endpoint carrier。

L14-L17:
  protocol strengthening，把 prefix-newline margin 推到更稳定区间。

L17-L20:
  readout-ready closure interval，是最关键必要段。

L20:
  endpoint carrier 极强，但不是唯一机制。
```

因此 Phase641 的“L10-L20 protocol trajectory”没有被推翻，而是被细化为：

```text
distributed trajectory + endpoint readout carrier
```

### 对附件内容的评估

附件对 Phase641 的整体判断正确：

```text
Phase641 是关键阶段；
下一步应该区分 endpoint dominance 与 distributed formation；
必须加入 remove_from_inline 方向。
```

附件中有少量公式排版错误，例如：

```text
M_newline = prefix max newline
Delta R = inline original
```

应修正为：

$$
M_{\text{newline}}
=
\ell_{\text{prefix}}
-
\max_{r\in G_{\text{newline}}}\ell_r
$$

以及：

$$
\Delta R_l^{\text{protocol}}
=
R_l^{\text{inline}}
-
R_l^{\text{original}}
$$

公式排版错误不影响附件的机制判断。

### 对语言编码机制研究的进展

当前机制链条应更新为：

```text
separator boundary
→ L10-L14 early protocol formation
→ L14-L17 protocol strengthening
→ L17-L20 necessary readout-ready closure
→ L20 endpoint carrier
→ prefix-vs-newline competition
→ token0 natural generation tendency
```

这对“相对编码 / 复用差分机制”的意义是：

```text
同一参数骨架通过 separator boundary 差分进入不同 residual trajectory；
该 trajectory 不是单点向量，而是分布式轨迹；
轨迹中存在强 endpoint carrier，但 endpoint 依赖前序轨迹形成；
最终读出表现为 prefix 与 newline 的竞争翻转。
```

可以写成：

$$
\Delta h_l^{\text{protocol}}
=
h_l^{\text{inline}}
-
h_l^{\text{original}}
$$

并且：

$$
\Delta h_{10:20}^{\text{protocol}}
\Rightarrow
\Delta \ell_{\text{prefix-newline}}>0
$$

从 necessity 方向看：

$$
h_{10:20}^{\text{inline}}
\leftarrow
h_{10:20}^{\text{original}}
\Rightarrow
\Delta \ell_{\text{prefix-newline}}<0
$$

这说明 protocol trajectory 不是外部格式标签，而是可因果移除、可因果恢复的内部状态轨道。

### 问题和硬伤

1. 仍是 target-only。

本阶段回答了失败目标样本的充分性和必要性，但没有补 non-target side effect。

2. 没有全量 exact generation。

本阶段为了控制规模，只测 token0 competition。自然生成闭环应后续用少数关键模式补测：

```text
original
inline
to_original L17-L20 full
remove_from_inline L17-L20 full
```

3. 仍未定位具体 writer。

Phase642 定位了 trajectory interval 和 endpoint carrier，但还没有拆出 attention / MLP / residual feedback 的具体写入器。

4. qwen3 / GLM4 仍只是边界对照。

跨模型统一结论不是“相同层相同方向”，而是：

```text
boundary condition 会写入模型特异的 protocol trajectory。
```

### 下一阶段任务

Phase643 应该进入最小自然生成闭环：

```text
Protocol Trajectory Natural Generation Closure
```

目标：

```text
把 Phase642 的 token0 competition 结果压到真实自然生成输出。
```

建议只测少数关键模式，避免测试规模过大：

```text
1. original
2. inline
3. to_original L17-L20 full
4. to_original L17-L20 middle
5. remove_from_inline L17-L20 full
6. remove_from_inline L17-L20 middle
7. random / reverse controls
```

核心指标：

```text
first token category
exact correct generation
newline/explanation rate
answer string stability
```

阶段目标：

```text
证明 L17-L20 protocol trajectory 不仅改变 logits，
还真实改变自然生成路径。
```

## Phase 643: Protocol Trajectory Natural Generation Closure [2026-06-25 21:17]

### 本阶段目标

根据附件分析，Phase642 的判断基本正确：

```text
DS7B 的 separator protocol state
= distributed trajectory + endpoint readout carrier。
```

Phase642 已经证明 L17-L20 / L10-L20 的 protocol trajectory 可以强烈改变 token0 的 prefix-vs-newline competition，但仍有一个硬伤：

```text
Phase642 主要是 teacher-forced logit audit，
还没有证明它真的改变 greedy natural generation。
```

因此 Phase643 的目标是：

```text
把 Phase642 的 L17-L20 protocol trajectory patch 压到自然生成闭环，
检查 exact generation、newline/explanation tendency 和生成文本分布。
```

### 生成脚本

主脚本：

```text
tests/gpt5/phase643_protocol_trajectory_natural_generation_closure.py
```

汇总脚本：

```text
tests/gpt5/phase643_protocol_trajectory_natural_generation_closure_summary.py
```

结果目录：

```text
results/glm5_phase643_protocol_trajectory_natural_generation_closure/
```

汇总文件：

```text
results/glm5_phase643_protocol_trajectory_natural_generation_closure/phase643_cross_model_summary.md
```

### 执行命令

烟测：

```bash
python tests/gpt5/phase643_protocol_trajectory_natural_generation_closure.py qwen3 --smoke --include-nontarget --hard-exit-after-model
```

正式测试，三个模型严格顺序执行：

```bash
python tests/gpt5/phase643_protocol_trajectory_natural_generation_closure.py qwen3 --confirm --hard-exit-after-model
python tests/gpt5/phase643_protocol_trajectory_natural_generation_closure.py glm4 --confirm --hard-exit-after-model
python tests/gpt5/phase643_protocol_trajectory_natural_generation_closure.py deepseek7b --confirm --hard-exit-after-model
python tests/gpt5/phase643_protocol_trajectory_natural_generation_closure_summary.py
```

### 测试原理

Phase643 固定 Phase642 中最关键的区间：

```text
L17-L20
```

测试少数关键模式，避免组合爆炸：

```text
original
inline
to_original_full_restore
to_original_middle_restore
to_original_full_random
to_original_full_reverse
remove_from_inline_full_restore
remove_from_inline_middle_restore
remove_from_inline_full_random
remove_from_inline_full_reverse
```

其中：

```text
to_original:
  把 inline 的 L17-L20 separator layer_out trajectory 恢复到 original prompt。

remove_from_inline:
  把 original 的 L17-L20 separator layer_out trajectory 写回 inline prompt。
```

生成长度沿用 Phase638/639 的精确答案口径：

```text
max_new_tokens = max(len(answer_ids(candidate_value)))
```

核心指标：

```text
tok0 correct_prefix
exact_correct generation
wrong_exact generation
newline_top0
generation_text distribution
prefix_minus_newline
```

### 客观结果

#### qwen3

样本：

```text
raw_cases = 256
target_seen = 17
cases_written = 17
max_new_tokens = 3
```

结果：

```text
original:
  tok0 = 14/17
  exact = 11/17
  wrong_exact = 3/17
  newline = 0/17

inline:
  tok0 = 1/17
  exact = 0/17
  newline = 9/17

to_original_full_restore:
  tok0 = 0/17
  exact = 0/17
  newline = 16/17

to_original_middle_restore:
  tok0 = 1/17
  exact = 1/17
  newline = 14/17
```

qwen3 延续前几阶段边界：

```text
inline separator 对 qwen3 目标子集是坏协议源。
把 inline trajectory 恢复到 original 会把自然生成拉坏。
```

这说明跨模型不能寻找固定字符规则，而要寻找模型特异 protocol trajectory。

#### GLM4

样本：

```text
raw_cases = 256
target_seen = 31
cases_written = 31
max_new_tokens = 2
```

结果：

```text
original:
  tok0 = 29/31
  exact = 28/31
  wrong_exact = 1/31
  newline = 0/31

inline:
  tok0 = 27/31
  exact = 26/31
  wrong_exact = 1/31
  newline = 0/31

to_original_full_restore:
  tok0 = 29/31
  exact = 26/31
  wrong_exact = 3/31
  newline = 0/31

remove_from_inline_full_restore:
  tok0 = 31/31
  exact = 28/31
  wrong_exact = 2/31
  newline = 0/31
```

GLM4 仍然是稳定对照：

```text
几乎没有 newline prior failure，
L17-L20 patch 主要造成 exact 小幅波动，
不是 DS7B 式 protocol gate failure。
```

#### DS7B

样本：

```text
raw_cases = 256
target_seen = 82
cases_written = 82
max_new_tokens = 3
```

baseline：

```text
original:
  tok0 = 20/82
  exact = 20/82
  wrong_exact = 0/82
  newline = 57/82
  prefix_minus_newline = -1.704
  generation_text 主要为 "?\\n\\nTo solve" 和 "?\\n\\nI think"

inline:
  tok0 = 75/82
  exact = 72/82
  wrong_exact = 0/82
  newline = 0/82
  prefix_minus_newline = 2.236
```

充分性方向：

```text
to_original_full_restore:
  tok0 = 77/82
  exact = 72/82
  wrong_exact = 3/82
  newline = 0/82
  prefix_minus_newline = 2.419

to_original_middle_restore:
  tok0 = 76/82
  exact = 76/82
  wrong_exact = 0/82
  newline = 1/82
  prefix_minus_newline = 2.281
```

控制：

```text
to_original_full_random:
  tok0 = 14/82
  exact = 13/82
  newline = 51/82
  prefix_minus_newline = -1.708

to_original_full_reverse:
  tok0 = 1/82
  exact = 0/82
  newline = 61/82
  prefix_minus_newline = -7.345
```

必要性方向：

```text
remove_from_inline_full_restore:
  tok0 = 19/82
  exact = 20/82
  wrong_exact = 0/82
  newline = 62/82
  prefix_minus_newline = -1.503

remove_from_inline_middle_restore:
  tok0 = 24/82
  exact = 24/82
  wrong_exact = 0/82
  newline = 53/82
  prefix_minus_newline = -1.012
```

必要性控制：

```text
remove_from_inline_full_random:
  tok0 = 73/82
  exact = 71/82
  newline = 0/82
  prefix_minus_newline = 2.225

remove_from_inline_full_reverse:
  tok0 = 75/82
  exact = 73/82
  newline = 0/82
  prefix_minus_newline = 4.232
```

### 阶段性结论

Phase643 证明：

```text
DS7B 的 L17-L20 separator protocol trajectory
不仅改变 teacher-forced token0 logits，
而且真实改变 greedy natural generation。
```

最强证据：

```text
original exact = 20/82
inline exact = 72/82
to_original L17-L20 full exact = 72/82
to_original L17-L20 middle exact = 76/82
remove_from_inline L17-L20 full exact = 20/82
remove_from_inline L17-L20 middle exact = 24/82
```

也就是说：

```text
把 inline 的 L17-L20 trajectory 写入 original，
可以让 original 生成表现接近甚至超过 inline。

把 original 的 L17-L20 trajectory 写回 inline，
可以让 inline 退回 original 式 newline/explanation failure。
```

这使当前链条完成了一个真正闭环：

```text
separator boundary
→ L17-L20 protocol trajectory
→ prefix-vs-newline competition
→ greedy natural generation
```

### 对附件内容的评估

附件对 Phase642 的总体判断正确，尤其正确指出：

```text
Phase643 必须验证 exact generation，
不能只停留在 token0 competition。
```

Phase643 已经完成这个验证。

附件中仍有少量公式排版错误，例如差分公式中缺少减号，但机制判断正确。正确写法是：

$$
\Delta h_l^{\text{protocol}}
=
h_l^{\text{inline}}
-
h_l^{\text{original}}
$$

以及：

$$
M_{\text{newline}}
=
\ell_{\text{prefix}}
-
\max_{r\in G_{\text{newline}}}\ell_r
$$

### 对语言编码机制研究的进展

当前 DS7B 局部机制链条可以写成：

```text
separator boundary
→ L10-L14 early protocol formation
→ L14-L17 strengthening
→ L17-L20 natural-generation-critical trajectory
→ L20 endpoint carrier
→ prefix-vs-newline competition
→ exact natural generation
```

这个结果对“复用差分机制”的意义非常直接：

```text
同一语义问题、同一答案、同一参数骨架，
只改变 separator boundary，
就能让模型进入不同 residual protocol trajectory；
这条 trajectory 不只是改变内部分数，
而是决定自然生成路径。
```

可以表达为：

$$
\Delta h_{17:20}^{\text{protocol}}
\Rightarrow
\Delta \ell_{\text{prefix-newline}}
\Rightarrow
\Delta \operatorname{Generate}
$$

其中：

$$
\Delta \operatorname{Generate}
=
\operatorname{Generate}(h^{\text{inline-protocol}})
-
\operatorname{Generate}(h^{\text{original-protocol}})
$$

### 问题和硬伤

1. 仍是 target-only。

Phase643 证明目标失败样本的自然生成闭环，但还没有检查 non-target side effect。

2. exact generation 使用短答案长度。

本阶段使用候选值 token 长度作为生成长度，适合判断值答案是否精确生成，但不能判断后续长文本稳定性。

3. 仍未拆出 writer。

L17-L20 trajectory 已证明自然生成关键，但 attention / MLP / residual feedback 的具体写入器还没有定位。

4. qwen3 / GLM4 是边界对照，不是同构复现。

qwen3 上 inline 是坏协议，GLM4 没有明显 newline failure。跨模型统一仍应抽象为：

```text
boundary condition → model-specific protocol trajectory → generation behavior
```

### 下一阶段任务

Phase644 应该补 non-target side effect 和任务边界：

```text
Protocol Trajectory Side-Effect and Boundary Audit
```

目标：

```text
检查 L17-L20 protocol trajectory patch 是否只修复目标失败样本，
还是会破坏原本正确样本、非值任务、或本应解释型的任务。
```

建议测试：

```text
1. target failure cases
2. original already-correct cases
3. inline already-bad cases
4. non-value relation cases
5. explanation-needed prompts
```

关键指标：

```text
exact_correct
wrong_exact
newline/explanation rate
over-short-answer rate
semantic value stability
```

阶段目标：

```text
把“能修复目标样本”
推进到
“知道修复边界和副作用”。
```

## Phase 644: Global Atlas Readiness Review and Side-Effect Boundary Plan [2026-06-25 21:20]

### 本阶段性质

本阶段未运行新的模型测试命令，也未新增测试脚本。工作内容是基于 Phase641、Phase642、Phase643 的客观结果，完成理论综合、全局图谱可启动性评估，以及下一阶段实验方案。

参考记录：

```text
Phase641: Separator Protocol Formation Interval Audit
Phase642: Endpoint Dominance vs Distributed Formation Audit
Phase643: Protocol Trajectory Natural Generation Closure
```

### 当前最重要的客观结果

DS7B 的核心闭环已经成立：

```text
original:
  exact = 20/82
  newline = 57/82

inline:
  exact = 72/82
  newline = 0/82

to_original L17-L20 full:
  exact = 72/82
  newline = 0/82

to_original L17-L20 middle:
  exact = 76/82
  newline = 1/82

remove_from_inline L17-L20 full:
  exact = 20/82
  newline = 62/82

remove_from_inline L17-L20 middle:
  exact = 24/82
  newline = 53/82
```

这说明：

```text
separator boundary
→ L17-L20 protocol trajectory
→ prefix-vs-newline competition
→ greedy natural generation
```

已经完成目标样本上的因果闭环。

### 测试原理的统一解释

前几阶段不是在证明“格式会影响输出”这种表面结论，而是在证明：

```text
极小 boundary difference 会写入 residual protocol trajectory，
这个 trajectory 会改变 token competition，
最终改变 natural generation。
```

形式化写法：

$$
\Delta h_l^{\text{protocol}}
=
h_l^{\text{inline}}
-
h_l^{\text{original}}
$$

前缀对换行竞争：

$$
M_{\text{newline}}
=
\ell_{\text{prefix}}
-
\max_{r\in G_{\text{newline}}}\ell_r
$$

自然生成闭环：

$$
\Delta h_{17:20}^{\text{protocol}}
\Rightarrow
\Delta M_{\text{newline}}
\Rightarrow
\Delta \operatorname{Generate}
$$

### 以上内容是否正确

总体正确，但必须保守解释。

正确部分：

```text
1. DS7B 的 separator protocol state 不是单点向量。
2. L10-L14 已出现早期协议形成。
3. L14-L17 继续增强协议状态。
4. L17-L20 是自然生成关键轨迹段。
5. L20 是强 endpoint carrier，但不是唯一机制。
6. random / reverse control 排除了普通扰动解释。
7. remove_from_inline 证明 L17-L20 对 inline protocol 接近必要。
```

必须保守的部分：

```text
1. 当前结果仍然是 target-only。
2. 还没有 non-target side effect audit。
3. 还没有解释型任务和非值任务边界。
4. 还没有拆出 L17-L20 内部具体 writer。
5. qwen3 / GLM4 是边界对照，不是同构复现。
```

### 是否可以开始全局图谱测试

可以开始，但不能直接做“全模型全任务大图谱”。更合理的启动方式是：

```text
局部闭环机制图谱
→ 边界副作用图谱
→ 跨任务复用图谱
→ 跨模型抽象图谱
```

原因：

```text
当前 DS7B 已经有一个完整局部闭环：
minimal causal unit
trajectory interval
sufficiency
necessity
natural generation closure
```

这足以作为全局图谱的第一个稳定锚点。

但还不能直接宣称完整图谱已经成立，因为缺少：

```text
non-target safety boundary
task boundary
writer graph
cross-model abstraction
```

### 当前核心拼图

1. residual stream 是状态总线。
2. hidden state 是动态轨迹，不是静态语义容器。
3. prompt 是条件化状态生成器。
4. boundary pattern 可以触发 protocol trajectory。
5. separator boundary 是 DS7B 格式协议的最小主因果单位。
6. protocol state 是独立机制对象。
7. protocol state 可以被 restore。
8. protocol state 可以被 remove。
9. protocol state 有形成区间。
10. protocol state 有 endpoint carrier。
11. L10-L14 是早期 formation interval。
12. L14-L17 是 strengthening interval。
13. L17-L20 是 natural-generation-critical interval。
14. L20 是强 endpoint carrier。
15. token0 是 format / prefix token。
16. token1 是 semantic value token。
17. token2 是 confirmation token。
18. correct value token attention 是语义值门核心来源。
19. result carrier 通过 layer_out 传播。
20. token0 失败主要是 correct prefix 输给 newline / explanation prior。
21. competitor ladder 比单一 competitor 更准确。
22. prefix_minus_newline 是协议门核心指标。
23. exact generation 可以被 protocol trajectory 因果改变。
24. qwen3 中 inline separator 是坏协议源。
25. GLM4 当前模板下没有明显 newline prior failure。
26. DS7B 对 separator boundary 极度敏感。
27. 同一 boundary 在不同模型中写入不同 protocol trajectory。
28. 跨模型统一对象不是固定字符，而是 boundary-conditioned trajectory。
29. restore 证明充分性。
30. remove 证明必要性。
31. random / reverse controls 证明方向性。
32. interval audit 定位形成区间。
33. endpoint audit 区分终点携带和分布式形成。
34. natural generation closure 证明真实生成效应。
35. target-only 必须补 side-effect audit。

### 统一数学公式更新

输入分解：

$$
x=(F,C,R,O,G,B)
$$

其中：

```text
F = format
C = concept
R = relation
O = output position
G = generation policy
B = boundary pattern
```

边界差分：

$$
\Delta h_l^{B}
=
h_l(B_{\text{inline}})
-
h_l(B_{\text{multi}})
$$

协议轨迹：

$$
\mathcal{T}_{\text{protocol}}
=
\{\Delta h_0^B,\Delta h_1^B,\dots,\Delta h_L^B\}
$$

区间分解：

$$
\Delta h_{a:b}^{\text{protocol}}
=
\Delta h_{a:b-1}^{\text{distributed}}
+
\Delta h_b^{\text{endpoint}}
$$

残差状态分解：

$$
h_l
=
h_l^{\text{semantic}}
+
h_l^{\text{protocol}}
+
h_l^{\text{syntax}}
+
h_l^{\text{competition}}
+
h_l^{\text{noise}}
$$

读出竞争：

$$
\ell_v
=
W_U(v)^\top y
$$

$$
M_{\text{newline}}
=
\ell_{\text{prefix}}
-
\max_{r\in G_{\text{newline}}}\ell_r
$$

协议门：

$$
G_{\text{protocol}}
=
\mathbb{1}[M_{\text{newline}}>0]
$$

完整生成近似：

$$
P(\hat{c}=c)
\approx
P(G_{\text{protocol}})
\cdot
P(G_{\text{prefix}})
\cdot
P(G_{\text{value}})
\cdot
P(G_{\text{confirm}})
$$

统一智能过程：

$$
\operatorname{Intelligence}
=
\operatorname{ReuseSkeleton}
\circ
\operatorname{BoundaryProtocolEncoder}
\circ
\operatorname{DistributedProtocolFormation}
\circ
\operatorname{EndpointCarrier}
\circ
\operatorname{SemanticSelector}
\circ
\operatorname{ReadoutCompetition}
\circ
\operatorname{TokenActor}
\circ
\operatorname{ContinuationController}
$$

### 最新完整理论

当前理论可命名为：

```text
相对编码—复用差分—分布式协议轨迹理论
```

核心表述：

```text
深度神经网络的语言能力不是由孤立概念向量、孤立语法模板或单个注意力头完成，
而是由同一参数骨架在不同输入边界和语义条件下生成不同状态轨迹。
这些状态轨迹包含语义轨迹、协议轨迹、竞争轨迹和确认轨迹。
语言生成不是简单读出知识，而是状态轨迹进入词表竞争后的自回归执行。
```

### 当前进度评估

```text
DS7B separator protocol natural generation closure:
92% 到 96%

DS7B protocol trajectory interval localization:
85% 到 92%

DS7B endpoint vs distributed formation:
78% 到 86%

DS7B side-effect boundary:
15% 到 25%

DS7B writer graph:
20% 到 35%

value semantic gate:
95% 到 99%

format / protocol gate:
82% 到 90%

cross-model abstraction:
55% 到 68%

global reuse-difference atlas:
45% 到 60%

language encoding mechanism:
76% 到 88%

complete intelligence theory:
55% 到 70%
```

### 对语言三大核心特性的反思

知识网络：

```text
知识不是单纯 (concept, relation) -> value。
知识输出必须经过 protocol trajectory 和 token competition。
```

更准确写法：

$$
(C,R,B)
\rightarrow
h^{\text{semantic}}
+
h^{\text{protocol}}
+
h^{\text{competition}}
\rightarrow
\hat{v}
$$

推理能力：

```text
推理不是只有语义选择。
推理还包括协议选择、状态轨迹构造、读出竞争和自回归执行。
```

语法系统：

```text
语法不是表面模板，而是边界协议轨迹系统。
它决定是否换行、是否解释、是否短答、哪个 token 先输出。
```

### 更好的方案

不要继续只做单点机制破解。应进入图谱化方案：

```text
1. 每个机制必须记录 causal unit。
2. 每个机制必须记录 trajectory interval。
3. 每个机制必须记录 sufficiency。
4. 每个机制必须记录 necessity。
5. 每个机制必须记录 generation closure。
6. 每个机制必须记录 side-effect boundary。
```

### 下一阶段方案

下一阶段应执行：

```text
Phase645: Protocol Trajectory Side-Effect and Boundary Atlas
```

目标：

```text
检查 L17-L20 protocol trajectory patch 是否只修复目标失败样本，
还是会破坏原本正确样本、非值任务、解释型任务和跨模板任务。
```

测试集合：

```text
1. target failure cases
2. original already-correct cases
3. inline already-bad cases
4. relation changed cases
5. explanation-needed prompts
6. non-value prompts
```

测试模式：

```text
original
inline
to_original L17-L20 middle restore
remove_from_inline L17-L20 middle restore
random control
reverse control
```

指标：

```text
exact_correct
wrong_exact
newline_rate
explanation_rate
over_short_answer_rate
semantic_stability
side_effect_rate
```

阶段目标：

```text
把“能修复目标样本”
推进到
“知道机制边界、适用范围和副作用”。
```

## Phase 645: Protocol Trajectory Side-Effect and Boundary Atlas [2026-06-25 21:49]

### 任务背景

用户上传的 Phase 644 评价基本正确：Phase 644 不是新实验，而是基于 Phase 641 到 Phase 643 的阶段性综合。它把 DS7B 的 separator boundary -> protocol trajectory -> prefix-vs-newline competition -> greedy natural generation 链条整理为一个可启动全局图谱的局部闭环。

但该评价中最重要的收紧也正确：不能只证明 target failure cases 可以被 L17-L20 protocol trajectory patch 修复，还必须测试副作用边界。否则当前结论可能只是“能把一批样本强行推入短答协议”，而不是完整解释语言编码机制。

### 本阶段生成脚本

```text
tests/gpt5/phase645_protocol_trajectory_side_effect_boundary_atlas.py
tests/gpt5/phase645_protocol_trajectory_side_effect_boundary_atlas_summary.py
```

### 执行命令

smoke：

```bash
python tests/gpt5/phase645_protocol_trajectory_side_effect_boundary_atlas.py qwen3 --smoke --hard-exit-after-model
```

正式顺序测试：

```bash
python tests/gpt5/phase645_protocol_trajectory_side_effect_boundary_atlas.py qwen3 --confirm --hard-exit-after-model
python tests/gpt5/phase645_protocol_trajectory_side_effect_boundary_atlas.py glm4 --confirm --hard-exit-after-model
python tests/gpt5/phase645_protocol_trajectory_side_effect_boundary_atlas.py deepseek7b --confirm --hard-exit-after-model
python tests/gpt5/phase645_protocol_trajectory_side_effect_boundary_atlas_summary.py
```

### 输出文件

```text
results/glm5_phase645_protocol_trajectory_side_effect_boundary_atlas/phase645_qwen3_protocol_trajectory_side_effect_boundary_atlas_confirm.json
results/glm5_phase645_protocol_trajectory_side_effect_boundary_atlas/phase645_glm4_protocol_trajectory_side_effect_boundary_atlas_confirm.json
results/glm5_phase645_protocol_trajectory_side_effect_boundary_atlas/phase645_deepseek7b_protocol_trajectory_side_effect_boundary_atlas_confirm.json
results/glm5_phase645_protocol_trajectory_side_effect_boundary_atlas/phase645_cross_model_summary.md
```

### 测试原理

本阶段复用 Phase 643 的核心 causal unit：

```text
separator protocol trajectory
component = layer_out
layers = [18, 19]
```

测试模式：

```text
original
inline
to_original_middle_restore
to_original_middle_random
to_original_middle_reverse
remove_from_inline_middle_restore
remove_from_inline_middle_random
remove_from_inline_middle_reverse
```

测试集合扩展为：

```text
target_failure
original_correct
inline_bad
relation_changed
explanation_needed
non_value
```

注意：

```text
relation_changed / explanation_needed / non_value 中的 exact 不是正向成功率，
而是旧 value 吸附、过短回答或协议误迁移风险指标。
```

核心问题是：

```text
同一个 L17-L20 protocol trajectory 是否只是修复 DS7B 的目标失败样本，
还是会把其它任务也强行推入 value short-answer protocol。
```

### 核心结果

#### qwen3

```text
raw_cases = 320
selected_items = 219
target_failure = 26
original_correct = 48
inline_bad = 1
relation_changed = 48
explanation_needed = 48
non_value = 48
```

qwen3 和 DS7B 方向相反。qwen3 的 inline 不是稳定修复，而经常触发 newline / Okay 路径：

```text
target_failure:
original exact = 19/26, newline_top0 = 0/26
inline exact = 0/26, newline_top0 = 15/26
to_original_middle_restore exact = 2/26, newline_top0 = 22/26
```

qwen3 说明：separator protocol trajectory 是模型相关状态，不是通用字符规则。不能把 DS7B 的 inline value protocol 直接推广到 qwen3。

#### GLM4

```text
raw_cases = 320
selected_items = 234
target_failure = 36
original_correct = 48
inline_bad = 6
relation_changed = 48
explanation_needed = 48
non_value = 48
```

GLM4 基本没有 newline 竞争问题，原始、inline 和 patch 之间差异较小：

```text
target_failure:
original exact = 29/36
inline exact = 27/36
to_original_middle_restore exact = 27/36
remove_from_inline_middle_restore exact = 31/36
newline_top0 = 0/36 for all key modes
```

GLM4 结果说明：当前 DS7B 的 protocol trajectory 不是所有模型共享的同构瓶颈。GLM4 更像是已经默认稳定短答或其格式门不依赖同一条 L17-L20 separator trajectory。

#### DS7B

```text
raw_cases = 320
selected_items = 241
target_failure = 48
original_correct = 48
inline_bad = 1
relation_changed = 48
explanation_needed = 48
non_value = 48
```

目标失败样本上，Phase 643 的闭环被复现：

```text
target_failure:
original exact = 12/48, newline_top0 = 34/48
inline exact = 45/48, newline_top0 = 0/48
to_original_middle_restore exact = 45/48, newline_top0 = 0/48
remove_from_inline_middle_restore exact = 14/48, newline_top0 = 30/48
```

原本正确样本上，restore 不是破坏，而是强力压制 newline 并提升短答：

```text
original_correct:
original exact = 8/48, newline_top0 = 36/48
inline exact = 47/48, newline_top0 = 0/48
to_original_middle_restore exact = 48/48, newline_top0 = 0/48
remove_from_inline_middle_restore exact = 14/48, newline_top0 = 31/48
```

这说明 DS7B 中原始模板的 newline / explanation prior 不只影响 target failure，也广泛影响 original_correct 集合。所谓 original_correct 中仍存在大量“表面正确但协议不稳定”的样本。

但是副作用边界非常明显：

```text
relation_changed:
original old_exact = 4/48, newline_top0 = 32/48
inline old_exact = 17/48, newline_top0 = 0/48
to_original_middle_restore old_exact = 17/48, newline_top0 = 0/48
```

说明 relation changed 场景下，inline / restore 会把模型推入旧 value 输出。这是语义边界风险。

```text
explanation_needed:
original old_exact = 0/48
inline old_exact = 44/48
to_original_middle_restore old_exact = 43/48
remove_from_inline_middle_restore old_exact = 0/48
```

说明当任务明确要求 explanation 时，inline / restore 会强烈过度短答，直接输出旧 value。这是任务协议边界风险。

```text
non_value:
original old_exact = 0/48, newline_top0 = 28/48
inline old_exact = 36/48, newline_top0 = 8/48
to_original_middle_restore old_exact = 23/48, newline_top0 = 8/48
remove_from_inline_middle_restore old_exact = 4/48, newline_top0 = 34/48
```

说明当任务要求 yes/no 而不是 category value 时，inline / restore 会错误吸附 category value。这是输出类型边界风险。

### 本阶段结论

Phase 645 是 Phase 644 的关键实证收紧：

```text
DS7B 的 L17-L20 separator protocol trajectory 确实是强因果机制；
它不仅能修复 target failure，也能修复一批 original_correct 中的隐藏协议不稳定样本。
```

但更重要的是：

```text
该 trajectory 不是纯语义恢复器，
而是 value short-answer protocol activator。
```

它的适用边界是：

```text
当任务确实要求直接输出 category value 时，restore 有效；
当任务要求解释、yes/no、或 relation 已改变时，restore 会产生明显副作用。
```

因此 Phase 644 中“可以启动全局图谱测试”的判断仍然成立，但必须附加边界条件：

```text
全局图谱不能只记录 effective patch。
必须同时记录 task boundary、semantic boundary、output-type boundary 和 side-effect boundary。
```

### 理论进展

原公式需要从单一 protocol gate 扩展为条件化协议门：

```text
G_protocol = 1[M_newline > 0]
```

更新为：

```text
G_protocol(B, T, O, R)
```

其中：

```text
B = boundary condition
T = task demand
O = output type
R = relation semantics
```

完整近似从：

```text
P(c_hat = c)
≈ P(G_protocol) * P(G_prefix) * P(G_value) * P(G_confirm)
```

收紧为：

```text
P(c_hat = c | B,T,O,R)
≈
P(G_protocol(B,T,O,R))
* P(G_prefix | G_protocol)
* P(G_value | R,T,O)
* P(G_confirm | context)
```

当前最重要的新洞察：

```text
protocol trajectory 是可复用差分机制的一部分，
但复用不是无条件复用。
同一条状态轨迹在 value task 中是修复器，
在 explanation / non-value / relation-changed task 中会变成副作用源。
```

这说明语言机制不是“找到一个正确方向并注入”，而是：

```text
同一参数骨架根据边界、任务、语义和输出类型选择不同协议态。
智能的关键不只是状态生成，而是状态选择边界。
```

### 问题和硬伤

1. relation_changed 目前只是在小关系集合内替换关系，尚不能证明真实语义关系的完整边界。
2. explanation_needed 的指标把 old_exact 作为风险指标，但还没有建立解释质量评分。
3. non_value 只测试 yes/no 型输出，输出类型边界还应覆盖数字、列表、JSON、自由文本等。
4. inline_bad 样本数量太少，说明该分支不能作为主要统计依据。
5. qwen3 与 DS7B 方向相反，说明跨模型统一理论还不能依赖固定层区间或固定 separator 字符。
6. 当前仍主要测试 layer_out trajectory，还没有完整 writer graph。

### 是否可以开始全局图谱测试

可以开始，但不是无约束开始。

全局图谱的每个机制节点必须至少记录：

```text
1. causal unit
2. trajectory interval
3. sufficiency
4. necessity
5. natural generation closure
6. semantic boundary
7. task boundary
8. output-type boundary
9. side-effect profile
10. cross-model polarity
```

如果只记录“patch 是否有效”，会重新回到单点机制陷阱。

### 下一阶段

Phase 646 应进入：

```text
Global Reuse-Difference Protocol Atlas Schema and First Batch
```

核心任务：

```text
把 value short-answer protocol 作为第一个 atlas node，
用统一表结构记录 DS7B / qwen3 / GLM4 的：
boundary condition
task demand
output type
semantic relation
trajectory interval
patch direction
generation closure
side-effect boundary
cross-model polarity
```

第一批 atlas 不要追求所有机制，而应覆盖三个机制族：

```text
1. value short-answer protocol
2. newline / explanation protocol
3. non-value answer protocol
```

阶段目标：

```text
从“单机制闭环”
进入
“机制图谱节点标准化”。
```

## Phase 646: Global Reuse-Difference Protocol Atlas Schema and First Batch [2026-06-25 22:13]

### 任务背景

用户上传的 Phase 645 评价基本正确。Phase 645 的关键意义不是继续证明 L17-L20 protocol trajectory 可以修复 DS7B 目标失败样本，而是证明该轨迹有清晰边界：

```text
在 value short-answer task 中是修复器；
在 explanation / non-value / relation_changed 中会变成副作用源。
```

因此下一步不能继续只做单点 patch，而应把已有结果整理成机制图谱节点。Phase 646 执行的是 atlas infrastructure，不重新运行 CUDA 模型。

### 本阶段脚本

```text
tests/gpt5/phase646_global_reuse_difference_protocol_atlas.py
```

### 执行命令

```bash
python tests/gpt5/phase646_global_reuse_difference_protocol_atlas.py
python -m py_compile tests/gpt5/phase646_global_reuse_difference_protocol_atlas.py
```

### 输入证据

```text
results/glm5_phase641_separator_protocol_formation_interval_audit/
results/glm5_phase642_endpoint_dominance_vs_distributed_formation/
results/glm5_phase643_protocol_trajectory_natural_generation_closure/
results/glm5_phase645_protocol_trajectory_side_effect_boundary_atlas/
```

### 输出文件

```text
results/glm5_phase646_global_reuse_difference_protocol_atlas/phase646_atlas_index.json
results/glm5_phase646_global_reuse_difference_protocol_atlas/atlas_nodes.jsonl
results/glm5_phase646_global_reuse_difference_protocol_atlas/atlas_edges.jsonl
results/glm5_phase646_global_reuse_difference_protocol_atlas/atlas_evidence.jsonl
results/glm5_phase646_global_reuse_difference_protocol_atlas/atlas_boundary_profiles.jsonl
results/glm5_phase646_global_reuse_difference_protocol_atlas/atlas_boundary_matrix.csv
results/glm5_phase646_global_reuse_difference_protocol_atlas/atlas_schema.json
results/glm5_phase646_global_reuse_difference_protocol_atlas/phase646_atlas_report.md
```

### 生成规模

```text
nodes = 54
edges = 80
boundary_profiles = 18
trajectory_evidence = 41
```

### Atlas 第一批机制节点

```text
mechanism:value_short_answer_protocol
mechanism:newline_explanation_protocol
mechanism:non_value_answer_protocol
```

这三个节点不是最终全局图谱，只是第一批 protocol atlas nodes。它们的目的不是总结全部语言机制，而是把当前最清楚的 value / newline / non-value 协议分化先标准化。

### 边界矩阵核心结果

#### qwen3

```text
target_failure:
original exact/newline = 19/0
to_original_restore exact/newline = 2/22
polarity = harmful_or_opposite_protocol

original_correct:
original exact/newline = 28/0
to_original_restore exact/newline = 8/36
polarity = harmful_or_opposite_protocol
```

qwen3 的结果再次证明：不能把 DS7B 的 separator trajectory 当作通用字符规则。对 qwen3 来说，该轨迹方向更接近反向或有害极性。

#### GLM4

```text
target_failure:
original exact/newline = 29/0
to_original_restore exact/newline = 27/0
polarity = weak_or_neutral

explanation_needed:
original exact/newline = 0/0
to_original_restore exact/newline = 0/0
polarity = boundary_respected_or_neutral
```

GLM4 没有明显 newline 竞争瓶颈，当前 atlas 中应记录为 weak / neutral polarity，而不是强行纳入 DS7B 同构机制。

#### DS7B

```text
target_failure:
original exact/newline = 12/34
to_original_restore exact/newline = 45/0
polarity = beneficial_value_protocol

original_correct:
original exact/newline = 8/36
to_original_restore exact/newline = 48/0
polarity = beneficial_value_protocol

relation_changed:
original old_exact/newline = 4/32
to_original_restore old_exact/newline = 17/0
polarity = side_effect_value_absorption

explanation_needed:
original old_exact/newline = 0/0
to_original_restore old_exact/newline = 43/0
polarity = side_effect_value_absorption

non_value:
original old_exact/newline = 0/28
to_original_restore old_exact/newline = 23/8
polarity = side_effect_value_absorption
```

DS7B 形成当前第一个清楚的 atlas node：

```text
value_short_answer_protocol:
  causal unit = separator boundary
  trajectory interval = L17-L20 / middle L18-L19
  component = layer_out
  generation closure = yes
  side effect = relation_changed / explanation_needed / non_value
  cross-model polarity = DS7B positive, qwen3 opposite, GLM4 weak-neutral
```

### 本阶段进展

Phase 646 把研究从散落的 Phase 结果推进到标准化图谱结构。现在每个机制节点至少需要记录：

```text
causal_unit
trajectory_interval
sufficiency
necessity
generation_closure
semantic_boundary
task_boundary
output_type_boundary
side_effect_profile
cross_model_polarity
```

这使得后续研究可以继续添加 evidence rows，而不是每次重新写一套孤立分析。

### 理论收紧

此前公式：

```text
P(c_hat = c | B,T,O,R)
≈
P(G_protocol(B,T,O,R))
* P(G_prefix | G_protocol)
* P(G_value | R,T,O)
* P(G_confirm | context)
```

Phase 646 后应加入 atlas node 维度：

```text
G_protocol(B,T,O,R; M_i)
```

其中：

```text
M_i = mechanism node in atlas
```

因此机制不再被看成单个向量或单个 patch，而是图谱节点：

```text
M_i =
{
  causal_unit,
  trajectory_interval,
  component,
  task_boundary,
  semantic_boundary,
  output_boundary,
  side_effect_profile,
  model_polarity
}
```

当前理论表述应更新为：

```text
语言能力来自同一参数骨架在不同边界、任务、语义和输出类型条件下生成的状态轨迹。
这些轨迹不是孤立正确状态，而是 atlas nodes。
破解语言编码机制的核心不是找到单个方向，而是建立机制节点之间的条件化图谱。
```

### 问题和硬伤

1. Phase 646 没有新增模型测试，只是结构化已有证据。
2. atlas 当前只有 protocol 机制的第一批节点，知识网络、推理链路、语法系统还没有统一接入。
3. writer graph 仍未完成，value_short_answer_protocol 目前主要停留在 layer_out trajectory 层级。
4. relation_changed 仍只有少量关系类型，语义边界需要扩展。
5. non_value 主要是 yes/no，输出类型边界还需要加入数字、列表、JSON、自由文本。
6. 当前极性分类是基础阈值规则，不是最终理论判断。

### 是否应该自动继续

本阶段目标是完成 atlas schema 和 first batch，已经完成。接下来可以继续自动进入 Phase 647，但 Phase 647 会重新涉及模型测试和 writer graph 拆解，工作量会明显增加。

### 下一阶段

Phase 647 应执行：

```text
Protocol Writer Graph Audit
```

目标：

```text
把 atlas 中的 value_short_answer_protocol 节点从 layer_out trajectory 继续拆到 attention / MLP / residual update writer。
```

具体任务：

```text
1. 固定 DS7B value_short_answer_protocol 目标样本。
2. 对 L17-L20 / L18-L19 进行 writer attribution。
3. 拆分 attention output、MLP update、residual carried state。
4. 测试 writer-level sufficiency / necessity。
5. 把结果追加到 Phase646 atlas，而不是另起孤立结论。
```

阶段目标：

```text
从 protocol trajectory atlas node
推进到
protocol writer graph node。
```

## Phase 647: Protocol Writer Graph Audit [2026-06-25 22:53]

### 任务背景

用户上传的 Phase 646 评价基本正确：Phase 646 的核心意义不是新因果实验，而是把 Phase 641 到 Phase 645 的局部闭环整理为 atlas node。它的保守边界也正确：当前图谱主要是 protocol atlas，writer graph 仍缺失。

因此本阶段继续执行 Phase 647：

```text
Protocol Writer Graph Audit
```

目标是把 atlas 中的：

```text
mechanism:value_short_answer_protocol
```

从：

```text
layer_out trajectory
```

进一步拆成：

```text
layer_input
attn_out
mlp_out
layer_out
```

并测试：

```text
to_original = sufficiency
remove_from_inline = necessity
```

### 本阶段脚本

```text
tests/gpt5/phase647_protocol_writer_graph_audit.py
tests/gpt5/phase647_protocol_writer_graph_audit_summary.py
tests/gpt5/phase647_protocol_writer_graph_atlas_update.py
```

### 执行命令

smoke：

```bash
python -m py_compile tests/gpt5/phase647_protocol_writer_graph_audit.py tests/gpt5/phase647_protocol_writer_graph_audit_summary.py
python tests/gpt5/phase647_protocol_writer_graph_audit.py qwen3 --smoke --hard-exit-after-model
python tests/gpt5/phase647_protocol_writer_graph_audit.py qwen3 --smoke --include-nontarget --hard-exit-after-model
```

正式三模型顺序测试：

```bash
python tests/gpt5/phase647_protocol_writer_graph_audit.py qwen3 --confirm --hard-exit-after-model
python tests/gpt5/phase647_protocol_writer_graph_audit.py glm4 --confirm --hard-exit-after-model
python tests/gpt5/phase647_protocol_writer_graph_audit.py deepseek7b --confirm --hard-exit-after-model
python tests/gpt5/phase647_protocol_writer_graph_audit_summary.py
python tests/gpt5/phase647_protocol_writer_graph_atlas_update.py
python -m py_compile tests/gpt5/phase647_protocol_writer_graph_atlas_update.py
```

### 输出文件

```text
results/glm5_phase647_protocol_writer_graph_audit/phase647_qwen3_protocol_writer_graph_audit_confirm.json
results/glm5_phase647_protocol_writer_graph_audit/phase647_glm4_protocol_writer_graph_audit_confirm.json
results/glm5_phase647_protocol_writer_graph_audit/phase647_deepseek7b_protocol_writer_graph_audit_confirm.json
results/glm5_phase647_protocol_writer_graph_audit/phase647_cross_model_summary.md
results/glm5_phase647_protocol_writer_graph_audit/phase647_writer_graph_atlas_update.md
results/glm5_phase647_protocol_writer_graph_audit/phase647_writer_graph_nodes.jsonl
results/glm5_phase647_protocol_writer_graph_audit/phase647_writer_graph_edges.jsonl
results/glm5_phase647_protocol_writer_graph_audit/phase647_writer_graph_evidence.jsonl
```

atlas update 规模：

```text
writer_candidate_nodes = 72
writer_edges = 72
writer_evidence_rows = 72
```

### 测试原理

在 target failure cases 上，对 separator boundary 位置的 L17-L20 轨迹进行组件级 patch：

```text
components = layer_input, attn_out, mlp_out, layer_out
layers = 17,18,19,20
intervals = L17-L20, L18-L19
```

每个 patch 分两类：

```text
to_original:
  把 inline 的组件状态写入 original。
  如果 exact 上升、newline 降低，说明该组件/位置具有 sufficiency。

remove_from_inline:
  把 original 的组件状态写入 inline。
  如果 exact 下降、newline 上升，说明该组件/位置具有 necessity。
```

核心观察不只是 exact，还包括：

```text
tok0_hit
newline_top0
mean_prefix_rank
generation_text
```

### 三模型总体结果

#### qwen3

```text
raw_cases = 320
target_seen = 26
cases_written = 26
mode_rows = 2444
```

baseline：

```text
original exact = 19/26, newline = 0/26
inline exact = 0/26, newline = 15/26
```

qwen3 再次证明它和 DS7B 极性相反：inline 本身是坏协议。

主要 sufficiency：

```text
to_original_interval_L18_19_attn_out_restore:
  exact = 21/26
  newline = 0/26

to_original_L19_mlp_out_restore:
  exact = 20/26
  newline = 0/26

to_original_L20_attn_out_restore:
  exact = 20/26
  newline = 0/26
```

主要 necessity：

```text
remove_from_inline_interval_L18_19_mlp_out_restore:
  exact = 0/26
  newline = 26/26

remove_from_inline_L19_mlp_out_restore:
  exact = 0/26
  newline = 26/26
```

但由于 qwen3 的 inline 本身是坏协议，所以这里不能解释为 DS7B 同构机制，只能记为：

```text
qwen3 opposite-polarity protocol writer pattern
```

#### GLM4

```text
raw_cases = 320
target_seen = 36
cases_written = 36
mode_rows = 3384
```

baseline：

```text
original exact = 29/36, newline = 0/36
inline exact = 27/36, newline = 0/36
```

GLM4 没有 newline bottleneck。

主要 sufficiency：

```text
to_original_interval_L18_19_attn_out_restore:
  exact = 30/36
  newline = 0/36

to_original_L19_attn_out_restore:
  exact = 29/36
  newline = 0/36

to_original_L18_attn_out_restore:
  exact = 29/36
  newline = 0/36
```

主要 necessity：

```text
remove_from_inline_interval_L17_20_mlp_out_restore:
  exact = 10/36
  newline = 0/36

remove_from_inline_interval_L17_20_attn_out_restore:
  exact = 17/36
  newline = 0/36
```

GLM4 的结果说明：有 writer-level 可影响读出，但不是 DS7B 那种 newline gate。它更像 value / label ranking 改变，而不是 protocol newline switch。

#### DS7B

```text
raw_cases = 320
target_seen = 48
cases_written = 48
mode_rows = 4512
```

baseline：

```text
original exact = 12/48, newline = 34/48
inline exact = 45/48, newline = 0/48
```

DS7B 的 sufficiency 最强结果：

```text
to_original_L17_layer_input_restore:
  exact = 46/48
  newline = 0/48

to_original_L18_layer_out_restore:
  exact = 46/48
  newline = 0/48

to_original_L19_layer_input_restore:
  exact = 46/48
  newline = 0/48

to_original_L17_layer_out_restore:
  exact = 46/48
  newline = 0/48

to_original_L18_layer_input_restore:
  exact = 46/48
  newline = 0/48

to_original_interval_L18_19_layer_out_restore:
  exact = 45/48
  newline = 0/48

to_original_interval_L17_20_layer_out_restore:
  exact = 43/48
  newline = 0/48
```

DS7B 的 necessity 最强结果：

```text
remove_from_inline_interval_L17_20_layer_out_restore:
  exact = 12/48
  newline = 35/48

remove_from_inline_L20_layer_out_restore:
  exact = 12/48
  newline = 35/48

remove_from_inline_interval_L18_19_layer_out_restore:
  exact = 14/48
  newline = 30/48

remove_from_inline_L19_layer_out_restore:
  exact = 14/48
  newline = 30/48

remove_from_inline_L20_layer_input_restore:
  exact = 14/48
  newline = 30/48
```

组件级结果：

```text
to_original_interval_L17_20_mlp_out_restore:
  exact = 33/48
  newline = 4/48

to_original_interval_L17_20_attn_out_restore:
  exact = 0/48
  newline = 17/48

remove_from_inline_interval_L17_20_attn_out_restore:
  exact = 8/48
  newline = 0/48

remove_from_inline_interval_L17_20_mlp_out_restore:
  exact = 22/48
  newline = 22/48
```

### 核心结论

Phase 647 是一个关键正结果，但结论必须谨慎。

DS7B 中最强 writer graph 证据不是：

```text
某一个 attention 或 MLP 组件单独写入完整 value short-answer protocol。
```

而是：

```text
协议态已经作为 residual carried state 在 layer_input / layer_out 之间传播。
```

特别是：

```text
L17 layer_input
L17 layer_out
L18 layer_input
L18 layer_out
L19 layer_input
L19 layer_out
L20 layer_input / layer_out
```

形成了一个高度连续的 residual protocol carrier chain。

这说明：

```text
value_short_answer_protocol 不是单层 writer；
它是已经成形的状态轨迹，在 L17-L20 的残差流中被携带和维护。
```

MLP 与 attention 的角色不对称：

```text
MLP 在部分 sufficiency 上有效，尤其 L17-L20 interval mlp_out 可达 33/48；
但不能单独解释完整闭合。

attention interval 在 DS7B to_original 中反而失败，说明注意力输出不是简单可移植的完整协议态；
但 remove_from_inline L17-L20 attn_out 会把 exact 降到 8/48，说明 attention 对 inline 轨道中的 value support / pattern stability 仍有必要性影响。
```

因此最新图谱节点应更新为：

```text
mechanism:value_short_answer_protocol
  primary carrier = residual stream layer_input/layer_out chain
  partial writers = MLP updates and attention updates
  strongest interval = L17-L20, especially L18-L19 / L20 readout boundary
  natural closure = yes
  side-effect boundary = yes
```

### 对 Phase 646 的修正

Phase 646 中说：

```text
trajectory_interval = L17-L20 / L18-L19
component = layer_out
```

Phase 647 后应收紧为：

```text
trajectory_interval = L17-L20
carrier = residual stream chain
observable states = layer_input and layer_out
partial component contribution = MLP / attention
not yet closed = exact writer decomposition
```

### 理论进展

Phase 646 的机制节点公式是：

```text
M_i =
{
  causal_unit,
  trajectory_interval,
  component,
  task_boundary,
  semantic_boundary,
  output_boundary,
  side_effect_profile,
  model_polarity
}
```

Phase 647 后需要加入 writer graph：

```text
M_i =
{
  causal_unit,
  trajectory_interval,
  carrier_chain,
  writer_candidates,
  sufficiency_profile,
  necessity_profile,
  task_boundary,
  semantic_boundary,
  output_boundary,
  side_effect_profile,
  model_polarity
}
```

对应到当前 DS7B：

```text
carrier_chain =
[
  L17.layer_input,
  L17.layer_out,
  L18.layer_input,
  L18.layer_out,
  L19.layer_input,
  L19.layer_out,
  L20.layer_input,
  L20.layer_out
]
```

```text
writer_candidates =
[
  L17-L20.mlp_out,
  L17-L20.attn_out,
  L18-L19.layer_out,
  L20.layer_out
]
```

但注意：

```text
writer_candidates != fully isolated writers
```

它们目前只是 candidate graph nodes，不是完整机制闭合。

### 问题和硬伤

1. `layer_input` 强效说明协议态已经在进入该层前形成，不能把该层称为唯一写入源。
2. `layer_out` 强效可能包含多个子组件和残差携带，仍不是纯 writer。
3. `attn_out` 和 `mlp_out` 出现方向不对称，说明组件 patch 可能扰动了组合状态，不能直接线性解释。
4. qwen3 是反向极性，GLM4 是弱/中性极性，因此跨模型统一对象仍应是功能图谱，而不是固定层号。
5. 当前只测试 separator 位置，没有把 writer graph 扩展到 answer label、prompt_last、question tail 等多位置。
6. 当前只在 target failure 上做 writer graph，还没有重新测试 side-effect writer 是否同源。

### 是否应该继续自动完成下一步

本阶段目标已经完成：

```text
从 protocol trajectory atlas node
推进到 protocol writer graph candidate node。
```

但下一步 Phase 648 会继续扩大到 multi-position writer graph，需要较大模型测试。若继续自动完成，应把重点从“单 separator writer”扩展到：

```text
separator
answer_label
prompt_last
question_mark_answer
relation_tail
```

### 下一阶段

Phase 648 应执行：

```text
Multi-Position Protocol Writer Graph Audit
```

目标：

```text
检查 DS7B 的 value_short_answer_protocol carrier chain 是否只在 separator boundary，
还是由 separator + answer_label + prompt_last 等多个边界位置共同维持。
```

测试应覆盖：

```text
positions:
  separator
  answer_label
  prompt_last
  question_mark_answer
  relation_tail

components:
  layer_input
  attn_out
  mlp_out
  layer_out

directions:
  to_original
  remove_from_inline

metrics:
  exact
  newline_top0
  prefix_rank
  natural generation text
```

阶段目标：

```text
从 single-boundary writer candidate graph
推进到 multi-position protocol writer graph。
```

## Phase 648: Multi-Position Protocol Writer Graph Audit [2026-06-25 23:40]

### 本阶段问题

本阶段分析了用户上传的 Phase 647 复核内容。复核内容基本正确，而且比 Phase 647 原始结论更严谨：

```text
Phase 647 不是找到了完整 writer，
而是把 value_short_answer_protocol 从 layer_out trajectory
推进到 residual carrier chain + writer candidate graph。
```

因此 Phase 648 继续完成下一步：

```text
检查 value_short_answer_protocol 是否只是 separator boundary 的局部现象，
还是在 prompt_last、question_mark_answer、relation_tail 等多个边界位置共同出现。
```

### 新增脚本

```text
tests/gpt5/phase648_multi_position_protocol_writer_graph_audit.py
tests/gpt5/phase648_multi_position_protocol_writer_graph_audit_summary.py
```

结果目录：

```text
results/glm5_phase648_multi_position_protocol_writer_graph_audit/
```

主要结果文件：

```text
phase648_qwen3_multi_position_protocol_writer_graph_audit_confirm.json
phase648_glm4_multi_position_protocol_writer_graph_audit_confirm.json
phase648_deepseek7b_multi_position_protocol_writer_graph_audit_confirm.json
phase648_cross_model_summary.md
```

### 执行命令

语法检查：

```bash
python -m py_compile tests/gpt5/phase648_multi_position_protocol_writer_graph_audit.py tests/gpt5/phase648_multi_position_protocol_writer_graph_audit_summary.py
```

smoke test：

```bash
python tests/gpt5/phase648_multi_position_protocol_writer_graph_audit.py qwen3 --smoke --include-nontarget --hard-exit-after-model
```

正式测试，三模型顺序执行：

```bash
python tests/gpt5/phase648_multi_position_protocol_writer_graph_audit.py qwen3 --confirm --hard-exit-after-model
python tests/gpt5/phase648_multi_position_protocol_writer_graph_audit.py glm4 --confirm --hard-exit-after-model
python tests/gpt5/phase648_multi_position_protocol_writer_graph_audit.py deepseek7b --confirm --hard-exit-after-model
```

汇总：

```bash
python tests/gpt5/phase648_multi_position_protocol_writer_graph_audit_summary.py
```

### 测试原理

Phase 647 只在 separator boundary 做 writer candidate audit。Phase 648 加入 position 维度：

```text
separator
answer_label
prompt_last
question_mark_answer
relation_tail
```

对每个位置测试两类方向：

```text
to_original:
  inline prompt 的状态 patch 到 original prompt，
  测充分性。

remove_from_inline:
  original prompt 的状态 patch 到 inline prompt，
  测必要性。
```

测试组件：

```text
single-layer components:
  layer_input
  layer_out

interval components:
  attn_out
  mlp_out
  layer_out
```

测试层区间：

```text
L17-L20
L18-L19
```

评价指标：

```text
exact
tok0_hit
newline_top0
prefix_rank
generation_text
```

### 客观结果

#### qwen3

```text
raw_cases = 320
target_seen = 26
cases_written = 26
mode_rows = 2964
total_time_min = 6.58
```

baseline：

```text
original exact = 19/26, newline = 0/26
inline   exact = 0/26,  newline = 15/26
```

最强充分性：

```text
question_mark_answer_to_original_interval_L17_20_attn_out_restore:
  exact = 23/26
  newline = 0/26

prompt_last_to_original_interval_L18_19_attn_out_restore:
  exact = 23/26
  newline = 0/26

separator_to_original_interval_L18_19_attn_out_restore:
  exact = 21/26
  newline = 0/26
```

最强必要性：

```text
separator_remove_from_inline_interval_L18_19_mlp_out_restore:
  exact = 0/26
  newline = 26/26

prompt_last_remove_from_inline_L17_layer_input_restore:
  exact = 0/26
  newline = 14/26
```

qwen3 的结果与 Phase 647 一致，仍然呈现与 DS7B 不同的极性结构：它不是 DS7B 式的 value short-answer protocol 正向修复模型，而是表现出更强的 newline / reasoning format 竞争。

#### GLM4

```text
raw_cases = 320
target_seen = 36
cases_written = 36
mode_rows = 4104
total_time_min = 10.41
```

baseline：

```text
original exact = 29/36, newline = 0/36
inline   exact = 27/36, newline = 0/36
```

最强充分性：

```text
question_mark_answer_to_original_interval_L18_19_attn_out_restore:
  exact = 34/36
  newline = 0/36

prompt_last_to_original_interval_L18_19_attn_out_restore:
  exact = 30/36
  newline = 0/36

separator_to_original_interval_L18_19_attn_out_restore:
  exact = 30/36
  newline = 0/36
```

最强必要性：

```text
relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore:
  exact = 0/36
  newline = 0/36

question_mark_answer_remove_from_inline_interval_L17_20_mlp_out_restore:
  exact = 2/36
  newline = 0/36
```

GLM4 仍然不是 newline gate 模型。它的变化主要体现为 value token / explanation / word 输出竞争，不是 DS7B 的换行格式门。

#### DS7B

```text
raw_cases = 320
target_seen = 49
cases_written = 48
mode_rows = 5472
total_time_min = 13.09
```

baseline：

```text
original exact = 12/48, newline = 34/48
inline   exact = 45/48, newline = 0/48
```

最强充分性：

```text
separator_to_original_L17_layer_input_restore:
  exact = 46/48
  newline = 0/48

separator_to_original_L18_layer_out_restore:
  exact = 46/48
  newline = 0/48

separator_to_original_L19_layer_input_restore:
  exact = 46/48
  newline = 0/48

prompt_last_to_original_L17_layer_out_restore:
  exact = 46/48
  newline = 0/48

question_mark_answer_to_original_interval_L18_19_layer_out_restore:
  exact = 45/48
  newline = 0/48

relation_tail_to_original_interval_L18_19_layer_out_restore:
  exact = 45/48
  newline = 0/48
```

最强必要性：

```text
relation_tail_remove_from_inline_interval_L17_20_attn_out_restore:
  exact = 1/48
  newline = 0/48

relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore:
  exact = 4/48
  newline = 39/48

prompt_last_remove_from_inline_interval_L17_20_attn_out_restore:
  exact = 5/48
  newline = 2/48

question_mark_answer_remove_from_inline_interval_L17_20_attn_out_restore:
  exact = 5/48
  newline = 0/48

relation_tail_remove_from_inline_interval_L17_20_layer_out_restore:
  exact = 7/48
  newline = 34/48

separator_remove_from_inline_interval_L17_20_layer_out_restore:
  exact = 12/48
  newline = 35/48
```

### 关键进展

Phase 648 证明：

```text
DS7B 的 value_short_answer_protocol 不是 separator-only 机制。
```

更准确的结构是：

```text
separator:
  强 residual carrier，尤其 L17 layer_input / L18 layer_out / L19 layer_input。

prompt_last:
  也能强恢复，说明最终提示边界参与协议态承载。

question_mark_answer:
  L18-L19 layer_out 强恢复，说明问号到答案标签之间的局部格式跨度参与协议态。

relation_tail:
  L18-L19 layer_out 强恢复，且 remove_from_inline 的 L17-L20 attn_out / mlp_out 非常强，
  说明关系尾部到答案边界不是语义尾巴，而是协议场的一部分。
```

因此，Phase 647 的图谱需要从：

```text
single-boundary residual carrier chain
```

升级为：

```text
multi-position protocol field
```

### 重要硬伤

1. `answer_label` 没有被有效测试。原因是 original prompt 和 inline prompt 中 `Answer:` 的 token span 长度不一致，全部进入 `position_len_mismatch`。因此本阶段不能声称 answer_label 独立位置已经被覆盖。

2. qwen3 / GLM4 / DS7B 仍不能用统一层号解释。跨模型统一对象应是 functional atlas node，而不是固定 L17-L20。

3. DS7B 中 `relation_tail_remove_from_inline_interval_L17_20_attn_out_restore` 使 exact 降到 1/48，但 newline 仍为 0/48。这说明它破坏的不是简单 newline gate，而可能是 value support / token choice / format-preference 的混合状态。

4. `relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore` 同时 exact 降到 4/48、newline 升到 39/48，说明 MLP 对格式门更直接，但仍不能单独解释所有输出竞争。

5. 本阶段没有随机 / reverse control，因为 Phase 648 主要目标是 position expansion。严格因果闭合仍需要下一阶段加 control。

### 理论更新

Phase 647 的机制节点：

```text
M_i =
{
  causal_unit,
  trajectory_interval,
  carrier_chain,
  writer_candidates,
  sufficiency_profile,
  necessity_profile,
  task_boundary,
  semantic_boundary,
  output_boundary,
  side_effect_profile,
  model_polarity
}
```

Phase 648 后应扩展为：

```text
M_i =
{
  causal_unit_set,
  position_field,
  trajectory_interval,
  carrier_chain,
  writer_candidate_graph,
  sufficiency_profile,
  necessity_profile,
  task_boundary,
  semantic_boundary,
  output_boundary,
  side_effect_profile,
  model_polarity
}
```

其中 DS7B 的当前 position_field：

```text
position_field =
{
  separator,
  prompt_last,
  question_mark_answer,
  relation_tail
}
```

暂时不能包含：

```text
answer_label
```

因为 answer_label 的 tokenization alignment 失败。

更具体地：

```text
value_short_answer_protocol_DS7B =
  residual_carrier(separator, L17-L20)
  + residual_carrier(prompt_last, L17-L18)
  + layer_out_carrier(question_mark_answer, L18-L19)
  + layer_out_carrier(relation_tail, L18-L19)
  + attn/mlp necessity(relation_tail, L17-L20)
```

### 当前结论

Phase 648 是关键正结果，但需要收紧：

```text
正确：
  DS7B 的 value_short_answer_protocol 是多位置协议场，
  不是单 separator 局部机制。

不能说：
  已经完成完整 writer graph。

不能说：
  answer_label 已经独立验证。

不能说：
  attn_out / mlp_out 是纯 writer。
```

### 下一阶段

Phase 649 应执行：

```text
Answer-Label Tokenization Alignment and Protocol Field Control Audit
```

目标：

```text
1. 修正 answer_label 的 token span 对齐问题。
2. 对 Phase 648 中最强位置加入 random / reverse control。
3. 区分 relation_tail 的 attention 破坏到底是 value support 破坏，还是 format gate 破坏。
```

建议测试对象：

```text
positions:
  answer_word
  colon
  answer_colon
  answer_label_aligned
  separator
  prompt_last
  question_mark_answer
  relation_tail

components:
  layer_input
  layer_out
  attn_out
  mlp_out

intervals:
  L17-L20
  L18-L19

controls:
  restore
  random
  reverse
```

阶段目标：

```text
从 multi-position protocol field
推进到 aligned protocol field with controls。
```

## Phase 649: Answer-Label Alignment and Protocol Field Control Audit [2026-06-26 01:00]

### 本阶段问题

用户上传的 Phase 648 复核内容基本正确：

```text
Phase 648 的关键结论是：
DS7B 的 value_short_answer_protocol 不是 separator-only，
而是 prompt tail multi-position protocol field。
```

但复核内容指出三个必须修正的问题：

```text
1. answer_label 未被有效测试。
2. Phase 648 缺少 random / reverse control。
3. relation_tail 的 attention / MLP 作用混合，需要继续拆分。
```

因此 Phase 649 的目标是：

```text
修正 answer_label tokenization alignment，
并对最强 protocol field 候选加入 restore / random / reverse controls。
```

### 新增脚本

```text
tests/gpt5/phase649_answer_label_alignment_protocol_field_control_audit.py
tests/gpt5/phase649_answer_label_alignment_protocol_field_control_audit_summary.py
```

结果目录：

```text
results/glm5_phase649_answer_label_alignment_protocol_field_control_audit/
```

主要结果文件：

```text
phase649_qwen3_answer_label_alignment_protocol_field_control_audit_confirm.json
phase649_glm4_answer_label_alignment_protocol_field_control_audit_confirm.json
phase649_deepseek7b_answer_label_alignment_protocol_field_control_audit_confirm.json
phase649_cross_model_summary.md
```

### 执行命令

语法检查：

```bash
python -m py_compile tests/gpt5/phase649_answer_label_alignment_protocol_field_control_audit.py tests/gpt5/phase649_answer_label_alignment_protocol_field_control_audit_summary.py
```

smoke test：

```bash
python tests/gpt5/phase649_answer_label_alignment_protocol_field_control_audit.py qwen3 --smoke --include-nontarget --hard-exit-after-model
```

正式测试：

```bash
python tests/gpt5/phase649_answer_label_alignment_protocol_field_control_audit.py qwen3 --confirm --hard-exit-after-model
python tests/gpt5/phase649_answer_label_alignment_protocol_field_control_audit.py glm4 --confirm --hard-exit-after-model
python tests/gpt5/phase649_answer_label_alignment_protocol_field_control_audit.py deepseek7b --confirm --hard-exit-after-model
```

汇总：

```bash
python tests/gpt5/phase649_answer_label_alignment_protocol_field_control_audit_summary.py
```

### 测试原理

Phase 648 中直接用 `"Answer:"` 做 token span，导致 original prompt 的 `"\nAnswer:"` 和 inline prompt 的 `" Answer:"` 被 tokenizer 切成不同长度，answer_label 进入 `position_len_mismatch`。

Phase 649 改为以冒号为锚点：

```text
colon = final ":"
answer_word = colon - 1
answer_colon = answer_word + colon
answer_label_aligned = answer_word + colon
```

这样不再让前面的 newline / space separator 影响 answer label 对齐。

测试位置：

```text
answer_word
colon
answer_colon
answer_label_aligned
separator
prompt_last
question_mark_answer
relation_tail
```

测试方向：

```text
to_original:
  inline -> original，测试充分性。

remove_from_inline:
  original -> inline，测试必要性。
```

关键 interval/component：

```text
L17-L20.attn_out
L17-L20.mlp_out
L17-L20.layer_out
L18-L19.layer_out
L17.layer_input
L17.layer_out
```

对 interval modes 加入：

```text
restore
random
reverse
```

### 对齐修复是否成功

smoke test 后：

```text
position_len_mismatch = 0
answer_word / colon / answer_colon / answer_label_aligned / separator / prompt_last / question_mark_answer / relation_tail
全部产生结果。
```

正式三模型也全部为：

```text
position_missing = 0
position_len_mismatch = 0
empty_patch = 0
```

因此 Phase 648 最大硬伤已经修复：

```text
answer_label_aligned 可以被有效测试。
```

### 客观结果

#### qwen3

```text
raw_cases = 320
target_seen = 26
cases_written = 26
mode_rows = 5876
total_time_min = 12.05
filtered = {
  not_target: 294,
  position_missing: 0,
  position_len_mismatch: 0,
  empty_patch: 0,
  case_cap: 0
}
```

baseline：

```text
original exact = 19/26, newline = 0/26
inline   exact = 0/26,  newline = 15/26
```

重要结果：

```text
answer_word_to_original_interval_L17_20_attn_out_restore:
  exact = 25/26
  newline = 0/26

answer_label_aligned_to_original_interval_L17_20_attn_out_restore:
  exact = 20/26
  newline = 0/26

answer_word_remove_from_inline_interval_L17_20_mlp_out_restore:
  exact = 0/26
  newline = 15/26

answer_label_aligned_remove_from_inline_interval_L17_20_attn_out_restore:
  exact = 7/26
  newline = 7/26
```

qwen3 仍然不是 DS7B 式正向协议模型，但 answer-aligned 区域确实有强影响。

#### GLM4

```text
raw_cases = 320
target_seen = 36
cases_written = 36
mode_rows = 8136
total_time_min = 19.16
filtered = {
  not_target: 284,
  position_missing: 0,
  position_len_mismatch: 0,
  empty_patch: 0,
  case_cap: 0
}
```

baseline：

```text
original exact = 29/36, newline = 0/36
inline   exact = 27/36, newline = 0/36
```

重要结果：

```text
answer_word_to_original_interval_L18_19_layer_out_restore:
  exact = 30/36
  newline = 0/36

answer_label_aligned_to_original_L17_layer_input_restore:
  exact = 28/36
  newline = 0/36

answer_label_aligned_remove_from_inline_interval_L17_20_mlp_out_restore:
  exact = 10/36
  newline = 0/36

relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore:
  exact = 0/36
  newline = 0/36
```

GLM4 仍然不是 newline gate 模型，主要表现为 value / word / explanation 竞争。

#### DS7B

```text
raw_cases = 320
target_seen = 49
cases_written = 48
mode_rows = 10848
total_time_min = 25.50
filtered = {
  not_target: 88,
  position_missing: 0,
  position_len_mismatch: 0,
  empty_patch: 0,
  case_cap: 1
}
```

baseline 延续 Phase 648：

```text
original exact = 12/48, newline = 34/48
inline   exact = 45/48, newline = 0/48
```

最关键充分性：

```text
answer_colon_to_original_L17_layer_input_restore:
  exact = 46/48
  newline = 0/48

answer_label_aligned_to_original_L17_layer_input_restore:
  exact = 46/48
  newline = 0/48

separator_to_original_L17_layer_input_restore:
  exact = 46/48
  newline = 0/48

answer_colon_to_original_L17_layer_out_restore:
  exact = 46/48
  newline = 0/48

answer_label_aligned_to_original_L17_layer_out_restore:
  exact = 46/48
  newline = 0/48

separator_to_original_L17_layer_out_restore:
  exact = 46/48
  newline = 0/48
```

这说明：

```text
answer_label_aligned 与 separator 几乎同强。
```

最关键必要性：

```text
answer_label_aligned_remove_from_inline_interval_L17_20_attn_out_restore:
  exact = 8/48
  newline = 0/48
  top0 = word:40, correct_prefix:8

answer_label_aligned_remove_from_inline_interval_L17_20_layer_out_restore:
  exact = 12/48
  newline = 35/48

answer_label_aligned_remove_from_inline_interval_L17_20_mlp_out_restore:
  exact = 22/48
  newline = 22/48

relation_tail_remove_from_inline_interval_L17_20_attn_out_restore:
  exact = 1/48
  newline = 0/48
  top0 = word:44, correct_prefix:4

relation_tail_remove_from_inline_interval_L17_20_mlp_out_restore:
  exact = 4/48
  newline = 39/48
```

random / reverse control：

```text
answer_label_aligned_remove_from_inline_interval_L18_19_layer_out_random:
  exact = 45/48
  newline = 0/48

answer_label_aligned_remove_from_inline_interval_L18_19_layer_out_reverse:
  exact = 45/48
  newline = 0/48
```

这说明最强 remove_from_inline 的 collapse 不是任意随机扰动或简单反向扰动能复现的。

### 关键进展

Phase 649 明确补上 Phase 648 的最大缺口：

```text
answer_label_aligned 是 DS7B value_short_answer_protocol 的强因果位置。
```

当前 DS7B 的 prompt tail protocol field 应更新为：

```text
position_field =
{
  answer_label_aligned,
  answer_colon,
  separator,
  prompt_last,
  question_mark_answer,
  relation_tail
}
```

其中：

```text
answer_label_aligned / answer_colon / separator:
  是同等级强协议承载点。

question_mark_answer / relation_tail:
  是更宽的 prompt-tail protocol field 区域。

answer_word:
  单独不稳定，不能作为完整 answer_label 的替代。

colon:
  比 answer_word 更强，但仍不等同于完整 answer_colon。
```

### relation_tail 的进一步拆分

Phase 649 支持 Phase 648 的判断：

```text
relation_tail attention:
  破坏 exact，但不主要触发 newline。

relation_tail MLP:
  破坏 exact，同时强触发 newline。
```

更准确：

```text
relation_tail.attn_out:
  更像 value/token-choice support 或 content-selection support。

relation_tail.mlp_out:
  更像 format/newline gate support。
```

这说明 relation_tail 不是单纯语义尾部，而是语义支持与格式协议共同耦合的位置。

### 理论更新

Phase 648：

```text
M_i =
{
  causal_unit_set,
  position_field,
  trajectory_interval,
  carrier_chain,
  writer_candidate_graph,
  sufficiency_profile,
  necessity_profile,
  task_boundary,
  semantic_boundary,
  output_boundary,
  side_effect_profile,
  model_polarity
}
```

Phase 649 后加入 alignment 和 control profile：

```text
M_i =
{
  causal_unit_set,
  aligned_position_field,
  tokenization_alignment_rule,
  trajectory_interval,
  carrier_chain,
  writer_candidate_graph,
  sufficiency_profile,
  necessity_profile,
  control_profile,
  task_boundary,
  semantic_boundary,
  output_boundary,
  side_effect_profile,
  model_polarity
}
```

DS7B 当前节点：

```text
value_short_answer_protocol_DS7B =
  aligned_carrier(answer_label_aligned, L17.layer_input/out)
  + aligned_carrier(answer_colon, L17.layer_input/out)
  + residual_carrier(separator, L17.layer_input/out)
  + residual_carrier(prompt_last, L17.layer_out)
  + layer_out_carrier(question_mark_answer, L18-L19)
  + layer_out_carrier(relation_tail, L18-L19)
  + attention_necessity(relation_tail, L17-L20, value/token-choice)
  + mlp_necessity(relation_tail, L17-L20, newline/format)
```

### 问题和硬伤

1. Phase 649 修复了 answer_label 对齐，但采用的是 `colon-1` 和 `colon` 的逻辑对齐，仍需要在更多模板中验证是否总是稳定。

2. random / reverse control 不是完整反事实对照。它能排除“任意扰动都有效”，但不能证明唯一 causal writer。

3. `answer_word` 单独在 DS7B 上不强，说明 answer label 的机制单位更接近 answer_colon / aligned answer label，而不是孤立 answer token。

4. qwen3 的 control 中仍出现一些 reverse / random 的高分现象，说明 qwen3 的机制极性和 DS7B 不同，不能用 DS7B 图谱硬套。

5. 仍然主要测试 target_failure 集合，还没有把 aligned answer_label 放进 side-effect boundary tasks。

### 当前结论

Phase 649 是关键正结果：

```text
answer_label_aligned 被重新找回，
并且在 DS7B 中与 separator 同等级强。
```

但结论必须收紧：

```text
已经证明：
  answer_colon / answer_label_aligned 是 protocol field 的强位置。

尚未证明：
  answer_label 是唯一 writer。

已经证明：
  relation_tail 的 attention 和 MLP 有可分离趋势。

尚未证明：
  attention = 语义，MLP = 格式。
```

### 下一阶段

Phase 650 应执行：

```text
Protocol Field Side-Effect Boundary and Template Generalization Audit
```

目标：

```text
1. 把 answer_label_aligned / answer_colon / separator / relation_tail 放入 side-effect tasks。
2. 测 explanation_needed、non_value、relation_changed 中这些位置是否同源。
3. 增加模板变化，确认 colon-1 / colon 对齐规则不是当前模板偶然现象。
4. 建立 protocol field 的 side-effect boundary profile。
```

阶段目标：

```text
从 aligned protocol field with controls
推进到 template-general and side-effect-aware protocol field atlas node。
```

## Phase 650: Protocol Field Template and Side-Effect Audit [2026-06-26 02:29]

### 任务来源

用户要求分析 Phase 649 是否正确，并综合当前进展继续完成任务。Phase 649 已经证明 DS7B 中 `answer_label_aligned`、`answer_colon`、`separator` 是强 protocol field 位置，但仍有三个硬伤：

```text
1. 只在 Answer 模板附近验证，缺少模板泛化。
2. 主要验证 target_failure，缺少 side-effect boundary。
3. relation_tail / label / separator 的作用边界仍混在一起。
```

因此 Phase 650 直接执行：

```text
Protocol Field Template and Side-Effect Audit
```

### 生成脚本

新增正式测试脚本：

```bash
tests/gpt5/phase650_protocol_field_template_side_effect_audit.py
```

新增汇总脚本：

```bash
tests/gpt5/phase650_protocol_field_template_side_effect_audit_summary.py
```

结果目录：

```bash
results/glm5_phase650_protocol_field_template_side_effect_audit/
```

跨模型汇总：

```bash
results/glm5_phase650_protocol_field_template_side_effect_audit/phase650_cross_model_summary.md
```

### 执行命令

脚本编译检查：

```bash
python -m py_compile \
  tests/gpt5/phase650_protocol_field_template_side_effect_audit.py \
  tests/gpt5/phase650_protocol_field_template_side_effect_audit_summary.py
```

冒烟检查：

```bash
python tests/gpt5/phase650_protocol_field_template_side_effect_audit.py \
  qwen3 --smoke --hard-exit-after-model
```

正式测试按顺序执行，均带 `--hard-exit-after-model`：

```bash
python tests/gpt5/phase650_protocol_field_template_side_effect_audit.py \
  qwen3 --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase650_protocol_field_template_side_effect_audit.py \
  glm4 --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase650_protocol_field_template_side_effect_audit.py \
  deepseek7b --confirm --save-rows --hard-exit-after-model
```

生成汇总：

```bash
python tests/gpt5/phase650_protocol_field_template_side_effect_audit_summary.py
```

### 测试原理

Phase 650 使用 Phase 649 的对齐位置规则：

```text
label_colon = token(colon - 1) + token(colon)
label_aligned = label_colon
separator = " ? Label:" 或 " ?\nLabel:"
relation_tail = relation + separator
```

模板扩展为：

```text
Answer:
Response:
Value:
```

任务边界扩展为：

```text
target_failure
original_correct
relation_changed
explanation_needed
non_value
```

每个模型正式测试：

```text
raw_cases = 320
selected_items = 40
每类 split = 8
templates = 3
position_units = label_aligned, label_colon, separator, relation_tail
components = layer_out, attn_out, mlp_out
layers = L17-L20
controls = restore, random, reverse
directions = to_original, remove_from_inline
mode_rows = 8880
```

这个设计的核心判断不是简单看 exact，而是比较同一 split / template 下 patch 相对 baseline 的变化：

```text
to_original:
  与 original baseline 比较

remove_from_inline:
  与 inline baseline 比较
```

其中 `relation_changed`、`explanation_needed`、`non_value` 中的 exact 不代表正向成功，而代表旧 value token 吸附风险。

### 客观结果

三模型均完成正式测试：

```text
qwen3:
  rows = 8880
  time = 17.45 min
  filtered = position_missing 0, position_len_mismatch 0, empty_patch 0

GLM4:
  rows = 8880
  time = 20.07 min
  filtered = position_missing 0, position_len_mismatch 0, empty_patch 0

DS7B:
  rows = 8880
  time = 21.02 min
  filtered = position_missing 0, position_len_mismatch 0, empty_patch 0
```

这说明 Phase 649 的 label alignment 规则在三种 label 模板上都能稳定形成可 patch 的位置，不再出现 Phase 648 的 tokenization alignment failure。

### DS7B 关键结果

DS7B 的 target_failure 中，Answer 和 Response 模板表现出强修复：

```text
target_failure / Answer:
  label_aligned_to_original_L17_20_layer_out_restore:
    base 0/8 -> patch 8/8
    newline delta = -7

  label_colon_to_original_L17_20_layer_out_restore:
    base 0/8 -> patch 8/8
    newline delta = -7

  separator_to_original_L17_20_layer_out_restore:
    base 0/8 -> patch 8/8
    newline delta = -7

target_failure / Response:
  label_aligned_to_original_L17_20_layer_out_restore:
    base 0/8 -> patch 8/8

  label_colon_to_original_L17_20_layer_out_restore:
    base 0/8 -> patch 8/8

  separator_to_original_L17_20_layer_out_restore:
    base 0/8 -> patch 8/8
```

DS7B 中 MLP 也能强修复部分模板：

```text
target_failure / Answer:
  label_aligned_to_original_L17_20_mlp_out_restore:
    base 0/8 -> patch 7/8

  label_colon_to_original_L17_20_mlp_out_restore:
    base 0/8 -> patch 7/8

target_failure / Response:
  label_aligned_to_original_L17_20_mlp_out_restore:
    base 0/8 -> patch 7/8

  separator_to_original_L17_20_mlp_out_restore:
    base 0/8 -> patch 7/8
```

但 Value 模板明显弱于 Answer / Response：

```text
target_failure / Value:
  label_aligned_to_original_L17_20_attn_out_restore:
    patch 3/8

  separator_to_original_L17_20_attn_out_restore:
    patch 3/8

  relation_tail_to_original_L17_20_attn_out_restore:
    patch 0/8
```

因此 DS7B 的协议场不是任意 label 完全等价，而是对 natural answer-like labels 更强，对 Value 这种语义更像字段名的标签较弱。

### qwen3 关键结果

qwen3 也出现模板泛化，但机制极性不同：

```text
target_failure / Response:
  label_aligned_to_original_L17_20_layer_out_restore:
    base 4/8 -> patch 8/8

  label_colon_to_original_L17_20_layer_out_restore:
    base 4/8 -> patch 8/8

target_failure / Answer:
  separator_to_original_L17_20_attn_out_restore:
    base 4/8 -> patch 7/8
```

qwen3 中一个非常重要的现象是，remove_from_inline 方向有时反而把 inline baseline 拉成正确短答：

```text
target_failure / Answer:
  label_aligned_remove_from_inline_L17_20_mlp_out_restore:
    base 0/8 -> patch 8/8

  label_colon_remove_from_inline_L17_20_mlp_out_restore:
    base 0/8 -> patch 8/8

  separator_remove_from_inline_L17_20_mlp_out_restore:
    base 0/8 -> patch 8/8
```

这说明 qwen3 的 original/inline 极性与 DS7B 不完全同向，不能把 DS7B 的协议轨迹方向直接移植到 qwen3。

### GLM4 关键结果

GLM4 的 target_failure 修复较温和，最高主要在 relation_tail / separator 的 layer_out：

```text
target_failure / Answer:
  relation_tail_to_original_L17_20_layer_out_restore:
    base 6/8 -> patch 7/8

  separator_to_original_L17_20_layer_out_restore:
    patch 6/8

target_failure / Response:
  relation_tail_to_original_L17_20_layer_out_restore:
    patch 6/8

target_failure / Value:
  separator_to_original_L17_20_layer_out_restore:
    patch 5/8

  relation_tail_to_original_L17_20_layer_out_restore:
    patch 5/8
```

GLM4 中 label_aligned / label_colon 有效果，但没有 DS7B 那样形成压倒性闭合。

### 副作用边界结果

Phase 650 最重要的新信息不是“修复更强”，而是发现 protocol field 会把一些非目标任务强行拉回 value short answer。

DS7B 中：

```text
non_value / Answer:
  separator_to_original_L17_20_mlp_out_restore:
    base 0/8 -> patch 8/8

  relation_tail_to_original_L17_20_mlp_out_restore:
    base 0/8 -> patch 7/8

explanation_needed / Answer:
  label_aligned_to_original_L17_20_mlp_out_restore:
    base 0/8 -> patch 8/8

  label_colon_to_original_L17_20_mlp_out_restore:
    base 0/8 -> patch 8/8

  separator_to_original_L17_20_mlp_out_restore:
    base 0/8 -> patch 8/8

explanation_needed / Response:
  label_aligned_to_original_L17_20_attn_out_restore:
    base 0/8 -> patch 8/8

  separator_to_original_L17_20_attn_out_restore:
    base 0/8 -> patch 8/8
```

这不是正向能力提升，而是强烈说明：

```text
protocol field 可以驱动“输出短值”的状态，
但它本身不包含 task intent boundary。
```

换句话说，DS7B 的 L17-L20 protocol field 更像短答读出协议执行场，而不是完整语言任务判断器。

qwen3 中也有明显副作用：

```text
explanation_needed / Answer:
  label_aligned_remove_from_inline_L17_20_layer_out_restore:
    base 1/8 -> patch 8/8

  separator_remove_from_inline_L17_20_attn_out_restore:
    base 1/8 -> patch 8/8

non_value / Value:
  relation_tail_to_original_L17_20_attn_out_restore:
    base 0/8 -> patch 4/8
```

GLM4 的副作用较弱，但 relation_changed 和 explanation_needed 仍有 old-value attraction：

```text
relation_changed / Response:
  label_aligned_remove_from_inline_L17_20_layer_out_restore:
    base 2/8 -> patch 4/8

explanation_needed / Value:
  separator_to_original_L17_20_layer_out_restore:
    base 0/8 -> patch 3/8
```

### 对 Phase 649 附件分析的评价

附件对 Phase 649 的总体判断基本正确：

```text
1. answer_label_aligned 被重新找回，正确。
2. answer_colon / separator 是同等级强位置，正确，尤其在 DS7B。
3. relation_tail 中 attention / MLP 有可分离趋势，基本正确，但不能过度解释成 attention=语义、MLP=格式。
4. 下一步应做 template generalization 和 side-effect boundary，正确。
```

Phase 650 对附件做了推进：

```text
1. label alignment 在 Answer / Response / Value 三模板中没有位置失配。
2. DS7B 在 Answer / Response 上形成强闭合，但 Value 较弱。
3. protocol field 的副作用边界非常明显：它能把 explanation / non-value 任务拉回短值输出。
```

### 理论进展

Phase 649 后的节点是：

```text
aligned protocol field with controls
```

Phase 650 后应更新为：

```text
template-conditioned protocol field with side-effect boundary
```

当前更准确的机制表达：

```text
value_short_answer_protocol(m, template, task)
  =
    field_strength(label_aligned, label_colon, separator, relation_tail)
    × template_compatibility(template)
    × task_intent_gate(task)
    × model_polarity(m)
```

其中 Phase 650 已经证明：

```text
field_strength 存在；
template_compatibility 存在；
side_effect_boundary 存在；
task_intent_gate 尚未定位。
```

新的结构图谱节点可以写成：

```text
M_i =
{
  aligned_position_field,
  template_compatibility_profile,
  side_effect_boundary_profile,
  target_repair_profile,
  non_target_absorption_profile,
  component_polarity,
  model_polarity,
  task_intent_gap
}
```

对 DS7B：

```text
DS7B_protocol_field:
  Answer / Response:
    label_aligned + label_colon + separator at L17-L20 layer_out
    strong target repair

  Value:
    weaker compatibility

  explanation_needed / non_value:
    strong old-value absorption risk

  conclusion:
    protocol field is execution-like, not intent-selective.
```

### 问题和硬伤

1. Phase 650 的 target_failure 是按原始 Answer 模板筛选，再迁移到 Response / Value 模板。因此跨模板结果是 template transfer，不是每个模板独立筛出的失败集。

2. `label_aligned = colon-1 + colon` 仍可能包含 tokenizer 的前置空白/换行痕迹，虽然本阶段没有位置失配，但还不能证明它是纯 label token。

3. side-effect 中的 exact 不是统一语义含义：在 original_correct 中 exact 是保持正确，在 explanation_needed / non_value 中 exact 是旧短值吸附风险。因此必须按 split 解读。

4. Phase 650 仍只测试 3 个 label 模板，尚未覆盖自然语言回答、长标签、无冒号格式、中文标签、JSON / list 等格式。

5. 当前 patch 是状态替换，不是自然生成路径中的 writer 因果链完整闭合。

### 当前结论

Phase 650 是关键推进，但不是简单正结果。

可以确认：

```text
1. Phase 649 的 aligned label protocol field 是真实可测结构，不是单模板位置偶然。
2. DS7B 对 Answer / Response 的 L17-L20 protocol field 修复非常强。
3. Value 模板较弱，说明协议字段有模板兼容性。
4. protocol field 会对 explanation_needed / non_value 产生短值吸附副作用。
```

必须收紧：

```text
protocol field 不是完整语言理解机制；
它更像短答值输出协议的执行场；
task intent boundary 仍未破解。
```

### 下一阶段

Phase 651 应执行：

```text
Task Intent Gate and Protocol Field Boundary Audit
```

目标：

```text
1. 把 explanation_needed / non_value / relation_changed 中的 intent signal 拆成可定位位置。
2. 测试 intent instruction、question type、answer format、label field 谁决定是否允许 value_short_answer_protocol 启动。
3. 对 DS7B 优先测试，因为 Phase 650 中 DS7B 副作用最强，最适合定位 task intent gate。
4. 对 qwen3 / GLM4 做同脚本对照，记录模型极性差异。
```

建议测试对象：

```text
positions:
  instruction_span
  answer_format_span
  question_type_span
  label_aligned
  separator
  relation_tail

splits:
  short_value_allowed
  explanation_required
  yes_no_required
  relation_changed
  format_changed

components:
  layer_out
  attn_out
  mlp_out

layers:
  L14-L22 粗扫
  对 DS7B 在 L17-L20 细扫
```

阶段目标：

```text
从 protocol execution field
推进到 intent-conditioned protocol gate。
```

## Phase 651: Task Intent Gate and Protocol Field Boundary Audit [2026-06-26 05:33]

### 任务来源

Phase 650 已经证明：

```text
protocol field 可以驱动短答 value output，
但会对 explanation_needed / non_value 等任务产生短值吸附副作用。
```

因此 Phase 651 不再继续扩大同类 protocol patch，而是转向定位 task intent gate：

```text
哪些内部位置决定“是否允许短答协议启动”？
```

### 生成脚本

新增正式测试脚本：

```bash
tests/gpt5/phase651_task_intent_gate_protocol_boundary_audit.py
```

新增汇总脚本：

```bash
tests/gpt5/phase651_task_intent_gate_protocol_boundary_audit_summary.py
```

结果目录：

```bash
results/glm5_phase651_task_intent_gate_protocol_boundary_audit/
```

跨模型汇总：

```bash
results/glm5_phase651_task_intent_gate_protocol_boundary_audit/phase651_cross_model_summary.md
```

### 执行命令

编译检查：

```bash
python -m py_compile \
  tests/gpt5/phase651_task_intent_gate_protocol_boundary_audit.py \
  tests/gpt5/phase651_task_intent_gate_protocol_boundary_audit_summary.py
```

冒烟检查：

```bash
python tests/gpt5/phase651_task_intent_gate_protocol_boundary_audit.py \
  qwen3 --smoke --hard-exit-after-model
```

正式测试按顺序执行，均带 `--hard-exit-after-model`：

```bash
python tests/gpt5/phase651_task_intent_gate_protocol_boundary_audit.py \
  qwen3 --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase651_task_intent_gate_protocol_boundary_audit.py \
  glm4 --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase651_task_intent_gate_protocol_boundary_audit.py \
  deepseek7b --confirm --save-rows --hard-exit-after-model
```

生成汇总：

```bash
python tests/gpt5/phase651_task_intent_gate_protocol_boundary_audit_summary.py
```

### 测试原理

Phase 651 使用等结构指令构造任务意图：

```text
Instruction: Answer with value.
Instruction: Answer with reason.
Instruction: Answer with yesno.
Instruction: Answer with sentence.
```

任务类型：

```text
short_value_allowed:
  value

explanation_required:
  reason

yes_no_required:
  yesno

full_sentence_required:
  sentence

relation_changed:
  value + changed relation
```

核心测试方向：

```text
value_to_task:
  把 short_value_allowed 的状态写入非短答任务。
  如果 correct value exact 或 rank 大幅提升，说明短值协议对该任务产生吸附。

task_to_value:
  把非短答任务状态写回 short_value_allowed。
  如果 correct value exact 下降或 rank 变差，说明任务意图状态能压制短答协议。
```

测试位置：

```text
intent_word
instruction_span
instruction_prefix
question_span
relation_text
label_aligned
separator
relation_tail
```

测试区间和组件：

```text
L14-L22 layer_out
L17-L20 layer_out
L17-L20 attn_out
L17-L20 mlp_out
```

对照：

```text
restore
random
reverse
```

正式测试规模：

```text
raw_cases = 320
selected_items = 12
pair_tasks = 4
mode_rows = 7872 / model
max_new_tokens = 12
```

### 客观结果

三模型均完成正式测试：

```text
qwen3:
  rows = 7872
  time = 43.65 min
  filtered = position_missing 48, position_len_mismatch 12, empty_patch 0

GLM4:
  rows = 7872
  time = 65.76 min
  filtered = position_missing 48, position_len_mismatch 12, empty_patch 0

DS7B:
  rows = 7872
  time = 54.14 min
  filtered = position_missing 48, position_len_mismatch 12, empty_patch 0
```

过滤主要来自自然指令下部分 position 的 token span 缺失或长度不一致；没有 empty_patch，说明进入统计的 patch 都有效生成。

### qwen3 结果

qwen3 的 value_to_task 吸附很强：

```text
full_sentence_required:
  instruction_span_value_to_task_L14_22_layer_out_restore:
    exact 0 -> 9 / 12
    rank 4.9 -> 1.2
    rank_improve = +3.8

yes_no_required:
  label_aligned_value_to_task_L17_20_attn_out_restore:
    exact 0 -> 7 / 12
    rank 6.3 -> 1.4
    rank_improve = +4.9

full_sentence_required:
  relation_tail_value_to_task_L17_20_attn_out_restore:
    exact 0 -> 6 / 12
    rank 4.9 -> 2.2
```

qwen3 的 task_to_value 在 exact 上没有下降空间，因为 value baseline exact 已经是 0，但 rank 显示明显压制：

```text
yes_no_required:
  label_aligned_task_to_value_L17_20_mlp_out_restore:
    rank 6.0 -> 25.0
    rank_improve = -19.0

yes_no_required:
  separator_task_to_value_L17_20_mlp_out_restore:
    rank 6.0 -> 18.6
    rank_improve = -12.6

relation_changed:
  relation_tail_task_to_value_L17_20_mlp_out_restore:
    rank 6.0 -> 17.6
    rank_improve = -11.6
```

这说明 qwen3 中 task intent 的压制更多体现在 rank / support 层，而不一定体现在 exact generation 层。

### GLM4 结果

GLM4 的结果最清晰，value_to_task 和 task_to_value 都有强效果。

value_to_task 吸附：

```text
explanation_required:
  label_aligned_value_to_task_L17_20_layer_out_restore:
    exact 0 -> 7 / 12
    rank 74.8 -> 2.0
    rank_improve = +72.8

explanation_required:
  label_aligned_value_to_task_L14_22_layer_out_restore:
    exact 0 -> 5 / 12
    rank 74.8 -> 2.0

yes_no_required:
  label_aligned_value_to_task_L14_22_layer_out_restore:
    exact 0 -> 3 / 12
    rank 204.2 -> 3.9
```

task_to_value 压制：

```text
yes_no_required:
  separator_task_to_value_L14_22_layer_out_restore:
    exact 2 -> 0 / 12
    rank 2.1 -> 137.2
    rank_improve = -135.1

yes_no_required:
  relation_tail_task_to_value_L14_22_layer_out_restore:
    exact 2 -> 0 / 12
    rank 2.1 -> 134.7

yes_no_required:
  label_aligned_task_to_value_L14_22_layer_out_restore:
    exact 2 -> 0 / 12
    rank 2.1 -> 93.8

explanation_required:
  relation_tail_task_to_value_L14_22_layer_out_restore:
    exact 2 -> 0 / 12
    rank 2.1 -> 56.6
```

GLM4 说明 task intent state 确实可以在中后层 strongly suppress value-token support。

### DS7B 结果

DS7B 中也出现明确吸附与压制，但 exact 闭合比 GLM4 弱，rank 变化非常强。

value_to_task 吸附：

```text
full_sentence_required:
  instruction_span_value_to_task_L14_22_layer_out_restore:
    exact 5 -> 9 / 12
    rank 7.6 -> 1.6
    rank_improve = +6.0

explanation_required:
  label_aligned_value_to_task_L14_22_layer_out_restore:
    exact 0 -> 3 / 12
    rank 87.3 -> 8.1
    rank_improve = +79.2

explanation_required:
  separator_value_to_task_L14_22_layer_out_restore:
    exact 0 -> 3 / 12
    rank 87.3 -> 8.5

yes_no_required:
  separator_value_to_task_L17_20_layer_out_restore:
    exact 0 -> 1 / 12
    rank 330.9 -> 22.8
    rank_improve = +308.1
```

task_to_value 压制：

```text
yes_no_required:
  separator_task_to_value_L14_22_layer_out_restore:
    exact 0 -> 0 / 12
    rank 8.0 -> 76.2
    rank_improve = -68.2

yes_no_required:
  relation_tail_task_to_value_L14_22_layer_out_restore:
    rank 8.0 -> 75.7

yes_no_required:
  label_aligned_task_to_value_L14_22_layer_out_restore:
    rank 8.0 -> 73.8

explanation_required:
  relation_tail_task_to_value_L14_22_layer_out_restore:
    rank 8.0 -> 65.2
```

DS7B 的关键现象：

```text
task intent patch 不一定直接改变 exact，
但能显著改变 correct value token 的 rank。
```

这说明 DS7B 的 task intent gate 可能首先作用于 value-token support landscape，而不是直接决定最终生成文本。

### 与 Phase 650 的关系

Phase 650 证明：

```text
protocol field 是 execution-like；
它能驱动短值输出；
它不自动携带 task intent boundary。
```

Phase 651 进一步证明：

```text
task intent state 是真实存在的；
它可以打开或压制 value_short_answer_protocol；
但它与 protocol field 高度耦合，不是单独的 instruction token 开关。
```

特别是 GLM4 / DS7B 的 L14-L22 layer_out 结果表明：

```text
task intent gate 更像中后层 residual trajectory 的状态场，
不是单个词或单个注意力头。
```

### 理论更新

Phase 650 的公式：

```text
value_short_answer_protocol(m, template, task)
  =
    field_strength(label_aligned, label_colon, separator, relation_tail)
    × template_compatibility(template)
    × task_intent_gate(task)
    × model_polarity(m)
```

Phase 651 后应改成：

```text
output_value_support(m, x, t)
  =
    protocol_execution_field(m, x, t)
    ⊙ intent_permission_field(m, x, t)
    ⊙ relation_content_field(m, x, t)
    ⊙ readout_competition_field(m, x, t)
```

其中：

```text
protocol_execution_field:
  label_aligned / separator / relation_tail 上的短答执行状态

intent_permission_field:
  instruction_span / intent_word / task-conditioned residual trajectory

relation_content_field:
  relation_text / relation_tail 上的内容选择状态

readout_competition_field:
  final logits / prefix rank / newline and non-value competitors
```

更完整的节点表达：

```text
M_i =
{
  protocol_execution_field,
  intent_permission_field,
  relation_content_field,
  readout_competition_field,
  template_compatibility_profile,
  task_absorption_profile,
  task_suppression_profile,
  rank_support_profile,
  exact_generation_profile,
  model_polarity
}
```

### 当前进展

Phase 651 的核心进展是：

```text
从“协议场会导致副作用”
推进到
“任务意图状态可以调制协议场，但调制首先表现为 rank/support 改变”。
```

这为全局图谱提供了新节点：

```text
intent-conditioned protocol gate
```

它不是独立模块，而是和 label_aligned / separator / relation_tail 共用一部分中后层 residual state。

### 问题和硬伤

1. 自然指令模板会引入生成格式差异，因此 exact 不能单独作为判据，必须同时看 rank delta。

2. 当前 instruction 只用了 `value/reason/yesno/sentence` 四个英文意图词，仍然可能带来模板偏置。

3. position_missing 和 position_len_mismatch 不是 0，说明自然指令比 Phase 650 的标准模板更难做严格位置对齐。

4. 当前还没有把 task intent gate 细分到 attention head / MLP neuron / residual subspace。

5. L14-L22 粗扫耗时很高，qwen3 43.65 分钟，GLM4 65.76 分钟，DS7B 54.14 分钟；下一阶段需要收缩到最有信息量的位置和层。

### 当前结论

可以确认：

```text
1. task intent state 真实存在。
2. task intent state 能改变 correct value token 的 rank/support。
3. value_to_task 能把非短答任务重新吸向短值输出。
4. task_to_value 能在 rank 层压制短值输出。
5. L14-L22 layer_out 是强候选区间，L17-L20 是局部执行区间。
```

必须谨慎：

```text
尚未证明 task intent gate 是单一门控；
尚未证明它能单独决定最终自然生成；
当前更像多位置 residual state field。
```

### 下一阶段

Phase 652 应执行：

```text
Intent Gate Layer Localization and Component Narrowing Audit
```

目标：

```text
1. 放弃全量 L14-L22 粗扫，改为单层/小区间定位。
2. 优先围绕 GLM4 和 DS7B 的强结果：
   - label_aligned
   - separator
   - relation_tail
   - instruction_span
3. 将 layer_out 拆为 attn_out / mlp_out / layer_input。
4. 用 rank delta 作为主要指标，exact 作为辅助指标。
5. 保留 qwen3 / GLM4 / DS7B 三模型，但减少生成长度和模式数量。
```

建议测试范围：

```text
layers:
  L14, L15, L16, L17, L18, L19, L20, L21, L22 单层

positions:
  instruction_span
  label_aligned
  separator
  relation_tail

directions:
  value_to_task
  task_to_value

tasks:
  explanation_required
  yes_no_required

metrics:
  rank_delta
  exact_delta
  top0_category
  generation_text_shortness
```

阶段目标：

```text
从 intent-conditioned protocol gate
推进到 localized intent-gate carrier map。
```

## Phase 652: Intent Gate Layer Localization and Component Narrowing Audit [2026-06-26 06:30]

### 任务来源

用户上传的 Phase 650 / Phase 651 分析总体正确。核心判断是：

```text
Phase 650:
  protocol field 不是单模板偶然结构，
  但它是 execution-like short-value field，
  会在非目标任务中产生副作用。

Phase 651:
  task intent state 真实存在，
  能调制 value_short_answer_protocol，
  但首先表现为 rank/support 变化，
  不一定直接闭合 exact generation。
```

因此继续执行 Phase 652：

```text
Intent Gate Layer Localization and Component Narrowing Audit
```

目标是把 Phase 651 的 L14-L22 粗区间结果压缩到：

```text
单层
单位置
单组件
rank delta 主指标
```

### 生成脚本

新增正式测试脚本：

```bash
tests/gpt5/phase652_intent_gate_layer_localization_audit.py
```

新增汇总脚本：

```bash
tests/gpt5/phase652_intent_gate_layer_localization_audit_summary.py
```

结果目录：

```bash
results/glm5_phase652_intent_gate_layer_localization_audit/
```

跨模型汇总：

```bash
results/glm5_phase652_intent_gate_layer_localization_audit/phase652_cross_model_summary.md
```

### 执行命令

编译检查：

```bash
python -m py_compile \
  tests/gpt5/phase652_intent_gate_layer_localization_audit.py \
  tests/gpt5/phase652_intent_gate_layer_localization_audit_summary.py
```

冒烟检查：

```bash
python tests/gpt5/phase652_intent_gate_layer_localization_audit.py \
  qwen3 --smoke --hard-exit-after-model
```

正式测试按顺序执行，均带 `--hard-exit-after-model`：

```bash
python tests/gpt5/phase652_intent_gate_layer_localization_audit.py \
  qwen3 --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase652_intent_gate_layer_localization_audit.py \
  glm4 --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase652_intent_gate_layer_localization_audit.py \
  deepseek7b --confirm --save-rows --hard-exit-after-model
```

生成汇总：

```bash
python tests/gpt5/phase652_intent_gate_layer_localization_audit_summary.py
```

### 测试原理

Phase 652 没有对每个 patch 都做长生成，而是只测最终 readout 前的 correct value prefix rank：

```text
rank_improvement = baseline_rank - patched_rank
```

其中：

```text
rank_improvement > 0:
  value token 支持增强

rank_improvement < 0:
  value token 支持被压制
```

测试任务：

```text
explanation_required
yes_no_required
```

测试方向：

```text
value_to_task:
  把 short_value_allowed 的状态写入非短答任务，
  测短值吸附增强。

task_to_value:
  把非短答任务状态写回 short_value_allowed，
  测短值支持压制。
```

测试位置：

```text
intent_word
instruction_span
label_aligned
separator
relation_tail
```

测试层：

```text
L14, L15, L16, L17, L18, L19, L20, L21, L22
```

测试组件：

```text
layer_input
attn_out
mlp_out
layer_out
```

正式测试规模：

```text
selected_items = 20
tasks = 2
positions = 5
directions = 2
layers = 9
components = 4
rows = 10160 / model
```

### 客观结果

三模型均完成正式测试：

```text
qwen3:
  rows = 10160
  time = 6.65 min
  filtered = position_missing 40, position_len_mismatch 20, empty_patch 0

GLM4:
  rows = 10160
  time = 9.20 min
  filtered = position_missing 40, position_len_mismatch 20, empty_patch 0

DS7B:
  rows = 10160
  time = 7.86 min
  filtered = position_missing 40, position_len_mismatch 20, empty_patch 0
```

Phase 651 的耗时：

```text
qwen3 43.65 min
GLM4 65.76 min
DS7B 54.14 min
```

Phase 652 的耗时显著下降，同时样本数增加到 20。说明把指标从 generation exact 转为 logits/rank，可以更高效地做图谱定位。

### qwen3 结果

qwen3 的 value_to_task 吸附主要集中在较前层：

```text
yes_no_required:
  label_aligned L16 layer_out:
    rank 12.9 -> 1.6
    rank_improvement = +11.3
    tok0 0 -> 10 / 20

  label_aligned L17 layer_input:
    rank 12.9 -> 1.6
    tok0 0 -> 10 / 20

  relation_tail L15 layer_out:
    rank 12.9 -> 1.9
    rank_improvement = +11.1
    tok0 0 -> 12 / 20

  separator L14 layer_input:
    rank 12.9 -> 1.9
    tok0 0 -> 11 / 20
```

qwen3 的 task_to_value 压制较弱，主要在 L16-L18 的 attention / MLP：

```text
yes_no_required:
  label_aligned L18 mlp_out:
    rank 9.2 -> 16.1
    rank_improvement = -6.9

explanation_required:
  label_aligned L16 attn_out:
    rank 9.2 -> 14.2
    rank_improvement = -5.0

  separator L16 attn_out:
    rank 9.2 -> 14.2
    rank_improvement = -5.0
```

qwen3 的定位特征：

```text
吸附峰更靠前、更分散；
压制峰较弱；
与 DS7B / GLM4 的后层强门控不同。
```

### GLM4 结果

GLM4 的 value_to_task 吸附峰非常集中，主要在 L20-L22 的 layer_out：

```text
yes_no_required:
  separator L22 layer_out:
    rank 188.1 -> 3.4
    rank_improvement = +184.8
    tok0 0 -> 3 / 20

  relation_tail L22 layer_out:
    rank 188.1 -> 3.5
    rank_improvement = +184.7

  label_aligned L22 layer_out:
    rank 188.1 -> 3.9
    rank_improvement = +184.2

  relation_tail L21 layer_out:
    rank 188.1 -> 4.1
    rank_improvement = +184.0
```

GLM4 的 task_to_value 压制也集中在同一后层区域：

```text
yes_no_required:
  relation_tail L21 layer_out:
    rank 2.2 -> 136.0
    rank_improvement = -133.8
    tok0 3 -> 0 / 20

  separator L22 layer_out:
    rank 2.2 -> 135.7
    rank_improvement = -133.5

  relation_tail L22 layer_out:
    rank 2.2 -> 135.2
    rank_improvement = -133.1

  separator L21 layer_out:
    rank 2.2 -> 132.3
    rank_improvement = -130.1
```

GLM4 的定位特征：

```text
后层 L21-L22 layer_out 是强门控区；
separator / relation_tail / label_aligned 三个位置共同承载 yes_no intent 对 value token 的开关。
```

### DS7B 结果

DS7B 的 value_to_task 吸附同样集中在后层 L20-L22：

```text
yes_no_required:
  relation_tail L22 layer_out:
    rank 295.6 -> 13.8
    rank_improvement = +281.7

  separator L22 layer_out:
    rank 295.6 -> 14.1
    rank_improvement = +281.4

  label_aligned L22 layer_out:
    rank 295.6 -> 14.7
    rank_improvement = +280.9

  relation_tail L21 layer_out:
    rank 295.6 -> 19.4
    rank_improvement = +276.2
```

DS7B 的 task_to_value 压制也在后层：

```text
yes_no_required:
  separator L22 layer_out:
    rank 8.0 -> 61.8
    rank_improvement = -53.8

  relation_tail L22 layer_out:
    rank 8.0 -> 61.8
    rank_improvement = -53.8

  label_aligned L22 layer_out:
    rank 8.0 -> 60.4
    rank_improvement = -52.4

  relation_tail L19 layer_out:
    rank 8.0 -> 58.2
    rank_improvement = -50.2
```

DS7B 的定位特征：

```text
rank delta 极大，但 tok0 exact 不一定同步闭合；
这与 Phase 651 一致：DS7B 的 intent gate 首先改变 support landscape。
```

### 对上传分析的评价

上传分析对 Phase 650 / 651 的判断基本正确：

```text
1. Phase 650 确认 protocol field 是 execution-like，而不是完整任务理解器。
2. Phase 651 确认 task intent state 存在，并能调制短答协议。
3. 必须把 exact 与 rank/support 分开看，正确。
4. 下一步应从 L14-L22 粗扫转向层定位，正确。
```

Phase 652 对上传分析做了实证推进：

```text
1. GLM4 / DS7B 的 intent-conditioned protocol gate 主要集中在后层 L20-L22。
2. qwen3 的吸附更靠前，主要在 L14-L17，压制较弱。
3. separator / relation_tail / label_aligned 是比 intent_word 更稳定的门控载体。
4. layer_out 是主载体，layer_input 的强结果多半反映下一层输入等价关系。
```

### 理论进展

Phase 651 的节点：

```text
intent-conditioned protocol gate
```

Phase 652 后应更新为：

```text
localized intent-gate carrier map
```

新的分层结构：

```text
intent_gate_carrier(m)
  =
    {layer, position, component, direction, task}
```

目前三模型可写成：

```text
qwen3:
  absorption:
    L14-L17, label_aligned / separator / relation_tail, layer_input/layer_out
  suppression:
    L16-L18, label_aligned / separator / relation_tail, attn_out/mlp_out

GLM4:
  absorption:
    L20-L22, separator / relation_tail / label_aligned, layer_out
  suppression:
    L21-L22, separator / relation_tail / label_aligned, layer_out

DS7B:
  absorption:
    L20-L22, separator / relation_tail / label_aligned, layer_out
  suppression:
    L19-L22, separator / relation_tail / label_aligned, layer_out
```

### 统一公式更新

Phase 651：

```text
output_value_support(m, x, t)
  =
    protocol_execution_field(m, x, t)
    ⊙ intent_permission_field(m, x, t)
    ⊙ relation_content_field(m, x, t)
    ⊙ readout_competition_field(m, x, t)
```

Phase 652 后应加入局部载体图：

```text
intent_permission_field(m, x, t)
  =
    Σ_{l,p,c}
      G_m(l,p,c,t)
      · h_m(l,p,c,x)
```

其中：

```text
l = layer
p = position
c = component
G_m(l,p,c,t) = 该模型、该任务下的意图门载体强度
h_m(l,p,c,x) = 对应隐藏状态
```

更完整的当前公式：

```text
support_value(m, x, t)
  =
    R_m(
      Σ_{l,p,c}
        [
          P_m(l,p,c,x)
          ⊙ I_m(l,p,c,t)
          ⊙ C_m(l,p,c,x)
        ]
    )
```

其中：

```text
P_m = protocol execution carrier
I_m = intent permission carrier
C_m = relation/content carrier
R_m = readout competition
```

Phase 652 的贡献是开始给 `I_m` 定位：

```text
I_GLM4, I_DS7B:
  high strength around L20-L22 layer_out

I_qwen3:
  more distributed, earlier absorption around L14-L17
```

### 问题和硬伤

1. Phase 652 是 restore-only localization，没有为每个单层位置加入 random / reverse controls。Phase 651 已有区间级 controls，但单层定位还需要后续对强峰做 controls。

2. 当前主要用 rank delta，不做长生成，因此它定位的是 readout support carrier，不等于完整自然生成闭环。

3. position_missing 和 position_len_mismatch 仍存在，主要来自自然 instruction tokenization。虽然 empty_patch = 0，但说明自然指令模板仍需更严格的对齐规则。

4. yes_no_required 的信号最强，explanation_required 相对弱，说明当前任务意图图谱仍偏向“禁止 value 输出”的 yes/no 门，而不是完整解释生成机制。

5. qwen3 与 GLM4 / DS7B 差异明显，不能把一个模型的层位图谱直接套到另一个模型。

### 当前结论

可以确认：

```text
1. Phase 650 / 651 的方向正确。
2. task intent gate 不是抽象假设，已有可定位载体。
3. GLM4 / DS7B 的核心载体集中在 L20-L22 layer_out。
4. separator / relation_tail / label_aligned 是稳定载体，比 intent_word 更强。
5. qwen3 的机制更早、更分散。
```

必须收紧：

```text
这不是完整语言意图机制；
这是 value-token support 层面的 task intent carrier map；
还需要对强峰做 controls 和生成闭环验证。
```

### 下一阶段

Phase 653 应执行：

```text
Localized Intent-Gate Control and Generation Closure Audit
```

目标：

```text
1. 只取 Phase 652 的强峰：
   GLM4:
     L21-L22 separator / relation_tail / label_aligned layer_out
   DS7B:
     L20-L22 separator / relation_tail / label_aligned layer_out
   qwen3:
     L14-L17 separator / relation_tail / label_aligned layer_out/input

2. 对强峰加入 restore / random / reverse controls。

3. 对强峰重新加入短生成闭环，但 max_new_tokens 控制在 4-6。

4. 测试 yes_no_required 与 explanation_required 是否都能复现。

5. 主指标：
   rank_delta
   tok0_delta
   short value generation rate
   yes/no or explanation signal rate
```

阶段目标：

```text
从 localized intent-gate carrier map
推进到 controlled localized intent gate with generation evidence。
```

## Phase 653: Localized Intent-Gate Control and Generation Closure Audit [2026-06-26 06:52]

### 触发问题

用户要求分析 Phase 652 的评估是否正确，并继续完成任务。Phase 652 已经把 task intent gate（任务意图门）从 L14-L22 粗区间收缩到单层、单位置、单组件的 rank-support carrier map（排名支持载体图谱），但它仍有两个硬伤：

```text
1. Phase 652 是 restore-only localization。
2. Phase 652 主要看 rank / tok0，不等于 natural generation closure。
```

因此 Phase 653 属于同一阶段性目标的必要闭环，不是另开新方向：

```text
localized intent-gate carrier map
  ->
controlled localized intent gate with generation evidence
```

### 生成脚本

```text
tests/gpt5/phase653_localized_intent_gate_generation_closure.py
tests/gpt5/phase653_localized_intent_gate_generation_closure_summary.py
```

### 运行命令

静态检查：

```bash
python -m py_compile tests/gpt5/phase653_localized_intent_gate_generation_closure.py tests/gpt5/phase653_localized_intent_gate_generation_closure_summary.py
```

qwen3 smoke：

```bash
python tests/gpt5/phase653_localized_intent_gate_generation_closure.py qwen3 --smoke --hard-exit-after-model
```

正式测试，严格顺序执行：

```bash
python tests/gpt5/phase653_localized_intent_gate_generation_closure.py qwen3 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase653_localized_intent_gate_generation_closure.py glm4 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase653_localized_intent_gate_generation_closure.py deepseek7b --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase653_localized_intent_gate_generation_closure_summary.py
```

### 结果文件

```text
results/glm5_phase653_localized_intent_gate_generation_closure/phase653_qwen3_localized_intent_gate_generation_closure_confirm.json
results/glm5_phase653_localized_intent_gate_generation_closure/phase653_glm4_localized_intent_gate_generation_closure_confirm.json
results/glm5_phase653_localized_intent_gate_generation_closure/phase653_deepseek7b_localized_intent_gate_generation_closure_confirm.json
results/glm5_phase653_localized_intent_gate_generation_closure/phase653_cross_model_summary.md
```

三模型正式测试均为：

```text
selected_items = 12
mode_rows = 480
position_missing = 0
position_len_mismatch = 0
empty_patch = 0
max_new_tokens = 6
```

运行时间：

```text
qwen3: 1.65 min
GLM4: 2.43 min
DS7B: 2.02 min
```

### 测试原理

Phase 653 不再全层粗扫，而是只取 Phase 652 的强峰站点，加入三类对照：

```text
restore
random
reverse
```

每个 patch 同时测：

```text
1. correct value prefix rank
2. tok0 hit
3. short greedy generation exact
4. generation text flags
```

核心判断标准：

```text
如果 restore 明显改变 rank / tok0 / generation，
且 random / reverse 不能产生同方向、同强度、同文本形态的变化，
则该 localized site 更接近受控 task intent gate。
```

测试站点来自 Phase 652：

```text
qwen3:
  early_peak_layer_out:
    L14-L17, label_aligned / separator / relation_tail, layer_out
  separator_input_edge:
    L14, separator, layer_input
  mid_suppression_attn_mlp:
    L16-L18, label_aligned / separator, attn_out / mlp_out

GLM4:
  late_peak_layer_out:
    L21-L22, label_aligned / separator / relation_tail, layer_out
  l22_peak_layer_out:
    L22, label_aligned / separator / relation_tail, layer_out
  relation_separator_l21_l22:
    L21-L22, separator / relation_tail, layer_out

DS7B:
  late_peak_layer_out:
    L20-L22, label_aligned / separator / relation_tail, layer_out
  l22_peak_layer_out:
    L22, label_aligned / separator / relation_tail, layer_out
  relation_separator_l22:
    L22, separator / relation_tail, layer_out
```

### 客观结果

#### qwen3

value_to_task absorption（值任务到其它任务吸附）：

```text
yes_no_required + separator_input_edge:
  rank 6.3 -> 1.2
  rank_improvement = +5.2
  exact 0 -> 10 / 12
  tok0 0 -> 10 / 12

yes_no_required + early_peak_layer_out:
  rank 6.3 -> 2.2
  rank_improvement = +4.2
  exact 0 -> 4 / 12
  tok0 0 -> 4 / 12
```

suppression / disruption（压制 / 破坏）：

```text
mid_suppression_attn_mlp:
  explanation_required:
    rank 1.3 -> 38.8
    exact 7 -> 0 / 12
    tok0 8 -> 0 / 12

  short_value_allowed under task_to_value:
    rank 6.0 -> 58.0 / 54.2
```

qwen3 的关键现象：

```text
1. separator_input_edge 已经能把 yes/no prompt 拉成短值生成。
2. early_peak_layer_out 有生成闭合，但弱于 separator_input_edge。
3. mid_suppression_attn_mlp 更像破坏性 / 抑制性写入区，不是干净的意图门。
4. qwen3 的 random / reverse 对照会产生较多异常文本，说明 qwen3 对早中层扰动非常敏感。
```

#### GLM4

value_to_task absorption：

```text
yes_no_required + late_peak_layer_out:
  rank 204.2 -> 3.8
  rank_improvement = +200.5
  exact 0 -> 2 / 12
  tok0 0 -> 2 / 12

explanation_required + late_peak_layer_out:
  rank 74.8 -> 2.2
  rank_improvement = +72.6
  exact 0 -> 3 / 12
  tok0 0 -> 3 / 12
```

task_to_value suppression：

```text
yes_no_required -> short_value_allowed + late_peak_layer_out:
  rank 2.1 -> 134.7
  rank_improvement = -132.6
  exact 2 -> 0 / 12
  tok0 2 -> 0 / 12

explanation_required -> short_value_allowed + late_peak_layer_out:
  rank 2.1 -> 56.6
  rank_improvement = -54.5
  exact 2 -> 0 / 12
  tok0 2 -> 0 / 12
```

GLM4 的关键现象：

```text
1. L22 layer_out 已经足以复现 L21-L22 的大部分 rank effect。
2. restore 方向与 Phase 652 完全一致。
3. generation exact 有闭合，但比例不高。
4. random / reverse 多数不能产生同向 value_to_task 生成闭合；但 task_to_value reverse 有时会提高 value exact，说明 reverse 不是中性对照，而是会改变输出协议极性。
```

#### DS7B

value_to_task absorption：

```text
yes_no_required + late_peak_layer_out:
  rank 330.9 -> 14.6
  rank_improvement = +316.3
  exact 0 -> 0 / 12
  tok0 0 -> 0 / 12

explanation_required + late_peak_layer_out:
  rank 87.3 -> 8.5
  rank_improvement = +78.8
  exact 0 -> 3 / 12
  tok0 0 -> 2 / 12
```

task_to_value suppression：

```text
yes_no_required -> short_value_allowed:
  rank 8.0 -> 75.7
  rank_improvement = -67.7
  exact 0 -> 0 / 12
  tok0 0 -> 0 / 12

explanation_required -> short_value_allowed:
  rank 8.0 -> 65.2
  rank_improvement = -57.2
  exact 0 -> 0 / 12
  tok0 0 -> 0 / 12
```

DS7B 的关键现象：

```text
1. L20-L22 / L22 layer_out 对 rank 的因果效应非常强。
2. yes_no_required 中 rank 从 330.9 拉到 14.6，但 exact / tok0 仍为 0。
3. explanation_required 中出现 3 / 12 的 exact 闭合。
4. DS7B 仍然存在 final generation gate / decoder preference gap。
5. random 的 L22 yes_no value_to_task 也能产生 rank +106.9，但没有 exact / tok0，说明 rank alone 不能作为 DS7B 生成闭合证据。
```

### 当前判断

Phase 652 的分析基本正确，但 Phase 653 后需要进一步收紧：

```text
Phase 652:
  localized intent-gate carrier map 成立。

Phase 653:
  controlled localized intent gate 部分成立。
```

跨模型结论：

```text
qwen3:
  已出现局部意图门的短生成闭合，尤其是 yes_no_required + separator_input_edge。

GLM4:
  L22 / L21-L22 layer_out 是强 task-intent protocol gate。
  rank、tok0、短生成 exact 同向，但 exact 比例不高。

DS7B:
  rank-level intent gate 非常强。
  generation-level closure 只在 explanation_required 中部分出现；
  yes_no_required 仍未闭合。
```

因此当前不能说三模型都完成了完整 natural generation controller（自然生成控制器）定位。更准确说：

```text
task intent gate 已经从抽象状态推进到可控局部载体；
但 readout / decoder preference / final generation policy 仍然是 DS7B 和部分 GLM4 的瓶颈。
```

### 理论进展

统一公式从 Phase 652 的：

```text
support_value(m, x, t)
  =
    R_m(
      Σ_{l,p,c}
        [
          P_m(l,p,c,x)
          ⊙ I_m(l,p,c,t)
          ⊙ C_m(l,p,c,x)
        ]
    )
```

推进到需要显式区分 support gate 与 generation gate：

```text
support_value(m, x, t)
  =
    R_m(
      Σ_{l,p,c}
        [
          P_m(l,p,c,x)
          ⊙ I_m(l,p,c,t)
          ⊙ C_m(l,p,c,x)
        ]
    )

generate_value(m, x, t)
  =
    D_m(
      support_value(m, x, t),
      F_m(x,t)
    )
```

其中：

```text
P_m = protocol execution carrier
I_m = intent permission carrier
C_m = relation/content carrier
R_m = readout competition
D_m = decoder / final generation policy
F_m = format / answer policy field
```

Phase 653 的核心贡献是证明：

```text
I_m 可以被局部控制；
但 I_m -> generate_value 之间还有 D_m / F_m 的门。
```

这解释了为什么 DS7B 中 rank 改变极强，但 yes/no 自然生成仍不闭合。

### 问题和硬伤

1. confirm 样本为 12 个 selected items，比 Phase 652 的 20 个少。原因是 Phase 653 加入了 generation，计算成本显著上升。当前结果足以判断趋势，但重要结论仍需要更大样本复测。

2. qwen3 的 random / reverse 对早中层扰动很敏感，会产生异常文本。这说明 qwen3 结果里有一部分是“状态破坏”而不是干净门控。

3. GLM4 的 exact generation rate 只有 2-3 / 12，虽然 rank effect 很强，但生成闭合仍不充分。

4. DS7B 的 yes_no_required 完全没有 exact / tok0 闭合，说明 DS7B 的 bottleneck 已经后移到 final generation gate。

5. reverse control 不是纯负对照。它会改变方向极性，有时能增强 value exact，因此后续不能把 reverse 简单解释成“无效对照”。

### 下一阶段

Phase 654 应继续属于当前阶段性目标，因为 Phase 653 已经证明：

```text
localized intent gate -> support_value 成立；
support_value -> generate_value 不稳定。
```

下一步应测试：

```text
Phase 654: Support-to-Generation Bridge and Final Policy Gate Audit
```

任务：

```text
1. 固定 Phase 653 的强峰，只测 restore。
2. 在 patch 后同时记录：
   final_norm input
   final_norm output
   lm_head logits
   top-k competition
   generated first token

3. 对 DS7B yes_no_required 特别检查：
   correct value rank 已经进入 top 15，
   但 top0 为什么仍不是 correct value。

4. 对 GLM4 检查：
   为什么 rank 进 top 3 后只有 2-3 / 12 exact。

5. 对 qwen3 检查：
   separator_input_edge 为什么能形成 10 / 12 exact，
   它是否绕过了后层 generation gate。
```

阶段目标：

```text
把瓶颈从 task intent carrier
推进到 support-to-generation bridge，
明确 D_m / F_m 的实际载体。
```

## Phase 654: Support-to-Generation Bridge and Final Policy Gate Audit [2026-06-26 07:00]

### 触发问题

Phase 653 已经证明 localized intent gate（局部意图门）能够强烈改变 correct value token support（正确值词元支持），并在 qwen3 / GLM4 上产生部分短生成闭合。但它同时暴露了新的瓶颈：

```text
rank 已经进入 top 15，
但 generation 仍不输出正确值。
```

因此 Phase 654 继续同一阶段性目标，直接审计：

```text
support_value -> generate_value
```

也就是 final readout / decoder policy gate（最终读出 / 解码策略门）。

### 生成脚本

```text
tests/gpt5/phase654_support_generation_bridge_policy_gate_audit.py
tests/gpt5/phase654_support_generation_bridge_policy_gate_audit_summary.py
```

### 运行命令

静态检查：

```bash
python -m py_compile tests/gpt5/phase654_support_generation_bridge_policy_gate_audit.py tests/gpt5/phase654_support_generation_bridge_policy_gate_audit_summary.py
```

qwen3 smoke：

```bash
python tests/gpt5/phase654_support_generation_bridge_policy_gate_audit.py qwen3 --smoke --hard-exit-after-model
```

正式测试，严格顺序执行：

```bash
python tests/gpt5/phase654_support_generation_bridge_policy_gate_audit.py qwen3 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase654_support_generation_bridge_policy_gate_audit.py glm4 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase654_support_generation_bridge_policy_gate_audit.py deepseek7b --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase654_support_generation_bridge_policy_gate_audit_summary.py
```

### 结果文件

```text
results/glm5_phase654_support_generation_bridge_policy_gate_audit/phase654_qwen3_support_generation_bridge_policy_gate_audit_confirm.json
results/glm5_phase654_support_generation_bridge_policy_gate_audit/phase654_glm4_support_generation_bridge_policy_gate_audit_confirm.json
results/glm5_phase654_support_generation_bridge_policy_gate_audit/phase654_deepseek7b_support_generation_bridge_policy_gate_audit_confirm.json
results/glm5_phase654_support_generation_bridge_policy_gate_audit/phase654_cross_model_summary.md
```

三模型正式测试均为：

```text
selected_items = 20
mode_rows = 240
position_missing = 0
position_len_mismatch = 0
empty_patch = 0
max_new_tokens = 6
```

运行时间：

```text
qwen3: 0.97 min
GLM4: 1.45 min
DS7B: 1.24 min
```

### 测试原理

Phase 654 只固定 Phase 653 的强峰 restore patch，不再测试 random / reverse：

```text
qwen3:
  separator_input_edge
  early_peak_layer_out

GLM4:
  l22_peak_layer_out
  late_peak_layer_out

DS7B:
  l22_peak_layer_out
  late_peak_layer_out
```

每个样本同时记录：

```text
1. correct value prefix rank
2. prefix margin vs top
3. top0 category
4. generated first token
5. exact generation
6. tok0 hit
7. final_norm output movement
8. support_without_generation
```

其中：

```text
support_without_generation = prefix_rank <= 15 and exact_correct = false
```

这个指标直接测量：

```text
支持已经进入高排名，
但自然生成仍然不闭合。
```

### 客观结果

#### qwen3

```text
yes_no_required + separator_input_edge:
  mean_rank = 1.9
  exact = 11 / 20
  tok0 = 11 / 20
  support_without_generation = 9 / 20
  top0 = correct_prefix:11, space:8, explanation:1

yes_no_required + early_peak_layer_out:
  mean_rank = 4.2
  exact = 5 / 20
  tok0 = 5 / 20
  support_without_generation = 15 / 20
  top0 = space:15, correct_prefix:5

explanation_required + early_peak_layer_out:
  mean_rank = 4.3
  exact = 7 / 20
  tok0 = 8 / 20
  support_without_generation = 12 / 20
  top0 = space:11, correct_prefix:8, word:1

explanation_required + separator_input_edge:
  mean_rank = 3.1
  exact = 4 / 20
  tok0 = 4 / 20
  support_without_generation = 16 / 20
  top0 = explanation:12, space:4, correct_prefix:4
```

qwen3 结论：

```text
1. separator_input_edge 对 yes_no_required 的 generation closure 最强。
2. 但即使 mean_rank = 1.9，仍有 9 / 20 support_without_generation。
3. qwen3 的瓶颈主要是 correct_prefix 与 space / explanation 的最终竞争。
```

#### GLM4

```text
explanation_required + l22_peak_layer_out:
  mean_rank = 2.1
  exact = 5 / 20
  tok0 = 5 / 20
  support_without_generation = 15 / 20
  top0 = space:15, correct_prefix:5

yes_no_required + l22_peak_layer_out:
  mean_rank = 3.5
  exact = 3 / 20
  tok0 = 3 / 20
  support_without_generation = 17 / 20
  top0 = space:15, correct_prefix:3, explanation:2

late_peak_layer_out 与 l22_peak_layer_out 结果相同。
```

GLM4 结论：

```text
1. L22 layer_out 已经足以代表 L21-L22 late_peak。
2. rank 已经非常高，但大多数样本仍被 space 抢走。
3. GLM4 的主要瓶颈不是 task intent carrier，而是 final token policy 中的 space/readout preference。
```

#### DS7B

```text
explanation_required + l22_peak_layer_out:
  mean_rank = 8.3
  exact = 3 / 20
  tok0 = 2 / 20
  support_without_generation = 14 / 20
  top0 = space:10, newline:8, correct_prefix:2

yes_no_required + l22_peak_layer_out:
  mean_rank = 13.8
  exact = 0 / 20
  tok0 = 0 / 20
  support_without_generation = 15 / 20
  top0 = newline:11, space:9

late_peak_layer_out 与 l22_peak_layer_out 结果相同。
```

DS7B 结论：

```text
1. DS7B 的 correct value support 已进入 top 15。
2. 但 yes_no_required 完全没有 exact / tok0 闭合。
3. top0 被 newline / space 占据。
4. DS7B 的核心瓶颈已经明确后移到 final policy gate。
```

### 关键进展

Phase 654 明确把问题从：

```text
task intent carrier 在哪里？
```

推进到：

```text
为什么 support 已经足够强，但 final generation 仍不选择它？
```

这说明当前图谱必须分成两层：

```text
1. support formation layer
   负责把 correct value token 拉进可竞争集合。

2. final policy layer
   负责决定第一个生成词元到底是 value、space、newline、yes/no、explanation，还是其它格式词。
```

Phase 654 证明：

```text
intent gate 不是最终生成门；
intent gate 只是把 value token 带入竞争场。
```

### 理论公式更新

Phase 653 的公式：

```text
support_value(m, x, t)
  =
    R_m(
      Σ_{l,p,c}
        [
          P_m(l,p,c,x)
          ⊙ I_m(l,p,c,t)
          ⊙ C_m(l,p,c,x)
        ]
    )

generate_value(m, x, t)
  =
    D_m(
      support_value(m, x, t),
      F_m(x,t)
    )
```

Phase 654 后要进一步拆开：

```text
candidate_set(m,x,t)
  =
    TopK(
      support_value(m,x,t)
    )

first_token(m,x,t)
  =
    argmax_z
      [
        support_z(m,x,t)
        +
        policy_bias_z(m,x,t)
        +
        format_prior_z(m,x,t)
      ]
```

其中：

```text
support_z = 语义 / 值词元支持
policy_bias_z = 任务输出策略偏置
format_prior_z = space / newline / explanation / yes-no 等格式先验
```

所以当前完整图景更接近：

```text
relation/content state
  ->
protocol / intent carrier
  ->
value support rank
  ->
format-policy competition
  ->
first generated token
```

### 问题和硬伤

1. Phase 654 没有新增 random / reverse controls，因为 Phase 653 已经做了。本阶段只审计 restore 后的 bridge failure。

2. final_norm movement 已记录，但当前总结还没有深入分析 final_norm 向量与 lm_head 各竞争 token 的投影方向，这需要下一阶段专门做。

3. support_without_generation 阈值使用 prefix_rank <= 15，是实用阈值，不是理论阈值。

4. 当前生成只看 max_new_tokens = 6，足以判断 first token bridge，但不足以判断长文本解释质量。

5. DS7B 的 newline / space 偏置仍未被直接定位到具体层或组件。

### 下一阶段

Phase 655 仍属于当前阶段性目标，因为 Phase 654 已经把瓶颈明确到：

```text
format-policy competition at final token
```

下一步应执行：

```text
Phase 655: Final Token Policy Decomposition and Format Prior Gate Audit
```

任务：

```text
1. 固定 Phase 654 的 bridge failure 样本。
2. 对 correct_prefix、space、newline、yes/no、explanation token 建立 top competitor set。
3. 在 final_norm output 上计算各 token unembedding projection。
4. 分解：
   support_value
   space_prior
   newline_prior
   explanation_prior
   yes_no_prior

5. 对 DS7B 重点定位 newline / space 为什么压过 correct_prefix。
6. 对 GLM4 重点定位 space 为什么在 rank 已进 top 3 时仍占 top0。
7. 对 qwen3 重点定位 separator_input_edge 为什么能突破 format prior。
```

阶段目标：

```text
从 support-to-generation bridge failure
推进到 final token policy decomposition。
```

## Phase 655: Final Token Policy Decomposition and Format Prior Gate Audit [2026-06-26 07:01]

### 触发问题

Phase 654 已经证明：

```text
correct value support 已经进入 top 15，
但 first generated token 仍然经常不是 correct_prefix。
```

因此 Phase 655 不再重新加载模型，而是离线分解 Phase 654 保存的 ladder / groups，分析：

```text
correct_prefix 被哪些 final-token policy groups 压过？
```

### 生成脚本

```text
tests/gpt5/phase655_final_token_policy_decomposition.py
```

### 运行命令

```bash
python -m py_compile tests/gpt5/phase655_final_token_policy_decomposition.py
python tests/gpt5/phase655_final_token_policy_decomposition.py
```

### 结果文件

```text
results/glm5_phase655_final_token_policy_decomposition/phase655_final_token_policy_decomposition.json
results/glm5_phase655_final_token_policy_decomposition/phase655_final_token_policy_decomposition_summary.md
```

### 测试原理

Phase 654 的每行结果已经保存：

```text
top-k tokens
top0_category
groups
prefix_minus_group_max
prefix_rank
exact_correct
generation_text
```

Phase 655 定义：

```text
bridge_failure = prefix_rank <= 15 and exact_correct = false
```

然后按 top0_category（最终胜出类别）统计：

```text
space
newline
explanation
word
punctuation
symbol
correct_prefix
```

核心指标：

```text
prefix_minus_group_max < 0
```

表示该类别的最强 token logit 高于 correct_prefix。

### 客观结果

bridge failures 数量：

```text
qwen3: 108
GLM4: 72
DS7B: 68
```

#### qwen3

失败类别：

```text
space:
  n = 92
  mean_rank = 6.74
  mean_margin_vs_top = -3.645

explanation:
  n = 13
  mean_rank = 3.62
  mean_margin_vs_top = -1.135

word:
  n = 1
```

关键模式：

```text
yes_no_required + separator_input_edge:
  mean_rank = 1.85
  exact = 11 / 20
  support_no_gen = 9 / 20
  winner_vs_prefix = space:8, explanation:1
```

qwen3 结论：

```text
qwen3 的生成失败主要不是 newline，
而是 space prior 和少量 explanation prior 压过 correct_prefix。
separator_input_edge 能突破一半以上样本，但没有完全消除 space prior。
```

#### GLM4

失败类别：

```text
space:
  n = 60
  mean_rank = 3.20
  mean_margin_vs_top = -1.217

explanation:
  n = 12
  mean_rank = 6.50
  mean_margin_vs_top = -2.396
```

关键模式：

```text
explanation_required + l22_peak_layer_out:
  mean_rank = 2.15
  exact = 5 / 20
  support_no_gen = 15 / 20
  winner_vs_prefix = space:15

yes_no_required + l22_peak_layer_out:
  mean_rank = 3.45
  exact = 3 / 20
  support_no_gen = 17 / 20
  winner_vs_prefix = space:15, explanation:2
```

GLM4 结论：

```text
GLM4 的 task intent support 已经很强，
但 final token policy 几乎固定偏向 leading space。

L22 layer_out 不是问题终点；
它把 correct_prefix 拉入竞争集合，
但没有压倒 space policy。
```

#### DS7B

失败类别：

```text
space:
  n = 38
  mean_rank = 4.37
  mean_margin_vs_top = -1.566

newline:
  n = 26
  mean_rank = 8.54
  mean_margin_vs_top = -2.625

explanation:
  n = 2

word:
  n = 2
```

关键模式：

```text
yes_no_required + l22_peak_layer_out:
  mean_rank = 13.85
  exact = 0 / 20
  support_no_gen = 15 / 20
  winner_vs_prefix = space:9, newline:6

explanation_required + l22_peak_layer_out:
  mean_rank = 8.30
  exact = 3 / 20
  support_no_gen = 14 / 20
  winner_vs_prefix = space:10, newline:4
```

DS7B 结论：

```text
DS7B 的 final policy gate 是双重门：
space prior + newline prior。

这解释了为什么 DS7B 的 rank 被大幅拉升后仍不生成 value：
correct_prefix 进入候选集合，但还没有压倒格式先验。
```

### 关键进展

Phase 655 把 Phase 654 的 bridge failure 从现象变成了可分解结构：

```text
support_without_generation
  =
    correct_prefix support exists
    but format_prior wins
```

当前三个模型的 final token policy map：

```text
qwen3:
  main blocker = space
  secondary blocker = explanation
  newline 不主导

GLM4:
  main blocker = space
  secondary blocker = explanation
  L22 已经足够形成 support，但不能解除 space prior

DS7B:
  main blocker = space + newline
  secondary blocker = explanation / word
  yes_no_required 中 newline 更突出
```

### 理论公式更新

Phase 654 的公式：

```text
first_token(m,x,t)
  =
    argmax_z
      [
        support_z(m,x,t)
        +
        policy_bias_z(m,x,t)
        +
        format_prior_z(m,x,t)
      ]
```

Phase 655 后可拆成：

```text
first_token(m,x,t)
  =
    argmax_z
      [
        semantic_support_z
        +
        value_protocol_support_z
        +
        task_intent_permission_z
        +
        format_prior_z
        +
        decoder_policy_bias_z
      ]
```

其中当前已观测到：

```text
format_prior_z =
  space_prior_z
  + newline_prior_z
  + explanation_prior_z
  + yesno_prior_z
  + word_prior_z
```

并且：

```text
generate_value 成功
  iff
    correct_prefix_logit
      >
    max(space, newline, explanation, word, punctuation, symbol ...)
```

更直观地写：

```text
value_generation_margin
  =
    logit(correct_prefix)
    -
    max_logit(format_policy_competitors)
```

当：

```text
value_generation_margin > 0
```

才会真正生成正确值。

### 对当前阶段的判断

Phase 652 到 Phase 655 形成了一个闭环：

```text
Phase 652:
  找到 intent-gate carrier map。

Phase 653:
  加入 controls 和短生成，证明局部 intent gate 可以改变生成，但不完全闭合。

Phase 654:
  定位 support-to-generation bridge failure。

Phase 655:
  分解 bridge failure 的最终竞争类别。
```

因此当前阶段性目标已经完成：

```text
task intent gate 的局部载体、生成桥接失败、最终格式竞争项已经被连成一张小图谱。
```

### 问题和硬伤

1. Phase 655 是离线分解，不是新的因果 patch 实验。

2. 当前只能证明 space / newline / explanation 是最终竞争胜出类别，还没有定位这些 priors 的写入层和写入组件。

3. prefix_rank <= 15 是经验阈值，后续可以改成 top5 / top10 / top20 多阈值审计。

4. 目前只分析 first token，不能代表完整回答文本质量。

5. qwen3 / GLM4 / DS7B 的 final policy map 差异很大，不能直接合并成单一模型结构。

### 下一阶段

下一阶段已经不是 Phase 652-655 的同一小阶段，而是新的阶段性大任务：

```text
Format Prior Writer Localization
```

建议 Phase 656：

```text
Phase 656: Space-Newline-Explanation Prior Writer Localization Audit
```

目标：

```text
1. 以 Phase 655 的失败样本为输入。
2. 分别定位 space_prior、newline_prior、explanation_prior 的写入层和组件。
3. 对 qwen3 重点查 space / explanation。
4. 对 GLM4 重点查 space。
5. 对 DS7B 重点查 space + newline。
6. 不再以 correct value rank 为唯一目标，而是直接测：
   logit(correct_prefix) - logit(space/newline/explanation)
```

这将把图谱从：

```text
value support graph
```

推进到：

```text
value support + format policy competition graph
```

## Phase 656: Space-Newline-Explanation Prior Writer Localization Audit [2026-06-26 07:23]

### 触发问题

Phase 655 证明 final token policy failure（最终词元策略失败）主要来自：

```text
qwen3: space / explanation
GLM4: space / explanation
DS7B: space / newline
```

但 Phase 655 只是离线分解 top0 category，并没有定位这些 format priors（格式先验）由哪些层和组件写入。因此 Phase 656 进入新的阶段性大任务：

```text
Format Prior Writer Localization
```

### 生成脚本

```text
tests/gpt5/phase656_format_prior_writer_localization_audit.py
tests/gpt5/phase656_format_prior_writer_localization_audit_summary.py
```

### 运行命令

静态检查：

```bash
python -m py_compile tests/gpt5/phase656_format_prior_writer_localization_audit.py tests/gpt5/phase656_format_prior_writer_localization_audit_summary.py
```

qwen3 smoke：

```bash
python tests/gpt5/phase656_format_prior_writer_localization_audit.py qwen3 --smoke --hard-exit-after-model
```

正式测试，严格顺序执行：

```bash
python tests/gpt5/phase656_format_prior_writer_localization_audit.py qwen3 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase656_format_prior_writer_localization_audit.py glm4 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase656_format_prior_writer_localization_audit.py deepseek7b --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase656_format_prior_writer_localization_audit_summary.py
```

### 结果文件

```text
results/glm5_phase656_format_prior_writer_localization_audit/phase656_qwen3_format_prior_writer_localization_audit_confirm.json
results/glm5_phase656_format_prior_writer_localization_audit/phase656_glm4_format_prior_writer_localization_audit_confirm.json
results/glm5_phase656_format_prior_writer_localization_audit/phase656_deepseek7b_format_prior_writer_localization_audit_confirm.json
results/glm5_phase656_format_prior_writer_localization_audit/phase656_cross_model_summary.md
```

正式测试规模：

```text
qwen3:
  selected_items = 20
  mode_rows = 2160
  scan_layers = L12-L24
  time = 1.61 min

GLM4:
  selected_items = 20
  mode_rows = 1680
  scan_layers = L18-L27
  time = 1.81 min

DS7B:
  selected_items = 20
  mode_rows = 1680
  scan_layers = L18-L27
  time = 1.56 min
```

三模型均为：

```text
position_missing = 0
position_len_mismatch = 0
empty_patch = 0
```

### 测试原理

Phase 656 固定 Phase 653 / 654 的 intent-gate restore patch（意图门恢复修补），然后在 final readout position（最终读出位置）逐层消融：

```text
attn_out
mlp_out
```

如果消融某个组件后：

```text
correct_prefix - previous_top_format_token
```

的 margin（差距）上升，说明该组件原本在支持 format prior（格式先验）或压制 correct prefix（正确前缀）。

核心指标：

```text
mean_top_margin_delta
  =
    margin_after_ablation
    -
    margin_before_ablation
```

若：

```text
mean_top_margin_delta > 0
```

则该组件是 format-prior writer candidate（格式先验写入候选）。

同时记录：

```text
mean_rank_improvement
flipped_to_correct
space / newline / explanation margin delta
```

### 客观结果

#### qwen3

最强候选：

```text
yes_no_required + early_peak_layer_out:
  L21 mlp_out
  baseline_top0 = space
  n = 15
  mean_top_margin_delta = +1.892
  mean_rank_improvement = +2.47
  flipped_to_correct = 5
  space_delta = +2.12
  explanation_delta = +1.40

explanation_required + early_peak_layer_out:
  L21 mlp_out
  baseline_top0 = space
  n = 11
  mean_top_margin_delta = +1.750
  mean_rank_improvement = +1.45
  flipped_to_correct = 2
  space_delta = +2.08

yes_no_required + early_peak_layer_out:
  L18 attn_out
  baseline_top0 = space
  n = 15
  mean_top_margin_delta = +1.500
  mean_rank_improvement = +2.67
  flipped_to_correct = 5
  space_delta = +1.63
```

qwen3 初步结论：

```text
qwen3 的 space prior writer 候选集中在 L18 attn_out 与 L21 mlp_out。
L21 mlp_out 是最稳定的 space/explanation format-prior writer candidate。
```

#### GLM4

最强候选：

```text
yes_no_required + l22_peak_layer_out:
  L27 attn_out
  baseline_top0 = space
  n = 14
  mean_top_margin_delta = +0.634
  mean_rank_improvement = +1.00
  flipped_to_correct = 1
  space_delta = +0.66

yes_no_required + l22_peak_layer_out:
  L23 attn_out
  baseline_top0 = space
  n = 14
  mean_top_margin_delta = +0.504
  mean_rank_improvement = +0.86
  flipped_to_correct = 1
  space_delta = +0.51

explanation_required + l22_peak_layer_out:
  L23 attn_out
  baseline_top0 = space
  n = 15
  mean_top_margin_delta = +0.487
  mean_rank_improvement = +0.40
  flipped_to_correct = 3
  space_delta = +0.50
```

GLM4 初步结论：

```text
GLM4 的 space prior writer 候选主要是 late attention outputs：
L23 attn_out 与 L27 attn_out。

效应小于 qwen3，但更集中。
```

#### DS7B

最强候选：

```text
explanation_required + l22_peak_layer_out:
  L24 mlp_out
  baseline_top0 = newline
  n = 7
  mean_top_margin_delta = +0.804
  mean_rank_improvement = +7.00
  flipped_to_correct = 0
  newline_delta = +0.80
  space_delta = +1.11

yes_no_required + l22_peak_layer_out:
  L24 mlp_out
  baseline_top0 = newline
  n = 11
  mean_top_margin_delta = +0.551
  mean_rank_improvement = +7.27
  flipped_to_correct = 0
  newline_delta = +0.55
  space_delta = +1.01

explanation_required + l22_peak_layer_out:
  L23 mlp_out
  baseline_top0 = newline
  n = 7
  mean_top_margin_delta = +0.661
  mean_rank_improvement = +1.14
  flipped_to_correct = 1
  newline_delta = +1.46
```

DS7B 初步结论：

```text
DS7B 的 newline / space prior writer 候选主要集中在 L23-L24 mlp_out，
其中 L24 mlp_out 对 newline/space 双重格式先验最明显。
```

### 关键进展

Phase 655 只知道：

```text
correct_prefix 输给 space/newline/explanation。
```

Phase 656 开始定位：

```text
哪些层和组件在写入这些 format priors。
```

当前模型差异：

```text
qwen3:
  L18 attn_out + L21 mlp_out
  space/explanation prior

GLM4:
  L23/L27 attn_out
  space prior

DS7B:
  L23/L24 mlp_out
  newline + space prior
```

这说明 final policy gate（最终策略门）不是单一结构，而是模型相关的格式先验写入图谱。

### 问题和硬伤

1. Phase 656 是 final-position component ablation（最终位置组件消融），不是 restore-style positive construction（恢复式正向构造）。

2. 消融某个组件后 margin 变好，只能说明该组件参与了格式先验或压制 correct_prefix，不能直接证明它是唯一写入者。

3. GLM4 的 effect size 较小，虽然集中在 attention，但需要生成级确认。

4. DS7B 的 L24 mlp_out 大幅改善 rank，但 flipped_to_correct 仍为 0，说明它降低了 newline/space prior，但还不足以让 correct_prefix 胜出。

5. qwen3 的 L21 mlp_out 与 L18 attn_out 都有效，说明可能存在多组件格式先验链，而不是单点写入。

### 下一阶段

Phase 657 仍属于同一个 Format Prior Writer Localization 阶段，因为 Phase 656 只完成 margin-level writer localization，还没有验证这些候选是否能改变短生成。

下一步：

```text
Phase 657: Format-Prior Writer Candidate Generation Confirmation
```

目标：

```text
1. 只取 Phase 656 的 top writer candidates。
2. 固定 intent-gate restore patch。
3. 对候选组件做 final-position ablation。
4. 加入短生成，测试 exact / tok0 是否提升。
5. 判断：
   margin-level writer candidate
   是否真的能变成 generation-level writer candidate。
```

## Phase 657: Format-Prior Writer Candidate Generation Confirmation [2026-06-26 07:38]

### 任务来源

用户要求分析 Phase 653-655 的判断是否正确，并在同一阶段内继续推进。附件判断基本正确：Phase 653-655 已经把问题从“intent-gate restore 能不能把 correct value token 拉进候选集”推进到“为什么最终生成仍输给 space/newline/explanation 等格式先验”。因此 Phase 657 接续 Phase 656，验证 Phase 656 的 format-prior writer candidates 是否真的能改变短生成，而不是只改变 final-position logit margin。

### 生成脚本

```text
tests/gpt5/phase657_format_prior_writer_generation_confirmation.py
tests/gpt5/phase657_format_prior_writer_generation_confirmation_summary.py
```

### 运行命令

```bash
python -m py_compile tests/gpt5/phase657_format_prior_writer_generation_confirmation.py tests/gpt5/phase657_format_prior_writer_generation_confirmation_summary.py
python tests/gpt5/phase657_format_prior_writer_generation_confirmation.py qwen3 --smoke --hard-exit-after-model
python tests/gpt5/phase657_format_prior_writer_generation_confirmation.py qwen3 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase657_format_prior_writer_generation_confirmation.py glm4 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase657_format_prior_writer_generation_confirmation.py deepseek7b --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase657_format_prior_writer_generation_confirmation_summary.py
```

### 测试原理

Phase 656 的测试只说明：

```text
在 fixed intent-gate restore patch 下，
消融某个 final-position component 后，
correct_prefix 相对当前 top format token 的 margin 变好。
```

Phase 657 进一步加入短生成：

```text
site_restore:
  只使用 intent-gate restore patch。

candidate_ablation:
  使用相同 intent-gate restore patch，
  并在第一步生成时消融 Phase 656 找到的候选 format-prior writer。
```

观察指标：

```text
exact_correct 是否增加
tok0 correct_prefix 是否增加
correct_prefix rank 是否改善
top0_category 是否从 space/newline/explanation 转向 correct_prefix
```

### 客观结果

输出目录：

```text
results/glm5_phase657_format_prior_writer_generation_confirmation/
results/glm5_phase657_format_prior_writer_generation_confirmation/phase657_cross_model_summary.md
```

三模型数据规模：

```text
qwen3:
  selected = 20
  rows = 240
  time = 0.97 min
  filtered = 0

GLM4:
  selected = 20
  rows = 240
  time = 1.42 min
  filtered = 0

DS7B:
  selected = 20
  rows = 240
  time = 1.20 min
  filtered = 0
```

qwen3 强正结果：

```text
explanation_required + separator_input_edge:
  L16 attn_out
  exact: 4 -> 12
  tok0: 5 -> 14
  rank: 3.1 -> 1.4
  top0: explanation/correct_prefix/space -> correct_prefix/space

yes_no_required + early_peak_layer_out:
  L18 attn_out
  exact: 5 -> 10
  tok0: 5 -> 10
  rank: 4.2 -> 2.1

yes_no_required + early_peak_layer_out:
  L21 mlp_out
  exact: 5 -> 9
  tok0: 5 -> 10
  rank: 4.2 -> 2.3
```

GLM4 弱正结果：

```text
explanation_required:
  L23 attn_out
  exact: 5 -> 6
  tok0: 5 -> 7
  rank: 2.1 -> 1.9

yes_no_required:
  L27 attn_out
  exact: 3 -> 4
  tok0: 3 -> 4
  rank: 3.5 -> 2.6

yes_no_required:
  L23 attn_out
  exact: 3 -> 4
  tok0: 3 -> 4
  rank: 3.5 -> 2.6
```

DS7B 仍未生成闭合：

```text
explanation_required:
  L23 mlp_out
  exact: 3 -> 3
  tok0: 2 -> 3
  rank: 8.3 -> 7.4

explanation_required:
  L24 mlp_out
  exact: 3 -> 3
  tok0: 2 -> 2
  rank: 8.3 -> 5.3

yes_no_required:
  L24 mlp_out
  exact: 0 -> 0
  tok0: 0 -> 0
  rank: 13.8 -> 9.6
```

### 阶段判断

Phase 657 证明：

```text
Phase 656 的 format-prior writer candidate
不是纯 logit artifact，
在 qwen3 和 GLM4 上能够传导到短生成。
```

但 DS7B 说明：

```text
降低 newline/space prior 或改善 rank
不等价于完成 generation closure。
```

### 问题和硬伤

1. Phase 657 使用的是 final-position ablation，不是正向构造 restore，因此只能证明“去掉该 component 会削弱格式先验”，不能证明自然机制就是靠单独该 component 写入。

2. qwen3 和 GLM4 支持 generation-level writer candidate，但 DS7B 不支持单点闭合。

3. DS7B 的 rank 从 13.8 到 9.6 是真实改善，但离 top1 仍很远，说明还有更强的首词元格式门或协议短路。

4. GLM4 效应稳定但小，需要组合测试判断是否是多 writer 累积。

### 下一阶段

Phase 658 与当前任务仍处于同一阶段，因为 Phase 657 只验证了单点候选，尚未验证最终格式先验是否由多个 writer 叠加形成。

下一步：

```text
Phase 658: Combined Format-Prior Suppression Generation Audit
```

目标：

```text
1. 读取 Phase 657 的 generation-level candidate。
2. 在同一个 restore site 内组合多个候选 writer。
3. 测试组合消融是否强于单点消融。
4. 判断 final format prior 是否是 multi-writer summed pressure。
5. 特别检查 DS7B 的 rank-only 改善能否被组合推进到 exact/tok0。
```

## Phase 658: Combined Format-Prior Suppression Generation Audit [2026-06-26 07:38]

### 生成脚本

```text
tests/gpt5/phase658_combined_format_prior_suppression_generation_audit.py
tests/gpt5/phase658_combined_format_prior_suppression_generation_audit_summary.py
```

### 运行命令

```bash
python -m py_compile tests/gpt5/phase658_combined_format_prior_suppression_generation_audit.py tests/gpt5/phase658_combined_format_prior_suppression_generation_audit_summary.py
python tests/gpt5/phase658_combined_format_prior_suppression_generation_audit.py qwen3 --smoke --hard-exit-after-model
python tests/gpt5/phase658_combined_format_prior_suppression_generation_audit.py qwen3 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase658_combined_format_prior_suppression_generation_audit.py glm4 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase658_combined_format_prior_suppression_generation_audit.py deepseek7b --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase658_combined_format_prior_suppression_generation_audit_summary.py
```

### 测试原理

Phase 657 发现单点 writer 有效，但最终格式先验可能不是单点机制，而是多个 component 共同写入。

Phase 658 的核心对照：

```text
site_restore:
  只使用同一个 intent-gate restore patch。

combo_ablation:
  使用同一个 intent-gate restore patch，
  并在第一步生成位置同时消融同 site 的 top1/top2/top3 format-prior writer candidates。
```

这能测试：

```text
single writer effect 是否可以叠加；
组合后是否把 correct_prefix 推到 top1；
DS7B 的 rank-only 改善是否能够被组合推进到 generation closure。
```

### 客观结果

输出目录：

```text
results/glm5_phase658_combined_format_prior_suppression_generation_audit/
results/glm5_phase658_combined_format_prior_suppression_generation_audit/phase658_cross_model_summary.md
```

三模型数据规模：

```text
qwen3:
  selected = 20
  rows = 240
  time = 0.97 min
  filtered = 0

GLM4:
  selected = 20
  rows = 240
  time = 1.42 min
  filtered = 0

DS7B:
  selected = 20
  rows = 240
  time = 1.21 min
  filtered = 0
```

qwen3 组合强正结果：

```text
explanation_required + separator_input_edge:
  L16 attn_out + L20 attn_out
  exact: 4 -> 18
  tok0: 5 -> 20
  rank: 3.1 -> 1.0
  top0: explanation/correct_prefix/space -> correct_prefix 20/20

yes_no_required + early_peak_layer_out:
  L18 attn_out + L21 mlp_out
  exact: 5 -> 16
  tok0: 5 -> 16
  rank: 4.2 -> 1.1

yes_no_required + early_peak_layer_out:
  L18 attn_out + L21 mlp_out + L23 mlp_out
  exact: 5 -> 16
  tok0: 5 -> 14
  rank: 4.2 -> 1.6
```

qwen3 关键事实：

```text
top2 > top1
top3 不一定优于 top2
说明 format prior writer 是可叠加的，但不是简单越多越好。
```

GLM4 组合正结果：

```text
yes_no_required + l22_peak_layer_out:
  L27 attn_out + L23 attn_out
  exact: 3 -> 8
  tok0: 3 -> 8
  rank: 3.5 -> 2.0

yes_no_required + late_peak_layer_out:
  L27 attn_out + L23 attn_out
  exact: 3 -> 8
  tok0: 3 -> 8
  rank: 3.5 -> 2.0
```

GLM4 关键事实：

```text
单点 attention writer 是弱正结果；
L27 + L23 attention 组合后明显增强。
```

DS7B 组合仍未 exact 闭合：

```text
explanation_required:
  L23 mlp_out + L24 mlp_out
  exact: 3 -> 3
  tok0: 2 -> 3
  rank: 8.3 -> 3.6

yes_no_required:
  L24 mlp_out
  exact: 0 -> 0
  tok0: 0 -> 0
  rank: 13.8 -> 9.6
```

DS7B 关键事实：

```text
组合可以大幅推进 rank，
但仍不能把 correct_prefix 稳定推到 top1。
```

### 关键进展

Phase 658 将 Phase 657 的判断收紧为：

```text
final format prior 不是单点门，
而是多个 writer 的叠加压力。
```

跨模型分化：

```text
qwen3:
  format writer suppression 可以直接生成闭合。

GLM4:
  单点弱，组合明显增强，说明 attention writer 叠加有效。

DS7B:
  组合只推动 rank，不推动 exact。
  DS7B 的瓶颈不只是 L23/L24 MLP format prior。
```

### 对附件判断的修正

附件认为 Phase 653-655 的方向正确，这一点成立。但现在需要补充：

```text
Phase 653-655 找到的是 task intent gate 与 final policy gate 的边界；
Phase 656-658 进一步证明 final policy gate 内部含有可定位的 format-prior writer graph。
```

不过不能把这个结果夸大为完整机制闭合，因为 DS7B 明确显示：

```text
format-prior writer graph 被削弱后，
correct value token 仍可能输给更深的 format/protocol/readout gate。
```

### 问题和硬伤

1. Phase 658 仍是消融型实验，不是自然路径的正向构造。

2. 组合测试只在同一个 restore site 内进行，没有做跨 site source patch union，因此还没有完整映射全局图谱。

3. qwen3 和 GLM4 支持 multi-writer pressure，但 DS7B 只支持 rank-level pressure，不支持 generation-level closure。

4. top3 不一定比 top2 好，说明 component 之间可能存在方向冲突，不能简单累加。

5. DS7B 的 correct_prefix rank 被推到 3.6 但仍不生成，说明接下来不能继续只找同类 writer，而要定位 top2/top3 到 top1 的最后读出桥。

### 当前理论进展

当前可更新为：

```text
语言输出不是：
  semantic support -> answer token

而是：
  semantic support
  -> task intent gate
  -> protocol / format prior writer graph
  -> final token competition
  -> generation closure
```

其中 Phase 653-658 已经把后半段拆成：

```text
intent-gate restore:
  让 correct_prefix 进入候选区

format-prior writer suppression:
  降低 space/newline/explanation 等格式先验

combined writer suppression:
  在 qwen3/GLM4 上把候选推进到生成，
  在 DS7B 上只推进到 rank 3-4，未完全闭合
```

### 下一阶段

Phase 659 仍属于同一个大阶段的后续，因为 DS7B 的 bottleneck 已经被压缩到：

```text
correct_prefix 已经从很低 rank 被推进到 rank 3-4，
但仍不能越过最终 top1。
```

下一阶段不应继续泛化总结，而应做更客观的最后竞争审计：

```text
Phase 659: DS7B Final Top1 Barrier and Residual Readout Bridge Audit
```

目标：

```text
1. 固定 Phase 658 的 DS7B best combo。
2. 记录 remaining top1/top2 competitors。
3. 在最后若干层分别测试：
   residual add
   ln_f / final_norm input
   lm_head projection margin
   top competitor category
4. 判断 DS7B 的最后瓶颈是：
   remaining format prior
   candidate token projection weakness
   readout normalization distortion
   还是 generation policy preference。
5. 只做客观定位，不做理论闭合。
```

## Phase 659: Final Top1 Barrier and Residual Readout Bridge Audit [2026-06-26 07:44]

### 任务来源

Phase 658 已经证明 format-prior writer suppression 具有组合效应：qwen3 和 GLM4 的组合明显强于单点，DS7B 也出现大幅 rank 改善。但是 DS7B 仍未完成 exact/tok0 generation closure。因此 Phase 659 不继续盲目扩大 patch 搜索，而是固定 Phase 658 的 best combo，审计最后 top1 barrier。

### 生成脚本

```text
tests/gpt5/phase659_final_top1_barrier_readout_audit.py
tests/gpt5/phase659_final_top1_barrier_readout_audit_summary.py
```

### 运行命令

```bash
python -m py_compile tests/gpt5/phase659_final_top1_barrier_readout_audit.py tests/gpt5/phase659_final_top1_barrier_readout_audit_summary.py
python tests/gpt5/phase659_final_top1_barrier_readout_audit.py qwen3 --smoke --hard-exit-after-model
python tests/gpt5/phase659_final_top1_barrier_readout_audit.py qwen3 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase659_final_top1_barrier_readout_audit.py glm4 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase659_final_top1_barrier_readout_audit.py deepseek7b --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase659_final_top1_barrier_readout_audit_summary.py
```

### 测试原理

Phase 659 比较三种状态：

```text
baseline_task:
  不加 restore，不加 format writer suppression。

site_restore:
  使用 Phase 653-656 的 intent-gate restore patch。

combo_ablation:
  在 site_restore 基础上，
  使用 Phase 658 的 best format-prior writer suppression combo。
```

核心测量：

```text
correct_prefix rank
correct_prefix 与 top1 的 logit gap
correct_prefix 是否成为 top1
剩余 top1 competitor 的类别和具体词元
```

### 客观结果

输出目录：

```text
results/glm5_phase659_final_top1_barrier_readout_audit/
results/glm5_phase659_final_top1_barrier_readout_audit/phase659_cross_model_summary.md
```

三模型数据规模：

```text
qwen3:
  selected = 20
  rows = 240
  time = 0.37 min
  filtered = 0

GLM4:
  selected = 20
  rows = 240
  time = 0.54 min
  filtered = 0

DS7B:
  selected = 20
  rows = 240
  time = 0.47 min
  filtered = 0
```

qwen3：

```text
explanation_required + separator_input_edge top2:
  L16 attn_out + L20 attn_out
  rank: 3.1 -> 1.0
  top1_gap: 1.14 -> 0.00
  correct_top1: 4 -> 20
  remaining top1: correct_prefix 20/20

yes_no_required + early_peak_layer_out top2:
  L18 attn_out + L21 mlp_out
  rank: 4.2 -> 1.1
  top1_gap: 2.22 -> 0.08
  correct_top1: 5 -> 17
  remaining top1: correct_prefix 17, space 2, newline 1
```

qwen3 结论：

```text
qwen3 的 final top1 barrier 基本被 Phase 658 best combo 打开。
explanation_required 已经完全 top1 closure。
yes_no_required 仍有少量 space/newline 残留。
```

GLM4：

```text
yes_no_required + l22_peak_layer_out top2:
  L27 attn_out + L23 attn_out
  rank: 3.5 -> 2.0
  top1_gap: 1.13 -> 0.43
  correct_top1: 3 -> 8
  remaining top1: space 11, correct_prefix 8, word 1

explanation_required + l22_peak_layer_out top1:
  L23 attn_out
  rank: 2.1 -> 1.9
  top1_gap: 0.80 -> 0.45
  correct_top1: 5 -> 7
  remaining top1: space 11, correct_prefix 7, word 2
```

GLM4 结论：

```text
GLM4 的 final top1 barrier 被削弱但未打开。
主要剩余竞争者仍是 space。
```

DS7B：

```text
explanation_required + l22_peak_layer_out top2:
  L23 mlp_out + L24 mlp_out
  rank: 8.3 -> 3.6
  top1_gap: 2.19 -> 0.98
  correct_top1: 2 -> 3
  remaining top1: space 15, correct_prefix 3, newline 2

yes_no_required + l22_peak_layer_out top1:
  L24 mlp_out
  rank: 13.8 -> 9.6
  top1_gap: 2.81 -> 2.31
  correct_top1: 0 -> 0
  remaining top1: newline 15, space 5
```

DS7B 结论：

```text
DS7B 的 bottleneck 已被定位得更清楚：

explanation_required:
  剩余 top1 主要是 space。

yes_no_required:
  剩余 top1 主要是 newline。

这说明 DS7B 不是没有 semantic value support，
而是最后读出仍被强格式协议词元占据。
```

### 关键进展

Phase 659 把 Phase 658 的结果压缩成一个更小的客观事实：

```text
final generation failure
不是单纯 correct value token 不存在，
也不是 intent gate 完全无效，
而是 correct_prefix 与最终格式协议 token 的 top1 competition 没有完全越过。
```

模型差异：

```text
qwen3:
  best combo 可以几乎完成 top1 closure。

GLM4:
  best combo 只能削弱 space barrier。

DS7B:
  explanation 的 rank 可以推进到 3.6，但 space 仍压住；
  yes/no 的 newline barrier 仍很强。
```

### 对当前理论的更新

当前链条应改写为：

```text
semantic value support
-> task intent gate
-> format-prior writer graph
-> residual readout top1 barrier
-> generation closure
```

其中：

```text
task intent gate:
  负责把 correct_prefix 拉入竞争区。

format-prior writer graph:
  写入 space/newline/explanation 等格式先验。

residual readout top1 barrier:
  决定 correct_prefix 是否真正成为第一个生成 token。
```

### 问题和硬伤

1. Phase 659 是 readout observation，不是新的修复构造。

2. top1_gap 是 logit 层观察，不能直接等同于自然机制中的内部门控变量。

3. DS7B 的 yes/no 仍有 2.31 logit gap，说明仅消融 L24 mlp_out 不够。

4. GLM4 与 DS7B 的剩余 barrier 都是 format token，说明接下来不能回退到语义支持层，必须继续审计 format/protocol readout path。

5. 当前还没有区分：
   - space/newline 是由最终几层 residual 写入；
   - 还是 lm_head 对这些 token 的 projection 过强；
   - 或者 final norm 改变了候选 token 的相对尺度。

### 阶段性判断

Phase 653-659 完成了一个阶段性目标：

```text
从 intent-gate restore 失败，
定位到 format-prior writer graph，
再压缩到 final top1 barrier。
```

这个阶段的结论不是“语言机制已闭合”，而是：

```text
在 value-answer 任务中，
最终失败点已经从广义语义支持层，
移动到明确的格式协议读出竞争层。
```

### 下一阶段

下一阶段仍然可以继续，但已经进入新的子阶段：

```text
Phase 660: Space/Newline Residual Readout Source Backtrace
```

目标：

```text
1. 固定 Phase 659 的 DS7B failure cases。
2. 针对 remaining top1 = space/newline 的样本。
3. 从最后读出向前回溯：
   final_norm input
   final_norm output
   last 4 residual additions
   lm_head projection contribution
4. 分别判断 space/newline barrier 是：
   residual state 强；
   final_norm 放大；
   unembedding projection 强；
   还是前层格式 writer 残留未消除。
5. 暂不做大范围搜索，先完成 top1 barrier 的局部图谱。
```

## Phase 660: Space/Newline Residual Readout Source Backtrace [2026-06-26 08:05]

### 任务来源

附件对 Phase 653-659 的分析基本正确：当前研究已经从 task intent gate、format-prior writer graph 推进到 residual readout top1 barrier。必须收紧的是：Phase 659 仍是 readout observation，不是完整因果闭合；DS7B 的最终读出桥仍未解决。因此 Phase 660 固定 Phase 659/658 的 best combo，回溯 space/newline top1 barrier 的来源。

### 生成脚本

```text
tests/gpt5/phase660_space_newline_readout_source_backtrace.py
tests/gpt5/phase660_space_newline_readout_source_backtrace_summary.py
```

### 运行命令

```bash
python -m py_compile tests/gpt5/phase660_space_newline_readout_source_backtrace.py tests/gpt5/phase660_space_newline_readout_source_backtrace_summary.py
python tests/gpt5/phase660_space_newline_readout_source_backtrace.py qwen3 --smoke --hard-exit-after-model
python tests/gpt5/phase660_space_newline_readout_source_backtrace.py qwen3 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase660_space_newline_readout_source_backtrace.py glm4 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase660_space_newline_readout_source_backtrace.py deepseek7b --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase660_space_newline_readout_source_backtrace_summary.py
```

### 测试原理

Phase 660 比较：

```text
site_restore:
  只使用 intent-gate restore patch。

combo_ablation:
  使用 Phase 658 best format-prior suppression combo。

last4_L{layer}_{component}:
  在 combo_ablation 基础上，
  继续消融最后 4 层的 attn_out / mlp_out。
```

同时记录：

```text
pre-final-norm projection gap
post-final-norm / lm_head gap
norm_gap_shift = post_gap - pre_gap
remaining top1 category
```

注意：pre-final-norm projection 是诊断指标，不是模型真实生成路径，因为真实路径必须经过 final_norm。

### 客观结果

输出目录：

```text
results/glm5_phase660_space_newline_readout_source_backtrace/
results/glm5_phase660_space_newline_readout_source_backtrace/phase660_cross_model_summary.md
```

数据规模：

```text
qwen3:
  selected = 20
  rows = 880
  time = 0.93 min
  last_layers = [32, 33, 34, 35]
  filtered = 0

GLM4:
  selected = 20
  rows = 880
  time = 1.22 min
  last_layers = [36, 37, 38, 39]
  filtered = 0

DS7B:
  selected = 20
  rows = 880
  time = 1.11 min
  last_layers = [24, 25, 26, 27]
  filtered = 0
```

qwen3：

```text
yes_no_required + early_peak_layer_out top2:
  gap: 2.22 -> 0.08
  rank: 4.2 -> 1.1
  top1: correct_prefix 16, space 3, newline 1
  norm_shift: -19.92

explanation_required + separator_input_edge top2:
  gap: 1.14 -> 0.00
  rank: 3.1 -> 1.0
  top1: correct_prefix 20
  norm_shift: -12.31
```

qwen3 last-writer 影响较小：

```text
L35 mlp_out / L32 mlp_out:
  对少量残留 space/newline 有改善，
  但整体已经接近闭合。
```

GLM4：

```text
yes_no_required + l22_peak_layer_out top2:
  gap: 1.13 -> 0.43
  rank: 3.5 -> 2.0
  top1: space 11, correct_prefix 8, word 1
  norm_shift: -36.40

explanation_required + l22_peak_layer_out top1:
  gap: 0.80 -> 0.45
  rank: 2.1 -> 1.9
  top1: space 12, correct_prefix 6, word 2
  norm_shift: -32.71
```

GLM4 strongest last-writer：

```text
L36 attn_out:
  explanation gap_delta = +0.18
  yes_no gap_delta = +0.14

L36 mlp_out:
  explanation gap_delta = +0.13
  yes_no gap_delta = +0.12
```

DS7B：

```text
explanation_required + top2:
  gap: 2.19 -> 0.98
  rank: 8.3 -> 3.6
  top1: space 15, correct_prefix 3, newline 2
  norm_shift: -394.97

yes_no_required + top1:
  gap: 2.81 -> 2.31
  rank: 13.8 -> 9.6
  top1: newline 17, space 3
  norm_shift: -518.00
```

DS7B strongest last-writer：

```text
yes_no_required:
  L25 attn_out:
    gap_delta = +0.35
    rank_delta = +2.50
    top1: space 11, newline 8, correct_prefix 1

  L26 mlp_out:
    gap_delta = +0.29
    rank_delta = +1.00
    top1: space 10, newline 7, correct_prefix 3

explanation_required:
  L26 mlp_out:
    gap_delta = +0.17
    rank_delta = +0.65
    top1: space 11, correct_prefix 9
```

### 关键进展

Phase 660 把 DS7B 的剩余 top1 barrier 从：

```text
space / newline 仍然压住 correct_prefix
```

进一步定位为：

```text
yes_no_required:
  L25 attn_out + L26 mlp_out 仍在读出端支撑 newline/space barrier。

explanation_required:
  L26 mlp_out 是最明显的剩余 space barrier contributor。
```

同时，final_norm 结论必须谨慎：

```text
pre_norm projection 与 post_norm projection 的尺度差异巨大。
这说明 raw residual state 不能直接解释最终 lm_head 读出；
final_norm 是真实读出链条中不可省略的强变换。
```

### 问题和硬伤

1. Phase 660 的 last-writer 测试仍是消融，不是正向恢复。
2. `norm_gap_shift` 数值很大，说明 pre_norm projection 只适合作诊断，不适合作直接理论变量。
3. DS7B yes_no 仍有明显 gap，说明 L25/L26 可能只是剩余 barrier 的一部分。
4. 当前尚未确认 last-writer candidates 叠加后是否能传到短生成。

### 下一步

Phase 661 与 Phase 660 处于同一子阶段，因为 Phase 660 只定位了 last writer candidates，还没有验证它们是否能传导到 generation closure。

目标：

```text
Phase 661: Last-Writer Combo Generation Closure

1. 读取 Phase 660 strongest last-writer candidates。
2. 与 Phase 658 best combo 叠加。
3. 测试 exact / tok0 / top1_gap 是否进一步改善。
4. 特别检查 DS7B 的 yes_no newline barrier 是否能被 L25 attn_out + L26 mlp_out 打开。
```

## Phase 661: Last-Writer Combo Generation Closure [2026-06-26 08:05]

### 生成脚本

```text
tests/gpt5/phase661_last_writer_combo_generation_closure.py
tests/gpt5/phase661_last_writer_combo_generation_closure_summary.py
```

### 运行命令

```bash
python -m py_compile tests/gpt5/phase661_last_writer_combo_generation_closure.py tests/gpt5/phase661_last_writer_combo_generation_closure_summary.py
python tests/gpt5/phase661_last_writer_combo_generation_closure.py qwen3 --smoke --hard-exit-after-model
python tests/gpt5/phase661_last_writer_combo_generation_closure.py qwen3 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase661_last_writer_combo_generation_closure.py glm4 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase661_last_writer_combo_generation_closure.py deepseek7b --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase661_last_writer_combo_generation_closure_summary.py
```

### 测试原理

Phase 661 比较：

```text
site_restore:
  只使用 intent-gate restore patch。

phase658_combo:
  使用 Phase 658 best format-prior suppression combo。

plus_last_writers:
  在 phase658_combo 基础上，
  叠加 Phase 660 strongest last-writer ablations。
```

测量：

```text
exact_correct
tok0 correct_prefix
prefix_rank
top1_gap
remaining top1 category
short greedy generation text
```

### 客观结果

输出目录：

```text
results/glm5_phase661_last_writer_combo_generation_closure/
results/glm5_phase661_last_writer_combo_generation_closure/phase661_cross_model_summary.md
```

数据规模：

```text
qwen3:
  selected = 20
  rows = 240
  time = 0.96 min
  filtered = 0

GLM4:
  selected = 20
  rows = 240
  time = 1.42 min
  filtered = 0

DS7B:
  selected = 20
  rows = 240
  time = 1.21 min
  filtered = 0
```

qwen3：

```text
yes_no_required + early_peak_layer_out top2:
  extra: L35 mlp_out + L32 mlp_out
  exact: 16 -> 20
  tok0: 16 -> 20
  gap: 0.08 -> 0.00
  top1: correct_prefix 20

explanation_required + separator_input_edge top1:
  extra: L35 mlp_out + L32 mlp_out
  exact: 12 -> 16
  tok0: 14 -> 19
  gap: 0.29 -> 0.03
```

GLM4：

```text
explanation_required + l22_peak_layer_out top1:
  extra: L36 attn_out + L36 mlp_out
  exact: 6 -> 14
  tok0: 7 -> 15
  gap: 0.45 -> 0.21
  top1: correct_prefix 15, space 5

yes_no_required + l22_peak_layer_out top2:
  extra: L36 attn_out + L36 mlp_out
  exact: 8 -> 13
  tok0: 8 -> 13
  gap: 0.43 -> 0.22
  top1: correct_prefix 13, space 5, word 2
```

DS7B：

```text
yes_no_required + l22_peak_layer_out top1:
  extra: L25 attn_out + L26 mlp_out
  exact: 0 -> 9
  tok0: 0 -> 9
  gap: 2.31 -> 1.17
  top1: correct_prefix 9, space 7, newline 4

explanation_required + l22_peak_layer_out top2:
  extra: L26 mlp_out
  exact: 3 -> 9
  tok0: 3 -> 9
  gap: 0.98 -> 0.81
  top1: space 11, correct_prefix 9
```

### 关键进展

Phase 661 是重要正结果：

```text
Phase 660 的 last-writer candidates
不是纯 readout observation，
叠加后可以传导到 short greedy generation。
```

尤其 DS7B：

```text
yes_no_required 从 0/20 exact 提升到 9/20 exact。
explanation_required 从 3/20 exact 提升到 9/20 exact。
```

这说明 DS7B 的 remaining barrier 确实包含：

```text
L25 attn_out
L26 mlp_out
```

等后段读出写入器，而不仅仅是 Phase 656 的 L23/L24 MLP。

### 对附件判断的修正

附件判断 Phase 653-659 已把问题推进到 final top1 barrier，这是正确的。Phase 660-661 进一步修正为：

```text
final top1 barrier 不是不可分解黑箱；
它至少还可以拆成：

1. Phase 658 format-prior writer combo。
2. Phase 660/661 last residual writer combo。
3. remaining projection / normalization / decoder preference。
```

### 问题和硬伤

1. DS7B 仍未完全闭合：

```text
yes_no_required:
  exact 9/20
  remaining top1: space 7, newline 4

explanation_required:
  exact 9/20
  remaining top1: space 11
```

2. Phase 661 仍使用消融式 suppression，不是自然正向机制构造。

3. 对 qwen3 和 GLM4，继续叠加 last-writer suppression 已经接近上限，但仍可能改变自然性。

4. DS7B 的剩余 gap 仍较大，说明还存在 projection-level 或更深层 policy preference。

### 当前理论进展

链条进一步细化为：

```text
semantic value support
-> task intent gate
-> format-prior writer graph
-> late residual writer graph
-> final_norm / lm_head readout
-> top1 token competition
-> generation closure
```

对于 DS7B：

```text
L23/L24 MLP:
  早段/中段 format-prior suppression 有效，但不足。

L25 attn_out + L26 mlp_out:
  后段 residual readout writer，对 yes_no newline/space barrier 有强影响。

remaining barrier:
  仍主要是 space/newline，可能还需要 projection-level audit。
```

### 阶段性判断

Phase 660-661 完成了当前子阶段目标：

```text
从 DS7B final top1 barrier
回溯到后段 residual readout writer，
并证明这些 writer 的组合能显著改善生成。
```

但尚未完成完整闭合：

```text
DS7B 仍不是 20/20 closure。
```

### 下一阶段

下一步已经进入新的子阶段，不应继续盲目扩大 writer suppression，而应审计剩余 projection-level barrier：

```text
Phase 662: Residual-to-LMHead Projection Barrier Audit
```

目标：

```text
1. 固定 Phase 661 的 DS7B partially repaired cases。
2. 只分析 remaining failure cases。
3. 比较 correct_prefix、space、newline 的 unembedding direction。
4. 测试 final_norm output 中是否已经接近 correct_prefix direction。
5. 判断剩余失败是：
   residual writer 未清除；
   final_norm output 方向不够；
还是 lm_head 对 space/newline 的 projection advantage。
```

## Phase 662: Residual-to-LMHead Projection Barrier Audit [2026-06-26 08:31]

### 任务背景

本阶段读取并审视了用户上传的 Phase 660-661 分析。总体判断：附件中的主判断基本正确。

Phase 660-661 的正确部分是：

```text
1. Phase 660 将 DS7B 的剩余 space/newline barrier 从最终输出现象回溯到后段 writer。
2. Phase 661 证明 last-writer combo 不是纯读出噪声，而能实质改善自然生成。
3. 但 Phase 661 仍未完成闭合：DS7B 的 yes_no / explanation 都仍有大量 space/newline 或后续生成失败。
4. 因此下一步不应继续盲目扩大 writer suppression，而应进入 residual-to-lm_head projection barrier audit。
```

本阶段目标是固定 Phase 661 的 partially repaired state，检查剩余失败到底来自：

```text
1. residual state 仍未对齐 correct value token；
2. final_norm 改写或压缩了读出差距；
3. lm_head/unembedding 对 space/newline 存在 projection advantage；
4. 第一 token 已正确，但后续自然生成仍失败。
```

### 生成脚本

```text
tests/gpt5/phase662_residual_to_lmhead_projection_barrier_audit.py
tests/gpt5/phase662_residual_to_lmhead_projection_barrier_audit_summary.py
```

### 执行命令

```bash
python -m py_compile tests/gpt5/phase662_residual_to_lmhead_projection_barrier_audit.py tests/gpt5/phase662_residual_to_lmhead_projection_barrier_audit_summary.py

python tests/gpt5/phase662_residual_to_lmhead_projection_barrier_audit.py qwen3 --smoke --hard-exit-after-model

python tests/gpt5/phase662_residual_to_lmhead_projection_barrier_audit.py qwen3 --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase662_residual_to_lmhead_projection_barrier_audit.py glm4 --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase662_residual_to_lmhead_projection_barrier_audit.py deepseek7b --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase662_residual_to_lmhead_projection_barrier_audit_summary.py
```

### 输出文件

```text
results/glm5_phase662_residual_to_lmhead_projection_barrier_audit/
results/glm5_phase662_residual_to_lmhead_projection_barrier_audit/phase662_cross_model_summary.md
results/glm5_phase662_residual_to_lmhead_projection_barrier_audit/phase662_qwen3_residual_to_lmhead_projection_barrier_audit_confirm.json
results/glm5_phase662_residual_to_lmhead_projection_barrier_audit/phase662_glm4_residual_to_lmhead_projection_barrier_audit_confirm.json
results/glm5_phase662_residual_to_lmhead_projection_barrier_audit/phase662_deepseek7b_residual_to_lmhead_projection_barrier_audit_confirm.json
```

### 测试原理

本阶段不重新寻找新的 patch，而是在 Phase 661 已经有效的 plus_last_writers 条件下，只审计剩余失败样本。

核心测量对象：

```text
1. post_gap:
   final_norm output 经 lm_head 后，competitor token 相对 correct_prefix 的真实 logit 优势。

2. pre_gap:
   final_norm input 直接接 lm_head 的诊断性 logit 差距。
   注意：pre_gap 不是模型真实输出，只用于观察 final_norm 前后压缩/改写。

3. norm_gap_change:
   post_gap - pre_gap。
   如果该值为巨大负数，说明 final_norm 强烈压缩了直接投影差距。

4. needed_unit_delta:
   如果想让 correct_prefix 追平 competitor，在 unembedding 差分方向上最少需要移动多少单位。

5. correct_cos / competitor_cos / competitor_norm_advantage:
   用于区分失败来自 hidden-state direction alignment，还是 competitor token unembedding norm/projection advantage。
```

### 主要结果

#### Qwen3

```text
样本数：20

yes_no early top2:
  phase658_combo exact_rate = 0.80
  plus_last_writers exact_rate = 1.00

yes_no early top3:
  phase658_combo exact_rate = 0.80
  plus_last_writers exact_rate = 0.95

explanation separator top1:
  phase658_combo exact_rate = 0.60
  plus_last_writers exact_rate = 0.80

explanation separator top2:
  phase658_combo exact_rate = 0.90
  plus_last_writers exact_rate = 0.90
```

Qwen3 的 remaining failures 很少。最重要现象是：部分失败样本的 first-token top1 已经是 correct_prefix，但 exact 仍失败。

```text
explanation top1 = correct_prefix:
  n = 5
  post_gap = 0.00
  pre_gap = 36.27
  norm_gap_change = -36.27
```

这说明 Qwen3 的剩余问题已经不主要是第一 token 的 projection barrier，而是后续自然生成质量或 answer continuation 的问题。

#### GLM4

```text
样本数：20

explanation l22 top1:
  phase658_combo exact_rate = 0.30
  plus_last_writers exact_rate = 0.70

yes_no l22:
  phase658_combo exact_rate = 0.40
  plus_last_writers exact_rate = 0.65
```

GLM4 的剩余失败仍主要是 space 和 word competitor。

```text
explanation top1 = space:
  n = 10
  post_gap = 0.82
  pre_gap = 25.65
  norm_gap_change = -24.82
  needed_unit_delta = 0.881
  correct_cos = 0.087
  competitor_cos = 0.094
  competitor_norm_advantage = -0.005

yes_no top1 = space:
  n = 10
  post_gap = 0.79
  pre_gap = 27.91
  norm_gap_change = -27.12
  needed_unit_delta = 0.841
  correct_cos = 0.090
  competitor_cos = 0.096
  competitor_norm_advantage = -0.005
```

GLM4 的 space barrier 更像 hidden-state direction alignment 问题，而不是 space token 本身的 norm advantage。因为 competitor_cos 高于 correct_cos，但 competitor_norm_advantage 近似为 0 或略负。

#### DS7B

```text
样本数：20

explanation l22/late top2:
  phase658_combo exact_rate = 0.15
  plus_last_writers exact_rate = 0.45

yes_no l22/late top1:
  phase658_combo exact_rate = 0.00
  plus_last_writers exact_rate = 0.45
```

DS7B 的剩余失败最关键，而且出现了清楚的二分：

```text
space failure:
  explanation top1 = space:
    n = 22
    post_gap = 1.47
    pre_gap = 385.67
    norm_gap_change = -384.20
    needed_unit_delta = 1.104
    correct_cos = 0.086
    competitor_cos = 0.073
    competitor_norm_advantage = 0.260

  yes_no top1 = space:
    n = 14
    post_gap = 1.70
    pre_gap = 395.14
    norm_gap_change = -393.45
    needed_unit_delta = 1.277
    correct_cos = 0.090
    competitor_cos = 0.077
    competitor_norm_advantage = 0.260
```

DS7B 的 space failure 中，correct_cos 反而高于 competitor_cos，但 space 仍赢。这说明 space 失败很可能不是 residual direction 没对齐，而是 space token 在 unembedding / projection norm 上有优势。

```text
newline failure:
  yes_no top1 = newline:
    n = 8
    post_gap = 2.86
    pre_gap = 454.56
    norm_gap_change = -451.70
    needed_unit_delta = 2.218
    correct_cos = 0.083
    competitor_cos = 0.103
    competitor_norm_advantage = -0.041
```

DS7B 的 newline failure 与 space failure 不同。newline 的 competitor_cos 明显高于 correct_cos，norm advantage 反而不是主要来源。这说明 newline failure 更像 final hidden direction alignment 问题。

### 客观进展

Phase 662 把 Phase 661 之后的剩余瓶颈从一个笼统说法：

```text
space/newline 仍然竞争 correct value token
```

进一步拆成两个不同机制：

```text
1. DS7B space barrier:
   correct direction 已不弱，但 space unembedding/projection norm advantage 仍能把它推到 top1。

2. DS7B newline barrier:
   newline 更像 hidden-state direction alignment 胜出，而不是 norm advantage 胜出。

3. GLM4 space barrier:
   主要像 hidden-state direction alignment 问题。

4. Qwen3 residual failure:
   很多已经越过 first-token barrier，剩余问题转向后续生成闭合。
```

这说明 “format/prefix/value gate” 不能再用单一机制解释。至少要拆成：

```text
semantic value path
format prior path
projection norm path
continuation generation path
```

### 理论进展

本阶段支持一个更细的读出公式：

```text
logit_i
=
W_i \cdot \mathrm{Norm}(h)
+ b_i
```

更细拆成：

```text
logit_i
=
\|W_i\| \cdot \|\mathrm{Norm}(h)\| \cdot \cos(\mathrm{Norm}(h), W_i)
+ b_i
```

因此 correct value token 是否胜出，不仅取决于语义方向是否存在，还取决于：

```text
1. hidden state 是否朝向 correct token；
2. competitor token 的 unembedding norm 是否有优势；
3. final_norm 是否压缩或改写了差距；
4. 后续 token generation 是否保持正确轨迹。
```

当前更准确的统一链条是：

```text
prompt state
-> value evidence path
-> format/protocol path
-> late writer residual repair
-> final_norm projection geometry
-> lm_head token competition
-> continuation generation closure
```

### 硬伤和边界

```text
1. pre_gap 是诊断指标，不是模型真实输出。
   它不能直接证明 final_norm “犯错”，只能证明 final_norm 前后的直接投影差距发生巨大变化。

2. 本阶段没有进行 causal projection intervention。
   因此目前只能说 DS7B space failure 显示 projection norm advantage 迹象，不能说已经因果证明。

3. DS7B 的 n=22 / n=14 统计来自不同 task/site 的 failure 聚合，不是 22 个完全独立 prompt。
   后续应按 task、site、competitor 分层验证。

4. Qwen3 的 exact failure with correct_prefix top1 说明第一 token 指标不足。
   后续必须把 first-token closure 与 continuation closure 分开。

5. GLM4 和 DS7B 的失败来源不同，不能把 DS7B 结论直接推广到所有模型。
```

### 下一阶段判断

Phase 662 已经完成当前 projection barrier audit 的阶段目标。

接下来如果继续同一大研究链条，应该进入一个新的干预子阶段：

```text
Phase 663: Projection-Specific Causal Intervention Audit
```

核心目标：

```text
1. 对 DS7B space failure 做 norm-neutralized lm_head readout。
2. 对 DS7B newline failure 做 direction-only correction。
3. 对 GLM4 space failure 做 hidden-direction correction。
4. 区分 projection norm barrier 与 hidden direction barrier 的因果贡献。
5. 对 Qwen3 correct-prefix-but-wrong-continuation 样本做 continuation-level generation audit。
```

这已经属于新的子阶段，不应在 Phase 662 内继续扩大测试，否则会把诊断审计和因果干预混在一起。

## Phase 663: Projection-Specific Causal Intervention Audit [2026-06-26 08:57]

### 任务背景

本阶段读取并审视了用户上传的 Phase 662 分析。总体判断：附件中的主判断基本正确，而且指出了 Phase 662 的关键硬伤：

```text
Phase 662 只是 projection barrier diagnosis，
还没有完成 projection-specific causal intervention。
```

因此本阶段继续同一研究链条，进入新的干预子阶段：

```text
Phase 663: Projection-Specific Causal Intervention Audit
```

目标不是继续扩大 writer suppression，而是直接验证 Phase 662 的两个分叉判断：

```text
1. DS7B space failure 是否真的主要来自 unembedding / projection norm advantage。
2. DS7B newline failure 是否真的主要来自 hidden-state direction alignment。
3. GLM4 space failure 是否主要是 hidden direction problem，而不是 norm problem。
4. Qwen3 是否已经进入 continuation-level failure。
```

### 生成脚本

```text
tests/gpt5/phase663_projection_specific_causal_intervention_audit.py
tests/gpt5/phase663_projection_specific_causal_intervention_audit_summary.py
```

### 执行命令

```bash
python -m py_compile tests/gpt5/phase663_projection_specific_causal_intervention_audit.py tests/gpt5/phase663_projection_specific_causal_intervention_audit_summary.py

python tests/gpt5/phase663_projection_specific_causal_intervention_audit.py qwen3 --smoke --hard-exit-after-model

python tests/gpt5/phase663_projection_specific_causal_intervention_audit.py qwen3 --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase663_projection_specific_causal_intervention_audit.py glm4 --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase663_projection_specific_causal_intervention_audit.py deepseek7b --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase663_projection_specific_causal_intervention_audit_summary.py
```

### 输出文件

```text
results/glm5_phase663_projection_specific_causal_intervention_audit/
results/glm5_phase663_projection_specific_causal_intervention_audit/phase663_cross_model_summary.md
results/glm5_phase663_projection_specific_causal_intervention_audit/phase663_qwen3_projection_specific_causal_intervention_audit_confirm.json
results/glm5_phase663_projection_specific_causal_intervention_audit/phase663_glm4_projection_specific_causal_intervention_audit_confirm.json
results/glm5_phase663_projection_specific_causal_intervention_audit/phase663_deepseek7b_projection_specific_causal_intervention_audit_confirm.json
```

### 测试原理

本阶段使用 Phase 661 的 plus_last_writers repaired state，不再寻找新 writer。

测试分两类：

```text
1. norm-neutralized pair readout:
   对 correct_prefix 和当前 top1 competitor 做范数中和比较。
   如果 actual logits 中 competitor 胜出，但去掉 unembedding norm 后 correct_prefix 胜出，
   则说明 competitor 的胜出很可能依赖 projection norm advantage。

2. direction correction:
   在 final_norm output 之后，沿 W_correct - W_competitor 方向移动隐藏状态，
   测试 correct_prefix 是否能成为 top1。
   这只是在读出端做因果反事实干预，不等于真实生成链路已经修复。
```

核心判断量：

```text
actual_gap = logit_competitor - logit_correct
neutral_cos_gap = cos(h, W_competitor) - cos(h, W_correct)

如果 actual_gap > 0 且 neutral_cos_gap < 0：
  competitor 实际赢，但去掉范数后 correct direction 更强。
```

### 主要结果

#### Qwen3

```text
selected_items = 32
rows = 128
```

实际 plus_last_writers 状态：

```text
yes_no top2:
  exact_rate = 1.000
  correct_top1_rate = 1.000

yes_no top3:
  exact_rate = 0.906
  correct_top1_rate = 0.875

explanation top1:
  exact_rate = 0.719
  correct_top1_rate = 0.812

explanation top2:
  exact_rate = 0.875
  correct_top1_rate = 0.938
```

剩余失败：

```text
explanation word:
  n = 7
  norm_neutral_flip_rate = 0.000
  actual_gap = 0.857
  neutral_cos_gap = 0.0103

yes_no explanation:
  n = 2
  norm_neutral_flip_rate = 1.000
  actual_gap = 0.156
  neutral_cos_gap = -0.0074

explanation space:
  n = 1
  norm_neutral_flip_rate = 1.000

yes_no newline:
  n = 1
  norm_neutral_flip_rate = 1.000
```

Qwen3 的主要剩余问题不是稳定的 projection norm barrier，而是两类小样本残留：

```text
1. explanation word competitor：方向仍偏向 word。
2. correct_prefix 已经 top1 但 exact wrong：continuation failure。
```

续写失败：

```text
explanation top1:
  correct_prefix_but_generation_wrong = 3

explanation top2:
  correct_prefix_but_generation_wrong = 2
```

这进一步支持：Qwen3 已经从 first-token barrier 转向 continuation-level audit。

#### GLM4

```text
selected_items = 32
rows = 128
```

实际 plus_last_writers 状态：

```text
explanation l22 top1:
  exact_rate = 0.781
  correct_top1_rate = 0.812

explanation late top1:
  exact_rate = 0.719
  correct_top1_rate = 0.812

yes_no l22 top2:
  exact_rate = 0.688
  correct_top1_rate = 0.688

yes_no late top2:
  exact_rate = 0.688
  correct_top1_rate = 0.688
```

范数中和结果：

```text
explanation space:
  n = 10
  norm_neutral_flip_rate = 0.000
  actual_gap = 0.825
  neutral_cos_gap = 0.0068
  norm_adv = -0.0045

yes_no space:
  n = 10
  norm_neutral_flip_rate = 0.000
  actual_gap = 0.787
  neutral_cos_gap = 0.0067
  norm_adv = -0.0045

yes_no word:
  n = 10
  norm_neutral_flip_rate = 0.000
  neutral_cos_gap = 0.0153
```

GLM4 的 space/word 失败在范数中和后完全不翻转，说明它不是 projection norm advantage 主导，而是 competitor direction alignment 主导。

方向修正结果：

```text
explanation space:
  scale 1.5 correct_top1_rate = 1.000

yes_no space:
  scale 1.5 correct_top1_rate = 0.800

yes_no word:
  scale 2.0 correct_top1_rate = 1.000
```

这说明 GLM4 的剩余问题可以通过 hidden direction correction 明显改善，和 Phase 662 的判断一致。

#### DS7B

```text
selected_items = 32
rows = 128
```

实际 plus_last_writers 状态：

```text
explanation l22 top2:
  exact_rate = 0.500
  correct_top1_rate = 0.500

explanation late top2:
  exact_rate = 0.500
  correct_top1_rate = 0.500

yes_no l22 top1:
  exact_rate = 0.469
  correct_top1_rate = 0.500

yes_no late top1:
  exact_rate = 0.469
  correct_top1_rate = 0.500
```

DS7B 的关键结果：

```text
explanation space:
  n = 32
  actual_gap = 1.453
  neutral_cos_gap = -0.0133
  norm_neutral_flip_rate = 1.000
  norm_adv = 0.2603

yes_no space:
  n = 24
  actual_gap = 1.750
  neutral_cos_gap = -0.0132
  norm_neutral_flip_rate = 0.917
  norm_adv = 0.2603

yes_no newline:
  n = 8
  actual_gap = 2.859
  neutral_cos_gap = 0.0197
  norm_neutral_flip_rate = 0.000
  norm_adv = -0.0408
```

这是本阶段最关键的结果。

DS7B space failure 中：

```text
actual logits 里 space 胜出；
但 norm-neutralized pair readout 后，correct_prefix 基本翻转胜出。
```

这给 Phase 662 的判断增加了更强证据：

```text
DS7B space barrier 很大部分确实来自 unembedding / projection norm advantage。
```

DS7B newline failure 中：

```text
norm-neutralized flip_rate = 0.000
neutral_cos_gap > 0
```

这说明 newline 不是靠 norm advantage 赢，而是 final hidden direction 真的更接近 newline direction。

方向修正结果：

```text
explanation space:
  scale 1.5 correct_top1_rate = 0.625
  scale 2.0 correct_top1_rate = 0.750

yes_no space:
  scale 1.5 correct_top1_rate = 0.417
  scale 2.0 correct_top1_rate = 0.417

yes_no newline:
  scale 2.0 correct_top1_rate = 0.250
```

这说明 DS7B 即使做 direction correction，仍然经常被 newline 或 space 抢走，尤其 yes_no 任务中存在多竞争者结构，不是单一 competitor 修正即可解决。

### 客观进展

Phase 663 将 Phase 662 的诊断推进为更强的反事实证据：

```text
1. DS7B space barrier:
   由 projection norm advantage 主导的证据很强。

2. DS7B newline barrier:
   不由 norm advantage 主导，而是 hidden direction alignment 主导。

3. GLM4 space/word barrier:
   不由 norm advantage 主导，而是 hidden direction alignment 主导。

4. Qwen3:
   first-token barrier 已明显弱化，continuation failure 开始成为主要新瓶颈。
```

因此 format prior 不能继续被视为一个单一门控，而应拆成：

```text
format writer
hidden direction alignment
projection norm geometry
multi-competitor readout competition
continuation controller
```

### 理论进展

Phase 663 支持把最终读出进一步拆成两个可区分项：

```text
logit_i
=
\|W_i\| \cdot \|\hat{h}\| \cdot \cos(\hat{h}, W_i)
+ b_i
```

其中：

```text
hidden direction alignment:
  \cos(\hat{h}, W_i)

projection norm advantage:
  \|W_i\|
```

这说明 token competition 至少有两种不同胜出方式：

```text
1. direction-win:
   competitor direction 与 hidden state 更接近。

2. norm-win:
   competitor direction 不一定更接近，但 unembedding norm / projection geometry 让它胜出。
```

当前更准确的统一链条更新为：

```text
prompt conditioned residual state
-> task / protocol state
-> semantic value support
-> format prior writer suppression
-> late residual writer repair
-> hidden direction alignment
-> projection norm geometry
-> multi-token competition
-> continuation generation controller
```

### 问题和硬伤

```text
1. 本阶段是 readout-level causal intervention，不是 full generation intervention。
   它能证明 projection geometry 对 top1 competition 有因果作用，
   但不能证明真实自回归轨迹会自动修复。

2. norm-neutralized pair readout 只比较 correct_prefix 和当前 top1 competitor。
   它没有完整重排整个 vocabulary。
   但对于判断当前 competitor 是否依赖 norm advantage 已经足够有信息量。

3. direction correction 是 post-final_norm intervention。
   它不是网络内部自然产生的修复路径。

4. DS7B yes_no space 在 direction correction 后经常转向 newline。
   说明 DS7B 是 multi-competitor barrier，不是 simple binary barrier。

5. Qwen3 continuation failure 还没有被机制定位。
   目前只能确认它已经不是纯 first-token projection problem。
```

### 下一阶段判断

Phase 663 完成了 projection-specific causal audit 的阶段目标。

接下来如果继续同一大阶段，应进入：

```text
Phase 664: Multi-Competitor Readout and Continuation Split Audit
```

核心目标：

```text
1. DS7B yes_no 中同时跟踪 correct_prefix、space、newline 三方竞争。
2. 测试 single-competitor correction 为什么会从 space 转向 newline。
3. 构造 multi-competitor margin：
   correct_prefix - max(space, newline, word, explanation)
4. 对 Qwen3 / GLM4 的 correct_prefix top1 but exact wrong 样本做 token1/token2 continuation audit。
5. 区分 first-token readout closure 与 continuation closure。
```

这仍属于当前 “读出竞争到生成闭合” 大阶段，但已经是新的子问题：从 pairwise projection intervention 转向 multi-competitor + continuation split。

## Phase 664: Multi-Competitor Readout and Continuation Split Audit [2026-06-26 09:05]

### 任务背景

Phase 663 已经证明：

```text
1. DS7B space failure 很大程度来自 projection norm advantage。
2. DS7B newline failure 更像 hidden direction alignment。
3. GLM4 space / word failure 更像 hidden direction alignment。
4. Qwen3 已出现 correct_prefix top1 但 exact wrong 的 continuation failure。
```

但 Phase 663 仍有一个硬伤：

```text
pairwise correction 只针对当前 top1 competitor。
```

如果 correct_prefix 打败当前 competitor 后，另一个 competitor 立刻接管 top1，那么 pairwise correction 会高估机制闭合度。

因此 Phase 664 的目标是：

```text
1. 构造 multi-competitor margin。
2. 同时追踪 space / newline / word / explanation。
3. 测试 multi-competitor correction 是否比 pairwise correction 更接近读出闭合。
4. 对 correct_prefix top1 但 exact wrong 的样本做 token1 / token2 continuation audit。
```

### 生成脚本

```text
tests/gpt5/phase664_multi_competitor_continuation_split_audit.py
tests/gpt5/phase664_multi_competitor_continuation_split_audit_summary.py
```

### 执行命令

```bash
python -m py_compile tests/gpt5/phase664_multi_competitor_continuation_split_audit.py tests/gpt5/phase664_multi_competitor_continuation_split_audit_summary.py

python tests/gpt5/phase664_multi_competitor_continuation_split_audit.py qwen3 --smoke --hard-exit-after-model

python tests/gpt5/phase664_multi_competitor_continuation_split_audit.py qwen3 --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase664_multi_competitor_continuation_split_audit.py glm4 --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase664_multi_competitor_continuation_split_audit.py deepseek7b --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase664_multi_competitor_continuation_split_audit_summary.py
```

### 输出文件

```text
results/glm5_phase664_multi_competitor_continuation_split_audit/
results/glm5_phase664_multi_competitor_continuation_split_audit/phase664_cross_model_summary.md
results/glm5_phase664_multi_competitor_continuation_split_audit/phase664_qwen3_multi_competitor_continuation_split_audit_confirm.json
results/glm5_phase664_multi_competitor_continuation_split_audit/phase664_glm4_multi_competitor_continuation_split_audit_confirm.json
results/glm5_phase664_multi_competitor_continuation_split_audit/phase664_deepseek7b_multi_competitor_continuation_split_audit_confirm.json
```

### 测试原理

本阶段引入 multi-competitor margin：

```text
multi_margin
=
logit(correct_prefix)
-
max(logit(space), logit(newline), logit(word), logit(explanation))
```

如果：

```text
multi_margin > 0
```

说明 correct_prefix 不只是打败当前 top1 competitor，而是打败主要格式/词元竞争集合。

同时做 multi-competitor correction：

```text
对所有当前超过 correct_prefix 的 competitor，
沿 W_correct - W_competitor 的方向构造联合移动。
```

这可以检查：

```text
pairwise correction 后是否只是从 space 转移到 newline；
multi correction 是否能真正清空主要 competitor set。
```

对 continuation failure，本阶段强制接上 correct_prefix 后，再检查 token1 / token2 是否命中真实正确答案的后续 token。

### 主要结果

#### Qwen3

```text
selected_items = 32
rows = 128
```

实际状态：

```text
yes_no top2:
  exact_rate = 1.000
  correct_top1_rate = 1.000

yes_no top3:
  exact_rate = 0.906
  correct_top1_rate = 0.906

explanation top1:
  exact_rate = 0.719
  correct_top1_rate = 0.781

explanation top2:
  exact_rate = 0.875
  correct_top1_rate = 0.938
```

多竞争者失败：

```text
explanation word:
  n = 7
  mean_multi_margin = -0.857
  winner_sets = word:6, space+word:1

yes_no explanation:
  n = 2
  mean_multi_margin = -0.156

yes_no newline:
  n = 1
  mean_multi_margin = -0.188
```

multi-correction 后：

```text
explanation word:
  scale 1.5 correct_top1_rate = 1.000

yes_no explanation:
  scale 2.0 correct_top1_rate = 1.000

yes_no newline:
  scale 1.5 correct_top1_rate = 1.000
```

续写审计：

```text
explanation top2:
  n = 2
  token1_match_rate = 0.000
  token2_match_rate = 0.000

explanation top1:
  n = 2
  token1_match_rate = 0.000
  token2_match_rate = 0.000
```

Qwen3 的关键结论：

```text
first-token barrier 已经很弱；
但 correct_prefix 后的 continuation token 仍不跟随正确答案轨迹。
```

#### GLM4

```text
selected_items = 32
rows = 128
```

实际状态：

```text
explanation l22 top1:
  exact_rate = 0.781
  correct_top1_rate = 0.781

explanation late top1:
  exact_rate = 0.719
  correct_top1_rate = 0.781

yes_no l22 top2:
  exact_rate = 0.688
  correct_top1_rate = 0.719

yes_no late top2:
  exact_rate = 0.688
  correct_top1_rate = 0.719
```

多竞争者失败：

```text
explanation space:
  n = 10
  mean_multi_margin = -0.825
  winner_sets = space+word:6, space:4

yes_no word:
  n = 10
  mean_multi_margin = -0.263
  winner_sets = word:8, space+word:2

yes_no space:
  n = 8
  mean_multi_margin = -0.984
  winner_sets = space+word:6, space:2
```

multi-correction 后：

```text
explanation space:
  scale 1.5 correct_top1_rate = 1.000

yes_no space:
  scale 1.5 correct_top1_rate = 1.000

yes_no word:
  scale 2.0 correct_top1_rate = 1.000
```

续写审计：

```text
explanation late top1:
  n = 2
  token1_match_rate = 0.000
  token2_match_rate = 0.000

yes_no l22 top2:
  n = 1
  token1_match_rate = 1.000
  token2_match_rate = 0.000

yes_no late top2:
  n = 1
  token1_match_rate = 1.000
  token2_match_rate = 0.000
```

GLM4 的关键结论：

```text
readout competition 可以被 multi-direction correction 充分压低；
但部分样本仍会在后续 token 处偏离。
```

#### DS7B

```text
selected_items = 32
rows = 128
```

实际状态：

```text
explanation l22 top2:
  exact_rate = 0.500
  correct_top1_rate = 0.500

explanation late top2:
  exact_rate = 0.500
  correct_top1_rate = 0.500

yes_no l22 top1:
  exact_rate = 0.469
  correct_top1_rate = 0.469

yes_no late top1:
  exact_rate = 0.469
  correct_top1_rate = 0.469
```

多竞争者失败：

```text
explanation space:
  n = 32
  mean_multi_margin = -1.453
  winner_sets =
    space:14
    space+newline:10
    space+newline+word:8

yes_no space:
  n = 24
  mean_multi_margin = -1.750
  winner_sets =
    space+newline:16
    space+newline+word+explanation:4
    space:2
    space+newline+explanation:2

yes_no newline:
  n = 8
  mean_multi_margin = -2.859
  winner_sets =
    space+newline+explanation:4
    space+newline+word+explanation:4
```

这是关键结果：DS7B 的失败几乎不是单竞争者失败，而是多竞争者集合同时压制 correct_prefix。

multi-correction 后：

```text
explanation space:
  scale 1.0 correct_top1_rate = 0.562
  scale 1.5 correct_top1_rate = 0.938
  scale 2.0 correct_top1_rate = 1.000

yes_no space:
  scale 1.0 correct_top1_rate = 1.000

yes_no newline:
  scale 1.0 correct_top1_rate = 1.000
```

Phase 663 中 pairwise correction 后经常出现 space -> newline 的替换，本阶段证明这是因为竞争者集合没有被同时处理。multi-competitor correction 能显著减少这种替换。

### 客观进展

Phase 664 完成了两个关键补洞：

```text
1. Pairwise correction 不足以描述 DS7B 的真实瓶颈。
   DS7B 是 multi-competitor readout barrier。

2. correct_prefix top1 不等于 full answer correct。
   Qwen3 / GLM4 都出现 token1/token2 continuation failure。
```

这把当前拼图从：

```text
correct_prefix vs current top1
```

推进到：

```text
correct_prefix vs competitor set
```

并且从：

```text
first-token closure
```

推进到：

```text
first-token closure + continuation closure
```

### 理论进展

读出竞争公式更新为：

```text
M_{\text{multi}}
=
\ell_{\text{correct}}
-
\max_{j \in C_{\text{format/readout}}} \ell_j
```

其中：

```text
C_{\text{format/readout}}
=
\{space, newline, word, explanation, punctuation, symbol, other-prefix\}
```

如果只检查：

```text
\ell_{\text{correct}} - \ell_{\text{top1}}
```

可能会漏掉 top2/top3 中随时接管的竞争者。

续写闭合需要新增公式：

```text
P(\text{answer})
=
P(y_0=\text{correct-prefix})
\cdot
\prod_{t=1}^{T}
P(y_t=\text{correct-continuation}_t \mid y_{<t}, h_t)
```

因此：

```text
first-token closure
\neq
continuation closure
```

当前统一链条更新为：

```text
semantic value support
-> protocol / format writer suppression
-> late residual readout repair
-> projection norm / direction geometry
-> multi-competitor token readout
-> continuation trajectory controller
```

### 问题和硬伤

```text
1. multi-correction 仍然是 post-final_norm readout intervention。
   它不是模型自然内部路径。

2. multi-correction 只覆盖 top_k 内识别到的 target categories。
   没有穷尽整个 vocabulary 的所有潜在竞争者。

3. continuation audit 只检查 token1/token2 的局部延续。
   尚未定位 continuation controller 的具体层、组件、位置。

4. DS7B 的 multi-correction 读出闭合很强，但这不等于生成闭合。
   下一步必须把 readout correction 和真实 autoregressive generation 区分开。

5. Qwen3 / GLM4 的续写失败样本数量不大，但现象稳定：token1/token2 明显偏离。
```

### 阶段性判断

Phase 660-664 组成的当前子链条已经完成阶段性目标：

```text
从 final top1 barrier
到 late writer
到 projection geometry
到 multi-competitor readout
到 continuation split
```

已经可以确认：

```text
读出端不是单一门，而是多竞争者集合；
生成闭合不是第一词元闭合，而必须包含续写轨迹闭合。
```

### 下一阶段

下一步不应继续在 post-final_norm readout 上做更多人工方向修正。新的阶段应转向真实自回归路径：

```text
Phase 665: Autoregressive Continuation Controller Localization
```

目标：

```text
1. 只选择 correct_prefix top1 but exact wrong 的样本。
2. 定位 token1/token2 偏离发生的层、组件、位置。
3. 比较 prompt-end state、after-token0 state、token1 generation state。
4. 判断 continuation failure 是：
   语义值轨迹丢失；
   格式协议重新接管；
   还是 answer-token sequence 没被绑定。
```

这是新的阶段：从 readout-level counterfactual 进入真实 autoregressive continuation path，因此不应与 Phase 664 混在同一测试内。

## Phase 665: Autoregressive Continuation Controller Localization [2026-06-26 09:30]

### 任务背景

本阶段读取并审视了用户上传的 Phase 663-664 分析。总体判断：附件中的判断基本正确。

正确部分：

```text
1. Phase 663-664 确认读出端不是单一门，而是多竞争者集合。
2. Phase 664 证明 first-token closure 不等于 continuation closure。
3. 当前最需要推进的是 correct_prefix top1 but exact wrong 的真实自回归路径定位。
```

附件指出的硬伤也成立：

```text
Phase 664 的 continuation audit 只检查 token1/token2 现象，
尚未定位 continuation controller 的层、组件、位置。
```

因此本阶段进入：

```text
Phase 665: Autoregressive Continuation Controller Localization
```

### 生成脚本

```text
tests/gpt5/phase665_autoregressive_continuation_controller_localization.py
tests/gpt5/phase665_autoregressive_continuation_controller_localization_summary.py
```

### 执行命令

```bash
python -m py_compile tests/gpt5/phase665_autoregressive_continuation_controller_localization.py tests/gpt5/phase665_autoregressive_continuation_controller_localization_summary.py

python tests/gpt5/phase665_autoregressive_continuation_controller_localization.py qwen3 --smoke --hard-exit-after-model

python tests/gpt5/phase665_autoregressive_continuation_controller_localization.py qwen3 --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase665_autoregressive_continuation_controller_localization.py glm4 --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase665_autoregressive_continuation_controller_localization.py deepseek7b --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase665_autoregressive_continuation_controller_localization_summary.py
```

### 执行过程说明

第一次 confirm 运行时，qwen3 已完成计算，但 JSON 写盘失败：

```text
TypeError: Object of type Tensor is not JSON serializable
```

原因：

```text
selected_failures 中误保存了 source_patches，
其中包含 tensor。
```

处理：

```text
移除 selected_failures 中的 source_patches 字段；
重新编译、冒烟、并重新顺序执行 qwen3、GLM4、DS7B。
```

该问题是结果保存层面的脚本问题，不是模型测试结果问题。

### 输出文件

```text
results/glm5_phase665_autoregressive_continuation_controller_localization/
results/glm5_phase665_autoregressive_continuation_controller_localization/phase665_cross_model_summary.md
results/glm5_phase665_autoregressive_continuation_controller_localization/phase665_qwen3_autoregressive_continuation_controller_localization_confirm.json
results/glm5_phase665_autoregressive_continuation_controller_localization/phase665_glm4_autoregressive_continuation_controller_localization_confirm.json
results/glm5_phase665_autoregressive_continuation_controller_localization/phase665_deepseek7b_autoregressive_continuation_controller_localization_confirm.json
```

### 测试原理

本阶段只选择：

```text
correct_prefix top1
but exact generation wrong
```

然后强制模型进入真实自回归续写输入：

```text
prompt + correct_prefix
prompt + correct_prefix + correct_token1
```

分别检查：

```text
token1 是否能成为 top1；
token2 是否能成为 top1；
```

之后，从 short_value_allowed source path 采集 continuation position 的 hidden state，并 patch 到 task path 的同一续写位置，扫描：

```text
layer_input
attn_out
mlp_out
layer_out
```

目标是定位：

```text
续写失败是 token1 轨迹丢失；
还是 token2 轨迹丢失；
以及哪些层/组件携带了修复 token1/token2 的状态。
```

### 主要结果

#### Qwen3

```text
raw_cases = 512
selected_items = 64
continuation_failures = 5
rows = 650
scan_layers = L20-L35
```

筛到的续写失败：

```text
explanation top1:
  n = 3
  generation_text:
    " v22\n\nWait,"
    " v22\n\nOkay,"
    " 05\n\nThe answer"

explanation top2:
  n = 2
  generation_text:
    " v22\n\nWait,"
    " v22\n\nOkay,"
```

续写基线：

```text
explanation top1 step1:
  n = 3
  expected_top1_rate = 0.333
  mean_expected_rank = 1.67
  mean_expected_minus_top1 = -0.833

explanation top1 step2:
  expected_top1_rate = 1.000

explanation top2 step1:
  n = 2
  expected_top1_rate = 0.000
  mean_expected_rank = 2.00
  mean_expected_minus_top1 = -1.875

explanation top2 step2:
  expected_top1_rate = 1.000
```

关键现象：

```text
Qwen3 的 continuation failure 主要发生在 token1。
一旦 token1 被强制正确，token2 基本能跟上。
```

最强 patch 候选：

```text
explanation top2 step1:
  L22 attn_out:
    n = 2
    mean_margin_delta = 1.875
    mean_rank_improvement = 1.00
    flip_rate = 1.00

  L23 layer_out / L24 layer_input 以后大量状态也能修复。
```

保守解释：

```text
L22 attn_out 是较早出现的有效候选；
后续 layer_input/layer_out 的广泛有效，说明修复状态一旦写入后会沿 residual stream 传播，
不能把所有后续层都解释成独立因果源。
```

#### GLM4

```text
raw_cases = 512
selected_items = 64
continuation_failures = 4
rows = 292
scan_layers = L22-L39
```

筛到的续写失败：

```text
explanation late top1:
  n = 3
  generation_text:
    " v05\n\nReason: According"
    " 22\n\nReason: The"

explanation l22 top1:
  n = 1
  generation_text:
    " 22\n\nReason: The"
```

续写基线：

```text
explanation l22 top1 step1:
  n = 1
  expected_top1_rate = 1.000

explanation late top1 step1:
  n = 3
  expected_top1_rate = 0.333
  mean_expected_rank = 1.67
  mean_expected_minus_top1 = -0.375
```

最强 patch 候选：

```text
explanation late top1 step1:
  L22 layer_input / attn_out / mlp_out / layer_out:
    n = 3
    mean_margin_delta = 0.375
    mean_rank_improvement = 0.67
    flip_rate = 0.67

  L23-L39 的 layer_input / layer_out 也表现为传播性有效。
```

保守解释：

```text
GLM4 的 continuation token1 修复入口最早出现在 L22 附近；
但广泛 layer_out / layer_input 有效说明这更像状态轨迹修复带，
不是一个单点 writer 已闭合。
```

#### DS7B

```text
raw_cases = 512
selected_items = 64
continuation_failures = 12
rows = 696
scan_layers = L14-L27
```

筛到的续写失败：

```text
explanation l22 top2:
  n = 3
  generation_text:
    " 22\nBut why"
    " v05.\n\nBut wait"
    " v05 or v4"

explanation late top2:
  n = 3
  same pattern

yes_no l22 top1:
  n = 3
  generation_text:
    " 48.\n\nQuestion:"

yes_no late top1:
  n = 3
  generation_text:
    " 48.\n\nQuestion:"
```

续写基线：

```text
explanation l22 top2 step1:
  n = 3
  expected_top1_rate = 0.333
  mean_expected_rank = 2.00
  mean_expected_minus_top1 = -1.042

explanation late top2 step1:
  n = 3
  expected_top1_rate = 0.333
  mean_expected_rank = 1.67
  mean_expected_minus_top1 = -0.771

yes_no l22 top1 step1:
  n = 3
  expected_top1_rate = 0.667
  mean_expected_rank = 1.33
  mean_expected_minus_top1 = -0.417

yes_no late top1 step1:
  n = 3
  expected_top1_rate = 0.667
  mean_expected_rank = 1.33
  mean_expected_minus_top1 = -0.292

所有 step2:
  expected_top1_rate = 1.000
```

关键现象：

```text
DS7B 不只是第一词元/多竞争者读出问题；
当它越过 first-token 后，也存在 token1 continuation failure。
```

最强 patch 候选：

```text
explanation l22 top2 step1:
  L21 layer_out:
    n = 3
    mean_margin_delta = 1.042
    mean_rank_improvement = 1.00
    flip_rate = 0.67

  L22 layer_input / layer_out:
    n = 3
    mean_margin_delta = 1.042
    mean_rank_improvement = 1.00
    flip_rate = 0.67

  L23-L27 layer_input / layer_out:
    同样有效。
```

保守解释：

```text
DS7B 的 continuation repair 状态最早在 L21 layer_out / L22 layer_input 附近显现。
这很可能是 continuation state 已进入 residual stream 的边界，
而不是证明 L21/L22 的每个 layer_out 都是独立控制器。
```

### 客观进展

Phase 665 把 Phase 664 的续写现象进一步拆开：

```text
1. 三个模型的 continuation failure 主要集中在 token1。
2. 当 token1 被强制正确后，token2 基本都能 top1。
3. continuation controller 更像 token1 接续轨迹控制器，而不是完整多步序列全部缺失。
4. qwen3 的较早有效候选在 L22 attn_out 附近。
5. GLM4 的较早有效候选在 L22 附近。
6. DS7B 的较早有效候选在 L21 layer_out / L22 layer_input 附近。
```

这说明：

```text
correct_prefix 并没有自动绑定后续 answer-token sequence；
模型必须在 token1 位置重新进入正确答案轨迹。
```

### 理论进展

续写闭合公式需要进一步拆分：

```text
P(answer)
=
P(y_0 = correct_prefix)
\cdot
P(y_1 = correct_token1 | y_0, h_1)
\cdot
P(y_2 = correct_token2 | y_0,y_1,h_2)
\cdots
```

Phase 665 的结果显示：

```text
P(y_0 = correct_prefix) 高
不推出
P(y_1 = correct_token1 | y_0, h_1) 高
```

但在当前测试中：

```text
如果 y_1 被强制正确，
P(y_2 = correct_token2 | y_0,y_1,h_2) 通常很高。
```

所以当前最精确的生成瓶颈是：

```text
token0 -> token1 transition gate
```

而不是泛泛的 continuation failure。

统一链条更新为：

```text
semantic value support
-> first-token readout closure
-> token0-to-token1 transition gate
-> later continuation stabilization
```

### 问题和硬伤

```text
1. 当前 patch 是 value-source continuation-position restore。
   它证明 value condition 中存在能修 token1 的状态，
   但还没有证明模型自然路径中的哪个 writer 负责写入。

2. 候选层呈现宽带传播。
   后续 layer_input / layer_out 大量有效，说明状态一旦修复会沿 residual stream 传播。
   因此不能把每个后续层都当作独立控制器。

3. 样本数仍偏少：
   qwen3 = 5
   GLM4 = 4
   DS7B = 12
   结论足以说明现象存在，但不足以精确排序所有层组件。

4. step2 基本正确可能部分来自强制 token1 后语义轨迹恢复，
   不能说明自然生成的后续全序列都已闭合。

5. 本阶段只做 restore，没有做 remove / ablation necessary test。
```

### 阶段性判断

Phase 665 完成了真实自回归续写路径的第一轮定位：

```text
continuation failure 主要不是 token2 后段崩溃，
而是 token0 -> token1 transition gate 没有稳定打开。
```

最重要的新增拼图：

```text
correct_prefix 是答案轨道入口，
但 token1 transition 才是真正进入答案序列的闸门。
```

### 下一阶段

下一步应进入更窄的边界定位：

```text
Phase 666: Token0-to-Token1 Transition Boundary Audit
```

目标：

```text
1. 不再扫描宽层带。
2. 聚焦 Phase 665 中最早有效边界：
   qwen3: L22 attn_out / L23 boundary
   GLM4: L22 boundary
   DS7B: L21 layer_out / L22 layer_input
3. 做 remove + restore 双向测试。
4. 区分：
   source state restoration 是否只是补足答案值；
   还是确实打开 token1 transition gate。
5. 进一步确认 token0 -> token1 gate 是否是独立于 first-token readout 的机制。
```

这属于新的精细边界阶段，不应继续用 Phase 665 的宽扫描方式扩大测试。

## Phase 666: Token1 Transition Boundary Remove/Restore Audit [2026-06-26 09:43]

### 本阶段目标

Phase 665 证明 correct_prefix top1 并不等于完整答案闭合，真实续写失败主要集中在：

```text
token0 -> token1 transition
```

本阶段不再扩大扫描层数，而是只审计 Phase 665 找到的最早有效边界，测试这些边界到底携带：

```text
1. correct value transition state
2. general continuation / format state
3. residual stream propagation artifact
```

### 生成脚本

```text
tests/gpt5/phase666_token1_transition_boundary_remove_restore.py
tests/gpt5/phase666_token1_transition_boundary_remove_restore_summary.py
```

### 运行命令

```bash
python -m py_compile tests/gpt5/phase666_token1_transition_boundary_remove_restore.py tests/gpt5/phase666_token1_transition_boundary_remove_restore_summary.py

python tests/gpt5/phase666_token1_transition_boundary_remove_restore.py qwen3 --smoke --hard-exit-after-model

python tests/gpt5/phase666_token1_transition_boundary_remove_restore.py qwen3 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase666_token1_transition_boundary_remove_restore.py glm4 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase666_token1_transition_boundary_remove_restore.py deepseek7b --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase666_token1_transition_boundary_remove_restore_summary.py
```

### 输出文件

```text
results/glm5_phase666_token1_transition_boundary_remove_restore/phase666_qwen3_token1_transition_boundary_remove_restore_confirm.json
results/glm5_phase666_token1_transition_boundary_remove_restore/phase666_glm4_token1_transition_boundary_remove_restore_confirm.json
results/glm5_phase666_token1_transition_boundary_remove_restore/phase666_deepseek7b_token1_transition_boundary_remove_restore_confirm.json
results/glm5_phase666_token1_transition_boundary_remove_restore/phase666_cross_model_summary.md
```

### 测试原理

测试样本直接复用 Phase 665 的 continuation failure：

```text
correct_prefix 已经 top1
但是完整生成错误
```

强制输入：

```text
task_prompt + correct_prefix
```

目标只看下一步：

```text
token1
```

对 Phase 665 的最早边界做五种干预：

```text
baseline:
  原始任务路径。

self_restore:
  用 task 自身同位置状态恢复自身，作为 no-op control。

zero_remove:
  把边界状态置零，测试该状态是否必要。

mismatch_restore:
  用其他 failure 的 value source 状态恢复，作为错误语义源对照。

correct_restore:
  用匹配的 short_value_allowed value source 状态恢复。
```

判断标准：

```text
如果 correct_restore 明显优于 mismatch_restore，
说明该边界更可能携带 correct value transition state。

如果 correct_restore 和 mismatch_restore 同样有效，
说明该边界更可能携带 general continuation / format state。

如果 zero_remove 强烈破坏，
说明该边界状态对 token1 读出路径是必要的，
但不自动说明它具有语义特异性。
```

### 客观结果

#### qwen3

测试 failure 数：

```text
5
```

边界：

```text
L22 attn_out
L23 layer_input
```

关键结果：

```text
L22 attn_out:
  correct_restore 和 mismatch_restore 基本等效。
  top1 / top2 两组 correct_minus_mismatch = 0.000。

L23 layer_input:
  出现明显 correct specificity。
  top2:
    correct_delta = 0.875
    mismatch_delta = -2.688
    zero_delta = -4.062
    correct_minus_mismatch = 3.562

  top1:
    correct_delta = 0.208
    mismatch_delta = -2.042
    zero_delta = -5.146
    correct_minus_mismatch = 2.250
```

解释：

```text
qwen3 的 L22 attn_out 更像一般续写/格式状态；
到 L23 layer_input 时，正确值转移状态开始变得更特异。
```

#### GLM4

测试 failure 数：

```text
4
```

边界：

```text
L22 layer_input
L22 attn_out
L22 layer_out
```

关键结果：

```text
late_peak_layer_out top1:
  L22_attn_out:
    correct_delta = 0.375
    mismatch_delta = -0.125
    zero_delta = 0.375
    correct_minus_mismatch = 0.500

  L22_layer_input:
    correct_delta = 0.375
    mismatch_delta = -0.042
    zero_delta = -3.354
    correct_minus_mismatch = 0.417

  L22_layer_out:
    correct_delta = 0.375
    mismatch_delta = 0.375
    zero_delta = -11.047
    correct_minus_mismatch = 0.000
```

解释：

```text
GLM4 的 L22 attention / layer_input 有弱 correct specificity；
L22 layer_out 已经更像共享续写轨迹，correct 和 mismatch 等效；
zero_remove 对 layer_out 破坏极强，说明该状态必要但语义特异性弱。
```

#### DS7B

测试 failure 数：

```text
12
```

边界：

```text
L21 layer_out
L22 layer_input
L22 layer_out
```

关键结果：

```text
explanation_required l22_peak_layer_out top2:
  L21_layer_out:
    correct_delta = 1.042
    mismatch_delta = -2.271
    zero_delta = -7.688
    correct_minus_mismatch = 3.312

  L22_layer_input:
    correct_delta = 1.042
    mismatch_delta = -2.271
    zero_delta = -7.688
    correct_minus_mismatch = 3.312

  L22_layer_out:
    correct_delta = 1.042
    mismatch_delta = -2.042
    zero_delta = -5.771
    correct_minus_mismatch = 3.083

explanation_required late_peak_layer_out top2:
  L21_layer_out:
    correct_delta = 0.771
    mismatch_delta = -2.479
    zero_delta = -7.833
    correct_minus_mismatch = 3.250

  L22_layer_input:
    correct_delta = 0.771
    mismatch_delta = -2.479
    zero_delta = -7.833
    correct_minus_mismatch = 3.250

  L22_layer_out:
    correct_delta = 0.771
    mismatch_delta = -2.312
    zero_delta = -6.042
    correct_minus_mismatch = 3.083
```

解释：

```text
DS7B 的 L21/L22 边界有强 correct specificity；
correct_restore 稳定提升 token1；
mismatch_restore 明显破坏；
zero_remove 强烈破坏。
```

这是本阶段最清晰的正结果。

### 阶段性进展

Phase 665 的结论是：

```text
续写失败主要集中在 token0 -> token1 transition。
```

Phase 666 进一步把这个结论拆成两层：

```text
general continuation / format state
correct value transition state
```

跨模型结果显示：

```text
qwen3:
  L22 attn_out 偏 general continuation / format state；
  L23 layer_input 开始出现 correct value transition specificity。

GLM4:
  L22 attention / layer_input 只有弱 correct specificity；
  L22 layer_out 更像共享续写轨迹。

DS7B:
  L21 layer_out / L22 layer_input / L22 layer_out 都表现出强 correct value transition specificity。
```

因此，当前不能再把 token1 transition gate 当成单一东西。更精确的结构是：

```text
token1 transition gate
=
format-continuation enabling state
+
value-specific transition state
```

### 对附件判断的评估

附件对 Phase 665 的判断基本正确：

```text
correct_prefix 是答案轨道入口；
token1 transition 才是真正进入答案序列的闸门。
```

但 Phase 666 说明需要进一步收紧：

```text
token1 transition gate 不是一个单纯语义门；
其中至少包含格式续写状态和正确值转移状态两部分。
```

### 当前硬伤

```text
1. qwen3 和 GLM4 的样本量仍偏小。
   qwen3 = 5
   GLM4 = 4
   DS7B = 12

2. mismatch_restore 采用其他 failure 的 value prompt 状态。
   它是错误语义源对照，但不保证只改变语义、不改变格式。

3. zero_remove 是强破坏操作。
   它能证明边界状态必要，但可能同时破坏格式、语义、位置和残差尺度。

4. GLM4 中 baseline 有一组已经 token1 top1。
   这些样本对 restore 的增益解释能力有限。

5. 当前仍然没有定位 natural writer。
   本阶段证明边界状态携带信息，但还没有证明哪个 attention head / MLP neuron 自然写入该状态。
```

### 理论更新

原链条：

```text
semantic value support
-> first-token readout closure
-> token0-to-token1 transition gate
-> later continuation stabilization
```

更新为：

```text
semantic value support
-> first-token readout closure
-> format-continuation enabling state
-> value-specific token1 transition state
-> later continuation stabilization
```

更保守的数学表达：

```text
P(y_1 | y_0, h)
由至少两个状态共同决定：
1. format continuation state
2. value-specific transition state
```

写成块级公式：

$$
P(y_1 = v_1 \mid y_0, h)
=
F_{\text{readout}}
\left(
h_{\text{format-cont}},
h_{\text{value-trans}},
h_{\text{other}}
\right)
$$

其中：

$$
h_{\text{format-cont}}
\neq
h_{\text{value-trans}}
$$

Phase 666 的核心证据是：

$$
\Delta_{\text{correct}}
-
\Delta_{\text{mismatch}}
> 0
$$

在 DS7B 上该差值稳定较大，在 qwen3 的 L23 layer_input 上也较明显，在 GLM4 上较弱。

### 下一阶段

下一步仍属于同一个阶段性目标：

```text
破解 token0 -> token1 transition gate。
```

但 Phase 667 不应继续只做边界 patch，而应定位 writer：

```text
Phase 667: Value-Specific Token1 Transition Writer Localization
```

目标：

```text
1. 只聚焦 Phase 666 证据最强的边界：
   qwen3:
     L23 layer_input
   GLM4:
     L22 attn_out / L22 layer_input
   DS7B:
     L21 layer_out / L22 layer_input

2. 对边界前一层的 attention heads 和 MLP 输出做 writer 分解。

3. 不只测试 correct_restore，
   还必须加入 mismatch_restore 和 zero_remove。

4. 如果某个 writer 的 correct_restore 有效、mismatch_restore 无效、zero_remove 破坏，
   才能认为它是 value-specific token1 transition writer 候选。

5. 如果所有 writer 都不满足三条件，
   则说明 transition state 是分布式合成状态，不应继续寻找单一 writer。
```

阶段性目标尚未完成，应继续自动推进 Phase 667。

## Phase 667: Value-Specific Token1 Transition Writer Localization [2026-06-26 09:49]

### 本阶段目标

Phase 666 证明 token1 transition gate 至少包含两类状态：

```text
format-continuation enabling state
value-specific token1 transition state
```

本阶段继续同一阶段性目标，追问：

```text
value-specific token1 transition state 是由单个 writer 写入，
还是由多个 writer 合成后在 residual stream 中形成？
```

### 生成脚本

```text
tests/gpt5/phase667_value_specific_token1_transition_writer_localization.py
tests/gpt5/phase667_value_specific_token1_transition_writer_localization_summary.py
```

### 运行命令

```bash
python -m py_compile tests/gpt5/phase667_value_specific_token1_transition_writer_localization.py tests/gpt5/phase667_value_specific_token1_transition_writer_localization_summary.py

python tests/gpt5/phase667_value_specific_token1_transition_writer_localization.py qwen3 --smoke --hard-exit-after-model

python tests/gpt5/phase667_value_specific_token1_transition_writer_localization.py qwen3 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase667_value_specific_token1_transition_writer_localization.py glm4 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase667_value_specific_token1_transition_writer_localization.py deepseek7b --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase667_value_specific_token1_transition_writer_localization_summary.py
```

### 输出文件

```text
results/glm5_phase667_value_specific_token1_transition_writer_localization/phase667_qwen3_value_specific_token1_transition_writer_localization_confirm.json
results/glm5_phase667_value_specific_token1_transition_writer_localization/phase667_glm4_value_specific_token1_transition_writer_localization_confirm.json
results/glm5_phase667_value_specific_token1_transition_writer_localization/phase667_deepseek7b_value_specific_token1_transition_writer_localization_confirm.json
results/glm5_phase667_value_specific_token1_transition_writer_localization/phase667_cross_model_summary.md
```

### 测试原理

样本仍然复用 Phase 665 的 continuation failure。

输入：

```text
task_prompt + correct_prefix
```

目标：

```text
token1
```

候选 writer 分两类：

```text
1. component writer:
   attn_out
   mlp_out
   layer_out
   layer_input

2. attention head writer:
   attention o_proj input 的单头 slice
```

干预仍然使用三类核心对照：

```text
zero_remove
mismatch_restore
correct_restore
```

判断标准：

```text
如果单头 correct_restore 明显优于 mismatch_restore，
说明可能存在单头级 writer。

如果整层 component 明显有效，但单头都弱，
说明更可能是多头/MLP/残差合成状态。

如果 correct_restore 和 mismatch_restore 等效，
说明该 writer 更可能写入 general continuation / format state。
```

### 客观结果

#### qwen3

测试规模：

```text
failures_tested = 5
rows = 540
```

最强 component 结果：

```text
top2:
  L22_layer_out:
    correct_top1 = 0.500
    mismatch_top1 = 0.000
    correct_minus_mismatch = 3.562
    correct_minus_zero = 4.938

  L23_layer_input:
    correct_top1 = 0.500
    mismatch_top1 = 0.000
    correct_minus_mismatch = 3.562
    correct_minus_zero = 4.938

top1:
  L22_layer_out:
    correct_top1 = 0.667
    mismatch_top1 = 0.333
    correct_minus_mismatch = 2.250
    correct_minus_zero = 5.354

  L23_layer_input:
    correct_top1 = 0.667
    mismatch_top1 = 0.333
    correct_minus_mismatch = 2.250
    correct_minus_zero = 5.354
```

最强 head 结果：

```text
L22_head11_o_input:
  top2:
    correct_top1 = 0.500
    mismatch_top1 = 0.000
    correct_minus_mismatch = 1.938

  top1:
    correct_top1 = 0.667
    mismatch_top1 = 0.333
    correct_minus_mismatch = 1.417

L22_head10_o_input:
  top2:
    correct_top1 = 0.500
    mismatch_top1 = 0.000
    correct_minus_mismatch = 1.750

  top1:
    correct_top1 = 0.667
    mismatch_top1 = 0.333
    correct_minus_mismatch = 0.958
```

解释：

```text
qwen3 存在少数 head 级贡献，尤其是 L22 head10 / head11；
但整层 L22_layer_out / L23_layer_input 明显更强。
因此更像多头混合后形成 value-specific transition state，
而不是单头独立闭合。
```

#### GLM4

测试规模：

```text
failures_tested = 4
rows = 432
```

最强结果：

```text
L22_attn_out:
  correct_top1 = 1.000
  mismatch_top1 = 0.333
  correct_minus_mismatch = 0.500

L22_head7_o_input:
  correct_top1 = 1.000
  mismatch_top1 = 0.667
  correct_minus_mismatch = 0.438

L21_layer_out:
  correct_top1 = 1.000
  mismatch_top1 = 0.667
  correct_minus_mismatch = 0.417

L22_mlp_out:
  correct_top1 = 1.000
  mismatch_top1 = 0.667
  correct_minus_mismatch = 0.396

L22_head13_o_input:
  correct_top1 = 1.000
  mismatch_top1 = 0.667
  correct_minus_mismatch = 0.354
```

解释：

```text
GLM4 有弱 writer 候选，但特异性不强。
许多 mismatch_restore 也能部分修复。
这说明 GLM4 的 token1 transition writer 可能更偏共享续写状态，
或样本量不足以稳定区分 value-specific writer。
```

#### DS7B

测试规模：

```text
failures_tested = 12
rows = 576
```

最强 component 结果：

```text
l22_peak_layer_out top2:
  L21_layer_out:
    correct_top1 = 1.000
    mismatch_top1 = 0.000
    correct_minus_mismatch = 3.312
    correct_minus_zero = 8.729

  L22_layer_input:
    correct_top1 = 1.000
    mismatch_top1 = 0.000
    correct_minus_mismatch = 3.312
    correct_minus_zero = 8.729

late_peak_layer_out top2:
  L21_layer_out:
    correct_top1 = 1.000
    mismatch_top1 = 0.000
    correct_minus_mismatch = 3.250
    correct_minus_zero = 8.604

  L22_layer_input:
    correct_top1 = 1.000
    mismatch_top1 = 0.000
    correct_minus_mismatch = 3.250
    correct_minus_zero = 8.604
```

最强 head / MLP 结果：

```text
L21_head14_o_input:
  correct_minus_mismatch = 0.188 / 0.146
  correct_top1 = 0.333
  mismatch_top1 = 0.333

L21_mlp_out:
  correct_minus_mismatch = 0.167 / 0.125
  correct_top1 = 0.333
  mismatch_top1 = 0.333
```

解释：

```text
DS7B 的 value-specific token1 transition state 在 L21_layer_out / L22_layer_input 上非常强；
但单个 attention head 和 MLP 输出都不能单独解释该状态。
这强烈支持分布式合成后进入 residual stream 的解释。
```

### 阶段性进展

Phase 667 没有把机制闭合到单个 head，但完成了一个重要排除：

```text
value-specific token1 transition state 不是简单的 single attention head writer。
```

更准确的拼图是：

```text
qwen3:
  少数 head 有贡献，但整层状态更强。

GLM4:
  弱 head / component 候选，但特异性不足。

DS7B:
  强状态在 layer_out / layer_input 边界；
  单头和单 MLP 输出远弱于整层。
```

因此，当前最保守结论：

```text
token1 transition writer 更可能是 distributed writer ensemble，
而不是单点 writer。
```

### 当前硬伤

```text
1. head slice 测的是 o_proj input 单头通道，不等价于完整 attention head 因果路径。
   它没有拆分 Q/K/V 和 attention pattern。

2. component_restore 仍是状态替换，不是自然写入过程追踪。

3. DS7B 的强结论来自 explanation_required top2 样本，
   还需要在更多任务类型中验证。

4. qwen3 的 head10/head11 是候选，不是闭合机制。
   它们只能解释部分增益。

5. GLM4 样本数偏少，且部分 baseline 已经较强，
   对 writer 排序不稳定。
```

### 理论更新

Phase 666 的公式：

$$
P(y_1 = v_1 \mid y_0, h)
=
F_{\text{readout}}
\left(
h_{\text{format-cont}},
h_{\text{value-trans}},
h_{\text{other}}
\right)
$$

Phase 667 进一步说明：

$$
h_{\text{value-trans}}
\neq
h_{\text{single-head}}
$$

更合理的表达是：

$$
h_{\text{value-trans}}
=
\sum_i W_i(x)
$$

其中：

$$
W_i(x)
\in
\{\text{attention head outputs},\ \text{MLP outputs},\ \text{residual mixing}\}
$$

并且当前证据显示：

$$
\left\|
\sum_i W_i(x)
\right\|
\text{ 的因果效果}
>
\max_i
\left\|
W_i(x)
\right\|
\text{ 的因果效果}
$$

通俗说：

```text
真正有效的 token1 转移状态，像是多个 writer 合成后的残差态，
不是某一个头单独写出来的简单开关。
```

### 下一阶段

同一阶段性目标还没有结束。下一步不应继续盲扫更多 head，而应做 ensemble 级验证：

```text
Phase 668: Token1 Transition Writer Ensemble Closure
```

目标：

```text
1. 以 Phase 667 的强边界为目标：
   qwen3:
     L22_layer_out / L23_layer_input
     L22 head10/head11 作为候选子集

   GLM4:
     L22_attn_out
     L22 head7/head13 作为弱候选子集

   DS7B:
     L21_layer_out / L22_layer_input
     不优先追单头，优先测 attn_out + mlp_out + layer_out 组合

2. 测试 writer ensemble 是否能恢复整层边界效果。

3. 加入三类组合：
   top-head ensemble
   component ensemble
   residual boundary full state

4. 如果 ensemble 仍不能接近 full boundary restore，
   则说明 value-specific transition state 可能还依赖 layernorm / residual scale / downstream readout。

5. 如果 ensemble 接近 full boundary restore，
   则可以把 token1 transition gate 从边界状态推进到 writer graph。
```

阶段性目标仍未完成，应继续自动推进 Phase 668。

## Phase 668: Token1 Transition Writer Ensemble Closure [2026-06-26 09:54]

### 本阶段目标

Phase 667 表明：

```text
value-specific token1 transition state
不是简单 single-head writer。
```

但 Phase 667 仍留下一个关键问题：

```text
少数 writer ensemble 能否接近 full boundary restore？
```

本阶段测试：

```text
full boundary state
top-head ensemble
component ensemble
```

是否能闭合 token1 transition。

### 生成脚本

```text
tests/gpt5/phase668_token1_transition_writer_ensemble_closure.py
tests/gpt5/phase668_token1_transition_writer_ensemble_closure_summary.py
```

### 运行命令

```bash
python -m py_compile tests/gpt5/phase668_token1_transition_writer_ensemble_closure.py tests/gpt5/phase668_token1_transition_writer_ensemble_closure_summary.py

python tests/gpt5/phase668_token1_transition_writer_ensemble_closure.py qwen3 --smoke --hard-exit-after-model

python tests/gpt5/phase668_token1_transition_writer_ensemble_closure.py qwen3 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase668_token1_transition_writer_ensemble_closure.py glm4 --confirm --save-rows --hard-exit-after-model
python tests/gpt5/phase668_token1_transition_writer_ensemble_closure.py deepseek7b --confirm --save-rows --hard-exit-after-model

python tests/gpt5/phase668_token1_transition_writer_ensemble_closure_summary.py
```

### 输出文件

```text
results/glm5_phase668_token1_transition_writer_ensemble_closure/phase668_qwen3_token1_transition_writer_ensemble_closure_confirm.json
results/glm5_phase668_token1_transition_writer_ensemble_closure/phase668_glm4_token1_transition_writer_ensemble_closure_confirm.json
results/glm5_phase668_token1_transition_writer_ensemble_closure/phase668_deepseek7b_token1_transition_writer_ensemble_closure_confirm.json
results/glm5_phase668_token1_transition_writer_ensemble_closure/phase668_cross_model_summary.md
```

### 测试原理

继续使用 Phase 665 的 continuation failure。

输入：

```text
task_prompt + correct_prefix
```

目标：

```text
token1
```

干预：

```text
zero_remove
mismatch_restore
correct_restore
```

比较：

```text
full boundary:
  完整边界状态恢复。

top-head ensemble:
  Phase 667 中最强 head 组合恢复。

component ensemble:
  attn_out + mlp_out 或多个组件组合恢复。
```

核心判据：

```text
如果 ensemble 的 correct_minus_mismatch 接近 full boundary，
说明 writer graph 已经接近闭合。

如果 ensemble 明显弱于 full boundary，
说明有效状态可能还依赖 residual mixing / layernorm / downstream scale。
```

### 客观结果

#### qwen3

测试规模：

```text
failures_tested = 5
rows = 60
```

关键结果：

```text
L22_heads10_11:
  top2:
    correct_top1 = 0.500
    mismatch_top1 = 0.000
    correct_minus_mismatch = 4.688
    correct_minus_zero = 1.375

  top1:
    correct_top1 = 0.667
    mismatch_top1 = 0.333
    correct_minus_mismatch = 3.000
    correct_minus_zero = 0.792

full_L22_layer_out:
  top2:
    correct_top1 = 0.500
    mismatch_top1 = 0.000
    correct_minus_mismatch = 3.562
    correct_minus_zero = 4.938

  top1:
    correct_top1 = 0.667
    mismatch_top1 = 0.333
    correct_minus_mismatch = 2.250
    correct_minus_zero = 5.354

full_L23_layer_input:
  与 full_L22_layer_out 基本相同。

L22_attn_mlp:
  top2:
    correct_top1 = 0.000
    mismatch_top1 = 0.000
    correct_minus_mismatch = 1.000

  top1:
    correct_top1 = 0.333
    mismatch_top1 = 0.333
    correct_minus_mismatch = 0.750
```

解释：

```text
qwen3 出现了比较清晰的小 head ensemble 闭合信号。
L22 head10 + head11 的 correct_minus_mismatch 甚至超过 full boundary。
但 correct_minus_zero 不如 full boundary，说明它更像语义特异子通道，
不是完整必要状态。
```

#### GLM4

测试规模：

```text
failures_tested = 4
rows = 48
```

关键结果：

```text
L22_heads7_13:
  correct_top1 = 1.000
  mismatch_top1 = 0.333
  correct_minus_mismatch = 0.604
  correct_minus_zero = 0.292

full_L22_attn_out:
  correct_top1 = 1.000
  mismatch_top1 = 0.333
  correct_minus_mismatch = 0.500

full_L22_layer_input:
  correct_top1 = 1.000
  mismatch_top1 = 0.667
  correct_minus_mismatch = 0.417

L21_layer_out_L22_attn_mlp:
  correct_top1 = 1.000
  mismatch_top1 = 1.000
  correct_minus_mismatch = 0.000
```

解释：

```text
GLM4 的 head7 + head13 组合略强于 full_L22_attn_out，
但整体 correct_minus_mismatch 很小。
这不是强机制闭合，只能视为弱候选。
```

#### DS7B

测试规模：

```text
failures_tested = 12
rows = 72
```

关键结果：

```text
full_L21_layer_out:
  l22_peak_layer_out top2:
    correct_top1 = 1.000
    mismatch_top1 = 0.000
    correct_minus_mismatch = 3.312
    correct_minus_zero = 8.729

  late_peak_layer_out top2:
    correct_top1 = 1.000
    mismatch_top1 = 0.000
    correct_minus_mismatch = 3.250
    correct_minus_zero = 8.604

full_L22_layer_input:
  与 full_L21_layer_out 基本相同。

L21_attn_mlp:
  l22_peak_layer_out top2:
    correct_top1 = 0.333
    mismatch_top1 = 0.000
    correct_minus_mismatch = 0.542

  late_peak_layer_out top2:
    correct_top1 = 0.333
    mismatch_top1 = 0.000
    correct_minus_mismatch = 0.479

L21_heads14_17:
  correct_top1 = 0.333
  mismatch_top1 = 0.333
  correct_minus_mismatch = 0.292
```

解释：

```text
DS7B 的 full boundary state 很强；
但 head ensemble 和 attn+mlp component ensemble 都远不能接近 full boundary。
因此 DS7B 的 token1 transition state 更像 residual-boundary integrated state，
而不是小 head ensemble。
```

### 阶段性进展

Phase 668 回答了 Phase 667 的核心问题：

```text
小 writer ensemble 是否能闭合 token1 transition？
```

答案是跨模型不一致：

```text
qwen3:
  L22 head10 + head11 能形成较强语义特异子通道，
  接近甚至超过 full boundary 的 correct-minus-mismatch。

GLM4:
  L22 head7 + head13 有弱闭合，
  但强度不足，仍不稳定。

DS7B:
  小 head ensemble 和 attn+mlp ensemble 都不能闭合；
  full layer_out / layer_input 边界远强于小组合。
```

这说明：

```text
token1 transition gate 的实现方式具有模型差异。
```

不能再假设所有模型共享同一种 writer topology。

### 理论更新

Phase 667 的表达：

$$
h_{\text{value-trans}}
=
\sum_i W_i(x)
$$

Phase 668 进一步把它拆成：

$$
h_{\text{value-trans}}
=
G_{\text{model}}
\left(
\{W_i(x)\},
R(x),
N(x)
\right)
$$

其中：

$$
R(x)
=
\text{residual mixing}
$$

$$
N(x)
=
\text{normalization / scale state}
$$

跨模型差异表现为：

$$
G_{\text{qwen3}}
\approx
\text{small head ensemble} + R
$$

$$
G_{\text{glm4}}
\approx
\text{weak head/component ensemble} + R
$$

$$
G_{\text{ds7b}}
\approx
\text{residual-boundary integrated state}
$$

通俗说：

```text
qwen3 的 token1 转移门可以下钻到少数 head 子通道；
GLM4 有弱子通道；
DS7B 主要要看边界整体状态，不能继续追单头。
```

### 当前硬伤

```text
1. qwen3 的 head10+head11 虽然 correct-minus-mismatch 强，
   但 correct-minus-zero 不如 full boundary。
   说明它不是完整必要状态。

2. GLM4 的正结果很弱，且样本量只有 4。

3. DS7B 的 full boundary 很强，但 ensemble 不闭合。
   这说明当前 writer 分解粒度仍不够。

4. 本阶段仍没有拆 layernorm / residual scale。

5. mismatch_restore 仍可能混入格式差异，不是纯语义差异。
```

### 阶段性判断

Phase 665-668 这条阶段链已经完成了一个阶段性目标：

```text
把 full-answer continuation failure 拆解为 token0 -> token1 transition gate，
并进一步区分 format-continuation state 与 value-specific transition state，
最后确认 writer 结构不是统一单头机制，而是跨模型差异化的 ensemble / residual-boundary 结构。
```

这已经足够作为阶段性收束。

### 下一阶段

下一阶段不应继续在同一个 token1 gate 内无限追小组件，而应进入更全局的图谱问题：

```text
Phase 669: Cross-Mechanism Language Encoding Graph Atlas
```

目标：

```text
1. 把已有拼图节点统一成 graph atlas：
   semantic value support
   first-token readout closure
   format-continuation state
   value-specific token1 transition state
   residual-boundary integrated state
   final readout competition

2. 不再只追单点机制。

3. 为每个节点记录：
   position
   layer
   component
   sufficiency
   necessity
   cross-model stability
   failure mode

4. 明确哪些机制是跨模型稳定结构，
   哪些机制是模型特异实现。

5. 为后续语言三大系统建立图谱入口：
   knowledge network
   reasoning route
   grammar / format protocol
```

Phase 669 属于新的全局图谱阶段，不再是当前 token1 transition gate 的局部阶段。

## Phase 669: Cross-Mechanism Language Encoding Graph Atlas [2026-06-26 10:04]

### 本阶段目标

分析用户上传的 Phase 666-668 阶段总结是否正确，并综合当前 Phase 626-668 的客观结果，把研究从局部 token1 transition gate 追踪收束到跨机制语言编码图谱。

本阶段没有进行新的模型推理测试。原因是 Phase 665-668 已经完成当前局部阶段目标，继续盲目追更小组件容易扩大噪声；当前更需要把已有拼图组织成可审计 graph atlas，再为下一阶段设计更干净的反事实控制。

### 生成脚本

```bash
tests/gpt5/phase669_cross_mechanism_language_encoding_graph_atlas.py
```

脚本输出：

```bash
results/glm5_phase669_cross_mechanism_language_encoding_graph_atlas/phase669_graph_atlas.json
results/glm5_phase669_cross_mechanism_language_encoding_graph_atlas/phase669_graph_atlas.md
```

### 执行命令

```bash
python -m py_compile tests/gpt5/phase669_cross_mechanism_language_encoding_graph_atlas.py
python tests/gpt5/phase669_cross_mechanism_language_encoding_graph_atlas.py
```

核对命令：

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('results/glm5_phase669_cross_mechanism_language_encoding_graph_atlas/phase669_graph_atlas.json')
d=json.loads(p.read_text(encoding='utf-8'))
ke=d['key_evidence']
print('nodes', len(d['nodes']))
print('edges', len(d['edges']))
print('available_phase_count', d['available_phase_count'])
print('phase666_models', {k: len(v) for k,v in ke['phase666_token1_boundary_specificity'].items()})
print('phase667_models', {k: len(v) for k,v in ke['phase667_writer_specificity'].items()})
print('phase668_models', {k: len(v) for k,v in ke['phase668_writer_ensemble_specificity'].items()})
print('next_phase', d['next_phase']['phase'])
PY
```

核对结果：

```text
nodes 10
edges 9
available_phase_count 40
phase666_models {'qwen3': 4, 'glm4': 6, 'deepseek7b': 6}
phase667_models {'qwen3': 8, 'glm4': 8, 'deepseek7b': 8}
phase668_models {'qwen3': 8, 'glm4': 8, 'deepseek7b': 8}
next_phase 670
```

### 对上传分析的判断

上传分析的主体判断基本正确：

```text
Phase 666:
token1 transition gate 不是单一语义修复，而至少包含 format-continuation state
和 value-specific token1 transition state。

Phase 667-668:
value-specific token1 transition state 不是统一 single-head writer。
qwen3 存在较强 head ensemble 子通道；
GLM4 有弱子通道；
DS7B 更像 residual-boundary integrated state。
```

需要收紧的地方：

```text
1. 上传分析中有少量公式排版错误，不影响核心判断，但不能作为正式数学表达。

2. “完成度百分比”只能作为主观机制闭合度，不是客观真理进度。

3. qwen3 的 L22 head10+11 不能外推为通用机制。

4. DS7B 的强结果主要在 full boundary，不支持继续用小 head ensemble 解释。

5. mismatch_restore 仍然不是纯语义对照，可能混入格式、位置、尺度状态。
```

### Phase 669 图谱节点

本阶段把已有拼图整理为 10 个机制节点：

```text
1. semantic_value_support
2. task_intent_gate
3. protocol_execution_field
4. first_token_readout_closure
5. multi_competitor_readout
6. format_continuation_state
7. value_specific_token1_transition_state
8. writer_topology
9. residual_boundary_integrated_state
10. continuation_controller
```

对应跨机制边：

```text
semantic_value_support
  -> task_intent_gate
  -> protocol_execution_field
  -> first_token_readout_closure
  -> multi_competitor_readout
  -> format_continuation_state
  -> value_specific_token1_transition_state
  -> writer_topology / residual_boundary_integrated_state
  -> continuation_controller
```

### 客观进展

1. 当前 token1 transition gate 局部阶段可以阶段性停止。

```text
Phase 665-668 已经证明：

correct_prefix top1 不等于 full answer generation；
失败主要前移到 token0 -> token1 transition；
token1 transition 同时包含一般格式续写状态和具体值转移状态；
writer 实现跨模型不同。
```

2. 当前机制图谱已经出现稳定的功能层级。

```text
semantic support:
给出候选值支持，但不保证生成。

task intent:
决定是否允许短值答案路线。

protocol field:
决定短答、解释、换行、格式路线。

readout competition:
决定正确前缀能否击败空格、换行、解释词等竞争项。

continuation gate:
决定正确前缀之后能否继续生成正确值序列。
```

3. 实现拓扑不是跨模型统一的。

```text
qwen3:
L22 head10+11 具有较强 value-specific 子通道。

GLM4:
有弱 head/component 信号，但边际较小。

DS7B:
full residual boundary 明显强于小组件 ensemble。
```

### 当前硬伤

```text
1. 图谱是由已有实验结果整理而来，不是新的因果测试。

2. 大量节点仍基于 ORV short-value 任务，需要扩展到 JSON、代码、自然解释、长答案。

3. mismatch_restore 和 zero_remove 仍然不是干净的纯语义 / 纯格式对照。

4. writer topology 尚未拆开 Q/K/V、attention pattern、layernorm、residual scale。

5. continuation_controller 只在 token1/token2 附近得到初步证据，尚未覆盖更长序列。
```

### 理论进展

当前最稳妥的语言编码机制表达应从单点机制改为图谱表达：

$$
\text{language output}
=
F(
S_{\text{semantic}},
G_{\text{intent}},
P_{\text{protocol}},
R_{\text{readout}},
C_{\text{continuation}}
)
$$

其中：

$$
S_{\text{semantic}}
\rightarrow
G_{\text{intent}}
\rightarrow
P_{\text{protocol}}
\rightarrow
R_{\text{readout}}
\rightarrow
C_{\text{continuation}}
$$

更贴近当前实证结果的展开式是：

$$
S_{\text{value}}
\rightarrow
G_{\text{task}}
\rightarrow
P_{\text{format}}
\rightarrow
R_{\text{multi-competitor}}
\rightarrow
C_{\text{format}}
\rightarrow
C_{\text{value-token}}
\rightarrow
W_{\text{writer/topology}}
$$

这说明语言能力不是一个单独“语义方向”产生的，而是语义、任务意图、格式协议、读出竞争、续写门控共同组成的动态图谱。

### 下一阶段

Phase 670 应进入：

```text
Graph Atlas Counterfactual Control Set
```

核心任务不是马上跑更大模型测试，而是先构造更干净的反事实控制集：

```text
1. same-value / different-format
   同一个值，不同格式路线。

2. different-value / same-format
   不同值，同一格式路线。

3. same-prefix / different-continuation
   相同 answer-entry token，不同后续值 token。

4. same-format / random-value
   同一格式协议，随机值内容。

5. protocol-only / value-only / intent-only controls
   分离格式协议、语义值、任务意图。
```

阶段目标：

```text
不再只问某个 patch 是否有效，
而是问每个 graph node 对哪些控制变量敏感，
从而建立真正的 language encoding atlas。
```

### 阶段性结论

Phase 666-668 的上传分析方向正确；Phase 669 完成了从局部 token1 gate 到跨机制图谱的阶段性收束。当前不应继续在同一局部机制里追单头，而应进入以 graph atlas 为中心的反事实控制测试。

## Phase 670: Graph Atlas Counterfactual Control Set [2026-06-26 10:22]

### 本阶段目标

分析用户给出的三份理论文档和 Phase 669 评价，并继续完成同一阶段任务：把 Phase 669 的 graph atlas 转化为可执行的反事实控制集。

本阶段不做模型推理。原因是当前主要瓶颈不是缺少新的 patch，而是缺少干净控制变量：

```text
same-value / different-format
different-value / same-format
same-prefix / different-continuation
same-format / random-value
value-only / intent-only / protocol-only
```

### 输入分析文件

```text
research/MainAnalysis/20260626_02_和主流研究的比较.md
research/MainAnalysis/20260626_03_当前理论的缺陷.md
research/MainAnalysis/20260626_04_接下来的计划.md
```

三份文件的主体判断基本正确：

```text
1. 当前路线和主流 mechanistic interpretability 的差异，
   是从 feature / circuit first 转向 field / gate / trajectory first。

2. 当前理论的最大风险不是局部结果错误，
   而是 ORV 微世界过拟合、patch 伪机制、自然轨迹预测不足。

3. 下一步不应继续盲目追局部 patch，
   而应进入理论验证、图谱控制集、自然轨迹预测和后续 SAE / circuit tracing 接轨。
```

需要收紧的地方：

```text
1. 文档中对主流研究的部分引用和日期应作为背景材料，不作为本阶段实验证据。

2. 部分数学公式存在 markdown 排版错误，例如 ======== 被插入公式中。

3. “语言机制完成度百分比”只能作为主观闭合度，不应作为客观科学进度。

4. SAE / circuit tracing 现在适合作为后续接轨方向，
   但本阶段仍应先完成 graph atlas 的干净反事实控制。
```

### 生成脚本

```bash
tests/gpt5/phase670_graph_atlas_counterfactual_control_set.py
```

### 执行命令

```bash
python -m py_compile tests/gpt5/phase670_graph_atlas_counterfactual_control_set.py
python tests/gpt5/phase670_graph_atlas_counterfactual_control_set.py
```

### 输出文件

```text
results/glm5_phase670_graph_atlas_counterfactual_control_set/phase670_counterfactual_control_set.json
results/glm5_phase670_graph_atlas_counterfactual_control_set/phase670_cases.jsonl
results/glm5_phase670_graph_atlas_counterfactual_control_set/phase670_pairs.jsonl
results/glm5_phase670_graph_atlas_counterfactual_control_set/phase670_counterfactual_control_set.md
```

### 客观结果

```text
n_cases = 630
n_pairs = 462
```

样本族：

```text
same_value_different_format: 432
different_value_same_format: 48
same_format_random_value: 72
same_prefix_different_continuation: 24
factor_isolation: 54
```

控制对：

```text
same_value_different_format: 360
different_value_same_format: 48
same_prefix_different_continuation: 18
factor_isolation: 36
```

覆盖的图谱节点：

```text
semantic_value_support: 570
protocol_execution_field: 468
format_continuation_state: 240
task_intent_gate: 180
value_specific_token1_transition_state: 144
first_token_readout_closure: 96
multi_competitor_readout: 72
continuation_controller: 24
```

未被 prompt-level control 直接覆盖的节点：

```text
writer_topology
residual_boundary_integrated_state
```

原因：

```text
这两个节点需要后续内部 activation / component / boundary restore 测试，
不能只靠输入输出反事实控制验证。
```

### 测试原理

Phase 670 使用合成 in-context record，避免模型真实知识记忆干扰。

例如：

```text
Record: daxor color is blue; daxor tool is hammer; ...
Question: What is the color of daxor?
Instruction: Answer with only the value.
Answer:
```

这样后续模型测试更接近：

```text
模型能否把上下文中的值通过正确的意图、协议、读出和续写路线输出
```

而不是：

```text
模型是否记得外部世界事实
```

### 阶段性判断

Phase 670 把 Phase 669 的图谱节点转化为可执行控制矩阵，是同一 graph-atlas validation 阶段的必要步骤。它本身不证明机制因果性，但为下一阶段自然轨迹预测和后续内部干预提供了干净样本基础。

## Phase 671: Graph Atlas Counterfactual Tokenizer Validation [2026-06-26 10:22]

### 本阶段目标

继续完成 Phase 670 的停止条件：正式模型前向测试前，必须验证控制集在 qwen3、GLM4、DS7B 三个 tokenizer 下是否有效。

本阶段只加载 tokenizer，不加载模型权重，不进行生成，不占用 GPU。

### 生成脚本

```bash
tests/gpt5/phase671_graph_atlas_counterfactual_tokenizer_validation.py
```

### 执行命令

严格按模型顺序执行，并加入 `--hard-exit-after-model`：

```bash
python -m py_compile tests/gpt5/phase671_graph_atlas_counterfactual_tokenizer_validation.py
python tests/gpt5/phase671_graph_atlas_counterfactual_tokenizer_validation.py --model qwen3 --hard-exit-after-model
python tests/gpt5/phase671_graph_atlas_counterfactual_tokenizer_validation.py --model glm4 --hard-exit-after-model
python tests/gpt5/phase671_graph_atlas_counterfactual_tokenizer_validation.py --model deepseek7b --hard-exit-after-model
python tests/gpt5/phase671_graph_atlas_counterfactual_tokenizer_validation.py --summarize-only
```

### 输出文件

```text
results/glm5_phase671_graph_atlas_counterfactual_tokenizer_validation/phase671_qwen3_tokenizer_validation_confirm.json
results/glm5_phase671_graph_atlas_counterfactual_tokenizer_validation/phase671_glm4_tokenizer_validation_confirm.json
results/glm5_phase671_graph_atlas_counterfactual_tokenizer_validation/phase671_deepseek7b_tokenizer_validation_confirm.json
results/glm5_phase671_graph_atlas_counterfactual_tokenizer_validation/phase671_cross_model_summary.json
results/glm5_phase671_graph_atlas_counterfactual_tokenizer_validation/phase671_cross_model_summary.md
```

### 客观结果

qwen3：

```text
n_cases = 630
n_pairs = 462
invalid_case_count = 0
invalid_pair_count = 0
same_prefix_valid_pair_count = 18 / 18
max_prompt_tokens = 63
max_expected_tokens = 16
status = pass
```

GLM4：

```text
n_cases = 630
n_pairs = 462
invalid_case_count = 0
invalid_pair_count = 0
same_prefix_valid_pair_count = 18 / 18
max_prompt_tokens = 63
max_expected_tokens = 16
status = pass
```

DS7B：

```text
n_cases = 630
n_pairs = 462
invalid_case_count = 0
invalid_pair_count = 0
same_prefix_valid_pair_count = 18 / 18
max_prompt_tokens = 63
max_expected_tokens = 16
status = pass
```

跨模型状态：

```text
status = pass
```

### 原理解释

Phase 671 验证三个条件：

```text
1. expected_output 不能在 tokenizer 下变成空 token。

2. prompt 长度不能过长，本阶段最大只有 63 tokens。

3. same-prefix / different-continuation 控制必须在 tokenizer 下真的共享首个 expected token，
   且后续 token 序列不同。
```

第三点非常关键，因为 Phase 665-668 的核心发现是：

```text
correct_prefix top1 不等于 token1 / continuation 正确。
```

如果 same-prefix 控制在 tokenizer 下不共享首 token，就不能作为 token1 transition / continuation controller 的干净测试。

本阶段结果显示：

```text
qwen3、GLM4、DS7B 三个模型均通过 tokenizer 验证；
Phase 670 控制集可以进入下一阶段自然轨迹测试。
```

### 当前硬伤

```text
1. Phase 671 只验证 tokenizer，不验证模型行为。

2. synthetic record 可以降低世界知识干扰，
   但也可能降低自然语言开放任务代表性。

3. writer_topology 和 residual_boundary_integrated_state 仍未被 prompt-level control 覆盖。

4. Phase 672 如果进行模型前向测试，需要记录自然生成、first-token、multi-competitor margin、
   token1/token2 continuation，而不能只看 exact match。
```

### 下一阶段

Phase 672 应进入：

```text
Graph Atlas Counterfactual Natural Trajectory Audit
```

测试目标：

```text
不用 patch，只看自然 forward trajectory，
验证 Phase 670 控制集能否区分：

1. semantic value change
2. format protocol change
3. task intent change
4. first-token readout competition
5. token1 continuation failure
```

执行要求：

```text
qwen3 -> GLM4 -> DS7B
每个模型单独运行
必须带 --hard-exit-after-model
```

### 阶段性结论

Phase 670-671 和 Phase 669 处于同一个 graph atlas validation 大阶段。当前已经完成：

```text
1. 图谱节点整理；
2. 反事实控制集构造；
3. 三模型 tokenizer 验证。
```

因此下一步可以进入真正的自然轨迹模型测试，但那将是新的模型前向测试阶段，需要完整记录生成、读出竞争和续写指标。

## Phase 672: Graph Atlas Counterfactual Natural Trajectory Audit [2026-06-26 10:36]

### 本阶段目标

分析用户上传的 Phase 670-671 评价是否正确，并在 Phase 670 控制集和 Phase 671 tokenizer 验证通过的基础上，继续完成同一 graph atlas validation 阶段的自然轨迹模型测试。

本阶段不做 patch，不做 restore，不做 ablation，只观察模型自然 forward / generation 轨迹。

目标是验证：

```text
1. graph atlas 控制集是否能区分 value / format / intent / continuation 变量；
2. first-token top1 是否能预测完整生成；
3. same-prefix / different-continuation 是否暴露 token1 / token2 续写问题；
4. 三个模型在自然轨迹中的失败模式是否不同。
```

### 对上传分析的判断

上传分析主体正确：

```text
Phase 670-671 没有新增机制因果结论，
但完成了从机制图谱到可执行反事实验证矩阵的关键准备。
```

其中最重要的判断是：

```text
如果没有 clean counterfactual controls，
后续 patch 会继续把语义、格式、意图、前缀、续写、残差尺度混在一起。
```

需要收紧的地方：

```text
1. Phase 670-671 只是控制集和 tokenizer 准备，不是模型行为证明。

2. synthetic record 能降低 world knowledge 干扰，
   但不能代表开放式自然语言任务。

3. writer_topology 和 residual_boundary_integrated_state 仍不能靠 prompt-level control 验证。

4. Phase 672 的 token1 / token2 指标对单 token answer 不完全等价于续写机制，
   因此 same-prefix / different-continuation 家族更适合解释续写指标。
```

### 生成脚本

```bash
tests/gpt5/phase672_graph_atlas_counterfactual_natural_trajectory_audit.py
```

### 执行命令

严格按 qwen3 -> GLM4 -> DS7B 顺序执行，每个模型都加入 `--hard-exit-after-model`：

```bash
python -m py_compile tests/gpt5/phase672_graph_atlas_counterfactual_natural_trajectory_audit.py

python tests/gpt5/phase672_graph_atlas_counterfactual_natural_trajectory_audit.py \
  --model qwen3 \
  --max-cases 630 \
  --batch-size 8 \
  --max-new-tokens 20 \
  --top-k 20 \
  --hard-exit-after-model

python tests/gpt5/phase672_graph_atlas_counterfactual_natural_trajectory_audit.py \
  --model glm4 \
  --max-cases 630 \
  --batch-size 4 \
  --max-new-tokens 20 \
  --top-k 20 \
  --hard-exit-after-model

python tests/gpt5/phase672_graph_atlas_counterfactual_natural_trajectory_audit.py \
  --model deepseek7b \
  --max-cases 630 \
  --batch-size 6 \
  --max-new-tokens 20 \
  --top-k 20 \
  --hard-exit-after-model

python tests/gpt5/phase672_graph_atlas_counterfactual_natural_trajectory_audit.py --summarize-only
```

### 输出文件

```text
results/glm5_phase672_graph_atlas_counterfactual_natural_trajectory_audit/phase672_qwen3_natural_trajectory_confirm.json
results/glm5_phase672_graph_atlas_counterfactual_natural_trajectory_audit/phase672_qwen3_natural_trajectory_rows.jsonl

results/glm5_phase672_graph_atlas_counterfactual_natural_trajectory_audit/phase672_glm4_natural_trajectory_confirm.json
results/glm5_phase672_graph_atlas_counterfactual_natural_trajectory_audit/phase672_glm4_natural_trajectory_rows.jsonl

results/glm5_phase672_graph_atlas_counterfactual_natural_trajectory_audit/phase672_deepseek7b_natural_trajectory_confirm.json
results/glm5_phase672_graph_atlas_counterfactual_natural_trajectory_audit/phase672_deepseek7b_natural_trajectory_rows.jsonl

results/glm5_phase672_graph_atlas_counterfactual_natural_trajectory_audit/phase672_cross_model_summary.json
results/glm5_phase672_graph_atlas_counterfactual_natural_trajectory_audit/phase672_cross_model_summary.md
```

### 测试指标

本阶段没有只看 exact match，而是记录：

```text
normalized_exact_rate
compact_exact_rate
contains_value_rate
first_expected_top1_rate
token1_match_rate
token2_match_rate
mean_expected_rank
mean_multi_margin
top1_category
generated_class
```

通俗解释：

```text
normalized_exact_rate:
  去掉多余空格后是否从期望答案开始。

compact_exact_rate:
  去掉所有空格后是否从期望答案开始，适合 JSON / label 等格式。

contains_value_rate:
  生成内容是否至少包含正确值。

first_expected_top1_rate:
  第一步读出时，期望首词元是否 top1。

token1 / token2:
  生成序列前几个词元是否跟期望序列一致。

mean_multi_margin:
  期望首词元相对最佳竞争词元的平均边际。
```

### 跨模型总体结果

```text
qwen3:
  cases = 630
  normalized_exact = 0.568
  compact_exact = 0.700
  contains_value = 0.710
  first_expected_top1 = 0.963
  token1_match = 0.546
  token2_match = 0.343
  mean_expected_rank = 1.09
  mean_multi_margin = 6.843

GLM4:
  cases = 630
  normalized_exact = 0.543
  compact_exact = 0.648
  contains_value = 0.668
  first_expected_top1 = 0.913
  token1_match = 0.592
  token2_match = 0.344
  mean_expected_rank = 1.11
  mean_multi_margin = 1.888

DS7B:
  cases = 630
  normalized_exact = 0.175
  compact_exact = 0.241
  contains_value = 0.540
  first_expected_top1 = 0.483
  token1_match = 0.357
  token2_match = 0.179
  mean_expected_rank = 70.44
  mean_multi_margin = -0.938
```

### 分家族结果

qwen3：

```text
different_value_same_format:
  normalized_exact = 1.000
  first_expected_top1 = 1.000

same_format_random_value:
  normalized_exact = 0.931
  first_expected_top1 = 0.931

same_prefix_different_continuation:
  normalized_exact = 1.000
  first_expected_top1 = 1.000
  token1_match = 1.000
  token2_match = 0.208

same_value_different_format:
  normalized_exact = 0.465
  first_expected_top1 = 1.000
  token1_match = 0.544
```

GLM4：

```text
different_value_same_format:
  normalized_exact = 0.625
  first_expected_top1 = 0.667

same_format_random_value:
  normalized_exact = 1.000
  first_expected_top1 = 1.000

same_prefix_different_continuation:
  normalized_exact = 0.958
  first_expected_top1 = 1.000
  token1_match = 0.958
  token2_match = 0.208

same_value_different_format:
  normalized_exact = 0.442
  first_expected_top1 = 0.928
  token1_match = 0.602
```

DS7B：

```text
different_value_same_format:
  normalized_exact = 0.042
  first_expected_top1 = 0.042
  mean_expected_rank = 120.21

same_format_random_value:
  normalized_exact = 0.111
  first_expected_top1 = 0.111
  mean_expected_rank = 459.96

same_prefix_different_continuation:
  normalized_exact = 0.042
  first_expected_top1 = 0.042
  token1_match = 0.042

same_value_different_format:
  normalized_exact = 0.204
  first_expected_top1 = 0.618
  token1_match = 0.458
```

### 关键客观现象

#### 现象一：qwen3 和 GLM4 的 first-token readout 很强，但完整生成明显掉落

```text
qwen3 first_expected_top1 = 0.963
qwen3 normalized_exact = 0.568

GLM4 first_expected_top1 = 0.913
GLM4 normalized_exact = 0.543
```

这说明：

```text
first-token readout closure 仍然不等于 full generation closure。
```

这个结论和 Phase 665-668 一致。

#### 现象二：same_value_different_format 是最稳定暴露协议问题的家族

qwen3：

```text
first_expected_top1 = 1.000
normalized_exact = 0.465
```

GLM4：

```text
first_expected_top1 = 0.928
normalized_exact = 0.442
```

DS7B：

```text
first_expected_top1 = 0.618
normalized_exact = 0.204
```

这说明：

```text
同一个 value 在不同 format 下，主要瓶颈不是值是否存在，
而是 protocol execution / format continuation / generation policy。
```

#### 现象三：same_prefix_different_continuation 对 qwen3 / GLM4 很干净，对 DS7B 很困难

```text
qwen3 normalized_exact = 1.000
GLM4 normalized_exact = 0.958
DS7B normalized_exact = 0.042
```

这说明：

```text
qwen3 / GLM4 在这个合成同前缀任务中能跟住短续写；
DS7B 在自然轨迹下被 word/explanation/newline 路线强烈吸走。
```

#### 现象四：DS7B 的自然轨迹失败是全局性的

DS7B 总体：

```text
first_expected_top1 = 0.483
mean_expected_rank = 70.44
mean_multi_margin = -0.938
```

DS7B top1 category：

```text
expected: 299
word_or_explanation: 228
newline: 42
other: 37
json_or_quote: 24
```

这说明 DS7B 的失败不是单纯 continuation failure，而是更靠前的：

```text
readout competition + protocol prior + explanation policy
```

共同压制短值答案路线。

#### 现象五：random-value 控制区分出“上下文值绑定能力”的模型差异

```text
qwen3 random-value normalized_exact = 0.931
GLM4 random-value normalized_exact = 1.000
DS7B random-value normalized_exact = 0.111
```

这说明：

```text
qwen3 / GLM4 能较好从 synthetic record 绑定 nonce value；
DS7B 对 nonce short-value 输出极不稳定，更倾向解释/普通词路线。
```

### 理论进展

Phase 672 第一次用 graph atlas 反事实控制集完成自然轨迹测试。当前理论可以从：

```text
patch 能不能修复某个点
```

推进到：

```text
自然轨迹是否按 graph node 分解出现可预测失败模式
```

新的实证链条是：

$$
S_{\text{value}}
\rightarrow
G_{\text{intent}}
\rightarrow
P_{\text{format}}
\rightarrow
R_{\text{multi}}
\rightarrow
C_{\text{continuation}}
\rightarrow
\hat{y}
$$

Phase 672 支持：

```text
1. value 支持和 format 生成可以明显分离。
2. first-token readout 和完整生成继续分离。
3. qwen3 / GLM4 / DS7B 共享功能节点，但自然失败模式不同。
4. DS7B 的短值失败更像全局协议/读出偏置，而不是单一 token1 writer 问题。
```

### 当前硬伤

```text
1. Phase 672 是自然前向测试，不是内部因果干预。

2. token1_match / token2_match 对单 token answer 的解释有限，
   更适合 same-prefix / different-continuation 家族。

3. normalized_exact 对 explanation 格式较严格，
   有些输出可能语义正确但措辞不同，因此 compact / contains_value 也必须保留。

4. synthetic record 仍不能代表开放式自然问答、代码、数学推理。

5. DS7B 的失败虽然强，但还没定位内部来源。
```

### 下一阶段

Phase 673 应进入：

```text
Graph Atlas Natural Failure Taxonomy and Internal Entry Selection
```

目标：

```text
1. 从 Phase 672 rows 中抽取高质量失败样本；
2. 按 failure class 分类：
   readout_failure
   protocol_failure
   format_continuation_failure
   value_binding_failure
   continuation_failure

3. 为每类失败选择后续内部测试入口：
   DS7B short/random-value failure
   DS7B same-prefix continuation failure
   qwen3 / GLM4 same-value different-format protocol failure

4. 不急着做 patch，先完成 failure taxonomy。
```

### 阶段性结论

Phase 672 是 graph atlas validation 的关键正结果。它证明 Phase 670 控制集不是形式准备，而能在自然生成中产生清晰、跨模型不同的失败图谱。

最重要的结论不是哪个模型准确率最高，而是：

```text
qwen3 / GLM4:
  first-token readout 多数已经能进正确路线，
  但 protocol / format / continuation 仍限制完整生成。

DS7B:
  自然短值路线整体被 explanation / word / newline 竞争压制，
  需要先做 failure taxonomy，再进入内部节点定位。
```

## Phase 673: Graph Atlas Natural Failure Taxonomy and Internal Entry Selection [2026-06-26 10:39]

### 本阶段目标

继续完成 Phase 672 后处理：不再增加模型测试，而是把自然轨迹结果整理成 failure taxonomy，并选择下一轮内部激活测试入口。

这是同一 graph atlas validation 阶段的阶段性收束。

### 生成脚本

```bash
tests/gpt5/phase673_graph_atlas_natural_failure_taxonomy.py
```

### 执行命令

```bash
python -m py_compile tests/gpt5/phase673_graph_atlas_natural_failure_taxonomy.py
python tests/gpt5/phase673_graph_atlas_natural_failure_taxonomy.py
```

### 输出文件

```text
results/glm5_phase673_graph_atlas_natural_failure_taxonomy/phase673_failure_taxonomy.json
results/glm5_phase673_graph_atlas_natural_failure_taxonomy/phase673_failure_taxonomy.md
```

### 分类原则

Phase 673 不是因果测试，而是对自然输出做启发式分类：

```text
success:
  normalized exact 成功。

readout_competitor_failure:
  期望首词元不是 top1，且 top1 被 space / newline / word / explanation / json 等竞争项占据。

readout_other_failure:
  期望首词元不是 top1，但竞争项不属于主要已知类别。

protocol_route_failure:
  首词元进入了期望路线，但生成格式类别偏离期望格式。

value_binding_failure:
  首词元进入或接近正确路线，但生成中不包含期望值。

format_surface_failure:
  包含值但表面格式不闭合。

continuation_transition_failure:
  same-prefix 样本中，入口后续 token 没有跟上。
```

### 客观结果

qwen3：

```text
success_rate = 0.568
dominant_failure = value_binding_failure

success: 358
value_binding_failure: 166
other_generation_failure: 83
readout_other_failure: 22
format_surface_failure: 1
```

GLM4：

```text
success_rate = 0.543
dominant_failure = value_binding_failure

success: 342
value_binding_failure: 160
other_generation_failure: 66
readout_competitor_failure: 30
readout_other_failure: 22
protocol_route_failure: 7
format_surface_failure: 2
continuation_transition_failure: 1
```

DS7B：

```text
success_rate = 0.175
dominant_failure = readout_competitor_failure

readout_competitor_failure: 281
value_binding_failure: 155
success: 110
other_generation_failure: 42
readout_other_failure: 36
format_surface_failure: 6
```

### 关键洞察

#### 洞察一：DS7B 不应立刻继续 token1 writer patch

DS7B 的主导失败是：

```text
readout_competitor_failure
```

这意味着大量样本还没有稳定进入 value route。此时直接追 token1 transition writer 会跳过更早的失败点。

更合理的顺序是：

```text
先定位 DS7B 为什么被 word / explanation / newline 竞争项截走，
再研究 token1 transition。
```

#### 洞察二：qwen3 和 GLM4 更适合做 protocol / format continuation 研究

qwen3 和 GLM4 的 first-token readout 相对强：

```text
qwen3 first_expected_top1 = 0.963
GLM4 first_expected_top1 = 0.913
```

但完整生成只有：

```text
qwen3 normalized_exact = 0.568
GLM4 normalized_exact = 0.543
```

因此 qwen3 / GLM4 更适合研究：

```text
first token 之后的 protocol surface formation
format continuation
value binding into full answer
```

#### 洞察三：自然轨迹已经能分流模型

Phase 672-673 的价值不是证明某个 patch 有效，而是把模型分成不同内部入口：

```text
qwen3:
  first-token mostly closed，下一步看 protocol / format continuation。

GLM4:
  first-token mostly closed，但仍有 space/newline 竞争和 value binding 问题。

DS7B:
  first-token route 本身大面积失败，优先看 readout competitor / protocol prior。
```

### 下一轮内部测试入口

优先级 1：

```text
model: DS7B
target: same_format_random_value
failure_class: readout_competitor_failure
next_internal_test:
  trace short-value first-token readout and compare word/explanation competitors at final residual.
```

优先级 2：

```text
model: DS7B
target: same_prefix_different_continuation
failure_class: readout_competitor_failure / continuation_transition_failure
next_internal_test:
  first fix or localize readout/protocol entry before token1 transition patching.
```

优先级 3：

```text
model: DS7B
target: list format
failure_class: protocol_route_failure
next_internal_test:
  compare list marker '-' against explanation word competitors at protocol field layers.
```

优先级 4：

```text
model: qwen3
target: same_value_different_format
failure_class: protocol_route_failure / format_surface_failure
next_internal_test:
  protocol surface formation after first expected token, especially explanation/list/json formatting.
```

优先级 5：

```text
model: GLM4
target: different_value_same_format
failure_class: readout_competitor_failure
next_internal_test:
  space/newline readout source under synthetic in-context value binding.
```

### 当前硬伤

```text
1. failure taxonomy 是启发式标签，不是因果归因。

2. value_binding_failure 可能混入 explanation 格式的宽松语义正确输出，
   因此后续要结合 contains_value、compact_exact 和人工抽样检查。

3. protocol_route_failure 和 format_surface_failure 之间仍有边界模糊。

4. DS7B 的 readout_competitor_failure 还没有定位到内部层、组件、读出几何或协议场。
```

### 阶段性结论

Phase 669-673 共同完成了一个阶段性目标：

```text
1. 建立 graph atlas；
2. 构造反事实控制集；
3. 完成三模型 tokenizer 验证；
4. 完成三模型自然轨迹测试；
5. 把自然失败分流成下一轮内部测试入口。
```

下一阶段不应继续扩大 prompt-level 控制集，而应进入内部机制定位：

```text
Phase 674: DS7B Synthetic Value Readout Competitor Source Localization
```

核心目标：

```text
解释 DS7B 为什么在 synthetic record 已给出 value 的情况下，
仍然被 word / explanation / newline 竞争项压制。
```

## Phase 674: Synthetic Value Readout Competitor Source Localization [2026-06-26 10:51]

### 任务来源

本阶段分析附件中关于 Phase 672-673 的判断，并继续完成下一步测试。

附件判断基本正确：

```text
Phase 672-673 的关键进展不是证明 graph atlas 已经闭合，
而是把 graph atlas 从静态图谱推进到自然生成轨迹，并通过失败分类暴露 DS7B 的主要瓶颈。
```

正确部分：

```text
1. Phase 672 已经完成三模型自然轨迹测试。
2. Phase 673 的 failure taxonomy 虽然仍是启发式，但能给出下一轮内部机制定位入口。
3. DS7B 的 same_format_random_value 失败主要不是 tokenizer 问题，
   也不是简单缺少 synthetic value，而是 readout competitor 压制。
4. qwen3 和 GLM4 在该类任务上明显更稳定，因此 DS7B 应作为 Phase 674 的主要定位对象。
```

需要收紧的部分：

```text
1. Phase 673 的 failure taxonomy 不是因果证明，只是入口定位。
2. readout_competitor_failure 还需要拆成 logits 几何项，而不能只看输出文本。
3. 如果只继续做 prompt 级测试，会重复确认现象，无法解释竞争项从哪里来。
```

### 生成脚本

```text
tests/gpt5/phase674_synthetic_value_readout_competitor_source_localization.py
```

脚本修正：

```text
测试中发现 top1_text 与 competitor.text 在少数行的记录含义容易混淆。
已修正为 raw top1 与 competitor 分开记录：

top1_id / top1_text / top1_category:
  表示真实 logits argmax。

competitor:
  表示排除 expected token 后的最高竞争 token。

competitor_is_top1:
  表示 competitor 是否就是真实 top1。
```

### 测试命令

```bash
python -m py_compile tests/gpt5/phase674_synthetic_value_readout_competitor_source_localization.py

python tests/gpt5/phase674_synthetic_value_readout_competitor_source_localization.py --model qwen3 --max-cases 72 --top-k 20 --hard-exit-after-model

python tests/gpt5/phase674_synthetic_value_readout_competitor_source_localization.py --model glm4 --max-cases 72 --top-k 20 --hard-exit-after-model

python tests/gpt5/phase674_synthetic_value_readout_competitor_source_localization.py --model deepseek7b --max-cases 72 --top-k 20 --hard-exit-after-model

python tests/gpt5/phase674_synthetic_value_readout_competitor_source_localization.py --summarize-only
```

三模型按顺序运行，并且都使用 `--hard-exit-after-model`，避免 GPU 显存残留。

### 测试数据

输入控制集：

```text
results/glm5_phase670_graph_atlas_counterfactual_control_set/phase670_counterfactual_control_set.json
```

本阶段只抽取：

```text
family = same_format_random_value
cases = 72
relations = color / tool / place / symbol
```

选择原因：

```text
Phase 673 显示 DS7B 在 same_format_random_value 中存在高比例 readout_competitor_failure。
该族控制了 format 基本一致、value 被 synthetic record 明确给出，
因此适合定位“正确 value token 为什么没有被读出”。
```

### 测试原理

对每个样本，在 prompt last token 位置读取：

```text
1. final_norm_input
2. final_norm_output
3. lm_head logits
4. expected token rank
5. top competitor token
```

对 expected token 与 competitor token 进行读出几何分解：

```text
logit_i = <h, W_i> + b_i
```

近似拆成：

```text
logit_i ≈ ||h|| · ||W_i|| · cos(h, W_i) + b_i
```

比较：

```text
expected token:
  logit_e, ||W_e||, cos(h, W_e)

competitor token:
  logit_c, ||W_c||, cos(h, W_c)

gap:
  logit_c - logit_e
```

诊断规则：

```text
1. 如果 logit_c - logit_e <= 0：
   expected_wins

2. 如果 competitor 的 unit_score 大于 expected：
   direction_alignment

3. 如果 competitor 主要靠更大的 unembedding norm 获胜：
   projection_norm_advantage

4. 其他情况：
   bias_or_other
```

### 结果文件

```text
results/glm5_phase674_synthetic_value_readout_competitor_source_localization/phase674_cross_model_summary.md
results/glm5_phase674_synthetic_value_readout_competitor_source_localization/phase674_cross_model_summary.json
results/glm5_phase674_synthetic_value_readout_competitor_source_localization/phase674_qwen3_synthetic_value_readout_source_confirm.json
results/glm5_phase674_synthetic_value_readout_competitor_source_localization/phase674_glm4_synthetic_value_readout_source_confirm.json
results/glm5_phase674_synthetic_value_readout_competitor_source_localization/phase674_deepseek7b_synthetic_value_readout_source_confirm.json
results/glm5_phase674_synthetic_value_readout_competitor_source_localization/phase674_*_synthetic_value_readout_source_rows.jsonl
```

### 核心结果

```text
model       cases  expected_top1_rate  mean_expected_rank  expected_minus_competitor  top1_category
deepseek7b  72     0.125               469.79              -6.301                     word_or_explanation=40, other=17, expected=9, newline=6
glm4        72     1.000               1.00                3.329                      expected=72
qwen3       72     0.931               1.49                5.341                      expected=67, other=5
```

DS7B 结果：

```text
expected_top1_rate = 0.125
mean_expected_rank = 469.79
mean_expected_minus_competitor = -6.301

top1_text:
  " The"   = 40
  " \"     = 12
  " nonce" = 9
  " ?\n\n" = 5
  others   = 6

diagnosed_source:
  direction_alignment = 63
  expected_wins       = 9
```

qwen3 结果：

```text
expected_top1_rate = 0.931
mean_expected_rank = 1.49
mean_expected_minus_competitor = 5.341

主要 top1:
  expected = 67
  other    = 5
```

GLM4 结果：

```text
expected_top1_rate = 1.000
mean_expected_rank = 1.00
mean_expected_minus_competitor = 3.329

主要 top1:
  expected = 72
```

### 客观进展

Phase 674 把 Phase 673 的 DS7B readout_competitor_failure 进一步拆开。

最重要的客观结果：

```text
DS7B 不是没有看到 synthetic value。
在 final_norm_input 层面，很多样本中 expected token 仍然可以强于 competitor。

但是经过 final_norm_output / lm_head 读出后，
competitor token 与最终 hidden state 的方向对齐更强，
导致 " The"、反斜杠、换行等格式/说明类 token 压过 expected value token。
```

这说明：

```text
DS7B 的 value 失败更接近“读出方向场竞争失败”，
而不是简单的 value memory 缺失、tokenizer 缺陷、或 unembedding norm 优势。
```

跨模型对比也支持这一点：

```text
qwen3 和 GLM4 在同一控制集上能稳定把 synthetic value 读成 top1。
DS7B 则被 explanation / format / continuation prior 强烈压制。
```

### 对当前理论的更新

原有链条：

```text
synthetic record
→ value binding
→ attention path
→ protocol field
→ final readout
→ natural generation
```

需要补充为：

```text
synthetic record
→ value binding
→ attention path
→ protocol / continuation field
→ final_norm geometry
→ lm_head direction competition
→ natural generation
```

本阶段新增拼图：

```text
final readout 不是被动读出。
它是一个方向竞争场。

value token 即使存在，也必须在 final hidden state 的方向上压过
format / explanation / continuation prior token。
```

### 严格问题和硬伤

```text
1. Phase 674 仍然不是 causal patch 实验。
   它定位了读出几何竞争，但没有证明哪个内部组件制造了这种方向偏转。

2. pre-final_norm 的 logits 只是诊断代理。
   它不能等价于真实中间层生成结果。

3. direction_alignment 是几何归因，不是组件归因。
   还不知道是 attention output、MLP output、RMSNorm 缩放、还是 residual prior 造成最终方向偏转。

4. 本阶段只测试 same_format_random_value。
   不能直接外推到所有 format protocol 或多 token continuation。

5. DS7B 中 expected_output 的首 token 有时是 newline 或无空格 nonce，
   说明后续仍需要更严格的 token variant 选择规则。
```

### 下一阶段方案

Phase 675 应继续当前阶段性目标，不需要回到 prompt-level 扩数据。

推荐标题：

```text
Phase 675: DS7B Final Readout Direction Field Component Attribution
```

核心目标：

```text
定位 DS7B final_norm_output 中把 hidden state 推向 " The" / "\\" / newline 的来源。
```

测试方案：

```text
1. 继续使用 Phase 674 的 72 个 same_format_random_value 样本。

2. 按 DS7B failure class 分组：
   word_or_explanation top1
   slash/quote top1
   newline top1
   expected top1

3. 在最后若干层分别记录：
   residual_pre_attn
   attn_output
   residual_post_attn
   mlp_output
   residual_post_mlp
   final_norm_input
   final_norm_output

4. 对每个组件计算对 competitor gap 的贡献：
   Δgap(component) = gap(after component) - gap(before component)

5. 找出最常把 expected_wins 推成 competitor_wins 的组件和层。
```

核心公式：

```text
gap_l = logit_c(h_l) - logit_e(h_l)
```

```text
Δgap_l^component = gap_l^after - gap_l^before
```

如果 Phase 675 找到稳定组件：

```text
再进入 causal patch：
把 expected-wins 样本中的该组件状态 patch 到 competitor-wins 样本，
观察 expected rank 是否恢复。
```

如果 Phase 675 仍找不到稳定组件：

```text
说明 DS7B 的 readout competitor 不是单层单组件产生，
需要转向多层轨迹场，而不是继续做单点 patch。
```

## Phase 675: Final Readout Direction Field Component Attribution [2026-06-26 10:56]

### 任务延续判断

根据 Phase 674 的结论，下一步 Phase 675 与当前任务属于同一阶段性目标：

```text
目标不是继续扩大 prompt 测试，
而是解释 DS7B final readout competitor 的内部来源。
```

因此继续自动完成。

### 生成脚本

```text
tests/gpt5/phase675_final_readout_direction_field_component_attribution.py
```

### 测试命令

```bash
python -m py_compile tests/gpt5/phase675_final_readout_direction_field_component_attribution.py

python tests/gpt5/phase675_final_readout_direction_field_component_attribution.py --model qwen3 --max-cases 72 --last-layers 8 --hard-exit-after-model

python tests/gpt5/phase675_final_readout_direction_field_component_attribution.py --model glm4 --max-cases 72 --last-layers 8 --hard-exit-after-model > results/glm5_phase675_final_readout_direction_field_component_attribution/phase675_glm4_run.log 2>&1

python tests/gpt5/phase675_final_readout_direction_field_component_attribution.py --model deepseek7b --max-cases 72 --last-layers 8 --hard-exit-after-model > results/glm5_phase675_final_readout_direction_field_component_attribution/phase675_deepseek7b_run.log 2>&1

python tests/gpt5/phase675_final_readout_direction_field_component_attribution.py --summarize-only
```

### 测试数据

直接使用 Phase 674 的三模型 rows：

```text
results/glm5_phase674_synthetic_value_readout_competitor_source_localization/phase674_*_synthetic_value_readout_source_rows.jsonl
```

每个模型：

```text
cases = 72
family = same_format_random_value
last_layers = 8
```

### 测试原理

Phase 674 已经确定：

```text
expected token 与 competitor token 的最终差距：

gap_final = logit_competitor - logit_expected
```

Phase 675 继续沿最后 8 层自然轨迹读取：

```text
layer_input
attn_plus_residual 后的 mlp_input
mlp_plus_residual 后的 layer_out
final_norm_input
final_norm_output
```

对每个位置计算：

```text
gap(h) = logit_c(h) - logit_e(h)
```

组件贡献定义为：

```text
Δgap_component = gap_after - gap_before
```

解释规则：

```text
Δgap_component > 0:
  该组件把状态推向 competitor。

Δgap_component < 0:
  该组件把状态拉回 expected。
```

### 结果文件

```text
results/glm5_phase675_final_readout_direction_field_component_attribution/phase675_cross_model_summary.md
results/glm5_phase675_final_readout_direction_field_component_attribution/phase675_cross_model_summary.json
results/glm5_phase675_final_readout_direction_field_component_attribution/phase675_qwen3_component_attribution_summary.json
results/glm5_phase675_final_readout_direction_field_component_attribution/phase675_glm4_component_attribution_summary.json
results/glm5_phase675_final_readout_direction_field_component_attribution/phase675_deepseek7b_component_attribution_summary.json
results/glm5_phase675_final_readout_direction_field_component_attribution/phase675_*_component_attribution_rows.jsonl
```

### 核心结果

```text
model       cases  expected_top1_rate  final_gap  strongest_overall_delta
deepseek7b  72     0.125               6.301      L27.attn_plus_residual = +107.620
glm4        72     1.000              -3.329      final_norm = +34.829
qwen3       72     0.931              -5.341      L35.attn_plus_residual = +96.587
```

DS7B overall 最强正向 competitor 推动项：

```text
L27.attn_plus_residual:
  before_gap = -103.740
  after_gap  = 3.879
  delta_gap  = 107.620
  positive_rate = 0.833

final_norm:
  before_gap = -67.681
  after_gap  = 6.301
  delta_gap  = 73.982
  positive_rate = 0.833

L26.attn_plus_residual:
  before_gap = -67.709
  after_gap  = 1.437
  delta_gap  = 69.146
  positive_rate = 0.833
```

DS7B overall 主要拉回 expected 的项：

```text
L26.mlp_plus_residual:
  delta_gap = -105.177

L27.mlp_plus_residual:
  delta_gap = -71.560

L25.mlp_plus_residual:
  delta_gap = -69.427
```

qwen3 对照：

```text
final_gap = -5.341
expected_top1_rate = 0.931

L35.attn_plus_residual:
  delta_gap = +96.587

final_norm:
  delta_gap = +60.114

但最终仍多数 expected top1。
```

GLM4 对照：

```text
final_gap = -3.329
expected_top1_rate = 1.000

final_norm:
  delta_gap = +34.829

L39.attn_plus_residual:
  delta_gap = +31.440

但最终全部 expected top1。
```

### 客观进展

Phase 675 的关键发现不是：

```text
attention 一定是坏的，MLP 一定是好的。
```

而是更精确的轨迹事实：

```text
在三模型中，late attention+residual 和 final_norm 普遍会把 gap 推向 competitor。

但是 qwen3 / GLM4 的整体轨迹仍然保留 expected 优势。

DS7B 的最后层 L27 attention+residual 与 final_norm 把原本 expected 优势翻成 competitor 优势，
因此 DS7B 的失败不是孤立 final lm_head 问题，而是 late residual trajectory + final norm readout 共同造成。
```

这把 Phase 674 的结论推进了一层：

```text
Phase 674:
  DS7B 的失败主要是 final readout direction alignment。

Phase 675:
  该 direction alignment 的最大轨迹来源集中在 DS7B L27 attention+residual 和 final_norm。
```

### 当前理论更新

现有链条应更新为：

```text
synthetic value record
→ value binding
→ attention path
→ protocol / continuation field
→ late attention residual competitor push
→ MLP expected correction
→ final norm direction reshaping
→ lm_head competition
→ natural generation
```

更通俗地说：

```text
最后阶段不是单纯“读出答案”。

模型内部像在做两股力的对抗：

1. attention / protocol / continuation 路线把状态推向解释、格式、续写词；
2. MLP / value 路线把状态拉回正确值；
3. final_norm 再改变方向和尺度，使最终 lm_head 竞争结果确定。
```

DS7B 的问题是：

```text
最后一步对抗中，解释/格式/续写方向赢了。
```

qwen3 和 GLM4 的区别是：

```text
它们也存在 competitor push，
但 expected 方向仍足够强，最终没有被翻盘。
```

### 严格问题和硬伤

```text
1. Phase 675 仍然是 natural trajectory attribution，不是 causal patch。

2. 中间层 gap 使用 lm_head 对中间状态直接投影，是诊断指标，
   不能等同于模型真实在中间层“已经输出这个 token”。

3. attn_plus_residual 包含 attention output 与 residual carry 的合成效果，
   还没有拆成 attention heads、residual state、position/protocol prior。

4. final_norm 的正向 delta 很大，但还没有拆成 RMSNorm 缩放、方向归一化和 readout coupling。

5. 本阶段仍只覆盖 same_format_random_value。
   不能直接代表所有语言机制。
```

### 下一阶段方案

Phase 676 与本阶段连续，但已经从“自然轨迹归因”进入“因果干预验证”，
应作为下一阶段启动，而不混入本轮 Phase 674-675 的定位任务。

推荐标题：

```text
Phase 676: DS7B L27 Attention and Final Norm Causal Suppression Audit
```

核心目标：

```text
验证 DS7B L27 attention+residual 与 final_norm 是否真的是 readout competitor 翻盘的因果来源。
```

测试方案：

```text
1. 选择 DS7B Phase 674 中 competitor_wins 样本：
   word_or_explanation
   other
   newline

2. 选择 DS7B expected_wins 样本作为 donor。

3. 对 L27 attention output / residual post-attn / final_norm input-output 做轻量 patch 或抑制：
   a. replace attention output
   b. subtract competitor direction component
   c. interpolate final_norm output toward expected-wins direction

4. 观察：
   expected_rank 是否下降到接近 1；
   competitor_gap 是否从正变负；
   natural first token 是否从 " The" / "\\" / newline 变成 expected value token。
```

如果 Phase 676 成功：

```text
说明 DS7B value readout failure 已经从现象 → 几何 → 组件 → 因果链条基本闭合。
```

如果 Phase 676 失败：

```text
说明 L27 / final_norm 是轨迹表象，不是可单点修复的因果源，
需要进入多层连续场修复，而不是继续单点 patch。
```

## Phase 676: Late Readout Competitor Causal Suppression Audit [2026-06-26 11:18]

### 任务来源与附件判断

本阶段分析附件中对 Phase 674-675 的判断，并继续推进。

附件判断基本正确：

```text
Phase 674-675 已经把 DS7B 的失败从自然生成失败，
推进到 final readout direction field 中 late attention / residual 与 final_norm 的共同翻盘。
```

需要加入的重要限制：

```text
当前测试模型都是小模型。
结果适合作为小模型内部失败机制定位，
不能直接外推到大模型，也不能把具体层号当成通用结构。
```

因此本阶段只把结论限定为：

```text
在当前三个本地小模型中，
DS7B 的 same_format_random_value 失败可被读出方向干预部分改变。
```

### 生成脚本

```text
tests/gpt5/phase676_late_readout_competitor_causal_suppression_audit.py
```

### 测试命令

```bash
python -m py_compile tests/gpt5/phase676_late_readout_competitor_causal_suppression_audit.py

mkdir -p results/glm5_phase676_late_readout_competitor_causal_suppression_audit

python tests/gpt5/phase676_late_readout_competitor_causal_suppression_audit.py --model qwen3 --max-cases 72 --hard-exit-after-model > results/glm5_phase676_late_readout_competitor_causal_suppression_audit/phase676_qwen3_run.log 2>&1

python tests/gpt5/phase676_late_readout_competitor_causal_suppression_audit.py --model glm4 --max-cases 72 --hard-exit-after-model > results/glm5_phase676_late_readout_competitor_causal_suppression_audit/phase676_glm4_run.log 2>&1

python tests/gpt5/phase676_late_readout_competitor_causal_suppression_audit.py --model deepseek7b --max-cases 72 --hard-exit-after-model > results/glm5_phase676_late_readout_competitor_causal_suppression_audit/phase676_deepseek7b_run.log 2>&1

python tests/gpt5/phase676_late_readout_competitor_causal_suppression_audit.py --summarize-only
```

### 测试原理

继续使用 Phase 674 的 same_format_random_value 样本：

```text
cases = 72
models = qwen3 / GLM4 / DS7B
```

对每个样本定义：

```text
d = W_competitor - W_expected
```

其中：

```text
W_competitor = competitor token 的 lm_head / unembedding 方向
W_expected = expected token 的 lm_head / unembedding 方向
```

干预方式：

```text
1. final_output_remove_comp_a1:
   在 final_norm_output 删除 hidden state 沿 d 的投影。

2. final_output_remove_random_a1:
   删除同范数 random direction，作为对照。

3. final_output_cancel_gap_a1:
   按当前 gap 直接沿 d 方向抵消 competitor gap，作为正控。

4. final_input_remove_comp_a1:
   在 final_norm_input 删除 d 投影。

5. attn_last_remove_comp_a1:
   在最后层 attention output 删除 d 投影。

6. attn_prev_remove_comp_a1:
   在倒数第二层 attention output 删除 d 投影。

7. attn_last_zero_a1:
   直接置零最后层 attention output 的当前 token 位置。
```

核心观测：

```text
expected_top1_rate
mean_expected_rank
mean_gap = logit_competitor - logit_expected
switch_to_expected
damage_success
```

### 结果文件

```text
results/glm5_phase676_late_readout_competitor_causal_suppression_audit/phase676_cross_model_summary.md
results/glm5_phase676_late_readout_competitor_causal_suppression_audit/phase676_cross_model_summary.json
results/glm5_phase676_late_readout_competitor_causal_suppression_audit/phase676_qwen3_causal_suppression_summary.json
results/glm5_phase676_late_readout_competitor_causal_suppression_audit/phase676_glm4_causal_suppression_summary.json
results/glm5_phase676_late_readout_competitor_causal_suppression_audit/phase676_deepseek7b_causal_suppression_summary.json
results/glm5_phase676_late_readout_competitor_causal_suppression_audit/phase676_*_causal_suppression_rows.jsonl
```

### 核心结果

DS7B：

```text
condition                         top1_rate  mean_rank   mean_gap  gap_delta
baseline                          0.125      469.79      6.301     0.000
final_output_remove_comp_a1        0.083      8.81       -0.003    -6.304
final_output_remove_random_a1      0.125      466.06      6.289    -0.012
final_output_cancel_gap_a1         0.472      2.90       -2.455    -8.756
final_input_remove_comp_a1         0.000      52810.07    16.405    10.104
attn_last_remove_comp_a1           0.000      18337.64    12.855    6.554
attn_prev_remove_comp_a1           0.000      782.47      8.018     1.717
attn_last_zero_a1                  0.000      10186.92    6.921     0.620
```

qwen3：

```text
baseline                    top1_rate = 0.931
final_output_remove_random  top1_rate = 0.931
attn_last_remove_comp       top1_rate = 0.931
final_input_remove_comp     top1_rate = 0.000
```

GLM4：

```text
baseline                    top1_rate = 1.000
final_output_remove_random  top1_rate = 1.000
attn_last_remove_comp       top1_rate = 1.000
final_input_remove_comp     top1_rate = 0.083
```

### 客观进展

Phase 676 的最重要结果是：

```text
random direction 基本无效；
final output 的 competitor direction 干预可以显著改变 gap；
但简单删除 projection 只能把 gap 拉到接近 0，不能稳定恢复 expected top1；
直接 gap cancel 正控能把 DS7B expected_top1_rate 从 0.125 提高到 0.472。
```

这说明：

```text
DS7B 的读出竞争确实有 readout-direction 因果成分。
```

但也说明：

```text
单点删除 L27 attention output 或 final_norm_input 的 competitor 方向并不会修复，
反而会破坏状态。
```

因此 Phase 675 的 L27 / final_norm 轨迹归因不能被简单解释成：

```text
只要抑制最后层 attention 就能修复。
```

更准确的解释是：

```text
最终读出方向是因果敏感点；
但上游 attention / norm / residual 是耦合系统，不能用粗暴单点删除修复。
```

### 严格问题和硬伤

```text
1. final_output_cancel_gap_a1 是强正控，使用了当前 gap 信息，
   不能视为自然机制修复。

2. final_output_remove_comp_a1 只把 competitor 和 expected 拉到接近打平，
   但 top1 仍可能被第三方 token 抢走。

3. final_input 和 attention output 的直接删除会破坏隐藏状态分布，
   不能简单认为“attention 不是因果源”。

4. qwen3 / GLM4 中同类干预会严重破坏本来正确的输出，
   说明 readout direction 干预不是通用无害操作。

5. 当前模型较小，DS7B 的失败可能包含小模型特有的不稳定读出和弱协议控制。
```

### 阶段判断

Phase 676 没有完成“自然因果修复”，但完成了一个重要边界：

```text
读出方向是有效杠杆；
但简单单点 suppression 不是正确修复形式。
```

所以不能直接进入理论闭合，应继续做 intervention strength scan。

## Phase 677: Readout Intervention Strength Scan [2026-06-26 11:18]

### 自动继续原因

Phase 676 只有单一 alpha 强度，且结果显示：

```text
final_output_cancel_gap_a1 对 DS7B 有明显部分恢复；
final_output_remove_comp_a1 只能打平 gap；
random control 无效。
```

这是同一阶段的强度问题，不应停在单点 alpha 结论。因此继续 Phase 677。

### 生成脚本

```text
tests/gpt5/phase677_readout_intervention_strength_scan.py
```

### 测试命令

```bash
python -m py_compile tests/gpt5/phase677_readout_intervention_strength_scan.py

mkdir -p results/glm5_phase677_readout_intervention_strength_scan

python tests/gpt5/phase677_readout_intervention_strength_scan.py --model qwen3 --max-cases 72 --hard-exit-after-model > results/glm5_phase677_readout_intervention_strength_scan/phase677_qwen3_run.log 2>&1

python tests/gpt5/phase677_readout_intervention_strength_scan.py --model glm4 --max-cases 72 --hard-exit-after-model > results/glm5_phase677_readout_intervention_strength_scan/phase677_glm4_run.log 2>&1

python tests/gpt5/phase677_readout_intervention_strength_scan.py --model deepseek7b --max-cases 72 --hard-exit-after-model > results/glm5_phase677_readout_intervention_strength_scan/phase677_deepseek7b_run.log 2>&1

python tests/gpt5/phase677_readout_intervention_strength_scan.py --summarize-only
```

### 扫描条件

```text
baseline

final_cancel_gap:
  alpha = 0.25 / 0.5 / 0.75 / 1.0 / 1.25 / 1.5 / 2.0 / 3.0

final_remove_comp:
  alpha = 0.5 / 1.0 / 1.5 / 2.0

final_remove_random:
  alpha = 1.0
```

### 结果文件

```text
results/glm5_phase677_readout_intervention_strength_scan/phase677_cross_model_summary.md
results/glm5_phase677_readout_intervention_strength_scan/phase677_cross_model_summary.json
results/glm5_phase677_readout_intervention_strength_scan/phase677_qwen3_strength_scan_summary.json
results/glm5_phase677_readout_intervention_strength_scan/phase677_glm4_strength_scan_summary.json
results/glm5_phase677_readout_intervention_strength_scan/phase677_deepseek7b_strength_scan_summary.json
results/glm5_phase677_readout_intervention_strength_scan/phase677_*_strength_scan_rows.jsonl
```

### 核心结果

DS7B gap cancel 强度曲线：

```text
condition                  top1_rate  mean_rank  mean_gap   failure_switch
baseline                   0.125      469.79     6.301      0.000
final_cancel_gap_a0p25      0.125      95.03      4.118      0.000
final_cancel_gap_a0p5       0.125      26.42      1.924      0.000
final_cancel_gap_a0p75      0.125      8.40      -0.276      0.095
final_cancel_gap_a1p0       0.472      2.90      -2.455      0.540
final_cancel_gap_a1p25      0.708      1.79      -4.659      0.810
final_cancel_gap_a1p5       0.722      1.57      -6.845      0.825
final_cancel_gap_a2p0       0.833      1.61      -11.224     0.952
final_cancel_gap_a3p0       0.833      3.10      -19.984     0.952
```

DS7B projection removal 曲线：

```text
condition                  top1_rate  mean_rank  mean_gap   failure_switch
final_remove_comp_a0p5      0.125      49.78      3.159      0.000
final_remove_comp_a1p0      0.083      8.81      -0.003      0.000
final_remove_comp_a1p5      0.569      1.92      -3.155      0.635
final_remove_comp_a2p0      0.819      1.43      -6.300      0.937
final_remove_random_a1p0    0.125      471.81     6.291      0.000
```

qwen3 对照：

```text
baseline top1_rate = 0.931

final_cancel_gap_a0p25 / a0p5:
  top1_rate 仍为 0.931

final_cancel_gap_a0p75 之后：
  top1_rate 降到 0.042 或更低，
  success_damage_rate = 1.000
```

GLM4 对照：

```text
baseline top1_rate = 1.000

final_cancel_gap_a0p25 / a0p5 / a0p75:
  top1_rate 仍为 1.000

final_cancel_gap_a1p0:
  top1_rate 降到 0.292

alpha >= 1.25:
  top1_rate = 0.000
```

### 客观进展

Phase 677 证明了 Phase 676 的结果不是 alpha=1 的偶然现象。

DS7B 存在清晰的剂量效应：

```text
干预强度越高，
mean_gap 越负，
failure_switch_rate 越高，
expected_top1_rate 越高。
```

random direction 对 DS7B 基本无效：

```text
baseline mean_gap = 6.301
random mean_gap = 6.291
```

因此可以更严格地说：

```text
DS7B 的失败确实依赖 expected-vs-competitor readout direction。
```

但是不能说：

```text
已经找到自然修复机制。
```

因为：

```text
强读出干预会破坏原本正确样本；
qwen3 / GLM4 在强干预下也会被破坏；
这说明干预是读出层强制转向，不是自然协议修复。
```

### 当前最谨慎结论

```text
Phase 674:
  DS7B 失败是 final readout direction competition。

Phase 675:
  该竞争在自然轨迹中主要由 late attention/residual 和 final_norm 推动。

Phase 676:
  读出方向干预能改变 gap，但简单单点 suppression 不稳定。

Phase 677:
  DS7B 存在稳定剂量效应，证明 readout direction 是有效因果杠杆；
  但强干预不是自然机制修复，会损伤原本正确样本和其他模型。
```

### 小模型限制

```text
1. DS7B 的强失败可能与小模型协议控制弱有关。
2. qwen3 / GLM4 在同一测试上更稳定，但也会被强读出干预破坏。
3. 具体层号、强度阈值、top1 竞争词不能外推到大模型。
4. 但“读出竞争方向可以因果控制输出”的机制层级有继续研究价值。
```

### 下一阶段方案

Phase 678 不应继续做全局强干预，而应做 selective repair。

推荐标题：

```text
Phase 678: Failure-Selective Readout Repair and Damage Control
```

核心目标：

```text
只对 baseline failure 样本施加读出方向修复，
对 baseline success 样本不动，
检查是否能得到高整体 top1_rate 且无 success damage。
```

测试方式：

```text
1. 使用 Phase 677 的 DS7B alpha curve。
2. 选 alpha = 1.25 / 1.5 / 2.0。
3. 只干预 baseline failure cases。
4. baseline success cases 保持原始输出。
5. 计算 conditional overall repair rate。
```

如果 selective repair 成功：

```text
说明 DS7B 的失败样本可以通过读出方向门控修复，
下一步寻找自然 gate 何时决定是否施加该方向。
```

如果 selective repair 仍失败：

```text
说明即使读出方向可控，也不能形成稳定机制，
需要回到多层协议场而不是 final readout。
```

## Phase 678: Failure-Selective Readout Repair Summary [2026-06-26 11:21]

### 自动继续原因

Phase 677 证明强读出干预有剂量效应，但也会损伤原本正确样本。

这自然提出一个同阶段问题：

```text
如果只对 baseline failure 样本施加修复，
baseline success 样本保持不动，
是否可以同时获得高修复率和低损伤？
```

该问题不需要重新运行模型，只需要对 Phase 677 rows 做后处理，因此继续完成 Phase 678。

### 生成脚本

```text
tests/gpt5/phase678_failure_selective_readout_repair_summary.py
```

### 运行命令

```bash
python -m py_compile tests/gpt5/phase678_failure_selective_readout_repair_summary.py

python tests/gpt5/phase678_failure_selective_readout_repair_summary.py
```

### 数据来源

```text
results/glm5_phase677_readout_intervention_strength_scan/phase677_*_strength_scan_rows.jsonl
```

### 计算原理

Phase 678 不做新的模型前向，而是模拟一个 selective gate：

```text
if baseline_success:
    keep baseline output

if baseline_failure:
    apply selected readout intervention result from Phase 677
```

观测：

```text
selective_expected_top1_rate
failure_repair_rate
success_damage_rate
selective_mean_rank
selective_mean_gap
```

这不是自然机制证明，而是回答：

```text
如果模型内部存在一个正确的 failure detector / selective gate，
readout intervention 是否足以修复失败样本，同时不破坏成功样本？
```

### 结果文件

```text
results/glm5_phase678_failure_selective_readout_repair_summary/phase678_selective_repair_summary.md
results/glm5_phase678_failure_selective_readout_repair_summary/phase678_selective_repair_summary.json
```

### 核心结果

DS7B：

```text
condition                  selective_top1  mean_rank  mean_gap   failure_repair  success_damage
baseline                   0.125           469.79     6.301      0.000           0.000
final_cancel_gap_a1p0       0.597           2.79      -2.660      0.540           0.000
final_cancel_gap_a1p25      0.833           1.61      -4.913      0.810           0.000
final_cancel_gap_a1p5       0.847           1.26      -7.155      0.825           0.000
final_cancel_gap_a2p0       0.958           1.04      -11.634     0.952           0.000
final_remove_comp_a1p5      0.681           1.81      -3.372      0.635           0.000
final_remove_comp_a2p0      0.944           1.15      -6.595      0.937           0.000
```

qwen3：

```text
baseline selective_top1 = 0.931
final_cancel_gap_a2p0 selective_top1 = 1.000
final_remove_comp_a2p0 selective_top1 = 1.000
success_damage = 0.000
```

GLM4：

```text
baseline selective_top1 = 1.000
所有 selective 条件保持 1.000
```

### 客观进展

Phase 678 说明：

```text
读出方向修复本身足以修复大量 failure 样本；
真正危险的是“对所有样本无差别施加强干预”。
```

DS7B 在 failure-selective 条件下：

```text
baseline top1_rate = 0.125
selective final_cancel_gap_a2p0 top1_rate = 0.958
success_damage = 0.000
```

这把瓶颈进一步后移：

```text
问题不再只是“能不能修复读出方向”；
问题变成“模型内部是否存在或缺失一个选择门，
能判断什么时候需要启动 readout repair”。
```

### 当前最谨慎结论

```text
1. DS7B 的失败样本可以通过 readout direction intervention 大量修复。
2. 成功样本不应该被同类强干预触碰。
3. 因此自然机制若存在，必须是 selective gate，而不是全局 readout shift。
4. 当前结果仍是小模型上的人工干预，不是大模型普遍机制证明。
```

### 当前硬伤

```text
1. Phase 678 是后处理选择，不是模型自己选择。

2. selective gate 是外部 oracle：
   我们用 baseline success/failure 标签决定是否干预。

3. 干预方向仍直接使用 expected-vs-competitor readout direction，
   不是模型自然生成的内部方向。

4. 结果说明“如果有选择门就能修”，
   但还没有找到模型内部是否有这个选择门。

5. 小模型中 failure detector 可能很弱，
   大模型可能通过更强 instruction/protocol control 自然完成类似选择。
```

### 下一阶段方案

Phase 679 应寻找 selective gate 的内部可测指标。

推荐标题：

```text
Phase 679: Internal Failure-Gate Predictor for Selective Readout Repair
```

核心问题：

```text
在不看最终答案是否正确的情况下，
能不能从内部状态预测当前样本是否需要 readout repair？
```

候选指标：

```text
1. final_norm_input gap
2. final_norm_output gap
3. L27.attn_plus_residual delta_gap
4. L27.mlp_plus_residual correction strength
5. competitor category
6. expected_rank before intervention
7. top1 entropy / margin
```

目标：

```text
建立一个不依赖人工标签的 failure-gate 指标，
把 Phase 678 的 oracle selective repair 推进成可测内部机制。
```

## Phase 679: Internal Failure-Gate Predictor for Selective Readout Repair [2026-06-26 11:28]

### 任务来源

本阶段继续分析 Phase 674-675 附件判断是否正确，并综合 Phase 676-678 的后续结果继续推进。

附件中对 Phase 674-675 的判断基本正确：Phase 674-675 确实把 DS7B 的错误定位到 final readout direction competition 附近，并证明 late attention/residual 与 final_norm 是主要轨迹贡献源。但附件停留在“需要因果验证”的位置；后续 Phase 676-678 已经补上因果与强度扫描：

```text
Phase 676: 粗 suppression 不能自然修复，但 cancel gap 能部分修复。
Phase 677: readout direction intervention 存在稳定剂量效应。
Phase 678: 如果存在 failure-selective gate，只修复失败样本，可以把 DS7B top1 从 0.125 提升到 0.958，且不损伤成功样本。
```

因此 Phase 679 的核心任务不是继续证明 readout intervention 有效，而是审计：

```text
是否存在一个可测的 failure gate，
能决定哪些样本需要 readout repair，
从而把 Phase 678 的 oracle selective repair 推进到可测机制。
```

### 命令

```bash
python -m py_compile tests/gpt5/phase679_internal_failure_gate_predictor.py
python tests/gpt5/phase679_internal_failure_gate_predictor.py
```

本阶段没有重新运行模型 forward，不占用 GPU；它复用 Phase 674、Phase 675、Phase 677 的逐样本结果，做跨模型后处理审计。

### 脚本

```text
tests/gpt5/phase679_internal_failure_gate_predictor.py
```

### 输入结果

```text
results/glm5_phase674_synthetic_value_readout_competitor_source_localization/
results/glm5_phase675_final_readout_direction_field_component_attribution/
results/glm5_phase677_readout_intervention_strength_scan/
```

### 输出结果

```text
results/glm5_phase679_internal_failure_gate_predictor/phase679_failure_gate_predictor.json
results/glm5_phase679_internal_failure_gate_predictor/phase679_failure_gate_predictor.md
```

### 测试原理

Phase 679 不使用复杂统计模型，不训练 classifier，只枚举基础可解释 gate。

每个样本先从已有结果中抽取以下可测量：

```text
1. baseline_success
2. expected_rank
3. final_gap
4. pre_gap
5. post_unit_gap
6. post_cos_advantage
7. final_norm_before_gap
8. final_norm_delta
9. last_attn_delta / prev_attn_delta / last_mlp_delta
10. top1_category
```

然后构造简单门控：

```text
top1_category_not_expected
top1_category_word_or_newline_or_other
expected_rank_gt_1
expected_rank_gt_10
final_gap_gt_0
若干 pre_final_gap / trajectory / readout_geometry 阈值门
```

每个 gate 与 Phase 677 中的 repair 条件组合：

```text
final_cancel_gap_a1p25
final_cancel_gap_a1p5
final_cancel_gap_a2p0
final_remove_comp_a2p0
```

评估指标：

```text
pred_rate: 被 gate 选中比例
failure_capture_rate: 失败样本捕获率
success_false_positive_rate: 成功样本误伤预测率
selective_top1_rate: 只对 gate 选中样本修复后的 top1
failure_repair_rate: 失败样本修复率
success_damage_rate: 成功样本损伤率
```

### 关键结果

#### qwen3

qwen3 baseline 只有少量失败样本。近读出门控能完整捕获失败，且不误伤成功样本：

```text
top1_category_not_expected + final_cancel_gap_a2p0:
pred_rate = 0.069
failure_capture_rate = 1.000
success_false_positive_rate = 0.000
selective_top1_rate = 1.000
failure_repair_rate = 1.000
success_damage_rate = 0.000

final_gap_gt_0 + final_cancel_gap_a2p0:
selective_top1_rate = 1.000
failure_repair_rate = 1.000
success_damage_rate = 0.000
```

#### GLM4

GLM4 baseline 已经全对，因此多数 gate 不触发，selective_top1 保持 1.000：

```text
top1_category_not_expected:
pred_rate = 0.000
selective_top1_rate = 1.000
success_damage_rate = 0.000
```

#### DS7B

DS7B 是本阶段核心。最稳定结果如下：

```text
top1_category_not_expected + final_cancel_gap_a2p0:
pred_rate = 0.875
failure_capture_rate = 1.000
success_false_positive_rate = 0.000
selective_top1_rate = 0.958
failure_repair_rate = 0.952
success_damage_rate = 0.000

final_gap_gt_0 + final_cancel_gap_a2p0:
pred_rate = 0.875
failure_capture_rate = 1.000
success_false_positive_rate = 0.000
selective_top1_rate = 0.958
failure_repair_rate = 0.952
success_damage_rate = 0.000

top1_category_not_expected + final_remove_comp_a2p0:
selective_top1_rate = 0.944
failure_repair_rate = 0.937
success_damage_rate = 0.000

expected_rank_gt_10 + final_cancel_gap_a2p0:
failure_capture_rate = 0.921
selective_top1_rate = 0.889
failure_repair_rate = 0.873
success_damage_rate = 0.000
```

### 负结果

更早的内部 gate 暂时没有稳定成功。

```text
pre_gap
post_unit_gap
post_cos_advantage
last_attn_delta
prev_attn_delta
last_mlp_delta
```

这些 trajectory/readout_geometry 指标在 DS7B 上经常出现：

```text
1. 捕获失败样本不完整；
2. 对成功样本误判较高；
3. 与强 repair 组合后会损伤原本成功样本；
4. 不能稳定复现 Phase 678 oracle selective repair。
```

因此目前最可靠的 gate 仍然是近读出 gate，而不是深层自然协议 gate。

### 当前判断

Phase 679 是一个“正结果 + 负结果”的阶段。

正结果：

```text
readout-side failure gate 可以把 Phase 678 的 oracle selective repair 变成可测规则。
```

具体说，DS7B 中：

```text
如果 top1_category != expected，或 final_gap > 0，
则对该样本执行 final_cancel_gap_a2p0；
否则保持 baseline。
```

可以得到：

```text
baseline top1 = 0.125
selective repair top1 = 0.958
success_damage = 0.000
```

负结果：

```text
当前尚未找到一个稳定的 pre-readout / trajectory gate。
```

也就是说，Phase 679 暂时没有证明模型内部已经自然形成了一个更早的、可直接读出的 failure detector。它只证明在最终读出竞争层面，失败状态已经可以被清楚识别。

### 对附件判断的修正

附件对 Phase 674-675 的核心判断正确，但现在需要补充三点：

```text
1. Phase 674-675 已经不是最新因果状态；
2. Phase 676-677 证明 readout direction 是因果杠杆；
3. Phase 679 证明选择性修复可以由 near-readout gate 实现，但还没有推进到自然内部 gate。
```

因此最谨慎的结论是：

```text
DS7B 的 synthetic value readout 错误，
很大程度上是 final readout competition 的问题；
这个问题可以被近读出门控选择性修复；
但自然语言系统内部更早的 gate / protocol signal 尚未被找到。
```

### 小模型限制

当前 qwen3、GLM4、DS7B 都是小模型或较小规模模型，必须保守解释：

```text
1. DS7B 的失败可能来自小模型协议能力不足，不一定代表大模型内部机制；
2. qwen3 / GLM4 的高正确率可能来自任务太简单或模板贴合，不等于机制闭合；
3. near-readout gate 在小模型中很清楚，但在大模型中可能被更早的 planner/protocol 层吸收；
4. 不能把 Phase 679 解释为“语言机制已破解”，只能解释为“一个局部 readout failure/recovery 拼图被测清楚”。
```

### 理论进展

当前链条从 Phase 674 到 Phase 679 可以压缩成：

```text
错误输出
  = final readout direction competition 偏向 competitor
  ≠ token memory 缺失
  ≠ 单纯 late attention 缺失
  ≠ 粗 suppression 可自然修复
  = 可以被 readout gap cancellation 选择性修复
  = 近读出 gate 可识别需要修复的样本
```

这说明语言生成末端至少存在两个可分层次：

```text
1. value / content path:
   正确值词元已经在内部路径中可被 attention 与 residual 支持。

2. readout / selection path:
   最终 logits 处仍会被 format / continuation / competitor direction 覆盖。
```

统一公式可暂时写成：

```text
h_t^L = S_{\text{content}}(x_t) + S_{\text{format}}(x_t) + S_{\text{continuation}}(x_t) + \epsilon_t
```

```text
z_t(v) = W_U(v)^\top \operatorname{Norm}(h_t^L)
```

```text
\Delta_{\text{value}} =
z_t(v_{\text{expected}}) - z_t(v_{\text{competitor}})
```

```text
G_{\text{fail}} =
\mathbf{1}\left[\Delta_{\text{value}} < 0
\lor \operatorname{Top1Category}(z_t) \ne \text{expected}\right]
```

```text
h_t^{L,\text{repair}} =
h_t^L - \alpha \cdot G_{\text{fail}} \cdot d_{\text{competitor-gap}}
```

其中 Phase 679 只验证了 \(G_{\text{fail}}\) 的 near-readout 版本，没有验证 pre-readout 自然版本。

### 当前硬伤

```text
1. gate 仍然接近最终输出，机制深度不足；
2. 没有证明模型内部自己使用该 gate；
3. 没有跨任务验证到自然语言问答、推理、语法续写；
4. gate 依赖 expected token / competitor category 的人工标注结构；
5. 强 repair 仍是外部干预，不是模型自身计算；
6. 当前模型较小，内部结构可能偏差明显。
```

### 是否自动继续

Phase 679 已完成 Phase 678 遗留的同阶段目标：

```text
寻找 selective readout repair 的可测 failure gate。
```

但下一步不应继续在同一个小阶段内自动展开，因为下一个目标已经从：

```text
near-readout gate 是否存在
```

转为：

```text
pre-readout / natural protocol gate 是否存在，
并且是否能跨任务族泛化。
```

这是一个新的阶段性大任务，需要重新设计数据族与脚本。

### 下一阶段方案

推荐 Phase 680：

```text
Phase 680: Pre-Readout Natural Gate and Cross-Family Generalization Audit
```

核心目标：

```text
把 Phase 679 的 near-readout failure gate
向更早的 natural internal gate 推进。
```

测试族应至少包含：

```text
1. same_format_random_value
2. same_value_different_format
3. different_value_same_format
4. short_answer / natural_answer 对照
5. syntax continuation / semantic value 对照
6. expected success 与 expected failure 混合样本
```

关键观测：

```text
1. gate 是否能在 final logits 之前预测失败；
2. gate 是否只在 DS7B failure 样本中触发；
3. gate 是否能避免损伤 qwen3 / GLM4 成功样本；
4. gate 是否和 L17-L20 protocol trajectory、L22 attention bridge、L27-L31 readout branch 有稳定对应；
5. gate 是否能在自然生成中预测 continuation 被 format/preference 覆盖的情况。
```

阶段性目标：

```text
如果 Phase 680 找不到 pre-readout gate，
则说明当前小模型中的 failure detector 主要只在 readout 端显化；
如果找到，
则可以把 readout repair 从外部补丁推进成内部协议图谱的一条可测边。
```

## Phase 680: Pre-Readout Natural Gate and Cross-Family Generalization Audit [2026-06-26 11:48]

### 任务来源

本阶段分析 Phase 676-679 附件内容是否正确，并继续推进。

附件判断基本正确：

```text
Phase 676-679 已经把 DS7B synthetic value failure
从最终输出失败推进到 final readout direction competition 的因果调节；
readout direction 是有效杠杆；
selective gate 是必要条件；
但当前只找到 near-readout gate，还没有找到自然 pre-readout gate。
```

需要补充的是：Phase 679 的 gate 仍然非常靠近最终输出，`top1_category_not_expected` 和 `final_gap_gt_0` 都是 near-readout diagnostic gate，不是模型内部更早的自然协议门。因此 Phase 680 继续做跨任务族 pre-readout gate 审计。

### 命令

```bash
python -m py_compile tests/gpt5/phase680_pre_readout_natural_gate_cross_family_audit.py

python tests/gpt5/phase680_pre_readout_natural_gate_cross_family_audit.py \
  --model qwen3 \
  --hard-exit-after-model \
  > results/glm5_phase680_pre_readout_natural_gate_cross_family_audit/phase680_qwen3_run.log 2>&1

python tests/gpt5/phase680_pre_readout_natural_gate_cross_family_audit.py \
  --model glm4 \
  --hard-exit-after-model \
  > results/glm5_phase680_pre_readout_natural_gate_cross_family_audit/phase680_glm4_run.log 2>&1

python tests/gpt5/phase680_pre_readout_natural_gate_cross_family_audit.py \
  --model deepseek7b \
  --hard-exit-after-model \
  > results/glm5_phase680_pre_readout_natural_gate_cross_family_audit/phase680_deepseek7b_run.log 2>&1

python tests/gpt5/phase680_pre_readout_natural_gate_cross_family_audit.py --summarize-only
```

三模型按 qwen3、GLM4、DS7B 顺序运行，每个模型独立进程并使用 `--hard-exit-after-model`。

### 脚本

```text
tests/gpt5/phase680_pre_readout_natural_gate_cross_family_audit.py
```

### 输出

```text
results/glm5_phase680_pre_readout_natural_gate_cross_family_audit/phase680_qwen3_pre_readout_rows.jsonl
results/glm5_phase680_pre_readout_natural_gate_cross_family_audit/phase680_glm4_pre_readout_rows.jsonl
results/glm5_phase680_pre_readout_natural_gate_cross_family_audit/phase680_deepseek7b_pre_readout_rows.jsonl
results/glm5_phase680_pre_readout_natural_gate_cross_family_audit/phase680_cross_model_summary.md
results/glm5_phase680_pre_readout_natural_gate_cross_family_audit/phase680_cross_model_summary.json
```

### 测试原理

Phase 680 不训练 classifier，也不使用复杂统计模型。它在跨任务族样本中枚举基础阈值门。

样本族：

```text
same_format_random_value: 72
same_value_different_format: 144
different_value_same_format: 48
same_prefix_different_continuation: 24
factor_isolation: 54
总计: 每模型 342 cases
```

每个样本记录最终 expected token 是否 top1，但候选 gate 不使用最终 top1。候选 pre-readout 特征包括：

```text
final_norm_input_gap
final_norm_input_rank
last_layer_gap
late_gap_shift
mid_to_late_shift
max_layer_gap
min_layer_gap
positive_layer_count
first_positive_layer_frac
```

near-readout gate 只作为参考线：

```text
REF_final_gap_gt_0
REF_top1_category_not_expected
```

核心指标：

```text
failure_capture_rate: 失败捕获率
success_false_positive_rate: 成功样本误触发率
failure_precision: 触发样本中真实失败比例
gate_score = failure_capture_rate - success_false_positive_rate
```

### 跨模型总体结果

```text
qwen3:
cases = 342
top1_rate = 0.933
failures = 23
best pre-readout gate = final_norm_input_gap_gt_0
pre score = 0.785
failure_capture = 0.826
false_pos = 0.041
best near-readout reference score = 1.000

GLM4:
cases = 342
top1_rate = 0.892
failures = 37
best pre-readout gate = final_norm_input_gap_lt_-36.05
pre score = 0.479
failure_capture = 0.676
false_pos = 0.197
best near-readout reference score = 1.000

DS7B:
cases = 342
top1_rate = 0.395
failures = 207
best pre-readout gate = final_norm_input_gap_gt_-118.8
pre score = 0.280
failure_capture = 0.850
false_pos = 0.570
best near-readout reference score = 1.000
```

### 分族 baseline

DS7B：

```text
different_value_same_format: top1 = 0.042, failure = 0.958
factor_isolation: top1 = 0.537, failure = 0.463
same_format_random_value: top1 = 0.125, failure = 0.875
same_prefix_different_continuation: top1 = 0.042, failure = 0.958
same_value_different_format: top1 = 0.653, failure = 0.347
```

GLM4：

```text
different_value_same_format: top1 = 0.667
factor_isolation: top1 = 0.796
same_format_random_value: top1 = 1.000
same_prefix_different_continuation: top1 = 1.000
same_value_different_format: top1 = 0.931
```

qwen3：

```text
different_value_same_format: top1 = 1.000
factor_isolation: top1 = 0.667
same_format_random_value: top1 = 0.931
same_prefix_different_continuation: top1 = 1.000
same_value_different_format: top1 = 1.000
```

### 分族 pre-readout gate 结果

DS7B 中有分族局部信号：

```text
different_value_same_format:
  final_norm_input_rank_gt_71445
  capture = 0.913
  false_pos = 0.000

same_prefix_different_continuation:
  final_norm_input_gap_gt_-118.8
  capture = 0.870
  false_pos = 0.000

same_format_random_value:
  final_norm_input_rank_gt_71445
  capture = 0.619
  false_pos = 0.000

same_value_different_format:
  final_norm_input_gap_gt_-118.8
  capture = 0.960
  false_pos = 0.628
```

这个结果说明：DS7B 不是完全没有 pre-readout 预警信号，但这些信号强烈依赖任务族；一旦跨族混合，误触发率很高。

qwen3 中 `final_norm_input_gap_gt_0` 比较干净：

```text
overall:
  capture = 0.826
  false_pos = 0.041

factor_isolation:
  capture = 1.000
  false_pos = 0.000
```

GLM4 中信号中等：

```text
overall:
  capture = 0.676
  false_pos = 0.197
```

### 当前判断

Phase 680 是一个重要的收紧阶段。

正结果：

```text
pre-readout diagnostic signal 并非完全不存在。
qwen3 有相对干净的 final_norm_input_gap gate。
DS7B 在个别任务族存在强局部门。
```

负结果：

```text
没有找到跨模型、跨任务族稳定成立的 natural pre-readout gate。
DS7B overall best pre gate false_pos = 0.570，不能作为选择性修复触发器。
near-readout reference 仍然完美，说明失败状态在最终读出端清楚显化，但上游预警不稳定。
```

最谨慎结论：

```text
Phase 680 没有完成自然 gate 闭合。
它把 Phase 679 的 near-readout gate 向上游推进了一步，
发现 qwen3/部分任务族存在 pre-readout 预警信号，
但 DS7B 和跨族泛化仍失败。
```

### 小模型限制

当前结果必须保守解释：

```text
1. DS7B 的 pre-readout gate 不稳定，可能是小模型协议监控能力弱；
2. qwen3 的干净 gate 可能来自任务简单或模型在 final_norm_input 已经接近最终答案；
3. 小模型中的层号和 gate 位置不能外推到大模型；
4. 当前只是 first-token / short-value 相关任务，不代表完整自然语言生成。
```

### 是否继续

Phase 680 的阈值是在同一批样本上枚举出来的，存在过拟合风险。这个问题仍属于同一个阶段目标：

```text
验证 pre-readout natural gate 是否真实存在。
```

因此继续自动完成 Phase 681：对 Phase 680 的 gate 做 deterministic holdout validation。

## Phase 681: Holdout Validation for Pre-Readout Failure Gates [2026-06-26 11:48]

### 任务来源

Phase 680 已经发现若干 pre-readout gate 候选，但阈值门是在全量数据上枚举的，必须进一步审视：

```text
这些 gate 是否只是同批样本上的阈值拟合？
还是在留出样本上仍然成立？
```

### 命令

```bash
python -m py_compile tests/gpt5/phase681_pre_readout_gate_holdout_validation.py
python tests/gpt5/phase681_pre_readout_gate_holdout_validation.py
```

本阶段不重新运行模型，只读取 Phase 680 rows 做 deterministic alternate split。

### 脚本

```text
tests/gpt5/phase681_pre_readout_gate_holdout_validation.py
```

### 输出

```text
results/glm5_phase681_pre_readout_gate_holdout_validation/phase681_holdout_validation.json
results/glm5_phase681_pre_readout_gate_holdout_validation/phase681_holdout_validation.md
```

### 测试原理

每个模型、每个任务组按 `case_id` 排序后交替切分：

```text
偶数位: train
奇数位: test
```

只在 train 中选择最优 pre-readout gate，然后把同一个 gate 直接应用到 test。

near-readout reference gate 同样做留出，用作上限参照。

### 留出验证结果

qwen3：

```text
overall:
  gate = final_norm_input_gap_gt_0
  train_score = 0.714
  test_score = 0.920
  test_capture = 1.000
  test_false_pos = 0.080
  ref_test_score = 1.000

factor_isolation:
  gate = final_norm_input_gap_gt_0
  train_score = 1.000
  test_score = 1.000
  test_capture = 1.000
  test_false_pos = 0.000
```

GLM4：

```text
overall:
  gate = max_layer_gap_lt_-0.5012
  train_score = 0.607
  test_score = 0.320
  test_capture = 0.429
  test_false_pos = 0.108
  ref_test_score = 1.000

different_value_same_format:
  gate = mid_to_late_shift_lt_-13.3
  train_score = 0.857
  test_score = 0.444
  test_capture = 0.833
  test_false_pos = 0.389
```

DS7B：

```text
overall:
  gate = final_norm_input_rank_gt_82560
  train_score = 0.535
  test_score = -0.106
  test_capture = 0.640
  test_false_pos = 0.746
  ref_test_score = 1.000

factor_isolation:
  gate = final_norm_input_rank_gt_96640
  train_score = 0.500
  test_score = 0.544
  test_capture = 0.615
  test_false_pos = 0.071

same_value_different_format:
  gate = final_norm_input_gap_gt_-118.8
  train_score = 0.684
  test_score = 0.000
  test_capture = 1.000
  test_false_pos = 1.000
```

### Cross-family holdout 结果

qwen3：

```text
factor_isolation -> same_format_random_value:
  gate = final_norm_input_gap_gt_0
  target_score = 0.200
  target_capture = 0.200
  target_false_pos = 0.000
```

GLM4：

```text
same_value_different_format -> different_value_same_format:
  target_score = 0.656
  target_capture = 0.938
  target_false_pos = 0.281

different_value_same_format -> same_value_different_format:
  target_score = 0.487
  target_capture = 0.800
  target_false_pos = 0.313
```

DS7B：

```text
same_value_different_format -> same_prefix_different_continuation:
  target_score = 0.870
  target_capture = 0.870
  target_false_pos = 0.000

factor_isolation -> different_value_same_format:
  target_score = 0.565
  target_capture = 0.565
  target_false_pos = 0.000

same_value_different_format -> same_format_random_value:
  target_score = 0.540
  target_capture = 0.651
  target_false_pos = 0.111
```

这些跨族结果有局部信号，但不够统一，也不能说明存在一个通用自然门。

### 当前判断

Phase 681 明显收紧了 Phase 680 的结论。

可以保留的结果：

```text
1. qwen3 的 final_norm_input_gap_gt_0 是目前最稳定的 pre-readout gate；
2. GLM4 存在中等强度 pre-readout 信号，但留出后明显变弱；
3. DS7B overall pre-readout gate 在留出上失败；
4. DS7B 只有少数 family-local / cross-family 局部信号，不能形成统一门。
```

必须删除或降级的结果：

```text
1. 不能说 DS7B 已找到自然 pre-readout gate；
2. 不能说 Phase 680 的 DS7B 分族高分 gate 都可靠；
3. 不能说 pre-readout gate 已经跨任务族泛化；
4. 不能把 near-readout reference 的完美结果混同为自然内部门。
```

### 理论进展

当前链条更新为：

```text
Phase 679:
near-readout failure gate 成立。

Phase 680:
pre-readout signal 存在局部候选，但跨模型/跨族不稳定。

Phase 681:
holdout 后，qwen3 保留较干净 pre-readout gate；
GLM4 降为中等信号；
DS7B overall gate 失败。
```

因此理论上应把 gate 拆成三层：

```text
1. G_readout:
   最终读出端失败诊断门，目前最强，跨模型较稳定。

2. G_pre:
   final_norm_input / late residual 读出前预警门，qwen3 强，GLM4 中等，DS7B 不稳定。

3. G_protocol:
   更早协议/规划自然门，目前尚未找到。
```

公式更新：

```text
G_{\text{fail}} =
G_{\text{readout}}
\;\; \text{or} \;\;
G_{\text{pre}}
\;\; \text{or} \;\;
G_{\text{protocol}}
```

当前实证状态：

```text
G_readout: 已验证
G_pre: 局部验证，未跨模型闭合
G_protocol: 未验证
```

### 硬伤

```text
1. Phase 681 仍是阈值门审计，不是因果证明；
2. pre-readout gate 仍依赖 expected token / competitor token 的人工定义；
3. DS7B 作为最关键失败模型，overall gate 留出失败；
4. cross-family 泛化只有局部结果，没有统一机制；
5. 当前测试仍是小模型，结构可能偏差；
6. 没有测更早层的真正 protocol monitor。
```

### 阶段性结论

Phase 680-681 完成了附件提出的下一阶段任务的一半：

```text
已经审计 pre-readout natural gate 是否存在；
结论是：存在局部 pre-readout 信号，但没有发现跨模型/跨族稳定自然门。
```

因此当前不能继续把理论推进为“自然失败门已找到”。更谨慎地说：

```text
小模型中，失败状态在 final readout 端最清晰；
读出前信号存在但碎片化；
DS7B 尤其缺少稳定的 pre-readout self-monitor。
```

### 下一阶段方案

当前任务和下一任务仍处于同一大方向，但 Phase 681 已完成本小阶段目标：排查 pre-readout gate 是否能从 Phase 680 的同批阈值中站稳。

下一步不应继续盲目扩大阈值搜索，而应进入新的大任务：

```text
Phase 682: Protocol-Level Failure Monitor Localization
```

核心问题：

```text
如果 G_pre 在 DS7B 中不稳定，
是否存在更早的 protocol-level monitor，
它不是 expected-vs-competitor lm_head gap，
而是 format route / value route / continuation route 的路径分叉信号？
```

推荐测试：

```text
1. 不再只用 expected-vs-competitor logit gap；
2. 构造 format-route 与 value-route 的方向集合；
3. 测 L17-L22 的 route margin；
4. 分离 short-answer、sentence、JSON、continuation 四类协议；
5. 检查失败样本是否在协议路由阶段已经偏向错误 route。
```

如果 Phase 682 成功，说明自然 gate 可能不是“值词元预警门”，而是“协议路径分叉门”。

## Phase 682: Protocol-Level Failure Monitor Localization [2026-06-26 12:01]

### 任务来源

本阶段分析 Phase 680-681 附件内容是否正确，并继续推进。

附件判断基本正确：Phase 680-681 不是继续证明 readout direction 可以修复错误，而是把 failure gate 拆成三层：

```text
G_readout: 最终读出端门，已验证；
G_pre: 读出前门，局部成立但跨模型/跨族不稳定；
G_protocol: 更早协议门，尚未验证。
```

附件指出下一步不能继续盲目扩大 expected-vs-competitor gap 阈值搜索，而应构造 format route / value route / continuation route 的路径分叉信号。因此 Phase 682 测试 protocol-level route monitor。

### 命令

```bash
python -m py_compile tests/gpt5/phase682_protocol_route_monitor_localization.py

python tests/gpt5/phase682_protocol_route_monitor_localization.py \
  --model qwen3 \
  --hard-exit-after-model \
  > results/glm5_phase682_protocol_route_monitor_localization/phase682_qwen3_run.log 2>&1

python tests/gpt5/phase682_protocol_route_monitor_localization.py \
  --model glm4 \
  --hard-exit-after-model \
  > results/glm5_phase682_protocol_route_monitor_localization/phase682_glm4_run.log 2>&1

python tests/gpt5/phase682_protocol_route_monitor_localization.py \
  --model deepseek7b \
  --hard-exit-after-model \
  > results/glm5_phase682_protocol_route_monitor_localization/phase682_deepseek7b_run.log 2>&1

python tests/gpt5/phase682_protocol_route_monitor_localization.py --summarize-only
```

三模型按 qwen3、GLM4、DS7B 顺序运行，每个模型独立进程并使用 `--hard-exit-after-model`。

### 脚本

```text
tests/gpt5/phase682_protocol_route_monitor_localization.py
```

### 输出

```text
results/glm5_phase682_protocol_route_monitor_localization/phase682_qwen3_protocol_route_rows.jsonl
results/glm5_phase682_protocol_route_monitor_localization/phase682_glm4_protocol_route_rows.jsonl
results/glm5_phase682_protocol_route_monitor_localization/phase682_deepseek7b_protocol_route_rows.jsonl
results/glm5_phase682_protocol_route_monitor_localization/phase682_cross_model_summary.md
results/glm5_phase682_protocol_route_monitor_localization/phase682_cross_model_summary.json
```

### 测试原理

Phase 682 不再用单个 expected token 对 competitor token 的 gap 作为主要门控，而是为每个样本标注目标协议路线：

```text
value
prose
json
label
list
yesno
continuation
```

目标路线由 expected_output 和 format_name 决定：

```text
短答/随机值/异值同格式 -> value
完整句/解释 -> prose
JSON -> json
Value: xxx -> label
- xxx -> list
yes/no -> yesno
```

每条路线由一组简单 token / phrase 的首 token 构成。对每个 L17-L22 层段和 final_norm_input 计算：

```text
route_margin =
score(target_route) - max(score(non_target_routes))
```

候选 protocol gate 使用：

```text
protocol_min_margin
protocol_max_margin
protocol_last_margin
protocol_negative_count
protocol_first_negative_frac
protocol_final_norm_input_margin
protocol_final_norm_input_rank
protocol_late_shift
```

near-readout route margin 只作为参考线。

脚本同时内置 holdout validation：

```text
按 case_id 交替切分 train/test；
只在 train 上选 protocol gate；
再在 test 上评估同一个 gate。
```

### 跨模型总体结果

```text
qwen3:
  cases = 342
  top1_rate = 0.933
  failures = 23
  best protocol gate = protocol_min_margin_gt_0
  score = 0.783
  capture = 0.783
  false_pos = 0.000
  holdout gate = protocol_min_margin_gt_0
  holdout score = 1.000
  holdout capture = 1.000
  holdout false_pos = 0.000
  reference holdout score = 0.821

GLM4:
  cases = 342
  top1_rate = 0.892
  failures = 37
  best protocol gate = protocol_late_shift_lt_16.94
  score = 0.275
  capture = 1.000
  false_pos = 0.725
  holdout gate = protocol_min_margin_gt_-0.1787
  holdout score = 0.201
  holdout capture = 0.500
  holdout false_pos = 0.299
  reference holdout score = 0.643

DS7B:
  cases = 342
  top1_rate = 0.395
  failures = 207
  best protocol gate = protocol_final_norm_input_margin_gt_12.5
  score = 0.394
  capture = 0.720
  false_pos = 0.326
  holdout gate = protocol_min_margin_gt_-8.531
  holdout score = 0.020
  holdout capture = 0.640
  holdout false_pos = 0.620
  reference holdout score = 0.930
```

### 分族路线结果

DS7B 失败样本的错误路线高度集中：

```text
different_value_same_format:
  target_route = value
  failure_best_other_route = prose: 46

same_format_random_value:
  target_route = value
  failure_best_other_route = prose: 63

same_prefix_different_continuation:
  target_route = value
  failure_best_other_route = prose: 23

same_value_different_format:
  failure_best_other_route = prose: 45, json: 5
```

这说明 DS7B 的很多失败不是随机路线错，而是系统性偏向 prose/explanation route。

GLM4 失败更偏向 continuation：

```text
different_value_same_format:
  failure_best_other_route = continuation: 16

factor_isolation:
  failure_best_other_route = continuation: 3, prose: 8

same_value_different_format:
  failure_best_other_route = continuation: 8, value: 2
```

qwen3 的失败主要在 factor_isolation：

```text
factor_isolation:
  failure_best_other_route = prose: 18

same_format_random_value:
  failure_best_other_route = continuation: 4, prose: 1
```

### 分族 holdout

DS7B：

```text
overall:
  protocol_min_margin_gt_-8.531
  test_score = 0.020
  capture = 0.640
  false_pos = 0.620

factor_isolation:
  protocol_late_shift_gt_-0.6875
  test_score = 0.566
  capture = 0.923
  false_pos = 0.357

same_value_different_format:
  protocol_min_margin_gt_-6.188
  test_score = -0.521
  capture = 0.000
  false_pos = 0.521
```

GLM4：

```text
overall:
  test_score = 0.201
  capture = 0.500
  false_pos = 0.299

factor_isolation:
  test_score = 0.524
  capture = 0.667
  false_pos = 0.143
```

qwen3：

```text
overall:
  protocol_min_margin_gt_0
  test_score = 1.000
  capture = 1.000
  false_pos = 0.000

factor_isolation:
  protocol_min_margin_gt_-0.3984
  test_score = 1.000
  capture = 1.000
  false_pos = 0.000
```

### 当前判断

Phase 682 是一个“路线级正结果 + 自然门负结果”的阶段。

正结果：

```text
1. route-level protocol signal 是可测的；
2. qwen3 中存在非常干净的 protocol-level failure monitor；
3. DS7B 的失败路线高度集中到 prose/explanation route；
4. route-level taxonomy 比 expected-vs-competitor gap 更接近语法/协议机制。
```

负结果：

```text
1. DS7B 这个关键失败模型的 overall protocol gate 留出失败；
2. GLM4 只有弱到中等 protocol signal；
3. DS7B 的 protocol-level 信号只有 family-local 局部效果；
4. 不能说 G_protocol 已经跨模型闭合。
```

最谨慎结论：

```text
G_protocol 在 qwen3 上强成立；
在 GLM4 上弱成立；
在 DS7B 上只表现为失败路线集中，而不是稳定可用的自然门控。
```

### 对附件判断的更新

附件提出：

```text
如果 Phase 682 成功，说明自然 gate 可能不是值词元预警门，而是协议路径分叉门。
```

现在需要收紧为：

```text
协议路径分叉信号确实存在；
但在 DS7B 中还没有形成稳定可泛化的选择门。
```

也就是说，Phase 682 成功定位了“失败偏向哪条路线”，但没有完成“自然门何时触发”的闭合。

### 理论进展

失败门三层继续保留，但状态更新：

```text
G_readout:
  已验证，最稳定。

G_pre:
  局部成立，qwen3 强，GLM4 中等，DS7B 不稳定。

G_protocol:
  路线偏置可测；
  qwen3 强；
  DS7B 失败路线集中到 prose route；
  但 DS7B overall gate 未闭合。
```

路线级公式：

```text
S_r(h_l) =
\max_{v \in \mathcal{V}_r}
W_U(v)^\top h_l
```

```text
M_{\text{route}}(h_l) =
S_{r^\*}(h_l)
-
\max_{r \ne r^\*} S_r(h_l)
```

其中：

```text
r^\*: 当前任务要求的目标协议路线
\mathcal{V}_r: 路线 r 的词元集合
```

当前实证显示：

```text
M_route 可以解释一部分失败路线，
但不能在 DS7B 中稳定作为 G_protocol。
```

### 硬伤

```text
1. route token set 仍是人工构造的粗粒度集合；
2. route score 仍通过 lm_head projection 读取，不等同模型真实中间层决策；
3. qwen3 的强结果可能来自失败样本少；
4. DS7B 的关键 overall protocol gate 失败；
5. 协议路线解释了“偏向 prose”，但没有找到“为什么偏向 prose”的更底层状态；
6. 当前仍是小模型，结构偏差可能很大。
```

### 是否继续

Phase 682 已完成附件提出的 protocol-level route monitor 审计目标。

下一步不应继续扩大 route token set 或阈值搜索，因为这会变成盲目枚举。新的阶段性任务应该是：

```text
Phase 683: Prose-Route Bias Source Decomposition
```

核心问题：

```text
DS7B 的失败几乎总是被 prose/explanation route 截走；
那么 prose route bias 来自哪里？
```

推荐方向：

```text
1. 区分 instruction text 中的 explanation prior 与模型自身 prose prior；
2. 对比 Answer with only the value / one sentence / short explanation；
3. 测 L17-L22 中 prose route score 的写入来源；
4. 对 prose route 进行 remove/restore，观察是否能释放 value route；
5. 不再寻找统一 gate，而是先定位 DS7B prose bias 的来源。
```

## Phase 683: Prose-Route Bias Source Decomposition [2026-06-26 12:24]

### 任务来源

附件对 Phase 682 的判断基本正确：

```text
Phase 682 证明了 protocol route monitor 可以测到路线偏置；
qwen3 的 protocol gate 很强；
DS7B 的失败高度集中到 prose/explanation route；
但 DS7B 的 G_protocol 没有闭合。
```

这一判断需要继续收紧。关键问题不是继续扩大 route token set，也不是继续搜索统一阈值，而是定位：

```text
DS7B 的 prose route bias 到底来自 instruction wording、模型默认输出协议，还是 final readout 附近的放大机制。
```

### 生成脚本

```text
tests/gpt5/phase683_prose_route_bias_source_decomposition.py
```

脚本规模：

```text
495 lines
```

### 测试命令

语法校验：

```bash
python -m py_compile tests/gpt5/phase683_prose_route_bias_source_decomposition.py
```

依次测试三个模型，均使用 CUDA，并添加 hard exit：

```bash
python tests/gpt5/phase683_prose_route_bias_source_decomposition.py --model qwen3 --hard-exit-after-model > results/glm5_phase683_prose_route_bias_source_decomposition/phase683_qwen3_run.log 2>&1
python tests/gpt5/phase683_prose_route_bias_source_decomposition.py --model glm4 --hard-exit-after-model > results/glm5_phase683_prose_route_bias_source_decomposition/phase683_glm4_run.log 2>&1
python tests/gpt5/phase683_prose_route_bias_source_decomposition.py --model deepseek7b --hard-exit-after-model > results/glm5_phase683_prose_route_bias_source_decomposition/phase683_deepseek7b_run.log 2>&1
python tests/gpt5/phase683_prose_route_bias_source_decomposition.py --summarize-only
```

### 输出文件

```text
results/glm5_phase683_prose_route_bias_source_decomposition/phase683_qwen3_prose_bias_rows.jsonl
results/glm5_phase683_prose_route_bias_source_decomposition/phase683_glm4_prose_bias_rows.jsonl
results/glm5_phase683_prose_route_bias_source_decomposition/phase683_deepseek7b_prose_bias_rows.jsonl
results/glm5_phase683_prose_route_bias_source_decomposition/phase683_cross_model_summary.md
```

每个模型：

```text
144 base cases
7 protocol variants
1008 rows
```

协议变体：

```text
short_only
terse_no_explain
bare_answer
sentence
explanation
json
label
```

### 测试原理

固定同一批 value query case，只改变输出协议指令，观察 correct value token 是否被 prose route 截走。

核心测量量：

```text
PMV_l =
S_prose(h_l) - S_value(h_l)
```

其中：

```text
S_r(h_l) =
\max_{v \in \mathcal{V}_r}
W_U(v)^\top h_l
```

如果：

```text
PMV_l > 0
```

说明当前位置更偏向 prose route。

如果：

```text
PMV_l < 0
```

说明当前位置更偏向 value route。

本阶段重点比较：

```text
protocol_pmv:
  L17-L22 的平均 prose_minus_value

final_norm_input_pmv:
  final norm 输入处的 prose_minus_value

final_pmv:
  lm_head final logits 上的 prose_minus_value
```

这个设计的目的，是区分：

```text
早期协议层已经写入 prose bias
```

和：

```text
final/readout 附近才把 prose bias 放大
```

### 核心结果

跨模型总表：

```text
model       rows   value_top1   value_final_pmv   short_top1   short_final_pmv   terse_final_pmv   bare_final_pmv
deepseek7b  1008   0.368        1.429             0.083        4.677             -0.930            0.540
glm4        1008   0.597       -2.075             0.889       -5.917             -4.135            3.828
qwen3       1008   0.650       -2.923             0.965       -6.504             -7.421            5.157
```

DS7B 的关键细节：

```text
short_only:
  top1 = 0.083
  protocol_pmv = -4.651
  final_norm_input_pmv = -51.511
  final_pmv = 4.677
  failure_best_other = prose:132

terse_no_explain:
  top1 = 0.583
  protocol_pmv = -5.263
  final_norm_input_pmv = -86.988
  final_pmv = -0.930
  failure_best_other = prose:60

bare_answer:
  top1 = 0.438
  protocol_pmv = -2.034
  final_norm_input_pmv = -92.613
  final_pmv = 0.540
  failure_best_other = prose:81
```

qwen3 的关键细节：

```text
short_only:
  top1 = 0.965
  final_pmv = -6.504

terse_no_explain:
  top1 = 0.986
  final_pmv = -7.421

bare_answer:
  top1 = 0.000
  final_pmv = 5.157
  failure_best_other = prose:144
```

GLM4 的关键细节：

```text
short_only:
  top1 = 0.889
  final_pmv = -5.917

terse_no_explain:
  top1 = 0.854
  final_pmv = -4.135

bare_answer:
  top1 = 0.049
  final_pmv = 3.828
  failure_best_other = prose:137
```

### 客观进展

第一，附件中关于 Phase 682 的主判断正确：

```text
DS7B 的错误确实高度集中到 prose/explanation route。
```

第二，Phase 683 修正了一个更细的机制判断：

```text
DS7B 的 prose bias 不是在 L17-L22 或 final_norm_input 已经稳定压倒 value route。
```

因为在 DS7B 的 short_only 条件下：

```text
protocol_pmv = -4.651
final_norm_input_pmv = -51.511
```

这两个位置都更偏向 value，而不是 prose。

但是 final logits 上：

```text
final_pmv = 4.677
```

说明 prose route 在 final/readout 附近被强烈放大。

第三，instruction wording 不是无关因素。

DS7B 从 short_only 到 terse_no_explain：

```text
top1: 0.083 -> 0.583
final_pmv: 4.677 -> -0.930
prose failures: 132 -> 60
```

这说明明确禁止 explanation 可以释放一部分 value route。

第四，bare_answer 不是纯净 value baseline。

qwen3 和 GLM4 在 bare_answer 下几乎都转向 prose：

```text
qwen3 bare_answer top1 = 0.000
GLM4 bare_answer top1 = 0.049
```

因此缺少显式协议时，模型会默认补全自然语言回答，而不是输出孤立值。

### 理论进展

当前拼图从：

```text
DS7B prose route bias 存在
```

推进到：

```text
DS7B prose route bias 具有 late amplification 特征。
```

更准确的机制链条是：

```text
residual semantic state
  -> protocol / route state
  -> final readout amplification
  -> output token competition
```

本阶段支持：

```text
G_protocol 不是 DS7B 的稳定统一门；
DS7B 的失败更像 final readout 对 prose/default answer manifold 的偏置放大。
```

### 问题和硬伤

```text
1. route token set 仍是人工构造，不能等同真实内部 route basis；
2. PMV 是 lm_head projection 读数，不等同真实 causal state；
3. Phase 683 仍是观测性分解，不是 remove/restore 因果证明；
4. DS7B 是小模型，内部结构可能和大模型有偏差；
5. final_pmv 的变化可能混合了 format prior、token frequency、instruction following 和 answer prior；
6. json/label 的 route margin 受 token set 粗糙影响，不能过度解释。
```

### 当前结论

Phase 683 证明：

```text
DS7B 的 prose failure 不是简单的早期协议路线错误；
更强证据指向 final/readout 附近的 prose/default-answer 放大。
```

同时：

```text
explicit no-explanation instruction 可以显著降低 DS7B 的 prose bias，
但不能完全消除。
```

所以附件提出的 Phase 683 方向正确，但需要把结论收紧为：

```text
已经定位到 prose bias 的主要表现位置和 instruction sensitivity；
还没有完成 causal source decomposition。
```

### 下一阶段任务

下一阶段仍属于同一个大阶段：

```text
Phase 684: Late Readout Prose Amplification Causal Audit
```

目标不是继续做统一 gate，而是验证：

```text
如果在 final_norm_output / lm_head 前后抑制 prose route 或增强 value route，
DS7B 的 short_only failure 是否能被释放。
```

最低要求：

```text
1. 只在 DS7B 的 short_only failure cases 上做 causal intervention；
2. 对比 terse_no_explain success cases；
3. 分别测试 final_norm_input、final_norm_output、logit-level route projection；
4. 区分 remove prose、add value、remove prose + add value；
5. 不做跨模型大结论，只定位 DS7B 的 late amplification 是否可因果改变。
```

如果 Phase 684 成功，说明：

```text
DS7B 的 failure 是 late readout amplification 可干预问题。
```

如果 Phase 684 失败，说明：

```text
prose bias 不是简单线性 route direction，
需要进入更完整的 graph atlas / activation subspace 分解。
```

## Phase 684: Late Readout Prose Amplification Causal Audit [2026-06-26 12:28]

### 任务来源

Phase 683 发现：

```text
DS7B short_only failure 在 L17-L22 和 final_norm_input 处并不是 prose-dominant；
但到 final logits 时 prose_minus_value 变成强正值。
```

因此 Phase 684 不再继续做观测性 route monitor，而是进行读出端因果审计：

```text
如果在 final_norm_output / final_logits 附近抑制 prose 或增强 value，
原本失败的 correct value token 能否恢复到 top1。
```

### 生成脚本

```text
tests/gpt5/phase684_late_readout_prose_amplification_causal_audit.py
```

脚本规模：

```text
410 lines
```

### 测试命令

语法校验：

```bash
python -m py_compile tests/gpt5/phase684_late_readout_prose_amplification_causal_audit.py
```

依次运行三个模型，均添加 hard exit：

```bash
python tests/gpt5/phase684_late_readout_prose_amplification_causal_audit.py --model qwen3 --hard-exit-after-model > results/glm5_phase684_late_readout_prose_amplification_causal_audit/phase684_qwen3_run.log 2>&1
python tests/gpt5/phase684_late_readout_prose_amplification_causal_audit.py --model glm4 --hard-exit-after-model > results/glm5_phase684_late_readout_prose_amplification_causal_audit/phase684_glm4_run.log 2>&1
python tests/gpt5/phase684_late_readout_prose_amplification_causal_audit.py --model deepseek7b --hard-exit-after-model > results/glm5_phase684_late_readout_prose_amplification_causal_audit/phase684_deepseek7b_run.log 2>&1
python tests/gpt5/phase684_late_readout_prose_amplification_causal_audit.py --summarize-only
```

### 输出文件

```text
results/glm5_phase684_late_readout_prose_amplification_causal_audit/phase684_qwen3_late_readout_rows.jsonl
results/glm5_phase684_late_readout_prose_amplification_causal_audit/phase684_glm4_late_readout_rows.jsonl
results/glm5_phase684_late_readout_prose_amplification_causal_audit/phase684_deepseek7b_late_readout_rows.jsonl
results/glm5_phase684_late_readout_prose_amplification_causal_audit/phase684_cross_model_summary.md
```

样本数量：

```text
qwen3 short_only failures: 5
GLM4 short_only failures: 16
DS7B short_only failures: 132
```

DS7B 行数：

```text
1980 intervention rows
```

### 测试原理

只取 Phase 683 中 short_only 且 expected_top1=false 的失败样本。

测试三类干预：

```text
logit remove_prose:
  只降低 prose route token logits

logit add_value:
  只提高 expected/value token logits

logit remove_prose_add_value:
  同时降低 prose route 并提高 value route
```

以及隐藏状态读出方向干预：

```text
d =
\operatorname{mean}(W_U[value])
-
\operatorname{mean}(W_U[prose])
```

在：

```text
final_norm_input
final_norm_output
```

上添加：

```text
h' = h + \rho \cdot \lVert h \rVert \cdot \frac{d}{\lVert d \rVert}
```

其中：

```text
\rho \in {0.02, 0.05, 0.10}
```

核心观察：

```text
repair_rate:
  baseline 失败样本被修复为 expected_top1 的比例

mean_rank_delta:
  baseline expected_rank - patched expected_rank

patched_pmv:
  patched 后 prose_minus_value
```

### 核心结果

跨模型最佳条件：

```text
model       failures  best_condition                                      repair_rate  patched_top1  rank_delta  patched_pmv
deepseek7b  132       logit add_value alpha=2.0                           0.992        0.992         301.64      -2.400
glm4        16        hidden final_norm_input add_value_minus_prose r=0.1  1.000        1.000         1.56        -14.148
qwen3       5         hidden final_norm_output add_value_minus_prose r=0.1 1.000        1.000         7.00        -17.363
```

DS7B 关键结果：

```text
baseline:
  failures = 132
  baseline_top1 = 0.000
  baseline_mean_rank = 302.64
  baseline_pmv = 5.192
  baseline_best_other_route = prose:132

logit add_value alpha=2.0:
  repair_rate = 0.992
  patched_top1 = 0.992
  mean_patched_rank = 1.01
  patched_pmv = -2.400

logit remove_prose_add_value alpha=2.0:
  repair_rate = 0.992
  patched_top1 = 0.992
  mean_patched_rank = 1.01
  patched_pmv = -12.785

hidden final_norm_output add_value_minus_prose r=0.10:
  repair_rate = 0.917
  patched_top1 = 0.917
  mean_patched_rank = 1.20
  patched_pmv = -9.990

hidden final_norm_input add_value_minus_prose r=0.10:
  repair_rate = 0.909
  patched_top1 = 0.909
  mean_patched_rank = 1.27
  patched_pmv = -9.830

logit remove_prose alpha=2.0:
  repair_rate = 0.038
  mean_patched_rank = 535.01
  patched_pmv = -5.195
```

### 最重要的客观现象

第一，DS7B 的失败可以在读出端被大幅修复。

```text
hidden final_norm_output add_value_minus_prose r=0.10:
  132 个失败样本中约 91.7% 修复。
```

这说明 Phase 683 定位的 late readout amplification 不是纯观察假象，而是可以被读出方向干预改变。

第二，单纯 remove prose 几乎无效，并且会让 rank 变差。

```text
remove_prose alpha=2.0:
  repair_rate = 0.038
  mean_patched_rank = 535.01
```

这非常关键。它说明 DS7B 的错误不是简单的：

```text
prose 太强，所以压掉 prose 就能恢复 value
```

更准确是：

```text
correct value token 自身读出不足；
只压掉 prose 后，竞争可能转移到 continuation 等其它路线，
并不会自动释放 correct value。
```

第三，add_value 比 remove_prose 更接近关键因果因素。

```text
add_value alpha=2.0:
  repair_rate = 0.992
```

这说明瓶颈更像：

```text
value readout activation insufficient
```

而不是单纯：

```text
prose suppression failure
```

第四，hidden direction 也有效，但需要较大比例。

```text
r=0.02:
  DS7B repair_rate ~= 0.26-0.27

r=0.05:
  DS7B repair_rate ~= 0.57

r=0.10:
  DS7B repair_rate ~= 0.91-0.92
```

这说明 final readout space 中确实存在可操作的 value-minus-prose 方向，但这个方向是否是自然机制仍未证明。

### 理论进展

Phase 684 把 Phase 683 的判断从：

```text
prose bias 在 final/readout 处放大
```

推进为：

```text
final/readout 处存在可干预的 value-minus-prose 读出方向；
但关键不是单独压 prose，而是增强 correct value 的读出。
```

当前更准确的机制链条：

```text
residual semantic state
  -> protocol route state
  -> final readout field
  -> value activation / prose activation / continuation activation competition
  -> token selection
```

旧假设：

```text
失败 = prose route 太强
```

需要修正为：

```text
失败 = correct value readout 不足 + prose/default-answer/continuation 竞争未被压制。
```

### 问题和硬伤

```text
1. Phase 684 的 add_value 使用了 expected/value token 的 lm_head direction，带有目标答案信息；
2. 这证明读出端可修复，不证明模型自然机制真的使用同一方向；
3. logit-level add_value 是强人工干预，不能等同内部 causal circuit；
4. hidden r=0.10 可能偏强，需要后续做自然性审计；
5. remove_prose 失败说明 route token set 仍然不完整，continuation 竞争也必须进入模型；
6. qwen3 和 GLM4 失败样本太少，只能作为旁证；
7. 当前模型是小模型，结构偏差仍必须保留。
```

### 当前结论

Phase 684 支持：

```text
DS7B short_only failure 的核心瓶颈在 final readout 端；
但不是单纯 prose route suppression 问题，
而是 correct value readout activation 不足。
```

因此，Phase 682-684 的连续进展可以压缩成：

```text
Phase 682:
  失败路线集中到 prose。

Phase 683:
  prose bias 在 final/readout 附近被放大。

Phase 684:
  读出端 value-minus-prose 方向可以大幅修复失败；
  但单纯 remove prose 不够，必须增强 value。
```

### 下一阶段任务

下一阶段仍属于当前阶段性目标，但应该从“读出端能不能修”转向“自然机制是否真的写入 value 方向”：

```text
Phase 685: Natural Value-Readout Writer Localization
```

目标：

```text
找到哪些 layer/component 在自然成功样本中写入 value readout direction，
并解释 DS7B short_only failure 中为什么没有足够写入。
```

最低测试要求：

```text
1. 对比 DS7B short_only failure 与 terse_no_explain success；
2. 以 Phase 684 的 value-minus-prose direction 作为 readout probe；
3. 分层测 layer_out、attn_out、mlp_out 对该方向的增量贡献；
4. 不做 PCA、不做复杂统计，只记录基础投影增量和 rank 变化；
5. 如果找到 writer，再做 remove/restore；
6. 同时记录 continuation competitor，避免只看 prose。
```

阶段性目标：

```text
从“读出端人工可修复”
推进到
“自然网络中哪个组件负责写入可修复方向”。
```

## Phase 685: Natural Value-Readout Writer Localization [2026-06-26 12:32]

### 任务来源

Phase 684 证明：

```text
DS7B short_only failure 可以被 final/readout 处的 value-minus-prose direction 大幅修复。
```

但 Phase 684 仍是人工干预。Phase 685 追问：

```text
自然成功样本中，到底哪些 layer/component 自然写入了这个 value-minus-prose direction？
```

### 生成脚本

```text
tests/gpt5/phase685_natural_value_readout_writer_localization.py
```

脚本规模：

```text
371 lines
```

### 测试命令

语法校验：

```bash
python -m py_compile tests/gpt5/phase685_natural_value_readout_writer_localization.py
```

依次运行三个模型，均添加 hard exit：

```bash
python tests/gpt5/phase685_natural_value_readout_writer_localization.py --model qwen3 --hard-exit-after-model > results/glm5_phase685_natural_value_readout_writer_localization/phase685_qwen3_run.log 2>&1
python tests/gpt5/phase685_natural_value_readout_writer_localization.py --model glm4 --hard-exit-after-model > results/glm5_phase685_natural_value_readout_writer_localization/phase685_glm4_run.log 2>&1
python tests/gpt5/phase685_natural_value_readout_writer_localization.py --model deepseek7b --hard-exit-after-model > results/glm5_phase685_natural_value_readout_writer_localization/phase685_deepseek7b_run.log 2>&1
python tests/gpt5/phase685_natural_value_readout_writer_localization.py --summarize-only
```

### 输出文件

```text
results/glm5_phase685_natural_value_readout_writer_localization/phase685_qwen3_writer_projection_rows.jsonl
results/glm5_phase685_natural_value_readout_writer_localization/phase685_glm4_writer_projection_rows.jsonl
results/glm5_phase685_natural_value_readout_writer_localization/phase685_deepseek7b_writer_projection_rows.jsonl
results/glm5_phase685_natural_value_readout_writer_localization/phase685_cross_model_summary.md
```

DS7B 投影行数：

```text
6048 rows
```

### 测试原理

只选同一个 case 中满足以下条件的配对：

```text
short_only:
  expected_top1 = false

terse_no_explain:
  expected_top1 = true
```

配对数：

```text
qwen3: 3
GLM4: 5
DS7B: 72
```

定义读出方向：

```text
d =
\operatorname{mean}(W_U[value])
-
\operatorname{mean}(W_U[prose])
```

对每个 layer/component 的最后位置输出计算：

```text
P_{l,c}(x) =
h_{l,c}(x)^\top
\frac{d}{\lVert d \rVert}
```

比较：

```text
\Delta P_{l,c}
=
P_{l,c}(terse)
-
P_{l,c}(short)
```

如果：

```text
\Delta P_{l,c} > 0
```

说明 terse 成功样本在该 layer/component 上比 short 失败样本更强地写入 value-minus-prose direction。

扫描组件：

```text
layer_out
attn_out
mlp_out
```

### 核心结果

跨模型摘要：

```text
model       paired_cases  short_rank  terse_rank  rank_delta  top_site       top_delta  top_positive_rate
deepseek7b  72            167.69      1.00        166.69      L27_layer_out  34.718     0.958
glm4        5             2.00        1.00        1.00        L38_layer_out  3.443      1.000
qwen3       3             2.00        1.00        1.00        L34_layer_out  8.932      1.000
```

DS7B top positive sites：

```text
L27 layer_out:
  mean_delta = 34.718
  positive_rate = 0.958
  short_proj = 25.644
  terse_proj = 60.362

L26 layer_out:
  mean_delta = 24.362
  positive_rate = 0.944
  short_proj = 37.825
  terse_proj = 62.187

L26 attn_out:
  mean_delta = 12.838
  positive_rate = 1.000
  short_proj = 10.975
  terse_proj = 23.813

L24 layer_out:
  mean_delta = 7.913
  positive_rate = 0.944

L25 layer_out:
  mean_delta = 7.285
  positive_rate = 0.903

L23 attn_out:
  mean_delta = 6.249
  positive_rate = 1.000

L23 layer_out:
  mean_delta = 5.511
  positive_rate = 0.931
```

DS7B component-level 平均：

```text
layer_out:
  mean_delta = 2.221
  positive_delta_rate = 0.270

attn_out:
  mean_delta = 0.898
  positive_delta_rate = 0.468

mlp_out:
  mean_delta = 0.342
  positive_delta_rate = 0.511
```

### 客观进展

第一，DS7B 的自然成功差异高度集中在晚层。

最强位置不是早层，也不是全层均匀分布，而是：

```text
L23-L27
```

其中最强：

```text
L26-L27 layer_out
L23/L26 attn_out
```

第二，attention 输出比 MLP 更像局部 writer 候选。

虽然 layer_out 的累计差异最大，但具体组件中：

```text
L26 attn_out:
  positive_rate = 1.000

L23 attn_out:
  positive_rate = 1.000
```

这说明 terse_no_explain 成功样本中，attention 可能负责把 value-readout 方向写入或搬运到最后位置。

第三，MLP 不是完全无关，但当前证据较弱。

```text
L27 mlp_out:
  mean_delta = 5.461
  positive_rate = 0.583

L26 mlp_out:
  mean_delta = 4.240
  positive_rate = 0.764
```

MLP 有贡献，但稳定性不如 L23/L26 attention。

第四，qwen3/GLM4 的旁证方向一致，但样本太少。

```text
qwen3 paired_cases = 3
GLM4 paired_cases = 5
```

不能作为强跨模型结论，只能说明晚层 layer_out 也出现类似现象。

### 理论进展

Phase 685 把 Phase 684 的人工读出方向推进为自然 writer 候选：

```text
人工可修复方向:
  final/readout value-minus-prose direction

自然写入候选:
  DS7B L23-L27 late layer_out / attn_out
```

当前机制链条进一步收紧为：

```text
instruction wording
  -> late attention / residual writer
  -> value-minus-prose readout direction
  -> final readout competition
  -> token selection
```

这比之前的：

```text
prose route bias
```

更精确，因为它指出：

```text
成功不是简单压 prose，
而是在晚层写入足够强的 value-readout direction。
```

### 问题和硬伤

```text
1. Phase 685 仍是观测性定位，不是因果 restore；
2. layer_out 是累计状态，不等于单一 writer；
3. L26/L23 attn_out 是候选，但还没有证明 patch 后能修复；
4. value-minus-prose direction 来自 lm_head token 集合，仍带有人造读出定义；
5. qwen3/GLM4 配对太少，不能宣称跨模型闭合；
6. 当前只看最后位置，未检查 attention 从哪里读取 value；
7. 当前模型为小模型，内部结构可能有偏差。
```

### 当前结论

Phase 685 支持：

```text
DS7B 的 terse_no_explain 成功样本在 L23-L27 晚层自然写入更强 value-minus-prose readout direction。
```

最可疑的 writer 候选是：

```text
L26 attn_out
L23 attn_out
L26-L27 layer_out
```

这不是机制闭合，但已经把搜索空间从全模型收缩到少数晚层组件。

### 下一阶段任务

下一阶段应进入因果确认：

```text
Phase 686: Late Attention Value-Readout Writer Restore
```

核心问题：

```text
把 terse_no_explain 成功样本中的 L23/L26 attention output 或 L26/L27 layer_out
restore 到 short_only 失败样本中，
是否能提高 correct value rank。
```

最低测试：

```text
1. 只在 DS7B 72 个 paired cases 上做；
2. patch sites:
   L23 attn_out
   L26 attn_out
   L26 layer_out
   L27 layer_out
   L23+L26 attn_out
   L26+L27 layer_out
3. 对比 random same-norm patch；
4. 记录 expected_rank、top1、prose_minus_value、continuation competitor；
5. 如果 restore 成功，再反向 ablate terse success 的同位置。
```

如果 Phase 686 成功：

```text
late attention/residual writer 是 value-readout direction 的因果来源候选。
```

如果失败：

```text
Phase 685 的投影差异只是伴随现象，
需要进入更细的 head-level 或 source-token graph atlas。
```

## Phase 686: Late Attention Value-Readout Writer Restore [2026-06-26 12:38]

### 任务来源

Phase 685 找到自然 writer 候选：

```text
DS7B:
  L26-L27 layer_out
  L23/L26 attn_out
```

但 Phase 685 仍是观测性投影差异。Phase 686 进行因果 restore：

```text
把 terse_no_explain 成功样本中的候选组件输出，
patch 到同一个 case 的 short_only 失败样本中，
观察 correct value rank 是否恢复。
```

### 生成脚本

```text
tests/gpt5/phase686_late_attention_value_readout_writer_restore.py
```

脚本规模：

```text
386 lines
```

### 测试命令

语法校验：

```bash
python -m py_compile tests/gpt5/phase686_late_attention_value_readout_writer_restore.py
```

依次运行三个模型，均添加 hard exit：

```bash
python tests/gpt5/phase686_late_attention_value_readout_writer_restore.py --model qwen3 --hard-exit-after-model > results/glm5_phase686_late_attention_value_readout_writer_restore/phase686_qwen3_run.log 2>&1
python tests/gpt5/phase686_late_attention_value_readout_writer_restore.py --model glm4 --hard-exit-after-model > results/glm5_phase686_late_attention_value_readout_writer_restore/phase686_glm4_run.log 2>&1
python tests/gpt5/phase686_late_attention_value_readout_writer_restore.py --model deepseek7b --hard-exit-after-model > results/glm5_phase686_late_attention_value_readout_writer_restore/phase686_deepseek7b_run.log 2>&1
python tests/gpt5/phase686_late_attention_value_readout_writer_restore.py --summarize-only
```

### 输出文件

```text
results/glm5_phase686_late_attention_value_readout_writer_restore/phase686_qwen3_writer_restore_rows.jsonl
results/glm5_phase686_late_attention_value_readout_writer_restore/phase686_glm4_writer_restore_rows.jsonl
results/glm5_phase686_late_attention_value_readout_writer_restore/phase686_deepseek7b_writer_restore_rows.jsonl
results/glm5_phase686_late_attention_value_readout_writer_restore/phase686_cross_model_summary.md
```

DS7B restore 行数：

```text
1512 rows
```

### 测试原理

从 Phase 685 自动读取每个模型的 top positive writer sites。

DS7B 候选：

```text
L26_attn_out
L23_attn_out
L27_layer_out
L26_layer_out
```

对每个 paired case：

```text
short_only:
  failure prompt

terse_no_explain:
  success prompt
```

先缓存：

```text
h_short(l,c)
h_terse(l,c)
```

然后对 short_only 前向过程做三种 patch：

```text
add_delta:
  h'_short = h_short + (h_terse - h_short)

replace:
  h'_short = h_terse

random_delta:
  h'_short = h_short + random_same_norm(h_terse - h_short)
```

其中 random_delta 是控制条件，用来排除“任意同范数扰动都能修复”的解释。

### 核心结果

跨模型最佳结果：

```text
model       pairs  best_condition             repair_rate  patched_top1  rank_delta  patched_pmv
deepseek7b  72     L27_layer_out add_delta    1.000        1.000         166.69      -1.884
glm4        5      L38_layer_out add_delta    1.000        1.000         1.00        -3.750
qwen3       3      L34_layer_out add_delta    1.000        1.000         1.00        -3.875
```

DS7B 关键条件：

```text
L27_layer_out add_delta:
  repair_rate = 1.000
  patched_top1 = 1.000
  mean_patched_rank = 1.00
  mean_rank_delta = 166.69
  patched_pmv = -1.884

L27_layer_out replace:
  repair_rate = 1.000
  patched_top1 = 1.000
  mean_patched_rank = 1.00
  patched_pmv = -1.886

L26_layer_out add_delta:
  repair_rate = 1.000
  patched_top1 = 1.000
  mean_patched_rank = 1.00
  patched_pmv = -1.975

L26_layer_out replace:
  repair_rate = 1.000
  patched_top1 = 1.000
  mean_patched_rank = 1.00
  patched_pmv = -1.972
```

DS7B attention-only 条件：

```text
top2_attn_out add_delta:
  repair_rate = 0.292
  mean_patched_rank = 5.14
  patched_pmv = 0.940

L26_attn_out add_delta:
  repair_rate = 0.069
  mean_patched_rank = 10.71
  patched_pmv = 1.849

L23_attn_out add_delta:
  repair_rate = 0.069
  mean_patched_rank = 44.33
  patched_pmv = 3.075
```

DS7B random_delta 控制：

```text
L27_layer_out random_delta:
  repair_rate = 0.097
  mean_patched_rank = 384.83
  patched_pmv = 3.471

L26_layer_out random_delta:
  repair_rate = 0.028
  mean_patched_rank = 358.79
  patched_pmv = 4.321

top2_attn_out random_delta:
  repair_rate = 0.000
  mean_patched_rank = 231.53
  patched_pmv = 4.397
```

### 客观进展

第一，Phase 685 的 layer_out 候选被因果确认。

```text
DS7B L26/L27 layer_out restore:
  72/72 paired failures 修复为 correct value top1。
```

这说明 Phase 685 的投影差异不是单纯伴随现象。

第二，random_same_norm 控制基本不能修复。

```text
L27 random_delta repair_rate = 0.097
L26 random_delta repair_rate = 0.028
```

因此有效的不是任意同范数扰动，而是 terse success 中特定方向和状态内容。

第三，attention-only patch 有部分效果，但不够闭合。

```text
top2_attn_out:
  rank 从 167.69 改善到 5.14，
  但 top1 repair_rate 只有 0.292。
```

这说明 attention 可能是上游搬运/写入来源之一，但最终闭合状态在 layer_out 中。

第四，当前最强因果候选不是单个 attention output，而是晚层 residual layer_out：

```text
L26 layer_out
L27 layer_out
```

### 理论进展

当前链条从 Phase 682 到 Phase 686 已形成较强闭环：

```text
Phase 682:
  failure route 偏向 prose。

Phase 683:
  prose/value 差异在 final/readout 处放大。

Phase 684:
  final readout 的 value-minus-prose direction 可以人工修复。

Phase 685:
  terse 成功样本在 L23-L27 自然写入更强 value-minus-prose direction。

Phase 686:
  restore L26/L27 layer_out 可以 100% 修复 DS7B paired failures。
```

更准确的机制图：

```text
instruction wording
  -> late writer state
  -> L26/L27 residual layer_out
  -> value-minus-prose readout direction
  -> final token competition
  -> correct value selection
```

这说明：

```text
正确 value token 的选择不是只由最终 lm_head 决定；
它依赖晚层 residual state 是否已经写入足够强的 value-readout support。
```

### 对附件内容的修正

附件中 Phase 682 的判断基本正确，但现在需要补充：

```text
DS7B 的 prose route bias 不是最终答案；
更深一层是 short_only prompt 没有在 L26/L27 形成足够的 value-support residual state。
```

所以当前最准确说法是：

```text
prose/default route 是失败表现；
L26/L27 value-support residual state 缺失是更靠近因果机制的原因。
```

### 问题和硬伤

```text
1. paired cases 只覆盖 short_fail -> terse_success，不覆盖所有失败类型；
2. layer_out patch 是强 restore，可能同时携带多种信息，不是纯 value direction；
3. 还没有拆出 L26/L27 layer_out 中哪些 head/MLP/source-token 负责；
4. attention-only patch 改善 rank 但未闭合，说明上游来源仍未定位；
5. qwen3/GLM4 paired 样本太少，不能作为强跨模型证明；
6. 当前模型为小模型，结构可能与大模型不同；
7. patch 使用同 case 的 terse success 作为 donor，仍需跨 case / cross value 测试判断抽象程度。
```

### 当前结论

Phase 686 给出本轮最强结论：

```text
DS7B short_only 的 correct value failure 可以通过 restore 同 case terse_no_explain 的 L26/L27 layer_out 完全修复。
```

因此当前阶段目标已经完成到：

```text
从 route-level failure
定位到
late residual value-support state 的因果缺失。
```

### 下一阶段任务

下一阶段不应继续做宏观 restore，而应进入更细图谱：

```text
Phase 687: L26/L27 Value-Support State Decomposition
```

核心问题：

```text
L26/L27 layer_out 中真正有效的是哪一部分？
```

最低方案：

```text
1. 将 L26/L27 layer_out restore 分解为 attn_out、mlp_out、residual carry；
2. 做 head-level attention output patch；
3. 做 source-token patch，定位是否来自 value token、instruction token、relation token；
4. 做 cross-case restore，判断是否是同 case 内容拷贝还是抽象 value-support state；
5. 做 random same-norm 和 unrelated terse donor 控制；
6. 不扩大理论，只记录哪些子图能修复 rank。
```

阶段性目标：

```text
把 L26/L27 residual layer_out 从“有效黑箱”
拆成可解释的 writer graph。
```

## Phase 687: L26/L27 Value-Support State Decomposition [2026-06-26 13:49]

### 任务来源

附件对 Phase 683-686 的判断基本正确：

```text
Phase 683-686 已经完成从 route-level prose bias 到 L26/L27 residual restore 的阶段性闭环。
```

但附件也指出，不能把：

```text
L26/L27 layer_out restore 100% 修复
```

直接解释为已经找到纯 value writer。因为 layer_out 是累计残差状态，可能混合：

```text
value support
instruction state
format state
position state
residual carry
continuation state
case-specific content
```

因此 Phase 687 的目标是继续拆解：

```text
L26/L27 layer_out 中真正有效的是哪一部分？
```

### 生成脚本

```text
tests/gpt5/phase687_l26_l27_value_support_state_decomposition.py
```

脚本规模：

```text
502 lines
```

### 测试命令

语法校验：

```bash
python -m py_compile tests/gpt5/phase687_l26_l27_value_support_state_decomposition.py
```

依次运行三个模型，均添加 hard exit：

```bash
python tests/gpt5/phase687_l26_l27_value_support_state_decomposition.py --model qwen3 --hard-exit-after-model > results/glm5_phase687_l26_l27_value_support_state_decomposition/phase687_qwen3_run.log 2>&1
python tests/gpt5/phase687_l26_l27_value_support_state_decomposition.py --model glm4 --hard-exit-after-model > results/glm5_phase687_l26_l27_value_support_state_decomposition/phase687_glm4_run.log 2>&1
python tests/gpt5/phase687_l26_l27_value_support_state_decomposition.py --model deepseek7b --hard-exit-after-model > results/glm5_phase687_l26_l27_value_support_state_decomposition/phase687_deepseek7b_run.log 2>&1
python tests/gpt5/phase687_l26_l27_value_support_state_decomposition.py --summarize-only
```

### 输出文件

```text
results/glm5_phase687_l26_l27_value_support_state_decomposition/phase687_qwen3_state_decomposition_rows.jsonl
results/glm5_phase687_l26_l27_value_support_state_decomposition/phase687_glm4_state_decomposition_rows.jsonl
results/glm5_phase687_l26_l27_value_support_state_decomposition/phase687_deepseek7b_state_decomposition_rows.jsonl
results/glm5_phase687_l26_l27_value_support_state_decomposition/phase687_cross_model_summary.md
```

DS7B 行数：

```text
2624 rows
```

### 测试原理

Phase 687 分两部分。

第一部分是 same-case component decomposition：

```text
在同一个 case 中，
把 terse_no_explain 成功样本的组件状态 patch 到 short_only 失败样本。
```

测试组件：

```text
layer_input
attn_out
mlp_out
layer_out
```

测试层：

```text
DS7B:
  L26, L27

GLM4:
  L38, L39

qwen3:
  L33, L34
```

第二部分是 cross-case donor control：

```text
使用其它 case 的 terse-short delta 或 terse state，
patch 到当前 short_only 失败样本。
```

donor 类型：

```text
same_value
same_relation_diff_value
same_family_diff_value
unrelated
```

这一步用于区分：

```text
抽象 value-support delta
```

和：

```text
同样本 / 同值 / 具体内容状态拷贝
```

### 核心结果

跨模型摘要：

```text
model       pairs  layers      best_component                         comp_repair  best_cross                                      cross_repair
deepseek7b  72     [26, 27]    L26_layer_input same_case_add_delta     1.000        same_value_replace L26_layer_out                1.000
glm4        5      [38, 39]    L38_layer_input same_case_add_delta     1.000        same_relation_diff_value_add_delta L38_layer_out 0.800
qwen3       3      [33, 34]    L33_layer_input same_case_add_delta     1.000        same_relation_diff_value_add_delta L33_layer_out 1.000
```

DS7B component decomposition：

```text
L26_layer_input add_delta:
  repair_rate = 1.000
  patched_rank = 1.00
  patched_pmv = -2.030

L26_layer_out add_delta:
  repair_rate = 1.000
  patched_rank = 1.00
  patched_pmv = -1.975

L27_layer_input add_delta:
  repair_rate = 1.000
  patched_rank = 1.00
  patched_pmv = -1.975

L27_layer_out add_delta:
  repair_rate = 1.000
  patched_rank = 1.00
  patched_pmv = -1.884

L26_attn_out add_delta:
  repair_rate = 0.069
  patched_rank = 10.71
  patched_pmv = 1.849

L26_mlp_out add_delta:
  repair_rate = 0.000
  patched_rank = 156.53
  patched_pmv = 4.827

L27_attn_out add_delta:
  repair_rate = 0.042
  patched_rank = 51.39
  patched_pmv = 3.275

L27_mlp_out add_delta:
  repair_rate = 0.014
  patched_rank = 164.44
  patched_pmv = 3.887
```

DS7B cross-donor control：

```text
same_value_replace L26_layer_out:
  n = 8
  repair_rate = 1.000
  patched_rank = 1.00
  patched_pmv = -1.180

same_value_add_delta L26_layer_out:
  n = 8
  repair_rate = 0.500
  patched_rank = 2.25
  patched_pmv = -0.641

same_family_diff_value_add_delta L26_layer_out:
  n = 72
  repair_rate = 0.361
  patched_rank = 14.83
  patched_pmv = -0.185

same_relation_diff_value_add_delta L26_layer_out:
  n = 72
  repair_rate = 0.097
  patched_rank = 165.31
  patched_pmv = 3.669

unrelated_add_delta L26_layer_out:
  n = 72
  repair_rate = 0.014
  patched_rank = 323.90
  patched_pmv = 4.359

unrelated_replace L26_layer_out:
  n = 72
  repair_rate = 0.000
  patched_rank = 1451.58
  patched_pmv = 10.079
```

### 客观进展

第一，关键状态已经在 L26 layer_input 出现。

```text
L26_layer_input restore:
  repair_rate = 1.000
```

这说明 L26/L27 layer_out 不是最早的有效位置。更准确地说：

```text
在进入 L26 时，value-support state 已经基本形成。
```

第二，单独 attn_out / mlp_out 不能解释 100% restore。

```text
L26_attn_out repair_rate = 0.069
L26_mlp_out repair_rate = 0.000
L27_attn_out repair_rate = 0.042
L27_mlp_out repair_rate = 0.014
```

因此 Phase 686 中的 layer_out 有效，不是因为单个 L26/L27 attn_out 或 mlp_out 自身闭合，而更像：

```text
residual carry / earlier accumulated state
```

第三，cross-case donor 结果强烈收紧抽象性。

same-value donor 只有 8 个可用样本，但：

```text
same_value_replace:
  repair_rate = 1.000
```

说明同值状态可迁移。

但 unrelated donor 基本失败，且强烈损伤：

```text
unrelated_replace L26:
  repair_rate = 0.000
  patched_rank = 1451.58
```

所以 L26/L27 state 不是一个通用 protocol delta。

第四，same_relation_diff_value 效果很弱。

```text
same_relation_diff_value_add_delta L26:
  repair_rate = 0.097
```

这说明 relation 或格式相似不足以替代 value-specific state。

### 理论进展

Phase 687 把 Phase 686 的结论从：

```text
L26/L27 layer_out restore 可以修复
```

推进到：

```text
有效状态在 L26 layer_input 已经存在；
L26/L27 attn_out / mlp_out 不是单独充分组件；
该状态具有很强 value-specific / case-specific 特征。
```

机制链条需要修正为：

```text
instruction wording
  -> pre-L26 accumulated residual state
  -> L26 layer_input value-support state
  -> L26/L27 residual propagation
  -> final value readout support
  -> token competition
```

这比 Phase 686 更保守，也更接近真实结构。

### 对附件内容的判断

附件说：

```text
L26/L27 layer_out 是有效黑箱，还需要 decomposition 和 cross-case donor。
```

这个判断正确。

Phase 687 进一步补充：

```text
有效黑箱的入口至少前移到 L26 layer_input；
同 case / same value 很强；
different value 和 unrelated donor 大多不能泛化。
```

因此不能把当前状态称为纯抽象 value-support direction。

更准确的名称是：

```text
value-specific late residual support state
```

### 问题和硬伤

```text
1. same_value donor 只有 8 个样本，不能过度解释；
2. L26_layer_input 有效说明来源更早，但 Phase 687 还没有定位更早来源；
3. layer_input 仍是累计状态，不是单一 writer；
4. cross-case replace 可能携带 donor token/content 干扰，因此 replace 的解释要比 add_delta 更谨慎；
5. qwen3/GLM4 paired 样本太少，不能做强跨模型结论；
6. 当前仍未做 necessity test；
7. 当前仍未做 source-token path 定位。
```

### 当前结论

Phase 687 支持：

```text
DS7B short_only failure 的可修复状态不是 L26/L27 内部新写入的单一 attn/MLP 输出，
而是在 L26 layer_input 已经形成或缺失的 late residual value-support state。
```

它还支持：

```text
该状态强烈依赖 value / case content，
不是一个可任意跨样本迁移的通用 protocol delta。
```

### 下一阶段任务

下一阶段仍属于同一个阶段目标，应做 necessity：

```text
Phase 688: L26 Input Necessity and Degradation Audit
```

核心问题：

```text
在 terse_no_explain 成功样本中，
如果把 L26_layer_input / L26_layer_out / L27_layer_input 替换回 short_only 失败状态，
是否会破坏 correct value top1？
```

最低方案：

```text
1. 对 DS7B 72 个 paired cases 做反向 patch；
2. target = terse_no_explain success；
3. donor = short_only failure；
4. sites:
   L26_layer_input
   L26_layer_out
   L27_layer_input
   L27_layer_out
   L26_attn_out
   L26_mlp_out
5. controls:
   random same-norm delta
   unrelated short donor
6. 记录 top1 drop、rank increase、PMV increase、best_other_route。
```

如果成功破坏 terse success：

```text
L26 input/state 不只是充分，而且接近必要。
```

如果不能破坏：

```text
restore 的充分性可能来自替换状态的强人工路径，
自然生成中还有冗余写入通道。
```

## Phase 688: L26 Input Necessity and Degradation Audit [2026-06-26 13:57]

### 任务来源

用户上传的 Phase 683-686 分析总体正确。它把当前链条概括为：

```text
Phase 683:
  DS7B short_only 失败表现为 prose route bias，
  但早期协议层和 final_norm_input 并不一定已经 prose-dominant。

Phase 684:
  readout 端 add_value / value-minus-prose 可以修复，
  remove_prose 单独无效。

Phase 685:
  terse_no_explain 成功样本在 L23-L27 自然写入更强 value-minus-prose projection。

Phase 686:
  同 case 的 L26/L27 layer_out restore 可以 100% 修复 DS7B paired failures。
```

附件中最重要的保守判断也正确：

```text
L26/L27 layer_out 不是纯 value direction；
同 case donor 可能携带样本内容；
attention-only 没有闭合；
qwen3 / GLM4 paired samples 太少；
当前都是小模型，不能直接外推大模型机制。
```

Phase 687 已经进一步证明：

```text
restore-effective state 在 L26 layer_input 已经存在；
L26/L27 attn_out 或 mlp_out 单独不充分；
状态强烈 value-specific / case-specific。
```

因此 Phase 688 继续同一个阶段目标，做 necessity / degradation audit：

```text
如果 terse_no_explain 成功样本的 L26/L27 状态被替换回 short_only 失败状态，
correct value top1 是否会被破坏？
```

### 测试脚本

```text
tests/gpt5/phase688_l26_input_necessity_degradation_audit.py
```

结果目录：

```text
results/glm5_phase688_l26_input_necessity_degradation_audit/
```

生成文件：

```text
phase688_qwen3_necessity_rows.jsonl
phase688_qwen3_necessity_summary.json
phase688_glm4_necessity_rows.jsonl
phase688_glm4_necessity_summary.json
phase688_deepseek7b_necessity_rows.jsonl
phase688_deepseek7b_necessity_summary.json
phase688_cross_model_summary.md
phase688_cross_model_summary.json
```

### 执行命令

语法检查：

```bash
python -m py_compile tests/gpt5/phase688_l26_input_necessity_degradation_audit.py
```

三模型严格按顺序执行，并且每个模型都使用 `--hard-exit-after-model`：

```bash
python tests/gpt5/phase688_l26_input_necessity_degradation_audit.py --model qwen3 --hard-exit-after-model > results/glm5_phase688_l26_input_necessity_degradation_audit/phase688_qwen3_run.log 2>&1

python tests/gpt5/phase688_l26_input_necessity_degradation_audit.py --model glm4 --hard-exit-after-model > results/glm5_phase688_l26_input_necessity_degradation_audit/phase688_glm4_run.log 2>&1

python tests/gpt5/phase688_l26_input_necessity_degradation_audit.py --model deepseek7b --hard-exit-after-model > results/glm5_phase688_l26_input_necessity_degradation_audit/phase688_deepseek7b_run.log 2>&1

python tests/gpt5/phase688_l26_input_necessity_degradation_audit.py --summarize-only
```

三模型均完成，没有脚本异常。

### 测试原理

Phase 686 / 687 的方向是正向 restore：

```text
short_only failure + terse_success_state -> repair
```

Phase 688 做反向破坏：

```text
terse_no_explain success + short_failure_state -> degradation
```

对每个 paired case：

```text
short_only:
  expected_top1 = false

terse_no_explain:
  expected_top1 = true
```

捕获两个 prompt 在目标层的状态：

```text
h_short(l,c)
h_terse(l,c)
```

然后在 terse_no_explain 的 forward 中 patch：

```text
same_case_replace_short:
  h'_terse(l,c) = h_short(l,c)

same_case_remove_delta:
  h'_terse(l,c) = h_terse(l,c) - (h_terse(l,c) - h_short(l,c))

random_same_norm_add:
  h'_terse(l,c) = h_terse(l,c) + random_same_norm(h_terse(l,c) - h_short(l,c))
```

记录：

```text
drop_rate:
  terse_top1 = true 且 patched_top1 = false 的比例

mean_patched_rank:
  patch 后 correct value token 的平均 rank

mean_rank_increase_from_terse:
  patch 后 rank 相对 terse baseline 增加多少

mean_pmv_increase_from_terse:
  patch 后 prose-minus-value 相对 terse baseline 增加多少

patched_best_other_route:
  patch 后压过 value 的主要 route
```

如果同 case replace_short 高 drop，而 random_same_norm_add 明显低 drop，说明破坏不是任意扰动造成，而是 short failure state 确实缺少成功路径所需的状态。

### 核心结果

#### DS7B

样本：

```text
paired cases = 72
layers = [26, 27]
```

同 case 反向替换：

```text
same_case_replace_short L26_layer_input:
  drop_rate = 0.958
  patched_top1_rate = 0.042
  mean_patched_rank = 168.11
  mean_pmv_increase_from_terse = 6.233
  patched_best_other_route = prose 72/72

same_case_replace_short L26_layer_out:
  drop_rate = 1.000
  patched_top1_rate = 0.000
  mean_patched_rank = 168.12
  mean_pmv_increase_from_terse = 6.207
  patched_best_other_route = prose 72/72

same_case_replace_short L27_layer_input:
  drop_rate = 1.000
  patched_top1_rate = 0.000
  mean_patched_rank = 168.12
  mean_pmv_increase_from_terse = 6.207
  patched_best_other_route = prose 72/72

same_case_replace_short L27_layer_out:
  drop_rate = 1.000
  patched_top1_rate = 0.000
  mean_patched_rank = 167.69
  mean_pmv_increase_from_terse = 6.165
  patched_best_other_route = prose 72/72
```

组件对照：

```text
same_case_replace_short L26_attn_out:
  drop_rate = 0.431
  patched_top1_rate = 0.569
  mean_patched_rank = 2.85

same_case_replace_short L27_attn_out:
  drop_rate = 0.194
  patched_top1_rate = 0.806
  mean_patched_rank = 1.57

same_case_replace_short L26_mlp_out:
  drop_rate = 0.042
  patched_top1_rate = 0.958
  mean_patched_rank = 1.24

same_case_replace_short L27_mlp_out:
  drop_rate = 0.236
  patched_top1_rate = 0.764
  mean_patched_rank = 1.54
```

随机同范数控制：

```text
random_same_norm_add L26_layer_input:
  drop_rate = 0.306
  patched_top1_rate = 0.694

random_same_norm_add L26_layer_out:
  drop_rate = 0.236
  patched_top1_rate = 0.764

random_same_norm_add L27_layer_input:
  drop_rate = 0.167
  patched_top1_rate = 0.833

random_same_norm_add L27_layer_out:
  drop_rate = 0.181
  patched_top1_rate = 0.819
```

这说明：

```text
L26/L27 residual state 的 same-case short replacement 会系统性破坏 terse success；
attention output 有中等破坏力；
MLP output 单独破坏力弱；
随机同范数扰动远弱于 short failure state 替换。
```

#### GLM4

样本很少：

```text
paired cases = 5
layers = [38, 39]
```

同 case 反向替换：

```text
same_case_replace_short L38_layer_input:
  drop_rate = 1.000
  patched_top1_rate = 0.000
  mean_patched_rank = 2.00
  patched_best_other_route = continuation 5/5

same_case_replace_short L38_layer_out:
  drop_rate = 1.000
  patched_top1_rate = 0.000
  mean_patched_rank = 2.00

same_case_replace_short L39_layer_input:
  drop_rate = 1.000
  patched_top1_rate = 0.000
  mean_patched_rank = 2.00

same_case_replace_short L39_layer_out:
  drop_rate = 1.000
  patched_top1_rate = 0.000
  mean_patched_rank = 2.00
```

但 GLM4 只有 5 个 paired cases，只能作为弱支持。

#### qwen3

样本更少：

```text
paired cases = 3
layers = [33, 34]
```

同 case 反向替换：

```text
same_case_replace_short L33_layer_input:
  drop_rate = 0.667
  patched_top1_rate = 0.333

same_case_replace_short L33_layer_out:
  drop_rate = 0.667
  patched_top1_rate = 0.333

same_case_replace_short L34_layer_input:
  drop_rate = 0.667
  patched_top1_rate = 0.333

same_case_replace_short L34_layer_out:
  drop_rate = 0.667
  patched_top1_rate = 0.333
```

qwen3 只有 3 个 paired cases，因此不能做强结论。

### 结果判断

Phase 688 是 Phase 686 / 687 的必要性闭环推进。

它支持：

```text
DS7B terse_no_explain 成功依赖 L26/L27 residual value-support state；
把该状态替换回 short_only failure state，会几乎完全破坏 correct value top1。
```

最强证据不是 cross donor，因为 cross donor replace 可能引入严重内容污染；最干净证据是同 case replace_short：

```text
L26_layer_input:
  drop_rate = 0.958

L26_layer_out:
  drop_rate = 1.000

L27_layer_input:
  drop_rate = 1.000

L27_layer_out:
  drop_rate = 1.000
```

这把 Phase 687 的：

```text
L26 layer_input 已经足以恢复 short failure
```

推进为：

```text
L26/L27 residual state 对 terse success 接近必要。
```

但是注意措辞必须保持保守：

```text
接近必要 != 绝对必要；
在当前 prompt pair 和 DS7B 小模型上接近必要；
不是证明所有语言机制都依赖同一状态。
```

### 与附件内容的综合判断

附件中“Phase 683-686 已完成路线偏置 -> 读出修复 -> 自然写入候选 -> 因果恢复闭环”的判断正确。

Phase 688 对附件的补充是：

```text
这个闭环不再只有 sufficient restore evidence，
还出现了 degradation / necessity evidence。
```

因此当前更准确的链条是：

```text
instruction wording
  -> pre-L26 accumulated residual state
  -> L26 layer_input value-support state
  -> L26/L27 residual propagation
  -> final value readout support
  -> value vs prose / continuation competition
```

其中：

```text
L26/L27 residual state:
  对 short failure 是 sufficient repair state；
  对 terse success 是 near-necessary support state。
```

### 理论进展

当前不是发现了一个“纯语义方向”，而是定位到一个更真实的结构：

```text
late residual value-support state
```

它有三个特征：

```text
1. 累计性：
   在 L26 layer_input 已经存在，不是 L26/L27 单层新写入。

2. 竞争性：
   缺失后 prose route 重新压过 value route。

3. 组合性：
   attn_out 有部分作用，但单独不闭合；
   mlp_out 单独作用弱；
   layer_input / layer_out 才接近闭合。
```

当前理论应从：

```text
寻找单个 writer / 单个 direction
```

进一步转向：

```text
追踪 residual state 的来源图谱和读出竞争图谱。
```

### 问题和硬伤

```text
1. L26_layer_input 仍是累计状态，Phase 688 不能说明它由哪些更早层写入；
2. layer_out / layer_input patch 是强干预，可能破坏多个变量，不是纯 value ablation；
3. cross donor replace 破坏力太强，解释价值低于 same-case replace_short；
4. random_same_norm_add 不是完美控制，因为随机扰动方向分布不等价于真实 failure state；
5. DS7B 结果最强，但仍是小模型结果；
6. qwen3 / GLM4 paired 样本分别只有 3 / 5，不能当作强跨模型证据；
7. 当前只测 first-token value readout，还没有测完整自然生成稳定性；
8. 当前没有 source-token path，也没有 head-level path；
9. 当前无法区分 value support、instruction state、format state、position state 在 residual state 中的比例。
```

### 阶段性结论

Phase 688 支持以下客观拼图：

```text
在 DS7B paired failures 中，
terse_no_explain 成功不是只靠最终读出端偶然偏移；
它依赖一个已经在 L26 layer_input 出现、并经 L26/L27 传播的 residual support state。
```

这个 state 被换回 short_only failure state 后：

```text
correct value token 从 top1 掉到平均 rank 约 168；
prose route 成为 72/72 个样本的主要竞争路线。
```

因此当前机制闭环从“充分性”推进到“接近必要性”：

```text
restore:
  short failure + terse state -> success

degradation:
  terse success + short state -> failure
```

### 接下来的阶段性任务

下一任务和当前任务属于同一阶段目标，应继续自动推进，不需要重新确认。

```text
Phase 689: Pre-L26 Source Path Localization
```

核心问题：

```text
既然 L26_layer_input 已经携带 near-necessary value-support state，
那么这个状态是从哪里写入 / 搬运 / 累积来的？
```

最低方案：

```text
1. 以 DS7B 72 个 paired cases 为主；
2. target site = L26_layer_input；
3. 扫描 L18-L25 的 layer_out / attn_out / mlp_out；
4. 做两类测试：
   a. restore:
      short_only failure 中 patch 更早层 terse-short delta，
      看能否恢复 L26_layer_input projection 和 final top1；
   b. degradation:
      terse_no_explain success 中把更早层状态替换回 short_only，
      看是否破坏 L26_layer_input 和 final top1；
5. 输出两个指标：
   upstream_to_L26_effect:
      patch 后 L26_layer_input 的 value-minus-prose projection 变化；
   final_readout_effect:
      patch 后 expected token rank / top1 / PMV 变化。
```

如果 Phase 689 成功：

```text
可以把 late residual state 从 L26 入口继续前溯到具体 source writer。
```

如果失败：

```text
说明 L26_layer_input state 可能是多层多通道小量累积，
下一步应从 graph atlas 的多边组合恢复，而不是单点 patch。
```

## Phase 689: Pre-L26 Source Path Localization [2026-06-26 14:07]

### 任务来源

Phase 688 已证明：

```text
DS7B terse_no_explain success 依赖 L26/L27 residual value-support state；
把 L26_layer_input / L26_layer_out / L27_layer_input / L27_layer_out 换回 short_only failure state，
会几乎完全破坏 correct value top1。
```

但是 Phase 688 仍然留下关键问题：

```text
L26_layer_input 已经携带 near-necessary state，
那么这个 state 是从哪里写入 / 搬运 / 累积来的？
```

Phase 689 继续同一阶段目标，扫描 L26 之前的 source path。

### 测试脚本

```text
tests/gpt5/phase689_pre_l26_source_path_localization.py
```

结果目录：

```text
results/glm5_phase689_pre_l26_source_path_localization/
```

生成文件：

```text
phase689_qwen3_source_path_rows.jsonl
phase689_qwen3_source_path_summary.json
phase689_glm4_source_path_rows.jsonl
phase689_glm4_source_path_summary.json
phase689_deepseek7b_source_path_rows.jsonl
phase689_deepseek7b_source_path_summary.json
phase689_cross_model_summary.md
phase689_cross_model_summary.json
```

### 执行命令

语法检查：

```bash
python -m py_compile tests/gpt5/phase689_pre_l26_source_path_localization.py
```

三模型严格按顺序执行，并且每个模型都使用 `--hard-exit-after-model`：

```bash
python tests/gpt5/phase689_pre_l26_source_path_localization.py --model qwen3 --hard-exit-after-model > results/glm5_phase689_pre_l26_source_path_localization/phase689_qwen3_run.log 2>&1

python tests/gpt5/phase689_pre_l26_source_path_localization.py --model glm4 --hard-exit-after-model > results/glm5_phase689_pre_l26_source_path_localization/phase689_glm4_run.log 2>&1

python tests/gpt5/phase689_pre_l26_source_path_localization.py --model deepseek7b --hard-exit-after-model > results/glm5_phase689_pre_l26_source_path_localization/phase689_deepseek7b_run.log 2>&1

python tests/gpt5/phase689_pre_l26_source_path_localization.py --summarize-only
```

三模型均完成。DS7B 因扫描 L18-L25 三类组件，运行时间明显长于 Phase 688，但没有异常。

### 测试原理

Phase 689 同时记录两个结果：

```text
1. upstream_to_target:
   patch 更早层后，L26_layer_input 的 value-minus-prose projection 是否被拉动。

2. final_readout:
   patch 更早层后，最终 expected value token 的 top1 / rank / PMV 是否改变。
```

对 DS7B：

```text
target_site = L26_layer_input
source_layers = L18-L25
components = layer_out / attn_out / mlp_out
```

对每个 paired case：

```text
short_only:
  expected_top1 = false

terse_no_explain:
  expected_top1 = true
```

正向 restore：

```text
short_only failure 中，在上游 site 添加：

h'_short(site) = h_short(site) + (h_terse(site) - h_short(site))
```

反向 degradation：

```text
terse_no_explain success 中，在上游 site 替换为：

h'_terse(site) = h_short(site)
```

同时在 patched forward 中捕获：

```text
L26_layer_input projection
final expected token rank
final prose-minus-value
```

### 核心结果

#### DS7B

样本：

```text
paired cases = 72
target = L26_layer_input
source_layers = L18-L25
```

最强 restore 结果：

```text
restore add_delta L24_layer_out:
  repair_rate = 1.000
  patched_top1_rate = 1.000
  mean_patched_rank = 1.00
  mean_target_effect = 8.105
  target_delta_fraction = 0.936
  mean_pmv_effect = 6.335

restore add_delta L25_layer_out:
  repair_rate = 1.000
  patched_top1_rate = 1.000
  mean_patched_rank = 1.00
  mean_target_effect = 7.285
  target_delta_fraction = 1.000
  mean_pmv_effect = 6.309

restore add_delta L23_layer_out:
  repair_rate = 0.986
  patched_top1_rate = 0.986
  mean_patched_rank = 1.01
  mean_target_effect = 8.195
  target_delta_fraction = 0.972

restore add_delta L22_layer_out:
  repair_rate = 0.986
  patched_top1_rate = 0.986
  mean_patched_rank = 1.01
  mean_target_effect = 7.810
  target_delta_fraction = 0.911

restore add_delta L21_layer_out:
  repair_rate = 0.986
  patched_top1_rate = 0.986
  mean_patched_rank = 1.01
  mean_target_effect = 7.307
  target_delta_fraction = 0.861

restore add_delta L20_layer_out:
  repair_rate = 0.972
  patched_top1_rate = 0.972
  mean_patched_rank = 1.06
  mean_target_effect = 7.385
  target_delta_fraction = 1.011

restore add_delta L19_layer_out:
  repair_rate = 0.889
  patched_top1_rate = 0.889
  mean_patched_rank = 1.31
  mean_target_effect = 6.892

restore add_delta L18_layer_out:
  repair_rate = 0.792
  patched_top1_rate = 0.792
  mean_patched_rank = 1.60
  mean_target_effect = 6.682
```

随机同范数 layer_out restore 控制很弱：

```text
random_same_norm L18_layer_out:
  repair_rate = 0.069

random_same_norm L19_layer_out:
  repair_rate = 0.028

random_same_norm L20_layer_out:
  repair_rate = 0.028

random_same_norm L21_layer_out:
  repair_rate = 0.069

random_same_norm L22_layer_out:
  repair_rate = 0.028

random_same_norm L23_layer_out:
  repair_rate = 0.000

random_same_norm L24_layer_out:
  repair_rate = 0.028

random_same_norm L25_layer_out:
  repair_rate = 0.014
```

这说明有效的不是任意扰动，而是 terse-short 的真实 residual trajectory delta。

组件级结果明显弱于 layer_out：

```text
restore add_delta L22_attn_out:
  repair_rate = 0.319
  mean_patched_rank = 9.54
  mean_target_effect = 1.917

restore add_delta L23_attn_out:
  repair_rate = 0.069
  mean_patched_rank = 44.33
  mean_target_effect = 5.786

restore add_delta L19_mlp_out:
  repair_rate = 0.125
  mean_patched_rank = 48.19
  mean_target_effect = 5.984

restore add_delta L18_mlp_out:
  repair_rate = 0.083
  mean_patched_rank = 47.46
```

反向 degradation 也支持同一结构：

```text
degradation replace_short L23_layer_out:
  drop_rate = 1.000
  patched_top1_rate = 0.000
  mean_patched_rank = 145.42
  mean_target_effect = 8.202
  best_other_route = prose 72/72

degradation replace_short L20_layer_out:
  drop_rate = 0.986
  patched_top1_rate = 0.014
  mean_patched_rank = 100.03
  mean_target_effect = 8.388

degradation replace_short L24_layer_out:
  drop_rate = 0.986
  patched_top1_rate = 0.014
  mean_patched_rank = 146.56
  mean_target_effect = 8.251

degradation replace_short L22_layer_out:
  drop_rate = 0.972
  patched_top1_rate = 0.028

degradation replace_short L21_layer_out:
  drop_rate = 0.972
  patched_top1_rate = 0.028

degradation replace_short L25_layer_out:
  drop_rate = 0.958
  patched_top1_rate = 0.042

degradation replace_short L19_layer_out:
  drop_rate = 0.931
  patched_top1_rate = 0.069

degradation replace_short L18_layer_out:
  drop_rate = 0.833
  patched_top1_rate = 0.167
```

#### GLM4

样本较少：

```text
paired cases = 5
target = L38_layer_input
source_layers = L30-L37
```

layer_out restore 全部有效：

```text
restore add_delta L30-L37 layer_out:
  repair_rate = 1.000
```

但 degradation 不如 DS7B 稳定：

```text
degradation replace_short L37_layer_out:
  drop_rate = 1.000

L33-L35 layer_out:
  drop_rate = 0.600

其它 layer_out:
  drop_rate = 0.400
```

GLM4 只有 5 个 paired cases，不能过度解释。

#### qwen3

样本更少：

```text
paired cases = 3
target = L33_layer_input
source_layers = L25-L32
```

layer_out restore 全部有效：

```text
restore add_delta L25-L32 layer_out:
  repair_rate = 1.000
```

degradation 中 L25-L27 layer_out 最强：

```text
degradation replace_short L25_layer_out:
  drop_rate = 1.000

degradation replace_short L26_layer_out:
  drop_rate = 1.000

degradation replace_short L27_layer_out:
  drop_rate = 1.000
```

但 qwen3 只有 3 个 paired cases，只能作为弱支持。

### 结果判断

Phase 689 是一个重要正结果，但它不是“找到单点 writer”，而是把机制从 L26 入口继续前移到一条残差轨迹：

```text
L18-L25 layer_out residual trajectory
  -> L26_layer_input value-support state
  -> L26/L27 residual propagation
  -> final value readout support
```

最关键的事实是：

```text
DS7B 中 L20-L25 任意 layer_out 的 terse-short delta，
都能强烈恢复 L26_layer_input 的 value-support projection，
并几乎完全恢复最终 correct value top1。
```

这说明 L26_layer_input 不是由 L25 单点突然写出，而是一个已经在更早层出现并沿 residual stream 连续携带的状态。

### 理论进展

当前理论应从：

```text
寻找 L26 writer
```

修正为：

```text
寻找 pre-L26 residual trajectory 的起点、分叉点和维持机制。
```

更准确的机制链条：

```text
instruction wording
  -> early/mid residual route bifurcation
  -> L18-L25 residual trajectory divergence
  -> L26_layer_input value-support state
  -> L26/L27 residual propagation
  -> final readout competition
```

其中：

```text
layer_out 是主闭合通道；
attn_out / mlp_out 是局部供料或调制通道；
单个 attn_out / mlp_out 通常不能闭合最终读出。
```

### 对附件与当前进展的综合修正

附件把 Phase 683-686 概括为：

```text
DS7B short_only 没有形成足够强的 L26/L27 value-support residual state。
```

这个判断仍正确，但 Phase 689 进一步修正为：

```text
L26/L27 不是状态起点；
状态差异至少可追溯到 L18-L25 的 residual trajectory。
```

因此当前不应说：

```text
L26/L27 写入了 value-support state。
```

更准确是：

```text
L26/L27 是 value-support state 的关键入口 / 放大 / 读出前承载区；
该状态的上游来源是一条 pre-L26 residual trajectory。
```

### 问题和硬伤

```text
1. L18-L25 layer_out 都有效，说明 patch 粒度仍太粗；
2. layer_out restore 可能只是 residual stream 的整体状态复制，不等于定位 writer；
3. attn_out / mlp_out 单点弱，不代表它们不重要，可能需要组合多边恢复；
4. Phase 689 没有做 head-level source-token path；
5. L18 已经有较强效果，说明还需要继续前溯；
6. 当前仍只看 final first-token readout，不等于完整自然生成；
7. qwen3 / GLM4 样本太少，只能作为方向性参考；
8. 当前模型都是小模型，内部结构可能存在偏差，不能直接外推大模型。
```

### 阶段性结论

Phase 689 支持：

```text
DS7B 的 short_only failure / terse_no_explain success 差异，
不是 L26 附近的单点写入差异，
而是从至少 L18 开始沿 residual stream 累积和携带的轨迹差异。
```

强结果：

```text
L20-L25 layer_out add_delta:
  几乎完全修复 short failure；
  同时恢复 L26_layer_input projection；
  随机同范数控制几乎无效。
```

因此当前拼图从：

```text
late residual value-support state
```

推进为：

```text
pre-L26 residual trajectory of value-support state
```

### 接下来的阶段性任务

下一任务仍属于同一阶段目标，应继续自动推进。

```text
Phase 690: Residual Trajectory Boundary and Multi-Edge Decomposition
```

核心问题：

```text
L18-L25 layer_out 都有效，是因为状态已经从更早层开始连续携带，
还是因为 layer_out patch 强复制了多个变量？
```

最低方案：

```text
1. 继续以 DS7B 72 paired cases 为主；
2. 扫描 L8-L18 的 layer_out；
3. 同时记录：
   a. final repair/drop；
   b. L18_layer_input projection；
   c. L26_layer_input projection；
4. 找到 trajectory divergence 的最早可见边界；
5. 对边界附近做组合 patch：
   attn_out + mlp_out + residual carry
6. 判断单点不足是否可以被多边组合闭合。
```

预期判断标准：

```text
如果 L8-L17 layer_out 也能强修复：
  说明路线分叉更早，下一步继续前溯到 prompt/instruction token path。

如果只有 L18 以后强修复：
  L18 附近可能是 value-support trajectory 的可见分叉边界。

如果组合 patch 才能闭合：
  当前机制应进入 graph atlas multi-edge decomposition，而不是单点定位。
```

## Phase 690: Residual Trajectory Boundary Scan [2026-06-26 14:14]

### 任务来源

Phase 689 证明：

```text
DS7B 的 L18-L25 layer_out 差分已经能强烈恢复 L26_layer_input value-support state，
并几乎完全修复 final value readout。
```

但 Phase 689 没有回答：

```text
这条 residual trajectory 是从 L18 才开始可见，
还是更早层已经存在？
```

Phase 690 因此继续前溯，目标是找到可见边界。

### 测试脚本

```text
tests/gpt5/phase690_residual_trajectory_boundary_scan.py
```

结果目录：

```text
results/glm5_phase690_residual_trajectory_boundary_scan/
```

生成文件：

```text
phase690_qwen3_boundary_rows.jsonl
phase690_qwen3_boundary_summary.json
phase690_glm4_boundary_rows.jsonl
phase690_glm4_boundary_summary.json
phase690_deepseek7b_boundary_rows.jsonl
phase690_deepseek7b_boundary_summary.json
phase690_cross_model_summary.md
phase690_cross_model_summary.json
```

### 执行命令

语法检查：

```bash
python -m py_compile tests/gpt5/phase690_residual_trajectory_boundary_scan.py
```

三模型严格按顺序执行，并且每个模型都使用 `--hard-exit-after-model`：

```bash
python tests/gpt5/phase690_residual_trajectory_boundary_scan.py --model qwen3 --hard-exit-after-model > results/glm5_phase690_residual_trajectory_boundary_scan/phase690_qwen3_run.log 2>&1

python tests/gpt5/phase690_residual_trajectory_boundary_scan.py --model glm4 --hard-exit-after-model > results/glm5_phase690_residual_trajectory_boundary_scan/phase690_glm4_run.log 2>&1

python tests/gpt5/phase690_residual_trajectory_boundary_scan.py --model deepseek7b --hard-exit-after-model > results/glm5_phase690_residual_trajectory_boundary_scan/phase690_deepseek7b_run.log 2>&1

python tests/gpt5/phase690_residual_trajectory_boundary_scan.py --summarize-only
```

三模型均完成，没有脚本异常。

### 测试原理

Phase 690 只做边界扫描，不再铺开全模型。

对 DS7B：

```text
early_target = L18_layer_input
final_target = L26_layer_input
layer_out_scan = L8-L18
boundary_components = L16-L18 attn_out / mlp_out
```

正向 restore：

```text
short_only failure 中，在更早 layer_out 添加：

h'_short(site) = h_short(site) + (h_terse(site) - h_short(site))
```

反向 degradation：

```text
terse_no_explain success 中，把更早 layer_out 替换为：

h'_terse(site) = h_short(site)
```

同时记录三个层级：

```text
1. early_target projection:
   L18_layer_input value-minus-prose projection

2. final_target projection:
   L26_layer_input value-minus-prose projection

3. final readout:
   expected token top1 / rank / prose-minus-value
```

### 核心结果

#### DS7B

样本：

```text
paired cases = 72
early_target = L18_layer_input
final_target = L26_layer_input
layer_out_scan = L8-L18
```

正向 restore 的 layer_out 边界：

```text
L8_layer_out:
  repair_rate = 0.083
  mean_patched_rank = 106.29
  final_target_effect = 0.604

L9_layer_out:
  repair_rate = 0.111
  mean_patched_rank = 89.44
  final_target_effect = 1.329

L10_layer_out:
  repair_rate = 0.056
  mean_patched_rank = 110.93
  final_target_effect = 0.923

L11_layer_out:
  repair_rate = 0.097
  mean_patched_rank = 88.04
  final_target_effect = 0.900

L12_layer_out:
  repair_rate = 0.097
  mean_patched_rank = 99.94
  final_target_effect = 2.071

L13_layer_out:
  repair_rate = 0.389
  mean_patched_rank = 13.99
  final_target_effect = 6.540

L14_layer_out:
  repair_rate = 0.431
  mean_patched_rank = 11.01
  final_target_effect = 5.901

L15_layer_out:
  repair_rate = 0.597
  mean_patched_rank = 5.90
  final_target_effect = 6.263

L16_layer_out:
  repair_rate = 0.486
  mean_patched_rank = 4.92
  final_target_effect = 4.871

L17_layer_out:
  repair_rate = 0.569
  mean_patched_rank = 3.69
  final_target_effect = 5.118

L18_layer_out:
  repair_rate = 0.792
  mean_patched_rank = 1.60
  final_target_effect = 6.682
```

反向 degradation 的 layer_out 边界：

```text
L8_layer_out:
  drop_rate = 0.056
  patched_top1_rate = 0.944

L9_layer_out:
  drop_rate = 0.194
  patched_top1_rate = 0.806

L10_layer_out:
  drop_rate = 0.250
  patched_top1_rate = 0.750

L11_layer_out:
  drop_rate = 0.361
  patched_top1_rate = 0.639

L12_layer_out:
  drop_rate = 0.347
  patched_top1_rate = 0.653

L13_layer_out:
  drop_rate = 0.875
  patched_top1_rate = 0.125
  mean_patched_rank = 25.18
  final_target_effect = 6.198

L14_layer_out:
  drop_rate = 0.847
  patched_top1_rate = 0.153
  mean_patched_rank = 16.57

L15_layer_out:
  drop_rate = 0.792
  patched_top1_rate = 0.208
  mean_patched_rank = 22.94

L16_layer_out:
  drop_rate = 0.750
  patched_top1_rate = 0.250

L17_layer_out:
  drop_rate = 0.681
  patched_top1_rate = 0.319

L18_layer_out:
  drop_rate = 0.833
  patched_top1_rate = 0.167
```

这给出非常清楚的边界现象：

```text
L8-L12:
  restore 弱，degradation 弱或中等；

L13-L18:
  restore 明显增强，degradation 明显增强；

L18:
  restore 最强，但不是唯一有效点。
```

#### GLM4

样本很少：

```text
paired cases = 5
early_target = L30_layer_input
final_target = L38_layer_input
layer_out_scan = L20-L30
```

GLM4 的 restore 非常宽：

```text
L20-L30 layer_out add_delta:
  repair_rate = 1.000
```

但 degradation 最强只有：

```text
L27-L29 layer_out:
  drop_rate = 0.600

L30_layer_out:
  drop_rate = 0.400
```

样本太少，不能做强跨模型判断。

#### qwen3

样本更少：

```text
paired cases = 3
early_target = L25_layer_input
final_target = L33_layer_input
layer_out_scan = L15-L25
```

qwen3 的边界大致在 L18 以后增强：

```text
L15_layer_out:
  repair_rate = 0.333

L16_layer_out:
  repair_rate = 0.333

L17_layer_out:
  repair_rate = 0.667

L18-L25 layer_out:
  repair_rate = 1.000
```

但 paired cases 只有 3，不能做强结论。

### 结果判断

Phase 690 是一个关键边界定位阶段。

它把 Phase 689 的：

```text
L18-L25 residual trajectory 有效
```

进一步收紧为：

```text
DS7B 可见强分叉边界大约在 L13-L18；
L8-L12 只有弱影响；
L13 起效果显著增强；
L18 之后进入强 residual carry 区间。
```

这说明当前路线不是：

```text
L26 单点写入
```

也不是：

```text
L18 突然写入
```

而更像：

```text
L13-L18 之间逐步形成可见 residual trajectory divergence，
随后 L18-L26 继续携带和放大。
```

### 理论进展

当前机制链条进一步修正为：

```text
instruction wording
  -> early route seed
  -> L13-L18 visible residual trajectory bifurcation
  -> L18-L25 residual carry / accumulation
  -> L26_layer_input value-support state
  -> L26/L27 residual propagation
  -> final readout competition
```

其中：

```text
L13-L18:
  可见分叉边界区

L18-L25:
  强残差轨迹携带区

L26/L27:
  近读出承载 / 放大 / 修复充分区
```

### 对当前理论的谨慎修正

之前说：

```text
pre-L26 residual trajectory
```

现在更具体：

```text
pre-L26 trajectory 的强可见边界不在 L8-L12，
而在 L13-L18 之间逐步出现。
```

但这仍不是最终机制闭合，因为：

```text
layer_out 是残差总状态；
它告诉我们轨迹在哪里可见，
还没有告诉我们哪条边、哪个 token、哪个 head / MLP 负责形成这个轨迹。
```

### 问题和硬伤

```text
1. L13-L18 是边界区，不是单点；
2. layer_out patch 仍然是粗粒度残差状态复制；
3. L16-L18 attn_out / mlp_out 单点仍不能解释全部 layer_out 效果；
4. 还缺少 source-token path；
5. 还缺少多边组合 patch；
6. 当前只测 first-token readout；
7. qwen3 / GLM4 样本太少；
8. 小模型结构可能偏向短答/散文路线不稳定，不能直接泛化到大模型。
```

### 阶段性结论

Phase 688-690 完成了当前阶段性目标：

```text
从 L26/L27 restore 黑箱，
推进到 L26 near-necessary state，
再推进到 pre-L26 residual trajectory，
最后定位到 DS7B 的 L13-L18 可见分叉边界。
```

当前客观拼图：

```text
DS7B short_only failure 和 terse_no_explain success 的关键差异，
不是 final readout 的孤立偶然事件；
也不是 L26 附近单点写入；
而是从 L13-L18 开始可见、经 L18-L25 携带、到 L26/L27 进入近读出承载区的 residual trajectory 差异。
```

### 接下来是否继续自动完成

当前 Phase 688-690 已经完成一个阶段性目标：

```text
定位 value-support residual trajectory 的可见边界。
```

下一步不再只是同一阶段内的顺延，而是进入新的阶段：

```text
Phase 691: Boundary Graph Multi-Edge Decomposition
```

新阶段目标：

```text
在 L13-L18 边界区拆解具体边：
attention head、MLP、residual carry、source-token path 的组合贡献。
```

最低方案：

```text
1. 以 DS7B 72 paired cases 为主；
2. 聚焦 L13-L18；
3. 做 head-level attn_out patch；
4. 做 MLP patch；
5. 做 layer_out minus component 的 residual carry patch；
6. 做组合 patch:
   best_attn + best_mlp
   best_attn + residual_carry
   best_mlp + residual_carry
   all_three
7. 同时记录:
   L18_layer_input projection
   L26_layer_input projection
   final expected token rank
   route competition
```

因为 Phase 691 是新阶段的图谱拆边任务，需要更复杂脚本和更长测试，建议下一次从 Phase 691 开始。
## Phase 691: Boundary Component and Residual-Carry Decomposition [2026-06-26 14:38]

### 命令

```bash
python -m py_compile tests/gpt5/phase691_boundary_component_residual_carry_decomposition.py
python tests/gpt5/phase691_boundary_component_residual_carry_decomposition.py --model qwen3 --hard-exit-after-model > results/glm5_phase691_boundary_component_residual_carry_decomposition/phase691_qwen3_run.log 2>&1
python tests/gpt5/phase691_boundary_component_residual_carry_decomposition.py --model glm4 --hard-exit-after-model > results/glm5_phase691_boundary_component_residual_carry_decomposition/phase691_glm4_run.log 2>&1
python tests/gpt5/phase691_boundary_component_residual_carry_decomposition.py --model deepseek7b --hard-exit-after-model > results/glm5_phase691_boundary_component_residual_carry_decomposition/phase691_deepseek7b_run.log 2>&1
python tests/gpt5/phase691_boundary_component_residual_carry_decomposition.py --summarize-only
```

### 生成脚本和结果

```text
脚本:
  tests/gpt5/phase691_boundary_component_residual_carry_decomposition.py

结果:
  results/glm5_phase691_boundary_component_residual_carry_decomposition/phase691_cross_model_summary.md
  results/glm5_phase691_boundary_component_residual_carry_decomposition/phase691_cross_model_summary.json
  results/glm5_phase691_boundary_component_residual_carry_decomposition/phase691_deepseek7b_boundary_component_rows.jsonl
  results/glm5_phase691_boundary_component_residual_carry_decomposition/phase691_deepseek7b_boundary_component_summary.json
```

### 测试原理

Phase 687-690 的正确部分是：

```text
不能把 L26/L27 layer_out restore 解释为单纯 value writer；
更保守的说法是：
short_only 与 terse_no_explain 在较早层形成 residual trajectory divergence，
然后经 L18-L25 携带到 L26/L27 near-readout carrier。
```

Phase 691 对 Phase 690 找到的边界层做组件拆分。

对每个 paired case，在同一个 case 内计算：

```text
delta_layer = terse_layer_out - short_layer_out
delta_attn  = terse_attn_out  - short_attn_out
delta_mlp   = terse_mlp_out   - short_mlp_out
delta_carry_est = delta_layer - delta_attn - delta_mlp
```

然后分别测试：

```text
restore:
  full_layer_delta
  attn_delta
  mlp_delta
  attn_mlp_delta
  carry_est_layerout
  layer_minus_attn_delta
  layer_minus_mlp_delta
  random_layer_same_norm

degradation:
  full_layer_replace_short
  attn_replace_short
  mlp_replace_short
  attn_mlp_replace_short
  remove_carry_est_layerout
  remove_attn_est_layerout
  remove_mlp_est_layerout
  random_layer_same_norm
```

注意：

```text
carry_est_layerout 是代数估计，不是证明存在一个独立 carry module。
它只回答：
如果从 layer_out 总差分中扣除本层 attn_out / mlp_out 差分，
剩余差分是否仍然携带主要修复信息。
```

### 客观结果

#### DS7B

```text
paired cases = 72
target = L26_layer_input
scan_layers = L13-L18
rows = 6912
```

按 mode 平均：

```text
restore|full_layer_delta:
  repair = 0.544
  mean_patched_rank = 6.85
  target_effect = 5.896

restore|carry_est_layerout:
  repair = 0.417
  mean_patched_rank = 31.35
  target_effect = 4.893

restore|layer_minus_mlp_delta:
  repair = 0.521
  mean_patched_rank = 11.87
  target_effect = 5.734

restore|layer_minus_attn_delta:
  repair = 0.421
  mean_patched_rank = 25.75
  target_effect = 5.147

restore|attn_delta:
  repair = 0.081
  mean_patched_rank = 158.78
  target_effect = 1.112

restore|mlp_delta:
  repair = 0.037
  mean_patched_rank = 211.79
  target_effect = 0.123

restore|attn_mlp_delta:
  repair = 0.111
  mean_patched_rank = 199.20
  target_effect = 1.038

restore|random_layer_same_norm:
  repair = 0.037
```

最强单层 restore：

```text
L18 full_layer_delta:
  repair = 0.792
  mean_patched_rank = 1.60
  target_effect = 6.682

L16 layer_minus_mlp_delta:
  repair = 0.778
  mean_patched_rank = 2.99
  target_effect = 10.315

L15 layer_minus_mlp_delta:
  repair = 0.750
  mean_patched_rank = 2.79
  target_effect = 8.333

L17 layer_minus_attn_delta:
  repair = 0.722
  mean_patched_rank = 2.88
  target_effect = 5.443

L15 carry_est_layerout:
  repair = 0.708
  mean_patched_rank = 3.21
  target_effect = 7.619
```

degradation 结果：

```text
degradation|full_layer_replace_short:
  drop = 0.796
  patched_top1 = 0.204
  target_effect = 6.314

degradation|remove_carry_est_layerout:
  drop = 0.683
  patched_top1 = 0.317
  target_effect = 4.358

degradation|attn_mlp_replace_short:
  drop = 0.315

degradation|attn_replace_short:
  drop = 0.120

degradation|mlp_replace_short:
  drop = 0.199
```

最强 degradation：

```text
L15 remove_carry_est_layerout:
  drop = 0.889

L17 remove_carry_est_layerout:
  drop = 0.875

L13 full_layer_replace_short:
  drop = 0.875

L16 remove_carry_est_layerout:
  drop = 0.861

L14 full_layer_replace_short:
  drop = 0.847
```

#### GLM4

```text
paired cases = 5
target = L38_layer_input
scan_layers = L23-L30
```

结果很宽，但样本太少：

```text
restore|full_layer_delta:
  repair = 1.000

restore|layer_minus_attn_delta:
  repair = 1.000

restore|carry_est_layerout:
  repair = 0.975

restore|attn_mlp_delta:
  repair = 0.850

degradation|full_layer_replace_short:
  drop = 0.375

degradation|remove_carry_est_layerout:
  drop = 0.100
```

#### qwen3

```text
paired cases = 3
target = L33_layer_input
scan_layers = L18-L25
```

样本更少，不能强解释：

```text
restore|full_layer_delta:
  repair = 1.000

restore|carry_est_layerout:
  repair = 1.000

restore|layer_minus_mlp_delta:
  repair = 1.000

restore|attn_delta:
  repair = 0.292

restore|mlp_delta:
  repair = 0.208

degradation|full_layer_replace_short:
  drop = 0.625

degradation|remove_carry_est_layerout:
  drop = 0.583
```

### 对附件判断的评估

附件对 Phase 687-690 的主判断基本正确：

```text
1. L26/L27 有效状态不能解释为纯 value writer；
2. 更合理的描述是 residual trajectory divergence；
3. DS7B 的可见边界在 L13-L18；
4. L18-L25 是强 carry / accumulation 区；
5. L26/L27 是 near-readout carrier；
6. qwen3 / GLM4 样本太少，不能做强跨模型结论；
7. 小模型结果可能存在内部结构偏差。
```

Phase 691 对附件的补充是：

```text
DS7B L13-L18 边界区中，本层 attn_out 和 mlp_out 单独不足以解释 layer_out 效应；
attn_out + mlp_out 单层组合仍明显不足；
layer_out 中扣除 attn / mlp 后的 carry_est 仍保留大量修复和破坏能力。
```

因此当前更保守的结论是：

```text
有效信息主要表现为 residual state carry / accumulated trajectory，
不是单层 attention output 或单层 MLP output 的可分离写入。
```

### 理论进展

当前链条更新为：

```text
instruction wording
  -> early route seed
  -> L13-L18 residual trajectory bifurcation
  -> nonlocal residual carry / accumulated state
  -> L18-L25 strong carry
  -> L26_layer_input value-support state
  -> L26/L27 near-readout carrier
  -> final readout competition
```

Phase 691 使图谱从：

```text
L13-L18 是边界区
```

推进到：

```text
L13-L18 的有效差异主要不在单层 attn_out / mlp_out，
而在 layer_out 总状态与 carry_est 中。
```

### 问题和硬伤

```text
1. carry_est 是代数残差估计，不是独立模块定位；
2. layer_out patch 仍然可能复制格式、位置、路线、续写偏置和值选择等多种变量；
3. 单层 attn_out / mlp_out 弱，不等于多层 attention / MLP 组合弱；
4. 没有 head-level source-token path；
5. 没有证明 carry_est 对自然生成的长程稳定性；
6. qwen3 / GLM4 paired cases 太少；
7. 当前模型是小模型，内部结构可能与更大模型不同。
```

### 接下来是否属于同一阶段

属于。

Phase 691 已经完成单层组件拆分，但还没有排除：

```text
多层 attention / MLP 组合是否可以共同重构 residual trajectory。
```

因此继续自动进入同一阶段的下一步：

```text
Phase 692: Boundary Window Component Combo Audit
```

## Phase 692: Boundary Window Component Combo Audit [2026-06-26 14:43]

### 命令

```bash
python -m py_compile tests/gpt5/phase692_boundary_window_component_combo_audit.py
python tests/gpt5/phase692_boundary_window_component_combo_audit.py --model qwen3 --hard-exit-after-model > results/glm5_phase692_boundary_window_component_combo_audit/phase692_qwen3_run.log 2>&1
python tests/gpt5/phase692_boundary_window_component_combo_audit.py --model glm4 --hard-exit-after-model > results/glm5_phase692_boundary_window_component_combo_audit/phase692_glm4_run.log 2>&1
python tests/gpt5/phase692_boundary_window_component_combo_audit.py --model deepseek7b --hard-exit-after-model > results/glm5_phase692_boundary_window_component_combo_audit/phase692_deepseek7b_run.log 2>&1
python tests/gpt5/phase692_boundary_window_component_combo_audit.py --summarize-only
```

### 生成脚本和结果

```text
脚本:
  tests/gpt5/phase692_boundary_window_component_combo_audit.py

结果:
  results/glm5_phase692_boundary_window_component_combo_audit/phase692_cross_model_summary.md
  results/glm5_phase692_boundary_window_component_combo_audit/phase692_cross_model_summary.json
  results/glm5_phase692_boundary_window_component_combo_audit/phase692_deepseek7b_window_combo_rows.jsonl
  results/glm5_phase692_boundary_window_component_combo_audit/phase692_deepseek7b_window_combo_summary.json
```

### 测试原理

Phase 691 证明：

```text
单层 attn_out / mlp_out patch 明显弱于 layer_out / carry_est。
```

但这个结论有一个潜在漏洞：

```text
如果 attention / MLP 的有效信息分布在多层，
单层 patch 弱并不能说明 attention / MLP 路径不重要。
```

Phase 692 因此做窗口组合审计。

对 Phase 690 / 691 的边界区切成：

```text
early window
late window
all window
```

然后一次性 patch 多层：

```text
attn_window
mlp_window
attn_mlp_window
layer_window
random_layer_window
```

同时做 restore 和 degradation。

### 客观结果

#### DS7B

```text
paired cases = 72
target = L26_layer_input
windows:
  early = L13-L15
  late  = L16-L18
  all   = L13-L18
rows = 2160
```

restore：

```text
layer_window|late:
  repair = 0.806
  mean_patched_rank = 1.61
  target_effect = 6.643

layer_window|all:
  repair = 0.806
  mean_patched_rank = 1.61
  target_effect = 6.643

layer_window|early:
  repair = 0.597
  mean_patched_rank = 6.31
  target_effect = 6.250

attn_mlp_window|all:
  repair = 0.625
  mean_patched_rank = 3.31
  target_effect = 6.844

attn_mlp_window|early:
  repair = 0.444
  mean_patched_rank = 10.21
  target_effect = 8.201

attn_mlp_window|late:
  repair = 0.236
  mean_patched_rank = 19.13
  target_effect = 2.672

attn_window|all:
  repair = 0.167

mlp_window|all:
  repair = 0.153

random_layer_window|all:
  repair = 0.069
```

degradation：

```text
layer_window|late:
  drop = 0.833
  patched_top1 = 0.167
  target_effect = 6.536

layer_window|all:
  drop = 0.833
  patched_top1 = 0.167
  target_effect = 6.536

layer_window|early:
  drop = 0.792
  patched_top1 = 0.208
  target_effect = 7.644

attn_mlp_window|early:
  drop = 0.750
  patched_top1 = 0.250
  target_effect = 8.676

attn_mlp_window|all:
  drop = 0.681
  patched_top1 = 0.319
  target_effect = 6.606

attn_mlp_window|late:
  drop = 0.472

attn_window|all:
  drop = 0.306

mlp_window|all:
  drop = 0.500
```

关键客观现象：

```text
1. 多层 attn_mlp_window 明显强于 Phase 691 的单层 attn_mlp；
2. 但是 attn_mlp_window|all 仍低于 layer_window|all / layer_window|late；
3. early attn_mlp 在 degradation 中很强，说明 L13-L15 对成功路线有关键支撑；
4. late layer_window 在 restore/degradation 中都非常强；
5. layer_window|all 与 layer_window|late 几乎相同，说明多层 layer_out patch 主要由后部窗口主导，或者后部 patch 覆盖了前部轨迹影响。
```

#### GLM4

```text
paired cases = 5
target = L38_layer_input
windows:
  early = L23-L26
  late  = L27-L30
  all   = L23-L30
```

restore 很宽：

```text
attn_mlp_window|all:
  repair = 1.000

attn_window|all:
  repair = 1.000

mlp_window|all:
  repair = 1.000

layer_window|all:
  repair = 1.000
```

degradation 较弱：

```text
layer_window|all:
  drop = 0.400

attn_mlp_window|all:
  drop = 0.200

attn_window|all:
  drop = 0.000

mlp_window|all:
  drop = 0.000
```

样本只有 5，仍不能强解释。

#### qwen3

```text
paired cases = 3
target = L33_layer_input
windows:
  early = L18-L21
  late  = L22-L25
  all   = L18-L25
```

restore：

```text
layer_window|all:
  repair = 1.000

attn_mlp_window|all:
  repair = 1.000

attn_window|all:
  repair = 1.000

mlp_window|all:
  repair = 0.667
```

degradation：

```text
layer_window|all:
  drop = 1.000

attn_mlp_window|all:
  drop = 1.000

attn_window|all:
  drop = 0.333

mlp_window|all:
  drop = 0.333
```

样本只有 3，只能作为提示。

### 对 Phase 691 的修正

Phase 691 中：

```text
单层 attn_out / mlp_out 弱。
```

Phase 692 修正为：

```text
DS7B 单层 attn_out / mlp_out 弱，
但多层 attn_mlp_window 组合明显有效。
```

因此不能说：

```text
attention / MLP 不重要。
```

更准确地说：

```text
有效机制不是单层组件可分离写入，
而是跨层组件组合 + residual layer state carry 的共同结果。
```

### 当前阶段判断

Phase 687-692 形成的新客观拼图：

```text
1. L26/L27 不是单纯 value writer；
2. L26/L27 是 near-readout carrier；
3. 有效轨迹可前溯到 L18-L25；
4. 可见边界在 L13-L18；
5. 单层 attn_out / mlp_out 不足；
6. 多层 attn_mlp_window 有明显因果效应；
7. layer_out / carry_est 仍比组件组合更接近完整轨迹；
8. DS7B 的 early boundary window 对 degradation 特别敏感；
9. late boundary layer_window 对 restore/degradation 都非常强；
10. 因此机制更像跨层轨迹图谱，而不是单点 writer。
```

### 问题和硬伤

```text
1. window patch 是粗粒度组合，仍没有定位 head；
2. attn_mlp_window 有效，但不能说明具体 source token；
3. layer_window 多层 patch 中，后层可能覆盖前层效应；
4. early / late 的 restore 与 degradation 不完全对称；
5. 只测试 first-token readout；
6. qwen3 / GLM4 paired cases 仍过少；
7. 当前都是小模型，结构可能存在偏差，不能直接外推。
```

### 理论进展

当前链条进一步修正为：

```text
instruction wording
  -> L13-L15 early boundary component interaction
  -> L16-L18 late residual state consolidation
  -> L18-L25 residual trajectory carry
  -> L26_layer_input near-readout value-support state
  -> final readout competition
```

关键变化：

```text
从“找单点 writer”
推进到
“定位跨层窗口中的组件组合和残差携带关系”。
```

### 接下来是否继续自动完成

Phase 691-692 已完成当前阶段性目标：

```text
判断 L13-L18 边界区是否能被单层组件解释，
以及多层组件组合是否显著补足。
```

答案是：

```text
单层组件不能解释；
多层 attn_mlp 组合显著有效；
但仍未达到 layer_window / residual state 的完整效应。
```

下一步不再是同一阶段的简单延续，而是进入新阶段：

```text
Phase 693: Boundary Attention Head and Source-Token Path Audit
```

新阶段目标：

```text
在 DS7B L13-L18 边界区，
把 attn_mlp_window 的有效性继续拆成 head-level / source-token path。
```
