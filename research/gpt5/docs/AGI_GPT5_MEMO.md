# AGI GPT5 Memo


## Phase 101: Value-to-Choice Bridge Mapping [2026-06-13 15:01]

### 本轮任务

结合 GPT5 Phase100 和 GLM5 Phase480 的最新进展继续推进。

GLM5 Phase480 的关键进展是：

```text
类别边界残差是普遍机制：
  Qwen3 8/8 类别 selectivity > 1
  GLM4 6/8 类别 selectivity > 1
  DS7B 5/8 类别 selectivity > 1

category_specific 方向有自然使用证据：
  Qwen3 8/8 类别在自身 specific 方向上投影最高。

反向注入有效：
  Qwen3 4/4 类别 -specific 方向能抑制对应类别。
```

这说明 value factor / semantic boundary path 不是虚构方向，而是模型自然使用的语义值路径之一。

GPT5 Phase100 的关键进展是：

```text
Qwen3 L24 head 29/31 是 choice/letter format interface。
它能恢复 letter，但不能恢复 value。
```

因此 Phase101 要测试：

```text
value path 与 choice interface 是否可分离？
value path 被破坏后，choice head restore 能不能救 letter？
choice head 被污染后，value path 是否仍保持？
```

### 生成脚本

```text
tests/gpt5/phase101_value_choice_bridge_mapping.py
tests/gpt5/phase101_value_choice_bridge_mapping_summary.py
tests/gpt5/run_phase101_value_choice_bridge_mapping_full.sh
```

### 测试设计

三模型节点：

```text
Qwen3:
  value path = L6 MLP prefix8
  choice interface = L24 head 29/31 prompt_tail

GLM4:
  value path = L39 MLP prefix8
  choice interface = L39 head 31/17 prompt_tail

DeepSeek7B:
  value path = L27 MLP prefix8
  choice interface = L27 head 21/26 prompt_tail
```

主测试条件：

```text
value_zero:
  清零 value node。

value_transplant:
  把 value node 替换为 same_slot_diff_target donor。

choice_transplant_heads:
  只替换 choice heads。

value_zero + choice_restore_clean_heads:
  value node 清零，同时把 choice heads 恢复为 clean。

value_transplant + choice_restore_clean_heads:
  value node 被 donor 替换，同时把 choice heads 恢复为 clean。

value_transplant + choice_transplant_heads:
  value node 和 choice heads 都被 donor 替换。

value_transplant + choice_transplant_all_restore_clean_heads:
  value node 被 donor 替换；
  choice attention 全部 donor；
  但 choice heads 恢复 clean。
```

判据：

```text
如果 value_zero 破坏 value 和 letter，
但 choice_restore_clean_heads 只恢复 letter、不恢复 value，
说明 choice interface 可绕过或覆盖 letter 输出格式，
但不能恢复语义值路径。

如果 choice_transplant_heads 只破坏 letter、不破坏 value，
说明 choice interface 与 value path 分离。
```

### 运行命令

Smoke：

```bash
python tests/gpt5/phase101_value_choice_bridge_mapping.py qwen3 \
  --value-layer 6 \
  --value-component mlp \
  --value-position prefix8 \
  --choice-layer 24 \
  --choice-heads 29,31 \
  --max-items 2 \
  --output-dir results/gpt5_phase101_smoke \
  --progress-every 1 \
  --hard-exit-after-model
```

三模型主测试：

```bash
chmod +x tests/gpt5/run_phase101_value_choice_bridge_mapping_full.sh
tests/gpt5/run_phase101_value_choice_bridge_mapping_full.sh
```

Qwen3 关键结果加大数据复测：

```bash
OUT=results/gpt5_phase101_value_choice_bridge_mapping_qwen3_validate_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
python tests/gpt5/phase101_value_choice_bridge_mapping.py qwen3 \
  --value-layer 6 \
  --value-component mlp \
  --value-position prefix8 \
  --choice-layer 24 \
  --choice-heads 29,31 \
  --max-items 240 \
  --choice-position prompt_tail \
  --donor-kind same_slot_diff_target \
  --choice-template choice_json_letter \
  --progress-every 40 \
  --output-dir "$OUT" \
  --hard-exit-after-model 2>&1 | tee "$OUT/qwen3_validate.log"
python tests/gpt5/phase101_value_choice_bridge_mapping_summary.py \
  --output-dir "$OUT" | tee "$OUT/summary.log"
```

### 数据规模

三模型主测试：

```text
results/gpt5_phase101_value_choice_bridge_mapping_full_20260613_141841

total_rows = 2520
total_bad_numeric_rows = 0

Qwen3 = 840 rows
GLM4 = 840 rows
DS7B = 840 rows
items/model = 120
```

Qwen3 复测：

```text
results/gpt5_phase101_value_choice_bridge_mapping_qwen3_validate_20260613_144631

rows = 1680
bad_numeric_rows = 0
items = 240
```

### 客观结果

#### 1. Qwen3 主测试：value path 与 choice interface 强分离

```text
value_zero:
  value_delta = -3.2103
  letter_delta = -4.0358
  value_top1_delta = -0.3250
  letter_top1_delta = -0.6333
```

L6 MLP value node 清零后，value 和 letter 都下降，说明 L6 MLP prefix8 是上游强 value path。

```text
choice_transplant_heads:
  value_delta = -0.0093
  letter_delta = -10.0734
  value_top1_delta = 0.0000
  letter_top1_delta = -0.8750
```

只污染 L24 head 29/31，value 几乎不动，但 letter 大崩。这再次说明 head 29/31 是 choice/letter interface，不是 value path。

最关键桥接条件：

```text
value_zero + choice_restore_clean_heads:
  value_delta = -3.1953
  letter_delta = +0.3755
  value_top1_delta = -0.3417
  letter_top1_delta = +0.0667
```

解释：

```text
value 已经被 L6 MLP zero 严重破坏，
但只要 L24 head 29/31 恢复 clean，
letter 反而恢复到接近 clean，甚至略正。
```

这说明：

```text
Qwen3 的 letter choice 可以被 L24 head 29/31 clean interface 强行恢复，
即使上游 value margin 没有恢复。
```

这不是完整语义恢复，而是输出接口恢复。

另一个关键条件：

```text
value_transplant + choice_transplant_all_restore_clean_heads:
  value_delta = -1.5215
  letter_delta = -0.3365
  value_top1_delta = -0.2000
  letter_top1_delta = -0.0333
```

即使 value path 被 donor 替换、choice attention 全部被 donor 污染，只恢复 head 29/31 clean 也能大幅救回 letter。

#### 2. Qwen3 240 items 复测确认

```text
value_zero:
  value_delta = -3.2425
  letter_delta = -4.0035
  value_top1_delta = -0.3042
  letter_top1_delta = -0.6208

choice_transplant_heads:
  value_delta = -0.0065
  letter_delta = -10.2411
  value_top1_delta = +0.0083
  letter_top1_delta = -0.8917

value_zero + choice_restore_clean_heads:
  value_delta = -3.2351
  letter_delta = +0.3130
  value_top1_delta = -0.3125
  letter_top1_delta = +0.0458

value_transplant + choice_transplant_all_restore_clean_heads:
  value_delta = -1.4074
  letter_delta = -0.4401
  value_top1_delta = -0.1417
  letter_top1_delta = -0.0333
```

复测与主测试一致：

```text
1. L6 MLP prefix8 是强 value path。
2. L24 head 29/31 是强 choice/letter interface。
3. choice interface restore 可以恢复 letter，但不能恢复 value。
```

#### 3. GLM4：value path 有效，但 head choice interface 很弱

```text
value_zero:
  value_delta = -0.3594
  letter_delta = 0.0000

value_transplant:
  value_delta = -0.3126
  letter_delta = 0.0000

choice_transplant_heads:
  value_delta = -0.0039
  letter_delta = -0.0510

value_transplant + choice_transplant_heads:
  value_delta = -0.3099
  letter_delta = -0.0510
```

GLM4 的 L39 MLP prefix8 对 value 有影响，但 L39 head 31/17 对 letter 几乎没有强接口作用。

这继续支持：

```text
GLM4 的 choice/output interface 不在 L39 attention heads。
```

#### 4. DS7B：本轮节点不是强 bridge

```text
value_zero:
  value_delta = -0.0666
  letter_delta = 0.0000

value_transplant:
  value_delta = -0.0835
  letter_delta = 0.0000

choice_transplant_heads:
  value_delta = +0.0202
  letter_delta = -0.0182

value_transplant + choice_transplant_all_restore_clean_heads:
  value_delta = +0.0324
  letter_delta = -0.1005
```

DS7B 的 L27 MLP prefix8 和 L27 head 21/26 没形成强 value-to-choice bridge。结合前面阶段，DS7B 仍更像深层多点轨迹/输出释放型。

### 本轮关键进展

1. Qwen3 中第一次得到跨层桥接分离证据：

```text
L6 MLP prefix8 = value path
L24 head 29/31 = choice/letter interface
```

2. Qwen3 的 value 和 letter 可以被独立破坏：

```text
choice_transplant_heads:
  value_delta ≈ 0
  letter_delta ≈ -10
```

3. Qwen3 的 letter 可以在 value 仍坏的情况下被恢复：

```text
value_zero + choice_restore_clean_heads:
  value_delta ≈ -3.2
  letter_delta ≈ +0.31 to +0.38
```

4. GLM4 有 MLP value effect，但没有 Qwen3 式 attention-head choice interface。

5. DS7B 在当前节点上没有明显 bridge，需要改用 segment trajectory 方法。

### 问题和硬伤

1. Qwen3 的 letter 恢复不等于语义正确恢复；value 仍然坏。
2. 现在测试的是 scoring margin，不是 generation 行为。
3. L6 MLP value path 的具体 factor 仍未拆成 category/color/function/material/location 子方向。
4. choice interface 为什么能在 value 坏时恢复 letter，需要进一步解释：可能是 clean head 中已经含有足够的 letter-format state，而不是实时读取 value。
5. GLM4 / DS7B 的 bridge 没找到，不等于不存在，只说明当前节点不是主桥。

### 理论进展

Phase101 明确支持三层结构：

```text
ValuePath:
  语义值路径，决定候选内容评分。

ChoiceInterface:
  选择格式接口，决定输出字母/格式。

Bridge:
  把语义值路径接入选择格式接口的跨层机制。
```

Qwen3 当前结构：

```text
L6 MLP prefix8:
  强 value path。

L24 head 29/31:
  强 choice/letter interface。
```

但本轮更微妙地说明：

```text
L24 head 29/31 clean state 本身已经携带足够强的 letter interface 信息，
可以在上游 value path 被破坏时恢复 letter margin。
```

因此完整公式需要区分：

```text
online bridge:
  当前 forward 中 value factor 实时进入 choice interface。

cached interface state:
  choice head 在 L24 时已经形成的格式/字母状态。
```

更新后的机制表达：

```text
h_l(t_readout, x)
= Base_l
+ ValuePath_l(x)
+ Bridge_l(ValuePath -> ChoiceInterface)
+ ChoiceState_l(letter_label, option_block, task_format)
+ OutputPolicy_l
+ U_l
```

其中 Phase101 已经较强定位：

```text
Qwen3:
  ValuePath ≈ L6 MLP prefix8
  ChoiceState / ChoiceInterface ≈ L24 head 29/31
```

但尚未定位：

```text
Bridge_l(ValuePath -> ChoiceInterface)
```

### 下一步

Phase102 应做：

```text
Qwen3 Value Factor Decomposition inside L6 MLP
```

目标：

```text
1. 把 L6 MLP prefix8 的 value path 拆成 slot-specific factors:
   category / color / function / material / location。

2. 对每个 slot 分别做 value subspace destroy/restore。

3. 判断哪些 slot 的 value factor 会传递到 L24 choice interface。

4. 与 GLM5 Phase480 的 category_specific / semantic boundary residual 对齐：
   看 object-attribute value path 是否也是 category boundary residual 的一种下游读出形式。

5. 对 Qwen3 做更细的:
   L6 factor destroy
   L24 head 29/31 clean restore
   generation audit
```

如果 Phase102 成功，就可以开始把：

```text
semantic boundary factor
object-attribute value path
choice/letter interface
```

三块拼图接成一条更完整的语言输出机制链。

## Phase 102: Value Factor Bridge Decomposition [2026-06-13 16:12]

### 触发问题

用户要求结合附件分析与 `research/glm5/docs/AGI_GLM5_MEMO.md` 最新记录，继续完成全局语义语法契约图谱任务。附件对 Phase100 的判断基本正确：Qwen3 L24 head 29/31 已较强定位为 `choice/letter interface`（选择/字母接口），不是 `semantic value heads`（语义值头）。GLM5 memo 最新 Phase480 进一步给出类别边界残差证据：category-specific semantic boundary direction（类别特异语义边界方向）在 Qwen3/GLM4/DS7B 上都有不同程度复现，尤其 Qwen3 最稳定。因此下一步应把 GPT5 侧的 value path（值路径）与 GLM5 侧的 category boundary residual（类别边界残差）连接起来。

### 生成脚本

```text
tests/gpt5/phase102_value_factor_bridge_decomposition.py
tests/gpt5/phase102_value_factor_bridge_decomposition_summary.py
tests/gpt5/run_phase102_value_factor_bridge_decomposition_full.sh
```

### 执行命令

第一次运行中，Qwen3 在 40/240 后发生 Python 进程段错误：

```text
tests/gpt5/run_phase102_value_factor_bridge_decomposition_full.sh
```

失败信息：

```text
Segmentation fault (core dumped), exit code 139
```

已保存 partial：

```text
results/gpt5_phase102_value_factor_bridge_decomposition_full_20260613_150644/qwen3_phase102_value_factor_bridge_decomposition.partial.json
```

随后以同一输出目录 resume，并提高 partial 落盘频率：

```text
PHASE102_OUTPUT_DIR=results/gpt5_phase102_value_factor_bridge_decomposition_full_20260613_150644 \
PHASE102_PROGRESS_EVERY=10 \
tests/gpt5/run_phase102_value_factor_bridge_decomposition_full.sh
```

三模型最终全部完成。

### 测试规模

```text
Qwen3:      240 items, 2850 rows, bad_numeric_rows=0
GLM4:       240 items, 2850 rows, bad_numeric_rows=0
DeepSeek7B: 240 items, 2850 rows, bad_numeric_rows=0
Total:      8550 rows, bad_numeric_rows=0
```

输出目录：

```text
results/gpt5_phase102_value_factor_bridge_decomposition_full_20260613_150644
```

### 测试原理

Phase101 已定位：

```text
Qwen3:
  L6 MLP prefix8 = value path（值路径）
  L24 head 29/31 = choice/letter interface（选择/字母接口）
```

Phase102 在 value path 内构建多个 rank-4 子空间：

```text
value_all: 全局目标值子空间
value_category / value_color / value_function / value_material / value_location: 当前 slot 的值子空间
relation: 关系/slot 子空间
object: 对象子空间
```

然后测试：

```text
destroy_own_value
transplant_own_value
destroy_all_value
transplant_all_value
destroy_relation
transplant_relation
destroy_object
transplant_object
destroy_own_value + choice_restore_clean_heads
transplant_own_value + choice_restore_clean_heads
destroy_all_value + choice_restore_clean_heads
transplant_all_value + choice_restore_clean_heads
```

读出同时包含：

```text
value_margin: 目标语义值候选的 full-sequence logprob margin
letter_margin: 选择题字母候选的 full-sequence logprob margin
```

这样可以区分：

```text
值路径坏了没有；
选择接口坏了没有；
恢复 choice heads 是否能在 value 仍坏时恢复 letter。
```

### 核心客观结果

#### 1. Qwen3：L6 MLP value factor 是强因果路径

```text
destroy_own_value:
  value_delta = -3.6314
  letter_delta = -3.0400
  value_top1_delta = -0.2833
  letter_top1_delta = -0.2833

destroy_all_value:
  value_delta = -3.7212
  letter_delta = -3.8145
  value_top1_delta = -0.2792
  letter_top1_delta = -0.5958

destroy_relation:
  value_delta = -3.6696
  letter_delta = -3.6072
  value_top1_delta = -0.2208
  letter_top1_delta = -0.4667

destroy_object:
  value_delta = -3.6865
  letter_delta = -3.6449
  value_top1_delta = -0.2625
  letter_top1_delta = -0.4000
```

Qwen3 的 L6 MLP prefix8 中，value_all、relation、object、own_slot_value 子空间都强烈影响 value 与 letter。说明此处不是一个单一语义轴，而是对象、关系、目标值共同参与的 value factor bundle（值因子束）。

#### 2. Qwen3：choice head restore 可以救 letter，但不能救 value

```text
destroy_own_value + choice_restore_clean_heads:
  value_delta = -3.5715
  letter_delta = +1.3842
  value_top1_delta = -0.2875
  letter_top1_delta = +0.0458

destroy_all_value + choice_restore_clean_heads:
  value_delta = -3.5629
  letter_delta = -0.0177
  value_top1_delta = -0.2333
  letter_top1_delta = +0.0208
```

这复现并加强 Phase101 的分离结论：

```text
L24 head 29/31 clean restore 能恢复 letter interface；
但 value path 仍然损坏。
```

因此 Qwen3 的选择输出至少分为：

```text
semantic value path（语义值路径）
choice/letter interface（选择/字母接口）
```

二者可被分离破坏和分离恢复。

#### 3. Qwen3：slot 差异明显

`destroy_all_value` 下按 slot：

```text
category:
  value_delta = -3.635
  letter_delta = -5.645

color:
  value_delta = -0.853
  letter_delta = -2.663

function:
  value_delta = -7.028
  letter_delta = -3.748

location:
  value_delta = -2.997
  letter_delta = -4.017

material:
  value_delta = -4.093
  letter_delta = -2.999
```

function 对 value 最敏感，category/location 对 letter interface 也很强。这说明不同关系槽位不是共用一条完全相同路径，而是共享 value factor bundle 后在输出接口上有不同投影。

#### 4. GLM4：同一测试中 value factor 效应弱很多

```text
destroy_own_value:
  value_delta = -0.1904
  letter_delta = -0.0443

destroy_all_value:
  value_delta = -0.1719
  letter_delta = -0.0323

destroy_relation:
  value_delta = -0.2570
  letter_delta = +0.0875

destroy_object:
  value_delta = -0.2136
  letter_delta = +0.0102
```

GLM4 在 L39 MLP prefix8 上有弱 value effect，但没有 Qwen3 式强 value-to-letter 耦合，也没有可见 choice-head restore 差异：

```text
destroy_own_value 和 destroy_own_value+choice_restore_clean_heads 完全相同；
destroy_all_value 和 destroy_all_value+choice_restore_clean_heads 完全相同。
```

这继续支持：GLM4 的选择接口不在当前 L39 head 31/17。

#### 5. DeepSeek7B：当前节点不是 value bridge，甚至 destroy 常常提升 margin

```text
destroy_own_value:
  value_delta = +0.1337
  letter_delta = +0.1172

destroy_all_value:
  value_delta = +0.1308
  letter_delta = +0.1326

destroy_relation:
  value_delta = +0.2310
  letter_delta = +0.1271

destroy_object:
  value_delta = +0.2382
  letter_delta = +0.1073
```

DeepSeek7B L27 MLP prefix8 与 L27 head 21/26 没有形成 Qwen3 式 value bridge。当前 destroy 子空间反而略微提升 margin，说明该位置更可能是输出竞争/噪声/压缩后接口的一部分，而不是可直接解释的语义值写入路径。

### 本轮进展

1. Qwen3 的 value path 不只是单一 value direction，而是可拆成 object/relation/value-slot 多因子束。
2. Qwen3 的 L6 MLP value factors 对 value 和 letter 都有强因果影响。
3. Qwen3 的 L24 head 29/31 restore 再次证明它们更像 choice/letter interface，而不是 semantic value restore。
4. GLM4 和 DS7B 在当前节点没有同构结构，说明三模型的 value-to-choice bridge 位置不同。
5. GPT5 侧结果与 GLM5 Phase480 的 category boundary residual 可以开始连接：Qwen3 的 category/value factor 不是孤立方向，而是多关系槽位 value bundle 中的一部分。

### 问题和硬伤

1. Qwen3 第一次运行出现 segmentation fault。虽然 resume 后三模型完成，但说明长 hook 会话仍有稳定性风险。
2. 当前是 rank-4 子空间 destroy/transplant，不是最小充分电路。
3. 子空间由 SVD 差分构造，仍可能混入模板、对象身份、候选分布和选项格式。
4. Qwen3 的 choice restore 可以救 letter，但这不等于语义正确；value_delta 仍很负。
5. GLM4/DS7B 没找到 bridge，不等于不存在，只说明当前节点不是主 bridge。
6. 本轮没有 generation audit，只测 full-sequence scoring margin。

### 理论进展

当前更稳的结构应写成：

```text
Output(x)
= Readout(
    ValueBundle_l(object, relation, slot, target)
    -> Bridge_l
    -> ChoiceInterface_l(letter_label, option_format)
  )
```

其中 Qwen3 已有较强定位：

```text
ValueBundle:
  L6 MLP prefix8

ChoiceInterface:
  L24 head 29/31
```

但 Bridge 仍未完全定位。Phase102 说明：

```text
破坏 L6 value bundle 会同时破坏 value 和 letter；
恢复 L24 choice heads 可以恢复 letter，但不能恢复 value。
```

因此语言输出机制不是：

```text
语义值 = 输出字母
```

而至少是：

```text
语义值因子束
→ 跨层桥接
→ 选择/格式接口
→ 输出策略
```

这与“相对编码”一致：单一 binding path 信息有限，必须比较 object / relation / slot / choice interface 多条路径，才能看到全局结构。

### 下一步 Phase103

建议进入：

```text
Qwen3 Bridge Localization Sweep
```

目标不是继续扩大宏观数据，而是在 Qwen3 中定位 `ValueBundle -> ChoiceInterface` 的中间桥：

```text
1. 固定 value destroy at L6 MLP prefix8。
2. 扫描 L8/L12/L16/L20/L22/L24 的 attention 与 MLP restore。
3. 测哪些层/模块能在 value 破坏后恢复 letter，哪些能恢复 value。
4. 对 category/function/location 三个强槽位分别跑。
5. 最后对最强桥接节点做 generation audit。
```

关键判据：

```text
如果某中间模块 restore 能同时恢复 value 和 letter:
  它更接近真正 Bridge。

如果只能恢复 letter:
  它仍是 ChoiceInterface / formatting state。

如果只能恢复 value:
  它是 ValuePath downstream，而不是最终接口。
```

## Phase 103: Bridge Localization Restore Sweep [2026-06-13 21:47]

### 触发问题

附件分析基本正确：Phase101/102 已经把 Qwen3 的机制分成三层：

```text
semantic value path（语义值路径）
→ value-to-choice bridge（值到选择桥）
→ choice/letter interface（选择/字母接口）
```

目前强定位为：

```text
Qwen3 L6 MLP prefix8:
  value path / value factor bundle（值路径/值因子束）

Qwen3 L24 head 29/31:
  choice/letter interface（选择/字母接口）
```

但中间 Bridge 仍未定位。因此 Phase103 固定破坏 value bundle，再扫描后续层模块 clean restore，看哪些模块能恢复 value，哪些只能恢复 letter。

### 生成脚本

```text
tests/gpt5/phase103_bridge_localization_restore_sweep.py
tests/gpt5/phase103_bridge_localization_restore_sweep_summary.py
tests/gpt5/run_phase103_bridge_localization_restore_sweep_full.sh
```

### 执行命令

```bash
tests/gpt5/run_phase103_bridge_localization_restore_sweep_full.sh
```

三模型按顺序运行，并使用 `--hard-exit-after-model`：

```text
qwen3 → glm4 → deepseek7b
```

输出目录：

```text
results/gpt5_phase103_bridge_localization_restore_sweep_full_20260613_202833
```

### 测试规模

```text
Qwen3:      180 items, 5040 rows, bad_numeric_rows=0
GLM4:       180 items, 2880 rows, bad_numeric_rows=0
DeepSeek7B: 180 items, 2880 rows, bad_numeric_rows=0
Total:      10800 rows, bad_numeric_rows=0
```

测试槽位：

```text
category / function / location
```

测试因子：

```text
value_all
own slot value
```

### 测试原理

对每个 item 先计算 clean value/letter margin。

然后破坏指定 value basis：

```text
destroy_only:
  在 value_layer 的 MLP 输出中删除 value factor 子空间投影。
```

再加 clean restore：

```text
destroy_restore:Lx:attn
destroy_restore:Lx:mlp
destroy_restore:Lx:choice_heads
```

判据：

```text
如果 restore 后 value_delta 接近 0:
  该节点可能在 value path downstream 或 bridge 内。

如果 restore 后 letter_delta 接近 0 或变正，但 value_delta 仍很负:
  该节点更像 choice/letter interface 或 format state。

如果 value 和 letter 都恢复:
  才是强 Bridge 候选。
```

### Qwen3 结果

Qwen3 设置：

```text
value destroy:
  L6 MLP prefix8

restore sweep:
  L8/L12/L16/L20/L22/L24 attention and MLP
  L24 choice_heads 29/31
```

#### 1. destroy baseline

```text
destroy_only:
  value_delta = -4.3472
  letter_delta = -4.0910
  value_top1_delta = -0.3639
  letter_top1_delta = -0.3500
```

这比 Phase102 更强，说明本轮在 category/function/location 强槽位上，L6 value bundle 破坏明显。

#### 2. 最强 letter restore 是 L24 attention

按 letter_delta 排序：

```text
L24:attn:
  value_delta = -4.0133
  letter_delta = +0.7901
  value_top1_delta = -0.3139
  letter_top1_delta = +0.0222

L24:choice_heads:
  value_delta = -4.1945
  letter_delta = -0.2206
  value_top1_delta = -0.3722
  letter_top1_delta = +0.0222

L8:attn:
  value_delta = -4.4491
  letter_delta = -3.1648
```

关键现象：

```text
恢复 L24 attention 可以把 letter_delta 从 -4.0910 拉到 +0.7901；
但 value_delta 仍为 -4.0133。
```

这说明 L24 attention 整体比 head 29/31 更能恢复 choice/letter interface，但仍不能恢复 semantic value。

#### 3. 最强 value restore 是 L22/L24 MLP，但恢复幅度有限

按 value_delta 排序：

```text
L22:mlp:
  value_delta = -3.7750
  letter_delta = -4.6763

L24:mlp:
  value_delta = -3.8677
  letter_delta = -3.8117

L24:attn:
  value_delta = -4.0133
  letter_delta = +0.7901
```

L22/L24 MLP 对 value 有一定缓解，但不能恢复到接近 clean。它们也不能恢复 letter。

#### 4. 中间层没有找到同时恢复 value 和 letter 的强 Bridge

```text
L24 attention:
  restore letter, not value

L22/L24 MLP:
  slight value relief, not letter

L8/L12/L16/L20:
  no stable joint restore
```

所以本轮没有定位到强 Bridge，只定位到更清楚的分工：

```text
late attention = choice/letter interface state
late MLP = weak downstream value relief
```

### GLM4 结果

GLM4 设置：

```text
value destroy:
  L33 MLP prefix8

restore sweep:
  L35/L37/L39 attention and MLP
  L39 choice_heads 31/17
```

结果整体很弱：

```text
destroy_only:
  value_delta = -0.0456
  letter_delta = +0.0014

best value restore L39:mlp:
  value_delta = -0.0372
  letter_delta = +0.0021

best letter restore L35:attn / L37:mlp:
  letter_delta = +0.0040
```

GLM4 在当前范式下没有明显 value destruction，也没有明显 bridge restore。说明该任务的 GLM4 value path 不在 L33 MLP prefix8，或者 GLM4 的候选评分路径不适合用本轮 Qwen3 式 value-basis destroy 捕捉。

### DeepSeek7B 结果

DeepSeek7B 设置：

```text
value destroy:
  L24 MLP prefix8

restore sweep:
  L25/L26/L27 attention and MLP
  L27 choice_heads 21/26
```

结果：

```text
destroy_only:
  value_delta = -0.0486
  letter_delta = -0.1387

best value restore L27:mlp:
  value_delta = -0.0034
  letter_delta = -0.1752

best letter restore L26:mlp:
  value_delta = -0.0686
  letter_delta = -0.0292
```

DS7B 的 L27 MLP 可以恢复 value margin 到接近 0，但 letter 更差；L26 MLP 对 letter 有一定缓解但 value 更差。没有发现 joint bridge。

### 本轮关键进展

1. Qwen3 的 L24 attention 是比 head 29/31 更宽的 choice/letter interface 恢复节点。
2. Qwen3 中没有发现能同时恢复 value 和 letter 的单一中间模块。
3. Qwen3 late MLP 对 value 有弱恢复，但和 letter interface 分离。
4. GLM4 当前扫描没有明显 value destroy/restore 结构。
5. DS7B 显示 value 和 letter 可能在 L26/L27 分离：L27 MLP 更接近 value relief，L26 MLP 更接近 letter relief。

### 问题和硬伤

1. Restore 使用 full-sequence clean state，可能包含 candidate-specific state；它能定位恢复节点，但不能直接等同自然重算机制。
2. Qwen3 没有找到强 Bridge，说明 Bridge 可能不是单层单模块，而是多层路径。
3. GLM4/DS7B 结果弱，不代表没有机制；可能是 value destroy 层或因子 basis 选错。
4. 当前仍是 scoring margin，不是 open generation。
5. 本轮只测 category/function/location 三个强槽位，不代表全部关系类型。

### 当前理论更新

Phase103 后，Qwen3 的结构应更谨慎地写成：

```text
L6 MLP:
  ValueBundle(object, relation, slot, target)

L22/L24 MLP:
  downstream value relief / partial value state

L24 attention:
  broad choice/letter interface

L24 head 29/31:
  concentrated letter-label sub-interface
```

也就是说，Bridge 不是一个已经定位的单点，而更可能是：

```text
ValueBundle 从 L6 开始；
沿多层 residual trajectory 传播；
晚层 MLP 保留部分 value state；
晚层 attention 将格式/选项/字母接口接入输出。
```

当前最稳结论仍然是结构分离：

```text
semantic value factor bundle
≠
choice/letter interface
```

### 下一步 Phase104

建议进入：

```text
Qwen3 Segment Dynamic Bridge Recompute
```

目标：

```text
不要再只 restore 单层 clean state。
改为 patch L6 value bundle 后，让 L8-L24 分段自然重算。
```

测试设计：

```text
1. destroy L6 value bundle。
2. restore / transplant segment:
   L8-L12
   L12-L16
   L16-L20
   L20-L24
   L8-L24
3. 比较 value_margin 与 letter_margin。
4. 对 L24 attention 和 L24 MLP 分别做 final restore。
```

关键问题：

```text
如果某段自然重算能同时恢复 value 与 letter:
  Bridge 是 segment-level trajectory。

如果只有 L24 attention 能恢复 letter:
  choice interface 仍是末端格式接口。

如果 value 只能由 MLP segment 恢复:
  value path 与 choice interface 的连接需要多模块组合。
```

## Phase 104: 全局类别分析与类别竞争图谱整合 [2026-06-13 23:59]

### 本阶段目标

读取 `research/glm5/docs/AGI_GLM5_MEMO.md` 最新 Phase 483-484 进展，并参考用户附加资料，完成第一版全局类别分析。重点不是重新运行模型，而是把已经完成的三模型实验拼成一张全局类别地图：

```text
类别 = 共享语义流形 + 类别边界残差 + 竞争释放关系
```

本轮只使用基础分析：读取 JSON、排序、正负号、简单幅度比较、人工归纳，不做复杂统计和高级数学建模。

### 命令记录

```bash
python tests/gpt5/phase104_global_category_analysis.py
python -m py_compile tests/gpt5/phase104_global_category_analysis.py
```

### 脚本与结果

- 脚本：`tests/gpt5/phase104_global_category_analysis.py`
- JSON 结果：`results/gpt5/phase104_global_category_analysis.json`
- Markdown 摘要：`results/gpt5/phase104_global_category_analysis.md`
- 输入结果：
  - `results/glm5/phase483_{qwen3,glm4,deepseek7b}_r1.json`
  - `results/glm5/phase483_{qwen3,glm4,deepseek7b}_r2.json`
  - `results/glm5/phase484_{qwen3,glm4,deepseek7b}_r1.json`
  - `results/glm5/phase484_{qwen3,glm4,deepseek7b}_r2.json`

### 分析原理

1. **Category-Layer Map**：读取 Phase 483 全 8 类最佳层位、目标类别移除幅度、选择性和边界范数，形成类别-层位图。
2. **Competition Graph**：对每个类别移除后的 DCF 变化取正值边，形成 `removed_category -> released_category` 图谱。
3. **Cross-model Stable Edges**：只按“几个模型中为正”做基础稳定性判断，不做统计显著性推断。
4. **Writer Map**：读取 Phase 484 的 MLP 重构 cos@k、显著神经元数、k=5 消融与方向级移除的一致性，粗分为 MLP 因果写入器、集中候选、非 MLP/反向、弥散/缺失、混合未解。
5. **Relation Slot Map**：读取 kind_of / used_for / found_in 下 B_c 注入 delta，判断关系槽位是否改变边界方向读出。
6. **Anomaly Map**：读取 food->vehicle、animal->clothing 的属性释放解释，避免把异常边直接判为错误。

### 核心结果

1. **全局图谱支持当前主假设**：类别不是孤立方向，更像“共享语义流形 + 类别边界残差 + 竞争释放”的组合结构。
2. **跨三模型都为正的释放边**：

```text
animal -> clothing
clothing -> furniture
tool -> vehicle
fruit -> animal
clothing -> plant
vehicle -> clothing
furniture -> clothing
fruit -> clothing
furniture -> fruit
```

这些边不都很强，但它们在 Qwen3、GLM4、DS7B 中方向一致，可能是最早显露的稳定竞争骨架。

3. **模型差异很大**：

```text
Qwen3: 释放幅度最大，竞争图最清楚。
GLM4: 释放幅度整体很小，但方向上仍有若干一致边。
DS7B: 幅度可大，但存在方向不干净和抑制性神经元问题。
```

4. **MLP 因果写入器不是全局统一机制**：

```text
Qwen3 clothing: MLP 因果写入器最清楚，k=5 cos_remove≈0.962。
GLM4 fruit: MLP 因果写入器最清楚，k=5 cos_remove≈0.924。
Qwen3 fruit/animal: MLP 消融方向为负，说明真正写入器可能在 attention 或 residual route。
DS7B animal: cos@50 高但 k=5 消融为负，说明“重构集中”不等于“因果写入”。
```

5. **类别最佳层位不是统一层**：

```text
Qwen3: fruit L32, animal L33, tool L23, vehicle L29, clothing L30, furniture L26, food L34, plant L28
GLM4: fruit L27, animal L38, tool L27, vehicle L29, clothing L39, furniture L34, food L38, plant L32
DS7B: fruit L26, animal L27, tool L26, vehicle L26, clothing L23, furniture L25, food L27, plant L25
```

这说明类别边界存在“类别-模型特异发育时间”，不能继续假设所有类别在同一层形成。

6. **关系槽位读出支持 prompt-invariant 边界，但仍需小尺度复核**：Phase 484 中 fruit 的 B_c 注入 delta 在 kind_of / used_for / found_in 基本不变；但 scale=1.0 可能过强，下一阶段必须做 scale sweep。

7. **异常边不是简单错误**：

```text
food -> vehicle: 可能来自地点/移动属性释放。
animal -> clothing: 可能来自商业/户外属性释放。
```

但 DS7B 的 food/animal 方向不够干净，不能把它作为强证据。

### 理论进展

当前理论应从“局部类别边界存在”升级为：

```text
语言模型内部可能存在类别竞争网络。
类别边界不是一个个孤立坐标轴，而是通过竞争释放关系互相定义。
一个类别的意义，部分来自它激活什么，部分来自它压制什么。
```

更严格地说：

```text
类别 C 的内部编码至少包含三层：

1. 共享属性簇 M_c：
   与邻近类别共用的语义材料。

2. 边界残差 B_c：
   把 C 从邻近类别中分离出来的方向。

3. 竞争抑制场 R_c->*：
   C 激活时对其他类别/属性的压制关系。
```

这很接近“相对编码”的第一性原理：意义不是靠绝对位置定义，而是靠一组可复用属性和一组差异边界共同定义。

### 最严格审视与硬伤

1. **本轮没有新跑模型**：只整合既有 Phase 483/484 结果，因此是全局拼图，不是新增因果证据。
2. **类别数太少**：目前只有 8 类，每类 8 个对象，只能看到雏形，不能证明完整语义大陆。
3. **DCF 词表仍可能制造偏置**：food->vehicle、animal->clothing 等边可能受候选词集合影响，需要宽词表和开放生成复核。
4. **关系不变性可能是假象**：scale=1.0 注入可能覆盖关系模板差异，必须用 0.05/0.1/0.2/0.5/1.0 重测。
5. **写入器证据只覆盖三类**：Phase 484 只对 fruit/animal/clothing 做 MLP 重构，tool/vehicle/furniture/food/plant 还没有写入器级因果图。
6. **MLP 重构与因果不等价**：DS7B animal 是关键反例，cos@50 高但消融为负。
7. **GLM4 幅度太弱**：方向一致不代表机制强，需要更多对象和更干净读出确认。

### 第一性原理判断

如果语言背后存在某种基础数学结构，它现在更像是：

```text
复用材料 + 差异边界 + 竞争抑制 + 层级发育
```

而不是简单的：

```text
词向量空间中有一个类别方向
```

要破解语言背后的数学理论，第一原则应从“寻找单一语义轴”转向“寻找语义如何通过相对差异闭合”。也就是说，核心问题不是 fruit 方向在哪里，而是：

```text
fruit 如何复用 plant/food 的材料；
fruit 如何排除 animal/tool/vehicle；
fruit 的边界在何层形成；
fruit 的边界由哪个模块写入；
fruit 激活后释放/压制哪些邻接类别；
这些关系是否跨模型稳定。
```

### 下一阶段大任务

下一阶段不应只做一个小功能，而应做 **Global Category Atlas v2**：

1. **扩展类别规模**：从 8 类扩展到至少 32 类，每类不少于 24 个对象，覆盖自然物、人造物、生物、身体、地点、材料、抽象概念、社会角色。
2. **建立四张图**：

```text
Category-Layer Map: 每类在哪些层形成边界。
Competition Graph: 每类压制/释放哪些类别。
Writer Map: MLP/attention/residual route 谁写入边界。
Relation Slot Map: 不同关系是否只改变 baseline，不改变 B_c 读出。
```

3. **做 scale sweep**：对 B_c 注入和移除使用 0.05/0.1/0.2/0.5/1.0，确认关系不变性不是强注入造成。
4. **分模块找写入器**：对非 MLP 主导类别，分别测试 attention output、MLP output、residual route，找 fruit/animal 的真正写入源。
5. **异常边宽词表审计**：对 food->vehicle、animal->clothing 做更宽属性词表和开放生成验证，区分真实属性释放与 DCF 偏置。
6. **三模型顺序执行**：若进入模型重测，必须按 Qwen3 -> GLM4 -> DS7B 顺序单模型运行，并添加 `--hard-exit-after-model`，避免 GPU 内存溢出。

## Phase 105: CUDA 全类型系统类别图谱与层位分布分析 [2026-06-14 00:12]

### 本阶段目标

根据用户要求，使用 CUDA 对“所有类型”做系统分析，重点回答：

```text
1. 每种类型分布在哪些层？
2. 每种类型的读出强度、边界强度、类内凝聚是什么样？
3. 类型之间的相对邻接关系是什么？
4. 不同模型是否有相同的类型层位规律？
```

本轮从 8 类扩展到 32 个大类，每类 24 个对象，三模型顺序运行：

```text
qwen3 -> glm4 -> deepseek7b
```

每个模型单独运行，并添加 `--hard-exit-after-model`，避免 GPU 显存残留。

### 执行命令

```bash
python tests/gpt5/phase105_global_category_atlas_cuda.py qwen3 \
  --max-categories 4 \
  --objects-per-category 3 \
  --batch-size 2 \
  --progress-every 1 \
  --output-dir results/gpt5_phase105_smoke \
  --hard-exit-after-model

python tests/gpt5/phase105_global_category_atlas_cuda.py qwen3 \
  --objects-per-category 24 \
  --batch-size 8 \
  --progress-every 12 \
  --output-dir results/gpt5_phase105_global_category_atlas \
  --hard-exit-after-model

python tests/gpt5/phase105_global_category_atlas_cuda.py glm4 \
  --objects-per-category 24 \
  --batch-size 8 \
  --progress-every 12 \
  --output-dir results/gpt5_phase105_global_category_atlas \
  --hard-exit-after-model

python tests/gpt5/phase105_global_category_atlas_cuda.py deepseek7b \
  --objects-per-category 24 \
  --batch-size 8 \
  --progress-every 12 \
  --output-dir results/gpt5_phase105_global_category_atlas \
  --hard-exit-after-model

python tests/gpt5/phase105_global_category_atlas_summary.py \
  --input-dir results/gpt5_phase105_global_category_atlas

python -m py_compile \
  tests/gpt5/phase105_global_category_atlas_cuda.py \
  tests/gpt5/phase105_global_category_atlas_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase105_global_category_atlas_cuda.py`
- 汇总脚本：`tests/gpt5/phase105_global_category_atlas_summary.py`
- Qwen3 结果：`results/gpt5_phase105_global_category_atlas/phase105_qwen3_atlas.json`
- GLM4 结果：`results/gpt5_phase105_global_category_atlas/phase105_glm4_atlas.json`
- DS7B 结果：`results/gpt5_phase105_global_category_atlas/phase105_deepseek7b_atlas.json`
- 跨模型汇总：`results/gpt5_phase105_global_category_atlas/phase105_cross_model_summary.md`

### 类别集合

本轮共 32 类，每类 24 个对象：

```text
fruit, animal, tool, vehicle, clothing, furniture, food, plant,
body, place, building, material, color, emotion, role, profession,
abstract, action, event, time, number, shape, sound, light,
weather, container, instrument, machine, communication, relation,
property, substance
```

### 测试原理

1. 对每个对象构造自然模板：

```text
The {obj} is a kind of
```

2. 使用 CUDA 前向，并设置 `output_hidden_states=True`，抓取所有层最后 token 的 hidden state。

3. 对每个类别、每层计算类别中心：

```text
center(category, layer) = mean(hidden_state(objects in category, layer))
```

4. 对每层计算基础指标：

```text
target margin:
  类别中心对自身类别 readout words 的分数
  -
  对其他类别 readout words 的最大分数

rank:
  自身类别 readout 在 32 类中的排名

cohesion:
  同类对象向类别中心的平均 cos

boundary norm:
  类别中心 - 其他类别中心平均值 的范数

nearest neighbors:
  类别中心与其他类别中心的 cos 排序

local boundary release:
  在最佳 margin 层做本层 logit-lens 边界移除，
  看其他类别 readout 是否上升
```

本轮仍坚持基础分析，不使用复杂统计建模。

### 三模型全局层位结果

```text
Qwen3:
  layers = 36
  best top1 layer = L36, 23/32 类 top1
  best mean margin layer = L36, mean margin = 0.68
  best mean boundary layer = L35, mean boundary norm = 161.17

GLM4:
  layers = 40
  best top1 layer = L40, 22/32 类 top1
  best mean margin layer = L0, mean margin ≈ 0
  best mean boundary layer = L19, mean boundary norm = 2.48

DS7B:
  layers = 28
  best top1 layer = L28, 8/32 类 top1
  best mean margin layer = L0, mean margin ≈ -0.02
  best mean boundary layer = L27, mean boundary norm = 238.80
```

### 关键类别层位图

#### Qwen3

Qwen3 呈现最清楚的晚层类别读出：

```text
fruit:      margin L32, boundary L35, margin 12.54
animal:     margin L32, boundary L35, margin 11.58
tool:       margin L35, boundary L35, margin 6.88
vehicle:    margin L33, boundary L35, margin 12.25
food:       margin L33, boundary L35, margin 16.44
plant:      margin L34, boundary L35, margin 15.91
building:   margin L35, boundary L35, margin 14.42
profession: margin L35, boundary L35, margin 22.24
sound:      margin L33, boundary L35, margin 23.19
shape:      margin L34, boundary L35, margin 8.63
```

Qwen3 中较弱或弥散的类型：

```text
role, abstract, action, time, number, relation
```

这些类型不是没有结构，而是当前 readout basis 下不形成强类别标签 margin。

#### GLM4

GLM4 的 rank 可出现正确，但 DCF margin 幅度极小：

```text
vehicle: margin L40, margin 1.05
body:    margin L40, margin 1.08
emotion: margin L40, margin 2.64
machine: margin L40, margin 1.11
```

多数类别 margin 接近 0，说明当前英文 DCF readout 对 GLM4 可能不够校准，不能简单说 GLM4 没有类别结构。

#### DS7B

DS7B 呈现强晚层 boundary norm，但类别标签 margin 普遍弱：

```text
boundary norm peak = L27
profession: margin L27, margin 26.42
animal:     margin L28, margin 0.91
plant:      margin L28, margin 0.75
property:   margin L12, margin 1.48
```

DS7B 的结构更像“中心和边界存在，但当前 DCF 标签读不干净”。

### 类型邻接关系示例

Qwen3 中一些强类别的最近邻：

```text
fruit -> plant, color, food
animal -> relation, plant, role
vehicle -> machine, container, building
food -> substance, material, container
plant -> color, fruit, relation
building -> place, container, action
profession -> role, relation, action
sound -> action, communication, light
shape -> property, number, light
```

这些邻接关系不是人类分类表的简单复制，而是模型内部类别中心的相对位置。

### 重要理论发现

1. **类别边界和类别读出不是同一件事**

Qwen3 中 margin 和 boundary norm 都在晚层很清楚；但 GLM4/DS7B 出现“boundary 或 rank 有信号，但 margin 很弱”的情况。说明：

```text
类别结构存在
≠
当前 DCF readout 能干净读出
```

2. **边界层普遍偏晚**

Qwen3 边界峰值在 L35，DS7B 在 L27，都是接近末层。类别差异不是只在早层产生，而是经过层级发育后在晚层变得最清楚。

3. **具体名词类比抽象关系类更容易形成强读出**

Qwen3 中 fruit/animal/vehicle/food/plant/building/sound/profession 很强，而 role/abstract/action/time/number/relation 较弱。这说明抽象和关系类可能更依赖上下文槽位，不适合用单一句式和类别标签 readout 直接测。

4. **“类型”不是统一形态**

当前可粗分：

```text
sharp_readout_cohesive:
  读出强、类内凝聚，典型如 Qwen3 fruit/animal/food/plant/sound/profession。

readout_clear:
  有清楚读出，但没有达到最强边界，例如 Qwen3 tool/color/weather/instrument。

cohesive_boundary_unclear_readout:
  有中心/边界，但类别标签读出弱，例如 Qwen3 clothing/furniture/body/machine。

diffuse_or_contextual:
  当前模板下弥散或依赖上下文，例如 GLM4 多数类、Qwen3 role/abstract/action/time/number/relation。
```

### 最严格审视与硬伤

1. **本轮是全层 logit-lens 图谱，不是下游因果干预**

local boundary release 只是本层读出变化，不等同于真正 forward patch 后的输出变化。不能把释放边直接当成因果机制。

2. **模板只有一个**

本轮为了快速完成 32 类三模型全图，只使用：

```text
The {obj} is a kind of
```

抽象类、关系类、动作类明显可能被模板压制。下一轮必须加多模板。

3. **readout words 对 GLM4/DS7B 可能严重不公平**

GLM4 和 DS7B 的 margin 弱，不一定说明类别弱，可能是英文 readout token、chat model 格式或输出头标定导致。

4. **cohesion 容易被共享模板抬高**

所有 prompt 共享前缀，类内凝聚可能混入模板相似性。需要做模板残差扣除或对象 token 位置测试。

5. **类别词表仍是人工定义**

32 类比 8 类更大，但仍不是完整语义空间；一些类别互相重叠，例如 role/profession/relation、material/substance/property。

6. **层位解释要谨慎**

`hidden_states[k]` 表示第 k-1 个 transformer block 之后的状态，L36/L40/L28 接近最终输出接口，可能混入 readout 适配，不完全等价于“语义生成层”。

### 第一性原理更新

本轮把“类别边界”从局部机制推进到全局层位图。当前更合理的第一性原理表述是：

```text
类型不是一个固定向量。
类型是对象集合在层级演化中逐步形成的相对闭合区域。

这个区域至少有四个可观察量：
1. 中心：同类对象是否聚到一起。
2. 边界：该中心和其他类型中心如何分离。
3. 读出：输出头是否能把它命名成类别。
4. 竞争：移除边界后哪些相邻类型被释放。
```

这说明语言背后的数学结构可能不是传统“向量空间 + 分类面”那么简单，而更像：

```text
对象轨道 -> 类别中心 -> 相对边界 -> 竞争网络 -> 输出读出接口
```

也就是意义并非静态存放，而是在层级计算中逐步闭合。

### 下一阶段大任务

下一阶段应做 **Phase 106: 多模板残差扣除 + 因果释放验证**：

1. **多模板重跑**

```text
The {obj} is a kind of
A {obj} belongs to the category of
The word {obj} refers to a type of
People use the word {obj} when talking about
```

2. **模板残差扣除**

对同一模板下所有类别中心求公共模板向量，再从对象表示中扣除，测试 cohesion 和 boundary 是否仍存在。

3. **对象 token 位置测试**

不要只看最后 token，也看对象 token 的首/尾位置，判断类别是在对象处形成，还是在答案槽位形成。

4. **挑选稳定边做真正 CUDA patch**

从 Phase 105 中选择强邻接/强释放边，例如：

```text
fruit -> plant/food
vehicle -> machine/container/building
food -> substance/material/container
profession -> role/relation/action
sound -> action/communication/light
```

在 Qwen3 先做真实 forward boundary removal，再扩展到 GLM4/DS7B。

5. **改进 GLM4/DS7B readout**

为 GLM4/DS7B 单独标定 readout words、中文/英文双语 readout、chat template readout，避免把读出失败误判成结构不存在。

## Phase 106: 多模板残差扣除与对象位置类别图谱复核 [2026-06-14 08:06]

### 本阶段目标

根据用户要求，先判断附加分析是否正确，再继续完成真实客观现象拼图。

对附加分析的收缩判断：

```text
正确部分：
1. Phase105 只是 logit-lens atlas，不是完整因果图谱。
2. 类别结构与 readout interface 需要分开。
3. 下一步必须做多模板、模板残差扣除、对象 token 位置测试。

需要谨慎部分：
1. 不能过早理论总结。
2. 不能把 Phase105 的 local boundary release 当成真实 forward causal edge。
3. GLM4/DS7B 的 weak margin 不能直接解释为类别结构不存在。
```

本轮 Phase106 使用 CUDA 对三模型完整重测，不分小批次实验，不在模型测试期间插入分析。

### 执行命令

```bash
python tests/gpt5/phase106_multitemplate_residual_cuda.py qwen3 \
  --objects-per-category 2 \
  --templates 2 \
  --batch-size 4 \
  --progress-every 2 \
  --output-dir results/gpt5_phase106_smoke \
  --hard-exit-after-model

python tests/gpt5/phase106_multitemplate_residual_cuda.py qwen3 \
  --objects-per-category 24 \
  --templates 4 \
  --batch-size 16 \
  --progress-every 16 \
  --output-dir results/gpt5_phase106_multitemplate_residual \
  --hard-exit-after-model

python tests/gpt5/phase106_multitemplate_residual_cuda.py glm4 \
  --objects-per-category 24 \
  --templates 4 \
  --batch-size 16 \
  --progress-every 16 \
  --output-dir results/gpt5_phase106_multitemplate_residual \
  --hard-exit-after-model

python tests/gpt5/phase106_multitemplate_residual_cuda.py deepseek7b \
  --objects-per-category 24 \
  --templates 4 \
  --batch-size 16 \
  --progress-every 16 \
  --output-dir results/gpt5_phase106_multitemplate_residual \
  --hard-exit-after-model

python tests/gpt5/phase106_multitemplate_residual_summary.py

python -m py_compile \
  tests/gpt5/phase106_multitemplate_residual_cuda.py \
  tests/gpt5/phase106_multitemplate_residual_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase106_multitemplate_residual_cuda.py`
- 汇总脚本：`tests/gpt5/phase106_multitemplate_residual_summary.py`
- Qwen3 结果：`results/gpt5_phase106_multitemplate_residual/phase106_qwen3_multitemplate_residual.json`
- GLM4 结果：`results/gpt5_phase106_multitemplate_residual/phase106_glm4_multitemplate_residual.json`
- DS7B 结果：`results/gpt5_phase106_multitemplate_residual/phase106_deepseek7b_multitemplate_residual.json`
- 跨模型汇总：`results/gpt5_phase106_multitemplate_residual/phase106_cross_model_summary.md`

### 测试规模

```text
models = qwen3, glm4, deepseek7b
categories = 32
objects/category = 24
templates = 4
positions = answer_last, object_last
prompts/model = 32 * 24 * 4 = 3072
total prompts = 9216
```

四个模板：

```text
The {obj} is a kind of
A {obj} belongs to the category of
The word {obj} refers to a type of
People use the word {obj} when talking about
```

两个位置：

```text
answer_last:
  答案槽位/类别读出槽位。

object_last:
  对象 token 最后位置。
```

两种基底：

```text
raw:
  原始 hidden state 类别中心。

template_residual:
  每个 template、每层、每个位置上，先减去该 template 的所有类别公共均值向量。
```

### 客观结果：全局层位

#### Qwen3

```text
answer_last / raw:
  top1 L36 = 21/32
  best mean margin L32 = 0.718
  best boundary L35 = 155.255

answer_last / template_residual:
  best mean margin L33 = 7.587
  best boundary L35 = 155.255

object_last / raw:
  top1 L0 = 18/32
  best mean margin L0 ≈ 0
  best boundary L35 = 119.261

object_last / template_residual:
  top1 L13 = 22/32
  best mean margin L32 = 0.946
  best boundary L35 = 119.261
```

#### GLM4

```text
answer_last / raw:
  top1 L40 = 25/32
  best mean margin L0 ≈ 0
  best boundary L18 = 2.644

answer_last / template_residual:
  top1 L19 = 32/32
  best mean margin L0 ≈ 0
  best boundary L18 = 2.644

object_last / raw:
  top1 L24 = 24/32
  best mean margin L0 ≈ 0
  best boundary L19 = 70.176

object_last / template_residual:
  top1 L20 = 32/32
  best mean margin L0 ≈ 0
  best boundary L19 = 70.176
```

#### DS7B

```text
answer_last / raw:
  top1 L28 = 9/32
  best mean margin L0 = -0.017
  best boundary L27 = 263.246

answer_last / template_residual:
  best mean margin L27 = 4.723
  best boundary L27 = 263.246

object_last / raw:
  top1 L4 = 5/32
  best mean margin L0 = -0.009
  best boundary L27 = 213.556

object_last / template_residual:
  top1 L28 = 15/32
  best mean margin L0 = -0.007
  best boundary L27 = 213.556
```

### 客观结果：Phase105 的直接修正

1. **Qwen3 的 Phase105 结论大体保留，但弱类被模板残差显著增强**

Phase105 中 Qwen3 的强类仍然强，例如：

```text
fruit:   raw 12.57 -> residual 14.07
vehicle: raw 10.10 -> residual 12.08
food:    raw 14.77 -> residual 16.22
plant:   raw 15.02 -> residual 11.99
sound:   raw 25.38 -> residual 14.57
```

但一些 Phase105 中偏弱的类，在 template_residual 后明显增强：

```text
clothing:      3.41 -> 14.79
furniture:     0.62 -> 7.67
body:          0.43 -> 4.87
place:         0.49 -> 7.84
action:       -0.08 -> 4.39
time:         -0.07 -> 8.93
number:       -0.07 -> 7.80
container:     0.16 -> 7.64
communication: 0.52 -> 6.50
property:     -0.08 -> 5.26
```

这说明 Phase105 对 Qwen3 的“弥散类”判断有一部分是模板公共向量污染造成的。

2. **Qwen3 object_last 明显弱于 answer_last，但不是空信号**

object_last / template_residual 中有多类仍有正 margin：

```text
weather 7.85
light 5.13
container 4.18
shape 3.68
vehicle 3.72
relation 3.51
color 3.25
profession 3.10
plant 3.05
```

说明类别信息在对象 token 位置已经存在，但在 answer_last 槽位被显著放大。

3. **GLM4 的问题不是简单模板公共向量污染**

GLM4 在 raw 与 template_residual 下 margin 仍接近 0：

```text
answer_last / template_residual best mean margin ≈ 0
object_last / template_residual best mean margin ≈ 0
```

但 top1 count 可达 32/32，说明 top1 在 margin 极小时会虚高，不能作为强证据。GLM4 更可能需要重新校准 readout words、chat template 或中英文 readout。

4. **DS7B 被 Phase105 明显低估**

DS7B 在 answer_last 做 template_residual 后：

```text
best mean margin: -0.017 -> 4.723
best layer: L27
```

大量类别从 raw 弱信号变成强信号：

```text
fruit:     -0.03 -> 9.18
vehicle:    0.00 -> 9.19
clothing:  -0.02 -> 10.21
plant:      1.01 -> 11.07
body:       0.44 -> 8.80
place:     -0.02 -> 6.93
building:  -0.03 -> 7.21
color:      0.20 -> 9.14
number:    -0.01 -> 9.63
weather:    0.56 -> 14.40
```

这说明 DS7B 内部类别结构并不弱，而是被公共模板/格式方向遮蔽。

5. **边界层结论稳定**

template_residual 不改变类别之间的相对差值，因此 boundary layer 基本不变：

```text
Qwen3 answer_last boundary peak: L35
Qwen3 object_last boundary peak: L35
DS7B answer_last boundary peak: L27
DS7B object_last boundary peak: L27
GLM4 answer_last boundary peak: L18
GLM4 object_last boundary peak: L19
```

### 当前最可靠客观事实

1. **answer_last 是类别读出放大槽位**：Qwen3 和 DS7B 的 answer_last margin 明显强于 object_last。
2. **模板公共向量会严重遮蔽类别方向**：尤其是 DS7B，也影响 Qwen3 弱类。
3. **boundary layer 比 margin layer 更稳定**：扣除模板公共向量后，boundary peak 基本不变。
4. **top1 count 不能单独作为证据**：GLM4 在 margin≈0 时也能出现 32/32 top1。
5. **GLM4 仍未被当前 readout 正确读出**：需要专门 readout 校准。
6. **Phase105 对 Qwen3 强类判断基本正确，但对弱类过于保守；对 DS7B 明显低估。**

### 硬伤分析

1. **仍不是真正因果 patch**：本轮是多模板/残差/位置图谱，不是 downstream forward intervention。
2. **template_residual 可能引入相对化增强**：减去公共均值后 margin 变大，说明差异更清楚，但不等于模型自然输出一定使用这个差异。
3. **object_last 定位是 token subsequence 近似**：多 token 对象或 tokenizer 差异可能影响位置定位。
4. **GLM4 readout 仍失败**：当前英文 readout words 可能不适配 GLM4，需要双语/聊天模板校准。
5. **没有测试跨模板一致对象轨道**：本轮只看中心，不看每个对象在多模板中的轨道是否闭合。

### 下一步任务

Phase107 不应做理论总结，应做真实因果验证：

```text
目标：从 Phase106 中选择最稳定、margin 高、boundary 层稳定的边，做 downstream forward boundary removal。
```

优先测试：

```text
Qwen3:
  clothing, furniture, time, number, action, container
  因为这些类在 template_residual 后从弱变强。

DS7B:
  fruit, vehicle, clothing, plant, body, place, building, weather
  因为这些类从 raw 弱信号变为 residual 强信号。

GLM4:
  暂不做因果边界测试，先做 readout calibration。
```

Phase107 应输出：

```text
1. 自然 forward baseline。
2. best boundary layer removal。
3. template residual boundary removal。
4. random same-norm control。
5. target DCF 下降和 competitor release 上升。
```

## Phase 107: 真实前向类别边界移除因果验证 [2026-06-14 08:43]

### 本阶段目标

根据用户要求，综合 Phase106 正确部分继续任务，不做过早理论总结，优先完成真实客观现象拼图。

Phase106 的正确部分：

```text
1. Phase105/106 仍是 atlas/readout 图谱，不是因果图。
2. 模板公共向量会遮蔽类别方向，尤其 DS7B。
3. answer_last 是类别读出放大槽位。
4. boundary layer 比 margin layer 更稳定。
```

Phase107 的目标：

```text
从 atlas 进入真实 forward causal intervention。
在自然前向传播中，于 boundary layer 的 answer_last 位置移除类别边界投影，
观察最终 logits 的类别 DCF 是否改变。
```

### 执行命令

```bash
python tests/gpt5/phase107_causal_boundary_removal_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --categories fruit,clothing \
  --output-dir results/gpt5_phase107_smoke \
  --hard-exit-after-model

python tests/gpt5/phase107_causal_boundary_removal_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 16 \
  --output-dir results/gpt5_phase107_causal_boundary_removal \
  --hard-exit-after-model

python tests/gpt5/phase107_causal_boundary_removal_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 16 \
  --output-dir results/gpt5_phase107_causal_boundary_removal \
  --hard-exit-after-model

# GLM4 fp16 logits 出现 NaN，改用 bf16 重新运行并覆盖结果
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase107_causal_boundary_removal_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 16 \
  --output-dir results/gpt5_phase107_causal_boundary_removal \
  --hard-exit-after-model

python tests/gpt5/phase107_causal_boundary_removal_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 16 \
  --output-dir results/gpt5_phase107_causal_boundary_removal \
  --hard-exit-after-model

python tests/gpt5/phase107_causal_boundary_removal_summary.py

python -m py_compile \
  tests/gpt5/phase107_causal_boundary_removal_cuda.py \
  tests/gpt5/phase107_causal_boundary_removal_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase107_causal_boundary_removal_cuda.py`
- 汇总脚本：`tests/gpt5/phase107_causal_boundary_removal_summary.py`
- Qwen3 结果：`results/gpt5_phase107_causal_boundary_removal/phase107_qwen3_causal_boundary_removal.json`
- GLM4 结果：`results/gpt5_phase107_causal_boundary_removal/phase107_glm4_causal_boundary_removal.json`
- DS7B 结果：`results/gpt5_phase107_causal_boundary_removal/phase107_deepseek7b_causal_boundary_removal.json`
- 跨模型汇总：`results/gpt5_phase107_causal_boundary_removal/phase107_cross_model_summary.md`

### 测试规模

```text
models = qwen3, glm4, deepseek7b
test categories = 12
train objects/category = 12
heldout test objects/category = 12
templates = 4
prompts/category = 48
conditions = baseline, remove_boundary, random_same_norm
```

测试类别：

```text
fruit, vehicle, clothing, furniture, plant, body,
place, building, time, number, weather, container
```

模型边界层：

```text
Qwen3: L35
GLM4: L18
DS7B: L27
```

### 方法

1. 使用前 12 个对象训练类别中心。
2. 对每个类别在 boundary layer 估计边界：

```text
B_c = mean_template(center(c, template) - mean_other_categories(template))
```

3. 在 heldout 对象上真实 forward。
4. 在 boundary layer 的 answer_last 位置注册 hook：

```text
h := h - projection(h, B_c)
```

5. 对照组：

```text
random_same_norm:
  使用确定性随机单位方向，做同样 projection removal。
```

6. 测量最终 logits 的 32 类 DCF 变化。

### 客观结果摘要

#### Qwen3

```text
fruit:     target Δ +0.16, top release sound +0.69
vehicle:   target Δ +0.11, top release role +0.22
clothing:  target Δ +0.49, top release tool +0.89
furniture: target Δ +1.37, top release building +1.49
plant:     target Δ +0.04, top release color +0.15
body:      target Δ +0.17, top release weather +0.73
place:     target Δ +0.05, top release shape +0.24
building:  target Δ +0.36, top release shape +0.59
time:      target Δ -0.51, top release animal +0.60
number:    target Δ -1.41, top release animal +0.23
weather:   target Δ +0.10, top release light +0.58
container: target Δ +0.03, top release fruit +1.12
```

Qwen3 只有 `time` 和 `number` 表现为目标下降，其中 `time` 同时有竞争释放。
多数具体类别是 release-only 或 target-up/opposed。

#### GLM4

GLM4 初次 fp16 运行 logits 出现 NaN，bf16 重跑后结果有限但正常。

```text
fruit:     target Δ +0.08, top release shape +0.39
vehicle:   target Δ -0.01, top release place +0.53
clothing:  target Δ -0.15, top release property +0.29
furniture: target Δ +0.01, top release material +0.07
plant:     target Δ -0.00, top release material +0.27
body:      target Δ +0.05, top release place +0.31
place:     target Δ +0.03, top release action +0.14
building:  target Δ -0.01, top release action +0.08
time:      target Δ -0.03, top release material +0.16
number:    target Δ +0.05, top release container +0.18
weather:   target Δ -0.26, top release shape +0.30
container: target Δ -0.01, top release role +0.23
```

GLM4 效应整体较小，不能作为强因果证据。

#### DS7B

```text
fruit:     target Δ +0.94, top release time +1.48
vehicle:   target Δ -0.04, top release machine +0.48
clothing:  target Δ +1.05, top release tool +1.58
furniture: target Δ +0.62, top release tool +1.02
plant:     target Δ +1.04, top release animal +1.19
body:      target Δ +0.65, top release container +1.00
place:     target Δ +0.21, top release emotion +0.23
building:  target Δ +0.22, top release fruit +0.55
time:      target Δ +0.10, top release clothing +0.23
number:    target Δ -2.58, no positive release
weather:   target Δ +0.01, top release clothing +0.40
container: target Δ -2.28, no positive release
```

DS7B 的 `number` 和 `container` 出现强目标下降，但没有清楚竞争释放。
多个具体类表现为 target-up/opposed。

### 当前最可靠客观事实

1. **atlas boundary vectors 能真实影响最终 logits**  
   boundary removal 的 release 幅度通常明显大于 random same-norm control。

2. **边界方向不是简单正支持方向**  
   很多类别移除边界后 target DCF 反而上升，例如 Qwen3 furniture、DS7B clothing/plant。

3. **干净的 target-down + competitor-release 很少**  
   本轮最接近的是：

```text
Qwen3 time: target Δ -0.51, animal release +0.60
```

4. **number 类跨模型更像可移除目标边界**

```text
Qwen3 number: target Δ -1.41
DS7B number: target Δ -2.58
```

但两者都缺少强竞争释放，因此更像 target boundary removal，不是完整 competition edge。

5. **GLM4 需要 bf16 才能避免 NaN**

GLM4 fp16 forward logits 不稳定，后续 GLM4 CUDA 测试应默认：

```bash
PROBE_TORCH_DTYPE=bfloat16
```

6. **Phase106 的强 margin 不等于简单因果支持**

Phase106 中 template_residual margin 很强的类别，在 Phase107 中不一定 target-down。
这说明 readout margin、boundary geometry、forward causal support 三者必须分开。

### 硬伤分析

1. **只移除 answer_last 单点**  
   类别边界可能分布在多 token、多层 residual trajectory 中，单点移除不一定能关闭类别。

2. **边界定义仍是 center-vs-others**  
   对 target-up/opposed 类别，说明此边界可能混入抑制方向或读出接口方向。

3. **没有 scale sweep**  
   本轮 scale=1.0，下一轮必须测试 0.25/0.5/1.0/1.5。

4. **没有 layer sweep**  
   只用了 boundary peak layer。真实因果操作层可能不是 boundary norm 最大层。

5. **没有多位置 patch**  
   object_last、answer_last、多 token 共同干预可能与单点干预不同。

### 下一步任务

Phase108 应继续客观测试，不做理论扩张：

```text
Boundary Causal Sweep:
  categories = number, time, container, clothing, furniture, plant
  models = Qwen3, DS7B
  GLM4 = bf16 only, optional calibration branch
```

必须测试：

```text
1. scale sweep: 0.25 / 0.5 / 1.0 / 1.5
2. layer sweep: boundary_layer-3 ... boundary_layer
3. position sweep: object_last, answer_last, both
4. controls: random_same_norm, neighbor_boundary_control
```

核心目标不是总结，而是判定：

```text
哪些类别的边界是正支持方向？
哪些类别的边界是抑制/竞争方向？
哪些类别需要多层/多位置共同移除才有因果效果？
```

## Phase 108: Boundary Causal Sweep 层位-位置-scale-对照系统扫描 [2026-06-14 09:03]

### 本阶段目标

根据用户要求，先判断附加分析是否正确，再继续完成客观现象拼图。

附加分析中正确部分：

```text
1. 分布情况是语言编码机制的核心拼图。
2. Phase107 已经从 atlas/readout 进入真实 forward causal intervention。
3. Phase107 的结果不能解释成“类别边界=简单正支持方向”。
4. 下一步必须做 scale、layer、position、control sweep。
```

本轮 Phase108 目标：

```text
判定哪些类别边界是正支持方向；
哪些是抑制/竞争/接口混合方向；
哪些需要多层/多位置共同移除才出现因果效果。
```

### 执行命令

```bash
python tests/gpt5/phase108_boundary_causal_sweep_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --layer-back 1 \
  --categories number,time \
  --output-dir results/gpt5_phase108_smoke \
  --hard-exit-after-model

python tests/gpt5/phase108_boundary_causal_sweep_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --output-dir results/gpt5_phase108_boundary_causal_sweep \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase108_boundary_causal_sweep_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --output-dir results/gpt5_phase108_boundary_causal_sweep \
  --hard-exit-after-model

python tests/gpt5/phase108_boundary_causal_sweep_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --output-dir results/gpt5_phase108_boundary_causal_sweep \
  --hard-exit-after-model

python tests/gpt5/phase108_boundary_causal_sweep_summary.py

python -m py_compile \
  tests/gpt5/phase108_boundary_causal_sweep_cuda.py \
  tests/gpt5/phase108_boundary_causal_sweep_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase108_boundary_causal_sweep_cuda.py`
- 汇总脚本：`tests/gpt5/phase108_boundary_causal_sweep_summary.py`
- Qwen3 结果：`results/gpt5_phase108_boundary_causal_sweep/phase108_qwen3_boundary_causal_sweep.json`
- GLM4 结果：`results/gpt5_phase108_boundary_causal_sweep/phase108_glm4_boundary_causal_sweep.json`
- DS7B 结果：`results/gpt5_phase108_boundary_causal_sweep/phase108_deepseek7b_boundary_causal_sweep.json`
- 跨模型汇总：`results/gpt5_phase108_boundary_causal_sweep/phase108_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, time, container, clothing, furniture, plant
train objects/category = 12
heldout test objects/category = 12
templates = 4
prompts/category = 48
layers = boundary_layer-3 ... boundary_layer
positions = answer_last, object_last, both
scales = 0.25, 0.5, 1.0, 1.5
controls = boundary, random_same_norm, neighbor_boundary
```

模型层位：

```text
Qwen3: L32-L35
GLM4: L15-L18
DS7B: L24-L27
```

### 客观结果

#### Qwen3

```text
number:
  strongest target down = L35 both scale1.5, target Δ -3.06
  same-setting random target Δ +0.02
  same-setting neighbor target Δ +0.28
  strongest release = animal +0.37

time:
  strongest target down = L35 both scale1.5, target Δ -1.35
  random +0.05
  neighbor +0.69
  strongest release = animal +0.61

container:
  strongest target down = L32 answer_last scale1.5, target Δ -0.34
  strongest release = clothing +2.03 at L34 both scale1.5

clothing:
  strongest target down = L33 answer_last scale1.5, target Δ -0.45
  strongest target up = +0.51
  strongest release = tool +1.08

furniture:
  no meaningful target down
  strongest target up = +2.10
  strongest release = clothing +2.22

plant:
  weak target down = -0.37 at L32 answer_last scale1.5
  strongest release = color +0.25
```

#### GLM4 bf16

```text
number:
  strongest target down = -0.02
  strongest target up = +0.17
  strongest release = material +0.48

time:
  strongest target down = -0.24
  strongest release = material +0.32

container:
  strongest target down = -0.05
  strongest release = event +0.25

clothing:
  strongest target down = -0.13
  strongest release = property +0.16

furniture:
  strongest target down = -0.08
  strongest target up = +0.14
  strongest release = material +0.22

plant:
  strongest target down = -0.06
  strongest release = shape +0.39
```

GLM4 效应仍然弱。

#### DS7B

```text
number:
  strongest target down = L27 both scale1.5, target Δ -4.75
  random -0.02
  neighbor -1.51
  strongest release = clothing +0.46

time:
  no meaningful boundary target down
  neighbor control itself can reduce target strongly
  strongest release = clothing +0.43

container:
  strongest target down = L27 both scale1.5, target Δ -3.21
  random -0.02
  neighbor +0.07
  strongest release weak = clothing +0.09

clothing:
  weak target down only at L25 object_last scale0.25, target Δ -0.17
  strongest target up = +1.61
  strongest release = tool +2.17

furniture:
  no meaningful target down
  strongest target up = +1.02
  strongest release = tool +1.48

plant:
  no meaningful target down
  strongest target up = +1.31
  strongest release = animal +1.51
```

### 当前最可靠客观事实

1. **number 是最稳定的可移除目标边界**

```text
Qwen3 number: L35 both scale1.5 target Δ -3.06
DS7B number: L27 both scale1.5 target Δ -4.75
```

两者都明显强于 random control，也强于 neighbor control。

2. **container 在 DS7B 是强 target-down 边界**

```text
DS7B container: L27 both scale1.5 target Δ -3.21
```

Qwen3 container 不是强 target-down，但有强 release：

```text
Qwen3 container -> clothing +2.03
```

3. **time 在 Qwen3 是 target-down + release，DS7B 不是**

```text
Qwen3 time: target Δ -1.35, animal release +0.61
DS7B time: boundary weak，neighbor control 影响更大
```

4. **clothing/furniture/plant 更像竞争/抑制混合边界**

这些类别常出现：

```text
target up
competitor release
缺少稳定 target down
```

例如：

```text
Qwen3 furniture: target up +2.10, clothing release +2.22
DS7B clothing: target up +1.61, tool release +2.17
DS7B plant: target up +1.31, animal release +1.51
```

5. **both-position 高 scale 对 target-down 很关键**

最强 target-down 基本出现在：

```text
answer_last + object_last
scale = 1.5
boundary peak layer
```

尤其 number 和 DS7B container。

6. **最佳因果层不一定是 boundary norm peak**

Qwen3 container/plant 的 target-down 出现在 L32，而不是 L35。

```text
boundary norm peak ≠ best causal layer
```

### 硬伤分析

1. **scale 最大只到 1.5**
   如果类别边界分布更宽，可能需要多层小 scale 累积，而不是单层大 scale。

2. **boundary vector 仍是 center-vs-others**
   对 clothing/furniture/plant 这类 target-up 类别，说明边界混入 suppressor/interface 成分，需要拆分。

3. **neighbor control 有时很强**
   DS7B time 中 neighbor control target down 更强，说明类别边界互相缠绕。

4. **没有同时做多层 patch**
   本轮是单层 sweep，不是 multi-layer cumulative patch。

5. **没有直接分解 support vs suppressor**
   只能从 target_down、target_up、release 模式推断，尚未直接分离成分。

### 下一步任务

Phase109 应继续客观测试：

```text
Support/Suppressor Decomposition
```

优先对象：

```text
number:
  作为较干净 target-support boundary。

clothing/furniture/plant:
  作为 suppressor/interface mixed boundary。

container:
  比较 Qwen3 release-only 与 DS7B target-down。
```

测试要求：

```text
1. 用 readout target direction 与 boundary vector 做分解。
2. 分别移除 boundary 中的 target-readout aligned component 和 orthogonal component。
3. 测 target_delta 与 release_delta。
4. 加 random_same_norm 和 neighbor_boundary control。
```

## Phase 109: 支持/抑制成分分解方案与条件化关系因子动力学更新 [2026-06-14 09:16]

### 本阶段性质

本阶段没有运行模型测试，而是根据 Phase 105-108 的客观结果，完成系统分析、公式更新和下一阶段研究方案设计。

### 对附加分析的判断

附加分析基本正确，尤其以下判断成立：

```text
1. 分布情况是语言编码机制的核心拼图。
2. Phase 107 已经证明 atlas boundary vector 进入真实 forward causal space。
3. Phase 108 证明类别边界不是简单正支持方向。
4. layer、position、scale、control 四个维度必须同时看。
5. number 是当前最稳定的 target-support boundary。
6. clothing/furniture/plant 更像 suppressor/interface mixed boundary。
```

需要收缩的部分：

```text
1. 不能把 CategoryCausalField 当成已被完整证明的理论对象。
2. 目前只证明了若干类别边界具有可测因果效应。
3. support / suppressor / interface 仍是基于干预模式的工作性分解，还不是直接电路分解。
4. 条件化关系因子动力学公式应更新为可测试公式，而不是最终数学理论。
```

### 当前客观进展

从 Phase 105 到 Phase 108，已经形成一条清楚路径：

```text
Phase 105:
  32 类全局类别图谱，发现层位分布、邻接关系、边界峰值。

Phase 106:
  多模板、模板残差、对象位置/答案位置复核。
  证明模板公共向量会遮蔽类别方向。

Phase 107:
  真实 forward boundary removal。
  证明 atlas boundary vector 能影响最终 logits。

Phase 108:
  layer/position/scale/control sweep。
  证明类别边界有不同因果类型。
```

当前最稳事实：

```text
1. number 是最稳定 target-support boundary:
   Qwen3: target_delta = -3.06
   DS7B:  target_delta = -4.75

2. time 在 Qwen3 中接近 target-down + release:
   target_delta = -1.35
   animal_release = +0.61

3. DS7B container 是强 target-down:
   target_delta = -3.21

4. clothing/furniture/plant 多数表现为 target-up 或 release-only。

5. both-position 高 scale 对强 target-down 很关键。

6. boundary norm peak 不一定是 best causal layer。
```

### 对深度神经网络内部结构研究的进展

当前内部结构研究从“有没有概念方向”推进到“方向的因果功能分类”：

```text
1. 表征几何:
   类别中心、边界、邻接关系存在。

2. 读出接口:
   answer_last 是类别读出放大槽位。

3. 分布式路径:
   object_last + answer_last 共同干预比单位置更强。

4. 因果分类:
   同样是类别边界，可能是支持、抑制、竞争、接口混合。

5. 模型差异:
   Qwen3 和 DS7B 对 number 一致，但对 container/clothing/plant 不一致。
```

这说明深度神经网络内部不是单一语义流，而至少有：

```text
object state
template/base state
category boundary
readout interface
competition/suppression field
final logit projection
```

### 条件化关系因子动力学公式更新

旧公式可以收缩为：

```text
h_{l,p} = Base_{l,p}(template)
        + Object_{l,p}(x)
        + Relation_{l,p}(r)
        + Category_{l,p}(c)
        + residual
```

但 Phase 106-108 表明这个公式不够，因为类别因子不是单一正方向。

更新为可测试公式：

```text
h_{l,p}(x,r,t)
= B_{l,p}(t)
+ O_{l,p}(x | t)
+ R_{l,p}(r | x,t)
+ C_{l,p}(c | x,r,t)
+ I_{l,p}(task | x,r,t)
+ ε
```

其中类别因子需要继续分解：

```text
C_{l,p}(c | x,r,t)
= S_{l,p}(c)
+ U_{l,p}(c)
+ K_{l,p}(c -> neighbors)
+ G_{l,p}(c -> readout)
```

含义：

```text
B:
  模板/基础状态。

O:
  对象状态。

R:
  关系条件状态。

C:
  类别条件状态。

I:
  任务/读出接口状态。

S:
  target-support component，支持目标类别的成分。

U:
  suppressor component，抑制或校准自身/邻居的成分。

K:
  competition component，压制或释放邻接类别的成分。

G:
  readout-interface component，连接输出词表读出的成分。
```

更直接的因果观测公式：

```text
ΔLogits_c
= A_c · Remove(S_c)
+ B_c · Remove(U_c)
+ D_c · Remove(K_c)
+ E_c · Remove(G_c)
```

当前观测对应：

```text
number:
  Remove(S_c) 主导，所以 target down。

clothing/furniture/plant:
  Remove(U_c 或 K_c) 主导，所以 target up 或 competitor release。

container:
  Qwen3 更像 K_c 主导，DS7B 更像 S_c 主导。
```

这不是最终理论，而是下一轮实验可直接证伪的工作公式。

### 当前最大问题和硬伤

1. **还没有直接分解 S/U/K/G**

目前只是通过 target_delta、release_delta、control 差异间接判断。

2. **边界仍由 center-vs-others 定义**

这个边界可能混合多个方向，不适合直接称为类别语义方向。

3. **邻居边界缠绕严重**

DS7B time 中 neighbor control 很强，说明类别边界不是独立坐标轴。

4. **多层累计效应未测**

Phase 108 是单层扫描，没有测试多层小尺度累积移除。

5. **读出词表仍可能影响结论**

DCF readout 仍是人工词表，不等于完整开放生成行为。

6. **GLM4 仍未解决**

GLM4 需要 bf16，且 readout 效应弱，必须做单独校准，不能和 Qwen3/DS7B 直接强比较。

### Phase 109 研究方案

目标：

```text
Support/Suppressor Decomposition
将类别边界拆成 target-readout aligned component 和 orthogonal component，
判断 support、suppressor、competition、interface 的相对贡献。
```

测试对象：

```text
number:
  稳定 target-support boundary。

time:
  Qwen3 中 target-down + animal release。

container:
  Qwen3 release-only, DS7B target-down。

clothing:
  tool release 明显，target-up/混合。

furniture:
  clothing release 明显，target-up/混合。

plant:
  animal/color release，target-up/混合。
```

核心方法：

```text
1. 计算类别边界 B_c。
2. 计算类别 readout direction W_c。
3. 将 B_c 分解为:

   B_parallel = projection(B_c, W_c)
   B_orth     = B_c - B_parallel

4. 分别移除:
   remove B_parallel
   remove B_orth
   remove full B_c

5. 测最终 logits:
   target_delta
   top competitor release
   random_same_norm control
   neighbor_boundary control
```

数据范围：

```text
models:
  qwen3, glm4, deepseek7b

GLM4:
  必须使用 PROBE_TORCH_DTYPE=bfloat16

categories:
  number, time, container, clothing, furniture, plant

train objects/category:
  12

heldout test objects/category:
  12

templates:
  4

positions:
  answer_last, both

layers:
  每个模型/类别采用 Phase 108 最强层 + boundary peak layer

scales:
  0.5, 1.0, 1.5
```

判据：

```text
如果 B_parallel 移除导致 target down:
  target-support component 成立。

如果 B_orth 移除导致 competitor release 或 target up:
  suppressor/competition component 成立。

如果 full B_c 效果大于两者单独效果:
  support 和 suppressor 存在非线性组合或接口耦合。

如果 neighbor control 接近或超过 B_c:
  该类别边界不是独立边界，而是邻接边界缠绕。
```

预期输出：

```text
1. 每类 support/suppressor/competition 类型表。
2. Qwen3 与 DS7B 的类别因果类型对照。
3. GLM4 readout 是否仍弱的客观确认。
4. 可用于 Phase 110 多层累计 patch 的候选类别。
```

## Phase 109: Support/Suppressor Decomposition 实测 [2026-06-14 09:23]

### 本阶段目标

根据 Phase108 的下一步任务，直接测试：

```text
类别边界 B_c 中，哪一部分是 target-readout aligned component，
哪一部分是 orthogonal component，
二者分别导致 target down、target up 还是 competitor release。
```

本轮重点不是理论总结，而是用真实 forward patch 继续客观拼图。

### 执行命令

```bash
python tests/gpt5/phase109_support_suppressor_decomposition_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --categories number,time \
  --output-dir results/gpt5_phase109_smoke \
  --hard-exit-after-model

python tests/gpt5/phase109_support_suppressor_decomposition_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --output-dir results/gpt5_phase109_support_suppressor_decomposition \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase109_support_suppressor_decomposition_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --output-dir results/gpt5_phase109_support_suppressor_decomposition \
  --hard-exit-after-model

python tests/gpt5/phase109_support_suppressor_decomposition_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --output-dir results/gpt5_phase109_support_suppressor_decomposition \
  --hard-exit-after-model

python tests/gpt5/phase109_support_suppressor_decomposition_summary.py

python -m py_compile \
  tests/gpt5/phase109_support_suppressor_decomposition_cuda.py \
  tests/gpt5/phase109_support_suppressor_decomposition_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase109_support_suppressor_decomposition_cuda.py`
- 汇总脚本：`tests/gpt5/phase109_support_suppressor_decomposition_summary.py`
- Qwen3 结果：`results/gpt5_phase109_support_suppressor_decomposition/phase109_qwen3_support_suppressor_decomposition.json`
- GLM4 结果：`results/gpt5_phase109_support_suppressor_decomposition/phase109_glm4_support_suppressor_decomposition.json`
- DS7B 结果：`results/gpt5_phase109_support_suppressor_decomposition/phase109_deepseek7b_support_suppressor_decomposition.json`
- 跨模型汇总：`results/gpt5_phase109_support_suppressor_decomposition/phase109_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, time, container, clothing, furniture, plant
train objects/category = 12
heldout test objects/category = 12
templates = 4
positions = answer_last, both
scales = 0.5, 1.0, 1.5
kinds = full_boundary, readout_parallel, orthogonal, random_same_norm, neighbor_boundary
```

### 分解方法

```text
B_c = category boundary
W_c = category readout direction

B_parallel = projection(B_c, W_c)
B_orth = B_c - B_parallel
```

分别移除：

```text
full_boundary
readout_parallel
orthogonal
random_same_norm
neighbor_boundary
```

### 客观结果

#### Qwen3

```text
number:
  cos(B,W)=0.165
  parallel_norm_fraction=0.165
  readout_parallel target_delta=-0.05
  orthogonal target_delta=-3.05
  full target_delta=-3.06

time:
  cos(B,W)=0.204
  readout_parallel target_delta=-0.18
  orthogonal target_delta=-0.64
  orthogonal release animal+0.96
  full target_delta=-1.35

container:
  readout_parallel target_delta=+0.01
  orthogonal target_delta=-0.33
  orthogonal release shape+2.81
  full target_delta=-0.34

clothing:
  readout_parallel target_delta=-0.37
  orthogonal target_delta=-0.57
  orthogonal release tool+1.46
  full target_delta=-0.45

furniture:
  readout_parallel target_delta=+0.02
  orthogonal target_delta=+1.00
  orthogonal release number+3.30
  full target_delta=+0.72

plant:
  readout_parallel target_delta=+0.11
  orthogonal target_delta=-0.41
  orthogonal release color+0.13
  full target_delta=-0.37
```

#### GLM4 bf16

```text
boundary-readout cos 接近 0。
所有类别效应整体很弱。

number:
  orthogonal target_delta=-0.01
  orthogonal release material+0.45

time:
  orthogonal target_delta=-0.23
  release material+0.33

container:
  orthogonal target_delta=-0.04
  release event+0.25

clothing:
  orthogonal target_delta=-0.13
  release property+0.17

furniture:
  orthogonal target_delta=-0.08
  release material+0.22

plant:
  orthogonal target_delta=-0.06
  release shape+0.39
```

#### DS7B

```text
number:
  cos(B,W)=0.130
  readout_parallel target_delta=-0.08
  orthogonal target_delta=-4.95
  full target_delta=-4.75

container:
  cos(B,W)=0.102
  readout_parallel target_delta=+0.06
  orthogonal target_delta=-3.15
  full target_delta=-3.21

clothing:
  readout_parallel target_delta=-0.87
  orthogonal target_delta=+0.40
  orthogonal release tool+2.24
  full target_delta=+0.39

furniture:
  readout_parallel target_delta=-1.11
  orthogonal target_delta=+0.16
  orthogonal release tool+1.09
  full target_delta=+0.31

plant:
  readout_parallel target_delta=-0.19
  orthogonal target_delta=+0.28
  orthogonal release animal+1.59
  full target_delta=+0.33
```

### 当前最可靠客观事实

1. **强 target-down 主要来自 orthogonal component，而不是 readout_parallel component**

```text
Qwen3 number:
  readout_parallel -0.05
  orthogonal -3.05

DS7B number:
  readout_parallel -0.08
  orthogonal -4.95

DS7B container:
  readout_parallel +0.06
  orthogonal -3.15
```

这推翻了一个简单假设：

```text
target-support boundary 不等于直接 output-readout aligned direction。
```

2. **boundary 与 readout word direction 的 cos 很低**

```text
Qwen3: 约 0.15-0.20
DS7B: 约 0.07-0.13
GLM4: 接近 0
```

说明类别因果边界多数不沿着输出词表 readout 方向。

3. **DS7B clothing/furniture 出现成分冲突**

```text
clothing:
  readout_parallel target down -0.87
  orthogonal release tool +2.24
  full boundary target up +0.39

furniture:
  readout_parallel target down -1.11
  orthogonal release tool +1.09
  full boundary target up +0.31
```

这说明 full boundary 是多个成分相互抵消/冲突后的结果。

4. **Qwen3 furniture 是典型 competition/interface 混合边界**

```text
orthogonal target up +1.00
orthogonal release number +3.30
full target up +0.72
```

5. **GLM4 仍然弱**

GLM4 的边界-readout cos 接近 0，效应小，仍需 readout calibration。

### 对公式的修正

Phase109 后，`S` 不应再简单等同于 readout_parallel。

需要改为：

```text
C_c = S_c + U_c + K_c + G_c
```

但：

```text
G_c ≈ readout_parallel component
S_c 不一定与 G_c 对齐
S_c 很可能主要位于 readout-orthogonal causal subspace
```

也就是说：

```text
target support 不是直接输出词方向；
它可能是通过内部因果子空间改变最终 readout。
```

这对破解编码机制非常关键。

### 硬伤分析

1. **readout direction 仍由 DCF 词表定义**

如果 readout words 不准，parallel/orthogonal 分解会受影响。

2. **orthogonal component 仍然太大**

因为 boundary-readout cos 很低，orthogonal 几乎包含大部分边界，仍需进一步分解。

3. **只分成两块还不够**

orthogonal 中同时包含 support、suppressor、competition、interface residual。

4. **未做多层累计**

number/container 的 orthogonal target-down 强，但是否来自单层或多层累积仍未知。

5. **GLM4 readout 问题未解决**

GLM4 不能用于强机制结论。

### 下一步任务

Phase110 应继续客观测试：

```text
Orthogonal Subspace Split
```

目标：

```text
把 B_orth 继续分成:
1. neighbor-aligned component
2. target-object trajectory component
3. residual component
```

优先测试：

```text
Qwen3 number/time/furniture
DS7B number/container/clothing/furniture/plant
```

方法：

```text
1. 用 neighbor boundary basis 分解 B_orth。
2. 用 object_last -> answer_last transport direction 分解 B_orth。
3. 分别移除各子成分。
4. 测 target_delta、release_delta、control_delta。
```

## Phase 110: Orthogonal Subspace Split 正交子空间拆分 [2026-06-14 09:34]

### 本阶段目标

根据 Phase109 的结果，`readout_parallel` 不是主要 target support，真正强因果成分主要位于 `readout-orthogonal` 子空间。

本阶段继续把 `B_orth` 拆成三类更基础成分：

```text
1. neighbor_aligned: 与邻近类别边界空间对齐的成分
2. transport_aligned: 与 object_last -> answer_last 平均传输方向对齐的成分
3. residual: 去除 neighbor 和 transport 后剩余的成分
```

核心问题：

```text
强 target-down 到底来自类别竞争边界、对象到答案位置的传输通道，还是剩余未知方向。
```

### 执行命令

```bash
python -m py_compile tests/gpt5/phase110_orthogonal_subspace_split_cuda.py

python tests/gpt5/phase110_orthogonal_subspace_split_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --categories number,time \
  --output-dir results/gpt5_phase110_smoke \
  --hard-exit-after-model

python tests/gpt5/phase110_orthogonal_subspace_split_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --output-dir results/gpt5_phase110_orthogonal_subspace_split \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase110_orthogonal_subspace_split_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --output-dir results/gpt5_phase110_orthogonal_subspace_split \
  --hard-exit-after-model

python tests/gpt5/phase110_orthogonal_subspace_split_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --output-dir results/gpt5_phase110_orthogonal_subspace_split \
  --hard-exit-after-model

python tests/gpt5/phase110_orthogonal_subspace_split_summary.py

python -m py_compile \
  tests/gpt5/phase110_orthogonal_subspace_split_cuda.py \
  tests/gpt5/phase110_orthogonal_subspace_split_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase110_orthogonal_subspace_split_cuda.py`
- 汇总脚本：`tests/gpt5/phase110_orthogonal_subspace_split_summary.py`
- Qwen3 结果：`results/gpt5_phase110_orthogonal_subspace_split/phase110_qwen3_orthogonal_subspace_split.json`
- GLM4 结果：`results/gpt5_phase110_orthogonal_subspace_split/phase110_glm4_orthogonal_subspace_split.json`
- DS7B 结果：`results/gpt5_phase110_orthogonal_subspace_split/phase110_deepseek7b_orthogonal_subspace_split.json`
- 跨模型汇总：`results/gpt5_phase110_orthogonal_subspace_split/phase110_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, time, container, clothing, furniture, plant
train objects/category = 12
heldout test objects/category = 12
templates = 4
prompts/category = 48
components = orthogonal_full, neighbor_aligned, transport_aligned, residual, random_same_norm
positions = answer_last, both
scales = 1.0, 1.5
```

模型层位：

```text
Qwen3: L35
GLM4: L18
DS7B: L27
```

### 客观结果

#### Qwen3

```text
number:
  norm fractions neighbor/transport/residual = 0.54/0.27/0.80
  best neighbor target Δ -1.91
  best transport target Δ -3.43
  best residual target Δ -0.37
  best orthogonal_full target Δ -3.05
  random_same_norm target Δ -0.12

time:
  fractions = 0.57/0.15/0.81
  neighbor target Δ -1.95
  transport target Δ -1.84
  residual target Δ +0.03
  orthogonal_full target Δ -0.64

container:
  fractions = 0.46/0.28/0.84
  neighbor target Δ +0.24
  transport target Δ -1.75
  residual target Δ +0.00
  orthogonal_full target Δ +0.25

clothing:
  fractions = 0.34/0.39/0.85
  neighbor target Δ +0.71
  transport target Δ -1.43
  residual target Δ +0.01
  orthogonal_full target Δ +0.72

furniture:
  fractions = 0.54/0.33/0.78
  neighbor target Δ +0.55
  transport target Δ -0.56
  residual target Δ -0.46
  orthogonal_full target Δ +1.93

plant:
  fractions = 0.49/0.28/0.83
  neighbor target Δ +0.10
  transport target Δ -5.97
  residual target Δ -0.29
  orthogonal_full target Δ -0.02
```

#### GLM4 bf16

```text
number:
  fractions = 0.64/0.04/0.77
  strongest target down = neighbor Δ -0.14

time:
  fractions = 0.74/0.07/0.67
  strongest target down = neighbor Δ -0.47

container:
  fractions = 0.84/0.06/0.53
  strongest target down = transport Δ -0.07

clothing:
  fractions = 0.89/0.01/0.45
  strongest target down = orthogonal_full Δ -0.08

furniture:
  fractions = 0.93/0.00/0.37
  strongest target down = transport Δ -0.03

plant:
  fractions = 0.90/0.04/0.44
  strongest target down = orthogonal_full Δ -0.06
```

GLM4 仍然弱，不能作为强机制结论来源。

#### DS7B

```text
number:
  fractions = 0.41/0.22/0.89
  neighbor target Δ -0.94
  transport target Δ +1.06
  residual target Δ -2.76
  orthogonal_full target Δ -4.95
  random_same_norm target Δ +0.07

time:
  fractions = 0.46/0.18/0.87
  neighbor target Δ -0.82
  transport target Δ -0.61
  residual target Δ -0.93
  orthogonal_full target Δ +0.06

container:
  fractions = 0.30/0.31/0.90
  neighbor target Δ -0.24
  transport target Δ -5.68
  residual target Δ -1.44
  orthogonal_full target Δ -3.15

clothing:
  fractions = 0.28/0.44/0.85
  neighbor target Δ -0.18
  transport target Δ -5.17
  residual target Δ -0.91
  orthogonal_full target Δ +1.22

furniture:
  fractions = 0.44/0.35/0.83
  neighbor target Δ +0.07
  transport target Δ -3.85
  residual target Δ -0.03
  orthogonal_full target Δ +0.31

plant:
  fractions = 0.42/0.34/0.84
  neighbor target Δ +0.66
  transport target Δ -3.28
  residual target Δ -0.12
  orthogonal_full target Δ +1.05
```

### 当前最可靠客观事实

1. **transport_aligned 是大量类别的强 target-down 成分**

典型结果：

```text
Qwen3 number transport Δ -3.43
Qwen3 plant transport Δ -5.97
DS7B container transport Δ -5.68
DS7B clothing transport Δ -5.17
DS7B furniture transport Δ -3.85
DS7B plant transport Δ -3.28
```

这说明 object_last 到 answer_last 的内部传输方向，是类别信息进入答案位置的重要候选通道。

2. **完整 orthogonal_full 会掩盖子成分**

例如：

```text
Qwen3 plant:
  transport Δ -5.97
  orthogonal_full Δ -0.02

DS7B clothing:
  transport Δ -5.17
  orthogonal_full Δ +1.22

DS7B plant:
  transport Δ -3.28
  orthogonal_full Δ +1.05
```

完整正交边界里混有方向相反的成分，直接移除整块会发生抵消甚至 target-up。

3. **DS7B number 是特殊模式**

```text
DS7B number:
  residual Δ -2.76
  orthogonal_full Δ -4.95
  transport Δ +1.06
```

number 在 DS7B 中不是 transport 主导，而更像剩余未知方向与完整正交边界共同形成强支撑。

4. **Qwen3 time 更像 neighbor/transport 混合**

```text
Qwen3 time:
  neighbor Δ -1.95
  transport Δ -1.84
  orthogonal_full Δ -0.64
```

time 与 number、event、weather 等邻近类别纠缠更强。

5. **GLM4 仍然低效应**

GLM4 的最大效应大多小于 0.5，继续证明当前 readout/intervention 框架下 GLM4 信号弱。

### 对 Phase109 附加分析的校正

Phase109 的核心判断仍然正确：

```text
target support 主要不在 readout_parallel；
readout-orthogonal causal subspace 是关键区域。
```

但 Phase110 进一步说明：

```text
readout-orthogonal 不是一个单一语义边界；
其中大量强因果效应来自 object_last -> answer_last transport component。
```

因此更准确的说法是：

```text
模型内部的类别信息，可能先在对象位置形成类别/对象状态，
再通过位置传输通道进入答案位置，
最后才改变输出词 readout。
```

### 条件化关系因子动力学公式更新

上一阶段：

```text
C_c = S_c + U_c + K_c + G_c
```

Phase110 后应拆成：

```text
C_c = G_c + N_c + T_c + R_c
```

含义：

```text
C_c: 类别边界整体
G_c: readout-parallel output gateway
N_c: neighbor-aligned competition/interface component
T_c: object_last -> answer_last transport component
R_c: residual unknown causal component
```

更接近当前结果的因果链：

```text
object state at object_last
  -> T_c transport to answer_last
  -> answer-position category state
  -> G_c/output gateway
  -> next-token category logits
```

中文解释：

```text
对象位置先承载对象/类别状态；
答案位置不是凭空生成类别，而是接收对象位置传来的类别状态；
输出词方向只是最后的门口，不是内部语义支撑本体。
```

### 硬伤分析

1. **transport direction 只是均值差分方向**

当前 `object_last -> answer_last` 是平均残差差分，不等于已经证明真实路径。

2. **neighbor basis 是人工邻接**

邻近类别由人为指定，可能漏掉模型内部真正的竞争类别。

3. **仍是单层干预**

如果类别传输跨多层累积，单层移除会低估或扭曲真实机制。

4. **子成分之间不是线性独立因果模块**

一些子成分移除比完整 orthogonal_full 更强，说明完整边界内部存在非线性或方向抵消。

5. **GLM4 仍然不能用于强结论**

GLM4 在当前框架下效应弱，需要单独校准。

### 下一步任务

Phase111 应做一个更大的阶段任务：

```text
Transport Path Causal Mapping
```

目标：

```text
确认 transport component 是否是真正的对象位置到答案位置类别传输通道。
```

建议测试：

```text
1. 对 object_last 单独写入/移除 transport component，观察 answer_last 与 logits 是否同步变化。
2. 对 answer_last 单独写入/移除 transport component，和 object_last 干预对照。
3. 做 layer-to-layer transport sweep，找出类别状态从对象位置迁移到答案位置的层段。
4. 做 multi-layer cumulative patch，确认单层结果是否低估或被抵消。
5. 对 Qwen3 number/time/plant 与 DS7B number/container/clothing/furniture/plant 扩大 heldout objects 做复测。
```

优先级：

```text
第一优先：DS7B container/clothing/furniture/plant 的 transport-dominant 现象
第二优先：Qwen3 plant 的 transport 强效但 orthogonal_full 近零现象
第三优先：DS7B number 的 residual-support 特殊模式
```

## Phase 111: Transport Path Causal Mapping 传输路径因果定位 [2026-06-14 10:43]

### 本阶段目标

根据用户附加分析与 Phase110 结果，先判断：

```text
Phase110 的 transport_aligned 强 target-down 是否等于真实 object_last -> answer_last 传输路径？
```

附加分析中正确部分：

```text
1. Phase110 的 transport_aligned 是目前最强候选语义通道之一。
2. readout_parallel 不是主要语义支持方向。
3. orthogonal_full 会掩盖强子成分。
4. transport direction 仍只是均值差分，不等于已经证明真实路径。
5. 下一步必须做 object-site 与 answer-site 的因果对照。
```

因此本阶段不再继续理论总结，而是直接测试：

```text
在 object_last 移除/写入 T_c，answer_last 的 T_c 投影和 final logits 是否同步变化。
```

### 执行命令

smoke：

```bash
python -m py_compile \
  tests/gpt5/phase111_transport_path_causal_mapping_cuda.py \
  tests/gpt5/phase111_transport_path_causal_mapping_summary.py

python tests/gpt5/phase111_transport_path_causal_mapping_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --layer-back 1 \
  --categories number,time \
  --scales 1.0 \
  --output-dir results/gpt5_phase111_smoke \
  --hard-exit-after-model
```

正式测试第一轮使用：

```bash
python tests/gpt5/phase111_transport_path_causal_mapping_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --output-dir results/gpt5_phase111_transport_path_causal_mapping \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase111_transport_path_causal_mapping_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --output-dir results/gpt5_phase111_transport_path_causal_mapping \
  --hard-exit-after-model

python tests/gpt5/phase111_transport_path_causal_mapping_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --output-dir results/gpt5_phase111_transport_path_causal_mapping \
  --hard-exit-after-model
```

第一轮发现 Phase110 的强效常在 scale=1.5，而默认范围只有 0.25/0.5/1.0。为避免错误否定，重新加入 1.5 完整复测：

```bash
python tests/gpt5/phase111_transport_path_causal_mapping_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --scales 0.25,0.5,1.0,1.5 \
  --output-dir results/gpt5_phase111_transport_path_causal_mapping \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase111_transport_path_causal_mapping_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --scales 0.25,0.5,1.0,1.5 \
  --output-dir results/gpt5_phase111_transport_path_causal_mapping \
  --hard-exit-after-model

python tests/gpt5/phase111_transport_path_causal_mapping_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --scales 0.25,0.5,1.0,1.5 \
  --output-dir results/gpt5_phase111_transport_path_causal_mapping \
  --hard-exit-after-model

python tests/gpt5/phase111_transport_path_causal_mapping_summary.py

python -m py_compile \
  tests/gpt5/phase111_transport_path_causal_mapping_cuda.py \
  tests/gpt5/phase111_transport_path_causal_mapping_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase111_transport_path_causal_mapping_cuda.py`
- 汇总脚本：`tests/gpt5/phase111_transport_path_causal_mapping_summary.py`
- Qwen3 结果：`results/gpt5_phase111_transport_path_causal_mapping/phase111_qwen3_transport_path_causal_mapping.json`
- GLM4 结果：`results/gpt5_phase111_transport_path_causal_mapping/phase111_glm4_transport_path_causal_mapping.json`
- DS7B 结果：`results/gpt5_phase111_transport_path_causal_mapping/phase111_deepseek7b_transport_path_causal_mapping.json`
- 跨模型汇总：`results/gpt5_phase111_transport_path_causal_mapping/phase111_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, time, container, clothing, furniture, plant
train objects/category = 12
heldout test objects/category = 12
templates = 4
prompts/category = 48
patch layers = peak-3 ... peak
patch sites = object_last, answer_last
patch modes = remove_target, amplify_target, wrong_inject_abs, random_remove
scales = 0.25, 0.5, 1.0, 1.5
monitor = answer_last transport projection at peak layer + final DCF logits
```

模型层位：

```text
Qwen3: monitor L35, patch L32-L35
GLM4: monitor L18, patch L15-L18
DS7B: monitor L27, patch L24-L27
```

### 测试原理

Phase110 中 `T_c` 的定义：

```text
B_orth = B - proj(B, readout_direction)
after_neighbor = B_orth - proj(B_orth, neighbor_boundary_basis)
T_c = proj(after_neighbor, mean(answer_last - object_last))
```

Phase111 做真实 forward 干预：

```text
1. 在 object_last 移除 target T_c。
2. 在 answer_last 移除 target T_c。
3. 写入 wrong-category T_d。
4. 使用 random_same_norm 作为对照。
5. 记录 final logits target_delta。
6. 同时记录 peak layer answer_last 的 T_c projection delta。
```

强路径闭包判据：

```text
object_last remove T_c
  -> answer_last T_c projection 同步下降
  -> target logits 同步下降
  -> 明显强于 random control
```

### 客观结果

#### Qwen3

```text
number:
  object_last remove: target Δ -0.00, answer projection Δ -0.10
  answer_last remove: target Δ -3.43
  wrong inject: target Δ -3.54
  random: target Δ -0.05

time:
  object_last remove: target Δ -0.03, answer projection Δ +0.15
  answer_last remove: target Δ -1.84
  wrong inject: target Δ -4.18
  random: target Δ -0.09

container:
  object_last remove: target Δ -0.05, answer projection Δ -0.16
  answer_last remove: target Δ -2.59
  wrong inject: target Δ -0.76
  random: target Δ -0.08

clothing:
  object_last remove: target Δ +0.01
  answer_last remove: target Δ -1.43
  wrong inject: target Δ -4.51

furniture:
  object_last remove: target Δ +0.01
  answer_last remove: target Δ -0.55
  wrong inject: target Δ -3.26

plant:
  object_last remove: target Δ -0.00, answer projection Δ +0.10
  answer_last remove: target Δ -5.97
  wrong inject: target Δ -2.52
```

#### GLM4 bf16

```text
all categories:
  object_last remove target effect near 0
  answer_last remove target effect near 0
  wrong inject weak
```

最大量级仍然很小：

```text
wrong inject clothing Δ -0.22
wrong inject furniture Δ -0.21
```

GLM4 在当前框架中仍不能支持强机制判断。

#### DS7B

```text
number:
  object_last remove: target Δ -0.07
  answer_last remove: target Δ +0.69
  wrong inject: target Δ -3.39
  random: target Δ -0.12

time:
  object_last remove: target Δ -0.02
  answer_last remove: target Δ -0.56
  wrong inject: target Δ -1.50

container:
  object_last remove: target Δ -0.21
  object-site strongest answer projection drop: Δ -1.70, but target Δ +0.08
  answer_last remove: target Δ -5.50
  random: target Δ -0.38

clothing:
  object_last remove: target Δ -0.23
  object-site strongest answer projection drop: Δ -2.23, but target Δ +0.05
  answer_last remove: target Δ -5.04

furniture:
  object_last remove: target Δ -0.17
  object-site strongest answer projection drop: Δ -2.16, but target Δ +0.11
  answer_last remove: target Δ -3.82

plant:
  object_last remove: target Δ -0.15
  answer projection Δ -0.75
  answer_last remove: target Δ -3.20
  wrong inject: target Δ -2.11
```

### 当前最可靠客观事实

1. **answer_last 是 transport_aligned 强 target-down 的直接作用位点**

强结果与 Phase110 基本对齐：

```text
Qwen3 number answer_last remove Δ -3.43
Qwen3 plant answer_last remove Δ -5.97
DS7B container answer_last remove Δ -5.50
DS7B clothing answer_last remove Δ -5.04
DS7B furniture answer_last remove Δ -3.82
DS7B plant answer_last remove Δ -3.20
```

2. **object_last remove 没有形成强 logits 因果闭包**

所有模型/类别中，object_last remove 的 target_delta 都很弱：

```text
Qwen3: roughly 0
GLM4: roughly 0
DS7B: strongest only around -0.23
```

3. **DS7B object_last 干预可以改变 answer projection，但不改变 target logits**

例如：

```text
DS7B container:
  object-site answer projection Δ -1.70
  target Δ +0.08

DS7B clothing:
  object-site answer projection Δ -2.23
  target Δ +0.05

DS7B furniture:
  object-site answer projection Δ -2.16
  target Δ +0.11
```

这说明“投影同步变化”本身不足以证明输出因果闭包。

4. **wrong-category injection 往往很强，但更像干扰/抑制，不是清晰类别替换**

例如：

```text
Qwen3 clothing wrong inject Δ -4.51
Qwen3 time wrong inject Δ -4.18
DS7B number wrong inject Δ -3.39
```

这些结果说明 wrong T_d 写入会强烈扰乱目标类别，但尚未证明它把输出推向指定错误类别。

5. **GLM4 继续弱**

GLM4 仍不能用于强结论。

### 对 Phase110 理论的校正

Phase110 的正确部分：

```text
T_c 是大量类别的强 target-down 成分。
T_c 位于 readout-orthogonal 子空间。
完整 orthogonal_full 会被其他成分抵消。
```

Phase111 的关键校正：

```text
当前还不能说 T_c 已被证明为 object_last -> answer_last 的真实传输路径。
```

更严格表述应为：

```text
T_c 是 answer_last 上非常强的类别状态/读出前状态成分；
它与 object_last -> answer_last 的均值差分对齐；
但 object_last 单点移除没有让 final logits 产生同步强变化。
```

因此当前理论从：

```text
object_last category state -> T_c transport -> answer_last
```

暂时回退为更谨慎的版本：

```text
object/answer positional contrast defines T_c;
T_c at answer_last is a strong causal pre-readout state;
object_last 单点 patch 尚未闭合到 logits。
```

### 条件化关系因子动力学公式更新

Phase110 公式：

```text
C_c = G_c + N_c + T_c + R_c
```

Phase111 后应加上位点区分：

```text
C_c(answer) = G_c(answer) + N_c(answer) + T_c(answer) + R_c(answer)
C_c(object) = O_c(object) + P_c(object)
```

当前已验证较强的是：

```text
T_c(answer) -> final logits
```

尚未验证的是：

```text
C_c(object) -> T_c(answer) -> final logits
```

因此完整链条应暂写为：

```text
object_state  --unclosed--> answer_transport_state -> output_gateway -> logits
```

中文解释：

```text
答案位置上的传输对齐状态具有强输出因果作用；
对象位置到答案位置的上游路径仍未闭合。
```

### 硬伤分析

1. **没有证明 object_last 单点移除足够打断路径**

object_last 可能只是路径起点之一，真实传输可能分布在多个层、多个 token、多个 attention head 中。

2. **monitor projection 不是完整 answer state**

本轮只监测 peak layer 的一个 T_c 投影。即使该投影下降，也不等于完整语义状态下降。

3. **patch at monitor layer 的 projection delta 记录有局限**

当 patch layer 等于 monitor layer 时，final logits 已改变，但记录的 hidden projection 可能显示 0，说明 hook 返回值与 hidden_states 记录顺序存在实现细节限制。

4. **wrong injection 未做目标错误类别释放分析**

wrong T_d 会压低目标，但还没有证明它提升了指定 wrong category。

5. **仍未做 generation audit**

目前仍是 DCF logits，不是开放生成闭包。

### 当前进展评价

Phase111 的结果不是对 Phase110 的否定，而是把结论变严格：

```text
Phase110 证明：T_c(answer) 是强因果成分。
Phase111 显示：object_last 单点 T_c patch 不能闭合到 final logits。
```

所以当前最可靠拼图是：

```text
读出前答案位置状态，是类别输出的关键因果位置；
对象位置上游路径仍未找到真正入口。
```

### 下一步任务

Phase112 应进入更细的路径搜索，而不是继续只做 residual stream 单点 patch：

```text
Attention Transport Head Mapping
```

目标：

```text
找出哪些 attention heads 把 object_last 信息写入 answer_last。
```

建议测试：

```text
1. 在 peak-3...peak 层记录 answer_last 对 object_last 的 attention 权重。
2. 对高权重 head 做 head output ablation。
3. 对高权重 head 做 object_last value patch。
4. 观察 answer_last T_c projection 与 final logits 是否同步变化。
5. 对 DS7B container/clothing/furniture/plant 与 Qwen3 plant/number 重点复测。
```

关键理由：

```text
如果真实路径是 attention transport，
那么 residual stream 的 object_last 单点 T_c 移除可能打不到真正写入 answer_last 的 head/value 通道。
```

## Phase 112: Attention Transport Head Mapping 注意力传输头定位 [2026-06-14 10:58]

### 本阶段目标

根据用户附加分析与 Phase111 结果，先判断：

```text
Phase111 的收缩是正确的：
T_c(answer) 是强因果读出前状态；
object_last 单点 residual patch 没有闭合到 logits。
```

附加分析中正确部分：

```text
1. 不应继续把 T_c 直接解释成已证明的 object_last -> answer_last 真实路径。
2. 下一步应从 residual direction 转向 attention route。
3. 需要测 answer_last 对 object/relation source 的 attention mass。
4. 需要做 head output ablation，而不只看注意力权重。
5. projection change 不等于 causal closure。
```

本阶段目标：

```text
定位哪些 attention heads 在 answer_last 读取 object source；
并测试这些 head 的单头消融是否降低 T_c(answer) 与 final logits。
```

### 执行命令

smoke：

```bash
python -m py_compile \
  tests/gpt5/phase112_attention_transport_head_mapping_cuda.py \
  tests/gpt5/phase112_attention_transport_head_mapping_summary.py

python tests/gpt5/phase112_attention_transport_head_mapping_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --layer-back 1 \
  --top-k-heads 2 \
  --categories number,time \
  --output-dir results/gpt5_phase112_smoke \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase112_attention_transport_head_mapping_cuda.py glm4 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --layer-back 1 \
  --top-k-heads 2 \
  --categories number,time \
  --output-dir results/gpt5_phase112_smoke \
  --hard-exit-after-model
```

正式测试：

```bash
python tests/gpt5/phase112_attention_transport_head_mapping_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --top-k-heads 8 \
  --output-dir results/gpt5_phase112_attention_transport_head_mapping \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase112_attention_transport_head_mapping_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --top-k-heads 8 \
  --output-dir results/gpt5_phase112_attention_transport_head_mapping \
  --hard-exit-after-model

python tests/gpt5/phase112_attention_transport_head_mapping_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --top-k-heads 8 \
  --output-dir results/gpt5_phase112_attention_transport_head_mapping \
  --hard-exit-after-model

python tests/gpt5/phase112_attention_transport_head_mapping_summary.py

python -m py_compile \
  tests/gpt5/phase112_attention_transport_head_mapping_cuda.py \
  tests/gpt5/phase112_attention_transport_head_mapping_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase112_attention_transport_head_mapping_cuda.py`
- 汇总脚本：`tests/gpt5/phase112_attention_transport_head_mapping_summary.py`
- Qwen3 结果：`results/gpt5_phase112_attention_transport_head_mapping/phase112_qwen3_attention_transport_head_mapping.json`
- GLM4 结果：`results/gpt5_phase112_attention_transport_head_mapping/phase112_glm4_attention_transport_head_mapping.json`
- DS7B 结果：`results/gpt5_phase112_attention_transport_head_mapping/phase112_deepseek7b_attention_transport_head_mapping.json`
- 跨模型汇总：`results/gpt5_phase112_attention_transport_head_mapping/phase112_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, time, container, clothing, furniture, plant
train objects/category = 12
heldout test objects/category = 12
templates = 4
prompts/category = 48
layers = peak-3 ... peak
selected heads/category = top 8 by answer_last attention to object_span + object_last
intervention = zero selected head slice at answer_last before o_proj
metrics = target_delta, release_delta, answer_last T_c projection delta
```

模型层位：

```text
Qwen3: monitor L35, scan/patch L32-L35, heads=32
GLM4: monitor L18, scan/patch L15-L18, heads=32
DS7B: monitor L27, scan/patch L24-L27, heads=28
```

### 测试原理

Phase112 分两步：

```text
1. attention source scan:
   对每个 head 记录 answer_last query 对 object_span/object_last/pre_object/post_object/self 的 attention mass。

2. head output ablation:
   在 o_proj 输入前，把该 head 在 answer_last 的 head slice 置零。
   然后测 final logits 和 answer_last T_c projection。
```

注意：

```text
attention mass 只用于候选选择；
真正因果判据来自 head ablation 后的 logits change。
```

### 客观结果

#### Qwen3

```text
number:
  top source head = L35 H21 object mass 0.057
  strongest target-down = L33 H24 target Δ -0.30, answer projection Δ -1.86
  strongest projection-down = L35 H8 answer projection Δ -4.83, target Δ +0.01

time:
  top source head = L33 H9 object mass 0.078
  strongest target-down = L35 H21 target Δ -0.02, answer projection Δ -2.15
  strongest projection-down = L35 H8 answer projection Δ -5.78, target Δ +0.00

container:
  top source head = L35 H27 object mass 0.093
  strongest target-down = L34 H21 target Δ -0.03
  strongest projection-down = L35 H8 answer projection Δ -4.73, target Δ +0.03

clothing:
  top source head = L33 H9 object mass 0.111
  strongest target-down = L33 H9 target Δ -0.07
  strongest projection-down = L35 H8 answer projection Δ -5.42, target Δ -0.02

furniture:
  top source head = L35 H21 object mass 0.101
  strongest target-down = L35 H28 target Δ -0.05
  strongest projection-down = L35 H8 answer projection Δ -4.85, target Δ -0.03

plant:
  top source head = L34 H21 object mass 0.117
  strongest target-down = L35 H27 target Δ -0.02
  strongest projection-down = L35 H21 answer projection Δ -2.18, target Δ +0.03
```

#### GLM4 bf16

```text
number/time/container/clothing/furniture/plant:
  top object-source attention heads exist, object mass roughly 0.12-0.16
  strongest target-down all near 0
  projection changes also near 0
```

GLM4 仍然不支持当前机制框架下的强结论。

#### DS7B

```text
number:
  top source head = L24 H17 object mass 0.174
  strongest target-down = L24 H22 target Δ -0.08, answer projection Δ -3.64

time:
  top source head = L25 H19 object mass 0.202
  strongest target-down = L24 H22 target Δ -0.06, answer projection Δ -6.62

container:
  top source head = L25 H19 object mass 0.228
  strongest target-down = L25 H15 target Δ -0.27, answer projection Δ -4.97
  strongest projection-down = L24 H22 answer projection Δ -5.84, target Δ -0.14

clothing:
  top source head = L25 H19 object mass 0.229
  strongest target-down = L24 H17 target Δ -0.40, answer projection Δ +0.62
  strongest projection-down = L25 H15 answer projection Δ -7.61, target Δ -0.02

furniture:
  top source head = L25 H19 object mass 0.273
  strongest target-down = L24 H2 target Δ -0.08
  strongest projection-down = L25 H15 answer projection Δ -6.33, target Δ +0.02

plant:
  top source head = L24 H6 object mass 0.311
  strongest target-down = L25 H24 target Δ -0.16
  strongest projection-down = L25 H15 answer projection Δ -6.65, target Δ +0.03
```

### 当前最可靠客观事实

1. **answer_last 确实会在 late layers 读取 object source**

DS7B 尤其明显：

```text
plant top object mass 0.311
furniture top object mass 0.273
clothing top object mass 0.229
container top object mass 0.228
```

Qwen3 也有较弱但可见的 object-source attention：

```text
plant 0.117
clothing 0.111
furniture 0.101
container 0.093
```

2. **单个高 object-source attention head 消融没有复现 Phase111 的强 target-down**

最强 target-down 仍很小：

```text
Qwen3 number: -0.30
DS7B clothing: -0.40
DS7B container: -0.27
```

这远弱于 Phase111 的 answer_last T_c removal：

```text
Qwen3 plant: -5.97
DS7B container: -5.50
DS7B clothing: -5.04
DS7B furniture: -3.82
```

3. **存在强 projection-only heads**

一些 head 消融会大幅降低 answer_last T_c projection，但 logits 几乎不变。

典型：

```text
Qwen3 L35 H8:
  number projection Δ -4.83, target Δ +0.01
  time projection Δ -5.78, target Δ +0.00
  container projection Δ -4.73, target Δ +0.03
  clothing projection Δ -5.42, target Δ -0.02
  furniture projection Δ -4.85, target Δ -0.03

DS7B L25 H15:
  clothing projection Δ -7.61, target Δ -0.02
  furniture projection Δ -6.33, target Δ +0.02
  plant projection Δ -6.65, target Δ +0.03
```

这再次证明：

```text
T_c projection change 不等于 logits causal closure。
```

4. **attention mass 不是因果强度**

高 object attention head 不一定有 target-down 效果。

例如：

```text
DS7B plant top source head L24 H6 object mass 0.311
但 strongest target-down 只有 -0.16
```

### 对 Phase111 的校正

Phase111 的判断继续成立：

```text
answer-site T_c 是强因果状态；
上游路径未闭合。
```

Phase112 进一步说明：

```text
单个 high object-attention head 不是足够的传输入口。
```

更严格说法：

```text
object source attention 存在；
但单头 answer_last output ablation 不能解释 answer-site T_c 的强 logits 因果效应。
```

因此上游路径可能是：

```text
1. 多头集合共同写入；
2. attention + MLP 接力；
3. value path 而非 head output 单点；
4. 多层 residual trajectory；
5. object_span/relation_span/template 多源共同构成。
```

### 条件化关系因子动力学公式更新

Phase111：

```text
object_state --unclosed--> answer_transport_state -> output_gateway -> logits
```

Phase112 后更细化为：

```text
source_tokens
  -> distributed_route_set
  -> A_c(answer)
  -> output_gateway
  -> logits
```

其中：

```text
distributed_route_set ≠ single high-attention head
A_c(answer) 包含强 causal state
projection(A_c, T_c) 不是充分因果指标
```

中文解释：

```text
对象源确实被答案位置读取；
但强类别因果状态不是由某一个明显高注意力头单独决定；
它更像多头、多层或注意力与 MLP 共同形成的答案位置状态。
```

### 硬伤分析

1. **只消融 top 8 object-source heads**

如果关键 head 不靠 object attention mass 排名，它可能被漏掉。

2. **只做单头消融**

强 T_c(answer) 可能由多个 head 累积写入，单头置零会低估。

3. **没有拆 Q/K/V**

本轮只在 o_proj 输入前置零 head slice，没有区分 attention pattern 与 value content。

4. **projection-only 现象仍未解释**

某些 head 强烈改变 T_c projection 但不改 logits，说明 T_c projection 本身不是完整因果状态。

5. **仍未做 generation audit**

尚未验证生成行为。

### 当前进展评价

Phase112 不是找到最终路径，而是排除了一个过于简单的假设：

```text
强 answer-site T_c 不是由单个高 object-attention head 直接控制。
```

当前最可靠拼图：

```text
1. answer_last 有强类别因果状态。
2. answer_last 确实读取 object source。
3. 单头 source attention 与 logits 因果之间不闭合。
4. projection-only heads 存在，投影不是充分指标。
```

### 下一步任务

Phase113 应测试：

```text
Head Set and MLP Relay Closure
```

目标：

```text
从单头转向 head set、多层累计与 MLP 接力，寻找能复现 Phase111 强 target-down 的最小路径集合。
```

建议测试：

```text
1. 对 top-k object-source heads 做 cumulative ablation。
2. 对 projection-only heads 与 source heads 分开/联合消融。
3. 对 attention output 与 MLP output 分别消融。
4. 测 answer_last T_c removal 与 head-set ablation 的 overlap。
5. 优先 DS7B container/clothing/furniture/plant 与 Qwen3 plant/number。
```

关键判据：

```text
如果 head set + MLP relay 能接近 Phase111 的 answer_last T_c remove 效果，
则上游路径开始闭合；
否则需要转向 residual trajectory / broader source span search。
```

## Phase 113: Head Set and MLP Relay Closure 注意力头集合与 MLP 接力闭包 [2026-06-14 11:27]

### 本阶段目标

根据用户附加分析与 Phase112 结果，先判断：

```text
Phase112 是正确的排除式进展：
object-source attention 存在；
但单个高 object-source head 不能解释 answer-site T_c 的强 logits 因果效应。
```

附加分析中正确部分：

```text
1. 单头不是基本单位，head set 可能才是基本单位。
2. attention mass 不是因果贡献。
3. projection-only heads 是重要现象，但不能直接解释成输出因果。
4. 下一步应测试 cumulative head-set ablation 与 MLP relay。
```

本阶段目标：

```text
测试 head set、MLP output、head set + MLP 是否能接近 Phase111 的 answer_last T_c removal 强效。
```

### 执行命令

smoke：

```bash
python -m py_compile \
  tests/gpt5/phase113_head_set_mlp_relay_closure_cuda.py \
  tests/gpt5/phase113_head_set_mlp_relay_closure_summary.py

python tests/gpt5/phase113_head_set_mlp_relay_closure_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --layer-back 1 \
  --candidate-heads 4 \
  --set-sizes 1,2,4 \
  --categories number,plant \
  --output-dir results/gpt5_phase113_smoke \
  --hard-exit-after-model
```

正式测试：

```bash
python tests/gpt5/phase113_head_set_mlp_relay_closure_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --candidate-heads 16 \
  --set-sizes 1,2,4,8,16 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase113_head_set_mlp_relay_closure \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase113_head_set_mlp_relay_closure_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --candidate-heads 16 \
  --set-sizes 1,2,4,8,16 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase113_head_set_mlp_relay_closure \
  --hard-exit-after-model

python tests/gpt5/phase113_head_set_mlp_relay_closure_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --candidate-heads 16 \
  --set-sizes 1,2,4,8,16 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase113_head_set_mlp_relay_closure \
  --hard-exit-after-model

python tests/gpt5/phase113_head_set_mlp_relay_closure_summary.py

python -m py_compile \
  tests/gpt5/phase113_head_set_mlp_relay_closure_cuda.py \
  tests/gpt5/phase113_head_set_mlp_relay_closure_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase113_head_set_mlp_relay_closure_cuda.py`
- 汇总脚本：`tests/gpt5/phase113_head_set_mlp_relay_closure_summary.py`
- Qwen3 结果：`results/gpt5_phase113_head_set_mlp_relay_closure/phase113_qwen3_head_set_mlp_relay_closure.json`
- GLM4 结果：`results/gpt5_phase113_head_set_mlp_relay_closure/phase113_glm4_head_set_mlp_relay_closure.json`
- DS7B 结果：`results/gpt5_phase113_head_set_mlp_relay_closure/phase113_deepseek7b_head_set_mlp_relay_closure.json`
- 跨模型汇总：`results/gpt5_phase113_head_set_mlp_relay_closure/phase113_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, clothing, plant
train objects/category = 12
heldout test objects/category = 12
templates = 4
prompts/category = 48
layers = peak-3 ... peak
candidate heads/category = 16
set sizes = 1, 2, 4, 8, 16
head sets = source, projection, target, mixed, random
relays = heads_only, mlp_only, heads_plus_mlp
reference = answer_last T_c removal, scale 1.5
```

模型层位：

```text
Qwen3: L32-L35
GLM4: L15-L18
DS7B: L24-L27
```

### 测试原理

每个类别先构造：

```text
T_c(answer)
```

并用 Phase111 的方式得到参考效应：

```text
answer_last T_c removal target_delta
```

然后选择候选 head：

```text
source heads:
  answer_last attention to object_span + object_last 最高的 heads。

projection heads:
  在候选池内，单头消融后 answer T_c projection 下降最多的 heads。

target heads:
  在候选池内，单头消融后 target logits 下降最多的 heads。

mixed heads:
  source + projection 的混合集合。

random heads:
  同规模随机对照。
```

干预：

```text
heads_only:
  在 o_proj 输入前，把 head set 在 answer_last 的 head slice 置零。

mlp_only:
  在 peak-3...peak 层，把 MLP output 在 answer_last 置零。

heads_plus_mlp:
  同时做 head set 消融和 MLP output 消融。
```

关键指标：

```text
effect_ratio = head_set_target_delta / T_c_remove_target_delta
```

### 客观结果

#### Qwen3

```text
number:
  T_c reference Δ -3.43
  best heads_only Δ -0.33, ratio 0.10
  best heads_plus_mlp Δ +4.00
  best mlp_only Δ +4.18
  random heads_only Δ -0.02

container:
  T_c reference Δ -1.75
  best heads_only Δ -0.15, ratio 0.09
  best heads_plus_mlp Δ +2.39
  best mlp_only Δ +2.63
  random heads_only Δ -0.01

clothing:
  T_c reference Δ -1.43
  best heads_only Δ -0.72, ratio 0.50
  best random heads_only Δ -0.35, ratio 0.25
  best heads_plus_mlp Δ +1.58
  best mlp_only Δ +2.49

plant:
  T_c reference Δ -5.97
  best heads_only Δ -0.59, ratio 0.10
  best heads_plus_mlp Δ +3.07
  best mlp_only Δ +3.48
  random heads_only Δ -0.02
```

Qwen3 中只有 clothing 出现局部闭合线索：

```text
heads_only ratio 0.50
random ratio 0.25
```

但这仍不能解释多数类别。

#### GLM4 bf16

```text
T_c reference 本身很弱：
number Δ -0.09
container Δ -0.07
clothing Δ -0.07
plant Δ +0.02
```

因此 GLM4 本轮不进入强机制结论。

#### DS7B

```text
number:
  T_c reference Δ +1.06
  reference 不是 target-down，因此不适合闭合判据。

container:
  T_c reference Δ -5.50
  best heads_only Δ -0.28, ratio 0.05
  best heads_plus_mlp Δ +0.34
  best mlp_only Δ +0.14
  random heads_only Δ -0.15

clothing:
  T_c reference Δ -5.04
  best heads_only Δ -0.78, ratio 0.16
  best random heads_only Δ -0.45, ratio 0.09
  best heads_plus_mlp Δ +1.39
  best mlp_only Δ +1.44

plant:
  T_c reference Δ -3.20
  best heads_only Δ -0.32, ratio 0.10
  best heads_plus_mlp Δ +0.55
  best mlp_only Δ +0.92
  random heads_only Δ -0.28
```

DS7B 的 head set 仍不能接近 T_c reference。

### 当前最可靠客观事实

1. **head set 消融比单头稍强，但大多数仍不能闭合**

典型：

```text
Qwen3 plant:
  T_c reference -5.97
  heads_only -0.59

DS7B container:
  T_c reference -5.50
  heads_only -0.28

DS7B clothing:
  T_c reference -5.04
  heads_only -0.78
```

2. **Qwen3 clothing 是局部例外**

```text
Qwen3 clothing:
  T_c reference -1.43
  target head set -0.72
  ratio 0.50
  random -0.35
```

这说明某些类别可能确实有 head-set 局部闭合，但不是普遍结构。

3. **coarse MLP output ablation 不支持 MLP relay 闭合**

MLP ablation 常常产生 target-up，而不是复现 T_c target-down：

```text
Qwen3 number mlp_only +4.18
Qwen3 plant mlp_only +3.48
DS7B clothing mlp_only +1.44
DS7B plant mlp_only +0.92
```

同时 answer projection 出现巨大变化：

```text
Qwen3 mlp_only answer projection Δ around -154 to -199
DS7B mlp_only answer projection Δ around -303 to -328
```

这说明粗 MLP 置零是强破坏，不是干净机制分解。

4. **projection change 继续不是充分因果指标**

很多条件 answer projection 大幅变化，但 target logits 不按 T_c reference 方向变化。

5. **GLM4 继续弱参考**

GLM4 的 T_c reference 太小，本轮不支持强机制结论。

### 对 Phase112 的校正

Phase112 的正确部分仍成立：

```text
单个高 object-source head 不是完整路径。
```

Phase113 进一步说明：

```text
top source/projection/target head set 也大多不是完整路径；
coarse MLP output ablation 也没有形成闭合。
```

更严格说法：

```text
answer-site T_c 的强因果效应，不能由当前 tested head-set + coarse MLP relay 解释。
```

因此当前路径可能在：

```text
1. 更宽的 residual trajectory；
2. 非 top-attention 的 value-content heads；
3. MLP 内部子方向，而非整个 MLP output；
4. 多层小尺度分布式累积；
5. answer-site 子空间而非单方向 T_c。
```

### 条件化关系因子动力学公式更新

Phase112：

```text
source_tokens -> distributed_route_set -> A_c(answer) -> output_gateway -> logits
```

Phase113 后应更谨慎：

```text
source_tokens
  -> unresolved_distributed_dynamics
  -> A_c(answer)
  -> output_gateway
  -> logits
```

其中：

```text
unresolved_distributed_dynamics
  不等于 single head
  不等于 tested top-k head set
  不等于 coarse whole-MLP output ablation
```

当前强验证仍是：

```text
A_c(answer) / T_c(answer) -> logits
```

未闭合部分仍是：

```text
source_tokens -> A_c(answer)
```

### 硬伤分析

1. **MLP ablation 太粗**

把整层 MLP output 在 answer_last 置零，会破坏大量非目标功能，不能说明 MLP 子方向机制。

2. **projection heads 仍来自 source candidate pool**

如果真正 projection heads 不在 top source candidate 中，仍可能漏掉。

3. **没有 Q/K/V value transplant**

仍未测试 value content 是否是关键。

4. **没有 answer-site 多维子空间**

T_c 是单方向；强因果场可能是多维子空间。

5. **没有 generation audit**

仍未验证生成行为。

### 当前进展评价

Phase113 是第二次排除式进展：

```text
单头不够；
top-k head set 大多也不够；
coarse MLP relay 也不够。
```

当前最可靠拼图：

```text
1. answer-site T_c 是强因果入口。
2. object-source attention 存在。
3. 单头与 top-k head set 多数不能闭合。
4. MLP 整体置零不是正确分解粒度。
5. Qwen3 clothing 有局部 head-set 线索。
```

### 下一步任务

Phase114 应转向：

```text
Answer-Site Causal Subspace Expansion
```

目标：

```text
不要再把 answer-site causal field 压缩成单方向 T_c；
构造多维 answer-site causal subspace，再测试子空间移除是否比单方向更稳定、更接近真实机制。
```

建议测试：

```text
1. 从多个强类别和多个模板中提取 answer-site causal directions。
2. 构造低秩子空间 rank 2/4/8/16。
3. 在 answer_last 移除整个子空间，和单方向 T_c 对照。
4. 测 target_delta、competitor release、random subspace control。
5. 优先 Qwen3 number/plant/clothing 与 DS7B container/clothing/plant。
```

关键理由：

```text
projection-only、head-set 不闭合、MLP 粗消融反向，
都说明当前单方向 T_c 只是强因果场的一个切片；
破解路径前，必须先把 answer-site causal field 的维度结构拼出来。
```

## Phase 114: Answer-Site Causal Subspace Expansion 答案位置因果子空间扩展 [2026-06-14 12:09]

### 本阶段目标

根据用户附加分析与 Phase113 结果，先判断：

```text
Phase113 的收缩是正确的：
单头不够；
top-k head set 大多不够；
coarse MLP relay 不够；
T_c(answer) 可能只是 answer-site causal field 的一维切片。
```

本阶段目标：

```text
构造 rank 1/2/4/8/16 的 answer-site category contrast subspace，
测试多维子空间移除是否比单方向 T_c 更稳定或更强。
```

### 执行命令

smoke：

```bash
python -m py_compile \
  tests/gpt5/phase114_answer_site_causal_subspace_cuda.py \
  tests/gpt5/phase114_answer_site_causal_subspace_summary.py

python tests/gpt5/phase114_answer_site_causal_subspace_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --ranks 1,2 \
  --scales 1.0 \
  --categories number,plant \
  --output-dir results/gpt5_phase114_smoke \
  --hard-exit-after-model
```

正式测试：

```bash
python tests/gpt5/phase114_answer_site_causal_subspace_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --ranks 1,2,4,8,16 \
  --scales 1.0,1.5 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase114_answer_site_causal_subspace \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase114_answer_site_causal_subspace_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --ranks 1,2,4,8,16 \
  --scales 1.0,1.5 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase114_answer_site_causal_subspace \
  --hard-exit-after-model

python tests/gpt5/phase114_answer_site_causal_subspace_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --ranks 1,2,4,8,16 \
  --scales 1.0,1.5 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase114_answer_site_causal_subspace \
  --hard-exit-after-model

python tests/gpt5/phase114_answer_site_causal_subspace_summary.py

python -m py_compile \
  tests/gpt5/phase114_answer_site_causal_subspace_cuda.py \
  tests/gpt5/phase114_answer_site_causal_subspace_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase114_answer_site_causal_subspace_cuda.py`
- 汇总脚本：`tests/gpt5/phase114_answer_site_causal_subspace_summary.py`
- Qwen3 结果：`results/gpt5_phase114_answer_site_causal_subspace/phase114_qwen3_answer_site_causal_subspace.json`
- GLM4 结果：`results/gpt5_phase114_answer_site_causal_subspace/phase114_glm4_answer_site_causal_subspace.json`
- DS7B 结果：`results/gpt5_phase114_answer_site_causal_subspace/phase114_deepseek7b_answer_site_causal_subspace.json`
- 跨模型汇总：`results/gpt5_phase114_answer_site_causal_subspace/phase114_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, clothing, plant
train objects/category = 12
heldout test objects/category = 12
templates = 4
prompts/category = 48
ranks = 1, 2, 4, 8, 16
scales = 1.0, 1.5
layer = model-specific causal peak
controls = random same-rank subspace
```

模型层位：

```text
Qwen3: L35
GLM4: L18
DS7B: L27
```

### 测试原理

对每个类别，在 answer-site peak layer 构造 target-vs-other contrast rows：

```text
每个 template:
  target_center - other_mean
  target_center - each_other_category_center
```

然后对这些 contrast rows 做 SVD，取：

```text
rank = 1, 2, 4, 8, 16
```

得到 answer-site category contrast subspace。

真实 forward 干预：

```text
在 answer_last 移除该子空间投影：
  h = h - scale * proj_subspace(h)
```

对照：

```text
1. 单方向 T_c removal
2. same-rank random subspace removal
```

指标：

```text
target_delta
max_other_release_delta
random control target_delta
```

### 客观结果

#### Qwen3

```text
number:
  T_c: r1 scale1.5 target Δ -3.43, release +0.87
  best subspace: rank2 scale1.5 target Δ -3.12, release +0.78
  random: rank16 scale1.5 target Δ -0.50

container:
  T_c: target Δ -1.74, release +0.12
  best subspace: rank16 scale1.5 target Δ -2.59, release +2.03
  random: target Δ -0.12

clothing:
  T_c: target Δ -1.42, release +1.12
  best subspace: rank8 scale1.5 target Δ -0.47, release +0.69
  random: target Δ -0.04

plant:
  T_c: target Δ -5.98, release +0.73
  best subspace: rank2 scale1.5 target Δ -1.26, release +0.00
  random: target Δ -0.21
```

Qwen3 结果是混合的：

```text
number: subspace 接近 T_c
container: subspace 稍强但 release 很大
clothing/plant: 单方向 T_c 更强
```

#### GLM4 bf16

```text
number:
  T_c -0.10
  subspace rank16 -0.86, release +1.22

container:
  T_c -0.08
  subspace rank16 -0.53

clothing:
  T_c -0.07
  subspace rank8 -0.34

plant:
  T_c +0.01
  subspace rank16 -0.13
```

GLM4 子空间有一些变化，但 T_c reference 本身弱，仍不作为强机制结论来源。

#### DS7B

```text
number:
  T_c: target Δ +1.11, release +1.22
  best subspace: rank16 scale1.5 target Δ -11.75, release +0.00
  random: target Δ -0.30

container:
  T_c: target Δ -5.60, release +0.00
  best subspace: rank16 scale1.5 target Δ -12.42, release +0.00
  random: target Δ -0.30

clothing:
  T_c: target Δ -5.22, release +0.18
  best subspace: rank8 scale1.5 target Δ -4.99, release +0.00
  random: target Δ -0.23

plant:
  T_c: target Δ -3.19, release +0.00
  best subspace: rank8 scale1.5 target Δ -7.93, release +0.00
  random: target Δ -0.43
```

DS7B 出现强正向结果：

```text
number/container/plant 的 answer-site 多维子空间显著强于单方向 T_c，
并且远强于 random subspace。
```

### 当前最可靠客观事实

1. **DS7B 的 answer-site causal field 明显是多维结构**

最强例子：

```text
DS7B container:
  T_c -5.60
  rank16 subspace -12.42
  random -0.30

DS7B plant:
  T_c -3.19
  rank8 subspace -7.93
  random -0.43

DS7B number:
  T_c +1.11
  rank16 subspace -11.75
  random -0.30
```

这说明 DS7B 的单方向 T_c 确实只是答案位置因果场的切片。

2. **Qwen3 更类别分化**

```text
Qwen3 number:
  T_c -3.43
  subspace -3.12

Qwen3 container:
  T_c -1.74
  subspace -2.59 but release +2.03

Qwen3 plant:
  T_c -5.98
  subspace -1.26
```

Qwen3 plant 仍然是强单方向模式。

3. **random subspace control 很弱**

典型：

```text
DS7B container random -0.30 vs subspace -12.42
DS7B plant random -0.43 vs subspace -7.93
Qwen3 number random -0.50 vs T_c/subspace around -3
```

说明强效不是单纯 rank 高或随机删除造成。

4. **container 类子空间 release 需要谨慎**

Qwen3 container：

```text
subspace target Δ -2.59
release +2.03
```

这说明该子空间混入强竞争释放，不是纯 target support。

### 对 Phase113 的校正

Phase113 的排除式判断仍正确：

```text
head set / coarse MLP relay 没有解释 T_c(answer) 强效。
```

Phase114 给出新的正向方向：

```text
尤其在 DS7B，answer-site causal field 不是单方向，而是低秩多维子空间。
```

因此上游路径未闭合，不一定是因为没有路由，而可能是因为我们追踪的目标状态维度太窄：

```text
source_tokens -> A_c(answer)
```

其中 `A_c(answer)` 应从单方向 `T_c` 改为多维子空间。

### 条件化关系因子动力学公式更新

Phase113：

```text
source_tokens
  -> unresolved_distributed_dynamics
  -> A_c(answer)
  -> output_gateway
  -> logits
```

Phase114 后：

```text
A_c(answer) ∈ Subspace_c^k(answer)
```

更具体：

```text
source_tokens
  -> unresolved_distributed_dynamics
  -> Subspace_c^k(answer)
  -> output_gateway
  -> logits
```

其中：

```text
T_c 是 Subspace_c^k 的一个强切片；
但在 DS7B 中，rank8/rank16 子空间比 T_c 更接近完整因果场。
```

中文解释：

```text
答案位置的类别因果状态不是一个方向，而可能是一个低秩子空间；
不同模型和类别的有效维度不同；
破解路径前，必须先确定要追踪的答案位置状态空间。
```

### 硬伤分析

1. **子空间来自类别几何，不是自动因果发现**

当前 subspace 是 target-vs-other answer center contrast 的 SVD，不等于已证明的最小因果子空间。

2. **random control 没有匹配谱结构**

random subspace 只匹配 rank，没有匹配奇异值谱或与 readout/transport 的夹角。

3. **高 rank 可能混入竞争释放**

Qwen3 container release +2.03 说明子空间会包含竞争/抑制成分。

4. **仍是 DCF logits**

没有 generation audit。

5. **上游路径仍未闭合**

本轮定位的是 answer-site field，不是 source -> answer 路径。

### 当前进展评价

Phase114 是关键正向进展：

```text
首次明确显示 answer-site causal field 在 DS7B 中是多维低秩结构；
并且多维子空间远强于随机子空间。
```

当前最可靠拼图：

```text
1. answer-site causal field 是核心因果入口。
2. DS7B 的 answer-site field 是多维结构。
3. Qwen3 存在类别分化：number 接近多维/单向都可，plant 强单向。
4. T_c 不是完整因果状态，只是某些模型/类别的强切片。
5. 上游路径搜索应改为追踪 Subspace_c^k(answer)，而不是单方向 T_c。
```

### 下一步任务

Phase115 应做：

```text
Causal Subspace Robustness and Release Decomposition
```

目标：

```text
验证 Phase114 的多维子空间是否稳定，并把 target support 与 competitor release 拆开。
```

建议测试：

```text
1. 对 DS7B number/container/plant 扩大 heldout objects 复测。
2. 对 rank8/rank16 子空间做 scale sweep: 0.25,0.5,1.0,1.5。
3. 对子空间做 leave-template-out 验证，确认不是模板过拟合。
4. 对 release 强的 Qwen3 container 做 target-support / release-component 分解。
5. 加 matched-spectrum random subspace control。
```

关键判据：

```text
如果 DS7B rank8/rank16 子空间在模板留出、扩大对象、matched random control 下仍强，
则可以把 answer-site causal field 从“单方向假设”正式升级为“低秩因果子空间”。
```

## Phase 115: Causal Subspace Robustness and Release Decomposition 因果子空间稳健性与释放分解 [2026-06-14 13:16]

### 本阶段目标

根据 Phase114 的结果继续验证：

```text
DS7B 的 answer-site causal field 是否真是稳健低秩子空间；
Qwen3 的子空间 target-down 是否混入 competitor release；
Phase114 强效是否会在扩大 heldout objects、leave-template-out、matched-spectrum random control 下保留。
```

### 执行命令

smoke：

```bash
python -m py_compile \
  tests/gpt5/phase115_causal_subspace_robustness_cuda.py \
  tests/gpt5/phase115_causal_subspace_robustness_summary.py

python tests/gpt5/phase115_causal_subspace_robustness_cuda.py qwen3 \
  --train-objects 4 \
  --test-objects 4 \
  --batch-size 4 \
  --ranks 2 \
  --scales 0.5 \
  --categories number,container \
  --output-dir results/gpt5_phase115_smoke \
  --hard-exit-after-model
```

正式测试：

```bash
python tests/gpt5/phase115_causal_subspace_robustness_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --ranks 8,16 \
  --scales 0.25,0.5,1.0,1.5 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase115_causal_subspace_robustness \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase115_causal_subspace_robustness_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --ranks 8,16 \
  --scales 0.25,0.5,1.0,1.5 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase115_causal_subspace_robustness \
  --hard-exit-after-model

python tests/gpt5/phase115_causal_subspace_robustness_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --ranks 8,16 \
  --scales 0.25,0.5,1.0,1.5 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase115_causal_subspace_robustness \
  --hard-exit-after-model

python tests/gpt5/phase115_causal_subspace_robustness_summary.py

python -m py_compile \
  tests/gpt5/phase115_causal_subspace_robustness_cuda.py \
  tests/gpt5/phase115_causal_subspace_robustness_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase115_causal_subspace_robustness_cuda.py`
- 汇总脚本：`tests/gpt5/phase115_causal_subspace_robustness_summary.py`
- Qwen3 结果：`results/gpt5_phase115_causal_subspace_robustness/phase115_qwen3_causal_subspace_robustness.json`
- GLM4 结果：`results/gpt5_phase115_causal_subspace_robustness/phase115_glm4_causal_subspace_robustness.json`
- DS7B 结果：`results/gpt5_phase115_causal_subspace_robustness/phase115_deepseek7b_causal_subspace_robustness.json`
- 跨模型汇总：`results/gpt5_phase115_causal_subspace_robustness/phase115_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, clothing, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
full prompts/category = 64
ranks = 8, 16
scales = 0.25, 0.5, 1.0, 1.5
layer = model-specific causal peak
controls = matched-spectrum random subspace
robustness = leave-template-out 4 folds
release decomposition = strongest-release-category excluded contrast
```

模型层位：

```text
Qwen3: L35
GLM4: L18
DS7B: L27
```

### 测试原理

Phase115 在 Phase114 基础上做四类验证：

```text
1. 扩大 heldout:
   train objects/category = 8
   test objects/category = 16

2. scale sweep:
   0.25, 0.5, 1.0, 1.5

3. leave-template-out:
   用 3 个模板构造子空间；
   在第 4 个模板上测试；
   4 个模板轮换。

4. matched-spectrum random:
   用 synthetic contrast matrix 保留奇异值谱，再生成随机子空间。
```

release 分解的初步版本：

```text
先找 full subspace 的最强 release category；
再从 contrast construction 中排除该 release category；
测试 release-excluded subspace。
```

注意：这还不是完整 support/release factorization，只是第一层排查。

### 客观结果

#### Qwen3

```text
number:
  full subspace: r8 scale1.5 target Δ -1.83, release +2.37
  matched random: target Δ +0.05
  release-excluded: target Δ -2.72, release +1.99
  LTO mean: target Δ -2.18, release +2.12
  LTO random mean: target Δ -0.19

container:
  full subspace: r16 scale1.5 target Δ -2.53, release +1.90
  matched random: target Δ +0.03
  release-excluded: target Δ -3.06, release +1.97
  LTO mean: target Δ -2.54, release +1.54
  LTO random mean: target Δ -0.07

clothing:
  full subspace: target Δ +0.22, release +0.51
  matched random: target Δ -0.38
  LTO mean: target Δ +0.24
  LTO random mean: target Δ -0.30

plant:
  full subspace: r16 scale1.5 target Δ -1.24, release +1.59
  matched random: target Δ -0.15
  release-excluded: target Δ -1.17, release +1.77
  LTO mean: target Δ -1.74, release +1.41
```

Qwen3 结论：

```text
number/container/plant 的子空间效应能跨模板保留，但 release 很大；
clothing 对照敏感；
release-excluded 没有解决 release，说明 release 不是单一类别导致。
```

#### GLM4 bf16

```text
number:
  full subspace Δ -0.90, release +0.68
  LTO mean Δ -0.58

container:
  full subspace Δ -0.32
  LTO mean Δ -0.36

clothing:
  full subspace Δ -0.28
  LTO mean Δ -0.19

plant:
  full subspace Δ -0.13
  LTO mean Δ -0.07
```

GLM4 仍然弱，但 number 出现小幅稳定信号。

#### DS7B

```text
number:
  full subspace: r16 scale1.5 target Δ -12.58, release +0.00
  matched random: target Δ -0.07
  LTO mean: target Δ -11.59, release +0.00
  LTO random mean: target Δ -0.20

container:
  full subspace: r16 scale1.5 target Δ -12.52, release +0.00
  matched random: target Δ -0.24
  LTO mean: target Δ -11.45, release +0.00
  LTO random mean: target Δ -0.37

clothing:
  full subspace: r8 scale1.0 target Δ -4.20, release +0.00
  matched random: target Δ -0.06
  LTO mean: target Δ -5.07, release +0.00
  LTO random mean: target Δ -0.37

plant:
  full subspace: r8 scale1.5 target Δ -9.40, release +0.00
  matched random: target Δ -0.29
  LTO mean: target Δ -8.71, release +0.00
  LTO random mean: target Δ -0.22
```

DS7B 结论：

```text
number/container/plant 是 robust_strong；
clothing 是 robust_moderate；
所有 matched-spectrum random controls 都很弱；
所有 release 都为 0。
```

### 当前最可靠客观事实

1. **DS7B answer-site low-rank causal subspace 已通过稳健性测试**

最强证据：

```text
DS7B number:
  full -12.58
  LTO mean -11.59
  random -0.07

DS7B container:
  full -12.52
  LTO mean -11.45
  random -0.24

DS7B plant:
  full -9.40
  LTO mean -8.71
  random -0.29
```

这说明 DS7B 的低秩子空间不是模板过拟合，也不是 rank/random 删除造成。

2. **DS7B 子空间几乎无 competitor release**

```text
number/container/clothing/plant:
  max_other_release = 0.00 in full and LTO mean
```

说明 DS7B 的子空间更像干净 target support removal。

3. **Qwen3 子空间混有强 release**

```text
Qwen3 number release +2.37
Qwen3 container release +1.90
Qwen3 plant release +1.59
```

release-excluded 后仍然 release 较高：

```text
number +1.99
container +1.97
plant +1.77
```

这说明 Qwen3 的 release 不是一个竞争类别造成，而是多竞争/接口混合结构。

4. **GLM4 仍然弱，但 number 有小信号**

```text
GLM4 number:
  full -0.90
  LTO -0.58
  random -0.02
```

仍不能与 DS7B 强结论同等对待。

### 理论进展

Phase115 支持把 DS7B 的 answer-site 表述升级为：

```text
Subspace_c^k(answer) 是稳健因果状态。
```

更具体：

```text
source_tokens
  -> unresolved_distributed_dynamics
  -> robust low-rank Subspace_c^k(answer)
  -> output_gateway
  -> logits
```

对 DS7B：

```text
number/container/plant:
  k ≈ 8-16
  robust across heldout objects and heldout templates
  target-down strong
  competitor release near zero
```

对 Qwen3：

```text
answer-site subspace 是 mixed support/release field；
需要进一步拆 support 与 release。
```

### 硬伤分析

1. **matched-spectrum random 仍使用 orthonormal basis 干预**

虽然通过 synthetic contrast matrix 匹配奇异值谱，但最终移除的是正交基投影，谱结构只影响基的生成过程。

2. **release-excluded 不是完整分解**

只排除最强 release category，无法分解多竞争释放。

3. **仍然没有生成审计**

目前仍是 DCF logits。

4. **上游路径仍未闭合**

Phase115 证明 answer-site 子空间稳健，但没有解释 source 如何写入该子空间。

5. **Qwen3 与 DS7B 机制分型明显不同**

不能把 DS7B 的干净低秩子空间结论直接套到 Qwen3。

### 当前进展评价

Phase115 是一次强确认：

```text
DS7B 的 answer-site low-rank causal subspace 已从“可能结构”升级为“稳健客观事实”。
```

当前最可靠拼图：

```text
1. DS7B number/container/plant 存在稳健、干净、低秩的 answer-site 因果子空间。
2. DS7B clothing 也有中强稳健子空间。
3. Qwen3 的 answer-site 子空间存在，但混入强 release。
4. GLM4 仍弱，只能作为小信号参考。
5. 下一步应从“是否有子空间”转向“子空间内部成分如何分解”。
```

### 下一步任务

Phase116 应做：

```text
Subspace Basis Component Audit
```

目标：

```text
把稳健低秩子空间拆成 rank component，
确定哪些基向量负责 target support，
哪些基向量负责 release/interface，
哪些是冗余或控制维度。
```

建议测试：

```text
1. 对 DS7B number/container/plant 的 rank16 子空间逐基向量 ablation。
2. 对 rank16 做 cumulative basis ablation: top1, top2, top4, top8, top16。
3. 对 Qwen3 number/container/plant 做 basis-level release decomposition。
4. 对每个 basis component 记录 target_delta、release_delta、readout cosine、transport cosine。
5. 加 matched random basis component control。
```

关键判据：

```text
如果少数 basis components 能复现大部分 target-down，
则子空间可以继续压缩；
如果必须 top8/top16 累积才强，
说明答案位置因果场确实是分布式低秩结构。
```

## Phase 116: Subspace Basis Component Audit 子空间基向量成分审计 [2026-06-14 13:28]

### 本阶段目标

根据用户附加分析与 Phase115 结果，继续完成：

```text
把稳健低秩 answer-site causal subspace 拆成 basis components；
确定哪些基向量负责 target support；
哪些负责 competitor release / interface；
以及完整强效是否来自少数基向量或 top-k 累积。
```

### 执行命令

smoke：

```bash
python -m py_compile \
  tests/gpt5/phase116_subspace_basis_component_audit_cuda.py \
  tests/gpt5/phase116_subspace_basis_component_audit_summary.py

python tests/gpt5/phase116_subspace_basis_component_audit_cuda.py qwen3 \
  --train-objects 4 \
  --test-objects 4 \
  --batch-size 4 \
  --ranks 4 \
  --set-sizes 1,2,4 \
  --categories number,container \
  --output-dir results/gpt5_phase116_smoke \
  --hard-exit-after-model
```

正式测试：

```bash
python tests/gpt5/phase116_subspace_basis_component_audit_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --ranks 8,16 \
  --set-sizes 1,2,4,8,16 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase116_subspace_basis_component_audit \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase116_subspace_basis_component_audit_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --ranks 8,16 \
  --set-sizes 1,2,4,8,16 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase116_subspace_basis_component_audit \
  --hard-exit-after-model

python tests/gpt5/phase116_subspace_basis_component_audit_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --ranks 8,16 \
  --set-sizes 1,2,4,8,16 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase116_subspace_basis_component_audit \
  --hard-exit-after-model

python tests/gpt5/phase116_subspace_basis_component_audit_summary.py

python -m py_compile \
  tests/gpt5/phase116_subspace_basis_component_audit_cuda.py \
  tests/gpt5/phase116_subspace_basis_component_audit_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase116_subspace_basis_component_audit_cuda.py`
- 汇总脚本：`tests/gpt5/phase116_subspace_basis_component_audit_summary.py`
- Qwen3 结果：`results/gpt5_phase116_subspace_basis_component_audit/phase116_qwen3_subspace_basis_component_audit.json`
- GLM4 结果：`results/gpt5_phase116_subspace_basis_component_audit/phase116_glm4_subspace_basis_component_audit.json`
- DS7B 结果：`results/gpt5_phase116_subspace_basis_component_audit/phase116_deepseek7b_subspace_basis_component_audit.json`
- 跨模型汇总：`results/gpt5_phase116_subspace_basis_component_audit/phase116_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, clothing, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
ranks = 8, 16
scale = 1.5
cumulative set sizes = 1, 2, 4, 8, 16
metrics = target_delta, max_release_delta, readout_cos, transport_cos, template_abs_cos
```

模型层位：

```text
Qwen3: L35
GLM4: L18
DS7B: L27
```

### 测试原理

对每个类别构造 answer-site contrast subspace：

```text
target_center - other_mean
target_center - each_other_category_center
```

SVD 后取 rank8/rank16 basis。

测试三类干预：

```text
1. basis-wise ablation:
   单独移除每个 basis vector。

2. cumulative basis ablation:
   按单基 target_delta 从强到弱排序，测试 top1/top2/top4/top8/top16。

3. split sets:
   根据单基效果标注 support / release / mixed / weak，
   分别移除这些集合。
```

基向量诊断：

```text
readout_cos
transport_cos
template_abs_cos
```

### 客观结果

#### Qwen3

```text
number:
  best single: basis0 target Δ -2.90, release +1.59
  best cumulative rank16: top4 target Δ -3.67, release +0.86
  release set: target Δ +0.70, release +1.77
  mixed set: target Δ -3.10, release +1.05
  best random single: target Δ -0.28

container:
  best single: basis1 target Δ -0.66, release +0.00
  support set rank16: target Δ -1.65, release +0.00
  release set rank16: target Δ +0.24, release +1.88
  mixed set: target Δ -0.46, release +2.05
  cumulative rank16 top8: target Δ -2.66, release +0.40

clothing:
  best single: basis1 target Δ -0.72, release +0.00
  support set: target Δ -1.03, release +0.00
  release set rank16: target Δ +2.00, release +2.64
  cumulative rank16 top8: target Δ -1.62, release +0.00

plant:
  best single: basis1 target Δ -1.02, release +0.00
  support set rank16: target Δ -1.39, release +0.00
  release set rank16: target Δ +1.09, release +1.62
  cumulative rank16 top8: target Δ -2.85, release +0.00
```

Qwen3 关键事实：

```text
1. release basis 可以直接分离出来。
2. support set 往往 target-down 干净。
3. number 的最强单基是 mixed，不是干净 support。
4. container/clothing/plant 都出现 clear support/release split。
```

#### GLM4 bf16

```text
number:
  best single target Δ -0.37
  cumulative rank16 top16 target Δ -0.90
  support set target Δ -0.60

container:
  best single target Δ -0.17
  cumulative rank16 top8 target Δ -0.46

clothing:
  best single target Δ -0.11
  cumulative rank16 top8 target Δ -0.44

plant:
  best single target Δ -0.07
  cumulative rank16 top8 target Δ -0.21
```

GLM4 仍弱。

#### DS7B

```text
number:
  best single: basis1 target Δ -5.55, release +0.00
  rank16 cumulative top16 target Δ -12.58, release +0.00
  support set rank16: target Δ -12.49, release +0.00
  release set rank16: target Δ +0.62, release +1.83
  best random single: target Δ -0.13

container:
  best single: basis6 target Δ -2.92, release +0.00
  rank16 cumulative top8 target Δ -13.55, release +0.00
  support set rank16: target Δ -13.55, release +0.00
  release set: target Δ -0.16, release +1.22
  best random single: target Δ -0.17

clothing:
  best single: basis0 target Δ -3.44, release +0.00
  rank16 cumulative top4 target Δ -5.31, release +0.00
  support set rank16: target Δ -5.58, release +0.00
  release set rank16: target Δ +2.07, release +1.67
  mixed set: target Δ -0.60, release +0.70

plant:
  best single: basis0 target Δ -4.93, release +0.00
  rank16 cumulative top8 target Δ -9.71, release +0.00
  support set rank16: target Δ -9.66, release +0.00
  release set: target Δ +0.17, release +0.53
  best random single: target Δ -0.14
```

DS7B 关键事实：

```text
1. 存在强单基 support component。
2. 完整强效仍需要多个 support basis 累积。
3. support set 非常干净，release=0。
4. release components 也存在，但不是 full subspace 强 target-down 的主要来源。
```

### 当前最可靠客观事实

1. **DS7B 是 clean distributed support subspace**

例如：

```text
container:
  single -2.92
  support set -13.55

plant:
  single -4.93
  support set -9.66

number:
  single -5.55
  support set -12.49
```

这说明少数强 basis 很重要，但完整效果需要多个 support basis。

2. **Qwen3 的 support/release 可在 basis level 分离**

典型：

```text
Qwen3 container:
  support set -1.65, release 0
  release set +0.24, release +1.88
  mixed set -0.46, release +2.05

Qwen3 clothing:
  support set -1.03, release 0
  release set +2.00, release +2.64

Qwen3 plant:
  support set -1.39, release 0
  release set +1.09, release +1.62
```

Phase115 中 Qwen3 的大 release，在 Phase116 被拆到了具体 basis sets。

3. **随机单基对照很弱**

典型：

```text
DS7B number random single -0.13 vs real single -5.55
DS7B plant random single -0.14 vs real single -4.93
Qwen3 number random single -0.28 vs real single -2.90
```

4. **readout/transport/template cos 都不高**

许多最强单基的 cos 仍低：

```text
DS7B number best single:
  readout_cos -0.06
  transport_cos -0.20
  template_abs_cos 0.35

DS7B container best single:
  readout_cos 0.00
  transport_cos 0.15
  template_abs_cos 0.13
```

说明强 causal basis 不是简单 readout/transport/template 方向。

### 理论进展

Phase115：

```text
Subspace_c^k(answer) 是稳健因果状态。
```

Phase116 后可进一步拆成：

```text
Subspace_c^k(answer)
=
SupportBasisSet_c
+ ReleaseBasisSet_c
+ MixedBasisSet_c
+ Weak/RedundantBasisSet_c
```

对 DS7B：

```text
SupportBasisSet_c 是主导；
release basis 存在但不主导；
target-down 几乎无 competitor release。
```

对 Qwen3：

```text
support 与 release basis 明显共存；
类别意义更像相对竞争场。
```

### 硬伤分析

1. **SVD basis 不是唯一基**

旋转同一子空间会改变单基解释，因此 basis-level 标签不是最终机制基。

2. **component labels 是启发式**

support/release/mixed/weak 由 target_delta 和 release_delta 阈值判定，需要后续验证。

3. **未做旋转不变审计**

需要测试 varimax/ICA/causal-optimized basis 等不同基选择。

4. **仍未做 generation audit**

目前仍是 DCF logits。

5. **上游路径仍未闭合**

本轮进一步理解 answer-site 子空间内部，但未解释 source 如何写入这些 basis sets。

### 当前进展评价

Phase116 是一次重要分解：

```text
DS7B: clean support basis set
Qwen3: support/release basis split
GLM4: weak
```

当前最可靠拼图：

```text
1. answer-site low-rank causal subspace 可拆成基向量功能成分。
2. DS7B 强效来自多个 support basis 累积。
3. Qwen3 的 release 是 basis-level 真实成分，不是统计噪声。
4. 强 basis 与 readout/transport/template 方向都不简单对齐。
```

### 下一步任务

Phase117 应做：

```text
Basis Rotation and Causal Axis Stabilization
```

目标：

```text
验证 Phase116 的 support/release basis 是否依赖 SVD 基选择；
寻找更稳定、更接近因果轴的 basis。
```

建议测试：

```text
1. 对 DS7B number/container/plant 做 SVD basis vs varimax-like rotation vs random orthogonal rotation。
2. 对 Qwen3 container/plant 做 support/release basis 在不同旋转下的稳定性。
3. 用 causal score 对子空间内方向做贪心搜索，找 causal-optimized basis。
4. 比较各 basis 的 target_delta、release_delta、readout_cos、transport_cos。
5. 保留 matched random control。
```

关键判据：

```text
如果 support/release 分解在不同合理旋转下稳定，
则 basis-level 功能分解可信度提高；
如果不稳定，则只能保留“子空间级”结论，不能解释单基功能。
```

## Phase 117: Basis Rotation and Causal Axis Stabilization 基旋转与因果轴稳定化 [2026-06-14 14:14]

### 本阶段目标

根据用户要求，先分析 Phase116 和附件判断是否正确，再继续客观测试。

判断：

```text
Phase116 的子空间内部 basis component 审计基本正确；
但 SVD basis 不是唯一基，因此 basis-level support/release 解释有硬伤；
必须测试同一 answer-site causal subspace 在正交旋转下是否保持因果效应，
以及单基 support/release 标签是否稳定。
```

本阶段 Phase117 目标：

```text
验证 Phase116 的 support/release basis 是否依赖 SVD 基选择；
比较 SVD、varimax-like rotation、random orthogonal rotation、causal_greedy basis；
区分“子空间级稳定事实”和“单基级可解释标签”。
```

### 生成脚本

- 主测试脚本：`tests/gpt5/phase117_basis_rotation_causal_axis_cuda.py`
- 汇总脚本：`tests/gpt5/phase117_basis_rotation_causal_axis_summary.py`

### 测试原理

```text
1. 复用 Phase116 的 answer-site category contrast matrix。
2. 取 rank16 SVD 子空间作为同一因果子空间。
3. 在该子空间内部构造不同正交基：
   - svd
   - varimax
   - random_rot_0
   - random_rot_1
   - causal_greedy
4. 对每个基向量做 answer_last projection removal。
5. 按 target_delta 和 max_other_delta 标注 support/release/mixed/weak。
6. 比较 single、top4、top8、top16、support set、release set。
```

关键判据：

```text
如果 top16 在不同旋转下保持一致：
  子空间级因果事实稳定。

如果 best single / support count / release count 随旋转改变：
  单基标签依赖基选择，不能当作最终机制轴。

如果 causal_greedy 用少量方向恢复大部分效果：
  子空间内存在更集中的 causal axis。
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase117_basis_rotation_causal_axis_cuda.py \
  tests/gpt5/phase117_basis_rotation_causal_axis_summary.py

python tests/gpt5/phase117_basis_rotation_causal_axis_cuda.py qwen3 \
  --train-objects 4 \
  --test-objects 4 \
  --batch-size 4 \
  --rank 8 \
  --categories number,container \
  --random-rotations 1 \
  --causal-candidates 8 \
  --set-sizes 1,2,4 \
  --output-dir results/gpt5_phase117_smoke \
  --hard-exit-after-model

python tests/gpt5/phase117_basis_rotation_causal_axis_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --categories number,container,clothing,plant \
  --random-rotations 2 \
  --causal-candidates 24 \
  --set-sizes 1,2,4,8,16 \
  --output-dir results/gpt5_phase117_basis_rotation_causal_axis \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase117_basis_rotation_causal_axis_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --categories number,container,clothing,plant \
  --random-rotations 2 \
  --causal-candidates 24 \
  --set-sizes 1,2,4,8,16 \
  --output-dir results/gpt5_phase117_basis_rotation_causal_axis \
  --hard-exit-after-model

python tests/gpt5/phase117_basis_rotation_causal_axis_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --categories number,container,clothing,plant \
  --random-rotations 2 \
  --causal-candidates 24 \
  --set-sizes 1,2,4,8,16 \
  --output-dir results/gpt5_phase117_basis_rotation_causal_axis \
  --hard-exit-after-model

python tests/gpt5/phase117_basis_rotation_causal_axis_summary.py

python -m py_compile \
  tests/gpt5/phase117_basis_rotation_causal_axis_cuda.py \
  tests/gpt5/phase117_basis_rotation_causal_axis_summary.py
```

### 结果文件

- Qwen3：`results/gpt5_phase117_basis_rotation_causal_axis/phase117_qwen3_basis_rotation_causal_axis.json`
- GLM4：`results/gpt5_phase117_basis_rotation_causal_axis/phase117_glm4_basis_rotation_causal_axis.json`
- DS7B：`results/gpt5_phase117_basis_rotation_causal_axis/phase117_deepseek7b_basis_rotation_causal_axis.json`
- 跨模型汇总：`results/gpt5_phase117_basis_rotation_causal_axis/phase117_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, clothing, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
rank = 16
scale = 1.5
set_sizes = 1, 2, 4, 8, 16
rotations = svd, varimax, random_rot_0, random_rot_1, causal_greedy
causal candidates/category = 24
```

### 客观结果

#### Qwen3

```text
number:
  top16 在所有旋转下固定为 target Δ -1.82, release +2.76
  svd single = -2.90 / release +1.59
  varimax single = -1.41 / release +2.53
  causal_greedy support set = -1.80 / release +0.31

container:
  top16 固定为 target Δ -2.53, release +1.90
  svd support set = -1.65 / release 0
  varimax single = -2.64 / release +1.33
  random_rot_0 top8 = -3.60 / release +0.15
  causal_greedy support set = -3.17 / release 0

clothing:
  top16 固定为 target Δ +1.69, release +1.49
  svd support set = -1.03 / release 0
  causal_greedy support set = -1.84 / release 0
  release set 仍强：causal_greedy +2.31 / release +2.10

plant:
  top16 固定为 target Δ -1.24, release +1.59
  svd support set = -1.39 / release 0
  random_rot_1 top8 = -3.07 / release 0
  causal_greedy top8 = -3.41 / release 0
```

Qwen3 结论：

```text
子空间级效果稳定，但 full-rank top16 常带 release；
support/release 单基标签对旋转敏感；
causal_greedy 可以找到更干净的局部 support set，
但完整子空间仍包含 release/interface 成分。
```

#### GLM4 bf16

```text
number:
  top16 约 -0.90 / release +0.68
  varimax top8 = -1.08 / release +0.78

container:
  top16 约 -0.22 / release +0.21

clothing:
  top16 约 -0.15 / release +0.20

plant:
  top16 约 -0.12 / release 0
```

GLM4 结论：

```text
旋转和 causal_greedy 没有挖出强隐藏因果轴；
Phase116 的“GLM4 效应弱”继续成立。
```

#### DS7B

```text
number:
  svd top16 = -12.58 / release 0
  varimax top16 = -12.58 / release 0
  random_rot_0 top16 = -12.59 / release 0
  random_rot_1 top16 = -12.58 / release 0
  causal_greedy top16 = -12.58 / release 0
  varimax single = -12.24 / release 0
  causal_greedy top4 = -10.65 / release 0

container:
  svd top16 = -12.52 / release 0
  varimax top16 = -12.52 / release 0
  random_rot_0 top16 = -12.54 / release 0
  random_rot_1 top16 = -12.54 / release 0
  causal_greedy top16 = -12.53 / release 0
  varimax single = -11.53 / release 0
  causal_greedy support set = -14.26 / release 0

clothing:
  top16 固定约 -2.46 / release 0
  svd support set = -5.58 / release 0
  full-rank 效果弱于 top4/top8，说明存在抵消成分
  release set 在多种基下仍存在

plant:
  top16 固定为 -7.87 / release 0
  svd top8 = -9.71 / release 0
  varimax single = -8.63 / release 0
  causal_greedy top8 = -9.91 / release 0
```

DS7B 结论：

```text
number/container/plant 是稳定的 causal subspace；
full-rank 因果效应对正交旋转不敏感；
但最强单基可从 SVD 的分布式形态变成 varimax 的集中单轴形态；
因此“强子空间存在”稳定，“SVD 单基就是机制轴”不稳定。
```

### 当前最可靠客观事实

1. **子空间级因果效应稳定**

同一 rank16 子空间经过正交旋转后，top16 基本不变。

典型：

```text
DS7B number:
  svd -12.58
  varimax -12.58
  random_rot_0 -12.59
  random_rot_1 -12.58
  causal_greedy -12.58

DS7B container:
  svd -12.52
  varimax -12.52
  random rotations -12.54
```

这说明 Phase114/115 的 answer-site causal subspace 不是 SVD 偶然产物。

2. **单基级标签明显依赖旋转**

例如 DS7B number：

```text
svd:
  best single -5.55
  support count 8

varimax:
  best single -12.24
  support count 1
```

同一子空间从“多个 support basis 累积”变成“一个极强单轴”，说明 Phase116 的 basis component 标签不能直接当作最终机制变量。

3. **DS7B 存在可集中化的强因果轴**

```text
number varimax single -12.24
container varimax single -11.53
plant varimax single -8.63
```

这不是随机方向，而是同一低秩子空间内部经过旋转后显露出的集中方向。

4. **DS7B 的 clean support 事实仍成立**

对于 number/container/plant：

```text
target_down 强；
release 接近 0；
top8/top16 稳定；
support set 强。
```

所以 Phase116 的“DS7B clean support subspace”需要改写为更严格表述：

```text
DS7B has a clean causal support subspace;
its basis-level distribution depends on the chosen orthogonal basis.
```

即：

```text
DS7B 有干净因果支持子空间；
但该支持在具体基向量上的分布依赖基选择。
```

5. **Qwen3 的 release/interface 是子空间级真实成分**

Qwen3 top16 在完整子空间下仍带明显 release：

```text
number top16: target -1.82, release +2.76
container top16: target -2.53, release +1.90
clothing top16: target +1.69, release +1.49
plant top16: target -1.24, release +1.59
```

虽然 causal_greedy 可以找到较干净 support set，但完整子空间仍包含 release/interface。

6. **GLM4 仍没有强因果轴**

旋转和 causal_greedy 都没有把 GLM4 提升到 DS7B/Qwen3 水平。

### 对 Phase116 的修正

Phase116 正确部分：

```text
1. answer-site low-rank subspace 内部确实含有功能不同的成分。
2. DS7B 的 number/container/plant 是干净支持型因果子空间。
3. Qwen3 的 release 是真实子空间成分，不是噪声。
4. GLM4 效应弱。
```

需要修正部分：

```text
1. “basis component” 不能直接解释成唯一机制轴。
2. support/release count 依赖基选择。
3. SVD 下的 distributed support 不一定表示机制本身必须分布式；
   varimax 可把 DS7B number/container/plant 压到强单轴或少数轴。
4. 更稳健的表述单位应从 basis component 上升到 causal subspace 和 causal axis family。
```

### 理论进展

Phase114/115/116/117 后，当前更稳健理论形式应改写为：

```text
Category causal state at answer site
=
low-rank causal subspace
+
rotation-dependent causal axis family
+
support/release/interface components
```

更具体：

```text
S_c(answer)
  是稳定对象；

Basis(S_c)
  不是稳定对象；

CausalAxisFamily(S_c)
  是下一步要寻找的对象。
```

对于 DS7B：

```text
S_c(answer) 是 clean support subspace；
在某些旋转下可集中成强 causal axis；
但 SVD basis 下显示为多个 support basis 累积。
```

对于 Qwen3：

```text
S_c(answer) 同时含 support 与 release/interface；
局部 support axis 可被 causal_greedy 找到；
但完整子空间不是干净 support。
```

### 硬伤分析

1. **causal_greedy 只是有限随机搜索**

```text
24 candidates/category 不等于全局最优因果轴。
```

2. **varimax 不是因果目标优化**

varimax 只是几何稀疏化旋转，不能保证就是真实机制变量。

3. **仍是 answer-site 单层测试**

尚未验证这些集中因果轴是否由上游层写入，或在多层路径中保持同一坐标。

4. **仍使用 DCF logits**

没有开放生成验证。

5. **Qwen3 的 release 仍未分解来源**

目前只知道 release/interface 是子空间级成分，尚不知道来自竞争类别、模板、词形、对象属性还是任务格式。

### 下一阶段任务

Phase118 应进入：

```text
Causal Axis Transport and Source-to-Answer Closure
```

目标：

```text
把 Phase117 找到的稳定 causal axes 从 answer site 往上游追踪，
测试这些轴是否在 object_last、middle layers、boundary layers 中被逐步写入；
并判断 DS7B 的强 support axis 是否是跨层同一坐标，
还是在 answer site 才重组出来。
```

建议测试：

```text
1. 选 DS7B number/container/plant 的 varimax/causal_greedy strong axes。
2. 在 object_last 与 answer_last 同时测：
   - source projection strength
   - answer causal effect
   - layer sweep
3. 做 axis patch：
   - remove at source layer
   - remove at answer layer
   - source+answer combined remove
4. 加 Qwen3 container/plant 对照，追踪 release/interface 是否来自上游竞争轴。
5. 加 random in-subspace axis 与 random ambient axis 对照。
```

Phase118 的关键问题：

```text
语言类别编码是否是：
  上游对象位置写入稳定因果轴，
  后续层传输并在答案位置读出；
还是：
  多个上游混合因素到答案位置才重组为 causal axis？
```

## Phase 118: Causal Axis Transport and Source-to-Answer Closure 因果轴传输与源到答案闭合 [2026-06-14 14:27]

### 本阶段目标

根据用户要求，先判断附件和 Phase117 分析是否正确，再继续客观测试。

判断：

```text
附件对 Phase117 的判断正确。
Phase117 没有推翻 Phase116，而是把结论收缩为：
  子空间级 causal effect 稳定；
  单个 SVD basis 的 support/release 标签不是旋转不变机制变量。
```

Phase118 目标：

```text
把 Phase117 找到的 answer-site causal axes 往上游追踪；
测试同一轴在 object_last、answer_last、both 三个位置的因果效果；
判断强轴是上游对象位置已经写入并直接传输，
还是主要在 answer_last 位置组装/读出。
```

### 生成脚本

- 主测试脚本：`tests/gpt5/phase118_causal_axis_transport_closure_cuda.py`
- 汇总脚本：`tests/gpt5/phase118_causal_axis_transport_closure_summary.py`

### 测试原理

```text
1. 在模型边界峰层构造 category answer-site rank16 causal subspace。
2. 对该子空间做 varimax rotation，选择 answer_last target-down 最强的 varimax_best axis。
3. 同时保留 svd_subspace 与 random_in_subspace 对照。
4. 在近峰层 sweep：
   Qwen3: L32-L35
   GLM4: L15-L18
   DS7B: L24-L27
5. 对每个 patch layer，在三个位置移除同一轴/子空间：
   object_last
   answer_last
   both
6. 记录 DCF logits target_delta、max_other_delta，并监控 answer-layer selected axis projection。
```

判据：

```text
如果 object_last removal 接近 answer_last removal：
  支持 source-to-answer 同坐标传输闭合。

如果 answer_last removal 很强而 object_last removal 很弱：
  支持 answer-site assembly/readout dominant。

如果 both 明显强于 answer_last：
  支持分布式位置共同因果。
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase118_causal_axis_transport_closure_cuda.py \
  tests/gpt5/phase118_causal_axis_transport_closure_summary.py

python tests/gpt5/phase118_causal_axis_transport_closure_cuda.py qwen3 \
  --train-objects 4 \
  --test-objects 4 \
  --batch-size 4 \
  --rank 8 \
  --layer-back 1 \
  --categories number,container \
  --output-dir results/gpt5_phase118_smoke \
  --hard-exit-after-model

python tests/gpt5/phase118_causal_axis_transport_closure_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase118_causal_axis_transport_closure \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase118_causal_axis_transport_closure_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase118_causal_axis_transport_closure \
  --hard-exit-after-model

python tests/gpt5/phase118_causal_axis_transport_closure_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase118_causal_axis_transport_closure \
  --hard-exit-after-model

python tests/gpt5/phase118_causal_axis_transport_closure_summary.py

python -m py_compile \
  tests/gpt5/phase118_causal_axis_transport_closure_cuda.py \
  tests/gpt5/phase118_causal_axis_transport_closure_summary.py
```

### 结果文件

- Qwen3：`results/gpt5_phase118_causal_axis_transport_closure/phase118_qwen3_causal_axis_transport_closure.json`
- GLM4：`results/gpt5_phase118_causal_axis_transport_closure/phase118_glm4_causal_axis_transport_closure.json`
- DS7B：`results/gpt5_phase118_causal_axis_transport_closure/phase118_deepseek7b_causal_axis_transport_closure.json`
- 跨模型汇总：`results/gpt5_phase118_causal_axis_transport_closure/phase118_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
rank = 16
scale = 1.5
axis_types = varimax_best, svd_subspace, random_in_subspace
patch_sites = object_last, answer_last, both
patch layers:
  qwen3 L32-L35
  glm4 L15-L18
  deepseek7b L24-L27
```

### 客观结果

#### Qwen3

```text
number:
  varimax_best selected = target Δ -1.41, release +2.53
  object_last best = -0.02, release +0.04
  answer_last best = -1.41, release +2.53
  both best = -1.41, release +2.57
  svd_subspace answer_last = -1.82, release +2.76

container:
  varimax_best selected = target Δ -2.64, release +1.33
  object_last best = -0.07, release +0.05
  answer_last best = -2.64, release +1.33
  both best = -2.73, release +1.28
  svd_subspace answer_last = -2.53, release +1.90

plant:
  varimax_best selected = target Δ -0.94, release +1.36
  object_last best = +0.01, release +0.08
  answer_last best = -0.94, release +1.36
  both best = -1.00, release +1.31
  svd_subspace answer_last = -1.24, release +1.59
```

Qwen3 结论：

```text
answer_last 明显强于 object_last；
both 基本不超过 answer_last；
Qwen3 的 release/interface 仍主要出现在 answer-site 轴移除中。
```

#### GLM4 bf16

```text
number:
  varimax_best selected = -0.38, release +0.26
  object_last = 0.00
  answer_last = -0.38
  svd_subspace answer_last = -0.90
  svd_subspace object_last = -0.31

container:
  varimax_best selected = -0.15, release +0.09
  object_last = -0.01
  answer_last = -0.15
  svd_subspace answer_last = -0.22

plant:
  varimax_best selected = -0.04
  object_last = -0.02
  answer_last = -0.04
```

GLM4 结论：

```text
整体仍弱；
没有出现强 source-to-answer closure。
```

#### DS7B

```text
number:
  varimax_best selected = target Δ -12.24, release 0
  object_last best = -0.74, release 0
  answer_last best = -12.24, release 0
  both best = -12.46, release 0
  svd_subspace:
    object_last -0.79
    answer_last -12.58
    both -12.78

container:
  varimax_best selected = target Δ -11.53, release 0
  object_last best = -0.47, release 0
  answer_last best = -11.53, release 0
  both best = -11.70, release 0
  svd_subspace:
    object_last -0.48
    answer_last -12.52
    both -12.68

plant:
  varimax_best selected = target Δ -8.63, release 0
  object_last best = -0.95, release 0
  answer_last best = -8.63, release 0
  both best = -8.91, release 0
  svd_subspace:
    object_last -0.90
    answer_last -7.87
    both -8.16
```

DS7B 结论：

```text
强因果轴在 answer_last 极强；
同一轴/子空间在 object_last 移除非常弱；
both 仅比 answer_last 小幅增强；
因此当前测试不支持“同一坐标从 object_last 直接传输到 answer_last”。
更支持 answer-site assembly/readout dominant。
```

### 当前最可靠客观事实

1. **DS7B 强轴主要是 answer-site 因果**

典型比例：

```text
number:
  object_last -0.74 vs answer_last -12.24

container:
  object_last -0.47 vs answer_last -11.53

plant:
  object_last -0.95 vs answer_last -8.63
```

object_last 不是完全没有信号，但远弱于 answer_last。

2. **both 不形成强加和**

```text
DS7B number:
  answer_last -12.24
  both -12.46

DS7B container:
  answer_last -11.53
  both -11.70

DS7B plant:
  answer_last -8.63
  both -8.91
```

这说明在当前同轴 patch 设计下，主要因果杠杆已经集中在 answer_last。

3. **Qwen3 同样是 answer_last 主导，但带 release**

```text
container:
  object_last -0.07
  answer_last -2.64, release +1.33
  both -2.73, release +1.28
```

Qwen3 的 release/interface 并没有在 object_last 同轴移除中显著出现，而是在 answer-site removal 中出现。

4. **GLM4 继续弱**

GLM4 没有强同轴闭合结果，延续 Phase116/117 的弱效应结论。

5. **同一 answer-site axis 不能简单当作 upstream source coordinate**

Phase118 的核心负结果：

```text
把 answer-site 选出的强 causal axis 直接拿到 object_last 移除，
不能复现 answer_last 的强 target_down。
```

这不等于上游没有类别信息，而是说明：

```text
上游对象位置的编码坐标可能不同；
answer-site 强轴可能是后续层重组/读出后的坐标。
```

### 对 Phase117 的修正和推进

Phase117 正确部分：

```text
answer-site causal subspace 稳定；
DS7B 有 clean support subspace；
varimax 可显露强单轴；
Qwen3 有 support/release/interface mixed subspace。
```

Phase118 新增限制：

```text
这些 answer-site strong axes 不能直接外推为 object_last source axes。
```

更严格表述：

```text
CausalAxis_c(answer)
  是答案位置强因果轴；
但不一定等于 CausalAxis_c(object)。
```

### 理论进展

当前条件化关系因子动力学公式应继续改写：

```text
Object state:
  O_c^l(object)

Transport / transformation:
  T_{object -> answer}^{l..L}

Answer state:
  S_c^L(answer)

Observed causal axis:
  A_c^L(answer) ∈ S_c^L(answer)
```

Phase118 表明：

```text
A_c^L(answer)
不能简单反向复制到 O_c^l(object)。
```

因此当前更稳健公式是：

```text
S_c^L(answer)
=
Transform_l_to_L(
  O_c^l(object),
  template/context,
  attention/MLP routing
)
```

而不是：

```text
S_c^L(answer)
=
direct_copy(O_c^l along same axis)
```

这对破解语言背后编码机制很关键：

```text
语言类别编码可能不是静态方向传输，
而是跨层坐标变换后在答案位置形成可读出的因果子空间。
```

### 硬伤分析

1. **只测试了 answer-site axis 在 upstream 的同坐标移除**

弱 object_last 不代表上游无类别信息，只说明同一答案轴在 object_last 不闭合。

2. **轴是在 monitor layer 拟合的**

每层可能有自己的局部坐标。如果要公平测试 source axis，需要在每层分别构造 local causal axis。

3. **object_last 可能不是唯一 source**

类别信息可能分散在 object span、post-object tokens、template tokens 或 attention output 中。

4. **Aproj 是均值投影监控**

投影均值变化不等于分布形状变化；target_down 可能来自方差/样本级排序变化。

5. **仍是 DCF logits**

尚未做开放生成和多任务验证。

### 下一阶段任务

Phase119 应进入：

```text
Layer-local Source Axis Discovery and Coordinate Transform Mapping
```

目标：

```text
不要再把 answer-site axis 直接搬到 object_last；
而是在每个 layer、每个 site 上分别学习 local category causal subspace/axis，
再测试 local source axis 是否能影响最终 answer logits。
```

建议测试：

```text
1. 对 DS7B number/container/plant，在 L24-L27 分别构造：
   - object_last local contrast subspace
   - answer_last local contrast subspace
2. 每层分别选 local varimax_best axis。
3. 测 local object axis removal 对最终 logits 的影响。
4. 测 local object axis 与 next-layer answer axis 的对齐和投影变化。
5. 加 object_span、post_object、answer_last 三类 source site。
6. 对 Qwen3 container/plant 做同样测试，追踪 release/interface 是否来自局部 source axis。
```

Phase119 的关键问题：

```text
如果 local object axis 有强因果效果：
  说明上游有类别源轴，只是坐标随层变换。

如果 local object axis 仍弱：
  说明类别因果子空间主要在 answer-site late assembly 中形成。
```

## Phase 119: Layer-local Source Axis Discovery 层局部源轴发现 [2026-06-14 14:58]

### 本阶段目标

根据用户要求，先判断附件与 Phase118 分析是否正确，再继续完成任务。

判断：

```text
附件对 Phase118 的判断正确。
Phase118 的负结果不能解释为 object/source 没有类别信息；
只能说明 answer-site axis 不能直接当作 object_last 的同坐标 source axis。
```

Phase119 目标：

```text
不再把 answer-site axis 直接搬到 object_last；
而是在每个 layer、每个 site 上分别学习 local category subspace/axis；
测试 local source axis 是否能影响最终 DCF logits。
```

### 生成脚本

- 主测试脚本：`tests/gpt5/phase119_layer_local_source_axis_cuda.py`
- 汇总脚本：`tests/gpt5/phase119_layer_local_source_axis_summary.py`

### 测试原理

```text
1. 对每个模型，在边界峰层前 3 层到峰层做 layer sweep。
2. 对每个 layer 和 site，分别捕获 train objects 的 hidden state centers。
3. 每个 site 单独构造 category contrast matrix。
4. 对 local contrast matrix 取 rank16 SVD subspace。
5. 对 local subspace 做 varimax rotation，并在同 layer/site 上选择 target-down 最强 local_varimax_best axis。
6. 同时测试：
   - local_varimax_best
   - local_svd_subspace
   - random_in_local_subspace
7. 对 heldout objects 测最终 DCF logits 的 target_delta 与 max_other_delta。
```

本阶段测试的 site：

```text
object_last
object_span_mean
post_object_mean
answer_last
```

其中：

```text
post_object_mean = object span 后到 answer_last 前/含 answer_last 的提示尾部区域平均。
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase119_layer_local_source_axis_cuda.py \
  tests/gpt5/phase119_layer_local_source_axis_summary.py

python tests/gpt5/phase119_layer_local_source_axis_cuda.py qwen3 \
  --train-objects 4 \
  --test-objects 4 \
  --batch-size 4 \
  --rank 8 \
  --layer-back 1 \
  --sites object_last,answer_last \
  --categories number,container \
  --output-dir results/gpt5_phase119_smoke \
  --hard-exit-after-model

python tests/gpt5/phase119_layer_local_source_axis_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --sites object_last,object_span_mean,post_object_mean,answer_last \
  --categories number,container,plant \
  --output-dir results/gpt5_phase119_layer_local_source_axis \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase119_layer_local_source_axis_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --sites object_last,object_span_mean,post_object_mean,answer_last \
  --categories number,container,plant \
  --output-dir results/gpt5_phase119_layer_local_source_axis \
  --hard-exit-after-model

python tests/gpt5/phase119_layer_local_source_axis_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --sites object_last,object_span_mean,post_object_mean,answer_last \
  --categories number,container,plant \
  --output-dir results/gpt5_phase119_layer_local_source_axis \
  --hard-exit-after-model

python tests/gpt5/phase119_layer_local_source_axis_summary.py

python -m py_compile \
  tests/gpt5/phase119_layer_local_source_axis_cuda.py \
  tests/gpt5/phase119_layer_local_source_axis_summary.py
```

### 结果文件

- Qwen3：`results/gpt5_phase119_layer_local_source_axis/phase119_qwen3_layer_local_source_axis.json`
- GLM4：`results/gpt5_phase119_layer_local_source_axis/phase119_glm4_layer_local_source_axis.json`
- DS7B：`results/gpt5_phase119_layer_local_source_axis/phase119_deepseek7b_layer_local_source_axis.json`
- 跨模型汇总：`results/gpt5_phase119_layer_local_source_axis/phase119_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
rank = 16
scale = 1.5
layers:
  qwen3 L32-L35
  glm4 L15-L18
  deepseek7b L24-L27
sites:
  object_last
  object_span_mean
  post_object_mean
  answer_last
axis_types:
  local_varimax_best
  local_svd_subspace
  random_in_local_subspace
```

### 客观结果

#### Qwen3

```text
number:
  object_last ≈ 0
  object_span_mean ≈ 0
  post_object_mean local_varimax_best: L35 target Δ -4.43, release +1.93
  post_object_mean local_svd_subspace: L35 target Δ -4.41, release +2.30
  answer_last local_varimax_best: L35 target Δ -1.41, release +2.53

container:
  object_last = -0.07
  object_span_mean = -0.07
  post_object_mean local_varimax_best: L32 target Δ -1.23, release +1.86
  post_object_mean local_svd_subspace: L32 target Δ -1.73, release +3.61
  answer_last local_varimax_best: L35 target Δ -2.64, release +1.33

plant:
  object_last = -0.02
  object_span_mean = -0.05
  post_object_mean local_varimax_best: L35 target Δ -5.29, release +1.37
  post_object_mean local_svd_subspace: L35 target Δ -4.66, release +1.83
  answer_last local_varimax_best: L35 target Δ -0.94, release +1.36
```

Qwen3 结论：

```text
object token 本身仍弱；
post_object_mean 出现强 local source axis；
但 release 很大，说明 Qwen3 的 source 区域仍是 support/release/interface 混合场。
```

#### GLM4 bf16

```text
number:
  object_last local_svd_subspace: -0.27, release +0.19
  object_span_mean local_svd_subspace: -0.20, release +0.23
  post_object_mean local_svd_subspace: -1.11, release +0.05
  answer_last local_svd_subspace: -0.90, release +0.68

container:
  best source weak，post_object_mean local_varimax_best -0.48, release +0.57
  answer_last local_svd_subspace -0.22

plant:
  all weak，post_object_mean local_varimax_best -0.29, release +0.31
```

GLM4 结论：

```text
仍弱；
只有 number 的 post_object_mean local_svd_subspace 有轻度信号。
```

#### DS7B

```text
number:
  object_last local_varimax_best: L27 target Δ -0.78, release 0
  object_span_mean local_varimax_best: L27 target Δ -0.81, release 0
  post_object_mean local_varimax_best: L27 target Δ -11.74, release 0
  post_object_mean local_svd_subspace: L27 target Δ -12.03, release 0
  answer_last local_varimax_best: L27 target Δ -12.24, release 0
  answer_last local_svd_subspace: L27 target Δ -12.58, release 0

container:
  object_last local_varimax_best: L27 target Δ -0.90, release 0
  object_span_mean local_varimax_best: L27 target Δ -0.95, release 0
  post_object_mean local_varimax_best: L27 target Δ -13.24, release 0
  post_object_mean local_svd_subspace: L27 target Δ -12.74, release 0
  answer_last local_varimax_best: L27 target Δ -11.53, release 0
  answer_last local_svd_subspace: L27 target Δ -12.52, release 0

plant:
  object_last local_varimax_best: L27 target Δ -0.97, release 0
  object_span_mean local_varimax_best: L27 target Δ -1.44, release 0
  post_object_mean local_varimax_best: L27 target Δ -10.58, release 0
  post_object_mean local_svd_subspace: L27 target Δ -9.57, release 0
  answer_last local_varimax_best: L27 target Δ -8.63, release 0
  answer_last local_svd_subspace: L27 target Δ -7.87, release 0
```

DS7B 结论：

```text
object_last/object_span 仍弱；
post_object_mean 出现与 answer_last 同量级甚至更强的 clean support source axis；
release = 0；
说明 Phase118 的负结果来自 source site 选窄了，而不是源轴不存在。
```

### 当前最可靠客观事实

1. **object_last 不是主要类别因果源点**

跨模型看：

```text
DS7B number object_last -0.78 vs answer_last -12.24
DS7B container object_last -0.90 vs answer_last -11.53
DS7B plant object_last -0.97 vs answer_last -8.63
```

即使重新学习 local object axis，object_last 仍远弱于 answer_last。

2. **object_span_mean 也不是主要源点**

DS7B：

```text
number object_span -0.81
container object_span -0.95
plant object_span -1.44
```

略强于 object_last，但仍远弱于 post_object/answer。

3. **post_object_mean 是强 source/control site**

DS7B：

```text
number post_object_mean -11.74 / -12.03
container post_object_mean -13.24 / -12.74
plant post_object_mean -10.58 / -9.57
```

这是 Phase119 的最大新发现。

4. **DS7B 的 post_object source axis 是 clean support**

```text
release = 0
```

对 number/container/plant 都成立。

5. **Qwen3 也有 post_object source effect，但混有 release**

Qwen3：

```text
number post_object -4.43, release +1.93
plant post_object -5.29, release +1.37
container post_object -1.23 to -1.73, release +1.86 to +3.61
```

Qwen3 的相对竞争/接口混合场不仅在 answer site，也出现在 post_object/source-control 区。

6. **random_in_local_subspace 对照显示 post_object 强效不是任意随机局部方向**

DS7B：

```text
number random post_object -4.07 vs local_varimax -11.74
container random post_object -1.48 vs local_varimax -13.24
plant random post_object -2.61 vs local_varimax -10.58
```

随机方向有时也有信号，说明局部子空间整体有因果性，但 local_varimax/local_svd 更强。

### 对 Phase118 的修正

Phase118 正确部分：

```text
answer-site axis 不能直接外推为 object_last axis；
object_last 同轴和 local axis 都弱；
answer_last 是强因果杠杆。
```

Phase119 修正部分：

```text
源位置不能只看 object_last 或 object_span；
post_object_mean 是强 source/control site；
在 DS7B 中 post_object_mean 与 answer_last 同量级。
```

更严格表述：

```text
类别因果源不在 object token 本身，
而更可能在 object 后的 prompt-tail / interface / pre-answer region 中形成。
```

### 理论进展

当前公式进一步改写：

```text
Object lexical state:
  O_c^l(object_span)

Prompt-tail / interface control state:
  P_c^l(post_object)

Answer readout state:
  A_c^L(answer)
```

Phase119 表明：

```text
O_c^l(object_span) 因果弱；
P_c^l(post_object) 因果强；
A_c^L(answer) 因果强。
```

因此当前更稳健的关系式是：

```text
A_c^L(answer)
=
Transform(
  P_c^l(post_object),
  O_c^l(object_span),
  template/context,
  route
)
```

而不是：

```text
A_c^L(answer)
=
Transform(O_c^l(object_span))
```

进一步：

```text
P_c^l(post_object)
可能是类别任务接口状态：
  它把 object lexical state 转成 category-query/readout-ready state。
```

中文解释：

```text
对象词本身更像提供语义材料；
对象后面的模板/接口区域把这些材料变成“准备回答类别”的控制状态；
答案位置再把控制状态读出为目标类别 logits。
```

### 对破解语言编码机制的关键洞察

1. **源不等于对象词本身**

当前证据显示：

```text
object token 是语义材料位置；
post_object region 是任务化/接口化控制位置；
answer token 是输出读出位置。
```

2. **语言编码可能是三段式**

```text
object semantic material
→ prompt-tail/interface control state
→ answer-site causal subspace
→ output logits
```

3. **DS7B 给出最干净版本**

```text
post_object_mean 与 answer_last 都是 clean support；
object_last/object_span 弱；
release = 0。
```

4. **Qwen3 给出竞争场版本**

```text
post_object 与 answer site 都有 target_down；
但 release 明显，说明类别状态包含竞争/接口混合。
```

### 硬伤分析

1. **post_object_mean 包含 answer_last**

当前 post_object_positions 定义为 object 后到 answer_last，包含最终位置。
这可能使 post_object_mean 受 answer_last 强轴影响。

2. **post_object 是 mean patch**

对所有 post-object tokens 使用同一 mean-derived axis，不能定位到底是哪一个 token 最关键。

3. **仍没有显式 transform mapping**

本轮发现了强 local source site，但没有拟合 post_object axis 到 answer_last axis 的变换。

4. **仍是 DCF logits**

没有开放生成验证。

5. **只测了三个类别**

number/container/plant 是关键类别，但还需要扩展到 clothing/furniture/time 等混合类别。

### 下一阶段任务

Phase120 应进入：

```text
Post-object Token Localization and Interface State Decomposition
```

目标：

```text
把 post_object_mean 拆开，定位到底是哪个 token 或哪类 token 形成强 source/control state；
排除“只是 answer_last 被平均进去”的可能。
```

建议测试：

```text
1. 将 post_object 区域拆为：
   - after_object_first
   - after_object_middle
   - pre_answer_last
   - answer_last
   - post_object_excluding_answer
2. 对每个 token/site 构造 local axis。
3. 在 DS7B number/container/plant 上优先测试。
4. 对 Qwen3 number/container/plant 做对照，观察 release/interface 来源。
5. 加 full post_object_mean 与 excluding_answer 对照。
6. 继续保留 random_in_local_subspace control。
```

Phase120 的关键问题：

```text
强 post_object source axis 是由 answer_last 混入造成，
还是确实存在于答案前的 prompt-tail/interface tokens？
```

## Phase 120: Post-object Token Localization and Interface State Decomposition 对象后词元定位与接口状态分解 [2026-06-14 15:59]

### 本阶段目标

根据用户要求，先判断附件与 Phase119 分析是否正确，再继续完成任务。

判断：

```text
附件对 Phase119 的判断基本正确：
  object_last/object_span 弱；
  post_object_mean 强；
  answer_last 强；
  说明 source/control site 不能只看对象词本身。

但 Phase119 的最大硬伤也非常关键：
  post_object_mean 包含 answer_last。
```

Phase120 目标：

```text
把 post_object_mean 拆开；
定位强效到底来自 answer_last 混入，
还是确实存在于答案前 prompt-tail/interface tokens。
```

### 生成脚本

- 主测试脚本：`tests/gpt5/phase120_post_object_token_localization_cuda.py`
- 汇总脚本：`tests/gpt5/phase120_post_object_token_localization_summary.py`

### 测试原理

```text
1. 延续 Phase119 的 layer-local local subspace/axis 方法。
2. 不再只测 post_object_mean。
3. 将对象后区域拆为：
   - object_last
   - after_object_first
   - after_object_middle
   - pre_answer_last
   - post_object_excluding_answer
   - answer_last
   - post_object_including_answer
4. 每个 layer/site 单独构造 local category contrast matrix。
5. 对 local subspace 取 rank16。
6. 测两类轴：
   - local_varimax_best
   - local_svd_subspace
7. 对 heldout objects 测最终 DCF logits。
```

关键对照：

```text
post_object_excluding_answer:
  对象后、答案前区域，不含 answer_last。

answer_last:
  最终答案位置。

post_object_including_answer:
  复现 Phase119 的 post_object_mean 风格，包含 answer_last。
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase120_post_object_token_localization_cuda.py \
  tests/gpt5/phase120_post_object_token_localization_summary.py

python tests/gpt5/phase120_post_object_token_localization_cuda.py qwen3 \
  --train-objects 4 \
  --test-objects 4 \
  --batch-size 4 \
  --rank 8 \
  --layer-back 1 \
  --sites post_object_excluding_answer,answer_last,post_object_including_answer \
  --axis-types local_varimax_best \
  --categories number,container \
  --output-dir results/gpt5_phase120_smoke \
  --hard-exit-after-model

python tests/gpt5/phase120_post_object_token_localization_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase120_post_object_token_localization \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase120_post_object_token_localization_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase120_post_object_token_localization \
  --hard-exit-after-model

python tests/gpt5/phase120_post_object_token_localization_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase120_post_object_token_localization \
  --hard-exit-after-model

python tests/gpt5/phase120_post_object_token_localization_summary.py

python -m py_compile \
  tests/gpt5/phase120_post_object_token_localization_cuda.py \
  tests/gpt5/phase120_post_object_token_localization_summary.py
```

### 结果文件

- Qwen3：`results/gpt5_phase120_post_object_token_localization/phase120_qwen3_post_object_token_localization.json`
- GLM4：`results/gpt5_phase120_post_object_token_localization/phase120_glm4_post_object_token_localization.json`
- DS7B：`results/gpt5_phase120_post_object_token_localization/phase120_deepseek7b_post_object_token_localization.json`
- 跨模型汇总：`results/gpt5_phase120_post_object_token_localization/phase120_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
rank = 16
scale = 1.5
layers:
  qwen3 L32-L35
  glm4 L15-L18
  deepseek7b L24-L27
sites:
  object_last
  after_object_first
  after_object_middle
  pre_answer_last
  post_object_excluding_answer
  answer_last
  post_object_including_answer
axis_types:
  local_varimax_best
  local_svd_subspace
```

### 客观结果

#### Qwen3

```text
number:
  post_object_excluding_answer best = -0.22, release +0.43
  answer_last = -1.41 / -1.82, release +2.53 / +2.76
  post_object_including_answer = -4.43 / -4.41, release +1.93 / +2.30

container:
  post_object_excluding_answer best = -0.14, release +0.40
  answer_last = -2.64 / -2.53, release +1.33 / +1.90
  post_object_including_answer = -1.23 / -1.73, release +1.86 / +3.61

plant:
  post_object_excluding_answer best = -0.22, release +0.29
  answer_last = -0.94 / -1.28, release +1.36 / +1.00
  post_object_including_answer = -5.29 / -4.66, release +1.37 / +1.83
```

Qwen3 结论：

```text
Phase119 的 post_object_mean 强效主要依赖 answer_last 混入；
纯答案前 tokens 效果弱；
Qwen3 的强 release/interface 仍集中在 answer_last 或 including_answer。
```

#### GLM4 bf16

```text
number:
  post_object_excluding_answer = -0.12 / -0.26
  answer_last = -0.38 / -0.90
  post_object_including_answer = -0.38 / -1.11

container:
  post_object_excluding_answer = -0.29 / -0.01
  answer_last = -0.15 / -0.22
  including_answer = -0.48 / -0.01

plant:
  post_object_excluding_answer = -0.24 / -0.02
  answer_last = -0.04 / -0.13
  including_answer = -0.29 / -0.06
```

GLM4 结论：

```text
整体继续弱；
没有强 post-object pre-answer source。
```

#### DS7B

```text
number:
  object_last = -0.78 / -0.76
  after_object_first = -1.14 / -1.24
  after_object_middle = -1.02 / -0.83
  pre_answer_last = -1.18 / -1.44
  post_object_excluding_answer = -2.35 / -2.51, release +0.57 / +0.54
  answer_last = -12.24 / -12.58, release 0
  post_object_including_answer = -11.74 / -12.03, release 0

container:
  object_last = -0.90 / -0.93
  after_object_first = -0.95 / -1.15
  after_object_middle = -0.61 / -0.50
  pre_answer_last = -0.62 / -0.87
  post_object_excluding_answer = -2.79 / -2.66, release +0.78 / +0.88
  answer_last = -11.53 / -12.52, release 0
  post_object_including_answer = -13.24 / -12.74, release 0

plant:
  object_last = -0.97 / -0.72
  after_object_first = -0.75 / -0.76
  after_object_middle = -0.73 / -0.46
  pre_answer_last = -1.41 / -1.62
  post_object_excluding_answer = -2.64 / -2.42, release +1.45 / +1.56
  answer_last = -8.63 / -7.87, release 0
  post_object_including_answer = -10.58 / -9.57, release 0
```

DS7B 结论：

```text
1. Phase119 的巨大 post_object_mean 强效主要来自 answer_last 被包含进去。
2. post_object_excluding_answer 仍有中等强度 target_down，大约 -2.3 到 -2.8。
3. 单个 after/pre-answer token 只有弱到中弱效应，大约 -0.6 到 -1.6。
4. answer_last 仍是主因果读出点。
5. including_answer 与 answer_last 同量级，说明 answer_last 是 post_object_mean 强效的主要来源。
```

### 当前最可靠客观事实

1. **answer_last 是 Phase119 post_object_mean 强效的主要来源**

DS7B：

```text
number:
  excluding_answer -2.35 / -2.51
  answer_last -12.24 / -12.58
  including_answer -11.74 / -12.03

container:
  excluding_answer -2.79 / -2.66
  answer_last -11.53 / -12.52
  including_answer -13.24 / -12.74

plant:
  excluding_answer -2.64 / -2.42
  answer_last -8.63 / -7.87
  including_answer -10.58 / -9.57
```

2. **答案前 prompt-tail/interface tokens 并非完全无效**

DS7B post_object_excluding_answer 稳定为：

```text
number: -2.35 / -2.51
container: -2.79 / -2.66
plant: -2.64 / -2.42
```

这比 object_last/object_span 强，但远低于 answer_last。

3. **单个 pre-answer token 不足以解释强效**

DS7B 单 token site：

```text
after_object_first: about -0.75 to -1.24
after_object_middle: about -0.46 to -1.02
pre_answer_last: about -0.62 to -1.62
```

说明答案前区域的中等效应更像分布式小效应，而不是某个单 token 独自形成强控制轴。

4. **Qwen3 的 pre-answer excluding effect 很弱**

Qwen3 excluding_answer 最大只有约：

```text
-0.22
```

其 including_answer 强效基本也来自 answer_last 混入与竞争读出场。

5. **GLM4 仍弱**

没有稳定强源点。

### 对 Phase119 的修正

Phase119 正确部分：

```text
object_last/object_span 不是主要控制点；
post_object_mean 包含强因果信号；
answer_last 强。
```

Phase120 修正部分：

```text
post_object_mean 不能直接解释为答案前 source/control state；
因为它包含 answer_last。
拆开后，答案前区域只有中等信号，主强效来自 answer_last。
```

更严格表述：

```text
类别因果控制主轴在 answer_last；
答案前 prompt-tail/interface 区域存在中等辅助控制场；
object token 本身弱。
```

### 理论进展

当前三段式理论需要细化为：

```text
object lexical material:
  weak direct causal control

pre-answer interface field:
  moderate distributed control

answer-site readout field:
  dominant causal support subspace
```

公式改写：

```text
h_L(answer)
=
ReadoutField_c(answer)
+
Influence(
  InterfaceField_c(pre-answer),
  template/context
)
+
ObjectMaterial_c(object_span)
```

更稳健地说：

```text
Output logits 主要由 answer-site readout field 控制；
pre-answer interface field 是辅助控制场；
object lexical state 是材料来源，不是强直接控制点。
```

### 对破解语言编码机制的关键洞察

1. **不要把 post_object_mean 当成纯源位置**

它包含 answer_last，所以会放大答案位置强效。

2. **answer_last 仍是主战场**

Phase114-120 共同支持：

```text
answer-site causal subspace 是当前最稳定、最强的类别输出控制结构。
```

3. **答案前区域是辅助接口场**

DS7B excluding_answer 的 -2.3 到 -2.8 不是噪声，说明答案前 tokens 有真实辅助因果作用。

4. **对象词不是控制点，而是材料点**

object_last/object_span 反复弱，说明对象词更像语义材料存放点，真正读出控制发生在后续 interface/readout 区。

### 硬伤分析

1. **post_object_excluding_answer 是多 token mean**

仍未定位这些中等效应是分散在多个 token，还是某些 token 小幅共同作用。

2. **没有做 combined pre-answer + answer patch**

还不知道 pre-answer interface field 与 answer readout field 是否加和、冗余或相互补偿。

3. **没有显式 attention/MLP 路由闭合**

目前只是在 residual stream site 上定位，没有追踪由哪些 head 或 MLP 写入。

4. **仍是 DCF logits**

尚未做开放生成验证。

5. **类别范围仍窄**

只测了 number/container/plant。

### 下一阶段任务

Phase121 应进入：

```text
Pre-answer Interface Additivity and Routing Closure
```

目标：

```text
测试 pre-answer interface field 与 answer-site readout field 是独立加和、冗余，还是前者通过路由写入后者。
```

建议测试：

```text
1. 对 DS7B number/container/plant 做：
   - pre-answer excluding only
   - answer_last only
   - pre-answer excluding + answer_last
2. 分别测试 local_varimax_best 与 local_svd_subspace。
3. 记录 target_delta 是否加和：
   - 如果 combined ≈ answer_last：
     pre-answer 与 answer 冗余或上游已被 answer 包含。
   - 如果 combined << answer_last：
     pre-answer 与 answer 有独立贡献。
4. 对 Qwen3 number/container/plant 做同样对照，观察 release 是否在 combined 中增强。
5. 后续再接 attention head / MLP 写入闭合。
```

Phase121 的关键问题：

```text
pre-answer interface field 是 answer readout field 的上游写入者，
还是与 answer readout field 并列但较弱的辅助控制场？
```

## Phase 121: Pre-answer Interface Additivity and Routing Closure 答案前接口加和性与路由闭合 [2026-06-14 16:14]

### 本阶段目标

根据用户要求，先判断附件与 Phase120 分析是否正确，再继续完成任务。

判断：

```text
附件对 Phase120 的判断正确。
Phase120 将 Phase119 的 post_object_mean 强效收缩为：
  answer_last 是主因果读出点；
  pre-answer / prompt-tail tokens 存在中等辅助控制场；
  object token 本身仍弱。
```

Phase121 目标：

```text
测试 pre-answer interface field 与 answer-site readout field 是独立加和、冗余，
还是 pre-answer 已被 answer field 吸收。
```

### 生成脚本

- 主测试脚本：`tests/gpt5/phase121_pre_answer_answer_additivity_cuda.py`
- 汇总脚本：`tests/gpt5/phase121_pre_answer_answer_additivity_summary.py`

### 测试原理

```text
1. 对每个 layer 分别构造：
   - post_object_excluding_answer local subspace
   - answer_last local subspace
2. 分别选择 local_varimax_best axis，并保留 local_svd_subspace。
3. 在同一 layer 做三种 causal patch：
   - pre_only
   - answer_only
   - pre_plus_answer
4. 比较 target_delta：
   combined-answer < -1:
     pre-answer 与 answer 有额外加和/独立贡献。
   combined ≈ answer:
     answer field 吸收 pre-answer，pre-answer 冗余或很弱。
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase121_pre_answer_answer_additivity_cuda.py \
  tests/gpt5/phase121_pre_answer_answer_additivity_summary.py

python tests/gpt5/phase121_pre_answer_answer_additivity_cuda.py qwen3 \
  --train-objects 4 \
  --test-objects 4 \
  --batch-size 4 \
  --rank 8 \
  --layer-back 1 \
  --axis-types local_varimax_best \
  --categories number,container \
  --output-dir results/gpt5_phase121_smoke \
  --hard-exit-after-model

python tests/gpt5/phase121_pre_answer_answer_additivity_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --axis-types local_varimax_best,local_svd_subspace \
  --categories number,container,plant \
  --output-dir results/gpt5_phase121_pre_answer_answer_additivity \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase121_pre_answer_answer_additivity_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --axis-types local_varimax_best,local_svd_subspace \
  --categories number,container,plant \
  --output-dir results/gpt5_phase121_pre_answer_answer_additivity \
  --hard-exit-after-model

python tests/gpt5/phase121_pre_answer_answer_additivity_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --axis-types local_varimax_best,local_svd_subspace \
  --categories number,container,plant \
  --output-dir results/gpt5_phase121_pre_answer_answer_additivity \
  --hard-exit-after-model

python tests/gpt5/phase121_pre_answer_answer_additivity_summary.py

python -m py_compile \
  tests/gpt5/phase121_pre_answer_answer_additivity_cuda.py \
  tests/gpt5/phase121_pre_answer_answer_additivity_summary.py
```

### 结果文件

- Qwen3：`results/gpt5_phase121_pre_answer_answer_additivity/phase121_qwen3_pre_answer_answer_additivity.json`
- GLM4：`results/gpt5_phase121_pre_answer_answer_additivity/phase121_glm4_pre_answer_answer_additivity.json`
- DS7B：`results/gpt5_phase121_pre_answer_answer_additivity/phase121_deepseek7b_pre_answer_answer_additivity.json`
- 跨模型汇总：`results/gpt5_phase121_pre_answer_answer_additivity/phase121_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
rank = 16
scale = 1.5
layers:
  qwen3 L32-L35
  glm4 L15-L18
  deepseek7b L24-L27
sites:
  post_object_excluding_answer
  answer_last
patch_modes:
  pre_only
  answer_only
  pre_plus_answer
axis_types:
  local_varimax_best
  local_svd_subspace
```

### 客观结果

#### Qwen3

```text
number:
  varimax:
    pre_only -0.22, release +0.43
    answer_only -1.41, release +2.53
    combined -1.58, release +2.20
    combined-answer -0.17
  svd:
    pre_only -0.13, release +0.45
    answer_only -1.82, release +2.76
    combined -2.00, release +2.49
    combined-answer -0.18

container:
  varimax:
    pre_only -0.14, release +0.40
    answer_only -2.64, release +1.33
    combined -2.64, release +1.31
    combined-answer 0.00
  svd:
    pre_only -0.04, release +0.26
    answer_only -2.53, release +1.90
    combined -2.29, release +1.27
    combined-answer +0.24

plant:
  varimax:
    pre_only -0.15, release +0.48
    answer_only -0.94, release +1.36
    combined -0.98, release +0.46
    combined-answer -0.04
  svd:
    pre_only -0.22, release +0.29
    answer_only -1.28, release +1.00
    combined -2.02, release +0.37
    combined-answer -0.74
```

Qwen3 结论：

```text
整体上 combined 接近 answer_only；
pre-answer 基本被 answer field 吸收；
少数 svd plant 有中等增强，但不稳定。
```

#### GLM4 bf16

```text
number:
  varimax combined-answer -0.06
  svd combined-answer +0.03

container:
  varimax combined-answer -0.28
  svd combined-answer +0.05

plant:
  varimax combined-answer -0.22
  svd combined-answer -0.16
```

GLM4 结论：

```text
整体弱；
没有形成强加和性结论。
```

#### DS7B

```text
number:
  varimax:
    pre_only -2.35, release +0.57
    answer_only -12.24, release 0
    combined -13.51, release 0
    combined-answer -1.27
  svd:
    pre_only -2.51, release +0.54
    answer_only -12.58, release 0
    combined -13.71, release 0
    combined-answer -1.13

container:
  varimax:
    pre_only -2.79, release +0.78
    answer_only -11.53, release 0
    combined -12.85, release 0
    combined-answer -1.33
  svd:
    pre_only -2.66, release +0.88
    answer_only -12.52, release 0
    combined -13.69, release 0
    combined-answer -1.17

plant:
  varimax:
    pre_only -2.64, release +1.45
    answer_only -8.63, release 0
    combined -10.15, release 0
    combined-answer -1.52
  svd:
    pre_only -2.42, release +1.56
    answer_only -7.87, release 0
    combined -9.32, release 0
    combined-answer -1.45
```

DS7B 结论：

```text
pre-answer interface field 不是完全冗余；
它与 answer-site readout field 有稳定额外贡献；
combined 比 answer_only 强约 1.1 到 1.5 logits；
但 answer_last 仍是主因果场。
```

### 当前最可靠客观事实

1. **DS7B pre-answer 有独立辅助贡献**

```text
number combined-answer: -1.27 / -1.13
container combined-answer: -1.33 / -1.17
plant combined-answer: -1.52 / -1.45
```

这说明 Phase120 中的 pre-answer 中等场不是纯噪声，也不是完全被 answer_last 吸收。

2. **DS7B answer_last 仍是主控制点**

```text
pre_only: about -2.4 to -2.8
answer_only: about -7.9 to -12.6
combined: about -9.3 to -13.7
```

pre-answer 是辅助，不是主读出。

3. **DS7B combined 会消除 pre_only 的 release**

pre_only 对 plant/container 有 release：

```text
plant pre_only release +1.45 / +1.56
container pre_only release +0.78 / +0.88
```

combined release 变为 0，说明 answer_last clean support field 可以压住 pre-answer interface 的竞争释放。

4. **Qwen3 pre-answer 基本被 answer field 吸收**

Qwen3 combined-answer 多数在：

```text
-0.18 到 +0.24
```

没有稳定独立加和。

5. **GLM4 继续弱**

没有强结论。

### 对 Phase120 的推进

Phase120 正确部分：

```text
answer_last 是主因果读出点；
pre-answer excluding 有中等辅助作用；
object token 本身弱。
```

Phase121 新增：

```text
在 DS7B 中，pre-answer auxiliary field 与 answer readout field 有稳定额外贡献；
不是完全冗余。
在 Qwen3 中，pre-answer 基本被 answer field 吸收。
```

更严格表述：

```text
DS7B:
  pre-answer interface field = independent auxiliary support/interface field
  answer_last readout field = dominant clean support field

Qwen3:
  pre-answer field weak
  answer-site competition/readout field dominates
```

### 理论进展

当前公式可更新为：

```text
LogitControl_c
=
ReadoutField_c(answer)
+
AuxInterfaceField_c(pre-answer)
+
WeakObjectMaterial_c(object)
```

在 DS7B 中：

```text
Effect(ReadoutField) >> Effect(AuxInterfaceField)
Effect(ReadoutField + AuxInterfaceField)
  =
Effect(ReadoutField) + small extra support
```

在 Qwen3 中：

```text
Effect(ReadoutField + AuxInterfaceField)
≈
Effect(ReadoutField)
```

这说明不同模型的语言编码路径不同：

```text
DS7B 更像 answer readout + auxiliary interface 双场；
Qwen3 更像 answer-site competition/readout 单主场。
```

### 硬伤分析

1. **还没有证明 pre-answer 写入 answer**

加和性只能说明存在额外贡献，不等于 pre-answer 是 answer field 的上游 writer。

2. **没有定位 writer module**

还不知道是 attention head、MLP，还是 residual stream 直接保留造成 combined 增强。

3. **combined 使用同层 patch**

没有做跨层 pre-answer -> later answer 的时序闭合。

4. **仍是 DCF logits**

没有开放生成验证。

5. **类别范围仍窄**

只测了 number/container/plant。

### 下一阶段任务

Phase122 应进入：

```text
Pre-answer-to-Answer Writer Path Closure
```

目标：

```text
测试 pre-answer interface field 是否通过 attention/MLP 写入 answer-site readout field。
```

建议测试：

```text
1. 以 DS7B number/container/plant 为主。
2. 选 L27 的 pre-answer local_varimax/local_svd 与 answer local_varimax/local_svd。
3. 做 pre-answer remove 后，监控 answer_last projection on answer axis 是否下降。
4. 做 attention head ablation：
   - answer token attending to pre-answer span 的 head
   - 对比 object span head
5. 做 MLP ablation：
   - answer-layer MLP
   - pre-answer-layer MLP
6. 判断：
   - pre-answer removal 是否降低 answer-axis projection
   - head/MLP ablation 是否同时降低 answer projection 和 logits
```

Phase122 的关键问题：

```text
pre-answer interface field 是通过具体 attention/MLP writer 写入 answer readout field，
还是只是与 answer field 并列存在的辅助控制场？
```

## Phase 122: Pre-answer-to-Answer Projection Closure 答案前到答案投影闭合 [2026-06-14 16:44]

### 本阶段目标

根据用户要求，先判断附件与 Phase121 分析是否正确，再继续完成任务。

判断：

```text
附件对 Phase121 的判断正确。
DS7B:
  pre-answer interface field 有稳定额外贡献；
  answer_last 仍是主因果读出场。
Qwen3:
  pre-answer 基本被 answer field 吸收。
GLM4:
  继续弱。
```

Phase122 目标：

```text
测试 pre-answer interface field 是否会写入 answer-site readout field。
做法：
  移除不同层的 pre-answer field；
  监控 peak-layer answer_last 在 answer causal axis/subspace 上的投影是否下降；
  同时记录最终 DCF logits。
```

### 生成脚本

- 主测试脚本：`tests/gpt5/phase122_pre_answer_to_answer_projection_closure_cuda.py`
- 汇总脚本：`tests/gpt5/phase122_pre_answer_to_answer_projection_closure_summary.py`

### 测试原理

```text
1. 对每个模型、类别、层位，分别构造：
   - post_object_excluding_answer local subspace
   - answer_last local subspace
2. 在 peak layer 构造 answer monitor axis/subspace。
3. 对 patch layers 做：
   - pre_remove
   - answer_remove
   - pre_plus_answer
4. 记录：
   - target_delta
   - max_other_delta
   - answer_proj_delta
5. 如果 pre_remove 降低 answer_proj 且 combined 强于 answer_only：
   支持 pre-answer 写入 answer projection。
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase122_pre_answer_to_answer_projection_closure_cuda.py \
  tests/gpt5/phase122_pre_answer_to_answer_projection_closure_summary.py

python tests/gpt5/phase122_pre_answer_to_answer_projection_closure_cuda.py qwen3 \
  --train-objects 4 \
  --test-objects 4 \
  --batch-size 4 \
  --rank 8 \
  --layer-back 1 \
  --axis-types local_varimax_best \
  --categories number,container \
  --output-dir results/gpt5_phase122_smoke \
  --hard-exit-after-model

python tests/gpt5/phase122_pre_answer_to_answer_projection_closure_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --axis-types local_varimax_best,local_svd_subspace \
  --categories number,container,plant \
  --output-dir results/gpt5_phase122_pre_answer_to_answer_projection_closure \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase122_pre_answer_to_answer_projection_closure_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --axis-types local_varimax_best,local_svd_subspace \
  --categories number,container,plant \
  --output-dir results/gpt5_phase122_pre_answer_to_answer_projection_closure \
  --hard-exit-after-model

python tests/gpt5/phase122_pre_answer_to_answer_projection_closure_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --axis-types local_varimax_best,local_svd_subspace \
  --categories number,container,plant \
  --output-dir results/gpt5_phase122_pre_answer_to_answer_projection_closure \
  --hard-exit-after-model

python tests/gpt5/phase122_pre_answer_to_answer_projection_closure_summary.py

python -m py_compile \
  tests/gpt5/phase122_pre_answer_to_answer_projection_closure_cuda.py \
  tests/gpt5/phase122_pre_answer_to_answer_projection_closure_summary.py
```

### 结果文件

- Qwen3：`results/gpt5_phase122_pre_answer_to_answer_projection_closure/phase122_qwen3_pre_answer_to_answer_projection_closure.json`
- GLM4：`results/gpt5_phase122_pre_answer_to_answer_projection_closure/phase122_glm4_pre_answer_to_answer_projection_closure.json`
- DS7B：`results/gpt5_phase122_pre_answer_to_answer_projection_closure/phase122_deepseek7b_pre_answer_to_answer_projection_closure.json`
- 跨模型汇总：`results/gpt5_phase122_pre_answer_to_answer_projection_closure/phase122_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
rank = 16
scale = 1.5
patch layers:
  qwen3 L32-L35
  glm4 L15-L18
  deepseek7b L24-L27
monitor layer:
  qwen3 L35
  glm4 L18
  deepseek7b L27
patch modes:
  pre_remove
  answer_remove
  pre_plus_answer
axis types:
  local_varimax_best
  local_svd_subspace
```

### 客观结果

#### Qwen3

```text
number:
  pre_remove target weak: -0.22 / -0.13
  answer_remove: -1.41 / -1.82
  combined: -1.58 / -2.00
  combined-answer: -0.17 / -0.18

container:
  pre_remove target weak: -0.14 / -0.04
  answer_remove: -2.64 / -2.53
  combined: -2.64 / -2.29
  combined-answer: 0.00 / +0.24

plant:
  pre_remove target weak: -0.15 / -0.22
  answer_remove: -0.94 / -1.28
  combined: -0.98 / -2.02
  combined-answer: -0.04 / -0.74
```

Qwen3 结论：

```text
pre-answer 没有稳定额外 target effect；
projection 有大幅变化的条件多与 answer/site patch 或 subspace norm 监控有关，
但没有形成稳定 pre -> answer 写入闭合。
```

#### GLM4 bf16

```text
整体 target_delta 很弱。
combined-answer:
  number -0.06 / +0.03
  container -0.28 / +0.05
  plant -0.22 / -0.16
```

GLM4 结论：

```text
继续弱；
没有强 writer path 结论。
```

#### DS7B

target additivity 复现 Phase121：

```text
number:
  pre_remove best target:
    L27 -2.35 / -2.51
  answer_remove:
    L27 -12.24 / -12.58
  combined:
    L27 -13.51 / -13.71
  combined-answer:
    -1.27 / -1.13

container:
  pre_remove best target:
    L27 -2.79 / -2.66
  answer_remove:
    L27 -11.53 / -12.52
  combined:
    L27 -12.85 / -13.69
  combined-answer:
    -1.33 / -1.17

plant:
  pre_remove best target:
    L27 -2.64 / -2.42
  answer_remove:
    L27 -8.63 / -7.87
  combined:
    L27 -10.15 / -9.32
  combined-answer:
    -1.52 / -1.45
```

answer projection closure:

```text
number:
  L24 pre_remove:
    target -2.06 / -2.05
    answer_proj_delta -55.89 / -51.27

container:
  L24 pre_remove:
    svd target -1.19
    answer_proj_delta -44.30
  varimax strongest target at L27:
    target -2.79
    answer_proj_delta 0.00

plant:
  L24 pre_remove:
    target -2.36 / -2.28
    answer_proj_delta -49.09 / -44.10
```

DS7B 结论：

```text
1. pre-answer field 的额外 logits 贡献稳定复现。
2. 早层 L24 pre_remove 会降低 peak answer projection，同时产生 target_down。
3. 最强 pre_only logits 常在 L27，同层 patch 不降低 peak answer projection，
   更像局部并列贡献或同层输出接口贡献。
4. 因此存在部分 pre -> answer projection closure，
   但 pre-answer 的全部贡献不能只解释为 answer-axis mean projection 下降。
```

### 当前最可靠客观事实

1. **DS7B 早层 pre-answer 有写入 answer projection 的证据**

```text
number L24 pre_remove:
  answer_proj_delta about -51 to -56
  target_delta about -2.05

plant L24 pre_remove:
  answer_proj_delta about -44 to -49
  target_delta about -2.3
```

2. **DS7B L27 pre-answer 有同层局部贡献**

```text
L27 pre_remove:
  target_delta about -2.4 to -2.8
  answer_proj_delta 0
```

这符合 hook 时序：同层 residual output 上 patch pre-answer token，不一定改变同层 answer token hidden state。

3. **combined 强于 answer-only 的现象稳定**

```text
DS7B combined-answer:
  -1.1 to -1.5
```

说明 pre-answer 辅助场不是完全冗余。

4. **Qwen3/GLM4 没有稳定 pre -> answer closure**

Qwen3 是 answer-site 主导吸收；GLM4 弱。

### 对 Phase121 的推进

Phase121 正确部分：

```text
DS7B pre-answer 与 answer 有稳定加和；
Qwen3 基本被 answer field 吸收；
GLM4 弱。
```

Phase122 新增：

```text
DS7B 的 pre-answer 辅助贡献分成两种：
  早层 pre-answer: 有 answer projection 写入证据；
  峰层 pre-answer: 有局部 logits 贡献，但不表现为同层 answer projection 下降。
```

更严格表述：

```text
pre-answer interface field partly writes into answer readout field across layers,
and partly contributes as a parallel/local interface field near the readout layer.
```

中文：

```text
答案前接口场一部分跨层写入答案读出场；
另一部分在读出层附近作为并列/局部接口场贡献 logits。
```

### 理论进展

当前 DS7B 类别读出链条可写为：

```text
EarlyPreAnswerField_l
  -> contributes to AnswerReadoutProjection_L

LatePreAnswerField_L
  -> local auxiliary logits control

AnswerReadoutField_L
  -> dominant clean support logits control
```

公式：

```text
LogitControl_c
=
ReadoutField_c(answer_L)
+
AuxInterfaceField_c(pre_L)
+
WritePath(
  EarlyInterfaceField_c(pre_l -> answer_L)
)
+
WeakObjectMaterial_c(object)
```

这比 Phase121 更细：

```text
pre-answer 不是单一场；
它至少包含 early writer component 与 late local component。
```

### 硬伤分析

1. **answer_proj_delta 是均值投影**

没有测样本级排序、方差、非线性投影变化。

2. **投影数值尺度很大**

尤其 subspace norm 监控下数值较大，需后续做标准化投影 delta。

3. **未定位具体 attention head / MLP**

Phase122 只证明 residual stream 层位上的部分闭合，没有找 writer module。

4. **同层 patch 的解释受 hook 时序影响**

L27 pre patch 不改变 L27 answer hidden 是预期现象，不能解释为无关系。

5. **仍是 DCF logits**

未做开放生成验证。

### 下一阶段任务

Phase123 应进入：

```text
Attention/MLP Writer Localization for Pre-answer-to-Answer Path
```

目标：

```text
定位 DS7B 中 early pre-answer field 写入 answer readout projection 的具体模块。
```

建议测试：

```text
1. 只先做 DS7B number/container/plant。
2. 重点层：
   - L24/L25 pre-answer writer candidate
   - L26/L27 answer readout candidate
3. 扫描 attention heads：
   - answer token attending to pre-answer span
   - answer token attending to object span
4. 对高 attention-mass heads 做 ablation。
5. 监控：
   - final target_delta
   - answer_proj_delta on answer axis
6. 再测 MLP ablation：
   - pre-answer positions at L24/L25
   - answer_last at L26/L27
```

Phase123 关键问题：

```text
pre-answer -> answer projection closure 是由少数 attention heads 写入，
还是由 MLP/residual 多模块共同形成？
```

## Phase 123: Attention MLP Writer Localization 注意力与 MLP 写入者定位 [2026-06-14 17:26]

### 本阶段目标

根据附加分析，Phase122 的正确结论应保持为：

```text
DS7B 的 pre-answer -> answer projection closure 只是部分闭合。
早层 pre-answer remove 会降低 peak answer projection 并伴随 target_down；
峰层 pre-answer remove 有 logits 贡献但不降低同层 answer projection。
```

因此 Phase123 继续测试：

```text
1. pre-answer -> answer 的写入是否能定位到少数 attention heads。
2. pre-answer 或 answer_last 位置的 MLP output 是否能解释 Phase122 的辅助贡献。
3. 单模块扰动是否同时满足：
   - target_delta 明显下降；
   - answer_proj_delta 明显下降；
   - 强于 object/random control。
```

### 生成脚本

- 主测试脚本：`tests/gpt5/phase123_attention_mlp_writer_localization_cuda.py`
- 汇总脚本：`tests/gpt5/phase123_attention_mlp_writer_localization_summary.py`

### 执行命令

```bash
python tests/gpt5/phase123_attention_mlp_writer_localization_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --rank 4 \
  --layer-back 1 \
  --top-k-heads 1 \
  --categories number \
  --output-dir results/gpt5_phase123_smoke \
  --hard-exit-after-model

python tests/gpt5/phase123_attention_mlp_writer_localization_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --top-k-heads 4 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase123_attention_mlp_writer_localization \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase123_attention_mlp_writer_localization_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --top-k-heads 4 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase123_attention_mlp_writer_localization \
  --hard-exit-after-model

python tests/gpt5/phase123_attention_mlp_writer_localization_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --top-k-heads 4 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase123_attention_mlp_writer_localization \
  --hard-exit-after-model

python tests/gpt5/phase123_attention_mlp_writer_localization_summary.py

python -m py_compile \
  tests/gpt5/phase123_attention_mlp_writer_localization_cuda.py \
  tests/gpt5/phase123_attention_mlp_writer_localization_summary.py
```

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
rank = 16
top heads/group = 4

Qwen3 layers = L32-L35, monitor L35
GLM4 layers = L15-L18, monitor L18
DS7B layers = L24-L27, monitor L27
```

### 测试原理

1. 在 peak answer_last 层构造类别 answer-site monitor axis。
2. 扫描 answer token 对各 source group 的 attention mass：

```text
post_object / pre-answer
object_span / object_last
self
```

3. 选择 top pre-answer heads、top object heads、top self heads 和 random heads。
4. 在 answer token 的 attention output projection 输入处，按 head slice 做单头 ablation。
5. 同时测试 MLP output ablation：

```text
pre_answer positions
answer_last position
```

6. 监控：

```text
final DCF target_delta
peak answer_last answer_proj_delta
max_other_delta
```

### 结果文件

- Qwen3：`results/gpt5_phase123_attention_mlp_writer_localization/phase123_qwen3_attention_mlp_writer_localization.json`
- GLM4：`results/gpt5_phase123_attention_mlp_writer_localization/phase123_glm4_attention_mlp_writer_localization.json`
- DS7B：`results/gpt5_phase123_attention_mlp_writer_localization/phase123_deepseek7b_attention_mlp_writer_localization.json`
- 跨模型汇总：`results/gpt5_phase123_attention_mlp_writer_localization/phase123_cross_model_summary.md`

### 客观结果

#### Qwen3

```text
number:
  best pre-head = L33 H24, target_delta -0.30, answer_proj_delta -1.15
  best object-head = -0.01
  best pre-MLP = -0.06
  best answer-MLP = -0.31, answer_proj_delta +31.25

container:
  best pre-head = -0.08
  best pre-MLP = -0.08
  best answer-MLP = -0.12

plant:
  best pre-head = -0.55, answer_proj_delta -1.77
  best pre-MLP = -0.00
  best answer-MLP = -0.24
```

Qwen3 没有出现强 writer localization。

#### GLM4 bf16

```text
number/container/plant:
  pre-head target_delta around -0.01 to -0.03
  object/random controls also around same level
  pre-MLP around -0.03 to -0.06
  answer-MLP around -0.03 to -0.07
```

GLM4 仍然弱。

#### DS7B

```text
number:
  best pre-head = L26 H17, target_delta -0.05, answer_proj_delta -11.14
  best object-head = L24 H22, target_delta -0.09, answer_proj_delta -14.81
  best random-head = -0.05
  best pre-MLP = L26, target_delta -0.33, answer_proj_delta -0.61
  best answer-MLP = L27, target_delta -1.16, answer_proj_delta -183.18

container:
  best pre-head = L25 H20, target_delta -0.09, answer_proj_delta -1.66
  best object-head = -0.14
  best random-head = -0.18
  best pre-MLP = L24, target_delta -0.34, answer_proj_delta -0.10
  best answer-MLP = -0.11

plant:
  best pre-head = L26 H17, target_delta -0.16, answer_proj_delta -11.85
  best object-head = -0.03
  best random-head = -0.06
  best pre-MLP = L26, target_delta -0.39, answer_proj_delta +0.31
  best answer-MLP = -0.15
```

### 当前最可靠客观事实

1. **DS7B 单个 pre-answer attention head 可以显著改变 answer projection，但 logits 因果效应很小**

```text
number L26 H17:
  answer_proj_delta -11.14
  target_delta only -0.05

plant L26 H17:
  answer_proj_delta -11.85
  target_delta only -0.16
```

这说明：

```text
answer projection drop 不是充分的 writer 判据；
必须同时要求 target_delta 明显下降。
```

2. **object/self heads 也能降低 answer projection**

```text
DS7B number object-head L24 H22:
  answer_proj_delta -14.81
  target_delta -0.09

DS7B number self-head L24 H25:
  answer_proj_delta -12.21
  target_delta -0.06
```

因此不能把 projection drop 简单解释为 pre-answer 专属写入。

3. **单点 pre-answer MLP ablation 只产生弱到中等 target_down**

```text
DS7B pre-MLP:
  number -0.33
  container -0.34
  plant -0.39
```

这小于 Phase122 中 residual subspace pre_remove 的约 -2 级别效果。

4. **DS7B number 的 answer_last MLP L27 有相对明显 readout effect**

```text
DS7B number answer-MLP L27:
  target_delta -1.16
  answer_proj_delta -183.18
```

这更像 answer-site readout module，而不是 pre-answer writer module。

5. **Phase123 没有找到单个模块能复现 Phase122 的强 pre-answer subspace effect**

尤其 DS7B：

```text
Phase122 pre_remove:
  target_delta about -2.0 to -2.8

Phase123 single pre-head:
  target_delta about -0.05 to -0.16

Phase123 single pre-MLP:
  target_delta about -0.33 to -0.39
```

### 对 Phase122 的判定更新

Phase122 的正确部分保留：

```text
DS7B 早层 pre-answer field 会影响后续 answer projection；
pre-answer 辅助贡献真实存在；
pre-answer 与 answer field 在 DS7B 上有稳定加和。
```

Phase123 对其增加限制：

```text
这个写入/辅助效应不是由单个高 pre-answer attention head 或单点 pre-answer MLP output 主导。
它更可能是 distributed multi-head + MLP + residual subspace 的组合效应。
```

更严格表述：

```text
Phase122 证明了 site/subspace level 的部分闭合；
Phase123 尚未证明 module-level 单点闭合。
```

### 硬伤分析

1. **只测了 single-head ablation**

如果真实 writer 是多个弱 head 共同形成，单头测试会低估。

2. **MLP ablation 是 whole-output zero，不是 category subspace removal**

这会混入非类别信息，也可能因为 LayerNorm/残差补偿导致效果变弱。

3. **head selection 只按 attention mass**

高 attention mass 不等于高 value-vector 类别写入。下一步需要按 value/output projection 与 answer axis 的对齐度选头。

4. **answer_proj_delta 标尺仍不稳定**

有些 answer-MLP 条件出现极大 projection delta，但 logits 不按比例变化，说明投影值需要标准化或样本级解释。

5. **没有测 head set cumulative ablation**

Phase113 做过 head set，但不是针对 Phase122 的 pre-answer writer path。Phase123 还没有做 pre-head set 累积测试。

### 理论进展

当前更可信的结构图是：

```text
DS7B:
  pre-answer residual subspace
    -> weak distributed attention/value/MLP contributions
    -> partly changes answer readout projection
    -> plus late local auxiliary contribution

  answer_last field
    -> dominant readout control
    -> answer MLP can affect number readout
```

不能再把 pre-answer writer 简化成：

```text
one head writes category to answer
```

更合理的假设是：

```text
category control is stored as a distributed residual subspace,
and modules expose only small slices of that subspace when ablated individually.
```

中文表述：

```text
类别控制更像残差流中的分布式子空间；
单个注意力头或单个 MLP 输出只暴露其中一小段。
```

### 下一阶段任务

Phase124 应做：

```text
Pre-answer Writer Set and Value-alignment Sweep
```

核心目标：

```text
从 single module 进入 module set；
从 attention mass selection 进入 value/readout alignment selection。
```

测试方案：

```text
1. DS7B 优先，Qwen3/GLM4 做对照。
2. 仍使用 number/container/plant。
3. 对每个 attention head 计算：
   - answer token attends to pre-answer mass
   - head output delta 与 answer monitor axis 的对齐
   - head ablation 的 target_delta
4. 选择三类 head set：
   - top attention-mass heads
   - top value-aligned heads
   - top ablation-discovered heads
5. 做 cumulative head-set ablation：
   - k = 1, 2, 4, 8, 16
6. 与 pre-answer subspace removal 做同层对照：
   - head set 是否接近 Phase122 的 -2 级别 target_down
7. 再做 MLP category-subspace removal：
   - 不再 zero whole MLP output
   - 只移除 MLP output 中与 pre-answer local category basis 对齐的部分
```

Phase124 的关键判据：

```text
如果 cumulative value-aligned head set 明显接近 Phase122 pre_remove，
则 pre-answer writer 主要在 attention value path。

如果 head set 仍弱，而 MLP category-subspace removal 强，
则 writer 更偏 MLP/residual transformation。

如果两者单独弱、合并强，
则 pre-answer writer 是多模块协同闭合。
```

## Phase 124: Pre-answer Writer Set and Value-alignment Sweep 写入集合与值对齐扫描 [2026-06-14 17:44]

### 本阶段目标

根据附加分析和 Phase123，当前正确判断是：

```text
Phase122 证明的是 site/subspace level 的部分闭合；
Phase123 证明这个闭合不是单个 attention head 或单点 MLP output 主导。
```

因此 Phase124 继续测试：

```text
1. 多个 attention heads 累积消融能否接近 Phase122 的 pre-answer subspace effect。
2. 按 attention mass 选头是否不如按 value/readout alignment 选头。
3. pre-answer MLP category-subspace removal 是否强于 whole-output zero ablation。
```

### 生成脚本

- 主测试脚本：`tests/gpt5/phase124_writer_set_value_alignment_cuda.py`
- 汇总脚本：`tests/gpt5/phase124_writer_set_value_alignment_summary.py`

### 执行命令

```bash
python tests/gpt5/phase124_writer_set_value_alignment_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --rank 4 \
  --layer-back 1 \
  --set-sizes 1,2 \
  --candidate-pool 4 \
  --categories number \
  --output-dir results/gpt5_phase124_smoke \
  --hard-exit-after-model

python tests/gpt5/phase124_writer_set_value_alignment_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --set-sizes 1,2,4,8,16 \
  --candidate-pool 24 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase124_writer_set_value_alignment \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase124_writer_set_value_alignment_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --set-sizes 1,2,4,8,16 \
  --candidate-pool 24 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase124_writer_set_value_alignment \
  --hard-exit-after-model

python tests/gpt5/phase124_writer_set_value_alignment_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --set-sizes 1,2,4,8,16 \
  --candidate-pool 24 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase124_writer_set_value_alignment \
  --hard-exit-after-model

python tests/gpt5/phase124_writer_set_value_alignment_summary.py

python -m py_compile \
  tests/gpt5/phase124_writer_set_value_alignment_cuda.py \
  tests/gpt5/phase124_writer_set_value_alignment_summary.py
```

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
rank = 16
candidate pool = 24
head set sizes = 1, 2, 4, 8, 16

Qwen3 layers = L32-L35, monitor L35
GLM4 layers = L15-L18, monitor L18
DS7B layers = L24-L27, monitor L27
```

### 测试原理

本轮不再只测单个 head，而是构造 head set：

```text
attention_mass:
  按 answer token 对 pre-answer tokens 的 attention mass 排序。

value_aligned:
  捕获 o_proj input 中每个 head 的 answer-position slice，
  乘以该 head 对应的 o_proj weight block，
  得到 head output contribution，
  再计算其与 answer monitor axis 的投影对齐。

target_discovered:
  在候选池内做 single-head ablation，
  按 target_delta 排序。

projection_discovered:
  在候选池内按 answer_proj_delta 排序。

object_control/random_control:
  对照组。
```

同时测试：

```text
pre-answer MLP category-subspace removal
```

即不再 zero whole MLP output，而是只移除 MLP output 中与 pre-answer local category basis 对齐的部分。

### 结果文件

- Qwen3：`results/gpt5_phase124_writer_set_value_alignment/phase124_qwen3_writer_set_value_alignment.json`
- GLM4：`results/gpt5_phase124_writer_set_value_alignment/phase124_glm4_writer_set_value_alignment.json`
- DS7B：`results/gpt5_phase124_writer_set_value_alignment/phase124_deepseek7b_writer_set_value_alignment.json`
- 跨模型汇总：`results/gpt5_phase124_writer_set_value_alignment/phase124_cross_model_summary.md`

### 客观结果

#### Qwen3

```text
number:
  attention set best = -0.28
  target-discovered set best = -0.64
  object control = -0.19
  random control = +0.04
  pre-MLP subspace = -0.00

container:
  target-discovered set best = -0.34
  value_aligned projection drop can be large, but target_delta not down
  pre-MLP subspace = -0.05

plant:
  attention set best = -0.49
  target-discovered set best = -0.93
  random control = -0.61
  pre-MLP subspace = -0.01
```

Qwen3 有 head-set target_down，但 control 也不弱，不能判为清洁写入闭合。

#### GLM4 bf16

```text
number:
  target-discovered set best = -0.39
  pre-MLP subspace = -0.06

container:
  target-discovered set best = -0.09
  pre-MLP subspace = -0.06

plant:
  target-discovered set best = -0.08
  pre-MLP subspace = -0.01
```

GLM4 仍弱。

#### DS7B

```text
number:
  attention_mass k16 = target_delta -0.16, answer_proj_delta -71.98
  value_aligned k16 = target_delta -0.64, answer_proj_delta -132.77
  target_discovered k16 = target_delta -0.67, answer_proj_delta -102.78
  projection_discovered k16 = target_delta -0.75, answer_proj_delta -192.08
  object_control k16 = target_delta -0.32
  random_control best = -0.04
  pre-MLP subspace L24 = target_delta -0.54, answer_proj_delta -19.50

container:
  attention_mass best = -0.03
  value/abs_value best = -0.25
  target_discovered k16 = -0.44
  object_control best = -0.15
  random_control best = -0.09
  pre-MLP subspace best = -0.40

plant:
  attention_mass k16 = -0.60
  value_aligned k16 = -0.67
  target_discovered k16 = -1.25
  projection_discovered k16 = -0.96
  object_control best = -0.03
  random_control k16 = -0.32
  pre-MLP subspace L24 = -0.67
```

### 当前最可靠客观事实

1. **DS7B plant 出现目前最强的模块集合证据**

```text
target_discovered head set k16:
  target_delta -1.25
  answer_proj_delta -132.29

object_control:
  target_delta -0.03

random_control:
  target_delta -0.32
```

这说明 plant 的 pre-answer/answer writer path 至少有一部分可以由 head set 捕获。

2. **DS7B number 有中等 head-set 效应，但仍不到 Phase122 强度**

```text
number projection_discovered k16:
  target_delta -0.75

Phase122 number pre_remove:
  about -2.0 to -2.8
```

所以 number 的 writer set 仍未闭合。

3. **DS7B container 仍弱**

```text
target_discovered k16:
  target_delta -0.44

pre-MLP subspace:
  target_delta -0.40
```

这说明 container 的 Phase122/Phase121 辅助贡献可能更分散，或当前 head/value/MLP 选择仍未命中。

4. **value alignment 通常强于纯 attention mass**

DS7B number：

```text
attention_mass k16 = -0.16
value_aligned k16 = -0.64
```

DS7B plant：

```text
attention_mass k16 = -0.60
value_aligned k16 = -0.67
target_discovered k16 = -1.25
```

这支持 Phase123 的硬伤判断：

```text
attention mass 不是 writer 的充分排序信号。
```

5. **pre-MLP category-subspace removal 明显强于 Phase123 whole-output zero 的解释力，但仍不够**

DS7B：

```text
number pre-MLP subspace L24 = -0.54
container pre-MLP subspace = -0.40
plant pre-MLP subspace L24 = -0.67
```

它比 Phase123 单点 whole-output MLP zero 更有结构意义，但仍不能单独复现 Phase122 的 -2 级别。

### 对 Phase123 的判定更新

Phase123 正确：

```text
单头/单点 MLP 不能解释 pre-answer writer。
```

Phase124 新增：

```text
head set 确实比 single head 更接近真实 effect；
value/readout alignment 比 attention mass 更有效；
但除 DS7B plant 外，仍未达到接近 Phase122 的闭合强度。
```

因此当前结论应写成：

```text
pre-answer writer 是 distributed module-set mechanism；
目前只有 plant 在 DS7B 上出现较强 head-set partial closure。
number/container 仍是 residual-subspace > module-set。
```

### 硬伤分析

1. **candidate pool 仍有限**

本轮候选池为 24，不是全头 exhaustive set search。

2. **head set 是简单累积 ablation**

没有测试 head 之间的非线性组合、补偿、冗余和顺序依赖。

3. **target_discovered 有选择偏差**

它用同一批 heldout prompts 做单头筛选和集合测试，可能高估。但 object/random controls 可以部分约束这个问题。

4. **value alignment 是 answer-position output alignment**

它没有直接证明这些 head 从 pre-answer tokens 读取 category value，只证明 head output 与 answer monitor axis 对齐。

5. **MLP subspace basis 来自 residual local centers，不是 MLP output centers**

更严格的 MLP 测试应直接采集 MLP output 的 category basis。

6. **仍未复现 Phase122 的强 residual subspace effect**

尤其 number/container：

```text
head set + pre-MLP subspace 仍未组合测试；
单独都不足以闭合。
```

### 理论进展

当前结构图更新为：

```text
DS7B plant:
  pre-answer/answer writer path
    -> detectable head-set partial closure
    -> value/readout aligned heads matter
    -> pre-MLP subspace contributes moderately

DS7B number:
  answer readout field strong
  pre-answer residual subspace strong
  module-set only captures part of it

DS7B container:
  residual-level effect remains more visible than module-level effect
```

更一般的机制判断：

```text
language/category control is not stored as a single writer module;
it appears as a residual subspace whose causal effect is only partially exposed
through identifiable attention/MLP module sets.
```

中文：

```text
语言类别控制不像单个写入模块；
更像残差流子空间，模块集合只能暴露其中一部分因果结构。
```

### 下一阶段任务

Phase125 应做：

```text
Joint Head-set + MLP-subspace Closure and Cross-heldout Validation
```

核心目标：

```text
验证 Phase124 的 head-set partial closure 是否真实泛化，
并测试 head set 与 MLP subspace 合并后能否接近 Phase122。
```

测试方案：

```text
1. 分离 selection prompts 与 evaluation prompts：
   - selection objects/category = 8
   - evaluation objects/category = 16 or 24

2. DS7B 为主，Qwen3/GLM4 对照。

3. 对 DS7B plant/number/container 分别测试：
   - best head set only
   - best pre-MLP subspace only
   - head set + pre-MLP subspace
   - head set + answer-side MLP/readout-side control

4. 加强 controls：
   - object head set
   - random head set
   - value-aligned but low-pre-attention set
   - same-size shuffled set

5. 直接对照 Phase122 residual pre_remove：
   - module combo / residual pre_remove 的 effect ratio
```

Phase125 关键判据：

```text
如果 head set + MLP subspace 在 independent evaluation prompts 上接近 residual pre_remove，
则 writer path 从 residual-level 推进到 module-set closure。

如果仍显著小于 residual pre_remove，
则说明当前模块级观测仍遗漏了关键 residual transformation，
下一步应进入 residual update decomposition / layernorm-mediated path。
```

## Phase 125: Joint Head-set MLP-subspace Cross-heldout Closure 联合模块集合独立验证 [2026-06-14 19:14]

### 本阶段目标

根据附加分析，Phase124 的结论基本正确，但存在一个关键硬伤：

```text
target_discovered head set 在同一批 heldout prompts 上选择和评估，
可能高估 head set closure。
```

因此 Phase125 做两件事：

```text
1. 将对象严格切成 train / selection / evaluation 三个不重叠集合。
2. 在 selection split 上选择 head set、pre-MLP layer、residual reference layer，
   只在 disjoint evaluation split 上报告效果。
```

核心问题：

```text
Phase124 的 DS7B plant head-set partial closure 是否真实泛化？
head set + pre-MLP subspace 是否能接近 Phase122 residual pre_remove？
```

### 生成脚本

- 主测试脚本：`tests/gpt5/phase125_joint_closure_crossheldout_cuda.py`
- 汇总脚本：`tests/gpt5/phase125_joint_closure_crossheldout_summary.py`

### 执行命令

```bash
python tests/gpt5/phase125_joint_closure_crossheldout_cuda.py qwen3 \
  --train-objects 2 \
  --selection-objects 2 \
  --eval-objects 2 \
  --batch-size 4 \
  --rank 4 \
  --layer-back 1 \
  --set-sizes 1,2 \
  --candidate-pool 4 \
  --categories number \
  --output-dir results/gpt5_phase125_smoke \
  --hard-exit-after-model

python tests/gpt5/phase125_joint_closure_crossheldout_cuda.py qwen3 \
  --train-objects 8 \
  --selection-objects 8 \
  --eval-objects 8 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --set-sizes 4,8,16 \
  --candidate-pool 24 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase125_joint_closure_crossheldout \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase125_joint_closure_crossheldout_cuda.py glm4 \
  --train-objects 8 \
  --selection-objects 8 \
  --eval-objects 8 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --set-sizes 4,8,16 \
  --candidate-pool 24 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase125_joint_closure_crossheldout \
  --hard-exit-after-model

python tests/gpt5/phase125_joint_closure_crossheldout_cuda.py deepseek7b \
  --train-objects 8 \
  --selection-objects 8 \
  --eval-objects 8 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --set-sizes 4,8,16 \
  --candidate-pool 24 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase125_joint_closure_crossheldout \
  --hard-exit-after-model

python tests/gpt5/phase125_joint_closure_crossheldout_summary.py

python -m py_compile \
  tests/gpt5/phase125_joint_closure_crossheldout_cuda.py \
  tests/gpt5/phase125_joint_closure_crossheldout_summary.py
```

### 测试范围

每个类别当前只有 24 个对象，因此严格对象不重叠时最大稳定切分为：

```text
train objects/category = 8
selection objects/category = 8
evaluation objects/category = 8
templates = 4
evaluation prompts/category = 32
```

模型层位：

```text
Qwen3: L32-L35, monitor L35
GLM4: L15-L18, monitor L18
DS7B: L24-L27, monitor L27
```

评估条件：

```text
residual_pre_reference
pre_mlp_subspace_only
head_set_only
head_set_plus_pre_mlp
object_control
random_control
low_pre_value_control
```

### 测试原理

1. train split 用来构造：

```text
answer monitor axis
pre-answer local category basis
```

2. selection split 用来选择：

```text
best head set
best pre-MLP subspace layer
best residual pre-remove reference layer
```

3. evaluation split 才报告最终效果。

4. 关键比例：

```text
effect_ratio_vs_residual_ref
= module_condition_target_delta / residual_pre_reference_target_delta
```

注意：

```text
只有 residual_pre_reference 本身足够强时，
ratio 才有解释意义。
```

Qwen3/GLM4 本轮 residual reference 很弱，因此它们的 ratio 不作为闭合比例解释。

### 结果文件

- Qwen3：`results/gpt5_phase125_joint_closure_crossheldout/phase125_qwen3_joint_closure_crossheldout.json`
- GLM4：`results/gpt5_phase125_joint_closure_crossheldout/phase125_glm4_joint_closure_crossheldout.json`
- DS7B：`results/gpt5_phase125_joint_closure_crossheldout/phase125_deepseek7b_joint_closure_crossheldout.json`
- 跨模型汇总：`results/gpt5_phase125_joint_closure_crossheldout/phase125_cross_model_summary.md`

### 客观结果

#### Qwen3

```text
residual reference weak:
  number -0.12
  container -0.02
  plant -0.19
```

因此 Qwen3 的 ratio 不解释为 closure ratio。

评估集上 head set 有 target_down：

```text
number head_set target_discovered k16 = -0.66
container head_set target_discovered k16 = -0.29
plant head_set target_discovered k16 = -0.89
```

但由于 pre-answer residual reference 本身弱，不能说明 pre-answer writer path closure。

#### GLM4 bf16

```text
residual reference weak or opposite:
  number +0.14
  container +0.05
  plant -0.02
```

GLM4 不形成可解释的 pre-answer residual closure。

#### DS7B

```text
number:
  residual reference = -2.55
  best head only = projection_discovered k16, -0.85, ratio 0.33
  best head+MLP = projection_discovered k16 + pre-MLP L24, -1.10, ratio 0.43
  best control = object_control k16, -0.47, ratio 0.18
  pre-MLP only = -0.43, ratio 0.17

container:
  residual reference = -2.71
  best head only = target_discovered k16, -0.11, ratio 0.04
  best head+MLP = -0.37, ratio 0.14
  best control = -0.17, ratio 0.06
  pre-MLP only = -0.32, ratio 0.12

plant:
  residual reference = -2.42
  best head only = target_discovered k16, -1.33, ratio 0.55
  best head+MLP = target_discovered k16 + pre-MLP L24, -1.80, ratio 0.74
  best control = low_pre_value_control k4, -0.19, ratio 0.08
  pre-MLP only = -0.69, ratio 0.28
```

DS7B evaluation objects：

```text
number:
  pair, triple, zero, first, second, third, many, few

container:
  carton, crate, vase, pot, pan, bowl, plate, tray

plant:
  ivy, weed, herb, lily, orchid, clover, palm, sapling
```

### 当前最可靠客观事实

1. **DS7B plant 的 Phase124 head-set 结果真实泛化**

selection split 选择，evaluation split 测试后：

```text
head only:
  target_delta -1.33
  ratio 0.55

head + pre-MLP:
  target_delta -1.80
  ratio 0.74

control:
  target_delta -0.19
  ratio 0.08
```

这是目前最强的 module-set closure 证据。

2. **DS7B number 有弱到中等泛化，但未闭合**

```text
head + pre-MLP:
  target_delta -1.10
  ratio 0.43

residual reference:
  -2.55
```

说明 number 的 residual subspace effect 仍有一半以上未被当前 head/MLP 组合解释。

3. **DS7B container 基本没有模块集合闭合**

```text
head + pre-MLP:
  target_delta -0.37
  ratio 0.14

residual reference:
  -2.71
```

container 仍是 residual-level effect 明显强于 module-level observable effect。

4. **pre-MLP subspace 与 head set 有加和**

DS7B plant：

```text
head only -1.33
pre-MLP only -0.69
head + pre-MLP -1.80
```

组合强于任一单项，但不是线性完全相加。

DS7B number：

```text
head only -0.85
pre-MLP only -0.43
combo -1.10
```

也有加和，但仍不足。

5. **Qwen3/GLM4 的 pre-answer residual reference 不稳定**

因此本轮不能用它们判断 module/residual closure ratio。

### 对 Phase124 的判定更新

Phase124 正确部分：

```text
head set 比 single head 更接近真实 effect；
value/readout alignment 比 attention mass 更有效；
DS7B plant 是最强 partial closure。
```

Phase125 新增：

```text
DS7B plant 的 head-set partial closure 经独立对象验证后仍成立；
head + pre-MLP 可以解释约 74% residual pre-answer reference。
```

但也要限制：

```text
number 只到约 43%；
container 只到约 14%；
所以不能说 pre-answer writer path 已整体模块闭合。
```

### 硬伤分析

1. **evaluation prompts/category 只有 32**

这是对象不重叠的代价。当前类别对象池只有 24 个，无法同时做到 train/selection/evaluation 都很大。

2. **selection split 仍参与选择**

虽然 evaluation 独立，但 head set 与 MLP layer 仍由 selection 决定，需要未来做多 seed/object split 复现。

3. **residual reference 使用同一 pre-answer basis**

它是合理 reference，但不是 Phase122 完全同构复刻，因为本轮做了 split 和 layer selection。

4. **head + MLP combo 仍未加入 LayerNorm / residual update 分解**

number/container 的缺口可能来自这些未测机制。

5. **Qwen3/GLM4 ratio 不可解释**

因为 residual reference 太弱，小分母会放大 ratio。

### 理论进展

当前可以更严格地写成：

```text
DS7B plant:
  pre-answer residual causal effect
    -> 约 55% 可由 selected head set 泛化解释
    -> 约 74% 可由 head set + pre-MLP subspace 泛化解释

DS7B number:
  module-set captures about 43%
  residual-level mechanism still contains unlocalized component

DS7B container:
  module-set captures little
  likely needs residual update / normalization / broader ensemble explanation
```

这说明：

```text
语言类别机制不是统一形态。
不同类别在同一模型中有不同程度的 module-level exposure。
```

中文：

```text
有些类别的因果子空间较容易被注意力头集合和 MLP 子空间暴露；
有些类别主要表现为残差流层面的结构，模块定位仍困难。
```

### 下一阶段任务

Phase126 应做：

```text
Residual Gap Decomposition for Number/Container
```

核心目标：

```text
解释为什么 number/container 的 residual pre-answer effect 远强于 head+MLP module combo。
```

测试方案：

```text
1. DS7B 优先，Qwen3/GLM4 只做轻量对照。

2. 对 number/container/plant 做 residual update 分解：
   - layer input residual
   - attention output
   - MLP output
   - layer output residual

3. 在 pre-answer positions 上分别移除 category subspace：
   - residual input subspace
   - attention output subspace
   - MLP output subspace
   - residual output subspace

4. 监控：
   - final target_delta
   - answer projection delta
   - module combo / residual output 的 ratio

5. 对 plant 作为正对照：
   - 验证 head+MLP 已接近 residual output；
   - 看残余 26% 来自哪里。
```

Phase126 关键判据：

```text
如果 residual output 强而 attention/MLP output 都弱，
则说明类别信息主要存在于 residual carry / normalization-mediated geometry。

如果 attention output + MLP output 合并接近 residual output，
则说明 Phase125 缺口来自模块组合不完整。

如果 layer input already strong，
则说明 writer 在更早层，需要回溯 upstream source。
```

## Phase 126: Residual Gap Decomposition 残差缺口分解 [2026-06-14 19:36]

### 本阶段目标

根据附加分析，Phase125 的结论正确：

```text
DS7B plant:
  head set + pre-MLP subspace 在独立 evaluation split 上解释约 74% residual reference。

DS7B number:
  约 43%，仍有明显 residual gap。

DS7B container:
  约 14%，基本未模块闭合。
```

Phase126 目标：

```text
解释 number/container 的 residual pre-answer effect 为什么远强于当前 head+MLP module combo。
```

具体做 residual update decomposition：

```text
layer_input residual
attention_output
MLP output
layer_output residual
attention_output + MLP output
```

### 生成脚本

- 主测试脚本：`tests/gpt5/phase126_residual_gap_decomposition_cuda.py`
- 汇总脚本：`tests/gpt5/phase126_residual_gap_decomposition_summary.py`

### 执行命令

```bash
python tests/gpt5/phase126_residual_gap_decomposition_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --rank 4 \
  --layer-back 1 \
  --categories number \
  --output-dir results/gpt5_phase126_smoke \
  --hard-exit-after-model

python tests/gpt5/phase126_residual_gap_decomposition_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase126_residual_gap_decomposition \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase126_residual_gap_decomposition_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase126_residual_gap_decomposition \
  --hard-exit-after-model

python tests/gpt5/phase126_residual_gap_decomposition_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase126_residual_gap_decomposition \
  --hard-exit-after-model

python tests/gpt5/phase126_residual_gap_decomposition_summary.py

python -m py_compile \
  tests/gpt5/phase126_residual_gap_decomposition_cuda.py \
  tests/gpt5/phase126_residual_gap_decomposition_summary.py
```

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
rank = 16
scale = 1.5

Qwen3 layers = L32-L35, monitor L35
GLM4 layers = L15-L18, monitor L18
DS7B layers = L24-L27, monitor L27
```

组件：

```text
layer_input
attention_output
mlp_output
layer_output
attention_plus_mlp
```

### 测试原理

对每个 model/category/layer/component：

```text
1. 在 train objects 上捕获 pre-answer positions 的 component activation。
2. 用 category contrast matrix 构造该 component 的 local category subspace。
3. 在 heldout prompts 上从对应 component 的 pre-answer positions 中移除该 subspace。
4. 监控 final target_delta 与 peak answer projection delta。
```

hook 位置：

```text
layer_input:
  transformer block forward_pre_hook，patch 输入 residual。

attention_output:
  self-attention module forward_hook，patch attention output。

mlp_output:
  MLP module forward_hook，patch MLP output。

layer_output:
  transformer block forward_hook，patch block output residual。
```

### 结果文件

- Qwen3：`results/gpt5_phase126_residual_gap_decomposition/phase126_qwen3_residual_gap_decomposition.json`
- GLM4：`results/gpt5_phase126_residual_gap_decomposition/phase126_glm4_residual_gap_decomposition.json`
- DS7B：`results/gpt5_phase126_residual_gap_decomposition/phase126_deepseek7b_residual_gap_decomposition.json`
- 跨模型汇总：`results/gpt5_phase126_residual_gap_decomposition/phase126_cross_model_summary.md`

### 客观结果

#### Qwen3

```text
number:
  layer_input best = -0.13
  attention_output best = -0.06
  mlp_output best = -0.00
  layer_output best = -0.13

container:
  layer_input best = -0.09
  attention_output best = -0.01
  mlp_output best = -0.07
  layer_output best = -0.04

plant:
  layer_input best = -0.22
  attention_output best = +0.00
  mlp_output best = +0.04
  layer_output best = -0.22
```

Qwen3 的 pre-answer residual path 仍弱。

#### GLM4 bf16

```text
number:
  layer_input best = -0.26
  attention_output best = -0.08
  mlp_output best = -0.09
  layer_output best = -0.26

container:
  layer_input best = -0.01
  attention_output best = -0.02
  mlp_output best = -0.04
  layer_output best = -0.01

plant:
  layer_input best = -0.02
  attention_output best = +0.01
  mlp_output best = -0.04
  layer_output best = -0.02
```

GLM4 仍弱。

#### DS7B

总体 best：

```text
number:
  layer_input best = L25, target_delta -2.05, answer_proj_delta -50.06
  attention_output best = L27, target_delta -0.03
  mlp_output best = L24, target_delta -0.28
  layer_output best = L27, target_delta -2.51
  attention_plus_mlp best = L24, target_delta -0.25

container:
  layer_input best = L26, target_delta -1.22
  attention_output best = L24, target_delta -0.07
  mlp_output best = L24, target_delta -0.32
  layer_output best = L27, target_delta -2.66
  attention_plus_mlp best = L24, target_delta -0.25

plant:
  layer_input best = L25, target_delta -2.28
  attention_output best = L24, target_delta -0.09
  mlp_output best = L25, target_delta -0.18
  layer_output best = L27, target_delta -2.42
  attention_plus_mlp best = L26, target_delta -0.13
```

DS7B 逐层关键值：

```text
number:
  L24 layer_input -1.45 -> layer_output -2.05
  L25 layer_input -2.05 -> layer_output -1.33
  L26 layer_input -1.33 -> layer_output -0.96
  L27 layer_input -0.96 -> layer_output -2.51
  attention_output all about 0
  mlp_output max about -0.28

container:
  L24 layer_input -1.05 -> layer_output -1.19
  L25 layer_input -1.19 -> layer_output -1.22
  L26 layer_input -1.22 -> layer_output -0.98
  L27 layer_input -0.98 -> layer_output -2.66
  attention_output all about 0
  mlp_output max about -0.32

plant:
  L24 layer_input -0.93 -> layer_output -2.28
  L25 layer_input -2.28 -> layer_output -2.25
  L26 layer_input -2.25 -> layer_output -1.21
  L27 layer_input -1.21 -> layer_output -2.42
  attention_output all about 0
  mlp_output max about -0.18
```

### 当前最可靠客观事实

1. **DS7B 的 pre-answer residual effect 在 layer_input 已经很强**

```text
number L25 layer_input -2.05
container L26 layer_input -1.22
plant L25 layer_input -2.28
```

这说明强残差场不是当前层 attention/MLP output 刚写出来的，而是已经进入该层。

2. **DS7B attention_output category subspace 几乎不能解释 residual effect**

```text
number best attention_output -0.03
container best attention_output -0.07
plant best attention_output -0.09
```

这说明 Phase125 的 head-set effect 不等价于单层 attention_output category subspace removal。

3. **DS7B MLP output category subspace 只有弱效应**

```text
number best mlp_output -0.28
container best mlp_output -0.32
plant best mlp_output -0.18
```

比 layer_output 的 -2 级别弱很多。

4. **attention_output + MLP output 仍然不能接近 layer_output**

```text
number attention_plus_mlp -0.25 vs layer_output -2.51
container attention_plus_mlp -0.25 vs layer_output -2.66
plant attention_plus_mlp -0.13 vs layer_output -2.42
```

因此 Phase125 缺口不是简单“同层 attention output + MLP output 子空间未合并”造成的。

5. **L27 layer_output 有 hook 时序特殊性**

L27 layer_output patch 直接改变同层最终 hidden state，因此 answer_proj_delta 为 0 是预期：

```text
patch 与 monitor 位于同一 hidden_state 输出边界，
不能用 answer_proj_delta=0 解释为无关系。
```

### 对 Phase125 的判定更新

Phase125 正确部分保留：

```text
plant 的 module-set partial closure 可泛化；
number 中等；
container 弱。
```

Phase126 增加限制：

```text
同层 attention_output / MLP_output 的 category subspace removal
不能解释 residual output 强效。
```

更准确说：

```text
Phase125 的 head-set 效应可能是对 residual stream 中已存在因果子空间的局部扰动，
而不是当前层 raw attention_output category subspace 的直接完整写入。
```

### 硬伤分析

1. **attention_output / MLP_output basis 是各自 raw output basis**

如果真实写入经过 residual addition、LayerNorm 或后续几何变换才成形，raw output basis 会低估模块贡献。

2. **layer_input 强说明需要回溯更早层**

本轮只扫 peak-3 到 peak，没有向更早层寻找最初写入点。

3. **layer_output patch 与 monitor 层重合时 answer_proj_delta 不可直接解释**

尤其 L27。

4. **attention head set 与 attention_output subspace 不是同一干预**

head set zero 是按 head slice 切断贡献；
attention_output subspace removal 是按 category basis 移除 raw attention output 中的低秩方向。
二者不能简单等价。

5. **没有直接分解 residual carry 与 LayerNorm**

当前结果提示这些机制重要，但尚未直接测。

### 理论进展

当前 DS7B pre-answer 机制要从：

```text
attention/MLP writer writes category subspace in current layer
```

修正为：

```text
category subspace is already present in residual input;
current layer modules expose or perturb it only weakly;
strong causal control is carried by residual stream and transformed near layer output/readout boundary.
```

中文：

```text
类别因果子空间已经在层输入残差中存在；
当前层注意力和 MLP 原始输出只暴露很小一部分；
强因果效应主要沿残差流携带，并在层输出或读出边界附近重新显现。
```

这解释了为什么：

```text
Phase125 plant head+MLP 有 74% closure，
但 Phase126 raw attention_output + raw MLP_output 很弱。
```

二者测的不是同一层面：

```text
head set ablation:
  切断模块对后续残差流的贡献，可能扰动累积路径。

raw output subspace removal:
  只移除当前输出张量中的线性类别方向，不能覆盖残差携带几何。
```

### 下一阶段任务

Phase127 应做：

```text
Upstream Residual Carry Backtrace
```

核心目标：

```text
沿更早层回溯 pre-answer layer_input causal subspace，
找到 DS7B number/container/plant 的 residual causal field 首次形成层。
```

测试方案：

```text
1. DS7B 为主，Qwen3/GLM4 轻量对照。
2. 扫描更宽层位：
   - DS7B L12-L27
   - Qwen3 L20-L35
   - GLM4 L8-L18
3. 只测 layer_input 与 layer_output 两个 residual sites。
4. 对 number/container/plant 构造 pre-answer local category subspace。
5. 记录每层 target_delta 曲线：
   - onset layer
   - peak layer
   - decay/re-emergence layer
6. 对 DS7B L27 特殊重显现做细查：
   - L26 output -> L27 input -> L27 output
   - 是否存在 final block normalization/readout gateway。
```

Phase127 关键判据：

```text
如果 layer_input effect 在早层已经出现并逐层携带，
说明 pre-answer field 是长期 residual memory。

如果中晚层突然出现，
说明有明确 upstream writer layer。

如果 L27 output 大幅强于 L27 input，
说明 final block / normalization / readout gateway 重新放大 residual category field。
```

## Phase 127: Upstream Residual Carry Backtrace 上游残差携带回溯 [2026-06-14 20:42]

### 本阶段目标

根据附加分析，Phase126 的判断正确：

```text
DS7B number/container/plant 的强 pre-answer causal field
不是当前层 attention_output 或 MLP_output 直接写出；
它已经存在于 layer_input residual，并沿 residual stream 携带。
```

Phase127 继续做宽层回溯：

```text
沿更早层扫描 pre-answer layer_input / layer_output category subspace，
寻找 residual causal field 的 onset layer、peak layer、衰减和 final re-emergence。
```

### 生成脚本

- 主测试脚本：`tests/gpt5/phase127_upstream_residual_carry_backtrace_cuda.py`
- 汇总脚本：`tests/gpt5/phase127_upstream_residual_carry_backtrace_summary.py`

### 执行命令

```bash
python tests/gpt5/phase127_upstream_residual_carry_backtrace_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --rank 4 \
  --layer-from 34 \
  --layer-to 35 \
  --categories number \
  --output-dir results/gpt5_phase127_smoke \
  --hard-exit-after-model

python tests/gpt5/phase127_upstream_residual_carry_backtrace_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-from 20 \
  --layer-to 35 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase127_upstream_residual_carry_backtrace \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase127_upstream_residual_carry_backtrace_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-from 8 \
  --layer-to 18 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase127_upstream_residual_carry_backtrace \
  --hard-exit-after-model

python tests/gpt5/phase127_upstream_residual_carry_backtrace_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-from 12 \
  --layer-to 27 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase127_upstream_residual_carry_backtrace \
  --hard-exit-after-model

python tests/gpt5/phase127_upstream_residual_carry_backtrace_summary.py

python -m py_compile \
  tests/gpt5/phase127_upstream_residual_carry_backtrace_cuda.py \
  tests/gpt5/phase127_upstream_residual_carry_backtrace_summary.py
```

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
rank = 16
scale = 1.5
onset threshold = target_delta <= -0.5
```

层位：

```text
Qwen3: L20-L35, monitor L35
GLM4: L8-L18, monitor L18
DS7B: L12-L27, monitor L27
```

只测 residual sites：

```text
layer_input
layer_output
```

### 测试原理

1. 在 peak answer layer 构造 answer monitor axis。
2. 对每个扫描层：

```text
capture pre-answer layer_input residual centers
capture pre-answer layer_output residual centers
```

3. 对每个 category 构造 local category subspace。
4. 在 heldout prompts 上从对应 residual site 移除 category subspace。
5. 记录每层 target_delta 曲线。

关键指标：

```text
onset layer:
  第一个 target_delta <= -0.5 的层。

best layer:
  target_delta 最低的层。

final input/output:
  最后扫描层的输入/输出差异，用于判断 final re-emergence。
```

### 结果文件

- Qwen3：`results/gpt5_phase127_upstream_residual_carry_backtrace/phase127_qwen3_upstream_residual_carry_backtrace.json`
- GLM4：`results/gpt5_phase127_upstream_residual_carry_backtrace/phase127_glm4_upstream_residual_carry_backtrace.json`
- DS7B：`results/gpt5_phase127_upstream_residual_carry_backtrace/phase127_deepseek7b_upstream_residual_carry_backtrace.json`
- 跨模型汇总：`results/gpt5_phase127_upstream_residual_carry_backtrace/phase127_cross_model_summary.md`

### 客观结果

#### Qwen3

```text
number:
  input onset L21
  output onset L20
  best input L30 target_delta -0.92
  best output L29 target_delta -0.92
  final input L35 -0.05
  final output L35 -0.07

container:
  input onset L21
  output onset L20
  best input L21 target_delta -0.79
  best output L20 target_delta -0.79
  final input L35 -0.04
  final output L35 +0.07

plant:
  input onset L20
  output onset L20
  best input L26 target_delta -0.88
  best output L25 target_delta -0.88
  final input L35 +0.27
  final output L35 +0.24
```

Qwen3 有中层 residual effects，但 final pre-answer residual path 不稳定，与前几阶段一致。

#### GLM4 bf16

```text
number:
  no onset by -0.5 threshold
  best input L18 -0.26
  best output L17 -0.26

container:
  no onset
  best input L13 -0.08
  best output L12 -0.08

plant:
  no onset
  best input L13 -0.08
  best output L12 -0.08
```

GLM4 继续弱。

#### DS7B

汇总：

```text
number:
  input onset L21
  output onset L20
  best input L25 -2.05
  best output L27 -2.51
  final input L27 -0.96
  final output L27 -2.51

container:
  input onset L20
  output onset L19
  best input L26 -1.22
  best output L27 -2.66
  final input L27 -0.98
  final output L27 -2.66

plant:
  input onset L23
  output onset L22
  best input L25 -2.28
  best output L27 -2.42
  final input L27 -1.21
  final output L27 -2.42
```

DS7B 逐层曲线：

```text
number layer_input:
  L12 +0.02, L13 +0.10, L14 +0.03, L15 +0.15,
  L16 +0.07, L17 +0.04, L18 -0.14, L19 +0.08,
  L20 -0.45, L21 -0.56, L22 -0.74, L23 -1.02,
  L24 -1.45, L25 -2.05, L26 -1.33, L27 -0.96

number layer_output:
  L12 +0.10, L13 +0.03, L14 +0.15, L15 +0.07,
  L16 +0.04, L17 -0.14, L18 +0.08, L19 -0.45,
  L20 -0.56, L21 -0.74, L22 -1.02, L23 -1.45,
  L24 -2.05, L25 -1.33, L26 -0.96, L27 -2.51

container layer_input:
  L12 +0.02, L13 +0.33, L14 +0.01, L15 +0.08,
  L16 +0.14, L17 -0.09, L18 -0.26, L19 -0.28,
  L20 -0.60, L21 -0.80, L22 -0.86, L23 -0.88,
  L24 -1.05, L25 -1.19, L26 -1.22, L27 -0.98

container layer_output:
  L12 +0.33, L13 +0.01, L14 +0.08, L15 +0.14,
  L16 -0.09, L17 -0.26, L18 -0.28, L19 -0.60,
  L20 -0.80, L21 -0.86, L22 -0.88, L23 -1.05,
  L24 -1.19, L25 -1.22, L26 -0.98, L27 -2.66

plant layer_input:
  L12 +0.19, L13 +0.70, L14 +0.47, L15 +0.52,
  L16 +0.64, L17 +0.47, L18 +0.16, L19 +0.16,
  L20 -0.23, L21 -0.34, L22 -0.31, L23 -0.67,
  L24 -0.93, L25 -2.28, L26 -2.25, L27 -1.21

plant layer_output:
  L12 +0.70, L13 +0.47, L14 +0.52, L15 +0.64,
  L16 +0.47, L17 +0.16, L18 +0.16, L19 -0.23,
  L20 -0.34, L21 -0.31, L22 -0.67, L23 -0.93,
  L24 -2.28, L25 -2.25, L26 -1.21, L27 -2.42
```

### 当前最可靠客观事实

1. **DS7B pre-answer residual field 不是从很早层开始**

L12-L18 基本无效或弱：

```text
number L12-L18 mostly around 0
container L12-L18 mostly weak
plant L12-L18 positive/weak
```

说明它不是全程长期 memory，而是在中后层逐渐形成。

2. **onset 出现在中后层**

```text
number input onset L21, output onset L20
container input onset L20, output onset L19
plant input onset L23, output onset L22
```

3. **layer_output Lk 基本传递为 layer_input L(k+1)**

例如 number：

```text
L23 output -1.45 ≈ L24 input -1.45
L24 output -2.05 ≈ L25 input -2.05
L25 output -1.33 ≈ L26 input -1.33
L26 output -0.96 ≈ L27 input -0.96
```

这直接支持：

```text
pre-answer causal field 沿 residual stream carry。
```

4. **L27 output 出现 final re-emergence / readout-boundary amplification**

DS7B：

```text
number:
  L27 input -0.96 -> L27 output -2.51

container:
  L27 input -0.98 -> L27 output -2.66

plant:
  L27 input -1.21 -> L27 output -2.42
```

这是 Phase127 的最大新发现。

5. **DS7B 三类的形成曲线不同**

```text
number:
  L20-L25 逐步增强，L26-L27 input 衰减，L27 output 重显现。

container:
  L19-L26 缓慢增强，L27 output 突然大幅增强。

plant:
  L22-L24 开始，L25-L26 强峰，L27 input 衰减，L27 output 重显现。
```

### 对 Phase126 的判定更新

Phase126 正确部分：

```text
强效在 layer_input 已经存在；
当前层 raw attention/MLP output 很弱；
机制应转向 residual carry。
```

Phase127 新增：

```text
这个 residual carry 不是从很早层就存在；
它在 DS7B 中后层形成，并在 L27 output/readout boundary 重新放大。
```

更严格表述：

```text
DS7B pre-answer causal field has a mid-late onset, residual carry phase, and final output re-emergence.
```

中文：

```text
DS7B 的答案前因果场有中后层起点、残差携带阶段、末层输出重显现三个阶段。
```

### 硬伤分析

1. **只测 L12 起**

虽然 L12-L18 基本弱，但仍未证明 L1-L11 绝对无关。

2. **onset threshold 是经验阈值 -0.5**

不同阈值会改变 onset layer 的精确位置，但不改变中后层增强趋势。

3. **仍是 residual site patch**

没有直接定位导致 onset 的具体模块或 LayerNorm 操作。

4. **L27 output 与 monitor 边界重合**

final re-emergence 很强，但需要进一步确认是否来自 final block residual output、final norm、或 readout coupling。

5. **Qwen3 中层有 residual effects 但 final 弱**

这可能说明它有中层临时 residual field，但没有 DS7B 式 final carry/readout closure。

### 理论进展

当前可以把 DS7B pre-answer causal pathway 分为三段：

```text
1. Mid-late formation:
   number/container around L19-L21;
   plant around L22-L23.

2. Residual carry:
   layer_output Lk -> layer_input L(k+1)
   causal field 被残差流携带、增强或衰减。

3. Final re-emergence:
   L27 input 较弱；
   L27 output 强烈重新放大并接近 Phase120/122 的 pre-answer residual effect。
```

这对破解语言编码机制的意义：

```text
类别信息不是简单由某层某模块一次写入；
它在中后层形成为 residual causal field，
沿残差流传递，
并在读出边界附近被重新对齐/放大。
```

### 下一阶段任务

Phase128 应做：

```text
Final Block Re-emergence and Norm/Readout Gateway Test
```

核心目标：

```text
解释 L27 output 为什么比 L27 input 强很多。
```

测试方案：

```text
1. DS7B 优先，number/container/plant。
2. 精细拆 L27：
   - L27 input
   - L27 attention output
   - L27 post-attention residual
   - L27 MLP input
   - L27 MLP output
   - L27 block output
   - final norm output if accessible
3. 构造 pre-answer category subspace 并 patch。
4. 监控 final logits 和 answer projection。
5. 对比：
   - L26 output
   - L27 input
   - L27 output
   - final norm
```

Phase128 关键判据：

```text
如果 L27 block output 强但 final norm 不改变结构，
则 re-emergence 来自 final block 内部。

如果 final norm output 才强，
则 normalization/readout gateway 是关键。

如果 L27 MLP output 或 post-MLP residual 是转折点，
则 final MLP/residual addition 是放大器。
```

## Phase 128: Boundary Peak Gateway Split 边界峰值层门控拆分 [2026-06-14 21:11]

### 本阶段目标

根据附件分析，Phase127 的判断基本正确：DS7B 的 pre-answer causal field 不是早层长期静态存储，而是中后层形成、沿 residual stream 携带，并在输出边界附近增强。

本阶段先按附件建议拆分所谓 final block / final norm / readout gateway。执行后发现一个关键修正：

```text
Qwen3 boundary peak = L35, true last = L36
GLM4 boundary peak = L18, true last = L40
DS7B boundary peak = L27, true last = L28
```

因此 Phase127 中的 L27 output 对 DS7B 来说不是模型真实最后一层输出，而是 true last layer L28 的输入。这一点非常关键，不能把 L27 output 直接解释成 final norm/readout 后状态。

### 执行命令

```bash
python tests/gpt5/phase128_final_block_gateway_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --categories number \
  --output-dir results/gpt5_phase128_smoke \
  --hard-exit-after-model

python tests/gpt5/phase128_final_block_gateway_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase128_final_block_gateway \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase128_final_block_gateway_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase128_final_block_gateway \
  --hard-exit-after-model

python tests/gpt5/phase128_final_block_gateway_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase128_final_block_gateway \
  --hard-exit-after-model

python tests/gpt5/phase128_final_block_gateway_summary.py

python -m py_compile \
  tests/gpt5/phase128_final_block_gateway_cuda.py \
  tests/gpt5/phase128_final_block_gateway_summary.py
```

### 生成脚本与结果

- 主脚本：`tests/gpt5/phase128_final_block_gateway_cuda.py`
- 汇总脚本：`tests/gpt5/phase128_final_block_gateway_summary.py`
- Qwen3 结果：`results/gpt5_phase128_final_block_gateway/phase128_qwen3_final_block_gateway.json`
- GLM4 结果：`results/gpt5_phase128_final_block_gateway/phase128_glm4_final_block_gateway.json`
- DS7B 结果：`results/gpt5_phase128_final_block_gateway/phase128_deepseek7b_final_block_gateway.json`
- 跨模型汇总：`results/gpt5_phase128_final_block_gateway/phase128_cross_model_summary.md`

### 测试原理

在 boundary peak layer 上，对 pre-answer tokens 的类别子空间做 subspace removal，比较以下 site：

```text
block_input
attention_output
post_attention_norm_input
mlp_input
mlp_output
block_output
final_norm_input
final_norm_output
```

同时记录 position audit，检查 pre-answer positions 是否包含 answer token。

### 客观结果

#### Qwen3

```text
number:
  block_input target Δ -0.05
  block_output target Δ -0.07
  final_norm_input/output target Δ 0.00

container:
  block_input target Δ -0.04
  block_output target Δ +0.07
  final_norm_input/output target Δ 0.00

plant:
  block_input target Δ +0.27
  block_output target Δ +0.24
  final_norm_input/output target Δ 0.00
```

Qwen3 在 boundary peak 后段没有 DS7B 式强 pre-answer causal field。

#### GLM4

```text
number:
  block_input target Δ -0.26
  block_output target Δ -0.23
  final_norm_input/output target Δ 0.00

container:
  block_input target Δ -0.01
  block_output target Δ +0.00
  final_norm_input/output target Δ 0.00

plant:
  block_input target Δ -0.02
  block_output target Δ -0.01
  final_norm_input/output target Δ 0.00
```

Phase128 原始 GLM4 结果较弱，但随后 Phase129 发现 GLM4 旧 answer position 口径存在 left padding mismatch，因此 GLM4 旧结果只能作为低可信参考。

#### DS7B

```text
number:
  block_input target Δ -0.96
  block_output target Δ -2.51
  final_norm_input/output target Δ 0.00

container:
  block_input target Δ -0.98
  block_output target Δ -2.66
  final_norm_input/output target Δ 0.00

plant:
  block_input target Δ -1.21
  block_output target Δ -2.42
  final_norm_input/output target Δ 0.00
```

DS7B 的强效应仍然稳定，但 Phase128 暴露了一个解释硬伤：L27 不是 true last layer，不能把 L27 output 直接叫 final block output。

### 阶段判断

Phase128 的正确部分：

```text
1. DS7B L27 block_output 强效应复现。
2. pre-answer positions 没有直接包含 answer token。
3. final_norm_input/output patch 为 0，说明 final norm 后没有跨位置传播。
```

Phase128 的硬伤：

```text
1. boundary peak layer 被误称为 final block。
2. GLM4 使用旧 answer_pos = sum(mask)-1 口径，在 left padding 下不可靠。
3. 必须用真实最后非 pad token 和真实 batched token grid 重做审计。
```

因此继续执行 Phase129。

## Phase 129: Position-corrected True Last Gateway Audit 位置修正版真实末层门控审计 [2026-06-14 21:11]

### 本阶段目标

修正 Phase128 暴露的两个问题：

```text
1. 用真实最后一个非 pad token 作为 answer position。
2. 在 actual batched token grid 中重新定位 object 后 pre-answer tokens。
3. 区分 boundary peak layer 与 true last layer。
```

核心判据：

```text
如果 peak_block_output == last_block_input 且 last_block_output/final_norm 为 0，
说明 pre-answer causal field 在 true last layer 输入前有效，
但通过 true last layer 后不再能以 pre-answer 位置直接影响 answer logits。
```

### 执行命令

```bash
python tests/gpt5/phase129_position_corrected_gateway_audit_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --categories number \
  --output-dir results/gpt5_phase129_smoke \
  --hard-exit-after-model

python tests/gpt5/phase129_position_corrected_gateway_audit_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase129_position_corrected_gateway_audit \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase129_position_corrected_gateway_audit_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase129_position_corrected_gateway_audit \
  --hard-exit-after-model

python tests/gpt5/phase129_position_corrected_gateway_audit_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase129_position_corrected_gateway_audit \
  --hard-exit-after-model

python tests/gpt5/phase129_position_corrected_gateway_audit_summary.py

python -m py_compile \
  tests/gpt5/phase128_final_block_gateway_cuda.py \
  tests/gpt5/phase128_final_block_gateway_summary.py \
  tests/gpt5/phase129_position_corrected_gateway_audit_cuda.py \
  tests/gpt5/phase129_position_corrected_gateway_audit_summary.py
```

### 生成脚本与结果

- 主脚本：`tests/gpt5/phase129_position_corrected_gateway_audit_cuda.py`
- 汇总脚本：`tests/gpt5/phase129_position_corrected_gateway_audit_summary.py`
- Qwen3 结果：`results/gpt5_phase129_position_corrected_gateway_audit/phase129_qwen3_position_corrected_gateway_audit.json`
- GLM4 结果：`results/gpt5_phase129_position_corrected_gateway_audit/phase129_glm4_position_corrected_gateway_audit.json`
- DS7B 结果：`results/gpt5_phase129_position_corrected_gateway_audit/phase129_deepseek7b_position_corrected_gateway_audit.json`
- 跨模型汇总：`results/gpt5_phase129_position_corrected_gateway_audit/phase129_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
rank = 16
scale = 1.5
sites = peak_block_input, peak_block_output, last_block_input, last_block_output, final_norm_input, final_norm_output
```

真实层位：

```text
Qwen3: peak L35, true last L36
GLM4: peak L18, true last L40
DS7B: peak L27, true last L28
```

### 客观结果

#### Qwen3

```text
position audit:
  answer_in_pre = 0
  old_answer_pos_mismatch = 0

number:
  peak input -0.05
  peak output -0.07
  last input -0.07
  last output 0.00
  final norm 0.00

container:
  peak input -0.04
  peak output +0.07
  last input +0.07
  last output 0.00
  final norm 0.00

plant:
  peak input +0.27
  peak output +0.24
  last input +0.24
  last output 0.00
  final norm 0.00
```

Qwen3 的 boundary peak output 与 true last input 对齐，但效应很弱或为 target-up，不构成强 pre-answer causal pathway。

#### GLM4 bf16

```text
position audit:
  number old_answer_pos_mismatch = 32 / 64
  container old_answer_pos_mismatch = 62 / 64
  plant old_answer_pos_mismatch = 52 / 64
```

这说明 GLM4 早前使用旧位置口径的 pre-answer / answer-site 结果存在明显索引风险。

修正后：

```text
number:
  peak input -0.47
  peak output -0.61
  last input -0.05
  last output 0.00
  final norm 0.00

container:
  peak input -0.26
  peak output -0.32
  last input +0.02
  last output 0.00
  final norm 0.00

plant:
  peak input -0.17
  peak output -0.25
  last input -0.15
  last output 0.00
  final norm 0.00
```

GLM4 修正后出现比旧结果更清楚的 peak-layer 弱中等效应，尤其 number。但 true last output 与 final norm 仍为 0。

#### DS7B

```text
position audit:
  answer_in_pre = 0
  old_answer_pos_mismatch = 0
```

核心曲线：

```text
number:
  peak input -0.96
  peak output -2.51
  last input -2.51
  last output 0.00
  final norm 0.00

container:
  peak input -0.98
  peak output -2.66
  last input -2.66
  last output 0.00
  final norm 0.00

plant:
  peak input -1.21
  peak output -2.42
  last input -2.42
  last output 0.00
  final norm 0.00
```

这是本轮最关键的客观结果。

### 当前最可靠客观事实

1. **DS7B 的强 pre-answer causal field 位于 true last layer 输入之前**

```text
peak_block_output L27 == last_block_input L28
number: -2.51 == -2.51
container: -2.66 == -2.66
plant: -2.42 == -2.42
```

这说明 Phase127 的 L27 output 重显现，准确说是：

```text
boundary-peak residual field becomes the input consumed by the true last layer.
```

不是 final norm 后状态。

2. **true last layer output 的 pre-answer patch 为 0**

```text
DS7B last_block_output:
  number 0.00
  container 0.00
  plant 0.00
```

这符合 causal transformer 结构：最后一层输出后，pre-answer token 不再通过后续 attention 影响 answer token。

3. **final_norm_input/output 的 pre-answer patch 为 0**

```text
所有模型所有类别 final norm patch 均为 0.00
```

这排除了“final norm 自身让 pre-answer 位置继续跨位置影响 answer logits”的解释。

4. **GLM4 旧位置口径存在硬伤**

GLM4 left padding 和特殊 token 使旧口径错位：

```text
old_answer_pos_mismatch:
number 32/64
container 62/64
plant 52/64
```

因此 GLM4 之前所有依赖 answer_pos = sum(mask)-1 的绝对结论，需要用 Phase129 的 corrected position 口径复核。

5. **Qwen3/DS7B 旧位置口径在本测试中没有 mismatch**

```text
Qwen3 mismatch = 0
DS7B mismatch = 0
```

因此 DS7B Phase126/127 的核心曲线不是 padding 错位造成的。

### 对 Phase127 附件分析的判定

正确部分：

```text
1. DS7B 确实存在 mid-late formation、residual carry、boundary re-emergence。
2. L27 output 强于 L27 input 的现象真实复现。
3. 需要拆 final block / final norm / readout gateway 的方向正确。
```

需要修正部分：

```text
1. DS7B L27 不是 true final block，而是 true last L28 的输入前一层。
2. final re-emergence 应改称 boundary-peak re-emergence 或 pre-last-layer gateway。
3. final norm/readout gateway 不是 pre-answer 跨位置因果效应来源。
```

### 理论进展

当前更准确的结构是：

```text
1. 中后层形成 category residual causal field。
2. 该 field 沿 residual stream 携带。
3. 在 true last layer 前的 boundary peak layer 输出处变强。
4. true last layer 的 answer token 可以通过 attention 读取这些 pre-answer residual states。
5. true last layer 输出之后，pre-answer token 不再能改变 answer logits。
```

因此，Phase127 的公式应修正为：

```text
R_c^{l+1}(P)
=
T_l(R_c^l(P))
+ U_l^{write}(P)
+ η_l
```

并在输出端加入：

```text
A_c^{L}(answer)
=
ReadLastLayer(
  h_answer^{L-1},
  R_c^{L-1}(P)
)
```

其中：

```text
R_c^{L-1}(P):
  true last layer 输入前的 pre-answer residual causal field

ReadLastLayer:
  true last layer 内部把 pre-answer field 转换到 answer token / logits 的读取算子
```

这比“final norm/readout 直接放大 pre-answer field”更符合 Phase129 结果。

### 硬伤和瓶颈

1. **GLM4 历史结果需要重审**

只要旧脚本使用 `attention_mask.sum(dim=1)-1` 并且直接用未加特殊 token 的位置索引，就可能在 GLM4 上错位。后续跨模型脚本必须统一使用 corrected token grid。

2. **DS7B 的 true last layer 内部读取机制尚未定位**

Phase129 说明强场在 L28 input 有效，但 L28 output pre-answer 无效。下一步必须研究：

```text
L28 attention at answer token 是否读取 L27/L28 input 的 pre-answer field。
```

3. **当前只测 subspace removal，不测生成**

logits 现象已经稳定，但还没有真实 generation audit。

4. **center-vs-others 子空间仍可能混入竞争方向**

plant 的 release 很强，说明 target-down 之外仍有 competitor release / interface 成分。

### 下一阶段大任务

Phase130 应做：

```text
True Last Attention Read Gateway Mapping
```

目标不是再泛泛拆模块，而是直接验证：

```text
DS7B L28 answer token 是否通过 attention 从 pre-answer tokens 读取 L27/L28 input 的 category causal field。
```

测试方案：

```text
1. 使用 Phase129 corrected position 口径。
2. 固定 DS7B 优先，同时保留 Qwen3/GLM4 对照。
3. 在 true last layer 做 answer-token attention output patch，而不是 pre-answer output patch。
4. 分 head 测：
   - patch attention value/write at answer token
   - patch attention output head slice at answer token
   - mask or replace attention contribution from pre-answer tokens to answer token if implementation permits
5. 比较：
   - peak_block_output / last_block_input subspace
   - true last attention output at answer token
   - true last MLP input/output at answer token
6. 对 DS7B number/container/plant 使用 train 8, test 16 起步；关键 head 命中后扩大到 train 12, test 24。
```

关键判据：

```text
如果 true last attention answer-site patch 能复现 L28 input 的 target-down，
则可确认 pre-answer residual field 通过最后一层 attention 被读到 answer token。

如果 attention 不强而 MLP/answer residual 强，
则最后一层可能是 answer-site nonlinear readout，而不是跨位置读取。
```

## Phase 130: True Last Attention Read Gateway Mapping 真实末层读取门控定位 [2026-06-14 21:31]

### 本阶段目标

根据附件判断，Phase128/129 的修正是正确的：DS7B 的强 pre-answer causal field 不在 final norm 后继续跨位置传播，而是在 true last layer input 之前达到强状态。真正问题变成：

```text
true last layer 的 answer token 是否通过 attention / MLP / residual update
把 pre-answer residual field 转换为 answer-site readout field。
```

本阶段测试两个对象：

```text
1. answer-site components:
   last_attention_output_answer
   last_mlp_input_answer
   last_mlp_output_answer
   last_block_output_answer
   final_norm_output_answer

2. true last layer attention heads:
   按 answer token 对 pre-answer tokens 的 attention mass 选 top heads，
   在 o_proj input head slice 上做 answer-position head ablation。
```

### 执行命令

```bash
python tests/gpt5/phase130_true_last_attention_read_gateway_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --categories number \
  --top-k-heads 2 \
  --output-dir results/gpt5_phase130_smoke \
  --hard-exit-after-model

python tests/gpt5/phase130_true_last_attention_read_gateway_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --categories number,container,plant \
  --top-k-heads 8 \
  --output-dir results/gpt5_phase130_true_last_attention_read_gateway \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase130_true_last_attention_read_gateway_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --categories number,container,plant \
  --top-k-heads 8 \
  --output-dir results/gpt5_phase130_true_last_attention_read_gateway \
  --hard-exit-after-model

python tests/gpt5/phase130_true_last_attention_read_gateway_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --categories number,container,plant \
  --top-k-heads 8 \
  --output-dir results/gpt5_phase130_true_last_attention_read_gateway \
  --hard-exit-after-model

python tests/gpt5/phase130_true_last_attention_read_gateway_summary.py

python -m py_compile \
  tests/gpt5/phase130_true_last_attention_read_gateway_cuda.py \
  tests/gpt5/phase130_true_last_attention_read_gateway_summary.py
```

### 生成脚本与结果

- 主脚本：`tests/gpt5/phase130_true_last_attention_read_gateway_cuda.py`
- 汇总脚本：`tests/gpt5/phase130_true_last_attention_read_gateway_summary.py`
- Qwen3 结果：`results/gpt5_phase130_true_last_attention_read_gateway/phase130_qwen3_true_last_attention_read_gateway.json`
- GLM4 结果：`results/gpt5_phase130_true_last_attention_read_gateway/phase130_glm4_true_last_attention_read_gateway.json`
- DS7B 结果：`results/gpt5_phase130_true_last_attention_read_gateway/phase130_deepseek7b_true_last_attention_read_gateway.json`
- 跨模型汇总：`results/gpt5_phase130_true_last_attention_read_gateway/phase130_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
rank = 16
scale = 1.5
top heads/category = 8
```

真实层位：

```text
Qwen3: peak L35, true last L36
GLM4: peak L18, true last L40
DS7B: peak L27, true last L28
```

### 客观结果

#### Qwen3

```text
reference last_input_pre_answer:
  number -0.07
  container +0.07
  plant +0.24

best answer-site local component:
  number last_mlp_output_answer -8.91
  container last_mlp_output_answer -4.54
  plant last_mlp_output_answer -8.35

best top-head ablation:
  number H5 pre_mass 0.408 target Δ -0.06
  container H16 pre_mass 0.228 target Δ -0.02
  plant H2 pre_mass 0.308 target Δ -0.01
```

Qwen3 的 pre-answer reference 很弱，但 answer-site local MLP/block/final-norm 子空间很强。这说明 Qwen3 更像 answer-site readout/local computation，而不是强 pre-answer carry。

#### GLM4 bf16

```text
reference last_input_pre_answer:
  number -0.05
  container +0.02
  plant -0.15

best answer-site local component:
  number last_mlp_output_answer -1.13
  container last_mlp_input_answer -0.26
  plant last_block_output_answer -0.53

best top-head ablation:
  number H26 pre_mass 0.724 target Δ -0.01
  container H27 pre_mass 0.475 target Δ -0.01
  plant H8 pre_mass 0.454 target Δ -0.02
```

GLM4 仍然有 corrected position 下的 old mismatch：

```text
number 32/64
container 62/64
plant 52/64
```

因此 GLM4 的旧结论必须降级；本轮修正后显示 true last answer-site component 有弱到中等信号，但 top single head ablation 仍然弱。

#### DS7B

```text
reference last_input_pre_answer:
  number -2.51
  container -2.66
  plant -2.42

last_attention_output_answer:
  number -5.09
  container -4.83
  plant -4.16

last_mlp_input_answer:
  number -2.78
  container -8.21
  plant -3.63

last_mlp_output_answer:
  number -9.40
  container -11.45
  plant -7.67

last_block_output_answer:
  number -11.98
  container -11.33
  plant -9.62

final_norm_output_answer:
  number -7.82
  container -6.44
  plant -5.63
```

DS7B answer-site local components 非常强，尤其：

```text
last_block_output_answer:
  number -11.98
  container -11.33
  plant -9.62
```

但 top single head ablation 很弱：

```text
number H8 pre_mass 0.615 target Δ -0.25
container H25 pre_mass 0.413 target Δ -0.11
plant H8 pre_mass 0.580 target Δ -0.28
```

### 当前客观事实

1. **DS7B answer-site readout field 极强**

真实最后层的 answer token 上，attention output、MLP input/output、block output、final norm output 都可因果打掉类别 logits。

2. **单个 high-attention-mass head 不是主因**

即使 head 对 pre-answer region 有很高 attention mass：

```text
DS7B H8 pre_mass 0.615 / 0.580
```

单头 ablation 也只产生小 target-down：

```text
number -0.25
plant -0.28
```

这说明最后读取机制不是一个简单“单头搬运器”，更可能是多头分布式读取、attention 后残差合成、或 MLP/answer-site 非线性重编码。

3. **Qwen3 与 DS7B 形成对比**

Qwen3 pre-answer reference 很弱，但 answer-site MLP/output patch 强，说明它更偏答案位置局部读出，而不是 DS7B 式强 pre-answer carry。

### 硬伤

Phase130 使用的是各 answer-site component 自己的 local category basis。因此强 target-down 只能说明：

```text
answer-site component 中存在强类别读出子空间。
```

不能直接证明：

```text
这个子空间就是 pre-answer residual field 的同一方向搬运结果。
```

因此继续做 Phase131 cross-site basis transfer。

## Phase 131: Cross-site Basis Transfer 跨位点同基底转移测试 [2026-06-14 21:31]

### 本阶段目标

Phase130 证明 DS7B answer-site local basis 很强，但还不能证明 true last layer 把 pre-answer field 以同一坐标搬到 answer token。

本阶段使用更严格判据：

```text
用 true-last input pre-answer basis，
直接 patch answer-site components。
```

如果同一 basis 在 answer-site attention / MLP / block / final norm 上仍强 target-down，说明 pre-answer field 坐标保留较多。

如果同一 basis 在 answer site 弱或反向，但 local answer basis 很强，说明最后一层读取后发生了坐标变换。

### 执行命令

```bash
python tests/gpt5/phase131_cross_site_basis_transfer_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --categories number \
  --output-dir results/gpt5_phase131_smoke \
  --hard-exit-after-model

python tests/gpt5/phase131_cross_site_basis_transfer_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase131_cross_site_basis_transfer \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase131_cross_site_basis_transfer_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase131_cross_site_basis_transfer \
  --hard-exit-after-model

python tests/gpt5/phase131_cross_site_basis_transfer_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase131_cross_site_basis_transfer \
  --hard-exit-after-model

python tests/gpt5/phase131_cross_site_basis_transfer_summary.py

python -m py_compile \
  tests/gpt5/phase130_true_last_attention_read_gateway_cuda.py \
  tests/gpt5/phase130_true_last_attention_read_gateway_summary.py \
  tests/gpt5/phase131_cross_site_basis_transfer_cuda.py \
  tests/gpt5/phase131_cross_site_basis_transfer_summary.py
```

### 生成脚本与结果

- 主脚本：`tests/gpt5/phase131_cross_site_basis_transfer_cuda.py`
- 汇总脚本：`tests/gpt5/phase131_cross_site_basis_transfer_summary.py`
- Qwen3 结果：`results/gpt5_phase131_cross_site_basis_transfer/phase131_qwen3_cross_site_basis_transfer.json`
- GLM4 结果：`results/gpt5_phase131_cross_site_basis_transfer/phase131_glm4_cross_site_basis_transfer.json`
- DS7B 结果：`results/gpt5_phase131_cross_site_basis_transfer/phase131_deepseek7b_cross_site_basis_transfer.json`
- 跨模型汇总：`results/gpt5_phase131_cross_site_basis_transfer/phase131_cross_model_summary.md`

### 客观结果

#### Qwen3 same pre-answer basis

```text
reference:
  number -0.07
  container +0.07
  plant +0.24

attention answer:
  number -0.10
  container -0.02
  plant -0.19

best same-basis answer component:
  number last_mlp_input_answer -0.30
  container last_attention_output_answer -0.02
  plant last_mlp_input_answer -0.28
```

Qwen3 没有强 pre-answer basis transfer。

#### GLM4 same pre-answer basis

```text
reference:
  number -0.05
  container +0.02
  plant -0.15

attention answer:
  number -0.10
  container -0.17
  plant -0.08

best same-basis answer component:
  number last_mlp_output_answer -0.94
  container last_mlp_output_answer -0.34
  plant final_norm_output_answer -2.28
```

GLM4 的 same-basis 结果不稳定，且仍有 old position mismatch。不能用它作为主要理论依据。

#### DS7B same pre-answer basis

```text
reference last_input_pre_answer:
  number -2.51
  container -2.66
  plant -2.42

attention answer with same pre-answer basis:
  number +1.26
  container +0.26
  plant +1.44

mlp input answer:
  number -0.43
  container -0.03
  plant +1.72

mlp output answer:
  number +0.14
  container +0.38
  plant -0.24

block output answer:
  number -0.48
  container +0.85
  plant -0.69

final norm answer:
  number +1.59
  container -0.05
  plant +1.06
```

这是本阶段最关键结果：

```text
DS7B pre-answer basis 在 last_input_pre_answer 上强 target-down；
但同一 basis 到 answer-site attention output 后不但不 target-down，反而 target-up。
```

与 Phase130 对比：

```text
Phase130 local answer basis:
  DS7B attention answer number -5.09
  DS7B block output answer number -11.98

Phase131 same pre-answer basis:
  DS7B attention answer number +1.26
  DS7B block output answer number -0.48
```

因此，answer-site 强类别场不是 pre-answer basis 的简单同方向搬运。

### 当前最可靠客观事实

1. **DS7B pre-answer field 与 answer-site field 坐标不同**

同一个 pre-answer basis：

```text
pre-answer site 强 target-down
answer-site attention output 弱或反向
```

说明 true last layer 的读取不是简单复制方向，而是发生了显著 coordinate transform。

2. **DS7B true last answer-site local basis 非常强**

Phase130 已证明：

```text
last_attention_output_answer / last_mlp_output_answer / last_block_output_answer
都有强 target-down。
```

Phase131 则证明：

```text
这些强轴是 answer-site local axes，不是原 pre-answer basis 原样保留。
```

3. **单头 attention mass 不能解释主效应**

高 pre-answer attention mass head 的 ablation 弱，说明最后读取路径可能是：

```text
multi-head distributed read
attention mixing + MLP recoding
residual-to-answer coordinate transform
```

而不是单个 head 的线性搬运。

### 理论进展

Phase129 后的理论是：

```text
R_c^{L-1}(P)
  true-last input pre-answer residual causal field

A_c^L(a)
  true-last output answer-site readout field
```

Phase130/131 后要加入一个关键中间算子：

```text
Φ_L:
  true-last read-and-recode operator
```

更准确公式：

```text
A_c^L(a)
=
Φ_L(
  R_c^{L-1}(P),
  h_{L-1}(a),
  context
)
```

并且：

```text
Basis(A_c^L(a)) ≠ Basis(R_c^{L-1}(P))
```

也就是说，最后一层不是把答案前残差场按同一方向搬到答案位置，而是读取后重编码为 answer-site local readout coordinates。

### 硬伤和瓶颈

1. **还没有直接做 source-token-specific attention value patch**

Phase130 的 head ablation 是按 head output slice 消融，不区分该 head 从哪些 source tokens 读取。下一步需要更细：

```text
只干预 true last attention 中 answer token 从 pre-answer tokens 得到的 value contribution。
```

2. **attention 权重高不等于 value 因果强**

Phase130 已说明 high attention mass head ablation 弱。下一步必须测 value contribution，而不是只看 attention mass。

3. **answer-site local basis 太强，容易掩盖转移路径**

Phase130 的 local answer basis 能强烈影响 logits，但它更像结果态，不是路径解释。

4. **GLM4 仍需 corrected-position 专项复核**

GLM4 不能作为当前理论主支撑，只能作为提示：位置口径修正后它可能有弱中等信号。

### 下一阶段大任务

Phase132 应做：

```text
True Last Source-specific Value Contribution Test
```

核心目标：

```text
在 true last layer 中，
只干预 answer token 从 pre-answer tokens 读取到的 attention value contribution，
而不是消融整个 head。
```

测试要求：

```text
1. 使用 corrected token grid。
2. DS7B 优先，同时保留 Qwen3/GLM4 对照。
3. 对 top attention-mass heads 和全 head aggregate 都测试。
4. 构造 source groups:
   - object_span
   - post_object_pre_answer
   - all_pre_answer
   - self
5. 对 answer token attention output 分解 source contribution：
   contribution(source) = attention_weight(answer, source) * value(source)
6. 移除或替换 source contribution，再测 logits。
```

关键判据：

```text
如果移除 all_pre_answer value contribution 能接近 reference last_input_pre_answer target-down，
则最后读取路径主要是 attention value read。

如果 source-specific value contribution 仍弱，
则要转向 residual/MLP 的 answer-site reconstruction 或多层非局部坐标解释。
```

## Phase 132: True Last Source-specific Value Contribution 真实末层来源值贡献测试 [2026-06-14 22:01]

### 本阶段目标

附件对 Phase130/131 的判断基本正确：

```text
pre-answer residual field 强；
answer-site local readout field 强；
二者不是同一坐标；
true last layer 中存在 read-and-recode operator。
```

Phase130 的硬伤是 head ablation 过粗：消融整个 head，不能区分该 head 从哪个 source token 读取。Phase131 的硬伤是 same-basis transfer 只能判断坐标是否保留，不能直接测 value path。

本阶段直接测试：

```text
answer token 在 true last attention 中
从不同 source group 读取的 value contribution
是否具有因果作用。
```

### 执行命令

```bash
python tests/gpt5/phase132_source_value_contribution_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --categories number \
  --top-k-heads 2 \
  --output-dir results/gpt5_phase132_smoke \
  --hard-exit-after-model

python tests/gpt5/phase132_source_value_contribution_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 16 \
  --categories number,container,plant \
  --top-k-heads 8 \
  --output-dir results/gpt5_phase132_source_value_contribution \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase132_source_value_contribution_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 16 \
  --categories number,container,plant \
  --top-k-heads 8 \
  --output-dir results/gpt5_phase132_source_value_contribution \
  --hard-exit-after-model

python tests/gpt5/phase132_source_value_contribution_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 16 \
  --categories number,container,plant \
  --top-k-heads 8 \
  --output-dir results/gpt5_phase132_source_value_contribution \
  --hard-exit-after-model

python tests/gpt5/phase132_source_value_contribution_summary.py

python -m py_compile \
  tests/gpt5/phase132_source_value_contribution_cuda.py \
  tests/gpt5/phase132_source_value_contribution_summary.py
```

### 生成脚本与结果

- 主脚本：`tests/gpt5/phase132_source_value_contribution_cuda.py`
- 汇总脚本：`tests/gpt5/phase132_source_value_contribution_summary.py`
- Qwen3 结果：`results/gpt5_phase132_source_value_contribution/phase132_qwen3_source_value_contribution.json`
- GLM4 结果：`results/gpt5_phase132_source_value_contribution/phase132_glm4_source_value_contribution.json`
- DS7B 结果：`results/gpt5_phase132_source_value_contribution/phase132_deepseek7b_source_value_contribution.json`
- 跨模型汇总：`results/gpt5_phase132_source_value_contribution/phase132_cross_model_summary.md`

### 测试原理

对 true last layer attention 做两次 forward：

```text
1. 第一次 forward:
   capture true last attention weights
   capture v_proj output values

2. 计算 answer token 从 source group 得到的 pre-o_proj contribution:
   contribution(source)
   =
   attention_weight(answer, source) * value(source)

3. 第二次 forward:
   在 o_proj input 中，只从 answer token 的指定 head slice 减去该 source contribution。

4. 测 final logits 和 answer projection。
```

source groups：

```text
object_span
post_object_pre_answer
all_pre_answer
self
```

head modes：

```text
all_heads
top_heads
```

其中 top_heads 是按 answer token 对 pre-answer tokens 的 attention mass 选出的 top 8 heads。

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
batch size = 16
reference scale = 1.5
contribution scale = 1.0
top heads = 8
```

真实层位与 head 结构：

```text
Qwen3:
  peak L35, true last L36
  heads 32, kv_heads 8

GLM4:
  peak L18, true last L40
  heads 32, kv_heads 2

DS7B:
  peak L27, true last L28
  heads 28, kv_heads 4
```

### 客观结果

#### Qwen3

```text
reference last_input_pre_answer:
  number -0.07
  container +0.07
  plant +0.24

all_pre_answer all_heads:
  number +0.24
  container +0.13
  plant +0.30

post_object_pre_answer all_heads:
  number +0.03
  container +0.08
  plant +0.23

object_span all_heads:
  number +0.02
  container +0.02
  plant +0.04

self all_heads:
  number +0.86
  container +0.46
  plant +0.14
```

Qwen3 没有 DS7B 式 pre-answer value contribution target-down；相反 all_pre/self 多为 target-up 或弱效应。

#### GLM4 bf16

```text
position mismatch:
  number 32/64
  container 62/64
  plant 52/64

reference last_input_pre_answer:
  number -0.05
  container +0.02
  plant -0.14

all_pre_answer all_heads:
  number -0.05
  container +0.21
  plant +0.22
```

GLM4 在 corrected position 下仍没有稳定强 source-value path。由于 left padding mismatch 历史问题，GLM4 暂不作为主要理论依据。

#### DS7B

```text
reference last_input_pre_answer:
  number -2.36
  container -2.44
  plant -2.17

all_pre_answer all_heads:
  number -1.86
  container -2.65
  plant -2.01

post_object_pre_answer all_heads:
  number -0.34
  container -0.21
  plant -0.50

object_span all_heads:
  number -0.07
  container -0.02
  plant -0.05

self all_heads:
  number -0.36
  container -0.10
  plant -0.10

all_pre_answer top_heads:
  number -0.27
  container -0.10
  plant -0.38
```

这是本阶段最关键结果：

```text
DS7B all_pre_answer all_heads value contribution removal
接近复现 reference last_input_pre_answer target-down。
```

对照：

```text
number:
  reference -2.36
  all_pre all_heads -1.86

container:
  reference -2.44
  all_pre all_heads -2.65

plant:
  reference -2.17
  all_pre all_heads -2.01
```

而更窄的 source group 明显弱：

```text
object_span all_heads:
  around 0

post_object_pre_answer all_heads:
  weak -0.21 to -0.50

self all_heads:
  weak
```

top heads aggregate 也弱：

```text
all_pre_answer top_heads:
  number -0.27
  container -0.10
  plant -0.38
```

### 当前最可靠客观事实

1. **DS7B true last layer 的关键读取路径是 all_pre_answer value contribution 的全头聚合**

不是 object token 单独贡献，也不是 post-object tokens 单独贡献，也不是 self token 单独贡献。

```text
all_pre_answer all_heads ≈ reference last_input_pre_answer
```

2. **top attention-mass heads 不能解释主效应**

Phase130 的单头 ablation 弱，Phase132 的 top_heads aggregate 也弱：

```text
DS7B all_pre top_heads:
  number -0.27
  container -0.10
  plant -0.38
```

说明强路径不是少数 high-mass heads，而是全头分布式 value aggregation。

3. **object_span 不是主读取来源**

```text
DS7B object_span all_heads:
  number -0.07
  container -0.02
  plant -0.05
```

类别因果信号不是只在 object token 上被最后一层读取，而是分布在 answer 前整个上下文区域。

4. **post_object_pre_answer 有弱贡献但远小于 all_pre_answer**

```text
post_object_pre_answer:
  number -0.34
  container -0.21
  plant -0.50
```

说明对象后的模板词有贡献，但不能独立解释主效应。

### 理论进展

Phase131 的公式是：

```text
A_c^L(a)
=
Φ_L(
  R_c^{L-1}(P),
  h_{L-1}(a),
  context
)
```

Phase132 进一步把 Φ_L 的关键项实证化：

```text
Φ_L 主要包含 true-last attention 的 all-pre-token value aggregation。
```

更具体：

```text
A_c^L(a)
≈
Recode_L(
  Σ_{h∈all heads}
  Σ_{s∈all pre-answer}
  α_h(a,s) V_h(s)
)
```

其中：

```text
α_h(a,s):
  true last layer 中 answer token 对 source token s 的 attention weight

V_h(s):
  source token s 在 head h 的 value vector

Recode_L:
  o_proj + residual + MLP + norm 形成的 answer-site 重编码过程
```

这说明语言编码机制中的类别约束不是单点向量，也不是单头搬运，而是：

```text
distributed pre-answer value field
通过 true last attention 全头聚合
进入 answer-site readout/recode state。
```

### 硬伤和瓶颈

1. **all_pre_answer 是宽 source group**

它证明了全答案前区域整体重要，但还没有分解出更细的 source 组合：

```text
pre_object tokens
object tokens
post_object tokens
template structural tokens
special tokens
```

2. **移除 contribution 是基于第一次 forward 的 attention/value**

第二次 forward 中前层状态可能因 hook 变化而略有差异。不过 hook 只发生在 true last o_proj input，前面的 attention/value 计算应保持一致，因此这个近似较强。

3. **没有逐 head contribution 组合搜索**

top attention-mass heads 弱，但不排除因果强 head 不等于 attention-mass top head。下一步要按 contribution ablation effect 直接选 head。

4. **还没有 generation audit**

当前仍是 DCF logits。关键路径已经接近闭合，后续需要验证生成行为是否同步改变。

### 对附件判断的修正与确认

附件判断中正确部分：

```text
1. Phase130/131 正确说明 pre-answer field 和 answer-site field 坐标不同。
2. true last read-and-recode operator 是当前核心对象。
3. 单头搬运假说不可靠。
4. 下一步应测 source-specific value contribution。
```

Phase132 的新增修正：

```text
1. read-and-recode operator 的关键输入不是 object token 单独贡献，
   而是 all_pre_answer value contribution 的全头聚合。

2. top attention-mass heads 仍不能解释主效应，
   因此 attention mass 不能作为 causal head selection 的主标准。

3. DS7B 的类别约束场更像分布式上下文场，
   而不是对象词位置上的静态概念向量。
```

### 下一阶段大任务

Phase133 应做：

```text
True Last Value Contribution Head Effect Ranking
```

核心目标：

```text
不用 attention mass 选 head，
而是直接按 source-value contribution removal 的 target_delta 选 head。
```

测试方案：

```text
1. 固定 DS7B number/container/plant。
2. 对 true last layer 每个 head 单独移除 all_pre_answer value contribution。
3. 排名每个 head 的 target_delta、release_delta、answer_proj_delta。
4. 再测试 top causal heads aggregate：
   top1, top2, top4, top8, all_heads。
5. 对 Qwen3/GLM4 做同脚本对照，但 DS7B 是主对象。
```

关键判据：

```text
如果少数 causal heads aggregate 接近 all_heads，
说明是稀疏 head set，只是不能用 attention mass 找到。

如果 top causal heads 仍需很多 head 才接近 all_heads，
说明 true last read path 是真正分布式多头聚合。
```

## Phase 133: True Last Value Contribution Head Effect Ranking 真实末层值贡献头效应排名 [2026-06-14 22:27]

### 本阶段目标

附件对 Phase132 的判断基本正确：

```text
DS7B 的 true-last read path 不是 object_span 单点，
也不是 attention-mass top heads，
而是 all_pre_answer value contribution 的多头聚合。
```

Phase132 仍留下一个问题：

```text
all_heads 强，top attention-mass heads 弱。
但是否存在 top causal heads？
```

本阶段不再按 attention mass 选 head，而是：

```text
对 true last layer 每个 head 单独移除 all_pre_answer value contribution，
按 target_delta 排名，
再测试 top1/top2/top4/top8/all_heads 聚合。
```

### 执行命令

```bash
python tests/gpt5/phase133_value_head_effect_ranking_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --categories number \
  --output-dir results/gpt5_phase133_smoke \
  --hard-exit-after-model

python tests/gpt5/phase133_value_head_effect_ranking_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 16 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase133_value_head_effect_ranking \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase133_value_head_effect_ranking_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 16 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase133_value_head_effect_ranking \
  --hard-exit-after-model

python tests/gpt5/phase133_value_head_effect_ranking_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 16 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase133_value_head_effect_ranking \
  --hard-exit-after-model

python tests/gpt5/phase133_value_head_effect_ranking_summary.py

python -m py_compile \
  tests/gpt5/phase133_value_head_effect_ranking_cuda.py \
  tests/gpt5/phase133_value_head_effect_ranking_summary.py
```

### 生成脚本与结果

- 主脚本：`tests/gpt5/phase133_value_head_effect_ranking_cuda.py`
- 汇总脚本：`tests/gpt5/phase133_value_head_effect_ranking_summary.py`
- Qwen3 结果：`results/gpt5_phase133_value_head_effect_ranking/phase133_qwen3_value_head_effect_ranking.json`
- GLM4 结果：`results/gpt5_phase133_value_head_effect_ranking/phase133_glm4_value_head_effect_ranking.json`
- DS7B 结果：`results/gpt5_phase133_value_head_effect_ranking/phase133_deepseek7b_value_head_effect_ranking.json`
- 跨模型汇总：`results/gpt5_phase133_value_head_effect_ranking/phase133_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
batch size = 16
source group = all_pre_answer
reference scale = 1.5
contribution scale = 1.0
```

真实层位：

```text
Qwen3: peak L35, true last L36, heads 32, kv_heads 8
GLM4: peak L18, true last L40, heads 32, kv_heads 2
DS7B: peak L27, true last L28, heads 28, kv_heads 4
```

### 客观结果

#### Qwen3

```text
reference:
  number -0.07
  container +0.07
  plant +0.24

best single causal head:
  number H11 -0.08
  container H11 -0.07
  plant H11 -0.09

top causal aggregate:
  number top8 -0.25, all_heads +0.24
  container top8 -0.19, all_heads +0.13
  plant top8 -0.26, all_heads +0.30
```

Qwen3 的 causal-selected heads 与 all_heads 方向相反，说明它不是 DS7B 式 all_pre value read path。

#### GLM4 bf16

```text
reference:
  number -0.05
  container +0.02
  plant -0.14

best single causal head:
  number H1 -0.03
  container H0 -0.05
  plant H18 -0.03

top causal aggregate:
  number top8 -0.11, all_heads -0.05
  container top8 -0.19, all_heads +0.21
  plant top8 -0.15, all_heads +0.22
```

GLM4 仍受 left padding old-mismatch 历史问题影响，不作为主要理论依据。

#### DS7B

```text
reference:
  number -2.36
  container -2.44
  plant -2.17

best single causal head:
  number H13 -0.28
  container H13 -0.41
  plant H13 -0.29
```

H13 在三个类别中都是 best single head，这是一个非常强的稳定信号。

DS7B top causal aggregate：

```text
number:
  top1 -0.28
  top2 -0.91
  top4 -1.41
  top8 -2.16
  all_heads -1.86

container:
  top1 -0.41
  top2 -0.74
  top4 -1.90
  top8 -2.31
  all_heads -2.65

plant:
  top1 -0.29
  top2 -0.53
  top4 -1.15
  top8 -1.52
  all_heads -2.01
```

DS7B top causal head ids：

```text
number top8:
  H13, H12, H8, H11, H7, H25, H10, H21

container top8:
  H13, H10, H12, H11, H25, H26, H8, H24

plant top8:
  H13, H8, H12, H11, H26, H24, H25, H23
```

跨类别稳定核心：

```text
H13, H12, H11, H8, H25
```

### 当前最可靠客观事实

1. **attention mass 不是因果 head 排名标准**

Phase132 的 attention-mass top heads 很弱；Phase133 的 causal-ranked heads 明显更强。

2. **DS7B 存在稳定 causal head set**

不是单头：

```text
best single head 只有 -0.28 到 -0.41。
```

但也不是完全均匀 all-head 分布：

```text
top4/top8 已经接近 all_heads。
```

尤其：

```text
number top8 -2.16, all_heads -1.86
container top8 -2.31, all_heads -2.65
plant top8 -1.52, all_heads -2.01
```

3. **H13 是跨类别最稳定的入口 head**

三个类别 best single head 都是 H13：

```text
number H13
container H13
plant H13
```

这说明 true last read path 中可能存在一个稳定 gate-like head，但它单独不够强，必须与其他 heads 聚合。

4. **true last read path 是中等稀疏的多头聚合**

Phase132 的“全头分布式”需要修正为：

```text
不是少数 1-2 个 head；
也不是 28 个 head 完全均匀；
而是约 4-8 个 causal heads 形成主聚合通道。
```

### 理论进展

Phase132 公式：

```text
A_c^L(a)
≈
Recode_L(
  Σ_{h∈all heads}
  Σ_{s∈all pre-answer}
  α_h(a,s) V_h(s)
)
```

Phase133 修正为：

```text
A_c^L(a)
≈
Recode_L(
  Σ_{h∈H_causal}
  Σ_{s∈all pre-answer}
  α_h(a,s) V_h(s)
  +
  background
)
```

其中：

```text
H_causal:
  true-last all_pre_answer value contribution 的因果头集合

DS7B H_causal 约包含:
  H13, H12, H11, H8, H25, H10/H26/H24...
```

这比 Phase132 更精确：

```text
关键机制不是 attention-mass top heads，
而是 causal value-contribution heads。
```

### 对附件判断的确认与修正

附件正确部分：

```text
1. Phase132 正确定位 all_pre_answer value aggregation。
2. object token 单点模型被否定。
3. high-attention-head 模型被否定。
4. 概念约束更像分布式上下文场。
```

Phase133 新增修正：

```text
1. all_heads 强不等于完全均匀分布。
2. 按真实 value contribution 因果效应排名后，存在稳定 causal head set。
3. H13 是 DS7B true last read path 的跨类别核心候选头。
4. top causal 4/8 接近 all_heads，因此下一步要分析 causal head set 的 source composition 和 value geometry。
```

### 硬伤和瓶颈

1. **仍只测试 all_pre_answer，不知道 causal heads 具体读哪些 token**

Phase133 确定了 head set，但没分解每个 causal head 的 source token 结构。

2. **没有测试 head interaction**

top-k 聚合不等于简单相加，可能存在 head 之间的协同或抵消。

3. **没有扩大数据量复验 H13**

H13 跨三类稳定，但仍需更大 train/test 和更多类别验证。

4. **没有 generation audit**

当前还是 DCF logits。

### 下一阶段大任务

Phase134 应做：

```text
Causal Head Source Composition and Expansion Audit
```

核心目标：

```text
验证 H13/H12/H11/H8/H25 等 causal heads 到底读取 all_pre_answer 中哪些细分 token，
并扩大类别与样本确认稳定性。
```

测试方案：

```text
1. 重点 DS7B，保留 Qwen3/GLM4 对照。
2. source groups 细分：
   - special_prefix
   - pre_object
   - object_span
   - object_to_template_bridge
   - post_object_structural_tokens
   - answer_prompt_tail
   - all_pre_answer
3. 对 causal heads:
   H13,H12,H11,H8,H25,H10,H26,H24
   分别做 source-specific value removal。
4. 扩大类别：
   number, container, plant, time, clothing, furniture
5. DS7B 关键结果加大到 train 12 / test 24。
```

关键判据：

```text
如果 causal heads 的强效应集中在某些结构 token，
说明语言编码依赖模板/关系结构接口。

如果强效应分散在多个 source group，
说明真正的类别约束是更宽的上下文场。
```


## Phase 134: Causal Head Source Composition and Expansion Audit 因果头来源构成扩展审计 [2026-06-14 22:49]

### 本阶段目标

附件对 Phase133 的判断基本正确：

```text
Phase133 把 Phase132 的 all-head distributed read
修正为 medium-sparse causal head ensemble。
```

DS7B true last read path 的稳定候选头：

```text
H13, H12, H11, H8, H25
```

以及类别补充头：

```text
H10, H26, H24, H23, H21
```

本阶段目标：

```text
1. 固定 causal head set。
2. 将 all_pre_answer 拆成更细 source groups。
3. 扩大类别到 6 类。
4. DS7B 使用更大样本 train12/test24 复核。
```

### 执行命令

```bash
python tests/gpt5/phase134_causal_head_source_composition_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --categories number \
  --output-dir results/gpt5_phase134_smoke \
  --hard-exit-after-model

python tests/gpt5/phase134_causal_head_source_composition_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --categories number,container,plant,time,clothing,furniture \
  --output-dir results/gpt5_phase134_causal_head_source_composition \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase134_causal_head_source_composition_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --categories number,container,plant,time,clothing,furniture \
  --output-dir results/gpt5_phase134_causal_head_source_composition \
  --hard-exit-after-model

python tests/gpt5/phase134_causal_head_source_composition_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 24 \
  --batch-size 24 \
  --categories number,container,plant,time,clothing,furniture \
  --output-dir results/gpt5_phase134_causal_head_source_composition \
  --hard-exit-after-model

python tests/gpt5/phase134_causal_head_source_composition_summary.py

python -m py_compile \
  tests/gpt5/phase134_causal_head_source_composition_cuda.py \
  tests/gpt5/phase134_causal_head_source_composition_summary.py
```

### 生成脚本与结果

- 主脚本：`tests/gpt5/phase134_causal_head_source_composition_cuda.py`
- 汇总脚本：`tests/gpt5/phase134_causal_head_source_composition_summary.py`
- Qwen3 结果：`results/gpt5_phase134_causal_head_source_composition/phase134_qwen3_causal_head_source_composition.json`
- GLM4 结果：`results/gpt5_phase134_causal_head_source_composition/phase134_glm4_causal_head_source_composition.json`
- DS7B 结果：`results/gpt5_phase134_causal_head_source_composition/phase134_deepseek7b_causal_head_source_composition.json`
- 跨模型汇总：`results/gpt5_phase134_causal_head_source_composition/phase134_cross_model_summary.md`

### 测试范围

```text
categories = number, container, plant, time, clothing, furniture

Qwen3:
  train/test = 8/16
  causal heads = H11,H10,H28,H3,H31,H2,H5,H20

GLM4:
  train/test = 8/16
  causal heads = H1,H28,H0,H18,H11,H27,H23,H4

DS7B:
  train/test = 12/24
  causal heads = H13,H12,H11,H8,H25,H10,H26,H24
```

source groups：

```text
special_prefix
pre_object
object_span
object_to_template_bridge
post_object_structural
answer_prompt_tail
all_pre_answer
```

### Source audit

DS7B 的 source group 平均长度：

```text
special_prefix: 0.0
pre_object: 2.0
object_span: 1.0-1.5
object_to_template_bridge: 2.0
post_object_structural: 0.0
answer_prompt_tail: 2.0
all_pre_answer: 6.25-6.75
```

这个 audit 很重要：当前模板太短，`post_object_structural` 被 bridge/tail 切分吃完，全部为空。因此本阶段不能对 post_object_structural 下结论，只能说明当前短模板下该组不可用。

### 客观结果

#### DS7B expanded 6 categories

```text
number:
  reference -2.56
  pre_object -1.88
  object_span -0.04
  bridge -0.17
  tail -0.19
  all_pre -1.97

container:
  reference -2.85
  pre_object -2.10
  object_span -0.07
  bridge -0.12
  tail -0.15
  all_pre -2.30

plant:
  reference -2.25
  pre_object -1.29
  object_span -0.07
  bridge -0.26
  tail -0.30
  all_pre -1.77

time:
  reference -2.62
  pre_object -1.76
  object_span -0.03
  bridge -0.21
  tail -0.23
  all_pre -2.02

clothing:
  reference +1.87
  pre_object -0.90
  object_span +0.06
  bridge -0.02
  tail +0.13
  all_pre -0.55

furniture:
  reference +1.49
  pre_object -0.77
  object_span -0.02
  bridge -0.04
  tail +0.13
  all_pre -0.59
```

#### Qwen3

Qwen3 causal heads 对照显示弱但稳定的 pre_object/all_pre 效应：

```text
number:
  pre_object -0.15
  all_pre -0.25

time:
  pre_object -0.19
  all_pre -0.29

clothing:
  pre_object -0.22
  all_pre -0.31
```

但 Qwen3 的 reference last_input_pre_answer 多数很弱或 target-up，因此不能解释为 DS7B 式 pre-answer carry。

#### GLM4

GLM4 corrected-position 下仍然弱，且 old mismatch 大：

```text
number old_mismatch 32
container old_mismatch 62
plant old_mismatch 52
clothing old_mismatch 60
furniture old_mismatch 52
```

GLM4 结果仍不作为主理论依据。

### 当前最可靠客观事实

1. **DS7B 的 object_span 不是主要 source**

6 类中 object_span 都接近 0：

```text
number -0.04
container -0.07
plant -0.07
time -0.03
clothing +0.06
furniture -0.02
```

这进一步确认：

```text
true last read path 不是从对象词元单点读取类别向量。
```

2. **DS7B 的主要可分来源落在 pre_object / all_pre**

对于强 target-down 类：

```text
number:
  pre_object -1.88, all_pre -1.97

container:
  pre_object -2.10, all_pre -2.30

plant:
  pre_object -1.29, all_pre -1.77

time:
  pre_object -1.76, all_pre -2.02
```

pre_object 在当前模板中主要是：

```text
The / A / The word / People use the word
```

也就是模板前缀和任务结构位置，而不是对象本体。

3. **clothing/furniture 与 number/container/plant/time 不同**

clothing/furniture 的 reference 是 target-up：

```text
clothing reference +1.87
furniture reference +1.49
```

但 pre_object 和 all_pre 移除仍产生 target-down：

```text
clothing pre_object -0.90, all_pre -0.55
furniture pre_object -0.77, all_pre -0.59
```

这说明这类类别的 pre-answer reference basis 与 causal head source contribution 关系更复杂，可能是 suppressor/interface mixed category。

4. **all_pre 强于任一窄 source，但 pre_object 贡献占比很大**

对于 DS7B 强类别：

```text
pre_object ≈ all_pre 的主要部分
```

但 all_pre 仍通常更强，说明多个 source 共同作用。

### 理论进展

Phase133 的理论：

```text
A_c^L(a)
≈
Recode_L(
  Σ_{h∈H_causal}
  Σ_{s∈all_pre_answer}
  α_h(a,s) V_h(s)
)
```

Phase134 进一步显示，在当前模板下：

```text
Σ_{s∈all_pre_answer}
主要可分为：
  pre_object/template-prefix contribution
  + weak bridge/tail contribution
  + near-zero object_span contribution
```

更谨慎的写法：

```text
ReadInput_c
≈
Σ_{h∈H_causal}
[
  Σ_{s∈template-prefix} α_h(a,s)V_h(s)
  +
  small Σ_{s∈post-object interface} α_h(a,s)V_h(s)
  +
  near-zero Σ_{s∈object-span} α_h(a,s)V_h(s)
]
```

这提示一个非常重要的方向：

```text
类别约束不是被最后一层直接从 object token 读出；
而是对象信息先被上游层扩散/编码到模板结构与上下文场中，
最后一层读取的是上下文结构化 value field。
```

### 对附件判断的确认与修正

附件正确部分：

```text
1. Phase133 正确把 all-head aggregation 修正为 causal head ensemble。
2. H13 是稳定候选入口头，但不是唯一头。
3. shared causal read core + category-specific supplement 是合理判断。
4. 下一步应分析 causal heads 的 source composition。
```

Phase134 新增修正：

```text
1. object_span 在 DS7B causal heads 中几乎无效。
2. 当前短模板下，主要可分来源是 pre_object/template-prefix 与 all_pre 聚合。
3. post_object_structural 为空，不能作为结论。
4. clothing/furniture 不是干净 target-down 类别，仍属于 mixed/suppressor/interface 类。
```

### 硬伤和瓶颈

1. **source split 受短模板限制**

当前模板很短：

```text
The {obj} is a kind of
A {obj} belongs to the category of
The word {obj} refers to a type of
People use the word {obj} when talking about
```

post_object 区域通常只有 2-4 个 token，因此 bridge/tail 切分后 structural 为空。

2. **pre_object 不是纯语义 source**

pre_object 主要是模板前缀/任务结构 token，强效应可能表示：

```text
answer token 读取任务框架和对象条件的组合坐标，
而不是读取“概念词”本身。
```

3. **还没有测试长模板和自然句**

必须用更长、更自然的模板重新切分 source groups，才能判断 structural tokens 是否真强。

4. **还没有生成审计**

路径已经较清楚，但仍需 generation audit。

### 下一阶段大任务

Phase135 应做：

```text
Long-template Source Field Decomposition
```

核心目标：

```text
用更长模板创造可分辨的 source regions，
重新测试 DS7B causal head set 的 source composition。
```

测试方案：

```text
1. 新增 6-8 个长模板，每个模板明确包含：
   - subject prefix
   - object mention
   - relation phrase
   - reasoning bridge phrase
   - answer prompt tail

2. source groups 改为：
   - prefix
   - object_span
   - relation_phrase
   - reasoning_bridge
   - answer_tail
   - all_pre_answer

3. 固定 DS7B causal heads:
   H13,H12,H11,H8,H25,H10,H26,H24

4. 类别：
   number, container, plant, time
   优先干净 target-down 类。

5. 数据：
   train12/test24 起步；
   如果结果稳定，再扩到 train16/test32。
```

关键判据：

```text
如果 prefix 仍主导，说明最后读取强依赖任务框架 token。

如果 relation/reasoning bridge 变强，说明短模板下 pre_object 强是模板切分伪影。

如果 object_span 仍弱，进一步否定对象词单点概念向量模型。
```


## Phase 135: Long-template Source Field Decomposition 长模板来源场分解 [2026-06-14 23:02]

### 本阶段目标

附件对 Phase134 的判断基本正确：

```text
Phase134 证明 object_span 不是主要来源，
但短模板导致 source split 不充分；
pre_object 强不能直接解释为最终理论。
```

Phase135 用更长模板重新分解 source field，目标是让下面区域真正非空且可分：

```text
prefix
object_span
relation_phrase
reasoning_bridge
answer_tail
all_pre_answer
```

### 执行命令

```bash
python tests/gpt5/phase135_long_template_source_field_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --categories number \
  --output-dir results/gpt5_phase135_smoke \
  --hard-exit-after-model

python tests/gpt5/phase135_long_template_source_field_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 16 \
  --categories number,container,plant,time \
  --output-dir results/gpt5_phase135_long_template_source_field \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase135_long_template_source_field_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 16 \
  --categories number,container,plant,time \
  --output-dir results/gpt5_phase135_long_template_source_field \
  --hard-exit-after-model

python tests/gpt5/phase135_long_template_source_field_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 24 \
  --batch-size 16 \
  --categories number,container,plant,time \
  --output-dir results/gpt5_phase135_long_template_source_field \
  --hard-exit-after-model

python tests/gpt5/phase135_long_template_source_field_summary.py

python -m py_compile \
  tests/gpt5/phase135_long_template_source_field_cuda.py \
  tests/gpt5/phase135_long_template_source_field_summary.py
```

### 生成脚本与结果

- 主脚本：`tests/gpt5/phase135_long_template_source_field_cuda.py`
- 汇总脚本：`tests/gpt5/phase135_long_template_source_field_summary.py`
- Qwen3 结果：`results/gpt5_phase135_long_template_source_field/phase135_qwen3_long_template_source_field.json`
- GLM4 结果：`results/gpt5_phase135_long_template_source_field/phase135_glm4_long_template_source_field.json`
- DS7B 结果：`results/gpt5_phase135_long_template_source_field/phase135_deepseek7b_long_template_source_field.json`
- 跨模型汇总：`results/gpt5_phase135_long_template_source_field/phase135_cross_model_summary.md`

### 长模板设置

使用 6 个长模板，每个模板包含：

```text
prefix
object mention
relation phrase
reasoning bridge
answer tail
```

例如：

```text
In this classification task, the item {obj}
should be interpreted by its ordinary meaning and everyday use,
so the broad semantic group that best fits this item
is
```

注意：Phase135 的 centers 也全部用长模板重新捕获，没有混用短模板 basis。

### Source audit

DS7B source group 平均长度：

```text
prefix: about 5.8
object_span: 1.0-1.5
relation_phrase: about 9.3
reasoning_bridge: about 9.7
answer_tail: 3.0
all_pre_answer: 28.8-29.3
```

这说明 Phase135 的 source split 有效，修复了 Phase134 中 `post_object_structural = 0` 的问题。

### 客观结果

#### DS7B

```text
number:
  reference -2.66
  prefix -0.45
  object_span +0.00
  relation_phrase -0.02
  reasoning_bridge -0.10
  answer_tail -0.02
  all_pre_answer -0.64

container:
  reference -2.67
  prefix -0.46
  object_span +0.00
  relation_phrase -0.11
  reasoning_bridge -0.21
  answer_tail -0.10
  all_pre_answer -1.05

plant:
  reference -2.70
  prefix -0.52
  object_span -0.00
  relation_phrase -0.12
  reasoning_bridge -0.27
  answer_tail -0.09
  all_pre_answer -1.08

time:
  reference -3.13
  prefix -0.49
  object_span -0.00
  relation_phrase -0.04
  reasoning_bridge -0.13
  answer_tail -0.05
  all_pre_answer -0.82
```

#### Qwen3

Qwen3 的 long-template reference 有时很强，但 causal head source effect 很弱：

```text
number:
  reference -2.80
  prefix -0.16
  all_pre -0.12

container:
  reference -0.70
  prefix -0.17
  all_pre -0.09

plant:
  reference -0.41
  prefix -0.23
  all_pre -0.15

time:
  reference -1.90
  prefix -0.18
  all_pre -0.14
```

说明 Qwen3 的固定 causal-head set 不解释 long-template reference，机制不同于 DS7B。

#### GLM4 bf16

GLM4 继续受 left padding old mismatch 影响：

```text
number old_mismatch 80
container old_mismatch 95
plant old_mismatch 90
time old_mismatch 80
```

修正定位下 source effects 仍弱，不作为主理论依据。

### 当前最可靠客观事实

1. **object_span 再次被否定为主要 source**

长模板下 DS7B：

```text
object_span:
  number +0.00
  container +0.00
  plant -0.00
  time -0.00
```

这比 Phase134 更强，因为现在模板更长、source split 更充分。

2. **prefix 是最强窄 source，但不能单独解释 reference**

```text
prefix:
  number -0.45
  container -0.46
  plant -0.52
  time -0.49

reference:
  number -2.66
  container -2.67
  plant -2.70
  time -3.13
```

prefix 是稳定来源，但只解释一小部分。

3. **all_pre_answer 仍是最强 source group**

```text
all_pre:
  number -0.64
  container -1.05
  plant -1.08
  time -0.82
```

all_pre 明显强于任一单独窄 source，说明类别约束仍是宽上下文场。

4. **relation/bridge/tail 有小贡献，但不是主通路**

```text
relation_phrase:
  around -0.02 to -0.12

reasoning_bridge:
  around -0.10 to -0.27

answer_tail:
  around -0.02 to -0.10
```

长模板下 reasoning_bridge 对 plant/container 有更明显弱贡献，但远小于 all_pre。

### 对 Phase134 附件判断的确认与修正

确认：

```text
1. Phase134 的 object_span 弱不是短模板偶然；
   长模板下 object_span 仍接近 0。

2. pre_object/prefix 强确实存在；
   不是完全切分伪影。

3. 但 prefix 不是全部；
   all_pre_answer 仍然最强。
```

修正：

```text
1. Phase134 中 pre_object 接近 all_pre 的现象部分来自短模板。
   长模板下 prefix 只解释一部分，all_pre 明显更强。

2. relation/bridge/tail 不是空组后，确实有小贡献；
   但不是主贡献。

3. DS7B true-last read path 更像：
   stable prefix/task-frame read + broad all-pre context aggregation。
```

### 理论进展

Phase135 后，当前公式应写成：

```text
A_c^L(a)
≈
Recode_L(
  Σ_{h∈H_causal}
  [
    Σ_{s∈prefix} α_h(a,s)V_h(s)
    +
    Σ_{s∈relation/bridge/tail} α_h(a,s)V_h(s)
    +
    Σ_{s∈other pre-answer context} α_h(a,s)V_h(s)
  ]
)
```

并且：

```text
Σ_{s∈object_span} α_h(a,s)V_h(s)
≈ 0
```

也就是说：

```text
对象词本身不是最终读取源；
对象意义已经在上游传播到上下文场中；
真实末层 causal heads 读取的是任务框架化的上下文 value field。
```

### 硬伤和瓶颈

1. **long-template reference 比 source removal 大很多**

DS7B reference 约 -2.6 到 -3.1，但 all_pre causal-head source removal 只有 -0.64 到 -1.08。

这说明：

```text
固定 Phase133 causal heads 不能解释全部 reference；
长模板下可能需要重新做 head ranking。
```

2. **Qwen3 reference 强但 source effect 弱**

说明 Qwen3 的 long-template pre-answer reference 可能走不同路径，不能套用 DS7B head set。

3. **source split 仍是 token-position 近似**

虽然比 Phase134 好，但 relation/bridge/tail 是按 object 后 token 区间切分，不是严格语义边界。

4. **还没有 generation audit**

路径已经更清楚，但仍需验证真实输出生成。

### 下一阶段大任务

Phase136 应做：

```text
Long-template Head Re-ranking and Path Closure
```

核心目标：

```text
在 long-template 条件下重新逐 head 排名，
判断 Phase133 的 short-template causal head set 是否迁移，
以及是否存在新的 long-template causal heads。
```

测试方案：

```text
1. DS7B 优先，类别 number/container/plant/time。
2. 使用 Phase135 长模板。
3. 对 true last layer 每个 head 单独移除 all_pre_answer value contribution。
4. 排名 top heads。
5. 比较：
   - Phase133 short-template H13,H12,H11,H8,H25
   - Phase136 long-template top heads
6. 测 top1/top2/top4/top8/all_heads。
7. 若 top heads 稳定，再做 generation audit。
```

关键判据：

```text
如果 H13/H12/H11/H8/H25 仍是 long-template top heads，
说明 causal head set 是模板稳健读取通道。

如果 head set 大幅改变，
说明 true-last read gateway 是 template-conditioned，
需要研究任务框架如何选择 head ensemble。
```


## Phase 136: Long-template Head Re-ranking 长模板因果头重排 [2026-06-14 23:28]

### 本阶段目标

根据用户要求，先分析 Phase135 附件判断是否正确，再继续完成客观测试。

附件中正确部分：

```text
1. Phase135 基本正确，确实排除了 Phase134 中短模板 pre_object 切分过短带来的偏置。
2. object_span 不是 true-last 主要 source，这一点在长模板下更可靠。
3. prefix 仍有稳定贡献，但不能解释全部 reference。
4. Phase135 的最大硬伤是固定 Phase133 causal heads 不能充分解释 long-template reference。
5. 下一步应在 long-template 条件下重新逐 head 排名。
```

本阶段目标：

```text
在 Phase135 长模板下重新扫描 true-last layer 全部 attention heads，
判断 short-template causal head set 是否迁移，
并找出 long-template all_pre_answer value field 的真实 head ensemble。
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase136_long_template_head_reranking_cuda.py \
  tests/gpt5/phase136_long_template_head_reranking_summary.py

python tests/gpt5/phase136_long_template_head_reranking_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --categories number \
  --output-dir results/gpt5_phase136_smoke \
  --hard-exit-after-model

python tests/gpt5/phase136_long_template_head_reranking_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 16 \
  --categories number,container,plant,time,clothing,furniture \
  --output-dir results/gpt5_phase136_long_template_head_reranking \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase136_long_template_head_reranking_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 16 \
  --categories number,container,plant,time,clothing,furniture \
  --output-dir results/gpt5_phase136_long_template_head_reranking \
  --hard-exit-after-model

python tests/gpt5/phase136_long_template_head_reranking_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 24 \
  --batch-size 16 \
  --categories number,container,plant,time,clothing,furniture \
  --output-dir results/gpt5_phase136_long_template_head_reranking \
  --hard-exit-after-model

python tests/gpt5/phase136_long_template_head_reranking_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase136_long_template_head_reranking_cuda.py`
- 汇总脚本：`tests/gpt5/phase136_long_template_head_reranking_summary.py`
- Qwen3 结果：`results/gpt5_phase136_long_template_head_reranking/phase136_qwen3_long_template_head_reranking.json`
- GLM4 结果：`results/gpt5_phase136_long_template_head_reranking/phase136_glm4_long_template_head_reranking.json`
- DS7B 结果：`results/gpt5_phase136_long_template_head_reranking/phase136_deepseek7b_long_template_head_reranking.json`
- 跨模型汇总：`results/gpt5_phase136_long_template_head_reranking/phase136_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant, time, clothing, furniture
templates = Phase135 six long templates
source_group = all_pre_answer
operation = remove per-head value contribution at true last layer answer position
aggregates = long_top1, long_top2, long_top4, long_top8, short_template_core, all_heads
```

数据量：

```text
Qwen3: train 8/object/category, test 16/object/category, prompts/category = 96
GLM4: train 8/object/category, test 16/object/category, prompts/category = 96
DS7B: train 12/object/category, test 24/object/category, prompts/category = 144
```

### 客观结果

#### DS7B

DS7B 的 long-template top heads 与 short-template core 部分迁移，不是完全换头。

```text
short_template_core = [13,12,11,8,25,10,26,24]
```

target-down 类别：

```text
number:
  reference -2.66
  best head H11 -0.15
  long_top4 [11,13,10,8] -0.60
  long_top8 [11,13,10,8,19,7,25,27] -0.90
  short_core -0.64
  all_heads -0.39

container:
  reference -2.67
  best head H10 -0.19
  long_top4 [10,11,13,8] -0.68
  long_top8 [10,11,13,8,24,21,26,25] -1.05
  short_core -1.05
  all_heads -1.04

plant:
  reference -2.70
  best head H13 -0.19
  long_top4 [13,10,11,8] -0.66
  long_top8 [13,10,11,8,26,22,24,21] -1.12
  short_core -1.08
  all_heads -1.03

time:
  reference -3.13
  best head H13 -0.20
  long_top4 [13,10,11,8] -0.70
  long_top8 [13,10,11,8,12,7,0,21] -0.94
  short_core -0.82
  all_heads -0.49
```

混合类别：

```text
clothing:
  reference +0.74
  best head H0 -0.07
  long_top8 [0,25,7,22,8,6,13,21] -0.33
  short_core +0.02
  all_heads +0.57

furniture:
  reference +1.15
  best head H0 -0.08
  long_top8 [0,8,7,10,22,13,25,6] -0.41
  short_core -0.09
  all_heads +0.64
```

最稳定 long-template target-down core：

```text
H13, H11, H10, H8
```

它们在 number/container/plant/time 的 top4 中反复出现。

#### Qwen3

Qwen3 结果与 DS7B 不同：

```text
number:
  reference -2.80
  best H11 -0.07
  long_top8 -0.20
  short_core -0.12
  all_heads +1.76

time:
  reference -1.90
  best H11 -0.05
  long_top8 -0.22
  short_core -0.14
  all_heads +1.66

furniture:
  reference -2.67
  best H3 -0.08
  long_top8 -0.25
  short_core -0.14
  all_heads +0.86
```

Qwen3 的 long-template reference 强，但 all_pre_answer per-head value contribution removal 很弱，且 all_heads removal 常使 target 上升。

这说明：

```text
Qwen3 的 long-template reference 不是由当前 DS-style true-last value-contribution head removal 充分解释。
```

#### GLM4 bf16

GLM4 继续存在 old_answer_pos_mismatch：

```text
number 80
container 95
plant 90
time 80
clothing 94
furniture 90
```

target effects 整体较弱：

```text
best head usually around -0.03 to -0.06
long_top8 usually around -0.07 to -0.19
reference also weak or positive in several categories
```

因此 GLM4 本轮仍作为跨模型对照，不作为主理论依据。

### 当前最可靠客观事实

1. **DS7B 的 long-template causal heads 部分继承 Phase133**

Phase133 short-template 核心中：

```text
H13, H11, H10, H8
```

在 Phase136 长模板 target-down 类别中仍然反复进入 top4。

2. **long-template top8 比 top4 更接近 Phase135 all_pre effect**

DS7B：

```text
container long_top8 -1.05, Phase135 all_pre -1.05
plant long_top8 -1.12, Phase135 all_pre -1.08
time long_top8 -0.94, Phase135 all_pre -0.82
number long_top8 -0.90, Phase135 all_pre -0.64
```

说明 Phase135 固定 head set 的不足主要不是核心头丢失，而是模板/类别条件下辅头组合变化。

3. **all_heads 不是上界**

DS7B：

```text
number long_top8 -0.90, all_heads -0.39
time long_top8 -0.94, all_heads -0.49
```

Qwen3：

```text
number long_top8 -0.20, all_heads +1.76
time long_top8 -0.22, all_heads +1.66
```

这说明全部 heads 同时 removal 会混入反向或补偿性成分，不能把 all_heads 当作 causal source 的自然上限。

4. **Qwen3 的机制不同于 DS7B**

Qwen3 有明显 reference effect，但 per-head value contribution removal 弱。

```text
reference strong
single/top-k heads weak
all_heads often target-up
```

因此 Qwen3 可能不是依赖同一类 true-last attention value readout，或者该脚本只捕捉了其中一部分路径。

5. **clothing/furniture 与 target-down 类别不是同一机制**

DS7B clothing/furniture reference 是 target-up：

```text
clothing reference +0.74
furniture reference +1.15
```

但 long_top heads 可产生 target-down：

```text
clothing long_top8 -0.33
furniture long_top8 -0.41
```

这继续支持它们是 suppressor/interface mixed boundary，不应与 number/container/plant/time 混成同一类。

### 对 Phase135 后理论的修正

Phase135 的公式保留，但 heads 需要条件化：

```text
A_c^L(a)
≈
Recode_L(
  Σ_{h∈H(c,T)}
  Σ_{s∈pre_answer}
    α_h(a,s) V_h(s)
)
```

其中：

```text
H(c,T) = stable core heads + category/template-conditioned auxiliary heads
```

DS7B 当前可写成：

```text
H_core ≈ {H13,H11,H10,H8}
```

辅头：

```text
number: H19,H7,H25,H27
container: H24,H21,H26,H25
plant: H26,H22,H24,H21
time: H12,H7,H0,H21
clothing/furniture: H0,H7,H22,H25/H6 等混合头更多
```

更谨慎的解释：

```text
true-last read gateway 不是单一头，也不是全部头；
它是一个条件化 head ensemble。
核心头负责稳定读取任务场，
辅头负责类别/模板/竞争结构的细化。
```

### 硬伤和瓶颈

1. **reference 仍大于 top-k contribution removal**

DS7B reference 约 -2.6 到 -3.1，long_top8 约 -0.9 到 -1.1。

说明仍有未闭合路径：

```text
可能包括 MLP residual recoding、上游层 value field、或非最后层 attention path。
```

2. **single head 效应很小**

DS7B best head 通常只有 -0.15 到 -0.20。

说明因果读取是分布式合成，不应过度解释单头。

3. **Qwen3 不被当前机制解释**

Qwen3 reference 强，但 value contribution path 弱甚至反向。

这可能是模型结构差异，也可能是 hook 位置、RoPE/GQA 映射、或 answer-site monitor basis 与 patch basis 不匹配导致。

4. **GLM4 仍受定位/模板适配问题影响**

GLM4 old_mismatch 很高，结果只能低权重参考。

5. **还没有 generation audit**

当前指标仍是 readout-score causal proxy，尚未证明真实输出 token 的生成概率/生成文本同步变化。

### 下一阶段大任务

Phase137 应做：

```text
Multi-layer Long-template Path Closure
```

不要只继续换小功能，而是沿当前最关键瓶颈做闭合：

```text
1. 以 DS7B number/container/plant/time 为主。
2. 使用 Phase136 long_top8 heads。
3. 同时测试 true-last layer、last-1、last-2、last-3 的 all_pre_answer contribution removal。
4. 比较：
   - single-layer top8
   - cumulative multi-layer top8
   - layer input reference removal
   - random heads / non-top heads control
5. 加 generation-logit audit：
   直接记录目标类别 token logits 和生成首 token 是否同步变化。
```

核心判据：

```text
如果 multi-layer top8 接近 reference，
说明缺口来自跨层累积读取。

如果 multi-layer 仍远小于 reference，
说明主要缺口在 residual/MLP recoding，
下一步应拆 MLP 与 attention 输出。
```




## Phase 137: Mechanism Variable and Transfer Map Closure 机制变量与转移映射闭合 [2026-06-15 00:06]

### 本阶段目标

根据用户要求，综合两个新分析与 Phase135/136 的当前进展，判断当前研究是否正确、有什么进展、有什么硬伤，并重新制定下一阶段大任务。

本阶段不新增模型测试，原因：

当前瓶颈不是缺少一个新的局部 head/source/layer sweep，
而是需要把已有局部发现压缩成可预测、可恢复、可迁移的机制模型。

对两个分析的判断

分析一：Phase135 判断

基本正确。

正确部分：

1. Phase135 成功排除了 Phase134 短模板 pre_object 切分偏置的一部分。
2. long-template 下 object_span 仍接近 0，说明对象词元不是 true-last 主要读取源。
3. prefix 仍有稳定贡献，但不再接近 all_pre_answer 的主效应。
4. 机制应从 prefix-dominant 修正为 broad context field。
5. 固定 Phase133 causal head set 无法完全解释 long-template reference，需要重新排头。

Phase136 已经补上了第 5 点：

DS7B long-template core heads:
H13, H11, H10, H8

结论：
short-template core 不是完全失效，
而是 stable core heads + category/template-conditioned auxiliary heads。

分析二：研究路线风险判断

非常关键，方向正确。

当前研究已经进入危险阶段：

局部结构越来越多，
但整体解释力不一定同步上升。

如果继续只做：

layer × head × source × template × category

会得到越来越大的表格，而不是语言编码机制。

因此必须从：

local ablation scan
局部消融扫描

升级到：

mechanism variable learning
机制变量学习

transfer map closure
转移映射闭合

restore / swap causal test
恢复 / 交换因果测试

generation audit
生成审计

当前研究的真实进展

进展一：否定对象词元单点语义向量

多阶段证据一致：

object_last 弱
object_span 弱
true-last object_span value contribution 接近 0
long-template 下仍接近 0

这说明：

模型不是在最后从对象词元直接读取概念向量。

更可靠的描述：

对象词元在上游触发约束形成；
约束随后扩散或重编码到上下文字段；
答案位置在后层读取这个上下文字段。

进展二：从方向理论升级到场理论

早期问题是：

某类别有没有固定方向？

现在问题已经变成：

残差场在哪里形成？
值场从哪里到答案位置？
哪些头执行读取？
答案位置如何重编码？
模板如何改变路径？

也就是从：

direction
方向

升级为：

field
场

进展三：发现分布式但可压缩的读取结构

Phase136 的 DS7B 结果显示：

H13, H11, H10, H8

在长模板 target-down 类别中反复出现。

这说明机制不是完全分散、不可解释，而是：

stable core
稳定核心

+

conditional auxiliary heads
条件化辅助头

这是当前最重要的正信号。

进展四：发现模型分型

DS7B：

true-last attention value readout path 明显存在。

Qwen3：

reference effect 强，
但 per-head value contribution removal 弱甚至反向。

GLM4：

定位与模板适配问题仍重，
当前链路不能作为主证据。

这说明不同模型内部可能实现不同编码策略。

当前最大硬伤

硬伤一：reference 仍未闭合

DS7B：

reference 约 -2.6 到 -3.1
long_top8 约 -0.9 到 -1.1

说明还有大量路径没有解释：

可能是跨层累积 attention path；
可能是 MLP/residual recoding；
可能是 final norm 非线性放大；
也可能是 value contribution hook 只捕捉了一部分真实通路。

硬伤二：局部归因过多

当前已经出现：

layer
head
source group
position
template
category
scale
control

这些变量都能产生局部结果，但不一定构成机制。

真正问题应该是：

哪些低维机制变量能预测干预结果？
哪些变量能恢复输出？
哪些变量能跨模板迁移？

硬伤三：缺少 sufficiency

当前大多数测试是：

remove 后是否变坏

但真正机制闭合还需要：

restore 后是否救回
swap 后是否换类
只保留最小路径是否足够

硬伤四：generation audit 不足

当前多用 readout score 或 logit proxy。

必须最终验证：

真实生成 token
目标类别概率
竞争类别释放
错误输出形态

否则仍可能只是代理指标。

条件化关系因子动力学公式更新

旧公式强调单一边界或局部头：

A_c^L(a)
≈
Recode_L(
  Σ_{h∈H(c,T)}
  Σ_{s∈pre_answer}
    α_h(a,s)V_h(s)
)

这个公式仍然正确，但不完整。

最新完整版本应写成四段：

1. 上游对象触发：
```text
R_c^l(P,T)
=
F_l(
  object,
  template,
  context
)
```
2. 答案前上下文字段形成：
```text
C_c^l(P,T)
=
G_l(
  R_c^l,
  T_frame,
  competitors
)

3. 条件化头集合读取：
```
```text
V_c^L(a)
=
Σ_{h∈H(c,T)}
Σ_{s∈P}
  α_h(a,s | c,T)
  V_h(s)

其中：
```
```text
H(c,T)
=
H_core
+
H_aux(c,T)

4. 答案位置重编码并输出：
```
```text
A_c^L(a)
=
Φ_L(
  C_c^{L-k}(P,T),
  V_c^L(a),
  MLP_L,
  Norm_L
)
```
```text
logit_c
=
Readout(
  A_c^L(a)
)

用中文解释：

对象词元不是最终概念存储点；
它先触发一个类别相关的上下文字段。

答案位置不是简单复制对象语义；
它通过条件化注意力头集合读取答案前字段。

读取头集合不是固定全集；
它由稳定核心头和类别/模板辅助头组成。

最终输出还受到 MLP、残差重编码和归一化非线性影响。

当前第一性原理表述：

语言编码不是固定概念向量，
而是在任务框架中形成的可转移约束场。

破解语言编码机制，
不是寻找一个永恒方向，
而是寻找：
对象意义如何被任务框架坐标化，
上下文字段如何被答案位置读取，
读取后的状态如何被映射为输出。

### 关键洞察

对象不是终点，而是触发器

object token 不是最后读取源，
它更像约束形成的触发点。

答案位置是汇聚点

answer site 是模型把上下文字段压成输出约束的地方。

模板不是噪声，而是坐标系

template 决定同一语义如何被重新坐标化。

头不是机制本体，而是读取算子

H13/H11/H10/H8 很重要，
但它们只是读取通道，
不是语言编码机制本身。

真正机制必须能预测、恢复、交换

只能解释已有消融结果还不够；
必须能预测新模板，
恢复被破坏输出，
交换类别约束。

### 接下来阶段方案

下一阶段不应只是继续原版 Phase137 的多层扫描，而应升级为：

Phase138:
Mechanism Variable and Transfer Map Closure
机制变量与转移映射闭合

Phase138 总目标

把已有局部发现压缩成低维机制模型，
验证它能否预测、恢复、交换类别输出。

主测模型

DS7B 为主。
Qwen3 作为分型对照。
GLM4 暂缓为低权重对照。

主测类别

干净 target-down 类别：
number, container, plant, time

混合 probe 类别：
clothing, furniture

数据范围

templates:
short templates
long templates
paraphrase templates

train objects/category:
16

test objects/category:
32

splits:
至少 3 个 object split

模块 A：机制变量表

每个 prompt 记录：

R_pre:
pre-answer residual field strength
答案前残差场强度

O_value:
source-to-answer value field
来源到答案值场

H_score:
causal head ensemble score
因果头集合分数

A_answer:
answer-site readout alignment
答案位置读出对齐

T_frame:
template frame factor
模板框架因子

C_release:
competitor release score
竞争类别释放分数

预测指标：

target_delta
release_delta
answer_proj_delta
generation_change

模块 B：低维预测模型

目标不是追求复杂预测器，而是寻找最小充分变量：
```
```text
target_delta
=
f(
  R_pre,
  O_value,
  H_score,
  A_answer,
  T_frame,
  C_release
)

成功标准：

少于 10 个机制变量解释 70% 以上 target_delta 方差。

如果失败，说明当前变量仍是外围指标。

模块 C：跨位点转移映射

学习：

A_answer ≈ W · R_pre

以及：

A_answer ≈ W_value · O_value

测试：

heldout objects
留出对象

heldout templates
留出模板

cross-template transfer
跨模板迁移

模块 D：restore / swap 因果闭合

三步：

1. remove R_pre
   移除答案前残差场

2. restore W·R_pre at answer site
   在答案位置恢复映射后的状态

3. swap W·R_pre between categories
   在类别之间交换映射后的状态

成功判据：

remove 后输出下降；
restore 后输出恢复；
swap 后输出偏向被换入类别。

模块 E：最小充分路径

学习稀疏 mask：

layers
heads
source groups
MLP components

目标：

只保留少数路径仍能复现 clean logits。

如果 mask 很大：

说明机制是宽场机制，不适合继续追单头。

模块 F：生成审计

对以下条件记录真实生成：

clean
remove
restore
swap
minimal mask

记录：

target probability
目标概率

competitor probability
竞争概率

generated token
生成词元

valid category rate
有效类别率

Phase138 成功判据

强成功：

1. 低维机制变量能预测多数 target_delta。
2. 转移映射能跨对象/模板迁移。
3. restore 能救回输出。
4. swap 能改变输出类别。
5. 最小充分路径可压缩。
6. generation audit 同步变化。

中等成功：

同模板有效，跨模板弱。
说明需要 template-conditioned transfer map。

失败：

变量预测差；
转移映射不稳定；
restore 无法救回。

失败时说明：

当前 head/source/subspace 变量仍偏外围，
需要转向更高层概念图或更低层神经元群体动力学。

### 本阶段结论

当前研究不是没有进展，而是已经从寻找概念方向，推进到寻找语言约束场。

但是下一步不能继续无限细分局部组件。

最重要的转向是：

从“哪里有效”
转向“哪些机制变量足以预测和恢复输出”。

只有完成：

预测
恢复
交换
生成同步

才能说真正开始接近语言背后的编码机制。

## Phase 138: Mechanism Transfer Closure 初测 [2026-06-15 00:29]

### 本阶段目标

根据 Phase137 的新方向，开始从局部 head/source/layer 扫描转向机制变量闭合。

本轮不做最小充分 mask，也不做完整 generation audit，而是先测试一个核心问题：

last-layer pre-answer residual field
答案前残差场

能否通过低秩 transfer map
低秩转移映射

预测并部分恢复 answer-site readout field
答案位置读出场

### 执行命令

python -m py_compile \
  tests/gpt5/phase138_mechanism_transfer_closure_cuda.py \
  tests/gpt5/phase138_mechanism_transfer_closure_summary.py

python tests/gpt5/phase138_mechanism_transfer_closure_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --rank 4 \
  --categories number,container \
  --output-dir results/gpt5_phase138_smoke \
  --hard-exit-after-model

python tests/gpt5/phase138_mechanism_transfer_closure_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 16 \
  --rank 8 \
  --categories number,container,plant,time,clothing,furniture \
  --output-dir results/gpt5_phase138_mechanism_transfer_closure \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase138_mechanism_transfer_closure_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 16 \
  --rank 8 \
  --categories number,container,plant,time,clothing,furniture \
  --output-dir results/gpt5_phase138_mechanism_transfer_closure \
  --hard-exit-after-model

python tests/gpt5/phase138_mechanism_transfer_closure_cuda.py deepseek7b \
  --train-objects 16 \
  --test-objects 32 \
  --batch-size 16 \
  --rank 8 \
  --categories number,container,plant,time,clothing,furniture \
  --output-dir results/gpt5_phase138_mechanism_transfer_closure \
  --hard-exit-after-model

python tests/gpt5/phase138_mechanism_transfer_closure_summary.py

### 脚本与结果

主测试脚本：tests/gpt5/phase138_mechanism_transfer_closure_cuda.py

汇总脚本：tests/gpt5/phase138_mechanism_transfer_closure_summary.py

Qwen3 结果：results/gpt5_phase138_mechanism_transfer_closure/phase138_qwen3_mechanism_transfer_closure.json

GLM4 结果：results/gpt5_phase138_mechanism_transfer_closure/phase138_glm4_mechanism_transfer_closure.json

DS7B 结果：results/gpt5_phase138_mechanism_transfer_closure/phase138_deepseek7b_mechanism_transfer_closure.json

跨模型汇总：results/gpt5_phase138_mechanism_transfer_closure/phase138_cross_model_summary.md

方法

对每个模型和类别：

1. 捕获 true-last layer input 的 all_pre_answer 平均残差向量：
   R_pre

2. 捕获 true-last layer output 的 answer-site 向量：
   A_answer

3. 用训练对象学习低秩转移映射：
   A_answer_coeff ≈ W · R_pre_coeff

4. 在 heldout objects 上测试：
   transfer R2
   transfer cosine

5. 做三种因果条件：
   remove:
     移除 R_pre 子空间

   restore:
     移除 R_pre 后，在 answer site 加回 W·R_pre

   swap:
     移除目标 R_pre 后，在 answer site 加入另一个类别的 answer prototype

本轮是 proxy closure，不是最终完整闭合。

### 测试范围

templates = Phase135 six long templates
categories = number, container, plant, time, clothing, furniture
rank = 8

数据量：

Qwen3:
  train 8 objects/category
  test 16 objects/category

GLM4:
  train 8 objects/category
  test 16 objects/category

DS7B:
  train 16 objects/category
  test 32 objects/category

客观结果

DS7B

number:
  transfer R2 +0.26, cosine +0.99
  remove target Δ -1.27
  restore target Δ +0.52
  recovery +1.41
  swap to container: container Δ -0.56

container:
  transfer R2 +0.41, cosine +0.98
  remove target Δ -1.41
  restore target Δ -0.52
  recovery +0.63
  swap to plant: plant Δ -0.74

plant:
  transfer R2 +0.32, cosine +0.98
  remove target Δ -1.59
  restore target Δ -0.42
  recovery +0.74
  swap to time: time Δ -1.79

time:
  transfer R2 +0.66, cosine +0.99
  remove target Δ -1.44
  restore target Δ -1.71
  recovery -0.19
  swap to clothing: clothing Δ +0.71

clothing:
  transfer R2 +0.49, cosine +0.98
  remove target Δ +0.42
  restore target Δ +0.88
  recovery +1.07
  swap to furniture: furniture Δ +0.64

furniture:
  transfer R2 +0.41, cosine +0.98
  remove target Δ -0.24
  restore target Δ +0.09
  recovery +1.36
  swap to number: number Δ +0.86

DS7B 最重要结果：

number/container/plant:
  remove 后 target 明显下降；
  restore 后明显救回。

这说明：

R_pre -> A_answer 的低秩转移映射捕捉到了一部分真实因果机制。

但：

swap 不稳定。
container/plant/time 的 swap 方向没有成功把目标推向换入类别。

因此还不能说完成类别交换闭合。

Qwen3

number:
  R2 +0.28
  remove -1.21
  restore -0.41
  recovery +0.66

container:
  R2 +0.30
  remove -0.55
  restore +0.90
  recovery +2.64

plant:
  R2 +0.60
  remove +0.51
  restore -0.07

time:
  R2 +0.35
  remove -0.24
  restore -0.40

Qwen3 有部分 restore 成功，但类别间模式不稳定。

这与 Phase136 一致：

Qwen3 存在 answer-site/pre-answer reference effect，
但不完全符合 DS7B 的 true-last value-readout 机制。

GLM4 bf16

R2 大约 +0.28 到 +0.53
cosine 大约 +0.84 到 +0.93
remove target effect 多数很弱
restore 常出现过度或反向变化

GLM4 仍作为低权重对照。

当前最可靠客观事实

低秩转移映射有预测能力，但不是充分机制

三模型多数类别：

transfer R2 > 0
cosine 很高

但 R2 不是很高，说明：

低秩 R_pre -> A_answer 映射捕捉方向结构，
但不能解释全部幅度和类别竞争结构。

DS7B number/container/plant restore 明显成立

DS7B：

number remove -1.27 -> restore +0.52
container remove -1.41 -> restore -0.52
plant remove -1.59 -> restore -0.42

说明：

pre-answer field 到 answer-site field 的映射不是纯相关；
它有可干预恢复的因果成分。

time 是异常类别

DS7B time：

R2 +0.66
remove -1.44
restore -1.71

虽然预测 R2 最高，但 restore 失败。

说明：

预测 answer-state 坐标
不等于
能够因果恢复 logits。

这可能是 time 类存在更强竞争/抑制结构，或者恢复向量的 scale/site 不合适。

swap 尚未闭合

swap 对 clothing/furniture/number 有部分正向迹象：

DS7B clothing -> furniture: furniture Δ +0.64
DS7B furniture -> number: number Δ +0.86
DS7B time -> clothing: clothing Δ +0.71

但对 target-down 核心类别：

number -> container: container Δ -0.56
container -> plant: plant Δ -0.74
plant -> time: time Δ -1.79

交换失败。

因此当前只达到：

partial restore closure
部分恢复闭合

尚未达到：

swap closure
交换闭合

### 对理论的修正

Phase137 公式中的：
```text
A_answer ≈ W · R_pre
```

现在得到部分支持，但必须加限制：
```text
A_answer_state ≈ W · R_pre
```

不等于：
```text
logit_c fully determined by W · R_pre
```

更完整写法：

```text
A_c^L(a)
=
Φ_L(
  W_c,T R_c^{pre},
  Q_c,T,
  B_competition,
  Norm
)
```
其中：

W_c,T:
  类别/模板条件化转移映射

Q_c,T:
  未被低秩 R_pre 捕捉的残差/MLP/attention 贡献

B_competition:
  类别竞争和抑制结构

Norm:
  final normalization 非线性缩放

因此当前最新理论是：

语言编码不是单纯 R_pre 到 A_answer 的线性转移；
低秩转移映射是其中一条真实路径，
但输出还需要竞争结构、归一化、MLP/残差重编码共同决定。

硬伤和问题

answer_proj_delta 数值异常大

本轮 answer projection 使用 rank-8 norm 型投影，remove 后出现很大的负数。

这说明：

projection norm 指标受尺度/归一化强烈影响，
不能单独作为机制成功指标。

更可靠的是：

target_delta
restore_recovery_ratio
swap_category_delta

restore scale 未调参

当前 restore_scale = 1.0。

time 失败可能只是 scale/site 不合适，也可能是真机制缺失。

swap 使用 prototype 太粗糙

当前 swap 加入的是换入类别 answer prototype，不是样本条件化的 swap vector。

所以 swap 失败不能直接否定类别交换机制。

只测 long templates

还没有验证 short/long/paraphrase 的模板迁移。

还没有 generation audit

本轮仍是 logit/readout proxy。

### 下一阶段任务

Phase139 应继续做：

Restore/Swap Calibration and Template Transfer
恢复/交换校准与模板迁移

核心目标：

判断 Phase138 的 partial restore closure 是否能变成稳定 restore/swap closure。

测试设计：

1. DS7B 为主。
2. 类别先聚焦 number/container/plant/time。
3. restore_scale sweep:
   0.25, 0.5, 1.0, 1.5, 2.0
4. restore site sweep:
   last_layer_input_answer
   last_attention_output_answer
   last_block_output_answer
   final_norm_input
5. swap 从 prototype 改成 sample-conditioned swap:
   用目标 prompt 的 template frame，
   换入类别的 object/category field。
6. 加 short/long/paraphrase template transfer:
   train on long, test on paraphrase；
   train on mixed, test on heldout templates。
7. 记录 generation logits 和真实首 token。

成功判据：

restore:
  remove 后下降；
  restore 后稳定恢复 50% 以上；
  不显著释放错误竞争类别。

swap:
  换入类别 logits 上升；
  原目标类别下降或不主导；
  生成首 token 随之改变。

template transfer:
  W 在 heldout templates 上仍有正 R2 和 restore 效果。

如果 Phase139 成功：

可以说已找到一条可恢复、可迁移的语言编码转移路径。

如果失败：

说明 Phase138 的 W 主要是相关映射，
需要转向 MLP/Norm/competition decomposition。


## Phase 139: Restore/Swap Calibration 恢复交换校准 [2026-06-15 06:28]

### 本阶段目标

根据用户要求，先判断附件分析是否正确，再继续完成客观测试。

附件判断基本正确：

```text
1. Phase137/138 的方向正确，研究已经从局部定位转向机制闭合。
2. Phase138 的 R_pre -> A_answer 低秩转移映射捕捉到真实路径的一部分。
3. Phase138 只达到 partial restore closure，没有达到 swap closure。
4. time 类 R2 高但 restore 失败，说明几何预测不等于因果恢复。
5. 下一步必须做 restore scale/site calibration 和 sample-conditioned swap。
```

本阶段目标：

```text
判断 Phase138 的 partial restore closure 是否能通过尺度/位点校准变成稳定 restore；
判断 sample-conditioned swap 是否优于 prototype swap；
继续以客观结果拼图，不急于总结完整理论。
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase139_restore_swap_calibration_cuda.py \
  tests/gpt5/phase139_restore_swap_calibration_summary.py

python tests/gpt5/phase139_restore_swap_calibration_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --rank 4 \
  --categories number,container \
  --restore-scales 0.5,1.0 \
  --swap-scales 1.0 \
  --restore-sites input_answer,block_output \
  --output-dir results/gpt5_phase139_smoke \
  --hard-exit-after-model

python tests/gpt5/phase139_restore_swap_calibration_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 16 \
  --rank 8 \
  --categories number,container,plant,time \
  --restore-scales 0.25,0.5,1.0,1.5,2.0 \
  --swap-scales 0.5,1.0,1.5 \
  --restore-sites input_answer,block_output \
  --output-dir results/gpt5_phase139_restore_swap_calibration \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase139_restore_swap_calibration_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 16 \
  --rank 8 \
  --categories number,container,plant,time \
  --restore-scales 0.25,0.5,1.0,1.5,2.0 \
  --swap-scales 0.5,1.0,1.5 \
  --restore-sites input_answer,block_output \
  --output-dir results/gpt5_phase139_restore_swap_calibration \
  --hard-exit-after-model

python tests/gpt5/phase139_restore_swap_calibration_cuda.py deepseek7b \
  --train-objects 16 \
  --test-objects 32 \
  --batch-size 16 \
  --rank 8 \
  --categories number,container,plant,time \
  --restore-scales 0.25,0.5,1.0,1.5,2.0 \
  --swap-scales 0.5,1.0,1.5 \
  --restore-sites input_answer,block_output \
  --output-dir results/gpt5_phase139_restore_swap_calibration \
  --hard-exit-after-model

python tests/gpt5/phase139_restore_swap_calibration_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase139_restore_swap_calibration_cuda.py`
- 汇总脚本：`tests/gpt5/phase139_restore_swap_calibration_summary.py`
- Qwen3 结果：`results/gpt5_phase139_restore_swap_calibration/phase139_qwen3_restore_swap_calibration.json`
- GLM4 结果：`results/gpt5_phase139_restore_swap_calibration/phase139_glm4_restore_swap_calibration.json`
- DS7B 结果：`results/gpt5_phase139_restore_swap_calibration/phase139_deepseek7b_restore_swap_calibration.json`
- 跨模型汇总：`results/gpt5_phase139_restore_swap_calibration/phase139_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant, time
templates = Phase135 six long templates
rank = 8
restore_sites = input_answer, block_output
restore_scales = 0.25, 0.5, 1.0, 1.5, 2.0
swap_scales = 0.5, 1.0, 1.5
swap pairs = number->container, container->plant, plant->time, time->number
```

数据量：

```text
Qwen3:
  train 8 objects/category
  test 16 objects/category

GLM4:
  train 8 objects/category
  test 16 objects/category

DS7B:
  train 16 objects/category
  test 32 objects/category
```

### 客观结果

#### DS7B

```text
number:
  transfer R2 +0.26
  remove target Δ -1.27
  best restore = block_output scale2.0, target Δ +1.12, recovery +1.88, release +1.74
  best sample swap = input_answer scale1.5, swap container Δ +0.34, target Δ -1.71

container:
  transfer R2 +0.41
  remove target Δ -1.41
  best restore = input_answer scale2.0, target Δ +0.66, recovery +1.47, release +0.00
  best sample swap = block_output scale1.5, swap plant Δ -0.46, target Δ -0.49

plant:
  transfer R2 +0.32
  remove target Δ -1.59
  best restore = block_output scale2.0, target Δ -0.11, recovery +0.93, release +1.37
  best sample swap = block_output scale0.5, swap time Δ -1.69, target Δ -0.70

time:
  transfer R2 +0.66
  remove target Δ -1.44
  best restore = block_output scale0.25, target Δ -1.49, recovery -0.04
  best sample swap = block_output scale1.5, swap number Δ +1.19, target Δ +0.91
```

DS7B restore 细节：

```text
number:
  block_output scale0.5 already recovers +0.92 ratio;
  scale2.0 over-recovers and releases competitors.

container:
  input_answer scale1.0 recovers +0.80 ratio with no competitor release;
  scale2.0 over-recovers but still no release.

plant:
  block_output scale1.0 recovers +0.74 ratio;
  scale2.0 recovers +0.93 but releases competitors +1.37.

time:
  all restore scales are <= 0 recovery;
  input_answer larger scale makes target worse.
```

#### Qwen3

Qwen3 restore/swap 都容易过度注入：

```text
number:
  best restore input_answer scale2.0, target Δ +7.26, release +8.33
  best swap container Δ +5.11, target Δ +4.14

container:
  best restore target Δ +6.45, release +8.21
  best swap plant Δ +2.93, target Δ +4.85

plant:
  remove target Δ +0.51 already不是 target-down；
  restore target Δ +4.84, release +7.19

time:
  restore target Δ +5.48, release +7.93
```

说明 Qwen3 的 input_answer 注入可以强烈改变 readout，但选择性差、竞争释放过大。

#### GLM4

GLM4 效应弱且不稳定：

```text
number:
  remove -0.03，best restore 仍 -0.29

container:
  remove -0.14，input_answer scale2.0 restore +0.32

plant:
  remove -0.14，input_answer scale2.0 restore +0.40

time:
  remove +0.12，best restore -0.19
```

GLM4 继续只作为低权重对照。

### 当前最可靠客观事实

1. **DS7B number/container/plant 的 restore 不是偶然**

经过 scale/site sweep 后：

```text
number recovery up to +1.88
container recovery up to +1.47
plant recovery up to +0.93
```

说明 Phase138 的 partial restore closure 得到加强。

2. **restore 的最佳 site 具有类别差异**

```text
number: block_output 最强
container: input_answer 最干净
plant: block_output 更强
time: 无有效 restore site
```

这说明：

```text
W·R_pre 不是一个全局统一注入点；
恢复路径与类别/竞争结构有关。
```

3. **time 失败不是简单 scale 问题**

DS7B time：

```text
input_answer scale0.25 target Δ -1.87
input_answer scale2.0 target Δ -3.75
block_output scale0.25 target Δ -1.49
block_output scale2.0 target Δ -1.94
```

所有 restore 都不能救回，说明 time 的失败不是 restore_scale=1.0 太小。

4. **sample-conditioned swap 有局部进展，但整体仍未闭合**

DS7B：

```text
number -> container:
  container Δ +0.34，target number Δ -1.71

time -> number:
  number Δ +1.19，target time Δ +0.91
```

但：

```text
container -> plant:
  plant Δ -0.46

plant -> time:
  time Δ -1.69
```

因此 sample-conditioned swap 优于 prototype swap 的局部迹象存在，但还没有形成稳定 swap closure。

5. **高 scale restore 会带来过度恢复和竞争释放**

例如 DS7B number：

```text
block_output scale2.0 target Δ +1.12
release +1.74
```

这说明 restore 成功不能只看 target_delta，还要看 competitor release。

### 理论修正

Phase139 后，公式应进一步加入 calibration 和 competition gate：

```text
A_c^L(a)
=
Φ_L(
  γ_{c,s} · Inject_s(W_{c,T}R_c^{pre}),
  Q_{c,T},
  B_comp,
  Norm
)
```

其中：

```text
γ_{c,s}:
  类别 c 在恢复位点 s 的有效尺度

Inject_s:
  注入/恢复位点函数

B_comp:
  竞争释放和抑制结构
```

当前更准确理论：

```text
R_pre -> A_answer 是真实可恢复路径；
但该路径必须经过类别条件化的尺度和位点校准；
并且不能单独决定 swap，因为 swap 还需要竞争结构同步改变。
```

### 硬伤和瓶颈

1. **best restore 可能是过度恢复**

高 target_delta 伴随高 release，不一定是机制闭合。

需要新增 clean-restore criterion：

```text
target recovery high
competitor release low
generation category stable
```

2. **swap 仍未闭合**

sample-conditioned swap 仍只在 number->container 和 time->number 有局部正向。

3. **time 类成为关键反例**

time 的 transfer R2 最高，但 restore 完全失败。

这说明：

```text
状态几何预测与输出因果控制之间有断层。
```

4. **仍只测 long templates**

template transfer 仍未完成。

5. **generation audit 仍未完成**

当前只记录 DCF readout score，不是真实生成首 token。

### 下一阶段任务

Phase140 应做：

```text
Clean Restore Criterion and Competition Decomposition
干净恢复判据与竞争结构分解
```

核心目标：

```text
把 restore 从“target 变大”升级为“干净恢复”；
解释 time 为什么 R2 高但 restore 失败；
判断 swap 失败是否来自竞争结构没有同步切换。
```

测试设计：

```text
1. DS7B 为主，类别 number/container/plant/time。
2. 对 Phase139 restore 条件计算 clean_restore_score:
   recovery_ratio - λ * competitor_release
3. 对每个类别记录 top competitor 的变化。
4. 构建 support/release/suppressor 三分量：
   support = target_delta
   release = max_other_delta
   suppressor = target_delta - max_other_delta
5. 对 time 做特殊审计：
   restore 后哪些 competitor 被释放？
   是否存在 time-specific suppressor direction？
6. 加 generation-logit audit：
   记录 target category token、top competitor token、generated first token。
7. 再做 template transfer：
   long train -> paraphrase test。
```

成功判据：

```text
number/container/plant:
  存在 clean_restore_score 高的 restore setting。

time:
  能解释 restore 失败来自竞争释放、抑制方向缺失，或位点不匹配。

swap:
  如果支持成分换入但竞争结构不换入，
  则 swap_category_delta 仍不稳定；
  这将支持 competition-conditioned transfer theory。
```


## Phase 140: Clean Restore and Competition Audit 干净恢复与竞争审计 [2026-06-15 06:47]

### 本阶段目标

根据用户要求，先判断附件分析是否正确，再继续完成客观测试。

附件判断基本正确：

```text
1. Phase139 方向正确，restore scale/site calibration 证明 Phase138 的恢复路径不是偶然。
2. restore 成功不等于 clean restore。
3. swap 仍未闭合，竞争结构可能没有同步切换。
4. time 是关键反例：R2 高但 restore 失败。
5. 下一步必须引入 clean_restore_score、competitor decomposition 和 first-token audit。
```

本阶段目标：

```text
把 restore 从“target_delta 最大”改成“目标恢复 + 竞争释放低”的 clean restore 判据；
记录 top competitor；
记录答案位置下一词 logits argmax，作为低成本 generation audit。
```

本轮不调用长文本 generate，只记录 first-token argmax，避免 token 成本和测试时间膨胀。

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase140_clean_restore_competition_cuda.py \
  tests/gpt5/phase140_clean_restore_competition_summary.py

python tests/gpt5/phase140_clean_restore_competition_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --rank 4 \
  --categories number,container \
  --restore-scales 0.5,1.0 \
  --restore-sites input_answer,block_output \
  --output-dir results/gpt5_phase140_smoke \
  --hard-exit-after-model

python tests/gpt5/phase140_clean_restore_competition_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 16 \
  --rank 8 \
  --categories number,container,plant,time,clothing,furniture \
  --restore-scales 0.25,0.5,1.0,1.5,2.0 \
  --restore-sites input_answer,block_output \
  --lambda-release 0.5 \
  --output-dir results/gpt5_phase140_clean_restore_competition \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase140_clean_restore_competition_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 16 \
  --rank 8 \
  --categories number,container,plant,time,clothing,furniture \
  --restore-scales 0.25,0.5,1.0,1.5,2.0 \
  --restore-sites input_answer,block_output \
  --lambda-release 0.5 \
  --output-dir results/gpt5_phase140_clean_restore_competition \
  --hard-exit-after-model

python tests/gpt5/phase140_clean_restore_competition_cuda.py deepseek7b \
  --train-objects 16 \
  --test-objects 32 \
  --batch-size 16 \
  --rank 8 \
  --categories number,container,plant,time,clothing,furniture \
  --restore-scales 0.25,0.5,1.0,1.5,2.0 \
  --restore-sites input_answer,block_output \
  --lambda-release 0.5 \
  --output-dir results/gpt5_phase140_clean_restore_competition \
  --hard-exit-after-model

python tests/gpt5/phase140_clean_restore_competition_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase140_clean_restore_competition_cuda.py`
- 汇总脚本：`tests/gpt5/phase140_clean_restore_competition_summary.py`
- Qwen3 结果：`results/gpt5_phase140_clean_restore_competition/phase140_qwen3_clean_restore_competition.json`
- GLM4 结果：`results/gpt5_phase140_clean_restore_competition/phase140_glm4_clean_restore_competition.json`
- DS7B 结果：`results/gpt5_phase140_clean_restore_competition/phase140_deepseek7b_clean_restore_competition.json`
- 跨模型汇总：`results/gpt5_phase140_clean_restore_competition/phase140_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant, time, clothing, furniture
templates = Phase135 six long templates
rank = 8
restore_sites = input_answer, block_output
restore_scales = 0.25, 0.5, 1.0, 1.5, 2.0
clean_restore_score = recovery_ratio - 0.5 * max_other_delta
```

数据量：

```text
Qwen3:
  train 8 objects/category
  test 16 objects/category

GLM4:
  train 8 objects/category
  test 16 objects/category

DS7B:
  train 16 objects/category
  test 32 objects/category
```

### 客观结果

#### DS7B

```text
number:
  transfer R2 +0.26
  remove target Δ -1.27, release +0.00
  best target restore = block_output scale2.0
    target Δ +1.12, release +1.74, recovery +1.88, clean +1.01
    top competitor = furniture +1.74
    first-token argmax mostly " often" 0.54

container:
  transfer R2 +0.41
  remove target Δ -1.41, release +0.00
  best clean restore = input_answer scale2.0
    target Δ +0.66, release +0.00, recovery +1.47, clean +1.47
    no positive top competitor
    first-token argmax distributed: quote/a/the

plant:
  transfer R2 +0.32
  remove target Δ -1.59, release +0.00
  best target restore = block_output scale2.0
    target Δ -0.11, release +1.37, recovery +0.93, clean +0.24
    top competitor = furniture +1.37
  best clean restore = input_answer scale0.5
    target Δ -0.71, release +0.17, recovery +0.56, clean +0.47
    top competitor = tool +0.17

time:
  transfer R2 +0.66
  remove target Δ -1.44, release +0.17
  best clean restore = block_output scale0.25
    target Δ -1.49, release +0.17, recovery -0.04, clean -0.12
    top competitor = furniture +0.17

clothing:
  transfer R2 +0.49
  remove target Δ +0.42, release +0.43
  best clean restore = block_output scale1.5
    target Δ +0.89, release +0.65, recovery +1.10, clean +0.78
    top competitor = furniture +0.65

furniture:
  transfer R2 +0.41
  remove target Δ -0.24, release +0.00
  best clean restore = block_output scale1.0
    target Δ +0.09, release +0.19, recovery +1.36, clean +1.27
    top competitor = clothing +0.19
```

#### Qwen3

Qwen3 的 clean_restore_score 在 lambda=0.5 下仍选高 scale，但 first-token argmax 明显异常：

```text
number:
  best clean = input_answer scale2.0
  target Δ +7.26, release +8.33
  top competitor communication +8.33
  first tokens include "clidean", "STRUCTOR", "theless"

container:
  target Δ +6.45, release +8.21
  top competitor communication +8.21

plant:
  target Δ +4.84, release +7.19
  top competitor building +7.19
```

说明 Qwen3 的 injection 高增益且失真，当前 restore 不是干净机制恢复。

#### GLM4

GLM4 效应继续弱且不稳定：

```text
number:
  remove -0.03，best clean 仍 -0.29

container:
  best clean target +0.32，release +1.21

plant:
  best clean target +0.40，release +1.18

time:
  remove already +0.12，best clean -0.19
```

仍作为低权重对照。

### 当前最可靠客观事实

1. **container 是最干净的 restore 成功样本**

DS7B container：

```text
target Δ +0.66
release +0.00
recovery +1.47
clean +1.47
```

这是目前最接近 clean restore closure 的类别。

2. **plant 的 target-best 与 clean-best 分离**

DS7B plant：

```text
target-best:
  block_output scale2.0
  target -0.11, release +1.37, clean +0.24

clean-best:
  input_answer scale0.5
  target -0.71, release +0.17, clean +0.47
```

这证明 Phase140 的 clean criterion 有实际价值：最大 target 恢复不是最佳机制恢复。

3. **number restore 强，但不干净**

DS7B number 的 target 恢复很强，但释放 furniture：

```text
target +1.12
furniture +1.74
```

说明 number 的恢复向量混入竞争释放，或缺少 suppressor component。

4. **time 仍是关键反例**

DS7B time：

```text
R2 +0.66
remove -1.44
best clean restore -1.49
```

这强烈支持：

```text
time 的 answer-state 几何可预测，
但输出控制路径不在当前 W·R_pre restore 通道中。
```

5. **first-token audit 暴露 qwen3 注入失真**

Qwen3 高 scale restore 后，first-token argmax 不是稳定类别词，而是异常碎片 token。

说明：

```text
logit/readout target_delta 变大不等于生成行为正确。
```

### 理论修正

Phase140 后，条件化关系因子动力学公式应增加 clean gate：

```text
A_c^L(a)
=
Φ_L(
  Inject_s(
    γ_{c,T,s} · W_{c,T}R_c^{pre}
  ),
  Q_{c,T},
  B_support(c),
  B_suppress(c),
  Norm_s
)
```

其中必须区分：

```text
B_support:
  目标支持成分

B_suppress:
  竞争抑制成分
```

当前结论：

```text
W·R_pre 可以恢复 support；
但 clean restore 还需要 suppressor/competition component；
time 的失败说明有些类别主要缺的是 competition/suppressor 结构，
不是 answer-state 几何。
```

### 硬伤和瓶颈

1. **lambda=0.5 仍不足以排除过度恢复**

number 仍选择 release +1.74 的高 scale setting。

下一步需要硬约束：

```text
release <= threshold
```

而不只是线性惩罚。

2. **first-token argmax 仍不是完整 generation**

本轮只看 argmax token，未调用 generate。

3. **类别 token readout 与真实 token argmax 不一致**

这说明 DCF readout 仍是 proxy，不能替代生成行为。

4. **time 机制尚未解释**

目前只确认当前通道失败，还没有定位 time 的真实通道。

5. **template transfer 仍未测试**

仍使用 long templates。

### 下一阶段任务

Phase141 应做：

```text
Constrained Clean Restore and Time Failure Localization
约束干净恢复与 time 失败定位
```

核心目标：

```text
1. 对 restore 加入 release threshold，寻找真正 clean settings。
2. 对 time 拆解失败来源：support 不足、suppressor 缺失、还是 restore site 错误。
3. 把 DCF readout 与真实 category token logits 对齐。
```

测试设计：

```text
1. DS7B 为主，类别 number/container/plant/time。
2. clean setting 选择规则：
   recovery_ratio >= 0.5
   max_other_delta <= 0.25
3. 对不满足条件的类别记录最小 release setting。
4. 对 time 做更细 site sweep：
   input_answer
   attention_output
   mlp_input
   mlp_output
   block_output
   final_norm_input
5. 对每个 setting 记录：
   DCF category score
   actual category token logits
   first-token argmax
   top competitor token
6. 若 time 所有 answer-site restore 都失败，
   回到 last-1 / last-2 layer 测试。
```

成功判据：

```text
number/container/plant:
  找到 constrained clean restore。

time:
  明确失败是位点问题、竞争抑制问题，还是当前转移映射根本不适用。
```


## Phase 141: Constrained Clean Restore and Time Localization 约束干净恢复与 time 定位 [2026-06-15 07:40]

### 本阶段目标

根据用户要求，先判断附件分析是否正确，再继续完成客观测试。

附件判断基本正确：

```text
1. Phase140 是关键分水岭：target_delta 最大不等于 clean restore。
2. W·R_pre 主要恢复 support component，不能自动恢复 suppressor/competition/format component。
3. Qwen3 高增益注入会导致 next-token 失真，不能解释为机制恢复。
4. time 是关键反例：R2 高但 restore 失败。
5. 下一步必须用硬约束，而不是 lambda soft penalty。
```

本阶段目标：

```text
1. 用硬约束筛选 constrained clean restore：
   recovery_ratio >= 0.5
   max_other_delta <= 0.25

2. 扩大 restore site：
   input_answer
   attention_output
   mlp_input
   mlp_output
   block_output
   final_norm_input

3. 增加数据范围，减少小样本结论被推翻：
   Qwen3/GLM4 train10 test20
   DS7B train20 test40

4. 测 core 与 mixed 类：
   number, container, plant, time, clothing, furniture
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase141_constrained_clean_restore_cuda.py \
  tests/gpt5/phase141_constrained_clean_restore_summary.py

python tests/gpt5/phase141_constrained_clean_restore_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --rank 4 \
  --categories number,container \
  --restore-scales 0.5,1.0 \
  --restore-sites input_answer,attention_output,mlp_input,mlp_output,block_output,final_norm_input \
  --output-dir results/gpt5_phase141_smoke \
  --hard-exit-after-model

python tests/gpt5/phase141_constrained_clean_restore_cuda.py qwen3 \
  --train-objects 10 \
  --test-objects 20 \
  --batch-size 20 \
  --rank 8 \
  --categories number,container,plant,time,clothing,furniture \
  --restore-scales 0.25,0.5,1.0,1.5,2.0 \
  --restore-sites input_answer,attention_output,mlp_input,mlp_output,block_output,final_norm_input \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase141_constrained_clean_restore \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase141_constrained_clean_restore_cuda.py glm4 \
  --train-objects 10 \
  --test-objects 20 \
  --batch-size 20 \
  --rank 8 \
  --categories number,container,plant,time,clothing,furniture \
  --restore-scales 0.25,0.5,1.0,1.5,2.0 \
  --restore-sites input_answer,attention_output,mlp_input,mlp_output,block_output,final_norm_input \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase141_constrained_clean_restore \
  --hard-exit-after-model

python tests/gpt5/phase141_constrained_clean_restore_cuda.py deepseek7b \
  --train-objects 20 \
  --test-objects 40 \
  --batch-size 20 \
  --rank 8 \
  --categories number,container,plant,time,clothing,furniture \
  --restore-scales 0.25,0.5,1.0,1.5,2.0 \
  --restore-sites input_answer,attention_output,mlp_input,mlp_output,block_output,final_norm_input \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase141_constrained_clean_restore \
  --hard-exit-after-model

python tests/gpt5/phase141_constrained_clean_restore_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase141_constrained_clean_restore_cuda.py`
- 汇总脚本：`tests/gpt5/phase141_constrained_clean_restore_summary.py`
- Qwen3 结果：`results/gpt5_phase141_constrained_clean_restore/phase141_qwen3_constrained_clean_restore.json`
- GLM4 结果：`results/gpt5_phase141_constrained_clean_restore/phase141_glm4_constrained_clean_restore.json`
- DS7B 结果：`results/gpt5_phase141_constrained_clean_restore/phase141_deepseek7b_constrained_clean_restore.json`
- 跨模型汇总：`results/gpt5_phase141_constrained_clean_restore/phase141_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant, time, clothing, furniture
templates = Phase135 six long templates
restore_sites = input_answer, attention_output, mlp_input, mlp_output, block_output, final_norm_input
restore_scales = 0.25, 0.5, 1.0, 1.5, 2.0
release_threshold = 0.25
```

### 客观结果

#### DS7B

```text
number:
  transfer R2 +0.49
  remove target Δ -1.34, release +0.15
  constrained clean count = 1
  best constrained = attention_output scale0.25
    target Δ -0.54
    release +0.00
    recovery +0.60
    top competitor vehicle -0.03
  best target = attention_output scale2.0
    target Δ +1.07
    release +1.15
    top competitor animal +1.15

container:
  transfer R2 +0.60
  remove target Δ -1.78, release +0.00
  constrained clean count = 1
  best constrained = input_answer scale1.0
    target Δ -0.09
    release +0.13
    recovery +0.95
    top competitor machine +0.13
  best target = mlp_input scale2.0
    target Δ +3.81
    release +2.64
    top competitor communication +2.64

plant:
  transfer R2 +0.52
  remove target Δ -1.84, release +0.13
  constrained clean count = 0
  min-release setting = attention_output scale0.25
    target Δ -0.95
    release +0.00
    recovery +0.48
  best target = attention_output scale2.0
    target Δ +0.67
    release +2.24
    top competitor tool +2.24

time:
  transfer R2 +0.60
  remove target Δ -2.07, release +0.75
  constrained clean count = 0
  min-release setting = mlp_input scale0.25
    target Δ -1.30
    release +0.00
    recovery +0.37
  best target = mlp_input scale2.0
    target Δ -0.76
    release +0.83
    top competitor clothing +0.83

clothing:
  transfer R2 +0.48
  remove target Δ +0.67, release +0.62
  constrained clean count = 0
  min-release setting = attention_output scale0.25
    target Δ -0.06
    release +0.00
    recovery -1.09

furniture:
  transfer R2 +0.52
  remove target Δ -0.25, release +0.00
  constrained clean count = 3
  best constrained = block_output scale0.25
    target Δ +0.10
    release +0.05
    recovery +1.40
    top competitor clothing +0.05
```

#### Qwen3

硬约束后 Qwen3 没有任何类别通过 clean restore：

```text
constrained clean count = 0 for all six categories
```

典型现象：

```text
best target restore 多在 mlp_input 高 scale；
target_delta 很大；
release 也极大；
top competitor 常为 sound/number/communication 等。
```

这支持 Phase140 判断：

```text
Qwen3 的 answer-site injection 当前主要是高增益扰动，不是干净机制恢复。
```

#### GLM4

GLM4 只有 plant 出现 constrained clean：

```text
plant:
  constrained clean count = 8
  best constrained = block_output scale1.0
    target Δ +0.03
    release +0.09
    recovery +1.24
```

但 GLM4 多数 remove effect 很弱或方向不稳定，因此仍作为低权重对照。

### 当前最可靠客观事实

1. **硬约束后，DS7B 真正 clean restore 只稳定出现在 number/container/furniture**

```text
number: attention_output scale0.25
container: input_answer scale1.0
furniture: block_output scale0.25
```

其中 container 最可靠：

```text
recovery +0.95
release +0.13
R2 +0.60
```

2. **plant 从“半干净”降为未通过硬约束**

plant 最接近通过的是：

```text
attention_output scale0.25:
  recovery +0.48
  release +0.00
```

只差 recovery 阈值 0.5，但严格说未通过。

3. **time 更细 site sweep 仍失败**

time 最低 release setting：

```text
mlp_input scale0.25:
  recovery +0.37
  release +0.00
```

最佳 target setting：

```text
mlp_input scale2.0:
  recovery +0.63
  release +0.83
```

这说明 time 不是简单 answer-site restore site 错误。

4. **高 scale best-target 多为 dirty restore**

DS7B：

```text
number best target release +1.15
container best target release +2.64
plant best target release +2.24
time best target release +0.83
```

所以 target 最大通常不可信。

5. **attention_output 成为重要 clean interface**

DS7B：

```text
number clean restore: attention_output scale0.25
plant min-release near-clean: attention_output scale0.25
```

这提示 support restore 的干净接口可能在 last attention output 附近，而不是 block_output 高 scale。

### 理论修正

Phase141 后，当前公式应更严格：

```text
A_c^L(a)
=
Φ_L(
  Z_support(c,T,s,γ),
  Z_suppress(c,T,s,γ),
  Z_format(T,s),
  Norm
)
```

其中：

```text
Z_support:
  当前 W·R_pre 能部分恢复的目标支持成分

Z_suppress:
  当前 W·R_pre 不稳定携带的竞争抑制成分

Z_format:
  next-token 格式和词元级读出约束
```

更谨慎结论：

```text
W·R_pre 是 support restore path；
clean restore 需要小 scale + 正确接口；
dirty restore 来自 support 过量注入且 suppressor/format 不匹配；
time 不是当前 answer-site support restore path 可解释的类别。
```

### 硬伤和瓶颈

1. **time 仍未解释**

即使增加 attention/MLP/final_norm 位点，time 也未通过 clean restore。

2. **plant 接近但未过阈值**

可能需要更细 scale：

```text
0.25, 0.3, 0.35, 0.4
```

或需要 support+suppressor 分解。

3. **actual generation 仍未完整测试**

本轮仍是 first-token argmax，不是 generate。

4. **template transfer 仍未测试**

本轮仍使用 long templates。

5. **clean restore 可能依赖类别专属 site**

不同类别最佳 clean site 不同，说明不能假设全局统一接口。

### 下一阶段任务

Phase142 应做：

```text
Support/Suppressor Split and Time Alternative Path
支持/抑制拆分与 time 替代路径
```

核心目标：

```text
1. 从 W·R_pre 中拆出 support 与 suppressor 成分。
2. 对 plant 测更细低 scale 和 attention_output 接口。
3. 对 time 回到 last-1 / last-2 layer，判断是否不是 true-last answer-site path。
```

测试设计：

```text
1. DS7B 为主。
2. 类别 number/container/plant/time。
3. 对 W·R_pre 方向做两种分解：
   target-aligned support component
   competitor-aligned suppressor component
4. 分别 restore：
   support only
   suppressor only
   support + suppressor
5. plant:
   attention_output scale = 0.25, 0.3, 0.35, 0.4, 0.5
6. time:
   last layer, last-1, last-2
   input_answer, attention_output, mlp_input
7. 记录 constrained clean restore、first-token argmax、top competitor。
```

成功判据：

```text
plant:
  找到通过硬约束的 clean setting。

time:
  如果 last-1/last-2 仍失败，说明 time 需要非 support-restore 路径；
  如果 earlier layer 成功，说明 time 的 restore site 在上游。

support/suppressor:
  如果 support only 提高 target 但释放 competitor，
  suppressor joint restore 降低 release，
  则 competition-conditioned theory 得到直接支持。
```

## Phase 142: Support/Suppressor Split and Time Alternative Path 支持抑制拆分与 time 替代路径 [2026-06-15 08:54]

### 本阶段目标

根据用户要求，先判断附加分析是否正确，再综合正确部分继续完成客观测试。

附加分析中正确部分：

```text
1. Phase141 的 hard clean-restore 判据是必要收紧。
2. DS7B number/container/plant/time 需要继续拆 support/suppressor。
3. time 不能只在 true-last layer 判断失败，必须测试 last-1/last-2 alternative path。
4. plant 接近 clean threshold，需要更细 low-scale sweep。
5. Qwen3 高 gain distortion 与 GLM4 弱效应不能作为主要闭合证据。
```

本轮目标：

```text
1. 判断 support-only 是否能形成 clean restore。
2. 判断 naive suppressor joint 是否能降低 competitor release。
3. 重新测试 time 是否存在 earlier-layer alternative path。
4. 用更大 DS7B 数据量验证 number/container/plant/time，避免小样本误判。
```

### 执行命令

```bash
python tests/gpt5/phase142_support_suppressor_timepath_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --rank 4 \
  --categories plant,time \
  --layer-offsets 0 \
  --restore-sites attention_output,mlp_input \
  --restore-scales 0.25,0.5 \
  --modes support,joint \
  --output-dir results/gpt5_phase142_smoke \
  --hard-exit-after-model

python tests/gpt5/phase142_support_suppressor_timepath_cuda.py qwen3 \
  --train-objects 10 \
  --test-objects 20 \
  --batch-size 20 \
  --rank 8 \
  --categories plant,time \
  --layer-offsets 0,-1 \
  --restore-sites attention_output,mlp_input \
  --restore-scales 0.25,0.3,0.35,0.4,0.5 \
  --modes support,joint \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase142_support_suppressor_timepath \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase142_support_suppressor_timepath_cuda.py glm4 \
  --train-objects 10 \
  --test-objects 20 \
  --batch-size 20 \
  --rank 8 \
  --categories plant,time \
  --layer-offsets 0,-1 \
  --restore-sites attention_output,mlp_input \
  --restore-scales 0.25,0.3,0.35,0.4,0.5 \
  --modes support,joint \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase142_support_suppressor_timepath \
  --hard-exit-after-model

python tests/gpt5/phase142_support_suppressor_timepath_cuda.py deepseek7b \
  --train-objects 20 \
  --test-objects 40 \
  --batch-size 20 \
  --rank 8 \
  --categories number,container,plant,time \
  --layer-offsets 0,-1,-2 \
  --restore-sites attention_output,mlp_input \
  --restore-scales 0.25,0.3,0.35,0.4,0.5 \
  --modes support,joint \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase142_support_suppressor_timepath \
  --hard-exit-after-model

python tests/gpt5/phase142_support_suppressor_timepath_summary.py

python -m py_compile \
  tests/gpt5/phase142_support_suppressor_timepath_cuda.py \
  tests/gpt5/phase142_support_suppressor_timepath_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase142_support_suppressor_timepath_cuda.py`
- 汇总脚本：`tests/gpt5/phase142_support_suppressor_timepath_summary.py`
- Qwen3 结果：`results/gpt5_phase142_support_suppressor_timepath/phase142_qwen3_support_suppressor_timepath.json`
- GLM4 结果：`results/gpt5_phase142_support_suppressor_timepath/phase142_glm4_support_suppressor_timepath.json`
- DS7B 结果：`results/gpt5_phase142_support_suppressor_timepath/phase142_deepseek7b_support_suppressor_timepath.json`
- 跨模型汇总：`results/gpt5_phase142_support_suppressor_timepath/phase142_cross_model_summary.md`

### 测试范围

```text
Qwen3:
  categories = plant,time
  train/test objects = 10/20
  layers = true-last, last-1

GLM4:
  categories = plant,time
  train/test objects = 10/20
  layers = true-last, last-1

DS7B:
  categories = number,container,plant,time
  train/test objects = 20/40
  layers = true-last, last-1, last-2

shared:
  restore sites = attention_output, mlp_input
  scales = 0.25, 0.3, 0.35, 0.4, 0.5
  modes = support, joint
  clean condition = target recovery high enough and competitor release <= 0.25
```

### 客观结果

#### Qwen3

```text
plant/time 在 L36/L35 均无 constrained clean restore。

plant L36:
  best support = mlp_input scale0.5, target +8.27, release +12.40
  best joint = attention_output scale0.25, target +1.05, release +1.40

time L36:
  best support = mlp_input scale0.5, target +9.97, release +12.34
  best joint = mlp_input scale0.4, target +1.79, release +1.91

time L35:
  removal target -3.05
  support mlp_input scale0.5 target -2.24, release +1.25, recovery +0.27
```

Qwen3 继续表现为 high-gain distortion：restore 方向可以大幅推高 target，但同时严重释放 competitor，不能算干净机制证据。

#### GLM4 bf16

```text
plant L40:
  clean count = 8
  best clean = joint attention_output scale0.5
  target +0.02, release +0.06, recovery +1.21

time L40:
  clean count = 0
  best support = mlp_input scale0.25
  target +0.02, release +0.48

plant L39/time L39:
  clean count = 0
```

GLM4 只有 plant L40 出现弱 clean restore，但 remove effect 本身较小，证据权重低。

#### DS7B

```text
number L28:
  clean count = 2
  best clean = support attention_output scale0.3
  target -0.43, release +0.12, recovery +0.68
  support scale0.5 target -0.05, release +0.61, recovery +0.97, dirty

container L28:
  clean count = 0
  best support = mlp_input scale0.5
  target +3.19, release +1.90, dirty
  best joint = attention_output scale0.25
  target -2.18, release +0.41, recovery negative

plant L28:
  clean count = 2
  best clean = support attention_output scale0.35
  target -0.69, release +0.11, recovery +0.62
  support scale0.5 target -0.37, release +0.62, dirty

time L28:
  clean count = 0
  best support = mlp_input scale0.25
  target -1.30, release +0.00, recovery +0.37

time L27:
  clean count = 3
  best clean = support mlp_input scale0.5
  target -0.18, release +0.19, recovery +0.92

time L26:
  clean count = 0
```

### 当前最可靠客观事实

1. **DS7B time 不是完全失败，而是路径位置错了**

```text
true-last L28: no clean restore
last-1 L27 mlp_input scale0.5: clean restore, recovery +0.92, release +0.19
last-2 L26: no clean restore
```

这说明 time 更像 last-1 MLP interface 上的可恢复路径，而不是 true-last attention/interface 上的路径。

2. **DS7B plant 从 near-clean 升级为 clean**

```text
Phase141 plant: near-clean
Phase142 plant L28 support attention_output scale0.35:
  target -0.69, release +0.11, recovery +0.62
```

plant 的可恢复窗口很窄，scale0.35 干净，scale0.5 变 dirty。

3. **DS7B number 仍是相对稳定 support restore 类别**

```text
number L28 support attention_output scale0.3:
  target -0.43, release +0.12, recovery +0.68
```

但 scale0.5 会释放 animal，说明 number 也不是无限线性通道。

4. **container 在本轮没有复现 Phase141 的 clean restore**

```text
Phase141 成功位点：input_answer
Phase142 测试位点：attention_output, mlp_input
Phase142 container clean count = 0
```

因此不能说 container 结论被推翻，只能说它不是 attention_output/mlp_input 上的 support/joint 可恢复路径；必须补测 input_answer。

5. **naive joint suppressor 没有得到支持**

本轮 joint 多数情况下没有降低 release，反而会：

```text
降低 target recovery
增加 competitor release
或造成 negative recovery
```

这不是 suppressor 不存在的证据，而是说明当前 suppressor 定义过粗，不能直接用预设 competitor basis 当作真实抑制成分。

### 对当前理论的修正

Phase142 支持以下更谨慎版本：

```text
类别编码不是单一正支持方向。
在 DS7B 中，部分类别存在低 scale support-only clean restore window。
不同类别的 clean restore site 不同：
  number/plant: true-last attention_output
  time: last-1 mlp_input
  container: 可能是 input_answer，需要复测
```

条件化关系因子动力学公式需要从：

```text
category factor -> last layer answer restore
```

修正为：

```text
conditioned relation factor =
  category support component
  + site-specific interface component
  + competitor-sensitive suppressor/control component
  + layer-position routing gate
```

其中 suppressor/control component 不能再用人工指定 competitor basis 直接替代，必须从真实 dirty-vs-clean contrast 或实际 released competitor 中反推出。

### 硬伤和瓶颈

1. **support/suppressor 拆分仍然粗糙**

本轮 support 主要来自 target-aligned component，joint 主要来自预设 competitor component。结果显示 naive joint 不可靠，说明 suppressor 需要重新定义。

2. **container 缺少 input_answer 复测**

Phase142 没有覆盖 Phase141 中 container 最强 clean site，因此 container 当前状态是 unresolved，不是 failed。

3. **clean restore 判据仍基于 first-token argmax/readout**

还没有完整 generate，也没有测试长输出稳定性。

4. **类别数量仍有限**

DS7B 本轮只扩展到 number/container/plant/time，不能把 clean window 直接推广到所有类别。

5. **scale window 很窄**

number/plant 低 scale 可 clean，高 scale 变 dirty，说明恢复不是单调线性注入，存在明显的非线性阈值或竞争释放。

### 下一阶段任务

Phase143 应做：

```text
Time Last-1 MLP Interface and Empirical Suppressor Redefinition
time 的 last-1 MLP 接口验证与经验抑制成分重定义
```

核心任务：

```text
1. 用更大 DS7B 数据复测 time L27 mlp_input clean restore。
2. 把 container 的 input_answer 位点加入 Phase142 框架，判断 Phase141 clean restore 是否可复现。
3. 从 dirty-vs-clean contrast 和实际 top released competitor 中学习 empirical suppressor direction。
4. 对比 support-only、naive-joint、empirical-joint。
5. 继续保持 hard clean criterion，不再只看 recovery 最大值。
```

建议测试范围：

```text
DS7B:
  categories = number,container,plant,time
  train/test objects = 30/60 or 40/80
  layers = L28,L27,L26
  sites = input_answer, attention_output, mlp_input
  scales = 0.2,0.25,0.3,0.35,0.4,0.45,0.5
  modes = support, naive_joint, empirical_joint

Qwen3/GLM4:
  只做 smaller confirmation，不作为主证据。
```

成功判据：

```text
time:
  L27 mlp_input 在更大数据上仍 clean，说明 time 的恢复路径是 earlier MLP interface。

container:
  input_answer clean 可复现，说明 container 是 answer-input interface 路径。

empirical suppressor:
  empirical_joint 比 support-only 更低 release，且不显著牺牲 target recovery，
  才能说明 suppressor/control component 被真正捕捉。
```

## Phase 143: Time Interface Empirical Suppressor time 接口验证与经验抑制重定义 [2026-06-15 09:28]

### 本阶段目标

根据用户要求，先分析附加判断是否正确，再继续完成客观测试。

附加分析基本正确：

```text
1. Phase142 已经证明不能只围绕 target support restore 推进。
2. number/plant 在 DS7B true-last attention_output 上有 low-scale clean restore window。
3. time 不是完全失败，而是有效恢复路径在 last-1 mlp_input。
4. naive suppressor 失败不能解释成 suppressor 不存在，只能说明人工固定 competitor basis 不可靠。
5. container 必须补测 input_answer，因为 Phase142 没覆盖 Phase141 的成功位点。
```

本轮 Phase143 目标：

```text
1. 复测 DS7B time L27 mlp_input 是否稳定 clean。
2. 把 input_answer 加入 container/number/plant/time 的恢复位点。
3. 对比 support、naive_joint、empirical_joint。
4. empirical_joint 使用 support-only 实际释放出的 top competitor 作为 suppressor source。
5. 继续使用 hard clean criterion，不只看 recovery 最大值。
```

### 执行命令

```bash
python tests/gpt5/phase143_time_interface_empirical_suppressor_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --rank 4 \
  --categories container,time \
  --layer-offsets 0 \
  --restore-sites input_answer,mlp_input \
  --restore-scales 0.25,0.5 \
  --modes support,naive_joint,empirical_joint \
  --output-dir results/gpt5_phase143_smoke \
  --hard-exit-after-model

python tests/gpt5/phase143_time_interface_empirical_suppressor_cuda.py qwen3 \
  --train-objects 10 \
  --test-objects 20 \
  --batch-size 20 \
  --rank 8 \
  --categories container,time \
  --layer-offsets 0,-1 \
  --restore-sites input_answer,mlp_input \
  --restore-scales 0.25,0.35,0.5 \
  --modes support,naive_joint,empirical_joint \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase143_time_interface_empirical_suppressor \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase143_time_interface_empirical_suppressor_cuda.py glm4 \
  --train-objects 10 \
  --test-objects 20 \
  --batch-size 20 \
  --rank 8 \
  --categories container,time \
  --layer-offsets 0,-1 \
  --restore-sites input_answer,mlp_input \
  --restore-scales 0.25,0.35,0.5 \
  --modes support,naive_joint,empirical_joint \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase143_time_interface_empirical_suppressor \
  --hard-exit-after-model

python tests/gpt5/phase143_time_interface_empirical_suppressor_cuda.py deepseek7b \
  --train-objects 30 \
  --test-objects 60 \
  --batch-size 20 \
  --rank 8 \
  --categories number,container,plant,time \
  --layer-offsets 0,-1,-2 \
  --restore-sites input_answer,attention_output,mlp_input \
  --restore-scales 0.2,0.25,0.3,0.35,0.4,0.45,0.5 \
  --modes support,naive_joint,empirical_joint \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase143_time_interface_empirical_suppressor \
  --hard-exit-after-model

# 30/60 超过部分类别可用 heldout 对象范围，实际主测试改为：
python tests/gpt5/phase143_time_interface_empirical_suppressor_cuda.py deepseek7b \
  --train-objects 20 \
  --test-objects 40 \
  --batch-size 20 \
  --rank 8 \
  --categories number,container,plant,time \
  --layer-offsets 0,-1,-2 \
  --restore-sites input_answer,attention_output,mlp_input \
  --restore-scales 0.2,0.25,0.3,0.35,0.4,0.45,0.5 \
  --modes support,naive_joint,empirical_joint \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase143_time_interface_empirical_suppressor \
  --hard-exit-after-model

python tests/gpt5/phase143_time_interface_empirical_suppressor_summary.py

python -m py_compile \
  tests/gpt5/phase143_time_interface_empirical_suppressor_cuda.py \
  tests/gpt5/phase143_time_interface_empirical_suppressor_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase143_time_interface_empirical_suppressor_cuda.py`
- 汇总脚本：`tests/gpt5/phase143_time_interface_empirical_suppressor_summary.py`
- Qwen3 结果：`results/gpt5_phase143_time_interface_empirical_suppressor/phase143_qwen3_time_interface_empirical_suppressor.json`
- GLM4 结果：`results/gpt5_phase143_time_interface_empirical_suppressor/phase143_glm4_time_interface_empirical_suppressor.json`
- DS7B 结果：`results/gpt5_phase143_time_interface_empirical_suppressor/phase143_deepseek7b_time_interface_empirical_suppressor.json`
- 跨模型汇总：`results/gpt5_phase143_time_interface_empirical_suppressor/phase143_cross_model_summary.md`

### 测试范围

```text
Qwen3:
  categories = container,time
  train/test objects = 10/20
  layers = true-last,last-1
  sites = input_answer,mlp_input

GLM4:
  categories = container,time
  train/test objects = 10/20
  layers = true-last,last-1
  sites = input_answer,mlp_input

DS7B:
  categories = number,container,plant,time
  train/test objects = 20/40
  layers = L28,L27,L26
  sites = input_answer,attention_output,mlp_input
  scales = 0.2,0.25,0.3,0.35,0.4,0.45,0.5
  modes = support,naive_joint,empirical_joint
```

### 客观结果

#### Qwen3

```text
container/time 均无 clean restore。

container L36:
  best support = mlp_input scale0.5
  target +7.22, release +11.91, dirty

time L36:
  best support = mlp_input scale0.5
  target +9.97, release +12.34, dirty

container L35:
  best empirical_joint input_answer scale0.5
  target -0.77, release +0.14, recovery +0.46, below clean threshold

time L35:
  best empirical_joint mlp_input scale0.5
  target -1.60, release +1.79, recovery +0.47, dirty
```

Qwen3 继续表现为高增益、强竞争释放，不能提供 clean restore 主证据。

#### GLM4 bf16

```text
container/time 均无 clean restore。

container L40:
  best support = mlp_input scale0.25
  target -0.30, release +0.26, recovery negative

time L40:
  best support = mlp_input scale0.25
  target +0.02, release +0.48, dirty

container/time 的 empirical_joint 经常带来更大 release。
```

GLM4 仍然不是本机制的强证据模型。

#### DS7B

```text
number L28:
  clean count = 4
  best clean = support attention_output scale0.3
  target -0.43, release +0.12, recovery +0.68

plant L28:
  clean count = 2
  best clean = support attention_output scale0.35
  target -0.69, release +0.11, recovery +0.62

time L27:
  clean count = 4
  best clean = support mlp_input scale0.5
  target -0.18, release +0.19, recovery +0.92

time L28:
  clean count = 1
  best clean = support mlp_input scale0.2
  target -0.98, release +0.06, recovery +0.53

container L28:
  clean count = 0
  best input_answer support:
    target -1.13, release +0.00, recovery +0.36
  best support overall:
    mlp_input scale0.5 target +3.19, release +1.90, dirty

container L27/L26:
  clean count = 0
```

### 当前最可靠客观事实

1. **time L27 mlp_input clean restore 被复现**

```text
Phase142:
  time L27 support mlp_input scale0.5
  recovery +0.92, release +0.19

Phase143:
  time L27 support mlp_input scale0.5
  recovery +0.92, release +0.19
```

这已经是当前最稳定的 time 路径证据。

2. **time true-last L28 不是绝对失败，但窗口更弱**

```text
time L28 support mlp_input scale0.2:
  recovery +0.53, release +0.06
```

Phase142 没有测 scale0.2，所以漏掉了这个弱 clean window。更强、更稳定的仍然是 L27 mlp_input。

3. **number/plant true-last attention_output 复现**

```text
number L28 attention_output scale0.3:
  recovery +0.68, release +0.12

plant L28 attention_output scale0.35:
  recovery +0.62, release +0.11
```

这说明 number/plant 的 clean window 不是偶然。

4. **container input_answer 没有稳定复现 Phase141**

```text
container L28 input_answer best support:
  recovery +0.36, release +0.00
```

这说明 Phase141 的 container clean restore 可能依赖：

```text
不同 restore 构造
不同数据切分
scale1.0 而非本轮最高0.5
或小样本偶然性
```

当前不能继续把 container 当作已闭合 clean path。

5. **empirical_joint 第一版没有成功**

本轮 empirical_joint 使用 support-only 实际释放出的 top competitor 选择 suppressor basis，但多数结果：

```text
没有提升 recovery
没有降低 release
常造成 target recovery 下降或 negative recovery
```

因此“按实际释放类选择类别基底”仍不等于真实 suppressor。

### 对当前理论的修正

当前最谨慎公式应写成：

```text
conditioned relation factor =
  category support channel
  + layer-site routing gate
  + scale-bounded clean window
  + unresolved competition-control component
```

已经较可靠的映射：

```text
number:
  DS7B L28 attention_output low-scale support channel

plant:
  DS7B L28 attention_output low-scale support channel

time:
  DS7B L27 mlp_input strong support channel
  DS7B L28 mlp_input weak support window

container:
  unresolved
```

这里还不能把 suppressor/control 写成已被捕捉的机制项，只能说它客观存在为 release/dirty 现象，但方向没有被正确分离。

### 硬伤和瓶颈

1. **DS7B 30/60 不能执行**

原因是部分类别 heldout 对象不足，导致测试集合为空。本轮主测试改为 20/40。后续如果要更大数据，必须先扩充 CATEGORY_OBJECTS。

2. **container 结论被削弱**

Phase143 补测 input_answer 后仍未 clean，因此 Phase141 的 container 需要重新审计，不能继续作为稳定通道。

3. **empirical suppressor 仍然不是机制 suppressor**

用 top released competitor 的类别基底做 suppressor 仍然失败，说明真实抑制方向可能不是类别方向，而是更局部的 dirty-clean 差分、logit margin 方向或 normalization gate。

4. **clean window 非常窄**

number/plant/time 都存在 scale-sensitive 现象，说明不能只用线性强注入解释。

5. **仍未做完整生成**

当前结果仍是 first-token/readout 层面的因果现象，还没有证明多 token 生成稳定。

### 下一阶段任务

Phase144 应做：

```text
Dirty-Clean Contrast Suppressor and Container Re-audit
脏恢复-干净恢复对比抑制方向与 container 重审计
```

核心目标：

```text
1. 不再用类别 basis 做 suppressor。
2. 从同一 category/site/layer 下的 dirty restore 与 clean restore 直接取差分方向。
3. 对 number/plant/time 的 clean-vs-dirty scale pair 做 contrast。
4. 对 container 单独重审 Phase141 条件：scale 扩到 1.0/1.5，恢复构造与 Phase141 对齐。
5. 判断 suppressor 是否是局部差分方向，而不是类别级方向。
```

建议测试范围：

```text
DS7B main:
  categories = number,plant,time,container
  train/test objects = 20/40
  number/plant:
    layer = L28
    site = attention_output
    clean scale = 0.3/0.35
    dirty scale = 0.5
  time:
    layer = L27
    site = mlp_input
    clean scale = 0.5
    dirty candidate = higher or alternative-site restore
  container:
    layers = L28,L27
    sites = input_answer,attention_output,mlp_input
    scales = 0.25,0.5,0.75,1.0,1.25,1.5

Qwen3/GLM4:
  optional confirmation only。
```

成功判据：

```text
dirty-clean contrast suppressor:
  joint restore 降低 release，
  target recovery 不显著下降，
  clean count 高于 support-only。

container:
  如果 scale1.0/1.5 仍不能复现 clean restore，
  则 Phase141 container 应降级为 unstable finding。
```

## Phase 144: Dirty-Clean Contrast Suppressor and Container Re-audit 脏干净对比抑制与 container 重审计 [2026-06-15 09:55]

### 本阶段目标

根据用户要求，先分析附加判断是否正确，再综合当前进展继续完成客观测试。

附加分析中正确部分：

```text
1. Phase143 的核心进展是 layer-site routing，而不是 empirical_joint 成功。
2. time 在 DS7B L27 mlp_input 的 clean restore 已经是稳定路径证据。
3. number/plant 在 DS7B L28 attention_output 上有 low-scale clean restore window。
4. container 应降级为 unresolved，必须按 Phase141 条件补测 scale1.0/1.5。
5. suppressor 不能继续用 category basis，需要 dirty-clean contrast。
```

本轮目标：

```text
1. 不再用类别基底作为 suppressor。
2. 从 dirty support restore 与 clean support restore 的 answer state 差分中构造 contrast suppressor。
3. 对 container 扩展 scale 到 1.5，复查 input_answer 是否能复现 clean。
4. 对 number/plant/time/container 保持同一轮 DS7B 主测试，不拆分批次。
```

### 执行命令

```bash
python tests/gpt5/phase144_dirty_clean_contrast_container_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --rank 4 \
  --categories container,time \
  --layer-offsets 0 \
  --restore-sites input_answer,mlp_input \
  --restore-scales 0.25,0.5 \
  --contrast-suppress-scales 0.5 \
  --output-dir results/gpt5_phase144_smoke \
  --hard-exit-after-model

python tests/gpt5/phase144_dirty_clean_contrast_container_cuda.py qwen3 \
  --train-objects 10 \
  --test-objects 20 \
  --batch-size 20 \
  --rank 8 \
  --categories container,time \
  --layer-offsets 0,-1 \
  --restore-sites input_answer,mlp_input \
  --restore-scales 0.25,0.5,1.0 \
  --contrast-suppress-scales 0.25,0.5,1.0 \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase144_dirty_clean_contrast_container \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase144_dirty_clean_contrast_container_cuda.py glm4 \
  --train-objects 10 \
  --test-objects 20 \
  --batch-size 20 \
  --rank 8 \
  --categories container,time \
  --layer-offsets 0,-1 \
  --restore-sites input_answer,mlp_input \
  --restore-scales 0.25,0.5,1.0 \
  --contrast-suppress-scales 0.25,0.5,1.0 \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase144_dirty_clean_contrast_container \
  --hard-exit-after-model

python tests/gpt5/phase144_dirty_clean_contrast_container_cuda.py deepseek7b \
  --train-objects 20 \
  --test-objects 40 \
  --batch-size 20 \
  --rank 8 \
  --categories number,plant,time,container \
  --layer-offsets 0,-1,-2 \
  --restore-sites input_answer,attention_output,mlp_input \
  --restore-scales 0.25,0.5,0.75,1.0,1.25,1.5 \
  --contrast-suppress-scales 0.25,0.5,1.0 \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase144_dirty_clean_contrast_container \
  --hard-exit-after-model

python tests/gpt5/phase144_dirty_clean_contrast_container_summary.py

python -m py_compile \
  tests/gpt5/phase144_dirty_clean_contrast_container_cuda.py \
  tests/gpt5/phase144_dirty_clean_contrast_container_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase144_dirty_clean_contrast_container_cuda.py`
- 汇总脚本：`tests/gpt5/phase144_dirty_clean_contrast_container_summary.py`
- Qwen3 结果：`results/gpt5_phase144_dirty_clean_contrast_container/phase144_qwen3_dirty_clean_contrast_container.json`
- GLM4 结果：`results/gpt5_phase144_dirty_clean_contrast_container/phase144_glm4_dirty_clean_contrast_container.json`
- DS7B 结果：`results/gpt5_phase144_dirty_clean_contrast_container/phase144_deepseek7b_dirty_clean_contrast_container.json`
- 跨模型汇总：`results/gpt5_phase144_dirty_clean_contrast_container/phase144_cross_model_summary.md`

### 测试范围

```text
Qwen3/GLM4:
  categories = container,time
  train/test objects = 10/20
  layers = true-last,last-1
  sites = input_answer,mlp_input
  scales = 0.25,0.5,1.0

DS7B:
  categories = number,plant,time,container
  train/test objects = 20/40
  layers = L28,L27,L26
  sites = input_answer,attention_output,mlp_input
  support scales = 0.25,0.5,0.75,1.0,1.25,1.5
  contrast suppress scales = 0.25,0.5,1.0
```

### 方法说明

```text
support-only:
  移除 pre-answer target basis 后，在指定 site 恢复 W·R_pre。

dirty-clean contrast:
  在同一 category/layer/site 下，
  找 clean support candidate 与 dirty support candidate，
  取 dirty answer state mean - clean answer state mean，
  作为 contrast suppressor basis。

contrast_joint:
  用 dirty scale 做 support restore，
  同时减去 dirty-clean contrast basis 投影。
```

### 客观结果

#### Qwen3

```text
container/time 均无 clean restore。

container L36:
  best support = mlp_input scale0.5
  target +7.22, release +11.91, dirty

time L36:
  best support = mlp_input scale1.0
  target +10.72, release +12.95, dirty

time L35:
  best support = mlp_input scale1.0
  recovery +0.92, release +1.03, dirty
```

Qwen3 仍是强 release 模型，本轮 contrast 没能降低 release。

#### GLM4 bf16

```text
container/time 均无 clean restore。

container L40:
  best support = mlp_input scale0.25
  target -0.30, release +0.26, recovery negative

time L40:
  best support = mlp_input scale0.25
  target +0.02, release +0.48, dirty
```

GLM4 仍未形成强机制证据。

#### DS7B

```text
number L28:
  clean count = 1
  best clean = support attention_output scale0.25
  target -0.54, release +0.00, recovery +0.60
  dirty support attention_output scale1.5:
    target +0.88, release +1.16, recovery +1.65
  contrast_joint attention_output scale1.5:
    target +0.26, release +0.68, recovery +1.19

plant L28:
  clean count = 2
  clean support input_answer scale0.75:
    target -0.89, release +0.13, recovery +0.52
  clean contrast_joint attention_output scale1.5:
    target -0.85, release +0.00, recovery +0.54
  dirty support attention_output scale1.5:
    target +0.53, release +2.06

time L27:
  clean count = 1
  support mlp_input scale0.5:
    target -0.18, release +0.19, recovery +0.92
  dirty support mlp_input scale1.5:
    target +1.63, release +2.06
  contrast_joint mlp_input scale1.5:
    target +2.05, release +2.51, worse

container L28:
  clean count = 3
  support input_answer scale0.75:
    target -0.62, release +0.00, recovery +0.65
  support input_answer scale1.0:
    target -0.09, release +0.13, recovery +0.95
  contrast_joint attention_output scale1.5:
    target -0.75, release +0.19, recovery +0.58

container L27/L26:
  no clean restore
```

### 当前最可靠客观事实

1. **container Phase141 结果被恢复**

Phase143 因为最高 scale 只有 0.5，没有复现 container。Phase144 把 scale 扩到 1.0 后：

```text
container L28 input_answer scale1.0:
  recovery +0.95, release +0.13
```

这说明 container 不是不稳定失败，而是依赖更高 scale 的 input_answer interface。

2. **time L27 mlp_input 再次复现**

```text
time L27 support mlp_input scale0.5:
  recovery +0.92, release +0.19
```

time 的主路径仍稳定。

3. **number/plant L28 support clean window 仍存在，但最佳 scale 有漂移**

```text
number:
  Phase143 best clean scale0.3
  Phase144 clean scale0.25

plant:
  Phase143 attention_output scale0.35 clean
  Phase144 input_answer scale0.75 clean
  Phase144 contrast_joint attention_output scale1.5 clean
```

这说明 clean window 不是单点常数，而是与测试尺度集合、site、contrast 构造有关。

4. **dirty-clean contrast 只有局部成功**

最有意义的正例：

```text
plant L28 attention_output:
  dirty support scale1.5 release +2.06
  contrast_joint scale1.5 release +0.00
  recovery +0.54
```

但 number/time/container 的 contrast 没有稳定成功，因此不能说 suppressor 机制已闭合。

5. **高 scale support 常把 target 推过头并释放 competitor**

例如：

```text
number L28 attention_output scale1.5:
  target +0.88, release +1.16

time L27 mlp_input scale1.5:
  target +1.63, release +2.06

container L28 mlp_input scale1.5:
  target +3.78, release +2.59
```

这进一步支持 scale-bounded clean window。

### 理论进展

当前理论应从 Phase143 的版本推进为：

```text
conditioned relation factor =
  category support channel
  + layer-site routing gate
  + scale-bounded clean window
  + interface-specific gain requirement
  + partially observed dirty-clean control direction
```

较可靠映射：

```text
number:
  DS7B L28 attention_output low-scale support。

plant:
  DS7B L28 support clean window；
  dirty-clean contrast 在 attention_output 上有局部成功。

time:
  DS7B L27 mlp_input stable support。

container:
  DS7B L28 input_answer high-scale support。
```

最重要修正：

```text
container 与 number/plant/time 不同：
  它不是 low-scale attention_output 路径，
  而是 high-scale input_answer interface 路径。
```

### 硬伤和瓶颈

1. **contrast suppressor 只有单点成功**

plant 的 dirty-clean contrast 成功，但 number/time/container 没有普遍成功。当前 contrast 方向仍然太粗，可能需要按 token、样本或 competitor 分组，而不是全样本均值差。

2. **clean window 受 scale grid 影响**

Phase143 漏掉 container，因为 scale 只到 0.5；Phase144 找回 container，因为扩到 1.0。以后重要结论必须覆盖足够 scale 范围。

3. **container 依赖高 scale**

高 scale 容易造成 dirty restore，因此 container 的 clean 可能更脆弱，需要在更多模板和对象上验证。

4. **仍是 first-token/readout 结果**

还没有完整 generation closure。

5. **Qwen3/GLM4 仍未闭合**

跨模型共性目前主要体现在现象类型，不体现在同样的 clean path。

### 下一阶段任务

Phase145 应做：

```text
Mechanism Stability Matrix and Generation Closure
机制稳定性矩阵与生成闭合
```

核心目标：

```text
1. 把 number/plant/time/container 四类的最佳 clean path 固定下来。
2. 不再只找最大 recovery，而是建立稳定性矩阵：
   category × layer × site × scale × template family × object split。
3. 对每个类别至少保留：
   best clean path
   nearest dirty path
   contrast path
4. 对最稳定路径做 small generation closure。
5. 明确哪些机制是稳定规律，哪些只是单轮窗口。
```

建议 DS7B 主测试：

```text
categories = number,plant,time,container
train/test objects = 20/40
template families = long, short, neutral
paths:
  number: L28 attention_output scale0.25/0.3
  plant: L28 attention_output scale0.35 and input_answer scale0.75
  time: L27 mlp_input scale0.5
  container: L28 input_answer scale0.75/1.0
dirty controls:
  number: L28 attention_output scale1.5
  plant: L28 attention_output scale1.5
  time: L27 mlp_input scale1.5
  container: L28 mlp_input scale1.5
```

成功判据：

```text
稳定性：
  clean path 在 template/object 扩展后仍满足 recovery >= 0.5, release <= 0.25。

生成闭合：
  patched generation 的 first generated token 或短输出类别偏向与 readout restore 同步。

理论推进：
  如果四类最佳 path 稳定，
  则 layer-site-scale routing 可以作为语言编码机制的第一张结构图。
```

## Phase 145: Mechanism Stability Matrix and Token Closure 机制稳定性矩阵与词元闭合审计 [2026-06-15 10:18]

### 本阶段目标

根据用户要求，先分析附加判断是否正确，再继续完成客观测试。

附加分析中正确部分：

```text
1. Phase144 正确恢复了 container，并进一步确认 time L27 mlp_input。
2. number/plant/time/container 的候选路径已经形成 layer-site-scale routing 雏形。
3. dirty-clean contrast 只有 plant 局部成功，suppressor 尚未闭合。
4. 下一步不能继续只看单点最大值，必须做 template/object stability matrix。
5. token-level closure 仍然缺失，需要补上类别读出和首词元审计。
```

本轮 Phase145 目标：

```text
1. 固定 Phase144 得到的候选 clean path 和 dirty path。
2. 扩展模板类型：
   long, short, neutral。
3. 扩展对象切分：
   front_back, back_front。
4. 用 clean_rate 评估路径稳定性，而不是只看 best case。
5. 记录 category_argmax_rate 作为轻量 token/readout closure 指标。
```

### 执行命令

```bash
python tests/gpt5/phase145_mechanism_stability_generation_cuda.py qwen3 \
  --train-objects 4 \
  --test-objects 4 \
  --batch-size 8 \
  --rank 4 \
  --categories number,time \
  --template-families long,short \
  --splits front_back \
  --output-dir results/gpt5_phase145_smoke \
  --hard-exit-after-model

python tests/gpt5/phase145_mechanism_stability_generation_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --rank 8 \
  --categories number,plant,time,container \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase145_mechanism_stability_generation \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase145_mechanism_stability_generation_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --rank 8 \
  --categories number,plant,time,container \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase145_mechanism_stability_generation \
  --hard-exit-after-model

python tests/gpt5/phase145_mechanism_stability_generation_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --rank 8 \
  --categories number,plant,time,container \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase145_mechanism_stability_generation \
  --hard-exit-after-model

python tests/gpt5/phase145_mechanism_stability_generation_summary.py

python -m py_compile \
  tests/gpt5/phase145_mechanism_stability_generation_cuda.py \
  tests/gpt5/phase145_mechanism_stability_generation_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase145_mechanism_stability_generation_cuda.py`
- 汇总脚本：`tests/gpt5/phase145_mechanism_stability_generation_summary.py`
- Qwen3 结果：`results/gpt5_phase145_mechanism_stability_generation/phase145_qwen3_mechanism_stability_generation.json`
- GLM4 结果：`results/gpt5_phase145_mechanism_stability_generation/phase145_glm4_mechanism_stability_generation.json`
- DS7B 结果：`results/gpt5_phase145_mechanism_stability_generation/phase145_deepseek7b_mechanism_stability_generation.json`
- 跨模型汇总：`results/gpt5_phase145_mechanism_stability_generation/phase145_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, plant, time, container
train/test objects = 12/12
object splits = front_back, back_front
template families = long, short, neutral
paths/category = 2 clean candidates + 1 dirty control
```

候选路径：

```text
number:
  clean_a = L28/Ltrue attention_output scale0.25
  clean_b = L28/Ltrue attention_output scale0.30
  dirty = L28/Ltrue attention_output scale1.50

plant:
  clean_attn = L28/Ltrue attention_output scale0.35
  clean_input = L28/Ltrue input_answer scale0.75
  dirty = L28/Ltrue attention_output scale1.50

time:
  clean_mlp = L27/last-1 mlp_input scale0.50
  weak_last = L28/Ltrue mlp_input scale0.20
  dirty = L27/last-1 mlp_input scale1.50

container:
  clean_input_a = L28/Ltrue input_answer scale0.75
  clean_input_b = L28/Ltrue input_answer scale1.00
  dirty = L28/Ltrue mlp_input scale1.50
```

### 客观结果

#### Qwen3

```text
所有主要 clean path 均未形成稳定 clean。
container clean_input_a clean_rate = 0.17，但 mean_release +2.89。
number/plant/time clean_rate = 0。
category_argmax_rate 全部为 0。
```

Qwen3 继续表现为高增益和不稳定竞争释放，不能作为当前机制闭合证据。

#### GLM4 bf16

```text
plant clean_attn clean_rate = 0.17。
其他路径 clean_rate 基本为 0。
container/time/number 没有稳定 clean。
category_argmax_rate 全部为 0。
```

GLM4 仍没有形成稳定路径闭合。

#### DS7B

跨模板族和对象切分后，Phase144 的单点路径明显降稳：

```text
container:
  clean_input_a clean_rate = 0.33
  clean_input_b clean_rate = 0.17
  dirty clean_rate = 0

number:
  clean_a clean_rate = 0.17
  clean_b clean_rate = 0.17
  dirty clean_rate = 0.17

plant:
  clean_attn clean_rate = 0.17
  clean_input clean_rate = 0.50
  dirty clean_rate = 0

time:
  clean_mlp clean_rate = 0
  weak_last clean_rate = 0
  dirty clean_rate = 0
```

分模板族结果：

```text
long family:
  number clean_rate = 0.50
  plant clean_rate = 0.25
  time clean_rate = 0
  container clean_rate = 0

short family:
  plant clean_input 有 1 个 clean
  container clean_input_a 有 1 个 clean
  number/time 不稳定

neutral family:
  plant clean_input clean_rate = 0.50
  container clean_input clean_rate = 0.50
  number/time 不稳定
```

### 当前最可靠客观事实

1. **Phase144 的路径不是跨模板/切分稳定机制**

单轮长模板中的 clean path，在 Phase145 扩展后明显降稳。尤其：

```text
time L27 mlp_input:
  Phase142/143/144 长模板稳定
  Phase145 template/split 扩展 clean_rate = 0
```

time 的路径依然是长模板局部强现象，不能说已经跨模板稳定。

2. **plant input_answer 是本轮最稳定路径**

```text
DS7B plant clean_input:
  clean_rate = 0.50
  mean recovery +0.63
  mean release +0.18
```

这是四类里最接近稳定机制的路径。

3. **container 仍有效但更脆弱**

```text
container clean_input_a clean_rate = 0.33
container clean_input_b clean_rate = 0.17
```

container 不是失败，但它不像 Phase144 单点结果那样稳定。

4. **number 只在 long/back_front 中较好**

```text
number long family clean_rate = 0.50
overall clean_rate = 0.17
```

number clean path 明显受模板族和对象切分影响。

5. **dirty controls 没有完全干净隔离**

number dirty path 也出现 clean_rate 0.17，说明当前 dirty control 不总是 dirty。dirty/clean 边界仍受模板和切分影响。

6. **category_argmax_rate 基本为 0**

当前 patch 能改变 readout score，但没有让类别读出成为 argmax。这说明：

```text
internal support restore != token-level output closure
```

### 对当前理论的修正

Phase145 对 Phase144 理论做了重要收紧。

之前可写成：

```text
category -> layer-site-scale path
```

现在必须写成：

```text
category + template family + object split
  -> layer-site-scale path
```

更谨慎版本：

```text
conditioned relation factor =
  context-field support channel
  + category-specific routing prior
  + template-conditioned routing shift
  + object-split sensitivity
  + scale-bounded clean window
  + unresolved token-level selection gate
```

当前最可靠说法：

```text
模型内部确实存在可因果恢复的支持通道；
这些通道在单一长模板条件下较清楚；
但跨模板和对象切分后，路径稳定性显著下降；
因此 layer-site-scale routing 还不能直接视为稳定语言编码结构图，
只能视为候选机制图。
```

### 硬伤和瓶颈

1. **token-level closure 未通过**

category_argmax_rate 基本为 0，说明恢复还没有接管最终 token selection。

2. **路径高度模板依赖**

long/short/neutral 的结果差异很大，说明模板本身是路由变量，不是噪声。

3. **对象切分敏感**

front_back 与 back_front 的差异说明对象集合会改变中心、basis 和 transfer。

4. **dirty path 定义仍不稳定**

dirty path 有时也会 clean，说明 dirty 不是固定 scale 即可定义。

5. **本轮 generation closure 仍是轻量版本**

记录了 category_argmax_rate 和 first-token ids，但还未做 autoregressive multi-token generate。

### 下一阶段任务

Phase146 应做：

```text
Template-Conditioned Router and Token Selection Gap
模板条件化路由与词元选择缺口
```

核心目标：

```text
1. 不再假设统一 path。
2. 对每个 template family 单独学习最佳 layer/site/scale。
3. 判断路径不稳定来自：
   layer/site 错位，
   scale 错位，
   还是 token selection gate 未打开。
4. 对稳定性最好的 plant input_answer 做完整 token-level audit。
5. 将 readout restore 与 final lm_head token logits 的差距拆出来。
```

建议测试范围：

```text
DS7B main:
  categories = number,plant,time,container
  template families = long,short,neutral
  objects = 12/12 with two splits
  per family sweep:
    layers = true-last,last-1
    sites = input_answer,attention_output,mlp_input
    scales = 0.2,0.25,0.3,0.35,0.5,0.75,1.0,1.25,1.5

重点输出：
  best path per category per template family
  readout recovery
  category_argmax_rate
  target token rank
  top released token/category
```

成功判据：

```text
如果 per-template best path 显著提高 clean_rate，
说明 Phase145 的不稳定主要来自 template-conditioned routing shift。

如果 readout recovery 高但 token rank 不提升，
说明 token selection gate 是独立瓶颈。
```

## Phase 146: Template-Conditioned Router and Token Selection Gap 模板条件化路由与词元选择缺口 [2026-06-15 10:58]

### 本阶段目标

根据用户要求，先分析附加内容是否正确，再综合当前进展继续测试。

附加分析基本正确：

```text
1. Phase145 的关键价值是把单点成功降级为模板/对象条件化候选机制。
2. template family 不是噪声，而是 routing condition。
3. category_argmax_rate 为 0 是重大负结果，说明 token selection gate 没闭合。
4. Phase146 应按 template family 单独搜索 layer/site/scale。
5. 应输出 target token rank、target token delta、top tokens，而不是只看内部 readout score。
```

本轮 Phase146 目标：

```text
1. 对每个 template family 单独搜索最佳路径。
2. 搜索范围覆盖 true-last/last-1、input_answer/attention_output/mlp_input、多 scale。
3. 判断 Phase145 的不稳定是否来自 template-conditioned routing shift。
4. 同时记录 target token rank、target token argmax、target token delta。
```

### 执行命令

```bash
python tests/gpt5/phase146_template_router_token_gap_cuda.py qwen3 \
  --train-objects 4 \
  --test-objects 4 \
  --batch-size 8 \
  --rank 4 \
  --categories number,time \
  --template-families long \
  --splits front_back \
  --layer-offsets 0,-1 \
  --sites input_answer,mlp_input \
  --scales 0.25,0.5 \
  --output-dir results/gpt5_phase146_smoke \
  --hard-exit-after-model

python tests/gpt5/phase146_template_router_token_gap_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --rank 8 \
  --categories plant,time \
  --template-families long,neutral \
  --splits front_back \
  --layer-offsets 0,-1 \
  --sites input_answer,attention_output,mlp_input \
  --scales 0.25,0.5,1.0 \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase146_template_router_token_gap \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase146_template_router_token_gap_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --rank 8 \
  --categories plant,time \
  --template-families long,neutral \
  --splits front_back \
  --layer-offsets 0,-1 \
  --sites input_answer,attention_output,mlp_input \
  --scales 0.25,0.5,1.0 \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase146_template_router_token_gap \
  --hard-exit-after-model

python tests/gpt5/phase146_template_router_token_gap_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --rank 8 \
  --categories number,plant,time,container \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --layer-offsets 0,-1 \
  --sites input_answer,attention_output,mlp_input \
  --scales 0.2,0.25,0.3,0.35,0.5,0.75,1.0,1.25,1.5 \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase146_template_router_token_gap \
  --hard-exit-after-model

python tests/gpt5/phase146_template_router_token_gap_summary.py

python -m py_compile \
  tests/gpt5/phase146_template_router_token_gap_cuda.py \
  tests/gpt5/phase146_template_router_token_gap_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase146_template_router_token_gap_cuda.py`
- 汇总脚本：`tests/gpt5/phase146_template_router_token_gap_summary.py`
- Qwen3 结果：`results/gpt5_phase146_template_router_token_gap/phase146_qwen3_template_router_token_gap.json`
- GLM4 结果：`results/gpt5_phase146_template_router_token_gap/phase146_glm4_template_router_token_gap.json`
- DS7B 结果：`results/gpt5_phase146_template_router_token_gap/phase146_deepseek7b_template_router_token_gap.json`
- 跨模型汇总：`results/gpt5_phase146_template_router_token_gap/phase146_cross_model_summary.md`

### 测试范围

```text
Qwen3/GLM4 confirmation:
  categories = plant,time
  template families = long,neutral
  split = front_back
  layers = true-last,last-1
  sites = input_answer,attention_output,mlp_input
  scales = 0.25,0.5,1.0

DS7B main:
  categories = number,plant,time,container
  template families = long,short,neutral
  splits = front_back,back_front
  layers = true-last,last-1
  sites = input_answer,attention_output,mlp_input
  scales = 0.2,0.25,0.3,0.35,0.5,0.75,1.0,1.25,1.5
```

### 客观结果

#### Qwen3

```text
plant:
  clean_rate = 0
  mean_release +12.52
  mean_token_rank 44062.7
  token_argmax 0

time:
  clean_rate = 0
  mean_release +12.81
  mean_token_rank 73654.5
  token_argmax 0
```

Qwen3 继续是高增益释放，不形成 clean 机制证据。

#### GLM4 bf16

```text
plant:
  clean_rate = 1.00
  mean_release +0.22
  mean_token_rank 1210.8
  token_argmax 0

time:
  clean_rate = 0
  mean_release +1.40
  mean_token_rank 9304.6
  token_argmax 0
```

GLM4 plant 出现 readout-clean，但 token argmax 仍为 0。

#### DS7B

模板条件化搜索显著提高 readout clean：

```text
container:
  clean_rate = 1.00
  mean recovery +1.07
  mean release +0.16
  mean token rank 19227.4
  token_argmax 0

plant:
  clean_rate = 1.00
  mean recovery +0.66
  mean release +0.03
  mean token rank 3178.8
  token_argmax 0

time:
  clean_rate = 1.00
  mean recovery +0.63
  mean release +0.10
  mean token rank 17927.9
  token_argmax 0

number:
  clean_rate = 0.50
  mean recovery +0.84
  mean release +1.04
  mean token rank 9589.2
  token_argmax 0
```

DS7B 最佳路径不再是固定全局路径，而是 template/split 条件化路径：

```text
plant:
  front_back long: L28 attention_output scale0.35
  front_back short: L28 input_answer scale1.0
  front_back neutral: L28 input_answer scale0.5
  back_front long: L28 attention_output scale0.5
  back_front short: L28 attention_output scale0.75
  back_front neutral: L28 input_answer scale1.0

time:
  front_back long: L28 attention_output scale0.75
  front_back short: L27 attention_output scale0.5
  front_back neutral: L28 input_answer scale0.75
  back_front long: L28 attention_output scale0.25
  back_front short: L28 attention_output scale0.5
  back_front neutral: L28 mlp_input scale1.5

container:
  front_back long: L28 mlp_input scale0.35
  front_back short: L28 input_answer scale1.0
  front_back neutral: L28 attention_output scale0.75
  back_front long: L28 input_answer scale0.35
  back_front short: L28 attention_output scale0.75
  back_front neutral: L28 attention_output scale0.75
```

### 当前最可靠客观事实

1. **Phase145 的不稳定主要来自模板条件化路由错位**

固定全局 path 时：

```text
plant clean_rate = 0.50
time clean_rate = 0
container clean_rate = 0.33/0.17
```

per-template sweep 后：

```text
plant clean_rate = 1.00
time clean_rate = 1.00
container clean_rate = 1.00
```

这说明 layer/site/scale 不是全局常数，而是 template-conditioned routing 的输出。

2. **number 仍没有闭合**

number clean_rate 只有 0.50，且 mean_release +1.04。它常选择高 scale attention_output scale1.5，容易释放 competitor。因此 number 当前比 plant/time/container 更不稳定。

3. **token selection gap 被直接确认**

虽然 DS7B readout clean_rate 大幅提高：

```text
plant/time/container = 1.00
```

但：

```text
token_argmax = 0
target token rank 仍在几千到几万
```

例如：

```text
plant mean token rank = 3178.8
time mean token rank = 17927.9
container mean token rank = 19227.4
```

这证明：

```text
internal readout restore != final token selection
```

4. **目标词元 logit 经常没有同步上升**

DS7B 目标词元 delta：

```text
number mean +0.316
plant mean -0.932
time mean -1.629
container mean -1.044
```

readout clean 并不保证目标词元 logit 上升。很多 top token 是空格、介词或格式 token。

5. **time 的长模板 L27 mlp_input 不是一般最优路径**

Phase142-144 的 L27 mlp_input 是长模板局部强路径。Phase146 中，time 的最佳路径随模板变化，多数转向 L28 attention_output 或 input_answer。

### 对当前理论的修正

Phase146 后，公式必须从：

```text
category + template family + object split
  -> layer-site-scale path
```

进一步写成：

```text
Router(c, T, O)
  -> (layer, site, scale)
```

并且：

```text
SupportRestore(Router(c,T,O)) -> readout clean
Readout clean -/-> token selection
```

也就是说，当前已经分离出两个不同层级：

```text
1. readout support routing
   内部读出支持路由

2. token selection gate
   词元选择门
```

当前理论更谨慎版本：

```text
语言编码机制至少包含：
  context field formation
  template-conditioned support router
  scale-sensitive interface injection
  competition/release control
  format-conditioned token selection gate
```

Phase146 已经强力支持前两项，但最后一项仍未破解。

### 硬伤和瓶颈

1. **token-level closure 仍然失败**

token_argmax 全部为 0，target token rank 仍很高。内部 readout clean 还没有转成真实输出。

2. **best path 有过拟合风险**

本轮 per-template sweep 在测试集上选 best path，因此它证明“存在模板条件路径”，但还没有证明 router 可以从训练数据预测到测试数据。

3. **number 仍不稳定**

number 常需要高 scale，且 release 大，说明 number 的 clean path 可能需要更精细 scale 或不同 token/readout 定义。

4. **top token 多是格式 token**

空格、介词、冒号换行等 token 常占 top，这说明 prompt format 和 generation surface 仍未对齐。

5. **没有 autoregressive multi-token generation**

本轮是 first-token logits/rank，不是完整生成。

### 下一阶段任务

Phase147 应做：

```text
Trainable Router Generalization and Format-Gated Token Closure
可泛化路由器与格式门控词元闭合
```

核心任务：

```text
1. 将 Phase146 的 best path 选择从 test-time search 改成 train-time router。
2. 在 train templates/objects 上选择 best layer/site/scale，
   再迁移到 heldout templates/objects。
3. 对 token selection gap 做格式对照：
   原始 prompt tail
   强制 category-token tail
   multiple-choice tail
   JSON/label tail
4. 判断 token gap 是否主要来自 format gate。
5. 对 plant/time/container 做 first-token + 2-token generation closure。
```

建议测试：

```text
DS7B main:
  categories = plant,time,container,number
  train templates vs heldout templates
  train objects vs heldout objects
  router candidates:
    layers = true-last,last-1
    sites = input_answer,attention_output,mlp_input
    scales = 0.2,0.25,0.3,0.35,0.5,0.75,1.0,1.25,1.5
  format tails:
    plain
    label_colon
    answer_is_one_word
    multiple_choice

Qwen3/GLM4:
  smaller confirmation。
```

成功判据：

```text
Router generalization:
  train-selected path 在 heldout templates/objects 上仍保持 clean_rate 高。

Token closure:
  如果格式 tail 改变后 target token rank 大幅下降或 argmax 出现，
  说明 token selection gap 主要来自 format gate。

如果 readout clean 保持但 token rank 仍不动，
  说明 LM-head/token selection 是独立机制层。
```

## Phase 147: Trainable Router Generalization and Format-Gated Token Closure 可泛化路由器与格式门控词元闭合 [2026-06-15 11:39]

### 本阶段目标

根据用户要求，先分析附加内容是否正确，再继续完成客观测试。

附加分析基本正确：

```text
1. Phase146 证明 template-conditioned routing 可以恢复内部 readout。
2. Phase146 同时证明 internal readout restore 与 token selection 之间有断层。
3. Phase146 的 per-template best path 是 test-time search，有过拟合风险。
4. 下一步必须做 train-time router selection -> heldout template/object generalization。
5. token selection gap 需要用 format tail 对照测试。
```

本轮目标：

```text
1. 在 train templates/objects 上选择 best layer/site/scale。
2. 把 train-selected router 迁移到 heldout template/object。
3. 测四种 format tail：
   plain
   label_colon
   answer_one_word
   multiple_choice
4. 同时记录 readout clean、target token rank、target token argmax。
```

### 执行命令

```bash
python tests/gpt5/phase147_train_router_format_token_cuda.py qwen3 \
  --train-objects 4 \
  --test-objects 4 \
  --batch-size 8 \
  --rank 4 \
  --categories plant,time \
  --template-families long \
  --splits front_back \
  --formats plain,label_colon \
  --layer-offsets 0,-1 \
  --sites input_answer,mlp_input \
  --scales 0.25,0.5 \
  --output-dir results/gpt5_phase147_smoke \
  --hard-exit-after-model

python tests/gpt5/phase147_train_router_format_token_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 8 \
  --batch-size 16 \
  --rank 8 \
  --categories plant,time \
  --template-families long,neutral \
  --splits front_back \
  --formats plain,label_colon \
  --layer-offsets 0,-1 \
  --sites input_answer,attention_output,mlp_input \
  --scales 0.25,0.5,1.0 \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase147_train_router_format_token \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase147_train_router_format_token_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 8 \
  --batch-size 16 \
  --rank 8 \
  --categories plant,time \
  --template-families long,neutral \
  --splits front_back \
  --formats plain,label_colon \
  --layer-offsets 0,-1 \
  --sites input_answer,attention_output,mlp_input \
  --scales 0.25,0.5,1.0 \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase147_train_router_format_token \
  --hard-exit-after-model

python tests/gpt5/phase147_train_router_format_token_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 8 \
  --batch-size 16 \
  --rank 8 \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats plain,label_colon,answer_one_word,multiple_choice \
  --layer-offsets 0,-1 \
  --sites input_answer,attention_output,mlp_input \
  --scales 0.2,0.25,0.3,0.35,0.5,0.75,1.0,1.25,1.5 \
  --release-threshold 0.25 \
  --output-dir results/gpt5_phase147_train_router_format_token \
  --hard-exit-after-model

python tests/gpt5/phase147_train_router_format_token_summary.py

python -m py_compile \
  tests/gpt5/phase147_train_router_format_token_cuda.py \
  tests/gpt5/phase147_train_router_format_token_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase147_train_router_format_token_cuda.py`
- 汇总脚本：`tests/gpt5/phase147_train_router_format_token_summary.py`
- Qwen3 结果：`results/gpt5_phase147_train_router_format_token/phase147_qwen3_train_router_format_token.json`
- GLM4 结果：`results/gpt5_phase147_train_router_format_token/phase147_glm4_train_router_format_token.json`
- DS7B 结果：`results/gpt5_phase147_train_router_format_token/phase147_deepseek7b_train_router_format_token.json`
- 跨模型汇总：`results/gpt5_phase147_train_router_format_token/phase147_cross_model_summary.md`

### 测试设计

```text
train router:
  train templates = template 0,1
  heldout template = template 2
  train objects = 8
  heldout objects = 8

candidate router:
  layers = true-last,last-1
  sites = input_answer,attention_output,mlp_input
  scales = 0.2,0.25,0.3,0.35,0.5,0.75,1.0,1.25,1.5

format tails:
  plain
  label_colon
  answer_one_word
  multiple_choice
```

### 客观结果

#### Qwen3

```text
plant:
  held_clean_rate = 0
  mean_release +12.57
  mean_token_rank 48998.9
  token_argmax 0

time:
  held_clean_rate = 0
  mean_release +13.20
  mean_token_rank 85218.6
  token_argmax 0
```

Qwen3 继续没有 clean 泛化。

#### GLM4 bf16

```text
plant:
  held_clean_rate = 0
  mean_release +1.80
  mean_token_rank 1956.5
  token_argmax 0

time:
  held_clean_rate = 0
  mean_release +1.35
  mean_token_rank 4942.8
  token_argmax 0
```

GLM4 的 label_colon 显著降低 token rank，但 readout clean 与 token argmax 都没有闭合。

#### DS7B

按类别：

```text
container:
  train_clean_rate = 0.75
  held_clean_rate = 0.17
  mean_recovery +1.18
  mean_release +1.49
  mean_token_rank 10498.3
  token_argmax 0

number:
  train_clean_rate = 0.67
  held_clean_rate = 0.17
  mean_recovery +0.88
  mean_release +1.36
  mean_token_rank 13163.3
  token_argmax 0

plant:
  train_clean_rate = 0.62
  held_clean_rate = 0.08
  mean_recovery +1.45
  mean_release +2.38
  mean_token_rank 2426.9
  token_argmax 0

time:
  train_clean_rate = 0.83
  held_clean_rate = 0.25
  mean_recovery +4.39
  mean_release +1.74
  mean_token_rank 17019.6
  token_argmax 0
```

按 format tail：

```text
answer_one_word:
  train_clean_rate = 0.50
  held_clean_rate = 0.29
  mean_token_rank 16506.2
  token_argmax 0

label_colon:
  train_clean_rate = 0.83
  held_clean_rate = 0.21
  mean_token_rank 6165.5
  token_argmax 0

multiple_choice:
  train_clean_rate = 0.67
  held_clean_rate = 0.08
  mean_token_rank 10654.8
  token_argmax 0

plain:
  train_clean_rate = 0.88
  held_clean_rate = 0.08
  mean_token_rank 9781.6
  token_argmax 0
```

### 当前最可靠客观事实

1. **train-time router 泛化明显不足**

训练侧经常能选到 clean path：

```text
plain train_clean_rate = 0.88
label_colon train_clean_rate = 0.83
```

但迁移到 heldout template/object 后：

```text
plain held_clean_rate = 0.08
label_colon held_clean_rate = 0.21
```

这说明 Phase146 的 test-time router 存在明显过拟合风险。

2. **格式尾部可以降低 token rank，但不能打开 argmax**

部分 case 中，target token rank 降到很低：

```text
plant multiple_choice: min rank 9.9
number label_colon: min rank 14.9
container multiple_choice: min rank 19.5
```

但：

```text
token_argmax = 0
```

所以 format gate 有影响，但不是充分条件。

3. **heldout readout clean 与 token rank 都未闭合**

DS7B heldout clean_rate 最高是：

```text
time 0.25
container 0.17
number 0.17
plant 0.08
```

这说明当前 train-selected router 还不能泛化到 heldout 模板/对象。

4. **类别差异明显**

```text
time:
  held_clean_rate 最高，但 mean_release 仍高。

plant:
  token rank 最低，但 held_clean_rate 最低。

container/number:
  readout 和 token 都不稳定。
```

这说明“读出泛化”和“词元接近”是两个不同轴。

5. **multiple_choice 不是万能格式**

multiple_choice 在若干 case 中显著降低 rank，但 held_clean_rate 只有 0.08，并且 release 不稳定。它更多改变 token surface，而不是稳定内部 readout。

### 对当前理论的修正

Phase147 后，理论必须进一步拆成三层：

```text
1. local support path existence
   局部支持路径存在

2. router generalization
   路由泛化

3. token selection gate
   词元选择门
```

Phase146 证明第 1 层在 test-time search 下很强。

Phase147 证明：

```text
第 2 层尚未解决：
  train-selected router 不能稳定泛化。

第 3 层也尚未解决：
  format tail 能降低 rank，但不能产生 argmax。
```

因此当前不能说已经破解语言输出机制。更准确说：

```text
已经观察到可恢复支持通道；
已经确认模板条件化路由存在；
但尚未找到可泛化路由规则；
也尚未打开词元选择门。
```

### 硬伤和瓶颈

1. **训练侧 router 过拟合**

train clean 高，heldout clean 低，说明 path 选择对模板和对象非常敏感。

2. **format tail 只改善 rank，不产生 argmax**

格式影响强，但仍不能把目标类别词元推到第一。

3. **token rank 和 readout clean 不同步**

plant token rank 较低但 clean_rate 低；time clean_rate 较高但 rank 高。

4. **对象数量仍有限**

8/8 是为了容纳模板和格式扩展，后续需要更平衡对象采样或交叉验证。

5. **仍未做完整 autoregressive generation**

当前是 first-token logits/rank；多 token 生成仍未闭合。

### 下一阶段任务

Phase148 应做：

```text
Router Feature Audit and LM-Head Alignment
路由特征审计与语言模型头对齐
```

核心目标：

```text
1. 不再只选 best path，而要分析 best path 的可预测特征。
2. 比较 train-success 与 heldout-fail 的差异：
   pre-answer basis shift
   answer-state norm shift
   transfer prediction error
   readout/token alignment
3. 直接测 LM-head category token direction 与 DCF readout direction 的夹角/投影。
4. 对 low-rank support restore 后，再加 LM-head aligned token steering 小尺度项。
5. 判断 token gap 是方向错配还是格式门控。
```

建议测试：

```text
DS7B main:
  categories = plant,time,container,number
  formats = label_colon,multiple_choice,answer_one_word
  splits = front_back,back_front
  template families = long,short,neutral

measure:
  train vs heldout basis cosine
  transfer R2 train/heldout
  answer norm delta
  DCF direction vs LM-head target token direction cosine
  token-rank before/after LM-head aligned small steering

Qwen3/GLM4:
  smaller confirmation。
```

成功判据：

```text
如果 heldout failure 与 basis/transfer drift 对齐，
  说明 router generalization 是主要瓶颈。

如果 LM-head aligned steering 显著降低 token rank，
  说明 token gap 主要是 DCF readout 与 LM-head direction 错配。

如果 steering 也无效，
  说明 token selection gate 还包含更深的格式/生成动力学。
```

## Phase 148: Router Feature and LM-Head Alignment Audit 路由特征与词元头对齐审计 [2026-06-15 11:56]

### 本阶段目标

根据 Phase147 的失败性进展，验证两个关键瓶颈：

```text
1. train-selected router 是否因为 train/heldout feature drift 而不能泛化。
2. readout restore 与最终 token selection 之间的 gap 是否来自 DCF support direction 与 LM-head target token direction 不对齐。
```

附件分析中正确部分：

```text
Phase147 的核心结论不是“没有找到机制”，而是把瓶颈拆成了两层：
  router generalization 未解决；
  token selection gate 未打开。

下一步不应继续只找更强 target_delta，而应同时审计：
  train-success vs heldout-fail 的特征漂移；
  DCF/readout direction 与 LM-head token direction 的对齐程度；
  简单 LM-head aligned steering 是否能改善 token rank 或 argmax。
```

### 脚本

```text
tests/gpt5/phase148_router_feature_lmhead_alignment_cuda.py
tests/gpt5/phase148_router_feature_lmhead_alignment_summary.py
```

### 执行命令

```bash
python tests/gpt5/phase148_router_feature_lmhead_alignment_cuda.py qwen3 \
  --categories plant,time \
  --template-families long \
  --splits front_back \
  --formats label_colon \
  --lm-scales 0.0,0.1 \
  --train-objects 4 --test-objects 4 --batch-size 8 --rank 4 \
  --output-dir results/gpt5_phase148_smoke \
  --hard-exit-after-model

python tests/gpt5/phase148_router_feature_lmhead_alignment_cuda.py qwen3 \
  --categories plant,time \
  --template-families long,neutral \
  --splits front_back \
  --formats label_colon \
  --lm-scales 0.0,0.05,0.1,0.2 \
  --train-objects 8 --test-objects 8 --batch-size 16 --rank 8 \
  --output-dir results/gpt5_phase148_router_feature_lmhead_alignment \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase148_router_feature_lmhead_alignment_cuda.py glm4 \
  --categories plant,time \
  --template-families long,neutral \
  --splits front_back \
  --formats label_colon \
  --lm-scales 0.0,0.05,0.1,0.2 \
  --train-objects 8 --test-objects 8 --batch-size 16 --rank 8 \
  --output-dir results/gpt5_phase148_router_feature_lmhead_alignment \
  --hard-exit-after-model

python tests/gpt5/phase148_router_feature_lmhead_alignment_cuda.py deepseek7b \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats label_colon,multiple_choice,answer_one_word \
  --lm-scales 0.0,0.05,0.1,0.2,0.5 \
  --train-objects 8 --test-objects 8 --batch-size 16 --rank 8 \
  --output-dir results/gpt5_phase148_router_feature_lmhead_alignment \
  --hard-exit-after-model

python tests/gpt5/phase148_router_feature_lmhead_alignment_summary.py

python -m py_compile \
  tests/gpt5/phase148_router_feature_lmhead_alignment_cuda.py \
  tests/gpt5/phase148_router_feature_lmhead_alignment_summary.py
```

### 结果文件

```text
results/gpt5_phase148_router_feature_lmhead_alignment/phase148_qwen3_router_feature_lmhead_alignment.json
results/gpt5_phase148_router_feature_lmhead_alignment/phase148_glm4_router_feature_lmhead_alignment.json
results/gpt5_phase148_router_feature_lmhead_alignment/phase148_deepseek7b_router_feature_lmhead_alignment.json
results/gpt5_phase148_router_feature_lmhead_alignment/phase148_cross_model_summary.md
```

### 测试范围

```text
Qwen3:
  categories = plant,time
  template_families = long,neutral
  split = front_back
  format = label_colon
  train/test objects = 8/8

GLM4:
  categories = plant,time
  template_families = long,neutral
  split = front_back
  format = label_colon
  train/test objects = 8/8
  dtype = bf16

DS7B main:
  categories = plant,time,container,number
  template_families = long,short,neutral
  splits = front_back,back_front
  formats = label_colon,multiple_choice,answer_one_word
  train/test objects = 8/8
  lm_scales = 0.0,0.05,0.1,0.2,0.5
```

### 客观结果

#### Qwen3

```text
plant:
  prev_clean 0.00
  best_clean 0.00
  pre_basis_overlap 0.53
  ans_basis_overlap 0.59
  heldout_transfer_R2 -4.28
  support_lm_cosine +0.11
  target_rank 68741.5 -> 68740.1
  argmax 0.00

time:
  prev_clean 0.00
  best_clean 0.00
  pre_basis_overlap 0.54
  ans_basis_overlap 0.59
  heldout_transfer_R2 -11.43
  support_lm_cosine +0.06
  target_rank 111794.3 -> 111794.3
  argmax 0.00
```

Qwen3 中 LM-head aligned steering 基本无效。

#### GLM4

```text
plant:
  prev_clean 0.00
  best_clean 0.00
  heldout_transfer_R2 -3.77
  support_lm_cosine +0.02
  target_rank 281.9 -> 276.1
  argmax 0.00

time:
  prev_clean 0.00
  best_clean 0.00
  heldout_transfer_R2 -4.98
  support_lm_cosine +0.03
  target_rank 363.1 -> 358.4
  argmax 0.00
```

GLM4 的 target token rank 本身较低，但 steering 只带来很小改善，没有 argmax closure。

#### DS7B by category

```text
container:
  n 18
  prev_clean 0.22
  best_clean 0.22
  pre_basis_overlap 0.47
  ans_basis_overlap 0.61
  heldout_transfer_R2 -4.92
  support_lm_cosine +0.06
  target_rank 11990.6 -> 11954.2
  argmax 0.00

number:
  n 18
  prev_clean 0.17
  best_clean 0.17
  pre_basis_overlap 0.48
  ans_basis_overlap 0.60
  heldout_transfer_R2 -27.03
  support_lm_cosine +0.08
  target_rank 13895.8 -> 13871.0
  argmax 0.00

plant:
  n 18
  prev_clean 0.11
  best_clean 0.11
  pre_basis_overlap 0.46
  ans_basis_overlap 0.60
  heldout_transfer_R2 -9.98
  support_lm_cosine +0.08
  target_rank 798.1 -> 785.0
  argmax 0.00

time:
  n 18
  prev_clean 0.28
  best_clean 0.28
  pre_basis_overlap 0.46
  ans_basis_overlap 0.59
  heldout_transfer_R2 -11.16
  support_lm_cosine +0.06
  target_rank 17750.8 -> 17703.2
  argmax 0.00
```

#### DS7B by format

```text
answer_one_word:
  n 24
  prev_clean 0.29
  best_clean 0.29
  pre_basis_overlap 0.44
  ans_basis_overlap 0.66
  heldout_transfer_R2 -9.03
  support_lm_cosine +0.06
  target_rank 16506.2 -> 16448.0
  argmax 0.00

label_colon:
  n 24
  prev_clean 0.21
  best_clean 0.21
  pre_basis_overlap 0.46
  ans_basis_overlap 0.59
  heldout_transfer_R2 -12.55
  support_lm_cosine +0.07
  target_rank 6165.5 -> 6141.2
  argmax 0.00

multiple_choice:
  n 24
  prev_clean 0.08
  best_clean 0.08
  pre_basis_overlap 0.50
  ans_basis_overlap 0.55
  heldout_transfer_R2 -18.24
  support_lm_cosine +0.08
  target_rank 10654.8 -> 10645.9
  argmax 0.00
```

### 关键现象

1. **train-selected router 的 heldout 泛化失败与特征漂移一致**

```text
heldout_transfer_R2 在三模型中大多为负。
DS7B number 平均 -27.03，time -11.16，plant -9.98，container -4.92。
```

这说明训练模板/训练对象上选出的 restore path，不能稳定预测保留模板/保留对象的状态变化。

2. **pre/answer basis overlap 只有中等强度**

```text
DS7B pre_basis_overlap 约 0.46-0.48
DS7B ans_basis_overlap 约 0.59-0.61
```

answer state 比 pre-answer state 稳一些，但仍不足以支撑稳定的跨模板泛化。

3. **DCF support direction 与 LM-head target token direction 对齐很弱**

```text
DS7B support_lm_cosine 约 +0.06 到 +0.08
Qwen3 约 +0.06 到 +0.11
GLM4 约 +0.02 到 +0.03
```

这说明 readout restore 的方向并不等于最终目标词元的直接上升方向。

4. **简单 LM-head aligned steering 不能打开 token gate**

```text
DS7B target rank 有轻微改善，但 argmax 始终 0.00。
Qwen3 基本无改善。
GLM4 rank 小幅下降，但 argmax 仍为 0.00。
```

例如：

```text
DS7B plant: 798.1 -> 785.0
DS7B number: 13895.8 -> 13871.0
GLM4 plant: 281.9 -> 276.1
```

这些改善不足以解释最终选择机制。

5. **低 rank 个案已经出现，但仍不是 argmax**

```text
DS7B plant multiple_choice 个别条件 rank 约 9-15。
DS7B number label_colon 个别条件 rank 约 14-20。
DS7B container multiple_choice 个别条件 rank 约 19。
```

这说明问题不只是“目标词元完全不可见”，而是候选竞争、格式约束、最终归一化或生成步骤仍未闭合。

### 对 Phase147 附件分析的判断

附件分析基本正确。

正确部分：

```text
Phase147 是失败性进展，不是无效进展。
router generalization 和 token selection gate 是两个独立瓶颈。
继续只追逐 readout recovery 会陷入局部结构堆叠。
需要审计 train/heldout feature drift 和 LM-head alignment。
```

需要修正的部分：

```text
LM-head aligned steering 并没有显著降低 target token rank，更没有产生 argmax closure。
因此 token gap 不能简单解释为“DCF/readout 方向与 LM-head 方向错配后，加一个 LM-head 方向即可解决”。
当前更像是：
  DCF support restore 负责内部语义状态的一部分；
  final token selection 还受 format、candidate competition、final norm、generation dynamics 共同控制。
```

### 理论进展

当前最稳妥的结构图应更新为：

```text
object/template condition
  -> route selection
  -> local support restore
  -> answer-state readout improvement
  -> final norm / LM-head / candidate competition
  -> token selection
```

Phase146-148 共同说明：

```text
readout restore 是真实内部现象；
template-conditioned router 是真实现象；
train-selected router 的跨模板泛化不稳；
最终 token selection 不是 readout restore 的直接线性后果；
LM-head direction 与 DCF support direction 只弱相关；
简单加 LM-head direction 不足以打开 token gate。
```

这迫使理论从单一支持路径模型，转向更严格的两级门控模型：

```text
第一级：内部语义/关系状态门控。
第二级：最终词元选择门控。
```

### 硬伤

1. **Phase148 仍依赖 Phase147 选出的路径**

```text
如果 Phase147 的 route selector 本身不稳，Phase148 审计的是“不稳定选择器的后果”，不是最优全局机制。
```

2. **LM-head target direction 仍是单词元近似**

```text
category label 可能不是模型实际生成类别的唯一 token route。
多 token、同义词、格式 token、选项 token 都可能参与竞争。
```

3. **没有直接审计 final norm 前后**

```text
当前只在内部 path 上加 support/LM-head steering。
如果 final norm 或最后一层 residual 重新缩放方向，内部 steering 可能被压扁。
```

4. **full-vocab argmax 过于严格但必要**

```text
argmax 0.00 说明完整生成闭合仍失败。
但如果 candidate-set argmax 已经成功，就应把问题定位到开放词表竞争。
Phase148 尚未完成 candidate-set audit。
```

5. **Qwen3/GLM4 只是小范围确认**

```text
主要结论来自 DS7B。
跨模型普遍趋势存在，但 Qwen3/GLM4 的测试范围小于 DS7B。
```

### 下一步 Phase149

Phase149 应进入：

```text
Final-Norm and Candidate-Set Token Gate Audit
最终归一化与候选集词元门审计
```

目标不是再找更大的 readout delta，而是分离 token gap 的来源：

```text
1. full-vocab competition:
   目标类别词元是否在候选集内已胜出，但被开放词表其他 token 压制。

2. final norm gate:
   support restore 在 final norm 前后是否被重缩放、旋转或压制。

3. candidate token basis:
   类别答案是否应由多个 label token、synonym token、option token 共同表示。

4. generation dynamics:
   单步 next-token rank 是否不能代表短答案生成闭合。
```

建议测试：

```text
models = qwen3, glm4, deepseek7b

DS7B main:
  categories = plant,time,container,number
  formats = label_colon,multiple_choice,answer_one_word
  template_families = long,short,neutral
  splits = front_back,back_front
  train/test objects >= 8/8

Qwen3/GLM4:
  categories = plant,time
  format = label_colon,multiple_choice
  train/test objects >= 8/8

measure:
  full_vocab_rank
  candidate_set_rank
  candidate_set_argmax
  final_norm_input vs final_norm_output logit lens
  target logit delta before/after support restore
  top competing tokens and their category relation
  one-token vs two-token constrained generation

interventions:
  support restore only
  final_norm_input LM-head steering
  final hidden state LM-head steering
  candidate competitor suppression
```

判据：

```text
如果 candidate_set_argmax 成功但 full_vocab_argmax 失败：
  token gap 主要来自开放词表竞争。

如果 final_norm 前有效、final_norm 后失效：
  final norm 是关键压制门。

如果 final hidden state steering 仍不能打开 argmax：
  说明目标答案不是单一 label token 方向，而是生成动力学/格式路径。

如果 two-token constrained generation 成功：
  下一步转向短答案生成闭合，而不是单 token argmax。
```

## Phase 149: Final-Norm Candidate Token Gate Audit 最终归一化与候选词元门审计 [2026-06-15 12:19]

### 本阶段目标

结合两个附件的正确部分继续推进。

附件一关于 AGI 路线和语言编码机制的判断，正确部分是：

```text
语言是理解智能编码机制的关键入口；
深度神经网络逆向工程比直接类脑测量更容易做因果实验；
语言能力不是单个词向量现象，而是概念、关系、语法、格式、任务模式共同形成的系统；
当前研究应继续积累可干预、可复现的客观拼图。
```

需要谨慎修正的部分：

```text
破解语言编码机制是 AGI 理论的核心入口，但不能直接等同于完整 AGI。
实时学习、稳定记忆更新、系统级世界模型、行动闭环仍需要额外机制。
“人脑接近物理最优解”和“两三年临界点”可以作为猜想，不能作为当前实验结论。
```

附件二对 Phase148 的判断基本正确：

```text
Phase148 是负结果推进；
router generalization 和 LM-head alignment 是两个独立瓶颈；
token selection gap 不能简单归因于少加了目标 token direction；
下一步必须分离 candidate-set gate、full-vocab competition 和 final norm gate。
```

因此 Phase149 的目标是：

```text
1. 测 candidate-set argmax 是否已经成功。
2. 测 full-vocab argmax 是否仍失败。
3. 比较 final_norm_input 与 final_norm_output 的候选集读出。
4. 测 support restore 后，在 final_norm_input/output 加 LM-head direction 是否能打开词元门。
5. 测简单 candidate competitor suppression 是否有效。
```

### 脚本

```text
tests/gpt5/phase149_final_norm_candidate_gate_cuda.py
tests/gpt5/phase149_final_norm_candidate_gate_summary.py
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase149_final_norm_candidate_gate_cuda.py \
  tests/gpt5/phase149_final_norm_candidate_gate_summary.py

python tests/gpt5/phase149_final_norm_candidate_gate_cuda.py qwen3 \
  --categories plant,time \
  --template-families long,neutral \
  --splits front_back \
  --formats label_colon,multiple_choice \
  --lm-scales 0.0,1.0,4.0 \
  --suppress-scales 1.0 \
  --train-objects 8 --test-objects 8 --batch-size 16 --rank 8 \
  --output-dir results/gpt5_phase149_final_norm_candidate_gate \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase149_final_norm_candidate_gate_cuda.py glm4 \
  --categories plant,time \
  --template-families long,neutral \
  --splits front_back \
  --formats label_colon,multiple_choice \
  --lm-scales 0.0,1.0,4.0 \
  --suppress-scales 1.0 \
  --train-objects 8 --test-objects 8 --batch-size 16 --rank 8 \
  --output-dir results/gpt5_phase149_final_norm_candidate_gate \
  --hard-exit-after-model

python tests/gpt5/phase149_final_norm_candidate_gate_cuda.py deepseek7b \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats label_colon,multiple_choice,answer_one_word \
  --lm-scales 0.0,1.0,4.0 \
  --suppress-scales 1.0 \
  --train-objects 8 --test-objects 8 --batch-size 16 --rank 8 \
  --output-dir results/gpt5_phase149_final_norm_candidate_gate \
  --hard-exit-after-model

python tests/gpt5/phase149_final_norm_candidate_gate_summary.py
```

### 结果文件

```text
results/gpt5_phase149_final_norm_candidate_gate/phase149_qwen3_final_norm_candidate_gate.json
results/gpt5_phase149_final_norm_candidate_gate/phase149_glm4_final_norm_candidate_gate.json
results/gpt5_phase149_final_norm_candidate_gate/phase149_deepseek7b_final_norm_candidate_gate.json
results/gpt5_phase149_final_norm_candidate_gate/phase149_cross_model_summary.md
```

### 测试范围

```text
Qwen3:
  categories = plant,time
  template_families = long,neutral
  split = front_back
  formats = label_colon,multiple_choice
  train/test objects = 8/8

GLM4:
  categories = plant,time
  template_families = long,neutral
  split = front_back
  formats = label_colon,multiple_choice
  train/test objects = 8/8

DS7B main:
  categories = plant,time,container,number
  template_families = long,short,neutral
  splits = front_back,back_front
  formats = label_colon,multiple_choice,answer_one_word
  train/test objects = 8/8
  total cases = 72
```

### 客观结果

#### Qwen3

```text
cases = 4

best candidate gate:
  candidate_argmax_rate mean = 0.781
  candidate_rank_mean = 1.219
  full_vocab_rank_mean = 37920.4
  full_vocab_argmax_rate = 0.000
  final_norm_input candidate_argmax = 0.500
  final_norm_output candidate_argmax = 0.781

support_only:
  candidate_argmax = 0.500
  full_vocab_rank_mean = 90267.9
  full_vocab_argmax = 0.000

final_norm_output_lm scale4:
  candidate_argmax = 0.781
  full_vocab_rank_mean = 37920.4
  full_vocab_argmax = 0.000
```

类别：

```text
plant:
  candidate_argmax = 1.000
  full_vocab_argmax = 0.000

time:
  candidate_argmax = 0.562
  full_vocab_argmax = 0.000
```

#### GLM4 bf16

```text
cases = 4

best candidate gate:
  candidate_argmax_rate mean = 0.969
  candidate_rank_mean = 1.031
  full_vocab_rank_mean = 90.8
  full_vocab_argmax_rate = 0.000
  final_norm_input candidate_argmax = 0.969
  final_norm_output candidate_argmax = 0.969

support_only:
  candidate_argmax = 0.906
  full_vocab_rank_mean = 322.5
  full_vocab_argmax = 0.000

final_norm_output_lm scale4:
  candidate_argmax = 0.969
  full_vocab_rank_mean = 90.8
  full_vocab_argmax = 0.000
```

GLM4 的目标类别词元已经进入较低 full-vocab rank，但仍没有 full-vocab argmax closure。

#### DS7B

```text
cases = 72

best candidate gate:
  candidate_argmax_rate mean = 0.679
  candidate_rank_mean = 1.481
  full_vocab_rank_mean = 4339.6
  full_vocab_argmax_rate = 0.000
  final_norm_input candidate_argmax = 0.319
  final_norm_output candidate_argmax = 0.679

support_only:
  candidate_argmax = 0.413
  candidate_rank_mean = 2.122
  full_vocab_rank_mean = 11108.8
  full_vocab_argmax = 0.000
  clean_rate = 0.194

final_norm_input_lm scale4:
  candidate_argmax = 0.424
  candidate_rank_mean = 2.083
  full_vocab_rank_mean = 10876.9
  full_vocab_argmax = 0.000

final_norm_output_lm scale4:
  candidate_argmax = 0.679
  candidate_rank_mean = 1.481
  full_vocab_rank_mean = 4334.7
  full_vocab_argmax = 0.000
  clean_rate = 0.139

final_norm_output_suppress:
  candidate_argmax = 0.516
  candidate_rank_mean = 1.979
  full_vocab_rank_mean = 11079.1
  full_vocab_argmax = 0.000
```

#### DS7B by category

```text
plant:
  candidate_argmax = 0.99
  candidate_rank = 1.01
  full_vocab_rank = 101.8
  full_vocab_argmax = 0.00
  final_norm_input/output candidate_argmax = 0.85 / 0.99

time:
  candidate_argmax = 0.66
  candidate_rank = 1.46
  full_vocab_rank = 7025.0
  full_vocab_argmax = 0.00
  final_norm_input/output candidate_argmax = 0.12 / 0.66

container:
  candidate_argmax = 0.61
  candidate_rank = 1.51
  full_vocab_rank = 3736.7
  full_vocab_argmax = 0.00
  final_norm_input/output candidate_argmax = 0.13 / 0.61

number:
  candidate_argmax = 0.46
  candidate_rank = 1.94
  full_vocab_rank = 6494.7
  full_vocab_argmax = 0.00
  final_norm_input/output candidate_argmax = 0.18 / 0.46
```

#### DS7B by format

```text
label_colon:
  candidate_argmax = 0.83
  full_vocab_rank = 1742.2
  full_vocab_argmax = 0.00

answer_one_word:
  candidate_argmax = 0.61
  full_vocab_rank = 5550.9
  full_vocab_argmax = 0.00

multiple_choice:
  candidate_argmax = 0.59
  full_vocab_rank = 5725.5
  full_vocab_argmax = 0.00
```

### 关键现象

1. **candidate-set gate 可以部分打开**

Phase148 只看到 full-vocab argmax 始终失败。Phase149 显示，在候选类别集合内部，目标类别经常已经成为第一：

```text
Qwen3 candidate_argmax 0.781
GLM4 candidate_argmax 0.969
DS7B candidate_argmax 0.679
```

这说明 token gap 不能简单说成“目标类别没有进入输出空间”。更准确是：

```text
目标类别在候选类别子空间内经常可胜出；
但在开放词表中仍被格式词、符号词、其他普通 token 压制。
```

2. **full-vocab argmax 仍然完全没有闭合**

```text
Qwen3 full_vocab_argmax = 0.000
GLM4 full_vocab_argmax = 0.000
DS7B full_vocab_argmax = 0.000
```

即使 DS7B plant 的 candidate_argmax 约 0.99，full-vocab_argmax 仍是 0。

这证明：

```text
候选集闭合 ≠ 真实开放生成闭合。
```

3. **final_norm_output 比 final_norm_input 更接近候选集闭合**

DS7B：

```text
final_norm_input candidate_argmax = 0.319
final_norm_output candidate_argmax = 0.679
```

类别上也一致：

```text
plant 0.85 -> 0.99
time 0.12 -> 0.66
container 0.13 -> 0.61
number 0.18 -> 0.46
```

这说明 final norm 不是简单压制门。它在很多条件下反而把状态推向候选类别可读空间。

4. **final_norm_output_lm 是最有效的干预，但仍不够**

最佳 variant 几乎都来自：

```text
final_norm_output_lm scale4
```

DS7B 72 个 case 中 71 个最佳候选集结果来自该 variant。

但即使这样：

```text
full_vocab_argmax 仍为 0。
```

说明 LM-head 方向能改善候选集竞争，但无法击穿开放词表竞争。

5. **support_only 已经有一部分候选集信号**

DS7B：

```text
support_only candidate_argmax = 0.413
final_norm_output_lm candidate_argmax = 0.679
```

这说明 support restore 不是无效，它确实把一部分内部语义状态推到类别候选集；但还需要 final output 层面的词元方向才能显著增强。

6. **plant 是当前最接近闭合的类别**

DS7B：

```text
plant candidate_argmax = 0.99
plant full_vocab_rank = 101.8
plant full_vocab_argmax = 0.00
```

plant 已经不是候选集问题，而是开放词表竞争问题。

number/time/container 仍同时存在候选集竞争和开放词表竞争。

### 对当前理论的修正

Phase149 后，结构图应更新为：

```text
object/template/format condition
  -> route selection
  -> support restore / internal semantic field
  -> final norm organizes candidate-readable state
  -> LM-head candidate category competition
  -> open-vocab competition
  -> actual next token
```

之前 Phase148 的判断：

```text
readout restore 不等于 token selection。
```

现在可以细化为：

```text
readout restore 有时可以进入 candidate-set selection；
candidate-set selection 仍不等于 full-vocab generation；
final norm output 是候选集门的重要接口；
开放词表竞争是当前最硬的最后一道门。
```

### 对破解语言编码机制的意义

这对语言背后的编码机制有一个重要启发：

```text
概念编码不是直接等于词元输出；
概念编码先在候选语义集合中形成相对优势；
真实语言生成还必须通过格式、词表、上下文和生成约束形成最终词元。
```

所以“语言编码机制”至少分三层：

```text
1. semantic support layer:
   概念/关系支持层。

2. candidate selection layer:
   候选语义类别选择层。

3. open generation layer:
   开放词表生成层。
```

当前已经对第 1 层和第 2 层有较多因果证据，第 3 层仍未闭合。

### 硬伤

1. **final_norm_output_lm scale4 是强干预**

```text
它证明 final_norm_output 是有效接口，但不能证明模型自然使用同样强度的方向。
```

2. **candidate set 是人为定义的**

```text
candidate categories = plant,time,container,number。
如果候选集扩大到所有 32 类，candidate_argmax 可能下降。
```

3. **full-vocab argmax 仍为 0**

```text
真实生成闭合仍未完成。
```

4. **top token 中有大量格式/符号/异常 token**

```text
说明开放词表竞争中，模型并不总是在“类别词”之间选择。
下一步必须分析这些 top token 的来源和格式门。
```

5. **Qwen3/GLM4 范围仍小于 DS7B**

```text
跨模型趋势一致，但主证据仍来自 DS7B 的 72 cases。
```

6. **没有做生成序列闭合**

```text
单步 next-token 可能低估 answer_one_word 或 multiple_choice 下的短答案生成路径。
```

### 下一步 Phase150

Phase150 应进入：

```text
Open-Vocab Competitor and Format Gate Decomposition
开放词表竞争者与格式门分解
```

核心目标：

```text
既然 candidate-set gate 已经部分打开，而 full-vocab argmax 仍为 0，
下一步必须找出开放词表中压制类别词元的东西是什么。
```

建议测试：

```text
models = qwen3, glm4, deepseek7b

DS7B main:
  categories = plant,time,container,number
  formats = label_colon,multiple_choice,answer_one_word
  template_families = long,short,neutral
  splits = front_back,back_front
  train/test objects = 8/8 or larger

measure:
  top 50 full-vocab tokens after support_only
  top 50 full-vocab tokens after final_norm_output_lm
  competitor token type:
    format token
    punctuation
    whitespace/newline
    category token
    object token
    abstract/common token
    unknown/noisy token
  target category token rank among:
    full vocab
    non-format vocab
    alphabetic tokens
    category readout tokens
    prompt option tokens

interventions:
  mask/suppress top format competitors
  mask/suppress punctuation/newline competitors
  constrain to alphabetic token subset
  constrain to option token subset
  run one-step and two-step generation audit
```

判据：

```text
如果移除格式/符号竞争后 full-vocab argmax 接近成功：
  最后一门主要是 format gate。

如果只在 option-token subset 成功：
  multiple-choice 格式需要选项路径，不是类别词路径。

如果 alphabetic subset 仍失败：
  类别答案本身不是当前目标 token set。

如果 two-step generation 成功但 one-step argmax 失败：
  真实输出是短序列路径，不是单 token 闭合。
```

## Phase 150: Open-Vocab Competitor and Format Gate Decomposition 开放词表竞争者与格式门分解 [2026-06-15 13:02]

### 本阶段目标

根据附件对 Phase149 的分析，继续完成任务。

附件判断基本正确：

```text
Phase149 是 Phase146-148 之后的关键分解；
readout restore 已经能部分进入 candidate-set selection；
candidate-set gate 部分打开；
full-vocab generation 仍失败；
final_norm_output 是候选集可读化接口；
下一步必须分解 open-vocab gate 中到底是谁压过目标类别词元。
```

需要谨慎修正的部分：

```text
candidate-set gate 打开不等于 token gate 完全打开；
Phase149 的 candidate set 仍偏窄；
final_norm_output_lm scale4 是强干预；
top token 类型必须系统分类，不能只从个例推断。
```

Phase150 目标：

```text
1. 对 support_only、final_norm_output_lm、final_norm_output_suppress 的 top50 full-vocab tokens 做类型分类。
2. 比较目标词元在 full vocab、non-format vocab、alphabetic vocab、4类 candidate vocab、32类 semantic vocab 中的 rank/argmax。
3. 判断 full-vocab argmax 失败主要来自格式词元、符号词元、同义词、其他类别词，还是普通字母词竞争。
4. 保持三模型顺序 CUDA 测试，DS7B 做主范围。
```

### 脚本

```text
tests/gpt5/phase150_open_vocab_competitor_gate_cuda.py
tests/gpt5/phase150_open_vocab_competitor_gate_summary.py
```

说明：

```text
第一次 DS7B 运行中发现脚本重复解码完整词表，效率过低。
已中止该低效运行，改为一次性缓存词表子集和 token 分类后重新运行。
修正后 DS7B 主测试完成。
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase149_final_norm_candidate_gate_cuda.py \
  tests/gpt5/phase150_open_vocab_competitor_gate_cuda.py \
  tests/gpt5/phase150_open_vocab_competitor_gate_summary.py

python tests/gpt5/phase150_open_vocab_competitor_gate_cuda.py qwen3 \
  --categories plant,time \
  --template-families long,neutral \
  --splits front_back \
  --formats label_colon \
  --train-objects 8 --test-objects 8 --batch-size 16 --rank 8 --top-k 50 \
  --output-dir results/gpt5_phase150_open_vocab_competitor_gate \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase150_open_vocab_competitor_gate_cuda.py glm4 \
  --categories plant,time \
  --template-families long,neutral \
  --splits front_back \
  --formats label_colon \
  --train-objects 8 --test-objects 8 --batch-size 16 --rank 8 --top-k 50 \
  --output-dir results/gpt5_phase150_open_vocab_competitor_gate \
  --hard-exit-after-model

python tests/gpt5/phase150_open_vocab_competitor_gate_cuda.py deepseek7b \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats label_colon,multiple_choice,answer_one_word \
  --train-objects 8 --test-objects 8 --batch-size 16 --rank 8 --top-k 50 \
  --output-dir results/gpt5_phase150_open_vocab_competitor_gate \
  --hard-exit-after-model

python tests/gpt5/phase150_open_vocab_competitor_gate_summary.py
```

### 结果文件

```text
results/gpt5_phase150_open_vocab_competitor_gate/phase150_qwen3_open_vocab_competitor_gate.json
results/gpt5_phase150_open_vocab_competitor_gate/phase150_glm4_open_vocab_competitor_gate.json
results/gpt5_phase150_open_vocab_competitor_gate/phase150_deepseek7b_open_vocab_competitor_gate.json
results/gpt5_phase150_open_vocab_competitor_gate/phase150_cross_model_summary.md
```

### 测试范围

```text
Qwen3:
  categories = plant,time
  template_families = long,neutral
  split = front_back
  format = label_colon
  train/test objects = 8/8
  cases = 4

GLM4:
  categories = plant,time
  template_families = long,neutral
  split = front_back
  format = label_colon
  train/test objects = 8/8
  cases = 4

DS7B main:
  categories = plant,time,container,number
  template_families = long,short,neutral
  splits = front_back,back_front
  formats = label_colon,multiple_choice,answer_one_word
  train/test objects = 8/8
  cases = 72
```

### 客观结果

#### Qwen3

```text
final_norm_output_lm:
  candidate4_argmax = 0.781
  semantic_all_categories_argmax = 0.250
  alphabetic_rank = 17652.9
  non_format_rank = 17654.9
  full_rank = 37920.4
  full_argmax = 0.000

support_only:
  candidate4_argmax = 0.500
  semantic_all_categories_argmax = 0.000
  full_rank = 90267.9
  full_argmax = 0.000

argmax token class:
  non_ascii_or_fragment = 3/4
  format_or_fragment = 1/4
```

Qwen3 的开放词表竞争主要表现为异常碎片/非 ASCII token 压制，candidate4 成功不稳定，扩展到 32 类 semantic set 后明显下降。

#### GLM4 bf16

```text
final_norm_output_lm:
  candidate4_argmax = 0.969
  semantic_all_categories_argmax = 0.906
  alphabetic_rank = 81.5
  non_format_rank = 81.9
  full_rank = 90.8
  full_argmax = 0.000

support_only:
  candidate4_argmax = 0.906
  semantic_all_categories_argmax = 0.438
  full_rank = 322.5
  full_argmax = 0.000

argmax token class:
  alphabetic_other = 3/4
  whitespace = 1/4
```

GLM4 是最接近开放词表闭合的模型，但仍不是目标类别 argmax。

#### DS7B overall

```text
support_only:
  candidate4_argmax = 0.413
  semantic_all_categories_argmax = 0.286
  alphabetic_rank = 6092.7
  non_format_rank = 7474.5
  full_rank = 11094.3
  full_argmax = 0.000

final_norm_output_lm:
  candidate4_argmax = 0.679
  semantic_all_categories_argmax = 0.503
  alphabetic_rank = 2364.4
  non_format_rank = 2928.8
  full_rank = 4334.0
  full_argmax = 0.000

final_norm_output_suppress:
  candidate4_argmax = 0.516
  semantic_all_categories_argmax = 0.311
  alphabetic_rank = 6100.0
  non_format_rank = 7471.5
  full_rank = 11065.2
  full_argmax = 0.000
```

DS7B argmax token class 分布：

```text
support_only:
  alphabetic_other 19
  punctuation 18
  whitespace 15
  target_synonym 14
  other_category 3
  non_ascii_or_fragment 2
  option_label 1

final_norm_output_lm:
  alphabetic_other 21
  target_synonym 17
  punctuation 15
  whitespace 13
  other_category 2
  non_ascii_or_fragment 2
  generic_continuation 1
  object_token 1
```

这说明开放词表竞争不是单一格式词元问题，而是多类竞争混合。

#### DS7B by category

```text
plant:
  candidate4_argmax = 0.99
  semantic_all_categories_argmax = 0.97
  alphabetic_rank = 85.4
  non_format_rank = 86.0
  full_rank = 101.8
  full_argmax = 0.00
  top_arg_class = target_synonym

time:
  candidate4_argmax = 0.66
  semantic_all_categories_argmax = 0.37
  alphabetic_rank = 3690.6
  non_format_rank = 4627.6
  full_rank = 7022.2
  full_argmax = 0.00
  top_arg_class = alphabetic_other

container:
  candidate4_argmax = 0.61
  semantic_all_categories_argmax = 0.41
  alphabetic_rank = 2286.3
  non_format_rank = 2725.4
  full_rank = 3717.4
  full_argmax = 0.00
  top_arg_class = punctuation

number:
  candidate4_argmax = 0.46
  semantic_all_categories_argmax = 0.27
  alphabetic_rank = 3395.2
  non_format_rank = 4276.5
  full_rank = 6494.7
  full_argmax = 0.00
  top_arg_class = alphabetic_other
```

#### DS7B by format

```text
label_colon:
  candidate4_argmax = 0.83
  semantic_all_categories_argmax = 0.61
  alphabetic_rank = 1073.6
  non_format_rank = 1261.2
  full_rank = 1742.2
  full_argmax = 0.00
  top_arg_class = alphabetic_other

multiple_choice:
  candidate4_argmax = 0.59
  semantic_all_categories_argmax = 0.51
  alphabetic_rank = 2749.3
  non_format_rank = 3562.8
  full_rank = 5709.0
  full_argmax = 0.00
  top_arg_class = target_synonym

answer_one_word:
  candidate4_argmax = 0.61
  semantic_all_categories_argmax = 0.39
  alphabetic_rank = 3270.3
  non_format_rank = 3962.5
  full_rank = 5550.9
  full_argmax = 0.00
  top_arg_class = alphabetic_other
```

### 关键现象

1. **4类候选集成功会被 32类 semantic set 明显削弱**

DS7B：

```text
candidate4_argmax = 0.679
semantic_all_categories_argmax = 0.503
```

Qwen3：

```text
candidate4_argmax = 0.781
semantic_all_categories_argmax = 0.250
```

这验证附件中的硬伤：Phase149 的候选集偏窄。扩展候选空间后，候选闭合强度下降。

2. **去掉格式词/限制字母词元仍不能闭合**

DS7B final_norm_output_lm：

```text
alphabetic_rank = 2364.4
non_format_rank = 2928.8
full_rank = 4334.0
full_argmax = 0.000
```

GLM4 虽然 rank 低：

```text
alphabetic_rank = 81.5
full_rank = 90.8
full_argmax = 0.000
```

但仍无法 argmax。说明最后一门不是简单移除空白/标点即可解决。

3. **open-vocab gate 是混合竞争，不是单一 format gate**

DS7B final_norm_output_lm 的 top arg class：

```text
alphabetic_other 21
target_synonym 17
punctuation 15
whitespace 13
```

这说明开放词表中压制目标类别词元的来源至少包括：

```text
普通字母词
目标同义词/相邻表述
标点/空白
其他类别词
异常片段
```

4. **plant 是特殊近闭合类别，但仍不是 full-vocab argmax**

DS7B plant：

```text
candidate4_argmax = 0.99
semantic_argmax = 0.97
full_rank = 101.8
top_arg_class = target_synonym
full_argmax = 0.00
```

这说明 plant 的真正问题不是语义类别竞争，而是：

```text
目标 label token set 与模型自然同义输出路径不完全一致；
或者模型更倾向输出 tree/flower/Plant 等同义或大小写变体。
```

5. **final_norm_output_lm 仍显著改善 rank，但不是生成闭合**

DS7B：

```text
support_only full_rank = 11094.3
final_norm_output_lm full_rank = 4334.0
```

GLM4：

```text
support_only full_rank = 322.5
final_norm_output_lm full_rank = 90.8
```

这说明 final_norm_output_lm 是有效接口，但距离完整生成仍有一道强竞争门。

### 对 Phase149 附件分析的判断

正确部分：

```text
Phase149 的分析基本正确；
candidate-set gate 与 open-vocab gate 应拆开；
final_norm_output 是候选可读接口；
plant 最适合作为 open-vocab gate 的突破对象；
必须分析 top-token ecology。
```

需要修正的部分：

```text
open-vocab gate 不只是 format token suppression 问题。
Phase150 中去格式/字母子集仍不能让 full-vocab argmax 闭合。
很多压制来自 alphabetic_other、target_synonym、punctuation、whitespace 的混合。
因此下一步不能只做格式词元抑制，还要重建真实答案词元集合和同义路径。
```

### 理论进展

Phase150 后，三门语言编码理论应再细化：

```text
semantic support gate:
  内部支持路径形成。

candidate semantic gate:
  小候选集或语义候选集中的相对选择。

surface realization gate:
  在开放词表中选择具体表面 token。
```

其中第三门不是单一门，而至少包括：

```text
format prior
punctuation/whitespace prior
synonym surface route
generic continuation route
category label route
object/context continuation route
tokenization artifact route
```

更准确公式：

```text
token*
=
argmax_v [
  y_semantic(v)
  + y_surface(v | format, prompt_tail)
  + y_continuation(v | local syntax)
  + y_tokenization(v)
]
```

当前实验说明：

```text
y_semantic 已经能让目标类在 candidate4/semantic set 中部分胜出；
但 y_surface + y_continuation + y_tokenization 仍经常压过目标 label token。
```

### 硬伤

1. **Phase150 尚未真正做 2-token/3-token 生成闭合**

```text
本轮做了真实 logits 与 top50 分类，但没有做多步生成。
```

2. **token 分类规则仍是粗粒度启发式**

```text
alphabetic_other 内部可能包含正确同义词、错误类别词、普通续写词、模板词。
需要更细的词表语义分类。
```

3. **target token set 可能过窄**

```text
plant 的 top_arg_class 经常是 target_synonym。
如果目标接受 tree/flower/Plant 等变体，生成闭合评价会不同。
```

4. **format suppression 简单版无效**

```text
final_norm_output_suppress 没有超过 final_norm_output_lm。
说明不能只压一个竞争方向，需要按竞争类型成组处理。
```

5. **Qwen3/GLM4 仍是小范围确认**

```text
跨模型趋势有参考价值，但主结论来自 DS7B 72 cases。
```

### 下一步 Phase151

Phase151 应进入：

```text
Surface-Answer Set and Multi-token Generation Closure
表面答案集合与多词元生成闭合
```

目标：

```text
既然 open-vocab gate 不是单纯格式门，
下一步必须重建模型真实可接受的答案表面集合，而不是只盯单个 category label token。
```

建议测试：

```text
models = qwen3, glm4, deepseek7b

DS7B main:
  categories = plant,time,container,number
  formats = label_colon,multiple_choice,answer_one_word
  template_families = long,short,neutral
  splits = front_back,back_front
  train/test objects = 8/8

surface answer sets:
  canonical label:
    plant, time, container, number

  readout synonyms:
    plant/tree/vegetation/flora
    time/date/period/moment
    container/vessel/box/holder
    number/amount/quantity/count

  object-near valid answers:
    flower/tree/rose/oak
    morning/year/hour/date
    box/bottle/cup/bag
    one/two/count/quantity

  format variants:
    leading space
    capitalized token
    article + label
    label + punctuation

measure:
  one-token argmax within expanded answer set
  two-token greedy generation contains answer surface
  three-token greedy generation contains answer surface
  target answer set rank
  wrong semantic answer rate
  format-first then answer-second rate

interventions:
  support_only
  final_norm_output_lm
  expanded-answer-set scoring
  optional constrained generation over answer surface tokens
```

判据：

```text
如果 expanded answer set 让 plant/time/container/number 显著闭合：
  当前瓶颈主要是 answer surface set 定义过窄。

如果 2-token/3-token 成功但 1-token 失败：
  语言生成闭合应从单 token 改为短序列路径。

如果 expanded set 和多词元仍失败：
  open-vocab gate 仍需要更深的格式/续写机制分解。
```

## Phase 151: Surface-Answer Set and Generation Closure 表面答案集合与生成闭合 [2026-06-15 14:36]

### 本阶段目标

根据附件对 Phase150 的分析继续推进。

附件正确部分：

```text
Phase150 正确地把 token selection failure 进一步定位到 surface realization gate。
当前失败不是模型完全没有内部语义，也不完全是候选语义类别失败。
失败主要发生在从候选语义优势到开放词表表面答案的转换。
target label token 过窄可能导致误判。
下一步必须测试 surface answer set 和生成闭合。
```

需要谨慎修正：

```text
Phase151 本轮先完成 one-token expanded surface set 的真实 logits 审计。
多词元部分只做了保守的 one-step greedy surface proxy，没有完成真正 2-token/3-token iterative generation。
因此不能把本轮解释为完整多词元生成闭合测试。
```

### 脚本

```text
tests/gpt5/phase151_surface_answer_generation_closure_cuda.py
tests/gpt5/phase151_surface_answer_generation_closure_summary.py
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase151_surface_answer_generation_closure_cuda.py \
  tests/gpt5/phase151_surface_answer_generation_closure_summary.py

python tests/gpt5/phase151_surface_answer_generation_closure_cuda.py qwen3 \
  --categories plant,time \
  --template-families long,neutral \
  --splits front_back \
  --formats label_colon \
  --train-objects 8 --test-objects 8 --batch-size 16 --rank 8 \
  --output-dir results/gpt5_phase151_surface_answer_generation_closure \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase151_surface_answer_generation_closure_cuda.py glm4 \
  --categories plant,time \
  --template-families long,neutral \
  --splits front_back \
  --formats label_colon \
  --train-objects 8 --test-objects 8 --batch-size 16 --rank 8 \
  --output-dir results/gpt5_phase151_surface_answer_generation_closure \
  --hard-exit-after-model

python tests/gpt5/phase151_surface_answer_generation_closure_cuda.py deepseek7b \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats label_colon,multiple_choice,answer_one_word \
  --train-objects 8 --test-objects 8 --batch-size 16 --rank 8 \
  --output-dir results/gpt5_phase151_surface_answer_generation_closure \
  --hard-exit-after-model

python tests/gpt5/phase151_surface_answer_generation_closure_summary.py
```

### 重要修正

第一次 Phase151 脚本中，`clean` 基线错误地调用了 remove+0 support 路径，不是真正 clean forward。

已修正为：

```text
clean = 原始 forward logits；
support_only = remove pre-answer projection + support restore；
final_norm_output_lm = support restore + final_norm_output LM-head direction。
```

三模型均使用修正版重新运行。

### 结果文件

```text
results/gpt5_phase151_surface_answer_generation_closure/phase151_qwen3_surface_answer_generation_closure.json
results/gpt5_phase151_surface_answer_generation_closure/phase151_glm4_surface_answer_generation_closure.json
results/gpt5_phase151_surface_answer_generation_closure/phase151_deepseek7b_surface_answer_generation_closure.json
results/gpt5_phase151_surface_answer_generation_closure/phase151_cross_model_summary.md
```

### 测试范围

```text
Qwen3:
  categories = plant,time
  template_families = long,neutral
  split = front_back
  format = label_colon
  train/test objects = 8/8
  cases = 4

GLM4:
  categories = plant,time
  template_families = long,neutral
  split = front_back
  format = label_colon
  train/test objects = 8/8
  cases = 4

DS7B main:
  categories = plant,time,container,number
  template_families = long,short,neutral
  splits = front_back,back_front
  formats = label_colon,multiple_choice,answer_one_word
  train/test objects = 8/8
  cases = 72
```

surface answer sets：

```text
canonical:
  plant / time / container / number

synonyms:
  CATEGORY_READOUT_WORDS 中的 readout synonyms

object_near:
  plant: flower, tree, rose, oak, pine, flora, vegetation
  time: morning, year, hour, date, period, moment, day
  container: box, bottle, cup, jar, vessel, holder, bag
  number: number, digit, amount, quantity, count, integer, one

format_variants:
  leading space
  capitalized
  label + punctuation
  article + label

option_like:
  A / A. / option A 等，仅 multiple_choice 使用
```

### 客观结果

#### Qwen3

```text
clean:
  expanded_argmax = 0.156
  expanded_rank = 11.8
  canonical_rank = 1200.2
  synonym_rank = 1182.4
  object_near_rank = 1935.4
  good_greedy_proxy = 0.156

support_only:
  expanded_argmax = 0.000
  expanded_rank = 279.0
  canonical_rank = 140872.8
  synonym_rank = 90267.9
  object_near_rank = 58010.8

final_norm_output_lm:
  expanded_argmax = 0.000
  expanded_rank = 901.4
  canonical_rank = 101752.9
  synonym_rank = 37920.4
  object_near_rank = 34404.4
```

Qwen3 的 expanded surface set 没有闭合，且 intervention 后比 clean 更差。

#### GLM4 bf16

```text
clean:
  expanded_argmax = 0.062
  expanded_rank = 17.4
  canonical_rank = 274.6
  synonym_rank = 269.6

support_only:
  expanded_argmax = 0.031
  expanded_rank = 18.0
  canonical_rank = 338.7
  synonym_rank = 322.5

final_norm_output_lm:
  expanded_argmax = 0.094
  expanded_rank = 12.5
  canonical_rank = 94.4
  synonym_rank = 90.8
```

GLM4 有小幅改善，但没有接近生成闭合。

#### DS7B overall

```text
clean:
  expanded_argmax = 0.250
  expanded_rank = 27.5
  canonical_rank = 6961.6
  synonym_rank = 4363.7
  object_near_rank = 3311.3
  good_greedy_proxy = 0.295

support_only:
  expanded_argmax = 0.163
  expanded_rank = 1472.7
  canonical_rank = 25581.5
  synonym_rank = 11108.8
  object_near_rank = 8178.6
  good_greedy_proxy = 0.245

final_norm_output_lm:
  expanded_argmax = 0.229
  expanded_rank = 1090.9
  canonical_rank = 16626.5
  synonym_rank = 4334.7
  object_near_rank = 3681.3
  good_greedy_proxy = 0.300
```

关键事实：

```text
expanded surface set 让 clean baseline 已经有 0.25 argmax；
但 support_only 和 final_norm_output_lm 没有超过 clean；
final_norm_output_lm 比 support_only 好，但仍明显弱于 clean 的 expanded rank。
```

#### DS7B by category

```text
plant:
  clean_exp_arg = 0.33
  support_exp_arg = 0.28
  final_exp_arg = 0.38
  final_exp_rank = 6.7
  final_canonical_rank = 178.2
  final_synonym_rank = 101.8
  good_greedy_proxy = 0.40
  top_class = canonical

container:
  clean_exp_arg = 0.28
  support_exp_arg = 0.12
  final_exp_arg = 0.15
  final_exp_rank = 950.6
  final_canonical_rank = 29464.9
  final_synonym_rank = 3717.4
  good_greedy_proxy = 0.19
  top_class = format_only

number:
  clean_exp_arg = 0.19
  support_exp_arg = 0.10
  final_exp_arg = 0.19
  final_exp_rank = 735.6
  final_canonical_rank = 12310.4
  final_synonym_rank = 6494.7
  good_greedy_proxy = 0.32
  top_class = other

time:
  clean_exp_arg = 0.20
  support_exp_arg = 0.15
  final_exp_arg = 0.19
  final_exp_rank = 2670.8
  final_canonical_rank = 24552.4
  final_synonym_rank = 7025.0
  good_greedy_proxy = 0.29
  top_class = format_only
```

#### DS7B by format

```text
multiple_choice:
  clean_exp_arg = 0.66
  support_exp_arg = 0.44
  final_exp_arg = 0.56
  final_exp_rank = 1624.4
  good_greedy_proxy = 0.77
  top_class = canonical

label_colon:
  clean_exp_arg = 0.05
  support_exp_arg = 0.03
  final_exp_arg = 0.10
  final_exp_rank = 605.3
  good_greedy_proxy = 0.10
  top_class = other

answer_one_word:
  clean_exp_arg = 0.04
  support_exp_arg = 0.02
  final_exp_arg = 0.03
  final_exp_rank = 1043.1
  good_greedy_proxy = 0.03
  top_class = format_only
```

### 关键现象

1. **expanded surface set 不足以全局闭合**

DS7B final_exp_arg 只有：

```text
0.229
```

即使放宽到 canonical + synonyms + object-near + format variants + option-like，整体仍未闭合。

2. **clean baseline 在表面答案集合上强于 intervention**

DS7B：

```text
clean_exp_arg = 0.250
support_exp_arg = 0.163
final_exp_arg = 0.229
```

这说明当前 support restore 路径虽然能增强候选语义门，但会破坏一部分自然 surface realization。

3. **multiple_choice 是当前最强表面闭合格式**

DS7B：

```text
multiple_choice clean_exp_arg = 0.66
multiple_choice final_exp_arg = 0.56
good_greedy_proxy = 0.77
top_class = canonical
```

相比：

```text
label_colon final_exp_arg = 0.10
answer_one_word final_exp_arg = 0.03
```

说明模型在 multiple_choice 下更容易进入明确表面答案路径。

4. **plant 仍是最接近闭合类别**

DS7B plant：

```text
final_exp_arg = 0.38
final_exp_rank = 6.7
good_greedy_proxy = 0.40
top_class = canonical
```

plant 扩展表面集合后更接近真实表面闭合，但仍远非稳定闭合。

5. **container/time 主要受 format_only 干扰**

DS7B：

```text
container top_class = format_only
time top_class = format_only
```

这说明不同类别的 surface gate 不同：

```text
plant 更像答案表面集合问题；
container/time 更像格式/续写路径问题；
number 更像抽象/普通词竞争问题。
```

### 对附件分析的判断

附件正确：

```text
Phase150 的判断正确；
surface answer set 是必要下一步；
plant 是主突破口；
单 canonical label token 会低估真实表面路径。
```

需要修正：

```text
expanded surface answer set 并没有让整体显著闭合。
真实结果显示：clean 自然表面路径比当前 intervention 更好。
因此问题不是简单“目标答案集合太窄”，而是 support restore 与自然 surface realization 没有对齐。
```

### 理论进展

Phase151 后，理论需要再拆一层：

```text
semantic candidate closure
  不等于
surface answer closure

surface answer closure
  还依赖 format-conditioned natural realization path。
```

当前可更新为：

```text
object/template/context field
  -> support restore
  -> candidate semantic advantage
  -> format-conditioned surface path
  -> expanded answer set selection
  -> actual generation
```

关键修正：

```text
support restore 不是越强越接近真实输出。
它可能恢复语义候选，但同时扰乱自然表面生成路径。
```

这说明语言生成机制不是单向“语义增强 -> 输出增强”，而是：

```text
语义支持和表面实现必须相互对齐。
```

### 硬伤

1. **没有完成真正 2-token/3-token iterative generation**

```text
本轮 greedy_class 是 one-step argmax 的表面分类代理。
不能证明多词元生成是否闭合。
```

2. **expanded surface set 仍可能不完整**

```text
尤其是 time、number、container 的自然答案空间可能更宽。
```

3. **intervention 可能破坏自然生成轨迹**

```text
support_only 和 final_norm_output_lm 在 expanded surface 上常弱于 clean。
这说明当前干预不等于自然机制。
```

4. **multiple_choice 的强结果可能来自提示格式本身**

```text
multiple_choice clean 已经很强，不能证明 support path 本身打开了生成闭合。
```

5. **Qwen3/GLM4 仍是小范围对照**

```text
主证据仍来自 DS7B 72 cases。
```

### 下一步 Phase152

Phase152 应进入：

```text
Natural Surface Path Preservation and Iterative Generation Closure
自然表面路径保持与迭代生成闭合
```

目标：

```text
既然 current support restore 会损伤自然 surface realization，
下一步必须同时保留自然表面路径并测试真正多词元生成。
```

建议测试：

```text
1. 不做 pre-answer removal，只做 additive support steering。
2. 对比：
   clean
   remove
   remove+restore
   additive_support_only
   final_norm_output_lm
   additive_support + final_norm_output_lm

3. 真正 iterative generation：
   step1 forward -> choose token -> append
   step2 forward -> choose token -> append
   step3 forward -> choose token -> append

4. 每一步都记录：
   surface answer set hit
   format-first then answer-second
   canonical/synonym/object_near/option_like
   wrong semantic answer
   fragment/format-only path

5. 主攻：
   DS7B plant multiple_choice / label_colon
   DS7B time/container format-only cases
   GLM4 plant/time label_colon
```

判据：

```text
如果 additive_support 保留 clean 的 surface path，又提升语义答案命中：
  说明 remove+restore 破坏了自然生成轨迹。

如果 iterative 2/3 token 比 one-step 明显更好：
  生成闭合应定义为短序列路径。

如果 multiple_choice 仍主要靠 clean，而干预无增益：
  当前机制只能解释候选分类，不能解释自然生成。
```

## Phase 152: Semantic-Format Path and Iterative Surface Generation 语义-格式路径与迭代表面生成 [2026-06-15 16:45]

### 本阶段目标

综合两个附件继续任务。

附件一关于标点/格式路径的批评基本正确：

```text
标点、空格、换行、选项字母、冒号等不应只当作噪声；
它们可能代表 format/syntax path；
语言生成不是单一语义路径，而是 semantic path、format-syntax path、surface realization path、tokenization path 的合流竞争。
```

需要谨慎修正：

```text
标点/格式 token 不一定完全跳过语义；
更准确是它们主要受 format/syntax/continuation path 控制，而不是主要受 category semantic path 控制。
```

附件二对 Phase151 的判断基本正确：

```text
expanded surface answer set 是必要但不充分；
remove+restore 可能破坏 natural surface realization；
下一步必须测试 additive support 和真实 iterative generation。
```

本轮 Phase152 目标：

```text
1. 比较 clean、remove、remove+restore、additive_support、additive_support_lm。
2. 测试 additive support 是否比 remove+restore 更保留自然表面路径。
3. 做真实 3-step iterative greedy generation：
   step1 forward -> argmax token -> append
   step2 forward -> argmax token -> append
   step3 forward -> argmax token -> append
4. 记录 surface answer set hit 与 format-first-answer-later。
```

### 脚本

```text
tests/gpt5/phase152_natural_surface_iterative_generation_cuda.py
tests/gpt5/phase152_natural_surface_iterative_generation_summary.py
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase152_natural_surface_iterative_generation_cuda.py \
  tests/gpt5/phase152_natural_surface_iterative_generation_summary.py

python tests/gpt5/phase152_natural_surface_iterative_generation_cuda.py qwen3 \
  --categories plant,time \
  --template-families long,neutral \
  --splits front_back \
  --formats label_colon \
  --train-objects 8 --test-objects 8 --batch-size 16 --rank 8 \
  --add-scales 0.05,0.2 \
  --steps 3 \
  --output-dir results/gpt5_phase152_natural_surface_iterative_generation \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase152_natural_surface_iterative_generation_cuda.py glm4 \
  --categories plant,time \
  --template-families long,neutral \
  --splits front_back \
  --formats label_colon \
  --train-objects 8 --test-objects 8 --batch-size 16 --rank 8 \
  --add-scales 0.05,0.2 \
  --steps 3 \
  --output-dir results/gpt5_phase152_natural_surface_iterative_generation \
  --hard-exit-after-model

python tests/gpt5/phase152_natural_surface_iterative_generation_cuda.py deepseek7b \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats label_colon,multiple_choice,answer_one_word \
  --train-objects 8 --test-objects 8 --batch-size 16 --rank 8 \
  --add-scales 0.05,0.1,0.2,0.5 \
  --steps 3 \
  --output-dir results/gpt5_phase152_natural_surface_iterative_generation \
  --hard-exit-after-model

python tests/gpt5/phase152_natural_surface_iterative_generation_summary.py
```

### 结果文件

```text
results/gpt5_phase152_natural_surface_iterative_generation/phase152_qwen3_natural_surface_iterative_generation.json
results/gpt5_phase152_natural_surface_iterative_generation/phase152_glm4_natural_surface_iterative_generation.json
results/gpt5_phase152_natural_surface_iterative_generation/phase152_deepseek7b_natural_surface_iterative_generation.json
results/gpt5_phase152_natural_surface_iterative_generation/phase152_cross_model_summary.md
```

### 测试范围

```text
Qwen3:
  categories = plant,time
  template_families = long,neutral
  split = front_back
  format = label_colon
  train/test objects = 8/8
  cases = 4
  add_scales = 0.05,0.2

GLM4:
  categories = plant,time
  template_families = long,neutral
  split = front_back
  format = label_colon
  train/test objects = 8/8
  cases = 4
  add_scales = 0.05,0.2

DS7B main:
  categories = plant,time,container,number
  template_families = long,short,neutral
  splits = front_back,back_front
  formats = label_colon,multiple_choice,answer_one_word
  train/test objects = 8/8
  cases = 72
  add_scales = 0.05,0.1,0.2,0.5
```

### 客观结果

#### Qwen3

```text
cases = 4

clean:
  hit_rate = 0.188
  format_first_answer_later = 0.000

remove:
  hit_rate = 0.219
  format_first_answer_later = 0.000

remove_restore:
  hit_rate = 0.000
  format_first_answer_later = 0.000

best_additive:
  hit_rate = 0.406
  format_first_answer_later = 0.000
  dominant scale = 0.05
```

Qwen3 中 additive support 明显优于 remove+restore。

#### GLM4 bf16

```text
cases = 4

clean:
  hit_rate = 0.156

remove:
  hit_rate = 0.125

remove_restore:
  hit_rate = 0.031

best_additive:
  hit_rate = 0.281
  dominant scale = 0.05
```

GLM4 也显示 additive support 优于 remove+restore。

#### DS7B overall

```text
cases = 72

clean:
  hit_rate = 0.330
  format_first_answer_later = 0.014

remove:
  hit_rate = 0.323
  format_first_answer_later = 0.007

remove_restore:
  hit_rate = 0.255
  format_first_answer_later = 0.002

best_additive:
  hit_rate = 0.438
  format_first_answer_later = 0.024
  dominant scale = 0.05
```

核心客观事实：

```text
best_additive > clean > remove_restore
```

这支持 Phase151 的怀疑：remove+restore 会破坏一部分自然表面路径，而小尺度 additive support 更像保留自然生成轨迹的干预。

#### DS7B by category

```text
container:
  clean_hit = 0.32
  remove_restore_hit = 0.18
  best_add_hit = 0.39
  best_fmt_later = 0.05

number:
  clean_hit = 0.31
  remove_restore_hit = 0.27
  best_add_hit = 0.43
  best_fmt_later = 0.01

plant:
  clean_hit = 0.38
  remove_restore_hit = 0.33
  best_add_hit = 0.56
  best_fmt_later = 0.01

time:
  clean_hit = 0.31
  remove_restore_hit = 0.24
  best_add_hit = 0.38
  best_fmt_later = 0.03
```

plant 仍是最强类别：

```text
plant best_add_hit = 0.56
```

但 container/number/time 也有 additive 增益。

#### DS7B by format

```text
multiple_choice:
  clean_hit = 0.88
  remove_restore_hit = 0.71
  best_add_hit = 0.98
  best_fmt_later = 0.02
  clean_class = canonical
  best_class = canonical

label_colon:
  clean_hit = 0.06
  remove_restore_hit = 0.03
  best_add_hit = 0.18
  best_fmt_later = 0.01

answer_one_word:
  clean_hit = 0.06
  remove_restore_hit = 0.03
  best_add_hit = 0.16
  best_fmt_later = 0.05
```

multiple_choice 的 3-step surface hit 接近闭合：

```text
best_add_hit = 0.98
```

但 clean 已经 0.88，说明格式本身是强控制变量。

### 关键现象

1. **additive support 明显优于 remove+restore**

三模型一致：

```text
Qwen3:
  remove_restore 0.000 -> best_additive 0.406

GLM4:
  remove_restore 0.031 -> best_additive 0.281

DS7B:
  remove_restore 0.255 -> best_additive 0.438
```

这说明：

```text
remove+restore 适合证明内部成分因果必要性；
但不适合直接代表自然生成机制。
```

2. **小尺度 additive support 最稳定**

DS7B dominant scale：

```text
0.05
```

这说明自然表面路径需要最小扰动，强注入未必更好。

3. **真正 3-step generation 比 one-step proxy 更能暴露 surface closure**

Phase151 DS7B final expanded argmax：

```text
0.229
```

Phase152 DS7B best additive 3-step hit：

```text
0.438
```

这说明单步指标低估了部分短生成闭合。

4. **format-first answer-later 路径并不常见**

DS7B：

```text
best_additive format_first_answer_later = 0.024
```

这修正了关于标点/格式路径的一个可能误读：

```text
格式路径很重要，但本轮没有证明主机制是“先格式 token，后答案 token”。
更多情况是格式约束改变整体生成轨迹，而不是简单两步顺序。
```

5. **multiple_choice 是最强表面闭合格式**

DS7B：

```text
multiple_choice best_add_hit = 0.98
```

这证明 format/syntax path 可以极大降低开放词表生成难度。

但 clean 已经：

```text
multiple_choice clean_hit = 0.88
```

所以 multiple_choice 成功主要来自格式自身，而 additive support 只是进一步提升。

### 对附件分析的判断

正确部分：

```text
Phase151 分析正确；
remove+restore 可能破坏自然表面路径；
additive support 必须测试；
真正 iterative generation 是必要的；
标点/格式路径不是噪声，应作为独立研究对象。
```

需要修正部分：

```text
format-first answer-second 路径并不高。
因此标点/格式路径的重要性不一定表现为“先输出标点/空格再输出答案”。
它更可能表现为：
  模板/格式先验整体改变可选 surface route；
  multiple_choice 直接压缩答案空间；
  label_colon / answer_one_word 仍容易进入解释/续写/碎片路径。
```

### 理论进展

Phase152 后，理论应从：

```text
remove+restore support mechanism
```

分裂成两个实验范式：

```text
1. necessity intervention:
   remove / remove+restore
   用于证明某成分是否因果相关。

2. natural-path steering:
   additive_support
   用于测试是否能在不破坏表面路径的情况下增强目标生成。
```

语言生成结构进一步更新为：

```text
context field
  -> semantic support route
  -> format/syntax route
  -> natural surface realization route
  -> iterative token trajectory
```

当前最可靠新事实：

```text
小尺度 additive support 可以提升真实 3-step surface hit；
remove+restore 会显著损伤 surface generation；
multiple_choice 的格式约束是最强自然表面路径控制器；
format-first answer-later 不是主要现象。
```

### 硬伤

1. **additive support 使用的仍是 Phase147 train-selected route**

```text
router generalization 问题仍未解决。
```

2. **格式路径尚未被直接定位**

```text
Phase152 证明 format 影响很大，但没有找到独立 format-syntax subspace。
```

3. **surface hit 的语义分类仍然较粗**

```text
有些 examples 包含错误类别、解释文本、混合答案。
需要更精细地判定正确/错误 semantic hit。
```

4. **multiple_choice 可能过度简化开放生成**

```text
它接近闭合，但不是一般自然语言生成。
```

5. **Qwen3/GLM4 范围仍小**

```text
主结论来自 DS7B 72 cases。
```

### 下一步 Phase153

Phase153 应进入：

```text
Format-Syntax Subspace Localization and Joint Steering
格式句法子空间定位与联合引导
```

目标：

```text
既然 additive semantic support 有效，
而 format/syntax path 也明显影响 surface realization，
下一步要直接定位 format-syntax subspace，并测试 semantic + format 联合引导。
```

建议测试：

```text
1. 语义不变，格式变化：
   same object/category, different prompt tails:
     label_colon
     answer_one_word
     multiple_choice
     quoted_answer
     list_answer

2. 格式不变，语义变化：
   same tail, categories:
     plant,time,container,number

3. 构造 format target token groups：
   whitespace
   newline
   colon
   period
   quote
   option labels A/B/C/D

4. 建立 format contrast basis：
   format centers vs semantic centers
   measure subspace overlap

5. 干预：
   semantic_additive only
   format_additive only
   semantic + format additive

6. 输出：
   3-step generation hit
   format token rank
   semantic answer hit
   wrong semantic hit
```

判据：

```text
如果 format basis 与 semantic basis overlap 低，且 format steering 主要影响空格/标点/选项标签：
  说明格式路径可分离。

如果 semantic + format 联合引导超过 semantic only：
  说明生成闭合需要多路径合流。

如果 format basis 不可分离：
  标点/格式可能主要是 LM-head/tokenization 层偏置，而非内部独立路径。
```

## Phase 153: Format-Syntax Subspace Localization and Joint Steering 格式句法子空间定位与联合引导 [2026-06-15 17:48]

### 本阶段目标

根据用户要求，分析 Phase152 附加判断是否正确，并继续完成任务，同时加入标点符号部分分析。

对附件判断的结论：

```text
Phase152 的核心判断正确：
  remove+restore 更适合证明必要性；
  additive support 更接近自然生成轨迹；
  真实 3-step generation 比 one-step proxy 更合理；
  标点/空格/换行/选项标签不是噪声，应进入独立分析。

需要修正的部分：
  Phase152 没有证明主机制是“先输出标点/格式 token，再输出答案 token”。
  format/syntax path 更像整体 surface route 约束，而不是简单顺序 token 现象。
```

因此 Phase153 直接测试：

```text
1. semantic answer basis 与 format/syntax basis 的重叠。
2. 标点/空白/换行/引号/列表符号/选项标签 token 的 rank 与 argmax 分布。
3. semantic-only、format-only、semantic+format joint steering 的真实 3-step generation hit。
```

### 重要修正

初次运行 Phase153 后发现：

```text
DS7B 覆盖 120 cases；
但 qwen3 和 GLM4 只有 4 cases。
```

原因是 qwen3/GLM4 依赖的 Phase147 router 文件此前不是全范围结果。为避免小样本误判，先补跑 qwen3/GLM4 的 Phase147 全范围 router，再重跑 Phase153。

### 生成脚本

```text
tests/gpt5/phase153_format_syntax_subspace_joint_steering_cuda.py
tests/gpt5/phase153_format_syntax_subspace_joint_steering_summary.py
```

补全依赖 router 的脚本：

```text
tests/gpt5/phase147_train_router_format_token_cuda.py
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase153_format_syntax_subspace_joint_steering_cuda.py \
  tests/gpt5/phase153_format_syntax_subspace_joint_steering_summary.py

python tests/gpt5/phase153_format_syntax_subspace_joint_steering_cuda.py qwen3 \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats label_colon,multiple_choice,answer_one_word,quoted_answer,list_answer \
  --train-objects 8 \
  --test-objects 8 \
  --batch-size 16 \
  --rank 8 \
  --format-rank 4 \
  --semantic-scale 0.05 \
  --format-scales 0.05,0.2 \
  --steps 3 \
  --output-dir results/gpt5_phase153_format_syntax_subspace_joint_steering \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase153_format_syntax_subspace_joint_steering_cuda.py glm4 \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats label_colon,multiple_choice,answer_one_word,quoted_answer,list_answer \
  --train-objects 8 \
  --test-objects 8 \
  --batch-size 16 \
  --rank 8 \
  --format-rank 4 \
  --semantic-scale 0.05 \
  --format-scales 0.05,0.2 \
  --steps 3 \
  --output-dir results/gpt5_phase153_format_syntax_subspace_joint_steering \
  --hard-exit-after-model

python tests/gpt5/phase153_format_syntax_subspace_joint_steering_cuda.py deepseek7b \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats label_colon,multiple_choice,answer_one_word,quoted_answer,list_answer \
  --train-objects 8 \
  --test-objects 8 \
  --batch-size 16 \
  --rank 8 \
  --format-rank 4 \
  --semantic-scale 0.05 \
  --format-scales 0.05,0.2 \
  --steps 3 \
  --output-dir results/gpt5_phase153_format_syntax_subspace_joint_steering \
  --hard-exit-after-model

python tests/gpt5/phase147_train_router_format_token_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 8 \
  --batch-size 16 \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats plain,label_colon,answer_one_word,multiple_choice \
  --layer-offsets 0,-1 \
  --sites input_answer,attention_output,mlp_input \
  --scales 0.2,0.35,0.5,0.75,1.0 \
  --output-dir results/gpt5_phase147_train_router_format_token \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase147_train_router_format_token_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 8 \
  --batch-size 16 \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats plain,label_colon,answer_one_word,multiple_choice \
  --layer-offsets 0,-1 \
  --sites input_answer,attention_output,mlp_input \
  --scales 0.2,0.35,0.5,0.75,1.0 \
  --output-dir results/gpt5_phase147_train_router_format_token \
  --hard-exit-after-model

python tests/gpt5/phase153_format_syntax_subspace_joint_steering_cuda.py qwen3 ...
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase153_format_syntax_subspace_joint_steering_cuda.py glm4 ...

python tests/gpt5/phase153_format_syntax_subspace_joint_steering_summary.py

python -m py_compile \
  tests/gpt5/phase147_train_router_format_token_cuda.py \
  tests/gpt5/phase153_format_syntax_subspace_joint_steering_cuda.py \
  tests/gpt5/phase153_format_syntax_subspace_joint_steering_summary.py
```

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = plant, time, container, number
template families = long, short, neutral
splits = front_back, back_front
formats = label_colon, multiple_choice, answer_one_word, quoted_answer, list_answer
train objects/category = 8
heldout test objects/category = 8
templates: train [0,1], heldout [2]
cases/model = 120
generation steps = 3
semantic additive scale = 0.05
format scales = 0.05, 0.2
format token groups = whitespace, newline, colon, period, quote, option_label, list_marker
```

### 输出文件

```text
results/gpt5_phase153_format_syntax_subspace_joint_steering/phase153_qwen3_format_syntax_subspace_joint_steering.json
results/gpt5_phase153_format_syntax_subspace_joint_steering/phase153_glm4_format_syntax_subspace_joint_steering.json
results/gpt5_phase153_format_syntax_subspace_joint_steering/phase153_deepseek7b_format_syntax_subspace_joint_steering.json
results/gpt5_phase153_format_syntax_subspace_joint_steering/phase153_cross_model_summary.md
```

### 客观结果

整体结果：

```text
Qwen3:
  cases = 120
  clean = 0.367
  semantic_additive = 0.326
  format_internal = 0.374
  format_lm = 0.270
  best_joint = 0.411
  semantic-format overlap max avg = 0.495

GLM4:
  cases = 120
  clean = 0.298
  semantic_additive = 0.286
  format_internal = 0.297
  format_lm = 0.285
  best_joint = 0.349
  semantic-format overlap max avg = 0.264

DS7B:
  cases = 120
  clean = 0.254
  semantic_additive = 0.227
  format_internal = 0.249
  format_lm = 0.136
  best_joint = 0.265
  semantic-format overlap max avg = 0.436
```

按格式结果：

```text
Qwen3:
  answer_one_word: clean 0.135, sem 0.208, joint 0.271
  label_colon: clean 0.292, sem 0.240, joint 0.333
  list_answer: clean 0.266, sem 0.177, joint 0.266
  multiple_choice: clean 0.979, sem 0.870, joint 0.964
  quoted_answer: clean 0.161, sem 0.135, joint 0.224

GLM4:
  answer_one_word: clean 0.073, sem 0.068, joint 0.156
  label_colon: clean 0.177, sem 0.172, joint 0.266
  list_answer: clean 0.161, sem 0.130, joint 0.229
  multiple_choice: clean 0.979, sem 0.974, joint 0.979
  quoted_answer: clean 0.099, sem 0.089, joint 0.115

DS7B:
  answer_one_word: clean 0.057, sem 0.031, joint 0.062
  label_colon: clean 0.057, sem 0.089, joint 0.125
  list_answer: clean 0.188, sem 0.115, joint 0.141
  multiple_choice: clean 0.875, sem 0.818, joint 0.875
  quoted_answer: clean 0.094, sem 0.083, joint 0.120
```

标点/格式词元结果：

```text
Qwen3 best_joint top format groups:
  other 101/120
  quote 11/120
  whitespace 5/120
  option_label 3/120

GLM4 best_joint top format groups:
  other 118/120
  whitespace 2/120

DS7B best_joint top format groups:
  other 92/120
  whitespace 26/120
  option_label 1/120
  list_marker 1/120
```

format rank 与 answer rank：

```text
Qwen3:
  multiple_choice answer_rank 1.2, format_rank 3.5
  list_answer format_rank 1.8, answer_rank 4.2

GLM4:
  multiple_choice answer_rank 1.2, format_rank 13.6
  list_answer format_rank 6.3, answer_rank 16.9

DS7B:
  multiple_choice answer_rank 2.1, format_rank 3.3
  label_colon answer_rank 31.0, format_rank 63.0
  quoted_answer answer_rank 31.5, format_rank 40.3
```

### 当前最可靠事实

1. **joint steering 比 semantic-only 更稳定**

三模型全范围都出现：

```text
Qwen3: semantic 0.326 -> joint 0.411
GLM4: semantic 0.286 -> joint 0.349
DS7B: semantic 0.227 -> joint 0.265
```

这支持：

```text
surface generation closure 不是单一路径完成；
语义支持需要与格式/表面路径合流。
```

2. **format-only internal steering 接近 clean，但单独不打开答案**

```text
Qwen3: clean 0.367, format_internal 0.374
GLM4: clean 0.298, format_internal 0.297
DS7B: clean 0.254, format_internal 0.249
```

这说明格式内部方向能保持或微调自然表面轨迹，但不是单独语义答案路径。

3. **format LM steering 单独较弱，尤其 DS7B**

```text
Qwen3 format_lm 0.270
GLM4 format_lm 0.285
DS7B format_lm 0.136
```

这反驳了一个简单解释：

```text
只要推高标点/空格/选项 token，就能打开答案。
```

结果不是这样。标点符号路径不是单个 token boost。

4. **multiple_choice 仍然是最强表面闭合格式**

```text
Qwen3 multiple_choice clean 0.979, joint 0.964
GLM4 multiple_choice clean 0.979, joint 0.979
DS7B multiple_choice clean 0.875, joint 0.875
```

这继续证明：

```text
format prior 可以大幅压缩开放词表竞争空间。
```

但 clean 已经极高，所以它不是语义机制闭合的充分证据。

5. **quoted/list 格式提供了新的标点现象，但不稳定**

```text
quoted_answer 在 Qwen3/GLM4/DS7B 都低：
  0.224 / 0.115 / 0.120

list_answer:
  Qwen3 0.266
  GLM4 0.229
  DS7B 0.141
```

说明引号、列表符号会改变 surface route，但没有形成稳定答案闭合。

6. **semantic-format overlap 不是零**

```text
Qwen3 overlap max avg = 0.495
GLM4 overlap max avg = 0.264
DS7B overlap max avg = 0.436
```

这说明格式路径与语义路径不是完全独立的正交通道。更谨慎的说法是：

```text
格式/标点路径可测、可影响生成；
但它与语义支持路径有共享接口或混合区域。
```

### 对标点符号机制的判断

当前结果支持：

```text
标点、空格、引号、列表符号、选项标签不是普通噪声。
它们是 surface route 的一部分。
```

但当前结果不支持：

```text
标点路径 = 独立于语义路径的简单格式 token 通道。
```

更符合现象的表述是：

```text
标点/格式是一类表面轨迹约束因子。
它既影响开放词表竞争，也影响后续 token 的可达状态。
但这种影响通常不是单步标点 token argmax，而是通过 prompt format、answer site state、LM head 竞争共同产生。
```

### 理论进展

Phase153 后，当前机制图从：

```text
semantic support route
  -> surface generation
```

推进为：

```text
context/object field
  -> semantic support route
  -> format/syntax constraint route
  -> answer-site surface state
  -> open-vocab competitor gate
  -> iterative token trajectory
```

其中：

```text
semantic support route:
  提供类别/对象相关支持，但单独不足以稳定开放生成。

format/syntax constraint route:
  改变输出空间、标点倾向、列表/引号/选项结构，但单独不等于答案。

joint steering:
  是目前最接近“多路径合流”的实验证据。
```

### 硬伤与问题

1. **format basis 是对比中心，不是直接找到真实格式电路**

当前 format basis 来自不同 prompt format 的 answer_vec center contrast。它能测“格式表征差异”，但还没有定位到具体 attention head / MLP writer。

2. **format token groups 仍然粗**

```text
whitespace/newline/colon/period/quote/option_label/list_marker
```

这些是人工分组，可能遗漏 tokenizer 中大量格式碎片。

3. **joint gain 不大**

```text
Qwen3 +0.085
GLM4 +0.063
DS7B +0.038
```

说明 joint steering 方向正确，但远没有机制闭合。

4. **multiple_choice 过强，容易掩盖机制**

multiple_choice clean 本身很高，应当作为格式上界或控制条件，而不是自然生成结论来源。

5. **semantic-format overlap 不低**

这说明不能简单宣称“语义路径和格式路径完全分离”。当前更像共享接口上的混合通道。

6. **DS7B 效果弱于 Qwen3/GLM4**

DS7B 的 joint 只有 0.265，说明它的开放词表竞争和表面生成缺口更深。

### 下一步任务

Phase154 应继续客观推进：

```text
Format Writer Localization and Surface Gate Closure
格式写入器定位与表面门闭合
```

目标不是再扩大格式种类，而是定位：

```text
哪些 attention heads / MLP blocks 写入 format/syntax constraint；
这些 writer 是否与 semantic support writer 合流；
joint steering 的增益来自哪一层、哪一类 writer。
```

建议测试：

```text
1. 在 Phase153 的格式条件上，对 final 4 layers 做 attention/MLP writer ablation。
2. 对 whitespace/quote/list_marker/option_label 的 rank 做 writer-level causal test。
3. 比较 semantic writer、format writer、joint writer 三类集合。
4. 对 DS7B 优先测试 label_colon、answer_one_word、quoted_answer，因为这些是瓶颈格式。
5. multiple_choice 只作为上界控制。
```

判据：

```text
如果某些 writer ablation 明显破坏 format rank，但不破坏 semantic answer rank：
  说明存在相对独立的格式写入器。

如果 writer 同时影响 format rank 与 answer hit：
  说明它是 surface gate 的共享接口。

如果找不到稳定 writer：
  format/syntax 更可能是分布式 LM-head/tokenization 竞争，而不是局部电路。
```

## Phase 154: Format Writer Surface Gate Localization 格式写入器与表面门定位 [2026-06-15 18:22]

### 本阶段目标

根据用户附加分析，Phase153 的判断基本正确，但需要继续收紧：

```text
Phase153 证明 format/syntax path 参与 surface generation；
但没有证明它是完全独立的 format circuit。
```

因此 Phase154 不再只测试 format basis 是否有用，而是进入 writer 级别：

```text
在最后 4 层 attention_output / mlp_output 上，
分别移除 semantic projection、format projection、joint projection，
观察 answer rank 与 format rank 的损伤。
```

核心问题：

```text
哪些 writer 写入 semantic answer support？
哪些 writer 写入 punctuation/format constraint？
二者是否在同一个 surface gate writer 上合流？
```

### 生成脚本

```text
tests/gpt5/phase154_format_writer_surface_gate_cuda.py
tests/gpt5/phase154_format_writer_surface_gate_summary.py
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase154_format_writer_surface_gate_cuda.py \
  tests/gpt5/phase154_format_writer_surface_gate_summary.py

python tests/gpt5/phase154_format_writer_surface_gate_cuda.py qwen3 \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats label_colon,answer_one_word,quoted_answer,list_answer,multiple_choice \
  --train-objects 8 \
  --test-objects 8 \
  --batch-size 16 \
  --rank 8 \
  --format-rank 4 \
  --layer-back 4 \
  --ablate-scale 1.0 \
  --output-dir results/gpt5_phase154_format_writer_surface_gate \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase154_format_writer_surface_gate_cuda.py glm4 \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats label_colon,answer_one_word,quoted_answer,list_answer,multiple_choice \
  --train-objects 8 \
  --test-objects 8 \
  --batch-size 16 \
  --rank 8 \
  --format-rank 4 \
  --layer-back 4 \
  --ablate-scale 1.0 \
  --output-dir results/gpt5_phase154_format_writer_surface_gate \
  --hard-exit-after-model

python tests/gpt5/phase154_format_writer_surface_gate_cuda.py deepseek7b \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats label_colon,answer_one_word,quoted_answer,list_answer,multiple_choice \
  --train-objects 8 \
  --test-objects 8 \
  --batch-size 16 \
  --rank 8 \
  --format-rank 4 \
  --layer-back 4 \
  --ablate-scale 1.0 \
  --output-dir results/gpt5_phase154_format_writer_surface_gate \
  --hard-exit-after-model

python tests/gpt5/phase154_format_writer_surface_gate_summary.py

python -m py_compile \
  tests/gpt5/phase154_format_writer_surface_gate_cuda.py \
  tests/gpt5/phase154_format_writer_surface_gate_summary.py
```

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = plant, time, container, number
template families = long, short, neutral
splits = front_back, back_front
formats = label_colon, answer_one_word, quoted_answer, list_answer, multiple_choice
cases/model = 120
patch layers:
  Qwen3 = L33-L36
  GLM4 = L37-L40
  DS7B = L25-L28
components = attention_output, mlp_output
ablation modes = semantic_proj, format_proj, joint_proj
metric = first-token expanded answer rank / format token rank
```

### 输出文件

```text
results/gpt5_phase154_format_writer_surface_gate/phase154_qwen3_format_writer_surface_gate.json
results/gpt5_phase154_format_writer_surface_gate/phase154_glm4_format_writer_surface_gate.json
results/gpt5_phase154_format_writer_surface_gate/phase154_deepseek7b_format_writer_surface_gate.json
results/gpt5_phase154_format_writer_surface_gate/phase154_cross_model_summary.md
```

### 客观结果

整体 writer 损伤：

```text
Qwen3:
  semantic_proj strongest answer damage avg = +1.70
  format_proj strongest format damage avg = +4.23
  joint_proj strongest answer damage avg = +2.55
  joint_proj strongest format damage avg = +4.86
  top writers:
    semantic answer: L36 mlp_output
    format rank: L36/L35 mlp_output
    joint answer: L36 mlp_output / L33 attention_output

GLM4:
  semantic_proj strongest answer damage avg = +8.21
  format_proj strongest format damage avg = +44.72
  joint_proj strongest answer damage avg = +13.81
  joint_proj strongest format damage avg = +124.52
  top writers:
    semantic answer: L40 mlp_output
    format rank: L39 attention_output / L40 mlp_output
    joint answer: L39 attention_output / L39-L40 mlp_output

DS7B:
  semantic_proj strongest answer damage avg = +14.11
  format_proj strongest format damage avg = +2078.86
  joint_proj strongest answer damage avg = +68.20
  joint_proj strongest format damage avg = +790.36
  top writers:
    semantic answer: L28 mlp_output / L28 attention_output
    format rank: L28 attention_output / L28 mlp_output
    joint answer: L28 mlp_output / L28 attention_output
```

按格式结果：

```text
Qwen3:
  label_colon:
    clean_answer_rank 11.4
    joint_answer_damage +8.0
    joint_format_damage +4.7
  quoted_answer:
    clean_format_rank 19.6
    joint_format_damage +12.6
  multiple_choice:
    clean_answer_rank 1.4
    joint_answer_damage +0.4

GLM4:
  label_colon:
    clean_answer_rank 30.1
    joint_answer_damage +22.0
    joint_format_damage +66.8
  quoted_answer:
    clean_answer_rank 55.4
    clean_format_rank 232.9
    joint_answer_damage +28.8
    joint_format_damage +334.7
  multiple_choice:
    clean_answer_rank 1.4
    joint_answer_damage +0.9

DS7B:
  answer_one_word:
    clean_answer_rank 21.7
    joint_answer_damage +23.8
    joint_format_damage +16.8
  label_colon:
    clean_answer_rank 58.4
    clean_format_rank 102.4
    joint_answer_damage +27.7
    joint_format_damage +77.8
  quoted_answer:
    clean_answer_rank 17.6
    clean_format_rank 669.9
    joint_answer_damage +275.0
    joint_format_damage +3837.8
  multiple_choice:
    clean_answer_rank 2.2
    joint_answer_damage +2.7
```

### 当前最可靠事实

1. **format/surface gate 有 writer 级别因果点**

如果 format projection 只是抽象统计差异，移除 writer 输出中的 format projection 不应稳定损伤 format rank。

但结果显示：

```text
Qwen3 format damage +4.23
GLM4 format damage +44.72
DS7B format damage +2078.86
```

这说明 format/syntax constraint 不是纯粹后验解释，而是在最后几层 writer 输出中有可测因果成分。

2. **最后一层或倒数第二层是主要 surface gate**

最稳定位置：

```text
Qwen3: L36/L35 mlp_output
GLM4: L39 attention_output, L40 mlp_output
DS7B: L28 attention_output, L28 mlp_output
```

说明表面路径不是早期形成后静态传递，而是在最后层附近完成强门控。

3. **semantic writer 与 format writer 部分合流**

DS7B 最明显：

```text
semantic answer writer:
  L28 mlp_output / L28 attention_output

format writer:
  L28 attention_output / L28 mlp_output

joint answer writer:
  L28 mlp_output / L28 attention_output
```

这支持 Phase153 的修正判断：

```text
format path 与 semantic path 可区分，
但在 answer-site surface gate 附近共享接口。
```

4. **multiple_choice 仍然是上界控制，不是瓶颈**

三模型 multiple_choice 的 clean_answer_rank 已经接近 1：

```text
Qwen3 1.4
GLM4 1.4
DS7B 2.2
```

所以 writer ablation 对 multiple_choice 的 answer damage 较小，不能代表一般开放生成。

5. **quoted_answer 是最强标点/格式瓶颈之一**

尤其 DS7B：

```text
quoted_answer clean_format_rank = 669.9
joint_format_damage = +3837.8
joint_answer_damage = +275.0
```

说明引号格式在 DS7B 中不是简单 token 选择，而是强表面门问题。

### 对标点/格式机制的更新

Phase154 后，对标点符号的判断应进一步收紧为：

```text
标点/格式不是普通噪声；
也不是简单的“推高标点 token”；
它是最后层附近 writer 写入的 surface constraint；
这个 surface constraint 与 semantic answer support 在 answer-site gate 附近合流。
```

当前更准确的结构是：

```text
semantic support writer
  -> answer-site semantic readiness

format/syntax writer
  -> answer-site surface constraint

semantic + format shared writer
  -> surface gate
  -> open-vocab rank
```

### 理论进展

Phase153 给出“多路径合流”的行为证据；Phase154 给出 writer 级别的因果证据。

当前语言生成机制应更新为：

```text
context/object field
  -> semantic support route
  -> format/syntax constraint route
  -> final-layer writer gate
  -> answer-site surface state
  -> open-vocab competitor gate
  -> iterative token trajectory
```

其中：

```text
semantic support:
  更影响 answer rank。

format constraint:
  更影响 punctuation/format rank。

joint projection:
  在困难格式下同时损伤 answer rank 与 format rank，
  是当前最接近 surface gate 的测量。
```

### 硬伤与问题

1. **当前是 projection ablation，不是完整 writer replacement**

移除 projection 只能证明该 writer 输出中含有该方向成分，不能证明它就是唯一来源。

2. **只测 first-token rank，尚未做 3-step generation writer ablation**

Phase154 为了大范围扫描只测 rank。它还没有证明这些 writer 对真实三步生成 hit 的直接影响。

3. **format token groups 仍然粗**

quoted/list/colon/whitespace 仍可能遗漏 tokenizer 中的复合格式词元。

4. **DS7B 的 rank damage 很大，需要防止被极端 rank 放大误导**

尤其 quoted_answer 中 clean_format_rank 已经很差，rank delta 会非常大。需要下一轮加入 hit/rank 双指标。

5. **还没有细拆 attention head**

Phase154 定位到 attention_output / mlp_output 层级，但没有定位到具体 head。

### 下一步任务

Phase155 应继续推进：

```text
Head-Level Surface Gate and Multi-step Causal Closure
头级表面门与多步因果闭合
```

目标：

```text
1. 在 Phase154 定位出的关键层上，拆 attention head。
2. 对 DS7B 的 label_colon / answer_one_word / quoted_answer 做重点验证。
3. 不只看 rank，还要做 3-step generation hit。
4. 比较：
   semantic writer ablation
   format writer ablation
   joint writer ablation
   random same-norm / random head control
```

优先测试对象：

```text
DS7B:
  L28 attention_output
  L28 mlp_output
  formats = label_colon, answer_one_word, quoted_answer

GLM4:
  L39 attention_output
  L40 mlp_output

Qwen3:
  L36 mlp_output
  L35 mlp_output
```

判据：

```text
如果某些 head ablation 同时降低 answer hit 与 format hit：
  它们是 surface gate heads。

如果 head 只影响 format rank，不影响 answer hit：
  它们是 format constraint heads。

如果 MLP projection ablation 强而 head ablation 弱：
  surface gate 更可能在 MLP writer 完成，而 attention 只是读入状态。
```

## Phase 155: Head-Level Surface Gate and Multi-step Causal Closure 头级表面门与多步因果闭合 [2026-06-15 19:29]

### 本阶段目标

根据用户提供的 Phase154 复核意见，继续验证以下判断：

```text
Phase154 的方向基本正确：
  它证明 format/surface constraint 不是纯噪声，
  而是在 final writer 附近有可测因果成分。

但 Phase154 仍停留在 writer-level：
  attention_output / mlp_output 级别，
  还没有拆到 head-level，
  也没有直接验证多步生成 hit。
```

本轮 Phase155 的目标：

```text
1. 在 Phase154 定位出的关键 attention 层上，逐 head 做 ablation。
2. 用 first-token rank 选择 top_answer / top_format / top_joint head。
3. 对这些 head 做真实 3-step greedy generation。
4. 比较 clean、top heads、random head control 的 answer hit。
5. 判断 surface gate 是否能被单个 attention head 闭合。
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase155_head_surface_gate_generation_cuda.py \
  tests/gpt5/phase155_head_surface_gate_generation_summary.py

python tests/gpt5/phase155_head_surface_gate_generation_cuda.py qwen3 \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats label_colon,answer_one_word,quoted_answer,list_answer,multiple_choice \
  --train-objects 8 \
  --test-objects 8 \
  --batch-size 16 \
  --steps 3 \
  --output-dir results/gpt5_phase155_head_surface_gate_generation \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase155_head_surface_gate_generation_cuda.py glm4 \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats label_colon,answer_one_word,quoted_answer,list_answer,multiple_choice \
  --train-objects 8 \
  --test-objects 8 \
  --batch-size 16 \
  --steps 3 \
  --output-dir results/gpt5_phase155_head_surface_gate_generation \
  --hard-exit-after-model

python tests/gpt5/phase155_head_surface_gate_generation_cuda.py deepseek7b \
  --categories plant,time,container,number \
  --template-families long,short,neutral \
  --splits front_back,back_front \
  --formats label_colon,answer_one_word,quoted_answer,list_answer,multiple_choice \
  --train-objects 8 \
  --test-objects 8 \
  --batch-size 16 \
  --steps 3 \
  --output-dir results/gpt5_phase155_head_surface_gate_generation \
  --hard-exit-after-model

python tests/gpt5/phase155_head_surface_gate_generation_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase155_head_surface_gate_generation_cuda.py`
- 汇总脚本：`tests/gpt5/phase155_head_surface_gate_generation_summary.py`
- Qwen3 结果：`results/gpt5_phase155_head_surface_gate_generation/phase155_qwen3_head_surface_gate_generation.json`
- GLM4 结果：`results/gpt5_phase155_head_surface_gate_generation/phase155_glm4_head_surface_gate_generation.json`
- DS7B 结果：`results/gpt5_phase155_head_surface_gate_generation/phase155_deepseek7b_head_surface_gate_generation.json`
- 跨模型汇总：`results/gpt5_phase155_head_surface_gate_generation/phase155_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = plant, time, container, number
template families = long, short, neutral
splits = front_back, back_front
formats = label_colon, answer_one_word, quoted_answer, list_answer, multiple_choice
train objects/category = 8
heldout test objects/category = 8
cases/model = 120
generation steps = 3
```

关键 attention 层：

```text
Qwen3: L36, 32 heads
GLM4: L39, 32 heads
DS7B: L28, 28 heads
```

测试方式：

```text
1. 对每个 case 扫描该层所有 attention head。
2. 根据 first-token answer_rank_delta 选择 top_answer head。
3. 根据 first-token format_rank_delta 选择 top_format head。
4. 根据 answer_rank_delta + format_rank_delta 选择 top_joint head。
5. 同时加入 deterministic random head control。
6. 对 clean / top_answer / top_format / top_joint / random 做 3-step greedy generation。
```

### 客观结果

#### Qwen3

```text
cases = 120
layer = L36
heads = 32

clean hit = 0.367
top_answer hit = 0.358, delta -0.008
top_format hit = 0.370, delta +0.003
top_joint hit = 0.367, delta +0.000
random hit = 0.368, delta +0.001
```

主要被选中的 head：

```text
top_answer: H0 38 cases, H24 13, H26 13, H8 9, H2 7
top_format: H0 58 cases, H25 18, H2 14, H5 11, H24 6
top_joint: H0 41 cases, H25 17, H24 15, H5 14, H2 10
```

按格式：

```text
answer_one_word: clean 0.14, top_answer 0.13, top_format 0.15, top_joint 0.15, random 0.14
label_colon: clean 0.29, top_answer 0.27, top_format 0.29, top_joint 0.28, random 0.29
list_answer: clean 0.27, top_answer 0.25, top_format 0.26, top_joint 0.26, random 0.27
multiple_choice: clean 0.98, top_answer 0.98, top_format 0.98, top_joint 0.98, random 0.98
quoted_answer: clean 0.16, top_answer 0.16, top_format 0.16, top_joint 0.16, random 0.16
```

#### GLM4 bf16

```text
cases = 120
layer = L39
heads = 32

clean hit = 0.298
top_answer hit = 0.290, delta -0.008
top_format hit = 0.295, delta -0.003
top_joint hit = 0.295, delta -0.003
random hit = 0.298, delta +0.000
```

主要被选中的 head：

```text
top_answer: H0 29 cases, H19 13, H28 10, H11 9, H26 8
top_format: H0 29 cases, H18 17, H7 16, H9 14, H19 10
top_joint: H0 28 cases, H7 18, H18 17, H9 15, H19 14
```

按格式：

```text
answer_one_word: clean 0.07, top_answer 0.05, top_format 0.06, top_joint 0.06, random 0.07
label_colon: clean 0.18, top_answer 0.17, top_format 0.18, top_joint 0.18, random 0.17
list_answer: clean 0.16, top_answer 0.14, top_format 0.15, top_joint 0.15, random 0.16
multiple_choice: clean 0.98, top_answer 0.98, top_format 0.98, top_joint 0.98, random 0.99
quoted_answer: clean 0.10, top_answer 0.10, top_format 0.10, top_joint 0.10, random 0.10
```

#### DS7B

```text
cases = 120
layer = L28
heads = 28

clean hit = 0.254
top_answer hit = 0.251, delta -0.003
top_format hit = 0.261, delta +0.007
top_joint hit = 0.256, delta +0.002
random hit = 0.254, delta +0.000
```

主要被选中的 head：

```text
top_answer: H13 16 cases, H12 15, H10 14, H11 14, H0 11
top_format: H27 23 cases, H12 17, H9 16, H21 10, H10 8
top_joint: H27 23 cases, H12 17, H13 15, H9 12, H11 12
```

按格式：

```text
answer_one_word: clean 0.06, top_answer 0.07, top_format 0.08, top_joint 0.07, random 0.06
label_colon: clean 0.06, top_answer 0.05, top_format 0.06, top_joint 0.06, random 0.05
list_answer: clean 0.19, top_answer 0.17, top_format 0.18, top_joint 0.18, random 0.18
multiple_choice: clean 0.88, top_answer 0.88, top_format 0.90, top_joint 0.89, random 0.88
quoted_answer: clean 0.09, top_answer 0.10, top_format 0.09, top_joint 0.09, random 0.10
```

### 当前最可靠客观事实

1. **单个 attention head 不能闭合 surface gate**

虽然 top head 是根据 rank damage 选择出来的，但真实 3-step hit 基本不变：

```text
Qwen3 top_answer delta = -0.008
GLM4 top_answer delta = -0.008
DS7B top_answer delta = -0.003
```

这些变化与 random head control 接近，不能证明存在单个决定性 surface gate head。

2. **rank-selected head 与 generation hit 之间出现断裂**

某些 head 对 first-token rank 有可见影响，但对三步生成命中率几乎无影响。

这说明：

```text
rank damage != generation closure
```

Phase154 中用 rank 看到的 writer-level 效应，不能直接解释成单 head 因果机制。

3. **Phase154 的 writer-level 结论没有被推翻，但被重新定位**

Phase154 中 final writer 的 projection ablation 有明显效果，尤其 DS7B 和 GLM4。

Phase155 说明：

```text
如果 surface gate 存在，
它大概率不是单个 attention head，
而是多头集合 + MLP/residual final gate 的组合。
```

4. **multiple_choice 仍是特殊 control，不代表一般语言生成**

multiple_choice 的 clean hit 长期接近上限：

```text
Qwen3 multiple_choice clean = 0.98
GLM4 multiple_choice clean = 0.98
DS7B multiple_choice clean = 0.88
```

这类格式更像选项复制或局部选择，不应被用来代表开放格式生成。

5. **DS7B 的格式路径仍然特殊，但不是单头特殊**

DS7B top_format / top_joint 经常选择 H27/H12/H9，但 ablation 后 hit 反而略升或基本不变：

```text
DS7B top_format delta = +0.007
DS7B top_joint delta = +0.002
```

这说明 DS7B 的 format rank 异常不等于单 head surface gate。

### 对 Phase154 复核意见的判断

复核意见中正确部分：

```text
1. Phase154 是 writer-level 进展，不是完整 circuit closure。
2. 需要 head-level 与 generation hit 双重验证。
3. 只看 rank 容易被极端 rank delta 误导。
4. multiple_choice 必须作为 control，而不是核心证据。
5. 下一步应从单点 ablation 走向多成分 cumulative closure。
```

需要修正的部分：

```text
Phase155 没有支持“存在单个 surface gate head”的强结论。
当前证据更支持：
  attention head 提供局部读写扰动，
  真正控制生成闭合的门更可能在多头集合、MLP 输出或残差最终状态中。
```

### 理论进展

Phase155 将当前理论从：

```text
final attention/MLP writer 中存在 surface gate component
```

推进为：

```text
surface gate 不是一个单 head 开关，
而是 final residual stream 上由多 writer 共同塑造的生成许可状态。
```

更通俗地说：

```text
模型不是靠某一个注意力头决定“现在该输出答案词元还是格式词元”。
它更像是在最后几层把语义支持、格式要求、候选词元竞争一起压到 residual state 中，
然后由 LM head 读出。
```

这与前面 Phase146-154 的现象兼容：

```text
semantic support 可以被恢复，
format/surface constraint 可以被扰动，
但 token selection gap 仍然存在。
```

因此当前瓶颈不是“还没找到那个神奇 head”，而是：

```text
需要测多成分集合如何共同改变 final residual state。
```

### 硬伤与问题

1. **本轮只测单 head**

如果机制是 3-8 个 head 的集合，单 head ablation 当然可能很弱。

2. **只测一个 attention 层**

虽然层位来自 Phase154，但 surface gate 可能跨 L28/L29 或 L35/L36/L37 等多层累积。

3. **没有拆 MLP channel**

Phase154 中 MLP writer 很强，Phase155 只拆 attention head，尚未解释 MLP 贡献。

4. **3-step greedy generation 仍然粗**

hit 指标比 rank 更接近真实生成，但仍只看短步数。它不能覆盖后续更长格式链。

5. **rank 选择 top head 可能不是 hit 最优 head**

本轮用 answer_rank_delta / format_rank_delta 选头。若 hit 对另一些不显著改变 rank 的 head 更敏感，则本轮会漏掉。

### 下一步任务

Phase156 应继续推进：

```text
Multi-head Set and MLP Gate Cumulative Closure
多头集合与 MLP 门的累积闭合
```

核心目标：

```text
从单 head ablation 改为 top-k head set + MLP projection 的联合干预，
判断 surface gate 是否需要集合级移除才影响真实生成。
```

测试设计：

```text
1. 继续使用 qwen3 / GLM4 / DS7B 三模型。
2. 保留 Phase155 的 120 cases/model 范围。
3. 对关键层做 top-k head cumulative ablation：
   k = 1, 2, 4, 8, all_selected
4. 选择 head 的依据同时包含：
   answer_rank_delta
   format_rank_delta
   joint_delta
   hit-sensitive pilot
5. 加入 MLP projection ablation：
   semantic_proj
   format_proj
   joint_proj
6. 测：
   clean
   top-k-heads
   MLP-only
   top-k-heads + MLP
   random-k-heads
```

优先层位：

```text
DS7B:
  L28 attention + L28 MLP
  必测 label_colon, answer_one_word, quoted_answer, list_answer

GLM4:
  L39 attention + L40 MLP

Qwen3:
  L36 attention + L36/L35 MLP
```

判据：

```text
如果 top-k-heads + MLP 明显降低 3-step hit，
而 random-k-heads 不降低：
  surface gate 是集合级 causal mechanism。

如果 MLP-only 强，top-k-heads 弱：
  attention 主要负责读入/路由，MLP 负责最终写入。

如果两者都弱：
  当前 surface gate 仍不在这些局部 writer 中，
  需要转向 final residual replacement 或 LM-head competition decomposition。
```
