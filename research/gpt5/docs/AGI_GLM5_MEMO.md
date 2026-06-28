## Phase 604: AI2050 Website Static Page Generation [2026-06-24 15:09]

### 命令

```bash
rg --files website && pwd
ls -la website
file website/logo.png website/pic1.png
tail -n 80 research/glm5/docs/AGI_GLM5_MEMO.md
rg '^## Phase [0-9]+' research/glm5/docs/AGI_GLM5_MEMO.md | tail -n 5
/snap/bin/chromium --headless --disable-gpu --no-sandbox --screenshot=/home/rankrank/Documents/OpenOne/Ai2050-OpenOne/website/_check_desktop.png --window-size=1440,1600 file:///home/rankrank/Documents/OpenOne/Ai2050-OpenOne/website/index.html
/snap/bin/chromium --headless --disable-gpu --no-sandbox --screenshot=/home/rankrank/Documents/OpenOne/Ai2050-OpenOne/website/_check_mobile.png --window-size=390,1300 file:///home/rankrank/Documents/OpenOne/Ai2050-OpenOne/website/index.html
rm -f website/_check_desktop.png website/_check_mobile.png
git diff --check -- website/index.html website/styles.css
date '+%Y-%m-%d %H:%M'
mkdir -p research/gpt5/docs
```

### 生成脚本与文件

本阶段没有生成测试脚本。

新增静态网页文件：

```text
website/index.html
website/styles.css
```

### 原理

参考 `website/pic1.png` 的视觉结构，拆解为：

```text
1. 顶部导航：品牌标识、栏目入口、参与建设按钮。
2. 首屏：左侧 AI2050 叙事，右侧使用 pic1.png 作为概念视觉资产。
3. 四个栏目卡片：AI2050计划、AGI项目、论坛、捐赠与开支。
4. 页脚：使命、研究、社区、订阅更新。
```

实现方式保持基础：

```text
1. 仅使用 HTML + CSS。
2. 不引入构建工具。
3. 不引入统计方法。
4. 不进行模型测试。
5. 使用 Chromium headless 做桌面与移动端截图目检。
```

### 结果

已完成一个可直接打开的静态网页：

```text
website/index.html
```

检查结果：

```text
1. 桌面截图生成成功。
2. 移动端截图生成成功。
3. git diff --check 没有发现空白格式错误。
4. 临时截图已删除。
```

### 理论研究进展

本阶段不是语言机制或模型行为研究，不产生 AGI 理论结论。

但网页表达层面对项目叙事做了结构化拆分：

```text
AI2050 = 开放研究 + 智能理论 + 公共讨论 + 透明治理
```

这对后续研究传播有辅助意义：把复杂研究目标拆成公众可理解的入口，有助于吸引协作者进入具体任务。

### 严格审视

本阶段硬伤：

```text
1. 没有新增科研证据。
2. 没有运行 qwen3、GLM4、DS7B。
3. 首屏右侧视觉仍依赖参考图本身，不是独立生成的精细分层素材。
4. 栏目图形为 CSS 抽象表达，不能替代真实项目数据或真实研究成果展示。
```

### 下一步阶段性任务

建议后续不要停留在单个网页组件，而是推进一个完整的 AI2050 公开研究门户：

```text
1. 研究路线页：展示语言数学结构破解路线。
2. 实验记录页：按 Phase 展示关键实验、结论、反例、硬伤。
3. 数据与脚本页：公开测试命令、脚本、模型、样本规模。
4. 贡献入口页：把研究任务拆成可领取的大任务，而不是零散小功能。
5. 治理透明页：展示捐赠、开支、决策记录和审计机制。
```

## Phase 711: Global Mechanism Atlas v0 and Marginal-Return Pivot [2026-06-27 11:27]

### 本阶段任务

分析两份新材料后，判断基本正确：

```text
1. 应该启动全局神经元图谱，但第一版应是 language-core verifiable micro-atlas，而不是“大而全图谱”。
2. Phase 709-710 有真实局部增量，但主要是边界收紧，patch 路线已经进入明显边际收益递减区。
```

因此本阶段没有继续新增模型 patch，而是把 Phase 698-710 的已有结果整理成可查询的机制图谱 v0。

### 新增脚本和命令

```text
tests/gpt5/phase711_global_mechanism_atlas_v0.py
```

```bash
python -m py_compile tests/gpt5/phase711_global_mechanism_atlas_v0.py
python tests/gpt5/phase711_global_mechanism_atlas_v0.py
```

输出目录：

```text
results/glm5_phase711_global_mechanism_atlas_v0/
```

### 测试原理

本阶段不重新运行 qwen3 / GLM4 / DS7B，而是整合既有跨模型结果：

```text
Phase 698: answer-last attention head source audit
Phase 707: full value phrase likelihood audit
Phase 709: natural generation write-in closure
Phase 710: natural write-in factor split
```

图谱单元形式：

```text
unit = model + layer + head/channel + source_group + target_position + role_scores + status
```

核心角色分数：

```text
route_gain_score
identity_score
format_or_prose_score
donor_residue_score
phrase_target_minus_donor
phrase_target_minus_prose
post_layer_target_value_rate
```

### 客观结果

```text
n_units = 288
```

按模型：

```text
deepseek7b: 96
glm4: 96
qwen3: 96
```

按类型：

```text
attention_head: 96
attention_channel: 192
```

按状态：

```text
deepseek7b: prose_or_format_route_carrier = 96
glm4: unresolved_or_weak = 96
qwen3: short_value_route_carrier = 96
```

### 理论进展

当前研究从：

```text
寻找单个可修复答案的 patch
```

转向：

```text
构建条件化残差轨迹上的机制图谱。
```

更稳妥的表达是：

```text
h_{l+1,p}
=
h_{l,p}
+ A_{l,p}(QK_{l,p}, V_{l,p})
+ M_{l,p}(h_{l,p})
```

图谱表达：

```text
G
=
{u_i, r_i, s_i, e_i}
```

其中：

```text
u_i = 单元，包括 head、channel、后续 neuron / MLP channel
r_i = 角色，包括 route、identity、format、readout、continuation
s_i = 状态分数，包括 target_value、prose_target、donor_residue、phrase margin
e_i = 证据层级，包括 attention、phrase likelihood、natural generation、causal patch
```

### 硬伤和下一步

当前 atlas v0 继承的是 condition-level evidence，不是 unit-local causal proof；attention_channel 也不是 neuron。因此它只能作为索引系统，不能直接宣称完成神经元图谱。

下一阶段：

```text
Phase 712: QK-V Split for Global Mechanism Atlas
```

核心目标：

```text
1. 固定 V content，只替换 Q/K attention pattern。
2. 固定 Q/K attention pattern，只替换 V content / value output。
3. 比较 target_value、prose_target、donor_value、other 的自然生成结果。
4. 把 addressing-role 和 content-role 回填到 atlas v0。
```

## Phase 712: QK-V Factor Atlas Audit [2026-06-27 12:23]

### 任务和脚本

Phase 712 继续 Phase 711 的 atlas v0 目标，拆分 Q/K addressing（查询 / 键寻址）和 V content（值内容）。

新增脚本：

```text
tests/gpt5/phase712_qkv_factor_atlas_audit.py
tests/gpt5/run_phase712_qkv_factor_atlas_audit_full.sh
tests/gpt5/phase712_update_atlas_with_qkv.py
```

执行：

```bash
python -m py_compile tests/gpt5/phase712_qkv_factor_atlas_audit.py
tests/gpt5/run_phase712_qkv_factor_atlas_audit_full.sh
python tests/gpt5/phase712_update_atlas_with_qkv.py
```

模型顺序：

```text
qwen3 -> GLM4 -> DS7B
```

每个模型使用：

```text
--hard-exit-after-model
```

### 原理

对 source contribution（源贡献）做分解：

```text
C = sum_s a_s v_s
```

比较 terse 和 short 状态：

```text
Delta C
=
sum_s (a_terse_s - a_short_s) v_short_s
+
sum_s a_short_s (v_terse_s - v_short_s)
+
sum_s (a_terse_s - a_short_s)(v_terse_s - v_short_s)
```

三项分别对应：

```text
QK/addressing term
V/content term
QK*V interaction term
```

### 客观结果

```text
DS7B source_top_channel:
  dominant = qk_addressing
  abs_qk_share = 0.505
  abs_v_share = 0.247
  abs_interaction_share = 0.248
  sum_total_direct = 34.536267

GLM4 source_top_channel:
  dominant = qk_addressing
  abs_qk_share = 0.291
  abs_v_share = 0.356
  abs_interaction_share = 0.353
  sum_total_direct = 3.135055

qwen3 source_top_channel:
  dominant = mixed_coupled
  abs_qk_share = 0.310
  abs_v_share = 0.345
  abs_interaction_share = 0.344
  sum_total_direct = 33.323272
```

DS7B 最强头：

```text
L26H15: qk_addressing, abs_qk_share = 0.948
L26H19: qk_addressing, abs_qk_share = 0.977
L23H11: qk_addressing, abs_qk_share = 0.844
L27H2: qk_addressing, abs_qk_share = 0.900
```

Atlas 回填：

```text
n_units = 288
n_units_with_qkv = 288

qk_addressing = 157
mixed_coupled = 131
```

按模型：

```text
deepseek7b: qk_addressing = 79, mixed_coupled = 17
glm4: qk_addressing = 36, mixed_coupled = 60
qwen3: qk_addressing = 42, mixed_coupled = 54
```

### 结论和硬伤

Phase 712 支持一个更具体的判断：

```text
DS7B 当前 source_top_channel effect 主要偏 QK/addressing；
qwen3 和 GLM4 更混合；
不能把该结论直接泛化为所有模型的语言机制。
```

硬伤：

```text
1. 这是 contribution decomposition，不是严格 causal Q/K replacement。
2. 数值受 short-state 基准选择影响。
3. attention_channel 仍不是 neuron。
4. 小模型偏差仍然存在。
```

下一步：

```text
Phase 713: Causal QK Pattern Replacement vs V Content Replacement
```

重点测试 DS7B：

```text
L26H15
L26H19
L23H11
L27H2
```

## Phase 713: IntelligentTheory 历史记录归纳与 QK/V 理论更新 [2026-06-28 09:01]

### 任务

根据以下三个历史记录文件归纳阶段理论、数学公式、非线性理论体系和有效理论，并更新 `research/IntelligentTheory.md`：

```text
research/glm5/docs/AGI_GLM5_MEMO_20260601.md
research/glm5/docs/AGI_GLM5_MEMO_20260625.md
research/glm5/docs/AGI_GLM5_MEMO.md
```

### 修改文件

```text
research/IntelligentTheory.md
```

本阶段未运行模型测试；这是理论文档更新阶段。

### 结果

已完成：

```text
1. 补充 Phase 708-712 后的语言编码机制更新。
2. 新增 QK/V 因子分解公式。
3. 新增基于词嵌入的完整计算流程例子。
4. 新增阶段十一：自然生成闭环、图谱 v0 与 QK/V 因子回填。
5. 重写第七章：最有效完整理论，以及问题硬伤和下一步。
```

最新理论收束为：

```text
语言智能
= 相对编码网络
+ 对象知识锚定
+ 关系/规则检索
+ 条件化状态变换
+ QK/V 源贡献机制
+ 候选短语竞争
+ 范数/格式/策略/生成读出门
+ 因果图谱
```

核心数学更新：

```text
Delta C_g
= Delta C_QK
+ Delta C_V
+ Delta C_QKxV
```

### 严格审视

有效主线：

```text
相对编码、条件化状态变换、源词元贡献、QK/V 因子分解、候选短语竞争、图谱化因果等级。
```

硬伤：

```text
1. 还没有 QK pattern replacement 与 V content replacement 的严格因果拆分。
2. attention channel 仍不是 neuron。
3. 小模型结构偏差仍然存在。
4. 自然生成端仍被 prose / format / continuation route 强烈干扰。
```

下一阶段：

```text
Phase 714: QK Pattern Replacement vs V Content Replacement Causal Audit
```

## Phase 717: IntelligentTheory 双文件比较与合并版更新 [2026-06-28 10:11]

### 任务

比较：

```text
research/IntelligentTheory.md
research/IntelligentTheory_20260628.md
```

并将更合理有效的最新理论更新到：

```text
research/IntelligentTheory.md
```

### 结论

`research/IntelligentTheory_20260628.md` 更适合作为主体，因为它包含更完整的非线性理论体系和 Phase 713-715 的严格硬伤审视。

旧 `research/IntelligentTheory.md` 中更有效的补充是：

```text
1. QK/V 因子分解公式。
2. 基于词嵌入入口的简化计算流程。
```

### 更新结果

`research/IntelligentTheory.md` 已更新为：

```text
主体 = IntelligentTheory_20260628.md
补充 = QK/V 因子分解 + 词嵌入计算入口 + Phase 708-712 因果等级更新
```

核心公式：

```text
Delta C_g
= Delta C_QK
+ Delta C_V
+ Delta C_QKxV
```

下一步仍是：

```text
QK Pattern Replacement vs V Content Replacement Causal Audit
```
