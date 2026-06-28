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

## Phase 719: 条件化相对状态—生成场闭合理论整理更新 [2026-06-28 14:37]

### 任务

分析附件提出的“条件化相对状态—生成场闭合理论”，并更新：

```text
research/IntelligentTheory.md
```

### 结论

理论方向基本正确，但需要收紧为“当前最有效组织框架”，不能视为完整闭合理论。

最合理整合方式：

```text
编码状态形成层 = 十模块理论
生成读出闭合层 = 生成场理论
```

核心定义：

```text
语言不是固定概念向量的直接读出，而是词嵌入在上下文中形成条件化相对状态轨迹；
这个轨迹进入生成场后，经过源贡献、路线增益、值身份、残差传播、读出投影、
多候选竞争和完整短语续写，最终完成或失败于生成闭合。
```

### 更新内容

已将第七章标题更新为：

```text
七，条件化相对状态—生成场闭合理论，以及问题硬伤和下一步
```

并新增：

```text
7.0 对新整合理论的判断
```

核心公式：

```text
Phi_L(a|x)
= {
  S_semantic, I_intent, P_protocol, F_format,
  G_route, V_identity, U_channel, T_residual,
  R_readout, M_competition, C_continuation
}
```

```text
GenerationClosure
<=> L(y_target|x) - max_{y != y_target} L(y|x) > delta
```

### 严格审视

```text
1. 生成场不是已证明的独立物理子空间，而是功能因子。
2. V_identity 尚未定位。
3. QK/V 分解仍需要 causal replacement 验证。
4. 当前结果仍受小模型偏差限制。
```

## Phase 718: AI2050 三个 PNG 参考页静态 HTML 转换 [2026-06-28 14:03]

### 命令

```bash
find frontend/website -maxdepth 2 -type f | sort
sed -n '1,260p' frontend/website/index.html
file frontend/website/*.png frontend/website/*.html
/snap/bin/chromium --headless --disable-gpu --no-sandbox --screenshot=frontend/website/.check/plan-desktop.png --window-size=1280,1600 file:///home/rankrank/Documents/OpenOne/Ai2050-OpenOne/frontend/website/ai2050计划.html
/snap/bin/chromium --headless --disable-gpu --no-sandbox --screenshot=frontend/website/.check/agi-desktop.png --window-size=1280,1500 file:///home/rankrank/Documents/OpenOne/Ai2050-OpenOne/frontend/website/agi研究.html
/snap/bin/chromium --headless --disable-gpu --no-sandbox --screenshot=frontend/website/.check/about-desktop.png --window-size=1280,1500 file:///home/rankrank/Documents/OpenOne/Ai2050-OpenOne/frontend/website/5-关于.html
/snap/bin/chromium --headless --disable-gpu --no-sandbox --screenshot=frontend/website/.check/plan-mobile.png --window-size=390,1400 file:///home/rankrank/Documents/OpenOne/Ai2050-OpenOne/frontend/website/ai2050计划.html
/snap/bin/chromium --headless --disable-gpu --no-sandbox --screenshot=frontend/website/.check/agi-mobile.png --window-size=390,1400 file:///home/rankrank/Documents/OpenOne/Ai2050-OpenOne/frontend/website/agi研究.html
/snap/bin/chromium --headless --disable-gpu --no-sandbox --screenshot=frontend/website/.check/about-mobile.png --window-size=390,1400 file:///home/rankrank/Documents/OpenOne/Ai2050-OpenOne/frontend/website/5-关于.html
rm -rf frontend/website/.check
git diff --check -- frontend/website/index.html frontend/website/styles.css frontend/website/ai2050计划.html frontend/website/agi研究.html frontend/website/5-关于.html
date '+%Y-%m-%d %H:%M'
```

### 生成脚本与文件

本阶段没有生成测试脚本。

新增静态页面：

```text
frontend/website/ai2050计划.html
frontend/website/agi研究.html
frontend/website/5-关于.html
```

修改文件：

```text
frontend/website/index.html
frontend/website/styles.css
```

### 原理

参考以下三张 PNG 的视觉结构：

```text
frontend/website/ai2050计划.png
frontend/website/agi研究.png
frontend/website/5-关于.png
```

将其拆解为可维护的 HTML + CSS 页面：

```text
1. 共用顶部导航、品牌标识和页脚。
2. AI2050计划页：首屏城市/网络概念视觉 + 十个计划列表。
3. AGI研究页：首屏研究视觉 + 路线分析 + 项目进展。
4. 关于页：使命、核心原则、团队介绍、统计数据与加入入口。
5. 首页导航和两个入口卡片改为链接到对应新页面。
```

### 结果

已完成三张 PNG 对应的静态 HTML 页面，并用 Chromium 做桌面与移动端截图目检。

检查结果：

```text
1. 三个页面桌面截图均正常渲染。
2. 三个页面移动端截图均正常渲染。
3. 修复了桌面计划页标题断行问题。
4. git diff --check 没有发现空白格式错误。
5. 临时截图目录已删除。
```

### 理论研究进展

本阶段是网站工程实现，不产生新的 AGI 理论结论。

但页面结构把当前项目表达拆成三个公开入口：

```text
AI2050计划 = 长期文明路线
AGI研究 = 语言编码机制 -> 智能统一理论 -> AGI
关于AI2050 = 开放研究 + 聚焦AGI + 长期价值
```

这有助于后续把研究路线、阶段进展和公共协作入口组织成可读的项目门户。

### 严格审视

当前硬伤：

```text
1. 页面为静态展示，未接入真实项目数据。
2. 部分图形由 CSS 抽象实现，不能完全等同于 PNG 中的精细插画。
3. 论坛、捐赠、项目进展等入口仍是占位链接。
4. 本阶段没有运行 qwen3、GLM4、DS7B，也没有产生模型证据。
```

### 下一步任务

建议进入网站信息架构阶段：

```text
1. 建立论坛、捐赠与开支、项目进展、研究路线四个真实页面。
2. 将 README 与 IntelligentTheory 中的路线分析和项目进展整理进 AGI研究页。
3. 加入 Phase 时间线，按证据等级展示研究推进。
4. 为每个页面建立统一导航状态和移动端菜单。
5. 后续再接前后端 API，把实验结果从静态展示推进到动态研究门户。
```

## Phase 720: Functional Atlas v1 Readiness and Head-to-Neuron Bridge [2026-06-28 15:07]

### 任务

分析用户提供的三部分内容：

```text
1. 条件化相对状态—生成场闭合理论。
2. 破解编码机制应转向功能图谱，例如 apple / fruit / color / translation 的复用与差异。
3. 最终目标应是 neuron 级全局图谱，但当前更适合先完成 head 级功能图谱。
```

结合当前 Phase 711-712 的跨模型图谱结果，判断这些内容是否正确，并继续推进任务。

### 结论判断

总体判断：

```text
上述方向基本正确。
破解语言编码机制不能继续只靠单点 patch。
下一阶段应进入 function atlas（功能图谱）工程。
但当前不能直接做 full neuron global atlas（全神经元全局图谱）。
更稳妥路线是：
head-level functional atlas -> channel/QK/V bridge -> targeted neuron/MLP/SAE atlas。
```

关键收紧：

```text
head 不是语义单元。
head 是 route / addressing / content / output projection / downstream nonlinear response 的混合接口。
所以 head 图谱只能作为全局路径骨架，不能直接等同于 neuron 编码机制。
```

### 脚本

新增脚本：

```text
tests/gpt5/phase720_functional_atlas_v1_builder.py
```

该脚本不加载模型，不进行新推理，只读取已有 Phase 712 跨模型图谱：

```text
results/glm5_phase712_qkv_factor_atlas_audit/phase712_atlas_units_with_qkv.jsonl
```

并生成 Phase 720 功能图谱 v1：

```text
results/glm5_phase720_functional_atlas_v1/phase720_functional_atlas_nodes.jsonl
results/glm5_phase720_functional_atlas_v1/phase720_functional_atlas_summary.json
results/glm5_phase720_functional_atlas_v1/phase720_functional_atlas_report.md
```

### 命令

```bash
python tests/gpt5/phase720_functional_atlas_v1_builder.py
python -m py_compile tests/gpt5/phase720_functional_atlas_v1_builder.py
```

### 客观结果

Phase 720 生成节点：

```text
n_nodes = 288
by_model:
  deepseek7b = 96
  glm4 = 96
  qwen3 = 96
by_unit_type:
  attention_head = 96
  attention_channel = 192
by_graph_level:
  head_route = 96
  channel_bridge = 192
```

QK/V 因子：

```text
qk_addressing = 157
mixed_coupled = 131
```

路线角色：

```text
deepseek7b:
  prose_or_format_route_carrier = 96

glm4:
  unresolved_or_weak = 96

qwen3:
  short_value_route_carrier = 96
```

当前已经实测的功能族只有：

```text
object_relation_value_short_answer
```

尚未实测的功能族：

```text
fruit_identity_reuse_difference
color_value_reuse_difference
translation_language_route
```

### 测试原理

本阶段不是新模型测试，而是图谱构建测试。

原理是把 Phase 711/712 中已有的单元证据整理成统一节点：

```text
function_family
model
unit_type
layer/head/channel
source_group
target_position
route_role
evidence_level
qkv_dominant_factor
route_gain_score
identity_score
format_or_prose_score
next_drilldown
```

### 理论进展

Phase 720 支持把当前路线正式改写为：

```text
局部机制验证 -> 功能图谱工程 -> 目标神经元/通道下钻
```

当前最接近真实语言编码机制的表述是：

```text
语言生成不是单个语义向量被读出，
而是输入条件在残差流中形成多条功能路线，
这些路线通过 QK addressing、V content、W_O readout、MLP nonlinear gate 和最终 logits/readout 竞争，
最终在生成场中闭合为可输出短语。
```

### 严格审视

硬伤：

```text
1. 当前功能图谱只覆盖 object_relation_value_short_answer 一个微型功能族。
2. apple / fruit / red / translation 等功能还没有实测，不能从现有结果直接外推。
3. head 级图谱只是路径骨架，不是 neuron 级编码机制。
4. channel 级结果仍未严格等同于神经元功能，需要继续拆到 QK/V、W_O、MLP、SAE feature。
5. 当前模型都是小模型，内部结构可能有模型规模和架构偏差。
```

### 下一步

Phase 721 应继续处在同一阶段：

```text
Global Functional Head Atlas Data Expansion
```

目标不是马上做全神经元图谱，而是先扩展功能族：

```text
1. fruit identity and category reuse/difference。
2. color value reuse/difference。
3. translation source/target language route。
4. simple grammar protocol route。
```

每个功能族最低证据要求：

```text
1. observational source contribution。
2. head/channel route score。
3. QK/V factor split。
4. top units causal patch。
5. phrase likelihood or natural generation closure。
```

如果 Phase 721 属于当前阶段，则应自动继续推进，不需要重新确认。

## Phase 721: Global Functional Head Atlas Data Expansion [2026-06-28 15:16]

### 任务

Phase 720 已确认当前阶段应从局部 patch 转向 function atlas（功能图谱）。本阶段继续推进同一阶段任务，把图谱从一个微型功能族扩展到四个功能族：

```text
1. fruit_identity_reuse_difference
2. color_value_reuse_difference
3. translation_language_route
4. simple_grammar_protocol_route
```

### 脚本

新增脚本：

```text
tests/gpt5/phase721_global_functional_head_atlas_expansion.py
tests/gpt5/run_phase721_global_functional_head_atlas_expansion_full.sh
```

输出目录：

```text
results/glm5_phase721_global_functional_head_atlas_expansion/
```

### 命令

```bash
python -m py_compile tests/gpt5/phase721_global_functional_head_atlas_expansion.py
python tests/gpt5/phase721_global_functional_head_atlas_expansion.py --dry-run --max-cases-per-family 24
bash -n tests/gpt5/run_phase721_global_functional_head_atlas_expansion_full.sh
bash tests/gpt5/run_phase721_global_functional_head_atlas_expansion_full.sh
```

脚本依次运行：

```text
qwen3 -> glm4 -> deepseek7b
```

每个模型都使用：

```text
--hard-exit-after-model
```

### 测试原理

本阶段是 observational attention atlas，不是 causal patch。

在 answer_last 位置统计所有层所有 head 对下列 token group 的注意力：

```text
record_line
question_line
instruction_line
answer_line
self_last
object_name
relation_name
target_value
source_value
target_language
grammar_marker
```

并计算 source_focus_score，用来筛选下一阶段 causal patch 候选：

```text
source_focus_score
= target_value_mass
+ 0.5 * object_name_mass
+ 0.5 * relation_name_mass
+ 0.5 * target_language_mass
+ 0.5 * grammar_marker_mass
- 0.5 * instruction_line_mass
- 0.25 * answer_line_mass
```

### 样本量

```text
total prompts = 96
color_value_reuse_difference = 24
fruit_identity_reuse_difference = 24
simple_grammar_protocol_route = 24
translation_language_route = 24
```

### 客观结果

```text
status = complete
models = qwen3, glm4, deepseek7b

qwen3 prompt-head rows = 110592
glm4 prompt-head rows = 122880
deepseek7b prompt-head rows = 75264
```

代表性候选：

```text
qwen3:
  simple_grammar_protocol_route L28H0 score=1.1737 target_value=0.8258 object=0.6809
  fruit_identity_reuse_difference L26H26 score=0.9459 target_value=0.9305
  translation_language_route L28H0 score=0.9079 target_value=0.8971

glm4:
  simple_grammar_protocol_route L29H26 score=1.2467 target_value=0.8817 object=0.7213
  simple_grammar_protocol_route L23H10 score=1.2119 target_value=0.8579 object=0.7082
  simple_grammar_protocol_route L29H18 score=1.1837 target_value=0.8136 object=0.7327

deepseek7b:
  simple_grammar_protocol_route L22H1 score=0.9293 target_value=0.6546 object=0.5469
  simple_grammar_protocol_route L21H25 score=0.8974 target_value=0.6091 object=0.5159
  color_value_reuse_difference L23H0 score=0.5027 target_value=0.1340 object=0.2956 relation=0.4517
```

### 理论进展

本阶段支持：

```text
不同模型可能复用相同功能路线类型，
但具体 head、层位置、source 侧重不同。
全局图谱必须记录复用部分和差异部分，
不能只寻找一个跨模型固定 head。
```

### 严格审视

```text
1. 本阶段是观测型图谱，不是因果证明。
2. 高注意力不等于 causal necessity。
3. source_focus_score 是候选筛选指标，不是机制公式。
4. prompt 模板仍偏显式 record copy，可能高估 target_value attention。
5. 当前模型为小模型，存在结构偏差风险。
```

### 下一步

Phase 722 属于下一小阶段，不再是当前 observational atlas expansion 的同一阶段：

```text
Functional Head Atlas Causal Patch Validation
```

目标：

```text
把 Phase 721 的 observational candidate
推进到 causal candidate。
```

## Phase 722: Functional Head Atlas Causal Ablation Validation [2026-06-28 15:43]

### 任务

用户提供的判断基本正确：当前路线不是原地打转，而是已经从局部 patch 进入 functional atlas（功能图谱）；但必须区分 visibility（可见性）、causal necessity（因果必要性）和 generation closure（生成闭合）。

Phase 721 只证明 top heads 在 answer_last 位置注意到了功能源词元，不证明这些 head 必要。因此本阶段做局部因果消融。

### 脚本

```text
tests/gpt5/phase722_functional_head_atlas_causal_ablation.py
tests/gpt5/run_phase722_functional_head_atlas_causal_ablation_full.sh
```

输出目录：

```text
results/glm5_phase722_functional_head_atlas_causal_ablation/
```

### 命令

```bash
python -m py_compile tests/gpt5/phase722_functional_head_atlas_causal_ablation.py
bash -n tests/gpt5/run_phase722_functional_head_atlas_causal_ablation_full.sh
bash tests/gpt5/run_phase722_functional_head_atlas_causal_ablation_full.sh
```

运行顺序：

```text
qwen3 -> glm4 -> deepseek7b
```

每个模型均使用：

```text
--hard-exit-after-model
```

### 测试原理

对 Phase 721 每个模型 / 每个功能族的 top 3 source-focus heads 做 answer_last o_proj input 局部置零消融，并与同层随机 head 对照。

观测指标：

```text
target_logprob_delta
target_rank_delta
margin_delta
top1_drop_rate
```

### 客观结果

```text
status = complete
models = qwen3, glm4, deepseek7b
n_rows per model = 576
```

qwen3：

```text
simple_grammar_protocol_route L28H0:
  mean_logprob_delta = -0.0130
  top1_drop_rate = 0.042

其他候选多数效应很弱。
```

GLM4：

```text
fruit_identity L24H19:
  mean_logprob_delta = -0.0106

translation L29H28:
  mean_logprob_delta = -0.0086

但 rank/top1 几乎不变。
```

DS7B：

```text
fruit_identity L20H17:
  mean_logprob_delta = -2.8909
  mean_rank_delta = 28.46
  top1_drop_rate = 0.250

fruit_identity L27H23:
  mean_logprob_delta = -0.8819
  mean_rank_delta = 5.62
  top1_drop_rate = 0.125

translation L24H21:
  mean_logprob_delta = -0.7647
  mean_rank_delta = 62.17
  top1_drop_rate = 0.083

grammar L22H24:
  mean_logprob_delta = -0.4171
  mean_rank_delta = 1.46
  top1_drop_rate = 0.083
```

### 关键洞察

```text
高 source attention 不等于高 causal necessity。
```

更准确现象：

```text
qwen3 / GLM4:
  观测可见性强，但消融效应弱。

DS7B:
  fruit / translation / grammar 出现强必要性 head。
```

### 严格审视

```text
1. 本阶段只测 first-token，不测完整短语。
2. zero ablation 可能 off-manifold。
3. 只测 top 3 heads / family。
4. qwen3 / GLM4 弱效应可能是冗余强，不是无机制。
5. DS7B 强效应需要 phrase likelihood 和 natural generation 闭合验证。
```

### 下一步

Phase 723 属于新的闭合验证小阶段，不自动继续。

建议：

```text
Functional Head Causal Closure Audit
```

只选强效应候选做：

```text
1. full phrase likelihood drop。
2. natural generation category change。
3. candidate / random / high-attention non-causal control。
4. channel / QK-V / W_O 下钻。
```

## Phase 723: Apple-Fruit-Attribute Reuse-Difference Micro-Atlas [2026-06-28 16:06]

### 触发问题

用户给出的 Phase 722 分析基本正确：Phase 722 把 functional head atlas（功能头图谱）从观测可见性推进到因果必要性筛选，但仍然不能说明完整语言编码机制。另一个建议也正确：下一阶段不应继续大而全扩展，而应收缩到“苹果-水果-属性”微世界，先把 reuse-difference mechanism（复用差分机制）测清楚。

### 本阶段脚本

```text
tests/gpt5/phase723_apple_fruit_attribute_micro_atlas.py
tests/gpt5/run_phase723_apple_fruit_attribute_micro_atlas_full.sh
```

### 运行命令

```bash
tests/gpt5/run_phase723_apple_fruit_attribute_micro_atlas_full.sh
```

该脚本按顺序运行：

```bash
python tests/gpt5/phase723_apple_fruit_attribute_micro_atlas.py --model qwen3 --hard-exit-after-model
python tests/gpt5/phase723_apple_fruit_attribute_micro_atlas.py --model glm4 --hard-exit-after-model
python tests/gpt5/phase723_apple_fruit_attribute_micro_atlas.py --model deepseek7b --hard-exit-after-model
python tests/gpt5/phase723_apple_fruit_attribute_micro_atlas.py --summarize-only
```

### 测试原理

本阶段构造 114 个 apple-fruit-attribute（苹果-水果-属性）微世界案例：

```text
1. explicit_profile：显式给出对象属性记录。
2. conflict_profile：显式给出反常事实，测试上下文绑定。
3. commonsense：不提供事实，只问常识。
```

对象分三组：

```text
apple：苹果。
other_fruit：banana / pear / grape / orange / lemon。
nonfruit：carrot / potato / stone / chair / car / spoon。
```

关系包括：

```text
category / color / taste / shape / edible / grows_on_tree
```

每个模型选 Phase 722 中 fruit_identity_reuse_difference（水果身份复用差分）最强的 3 个候选 head（注意力头），并加入 3 个同层随机 head 作为对照。对每个案例计算 teacher-forced answer phrase likelihood（教师强制答案短语似然），即逐词元计算正确答案短语的 logprob（对数概率），然后消融单个 head，观察答案短语似然下降。

核心指标：

```text
necessity = - mean_logprob_delta

apple_minus_other_fruit
  = apple_explicit_necessity - other_fruit_explicit_necessity

other_fruit_minus_nonfruit
  = other_fruit_explicit_necessity - nonfruit_explicit_necessity
```

其中：

```text
positive necessity 表示消融该 head 后答案短语似然下降。
other_fruit_minus_nonfruit > 0 表示更像水果共享骨架。
apple_minus_other_fruit > 0 表示更像苹果特异差分。
```

### 结果文件

```text
results/glm5_phase723_apple_fruit_attribute_micro_atlas/phase723_cross_model_summary.md
results/glm5_phase723_apple_fruit_attribute_micro_atlas/phase723_cross_model_summary.json
results/glm5_phase723_apple_fruit_attribute_micro_atlas/phase723_qwen3_micro_atlas_rows.jsonl
results/glm5_phase723_apple_fruit_attribute_micro_atlas/phase723_glm4_micro_atlas_rows.jsonl
results/glm5_phase723_apple_fruit_attribute_micro_atlas/phase723_deepseek7b_micro_atlas_rows.jsonl
```

每个模型：

```text
n_cases = 114
n_rows = 684
```

### 关键客观结果

qwen3：

```text
L24H29:
  mean_logprob_delta = -0.0611
  first_rank_delta = 0.08
  top1_drop = 0.035
  apple_need = 0.1146
  fruit_need = 0.0557
  nonfruit_need = 0.1195
  fruit_minus_nonfruit = -0.0638
  apple_minus_fruit = 0.0589
```

qwen3 的候选 head 有小幅必要性，但 fruit_need 小于 nonfruit_need，不支持清晰水果共享骨架；更像局部属性/格式/yes-no 等混合效应。

GLM4：

```text
L29H28:
  mean_logprob_delta = -0.0059
  first_rank_delta = 0.01
  top1_drop = 0.000
  apple_need = 0.0168
  fruit_need = 0.0104
  nonfruit_need = 0.0085
  fruit_minus_nonfruit = 0.0018
  apple_minus_fruit = 0.0065
```

GLM4 的效应非常弱，虽然 L29H28 有极小水果共享倾向，但不足以形成强机制结论。

DS7B：

```text
L20H17:
  mean_logprob_delta = -0.3161
  first_rank_delta = 4.11
  top1_drop = 0.061
  apple_need = 0.2269
  fruit_need = 0.2819
  nonfruit_need = 0.2440
  fruit_minus_nonfruit = 0.0379
  apple_minus_fruit = -0.0550

L27H23:
  mean_logprob_delta = -0.1885
  first_rank_delta = 2.16
  top1_drop = 0.044
  apple_need = 0.0840
  fruit_need = 0.3778
  nonfruit_need = 0.0437
  fruit_minus_nonfruit = 0.3341
  apple_minus_fruit = -0.2938

L23H0:
  mean_logprob_delta = -0.1253
  first_rank_delta = 11.56
  top1_drop = 0.061
  apple_need = 0.1541
  fruit_need = 0.1867
  nonfruit_need = 0.0317
  fruit_minus_nonfruit = 0.1549
  apple_minus_fruit = -0.0326
```

DS7B 出现清晰结果：

```text
L27H23 和 L23H0 更像 other_fruit 共享路线。
L20H17 是更强的全局答案/类别/属性支撑 head，但不是苹果特异 head。
```

同层随机对照中，DS7B 的 L20H18 也有较强效应：

```text
L20H18 random:
  mean_logprob_delta = -0.2207
  fruit_minus_nonfruit = 0.0823
```

这说明：

```text
DS7B 的 L20 附近可能存在层级通路或 head cluster（注意力头簇），不能把全部机制归因到单个 L20H17。
```

### 阶段判断

本阶段支持以下判断：

```text
1. Phase 722 的强 DS7B fruit heads 不是偶然 first-token 现象，在 phrase likelihood 上仍有明显效应。
2. “水果共享骨架”在 DS7B 上比 qwen3 / GLM4 更清楚。
3. 当前结果没有找到清晰“苹果特异 head”；苹果可能不是由单个 head 编码，而是由水果共享骨架 + 其他属性/词嵌入/MLP 差分共同决定。
4. qwen3 / GLM4 的弱效应不能直接解释为没有机制，更可能是冗余、分布式或小模型架构差异。
```

### 严格问题和硬伤

```text
1. 仍然是 head 级，不是 neuron / channel 级。
2. zero ablation 是 off-manifold 干预，可能制造非自然扰动。
3. 本阶段是 teacher-forced likelihood，不是自然生成闭环。
4. apple 特异差分没有闭合，说明当前候选更偏水果共享路线，而不是对象身份编码。
5. DS7B 的随机同层 head 也有较强效应，说明必须从单 head 进入 cluster / subspace / channel 级分析。
6. 当前模型都是小模型，内部结构可能和大模型存在偏差，不能把 DS7B 的结构直接当成通用语言机制。
```

### 理论进展

当前更合理的描述不是：

```text
苹果由某个 head 编码。
```

而是：

```text
对象答案生成 = 共享类别骨架 + 属性条件化差分 + 词嵌入先验 + 下游读出竞争
```

在 DS7B 中，当前已看到：

```text
fruit-shared route 的因果迹象 > apple-specific route 的因果迹象。
```

这与“条件化相对状态—生成场闭合理论”一致：语言不是单点概念向量读出，而是在上下文条件下形成相对状态，再由多个共享/差分通路共同闭合到输出。

### 下一步

Phase 724 与 Phase 723 属于同一个阶段性目标：完成 apple-fruit-attribute micro-atlas（苹果-水果-属性微图谱）的机制闭合。

下一步不应继续扩大全局图谱，而应下钻 DS7B 的强路线：

```text
Phase 724: DS7B Fruit Route Cluster and Channel Drilldown

目标：
1. 围绕 L20H17 / L20H18 / L27H23 / L23H0 建立 head cluster。
2. 分离 category / color / taste / shape / yes-no 属性路线。
3. 对 W_O 输出通道做 channel-level causal scan。
4. 判断水果共享骨架到底来自 head 输出、MLP 放大，还是 residual route。
5. 若 channel 级出现稳定子集，再进入 neuron-level atlas。
```

## Phase 724: Fruit Route Channel Group Drilldown [2026-06-28 16:15]

### 触发问题

Phase 723 已确认：apple-fruit-attribute（苹果-水果-属性）微世界中，DS7B 的 fruit route（水果路线）有明显因果效应，但效应仍停留在 head（注意力头）级。根据用户要求，如果下一任务仍处于同一阶段，就继续自动完成。因此本阶段继续在同一阶段性目标内推进：从 head 级进入 channel group（通道组）级。

### 本阶段脚本

```text
tests/gpt5/phase724_fruit_route_channel_group_drilldown.py
tests/gpt5/run_phase724_fruit_route_channel_group_drilldown_full.sh
```

### 运行命令

```bash
tests/gpt5/run_phase724_fruit_route_channel_group_drilldown_full.sh
```

按顺序运行：

```bash
python tests/gpt5/phase724_fruit_route_channel_group_drilldown.py --model qwen3 --hard-exit-after-model
python tests/gpt5/phase724_fruit_route_channel_group_drilldown.py --model glm4 --hard-exit-after-model
python tests/gpt5/phase724_fruit_route_channel_group_drilldown.py --model deepseek7b --hard-exit-after-model
python tests/gpt5/phase724_fruit_route_channel_group_drilldown.py --summarize-only
```

### 测试原理

本阶段复用 Phase 723 的 114 个 apple-fruit-attribute（苹果-水果-属性）案例。每个模型选 Phase 723 最强的 2 个候选 head，把每个 head 的 W_O 输入向量按连续 channel group（通道组）切成 8 段，逐段置零，然后测正确答案短语的 teacher-forced likelihood（教师强制似然）下降。

核心思想：

```text
Phase 723:
  哪些 head 重要？

Phase 724:
  这些 head 的效应是否集中在少数输出通道组？
```

本阶段不是 neuron-level（神经元级）解释，只是从 head 级向 channel/subspace（通道/子空间）级推进的粗筛。

### 结果文件

```text
results/glm5_phase724_fruit_route_channel_group_drilldown/phase724_cross_model_summary.md
results/glm5_phase724_fruit_route_channel_group_drilldown/phase724_cross_model_summary.json
results/glm5_phase724_fruit_route_channel_group_drilldown/phase724_qwen3_channel_group_rows.jsonl
results/glm5_phase724_fruit_route_channel_group_drilldown/phase724_glm4_channel_group_rows.jsonl
results/glm5_phase724_fruit_route_channel_group_drilldown/phase724_deepseek7b_channel_group_rows.jsonl
```

每个模型：

```text
n_cases = 114
n_rows = 1824
```

三模型总计：

```text
total_rows = 5472
```

### 关键客观结果

qwen3：

```text
L24H29 channel 112-128:
  mean_logprob_delta = -0.0151
  rank_delta = 0.03
  top1_drop = 0.009
  fruit_minus_nonfruit = 0.0015
  apple_minus_fruit = -0.0075
```

qwen3 的通道组效应很弱，且水果共享不清晰。

GLM4：

```text
L29H28 channel 32-48:
  mean_logprob_delta = -0.0028
  rank_delta = 0.01
  top1_drop = 0.000
  fruit_minus_nonfruit = 0.0009
  apple_minus_fruit = 0.0016
```

GLM4 的效应接近噪声级，暂不支持强通道定位。

DS7B：

```text
L20H17 channel 16-32:
  mean_logprob_delta = -0.1499
  rank_delta = 0.39
  top1_drop = 0.035
  fruit_minus_nonfruit = 0.0534
  apple_minus_fruit = 0.0026

L20H17 channel 0-16:
  mean_logprob_delta = -0.0411
  rank_delta = 5.13
  top1_drop = 0.026
  fruit_minus_nonfruit = -0.0023
  apple_minus_fruit = -0.0178

L27H23 channel 80-96:
  mean_logprob_delta = -0.0214
  rank_delta = 0.39
  top1_drop = 0.018
  fruit_minus_nonfruit = 0.0697
  apple_minus_fruit = -0.0494

L27H23 channel 16-32:
  mean_logprob_delta = -0.0180
  fruit_minus_nonfruit = 0.0393

L27H23 channel 32-48:
  mean_logprob_delta = -0.0147
  fruit_minus_nonfruit = 0.0361
```

DS7B 的结果最清晰：

```text
1. L20H17 的主要强效应集中在 channel 16-32。
2. L20H17 channel 16-32 对 category（类别）的必要性尤其强。
3. L27H23 的水果共享效应不集中在单一通道组，而是在 80-96、48-64、16-32、32-48 等多个组都有迹象。
4. L20H17 更像全局类别/答案支撑子空间；L27H23 更像分布式水果共享子空间。
```

其中 DS7B L20H17 channel 16-32 的 relation necessity（关系必要性）：

```text
category = 0.6019
color = 0.0072
edible = 0.0638
grows_on_tree = 0.0146
shape = 0.0599
taste = 0.0672
```

这说明该通道组更接近 category route（类别路线），不是均匀属性路线。

### 阶段判断

Phase 724 支持以下判断：

```text
1. DS7B 的强 fruit route 不是平均分布在整个 head_dim 中，至少 L20H17 存在明显的粗通道集中。
2. L20H17 channel 16-32 是目前最值得下钻的 category route 候选。
3. L27H23 不是单点强通道，而更像多个通道组共同形成水果共享路线。
4. qwen3 / GLM4 没有形成同等清晰结构，不能直接用于证明通用机制。
```

### 严格问题和硬伤

```text
1. channel group 是连续粗切片，不是真正自动发现的语义通道。
2. 置零仍是 off-manifold 干预。
3. 本阶段没有测试 W_O 输出后的 residual 传播，也没有测试 MLP 放大。
4. L20H17 channel 16-32 的 category 强效应可能是类别读出路线，也可能是答案格式/高频类别词路线。
5. L27H23 的分布式效应需要更细粒度 channel scan，否则不能确定是否存在稀疏子结构。
6. 当前还没有完成 neuron-level atlas（神经元级图谱）。
```

### 理论进展

当前从 Phase 723 到 Phase 724，机制拼图从：

```text
重要 head
```

推进到：

```text
重要 head 内部的粗通道组
```

最重要的新现象是：

```text
DS7B 的 category route 在 L20H17 channel 16-32 出现局部集中。
```

这支持“共享骨架 + 差分条件化”的路线，但目前更像 category shared subspace（类别共享子空间），还不是 apple-specific differential（苹果特异差分）。

### 下一步

Phase 725 仍属于同一阶段性目标，但需要更细：

```text
Phase 725: DS7B L20H17 Channel 16-32 Fine Scan and Residual Propagation

目标：
1. 对 L20H17 channel 16-32 做单 channel 或小组 channel scan。
2. 对 L27H23 的 80-96 / 16-32 / 32-48 做细扫。
3. 追踪这些 channel group 写入后，在后续 residual / MLP / readout 中是否被放大。
4. 区分类别词 route、格式 route、对象身份 route。
5. 如果细 channel 稳定，再进入 neuron-level graph atlas。
```

## Phase 725: Fine Channel Category Route Scan [2026-06-28 17:00]

### 触发问题

用户上传的 Phase 723-724 分析基本正确：当前路线已经从 global functional atlas（全局功能图谱）收缩到 apple-fruit-attribute（苹果-水果-属性）微世界，并从 head（注意力头）进入 channel group（通道组）。但 Phase 724 仍有硬伤：channel group 是粗切片，L20H17 channel 16-32 可能只是类别词 / 短答格式 / 高频词路线，而不一定是 category route（类别路线）。

因此 Phase 725 继续同一阶段性目标：对 Phase 724 的高效通道组做 single-channel fine scan（单通道细扫），并加入 category selectivity（类别选择性）指标。

### 本阶段脚本

```text
tests/gpt5/phase725_fine_channel_category_route_scan.py
tests/gpt5/run_phase725_fine_channel_category_route_scan_full.sh
```

### 运行命令

```bash
tests/gpt5/run_phase725_fine_channel_category_route_scan_full.sh
```

按顺序运行：

```bash
python tests/gpt5/phase725_fine_channel_category_route_scan.py --model qwen3 --hard-exit-after-model
python tests/gpt5/phase725_fine_channel_category_route_scan.py --model glm4 --hard-exit-after-model
python tests/gpt5/phase725_fine_channel_category_route_scan.py --model deepseek7b --hard-exit-after-model
python tests/gpt5/phase725_fine_channel_category_route_scan.py --summarize-only
```

### 测试原理

本阶段读取 Phase 724 中每个模型最强的 2 个 channel group（通道组），把每个 16 维通道组拆成 16 个 single channel（单通道），逐个置零，再测 114 个 apple-fruit-attribute（苹果-水果-属性）案例的答案短语似然变化。

核心指标：

```text
Need(channel) = - mean_logprob_delta
```

新增 category selectivity（类别选择性）：

```text
category_selectivity
  = Need(category)
    - mean(Need(color), Need(taste), Need(shape), Need(edible), Need(grows_on_tree))
```

解释：

```text
如果某个 channel 的 category_selectivity 明显为正，
说明它更偏 category route（类别路线），
而不是均匀影响所有短答关系。
```

### 结果文件

```text
results/glm5_phase725_fine_channel_category_route_scan/phase725_cross_model_summary.md
results/glm5_phase725_fine_channel_category_route_scan/phase725_cross_model_summary.json
results/glm5_phase725_fine_channel_category_route_scan/phase725_qwen3_fine_channel_rows.jsonl
results/glm5_phase725_fine_channel_category_route_scan/phase725_glm4_fine_channel_rows.jsonl
results/glm5_phase725_fine_channel_category_route_scan/phase725_deepseek7b_fine_channel_rows.jsonl
```

每个模型：

```text
n_cases = 114
n_rows = 3648
```

三模型总计：

```text
total_rows = 10944
```

### 关键客观结果

qwen3：

```text
最强 harmful channel:
  L24H29 channel 10
  mean_logprob_delta = -0.0096
  category_selectivity = -0.0356
  fruit_minus_nonfruit = -0.0068
  apple_minus_fruit = 0.0332
```

qwen3 没有找到 category-selective channel（类别选择性通道）。它的最强伤害更像 yes-no / edible（是否可食用）或格式混合效应。

GLM4：

```text
最强 harmful channel:
  L24H19 channel 73
  mean_logprob_delta = -0.0028
  category_selectivity = 0.0006
  fruit_minus_nonfruit = -0.0005
  apple_minus_fruit = 0.0004
```

GLM4 结果接近噪声级，不支持当前候选下的单通道类别路线。

DS7B：

```text
L20H17 channel 25:
  mean_logprob_delta = -0.0520
  rank_delta = 0.98
  top1_drop = 0.018
  category_selectivity = 0.2176
  category_need = 0.2271
  fruit_minus_nonfruit = 0.0001
  apple_minus_fruit = -0.0101

L20H17 channel 30:
  mean_logprob_delta = -0.0504
  rank_delta = 1.58
  top1_drop = 0.009
  category_selectivity = 0.1732
  category_need = 0.1901

L20H17 channel 24:
  mean_logprob_delta = -0.0430
  rank_delta = 1.20
  top1_drop = 0.018
  category_selectivity = 0.1793
  category_need = 0.1886
  fruit_minus_nonfruit = 0.0117
  apple_minus_fruit = 0.0231

L20H17 channel 23:
  mean_logprob_delta = -0.0275
  category_selectivity = 0.1250
  fruit_minus_nonfruit = 0.0193
  apple_minus_fruit = 0.0348
```

DS7B 的关键结果：

```text
1. Phase 724 的 L20H17 channel 16-32 强效应不是均匀分布。
2. 其中 channel 25 / 30 / 24 / 23 是最清晰的单通道候选。
3. channel 25 / 30 / 24 对 category 明显更强，支持 category route 候选。
4. fruit_minus_nonfruit 仍然很弱，说明这些单通道更像 category-selective route，而不是 fruit-specific shared route。
5. apple-specific code 仍未闭合。
```

### 阶段判断

Phase 725 支持以下判断：

```text
1. DS7B L20H17 channel 16-32 内部存在更细的高效单通道。
2. 最强单通道是 L20H17 channel 25 / 30 / 24。
3. 这些通道更像 category route（类别路线），而不是完整 fruit-shared route（水果共享路线）。
4. qwen3 / GLM4 仍没有同等清晰结构。
5. 当前机制从 head-level atlas 推进到 single-channel candidate，但还不是 neuron-level code。
```

### 严格问题和硬伤

```text
1. single-channel zero ablation 仍然是 off-manifold。
2. single channel 不是神经元；它只是 attention head output 子空间的坐标。
3. category_selectivity 仍基于模板任务，不等于自然语言类别理解。
4. 这些通道对 fruit_minus_nonfruit 不强，说明它们更可能是一般 category route，而不是水果专属路线。
5. 还没有测试 residual / MLP 是否放大这些单通道。
6. 没有做自然生成闭合。
7. 小模型偏差仍然存在，DS7B 结果不能直接推广为通用语言机制。
```

### 理论进展

从 Phase 723 到 Phase 725，当前拼图变成：

```text
fruit-shared head route:
  DS7B L27H23 / L23H0

category-support head route:
  DS7B L20H17

category-selective fine channels:
  DS7B L20H17 channel 25 / 30 / 24 / 23
```

更准确理论：

```text
苹果-水果-属性任务中，
category route 与 fruit-shared route 不是同一个东西。

L20H17 负责更一般的类别/短答候选支撑；
L27H23 更像水果共享路线；
苹果特异差分尚未定位，可能在词嵌入、MLP 或读出竞争中。
```

### 下一步

Phase 726 与当前阶段仍相关，但已经从定位进入传播验证：

```text
Phase 726: DS7B Category Channel Residual/MLP Propagation Audit

目标：
1. 追踪 L20H17 channel 25 / 30 / 24 的写入是否在后续 residual 中保持或放大。
2. 检查后续 MLP 是否把这些 category-selective channels 转换成最终 logits。
3. 对 category / format / high-frequency label 做更强对照。
4. 加入 natural generation closure，确认消融是否真的改变自然输出。
```

## Phase 726: Category Channel Natural Generation Closure [2026-06-28 17:04]

### 触发问题

Phase 725 找到了 DS7B L20H17 channel 25 / 30 / 24 等 category-selective channel（类别选择性通道），但仍然只是 teacher-forced likelihood（教师强制似然）层面的证据。用户上传的分析明确指出：必须验证 natural generation closure（自然生成闭合）。因此本阶段继续同一阶段目标，测试单通道消融是否真的改变自然贪婪生成输出。

### 本阶段脚本

```text
tests/gpt5/phase726_category_channel_generation_closure.py
tests/gpt5/run_phase726_category_channel_generation_closure_full.sh
```

### 运行命令

```bash
tests/gpt5/run_phase726_category_channel_generation_closure_full.sh
```

按顺序运行：

```bash
python tests/gpt5/phase726_category_channel_generation_closure.py --model qwen3 --hard-exit-after-model
python tests/gpt5/phase726_category_channel_generation_closure.py --model glm4 --hard-exit-after-model
python tests/gpt5/phase726_category_channel_generation_closure.py --model deepseek7b --hard-exit-after-model
python tests/gpt5/phase726_category_channel_generation_closure.py --summarize-only
```

### 测试原理

本阶段从 Phase 725 中读取每个模型 category_selectivity（类别选择性）最高的单通道：

```text
qwen3: L24H29 channel 119
GLM4: L24H19 channel 69
DS7B: L20H17 channel 25
```

然后只取 Phase 723 的 category（类别）问题，共 22 个案例，对 baseline（原始）和 single-channel ablation（单通道消融）分别做 greedy natural generation（贪婪自然生成），最多生成 4 个 token，比较：

```text
changed_rate：输出文本是否变化。
baseline_hit_rate：baseline 是否命中目标类别。
ablated_hit_rate：消融后是否命中目标类别。
hit_drop_rate：baseline 命中但消融后不命中的比例。
```

### 结果文件

```text
results/glm5_phase726_category_channel_generation_closure/phase726_cross_model_summary.md
results/glm5_phase726_category_channel_generation_closure/phase726_cross_model_summary.json
results/glm5_phase726_category_channel_generation_closure/phase726_qwen3_generation_rows.jsonl
results/glm5_phase726_category_channel_generation_closure/phase726_glm4_generation_rows.jsonl
results/glm5_phase726_category_channel_generation_closure/phase726_deepseek7b_generation_rows.jsonl
```

每个模型：

```text
n_cases = 22
```

三模型总计：

```text
total_rows = 66
```

### 关键客观结果

```text
qwen3 L24H29:119:
  changed_rate = 0.000
  baseline_hit_rate = 0.955
  ablated_hit_rate = 0.955
  hit_drop_rate = 0.000

GLM4 L24H19:69:
  changed_rate = 0.000
  baseline_hit_rate = 0.955
  ablated_hit_rate = 0.955
  hit_drop_rate = 0.000

DS7B L20H17:25:
  changed_rate = 0.045
  baseline_hit_rate = 0.500
  ablated_hit_rate = 0.545
  hit_drop_rate = 0.000
```

DS7B 只有 1/22 个案例发生生成文本变化：

```text
case: lemon commonsense category
baseline: Fruits.
ablated: Fruit.
target: fruit
```

这个变化不是错误增加，而是从复数 Fruits 变成单数 Fruit。

### 阶段判断

Phase 726 是重要负结果：

```text
1. Phase 725 的 DS7B L20H17 channel 25 确实影响 category likelihood。
2. 但单独消融该 channel 几乎不改变自然贪婪生成。
3. 因此它不是单通道 generation switch（生成开关）。
4. 它更像 category candidate likelihood support（类别候选似然支撑）的一部分。
5. 自然生成闭合需要多个 channel / head cluster / downstream MLP / readout 共同干预。
```

### 严格问题和硬伤

```text
1. 只测 greedy decoding（贪婪解码），未测采样。
2. 只测 category 关系，未测其他属性。
3. 只消融单通道，可能太弱。
4. DS7B baseline category generation 本身只有 0.500 hit rate，说明自然生成任务更难、更不稳定。
5. 仍未完成 residual / MLP propagation（残差 / MLP 传播）验证。
```

### 理论进展

最新判断需要收紧：

```text
DS7B L20H17 channel 25 / 30 / 24
不是完整类别生成机制，
而是类别候选似然支撑子通道。
```

更准确机制图：

```text
category likelihood support:
  L20H17 channels 24/25/30/23

fruit-shared route:
  L27H23 / L23H0 and distributed subspace

natural generation closure:
  not closed by single channel
```

### 下一步

当前 apple-fruit-attribute micro-atlas（苹果-水果-属性微图谱）阶段已经完成从：

```text
head -> channel group -> single channel -> natural generation sanity check
```

的第一轮闭合。

下一步如果继续同一大方向，应进入 cluster-level intervention（簇级干预），而不是继续单通道：

```text
Phase 727: DS7B Category/Fruit Route Cluster Intervention

目标：
1. 同时消融 L20H17 channels 24/25/30/23。
2. 同时消融 L27H23 fruit-shared channel groups。
3. 比较 single-channel、multi-channel、full-head 的 generation closure。
4. 观察是否只有 cluster-level 才能改变自然生成。
```

## Phase 727: Category/Fruit Route Cluster Intervention [2026-06-28 17:18]

### 触发问题

用户上传的 Phase 725-726 分析基本正确：单通道能影响 category likelihood（类别似然），但不能闭合 natural generation（自然生成）。因此下一步不能继续单通道，而应测试 cluster-level intervention（簇级干预）：如果多通道/多头簇仍不能改变自然生成，就说明生成闭合瓶颈更可能在 full head（整头）、downstream MLP（下游多层感知机）或 readout gate（读出门）。

### 本阶段脚本

```text
tests/gpt5/phase727_category_fruit_cluster_intervention.py
tests/gpt5/run_phase727_category_fruit_cluster_intervention_full.sh
```

### 运行命令

```bash
tests/gpt5/run_phase727_category_fruit_cluster_intervention_full.sh
```

按顺序运行：

```bash
python tests/gpt5/phase727_category_fruit_cluster_intervention.py --model qwen3 --hard-exit-after-model
python tests/gpt5/phase727_category_fruit_cluster_intervention.py --model glm4 --hard-exit-after-model
python tests/gpt5/phase727_category_fruit_cluster_intervention.py --model deepseek7b --hard-exit-after-model
python tests/gpt5/phase727_category_fruit_cluster_intervention.py --summarize-only
```

### 测试原理

本阶段只测试 category（类别）问题，共 22 个案例。每个模型比较以下干预：

```text
baseline：无干预。
category_single：最高 category-selective 单通道。
category_cluster：top category channels 簇。
category_full_head：对应 category head 整头消融。
fruit_cluster：fruit-shared channel groups。
category_plus_fruit_cluster：category cluster + fruit cluster。
```

对每个干预同时记录：

```text
1. teacher-forced answer phrase likelihood delta。
2. greedy natural generation hit rate。
3. changed_rate_vs_baseline。
4. hit_drop_rate_vs_baseline。
```

### 结果文件

```text
results/glm5_phase727_category_fruit_cluster_intervention/phase727_cross_model_summary.md
results/glm5_phase727_category_fruit_cluster_intervention/phase727_cross_model_summary.json
results/glm5_phase727_category_fruit_cluster_intervention/phase727_qwen3_cluster_rows.jsonl
results/glm5_phase727_category_fruit_cluster_intervention/phase727_glm4_cluster_rows.jsonl
results/glm5_phase727_category_fruit_cluster_intervention/phase727_deepseek7b_cluster_rows.jsonl
```

每个模型：

```text
n_cases = 22
n_rows = 132
```

三模型总计：

```text
total_rows = 396
```

### 关键客观结果

qwen3：

```text
category_cluster:
  mean_logprob_delta = +0.0184
  hit_rate = 0.955
  changed_rate = 0.000
  hit_drop = 0.000

category_full_head:
  mean_logprob_delta = -0.0159
  hit_rate = 0.955
  changed_rate = 0.000
  hit_drop = 0.000
```

qwen3 没有生成闭合效应。

GLM4：

```text
category_cluster:
  mean_logprob_delta = +0.0006
  hit_rate = 0.955
  changed_rate = 0.000
  hit_drop = 0.000

category_full_head:
  mean_logprob_delta = +0.0016
  hit_rate = 0.955
  changed_rate = 0.000
  hit_drop = 0.000
```

GLM4 仍接近噪声。

DS7B：

```text
category_single:
  mean_logprob_delta = -0.2271
  hit_rate = 0.545
  changed_rate = 0.045
  hit_drop = 0.000
  rank_delta = 3.14

category_cluster:
  mean_logprob_delta = -0.2613
  hit_rate = 0.500
  changed_rate = 0.000
  hit_drop = 0.000
  rank_delta = 0.68

fruit_cluster:
  mean_logprob_delta = -0.0582
  hit_rate = 0.500
  changed_rate = 0.000
  hit_drop = 0.000

category_plus_fruit_cluster:
  mean_logprob_delta = -0.3463
  hit_rate = 0.500
  changed_rate = 0.000
  hit_drop = 0.000
  rank_delta = 1.00

category_full_head:
  mean_logprob_delta = -1.0575
  hit_rate = 0.500
  changed_rate = 0.409
  hit_drop = 0.091
  rank_delta = 4.50
```

### 阶段判断

Phase 727 给出非常关键的边界：

```text
1. DS7B category_cluster 能明显降低 category likelihood。
2. category_plus_fruit_cluster 降低似然更强。
3. 但这些 cluster 仍没有改变 greedy natural generation。
4. 只有 category_full_head 才明显改变生成文本，并带来少量 hit_drop。
```

因此：

```text
category likelihood support
  可以由少数 channel cluster 承担；

natural generation closure
  不是这些 channel cluster 单独完成；
  更接近 full head output / downstream MLP / readout gate 级别。
```

### 严格问题和硬伤

```text
1. 本阶段仍只测 category 问题。
2. natural generation 使用 greedy decoding，不含采样。
3. category_full_head 更强但定位更粗。
4. DS7B baseline hit_rate 只有 0.5，说明自然生成任务本身不稳定。
5. cluster 不改变生成，可能是下游补偿，也可能是当前 prompt / decoding 不敏感。
6. 仍未完成 residual / MLP propagation。
```

### 理论进展

从 Phase 725-727，理论进一步收紧为：

```text
likelihood support cluster
  !=
generation closure mechanism
```

更准确机制图：

```text
category likelihood support:
  L20H17 channels 24/25/30/23

fruit-shared support:
  L27H23 channel groups

generation-sensitive unit:
  closer to full L20H17 head output,
  or downstream MLP / readout gate.
```

这说明语言编码机制不是“单通道语义开关”，而是：

```text
局部候选似然支撑
→ 通路/整头级整合
→ 下游残差和 MLP 转换
→ 读出竞争
→ 自然生成闭合
```

### 下一步

当前 apple-fruit-attribute micro-atlas 的 head-to-channel-to-cluster 第一轮已经完成。下一步仍处于同一大方向，但应转向 propagation（传播）：

```text
Phase 728: DS7B Full-Head vs Channel-Cluster Residual Propagation

目标：
1. 比较 category_cluster 和 category_full_head 在后续层 residual 上造成的差异。
2. 测试 L20H17 full-head 为什么能改变生成，而 channel cluster 不能。
3. 检查 L21-L24 MLP 是否放大或恢复 category cluster 扰动。
4. 明确 generation closure 的瓶颈位置。
```

## Phase 728: Atlas Graph v1 3D Visualization Client [2026-06-28 17:50]

### 触发问题

用户要求：

```text
可视化客户端中，可以加载图谱的测试结果，在3d空间中显示出来；
同时生成一份文件格式说明，保证后面会生成相同格式的数据，方便查看和研究。
```

### 生成和修改的文件

```text
frontend/src/neural_vis/hooks/useVisData.js
frontend/src/neural_vis/index.jsx
frontend/src/neural_vis/renderers/AtlasGraphRenderer.jsx
frontend/ATLAS_GRAPH_FORMAT.md
tests/gpt5/build_phase727_atlas_graph.py
results/glm5_phase727_category_fruit_cluster_intervention/phase727_atlas_graph.json
```

### 命令

```bash
python tests/gpt5/build_phase727_atlas_graph.py
npm run build
```

### 原理

本阶段没有进行新模型推理，而是把已有 Phase 727 测试结果转换成统一图谱格式：

```text
atlas_graph_v1
  graph.nodes = 机制对象
  graph.edges = 机制关系 / 因果证据 / 失败边界
```

3D 坐标采用可解释布局：

```text
x = component offset + head/channel index
y = layer index
z = model lane
```

因此图谱不是普通展示图，而是可以直接观察：

```text
模型 → Phase → task → intervention → head/channel/cluster → failure boundary
```

### 客观结果

生成 Phase 727 样例图谱：

```text
schema_version = atlas_graph_v1
node_count = 54
edge_count = 71
source_phase = 727
```

前端构建验证：

```text
npm run build
结果：通过
```

### 理论和工程进展

本阶段把研究从文字 memo 和 JSON summary 推进到可累计的 mechanism atlas 数据层：

```text
Phase 727 原始结果
  → atlas_graph_v1
  → 3D client render
  → hover 查看 role / evidence / Δlogp / generation changed
```

这使后续 apple-fruit-attribute micro-atlas 可以持续积累，不再只是每个 Phase 孤立记录。

### 严格问题和硬伤

```text
1. 当前只是第一版图谱格式，不是完整全局神经元图谱。
2. 3D 布局是可解释坐标，不是从真实激活空间降维得到。
3. Phase 727 图谱主要来自 summary 和 Phase 724/725 component hints，仍依赖前序实验质量。
4. 当前客户端支持 JSON 文件加载，不支持 JSONL 原始行直接加载。
5. atlas_graph_v1 目前只覆盖 head/channel/cluster/intervention 层级，还没有接入 neuron 级别。
```

### 下一步

下一阶段可以继续 Phase 728 原计划的 propagation 测试，但编号应顺延为 Phase 729：

```text
Phase 729: DS7B Full-Head vs Channel-Cluster Residual Propagation

目标：
1. 比较 category_cluster 和 category_full_head 对 L21-L24 residual 的影响。
2. 判断 channel cluster 的扰动是在下游被冲洗、恢复，还是从未进入关键读出路径。
3. 把传播结果继续输出为 atlas_graph_v1。
4. 在 3D 图谱中显示 likelihood support 和 generation closure 的分叉位置。
```

## Phase 729: Full-Head vs Channel-Cluster Residual Propagation [2026-06-28 18:14]

### 触发问题

用户提供的 Phase 727 分析基本正确：

```text
likelihood support cluster
  !=
generation closure mechanism
```

Phase 727 已经证明 category_cluster 能影响 likelihood，但不能改变 natural generation；只有 category_full_head 明显改变生成文本。因此本阶段继续完成同一阶段目标中的 propagation 测量。

### 脚本和结果文件

```text
tests/gpt5/phase729_full_head_vs_cluster_residual_propagation.py
tests/gpt5/run_phase729_full_head_vs_cluster_residual_propagation_full.sh
tests/gpt5/build_phase729_atlas_graph.py
results/glm5_phase729_full_head_vs_cluster_residual_propagation/
results/glm5_phase729_full_head_vs_cluster_residual_propagation/phase729_cross_model_summary.md
results/glm5_phase729_full_head_vs_cluster_residual_propagation/phase729_cross_model_summary.json
results/glm5_phase729_full_head_vs_cluster_residual_propagation/phase729_atlas_graph.json
```

### 命令

```bash
bash tests/gpt5/run_phase729_full_head_vs_cluster_residual_propagation_full.sh
python tests/gpt5/build_phase729_atlas_graph.py
```

运行顺序：

```text
qwen3 → GLM4 → DS7B
```

并且每个模型都使用：

```text
--hard-exit-after-model
```

### 测试原理

本阶段不做新自然生成，而是对 Phase 727 的 category cases 做一次前向传播采样。

对同一 prompt 分别运行：

```text
baseline
category_cluster
category_full_head
category_plus_fruit_cluster
```

然后比较：

```text
delta(h_l) = h_l(intervention) - h_l(baseline)
```

核心指标：

```text
source_delta_norm:
  干预源层后的扰动范数。

delta_norm:
  后续层 / component 的扰动范数。

amplification_vs_source:
  delta_norm / source_delta_norm

component_vs_layer_input:
  attention 或 MLP output 扰动 / 同层 input 扰动

cos_with_final_delta:
  当前扰动和最终 hidden 扰动的方向一致性。
```

### 客观结果

跨模型汇总：

```text
qwen3:
  category_cluster:
    max_layer_amp = 7.112
    top_site = hidden_33
    top_delta = 5.891
    MLP/input = 0.536
    attn/input = 0.461

  category_full_head:
    max_layer_amp = 2.652
    top_site = hidden_33
    top_delta = 27.987
    MLP/input = 0.517
    attn/input = 0.407

GLM4:
  category_cluster:
    max_layer_amp = 10.784
    top_site = hidden_40
    top_delta = 2.599
    MLP/input = 0.465
    attn/input = 0.389

  category_full_head:
    max_layer_amp = 3.978
    top_site = hidden_40
    top_delta = 5.406
    MLP/input = 0.405
    attn/input = 0.343

DS7B:
  category_cluster:
    max_layer_amp = 4.105
    top_site = hidden_27
    top_delta = 31.437
    MLP/input = 0.683
    attn/input = 0.329

  category_full_head:
    max_layer_amp = 3.042
    top_site = hidden_27
    top_delta = 97.070
    MLP/input = 0.626
    attn/input = 0.255
```

Phase 729 atlas graph：

```text
schema_version = atlas_graph_v1
node_count = 72
edge_count = 72
```

### 阶段判断

本阶段结果进一步支持 Phase 727 的边界结论。

关键观察：

```text
1. category_cluster 不是没有传播。
2. category_cluster 在 residual trajectory 中会被后续层放大。
3. 但是 category_full_head 的绝对扰动幅度显著更大。
4. DS7B 中 full head top_delta = 97.070，而 cluster top_delta = 31.437。
5. MLP/input 明显高于 attention/input，特别是 DS7B: 0.683 vs 0.329。
```

因此，当前更准确的图像是：

```text
channel cluster
  形成可传播的 likelihood-support perturbation；

full head
  形成更大、更完整的 residual trajectory perturbation；

downstream MLP
  对扰动有更强响应；

generation closure
  仍不等于 residual perturbation 本身，
  还需要 readout competition / decoding route 级别闭合。
```

### 严格问题和硬伤

```text
1. Phase 729 是传播测量，不是自然生成闭合测试。
2. max_layer_amp 是相对源扰动的放大比例，不能单独解释为因果强度。
3. full_head 绝对扰动更大，但也更粗，不能说明具体 channel 已定位完整。
4. component_vs_layer_input 是诊断指标，不是 MLP 因果证明。
5. 小模型内部结构可能有偏差，尤其 qwen3/GLM4 的候选头未必是同构机制。
6. DS7B category_plus_fruit_cluster 和 category_cluster 在本阶段传播指标接近，说明新增 fruit cluster 未显著改变传播主轨迹。
```

### 理论进展

Phase 727 到 Phase 729 后，机制拼图更新为：

```text
category channel cluster:
  likelihood support
  + residual propagation
  - generation closure

category full head:
  stronger residual perturbation
  + visible generation sensitivity
  - still too coarse

downstream MLP:
  stronger diagnostic response than attention output
  but not yet causal closure
```

因此语言编码机制更像：

```text
局部通道簇支撑似然
→ 整头输出形成完整传播轨迹
→ MLP / residual 更新扩大或改写轨迹
→ 读出端竞争决定自然生成
```

### 下一步

下一阶段应继续同一大任务，但从“传播测量”推进到“传播节点干预”：

```text
Phase 730: DS7B Downstream MLP/Residual Bottleneck Intervention

目标：
1. 对 DS7B L22-L27 中 top propagation sites 做局部干预。
2. 比较 hidden_27、L24_mlp_out、L22_mlp_out 等节点是否真正影响 category likelihood 和 generation。
3. 测试 full_head trajectory 是否可以被下游 MLP/residual 单点拦截。
4. 输出 atlas_graph_v1，把 propagation node 和 generation boundary 接起来。
```

## Phase 730: Downstream Propagation Node Cancellation [2026-06-28 18:58]

### 输入内容判断

本阶段分析了两份新材料：

```text
1. Phase 729 是关键分水岭实验：
   likelihood support cluster != generation closure mechanism

2. 差分方法只能看到两个条件之间的差异，
   不能直接看到神经网络内部完整功能脉络。
```

总体判断：两份材料的方向基本正确。Phase 729 的真实增量不是证明 channel cluster 已经闭合生成，而是把机制分成了四层：

```text
likelihood support
→ residual propagation
→ downstream MLP / residual mediation
→ readout / generation closure
```

第二份材料也正确指出：继续只做 pairwise difference 容易遗漏共享骨架，因此后续必须从 difference atlas 过渡到 full-path functional atlas。但在进入全路径图谱前，先做一次 downstream cancellation 是必要的，因为它可以判断 Phase 729 找到的传播节点到底是不是瓶颈。

### 生成脚本

```text
tests/gpt5/phase730_downstream_node_cancellation.py
tests/gpt5/run_phase730_downstream_node_cancellation_full.sh
tests/gpt5/build_phase730_atlas_graph.py
```

### 执行命令

```bash
PHASE730_MAX_CASES=1 bash tests/gpt5/run_phase730_downstream_node_cancellation_full.sh
unset PHASE730_MAX_CASES
bash tests/gpt5/run_phase730_downstream_node_cancellation_full.sh
python tests/gpt5/build_phase730_atlas_graph.py
python -m py_compile tests/gpt5/build_phase730_atlas_graph.py tests/gpt5/phase730_downstream_node_cancellation.py
```

三个模型按 qwen3、GLM4、DS7B 顺序运行，并使用 `--hard-exit-after-model`，避免 GPU 显存残留。

### 测试原理

Phase 730 不是简单 ablation，而是 downstream cancellation。

具体做法：

```text
1. 先运行 baseline。
2. 再运行 upstream intervention：
   category_cluster 或 category_full_head。
3. 在上游扰动已经发生后，
   把下游某个传播节点的输出替换回 baseline 对应输出。
4. 如果目标答案 likelihood / generation 恢复，
   说明该下游节点承载了上游扰动的因果路径。
```

本阶段测试的下游节点来自 Phase 729 的 top propagation sites：

```text
qwen3:
  hidden_33
  L28_mlp_out

GLM4:
  hidden_40
  L28_mlp_out / L25_mlp_out

DS7B:
  hidden_27
  L24_mlp_out
```

核心指标：

```text
mean_logprob_delta
recovery_fraction_vs_upstream
hit_rate
changed_rate_vs_baseline
hit_drop_rate_vs_baseline
```

### 客观结果

结果目录：

```text
results/glm5_phase730_downstream_node_cancellation/
```

图谱文件：

```text
results/glm5_phase730_downstream_node_cancellation/phase730_atlas_graph.json
schema_version = atlas_graph_v1
node_count = 45
edge_count = 62
```

关键汇总：

```text
qwen3:
  category_cluster upstream_only:
    mean_delta = +0.0184
    generation changed = 0.000

  category_cluster cancel_top_layer_out:
    mean_delta = 0.0000
    recovery = 1.000

  category_cluster cancel_top_mlp_out:
    mean_delta = +0.0032
    recovery = 0.829

  category_full_head upstream_only:
    mean_delta = -0.0159
    generation changed = 0.000

  category_full_head cancel_top_mlp_out:
    mean_delta = -0.0209
    recovery = -0.318
```

```text
GLM4:
  category_cluster upstream_only:
    mean_delta = +0.0006
    generation changed = 0.000

  category_cluster cancel_top_layer_out:
    mean_delta = 0.0000
    recovery = 1.000

  category_cluster cancel_top_mlp_out:
    mean_delta = +0.0047
    recovery = -6.311

  category_full_head upstream_only:
    mean_delta = +0.0016
    generation changed = 0.000

  category_full_head cancel_top_mlp_out:
    mean_delta = +0.0050
    recovery = -2.051
```

```text
DS7B:
  category_cluster upstream_only:
    mean_delta = -0.2613
    generation changed = 0.000
    hit_drop = 0.000

  category_cluster cancel_top_layer_out:
    mean_delta = 0.0000
    recovery = 1.000

  category_cluster cancel_top_mlp_out:
    mean_delta = -0.3281
    recovery = -0.256

  category_full_head upstream_only:
    mean_delta = -1.0575
    generation changed = 0.409
    hit_drop = 0.091

  category_full_head cancel_top_layer_out:
    mean_delta = 0.0000
    recovery = 1.000
    generation changed = 0.000

  category_full_head cancel_top_mlp_out:
    mean_delta = -0.9339
    recovery = 0.117
    generation changed = 0.273
    hit_drop = 0.091
```

### 阶段结论

最稳健的客观结论：

```text
1. cancel_top_layer_out 在三个模型中都能把 likelihood delta 恢复到 0。
2. 这说明 late residual state 确实承载上游扰动后的状态。
3. 但它不是机制起点，因为它是很靠后的 residual overwrite。
4. top MLP cancellation 没有形成完整恢复。
5. DS7B full_head 的 L24_mlp_out 只恢复约 11.7% likelihood drop，
   并把 generation changed_rate 从 0.409 降到 0.273，
   但 hit_drop 仍为 0.091。
```

因此 Phase 730 把 Phase 729 的传播结论进一步收紧为：

```text
late residual node:
  carries downstream state
  but too late / too coarse

top MLP node:
  partial mediator in DS7B full-head path
  not full bottleneck

category cluster:
  supports likelihood trajectory
  but still does not close generation

full head:
  generation-sensitive perturbation source
  but mechanism still distributed
```

### 严格问题和硬伤

```text
1. cancel_top_layer_out 的 100% recovery 不能解释为发现核心机制。
   它只是证明 late residual state 是扰动承载点。

2. qwen3 / GLM4 的 upstream delta 很小，
   recovery_fraction 存在小分母不稳定问题。

3. DS7B 结果最有信息量，但 DS7B 也是小模型，
   内部结构可能偏离更大模型的真实语言机制。

4. top MLP 只解释一部分 full_head effect，
   说明 generation closure 不是单个 MLP 节点能完全解释的。

5. 本阶段仍然是差分 + cancellation，
   还不是完整自然路径图谱。
```

### 理论进展

当前理论更应表述为：

```text
语言生成不是单个语义向量直接被读出。
它更像条件化 residual state 在多层中被逐步改写，
其中注意力头提供源词元/路线选择，
MLP 对状态进行非线性重写，
late residual state 承载最终读出条件，
最后由 readout competition 决定自然生成。
```

简化链条：

```text
source token / concept cue
→ head route selection
→ channel / subspace likelihood support
→ full-head residual trajectory
→ MLP partial mediation
→ late residual state
→ readout competition
→ generation
```

这也支持第二份材料的关键提醒：

```text
差分方法只能看到某一条边的变化，
不能看到共享的完整功能骨架。
```

### 下一步

Phase 730 的阶段目标已经完成。下一步属于同一总研究路线，但已经不是简单传播节点取消，而是图谱范式升级：

```text
Phase 731: Full-Path Functional Atlas v0

目标：
1. 对 apple / fruit / color / attribute 任务记录自然完整路径。
2. 同时保存 absolute trajectory 与 differential trajectory。
3. 把 head、channel、MLP、residual、readout 统一进 atlas_graph_v1。
4. 区分 shared skeleton、concept-specific branch、format-specific branch。
5. 用少量关键 causal validation 验证图谱边，不再盲目扩大 patch 搜索。
```

Phase 731 的核心不是继续寻找单点开关，而是开始构建：

```text
full-path functional atlas
```

即：

```text
完整功能图谱 > 单点 patch 成功率
```

## Phase 731: Full-Path Functional Atlas v0 [2026-06-28 19:21]

### 输入内容判断

本阶段分析的 Phase 730 复盘材料总体正确，尤其是以下三点：

```text
1. Phase 730 的 late residual cancellation 是承载点证明，不是机制源点证明。
2. top MLP cancellation 只说明部分中介，不说明完整瓶颈。
3. 差分 + cancellation 已经暴露局限，下一步必须转向 full-path functional atlas。
```

材料中提出的核心修正是正确的：

```text
传播大，不等于机制瓶颈；
能恢复，不等于机制起点；
差分边，不等于完整功能骨架。
```

因此本阶段不继续扩大单点 patch，而是启动 full-path functional atlas v0。

### 生成脚本

```text
tests/gpt5/phase731_full_path_functional_atlas_v0.py
tests/gpt5/run_phase731_full_path_functional_atlas_v0_full.sh
```

### 执行命令

```bash
python -m py_compile tests/gpt5/phase731_full_path_functional_atlas_v0.py
python tests/gpt5/phase731_full_path_functional_atlas_v0.py --dry-run
bash tests/gpt5/run_phase731_full_path_functional_atlas_v0_full.sh
python tests/gpt5/phase731_full_path_functional_atlas_v0.py --summarize-only
```

三个模型按 qwen3、GLM4、DS7B 顺序运行，并使用 `--hard-exit-after-model`。

### 测试目标

Phase 731 的目标不是证明某个节点闭合生成，而是建立第一版全路径功能图谱：

```text
absolute trajectory（绝对轨迹）
+ descriptive factor effect（描述性因素效应）
+ candidate-head attention route（候选注意力头路线）
+ natural generation summary（自然生成摘要）
```

案例数量：

```text
total = 66
category = 22
color = 22
taste = 22

explicit_profile = 36
conflict_profile = 12
commonsense = 18
```

这比单一 category 测试更接近 apple-fruit-attribute 微世界。

### 测试原理

本阶段同时记录四类数据：

```text
1. natural generation:
   贪婪生成文本、hit_rate、目标答案 phrase likelihood。

2. absolute trajectory:
   hidden state、attention output、MLP output、residual trajectory 的范数和相对关系。

3. factor mean-difference:
   对 object_group、relation、prompt_type 做基础均值差：

   effect_norm = || mean(h | factor=level) - mean(h) ||

4. candidate-head attention:
   只记录候选 head 对 object_name、relation_name、target_value、record_line 等 token group 的注意力质量。
```

注意：这里没有使用复杂统计模型，只做基础均值差，目的是先拼出客观结构。

### 输出文件

结果目录：

```text
results/glm5_phase731_full_path_functional_atlas_v0/
```

核心文件：

```text
phase731_qwen3_case_summary.jsonl
phase731_qwen3_trajectory_rows.jsonl
phase731_qwen3_attention_rows.jsonl
phase731_qwen3_factor_effect_rows.jsonl

phase731_glm4_case_summary.jsonl
phase731_glm4_trajectory_rows.jsonl
phase731_glm4_attention_rows.jsonl
phase731_glm4_factor_effect_rows.jsonl

phase731_deepseek7b_case_summary.jsonl
phase731_deepseek7b_trajectory_rows.jsonl
phase731_deepseek7b_attention_rows.jsonl
phase731_deepseek7b_factor_effect_rows.jsonl

phase731_cross_model_summary.json
phase731_cross_model_summary.md
phase731_atlas_graph.json
```

图谱输出：

```text
schema_version = atlas_graph_v1
node_count = 70
edge_count = 78
```

### 客观结果

跨模型自然生成结果：

```text
qwen3:
  category hit = 0.955
  color hit = 1.000
  taste hit = 0.909
  top factor effect = prompt_type/commonsense@hidden_35
  effect_norm = 190.465

GLM4:
  category hit = 0.955
  color hit = 1.000
  taste hit = 1.000
  top factor effect = prompt_type/commonsense@hidden_39
  effect_norm = 89.866

DS7B:
  category hit = 0.500
  color hit = 0.682
  taste hit = 0.727
  top factor effect = prompt_type/commonsense@L27_mlp_out
  effect_norm = 394.751
```

按 prompt_type 分组：

```text
qwen3:
  commonsense hit = 0.944
  conflict_profile hit = 1.000
  explicit_profile hit = 0.944

GLM4:
  commonsense hit = 0.944
  conflict_profile hit = 1.000
  explicit_profile hit = 1.000

DS7B:
  commonsense hit = 0.222
  conflict_profile hit = 0.750
  explicit_profile hit = 0.806
```

候选 head 注意力摘要中的关键现象：

```text
DS7B:
  L20H17:
    relation mass = 0.574
    target_value mass = 0.024
    object mass = 0.015

  L23H0:
    relation mass = 0.411
    object mass = 0.356
    target_value mass = 0.119

  L24H21:
    record_line mass = 0.839
    target_value mass = 0.157

GLM4:
  L29H18 / L29H28 / L24H19:
    target_value mass 高，record_line mass 接近 0.95 到 0.99。

qwen3:
  L24H29:
    target_value mass = 0.504
    record_line mass = 0.989。
```

### 阶段结论

Phase 731 的最重要客观发现不是某个 fruit head，而是：

```text
prompt_type / knowledge source 的全路径效应非常强。
```

三模型 top factor effect 都集中在 commonsense 条件：

```text
qwen3: commonsense@hidden_35
GLM4: commonsense@hidden_39
DS7B: commonsense@L27_mlp_out
```

这说明在当前微世界中，模型首先强烈区分：

```text
explicit facts / conflict facts / commonsense
```

然后才在这个路径上处理：

```text
category / color / taste
```

因此此前只围绕 category channel cluster 做 patch，确实容易漏掉共享骨架。

更准确的全路径图像是：

```text
prompt_type skeleton
→ relation route
→ object / value binding
→ candidate head addressing
→ MLP / residual trajectory
→ readout
→ generation
```

### 严格问题和硬伤

```text
1. Phase 731 是描述性 full-path atlas v0，不是因果闭合。
2. factor effect_norm 是均值向量差，不能直接解释为因果强度。
3. candidate-head attention 仍是 observational attention，不等价于必要性。
4. DS7B 的 commonsense 表现很弱，可能放大了 prompt_type effect。
5. 本阶段没有验证具体因果边，例如 L20H17 -> L24_mlp_out。
6. 小模型结构可能偏差很大，不能外推到大模型或人脑。
```

### 理论进展

Phase 731 后，理论应进一步收紧：

```text
语言编码机制不是先有单个概念向量再读出，
而是先形成任务/知识来源/格式协议骨架，
再在骨架中绑定对象、关系和值，
最后通过 head route、MLP rewrite、residual carrier 和 readout competition 生成答案。
```

当前最接近真实机制的公式应是：

```text
h_l(o,r,f,k)
= S_l
+ K_l(k)
+ P_l(f)
+ R_l(r)
+ O_l(o)
+ V_l(v)
+ B_l(o,r,v)
+ M_l
+ I_l
+ epsilon_l
```

其中本阶段最强客观项是：

```text
K_l(k) / P_l(f)
```

也就是 knowledge source / prompt protocol 对后段 residual / MLP 轨迹的影响。

### 下一步

Phase 731 已完成 full-path atlas v0。下一阶段仍属于同一大任务，但应从“描述性图谱”推进到“关键边因果验证”：

```text
Phase 732: Full-Path Atlas Causal Edge Validation

核心边：
1. prompt_type skeleton -> late hidden / MLP
2. relation route -> DS7B L20H17 / L23H0
3. DS7B L20H17 full head -> L24_mlp_out
4. L24_mlp_out -> late residual / readout

目标：
把 Phase 731 的描述性 factor edge，
转化为少量可验证 causal edge。
```

阶段性大任务仍然是：

```text
从局部 patch 研究，升级为全路径功能图谱工程。
```
