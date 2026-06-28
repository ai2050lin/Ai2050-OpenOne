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
