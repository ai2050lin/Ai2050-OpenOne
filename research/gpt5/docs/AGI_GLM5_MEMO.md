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

## Phase 732: Full-Path Atlas Causal Edge Validation [2026-06-28 20:09]

### 输入内容判断

本阶段分析的 Phase 731 复盘材料总体正确。它指出 Phase 731 的范式切换是必要的，但必须严格降级理解：

```text
Phase 731 = descriptive atlas v0
not causal atlas
not neuron-level global atlas
```

材料中最重要的判断是正确的：

```text
prompt_type / knowledge source skeleton
是当前 apple-fruit-attribute 微世界中最强的全路径因素。
```

因此 Phase 732 不继续扩大单点 channel / head 搜索，而是把 Phase 731 的描述性边转成少量因果边验证。

### 生成脚本

```text
tests/gpt5/phase732_full_path_atlas_causal_edge_validation.py
tests/gpt5/run_phase732_full_path_atlas_causal_edge_validation_full.sh
```

### 执行命令

```bash
python -m py_compile tests/gpt5/phase732_full_path_atlas_causal_edge_validation.py
python tests/gpt5/phase732_full_path_atlas_causal_edge_validation.py --dry-run
bash tests/gpt5/run_phase732_full_path_atlas_causal_edge_validation_full.sh
```

三个模型按 qwen3、GLM4、DS7B 顺序运行，并使用 `--hard-exit-after-model`。

### 测试原理

Phase 732 验证三类边：

```text
1. prompt_type skeleton -> late hidden / MLP
   用 explicit_profile 与 commonsense 配对 prompt，
   在关键 late site 做 donor state replacement。

2. candidate head -> readout
   对 Phase 731 的候选 head 做整头消融，
   测 phrase likelihood 和 natural generation 是否变化。

3. DS7B L20H17 -> L24_mlp_out
   消融 L20H17，
   测 L24_mlp_out、final logits 和 category likelihood 的变化。
```

prompt transfer 公式：

```text
h_v^{commonsense}
← h_v^{explicit}
```

若 target likelihood 提升并且 generation 改变，说明 prompt_type skeleton 不是纯观察相关，而是能沿 late hidden / MLP site 改变输出路径。

### 输出文件

结果目录：

```text
results/glm5_phase732_full_path_atlas_causal_edge_validation/
```

核心文件：

```text
phase732_qwen3_prompt_transfer_rows.jsonl
phase732_qwen3_head_ablation_rows.jsonl
phase732_qwen3_summary.json

phase732_glm4_prompt_transfer_rows.jsonl
phase732_glm4_head_ablation_rows.jsonl
phase732_glm4_summary.json

phase732_deepseek7b_prompt_transfer_rows.jsonl
phase732_deepseek7b_head_ablation_rows.jsonl
phase732_deepseek7b_edge_mediation_rows.jsonl
phase732_deepseek7b_summary.json

phase732_cross_model_summary.json
phase732_cross_model_summary.md
phase732_atlas_graph.json
```

图谱输出：

```text
schema_version = atlas_graph_v1
node_count = 55
edge_count = 52
```

### 客观结果

#### 1. prompt_type skeleton 因果替换

qwen3：

```text
commonsense <- explicit @ hidden_35:
  mean_delta = +3.259
  changed_rate = 0.056
  hit_rate: 0.944 -> 1.000
  hit_gain = 0.056

explicit <- commonsense @ hidden_35:
  mean_delta = -3.698
  changed_rate = 0.056
  hit_rate: 1.000 -> 0.944
  hit_loss = 0.056
```

GLM4：

```text
commonsense <- explicit @ hidden_39:
  mean_delta = +2.374
  changed_rate = 0.167
  hit_rate: 0.944 -> 1.000
  hit_gain = 0.056

explicit <- commonsense @ hidden_39:
  mean_delta = -2.387
  changed_rate = 0.167
  hit_rate: 1.000 -> 0.944
  hit_loss = 0.056
```

DS7B：

```text
commonsense <- explicit @ L27_mlp_out:
  mean_delta = +1.472
  changed_rate = 0.667
  hit_rate: 0.222 -> 0.389
  hit_gain = 0.167

commonsense <- explicit @ hidden_27:
  mean_delta = +4.129
  changed_rate = 0.889
  hit_rate: 0.222 -> 0.444
  hit_gain = 0.444
  hit_loss = 0.222

explicit <- commonsense @ hidden_27:
  mean_delta = -4.559
  changed_rate = 0.889
  hit_rate: 0.667 -> 0.222
  hit_loss = 0.556
```

结论：

```text
prompt_type / knowledge-source skeleton 是因果有效边，
不是 Phase 731 中的纯描述性均值差。
```

但是 DS7B hidden_27 替换同时有 hit_gain 和 hit_loss，说明该 site 很强但很粗，替换会造成分布偏移。

#### 2. 候选 head 必要性

qwen3 最强 head edge：

```text
L24H29 / taste / explicit_profile:
  mean_logprob_delta = -0.099
  changed_rate = 0.000
  hit_drop = 0.000
```

GLM4 最强 head edge：

```text
L24H19 / taste / commonsense:
  mean_logprob_delta = -0.027
  changed_rate = 0.000
  hit_drop = 0.000
```

DS7B 最强 head edges：

```text
L20H17 / category / conflict_profile:
  mean_logprob_delta = -1.957
  rank_delta = +5.000
  changed_rate = 0.250

L20H17 / category / explicit_profile:
  mean_logprob_delta = -0.929
  rank_delta = +2.250
  changed_rate = 0.333
  hit_drop = 0.167

L20H17 / category / commonsense:
  mean_logprob_delta = -0.715
  rank_delta = +8.667
  changed_rate = 0.667

L23H0 / taste / commonsense:
  mean_logprob_delta = -0.911
  rank_delta = +220.500
```

结论：

```text
qwen3 / GLM4 candidate heads 的因果必要性较弱；
DS7B L20H17 是 category route 的强因果 head；
DS7B L23H0 在 taste commonsense 中也有强 rank effect。
```

#### 3. DS7B L20H17 -> L24_mlp_out

```text
n = 22
mean_source_delta_norm = 26.636
mean_target_delta_norm = 32.777
mean_source_target_delta_cos = -0.003
mean_target_logprob_delta = -1.058
mean_target_rank_delta = +4.500
```

结论：

```text
消融 L20H17 会显著改变 L24_mlp_out，
并造成 category target likelihood 下降。
```

但是 source-target delta cosine 接近 0，说明这不是简单同方向线性传递，更像经过 MLP / residual 重写后的非线性变换。

### 阶段结论

Phase 732 至少验证了两类关键 causal edge：

```text
1. prompt_type skeleton -> late hidden / MLP site
2. DS7B L20H17 -> category readout / L24_mlp_out
```

这把 Phase 731 从 descriptive atlas v0 推进到 causal edge atlas v0。

当前图谱结构应更新为：

```text
prompt_type / knowledge-source skeleton
  has causal control over late hidden / MLP state

DS7B L20H17
  is a causal category-route head

L24_mlp_out
  receives strong transformed perturbation from L20H17 ablation

generation closure
  still not fully closed, but causal edges are now visible
```

### 严格问题和硬伤

```text
1. site replacement 是强干预，可能造成分布偏移。
2. hidden_27 / hidden_35 / hidden_39 很靠后，仍然偏 carrier，不是源点。
3. prompt transfer 提升 likelihood，不等于完整生成修复。
4. DS7B hidden_27 替换同时有 hit_gain 和 hit_loss，说明过粗。
5. qwen3 / GLM4 head ablation 效应弱，说明它们可能更冗余或候选 head 不是必要点。
6. L20H17 -> L24_mlp_out 的 cosine 接近 0，说明不能用线性方向解释。
7. 当前仍是小模型结果，不能外推到大模型或人脑。
```

### 理论进展

Phase 732 后，理论更明确：

```text
1. knowledge-source / prompt-type skeleton 是真实因果控制层。
2. relation/category route 可以由特定 head 强因果支撑，DS7B 中是 L20H17。
3. MLP/residual 不是简单传递，而是重写扰动。
4. 语言编码机制必须用 causal edge graph，而不是只用 node importance。
```

当前最简链条：

```text
knowledge-source skeleton
→ late residual / MLP state
→ relation route head
→ MLP nonlinear rewrite
→ readout competition
→ generation
```

### 下一步

Phase 732 已经完成 causal edge validation v0。下一阶段仍属于同一大任务，但应继续收紧到：

```text
Phase 733: Prompt-Type Skeleton Source Localization

目标：
1. 不再只替换 late hidden。
2. 向前追踪 prompt_type skeleton 的起点。
3. 找到 commonsense / explicit / conflict 分离最早出现的层和组件。
4. 区分 attention route、MLP rewrite、residual carrier 的先后顺序。
5. 用更早层 site replacement 验证是否能保留 Phase 732 的因果效应。
```

核心问题：

```text
prompt_type skeleton 是在哪里被写入的？
它是 attention 写入，MLP 写入，还是 residual 中逐层积累？
```

## Phase 733: Prompt-Type Skeleton Source Localization [2026-06-28 20:36]

### 任务来源和判断

用户给出的 Phase 732 分析基本正确：Phase 732 不是普通 patch 堆叠，而是把 Phase 731 的 descriptive atlas（描述性图谱）推进到 causal edge atlas v0（因果边图谱0版）。但必须收紧解释：late hidden replacement（晚期隐藏态替换）证明的是后段 carrier / controller（载体/控制器）有因果作用，不等于证明 prompt-type skeleton（提示类型骨架）的源点已经找到；DS7B L20H17 -> L24_mlp_out 的 cosine（余弦相似度）接近 0，也说明下游不是线性传递，而是非线性重写。

因此本阶段继续完成同一大任务：向前定位 prompt-type / knowledge-source skeleton（提示类型/知识来源骨架）最早明显形成的位置，并验证较早 site（位置）是否仍有因果迁移效应。

### 脚本和命令

生成脚本：

```text
tests/gpt5/phase733_prompt_type_skeleton_source_localization.py
tests/gpt5/run_phase733_prompt_type_skeleton_source_localization_round.sh
```

输出目录：

```text
results/glm5_phase733_prompt_type_skeleton_source_localization/
```

关键命令：

```bash
python -m py_compile tests/gpt5/phase733_prompt_type_skeleton_source_localization.py
python tests/gpt5/phase733_prompt_type_skeleton_source_localization.py --dry-run --round-name smoke --max-scan-cases 9 --max-pairs 2

PHASE733_ROUND_NAME=smoke PHASE733_MAX_SCAN_CASES=9 PHASE733_MAX_PAIRS=2 PHASE733_MAX_CANDIDATE_SITES=4 PHASE733_LOG_EVERY=1 bash tests/gpt5/run_phase733_prompt_type_skeleton_source_localization_round.sh
PHASE733_ROUND_NAME=main PHASE733_MAX_SCAN_CASES=36 PHASE733_MAX_PAIRS=12 PHASE733_MAX_CANDIDATE_SITES=10 PHASE733_LOG_EVERY=4 bash tests/gpt5/run_phase733_prompt_type_skeleton_source_localization_round.sh
PHASE733_ROUND_NAME=confirm PHASE733_MAX_SCAN_CASES=66 PHASE733_MAX_PAIRS=18 PHASE733_MAX_CANDIDATE_SITES=10 PHASE733_LOG_EVERY=6 bash tests/gpt5/run_phase733_prompt_type_skeleton_source_localization_round.sh
```

加载策略：bf16（bfloat16，脑浮点16位），quantization=off（量化关闭），先尝试 flash_attention_2（闪存注意力2），本地环境缺少 FlashAttention2 包，因此三模型均自动退到 sdpa（缩放点积注意力）。每个模型均使用 `--hard-exit-after-model`，按 qwen3 -> GLM4 -> DS7B 顺序执行，避免显存残留。

### 测试原理

本阶段不再只看某个 late hidden（晚期隐藏态）能不能修复输出，而是分两步：

```text
第一步：formation scan（形成扫描）
对 explicit_profile / conflict_profile / commonsense 三类 prompt（提示）分别记录每层 layer_input、attn_out、mlp_out、layer_out 的最后词元向量。
计算 commonsense_vs_explicit_profile 的平均向量距离，寻找 prompt-type skeleton 最早明显分离的位置。

第二步：site replacement validation（位置替换验证）
把 explicit prompt 在候选 site 的向量替换到 commonsense prompt 的对应 site，观察正确答案 logprob、生成文本、hit rate 是否变化。
反向也做 explicit<-commonsense，判断方向性。
```

候选 site 选择规则：

```text
对 layer_input / attn_out / mlp_out / layer_out 四类 site：
1. 找到 commonsense_vs_explicit_profile effect_norm 最大点。
2. 找到达到最大 effect_norm 35% 的最早层。
3. 同时加入 Phase 732 的 late hidden 参考点。
```

### 客观结果

三轮均完成：smoke、main、confirm。

confirm 轮关键结果：

```text
qwen3:
layer_input earliest = L28_layer_input, max = L35_layer_input
attn_out earliest = L26_attn_out, max = L34_attn_out
mlp_out earliest = L28_mlp_out, max = L35_mlp_out
layer_out earliest = hidden_28, max = hidden_36
最强 commonsense<-explicit transfer：hidden_36 / L35_layer_input
logprob delta 约 +3.11，hit 0.944 -> 1.000，changed_rate 0.056

GLM4:
layer_input earliest = L31_layer_input, max = L39_layer_input
attn_out earliest = L23_attn_out, max = L35_attn_out
mlp_out earliest = L38_mlp_out, max = L39_mlp_out
layer_out earliest = hidden_39, max = hidden_40
最强 commonsense<-explicit transfer：hidden_40 / L39_layer_input
logprob delta 约 +2.34，hit 0.944 -> 1.000，changed_rate 0.167

DS7B:
layer_input earliest = L22_layer_input, max = L27_layer_input
attn_out earliest = L21_attn_out, max = L27_attn_out
mlp_out earliest = L25_mlp_out, max = L27_mlp_out
layer_out earliest = hidden_24, max = hidden_28
最强 commonsense<-explicit transfer：hidden_28
logprob delta 约 +5.90，hit 0.222 -> 0.722，hit_gain 0.611，hit_loss 0.111
L27_layer_input 也有效：logprob delta 约 +4.92，hit 0.222 -> 0.444
```

main 与 confirm 的层位基本一致：

```text
qwen3 稳定在 L26-L36 段。
GLM4 稳定在 L23-L40 段，但真正强的 residual / MLP 位点靠最后两层。
DS7B 稳定在 L21-L28 段，且生成 hit 改善最明显。
```

### 当前进展

Phase 733 把 Phase 732 的结论向前推进了一步：

```text
Phase 732:
late hidden / late MLP 有 prompt-type causal effect。

Phase 733:
prompt-type skeleton 在不同模型中有稳定的中后层形成轨迹；
早期阈值点可定位，但最强因果迁移仍在晚层 residual / layer_input；
DS7B 中该骨架对自然生成有明显修复效应，qwen3 / GLM4 中主要表现为 likelihood 支撑。
```

更新后的机制拼图：

```text
prompt-type / knowledge-source skeleton
  begins to separate in mid-late layers
  becomes strongest in late residual / layer_input
  can causally bias correct value likelihood
  in DS7B can partially repair generation

attention route
  can show earlier separation
  but not yet proven to be primary writer

MLP route
  tends to peak late
  more像 nonlinear rewrite（非线性重写）而不是线性转运
```

### 严格问题和硬伤

```text
1. earliest_35pct 是工程阈值，不是自然边界。
2. site replacement 仍然是强干预，可能产生分布偏移。
3. qwen3 / GLM4 的 hit 已接近天花板，因此 logprob 提升不等价于生成能力修复。
4. DS7B 的修复最强，但同时仍有 hit_loss，说明 late carrier 过粗。
5. formation scan 使用均值距离，能定位分离轨迹，但不能证明 writer neuron（写入神经元）。
6. 当前仍是小模型，内部结构可能偏离大模型。
7. FlashAttention2 未安装，实际使用 sdpa；这不影响结论方向，但需记录运行环境差异。
```

### 理论进展

语言编码机制更接近如下结构：

```text
语义不是单个固定向量；
生成不是单点读出；
模型内部存在条件化相对状态轨迹；
prompt-type skeleton 是轨迹的全局控制骨架之一；
relation / value / format 等局部机制在该骨架上被差分复用；
最终输出来自 late residual field 中多个因果边的闭合。
```

当前最简链条更新为：

```text
prompt condition
→ mid-late prompt-type skeleton formation
→ late residual / layer_input carrier
→ attention / MLP nonlinear rewrite
→ value / route competition
→ readout
→ generation closure
```

### 下一步

Phase 733 和 Phase 732 属于同一阶段性目标：从局部 patch 转向 causal atlas（因果图谱）。当前已完成 prompt-type skeleton source localization v0，下一步仍应继续自动推进，但要从“找到 site”转向“找到 writer mechanism”。

建议 Phase 734：

```text
Phase 734: Prompt-Type Skeleton Writer Decomposition

目标：
1. 以 qwen3 L28-L35、GLM4 L31-L40、DS7B L22-L28 为窗口。
2. 对 candidate attention heads / MLP channels 做 source-restricted ablation。
3. 不再只问 site replacement 是否有效，而是问：
   哪些组件负责写入 prompt-type skeleton？
   哪些组件只是 carrier？
   哪些组件负责 nonlinear rewrite？
4. 对 DS7B 优先，因为它有最强生成 hit_gain。
5. 输出 writer-edge atlas v0。
```

## Phase 734: Prompt-Type Skeleton Writer Decomposition [2026-06-28 21:25]

### 任务来源和判断

用户给出的 Phase 733 复盘总体正确。Phase 733 的性质不是完整机制闭合，而是把 prompt-type / knowledge-source skeleton（提示类型 / 知识来源骨架）从 late carrier（后期承载器）推进到 mid-late formation trajectory（中后层形成轨迹）。它证明了：

```text
prompt-type skeleton 不是最后一层突然出现；
它在中后层逐步分离，并在晚层 residual / layer_input 中达到最强；
DS7B 中该骨架能明显修复 commonsense path failure。
```

但复盘中也指出一个关键硬伤：Phase 733 仍然没有回答 writer mechanism（写入机制）。因此 Phase 734 继续同一阶段性目标，从 site localization（位置定位）推进到 writer decomposition v0（写入器分解0版）。

### 生成脚本

```text
tests/gpt5/phase734_prompt_type_skeleton_writer_decomposition.py
tests/gpt5/run_phase734_prompt_type_skeleton_writer_decomposition_round.sh
```

输出目录：

```text
results/glm5_phase734_prompt_type_skeleton_writer_decomposition/
```

核心输出文件：

```text
phase734_{model}_attention_writer_rows.jsonl
phase734_{model}_mlp_writer_rows.jsonl
phase734_{model}_summary.json
phase734_cross_model_summary.json
phase734_cross_model_summary.md
phase734_atlas_graph.json
```

### 执行命令

```bash
python -m py_compile tests/gpt5/phase734_prompt_type_skeleton_writer_decomposition.py
python tests/gpt5/phase734_prompt_type_skeleton_writer_decomposition.py --dry-run --round-name smoke --max-pairs 2 --max-layers 3 --max-heads-per-layer 4 --mlp-groups-per-layer 4

PHASE734_ROUND_NAME=smoke PHASE734_MAX_PAIRS=2 PHASE734_MAX_LAYERS=3 PHASE734_MAX_HEADS_PER_LAYER=4 PHASE734_MLP_GROUPS_PER_LAYER=4 PHASE734_LOG_EVERY=1 bash tests/gpt5/run_phase734_prompt_type_skeleton_writer_decomposition_round.sh

PHASE734_ROUND_NAME=main PHASE734_MAX_PAIRS=8 PHASE734_MAX_LAYERS=4 PHASE734_MAX_HEADS_PER_LAYER=8 PHASE734_MLP_GROUPS_PER_LAYER=8 PHASE734_LOG_EVERY=2 bash tests/gpt5/run_phase734_prompt_type_skeleton_writer_decomposition_round.sh

PHASE734_ROUND_NAME=confirm PHASE734_MAX_PAIRS=12 PHASE734_MAX_HEADS_PER_LAYER=10 PHASE734_MLP_GROUPS_PER_LAYER=10 PHASE734_LOG_EVERY=3 bash tests/gpt5/run_phase734_prompt_type_skeleton_writer_decomposition_round.sh
```

加载策略：

```text
bf16
quantization = off
先尝试 flash_attention_2
本地缺少 FlashAttention2 包，三模型实际均回退为 sdpa
每个模型使用 --hard-exit-after-model
执行顺序 qwen3 -> GLM4 -> DS7B
```

### 测试原理

Phase 734 不再问“哪个 late hidden 替换有效”，而是问：

```text
消融某个 attention head 或 MLP output group 后，
explicit path 在 downstream prompt-type carrier 上是否失去 explicit-vs-commonsense 方向？
```

具体流程：

```text
1. 从 Phase 733 confirm summary 自动读取每个模型的 target carrier：
   qwen3: hidden_36
   GLM4: hidden_40
   DS7B: hidden_28

2. 对同一 object-relation-answer 的 explicit_profile / commonsense pair，
   计算 downstream skeleton direction：
   d = h_target(explicit) - h_target(commonsense)

3. 对候选窗口内 attention head 做 zero ablation，
   重新计算 explicit path 的 target carrier shift：
   Δh = h_target(explicit, ablated) - h_target(explicit)

4. 计算 projection loss：
   explicit_skeleton_loss = - <Δh, d / |d|>

5. 同时测 answer first-token logprob delta。
```

判据：

```text
explicit_skeleton_loss > 0：
  消融使 explicit path 远离 explicit-vs-commonsense 方向，
  说明该组件可能参与写入或维持 prompt-type skeleton。

explicit_logprob_delta < 0：
  消融同时伤害目标答案支持。

二者同时成立：
  writer_candidate / contributor_candidate。
```

注意：本阶段扫描的是 attention head 与 MLP output group，不是单神经元，因此仍是 component-level v0。

### 客观结果

三轮均完成：smoke、main、confirm。

confirm 轮关键结果：

#### qwen3

```text
target_site = hidden_36
scan_layers = [26, 27, 28, 34, 35]

top attention:
L35H0:
  explicit_skeleton_loss = 7.007
  explicit_logprob_delta = -0.001
  commonsense_logprob_delta = -0.421
  role = writer_candidate

top MLP:
L28:mlp[256:512]:
  explicit_skeleton_loss = 6.347
  explicit_logprob_delta = -0.0008
  role = writer_candidate

L34:mlp[0:256]:
  explicit_skeleton_loss = 4.775
  explicit_logprob_delta = -0.019
  role = writer_candidate
```

解释：

```text
qwen3 有 attention writer candidate，最稳定的是 late L35H0。
MLP 也有候选，但 logprob 效应较小，更多像弱写入 / 放大混合。
```

#### GLM4

```text
target_site = hidden_40
scan_layers = [23, 31, 35, 38, 39]

top attention:
L39H21:
  explicit_skeleton_loss = 9.445
  explicit_logprob_delta = -0.0075
  commonsense_logprob_delta = -0.152
  role = writer_candidate

L23H17:
  explicit_skeleton_loss = 2.659
  explicit_logprob_delta = -0.015
  role = writer_candidate

top MLP:
L38:mlp[2870:3280]:
  explicit_skeleton_loss = 8.267
  explicit_logprob_delta = -0.0033
  role = writer_candidate
```

解释：

```text
GLM4 的最稳定 attention writer candidate 在 L39H21。
L23 也有早期候选，但效应较弱。
L38 MLP group 明显扰动 downstream carrier，更像 late rewriter / amplifier。
```

#### DS7B

```text
target_site = hidden_28
scan_layers = [21, 22, 23, 25, 27]

top attention:
L22H24:
  explicit_skeleton_loss = 24.410
  explicit_logprob_delta = -0.132
  commonsense_logprob_delta = -0.064
  role = writer_candidate

L21H12:
  explicit_skeleton_loss = 7.332
  explicit_logprob_delta = -0.019
  role = writer_candidate

top MLP:
L27:mlp[2872:3231]:
  explicit_skeleton_loss = 108.435
  explicit_logprob_delta = -0.698
  commonsense_logprob_delta = -1.083
  explicit_rank_delta = +1.25
  commonsense_rank_delta = +103.33
  role = writer_candidate

L22:mlp[718:1077]:
  explicit_skeleton_loss = 22.994
  explicit_logprob_delta = -0.092
  role = writer_candidate
```

解释：

```text
DS7B 是本阶段最强结果。
L22H24 是明确 attention writer / contributor candidate。
L27 MLP output group 对 downstream skeleton 和 answer likelihood 的影响极强，
但由于它靠近 target hidden_28，更保守地说应解释为 late nonlinear rewriter / amplifier，
不是纯早期 writer。
L22 MLP group 更接近真正中层写入候选。
```

### 阶段进展

Phase 734 把 Phase 733 的问题推进了一层：

```text
Phase 733:
知道 prompt-type skeleton 在哪里开始可见、哪里最强。

Phase 734:
开始定位哪些 attention head / MLP output group 会削弱 downstream skeleton，
即找到 writer / contributor candidates。
```

新的图谱边：

```text
attention_head / MLP_group
  -> downstream prompt-type skeleton carrier
  -> answer likelihood / readout support
```

当前最强边：

```text
DS7B L22H24 -> hidden_28 prompt-type skeleton
DS7B L27 MLP[2872:3231] -> hidden_28 prompt-type skeleton / readout
GLM4 L39H21 -> hidden_40 prompt-type skeleton
qwen3 L35H0 -> hidden_36 prompt-type skeleton
```

### 严格问题和硬伤

```text
1. 本阶段是 component-level writer decomposition v0，不是 neuron-level atlas。
2. MLP output group 很粗，一个 group 包含数百维，不是单通道或单神经元。
3. attention head zero ablation 是强干预，可能同时破坏多个功能。
4. explicit_skeleton_loss 说明该组件影响 downstream skeleton，不等于证明它是唯一写入源。
5. DS7B L27 MLP group 太靠近 hidden_28，更像 late rewriter / amplifier。
6. qwen3 / GLM4 的 logprob 伤害较小，因为 baseline hit 接近天花板，不能过度否定其机制作用。
7. 当前没有 source-token restricted attribution，因此还不能说这些 head 具体从哪些 token 写入。
8. 当前仍是小模型结果，层号和头号不能外推。
```

### 理论进展

Phase 734 支持一个更细的分工：

```text
attention:
  有一批 head 会影响 prompt-type skeleton 的 downstream carrier，
  更像 route / writer / contributor。

MLP:
  中层 MLP group 可能参与写入；
  晚层 MLP group 可能参与 nonlinear rewrite / amplification。

residual:
  仍是承载与累积通道。
```

更新后的机制链：

```text
prompt condition
→ attention writer / route contributor
→ MLP writer / nonlinear rewriter
→ residual carrier
→ relation / value route
→ readout competition
→ generation closure
```

### 下一步

Phase 734 仍属于 causal atlas 大阶段，并且已经完成 writer decomposition v0。下一步不能简单扩大 group 扫描，应该从 component-level 下钻到 finer-grained writer validation。

建议 Phase 735：

```text
Phase 735: Source-Restricted Writer Validation

目标：
1. 对 DS7B L22H24、L21H12、L22 MLP group、L27 MLP group 做重点验证。
2. 加入 source token groups：
   instruction tokens
   record line tokens
   question tokens
   object/relation/value tokens
3. 测这些 writer candidates 是否从特定 token group 写入 prompt-type skeleton。
4. 对 top MLP group 做更细通道划分，缩小到几十维甚至单通道候选。
5. 用 holdout objects / relations 验证候选是否跨样本稳定。
```

第一性原理判断：

```text
要破解语言编码机制，不能只知道某个状态有效；
必须建立 writer -> carrier -> rewriter -> readout 的完整因果链。
Phase 734 已经开始从 carrier 追到 writer，
Phase 735 应继续追 writer 的 source token 与细粒度通道。
```

## Phase 735: Source-Restricted Writer Validation and MLP Fine Decomposition [2026-06-28 22:24]

### 命令

```bash
python -m py_compile tests/gpt5/phase735_source_restricted_writer_validation.py
python tests/gpt5/phase735_source_restricted_writer_validation.py --dry-run --round-name smoke --max-pairs 2 --top-attn 1 --top-mlp 1 --mlp-subgroups 2

PHASE735_ROUND_NAME=smoke PHASE735_MAX_PAIRS=2 PHASE735_TOP_ATTN=1 PHASE735_TOP_MLP=1 PHASE735_MLP_SUBGROUPS=2 PHASE735_LOG_EVERY=1 bash tests/gpt5/run_phase735_source_restricted_writer_validation_round.sh

PHASE735_ROUND_NAME=main PHASE735_MAX_PAIRS=12 PHASE735_TOP_ATTN=2 PHASE735_TOP_MLP=2 PHASE735_MLP_SUBGROUPS=4 PHASE735_LOG_EVERY=2 bash tests/gpt5/run_phase735_source_restricted_writer_validation_round.sh

PHASE735_ROUND_NAME=confirm PHASE735_MAX_PAIRS=18 PHASE735_TOP_ATTN=2 PHASE735_TOP_MLP=2 PHASE735_MLP_SUBGROUPS=6 PHASE735_LOG_EVERY=3 bash tests/gpt5/run_phase735_source_restricted_writer_validation_round.sh
```

### 生成脚本和结果

```text
script:
  tests/gpt5/phase735_source_restricted_writer_validation.py
  tests/gpt5/run_phase735_source_restricted_writer_validation_round.sh

results:
  results/glm5_phase735_source_restricted_writer_validation/smoke/
  results/glm5_phase735_source_restricted_writer_validation/main/
  results/glm5_phase735_source_restricted_writer_validation/confirm/

confirm outputs:
  phase735_cross_model_summary.json
  phase735_cross_model_summary.md
  phase735_atlas_graph.json
  phase735_{model}_attention_source_rows.jsonl
  phase735_{model}_mlp_fine_rows.jsonl
  phase735_{model}_summary.json
```

### 测试原理

Phase 735 直接验证 Phase 734 的 writer candidate（写入器候选）是否真的从某些 source token group（源词元组）把 prompt-type skeleton（提示类型骨架）写入 downstream carrier（下游承载器）。

对 attention head（注意力头）使用 source-restricted contribution erasure（源限制贡献擦除）：

```text
C_G(l,h) = sum_{t in G} alpha_{l,h}(a,t) V_{l,h}(t)
```

其中：

```text
G:
  instruction / records_all / target_record_line / records_other
  question / object_tokens / relation_tokens / target_value_tokens
  answer_prefix / all_pre_answer / self_last

a:
  answer prompt last token（答案提示最后词元）
```

然后只擦除某个 head 从某个 G 接收到的 value contribution（值贡献），观察 target_site（目标位置）上的 explicit-vs-commonsense skeleton direction（显式-常识骨架方向）是否损失：

```text
Loss_K(h,G) = - < h_T^{explicit, erase(h,G)} - h_T^{explicit}, d_K >
```

同时记录 answer first-token logprob delta（答案首词元对数概率变化）。

对 MLP group（多层感知机组）使用 fine output-channel ablation（输出通道细分消融），把 Phase 734 的粗组继续切成 2 / 4 / 6 个子组，观察每个子组的 skeleton loss（骨架损失）和 logprob delta（对数概率变化）。

注意：因为 source-restricted attribution（源限制归因）需要 output_attentions（输出注意力），本阶段使用 eager attention（eager 注意力）。模型仍为 bf16（脑浮点16）、无量化、单模型顺序运行，并使用 --hard-exit-after-model 防止显存残留。

### 三轮测试规模

```text
smoke:
  2 pairs, 1 attention candidate, 1 MLP candidate, 2 MLP subgroups

main:
  12 pairs, 2 attention candidates, 2 MLP candidates, 4 MLP subgroups

confirm:
  18 pairs, 2 attention candidates, 2 MLP candidates, 6 MLP subgroups
```

confirm 轮每个模型生成：

```text
attention source rows:
  396 rows/model

MLP fine rows:
  216 rows/model

total:
  1836 detailed rows
```

### 关键结果

confirm 轮摘要：

```text
qwen3:
  target_site = hidden_36
  top source path:
    L35H0 <- self_last
    skeleton_loss = 6.537
    logprob_delta = -0.002
    attention_mass = 0.947
  secondary source path:
    L28H28 <- instruction
    skeleton_loss = 4.369
    logprob_delta ~= 0
    attention_mass = 0.873
  top MLP fine:
    L34:mlp[43:85]
    skeleton_loss = 3.681
    logprob_delta = +0.005

GLM4:
  target_site = hidden_40
  top source path:
    L39H21 <- self_last
    skeleton_loss = 8.997
    logprob_delta = -0.005
    attention_mass = 1.000
  secondary source path:
    L23H17 <- all_pre_answer / instruction
    skeleton_loss = 3.111 / 3.021
    logprob_delta = -0.018 / -0.024
  top MLP fine:
    L38:mlp[3212:3280]
    skeleton_loss = 3.267
    logprob_delta = +0.003

DS7B:
  target_site = hidden_28
  top source path:
    L22H24 <- all_pre_answer
    skeleton_loss = 24.993
    logprob_delta = -0.187
    attention_mass = 0.981
  record-specific source paths:
    L22H24 <- records_all
      skeleton_loss = 20.426
      logprob_delta = -0.289
      attention_mass = 0.725
    L22H24 <- target_record_line
      skeleton_loss = 16.386
      logprob_delta = -0.295
      attention_mass = 0.595
    L22H24 <- target_value_tokens
      skeleton_loss = 12.636
      logprob_delta = -0.212
      attention_mass = 0.354
  secondary path:
    L21H12 <- self_last
    skeleton_loss = 5.913
    logprob_delta = -0.018
  top MLP fine:
    L27:mlp[2872:2932]
      skeleton_loss = 66.368
      logprob_delta = -0.141
    L22:mlp[957:1017]
      skeleton_loss = 7.110
      logprob_delta = -0.014
```

### 客观进展

Phase 735 把 Phase 734 的 component-level writer candidate（组件级写入器候选）推进到 source-restricted writer path（源限制写入路径）：

```text
DS7B:
  record/value source token
  -> L22H24 attention contribution
  -> hidden_28 prompt-type skeleton carrier
  -> answer logprob support

qwen3:
  self_last / instruction source
  -> L35H0 / L28H28
  -> hidden_36 skeleton carrier

GLM4:
  self_last / instruction source
  -> L39H21 / L23H17
  -> hidden_40 skeleton carrier
```

最强客观结果是 DS7B：L22H24 不是普通 head 排名，而是能被分解到 records_all、target_record_line、target_value_tokens 的源限制因果路径。尤其 target_value_tokens 单独擦除后仍有 skeleton_loss = 12.636 且 logprob_delta = -0.212，说明它确实携带了显式记录中的值信息。

MLP 方面，DS7B L27:mlp[2872:2932] 是当前最强 late nonlinear rewriter / amplifier（后期非线性重写器 / 放大器）细分候选，L22:mlp[957:1017] 是更靠近 writer 层的细分候选。

### 严格审视和硬伤

```text
1. qwen3 / GLM4 的强源路径主要集中在 self_last 或 instruction，
   说明它们当前更像 prompt-control skeleton path（提示控制骨架路径），
   不是明确的 record-value path（记录值路径）。

2. source groups 仍然不是完全正交：
   all_pre_answer 包含 records、instruction、question 等所有前文，
   所以它只能作为总源贡献，不应当单独解释为具体语义路径。

3. attention source erasure 用的是 value contribution（值贡献）擦除，
   还没有做 source-restricted replacement（源限制替换）。

4. MLP fine decomposition 仍然是 output-channel group（输出通道组），
   不是 hidden neuron activation（隐藏神经元激活）级别。

5. 当前模型是小模型，
   层号、头号、通道号不能外推到大模型；
   只能外推功能阶段：source -> writer -> carrier -> rewriter -> readout。

6. logprob delta 和 skeleton loss 有时不一致，
   说明 skeleton carrier 与最终 answer readout 之间仍有非线性重写和竞争。
```

### 理论进展

当前最接近语言编码机制的链条更新为：

```text
source token group
-> attention writer / route contributor
-> prompt-type skeleton carrier
-> MLP nonlinear rewrite / amplification
-> residual state accumulation
-> readout competition
-> generation closure
```

更具体地说，语言编码不是单个 semantic vector（语义向量），而是 condition-specific causal path（条件化因果路径）。同一个词嵌入进入模型后，会被 prompt type（提示类型）、source token（源词元）、relation（关系）、value（值）共同条件化，沿不同 head / MLP group 形成不同状态。

### 下一步

Phase 735 和当前任务仍属于同一个阶段性目标：从 component atlas（组件图谱）推进到 source-resolved functional atlas（源分辨功能图谱）。下一步不需要重新确认方向，应继续自动进入：

```text
Phase 736: Source-Restricted Replacement and Generation Closure

目标：
1. 对 DS7B L22H24 做 source-restricted replacement：
   records_all / target_record_line / target_value_tokens
   donor -> recipient 替换。
2. 检查替换是否能恢复 hidden_28 skeleton 和 answer likelihood。
3. 对 L27:mlp[2872:2932] 与 L22:mlp[957:1017] 做联合路径验证：
   L22H24 source path -> L22 MLP fine writer -> L27 MLP rewriter。
4. 加入 generation closure：
   不只看 logprob，还看自然生成是否被拉回目标答案。
5. 若 DS7B 成立，再做 qwen3 / GLM4 轻量对照，
   但不要强行要求三模型同构。
```

第一性原理判断：

```text
破解语言背后编码机制的关键不是找到一个“语义向量”，
而是找到 source-conditioned route（源条件化路径）
如何经过 writer、carrier、rewriter、readout 形成可生成答案。
Phase 735 第一次把 DS7B 的显式记录值路径从 source token 追到了 downstream carrier，
这是比单纯 head 排名更接近真实编码机制的一步。
```

## Phase 736: Source-Restricted Replacement and Generation Closure [2026-06-28 22:54]

### 任务背景

本阶段分析 Phase 735 的判断后继续推进。Phase 735 的核心结论基本正确：它不是普通 head 排名，而是把 DS7B 的 L22H24 分解到了 `records_all`、`target_record_line`、`target_value_tokens` 等 source group（源词元组），形成 source-resolved functional path v0（源分辨功能路径初版）。但 Phase 735 仍有一个硬缺口：只证明 source-restricted erasure（源限制擦除）会破坏 hidden / likelihood，没有证明 donor source contribution（供体源贡献）替换到 recipient prompt（受体提示）后能否恢复目标状态、答案似然，甚至自然生成。

因此 Phase 736 的目标是：

```text
donor source contribution
-> recipient source contribution replacement
-> target hidden restore
-> donor answer likelihood shift
-> greedy generation closure
```

### 生成脚本

新增正式脚本：

```text
tests/gpt5/phase736_source_replacement_generation_closure.py
```

新增跨模型运行脚本：

```text
tests/gpt5/run_phase736_source_replacement_generation_closure_round.sh
```

输出目录：

```text
results/glm5_phase736_source_replacement_generation_closure/
```

确认轮主要结果文件：

```text
results/glm5_phase736_source_replacement_generation_closure/confirm/phase736_cross_model_summary.md
results/glm5_phase736_source_replacement_generation_closure/confirm/phase736_cross_model_summary.json
results/glm5_phase736_source_replacement_generation_closure/confirm/phase736_atlas_graph.json
results/glm5_phase736_source_replacement_generation_closure/confirm/phase736_qwen3_replacement_rows.jsonl
results/glm5_phase736_source_replacement_generation_closure/confirm/phase736_glm4_replacement_rows.jsonl
results/glm5_phase736_source_replacement_generation_closure/confirm/phase736_deepseek7b_replacement_rows.jsonl
```

### 执行命令

静态检查：

```bash
python -m py_compile tests/gpt5/phase736_source_replacement_generation_closure.py
```

干运行：

```bash
PHASE736_DRY_RUN=1 PHASE736_ROUND_NAME=smoke PHASE736_MAX_PAIRS=2 PHASE736_TOP_PATHS=2 python tests/gpt5/phase736_source_replacement_generation_closure.py --models qwen3 glm4 deepseek7b --hard-exit-after-model
```

冒烟测试：

```bash
PHASE736_ROUND_NAME=smoke PHASE736_MAX_PAIRS=2 PHASE736_TOP_PATHS=2 PHASE736_MAX_NEW_TOKENS=2 PHASE736_LOG_EVERY=1 bash tests/gpt5/run_phase736_source_replacement_generation_closure_round.sh
```

主测试：

```bash
PHASE736_ROUND_NAME=main PHASE736_MAX_PAIRS=8 PHASE736_TOP_PATHS=4 PHASE736_MAX_NEW_TOKENS=3 PHASE736_LOG_EVERY=2 bash tests/gpt5/run_phase736_source_replacement_generation_closure_round.sh
```

确认测试：

```bash
PHASE736_ROUND_NAME=confirm PHASE736_MAX_PAIRS=12 PHASE736_TOP_PATHS=4 PHASE736_MAX_NEW_TOKENS=3 PHASE736_LOG_EVERY=3 bash tests/gpt5/run_phase736_source_replacement_generation_closure_round.sh
```

说明：本阶段没有使用量化。由于 source contribution replacement（源贡献替换）必须读取 attention weights（注意力权重），脚本使用 eager attention（急切注意力）而不是 flash attention（闪存注意力）。三模型按 qwen3、GLM4、DS7B 顺序运行，并传入 `--hard-exit-after-model`，每个模型结束后释放 GPU。

### 测试原理

对每个 donor / recipient case（供体 / 受体样本），先在目标 head 上计算 source group 的注意力值贡献：

```text
C_G(l,h)=sum_{s in G} A_{t,s}^{l,h} V_s^{l,h}
```

然后在 recipient prompt 的同一层同一头，将对应 head input 做源限制替换：

```text
head_input' = head_input - C_G(recipient) + C_G(donor)
```

之后测量四类指标：

```text
1. hidden restore projection：
   recipient patched hidden 是否朝 donor hidden 移动。

2. donor answer logprob delta：
   donor answer token 在 recipient context 中是否被增强。

3. recipient answer logprob delta：
   原 recipient answer 是否被压低或扰动。

4. generation closure：
   greedy generation 是否真正转向 donor answer。
```

这比 Phase 735 的擦除更严格，因为它不是问“删掉是否破坏”，而是问“替换是否携带可迁移的功能信息”。

### 客观结果

三轮数据量：

```text
smoke：每模型 8 行 replacement rows
main：每模型 64 行 replacement rows
confirm：每模型 96 行 replacement rows
total：504 行 replacement rows
```

确认轮最强结果：

```text
qwen3:
  target_site = hidden_36
  top path = L35H0 <- self_last, conflict<-explicit
  restore = 1.803
  restore_fraction = 0.0070
  donor_logprob_delta = -0.151
  donor_hit_gain = 0.000
  changed_rate = 0.000
  role = state_transfer_only

GLM4:
  target_site = hidden_40
  top path = L23H17 <- all_pre_answer, explicit<-conflict
  restore = 0.805
  restore_fraction = 0.0055
  donor_logprob_delta = 0.041
  donor_hit_gain = 0.000
  changed_rate = 0.000
  role = content_transfer_candidate

DS7B:
  target_site = hidden_28
  top path = L22H24 <- all_pre_answer, conflict<-explicit
  restore = 15.831
  restore_fraction = 0.0270
  donor_logprob_delta = 0.286
  donor_hit_gain = 0.000
  changed_rate = 0.167
  role = content_transfer_candidate
```

DS7B 确认轮的 source group 排序非常稳定：

```text
L22H24 all_pre_answer conflict<-explicit:
  restore = 15.831
  donor_logprob_delta = 0.286
  changed_rate = 0.167

L22H24 records_all conflict<-explicit:
  restore = 14.701
  donor_logprob_delta = 0.249
  changed_rate = 0.167

L22H24 target_record_line conflict<-explicit:
  restore = 11.129
  donor_logprob_delta = 0.244
  changed_rate = 0.083

L22H24 target_value_tokens conflict<-explicit:
  restore = 7.035
  donor_logprob_delta = 0.144
  changed_rate = 0.000
```

主测试和确认测试一致：DS7B 的 L22H24 源贡献替换能显著推动 hidden_28 skeleton（隐藏态骨架）与 donor answer likelihood（供体答案似然），但不能稳定实现 donor answer generation（供体答案自然生成）。

### 对 Phase 735 判断的校正

Phase 735 关于 DS7B L22H24 是 source-resolved writer path（源分辨写入路径）的判断得到加强，但必须收紧：

```text
正确部分：
  L22H24 的 records_all / target_record_line / target_value_tokens
  确实携带可替换的源贡献。

新增进展：
  这些源贡献不是只在擦除时有破坏作用，
  也能在 donor -> recipient 替换时推动目标 hidden 和 donor likelihood。

必须收紧：
  这种推动还没有形成稳定 generation closure。
  因此它是 content transfer candidate（内容迁移候选路径），
  不是完整答案生成路径。
```

### 严格审视和硬伤

```text
1. generation hit gain 全部为 0。
   这说明当前替换只改变了隐藏态方向和答案似然，
   还没有让自然生成稳定转向 donor value。

2. DS7B changed_rate 只有 0.083 到 0.167。
   即使输出发生变化，也不是稳定命中 donor answer。

3. qwen3 最强路径 donor_logprob_delta 为负，
   更像 state transfer only（状态扰动 / 骨架迁移），
   不应解释为内容路径。

4. GLM4 的 L23H17 有正向 restore 与 logprob，
   但幅度很小且生成完全不变，
   只能视为弱 content transfer candidate。

5. source groups 仍有包含关系：
   all_pre_answer 包含 records_all、instruction、question 等，
   不能直接等同于语义值路径。

6. 替换位置只覆盖注意力 head input，
   没有同步替换后续 MLP rewriter、residual carrier、readout competition。
   这可能是生成闭环失败的主要原因。

7. 当前模型是小模型，
   层号、头号不能外推到大模型；
   可外推的只是功能链条：
   source contribution -> writer -> carrier -> nonlinear rewrite -> readout -> generation。
```

### 理论进展

当前拼图进一步收紧为：

```text
source token group
-> source contribution in attention head
-> prompt-type / value skeleton movement
-> answer likelihood bias
-> generation competition
```

Phase 736 的关键贡献是把“源分辨路径”从 erasure evidence（擦除证据）推进到 replacement evidence（替换证据）。这说明 DS7B L22H24 至少部分保存了可迁移的源词元贡献；但完整语言生成不是单个 head source contribution 决定的，后面仍有 MLP nonlinear rewrite（非线性重写）、readout competition（读出竞争）和 decoding attractor（解码吸引子）在限制输出闭合。

第一性原理判断：

```text
语言编码机制不是一个静态语义向量，
也不是单个 head 的注意力统计。
它更像一个条件化生成场：
source token contribution 提供局部内容势能，
residual / MLP / readout 共同决定这个势能是否能进入生成闭环。
```

### 下一步

Phase 736 已经完成当前阶段的 source-restricted replacement v0（源限制替换初版）。如果继续推进同一阶段的最终目标，下一步应进入：

```text
Phase 737: Writer-Rewriter Joint Replacement and Generation Closure

目标：
1. 以 DS7B L22H24 records_all / target_record_line / target_value_tokens 为入口。
2. 联合替换后续 MLP 候选：
   L22:mlp[957:1017]
   L27:mlp[2872:2932]
3. 同时检查 hidden_28、final hidden、answer logprob、greedy generation。
4. 判断 generation closure 失败是：
   a. source writer 不足；
   b. downstream nonlinear rewrite 不足；
   c. readout competition 抵消；
   d. decoding attractor 太强。
5. 只在 DS7B 上做完整验证；
   qwen3 / GLM4 只做轻量对照，避免强行同构。
```

阶段性结论：

```text
Phase 736 支持“源词元贡献可迁移”，
但反证了“单个源限制 head 替换即可闭合生成”。
下一阶段必须从单 head replacement 进入 writer + rewriter + readout 的联合路径闭环。
```

## Phase 737: Writer-Rewriter Joint Replacement and Generation Closure [2026-06-28 23:40]

### 任务背景

本阶段首先分析 Phase 736 的复盘内容。该复盘判断基本正确：Phase 736 同时给出了重要正结果和关键负结果。正结果是 source-restricted replacement（源限制替换）确实能推动 DS7B 的 hidden_28 skeleton（隐藏态28骨架）和 donor answer likelihood（供体答案似然）；负结果是 generation hit gain（生成命中增益）仍为 0，说明单个 L22H24 source contribution（源贡献）不是完整答案生成路径。

因此 Phase 737 继续同一阶段性目标，不再只测 single writer replacement（单写入器替换），而是测试：

```text
source writer
+ MLP rewriter
-> readout margin
-> generation closure
```

核心问题：

```text
Phase 736 生成闭合失败，
是因为 source writer 不足，
还是因为没有同步替换后续 MLP rewriter / readout competition？
```

### 生成脚本

新增正式脚本：

```text
tests/gpt5/phase737_writer_rewriter_joint_replacement.py
```

新增跨模型运行脚本：

```text
tests/gpt5/run_phase737_writer_rewriter_joint_replacement_round.sh
```

输出目录：

```text
results/glm5_phase737_writer_rewriter_joint_replacement/
```

确认轮主要结果：

```text
results/glm5_phase737_writer_rewriter_joint_replacement/confirm/phase737_cross_model_summary.md
results/glm5_phase737_writer_rewriter_joint_replacement/confirm/phase737_cross_model_summary.json
results/glm5_phase737_writer_rewriter_joint_replacement/confirm/phase737_atlas_graph.json
results/glm5_phase737_writer_rewriter_joint_replacement/confirm/phase737_qwen3_joint_rows.jsonl
results/glm5_phase737_writer_rewriter_joint_replacement/confirm/phase737_glm4_joint_rows.jsonl
results/glm5_phase737_writer_rewriter_joint_replacement/confirm/phase737_deepseek7b_joint_rows.jsonl
```

### 执行命令

静态检查：

```bash
python -m py_compile tests/gpt5/phase737_writer_rewriter_joint_replacement.py
bash -n tests/gpt5/run_phase737_writer_rewriter_joint_replacement_round.sh
```

干运行：

```bash
python tests/gpt5/phase737_writer_rewriter_joint_replacement.py --dry-run --round-name smoke --max-pairs 1 --top-paths 2 --top-mlp 1 --mode-set compact
```

冒烟测试：

```bash
PHASE737_ROUND_NAME=smoke PHASE737_MAX_PAIRS=1 PHASE737_TOP_PATHS=2 PHASE737_TOP_MLP=1 PHASE737_MAX_NEW_TOKENS=2 PHASE737_MODE_SET=compact PHASE737_LOG_EVERY=1 bash tests/gpt5/run_phase737_writer_rewriter_joint_replacement_round.sh
```

主测试：

```bash
PHASE737_ROUND_NAME=main PHASE737_MAX_PAIRS=4 PHASE737_TOP_PATHS=3 PHASE737_TOP_MLP=2 PHASE737_MAX_NEW_TOKENS=3 PHASE737_MODE_SET=compact PHASE737_LOG_EVERY=1 bash tests/gpt5/run_phase737_writer_rewriter_joint_replacement_round.sh
```

确认测试：

```bash
PHASE737_ROUND_NAME=confirm PHASE737_MAX_PAIRS=8 PHASE737_TOP_PATHS=3 PHASE737_TOP_MLP=2 PHASE737_MAX_NEW_TOKENS=3 PHASE737_MODE_SET=compact PHASE737_LOG_EVERY=2 bash tests/gpt5/run_phase737_writer_rewriter_joint_replacement_round.sh
```

说明：本阶段没有使用量化。由于仍需读取 attention weights（注意力权重）计算 source contribution（源贡献），脚本继续使用 eager attention（急切注意力）。这不是 flash attention（闪存注意力），但这是当前替换算法的必要条件。三模型按 qwen3、GLM4、DS7B 顺序运行，并传入 `--hard-exit-after-model`。

### 测试原理

Phase 737 比 Phase 736 多了 MLP donor -> recipient replacement（多层感知机供体到受体替换）。

source writer 替换仍为：

```text
head_input' = head_input - C_G(recipient) + C_G(donor)
```

MLP group 替换为：

```text
MLP_l[start:end]' = MLP_l[start:end]^{donor}
```

联合替换包括：

```text
source_only
mlp_only
mlp_all
source_plus_top_mlp
source_plus_all_mlp
```

新增 readout competition（读出竞争）指标：

```text
donor_vs_recipient_margin
= logit(donor_answer) - logit(recipient_answer)
```

如果 donor_logprob 上升但 margin 仍为强负，说明 donor answer 仍无法突破 recipient / format / prose attractor（受体 / 格式 / 叙述吸引子）。

### 客观数据量

```text
smoke：每模型 10 行 joint rows
main：每模型 96 行 joint rows
confirm：每模型 192 行 joint rows
total：894 行 joint rows
```

### 确认轮关键结果

```text
qwen3:
  target_site = hidden_36
  top intervention =
    source_plus_all_mlp
    L35H0 <- self_last
    + L34:mlp[85:128]
    + L28:mlp[299:341]
    explicit<-conflict
  restore = 1.091
  donor_logprob_delta = 0.119
  margin_delta = 0.125
  patched_margin = -17.016
  donor_hit_gain = 0.000
  changed_rate = 0.000

GLM4:
  target_site = hidden_40
  top intervention =
    source_plus_all_mlp
    L23H17 <- all_pre_answer
    + L38:mlp[2597:2665]
    + L38:mlp[3007:3075]
    conflict<-explicit
  restore = 2.832
  donor_logprob_delta = 0.333
  margin_delta = 0.351
  patched_margin = -8.867
  donor_hit_gain = 0.000
  changed_rate = 0.000

DS7B:
  target_site = hidden_28
  top summary intervention =
    mlp_only
    L27:mlp[2872:2932]
    conflict<-explicit
  restore = 8.444
  donor_logprob_delta = 0.108
  margin_delta = 0.231
  patched_margin = -9.100
  donor_hit_gain = 0.000
  recipient_hit_loss = 0.125
  changed_rate = 0.250
```

DS7B 更细结果：

```text
source_plus_all_mlp
L22H24 <- target_record_line
+ L27:mlp[2872:2932]
+ L22:mlp[957:1017]
explicit<-conflict:
  restore = 20.723
  donor_logprob_delta = 0.428
  margin_delta = 0.520
  patched_margin = -12.933
  donor_hit_gain = 0.000
  recipient_hit_loss = 0.125
  changed_rate = 0.125

source_plus_all_mlp
L22H24 <- all_pre_answer
+ L27:mlp[2872:2932]
+ L22:mlp[957:1017]
conflict<-explicit:
  restore = 26.561
  donor_logprob_delta = 0.374
  margin_delta = 0.414
  patched_margin = -8.917
  donor_hit_gain = 0.000
  changed_rate = 0.125
```

按 intervention mode 聚合：

```text
qwen3:
  donor hit gains = 0
  changed = 0
  recipient hit loss = 0

GLM4:
  donor hit gains = 0
  changed = 0
  recipient hit loss = 0
  mlp_all mean_margin_delta = 0.291
  source_plus_all_mlp mean_margin_delta = 0.303

DS7B:
  donor hit gains = 0
  changed = 17 / 192
  recipient hit loss = 4 / 192
  source_only mean_margin_delta = 0.293
  source_plus_top_mlp mean_margin_delta = 0.404
  source_plus_all_mlp mean_margin_delta = 0.430
```

### 样本级观察

DS7B 的 changed samples（生成变化样本）不是 donor answer 命中，而多为格式 / 前缀轨迹变化：

```text
carrot:taste explicit<-conflict
baseline: 'earthy'
patched : 'The taste of'
donor = sweet
recipient = earthy

stone:category conflict<-explicit
baseline: 'The category of'
patched : 'stone.category'
donor = object
recipient = fruit
```

这说明联合替换确实能扰动 generation trajectory（生成轨迹），但扰动方向还不是 donor value answer（供体值答案），更像 format/prose route（格式 / 叙述路线）被激活。

### 对 Phase 736 复盘的判断

Phase 736 复盘文本基本正确。Phase 737 进一步验证了其中最关键的判断：

```text
source writer replacement 可以推动 hidden 和 likelihood；
加入 MLP rewriter 后可以进一步推动 readout margin；
但 generation closure 仍然没有成立。
```

所以当前最准确结论是：

```text
DS7B L22H24 + L27/L22 MLP group
构成了可检测的 writer-rewriter-readout subpath（写入器-重写器-读出子路径），
但还不是完整 generation path（生成路径）。
```

### 严格审视和硬伤

```text
1. donor_hit_gain 仍然为 0。
   这是最强负结果，说明当前联合替换没有真正恢复 donor answer generation。

2. patched_margin 仍然为强负。
   qwen3 top patched_margin = -17.016
   GLM4 top patched_margin = -8.867
   DS7B top patched_margin = -9.100
   虽然 margin_delta 为正，但 donor answer 仍远远输给 recipient / 其他输出。

3. DS7B 的 changed_rate 来自格式或前缀扰动，
   不是 donor value 命中。

4. MLP replacement 是 output-channel group（输出通道组）替换，
   仍不是 neuron activation（神经元激活）级别证明。

5. donor MLP slice 直接替换到 recipient context 可能 off-manifold（离流形），
   会制造状态扰动，不等于自然计算路径完全存在。

6. source_plus_all_mlp 对 margin 的提升强于 source_only，
   但提升幅度仍不足以翻转 readout competition。

7. 小模型偏差仍然很强。
   DS7B 的生成扰动更明显，可能来自小模型显式记录依赖强和解码稳定性弱。
```

### 理论进展

Phase 737 使当前图谱从：

```text
source writer -> hidden / likelihood
```

推进到：

```text
source writer + MLP rewriter -> readout margin -> partial generation perturbation
```

最重要的拼图是：

```text
readout margin 是 generation closure 的硬瓶颈。
```

也就是说，当前不是完全没有内容路径；相反，内容路径已经能稳定推动 hidden、logprob、margin。但是 margin 仍停留在强负区间，说明 donor answer 没有成为最强竞争输出。因此自然生成不会闭合。

更准确的机制链条更新为：

```text
source token group
-> attention writer
-> MLP rewriter / amplifier
-> residual carrier
-> readout margin competition
-> format/prose/value route selection
-> generation closure
```

第一性原理判断：

```text
语言生成不是“某个值被写入就输出”，
而是多个候选生成场在 readout 端竞争。
source writer 和 MLP rewriter 提供的是势能偏置，
只有当该偏置足以翻转 readout margin，
并且不被 format/prose attractor 吸走时，
才会变成自然生成闭合。
```

### 下一步

Phase 737 已经完成 Phase 736 规划中的 writer + rewriter 联合路径验证。继续在同一思路上盲目扩大替换组合，边际收益会下降。下一步应进入更靠近瓶颈的阶段：

```text
Phase 738: Readout Margin and Token Continuation Gate Audit

目标：
1. 不再优先寻找更多 writer / MLP 替换组合。
2. 直接测 donor answer 为什么 margin 仍为强负。
3. 拆分竞争对象：
   donor answer
   recipient answer
   format/prose prefix
   relation echo
   object echo
4. 测 token0 和 token1：
   donor first token 是否被提升；
   donor second token continuation 是否被阻断。
5. 对 DS7B changed samples 做 focused audit：
   为什么从答案路线转向 'The taste of'、'stone.category' 等格式路线。
6. 输出 readout competition atlas（读出竞争图谱），
   把 writer / rewriter 路径和最终竞争失败点接起来。
```

阶段性结论：

```text
Phase 737 支持“writer + rewriter 能增强 readout margin”，
但反证了“writer + rewriter 联合替换足以闭合自然生成”。
当前瓶颈已经从路径写入后移到 readout margin 和 token continuation gate。
```

## Phase 738: Readout Margin and Token Continuation Gate Audit [2026-06-29 00:08]

### 触发原因

Phase 737 证明 source writer（源写入器）和 MLP rewriter（多层感知机重写器）联合替换能够稳定提高 donor answer（供体答案）的 readout margin（读出边际），但仍不能自然闭合 generation（生成）。

因此本阶段不继续扩大 writer / rewriter（写入器 / 重写器）组合，而是直接检查两个更靠近输出端的问题：

```text
1. token0（第一个词元）处 donor answer 为什么仍然输给其他候选。
2. 如果强制给出 donor token0，token1（第二个词元）的续写路线会走向答案闭合，还是走向格式 / 关系 / 回声路线。
```

### 生成脚本

```text
tests/gpt5/phase738_readout_margin_continuation_audit.py
tests/gpt5/run_phase738_readout_margin_continuation_audit_round.sh
```

结果目录：

```text
results/glm5_phase738_readout_margin_continuation_audit/
```

### 执行命令

静态检查：

```bash
python -m py_compile tests/gpt5/phase738_readout_margin_continuation_audit.py
bash -n tests/gpt5/run_phase738_readout_margin_continuation_audit_round.sh
```

冒烟测试：

```bash
PHASE738_ROUND_NAME=smoke PHASE738_MAX_PAIRS=1 PHASE738_TOP_AUDITS=2 PHASE738_LOG_EVERY=1 bash tests/gpt5/run_phase738_readout_margin_continuation_audit_round.sh
```

主测试：

```bash
PHASE738_ROUND_NAME=main PHASE738_MAX_PAIRS=8 PHASE738_TOP_AUDITS=5 PHASE738_LOG_EVERY=2 bash tests/gpt5/run_phase738_readout_margin_continuation_audit_round.sh
```

确认测试：

```bash
PHASE738_ROUND_NAME=confirm PHASE738_MAX_PAIRS=12 PHASE738_TOP_AUDITS=5 PHASE738_LOG_EVERY=3 bash tests/gpt5/run_phase738_readout_margin_continuation_audit_round.sh
```

三轮均按 qwen3 -> GLM4 -> DS7B 顺序运行，并使用 `--hard-exit-after-model`（每个模型完成后硬退出）释放 GPU（图形处理器）内存。脚本未使用 quantization（量化）。本阶段需要 attention weights（注意力权重）来复用 Phase 737 的 source contribution（源贡献）替换，因此使用 eager attention（急切注意力路径），没有强行使用 flash attention（闪存注意力）。

### 测试原理

本阶段从 Phase 737 的确认结果中读取每个模型最强的 joint intervention（联合干预）候选，然后做两类审计：

```text
第一类：token0 candidate competition（第一个词元候选竞争）

比较 donor_answer（供体答案）、recipient_answer（受体答案）、
object_echo（对象回声）、relation_echo（关系回声）、
format_the / format_answer / format_value / format_it / format_of（格式前缀）
在 patched readout（修补后读出）处谁最强。
```

```text
第二类：forced donor-token continuation（强制供体词元后的续写）

先强制生成 donor answer 的第一个词元，
再比较下一步 token1（第二个词元）更倾向于：
donor_answer_token1（供体答案后续词元）、
recipient_answer_token1（受体答案后续词元）、
stop token（停止词元）、
is / of / colon / comma（格式或散文连接词）、
relation_echo（关系回声）、
object_echo（对象回声）。
```

核心指标：

```text
mean_token0_donor_logprob_delta
mean_token0_margin_delta_donor_vs_recipient
mean_token0_patched_margin_donor_vs_recipient
token0_donor_top_candidate_rate
token0_patched_best_counts
token1_patched_best_counts
```

### 客观结果

行数：

```text
smoke：每个模型 2 行，共 6 行
main：每个模型 40 行，共 120 行
confirm：每个模型 60 行，共 180 行
三轮总计 306 行 readout audit（读出审计）记录
```

确认轮汇总：

```text
qwen3:
target_site = hidden_36
最佳审计 = source_plus_all_mlp, L35H0<-self_last, L34/L28 MLP
mean_token0_margin_delta_donor_vs_recipient = +0.099
mean_token0_patched_margin_donor_vs_recipient = -15.651
token0_donor_top_candidate_rate = 0.000
token0_patched_best_counts = {'recipient_answer': 12}
token1_patched_best_counts = {'cont_is': 6, 'cont_stop_newline': 6}
```

```text
GLM4:
target_site = hidden_40
最佳审计 = source_plus_all_mlp, L39H21<-self_last, L38 MLP pair
mean_token0_margin_delta_donor_vs_recipient = +0.310
mean_token0_patched_margin_donor_vs_recipient = -8.244
token0_donor_top_candidate_rate = 0.000
token0_patched_best_counts = {'recipient_answer': 12}
token1_patched_best_counts = {'cont_is': 3, 'cont_of': 1, 'object_echo': 1, 'relation_echo': 7}
```

```text
DS7B:
target_site = hidden_28
最佳审计 = source_plus_all_mlp, L22H24<-target_record_line, L27/L22 MLP
mean_token0_margin_delta_donor_vs_recipient = +0.411
mean_token0_patched_margin_donor_vs_recipient = -11.932
token0_donor_top_candidate_rate = 0.000
token0_patched_best_counts = {'object_echo': 1, 'recipient_answer': 10, 'relation_echo': 1}
token1_patched_best_counts = {'cont_is': 11, 'cont_of': 1}
```

跨模型总计：

```text
qwen3 token0 patched best:
recipient_answer = 60 / 60
donor_answer = 0 / 60

GLM4 token0 patched best:
recipient_answer = 60 / 60
donor_answer = 0 / 60

DS7B token0 patched best:
recipient_answer = 39 / 60
object_echo = 11 / 60
format_the = 8 / 60
relation_echo = 2 / 60
donor_answer = 0 / 60
```

forced donor token0 后的 token1：

```text
qwen3:
cont_is = 32 / 60
cont_stop_newline = 28 / 60

GLM4:
relation_echo = 30 / 60
cont_is = 20 / 60
object_echo = 4 / 60
cont_of = 4 / 60
cont_stop_newline = 2 / 60

DS7B:
cont_is = 49 / 60
cont_of = 8 / 60
object_echo = 2 / 60
cont_stop_newline = 1 / 60
```

本阶段还生成了 atlas（图谱）文件：

```text
node_count = 30
edge_count = 46
source_phase = 738
```

### 对 Phase 737 分析的判断

上传的 Phase 737 分析基本正确，但需要收紧：

```text
正确部分：
source writer + MLP rewriter 的确能提升 donor answer 的读出势能。
这种提升在 DS7B 上最明显，在 GLM4 上中等，在 qwen3 上较弱。
```

```text
需要修正的部分：
不能把 readout margin 改善理解为 generation closure（生成闭合）。
Phase 738 证明 donor answer 虽然被提高，但仍没有赢得 token0 候选竞争。
即使强制 donor token0，token1 也主要进入 is / of / relation_echo / newline 等路线。
```

因此，Phase 737 的真实位置应写成：

```text
writer / rewriter 路径已经找到一部分，
但它只把内容推近输出端，
还没有穿过 readout competition（读出竞争）和 continuation gate（续写门控）。
```

### 理论进展

当前机制链条应更新为：

```text
source token group
-> attention writer
-> MLP rewriter / amplifier
-> residual carrier
-> readout candidate field
-> token0 competition
-> token1 continuation gate
-> natural generation closure
```

Phase 738 的关键拼图是：

```text
readout margin 改善不是生成闭合；
生成闭合至少需要两个门槛：
1. donor answer 在 token0 竞争中胜出；
2. donor token0 后的 token1 续写不被格式 / 关系 / 散文路线吸走。
```

这解释了为什么 Phase 737 中 DS7B 会出现 changed output（输出被改变），但 donor answer hit gain（供体答案命中增益）仍为 0：

```text
干预已经改变了生成场，
但改变方向不是稳定答案闭合，
而是把模型推向 object echo、relation echo、format prefix 或 prose connector。
```

### 问题和硬伤

1. 当前候选集合仍是人工列出的，虽然包含 answer、echo、format、continuation 等主要竞争项，但不等于完整 vocabulary（词表）竞争图。

2. 很多答案是单词，token1 审计不能简单解释为“答案第二词是否正确”，更准确说它测的是强制供体首词元之后模型进入哪条续写路线。

3. 本阶段复用 Phase 737 的 patch（修补）干预，干预后状态可能偏离自然流形，因此结果说明“当前干预不能闭合生成”，不能直接说明自然模型不存在对应闭合路径。

4. 当前模型都是小模型，内部结构可能偏粗糙。DS7B 的 object_echo / format_the 竞争较明显，可能包含小模型偏差。

5. 本阶段没有直接修改 unembedding（反嵌入）或 final norm（最终归一化），所以还不能判断读出端本身的最小翻转阈值。

### 下一步

Phase 738 已完成 Phase 737 后的同阶段闭环验证。下一阶段不应继续盲目扩大 writer / MLP patch（写入器 / 多层感知机修补）组合，而应进入新的瓶颈阶段：

```text
Phase 739: Readout Threshold and Closure Boundary Test

目标：
1. 直接测 donor answer 从负 margin 到 token0 top1 需要多大 logit shift。
2. 分离 final hidden（最终隐藏态）、final norm（最终归一化）、unembedding（反嵌入）三个环节。
3. 对 donor answer direction（供体答案方向）做最小强度扫描，而不是继续做大规模局部 patch。
4. 测 token0 top1 被翻转后，token1 是否仍进入 format/prose route（格式 / 散文路线）。
5. 输出 readout threshold atlas（读出阈值图谱），连接 source writer、MLP rewriter、readout competition、continuation gate。
```

阶段性结论：

```text
Phase 738 支持 Phase 737 的正结果：
writer + rewriter 能推动 donor answer 的读出势能。

但 Phase 738 同时给出更强边界：
当前推动远不足以让 donor answer 赢得 token0 竞争；
并且强制 donor token0 后，token1 仍偏向格式 / 关系 / 散文续写。

所以当前真正瓶颈已经从“有没有内容路径”移动到：
readout threshold（读出阈值）和 continuation closure（续写闭合）。
```

## Phase 739: Readout Threshold and Closure Boundary Test [2026-06-29 00:37]

### 触发原因

用户上传的两份分析总体正确：Phase 738 已经把生成失败定位到 token0 competition（第一个词元竞争）和 token1 continuation gate（第二词元续写门）。其中最重要的正确部分是：

```text
1. 当前苹果—水果—属性图谱已经不是普通 patch（修补）实验，
   而是进入 source token（源词元）-> writer（写入器）-> rewriter（重写器）-> readout competition（读出竞争）的路径级图谱。

2. Phase 738 证明 writer + rewriter（写入器 + 重写器）能提高 donor answer（供体答案）势能，
   但 donor answer 仍不能自然赢得 token0（第一个词元）竞争。

3. 下一步应测 readout threshold（读出阈值），而不是继续扩大 writer / MLP patch（写入器 / 多层感知机修补）组合。
```

需要收紧的地方是：

```text
Phase 738 中 token1（第二词元）偏向 is / of / relation_echo（关系回声）等路线，
不能直接说明 continuation gate（续写门）永远失败；
因为如果 token0（第一个词元）本身没有赢，后续状态仍可能不是答案闭合状态。

所以 Phase 739 直接测试：
如果人工跨过 final readout threshold（最终读出阈值），生成是否能够闭合？
```

### 生成脚本

```text
tests/gpt5/phase739_readout_threshold_closure_boundary.py
tests/gpt5/run_phase739_readout_threshold_closure_boundary_round.sh
```

结果目录：

```text
results/glm5_phase739_readout_threshold_closure_boundary/
```

### 执行命令

静态检查：

```bash
python -m py_compile tests/gpt5/phase739_readout_threshold_closure_boundary.py
bash -n tests/gpt5/run_phase739_readout_threshold_closure_boundary_round.sh
```

dry run（空跑）：

```bash
python tests/gpt5/phase739_readout_threshold_closure_boundary.py --dry-run --max-pairs 1 --top-audits 2 --round-name dry
```

smoke（冒烟测试）：

```bash
PHASE739_ROUND_NAME=smoke PHASE739_MAX_PAIRS=1 PHASE739_TOP_AUDITS=1 PHASE739_LOG_EVERY=1 bash tests/gpt5/run_phase739_readout_threshold_closure_boundary_round.sh
```

main（主测试）：

```bash
PHASE739_ROUND_NAME=main PHASE739_MAX_PAIRS=6 PHASE739_TOP_AUDITS=2 PHASE739_LOG_EVERY=2 bash tests/gpt5/run_phase739_readout_threshold_closure_boundary_round.sh
```

confirm（确认测试）：

```bash
PHASE739_ROUND_NAME=confirm PHASE739_MAX_PAIRS=10 PHASE739_TOP_AUDITS=2 PHASE739_LOG_EVERY=3 bash tests/gpt5/run_phase739_readout_threshold_closure_boundary_round.sh
```

三轮均按 qwen3 -> GLM4 -> DS7B 顺序执行，并使用 `--hard-exit-after-model`（每个模型完成后硬退出）。脚本未使用 quantization（量化）。由于 Phase 739 复用 Phase 738 选出的 joint path（联合路径）状态，其中可能包含 source contribution replacement（源贡献替换），仍需 attention weights（注意力权重），所以使用 eager attention（急切注意力路径）。

### 测试原理

Phase 739 不再寻找更多中间节点，而是在 Phase 738 的最佳 joint intervention（联合干预）基础上，直接测 final_norm_output（最终归一化输出）处需要多大的 readout boost（读出增强）才能让 donor answer（供体答案）成为第一个词元的 top1（第一候选）。

设修补后 logits（词表分数）为：

```text
l(y) = W_U(y)^T h_final
```

其中：

```text
W_U(y) 是 token y（词元 y）的 unembedding（反嵌入）方向。
h_final 是 final_norm_output（最终归一化输出）。
```

对当前 top competitor（最强竞争词元）c，构造方向：

```text
d = normalize(W_U(y_donor) - W_U(c))
```

然后扫描：

```text
h_final' = h_final + alpha * d
```

记录：

```text
1. donor answer（供体答案）何时成为 candidate top1（候选集合第一）。
2. donor answer（供体答案）何时成为 vocabulary top1（全词表第一）。
3. 一旦 token0（第一个词元）被翻转，继续 greedy generation（贪心生成）是否能闭合短答。
```

这个测试的意义不是证明自然路径已经存在，而是测量：

```text
自然 writer / rewriter path（写入器 / 重写器路径）距离真正生成闭合还差多大的 readout threshold（读出阈值）。
```

### 客观结果

行数：

```text
smoke：每个模型 1 行，共 3 行
main：每个模型 12 行，共 36 行
confirm：每个模型 20 行，共 60 行
三轮总计 99 行 threshold audit（阈值审计）记录
```

confirm（确认轮）汇总：

```text
qwen3:
target_site = hidden_36
最佳审计 = source_plus_all_mlp, L35H0<-self_last, L34/L28 MLP
patched_candidate_best_counts = {'recipient_answer': 10}
mean_patched_margin_donor_vs_vocab_top = -15.244
mean_alpha_star_vocab_top = 11.196
mean_first_alpha_donor_vocab_top = 17.986
vocab_flip_found_rate = 1.000
boosted_generation_donor_hit_rate = 1.000
boosted_generation_class_counts = {'answer_stop': 10}
```

```text
GLM4:
target_site = hidden_40
最佳审计 = source_plus_all_mlp, L23H17<-instruction, L38 MLP pair
patched_candidate_best_counts = {'recipient_answer': 10}
mean_patched_margin_donor_vs_vocab_top = -7.825
mean_alpha_star_vocab_top = 8.911
mean_first_alpha_donor_vocab_top = 11.922
vocab_flip_found_rate = 1.000
boosted_generation_donor_hit_rate = 1.000
boosted_generation_class_counts = {'answer_stop': 10}
```

```text
DS7B:
target_site = hidden_28
最佳审计 = source_plus_all_mlp, L22H24<-all_pre_answer, L27/L22 MLP
patched_candidate_best_counts = {'recipient_answer': 6, 'object_echo': 2, 'format_the': 2}
mean_patched_margin_donor_vs_vocab_top = -8.016
mean_alpha_star_vocab_top = 5.962
mean_first_alpha_donor_vocab_top = 11.654
vocab_flip_found_rate = 1.000
boosted_generation_donor_hit_rate = 1.000
boosted_generation_class_counts = {'answer_stop': 10}
```

跨模型行级统计：

```text
qwen3:
patched_candidate_best = recipient_answer 20 / 20
boosted_generation = answer_stop 20 / 20
mean_first_alpha_donor_vocab_top = 17.988

GLM4:
patched_candidate_best = recipient_answer 20 / 20
boosted_generation = answer_stop 20 / 20
mean_first_alpha_donor_vocab_top = 12.106

DS7B:
patched_candidate_best = recipient_answer 15 / 20, object_echo 2 / 20, format_the 2 / 20, relation_echo 1 / 20
boosted_generation = answer_stop 15 / 20, answer_mentioned 5 / 20
mean_first_alpha_donor_vocab_top = 14.526
```

本阶段生成 atlas（图谱）：

```text
node_count = 12
edge_count = 9
source_phase = 739
```

### 对当前分析的修正

Phase 738 的“续写门失败”需要被 Phase 739 收紧：

```text
Phase 738 看到的是：
在自然 writer + rewriter 势能不足时，
强制 donor token0 后，token1 仍容易进入 is / of / echo route。

Phase 739 看到的是：
如果直接在 final readout（最终读出）层给足够大的 donor-vs-current-top boost（供体对当前第一候选的增强），
三个模型都能让 donor answer 成为 top1，并且短答生成可以闭合。
```

因此更准确的结论是：

```text
continuation gate（续写门）不是完全不可通过；
当前主要硬瓶颈是自然路径没有给 final readout（最终读出）足够强的 token0 flip force（第一词元翻转力）。

一旦 token0 真正以足够强的读出优势进入答案路线，
短答闭合可以发生。
```

但这个结论必须谨慎：

```text
Phase 739 的 readout boost（读出增强）是人工外力，
不等于模型自然内部已经存在同等强度的路径。
```

### 理论进展

当前机制链条进一步收紧为：

```text
source token group
-> attention writer
-> MLP rewriter / amplifier
-> residual carrier
-> readout potential
-> readout threshold
-> token0 route entry
-> short-answer closure
```

Phase 739 的关键拼图是：

```text
自然路径不是完全错路；
它缺的是足够强的 final readout threshold crossing（最终读出阈值跨越）。
```

数学上，生成闭合至少需要：

```text
l(y_donor) + Delta_readout(y_donor) > max_c l(c)
```

其中 Phase 739 测得的 Delta_readout（读出增量）在三个模型中都不小：

```text
qwen3 约需要 18 的 first_alpha（实际首次翻转增强）
GLM4 约需要 12
DS7B 约需要 12 到 17，取决于路径方向
```

这说明 Phase 737 / 738 的 writer + rewriter 已经把答案推向读出端，但离真正 top1 还有明显距离。

### 问题和硬伤

1. Phase 739 是 readout intervention（读出干预），属于人工边界测试，不是自然机制证明。

2. alpha（增强强度）较大，说明最终状态可能已经 off-manifold（离自然流形）。它只能说明“读出端具备答案闭合能力”，不能说明自然路径能产生这么大的增强。

3. 当前 boosted_generation_donor_hit_rate（增强后供体生成命中率）为 1.0，但很多答案是单词短答，因此 answer_stop（答案后停止）不等于复杂多词答案闭合。

4. DS7B 有 5 / 20 是 answer_mentioned（答案被提及）而不是严格 answer_stop（答案停止），说明 DS7B 的格式 / 散文吸引子仍然存在。

5. 本阶段仍未定位哪个自然节点能提供这 12 到 18 量级的 readout boost（读出增强）。

6. 当前模型是小模型，阈值大小和具体路径不能直接外推到大模型。

### 下一步

Phase 739 已经完成 Phase 738 后的读出阈值闭环。下一阶段应继续沿同一大阶段推进，但不要再做单纯人工 readout boost（读出增强），而要寻找自然来源：

```text
Phase 740: Natural Readout Boost Source Backtrace

目标：
1. 以 Phase 739 的 donor-vs-current-top readout direction（供体对当前第一候选读出方向）为目标方向。
2. 回溯哪些 late attention / late MLP / residual stream（后期注意力 / 后期多层感知机 / 残差流）自然贡献沿这个方向。
3. 区分三类来源：
   a. value content boost（值内容增强）
   b. recipient suppression（受体抑制）
   c. format / echo suppression（格式 / 回声抑制）
4. 不再只看 logprob delta（对数概率增量），而看是否能提供接近 Phase 739 测得阈值的自然增量。
5. 输出 natural threshold source atlas（自然阈值来源图谱）。
```

阶段性结论：

```text
Phase 739 证明：
当前生成失败不是因为答案路线无法被读出，
而是自然 writer + rewriter 给出的读出势能不足以跨过 token0 top1 阈值。

人工跨过阈值后，三模型都能短答闭合。

所以真正的下一瓶颈是：
寻找自然计算中谁负责提供 readout threshold crossing force（读出阈值跨越力）。
```

## Phase 740: Natural Readout Boost Source Backtrace [2026-06-29 07:11]

### 问题来源

用户给出的 Phase 739 分析基本正确。Phase 739 已经证明：

```text
人工 final readout boost（最终读出增强）可以让 donor answer（供体答案）进入 token0 top1（第一个生成词元第一名），
但自然 writer + rewriter（写入器 + 重写器）路径没有提供足够强的 readout threshold crossing force（读出阈值跨越力）。
```

因此 Phase 740 没有继续增加人工 readout boost（读出增强），而是回溯：

```text
自然 donor path（供体路径）里谁提供了目标答案方向？
当前 patched path（修补路径）到底转移了多少这种方向？
```

### 脚本

```text
tests/gpt5/phase740_natural_readout_boost_source_backtrace.py
tests/gpt5/run_phase740_natural_readout_boost_source_backtrace_round.sh
```

### 命令

静态检查：

```bash
python -m py_compile tests/gpt5/phase740_natural_readout_boost_source_backtrace.py
bash -n tests/gpt5/run_phase740_natural_readout_boost_source_backtrace_round.sh
python tests/gpt5/phase740_natural_readout_boost_source_backtrace.py --model qwen3 --round-name dry --max-pairs 1 --top-audits 2 --dry-run
python tests/gpt5/phase740_natural_readout_boost_source_backtrace.py --model glm4 --round-name dry --max-pairs 1 --top-audits 2 --dry-run
python tests/gpt5/phase740_natural_readout_boost_source_backtrace.py --model deepseek7b --round-name dry --max-pairs 1 --top-audits 2 --dry-run
```

冒烟测试：

```bash
PHASE740_ROUND_NAME=smoke PHASE740_MAX_PAIRS=1 PHASE740_TOP_AUDITS=1 PHASE740_LOG_EVERY=1 bash tests/gpt5/run_phase740_natural_readout_boost_source_backtrace_round.sh
```

主测试：

```bash
PHASE740_ROUND_NAME=main PHASE740_MAX_PAIRS=6 PHASE740_TOP_AUDITS=1 PHASE740_LOG_EVERY=2 bash tests/gpt5/run_phase740_natural_readout_boost_source_backtrace_round.sh
```

确认测试：

```bash
PHASE740_ROUND_NAME=confirm PHASE740_MAX_PAIRS=10 PHASE740_TOP_AUDITS=2 PHASE740_LOG_EVERY=3 bash tests/gpt5/run_phase740_natural_readout_boost_source_backtrace_round.sh
```

脚本按 qwen3、GLM4、DS7B 顺序运行，并使用 `--hard-exit-after-model` 避免显存累积。

### 测试原理

Phase 740 直接复用 Phase 739 的 top threshold audit（阈值审计）结果。对每个模型选取 Phase 739 中最重要的路径，然后定义目标方向：

```text
d = normalize(W_U(y_donor) - W_U(c_top))
```

其中：

```text
y_donor = donor answer token（供体答案词元）
c_top = patched state 当前 vocab top token（修补状态当前词表第一词元）
W_U = unembedding（反嵌入矩阵）
```

然后测量三类量：

```text
1. donor_final_delta_proj
   donor natural final output（供体自然最终输出）相对 recipient（受体）沿 d 的投影增量。

2. patched_final_delta_proj
   Phase 739 选中的 patched path（修补路径）相对 recipient（受体）沿 d 的投影增量。

3. late component raw projection
   late attention / MLP output（后期注意力 / 多层感知机输出）沿 d 的原始投影。
```

为了和 Phase 739 可比，所有关键结果都除以 Phase 739 测得的 first_alpha / threshold（首次翻转阈值）：

```text
fraction = projected_delta / threshold
```

所以：

```text
fraction >= 1
```

表示该自然差异在读出方向上的强度已经接近或超过 Phase 739 人工翻转需要的阈值。

### 确认轮结果

结果目录：

```text
results/glm5_phase740_natural_readout_boost_source_backtrace/confirm/
```

跨模型摘要：

```text
results/glm5_phase740_natural_readout_boost_source_backtrace/confirm/phase740_cross_model_summary.md
```

核心结果：

| model | target site | threshold | patched final fraction | donor final fraction | top component | component patched fraction |
|---|---:|---:|---:|---:|---|---:|
| qwen3 | hidden_36 | 17.986 | 0.004 | 1.292 | L34:attn_out | 0.009 |
| GLM4 | hidden_40 | 12.291 | 0.029 | 1.892 | L38:mlp_out | 0.042 |
| DS7B | hidden_28 | 11.654 | 0.020 | 1.101 | L26:attn_out | 0.057 |

更细结果：

```text
qwen3:
donor final fraction ≈ 1.29
patched final fraction ≈ 0.003-0.004
候选组件：L34:attn_out、L31:attn_out、L33:mlp_out

GLM4:
donor final fraction ≈ 1.89-1.94
patched final fraction ≈ 0.029
候选组件：L38:mlp_out，其它 late components（后期组件）明显弱很多

DS7B:
donor final fraction ≈ 0.90-1.10
patched final fraction ≈ 0.015-0.020
候选组件：L26:attn_out、L27:mlp_out、L27:attn_out
```

### 客观结论

Phase 740 的最关键事实是：

```text
donor natural context（供体自然上下文）里确实存在足够强的 readout direction（读出方向）；
但是当前 writer + rewriter patch（写入器 + 重写器修补）只把极小比例传递到 final readout（最终读出）。
```

这把 Phase 739 的瓶颈进一步定位为：

```text
不是模型完全没有答案方向；
也不是 unembedding readout（反嵌入读出）完全不能读出答案；
而是当前已定位的路径没有把自然答案方向稳定送入最终读出阈值。
```

更具体地说：

```text
qwen3:
自然 donor 已经超过阈值，但 patch 只传递约 0.3%-0.4%。

GLM4:
自然 donor 明显超过阈值，patch 传递约 3%。

DS7B:
自然 donor 接近或略超阈值，patch 传递约 1.5%-2%。
```

### 理论进展

Phase 740 支持把当前语言编码路径写成：

```text
source evidence（源证据）
-> writer / local router（写入器 / 局部路由器）
-> rewriter / amplifier（重写器 / 放大器）
-> late carrier（后期承载器）
-> final readout direction（最终读出方向）
-> token0 threshold crossing（第一个词元阈值跨越）
```

其中 Phase 740 新增的拼图是：

```text
late carrier（后期承载器）不是没有目标方向；
目标方向在自然 donor path（供体路径）中很强；
当前失败点在于路径传递、放大和竞争抑制没有闭合。
```

这说明当前研究已经从：

```text
找哪个 head / channel 有用
```

推进到：

```text
找自然阈值方向如何被生成、传递、放大、冲洗或读出。
```

### 问题和硬伤

1. Phase 740 的 late component projection（后期组件投影）仍然是回溯证据，不是因果证明。

2. component raw projection（组件原始投影）发生在 final norm（最终归一化）之前，不能直接等价于最终 logit（对数几率）变化。

3. qwen3 和 DS7B 的候选 attention component（注意力组件）有明显 donor signal（供体信号），但 patched signal（修补信号）仍很弱，说明“看到了方向”不等于“完成闭合”。

4. GLM4 的 L38:mlp_out 是最清晰候选，但也只解释约 4% 的阈值，需要继续验证是否存在组合放大或竞争抑制。

5. 当前模型仍是小模型，内部结构可能有偏差，不能把具体层号直接外推为通用结论。

### 下一步

Phase 741 应继续处于同一阶段，目标不是扩大搜索，而是做因果验证：

```text
Phase 741: Causal Validation of Natural Threshold Source Candidates
```

具体任务：

```text
1. 对 Phase 740 的 top candidate components（最高候选组件）做 donor->recipient component transplant（供体到受体组件移植）。
2. 做 component erasure / suppression（组件擦除 / 抑制），观察 final readout fraction（最终读出阈值比例）是否下降。
3. 区分三种机制：
   a. value boost（值增强）
   b. competitor suppression（竞争者抑制）
   c. format / echo route suppression（格式 / 回声路线抑制）
4. 不以单个 logprob delta（对数概率增量）作为主要结论，而以 threshold fraction（阈值比例）和 token0 route change（第一个词元路线变化）作为核心指标。
```

阶段性判断：

```text
Phase 740 没有完成自然闭环，
但它把瓶颈从“读出端不够强”推进为“自然读出方向存在，但当前路径没有把它因果传递到最终阈值”。
```

## Phase 741: Threshold Candidate Causal Validation [2026-06-29 07:21]

### 问题来源

Phase 740 找到了 natural readout threshold source candidates（自然读出阈值来源候选），但那些结果仍然是 projection backtrace（投影回溯），不是严格因果证据。

因此 Phase 741 的问题是：

```text
这些 late component（后期组件）只是和目标方向相关，
还是它们的 donor-recipient delta（供体-受体差分）真的能推动 final readout threshold（最终读出阈值）？
```

### 脚本

```text
tests/gpt5/phase741_threshold_candidate_causal_validation.py
tests/gpt5/run_phase741_threshold_candidate_causal_validation_round.sh
```

### 命令

静态检查：

```bash
python -m py_compile tests/gpt5/phase741_threshold_candidate_causal_validation.py
bash -n tests/gpt5/run_phase741_threshold_candidate_causal_validation_round.sh
python tests/gpt5/phase741_threshold_candidate_causal_validation.py --round-name dry --max-pairs 1 --top-audits 2 --top-candidates 3 --dry-run
```

冒烟测试：

```bash
PHASE741_ROUND_NAME=smoke PHASE741_MAX_PAIRS=1 PHASE741_TOP_AUDITS=1 PHASE741_TOP_CANDIDATES=1 PHASE741_LOG_EVERY=1 bash tests/gpt5/run_phase741_threshold_candidate_causal_validation_round.sh
```

主测试：

```bash
PHASE741_ROUND_NAME=main PHASE741_MAX_PAIRS=6 PHASE741_TOP_AUDITS=2 PHASE741_TOP_CANDIDATES=3 PHASE741_LOG_EVERY=2 bash tests/gpt5/run_phase741_threshold_candidate_causal_validation_round.sh
```

确认测试：

```bash
PHASE741_ROUND_NAME=confirm PHASE741_MAX_PAIRS=10 PHASE741_TOP_AUDITS=2 PHASE741_TOP_CANDIDATES=3 PHASE741_LOG_EVERY=3 bash tests/gpt5/run_phase741_threshold_candidate_causal_validation_round.sh
```

脚本仍按 qwen3、GLM4、DS7B 顺序运行，并使用 `--hard-exit-after-model`。

### 测试原理

Phase 741 对 Phase 740 排名前三的 candidate components（候选组件）做四类粗粒度因果操作：

```text
recipient_add_donor_delta:
在 recipient（受体）自然路径中加入 donor - recipient component delta（供体-受体组件差分）。

joint_add_donor_delta:
在 Phase739/740 的 joint patch（联合修补路径）基础上，再加入 donor - recipient component delta。

joint_erase_to_recipient_component:
在 joint patch（联合修补路径）中，把该组件强制替换回 recipient component（受体组件）。

donor_erase_to_recipient_component:
在 donor（供体）自然路径中，把该组件替换成 recipient component（受体组件）。
```

核心指标仍然是：

```text
effect_vs_joint_fraction = projection(condition_final - joint_final, d) / threshold
effect_vs_donor_fraction = projection(condition_final - donor_final, d) / threshold
```

其中 `d` 是 Phase 739 / 740 使用的 donor-vs-current-top readout direction（供体对当前第一候选读出方向），`threshold` 是 Phase 739 的 token0 top1（第一个生成词元第一名）翻转阈值。

### 确认轮结果

结果目录：

```text
results/glm5_phase741_threshold_candidate_causal_validation/confirm/
```

跨模型摘要：

```text
results/glm5_phase741_threshold_candidate_causal_validation/confirm/phase741_cross_model_summary.md
```

确认轮共生成：

```text
qwen3: 360 rows
GLM4: 360 rows
DS7B: 360 rows
```

核心表：

| model | component | joint add effect | joint erase effect | donor erase effect | role |
|---|---|---:|---:|---:|---|
| qwen3 | L31:attn_out | 0.124 | 0.001 | -0.131 | causal_boost_candidate |
| qwen3 | L33:mlp_out | 0.019 | 0.001 | -0.038 | weak_boost_candidate |
| qwen3 | L34:attn_out | 0.164 | 0.001 | -0.230 | causal_boost_candidate |
| GLM4 | L37:mlp_out | 0.053 | -0.000 | -0.039 | causal_boost_candidate |
| GLM4 | L38:mlp_out | 0.671 | -0.028 | -0.678 | causal_boost_candidate |
| GLM4 | L39:mlp_out | 0.204 | -0.005 | -0.135 | causal_boost_candidate |
| DS7B | L26:attn_out | 0.310 | -0.005 | -0.320 | causal_boost_candidate |
| DS7B | L27:attn_out | 0.211 | -0.002 | -0.204 | causal_boost_candidate |
| DS7B | L27:mlp_out | 0.034 | -0.001 | -0.031 | weak_boost_candidate |

### 客观结论

Phase 741 的关键正结果：

```text
Phase 740 的候选组件不是纯相关信号；
其中多个组件的 donor-recipient delta（供体-受体差分）具有直接因果读出增强作用。
```

最强结果：

```text
GLM4 L38:mlp_out:
joint_add_donor_delta 平均增加约 0.671 个阈值比例，
target_top1_rate（目标第一名率）达到 0.20。

DS7B L26:attn_out:
joint_add_donor_delta 平均增加约 0.310 个阈值比例。

qwen3 L34:attn_out:
joint_add_donor_delta 平均增加约 0.164 个阈值比例。

qwen3 L31:attn_out:
joint_add_donor_delta 平均增加约 0.124 个阈值比例。
```

关键负结果：

```text
joint_erase_to_recipient_component 的下降很小。
```

这说明：

```text
当前 Phase739/740 的 joint patch（联合修补路径）并没有真正利用这些自然阈值组件；
这些组件更像是自然 donor path（供体路径）中存在的有效读出方向源，
但原 patch 路线没有把它们接入最终闭合路径。
```

### 理论进展

Phase 741 把当前图谱推进到：

```text
source writer（源写入器）
-> rewriter（重写器）
-> natural threshold component（自然阈值组件）
-> final readout threshold（最终读出阈值）
```

其中 natural threshold component（自然阈值组件）首次获得了粗粒度因果验证。

三个模型出现不同结构：

```text
qwen3:
主要是 late attention outputs（后期注意力输出）提供中等强度读出增强。

GLM4:
L38:mlp_out 是非常清晰的 readout amplifier（读出放大器）。

DS7B:
L26 / L27 attention outputs（注意力输出）是主要读出增强源，MLP 较弱。
```

这说明语言编码机制很可能不是单一模块：

```text
同一种功能闭合，在不同模型中可以由 attention carrier（注意力承载器）或 MLP amplifier（多层感知机放大器）实现。
```

### 问题和硬伤

1. Phase 741 是 component-output granularity（组件输出颗粒度），还不是 neuron-level（神经元级）证明。

2. donor delta add（供体差分加入）是人工插入，可能 off-manifold（离自然流形）。

3. 单个组件仍然没有普遍完成 top1 closure（第一名闭合）。GLM4 L38 最接近，但也只有 0.20 top1 rate。

4. joint erase（联合擦除）效应很小，说明当前 joint patch 与自然候选路径没有真正接上。

5. qwen3 / DS7B 的增强主要来自 attention output（注意力输出），但这不等于已经定位到具体 head / channel / neuron（注意力头 / 通道 / 神经元）。

6. 当前是小模型，具体层号不能外推为大模型通用结构。

### 下一步

Phase 742 仍属于同一个阶段，应该继续完成：

```text
Combined Threshold Component Closure
```

核心问题：

```text
单个自然阈值组件不能完全闭合，
那么多个 causally validated components（因果验证组件）组合后是否能跨过 threshold（阈值）？
```

测试设计：

```text
1. 对 Phase 741 的 top components（最高组件）做 top1 / top2 / top3 cumulative donor delta add（累计供体差分加入）。
2. 测量 final threshold fraction（最终阈值比例）是否接近或超过 1。
3. 测量 token0 target_top1_rate（第一个词元目标第一名率）是否明显提升。
4. 如果组合仍不能闭合，则说明缺失的是：
   a. 更早的路线选择机制；
   b. format / competitor suppression（格式 / 竞争者抑制）；
   c. final norm / readout geometry（最终归一化 / 读出几何）。
```

阶段性判断：

```text
Phase 741 已经证明自然阈值候选组件具备因果增强作用；
但还没有证明完整自然闭合。

所以当前最关键问题从“有没有因果读出组件”
推进为“这些组件组合后是否足以完成 token0 threshold closure”。
```

## Phase 742: Combined Threshold Component Closure [2026-06-29 07:29]

### 问题来源

Phase 741 证明多个 Phase 740 候选组件具备 causal boost（因果增强）作用，但单个组件大多不能完成 token0 top1 closure（第一个生成词元第一名闭合）。

因此 Phase 742 继续同一阶段的问题：

```text
如果把已经因果验证的组件组合起来，
是否足以跨过 final readout threshold（最终读出阈值）？
```

### 脚本

```text
tests/gpt5/phase742_combined_threshold_component_closure.py
tests/gpt5/run_phase742_combined_threshold_component_closure_round.sh
```

### 命令

静态检查：

```bash
python -m py_compile tests/gpt5/phase742_combined_threshold_component_closure.py
bash -n tests/gpt5/run_phase742_combined_threshold_component_closure_round.sh
python tests/gpt5/phase742_combined_threshold_component_closure.py --round-name dry --max-pairs 1 --top-audits 2 --top-candidates 3 --dry-run
```

冒烟测试：

```bash
PHASE742_ROUND_NAME=smoke PHASE742_MAX_PAIRS=1 PHASE742_TOP_AUDITS=1 PHASE742_TOP_CANDIDATES=3 PHASE742_LOG_EVERY=1 bash tests/gpt5/run_phase742_combined_threshold_component_closure_round.sh
```

主测试：

```bash
PHASE742_ROUND_NAME=main PHASE742_MAX_PAIRS=6 PHASE742_TOP_AUDITS=2 PHASE742_TOP_CANDIDATES=3 PHASE742_LOG_EVERY=2 bash tests/gpt5/run_phase742_combined_threshold_component_closure_round.sh
```

确认测试：

```bash
PHASE742_ROUND_NAME=confirm PHASE742_MAX_PAIRS=10 PHASE742_TOP_AUDITS=2 PHASE742_TOP_CANDIDATES=3 PHASE742_LOG_EVERY=3 bash tests/gpt5/run_phase742_combined_threshold_component_closure_round.sh
```

脚本仍按 qwen3、GLM4、DS7B 顺序运行，并使用 `--hard-exit-after-model`。

### 测试原理

Phase 742 不再扩大候选空间，而是读取 Phase 741 的确认轮结果，按 `joint_add_donor_delta`（联合路径加入供体差分）的因果强度排序。

排序结果：

```text
qwen3:
L34:attn_out -> L31:attn_out -> L33:mlp_out

GLM4:
L38:mlp_out -> L39:mlp_out -> L37:mlp_out

DS7B:
L26:attn_out -> L27:attn_out -> L27:mlp_out
```

然后测试：

```text
joint_add_top1:
在 joint patch（联合修补路径）上加入最强组件 donor-recipient delta（供体-受体差分）。

joint_add_top2:
累计加入前两个组件差分。

joint_add_top3:
累计加入前三个组件差分。
```

核心指标：

```text
fraction = projection(condition_final - recipient_final, d) / threshold
target_top1_rate = 目标答案 token 在 token0 的 top1 比例
margin_donor_vs_top = 目标答案与当前 top token 的 logit margin（对数几率差）
```

### 确认轮结果

结果目录：

```text
results/glm5_phase742_combined_threshold_component_closure/confirm/
```

跨模型摘要：

```text
results/glm5_phase742_combined_threshold_component_closure/confirm/phase742_cross_model_summary.md
```

确认轮共生成：

```text
qwen3: 200 rows
GLM4: 200 rows
DS7B: 200 rows
```

核心结果：

| model | condition | components | fraction | joint add effect | target top1 rate | margin donor vs top |
|---|---|---|---:|---:|---:|---:|
| qwen3 | joint_base | - | 0.004 | 0.000 | 0.000 | -15.250 |
| qwen3 | joint_add_top1 | L34:attn_out | 0.167 | 0.164 | 0.000 | -11.541 |
| qwen3 | joint_add_top2 | L34:attn_out,L31:attn_out | 0.279 | 0.275 | 0.000 | -8.316 |
| qwen3 | joint_add_top3 | L34:attn_out,L31:attn_out,L33:mlp_out | 0.312 | 0.309 | 0.000 | -7.534 |
| GLM4 | joint_base | - | 0.029 | 0.000 | 0.000 | -7.825 |
| GLM4 | joint_add_top1 | L38:mlp_out | 0.700 | 0.671 | 0.200 | -1.678 |
| GLM4 | joint_add_top2 | L38:mlp_out,L39:mlp_out | 0.877 | 0.848 | 0.300 | -1.116 |
| GLM4 | joint_add_top3 | L38:mlp_out,L39:mlp_out,L37:mlp_out | 0.931 | 0.901 | 0.500 | -0.706 |
| DS7B | joint_base | - | 0.018 | 0.000 | 0.000 | -9.863 |
| DS7B | joint_add_top1 | L26:attn_out | 0.328 | 0.310 | 0.000 | -5.125 |
| DS7B | joint_add_top2 | L26:attn_out,L27:attn_out | 0.553 | 0.536 | 0.100 | -2.513 |
| DS7B | joint_add_top3 | L26:attn_out,L27:attn_out,L27:mlp_out | 0.587 | 0.569 | 0.050 | -2.356 |

### 客观结论

Phase 742 得到一个强收束结果：

```text
因果组件组合可以显著推动 readout threshold（读出阈值），
但三模型闭合程度不同。
```

具体：

```text
qwen3:
top3 组合只达到约 0.312 个阈值比例，target_top1_rate = 0。
说明已知组件只解释约三分之一阈值缺口。

GLM4:
top3 组合达到约 0.931 个阈值比例，target_top1_rate = 0.5。
说明 GLM4 的自然阈值组件图谱已经非常接近 token0 top1 closure。

DS7B:
top3 组合达到约 0.587 个阈值比例，target_top1_rate = 0.05。
说明 DS7B 的 L26/L27 attention（注意力）组合有明显推进，但还远未稳定闭合。
```

### 理论进展

Phase 742 把当前机制图谱推进为：

```text
source writer（源写入器）
-> rewriter（重写器）
-> natural threshold components（自然阈值组件群）
-> cumulative readout force（累计读出力）
-> token0 competition（第一个词元竞争）
```

新的关键拼图：

```text
读出闭合不是单组件事件，
而是多个自然阈值组件的累计效应。
```

但 Phase 742 同时说明：

```text
组件累计读出力 和 最终 token0 top1 闭合 不是同一件事。
```

GLM4 的 top3 组件已经接近 1.0 阈值比例，但仍只有 50% top1，说明除了 value boost（值增强），还存在：

```text
competitor suppression（竞争者抑制）
format suppression（格式抑制）
final norm / readout geometry（最终归一化 / 读出几何）
```

### 问题和硬伤

1. Phase 742 使用的是 whole-component cumulative delta（整组件累计差分），还不是神经元级机制。

2. top3 组合仍然是人工加入，不等于自然路径自动产生。

3. qwen3 和 DS7B 未闭合，说明当前图谱缺少关键机制。

4. GLM4 虽然接近闭合，但仍有 50% 未 top1，说明读出阈值比例接近 1 不等于稳定生成闭合。

5. 当前仍未拆出 competitor token（竞争词元）为什么仍然占优。

6. 当前模型是小模型，具体层号和组件分工不能直接外推。

### 阶段性判断

从 Phase 739 到 Phase 742，当前阶段已经完成以下闭环：

```text
Phase 739:
人工 readout boost 可以闭合，瓶颈是阈值跨越。

Phase 740:
自然 donor path 中存在足够强的 readout direction（读出方向），但当前 patch 只传递很小比例。

Phase 741:
多个自然阈值组件具有因果增强作用。

Phase 742:
组件组合能显著接近闭合，其中 GLM4 已接近闭合，但 qwen3 / DS7B 仍不足。
```

因此当前大阶段已经达到一个自然收束点：

```text
已经确认“自然阈值组件群”是语言值输出闭合的重要机制拼图；
但完整闭合还需要解释 competitor / format / readout geometry（三类竞争和读出几何问题）。
```

### 下一步

下一阶段不应继续盲目增加 component add（组件加入），而应该转向：

```text
Phase 743: Competitor and Format Suppression Audit
```

核心问题：

```text
为什么 readout fraction（读出阈值比例）已经接近 1，
但 token0 top1 仍不稳定？
```

建议测试：

```text
1. 对 GLM4 top3 near-closure cases（接近闭合样本）审计当前 top token 类型。
2. 区分 competitor（竞争者）：
   a. wrong semantic value（错误语义值）
   b. format token（格式词元）
   c. echo token（回声词元）
   d. generic noun / category token（泛化名词 / 类别词）
3. 测量这些 competitor 的 logit 来源：
   late MLP（后期多层感知机）
   attention output（注意力输出）
   final norm（最终归一化）
   unembedding geometry（反嵌入几何）
4. 只在必要时做 suppression intervention（抑制干预），不要继续盲目增强 donor answer。
```

阶段性结论：

```text
Phase 742 证明：
自然阈值组件群可以把答案推到读出闭合边缘；
真正剩下的瓶颈不是“没有答案方向”，而是“答案方向和竞争/格式路线之间的最后选择机制”。
```

## Phase 743: 竞争者与格式抑制审计 [2026-06-29 08:15]

### 背景和外部分析判断

本阶段分析了用户提供的 Phase 740-742 总结。该总结总体正确：Phase 740-742 不是普通 patch（修补）堆叠，而是把瓶颈从 writer / rewriter（写入器 / 重写器）后移到 token0 competition（第一个词元竞争）。更准确的当前判断是：

```text
自然 donor path（供体路径）中存在足够的 answer readout direction（答案读出方向）。
Phase 741/742 找到的自然阈值组件具有因果增强作用。
但是 target answer（目标答案）是否成为 top1（第一名），还取决于 recipient answer（受体答案）、format token（格式词元）、echo token（回声词元）、punctuation / prose route（标点 / 散文路线）是否被压制。
```

因此本阶段不继续盲目加入更多组件，而是审计：

```text
joint + topK natural threshold components（联合路径 + 前K个自然阈值组件）之后，
当前 top token（最高词元）到底是谁？
属于哪一类竞争路线？
如果只抑制当前 top competitor（最高竞争者），donor answer 是否能闭合？
```

### 生成脚本

```text
tests/gpt5/phase743_competitor_format_suppression_audit.py
tests/gpt5/run_phase743_competitor_format_suppression_audit_round.sh
```

输出目录：

```text
results/glm5_phase743_competitor_format_suppression_audit/
```

关键输出：

```text
results/glm5_phase743_competitor_format_suppression_audit/confirm/phase743_cross_model_summary.md
results/glm5_phase743_competitor_format_suppression_audit/confirm/phase743_cross_model_summary.json
results/glm5_phase743_competitor_format_suppression_audit/confirm/phase743_atlas_graph.json
```

### 执行命令

冒烟测试：

```bash
tests/gpt5/run_phase743_competitor_format_suppression_audit_round.sh smoke --max-pairs 1 --top-audits 1 --top-candidates 2 --top-k-vocab 8 --suppress-scales 1.0 1.25 --log-every 1
```

主测试：

```bash
tests/gpt5/run_phase743_competitor_format_suppression_audit_round.sh main --max-pairs 6 --top-audits 2 --top-candidates 3 --top-k-vocab 12 --suppress-scales 1.0 1.25 --log-every 2
```

确认测试：

```bash
tests/gpt5/run_phase743_competitor_format_suppression_audit_round.sh confirm --max-pairs 10 --top-audits 2 --top-candidates 3 --top-k-vocab 12 --suppress-scales 1.0 1.25 --log-every 2
```

三模型按 qwen3、GLM4、DS7B 顺序执行，并使用：

```text
--hard-exit-after-model
```

模型加载采用 BF16（bfloat16）非量化方案，复用前面阶段的 hooks（钩子）和 component replacement（组件替换）框架。

### 测试原理

Phase 742 已经得到：

```text
h_combo = h_joint + sum(topK donor-recipient component delta)
```

本阶段在该状态上读取全词表 top-k token，并分类为：

```text
donor_answer
recipient_answer
other_semantic_value
format_or_schema
echo_object_or_relation
punctuation_or_stop
prose_prefix
other_vocab
```

对于当前最高竞争词元 c，定义竞争方向：

```text
d_c = normalize(W_U(c) - W_U(y_donor))
```

当前竞争缺口为：

```text
gap_c = logit(c) - logit(y_donor)
```

最小抑制量近似为：

```text
alpha_c = gap_c / dot(W_U(c) - W_U(y_donor), d_c)
```

然后在 final norm output（最终归一化输出）处做：

```text
h' = h - scale * alpha_c * d_c
```

scale 取：

```text
1.0, 1.25
```

这不是自然机制证明，而是读出几何审计：如果压掉当前最高竞争者后 donor answer 仍不稳定，说明失败不是单个 competitor（竞争者）造成，而是多竞争路线或全局读出几何造成。

### 确认轮客观结果

#### qwen3

```text
joint_add_topK:
  n = 20
  donor_top1_rate = 0.000
  mean_donor_rank = 14.45
  mean_margin_donor_vs_top = -7.534
  top_token_class = recipient_answer: 20/20

suppress_current_top scale=1.0:
  donor_top1_rate = 0.100
  mean_donor_rank = 2.70
  top_token_class:
    format_or_schema = 16
    recipient_answer = 3
    donor_answer = 1

suppress_current_top scale=1.25:
  donor_top1_rate = 0.300
  mean_donor_rank = 2.15
  top_token_class:
    format_or_schema = 13
    donor_answer = 7
```

解释：

```text
qwen3 的 top3 阈值组件组合后，首先完全输给 recipient answer（受体答案）。
但压掉 recipient answer 后，format_or_schema（格式 / 模板路线）大面积接管。
所以 qwen3 不是单一 recipient competition（受体竞争）问题，而是 recipient + format 多路线竞争问题。
```

#### GLM4

```text
joint_add_topK:
  n = 20
  donor_top1_rate = 0.500
  mean_donor_rank = 2.75
  mean_margin_donor_vs_top = -0.706
  top_token_class:
    donor_answer = 10
    echo_object_or_relation = 6
    other_vocab = 4

suppress_current_top scale=1.25:
  n = 9
  donor_top1_rate = 0.667
  mean_donor_rank = 1.22
  top_token_class:
    donor_answer = 6
    echo_object_or_relation = 2
    other_vocab = 1
```

解释：

```text
GLM4 已经最接近闭合。
失败样本主要不是 recipient answer，而是 echo_object_or_relation（回声 / 对象 / 关系）和 other_vocab 中的 "B" 路线。
对失败样本压制当前 top competitor 后，有 66.7% 转为 donor top1。
这说明 GLM4 的剩余瓶颈主要是局部竞争者抑制不足，而不是缺少大量 donor readout force（供体读出力）。
```

#### DS7B

```text
joint_add_topK:
  n = 20
  donor_top1_rate = 0.050
  mean_donor_rank = 9.30
  mean_margin_donor_vs_top = -2.356
  top_token_class:
    format_or_schema = 9
    echo_object_or_relation = 7
    punctuation_or_stop = 1
    recipient_answer = 1
    other_vocab = 1
    donor_answer = 1

suppress_current_top scale=1.25:
  n = 18
  donor_top1_rate = 0.500
  mean_donor_rank = 2.06
  top_token_class:
    donor_answer = 9
    echo_object_or_relation = 3
    format_or_schema = 2
    punctuation_or_stop = 2
    other_semantic_value = 1
    other_vocab = 1
```

解释：

```text
DS7B 的主要阻挡不是 recipient answer。
最大竞争路线是 format_or_schema（尤其 "The"）和 echo_object_or_relation（category / taste / carrot / stone）。
单独压制当前最高竞争者后，donor_top1_rate 从 0.05 提升到 0.50，但仍有格式、回声、标点路线接管。
所以 DS7B 是多竞争吸引子问题，尤其是格式 / 回声路线控制不足。
```

### 总体结论

Phase 743 支持并收紧 Phase 740-742 的判断：

```text
答案方向存在；
自然阈值组件有因果增强作用；
但生成闭合还必须包含 competitor suppression（竞争者抑制）。
```

三模型分化清楚：

```text
qwen3:
  recipient answer competition 是第一阻挡；
  recipient 被压下后 format route 立刻接管。

GLM4:
  donor force 已较强；
  剩余失败多是 echo / relation / other_vocab 局部竞争；
  最接近自然闭合。

DS7B:
  format route 和 echo route 是主要阻挡；
  抑制当前 top competitor 可显著提升，但仍不稳定。
```

因此，输出闭合不只是：

```text
Boost donor answer
```

而是：

```text
Boost donor answer
+ Suppress recipient answer
+ Suppress format route
+ Suppress echo route
+ Suppress punctuation / prose continuation
```

### 关键进展

本阶段第一次把 Phase 742 的“fraction 接近阈值但 top1 不稳定”拆成可观测竞争类别：

```text
qwen3: recipient_answer -> format_or_schema
GLM4: echo_object_or_relation / other_vocab
DS7B: format_or_schema + echo_object_or_relation
```

这说明：

```text
readout fraction（读出阈值比例）只解释 donor force（供体力）；
token0 closure（第一个词元闭合）还需要 competitor field（竞争场）的压制结构。
```

### 问题、硬伤和边界

1. 本阶段的 suppression（抑制）仍是在 final norm output（最终归一化输出）上的人工读出几何干预，不证明模型自然内部有同样的 suppression circuit（抑制回路）。

2. 当前只压制“当前 top competitor”。如果压掉后另一个 format / echo token 接管，说明竞争场是多峰结构，不能用单竞争者解释。

3. 分类规则仍是启发式。比如 GLM4 的 `" B"` 暂记为 other_vocab，但它可能是 option / format route（选项 / 格式路线），后续需要细分。

4. 当前仍是 whole component（整组件）级路径，未进入 head/channel/neuron（注意力头 / 通道 / 神经元）级 suppression atlas（抑制图谱）。

5. 小模型结果不能直接外推层号和组件形态，只能外推功能结构：

```text
writer -> rewriter -> threshold components -> competitor suppression -> token0 closure
```

### 理论更新

当前完整路径应从：

```text
source -> writer -> rewriter -> threshold components -> readout boost
```

更新为：

```text
source -> writer -> rewriter -> threshold components
-> donor readout force
-> competitor suppression field
-> token0 closure
-> continuation closure
```

最小闭合公式：

```text
Closure = donor_force - max(competitor_force) > 0
```

更完整地写：

```text
logit(y_donor)
-
max_c logit(c)
> 0
```

其中：

```text
logit(y_donor) = W_U(y_donor)^T h_final
logit(c)       = W_U(c)^T h_final
```

而：

```text
h_final
= h_base
+ source_writer
+ mlp_rewriter
+ threshold_components
+ suppression_components
+ residual_noise
```

Phase 743 的核心新项是：

```text
suppression_components
```

没有这个项，读出增强可以接近阈值，但不一定稳定生成。

### 下一步

下一阶段应进入：

```text
Phase 744: Competitor Suppression Source Localization
竞争者抑制来源定位
```

核心目标：

```text
不再只在 final norm output 人工压制 competitor，
而是定位哪些 head / MLP / channel 自然负责压制：
1. recipient_answer
2. format_or_schema
3. echo_object_or_relation
4. punctuation_or_stop
```

建议优先顺序：

```text
1. qwen3:
   找 recipient_answer suppression source。

2. DS7B:
   找 "The" format route 和 category/taste echo route 的抑制来源。

3. GLM4:
   找 echo_object_or_relation 和 "B" route 的局部抑制来源。
```

成功判据：

```text
找到至少一种自然 suppression component（抑制组件），其 donor-recipient delta 能降低 competitor logit 或提升 donor-vs-competitor margin。
```

## Phase 744: 竞争者抑制来源定位 [2026-06-29 08:36]

### 背景和判断

本阶段分析了用户提供的 Phase 743 复盘。该复盘总体正确：Phase 743 的关键价值不是继续增强 donor answer（供体答案），而是把输出闭合改写为：

```text
donor boost（供体增强）
+
competitor suppression（竞争者抑制）
=
token0 closure（第一个词元闭合）
```

Phase 743 已经证明：

```text
qwen3:
  第一阻挡是 recipient_answer（受体答案），压掉后 format route（格式路线）接管。

GLM4:
  已经最接近闭合，剩余主要是 echo / other_vocab 竞争。

DS7B:
  主要是 format route + echo route 多吸引子竞争。
```

因此 Phase 744 的目标是继续同一阶段性任务：不再只问“要压谁”，而是定位“自然路径中哪些 component（组件）能压制这些竞争者，或至少提高 donor-vs-competitor margin（供体相对竞争者边际）”。

### 生成脚本

```text
tests/gpt5/phase744_competitor_suppression_source_localization.py
tests/gpt5/run_phase744_competitor_suppression_source_localization_round.sh
```

输出目录：

```text
results/glm5_phase744_competitor_suppression_source_localization/
```

关键输出：

```text
results/glm5_phase744_competitor_suppression_source_localization/confirm/phase744_cross_model_summary.md
results/glm5_phase744_competitor_suppression_source_localization/confirm/phase744_cross_model_summary.json
results/glm5_phase744_competitor_suppression_source_localization/confirm/phase744_atlas_graph.json
```

### 执行命令

冒烟测试：

```bash
tests/gpt5/run_phase744_competitor_suppression_source_localization_round.sh smoke --max-pairs 1 --top-audits 1 --top-candidates 2 --top-k-vocab 8 --max-scan-candidates 6 --log-every 1
```

主测试：

```bash
tests/gpt5/run_phase744_competitor_suppression_source_localization_round.sh main --max-pairs 6 --top-audits 2 --top-candidates 3 --top-k-vocab 12 --log-every 2
```

确认测试：

```bash
tests/gpt5/run_phase744_competitor_suppression_source_localization_round.sh confirm --max-pairs 10 --top-audits 2 --top-candidates 3 --top-k-vocab 12 --log-every 2
```

三模型按 qwen3、GLM4、DS7B 顺序执行，并使用：

```text
--hard-exit-after-model
```

加载方式：

```text
BF16（bfloat16）
非量化
attn=eager
device_map=auto
```

### 测试原理

Phase 743 在下列状态上得到当前最高竞争者：

```text
h_combo = h_joint + sum(topK donor-recipient threshold component delta)
```

Phase 744 固定该状态下的当前 top competitor（最高竞争者）c，然后扫描 late attn / mlp component（后期注意力 / 多层感知机组件），把自然 donor-recipient delta 加入 recipient combo state：

```text
h' = h_combo + delta_component(donor - recipient)
```

核心指标：

```text
base_margin = logit(y_donor) - logit(c)
new_margin  = logit'(y_donor) - logit'(c)
delta_margin = new_margin - base_margin

delta_donor_logit = logit'(y_donor) - logit(y_donor)
delta_comp_logit  = logit'(c) - logit(c)
```

解释规则：

```text
delta_margin > 0:
  该组件能改善 donor-vs-current-competitor competition（供体对当前竞争者竞争）。

delta_comp_logit < 0:
  有直接 competitor suppression（竞争者抑制）证据。

delta_donor_logit > 0 且 delta_comp_logit 不明显下降:
  主要是 donor boost（供体增强），不是纯抑制。

new_donor_top1_rate 上升:
  该组件能把部分失败样本推向 token0 closure（第一个词元闭合）。
```

### 确认轮结果

#### qwen3

确认轮：

```text
n = 20
base competitor class = recipient_answer: 20/20
```

主要候选：

```text
L33:attn_out:
  delta_margin = +5.144
  delta_donor_logit = +3.038
  delta_competitor_logit = -2.106
  new_donor_top1_rate = 0.000
  解释：强烈压低 recipient answer，同时增强 donor，但 format route 接管，所以不闭合。

L32:attn_out:
  delta_margin = +3.259
  delta_donor_logit = +2.959
  delta_competitor_logit = -0.300
  new_donor_top1_rate = 0.100
  解释：混合 boost + suppression，并能少量闭合。

L30:attn_out:
  delta_margin = +1.991
  delta_donor_logit = +1.428
  delta_competitor_logit = -0.562
  new_donor_top1_rate = 0.000

L32:mlp_out:
  delta_margin = +0.975
  delta_donor_logit = +0.175
  delta_competitor_logit = -0.800
  new_donor_top1_rate = 0.000
  解释：这是更接近 pure recipient suppression（纯受体抑制）的候选，但强度不足。
```

结论：

```text
qwen3 的 recipient_answer suppression source（受体答案抑制来源）候选成立。
最强的是 L33:attn_out，但它压掉 recipient 后引出 format route。
L32:mlp_out 是较纯的 recipient suppression candidate（受体抑制候选），但不能单独闭合。
```

#### GLM4

确认轮中 GLM4 有 10 个 case 已经 donor top1，被跳过；剩余 10 个失败 case 用于来源定位。

主要候选：

```text
L34:attn_out:
  delta_margin = +2.306
  delta_donor_logit = +2.131
  delta_competitor_logit = -0.175
  new_donor_top1_rate = 0.800
  解释：最强闭合候选，主要是 donor boost，附带轻度 competitor suppression。

L35:attn_out:
  delta_margin = +0.794
  delta_donor_logit = +0.581
  delta_competitor_logit = -0.212
  new_donor_top1_rate = 0.400

L35:mlp_out:
  delta_margin = +0.231
  delta_donor_logit = +0.188
  delta_competitor_logit = -0.044
  new_donor_top1_rate = 0.400

L36:attn_out:
  delta_margin = +0.475
  delta_donor_logit = +0.412
  delta_competitor_logit = -0.062
  new_donor_top1_rate = 0.300
```

按竞争类别：

```text
other_vocab:
  L34:attn_out 可达 donor_top1_rate = 1.000，但主要是 donor boost。
  L34:mlp_out 和 L39:attn_out 对 other_vocab 有较纯 suppression 迹象：
    L34:mlp_out: competitor delta = -0.203, donor_top1_rate = 0.500
    L39:attn_out: competitor delta = -0.281, donor_top1_rate = 0.500

echo_object_or_relation:
  L34:attn_out:
    delta_margin = +2.885
    delta_competitor_logit = -0.323
    donor_top1_rate = 0.667
```

结论：

```text
GLM4 的失败样本可以被 L34/L35 attention route 显著修复。
但 L34:attn_out 多数是 boost-dominant（增强主导），不是纯抑制。
较纯的 suppression 候选在 L34:mlp_out / L39:attn_out，但强度较小。
```

#### DS7B

确认轮中 1 个 case 已经 donor top1，被跳过，剩余 19 个失败 case。

主要候选：

```text
L23:attn_out:
  delta_margin = +1.977
  delta_donor_logit = +1.671
  delta_competitor_logit = -0.306
  new_donor_top1_rate = 0.421
  解释：最强 boost + suppression 混合候选。

L22:attn_out:
  delta_margin = +1.688
  delta_donor_logit = +1.303
  delta_competitor_logit = -0.385
  new_donor_top1_rate = 0.316

L24:attn_out:
  delta_margin = +1.118
  delta_donor_logit = +0.993
  delta_competitor_logit = -0.125
  new_donor_top1_rate = 0.263

L24:mlp_out:
  delta_margin = +0.868
  delta_donor_logit = +0.753
  delta_competitor_logit = -0.115
  new_donor_top1_rate = 0.105

L25:mlp_out:
  delta_margin = +0.658
  delta_donor_logit = +0.401
  delta_competitor_logit = -0.257
  new_donor_top1_rate = 0.105
```

结论：

```text
DS7B 的 L22-L24 attention 是主要竞争边际改善来源。
L23:attn_out 最强，可把失败 case 的 donor_top1_rate 推到 0.421。
这些组件不是纯 suppression，而是 donor boost + competitor suppression 混合。
```

### 总体结论

Phase 744 支持 Phase 743 的路线，并给出新的客观拼图：

```text
1. 自然 donor-recipient component delta 中确实存在能改善 donor-vs-competitor margin 的来源。
2. 这些来源多数不是纯 suppression，而是 boost + suppression 混合组件。
3. 纯 suppression 候选存在，但通常强度较弱，单独不能闭合。
4. GLM4 最接近闭合，L34:attn_out 对失败样本有 0.800 top1 修复率。
5. DS7B 的 L23:attn_out 是最强混合候选，能把失败样本 donor_top1_rate 推到 0.421。
6. qwen3 虽然 L33:attn_out 强烈压低 recipient answer，但 format route 接管，说明 qwen3 的瓶颈是 recipient + format 双层竞争。
```

### 理论更新

Phase 743 的公式是：

```text
ClosureForce = Boost_donor + Suppress_competitors
```

Phase 744 进一步说明，实际组件通常不是纯分离的：

```text
component_effect(u)
= donor_boost(u)
+ competitor_suppression(u)
+ route_shift(u)
```

其中：

```text
donor_boost(u) = delta logit(y_donor)
competitor_suppression(u) = - delta logit(c)
route_shift(u) = top competitor 从一种 route 切换到另一种 route
```

所以更接近当前实证结果的闭合公式是：

```text
delta_margin(u, c)
= delta logit(y_donor)
- delta logit(c)
```

而稳定闭合需要：

```text
sum_u delta_margin(u, c_i) > gap(c_i)
for all dominant competitor routes c_i
```

这说明：

```text
语言生成闭合不是单一方向增强，
而是多个竞争路线同时被重新排序。
```

### 问题和硬伤

1. Phase 744 仍是 whole-component（整组件）级别，尚未定位到 head / channel / neuron（注意力头 / 通道 / 神经元）。

2. donor-recipient delta add（供体-受体差分加入）仍是人工 transplant（移植），不等于证明自然生成中这些组件会自动被激活。

3. 多数候选是 boost + suppression 混合，不是干净的 suppression circuit（抑制回路）。

4. qwen3 的强 recipient suppression 会引出 format route，说明单类竞争者定位仍不够，必须进入 route-level multi-suppression（路线级多竞争抑制）。

5. GLM4 的 `"B"` 仍归为 other_vocab，但它可能是格式/选项路线，需要更细 route classifier（路线分类器）。

6. 小模型层号不可外推到大模型，只能外推功能结构：

```text
late component delta can jointly boost donor and suppress competitors.
```

### 下一步

下一阶段应进入：

```text
Phase 745: Route-Level Multi-Competitor Suppression Validation
路线级多竞争者抑制验证
```

核心目标：

```text
不再只针对 current top competitor（当前最高竞争者），
而是同时处理一个 route class（路线类）：
recipient route
format route
echo route
other_vocab / option route
```

建议优先测试：

```text
qwen3:
  L33:attn_out 压 recipient 后 format 接管，下一步要组合 L33 recipient suppression + format route suppression。

GLM4:
  L34:attn_out 可强闭合，但 boost 主导；需要找 L34/L39 的 pure suppression 子来源。

DS7B:
  L23:attn_out / L22:attn_out 是核心混合候选，下一步验证它们是否同时处理 format + echo route。
```

成功判据：

```text
1. 对每个模型至少确定一个 route-level suppressor set（路线级抑制集合）。
2. 证明它比单 top competitor suppression 更稳定提升 donor_top1_rate。
3. 对 qwen3，必须解释 recipient 被压制后 format 接管的问题。
4. 对 GLM4，区分 boost-dominant closure 与 true suppression closure。
5. 对 DS7B，验证 L23/L22 attention 是否是 format+echo mixed suppressor。
```

## Phase 745: 路线级多竞争者抑制验证 [2026-06-29 09:18]

### 任务来源

用户提供的 Phase 744 复盘判断基本正确：

```text
Phase 744 证明了 donor answer 增强之后仍可能失败，
因为 recipient / format / echo / punctuation / other_vocab 等竞争路线会接管。
```

因此本阶段继续同一大阶段目标，进入：

```text
Route-Level Multi-Competitor Suppression Validation
路线级多竞争者抑制验证
```

核心问题：

```text
单独压制 current top competitor 是否足够？
还是必须同时压制多个 route class，donor answer 才能稳定 token0 top1？
```

### 新增脚本

```text
tests/gpt5/phase745_route_level_multi_competitor_suppression.py
tests/gpt5/run_phase745_route_level_multi_competitor_suppression_round.sh
```

### 输出目录

```text
results/glm5_phase745_route_level_multi_competitor_suppression/
```

确认轮核心文件：

```text
results/glm5_phase745_route_level_multi_competitor_suppression/confirm/phase745_cross_model_summary.md
results/glm5_phase745_route_level_multi_competitor_suppression/confirm/phase745_cross_model_summary.json
results/glm5_phase745_route_level_multi_competitor_suppression/confirm/phase745_atlas_graph.json
```

### 执行命令

冒烟测试：

```bash
tests/gpt5/run_phase745_route_level_multi_competitor_suppression_round.sh smoke --max-pairs 1 --top-audits 1 --top-candidates 2 --top-k-vocab 8 --max-route-classes 3 --max-topk-tokens 5 --suppress-scales 1.0 1.25 --log-every 1
```

主测试：

```bash
tests/gpt5/run_phase745_route_level_multi_competitor_suppression_round.sh main --max-pairs 8 --top-audits 2 --top-candidates 3 --top-k-vocab 12 --max-route-classes 5 --max-topk-tokens 8 --suppress-scales 1.0 1.25 --log-every 2
```

确认测试：

```bash
tests/gpt5/run_phase745_route_level_multi_competitor_suppression_round.sh confirm --max-pairs 10 --top-audits 2 --top-candidates 3 --top-k-vocab 12 --max-route-classes 5 --max-topk-tokens 8 --suppress-scales 1.0 1.25 --log-every 2
```

三模型均按顺序执行：

```text
qwen3 -> GLM4 -> DS7B
```

每个模型命令均带：

```text
--hard-exit-after-model
```

加载方式：

```text
BF16
quantization = off
attn_implementation = eager
```

### 测试原理

Phase 745 复用 Phase 743 / Phase 744 的 near-closure state：

```text
h_combo = h_joint + topK threshold component deltas
```

然后读取 top-k vocab，将非 donor token 分类为：

```text
recipient_answer
format_or_schema
echo_object_or_relation
punctuation_or_stop
prose_prefix
other_semantic_value
other_vocab
```

本阶段不只压当前最高竞争 token，而是比较五类条件：

```text
1. joint_add_topK
   不额外抑制，作为基线。

2. suppress_current_top
   只压当前 top competitor。

3. suppress_current_top_class
   压当前 top competitor 所在类。

4. suppress_route_representatives
   每个竞争 route class 取一个代表 token 同时压制。

5. suppress_route_centroids
   每个 route class 构造 centroid direction，同时压制。

6. suppress_all_topk_competitors
   压 top-k 内全部非 donor 竞争 token。
```

单 token 抑制方向：

```text
d_c = normalize(W_U(c) - W_U(y_donor))
```

需要抑制量：

```text
alpha_c = (logit(c) - logit(y_donor)) / dot(W_U(c) - W_U(y_donor), d_c)
```

路线 centroid 方向：

```text
d_R = normalize(mean_{c in R} W_U(c) - W_U(y_donor))
```

最终测试：

```text
h_final' = h_final - scale * sum(alpha_i * d_i)
```

观测：

```text
donor_top1_rate
mean_donor_rank
top_token_class_counts
route_shift_rate
margin_gain_vs_base_top
```

### 确认轮结果

#### qwen3

基线：

```text
joint_add_topK donor_top1_rate = 0.000
base top class = recipient_answer: 20/20
```

单 top 压制：

```text
suppress_current_top scale=1.00:
  donor_top1_rate = 0.100
  new top classes = donor 1, format 16, recipient 3

suppress_current_top scale=1.25:
  donor_top1_rate = 0.300
  new top classes = donor 7, format 13
```

多路线压制：

```text
suppress_route_representatives scale=1.00:
  donor_top1_rate = 1.000

suppress_route_centroids scale=1.00:
  donor_top1_rate = 1.000

suppress_all_topk_competitors scale=1.00:
  donor_top1_rate = 1.000
```

解释：

```text
qwen3 明确不是单 recipient_answer 竞争。
压掉 recipient 后 format route 接管。
同时压 route representatives / route centroids 后 donor 才稳定闭合。
```

#### GLM4

基线：

```text
joint_add_topK donor_top1_rate = 0.500
base top classes = donor 10, echo 6, other_vocab 4
```

单 top 压制：

```text
suppress_current_top scale=1.00:
  donor_top1_rate = 0.500

suppress_current_top scale=1.25:
  donor_top1_rate = 0.800
```

多路线压制：

```text
suppress_route_representatives scale=1.00:
  donor_top1_rate = 0.800

suppress_route_representatives scale=1.25:
  donor_top1_rate = 0.900

suppress_route_centroids scale=1.25:
  donor_top1_rate = 0.900

suppress_all_topk_competitors scale=1.25:
  donor_top1_rate = 0.900
```

解释：

```text
GLM4 本来已经半闭合。
路线级抑制能把 donor_top1_rate 从 0.5 推到 0.9。
剩余失败主要仍在 other_vocab，说明 other_vocab 可能包含更细的 option / label / format 子路线。
```

#### DS7B

基线：

```text
joint_add_topK donor_top1_rate = 0.050
base top classes = format 9, echo 7, donor 1, punctuation 1, recipient 1, other_vocab 1
```

单 top 压制：

```text
suppress_current_top scale=1.00:
  donor_top1_rate = 0.150

suppress_current_top scale=1.25:
  donor_top1_rate = 0.500
```

多路线压制：

```text
suppress_route_representatives scale=1.00:
  donor_top1_rate = 0.850

suppress_route_representatives scale=1.25:
  donor_top1_rate = 0.900

suppress_route_centroids scale=1.25:
  donor_top1_rate = 0.900

suppress_all_topk_competitors scale=1.25:
  donor_top1_rate = 0.900
```

解释：

```text
DS7B 是典型多路线竞争场。
单 top 压制只能达到 0.5，压掉一个路线后 echo / format / punctuation 会接管。
路线级压制后能达到 0.9，说明 DS7B 的失败主要来自多个竞争 route 同时占优。
```

### 核心客观结论

1. Phase 745 强力支持 Phase 744 的判断：

```text
token0 closure 不是单 donor boost 问题，
也不是单 top competitor suppression 问题，
而是 route-level competition field reordering。
```

2. qwen3：

```text
recipient_answer 是第一阻塞项，
format_or_schema 是第二阻塞项。
单压 recipient 会让 format 接管。
```

3. GLM4：

```text
已经接近闭合；
路线级抑制从 0.5 提升到 0.9；
剩余 other_vocab 需要更细分类。
```

4. DS7B：

```text
format + echo 是主竞争场；
单 top 压制不足；
路线级压制大幅提升到 0.85-0.90。
```

5. route centroid 和 route representative 的效果非常接近：

```text
说明 top-k 内每类 route 的代表 token 已经能近似该路线的读出竞争方向。
```

### 理论进展

Phase 744 的公式：

```text
component_effect(u)
= donor_boost(u)
+ competitor_suppression(u)
+ route_shift(u)
```

Phase 745 将闭合条件收紧为：

```text
Closure =
logit(y_donor)
- max_R max_{c in R} logit(c)
> 0
```

其中：

```text
R in {recipient, format, echo, punctuation, prose, other}
```

因此完整 token0 闭合不是：

```text
donor > current_top
```

而是：

```text
for every dominant route R:
  donor > max competitor in R
```

更接近当前现象的公式：

```text
ClosureForce =
DonorBoost
+ MultiRouteSuppression
- RouteTakeoverRisk
```

其中：

```text
MultiRouteSuppression =
sum_R Suppress(R)

RouteTakeoverRisk =
max_{R not suppressed} max_{c in R} logit(c)
```

### 问题和硬伤

1. Phase 745 仍然是 final-norm readout geometry intervention（最终读出几何干预），不是自然 circuit proof（自然回路证明）。

2. route classifier 仍较粗：

```text
other_vocab 可能混合 option marker / label route / tokenizer artifact。
format_or_schema 也可能包含多种格式子路线。
```

3. 压制 top-k 内的路线不等于压制完整词表路线。若 top-k 之外存在潜在竞争路线，本阶段无法看到。

4. 当前只测 token0，尚未把 continuation closure（续写闭合）纳入 route-level suppression。

5. 小模型可能有更强格式/回声吸引子，不能把具体层号或具体 token 竞争模式外推到大模型。

### 下一步

Phase 746 应进入：

```text
Natural Route Suppressor Localization
自然路线级抑制器定位
```

目标不是继续人工压 route，而是找到自然组件是否能分别降低：

```text
recipient route max
format route max
echo route max
other_vocab route max
```

建议做法：

```text
1. 复用 Phase 744 的 late attn/mlp component scan。
2. 不再只记录 current top competitor delta。
3. 对每个组件 u，计算：

   boost_donor(u)
   suppress_recipient_route(u)
   suppress_format_route(u)
   suppress_echo_route(u)
   suppress_other_route(u)

4. 找到自然 route suppressor set。
5. 再下钻到 head/channel/neuron。
```

## Phase 746: 自回归训练—条件化相对状态—生成场闭合理论整合 [2026-06-29 09:46]

### 任务

用户提供附件理论，要求比较它与：

```text
research/IntelligentTheory.md
```

中的现有理论差异，并综合为最新理论，更新：

```text
research/IntelligentTheory.md
```

### 使用材料

```text
/home/rankrank/.codex/attachments/00c38cef-400f-48c0-9252-e7d85b8b32b0/pasted-text.txt
research/IntelligentTheory.md
```

### 执行命令

```bash
sed -n '1,260p' /home/rankrank/.codex/attachments/00c38cef-400f-48c0-9252-e7d85b8b32b0/pasted-text.txt
sed -n '260,760p' /home/rankrank/.codex/attachments/00c38cef-400f-48c0-9252-e7d85b8b32b0/pasted-text.txt
sed -n '760,1200p' /home/rankrank/.codex/attachments/00c38cef-400f-48c0-9252-e7d85b8b32b0/pasted-text.txt
sed -n '1120,1635p' research/IntelligentTheory.md
rg -n "^七|^\\s*7\\.[0-9]|自回归训练|route-level|自然路线级|机制闭合度" research/IntelligentTheory.md
```

本阶段没有进行模型测试。

### 理论差异判断

附件理论的核心价值：

```text
把机制图谱接回自回归训练闭环。
它解释为什么 attention / MLP / residual / lm_head 会在训练中形成关系寻址、内容搬运、竞争抑制和生成闭合路径。
```

原 IntelligentTheory.md 的核心价值：

```text
把推理时的机制现象整理为：
相对编码、复用差分、条件化状态变换、QK/V 分解、源贡献路线增益、候选竞争、生成场闭合。
```

两者关系：

```text
附件理论解释机制为什么被训练出来；
原理论解释机制推理时如何运行和如何失败。
```

### 文档更新

已将第七章标题从：

```text
条件化相对状态—生成场闭合理论
```

升级为：

```text
自回归训练—条件化相对状态—生成场闭合理论
```

新增 7.0：

```text
2026-06-29 最新整合：从训练闭环到生成闭合
```

并将后续小节顺延：

```text
7.1 对前一版整合理论的保留与修正
7.2 当前最有效完整理论的明确陈述
7.3 完整计算例子
7.4 问题与硬伤
7.5 下一步阶段性大任务
7.6 当前整体进展评估
```

### 最新理论

理论名称：

```text
自回归训练—条件化相对状态—生成场闭合理论
```

简称：

```text
自回归相对状态闭合理论
```

三层结构：

```text
1. 自回归训练塑形层
   真实下一个 token 作为监督信号，通过 cross entropy 同时增强正确 token、压制高概率错误 token。

2. 条件化相对状态形成层
   词嵌入进入上下文后，经 attention / MLP / residual 形成对象、关系、格式、绑定、路线和候选竞争状态。

3. 生成场竞争闭合层
   目标 token / phrase 必须压过 recipient、format、echo、prose、punctuation、other_vocab 等多路线竞争者。
```

核心公式：

```text
P_theta(x_{t+1}|x_{<=t})
= softmax(W_U · LN(h_t^L))

L_t = -log P_theta(x_{t+1}|x_{<=t})

dL_t / d logit(y)
= P_theta(y|x_{<=t}) - 1[y=x_{t+1}]
```

含义：

```text
正确 token 被推高；
高概率错误竞争 token 被压低；
梯度沿 lm_head、final norm、residual、MLP、attention、Q/K/V、embedding 反传；
长期训练后形成 source -> writer -> rewriter -> route competition -> generation closure。
```

Phase 745 后的路线级闭合公式：

```text
Token0Closure
⇔
logit(y_target)
- max_R max_{c in R} logit(c)
> 0

R ∈ {recipient, format, echo, prose, punctuation, other}
```

短语闭合公式：

```text
L(y|x)
=
(1/m) * Σ_i log P(y_i | x, y_<i)

GenerationClosure
⇔
L(y_target|x)
- max_{y != y_target} L(y|x)
> δ
```

### 关键更新

1. 自回归训练不是简单记忆下一个词，而是在海量文本中塑造关系寻址、内容搬运、状态重写、竞争抑制和生成闭合路径。

2. 注意力更准确地说是“对当前预测有用的关系性寻址”，不是完整理解本身。

3. MLP 是 rewriter，把 attention 搬运来的信息变成读出可用状态，并参与路线增强或压制。

4. 残差流是 carrier，承载跨层状态累积。

5. lm_head / unembedding 是读出竞争接口。

6. Phase 745 后必须把 candidate competition 升级为 route-level competition field。

7. 当前机制闭合度从 65%-75% 更新为 70%-78%。

### 新增硬伤

1. 训练塑形解释仍是机制合理性推断，不是直接证明某个自然组件由该训练压力形成。

2. Phase 745 的 route-level suppression 仍是 final-norm readout geometry intervention，不是 natural circuit proof。

3. route classifier 仍粗，other_vocab 和 format_or_schema 可能混合多个子路线。

### 下一步

新增两个优先任务：

```text
大任务0：自然路线级抑制器定位
  对每个自然组件 u 测：
    boost_target(u)
    suppress_recipient_route(u)
    suppress_format_route(u)
    suppress_echo_route(u)
    suppress_other_route(u)

大任务00：训练塑形证据回溯
  比较 target-vs-competitor gradient 与已发现 writer / rewriter / suppressor 组件方向是否对齐，
  通过小规模可控训练或微调观察 route suppressor 是否随 loss pressure 形成。
```

### 最严格结论

本阶段没有新增模型证据，但完成了理论结构上的关键补环：

```text
训练解释机制来源；
推理解释机制运行；
生成场解释机制为何失败或闭合；
路线级竞争解释为什么单 top competitor suppression 不足。
```

当前理论仍不能宣称已经破解语言编码机制。最关键的未闭合点仍是：

```text
自然 route suppressor 的组件级来源；
训练塑形层与推理图谱节点的直接因果连接；
神经元级/通道级全局图谱。
```

## Phase 747: 预测充分状态不变量与生命闭合数学整合 [2026-06-29 09:54]

### 任务

用户提供三份附件理论，要求比较它们与：

```text
research/IntelligentTheory.md
```

中的现有理论差异，并综合为最新理论，更新：

```text
research/IntelligentTheory.md
```

### 使用材料

```text
/home/rankrank/.codex/attachments/b84c33d0-1cdb-489f-86a7-63a49faa60f3/pasted-text.txt
/home/rankrank/.codex/attachments/257aab2e-6353-4120-aec8-866a44e89c8b/pasted-text.txt
/home/rankrank/.codex/attachments/1d34c39f-3ab7-4d52-8542-f94baf0f598f/pasted-text.txt
research/IntelligentTheory.md
```

### 执行命令

```bash
wc -l /home/rankrank/.codex/attachments/b84c33d0-1cdb-489f-86a7-63a49faa60f3/pasted-text.txt /home/rankrank/.codex/attachments/257aab2e-6353-4120-aec8-866a44e89c8b/pasted-text.txt /home/rankrank/.codex/attachments/1d34c39f-3ab7-4d52-8542-f94baf0f598f/pasted-text.txt research/IntelligentTheory.md
sed -n '1,240p' /home/rankrank/.codex/attachments/b84c33d0-1cdb-489f-86a7-63a49faa60f3/pasted-text.txt
sed -n '241,520p' /home/rankrank/.codex/attachments/b84c33d0-1cdb-489f-86a7-63a49faa60f3/pasted-text.txt
sed -n '521,980p' /home/rankrank/.codex/attachments/b84c33d0-1cdb-489f-86a7-63a49faa60f3/pasted-text.txt
sed -n '981,1460p' /home/rankrank/.codex/attachments/b84c33d0-1cdb-489f-86a7-63a49faa60f3/pasted-text.txt
sed -n '1461,2170p' /home/rankrank/.codex/attachments/b84c33d0-1cdb-489f-86a7-63a49faa60f3/pasted-text.txt
sed -n '1,260p' /home/rankrank/.codex/attachments/257aab2e-6353-4120-aec8-866a44e89c8b/pasted-text.txt
sed -n '261,620p' /home/rankrank/.codex/attachments/257aab2e-6353-4120-aec8-866a44e89c8b/pasted-text.txt
sed -n '621,980p' /home/rankrank/.codex/attachments/257aab2e-6353-4120-aec8-866a44e89c8b/pasted-text.txt
sed -n '1,300p' /home/rankrank/.codex/attachments/1d34c39f-3ab7-4d52-8542-f94baf0f598f/pasted-text.txt
sed -n '301,760p' /home/rankrank/.codex/attachments/1d34c39f-3ab7-4d52-8542-f94baf0f598f/pasted-text.txt
sed -n '761,1120p' /home/rankrank/.codex/attachments/1d34c39f-3ab7-4d52-8542-f94baf0f598f/pasted-text.txt
sed -n '1121,1400p' /home/rankrank/.codex/attachments/1d34c39f-3ab7-4d52-8542-f94baf0f598f/pasted-text.txt
rg -n "I_\\{|I_\\w|Id_|Int_|identity|intent|预测充分|生命闭合|理论组织完整度" research/IntelligentTheory.md
```

本阶段没有进行模型测试。

### 理论差异判断

第一份附件的价值：

```text
把“条件化相对状态—生成场闭合理论”的公式逐条拆开；
明确这些变量是机制分解变量，不是模型内部显式存储变量；
修正 Id(identity，身份) 与 Int(intent，意图许可) 的符号冲突；
补充 source contribution、route gain、identity binding、readout、softmax、phrase likelihood、generation closure 的完整计算链。
```

第二份附件的价值：

```text
提出最接近全局不变量的对象不是固定神经元、固定 head 或固定语义向量，
而是“预测闭合不变量”：
任意前缀状态都必须形成足以预测下一个 token 的读出分布。
```

第三份附件的价值：

```text
把预测闭合推广到生命系统的上位框架：
生命系统不是由单一守恒量定义，而是由边界维持、可生存域、功能闭合、修复和生成能力定义。
这对智能理论有启发，但仍是外推，不是语言模型机制实验证据。
```

原 IntelligentTheory.md 的核心价值：

```text
已有文件主要整理了自回归训练塑形、条件化相对状态、复用差分、机制图谱、QK/V 拆分、生成场竞争闭合和非线性理论。
它更偏向“语言模型推理机制如何运行和失败”。
```

### 文档更新

已将第七章标题升级为：

```text
预测充分状态—自回归训练—条件化相对状态—生成场闭合理论，以及问题硬伤和下一步
```

新增小节：

```text
7.0.1 2026-06-29 再整合：预测充分状态不变量与生命闭合数学
```

新增理论名称：

```text
预测充分状态—自回归训练—条件化相对状态—生成场闭合理论
```

简称：

```text
预测充分相对状态闭合理论
```

### 最新核心公式

预测误差势能：

```text
Q(x_{<=t})
=
-log P_theta(x_{t+1}|x_{<=t})
```

统一词表读出坐标系：

```text
ell_t
=
W_U Norm(h_t)
in R^{|V|}
```

读出等价类：

```text
h ~_readout h'
iff
softmax(W_U Norm(h))
approx
softmax(W_U Norm(h'))
```

预测充分状态：

```text
P(x_{t+1}|h_t)
approx
P(x_{t+1}|x_{<=t})
```

预测充分等价类：

```text
H_suf(x_{<=t})
=
{
  h:
  D(
    P_theta(.|h),
    P_theta(.|x_{<=t})
  )
  < epsilon
}
```

生成闭合：

```text
GenerationClosure
iff
L(y_target|x)
-
max_{y != y_target} L(y|x)
>
delta
```

生命闭合不变量：

```text
I_life
=
(
  B,
  V,
  F,
  R,
  G
)
```

其中：

```text
B = boundary，边界；
V = viability domain，可生存域；
F = functional closure，功能闭合；
R = repairability，可修复性；
G = generation / reproduction，生成 / 复制性。
```

生命作用量：

```text
A_life
=
integral [
  D_V
  +
  D_B
  +
  E_closure
  +
  E_prediction
  +
  E_repair
  +
  E_generation
  +
  E_cost
] dt
```

语言模型是退化特例：

```text
A_LM
=
sum_t -log P_theta(x_{t+1}|x_{<=t})
```

### 关键结论

1. 当前语言模型理论最接近不变量的对象，是给定前缀后的读出分布，以及能够产生该分布的预测充分状态等价类。

2. 条件化相对状态轨迹仍然重要，但它现在应被理解为“进入某个预测充分等价类的路径”。

3. 生成场闭合仍然重要，因为预测充分不等于自然生成成功；目标路线还必须压过 format、echo、recipient、prose、punctuation、other_vocab 等竞争路线。

4. 生命闭合数学提供上位启发，但不能直接替代语言模型内部机制测试。

### 新增硬伤

1. 预测充分状态不变量是全局等价类，不是局部机制定位。它不能直接告诉我们哪个 head、channel、MLP 神经元负责预测充分。

2. 生命闭合数学是上位类比，不是语言模型实验结论。当前模型实验证据只支持预测闭合、生成闭合、路线竞争和局部机制图谱。

3. 预测充分等价类目前尚未组件级测量，需要把读出分布距离、top-k logit margin distance 和 generation closure 联动起来。

### 下一步

新增大任务：

```text
大任务-1：预测充分等价类测量
```

目标：

```text
把“预测充分状态不变量”从理论概念变成可测对象。
```

核心做法：

```text
1. 对每个样本记录完整 top-k 读出分布 P_theta(.|x_{<=t})。
2. 对每个候选组件 u 做干预，得到 P_theta(.|do(u))。
3. 计算读出分布距离 D_readout(u)。
4. 计算目标预测充分增益 G_suf(u)。
5. 区分：
   只改变 hidden state 的组件；
   改变读出分布但不闭合的组件；
   真正把状态推入目标预测充分等价类并提升生成闭合的组件。
```

成功判据：

```text
若某类 head / channel / MLP 组件能跨样本稳定降低 target distribution distance，
并同时提升 token0 closure 与 phrase closure，
则“预测充分状态”从全局不变量进入组件级图谱。
```

### 最严格结论

本阶段没有新增模型实验，但完成了理论上位层的关键收紧：

```text
局部机制图谱
不再只是解释某个 head 或 channel 的功能；
而是要解释这些局部机制如何共同维持预测充分状态，
并如何在生成场中完成或失败于闭合。
```

当前仍不能宣称已经建立新的完整数学体系。
更准确地说：

```text
预测充分状态不变量
是语言模型方向最可操作的候选全局不变量；
生命闭合不变量
是生物学和智能理论方向的上位候选框架；
二者之间还缺少直接实验桥梁。
```

## Phase 748: Natural Route Suppressor Matrix（自然路线抑制器矩阵） [2026-06-29 10:20]

### 本阶段问题

前面 Phase 745 已经证明：只压制单个 top competitor（最高竞争者）不足以完成 token0 closure（首词元闭合），需要 route-level multi-competitor suppression（路线级多竞争者抑制）。

本阶段继续问：

```text
是否能把 suppressor（抑制器）从“某个 patch 是否成功”
推进为可测量的 route suppressor matrix（路线抑制矩阵）？
```

### 新增脚本

```text
tests/gpt5/phase748_natural_route_suppressor_matrix.py
tests/gpt5/run_phase748_natural_route_suppressor_matrix_round.sh
```

结果目录：

```text
results/glm5_phase748_natural_route_suppressor_matrix/
```

### 运行命令

冒烟测试：

```bash
tests/gpt5/run_phase748_natural_route_suppressor_matrix_round.sh smoke --max-pairs 1 --top-audits 1 --top-candidates 1 --top-k-vocab 10 --max-topk-tokens 6 --max-route-classes 4 --max-scan-candidates 4 --log-every 1
```

主测试：

```bash
tests/gpt5/run_phase748_natural_route_suppressor_matrix_round.sh main --max-pairs 6 --top-audits 2 --top-candidates 3 --top-k-vocab 16 --max-topk-tokens 10 --max-route-classes 6 --log-every 2
```

确认测试：

```bash
tests/gpt5/run_phase748_natural_route_suppressor_matrix_round.sh confirm --max-pairs 8 --top-audits 2 --top-candidates 3 --top-k-vocab 20 --max-topk-tokens 12 --max-route-classes 7 --log-every 2
```

三模型按 qwen3、GLM4、DS7B 顺序运行；每个模型结束后释放 GPU，并通过 `--hard-exit-after-model` 避免显存残留。

### 测试原理

对每个候选 whole component（整体组件）做 donor-recipient delta（供体-受体差分）干预，并测量它对不同 route（路线）的读出最大值影响。

路线读出分数：

```text
S_R(h)=max_{y in V_R} W_U(y)^T Norm(h)
```

组件对路线的抑制：

```text
Suppress_u(R)=S_R(h_base)-S_R(h_do(u))
```

组件对目标答案的增强：

```text
Boost_u=logit_target(h_do(u))-logit_target(h_base)
```

路线抑制矩阵：

```text
M_{u,R}=Suppress_u(R)
```

测量路线包括：

```text
recipient_answer
format_or_schema
echo_object_or_relation
punctuation_or_stop
other_vocab
other_semantic_value
```

### 确认轮核心结果

qwen3：

```text
L32:mlp_out
  target boost = -0.047
  route suppression = 2.961
  route coverage = 3.875
  donor top1 = 0.000

L33:attn_out
  target boost = 2.547
  route suppression = 2.531
  route coverage = 2.188
  donor top1 = 0.000

L32:attn_out
  target boost = 3.066
  route suppression = 1.180
  route coverage = 1.938
  donor top1 = 0.000
```

qwen3 的 L32:mlp_out 更像纯路线 suppressor：目标答案没有被增强，但 format、other_vocab、punctuation、recipient、echo 等路线被稳定压低。L32/L33 attention output 更像 booster + suppressor 混合组件。

GLM4：

```text
L34:attn_out
  target boost = 1.504
  route suppression = 0.525
  route coverage = 2.500
  donor top1 = 0.875

L35:attn_out
  target boost = 0.543
  route suppression = 0.611
  route coverage = 2.812
  donor top1 = 0.625

L36:mlp_out
  target boost = -0.441
  route suppression = 1.332
  route coverage = 3.125
  donor top1 = 0.312
```

GLM4 的 donor answer 基线较强，所以更多表现为 maintenance candidate（维持候选），而不是从错误状态修复到正确状态的 closure candidate（闭合候选）。

DS7B：

```text
L23:attn_out
  target boost = 1.762
  route suppression = 1.637
  route coverage = 2.875
  donor top1 = 0.250

L22:attn_out
  target boost = 1.531
  route suppression = 2.809
  route coverage = 2.688
  donor top1 = 0.188

L26:mlp_out
  target boost = 0.586
  route suppression = 2.215
  route coverage = 3.438
  donor top1 = 0.000

L25:mlp_out
  target boost = 0.414
  route suppression = 2.055
  route coverage = 3.125
  donor top1 = 0.000
```

DS7B 最支持“路线级 suppressor matrix 可测”：L22-L23 attention output 同时有目标增强和 recipient_answer 抑制，L25-L26 MLP output 更像广域路线抑制器。

### 关键结论

1. 当前结果支持：suppressor 不是单点 token 压制，而是 route-level suppressor matrix（路线级抑制矩阵）。

2. 全局 suppressor 不是完整语言编码机制，但它很可能是语言编码中“路线选择与生成闭合”的核心入口。

3. 真实机制不是单纯抑制，而是：

```text
target boost（目标增强）
+ route suppression（路线抑制）
+ readout field reordering（读出场重排）
```

4. qwen3、GLM4、DS7B 都出现了可测路线抑制，但三者形态不同：qwen3 抑制强但闭合弱，GLM4 维护强，DS7B 最接近 booster + suppressor 混合闭合。

### 硬伤

1. 当前证据仍是 whole-component donor-recipient delta，不是自然前向因果链的直接证明。

2. 组件粒度仍然太粗，不能证明具体 head、channel、neuron 承担 suppressor。

3. route taxonomy 基于 top-k 词元和人工分类，可能漏掉真实竞争路线。

4. 当前模型是小模型，内部结构可能存在偏差，不能直接外推到大模型。

5. donor top1 rate 仍然偏低，说明 suppressor alone 不足以完成 closure。

### 理论进展

本阶段把“全局 suppressor 就是破解语言编码机制”的说法收紧为：

```text
全局 suppressor 不是全部语言编码机制；
但它是语言编码中路线选择、竞争压制、生成闭合的关键结构。
```

完整图谱应是：

```text
source / writer / rewriter / booster / suppressor / readout / continuation / closure
```

其中 suppressor 对应 selective mechanism（选择机制），writer/rewriter/booster 对应 constructive mechanism（构造机制）。

### 下一阶段

Phase 749 建议：

```text
Suppressor Component Decomposition
抑制器组件分解
```

优先下钻：

```text
qwen3:
  L32:mlp_out
  L33:attn_out
  L32:attn_out

GLM4:
  L34:attn_out
  L35:attn_out
  L36:mlp_out

DS7B:
  L22:attn_out
  L23:attn_out
  L25:mlp_out
  L26:mlp_out
```

核心问题：

```text
1. attention output 是否能分解到 head 级 suppressor？
2. MLP output 是否能分解到 channel / neuron 级 suppressor？
3. booster 和 suppressor 是否能被分离？
4. route-specific suppressor 与 broad/global suppressor 是否可区分？
```

## Phase 749: Suppressor Component Decomposition（抑制器组件分解） [2026-06-29 10:33]

### 任务背景

Phase 748 证明 whole-component donor-recipient delta（整体组件供体-受体差分）可以形成 route-level suppressor matrix（路线级抑制矩阵），但仍然停留在粗粒度组件层：

```text
L33:attn_out / L32:mlp_out 有 suppressor 效果，
不等于已经知道内部哪个 head（注意力头）、channel（通道）或 neuron（神经元）承担该效果。
```

本阶段继续同一阶段目标：从 whole-component suppressor（整体组件抑制器）下钻到 subunit suppressor（子单元抑制器）。

### 生成脚本

```text
tests/gpt5/phase749_suppressor_component_decomposition.py
tests/gpt5/run_phase749_suppressor_component_decomposition_round.sh
```

结果目录：

```text
results/glm5_phase749_suppressor_component_decomposition/
```

确认轮摘要：

```text
results/glm5_phase749_suppressor_component_decomposition/confirm/phase749_cross_model_summary.md
```

### 执行命令

冒烟测试：

```bash
tests/gpt5/run_phase749_suppressor_component_decomposition_round.sh smoke --max-pairs 1 --top-audits 1 --top-candidates 1 --max-components 1 --top-k-vocab 10 --max-topk-tokens 6 --max-route-classes 4 --top-heads-per-component 2 --random-heads-per-component 1 --headset-sizes 1 2 --channelset-sizes 1 4 --individual-channels 2 --log-every 1
```

主测试：

```bash
tests/gpt5/run_phase749_suppressor_component_decomposition_round.sh main --max-pairs 4 --top-audits 2 --top-candidates 3 --max-components 3 --top-k-vocab 16 --max-topk-tokens 10 --max-route-classes 6 --top-heads-per-component 4 --random-heads-per-component 2 --headset-sizes 1 2 4 --channelset-sizes 1 4 16 64 --individual-channels 4 --log-every 2
```

确认测试：

```bash
tests/gpt5/run_phase749_suppressor_component_decomposition_round.sh confirm --max-pairs 6 --top-audits 2 --top-candidates 3 --max-components 3 --top-k-vocab 18 --max-topk-tokens 12 --max-route-classes 7 --top-heads-per-component 4 --random-heads-per-component 2 --headset-sizes 1 2 4 --channelset-sizes 1 4 16 64 --individual-channels 4 --log-every 2
```

三个模型按 qwen3、GLM4、DS7B 顺序运行，并使用 `--hard-exit-after-model` 避免 GPU 显存残留。

### 测试原理

attention output（注意力输出）分解：

```text
1. 捕获 donor（供体）和 recipient（受体）在 attention o_proj（注意力输出投影）前的 per-head 表示。
2. 计算每个 head 的 donor-recipient delta。
3. 将单个 head 或 topH{1,2,4} headset（注意力头集合）的 delta 经 o_proj.weight 投影回 residual stream（残差流）。
4. 用 route-level max-logit matrix（路线级最大 logit 矩阵）测量 target boost（目标增强）、route suppression（路线抑制）、margin gain（间隔增益）和 delta fraction（相对整体效果比例）。
```

MLP output（多层感知机输出）分解：

```text
1. 捕获 donor 和 recipient 的 MLP residual output delta（残差输出差分）。
2. 按通道估计 route suppression / margin gain。
3. 测试单通道和 topC{1,4,16,64} channelset（通道集合）。
4. 与 whole-component（整体组件）效果比较。
```

注意：MLP channelset 只是 residual output channel evidence（残差输出通道证据），不是严格 neuron（神经元）证据。

### 关键结果

qwen3 最强 attention 子组件：

```text
L33:attn_out:topH4
donor top1 = 0.000
target boost = 1.589
route suppression = 3.406
coverage = 3.92
margin gain = 2.261
delta fraction = 0.856
effect = global_suppressor_margin_candidate
```

qwen3 对照整体组件：

```text
L33:attn_out whole-component
target boost = 2.948
route suppression = 2.656
margin gain = 3.343
```

解释：qwen3 的 L33 attention suppressor（第33层注意力抑制器）可以被 topH4 headset（前4个注意力头集合）复现大部分路线抑制效果，甚至在 route suppression 指标上超过整体组件，但 target boost 和最终闭合仍依赖整体组件。

qwen3 MLP 结果：

```text
L32:mlp_out whole-component:
  route suppression = 3.094

L32:mlp_out:topC64:
  route suppression = 1.979
  coverage = 4.50
  delta fraction = 0.365
```

GLM4 结果：

```text
L36:mlp_out whole-component:
  donor top1 = 0.333
  target boost = -0.344
  route suppression = 1.086
  margin gain = -0.359

L34:attn_out:topH4:
  donor top1 = 0.667
  target boost = 0.729
  route suppression = 0.659
  delta fraction = 0.668

L35:attn_out:topH4:
  donor top1 = 0.333
  target boost = 0.099
  route suppression = 0.841
  delta fraction = 0.662
```

解释：GLM4 的子组件能复现部分抑制形状，但更像 maintenance（维护）而不是强纠偏。

DS7B 最强 attention 子组件：

```text
L22:attn_out whole-component:
  donor top1 = 0.167
  target boost = 1.208
  route suppression = 3.323
  margin gain = 1.739

L22:attn_out:topH4:
  donor top1 = 0.167
  target boost = 0.958
  route suppression = 2.768
  margin gain = 1.432
  delta fraction = 0.708
```

另一个 DS7B 结果：

```text
L23:attn_out whole-component:
  donor top1 = 0.250
  target boost = 1.620
  route suppression = 2.057

L23:attn_out:topH4:
  donor top1 = 0.250
  target boost = 1.266
  route suppression = 1.911
  delta fraction = 0.739
```

DS7B MLP 结果：

```text
L25:mlp_out whole-component:
  donor top1 = 0.167
  target boost = 0.354
  route suppression = 2.292

L25:mlp_out:topC64:
  donor top1 = 0.167
  target boost = 0.359
  route suppression = 1.177
  delta fraction = 0.310
```

### 客观结论

1. Phase 748 的 route suppressor matrix（路线抑制矩阵）不是只能在整体组件层观察到。

2. qwen3 和 DS7B 中，attention suppressor（注意力抑制器）可以明显下钻到少量 headset（注意力头集合）：

```text
qwen3:
  L33:attn_out:topH4

DS7B:
  L22:attn_out:topH4
  L23:attn_out:topH4
```

3. MLP suppressor（多层感知机抑制器）可以部分下钻到 residual output channelset（残差输出通道集合），但解释力弱于 attention headset。

4. booster（增强器）和 suppressor（抑制器）仍然纠缠，尚未完全分离。

5. donor top1 rate（供体第一名率）仍偏低，说明 suppressor 可以重排竞争路线，但不能单独保证自然生成闭合。

### 理论进展

Phase 749 把前一阶段的图谱从：

```text
whole-component suppressor
```

推进到：

```text
headset-level suppressor candidate
channelset-level partial suppressor candidate
```

因此对“全局 suppressor 就是破解语言编码机制”的判断应收紧为：

```text
全局 suppressor 是破解语言编码机制的关键入口，
但不是完整语言编码机制本身。
```

更完整的当前图谱是：

```text
source（源词元）
  -> writer（写入器）
  -> skeleton carrier（骨架承载器）
  -> rewriter / booster（重写器 / 增强器）
  -> route suppressor matrix（路线抑制矩阵）
  -> readout competition（读出竞争）
  -> continuation closure（续写闭合）
```

### 硬伤

1. attention head evidence（注意力头证据）来自 o_proj projected delta（输出投影差分），还不是自然前向因果链证明。

2. MLP channel evidence（多层感知机通道证据）不是 neuron evidence（神经元证据）。

3. 路线分类仍依赖 top-k token（前 k 个词元）和人工 taxonomy（分类表）。

4. 当前模型为小模型，内部结构可能偏离大模型。

5. suppressor 只解释选择和闭合的一部分，不解释全部语义构造。

### 下一阶段

Phase 750 应继续同一阶段目标：

```text
Natural Subunit Suppressor Necessity Test
自然子单元抑制器必要性测试
```

核心问题：

```text
1. topH4 / topC64 是否在自然 recipient forward（受体自然前向）中必要？
2. 移除这些子单元是否会破坏目标 token、竞争路线和生成闭合？
3. attention headset 的效果是否来自 source token attention（源词元注意力）和 value vector（值向量）？
4. MLP channelset 是否能进一步追到中间 neuron（神经元），还是只是分布式残差投影？
```

如果 Phase 750 成立，研究可以从：

```text
差分修补图谱
```

推进到：

```text
自然前向功能图谱
```

## Phase 750: Natural Subunit Suppressor Necessity Test（自然子单元抑制器必要性测试） [2026-06-29 10:44]

### 任务背景

Phase 749 证明部分 attention headset（注意力头集合）和 MLP channelset（多层感知机通道集合）可以复现 donor-recipient delta（供体-受体差分）里的 route suppression（路线抑制）。但这还不能证明它们是自然前向机制的一部分。

本阶段的目标是把问题从：

```text
patch candidate（修补候选）
```

推进到：

```text
natural necessity candidate（自然必要性候选）
```

因此 Phase 750 不再注入 donor delta，而是在 natural donor / natural recipient forward（自然供体/受体前向）中直接擦除子单元，观察目标答案、竞争路线和读出间隔是否退化。

### 生成脚本

```text
tests/gpt5/phase750_natural_subunit_suppressor_necessity.py
tests/gpt5/run_phase750_natural_subunit_suppressor_necessity_round.sh
```

结果目录：

```text
results/glm5_phase750_natural_subunit_suppressor_necessity/
```

确认轮摘要：

```text
results/glm5_phase750_natural_subunit_suppressor_necessity/confirm/phase750_cross_model_summary.md
```

### 执行命令

冒烟测试：

```bash
tests/gpt5/run_phase750_natural_subunit_suppressor_necessity_round.sh smoke --max-pairs 1 --top-audits 1 --max-components 1 --top-k-vocab 10 --max-topk-tokens 6 --max-route-classes 4 --headset-sizes 1 2 --channelset-sizes 16 --individual-heads 1 --individual-channels 1 --log-every 1
```

主测试：

```bash
tests/gpt5/run_phase750_natural_subunit_suppressor_necessity_round.sh main --max-pairs 4 --top-audits 2 --max-components 3 --top-k-vocab 16 --max-topk-tokens 10 --max-route-classes 6 --headset-sizes 1 2 4 --channelset-sizes 16 64 --individual-heads 1 --individual-channels 1 --log-every 2
```

确认测试：

```bash
tests/gpt5/run_phase750_natural_subunit_suppressor_necessity_round.sh confirm --max-pairs 6 --top-audits 2 --max-components 3 --top-k-vocab 18 --max-topk-tokens 12 --max-route-classes 7 --headset-sizes 1 2 4 --channelset-sizes 16 64 --individual-heads 1 --individual-channels 1 --log-every 2
```

三个模型按 qwen3、GLM4、DS7B 顺序运行，并使用 `--hard-exit-after-model`。

### 测试原理

attention erase（注意力擦除）：

```text
在 o_proj（输出投影）前，把指定 head 的 final token 输入 slice 置零。
```

MLP erase（多层感知机擦除）：

```text
在 MLP output（多层感知机输出）里，把指定 residual output channels（残差输出通道）置零。
```

衡量指标：

```text
target_logit_drop_after_erase:
  擦除后目标答案 logit 下降量。

total_positive_route_release_after_erase:
  擦除后竞争路线最大 logit 正向释放总量。

mean_margin_drop_target_vs_routes:
  目标答案相对竞争路线的平均间隔下降。

top1_loss_rate:
  原来目标答案 top1，擦除后失去 top1 的比例。
```

### 确认轮关键结果

qwen3：

```text
L32:attn_out:topH4 / natural_donor
base top1 = 1.000
after top1 = 1.000
top1 loss = 0.000
target drop = -1.292
route release = 1.979
coverage = 3.00
margin drop = -0.919
effect = erase_improves_or_inverse_effect
```

解释：qwen3 出现路线释放，但 target drop 为负，margin 也不退化，说明 Phase 749 的 patch suppressor 不等于自然必要 suppressor。

qwen3 MLP：

```text
L32:mlp_out:topC64 / natural_donor
base top1 = 1.000
after top1 = 1.000
target drop = -0.417
route release = 1.854
margin drop = 0.022
effect = small_or_no_effect
```

GLM4：

```text
L36:mlp_out:topC64 / natural_donor
base top1 = 1.000
after top1 = 1.000
target drop = -0.198
route release = 0.969
coverage = 4.33
margin drop = 0.023
effect = small_or_no_effect
```

少量 GLM4 单 head 有自然必要性候选，但样本数太少：

```text
L34:attn_out:H4 / natural_donor
n = 2
target drop = 0.312
route release = 0.375
margin drop = 0.250
effect = natural_suppressor_necessity_candidate
```

DS7B 最强自然必要性结果：

```text
L22:attn_out:topH4 / natural_donor
base top1 = 0.750
after top1 = 0.500
top1 loss = 0.250
target drop = 1.438
route release = 0.682
coverage = 1.58
margin drop = 1.423
effect = target_support_necessity_candidate
```

DS7B 单 head：

```text
L22:attn_out:H1 / natural_donor
base top1 = 0.857
after top1 = 0.714
top1 loss = 0.143
target drop = 0.518
route release = 0.777
coverage = 2.29
margin drop = 0.616
effect = natural_suppressor_necessity_candidate
```

DS7B L23：

```text
L23:attn_out:topH4 / natural_recipient
base top1 = 0.750
after top1 = 0.667
top1 loss = 0.083
target drop = 0.542
route release = 0.740
coverage = 2.08
margin drop = 0.542
effect = target_support_necessity_candidate
```

DS7B MLP 较弱：

```text
L25:mlp_out:topC64 / natural_donor
base top1 = 0.750
after top1 = 0.750
target drop = -0.104
route release = 0.578
margin drop = -0.076
effect = small_or_no_effect
```

### 客观结论

1. Phase 750 是对 Phase 749 的重要纠偏：patch-visible suppressor（修补可见抑制器）不能直接等同 natural suppressor unit（自然抑制单元）。

2. qwen3 的自然擦除多数是 erase improves / inverse effect（擦除改善或反向效果），不支持强自然必要性。

3. GLM4 多数是 weak route release（弱路线释放），不破坏 top1 closure（第一名闭合）。

4. DS7B 的 L22 / L23 attention subunits（第22/23层注意力子单元）出现最接近自然机制的证据：擦除后目标下降、竞争路线释放、margin 退化，并出现部分 top1 loss。

5. MLP channelset（多层感知机通道集合）在自然必要性测试中明显弱于 attention headset（注意力头集合）。

### 理论进展

当前图谱应更新为：

```text
patch-visible suppressor（修补可见抑制器）
  说明某个差分方向可以压制竞争路线；

natural suppressor unit（自然抑制单元）
  必须在自然前向擦除中造成目标下降、竞争释放或闭合丢失；

full coding mechanism（完整编码机制）
  还需要解释 source、writer、rewriter、booster、suppressor、readout 的自然协同。
```

最稳妥判断：

```text
suppressor 是破解语言编码机制的关键拼图；
DS7B 的 L22/L23 attention 子单元是目前最接近自然 suppressor 的证据；
但完整语言编码机制尚未闭合。
```

### 硬伤

1. 擦除是人工干预，可能造成 off-manifold perturbation（离流形扰动）。

2. qwen3 和 GLM4 未出现稳定自然必要性，说明模型结构差异很大。

3. DS7B 是小模型，不能直接外推到大模型。

4. attention head（注意力头）证据仍不是 neuron（神经元）证据。

5. MLP channelset 不是 MLP neuron，且自然必要性弱。

6. route taxonomy（路线分类）仍可能漏掉真实竞争路线。

### 下一阶段

Phase 751 建议：

```text
Natural Attention Head Mechanism Backtrace
自然注意力头机制回溯
```

聚焦：

```text
L22:attn_out:H1
L22:attn_out:topH2
L22:attn_out:topH4
L23:attn_out:topH4
```

核心问题：

```text
1. 这些 head 在自然前向中 attend 哪些 source token？
2. Q/K pattern（查询/键模式）和 V/O content（值/输出内容）能否分离？
3. 它们是 target support（目标支持）、route suppression（路线抑制），还是二者混合？
4. L22 和 L23 是否构成连续机制链？
```

## Phase 751: 自然注意力头机制回溯 [2026-06-29 11:12]

### 任务判断

本阶段分析的上传内容方向基本正确：当前路线已经从普通 patch（修补）推进到 route-level suppressor（路线级抑制器）和 subunit natural necessity（子单元自然必要性）。但结论必须收紧：现在还不是完整语言编码机制，只是小模型、局部任务、head 级路径图谱。最关键的修正是：

```text
attention mass 只是 Q/K pattern（查询/键模式）观察证据；
真正更接近因果的是 source-restricted V/O contribution removal（源限制值/输出贡献移除）。
```

因此继续 Phase751 属于同一阶段性目标，不需要另行确认。

### 新增脚本

```text
tests/gpt5/phase751_natural_attention_head_mechanism_backtrace.py
tests/gpt5/run_phase751_natural_attention_head_mechanism_backtrace_round.sh
```

### 执行命令

静态检查：

```bash
python -m py_compile tests/gpt5/phase751_natural_attention_head_mechanism_backtrace.py
```

dry-run：

```bash
python tests/gpt5/phase751_natural_attention_head_mechanism_backtrace.py --dry-run --round-name smoke --max-pairs 1 --top-audits 1 --max-components 1 --top-k-vocab 10 --max-topk-tokens 6 --max-route-classes 4 --headset-sizes 1 2 --individual-heads 1 --max-focus-heads 1 --max-source-groups 4
```

smoke：

```bash
tests/gpt5/run_phase751_natural_attention_head_mechanism_backtrace_round.sh smoke --max-pairs 1 --top-audits 1 --max-components 1 --top-k-vocab 10 --max-topk-tokens 6 --max-route-classes 4 --headset-sizes 1 2 --individual-heads 1 --max-focus-heads 1 --max-source-groups 4 --log-every 1
```

main：

```bash
tests/gpt5/run_phase751_natural_attention_head_mechanism_backtrace_round.sh main --max-pairs 4 --top-audits 1 --max-components 2 --top-k-vocab 14 --max-topk-tokens 8 --max-route-classes 5 --headset-sizes 1 2 4 --individual-heads 1 --max-focus-heads 2 --max-source-groups 8 --log-every 1
```

confirm：

```bash
tests/gpt5/run_phase751_natural_attention_head_mechanism_backtrace_round.sh confirm --max-pairs 6 --top-audits 1 --max-components 2 --top-k-vocab 16 --max-topk-tokens 10 --max-route-classes 6 --headset-sizes 1 2 4 --individual-heads 1 --max-focus-heads 2 --max-source-groups 8 --log-every 2
```

说明：为了读取逐 head attention map（注意力图）并在 o_proj（输出投影）前移除 source contribution（源贡献），本阶段使用 bf16、非量化、eager attention。三模型均按 qwen3 -> GLM4 -> DS7B 顺序运行，并带 `--hard-exit-after-model`。

### 测试原理

本阶段把注意力头拆成两部分：

```text
Q/K pattern:
head 在自然前向中看向哪些 source token。

V/O content:
head 从这些 source token 取出的 value 内容，经 o_proj 写入 residual 后，对 target logit 和 route competition 有什么影响。
```

核心流程：

```text
1. 捕获自然前向中的 attention weights 和 v_proj 输出。
2. 把 prompt 切成 target_record_line、target_value_tokens、records_all、object_tokens、relation_tokens 等 source group。
3. 对候选 head / headset 计算 source group attention mass。
4. 计算 attention * value 得到 source contribution。
5. 在 o_proj 输入处只移除某 head 从某 source group 带来的 contribution。
6. 比较 target logit drop、route release、margin drop、top1 loss。
```

### 结果文件

```text
results/glm5_phase751_natural_attention_head_mechanism_backtrace/smoke/
results/glm5_phase751_natural_attention_head_mechanism_backtrace/main/
results/glm5_phase751_natural_attention_head_mechanism_backtrace/confirm/
```

confirm 轮行数：

```text
qwen3:      768
GLM4:      912
DS7B:     1040
```

### 主要结果

qwen3 confirm：

```text
L33:attn_out:H15 / natural_donor / target_record_line
attention mass = 0.307
source target contribution = 10.700
source route suppression contribution = 0.494
remove target drop = 1.125
route release = 0.125
top1 loss = 0.000
role = target_support_content
```

qwen3 的 L33:topH4 也出现混合机制：

```text
L33:attn_out:topH4 / natural_donor / target_record_line
n = 6
source target contribution = 9.563
source route suppression contribution = 1.140
remove target drop = 0.583
route release = 0.625
role = mixed_target_support_and_route_guard
```

GLM4 confirm：

```text
L35:attn_out:H29 / natural_recipient / records_all
attention mass = 0.622
source target contribution = 0.986
remove target drop = 0.250
route release = 0.000
role = target_support_content
```

GLM4 证据弱于 qwen3 和 DS7B，并且 L34:H4 多数表现为 high Q/K attention but weak causal content（高注意力但弱因果内容）。

DS7B confirm 最强单头：

```text
L22:attn_out:H24 / natural_donor / target_record_line
attention mass = 0.817
source target contribution = 0.281
source route suppression contribution = 0.320
remove target drop = 1.250
route release = 0.000
margin drop = 0.922
top1 loss = 0.000
role = target_support_content
```

DS7B confirm 最强 headset：

```text
L22:attn_out:topH4 / natural_donor / records_all
n = 6
attention mass = 0.427
source target contribution = 1.918
source route suppression contribution = 0.922
remove target drop = 1.031
route release = 0.177
margin drop = 0.803
top1 loss = 0.000
role = target_support_content
```

Phase750 的 H1 得到延续：

```text
L22:attn_out:H1 / natural_recipient / target_record_line
n = 6
attention mass = 0.384
source target contribution = 1.602
source route suppression contribution = 0.368
remove target drop = 0.708
route release = 0.188
top1 loss = 0.167
role = target_support_content
```

出现一个 L23 候选：

```text
L23:attn_out:H6 / natural_recipient / relation_tokens
attention mass = 0.600
source target contribution = 3.754
remove target drop = 0.688
route release = 0.125
role = target_support_content
```

### 理论进展

Phase751 把图谱从：

```text
head erase proves necessity
```

推进到：

```text
source-group restricted V/O contribution removal proves source-content path
```

现在可以更具体地描述一条自然路径：

```text
source token group
  -> attention head Q/K selects source
  -> V vector carries content
  -> O projection writes residual
  -> target logit / route competition changes
```

这说明 suppressor（抑制器）不是唯一核心单元。更稳的说法是：

```text
自然语言生成中存在 source-conditioned writer / supporter / guard path；
suppressor 是 route competition 中的关键功能，但最强自然头也可能首先表现为 target support。
```

### 严格审视

1. 当前仍是 head 级和 source group 级证据，不是 neuron（神经元）级证据。
2. source group 来自人工 token span 规则，可能不等同于模型内部真实分组。
3. source contribution removal 仍是人工干预，可能存在 off-manifold effect（离流形效应）。
4. qwen3 和 DS7B 的贡献值尺度不同，不能直接比较绝对值。
5. DS7B L22:H24 是新出现的强候选，需要确认它是不是稳定 writer，而不是动态筛选偏差。
6. L23:H6 很有价值，但当前 n=1，不能作为稳定结论。

### 下一阶段

Phase 752 建议：

```text
Natural Writer Stability and Path Chain Validation
自然写入器稳定性与路径链验证
```

任务：

```text
1. 固定 DS7B L22:H24、L22:H1、L22:topH4、L23:H6。
2. 扩展苹果—水果—属性任务中的 object、relation、answer 类型。
3. 判断这些 head 是 object-specific、relation-specific，还是 answer-value-specific。
4. 追踪 L22:H24 写入后，L23、MLP、readout 是否继续使用同一路径。
5. 对比 head removal、source contribution removal、downstream residual patch 是否闭合。
```

阶段性目标是把图谱从单层 source-content path（源内容路径）推进到：

```text
source -> writer head -> downstream carrier/rewriter -> readout competition
```

## Phase 752: 自然写入器稳定性与路径链验证 [2026-06-29 11:25]

### 对上传内容的判断

上传内容方向基本正确：当前“苹果—水果—属性”图谱已经应该从普通 patch（修补）转向自然路径链验证。需要保守收紧的是：attention mass（注意力质量）只能说明 Q/K pattern（查询/键模式）自然看向哪里；source-restricted V/O contribution removal（源限制值/输出贡献移除）才更接近因果证据，但仍然只是 head-source path（注意力头-源路径）层级证据，不等于神经元级完整图谱。

### 脚本

```text
tests/gpt5/phase752_natural_writer_stability_path_chain.py
tests/gpt5/run_phase752_natural_writer_stability_path_chain_round.sh
```

脚本不使用量化方案；三个模型按 qwen3、GLM4、DS7B 顺序运行，并使用 `--hard-exit-after-model` 避免显存叠加。

### 命令

```bash
python -m py_compile tests/gpt5/phase752_natural_writer_stability_path_chain.py
```

```bash
tests/gpt5/run_phase752_natural_writer_stability_path_chain_round.sh smoke --max-pairs 1 --top-audits 1 --max-candidates 2 --max-source-groups 3 --top-k-vocab 10 --max-topk-tokens 6 --max-route-classes 4 --log-every 1
```

```bash
tests/gpt5/run_phase752_natural_writer_stability_path_chain_round.sh main --max-pairs 8 --include-extended-relations --top-audits 1 --max-candidates 4 --max-source-groups 5 --top-k-vocab 14 --max-topk-tokens 8 --max-route-classes 5 --log-every 2
```

```bash
tests/gpt5/run_phase752_natural_writer_stability_path_chain_round.sh confirm --max-pairs 12 --include-extended-relations --top-audits 1 --max-candidates 4 --max-source-groups 5 --top-k-vocab 16 --max-topk-tokens 10 --max-route-classes 6 --log-every 3
```

### 原理

固定 Phase 751 找到的候选 head（注意力头），跨 object / relation / answer（对象/关系/答案）扩展测试。对每个 head 和 source group（源组）记录自然 attention（注意力）和 V/O contribution（值/输出贡献），然后只移除该 head 从指定源组写入的贡献，观察 target logit drop（目标词元 logit 下降）、route release（路线释放）、top1 loss（第一名丢失）和 final hidden delta（最终隐藏态扰动）。

固定候选：

```text
qwen3: L33:H15, L33:H23, L32:H11, L32:H0
GLM4: L35:H29, L34:H4, L34:H9
DS7B: L22:H24, L22:H1, L22:H7, L23:H6
```

### 结果

确认轮：

```text
qwen3: 480 rows
GLM4: 360 rows
DS7B: 480 rows
```

汇总文件：

```text
results/glm5_phase752_natural_writer_stability_path_chain/confirm/phase752_cross_model_summary.md
```

qwen3：

```text
natural_donor L32:H11 target_record_line
n=12, relations=6
support rate=0.333
mean target drop=0.062
route guard rate=0.250
final delta=5.809
判断：relation_conditioned_writer
```

GLM4：

```text
natural_donor L35:H29 records_all
n=12, relations=6
support rate=0.167
mean target drop=0.094
route guard rate=0.000
final delta=2.775
判断：weak_or_unstable
```

DS7B：

```text
natural_recipient L22:H24 records_all
n=12, relations=6
support rate=0.917
mean target drop=0.688
route guard rate=0.500
mean route release=0.516
top1 loss=0.167
final delta=14.240
判断：stable_mixed_writer_guard
```

```text
natural_recipient L22:H24 target_record_line
n=12, relations=6
support rate=0.833
mean target drop=0.594
route guard rate=0.417
mean route release=0.484
final delta=12.786
判断：stable_mixed_writer_guard
```

```text
natural_recipient L22:H1 records_all
n=12, relations=6
support rate=0.750
mean target drop=0.474
route guard rate=0.667
mean route release=0.620
final delta=11.413
判断：stable_mixed_writer_guard
```

```text
natural_donor L22:H24 records_all
n=12, relations=6
support rate=0.750
mean target drop=0.401
route guard rate=0.167
mean route release=0.104
final delta=9.938
判断：stable_target_writer
```

### 进展

Phase 752 把 Phase 751 的自然注意力头回溯推进到稳定性验证：

```text
1. DS7B L22:H24 是当前最稳定的自然写入器候选。
2. DS7B L22:H1 更像 target writer + route guard（目标写入器 + 路线守门器）的混合组件。
3. qwen3 的相关 head 更多是 relation-conditioned（关系条件化）或 answer-value-specific（答案值特异）片段。
4. GLM4 没有出现强稳定写入器。
5. DS7B 的 final hidden delta 更大，说明 L22 写入扰动会向后层传播。
```

### 严格问题

```text
1. hidden delta 只证明扰动传播，不证明下游模块正确使用同一路径。
2. source group 仍然是外部 token span 标签，不是模型内部自带语义单元。
3. DS7B 是小模型，内部结构可能存在偏差。
4. qwen3 / GLM4 没有复现 DS7B 的稳定 L22 写入器，机制可能模型特异。
5. 当前仍然是 head-source path 图谱，不是 neuron-level graph（神经元级图谱）。
```

### 理论进展

当前最稳妥的表达：

```text
语言生成不是单个语义向量被直接读出，而是由 source-conditioned writer（源条件化写入器）、route guard（路线守门器）、downstream carrier（下游承载器）和 readout competition（读出竞争）共同形成的条件化路径。
```

DS7B 局部证据已经支持：

```text
source record/value
-> L22:H24 / L22:H1
-> downstream residual perturbation
-> route competition changes
-> final answer logit changes
```

但还没有证明：

```text
source -> writer -> exact downstream carrier -> exact rewriter -> exact readout
```

### 下一步

Phase 753：

```text
Downstream Carrier Closure for L22 Writer Path
L22 写入器路径的下游承载闭合验证
```

任务：

```text
1. 固定 DS7B L22:H24 和 L22:H1。
2. 移除它们的 source contribution 后，在 L23-L27 的 residual / attn / MLP 位置尝试恢复。
3. 如果某个下游位置能恢复 target logit，同时压回 route release，说明它更可能是 carrier / rewriter。
4. 对比 target_record_line、target_value_tokens、records_all 三种源组，判断下游承载是否依赖源粒度。
5. 不扩大模型范围，先把 DS7B 的自然路径链闭合，再考虑跨模型复现。
```

## Phase 755: 跨语义域路线不变量图谱第一版 [2026-06-29 12:17]

### 对上传内容的判断

上传内容中“必须从苹果-水果局部图谱扩展到植物、动物、物品、工具、抽象概念”的判断正确。当前苹果-水果图谱只能说明 fruit-domain（水果域）内部的 route competition（路线竞争），无法判断 format / echo / suppression route 是否跨语义域复用，也无法判断 DS7B L22:H24 / L22:H1 是否是跨域 writer（写入器）。

需要收紧的是：Graph Atlas v1.0 还不能说已经完成 language graph（语言全图）。本轮只能称为第一版 Cross-Domain Route Invariance Atlas（跨语义域路线不变量图谱）。

### 脚本

```text
tests/gpt5/phase755_cross_domain_route_invariance_atlas.py
tests/gpt5/run_phase755_cross_domain_route_invariance_atlas_round.sh
```

脚本使用 bf16，不使用量化。由于需要 `output_attentions` 和 V/O contribution hook（值/输出贡献钩子），实际使用 eager attention。

### 命令

```bash
python -m py_compile tests/gpt5/phase755_cross_domain_route_invariance_atlas.py
```

```bash
tests/gpt5/run_phase755_cross_domain_route_invariance_atlas_round.sh smoke --max-pairs 2 --max-candidates 1 --max-source-groups 2 --top-k-vocab 8 --max-topk-tokens 5 --max-route-classes 4 --log-every 1
```

```bash
tests/gpt5/run_phase755_cross_domain_route_invariance_atlas_round.sh main --max-pairs 30 --max-candidates 3 --max-source-groups 3 --top-k-vocab 16 --max-topk-tokens 10 --max-route-classes 6 --log-every 5
```

```bash
tests/gpt5/run_phase755_cross_domain_route_invariance_atlas_round.sh confirm --max-pairs 60 --max-candidates 3 --max-source-groups 3 --top-k-vocab 18 --max-topk-tokens 12 --max-route-classes 6 --log-every 10
```

实际有效 pair 数为 58。

### 原理

测试域：

```text
fruit, animal, plant, object, tool, abstract
```

测试关系：

```text
category, color, taste, shape, edible, grows_on_tree
```

每个样本构造 explicit_profile（显式事实）和 conflict_profile（冲突事实），主要观察 explicit_profile 下的自然路线结构。

测试两类现象：

```text
1. route profile（路线轮廓）
   统计 top token class，并计算不同 domain 的 route profile JS divergence。

2. fixed head/source contribution removal（固定注意力头/源贡献移除）
   qwen3: L33:H15, L33:H23, L32:H11
   GLM4: L35:H29, L34:H4, L34:H9
   DS7B: L22:H24, L22:H1, L22:H7
```

### 结果

确认轮：

```text
qwen3: 580 rows, 58 route observations, 522 source removals
GLM4: 580 rows, 58 route observations, 522 source removals
DS7B: 580 rows, 58 route observations, 522 source removals
```

结果文件：

```text
results/glm5_phase755_cross_domain_route_invariance_atlas/confirm/phase755_cross_model_summary.md
```

route profile：

```text
qwen3 mean pairwise domain JS = 0.0433
top class counts = donor_answer 45, format_or_schema 13

GLM4 mean pairwise domain JS = 0.0970
top class counts = donor_answer 58

DS7B mean pairwise domain JS = 0.0542
top class counts = donor_answer 34, format_or_schema 18, echo_object_or_relation 3, punctuation_or_stop 2, other_vocab 1
```

qwen3：

```text
L33:H23 records_all
n=58, domains=6
support rate=0.224
mean target drop=0.110
route guard rate=0.241
mean route release=0.108
判断：domain_specific_or_weak
```

GLM4：

```text
L35:H29 records_all
n=58, domains=6
support rate=0.034
mean target drop=0.023
route guard rate=0.069
mean route release=0.055
判断：domain_specific_or_weak
```

DS7B：

```text
L22:H24 records_all
n=58, domains=6
support rate=0.862
mean target drop=0.528
route guard rate=0.310
mean route release=0.200
top1 loss=0.121
判断：cross_domain_writer_candidate
```

```text
L22:H1 records_all
n=58, domains=6
support rate=0.810
mean target drop=0.554
route guard rate=0.328
mean route release=0.189
top1 loss=0.121
判断：cross_domain_writer_candidate
```

```text
L22:H24 target_record_line
n=58, domains=6
support rate=0.810
mean target drop=0.448
route guard rate=0.328
mean route release=0.223
判断：cross_domain_writer_candidate
```

```text
L22:H1 target_record_line
n=58, domains=6
support rate=0.672
mean target drop=0.435
route guard rate=0.379
mean route release=0.204
判断：cross_domain_mixed_writer_guard
```

```text
L22:H24 target_value_tokens
n=58, domains=6
support rate=0.534
mean target drop=0.268
route guard rate=0.379
mean route release=0.218
判断：cross_domain_route_guard_candidate
```

### 进展

Phase 755 将 Phase 752 的 DS7B L22:H24 / L22:H1 从 fruit-domain（水果域）推进到六个 domain 的跨域验证。结果显示：

```text
DS7B L22:H24 / L22:H1 不只是苹果-水果局部 writer。
它们在 fruit / animal / plant / object / tool / abstract 上都有稳定 target-support effect。
```

但 qwen3 和 GLM4 没有同等复现，所以当前结论必须写成：

```text
DS7B-local cross-domain writer candidate
```

不能写成：

```text
cross-model universal invariant
```

### 严格问题

```text
1. route JS 低只能说明路线轮廓相似，不能单独证明数学不变量。
2. DS7B 强成立，qwen3 / GLM4 不强。
3. domain 词表仍小，abstract domain 尤其粗糙。
4. 显式事实提示不等价于自然知识网络。
5. 当前仍然是 head-source path 图谱，不是 neuron-level atlas。
6. 下游 carrier / rewriter 仍未定位。
```

### 理论进展

当前最稳妥表达：

```text
语言模型中可能存在跨语义域复用的 route competition skeleton（路线竞争骨架）。
DS7B 中 L22:H24 / L22:H1 是目前最强的跨域 writer / guard 候选。
```

DS7B 支持的局部链条：

```text
domain facts
-> records_all / target_record_line
-> L22:H24 / L22:H1
-> target logit support
-> route competition changes
```

### 下一步

Phase 756：

```text
Cross-Domain Writer Control and Downstream Carrier Test
跨域写入器控制组与下游承载验证
```

任务：

```text
1. 固定 DS7B L22:H24 / L22:H1。
2. 加入 same-layer random head、L22:H7、L23:H6 作为控制。
3. 对 records_all / target_record_line 做 source removal + downstream residual restore。
4. 检查 L23-L27 哪些位置能恢复 target logit，同时压回 route release。
5. 如果 L22:H24/H1 强于随机头，且存在可恢复的 downstream carrier，才可以把 Phase 755 的跨域 writer 结果写入 Graph Atlas v1。
```

## Phase 756: 跨域写入器控制组与下游承载验证 [2026-06-29 14:26]

### 附件判断审视

附件对 Phase 755 的收紧判断基本正确：Phase 755 不是语言全图完成，而是把苹果-水果局部图谱推进到六个语义域的 route competition skeleton（路线竞争骨架）候选。最稳妥表达仍然是：

```text
DS7B 出现强跨域 writer / guard 候选；
qwen3 和 GLM4 没有同等复现；
因此不能称为 cross-model universal invariant。
```

附件提出 Phase 756 需要补 random / same-layer control（随机 / 同层控制）和 downstream carrier（下游承载者）验证，这一点是正确的。本轮执行了该任务。

### 生成脚本

```text
tests/gpt5/phase756_cross_domain_writer_control_downstream_carrier.py
tests/gpt5/run_phase756_cross_domain_writer_control_downstream_carrier_round.sh
```

### 执行命令

```bash
python -m py_compile tests/gpt5/phase756_cross_domain_writer_control_downstream_carrier.py
python tests/gpt5/phase756_cross_domain_writer_control_downstream_carrier.py --dry-run --round-name dry --max-pairs 6 --max-candidates 3 --max-source-groups 2 --max-downstream-sites 4

tests/gpt5/run_phase756_cross_domain_writer_control_downstream_carrier_round.sh smoke --max-pairs 2 --max-candidates 1 --max-source-groups 1 --max-downstream-sites 2 --top-k-vocab 8 --max-topk-tokens 5 --max-route-classes 4 --log-every 1

tests/gpt5/run_phase756_cross_domain_writer_control_downstream_carrier_round.sh main --max-pairs 24 --max-candidates 2 --max-source-groups 2 --max-downstream-sites 4 --top-k-vocab 14 --max-topk-tokens 10 --max-route-classes 6 --log-every 4

tests/gpt5/run_phase756_cross_domain_writer_control_downstream_carrier_round.sh confirm --max-pairs 48 --max-candidates 2 --max-source-groups 2 --max-downstream-sites 4 --top-k-vocab 16 --max-topk-tokens 10 --max-route-classes 6 --log-every 8
```

### 测试原理

本轮测试分两步：

```text
1. source removal + same-layer control
   固定 Phase 755 候选头，移除 records_all / target_record_line 到 answer position 的 V/O 源贡献。
   同时加入同层 deterministic control head，比较 target logit drop 和 route release。

2. downstream component restore
   在保持源贡献移除的条件下，把 L23-L27 或对应模型下游层的 attn_out / mlp_out 输出恢复为 base 状态。
   如果恢复某个下游组件能显著恢复 target logit，并压回 route release，说明该组件是 coarse downstream carrier。
```

注意：downstream restore 是整组件、答案位置恢复，不是神经元级充分性证明。

### 客观结果

确认轮结果路径：

```text
results/glm5_phase756_cross_domain_writer_control_downstream_carrier/confirm/phase756_cross_model_summary.md
results/glm5_phase756_cross_domain_writer_control_downstream_carrier/confirm/phase756_cross_model_summary.json
```

控制组基线：

```text
qwen3:
  Phase755 candidates: support=0.146, mean drop=0.058, guard=0.281, release=0.131
  same-layer controls: support=0.083, mean drop=0.034, guard=0.188, release=0.090
  判断：候选略强，但整体弱。

GLM4:
  Phase755 candidates: support=0.042, mean drop=0.002, guard=0.062, release=0.073
  same-layer controls: support=0.000, mean drop=0.007, guard=0.010, release=0.029
  判断：整体弱。

DS7B:
  Phase755 candidates: support=0.771, mean drop=0.449, guard=0.339, release=0.207
  same-layer controls: support=0.073, mean drop=0.004, guard=0.281, release=0.190
  判断：target-support 明显强于控制；route-release 本身并不完全特异。
```

DS7B top writer / guard：

```text
L22:H24 records_all
  n=48, domains=6
  support=0.854, mean target drop=0.500
  guard=0.333, mean release=0.212
  top1 loss=0.146
  判断：cross_domain_writer_guard_candidate

L22:H24 target_record_line
  n=48, domains=6
  support=0.812, mean target drop=0.428
  guard=0.333, mean release=0.224
  top1 loss=0.146
  判断：cross_domain_writer_guard_candidate

L22:H1 records_all
  n=48, domains=6
  support=0.771, mean target drop=0.480
  guard=0.333, mean release=0.188
  top1 loss=0.104
  判断：cross_domain_writer_guard_candidate

L22:H1 target_record_line
  n=48, domains=6
  support=0.646, mean target drop=0.385
  guard=0.354, mean release=0.206
  top1 loss=0.125
  判断：cross_domain_writer_guard_candidate
```

下游恢复结果：

```text
DS7B L22:H24 records_all -> L23:attn_out
  effective restore rate=0.354
  erase drop=0.500
  recovered=0.034
  recovery fraction=0.020
  route release reduced=-0.010
  判断：weak_or_unclear

DS7B L22:H24 target_record_line -> L23:attn_out
  effective restore rate=0.312
  erase drop=0.428
  recovered=0.046
  recovery fraction=0.267
  route release reduced=-0.039
  判断：weak_or_unclear

DS7B L22:H1 target_record_line -> L23:mlp_out
  effective restore rate=0.250
  erase drop=0.385
  recovered=0.052
  recovery fraction=0.197
  route release reduced=-0.044
  判断：weak_or_unclear
```

### 严格结论

```text
1. Phase 756 强化了 DS7B L22:H24 / L22:H1 的跨域 writer / guard 必要性证据。
2. 同层控制头没有 target-support 效果，说明 DS7B 的 target-support 不是普通同层扰动。
3. 但 route-release 不是完全特异：DS7B 控制头也有一定 route release，因此 suppressor / guard 结论必须更谨慎。
4. 下游单组件 restore 不能稳定恢复 target，也经常不能压回 route release。
5. 因此当前只完成 controlled writer evidence，没有完成 downstream carrier closure。
```

### 理论进展

Phase 756 支持的链条：

```text
跨域 records
-> DS7B L22:H24 / L22:H1 source contribution
-> target answer logit support
-> partial route competition shift
```

但没有完成：

```text
L22 writer
-> L23-L27 单一 downstream carrier
-> readout closure
```

这说明 DS7B 的跨域 writer 很可能是真实路径节点，但下游承载不是单组件线性转移，更可能是分布式、多组件、非线性闭合。

### 硬伤

```text
1. 当前恢复的是整组件输出，不是神经元级恢复。
2. 单个下游组件恢复弱，不能证明路径闭合。
3. qwen3 / GLM4 没有同等级复现，结论仍然是 DS7B-local。
4. 显式事实 prompt 仍然偏上下文搬运，不等于自然知识图谱。
5. route release 在控制头上也存在，说明 suppressor 可能是更宽的竞争场现象。
```

### 下一步

Phase 757 应进入同一阶段的下一个收束任务：

```text
Multi-Site Downstream Carrier Closure Test
多组件下游承载闭合测试
```

核心任务：

```text
1. 以 DS7B L22:H24 / L22:H1 为主候选。
2. 对 L23:attn_out、L23:mlp_out、L24:attn_out、L24:mlp_out 做 single-site 与 multi-site restore 对比。
3. 增加 off-path downstream control。
4. 如果多组件恢复显著强于单组件和 off-path control，说明下游承载是分布式路径。
5. 如果多组件仍不能恢复，则说明 bottleneck 可能在 readout threshold / phrase likelihood / generation closure，而不是 L23-L27 component carrier。
```

## Phase 757: 多组件下游承载闭合测试 [2026-06-29 14:47]

### 背景

Phase 756 已经确认：DS7B 的 L22:H24 / L22:H1 在跨域 records source removal 中具有明显 target-support；同层控制头没有同等级 target-support。因此 Phase 757 不再继续扩大 writer 搜索，而是测试一个更靠后的闭合问题：

```text
如果 L22 writer 真的写入了正确答案支持，那么这些支持是否被 L23-L24 的 downstream components 承载？
```

### 脚本

```text
tests/gpt5/phase757_multisite_downstream_carrier_closure.py
tests/gpt5/run_phase757_multisite_downstream_carrier_closure_round.sh
```

### 命令

```bash
python -m py_compile tests/gpt5/phase757_multisite_downstream_carrier_closure.py

tests/gpt5/run_phase757_multisite_downstream_carrier_closure_round.sh smoke \
  --max-pairs 2 --max-candidates 1 --max-source-groups 1 \
  --max-combos 4 --top-k-vocab 8 --max-topk-tokens 5 \
  --max-route-classes 4 --log-every 1

tests/gpt5/run_phase757_multisite_downstream_carrier_closure_round.sh main \
  --max-pairs 24 --max-candidates 2 --max-source-groups 2 \
  --max-combos 8 --top-k-vocab 14 --max-topk-tokens 10 \
  --max-route-classes 6 --log-every 4

tests/gpt5/run_phase757_multisite_downstream_carrier_closure_round.sh confirm \
  --max-pairs 48 --max-candidates 2 --max-source-groups 2 \
  --max-combos 8 --top-k-vocab 16 --max-topk-tokens 10 \
  --max-route-classes 6 --log-every 8
```

### 测试原理

测试分三步：

```text
1. 在 candidate writer 的 source token 上移除 V/O source contribution。
2. 观察 correct target logit drop 与 competing route release。
3. 在移除条件下，恢复下游组件输出：
   single primary site
   same-layer primary pair
   primary_all
   off_path_same_count
```

其中 DS7B 的 primary path 设为：

```text
L23:attn_out
L23:mlp_out
L24:attn_out
L24:mlp_out
```

off-path control 设为：

```text
L25:attn_out
L25:mlp_out
L26:attn_out
L26:mlp_out
```

### 结果文件

```text
results/glm5_phase757_multisite_downstream_carrier_closure/smoke/
results/glm5_phase757_multisite_downstream_carrier_closure/main/
results/glm5_phase757_multisite_downstream_carrier_closure/confirm/
```

确认轮摘要：

```text
results/glm5_phase757_multisite_downstream_carrier_closure/confirm/phase757_cross_model_summary.md
```

### 确认轮客观结果

qwen3：

```text
off_path_control restore rate=0.070, recovered=-0.033
same_layer_primary_pair restore rate=0.055, recovered=-0.076
single_primary_site restore rate=0.052, recovered=-0.029

判断：没有 downstream carrier closure。
```

GLM4：

```text
off_path_control restore rate=0.010, recovered=-0.011
primary_multisite_all restore rate=0.000, recovered=-0.013
same_layer_primary_pair restore rate=0.000, recovered=-0.010
single_primary_site restore rate=0.001, recovered=-0.006

判断：没有 downstream carrier closure。
```

DS7B：

```text
off_path_control restore rate=0.297, recovered=0.104, recovery fraction=0.532
primary_multisite_all restore rate=0.237, recovered=0.052, recovery fraction=0.387
same_layer_primary_pair restore rate=0.148, recovered=0.027
single_primary_site restore rate=0.128, recovered=0.014
```

DS7B top writer 仍然稳定：

```text
L22:H24 records_all
  target drop=0.500
  support rate=0.854
  route release=0.212
  top1 loss=0.146
  role=cross_domain_writer_guard_candidate

L22:H1 records_all
  target drop=0.480
  support rate=0.771
  route release=0.188
  top1 loss=0.104
  role=cross_domain_writer_guard_candidate
```

DS7B top restore：

```text
L22:H24 records_all -> L25/L26 off_path_same_count
  restore rate=0.604
  erase drop=0.500
  recovered=0.262
  recovery fraction=0.614
  route release reduced=-0.147
  role=off_path_control_suspicious

L22:H1 records_all -> L25/L26 off_path_same_count
  restore rate=0.542
  erase drop=0.480
  recovered=0.178
  recovery fraction=0.274
  route release reduced=0.010
  role=off_path_control_suspicious

L22:H24 records_all -> L23/L24 primary_all
  restore rate=0.479
  erase drop=0.500
  recovered=0.115
  recovery fraction=0.139
  route release reduced=-0.020
  role=weak_or_unclear
```

### 严格结论

```text
1. Phase 757 没有证明 L23-L24 primary downstream components 完成闭合。
2. DS7B 的 L22:H24 / L22:H1 writer 仍然稳定，通过 confirm 复现。
3. L23-L24 primary_all 有弱恢复，但恢复幅度明显低于 L25-L26 组合。
4. L25-L26 原本设计为 off-path control，但确认轮中稳定恢复更强，因此不能再当成普通对照。
5. 更合理的解释是：L25-L26 很可能是 source writer 之后的 late carrier / rewrite / washout stage 候选。
6. 但是 L25-L26 恢复 target logit 的同时，经常没有压回 route release，甚至增加 route release，因此它不是严格的“正确路线闭合器”。
```

### 理论进展

Phase 757 把链条从：

```text
source facts
-> L22 writer
-> unknown downstream
```

推进为：

```text
source facts
-> DS7B L22:H24 / L22:H1 writer
-> L23-L24 weak partial carrier
-> L25-L26 stronger late carrier / rewrite candidate
-> route competition still unresolved
```

这说明当前瓶颈不是“找不到 writer”，而是：

```text
writer support 如何被后续层改写、冲洗、竞争，并最终变成 token0 readout。
```

### 硬伤

```text
1. L25-L26 是事后由 off-path control 反转出的候选，必须重新设计测试，不能直接当成机制结论。
2. 当前 restore 是整组件恢复，不是神经元级恢复。
3. target logit recovery 与 route release closure 分离，说明“恢复正确值”和“关闭错误路线”不是同一个机制。
4. qwen3 / GLM4 没有复现 DS7B 的强路径，当前仍是小模型 DS7B-local 结构。
5. 显式 facts prompt 仍然更接近上下文机制，不等于自然知识网络。
```

### 下一步

Phase 758 属于同一阶段，应直接验证：

```text
Late Carrier / Rewrite Relabel Test
晚期承载 / 重写重标定测试
```

核心任务：

```text
1. 把 L25-L26 从 off-path control 改为 late carrier candidate。
2. 增加真正的 late off-path control，例如 L27-L28 或同数量随机组件。
3. 比较 L23-L24、L25-L26、L27-L28 对 target recovery 与 route release closure 的作用。
4. 如果 L25-L26 稳定强于 L23-L24 和真正对照，说明 source writer 后存在晚期重写层。
5. 如果 L25-L26 只恢复 target 但不关闭 route release，则说明下一瓶颈是 readout competition / route suppression matrix。
```

## Phase 758: 晚期承载 / 重写重标定测试 [2026-06-29 15:12]

### 背景

Phase 757 中，原本作为 off-path control 的 L25-L26 在 DS7B 上稳定恢复 target logit，且强于 L23-L24 primary path。因此 Phase 758 不继续把 L25-L26 当作对照，而是重新命名为：

```text
late carrier / rewrite candidate
晚期承载 / 重写候选
```

核心问题：

```text
L25-L26 是真正的晚期承载 / 重写阶段，还是只是普通后段扰动？
```

### 脚本

```text
tests/gpt5/phase758_late_carrier_rewrite_relabel.py
tests/gpt5/run_phase758_late_carrier_rewrite_relabel_round.sh
```

### 命令

```bash
python -m py_compile tests/gpt5/phase758_late_carrier_rewrite_relabel.py

python tests/gpt5/phase758_late_carrier_rewrite_relabel.py \
  --dry-run --round-name dry --max-pairs 3 --max-combos 10

tests/gpt5/run_phase758_late_carrier_rewrite_relabel_round.sh smoke \
  --max-pairs 2 --max-candidates 1 --max-source-groups 1 \
  --max-combos 6 --top-k-vocab 8 --max-topk-tokens 5 \
  --max-route-classes 4 --log-every 1

tests/gpt5/run_phase758_late_carrier_rewrite_relabel_round.sh main \
  --max-pairs 24 --max-candidates 2 --max-source-groups 2 \
  --max-combos 12 --top-k-vocab 14 --max-topk-tokens 10 \
  --max-route-classes 6 --log-every 4

tests/gpt5/run_phase758_late_carrier_rewrite_relabel_round.sh confirm \
  --max-pairs 48 --max-candidates 2 --max-source-groups 2 \
  --max-combos 12 --top-k-vocab 16 --max-topk-tokens 10 \
  --max-route-classes 6 --log-every 8
```

### 测试设计

Phase 758 把下游 restore 分成四类：

```text
primary path:
  DS7B L23-L24

late candidate:
  DS7B L25-L26

primary + late joint:
  DS7B L23-L26

true late control:
  DS7B L27
```

判断标准：

```text
1. 如果 late candidate 强于 primary path，说明 Phase 757 的 off-path 不是普通对照。
2. 如果 primary + late joint 最强，说明 L23-L24 与 L25-L26 可能组成连续重写链。
3. 如果 target recovery 上升但 route release 没有下降，则只能叫 target rewrite，不能叫 route closure。
```

### 结果文件

```text
results/glm5_phase758_late_carrier_rewrite_relabel/smoke/
results/glm5_phase758_late_carrier_rewrite_relabel/main/
results/glm5_phase758_late_carrier_rewrite_relabel/confirm/
```

确认轮摘要：

```text
results/glm5_phase758_late_carrier_rewrite_relabel/confirm/phase758_cross_model_summary.md
```

### 确认轮客观结果

qwen3：

```text
late_candidate_all restore rate=0.070, recovered=-0.033
primary_multisite_all restore rate=0.055, recovered=-0.076
primary_plus_late_all restore rate=0.060, recovered=-0.140

判断：没有稳定 late rewrite 证据。
```

GLM4：

```text
late_candidate_all restore rate=0.010, recovered=-0.011
primary_multisite_all restore rate=0.000, recovered=-0.013
primary_plus_late_all restore rate=0.000, recovered=-0.020

判断：没有稳定 late rewrite 证据。
```

DS7B baseline：

```text
late_candidate_all restore rate=0.297, recovered=0.104, recovery fraction=0.532, release reduced=-0.012
primary_multisite_all restore rate=0.237, recovered=0.052, recovery fraction=0.387, release reduced=0.011
primary_plus_late_all restore rate=0.378, recovered=0.145, recovery fraction=0.666, release reduced=-0.003
same_layer_late_candidate_pair restore rate=0.228, recovered=0.056
same_layer_primary_pair restore rate=0.148, recovered=0.027
true_late_control restore rate=0.193, recovered=0.022, release reduced=0.048
```

DS7B writer 仍然稳定：

```text
L22:H24 records_all
  target drop=0.500
  support rate=0.854
  route release=0.212
  role=cross_domain_writer_guard_candidate

L22:H1 records_all
  target drop=0.480
  support rate=0.771
  route release=0.188
  role=cross_domain_writer_guard_candidate
```

DS7B top restore：

```text
L22:H24 records_all -> L23-L26 primary_plus_late_all
  restore rate=0.792
  erase drop=0.500
  recovered=0.363
  recovery fraction=0.739
  release reduced=-0.107
  role=primary_late_joint_target_candidate

L22:H24 target_record_line -> L23-L26 primary_plus_late_all
  restore rate=0.771
  erase drop=0.428
  recovered=0.285
  recovery fraction=0.734
  release reduced=-0.055
  role=primary_late_joint_target_candidate

L22:H24 records_all -> L26 late_attn+mlp
  restore rate=0.625
  erase drop=0.500
  recovered=0.203
  recovery fraction=0.419
  release reduced=-0.036
  role=late_target_rewrite_candidate

L22:H24 records_all -> L25-L26 late_candidate_all
  restore rate=0.604
  erase drop=0.500
  recovered=0.262
  recovery fraction=0.614
  release reduced=-0.147
  role=late_target_rewrite_candidate
```

### 严格结论

```text
1. DS7B 上 L25-L26 的 late candidate 证据被确认，强于 L23-L24 primary path。
2. L23-L26 joint restore 最强，说明 L23-L24 与 L25-L26 更像连续链，而不是互斥候选。
3. 但恢复 target logit 不等于完成 route closure。
4. 多数关键 restore 的 release reduced 为负或接近 0，说明错误路线 / 竞争路线没有被同步压回。
5. true late control L27 也有少量 suspicious recovery，因此后段 residual / readout 区域存在宽泛扰动敏感性。
6. 结论只能写成：DS7B 存在 late target rewrite 候选，不能写成已经找到完整 suppressor closure。
```

### 机制进展

当前更稳妥的链条是：

```text
source facts
-> DS7B L22:H24 / L22:H1 writer / guard
-> L23-L24 weak primary carrier
-> L25-L26 late target rewrite
-> L23-L26 joint target recovery
-> route competition unresolved
-> token0 readout closure 未完成
```

这个结果对语言编码机制的意义是：

```text
正确值不是一次写入后直接读出；
它会在后续层经历至少一次晚期重写。
而“正确值增强”和“竞争路线关闭”是可分离机制。
```

### 硬伤

```text
1. 仍然是整组件级 restore，不是神经元级图谱。
2. DS7B 结果强，qwen3 / GLM4 没有复现，不能说跨模型不变量。
3. L27 control 有少量恢复，说明越靠近读出端，restore 可能混入通用 readout perturbation。
4. 显式 facts prompt 仍然偏上下文搬运，不等于自然知识网络。
5. route release closure 没有完成，因此 suppressor matrix 仍未真正闭合。
```

### 下一步

Phase 759 仍属于同一阶段，但应该换问题：

```text
Late Rewrite vs Route Suppression Matrix Split
晚期重写与路线抑制矩阵分离测试
```

任务：

```text
1. 固定 DS7B L22:H24 / L22:H1 与 L25-L26 late rewrite。
2. 分别测 target recovery 与 route release closure，不再混成一个指标。
3. 找出哪些组件只恢复 correct target，哪些组件负责关闭 format / echo / wrong-category route。
4. 如果能找到 route release closure 组件，才进入 suppressor matrix 图谱。
5. 如果找不到，说明最终瓶颈在 token0 readout competition / phrase likelihood。
```

## Phase 759: 晚期重写与路线抑制矩阵分离测试 [2026-06-29 15:16]

### 背景

Phase 758 已经确认：DS7B 的 L25-L26 是 late target rewrite 候选，L23-L26 joint restore 最强。但 Phase 758 同时显示：

```text
target recovery 上升
不等于
route release closure 完成
```

因此 Phase 759 不再运行新模型，而是对 Phase 758 confirm 的 JSONL 结果做离线重分析，把两个指标拆开：

```text
1. target recovery
2. route suppression / route release closure
```

### 脚本

```text
tests/gpt5/phase759_rewrite_vs_route_suppression_split.py
```

### 命令

```bash
python -m py_compile tests/gpt5/phase759_rewrite_vs_route_suppression_split.py
python tests/gpt5/phase759_rewrite_vs_route_suppression_split.py --round-name confirm
```

### 结果文件

```text
results/glm5_phase759_rewrite_vs_route_suppression_split/confirm/phase759_cross_model_summary.json
results/glm5_phase759_rewrite_vs_route_suppression_split/confirm/phase759_cross_model_summary.md
```

### 测试原理

Phase 759 对 Phase 758 的每条 restore row 重新打分：

```text
target_success:
  erase_target_logit_drop >= 0.20
  target_logit_recovered_by_restore >= 0.10
  target_recovery_fraction >= 0.25

route_success:
  erase_total_positive_route_release >= 0.10
  route_release_reduced_by_restore >= 0.05
  restored route release 确实下降
```

然后按 combo_kind 和 writer/combo 聚合。

### 客观结果

qwen3：

```text
late_candidate_all:
  target rate=0.073
  route rate=0.268
  recovered=-0.033
  role=weak_split_signal

primary_plus_late_all:
  target rate=0.060
  route rate=0.323
  recovered=-0.140
  role=route_suppression_only_candidate

判断：没有 target rewrite，只有弱 route-side 信号。
```

GLM4：

```text
late_candidate_all:
  target rate=0.010
  route rate=0.159
  recovered=-0.011
  role=weak_split_signal

primary_plus_late_all:
  target rate=0.000
  route rate=0.161
  recovered=-0.020
  role=weak_split_signal

判断：没有 target rewrite，也没有强 route closure。
```

DS7B：

```text
late_candidate_all:
  target rate=0.315
  route rate=0.250
  recovered=0.104
  route reduced=-0.012
  role=weak_split_signal

primary_multisite_all:
  target rate=0.250
  route rate=0.289
  recovered=0.052
  route reduced=0.011
  role=weak_split_signal

primary_plus_late_all:
  target rate=0.385
  route rate=0.302
  recovered=0.145
  route reduced=-0.003
  role=joint_target_and_route_candidate

true_late_control:
  target rate=0.206
  route rate=0.323
  recovered=0.022
  route reduced=0.048
  role=weak_split_signal
```

DS7B 关键单项：

```text
L22:H24 records_all -> L23-L26 primary_plus_late_all
  target rate=0.833
  route rate=0.208
  recovered=0.363
  route reduced=-0.107
  role=target_rewrite_only_candidate

L22:H24 records_all -> L25-L26 late_candidate_all
  target rate=0.667
  route rate=0.104
  recovered=0.262
  route reduced=-0.147
  role=target_rewrite_only_candidate

L22:H24 records_all -> L26 late_attn+mlp
  target rate=0.625
  route rate=0.208
  recovered=0.203
  route reduced=-0.036
  role=target_rewrite_only_candidate

L22:H1 target_record_line -> L23-L26 primary_plus_late_all
  target rate=0.583
  route rate=0.312
  recovered=0.207
  route reduced=0.009
  role=joint_target_and_route_candidate
```

### 严格结论

```text
1. Phase 759 支持 target recovery 和 route suppression 是可分离机制。
2. DS7B L22:H24 + L25-L26 / L23-L26 更像 target rewrite path，不是强 route suppression path。
3. L22:H1 的部分组合有 joint signal，但 route reduced 很小，不能当成完整 suppressor matrix。
4. qwen3 / GLM4 没有同等级 target rewrite 复现。
5. route-side 信号在 qwen3 / GLM4 上也有一些弱出现，说明 route suppression 可能更宽、更分散，不一定跟 target writer 同构。
```

### 当前机制图谱更新

更稳妥的链条应写成双路径：

```text
Path A: target rewrite path
source facts
-> DS7B L22:H24 / L22:H1
-> L23-L24 weak carrier
-> L25-L26 late rewrite
-> target logit recovery

Path B: route suppression path
format / echo / wrong-category route
-> distributed suppression / release field
-> not closed by L25-L26 alone
-> token0 competition remains unresolved
```

### 硬伤

```text
1. Phase 759 是离线重分析，不是新 causal intervention。
2. route_success 阈值是人为设定，不能当作自然机制边界。
3. 仍然是组件级，不是神经元级。
4. DS7B-local 结构明显，不能上升为跨模型不变量。
5. 显式 prompt 仍然不等于自然知识网络。
```

### 下一步

阶段性目标已经从“找 late rewrite”推进到“双路径拆分”。下一阶段应转入：

```text
Route Suppression Matrix Atlas
路线抑制矩阵图谱
```

核心任务：

```text
1. 不再只看 correct target logit。
2. 单独构造 format / echo / wrong-category / generic-answer route tokens。
3. 对每类 route 做 source writer、late rewrite、readout-layer restore / ablation。
4. 寻找只压制错误路线、不增强正确值的组件。
5. 如果找到稳定组件，才进入全局 suppressor matrix。
```

## Phase 760: 路线抑制矩阵图谱测试 [2026-06-29 16:12]

### 触发问题

Phase 756-759 已经把机制收紧为双路径：

```text
Path A: target rewrite path
source facts -> DS7B L22:H24 / L22:H1 -> L23-L24 weak carrier -> L25-L26 late rewrite -> target logit recovery

Path B: route suppression path
format / echo / wrong-category / generic route -> distributed suppression field -> token0 readout competition
```

Phase 760 不再只看 correct target logit，而是把竞争路线拆成显式矩阵：

```text
contrast_answer
object_relation_echo
other_record_value
format_schema
generic_answer
top_non_target
top_class:* dynamic route classes
```

### 脚本与命令

```bash
tests/gpt5/phase760_route_suppression_matrix_atlas.py
tests/gpt5/run_phase760_route_suppression_matrix_atlas_round.sh

python -m py_compile tests/gpt5/phase760_route_suppression_matrix_atlas.py

bash tests/gpt5/run_phase760_route_suppression_matrix_atlas_round.sh smoke \
  --max-pairs 2 --max-candidates 1 --max-source-groups 1 --max-combos 4 \
  --top-k-vocab 8 --max-topk-tokens 5 --max-dynamic-route-classes 3 --log-every 1

bash tests/gpt5/run_phase760_route_suppression_matrix_atlas_round.sh main \
  --max-pairs 24 --max-candidates 2 --max-source-groups 2 --max-combos 10 \
  --top-k-vocab 14 --max-topk-tokens 8 --max-dynamic-route-classes 5 --log-every 4

bash tests/gpt5/run_phase760_route_suppression_matrix_atlas_round.sh confirm \
  --max-pairs 48 --max-candidates 2 --max-source-groups 2 --max-combos 10 \
  --top-k-vocab 16 --max-topk-tokens 10 --max-dynamic-route-classes 5 --log-every 8
```

模型顺序为 qwen3 -> GLM4 -> DS7B，每个模型使用 `--hard-exit-after-model`，未使用量化。

### 输出

```text
results/glm5_phase760_route_suppression_matrix_atlas/smoke/
results/glm5_phase760_route_suppression_matrix_atlas/main/
results/glm5_phase760_route_suppression_matrix_atlas/confirm/
results/glm5_phase760_route_suppression_matrix_atlas/confirm/phase760_cross_model_summary.md
results/glm5_phase760_route_suppression_matrix_atlas/confirm/phase760_cross_model_summary.json
```

### 测试原理

对每个样本：

```text
1. 捕获 answer position 的 baseline logits。
2. 移除候选 writer head 来自 source group 的 V/O contribution。
3. 测量 target logit drop 和各 route group release。
4. 在移除基础上恢复 L23-L24、L25-L26、L23-L26 或控制层组件。
5. 分别计算：
   target recovered = erase target drop - restored target drop
   route reduced = erase route release - restored route release
```

### 关键结果

```text
qwen3:
  route-only 信号集中在少量 fruit/edible wrong-answer route，n 很小，target_recovered 多为负。

GLM4:
  确认轮后 route-only 信号衰减到约 0.09，全部 weak_or_unclear。

DS7B:
  L22:H24 没有成为 route suppression matrix 顶部节点。
  L22:H24 更多出现在 generic_answer route，role_guess 仍为 weak_or_unclear。
  L22:H1 有部分 recipient_answer route 抑制候选：
    route_only_success_rate ≈ 0.333
    mean_route_reduced ≈ 0.056-0.076
    target_recovered 为负或很弱。
  L22:H9 / H14 同层控制头也出现类似甚至更强 route-only 信号。
```

### 严格结论

```text
1. Phase 760 支持 target rewrite 和 route suppression 分离。
2. DS7B L22:H24 + L25-L26 的 target rewrite 证据仍强于 route suppression 证据。
3. route suppression 不是由 L25-L26 单独闭合。
4. route-only 信号更多表现为分布式、路线类别相关、控制头也可出现的现象。
5. 目前不能说已经找到 global suppressor matrix。
```

### 硬伤

```text
1. route group 仍由 first token 和 top-k token 构造，不能保证覆盖真实路线全貌。
2. 部分 route-only cell 的 target drop 为负，不能直接支持 writer-path suppressor。
3. DS7B 的 route-only 顶部节点出现 H9/H14 控制头，削弱了 H24/H1 特异性。
4. 仍然是 head/component 级，不是 neuron/channel 级机制。
5. 小模型内部结构可能偏移，不能直接上升为大模型语言机制不变量。
```

### 当前理论进展

当前“苹果-水果-跨域路线图谱”应更新为：

```text
source facts
  -> target writer / rewrite path
      DS7B L22:H24/H1 -> L23-L26 -> target/generic-answer support

source facts
  -> route competition field
      wrong-answer / semantic-value / format / echo routes
      -> distributed suppression / release components
      -> not closed by L22:H24 or L25-L26 alone
```

语言编码机制更像：

```text
多路线竞争场 + 局部写入器 + 分布式抑制矩阵 + 读出端阈值闭合
```

而不是：

```text
单个 writer head -> 单个 suppressor head -> 正确输出
```

### 下一步

```text
Phase 761: Route Suppression Source-Target Disentanglement
路线抑制的源-目标解耦测试
```

核心任务：

```text
1. 对 route-only cell 增加反向验证：
   只移除 route token 的源贡献，看是否能复现 route release。
2. 对 H1/H9/H14/H24 做同层控制矩阵：
   判断 route suppression 是特异 head，还是 L22 层级场效应。
3. 把 target drop 为负的样本单独分离：
   不允许它们直接支持 writer-path suppressor。
4. 如果 route suppression 仍分散，应转向 source-token route graph，而不是继续找单个 suppressor。
```

## Phase 761: 路线源贡献与目标源贡献拆分测试 [2026-06-29 16:49]

### 触发问题

Phase 760 已经证明 target rewrite 与 route suppression 不能简单合并，但仍有一个关键混淆：

```text
路线释放到底来自 target source removal，
还是来自 route-token / format / echo source removal 本身？
```

如果 route-source removal 可以在不降低 target logit 的情况下稳定释放错误路线，说明路线竞争场有独立源结构；如果现象主要伴随 target boost artifact 或控制头同样出现，则不能称为全局 suppressor。

### 脚本与命令

```bash
tests/gpt5/phase761_route_source_target_disentanglement.py
tests/gpt5/run_phase761_route_source_target_disentanglement_round.sh

python -m py_compile tests/gpt5/phase761_route_source_target_disentanglement.py

python tests/gpt5/phase761_route_source_target_disentanglement.py \
  --dry-run --round-name smoke --max-pairs 2 --max-candidates 1 --max-route-source-groups 3

tests/gpt5/run_phase761_route_source_target_disentanglement_round.sh smoke \
  --max-pairs 2 --max-candidates 1 --controls-per-candidate 1 \
  --max-route-source-groups 3 --max-total-source-groups 6 --log-every 1

tests/gpt5/run_phase761_route_source_target_disentanglement_round.sh main \
  --max-pairs 24 --max-candidates 2 --controls-per-candidate 1 \
  --max-route-source-groups 5 --max-total-source-groups 9 --log-every 4

tests/gpt5/run_phase761_route_source_target_disentanglement_round.sh confirm \
  --max-pairs 48 --max-candidates 2 --controls-per-candidate 1 \
  --max-route-source-groups 5 --max-total-source-groups 9 --log-every 8
```

所有模型按 qwen3 -> GLM4 -> DS7B 顺序执行，均使用 BF16 eager、无量化、`--hard-exit-after-model`，每个模型结束后释放 GPU 显存。

### 测试原理

对同一个 candidate head，分别移除不同源位置贡献：

```text
target_record_line
target_value_tokens
records_all
route_src:any_route
route_src:contrast_answer
route_src:recipient_answer
route_src:other_record_value
route_src:format_schema
route_src:object_relation_echo
```

然后分别测量：

```text
target_logit_drop = base_target_logit - after_target_logit
route_release = after_route_max_logit - base_route_max_logit
margin_drop = (base_target - base_route) - (after_target - after_route)
```

解释规则：

```text
target_logit_drop > 0: 源贡献支持目标答案。
target_logit_drop < 0: 移除后目标反而增强，属于 target boost artifact 风险。
route_release > 0: 某类竞争路线释放。
route_source_release_without_target_drop: 可能是路线源证据。
negative_target_drop_route_artifact: 不能作为 suppressor 证据。
```

### 生成结果

```text
results/glm5_phase761_route_source_target_disentanglement/smoke/
results/glm5_phase761_route_source_target_disentanglement/main/
results/glm5_phase761_route_source_target_disentanglement/confirm/
results/glm5_phase761_route_source_target_disentanglement/confirm/phase761_cross_model_summary.md
results/glm5_phase761_route_source_target_disentanglement/confirm/phase761_cross_model_summary.json
```

确认轮规模：

```text
qwen3:      48 pairs, 17812 rows, 16104 route-source cells
GLM4:       48 pairs, 18712 rows, 16988 route-source cells
DS7B:       48 pairs, 19348 rows, 17588 route-source cells
```

### 客观结果

```text
qwen3:
  route_token_source -> format / top_non_target 的 route_release_rate 约 0.33-0.35，
  但 target_boost_rate 约 0.22，且 top cells 多为 n=1 或同层 control head。
  H15 的 records_all / target_value_tokens 有较高 route release，
  但大量属于 negative_target_drop_route_artifact。

GLM4:
  确认轮中所有 source family 的 route_release_rate 很低。
  最高大约 0.027，全部 weak_or_unclear。
  没有支持稳定 route-source suppressor。

DS7B:
  route_token_source -> recipient_answer 的 route_release_rate 约 0.444，
  但 target_boost_rate 约 0.544，严重污染。
  L22:H24 route_src:any_route -> recipient_answer 在 n=9 上 route_release_rate 0.667，
  但 target_boost_rate 0.556，属于 artifact。
  L22:H1 target_value_tokens -> object_relation_echo 在 n=48 上 route_release_rate 0.50，
  target_drop_rate 0.583，说明更像 target-source writer / mixed route effect。
  L22:H9 / H14 控制头仍出现强 route release artifact。
```

### 严格结论

```text
1. Phase 761 没有找到稳定的 route-token-source-only suppressor。
2. qwen3 和 DS7B 的若干路线释放现象主要被 target boost artifact 和低样本强信号污染。
3. GLM4 基本给出负结果，说明该现象不具备跨模型稳健性。
4. DS7B L22:H24 / L22:H1 仍不能解释为全局 suppressor。
5. 目标源贡献与路线源贡献确实可分开测量，但分开后没有出现清晰单点闭合。
```

### 理论进展

Phase 761 支持以下更谨慎的结构：

```text
target rewrite path:
  target source -> writer head -> late rewrite -> target logit

route competition path:
  route / format / echo tokens -> distributed route field -> readout competition
```

但当前结果不支持：

```text
route token source -> single head suppressor -> universal route suppression
```

更合理的解释是：

```text
路线抑制不是单个 head 的属性，
而是 source-token group、attention transport、MLP rewrite、late readout geometry 共同形成的分布式场。
```

### 问题和硬伤

```text
1. route source group 仍然由 token id 匹配构造，存在粗粒度问题。
2. route group 仍然是 top-k / first-token 近似，不是完整短语路线。
3. target boost artifact 在 qwen3 和 DS7B 中非常明显。
4. 同层控制头仍能产生相似现象，说明 head 特异性不足。
5. 小模型结构可能有压缩偏差，不能直接推断大模型机制。
```

### 下一步

Phase 762 不应继续寻找单点 global suppressor，而应转向：

```text
Source-Token Route Graph Atlas
```

具体任务：

```text
1. 把源位置从粗 source_group 拆成逐 token 节点。
2. 对每个 source token 建立它影响的 route class 分布。
3. 区分 target-support token、format token、echo token、competitor token。
4. 观察是否存在稳定的 source-token -> route-class 边，而不是 head -> route-class 单点。
5. 如果逐 token 图谱仍混杂，再进入 phrase-level route graph。
```

## Phase 762: 语义—数值接口与因果纤维图谱确认测试 [2026-06-29 17:18]

### 本阶段判断

用户上传内容的核心判断基本正确：当前最大瓶颈不是继续寻找单个 head（注意力头）、channel（通道）或 patch（补丁），而是要建立：

```text
语义对象 -> 可测量数值结构 -> 可定位因果结构 -> 可复用图谱节点
```

因此本阶段没有继续寻找 single global suppressor（单一全局抑制器），而是把对象转换成跨关系任务族上的 causal functional fingerprint（因果功能指纹）。

### 生成脚本

```text
tests/gpt5/phase762_semantic_numeric_fiber_atlas.py
tests/gpt5/run_phase762_semantic_numeric_fiber_atlas_round.sh
```

### 测试命令

语法检查：

```bash
python -m py_compile tests/gpt5/phase762_semantic_numeric_fiber_atlas.py
```

冒烟测试：

```bash
python tests/gpt5/phase762_semantic_numeric_fiber_atlas.py \
  --dry-run --round-name smoke --max-pairs 6 --max-candidates 1 --max-source-groups 3

tests/gpt5/run_phase762_semantic_numeric_fiber_atlas_round.sh smoke \
  --max-pairs 6 --max-candidates 1 --controls-per-candidate 1 \
  --max-source-groups 3 --top-k-vocab 8 --max-topk-tokens 5 \
  --max-dynamic-route-classes 3 --log-every 2
```

主测试：

```bash
tests/gpt5/run_phase762_semantic_numeric_fiber_atlas_round.sh main \
  --max-pairs 54 --max-candidates 2 --controls-per-candidate 1 \
  --max-source-groups 5 --top-k-vocab 14 --max-topk-tokens 8 \
  --max-dynamic-route-classes 5 --log-every 9
```

确认测试：

```bash
tests/gpt5/run_phase762_semantic_numeric_fiber_atlas_round.sh confirm \
  --max-pairs 108 --max-candidates 2 --controls-per-candidate 1 \
  --max-source-groups 5 --top-k-vocab 16 --max-topk-tokens 10 \
  --max-dynamic-route-classes 5 --log-every 18
```

脚本逐个运行 qwen3、GLM4 和 DS7B，并保留 `--hard-exit-after-model`，不使用量化方案，采用 bfloat16 eager 路径。

### 测试原理

对象不再被当成一个静态向量，而被定义为：

```text
object causal fiber =
同一对象在多种 relation task 中，
不同 head/source group removal 对 target logit、route release、margin drop、attention mass、direct score 的平均因果效应谱。
```

本阶段构造了 18 个对象、6 个语义域和 6 类关系：

```text
fruit: apple / banana / pear
animal: cat / bird / dog
plant: oak / rose / wheat
object: chair / stone / cup
tool: hammer / knife / scissors
abstract: freedom / time / justice

relations:
category / color / taste / shape / edible / grows_on_tree
```

完整确认轮一共 108 个 object-relation task。对每个对象建立因果纤维向量，然后比较同域对象和异域对象的相似度，并与 first-token embedding baseline（首词元嵌入基线）比较。

### 确认轮结果

结果目录：

```text
results/glm5_phase762_semantic_numeric_fiber_atlas/confirm/
```

核心结果：

```text
qwen3:
  causal NN = 0.556
  embedding NN = 0.667
  causal same = 0.531
  causal diff = -0.122
  causal separation = 0.653
  embedding separation = 0.092

GLM4:
  causal NN = 0.611
  embedding NN = 0.611
  causal same = 0.411
  causal diff = -0.090
  causal separation = 0.501
  embedding separation = 0.048

DS7B:
  causal NN = 0.556
  embedding NN = 0.611
  causal same = 0.214
  causal diff = -0.093
  causal separation = 0.307
  embedding separation = 0.043
```

跨模型 centered object-topology correlation（中心化对象拓扑相关）：

```text
qwen3__GLM4 = 0.344
qwen3__DS7B = 0.292
GLM4__DS7B = 0.287
```

### 严格解释

确认轮支持一个弱正结果：

```text
因果纤维的 same-domain / different-domain 均值分离明显强于 embedding baseline。
```

但它不支持强结论：

```text
语义—数值接口已经闭合。
```

原因是：

```text
1. 最近邻 domain accuracy 没有稳定超过 embedding baseline。
2. 跨模型拓扑相关只有弱到中等强度。
3. DS7B 的信号明显弱于 qwen3 和 GLM4。
4. 当前仍是 head/source-level fingerprint，不是 neuron-level atlas。
5. 对象、关系和答案仍由人工 profile 构造，存在任务模板偏差。
```

### 理论进展

本阶段把“语义对象”的研究单位从：

```text
一个词 / 一个向量 / 一个 head
```

推进为：

```text
对象在关系任务族中的条件化因果功能纤维。
```

更谨慎的结论是：

```text
语义对象不是点，而是一束随任务条件展开的因果效应谱。
```

但这束纤维目前只能提供弱语义域结构，还没有达到全局语义动力系统的可闭合程度。

## Phase 763: 语义因果纤维特征消融审计 [2026-06-29 17:18]

### 生成脚本

```text
tests/gpt5/phase763_semantic_numeric_fiber_ablation.py
```

### 测试命令

```bash
python -m py_compile tests/gpt5/phase763_semantic_numeric_fiber_ablation.py
python tests/gpt5/phase763_semantic_numeric_fiber_ablation.py --round-name confirm
```

### 测试原理

本阶段不重新加载模型，只读取 Phase 762 confirm 的 jsonl 结果，按特征家族重新构造对象纤维：

```text
phase762_exact
target_drop_only
attention_mass_only
direct2_scores_only
direct4_scores_only
route_release_only
margin_drop_only
records_only
object_relation_sources_only
all_relation_collapsed
```

目的不是新增因果干预，而是审计 Phase 762 的弱正结果到底由哪些数值成分支撑。

### 结果

结果目录：

```text
results/glm5_phase763_semantic_numeric_fiber_ablation/confirm/
```

关键结果：

```text
qwen3:
  phase762_exact: NN 0.556, separation 0.653
  attention_mass_only: NN 0.667, separation 0.583
  direct2_scores_only: NN 0.611, separation 0.777
  route_release_only: NN 0.333, separation 0.159
  object_relation_sources_only: NN 0.444, separation 0.198

GLM4:
  phase762_exact: NN 0.611, separation 0.501
  attention_mass_only: NN 0.667, separation 0.673
  direct2_scores_only: NN 0.722, separation 0.831
  route_release_only: NN 0.444, separation 0.316
  object_relation_sources_only: NN 0.056, separation 0.013

DS7B:
  phase762_exact: NN 0.556, separation 0.307
  attention_mass_only: NN 0.667, separation 0.550
  direct2_scores_only: NN 0.667, separation 0.402
  route_release_only: NN 0.667, separation 0.115
  object_relation_sources_only: NN 0.278, separation 0.110
```

跨模型相关：

```text
phase762_exact:
  DS7B__GLM4 = 0.287
  DS7B__qwen3 = 0.292
  GLM4__qwen3 = 0.344

direct2_scores_only:
  DS7B__GLM4 = 0.151
  DS7B__qwen3 = 0.229
  GLM4__qwen3 = 0.262
```

### 严格解释

Phase 763 把 Phase 762 的结论进一步收紧：

```text
语义域结构主要不是由 route_release_only 单独承载。
```

更稳定的支撑来自：

```text
1. attention_mass_only
2. direct target / route score
3. records_only source group
```

而单独使用：

```text
object_tokens + relation_tokens
```

效果很弱，尤其 GLM4 几乎失效。这说明当前“语义对象纤维”更像是：

```text
记录侧条件化读出几何 + 注意力分配质量 + 直接 logit 几何
```

而不是已经定位到对象词元自身或 route release（路线释放）本身。

### 当前硬伤

```text
1. 语义域分离仍可能来自 profile 记录格式，而不是真实自然语义。
2. direct score 很强，但跨模型相关不强，可能是局部读出几何而非全局语义不变量。
3. route_release 单独较弱，说明路线动力系统尚未真正被捕获。
4. object/relation token source 单独较弱，说明“对象词元就是语义入口”的假设仍不成立。
5. 当前仍是小模型结果，存在结构压缩和机制偏移风险。
```

### 下一步

下一阶段不应继续扩大对象数量，而应先做：

```text
Phase 764:
Record-Format Control and Natural-Context Fiber Test
（记录格式控制与自然语境因果纤维测试）
```

核心目标：

```text
1. 保留同一对象和关系，改变 profile 格式，测试 records_only 信号是否稳定。
2. 去掉显式 key-value records，换成自然句子上下文。
3. 比较 causal fiber 在 explicit profile / paraphrase / natural sentence 三种上下文中的一致性。
4. 如果一致，语义—数值接口更可信。
5. 如果不一致，当前图谱主要是格式条件图谱，不是自然语义图谱。
```

## Phase 764: 记录格式控制与自然语境因果纤维测试 [2026-06-29 17:39]

### 本阶段判断

Phase 762-763 已经证明：

```text
语义对象因果纤维存在弱语义域结构；
但最强支撑来自 records / attention / direct score，
不是 object token 本身，也不是 route_release 单独特征。
```

因此 Phase 764 的关键问题是：

```text
这个信号是不是 key-value records 格式造成的？
```

如果只在 `apple.category = fruit` 这种显式格式中成立，那么它更像 format-conditioned atlas（格式条件图谱）；如果在句子行和紧凑自然描述中也成立，才更接近 natural semantic fiber（自然语义纤维）。

### 生成脚本

```text
tests/gpt5/phase764_record_format_natural_context_fiber_test.py
tests/gpt5/run_phase764_record_format_natural_context_fiber_test_round.sh
```

### 测试命令

语法检查：

```bash
python -m py_compile tests/gpt5/phase764_record_format_natural_context_fiber_test.py
```

dry-run：

```bash
python tests/gpt5/phase764_record_format_natural_context_fiber_test.py \
  --dry-run --max-cases 9 --relations category,color,edible
```

冒烟测试：

```bash
tests/gpt5/run_phase764_record_format_natural_context_fiber_test_round.sh smoke \
  --max-cases 9 --max-candidates 1 --controls-per-candidate 1 \
  --max-source-groups 4 --relations category,color,edible --log-every 3
```

冒烟阶段发现 `expanded_candidates` 需要 `include_controls` 和 `control_offset` 参数，已修复并重跑通过。模型路径、三种上下文格式和三模型顺序运行均正常。

主测试：

```bash
tests/gpt5/run_phase764_record_format_natural_context_fiber_test_round.sh main \
  --max-candidates 1 --controls-per-candidate 1 --max-source-groups 4 \
  --relations category,color,edible --log-every 27
```

确认测试：

```bash
tests/gpt5/run_phase764_record_format_natural_context_fiber_test_round.sh confirm \
  --max-candidates 1 --controls-per-candidate 1 --max-source-groups 6 \
  --relations category,color,edible --log-every 27
```

三模型按 qwen3、GLM4、DS7B 顺序运行，并使用 `--hard-exit-after-model`。本阶段不使用量化方案，采用 bfloat16 eager 路径。

### 测试原理

同一批对象和答案，用三种上下文格式表达：

```text
1. key_value:
   apple.category = fruit
   apple.color = red

2. sentence_lines:
   Apple is in the category fruit.
   The color of apple is red.

3. compact_sentence:
   Profile for apple: category fruit; color red; ...
```

然后对同一对象构造跨格式 causal fiber（因果纤维）：

```text
source group:
context_all
target_context_line
target_value_tokens
object_tokens
relation_tokens
question

feature:
target_logit_drop
attention_mass
direct_target_boost
direct_total_route_suppression
```

测试两件事：

```text
1. 每种上下文内部是否仍然有 same-domain > different-domain 的分离。
2. 不同上下文之间，同一个对象 / 同一语义域 / 不同语义域的相似度关系是否稳定。
```

本阶段主测试和确认测试均使用：

```text
18 objects * 3 relations * 3 context formats = 162 cases per model
```

其中 relations 为：

```text
category / color / edible
```

### 确认轮结果

结果目录：

```text
results/glm5_phase764_record_format_natural_context_fiber_test/confirm/
```

上下文内部语义域分离：

```text
qwen3:
  key_value:        NN 0.722, separation 0.687
  sentence_lines:   NN 0.833, separation 0.684
  compact_sentence: NN 0.611, separation 0.681

GLM4:
  key_value:        NN 0.500, separation 0.331
  sentence_lines:   NN 0.556, separation 0.453
  compact_sentence: NN 0.667, separation 0.425

DS7B:
  key_value:        NN 0.778, separation 0.464
  sentence_lines:   NN 0.889, separation 0.597
  compact_sentence: NN 0.667, separation 0.515
```

跨上下文稳定性：

```text
qwen3:
  compact_sentence__key_value:
    same_object = 0.874
    same_domain_other = 0.798
    diff_domain = 0.137
    domain_gap = 0.661

  compact_sentence__sentence_lines:
    same_object = 0.777
    same_domain_other = 0.718
    diff_domain = 0.087
    domain_gap = 0.631

  key_value__sentence_lines:
    same_object = 0.882
    same_domain_other = 0.797
    diff_domain = 0.132
    domain_gap = 0.665

GLM4:
  compact_sentence__key_value:
    same_object = 0.240
    same_domain_other = 0.002
    diff_domain = -0.194
    domain_gap = 0.196

  compact_sentence__sentence_lines:
    same_object = -0.066
    same_domain_other = -0.176
    diff_domain = -0.301
    domain_gap = 0.125

  key_value__sentence_lines:
    same_object = 0.081
    same_domain_other = -0.110
    diff_domain = -0.299
    domain_gap = 0.189

DS7B:
  compact_sentence__key_value:
    same_object = 0.165
    same_domain_other = 0.045
    diff_domain = -0.201
    domain_gap = 0.246

  compact_sentence__sentence_lines:
    same_object = 0.248
    same_domain_other = 0.185
    diff_domain = -0.171
    domain_gap = 0.356

  key_value__sentence_lines:
    same_object = 0.101
    same_domain_other = -0.000
    diff_domain = -0.247
    domain_gap = 0.246
```

### 客观进展

Phase 764 给出一个比 Phase 762 更强、但仍需谨慎的正结果：

```text
语义域结构不是 key-value records 独有。
```

证据是：

```text
1. qwen3 / GLM4 / DS7B 三个模型在 key_value、sentence_lines、compact_sentence 三种上下文中都有正 separation。
2. sentence_lines 往往不弱于 key_value，DS7B 中甚至更强。
3. 跨上下文对比中，同语义域对象普遍高于异语义域对象。
```

因此 Phase 762 中的语义纤维弱信号不能简单归因于 `apple.category = fruit` 这种单一格式。

### 严格解释

但当前仍不能说：

```text
自然语义对象编码机制已经闭合。
```

必须收紧为：

```text
在显式给定事实的多种上下文表述中，
模型形成了跨格式保留的语义域级因果纤维结构。
```

关键限制：

```text
1. qwen3 的跨格式 same-object 稳定性很强，但 GLM4 和 DS7B 只达到弱到中等。
2. 当前更稳的是 domain-level structure，不是 object-level identity closure。
3. 只测试了 category / color / edible 三种关系，还不是完整六关系图谱。
4. 上下文仍然显式给定事实，不是纯 commonsense 自然知识。
5. 特征仍是 head/source 级别，不是 neuron-level 或 parameter-level 图谱。
6. 小模型可能存在结构压缩偏差，不能直接外推到大模型。
```

### 理论进展

Phase 764 把语义—数值接口从：

```text
格式化记录中的因果纤维
```

推进到：

```text
跨表述格式仍部分保持的语义域因果纤维。
```

这说明当前最接近真实语言编码机制的对象不是：

```text
单个语义词元
单个 head
单条 route
```

而是：

```text
在上下文条件下形成的 domain-level causal fiber field
（语义域级因果纤维场）
```

### 下一步

下一阶段应继续同一大阶段，但不要盲目扩大对象数。最关键任务是：

```text
Phase 765:
Commonsense Context and Object-Identity Closure Test
（常识语境与对象身份闭合测试）
```

核心问题：

```text
如果不显式给定 profile，只要求模型使用常识回答，
对象因果纤维是否仍然保持语义域结构和对象身份稳定性？
```

如果成立，说明语义—数值接口开始接近模型内部已有知识结构。

如果不成立，说明当前图谱主要是 context-conditioned retrieval / readout mechanism（上下文条件检索/读出机制），还不是完整自然语义编码机制。

## Phase 765: 常识语境与对象身份闭合测试 [2026-06-29 18:10]

### 本阶段判断

两份新附件的判断基本正确，而且与 Phase 761-764 的结果一致：

```text
静态 semantic graph（语义图谱）不是正确第一目标。
更正确的目标是 context-route-causal-fiber-generation-closure graph
（脉络—路线—因果纤维—生成闭合图谱）。
```

Phase 764 已经证明，在显式给定事实的多种格式中，语义域因果纤维不是 key-value records 独有。但它仍然没有回答：

```text
如果不显式给定 profile（档案事实），
模型只依赖 commonsense（常识）回答，
对象因果纤维是否仍然保留语义域结构和对象身份稳定性？
```

因此 Phase 765 继续同一阶段目标：从显式上下文检索机制，推进到内部常识语义机制的第一轮测试。

### 生成脚本

```text
tests/gpt5/phase765_commonsense_context_identity_closure_test.py
tests/gpt5/run_phase765_commonsense_context_identity_closure_test_round.sh
```

### 测试命令

语法检查：

```bash
python -m py_compile tests/gpt5/phase765_commonsense_context_identity_closure_test.py
```

dry-run：

```bash
python tests/gpt5/phase765_commonsense_context_identity_closure_test.py \
  --dry-run --max-cases 8 --relations category,edible,grows_on_tree
```

第一次冒烟测试：

```bash
tests/gpt5/run_phase765_commonsense_context_identity_closure_test_round.sh smoke \
  --max-cases 12 --max-candidates 1 --controls-per-candidate 1 \
  --max-source-groups 4 --relations category,edible,grows_on_tree --log-every 4
```

第一次冒烟发现原始 commonsense prompt（常识提示）下 target_top1_rate 基本为 0，说明目标答案状态没有可靠形成。随后加入 allowed values（候选值列表），但不加入对象事实；这仍然是常识测试，因为只限定候选集合，没有告诉模型 `apple -> fruit`。

修正后冒烟测试：

```bash
tests/gpt5/run_phase765_commonsense_context_identity_closure_test_round.sh smoke \
  --max-cases 12 --max-candidates 1 --controls-per-candidate 1 \
  --max-source-groups 4 --relations category,edible,grows_on_tree --log-every 4
```

主测试：

```bash
tests/gpt5/run_phase765_commonsense_context_identity_closure_test_round.sh main \
  --max-candidates 1 --controls-per-candidate 1 --max-source-groups 4 \
  --relations category,edible,grows_on_tree --log-every 18
```

确认测试：

```bash
tests/gpt5/run_phase765_commonsense_context_identity_closure_test_round.sh confirm \
  --max-candidates 1 --controls-per-candidate 1 --max-source-groups 5 \
  --relations category,edible,grows_on_tree --log-every 18
```

三模型按 qwen3、GLM4、DS7B 顺序运行，并使用 `--hard-exit-after-model`。本阶段不使用量化方案，采用 bfloat16 eager 路径。

### 测试原理

本阶段不提供显式事实记录，而只使用常识提示：

```text
commonsense_question:
  Answer using common everyday knowledge.
  Allowed values: ...
  Question: What is the category of apple?
  Answer:

commonsense_statement:
  Use common everyday knowledge.
  Allowed values: ...
  Task: For apple, give category.
  Answer:
```

测试对象仍为 18 个对象、6 个语义域：

```text
fruit / animal / plant / object / tool / abstract
```

测试关系为：

```text
category / edible / grows_on_tree
```

确认轮任务量：

```text
18 objects * 3 relations * 2 commonsense prompt formats = 108 cases per model
```

source groups：

```text
instruction
question
object_tokens
relation_tokens
answer_prefix
```

特征：

```text
target_logit_drop
attention_mass
direct_target_boost
direct_total_route_suppression
```

评估两件事：

```text
1. base answer reliability：
   常识 prompt 是否真的让目标答案进入可预测状态。

2. commonsense causal fiber structure：
   不同常识 prompt 格式下，
   对象因果纤维是否保留 same-domain separation 和 same-object stability。
```

### 确认轮结果

结果目录：

```text
results/glm5_phase765_commonsense_context_identity_closure_test/confirm/
```

base answer reliability（基础答案可靠性）：

```text
qwen3:
  target_top1_rate = 0.806
  mean_target_rank = 1.324
  mean_contrast_rank = 51.741

GLM4:
  target_top1_rate = 0.593
  mean_target_rank = 1.657
  mean_contrast_rank = 5.046

DS7B:
  target_top1_rate = 0.185
  mean_target_rank = 5.833
  mean_contrast_rank = 66.102
```

commonsense domain separation（常识语义域分离）：

```text
qwen3:
  commonsense_question:
    NN = 0.556
    same = 0.801
    diff = 0.198
    separation = 0.603

  commonsense_statement:
    NN = 0.611
    same = 0.832
    diff = 0.226
    separation = 0.606

GLM4:
  commonsense_question:
    NN = 0.222
    same = 0.154
    diff = -0.053
    separation = 0.208

  commonsense_statement:
    NN = 0.278
    same = 0.087
    diff = -0.070
    separation = 0.156

DS7B:
  commonsense_question:
    NN = 0.667
    same = 0.258
    diff = -0.097
    separation = 0.355

  commonsense_statement:
    NN = 0.667
    same = 0.301
    diff = -0.100
    separation = 0.401
```

cross-context stability（跨常识提示格式稳定性）：

```text
qwen3:
  same_object = 0.877
  same_domain_other = 0.789
  diff_domain = 0.203
  object_gap = 0.088
  domain_gap = 0.585

GLM4:
  same_object = -0.261
  same_domain_other = -0.385
  diff_domain = -0.422
  object_gap = 0.124
  domain_gap = 0.037

DS7B:
  same_object = -0.115
  same_domain_other = -0.186
  diff_domain = -0.312
  object_gap = 0.071
  domain_gap = 0.126
```

### 客观进展

Phase 765 给出一个“模型分化明显”的关键结果：

```text
qwen3 支持 commonsense semantic fiber closure（常识语义纤维闭合）的初步正结果。
GLM4 只支持弱语义域分离，不支持对象级闭合。
DS7B 虽有 domain separation，但 base answer reliability 太低，不能作为强闭合证据。
```

这说明：

```text
自然常识语义机制不是完全不存在；
但它不是三模型稳定共享的不变量。
```

### 严格解释

不能说：

```text
已经完成自然语义对象编码机制。
```

更准确的结论是：

```text
在 qwen3 中，常识语境下对象因果纤维已经表现出强 domain-level structure 和跨 prompt 的 object-level stability。

在 GLM4 中，该结构很弱。

在 DS7B 中，虽然对象纤维能形成语义域分离，但目标答案本身不可靠，所以结果更像弱候选集合几何，而不是稳定常识闭合。
```

### 对附件理论的判断

附件中“静态语义图谱不是底层单位”的判断是正确的。Phase 765 进一步支持这一点：

```text
即使没有显式 profile，
模型是否形成语义结构，也取决于：
prompt 脉络
候选集合
目标答案可靠性
source group
attention/direct geometry
跨格式闭合
```

所以真正的图谱对象不是：

```text
apple node
fruit node
```

而是：

```text
apple-category commonsense trajectory
apple-edible commonsense trajectory
apple-answer candidate competition
apple prompt-format closure
```

### 当前硬伤

```text
1. 候选值列表是必要的，否则 target_top1 基本不成立；这说明自然生成闭合仍然脆弱。
2. DS7B 的 target_top1_rate 只有 0.185，不能把它的 domain separation 当成强语义证据。
3. GLM4 的 domain_gap 只有 0.037，跨 prompt 稳定性很弱。
4. 当前只做了 head/source 级别，不是 neuron-level atlas。
5. allowed values 虽不提供对象事实，但会引入 candidate-set conditioning（候选集合条件化）。
6. 当前只测 category / edible / grows_on_tree，尚未覆盖完整关系空间。
```

### 理论进展

Phase 765 把图谱目标进一步收紧为：

```text
不是静态语义图谱；
而是条件化常识脉络中的因果纤维闭合图谱。
```

目前最准确的机制表述是：

```text
语义对象 =
在给定任务脉络和候选集合下，
由 attention transport、direct readout geometry、source group effect
共同形成的跨关系因果纤维；
其强度取决于模型是否已经形成可靠预测充分状态。
```

### 下一步

下一阶段仍属于同一个大阶段，但应从“是否有因果纤维”转向：

```text
Phase 766:
Prediction-Sufficient State Reliability Audit
（预测充分状态可靠性审计）
```

核心问题：

```text
为什么 qwen3 能形成可靠 commonsense prediction-sufficient state，
而 GLM4 / DS7B 不能稳定形成？
```

具体任务：

```text
1. 按 relation 分开审计 category / edible / grows_on_tree。
2. 找出 target_top1 成功样本和失败样本的上下文状态差异。
3. 比较成功样本与失败样本中的 attention_mass、direct_score、source_group contribution。
4. 判断失败来自知识缺失、候选竞争、格式协议，还是读出阈值不足。
5. 只在 target state 已可靠形成的样本上继续做机制图谱。
```

## Phase 766: 预测充分状态可靠性审计 [2026-06-29 18:13]

### 本阶段判断

Phase 765 给出一个模型分化结果：

```text
qwen3 支持常识语义纤维闭合的初步正结果；
GLM4 只有弱语义域结构；
DS7B 虽有 domain separation，但 target_top1_rate 很低。
```

因此 Phase 766 不再加载模型，而是对 Phase 765 confirm 结果做离线审计，回答：

```text
成功形成 target_top1 的样本，
和失败样本在 attention / direct score / source contribution 上有什么差异？
```

### 生成脚本

```text
tests/gpt5/phase766_prediction_sufficient_state_reliability_audit.py
```

### 测试命令

```bash
python -m py_compile tests/gpt5/phase766_prediction_sufficient_state_reliability_audit.py
python tests/gpt5/phase766_prediction_sufficient_state_reliability_audit.py --round-name confirm
```

### 测试原理

读取 Phase 765 confirm 的 jsonl：

```text
results/glm5_phase765_commonsense_context_identity_closure_test/confirm/
```

把每个 commonsense case 按 target_top1 分成：

```text
success: 目标答案是 top1
failure: 目标答案不是 top1
```

然后比较：

```text
target_logit_drop
attention_mass_to_source
direct_target_boost
direct_total_route_suppression
direct_mean_margin_gain
source_positions_n
```

按三个层次聚合：

```text
1. by relation
2. by source_group
3. by relation + source_group
```

注意：本阶段是 observational audit（观察性审计），不是新的 causal intervention（因果干预）。

### 结果

结果目录：

```text
results/glm5_phase766_prediction_sufficient_state_reliability_audit/confirm/
```

基础可靠性按关系：

```text
qwen3:
  category top1 = 0.861, mean rank = 1.389
  edible top1 = 0.694, mean rank = 1.306
  grows_on_tree top1 = 0.861, mean rank = 1.278

GLM4:
  category top1 = 0.806, mean rank = 1.444
  edible top1 = 0.500, mean rank = 1.667
  grows_on_tree top1 = 0.472, mean rank = 1.861

DS7B:
  category top1 = 0.361, mean rank = 5.000
  edible top1 = 0.111, mean rank = 7.028
  grows_on_tree top1 = 0.083, mean rank = 5.472
```

success - failure source group gaps：

```text
qwen3:
  instruction:
    target_drop_gap = 0.049
    attention_gap = 0.002
    direct_boost_gap = 0.111

  object_tokens:
    target_drop_gap = 0.019
    attention_gap = 0.005
    direct_boost_gap = 0.015

  question:
    target_drop_gap = 0.032
    attention_gap = -0.007
    direct_boost_gap = 0.002

GLM4:
  all source groups:
    target_drop_gap mostly negative or near zero
    direct_boost_gap near zero
    attention gaps small

DS7B:
  instruction:
    attention_gap = 0.132
    direct_boost_gap = -0.085

  object_tokens:
    target_drop_gap = 0.058
    attention_gap = -0.039
    direct_boost_gap = -0.090

  question:
    attention_gap = -0.094
    direct_boost_gap = -0.132
```

### 客观解释

qwen3：

```text
成功样本更像真的形成了 prediction-sufficient state（预测充分状态）：
instruction / object_tokens / question 的移除更会造成 target_logit_drop，
instruction direct_target_boost 明显更强。
```

GLM4：

```text
成功和失败样本在 source contribution 上差异很小。
这说明 GLM4 的失败不一定是“没看对象”或“没看关系”，
更可能是读出阈值、候选竞争或任务协议不稳定。
```

DS7B：

```text
DS7B 的 target_top1 很低，
但成功样本并没有表现为 object/question direct boost 更强。
部分 source gap 甚至为负。
这说明 DS7B 的 commonsense 结果不能解释为稳定语义纤维闭合，
更可能是候选集合、格式协议或 readout competition（读出竞争）造成的弱结构。
```

### 对附件理论的进一步收紧

附件说“语义不是底层节点，而是脉络—路线—轨迹—竞争—闭合上的功能标签”，这个判断继续得到支持。

Phase 766 说明：

```text
真正需要进入图谱的不是 apple / fruit 静态节点，
而是：
1. target state 是否可靠形成；
2. 成功样本中哪些 source group 对 target 有必要贡献；
3. 失败样本失败在 source reading、candidate competition 还是 readout threshold；
4. 只有进入 prediction-sufficient state 的样本，才适合继续做机制图谱。
```

### 当前硬伤

```text
1. Phase 766 是离线观察性审计，不是新的因果干预。
2. success/failure 按 target_top1 划分，可能混入 tokenizer 和候选词首词元偏差。
3. DS7B failure 样本太多，成功样本只有 40 条 effect rows 对应的 case 较少，结论要谨慎。
4. 目前还没有把 failure 分解成知识缺失、候选竞争、格式协议、读出阈值四类。
5. 仍停留在 head/source 级别，未进入 neuron/channel 级。
```

### 理论进展

现在“语言机制图谱”的第一层应当明确为：

```text
prediction-sufficient state reliability layer
（预测充分状态可靠性层）
```

只有当一个样本已经形成可靠目标状态，后续的 route / causal fiber / closure 图谱才有解释意义。

这把当前理论从：

```text
对象因果纤维图谱
```

进一步收紧为：

```text
预测充分状态条件下的对象因果纤维图谱。
```

### 下一步

下一阶段应该是：

```text
Phase 767:
Failure-Type Decomposition for Commonsense Closure
（常识闭合失败类型分解）
```

具体任务：

```text
1. 对 Phase 765 的失败样本进行分类：
   knowledge miss（知识缺失）
   candidate competition（候选竞争）
   format/protocol miss（格式协议失败）
   readout threshold miss（读出阈值不足）

2. 对每类失败分别做 top-k token 审计：
   看模型到底倾向输出什么。

3. 对 qwen3 成功样本建立 clean subset：
   只在可靠预测充分状态上继续追踪 route / closure。

4. 对 GLM4 / DS7B 只在 target rank <= 2 的样本上重算纤维，
   避免把未形成目标状态的样本误当成语义机制。
```

## Phase 767: 常识闭合失败类型 top-k 审计 [2026-06-29 18:35]

### 背景

本阶段分析了最新附件中对 Phase 765 / Phase 766 的判断。附件的核心判断基本正确：

```text
语言机制图谱的第一层不应该是静态语义节点，
而应该是 prediction-sufficient state reliability layer（预测充分状态可靠性层）。
```

Phase 765 证明常识 prompt（提示）下 qwen3 有较强语义域因果纤维结构，GLM4 较弱，DS7B 的基础答案可靠性不足。
Phase 766 进一步说明，未形成 target_top1（目标第一名）的样本不应直接进入机制图谱。

但 Phase 766 仍然有一个硬伤：

```text
它只知道 target rank（目标排名）和 source effect（源组效应），
不知道失败时模型真正 top-k 倾向输出什么。
```

因此本阶段执行 Phase 767：对 Phase 765 的同一批常识 prompt 做 logits-only top-k 审计，用于失败类型分解。

### 测试脚本

新增脚本：

```text
tests/glm5/phase767_commonsense_failure_type_topk_audit.py
tests/glm5/run_phase767_commonsense_failure_type_topk_audit_round.sh
```

结果目录：

```text
tests/result/phase767_commonsense_failure_type_topk_audit/main/
results/glm5_phase767_commonsense_failure_type_topk_audit/main/
```

主结果文件：

```text
tests/result/phase767_commonsense_failure_type_topk_audit/main/phase767_cross_model_summary.md
tests/result/phase767_commonsense_failure_type_topk_audit/main/phase767_cross_model_summary.json
```

执行命令：

```bash
python -m py_compile tests/glm5/phase767_commonsense_failure_type_topk_audit.py

tests/glm5/run_phase767_commonsense_failure_type_topk_audit_round.sh smoke_alias \
  --max-cases 12 \
  --top-k 30 \
  --relations category,edible,grows_on_tree \
  --log-every 6

tests/glm5/run_phase767_commonsense_failure_type_topk_audit_round.sh main \
  --top-k 80 \
  --relations category,edible,grows_on_tree \
  --log-every 18
```

模型顺序：

```text
qwen3 -> GLM4 -> DS7B
```

加载方式：

```text
bf16；
quantization=off（无量化）；
优先尝试 flash_attention_2；
本机 FlashAttention2 未安装，自动降级为 sdpa；
每个模型独立进程运行，完成后释放显存。
```

### 测试原理

本阶段不做 attention hook（注意力钩子）、不做 patch（修补）、不做 source removal（源移除）。
只对 Phase 765 的常识 prompt 做下一词 logits（分数）读取。

每个样本记录：

```text
1. target answer（目标答案）首词元排名；
2. contrast answer（对照答案）首词元排名；
3. top-k token（前 k 个词元）；
4. allowed value set（候选值集合）内部排名；
5. target margin（目标相对最佳非目标的边际）；
6. failure type（失败类型）。
```

由于测试中发现 DS7B 和 GLM4 经常输出 `Yes`，而目标答案可能写作 `yes`，
因此本阶段新增了 semantic alias（语义别名）合并：

```text
yes / Yes / YES
no / No / NO
fruit / Fruit / FRUIT
```

同时保留 exact top1（精确首词元第一名）和 semantic top1（语义别名第一名）。
这是一个重要纠偏：否则会把大小写 / 词形差异误判成语义失败。

### 失败类型定义

本阶段使用保守规则：

```text
success_top1:
  语义别名 target 为 top1。

readout_threshold_miss:
  语义 target rank = 2。

known_contrast_competition:
  已知 contrast answer 排在 target 前面。

allowed_value_candidate_competition:
  target 在 allowed value set 内部不是第一，且整体排名靠前。

candidate_competition_other:
  target 在 top-k 内，但不是已知 contrast 或 allowed value 竞争。

format_protocol_miss:
  top1 是 The / Answer / 标点 / 空白 等格式或解释起手词。

knowledge_or_state_formation_miss:
  target 不在 top-k，或没有形成可解释的目标状态。
```

本轮 top-k=80，所有 target 都进入 top-k，所以没有出现强 `knowledge_or_state_formation_miss`。
这说明本轮失败主要不是“完全不知道答案”，而是 readout / candidate / format competition（读出、候选、格式竞争）。

### 数学形式

预测充分状态的第一过滤仍然是：

```text
R_suf(x) = 1[rank(y_target | x) = 1]
```

本阶段修正为语义别名版本：

```text
A(y) = { token aliases of y }
```

```text
rank_sem(y | x)
=
min_{a in A(y)} rank(a | x)
```

```text
R_suf_sem(x)
=
1[rank_sem(y_target | x) = 1]
```

失败类型分解为：

```text
Failure(x)
in
{
  readout_threshold,
  known_contrast_competition,
  allowed_value_candidate_competition,
  candidate_competition_other,
  format_protocol,
  knowledge_or_state_formation
}
```

目标边际为：

```text
margin_sem(x)
=
max_{a in A(y_target)} logit(a | x)
-
max_{z notin A(y_target)} logit(z | x)
```

### 客观结果

主测试共：

```text
18 objects * 3 relations * 2 context formats = 108 cases / model
```

整体可靠性：

| model | semantic top1 | exact top1 | semantic rank | exact rank | clean n | rank<=2 n |
|---|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.824 | 0.824 | 1.315 | 1.315 | 89 | 102 |
| GLM4 | 0.750 | 0.611 | 1.380 | 1.630 | 81 | 99 |
| DS7B | 0.352 | 0.176 | 3.806 | 6.407 | 38 | 55 |

最重要的客观现象：

```text
1. qwen3 精确 top1 和语义 top1 一致，说明词形偏差较小。
2. GLM4 语义 top1 从 exact 0.611 提升到 semantic 0.750，说明大量失败其实是大小写 / 词形读出问题。
3. DS7B 语义 top1 从 exact 0.176 提升到 semantic 0.352，说明 DS7B 的一部分失败也是词形问题，但仍然远弱于 qwen3 / GLM4。
4. 三个模型 target_in_topk_rate 都是 1.000，说明目标答案通常在候选空间内，不是完全消失。
5. DS7B 的 mean semantic margin 仍为负数，说明它即使知道候选，也经常不能把正确语义候选推到第一。
```

失败类型分布：

| model | success_top1 | readout_threshold | known_contrast | allowed_value_comp | other_comp | format_protocol |
|---|---:|---:|---:|---:|---:|---:|
| qwen3 | 89 | 13 | 2 | 4 | 0 | 0 |
| GLM4 | 81 | 18 | 8 | 0 | 1 | 0 |
| DS7B | 38 | 17 | 41 | 7 | 1 | 4 |

### 结果解释

#### qwen3

qwen3 是当前最稳定的模型：

```text
semantic top1 = 0.824
clean subset = 89 / 108
rank <= 2 = 102 / 108
```

这说明 qwen3 大多数常识样本已经进入 prediction-sufficient state（预测充分状态）。
失败样本主要是：

```text
readout_threshold_miss: 13
allowed_value_candidate_competition: 4
known_contrast_competition: 2
```

因此 qwen3 的瓶颈不是“目标答案不存在”，而是少量候选竞争和读出阈值问题。

#### GLM4

GLM4 的关键发现是：

```text
exact top1 = 0.611
semantic top1 = 0.750
```

这说明 GLM4 有相当一部分样本语义上已经闭合，但首词元大小写 / 词形不同。
因此 Phase 765 / 766 对 GLM4 的严格首词元判断可能低估了它的语义预测能力。

但 GLM4 仍有：

```text
readout_threshold_miss: 18
known_contrast_competition: 8
```

所以它比 qwen3 更容易卡在读出阈值和候选竞争。

#### DS7B

DS7B 的 exact top1 很低，但 semantic top1 翻倍：

```text
exact top1 = 0.176
semantic top1 = 0.352
```

这证明 DS7B 的一部分失败是词形 / 大小写输出问题，而不是完全语义失败。

但是 DS7B 仍有大量竞争失败：

```text
known_contrast_competition: 41
readout_threshold_miss: 17
allowed_value_candidate_competition: 7
format_protocol_miss: 4
```

这说明 DS7B 的核心瓶颈仍然是：

```text
候选竞争场没有稳定收敛；
正确答案经常存在，但不能压过错误候选或格式路线。
```

### 对附件判断的修正

附件说 Phase 765 / 766 的下一步应做 failure taxonomy（失败类型分类），这是正确的。
但 Phase 767 发现一个新的关键细节：

```text
failure taxonomy 不能只看 exact first-token rank。
必须区分：
1. exact token closure（精确词元闭合）；
2. semantic alias closure（语义别名闭合）；
3. phrase-level closure（短语级闭合）。
```

否则会把 `Yes` 与 `yes`、`No` 与 `no` 的差异误判为语义机制失败。

这说明语言机制图谱的第一层还要再细分：

```text
prediction-sufficient state reliability layer
  -> semantic target state reliability
  -> lexical realization reliability
  -> phrase-level generation reliability
```

### 当前硬伤

```text
1. 本阶段仍然是 logits-only observation（只读分数观察），不是因果干预。
2. semantic alias 只覆盖简单大小写 / 词形别名，未覆盖同义表达。
3. top-k 只看首词元，不能证明 phrase-level generation closure（短语级生成闭合）。
4. allowed values（候选值列表）仍然存在，会增强候选集合条件化。
5. format_protocol_miss 只能识别 top1 显式格式词，不能覆盖所有协议失败。
6. 没有新增关系类型，仍只覆盖 category / edible / grows_on_tree。
7. 当前模型都是小模型，DS7B 等结果可能包含小模型压缩和格式捷径偏差。
```

### 理论进展

本阶段把 Phase 766 的“预测充分状态”继续细化为三层：

```text
1. semantic prediction-sufficient state
   语义目标是否已经成为第一候选。

2. lexical realization state
   语义目标是否以预期词形 / 大小写输出。

3. phrase generation closure
   首词元之后是否能完成完整短语生成。
```

因此最新机制图谱第一层应写成：

```text
State reliability layer
=
semantic closure
+
lexical closure
+
phrase closure
```

这比单纯 target_top1 更严格，也更接近真实语言生成机制。

### 下一步

下一阶段应进入：

```text
Phase 768:
Semantic-Alias Clean Subset Fiber Reanalysis and Phrase Closure
（语义别名干净子集因果纤维重算与短语闭合测试）
```

任务：

```text
1. 用 Phase 767 的 semantic top1 重新定义 clean subset。
2. 在 qwen3 / GLM4 的 semantic clean subset 上重算 Phase 765 的因果纤维。
3. 对 DS7B 只保留 semantic rank <= 2 的样本，避免失败样本污染图谱。
4. 增加 phrase likelihood（短语似然）测试，确认首词元闭合是否能延续成完整答案。
5. 对 exact success 和 semantic-only success 分开比较：
   如果因果纤维相似，说明大小写只是输出层词形问题；
   如果因果纤维不同，说明词形闭合也是机制的一部分。
```

阶段性结论：

```text
Phase 767 完成了常识闭合失败类型分解。
当前最重要的新拼图是：
语义闭合与精确词元闭合不是同一层机制。

qwen3 已经具备较强语义预测充分状态；
GLM4 被 strict first-token metric 明显低估；
DS7B 目标答案常在 top-k 内，但候选竞争和格式路线仍然很强。
```

## Phase 768: 语义别名干净子集短语闭合测试 [2026-06-29 19:03]

### 背景

本阶段继续分析最新附件。附件对 Phase 767 的判断基本正确：

```text
Phase 767 的关键进展不是又找到一个 patch（补丁），
而是把 prediction-sufficient state（预测充分状态）拆成：
semantic closure（语义闭合）
lexical closure（词形闭合）
phrase closure（短语闭合）。
```

Phase 767 已经证明：

```text
exact first-token top1（精确首词元第一名）
不等于
semantic top1（语义别名第一名）。
```

但它仍然只在 first-token（首词元）层面。
因此本阶段执行 Phase 768，目标是检验：

```text
Phase 767 中已经 semantic closed（语义闭合）的样本，
是否能继续完成 phrase-level closure（短语级闭合）和 short greedy generation（短生成闭合）。
```

### 测试脚本

新增脚本：

```text
tests/glm5/phase768_semantic_alias_phrase_closure.py
tests/glm5/run_phase768_semantic_alias_phrase_closure_round.sh
```

结果目录：

```text
tests/result/phase768_semantic_alias_phrase_closure/main/
results/glm5_phase768_semantic_alias_phrase_closure/main/
```

主结果文件：

```text
tests/result/phase768_semantic_alias_phrase_closure/main/phase768_cross_model_summary.md
tests/result/phase768_semantic_alias_phrase_closure/main/phase768_cross_model_summary.json
```

执行命令：

```bash
python -m py_compile tests/glm5/phase768_semantic_alias_phrase_closure.py

tests/glm5/run_phase768_semantic_alias_phrase_closure_round.sh smoke2 \
  --max-cases 12 \
  --phase767-round main \
  --max-new-tokens 6 \
  --relations category,edible,grows_on_tree \
  --log-every 6

tests/glm5/run_phase768_semantic_alias_phrase_closure_round.sh main \
  --phase767-round main \
  --max-new-tokens 6 \
  --relations category,edible,grows_on_tree \
  --log-every 18
```

模型顺序：

```text
qwen3 -> GLM4 -> DS7B
```

加载方式：

```text
bf16；
quantization=off（无量化）；
优先尝试 flash_attention_2；
本机 FlashAttention2 未安装，自动降级为 sdpa；
每个模型独立进程运行，完成后释放显存。
```

### 测试原理

本阶段使用 Phase 765 的同一批常识 prompt 和 Phase 767 的 clean subset 标注。

每个 case 做两类测试：

```text
1. phrase likelihood（短语似然）：
   对 allowed values（候选值集合）中的每个值构造 phrase forms（短语形式），
   例如 yes / Yes / YES / yes. / Yes. / YES.，
   计算完整短语的 log probability（对数概率），
   看 target value（目标值）是否是短语级第一。

2. short greedy generation（短贪婪生成）：
   让模型生成最多 6 个新词元，
   解析第一条非空答案片段，
   判断是否以目标语义值开头。
```

本阶段修正了一个解析问题：

```text
生成文本经常形如：
"fruit\nQuestion: ..."
"no. The oak tree is ..."

如果直接解析整段文本，会误把第二行或解释内容当作答案。
现在只取第一条非空答案片段进行 generation match（生成匹配）判断。
```

### 数学形式

Phase 767 的语义别名集合：

```text
A(y) = {alias tokens / alias phrases of y}
```

短语候选集合：

```text
P(y) = { phrase forms of y }
```

短语似然：

```text
L(p | x)
=
sum_i log P_theta(p_i | x, p_<i)
```

目标值短语分数：

```text
Score_phrase(y | x)
=
max_{p in P(y)} L(p | x)
```

短语闭合：

```text
R_phrase(x)
=
1[
Score_phrase(y_target | x)
>
max_{y != y_target} Score_phrase(y | x)
]
```

短生成闭合：

```text
R_gen(x)
=
1[
first_generated_value(x) = y_target
]
```

完整状态可靠性现在应写成：

```text
R_state(x)
=
R_sem(x)
and
R_phrase(x)
and
R_gen(x)
```

### 客观结果

主测试共：

```text
108 cases / model
```

整体结果：

| model | semantic top1 | exact top1 | phrase top1 | generation match |
|---|---:|---:|---:|---:|
| qwen3 | 0.824 | 0.824 | 0.806 | 0.824 |
| GLM4 | 0.750 | 0.611 | 0.750 | 0.750 |
| DS7B | 0.352 | 0.176 | 0.380 | 0.352 |

semantic clean subset（语义干净子集）结果：

| model | clean n | phrase top1 | generation match | mean phrase rank | mean phrase margin |
|---|---:|---:|---:|---:|---:|
| qwen3 | 89 | 0.978 | 1.000 | 1.022 | 5.709 |
| GLM4 | 81 | 0.975 | 1.000 | 1.025 | 2.307 |
| DS7B | 38 | 0.974 | 1.000 | 1.026 | 3.350 |

semantic-only subset（语义闭合但 exact token 不闭合）结果：

| model | semantic-only n | phrase top1 | generation match |
|---|---:|---:|---:|
| qwen3 | 0 | null | null |
| GLM4 | 15 | 1.000 | 1.000 |
| DS7B | 19 | 0.947 | 1.000 |

semantic fail subset（语义未闭合）结果：

| model | fail n | phrase top1 | generation match |
|---|---:|---:|---:|
| qwen3 | 19 | 0.000 | 0.000 |
| GLM4 | 27 | 0.074 | 0.000 |
| DS7B | 70 | 0.057 | 0.000 |

### 关键客观现象

```text
1. 一旦进入 semantic clean subset，三个模型几乎都能完成短语级闭合和短生成闭合。

2. GLM4 的 semantic-only 样本：
   phrase top1 = 1.000
   generation match = 1.000
   说明之前 exact token 失败主要是 lexical realization（词形实现）问题，不是语义机制失败。

3. DS7B 的 semantic-only 样本：
   phrase top1 = 0.947
   generation match = 1.000
   说明 DS7B 只要进入语义闭合状态，也能自然生成正确答案。

4. semantic fail subset 中 generation match 全部为 0。
   说明 first-token semantic closure 是短语 / 生成闭合的强前置条件。

5. qwen3 / GLM4 / DS7B 的 semantic clean subset 生成匹配均为 1.000。
   这支持：
   semantic closure 是 phrase generation closure 的有效前置层。
```

### 需要谨慎解释的细节

有少量 semantic clean 样本的 phrase top1 为 false，但 generation match 为 true。
检查发现它们多为 phrase likelihood 的零边际 tie（并列）或标点形式差异，例如：

```text
target = no
best phrase value = yes
margin = 0.0
generated = "no. ..."
```

因此：

```text
phrase_top1 是严格短语似然指标；
generation_match 是自然短生成指标；
二者不完全等价。
```

这不是推翻结果，而是说明下一步要引入 tie-aware phrase closure（并列感知短语闭合）。

### 对附件判断的修正

附件判断“必须进入 phrase-level closure”是正确的。
Phase 768 的结果进一步说明：

```text
真正的主要分界不是 phrase closure 是否存在，
而是 semantic closure 是否先成立。
```

更精确地说：

```text
semantic closure 成立 -> phrase / generation closure 大概率成立；
semantic closure 不成立 -> phrase / generation closure 基本不成立。
```

因此语言机制图谱第一层应进一步写成：

```text
semantic state formation
  -> lexical realization
  -> phrase continuation
  -> generation closure
```

其中 semantic state formation 是最关键的门。

### 当前硬伤

```text
1. 本阶段仍然使用 allowed values（候选值列表），不是完全自由生成。
2. phrase forms 只覆盖简单值、大小写、句点形式，未覆盖同义短语。
3. phrase likelihood 使用候选集合内部比较，不等于全词表自然生成概率。
4. generation match 只解析第一条非空答案片段，不能评价后续解释是否一致。
5. 没有做 causal intervention（因果干预），只能说明闭合层级关系，不能定位闭合原因。
6. 仍只覆盖 category / edible / grows_on_tree 三类关系。
7. 小模型结果不能直接外推为大模型或大脑机制。
```

### 理论进展

当前理论应从：

```text
prediction-sufficient state reliability layer
```

进一步收紧为：

```text
semantic-state-first closure theory
（语义状态优先闭合理论）
```

即：

```text
语言生成的关键第一门不是词形，也不是短语，
而是目标语义候选是否成为当前脉络中的主导状态。

一旦语义状态闭合，
词形和短语生成多数情况下会自然跟随；
如果语义状态未闭合，
短语级和生成级闭合几乎不会凭空出现。
```

这说明：

```text
Phase 767 发现语义闭合与 exact token 闭合不同层；
Phase 768 证明语义闭合是 phrase / generation closure 的强前置层。
```

### 下一步

下一阶段应进入：

```text
Phase 769:
Semantic Clean Subset Causal Fiber Reanalysis
（语义干净子集因果纤维重算）
```

具体任务：

```text
1. 用 Phase 767 / 768 的 semantic clean subset 重新过滤 Phase 765 因果纤维。
2. 对比：
   all cases
   exact clean
   semantic clean
   semantic-only
   semantic fail
3. 检查 semantic clean subset 是否有更强 domain separation / object stability。
4. 对 semantic-only 样本单独分析：
   如果纤维与 exact clean 相似，说明词形只是输出层问题；
   如果纤维不同，说明 lexical closure 有独立机制。
5. 对 DS7B 只在 semantic clean 或 rank<=2 子集上重算，避免失败样本污染。
```

阶段性结论：

```text
Phase 768 证明：
语义闭合不是终点，但它是短语闭合和自然生成闭合的强前置条件。

GLM4 和 DS7B 被 exact first-token metric 低估；
只要进入 semantic clean subset，它们也能完成短语与生成闭合。

因此后续机制图谱必须以 semantic clean subset 为第一过滤层。
```

## Phase 769: 语义干净子集因果纤维离线重分析 [2026-06-29 19:07]

### 背景

Phase 768 证明：

```text
semantic clean subset（语义干净子集）
几乎都能完成 phrase closure（短语闭合）和 generation closure（生成闭合）。
```

但这还没有回答一个更严格的问题：

```text
语义闭合样本的因果纤维图谱是否也更稳定？
```

因此本阶段执行 Phase 769：

```text
不加载模型；
直接读取 Phase 765 的因果纤维 effect rows；
再用 Phase 767 的 semantic / exact clean subset 标签过滤；
比较不同子集的 domain separation（语义域分离）和 cross-context stability（跨上下文稳定性）。
```

### 测试脚本

新增脚本：

```text
tests/glm5/phase769_semantic_clean_fiber_reanalysis.py
```

结果目录：

```text
tests/result/phase769_semantic_clean_fiber_reanalysis/confirm_x_main/
results/glm5_phase769_semantic_clean_fiber_reanalysis/confirm_x_main/
```

执行命令：

```bash
python -m py_compile tests/glm5/phase769_semantic_clean_fiber_reanalysis.py

python tests/glm5/phase769_semantic_clean_fiber_reanalysis.py \
  --phase765-round confirm \
  --phase767-round main
```

### 测试原理

输入：

```text
Phase 765:
commonsense causal-fiber effect rows

Phase 767:
semantic top1 / exact top1 / semantic_only / semantic_fail 标签
```

按以下子集过滤 Phase 765 因果纤维：

```text
all
exact_clean
semantic_clean
semantic_only
semantic_fail
rank_le2
```

然后重新计算：

```text
1. context 内 same-domain vs different-domain separation；
2. nearest-neighbor domain accuracy；
3. cross-context object stability gap；
4. cross-context domain stability gap。
```

### 客观结果

| model | subset | cases | mean sep | mean NN | object gap | domain gap |
|---|---:|---:|---:|---:|---:|---:|
| qwen3 | all | 108 | 0.605 | 0.583 | 0.088 | 0.585 |
| qwen3 | semantic_clean | 89 | 0.471 | 0.500 | 0.084 | 0.437 |
| qwen3 | semantic_fail | 19 | 0.868 | 0.436 | -0.237 | 1.126 |
| GLM4 | all | 108 | 0.182 | 0.250 | 0.124 | 0.037 |
| GLM4 | exact_clean | 66 | 0.163 | 0.149 | 0.482 | 0.085 |
| GLM4 | semantic_clean | 81 | 0.258 | 0.167 | 0.389 | 0.307 |
| GLM4 | semantic_only | 15 | 0.647 | 0.364 | null | null |
| DS7B | all | 108 | 0.378 | 0.667 | 0.071 | 0.126 |
| DS7B | exact_clean | 19 | 0.814 | 0.633 | 0.125 | -0.084 |
| DS7B | semantic_clean | 38 | 0.228 | 0.300 | 0.097 | 0.124 |
| DS7B | semantic_only | 19 | -0.061 | 0.231 | null | null |

### 关键客观现象

```text
1. semantic clean subset 并没有在所有模型上单调增强因果纤维分离。

2. qwen3 的 all subset 分离度高于 semantic_clean：
   all sep = 0.605
   semantic_clean sep = 0.471
   说明 qwen3 的失败样本仍然可能带有强 domain-level 结构，
   但 object stability 可能不可靠。

3. GLM4 的 semantic_clean 比 all 更好：
   all sep = 0.182
   semantic_clean sep = 0.258
   domain gap 从 0.037 提升到 0.307。
   说明 GLM4 中 failed states 确实可能污染了部分机制图谱。

4. DS7B 的 exact_clean 分离度最高：
   exact_clean sep = 0.814
   semantic_clean sep = 0.228
   semantic_only sep = -0.061。
   这说明 DS7B 的 semantic-only 样本虽然能短语生成正确，
   但其因果纤维不像 exact clean 那样稳定。
```

### 最重要的收紧

Phase 768 容易让人得出：

```text
只要 semantic closure 成立，就可以进入机制图谱。
```

Phase 769 纠正了这个过强结论。

更准确的判断是：

```text
semantic closure 是 phrase / generation closure 的强前置条件，
但不是因果纤维图谱稳定性的充分条件。
```

也就是说：

```text
输出闭合层
和
内部因果纤维层
仍然不能完全等同。
```

### 当前硬伤

```text
1. Phase 769 是离线重分析，不是新的因果干预。
2. 子集过滤会改变 object / relation 分布，可能造成统计偏差。
3. semantic_only 子集样本很少，特别是 qwen3 为 0，不能泛化。
4. DS7B exact_clean 只有 19 个 case，分离度高但样本少。
5. 当前因果纤维仍然是 head/source 级，不是 neuron/channel 级。
6. Phase 765 的 effect rows 来自固定候选组件，不是全模型完整图谱。
```

### 理论进展

当前闭合层级必须拆成两条线：

```text
输出闭合线：
semantic closure -> lexical closure -> phrase closure -> generation closure

内部机制线：
source contribution -> causal fiber -> route competition -> readout geometry
```

Phase 768 证明输出闭合线中：

```text
semantic closure 是 phrase / generation closure 的强前置层。
```

Phase 769 进一步证明：

```text
输出闭合线不能直接等同于内部机制线。
```

因此最新理论应继续收紧为：

```text
双层闭合理论：
1. output closure reliability（输出闭合可靠性）
2. causal-fiber structural reliability（因果纤维结构可靠性）

只有两者同时成立，样本才适合进入更深的 neuron / channel atlas。
```

### 下一步

下一阶段应进入：

```text
Phase 770:
Balanced Semantic-Clean Fiber Reanalysis
（平衡语义干净子集因果纤维重分析）
```

任务：

```text
1. 按 relation / domain 对 clean 与 fail 子集做配平，避免子集分布偏差。
2. 对 exact_clean / semantic_only / semantic_fail 做同对象同关系配对比较。
3. 分开计算：
   domain-level fiber structure
   object-level identity stability
   relation-level fiber structure
4. 如果配平后 semantic_clean 仍不能增强纤维结构，
   说明输出闭合与内部纤维结构存在更深分离。
```

阶段性结论：

```text
Phase 769 是一个必要负结果 / 收紧结果。
它没有推翻 Phase 768，
但证明 phrase / generation closure 不能直接替代 causal-fiber stability。

下一步不能只按输出成功过滤，
必须做 balanced subset（配平子集）和 paired comparison（配对比较）。
```

## Phase 770: 平衡语义干净子集因果纤维重分析 [2026-06-29 19:43]

### 总体判断

本阶段继续分析最新附件。附件对 Phase 768 和 Phase 769 的判断基本正确：

```text
Phase 768 证明 semantic closure 是 phrase / generation closure 的强前置条件；
Phase 769 证明 output closure 不能直接等同于 causal-fiber stability。
```

但 Phase 769 有一个重要硬伤：

```text
semantic_clean、semantic_fail、exact_clean、semantic_only 的 object / domain / relation / context_format 分布不同。
```

因此，本阶段执行 Phase 770：不重新加载模型，只读取 Phase 765 的 causal-fiber effect rows 和 Phase 767 的 semantic / exact labels，做 balanced subset 与 paired context 的离线重分析。

### 测试脚本

新增脚本：

```text
tests/glm5/phase770_balanced_semantic_clean_fiber_reanalysis.py
```

结果目录：

```text
tests/result/phase770_balanced_semantic_clean_fiber_reanalysis/confirm_x_main/
results/glm5_phase770_balanced_semantic_clean_fiber_reanalysis/confirm_x_main/
```

执行命令：

```bash
python -m py_compile tests/glm5/phase770_balanced_semantic_clean_fiber_reanalysis.py

python tests/glm5/phase770_balanced_semantic_clean_fiber_reanalysis.py \
  --phase765-round confirm \
  --phase767-round main
```

本阶段没有加载模型，没有占用 GPU。

### 测试原理

第一部分是分层配平。对两个待比较子集 A 和 B，在同一个 stratum 内取相同数量样本：

```text
stratum = domain + relation + context_format
```

数学形式：

$$
S_k
=
\{x\mid domain(x)=d_k,\ relation(x)=r_k,\ context(x)=c_k\}
$$

$$
B_k^{A,B}
=
\min
\left(
|A\cap S_k|,
|B\cap S_k|
\right)
$$

$$
A'
=
\bigcup_k
sample(A\cap S_k,\ B_k^{A,B})
$$

$$
B'
=
\bigcup_k
sample(B\cap S_k,\ B_k^{A,B})
$$

第二部分是 paired context stability。对同一个 object 和同一个 relation，在两个 context_format 之间比较因果纤维向量相似度：

$$
S_{pair}(o,r)
=
cos
\left(
Fiber(o,r\mid commonsense\_question),
Fiber(o,r\mid commonsense\_statement)
\right)
$$

### 客观结果

| model | exact clean | semantic clean | semantic only | semantic fail |
|---|---:|---:|---:|---:|
| qwen3 | 89 | 89 | 0 | 19 |
| GLM4 | 66 | 81 | 15 | 27 |
| DS7B | 19 | 38 | 19 | 70 |

配平后的 semantic_clean - semantic_fail：

| model | strata | cases each | delta sep | delta object gap | delta domain gap |
|---|---:|---:|---:|---:|---:|
| qwen3 | 11 | 11 | +0.310 | -0.569 | -0.459 |
| GLM4 | 16 | 16 | +0.325 | +0.331 | +0.045 |
| DS7B | 18 | 18 | +0.064 | -0.069 | -0.209 |

同对象同关系跨上下文稳定性：

| model | group | pairs | mean context cosine |
|---|---|---:|---:|
| qwen3 | both semantic clean | 40 | 0.951 |
| qwen3 | both semantic fail | 5 | 0.959 |
| GLM4 | both semantic clean | 38 | 0.931 |
| GLM4 | both semantic fail | 11 | 0.918 |
| DS7B | both semantic clean | 10 | 0.636 |
| DS7B | both semantic fail | 26 | 0.640 |

### 严格分析

GLM4 的结果支持一个较稳判断：

```text
在配平 domain / relation / context_format 后，
semantic_clean 仍比 semantic_fail 有更高的 separation 和 object gap。
```

但 qwen3 和 DS7B 都没有给出一致增强。qwen3 的 semantic_clean separation 更高，但 object gap 和 domain gap 更低；DS7B 的增强很弱，且稳定性指标没有改善。

paired context stability 也没有证明 both semantic clean 一定比 both semantic fail 更稳定。因此不能把 semantic closure 当作内部纤维稳定的充分条件。

### 当前最新收紧

```text
1. semantic closure 是 phrase / generation closure 的强前置条件。
2. semantic closure 不是 causal-fiber stability 的充分条件。
3. exact closure 也不是绝对充分条件，但在 GLM4 / DS7B 的 paired context 中比 semantic_only 更接近稳定纤维。
4. 输出成功样本可以作为图谱入口候选，但不能直接作为神经元 / 通道图谱样本。
5. 真正进入 atlas 的样本必须同时通过 output closure 和 balanced fiber reliability。
```

对应公式收紧为：

$$
R_{atlas}(x)
=
R_{output}(x)
\land
R_{fiber}^{balanced}(x)
\land
R_{paired}(x)
$$

### 问题和硬伤

```text
1. Phase 770 仍是离线重分析，不是新 causal intervention。
2. 配平后样本数明显变小，特别是 GLM4 exact_clean vs semantic_only 只有 3 个样本。
3. NN 指标在小配平子集上多为 0，说明该指标不适合解释小子集。
4. 当前仍是 head/source 级，不是 neuron/channel 级。
5. 小模型结构可能有压缩偏差，尤其 DS7B 不能直接代表大模型。
```

### 阶段性结论

```text
Phase 770 没有推翻 Phase 768，也没有推翻 Phase 769。
它进一步证明：
输出闭合线和内部机制线确实必须分开审计。

semantic closure 可以作为输出闭合入口；
balanced fiber reliability 才是进入机制图谱的关键门槛。
```

下一阶段不应继续只做 semantic clean 过滤，而应进入：

```text
Phase 771:
Matched Causal Intervention Reliability Test
（配对因果干预可靠性测试）
```

## Phase 771: 配对因果干预可靠性测试 [2026-06-29 19:57]

### 总体判断

本阶段继续分析最新附件。附件对 Phase 770 的判断是正确的：

```text
semantic closure 可以作为 output closure 的入口，
但不能直接作为 mechanism atlas 的入口。
```

Phase 770 的关键硬伤是：

```text
它仍然是离线重分析，不是新的 causal intervention。
```

因此本阶段执行 Phase 771：在 Phase 767 / Phase 770 已经配平的 semantic_clean vs semantic_fail cases 上，重新加载 qwen3、GLM4、DS7B，直接做 source-contribution removal 因果干预。

### 测试脚本

新增脚本：

```text
tests/glm5/phase771_matched_causal_intervention_reliability_test.py
tests/glm5/run_phase771_matched_causal_intervention_reliability_round.sh
```

结果目录：

```text
tests/result/phase771_matched_causal_intervention_reliability_test/
results/glm5_phase771_matched_causal_intervention_reliability_test/
```

执行三轮：

```text
smoke：每模型 1 对，验证脚本正常。
main：每模型 6 对，主要测试。
confirm：每模型 10 对，并加入 same-layer control head。
```

模型顺序：

```text
qwen3 -> GLM4 -> DS7B
```

加载设置：

```text
bf16
quantization off
device_map auto
attention eager
```

本阶段需要 output_attentions 来计算 source contribution，因此使用 eager attention。

### 测试原理

对配平后的 clean / fail cases，选择 Phase755 top candidate head，并在 confirm 轮加入 same-layer control head。对 source group 执行：

```text
instruction
question
object_tokens
relation_tokens
```

计算源贡献：

$$
C_g(l,h\mid x)
=
\sum_{t\in g}
\alpha_{l,h}(p,t\mid x)
V_{l,h}(t\mid x)
$$

执行移除：

$$
h' = h - Proj_O(C_g(l,h\mid x))
$$

记录：

```text
target_logit_drop
margin_drop_target_vs_contrast
top1_loss
attention_mass
direct_target_boost
direct_route_suppression
```

### Confirm 轮关键结果

| model | arm | rows | target drop | margin drop | top1 loss | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | clean | 80 | -0.023 | -0.022 | 0.000 | 0.240 | 0.070 |
| qwen3 | fail | 80 | -0.030 | -0.042 | 0.000 | 0.166 | 0.068 |
| GLM4 | clean | 80 | 0.007 | 0.018 | 0.000 | 0.001 | 0.002 |
| GLM4 | fail | 80 | -0.002 | 0.009 | 0.000 | 0.001 | 0.001 |
| DS7B | clean | 80 | 0.025 | -0.006 | 0.000 | 0.058 | 0.131 |
| DS7B | fail | 80 | 0.013 | 0.002 | 0.013 | 0.098 | 0.118 |

Fiber bucket：

| model | fiber bucket | rows | target drop | margin drop | top1 loss |
|---|---|---:|---:|---:|---:|
| qwen3 | fiber_high | 96 | -0.008 | -0.021 | 0.000 |
| qwen3 | fiber_low | 64 | -0.055 | -0.049 | 0.000 |
| GLM4 | fiber_high | 88 | 0.007 | 0.013 | 0.000 |
| GLM4 | fiber_low | 72 | -0.003 | 0.015 | 0.000 |
| DS7B | fiber_high | 72 | 0.001 | 0.000 | 0.000 |
| DS7B | fiber_low | 88 | 0.034 | -0.004 | 0.011 |

Candidate vs control：

| model | candidate kind | arm | rows | target drop | margin drop | top1 loss |
|---|---|---|---:|---:|---:|---:|
| qwen3 | Phase755 top candidate | clean | 40 | -0.041 | -0.048 | 0.000 |
| qwen3 | same-layer control | clean | 40 | -0.006 | 0.005 | 0.000 |
| GLM4 | Phase755 top candidate | clean | 40 | 0.013 | 0.022 | 0.000 |
| GLM4 | same-layer control | clean | 40 | 0.002 | 0.015 | 0.000 |
| DS7B | Phase755 top candidate | clean | 40 | 0.052 | -0.018 | 0.000 |
| DS7B | same-layer control | clean | 40 | -0.002 | 0.006 | 0.000 |

### 严格结论

Phase 771 是关键负结果 / 收紧结果。

```text
1. 三模型 clean rows 都没有出现 top1 loss。
2. qwen3 source removal 效果为负或接近 0。
3. GLM4 clean 的 target drop / margin drop 略高于 fail，但数值很小。
4. DS7B top candidate 比 control 更有 target drop，但 clean 仍没有 top1 loss。
5. fiber_high 没有稳定预测更强 intervention sensitivity。
```

因此：

```text
output-clean / fiber-high 是机制图谱候选入口；
但不是当前候选 head/source path 必然可因果验证的充分条件。
```

Phase 770 的双门槛需要升级为三门槛：

$$
R_{atlas}(x)
=
R_{output}(x)
\land
R_{fiber}^{balanced}(x)
\land
R_{component}^{causal}(x)
$$

### 硬伤

```text
1. 只测试 Phase755 top candidate 和同层 control，不是全组件扫描。
2. 仍是 head/source contribution，不是 neuron/channel intervention。
3. top1_loss 几乎为 0，说明干预不够强或组件不是必要路径。
4. 仍依赖 allowed values 常识任务，不是自由生成机制。
```

### 下一步

下一阶段应进入：

```text
Phase 772:
Matched Component Discovery Scan
（配对组件发现扫描）
```

核心任务：

```text
1. 在 matched output-clean + fiber-high cases 内扫描更多 layer/head/source_group。
2. 不再只沿用 Phase755 top candidate。
3. 对每个 case 直接排名 component causal effect。
4. 找到可复现 component 后，再进入 neuron/channel-level atlas。
```

## Phase 772: Matched Component Discovery Scan（配对组件发现扫描） [2026-06-29 20:39]

### 任务来源

Phase 770 和 Phase 771 的合并判断基本正确：语言机制图谱的准入条件不能停留在 output closure（输出闭合）和 balanced fiber reliability（配平因果纤维可靠性），还必须加入 component-level causal validation（组件级因果验证）。

原因是：

```text
Phase 770:
semantic closure（语义闭合）不能直接等同内部 fiber stability（纤维稳定）。

Phase 771:
即使 output-clean（输出干净）和 fiber-high（纤维高稳定）成立，
沿用 Phase 755 的 top head/source path（顶部注意力头 / 源路径）
仍不能稳定解释生成闭合。
```

所以 Phase 772 不再继续复用 Phase 755 的旧候选，而是在 matched cases（配对样本）内部重新扫描 layer/head/source_group（层 / 注意力头 / 源组），先发现候选，再做 source contribution removal（源贡献移除）因果验证。

### 测试脚本和结果文件

新增脚本：

```text
tests/glm5/phase772_matched_component_discovery_scan.py
tests/glm5/run_phase772_matched_component_discovery_scan_round.sh
```

结果目录：

```text
tests/result/phase772_matched_component_discovery_scan/
results/glm5_phase772_matched_component_discovery_scan/
```

完成三轮测试：

```text
smoke:
tests/glm5/run_phase772_matched_component_discovery_scan_round.sh smoke \
  --max-pairs 1 --max-cases 2 --max-per-stratum 1 \
  --max-scan-layers 2 --max-source-groups 2 \
  --top-components-per-case 1 --log-every 1

main:
tests/glm5/run_phase772_matched_component_discovery_scan_round.sh main \
  --max-pairs 6 --max-cases 8 --max-per-stratum 1 \
  --max-scan-layers 4 --max-source-groups 4 \
  --top-components-per-case 3 --log-every 2

confirm:
tests/glm5/run_phase772_matched_component_discovery_scan_round.sh confirm \
  --focus clean_fiber_high --max-pairs 10 --max-cases 5 \
  --max-per-stratum 1 --max-scan-layers 4 --max-source-groups 4 \
  --top-components-per-case 2 --include-controls \
  --control-offset 5 --log-every 1
```

三轮测试均按 qwen3、GLM4、DS7B 顺序运行；bf16（bfloat16）加载；未使用量化；attention extraction（注意力抽取）需要 eager attention（急切注意力实现）。

### 测试原理

对每个 matched case（配对样本），先捕获指定层的 attention（注意力）和 value（值向量），然后对每个 source group（源组）计算某个 head（注意力头）从源词元到 answer position（答案位置）的源贡献：

$$
C_g(l,h \mid x)
=
\sum_{t \in g}
\alpha_{l,h}(p,t \mid x)
V_{l,h}(t \mid x)
$$

其中：

```text
g = source group（源组）
l = layer（层）
h = head（注意力头）
p = answer position（答案位置）
t = source token position（源词元位置）
```

然后把该源贡献投影回 residual stream（残差流），并用 unembedding（反嵌入读出矩阵）估计它对 target token（目标词元）和 route tokens（路线词元）的直接影响：

$$
S(l,h,g \mid x)
=
w_t \cdot max(0,\Delta y)
+
w_r \cdot max(0,-\Delta R)
+
w_m \cdot max(0,\Delta M)
+
w_a \cdot A_g
$$

其中：

```text
Delta y = direct target boost（直接目标增强）
Delta R = route suppression（路线抑制）
Delta M = margin gain（边际增强）
A_g = attention mass to source group（到源组的注意力质量）
```

这个分数只用于 discovery（发现），不作为最终证据。真正证据来自 causal removal（因果移除）：

$$
h' = h - Proj_O(C_g(l,h \mid x))
$$

观察：

$$
\Delta \ell_y
=
\ell_y(x)-\ell_y(x')
$$

$$
\Delta Margin
=
[\ell_y(x)-\ell_c(x)]
-
[\ell_y(x')-\ell_c(x')]
$$

如果扫描出来的组件是真正的机制组件，它应当比 same-layer control head（同层控制头）产生更大的 target logit drop（目标分数下降）、margin drop（边际下降）或 top1 loss（第一名损失）。

### 客观结果

#### 冒烟测试

smoke（冒烟轮）确认脚本可以完整跑通：

```text
qwen3:
scan top component target drop = 2.562
margin drop = 2.781
top1 loss = 0.000

GLM4:
scan top component target drop = 0.375
margin drop = 0.344
top1 loss = 0.000

DS7B:
scan top component target drop = -0.250
margin drop = -0.094
top1 loss = 0.000
```

冒烟轮只验证链路，不用于机制结论。

#### 主测试

main（主测试）在 all_matched（全部配对样本）上得到：

| model（模型） | rows（行数） | cases（样本数） | mean target drop（平均目标下降） | mean margin drop（平均边际下降） | top1 loss rate（第一名损失率） |
|---|---:|---:|---:|---:|---:|
| qwen3 | 24 | 8 | 0.740 | 0.656 | 0.042 |
| GLM4 | 24 | 8 | 0.070 | 0.109 | 0.000 |
| DS7B | 24 | 8 | 0.557 | 0.289 | 0.083 |

主要复现组件：

```text
qwen3:
L32:attn_out:H5:instruction
n = 6 cases
target drop = 2.125
margin drop = 1.812
top1 loss rate = 0.167

GLM4:
L34:attn_out:H1:instruction
n = 5 cases
target drop = 0.0875
margin drop = 0.075
top1 loss rate = 0

DS7B:
L27:attn_out:H23:instruction
n = 4 cases
target drop = 0.469
margin drop = -0.344
top1 loss rate = 0

DS7B:
L23:attn_out:H11:instruction 和 L27:attn_out:H24:instruction
各自 n = 1
target drop = 0.938
top1 loss rate = 1.0
```

主测试说明：Phase 772 的重新扫描确实比 Phase 771 沿用旧候选更有信号，尤其 qwen3 和 DS7B。但 DS7B 的强 top1 loss 组件样本数仍只有 1，不能当作稳定通路。

#### 确认测试

confirm（确认轮）聚焦 clean_fiber_high（输出干净且纤维高稳定）样本，并加入 same-layer control head（同层控制头）。

| model（模型） | candidate kind（候选类型） | rows（行数） | cases（样本数） | target drop（目标下降） | margin drop（边际下降） | top1 loss（第一名损失） |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | scan_top_component | 10 | 5 | 0.600 | 0.825 | 0.200 |
| qwen3 | same_layer_control_head | 10 | 5 | 0.037 | -0.006 | 0.000 |
| GLM4 | scan_top_component | 10 | 5 | 0.025 | 0.087 | 0.000 |
| GLM4 | same_layer_control_head | 10 | 5 | 0.013 | 0.009 | 0.000 |
| DS7B | scan_top_component | 10 | 5 | 0.512 | 0.156 | 0.000 |
| DS7B | same_layer_control_head | 10 | 5 | 0.044 | 0.025 | 0.000 |

确认轮最重要结果：

```text
qwen3:
扫描候选明显强于同层控制头，
并出现 top1 loss = 0.20。

DS7B:
扫描候选明显强于同层控制头，
但没有 top1 loss。

GLM4:
扫描候选略强于控制头，
但绝对效应非常弱。
```

确认轮也显示所有 top components（顶部组件）几乎都来自 instruction source group（指令源组）。这是一个重要现象，但必须谨慎解释：可能是任务模板中的 instruction token（指令词元）承载了格式 / 目标约束，也可能是当前 source group 切分没有细分到真正语义载体。

### 是否正确

Phase 772 的方向正确，而且比 Phase 771 前进了一步。

正确部分：

```text
1. 不再复用 Phase 755 的旧候选，而是在 matched cases 内重新发现组件。
2. 先 discovery（发现）再 intervention（干预），证据层级更清楚。
3. 加入 same-layer control head 后，qwen3 和 DS7B 的扫描候选明显强于控制。
4. 结果支持 component-level causal validation 作为第三门槛。
```

需要收紧的部分：

```text
1. qwen3 有最强正结果，但 confirm 轮只有 5 个 clean_fiber_high cases。
2. DS7B 有稳定 target drop，但 top1 loss 在 confirm 轮消失。
3. GLM4 整体效应弱，不能证明三模型共享同一强组件机制。
4. 当前发现的是 head/source 级组件，不是 neuron/channel 级组件。
5. 当前 source group 几乎全落在 instruction，可能受模板结构影响。
6. direct score 只是筛选器，不是因果证据。
```

因此最稳妥结论是：

```text
Phase 772 找到了比 Phase 755 旧候选更强的 matched component candidates。
qwen3 和 DS7B 支持“组件发现扫描”这条路线。
GLM4 只支持弱正结果。
当前还没有完成 neuron/channel atlas。
```

### 当前理论收紧

机制图谱准入公式应从 Phase 771 的三门槛继续细化为：

$$
R_{atlas}(x,c)
=
R_{output}(x)
\land
R_{fiber}^{balanced}(x)
\land
R_{component}^{discovered}(x,c)
\land
R_{component}^{causal}(x,c)
$$

其中：

$$
R_{component}^{discovered}(x,c)
=
\mathbf{1}
\left[
Score(c \mid x)
>
Score(control(c) \mid x)
\right]
$$

$$
R_{component}^{causal}(x,c)
=
\mathbf{1}
\left[
\Delta \ell_y^{do(c)} > \tau_l
\lor
\Delta Margin^{do(c)} > \tau_m
\lor
Top1Loss^{do(c)} = 1
\right]
$$

也就是说，component（组件）不能只因为在旧图谱里排名高就进入 atlas（图谱）；它必须在当前 matched context（配对语境）内被重新发现，并在 causal removal（因果移除）中强于控制。

### 硬伤和瓶颈

```text
1. 仍是小模型结果，内部机制可能偏移。
2. 仍是 head/source 粒度，不能直接推出神经元级编码机制。
3. source contribution removal 是局部干预，不等于完全删除该信息流。
4. instruction source group 过强，说明当前任务格式可能主导了发现。
5. clean/fiber_high 不是充分条件，必须进入组件级验证。
6. top1 loss 稀少，说明很多组件只影响读出强度，不一定控制最终生成。
7. scan layers 是根据近期证据选的局部层，不是全层全头扫描。
8. direct score 和 causal effect 之间仍可能错配。
```

### 智能理论角度的关键洞察

本阶段支持一个更谨慎的判断：

```text
语言机制图谱不是静态语义图谱，
也不是单个 head / channel 的排名表，
而是 matched context 中可重复发现、可因果干预的路线组件图谱。
```

更接近真实编码机制的对象不是：

```text
apple -> fruit 的静态边
```

而是：

```text
在给定语境下，
哪些 source tokens（源词元）
通过哪些 components（组件）
改变 target / contrast / route competition（目标 / 对照 / 路线竞争）。
```

当前最重要的拼图是：

```text
instruction source group 在 qwen3 和 DS7B 中反复成为强 causal component source。
```

这提示语言系统里可能存在一种“任务约束 / 输出协议”路线，它不等于语义本身，但会强烈调制语义读出。

### 下一步

下一阶段不应马上进入 neuron/channel atlas（神经元 / 通道图谱），而应先解决 instruction source group 过强的问题。

建议 Phase 773：

```text
Phase 773:
Instruction Source Disentanglement and Object-Route Reweighting
（指令源解耦与对象路线重加权）
```

核心任务：

```text
1. 把 instruction source group 细分为 task prefix / format cue / answer constraint。
2. 把 object_tokens / relation_tokens 的扫描权重和样本结构单独加强。
3. 比较 instruction-only、object-only、relation-only、mixed-source 的 causal effect。
4. 对 qwen3 的 L32H5/L33H16 和 DS7B 的 L27H21/L27H23 做组件复测。
5. 只有当 object/relation source 也出现稳定因果组件后，再进入 neuron/channel 粒度。
```

阶段性结论：

```text
Phase 772 是正结果，但不是最终闭合。

它证明：
matched component discovery scan 比复用旧 top candidates 更有效。

它没有证明：
当前已经找到完整语言编码机制或神经元级图谱。

下一步必须拆开 instruction source 的格式效应和 object/relation source 的语义效应。
```

## Phase 773: Instruction Source Disentanglement and Object-Route Reweighting（指令源解耦与对象路线重加权） [2026-06-29 21:03]

### 任务来源

新上传的 Phase 772 分析基本正确。Phase 772 是正结果，但不是最终闭合。它证明了：

```text
不能继续沿用旧阶段 top head（顶部注意力头）；
必须在 matched context（配对语境）内部重新发现 component（组件），
再做 causal removal（因果移除）验证。
```

但 Phase 772 的最大硬伤也很清楚：

```text
最强 source group（源组）几乎都来自 instruction（指令）。
```

这可能意味着两种完全不同的机制：

```text
1. instruction（指令）本身是任务约束 / 输出协议路线；
2. instruction 里混入了 candidate list（候选值列表），真正强的是候选列表，而不是自然语义。
```

因此 Phase 773 的目标不是继续找更强 head，而是拆开 instruction source（指令源），比较：

```text
instruction_core（常识指令）
format_cue（短答案格式约束）
candidate_list（候选值列表）
answer_prefix（答案前缀）
object_tokens（对象词元）
relation_tokens（关系词元）
semantic_pair（对象 + 关系）
task_frame_without_semantic（去掉对象/关系后的任务框架）
```

### 测试脚本和结果文件

新增脚本：

```text
tests/glm5/phase773_instruction_source_disentanglement.py
tests/glm5/run_phase773_instruction_source_disentanglement_round.sh
```

结果目录：

```text
tests/result/phase773_instruction_source_disentanglement/
results/glm5_phase773_instruction_source_disentanglement/
```

完成三轮测试：

```text
smoke:
tests/glm5/run_phase773_instruction_source_disentanglement_round.sh smoke \
  --focus clean_fiber_high --max-pairs 4 --max-cases 2 \
  --max-per-stratum 1 --max-scan-layers 2 --max-source-groups 4 \
  --top-components-per-source 1 --top-global-components-per-case 0 \
  --log-every 1

main:
tests/glm5/run_phase773_instruction_source_disentanglement_round.sh main \
  --focus clean_fiber_high --max-pairs 10 --max-cases 5 \
  --max-per-stratum 1 --max-scan-layers 4 --max-source-groups 8 \
  --top-components-per-source 1 --top-global-components-per-case 0 \
  --log-every 1

confirm:
tests/glm5/run_phase773_instruction_source_disentanglement_round.sh confirm \
  --focus clean_fiber_high --max-pairs 10 --max-cases 4 \
  --max-per-stratum 1 --max-scan-layers 4 --max-source-groups 8 \
  --top-components-per-source 1 --top-global-components-per-case 0 \
  --include-controls --control-offset 5 --log-every 1
```

三轮均按 qwen3、GLM4、DS7B 顺序运行；bf16（bfloat16）加载；未使用量化；attention extraction（注意力抽取）使用 eager attention（急切注意力实现）。

### 测试原理

Phase 772 的 source group（源组）太粗：

```text
instruction = common instruction + format cue + allowed values
```

Phase 773 把它拆成：

$$
G_{instruction}
=
G_{core}
\cup
G_{format}
\cup
G_{candidate}
$$

并继续保留语义源：

$$
G_{semantic}
=
G_{object}
\cup
G_{relation}
$$

对每个 source group（源组）、每个 layer/head（层 / 注意力头）计算源贡献：

$$
C_g(l,h \mid x)
=
\sum_{t \in g}
\alpha_{l,h}(p,t \mid x)
V_{l,h}(t \mid x)
$$

然后每个 source group 至少取一个 top component（顶部组件）做因果移除：

$$
h' = h - Proj_O(C_g(l,h \mid x))
$$

观察：

$$
\Delta \ell_y
=
\ell_y(x)-\ell_y(x')
$$

$$
\Delta Margin
=
[\ell_y(x)-\ell_c(x)]
-
[\ell_y(x')-\ell_c(x')]
$$

关键改进是：不再让 global ranking（全局排序）把 object / relation（对象 / 关系）源压掉，而是强制每个源组都有被测试机会。

### 客观结果

#### 主测试

main（主测试）在 clean_fiber_high（输出干净且纤维高稳定）样本上，每个模型 5 个 case（样本），每个 case 覆盖 8 个源组。

| model（模型） | 最强源组 | target drop（目标下降） | margin drop（边际下降） | top1 loss（第一名损失） |
|---|---|---:|---:|---:|
| qwen3 | candidate_list | 0.975 | 1.512 | 0.200 |
| GLM4 | candidate_list | 0.000 | 0.106 | 0.000 |
| DS7B | candidate_list | 0.650 | 0.456 | 0.000 |

qwen3 的其他源组：

```text
instruction_core:
target drop = 0.075
margin drop = 0.138

semantic_pair:
target drop = 0.025
margin drop = 0.062

object_tokens:
target drop = -0.025
margin drop = 0.013

relation_tokens:
target drop = 0.025
margin drop = 0.037
```

DS7B 的其他源组：

```text
semantic_object:
target drop = 0.037
margin drop = 0.188

semantic_mixed:
target drop = -0.013
margin drop = 0.144

semantic_relation:
target drop = -0.037
margin drop = -0.013
```

GLM4 整体仍然弱：

```text
candidate_list:
target drop = 0.000
margin drop = 0.106

其他源组 target drop 基本在 -0.013 到 0.025 之间。
```

#### 确认测试

confirm（确认轮）加入 same-layer control head（同层控制头），每模型 4 个 case。

整体候选 vs 控制：

| model（模型） | candidate target drop（候选目标下降） | control target drop（控制目标下降） | candidate margin drop（候选边际下降） | control margin drop（控制边际下降） |
|---|---:|---:|---:|---:|
| qwen3 | 0.176 | 0.020 | 0.248 | -0.004 |
| GLM4 | 0.021 | 0.008 | 0.025 | 0.008 |
| DS7B | 0.139 | 0.016 | 0.137 | 0.012 |

candidate_list（候选值列表）确认结果：

```text
qwen3:
candidate_list top component:
target drop = 1.000
margin drop = 1.750

candidate_list same-layer control:
target drop = -0.094
margin drop = -0.141

GLM4:
candidate_list top component:
target drop = 0.125
margin drop = 0.141

candidate_list same-layer control:
target drop = 0.016
margin drop = -0.016

DS7B:
candidate_list top component:
target drop = 0.844
margin drop = 0.547

candidate_list same-layer control:
target drop = 0.000
margin drop = 0.031
```

这说明 Phase 772 的 instruction dominance（指令源主导）需要进一步改写为：

```text
不是 instruction_core 主导；
也不是 format_cue 主导；
而是 candidate_list / allowed-values protocol（候选列表 / 允许值协议）主导。
```

### 是否正确

Phase 773 的方向正确，并且是 Phase 772 之后必要的收紧。

正确部分：

```text
1. 它拆开了 instruction 的内部结构。
2. 它强制每个 source group 都被测试，避免 candidate_list 压制语义源。
3. 它用 same-layer control 验证 candidate_list 不是简单同层噪声。
4. 它客观证明 qwen3 和 DS7B 的强效应主要来自 candidate_list。
```

最重要的新结果：

```text
Phase 772 所谓 instruction source 强，
更准确地说是 candidate_list source 强。
```

这对当前语言机制图谱是关键纠偏。

### 严格问题和硬伤

```text
1. 当前任务包含 allowed values（允许值列表），这本身就是强外部候选集合。
2. candidate_list 强可能是候选集合读出机制，不是自然语言语义机制。
3. object/relation 源弱，不代表语义不存在，可能只是当前 allowed-values prompt 把语义压力转移给候选列表。
4. GLM4 整体仍弱，三模型不形成统一强机制。
5. qwen3 的 main 有 top1 loss，但 confirm 中 top1 loss 消失，说明 winner control 仍不稳定。
6. DS7B 有 strong target/margin drop，但没有 top1 loss，更像读出强度调节。
7. 仍是 head/source 粒度，不是 neuron/channel 粒度。
8. clean_fiber_high 样本数量仍有限。
```

### 理论收紧

Phase 773 后，机制图谱准入不能只写成：

$$
R_{component}^{causal}(x,c)
$$

还必须标明 source family（源族）：

$$
R_{component}^{causal}(x,c,g)
$$

其中：

$$
g \in
\{
candidate,
instruction,
format,
object,
relation,
query,
prefix
\}
$$

当前结果支持：

$$
R_{candidate}^{causal}
>
R_{object}^{causal}
\approx
R_{relation}^{causal}
$$

至少在 allowed-values commonsense task（带候选值列表的常识任务）中成立。

更严格的图谱准入应写成：

$$
R_{atlas}(x,c,g)
=
R_{output}(x)
\land
R_{fiber}^{balanced}(x)
\land
R_{component}^{discovered}(x,c,g)
\land
R_{component}^{causal}(x,c,g)
\land
R_{source}^{identified}(g)
$$

其中：

$$
R_{source}^{identified}(g)
=
\mathbf{1}
[
g \text{ 已经从 protocol / semantic / candidate 中解耦}
]
$$

### 智能理论角度的关键洞察

本阶段对“语言编码机制”有一个重要提醒：

```text
模型在当前任务里不一定直接从 object/relation semantic source 读出答案；
它可能先利用 candidate list 建立候选竞争空间，
再用较弱语义信号或格式信号完成 winner selection（胜出选择）。
```

也就是说，当前 allowed-values 任务更像：

```text
候选集合约束下的读出竞争机制
```

而不是完整自然语义生成机制。

这不推翻前面的结果，但要求我们把机制图谱分成两层：

```text
1. candidate-conditioned closure（候选条件闭合）
2. free semantic closure（自由语义闭合）
```

当前强结果主要属于第一层。

### 下一步

下一阶段仍属于当前大阶段，因为它直接解决 Phase 773 暴露出的核心硬伤。

建议 Phase 774：

```text
Candidate-List Ablation and Free-Semantic Transfer
（候选列表消融与自由语义迁移）
```

核心任务：

```text
1. 对同一批 clean_fiber_high cases，构造 with_candidate_list / without_candidate_list 两种 prompt。
2. 比较 candidate_list 移除后，object_tokens / relation_tokens 是否增强。
3. 检查 qwen3 和 DS7B 的 target drop 是否从 candidate_list 转移到 object/relation。
4. 如果无候选列表后机制崩溃，说明当前图谱主要是候选条件闭合。
5. 如果无候选列表后 object/relation 增强，说明可以继续进入自由语义图谱。
```

阶段性结论：

```text
Phase 773 是关键纠偏阶段。

它证明：
Phase 772 的强 instruction source 主要来自 candidate_list，
不是自然 instruction_core，也不是 object/relation semantic source。

当前研究应暂时暂停 neuron/channel atlas，
先确认 candidate-conditioned closure 是否能迁移到 free semantic closure。
```

## Phase 774: Candidate-List Ablation and Free-Semantic Transfer（候选列表消融与自由语义迁移） [2026-06-29 21:29]

### 任务来源

Phase 773 证明 Phase 772 中最强的 source（源）不是自然 instruction_core（指令核心），也不是 object / relation semantic source（对象 / 关系语义源），而是 candidate_list（候选列表）这一类 allowed-values protocol（允许值协议）。

因此 Phase 774 的核心问题变成：

```text
如果移除 candidate_list，
当前图谱是否仍能依靠 object_tokens / relation_tokens 形成自由语义闭合？
```

### 测试脚本和结果位置

新增脚本：

```text
tests/glm5/phase774_candidate_list_ablation_transfer.py
tests/glm5/run_phase774_candidate_list_ablation_transfer_round.sh
```

测试结果：

```text
tests/result/phase774_candidate_list_ablation_transfer/smoke/
tests/result/phase774_candidate_list_ablation_transfer/main/
tests/result/phase774_candidate_list_ablation_transfer/confirm/
results/glm5_phase774_candidate_list_ablation_transfer/
```

### 测试原理

Phase 774 比较两个条件：

```text
P_with  = prompt(object, relation, candidate_list)
P_free  = prompt(object, relation, no_candidate_list)
```

如果语义机制已经自然闭合，应满足：

```text
BaseTop1(P_free) 接近 BaseTop1(P_with)
```

并且：

```text
Effect(object/relation | P_free) >= Effect(object/relation | P_with)
```

### 数学表示

```text
y = R(h_L, A)
```

其中：

```text
h_L = 最终层隐藏状态
A   = allowed answer set（允许答案集合 / 候选列表）
R   = readout closure（读出闭合函数）
```

候选条件闭合：

```text
C_candidate(x) =
1[
  Top1(R(h_L(x), A)) = y*
]
```

自由语义闭合：

```text
C_free(x) =
1[
  Top1(R(h_L(x), empty)) = y*
]
```

### 主轮结果

主轮使用每模型 5 个 clean_fiber_high cases。

```text
qwen3:
  with_candidate_list:
    base_top1_rate = 1.000
    mean_base_target_rank = 1.000
    target_drop = 0.138
  without_candidate_list:
    base_top1_rate = 0.000
    mean_base_target_rank = 1168.800
    target_drop = 0.031

GLM4:
  with_candidate_list:
    base_top1_rate = 1.000
    mean_base_target_rank = 1.000
    target_drop = 0.005
  without_candidate_list:
    base_top1_rate = 0.000
    mean_base_target_rank = 45.800
    target_drop = -0.004

DS7B:
  with_candidate_list:
    base_top1_rate = 0.600
    mean_base_target_rank = 2.200
    target_drop = 0.102
  without_candidate_list:
    base_top1_rate = 0.000
    mean_base_target_rank = 4177.000
    target_drop = 0.033
```

主轮最关键结果：

```text
qwen3 / with_candidate_list / candidate_protocol:
  target_drop = 0.975
  margin_drop = 1.512
  direct_boost = 16.360

DS7B / without_candidate_list / semantic_mixed:
  target_drop = 0.437
  margin_drop = 0.905
  direct_boost = 3.419
```

### 确认轮结果

确认轮使用每模型 3 个 cases，并加入 same_layer_control_head 对照。

```text
qwen3:
  with_candidate_list / source_group_top_component:
    base_top1_rate = 1.000
    target_drop = 0.214
    margin_drop = 0.268
  without_candidate_list / source_group_top_component:
    base_top1_rate = 0.000
    mean_base_target_rank = 1941.667
    target_drop = 0.058
    margin_drop = 0.097

GLM4:
  with_candidate_list / source_group_top_component:
    base_top1_rate = 1.000
    target_drop = 0.010
  without_candidate_list / source_group_top_component:
    base_top1_rate = 0.000
    mean_base_target_rank = 72.667
    target_drop = -0.010

DS7B:
  with_candidate_list / source_group_top_component:
    base_top1_rate = 0.333
    mean_base_target_rank = 3.000
    target_drop = 0.109
  without_candidate_list / source_group_top_component:
    base_top1_rate = 0.000
    mean_base_target_rank = 6958.667
    target_drop = 0.091
```

DS7B 在无候选列表下仍有 semantic_mixed 强信号：

```text
without_candidate_list / semantic_mixed:
  target_drop = 0.916
  margin_drop = 1.190
  direct_boost = 3.360
```

但由于 base_top1_rate = 0.000，mean_base_target_rank = 6958.667，所以它不是稳定自由语义输出闭合。

### 客观结论

```text
Phase 774 支持：
当前 Phase 772-773 找到的强图谱主要是 candidate-conditioned closure，
不是完整 free semantic closure。
```

更具体地说：

```text
1. qwen3：
   候选列表是输出闭合的强必要条件。
   移除 candidate_list 后，object / relation 没有接管。

2. GLM4：
   候选列表可以让输出变得稳定，
   但内部因果效应整体较弱。

3. DS7B：
   移除候选列表后仍有部分 semantic_mixed 效应，
   但正确目标没有进入自然 top1，
   因此不是自由语义闭合。
```

### 问题、硬伤和瓶颈

```text
1. 无候选列表同时改变了输出空间，不能简单解释为模型没有语义。
2. DS7B 的 semantic_mixed 是重要线索，但不是输出闭合证据。
3. 当前仍是 head/source 级别，不是 neuron/channel 级别。
4. 小模型可能存在语义路线不完整、读出竞争不稳定、格式协议过强的问题。
```

### 阶段性判断

```text
Phase 772: 证明 matched component discovery scan 能找到强候选组件。
Phase 773: 证明强 source 主要来自 candidate_list。
Phase 774: 证明 candidate_list 移除后，自由语义闭合没有自然接管。
```

当前真实结论：

```text
已经找到 candidate-conditioned route atlas 的关键组件，
但还没有完成 free-semantic route atlas。
```

### 下一步

建议 Phase 775：

```text
Semantic Latent Route vs Output Closure Audit
（语义潜在路线与输出闭合审计）
```

核心测试：

```text
1. 保持 without_candidate_list。
2. 不只看 top1 输出，而是检查 correct target 在候选词、同类词、属性词附近的 logit / rank / hidden-state direction。
3. 测试 semantic_mixed 强组件是否能提升目标 rank，即使不能直接提升 top1。
4. 增加 constrained_free_prompt（弱约束自由提示），避免 candidate_list 直接提供答案集合，但保留输出格式。
5. 对 DS7B 的 semantic_mixed 线索做确认轮。
```

如果 Phase 775 证明 hidden state 中已有语义方向，但 readout closure 失败，后续任务应转向 readout bridge（读出桥），而不是继续堆更多 candidate-conditioned head atlas。

## Phase 775: Semantic Latent Route vs Output Closure Audit（语义潜在路线与输出闭合审计） [2026-06-29 22:05]

### 任务来源

Phase 774 证明 candidate_list 移除后，开放词表 top1 输出闭合失败。但这不能直接解释为“没有语义”，因为无候选列表同时改变了输入信息源和输出空间。

Phase 775 的核心任务是区分：

```text
1. 语义路线不存在。
2. 语义路线存在，但没有完成开放词表读出闭合。
```

### 测试脚本和结果位置

新增脚本：

```text
tests/glm5/phase775_semantic_latent_route_output_closure.py
tests/glm5/run_phase775_semantic_latent_route_output_closure_round.sh
```

测试结果：

```text
tests/result/phase775_semantic_latent_route_output_closure/smoke/
tests/result/phase775_semantic_latent_route_output_closure/main/
tests/result/phase775_semantic_latent_route_output_closure/confirm/
results/glm5_phase775_semantic_latent_route_output_closure/
```

### 测试原理

定义：

```text
V_all = 全词表
V_rel = 当前 relation 对应的合理值池
y*    = 正确答案 token
```

开放词表输出闭合：

```text
C_open(x) =
1[
  argmax_{v in V_all} logit(v | x) = y*
]
```

关系值池内语义选择：

```text
C_pool(x) =
1[
  argmax_{v in V_rel} logit(v | x) = y*
]
```

潜在语义命中：

```text
L_latent(x) =
1[
  C_open(x) = 0
  and
  C_pool(x) = 1
]
```

### 主轮结果

主轮每模型 5 个 clean_fiber_high cases。

```text
qwen3:
  without_candidate_list:
    base_top1_rate = 0.000
    latent_pool_hit_rate = 0.800
    pool_target_top1_rate = 0.800
    mean_base_target_rank = 1168.800
  constrained_free_prompt:
    base_top1_rate = 0.000
    latent_pool_hit_rate = 0.800
    pool_target_top1_rate = 0.800
    mean_base_target_rank = 498.200

GLM4:
  without_candidate_list:
    base_top1_rate = 0.000
    latent_pool_hit_rate = 0.800
    pool_target_top1_rate = 0.800
    mean_base_target_rank = 45.800
  constrained_free_prompt:
    base_top1_rate = 0.000
    latent_pool_hit_rate = 0.600
    pool_target_top1_rate = 0.600
    mean_base_target_rank = 42.000

DS7B:
  without_candidate_list:
    base_top1_rate = 0.000
    latent_pool_hit_rate = 0.800
    pool_target_top1_rate = 0.800
    mean_base_target_rank = 4177.000
  constrained_free_prompt:
    base_top1_rate = 0.000
    latent_pool_hit_rate = 0.800
    pool_target_top1_rate = 0.800
    mean_base_target_rank = 1420.000
```

主轮结论：

```text
三模型在无候选列表或弱约束自由提示下，开放 top1 都失败；
但关系值池内经常已经选中正确答案。
```

### 确认轮结果

确认轮每模型 3 个 cases，并加入 same_layer_control_head。

```text
qwen3:
  without_candidate_list:
    base_top1_rate = 0.000
    latent_pool_hit_rate = 0.667
    pool_target_top1_rate = 0.667
  constrained_free_prompt:
    base_top1_rate = 0.000
    latent_pool_hit_rate = 0.667
    pool_target_top1_rate = 0.667

GLM4:
  without_candidate_list:
    base_top1_rate = 0.000
    latent_pool_hit_rate = 0.667
    pool_target_top1_rate = 0.667
  constrained_free_prompt:
    base_top1_rate = 0.000
    latent_pool_hit_rate = 1.000
    pool_target_top1_rate = 1.000

DS7B:
  without_candidate_list:
    base_top1_rate = 0.000
    latent_pool_hit_rate = 0.667
    pool_target_top1_rate = 0.667
  constrained_free_prompt:
    base_top1_rate = 0.000
    latent_pool_hit_rate = 0.667
    pool_target_top1_rate = 0.667
```

DS7B 的因果信号最强：

```text
without_candidate_list / semantic_mixed:
  target_drop = 0.916
  pool_margin_drop = 0.960
  direct_boost = 3.360

constrained_free_prompt / semantic_object:
  target_drop = 0.365
  pool_margin_drop = 0.594
  direct_boost = 2.654
```

### 综合客观结论

Phase 775 对 Phase 774 做了关键修正：

```text
Phase 774 证明：
candidate_list 移除后，开放词表输出闭合失败。

Phase 775 进一步证明：
开放输出失败不等于语义路线不存在；
三模型在无候选列表或弱约束自由提示下，经常已经在关系值池内选中正确答案。
```

当前更准确的机制图谱：

```text
object / relation / semantic_pair
        ↓
semantic latent route（潜在语义路线）
        ↓
relation value pool preference（关系值池偏好）
        ↓
readout bridge 缺失
        ↓
open-vocabulary closure 失败
```

candidate_list 的真实作用应收紧为：

```text
candidate_list 更像是把潜在语义偏好接到有限候选读出空间上的 readout bridge，
而不是简单的“全部语义来源”。
```

### 问题和硬伤

```text
1. value pool 是外部评估集合，不是模型自然生成集合。
2. 当前只检查 first token，后续需要 multi-token answer scoring。
3. pool_top1 不等于自然输出闭合。
4. GLM4 observation 强，但 causal effect 弱。
5. DS7B 有较强 semantic latent route，但仍没有完成开放词表读出。
```

### 阶段性结论

```text
Phase 772: matched component discovery scan 可以找到强组件。
Phase 773: 强 source 很大程度来自 candidate_list。
Phase 774: 移除 candidate_list 后，开放词表输出闭合失败。
Phase 775: 开放输出失败不等于没有语义；三模型经常已经在 relation value pool 内选中正确值。
```

当前最稳妥结论：

```text
已经观察到自由语义潜在选择，
但还没有完成自由语义开放读出闭合。
```

### 下一步

Phase 775 已完成当前小阶段目标。下一阶段应转入：

```text
Phase 776: Readout-Bridge Competition Audit
（读出桥竞争审计）
```

核心任务：

```text
1. 在 without_candidate_list / constrained_free_prompt 下，记录全词表 top-k 竞争 token。
2. 判断正确值被哪些 token 压制。
3. 对 top-k competitor 做 route 分类。
4. 测试 semantic latent route 已经 pool_top1 的 case 中，哪些 competitor route 阻止 C_open。
5. 找从 C_pool 到 C_open 的 readout bridge。
```
