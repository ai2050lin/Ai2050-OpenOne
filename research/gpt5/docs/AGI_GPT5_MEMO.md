# AGI GPT5 Memo

## Phase 195: Phase944 Integrated Mechanism Trace and Visualization Client [2026-07-05]

### 目标

本阶段把前面讨论的三个方案整合为一个可运行、可保存、可展示的机制分析闭环：

```text
方案一：不只记录最大神经元，而是记录 embedding -> residual stream -> attention/MLP -> W_U -> next token 的完整计算脉络。
方案二：同步记录 O/R/A/C/F/M/K/S_answer/B/G/N/P/T 机制因子。
方案三：把生成过程组织成 上文状态 -> 条件化路由 -> 候选空间打开 -> 知识路径激活 -> 输出边界竞争 -> 下一 token。
```

本阶段优先目标是颜色编码机制，测试 prompt 为：

```text
A red cube is placed on the table. The color of the cube is
```

目标 token 为：

```text
red
```

### 生成脚本与文件

新增核心脚本：

```text
tests/glm5/phase944_integrated_mechanism_trace.py
```

新增前端可视化组件：

```text
frontend/src/components/mechanism/MechanismTraceExplorer.jsx
frontend/src/components/mechanism/MechanismTraceExplorer.css
```

修改前端入口：

```text
frontend/src/App.jsx
```

生成前端静态数据：

```text
frontend/public/vis_data/mechanism_trace/manifest.json
frontend/public/vis_data/mechanism_trace/phase944_qwen3_red_cube_trace.json
frontend/public/vis_data/mechanism_trace/phase944_glm4_red_cube_trace.json
frontend/public/vis_data/mechanism_trace/phase944_deepseek7b_red_cube_trace.json
```

生成原始实验结果：

```text
tests/result/phase944_integrated_mechanism_trace/phase944_qwen3_red_cube_trace/phase944_qwen3_integrated_mechanism_trace.json
tests/result/phase944_integrated_mechanism_trace/phase944_glm4_red_cube_trace/phase944_glm4_integrated_mechanism_trace.json
tests/result/phase944_integrated_mechanism_trace/phase944_deepseek7b_red_cube_trace/phase944_deepseek7b_integrated_mechanism_trace.json
```

### 测试原理

本阶段不再只寻找“最大的几个神经元”，而是记录完整的生成链路。

核心流程：

```text
1. 输入 prompt，得到 token embedding。
2. 对每一层保存 answer position 的 residual state。
3. 通过 forward hook 捕获每层 attention output 和 MLP output。
4. 将每层 residual、attention、MLP 向量投影到 W_U 的候选 token 行。
5. 构造颜色候选场，计算 target color 与 strongest competitor 的竞争关系。
6. 同步计算 O/R/A/C/F/M/K/S_answer/B/G/N/P/T 机制因子。
7. 保存为 mechanism_trace_v1 JSON，供前端直接读取和展示。
```

核心读出公式：

```text
score_l(token) = h_l · W_U[token]
```

颜色候选场：

```text
p_l(c) = softmax({h_l · W_U[c] | c in Colors})
```

目标颜色边际：

```text
K_l = score_l(target_color) - max_{c != target_color} score_l(c)
```

候选门控：

```text
M_l = 1 - H(p_l) / log(|Colors|)
```

边界齿轮：

```text
G_l = K_l - K_{l-1}
```

自然 gate：

```text
N_l = ||component_l|| / ||residual_l||
```

全词表闭合与候选闭合分开记录：

```text
global_closed = target_global_rank == 1
candidate_closed = target_rank_within_color_candidates == 1
```

这个区分很关键：候选颜色内部 red 排第一，不代表全词表下一 token 一定输出 red。

### 机制因子定义

```text
O = object factor，对象因子，h_l · W_U[token(object)]
R = relation factor，关系因子，h_l · W_U[token(relation)]
A = attribute factor，属性因子，h_l · (W_U[target_attribute] - mean(W_U[other_attributes]))
C = category factor，类别因子，颜色类别整体读出均值
F = function factor，功能因子，answer/word/color 等协议 token 的读出均值
M = candidate gate，候选门控，颜色候选分布的尖锐程度
K = knowledge path，知识路径，目标颜色相对最强颜色竞争者的 margin
S_answer = semantic answer field，语义答案场，目标答案 token 的直接读出
B = blocker field，阻断场，最强错误颜色候选读出
G = boundary gear，边界齿轮，本层 margin 相对上一层的变化
N = natural gate，自然 gate，attention/MLP 更新幅度相对 residual 的比例
P = output protocol，输出协议，协议 token 相对颜色 token 的读出偏置
T = termination action，终止动作，句号、换行、EOS 等终止 token 的读出强度
```

### 三模型结果

Qwen3：

```text
target_global_rank = 1
next_token = " red"
final_margin_vs_color_competitor = 5.172491073608398
final_competitor_label = blue
global_closed = true
candidate_closed = true
recorded_layers = 37
```

GLM4：

```text
target_global_rank = 1
next_token = " red"
final_margin_vs_color_competitor = 3.9285354614257812
final_competitor_label = blue
global_closed = true
candidate_closed = true
recorded_layers = 41
```

DeepSeek7B：

```text
target_global_rank = 5
next_token = " determined"
final_margin_vs_color_competitor = 0.4698371887207031
final_competitor_label = white
global_closed = false
candidate_closed = true
recorded_layers = 29
```

### 关键发现

本阶段最重要发现是：

```text
候选场闭合不等于全词表输出闭合。
```

DeepSeek7B 在颜色候选集合内部已经让 red 排第一，但全词表真实 next token 是 " determined"，red 的全局 rank 只有 5。

这说明颜色机制至少有两层闭合：

```text
1. 局部语义候选闭合：颜色候选空间内部选对。
2. 全局输出协议闭合：全词表竞争中目标 token 真正成为 next token。
```

因此后续破解编码机制时，不能只看某个语义类别内部的 margin，还必须检查全词表输出协议、终止动作、自然语言续写惯性和 blocker field。

### 前端可视化

前端新增“机制 Trace”入口：

```text
左上角 GitBranch 图标按钮
```

可视化客户端现在可以：

```text
1. 读取 frontend/public/vis_data/mechanism_trace/manifest.json。
2. 选择 qwen3 / glm4 / deepseek7b 的测试结果。
3. 查看 prompt、tokens、目标对象、目标属性、下一 token。
4. 查看每层 residual margin 曲线。
5. 查看 O/R/A/C/F/M/K/S_answer/B/G/N/P/T 每个机制因子。
6. 查看颜色候选分数、概率、target rank、competitor。
7. 查看 attention 和 MLP component 对候选场的贡献。
8. 查看当前层 raw JSON 和完整 Trace JSON。
```

本地客户端：

```text
http://127.0.0.1:5174/
```

数据接口验证：

```text
/vis_data/mechanism_trace/manifest.json -> 200 OK
/vis_data/mechanism_trace/phase944_qwen3_red_cube_trace.json -> 200 OK
/vis_data/mechanism_trace/phase944_glm4_red_cube_trace.json -> 200 OK
/vis_data/mechanism_trace/phase944_deepseek7b_red_cube_trace.json -> 200 OK
```

### 工程验证

脚本验证：

```text
python -m py_compile tests/glm5/phase944_integrated_mechanism_trace.py -> OK
```

前端构建：

```text
npm run build -> OK
```

注意：

```text
Vite build 仍提示部分 bundle 大于 500 kB，这是既有前端体积问题，不影响本阶段功能。
```

### 理论进展

本阶段把“颜色编码特征发现”推进为“可复用的全链路机制 Trace”。

新的理论判断：

```text
同一个神经网络参数能在不同上下文中生成正确下文，
不是因为单个神经元保存了完整答案，
而是因为上下文状态在 residual stream 中打开了特定候选空间，
再通过 attention/MLP 的条件化路由、
W_U 输出边界竞争、
输出协议和终止动作共同完成 next-token closure。
```

因此一个更准确的破解路线应是：

```text
context state
  -> conditioned routing
  -> candidate-space opening
  -> knowledge-path activation
  -> output-boundary competition
  -> next-token closure
```

而不是：

```text
找到最大神经元 = 找到编码机制
```

### 下一步任务

建议下一阶段进入 Phase945：

```text
1. 扩展 prompt 数据集，不只测试 red cube。
2. 对颜色、形状、材质、类别、关系分别生成批量 Trace。
3. 对每个模型统计哪些层稳定打开 candidate field。
4. 区分 candidate_closed 和 global_closed 的失败类型。
5. 对 DeepSeek7B 的 global_closed=false 做专项审计：
   - 是否 output protocol 太弱
   - 是否 continuation inertia 太强
   - 是否 answer position 需要更明确的 instruction prompt
   - 是否 blocker field 来自非颜色 token，而不是颜色候选内部
6. 将 Trace JSON 进一步压缩为可检索机制图谱：
   node = factor/layer/token/component
   edge = margin gain/component contribution/routing relation
```

阶段性结论：

```text
Phase944 已经完成从“特征神经元列表”到“全链路机制脉络”的第一版转型。
它还没有完成全局编码图谱，但已经提供了可预测、可验证、可复用的图谱数据格式和可视化入口。
```

## Phase 196: Phase944 机制 Trace 后的研究方案与客户端改进审计 [2026-07-05 07:19]

### 一、任务范围

本阶段读取并分析了 `research/gpt5/docs/AGI_GPT5_MEMO.md` 的最新记录。当前 GPT5 备忘录最新阶段是 Phase195，标题为：

```text
Phase944 Integrated Mechanism Trace and Visualization Client
```

需要注意编号关系：

```text
GPT5 memo phase = 195
GLM5 / test script phase = 944
```

因此本阶段作为 GPT5 备忘录的 Phase196 追加，不把两个阶段序列混用。

本阶段没有运行新的 CUDA 模型测试，也没有改动前端代码；工作性质是研究方案审计和客户端方案设计。读取对象包括：

```text
research/gpt5/docs/AGI_GPT5_MEMO.md
frontend/src/components/mechanism/MechanismTraceExplorer.jsx
frontend/src/components/mechanism/MechanismTraceExplorer.css
frontend/public/vis_data/mechanism_trace/manifest.json
tests/glm5/phase944_integrated_mechanism_trace.py
```

### 二、最新记录的核心结论复核

Phase195 / Phase944 的核心进展是把单一颜色任务拆成了可观察的机制链：

```text
context state
  -> conditioned routing
  -> candidate-space opening
  -> knowledge-path activation
  -> output-boundary competition
  -> next-token closure
```

其中最关键的发现是：

```text
candidate_closed != global_closed
```

也就是：

```text
候选颜色空间内部选对 red
不等于
全词表真实 next token 一定输出 red
```

三模型结果中，Qwen3 和 GLM4 同时完成候选闭合与全局闭合；DeepSeek7B 只完成候选闭合，没有完成全词表闭合：

```text
Qwen3:
candidate_closed = true
global_closed = true
next_token = " red"

GLM4:
candidate_closed = true
global_closed = true
next_token = " red"

DeepSeek7B:
candidate_closed = true
global_closed = false
next_token = " determined"
target_global_rank = 5
```

这说明“语义答案场”与“输出协议场”至少是两个不同齿轮。破解语言编码不能只看语义候选集合内的 margin，还必须同时追踪全词表阻塞者、协议续写惯性、终止动作和 answer-position 条件。

### 三、现有研究方案的主要问题

当前 Phase944 的方案已经比单点神经元搜索更接近真实机制，但仍有几个硬伤。

#### 1. 样本太小，不能外推为机制规律

当前可视化数据只有：

```text
red cube / color
3 个模型
每个模型 1 条 prompt
```

这个结果只能证明 Trace 框架可用，不能证明颜色机制、属性机制或输出协议机制已经稳定定位。尤其 DeepSeek7B 的失败可能来自 prompt 风格、tokenization、模型 instruction habit、颜色词竞争，也可能来自真正的 output protocol gap。

#### 2. 候选空间过窄

当前 candidate field 主要是颜色候选。这个设计适合验证 red cube，但还不能解释语言能力的完整结构。至少需要扩展到：

```text
颜色: red / blue / green / white / black ...
形状: cube / sphere / cylinder / pyramid ...
材质: wood / metal / plastic / glass ...
类别: animal / tool / fruit / vehicle ...
关系: color-of / shape-of / made-of / located-in / used-for ...
动作: move / fall / open / close ...
否定: is / is not
比较: bigger / smaller / same / different
```

只有当同一套 Trace 因子能覆盖这些任务，才能说它在逼近语言背后的通用数学结构，而不是为颜色任务定制了一套解释。

#### 3. 缺少失败分类学

Phase944 已经发现 DeepSeek7B 的失败，但还没有把失败分成可检索类型。建议把失败类型至少拆成：

```text
F1: candidate_not_open
候选空间没有打开，目标在候选内部也不占优。

F2: candidate_open_global_blocked
候选空间内部选对，但全词表被非候选 token 阻塞。

F3: candidate_competitor_error
候选空间打开，但强竞争者是同类错误候选。

F4: protocol_inertia_error
模型进入解释、续写、列表、限定词等协议惯性。

F5: termination_or_format_error
语义答案已出现，但停止/格式控制失败。

F6: tokenization_alias_error
目标词的空格 token、大小写、同义词、词片段导致评估口径错位。
```

其中 DeepSeek7B 当前最像：

```text
F2 + F4
candidate_open_global_blocked + protocol_inertia_error
```

但还不能确认，需要下一阶段通过批量 prompt 和 blocker token 审计验证。

#### 4. 当前因子多，但缺少“必要性”和“可替代性”测试

Phase944 记录了 O/R/A/C/F/M/K/S_answer/B/G/N/P/T 这些机制因子。但这些因子目前主要是观测量，还不是因果变量。

下一步要从：

```text
factor trace
```

升级为：

```text
factor necessity / sufficiency / substitutability test
```

也就是分别问：

```text
去掉这个因子，闭合是否消失？
增强这个因子，闭合是否恢复？
换成另一个同类因子，行为是否按预测改变？
```

否则因子曲线容易变成解释性图表，而不是编码机制。

### 四、改进后的研究方案

建议下一阶段不要只做一个更漂亮的 Trace，而是建立“批量机制审计流水线”。阶段目标应是：

```text
从单样本机制 Trace
升级为
跨任务、跨模型、跨失败类型的机制图谱构建器
```

#### 1. 数据集设计

先做一个中等规模、结构清晰的数据集，不急于复杂自然语言。建议：

```text
属性类:
  颜色 8 类 × 物体 20 类 = 160 条
  形状 6 类 × 物体 20 类 = 120 条
  材质 6 类 × 物体 20 类 = 120 条

关系类:
  located-in / used-for / part-of / made-of 各 80 条 = 320 条

控制类:
  改写 prompt 5 个模板
  answer-prefix 明确/不明确 2 个条件
  instruction / plain completion 2 个条件
```

最小第一轮：

```text
每模型 300-500 条
三模型合计 900-1500 条
```

重要结论复测：

```text
每模型 1000+ 条
三模型合计 3000+ 条
```

这符合当前项目要求：重要测试不能用太少样本得结论。

#### 2. 核心指标

保留 Phase944 的指标，但增加闭合分层：

```text
candidate_rank = 目标在候选集合内的 rank
global_rank = 目标在全词表内的 rank
blocker_type = 全词表 top1 的类型
margin_candidate = target_score - max(other_candidate_score)
margin_global = target_logit - top_non_target_logit
protocol_bias = mean(protocol_token_logits) - mean(answer_candidate_logits)
termination_bias = mean(termination_logits) - target_logit
closure_gap = global_rank - candidate_rank
```

最重要的新公式是：

```text
closure_gap = global_rank - candidate_rank
```

若：

```text
candidate_rank = 1
global_rank > 1
```

则说明语义候选已经闭合，但输出边界没有闭合。

再定义：

```text
protocol_block = logit(top_protocol_or_continuation) - logit(target)
```

若：

```text
protocol_block > 0
```

则说明全局失败来自协议/续写 token，而不是同类语义候选。

#### 3. 实验分层

建议按三层推进：

```text
Layer A: Observation Trace
记录每层 O/R/A/C/F/M/K/S_answer/B/G/N/P/T。

Layer B: Failure Taxonomy
把每条样本归类为 F1-F6。

Layer C: Causal Repair
对每类失败做最小干预：增强答案场、抑制协议场、移动 answer-position、替换 prompt 模板。
```

这比直接寻找“大统一神经元”更稳，因为它先把失败形状拆开，再定位因果齿轮。

#### 4. DeepSeek7B 专项

DeepSeek7B 当前是最有价值的失败样本。下一步不应只记录它失败，而要做四个对照：

```text
1. prompt 明确化:
   "Answer with one color word only:"

2. answer prefix 对照:
   "The answer is"
   "Color:"
   "It is"

3. blocker 审计:
   top1 token 是否属于 explanation / adjective / continuation / format token

4. target injection repair:
   在不破坏 prompt 的前提下增强 S_answer 或 P，看 global_closed 是否恢复
```

如果 prompt 明确化即可修复 DeepSeek7B，那么失败更偏协议场；如果必须增强 S_answer 才修复，则说明答案场强度不足；如果两者都不能修复，则需要追踪更早层的 routing / candidate opening。

### 五、客户端改进方案

当前客户端 `MechanismTraceExplorer` 已经完成单 Trace 展示，包括：

```text
manifest 读取
模型选择
prompt / token / summary 展示
层列表
factor tiles
candidate table
attention / mlp component table
raw JSON 查看
```

下一步客户端应从“查看器”升级为“机制审计工作台”。

#### 1. 增加跨模型对比视图

当前每次只能看一个 trace。建议增加：

```text
Compare Mode
```

核心展示：

```text
Qwen3 / GLM4 / DeepSeek7B 同一 prompt 的并排摘要
candidate_closed / global_closed 对比
target_global_rank 对比
final_margin 对比
top blocker 对比
关键层 margin 曲线对比
```

这能直接暴露“同一语义候选闭合，为什么某模型全局失败”。

#### 2. 增加失败类型过滤器

manifest item 需要追加字段：

```text
failure_type
closure_gap
blocker_type
task_type
attribute_type
prompt_template
```

前端增加筛选：

```text
只看 candidate_closed=true && global_closed=false
只看 protocol_blocker
只看 color / shape / material
只看 DeepSeek7B 失败样本
```

这样客户端才能支持批量研究，而不是打开一个 JSON 看一次。

#### 3. 增加关键层自动定位

当前层列表只显示 margin。建议自动标出：

```text
candidate_open_layer:
M 第一次超过阈值的层

knowledge_jump_layer:
G 最大正跃迁层

global_block_layer:
target_global_rank 开始进入 topK 但没有 top1 的层

protocol_takeover_layer:
P 或 blocker field 超过 S_answer 的层
```

前端可用不同小标签标记：

```text
OPEN
JUMP
BLOCK
TAKEOVER
```

#### 4. 增加曲线视图

目前因子以 tile 展示，不利于看动力学。建议增加：

```text
Factor Curves
```

至少画：

```text
K_l: target vs competitor margin
M_l: candidate gate
S_answer_l: answer field
B_l: blocker field
P_l: protocol field
G_l: boundary gear
```

研究上最有用的是叠加：

```text
S_answer_l - B_l
K_l
P_l
```

如果：

```text
K_l > 0
但
S_answer_l - B_l < 0
```

就说明候选语义正确，但全词表边界仍被阻塞。

#### 5. 增加 Trace 到图谱的导出入口

Phase195 已经提出 node/edge 图谱：

```text
node = factor/layer/token/component
edge = margin gain/component contribution/routing relation
```

客户端应该提供：

```text
Export Graph
```

生成：

```text
mechanism_graph_v1.json
```

基本结构：

```text
nodes:
  layer_factor nodes
  token nodes
  component nodes

edges:
  layer_to_layer margin_delta
  component_to_candidate contribution
  blocker_to_output competition
```

这能把单条 Trace 压缩成可检索、可比较的机制图谱。

#### 6. 增加审计标注层

客户端需要允许研究者给样本添加人工判定：

```text
failure_type
important_layer
suspected_mechanism
next_test
notes
```

保存位置建议为：

```text
tests/result/mechanism_trace_audit/
```

或者导出为浏览器下载 JSON，后续再统一汇总进仓库。

这一步很重要，因为当前项目处于“拼图阶段”，许多洞察需要人工严谨标注，而不是一开始就强行自动化。

### 六、理论进展

本阶段的理论收紧是：

```text
语言输出不是单一语义向量读出，
而是至少由候选语义场、全词表边界场、协议续写场共同决定。
```

更基础地说，当前证据支持一个三层模型：

```text
1. semantic candidate field
   决定“可能答案集合里哪个对”

2. global output boundary
   决定“全词表下一 token 真实输出什么”

3. protocol / termination field
   决定“模型是在回答、解释、续写、列举还是停止”
```

因此破解语言背后的数学结构，第一性问题不应写成：

```text
red 编码在哪个神经元？
```

而应写成：

```text
在给定上下文状态下，模型如何把一个候选语义场提升为全词表边界中的实际动作？
```

这个问题更接近语言能力的本体，因为语言不是静态表征，而是状态到动作的闭合。

### 七、严格审视：问题、硬伤和瓶颈

#### 1. 当前 Trace 仍然偏 logit lens

很多指标来自 residual 到 W_U 的投影。这能提供有用线索，但不能直接等价于模型自然计算过程。LayerNorm、后续层变换和非线性路由可能改变解释。

#### 2. attention / MLP component 粒度仍粗

当前只看到 attention 和 MLP 总贡献，还没有分 head、MLP channel、source token。真正机制很可能藏在更细粒度的组合中。

#### 3. 因果闭合不足

Phase944 是强观察框架，不是强因果框架。后续必须加入最小干预和 repair 测试，否则容易形成“看起来合理”的解释图。

#### 4. tokenization 可能污染结论

不同模型对 `" red"`、`"red"`、大小写、标点、换行的 tokenization 不同。必须在每个样本中记录 target token variants，否则 global_rank 可能被评估口径影响。

#### 5. 客户端还不能承载大样本

当前单 JSON 展示没问题，但当 trace 增加到几百或几千条时，需要索引、过滤、聚合和延迟加载。否则研究者仍然只能人工翻样本，无法形成机制地图。

### 八、阶段性大任务

下一阶段建议不是“小修 UI”，而是启动一个大任务：

```text
Phase197:
Batch Mechanism Trace Atlas and Failure Taxonomy
批量机制 Trace 图谱与失败分类学
```

阶段目标：

```text
1. 生成 300-500 条结构化 prompt；
2. 三模型逐个 CUDA 测试，避免显存溢出；
3. 输出 mechanism_trace_batch_v1；
4. 自动归类 closure state 和 failure_type；
5. 前端支持 batch manifest、过滤、跨模型对比和关键层标记；
6. 对 DeepSeek7B 的 candidate_closed=true/global_closed=false 做专项复测；
7. 从 batch trace 生成 mechanism_graph_v1。
```

成功标准不是“画出更多曲线”，而是：

```text
能回答每一类失败为什么失败；
能预测某类 prompt 是否会 global_closed；
能用最小干预修复至少一类失败；
能把单样本 Trace 压缩成跨模型可比较的机制图谱。
```

### 九、通俗总结

Phase944 已经证明：模型不是简单地“知道 red 就输出 red”。它可能已经在颜色候选里选中了 red，但最后真正输出时，被解释性词、续写习惯或其他全词表 token 抢走了位置。

所以后续研究要从“找答案在哪里”升级为“答案怎样赢得输出权”。客户端也要从“看一条曲线”升级为“批量比较谁赢、谁输、为什么输、在哪一层开始输”。这才更接近破解语言背后数学结构的主路。

## Phase 197: 最新完整智能理论与全局图谱完成方案 [2026-07-05 07:26]

### 一、任务范围

本阶段读取并整理：

```text
research/glm5/docs/AGI_GLM5_MEMO.md
research/IntelligentTheory.md
```

目标是从最新记录中整理当前最完整的理论，并给出完成全局图谱的最可行方案。

本阶段没有运行新的 CUDA 模型测试，也没有新增测试脚本；性质是理论整理、路线收敛和图谱工程方案设计。

### 二、最新事实基础

`research/IntelligentTheory.md` 当前总框架已经从早期的“相对编码 + 条件化变换 + 候选竞争”扩展为：

```text
预测充分相对状态
  -> 全局齿轮图谱
  -> 条件化路线门控
  -> 全词表竞争闭合
  -> 自然生成一致性
```

GLM5 最新 Phase944 则进一步把图谱从 residual coordinate 层推进到 component candidate 层：

```text
static component candidate
  -> activation-weighted component candidate
  -> partial causal gear evidence
```

Phase944 最强正结果来自 qwen3：

```text
qwen3 color en->en
hidden 36 / channels 2509, 16, 249
activation lift = 4.8014
boundary slope = +0.6779
candidate_ablate boundary delta = -0.4471
candidate_boost boundary delta = +0.2308
```

但 GLM4 和 DS7B 给出重要负向约束：

```text
GLM4:
activation lift 强，但 candidate boundary slope 接近 0，
说明强通道信号可能混入通用边界敏感性。

DS7B:
coordinate concentration 很强，
但没有形成稳定正向 boundary causal slope。
```

因此当前不能说已经完成语言编码机制闭合，只能说：

```text
语义残差坐标已经开始接到可测 MLP 通道齿轮；
但自然门控、全词表 blocker、strict-clean rollout 仍未闭合。
```

### 三、当前最新完整理论

当前最稳妥的完整理论应写成：

```text
语言输出不是语义向量直接读出，
而是相对状态网络在上下文中形成候选场，
再经由组件齿轮、协议场、全词表边界和自然生成门共同完成动作闭合。
```

更具体的机制链是：

```text
Input / Prompt Protocol
  -> State Variables
     identity / role / frame / operator / scope / binding
  -> Domain Route Axes
     color / material / animal / tool / geometry / abstract ...
  -> Candidate Field
     candidate_open + candidate_specific_ranking
  -> Component Gears
     MLP channel / attention head / source token route
  -> Boundary Field
     target token vs candidate competitor vs full-vocab blocker
  -> Protocol Field
     short answer / prose / list / echo / punctuation / EOS
  -> Natural Rollout
     first token + phrase likelihood + multi-token clean answer
```

这条链路说明：语言能力不是单点编码，而是“状态-路线-齿轮-边界-生成”的动态系统。

### 四、核心数学表达

#### 1. 状态层

第 `l` 层、位置 `p` 的状态可写成：

```text
h_l(p)
  =
  I_l(p)
  + R_l(p)
  + F_l(p)
  + C_l(p)
  + O_l(p)
  + S_l(p)
  + K_l(p)
  + B_l(p)
  + Q_l(p)
  + N_l(p)
  + eps_l(p)
```

其中：

```text
I = identity，对象/token 身份
R = role，功能角色
F = frame，局部格式
C = construction，构式/语法
O = operator，否定/规则/控制符
S = scope，作用域
K = knowledge anchor，知识锚
B = binding，对象-关系-值绑定
Q = candidate competition，候选竞争
N = norm/gain，范数和读出增益
```

#### 2. 语义答案场

旧公式：

```text
A(y|x) = sum_{o,r} P(o|x) P(r|x) K(o,r,y) g(y|x)
```

仍可作为语义答案场骨架，但 Phase944 证明必须增加可测组件层：

```text
K(o,r,y)
  -> C_consensus(o,r)
  -> Contribution_G(x)
  -> BoundaryMovement(y, B_x)
```

因此更准确的输出闭合公式是：

```text
CleanOutput(y|x)
  =
  SemanticAnswer(y|x)
  and CandidateWinner(y|x)
  and ActivationWeightedGear(G,x)
  and FieldAdmissible(B_x)
  and NaturalGate(G,x)
  and NoProtocolDrift(x)
```

当前 Phase944 主要推进了：

```text
ActivationWeightedGear(G,x)
```

尚未闭合：

```text
NaturalGate(G,x)
NoProtocolDrift(x)
StrictCleanRollout(x)
```

#### 3. 领域坐标轴层

IntelligentTheory 最新 Phase897-898 已经修正了“一个 universal pair 解决所有领域”的过强假设。当前应写成：

```text
语言图谱不是 one universal pair，
而是 domain-specific route axes
和 shared blocker/protocol field 的组合。
```

领域轴评分：

```text
AxisScore(g,d)
  =
  MeanAbsActivation(g | domain=d)
  -
  MeanAbsActivation(g | domain!=d)
```

领域候选集合：

```text
U_d
  =
  TopK_g AxisScore(g,d)
  union
  HistoryAxes(d)
```

领域最小 pair 判定：

```text
DomainNoSinglePair(a,b,x,d)
  =
  BoundaryClosed({a,b},x,d)
  and not BoundaryClosed({a},x,d)
  and not BoundaryClosed({b},x,d)
```

#### 4. 干净因果边标准

全局图谱中的边不能只记录“有效”，必须记录证据等级。当前严格标准应保持为：

```text
CleanCausalEdge
  =
  GearEffect
  and FieldAdmissible
  and OutputTransition
  and NoSideEffect
```

用于全局编码图谱时，还要加：

```text
SemanticFactorMeasurable
and CrossObjectHoldout
and CrossRelationHoldout
and CrossTemplateHoldout
and CrossModelRobustness
and NaturalGate
and StrictCleanRollout
```

### 五、当前理论的最重要结论

#### 1. 编码本体不是单神经元，而是可复用坐标接口 + 组件齿轮

当前证据不支持：

```text
apple neuron
red neuron
one universal semantic vector
```

更接近事实的是：

```text
object-relation coordinate
  -> residual consensus coordinate
  -> sparse MLP / attention component candidates
  -> activation-weighted causal gear
  -> full-vocab boundary movement
```

#### 2. 语义候选闭合不等于输出闭合

候选集合内 target 排第一，只能说明语义候选场有效；全词表里仍可能被 protocol、prose、punctuation、EOS、echo、generic token 抢走输出权。

因此全局图谱必须同时记录：

```text
candidate_closed
global_closed
protocol_drift
blocker_class
rollout_clean
```

#### 3. 领域轴优先于通用轴

Phase897-898 已经说明，非颜色 domain 需要先建立各自的 candidate_U。当前路线不应强行复用 color pair 到 material / animal / tool / abstract，而应先为每个 domain 找本域坐标轴，再比较结构同构。

#### 4. 图谱比闭合更优先

目前最可行路线不是直接宣称语言机制闭合，而是先完成：

```text
证据校准全局齿轮图谱
```

当图谱足够完整后，闭合会自然变成若干可测边的组合问题。

### 六、完成图谱的最可行方案

最可行方案是四层推进：

```text
Layer 1: Domain Axis Atlas
Layer 2: Component Gear Atlas
Layer 3: Boundary / Protocol Atlas
Layer 4: Natural Rollout Closure Atlas
```

不要先追求“大统一公式”，而是把每条边做成可审计对象。

#### Layer 1: Domain Axis Atlas

目标：

```text
为 color / material / animal / tool / geometry / category / function / abstract
分别建立领域坐标轴集合 U_d。
```

每个 domain 记录：

```text
axis_id
model
layer
channel_or_coordinate
domain
AxisScore
single_axis_support
pair_support
holdout_support
failure_cases
```

最小测试规模：

```text
每 domain 至少 100-200 条结构化样本；
每模型至少 800-1500 条；
三模型逐个 CUDA 测试，避免显存溢出。
```

#### Layer 2: Component Gear Atlas

目标：

```text
把 residual consensus coordinate 映射到 MLP channel / attention head / source-token route。
```

每个候选组件必须通过：

```text
1. static contribution
2. activation-weighted contribution
3. ablate/boost causal slope
4. random same-layer control
5. single-channel sign decomposition
6. object/template holdout
```

Phase945 最应优先做：

```text
qwen3 color en->en channels 2509, 16, 249 单通道 ablate/boost；
qwen3 function zh->en channels 106, 2, 3 单通道 ablate/boost；
区分 support / suppressor / mixed side-effect channel。
```

#### Layer 3: Boundary / Protocol Atlas

目标：

```text
解释为什么候选选对后，真实输出仍可能失败。
```

每个样本记录：

```text
candidate_rank
global_rank
target_logit
top_blocker_token
blocker_class
protocol_score
termination_score
closure_gap
```

关键公式：

```text
closure_gap = global_rank - candidate_rank
```

若：

```text
candidate_rank = 1
global_rank > 1
```

说明语义候选闭合，但全词表边界失败。

blocker 分类至少包括：

```text
semantic_competitor
protocol_token
prose_continuation
punctuation
eos_or_stop
echo_token
generic_token
tokenization_alias
```

#### Layer 4: Natural Rollout Closure Atlas

目标：

```text
把 first-token 边界推进到完整自然生成。
```

不能只看 next token，要同时记录：

```text
first_token_closed
phrase_likelihood_winner
short_answer_hit
clear_answer
object_echo
protocol_drift
multi_token_clean
```

长 rollout 稳定性公式：

```text
LongRolloutStable(S,x,T)
  =
  ClassHit(S,x,T)
  and ClearAnswer(S,x,T)
  and not ObjectEcho(S,x,T)
  and not ProtocolDrift(S,x,T)
```

只有到这一层，才接近真正的语言输出闭合。

### 七、图谱数据结构建议

建议把图谱统一为：

```text
mechanism_atlas_v1
```

核心节点：

```text
state_node:
  identity / role / relation / binding / candidate_field

axis_node:
  domain axis / known-axis pair / single-axis route

component_node:
  MLP channel / attention head / source token group

boundary_node:
  target / semantic competitor / blocker class / protocol token

rollout_node:
  first token / phrase / multi-token answer
```

核心边：

```text
state_to_axis
axis_to_component
component_to_boundary
boundary_to_protocol
protocol_to_rollout
```

每条边必须带证据等级：

```text
L1 correlation / clustering
L2 projection node
L3 transition node
L4 component causal node
L5 hidden causal repair node
L6 generation closure
```

当前最重要的是把 L2/L3/L4 分清楚，不要把 projection node 误写成 component causal node。

### 八、最短可执行路线

如果以最快完成“可用图谱”为目标，建议按下面顺序：

```text
Step 1:
整理 Phase897-898 的 domain axis 候选，生成 domain_axis_atlas_v1。

Step 2:
执行 Phase945，先只做 qwen3 最强通道的 single-channel sign decomposition。

Step 3:
为每条样本补充 blocker_class 和 closure_gap，生成 boundary_protocol_atlas_v1。

Step 4:
把 Phase944 integrated mechanism trace 接入 batch manifest，客户端支持按 domain/model/failure_type 过滤。

Step 5:
对 qwen3 的强通道做 300-500 条 holdout；
若稳定，再扩到 GLM4；
DS7B 暂作弱参考。

Step 6:
只选择通过 L4 的边进入 natural rollout；
不要把所有候选都跑长生成，节省 GPU。

Step 7:
把通过 rollout 的边标记为 L6 generation closure candidate。
```

这条路线比全面搜索所有模型所有通道更可行，因为它先抓最强证据链，把图谱格式和证据标准跑通。

### 九、严格审视：问题和硬伤

#### 1. 变量仍不够可测

理论中的：

```text
P(o|x), P(r|x), K(o,r,y), g(y|x)
```

仍需要映射到：

```text
residual coordinate
MLP channel contribution
attention head route
candidate blocker class
natural gate score
rollout transition
```

否则公式仍只是漂亮骨架。

#### 2. group-level 通道可能混合多种功能

同一组 channels 里可能同时存在：

```text
answer lift
blocker weakening
protocol drift
format side-effect
suppressor release
```

所以必须做 single-channel sign decomposition。

#### 3. 人工 ablate/boost 不等于自然门控

当前干预：

```text
a'_j = f a_j
```

只能说明通道能推动边界，不能说明模型自然运行时为什么启动该通道。自然门控仍是最大缺口。

#### 4. 小模型偏差不能忽视

qwen3、GLM4、DS7B 的结构都可能与更大模型或人脑差异很大。GLM4 的通用边界敏感性、DS7B 的坐标集中但边界弱，都说明跨模型结论必须谨慎。

#### 5. 客户端还缺图谱级工作流

现在客户端能看单条 Trace，但图谱完成需要：

```text
batch manifest
domain/failure filters
cross-model comparison
evidence-level badge
edge drill-down
rollout result table
manual audit annotation
```

否则研究者仍然无法从大量碎片中完成拼图。

### 十、理论进展

本阶段的理论收敛是：

```text
智能的基本单位不是概念向量，
而是上下文状态中可复用的对象-关系坐标接口；
这些接口只有通过组件齿轮、边界竞争、协议门和自然生成，
才变成语言动作。
```

从第一性原理看，语言背后的数学结构更像：

```text
相对状态网络上的条件化路线选择系统
```

而不是：

```text
固定欧氏空间中的全局语义坐标系
```

如果要破解这个结构，下一步最关键的不是再提出更高层数学名词，而是把每个变量都落到可测节点、可测边和可复现实验上。

### 十一、下一阶段任务

建议下一阶段进入：

```text
Phase198:
Evidence-Calibrated Mechanism Atlas v1
证据校准机制图谱 v1
```

任务：

```text
1. 定义 mechanism_atlas_v1 JSON schema；
2. 汇总 domain_axis_atlas、component_gear_atlas、boundary_protocol_atlas；
3. 对 qwen3 Phase944 强通道做 Phase945 single-channel sign decomposition；
4. 加入 blocker_class 和 closure_gap；
5. 客户端支持 batch atlas 浏览；
6. 只把 L4 以上边送入 natural rollout；
7. 建立 L1-L6 证据等级面板。
```

阶段成功标准：

```text
不是证明 AGI 理论完成，
而是让每个机制候选都有统一位置、统一证据等级、统一失败原因和下一步测试入口。
```

### 十二、通俗总结

当前最完整的理论可以简单说成：

```text
模型先在上下文里形成“这个对象在这个关系下应该找什么答案”的状态，
再用某些 MLP 通道或 attention 路线推动候选答案，
最后这个候选还要打赢全词表里的解释、续写、标点、停止符和错误答案，
才能真正输出干净答案。
```

所以完成图谱的最可行方法不是一次找出终极公式，而是把这条链拆成可测节点：

```text
领域坐标轴 -> 组件齿轮 -> 边界竞争 -> 协议控制 -> 自然生成
```

每条边都标清楚证据等级。等这些边足够多、足够干净、能跨对象和模板复现，语言背后的数学结构才会自然浮现。

## Phase 198: 单通道符号分解机制图谱初测 [2026-07-05 07:42]

### 一、任务判断

本阶段读取并分析了附件内容。附件对 Phase197 的总体判断基本正确：

```text
1. Phase197 的方向正确：从“语义向量/概念神经元”推进到“相对状态 -> 领域路线 -> 候选场 -> 组件齿轮 -> 全词表边界 -> 协议控制 -> 自然生成”。
2. 当前不能说已经破解语言编码机制。
3. 当前第一优先级应是完成证据校准图谱，而不是继续无目标堆 patch。
4. Phase944 的 group-level MLP channel 结果必须做 single-channel sign decomposition。
```

附件需要收紧的一点是：它提出的多个公式仍有理论化倾向。当前阶段更应优先把它们落到客观可测字段：

```text
channel_id
boundary_slope
relation_slope
target_logit_slope
protocol_margin_slope
single_channel_sign
random_control
evidence_level
```

因此本阶段没有继续做抽象理论总结，而是直接执行一个跨模型单通道符号分解测试。

### 二、测试脚本与结果位置

新增测试脚本：

```text
tests/gpt5/phase198_single_channel_sign_decomposition_atlas.py
tests/gpt5/run_phase198_single_channel_sign_decomposition_atlas.sh
```

结果目录：

```text
tests/result/phase198_single_channel_sign_decomposition_atlas/single_channel_sign_decomposition_atlas/
```

核心输出：

```text
phase198_qwen3_summary.json
phase198_qwen3_rows.jsonl
phase198_qwen3_mechanism_atlas_v1.json

phase198_glm4_summary.json
phase198_glm4_rows.jsonl
phase198_glm4_mechanism_atlas_v1.json

phase198_deepseek7b_summary.json
phase198_deepseek7b_rows.jsonl
phase198_deepseek7b_mechanism_atlas_v1.json

phase198_cross_model_summary.json
phase198_cross_model_summary.md
phase198_cross_model_mechanism_atlas_v1.json
```

工程验证：

```text
python -m py_compile tests/gpt5/phase198_single_channel_sign_decomposition_atlas.py -> OK
```

模型测试顺序：

```text
qwen3 -> GLM4 -> DS7B
```

每个模型测试后释放 GPU，再加载下一个模型，避免显存叠加。

### 三、测试原理

Phase944 测试的是通道组：

```text
G = {c1, c2, c3}
```

本阶段把它拆成单通道：

```text
g = ci
```

对每个通道做：

```text
ablate: a'_g = 0.0 * a_g
boost : a'_g = 1.5 * a_g
```

边界斜率定义为：

```text
boundary_slope(g)
  =
  DeltaBoundary(boost_g)
  -
  DeltaBoundary(ablate_g)
```

关系候选斜率定义为：

```text
relation_slope(g)
  =
  DeltaRelationMargin(boost_g)
  -
  DeltaRelationMargin(ablate_g)
```

单通道符号判定：

```text
support_channel:
  boundary_slope > 0.02
  and relation_slope >= -0.02

suppressor_or_blocker_channel:
  boundary_slope < -0.02

near_zero_or_correlational:
  abs(boundary_slope) <= 0.02
  and abs(relation_slope) <= 0.02

mixed_side_effect_channel:
  其他混合情况
```

这比 Phase944 更严格，因为它不允许把一个通道组的平均正结果直接解释为每个通道都是真正支持齿轮。

### 四、测试规模

本阶段复用 Phase944 的 activation-weighted MLP channel records：

```text
qwen3:
  records = 7
  row-level interventions = 1392

GLM4:
  records = 5
  row-level interventions = 1044

DS7B:
  records = 1
  row-level interventions = 144
```

合计：

```text
records = 13
row-level interventions = 2580
```

每条 record 拆分：

```text
3 candidate channels
3 same-layer random channels
每个 channel 做 ablate / boost
```

### 五、跨模型总结果

跨模型 evidence：

```text
single_channel_support_positive: 1
single_channel_mixed_sign_decomposition_positive: 1
single_channel_suppressor_only: 1
```

对应模型：

```text
qwen3:
  single_channel_support_positive

GLM4:
  single_channel_mixed_sign_decomposition_positive

DS7B:
  single_channel_suppressor_only
```

这说明附件提出的关键担心成立：

```text
group-level 通道正结果确实会混入 support / suppressor / mixed / near-zero 多种通道。
```

### 六、qwen3 客观结果

qwen3 总体结果：

```text
candidate_sign_counts:
  support_channel: 4
  mixed_side_effect_channel: 2
  near_zero_or_correlational: 15
```

最强单通道：

```text
relation = color
language_pair = en->en
hidden_idx = 36
layer_idx = 35
channel = 249
sign = support_channel
rows = 26
boundary_slope = 0.5432692307692308
relation_slope = 0.03245192307692307
target_logit_slope = 0.5865384615384616
ablate_boundary_delta = -0.3581730769230769
boost_boundary_delta = 0.18509615384615385
```

这说明 Phase944 中 qwen3 color en->en 的组级正结果，主要不是平均分散到 2509 / 16 / 249 三个通道，而是由 `channel 249` 提供最强边界推动。

同组内另外两个通道：

```text
channel 16:
  sign = mixed_side_effect_channel
  boundary_slope = 0.12980769230769232
  relation_slope = -0.03365384615384616

channel 2509:
  sign = near_zero_or_correlational
  boundary_slope = 0.0
  relation_slope = 0.0
```

因此 Phase944 里提到的：

```text
qwen3 color en->en hidden 36 / channels 2509,16,249
```

必须修正为：

```text
qwen3 color en->en hidden 36:
  channel 249 是当前最强 support gear；
  channel 16 是 mixed side-effect gear；
  channel 2509 在本阶段单通道测试中近似 near-zero。
```

qwen3 function 方向也有弱正单通道：

```text
function zh->zh channel 2:
  boundary_slope = 0.05598958333333333
  relation_slope = -0.009114583333333332
  sign = support_channel

function zh->zh channel 58:
  boundary_slope = 0.02734375
  relation_slope = 0.013020833333333332
  sign = support_channel

function en->en channel 3:
  boundary_slope = 0.020833333333333336
  relation_slope = 0.04166666666666667
  sign = support_channel
```

qwen3 结论：

```text
qwen3 支持“少数单通道可作为 L4 component causal edge 候选”，
但不支持“Phase944 组内所有 top channels 都是语义支持齿轮”。
```

### 七、GLM4 客观结果

GLM4 总体结果：

```text
candidate_sign_counts:
  mixed_side_effect_channel: 2
  support_channel: 1
  near_zero_or_correlational: 9
  suppressor_or_blocker_channel: 3
```

最强正边界斜率来自：

```text
relation = function
language_pair = zh->en
hidden_idx = 30
layer_idx = 29
channel = 1165
sign = mixed_side_effect_channel
rows = 12
boundary_slope = 0.04817708333333348
relation_slope = -0.02473958333333326
target_logit_slope = -0.05078125
```

它不能解释为干净 support channel，因为：

```text
boundary_slope > 0
但 relation_slope < 0
且 target_logit_slope < 0
```

较干净的弱 support：

```text
relation = color
language_pair = en->en
channel = 1165
sign = support_channel
boundary_slope = 0.03125
relation_slope = 0.004807692307692307
target_logit_slope = 0.021634615384615384
```

明显 suppressor / blocker：

```text
function zh->en channel 5532:
  boundary_slope = -0.02994791666666652

color en->en channel 5532:
  boundary_slope = -0.040865384615384616

function en->en channel 5532:
  boundary_slope = -0.08333333333333393
```

GLM4 结论：

```text
GLM4 的 Phase944 group-level 信号确实高度混合；
同一候选组中存在 support、suppressor、near-zero 和 mixed side-effect。
因此 GLM4 不能作为干净语义齿轮闭合证据，但非常适合作为“通用边界敏感性/混合齿轮”的负向校准样本。
```

### 八、DS7B 客观结果

DS7B 总体结果：

```text
candidate_sign_counts:
  near_zero_or_correlational: 1
  mixed_side_effect_channel: 1
  suppressor_or_blocker_channel: 1
```

三个候选通道：

```text
channel 16221:
  sign = near_zero_or_correlational
  boundary_slope = 0.015625
  relation_slope = 0.0

channel 3033:
  sign = mixed_side_effect_channel
  boundary_slope = -0.005208333333333336
  relation_slope = 0.03125

channel 6030:
  sign = suppressor_or_blocker_channel
  boundary_slope = -0.04166666666666667
  relation_slope = -0.010416666666666668
```

DS7B 结论：

```text
DS7B 的 Phase944 coordinate concentration 没有转化为正向单通道边界齿轮；
本阶段更支持“DS7B 只有弱/混合/抑制性参考”，不能作为机制闭合依据。
```

### 九、对附件判断的核验

附件正确部分：

```text
1. Phase197 主路线正确。
2. 不能把 group-level channel 直接写成语义齿轮。
3. 必须做 single-channel sign decomposition。
4. GLM4 / DS7B 的负结果非常重要，能防止 qwen3 过拟合。
5. 当前测试仍是小模型机制候选图谱，不是语言编码最终图谱。
```

本阶段进一步修正：

```text
1. qwen3 的 strongest gear 从 group {2509,16,249} 收紧到 channel 249。
2. qwen3 channel 2509 在单通道层面不再是强支持齿轮。
3. GLM4 的 channel 1165 在不同 relation / language_pair 下可表现为 support 或 mixed side-effect。
4. GLM4 channel 5532 多处表现为 suppressor / blocker。
5. DS7B 的候选组没有出现 support_channel。
```

### 十、理论进展

本阶段不做大的理论改名，只给出一个客观收紧：

```text
组件齿轮不是“通道组 = 功能单元”，
而是组内存在不同符号的微齿轮。
```

更准确的图谱边应从：

```text
residual consensus coordinate
  -> top MLP channel group
  -> boundary movement
```

收紧为：

```text
residual consensus coordinate
  -> single MLP channel sign
  -> boundary / relation / protocol slope vector
```

其中每个单通道至少要记录：

```text
(boundary_slope, relation_slope, target_logit_slope, protocol_margin_slope)
```

而不是只记录一个“有效/无效”标签。

### 十一、严格审视与硬伤

#### 1. 仍不是自然门控

本阶段仍使用人工缩放：

```text
a'_g = factor * a_g
```

它只能证明通道被缩放时会影响边界，不能证明模型自然状态下何时启动该通道。

#### 2. 仍不是自然 rollout 闭合

本阶段只测 first-token boundary metrics，没有测试完整短答生成。因此最高只能算 L4 component causal candidate，不能算 L6 generation closure。

#### 3. 样本规模仍偏中等

本阶段 row-level interventions 为 2580，但实际 record 数只有 13，DS7B 只有 1 条 record。因此强结论只能给 qwen3 color/function 局部图谱，不能扩展为跨模型通用规律。

#### 4. 随机对照也出现少量 support

例如 DS7B same-layer random channel 3380 出现：

```text
boundary_slope = 0.020833333333333336
relation_slope = 0.03125
```

这说明阈值附近的 weak support 必须谨慎处理，不能只凭 `boundary_slope > 0.02` 就写成干净语义机制。

#### 5. 小模型偏差仍然明显

GLM4 和 DS7B 的结果支持附件提醒：小模型内部机制可能更粗糙，通道更混合，协议/边界场更容易污染语义解释。

### 十二、阶段性结论

Phase198 完成了 Phase197 提出的证据校准图谱中的一个关键子任务：

```text
group-level component candidate
  -> single-channel sign decomposition
  -> mechanism_atlas_v1 edge
```

当前最可靠的正边：

```text
qwen3 color en->en h36 / L35 / MLP channel 249
```

证据：

```text
support_channel
boundary_slope = 0.5432692307692308
relation_slope = 0.03245192307692307
rows = 26
```

当前最重要的负边 / 混合边：

```text
GLM4 channel 5532:
  多处 suppressor_or_blocker_channel

DS7B candidate channels:
  无 support_channel
```

因此当前图谱应新增：

```text
support edge:
  qwen3 L35 C249 color en->en

mixed edge:
  qwen3 L35 C16 color en->en
  GLM4 L29 C1165 function zh->en

suppressor edge:
  GLM4 L29 C5532
  DS7B L13 C6030
```

### 十三、下一阶段是否属于同一阶段

当前阶段目标是：

```text
把 Phase944 的 group-level 通道候选拆成 single-channel sign atlas。
```

这个目标已经完成。

接下来的任务是：

```text
Phase199:
L4 support edge natural-gate and rollout closure audit
```

它与当前任务属于同一条“大路线”：

```text
证据校准全局机制图谱
```

但不属于同一个直接子阶段。原因是它要从：

```text
single-channel first-token boundary
```

进入：

```text
natural gate + multi-token rollout
```

证据等级从 L4 推向 L5/L6，测试目标、指标和脚本结构都需要单独设计。因此本阶段不把 Phase199 混入 Phase198 结果，避免把单通道边界正结果提前写成自然生成闭合。

### 十四、下一阶段任务

Phase199 建议只选最强边，不要大范围铺开：

```text
primary:
  qwen3 color en->en L35 C249

secondary:
  qwen3 function zh->zh L26 C2
  qwen3 function zh->zh L26 C58
  qwen3 function en->en L26 C3

negative controls:
  qwen3 color en->en L35 C2509
  GLM4 L29 C5532
  DS7B L13 C6030
```

测试目标：

```text
1. 自然激活强度是否预测 success / failure；
2. support channel 在自然样本中是否由特定 state variable 启动；
3. ablate C249 是否破坏自然短答；
4. boost C249 是否修复 candidate_closed=true/global_closed=false；
5. 多 token rollout 是否保持 clear_answer 且无 protocol_drift。
```

只有 Phase199 通过后，qwen3 L35 C249 才能从：

```text
L4 component causal candidate
```

升级为：

```text
L5/L6 generation closure candidate
```

### 十五、通俗总结

这次测试证明了一件很关键的事：

```text
之前看到的一组“可能有用的齿轮”，其实不是一个整体齿轮。
里面有真正推动答案的齿轮，也有副作用齿轮、刹车齿轮，还有看起来相关但实际不动边界的齿轮。
```

最清楚的正结果是：

```text
qwen3 的 color en->en 任务里，L35 的 MLP channel 249 是当前最像真齿轮的单通道。
```

但这仍然不是最终破解。下一步要问：

```text
模型自然运行时为什么会启动 C249？
启动后能不能真的让完整答案稳定输出？
```

这就是从“单通道边界齿轮”走向“自然生成闭合”的下一块拼图。

## Phase 199: L4 单通道边到自然门控与短 rollout 闭合审计 [2026-07-05 07:58]

### 一、任务判断

本阶段读取并分析了新附件。附件对 Phase197 和 Phase198 的判断基本正确：

```text
1. Phase197 的机制链路线正确；
2. Phase198 的单通道符号分解是实质进展；
3. qwen3 L35 C249 是当前最清楚的 L4 component causal candidate；
4. 但 C249 仍没有自然门控和自然生成闭合证据；
5. 下一步应从 L4 first-token boundary 推进到 natural gate + rollout audit。
```

这与 Phase198 属于同一条大路线：

```text
证据校准全局机制图谱
```

并且附件明确要求继续完成阶段性目标。因此本阶段继续执行 Phase199。

### 二、测试脚本与结果位置

新增脚本：

```text
tests/gpt5/phase199_l4_edge_natural_gate_rollout_audit.py
tests/gpt5/run_phase199_l4_edge_natural_gate_rollout_audit.sh
```

正式结果目录：

```text
tests/result/phase199_l4_edge_natural_gate_rollout_audit/l4_edge_natural_gate_rollout_audit_strict/
```

注意：本阶段先跑了一轮非严格口径，发现 `clean_generated` 会在句号处截断，例如：

```text
raw generated = "red. What is the most common color"
cleaned = "red"
```

这会把明显的后续协议漂移误判为 clean rollout。因此脚本已修正为同时检查 raw continuation，并以 strict round 作为正式结果。

工程验证：

```text
python -m py_compile tests/gpt5/phase199_l4_edge_natural_gate_rollout_audit.py -> OK
```

模型测试顺序：

```text
qwen3 -> GLM4 -> DS7B
```

### 三、测试目标

Phase198 证明若干单通道可以推动 first-token boundary。Phase199 问更严格的问题：

```text
这些 L4 单通道边是否能影响自然短生成？
```

具体测试：

```text
1. baseline 生成；
2. ablate: a'_g = 0.0 * a_g；
3. boost : a'_g = 1.5 * a_g；
4. 检查 clear_answer；
5. 检查 raw protocol_drift；
6. 检查 long_rollout_stable；
7. 记录自然激活 activation_abs 是否区分 stable / unstable。
```

严格 rollout 稳定标准：

```text
long_rollout_stable
  =
  rollout_clear_answer_class
  and not rollout_object_echo
  and not protocol_drift(raw_generated)
```

其中 protocol_drift 使用 raw generated，不只看清洗后的首短语。

### 四、测试规模

测试边：

```text
qwen3: 7 edges
GLM4: 6 edges
DS7B: 2 edges
total: 15 edges
```

row-level rollout：

```text
qwen3: 360 rows
GLM4: 270 rows
DS7B: 72 rows
total: 702 rows
```

每条边最多 24 个样本，每个样本测试：

```text
baseline
ablate
boost
```

### 五、跨模型总结果

正式 strict round 的核心结果非常清楚：

```text
15 / 15 条边：
  baseline long_rollout_stable = 0
  ablate   long_rollout_stable = 0
  boost    long_rollout_stable = 0
```

也就是说：

```text
没有任何一条 L4 单通道边通过严格自然 rollout 闭合。
```

这不是小修正，而是重要负结果：

```text
first-token boundary gear
  !=
natural generation closure gear
```

### 六、qwen3 结果

qwen3 测试：

```text
rows = 360
edges = 7
```

总计：

```text
baseline stable = 0
ablate stable   = 0
boost stable    = 0

baseline clear_answer = 36
ablate clear_answer   = 36
boost clear_answer    = 36
```

最关键边：

```text
qwen3 color en->en h36 L35 C249
sign = support_channel
Phase198 boundary_slope = 0.5432692307692308
```

Phase199 strict rollout：

```text
baseline_stable = 0
ablate_stable   = 0
boost_stable    = 0

baseline_clear = 11
ablate_clear   = 11
boost_clear    = 11
```

解释：

```text
C249 能推动 first-token boundary；
但在当前短 rollout 中，ablate / boost 都没有改变 clear answer 数量；
raw continuation 中存在续写，因此 strict stable = 0。
```

负控：

```text
qwen3 color en->en C2509 near_zero:
  baseline_clear = 11
  ablate_clear = 11
  boost_clear = 11
  stable 全为 0
```

这说明：

```text
C249 的 L4 边界正效应是真实的；
但它没有自动转化为 L6 natural rollout closure。
```

### 七、GLM4 结果

GLM4 测试：

```text
rows = 270
edges = 6
```

总计：

```text
baseline stable = 0
ablate stable   = 0
boost stable    = 0

baseline clear_answer = 28
ablate clear_answer   = 28
boost clear_answer    = 28
```

GLM4 color en->en C1165：

```text
sign = support_channel
baseline_clear = 10
ablate_clear   = 10
boost_clear    = 10
stable 全为 0
```

GLM4 C5532 suppressor / blocker：

```text
color en->en:
  baseline_clear = 10
  ablate_clear = 10
  boost_clear = 10
  stable 全为 0

function en->en:
  clear 全为 0
  stable 全为 0
```

结论：

```text
GLM4 的边界敏感性没有在本阶段转化为自然短生成控制；
Phase198 的 mixed / suppressor 分类仍然成立，但不是 rollout closure。
```

### 八、DS7B 结果

DS7B 测试：

```text
rows = 72
edges = 2
```

总计：

```text
baseline stable = 0
ablate stable   = 0
boost stable    = 0

baseline clear_answer = 0
ablate clear_answer   = 0
boost clear_answer    = 0
```

测试边：

```text
DS7B function en->en C3033 mixed_side_effect
DS7B function en->en C6030 suppressor_or_blocker
```

结果：

```text
两条边都没有 clear_answer，也没有 stable rollout。
```

这进一步支持 Phase198 的判断：

```text
DS7B 当前只能作为弱/混合/抑制性参考，
不能作为机制闭合依据。
```

### 九、评估口径修正

本阶段最重要的方法论修正是：

```text
不能只用 cleaned generated 判断 rollout closure。
```

原因：

```text
clean_generated 会在 ".", ",", ";", ":" 等位置截断；
这适合判断 first answer span，
但不适合判断 protocol drift。
```

例如：

```text
generated = "red. What is the most common color"
```

清洗后可能得到：

```text
red
```

如果只看 cleaned，就会误以为 clean rollout 成立。严格口径必须同时检查 raw continuation。

因此后续所有 rollout closure 都应同时记录：

```text
generated_raw
generated_clean
clear_answer_from_clean
protocol_drift_from_raw
strict_rollout_stable
```

### 十、理论进展

本阶段不提出新理论，只给出一个重要负边：

```text
L4 first-token boundary edge
  -/-> L6 natural rollout closure
```

当前更准确的图谱层级是：

```text
single MLP channel sign
  -> first-token boundary movement
  -> first answer span
  -> raw continuation protocol control
  -> strict rollout closure
```

Phase198 完成了：

```text
single MLP channel sign -> first-token boundary movement
```

Phase199 证明尚未完成：

```text
first-token boundary movement -> strict rollout closure
```

### 十一、严格审视与硬伤

#### 1. 当前 prompt 本身容易诱发续写

例如：

```text
A common color for an apple is
```

模型自然输出：

```text
red. What is the most common color ...
```

这说明 prompt protocol 本身没有强约束“只回答一个词并停止”。因此 Phase199 的负结果可能部分来自 prompt protocol，而不是 C249 完全无用。

#### 2. 单通道缩放可能不足以控制多 token 策略

C249 可能只影响第一步答案 token 边界，而后续停止/续写由其他协议齿轮控制。不能要求 C249 单独完成：

```text
answer selection + stop control
```

#### 3. strict stable = 0 是真实负结果，但不是最终否定

它证明：

```text
当前测试条件下，L4 边没有升级到 L6。
```

但不能证明：

```text
C249 对所有生成闭合都无用。
```

下一步需要加入更明确的 prompt protocol 和 stop/protocol edge。

#### 4. 小模型偏差仍然重要

qwen3 / GLM4 / DS7B 在短生成中都有明显续写、格式、协议问题。小模型可能把答案选择和终止控制拆得更粗糙，因此结果不能直接外推到大模型或人脑。

### 十二、阶段性结论

Phase199 是一个重要负结果：

```text
qwen3 L35 C249 是强 L4 first-token boundary support edge，
但不是 L6 natural rollout closure edge。
```

跨模型共同事实：

```text
1. first-token clear answer 可以出现；
2. ablate / boost 单通道不改变 clear answer 数量；
3. raw continuation 产生 protocol drift；
4. strict long_rollout_stable 全部为 0。
```

因此当前图谱应新增一条明确边界：

```text
L4 component causal edge
  does not imply
L6 generation closure
```

### 十三、下一阶段是否属于同一阶段

当前 Phase199 的直接目标是：

```text
测试 L4 support edge 是否能自然升级到 L5/L6 rollout closure。
```

该目标已经完成，结果为负。

下一阶段仍属于同一条大路线：

```text
证据校准全局机制图谱
```

但直接子目标应切换为：

```text
Protocol-Gated Rollout Repair Audit
协议门控的 rollout 修复审计
```

它不应继续盲目扩大 C249 patch，而应加入：

```text
1. 明确 short-answer prompt protocol；
2. stop / punctuation / EOS / prose continuation 分类；
3. C249 + protocol edge 的组合测试；
4. raw continuation drift 作为硬指标。
```

因此本阶段不继续自动混入下一子阶段，避免把 Phase199 的负结果用更强 prompt 直接掩盖。

### 十四、下一阶段任务

建议 Phase200：

```text
Protocol-Gated Rollout Repair Audit
协议门控的 rollout 修复审计
```

优先测试：

```text
qwen3 color en->en L35 C249
```

对照条件：

```text
prompt_plain:
  A common color for an apple is

prompt_short_answer:
  Answer with one English color word only. The color of an apple is

prompt_stop_explicit:
  Answer with one English color word and then stop. The color of an apple is
```

干预组合：

```text
baseline
C249 ablate
C249 boost
protocol suppress / stop edge candidate
C249 boost + protocol suppress
```

目标不是证明 C249 单独闭合，而是测试：

```text
answer-selection gear + protocol gate
```

是否必须组合才能形成 strict rollout closure。

### 十五、通俗总结

这次测试说明：

```text
C249 像是能把第一个答案词推向 red 的齿轮，
但它不是让模型“只回答 red 然后停下”的完整机制。
```

模型会先说出正确开头，然后继续写别的东西。也就是说：

```text
选答案的齿轮
和
回答到哪里停止的齿轮
不是同一个东西。
```

下一步要找的是这两个齿轮如何配合，而不是继续单独放大 C249。

## Phase 200: 协议门控 rollout 修复审计 [2026-07-05 17:10]

### 一、任务来源与判断

本阶段分析了附件中对 Phase199 的判断。附件的核心判断基本正确：

```text
L4 first-token boundary gear != L6 natural rollout closure gear
```

Phase199 的严格 raw generated 审计说明，qwen3 L35 C249 等边可以影响第一答案词边界，但不能自动保证模型只输出一个答案词后停止。因此不能把 C249 直接解释为完整回答闭合机制。正确的下一步不是继续盲目放大 C249，而是测试：

```text
answer-selection gear + prompt/protocol gate
```

能否形成严格 rollout closure。

### 二、本阶段完成内容

新增正式测试脚本：

```text
tests/gpt5/phase200_protocol_gated_rollout_repair_audit.py
tests/gpt5/run_phase200_protocol_gated_rollout_repair_audit.sh
```

结果保存位置：

```text
tests/result/phase200_protocol_gated_rollout_repair_audit/protocol_gated_rollout_repair_audit/
```

跨模型汇总文件：

```text
phase200_cross_model_summary.json
phase200_cross_model_summary.md
```

执行顺序为：

```text
qwen3 -> GLM4 -> DS7B
```

每个模型测试完成后释放 GPU 显存，再加载下一个模型。

### 三、测试原理

本阶段不再只测试自然 prompt，而是加入三类 prompt protocol：

```text
plain:
  原始自然问题

short_answer:
  Answer with one English color/verb/category word only...

stop_explicit:
  Answer with exactly one English color/verb/category word and no explanation...
```

每条候选边测试三种条件：

```text
baseline
ablate: channel factor = 0.0
boost: channel factor = 1.5
```

严格稳定性仍使用 Phase199 的 raw generated 判据，不使用清洗后的短 answer span 判断闭合。

### 四、基本数学记号

设候选通道为第 \(l\) 层 MLP 通道 \(c\)，通道激活为：

```text
a_{l,c}
```

干预后激活为：

```text
a'_{l,c} = s · a_{l,c}
```

其中：

```text
s = 0.0   ablate
s = 1.5   boost
```

对每个样本记录三个核心指标：

```text
clear = 是否出现目标答案类别
drift = raw continuation 是否发生协议漂移
stable = clear 且没有 protocol drift 且没有 object echo
```

本阶段真正关心的因果差分是：

```text
Δstable_boost = stable(boost) - stable(baseline)
Δstable_ablate = stable(baseline) - stable(ablate)
```

若 prompt protocol 能提升 stable，但通道 ablate/boost 的 Δstable 为 0，则说明修复主要来自外部协议提示，不是该单通道内部机制。

### 五、客观结果

总数据量：

```text
qwen3: 612 rows
GLM4: 585 rows
DS7B: 216 rows
cross-model condition rows: 108
protocol eval rows: 36
```

#### 1. qwen3

聚合结果：

```text
plain baseline:        rows=68, stable=0, clear=19, drift=63
short_answer baseline: rows=68, stable=0, clear=22, drift=68
stop_explicit baseline:rows=68, stable=0, clear=16, drift=68
```

ablate 与 boost 后：

```text
所有 qwen3 条件 stable = 0
所有 qwen3 protocol/channel 组合 Δstable_boost = 0
所有 qwen3 protocol/channel 组合 Δstable_ablate = 0
```

特别是核心边：

```text
qwen3|color|en->en|h36|c249|support_channel
```

在 plain、short_answer、stop_explicit 三种协议下，baseline/ablate/boost 的 stable 全部为 0。

#### 2. GLM4

GLM4 出现了有限的 prompt protocol 修复：

```text
plain baseline:        rows=65, stable=0,  clear=22, drift=65
short_answer baseline: rows=65, stable=6,  clear=14, drift=45
stop_explicit baseline:rows=65, stable=23, clear=29, drift=16
```

但是通道干预没有带来 stable 增益：

```text
short_answer:
  baseline stable=6
  ablate stable=6
  boost stable=6

stop_explicit:
  baseline stable=23
  ablate stable=23
  boost stable=23
```

非零 stable 主要集中在 GLM4 的 color en->en 与部分 function 条件上，例如：

```text
glm4|color|en->en|h30|c1165|support_channel
stop_explicit: baseline/ablate/boost stable = 8/8/8

glm4|color|en->en|h30|c5532|suppressor_or_blocker_channel
stop_explicit: baseline/ablate/boost stable = 8/8/8
```

这说明显式停止 prompt 可以让 GLM4 更容易闭合，但当前这些单通道边没有解释闭合增益。

#### 3. DS7B

聚合结果：

```text
plain baseline:        rows=24, stable=0, clear=0, drift=20
short_answer baseline: rows=24, stable=0, clear=8, drift=24
stop_explicit baseline:rows=24, stable=0, clear=8, drift=24
```

DS7B 的 short_answer / stop_explicit 能提高 clear，但 drift 仍然很高，stable 仍为 0。

### 六、结果分析

本阶段得到两个可靠现象：

```text
1. qwen3 的 C249 类边不是严格回答闭合机制；
2. GLM4 的显式停止 prompt 可以产生部分稳定闭合，但不是由当前测试的 C1165/C5532 单通道干预造成。
```

这进一步支持 Phase199 的负结果：

```text
answer selection 与 rollout closure 是可分离机制。
```

更准确地说：

```text
first-token answer boundary
prompt-level instruction following
raw continuation stopping / anti-prose drift
```

至少应当作为三个不同层面的现象处理。

### 七、问题、硬伤与瓶颈

1. 本阶段的 protocol gate 是外部 prompt gate，不是内部 stop/protocol edge。

因此 GLM4 的 stable 提升不能解释为内部闭合通道被定位，只能说明模型在强提示下具备部分短答闭合能力。

2. qwen3 在 stop_explicit 下仍然 stable=0，说明其问题不只是缺少短答提示，也可能是小模型 instruction-following 或停止策略较粗糙。

3. DS7B 的 clear 可被短答提示提高，但 drift 同时保持很高，说明“答对类别”和“停止生成”仍然脱耦。

4. 当前候选边来自 Phase198 的单通道符号分解，并非专门从 stop/EOS/punctuation/prose continuation 维度定位。因此没有真正测试到内部停止齿轮。

### 八、理论进展

Phase200 把 Phase199 的负结果推进了一步：

```text
C249 不是完整闭合机制
外部 stop prompt 可在 GLM4 上修复部分闭合
但当前单通道边不承担这个修复
```

这意味着语言背后的机制图谱不能只画“语义答案方向”，还必须加入：

```text
1. 答案选择坐标
2. 协议约束坐标
3. 停止/标点/EOS 坐标
4. prose continuation 抑制坐标
```

更接近第一性原理的表述是：

```text
一个完整语言行为不是单个语义向量，而是多个机制坐标在生成时序上的组合闭包。
```

目前已经看到的拼图是：

```text
模型可以先选中正确答案词，
但如果没有停止坐标或反续写坐标配合，
它仍会继续生成解释、问题回声或协议漂移。
```

### 九、下一阶段大任务

当前 Phase200 与 Phase199 属于同一阶段性目标：

```text
从 L4 first-token edge 推进到 L6 natural rollout closure mechanism
```

Phase200 已完成该阶段中的“外部协议门控是否足够”的测试。结果表明外部协议门控在 GLM4 上部分有效，但没有定位内部机制。因此下一步不应继续自动做同类 prompt 强化，而应进入新的子阶段：

```text
Phase201: Stop / EOS / Punctuation / Prose-Continuation Component Localization
```

建议下一阶段优先构造四类对照 logit/rollout 指标：

```text
target answer token
period / punctuation token
EOS token
prose continuation token set: because, and, it, usually, which, that, what, is
```

并用通道级因果测试寻找：

```text
提高 period/EOS 或压低 prose continuation 的内部组件
```

只有找到这类组件后，才适合做：

```text
answer-selection edge + stop/prose-control edge
```

的组合 patch。

### 十、通俗总结

这次测试可以理解成：

```text
我们已经找到一些“让模型说出正确第一个词”的零件，
但还没找到“让模型说完就停”的零件。
```

给 GLM4 加一句“只回答一个词，不要解释”，确实能让它部分停住；但把当前找到的单个通道放大或关掉，并不会改变这个停住能力。

所以现在最重要的下一步不是继续追 C249，而是专门去找：

```text
停止、句号、EOS、抑制继续解释
```

这些机制坐标。只有把“选答案”和“控制停止”两类坐标拼起来，才可能画出真正的自然语言生成闭合图谱。

## Phase 201: 停止、结束符、标点与解释续写组件定位图谱 [2026-07-05 17:45]

### 一、任务判断

本阶段分析了附件中对 Phase200 的判断。附件结论基本正确：

```text
Phase199 证明首词元答案边界不等于自然生成闭合；
Phase200 证明外部提示协议可部分修复 GLM4 闭合，但当前答案选择通道不解释该修复；
下一步必须定位内部 stop / punctuation / EOS / anti-prose 组件。
```

这与当前进展处于同一个阶段性目标：

```text
从 L4 answer-selection edge 推进到 L5c/L5d stop/prose control edge，
再为 L6 strict rollout closure 做组合准备。
```

因此本阶段继续自动完成 Phase201，而不是继续围绕 C249 做单通道闭合 patch。

### 二、本阶段完成内容

新增正式测试脚本：

```text
tests/gpt5/phase201_stop_prose_component_atlas.py
tests/gpt5/run_phase201_stop_prose_component_atlas.sh
```

结果保存位置：

```text
tests/result/phase201_stop_prose_component_atlas/stop_prose_component_atlas/
```

跨模型汇总：

```text
phase201_cross_model_summary.json
phase201_cross_model_summary.md
```

执行顺序：

```text
qwen3 -> GLM4 -> DS7B
```

每个模型测试完成后释放 GPU 显存，再加载下一个模型。

### 三、测试原理

Phase201 不再在“回答前”位置只测答案 token，而是在“回答后”位置测停止控制。

例如：

```text
The color of an apple is red
```

在这个位置，下一 token 的竞争不再是 red 是否胜出，而是：

```text
句号 / 换行 / EOS
vs
because / and / usually / which / that / what / is / it / the / a ...
```

本阶段构造三种协议：

```text
plain
short_answer
stop_explicit
```

每个样本先把目标答案追加到 prompt 尾部，然后计算下一 token logits。

### 四、数学指标

停止集合：

```text
V_stop = {EOS, period, newline}
```

解释续写集合：

```text
V_prose = {because, and, usually, which, that, what, is, it, the, a, an, this ...}
```

对象复读集合：

```text
V_echo = object token set
```

停止边界：

```text
m_stop = max z(V_stop) - max z(V_prose)
```

解释续写边界：

```text
m_prose = max z(V_prose) - max z(V_stop)
```

对象复读边界：

```text
m_echo = max z(V_echo) - max z(V_stop)
```

组件搜索先计算通道激活与这些 margin 的相关性：

```text
stop_candidate:       Corr(a_g, m_stop)
anti_prose_candidate: Corr(a_g, -m_prose)
echo_suppress:        Corr(a_g, -m_echo)
```

然后只对候选通道做因果验证：

```text
ablate: a'_g = 0.0 * a_g
boost:  a'_g = 1.5 * a_g
```

评价不再只看答案 token，而看：

```text
Δstop_margin
Δprose_margin
Δecho_margin
Δperiod_vs_prose_margin
Δeos_vs_prose_margin
```

### 五、数据量

token group 大小：

```text
EOS: 1
period: 4
newline: 2
stop: 7
prose: 48
```

三模型数据量：

```text
qwen3:     metric_rows=192, scan_rows=576, causal_rows=252
GLM4:      metric_rows=177, scan_rows=576, causal_rows=242
DS7B:      metric_rows=36,  scan_rows=144, causal_rows=216
```

### 六、客观结果

#### 1. qwen3

qwen3 的 post-answer stop margin 本身并不低，尤其在 stop_explicit 下更高：

```text
color en->en:
plain        stop_margin=4.222
short_answer stop_margin=6.896
stop_explicit stop_margin=8.694
```

但候选通道因果验证没有得到正向 stop/prose 修复：

```text
positive causal candidate rows: 0
```

典型候选反而表现为：

```text
qwen3 color en->zh stop_explicit L26 C2192
ablate: Δstop=-0.054, Δprose=+0.054
boost:  Δstop=-0.071, Δprose=+0.071
```

这说明 qwen3 的问题可能不在“回答后 logit 停止边界”本身，而在真实 rollout 的时序执行、协议服从或生成策略上。

#### 2. GLM4

GLM4 得到本阶段最强正结果。

基础 post-answer 指标中，function en->en 的 plain 条件明显偏向续写：

```text
GLM4 function en->en plain:
stop_margin=-9.240
prose_margin=+9.240
echo_margin=+5.145
```

加入 short_answer / stop_explicit 后转为正 stop margin：

```text
short_answer:  stop_margin=1.389
stop_explicit: stop_margin=1.764
```

这与 Phase200 中 GLM4 可被外部协议部分修复相互印证。

最强因果候选集中在：

```text
GLM4 color zh->en plain
```

代表性结果：

```text
anti_prose_candidate L35 C1018 boost:
rows=14
Δstop_margin=+2.048
Δprose_margin=-2.048
Δecho_margin=-1.184
score=4.095

anti_prose_candidate L23 C616 ablate:
rows=14
Δstop_margin=+2.039
Δprose_margin=-2.039
Δecho_margin=-1.182
score=4.078

anti_prose_candidate L29 C7118 ablate:
rows=14
Δstop_margin=+2.034
Δprose_margin=-2.034
Δecho_margin=-1.186
score=4.069
```

同一批通道也会被 stop_candidate 规则选出，说明这里更准确的命名应是：

```text
stop-vs-prose boundary control candidate
```

而不是纯停止神经元或纯反续写神经元。

#### 3. DS7B

DS7B 只有极弱正结果：

```text
echo_suppress_candidate L16 C6402 boost:
rows=12
Δstop_margin=+0.021
Δprose_margin=-0.021
Δecho_margin=+0.005
score=0.016
```

这个强度太低，只能作为弱参考，不能作为稳定机制证据。

### 七、结果分析

本阶段得到三个客观现象。

第一，Phase201 初步建立了可操作的内部 stop/prose 指标：

```text
回答后位置的 stop/prose logit margin 可以被测量；
通道激活可以与该 margin 做相关筛选；
候选通道可以做 ablate/boost 因果验证。
```

第二，GLM4 存在明显的 stop-vs-prose boundary control 候选，尤其是：

```text
L35 C1018
L23 C616
L29 C7118
```

这些候选能在 post-answer 位置显著提高 stop margin、降低 prose margin，并降低 echo margin。

第三，跨模型稳健性仍弱：

```text
qwen3 没有正候选；
GLM4 有强候选；
DS7B 只有极弱候选。
```

因此当前不能把 GLM4 候选写成通用语言机制，只能写成：

```text
GLM4 小模型中的 stop/prose 边界候选。
```

### 八、硬伤与风险

1. 当前测试仍是 post-answer logit 测试，不是完整 rollout closure。

它证明的是：

```text
组件可改变回答后下一 token 的 stop/prose 竞争。
```

还没有证明：

```text
组件可在自然生成中稳定输出正确答案并停止。
```

2. GLM4 的强候选存在 ablate 和 boost 同向改善的现象。

这说明通道可能不是简单线性单调齿轮，而可能涉及：

```text
激活符号混合
层归一化 / 残差重排
局部非线性
候选筛选偏差
同一通道承担多功能
```

因此不能过早命名为“停止神经元”。

3. qwen3 的 post-answer stop margin 高，但 Phase199/200 rollout stable 仍为 0。

这说明：

```text
logit 停止倾向
和
实际多 token 生成轨迹闭合
仍然不是同一个指标。
```

后续必须把 stop/prose logit 指标接回 raw rollout。

4. DS7B 数据量和候选强度都偏弱，只能当负控或弱参照。

5. 当前扫描层数有限，只扫中后层若干位置，还不是完整全层图谱。

### 九、理论进展

Phase201 把 Phase200 后提出的缺口变成了可测对象：

```text
停止控制不是抽象猜测；
它可以被具体写成 stop/prose/echo logit margin；
也可以在 MLP 通道层面寻找候选边。
```

最新拼图应更新为：

```text
答案选择边：qwen3 C249 是强 L4 候选；
外部协议层：GLM4 stop_explicit 可部分修复行为；
内部 stop/prose 层：GLM4 L35 C1018 / L23 C616 / L29 C7118 是 post-answer 候选；
完整 rollout 闭合：仍未完成。
```

这支持当前统一理论：

```text
语言动作闭合 = 答案选择 + 协议服从 + 停止控制 + 反续写抑制 + 反复读控制。
```

但必须强调：

```text
Phase201 只完成了 L5c/L5d 的候选定位初步；
还没有完成 L6。
```

### 十、下一阶段任务

当前 Phase201 仍属于同一阶段性目标：

```text
把答案选择边与停止 / 反续写边拼成可验证组合机制。
```

Phase201 已经完成“post-answer stop/prose 候选定位初测”。下一阶段应继续同一阶段目标，进入：

```text
Phase202: Stop-Prose Candidate Natural Rollout Repair Audit
```

核心任务：

```text
1. 选 GLM4 L35 C1018、L23 C616、L29 C7118；
2. 在自然生成 prompt 上测试单独 ablate/boost；
3. 在 answer-selected prompt 上测试 post-answer continuation；
4. 检查 raw generated 的 stable / drift / echo；
5. 若 GLM4 候选能降低 drift，再测试与答案选择边的组合。
```

对于 qwen3，不应直接组合 C249，因为 Phase201 没有找到 qwen3 的正 stop/prose 候选。qwen3 应先扩大 stop/prose 搜索层和样本，而不是硬做组合。

### 十一、通俗总结

这次测试相当于问：

```text
模型已经说出答案后，下一步更想打句号，还是更想继续解释？
```

结果是：

```text
GLM4 里找到了一些能明显把“继续解释”推向“更像停止”的候选通道；
qwen3 没找到；
DS7B 只有很弱信号。
```

所以我们向前走了一步：停止/反续写不再只是理论缺口，而是已经有了可测的候选边。但这还不是最终闭合。下一步要看这些候选边能不能真的让 raw generation 少漂移、少解释、少复读。

## Phase 202: 停止/解释续写候选的自然生成修复审计 [2026-07-05 19:59]

### 一、任务判断

本阶段分析了附件中对 Phase201 的判断。附件核心判断正确：

```text
Phase201 是实质进展，但不是闭合。
post-answer stop/prose margin
≠
natural rollout stop
≠
strict language action closure
```

Phase201 已经把停止控制、解释续写抑制、对象复读抑制变成可测的内部指标，并在 GLM4 中找到：

```text
L35 C1018
L23 C616
L29 C7118
```

但这些只是 L5c/L5d 候选，还没有证明能修复 raw rollout。因此本阶段继续同一阶段性目标，进入 Phase202：

```text
把 Phase201 的 stop/prose 候选接回自然生成轨迹。
```

### 二、本阶段完成内容

新增正式脚本：

```text
tests/gpt5/phase202_stop_prose_rollout_repair_audit.py
tests/gpt5/run_phase202_stop_prose_rollout_repair_audit.sh
```

结果保存位置：

```text
tests/result/phase202_stop_prose_rollout_repair_audit/stop_prose_rollout_repair_audit/
```

跨模型汇总：

```text
phase202_cross_model_summary.json
phase202_cross_model_summary.md
```

执行顺序：

```text
qwen3 -> GLM4 -> DS7B
```

每个模型完成后释放 GPU 显存，再加载下一个模型。

### 三、测试原理

Phase202 对 Phase201 候选做 raw generation 审计。

每个候选通道测试：

```text
baseline
ablate: channel factor = 0.0
boost:  channel factor = 1.5
```

每个样本测试两类 rollout mode：

```text
natural:
  从普通 prompt 直接生成。

post_answer:
  先把答案 teacher-forced 到 prompt 尾部，再生成后续 token。
```

例如：

```text
natural:
  The color of a horse is

post_answer:
  The color of a horse is brown
```

三类 prompt protocol：

```text
plain
short_answer
stop_explicit
```

本阶段不再只看单步 stop margin，而看 raw generated 文本是否真的：

```text
答案清楚
不漂移
不解释续写
不复读对象
严格稳定
```

### 四、判定公式

候选修复有效定义为：

```text
RepairEffective(g)
=
stable_delta > 0
and drift_delta < 0
and clear_delta >= 0
```

同时记录：

```text
echo_delta
prose_delta
clear_delta
stable_delta
drift_delta
```

如果候选只改变 Phase201 的 post-answer logit margin，但不能改变 raw rollout，则仍停留在 L5c/L5d，不能升级为 L6。

### 五、测试对象与数据量

qwen3：

```text
3 个弱/负候选
rows = 756
```

GLM4：

```text
3 个 Phase201 强候选：
L35 C1018
L23 C616
L29 C7118
rows = 756
```

DS7B：

```text
3 个弱候选
rows = 648
```

总结果：

```text
repair_effective count = 0
```

### 六、客观结果

#### 1. qwen3

qwen3 没有任何有效修复：

```text
repair_effective = 0
stable_delta 全部为 0
```

少数条件出现极小 drift 下降：

```text
qwen3 L26 C2192 natural plain boost:
stable_delta = 0
drift_delta = -1
prose_delta = -1
clear_delta = 0
repair_effective = false
```

但因为 stable 没有提升，所以不能算修复。

qwen3 在 post_answer stop_explicit 中常见输出：

```text
".\nThe answer is brown. The color"
". The color of a flower is red"
```

这说明它即使先出现停止标点，也会继续进入解释、答案声明或新问题轨迹。

#### 2. GLM4

GLM4 的 Phase201 强候选没有升级为 rollout 修复：

```text
repair_effective = 0
```

聚合结果显示，候选通道 ablate/boost 对 natural rollout 的 stable/drift 基本没有影响：

```text
natural plain:
baseline stable=0, drift=42, clear=6
ablate   stable=0, drift=42, clear=6
boost    stable=0, drift=42, clear=6

natural short_answer:
baseline stable=12, drift=24, clear=18
ablate   stable=12, drift=24, clear=18
boost    stable=12, drift=24, clear=18

natural stop_explicit:
baseline stable=18, drift=6, clear=24
ablate   stable=18, drift=6, clear=24
boost    stable=18, drift=6, clear=24
```

这说明 GLM4 的稳定提升主要仍来自外部 prompt protocol，而不是这三个单通道候选。

post_answer 抽样显示：

```text
".Brown. Brown. Brown. Brown"
".brown. brown. brown. brown"
```

即 GLM4 在回答后常能先打句号，但随后重复答案词。因此：

```text
句号出现
≠
序列真正停止
```

这解释了为什么 Phase201 的 stop margin 正结果没有变成 L6。

#### 3. DS7B

DS7B 也没有有效修复：

```text
repair_effective = 0
stable_delta 全部为 0
```

部分 ablate 还让 drift 增加：

```text
natural plain ablate:
drift_delta = +1
prose_delta = +1
```

DS7B 仍只能作为弱参考或负控。

### 七、关键负结果

Phase202 是一个重要负结果：

```text
Phase201 的 post-answer stop/prose 单步边界候选
没有直接修复 raw rollout。
```

更精确地说：

```text
L5c/L5d single-step stop/prose margin candidate
≠
L6 sequence-level rollout closure candidate
```

这进一步收紧了机制层级：

```text
答案选择边界
≠
回答后停止边界
≠
序列级停止链
≠
严格自然生成闭合
```

### 八、问题、硬伤与瓶颈

1. GLM4 的 Phase201 强候选只影响单步 logit 边界，不能保证后续 token 序列停止。

2. post_answer 模式显示“先句号、后复读”的现象：

```text
stop token 可以出现，
但生成器仍继续输出后续 token。
```

这说明停止控制可能不在单个 MLP 通道，而在：

```text
EOS 触发
generation stopping rule
多步状态链
attention / residual / norm 协同
```

3. 外部 prompt protocol 仍然比单通道干预更强：

```text
GLM4 stop_explicit baseline stable=18
ablate/boost stable 仍然=18
```

4. qwen3 再次证明 post-answer stop 倾向不能推出 strict rollout stable。

5. 当前测试仍是单通道干预，尚未测试多组件组合、全层搜索、注意力头和层归一化/增益。

### 九、理论进展

Phase202 没有给出 L6 正结果，但它补上了一块关键拼图：

```text
单步停止边界不是完整停止机制。
```

因此统一理论需要把停止控制再拆成两层：

```text
token-level stop/prose boundary
sequence-level stop-chain controller
```

语言动作闭合现在至少需要：

```text
1. answer selection
2. protocol following
3. post-answer stop/prose boundary
4. sequence-level stop chain
5. anti-echo control
6. generation termination execution
```

Phase201 找到了第 3 层候选。
Phase202 证明第 3 层单独不足以完成第 4-6 层。

### 十、下一阶段任务

Phase202 仍属于同一阶段性目标：

```text
从答案选择边推进到严格自然生成闭合。
```

但当前子阶段已经完成：

```text
Phase201: 找 post-answer stop/prose 候选；
Phase202: 检查这些候选是否修复 raw rollout；
结果：没有修复。
```

下一步不应继续盲目 boost 这三个 GLM4 通道，而应进入：

```text
Phase203: Sequence-Level Stop Chain and EOS Execution Audit
```

建议 Phase203 直接测生成过程每一步：

```text
t=1,2,3,4...
stop margin
EOS rank
period rank
prose token rank
echo token rank
actual emitted token
whether generation should have stopped but did not
```

核心问题从：

```text
哪个通道提高句号/停止边界？
```

升级为：

```text
为什么模型已经生成句号后还继续生成？
```

### 十一、通俗总结

这次测试说明：

```text
我们找到的 GLM4 通道，确实像是能让“句号/停止”在下一步竞争中更强；
但它们不能让模型真正停下来。
```

模型会出现一种很关键的失败：

```text
brown. Brown. Brown. Brown
```

也就是说：

```text
打了句号
不等于
生成停止。
```

所以真正缺的不是单步“想打句号”的齿轮，而是序列级“打完句号就结束”的控制链。下一步要研究的就是这条停止链。

## Phase 203: 阶段性总总结与全局图谱破解方案 [2026-07-05 20:13]

### 一、总判断

截至 Phase202，当前研究已经完成了一个重要转向：

```text
从“寻找语义答案齿轮”
转向
“构建语言动作的全局机制图谱”
```

最核心的成果不是已经闭合，而是已经证明了多层机制之间不能混为一谈：

```text
答案选择边界
≠
回答后停止边界
≠
序列级停止链
≠
严格自然生成闭合
```

这说明早期“找到语义方向 / 找到答案通道 / patch 一个组件就能闭合”的路线已经被实验证伪。当前最可靠的路线是：

```text
先完成全局图谱拼图，
再做组合闭合验证。
```

### 二、已经完成的主要成果

#### 1. 排除了固定概念向量路线

已有结果反复说明，模型内部不像存着一个固定“apple = 苹果概念向量”。更合理的结构是：

```text
对象身份
+ 领域路线
+ 关系差分
+ 候选场
+ 协议/停止/续写控制
```

也就是相对状态中的条件化组合，而不是孤立概念向量。

#### 2. 建立了答案选择边界证据

Phase198 找到 qwen3 的强 L4 答案选择边：

```text
qwen3 L35 C249
```

它能推动首词元答案边界，例如颜色任务中的目标答案词。

但 Phase199/200 证明：

```text
C249 不能形成严格自然生成闭合。
```

所以它是答案选择候选，不是完整语言行为候选。

#### 3. 建立了 raw generated（原始生成）审计标准

Phase199 纠正了一个重要方法问题：

```text
cleaned generated（清洗后文本）
不能用于判断生成闭合。
```

必须看 raw generated（原始生成），因为模型常常先输出正确答案，然后继续解释、复读、换题或补全新问题。

#### 4. 证明了外部提示协议有效但不是内部机制定位

Phase200 发现：

```text
GLM4 在 stop_explicit（显式停止提示）下 stable（稳定生成）提升。
```

但 ablate/boost 当前候选通道不改变稳定性。因此：

```text
外部 prompt protocol（提示协议）
≠
内部 protocol edge（协议边）
```

#### 5. 建立了 stop/prose/echo 边界指标

Phase201 把停止控制从抽象理论变成可测指标：

```text
stop margin（停止边界）
prose margin（解释续写边界）
echo margin（复读边界）
```

并在 GLM4 中找到 post-answer（回答后）候选：

```text
GLM4 L35 C1018
GLM4 L23 C616
GLM4 L29 C7118
```

这些候选能改变回答后单步 stop/prose 竞争。

#### 6. 证明单步停止边界不等于序列停止链

Phase202 把 Phase201 候选接回 raw rollout（原始生成展开），结果：

```text
repair_effective count = 0
```

GLM4 的强候选没有修复自然生成。典型失败是：

```text
brown. Brown. Brown. Brown
```

这说明：

```text
打句号
不等于
真正停止。
```

### 三、当前积累的核心拼图

当前核心拼图完整列出如下。

```text
1. 相对编码：固定概念向量路线被削弱，条件化相对状态路线增强。
2. 复用差分：概念更像共享路线 + 对象差分 + 关系差分。
3. 条件化路线：领域、模板、语言、关系会启动不同路线。
4. 预测充分状态：状态包含完成预测动作所需变量，但变量仍未完全可测化。
5. 身份变量：对象身份参与生成，不只是类别标签。
6. 角色变量：答案槽、主语、宾语等角色影响输出。
7. 领域变量：颜色领域最强，功能领域较弱，其他领域不足。
8. 关系变量：color（颜色）、function（功能）等关系路线可测。
9. 绑定变量：对象-关系-值绑定可破坏和部分恢复。
10. 候选场：模型会形成候选答案集合。
11. 候选闭合：候选集合内目标可胜出，但不等于全词表胜出。
12. 全词表边界：阻断项和全词表竞争很关键。
13. 闭合间隙：候选闭合与真实生成闭合不同。
14. 协议场：短答、解释、标点、停止会改变输出。
15. 外部协议修复：GLM4 可被显式停止提示部分修复。
16. 内部协议边：尚未稳定定位。
17. 协议漂移：raw generated 中的解释、换题、复读是硬指标。
18. 答案选择边：qwen3 C249 是强 L4 候选。
19. 混合副作用边：qwen3 C16 等不能写成纯语义齿轮。
20. 近零负控：qwen3 C2509 修正了组级误判。
21. GLM4 C1165/C5532：有支持/压制迹象，但不解释闭合。
22. DS7B 通道组：信号弱，主要作负控。
23. L4 不推出 L6：首词元答案边界不推出自然闭合。
24. clear（答案清楚）与 stable（稳定生成）分离。
25. post-answer stop margin（回答后停止边界）可测。
26. prose margin（解释续写边界）可测。
27. echo margin（复读边界）可测。
28. GLM4 stop/prose 候选：C1018/C616/C7118。
29. 单步 stop/prose 候选不推出 raw rollout 修复。
30. 序列级停止链是新缺口。
31. 句号不是终止执行：period（句号）出现后模型仍可继续生成。
32. EOS（结束符）执行机制未定位。
33. 注意力路由仍可能参与停止链，但未系统测试。
34. 残差流可能承载状态协调和停止状态传播。
35. MLP（多层感知机）通道能影响边界，但功能混合严重。
36. LayerNorm/gain（层归一化/增益）可能控制输出强度和续写倾向。
37. 词表读出层承载答案词、停止词、解释词、复读词竞争。
38. 生成轨迹本身成为必须建图的对象。
39. patch（补丁式干预）边际收益递减已经出现。
40. 全局图谱优先级高于单点闭合。
41. 小模型偏差显著，跨模型稳健性弱。
42. 证据等级 L1-L6 已基本清楚，但 L6 仍缺。
```

### 四、统一机制公式的改进

当前不应改理论名词。理论主体仍保持：

```text
预测充分相对状态
→ 全局齿轮图谱
→ 条件化路线门控
→ 全词表竞争闭合
→ 自然生成一致性
```

但公式必须从线性语义读出升级为条件化动态图谱。

#### 1. 条件化状态转移公式

$$
s_{l+1}
=
\Phi_l
\left(
s_l,\,
\rho_l(s_l,x),\,
\gamma_l(s_l,x),\,
\pi(x),\,
\tau_l(s_l,x),\,
\kappa_l(s_l,x),\,
\eta_l(s_l,x),\,
\chi_l(s_l,x)
\right)
+
\varepsilon_l
$$

其中：

```text
s_l = 第 l 层有效状态
ρ_l = 路线门控
γ_l = 增益 / 归一化控制
π(x) = 协议场
τ_l = 停止 / 续写控制器
κ_l = 反解释续写控制器
η_l = 复读抑制控制器
χ_l = 序列级停止链控制器
ε_l = 未解释残差
```

关键新增是：

```text
χ_l = 序列级停止链控制器
```

Phase202 证明，只测单步停止边界不足以解释真实停止。

#### 2. 答案选择公式

$$
m_{\mathrm{ans}}(y \mid x)
=
z_1(y \mid x)
-
\max_{v \in \mathcal{V},\,v \ne y}
z_1(v \mid x)
$$

这个公式回答：

```text
第一个答案词是否胜出？
```

qwen3 C249 主要属于这一层。

#### 3. 回答后停止边界公式

$$
m_{\mathrm{stop}}(t)
=
\max_{v \in V_{\mathrm{stop}}} z_t(v)
-
\max_{v \in V_{\mathrm{prose}}} z_t(v)
$$

其中：

```text
V_stop = EOS（结束符）、period（句号）、newline（换行）
V_prose = because（因为）、and（和）、usually（通常）等解释续写词
```

#### 4. 解释续写边界公式

$$
m_{\mathrm{prose}}(t)
=
\max_{v \in V_{\mathrm{prose}}} z_t(v)
-
\max_{v \in V_{\mathrm{stop}}} z_t(v)
$$

#### 5. 复读边界公式

$$
m_{\mathrm{echo}}(t)
=
\max_{v \in V_{\mathrm{echo}}} z_t(v)
-
\max_{v \in V_{\mathrm{stop}}} z_t(v)
$$

#### 6. 序列级停止链公式

$$
\mathrm{StopChain}(x)
=
\prod_{t=1}^{T}
\mathbf{1}
\left[
m_{\mathrm{stop}}(t)>0
\land
m_{\mathrm{prose}}(t)<0
\land
m_{\mathrm{echo}}(t)<0
\land
\mathrm{ExecStop}(t)=1
\right]
$$

这里最重要的是：

```text
ExecStop(t) = 停止执行是否真正发生
```

Phase202 说明：

```text
period（句号）出现
不等于
ExecStop（停止执行）发生。
```

#### 7. 严格语言动作闭合公式

$$
\mathrm{StrictClosure}(x,y)
=
\mathrm{AnswerSelected}(x,y)
\land
\mathrm{AnswerSpanClear}(x,y)
\land
\mathrm{ProtocolSatisfied}(x)
\land
\mathrm{StopControlled}(x)
\land
\mathrm{ProseSuppressed}(x)
\land
\mathrm{EchoSuppressed}(x)
\land
\mathrm{StopChainExecuted}(x)
\land
\mathrm{RawRolloutStable}(x)
$$

#### 8. 全局图谱公式

$$
\mathcal{G}_{atlas}
=
\left(
\mathcal{V}_{state},
\mathcal{V}_{axis},
\mathcal{V}_{answer},
\mathcal{V}_{boundary},
\mathcal{V}_{protocol},
\mathcal{V}_{stop},
\mathcal{V}_{anti\text{-}prose},
\mathcal{V}_{anti\text{-}echo},
\mathcal{V}_{stop\text{-}chain},
\mathcal{V}_{rollout},
\mathcal{E},
\mathcal{Q}
\right)
$$

每条边记录：

$$
e
=
\left(
v_i \rightarrow v_j,\,
m,\,
d,\,
l,\,
p,\,
g,\,
\boldsymbol{\beta}_g,\,
q,\,
f
\right)
$$

功能斜率向量：

$$
\boldsymbol{\beta}_g
=
\left(
\beta_{\mathrm{ans}},
\beta_{\mathrm{stop}},
\beta_{\mathrm{prose}},
\beta_{\mathrm{echo}},
\beta_{\mathrm{protocol}},
\beta_{\mathrm{drift}},
\beta_{\mathrm{rollout}},
\beta_{\mathrm{exec}}
\right)
$$

其中：

```text
β_exec = 停止执行斜率
```

这是 Phase202 后必须加入的新维度。

### 五、最新完整理论

当前最新理论可以表述为：

```text
语言智能不是固定语义向量空间中的直接读出，
而是预测充分的相对状态网络，
在上下文协议中形成领域路线、候选答案场和答案选择边界；
随后还必须通过协议控制、停止边界、反解释续写、反复读和序列级停止执行，
把正确答案转化为干净、有限、符合协议的语言动作。
```

核心更新是：

```text
自然生成一致性必须拆成：
答案选择一致性
协议一致性
停止边界一致性
解释续写抑制一致性
复读抑制一致性
停止执行一致性
```

### 六、闭合标准与当前距离

当前闭合等级仍合理，但需要加入停止执行层：

```text
L1: 相关证据
L2: 投影节点
L3: 状态转移节点
L4: 答案边界组件因果边
L5a: 自然门控证据
L5b: 协议门控证据
L5c: 停止边界候选
L5d: 反解释续写候选
L5e: 复读抑制候选
L5f: 序列级停止执行候选
L6: 严格自然生成闭合
```

当前完成度估计：

```text
理论主体：约 77%
统一机制公式：约 65%
全局图谱框架：约 48%
答案选择层：约 38%
协议层：约 22%
停止边界层：约 24%
反解释续写层：约 22%
复读抑制层：约 12%
序列级停止链：约 5%
严格自然生成闭合：约 7%
跨模型稳健性：约 20%
```

综合评估：

```text
整体语言编码机制破解进度：约 33% 到 34%
距离完整闭合：约 66% 到 67%
```

考虑当前 qwen3、GLM4、DS7B 都是小模型，外推到更大模型或真实语言编码机制时应打折：

```text
小模型偏差折扣：30% 到 50%
```

因此，当前结果应写成：

```text
小模型机制候选图谱中的阶段性规律，
不能直接写成通用语言智能机制定律。
```

### 七、围绕语言三大核心特性的反思

#### 1. 知识网络

知识网络解决的是：

```text
应该答什么？
```

它包含：

```text
对象身份
类别复用
领域路线
关系路线
候选值场
上下文绑定
```

但 Phase199-202 证明：

```text
知道答案
≠
完成语言动作。
```

知识网络只能把 red（红色）、brown（棕色）、green（绿色）等候选推上来，不能保证模型回答后停止。

#### 2. 推理能力

推理能力不应只理解为链式逻辑推导，而应理解为：

```text
状态更新
约束传播
候选过滤
边界重排
协议检查
停止检查
执行检查
```

也就是说，真正的语言推理不仅要判断答案，还要判断：

```text
当前是否已经回答完？
是否应该解释？
是否应该停止？
是否已经违反协议？
是否正在复读？
```

#### 3. 语法系统

语法系统不只是词序规则，而是语言动作控制系统。

它至少包含：

```text
答案槽控制
短答协议
标点控制
EOS（结束符）控制
解释续写抑制
对象复读抑制
停止执行控制
```

Phase202 的关键洞察是：

```text
句号属于语法表层；
停止执行属于生成控制。
```

二者不同。

### 八、全局图谱总结

#### 1. 特征分布在哪里

当前判断：

```text
残差流：状态协调、跨层传递、候选接口。
MLP（多层感知机）门控：领域路线、候选推动、边界调节。
MLP 写回：词表方向、答案/停止/续写竞争。
注意力头：源词路由、绑定、作用域、答案槽位置。
LayerNorm/gain（层归一化/增益）：边界放大、续写倾向、停止倾向。
词表读出：答案词、句号、EOS、解释词、复读词竞争。
生成器循环：序列级停止链和执行层。
```

#### 2. 复用差分机制

当前更合理的复用差分公式是：

$$
\mathrm{ConceptUse}(o,r,d,\pi)
=
\mathrm{SharedDomainRoute}(d)
+
\mathrm{CategoryReuse}(o,d)
+
\mathrm{ObjectDifference}(o)
+
\mathrm{RelationDifference}(r)
+
\mathrm{ProtocolModifier}(\pi)
+
\mathrm{StopModifier}(\pi)
+
\mathrm{AntiProseModifier}(\pi)
+
\mathrm{EchoSuppressor}(\pi,o)
+
\mathrm{ExecutionController}(\pi,t)
$$

#### 3. 整体形状

当前全局图谱整体形状是：

```text
高维相对状态网络
→ 领域路线漏斗
→ 稀疏答案选择齿轮
→ 全词表竞争场
→ 协议 / 停止 / 反续写 / 反复读控制场
→ 序列级停止执行链
→ 自然生成轨迹
```

#### 4. 如何改进特征分析算法

下一步算法应从“找强通道”改成“画轨迹图谱”：

```text
1. 对每一步生成记录 logits、rank、实际 token。
2. 同时记录 answer/stop/prose/echo/protocol 多维边界。
3. 按成功轨迹、漂移轨迹、复读轨迹分组。
4. 找差异组件，而不是先 patch。
5. patch 只用于最后证据升级。
6. 建立组件的多维功能斜率向量。
```

### 九、接下来的研究方案

下一阶段应放在一个 Phase 中：

```text
Phase204: Global Trajectory Atlas and Stop-Execution Mechanism Mapping
阶段 204：全局生成轨迹图谱与停止执行机制定位
```

阶段目标：

```text
第一优先级：完成全局轨迹图谱拼图。
第二优先级：定位序列级停止执行机制。
第三优先级：为组合闭合准备候选。
```

核心任务：

```text
1. 对 qwen3、GLM4、DS7B 逐 token 记录生成轨迹。
2. 每一步记录 answer、stop、prose、echo、EOS、period 的 logit/rank。
3. 记录实际输出 token 与理论 stop margin 是否一致。
4. 专门统计“句号后继续生成”的失败类型。
5. 比较 stable、drift、echo、prose 四类轨迹的内部状态差异。
6. 扩展组件候选到 attention head（注意力头）、residual stream（残差流）、LayerNorm/gain（层归一化/增益）。
7. 建立 stop-chain atlas（停止链图谱）。
8. 只在候选稳定后做最小组合 patch。
```

阶段成功标准：

```text
不是立刻 L6 闭合，
而是建立可解释的序列级失败图谱：
为什么答对后继续解释？
为什么打句号后继续生成？
为什么 EOS 不触发？
为什么对象会被复读？
```

### 十、最终关键洞察

当前最重要的洞察是：

```text
语言编码机制不是“语义答案编码”一个问题，
而是“语言动作闭合编码”问题。
```

要破解深度神经网络中的脉络和编码机制，必须从单点特征转为全局轨迹：

```text
答案从哪里来？
为什么这个答案胜出？
为什么输出成这种格式？
为什么继续解释？
为什么复读？
为什么句号后还不停？
EOS 为什么没有执行？
```

只有把这些问题放在同一张全局图谱中，才可能真正接近语言背后的数学结构。

## Phase 204: 全局生成轨迹图谱与停止执行机制定位 [2026-07-06 00:03]

### 一、任务判断

本阶段分析了两个附件中对 Phase202/203 的判断。核心判断正确：

```text
Phase202 是关键负结果；
Phase203 的阶段性收束方向正确；
下一步必须从单步 stop/prose margin 转向逐 token 生成轨迹。
```

因此本阶段继续同一阶段性目标：

```text
从答案选择边推进到严格自然生成闭合。
```

但本阶段不做新的 patch，而是优先完成全局轨迹拼图。

### 二、本阶段完成内容

新增正式脚本：

```text
tests/gpt5/phase204_global_trajectory_stop_execution_atlas.py
tests/gpt5/run_phase204_global_trajectory_stop_execution_atlas.sh
```

结果保存位置：

```text
tests/result/phase204_global_trajectory_stop_execution_atlas/global_trajectory_stop_execution_atlas/
```

跨模型汇总：

```text
phase204_cross_model_summary.json
phase204_cross_model_summary.md
```

执行顺序：

```text
qwen3 -> GLM4 -> DS7B
```

### 三、算法原理

Phase204 不再只问：

```text
下一 token 的 stop margin 是否为正？
```

而是逐 token 记录：

```text
actual emitted token（实际输出词元）
EOS rank（结束符排名）
period rank（句号排名）
prose rank（解释续写词排名）
echo rank（复读词排名）
stop margin（停止边界）
prose margin（解释续写边界）
echo margin（复读边界）
period_seen（是否已经出现句号）
continued_after_period（句号后是否继续生成）
ended_with_eos（是否真正 EOS 结束）
```

本阶段使用两种轨迹模式：

```text
natural:
  从原始 prompt 自然生成。

post_answer:
  先把目标答案追加到 prompt 尾部，再继续生成。
```

这分别测试：

```text
模型能否自己选答案并停止；
答案已经给出后模型能否真正停止。
```

### 四、核心公式

逐步停止边界：

$$
m_{\mathrm{stop}}(t)
=
\max_{v \in V_{\mathrm{stop}}} z_t(v)
-
\max_{v \in V_{\mathrm{prose}}} z_t(v)
$$

解释续写边界：

$$
m_{\mathrm{prose}}(t)
=
\max_{v \in V_{\mathrm{prose}}} z_t(v)
-
\max_{v \in V_{\mathrm{stop}}} z_t(v)
$$

复读边界：

$$
m_{\mathrm{echo}}(t)
=
\max_{v \in V_{\mathrm{echo}}} z_t(v)
-
\max_{v \in V_{\mathrm{stop}}} z_t(v)
$$

序列级停止执行判据：

$$
\mathrm{StopExecuted}(x)
=
\exists t
\left[
\mathrm{EOS}_t = 1
\lor
\left(
\mathrm{Period}_t = 1
\land
\neg \mathrm{Continue}_{t+1:T}
\right)
\right]
$$

Phase204 特别关注失败类型：

$$
\mathrm{PeriodButContinue}(x)
=
\exists t
\left[
\mathrm{Period}_t = 1
\land
\mathrm{Continue}_{t+1:T}=1
\right]
$$

### 五、数据量

本阶段生成轨迹数据量：

```text
qwen3:     token_rows=2304, trajectory_rows=288
GLM4:      token_rows=2160, trajectory_rows=270
DS7B:      token_rows=576,  trajectory_rows=72
```

每条轨迹最多生成 8 步。

### 六、客观结果

#### 1. qwen3

qwen3 没有 strict stable：

```text
natural plain:        rows=48, stable=0, drift=44, clear=9
natural short_answer: rows=48, stable=0, drift=48, clear=20
natural stop_explicit:rows=48, stable=0, drift=48, clear=24
```

句号后继续生成非常普遍：

```text
post_answer short_answer:
period_seen=48
continued_after_period=48
ended_with_eos=0

post_answer stop_explicit:
period_seen=48
continued_after_period=48
ended_with_eos=0
```

典型轨迹：

```text
natural plain:
red. What is the most common color

post_answer short_answer:
. The color of a banana is yellow
```

这说明 qwen3 常常能生成答案和句号，但句号后会继续进入新问题或解释轨迹。

#### 2. GLM4

GLM4 同样没有 EOS 结束：

```text
ended_with_eos=0
```

自然 short_answer 条件中：

```text
rows=45
period_seen=44
continued_after_period=44
```

post_answer stop_explicit 条件中：

```text
rows=45
period_seen=45
continued_after_period=45
ended_with_eos=0
```

典型轨迹：

```text
post_answer stop_explicit:
.\nWhat is the color of a banana

post_answer short_answer:
object is ______.\n.\n red__The
```

GLM4 的问题不是不会输出句号，而是句号后仍继续生成。

#### 3. DS7B

DS7B 也没有 EOS 结束：

```text
ended_with_eos=0
```

自然 short_answer 与 stop_explicit：

```text
period_seen=12
continued_after_period=12
stable=0
```

典型轨迹：

```text
natural plain:
drink from it.1234

natural short_answer:
hold something. So, the verb should

post_answer short_answer:
something. So, the verb should be
```

DS7B 的失败形态是句号后进入数字序列、解释句或继续补全。

### 七、关键发现

Phase204 得到一个跨模型强现象：

```text
period（句号）出现很常见；
continued_after_period（句号后继续）也很常见；
ended_with_eos（EOS 结束）几乎为 0。
```

这说明停止执行机制至少分为两层：

```text
1. 生成停止形态 token，例如句号；
2. 真正终止 generation loop（生成循环）。
```

Phase201/202 主要触及第 1 层。
Phase204 证明第 2 层仍未定位。

### 八、问题和硬伤

1. 当前测试仍只记录 logits 和实际 token，还没有定位导致停止执行失败的内部组件。

2. post_answer 模式中，clear 指标不完全适合作为答案正确性指标，因为答案已经被 teacher-forced 到 prompt 中；该模式更适合分析停止执行。

3. qwen3、GLM4、DS7B 都是小模型，停止执行机制可能比大模型更粗糙，因此外推需要 30% 到 50% 折扣。

4. 当前最多生成 8 步，只能观察短轨迹失败，尚未覆盖更长生成。

5. 当前还没有比较 attention head（注意力头）、residual stream（残差流）、LayerNorm/gain（层归一化/增益）在停止链中的作用。

### 九、理论进展

Phase204 把“停止控制”进一步拆成：

```text
stop token boundary（停止词元边界）
period emission（句号输出）
EOS competitiveness（结束符竞争）
termination execution（终止执行）
post-period continuation（句号后继续）
```

因此，语言动作闭合公式必须包含：

```text
StopChainExecuted
```

而不能只包含：

```text
StopMarginPositive
```

更精确地说：

$$
\mathrm{StrictClosure}(x,y)
=
\mathrm{AnswerSelected}(x,y)
\land
\mathrm{ProtocolSatisfied}(x)
\land
\mathrm{PeriodOrEOSAvailable}(x)
\land
\mathrm{StopChainExecuted}(x)
\land
\neg \mathrm{PostPeriodContinuation}(x)
\land
\mathrm{RawRolloutStable}(x)
$$

### 十、下一阶段任务

Phase204 已经完成轨迹图谱初测，下一步仍属于同一阶段性目标。建议进入：

```text
Phase205: Stop-Execution Source Localization Audit
阶段 205：停止执行失败的源头定位审计
```

核心任务：

```text
1. 选择 period 后继续生成的失败轨迹；
2. 捕获 period 前后每一步的 hidden state（隐藏状态）、MLP activation（多层感知机激活）、attention output（注意力输出）；
3. 比较 period 后继续轨迹与少数短停轨迹的差异；
4. 搜索 EOS rank 上不去的源头；
5. 检查是否存在专门压制 EOS 或推动 next-question/prose 的组件；
6. 暂时不做大规模 patch，先完成源头图谱。
```

### 十一、通俗总结

这次测试回答了一个非常具体的问题：

```text
模型打了句号以后，会不会真的停？
```

答案是：

```text
大多数时候不会。
```

qwen3 会生成：

```text
red. What is the most common color
```

GLM4 会生成：

```text
.\nWhat is the color of a banana
```

DS7B 会生成：

```text
something. So, the verb should be
```

因此，当前真正要破解的不是“句号怎么生成”，而是：

```text
为什么句号没有触发停止执行？
```

这就是下一步全局图谱中最关键的脉络。

## Phase 205: 停止执行失败的源头定位审计 [2026-07-06 03:03]

### 一、对附件判断的审查

附件中对 Phase204 的核心判断基本正确，而且需要保留其中最重要的修正：

```text
句号不是模型内部天然停止信号。
句号、EOS、外部 stop sequence、生成循环终止执行必须分层分析。
```

Phase204 已经证明三个小模型普遍存在：

```text
period_seen 高；
continued_after_period 高；
ended_with_eos 近似为 0。
```

但这不能简单写成“模型停止机制坏了”。更谨慎的解释是：当前调用端没有设置句号 stop sequence，模型也没有稳定把短答任务转化为 EOS 竞争胜出。因此 Phase205 的任务不是继续 patch，而是定位句号前后 EOS 竞争和续写竞争如何变化。

### 二、测试脚本和结果路径

新增跨模型脚本：

```text
tests/gpt5/phase205_stop_execution_source_localization_audit.py
tests/gpt5/run_phase205_stop_execution_source_localization_audit.sh
```

结果保存：

```text
tests/result/phase205_stop_execution_source_localization_audit/stop_execution_source_localization_audit/
```

主要结果文件：

```text
phase205_qwen3_summary.json
phase205_glm4_summary.json
phase205_deepseek7b_summary.json
phase205_cross_model_summary.json
phase205_cross_model_summary.md
phase205_*_state_rows.jsonl
phase205_*_transition_rows.jsonl
phase205_*_top_mlp_delta_rows.jsonl
```

三个模型按 qwen3、GLM4、DS7B 顺序加载和释放，避免 GPU 显存叠加。FlashAttention2 不可用时自动回退到 sdpa。每个模型选取 Phase204 中 period 后继续生成的失败轨迹 36 条，记录句号前、句号后、继续一个 token 后三个状态。

### 三、算法原理

Phase205 将停止执行失败拆成三类状态：

```text
before_period：即将输出句号之前；
after_period：已经输出句号之后；
after_continue1：句号后又继续输出一个 token 之后。
```

对每个状态记录：

```text
EOS rank；
period rank；
prose rank；
echo rank；
stop margin；
prose margin；
echo margin；
selected layers residual norm；
selected layers MLP down_proj input RMS；
selected layers attention output norm；
MLP 通道绝对跃迁最大项。
```

核心不是一次性证明闭合，而是记录状态跃迁：

$$
\Delta s_{\text{句号}}
=
s_{\text{after period}}
-
s_{\text{before period}}
$$

$$
\Delta s_{\text{继续}}
=
s_{\text{after continue1}}
-
s_{\text{after period}}
$$

EOS 竞争力变化：

$$
\Delta r_{\text{EOS}}
=
r_{\text{EOS}}(t+1)-r_{\text{EOS}}(t)
$$

其中 rank 越小竞争力越强，所以：

$$
\Delta r_{\text{EOS}} > 0
$$

表示 EOS 竞争力变差。

停止执行需要区分两个目标：

$$
\mathrm{ModelStopExecuted}(x)
=
\exists t[\mathrm{EOS}_t=1]
$$

$$
\mathrm{TaskStopSatisfied}(x)
=
\exists t[
\mathrm{Period}_t=1
\land
\neg \mathrm{Continue}_{t+1:T}
]
$$

更完整的序列状态可写成：

$$
h_{t+1}
=
F_\theta(h_t, y_t, p_t, c_t)
$$

$$
\delta_t
=
D_\theta(h_t, y_{\le t}, x)
$$

$$
\xi_t
=
E_\theta(\delta_t, z_t(\mathrm{EOS}), R_{\text{decode}})
$$

其中：

```text
h_t 是生成状态；
y_t 是当前输出 token；
p_t 是协议变量；
c_t 是上下文变量；
delta_t 是完成状态；
xi_t 是终止执行状态；
R_decode 是解码器规则，包括 EOS、max_new_tokens、外部 stop sequence。
```

### 四、客观结果

#### 1. qwen3

样本：

```text
state rows = 108
transition rows = 72
scanned layers = 12, 18, 23, 28, 33
```

状态均值：

```text
before_period: eos_rank_mean = 57029.17, period_rank_mean = 1.56, prose_rank_mean = 10.08
after_period: eos_rank_mean = 43261.58, period_rank_mean = 437.14, prose_rank_mean = 1.06
after_continue1: eos_rank_mean = 97597.78, period_rank_mean = 518.92, prose_rank_mean = 148.50
```

关键跃迁：

```text
before_period -> after_period:
EOS rank mean delta = -13767.58
stop margin mean delta = -27.01
prose margin mean delta = +27.01

after_period -> after_continue1:
EOS rank mean delta = +54336.19
stop margin mean delta = +24.55
prose margin mean delta = -24.55
```

解释：qwen3 在句号前 period rank 已经非常强，但 after_period 状态里 prose rank 反而变成最强，说明句号并没有维持完成状态；继续一个 token 后 EOS 排名大幅恶化。

最大 MLP 跃迁通道：

```text
L12 C22 mean_abs_delta = 1.04
L18 C5159 mean_abs_delta = 2.03
L23 C1283 mean_abs_delta = 8.32
L28 C205 mean_abs_delta = 9.54
L33 C1986 mean_abs_delta = 13.60
```

#### 2. GLM4

样本：

```text
state rows = 108
transition rows = 72
scanned layers = 14, 20, 25, 31, 37
```

状态均值：

```text
before_period: eos_rank_mean = 1413.31, period_rank_mean = 17.14, prose_rank_mean = 1.44
after_period: eos_rank_mean = 326.72, period_rank_mean = 2.00, prose_rank_mean = 5.86
after_continue1: eos_rank_mean = 9495.08, period_rank_mean = 333.14, prose_rank_mean = 9.28
```

关键跃迁：

```text
before_period -> after_period:
EOS rank mean delta = -1086.58
stop margin mean delta = +7.09
prose margin mean delta = -7.09

after_period -> after_continue1:
EOS rank mean delta = +9168.36
stop margin mean delta = -13.60
prose margin mean delta = +13.60
```

解释：GLM4 在 after_period 时比 qwen3 更接近停止状态，EOS rank 也明显改善，但只要继续一个 token，EOS rank 迅速恶化，说明完成状态不稳定，不能跨步保持。

最大 MLP 跃迁通道：

```text
L14 C2167 mean_abs_delta = 0.30
L20 C1865 mean_abs_delta = 1.16
L25 C9938 mean_abs_delta = 3.74
L31 C11903 mean_abs_delta = 5.36
L37 C8035 mean_abs_delta = 10.82
```

#### 3. DS7B

样本：

```text
state rows = 108
transition rows = 72
scanned layers = 9, 14, 18, 22, 26
```

状态均值：

```text
before_period: eos_rank_mean = 6833.97, period_rank_mean = 1.03, prose_rank_mean = 12.22
after_period: eos_rank_mean = 889.97, period_rank_mean = 534.92, prose_rank_mean = 2.28
after_continue1: eos_rank_mean = 6085.58, period_rank_mean = 606.06, prose_rank_mean = 76.25
```

关键跃迁按协议分组后均显示同一方向：before_period 到 after_period 时 EOS rank 改善，但 after_period 到 after_continue1 时 EOS rank 重新变差。

```text
natural + stop_explicit:
after_period -> after_continue1 EOS rank delta = +1785.00

post_answer + plain:
after_period -> after_continue1 EOS rank delta = +15036.75

post_answer + short_answer:
after_period -> after_continue1 EOS rank delta = +3942.17

post_answer + stop_explicit:
after_period -> after_continue1 EOS rank delta = +873.20
```

最大 MLP 跃迁通道：

```text
L9 C271 mean_abs_delta = 6.55
L14 C11019 mean_abs_delta = 2.64
L18 C17901 mean_abs_delta = 4.50
L22 C15320 mean_abs_delta = 25.57
L26 C264 mean_abs_delta = 44.56
```

### 五、阶段性判断

Phase205 支持附件中的核心修正：

```text
period emission 不等于 EOS competition；
EOS competition 不等于 generation-loop termination；
generation-loop termination 还受解码规则和外部 stop sequence 控制。
```

三模型共同现象：

```text
1. before_period 阶段，period 通常很强；
2. after_period 阶段，EOS rank 往往改善，但很少直接成为稳定终止；
3. after_continue1 阶段，EOS rank 普遍恶化；
4. 句号后的完成状态不能稳定保持；
5. 中后层 MLP 通道和残差状态出现较大跃迁，是下一步建图候选区域。
```

最重要的新拼图是：

```text
停止执行失败不是“没有句号齿轮”，而是“句号后完成状态没有被锁住，EOS 没有稳定胜出，继续生成一步后状态会重新滑入叙述/复读/新任务轨道”。
```

### 六、问题、硬伤和谨慎解释

第一，Phase205 主要使用失败轨迹，短停或 EOS 正样本不足，因此目前更像失败源头图谱，不是成功机制图谱。

第二，attention output 只按层记录整体 norm，没有分 head，不足以定位注意力头是否携带完成状态。

第三，MLP 通道跃迁是绝对变化排序，不等于因果证明。下一步仍需做通道级 ablation、boost、direction patch 或路径 patch 验证。

第四，teacher-forced post-answer 和 self-generated natural 仍未完全解耦。虽然 DS7B 分组显示多个协议方向一致，但 qwen3 和 GLM4 当前样本主要集中在 post_answer + stop_explicit。

第五，当前模型都是小模型，短答协议、EOS 控制、任务完成状态可能与更大模型有 30% 到 50% 偏差。因此结论应限定为：

```text
小模型中的停止执行失败轨迹图谱。
```

不能直接上升为所有语言模型的停止机制定律。

### 七、对智能理论的更新

当前理论不需要换名词，但需要把“动作完成”正式并入统一机制。语言能力不只是选出正确 token，而是要在知识网络、推理链、语法系统之后执行正确动作。

统一机制应暂时写成：

$$
s_{t+1}
=
\Phi_\theta(
s_t,
x,
y_{\le t},
r_t,
p_t,
b_t
)
$$

$$
z_t
=
W_U s_t
$$

$$
a_t
=
\arg\max_{v \in V} z_t(v)
$$

$$
\mathrm{Close}(t)
=
\mathrm{AnswerCorrect}(t)
\land
\mathrm{BoundaryStable}(t)
\land
\mathrm{DoneStateStable}(t)
\land
\mathrm{NoDrift}(t+1:T)
$$

这里的关键更新是：

```text
DoneStateStable 不能由 period 或单步 stop margin 代替。
```

语言的三个核心特性在本阶段的对应关系：

```text
知识网络：决定候选答案和对象-属性绑定；
推理能力：决定关系路径和答案选择；
语法系统：决定句号、续写词、复读词、换行等形式；
编码机制：必须同时解释答案选择和完成状态保持。
```

Phase205 说明语法系统能产出句号，但智能动作系统没有稳定完成闭合。这是“语言形式正确但动作未完成”的典型失败。

### 八、下一阶段任务

当前任务和下一任务仍属于同一阶段性目标：完成全局轨迹图谱，第一优先级仍是拼图，不是闭合。

建议进入：

```text
Phase206: 成功/失败对照的完成状态保持图谱
```

核心目标：

```text
1. 增加成功对照：收集 EOS 结束、外部 stop sequence 结束、句号后无续写的轨迹；
2. 在相同 prompt 上比较 natural、post_answer、forced_period、forced_eos；
3. 全层扫描 EOS rank、prose rank、echo rank 的跃迁源；
4. 对 Phase205 的候选通道做小规模因果验证；
5. 将 attention head 拆开，定位是否存在完成状态读取头；
6. 明确区分内部 ModelStopExecuted 和外部 TaskStopSatisfied；
7. 不做大规模线性 patch，优先完成成功/失败状态差分图谱。
```

通俗总结：模型不是不会打句号，而是打完句号以后没有稳定进入“我已经答完了”的状态。句号像写在纸上的标点，EOS 才像真正按下停止键。现在看到的问题是：三个小模型经常写了句号，却没有按停止键；一旦多走一步，状态会重新滑向续写。下一步要找的就是“答完了”这个状态在网络里到底在哪里、怎么保持、为什么会丢失。

## Phase 206: 成功/失败对照的完成状态保持图谱 [2026-07-06 14:09]

### 一、对附件判断的审查

附件对 Phase205 的判断基本正确。Phase205 的真正价值不是 L6 闭合，而是把停止失败拆成了状态跃迁问题：

```text
before_period -> after_period -> after_continue1
```

本阶段保留附件中最关键的修正：

```text
句号不是停止。
EOS 也不会自动胜出。
停止执行必须区分内部 ModelStopExecuted 和外部 TaskStopSatisfied。
```

因此 Phase206 继续同一阶段性目标：优先完成全局轨迹图谱拼图，而不是直接做大规模 patch。

### 二、测试脚本和结果路径

新增脚本：

```text
tests/gpt5/phase206_done_state_contrast_atlas.py
tests/gpt5/run_phase206_done_state_contrast_atlas.sh
```

结果目录：

```text
tests/result/phase206_done_state_contrast_atlas/done_state_contrast_atlas/
```

主要输出：

```text
phase206_qwen3_summary.json
phase206_glm4_summary.json
phase206_deepseek7b_summary.json
phase206_cross_model_summary.json
phase206_cross_model_summary.md
phase206_*_trajectory_rows.jsonl
phase206_*_token_rows.jsonl
phase206_*_state_rows.jsonl
phase206_*_forced_delta_rows.jsonl
```

三模型按 qwen3、GLM4、DS7B 顺序测试，每个模型测试后释放显存。

### 三、算法原理

Phase206 构造两类停止定义：

$$
\mathrm{ModelStopExecuted}(x)
=
\exists t[\mathrm{EOS}_t=1]
$$

$$
\mathrm{TaskStopSatisfied}(x)
=
\mathrm{ModelStopExecuted}(x)
\lor
\mathrm{ExternalStopExecuted}(x)
\lor
[
\mathrm{PeriodSeen}(x)
\land
\neg \mathrm{ContinuedAfterPeriod}(x)
]
$$

其中：

```text
ModelStopExecuted = 模型自然生成 EOS；
TaskStopSatisfied = 任务层面停止满足，可以由客户端 stop rule 截断。
```

测试对照：

```text
external_stop_rule = none：不设置客户端停止序列，只看模型内部是否 EOS；
external_stop_rule = period：客户端把句号作为停止规则，模拟外部 stop sequence。
```

并重放状态：

```text
before_answer
after_answer
forced_period
forced_eos
after_period
after_continue1
after_continue2
```

注意：forced_eos 是把 EOS 符号放入上下文后的状态代理，不等于模型自然生成 EOS 的成功样本。这一点后面作为硬伤保留。

### 四、客观结果

#### 1. 总体结果

```text
qwen3:
none rows = 504, model_stop_executed = 0, task_stop_satisfied = 12
period rows = 504, model_stop_executed = 0, task_stop_satisfied = 374

GLM4:
none rows = 402, model_stop_executed = 0, task_stop_satisfied = 11
period rows = 402, model_stop_executed = 0, task_stop_satisfied = 233

DS7B:
none rows = 72, model_stop_executed = 0, task_stop_satisfied = 1
period rows = 72, model_stop_executed = 0, task_stop_satisfied = 61
```

最强结果：

```text
三个模型的 ModelStopExecuted 全部为 0。
客户端 period stop rule 大幅提高 TaskStopSatisfied。
```

这说明：当前主要瓶颈不是“客户端无法让任务停止”，而是“模型内部没有自然 EOS 闭合”。

#### 2. qwen3

无客户端停止：

```text
natural plain: rows 84, task_stop_satisfied 7, period_seen 36, continued_after_period 29
natural short_answer: rows 84, task_stop_satisfied 0, period_seen 56, continued_after_period 56
natural stop_explicit: rows 84, task_stop_satisfied 2, period_seen 68, continued_after_period 66
post_answer plain: rows 84, task_stop_satisfied 1, period_seen 51, continued_after_period 50
post_answer short_answer: rows 84, task_stop_satisfied 0, period_seen 78, continued_after_period 78
post_answer stop_explicit: rows 84, task_stop_satisfied 2, period_seen 84, continued_after_period 82
```

客户端句号停止：

```text
natural plain: task_stop_satisfied 37 / 84
natural short_answer: task_stop_satisfied 56 / 84
natural stop_explicit: task_stop_satisfied 68 / 84
post_answer plain: task_stop_satisfied 51 / 84
post_answer short_answer: task_stop_satisfied 78 / 84
post_answer stop_explicit: task_stop_satisfied 84 / 84
```

状态均值：

```text
after_answer: eos_rank_mean = 75597.58, prose_rank_mean = 8.48, stop_margin_mean = 6.64
forced_period: eos_rank_mean = 53235.81, prose_rank_mean = 1.21, stop_margin_mean = -10.53
after_period none fail: eos_rank_mean = 49110.39, prose_rank_mean = 3.53, stop_margin_mean = -11.35
after_continue1 none fail: eos_rank_mean = 115744.13, prose_rank_mean = 102.53, stop_margin_mean = -0.44
```

qwen3 的关键现象：

```text
客户端句号停止几乎可以修复任务输出；
但内部 EOS 从未自然执行；
forced_period 和 after_period 都显示 prose 竞争很强，说明句号后不是稳定完成状态。
```

#### 3. GLM4

无客户端停止：

```text
natural plain: task_stop_satisfied 0 / 67
natural short_answer: task_stop_satisfied 1 / 67
natural stop_explicit: task_stop_satisfied 10 / 67
post_answer plain: task_stop_satisfied 0 / 67
post_answer short_answer: task_stop_satisfied 0 / 67
post_answer stop_explicit: task_stop_satisfied 0 / 67
```

客户端句号停止：

```text
natural plain: task_stop_satisfied 14 / 67
natural short_answer: task_stop_satisfied 62 / 67
natural stop_explicit: task_stop_satisfied 21 / 67
post_answer plain: task_stop_satisfied 9 / 67
post_answer short_answer: task_stop_satisfied 60 / 67
post_answer stop_explicit: task_stop_satisfied 67 / 67
```

状态均值：

```text
after_answer: eos_rank_mean = 4322.29, prose_rank_mean = 7.73, stop_margin_mean = -0.65
forced_period: eos_rank_mean = 933.40, prose_rank_mean = 1.25, stop_margin_mean = -6.46
after_period none fail: eos_rank_mean = 4579.63, prose_rank_mean = 17.62, stop_margin_mean = -0.93
after_continue1 none fail: eos_rank_mean = 9600.62, prose_rank_mean = 49.53, stop_margin_mean = -1.71
```

GLM4 的关键现象：

```text
post_answer + stop_explicit 在无客户端停止时 67 / 67 都继续；
客户端 period stop 后 67 / 67 任务停止满足；
说明 GLM4 的短答任务更像客户端协议问题，而不是内部 EOS 成功。
```

#### 4. DS7B

无客户端停止：

```text
natural plain: task_stop_satisfied 0 / 12
natural short_answer: task_stop_satisfied 0 / 12
natural stop_explicit: task_stop_satisfied 0 / 12
post_answer plain: task_stop_satisfied 1 / 12
post_answer short_answer: task_stop_satisfied 0 / 12
post_answer stop_explicit: task_stop_satisfied 0 / 12
```

客户端句号停止：

```text
natural plain: task_stop_satisfied 6 / 12
natural short_answer: task_stop_satisfied 12 / 12
natural stop_explicit: task_stop_satisfied 12 / 12
post_answer plain: task_stop_satisfied 9 / 12
post_answer short_answer: task_stop_satisfied 12 / 12
post_answer stop_explicit: task_stop_satisfied 10 / 12
```

状态均值：

```text
after_answer: eos_rank_mean = 2097.67, prose_rank_mean = 6.69, stop_margin_mean = -1.41
forced_period: eos_rank_mean = 657.61, prose_rank_mean = 4.17, stop_margin_mean = -5.45
after_period none fail: eos_rank_mean = 824.25, prose_rank_mean = 2.42, stop_margin_mean = -7.33
after_continue1 none fail: eos_rank_mean = 8614.40, prose_rank_mean = 51.38, stop_margin_mean = -4.66
```

DS7B 的样本数较少，但方向和前两者一致：无客户端停止时几乎没有任务闭合，period stop rule 显著提高任务停止满足。

### 五、forced_period 与 forced_eos 的结果

forced_period 的方向整体仍支持 Phase205：

```text
qwen3 after_answer -> forced_period:
EOS rank delta = -4876.25 到 -32599.25，但 stop margin delta 为负，prose margin 上升。

GLM4 after_answer -> forced_period:
EOS rank delta = -1627.06 到 -5596.63，但 stop margin delta 为负，prose margin 上升。

DS7B after_answer -> forced_period:
EOS rank delta 约 -1172.33 到 -1634.50，部分协议下 stop margin 仍下降。
```

解释：强制句号通常能让 EOS 排名改善，但并不稳定提升停止边界；句号后仍可能激活续写轨道。

forced_eos 的结果不一致：

```text
qwen3 forced_eos 使 EOS rank 大幅改善；
GLM4 forced_eos 在 short_answer / stop_explicit 下反而使 EOS rank 变差；
DS7B forced_eos 明显使 EOS rank 变差。
```

这个结果不能直接解释为“EOS 破坏完成状态”。更谨慎的解释是：

```text
把 EOS token 放入上下文后再预测下一 token，可能进入训练分布之外；
不同 tokenizer 对 eos_token 字符串的处理可能不同；
自然生成 EOS 成功样本仍然缺失。
```

因此 forced_eos 目前只能作为上下文代理审计，不是成功停止正样本。

### 六、阶段性结论

Phase206 的最重要进展是把 Phase205 的失败图谱扩展为成功/失败对照图谱，并得到一个非常清楚的拆分：

```text
客户端 TaskStopSatisfied 可以通过 period stop rule 大幅提高；
模型内部 ModelStopExecuted 在本轮三个模型中仍为 0；
句号后无续写可以由客户端规则实现，但这不等于模型内部完成状态闭合。
```

这说明当前的全局图谱至少需要三条分离边：

```text
1. period emission -> client task stop
句号输出 -> 客户端任务停止

2. period emission -> temporary EOS improvement
句号输出 -> EOS 竞争短暂改善

3. period emission -/-> model EOS execution
句号输出不能推出模型内部 EOS 执行
```

当前最强负结果：

```text
在 qwen3、GLM4、DS7B 本轮样本中，自然生成 EOS 结束为 0。
```

当前最强正结果：

```text
外部 period stop rule 能显著提升任务级停止满足。
```

### 七、问题和硬伤

第一，Phase206 仍然没有收集到自然 EOS 成功样本，因此 done state 的真正成功正样本仍不足。

第二，period stop rule 是客户端截断，不是模型内部机制。它可以让产品体验闭合，但不能证明内部智能动作闭合。

第三，forced_eos 是上下文代理，不是自然 EOS 成功。特别是 DS7B 上 forced_eos 后 EOS rank 反而大幅恶化，提示该指标必须谨慎。

第四，attention head 仍未拆分，本轮只记录了多层 residual norm，尚未定位完成状态读取头。

第五，DS7B 可用样本只有 72 条轨迹，明显少于 qwen3 和 GLM4，跨模型统计权重应降低。

第六，当前三个模型均为小模型，EOS 控制和短答协议可能有 30% 到 50% 偏差，不能直接外推到大模型或人脑语言机制。

### 八、对研究方案和客户端方案的改进

研究方案：

```text
1. 继续把 ModelStopExecuted 和 TaskStopSatisfied 分开记录；
2. 不再把句号后无续写当作内部闭合；
3. 优先寻找自然 EOS 正样本，或者构造更接近训练分布的 EOS 诱导提示；
4. 对 after_period success/fail 做方向差分，但必须标注 success 很少；
5. 下一步拆 attention head，检查是否存在完成状态读取头；
6. 对 Phase205/206 的候选层做小规模因果验证，而不是大规模盲目 patch。
```

客户端方案：

```text
1. 对短答任务显式设置 stop sequence，例如 "."、"。" 或 "\n"；
2. 将客户端停止称为 TaskStopSatisfied，不称为 ModelStopExecuted；
3. 保留生成后校验：答案是否清楚、是否漂移、是否复读；
4. 对 GLM4 和 qwen3 的 post_answer + stop_explicit 类任务，period stop rule 能显著改善用户可见输出；
5. 不能依赖模型自然 EOS 停止，至少在当前小模型上不可靠。
```

### 九、统一公式更新

当前闭合公式必须拆成模型闭合和任务闭合：

$$
\mathrm{ModelClose}(x,y)
=
\mathrm{AnswerCorrect}(x,y)
\land
\mathrm{BoundaryStable}(x,y)
\land
\mathrm{DoneStateStable}(x)
\land
\mathrm{ModelStopExecuted}(x)
\land
\mathrm{NoDrift}(x)
$$

$$
\mathrm{TaskClose}(x,y)
=
\mathrm{AnswerCorrect}(x,y)
\land
\mathrm{BoundaryStable}(x,y)
\land
\mathrm{TaskStopSatisfied}(x)
\land
\mathrm{NoDrift}(x)
$$

其中：

$$
\mathrm{TaskStopSatisfied}(x)
=
\mathrm{ModelStopExecuted}(x)
\lor
\mathrm{ExternalStopExecuted}(x)
\lor
[
\mathrm{PeriodSeen}(x)
\land
\neg \mathrm{ContinuedAfterPeriod}(x)
]
$$

Phase206 证明：在当前三个小模型中，TaskClose 可以通过客户端规则接近，但 ModelClose 仍未出现。

### 十、下一阶段任务

当前任务和下一任务仍属于同一阶段性目标：完成全局图谱拼图。建议进入：

```text
Phase207: 自然 EOS 正样本搜索与注意力头完成状态图谱
```

核心任务：

```text
1. 扩大提示类型，专门搜索自然 EOS 结束样本；
2. 比较 EOS 成功、period 客户端成功、period 后继续失败三类轨迹；
3. 将 attention output 从层级 norm 拆成 head 级输出；
4. 对 after_period success/fail 做残差方向差分；
5. 对 qwen3 L33 C1986、GLM4 L37 C8035、DS7B L26 C264 等候选只做小规模验证；
6. 输出 done_state_success_fail_atlas_v2 和 eos_positive_sample_bank。
```

通俗总结：这次结果说明，客户端可以“看见句号就停”，所以产品层面的短答可以被修好；但模型自己并没有真正按下 EOS 停止键。也就是说，外部可以把回答剪干净，内部却还没有稳定的“我答完了”机制。下一步要先找到自然 EOS 正样本，否则完成状态图谱仍然缺真正的正例。

## Phase 207: 自然 EOS 正样本搜索与注意力头完成状态图谱 [2026-07-06 16:07]

### 一、对附件判断的审查

附件对 Phase206 的判断正确。Phase206 已经把三种停止彻底拆开：

```text
模型内部停止 != 任务层面停止 != 客户端截断
```

本阶段继续同一阶段性目标：完成全局轨迹图谱拼图。Phase207 的核心不是闭合，而是专门回答：

```text
扩大提示类型和生成步数后，能否找到自然 EOS 正样本？
如果找不到，是否可以至少建立失败轨迹的 attention head 候选图谱？
```

### 二、测试脚本和结果路径

新增脚本：

```text
tests/gpt5/phase207_eos_positive_head_atlas.py
tests/gpt5/run_phase207_eos_positive_head_atlas.sh
```

结果目录：

```text
tests/result/phase207_eos_positive_head_atlas/eos_positive_head_atlas/
```

主要输出：

```text
phase207_qwen3_summary.json
phase207_glm4_summary.json
phase207_deepseek7b_summary.json
phase207_cross_model_summary.json
phase207_cross_model_summary.md
phase207_*_trajectory_rows.jsonl
phase207_*_token_rows.jsonl
phase207_*_state_rows.jsonl
phase207_*_head_rows.jsonl
phase207_*_eos_positive_sample_bank.jsonl
```

本轮按 qwen3、GLM4、DS7B 顺序测试，每个模型测试后释放显存。

### 三、算法原理

Phase207 扩大了自然 EOS 搜索空间：

```text
prompt_protocols =
plain,
short_answer,
stop_explicit,
eos_instruction,
final_answer,
chat_eos

max_steps = 32
decoding = greedy_manual_argmax
do_sample = false
external_stop_sequence = none
```

记录解码配置：

```text
eos_token_id
eos_token
pad_token_id
pad_token
generation_config_eos_token_id
generation_config_pad_token_id
max_steps
do_sample
temperature
```

轨迹被分为：

```text
model_eos_success：自然生成 EOS；
period_client_success_proxy：出现句号且未继续；
period_continue_fail：出现句号后继续；
no_period_fail：没有句号也没有 EOS。
```

注意：本轮不设置客户端 stop sequence，所以 `period_client_success_proxy` 只表示如果客户端截断可以任务停止，不表示模型内部停止。

注意力头图谱采用 `self_attn.o_proj` 的输入作为 head 级输出代理。对每个选中层，把 o_proj 输入按 attention head 拆分，记录每个 head 的范数。该指标是：

```text
attention head output proxy
注意力头输出代理
```

不是因果证明。

### 四、客观结果

#### 1. 总体结果

```text
qwen3:
prompt_count = 432
trajectory_rows = 432
eos_positive_count = 0

GLM4:
prompt_count = 342
trajectory_rows = 342
eos_positive_count = 0

DS7B:
prompt_count = 72
trajectory_rows = 72
eos_positive_count = 0
```

跨模型汇总：

```text
total_eos_positive_count = 0
```

这是一条强负结果：在扩大提示族、最大生成 32 步、无客户端 stop sequence 的条件下，三个小模型仍没有自然生成 EOS。

#### 2. qwen3

轨迹统计：

```text
chat_eos: period_continue_fail 72 / 72
eos_instruction: no_period_fail 2, period_continue_fail 70
final_answer: no_period_fail 26, period_continue_fail 46
plain: no_period_fail 9, period_continue_fail 63
short_answer: no_period_fail 4, period_continue_fail 68
stop_explicit: no_period_fail 6, period_continue_fail 66
```

典型现象：

```text
chat_eos 触发 <think> 轨迹和长续写；
eos_instruction 常出现 “Done.” 复读；
short_answer 和 stop_explicit 仍会转入连续问答/解释轨道。
```

关键状态均值：

```text
chat_eos after_prompt:
eos_rank_mean = 5.5
prose_rank_mean = 66909.17
stop_margin_mean = 11.44

chat_eos after_period:
eos_rank_mean = 84756.83
prose_rank_mean = 7.83
stop_margin_mean = -19.43

chat_eos after_continue1:
eos_rank_mean = 111466.67
prose_rank_mean = 72.67
stop_margin_mean = -1.31
```

解释：qwen3 在 chat prompt 起点上 EOS rank 可以很强，但模型仍没有生成 EOS；一旦进入句号后状态，EOS 竞争迅速丢失。这说明高 EOS rank 起点不等于 ModelStopExecuted。

头部候选：

```text
L26 H22 norm_mean = 22.55
L33 H21 norm_mean = 18.61
L33 H6 norm_mean = 16.93
L26 H20 norm_mean = 16.87
L33 H7 norm_mean = 15.95
```

这些是失败轨迹中的高范数注意力头候选，不是 done-state 成功头。

#### 3. GLM4

轨迹统计：

```text
chat_eos: no_period_fail 29, period_continue_fail 28
eos_instruction: no_period_fail 1, period_continue_fail 56
final_answer: no_period_fail 56, period_continue_fail 1
plain: no_period_fail 48, period_continue_fail 9
short_answer: period_continue_fail 57 / 57
stop_explicit: period_continue_fail 57 / 57
```

典型现象：

```text
eos_instruction 多次生成 “Do not add any additional information...” 但仍继续；
final_answer 倾向 Answer: red: red 复读；
chat_eos 会出现 <|user|> 复现和继续对话轨道。
```

头部候选：

```text
L29 H19 norm_mean = 39.80
L37 H10 norm_mean = 33.21
L29 H31 norm_mean = 32.30
L37 H7 norm_mean = 26.41
L37 H29 norm_mean = 26.38
```

解释：GLM4 的强活动集中在中后层若干 attention head，但由于没有 EOS 成功样本，只能作为失败/续写轨道候选。

#### 4. DS7B

轨迹统计：

```text
chat_eos: period_continue_fail 12 / 12
eos_instruction: no_period_fail 8, period_continue_fail 4
final_answer: no_period_fail 8, period_continue_fail 4
plain: no_period_fail 2, period_continue_fail 10
short_answer: period_continue_fail 12 / 12
stop_explicit: period_continue_fail 12 / 12
```

典型现象：

```text
chat_eos 进入 “Okay, so I need to figure out...” 推理续写；
short_answer 继续输出 “So, the answer is...” 和步骤解释；
stop_explicit 仍可能进入问题复述或解释。
```

头部候选：

```text
L26 H2 norm_mean = 26.84
L26 H0 norm_mean = 21.72
L26 H20 norm_mean = 18.79
L26 H18 norm_mean = 17.08
L26 H8 norm_mean = 16.88
```

DS7B 样本量较小，候选头只能低权重参考。

### 五、阶段性判断

Phase207 得到的是负结果，但价值很高：

```text
扩大提示族 + 32 步生成 + 无外部 stop sequence
仍然找不到自然 EOS 正样本。
```

这进一步强化 Phase206 的分层结论：

```text
客户端可以任务闭合；
模型内部 EOS 闭合仍未出现；
句号、显式停止提示、chat 模板都不能可靠诱导 ModelStopExecuted。
```

本阶段还得到一个新的细节：

```text
chat_eos 并不等于 EOS 诱导；
在 qwen3 和 DS7B 上它常触发 reasoning/prose 轨道；
在 GLM4 上它可能触发对话标记复现。
```

### 六、问题和硬伤

第一，Phase207 没有找到自然 EOS 正样本，所以仍不能构建真正的 done-state 成功方向。

第二，attention head 结果只是 o_proj 输入范数代理，不能说明某个 head 因果控制完成状态。

第三，因为没有成功类，head 图谱主要是失败轨道候选图谱，不是成功/失败差分图谱。

第四，本轮只用 greedy decoding。采样可能偶然产生 EOS，但那会引入随机性；如果后续使用 sampling，必须固定 seed、多轮重复，并与 greedy 分开记录。

第五，当前模型都是小模型，EOS 控制可能显著弱于更大模型，因此不能把 “EOS 正样本为 0” 外推为语言模型普遍规律。

### 七、对理论和方案的更新

当前统一公式不需要改名，但需要增加一条更严格的观测：

$$
\mathrm{EOSRankHigh}(t)
\not\Rightarrow
\mathrm{ModelStopExecuted}(t)
$$

qwen3 的 chat_eos after_prompt 中 EOS rank 很高，但仍没有自然生成 EOS，说明 EOS 竞争力只是必要候选，不是执行停止本身。

模型闭合仍保持：

$$
\mathrm{ModelClose}(x,y)
=
\mathrm{AnswerCorrect}(x,y)
\land
\mathrm{BoundaryStable}(x,y)
\land
\mathrm{DoneStateStable}(x)
\land
\mathrm{ModelStopExecuted}(x)
\land
\mathrm{NoDrift}(x)
$$

任务闭合仍保持：

$$
\mathrm{TaskClose}(x,y)
=
\mathrm{AnswerCorrect}(x,y)
\land
\mathrm{BoundaryStable}(x,y)
\land
\mathrm{TaskStopSatisfied}(x)
\land
\mathrm{NoDrift}(x)
$$

Phase207 说明：当前只能比较任务闭合和失败轨迹，模型闭合正例仍缺失。

### 八、客户端方案更新

工程上更明确：

```text
不要期待小模型自然 EOS 停止；
短答任务必须设置客户端 stop sequence；
句号、中文句号、换行都应纳入可配置停止规则；
客户端还要做答案清楚度、复读、漂移校验；
chat template 不一定提升短答闭合，可能触发思维链或对话续写。
```

### 九、下一阶段任务

当前任务仍属于同一阶段性目标：完成全局图谱拼图。Phase207 未找到 EOS 正样本，因此下一步不应立即做 done-state 成功差分，而应先改进正样本搜索策略。

建议进入：

```text
Phase208: 解码配置审计与 EOS 诱导边界搜索
```

核心任务：

```text
1. 使用 model.generate 与手写 greedy 对照，确认 EOS 停止配置没有误差；
2. 记录 generation_config、eos_token_id、pad_token_id、forced_eos_token_id、stopping_criteria；
3. 加入 sampling 但固定 seed，分温度搜索自然 EOS；
4. 测试极短 completion、空白 completion、特殊结束提示、chat template 结束边界；
5. 明确区分 greedy 无 EOS、sampling 偶发 EOS、客户端 stop 三类机制；
6. 若仍无 EOS，转向寻找“高 EOS rank 但未执行 EOS”的解码/词表读出原因。
```

通俗总结：这次专门去找“模型自己按下 EOS 停止键”，但三个小模型都没有按。更有意思的是，qwen3 在某些 chat 起点上 EOS 排名很高，却仍然不选 EOS，说明“停的信号在附近”不等于“真正停”。现在最稳的结论是：产品可以靠客户端停止规则闭合，但模型内部自然停止还没有找到正样本。下一步要先审计解码配置，再尝试更系统的 EOS 诱导搜索。

## Phase 208: 解码配置审计与 EOS 诱导边界搜索 [2026-07-06 17:35]

### 一、对附件判断的审查

附件对 Phase207 的判断基本正确。Phase207 的强负结果是：

```text
扩大 prompt 类型；
max_steps 提高到 32；
greedy decoding；
无 external stop sequence；
仍找不到自然 EOS 正样本。
```

但附件也指出一个关键风险：必须先排查解码配置和 EOS/pad token 处理，否则可能把实现问题误判成模型机制。Phase208 因此继续同一阶段性目标：完成全局图谱拼图，优先做解码配置审计，而不是理论收束或大规模 patch。

### 二、测试脚本和结果路径

新增脚本：

```text
tests/gpt5/phase208_decode_config_eos_boundary_audit.py
tests/gpt5/run_phase208_decode_config_eos_boundary_audit.sh
```

结果目录：

```text
tests/result/phase208_decode_config_eos_boundary_audit/decode_config_eos_boundary_audit_fixed/
```

说明：第一次运行发现 `pad_token_id == eos_token_id` 时，generate 输出解析会把 EOS 当作 padding 误删。已修正脚本：当 pad 与 EOS 相同，generate 解析强制 batch=1，并保留 EOS token。以下结论以 fixed round 为准。

主要输出：

```text
phase208_qwen3_summary.json
phase208_glm4_summary.json
phase208_deepseek7b_summary.json
phase208_cross_model_summary.json
phase208_cross_model_summary.md
phase208_*_decode_rows.jsonl
phase208_*_manual_token_rows.jsonl
phase208_*_manual_generate_compare_rows.jsonl
phase208_*_eos_positive_rows.jsonl
```

### 三、算法原理

Phase208 对比四类解码机制：

```text
manual_greedy：手写逐步 argmax；
generate_greedy：model.generate 的 greedy；
generate_beam：beam search；
generate_sample：固定 seed 的 sampling。
```

采样参数：

```text
temperature = 0.7, 1.0
seed = 11, 23, 37
top_p = 0.95
```

记录配置：

```text
tokenizer_eos_token_id
tokenizer_pad_token_id
generation_config_eos_token_id
generation_config_pad_token_id
generation_config_forced_eos_token_id
model_config_eos_token_id
model_config_pad_token_id
```

核心判据：

$$
\mathrm{ModelStopExecuted}(x)
=
\exists t[y_t=\mathrm{EOS}]
$$

在 greedy 下：

$$
\mathrm{ModelStopExecuted}(t)
=
\mathbf{1}
[
\arg\max_{v \in \mathcal{V}} z_t(v)=\mathrm{EOS}
]
$$

Phase208 特别区分：

```text
greedy EOS；
generate API EOS；
beam EOS；
sampling EOS；
client stop。
```

这些不能混合解释。

### 四、配置审计结果

#### 1. qwen3

```text
tokenizer_eos_token_id = 151645
tokenizer_pad_token_id = 151643
generation_config_eos_token_id = [151645, 151643]
generation_config_pad_token_id = 151643
```

manual 与 generate greedy：

```text
rows = 96
first_token_matches = 96
manual_eos = 0
generate_eos = 0
```

结论：qwen3 的手写 greedy 和 generate greedy 首 token 完全一致，且都没有 EOS。Phase207 对 qwen3 的 0 EOS 判断基本稳固。

#### 2. GLM4

```text
tokenizer_eos_token_id = 151329
tokenizer_pad_token_id = 151329
generation_config_eos_token_id = [151329, 151336, 151338]
generation_config_pad_token_id = 151329
```

manual 与 generate greedy：

```text
rows = 96
first_token_matches = 92
manual_eos = 0
generate_eos = 0
```

结论：GLM4 的 pad 和 EOS 相同，而且 generation_config 有多个 EOS token。修正解析后发现 greedy 仍无 EOS，但 sampling 有少量 EOS。

#### 3. DS7B

```text
tokenizer_eos_token_id = 151643
tokenizer_pad_token_id = 151643
generation_config_eos_token_id = 151643
generation_config_pad_token_id = null
```

manual 与 generate greedy：

```text
rows = 36
first_token_matches = 36
manual_eos = 1
generate_eos = 1
```

结论：DS7B 的 pad 和 EOS 相同。修正解析后，manual greedy 与 generate greedy 在 `End the response now.` 特殊提示上都能生成 EOS。

### 五、客观结果

跨模型 fixed 汇总：

```text
qwen3 eos_positive_count = 0
GLM4 eos_positive_count = 5
DS7B eos_positive_count = 37
total_eos_positive_count = 42
```

#### 1. qwen3

```text
manual_greedy: EOS = 0
generate_greedy: EOS = 0
generate_beam: EOS = 0
generate_sample: EOS = 0
```

qwen3 在本轮所有解码模式中仍无 EOS。chat_eos 继续稳定触发 `<think>`：

```text
chat_eos first token = <think>
```

结论：qwen3 的内部 EOS 闭合仍未找到正样本。它是本轮最稳的负结果。

#### 2. GLM4

GLM4 fixed 后出现 5 条 EOS，全部来自 sampling：

```text
generate_sample + eos_instruction + temperature 0.7: 2 / 57
generate_sample + final_answer + temperature 1.0: 1 / 57
generate_sample + short_answer + temperature 1.0: 2 / 60
```

典型正样本：

```text
No additional comments or questions are allowed.
Greenish brown. END.<|endoftext|>
```

也有低质量样本：

```text
?\n<|endoftext|>
\n<|endoftext|>
1.5g<|endoftext|>
```

结论：GLM4 不是完全不能 EOS，但 EOS 主要是 sampling 偶发，而且内容质量不稳定。不能把它当成稳定 ModelClose。

#### 3. DS7B

DS7B fixed 后出现 37 条 EOS：

```text
manual_greedy + end_now: 1 / 1
generate_greedy + end_now: 1 / 1
generate_beam + end_now: 1 / 1
generate_beam + eos_instruction: 4 / 6
generate_beam + final_answer: 4 / 6
generate_sample 多协议下均有若干 EOS。
```

关键正样本：

```text
Prompt:
End the response now.

Output:
</think>

Hello! How can I assist you today?<｜end▁of▁sentence｜>
```

另一些 beam 正样本：

```text
drink<｜end▁of▁sentence｜>
to hold liquids.<｜end▁of▁sentence｜>
What is the function of a horse?<｜end▁of▁sentence｜>
```

结论：DS7B 可以自然生成 EOS，但很多 EOS 正样本仍不是严格短答闭合，有的先进入 `</think>`、问句复述或解释轨道，然后才 EOS。因此这是 ModelStopExecuted 正样本，不等于 L6 ModelClose。

### 六、阶段性判断

Phase208 修正了 Phase207 的一个过强结论：

```text
“三个模型都没有自然 EOS” 不再成立。
```

更准确的新结论是：

```text
qwen3：多解码模式下仍未找到 EOS；
GLM4：greedy 无 EOS，sampling 偶发 EOS；
DS7B：在特殊提示、beam、sampling 下可生成 EOS；
但三者都没有证明稳定严格 ModelClose。
```

因此当前图谱应新增三类 EOS：

```text
1. GreedyEOS：贪心 EOS；
2. SamplingEOS：采样偶发 EOS；
3. BeamEOS：束搜索 EOS。
```

它们和客户端停止仍然不同：

```text
ClientStop != GreedyEOS != SamplingEOS != BeamEOS
```

### 七、问题和硬伤

第一，GLM4 的 EOS 正样本主要来自 sampling，随机性较强，且质量不稳定。

第二，DS7B 的 EOS 正样本虽然多，但不少先进入推理/复述/问句轨道，然后才 EOS，不等于完成状态稳定。

第三，qwen3 仍没有 EOS 正样本，因此 qwen3 的 done-state 成功图谱仍无法建立。

第四，beam search 改变了解码目标，不等于模型自然 greedy 行为，不能和 greedy EOS 混用。

第五，本轮发现 pad/eos 解析是重大实现风险。以后所有 EOS 研究必须显式记录：

```text
pad_token_id == eos_token_id ?
generation_config.eos_token_id 是否为列表？
输出解析是否保留 EOS？
batch padding 是否引入假 EOS 或删掉真 EOS？
```

第六，当前仍然是小模型图谱，EOS 控制和 chat template 行为可能与大模型有明显偏差。

### 八、理论更新

Phase207 的公式：

$$
\mathrm{EOSRankHigh}(t)
\not\Rightarrow
\mathrm{ModelStopExecuted}(t)
$$

仍然成立。

Phase208 进一步说明：

$$
\mathrm{ModelStopExecuted}
=
\mathrm{GreedyEOS}
\lor
\mathrm{BeamEOS}
\lor
\mathrm{SamplingEOS}
$$

但这只是停止执行，不是完整模型闭合。

严格模型闭合仍是：

$$
\mathrm{ModelClose}(x,y)
=
\mathrm{AnswerCorrect}(x,y)
\land
\mathrm{BoundaryStable}(x,y)
\land
\mathrm{DoneStateStable}(x)
\land
\mathrm{ModelStopExecuted}(x)
\land
\mathrm{NoDrift}(x)
$$

Phase208 找到了一些 ModelStopExecuted 正样本，但还没有证明：

```text
AnswerCorrect
BoundaryStable
DoneStateStable
NoDrift
```

同时成立。

### 九、客户端方案更新

工程方案不变，反而更明确：

```text
1. qwen3 不能依赖自然 EOS；
2. GLM4 不能依赖 sampling 偶发 EOS；
3. DS7B 虽能 EOS，但输出质量不稳定，仍需客户端 stop 与校验；
4. 短答产品层仍应使用 stop sequence；
5. EOS 只能作为额外停止条件，不能替代输出校验。
```

### 十、下一阶段任务

当前任务仍属于同一阶段性目标：完成全局图谱拼图。Phase208 已找到可用但不完美的 EOS 正样本，下一步可以从“找不到正样本”转为“正样本质量分层和 done-state 对照”。

建议进入：

```text
Phase209: EOS 正样本质量分层与 ModelClose 近邻图谱
```

核心任务：

```text
1. 将 EOS 正样本分为 clean_eos、late_eos、drift_eos、bad_answer_eos；
2. 比较 qwen3 无 EOS、GLM4 sampling EOS、DS7B greedy/beam/sampling EOS；
3. 对 EOS 前状态、句号后状态、续写失败状态做 residual 和 attention head 差分；
4. 只对 clean 或 near-clean EOS 建立 done-state 候选方向；
5. 明确区分 ModelStopExecuted 与 ModelClose；
6. 输出 eos_quality_atlas_v1 和 modelclose_neighbor_atlas_v1。
```

通俗总结：这轮最大的收获不是“模型会停了”，而是发现之前可能把 EOS 当 pad 误删。修正以后，qwen3 仍然不会自然停，GLM4 偶尔会停但不稳，DS7B 确实能停但常常不是干净短答。也就是说，我们终于找到了 EOS 正样本，但它们还不等于真正的“答完了”。下一步要给这些 EOS 正样本分质量等级，再找接近 ModelClose 的内部状态。

## Phase 209: 模式运行对比图谱与单点组件路线重估 [2026-07-06 18:49]

### 一、任务来源和总判断

本轮输入提出一个关键修正：

```text
语言的本质可能不是单个语义向量、单个概念神经元、单个停止符或单个通道，而是模式。
语法规则是模式，逻辑推理是模式，知识网络也是模式网络。
深度神经网络的核心工作方式，是对动态特征网络进行处理、组合、竞争和路由。
```

这个判断总体正确，而且比继续追 C249、句号、EOS rank、单个 attention head 更接近当前证据。

Phase198 到 Phase208 的价值不是被推翻，而是被重新解释：

```text
单点组件不是语言机制的基本单位。
单点组件是模式运行轨迹的底层证据。
```

前面负结果共同说明：

```text
答案选择通道不能闭合完整回答；
句号不是停止；
EOS rank 高不等于 EOS 被执行；
客户端 stop 可以修复产品输出，但不是模型内部闭合；
自然 EOS 正样本存在，但不等于 clean ModelClose。
```

因此，本轮将研究对象从“停止符 / 单点组件”上移到：

```text
对象-关系-值问答模式，在不同输出约束下如何运行、竞争和漂移。
```

### 二、模式的严格工作定义

为了避免“模式”变成过大的词，本轮把模式定义为可测对象：

$$
\boxed{
\mathrm{Pattern}
=
\left(
\mathrm{Trigger},
\mathrm{StateVariables},
\mathrm{FeatureTrajectory},
\mathrm{PriorityProxy},
\mathrm{OutputConstraint},
\mathrm{FailureModes}
\right)
}
$$

其中：

```text
Trigger = 触发条件，本轮为 relation + prompt pattern；
StateVariables = 对象、关系、目标答案、语言方向；
FeatureTrajectory = 贪心生成轨迹，以及每一步 target/stop/prose/echo/EOS 等排名代理；
PriorityProxy = 多个模式竞争时，最终胜出的输出模式；
OutputConstraint = 预期输出模式，例如短答、解释、复读、列表；
FailureModes = 实际输出模式与预期不一致时的漂移类型。
```

统一机制公式暂时改写为：

$$
\boxed{
h_{t+1}
=
\sum_{k \in \mathcal{P}}
\alpha_k(x,t) T_k(h_t)
+
\varepsilon_t
}
$$

其中：

```text
\alpha_k(x,t) = 第 k 个模式在当前上下文和生成状态下的优先级；
T_k(h_t) = 第 k 个模式对应的状态转移；
\varepsilon_t = 当前未解释残差。
```

这比线性单点公式更能解释当前现象：同一个答案词可能已经被选中，但输出轨迹仍会漂移到解释、列表、复读或续写。

### 三、测试脚本和数据

新增脚本：

```text
tests/gpt5/phase209_pattern_running_contrast_atlas.py
tests/gpt5/run_phase209_pattern_running_contrast_atlas.sh
```

结果目录：

```text
tests/result/phase209_pattern_running_contrast_atlas/pattern_running_contrast_atlas/
```

核心结果文件：

```text
phase209_qwen3_summary.json
phase209_glm4_summary.json
phase209_deepseek7b_summary.json
phase209_cross_model_summary.json
phase209_cross_model_summary.md
phase209_*_token_rows.jsonl
phase209_*_trajectory_rows.jsonl
```

测试模式：

```text
answer_short          = 短答模式；
answer_stop           = 短答 + 停止模式；
answer_explain        = 解释模式；
answer_repeat         = 复读模式；
answer_list           = 列表模式；
answer_echo_control   = 对象回声 + 回答模式；
answer_target_seeded  = 目标答案种子 + 最终答案模式。
```

模型按顺序测试：

```text
qwen3 -> GLM4 -> DS7B
```

所有模型均使用本地 CUDA 加载，测试完一个释放显存后再加载下一个。

### 四、客观结果

跨模型总结果：

```text
总轨迹数: 833
目标模式匹配: 140
模式漂移: 693
答案出现: 415
自然 EOS 结束: 8
```

按模型：

```text
qwen3:
  轨迹 420
  模式匹配 79
  模式漂移 341
  答案出现 248
  EOS 0

GLM4:
  轨迹 343
  模式匹配 42
  模式漂移 301
  答案出现 145
  EOS 0

DS7B:
  轨迹 70
  模式匹配 19
  模式漂移 51
  答案出现 22
  EOS 8
```

按模式的核心现象：

```text
answer_short:
  三个模型均 0 命中；
  经常漂移到 explain_answer、list_answer、repeat_answer 或 other_or_wrong。

answer_stop:
  三个模型均 0 命中；
  说明“停止指令”没有形成稳定内部停止模式。

answer_explain:
  qwen3 27/60 命中；
  GLM4 24/49 命中；
  DS7B 10/10 命中；
  解释模式比短答/停止模式更容易占优。

answer_list:
  qwen3 24/60 命中；
  GLM4 4/49 命中；
  DS7B 9/10 命中；
  列表模式在 qwen3 和 DS7B 上较强，但 GLM4 输出格式混乱。

answer_repeat:
  qwen3 28/60 命中；
  GLM4 10/49 命中；
  DS7B 0/10 命中；
  复读模式容易被列表模式吸收。

answer_target_seeded:
  qwen3 0/60 命中，但答案出现 57/60；
  GLM4 4/49 命中，答案出现 42/49；
  DS7B 0/10 命中，答案出现 10/10，且 EOS 8/10；
  说明目标答案种子能强力提升答案出现，但不等于短答闭合。
```

### 五、关键进展

第一，模式路线是可测试的，不只是理论说法。

本轮已经把“模式”压成了可测对象：

```text
同一批对象-关系-值样本
叠加不同输出约束
观察最终胜出的输出模式和漂移模式
```

第二，结果支持“模式优先级竞争”。

短答、停止、解释、列表、复读不是同一条轨迹上的简单参数变化，而更像不同模式之间的竞争。

第三，前面停止机制负结果得到重新解释。

短答和 stop 指令在本轮几乎完全失败，这说明：

```text
停止控制不是答案选择模式的自然尾部；
停止控制也不是句号或 EOS 排名的简单结果；
停止控制需要一个更高层的完成状态模式或外部客户端规则。
```

第四，答案出现和模式闭合被清楚拆开。

例如 qwen3 的 answer_target_seeded：

```text
答案出现 57/60
短答模式命中 0/60
```

这说明模型可以知道答案，但不能稳定执行目标输出模式。

第五，解释模式和列表模式可能是小模型中的强默认语言模式。

尤其 DS7B：

```text
answer_explain 10/10
answer_list 9/10
answer_short 0/10
answer_stop 0/10
```

这说明 DS7B 更像被训练成“解释/展开/思考”优先，而不是短答优先。

### 六、问题、硬伤和谨慎点

第一，本轮分类器仍然是启发式分类器。

它依赖 answer_mentions、comma_count、because_like、word_count 等规则，不能完全等价于真实语义模式。

第二，短答判定较严格。

如果模型先给出正确答案再继续展开，会被判为漂移。这是合理的，因为本轮研究的是模式闭合，不只是答案出现；但它会降低“短答模式命中率”。

第三，DS7B 样本量偏小。

DS7B 只有 70 条轨迹，不能和 qwen3 的 420 条、GLM4 的 343 条直接等权比较。DS7B 的方向有价值，但不能过度泛化。

第四，当前样本仍主要来自对象-关系-值问答。

这只是语言模式图谱的第一块，不覆盖否定、条件、比较、递推、语法嵌套等模式。

第五，当前没有直接记录完整 hidden-state 轨迹。

本轮 FeatureTrajectory 主要是输出轨迹和 token-rank 代理，还没有进入 layer/head/channel 级的动态轨迹分析。

第六，小模型偏差仍然很大。

当前三个模型的内部编码机制可能与更强模型存在 30% 到 50% 偏差，尤其 DS7B 的 reasoning trace 会强烈污染短答和停止模式。

### 七、理论进展

本轮理论不改名，只更新主体：

```text
智能理论当前核心主体：
语言能力来自深度神经网络中可复用的动态模式网络。
知识、语法、推理不是三个完全分离模块，而是不同类型的模式族。
模型输出不是单个通道决定，而是多个模式在上下文中竞争后的轨迹结果。
```

最新机制公式：

$$
\boxed{
P_k(x,t)
=
\left[
\alpha_k(x,t),
\phi_k(x,t),
\Delta h_k(x,t),
b_k(x,t),
o_k(x,t)
\right]
}
$$

其中：

```text
P_k = 第 k 个模式；
\alpha_k = 模式激活强度；
\phi_k = 模式特征集合；
\Delta h_k = 模式造成的状态变化；
b_k = 边界/停止/切换信号；
o_k = 输出倾向。
```

全局运行公式：

$$
\boxed{
h_{t+1}
=
\sum_k
\alpha_k(x,t) T_k(h_t)
+
\varepsilon_t
}
$$

输出竞争公式：

$$
\boxed{
o_t
=
\operatorname{Readout}
\left(
\sum_k
\alpha_k(x,t) T_k(h_t)
\right)
}
$$

闭合公式仍然保持严格：

$$
\boxed{
\mathrm{ModelClose}(x,y)
=
\mathrm{AnswerCorrect}(x,y)
\land
\mathrm{PatternMatched}(x,y)
\land
\mathrm{BoundaryStable}(x,y)
\land
\mathrm{DoneStateStable}(x)
\land
\mathrm{ModelStopExecuted}(x)
\land
\mathrm{NoDrift}(x)
}
$$

Phase209 说明：

```text
AnswerCorrect 或 AnswerPresent 远远不够。
真正闭合必须要求 PatternMatched 和 NoDrift。
```

### 八、客户端改进方案

客户端方案进一步明确：

```text
1. 短答产品不能依赖模型自然短答模式；
2. stop sequence 仍然必要；
3. EOS 只能作为辅助停止条件，不能作为唯一闭合标准；
4. 对短答任务，应增加输出模式校验；
5. 如果模型进入解释、列表、复读、格式循环，应客户端截断或重问；
6. target-seeded prompt 可以提升答案出现率，但容易造成复读和二次展开，不能直接当闭合方案。
```

推荐工程规则：

$$
\boxed{
\mathrm{ClientAccept}(y)
=
\mathrm{AnswerPresent}(y)
\land
\mathrm{FormatValid}(y)
\land
\mathrm{LengthValid}(y)
\land
\neg \mathrm{DriftPattern}(y)
}
$$

其中 DriftPattern 至少包括：

```text
解释漂移；
列表漂移；
复读漂移；
继续问答漂移；
格式循环；
reasoning trace 泄漏。
```

### 九、接下来的阶段任务

当前任务和下一任务仍属于同一阶段性目标：

```text
完成语言编码机制的全局图谱拼图。
```

但下一步不应继续 patch 单点参数，而应进入：

```text
Phase210: 最小稳定模式的内部轨迹定位
```

目标：

```text
1. 从 Phase209 中挑出最稳定的正模式：
   qwen3 answer_explain / answer_repeat / answer_list；
   GLM4 answer_explain；
   DS7B answer_explain / answer_list。

2. 为每个稳定模式建立最小对照：
   explain vs short；
   list vs short；
   repeat vs list；
   target_seeded vs short。

3. 不再首先找 EOS，而是找模式切换点：
   什么时候从 answer selection 进入 explain；
   什么时候从 repeat 进入 list；
   什么时候从 short 失败进入 continuation。

4. 记录隐藏层轨迹：
   residual norm；
   target rank；
   prose rank；
   echo rank；
   stop/EOS rank；
   关键层差分。

5. 目标输出：
   minimal_pattern_transition_atlas_v1；
   stable_pattern_positive_set_v1；
   pattern_drift_negative_set_v1。
```

通俗总结：这轮证明“语言是模式网络”不是空话。模型经常已经知道答案，但并不会按短答或停止模式完成任务；它更容易被解释、列表、复读这些强模式带走。要破解编码机制，下一步不应继续死追某个停止符，而要选一个最稳定模式，观察它在网络内部从触发、竞争、占优到输出的完整轨迹。

## Phase 210: 最小稳定模式的内部轨迹定位 [2026-07-06 19:23]

### 一、任务判断

本轮输入继续确认 Phase209 的路线：

```text
语言是多层动态模式网络；
知识、语法、逻辑、标点、EOS、客户端停止都可以放入模式网络；
但模式必须分层，不能把所有机制压成同一个平面。
```

这个判断正确。需要补充的是：

```text
模式网络不是放弃组件研究，而是改变研究顺序：
先定义模式；
再记录模式轨迹；
再找模式切换点；
最后把 attention head、MLP channel、EOS、句号等组件挂回模式图谱。
```

Phase209 已经证明输出模式漂移非常普遍，但它主要是行为层和 token-rank 代理。本轮 Phase210 的任务是继续同一阶段目标：

```text
从输出模式图谱进入内部 hidden-state 轨迹图谱。
```

### 二、算法原理

本轮继续使用对象-关系-值问答模式族，测试五类模式：

```text
answer_short
answer_explain
answer_list
answer_repeat
answer_target_seeded
```

每条轨迹逐步生成，并在每一步记录：

```text
1. 当前输出 token；
2. target rank；
3. stop/prose/echo/EOS 等边界代理；
4. 选定层的 last-token hidden state；
5. 每层 hidden state 的 residual_norm、mean、std；
6. 每个 pattern 相对 answer_short 的均值向量差分。
```

本轮模式轨迹代理为：

$$
\boxed{
\Gamma(P,x,t,l)
=
\left[
h_{l,t},
m_{\mathrm{target},t},
m_{\mathrm{stop},t},
m_{\mathrm{prose},t},
m_{\mathrm{echo},t},
m_{\mathrm{EOS},t},
o_t
\right]
}
$$

其中：

```text
P = 模式；
x = 样本；
t = 生成步；
l = 层；
h_l,t = 第 l 层 last-token hidden state；
m = 各类 rank/margin 代理；
o_t = 实际输出 token。
```

模式差分计算为：

$$
\boxed{
\Delta h_{P,\mathrm{short}}(t,l)
=
\mathbb{E}[h_{P,t,l}]
-
\mathbb{E}[h_{\mathrm{short},t,l}]
}
$$

并记录：

$$
\boxed{
D_{P,\mathrm{short}}(t,l)
=
\left\|
\Delta h_{P,\mathrm{short}}(t,l)
\right\|_2
}
$$

以及：

$$
\boxed{
C_{P,\mathrm{short}}(t,l)
=
\cos
\left(
\mathbb{E}[h_{P,t,l}],
\mathbb{E}[h_{\mathrm{short},t,l}]
\right)
}
$$

这些不是因果证明，只是内部状态分离度代理。

### 三、脚本和结果位置

新增脚本：

```text
tests/gpt5/phase210_minimal_pattern_transition_atlas.py
tests/gpt5/run_phase210_minimal_pattern_transition_atlas.sh
```

结果目录：

```text
tests/result/phase210_minimal_pattern_transition_atlas/minimal_pattern_transition_atlas/
```

核心结果：

```text
phase210_cross_model_summary.json
phase210_cross_model_summary.md
phase210_qwen3_summary.json
phase210_glm4_summary.json
phase210_deepseek7b_summary.json
phase210_*_state_rows.jsonl
phase210_*_contrast_rows.jsonl
phase210_*_trajectory_rows.jsonl
phase210_*_token_rows.jsonl
```

模型顺序：

```text
qwen3 -> GLM4 -> DS7B
```

每个模型测试完成后释放 GPU 显存，再加载下一个模型。

### 四、客观结果

跨模型总结果：

```text
总轨迹数: 440
内部状态记录: 36848
模式差分记录: 1008
目标模式匹配: 119
模式漂移: 321
答案出现: 267
自然 EOS: 6
```

模式匹配率：

$$
\boxed{
R_{\mathrm{pattern}}
=
\frac{119}{440}
\approx
27.0\%
}
$$

模式漂移率：

$$
\boxed{
R_{\mathrm{drift}}
=
\frac{321}{440}
\approx
73.0\%
}
$$

答案出现率：

$$
\boxed{
R_{\mathrm{answer}}
=
\frac{267}{440}
\approx
60.7\%
}
$$

自然 EOS 率：

$$
\boxed{
R_{\mathrm{EOS}}
=
\frac{6}{440}
\approx
1.36\%
}
$$

### 五、按模型结果

qwen3：

```text
轨迹数: 200
状态记录: 16800
模式匹配: 62
模式漂移: 138

answer_short: 0/40
answer_explain: 20/40
answer_list: 14/40
answer_repeat: 28/40
answer_target_seeded: 0/40
```

GLM4：

```text
轨迹数: 200
状态记录: 16800
模式匹配: 45
模式漂移: 155

answer_short: 0/40
answer_explain: 7/40
answer_list: 1/40
answer_repeat: 20/40
answer_target_seeded: 17/40
```

DS7B：

```text
轨迹数: 40
状态记录: 3248
模式匹配: 12
模式漂移: 28

answer_short: 0/8
answer_explain: 6/8
answer_list: 6/8
answer_repeat: 0/8
answer_target_seeded: 0/8
```

关键事实：

```text
三模型 answer_short 仍然全部 0 命中。
answer_repeat 在 qwen3 和 GLM4 中较强。
answer_explain / answer_list 在 DS7B 中较强。
GLM4 的 answer_target_seeded 出现 17/40 短答命中，是本轮新增的局部正结果。
DS7B 的 answer_target_seeded 仍然答案出现高，但多为 repeat/echo，并且 6/8 EOS。
```

### 六、内部轨迹结果

本轮最重要的内部结果是：

```text
相对 answer_short，其他模式在后层出现明显 hidden-state 均值向量差分。
```

最大差分集中层：

```text
qwen3:
  layer 34 / 32 / 26 最明显；
  answer_repeat、answer_list、answer_explain、answer_target_seeded 都和 short 分离。

GLM4:
  layer 38 / 35 / 29 最明显；
  answer_list、answer_repeat、answer_target_seeded、answer_explain 都和 short 分离。

DS7B:
  layer 26 / 24 / 20 最明显；
  answer_target_seeded、answer_explain、answer_list、answer_repeat 都和 short 分离。
```

跨模型 top contrast：

```text
DS7B answer_target_seeded vs short, layer 26:
  mean_l2_diff = 708.77
  cosine = 0.789

DS7B answer_explain vs short, layer 26:
  mean_l2_diff = 680.80
  cosine = 0.805

DS7B answer_list vs short, layer 26:
  mean_l2_diff = 675.62
  cosine = 0.806

qwen3 answer_repeat vs short, layer 34:
  mean_l2_diff = 260.20
  cosine = 0.892

qwen3 answer_list vs short, layer 34:
  mean_l2_diff = 248.09
  cosine = 0.896

GLM4 answer_list vs short, layer 38:
  mean_l2_diff = 143.04
  cosine = 0.880

GLM4 answer_repeat vs short, layer 38:
  mean_l2_diff = 134.87
  cosine = 0.900
```

解释：

```text
模式差异不只是输出文本分类差异；
在 selected hidden layers 的 last-token state 中也能看到稳定分离代理。
```

但必须谨慎：

```text
这是均值向量差分，不是因果定位；
不能直接说 layer 34 或 layer 38 “控制”模式；
只能说这些层是当前观测下的模式分离高响应区。
```

### 七、和 Phase209 的关系

Phase209 证明：

```text
模式路线可测；
短答弱；
解释、列表、复读是强竞争模式；
答案出现不等于模式闭合。
```

Phase210 增加：

```text
模式差异可在 hidden-state 轨迹中观察到；
差异主要出现在中后层到后层；
answer_short 不是没有被提示，而是相对其他模式缺少稳定占优轨迹；
target_seeded 能显著改变答案出现和 hidden-state 轨迹，但不保证短答闭合。
```

因此，当前拼图从：

```text
输出模式图谱
```

推进到：

```text
输出模式 + 内部状态差分图谱。
```

### 八、问题和硬伤

第一，Phase210 仍然是相关性图谱。

虽然记录了 hidden-state 差分，但没有做因果 patch、ablation 或 activation steering。

第二，差分是均值差分。

均值向量可能掩盖多子模式、多簇结构和样本内部差异。后续需要聚类或按输出成功/失败分组。

第三，当前使用 last-token hidden state。

这适合观察生成决策点，但不能覆盖 prompt 内部各 token 的模式触发过程，例如“because”“comma”“target seed”等 token 的局部作用。

第四，DS7B 样本量仍偏小。

DS7B 只有 40 条轨迹，虽然内部差分很大，但不能和 qwen3 / GLM4 等权比较。

第五，模式分类器仍然是启发式。

分类规则会影响 pattern_match / drift。尤其 short、other_or_wrong、echo_then_answer 的边界需要继续校正。

第六，小模型偏差仍然存在。

当前结果只能说明小模型模式机制图谱的一部分，不能直接外推到大模型或真实通用语言机制。

### 九、理论进展

当前理论主体不改名，只更新公式解释。

语言编码机制当前可写为：

$$
\boxed{
\mathrm{LanguageMechanism}
=
\mathrm{PatternFamily}
+
\mathrm{PatternCompetition}
+
\mathrm{StateTrajectory}
+
\mathrm{OutputClosure}
}
$$

模式运行公式保持：

$$
\boxed{
h_{t+1}
=
\sum_k
\alpha_k(x,t)T_k(h_t)
+
\varepsilon_t
}
$$

Phase210 增加内部轨迹观测项：

$$
\boxed{
\Gamma(P,x)
=
\left\{
h_{l,t},
m_{r,t},
o_t
\right\}_{l,t,r}
}
$$

其中：

```text
h_l,t = 层 l、生成步 t 的 hidden state；
m_r,t = target、stop、prose、echo、EOS 等边界代理；
o_t = 输出 token。
```

模式差分：

$$
\boxed{
\Delta \Gamma(P_i,P_j)
=
\mathbb{E}[\Gamma(P_i)]
-
\mathbb{E}[\Gamma(P_j)]
}
$$

当前最可靠的结论是：

```text
PatternMatched 必须成为闭合标准；
NoDrift 必须成为闭合标准；
hidden-state 分离是模式存在的候选证据；
但只有因果干预才能把候选证据升级为机制证据。
```

### 十、客户端方案更新

客户端层仍然不能依赖自然短答或自然 EOS。

本轮进一步说明：

```text
1. short prompt 本身不足以触发短答闭合；
2. target_seeded 可以改善答案出现，GLM4 上甚至改善短答命中，但容易引发 repeat/echo；
3. explain/list/repeat 是强竞争模式，需要被客户端识别并截断；
4. 产品层应使用输出模式校验，而不是只判断答案是否出现；
5. 对短答任务，应组合 stop sequence、长度限制、格式校验、漂移模式过滤。
```

客户端接受公式保持：

$$
\boxed{
\mathrm{ClientAccept}(y)
=
\mathrm{AnswerPresent}(y)
\land
\mathrm{FormatValid}(y)
\land
\mathrm{LengthValid}(y)
\land
\neg \mathrm{DriftPattern}(y)
}
$$

### 十一、下一阶段任务

当前任务和下一任务仍属于同一阶段性目标：

```text
完成语言编码机制的全局图谱拼图。
```

Phase210 已完成第一版内部轨迹代理。下一步应进入：

```text
Phase211: 模式成功/失败分组与切换点定位
```

核心任务：

```text
1. 把同一模式分成 success 和 drift 两组；
2. 比较 answer_repeat 成功 vs answer_repeat 漂移到 list；
3. 比较 answer_explain 成功 vs explain 漂移到 repeat/list/other；
4. 比较 target_seeded 短答成功 vs target_seeded repeat/echo；
5. 在 layer-step 网格上找差分峰值；
6. 只对差分峰值做小规模 causal patch / ablation；
7. 输出 pattern_transition_switchpoint_atlas_v1。
```

通俗总结：Phase210 证明模式差异已经可以在模型内部状态轨迹中看见，尤其在中后层/后层最明显。但这还不是“找到了机制”，只是找到了可能的机制地形。下一步要把成功轨迹和失败轨迹分开，找“从短答转向解释/列表/复读”的具体切换点，再用小规模干预验证。

## Phase 211: 模式成功/失败分组与切换点候选图谱 [2026-07-06 19:38]

### 一、任务判断

本轮输入对 Phase210 的判断基本正确：

```text
Phase210 证明模式差异不只是输出文本差异；
在 hidden-state 轨迹中也能看到分离代理；
但 Phase210 仍然是相关性图谱，不是因果闭合。
```

当前最应该继续的任务不是重新总结理论，也不是马上大规模 patch，而是：

```text
把同一模式的成功轨迹和失败轨迹分开；
在 layer-step 网格中找模式漂移切换点候选；
为下一轮小规模因果验证缩小搜索空间。
```

由于 Phase210 已经产生：

```text
440 条轨迹；
36848 条 state rows；
1008 条 pattern contrast rows。
```

本轮不需要重新加载 qwen3、GLM4、DS7B。直接离线分析 Phase210 结果更合理，避免重复消耗 GPU，并且符合“不要轻易进入 patch”的要求。

### 二、脚本和结果

新增脚本：

```text
tests/gpt5/phase211_pattern_switchpoint_atlas.py
tests/gpt5/run_phase211_pattern_switchpoint_atlas.sh
```

结果目录：

```text
tests/result/phase211_pattern_switchpoint_atlas/pattern_switchpoint_atlas/
```

核心结果：

```text
phase211_cross_model_summary.json
phase211_cross_model_summary.md
phase211_cross_model_outcome_rows.jsonl
phase211_cross_model_state_outcome_summary_rows.jsonl
phase211_cross_model_switchpoint_rows.jsonl
phase211_qwen3_summary.json
phase211_glm4_summary.json
phase211_deepseek7b_summary.json
```

### 三、算法原理

Phase211 读取 Phase210 的：

```text
trajectory_rows；
state_rows；
token_rows 中已经写入 state rows 的 rank/margin 代理。
```

先给每条轨迹打 outcome label：

```text
success = pattern_match；
drift:{failure_mode} = pattern_drift。
```

然后在每个：

```text
model / pattern_id / outcome_group / step / layer_idx
```

上统计：

```text
residual_norm_mean；
residual_mean_mean；
residual_std_mean；
target_rank_mean；
stop_margin_mean；
prose_margin_mean；
echo_margin_mean；
eos_rank_mean；
period_rank_mean。
```

再做 success-vs-drift 对比：

$$
\boxed{
\Delta M
=
M_{\mathrm{drift}}
-
M_{\mathrm{success}}
}
$$

切换点候选分数定义为简单代理：

$$
\boxed{
S_{\mathrm{switch}}
=
\frac{|\Delta \mathrm{ResidualNorm}|}{\max(1,|\mathrm{ResidualNorm}_{success}|)}
+
|\Delta \mathrm{ProseMargin}|
+
|\Delta \mathrm{EchoMargin}|
+
|\Delta \mathrm{StopMargin}|
+
\frac{|\Delta \mathrm{TargetRank}|}{1000}
+
\frac{|\Delta \mathrm{EOSRank}|}{1000}
}
$$

解释：

```text
分数越高，表示成功轨迹和某类漂移轨迹在该层、该步的状态/边界代理差异越大。
```

注意：

```text
这是 switchpoint candidate score（切换点候选分数）；
不是因果强度；
不是机制闭合。
```

### 四、客观结果

本轮输出：

```text
outcome rows: 50
state summary rows: 4186
switchpoint rows: 2268
```

最高分切换点候选：

```text
qwen3 answer_list -> drift:other_or_wrong
  best step = 11
  best layer = 32
  score = 181.73
  residual_norm_delta = -103.93
  prose_margin_delta = -38.29
  echo_margin_delta = -16.45

GLM4 answer_list -> drift:short_answer
  best step = 8
  best layer = 29
  score = 157.20
  residual_norm_delta = 17.62
  prose_margin_delta = 8.62
  echo_margin_delta = 2.29

GLM4 answer_list -> drift:echo_then_answer
  best step = 8
  best layer = 35
  score = 153.64
  residual_norm_delta = 50.34
  prose_margin_delta = 5.53
  echo_margin_delta = 11.63

GLM4 answer_list -> drift:next_task_or_format
  best step = 10
  best layer = 35
  score = 149.42
  residual_norm_delta = -31.03
  prose_margin_delta = -5.09
  echo_margin_delta = -3.78

GLM4 answer_list -> drift:repeat_answer
  best step = 8
  best layer = 29
  score = 135.61
  residual_norm_delta = 11.45
  prose_margin_delta = 8.01
  echo_margin_delta = 2.08

DS7B answer_explain -> drift:other_or_wrong
  best step = 7
  best layer = 26
  score = 133.46
  residual_norm_delta = 132.97
  prose_margin_delta = -3.72
  echo_margin_delta = 11.00

qwen3 answer_list -> drift:short_answer
  best step = 9
  best layer = 32
  score = 126.87
  residual_norm_delta = -74.96
  prose_margin_delta = -38.27
  echo_margin_delta = -30.79

DS7B answer_list -> drift:other_or_wrong
  best step = 7
  best layer = 24
  score = 111.87
  residual_norm_delta = 99.68
  prose_margin_delta = 0.65
  echo_margin_delta = 10.21
```

### 五、主要发现

第一，Phase211 和 Phase210 的层级结果基本一致。

Phase210 发现模式分离高响应区：

```text
qwen3: layer 34 / 32 / 26；
GLM4: layer 38 / 35 / 29；
DS7B: layer 26 / 24 / 20。
```

Phase211 的高分切换点也主要集中在：

```text
qwen3 layer 32；
GLM4 layer 29 / 35；
DS7B layer 24 / 26。
```

这说明 Phase210 的后层/中后层分离不是孤立现象，成功/漂移分组后仍然复现。

第二，answer_list 是当前最适合继续深挖的模式。

高分候选里，大量来自：

```text
qwen3 answer_list；
GLM4 answer_list；
DS7B answer_list。
```

它有两个优点：

```text
有成功样本；
有多种漂移方向；
能形成 success-vs-drift 对照。
```

第三，GLM4 的 answer_list 很适合研究模式漂移。

GLM4 的 answer_list 有多个 drift group：

```text
short_answer；
echo_then_answer；
next_task_or_format；
repeat_answer；
other_or_wrong。
```

这些漂移在 layer 29 / 35、step 8 / 10 附近出现较强候选差异。

第四，qwen3 的 answer_list 漂移有强边界特征。

例如：

```text
qwen3 answer_list -> other_or_wrong:
prose_margin_delta = -38.29
echo_margin_delta = -16.45

qwen3 answer_list -> short_answer:
prose_margin_delta = -38.27
echo_margin_delta = -30.79
```

这说明 qwen3 的 list 成功和 drift 之间，可能不是单纯 residual norm 差异，而是 prose/echo/target 等边界代理共同变化。

第五，DS7B 的高分候选样本量仍小，但层位非常一致。

DS7B 的 answer_explain / answer_list 漂移候选出现在：

```text
layer 26 / 24；
step 7。
```

这和 Phase210 的 DS7B 高差分层一致。但由于 DS7B 样本量小，仍只能作为候选证据。

### 六、问题和硬伤

第一，本轮没有重新跑模型，也没有做因果干预。

这不是缺点，而是阶段选择：Phase211 是切换点候选图谱，不是闭合验证。

第二，本轮只使用 Phase210 已保存的 scalar metrics。

也就是说，Phase211 没有原始 hidden-state 向量，只能比较：

```text
residual_norm；
residual_mean；
residual_std；
rank/margin。
```

不能计算 success-vs-drift 的真实向量方向。

第三，switchpoint_score 是启发式。

它用于排序候选点，不是严格数学定理。不同归一化方式可能改变排名。

第四，成功/失败标签仍依赖 Phase210 的启发式模式分类器。

如果分类器误判，switchpoint 候选也会受影响。

第五，部分 drift group 样本数很少。

例如 DS7B 的样本总量仍偏小，GLM4 某些 drift group 也可能样本不足。不能把单个高分点当成结论。

第六，仍未覆盖 prompt token-level 触发过程。

Phase211 仍然基于生成步 last-token state，没有观察 prompt 中模式触发词的局部状态。

### 七、理论进展

当前理论不改名，只增加“切换点”概念。

模式运行公式保持：

$$
\boxed{
h_{t+1}
=
\sum_k
\alpha_k(x,t)T_k(h_t)
+
\varepsilon_t
}
$$

Phase211 增加：

$$
\boxed{
\mathrm{SwitchPoint}(P_i \to P_j)
=
\arg\max_{l,t}
D
\left(
\Gamma_{\mathrm{success}}(P_i,l,t),
\Gamma_{\mathrm{drift}}(P_j,l,t)
\right)
}
$$

其中：

```text
SwitchPoint = 模式从目标轨迹分叉到漂移轨迹的候选层-步位置；
D = 状态/边界代理差异；
\Gamma_success = 成功轨迹；
\Gamma_drift = 漂移轨迹。
```

当前严格表述应为：

```text
Phase211 找到的是 SwitchPointCandidate（切换点候选），不是 SwitchPointCause（切换点因果）。
```

闭合标准不变：

$$
\boxed{
\mathrm{ModelClose}
=
\mathrm{AnswerCorrect}
\land
\mathrm{PatternMatched}
\land
\mathrm{BoundaryStable}
\land
\mathrm{DoneStateStable}
\land
\mathrm{ModelStopExecuted}
\land
\mathrm{NoDrift}
}
$$

Phase211 的作用是服务于：

```text
PatternMatched；
NoDrift；
BoundaryStable。
```

### 八、当前阶段目标判断

当前任务和下一任务仍属于同一阶段性目标：

```text
完成语言编码机制的全局图谱拼图。
```

Phase209 到 Phase211 的连续进展是：

```text
Phase209:
  输出模式图谱，证明模式竞争和漂移。

Phase210:
  内部状态差分图谱，证明模式在 hidden-state 代理中可分。

Phase211:
  成功/失败切换点候选图谱，缩小因果验证范围。
```

### 九、下一阶段任务

下一阶段应进入：

```text
Phase212: 小规模切换点因果验证
```

不建议大规模 patch。应选择 2 到 4 个最清楚候选：

```text
1. qwen3 answer_list success vs other_or_wrong
   layer 32, step 11

2. qwen3 answer_list success vs short_answer
   layer 32, step 9

3. GLM4 answer_list success vs repeat_answer / echo_then_answer
   layer 29 / 35, step 8

4. DS7B answer_explain success vs other_or_wrong
   layer 26, step 7
   但 DS7B 样本少，作为低权重验证。
```

因果验证方式：

```text
1. success mean state -> drift trajectory patch；
2. drift mean state -> success trajectory patch；
3. 只 patch 一个 layer-step；
4. 观察输出模式是否改变；
5. 记录 target/prose/echo/stop/EOS 边界变化；
6. 不追求一次闭合，只判断候选点是否有方向性因果影响。
```

通俗总结：Phase211 没有证明“某个层控制某个模式”，但把搜索范围缩小了。现在最值得验证的是 list 模式的成功/漂移分叉，尤其 qwen3 的 layer 32 和 GLM4 的 layer 29/35。下一步只做少量精确干预，看看这些候选点是不是能真的推动模式从漂移回到目标轨迹。

## Phase 212: 小规模切换点因果验证 [2026-07-06 19:53]

### 一、任务判断

本轮输入对 Phase211 的判断正确：

```text
Phase211 已经从“模式差异”推进到“成功轨迹和漂移轨迹在哪些层-步位置分叉”；
但 Phase211 仍然只是 SwitchPointCandidate（切换点候选），不是 SwitchPointCause（切换点因果）。
```

因此，当前继续任务仍属于同一阶段目标：

```text
完成语言编码机制的全局图谱拼图。
```

但本轮不适合大规模 patch。更合理的做法是：

```text
只选 Phase211 中最清楚的少数候选点；
做单 layer-step hidden-state mean patch；
验证候选点是否有方向性因果影响。
```

### 二、脚本和结果

新增脚本：

```text
tests/gpt5/phase212_switchpoint_causal_validation.py
tests/gpt5/run_phase212_switchpoint_causal_validation.sh
```

结果目录：

```text
tests/result/phase212_switchpoint_causal_validation/switchpoint_causal_validation/
```

核心文件：

```text
phase212_cross_model_summary.json
phase212_cross_model_summary.md
phase212_qwen3_summary.json
phase212_glm4_summary.json
phase212_deepseek7b_summary.json
phase212_*_patch_rollout_rows.jsonl
phase212_*_candidate_summary_rows.jsonl
```

### 三、验证方法

对每个候选点：

```text
1. 选择同一 model / pattern / failure_mode / layer / step；
2. 从 Phase210 trajectory rows 中取 success 轨迹和 drift 轨迹；
3. 重新加载模型；
4. 在该 layer-step 重新计算 success mean hidden state；
5. 重新计算 drift mean hidden state；
6. 对 drift 轨迹做 success_mean patch；
7. 对 success 轨迹做 drift_mean patch；
8. 比较 patch 前后输出模式是否改变。
```

核心干预：

$$
\boxed{
h_{l,t}^{patched}
=
\mathbb{E}
\left[
h_{l,t}
\mid
\mathrm{success}
\right]
}
$$

用于 drift trajectory repair（漂移轨迹修复）。

反向干预：

$$
\boxed{
h_{l,t}^{patched}
=
\mathbb{E}
\left[
h_{l,t}
\mid
\mathrm{drift}
\right]
}
$$

用于 success trajectory damage（成功轨迹破坏）。

本轮只 patch 一个 layer-step，不做多层、多步组合。

### 四、候选点

验证候选：

```text
qwen3:
  answer_list -> other_or_wrong
  layer 32, step 11

qwen3:
  answer_list -> short_answer
  layer 32, step 9

GLM4:
  answer_list -> repeat_answer
  layer 29, step 8

GLM4:
  answer_list -> echo_then_answer
  layer 35, step 8

DS7B:
  answer_explain -> other_or_wrong
  layer 26, step 7

DS7B:
  answer_list -> other_or_wrong
  layer 24, step 7
```

### 五、客观结果

总结果：

```text
rollout rows: 92
total repair match gain: 0
total damage match loss: 2
```

按候选点：

```text
qwen3 answer_list -> other_or_wrong, L32 S11:
  success rows = 8
  drift rows = 6
  repair gain = 0
  damage loss = 0

qwen3 answer_list -> short_answer, L32 S9:
  success rows = 8
  drift rows = 2
  repair gain = 0
  damage loss = 0

GLM4 answer_list -> repeat_answer, L29 S8:
  success rows = 1
  drift rows = 8
  repair gain = 0
  damage loss = 0

GLM4 answer_list -> echo_then_answer, L35 S8:
  success rows = 1
  drift rows = 2
  repair gain = -2
  damage loss = 0

DS7B answer_explain -> other_or_wrong, L26 S7:
  success rows = 6
  drift rows = 2
  repair gain = 2
  damage loss = 0

DS7B answer_list -> other_or_wrong, L24 S7:
  success rows = 6
  drift rows = 2
  repair gain = 0
  damage loss = 2
```

### 六、核心发现

第一，qwen3 的两个高分候选被初步否定。

```text
qwen3 answer_list L32 S11:
  success_mean patch 不能把 other_or_wrong 修复为 list；
  drift_mean patch 也不能破坏成功 list。

qwen3 answer_list L32 S9:
  success_mean patch 不能把 short_answer 修复为 list；
  还会把 short_answer 推到 echo_then_answer；
  drift_mean patch 不能破坏成功 list。
```

这说明：

```text
qwen3 layer 32 的高 switchpoint score 更像相关/放大/读出代理；
不是单点充分因果控制点。
```

第二，GLM4 的 list 候选也没有正向修复。

```text
GLM4 answer_list -> repeat_answer, L29 S8:
  repair gain = 0
  damage loss = 0

GLM4 answer_list -> echo_then_answer, L35 S8:
  repair gain = -2
  damage loss = 0
```

其中第二个候选出现负向结果：

```text
success_mean patch 反而让 baseline 判为 list 的 drift eval rows 变回 echo_then_answer。
```

这说明：

```text
GLM4 的候选点很可能不是简单“success mean state 越多越好”；
模式状态可能依赖样本、位置、前文轨迹和多点组合。
```

第三，DS7B answer_explain 出现弱正因果信号。

```text
DS7B answer_explain -> other_or_wrong, L26 S7:
  drift baseline: 0/2 match；
  success_mean patch: 2/2 match；
  repair gain = 2。
```

生成前缀从：

```text
Horses are primarily used for transportation, but they are also...
```

变为：

```text
Horses are primarily used for because they are excellent at carrying...
```

这不是高质量答案，但分类上进入 explain_answer。

严格结论：

```text
DS7B layer 26 step 7 对 explain 模式有方向性因果影响候选；
但样本只有 2 条 drift，且输出质量不高，不能视为闭合。
```

第四，DS7B answer_list 出现反向破坏信号。

```text
DS7B answer_list -> other_or_wrong, L24 S7:
  success baseline: 4/6 match；
  drift_mean patch: 2/6 match；
  damage loss = 2。
```

这说明：

```text
drift mean state 可以削弱 list 成功轨迹；
但 success mean state 不能修复 drift trajectory。
```

这更像：

```text
list 成功需要多条件共同维持；
单点 drift state 足以扰乱部分成功轨迹；
单点 success state 不足以修复完整漂移。
```

### 七、对 Phase211 的校正

Phase211 的 switchpoint score 有用，但不能直接等同因果强度。

本轮结果显示：

```text
高分候选可能是：
1. 相关点；
2. 放大点；
3. 读出点；
4. 局部扰动敏感点；
5. 真实因果切换点。
```

Phase212 只支持少量弱因果信号：

```text
DS7B explain L26 S7: repair positive；
DS7B list L24 S7: damage positive。
```

不支持：

```text
qwen3 list L32 S9/S11 是单点因果开关；
GLM4 list L29/L35 是单点因果开关。
```

### 八、问题和硬伤

第一，样本量仍小。

尤其 GLM4 的 success rows 只有 1，DS7B drift rows 只有 2。不能把单个候选的结果过度泛化。

第二，patch 方式很粗糙。

本轮直接替换 layer output 的 last-token hidden state：

```text
h_l,t = mean_success 或 mean_drift
```

这可能破坏样本特异性，也可能引入分布外状态。

第三，单点 patch 可能不足。

模式切换可能不是一个 layer-step，而是：

```text
多层连续窗口；
多 token 累积；
attention route + residual state + readout margin 的组合。
```

第四，分类器仍然启发式。

DS7B explain repair 的文本质量并不好，只是进入 explain_answer 分类。因此它是方向性证据，不是质量闭合。

第五，小模型偏差仍然很大。

DS7B 的 positive result 可能与 reasoning/explain 偏置有关，不一定代表通用语言机制。

### 九、理论进展

当前理论不改名。Phase212 增加因果层级区分：

```text
SwitchPointCandidate != SwitchPointCause
```

更严格地说：

$$
\boxed{
\mathrm{Candidate}(l,t)
\not\Rightarrow
\mathrm{Cause}(l,t)
}
$$

因果证据需要：

$$
\boxed{
\mathrm{Patch}
\left(
h_{l,t}^{success}
\to
h_{l,t}^{drift}
\right)
\Rightarrow
\Delta \mathrm{PatternMatch} > 0
}
$$

或：

$$
\boxed{
\mathrm{Patch}
\left(
h_{l,t}^{drift}
\to
h_{l,t}^{success}
\right)
\Rightarrow
\Delta \mathrm{PatternMatch} < 0
}
$$

Phase212 目前只得到弱版本：

```text
部分候选有方向性影响；
多数候选不具备单点充分因果控制。
```

### 十、当前阶段判断

当前任务和下一任务仍属于同一阶段性目标：

```text
完成语言编码机制的全局图谱拼图。
```

Phase209 到 Phase212 的进展链条：

```text
Phase209:
  输出模式图谱。

Phase210:
  hidden-state 模式差分图谱。

Phase211:
  成功/漂移切换点候选图谱。

Phase212:
  小规模切换点因果验证，发现多数高分点非单点因果，少数 DS7B 点有弱方向性影响。
```

### 十一、下一阶段任务

下一阶段不应继续盲目扩大单点 patch。

建议进入：

```text
Phase213: 多点窗口因果验证与 prompt 触发轨迹分析
```

重点：

```text
1. 对 DS7B positive candidate 做复测，增加样本和相邻 layer/step 窗口；
2. 对 qwen3 / GLM4 的 list 失败，改用多点窗口 patch，而不是单点；
3. 对 prompt 中 list/explain/repeat/target_seeded 触发词做 token-level hidden-state 轨迹分析；
4. 区分 cause point、amplification point、readout point；
5. 输出 multi_site_pattern_causal_window_atlas_v1。
```

通俗总结：Phase212 是一次必要的“踩刹车”。它没有证明 qwen3/GLM4 的高分切换点是单点开关，反而说明很多高分点只是相关或放大位置。真正有价值的是 DS7B 的 explain/list 出现弱方向性因果信号。下一步要从“单点”升级到“小窗口”和“prompt 触发轨迹”，否则会在单点 patch 上进入边际收益递减。

## Phase 213: 窗口方向补丁与 Prompt 触发态图谱 [2026-07-06 20:07]

### 一、任务判断

本轮输入对 Phase212 的判断正确：

```text
Phase212 没有否定动态模式网络路线；
它校准了因果层级；
高分切换点不等于单点因果开关；
后续应从单点 patch 升级为小窗口方向补丁和 prompt token 触发轨迹分析。
```

因此，本轮继续同一阶段目标：

```text
完成语言编码机制的全局图谱拼图。
```

但不继续盲目扩大单点 patch，而是测试：

```text
1. 多 layer-step 小窗口方向补丁是否比单点均值替换更有效；
2. prompt 结束态是否已经携带模式成败差异。
```

### 二、脚本和结果

新增脚本：

```text
tests/gpt5/phase213_window_direction_prompt_trigger.py
tests/gpt5/run_phase213_window_direction_prompt_trigger.sh
```

结果目录：

```text
tests/result/phase213_window_direction_prompt_trigger/window_direction_prompt_trigger/
```

核心文件：

```text
phase213_cross_model_summary.json
phase213_cross_model_summary.md
phase213_qwen3_summary.json
phase213_glm4_summary.json
phase213_deepseek7b_summary.json
phase213_*_window_rollout_rows.jsonl
phase213_*_window_summary_rows.jsonl
phase213_*_prompt_trigger_rows.jsonl
```

### 三、方法

Phase212 使用直接均值替换：

$$
h_{l,t}^{patched}
=
\mu_{\mathrm{success}}
$$

Phase213 改为更温和的方向补丁：

$$
\boxed{
h_{l,t}^{\prime}
=
h_{l,t}
+
\lambda
\left(
\mu_{\mathrm{success}}(l,t)
-
\mu_{\mathrm{drift}}(l,t)
\right)
}
$$

用于 drift trajectory repair。

反向破坏：

$$
\boxed{
h_{l,t}^{\prime}
=
h_{l,t}
-
\lambda
\left(
\mu_{\mathrm{success}}(l,t)
-
\mu_{\mathrm{drift}}(l,t)
\right)
}
$$

用于 success trajectory damage。

本轮使用：

```text
direction_scale = 0.7
每个窗口包含 3 层 x 2 步 = 6 个 direction sites。
```

同时记录 prompt 结束处 hidden state：

```text
prompt_trigger_rows = 920
```

这只是 prompt last-token trigger state（提示末词元触发态），还不是完整 token-level 触发词轨迹。

### 四、候选窗口

qwen3：

```text
answer_list -> other_or_wrong
layers = 31, 32, 33
steps = 10, 11

answer_list -> short_answer
layers = 31, 32, 33
steps = 8, 9
```

GLM4：

```text
answer_list -> repeat_answer
layers = 28, 29, 30
steps = 7, 8

answer_list -> echo_then_answer
layers = 34, 35, 36
steps = 7, 8
```

DS7B：

```text
answer_explain -> other_or_wrong
layers = 25, 26, 27
steps = 6, 7

answer_list -> other_or_wrong
layers = 23, 24, 25
steps = 6, 7
```

### 五、客观结果

总结果：

```text
rollout rows: 92
prompt trigger rows: 920
total repair match gain: 0
total damage match loss: 0
```

按模型：

```text
qwen3:
  rollout rows = 40
  prompt trigger rows = 240
  repair gain = 0
  damage loss = 0

GLM4:
  rollout rows = 20
  prompt trigger rows = 480
  repair gain = -2
  damage loss = 0

DS7B:
  rollout rows = 32
  prompt trigger rows = 200
  repair gain = 2
  damage loss = 0
```

按候选：

```text
qwen3 answer_list -> other_or_wrong:
  window L31/32/33 S10/11
  repair gain = 0
  damage loss = 0

qwen3 answer_list -> short_answer:
  window L31/32/33 S8/9
  repair gain = 0
  damage loss = 0

GLM4 answer_list -> repeat_answer:
  window L28/29/30 S7/8
  repair gain = 0
  damage loss = 0

GLM4 answer_list -> echo_then_answer:
  window L34/35/36 S7/8
  repair gain = -2
  damage loss = 0

DS7B answer_explain -> other_or_wrong:
  window L25/26/27 S6/7
  repair gain = 2
  damage loss = 0

DS7B answer_list -> other_or_wrong:
  window L23/24/25 S6/7
  repair gain = 0
  damage loss = 0
```

### 六、核心发现

第一，窗口方向补丁没有带来跨模型净收益。

```text
总 repair gain = 0
总 damage loss = 0
```

这说明：

```text
把单点扩展成 3 层 x 2 步窗口，仍不足以稳定修复 qwen3/GLM4 的 list 漂移。
```

第二，qwen3 的 list 漂移进一步被确认不是简单后层窗口问题。

```text
qwen3 两个窗口都是 0 repair / 0 damage。
```

但 window patch 会改变漂移类型：

```text
other_or_wrong 有部分转为 next_task_or_format；
short_answer 有部分转为 explain_answer。
```

所以窗口方向不是完全无效，而是没有沿目标 list 模式方向闭合。

第三，GLM4 复现负向结果。

```text
GLM4 answer_list -> echo_then_answer:
repair gain = -2
```

这和 Phase212 的负向结果一致。

说明：

```text
GLM4 的 list/echo 边界不是简单 success-minus-drift 方向；
方向补丁可能破坏局部样本结构，反而加强 echo_then_answer。
```

第四，DS7B explain 的弱正因果信号被复现。

```text
Phase212:
DS7B explain L26 S7 repair gain = 2

Phase213:
DS7B explain L25/26/27 S6/7 repair gain = 2
```

这说明：

```text
DS7B explain 模式确实存在可干预的局部窗口；
但样本仍只有 2 条 drift，且输出质量仍未证明干净闭合。
```

第五，DS7B list 的 damage 信号没有在窗口方向补丁中复现。

Phase212：

```text
DS7B list L24 S7 damage loss = 2
```

Phase213：

```text
DS7B list L23/24/25 S6/7 damage loss = 0
```

说明：

```text
DS7B list 的单点 damage 可能较脆弱；
窗口方向补丁没有稳定复现。
```

第六，prompt 结束态已产生 920 条记录。

初步观察显示：

```text
不同 pattern 的 prompt_last_residual_norm 存在差异；
但当前只记录 prompt 末词元，不足以定位 list/explain/repeat 等触发词本身。
```

它可作为下一轮 prompt token-level trajectory 的底层材料。

### 七、问题和硬伤

第一，窗口方向仍不够细。

本轮使用：

```text
success_mean - drift_mean
```

这仍是均值方向，可能掩盖子簇、样本特异结构和具体触发词作用。

第二，窗口选择仍基于 Phase211 高分点。

如果 Phase211 高分点主要是 amplification/readout point（放大/读出点），窗口 patch 仍可能无法触及 cause point（原因点）。

第三，prompt 分析仍然粗糙。

本轮只看 prompt last-token state，没有逐 token 分析：

```text
three；
plausible；
answers；
because；
same answer word；
final answer only；
one word；
Answer:
```

第四，样本量仍受 Phase210 限制。

尤其 GLM4 success rows 和 DS7B drift rows 太少。需要更平衡的数据才能强验证。

第五，小模型偏差仍然明显。

DS7B explain 正信号可能来自模型的强解释/推理偏置，不能直接外推。

### 八、理论进展

Phase213 进一步支持：

```text
模式因果不是单点，也不只是简单小窗口均值方向。
```

更严格公式应从点扩展到路径：

$$
\boxed{
\mathrm{PatternCause}
\neq
\mathrm{Point}(l,t)
}
$$

也不一定等于简单窗口：

$$
\boxed{
\mathrm{PatternCause}
\neq
\sum_{(l,t)\in W}
\left(
\mu_{\mathrm{success}}(l,t)
-
\mu_{\mathrm{drift}}(l,t)
\right)
}
$$

更可能是：

$$
\boxed{
\mathrm{PatternCause}
=
\mathrm{TriggerPath}
\circ
\mathrm{RoutePath}
\circ
\mathrm{StatePath}
\circ
\mathrm{ReadoutPath}
}
$$

其中：

```text
TriggerPath = prompt 触发词形成的模式启动路径；
RoutePath = attention / residual 路由路径；
StatePath = 多层多步状态维持路径；
ReadoutPath = 输出边界和词表竞争路径。
```

当前闭合标准不变：

$$
\boxed{
\mathrm{ModelClose}
=
\mathrm{AnswerCorrect}
\land
\mathrm{PatternMatched}
\land
\mathrm{BoundaryStable}
\land
\mathrm{DoneStateStable}
\land
\mathrm{ModelStopExecuted}
\land
\mathrm{NoDrift}
}
$$

### 九、当前阶段判断

当前任务和下一任务仍属于同一阶段性目标：

```text
完成语言编码机制的全局图谱拼图。
```

Phase209 到 Phase213 的连续进展：

```text
Phase209:
  输出模式图谱。

Phase210:
  hidden-state 模式差分图谱。

Phase211:
  成功/漂移切换点候选图谱。

Phase212:
  单点因果校准，发现多数候选非单点因果。

Phase213:
  窗口方向补丁验证，发现简单窗口方向仍不能跨模型修复，DS7B explain 弱正信号复现。
```

### 十、下一阶段任务

下一步不应继续扩大 patch 规模，而应转向：

```text
Phase214: Prompt 触发词级轨迹与模式路径分解
```

重点：

```text
1. 对 prompt 中的触发词做 token-level hidden-state 轨迹；
2. 对 list/explain/repeat/short/target_seeded 提取触发词位置；
3. 比较 trigger token、Answer: token、生成 step 1、漂移 step 的状态路径；
4. 区分 TriggerPath、RoutePath、StatePath、ReadoutPath；
5. 只在路径明确后再做组件级因果验证。
```

通俗总结：Phase213 再次提醒我们，模式不是一个点，也不是简单小窗口。qwen3 和 GLM4 的 list 漂移无法靠后层方向补丁修复；DS7B explain 有可复现弱因果信号，但仍不闭合。下一步要回到模式触发源头，看 prompt 里的触发词如何启动路径，否则 patch 会一直在后层放大点附近打转。

## Phase 214: Prompt 触发词级轨迹与模式路径分解 [2026-07-06 21:12]

### 一、任务判断

本轮分析的附件判断基本正确。

Phase213 的核心意义不是简单的“窗口方向补丁失败”，而是说明：

```text
语言模式因果不是单点；
也不是简单 3 层 x 2 步窗口均值方向；
更可能是 prompt 触发、路由、状态维持、读出共同形成的路径机制。
```

因此继续扩大同类 patch（补丁）不是第一优先级。

当前任务和下一步任务仍处于同一阶段：

```text
完成语言编码机制的全局图谱拼图。
```

Phase214 进入：

```text
Prompt 触发词级轨迹与模式路径分解。
```

它不是闭合验证，而是路径图谱初版。

### 二、测试脚本

新增脚本：

```text
tests/gpt5/phase214_prompt_trigger_token_path_atlas.py
tests/gpt5/run_phase214_prompt_trigger_token_path_atlas.sh
```

结果目录：

```text
tests/result/phase214_prompt_trigger_token_path_atlas/prompt_trigger_token_path_atlas/
```

运行方式：

```text
qwen3 -> GLM4 -> DS7B
```

三模型按顺序加载和释放，没有并发占用 GPU。

### 三、算法原理

Phase214 不做干预，只做轨迹记录。

核心问题从：

```text
能否用后层 patch 修复漂移？
```

改为：

```text
prompt 中的触发词状态，是否和后续生成模式成功/漂移有关？
```

对每个 prompt（提示）定位触发词：

```text
answer_short:
  one English color word
  only

answer_explain:
  answer first
  short reason
  because

answer_list:
  three
  plausible
  short answers
  commas

answer_repeat:
  exactly
  same answer word
  twice
  comma

answer_target_seeded:
  likely
  final answer
  only

common:
  Answer:
```

对每个触发词 token（词元）记录多层 hidden state（隐藏状态）：

$$
\boxed{
h^{prompt}_{l,p}
}
$$

其中：

```text
l = layer（层）；
p = prompt 内触发词 token 位置。
```

再记录触发词到生成阶段 anchor（锚点）的路径相似度：

```text
prompt_last；
gen_after_step_1；
gen_after_step_2；
gen_after_step_3；
gen_after_step_6；
gen_after_final。
```

核心指标：

$$
\boxed{
R_{trigger \to anchor}(l)
=
\cos
\left(
h^{prompt}_{l,p},
h^{anchor}_{l}
\right)
}
$$

以及成功/漂移差分：

$$
\boxed{
\Delta R(l)
=
\mathbb{E}
\left[
R_{trigger \to anchor}(l)
\mid success
\right]
-
\mathbb{E}
\left[
R_{trigger \to anchor}(l)
\mid drift
\right]
}
$$

这不是因果证明，只是路径相关图谱。

### 四、客观结果

总结果：

```text
selected trajectory rows = 340
trigger token rows = 14794
path rows = 88764
success/drift delta rows = 2754
```

分模型：

```text
qwen3:
  selected trajectory rows = 150
  trigger token rows = 5400
  path rows = 32400
  selected layers = 3,6,11,18,24,29,31,32,33

GLM4:
  selected trajectory rows = 150
  trigger token rows = 7722
  path rows = 46332
  selected layers = 3,7,12,20,27,28,29,30,32,34,35,36,37

DS7B:
  selected trajectory rows = 40
  trigger token rows = 1672
  path rows = 10032
  selected layers = 2,5,9,14,18,22,23,24,25,26,27
```

### 五、主要现象

第一，Phase214 确认 prompt 触发路径值得继续追。

三模型都能形成大量：

```text
trigger token -> generation anchor
```

的 success/drift（成功/漂移）差分记录。

这说明 Phase213 后转向 prompt token-level trajectory（提示词元级轨迹）是正确方向。

第二，DS7B 的 explain/list 触发路径出现强差异，但样本很少。

未过滤时，DS7B 最强差异集中在 L27：

```text
answer_list:
  list_three -> gen_after_step_1 L27
  success rows = 6
  drift rows = 2
  cosine delta = +0.869056

answer_list:
  list_plausible -> gen_after_step_1 L27
  success rows = 6
  drift rows = 2
  cosine delta = +0.807151

answer_explain:
  explain_because -> gen_after_step_1 L27
  success rows = 6
  drift rows = 2
  cosine delta = +0.799727
```

解释：

```text
DS7B 的 explain/list 成功样本中，prompt 触发词状态与生成第 1 步后状态更接近；
这和 Phase212/213 中 DS7B explain 有弱可干预信号相互呼应。
```

但硬伤也很明显：

```text
drift rows 只有 2；
不能作为强因果证据；
更像路径候选。
```

第三，过滤到 success/drift 都不少于 5 后，qwen3 和 GLM4 的稳定信号更有参考价值。

qwen3 较强信号：

```text
answer_repeat:
  answer_slot -> gen_after_step_6 L32
  success rows = 18
  drift rows = 12
  cosine delta = -0.23044

answer_repeat:
  answer_slot -> gen_after_step_6 L29
  success rows = 18
  drift rows = 12
  cosine delta = -0.22032

answer_explain:
  explain_because -> gen_after_step_3 L3
  success rows = 15
  drift rows = 15
  cosine delta = -0.21612
```

GLM4 较强信号：

```text
answer_explain:
  answer_slot -> gen_after_step_6 L7
  success rows = 7
  drift rows = 23
  cosine delta = +0.31286

answer_target_seeded:
  answer_slot -> gen_after_step_6 L7
  success rows = 15
  drift rows = 15
  cosine delta = -0.30322

answer_target_seeded:
  answer_slot -> gen_after_step_6 L3
  success rows = 15
  drift rows = 15
  cosine delta = -0.29609
```

这些结果说明：

```text
触发路径差异不只在后层；
早层和中层已经出现 success/drift 分叉；
Answer: 这个回答槽位置本身可能是重要的模式路由节点。
```

第四，GLM4 list 的极大差异不可靠。

未过滤时，GLM4 answer_list 有很大的差异：

```text
answer_list answer_slot -> gen_after_step_6 L7
success rows = 1
drift rows = 29
cosine delta = -0.682425
```

但 success rows 只有 1。

严格判断：

```text
这不能作为强现象；
只能说明 GLM4 list 成功样本过少，当前图谱不平衡。
```

第五，qwen3 的 repeat 和 explain 比 list 更适合做下一轮路径拆解。

qwen3 list 有成功和漂移，但强差异不如 repeat/explain 稳定。

这提示：

```text
如果下一轮做 attention/head route（注意力头路由）或 component-level validation（组件级验证），
qwen3 repeat/explain 可能比 list 更适合作为第一批样本。
```

### 六、理论进展

Phase214 支持把模式机制从点和窗口继续升级到路径。

当前公式保持：

$$
\boxed{
h_{t+1}
=
\sum_k
\alpha_k(x,t)T_k(h_t)
+
\varepsilon_t
}
$$

Phase214 进一步补充：

$$
\boxed{
\alpha_k(x,t)
\text{ 不是只在生成阶段产生，}
\text{而可能从 prompt trigger token 开始形成。}
}
$$

因此模式因果公式应写成：

$$
\boxed{
\mathrm{PatternCause}
=
\mathrm{TriggerPath}
\circ
\mathrm{RoutePath}
\circ
\mathrm{StatePath}
\circ
\mathrm{ReadoutPath}
}
$$

Phase214 当前只完成：

$$
\boxed{
\mathrm{TriggerPath}
\to
\mathrm{StatePath}
\text{ 的相关图谱初版}
}
$$

尚未完成：

```text
RoutePath 的 attention head（注意力头）分解；
MLP channel（多层感知机通道）分解；
ReadoutPath 的词表竞争分解；
因果干预验证。
```

### 七、问题和硬伤

第一，Phase214 仍是相关性图谱，不是因果闭合。

当前只能说明：

```text
某些 prompt 触发词状态和后续成功/漂移有差异。
```

不能说明：

```text
改写这些触发词状态一定会修复漂移。
```

第二，样本仍不平衡。

最明显的是：

```text
GLM4 answer_list success rows 太少；
DS7B answer_explain/list drift rows 太少；
answer_short 几乎没有成功闭合样本。
```

第三，触发词定位仍是字符串级启发式。

本轮定位：

```text
three；
plausible；
because；
same answer word；
final answer；
Answer:
```

但真实触发可能跨多个 token，甚至不是单个短语。

第四，尚未加入 attention head。

附件建议记录：

```text
A^{prompt}_{l,h,p}
```

本轮没有做 attention head 输出或注意力权重图谱，只完成 residual stream（残差流）触发轨迹。

第五，小模型偏差仍需 30% 到 50% 折扣。

这些结果更准确地说是：

```text
小模型 prompt-trigger path atlas（提示触发路径图谱）。
```

不能直接外推为通用语言机制。

### 八、阶段性结论

Phase214 的核心结论：

```text
prompt 触发词状态确实能形成 success/drift 路径差异；
模式分叉很可能早于后层 readout；
后层 patch 失败的原因，可能是它在已经分叉后的状态路径上修补，而不是在触发/路由源头修补。
```

更通俗地说：

```text
模型不是到输出末端才决定“解释、列表、复读、短答”；
这些模式很可能在 prompt 中的触发词和 Answer: 槽位附近就开始分叉。
```

但这仍不是闭合。

### 九、下一阶段任务

下一步仍属于同一阶段性目标，应该继续自动推进到：

```text
Phase215: TriggerPath -> RoutePath 的注意力路由图谱
```

优先选择样本更平衡、信号更稳的模式：

```text
qwen3 answer_repeat；
qwen3 answer_explain；
GLM4 answer_target_seeded；
GLM4 answer_explain；
DS7B answer_explain/list 只作为弱候选保留。
```

Phase215 应完成：

```text
1. 对触发词 token 和 Answer: token 提取 attention pattern（注意力模式）；
2. 记录触发词是否被生成阶段早期 token 回读；
3. 区分 trigger point（触发点）、route point（路由点）、state point（状态点）、readout point（读出点）；
4. 只在路由图谱明确后，再做组件级因果干预。
```

当前总体进展估计：

```text
小模型模式机制图谱：约 56%
路径因果机制：约 9%
模型内部自然闭合：约 30%
任务层产品闭合：约 55%
通用语言机制外推置信：约 33% 到 38%
```

## Phase 215: Prompt 触发词注意力路由图谱 [2026-07-06 21:28]

### 一、任务判断

本轮附件对 Phase214 的评估基本正确。

Phase214 的价值不是闭合，而是把研究从：

```text
后层 patch 修补
```

推进到：

```text
prompt trigger path（提示触发路径）
```

它说明模式分叉很可能早于后层 readout（读出）。

因此 Phase215 继续同一阶段性目标：

```text
完成语言编码机制的全局图谱拼图。
```

本轮不做干预，而是补齐 Phase214 缺失的 RoutePath（路由路径）初版。

### 二、测试脚本

新增脚本：

```text
tests/gpt5/phase215_prompt_attention_route_atlas.py
tests/gpt5/run_phase215_prompt_attention_route_atlas.sh
```

结果目录：

```text
tests/result/phase215_prompt_attention_route_atlas/prompt_attention_route_atlas/
```

运行方式：

```text
qwen3 -> GLM4 -> DS7B
```

三模型顺序运行，使用 eager attention（逐头注意力输出），每个模型完成后释放显存。

### 三、算法原理

Phase215 的核心问题：

```text
生成早期 token 是否通过 attention head 回读 prompt 中的触发词、Answer:、对象词、关系词、目标答案词？
```

对每条轨迹，在以下 anchor（锚点）取最后 token 作为 query（查询位置）：

```text
prompt_last；
gen_after_step_1；
gen_after_step_3；
gen_after_step_6。
```

对 prompt 侧 source group（源位置组）计算每层每头注意力质量：

```text
trigger:any；
trigger:具体触发词；
answer_slot；
object；
target_label；
relation；
question_prefix；
instruction_to_answer；
prompt_all。
```

核心公式：

$$
\boxed{
M_{l,h}^{G}(q)
=
\sum_{p\in G}
A_{l,h}(q,p)
}
$$

其中：

```text
M = 某层某头对源位置组 G 的注意力质量；
l = layer（层）；
h = head（注意力头）；
q = 当前 anchor 的 query 位置；
p = prompt 中的 source token 位置；
A = attention matrix（注意力矩阵）。
```

成功/漂移路由差分：

$$
\boxed{
\Delta M_{l,h}^{G}
=
\mathbb{E}
\left[
M_{l,h}^{G}
\mid success
\right]
-
\mathbb{E}
\left[
M_{l,h}^{G}
\mid drift
\right]
}
$$

这仍是相关图谱，不是因果证明。

### 四、客观结果

总结果：

```text
selected trajectory rows = 111
attention route rows = 113408
summary rows = 16320
route delta rows = 8160
```

分模型：

```text
qwen3:
  selected trajectory rows = 48
  attention route rows = 49152
  route delta rows = 3072
  selected layers = 3,6,11,24,29,31,32,33

GLM4:
  selected trajectory rows = 47
  attention route rows = 48128
  route delta rows = 3072
  selected layers = 3,7,12,20,27,28,29,30

DS7B:
  selected trajectory rows = 16
  attention route rows = 16128
  route delta rows = 2016
  selected layers = 2,5,14,22,23,24,25,26,27
```

样本分布：

```text
qwen3:
  answer_explain success/drift = 8/8
  answer_list success/drift = 8/8
  answer_repeat success/drift = 8/8

GLM4:
  answer_explain success/drift = 7/8
  answer_repeat success/drift = 8/8
  answer_target_seeded success/drift = 8/8

DS7B:
  answer_explain success/drift = 6/2
  answer_list success/drift = 6/2
```

### 五、主要现象

第一，RoutePath 候选成立。

三模型都出现：

```text
success 与 drift 在生成早期对 prompt 源 token 的注意力质量不同。
```

这说明 Phase214 的 TriggerPath（触发路径）不是孤立相关现象。

更合理的图谱是：

```text
prompt trigger token
-> attention route
-> generation state
-> readout/output
```

第二，GLM4 的 target_seeded 路由信号最稳。

GLM4 在 answer_target_seeded 中出现多个强触发词路由差异：

```text
answer_target_seeded gen_after_step_6 L29H28
success/drift = 8/8
trigger:any delta = +0.6162

answer_target_seeded gen_after_step_6 L29H10
success/drift = 8/8
trigger:any delta = +0.5537

answer_target_seeded gen_after_step_6 L29H18
success/drift = 8/8
trigger:any delta = +0.5504

answer_target_seeded gen_after_step_6 L29H11
success/drift = 8/8
trigger:any delta = +0.5383

answer_target_seeded gen_after_step_6 L29H25
success/drift = 8/8
trigger:any delta = +0.5132
```

这些 head 集中在：

```text
L29 附近；
gen_after_step_6；
target_final_answer / trigger:any。
```

解释：

```text
GLM4 的目标答案种子模式，可能存在较清楚的触发词回读路由。
成功轨迹在生成后期更强回读 final answer / 触发指令区域。
```

第三，GLM4 的 repeat/explain 也有路由候选。

例如：

```text
answer_repeat gen_after_step_3 L12H21
success/drift = 8/8
trigger:any delta = +0.5219

answer_explain gen_after_step_3 L12H18
success/drift = 7/8
trigger:any delta = -0.4386

answer_explain gen_after_step_1 L20H25
success/drift = 7/8
trigger:any delta = -0.3503
```

这说明 GLM4 的路由不只限于 target_seeded。

但 target_seeded 最集中、最稳定。

第四，qwen3 的 explain/repeat 有较稳 RoutePath 候选。

qwen3 强触发词路由差异：

```text
answer_explain gen_after_step_3 L3H15
success/drift = 8/8
trigger:any delta = -0.4903

answer_explain gen_after_step_6 L29H11
success/drift = 8/8
trigger:any delta = +0.3799

answer_explain gen_after_step_1 L11H3
success/drift = 8/8
trigger:any delta = +0.3744

answer_repeat prompt_last L31H26
success/drift = 8/8
trigger:any delta = -0.3544

answer_repeat gen_after_step_1 L29H11
success/drift = 8/8
trigger:any delta = +0.3533
```

这与 Phase214 的判断一致：

```text
qwen3 repeat/explain 比 list 更适合作为下一轮路径拆解对象。
```

第五，DS7B 信号很强但仍不稳。

DS7B 的 answer_explain/list 出现大量强差异：

```text
answer_explain gen_after_step_1 L24H20
success/drift = 6/2
trigger:any delta = -0.8713
answer_slot delta = -0.8745

answer_explain gen_after_step_1 L24H16
success/drift = 6/2
trigger:any delta = -0.8330

answer_list gen_after_step_1 L24H20
success/drift = 6/2
trigger:any delta = -0.6579
```

这和 Phase212/213/214 中 DS7B explain/list 的弱正信号互相呼应。

但必须谨慎：

```text
drift rows 只有 2；
这些强差异只能作为候选，不能作为强结论。
```

第六，RoutePath 不只是触发词，也包括 question_prefix（问题区域）和 instruction_to_answer（指令到回答槽区域）。

聚合统计显示，强差异频繁出现在：

```text
question_prefix；
instruction_to_answer；
trigger:any；
answer_slot；
object。
```

这说明真实路由路径可能不是“只回读一个触发词”，而是：

```text
问题内容 + 指令结构 + 回答槽 + 触发词
```

共同形成模式路由。

### 六、理论进展

Phase215 把 Phase214 的路径公式补上了一块。

此前：

$$
\boxed{
\mathrm{TriggerPath}
\to
\mathrm{StatePath}
}
$$

现在增加：

$$
\boxed{
\mathrm{TriggerPath}
\to
\mathrm{RoutePath}
\to
\mathrm{StatePath}
}
$$

模式因果主公式保持：

$$
\boxed{
\mathrm{PatternCause}
=
\mathrm{TriggerPath}
\circ
\mathrm{RoutePath}
\circ
\mathrm{StatePath}
\circ
\mathrm{ReadoutPath}
}
$$

Phase215 的贡献是：

```text
RoutePath 有了逐头注意力候选；
但还没有证明这些 head 是因果组件。
```

更精确地说：

$$
\boxed{
\mathrm{RouteCandidate}_{l,h,G}
=
\Delta M_{l,h}^{G}
}
$$

其中：

```text
G = prompt source group（提示源位置组）。
```

如果某个 head 在多个样本中对 G 的注意力差分稳定，则它是 RoutePath 候选。

### 七、问题和硬伤

第一，本轮仍是注意力相关图谱，不是因果验证。

注意力质量差异不能自动等于：

```text
该 head 决定模式成功/漂移。
```

下一步必须做 head-level causal validation（注意力头级因果验证）。

第二，attention mass（注意力质量）只说明读了哪里，不说明写了什么。

即使 head 回读 trigger token（触发词），也可能只是旁观。

需要后续结合：

```text
head output；
residual delta；
readout margin；
generation behavior。
```

第三，source group 仍较粗。

本轮分组：

```text
trigger:any；
question_prefix；
instruction_to_answer；
answer_slot；
object；
target_label。
```

但真实路由可能依赖：

```text
短语结构；
相邻 token；
标点；
对象-关系组合；
句式整体。
```

第四，DS7B 样本仍不足。

DS7B 强结果不能直接当强证据。

第五，小模型偏差仍需 30% 到 50% 折扣。

当前结果仍是：

```text
small-model route atlas（小模型路由图谱）。
```

### 八、阶段性结论

Phase215 的核心结论：

```text
prompt 触发词和回答槽不只是静态提示；
生成早期确实存在对这些 prompt 源位置的差分回读路由；
这种路由差异和 success/drift 模式分叉相关。
```

这支持当前路线：

```text
语言编码机制不是点；
不是简单窗口；
而是触发、路由、状态维持、读出的路径网络。
```

通俗说：

```text
模型在回答时会回头看 prompt 中的关键位置；
成功和漂移的区别，可能部分来自“看回哪里、哪个 head 看、什么时候看”。
```

### 九、下一阶段任务

下一步仍属于同一阶段目标。

建议进入：

```text
Phase216: 路由头因果校准与写入作用验证
```

优先候选：

```text
GLM4 answer_target_seeded:
  L29H28
  L29H10
  L29H18
  L29H11
  L29H25

qwen3 answer_explain:
  L3H15
  L29H11
  L11H3

qwen3 answer_repeat:
  L31H26
  L29H11

DS7B answer_explain/list:
  L24H20
  L24H16
  L25H1
  仅作为弱候选保留。
```

Phase216 不应直接做大规模 patch。

应先做小规模校准：

```text
1. head ablation（注意力头消融）是否降低 PatternMatched；
2. source-restricted ablation（限制源位置的消融）是否比全 head 更精确；
3. head output norm / residual delta 是否和模式差异一致；
4. 如果因果弱，则回退到更完整的 source group，而不是继续硬 patch。
```

当前进度估计：

```text
小模型模式机制图谱：约 58%
TriggerPath：约 38%
RoutePath：约 18%
StatePath：约 35%
ReadoutPath：约 20%
路径因果机制：约 12%
模型内部自然闭合：约 30%
任务层产品闭合：约 55%
通用语言机制外推置信：约 34% 到 39%
```

## Phase 216: 路由头因果校准与写入作用验证 [2026-07-06 21:44]

### 一、任务判断

本轮附件对 Phase215 的判断基本正确。

Phase215 已经证明：

```text
success/drift 在 prompt source 回读路径上存在差异；
RoutePath 候选成立；
但注意力质量只说明“读哪里”，不说明“写什么”，更不说明因果控制。
```

因此 Phase216 继续同一阶段目标：

```text
完成语言编码机制的全局图谱拼图。
```

本轮进入小规模 head-level causal calibration（注意力头级因果校准），但仍不追求闭合。

### 二、测试脚本

新增脚本：

```text
tests/gpt5/phase216_route_head_causal_calibration.py
tests/gpt5/run_phase216_route_head_causal_calibration.sh
```

结果目录：

```text
tests/result/phase216_route_head_causal_calibration/route_head_causal_calibration/
```

运行顺序：

```text
qwen3 -> GLM4 -> DS7B
```

三模型顺序加载和释放，没有并发占用 GPU。

### 三、算法原理

Phase216 从 Phase215 的 RouteCandidate（路由候选）中选取 top heads（高分注意力头），做小规模消融。

消融方式：

```text
在 self_attn.o_proj 输入前，把指定 head 的 head slice 置零。
```

对每个候选 head 测试三种条件：

```text
none:
  不消融。

ablate_anchor_step:
  只在候选 anchor 对应生成步消融。

ablate_all_steps:
  每个生成步都消融该 head。
```

核心公式：

$$
\boxed{
z'_{l,h,t}
=
0
}
$$

其中：

```text
z_{l,h,t} = 第 l 层第 h 个注意力头在生成步 t 的 o_proj 输入切片；
z' = 消融后的 head 输出切片。
```

观测指标：

```text
PatternMatched 是否下降；
drift 是否被修复；
输出模式是否改变。
```

定义：

$$
\boxed{
\mathrm{Damage}
=
\mathrm{Match}_{success,none}
-
\mathrm{Match}_{success,ablate}
}
$$

$$
\boxed{
\mathrm{Repair}
=
\mathrm{Match}_{drift,ablate}
-
\mathrm{Match}_{drift,none}
}
$$

如果：

```text
Damage > 0
```

说明该 head 对成功模式可能有必要性。

如果：

```text
Repair > 0
```

说明该 head 可能支持漂移或竞争模式。

### 四、候选范围

qwen3：

```text
answer_explain:
  L3H15
  L29H11
  L11H3

answer_repeat:
  L31H26
  L29H11
```

GLM4：

```text
answer_target_seeded:
  L29H28
  L29H10
  L29H18
  L29H11
  L29H25

answer_repeat:
  L12H21

answer_explain:
  L12H18
```

DS7B：

```text
answer_explain:
  L24H20
  L24H16

answer_list:
  L24H20
```

其中 DS7B 仍为弱样本候选。

### 五、客观结果

总体：

```text
candidate count = 15
rollout rows = 342
effect rows = 30
total damage match loss = 1
total repair match gain = 0
```

分模型：

```text
qwen3:
  rollout rows = 120
  damage = 0
  repair = 0

GLM4:
  rollout rows = 168
  damage = -3
  repair = 0

DS7B:
  rollout rows = 54
  damage = 4
  repair = 0
```

### 六、主要现象

第一，qwen3 的 RoutePath 候选没有单头必要性证据。

qwen3 所有候选：

```text
damage = 0
repair = 0
```

具体基线可复现性较好：

```text
answer_explain:
  success none = 4/4 explain_answer

answer_repeat:
  success none = 4/4 repeat_answer
```

但消融：

```text
ablate_anchor_step；
ablate_all_steps；
```

都没有降低 PatternMatched。

严格结论：

```text
qwen3 的 L3H15、L29H11、L11H3、L31H26 等 head 是路由差异候选，
但不是当前测试下的单头必要因果组件。
```

第二，GLM4 target_seeded 的强路由候选没有通过必要性验证。

Phase215 中 GLM4 target_seeded 的 L29 heads 是最强 RoutePath 候选。

但 Phase216 发现：

```text
GLM4 answer_target_seeded success rows 在 none 条件下并未复现成功；
success none = 0/4；
输出全部是 repeat_answer。
```

例如：

```text
L29H28:
  success none output = repeat_answer 4/4

L29H10:
  success none output = repeat_answer 4/4

L29H18:
  success none output = repeat_answer 4/4
```

因此这些候选不能用于强必要性判断。

严格结论：

```text
GLM4 target_seeded 的 route delta 很强，
但当前小规模重生成基线不稳定；
不能说明 L29 heads 是 target_seeded 成功模式的必要因果组件。
```

第三，GLM4 explain L12H18 出现“负 damage”。

结果：

```text
glm4 answer_explain L12H18 ablate_all_steps:
  damage = -3
  repair = 0
```

解释：

```text
none 条件下 success rows 只有 1/4 匹配 explain_answer；
ablate_all_steps 后变为 4/4 explain_answer。
```

这不是必要性证据，而是说明：

```text
该 head 可能支持竞争/漂移成分；
或者 none 基线本身不稳定。
```

这值得保留为后续“竞争路由头”候选，但不能当闭合。

第四，DS7B explain L24H16 出现弱正因果信号。

结果：

```text
deepseek7b answer_explain L24H16 ablate_all_steps:
  success none = 4/4 explain_answer
  success ablate_all_steps = 0/4 explain_answer
  damage = 4
  repair = 0
```

输出从：

```text
explain_answer 4/4
```

变为：

```text
echo_then_answer 2/4
other_or_wrong 2/4
```

这说明：

```text
DS7B L24H16 对 explain 模式有必要性候选。
```

但必须低权重：

```text
DS7B drift rows 只有 2；
小模型 explain 偏置明显；
只有一个 head 在本轮显示强 damage。
```

第五，repair 仍然完全没有出现。

总体：

```text
total repair match gain = 0
```

这说明：

```text
消融高路由差异 head 不能把 drift 修复成 success。
```

换句话说：

```text
这些 head 即便参与成功模式，也不是简单“漂移开关”。
```

### 七、理论进展

Phase216 进一步收紧 Phase215 的结论。

Phase215：

$$
\boxed{
\mathrm{RouteCandidate}_{l,h,G}
=
\Delta M_{l,h}^{G}
}
$$

Phase216 证明：

$$
\boxed{
\mathrm{RouteCandidate}
\not\Rightarrow
\mathrm{RouteCause}
}
$$

更严格地说：

$$
\boxed{
\Delta M_{l,h}^{G}
\text{ 高}
\not\Rightarrow
\mathrm{Damage}_{l,h}>0
}
$$

当前更合理的路径因果判断应是：

$$
\boxed{
\mathrm{RouteCause}_{l,h}
=
\mathrm{RouteDifference}_{l,h}
\land
\mathrm{WriteEffect}_{l,h}
\land
\mathrm{BehaviorEffect}_{l,h}
}
$$

其中：

```text
RouteDifference = 注意力路由差异；
WriteEffect = head output 对 residual state 有可测写入；
BehaviorEffect = 消融或干预改变输出模式。
```

Phase216 当前只在 DS7B L24H16 上看到弱 BehaviorEffect。

### 八、问题和硬伤

第一，source-restricted ablation 尚未完成。

本轮做的是：

```text
full head ablation（整头消融）
```

还没有做：

```text
只阻断该 head 对 trigger/Answer:/instruction 的读取。
```

所以不能区分：

```text
head 本身重要；
还是 head 对某个 source group 的读取重要。
```

第二，head output 写入仍未直接测量。

本轮只看行为变化，没有直接记录：

```text
head output norm；
head output direction；
residual delta；
readout margin。
```

第三，GLM4 target_seeded 基线不可复现。

这是本轮最大硬伤之一。

Phase215 的强路由信号不能直接进入 Phase216 的因果结论，因为：

```text
success rows 在重新生成时已经不 success。
```

后续必须先做：

```text
baseline reproducibility filter（基线可复现过滤）。
```

第四，单头消融可能太弱。

模式路径可能由多个 head 共同实现。

因此：

```text
单头 damage = 0
```

不能证明该 head 完全无用，只能说明它不是单头必要组件。

第五，小模型偏差仍然存在。

当前结果仍只能叫：

```text
small-model route head calibration（小模型路由头校准）。
```

### 九、阶段性结论

Phase216 的核心结论：

```text
强 RoutePath 候选多数没有通过单头因果校准；
路由差异不是单头因果；
DS7B explain L24H16 是唯一弱正必要性候选；
GLM4 target_seeded 的强路由信号受基线不可复现限制；
qwen3 explain/repeat 路由候选更像分布式路径，不是单头开关。
```

这和前面 Phase212/213 的负结果一致：

```text
语言模式机制不是单点；
不是单窗口；
也不是单 head。
```

更可能是：

```text
多源 prompt 信息
-> 多头路由集合
-> residual state 写入
-> MLP/attention 状态维持
-> readout 竞争
```

### 十、下一阶段任务

下一步仍属于同一阶段目标。

不应继续盲目扩大 head patch。

建议进入：

```text
Phase217: 可复现基线过滤与多头路由集合验证
```

核心任务：

```text
1. 先重新生成所有 Phase215 候选样本，保留 none 条件下可复现 success/drift 的样本；
2. 对同一 pattern 的多个 route heads 做 head set ablation（头集合消融）；
3. 记录 head output norm 与 residual delta；
4. 对 GLM4 target_seeded 重新建立可复现样本集；
5. 对 DS7B L24H16 做扩大样本复测，但仍低权重解释。
```

当前进度估计：

```text
小模型模式机制图谱：约 59%
TriggerPath：约 38%
RoutePath：约 21%
RouteCause：约 6%
StatePath：约 35%
ReadoutPath：约 20%
路径因果机制：约 13%
模型内部自然闭合：约 30%
任务层产品闭合：约 55%
通用语言机制外推置信：约 34% 到 39%
```

## Phase 217: 可复现基线过滤与多头路由集合验证 [2026-07-06 22:12]

### 一、任务判断

本轮附件对 Phase216 的判断基本正确。

Phase216 的关键结果是：

```text
RouteCandidate 不等于 RouteCause；
单个 attention head 的高路由差异，大多不能推出行为因果；
GLM4 target_seeded 的强路由信号受 baseline reproducibility 失败限制。
```

因此 Phase217 继续同一阶段目标：

```text
完成语言编码机制的全局图谱拼图。
```

本轮优先解决：

```text
1. none 条件下 success/drift 是否可复现；
2. head set 是否比 single head 更有行为影响；
3. head output norm 是否能作为 WriteEffect 的轻量代理。
```

### 二、测试脚本

新增脚本：

```text
tests/gpt5/phase217_reproducible_headset_validation.py
tests/gpt5/run_phase217_reproducible_headset_validation.sh
```

结果目录：

```text
tests/result/phase217_reproducible_headset_validation/reproducible_headset_validation/
```

运行顺序：

```text
qwen3 -> GLM4 -> DS7B
```

三模型顺序加载和释放，避免显存重叠。

### 三、算法原理

Phase217 分两步。

第一步，baseline reproducibility filter（基线可复现过滤）。

对 Phase215/216 的候选样本重新生成 none 条件：

```text
原 success 样本：
  none 条件仍 PatternMatched 才保留。

原 drift 样本：
  none 条件仍 PatternDrift 才保留。
```

公式：

$$
\boxed{
\mathrm{KeepSuccess}(x)
=
\mathrm{PatternMatched}
\left(
G_{\mathrm{none}}(x)
\right)
}
$$

$$
\boxed{
\mathrm{KeepDrift}(x)
=
\neg
\mathrm{PatternMatched}
\left(
G_{\mathrm{none}}(x)
\right)
}
$$

第二步，head set ablation（头集合消融）。

对同一模式的多个 route heads 组成集合：

$$
\boxed{
H_{\mathrm{set}}
=
\{(l_i,h_i)\}_{i=1}^{n}
}
$$

消融：

$$
\boxed{
z'_{l,h,t}=0,
\quad
(l,h)\in H_{\mathrm{set}}
}
$$

测试条件：

```text
none；
headset_anchor_step；
headset_all_steps。
```

同时在 none 条件记录 head output norm：

$$
\boxed{
\|O_{l,h,t}\|_2
}
$$

这只是 WriteEffect（写入作用）的轻量代理，不是完整方向验证。

### 四、测试范围

qwen3：

```text
qwen3_explain_route_set:
  L3H15
  L11H3
  L29H11

qwen3_repeat_route_set:
  L31H26
  L29H11
```

GLM4：

```text
glm4_target_seeded_l29_route_set:
  L29H28
  L29H10
  L29H18
  L29H11
  L29H25

glm4_repeat_route_set:
  L12H21

glm4_explain_competition_route_set:
  L12H18
```

DS7B：

```text
deepseek7b_explain_l24_route_set:
  L24H20
  L24H16

deepseek7b_list_l24_route_set:
  L24H20
```

### 五、客观结果

总体：

```text
headset count = 7
filter rows = 95
reproducible success rows = 38
reproducible drift rows = 40
rollout rows = 198
head output norm rows = 2754
effect rows = 14
total damage match loss = 4
total repair match gain = 0
```

分模型：

```text
qwen3:
  filter rows = 32
  reproducible success rows = 16
  reproducible drift rows = 14
  rollout rows = 72
  norm rows = 1200
  damage = 0
  repair = 0

GLM4:
  filter rows = 47
  reproducible success rows = 12
  reproducible drift rows = 22
  rollout rows = 84
  norm rows = 1092
  damage = 0
  repair = 0

DS7B:
  filter rows = 16
  reproducible success rows = 10
  reproducible drift rows = 4
  rollout rows = 42
  norm rows = 462
  damage = 4
  repair = 0
```

### 六、主要现象

第一，qwen3 在可复现过滤后仍无 head set 因果信号。

qwen3：

```text
qwen3_explain_route_set:
  kept success = 6
  kept drift = 6
  damage = 0
  repair = 0

qwen3_repeat_route_set:
  kept success = 6
  kept drift = 6
  damage = 0
  repair = 0
```

输出保持：

```text
explain success -> explain_answer 6/6
repeat success -> repeat_answer 6/6
```

head set 消融后仍没有破坏成功，也没有修复漂移。

严格结论：

```text
qwen3 explain/repeat 的路由差异不是这些小 head set 的必要因果。
```

这比 Phase216 的单头负结果更强。

第二，GLM4 target_seeded 的基线不可复现被确认。

GLM4：

```text
glm4_target_seeded_l29_route_set:
  kept success = 0
  kept drift = 6
```

也就是说：

```text
Phase215 中最强的 GLM4 target_seeded RoutePath 候选，
在 Phase217 的 none 重生成中没有可复现 success 样本。
```

因此：

```text
它不能进入成功必要性验证；
只能说明 GLM4 的 target_seeded 轨迹在当前生成设置下非常不稳定。
```

第三，GLM4 repeat/explain 在可复现过滤后也无 head set 行为影响。

```text
glm4_repeat_route_set:
  kept success = 6
  kept drift = 6
  damage = 0
  repair = 0

glm4_explain_competition_route_set:
  kept success = 4
  kept drift = 6
  damage = 0
  repair = 0
```

Phase216 中 GLM4 L12H18 的负 damage 没有在可复现过滤 + head set 条件下形成稳定结论。

严格结论：

```text
GLM4 当前没有可靠 RouteCause。
```

第四，DS7B explain L24 head set 复现弱正必要性信号。

结果：

```text
deepseek7b_explain_l24_route_set:
  kept success = 6
  kept drift = 2
  headset_all_steps damage = 4
  repair = 0
```

输出从：

```text
success none:
  explain_answer 6/6
```

变成：

```text
headset_all_steps:
  explain_answer 2/6
  echo_then_answer 2/6
  other_or_wrong 2/6
```

这复现并扩展了 Phase216 的 DS7B explain L24H16 信号。

但仍要低权重解释：

```text
drift rows = 2；
DS7B 是小模型；
explain 模式可能有训练偏置；
repair 仍然为 0。
```

第五，DS7B list L24 head set 没有行为影响。

```text
deepseek7b_list_l24_route_set:
  kept success = 4
  kept drift = 2
  damage = 0
  repair = 0
```

说明：

```text
DS7B 的 explain 路由信号比 list 更接近因果。
```

第六，head output norm 记录显示“强写入不等于修复”。

head output norm 已正常记录：

```text
norm rows = 2754
```

典型现象：

```text
GLM4 target_seeded L29 heads 在 drift_repro 中 norm 很高：
  L29H18 step4 norm ≈ 30.82
  L29H28 step4 norm ≈ 30.57
  L29H10 step5 norm ≈ 30.10
```

但：

```text
damage = 0
repair = 0
```

这说明：

```text
强 head output / 强路由可能服务于漂移模式，而不是目标成功模式；
仅看范数和注意力质量都不够。
```

### 七、理论进展

Phase217 进一步收紧路径因果公式。

Phase216 给出：

$$
\boxed{
\mathrm{RouteCause}_{l,h}
=
\mathrm{RouteDifference}_{l,h}
\land
\mathrm{WriteEffect}_{l,h}
\land
\mathrm{BehaviorEffect}_{l,h}
}
$$

Phase217 进一步加入可复现条件：

$$
\boxed{
\mathrm{RouteCause}
=
\mathrm{ReproducibleBaseline}
\land
\mathrm{RouteDifference}
\land
\mathrm{WriteEffect}
\land
\mathrm{BehaviorEffect}
}
$$

并且：

$$
\boxed{
\mathrm{RouteSetCandidate}
\not\Rightarrow
\mathrm{RouteSetCause}
}
$$

当前只有：

```text
DS7B explain L24 head set
```

达到弱 RouteSetCause 候选。

### 八、问题和硬伤

第一，source-restricted ablation 仍未完成。

本轮是 head set 整体消融，仍不能区分：

```text
head 本身重要；
还是 head 对某个 source group 的读取重要。
```

第二，WriteEffect 只用了 norm 代理。

本轮记录：

```text
head output norm
```

但没有计算：

```text
head output direction；
success-drift 写入方向；
residual delta；
readout margin delta。
```

第三，GLM4 target_seeded 不稳定问题仍未解决。

它可能是：

```text
采样/贪心展开导致；
模式本身不稳；
分类启发式偏差；
prompt seed 与 repeat 模式冲突。
```

第四，repair 仍然完全为 0。

Phase217：

```text
total repair match gain = 0
```

说明当前消融方法不能把 drift 修成 success。

第五，小模型偏差仍然需要 30% 到 50% 折扣。

尤其 DS7B explain 的正信号不能过度外推。

### 九、阶段性结论

Phase217 的核心结论：

```text
可复现过滤是必要的；
qwen3 和 GLM4 的路由候选在 head set 消融下仍没有行为因果；
DS7B explain L24 head set 是当前唯一可复现弱必要性候选；
强注意力路由和强 head output norm 都不等于目标模式因果；
RouteCause 需要更细的 source-restricted 和方向写入验证。
```

这再次支持：

```text
语言模式机制不是单点；
不是单窗口；
不是单 head；
也不是简单 head set。
```

更可能是：

```text
多源触发
-> 多头路由
-> 有方向的状态写入
-> MLP/attention 状态维持
-> readout 竞争
```

### 十、下一阶段任务

下一步仍属于同一阶段目标。

建议进入：

```text
Phase218: 源位置限制消融与写入方向验证
```

优先对象：

```text
DS7B explain L24H16/L24H20：
  低权重但目前唯一可复现弱必要性候选。

qwen3 explain/repeat：
  作为负对照，验证 source-restricted 是否仍无效。

GLM4 target_seeded：
  先改进 prompt 或过滤策略，暂不作为因果主对象。
```

Phase218 应完成：

```text
1. 对 DS7B explain L24 head set 做 source group restricted ablation；
2. 比较阻断 answer_slot / instruction_to_answer / trigger:any 的效果；
3. 计算 head output 与 success-drift 方向的 cosine；
4. 同时记录 residual delta 和输出模式变化；
5. 如果 source-restricted 仍不能定位，则转向 MLP/StatePath。
```

当前进度估计：

```text
小模型模式机制图谱：约 60%
TriggerPath：约 38%
RoutePath：约 23%
RouteCause：约 8%
StatePath：约 35%
ReadoutPath：约 20%
路径因果机制：约 14%
模型内部自然闭合：约 30%
任务层产品闭合：约 55%
通用语言机制外推置信：约 34% 到 39%
```

## Phase 218: 源位置限制消融与写入方向验证 [2026-07-06 22:26]

### 0. 对附件判断的校准

Phase217 的判断基本正确，而且 Phase218 的任务方向也是正确的。

Phase217 已经把一个关键混淆拆开：

```text
RouteCandidate（路由候选） != RouteCause（路由因果）
HeadSetCandidate（头集合候选） != HeadSetCause（头集合因果）
```

Phase217 的 head set 消融中，qwen3 和 GLM4 的可复现候选没有稳定因果效应；DS7B 的 explain L24 head set 出现过弱必要性信号，但样本量和效应都偏弱。因此附件提出的 Phase218 是合理的：不能继续只看“哪些 head 注意到哪些 token”，而要看候选 head 是否真的从特定源位置读取并写入了会改变输出模式的状态。

本阶段继续同一阶段性目标：从模式候选推进到路径因果候选。任务仍属于 Phase209 到 Phase217 形成的“语言是动态模式网络”的机制图谱阶段，没有切换到最终理论闭合阶段。

### 1. 测试文件和结果文件

测试脚本：

```text
tests/gpt5/phase218_source_restricted_value_ablation.py
tests/gpt5/run_phase218_source_restricted_value_ablation.sh
```

结果目录：

```text
tests/result/phase218_source_restricted_value_ablation/source_restricted_value_ablation/
```

关键结果文件：

```text
phase218_cross_model_summary.json
phase218_cross_model_summary.md
phase218_qwen3_summary.json
phase218_glm4_summary.json
phase218_deepseek7b_summary.json
```

### 2. 算法原理

本阶段实现的是近似的 source-restricted value ablation（源位置限制 value 消融）。

由于不同模型的注意力实现和缓存结构不完全一致，直接修改 attention probability（注意力概率）容易在模型间失效。因此本阶段选择一个更稳定的跨模型接口：在候选层的 `v_proj` 输出处，对指定源 token 位置、指定 KV head 的 value 向量置零。

设某层第 \(l\) 层的 value 投影为：

$$
V_l = X_l W^V_l
$$

对候选 head set 中的 query head \(h\)，根据 GQA/MQA 的 head 分组映射到 KV head：

$$
k(h)=\left\lfloor \frac{h}{H_q / H_{kv}} \right\rfloor
$$

对源位置集合 \(S\) 进行限制消融：

$$
V'_{l,t,k(h)} =
\begin{cases}
0, & t \in S \\
V_{l,t,k(h)}, & t \notin S
\end{cases}
$$

然后比较 baseline（基线）输出和 patch（补丁/消融）输出：

$$
\Delta_{\text{damage}} =
\text{match}_{success}^{base}
-
\text{match}_{success}^{patch}
$$

$$
\Delta_{\text{repair}} =
\text{match}_{drift}^{patch}
-
\text{match}_{drift}^{base}
$$

本阶段测试的源位置组：

```text
answer_slot：答案槽位附近
instruction_to_answer：指令到答案之间的区域
trigger:any：Phase214/215 识别的触发 token 区域
question_prefix：问题前缀区域
```

测试对象：

```text
qwen3:
  qwen3_explain_route_set
  qwen3_repeat_route_set

GLM4:
  glm4_repeat_route_set
  glm4_explain_competition_route_set

DS7B:
  deepseek7b_explain_l24_route_set
  deepseek7b_list_l24_route_set
```

### 3. 客观测试结果

跨模型汇总：

```text
headset_count: 6
filter_rows: 79
reproducible_success_rows: 30
reproducible_drift_rows: 26
rollout_rows: 180
source_value_rows: 1920
effect_rows: 24
total_damage_match_loss: 0
total_repair_match_gain: 0
```

分模型结果：

```text
qwen3:
  filter_rows: 32
  reproducible_success_rows: 16
  reproducible_drift_rows: 12
  rollout_rows: 80
  source_value_rows: 1280
  damage: 0
  repair: 0

GLM4:
  filter_rows: 31
  reproducible_success_rows: 12
  reproducible_drift_rows: 10
  rollout_rows: 70
  source_value_rows: 448
  damage: 0
  repair: 0

DS7B:
  filter_rows: 16
  reproducible_success_rows: 2
  reproducible_drift_rows: 4
  rollout_rows: 30
  source_value_rows: 192
  damage: 0
  repair: 0
```

重要现象：

```text
1. qwen3 explain/repeat 的候选 head set 继续为负对照：
   阻断 answer_slot、instruction_to_answer、trigger:any、question_prefix 后，
   成功样本仍保持目标模式，漂移样本没有被修复。

2. GLM4 repeat/explain_competition 的候选 head set 也没有行为因果效应：
   所有源位置限制消融下 damage=0，repair=0。

3. DS7B explain L24 head set 在 Phase217 的 all-step head set 消融中曾有弱 damage，
   但本阶段只阻断候选源位置 value 后，damage 消失。

4. DS7B list L24 head set 没有可用成功样本保留，只能作为漂移侧观察；
   结果同样没有 repair。
```

### 4. 当前结论

本阶段得到一个重要负结果：

```text
Phase217 中的弱必要性 head set，
不能被进一步定位为“从 answer_slot / instruction_to_answer / trigger:any /
question_prefix 这些源位置读取 value 并造成输出模式”的因果路径。
```

这说明当前 RoutePath（路由路径）仍主要是相关图谱，不是闭合因果图谱。

更严格地说：

```text
候选 head 注意到某些源位置
  不等于
这些源位置的 value 写入决定了最终模式。

全 head 输出消融有弱效应
  不等于
源位置限制 value 消融可以定位原因。
```

这进一步支持 Phase217 的校准：RouteCandidate 和 HeadSetCandidate 还不能升级为 RouteCause。

### 5. 进展

本阶段推进了三个拼图：

```text
1. 方法上：
   建立了跨 qwen3、GLM4、DS7B 的源位置限制 value 消融框架。

2. 证据上：
   对 Phase217 中最值得追踪的 DS7B explain L24 候选做了更严格定位，
   结果没有定位出源位置因果。

3. 路线判断上：
   继续沿 RoutePath 打补丁的收益下降，
   下一步应转向 StatePath / MLP / residual write 的因果验证。
```

### 6. 问题、硬伤和限制

本阶段结果不能过度解释，主要硬伤如下：

```text
1. 本阶段是 value 投影处的近似源位置消融，
   不是直接在 attention probability 上做精确源位置屏蔽。

2. 对 GQA/MQA 模型，query head 到 KV head 的映射可能造成共享 KV head 被一起阻断；
   因此它既可能过度阻断，也可能无法精确对应单个 query head。

3. source group 仍然较粗：
   answer_slot、instruction_to_answer、trigger:any、question_prefix
   可能没有覆盖真正的因果源 token。

4. DS7B 可复现成功样本只有 2 条，
   所以 DS7B 的负结果方向有价值，但强度有限。

5. 当前小模型内部结构可能比真实强模型粗糙，
   对通用语言编码机制的外推仍可能有 30% 到 50% 偏差。

6. 本阶段只证明“当前源位置 value 阻断未定位到因果”，
   不能证明注意力完全无因果作用。
```

### 7. 对语言模式网络图谱的含义

Phase209 到 Phase218 的累计结果显示：

```text
语言模式确实可以在模型内部形成可观测分叉：
  repeat / list / explain / target_seeded 等模式有稳定轨迹差异。

触发 token 和注意力路由可以构成候选图谱：
  TriggerPath 和 RoutePath 能描述模式分布。

但是当前候选图谱还没有闭合到因果：
  阻断窗口方向、阻断候选 head、阻断候选源位置 value，
  都没有稳定修复 drift，也没有跨模型稳定 damage。
```

这对“语言是动态模式网络”的理论是一次收紧，而不是否定。

更接近当前事实的表述应为：

```text
语言任务中的模式不是单个语义向量，也不是单个概念神经元；
它更像由触发状态、候选路由、残差状态写入、读出竞争共同构成的动态模式网络。

当前已较清楚的是模式候选图谱；
尚未闭合的是模式因果路径。
```

### 8. 当前进度估计

只根据当前测试进展估计：

```text
小模型模式机制图谱：约 61%
TriggerPath：约 38%
RoutePath：约 25%
RouteCause：约 8%
SourcePath：约 10%
StatePath：约 35%
MLP/ResidualWrite：约 18%
ReadoutPath：约 20%
路径因果机制：约 14%
模型内部自然闭合：约 30%
任务层产品闭合：约 55%
通用语言机制外推置信：约 34% 到 39%
```

### 9. 下一阶段任务：Phase219

Phase219 不应继续优先扩大 head route 补丁，而应转向 StatePath / MLP / residual write。

建议阶段目标：

```text
Phase219: 模式状态写入路径与 MLP 因果验证
```

核心任务：

```text
1. 从 Phase209/214/215 的 success-drift 样本中，抽取每层 residual delta。

2. 在关键 token 窗口比较：
   success 模式状态
   drift 模式状态
   baseline 状态

3. 对 MLP 输出、attention 输出、residual stream 分别做 patch/ablation：
   不再只问“哪个 head 注意了哪里”，
   而是问“哪个模块把状态写成了目标模式”。

4. 先做必要性测试：
   success 状态中移除某层 MLP/attention/residual 写入，
   看目标模式是否坍塌。

5. 再做充分性测试：
   drift 状态中加入 success-drift 方向，
   看是否能把漂移样本拉回目标模式。

6. 如果 MLP/StatePath 能产生稳定 damage 或 repair，
   再回头解释 RoutePath 是读源、调度、还是旁路相关信号。
```

Phase219 的优先模型顺序：

```text
1. qwen3 explain/repeat：负对照和样本量较稳定。
2. GLM4 repeat/explain_competition：检查是否存在状态写入竞争。
3. DS7B explain：保留 L24 线索，但不再把它当作主因果假设。
```

阶段性判断：

```text
当前任务与下一任务仍处于同一阶段：
  完成语言模式网络的全局图谱和因果拼图。

可以继续自动推进：
  下一步应实现 Phase219 的 StatePath/MLP 写入因果脚本，
  而不是继续在 RoutePath 上做边际收益很低的 head/source patch。
```

## Phase 219: 模式状态写入路径与 MLP 因果验证 [2026-07-06 22:46]

### 0. 对附件判断的校准

附件对 Phase218 的判断基本正确。Phase218 不是普通失败，而是一次路线止损：它证明当前 RoutePath（路由路径）和 SourcePath（源位置路径）还不能升级为 RouteCause（路由因果）。

更准确地说：

```text
模型看回哪里
  不等于
这些信息已经被写入成决定输出模式的状态。
```

因此附件提出的 Phase219 方向是正确的：下一步不应继续优先做 head/source patch，而应转向 StatePath（状态路径）、MLP write（多层感知机写入）、residual write（残差写入）和 ReadoutPath（读出路径）。

本阶段仍属于同一个阶段性目标：完成语言模式网络的全局机制图谱和路径因果拼图。当前不是最终理论闭合阶段，而是从候选图谱进入因果图谱的关键阶段。

### 1. 测试文件和结果文件

新增测试脚本：

```text
tests/gpt5/phase219_state_write_mlp_causal_validation.py
tests/gpt5/run_phase219_state_write_mlp_causal_validation.sh
```

第一轮结果目录：

```text
tests/result/phase219_state_write_mlp_causal_validation/state_write_mlp_causal_validation/
```

扩大确认轮结果目录：

```text
tests/result/phase219_state_write_mlp_causal_validation/state_write_mlp_causal_validation_confirm/
```

关键结果文件：

```text
phase219_cross_model_summary.json
phase219_cross_model_summary.md
phase219_qwen3_summary.json
phase219_glm4_summary.json
phase219_deepseek7b_summary.json
phase219_*_effect_rows.jsonl
phase219_*_write_score_rows.jsonl
phase219_*_write_summary_rows.jsonl
```

### 2. 算法原理

本阶段目标是验证：

```text
到底是哪个模块把当前状态写成了目标输出模式？
```

不再优先问：

```text
哪个 attention head（注意力头）看了哪个 token？
```

本阶段对每个模型和模式选择少量高价值层，对可复现 success（成功）和 drift（漂移）样本计算每层每步的状态方向。

定义第 \(l\) 层、第 \(t\) 步的 success-drift 状态方向：

$$
v_{l,t}^{S-D}
=
\mathbb{E}\left[h_{l,t}\mid success\right]
-
\mathbb{E}\left[h_{l,t}\mid drift\right]
$$

模块写入分数：

$$
\mathrm{WriteScore}_{m,l,t}
=
\cos\left(
O_{m,l,t},
v_{l,t}^{S-D}
\right)
$$

其中：

```text
m = resid / mlp / attn
resid = residual stream（残差流）
mlp = MLP output（多层感知机输出）
attn = attention output（注意力输出）
```

行为干预分成四类：

```text
1. resid_add_Lx:
   在第 x 层残差状态加入 success-drift 方向。

2. resid_sub_Lx:
   在第 x 层残差状态减去 success-drift 方向。

3. mlp_zero_Lx:
   将第 x 层 MLP 在当前生成位置的输出置零。

4. attn_zero_Lx:
   将第 x 层 attention output 在当前生成位置的输出置零。
```

对应公式：

$$
h'_{l,t}
=
h_{l,t}
+
\lambda v_{l,t}^{S-D}
$$

$$
h'_{l,t}
=
h_{l,t}
-
\lambda v_{l,t}^{S-D}
$$

$$
O'_{mlp,l,t}=0
$$

$$
O'_{attn,l,t}=0
$$

评价仍使用：

$$
\Delta_{\mathrm{damage}}
=
\mathrm{match}_{success}^{base}
-
\mathrm{match}_{success}^{patch}
$$

$$
\Delta_{\mathrm{repair}}
=
\mathrm{match}_{drift}^{patch}
-
\mathrm{match}_{drift}^{base}
$$

### 3. 测试对象

本阶段选择对象：

```text
qwen3:
  qwen3_explain_state_write
  qwen3_repeat_state_write

GLM4:
  glm4_repeat_state_write
  glm4_explain_competition_state_write

DS7B:
  deepseek7b_explain_state_write
```

候选层来自前面 Phase213、Phase217 的高信号层：

```text
qwen3 explain: L11 / L29 / L31 / L33
qwen3 repeat: L29 / L31 / L32 / L33
GLM4 repeat/explain: L12 / L28 / L29 / L30
DS7B explain: L24 / L25 / L26 / L27
```

### 4. 第一轮客观结果

第一轮参数：

```text
max_filter_rows: 8
max_direction_rows: 6
max_eval_rows: 3
max_steps: 6
```

跨模型汇总：

```text
spec_count: 5
filter_rows: 71
reproducible_success_rows: 19
reproducible_drift_rows: 24
rollout_rows: 357
write_score_rows: 540
effect_rows: 64
write_summary_rows: 96
total_damage_match_loss: 32
total_repair_match_gain: 29
```

分模型：

```text
qwen3:
  filter_rows: 32
  reproducible_success_rows: 10
  reproducible_drift_rows: 12
  rollout_rows: 204
  write_score_rows: 288
  damage: 24
  repair: 19

GLM4:
  filter_rows: 31
  reproducible_success_rows: 9
  reproducible_drift_rows: 10
  rollout_rows: 153
  write_score_rows: 252
  damage: 8
  repair: 10

DS7B:
  filter_rows: 8
  reproducible_success_rows: 0
  reproducible_drift_rows: 2
  rollout_rows: 0
  write_score_rows: 0
  damage: 0
  repair: 0
```

第一轮已经出现与 Phase215 到 Phase218 完全不同的现象：

```text
RoutePath / SourcePath 干预基本为 0；
StatePath / residual direction 干预出现明显 damage 和 repair。
```

### 5. 扩大确认轮结果

由于第一轮出现重要正结果，按项目要求加大样本进行确认。

确认轮参数：

```text
max_filter_rows: 12
max_direction_rows: 8
max_eval_rows: 5
max_steps: 6
```

确认轮跨模型汇总：

```text
spec_count: 5
filter_rows: 99
reproducible_success_rows: 28
reproducible_drift_rows: 36
rollout_rows: 595
write_score_rows: 792
effect_rows: 64
write_summary_rows: 96
total_damage_match_loss: 66
total_repair_match_gain: 50
```

确认轮分模型：

```text
qwen3:
  filter_rows: 48
  rollout_rows: 340
  write_score_rows: 432
  damage: 48
  repair: 30

GLM4:
  filter_rows: 43
  rollout_rows: 255
  write_score_rows: 360
  damage: 18
  repair: 20

DS7B:
  filter_rows: 8
  reproducible_success_rows: 0
  reproducible_drift_rows: 2
  rollout_rows: 0
  write_score_rows: 0
```

确认轮关键现象：

```text
1. qwen3 explain:
   resid_add_L31 将 drift 从 0/5 修复到 5/5 explain_answer；
   resid_add_L29 将 drift 修复到 4/5；
   resid_sub_L29/L31/L33 将 success 从 5/5 破坏到 0/5。

2. qwen3 repeat:
   resid_sub_L31 将 repeat success 从 5/5 破坏到 0/5；
   resid_sub_L33 也将 repeat success 从 5/5 破坏到 0/5。

3. GLM4 repeat:
   resid_sub_L28 将 repeat success 从 5/5 破坏到 0/5；
   resid_sub_L29/L30 也有明显 damage。

4. GLM4 explain_competition:
   成功样本只有 1 条，权重较低；
   但 drift 样本中 resid_add_L28/L29/L30 可将 4/4 修复成 explain_answer。

5. DS7B:
   explain 任务没有可复现成功样本，不能参与本阶段行为因果判断。
```

### 6. WriteScore 结果

确认轮最高的 write score（写入分数）集中在 residual stream（残差流），其次是部分 MLP 和 attention output：

```text
qwen3 explain drift:
  L33 resid: cosine approx -0.387
  L31 resid: cosine approx -0.365
  L29 resid: cosine approx -0.331

GLM4 repeat success:
  L12 resid: cosine approx 0.361
  L28 resid: cosine approx 0.315
  L30 mlp: cosine approx 0.302
  L30 resid: cosine approx 0.301

qwen3 repeat success:
  L32 resid: cosine approx 0.300
  L29 resid: cosine approx 0.289
  L31 resid: cosine approx 0.283
  L32 mlp: cosine approx 0.264
```

这说明：

```text
模式成功/漂移的方向在 residual stream 中最明显；
MLP 有可见写入关系，但当前第一轮还没有证明 MLP 是唯一或主因；
attention output 有局部作用，但不如 residual direction 稳定。
```

### 7. 当前结论

Phase219 是 Phase209 以来最重要的正结果之一。

当前可以谨慎升级：

```text
StatePath（状态路径）从候选图谱升级为初步因果路径。
```

更精确的结论：

```text
1. 对 qwen3 explain，success-drift residual direction 具有强充分性：
   加入方向可将 drift 修复成 explain_answer。

2. 对 qwen3 explain/repeat 和 GLM4 repeat，success-drift residual direction 具有强必要性：
   减去方向可破坏原本成功的输出模式。

3. MLP output 与状态方向有明显相关和部分行为效应，
   但当前不能证明 MLP 是唯一源头。

4. attention output 也有局部修复作用，
   但它更像参与状态写入或竞争路径，而不是前几轮设想中的独立 RouteCause。
```

与 Phase218 的组合结论是：

```text
RoutePath 是候选调度层；
StatePath / residual write 更接近当前模式因果层。
```

### 8. 重要问题和硬伤

本阶段虽是正结果，但仍不能过度总结。

主要硬伤：

```text
1. residual direction patch 是人工加入方向，
   证明的是状态方向具备行为因果作用，
   还没有证明模型自然运行时具体由哪个模块生成该方向。

2. MLP zero 和 attention zero 是粗粒度模块消融，
   可能破坏多个功能，不是精细通道级因果。

3. qwen3 结果最强，GLM4 次之；
   DS7B 因没有可复现成功样本，本阶段无法验证。

4. GLM4 explain_competition 的 success 样本只有 1 条，
   其中 repair 信号需要低权重处理。

5. 当前方向按 success-drift 均值构造，
   可能包含多个混合因素：答案内容、输出格式、解释模式、长度倾向、停止倾向。

6. max_steps 只有 6，
   对长解释、列表和停止闭合仍不足。

7. 当前小模型结构粗糙，
   对真实语言编码机制外推仍需 30% 到 50% 折扣。
```

### 9. 对全局图谱的更新

当前全局机制图谱应更新为：

```text
PromptTrigger（提示触发）
→ CandidateRoutePath（候选路由）
→ StateWritePath（状态写入）
→ StateMaintainPath（状态维持）
→ ReadoutCompetition（读出竞争）
→ OutputPattern（输出模式）
→ ClosureOrDrift（闭合或漂移）
```

Phase219 后，权重应调整：

```text
TriggerPath:
  仍是模式启动层。

RoutePath:
  降级为候选调度层和相关图谱。

StatePath:
  升级为当前最接近行为因果的路径。

MLP/Attention write:
  是 StatePath 的候选生成模块，需要下一阶段细化。

ReadoutPath:
  仍未闭合，但已经可以被 residual direction 明显影响。
```

### 10. 统一机制公式更新

保持理论名词不变，但把状态写入放到更核心位置。

当前模式因果公式：

$$
\mathrm{PatternCause}
=
\mathrm{TriggerPath}
\circ
\mathrm{CandidateRoutePath}
\circ
\mathrm{StateWritePath}
\circ
\mathrm{StateMaintainPath}
\circ
\mathrm{ReadoutPath}
$$

StateWriteCause（状态写入因果）暂定为：

$$
\mathrm{StateWriteCause}_{m,l,t}
=
\mathrm{ReproducibleBaseline}
\land
\mathrm{StateDifference}_{l,t}
\land
\mathrm{WriteScore}_{m,l,t}
\land
\mathrm{BehaviorEffect}_{m,l,t}
$$

本阶段已经初步满足：

```text
ReproducibleBaseline: qwen3/GLM4 部分成立
StateDifference: 成立
BehaviorEffect: qwen3/GLM4 明显成立
WriteScore: residual 强，MLP/attention 部分成立
```

但还未完全满足：

```text
ModuleWriteEffect 的自然来源定位。
```

因此不能说完整闭合，只能说 StatePath 进入强候选因果层。

### 11. 当前进度估计

只根据当前测试进展估计：

```text
小模型模式机制图谱：约 65%
TriggerPath：约 38%
RoutePath：约 25%
RouteCause：约 8%
SourcePath：约 10%
StatePath：约 48%
StateWriteCause：约 28%
MLP/ResidualWrite：约 32%
ReadoutPath：约 23%
路径因果机制：约 25%
模型内部自然闭合：约 33%
任务层产品闭合：约 55%
通用语言机制外推置信：约 38% 到 43%
```

### 12. 下一阶段任务：Phase220

Phase219 和下一步仍处于同一个阶段性目标：完成语言模式网络的全局图谱和因果拼图。

下一步不应立刻理论收束，而应继续客观拼图。

建议 Phase220：

```text
Phase220: StateWrite 来源定位与 MLP/Attention 精细因果分解
```

核心任务：

```text
1. 对 qwen3 explain 的 L29/L31/L33 residual direction 做精细定位：
   分解该方向来自 MLP、attention output、还是前层 residual accumulation。

2. 对 qwen3 repeat 的 L31/L33 residual_sub 强 damage 做复测：
   判断 repeat 模式是否依赖同一类状态方向。

3. 对 GLM4 repeat 的 L28/L29/L30 做确认：
   检查是否是跨模型的 repeat StatePath。

4. 对 MLP 做更细粒度通道级测试：
   找出 top channels，再做 channel zero / channel boost。

5. 对 residual direction 做 scale sweep：
   测试 0.25 / 0.5 / 1.0 / 1.5 / 2.0，
   判断是否存在连续剂量效应。

6. 对 direction 做因子拆分：
   区分答案内容方向、解释格式方向、复读格式方向、长度/停止方向。
```

Phase220 的优先级：

```text
第一优先级：
  qwen3 explain L29/L31/L33 residual direction。

第二优先级：
  qwen3 repeat L31/L33 和 GLM4 repeat L28/L29/L30。

第三优先级：
  DS7B 需要先改造 prompt 或样本选择，让 explain success 可复现，否则暂不作为主因果对象。
```

阶段性判断：

```text
可以继续自动推进 Phase220。
当前最重要的问题已经从“模式是否有状态方向”变成：
  这个状态方向由哪些自然模块写入？
```

## Phase 220: StateWrite 来源定位与 MLP/Attention 精细因果分解 [2026-07-06 23:24]

### 0. 对附件判断的校准

附件对 Phase219 的判断基本正确。Phase219 是 Phase209 以来最重要的正结果之一，因为它第一次把模式机制从：

```text
候选图谱 / 路由相关
```

推进到：

```text
可干预的 residual state direction（残差状态方向）
```

但附件的谨慎意见也正确：Phase219 证明了 residual direction（残差方向）具有行为因果作用，却没有证明这个方向在模型自然运行中由哪个模块写入。

因此 Phase220 的任务是合理的：

```text
不再问“状态方向是否有效”，
而是问“状态方向是谁写进去的”。
```

当前任务与 Phase219 处于同一阶段性目标：完成语言模式网络的全局机制图谱和路径因果拼图，尚未进入最终理论闭合阶段。

### 1. 新增测试文件和结果文件

新增脚本：

```text
tests/gpt5/phase220_state_write_source_decomposition.py
tests/gpt5/run_phase220_state_write_source_decomposition.sh
```

第一轮结果目录：

```text
tests/result/phase220_state_write_source_decomposition/state_write_source_decomposition/
```

扩大确认轮结果目录：

```text
tests/result/phase220_state_write_source_decomposition/state_write_source_decomposition_confirm/
```

关键结果文件：

```text
phase220_cross_model_summary.json
phase220_cross_model_summary.md
phase220_*_effect_rows.jsonl
phase220_*_source_alignment_rows.jsonl
phase220_*_source_summary_rows.jsonl
```

### 2. 测试目标

Phase220 做三件事：

```text
1. residual direction 剂量扫描：
   测试 StatePath 是否存在连续剂量/阈值效应。

2. 模块来源对齐：
   比较 MLP/attention 的 success-drift 模块方向与 residual direction 的对齐程度。

3. 模块方向干预：
   测试移除模块输出在 residual direction 上的投影，或加入模块 success-drift 方向，
   是否能复制 residual direction 的行为效应。
```

### 3. 算法公式

残差方向仍定义为：

$$
v_{l,t}^{S-D}
=
\mathbb{E}\left[h_{l,t}\mid success\right]
-
\mathbb{E}\left[h_{l,t}\mid drift\right]
$$

模块方向定义为：

$$
u_{m,l,t}^{S-D}
=
\mathbb{E}\left[O_{m,l,t}\mid success\right]
-
\mathbb{E}\left[O_{m,l,t}\mid drift\right]
$$

模块来源对齐分数：

$$
\mathrm{SourceAlign}_{m,l,t}
=
\cos\left(
u_{m,l,t}^{S-D},
v_{l,t}^{S-D}
\right)
$$

残差剂量扫描：

$$
h'_{l,t}
=
h_{l,t}
+
\lambda v_{l,t}^{S-D}
$$

$$
h'_{l,t}
=
h_{l,t}
-
\lambda v_{l,t}^{S-D}
$$

本阶段使用：

```text
lambda = 0.25 / 0.5 / 1.0 / 1.5 / 2.0
```

模块投影移除：

$$
O'_{m,l,t}
=
O_{m,l,t}
-
\left(
O_{m,l,t}\cdot \hat{v}_{l,t}^{S-D}
\right)
\hat{v}_{l,t}^{S-D}
$$

模块方向加入：

$$
O'_{m,l,t}
=
O_{m,l,t}
+
u_{m,l,t}^{S-D}
$$

其中：

```text
m = MLP 或 attention output
```

### 4. 测试对象

本阶段优先对象：

```text
qwen3 explain:
  L29 / L31 / L33
  scale layer: L31

qwen3 repeat:
  L31 / L33
  scale layer: L31

GLM4 repeat:
  L28 / L29 / L30
  scale layer: L28

DS7B explain:
  L24
  仍作为低权重观察对象
```

### 5. 第一轮客观结果

第一轮参数：

```text
max_filter_rows: 12
max_direction_rows: 8
max_eval_rows: 4
max_source_steps: 3
max_steps: 6
```

跨模型汇总：

```text
spec_count: 4
filter_rows: 80
reproducible_success_rows: 27
reproducible_drift_rows: 32
rollout_rows: 520
source_alignment_rows: 48
effect_rows: 62
source_summary_rows: 16
total_damage_match_loss: 52
total_repair_match_gain: 12
```

分模型：

```text
qwen3:
  filter_rows: 48
  reproducible_success_rows: 17
  reproducible_drift_rows: 18
  rollout_rows: 336
  source_alignment_rows: 30
  damage: 34
  repair: 12

GLM4:
  filter_rows: 24
  reproducible_success_rows: 10
  reproducible_drift_rows: 12
  rollout_rows: 184
  source_alignment_rows: 18
  damage: 18
  repair: 0

DS7B:
  filter_rows: 8
  reproducible_success_rows: 0
  reproducible_drift_rows: 2
  rollout_rows: 0
  source_alignment_rows: 0
```

### 6. 扩大确认轮结果

因为第一轮继续出现强行为效应和稳定 MLP 对齐，所以增加评估样本与 source steps。

确认轮参数：

```text
max_filter_rows: 12
max_direction_rows: 10
max_eval_rows: 5
max_source_steps: 4
max_steps: 6
```

确认轮跨模型汇总：

```text
spec_count: 4
filter_rows: 80
reproducible_success_rows: 27
reproducible_drift_rows: 32
rollout_rows: 650
source_alignment_rows: 64
effect_rows: 62
source_summary_rows: 16
total_damage_match_loss: 64
total_repair_match_gain: 16
```

确认轮分模型：

```text
qwen3:
  rollout_rows: 420
  source_alignment_rows: 40
  damage: 43
  repair: 16

GLM4:
  rollout_rows: 230
  source_alignment_rows: 24
  damage: 21
  repair: 0

DS7B:
  仍无可复现 explain success，不能参与本阶段因果判断。
```

### 7. 剂量扫描结果

qwen3 explain L31 的 residual add 呈现明显阈值/剂量效应：

```text
resid_add_L31_s0.25:
  drift repair = 0/5

resid_add_L31_s0.5:
  drift repair = 4/5

resid_add_L31_s1.0:
  drift repair = 5/5

resid_add_L31_s1.5:
  drift repair = 1/5，并出现 echo_then_answer 增加

resid_add_L31_s2.0:
  drift repair = 5/5，但 success damage = 3/5
```

这说明 qwen3 explain 的 residual direction 不是单点偶然补丁，而是有可观察的剂量响应；但剂量过大时也会破坏原有成功轨迹或引入格式漂移。

qwen3 explain L31 的 residual sub：

```text
scale 0.25: success damage = 0/5
scale 0.5: success damage = 0/5
scale 1.0: success damage = 5/5
scale 1.5: success damage = 5/5
scale 2.0: success damage = 5/5
```

qwen3 repeat L31 的 residual sub：

```text
scale 0.25: success damage = 1/5
scale 0.5: success damage = 1/5
scale 1.0: success damage = 5/5
scale 1.5: success damage = 5/5
scale 2.0: success damage = 5/5
```

GLM4 repeat L28 的 residual sub：

```text
scale 0.25: success damage = 1/5
scale 0.5: success damage = 3/5
scale 1.0: success damage = 5/5
scale 1.5: success damage = 5/5
scale 2.0: success damage = 5/5
```

结论：

```text
residual direction 存在稳定阈值效应；
大约在 scale 1.0 附近进入强行为控制区。
```

### 8. 模块来源对齐结果

确认轮中，MLP 与 residual direction 的对齐显著高于 attention。

最高来源对齐：

```text
GLM4 repeat L28 MLP:
  cosine approx 0.497
  norm_ratio approx 0.289

GLM4 repeat L30 MLP:
  cosine approx 0.468
  norm_ratio approx 0.284

qwen3 explain L29 MLP:
  cosine approx 0.452
  norm_ratio approx 0.415

qwen3 repeat L31 MLP:
  cosine approx 0.430
  norm_ratio approx 0.343

qwen3 explain L31 MLP:
  cosine approx 0.430
  norm_ratio approx 0.376

qwen3 explain L29 attention:
  cosine approx 0.409
  norm_ratio approx 0.262

qwen3 explain L33 MLP:
  cosine approx 0.391
  norm_ratio approx 0.427
```

整体观察：

```text
1. MLP 的 module-to-residual alignment 普遍高于 attention。
2. MLP 的 norm_ratio 大约在 0.26 到 0.43 之间，说明单层 MLP 方向能解释 residual direction 的一部分，但不是全部。
3. attention 也有局部对齐，尤其 qwen3 explain L29 attention，但整体弱于 MLP。
```

### 9. 模块干预结果

模块投影移除和模块方向加入没有完整复制 residual direction 的强效应。

典型结果：

```text
qwen3 explain:
  mlp_proj_remove_L29/L31/L33: damage=0, repair=0
  attn_proj_remove_L29/L31/L33: damage=0, repair=0
  mlp_sdm_add_L33: damage=2/5, repair=0

GLM4 repeat:
  mlp_proj_remove_L28: damage=1/5
  mlp_proj_remove_L30: damage=1/5
  repair=0
```

这说明：

```text
MLP/attention 模块方向与 residual direction 对齐，
但单步、单模块、粗投影级干预还不能完全解释 residual 状态方向。
```

换句话说：

```text
residual direction 更像多层累积状态；
MLP 是强候选写入源；
attention 是局部参与者；
但自然 StateWriteSource 还没有闭合。
```

### 10. 当前结论

Phase220 的主要结论：

```text
1. Phase219 的 residual direction 行为因果被再次确认。

2. residual direction 不是随机单点补丁，而是存在明确阈值/剂量效应。

3. MLP 与 residual direction 的对齐稳定高于 attention，
   因此 MLP 应升级为 StateWriteSource 的第一候选。

4. 但模块投影移除和模块方向加入没有完全复制 residual direction 的行为效应，
   说明状态方向更可能来自多层 MLP + residual accumulation，而不是单层单模块。

5. DS7B 继续由于缺少可复现 success 样本而无法参与该阶段判断。
```

严谨表述：

```text
StatePath 已有强行为因果；
MLP 是 StateWriteSource 的强候选；
但 StateWriteSource 还未闭合。
```

### 11. 问题、硬伤和限制

本阶段仍有重要限制：

```text
1. 模块投影移除只移除模块输出在 residual direction 上的线性投影，
   不能覆盖非线性门控、LayerNorm 变化、后续层放大等机制。

2. module_sdm_add 使用模块 success-drift 均值方向，
   它可能比真正自然写入更粗糙，也可能没有正确的上下文条件门控。

3. MLP source alignment 高，只说明它方向相近，
   不能证明它独立生成完整 residual direction。

4. residual direction 可能混合解释格式、答案内容、长度、停止和漂移抑制等多个因子。

5. qwen3 explain 的 resid_add 在 scale 1.5/2.0 时会引入 success damage 或 echo drift，
   说明方向不是纯解释模式因子。

6. 当前仍以对象-关系-值模式族为主，通用语言机制外推需要谨慎。

7. 小模型内部结构可能较粗糙，对真实强模型语言编码机制仍需 30% 到 50% 折扣。
```

### 12. 对全局图谱的更新

当前图谱可以进一步细化为：

```text
PromptTrigger
→ CandidateRoutePath
→ MLP-dominant StateWriteSource
→ ResidualAccumulation
→ StateMaintainPath
→ ReadoutCompetition
→ OutputPattern
```

更细的候选机制：

```text
MLP 写入一部分模式方向；
attention 提供局部上下文调度/补充写入；
residual stream 累积多层方向；
读出层将累积状态转换成 explain/repeat 等输出模式。
```

### 13. 当前进度估计

只根据当前测试进展估计：

```text
小模型模式机制图谱：约 67%
TriggerPath：约 38%
CandidateRoutePath：约 25%
RouteCause：约 8%
SourcePath：约 10%
StatePath：约 52%
StateWriteCause：约 34%
StateWriteSource：约 22%
MLP/ResidualWrite：约 38%
ReadoutPath：约 24%
路径因果机制：约 29%
模型内部自然闭合：约 35%
任务层产品闭合：约 55%
通用语言机制外推置信：约 39% 到 44%
```

### 14. 下一阶段任务：Phase221

Phase221 仍属于同一个阶段性目标，应继续自动推进。

建议阶段标题：

```text
Phase221: MLP 通道级 StateWriteSource 定位与因子拆分
```

核心任务：

```text
1. 对 qwen3 explain L29/L31/L33 MLP 做 channel-level scan：
   找出对 residual direction 投影贡献最大的通道。

2. 对 GLM4 repeat L28/L30 MLP 做 channel-level scan：
   检查 repeat 模式是否存在跨模型 MLP 通道写入结构。

3. 对 top channels 做 channel zero / channel boost：
   验证它们是否能产生 damage 或 repair。

4. 做方向因子拆分：
   将 residual direction 分成 content / format / drift-control / length-stop。

5. 将 Phase220 的 scale sweep 作为剂量基准：
   对通道补丁也做 0.25 / 0.5 / 1.0 / 1.5 / 2.0。
```

优先级：

```text
第一优先级：
  qwen3 explain L29/L31 MLP channel scan。

第二优先级：
  GLM4 repeat L28/L30 MLP channel scan。

第三优先级：
  qwen3 repeat L31 MLP channel scan。

DS7B:
  先解决可复现 success 样本问题，否则暂不作为主线。
```

## Phase 221: MLP 通道级 StateWriteSource 定位与因子拆分初测 [2026-07-07 00:02]

### 0. 对附件判断的校准

附件对 Phase220 的判断基本正确。

Phase219 证明：

```text
success-drift residual direction 具有行为因果作用。
```

Phase220 进一步证明：

```text
这个 residual direction 存在剂量/阈值效应；
MLP 比 attention 更像 StateWriteSource 的第一候选；
但单层模块方向干预不能完整复制 residual direction 的强效应。
```

因此 Phase221 的方向是合理的：继续向 MLP channel（多层感知机通道）级下钻，测试是否存在更精细的 StateWriteSource 候选。

需要强调：本阶段仍属于同一阶段性目标，即完成语言模式网络的全局图谱和路径因果拼图。当前不应理论收束，而应继续积累客观拼图。

### 1. 新增测试文件和结果文件

新增脚本：

```text
tests/gpt5/phase221_mlp_channel_statewrite_source.py
tests/gpt5/run_phase221_mlp_channel_statewrite_source.sh
```

第一轮结果目录：

```text
tests/result/phase221_mlp_channel_statewrite_source/mlp_channel_statewrite_source/
```

扩大确认轮结果目录：

```text
tests/result/phase221_mlp_channel_statewrite_source/mlp_channel_statewrite_source_confirm/
```

关键结果文件：

```text
phase221_cross_model_summary.json
phase221_cross_model_summary.md
phase221_*_channel_score_rows.jsonl
phase221_*_effect_rows.jsonl
phase221_*_rollout_rows.jsonl
```

### 2. 算法原理

Phase221 的核心问题：

```text
如果 MLP 是 StateWriteSource 的第一候选，
那么哪些 MLP channel 对 residual direction 的写入贡献最大？
```

对 MLP down projection（下投影）输入通道 \(c\)，设：

$$
z_{l,t,c}
$$

为第 \(l\) 层、第 \(t\) 步 MLP down-proj 输入的第 \(c\) 个通道。

success-drift 通道差分：

$$
\Delta z_{l,t,c}
=
\mathbb{E}\left[z_{l,t,c}\mid success\right]
-
\mathbb{E}\left[z_{l,t,c}\mid drift\right]
$$

down projection 中该通道的输出方向：

$$
W^{down}_{l,:,c}
$$

通道对 residual direction 的写入分数：

$$
\mathrm{ChannelWriteScore}_{l,t,c}
=
\left|
\Delta z_{l,t,c}
\cdot
\left(
W^{down}_{l,:,c}
\cdot
\hat{v}_{l,t}^{S-D}
\right)
\right|
$$

其中：

```text
v_{l,t}^{S-D} = success-drift residual direction；
W_down column = 单个 MLP channel 写入 residual stream 的方向；
ChannelWriteScore = 通道激活差分乘以该通道输出方向与 residual direction 的对齐。
```

然后选择 top channels，测试：

```text
mlpchan_zero:
  将 top K 通道在当前生成位置置零。

mlpchan_boost:
  将 top K 通道按 success-drift Δz 方向增强。
```

测试 \(K\)：

```text
K = 4 / 16 / 64
```

### 3. 测试对象

本阶段测试：

```text
qwen3:
  qwen3_explain_l29_l31_mlp_channels
  qwen3_repeat_l31_mlp_channels

GLM4:
  glm4_repeat_l28_l30_mlp_channels

DS7B:
  deepseek7b_explain_l24_mlp_channels
```

### 4. 第一轮客观结果

第一轮参数：

```text
max_filter_rows: 12
max_direction_rows: 8
max_eval_rows: 4
max_channel_steps: 3
top_channels: 64
```

跨模型汇总：

```text
spec_count: 4
filter_rows: 80
reproducible_success_rows: 27
reproducible_drift_rows: 32
rollout_rows: 264
channel_score_rows: 960
effect_rows: 30
channel_summary_rows: 5
total_damage_match_loss: 6
total_repair_match_gain: 8
```

分模型：

```text
qwen3:
  rollout_rows: 160
  channel_score_rows: 576
  damage: 0
  repair: 8

GLM4:
  rollout_rows: 104
  channel_score_rows: 384
  damage: 6
  repair: 0

DS7B:
  无可复现 explain success，未进入通道因果测试。
```

第一轮主要效应：

```text
qwen3 explain L29:
  mlpchan_zero_L29_K16 repair = 3/4
  mlpchan_zero_L29_K64 repair = 3/4

GLM4 repeat L30:
  mlpchan_zero_L30_K4 damage = 2/4
  mlpchan_zero_L30_K16 damage = 2/4
  mlpchan_zero_L30_K64 damage = 2/4

qwen3 explain L31:
  mlpchan_boost_L31_K16 repair = 1/4
  mlpchan_boost_L31_K64 repair = 1/4
```

### 5. 扩大确认轮结果

确认轮参数：

```text
max_filter_rows: 12
max_direction_rows: 10
max_eval_rows: 5
max_channel_steps: 4
top_channels: 96
```

确认轮跨模型汇总：

```text
spec_count: 4
filter_rows: 80
reproducible_success_rows: 27
reproducible_drift_rows: 32
rollout_rows: 330
channel_score_rows: 1920
effect_rows: 30
channel_summary_rows: 5
total_damage_match_loss: 6
total_repair_match_gain: 10
```

确认轮主要效应：

```text
qwen3 explain L29:
  mlpchan_zero_L29_K16 repair = 4/5
  mlpchan_zero_L29_K64 repair = 4/5
  success 保持 5/5，不产生 damage。

GLM4 repeat L30:
  mlpchan_zero_L30_K4 damage = 2/5
  mlpchan_zero_L30_K16 damage = 2/5
  mlpchan_zero_L30_K64 damage = 2/5

qwen3 explain L31:
  mlpchan_boost_L31_K16 repair = 1/5
  mlpchan_boost_L31_K64 repair = 1/5
```

### 6. 通道候选

确认轮中重复出现的高分通道包括：

```text
qwen3 explain L29:
  channel 6627
  channel 5880
  channel 1070
  channel 4057
  channel 199
  channel 5347

qwen3 explain L31:
  channel 580
  channel 1735
  channel 4800
  channel 9219
  channel 8384
  channel 2779

qwen3 repeat L31:
  channel 6567
  channel 9407
  channel 4350
  channel 3298
  channel 9219

GLM4 repeat L28:
  channel 12792
  channel 742
  channel 13262
  channel 5867
  channel 1260

GLM4 repeat L30:
  channel 7088
  channel 9374
  channel 9892
  channel 670
  channel 5760
  channel 6118
```

这些不是单神经元机制证明，只是 MLP channel-level StateWriteSource 候选。

### 7. 当前结论

Phase221 得到的是弱正结果，而不是强闭合。

主要结论：

```text
1. MLP channel 组能产生部分行为效应。

2. qwen3 explain L29 top channel zero 可以修复 drift，
   但这更像移除了竞争/漂移写入，而不是直接加入 explain 写入。

3. GLM4 repeat L30 top channel zero 可以破坏 repeat success，
   说明该层 top channels 对 repeat 模式有部分必要性。

4. qwen3 explain L31 top channel boost 只有很弱 repair，
   说明简单通道增强不能复制 residual direction 的强充分性。

5. qwen3 repeat L31 通道组虽然有高 ChannelWriteScore，
   但 zero/boost 行为效应基本为 0，提示 repeat 的必要性可能来自更分布式 residual accumulation。
```

严谨表述：

```text
MLP channel 是 StateWriteSource 的局部候选；
但单层 top channel 组只能解释 residual direction 的一部分行为效应；
StateWriteSource 仍未闭合。
```

### 8. 问题和硬伤

本阶段限制很明显：

```text
1. 通道选择基于线性 down-proj 贡献分数，
   不能覆盖 MLP 内部门控、非线性和 LayerNorm 影响。

2. top channel zero 有时修复 drift，
   说明这些通道可能支持竞争模式或漂移抑制失败，
   不能简单解释成“目标模式写入通道”。

3. top channel boost 效果弱，
   说明通道 Δz 缺少上下文门控或多层协同。

4. K=4/16/64 效果差异不大，
   说明当前 topK 不是非常干净的稀疏机制，也可能是通道排序还不够准确。

5. DS7B 继续缺少可复现 success，无法参与判断。

6. 当前测试仍集中在对象-关系-值模式族，外推到完整语言机制仍要谨慎。

7. 小模型结构可能粗糙，对真实强模型编码机制仍需 30% 到 50% 折扣。
```

### 9. 对全局图谱的更新

当前图谱不应改成“单通道机制”，而应更新为：

```text
PromptTrigger
→ CandidateRoutePath
→ distributed MLP channel groups
→ ResidualAccumulation
→ StateMaintainPath
→ ReadoutCompetition
→ OutputPattern
```

更具体地说：

```text
MLP channel groups 提供局部写入或竞争抑制；
residual stream 才是强行为控制载体；
StateWriteSource 可能是多层、多通道、带门控的组合机制。
```

### 10. 当前进度估计

只根据当前测试进展估计：

```text
小模型模式机制图谱：约 68%
TriggerPath：约 38%
CandidateRoutePath：约 25%
RouteCause：约 8%
SourcePath：约 10%
StatePath：约 53%
StateWriteCause：约 35%
StateWriteSource：约 25%
MLP/ResidualWrite：约 40%
MLP channel source：约 14%
ReadoutPath：约 24%
路径因果机制：约 30%
模型内部自然闭合：约 35%
任务层产品闭合：约 55%
通用语言机制外推置信：约 39% 到 44%
```

### 11. 下一阶段任务：Phase222

Phase222 仍属于同一阶段，应继续客观拼图。

建议阶段标题：

```text
Phase222: StateWrite 因子拆分与竞争通道分离
```

核心任务：

```text
1. 将 qwen3 explain 的 residual direction 拆分为：
   explain-format 因子、answer-content 因子、echo/competition 因子、length/stop 因子。

2. 对 qwen3 explain L29 的 top zero-repair channels 做竞争通道分析：
   判断它们是在写入 explain，还是在写入 other_or_wrong/echo 竞争模式。

3. 对 GLM4 repeat L30 的 top damage channels 做必要性复测：
   分离 repeat-format 与 next_task_or_format 竞争。

4. 做正负通道分组：
   positive target channels；
   negative competitor channels；
   mixed channels。

5. 不再只看总 repair/damage，
   还要看 output_pattern 从哪一类漂移变到哪一类目标或竞争模式。
```

当前最关键的新问题：

```text
MLP 通道到底是在写目标模式，
还是在抑制竞争模式？
```

## Phase 222: StateWrite 因子拆分与竞争通道分离 [2026-07-07 00:21]

### 1. 本阶段任务

本阶段继续 Phase221 的同一阶段目标：分析 Phase221 关于 MLP channel（多层感知机通道）结果的判断是否正确，并继续完成 StateWriteSource（状态写入来源）的客观拼图。

Phase221 的判断基本正确：

```text
MLP channel group（多层感知机通道组）已经能产生局部因果效应；
但 top channel（高分通道）不能直接解释成目标模式写入通道。
```

Phase222 的改进是把 Phase221 的绝对值通道排序改成 signed channel split（带符号通道拆分），分别测试 positive channel group（正向通道组）和 negative channel group（负向通道组）的 zero（置零）与 boost（增强）效应。

### 2. 脚本与结果

新增脚本：

```text
tests/gpt5/phase222_statewrite_factor_competition.py
tests/gpt5/run_phase222_statewrite_factor_competition.sh
```

结果目录：

```text
tests/result/phase222_statewrite_factor_competition/statewrite_factor_competition/
```

已完成检查：

```text
python -m py_compile tests/gpt5/phase222_statewrite_factor_competition.py
bash -n tests/gpt5/run_phase222_statewrite_factor_competition.sh
```

已按 qwen3、GLM4、DS7B 顺序加载本地 CUDA 模型测试，每个模型测试后释放显存。

### 3. 算法原理和公式

成功轨迹和漂移轨迹之间的通道激活差分为：

$$
\Delta z_{l,t,c}
=
\mathbb{E}[z_{l,t,c}\mid success]
-
\mathbb{E}[z_{l,t,c}\mid drift]
$$

成功-漂移 residual direction（残差方向）单位向量为：

$$
\hat v_{l,t}^{S-D}
=
\frac{
v_{l,t}^{S-D}
}{
\left\|v_{l,t}^{S-D}\right\|
}
$$

带符号通道写入分数为：

$$
\boxed{
\mathrm{SignedChannelScore}_{l,t,c}
=
\Delta z_{l,t,c}
\cdot
\left(
W^{down}_{l,:,c}
\cdot
\hat v_{l,t}^{S-D}
\right)
}
$$

正向候选通道和负向候选通道分别为：

$$
\boxed{
C^{+}_{l,t,K}
=
\operatorname{TopK}_{c}
\left(
\mathrm{SignedChannelScore}_{l,t,c}
\right)
}
$$

$$
\boxed{
C^{-}_{l,t,K}
=
\operatorname{TopK}_{c}
\left(
-
\mathrm{SignedChannelScore}_{l,t,c}
\right)
}
$$

置零干预：

$$
\boxed{
z'_{l,t,c}=0,
\quad c\in C^{+}_{l,t,K}\ \text{or}\ C^{-}_{l,t,K}
}
$$

增强干预：

$$
\boxed{
z'_{l,t,c}
=
z_{l,t,c}
\alpha\Delta z_{l,t,c},
\quad c\in C^{+}_{l,t,K}\ \text{or}\ C^{-}_{l,t,K}
}
$$

本轮参数：

```text
alpha = 1.0
K = 4, 16, 64
max_eval_rows = 5
max_channel_steps = 4
max_steps = 6
top_channels = 96
```

本阶段还新增 output_pattern transition（输出模式转移）记录，不只看总 damage/repair（破坏/修复）。

### 4. 总体结果

跨模型汇总：

```text
spec_count = 4
filter_rows = 80
reproducible_success_rows = 27
reproducible_drift_rows = 32
rollout_rows = 630
channel_score_rows = 3840
total_damage_match_loss = 6
total_repair_match_gain = 10
```

分模型结果：

```text
qwen3:
  rollout_rows = 380
  channel_score_rows = 2304
  damage = 0
  repair = 10

GLM4:
  rollout_rows = 250
  channel_score_rows = 1536
  damage = 6
  repair = 0

DS7B:
  reproducible_success_rows = 0
  rollout_rows = 0
  channel_score_rows = 0
```

DS7B 仍缺少可复现 success（成功样本），所以本阶段不把 DS7B 纳入通道因果判断。

### 5. 关键客观结果

#### 5.1 qwen3 explain L29 positive zero 修复 drift

```text
qwen3_explain_l29_l31_signed_channel_split

mlpchan_pos_zero_L29_K16:
  success: 5/5 -> 5/5
  drift:   0/5 -> 4/5
  damage = 0
  repair = 4

mlpchan_pos_zero_L29_K64:
  success: 5/5 -> 5/5
  drift:   0/5 -> 4/5
  damage = 0
  repair = 4
```

输出模式转移：

```text
other_or_wrong -> explain_answer: 4 rows
```

这个结果确认 Phase221 的现象稳定存在：qwen3 explain（解释模式）L29 的一组 MLP channel 被置零后，可以把 drift（漂移）修复成 explain_answer（解释回答）。

但新的关键校准是：

```text
产生 repair 的不是 negative group（负向组），而是 positive group（正向组）。
```

因此不能简单认为：

```text
positive = 目标通道；
negative = 竞争通道。
```

更准确的解释是：

```text
signed score 只描述成功-漂移均值差分沿 residual direction 的线性投影；
zero 干预改变的是当前样本的实际通道激活；
所以 positive channel 被置零后仍可能修复 drift。
```

#### 5.2 qwen3 explain L31 positive boost 弱修复

```text
mlpchan_pos_boost_L31_K16:
  drift: 0/5 -> 1/5
  repair = 1

mlpchan_pos_boost_L31_K64:
  drift: 0/5 -> 1/5
  repair = 1
```

这说明 L31 positive channel（正向通道）可能包含少量 explain target factor（解释目标因子），但增强通道不能复制 Phase219/Phase220 的 residual direction（残差方向）强效应。

#### 5.3 qwen3 repeat L31 高分通道仍无目标修复

qwen3 repeat（复读模式）L31 的 positive channel（正向通道）有高分候选：

```text
channel 6567:
  signed = 5.4276, step = 3
  signed = 4.9498, step = 1

channel 9407:
  signed = 3.1673, step = 3
```

但行为层面：

```text
damage = 0
repair = 0
```

这继续证明：

```text
高 signed channel score（带符号通道分数）不等于行为因果闭合。
```

#### 5.4 GLM4 repeat L30 positive zero 破坏 repeat success

```text
glm4_repeat_l28_l30_signed_channel_split

mlpchan_pos_zero_L30_K4:
  success: 5/5 -> 3/5
  damage = 2

mlpchan_pos_zero_L30_K16:
  success: 5/5 -> 3/5
  damage = 2

mlpchan_pos_zero_L30_K64:
  success: 5/5 -> 3/5
  damage = 2
```

输出模式转移包括：

```text
repeat_answer -> next_task_or_format
repeat_answer -> echo_then_answer
repeat_answer -> other_or_wrong
```

说明 GLM4 L30 positive channel group（正向通道组）对 repeat_answer（复读回答）有局部必要性，但它不是完整 repeat（复读）机制。

#### 5.5 negative channel group 基本无行为效应

本轮重要负结果：

```text
qwen3 explain L29/L31 negative zero/boost:
  damage = 0
  repair = 0

GLM4 repeat L28/L30 negative zero/boost:
  damage = 0
  repair = 0
```

因此 Phase221 附件里的“竞争/漂移通道可能在 negative group（负向组）”需要进一步校准：

```text
竞争效应不一定落在 signed negative group；
它可能落在 positive group 中，
因为竞争效应可能由当前激活值、上下文门控和置零后的后续层反应共同决定。
```

### 6. 关键通道候选

qwen3 explain L29 positive group（正向组）：

```text
channel 6627:
  step = 3
  signed = 3.7097
  delta_z = -19.8707
  down_dot = -0.1867

channel 5880:
  step = 2
  signed = 2.4408
  delta_z = 10.4733
  down_dot = 0.2330

channel 1070:
  step = 3
  signed = 2.0806
  delta_z = -11.3188
  down_dot = -0.1838
```

qwen3 explain L31 positive group（正向组）：

```text
channel 580:
  step = 2
  signed = 5.1658
  delta_z = 19.5678
  down_dot = 0.2640

channel 1735:
  step = 3
  signed = 2.9321
  delta_z = 16.0600
  down_dot = 0.1826
```

GLM4 repeat L30 positive group（正向组）：

```text
channel 7088:
  step = 1
  signed = 1.2031
  delta_z = 3.4872
  down_dot = 0.3450

channel 9374:
  step = 1
  signed = 0.7667
  delta_z = 2.4661
  down_dot = 0.3109
```

这些通道进入后续候选池，但不能称为单神经元机制。

### 7. 对当前路线的判断

Phase222 支持 Phase221 的主判断，但给出一个更重要的修正：

```text
MLP channel group 是局部因果候选；
但是 positive / negative score 不是通道语义标签。
```

当前更合理的三因子解释是：

```text
1. 成功-漂移均值差分；
2. down projection 写入方向；
3. 当前样本的实际通道激活与上下文门控。
```

因此，StateWriteSource（状态写入来源）不能只靠线性通道分数闭合。

### 8. 问题和硬伤

```text
1. signed score 仍是线性近似，只看 down projection 与 residual direction 的点积。

2. zero 干预和 boost 干预回答的问题不同：
   zero 更像必要性/移除测试；
   boost 更像充分性/增强测试。

3. positive / negative 的符号不是语义符号，只是相对 residual direction 的投影符号。

4. qwen3 explain L29 positive zero 能 repair，
   说明通道角色混有门控、竞争、格式切换或漂移抑制解除。

5. GLM4 repeat L30 positive zero 只 damage 2/5，
   说明 repeat 模式不是单层单通道闭合。

6. qwen3 repeat L31 高分通道行为无效，
   说明通道高分可能只是相关项、可补偿项或被后续层吸收。

7. 当前模型是小模型，
   内部编码机制可能比强模型粗糙，结果外推到真实语言编码机制需要保留 30% 到 50% 不确定性。
```

### 9. 全局图谱更新

截至 Phase222，全局图谱应更新为：

```text
PromptTrigger
→ CandidateRoutePath
→ MLP gated channel groups
→ ResidualAccumulation
→ StateMaintainPath
→ ReadoutCompetition
→ OutputPattern
```

局部图谱为：

```text
success-drift residual direction
→ signed channel candidates
→ context-gated channel activation
→ channel zero/boost causal response
→ output_pattern transition
```

核心进展是：

```text
MLP channel 的符号分数不是最终机制；
真正机制更可能是“通道候选 + 当前激活 + 上下文门控 + 残差累积”的组合。
```

### 10. 智能理论角度的关键洞察

如果语言是动态模式网络，那么局部通道不应理解成一个概念或一个语义标签，而更像模式网络中的小控制因子。

当前结果支持：

```text
知识网络：对象-关系-值模式在 residual state 中被激活；
推理能力：多步模式状态在 residual stream 中维持和转移；
语法系统：格式/边界/续写模式参与 ReadoutCompetition；
编码机制：MLP channel groups 负责局部状态写入或竞争平衡。
```

但这仍不是闭合。当前公式只能解释部分局部行为，不能完整模拟真实运行机制。

### 11. 当前进度估计

```text
小模型模式机制图谱：约 69%
StatePath：约 54%
StateWriteCause：约 37%
StateWriteSource：约 28%
MLP/ResidualWrite：约 42%
MLP channel source：约 18%
signed channel factor split：约 12%
context-gated channel mechanism：约 8%
ReadoutPath：约 24%
路径因果机制：约 31%
模型内部自然闭合：约 36%
任务层产品闭合：约 55%
通用语言机制外推置信：约 39% 到 44%
```

### 12. 下一阶段任务：Phase223

Phase223 仍属于同一阶段，应继续客观拼图。

建议标题：

```text
Phase223: 通道因果角色的激活态分层与门控验证
```

核心任务：

```text
1. 对 qwen3 explain L29 positive zero-repair 通道做激活态分层：
   分别统计 success、drift、patched drift 中这些通道的实际 z 值。

2. 不只按 signed score 选通道，
   还要按 drift 当前激活强度、success 当前激活强度、zero 后变化方向选通道。

3. 对同一通道做三种干预：
   zero 当前值；
   clamp 到 success 均值；
   clamp 到 drift 均值。

4. 判断 qwen3 explain L29 的 repair 来自：
   移除竞争激活，
   恢复目标激活，
   解除后续层门控，
   还是改变首词元/格式边界。

5. 对 GLM4 repeat L30 positive damage 通道做同样分析：
   判断它们是 repeat-format 必要通道，
   还是维持 repeat 与 echo 之间边界的门控通道。
```

Phase223 的重点不是继续给公式 patch（补丁），而是把通道角色从分数排序推进到激活态角色分类。

## Phase 223: 通道因果角色的激活态分层与门控验证 [2026-07-07 00:31]

### 1. 本阶段任务

Phase223 继续 Phase222 的同一阶段目标：解释为什么 qwen3 explain（解释模式）L29 的 positive channel（正向通道）被 zero（置零）后可以修复 drift（漂移），以及 GLM4 repeat（复读模式）L30 positive channel 为什么具有局部必要性。

Phase222 已经证明：

```text
signed positive / signed negative 不能直接解释为目标通道 / 竞争通道。
```

Phase223 因此把通道角色从“分数排序”推进到“激活态分层”，对同一批通道比较三类干预：

```text
zero 当前激活；
clamp 到 success 均值；
clamp 到 drift 均值。
```

### 2. 脚本与结果

新增脚本：

```text
tests/gpt5/phase223_channel_activation_gate_validation.py
tests/gpt5/run_phase223_channel_activation_gate_validation.sh
```

结果目录：

```text
tests/result/phase223_channel_activation_gate_validation/channel_activation_gate_validation/
```

已完成检查：

```text
python -m py_compile tests/gpt5/phase223_channel_activation_gate_validation.py
bash -n tests/gpt5/run_phase223_channel_activation_gate_validation.sh
```

已按 qwen3、GLM4、DS7B 顺序运行，模型之间释放 GPU 显存。

### 3. 算法原理和公式

Phase223 继续使用 Phase222 的 signed channel score（带符号通道分数）选择候选通道：

$$
\mathrm{SignedChannelScore}_{l,t,c}
=
\Delta z_{l,t,c}
\cdot
\left(
W^{down}_{l,:,c}
\cdot
\hat v_{l,t}^{S-D}
\right)
$$

但新增激活态统计：

$$
\boxed{
\mu^{S}_{l,t,c}
=
\mathbb{E}[z_{l,t,c}\mid success]
}
$$

$$
\boxed{
\mu^{D}_{l,t,c}
=
\mathbb{E}[z_{l,t,c}\mid drift]
}
$$

三类干预为：

$$
\boxed{
z'_{l,t,c}=0
}
$$

$$
\boxed{
z'_{l,t,c}=\mu^{S}_{l,t,c}
}
$$

$$
\boxed{
z'_{l,t,c}=\mu^{D}_{l,t,c}
}
$$

测试参数：

```text
max_eval_rows = 5
max_channel_steps = 4
max_steps = 6
top_channels = 96
K = 4, 16, 64
```

### 4. 总体结果

跨模型汇总：

```text
spec_count = 3
filter_rows = 56
reproducible_success_rows = 17
reproducible_drift_rows = 20
rollout_rows = 560
channel_score_rows = 2304
activation_stat_rows = 288
total_damage_match_loss = 15
total_repair_match_gain = 12
```

分模型结果：

```text
qwen3:
  rollout_rows = 370
  channel_score_rows = 1536
  activation_stat_rows = 192
  damage = 2
  repair = 12

GLM4:
  rollout_rows = 190
  channel_score_rows = 768
  activation_stat_rows = 96
  damage = 13
  repair = 0

DS7B:
  reproducible_success_rows = 0
  rollout_rows = 0
```

DS7B 仍没有可复现 success（成功样本），继续作为客观空缺记录。

### 5. 关键客观结果

#### 5.1 qwen3 explain L29：success clamp 与 zero 都能修复 drift

最强结果：

```text
qwen3_explain_l29_l31_activation_gate

mlpchan_pos_success_L29_K64:
  success: 5/5 -> 5/5
  drift:   0/5 -> 4/5
  damage = 0
  repair = 4

mlpchan_pos_zero_L29_K16:
  success: 5/5 -> 5/5
  drift:   0/5 -> 4/5
  damage = 0
  repair = 4

mlpchan_pos_zero_L29_K64:
  success: 5/5 -> 5/5
  drift:   0/5 -> 4/5
  damage = 0
  repair = 4
```

这比 Phase222 更进一步：

```text
zero 修复说明移除当前激活可以释放 explain；
success clamp 也修复说明把通道拉到 success 激活态同样可以释放 explain。
```

因此 qwen3 L29 positive channels（正向通道）不是纯目标写入通道，也不是纯竞争通道，而更像 gating-sensitive state channels（门控敏感状态通道）。

#### 5.2 qwen3 explain L29：drift clamp 会轻微破坏 success

```text
mlpchan_pos_drift_L29_K16:
  success: 5/5 -> 4/5
  damage = 1

mlpchan_pos_drift_L29_K64:
  success: 5/5 -> 4/5
  damage = 1
```

这说明：

```text
L29 positive channel 的 drift 激活态确实带有破坏 explain 的成分。
```

但破坏只有 1/5，说明它不是完整控制器，只是局部状态因子。

#### 5.3 GLM4 repeat L30：drift clamp 强破坏 repeat success

最强 GLM4 结果：

```text
glm4_repeat_l30_activation_gate

mlpchan_pos_drift_L30_K4:
  success: 5/5 -> 1/5
  damage = 4

mlpchan_pos_drift_L30_K64:
  success: 5/5 -> 3/5
  damage = 2

mlpchan_pos_zero_L30_K4/K16/K64:
  success: 5/5 -> 3/5
  damage = 2
```

这说明 GLM4 L30 positive channels（正向通道）具有更明确的 repeat state（复读状态）必要性：

```text
把 success 样本中的这些通道拉到 drift 均值，
比单纯 zero 更强地破坏 repeat_answer。
```

这比 Phase222 的 zero 结果更接近“激活态因果”。

#### 5.4 negative channel group 继续基本无效

```text
qwen3 negative zero/success/drift:
  damage = 0
  repair = 0

GLM4 negative zero/success/drift:
  damage = 0
  repair = 0
```

这连续两阶段说明：

```text
当前 signed negative group 不是主要行为因果组。
```

竞争或漂移因素更可能藏在 positive group 的不同激活态中，而不是简单落在 negative group。

### 6. 激活态证据

qwen3 explain L29 的关键通道出现明显 success/drift 均值差：

```text
L29 step=3 channel 6627:
  success_z = -24.3968
  drift_z   = -4.5260
  delta     = -19.8707
  signed    = 3.7097

L29 step=2 channel 5880:
  success_z = -5.5684
  drift_z   = -16.0417
  delta     = 10.4733
  signed    = 2.4408

L29 step=3 channel 1070:
  success_z = -9.7355
  drift_z   = 1.5833
  delta     = -11.3188
  signed    = 2.0806
```

GLM4 repeat L30 关键通道：

```text
L30 step=1 channel 7088:
  success_z = 2.8844
  drift_z   = -0.6029
  delta     = 3.4872
  signed    = 1.2031
```

这些结果说明：通道差异不是微小噪声，而是具有较大激活态差分的局部结构。

### 7. 当前结论

Phase223 的核心结论：

```text
StateWriteSource 不是简单的“高分通道写目标模式”；
更像是 MLP positive channel group 的激活态门控。
```

对 qwen3 explain：

```text
L29 positive channels 的 zero 和 success clamp 都能 repair drift；
drift clamp 会轻微 damage success；
说明 L29 更像 explain/drift 的状态门控层。
```

对 GLM4 repeat：

```text
L30 positive channels 的 drift clamp 强破坏 repeat success；
说明 L30 更像 repeat 状态的必要激活态层。
```

这比 Phase222 更进一步，因为它把通道角色从“符号分数”推进到了“激活态因果”。

### 8. 问题和硬伤

```text
1. clamp 到均值仍然是粗干预，不等于真实上下文门控。

2. success_z / drift_z 是样本均值，可能掩盖多种子模式。

3. K=64 的 success clamp 才在 qwen3 L29 repair 4/5，
   K=4/K16 的 success clamp 未同样强，说明有效因子可能较分布式。

4. GLM4 L30 K4 drift clamp damage 4/5，
   但 K16 反而只有 1/5，说明 topK 通道之间可能存在相互抵消。

5. 当前只测对象-关系-值任务族，
   还不能外推到完整语法、推理和知识网络。

6. 小模型内部编码机制可能粗糙，
   对真实语言编码机制仍需保留 30% 到 50% 不确定性。
```

### 9. 全局图谱更新

当前图谱应进一步更新为：

```text
PromptTrigger
→ CandidateRoutePath
→ MLP gated channel activation
→ ResidualAccumulation
→ StateMaintainPath
→ ReadoutCompetition
→ OutputPattern
```

局部机制图谱：

```text
signed channel candidate
→ success/drift activation state
→ zero / success-clamp / drift-clamp response
→ residual state shift
→ output pattern transition
```

这说明全局图谱的第一优先级应继续放在：

```text
激活态图谱；
门控图谱；
多层残差累积图谱。
```

### 10. 智能理论角度的关键洞察

本阶段更支持“语言是动态模式网络”的路线。模式不是某个静态概念向量，而是：

```text
局部通道激活态
+ 残差状态累积
+ 输出读出竞争
```

共同形成的运行状态。

对语言三核心特性的反思：

```text
知识网络：对象-关系-值不是单点知识，而是可被激活态维持的状态模式。

推理能力：推理可能依赖状态路径在多层中的连续转移，而不是单步符号规则。

语法系统：格式、解释、复读、停止等语法/输出结构，可能共享同一套状态门控机制。
```

所以破解编码机制的关键，不是找一个“概念神经元”，而是完成：

```text
通道激活态图谱
→ 残差状态图谱
→ 输出竞争图谱
```

### 11. 当前进度估计

```text
小模型模式机制图谱：约 70%
StatePath：约 55%
StateWriteCause：约 39%
StateWriteSource：约 31%
MLP/ResidualWrite：约 45%
MLP channel source：约 22%
activation-gated channel mechanism：约 14%
ReadoutPath：约 24%
路径因果机制：约 32%
模型内部自然闭合：约 37%
任务层产品闭合：约 55%
通用语言机制外推置信：约 40% 到 45%
```

### 12. 阶段性判断与下一步

Phase222 和 Phase223 已经完成了一个阶段性小闭环：

```text
绝对值通道分数
→ 带符号通道分组
→ 激活态 clamp 验证
```

结论是：

```text
StateWriteSource 的局部候选已经从 MLP 模块推进到 MLP positive channel activation state；
但它仍未闭合到完整机制公式。
```

下一步如果继续同一大阶段，应进入 Phase224：

```text
Phase224: 多层激活态传播与 ResidualAccumulation 闭环验证
```

核心任务：

```text
1. 检查 qwen3 L29 通道激活态改变后，L31/L33 residual direction 是否随之改变。

2. 检查 GLM4 L30 drift clamp 破坏 repeat 后，后续层 readout competition 如何变化。

3. 从单层通道因果推进到多层传播因果：
   channel activation -> residual shift -> later layer state -> output readout。
```

由于 Phase222-223 已经完成本轮“通道符号与激活态校准”的阶段性目标，下一轮应把重点从单层通道转移到多层传播闭环。

## Phase 224: 多层激活态传播与 ResidualAccumulation 闭环验证 [2026-07-07 00:47]

### 1. 本阶段任务

本阶段分析附件中关于 Phase222/223 的判断是否正确，并继续推进同一大阶段目标。

附件的核心判断基本正确：

```text
Phase222/223 没有完成最终闭合；
但已经把 MLP channel 机制从高分排序推进到带符号分组 + 激活态因果验证。
```

需要保留的关键校准是：

```text
positive / negative score 不是通道语义标签；
真正有效的是通道激活态、上下文门控、残差累积和读出竞争。
```

因此 Phase224 不再继续扩大单层通道 patch（补丁），而是检查：

```text
单层通道激活态改变后，
是否沿后续层传播成 residual state shift（残差状态迁移），
并最终影响 readout competition（读出竞争）。
```

### 2. 脚本与结果

新增脚本：

```text
tests/gpt5/phase224_multilayer_activation_propagation.py
tests/gpt5/run_phase224_multilayer_activation_propagation.sh
```

结果目录：

```text
tests/result/phase224_multilayer_activation_propagation/multilayer_activation_propagation/
```

已完成语法检查：

```text
python -m py_compile tests/gpt5/phase224_multilayer_activation_propagation.py
bash -n tests/gpt5/run_phase224_multilayer_activation_propagation.sh
```

已按 qwen3、GLM4、DS7B 顺序使用本地 CUDA 模型测试，并在每个模型后释放显存。

### 3. 算法原理

Phase224 的核心测量对象是：通道干预后，后续层 hidden state（隐藏状态）相对 success-drift residual direction（成功-漂移残差方向）的投影变化。

对第 \(l_s\) 层 source channel（源通道）做干预：

$$
z'_{l_s,t,c}
\in
\left\{
0,\ \mu^{S}_{l_s,t,c},\ \mu^{D}_{l_s,t,c}
\right\}
$$

然后在后续观测层 \(l_o\) 捕获 hidden state：

$$
h^{base}_{l_o,t},
\quad
h^{patched}_{l_o,t}
$$

传播差分为：

$$
\boxed{
\Delta h^{patch}_{l_o,t}
=
h^{patched}_{l_o,t}
-
h^{base}_{l_o,t}
}
$$

相对 success-drift residual direction 的投影迁移为：

$$
\boxed{
\mathrm{ProjectionShift}_{l_o,t}
=
\Delta h^{patch}_{l_o,t}
\cdot
\hat v^{S-D}_{l_o,t}
}
$$

方向余弦为：

$$
\boxed{
\mathrm{PropagationCos}_{l_o,t}
=
\cos
\left(
\Delta h^{patch}_{l_o,t},
v^{S-D}_{l_o,t}
\right)
}
$$

同时记录 readout competition（读出竞争）指标：

```text
top token 是否改变；
target rank 是否改善；
prose / echo / stop margin 是否变化。
```

本阶段重点不是直接生成完整回答，而是做更细的多层传播观测。

### 4. 测试配置

```text
qwen3:
  pattern = answer_explain
  source layer = L29
  observe layers = L29, L31, L33

GLM4:
  pattern = answer_repeat
  source layer = L30
  observe layers = L30, L31, L32

DS7B:
  pattern = answer_explain
  source layer = L24
  observe layers = L24, L25, L26
```

干预条件：

```text
mlpchan_pos_zero
mlpchan_pos_success
mlpchan_pos_drift
K = 4, 16, 64
```

### 5. 总体结果

跨模型汇总：

```text
spec_count = 3
filter_rows = 56
reproducible_success_rows = 17
reproducible_drift_rows = 20
propagation_rows = 1296
channel_score_rows = 1152
total_top_token_changed = 141
total_target_rank_improved = 219
```

分模型结果：

```text
qwen3:
  propagation_rows = 648
  channel_score_rows = 576
  top_token_changed = 27
  target_rank_improved = 198

GLM4:
  propagation_rows = 648
  channel_score_rows = 576
  top_token_changed = 114
  target_rank_improved = 21

DS7B:
  reproducible_success_rows = 0
  propagation_rows = 0
```

DS7B 仍然缺少可复现 success（成功样本），因此不能参与传播因果判断。

### 6. qwen3 explain：L29 激活态会传播到 L31/L33

qwen3 最强传播结果来自 drift 样本上的 L29 success clamp：

```text
qwen3_explain_l29_to_l31_l33_propagation
condition = mlpchan_pos_success_L29_K64
source_group = drift_repro

observe L29:
  mean_projection_shift = +19.0338
  mean_cos = +0.3932
  top_token_changed = 3
  target_rank_improved = 5

observe L31:
  mean_projection_shift = +22.7522
  mean_cos = +0.3898
  top_token_changed = 3
  target_rank_improved = 5

observe L33:
  mean_projection_shift = +25.0593
  mean_cos = +0.3530
  top_token_changed = 3
  target_rank_improved = 5
```

这说明：

```text
把 qwen3 drift 样本的 L29 positive channels 拉到 success 激活态，
会在 L29、L31、L33 形成沿 success-drift direction 的正向 residual shift。
```

这正是 Phase224 要找的多层传播证据。

qwen3 L29 zero 也产生同向传播：

```text
condition = mlpchan_pos_zero_L29_K64
source_group = drift_repro

observe L29:
  mean_projection_shift = +11.4991
  mean_cos = +0.2563
  target_rank_improved = 7

observe L31:
  mean_projection_shift = +15.1672
  mean_cos = +0.2715
  target_rank_improved = 7

observe L33:
  mean_projection_shift = +18.0175
  mean_cos = +0.2693
  target_rank_improved = 7
```

这解释了 Phase222/223 中 L29 zero 可以 repair drift 的原因：

```text
zero 不是简单删除目标通道；
它会把 drift residual state 推向 success-drift direction 的正向区域，
并且这种迁移会传播到 L31/L33。
```

反向验证也成立。对 success 样本做 L29 drift clamp：

```text
condition = mlpchan_pos_drift_L29_K64
source_group = success_repro

observe L29:
  mean_projection_shift = -18.4517
  mean_cos = -0.3840

observe L31:
  mean_projection_shift = -21.8817
  mean_cos = -0.3688

observe L33:
  mean_projection_shift = -25.9240
  mean_cos = -0.3535
```

这说明 qwen3 L29 positive channel activation state（正向通道激活态）确实是 explain/drift residual state 的可传播因子。

### 7. GLM4 repeat：L30 干预产生反向 residual shift 和强 readout 扰动

GLM4 的主要结果不同于 qwen3。

success 样本上 L30 zero：

```text
condition = mlpchan_pos_zero_L30_K64
source_group = success_repro

observe L30:
  mean_projection_shift = -3.4881
  mean_cos = -0.3392
  top_token_changed = 2

observe L31:
  mean_projection_shift = -3.5755
  mean_cos = -0.3115
  top_token_changed = 2

observe L32:
  mean_projection_shift = -3.9675
  mean_cos = -0.3158
  top_token_changed = 2
```

success 样本上 L30 drift clamp：

```text
condition = mlpchan_pos_drift_L30_K64
source_group = success_repro

observe L30:
  mean_projection_shift = -3.5139
  mean_cos = -0.4103

observe L31:
  mean_projection_shift = -3.6013
  mean_cos = -0.3831

observe L32:
  mean_projection_shift = -3.7960
  mean_cos = -0.3775
```

这与 Phase223 的行为结果一致：

```text
GLM4 L30 drift clamp 会强破坏 repeat success；
Phase224 进一步显示，它会把 L30/L31/L32 residual state 推离 repeat success direction。
```

GLM4 的 top token changed（首选词元改变）总数为 114，远高于 qwen3 的 27，但 target rank improved（目标排名改善）只有 21，远低于 qwen3 的 198。

这说明：

```text
qwen3 explain L29 干预更像目标方向修复；
GLM4 repeat L30 干预更像读出竞争扰动和 repeat 状态破坏。
```

### 8. 本阶段是否证明闭合

没有闭合，但明显推进。

已经得到的客观拼图：

```text
1. qwen3 L29 positive channel activation state 可以沿 L31/L33 传播成正向 residual shift。

2. qwen3 L29 zero 和 success clamp 都能让 drift residual state 靠近 explain success direction。

3. qwen3 L29 drift clamp 会让 success residual state 远离 explain success direction。

4. GLM4 L30 zero / drift clamp 会让 repeat success residual state 远离 repeat direction。

5. readout competition 已经受到传播结果影响：
   qwen3 主要表现为 target rank improved；
   GLM4 主要表现为 top token changed。
```

但仍未闭合，因为：

```text
1. 只观测到传播相关，不等于完整自然生成链条；
2. 只覆盖 qwen3 explain 和 GLM4 repeat 两个强对象；
3. 没有证明哪些上游机制自然产生这些 channel activation state；
4. 没有解决 DoneStateStable 和 ModelStopExecuted；
5. DS7B 仍然缺少可复现 success，跨三模型闭合不足。
```

### 9. 对附件判断的校准

附件中关于 Phase222/223 的判断总体正确：

```text
当前路线应从单通道得分转到 activation state + residual accumulation + readout competition。
```

Phase224 给出的新增校准是：

```text
qwen3 explain L29 已经出现较清楚的多层传播链；
GLM4 repeat L30 也有传播链，但表现为破坏 repeat state 和扰动 readout；
两者不是同一种机制形状。
```

因此不能简单说“找到了通用 MLP 通道机制”。更严谨的说法是：

```text
不同模式族可能共享“通道激活态 -> 残差传播 -> 读出竞争”的大框架，
但具体方向、强度、修复/破坏形态不同。
```

### 10. 当前全局图谱

更新后的图谱：

```text
PromptTrigger
→ CandidateRoutePath
→ MLP gated channel activation
→ ResidualPropagation
→ ResidualAccumulation
→ StateMaintainPath
→ ReadoutCompetition
→ OutputPattern
```

局部因果链：

```text
L29/L30 positive channel activation state
→ same-layer residual shift
→ later-layer residual shift
→ readout metric change
→ output pattern change
```

其中 Phase224 主要补上：

```text
same-layer residual shift
→ later-layer residual shift
→ readout metric change
```

这比 Phase223 的单层激活态验证更接近机制闭环。

### 11. 问题和硬伤

```text
1. Phase224 仍是人工干预，不是自然因果链完整追踪。

2. residual direction 仍然是 success-drift 均值方向，可能混合内容、格式、长度、停止、读出竞争等因子。

3. qwen3 target rank improved 很多，但 top token changed 较少，说明 readout 变化可能还没有完全跨过解码阈值。

4. GLM4 top token changed 很多，但 target rank improved 少，说明 GLM4 干预更多是扰动，而不是目标修复。

5. 当前只测强对象：
   qwen3 explain L29；
   GLM4 repeat L30。
   还不能外推到完整语言机制。

6. DS7B 继续无稳定 success，三模型闭合不足。

7. 小模型内部结构可能较粗糙，对强模型和真实语言编码机制仍需 30% 到 50% 折扣。
```

### 12. 智能理论角度的关键洞察

Phase224 支持“语言是动态模式网络”的更具体版本：

```text
语言模式不是只存在于单层；
它会通过局部通道激活态改变残差状态，
再沿后续层传播并改变读出竞争。
```

这对知识网络、推理能力、语法系统的统一解释是：

```text
知识网络：对象-关系-值状态可以通过 residual propagation 被维持和放大。

推理能力：推理可能是多层状态传播，而不是单步符号操作。

语法系统：解释、复读、格式、边界等输出模式可能通过同一类传播链进入读出竞争。
```

因此破解语言编码机制的下一步，不是再找更多单通道，而是完成：

```text
通道激活态图谱
→ 多层残差传播图谱
→ 读出竞争图谱
```

### 13. 当前进度估计

```text
小模型模式机制图谱：约 71%
StatePath：约 57%
StateWriteCause：约 41%
StateWriteSource：约 33%
MLP/ResidualWrite：约 47%
MLP channel source：约 24%
activation-gated channel mechanism：约 17%
ResidualPropagation：约 15%
ReadoutPath：约 26%
路径因果机制：约 34%
模型内部自然闭合：约 38%
任务层产品闭合：约 55%
通用语言机制外推置信：约 40% 到 45%
```

### 14. 下一阶段任务

Phase224 仍属于同一大阶段，并补上了多层传播证据。

下一步 Phase225 应继续同一阶段，但要从“传播观测”推进到“读出竞争闭环”：

```text
Phase225: ReadoutCompetition 与输出阈值跨越验证
```

核心任务：

```text
1. 对 qwen3 explain L29 干预后 target rank improved 但 top token 未必改变的样本，
   测试需要多大 residual/readout margin 才能跨过输出阈值。

2. 对 GLM4 repeat L30 干预后 top token changed 但 target rank 不改善的样本，
   分析它是在破坏 repeat、触发 echo、还是转向 next_task_or_format。

3. 将 propagation shift 与 output_pattern transition 直接关联，
   建立 residual shift -> readout margin -> output token 的最小闭环。
```

## Phase 225: ReadoutCompetition 与输出阈值跨越验证 [2026-07-07 01:19]

### 1. 本阶段任务

本阶段分析附件中对 Phase224 的判断，并继续完成同一大阶段任务。附件判断基本正确：

```text
Phase224 已经从单层通道激活态推进到多层 residual propagation（残差传播）候选闭环；
但仍未完成自然机制闭合。
```

Phase224 的关键缺口是：

```text
qwen3 explain:
  target rank improved 很多，但 top token changed 较少。

GLM4 repeat:
  top token changed 很多，但 target rank improved 很少。
```

因此 Phase225 不重新加载模型，而是复用 Phase224 的 readout metric（读出指标）和 Phase223 的 output_pattern（输出模式）结果，分析：

```text
residual shift
→ readout margin
→ target rank
→ top token threshold
→ output pattern
```

### 2. 脚本与结果

新增脚本：

```text
tests/gpt5/phase225_readout_competition_threshold.py
tests/gpt5/run_phase225_readout_competition_threshold.sh
```

结果目录：

```text
tests/result/phase225_readout_competition_threshold/readout_competition_threshold/
```

已完成检查：

```text
python -m py_compile tests/gpt5/phase225_readout_competition_threshold.py
bash -n tests/gpt5/run_phase225_readout_competition_threshold.sh
```

本阶段属于结果再分析，不需要重新加载 qwen3、GLM4、DS7B；它直接读取 Phase224/223 的客观结果。

### 3. 算法原理

Phase225 继续使用 Phase224 的传播指标：

$$
\boxed{
\mathrm{ProjectionShift}_{l,t}
=
\Delta h^{patch}_{l,t}
\cdot
\hat v^{S-D}_{l,t}
}
$$

然后增加 readout threshold（读出阈值）分析：

$$
\boxed{
\Delta r
=
r_{base}
-
r_{patch}
}
$$

其中 \(r\) 是 target rank（目标排名），数值越小表示目标越接近 top token。

同时记录：

$$
\boxed{
\Delta m_{prose}
=
m^{patch}_{prose}
-
m^{base}_{prose}
}
$$

$$
\boxed{
\Delta m_{echo}
=
m^{patch}_{echo}
-
m^{base}_{echo}
}
$$

$$
\boxed{
\Delta m_{stop}
=
m^{patch}_{stop}
-
m^{base}_{stop}
}
$$

并统计：

```text
top_token_changed；
target_rank_improved；
min_abs_shift_for_top_change；
min_abs_shift_for_rank_improve；
top_token_pairs。
```

### 4. 总体结果

跨模型汇总：

```text
propagation_rows = 1296
threshold_rows = 108
behavior_correlation_rows = 18
total_top_token_changed = 141
total_rank_improved = 219
```

分模型：

```text
qwen3:
  propagation_rows = 648
  threshold_rows = 54
  top_token_changed = 27
  rank_improved = 198

GLM4:
  propagation_rows = 648
  threshold_rows = 54
  top_token_changed = 114
  rank_improved = 21

DS7B:
  propagation_rows = 0
```

DS7B 仍因没有可复现 success 样本而不能参与因果判断。

### 5. qwen3 explain：rank 改善先于 top token 跨阈值

qwen3 中最关键的条件仍是 L29 positive zero/success clamp。

```text
qwen3_explain_l29_to_l31_l33_propagation
condition = mlpchan_pos_zero_L29_K64
source_group = drift_repro

observe L29:
  projection_shift = +11.4991
  top_token_changed = 3
  rank_improved = 7
  prose_margin_delta = +1.4427
  top_token_pairs = can -> is

observe L31:
  projection_shift = +15.1672
  top_token_changed = 3
  rank_improved = 7

observe L33:
  projection_shift = +18.0175
  top_token_changed = 3
  rank_improved = 7
```

success clamp 也类似：

```text
condition = mlpchan_pos_success_L29_K64
source_group = drift_repro

observe L29:
  projection_shift = +19.0338
  top_token_changed = 3
  rank_improved = 5
  prose_margin_delta = +1.1510

observe L31:
  projection_shift = +22.7522
  top_token_changed = 3
  rank_improved = 5

observe L33:
  projection_shift = +25.0593
  top_token_changed = 3
  rank_improved = 5
```

重要细节：

```text
qwen3 的 rank_improved 远多于 top_token_changed。
```

例如：

```text
mlpchan_pos_success_L29_K16:
  total_rank_improved = 27
  total_top_token_changed = 0
  behavior repair = 0

mlpchan_pos_success_L29_K64:
  total_rank_improved = 21
  total_top_token_changed = 9
  behavior repair = 4
```

这说明：

```text
qwen3 explain 的 residual shift 会先改善目标排名；
只有当 readout margin 足够大时，才跨过 top token 阈值并修复 output pattern。
```

所以 Phase224 看到的“target rank improved 但 top token changed 少”不是矛盾，而是读出阈值效应。

### 6. GLM4 repeat：top token 改变多，但多为扰动

GLM4 的最强 readout 变化来自 L30 zero：

```text
glm4_repeat_l30_to_l31_l32_propagation
condition = mlpchan_pos_zero_L30_K64

total_top_token_changed = 18
total_rank_improved = 3
success_damage_match_loss = 2
drift_repair_match_gain = 0
success_patch_outputs:
  repeat_answer = 3
  echo_then_answer = 1
  other_or_wrong = 1
drift_patch_outputs:
  echo_then_answer = 2
  next_task_or_format = 1
  other_or_wrong = 2
```

top token pairs（首选词元转移）包括：

```text
Green -> The
White -> Blue
White -> The
Red -> Cardinal
Red -> The
```

这说明 GLM4 的 top token changed 不是目标修复，而是：

```text
repeat state 被破坏；
echo / next_task_or_format / other_or_wrong 竞争增强；
readout 被扰动。
```

GLM4 drift clamp 更明显：

```text
mlpchan_pos_drift_L30_K4:
  success_damage_match_loss = 4
  total_top_token_changed = 15
  total_rank_improved = 0
```

这与 Phase223 的行为结果一致：

```text
GLM4 L30 positive channel activation state 对 repeat success 有必要性；
把它拉到 drift 激活态会破坏 repeat，而不是修复 drift。
```

### 7. 输出阈值规律

当前可以得到一个谨慎规律：

```text
qwen3 explain:
  positive residual shift
  → prose margin 上升
  → target rank 改善
  → 只有部分样本跨 top token 阈值
  → 部分 drift 修复为 explain_answer

GLM4 repeat:
  negative residual shift
  → repeat state 被破坏
  → top token 频繁改变
  → 但 target rank 不改善
  → 输出转向 echo / format / other
```

这说明 ReadoutCompetition（读出竞争）不是单一指标，需要分成：

```text
目标排名改善；
最高词元阈值跨越；
竞争模式增强；
输出模式转移。
```

### 8. 与行为输出的关联

Phase225 将 Phase224 的传播指标与 Phase223 的输出模式关联起来后，得到：

```text
qwen3 mlpchan_pos_zero_L29_K64:
  mean_layer_projection_shift = +3.7813
  total_top_token_changed = 9
  total_rank_improved = 33
  drift_repair_match_gain = 4

qwen3 mlpchan_pos_success_L29_K64:
  mean_layer_projection_shift = +10.6432
  total_top_token_changed = 9
  total_rank_improved = 21
  drift_repair_match_gain = 4

GLM4 mlpchan_pos_zero_L30_K64:
  mean_layer_projection_shift = -2.5523
  total_top_token_changed = 18
  total_rank_improved = 3
  success_damage_match_loss = 2

GLM4 mlpchan_pos_drift_L30_K4:
  mean_layer_projection_shift = -1.2371
  total_top_token_changed = 15
  total_rank_improved = 0
  success_damage_match_loss = 4
```

这说明：

```text
qwen3 的正向 residual/readout 变化与 repair 相关；
GLM4 的 readout 改变主要与 damage 和模式漂移相关。
```

### 9. 问题和硬伤

```text
1. Phase225 是对 Phase224/223 结果的再分析，不是新模型干预。

2. readout margin 指标仍是代理指标，不能完全等价于完整输出概率地形。

3. target rank improved 不一定意味着正确输出，尤其在 GLM4 中已经证明会出现强扰动。

4. qwen3 的 top token changed 仍然只有一部分，说明还有未解释的解码阈值、后续生成稳定性和模式维持问题。

5. GLM4 的 repeat 结果说明 readout competition 可以被扰动，但目标修复路径仍不清楚。

6. DS7B 仍缺少可复现 success。

7. 当前仍是小模型、对象-关系-值任务族，外推到完整语言编码机制需要 30% 到 50% 折扣。
```

### 10. 当前结论

Phase225 完成了一个更小的 readout 闭环：

```text
residual propagation
→ readout rank / margin change
→ top token threshold crossing
→ output pattern repair or damage
```

但这个闭环不是全局闭合。

最重要的客观结论：

```text
1. qwen3 explain 的 L29 通道激活态可以通过 residual propagation 改善 target rank，
   但需要更大 readout margin 才能跨 top token 阈值。

2. GLM4 repeat 的 L30 通道干预会强烈改变 top token，
   但多数不是目标修复，而是 repeat state 破坏和竞争模式增强。

3. ReadoutCompetition 必须拆成 rank improvement、top-token threshold、competitor transition 三层。
```

### 11. 全局图谱更新

当前图谱：

```text
PromptTrigger
→ CandidateRoutePath
→ MLP gated channel activation
→ ResidualPropagation
→ ResidualAccumulation
→ ReadoutRankShift
→ TopTokenThreshold
→ OutputPattern
```

局部闭环：

```text
L29/L30 positive channel activation
→ later-layer residual shift
→ readout margin shift
→ target rank / competitor rank change
→ top token threshold crossing
→ pattern repair or damage
```

### 12. 当前进度估计

```text
小模型模式机制图谱：约 72%
StatePath：约 58%
StateWriteCause：约 42%
StateWriteSource：约 34%
MLP/ResidualWrite：约 48%
MLP channel source：约 25%
activation-gated channel mechanism：约 18%
ResidualPropagation：约 18%
ReadoutPath：约 30%
路径因果机制：约 36%
模型内部自然闭合：约 39%
任务层产品闭合：约 55%
通用语言机制外推置信：约 40% 到 45%
```

### 13. 下一阶段任务

Phase225 仍属于同一大阶段，并完成了 readout threshold（读出阈值）初步闭环。

下一步 Phase226 应避免继续堆 patch，而要验证自然链条来源：

```text
Phase226: Natural Trigger 到 Channel Activation 的上游来源验证
```

核心任务：

```text
1. 找出 qwen3 explain L29 positive channel activation state 是由哪些 prompt token / 上游层状态自然触发的。

2. 对 GLM4 repeat L30 positive channel activation state 做同样的上游来源分析。

3. 连接：
   PromptTrigger
   → upstream residual state
   → MLP gate/up activation
   → positive channel activation
   → residual propagation
   → readout threshold。
```

## Phase 226: 自然触发源到通道激活的初步验证 [2026-07-07 01:44]

### 1. 对 Phase225 附件判断的校准

Phase225 的核心判断基本正确。

它补上了 Phase224 之后必须区分的一层：

```text
residual shift（残差状态移动）
≠ rank improvement（目标排名改善）
≠ top-token threshold crossing（跨过最高词元阈值）
≠ target repair（目标模式修复）
```

Phase225 证明：

```text
1. qwen3 explain 中，L29 positive channel（正向通道）干预可以多次改善目标 rank，
   但不一定让目标词元跨过 top token threshold。

2. GLM4 repeat 中，L30 positive channel 干预容易改变 top token，
   但大量变化不是目标修复，而是 repeat state（重复状态）被破坏后竞争词元切换。

3. 因此继续堆 readout patch（读出补丁）会进入边际收益递减区。
```

所以 Phase226 的方向应从“继续修输出”转为验证自然链条来源：

```text
PromptTrigger（提示触发）
→ upstream residual state（上游残差状态）
→ MLP channel activation（MLP 通道激活）
→ residual propagation（残差传播）
→ readout threshold（读出阈值）
```

### 2. 本阶段测试目标

本阶段不再人工 clamp（钳制）或 zero（清零）通道，而是只改变自然 prompt（提示词）结构，观察关键通道是否自然进入 success-like state（成功样本相似状态）或 drift-like state（漂移样本相似状态）。

测试对象：

```text
qwen3: explain 模式，source layer L29，observe layers L11/L29/L31/L33
GLM4: repeat 模式，source layer L30，observe layers L12/L28/L30/L32
DS7B: explain 模式，source layer L24，observe layers L20/L24/L26/L27
```

prompt 变体：

```text
full（完整提示）
no_instruction（去掉任务说明）
short_answer_instruction（短答指令）
no_answer_anchor（去掉 Answer: 锚点）
repeat_instruction（把 explain 改成 repeat）
because_removed（移除 because 相关结构）
explain_instruction（把 repeat 改成 explain）
comma_removed（移除逗号结构）
```

### 3. 算法原理

核心思想：

```text
如果某个自然 prompt 变体能把 MLP channel state 推向 success-like state，
说明它可能包含该通道激活的自然上游触发源。

如果 channel state 接近 success-like state，但 readout 仍失败，
说明通道激活只是必要拼图之一，不是完整闭合机制。
```

#### 3.1 通道激活轴公式

令：

```text
z_l(x)
```

表示第 l 层 MLP down projection input（下投影输入）中的通道激活向量。

令：

```text
μ_S
```

表示 success rows（成功样本）的平均通道状态。

令：

```text
μ_D
```

表示 drift rows（漂移样本）的平均通道状态。

对 top-K positive channels（前 K 个正向通道）计算：

```math
\mathrm{ActivationAxis}_K(x)
=
\frac{1}{K}
\sum_{i \in C_K}
\frac{z_i(x)-\mu_{D,i}}{\mu_{S,i}-\mu_{D,i}+\epsilon}
```

解释：

```text
ActivationAxis ≈ 0：接近 drift-like state
ActivationAxis ≈ 1：接近 success-like state
ActivationAxis < 0：比 drift 更偏离 success
ActivationAxis > 1：超过 success 平均方向，可能是过强激活或异常状态
```

自然变体相对 full prompt 的变化为：

```math
\Delta \mathrm{Axis}_K
=
\mathrm{ActivationAxis}_K(x_{\mathrm{variant}})
-
\mathrm{ActivationAxis}_K(x_{\mathrm{full}})
```

#### 3.2 隐状态投影公式

令：

```math
d_l = \mu_{S,l}^{h} - \mu_{D,l}^{h}
```

表示第 l 层 hidden state（隐状态）的 success-drift 方向。

自然变体的投影变化：

```math
\Delta \mathrm{Proj}_l(x)
=
\langle h_l(x_{\mathrm{variant}})-h_l(x_{\mathrm{full}}), d_l \rangle
```

解释：

```text
正值：自然变体把 hidden state 推向 success direction（成功方向）
负值：自然变体把 hidden state 推离 success direction
```

#### 3.3 读出层指标

记录：

```text
target_rank_delta（目标词排名变化）
target_logit_delta（目标 logit 变化）
top_token_changed（最高词元是否改变）
prose_logit_delta（prose 类竞争词变化）
echo_logit_delta（echo/repeat 类竞争词变化）
```

其中：

```text
rank_delta < 0 表示目标排名改善
rank_delta > 0 表示目标排名变差
```

### 4. 脚本和结果文件

新增脚本：

```text
/tests/gpt5/phase226_natural_trigger_channel_activation.py
/tests/gpt5/run_phase226_natural_trigger_channel_activation.sh
```

结果目录：

```text
/tests/result/phase226_natural_trigger_channel_activation/natural_trigger_channel_activation/
```

三模型依次测试，避免 GPU 显存叠加：

```text
qwen3 → GLM4 → DS7B
```

规模：

```text
qwen3: activation_rows=648, hidden_rows=864, readout_rows=216
GLM4: activation_rows=648, hidden_rows=864, readout_rows=216
DS7B: activation_rows=432, hidden_rows=576, readout_rows=144
cross-model total:
activation_rows=1728
hidden_rows=2304
readout_rows=576
channel_score_rows=1728
```

脚本检查：

```text
python -m py_compile: 通过
bash -n runner: 通过
```

### 5. 关键客观结果

#### 5.1 GLM4 repeat：Answer 锚点是强自然触发源

GLM4 repeat success rows 中，移除 Answer anchor 后：

```text
variant=no_answer_anchor, step=1, layer=L30

K=4:
axis=-0.6807
delta_axis=-1.6599
success_closer=0

K=16:
axis=-0.3347
delta_axis=-1.3872
success_closer=0

K=64:
axis=0.0092
delta_axis=-1.0138
success_closer=0
```

GLM4 repeat drift rows 中，同一变体：

```text
variant=no_answer_anchor, step=1, layer=L30

K=4:
axis=-0.6748
delta_axis=-0.8954
success_closer=0

K=16:
axis=-0.3417
delta_axis=-0.5685
success_closer=0
```

说明：

```text
对 GLM4 repeat 来说，Answer: 不是普通文本边界，
而是 L30 repeat-related positive channel state 的强自然触发源之一。

移除 Answer anchor 后，success rows 的通道状态从 success-like 明显塌回 drift-like 或更低。
```

读出层也同步发生大变化：

```text
GLM4 success no_answer_anchor step=1:
top_token_changed=6/6
rank_delta=-5285.67
prose_delta=+1.7344
echo_delta=-2.25
top token 主要变为 For

GLM4 drift no_answer_anchor step=1:
top_token_changed=6/6
rank_delta=-5263.67
prose_delta=+0.9948
echo_delta=-3.1771
top token 主要变为 For
```

谨慎解释：

```text
rank_delta 改善并不等于 repeat 修复。
这里更像是 Answer anchor 被移除后，模型进入新的 prose/continuation readout regime（散文续写读出机制）。
```

#### 5.2 qwen3 explain：Answer 锚点控制的不是单一通道，而是通道与读出的边界条件

qwen3 explain drift rows 中，移除 Answer anchor 后：

```text
variant=no_answer_anchor, step=1, layer=L29

K=64:
axis=0.7352
delta_axis=+0.7388
success_closer=6

K=16:
axis=0.6337
delta_axis=+0.6378
success_closer=6

K=4:
axis=0.3203
delta_axis=+0.4749
success_closer=6
```

这说明：

```text
在 qwen3 drift rows 中，移除 Answer anchor 反而能把 L29 选定通道推向 success-like side。
```

但读出结果并没有闭合，反而明显破坏：

```text
qwen3 success no_answer_anchor step=1:
top_token_changed=6/6
rank_delta=-740.67
prose_delta=-5.375
echo_delta=-10.7708
top token 主要变为 Then

qwen3 drift no_answer_anchor step=1:
top_token_changed=6/6
rank_delta=-691.67
prose_delta=-5.6667
echo_delta=-9.0
top token 主要为 Then / The
```

隐状态投影同样出现强移动：

```text
qwen3 drift no_answer_anchor step=1:
L33 projection delta=+89.1919
L31 projection delta=+73.0376
L29 projection delta=+46.3266
```

说明：

```text
qwen3 explain 的 Answer anchor 不是简单的“激活正向通道”按钮。
它同时参与 readout boundary（读出边界）、文本续写 regime（续写机制）和 explain/repeat 竞争模式切换。

所以 qwen3 中出现了：

channel state 更像 success，
hidden projection 更像 success，
但 top token 仍然进入 Then/The 续写模式。
```

这是非常重要的负结果：

```text
ChannelActivation alone is not closure.
（单独通道激活不是闭合。）
```

#### 5.3 qwen3 指令内容影响后续生成步的自然状态

qwen3 drift rows 中，repeat_instruction 在 step=3：

```text
variant=repeat_instruction, step=3
L33 projection delta=+63.5865
L31 projection delta=+42.4876
L29 projection delta=+36.0306

readout:
top_token_changed=4/6
rank_delta=+402.33
prose_delta=-15.3542
top tokens 包括 Answer / be
```

qwen3 success rows 中，同一变体在 step=3：

```text
L33 projection delta=-43.1403
L31 projection delta=-37.9783
L29 projection delta=-27.2162
```

说明：

```text
自然触发源不是单个 token。
它至少包含：

1. instruction content（任务说明内容）
2. answer anchor（回答锚点）
3. generated prefix（已生成前缀）
4. current decoding step（当前解码步）
```

#### 5.4 DS7B 结果只能作为弱参考

DS7B explain drift rows 中，no_answer_anchor 在 L24：

```text
variant=no_answer_anchor, step=1, layer=L24

K=64:
axis=0.7240
delta_axis=+0.7240
success_closer=2

K=16:
axis=0.6035
delta_axis=+0.6035
success_closer=2

K=4:
axis=0.5988
delta_axis=+0.5988
success_closer=2
```

但 DS7B 的 drift 样本数量只有 2，且此前行为闭合较弱，所以只能说明：

```text
DS7B 对自然 prompt trigger 也高度敏感。
```

不能据此推出稳定机制。

### 6. 阶段性结论

Phase226 得到的是：

```text
自然触发源存在。
```

但不是：

```text
自然触发源已经闭合。
```

更准确的结果是：

```text
1. GLM4 repeat 的 Answer anchor 是 L30 repeat-related channel state 的强自然触发源。

2. qwen3 explain 的自然触发机制更复杂，
   Answer anchor 同时影响 channel activation、hidden state projection 和 readout boundary。

3. MLP channel activation 可以与 readout threshold 脱钩。

4. 因此 “找到正向通道” 不是终点，
   还必须找到该通道写入 residual 后如何进入 readout regime。
```

新的拼图更新：

```text
PromptTrigger
→ InstructionFrame
→ AnswerAnchor
→ GeneratedPrefix
→ StepCondition
→ MLPChannelActivation
→ ResidualStateShift
→ ReadoutRegimeSelection
→ TopTokenThreshold
→ OutputPattern
```

其中关键新增节点是：

```text
ReadoutRegimeSelection（读出机制选择）
```

它解释了为什么：

```text
channel state success-like
hidden projection success-like
但 top token 仍可能进入 Then / The / For 这类续写模式。
```

### 7. 问题、硬伤和瓶颈

#### 7.1 prompt 变体仍然过粗

当前变体是结构级干预：

```text
去掉 instruction
去掉 Answer:
替换 explain/repeat
移除 comma/because
```

它能定位自然来源的大区域，但不能精确定位：

```text
哪个 token
哪个位置
哪个上游层
哪个 gate/up channel
哪个 residual write path
```

#### 7.2 ActivationAxis 可能混合多个子模式

Phase226 使用 Phase210 的 success/drift 均值构造方向。

问题是：

```text
success rows 内部可能不止一种 success mode。
drift rows 内部也可能有多个 drift mode。
```

所以 axis 接近 1 不一定代表真正功能闭合，只代表接近某个均值方向。

#### 7.3 DS7B 证据弱

DS7B 的样本量和行为闭合都偏弱。

本阶段只能把 DS7B 当作：

```text
跨模型敏感性参考
```

不能作为机制闭合证据。

#### 7.4 还没有拆开 gate/up 机制

当前抓取的是 MLP down projection input。

但真实链条应拆成：

```text
input hidden state
→ gate projection
→ up projection
→ activation function
→ gated product
→ down projection
→ residual write
```

现在只看到了中后段，还没有看到上游 gate 如何打开。

### 8. 智能理论角度的谨慎洞察

从智能理论看，语言更像动态模式网络，而不是单一语义向量。

本阶段支持：

```text
语言输出 = 模式触发 + 状态写入 + 读出竞争
```

更具体：

```text
Prompt 不是单纯提供语义。
Prompt 在模型内部同时设置：

1. 任务模式
2. 输出边界
3. 生成步状态
4. 候选读出空间
5. MLP channel activation 条件
```

因此语法、逻辑、标点、结束、复述、解释，可能不是彼此独立的模块，而是同一个动态模式网络在不同层级上的表现。

但当前不能把这一点总结成最终理论，因为还缺：

```text
1. token-level trigger attribution（词元级触发归因）
2. gate/up causal path（门控/上投影因果路径）
3. residual write map（残差写入图谱）
4. readout regime switch（读出机制切换）
5. 跨模型稳定复现
```

### 9. 当前进度估计

```text
小模型模式机制图谱：约 73%
StatePath：约 59%
StateWriteCause：约 43%
StateWriteSource：约 36%
NaturalTrigger → ChannelActivation：约 12%
MLP/ResidualWrite：约 49%
activation-gated channel mechanism：约 20%
ResidualPropagation：约 19%
ReadoutPath：约 32%
ReadoutRegimeSelection：约 8%
路径因果机制：约 37%
模型内部自然闭合：约 40%
任务层产品闭合：约 55%
通用语言机制外推置信：约 40% 到 45%
```

### 10. 下一阶段任务

Phase226 和 Phase225 属于同一阶段性目标：

```text
从人工干预修复，转向自然机制来源定位。
```

Phase226 已完成第一轮自然触发源粗定位，但没有完成闭合。

下一阶段仍属于同一阶段，应继续自动推进：

```text
Phase227: token-level trigger attribution and gate/up decomposition
（词元级触发归因与 gate/up 分解）
```

核心任务：

```text
1. 对 Answer:、instruction、generated prefix 做 token-level ablation（词元级消融）。

2. 在 qwen3 L29、GLM4 L30、DS7B L24 拆分：
   gate projection
   up projection
   activation product
   down projection input
   residual write。

3. 找出自然 prompt token 如何打开 positive channel。

4. 检查该 positive channel 是否真的写入目标 residual direction，
   以及为何有时进入 Then/The/For 读出机制。

5. 把目标从 “修复输出” 改成 “画出自然触发到读出的机制图谱”。
```

## Phase 227: 词元触发与 gate/up/product 分解 [2026-07-07 01:51]

### 1. 阶段目标

Phase226 已经证明：

```text
自然 prompt trigger（自然提示触发源）确实会改变关键 MLP channel state（通道状态）。
```

但 Phase226 仍然是粗定位，它没有回答：

```text
1. 到底是哪个 token / 哪个片段触发了通道变化？
2. 通道变化来自 gate projection、up projection，还是 gated product？
3. channel state 变化为什么有时不能进入正确 readout regime？
```

Phase227 因此继续同一阶段性目标：

```text
从人工干预修复，转向自然机制来源定位。
```

但本阶段不继续修输出，只拆内部机制。

### 2. 测试脚本和结果

新增脚本：

```text
/tests/gpt5/phase227_token_trigger_gateup_decomposition.py
/tests/gpt5/run_phase227_token_trigger_gateup_decomposition.sh
```

结果目录：

```text
/tests/result/phase227_token_trigger_gateup_decomposition/token_trigger_gateup_decomposition/
```

三模型顺序测试：

```text
qwen3 → GLM4 → DS7B
```

输出规模：

```text
qwen3:
activation_rows=5184
hidden_rows=1728
readout_rows=432

GLM4:
activation_rows=1296
hidden_rows=1728
readout_rows=432

DS7B:
activation_rows=3888
hidden_rows=1296
readout_rows=324

cross-model:
activation_rows=10368
hidden_rows=4752
readout_rows=1188
channel_score_rows=1728
```

脚本检查：

```text
python -m py_compile: 通过
bash -n runner: 通过
```

### 3. 算法原理

本阶段把 Phase226 的 down projection input（下投影输入）扩展为四个组件：

```text
gate projection output（门控投影输出）
up projection output（上投影输出）
product / down projection input（门控乘积 / 下投影输入）
recomputed product（重算门控乘积）
```

对 LLaMA/Qwen/DeepSeek 类 MLP，基本结构是：

```math
g_l(x) = W^{gate}_l h_l(x)
```

```math
u_l(x) = W^{up}_l h_l(x)
```

```math
z_l(x) = \phi(g_l(x)) \odot u_l(x)
```

```math
\Delta h_l(x) = W^{down}_l z_l(x)
```

其中：

```text
g_l: gate output（门控输出）
u_l: up output（上投影输出）
z_l: gated product（门控乘积，也就是 down input）
φ: activation function（激活函数）
⊙: elementwise product（逐元素乘法）
```

对每个组件独立计算 success-drift axis（成功-漂移轴）：

```math
\mathrm{Axis}_{K,c}(x)
=
\frac{1}{K}
\sum_{i \in C_K}
\frac{c_i(x)-\mu_{D,c,i}}{\mu_{S,c,i}-\mu_{D,c,i}+\epsilon}
```

其中：

```text
c ∈ {gate, up, product, recomputed_product}
C_K 是 Phase221/222 选出的 positive write channels（正向写入通道）
```

自然变体的影响：

```math
\Delta \mathrm{Axis}_{K,c}
=
\mathrm{Axis}_{K,c}(x_{\mathrm{variant}})
-
\mathrm{Axis}_{K,c}(x_{\mathrm{full}})
```

词元触发方式：

```text
1. 保留 Phase226 的结构变体：
   no_instruction
   short_answer_instruction
   no_answer_anchor
   repeat_instruction / explain_instruction
   because_removed / comma_removed

2. 新增 token-level removal（词元级删除）：
   对 Answer、冒号、because、same、twice、comma、reason、以及 prompt 尾部邻近 token 做逐个删除。
```

### 4. 关键客观结果

#### 4.1 qwen3 explain：gate/up/product 不同步，up 组件波动最大

qwen3 L29 暴露出完整组件：

```text
components = gate, up, product, recomputed_product
```

最强变化集中在 up component（上投影组件），尤其是 step=3 的 K=4：

```text
drift short_answer_instruction step=3 up K=4:
axis=3.3289
delta_axis=+29.4829
success_closer=0

drift repeat_instruction step=3 up K=4:
axis=-4.3436
delta_axis=+21.8105
success_closer=0

drift no_instruction step=3 up K=4:
axis=-5.7025
delta_axis=+20.4515
success_closer=0

success no_instruction step=3 up K=4:
axis=15.2370
delta_axis=+18.4728
success_closer=4
```

但是 product / recomputed_product 的变化明显更小：

```text
drift no_answer_anchor step=1 product K=4:
axis=0.3254
delta_axis=+0.8706
success_closer=4

drift drop_tok_3_apple/cardinal step=1 product K=64:
axis≈0.894
delta_axis≈+0.882
success_closer=2
```

这说明：

```text
qwen3 的 up projection 对 prompt token 非常敏感，
但真正进入 down input 的 product 会被 gate 调制压缩。
```

因此不能只看 up 或 gate 的大幅变化，必须看：

```text
gate × up → product → residual write
```

#### 4.2 qwen3 explain：词元删除能改变隐藏状态，但不等于读出闭合

qwen3 hidden projection（隐状态投影）中，多个词元级删除造成强移动：

```text
drift drop_tok_3_cardinal step=1:
L33 projection_delta=+126.9386
L31 projection_delta=+93.4874
L29 projection_delta=+63.7177

drift no_answer_anchor step=1:
L33 projection_delta=+88.0596
L31 projection_delta=+71.8746
L29 projection_delta=+44.5747
```

读出层中：

```text
drift no_answer_anchor step=1:
top_token_changed=4/4
rank_delta=-646.5
top_tokens=The / Then

success no_answer_anchor step=1:
top_token_changed=4/4
rank_delta=-622.5
top_tokens=Then

drift repeat_instruction step=3:
top_token_changed=4/4
rank_delta=+620.5
top_tokens=Answer
```

结论：

```text
qwen3 中，删除 Answer anchor 或修改 instruction 会强烈改变 hidden state 和 readout，
但它经常把输出推入 Then/The/Answer 等读出机制，
不是稳定目标修复。
```

#### 4.3 GLM4 repeat：组件捕获只稳定得到 product，但 Answer anchor 仍是最强触发源

GLM4 本轮只稳定捕获到：

```text
component = product
```

这可能是模型 MLP 模块命名或结构封装不同，gate/up 没有被当前 hook 命中。

但 product 结果非常清楚：

```text
success no_answer_anchor step=1 product K=4:
axis=-0.6802
delta_axis=-1.6802
success_closer=0

drift no_answer_anchor step=1 product K=4:
axis=-0.6818
delta_axis=-1.3373
success_closer=0

success no_answer_anchor step=1 product K=16:
axis=-0.3351
delta_axis=-1.3351
success_closer=0

success no_answer_anchor step=1 product K=64:
axis=0.0062
delta_axis=-0.9938
success_closer=0
```

读出层同步显示：

```text
GLM4 drift no_answer_anchor step=1:
top_token_changed=4/4
rank_delta=-6151.25
top_token=For

GLM4 success no_answer_anchor step=1:
top_token_changed=4/4
rank_delta=-5467.0
top_token=For
```

这进一步支持 Phase226：

```text
GLM4 repeat 的 Answer anchor 是强自然触发源。
移除它会让 L30 product state 从 repeat-related success state 塌回非目标读出机制。
```

但这仍不是 repeat 修复，而是 readout regime switch（读出机制切换）。

#### 4.4 DS7B：gate/product 都对 Answer anchor 敏感，但证据仍弱

DS7B L24 暴露出完整组件：

```text
components = gate, up, product, recomputed_product
```

drift no_answer_anchor step=1：

```text
gate K=64:
axis=0.9797
delta_axis=+0.9797
success_closer=2

gate K=16:
axis=0.9540
delta_axis=+0.9540
success_closer=2

product K=64:
axis=0.7240
delta_axis=+0.7240
success_closer=2

product K=16:
axis=0.6035
delta_axis=+0.6035
success_closer=2
```

这说明 DS7B 也对 Answer anchor 高度敏感。

但 DS7B 的读出 rank 变化极大：

```text
drop_tok_6_Answer step=1 rank_delta=-51866
no_answer_anchor step=1 rank_delta 可达 ±27000 级别
```

结合此前 DS7B drift 样本量偏小和行为闭合弱，本阶段仍只能把 DS7B 作为：

```text
弱参考 / 敏感性证据
```

不能作为稳定机制闭合证据。

### 5. 本阶段新增核心拼图

新增拼图 1：

```text
up projection sensitivity（上投影敏感性）很高，
但 product 才是实际进入 residual write 的关键前态。
```

新增拼图 2：

```text
gate 调制会压缩或重塑 up 的大幅变化。
所以只看单个投影输出会夸大 prompt token 的真实写入效果。
```

新增拼图 3：

```text
Answer anchor 在 GLM4 repeat 中是强 product-state trigger。
```

新增拼图 4：

```text
qwen3 explain 的自然触发不是单 token 控制，
而是 answer anchor + instruction + generated prefix + step condition 的组合。
```

新增拼图 5：

```text
hidden state 朝 success direction 移动后，
仍可能进入 Then/The/For/Answer 等非目标 readout regime。
```

所以当前全局图谱更新为：

```text
PromptToken / PromptSpan
→ InstructionFrame
→ AnswerAnchor
→ StepCondition
→ gate projection
→ up projection
→ gated product
→ down projection
→ residual write
→ hidden trajectory
→ readout regime selection
→ top-token threshold
→ output pattern
```

### 6. 问题和硬伤

#### 6.1 qwen3 up K=4 的极端值不能过度解释

qwen3 up component 在 K=4 上出现很大的 axis 和 delta_axis。

原因可能是：

```text
1. K=4 太小，少数通道分母很小会放大 axis。
2. up projection 不是最终写入态。
3. gate 可能把 up 的大幅变化压制掉。
```

所以更可靠的是：

```text
product / recomputed_product
```

而不是单独的 up 极值。

#### 6.2 GLM4 gate/up 没有被当前 hook 完整捕获

GLM4 本轮只稳定得到 product。

这不是机制结论，而是工具限制：

```text
当前 hook 可能没有覆盖 GLM4 的真实 gate/up 命名或封装路径。
```

后续如果要精细拆 GLM4，必须先做 module tree audit（模块树审计）。

#### 6.3 token 删除不是严格因果归因

删除一个 token 会同时改变：

```text
tokenization（分词）
position（位置）
attention context（注意力上下文）
prompt formatting（提示格式）
readout boundary（读出边界）
```

所以当前只能叫：

```text
token-level sensitivity（词元级敏感性）
```

还不能叫严格因果来源。

#### 6.4 小模型偏差仍然明显

当前三模型都是小模型，内部结构可能更粗糙。

尤其 DS7B 的巨大 rank 跳变说明：

```text
小模型可能把多个机制压在同一个粗糙方向上。
```

因此外推到真实语言编码机制时，需要保留 30% 到 50% 偏差空间。

### 7. 阶段性结论

Phase227 的结论：

```text
自然触发到 MLP product 的链条已经看到，但还没有闭合。
```

更准确地说：

```text
1. GLM4 repeat:
   Answer anchor → L30 product state 的证据较强。

2. qwen3 explain:
   prompt token 会强烈改变 gate/up/product 和 hidden trajectory，
   但 readout regime selection 仍是主要未解节点。

3. DS7B:
   Answer anchor 敏感性存在，但稳定性不足。
```

本阶段最重要的负结果：

```text
不是所有 channel/product success-like shift 都能形成正确输出。
```

这说明最终图谱里必须保留：

```text
ReadoutRegimeSelection
```

不能把它简化成：

```text
residual direction → top token
```

### 8. 当前进度估计

```text
小模型模式机制图谱：约 74%
StatePath：约 60%
StateWriteCause：约 44%
StateWriteSource：约 38%
NaturalTrigger → ChannelActivation：约 18%
TokenTrigger → Gate/Up/Product：约 10%
MLP/ResidualWrite：约 51%
activation-gated channel mechanism：约 23%
ResidualPropagation：约 20%
ReadoutPath：约 33%
ReadoutRegimeSelection：约 10%
路径因果机制：约 38%
模型内部自然闭合：约 41%
任务层产品闭合：约 55%
通用语言机制外推置信：约 40% 到 45%
```

### 9. 阶段边界判断和下一步

Phase226 + Phase227 已完成当前阶段的第一轮目标：

```text
自然 prompt trigger 到 MLP channel/product state 的初步图谱。
```

下一个任务仍然相关，但已经进入新的验证阶段，不应继续无限自动堆测试。

下一阶段建议：

```text
Phase228: module-tree calibrated gate/up causal validation
（模块树校准后的 gate/up 因果验证）
```

核心任务：

```text
1. 先审计 qwen3、GLM4、DS7B 的 MLP module tree，确认 gate/up/down 的真实路径。

2. 对 qwen3 和 DS7B 做 gate-only、up-only、product-only 的小幅 patch，
   验证 product 是否真是 residual write 的有效前态。

3. 对 GLM4 先补齐 gate/up 捕获，再重复同样验证。

4. 不以输出修复为第一目标，而以：
   token trigger → gate/up/product → residual write → readout regime
   的机制图谱为第一目标。
```

## Phase 228: 模块树校准后的 gate/up/product 因果验证 [2026-07-07 02:39]

### 1. 对附件判断的校准

本次附件对 Phase226 和 Phase227 的判断基本正确。

正确部分：

```text
1. Phase226/227 的方向确实是必要校准：
   从人工干预修复输出，
   转向自然触发源如何形成关键 MLP product state。

2. Phase226 的核心负结果成立：
   channel activation / hidden projection success-like
   不等于最终 readout closure。

3. Phase227 的核心推进成立：
   只看 down projection input 不够，
   必须拆 gate projection、up projection、product、down projection。

4. qwen3 的 up projection 大幅波动不能过度解释，
   product / recomputed_product 更接近真实 residual write 前态。
```

需要修正和补充的部分：

```text
1. Phase227 中 GLM4 只捕获到 product，不代表 GLM4 没有 gate/up。
   更可能是 GLM4 使用 fused gate_up_proj，需要拆分融合输出。

2. token deletion 只能证明 token-level sensitivity，
   不能直接称为严格因果来源。

3. DS7B 的巨大 rank 变化仍然只能作为敏感性证据，
   不能作为闭合证据。
```

因此 Phase228 的任务是：

```text
先校准模型 MLP module tree，
再做小幅 gate/up/product/down_out 因果 patch。
```

### 2. 本阶段脚本和结果

新增脚本：

```text
/tests/gpt5/phase228_module_tree_gateup_causal_validation.py
/tests/gpt5/run_phase228_module_tree_gateup_causal_validation.sh
```

结果目录：

```text
/tests/result/phase228_module_tree_gateup_causal_validation/module_tree_gateup_causal_validation/
```

三模型顺序测试：

```text
qwen3 → GLM4 → DS7B
```

输出规模：

```text
qwen3:
patch_rows=540

GLM4:
patch_rows=540

DS7B:
patch_rows=450

cross-model:
patch_rows=1530
channel_score_rows=1152
```

脚本检查：

```text
python -m py_compile: 通过
bash -n runner: 通过
```

### 3. 算法原理

#### 3.1 模块树审计

先检查每个模型目标层的 MLP 结构：

```text
qwen3 L29:
split_gate_up
gate_proj + up_proj + down_proj

GLM4 L30:
merged_gate_up
gate_up_proj + down_proj

DS7B L24:
split_gate_up
gate_proj + up_proj + down_proj
```

其中 GLM4 的 `gate_up_proj` 需要按最后一维拆成：

```math
\mathrm{gate}, \mathrm{up}
=
\mathrm{split}(\mathrm{gate\_up\_proj}(h))
```

#### 3.2 product 重算校准

对每个模型检查：

```math
z_{\mathrm{recompute}}
=
\phi(g) \odot u
```

是否等于真实 down projection input：

```math
z_{\mathrm{down\_input}}
```

误差：

```math
\mathrm{RelError}
=
\frac{
\|z_{\mathrm{recompute}} - z_{\mathrm{down\_input}}\|
}{
\|z_{\mathrm{down\_input}}\|+\epsilon
}
```

如果误差很小，说明 gate/up 拆分和 product 捕获是可信的。

#### 3.3 小幅因果 patch

对 success/drift 均值差分：

```math
\Delta c
=
\mu_{S,c} - \mu_{D,c}
```

其中：

```text
c ∈ {gate, up, gate_up_pair, product, down_out}
```

对 drift rows 使用：

```math
c' = c + \alpha \Delta c
```

对 success rows 使用：

```math
c' = c - \alpha \Delta c
```

其中：

```text
α ∈ {0.25, 0.5, 1.0}
```

通道范围：

```text
all
top16 positive write channels
top64 positive write channels
```

读出指标：

```text
target_rank_delta
target_logit_delta
prose_margin_delta
echo_margin_delta
top_token_changed
```

解释：

```text
drift rows:
target_logit_delta > 0 / rank_delta > 0 表示朝目标修复方向移动。

success rows:
target_logit_delta < 0 / rank_delta < 0 表示破坏成功态。
```

### 4. 模块树审计结果

#### 4.1 qwen3

```text
layer=L29
mlp_class=Qwen3MLP
mlp_type=split_gate_up

gate_proj: [9728, 2560]
up_proj:   [9728, 2560]
down_proj: [2560, 9728]
```

#### 4.2 GLM4

```text
layer=L30
mlp_class=GlmMLP
mlp_type=merged_gate_up

gate_up_proj: [27392, 4096]
down_proj:    [4096, 13696]
```

校准结论：

```text
Phase227 没捕获到 GLM4 gate/up 是工具限制，
不是模型没有 gate/up。

GLM4 的 gate/up 藏在 fused gate_up_proj 中。
```

#### 4.3 DS7B

```text
layer=L24
mlp_class=Qwen2MLP
mlp_type=split_gate_up

gate_proj: [18944, 3584]
up_proj:   [18944, 3584]
down_proj: [3584, 18944]
```

### 5. product 重算校准结果

三模型的 product 重算都高度吻合真实 down-input：

```text
qwen3:
n=4
rel_error_mean=0.001675
rel_error_max=0.001717
cosine_min=0.999998

GLM4:
n=4
rel_error_mean=0.001352
rel_error_max=0.001648
cosine_min=0.999998

DS7B:
n=4
rel_error_mean=0.001937
rel_error_max=0.002512
cosine_min=0.999997
```

这说明：

```text
1. qwen3 / DS7B 的 gate_proj + up_proj 捕获可信。
2. GLM4 的 gate_up_proj fused split 捕获可信。
3. product 确实是 gate/up 之后、down_proj 之前的真实写入前态。
```

这是 Phase228 最硬的正结果。

### 6. 因果 patch 结果

#### 6.1 qwen3 explain：product patch 有弱正向修复效果

qwen3 drift rows 中，L29 product patch 在 step=2 有最清楚的正向效果：

```text
drift product top16 alpha=1.0 step=2:
target_logit_delta=+0.9375
target_rank_delta=+98.33
top_token_changed=0/3

drift product top64 alpha=1.0 step=2:
target_logit_delta=+0.7917
target_rank_delta=+82.00
top_token_changed=2/3

drift product top16 alpha=0.5 step=2:
target_logit_delta=+0.5625
target_rank_delta=+58.67
top_token_changed=0/3
```

同一方向的 gate_up_pair 也有较弱正向效果：

```text
drift gate_up_pair top16 alpha=1.0 step=2:
target_logit_delta=+0.4583
target_rank_delta=+47.67
top_token_changed=2/3
```

解释：

```text
qwen3 L29 product delta 能因果性提高目标 logit 和目标 rank，
但多数情况下仍没有稳定跨过 top-token threshold。
```

这与 Phase225/226 一致：

```text
product/residual shift 可以改善目标排名，
但不自动完成最终读出闭合。
```

#### 6.2 qwen3 success rows：反向 patch 能破坏成功态

qwen3 success rows 中，反向 patch 在 step=2 能降低目标：

```text
success gate_up_pair all alpha=1.0:
target_logit_delta=-1.1042
target_rank_delta=-50.33

success down_out all alpha=1.0:
target_logit_delta=-0.7917
target_rank_delta=-61.67

success product all alpha=1.0:
target_logit_delta=-0.7292
target_rank_delta=-61.33
```

说明：

```text
qwen3 L29 gate_up/product/down_out 对 explain 目标状态有因果影响。
```

但它仍是局部因果，不是闭合。

#### 6.3 GLM4 repeat：gate/up 捕获已修复，但 patch 效果弱且方向不稳定

GLM4 drift rows 中，正向 logit 改善很小：

```text
drift gate top16 alpha=0.25 step=1:
target_logit_delta=+0.0208
target_rank_delta=0

drift up top16 alpha=0.25 step=2:
target_logit_delta=+0.0208
target_rank_delta=-0.33

drift gate_up_pair top16 alpha=0.25 step=2:
target_logit_delta=+0.0208
target_rank_delta=-0.33
```

更大 alpha 下，很多 patch 反而降低 target logit：

```text
drift gate_up_pair all alpha=1.0 step=2:
target_logit_delta=-0.5625
target_rank_delta=-9.67

drift product all alpha=1.0 step=2:
target_logit_delta=-0.3750
target_rank_delta=-7.67
```

解释：

```text
GLM4 L30 repeat 的 fused gate/up/product 已经可观测，
但 success-drift delta 不是稳定修复方向。
```

这说明 Phase226 中看到的 Answer anchor 强效，更可能是：

```text
prompt-level readout regime / continuation regime 触发
```

而不是单靠 L30 product delta 就能修复 repeat。

#### 6.4 DS7B：数值反应强，但稳定性仍不足

DS7B drift rows 中有正向 logit 效果：

```text
drift up all alpha=1.0 step=2:
target_logit_delta=+0.9805
target_rank_delta=+9581

drift gate_up_pair all alpha=1.0 step=1:
target_logit_delta=+0.4063
target_rank_delta=+2509

drift product all alpha=1.0 step=1:
target_logit_delta=+0.2148
target_rank_delta=+1944
```

但 success rows 反向 patch 的破坏极强：

```text
success gate all alpha=1.0 step=2:
target_logit_delta=-4.2813
target_rank_delta=-19845.67

success product all alpha=1.0 step=2:
target_logit_delta=-3.5000
target_rank_delta=-30977.33

success down_out all alpha=1.0 step=2:
target_logit_delta=-3.4531
target_rank_delta=-30359.67
```

这说明：

```text
DS7B 的 L24 MLP 内部组件对目标读出高度敏感。
```

但因为：

```text
1. drift rows 只有 2 条；
2. rank_delta 巨大；
3. 此前 DS7B 行为闭合弱；
```

所以仍只能作为敏感性证据，不能作为稳定机制闭合。

### 7. 本阶段核心进展

核心进展 1：

```text
GLM4 gate/up 捕获问题已解决：
GLM4 是 merged_gate_up，不是没有 gate/up。
```

核心进展 2：

```text
三模型 product 重算均与真实 down-input 高度一致。
product 是可信的 residual write 前态。
```

核心进展 3：

```text
qwen3 L29 product patch 对 explain drift 有弱正向因果效果：
target logit 和 rank 均改善。
```

核心进展 4：

```text
success rows 的反向 patch 可以破坏目标状态，
说明 gate/up/product/down_out 不是旁观变量，而是因果链条的一部分。
```

核心负结果：

```text
GLM4 和 DS7B 的 success-drift product delta 不能稳定解释为通用修复方向。
```

因此当前还不能写成：

```text
PromptTrigger → gate/up/product → correct output
```

只能写成：

```text
PromptTrigger → gate/up/product → local readout pressure
```

其中 final output 仍受：

```text
ReadoutRegimeSelection
TopTokenThreshold
CompetitorTransition
```

控制。

### 8. 问题、硬伤和瓶颈

#### 8.1 patch 是局部线性近似

本阶段 patch 是：

```math
c' = c + \alpha(\mu_S-\mu_D)
```

这仍然是假设均值差分可以代表自然因果方向。

问题：

```text
真实运行可能不是线性差分；
尤其 gate/up/product 是非线性耦合。
```

因此 patch 有用只能说明“相关方向有局部因果效应”，不能说明它就是自然算法。

#### 8.2 success-drift 均值可能混合多个子模式

如果 success rows 内部有多个模式：

```text
success mode A
success mode B
success mode C
```

那么：

```math
\mu_S-\mu_D
```

可能不是任何真实运行路径，而是混合平均方向。

GLM4 的弱/反向 patch 很可能与此有关。

#### 8.3 top token 仍很少闭合

qwen3 product patch 能改善 logit/rank，但：

```text
top_token_changed 很少稳定变成目标 token。
```

这继续证明：

```text
ReadoutRegimeSelection 是独立瓶颈。
```

#### 8.4 DS7B 仍有小模型粗糙性

DS7B 的 rank 变化过大，说明内部编码可能更粗糙。

分析 DS7B 时应保留：

```text
30% 到 50% 偏差空间
```

### 9. 当前完整图谱更新

```text
PromptToken / PromptSpan
→ InstructionFrame
→ AnswerAnchor
→ StepCondition
→ gate projection
→ up projection
→ gated product
→ down projection
→ residual write
→ local readout pressure
→ readout regime selection
→ top-token threshold
→ output pattern
```

其中 Phase228 确认较强的是：

```text
gate/up → product → down_out
```

以及：

```text
product/down_out 对 local readout pressure 有因果影响。
```

仍未确认的是：

```text
local readout pressure 如何切换到正确 readout regime。
```

### 10. 当前进度估计

```text
小模型模式机制图谱：约 75%
StatePath：约 61%
StateWriteCause：约 45%
StateWriteSource：约 39%
NaturalTrigger → ChannelActivation：约 20%
TokenTrigger → Gate/Up/Product：约 16%
MLP/ResidualWrite：约 54%
activation-gated channel mechanism：约 27%
ResidualPropagation：约 21%
ReadoutPath：约 34%
ReadoutRegimeSelection：约 11%
路径因果机制：约 40%
模型内部自然闭合：约 42%
任务层产品闭合：约 55%
通用语言机制外推置信：约 40% 到 45%
```

### 11. 阶段边界判断和下一步

Phase228 仍属于 Phase226/227 的同一大阶段：

```text
自然触发源 → MLP product → residual write 的机制图谱。
```

但 Phase228 已经完成了该阶段的工具校准和第一轮因果验证。

下一步如果继续同阶段，不应再做普通 gate/up/product patch，而应转向 readout regime 的来源：

```text
Phase229: readout regime selection source atlas
（读出机制选择来源图谱）
```

核心任务：

```text
1. 固定 qwen3 L29 product patch 有效样本，
   追踪为什么 rank/logit 改善仍不跨 top token threshold。

2. 对 Then / The / For / Answer 等 competitor token 建立 readout competitor atlas。

3. 区分：
   target pressure
   prose pressure
   continuation pressure
   echo/repeat pressure
   answer-boundary pressure。

4. 不再以“目标 token 是否 top1”作为唯一指标，
   而是画出多个读出机制之间的竞争图谱。
```

## Phase 229: 读出机制选择来源图谱 [2026-07-07 03:08]

### 1. 对 Phase228 附件判断的校准

附件对 Phase228 的判断基本正确。

Phase228 的性质应定义为：

```text
工具校准 + 局部因果验证
```

它完成了两个硬进展：

```text
1. GLM4 的 gate/up 捕获问题被修复：
   GLM4 是 merged_gate_up，不是没有 gate/up。

2. 三模型 product recompute 与真实 down-input 高度一致：
   product 是 residual write 前态。
```

但附件指出的一个问题也成立：

```text
Phase226 / Phase228 的 rank_delta 文字说明容易混淆。
```

本阶段统一记法为：

```math
\Delta r_{\mathrm{improve}}
=
r_{\mathrm{base}} - r_{\mathrm{new}}
```

因此：

```text
rank_improve > 0 表示目标排名改善。
rank_improve < 0 表示目标排名变差。
```

### 2. 本阶段目标

Phase228 已经证明：

```text
gate/up/product → residual write → local readout pressure
```

但没有闭合：

```text
local readout pressure → correct output
```

Phase229 的任务是把 readout regime selection（读出机制选择）拆开，不再只看 top1 是否目标词，而是观察目标词被哪些机制压住：

```text
Then / The / For / Answer / Because / comma / period / newline / echo / prose / be-continuation
```

### 3. 脚本和结果

新增脚本：

```text
/tests/gpt5/phase229_readout_regime_selection_atlas.py
/tests/gpt5/run_phase229_readout_regime_selection_atlas.sh
```

结果目录：

```text
/tests/result/phase229_readout_regime_selection_atlas/readout_regime_selection_atlas/
```

三模型顺序测试：

```text
qwen3 → GLM4 → DS7B
```

结果规模：

```text
qwen3: regime_rows=432
GLM4: regime_rows=432
DS7B: regime_rows=324
cross-model: regime_rows=1188
```

脚本检查：

```text
python -m py_compile: 通过
bash -n runner: 通过
```

### 4. 算法原理

对每个上下文或 patch 结果，计算：

```text
target_logit
target_rank
top_token
winning_regime
target_margin_vs_winner
```

其中：

```math
\mathrm{TargetMargin}
=
\mathrm{logit}_{target}
-
\max_{r \in R, r \ne target} \mathrm{logit}_r
```

如果：

```text
target_logit_delta > 0
rank_improve > 0
target_margin_vs_winner < 0
```

说明：

```text
目标压力已经增强，但仍没有跨过读出机制 winner。
```

这就是 Phase225 到 Phase229 一直追踪的关键断点。

读出机制集合：

```text
target
then_continuation
the_continuation
for_continuation
answer_boundary
because_reason
be_continuation
comma_repeat
period_stop
colon_boundary
newline_boundary
space_boundary
prose
echo
stop
```

### 5. 关键结果

#### 5.1 qwen3：目标压力增强后常被 period / because / comma / echo 压住

qwen3 drift 中，product patch 能改善目标，但 winner 仍是 period_stop：

```text
drift patch_product_top16_a1 step=2:
target_logit_delta=+0.8438
rank_improve=+93.5
target_margin_vs_winner=-20.3438
winner=period_stop
top_token=.\n
```

同一类结果：

```text
patch_product_top64_a1 step=2:
target_logit_delta=+0.6562
rank_improve=+76.0
target_margin_vs_winner=-20.4688
winner=period_stop

patch_gate_up_pair_top64_a1 step=2:
target_logit_delta=+0.8750
rank_improve=+124.0
target_margin_vs_winner=-21.8750
winner=period_stop
```

qwen3 drift 在 step=3 常被 because_reason 压住：

```text
patch_product_top16_a1 step=3:
target_logit_delta=+0.4375
rank_improve=+170.0
target_margin_vs_winner=-28.2500
winner=because_reason
top_token=Because
```

解释：

```text
qwen3 的 product patch 确实增强目标压力，
但 period_stop 和 because_reason 的读出机制仍大幅领先。
```

这解释了为什么 Phase228 看到 rank/logit 改善，却很少 top1 闭合。

#### 5.2 qwen3 自然变体可强烈切换读出机制

qwen3 drift repeat_instruction：

```text
step=3:
target_logit_delta=+5.6562
rank_improve=+620.5
target_margin_vs_winner=-5.9688
winner=answer_boundary
top_token=Answer
```

qwen3 drift no_answer_anchor：

```text
step=2:
target_logit_delta=+3.8125
rank_improve=+243.0
target_margin_vs_winner=-10.0000
winner=period_stop
top_token=.

step=3:
target_logit_delta=+2.4375
rank_improve=+444.5
target_margin_vs_winner=-16.1875
winner=because_reason
top_token=Because
```

说明：

```text
自然 prompt 变体能同时增强 target pressure 和切换 readout regime。
但 regime winner 仍可能不是 target。
```

#### 5.3 GLM4：Answer anchor 移除稳定切到 For continuation

GLM4 repeat 中，no_answer_anchor 的结果非常稳定：

```text
success no_answer_anchor step=1:
target_logit_delta=-8.4224
rank_improve=-5467.0
target_margin_vs_winner=-12.4849
winner=for_continuation
top_token=For

drift no_answer_anchor step=1:
target_logit_delta=-7.2902
rank_improve=-6151.25
target_margin_vs_winner=-12.6027
winner=for_continuation
top_token=For
```

这把 Phase226/227 的判断进一步收紧：

```text
GLM4 repeat 的 Answer anchor 不是单纯通道触发；
它还强控制 readout regime selection。
```

GLM4 中，explain_instruction 会把 winner 切到 because_reason 或 newline_boundary：

```text
success explain_instruction step=3:
winner=because_reason
top_token=because

success explain_instruction step=2:
winner=newline_boundary
top_token=\n
```

#### 5.4 DS7B：目标压力和排名变化很大，但 winner 通常仍是 be/echo/newline

DS7B drift product patch：

```text
patch_product_top64_a1 step=3:
target_logit_delta=+1.3750
rank_improve=+322.0
target_margin_vs_winner=-16.9375
winner=be_continuation
top_token=are
```

DS7B natural repeat_instruction：

```text
drift repeat_instruction step=2:
target_logit_delta=+1.1055
rank_improve=+14899.0
target_margin_vs_winner=-11.2148
winner=newline_boundary
top_token=orses
```

说明：

```text
DS7B 的目标压力变化很大，
但读出 winner 仍经常是 be_continuation / echo / newline_boundary。
```

由于 DS7B 样本少且 rank 变化巨大，仍只作为敏感性参考。

### 6. Phase229 结论

Phase229 的核心结果：

```text
local readout pressure 和 readout regime winner 是两层不同机制。
```

更具体：

```text
1. qwen3:
   product patch 增强 target pressure，
   但 period_stop / because_reason / echo 仍常压过 target。

2. GLM4:
   Answer anchor 强控制 for_continuation / repeat regime 的切换。

3. DS7B:
   目标压力敏感，但 winner 常被 be / echo / newline 控制。
```

因此当前图谱应更新为：

```text
product/down_out
→ target pressure
→ competitor pressure field
→ readout regime winner
→ top-token threshold
```

不能再写成：

```text
product/down_out → target token
```

## Phase 230: 读出阈值障碍量化 [2026-07-07 03:09]

### 1. 阶段目标

Phase229 画出了读出机制 winner。

Phase230 不再重新跑模型，而是基于 Phase229 的 1188 条记录量化：

```text
目标压力已经增加，但距离跨过 winner 还差多少。
```

这一步仍属于同一阶段，因为它继续处理：

```text
local readout pressure → top-token threshold
```

### 2. 脚本和结果

新增脚本：

```text
/tests/gpt5/phase230_readout_threshold_barrier_analysis.py
```

结果目录：

```text
/tests/result/phase230_readout_threshold_barrier_analysis/readout_threshold_barrier_analysis/
```

输入：

```text
Phase229 regime_rows=1188
```

输出：

```text
barrier_rows=321
closure_candidate_rows=59
```

其中 barrier row 定义为：

```text
target_logit_delta > 0
target_margin_vs_winner < 0
```

即：

```text
目标压力增加了，但仍未跨过 winner。
```

### 3. 公式

剩余阈值障碍：

```math
\mathrm{RemainingGap}
=
-
\mathrm{TargetMargin}
```

其中：

```math
\mathrm{TargetMargin}
=
\mathrm{logit}_{target}
-
\mathrm{logit}_{winner}
```

压力效率：

```math
\mathrm{PressureEfficiency}
=
\frac{
\Delta \mathrm{TargetMargin}
}{
\Delta \mathrm{TargetLogit}
}
```

解释：

```text
RemainingGap 越大，说明目标距离 winner 越远。
PressureEfficiency 越高，说明目标 logit 增加更有效地缩小了与 winner 的差距。
```

### 4. 关键 barrier 结果

#### 4.1 qwen3: because_reason 是最大阈值障碍之一

qwen3 drift step=3：

```text
patch_product_top16_a1:
target_logit_delta=+0.4375
rank_improve=+170.0
winner=because_reason
remaining_gap=28.25
top_token=Because
```

同类结果：

```text
patch_product_top16_a0.5:
target_logit_delta=+0.3125
rank_improve=+89.0
remaining_gap=29.375
winner=because_reason

patch_gate_up_pair_top16_a1:
target_logit_delta=+0.2500
rank_improve=+121.0
remaining_gap=28.6875
winner=because_reason
```

说明：

```text
qwen3 explain 中，because_reason 是非常强的后续解释机制。
即使 product patch 提升目标，也很难越过 because readout。
```

#### 4.2 qwen3: period_stop 是 step=2 的主要障碍

qwen3 drift step=2：

```text
patch_product_top16_a1:
target_logit_delta=+0.8438
rank_improve=+93.5
winner=period_stop
remaining_gap=20.3438
top_token=.\n

patch_product_top64_a1:
target_logit_delta=+0.6562
rank_improve=+76.0
winner=period_stop
remaining_gap=20.4688
```

说明：

```text
step=2 的失败不是目标完全没动，
而是 period_stop 阈值障碍太强。
```

#### 4.3 DS7B: be_continuation 是常见障碍

DS7B drift step=3：

```text
patch_product_top64_a1:
target_logit_delta=+1.3750
rank_improve=+322.0
winner=be_continuation
remaining_gap=16.9375
top_token=are
```

多个 gate_up/product patch 都落在：

```text
winner=be_continuation
remaining_gap≈17
```

说明：

```text
DS7B 的目标压力增长常被 be_continuation 机制压住。
```

#### 4.4 GLM4: barrier 更少，主要问题是 regime switch

GLM4 的 barrier_rows 明显少于 qwen3 / DS7B。

这不是说明 GLM4 更闭合，而是说明：

```text
GLM4 经常不是“目标压力增加但还差一点”，
而是直接被 no_answer_anchor / explain_instruction 切换到 For / newline / because 等 regime。
```

例如：

```text
no_answer_anchor step=1:
winner=for_continuation
target_logit_delta 大幅下降
```

这属于：

```text
regime switch failure
```

而不是：

```text
threshold barrier failure
```

### 5. 本阶段新增拼图

新增拼图 1：

```text
读出失败至少分两类：

1. threshold barrier failure:
   target pressure 增强，但 winner 仍领先。

2. regime switch failure:
   prompt/anchor 改变后，模型直接切到另一个读出机制。
```

新增拼图 2：

```text
qwen3 explain 的主要 threshold barrier:
period_stop
because_reason
echo
```

新增拼图 3：

```text
GLM4 repeat 的主要失败:
For continuation / newline / because 的 regime switch。
```

新增拼图 4：

```text
DS7B explain 的主要 threshold barrier:
be_continuation
echo
newline_boundary。
```

### 6. 问题和硬伤

#### 6.1 regime groups 仍是人工定义

当前 Then / The / For / Answer / because 等组是人工指定。

它能解释当前现象，但还不是自动发现的完整读出机制图谱。

#### 6.2 winner 是词元级 winner，不等于完整生成轨迹

本阶段只看 next-token readout。

真实输出模式还包括：

```text
后续多 token rollout
```

所以 Phase230 仍不是最终输出闭合。

#### 6.3 barrier 不能直接用 logit 差线性修复

RemainingGap 很大时，不能简单认为补一个等量 logit 就能闭合。

因为补 target 可能同时改变 competitor field。

### 7. 当前图谱更新

```text
PromptTrigger
→ gate/up/product
→ residual write
→ local target pressure
→ competitor pressure field
→ readout regime winner
→ threshold barrier / regime switch
→ top token
→ rollout pattern
```

### 8. 当前进度估计

```text
小模型模式机制图谱：约 76%
StatePath：约 62%
StateWriteCause：约 46%
StateWriteSource：约 40%
NaturalTrigger → ChannelActivation：约 21%
TokenTrigger → Gate/Up/Product：约 17%
MLP/ResidualWrite：约 55%
ReadoutPressure：约 22%
ReadoutRegimeSelection：约 17%
TopTokenThreshold：约 19%
路径因果机制：约 41%
模型内部自然闭合：约 43%
任务层产品闭合：约 55%
通用语言机制外推置信：约 40% 到 45%
```

### 9. 阶段边界判断

Phase229 和 Phase230 仍属于同一大阶段：

```text
自然触发源 → MLP product → residual write → readout pressure → readout regime
```

本阶段已经完成：

```text
1. readout regime 初步图谱；
2. threshold barrier 初步量化；
3. threshold barrier failure 与 regime switch failure 的区分。
```

下一步如果继续，应进入更具体的因果验证：

```text
Phase231: competitor pressure causal suppression
（竞争读出压力因果抑制验证）
```

但这已经是新的子阶段：

```text
从“画出读出竞争图谱”
转向“因果抑制竞争机制”。
```

因此本轮自动推进到 Phase230 后可以收束，不继续无限增加 patch。

## Phase 231: 竞争读出压力理想抑制预算 [2026-07-07 03:21]

### 1. 本阶段任务

本阶段继续分析 Phase229/230 的结果是否正确，并把正确部分向前推进一步。

Phase229/230 的核心判断是正确的：

```text
target logit 增强
target rank 改善
不等于 target token 成为 top token。
```

更准确地说，读出失败至少分为两类：

```text
1. threshold barrier failure:
   目标压力增强，但竞争读出机制仍然领先。

2. regime switch failure:
   prompt 或 anchor 改变后，模型直接进入另一套读出机制。
```

Phase231 不继续盲目增加 product patch，而是检查一个更直接的问题：

```text
如果只在读出层压低当前 winner，需要多大竞争抑制预算，
target margin 才能过零？
```

注意：本阶段是 readout-level oracle suppression（读出层理想抑制）测试，不等于已经找到模型内部真实抑制通道。

### 2. 测试脚本和结果文件

脚本：

```text
tests/gpt5/phase231_competitor_pressure_oracle_suppression.py
```

输入：

```text
tests/result/phase230_readout_threshold_barrier_analysis/readout_threshold_barrier_analysis/
```

输出：

```text
tests/result/phase231_competitor_pressure_oracle_suppression/competitor_pressure_oracle_suppression/
```

本阶段没有重新加载模型，因为它复用 Phase229/230 已经完成的 qwen3、GLM4、DS7B 三模型读出结果。

总数据量：

```text
input_barrier_rows = 321
suppression_rows = 3210
```

### 3. 算法原理

Phase230 已经给出每条 threshold barrier row：

```text
target_margin_vs_winner < 0
target_logit_delta > 0
```

其中：

```text
remaining_margin_gap = - target_margin_vs_winner
```

Phase231 假设对当前 winning_regime 施加一个读出层理想抑制预算：

```text
b ∈ {1,2,4,8,12,16,20,24,28,32}
```

核心公式：

$$
\mathrm{PostMargin}(b)
=
\mathrm{TargetMargin}
+
b
$$

闭合判定：

$$
\mathrm{OracleClose}(b)
=
\mathbf{1}
\left[
\mathrm{PostMargin}(b) \ge 0
\right]
$$

也就是：

```text
如果只压低当前 winner 的读出压力 b 个 logit 单位，
target 是否能超过 winner。
```

这个测试的意义不是证明真实模型可以线性抑制，而是量化：

```text
当前失败到底是小阈值差，还是巨大竞争场。
```

### 4. 关键结果

#### 4.1 qwen3

qwen3 的 threshold barrier rows：

```text
rows = 188
```

整体预算闭合率：

```text
b=8:  closed 48 / 188, closure_rate=0.2553
b=16: closed 122 / 188, closure_rate=0.6489
b=20: closed 144 / 188, closure_rate=0.7660
b=24: closed 170 / 188, closure_rate=0.9043
b=28: closed 170 / 188, closure_rate=0.9043
b=32: closed 188 / 188, closure_rate=1.0000
```

主要竞争机制：

```text
success / echo:
rows=52
median_remaining_gap=12.6250

drift / period_stop:
rows=40
median_remaining_gap=20.5000
p90_remaining_gap=22.6875

drift / because_reason:
rows=24
median_remaining_gap=29.1875
p90_remaining_gap=29.8125
```

解释：

```text
qwen3 的 period_stop 和 because_reason 不是小障碍。
尤其 because_reason，需要接近 30 个 logit 级别的 winner 抑制才稳定过阈值。
```

因此，qwen3 中单纯继续增强 target pressure 很可能进入边际收益递减区。
下一步更应该找：

```text
because_reason / period_stop 的自然触发源和写入路径。
```

#### 4.2 GLM4

GLM4 的 threshold barrier rows 很少：

```text
rows = 13
```

整体预算闭合率：

```text
b=1: closed 7 / 13, closure_rate=0.5385
b=2: closed 9 / 13, closure_rate=0.6923
b=4: closed 11 / 13, closure_rate=0.8462
b=8: closed 13 / 13, closure_rate=1.0000
```

主要 gap：

```text
drift / the_continuation:
rows=4
median_remaining_gap=1.7656

drift / because_reason:
rows=4
median_remaining_gap=0.8750

drift / comma_repeat:
rows=2
median_remaining_gap=4.5625
```

解释：

```text
GLM4 在 threshold barrier 行里并不厚。
它的问题更像 Phase230 判断的 regime switch failure：
模型经常不是“差一点过不了阈值”，
而是直接切到 For / newline / because 等读出机制。
```

所以 GLM4 后续不应优先做大幅 competitor suppression，而应优先研究：

```text
prompt anchor 如何触发 regime switch。
```

#### 4.3 DS7B

DS7B 的 threshold barrier rows：

```text
rows = 120
```

整体预算闭合率：

```text
b=8:  closed 0 / 120, closure_rate=0.0000
b=12: closed 10 / 120, closure_rate=0.0833
b=16: closed 78 / 120, closure_rate=0.6500
b=20: closed 118 / 120, closure_rate=0.9833
b=24: closed 120 / 120, closure_rate=1.0000
```

主要竞争机制：

```text
drift / be_continuation:
rows=28
median_remaining_gap=17.3438

drift / the_continuation:
rows=28
median_remaining_gap=12.4043

success / be_continuation:
rows=20
median_remaining_gap=15.3438

success / prose:
rows=20
median_remaining_gap=15.3066
```

解释：

```text
DS7B 的主要障碍不是 because_reason，而是 be_continuation / the_continuation / prose 续写场。
```

这说明三模型读出竞争结构不同：

```text
qwen3: period_stop / because_reason 很强。
GLM4: threshold barrier 薄，regime switch 更关键。
DS7B: be/the/prose continuation field 更强。
```

### 5. 本阶段新增拼图

新增拼图 1：

```text
读出失败不是一个统一障碍。
不同模型的竞争读出场形状不同。
```

新增拼图 2：

```text
qwen3 的 because_reason 是大阈值障碍，
不是小幅 target pressure patch 能自然闭合的对象。
```

新增拼图 3：

```text
GLM4 的 threshold barrier 很薄，
说明它的主要失败更可能在 regime switch 层，
不是 winner margin 层。
```

新增拼图 4：

```text
DS7B 的 continuation field 更强，
尤其 be_continuation / the_continuation / prose。
```

新增拼图 5：

```text
competitor suppression 的下一步不能直接假设全局有效，
必须按 winner regime 分别找内部来源。
```

### 6. 问题、硬伤和边界

#### 6.1 这是 oracle，不是真实内部抑制

本阶段只在读出层做理想预算：

```text
PostMargin = TargetMargin + b
```

真实模型内部抑制某个通道时，可能同时改变：

```text
target pressure
competitor pressure
其他读出机制
后续 rollout
```

所以不能把本阶段当成真实因果闭合。

#### 6.2 只压 winner 可能过于理想

真实读出竞争不是只存在一个 winner。

压低当前 winner 后，可能出现第二个 competitor 继续压住 target。

因此 Phase231 的 closure_rate 是上界，不是实际干预成功率。

#### 6.3 仍是 next-token 读出，不是完整输出轨迹

本阶段仍然只处理 top token 阈值，不处理后续多 token 生成轨迹。

### 7. 理论进展

当前更稳妥的机制链条是：

```text
PromptPattern
→ Gate/Up/Product
→ ResidualWrite
→ TargetPressure
→ CompetitorPressureField
→ ReadoutRegimeSelection
→ ThresholdBarrier
→ TopToken
→ RolloutPattern
```

Phase231 对这条链的贡献是：

```text
把 CompetitorPressureField 的强度第一次按模型和 winner regime 量化。
```

这比继续讨论“目标向量是否存在”更接近真实编码机制，因为它承认：

```text
语言输出不是单一目标方向，
而是多个模式场同时竞争。
```

### 8. 下一步任务

Phase231 仍属于：

```text
readout pressure / readout regime / threshold barrier
```

这一大阶段。

但下一步应从读出层 oracle 进入内部来源验证：

```text
Phase232: competitor source localization
（竞争读出压力来源定位）
```

目标不是继续调 target patch，而是分别定位：

```text
qwen3:
because_reason / period_stop 的内部来源。

GLM4:
regime switch 的 prompt anchor 来源。

DS7B:
be_continuation / the_continuation / prose 的内部来源。
```

优先级：

```text
1. 先定位 competitor source；
2. 再做 source-level causal suppression；
3. 最后才回到 target closure。
```

这样可以避免继续在线性补丁上做边际收益递减的 patch。

### 9. 阶段结论

Phase231 证明：

```text
Phase229/230 的判断基本正确。
```

更具体地说：

```text
qwen3 的失败是厚阈值障碍；
GLM4 的失败主要是机制切换；
DS7B 的失败是 continuation field 压制。
```

所以，破解语言编码机制的下一块拼图不是“再增强目标词”，而是：

```text
找出不同 competitor regime 的自然触发源和内部写入路径。
```

这一步如果完成，才有可能从：

```text
读出层图谱
```

推进到：

```text
真实内部因果机制图谱。
```

## Phase 232: 竞争读出来源候选定位 [2026-07-07 03:34]

### 1. 本阶段任务

本阶段分析 Phase231 的判断是否正确，并继续推进同一阶段任务。

Phase231 的方向是正确的：

```text
target logit 增强
target rank 改善
不等于 target token 胜出；
读出失败需要区分 target pressure 和 competitor pressure。
```

但 Phase231 只是 readout-level oracle suppression（读出层理想抑制），不能说明模型内部哪里产生了 because_reason、period_stop、for_continuation、be_continuation 等竞争压力。

因此 Phase232 的目标是：

```text
从 Phase229 的自然 prompt 变体和 MLP patch 变体中，
定位 competitor regime 的来源候选。
```

注意：本阶段仍不是内部因果闭合，而是 source candidate localization（来源候选定位）。

### 2. 脚本和结果

脚本：

```text
tests/gpt5/phase232_competitor_source_localization.py
```

输入：

```text
tests/result/phase229_readout_regime_selection_atlas/readout_regime_selection_atlas/
```

输出：

```text
tests/result/phase232_competitor_source_localization/competitor_source_localization/
```

数据量：

```text
input_rows = 1188
pressure_source_rows = 4536
switch_source_rows = 51
coupling_rows = 3024
priority_rows = 160
```

本阶段没有重新加载模型，原因是 Phase232 是对 Phase229 已经完成的三模型读出行进行来源候选分析。

### 3. 算法原理

Phase229 每一行已经包含：

```text
regime_delta[r]
target_logit_delta
winning_regime
base_winning_regime
winning_regime_changed
variant
step
component
channel_scope
```

Phase232 对每个候选 regime 计算三类指标。

#### 3.1 pressure source（压力来源）

对某个 regime：

$$
\Delta z_r
=
z_r^{variant}
-
z_r^{base}
$$

竞争相对目标优势：

$$
\mathrm{CompAdv}_r
=
\Delta z_r
-
\Delta z_{target}
$$

如果：

```text
CompAdv_r 明显为正
```

说明该 prompt 变体或 patch 变体更偏向抬高 competitor，而不是抬高 target。

#### 3.2 switch source（切换来源）

如果：

$$
\mathrm{Winner}_{base}
\ne
\mathrm{Winner}_{variant}
$$

则记录：

```text
base_winning_regime → winning_regime
```

这用于定位 regime switch failure 的来源。

#### 3.3 patch coupling（补丁耦合）

对 MLP patch 行计算：

```text
target_delta > 0 且 competitor_delta > 0
```

如果 target 和 competitor 同涨，说明当前 patch 不是纯 target 修复，而是可能同时增强某个竞争机制。

### 4. 关键结果

#### 4.1 GLM4：regime switch 来源最清楚

GLM4 的来源候选最强，且主要来自自然 prompt 变体。

最强结果：

```text
success / no_answer_anchor / step=1
regime = for_continuation
rows = 4
changed_to_rate = 1.0000
winner_rate = 1.0000
mean_target_delta = -8.4224
mean_regime_delta = +9.0820
mean_competitor_minus_target_delta = +17.5044
top_token = For
```

drift 组同样成立：

```text
drift / no_answer_anchor / step=1
regime = for_continuation
rows = 4
changed_to_rate = 1.0000
winner_rate = 1.0000
mean_target_delta = -7.2902
mean_regime_delta = +8.8633
mean_competitor_minus_target_delta = +16.1535
top_token = For
```

这说明：

```text
GLM4 去掉 AnswerAnchor 后，
不是薄阈值失败，
而是强烈切换到 For continuation。
```

另外：

```text
success / explain_instruction / step=3
regime = because_reason
rows = 4
changed_to_rate = 1.0000
winner_rate = 1.0000
mean_target_delta = -4.9531
mean_regime_delta = +8.5781
top_token = because
```

以及：

```text
success / explain_instruction / step=2
regime = newline_boundary
rows = 4
changed_to_rate = 1.0000
winner_rate = 1.0000
mean_target_delta = -6.0078
mean_regime_delta = +3.6719
top_token = newline
```

GLM4 的结论比较稳定：

```text
AnswerAnchor 控制 repeat regime；
no_answer_anchor 触发 For continuation；
explain_instruction 触发 newline / because。
```

这与 Phase230/231 的判断一致：GLM4 不是厚 threshold barrier 为主，而是 regime switch 为主。

#### 4.2 qwen3：because/period 来源更分散

qwen3 的强候选主要来自自然 prompt 变体，但不是直接 winner switch，而是 competitor 相对 target 的压力优势。

代表结果：

```text
drift / no_answer_anchor / step=1
regime = because_reason
rows = 4
winner_rate = 0.0000
mean_target_delta = -12.0625
mean_regime_delta = +4.4688
mean_competitor_minus_target_delta = +16.5312
top_token = The / Then
```

同一条件下 period_stop 也增强：

```text
drift / no_answer_anchor / step=1
regime = period_stop
rows = 4
mean_target_delta = -12.0625
mean_regime_delta = +0.6719
mean_competitor_minus_target_delta = +12.7344
```

另外：

```text
drift / because_removed / step=2
regime = period_stop
rows = 4
winner_rate = 1.0000
mean_target_delta = -1.2812
mean_regime_delta = +2.5000
top_token = . / .\n
```

这说明：

```text
qwen3 的 because_reason / period_stop 不是单一 prompt 开关。
no_answer_anchor 会削弱 target 并相对增强 because/period；
because_removed 反而会让 period_stop 直接成为 winner。
```

qwen3 的一个重要正面切换：

```text
drift / repeat_instruction / step=3
base_winning_regime = because_reason
winning_regime = answer_boundary
rows = 4
mean_target_delta = +5.6562
mean_margin_delta_vs_winner = +24.2188
top_token = Answer
```

但它仍没有完成最终目标闭合，因为：

```text
target_margin_vs_winner = -5.9688
```

解释：

```text
repeat_instruction 可以把 qwen3 从 because_reason 拉到 answer_boundary，
但 answer_boundary 不是目标答案本身。
```

这说明 qwen3 的下一步不能只找 because 抑制，还要区分：

```text
answer boundary
target answer
period stop
because reason
```

四者之间的竞争关系。

#### 4.3 DS7B：continuation 来源仍弱，样本权重较低

DS7B 的候选结果明显更弱，且每桶 rows 多为 2 或 4。

可观察结果：

```text
drift / patch_product_top64_a1 / step=2
regime = be_continuation
rows = 2
mean_target_delta = -1.6484
mean_regime_delta = +2.8906
mean_competitor_minus_target_delta = +4.5391
top_token = orses
```

类似结果：

```text
drift / patch_product_all_a1 / step=2
regime = be_continuation
rows = 2
mean_target_delta = -1.0078
mean_regime_delta = +3.0156
mean_competitor_minus_target_delta = +4.0234
```

以及：

```text
drift / patch_gate_up_pair_top64_a1 / step=2
regime = be_continuation
rows = 2
mean_target_delta = -1.5078
mean_regime_delta = +2.6094
mean_competitor_minus_target_delta = +4.1172
```

解释：

```text
DS7B 的 be_continuation 可能和 product / gate_up patch 有耦合，
但当前样本太少，只能作为低权重候选。
```

DS7B 的 echo 候选也存在：

```text
drift / no_instruction / step=1
regime = echo
rows = 2
changed_to_rate = 1.0000
winner_rate = 1.0000
mean_regime_delta = +2.1250
```

但仍需加大样本后再判断。

### 5. 本阶段新增拼图

新增拼图 1：

```text
GLM4 的主要失败来源从读出层进一步定位到 prompt anchor：
no_answer_anchor → For continuation；
explain_instruction → newline / because。
```

新增拼图 2：

```text
qwen3 的 because/period 不是简单开关，
而是多 prompt 条件共同改变 target 与 competitor 的相对压力。
```

新增拼图 3：

```text
qwen3 repeat_instruction 可以把 because_reason winner 拉到 answer_boundary，
但 answer_boundary 不等于 target answer。
```

新增拼图 4：

```text
DS7B 的 be_continuation 与 product/gate_up patch 有候选耦合，
但当前样本量不足，只能低权重记录。
```

新增拼图 5：

```text
competitor source localization 必须区分三类来源：
prompt source；
MLP patch coupling source；
winner switch source。
```

### 6. 问题和硬伤

#### 6.1 每桶样本量偏小

很多候选每桶只有：

```text
rows = 2 到 4
```

所以 Phase232 是候选定位，不是强统计结论。

#### 6.2 来源定位仍停留在读出结果层

本阶段没有重新 hook 模型内部激活。

因此当前只能说：

```text
某些 prompt/patch 条件与 competitor pressure 上升或 winner switch 有关。
```

还不能说：

```text
某个 layer/channel/head 是 because_reason 或 period_stop 的真实来源。
```

#### 6.3 regime group 仍是人工集合

because_reason、period_stop、for_continuation 等仍是人工词元组。

这适合当前机制追踪，但还不是自动发现的完整读出图谱。

#### 6.4 qwen3 的来源关系复杂

qwen3 中：

```text
no_answer_anchor 会相对增强 because/period；
because_removed 会触发 period_stop；
repeat_instruction 会拉到 answer_boundary。
```

这说明 qwen3 的读出机制不是单一 competitor source，而是多个边界机制联动。

### 7. 理论进展

Phase232 对机制链条的更新是：

```text
PromptPattern
→ PromptAnchor / InstructionFrame
→ Gate/Up/Product
→ ResidualWrite
→ TargetPressure
→ CompetitorPressureField
→ ReadoutRegimeSwitch
→ ThresholdBarrier
→ TopToken
```

核心进展不是提出新理论，而是把：

```text
CompetitorPressureField
```

进一步拆成：

```text
prompt source
patch coupling source
winner switch source
```

更谨慎的公式：

$$
\mathrm{CompetitorSource}_r
=
S_{prompt,r}
+
S_{patch,r}
+
S_{switch,r}
$$

其中：

```text
S_prompt,r = prompt / anchor 对 regime r 的压力影响；
S_patch,r = MLP patch 对 regime r 的耦合影响；
S_switch,r = winner 从其他 regime 切换到 r 的影响。
```

当前只完成候选定位，还没有完成内部因果来源闭合。

### 8. 下一步任务

Phase232 仍属于同一阶段：

```text
readout pressure / readout regime / threshold barrier / competitor source
```

下一步应进入：

```text
Phase233: competitor source hook causal validation
（竞争来源 hook 级因果验证）
```

优先测试对象：

```text
GLM4:
no_answer_anchor → for_continuation；
explain_instruction → newline / because。

qwen3:
no_answer_anchor → because/period 相对增强；
because_removed → period_stop；
repeat_instruction → answer_boundary。

DS7B:
product/gate_up patch → be_continuation。
```

测试要求：

```text
1. 三模型依次运行，避免 GPU OOM；
2. 样本量需要大于 Phase232 当前每桶 2 到 4 的规模；
3. 先做自然 prompt 触发下的 layer/channel pressure map；
4. 再做 source-level suppression，不要直接做 target patch；
5. 记录 target、winner、second competitor，避免只压 winner 后被第二机制接管。
```

### 9. 阶段结论

Phase232 证明：

```text
Phase231 的判断基本正确，但还只是上界；
真正下一步必须定位 competitor source。
```

当前最可靠的来源候选是：

```text
GLM4 的 no_answer_anchor / explain_instruction regime switch。
```

当前中等可靠的候选是：

```text
qwen3 的 no_answer_anchor / because_removed / repeat_instruction 对 because、period、answer_boundary 的联动。
```

当前低权重候选是：

```text
DS7B 的 product/gate_up 与 be_continuation 耦合。
```

因此，接下来最可行的突破不是继续增加读出层公式，而是做 hook 级来源验证：

```text
从 prompt source 找到内部写入位置，
再从内部写入位置做 source-level causal suppression。
```

## Phase 233: 竞争来源 hook 级因果验证 [2026-07-07 03:51]

### 1. 本阶段任务

本阶段分析 Phase232 的判断是否正确，并继续完成同一阶段任务。

Phase232 的结论基本正确：

```text
Phase232 不是闭合验证，
而是 competitor source localization（竞争来源候选定位）。
```

它指出：

```text
GLM4:
主要是 prompt anchor 触发 regime switch。

qwen3:
because_reason / period_stop 来源更分散。

DS7B:
be_continuation 与 product / gate_up patch 有低权重候选耦合。
```

Phase233 的目标是把这些候选推进到 hook 级验证：

```text
在模型内部采集 full prompt 与 variant prompt 的 gate/up/product/down_out 差分；
再把这个差分从 variant prompt 中减回去；
观察 competitor regime 是否下降、target margin 是否改善、winner 是否改变。
```

这一步不是继续增强 target，而是做 source-level causal suppression（来源级因果抑制）。

### 2. 脚本和运行方式

脚本：

```text
tests/gpt5/phase233_competitor_source_hook_causal_validation.py
```

顺序运行脚本：

```text
tests/gpt5/run_phase233_competitor_source_hook_causal_validation.sh
```

结果目录：

```text
tests/result/phase233_competitor_source_hook_causal_validation/competitor_source_hook_causal_validation/
```

本阶段按顺序加载模型：

```text
1. qwen3
2. GLM4
3. DS7B
```

运行记录：

```text
qwen3:
observation_rows = 540
suppression_rows = 3240

GLM4:
observation_rows = 540
suppression_rows = 3240

DS7B:
observation_rows = 288
suppression_rows = 1728

cross-model:
observation_rows = 1368
suppression_rows = 8208
```

DS7B 样本仍少：

```text
success_rows = 6
drift_rows = 2
```

因此 DS7B 结果继续低权重处理。

### 3. 算法原理

对同一个样本、同一个 step，比较：

```text
full prompt
variant prompt
```

在 source layer 的内部状态差分：

$$
\Delta h_c
=
h_c^{variant}
-
h_c^{full}
$$

其中组件：

```text
c ∈ {gate, up, product, down_out}
```

然后在 variant prompt 上做来源抑制：

$$
h_c^{patched}
=
h_c^{variant}
-
\alpha \Delta h_c
$$

其中：

```text
alpha ∈ {0.5, 1.0}
```

观察读出变化：

$$
\Delta z_r^{suppress}
=
z_r^{patched}
-
z_r^{variant}
$$

以及：

$$
\Delta \mathrm{Margin}^{suppress}
=
\mathrm{TargetMargin}^{patched}
-
\mathrm{TargetMargin}^{variant}
$$

如果：

```text
regime_suppression_delta < 0
target_margin_delta_after_suppression > 0
```

说明该内部差分对 competitor regime 有因果贡献，而且抑制它能改善 target margin。

但更严格还要看：

```text
winner_changed_by_suppression
```

因为只降低 competitor 不等于输出闭合。

### 4. 关键结果

#### 4.1 GLM4: no_answer_anchor → For continuation 得到最强 hook 支持

Prompt source observation：

```text
success / no_answer_anchor / step=1 / for_continuation
rows = 10
winner_switch_rate = 1.0000
mean_target_delta = -8.3561
mean_regime_delta = +9.0469
mean_competitor_minus_target_delta = +17.4029
variant_winner = for_continuation
top_token = For
```

drift 组也成立：

```text
drift / no_answer_anchor / step=1 / for_continuation
rows = 10
winner_switch_rate = 1.0000
mean_target_delta = -3.9463
mean_regime_delta = +7.1797
mean_competitor_minus_target_delta = +11.1260
top_token = For
```

来源抑制结果：

```text
success / no_answer_anchor / step=1 / down_out / alpha=1.0 / for_continuation
rows = 10
regime_reduction_rate = 1.0000
target_margin_help_rate = 1.0000
mean_regime_suppression_delta = -1.1250
mean_target_delta_after_suppression = +3.3186
mean_target_margin_delta_after_suppression = +4.4436
winner_changed_rate = 0.0000
suppressed_winner = for_continuation
```

product 与 gate_up_pair 也类似：

```text
product alpha=1.0:
mean_regime_suppression_delta = -1.1062
mean_target_delta_after_suppression = +3.3467
mean_target_margin_delta_after_suppression = +4.4529

gate_up_pair alpha=1.0:
mean_regime_suppression_delta = -1.1187
mean_target_delta_after_suppression = +3.2936
mean_target_margin_delta_after_suppression = +4.4123
```

判断：

```text
GLM4 的 no_answer_anchor → For continuation 有明确 hook 级因果贡献。
```

但也有一个重要硬结果：

```text
winner_changed_rate = 0
```

说明 source suppression 能降低 For pressure、改善 target margin，但还不足以把 winner 从 For continuation 拉走。

所以这是强因果贡献，不是闭合。

#### 4.2 GLM4: explain_instruction → newline / because 也得到支持

Prompt observation：

```text
success / explain_instruction / step=2 / because_reason
rows = 10
winner_switch_rate = 1.0000
mean_target_delta = -5.5812
mean_regime_delta = +7.5422
mean_competitor_minus_target_delta = +13.1234
variant_winner = newline_boundary
top_token = newline
```

```text
success / explain_instruction / step=3 / because_reason
rows = 10
winner_switch_rate = 1.0000
mean_target_delta = -4.4062
mean_regime_delta = +8.6594
mean_competitor_minus_target_delta = +13.0656
variant_winner = because_reason
top_token = because
```

来源抑制：

```text
success / explain_instruction / step=2 / down_out / because_reason
rows = 10
regime_reduction_rate = 1.0000
target_margin_help_rate = 1.0000
mean_regime_suppression_delta = -0.2437
mean_target_delta_after_suppression = +1.8500
mean_target_margin_delta_after_suppression = +1.9688
winner_changed_rate = 0.0000
```

判断：

```text
explain_instruction 的 newline / because 来源也有 hook 级因果贡献，
但仍未让 winner 改变。
```

#### 4.3 qwen3: no_answer_anchor 的 because/period 来源得到部分支持

Prompt observation：

```text
success / no_answer_anchor / step=1 / because_reason
rows = 10
winner_switch_rate = 1.0000
mean_target_delta = -5.3719
mean_regime_delta = +2.7375
mean_competitor_minus_target_delta = +8.1094
variant_winner = then_continuation
top_token = Then
```

```text
drift / no_answer_anchor / step=1 / because_reason
rows = 10
winner_switch_rate = 0.8000
mean_target_delta = -10.9062
mean_regime_delta = +3.5000
mean_competitor_minus_target_delta = +14.4062
variant_winner = then_continuation / the_continuation
```

来源抑制：

```text
success / no_answer_anchor / step=1 / product / alpha=1.0 / because_reason
rows = 10
regime_reduction_rate = 1.0000
target_margin_help_rate = 1.0000
mean_regime_suppression_delta = -0.9062
mean_target_delta_after_suppression = -0.0906
mean_target_margin_delta_after_suppression = +0.6844
winner_changed_rate = 0.5000
```

period_stop 也类似：

```text
success / no_answer_anchor / step=1 / product / alpha=1.0 / period_stop
rows = 10
regime_reduction_rate = 1.0000
target_margin_help_rate = 1.0000
mean_regime_suppression_delta = -1.2000
mean_target_margin_delta_after_suppression = +0.6844
winner_changed_rate = 0.5000
```

判断：

```text
qwen3 的 no_answer_anchor 内部差分确实影响 because/period pressure。
```

但 winner 改变后并不稳定进入 target answer，而是分散到：

```text
answer_boundary
echo
the_continuation
then_continuation
```

这验证了 Phase232 的判断：

```text
qwen3 不是单一 competitor source；
它是多个边界机制联动。
```

#### 4.4 qwen3: repeat_instruction 不是简单 answer 修复

Prompt observation：

```text
success / repeat_instruction / step=2
winner = comma_repeat
top_token = comma
target_delta = +4.9656
```

这说明 repeat_instruction 能显著提高 target pressure，但也强烈触发：

```text
comma_repeat
```

drift step=3 有部分 answer_boundary：

```text
drift / repeat_instruction / step=3 / answer_boundary
rows = 10
winner_switch_rate = 0.4000
mean_target_delta = +1.1625
mean_regime_delta = +3.9375
top_token 包含 Answer
```

判断：

```text
repeat_instruction 不是纯 answer 修复；
它会同时打开 comma_repeat / answer_boundary / be_continuation 等机制。
```

#### 4.5 DS7B: 有弱因果信号，但仍低权重

Prompt observation：

```text
drift / no_instruction / step=1 / be_continuation
rows = 2
winner_switch_rate = 1.0000
mean_target_delta = -0.0625
mean_regime_delta = +2.7500
variant_winner = echo
```

来源抑制中：

```text
success / short_answer_instruction / step=3 / product / alpha=1.0 / be_continuation
rows = 6
regime_reduction_rate = 1.0000
target_margin_help_rate = 1.0000
mean_regime_suppression_delta = -0.3958
mean_target_margin_delta_after_suppression = +0.1836
winner_changed_rate = 0.0000
```

判断：

```text
DS7B 的 be_continuation 有内部来源信号，
但样本少、效应小、winner 不变，因此只能低权重记录。
```

### 5. 本阶段新增拼图

新增拼图 1：

```text
GLM4 的 no_answer_anchor → For continuation 已从候选定位推进到 hook 级因果贡献。
```

新增拼图 2：

```text
GLM4 的 source suppression 可以降低 For pressure 并改善 target margin，
但不足以改变 winner。
```

新增拼图 3：

```text
qwen3 的 no_answer_anchor 对 because/period 有内部来源贡献，
但抑制后会被 answer_boundary / echo / the / then 等第二竞争机制接管。
```

新增拼图 4：

```text
repeat_instruction 对 qwen3 不是纯修复，
它同时触发 comma_repeat / answer_boundary 等模式。
```

新增拼图 5：

```text
DS7B 的 be_continuation 有弱 source-level 因果信号，
但当前不能作为强证据。
```

### 6. 问题和硬伤

#### 6.1 source suppression 是均值差分，不是精确通道定位

本阶段使用：

```text
mean(variant - full)
```

作为来源差分。

这能验证该内部组件是否有因果贡献，但还没有精确到：

```text
具体 channel
具体 head
具体 token position
```

#### 6.2 抑制能改善 margin，但多数不能改变 winner

最典型的是 GLM4：

```text
For pressure 降低；
target margin 改善；
winner 仍是 For。
```

这说明 competitor field 很可能不是单点来源，而是：

```text
prompt anchor 改变后形成的整体状态场。
```

#### 6.3 qwen3 有第二竞争机制接管

qwen3 抑制 because/period 后，经常转到：

```text
answer_boundary
echo
the_continuation
then_continuation
```

这证明 Phase231 的担忧成立：

```text
只压当前 winner 后，第二 competitor 可能接管。
```

#### 6.4 DS7B 样本仍不足

DS7B 只有：

```text
success_rows = 6
drift_rows = 2
```

因此只作为敏感性参考。

### 7. 理论进展

Phase233 对机制链的推进是：

```text
PromptAnchor
→ MLP gate/up/product/down_out source delta
→ CompetitorPressure
→ TargetMargin
→ WinnerRegime
```

更准确的公式：

$$
\Delta z_r
=
F_r
\left(
h^{variant}
\right)
-
F_r
\left(
h^{full}
\right)
$$

来源抑制：

$$
h^{patched}
=
h^{variant}
-
\alpha
\left(
h^{variant}
-
h^{full}
\right)
$$

如果：

$$
z_r(h^{patched}) < z_r(h^{variant})
$$

则说明该内部差分对 regime r 有因果贡献。

但闭合还需要：

$$
\mathrm{Winner}(h^{patched})
=
\mathrm{TargetRegime}
$$

当前大多数结果还没有满足这个条件。

### 8. 阶段边界判断

Phase233 仍属于同一阶段：

```text
readout pressure / readout regime / threshold barrier / competitor source
```

本阶段完成了：

```text
source candidate
→ hook-level causal contribution
```

但没有完成：

```text
source-level full closure
```

所以可以继续自动推进到下一步，但下一步必须收紧目标，不应继续扩展理论名词。

### 9. 下一步任务

下一阶段建议：

```text
Phase234: second competitor takeover atlas
（第二竞争机制接管图谱）
```

原因：

```text
Phase233 已经证明压低当前 competitor 后，
很多情况下 winner 不变或被第二 competitor 接管。
```

下一步应记录：

```text
1. 当前 winner；
2. 抑制后 winner；
3. 抑制前 second competitor；
4. 抑制后 second competitor；
5. target margin 是否改善；
6. 是否进入 target answer、answer_boundary、the/then、echo、comma、newline 等机制。
```

优先模型：

```text
GLM4:
no_answer_anchor / For continuation。

qwen3:
no_answer_anchor / because-period-the-then-answer boundary 联动。

DS7B:
低权重保留。
```

### 10. 阶段结论

Phase233 是一个正结果，但不是闭合结果。

它证明：

```text
Phase232 找到的部分 competitor source 不是纯相关；
至少在 GLM4 和 qwen3 中，内部 gate/up/product/down_out 差分对 competitor pressure 有因果贡献。
```

同时它也证明：

```text
只找到当前 competitor source 还不够；
抑制当前 competitor 后，winner 可能仍不变，或被第二竞争机制接管。
```

因此下一步不能再只问：

```text
如何压低当前 winner？
```

而要问：

```text
整个 competitor field 的接管顺序是什么？
```

这是完成真实读出图谱和语言编码机制闭合前必须补上的拼图。

## Phase 234: 模式族图谱计划与测试矩阵 [2026-07-07 04:12]

### 1. 本阶段任务

本阶段综合两部分内容：

```text
1. Phase233 的校准：
   hook-level causal contribution 成立，但 source-level full closure 未完成。

2. 模式族系统规划：
   不能继续只围绕单个模式、单个 patch、单个 competitor 迭代，
   需要进入 PatternFamily Atlas Program（模式族图谱计划）。
```

Phase233 的判断基本正确：

```text
GLM4 和 qwen3 的部分 competitor source 不是纯相关；
内部 gate/up/product/down_out 差分对 competitor pressure 有因果贡献。
```

但 Phase233 同时证明：

```text
抑制当前 competitor 后，
winner 可能仍不变，
也可能被第二竞争机制接管。
```

所以原本建议的：

```text
second competitor takeover atlas
```

不应丢掉，而应纳入更大的：

```text
readout_competition（竞争读出模式族）
```

作为后续模式族图谱计划中的优先模式。

### 2. 本阶段是否进行模型测试

本阶段没有重新进行 CUDA 模型测试。

原因：

```text
当前任务是建立模式族分类、模式清单、测试层级和机器可读测试矩阵。
```

这属于实验设计和数据结构阶段。真正的三模型测试应放到下一阶段：

```text
Phase235: behavior_family_benchmark
```

届时再按要求依次运行：

```text
qwen3 → GLM4 → DS7B
```

### 3. 新增脚本和结果

新增脚本：

```text
tests/gpt5/phase234_pattern_family_atlas_matrix.py
```

输出目录：

```text
tests/result/phase234_pattern_family_atlas_matrix/pattern_family_atlas_matrix/
```

生成结果：

```text
families = 9
modes = 72
seed_test_cases = 36
test_levels = 8
```

输出文件：

```text
phase234_pattern_family_atlas_matrix.json
phase234_pattern_family_atlas_matrix.md
phase234_pattern_family_rows.jsonl
phase234_pattern_mode_rows.jsonl
phase234_seed_test_case_rows.jsonl
phase234_program_phase_rows.jsonl
```

同时更新了：

```text
research/MainAnalysis/20260707_03_语言的模式族测试方案.md
```

在尾部补充了 Phase234 可执行测试矩阵。

### 4. 九大模式族

Phase234 将语言模式族整理为九类：

```text
1. content_knowledge（内容知识模式族）
2. output_protocol（输出协议模式族）
3. reasoning_constraint（推理约束模式族）
4. syntax_structure（语法结构模式族）
5. language_action（语言动作模式族）
6. cross_lingual（跨语言模式族）
7. readout_competition（竞争读出模式族）
8. state_drift（状态维持与漂移模式族）
9. closure（闭合模式族）
```

这些不是语义分类，而是内部运行机制分类。

其中当前最高优先级是：

```text
content_knowledge
output_protocol
readout_competition
closure
```

原因：

```text
它们直接连接当前已有结果：
对象-关系-值任务、
explain/repeat 输出协议、
读出竞争、
模型闭合失败。
```

### 5. 统一测试层级

每个模式都按八层记录：

```text
Level 1: behavior
Level 2: prompt_trigger
Level 3: gate_up_product
Level 4: residual_state
Level 5: readout_competition
Level 6: competitor_source
Level 7: rollout
Level 8: closure
```

核心目的是避免：

```text
每个模式重新发明一套指标。
```

以后每个模式都按同一套字段积累：

```text
行为表现；
prompt 触发；
MLP 内部状态；
残差写入；
读出竞争；
竞争来源；
多 token 展开；
闭合状态。
```

### 6. 机器可读种子测试集

Phase234 先生成 36 个种子测试用例。

主体来自：

```text
content_knowledge / object_relation_value
```

并交叉四种输出协议：

```text
short
explain
repeat
list
```

种子对象包括：

```text
apple → color → red
banana → color → yellow
grass → color → green
snow → color → white
coal → color → black
lemon → taste → sour
hammer → function → hit
wheel → part_of → car
```

另外加入特殊样例：

```text
reason_negation_0001
syntax_boundary_0001
readout_takeover_0001
closure_stop_0001
```

这些样例用于把已有 readout competition 和 closure 问题接入模式族计划。

### 7. 与 Phase233 的衔接

Phase233 的关键问题：

```text
current winner 被压低后，
second competitor 接管。
```

已经在 Phase234 中进入：

```text
readout_competition / second_competitor_takeover
```

这说明下一步不应只做单一：

```text
For suppression
because suppression
period suppression
```

而要做：

```text
winner sequence / competitor takeover sequence
```

也就是记录：

```text
winner_before
second_before
winner_after
second_after
target_margin_delta
rollout_pattern
```

### 8. 后续阶段安排

Phase234 生成了如下阶段路线：

```text
Phase234:
pattern_family_matrix
建立模式族、模式、测试层级、样例任务的机器可读矩阵。

Phase235:
behavior_family_benchmark
先跑行为层模式分类，覆盖 qwen3、GLM4、DS7B。

Phase236:
prompt_trigger_family_atlas
对高差异模式做 prompt trigger 和 anchor 消融。

Phase237:
gate_product_family_atlas
采集 gate/up/product/down_out 跨模式差分。

Phase238:
readout_competition_family_atlas
统一记录 winner、second competitor、remaining gap。

Phase239:
source_suppression_family_validation
选择最稳定模式做 source-level suppression。

Phase240:
closure_candidate_family_validation
选择少数模式尝试完整闭合。
```

### 9. 当前判断

这次规划是必要的。

原因：

```text
Phase209 到 Phase233 已经证明：
单个模式中的局部因果链可以被追踪，
但语言机制不可能只靠单个模式闭合。
```

要破解语言编码机制，必须比较：

```text
模式族之间的共用机制；
模式之间的差分机制；
模式组合时的竞争机制；
闭合模式和漂移模式的区别。
```

### 10. 问题和硬伤

#### 10.1 目前只是矩阵，不是测试结果

Phase234 生成的是：

```text
测试结构
```

不是模型行为结果。

下一阶段必须实际跑三模型行为层测试。

#### 10.2 种子测试仍偏内容知识

当前 36 个种子测试主要从对象-关系-值扩展。

否定、翻译、条件推理、语法嵌套还只是少量特殊样例。

Phase235 需要扩充这些模式族的数据量。

#### 10.3 小模型偏差仍需保留

即使后续跑 qwen3、GLM4、DS7B，也必须保留：

```text
30% 到 50% 小模型偏差空间。
```

不能把小模型中的粗糙竞争场直接当成真实语言编码机制。

### 11. 阶段结论

Phase234 完成了从：

```text
单模式机制追踪
```

到：

```text
模式族图谱计划
```

的结构化过渡。

当前最可行路线是：

```text
先用 Phase234 的矩阵做 Phase235 行为层大样本测试；
再选择稳定差异最大的模式进入 hook 级机制测试；
最后用少数模式做闭合验证。
```

这比继续单点 patch 更有希望积累真实客观现象的拼图。

## Phase 235: 模式图谱固定数据契约与客户端索引 [2026-07-07 04:43]

### 1. 本阶段任务

本阶段分析了 Phase234 之后的补充要求：

```text
1. 所有测试必须输出固定数据格式，方便可视化客户端读取；
2. 可视化客户端需要独立的模式图谱界面；
3. 核心任务是追踪每个模式在深度神经网络内部的脉络，并进行跨数据统计。
```

判断结果：

```text
以上补充是正确的，而且应该优先于继续扩大行为测试。
```

原因是，如果没有固定数据契约，后续 qwen3、GLM4、DS7B 的每次测试都会变成孤立记录，不能稳定进入图谱，也不能积累全局模式脉络。

### 2. 已完成工作

新增脚本：

```text
tests/gpt5/phase235_pattern_atlas_data_contract.py
```

生成固定数据包：

```text
tests/result/pattern_family_atlas/v1/
```

输出文件：

```text
manifest.json
schema.json
client_index.json
families.jsonl
modes.jsonl
test_cases.jsonl
runs.jsonl
observations.jsonl
metrics.jsonl
graph_nodes.jsonl
graph_edges.jsonl
progress.json
summary.md
phase235_summary.json
```

新增客户端规范：

```text
frontend/PATTERN_ATLAS_CLIENT_SPEC.md
```

同时更新主研究文档：

```text
research/MainAnalysis/20260707_04_语言模式图谱+可视化客户端方案.md
```

### 3. 客观输出结果

本阶段生成的数据规模：

```text
模式族：9
模式：72
种子测试项：36
图谱节点：81
已知机制边：5
JSONL 总行数：208
```

数据包入口：

```text
tests/result/pattern_family_atlas/v1/manifest.json
```

客户端索引：

```text
tests/result/pattern_family_atlas/v1/client_index.json
```

当前支持的客户端页面：

```text
模式图谱总览；
模式族详情；
单模式详情；
机制链路图；
跨模型对比。
```

### 4. 已导入的机制边

本阶段没有宣称发现新的模型机制，而是把 Phase229 到 Phase233 中较稳定的机制证据导入图谱：

```text
GLM4 no_answer_anchor -> For 延续竞争；
GLM4 explain 指令 -> because 竞争；
qwen3 no_answer_anchor -> because/period 竞争；
qwen3 period 抑制 -> second competitor takeover；
DS7B short_answer -> be continuation 弱候选。
```

这些边目前主要属于：

```text
输出协议；
读出竞争；
局部 hook 因果支持。
```

它们还不能代表完整语言编码机制。

### 5. 固定证据公式

本阶段采用的证据分数为：

$$
\boxed{
E
=
0.15B
+
0.15S
+
0.20R
+
0.25H
+
0.15C
+
0.10M
}
$$

其中：

```text
B = 行为一致性；
S = 状态一致性；
R = 读出一致性；
H = hook 因果支持；
C = 闭合支持；
M = 跨模型一致性。
```

需要注意：

```text
这个公式只是证据聚合公式，不是最终智能理论公式。
```

### 6. 当前统一机制表达

当前仍然保留 Phase234 的核心骨架：

$$
\boxed{
\mathrm{LanguageMechanism}
=
\sum_i
\alpha_i(x,t)
P_i(x,t)
}
$$

其中每个模式脉络为：

$$
\boxed{
P_i
=
\mathrm{TriggerTrace}_i
\circ
\mathrm{GateProductTrace}_i
\circ
\mathrm{ResidualWriteTrace}_i
\circ
\mathrm{ReadoutTrace}_i
\circ
\mathrm{CompetitorTrace}_i
\circ
\mathrm{RolloutTrace}_i
\circ
\mathrm{ClosureTrace}_i
}
$$

模式族之间的关系仍然表达为：

$$
\boxed{
P_i
=
P_{\mathrm{shared}}
+
\Delta P_i
}
$$

本阶段的进展不是修改理论名词，而是把这个公式落到可积累的数据结构中。

### 7. 严格审视

第一，Phase235 不是模型行为测试。

`observations.jsonl` 当前只有格式占位记录，不能当作 qwen3、GLM4、DS7B 的新实验结论。

第二，当前机制边分布不均衡。

读出竞争和输出协议证据较多，但知识网络、语法系统、推理约束、跨语言模式仍然缺少等价强度的 hook 级证据。

第三，小模型偏差仍然很大。

当前测试模型可能存在：

```text
30% 到 50% 的内部结构偏差。
```

所以图谱中的边应当被看作“小模型可观测脉络”，不能直接等同于真实语言能力的完整数学结构。

### 8. 阶段判断

Phase235 与 Phase234 属于同一阶段性目标：

```text
从单模式机制追踪转向模式族全局图谱。
```

本阶段完成了这一目标中最基础的一环：

```text
让后续所有测试可以进入同一套固定数据格式和同一个客户端入口。
```

因此当前阶段可以继续自动推进到下一步。

### 9. 下一阶段任务

Phase236 应该执行：

```text
模式族行为基准测试。
```

具体要求：

```text
1. 使用 qwen3、GLM4、DS7B 依次测试，避免 GPU 内存溢出；
2. 使用 Phase234 的模式族和模式矩阵；
3. 扩大重点模式族的数据量；
4. 每个 case 的输出写入 observations.jsonl；
5. 每个模式和模式族的统计写入 metrics.jsonl；
6. 更新 progress.json、graph_nodes.jsonl、graph_edges.jsonl；
7. 不急于理论闭合，优先找稳定成功、稳定漂移和跨模型分歧。
```

Phase236 的核心问题不是：

```text
某个 patch 能不能修复输出。
```

而是：

```text
不同语言模式族在三模型中的自然行为分布是什么，
哪些模式值得进入 hook 级机制追踪。
```

## Phase 236: 模式族行为基准测试与图谱回写 [2026-07-07 04:51]

### 1. 本阶段任务

Phase235 完成固定数据契约后，Phase236 继续同一阶段目标：

```text
把模式族图谱从结构计划推进到真实跨模型行为观测。
```

本阶段执行了 qwen3、GLM4、DS7B 三个模型的顺序测试，并把测试结果写入固定数据包：

```text
tests/result/pattern_family_atlas/v1/
```

### 2. 新增脚本

测试脚本：

```text
tests/gpt5/phase236_pattern_family_behavior_benchmark.py
```

顺序运行脚本：

```text
tests/gpt5/run_phase236_pattern_family_behavior_benchmark.sh
```

三模型执行顺序：

```text
qwen3 -> GLM4 -> DS7B
```

每个模型完成后释放 CUDA 显存，再加载下一个模型。

脚本检查：

```text
python -m py_compile: 通过
bash -n: 通过
```

### 3. 输出目录

独立 Phase236 结果：

```text
tests/result/phase236_pattern_family_behavior_benchmark/pattern_family_behavior_benchmark/
```

客户端图谱回写：

```text
tests/result/pattern_family_atlas/v1/
```

回写文件包括：

```text
observations.jsonl
metrics.jsonl
graph_edges.jsonl
progress.json
summary.md
```

### 4. 算法原理

本阶段不做 hook，不进入内部激活干预，只做行为层和读出层轻量观测。

对每个 case：

```text
1. 读取 prompt 和 target；
2. 计算下一 token logits；
3. 记录 target token rank、target logit、target margin；
4. greedy 生成最多 24 个新 token；
5. 根据输出是否包含目标词、是否以目标词开始、是否符合协议、是否过生成，得到行为评分；
6. 写入固定 observations / metrics / graph_edges。
```

行为评分公式为：

$$
\boxed{
S_{\mathrm{behavior}}
=
0.45 I_{\mathrm{contains}}
+
0.25 I_{\mathrm{starts}}
+
0.30 I_{\mathrm{pattern}}
}
$$

其中：

```text
I_contains = 输出包含目标词；
I_starts = 输出以目标词或目标首词元开始；
I_pattern = 输出协议匹配。
```

读出边距为：

$$
\boxed{
M_{\mathrm{target}}
=
\ell_{\mathrm{target}}
-
\max_j \ell_{\mathrm{regime}_j}
}
$$

其中 `regime_j` 包括：

```text
because
period
comma
For
The
be/is/are
newline
Answer
```

### 5. 客观测试结果

跨模型总量：

```text
case_rows: 132
observation_rows: 1056
metric_rows: 60
graph_edges: 36
```

总分布：

```text
mean_behavior_score: 0.6462
pattern_match_rate: 0.6288
```

漂移类型：

```text
none: 83
wrong_or_missing_target: 30
over_generation: 19
```

分模型结果：

```text
qwen3:
mean_behavior_score = 0.7307
pattern_match_rate = 0.6591

GLM4:
mean_behavior_score = 0.6580
pattern_match_rate = 0.6136

DS7B:
mean_behavior_score = 0.5500
pattern_match_rate = 0.6136
```

### 6. 主要现象

第一，简单模式存在一定跨模型稳定性。

少量 closure、cross_lingual、language_action、reasoning_constraint、syntax_structure 样例表现较好。

但这些模式族样例数量仍少，不能提前理论总结。

第二，content_knowledge 暴露明显目标定义问题。

失败集中出现在：

```text
hammer -> hit
lemon taste -> sour
wheel part_of -> car
```

模型常给出：

```text
drive
strike
tart
rim
vehicle
```

这些不一定是模型机制失败，可能是测试目标词过窄或关系定义歧义。

第三，output_protocol 仍存在过生成。

即使要求 one word，模型仍可能继续解释或重复提示。这与前面阶段关于停止控制和输出协议分离的结论一致。

### 7. 严格审视

第一，Phase236 是行为层基准，不是机制闭合。

它只能回答：

```text
哪些模式输出稳定；
哪些模式容易漂移；
哪些模式值得进入 hook 追踪。
```

不能证明内部真实路径。

第二，评分规则存在硬伤。

当前评分过度依赖目标字符串匹配，不能正确处理：

```text
同义词；
上位词；
解释性正确答案；
中文翻译变体；
关系歧义。
```

第三，模式族数据不均衡。

content_knowledge 样例多，其他模式族样例少，因此跨模式族比较还不公平。

第四，小模型偏差仍然存在。

当前模型内部结构可能较粗糙，行为漂移可能混合了：

```text
真实语言机制；
小模型能力不足；
提示格式脆弱；
评分规则误差。
```

因此仍然保留：

```text
30% 到 50% 的偏差空间。
```

### 8. 理论进展

本阶段没有修改统一理论名称，也不引入新的大理论。

更准确的进展是：

```text
语言模式图谱开始有了可持续积累的行为层观测。
```

当前机制公式仍保持：

$$
\boxed{
\mathrm{LanguageMechanism}
=
\sum_i
\alpha_i(x,t)
P_i(x,t)
}
$$

其中：

$$
\boxed{
P_i
=
\mathrm{TriggerTrace}_i
\circ
\mathrm{GateProductTrace}_i
\circ
\mathrm{ResidualWriteTrace}_i
\circ
\mathrm{ReadoutTrace}_i
\circ
\mathrm{CompetitorTrace}_i
\circ
\mathrm{RolloutTrace}_i
\circ
\mathrm{ClosureTrace}_i
}
$$

Phase236 只补上了：

$$
\boxed{
\mathrm{BehaviorObservation}
\rightarrow
\mathrm{PatternAtlas}
}
$$

还没有补上：

$$
\boxed{
\mathrm{BehaviorObservation}
\rightarrow
\mathrm{InternalTrace}
\rightarrow
\mathrm{CausalClosure}
}
$$

### 9. 阶段结论

Phase236 是正确推进。

它完成了：

```text
固定格式
-> 跨模型行为测试
-> 客户端图谱回写
```

这一条基础链路。

但它也暴露了一个关键问题：

```text
如果测试目标本身有歧义，后续 hook 追踪会追踪到评分误差，而不是真实机制差异。
```

### 10. 下一阶段任务

Phase237 应该进入：

```text
prompt trigger / anchor 消融图谱。
```

但在进入更深 hook 之前，必须先修正 Phase236 暴露的问题：

```text
1. 为 content_knowledge 增加同义目标集合；
2. 消除 function、taste、part_of 等关系歧义；
3. 平衡 reasoning、syntax、cross_lingual、language_action 的样例数量；
4. 区分“语义等价但目标词不同”和“真实模式漂移”；
5. 再选稳定漂移模式进入 prompt trigger 和内部脉络追踪。
```

## Phase 237: 可视化客户端运行环境与模式图谱数据发布配置 [2026-07-07 05:32]

### 1. 本阶段任务

本阶段任务不是新的模型机制测试，而是安装和配置可视化客户端运行环境，使 Phase235/236 生成的 Pattern Atlas（模式图谱）可以被前端稳定读取。

核心目标：

```text
1. 安装并验证前端依赖；
2. 配置模式图谱数据的前端 public 访问路径；
3. 启动并验证 Vite 可视化客户端；
4. 启动并验证 FastAPI 后端；
5. 补齐后端 GPT-2 本地缓存，使模型相关 API 不因缺权重而降级。
```

### 2. 前端环境

前端目录：

```text
frontend/
```

执行：

```text
npm install
npm run sync:pattern-atlas
npm run build
npm run dev -- --host 0.0.0.0
```

结果：

```text
npm install: 完成，依赖已是最新；
npm run build: 通过；
Vite dev server: 运行于 5173。
```

当前访问地址：

```text
http://127.0.0.1:5173/
http://192.168.101.116:5173/
```

注意：当前 shell 环境存在 `http_proxy/https_proxy`，所以命令行测试本地服务时需要使用：

```text
curl --noproxy '*'
```

否则 `localhost` 可能被代理拦截并返回 502。

### 3. 模式图谱数据发布

新增同步脚本：

```text
frontend/scripts/sync_pattern_atlas.mjs
```

新增 npm 命令：

```text
npm run sync:pattern-atlas
```

源数据目录：

```text
tests/result/pattern_family_atlas/v1/
```

前端发布目录：

```text
frontend/public/vis_data/pattern_family_atlas/v1/
```

前端可访问入口：

```text
/vis_data/pattern_family_atlas/v1/manifest.json
/vis_data/pattern_family_atlas/v1/progress.json
/vis_data/pattern_family_atlas/v1/public_manifest.json
```

同步结果：

```text
14 个 pattern atlas 文件已同步到 frontend/public/vis_data/pattern_family_atlas/v1。
```

新增环境变量：

```text
VITE_PATTERN_ATLAS_BASE=/vis_data/pattern_family_atlas/v1
VITE_PATTERN_ATLAS_MANIFEST=/vis_data/pattern_family_atlas/v1/manifest.json
```

### 4. 后端环境

后端入口：

```text
python -m server.server
```

服务地址：

```text
http://127.0.0.1:5001/
```

已验证：

```text
http://127.0.0.1:5001/docs
http://127.0.0.1:5001/agi/progress
```

第一次启动时发现：

```text
GPT-2 model.safetensors 本地缓存缺失；
fallback 下载因 HF_ENDPOINT=hf-mirror.com 不可达失败。
```

修复方式：

```text
使用官方 HuggingFace endpoint 下载 gpt2 snapshot。
```

缓存路径：

```text
/home/rankrank/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e
```

重启后结果：

```text
Forced 12-layer GPT-2 loaded successfully on cuda；
ManifoldSurgeon and GeometricInterceptor initialized；
AGI Core Engine and RLMF Provider initialized；
AGI Chat Engine initialization started。
```

### 5. 当前验证结果

前端：

```text
HTTP 200:
http://127.0.0.1:5173/
```

后端：

```text
HTTP 200:
http://127.0.0.1:5001/docs
http://127.0.0.1:5001/agi/progress
```

模式图谱数据：

```text
HTTP 200:
http://127.0.0.1:5173/vis_data/pattern_family_atlas/v1/public_manifest.json
http://127.0.0.1:5173/vis_data/pattern_family_atlas/v1/manifest.json
http://127.0.0.1:5173/vis_data/pattern_family_atlas/v1/progress.json
```

### 6. 问题和风险

第一，前端 `npm install` 报告：

```text
15 vulnerabilities
```

其中：

```text
1 low
6 moderate
8 high
```

本阶段没有自动执行 `npm audit fix`，因为自动修复可能升级依赖并破坏现有可视化客户端。

第二，当前只是完成环境和数据发布配置。

Pattern Atlas 独立页面还没有正式实现到 React UI 中，目前只是数据已经可以被前端通过 public URL 读取。

第三，后端启动会加载 GPT-2 到 CUDA，会占用显存。

如果后续要运行 qwen3、GLM4、DS7B 测试，应该先停止后端，避免 GPU 内存竞争。

### 7. 阶段结论

Phase237 完成了可视化客户端运行环境配置。

当前链路已经打通：

```text
tests/result/pattern_family_atlas/v1
-> frontend/public/vis_data/pattern_family_atlas/v1
-> Vite client 5173
-> FastAPI backend 5001
```

这为下一步实现真正的 Pattern Atlas 独立界面提供了运行基础。

### 8. 下一阶段

下一阶段应该实现：

```text
Pattern Atlas 独立可视化页面。
```

最低要求：

```text
1. 读取 VITE_PATTERN_ATLAS_MANIFEST；
2. 展示 global_progress；
3. 展示 9 个模式族卡片；
4. 展示 72 个模式表格；
5. 展示 graph_nodes / graph_edges 的机制链路；
6. 展示 Phase236 行为基准统计；
7. 每次运行 npm run sync:pattern-atlas 后自动看到最新结果。
```

## Phase 238: 行为评分校准、歧义样例标记与独立 Pattern Atlas 页面 [2026-07-07 06:45]

### 1. 对附件判断的校准

附件对 Phase236 和 Phase237 的判断基本正确。

正确部分：

```text
1. Phase236 是行为层基准，不是内部机制闭合；
2. Phase237 是工程基础设施，不是新的机制实验；
3. 当前真正完成的是固定格式、跨模型行为观测、图谱回写、前端 public 数据发布；
4. 下一步不能直接进入深层 hook，必须先修正行为评分和目标歧义。
```

需要补充的一点：

```text
评分校准不能简单扩大 aliases。
如果 aliases 太宽，会把解释中的相关词误判成答案本身。
```

例如初版校准把 `citric` 纳入 lemon taste 的同义集合，会把 “sweet because citric acid” 误判为 sour/tart 等价答案。已在本阶段收紧。

### 2. 本阶段任务

本阶段目标：

```text
1. 增加 target_aliases / acceptable_answers / relation_schema；
2. 重新计算 Phase236 行为得分；
3. 标记 semantic_correct_but_target_mismatch；
4. 标记 ambiguous target / relation；
5. 导出 stable failure candidates；
6. 实现独立 Pattern Atlas 页面；
7. 根据固定格式回写语言模式图谱测试数据。
```

本阶段没有重新跑 qwen3、GLM4、DS7B，因为目标是校准 Phase236 已有输出，不是新增模型行为测试。

### 3. 新增脚本和页面

新增评分校准脚本：

```text
tests/gpt5/phase238_pattern_atlas_scoring_calibration.py
```

新增独立页面：

```text
frontend/public/pattern_atlas.html
```

更新客户端规范：

```text
frontend/PATTERN_ATLAS_CLIENT_SPEC.md
```

同步命令：

```text
cd frontend
npm run sync:pattern-atlas
```

构建验证：

```text
npm run build: 通过
```

页面访问：

```text
http://127.0.0.1:5173/pattern_atlas.html
```

### 4. 输出目录

独立结果目录：

```text
tests/result/phase238_pattern_atlas_scoring_calibration/pattern_atlas_scoring_calibration/
```

回写图谱目录：

```text
tests/result/pattern_family_atlas/v1/
```

前端发布目录：

```text
frontend/public/vis_data/pattern_family_atlas/v1/
```

### 5. 新增固定格式数据

新增文件：

```text
case_aliases.jsonl
semantic_equivalence_flags.jsonl
stable_failure_candidates.jsonl
```

独立结果中还包含：

```text
phase238_calibrated_case_rows.jsonl
phase238_calibrated_observations.jsonl
phase238_calibrated_metrics.jsonl
phase238_ambiguous_case_report.md
phase238_scoring_calibration_summary.json
```

这些文件解决：

```text
target_aliases；
acceptable_answers；
relation_schema；
semantic_equivalence_label；
ambiguous_case；
stable_failure_candidate。
```

### 6. 校准评分公式

Phase238 使用：

$$
\boxed{
S_{\mathrm{calibrated}}
=
0.35 I_{\mathrm{answer}}
+
0.25 I_{\mathrm{protocol}}
+
0.25 I_{\mathrm{semantic}}
+
0.15 I_{\mathrm{closure}}
}
$$

其中：

```text
I_answer = 是否命中 target_aliases / acceptable_answers；
I_protocol = 是否符合输出协议；
I_semantic = 是否语义等价；
I_closure = 是否出现粗略闭合信号。
```

注意：

```text
这个公式只是行为层评分校准工具，不是语言机制公式。
```

### 7. 客观结果

输入规模：

```text
case_rows: 132
unique_cases: 44
```

输出规模：

```text
calibrated_observation_rows: 792
metric_rows: 68
case_aliases: 44
semantic_equivalence_flags: 132
stable_failure_candidates: 12
```

校准前后：

```text
mean_original_behavior_score: 0.6462
mean_calibrated_behavior_score: 0.8133
```

校准后漂移类型：

```text
none: 69
protocol_or_over_generation: 27
semantic_or_target_failure: 17
closure_or_rollout_failure: 11
semantic_correct_but_target_mismatch: 8
```

歧义行：

```text
ambiguous_rows: 13
semantic_mismatch_rows: 13
```

### 8. 稳定失败候选

Phase238 导出 12 个 stable failure candidates。

主要类型：

```text
stable_protocol_failure；
stable_semantic_or_target_failure。
```

当前最值得进入 Phase239 的方向：

```text
short_answer 过生成；
Answer boundary；
one-word constraint；
period / newline / Answer 触发边界；
wheel part_of 关系歧义。
```

其中更适合机制追踪的是：

```text
stable_protocol_failure
```

因为它更可能对应输出协议、边界控制和读出机制，而不是语义标注歧义。

### 9. 独立 Pattern Atlas 页面

页面：

```text
frontend/public/pattern_atlas.html
```

访问：

```text
http://127.0.0.1:5173/pattern_atlas.html
```

已验证：

```text
HTTP 200；
public_manifest.json 可访问；
progress.json 可访问；
stable_failure_candidates.jsonl 可访问。
```

页面显示：

```text
global_progress；
9 个模式族；
72 个模式；
calibrated metrics；
stable failure candidates；
mechanism graph edges；
case aliases；
ambiguity risk。
```

### 10. 当前图谱进度

Phase238 回写后：

```text
pattern_family_atlas: 0.48
behavior level: 0.54
general_language_mechanism_confidence: 0.46
model_internal_closure: 0.46
```

当前完成内容：

```text
1. 固定数据契约；
2. 行为层跨模型基准；
3. 行为评分校准；
4. 语义等价标记；
5. 歧义样例报告；
6. 稳定失败候选筛选；
7. 前端 public 数据发布；
8. 独立 Pattern Atlas 页面。
```

仍未完成：

```text
1. 模式族样本均衡；
2. prompt trigger / anchor 系统消融；
3. gate/up/product 跨模式族追踪；
4. residual state 传播图谱；
5. rollout 多 token 轨迹；
6. closure 因果闭合。
```

### 11. 严格审视

第一，校准后分数升高不等于模型能力变强。

它说明：

```text
Phase236 的原始评分低估了语义等价答案。
```

第二，semantic_equivalence 仍然是启发式。

同义集合过窄会误判正确答案，同义集合过宽会误判解释性相关词。当前只是第一版校准。

第三，模式族仍然不均衡。

当前 content_knowledge 行数仍然明显多于 reasoning、syntax、cross_lingual、language_action、closure。

第四，小模型偏差仍然存在。

即使评分校准后，也必须保留：

```text
30% 到 50% 小模型结构偏差空间。
```

### 12. 阶段判断

Phase238 与 Phase236/237 属于同一个阶段性目标：

```text
把语言模式研究沉淀成可持续积累、可视化、可校准的 Pattern Atlas。
```

这一阶段的基础目标已经完成：

```text
固定格式
-> 行为观测
-> 评分校准
-> 图谱回写
-> 前端查看。
```

Phase239 已经进入新的机制测试阶段，应该单独执行。

### 13. 下一阶段任务

Phase239 建议执行：

```text
prompt trigger / anchor 消融。
```

但不应大范围铺样本，而应从 Phase238 的 stable failure candidates 中选择少量高价值样例。

优先级：

```text
1. stable_protocol_failure；
2. short_answer / one-word 过生成；
3. Answer boundary；
4. period / newline 触发；
5. readout regime takeover。
```

Phase239 如果进行模型测试，应依次运行：

```text
qwen3 -> GLM4 -> DS7B
```

并继续写入：

```text
observations.jsonl
metrics.jsonl
graph_edges.jsonl
progress.json
```

## Phase 239: 稳定协议失败的 prompt trigger / anchor 消融图谱 [2026-07-07 07:23]

### 1. 对附件判断的校准

附件对 Phase238 的判断正确：

```text
Phase238 是行为观测校准，不是机制实验；
Phase239 应该从 stable_protocol_failure 中选少量样例，
进入 prompt trigger / anchor 消融。
```

本阶段执行了这个方案。

需要特别记录一个关键校准：

```text
初版统计把 explain_instruction 按解释任务协议评分，
会把“换任务成功”误判成“原始短答协议修复”。
```

因此脚本已修正为：

```text
所有 prompt 变体都按 original_protocol_match 评价，
即是否修复原始 short-answer / one-word 协议。
```

### 2. 新增脚本

新增测试脚本：

```text
tests/gpt5/phase239_stable_protocol_prompt_trigger_atlas.py
```

新增顺序运行脚本：

```text
tests/gpt5/run_phase239_stable_protocol_prompt_trigger_atlas.sh
```

三模型顺序：

```text
qwen3 -> GLM4 -> DS7B
```

每个模型完成后释放 CUDA 显存。

脚本检查：

```text
python -m py_compile: 通过
bash -n: 通过
```

### 3. 测试设计

输入来自：

```text
tests/result/pattern_family_atlas/v1/stable_failure_candidates.jsonl
```

只选择：

```text
stable_protocol_failure
```

测试规模：

```text
8 个 stable_protocol_failure 样例；
每个样例 11 个 prompt 变体；
三模型共 264 条 variant rows。
```

prompt 变体：

```text
full
no_answer_anchor
strong_answer_anchor
one_word_strict
one_word_no_explain
period_forced
newline_removed
colon_removed
short_answer_instruction
explain_instruction
target_seeded
```

记录指标：

```text
target_rank；
target_logit；
target_margin_vs_winner；
winner_regime；
second_competitor；
top_token；
original_protocol_match；
variant_protocol_match；
over_generation；
closure_signal；
baseline_score_delta；
baseline_margin_delta。
```

### 4. 输出文件

结果目录：

```text
tests/result/phase239_stable_protocol_prompt_trigger_atlas/stable_protocol_prompt_trigger_atlas/
```

核心文件：

```text
phase239_cross_model_summary.json
phase239_cross_model_prompt_trigger_rows.jsonl
phase239_cross_model_observations.jsonl
phase239_cross_model_metrics.jsonl
phase239_cross_model_graph_edges.jsonl
phase239_protocol_failure_report.md
phase239_stable_failure_selection.json
```

同时回写：

```text
tests/result/pattern_family_atlas/v1/observations.jsonl
tests/result/pattern_family_atlas/v1/metrics.jsonl
tests/result/pattern_family_atlas/v1/graph_edges.jsonl
tests/result/pattern_family_atlas/v1/progress.json
tests/result/pattern_family_atlas/v1/summary.md
```

并已同步到前端：

```text
cd frontend
npm run sync:pattern-atlas
npm run build
```

构建通过。

### 5. 客观结果

总量：

```text
variant_rows: 264
observation_rows: 2376
metric_rows: 33
graph_edges: 33
```

修正版跨模型结果：

```text
mean_score: 0.6152
protocol_match_rate: 0.0038
```

分模型：

```text
qwen3:
mean_score = 0.6869
protocol_match_rate = 0.0

GLM4:
mean_score = 0.7256
protocol_match_rate = 0.0114

DS7B:
mean_score = 0.4330
protocol_match_rate = 0.0
```

### 6. 关键现象

第一，几乎没有 prompt 变体能恢复原始短答协议。

```text
protocol_match_rate = 0.0038
```

说明 stable_protocol_failure 不是简单 prompt wording 问题。

第二，explain_instruction 不是短答修复。

修正后：

```text
explain_instruction:
original_protocol_match = 0.0
variant_protocol_match = 0.9167
over_generation_rate = 0.9583
```

这说明它只是把任务切换成解释输出，不是修复原始 one-word / short-answer 协议。

第三，强指令和强 Answer 锚点没有修复。

```text
one_word_strict:
protocol_match_rate = 0.0
over_generation_rate = 0.9583

strong_answer_anchor:
protocol_match_rate = 0.0
over_generation_rate = 0.7083
```

这说明短答失败不只是缺少显式指令。

第四，主要竞争机制仍然是：

```text
the_continuation；
newline_boundary；
answer_boundary；
be_continuation；
period_stop；
for_continuation。
```

### 7. 严格审视

第一，这是 prompt trigger 层负结果，不是内部机制闭合。

它只能说明：

```text
浅层 prompt / anchor 修改不足以修复稳定协议失败。
```

第二，测试样例仍来自小模型。

需要保留：

```text
30% 到 50% 小模型结构偏差空间。
```

第三，当前没有进入 gate/up/product 或 residual write。

所以不能断言具体内部通道或层已经定位。

第四，稳定协议失败的样例主要来自 content_knowledge 的 short protocol 和 output_protocol short_answer，模式族仍不均衡。

### 8. 理论进展

本阶段补上的拼图是：

$$
\boxed{
\mathrm{StableProtocolFailure}
\not\approx
\mathrm{PromptWordingFailure}
}
$$

更准确地说：

$$
\boxed{
\mathrm{ProtocolFailure}
\rightarrow
\mathrm{ReadoutRegimeTakeover}
\lor
\mathrm{ResidualStateBias}
\lor
\mathrm{RolloutPatternBias}
\lor
\mathrm{ClosureStateMissing}
}
$$

这只是方向判断，还不是闭合证明。

### 9. 当前图谱进度

Phase239 回写后：

```text
pattern_family_atlas: 0.52
prompt_trigger: 0.32
behavior: 0.54
readout_competition: 0.32
model_internal_closure: 0.46
general_language_mechanism_confidence: 0.47
```

已完成：

```text
固定数据契约；
行为层测试；
评分校准；
稳定失败筛选；
prompt trigger / anchor 消融；
前端 Pattern Atlas 查看。
```

未完成：

```text
gate/up/product 协议失败追踪；
residual write 机制；
rollout 多 token 轨迹；
closure 因果闭合。
```

### 10. 阶段结论

Phase239 是一个重要负结果。

它证明：

```text
stable_protocol_failure 不能靠简单 prompt / anchor 变体修复。
```

这把下一步研究从：

```text
继续调提示词
```

推进到：

```text
追踪协议状态是否被写入 residual stream，
以及读出竞争机制为什么接管 rollout。
```

### 11. 下一阶段任务

Phase240 应该进入：

```text
gate/up/product protocol trace
```

核心问题：

```text
1. one-word / short-answer 协议状态是否进入 MLP product；
2. Answer boundary 是否有可定位的 residual write；
3. the_continuation / newline_boundary 为什么压过目标闭合；
4. 是否存在 stopping / boundary state 的早期写入失败；
5. 哪些稳定协议失败值得进入更深 hook 因果验证。
```

Phase240 仍应继续使用少量 stable_protocol_failure 样例，避免大范围铺数据。

## Phase 240: gate/up/product 协议状态追踪与读出竞争校准 [2026-07-07 14:57]

### 1. 阶段目标

本阶段承接 Phase239 的负结果。

Phase239 已经证明：

```text
stable_protocol_failure 不是简单 prompt / anchor 变体可以修复的问题。
```

Phase240 的目标不是继续改提示词，而是进入内部状态链条：

```text
PromptProtocol
→ Gate / Up / Product
→ ResidualState
→ ReadoutCompetition
→ Rollout / Closure
```

核心问题：

```text
1. one-word / short-answer 协议是否进入 MLP product；
2. 协议进入 product / residual 后，是否能改善目标读出竞争；
3. 如果目标读出没有改善，稳定协议失败更像写入失败、读出竞争失败，还是 rollout / closure 失败。
```

### 2. 测试脚本与输出

新增脚本：

```text
tests/gpt5/phase240_gate_product_protocol_trace.py
tests/gpt5/run_phase240_gate_product_protocol_trace.sh
```

测试顺序：

```text
qwen3 → GLM4 → DS7B
```

固定输出目录：

```text
tests/result/phase240_gate_product_protocol_trace/gate_product_protocol_trace/
```

核心输出文件：

```text
phase240_cross_model_summary.json
phase240_cross_model_behavior_rows.jsonl
phase240_cross_model_gate_product_protocol_rows.jsonl
phase240_cross_model_residual_protocol_rows.jsonl
phase240_cross_model_observations.jsonl
phase240_cross_model_metrics.jsonl
phase240_cross_model_graph_edges.jsonl
phase240_protocol_trace_report.md
```

并回写 Pattern Atlas：

```text
tests/result/pattern_family_atlas/v1/observations.jsonl
tests/result/pattern_family_atlas/v1/metrics.jsonl
tests/result/pattern_family_atlas/v1/graph_edges.jsonl
tests/result/pattern_family_atlas/v1/progress.json
tests/result/pattern_family_atlas/v1/summary.md
```

前端同步和构建：

```text
cd frontend
npm run sync:pattern-atlas
npm run build
```

构建通过；只有 Vite chunk size 警告，不影响本阶段结果。

### 3. 测试原理

从 Phase238 / Phase239 标记的 stable_protocol_failure 中选取 6 个高价值样例。

对每个样例比较 6 种输入变体：

```text
full
strong_answer_anchor
one_word_strict
short_answer_instruction
explain_instruction
target_seeded
```

内部追踪层：

```text
qwen3: layer 29，观察 29 / 31 / 33
GLM4: layer 30，观察 28 / 30 / 32
DS7B: layer 24，观察 24 / 26 / 27
```

采集组件：

```text
gate
up
product
down_out
recomputed_product
residual_state
readout_winner
second_competitor
target_margin_vs_winner
rollout output
```

基础差分公式：

$$
\Delta z_{c,l,v}
=
z_{c,l,v}
-
z_{c,l,\mathrm{full}}
$$

其中：

```text
c: component，例如 gate / up / product / down_out / residual_state；
l: layer；
v: prompt variant。
```

相对差分：

$$
r_{c,l,v}
=
\frac{
\|\Delta z_{c,l,v}\|
}{
\|z_{c,l,\mathrm{full}}\|+\epsilon
}
$$

读出竞争变化：

$$
\Delta M_v
=
M_v - M_{\mathrm{full}}
$$

其中：

$$
M
=
\mathrm{logit}(\mathrm{target})
-
\mathrm{logit}(\mathrm{winning\ competitor})
$$

判断逻辑：

```text
如果 strict protocol prompt 的 product / down_out 差分很小：
    倾向 protocol_state_weak_or_not_written；

如果 product / down_out 差分明显，但 target margin 不改善，protocol_match 仍然为 0：
    倾向 protocol_state_written_but_readout_competition_failed；

如果 target margin 改善，但生成仍过长：
    倾向 readout_or_rollout_closure_failed。
```

### 4. 主要结果

跨模型总量：

```text
behavior_rows: 108
gate_product_trace_rows: 540
residual_trace_rows: 324
observation_rows: 972
metric_rows: 162
graph_edges: 24
mean_behavior_score: 0.6278
protocol_match_rate: 0.0
```

三模型判断：

```text
qwen3:
  decision: protocol_state_written_but_readout_competition_failed
  strict_mean_product_down_relative_delta: 0.172966
  strict_mean_margin_delta: -0.066
  strict_protocol_match_rate: 0.0
  strict_over_generation_rate: 1.0

GLM4:
  decision: protocol_state_written_but_readout_competition_failed
  strict_mean_product_down_relative_delta: 0.604032
  strict_mean_margin_delta: -1.0608
  strict_protocol_match_rate: 0.0
  strict_over_generation_rate: 1.0

DS7B:
  decision: protocol_state_written_but_readout_competition_failed
  strict_mean_product_down_relative_delta: 0.682197
  strict_mean_margin_delta: -2.6493
  strict_protocol_match_rate: 0.0
  strict_over_generation_rate: 0.6667
```

最重要的客观现象：

```text
1. 严格协议提示确实改变 gate/up/product/down_out/residual_state；
2. 但 protocol_match_rate 仍然为 0；
3. strict protocol prompt 没有稳定改善 target_margin_vs_winner；
4. GLM4 和 DS7B 的 product/down_out 差分很大，但读出竞争反而更差；
5. target_seeded 产生最大内部差分，但它不是自然短答闭合，只能作为“目标已进入状态”的上界参照。
```

因此 Phase240 支持的判断是：

```text
stable_protocol_failure 不是协议状态完全没有进入内部层；
更像是协议状态进入了 gate/up/product/residual，
但没有转化为目标读出优势和 rollout 闭合。
```

### 5. 与 Phase239 的关系

Phase239 证明：

```text
继续加强 prompt wording / answer anchor 不能修复稳定协议失败。
```

Phase240 进一步说明：

```text
这不是因为提示词完全没有影响内部状态；
提示词影响了内部状态，
但这个影响没有赢得 readout competition，
也没有形成 closure control。
```

这把问题从：

```text
协议提示是否足够强
```

推进到：

```text
协议状态如何被读出、如何压过 continuation competitor、如何触发停止闭合。
```

### 6. 当前核心拼图

已经积累的核心拼图：

```text
1. 语言模式不是单一语义向量，而是模式网络；
2. 输出协议是一类模式，不等同于答案语义；
3. stable_protocol_failure 是稳定失败族，不是随机噪声；
4. target_alias 校准能修复部分假失败，但不能修复协议失败；
5. prompt trigger / anchor 变体不能解决稳定协议失败；
6. gate/up/product/residual 会响应协议变体；
7. 内部响应不等于读出成功；
8. target_seeded 是目标状态上界参照，不代表自然闭合；
9. readout competition 和 rollout / closure 是下一层关键障碍。
```

当前全局图谱链条更新为：

```text
TestCase
→ ScoringCalibration
→ PromptProtocol
→ ProtocolState
→ GateUpProduct
→ ResidualWrite
→ TargetPressure
→ CompetitorPressureField
→ ReadoutRegimeSelection
→ RolloutControl
→ ClosureTrace
```

### 7. 问题与硬伤

本阶段仍不是闭合证明，主要硬伤如下：

```text
1. Phase240 是 trace，不是 causal patch；
2. 只追踪最后输入位置的状态，尚未覆盖多 token rollout 轨迹；
3. target_seeded 与自然短答协议不同，不能混作成功样本；
4. 当前模型是小模型，内部编码机制可能比大模型粗糙 30% 到 50%；
5. product / residual 的大差分不必然表示“正确协议方向”，可能包含格式、目标、边界、续写等混合因素；
6. 还没有分离 target pressure 与 competitor pressure 的方向；
7. 还没有确认 stopping / closure state 是否存在可定位写入。
```

严格结论只能写成：

```text
协议变体能够改变内部状态；
但该内部变化没有稳定转化为目标读出优势和短答闭合。
```

不能写成：

```text
已经找到完整协议机制。
```

### 8. 图谱进度

Phase240 回写后：

```text
pattern_family_atlas: 0.56
behavior: 0.54
prompt_trigger: 0.32
gate_up_product: 0.30
residual_state: 0.30
readout_competition: 0.48
rollout: 0.05
closure: 0.10
model_internal_closure: 0.46
general_language_mechanism_confidence: 0.48
```

总体判断：

```text
语言模式图谱完成度约 56%；
模型内部闭合完成度约 46%；
对一般语言机制的信心约 48%。
```

### 9. 智能理论角度的关键洞察

语言能力至少包含三层不同机制：

```text
1. 内容机制：知道答案是什么；
2. 协议机制：知道答案应该以什么形式出现；
3. 闭合机制：知道什么时候停止，不进入续写或解释。
```

Phase240 的关键洞察是：

```text
协议状态可以进入内部网络，但不一定成为最终输出制度。
```

这说明智能系统中的“规则”可能不是单点开关，而是分布式竞争过程：

$$
\mathrm{Output}
=
\arg\max
\left(
P_{\mathrm{target}}
-
P_{\mathrm{competitor}}
-
B_{\mathrm{closure\ failure}}
\right)
$$

更接近当前结果的统一机制公式：

$$
\boxed{
\mathrm{LanguageState}_{t+1}
=
F(
\mathrm{ContentState}_t,
\mathrm{ProtocolState}_t,
\mathrm{BoundaryState}_t,
\mathrm{CompetitorField}_t,
\mathrm{ResidualWrite}_t,
\mathrm{RolloutControl}_t
)
}
$$

其中 Phase240 主要补上：

```text
ProtocolState → GateUpProduct → ResidualWrite
```

但仍没有闭合：

```text
ResidualWrite → ReadoutCompetition → RolloutControl → ClosureTrace
```

### 10. 下一阶段任务

Phase241 应进入：

```text
protocol rollout / closure trace
```

优先任务：

```text
1. 对 stable_protocol_failure 做多 token rollout 逐步追踪；
2. 比较第 1 个输出 token 正确但后续过生成的样例；
3. 分离 target pressure、continuation pressure、newline / period boundary pressure；
4. 追踪停止符、句号、换行、because、the 等竞争 token 的逐步 logit 变化；
5. 判断 closure failure 是第一步读出失败，还是后续 rollout 控制失败；
6. 只在定位清楚后再进入 causal suppression / patch。
```

阶段性目标不变：

```text
第一优先级：完成全局图谱；
第二优先级：机制闭合验证；
第三优先级：理论总结。
```

## Phase 241: 大规模模式族行为与读出图谱基准 [2026-07-07 16:12]

### 1. 阶段校正

Phase240 之后原计划进入：

```text
protocol rollout / closure trace
```

但结合最新判断，本阶段更合理的优先级是：

```text
先做语言模式图谱的大数据量测试，
再从大数据负面结果中筛选高价值内部追踪样本。
```

原因是前面多轮局部实验已经反复出现：

```text
小样本有效；
扩量后失效；
单模型有效；
跨模型失效；
局部 patch 有效；
全局闭合失败。
```

因此 Phase241 不继续局部追闭合，而是建立第一版：

```text
Large-Scale Pattern Atlas Benchmark
大规模模式族图谱基准
```

### 2. 新增脚本

新增：

```text
tests/gpt5/phase241_large_scale_pattern_atlas_benchmark.py
tests/gpt5/run_phase241_large_scale_pattern_atlas_benchmark.sh
```

运行顺序：

```text
qwen3 → GLM4 → DS7B
```

本轮正式规模：

```text
9 个模式族
72 个模式
每个模式 4 个样本
每个样本 6 个 prompt / protocol 变体
3 个模型
```

总运行量：

```text
72 × 4 × 6 × 3 = 5184
```

脚本参数支持继续扩展：

```text
SAMPLES_PER_MODE=50
SAMPLES_PER_MODE=100
```

但本轮先用 5184 条运行建立第一版全图谱分布，避免一次性把测试成本推得过高。

### 3. 覆盖的九大模式族

本轮覆盖：

```text
1. content_knowledge
2. output_protocol
3. reasoning_constraint
4. syntax_structure
5. language_action
6. cross_lingual
7. readout_competition
8. state_drift
9. closure
```

每个模式族 8 个模式，共 72 个模式。

本阶段不做 hook，只做：

```text
behavior；
next-token readout；
rollout up to 24 tokens；
winner / second competitor；
calibrated scoring；
negative result taxonomy。
```

### 4. 输出文件

结果目录：

```text
tests/result/phase241_large_scale_pattern_atlas_benchmark/large_scale_pattern_atlas_benchmark/
```

核心输出：

```text
phase241_cross_model_summary.json
phase241_large_scale_case_rows.jsonl
phase241_large_scale_behavior_rows.jsonl
phase241_large_scale_readout_rows.jsonl
phase241_negative_result_rows.jsonl
phase241_mode_trace_vectors.jsonl
phase241_family_failure_matrix.json
phase241_readout_regime_matrix.json
phase241_cross_model_observations.jsonl
phase241_cross_model_metrics.jsonl
phase241_cross_model_graph_edges.jsonl
phase241_large_scale_summary.md
```

同时回写：

```text
tests/result/pattern_family_atlas/v1/test_cases.jsonl
tests/result/pattern_family_atlas/v1/observations.jsonl
tests/result/pattern_family_atlas/v1/metrics.jsonl
tests/result/pattern_family_atlas/v1/graph_edges.jsonl
tests/result/pattern_family_atlas/v1/progress.json
tests/result/pattern_family_atlas/v1/summary.md
```

前端同步和构建：

```text
cd frontend
npm run sync:pattern-atlas
npm run build
```

构建通过，只有 chunk size 警告。

### 5. 测试原理

每条样本记录：

```text
case_id
family_id
mode_id
model
prompt_variant
target
target_aliases
output
semantic_match
protocol_match
winner_regime
second_competitor
target_margin_vs_winner
failure_type
negative_result
negative_category
mechanism_hint
should_enter_hook
```

基础行为分数：

$$
\boxed{
S
=
0.40 A
+
0.25 F
+
0.25 P
+
0.10 C
}
$$

其中：

```text
A = semantic_match
F = answer starts / reachable
P = protocol_match
C = closure_signal
```

读出边界：

$$
\boxed{
M
=
\mathrm{logit}(\mathrm{target})
-
\mathrm{logit}(\mathrm{winning\ competitor})
}
$$

模式脉络向量：

$$
\boxed{
\tau(P_i)
=
[
B_i,
T_i,
G_i,
R_i,
C_i,
O_i,
K_i
]
}
$$

本阶段实际填充：

```text
B_i = behavior signature
T_i = prompt variant trigger signature
C_i = readout competition signature
O_i = rollout signature
K_i = closure signature
```

其中：

```text
G_i 和 R_i 暂时为空，留给 Phase242 的内部追踪。
```

### 6. 负面结果分类

本阶段正式把负面结果纳入图谱，而不是视为失败。

分类包括：

```text
semantic_failure
protocol_negative
readout_negative
rollout_negative
closure_negative
```

负面结果公式：

$$
\boxed{
N_i
=
f(
\neg A_i,
\neg P_i,
M_i < 0,
\mathrm{OverGenerate}_i,
\neg C_i
)
}
$$

其中：

```text
A_i: 语义目标匹配
P_i: 输出协议匹配
M_i: 目标相对竞争机制边界
C_i: 闭合信号
```

### 7. 客观结果

跨模型汇总：

```text
case_count: 288
behavior_rows: 5184
readout_rows: 5184
negative_rows: 4223
mean_score: 0.5386
semantic_match_rate: 0.7355
protocol_match_rate: 0.1854
negative_rate: 0.8146
```

负面结果分类：

```text
rollout_negative: 1863
semantic_failure: 1371
closure_negative: 398
readout_negative: 363
protocol_negative: 228
```

单模型负面率：

```text
qwen3: 0.8038
GLM4: 0.7824
DS7B: 0.8576
```

这证明：

```text
扩量后负面结果不是偶然；
负面结果是当前图谱的主体材料。
```

### 8. 重要客观现象

1. 大量负面来自 rollout_negative。

```text
模型经常能触达答案或部分触达答案，
但输出展开、格式、停止和闭合不稳定。
```

2. semantic_failure 数量也很高。

```text
这说明第一版 case bank 中有一部分模式目标定义偏粗，
尤其 location_fact、causal_fact、classify 等模式，
需要继续做 target_alias 和 relation_schema 校准。
```

3. closure_negative 和 readout_negative 成为独立大类。

```text
这支持 Phase239 / Phase240 的方向：
语言机制不只是答案内容，
还包括协议、读出竞争、展开控制和停止闭合。
```

4. output_protocol / closure 中 one_word、short_answer、done_state_stable、boundary_stable 继续高失败。

```text
短答和闭合仍是核心困难。
```

5. 主要竞争机制仍集中在：

```text
the_continuation
period_stop
comma_repeat
answer_boundary
newline_boundary
be_continuation
```

### 9. 当前硬伤

本阶段是大规模行为/读出图谱，不是闭合证明。

主要硬伤：

```text
1. 每模式 4 个样本仍不是最终大数据规模；
2. 6 个 prompt 变体中 target_seeded 是上界参照，不是自然成功；
3. location_fact / causal_fact / classify 的目标定义偏粗，会增加 semantic_failure；
4. 当前 scoring 仍是规则评分，不是人工审阅或语义模型复核；
5. readout regime 集合仍是人工定义，还未自动发现；
6. 没有内部 hook，因此 G_i / R_i 还未填充；
7. 小模型内部编码粗糙，仍需保留 30% 到 50% 偏差空间。
```

因此不能说：

```text
已经完成语言模式图谱。
```

只能说：

```text
已经建立第一版跨九大模式族的大规模行为/读出负面结果图谱。
```

### 10. 图谱进度

Phase241 回写后：

```text
pattern_family_atlas: 0.62
behavior: 0.68
readout_competition: 0.56
large_scale_negative_taxonomy: 0.35
prompt_trigger: 0.32
gate_up_product: 0.30
residual_state: 0.30
rollout: 0.05
closure: 0.10
model_internal_closure: 0.46
general_language_mechanism_confidence: 0.50
```

当前总体判断：

```text
语言模式图谱约 62%；
行为层覆盖明显提高；
内部机制闭合仍约 46%；
一般语言机制信心约 50%。
```

### 11. 理论进展

Phase241 最重要的理论进展不是提出新名词，而是校正研究方法：

```text
负面结果不是实验失败；
负面结果是语言机制图谱的一部分。
```

当前更合理的语言机制公式：

$$
\boxed{
\mathrm{LanguageMechanism}
=
\sum_i
\alpha_i(x,t)
P_i(x,t)
}
$$

其中：

$$
\boxed{
P_i
=
\mathrm{TriggerTrace}_i
\circ
\mathrm{GateProductTrace}_i
\circ
\mathrm{ResidualWriteTrace}_i
\circ
\mathrm{ReadoutTrace}_i
\circ
\mathrm{CompetitorTrace}_i
\circ
\mathrm{RolloutTrace}_i
\circ
\mathrm{ClosureTrace}_i
}
$$

Phase241 主要补上：

```text
BehaviorTrace；
ReadoutTrace；
CompetitorTrace；
NegativeTaxonomy；
ModeTraceVector 的外部部分。
```

但仍缺：

```text
GateProductTrace；
ResidualWriteTrace；
ClosureTrace 的内部闭合。
```

### 12. 下一阶段任务

Phase242 应进入：

```text
High-Value Internal Trace Selection
高价值内部脉络选择
```

不是直接全量 hook，而是从 Phase241 结果中筛：

```text
1. 跨模型稳定失败；
2. 高 target pressure 但 winner 压制；
3. protocol_negative 高发模式；
4. closure_negative 高发模式；
5. rollout_negative 高发但 semantic_match 高的模式；
6. scoring 可能错误的 semantic_failure 高发模式；
7. 理论冲突样本。
```

优先补两件事：

```text
1. 校准 case bank：修复目标定义粗糙的模式；
2. 选出 100 到 300 条 hook 候选，进入 gate/up/product、residual、rollout trace。
```

阶段目标保持：

```text
先完成全局图谱；
再做高价值内部追踪；
最后做少数模式闭合验证。
```

## Phase 242: 负面结果多标签化与高价值内部脉络选择 [2026-07-07 17:11]

### 1. 阶段目标

本阶段承接 Phase241。

Phase241 已经完成：

```text
9 个模式族；
72 个模式；
288 个基础样例；
5184 条跨模型行为 / 读出记录；
4223 条单标签负面结果。
```

但 Phase241 的主要硬伤是：

```text
1. 负面结果还是单标签；
2. semantic_failure 中混有 case bank / scoring 风险；
3. 还没有从负面结果中筛出高价值内部追踪候选；
4. 不能直接把所有负面结果送进 hook。
```

因此 Phase242 的目标是：

```text
把 Phase241 的第一版大图谱升级成可筛选、可校准、可进入内部追踪的数据基础。
```

本阶段不重新跑模型，不做 hook，不做 probe，不做 ablation。

### 2. 新增脚本

新增：

```text
tests/gpt5/phase242_negative_multilabel_and_trace_selection.py
```

输入：

```text
tests/result/phase241_large_scale_pattern_atlas_benchmark/large_scale_pattern_atlas_benchmark/
```

输出目录：

```text
tests/result/phase242_negative_multilabel_and_trace_selection/negative_multilabel_and_trace_selection/
```

核心输出：

```text
phase242_summary.json
phase242_multilabel_negative_rows.jsonl
phase242_high_value_hook_candidates.jsonl
phase242_case_bank_calibration_rows.jsonl
phase242_trace_selection_matrix.json
phase242_internal_trace_plan.md
phase242_observations.jsonl
phase242_metrics.jsonl
phase242_graph_edges.jsonl
```

并回写：

```text
tests/result/pattern_family_atlas/v1/observations.jsonl
tests/result/pattern_family_atlas/v1/metrics.jsonl
tests/result/pattern_family_atlas/v1/graph_edges.jsonl
tests/result/pattern_family_atlas/v1/progress.json
tests/result/pattern_family_atlas/v1/summary.md
```

前端同步和构建通过：

```text
cd frontend
npm run sync:pattern-atlas
npm run build
```

### 3. 算法原理

Phase241 的单标签负面结果：

```text
semantic_failure
protocol_negative
readout_negative
rollout_negative
closure_negative
```

升级为多标签向量：

$$
\boxed{
\mathbf{N}_i
=
[
N_{\mathrm{semantic}},
N_{\mathrm{protocol}},
N_{\mathrm{readout}},
N_{\mathrm{rollout}},
N_{\mathrm{closure}},
N_{\mathrm{scoring}}
]
}
$$

其中：

```text
N_semantic: semantic_match = false；
N_protocol: semantic_match = true 且 protocol_match = false；
N_readout: target_margin_vs_winner < -1 或 target_rank 较差；
N_rollout: over_generation 或输出过长；
N_closure: semantic_match = true 且 closure_signal = false；
N_scoring: 目标定义、跨语言、动作类型、target_rank 与字符串匹配冲突等评分风险。
```

候选筛选公式：

$$
\boxed{
\mathrm{CandidateScore}
=
0.25 S_{\mathrm{crossmodel}}
+
0.20 S_{\mathrm{semantic}}
+
0.15 S_{\mathrm{failure}}
+
0.15 S_{\mathrm{winner}}
+
0.15 S_{\mathrm{divergence}}
+
0.10 S_{\mathrm{margin}}
-
0.20 S_{\mathrm{scoring}}
}
$$

其中：

```text
S_crossmodel: 三模型稳定失败程度；
S_semantic: 语义正确程度；
S_failure: protocol/readout/rollout/closure 负面强度；
S_winner: winner_regime 跨模型稳定程度；
S_divergence: 跨模型成功/失败差异；
S_margin: target margin 接近或大于 0 但仍失败；
S_scoring: 样例/评分风险惩罚。
```

筛选原则：

```text
优先选择语义正确但协议、读出、展开或闭合失败的样本；
不优先选择 target_seeded；
不优先选择明显 scoring risk 且没有机制价值的 semantic_failure。
```

### 4. 客观结果

输入：

```text
source_behavior_rows: 5184
source_negative_rows: 4223
```

输出：

```text
multilabel_rows: 5184
high_value_candidates: 300
hook_ready_candidates: 300
case_bank_review_rows: 288
manual_review_cases: 95
```

多标签计数：

```text
semantic: 1371
protocol: 2852
readout: 4644
rollout: 3476
closure: 3345
scoring: 930
```

这说明 Phase241 的单标签低估了失败的重叠程度。

尤其重要的是：

```text
readout、rollout、closure 大量重叠；
很多样本不是单一失败，而是“语义触达 + 协议失败 + 读出竞争 + 展开/闭合失败”的复合状态。
```

候选原因统计：

```text
semantic_correct_rollout_failure: 286
cross_model_stable_failure: 286
stable_readout_competitor: 216
semantic_correct_closure_failure: 209
semantic_correct_protocol_failure: 194
high_target_pressure_protocol_failure: 38
cross_model_divergence: 14
```

推荐下一步测试：

```text
readout_competitor_trace: 216
protocol_gate_product_residual_trace: 36
stepwise_rollout_trace: 29
rollout_closure_trace: 13
cross_model_structure_comparison: 6
```

样例库复核热点：

```text
content_knowledge::function_answer
content_knowledge::part_whole
content_knowledge::material_answer
content_knowledge::location_fact
content_knowledge::causal_fact
language_action::summarize
language_action::translate
language_action::classify
language_action::rewrite
language_action::compare
cross_lingual::ZH_to_ZH
cross_lingual::EN_to_FR
cross_lingual::FR_to_EN
cross_lingual::cross_lingual_reasoning
```

### 5. 关键进展

Phase242 的进展不是模型新结果，而是数据结构进展。

核心进展：

```text
1. 负面结果从单标签升级为多标签；
2. 证明大量失败是复合失败，而不是单点失败；
3. 将 scoring / case bank 风险从机制失败中分离出来；
4. 从 5184 条记录中筛出 300 条高价值内部追踪候选；
5. 明确下一步内部测试优先级：readout competitor trace 最多，其次 protocol gate/product/residual trace；
6. 产出 case bank 校准清单，避免把目标定义粗糙带进 hook。
```

### 6. 问题和硬伤

本阶段仍有明显硬伤：

```text
1. 没有重新跑模型，完全依赖 Phase241 数据；
2. 多标签规则仍是启发式，不是人工复核；
3. 300 条 hook-ready candidates 中仍可能混有评分误差；
4. readout 标签很宽，因为 target_margin_vs_winner < -1 会覆盖大量样本；
5. case bank 校准只是标记 review，不是实际修正样本；
6. 还没有做模式脉络聚类；
7. 还没有做内部 G_i / R_i 填充。
```

因此不能说：

```text
已经找到内部机制候选的最终集合。
```

只能说：

```text
已经形成第一版内部追踪候选池和样例库校准任务池。
```

### 7. 当前图谱进度

Phase242 回写后：

```text
pattern_family_atlas: 0.66
behavior: 0.68
readout_competition: 0.56
large_scale_negative_taxonomy: 0.50
case_bank_calibration: 0.28
high_value_trace_selection: 0.34
prompt_trigger: 0.32
gate_up_product: 0.30
residual_state: 0.30
rollout: 0.05
closure: 0.10
model_internal_closure: 0.46
general_language_mechanism_confidence: 0.51
```

阶段判断：

```text
全局图谱继续前进；
闭合没有明显前进；
内部机制仍未进入新一轮实测。
```

### 8. 智能理论角度的关键洞察

Phase242 说明：

```text
语言失败不是单轴失败。
```

同一个输出可以同时满足：

```text
语义接近；
协议失败；
读出被压；
展开漂移；
闭合失败；
评分存在风险。
```

因此语言模式不能用单个 success / failure 标签描述，而应表示为多维状态：

$$
\boxed{
\mathrm{LanguageFailure}_i
=
(
N_{\mathrm{semantic}},
N_{\mathrm{protocol}},
N_{\mathrm{readout}},
N_{\mathrm{rollout}},
N_{\mathrm{closure}},
N_{\mathrm{scoring}}
)
}
$$

这也反过来说明，语言机制应表示为：

$$
\boxed{
P_i
=
\mathrm{TriggerTrace}_i
\circ
\mathrm{ReadoutTrace}_i
\circ
\mathrm{CompetitorTrace}_i
\circ
\mathrm{RolloutTrace}_i
\circ
\mathrm{ClosureTrace}_i
\circ
\mathrm{InternalTrace}_i
}
$$

当前 Phase242 主要补强：

```text
NegativeTaxonomy；
CaseBankCalibration；
HighValueInternalTraceSelection。
```

### 9. 下一阶段任务

Phase243 应进入：

```text
mode trace clustering and candidate validation
模式脉络聚类与候选验证
```

优先任务：

```text
1. 对 300 条候选做去重、分层、聚类；
2. 建立 explore / validate / frozen 数据划分；
3. 从 300 条中选 60 到 120 条第一批内部追踪样本；
4. 重点覆盖 readout_competitor_trace、protocol_gate_product_residual_trace、stepwise_rollout_trace；
5. 对 95 个 manual_review case 做样例库 v2 修正；
6. 为后续每模式 50 样本的大规模二版测试准备冻结 case bank。
```

仍然不要直接闭合。

## Phase 243: 候选聚类、数据划分与 case bank v2 [2026-07-07 17:30]

### 1. 阶段目标

本阶段承接 Phase242。

Phase242 已经完成：

```text
5184 条记录多标签化；
300 条高价值内部追踪候选；
95 个 manual_review case；
readout / protocol / rollout / closure 候选优先级。
```

但 Phase242 的候选池仍然不能直接进入 hook，因为：

```text
1. 300 条候选过多；
2. 候选之间有重复机制；
3. 还没有 explore / validate / frozen 数据划分；
4. case bank 仍有评分风险；
5. 内部追踪需要按测试类型平衡。
```

Phase243 的目标是：

```text
候选去重；
候选聚类；
数据划分；
选出第一批内部追踪样本；
生成 case bank v2 标记。
```

本阶段仍不重新跑模型，不做 hook，不做 probe，不做 ablation，不做闭合。

### 2. 新增脚本

新增：

```text
tests/gpt5/phase243_candidate_clustering_and_casebank_v2.py
```

输入：

```text
tests/result/phase242_negative_multilabel_and_trace_selection/negative_multilabel_and_trace_selection/
```

输出目录：

```text
tests/result/phase243_candidate_clustering_and_casebank_v2/candidate_clustering_and_casebank_v2/
```

核心输出：

```text
phase243_pattern_mining_summary.json
phase243_candidate_dedup_rows.jsonl
phase243_candidate_cluster_rows.jsonl
phase243_trace_selection_rows.jsonl
phase243_case_bank_v2_rows.jsonl
phase243_data_split_rows.jsonl
phase243_internal_trace_plan.md
phase243_observations.jsonl
phase243_metrics.jsonl
phase243_graph_edges.jsonl
```

前端同步和构建通过：

```text
cd frontend
npm run sync:pattern-atlas
npm run build
```

### 3. 聚类算法

候选聚类使用外部模式脉络特征：

```text
family_id；
mode_id；
recommended_next_test；
stable_winner_regime；
failure_group；
margin_bucket。
```

聚类键：

$$
\boxed{
C_i
=
(
F_i,
M_i,
T_i,
W_i,
G_i,
B_i
)
}
$$

其中：

```text
F_i = family_id；
M_i = mode_id；
T_i = recommended_next_test；
W_i = stable_winner_regime；
G_i = failure_group；
B_i = margin_bucket。
```

数据划分：

$$
\boxed{
D
=
D_{\mathrm{explore}}
\cup
D_{\mathrm{validate}}
\cup
D_{\mathrm{frozen}}
}
$$

实际使用稳定 hash 划分，并根据 case bank 风险调整：

```text
explore: 用于发现规律；
validate: 用于验证候选规律；
frozen: 保留，不参与调参。
```

第一批内部追踪样本按附件建议比例选择：

```text
readout_competitor_trace: 40%
protocol_gate_product_residual_trace: 25%
stepwise_rollout_trace: 20%
rollout_closure_trace: 10%
cross_model_structure_comparison: 5%
```

### 4. 客观结果

```text
input_candidates: 300
dedup_candidates: 300
cluster_count: 157
trace_selection_rows: 100
case_bank_v2_rows: 288
manual_review_cases: 95
```

数据划分：

```text
explore: 168
validate: 70
frozen: 62
```

第一批内部追踪样本：

```text
readout_competitor_trace: 40
protocol_gate_product_residual_trace: 25
stepwise_rollout_trace: 20
rollout_closure_trace: 10
cross_model_structure_comparison: 5
```

高频聚类现象：

```text
reasoning_constraint::if_then → be_continuation / period_stop 稳定读出竞争；
reasoning_constraint::counterfactual → the_continuation 稳定读出竞争；
state_drift::early_correct_late_drift → the_continuation 稳定读出竞争；
readout_competition::because_reason → the_continuation 稳定读出竞争；
content_knowledge::object_attribute → period_stop / comma_repeat 稳定竞争；
output_protocol::explain_answer → the_continuation 稳定竞争。
```

### 5. 关键进展

Phase243 的关键进展：

```text
1. 300 条候选被压缩成 157 个机制簇；
2. 建立 explore / validate / frozen 划分；
3. 选出 100 条第一批内部追踪样本；
4. 内部追踪样本按 readout / protocol / rollout / closure / cross-model 比例平衡；
5. 形成 case bank v2 标记；
6. 为 Phase244 第一批内部 trace 提供明确输入。
```

这一步使后续 hook 不再是盲目抽样，而是来自大数据图谱、负面多标签和聚类筛选。

### 6. 问题和硬伤

本阶段仍有硬伤：

```text
1. dedup_candidates 仍为 300，说明候选按 case+variant 没有明显重复，机制层重复主要体现在 cluster；
2. 聚类是规则聚类，不是自动 embedding 聚类；
3. case bank v2 仍是标记层，还没有人工修正 target_aliases；
4. 100 条 trace selection 仍可能有评分风险；
5. 没有产生新的模型行为结果；
6. 没有填充 G_i / R_i；
7. 没有内部因果验证。
```

因此不能说：

```text
已经完成内部机制选择。
```

只能说：

```text
已经完成第一批内部追踪样本的结构化选择。
```

### 7. 图谱进度

Phase243 回写后：

```text
pattern_family_atlas: 0.69
behavior: 0.68
readout_competition: 0.56
large_scale_negative_taxonomy: 0.50
case_bank_calibration: 0.36
high_value_trace_selection: 0.48
candidate_clustering: 0.40
prompt_trigger: 0.32
gate_up_product: 0.30
residual_state: 0.30
rollout: 0.05
closure: 0.10
model_internal_closure: 0.46
general_language_mechanism_confidence: 0.52
```

总体判断：

```text
全局图谱继续推进；
候选选择明显前进；
内部机制闭合仍没有前进。
```

### 8. 理论进展

Phase243 进一步确认：

```text
语言模式图谱不能只列样本；
必须把样本组织成候选簇、验证集和冻结集。
```

当前研究循环更清楚：

$$
\boxed{
D_{\mathrm{large}}
\rightarrow
\mathrm{NegativeMultilabel}
\rightarrow
\mathrm{CandidateCluster}
\rightarrow
D_{\mathrm{explore/validate/frozen}}
\rightarrow
\mathrm{InternalTraceBatch}
\rightarrow
\mathrm{CausalTest}
}
$$

这意味着后续规律必须先经过：

```text
候选聚类；
验证集复核；
冻结集保留；
内部 trace；
因果测试。
```

不能直接从局部候选上升为理论。

### 9. 下一阶段任务

Phase244 应进入：

```text
first internal trace batch
第一批内部追踪样本测试
```

但仍不要直接闭合。

建议优先顺序：

```text
1. 先做 40 条 readout_competitor_trace；
2. 再做 25 条 protocol_gate_product_residual_trace；
3. 再做 20 条 stepwise_rollout_trace；
4. 少量做 rollout_closure_trace 和 cross_model_structure_comparison；
5. 每次仍按 qwen3 → GLM4 → DS7B 顺序运行；
6. 输出必须继续写入 Pattern Atlas 固定格式。
```

Phase244 的目标不是证明闭合，而是填充：

```text
G_i: gate/product signature；
R_i: residual signature；
更细的 readout competitor trace；
stepwise rollout trace。
```

## Phase 244: 第一批内部追踪样本测试 [2026-07-07 17:47]

### 1. 任务判断

本阶段分析的 Phase243 判断基本正确。

Phase243 的价值不是直接发现新的语言理论，而是把 Phase241/242 的大数据负结果整理成可以进入内部追踪的高价值样本集。它把下一步任务从“继续扩行为数据”推进到：

```text
高价值候选样本
→ 内部组件追踪
→ 读出竞争追踪
→ rollout 早期轨迹追踪
→ 再进入因果验证
```

因此 Phase244 不应该追求闭合，而应该先完成第一批内部 trace 数据。

### 2. 测试原理

本阶段使用 Phase243 选出的 100 条 trace selection 样本，并回连 Phase241 的固定行为数据行。每个样本在每个模型上执行：

```text
1. 找到 selected variant 的 prompt_variant；
2. 找到同 case_id 的 full baseline；
3. 捕获指定层的 gate/up/product/down_out/recomputed_product；
4. 捕获 observe layers 的 residual state；
5. 计算 selected variant 相对 full baseline 的 delta；
6. 重新读取 lm_head 的 readout competitor；
7. 对 stepwise_rollout_trace / rollout_closure_trace / cross_model_structure_comparison 样本做 4 步 argmax rollout；
8. 输出 Pattern Atlas 固定格式 observations / metrics / graph_edges。
```

核心度量为：

$$
\Delta_{\mathrm{rel}}(x, x_{\mathrm{full}})
=
\frac{\lVert x - x_{\mathrm{full}} \rVert_2}
{\max(\lVert x_{\mathrm{full}} \rVert_2,\epsilon)}
$$

读出边界变化为：

$$
\Delta m
=
m_{\mathrm{variant}} - m_{\mathrm{full}}
=
\left(z_{\mathrm{target}} - z_{\mathrm{winner}}\right)_{\mathrm{variant}}
-
\left(z_{\mathrm{target}} - z_{\mathrm{winner}}\right)_{\mathrm{full}}
$$

本阶段仍然只证明 trace 相关性，不证明因果闭合。

### 3. 脚本和结果文件

新增脚本：

```text
tests/gpt5/phase244_first_internal_trace_batch.py
tests/gpt5/run_phase244_first_internal_trace_batch.sh
```

结果目录：

```text
tests/result/phase244_first_internal_trace_batch/first_internal_trace_batch/
```

主要输出：

```text
phase244_cross_model_summary.json
phase244_cross_model_component_trace_rows.jsonl
phase244_cross_model_residual_trace_rows.jsonl
phase244_cross_model_readout_trace_rows.jsonl
phase244_cross_model_stepwise_rollout_rows.jsonl
phase244_cross_model_observations.jsonl
phase244_cross_model_metrics.jsonl
phase244_cross_model_graph_edges.jsonl
phase244_first_internal_trace_report.md
```

同时同步到：

```text
tests/result/pattern_family_atlas/v1/
frontend/public/vis_data/pattern_family_atlas/v1/
```

### 4. 测试规模

按要求依次运行：

```text
qwen3 → GLM4 → DS7B
```

完成情况：

```text
selected rows: 100
completed cases per model: 100
missing rows: 0
models: qwen3, glm4, deepseek7b
```

跨模型总输出：

```text
component_trace_rows: 1500
residual_trace_rows: 900
readout_trace_rows: 300
stepwise_rollout_rows: 420
observation_rows: 3120
metric_rows: 144
graph_edges: 45
```

trace 任务分布：

```text
readout_competitor_trace: 120
protocol_gate_product_residual_trace: 75
stepwise_rollout_trace: 60
rollout_closure_trace: 30
cross_model_structure_comparison: 15
```

### 5. 客观结果

总体均值：

```text
mean_component_relative_delta: 0.479805
mean_residual_relative_delta: 0.338150
mean_readout_margin_delta_vs_full: 7.433960
stable_winner_match_rate: 0.5533
```

分模型结果：

```text
qwen3:
  component delta: 0.419004
  residual delta: 0.303248
  readout margin delta: 12.172012
  stable winner match rate: 0.6600

GLM4:
  component delta: 0.449837
  residual delta: 0.338994
  readout margin delta: 6.471758
  stable winner match rate: 0.4800

DS7B:
  component delta: 0.570575
  residual delta: 0.372209
  readout margin delta: 3.658110
  stable winner match rate: 0.5200
```

最强内部组件变化主要出现在 DS7B 的 stepwise_rollout_trace / rollout_closure_trace / cross_model_structure_comparison 样本中，product 与 recomputed_product 的相对变化最高接近 0.90。

但 readout margin delta 最大的是 qwen3，而不是 DS7B。这说明：

```text
内部组件变化强
≠
读出边界改善强
```

这是一个重要校准结果。

### 6. 进展分析

本阶段完成的核心拼图：

```text
1. Phase243 的候选样本可以无缺失地回连 Phase241 行为数据；
2. 可以在三模型上稳定生成固定格式内部 trace；
3. gate/product、residual、readout、stepwise rollout 四类数据已经进入同一个图谱格式；
4. 首次得到 100 个候选样本 × 3 模型的内部组件变化矩阵；
5. 发现“组件变化强度”和“读出改善强度”存在明显模型差异。
```

这比 Phase240 的局部 protocol trace 更进一步，因为 Phase244 不再只测短答协议，而是覆盖 Phase243 挑出的多类高价值失败样本。

### 7. 问题和硬伤

第一，本阶段仍是 trace，不是 causal test。内部向量变化与输出变化同现，不等于证明该组件是必要原因。

第二，当前 baseline 使用同 case 的 full variant。它适合观察 prompt variant 变化，但不能完全分离：

```text
语义内容变化；
格式约束变化；
目标词提示变化；
长度和边界变化。
```

第三，stepwise rollout 只做 argmax 前 4 步。它能观察早期漂移，但还不能覆盖长程生成闭合。

第四，小模型偏差仍然很大。qwen3、GLM4、DS7B 的内部机制可能比真实大型模型更粗糙，当前机制图谱可能有 30% 到 50% 的偏差，尤其不能把单模型强信号直接提升为通用语言机制。

第五，stable_winner_match_rate 只有 0.5533，说明 Phase243 的 stable winner 聚类并不是强闭合变量，只能作为候选分组依据。

### 8. 图谱进度

当前 Pattern Atlas 进度评估：

```text
pattern_family_atlas: 0.72
candidate_clustering: 0.40
case_bank_calibration: 0.36
high_value_trace_selection: 0.55
first_internal_trace_batch: 0.30
gate_up_product_signature: 0.38
residual_state_signature: 0.37
readout_competition_trace: 0.58
stepwise_rollout_trace: 0.18
causal_closure: 0.10
general_language_mechanism_confidence: 0.53
```

总体判断：

```text
语言模式图谱已经进入内部追踪阶段；
但机制闭合仍然很早；
当前最强进展是数据管线和第一批 G/R/readout/rollout trace 拼图。
```

### 9. 智能理论视角

Phase244 支持“语言是动态模式网络”的路线，但进一步校准了一个关键点：

```text
模式不是单一向量；
模式更像多层、多组件、多读出边界共同决定的运行轨迹。
```

当前更合理的机制表达是：

$$
P_i(t)
=
\left[
G_i(t),
U_i(t),
M_i(t),
D_i(t),
R_i(t),
O_i(t)
\right]
$$

其中：

```text
G_i: gate signature
U_i: up projection signature
M_i: product signature
D_i: down output signature
R_i: residual trajectory
O_i: output/readout regime
```

模式是否成功，不应只看某个组件是否变化，而要看：

$$
\mathrm{Success}(P_i)
=
f\left(
\Delta G_i,
\Delta M_i,
\Delta R_i,
\Delta O_i,
\mathrm{Rollout}_i
\right)
$$

本阶段最重要的洞察是：

```text
内部变化可能负责“写入候选状态”；
读出边界负责“选择输出 regime”；
rollout 轨迹负责“是否维持正确模式”。
```

这三者不能再混成一个变量。

### 10. 下一阶段任务

Phase245 应继续处于同一阶段性目标：

```text
完成语言模式图谱的内部追踪拼图；
暂不进入最终闭合。
```

建议任务：

```text
Phase245: internal trace validation and frozen split audit
```

具体步骤：

```text
1. 从 Phase244 中选出 component delta 高、readout margin 高、二者不一致的三类样本；
2. 在 validate split 和 frozen split 上复测这些 trace signature；
3. 检查 product/down_out/residual/readout 四类指标是否稳定；
4. 对最稳定的少数候选再设计 causal patch/ablation；
5. 不再盲目扩大局部 patch，先判断 trace signature 是否跨样本稳定。
```

当前阶段结论：

```text
Phase244 完成了第一批内部追踪数据；
证明图谱管线可运行；
发现内部组件变化与读出改善不等价；
但距离机制闭合仍然较远。
```

## Phase 245: 内部追踪签名验证与冻结集审计 [2026-07-07 18:17]

### 1. 任务判断

本阶段分析的 Phase244 复盘内容总体正确。

Phase244 已经完成：

```text
大数据负面结果图谱
→ 候选聚类
→ 第一批内部追踪
```

但附件中的关键提醒也正确：

```text
Phase244 是 trace；
不是 causal test；
不是 probe；
不是 ablation；
不是 closure validation。
```

因此 Phase245 继续属于同一阶段性目标：

```text
完成语言模式图谱的内部追踪拼图；
暂不进入最终闭合。
```

本阶段没有重新加载模型，因为 Phase244 已经完成 100 个候选 × 3 模型的内部追踪。Phase245 先对已有 trace 数据做验证、审计和候选筛选，避免过早进入 patch。

### 2. 测试原理

输入数据：

```text
tests/result/phase244_first_internal_trace_batch/first_internal_trace_batch/
```

主要使用：

```text
phase244_cross_model_component_trace_rows.jsonl
phase244_cross_model_residual_trace_rows.jsonl
phase244_cross_model_readout_trace_rows.jsonl
phase244_cross_model_stepwise_rollout_rows.jsonl
```

Phase245 将每个：

```text
model + case_id + variant_id
```

聚合成一个 trace signature：

```text
component_mean_delta
product_down_mean_delta
residual_mean_delta
readout_margin_delta_vs_full
winner_sequence
rollout_drift_score
closure_proxy_score
```

然后划分为五类：

```text
high_component_high_readout
high_component_low_readout
low_component_high_readout
readout_boundary_weak_change
mixed_signature
```

核心分析公式：

$$
\rho_{\mathrm{group}}
=
\mathrm{corr}
\left(
\Delta_{\mathrm{component}},
\Delta_{\mathrm{readout}}
\right)
$$

按以下维度分组：

```text
model
model + recommended_next_test
model + family_id
model + family_id + recommended_next_test
model + winning_regime
model + signature_class
```

冻结集稳定性采用 validate / frozen 之间的均值差距近似：

$$
\mathrm{Stability}
=
\frac{1}
{1 + \mathrm{mean}
\left(
|\Delta_{\mathrm{validate}}-\Delta_{\mathrm{frozen}}|
\right)}
$$

注意：本阶段的 factor projection 是代理分解，不是真正的原始向量正交分解。由于 Phase244 没有保存原始向量，只能用 trace/readout/rollout 元数据估计：

```text
target_proxy
protocol_proxy
boundary_proxy
competitor_proxy
closure_proxy
rollout_drift_proxy
```

### 3. 新增脚本和输出

新增脚本：

```text
tests/gpt5/phase245_trace_signature_validation_and_frozen_audit.py
```

结果目录：

```text
tests/result/phase245_trace_signature_validation_and_frozen_audit/trace_signature_validation_and_frozen_audit/
```

主要输出：

```text
phase245_summary.json
phase245_trace_signature_rows.jsonl
phase245_component_readout_correlation_rows.jsonl
phase245_validate_frozen_audit_rows.jsonl
phase245_factor_projection_rows.jsonl
phase245_causal_test_candidate_rows.jsonl
phase245_observations.jsonl
phase245_metrics.jsonl
phase245_graph_edges.jsonl
phase245_trace_signature_report.md
```

并同步到 Pattern Atlas 固定格式：

```text
tests/result/pattern_family_atlas/v1/
frontend/public/vis_data/pattern_family_atlas/v1/
```

### 4. 客观结果

本阶段输出规模：

```text
signature_rows: 300
correlation_rows: 94
validate_frozen_audit_rows: 79
proxy_factor_projection_rows: 300
causal_test_candidate_rows: 30
observation_rows: 3000
metric_rows: 178
graph_edges: 40
```

signature class 分布：

```text
mixed_signature: 143
high_component_low_readout: 71
high_component_high_readout: 46
low_component_high_readout: 36
readout_boundary_weak_change: 4
```

数据划分分布：

```text
explore: 177
frozen: 66
validate: 57
```

模型级 component-readout 相关性：

```text
qwen3: 0.117579
GLM4: -0.071452
DS7B: -0.005691
```

这个结果非常关键：

```text
在模型级整体上，component delta 与 readout margin delta 基本不相关。
```

虽然局部分组中出现接近 1 或 -1 的相关性，但很多分组只有 3 到 4 条样本，不能当成稳定机制结论。

### 5. 关键进展

Phase245 进一步确认 Phase244 的核心校准：

$$
\Delta_{\mathrm{component}}
\not\Rightarrow
\Delta_{\mathrm{readout}}
$$

并把它从观察结论推进到分组统计结论：

```text
整体模型级相关性弱；
局部分组可能强相关；
但强相关分组常常样本量过小；
因此必须先做 validate/frozen 稳定性审计。
```

本阶段筛出 30 条下一步 causal test 候选。最高分候选包括：

```text
glm4 / reasoning_constraint / if_then / no_answer_anchor / frozen
qwen3 / output_protocol / table_answer / explain_instruction / explore
glm4 / reasoning_constraint / if_then / no_answer_anchor / explore
qwen3 / reasoning_constraint / if_then / explain_instruction / frozen
qwen3 / state_drift / boundary_takeover / short_answer_instruction / validate
```

推荐的下一步因果测试主要是：

```text
component_ablation_and_readout_margin_test
target_injection_vs_competitor_suppression
```

### 6. 稳定性审计

较稳定的候选组包括：

```text
DS7B / high_component_high_readout
DS7B / high_component_low_readout
DS7B / the_continuation
qwen3 / answer_boundary
qwen3 / output_protocol
GLM4 / because_reason
```

但需要谨慎：

```text
有些稳定性高的组 frozen 行数很少；
有些强相关组 row_count 只有 3；
这些只能作为下一步采样优先级，不能作为机制结论。
```

### 7. 问题和硬伤

第一，Phase245 没有新模型 forward，只是对 Phase244 结果做审计。因此它不能替代真正复测，也不能证明因果。

第二，factor projection 只是代理分解。真正的方向分解需要在 Phase246 或之后保存 raw delta vectors，并构造：

```text
target direction
protocol direction
boundary direction
competitor direction
closure direction
```

第三，局部分组相关性容易被小样本放大。Phase245 已经暴露这个问题，所以后续必须要求：

```text
高相关 + 足够样本 + validate/frozen 稳定
```

三者同时满足，才进入因果测试。

第四，小模型偏差仍然存在。当前所有结论都是 qwen3 / GLM4 / DS7B 的小模型内部图谱，不能直接等价为大型模型或真实语言机制。

### 8. 图谱进度

Phase245 后 Pattern Atlas 进度：

```text
pattern_family_atlas: 0.73
candidate_clustering: 0.42
case_bank_calibration: 0.38
high_value_trace_selection: 0.58
first_internal_trace_batch: 0.36
trace_signature_validation: 0.32
gate_up_product_signature: 0.42
residual_state_signature: 0.40
readout_competition_trace: 0.61
stepwise_rollout_trace: 0.21
proxy_factor_decomposition: 0.16
causal_closure: 0.10
general_language_mechanism_confidence: 0.54
```

总体进度判断：

```text
图谱阶段继续推进；
内部追踪签名开始可筛选；
但 causal closure 仍然只有 0.10。
```

### 9. 智能理论视角

Phase245 的关键洞察是：

```text
语言模式轨迹不能用单一“内部变化强度”解释；
必须同时看组件变化、残差变化、读出边界、winner 序列、rollout 漂移和闭合代理。
```

当前更稳妥的模式轨迹公式是：

$$
P_i(t)
=
\left[
G_i(t),
U_i(t),
M_i(t),
D_i(t),
R_i(t),
O_i(t),
L_i(t),
K_i(t)
\right]
$$

其中：

```text
G_i: gate signature
U_i: up projection signature
M_i: product signature
D_i: down output signature
R_i: residual trajectory
O_i: readout regime
L_i: rollout trajectory
K_i: closure state
```

Phase245 进一步说明：

```text
如果只看 G/M/D 的范数差分，很容易误判；
如果只看 O 的 readout margin，也会忽略内部写入；
如果不看 L/K，无法判断输出是否真正闭合。
```

### 10. 下一阶段任务

下一阶段仍属于同一大阶段：

```text
内部追踪图谱 → 稳定签名 → 小规模因果测试准备。
```

建议 Phase246 进入：

```text
focused causal validation for stable trace signatures
稳定追踪签名的小规模因果验证
```

但只选 10 到 15 条候选，不要大规模 patch。

优先样本：

```text
1. frozen / validate 中的高分 causal candidates；
2. high_component_low_readout，用 component ablation 测是否只是噪声写入；
3. low_component_high_readout，用 target injection vs competitor suppression 测读出是否主导；
4. high_component_high_readout，用 component replacement 测是否能改变 readout margin。
```

Phase246 的成功标准应是：

```text
不是闭合语言机制；
而是确认至少 2 到 3 类 signature 是否具有因果必要性或充分性迹象。
```

当前结论：

```text
Phase245 没有推进闭合；
但它把 Phase244 的内部追踪数据转成了可审计的签名、稳定性和因果候选。
下一步可以更克制地进入小规模因果验证。
```

## Phase 246: 稳定追踪签名的小规模因果验证 [2026-07-07 21:05]

### 1. 任务判断

本阶段分析的 Phase245 复盘内容总体正确。

Phase245 已经把 Phase244 的内部 trace 转成：

```text
trace signature；
component-readout correlation；
validate / frozen audit；
proxy factor decomposition；
causal test candidates。
```

它的结论也正确：

```text
整体模型级 component delta 与 readout margin delta 基本不相关；
局部强相关常常样本量太小；
下一步应该做 focused causal validation；
不能直接做 closure validation。
```

因此 Phase246 继续处于同一阶段性目标：

```text
语言模式图谱 → 内部追踪 → 签名审计 → 小规模因果迹象验证。
```

本阶段仍不追求闭合，只验证少数稳定签名是否有必要性或充分性迹象。

### 2. 测试原理

Phase246 从 Phase245 的 30 条 causal candidates 中选取全局前 15 条，按模型依次运行：

```text
qwen3 → GLM4 → DS7B
```

每个候选执行：

```text
1. 找到 selected variant 和 full baseline；
2. 捕获 source layer 的 product / down_out；
3. 捕获 final observe layer 的 residual；
4. 保存 raw delta vectors；
5. 执行 no_intervention；
6. 执行 down_out_delta_ablation；
7. 执行 target_unembed_injection；
8. 执行 top_competitor_suppression；
9. 比较 target margin、winner regime、短 rollout。
```

关键差分向量：

$$
\Delta d
=
d_{\mathrm{variant}} - d_{\mathrm{full}}
$$

down_out 差分消融：

$$
h' = h - \lambda \Delta d
$$

target 注入：

$$
h' = h + \lambda v_{\mathrm{target}}
$$

competitor 抑制：

$$
h' = h - \lambda v_{\mathrm{competitor}}
$$

本阶段保存了 raw delta vectors，用于后续真正的方向分解。

### 3. 新增脚本和输出

新增脚本：

```text
tests/gpt5/phase246_focused_causal_validation.py
tests/gpt5/run_phase246_focused_causal_validation.sh
```

结果目录：

```text
tests/result/phase246_focused_causal_validation/focused_causal_validation/
```

主要输出：

```text
phase246_cross_model_summary.json
phase246_causal_validation_rows.jsonl
phase246_component_ablation_rows.jsonl
phase246_target_injection_rows.jsonl
phase246_competitor_suppression_rows.jsonl
phase246_rollout_closure_perturbation_rows.jsonl
phase246_raw_delta_vector_manifest.json
phase246_causal_validation_report.md
raw_vectors/
```

并继续写入 Pattern Atlas：

```text
observations.jsonl
metrics.jsonl
graph_edges.jsonl
progress.json
```

### 4. 测试规模

正式测试规模：

```text
candidate_count: 15
validation_rows: 60
component_ablation_rows: 15
target_injection_rows: 15
competitor_suppression_rows: 15
rollout_closure_perturbation_rows: 60
raw_delta_vectors: 15
missing_rows: 0
```

分模型：

```text
qwen3: 10 candidates
GLM4: 4 candidates
DS7B: 1 candidate
```

这说明 Phase245 的前 15 个高分候选偏向 qwen3，DS7B 候选较少。这个分布来自候选评分，不应解释为模型能力差异。

### 5. 客观结果

总效果：

```text
necessity_signal_count: 3
target_injection_gain_count: 14
competitor_suppression_gain_count: 10
```

平均 margin 变化：

```text
mean_ablation_margin_delta: 1.768229
mean_target_injection_margin_delta: 8.764063
mean_competitor_suppression_margin_delta: -0.696354
```

effect label 分布：

```text
sufficiency_or_readout_gain_signal: 24
weak_or_no_sufficiency_signal: 15
ablation_improved_margin_opposite_signal: 8
intervention_harmed_margin: 6
weak_or_no_necessity_signal: 4
necessity_signal_margin_dropped: 3
```

### 6. 关键发现

第一，target_unembed_injection 是最稳定的正向干预：

```text
15 条中 14 条出现 sufficiency_or_readout_gain_signal；
平均 readout margin 增益为 8.764063。
```

这说明当前候选中大量问题更像：

```text
target pressure 不足；
或 target direction 没有足够进入最终读出。
```

第二，down_out_delta_ablation 只有 3 条出现 necessity_signal_margin_dropped。

更重要的是，有 8 条出现：

```text
ablation_improved_margin_opposite_signal
```

也就是说，抑制原本的 down_out 差分反而改善了目标边界。这是一个关键负结果：

```text
高 component delta 经常不是必要目标写入；
它可能包含 competitor / continuation / protocol 噪声写入。
```

第三，top_competitor_suppression 分化明显：

```text
10 条正向；
5 条 harmed；
平均值为 -0.696354。
```

说明“抑制 top token / top competitor”这个粗方法不稳定。它有时压住竞争，有时也破坏了目标相关结构。

### 7. 进展分析

Phase246 完成了 Phase245 要求的第一轮小规模因果迹象验证。

本阶段新增拼图：

```text
1. raw delta vectors 已保存；
2. down_out 差分消融开始给出必要性迹象；
3. target unembed 注入给出较稳定充分性迹象；
4. competitor suppression 被证明是混合且不稳定的粗干预；
5. high component delta 中混入非目标写入的可能性大幅提高。
```

这比 Phase245 前进了一步，因为 Phase245 只是候选审计，Phase246 已经开始测试：

```text
某个内部差分被削弱或增强时，readout margin 是否改变。
```

但这仍然不是闭合。

### 8. 问题和硬伤

第一，Phase246 的干预仍然很粗。target_unembed_injection 使用的是输出嵌入方向，不等于真实 target factor direction。

第二，top_competitor_suppression 使用 top token 方向，不能代表完整 competitor regime。一个 regime 可能由多个 token、多个方向、多个层共同构成。

第三，down_out_delta_ablation 只消融 source layer 的 last-token down_out 差分，不能覆盖多层传播和后续重写。

第四，候选分布不均衡：

```text
qwen3: 10
GLM4: 4
DS7B: 1
```

因此跨模型结论必须谨慎。

第五，当前 rollout 仍然很短，不能说明 closure。

第六，当前模型都是小模型或中小模型，需要保留 30% 到 50% 偏差空间。

### 9. 图谱进度

Phase246 后 Pattern Atlas 进度：

```text
pattern_family_atlas: 0.74
candidate_clustering: 0.42
case_bank_calibration: 0.39
high_value_trace_selection: 0.60
first_internal_trace_batch: 0.38
trace_signature_validation: 0.35
focused_causal_validation: 0.20
raw_delta_vector_archive: 0.18
gate_up_product_signature: 0.44
residual_state_signature: 0.41
readout_competition_trace: 0.63
stepwise_rollout_trace: 0.23
proxy_factor_decomposition: 0.18
causal_closure: 0.12
general_language_mechanism_confidence: 0.55
```

总体判断：

```text
已经进入小规模因果迹象阶段；
target pressure 路线获得较强正信号；
component delta 必要性只得到弱正 + 强校准；
closure 仍然很早。
```

### 10. 智能理论视角

Phase246 的关键洞察是：

```text
内部大差分不一定是“正确模式写入”；
它可能是目标、竞争、续写、协议、格式等多种状态的混合。
```

当前更合理的链条是：

$$
\Delta \mathrm{Component}
\rightarrow
\Delta \mathrm{Residual}
\rightarrow
\Delta \mathrm{Readout}
\rightarrow
\Delta \mathrm{Rollout}
\rightarrow
\Delta \mathrm{Closure}
$$

但 Phase246 说明：

```text
第一段箭头不稳定；
第二段 readout 更容易被 target direction 直接影响；
competitor suppression 必须做 regime 级分解，不能只压 top token。
```

这对破解语言编码机制很关键：

```text
真正的机制不在“某个大向量变化”；
而在“哪些变化沿 target / competitor / boundary / closure 方向传播并进入读出与展开”。
```

### 11. 下一阶段任务

下一阶段仍属于同一大阶段，但应从粗因果干预转向真实方向分解：

```text
Phase247: raw-vector factor direction decomposition
```

任务：

```text
1. 使用 Phase246 保存的 raw delta vectors；
2. 构造 target / competitor / boundary / protocol / closure 方向；
3. 对 delta_down_out、delta_product、delta_residual 做投影；
4. 检查 target projection 是否解释 target_injection_gain；
5. 检查 competitor projection 是否解释 suppression harmed / gain 的分化；
6. 只对方向分解稳定的候选进入下一轮因果验证。
```

Phase247 不应扩大闭合测试，而应先回答：

```text
大差分到底是什么方向？
target pressure 和 competitor pressure 如何混在一起？
```

当前阶段结论：

```text
Phase246 获得了第一批因果迹象；
target direction 充分性迹象较强；
down_out 差分必要性较弱且混合；
competitor suppression 需要 regime 级分解；
机制闭合仍未完成。
```

## Phase 247: 原始向量因子方向分解 [2026-07-07 21:26]

### 1. 任务判断

本阶段分析的 Phase246 复盘内容总体正确。

Phase246 的关键进展是：

```text
target_unembed_injection 有强正向读出迹象；
down_out_delta_ablation 必要性较弱且经常反向；
top_competitor_suppression 不稳定；
raw delta vectors 已经保存。
```

因此 Phase247 不应继续扩大 patch，也不应进入 closure validation，而应先回答：

```text
大差分到底是什么方向？
target pressure 和 competitor pressure 如何混在一起？
为什么 target injection 有效，而 down_out ablation 必要性弱？
为什么 competitor suppression 有时有效、有时有害？
```

本阶段没有重新加载模型，也没有新的 forward。它直接分析 Phase246 保存的 raw vectors。

### 2. 测试原理

输入：

```text
tests/result/phase246_focused_causal_validation/focused_causal_validation/phase246_raw_delta_vector_manifest.json
tests/result/phase246_focused_causal_validation/focused_causal_validation/raw_vectors/
tests/result/phase246_focused_causal_validation/focused_causal_validation/phase246_causal_validation_rows.jsonl
```

每个 raw vector 文件包含：

```text
delta_down_out
delta_product
delta_residual
target_direction
competitor_direction
target_token_id
top_token_id
```

Phase247 对三类差分向量做投影：

```text
delta_down_out
delta_product
delta_residual
```

投影到：

```text
target direction
top competitor direction
empirical regime direction
```

其中 empirical regime direction 来自 Phase246 中同一模型、同一 regime group 的 top competitor directions 平均。

核心投影：

$$
a_j
=
\left\langle
\Delta x,
v_j
\right\rangle
$$

正交化后投影：

$$
a^{\perp}_j
=
\left\langle
\Delta x,
v^{\perp}_j
\right\rangle
$$

注意：Phase247 只对已经保存的 raw target / top competitor directions 做真实投影。protocol / boundary / closure 的真实原始方向仍未捕获，不能冒充已解决。

### 3. 新增脚本和输出

新增脚本：

```text
tests/gpt5/phase247_raw_vector_factor_decomposition.py
```

结果目录：

```text
tests/result/phase247_raw_vector_factor_decomposition/raw_vector_factor_decomposition/
```

主要输出：

```text
phase247_factor_decomposition_summary.json
phase247_factor_direction_rows.jsonl
phase247_regime_direction_rows.jsonl
phase247_raw_delta_projection_rows.jsonl
phase247_projection_prediction_rows.jsonl
phase247_regime_test_candidate_rows.jsonl
phase247_observations.jsonl
phase247_metrics.jsonl
phase247_graph_edges.jsonl
phase247_factor_direction_report.md
```

并同步 Pattern Atlas 固定格式。

### 4. 客观结果

输出规模：

```text
raw_vector_rows: 15
factor_direction_rows: 20
regime_direction_rows: 5
projection_rows: 135
prediction_rows: 45
next_test_candidate_rows: 15
observation_rows: 135
metric_rows: 46
graph_edges: 32
```

方向覆盖：

```text
target directions: available
top competitor directions: available but regime incomplete
empirical continuation_regime directions: 3
empirical boundary_regime directions: 2
protocol / boundary / closure raw directions: missing
direction_gap_count: 9
```

候选路线分布：

```text
competitor_regime_candidate: 8
target_pressure_direction_candidate: 4
mixed_or_weak_candidate: 3
```

这说明 Phase246 后真正优先级已经从：

```text
单 token competitor suppression
```

转向：

```text
regime-level competitor decomposition
```

### 5. 关键发现

第一，Phase247 证明 Phase246 已保存的 raw vectors 可以进入真实方向投影流程。

这意味着研究不再停留在：

```text
component delta norm
```

而开始进入：

```text
delta direction composition
```

第二，target projection 能筛出一批 target_pressure_direction_candidate，例如：

```text
qwen3 / output_protocol_json_answer_0000 / one_word_strict
qwen3 / output_protocol_table_answer_0002 / explain_instruction
qwen3 / state_drift_boundary_takeover_0001 / short_answer_instruction
deepseek7b / reasoning_constraint_if_then_0001 / no_answer_anchor
```

第三，更多候选被归到 competitor_regime_candidate。这与 Phase246 中 top_competitor_suppression 分化明显相吻合：

```text
单 top token 不够；
需要机制场方向。
```

第四，best_target_prediction_corr 和 best_competitor_suppression_prediction_corr 接近 1，但这些主要来自小样本分组，不应上升为机制规律。

更稳妥的结论是：

```text
Phase247 已经能把候选分成 target pressure、competitor regime、mixed 三类；
但分组相关性还需要更大样本验证。
```

### 6. 问题和硬伤

第一，Phase247 没有新增模型 forward，只是 raw vector 二次分析。

第二，empirical regime directions 只来自 15 个 Phase246 候选，样本太少。

第三，protocol / boundary / closure 的 raw directions 缺失。这是当前最大缺口。

第四，continuation_regime 和 boundary_regime 方向仍然由 top competitor directions 聚合而来，不是真正的多 token regime bank。

第五，DS7B 只有 1 个候选，不能做可靠跨模型方向结论。

### 7. 图谱进度

Phase247 后 Pattern Atlas 进度：

```text
pattern_family_atlas: 0.75
candidate_clustering: 0.42
case_bank_calibration: 0.39
high_value_trace_selection: 0.61
first_internal_trace_batch: 0.38
trace_signature_validation: 0.36
focused_causal_validation: 0.22
raw_delta_vector_archive: 0.24
raw_vector_factor_decomposition: 0.20
regime_field_direction_bank: 0.10
gate_up_product_signature: 0.45
residual_state_signature: 0.42
readout_competition_trace: 0.64
stepwise_rollout_trace: 0.23
causal_closure: 0.12
general_language_mechanism_confidence: 0.56
```

总体判断：

```text
已进入原始向量方向分解阶段；
target / top competitor 方向可分析；
regime field 方向仍弱；
protocol / boundary / closure 方向仍缺失；
closure 没有实质推进。
```

### 8. 智能理论视角

Phase247 对智能理论的关键校准是：

```text
语言模式不是“大向量变化”；
语言模式是多方向因子的混合写入和竞争读出。
```

更准确的内部差分公式是：

$$
\Delta x
=
a_t v_{\mathrm{target}}
+
a_c v_{\mathrm{competitor}}
+
a_p v_{\mathrm{protocol}}
+
a_b v_{\mathrm{boundary}}
+
a_k v_{\mathrm{closure}}
+
\epsilon
$$

但 Phase247 只能实测其中一部分：

```text
v_target: 已有 raw direction
v_competitor: 已有 top-token raw direction，但 regime 不完整
v_protocol: 缺失
v_boundary: 缺失
v_closure: 缺失
```

因此当前理论进展不是“公式闭合”，而是明确了缺失项。

### 9. 下一阶段任务

下一阶段仍属于同一大阶段，不应进入闭合。

建议 Phase248：

```text
regime-level direction bank construction
机制级方向库构建
```

任务：

```text
1. 为 each model 构造 regime token bank；
2. 至少覆盖 continuation、answer_boundary、newline_boundary、period_stop、because_reason、comma_repeat；
3. 使用输出嵌入或自然 trace 方向构造 regime field direction；
4. 重新投影 Phase246 raw deltas；
5. 检查 regime projection 是否解释 suppression gain/harm；
6. 选择少数候选进入 regime-level causal test。
```

Phase248 的成功标准不是闭合，而是：

```text
把 competitor 从 top token 升级为 regime field；
减少 Phase246 中 competitor suppression 的不稳定性；
明确哪些 regime 方向需要真正因果测试。
```

当前结论：

```text
Phase247 完成了第一版 raw-vector factor decomposition；
证明大差分可被方向投影审计；
发现下一步关键缺口是 regime-level direction bank；
机制闭合仍未完成。
```

## Phase 248: 机制级方向库构建 [2026-07-07 22:59]

### 1. 任务判断

本阶段分析的 Phase247 复盘内容总体正确。

Phase247 的核心进展是：

```text
研究对象从 component delta norm
升级为 delta direction composition。
```

它证明 raw delta vectors 可以进入方向投影流程，但也暴露了核心缺口：

```text
target direction 已有；
top competitor direction 已有但过粗；
protocol / boundary / closure 真实方向仍缺失；
competitor 需要从 top token 升级为 regime field。
```

因此 Phase248 继续属于同一阶段目标：

```text
语言模式图谱 → 原始向量方向分解 → 机制级方向库 → 机制级因果测试候选。
```

本阶段不做闭合验证，只构建第一版 regime-level direction bank，并重新投影 Phase246 raw deltas。

### 2. 测试原理

Phase248 顺序加载：

```text
qwen3 → GLM4 → DS7B
```

但不做生成 forward，只使用 tokenizer 和 output embedding 构造 regime token-bank directions。

覆盖的 regime：

```text
continuation_regime
answer_boundary_regime
newline_boundary_regime
period_stop_regime
because_reason_regime
comma_repeat_regime
protocol_short_regime
```

机制方向构造公式：

$$
v_{\mathrm{regime}}
=
\frac{1}{|R|}
\sum_{t\in R}
v_t
$$

其中：

```text
R: regime token bank
v_t: token t 的 output embedding direction
```

然后对 Phase246 保存的 raw deltas 重新投影：

```text
delta_down_out
delta_product
delta_residual
```

投影公式：

$$
a_{\mathrm{regime}}
=
\left\langle
\Delta x,
v_{\mathrm{regime}}
\right\rangle
$$

### 3. 新增脚本和输出

新增脚本：

```text
tests/gpt5/phase248_regime_level_direction_bank.py
tests/gpt5/run_phase248_regime_level_direction_bank.sh
```

结果目录：

```text
tests/result/phase248_regime_level_direction_bank/regime_level_direction_bank/
```

主要输出：

```text
phase248_cross_model_summary.json
phase248_regime_direction_rows.jsonl
phase248_regime_projection_rows.jsonl
phase248_projection_prediction_rows.jsonl
phase248_regime_test_candidate_rows.jsonl
phase248_observations.jsonl
phase248_metrics.jsonl
phase248_graph_edges.jsonl
phase248_regime_direction_report.md
```

并同步 Pattern Atlas 固定格式。

### 4. 客观结果

总体输出：

```text
regime_bank_rows: 21
raw_vector_rows: 15
projection_rows: 315
prediction_rows: 84
regime_test_candidate_rows: 15
observation_rows: 315
metric_rows: 105
graph_edges: 55
```

每个模型 7 个 regime bank，token 覆盖：

```text
continuation_regime: 10
answer_boundary_regime: 7
newline_boundary_regime: 3
period_stop_regime: 5
because_reason_regime: 6
comma_repeat_regime: 4
protocol_short_regime: 6
```

下一轮候选路线分布：

```text
continuation_regime_test: 9
protocol_regime_test: 5
reason_regime_test: 1
```

这说明当前最突出的 regime-level 问题仍然是：

```text
continuation / protocol 接管；
而不是单个 top token 竞争。
```

### 5. 关键发现

第一，Phase248 成功把 competitor 从 top token 推进到 token-bank regime direction。

这比 Phase247 的 empirical regime direction 更稳，因为 Phase247 的 regime direction 只来自 15 个候选中的 top competitor，而 Phase248 使用显式 token bank。

第二，continuation_regime_test 成为最大候选类：

```text
9 / 15
```

这与早期 Phase209 以来反复出现的 continuation takeover、over-generation、the_continuation、be_continuation 现象一致。

第三，protocol_regime_test 有 5 条，说明输出协议不是表面格式问题，而可能是一个读出方向/边界方向问题。

第四，reason_regime_test 只有 1 条，且来自 DS7B 单样本，不能强化解释。

第五，部分 projection-prediction correlation 很高，但仍受样本数限制。尤其 GLM4 只有 4 个 raw vectors，不能把高相关当成机制定律。

### 6. 问题和硬伤

第一，Phase248 的 regime bank 仍然来自 output embedding token bank，不等于自然内部 regime field。

第二，每个 regime token bank 仍然较小，尤其 newline、comma、period 等边界类还很粗。

第三，当前没有多层、多步、多位置的 regime field，只是静态 output direction。

第四，DS7B 只有 1 个 raw vector，因此 DS7B 的 prediction rows 为 0，不能做统计判断。

第五，本阶段仍然没有 closure validation，没有 DoneStateStable，也没有 ModelStopExecuted。

### 7. 图谱进度

Phase248 后 Pattern Atlas 进度：

```text
pattern_family_atlas: 0.76
candidate_clustering: 0.42
case_bank_calibration: 0.39
high_value_trace_selection: 0.62
first_internal_trace_batch: 0.38
trace_signature_validation: 0.36
focused_causal_validation: 0.22
raw_delta_vector_archive: 0.25
raw_vector_factor_decomposition: 0.22
regime_field_direction_bank: 0.22
gate_up_product_signature: 0.45
residual_state_signature: 0.42
readout_competition_trace: 0.66
stepwise_rollout_trace: 0.23
causal_closure: 0.12
general_language_mechanism_confidence: 0.57
```

总体判断：

```text
机制级方向库已经有第一版；
continuation/protocol 是当前最强候选方向；
target/competitor 方向分解继续推进；
闭合仍没有实质推进。
```

### 8. 智能理论视角

Phase248 进一步支持：

```text
语言机制不是单词级竞争，而是 regime field 级竞争。
```

更合理的读出竞争公式是：

$$
O(t)
=
\arg\max_r
\left\langle
h_t,
v_{\mathrm{regime},r}
\right\rangle
$$

其中：

```text
r: continuation / boundary / reason / protocol / closure 等机制；
v_regime,r: 机制方向场。
```

这比单 token readout 更接近语言运行机制。

### 9. 下一阶段任务

下一阶段仍属于同一阶段目标，不应闭合。

建议 Phase249：

```text
regime-level causal validation
机制级因果验证
```

任务：

```text
1. 选取 Phase248 的 continuation_regime_test 和 protocol_regime_test 高分候选；
2. 用 regime direction 替代 top token direction 做 suppression / injection；
3. 比较 target margin、winner regime、rollout token 序列；
4. 判断 regime-level 干预是否比 top-token suppression 更稳定；
5. 只保留能稳定减少 continuation takeover 或 protocol drift 的候选。
```

成功标准：

```text
不是闭合；
而是证明 regime-level intervention 比 top-token intervention 更稳定或更可解释。
```

当前结论：

```text
Phase248 完成了第一版机制级方向库；
把竞争机制从 top token 推进到 regime token bank；
发现 continuation/protocol 是下一步最优先测试方向；
机制闭合仍未完成。
```

## Phase 249: 机制级因果验证 [2026-07-07 23:14]

### 任务来源

本阶段接续 Phase248 的结论：

```text
top token intervention（最高词元干预）不够稳定；
需要验证 regime-level intervention（机制级干预）是否更接近真实竞争机制。
```

因此 Phase249 不再只看 output embedding token bank（输出嵌入词元库）的投影相关性，而是在同一批 Phase248 候选上进行因果式读出干预。

### 测试脚本与结果文件

新增脚本：

```text
tests/gpt5/phase249_regime_level_causal_validation.py
tests/gpt5/run_phase249_regime_level_causal_validation.sh
```

结果目录：

```text
tests/result/phase249_regime_level_causal_validation/regime_level_causal_validation/
```

核心输出：

```text
phase249_regime_causal_validation_rows.jsonl
phase249_regime_suppression_rows.jsonl
phase249_regime_injection_rows.jsonl
phase249_target_vs_regime_comparison_rows.jsonl
phase249_rollout_effect_rows.jsonl
phase249_regime_causal_validation_report.md
phase249_cross_model_summary.json
```

并已同步写入 Pattern Atlas（模式图谱）固定格式：

```text
tests/result/pattern_family_atlas/v1/observations.jsonl
tests/result/pattern_family_atlas/v1/metrics.jsonl
tests/result/pattern_family_atlas/v1/graph_edges.jsonl
tests/result/pattern_family_atlas/v1/progress.json
frontend/public/vis_data/pattern_family_atlas/v1/
```

前端可视化数据已同步，`frontend` 构建通过。构建仍有既有大 chunk（大代码块）警告，不影响本阶段数据读取。

### 测试原理

Phase249 使用 Phase246 保存的 raw vector（原始向量）与 Phase248 的 regime direction（机制方向）：

```text
continuation_regime_test -> continuation_regime
protocol_regime_test -> protocol_short_regime
reason_regime_test -> because_reason_regime
```

在最终观测层对 hidden state（隐藏状态）做四类对照：

```text
no_intervention（无干预）
regime_suppression（机制抑制）
regime_injection（机制注入）
target_injection_replay（目标注入复现）
top_token_suppression_replay（最高竞争词元抑制复现）
```

核心公式：

$$
\boxed{
h' =
h - \lambda v_{\mathrm{regime}}
}
$$

$$
\boxed{
h' =
h + \lambda v_{\mathrm{regime}}
}
$$

对照指标：

$$
\boxed{
\Delta M =
M(h') - M(h)
}
$$

其中：

```text
M = target_logit - winning_regime_logit
```

即目标相对胜出机制的 readout margin（读出边界）。

### 客观结果

三模型顺序完成：

```text
qwen3 -> GLM4 -> DS7B
```

跨模型统计：

```text
candidate_count: 15
validation_rows: 75
regime_suppression_rows: 15
regime_injection_rows: 15
target_vs_regime_comparison_rows: 15
rollout_effect_rows: 75
missing_rows: 0
```

总体均值：

```text
mean_regime_suppression_margin_delta: +1.464583
mean_regime_injection_margin_delta: -1.470833
mean_top_token_replay_margin_delta: -2.23125
regime_better_than_top_token_replay_count: 7 / 15
regime_better_than_phase246_top_token_count: 6 / 15
```

候选路线：

```text
continuation_regime_test: 9
protocol_regime_test: 5
reason_regime_test: 1
```

失败类型提示：

```text
competitor_regime_failure: 7
target_pressure_failure: 7
mixed_or_unresolved: 1
```

模型分解：

```text
qwen3:
  candidate_count: 10
  mean_regime_suppression_margin_delta: +2.275
  mean_regime_injection_margin_delta: -2.0125
  regime_better_than_top_token_replay_count: 6 / 10

GLM4:
  candidate_count: 4
  mean_regime_suppression_margin_delta: -0.25
  mean_regime_injection_margin_delta: -0.453125
  regime_better_than_top_token_replay_count: 1 / 4

DS7B:
  candidate_count: 1
  mean_regime_suppression_margin_delta: +0.21875
  mean_regime_injection_margin_delta: -0.125
  regime_better_than_top_token_replay_count: 0 / 1
```

最强正例集中在 qwen3 的 continuation_regime（续写机制）：

```text
phase241_output_protocol_table_answer_0002 / explain_instruction:
  regime_suppression_margin_delta: +7.125

phase241_state_drift_boundary_takeover_0001 / short_answer_instruction:
  regime_suppression_margin_delta: +6.0625

phase241_reasoning_constraint_if_then_0000 / no_answer_anchor:
  regime_suppression_margin_delta: +4.0625

phase241_output_protocol_json_answer_0000 / one_word_strict:
  regime_suppression_margin_delta: +3.0625
```

最明显问题集中在 protocol_short_regime（短答协议机制）：

```text
qwen3 / protocol_short_regime 多个样本出现 margin 下降；
GLM4 / protocol_short_regime 也不稳定；
说明 protocol direction（协议方向）不能直接等同于“修复方向”。
```

### 正确性分析

Phase248 的判断基本正确：

```text
语言输出确实更像 regime-level competition（机制级竞争），
而不是单一 top-token competition（最高词元竞争）。
```

Phase249 给出了第一层因果迹象：

```text
在 qwen3 上，抑制 continuation_regime 经常改善目标 readout margin；
机制级抑制比 top-token replay 更稳定；
regime injection 与 suppression 呈相反方向，说明方向不是完全随机。
```

但跨模型不闭合：

```text
GLM4 不支持稳定正效应；
DS7B 只有 1 个样本，不能统计；
总体正均值主要由 qwen3 拉动。
```

因此 Phase249 的结论只能写成：

```text
弱正结果 + 强校准结果。
```

### 主要硬伤

1. token-bank regime direction（词元库机制方向）仍是 proxy（代理），不是自然内部方向。

2. regime_suppression 的正效应主要来自 qwen3，不能作为跨模型机制闭合。

3. protocol_short_regime 不是单调修复方向，可能混合了：

```text
短答协议；
边界锚点；
目标词压力；
停止控制；
格式遵循。
```

4. rollout（生成展开）只测了早期 4 token，不能证明 closure（闭合）。

5. 当前模型仍是小模型，内部编码机制可能存在 30% 到 50% 偏差。

### 图谱进度

本阶段后图谱进度更新为：

```text
pattern_family_atlas: 0.77
regime_field_direction_bank: 0.24
regime_level_causal_validation: 0.18
readout_competition_trace: 0.67
stepwise_rollout_trace: 0.24
causal_closure: 0.12
general_language_mechanism_confidence: 0.58
```

### 阶段结论

Phase249 证明：

```text
continuation_regime 是真实竞争机制候选；
机制级抑制在部分模型和样本上优于 top-token 抑制；
但是 token-bank direction 还不能代表自然内部机制场。
```

下一步仍属于同一大阶段：机制方向图谱阶段。

因此继续自动进入 Phase250：

```text
从静态 token-bank direction
推进到 natural contrast direction（自然对照方向）。
```

## Phase 250: 自然机制方向提取 [2026-07-07 23:18]

### 阶段目标

Phase249 的最大硬伤是：

```text
regime direction 来自 output embedding token bank，
不是自然运行轨迹中形成的内部方向。
```

Phase250 因此构造自然对照：

```text
one_word_strict vs explain_instruction
full vs no_answer_anchor
target_seeded vs full
short_answer_instruction vs explain_instruction
```

从同一 case（案例）的不同 prompt variant（提示变体）中提取 hidden-state delta（隐藏状态差分），建立 natural regime direction bank（自然机制方向库）。

### 测试脚本与结果文件

新增脚本：

```text
tests/gpt5/phase250_natural_regime_direction_extraction.py
tests/gpt5/run_phase250_natural_regime_direction_extraction.sh
```

结果目录：

```text
tests/result/phase250_natural_regime_direction_extraction/natural_regime_direction_extraction/
```

核心输出：

```text
phase250_natural_direction_sample_rows.jsonl
phase250_natural_direction_rows.jsonl
phase250_natural_projection_rows.jsonl
phase250_natural_prediction_rows.jsonl
phase250_natural_regime_direction_report.md
phase250_cross_model_summary.json
```

自然方向张量保存于：

```text
tests/result/phase250_natural_regime_direction_extraction/natural_regime_direction_extraction/natural_vectors/
```

并已同步 Pattern Atlas（模式图谱）固定格式和前端可视化数据。

### 测试设计

每个模型从 Phase241 大规模图谱中抽取：

```text
9 个模式族；
每个模式族 2 个 case；
每个 case 5 类自然对照；
每模型 90 个自然对照样本；
三模型共 270 个自然对照样本。
```

五类自然对照：

```text
natural_protocol_short:
  one_word_strict - explain_instruction

natural_continuation_explain:
  explain_instruction - one_word_strict

natural_answer_boundary:
  full - no_answer_anchor

natural_target_seed:
  target_seeded - full

natural_concise_answer:
  short_answer_instruction - explain_instruction
```

核心公式：

$$
\boxed{
v_{\mathrm{natural}, r}
=
\mathrm{Normalize}
\left(
\frac{1}{N}
\sum_{i=1}^{N}
\left(
h_i^{(+)} - h_i^{(-)}
\right)
\right)
}
$$

再将 Phase246 的 raw delta vector（原始差分向量）投影到自然方向：

$$
\boxed{
s_{i,r}
=
\cos
\left(
\Delta h_i,
v_{\mathrm{natural}, r}
\right)
}
$$

观察：

```text
自然方向投影是否能解释 target_injection_gain；
自然方向投影是否能解释 competitor_suppression_gain；
自然方向是否比 token-bank direction 更接近真实轨迹。
```

### 客观结果

跨模型统计：

```text
sample_rows: 270
direction_rows: 150
projection_rows: 225
prediction_rows: 30
observation_rows: 225
metric_rows: 180
graph_edges: 30
missing_rows: 0
```

每类自然对照样本数：

```text
natural_protocol_short: 54
natural_continuation_explain: 54
natural_answer_boundary: 54
natural_target_seed: 54
natural_concise_answer: 54
```

模型分解：

```text
qwen3:
  sample_rows: 90
  direction_rows: 50
  projection_rows: 150
  prediction_rows: 15

GLM4:
  sample_rows: 90
  direction_rows: 50
  projection_rows: 60
  prediction_rows: 15

DS7B:
  sample_rows: 90
  direction_rows: 50
  projection_rows: 15
  prediction_rows: 0
```

DS7B 仍因 Phase246 raw vectors 太少，无法形成预测相关性。

### 主要观测

GLM4 的若干自然方向相关性较大，但只有 4 个 raw-vector 行：

```text
delta_down_out x natural_concise_answer:
  corr_projection_competitor_suppression_gain: -0.786991

delta_down_out x natural_target_seed:
  corr_projection_competitor_suppression_gain: +0.782785

delta_down_out x natural_answer_boundary:
  corr_projection_competitor_suppression_gain: -0.694349

delta_residual x natural_target_seed:
  corr_projection_competitor_suppression_gain: +0.656338
```

qwen3 的自然方向相关性更弱，但样本行更多：

```text
delta_down_out x natural_target_seed:
  rows: 10
  corr_projection_competitor_suppression_gain: -0.292339

delta_down_out x natural_answer_boundary:
  rows: 10
  corr_projection_competitor_suppression_gain: -0.287995

delta_residual x natural_target_seed:
  rows: 10
  corr_projection_competitor_suppression_gain: -0.203867
```

这说明：

```text
自然方向库已经建立；
但自然方向对 Phase246 少量 raw vectors 的解释力还不强；
当前更像方向素材库，而不是可闭合公式。
```

### 正确性分析

Phase250 的方向是正确的，因为它修复了 Phase248/249 的核心缺口：

```text
静态 output direction（输出方向）
不能代表自然内部轨迹；
必须从真实 prompt contrast（提示对照）提取方向。
```

但是 Phase250 不应被过度解释。

原因：

```text
自然对照方向来自 prompt-level contrast（提示层对照）；
它可能同时包含语义、格式、边界、目标词、停止控制等多个因素；
还没有完成 orthogonalization（正交化）与因果干预。
```

因此 Phase250 的结论是：

```text
自然机制方向库已建立；
方向库具备后续因果验证价值；
尚未证明自然方向能闭合语言编码机制。
```

### 当前核心拼图更新

新增拼图：

```text
NaturalRegimeDirectionBank（自然机制方向库）
NaturalProtocolShort（自然短答协议方向）
NaturalContinuationExplain（自然解释续写方向）
NaturalAnswerBoundary（自然回答边界方向）
NaturalTargetSeed（自然目标种子方向）
NaturalConciseAnswer（自然简答方向）
```

当前机制图谱从：

```text
token-bank regime direction
```

推进到：

```text
token-bank direction
+ natural contrast direction
+ raw delta projection
+ intervention comparison
```

机制公式更新为：

$$
\boxed{
\Delta h
=
\sum_r
\beta_r
v_{\mathrm{natural}, r}
+
\sum_k
\gamma_k
v_{\mathrm{tokenbank}, k}
+
\epsilon
}
$$

其中：

```text
v_natural,r 是自然对照方向；
v_tokenbank,k 是输出词元库方向；
epsilon 是未解释混合残差。
```

### 图谱进度

本阶段后图谱进度更新为：

```text
pattern_family_atlas: 0.78
case_bank_calibration: 0.40
high_value_trace_selection: 0.63
raw_delta_vector_archive: 0.26
raw_vector_factor_decomposition: 0.23
regime_field_direction_bank: 0.30
natural_regime_direction_bank: 0.20
regime_level_causal_validation: 0.18
residual_state_signature: 0.45
readout_competition_trace: 0.68
causal_closure: 0.12
general_language_mechanism_confidence: 0.59
```

### 硬伤与瓶颈

1. 当前自然方向样本虽覆盖 9 个模式族，但每族只有 2 个 case，仍是第一版方向库。

2. 自然对照不是纯变量对照，例如：

```text
one_word_strict - explain_instruction
```

同时混合了短答协议、解释欲望、长度控制和结束倾向。

3. GLM4 的相关性较大但 raw vector 行数只有 4，不能强化结论。

4. DS7B 仍缺 raw vector 样本，不能参与相关性判断。

5. Phase250 没有做自然方向 causal intervention（因果干预），所以仍不是闭合。

### 阶段结论

Phase249 和 Phase250 合起来说明：

```text
continuation/protocol 等机制场是当前最有价值的候选；
token-bank direction 可以产生部分因果迹象；
natural contrast direction 已经可以建立方向库；
但两者都还没有达到闭合标准。
```

当前最谨慎结论：

```text
语言编码机制不是单词元竞争；
更像多个机制场在 residual state（残差状态）中的混合竞争；
其中 continuation field（续写场）是当前最强候选；
protocol / boundary / closure 仍需拆分。
```

下一阶段建议进入 Phase251：

```text
对自然机制方向做正交化和因果干预。
```

具体任务：

```text
1. 对 natural_protocol_short、natural_continuation_explain、natural_answer_boundary、natural_target_seed 做正交化；
2. 用正交后的自然方向重跑 regime suppression / injection；
3. 比较 token-bank direction 与 natural direction 的因果稳定性；
4. 筛出进入 rollout / closure trace 的少数高置信候选；
5. 暂时不要总结闭合理论，优先继续完善图谱拼图。
```

## Phase 251: 正交化自然方向因果验证 [2026-07-07 23:56]

### 任务来源

本阶段接续 Phase249 和 Phase250。

Phase249 证明：

```text
token-bank regime direction（词元库机制方向）有部分因果迹象；
continuation_regime（续写机制）是当前最强候选；
但机制方向仍是 output embedding proxy（输出嵌入代理方向）。
```

Phase250 证明：

```text
可以从自然 prompt contrast（提示对照）中提取 natural regime direction（自然机制方向）；
但自然方向仍然是混合方向，没有完成拆分。
```

因此 Phase251 的目标是：

```text
对自然方向做正交化；
比较 token-bank direction、natural raw direction、natural orthogonal direction；
验证正交化是否能减少混合污染并提升因果稳定性。
```

### 测试脚本与结果文件

新增脚本：

```text
tests/gpt5/phase251_orthogonalized_natural_direction_causal_validation.py
tests/gpt5/run_phase251_orthogonalized_natural_direction_causal_validation.sh
```

结果目录：

```text
tests/result/phase251_orthogonalized_natural_direction_causal_validation/orthogonalized_natural_direction_causal_validation/
```

核心输出：

```text
phase251_orthogonalized_natural_direction_rows.jsonl
phase251_tokenbank_vs_natural_direction_rows.jsonl
phase251_natural_direction_causal_rows.jsonl
phase251_rollout_effect_rows.jsonl
phase251_high_confidence_rollout_candidates.jsonl
phase251_natural_direction_report.md
phase251_cross_model_summary.json
```

并已同步：

```text
tests/result/pattern_family_atlas/v1/
frontend/public/vis_data/pattern_family_atlas/v1/
```

前端 `npm run build` 已通过，仍只有既有大 chunk（大代码块）警告。

### 算法原理

Phase251 使用 Phase250 的五类自然方向：

```text
natural_protocol_short
natural_continuation_explain
natural_answer_boundary
natural_target_seed
natural_concise_answer
```

按固定顺序做 Gram-Schmidt orthogonalization（格拉姆-施密特正交化）：

$$
\boxed{
v_i^{\perp}
=
\mathrm{Normalize}
\left(
v_i
-
\sum_{j<i}
\langle v_i, v_j^{\perp} \rangle
v_j^{\perp}
\right)
}
$$

然后在同一批 Phase248/249 候选上比较三类方向：

```text
tokenbank（词元库方向）
natural_raw（自然原始方向）
natural_orth（正交化自然方向）
```

每类方向做两种干预：

```text
suppression（抑制）
injection（注入）
```

核心干预公式：

$$
\boxed{
h'
=
h
-
\lambda v
}
$$

$$
\boxed{
h'
=
h
+
\lambda v
}
$$

读出边界仍使用：

$$
\boxed{
\Delta M
=
M(h') - M(h)
}
$$

其中：

$$
\boxed{
M
=
z_{\mathrm{target}}
-
z_{\mathrm{winning\ regime}}
}
$$

同时记录 early rollout（早期生成展开）8 token，用于筛选下一阶段 rollout / closure trace（生成展开 / 闭合追踪）候选。

### 客观结果

三模型顺序完成：

```text
qwen3 -> GLM4 -> DS7B
```

跨模型统计：

```text
candidate_count: 15
validation_rows: 90
comparison_rows: 15
rollout_effect_rows: 90
orthogonalized_direction_rows: 15
high_confidence_rollout_candidates: 9
missing_rows: 0
```

总体均值：

```text
mean_tokenbank_suppression_delta: +1.464583
mean_natural_raw_suppression_delta: +1.220833
mean_natural_orth_suppression_delta: -1.8375

natural_orth_better_than_tokenbank_count: 2 / 15
natural_orth_better_than_raw_count: 0 / 15
```

最佳方向来源：

```text
natural_raw: 8
tokenbank: 7
natural_orth: 0
```

路线分布：

```text
continuation_regime_test: 9
protocol_regime_test: 5
reason_regime_test: 1
```

### 模型分解

qwen3：

```text
candidate_count: 10
mean_tokenbank_suppression_delta: +2.275
mean_natural_raw_suppression_delta: +1.059375
mean_natural_orth_suppression_delta: -2.578125
natural_orth_better_than_tokenbank_count: 0 / 10
best_suppression_sources:
  tokenbank: 6
  natural_raw: 4
```

GLM4：

```text
candidate_count: 4
mean_tokenbank_suppression_delta: -0.25
mean_natural_raw_suppression_delta: +1.734375
mean_natural_orth_suppression_delta: -0.265625
natural_orth_better_than_tokenbank_count: 2 / 4
best_suppression_sources:
  natural_raw: 3
  tokenbank: 1
```

DS7B：

```text
candidate_count: 1
mean_tokenbank_suppression_delta: +0.21875
mean_natural_raw_suppression_delta: +0.78125
mean_natural_orth_suppression_delta: -0.71875
best_suppression_sources:
  natural_raw: 1
```

### 高置信候选

筛出 9 个 high-confidence rollout candidates（高置信生成展开候选）。

最强样本：

```text
qwen3 / phase241_output_protocol_table_answer_0002 / explain_instruction
best_source: natural_raw
best_suppression_delta: +11.25
rollout_word_delta_vs_original: -1
```

其他强样本：

```text
qwen3 / reasoning_constraint_if_then_0001 / explain_instruction
best_source: natural_raw
best_suppression_delta: +7.125

qwen3 / state_drift_boundary_takeover_0001 / short_answer_instruction
best_source: natural_raw
best_suppression_delta: +6.625

qwen3 / reasoning_constraint_if_then_0000 / no_answer_anchor
best_source: natural_raw
best_suppression_delta: +6.5

GLM4 / output_protocol_explain_answer_0000 / no_answer_anchor
best_source: natural_raw
best_suppression_delta: +4.8125
```

### 正确性分析

Phase251 最重要的结果不是正交化成功，而是正交化失败。

客观现象是：

```text
natural_raw direction（自然原始方向）经常有效；
tokenbank direction（词元库方向）仍然有效；
natural_orth direction（正交化自然方向）整体明显变差。
```

这说明前一阶段的“做正交化可能减少污染”这个想法只对了一半：

```text
正交化确实减少混合；
但它也可能剥掉真实机制运行所需的混合成分。
```

因此当前不能把语言机制理解成简单线性独立基底：

```text
protocol field（协议场）
continuation field（续写场）
boundary field（边界场）
target field（目标场）
```

很可能不是彼此独立的正交方向，而是存在共享子空间、耦合方向或层级复用。

### 关键负结果

本阶段的核心负结果：

```text
简单 Gram-Schmidt 正交化不能提升机制方向因果稳定性。
```

更准确地说：

```text
natural_orth 不仅没有优于 natural_raw，
而且在 15 个候选中没有一次优于 natural_raw。
```

这说明：

```text
当前机制方向不是“越独立越好”；
真实机制可能依赖混合子空间；
简单线性拆分会破坏有效脉络。
```

### 当前机制公式修正

Phase250 的公式是：

$$
\boxed{
\Delta h
=
\sum_r
\beta_r
v_{\mathrm{natural},r}
+
\sum_k
\gamma_k
v_{\mathrm{tokenbank},k}
+
\epsilon
}
$$

Phase251 后需要补充一项：

$$
\boxed{
v_{\mathrm{effective}}
\neq
\mathrm{Orthogonalize}
\left(
v_{\mathrm{protocol}},
v_{\mathrm{continuation}},
v_{\mathrm{boundary}},
v_{\mathrm{target}}
\right)
}
$$

更合理的暂定形式是：

$$
\boxed{
\Delta h
=
\sum_r
\beta_r
v_r
+
\sum_{r,s}
\eta_{rs}
C(v_r, v_s)
+
\epsilon
}
$$

其中：

```text
C(v_r, v_s) 表示机制之间的耦合成分；
eta_rs 表示耦合强度；
epsilon 表示仍未解释的残差。
```

这不是最终理论，只是对当前负结果的最低限度解释。

### 硬伤与瓶颈

1. 当前正交化顺序固定，Gram-Schmidt 对顺序敏感。

2. 每个自然方向来自 prompt-level contrast（提示层对照），仍然混合语义、格式、长度、边界、目标压力。

3. qwen3 和 GLM4 的最佳方向来源不同：

```text
qwen3 更偏 tokenbank + natural_raw 混合；
GLM4 更偏 natural_raw。
```

不能直接上升为跨模型统一机制。

4. DS7B 只有 1 个候选，仍不能统计。

5. rollout 只记录 8 token，仍不是 closure validation（闭合验证）。

6. 当前小模型内部机制可能比大模型粗糙，有 30% 到 50% 偏差空间。

### 图谱进度

本阶段后图谱进度更新为：

```text
pattern_family_atlas: 0.79
candidate_clustering: 0.43
high_value_trace_selection: 0.64
trace_signature_validation: 0.37
focused_causal_validation: 0.24
raw_vector_factor_decomposition: 0.24
regime_field_direction_bank: 0.33
natural_regime_direction_bank: 0.28
regime_level_causal_validation: 0.23
orthogonalized_direction_validation: 0.16
residual_state_signature: 0.46
readout_competition_trace: 0.69
stepwise_rollout_trace: 0.26
causal_closure: 0.12
general_language_mechanism_confidence: 0.60
```

### 阶段结论

Phase251 完成了一个重要校准：

```text
自然方向有效；
词元库方向也有效；
简单正交化方向无效。
```

因此当前路线应从：

```text
寻找独立机制方向
```

转向：

```text
分析机制方向的共享子空间、耦合成分和层级复用。
```

下一阶段 Phase252 建议：

```text
共享子空间与耦合机制分析。
```

具体任务：

```text
1. 对 tokenbank direction 与 natural_raw direction 做 pairwise cosine（成对余弦）和 residual overlap（残差重叠）分析；
2. 不再直接正交化，而是寻找 shared effective subspace（共享有效子空间）；
3. 对 9 个高置信候选做 rollout / closure trace；
4. 检查 continuation/protocol/boundary 是否共享同一个底层展开场；
5. 继续保持闭合后置，优先补齐机制图谱。
```

## Phase 252: 共享子空间与耦合机制场分析 [2026-07-08 00:24]

### 任务来源

Phase251 给出一个关键负结果：

```text
natural_raw direction（自然原始方向）有效；
tokenbank direction（词元库方向）有效；
natural_orth direction（正交化自然方向）无效。
```

这说明简单正交化没有拆出更好的机制方向，反而可能破坏了有效混合成分。因此 Phase252 不再继续做“独立方向”假设，而是转向：

```text
shared subspace（共享子空间）
coupled regime field（耦合机制场）
rollout / closure trace（生成展开 / 闭合追踪）
```

目标不是闭合，而是解释 Phase251 的负结果：

```text
为什么自然原始方向有效；
为什么正交化方向失效；
tokenbank direction 与 natural direction 是否处于不同子空间；
高置信候选的 readout 改善是否会进入 rollout / closure。
```

### 脚本与结果文件

新增脚本：

```text
tests/gpt5/phase252_shared_subspace_coupled_regime_analysis.py
tests/gpt5/run_phase252_shared_subspace_coupled_regime_analysis.sh
```

结果目录：

```text
tests/result/phase252_shared_subspace_coupled_regime_analysis/shared_subspace_coupled_regime_analysis/
```

核心输出：

```text
phase252_direction_rows.jsonl
phase252_direction_cosine_rows.jsonl
phase252_subspace_overlap_rows.jsonl
phase252_shared_effective_subspace_rows.jsonl
phase252_shared_subspace_projection_rows.jsonl
phase252_rollout_closure_trace_rows.jsonl
phase252_coupled_regime_field_report.md
phase252_cross_model_summary.json
```

并已同步：

```text
tests/result/pattern_family_atlas/v1/
frontend/public/vis_data/pattern_family_atlas/v1/
```

前端构建通过，仍只有既有大 chunk（大代码块）警告。

### 算法原理

Phase252 做四类分析。

第一类：方向余弦矩阵。

覆盖方向：

```text
tokenbank_core:
  continuation_regime
  protocol_short_regime
  answer_boundary_regime
  period_stop_regime
  because_reason_regime

natural_raw_core:
  natural_protocol_short
  natural_continuation_explain
  natural_answer_boundary
  natural_target_seed
  natural_concise_answer

natural_orth_core:
  上述自然方向的正交化版本
```

方向余弦：

$$
\boxed{
\mathrm{cos}(v_i, v_j)
=
\frac{
\langle v_i, v_j \rangle
}{
\|v_i\|\|v_j\|
}
}
$$

第二类：子空间重叠。

对每组方向构造正交基：

$$
\boxed{
Q_S
=
\mathrm{QR}(V_S)
}
$$

子空间重叠：

$$
\boxed{
\mathrm{Overlap}(S_a,S_b)
=
\frac{
\|Q_a^T Q_b\|_F^2
}{
\min(\dim S_a,\dim S_b)
}
}
$$

第三类：共享有效子空间。

对方向集合做 SVD/PCA：

$$
\boxed{
S_{\mathrm{shared}}
=
\mathrm{SVD}
(
[v_1, v_2, \ldots, v_n]
)
}
$$

观察前几维是否能解释 raw delta vector（原始差分向量）和 Phase251 的 best suppression delta（最佳抑制边界变化）。

第四类：高置信候选 rollout / closure trace。

对 Phase251 的 high-confidence rollout candidates（高置信生成展开候选）做：

```text
8 token rollout
16 token rollout
no_intervention vs best_suppression
EOS / period / newline / continuation logit
closure_proxy_margin
```

闭合代理：

$$
\boxed{
M_{\mathrm{closure}}
=
\max(z_{\mathrm{eos}}, z_{\mathrm{period}}, z_{\mathrm{newline}})
-
z_{\mathrm{continuation}}
}
$$

### 客观结果

三模型顺序完成：

```text
qwen3 -> GLM4 -> DS7B
```

跨模型统计：

```text
direction_rows: 45
direction_cosine_rows: 315
subspace_overlap_rows: 9
shared_effective_subspace_rows: 15
shared_subspace_projection_rows: 45
rollout_closure_trace_rows: 384
observation_rows: 708
metric_rows: 32
graph_edges: 54
missing_rows: 0
```

rollout / closure trace 覆盖：

```text
qwen3: 5 个高置信候选
GLM4: 3 个高置信候选
DS7B: 0 个高置信候选
```

DS7B 没有 rollout trace，不是模型无效，而是 Phase251 没有筛出高置信候选。

### 关键观测一：自然方向内部高度重叠

最强方向余弦：

```text
natural_raw:natural_continuation_explain
vs
natural_raw:natural_protocol_short

qwen3: -1.0
GLM4: -1.0
DS7B: -1.0
```

这说明：

```text
natural_protocol_short 与 natural_continuation_explain 基本是同一轴的相反方向。
```

这非常重要。它解释了 Phase251 的正交化失败：

```text
如果两个自然方向本来就是同一控制轴的两端，
把它们强行正交化，就会破坏真实机制轴。
```

其他强重叠：

```text
qwen3:
  natural_concise_answer vs natural_continuation_explain: -0.964491
  natural_concise_answer vs natural_protocol_short: +0.964491

GLM4:
  natural_concise_answer vs natural_continuation_explain: -0.883079
  natural_concise_answer vs natural_protocol_short: +0.883079

DS7B:
  natural_concise_answer vs natural_continuation_explain: -0.717246
  natural_concise_answer vs natural_protocol_short: +0.717246
```

这说明：

```text
短答协议、解释续写、简答控制高度耦合；
它们不是三个独立方向，更像一个长度/展开控制轴上的不同方向。
```

### 关键观测二：自然方向子空间与 tokenbank 子空间重叠很低

子空间重叠：

```text
qwen3:
  natural_raw_core vs tokenbank_core: 0.008414
  natural_orth_core vs tokenbank_core: 0.009526
  natural_orth_core vs natural_raw_core: 1.0

GLM4:
  natural_raw_core vs tokenbank_core: 0.008555
  natural_orth_core vs tokenbank_core: 0.01032
  natural_orth_core vs natural_raw_core: 1.0

DS7B:
  natural_raw_core vs tokenbank_core: 0.007336
  natural_orth_core vs tokenbank_core: 0.007687
  natural_orth_core vs natural_raw_core: 0.80008
```

这说明：

```text
tokenbank direction 和 natural direction 都有效，
但它们可能不是同一个子空间。
```

更合理的解释是：

```text
tokenbank direction 更接近 output readout axis（输出读出轴）；
natural direction 更接近 internal control axis（内部控制轴）；
二者通过模型后层读出机制耦合，而不是几何上直接重合。
```

### 关键观测三：闭合代理有改善，但不能算闭合

跨模型 closure_proxy_margin（闭合代理边界）均值：

```text
no_intervention: -3.221354
natural_raw_suppression: -1.554199
tokenbank_suppression: +0.572266
```

按模型分解：

```text
qwen3:
  no_intervention: -3.491146
  natural_raw_suppression: -2.449870
  tokenbank_suppression: +0.572266

GLM4:
  no_intervention: -2.771701
  natural_raw_suppression: -0.359972
```

说明：

```text
机制抑制不仅影响第一步 readout margin；
也会影响早期 rollout 中 closure proxy；
但 closure proxy 不是真实 ModelClose（模型闭合）。
```

尤其 qwen3 的 tokenbank_suppression 把 closure proxy 推到正值，说明它可能更直接压制 continuation pressure（续写压力）或抬高 boundary/stop pressure（边界/停止压力）。

### 正确性分析

Phase252 支持 Phase251 的核心解释：

```text
自然机制方向不是独立功能方向；
它们内部高度耦合；
正交化失败不是偶然，而是因为正交化破坏了真实共享轴。
```

同时 Phase252 也补充了一个新判断：

```text
tokenbank direction 与 natural direction 可能分别处于读出轴和控制轴；
二者都是有效拼图，但不能简单相加或互相替代。
```

这使当前语言机制图谱从：

```text
单 token
单方向
单机制场
```

推进到：

```text
内部控制轴
+ 输出读出轴
+ 机制耦合轴
+ rollout 闭合代理
```

### 主要硬伤

1. 当前子空间分析方向数量仍少，每模型 15 个全局方向。

2. natural_protocol_short 与 natural_continuation_explain 是由互反 prompt contrast 构造出来的，因此余弦 -1.0 有一部分来自实验设计本身，不能过度解释为模型自然发现。

3. 子空间 overlap 很低，可能受 output embedding space（输出嵌入空间）与 hidden contrast space（隐藏对照空间）尺度差异影响。

4. rollout trace 只覆盖 qwen3 和 GLM4 的高置信候选，DS7B 没有高置信候选。

5. closure proxy 只是 EOS/句号/换行相对 continuation 的代理，不等于真实停止执行。

6. 当前仍是小模型结果，内部结构可能粗糙，有 30% 到 50% 偏差空间。

### 当前机制公式更新

Phase251 后的公式是：

$$
\boxed{
\Delta h
=
\sum_r
\beta_r
v_r
+
\sum_{r,s}
\eta_{rs}
C(v_r, v_s)
+
\epsilon
}
$$

Phase252 后需要区分两个空间：

$$
\boxed{
\Delta h
=
\Delta h_{\mathrm{control}}
+
\Delta h_{\mathrm{readout}}
+
\Delta h_{\mathrm{coupling}}
+
\epsilon
}
$$

其中：

```text
control = natural contrast direction 对应的内部控制轴；
readout = tokenbank direction 对应的输出读出轴；
coupling = 控制轴影响读出轴的耦合机制；
epsilon = 未解释残差。
```

更具体地：

$$
\boxed{
M_{\mathrm{target}}
=
R
(
h
+
\alpha v_{\mathrm{control}}
+
\gamma v_{\mathrm{readout}}
+
\eta C_{\mathrm{control,readout}}
)
}
$$

这里的 \(R\) 是 readout map（读出映射），不是简单的方向投影。

### 图谱进度

本阶段后图谱进度更新为：

```text
pattern_family_atlas: 0.80
high_value_trace_selection: 0.65
raw_vector_factor_decomposition: 0.25
regime_field_direction_bank: 0.34
natural_regime_direction_bank: 0.29
regime_level_causal_validation: 0.24
orthogonalized_direction_validation: 0.17
shared_subspace_analysis: 0.18
coupled_regime_field_analysis: 0.16
residual_state_signature: 0.47
readout_competition_trace: 0.70
stepwise_rollout_trace: 0.30
causal_closure: 0.13
general_language_mechanism_confidence: 0.61
```

### 阶段结论

Phase252 完成了三个关键拼图：

```text
1. 自然方向内部存在强耦合轴；
2. tokenbank 与 natural direction 处于低重叠子空间；
3. 高置信候选的机制抑制会改变 rollout closure proxy。
```

这说明当前最合理路线不是继续找单一闭合方向，而是：

```text
内部控制轴 -> 读出轴 -> rollout/closure 代理
```

下一阶段 Phase253 建议：

```text
控制轴到读出轴的耦合映射验证。
```

具体任务：

```text
1. 对 high-confidence candidates 记录多层 hidden trajectory（隐藏轨迹）；
2. 分析 natural_raw suppression 后，tokenbank readout axis 的投影如何随层变化；
3. 判断 control axis 是否在后层被映射到 readout axis；
4. 对 closure proxy 改善的样本做更长 rollout（32 token）；
5. 暂时仍不做闭合理论总结，继续补全机制图谱。
```

## Phase 253: 控制轴到读出轴的耦合映射验证 [2026-07-08 00:59]

### 任务来源

Phase252 的核心结论是：

```text
natural direction 更像 internal control axis（内部控制轴）；
tokenbank direction 更像 output readout axis（输出读出轴）；
二者几何子空间重叠很低，但都能影响输出。
```

因此 Phase253 不再继续寻找单一方向，而是验证：

```text
控制轴是否会在后层映射为读出轴；
控制轴 / 读出轴联合抑制是否能改善 32 token rollout 的 closure proxy（闭合代理）。
```

### 脚本与结果文件

新增脚本：

```text
tests/gpt5/phase253_control_readout_coupling_validation.py
tests/gpt5/run_phase253_control_readout_coupling_validation.sh
```

结果目录：

```text
tests/result/phase253_control_readout_coupling_validation/control_readout_coupling_validation/
```

核心输出：

```text
phase253_control_readout_projection_rows.jsonl
phase253_layerwise_coupling_rows.jsonl
phase253_suppression_projection_effect_rows.jsonl
phase253_32token_rollout_rows.jsonl
phase253_closure_validation_candidates.jsonl
phase253_control_readout_coupling_report.md
phase253_cross_model_summary.json
```

并已同步：

```text
tests/result/pattern_family_atlas/v1/
frontend/public/vis_data/pattern_family_atlas/v1/
```

前端构建通过，仍只有既有大 chunk（大代码块）警告。

### 算法原理

Phase253 做三类测试。

第一类：多层 hidden projection（隐藏状态投影）。

对高置信候选记录多个层的 hidden state：

```text
qwen3: L20, L26, L29, L31, L33
GLM4: L20, L26, L28, L30, L32
DS7B: L16, L22, L24, L26, L27
```

计算控制轴投影：

$$
\boxed{
a_l^{\mathrm{control}}
=
\langle h_l, v_{\mathrm{control}} \rangle
}
$$

计算读出轴投影：

$$
\boxed{
a_l^{\mathrm{readout}}
=
\langle h_l, v_{\mathrm{readout}} \rangle
}
$$

其中：

```text
v_control = natural_raw direction（自然原始控制方向）
v_readout = tokenbank direction（词元库读出方向）
```

第二类：控制轴抑制后的读出变化。

在不同层做：

$$
\boxed{
h_l'
=
h_l
-
\lambda v_{\mathrm{control}}
}
$$

观察：

```text
target margin delta（目标边界变化）
closure proxy delta（闭合代理变化）
readout projection delta（读出投影变化）
```

第三类：32 token rollout（32 词元生成展开）。

对 high-confidence candidates（高置信候选）做四种条件：

```text
no_intervention（无干预）
tokenbank_suppression（读出轴抑制）
natural_raw_suppression（控制轴抑制）
combined_suppression（控制轴 + 读出轴联合抑制）
```

闭合代理仍使用：

$$
\boxed{
M_{\mathrm{closure}}
=
\max(z_{\mathrm{eos}}, z_{\mathrm{period}}, z_{\mathrm{newline}})
-
z_{\mathrm{continuation}}
}
$$

并记录 continuation token rate（续写词元率）。

### 客观结果

三模型顺序完成：

```text
qwen3 -> GLM4 -> DS7B
```

跨模型统计：

```text
candidate_count: 8
control_readout_projection_rows: 240
layerwise_coupling_rows: 48
suppression_projection_effect_rows: 48
rollout_32token_rows: 1015
closure_validation_candidate_rows: 15
observation_rows: 1303
metric_rows: 22
graph_edges: 88
missing_rows: 0
```

DS7B 没有高置信候选，因此 candidate_count 为 0，不参与 rollout 判断。

### 关键结果一：32 token closure proxy 明显改善

跨模型 32 token rollout 的 closure proxy 均值：

```text
no_intervention: -3.431720
tokenbank_suppression: -0.605981
natural_raw_suppression: -1.594360
combined_suppression: +0.374603
```

续写词元率：

```text
no_intervention: 0.199219
tokenbank_suppression: 0.137652
natural_raw_suppression: 0.156250
combined_suppression: 0.132812
```

这说明：

```text
combined suppression（控制轴 + 读出轴联合抑制）
在 32 token rollout 中对 closure proxy 最强，
并且 continuation token rate 最低。
```

这支持 Phase252 的判断：

```text
控制轴和读出轴不是同一方向；
但联合使用时能更好影响生成展开和闭合代理。
```

### 关键结果二：中后层控制轴抑制更有效

跨模型 target margin delta：

```text
natural_raw_suppression_at_L20: -0.320312
natural_raw_suppression_at_L26: +6.656250
natural_raw_suppression_at_L28: +3.312500
natural_raw_suppression_at_L29: +8.062500
natural_raw_suppression_at_L30: +3.250000
natural_raw_suppression_at_L31: +7.712500
natural_raw_suppression_at_L32: +2.666667
natural_raw_suppression_at_L33: +6.675000
```

跨模型 closure proxy delta：

```text
natural_raw_suppression_at_L20: +0.621094
natural_raw_suppression_at_L26: +3.470947
natural_raw_suppression_at_L28: -0.102214
natural_raw_suppression_at_L29: +4.787500
natural_raw_suppression_at_L30: +0.700521
natural_raw_suppression_at_L31: +4.450000
natural_raw_suppression_at_L32: +0.902995
natural_raw_suppression_at_L33: +4.450000
```

模型分解：

```text
qwen3:
  L26 target margin delta: +9.4625
  L26 closure proxy delta: +6.0125
  L29 target margin delta: +8.0625
  L31 target margin delta: +7.7125
  L33 target margin delta: +6.675

GLM4:
  L20 target margin delta: -0.979167
  L20 closure proxy delta: -7.177083
  L28 target margin delta: +3.3125
  L30 closure proxy delta: +0.700521
  L32 closure proxy delta: +0.902995
```

这说明：

```text
控制轴干预不是越早越好；
中后层更像 control-to-readout coupling（控制到读出耦合）的有效区间。
```

### 关键结果三：筛出闭合验证候选

Phase253 输出：

```text
closure_validation_candidate_rows: 15
```

强候选包括：

```text
qwen3 / reasoning_constraint_if_then_0001 / explain_instruction / tokenbank_suppression
  final_closure_proxy_margin: +23.875

qwen3 / state_drift_boundary_takeover_0001 / short_answer_instruction / combined_suppression
  final_closure_proxy_margin: +10.46875
  mean_closure_proxy_margin: +1.293945
  continuation_token_rate: 0.09375

qwen3 / output_protocol_table_answer_0002 / explain_instruction / combined_suppression
  final_closure_proxy_margin: +5.046875
  mean_closure_proxy_margin: +1.936768

GLM4 / output_protocol_explain_answer_0000 / no_answer_anchor / combined_suppression
  final_closure_proxy_margin: +6.296875
  mean_closure_proxy_margin: +0.673340

GLM4 / output_protocol_explain_answer_0000 / no_answer_anchor / natural_raw_suppression
  final_closure_proxy_margin: +5.070312
  mean_closure_proxy_margin: +1.600647
  continuation_token_rate: 0.0625
```

这些候选适合下一阶段做更严格 closure validation（闭合验证）。

### 正确性分析

Phase253 的方向是正确的。

它把 Phase252 的静态几何判断：

```text
control axis 与 readout axis 子空间低重叠
```

推进到动态验证：

```text
中后层 control suppression 会改变 readout margin 和 closure proxy；
combined suppression 在 32 token rollout 中最好。
```

因此当前最谨慎结论是：

```text
语言机制可能存在 control-to-readout coupling path（控制到读出耦合路径）；
这个路径主要在中后层显现；
closure proxy 已经能被改善，但还不是 ModelClose。
```

### 主要硬伤

1. `phase253_closure_validation_candidates.jsonl` 中 `closure_proxy_delta` 和 `target_margin_delta` 字段没有成功回填，因为 32 token rollout 条件名与单步 projection effect 条件名不同。核心 rollout closure proxy 数据完整，但候选表的两个 delta 字段需要下一阶段修复。

2. 32 token rollout 仍是代理，不等于模型真实停止。

3. DS7B 没有高置信候选，不能参与 Phase253 的 rollout 判断。

4. qwen3 与 GLM4 的有效层不同，不能直接当作统一层级规律。

5. combined suppression 使用简单方向相加，尚未校准最佳权重。

6. 当前测试模型仍是小模型，内部机制可能存在 30% 到 50% 偏差。

### 当前机制公式更新

Phase252 的公式：

$$
\boxed{
M_{\mathrm{target}}
=
R
(
h
+
\alpha v_{\mathrm{control}}
+
\gamma v_{\mathrm{readout}}
+
\eta C_{\mathrm{control,readout}}
)
}
$$

Phase253 后可以加入层级映射：

$$
\boxed{
h_{l+k}^{\mathrm{readout}}
=
T_{l \rightarrow l+k}
(
h_l
+
\alpha v_{\mathrm{control}}
)
}
$$

以及闭合代理：

$$
\boxed{
M_{\mathrm{closure}}^{(t)}
=
B
(
h_t,
v_{\mathrm{control}},
v_{\mathrm{readout}}
)
}
$$

其中：

```text
T 是层间控制到读出映射；
B 是生成步上的闭合代理读出；
二者都还不是完整机制公式。
```

### 图谱进度

本阶段后图谱进度更新为：

```text
pattern_family_atlas: 0.81
high_value_trace_selection: 0.66
trace_signature_validation: 0.38
focused_causal_validation: 0.25
raw_vector_factor_decomposition: 0.25
regime_field_direction_bank: 0.35
natural_regime_direction_bank: 0.30
regime_level_causal_validation: 0.25
shared_subspace_analysis: 0.20
coupled_regime_field_analysis: 0.22
control_readout_coupling: 0.18
residual_state_signature: 0.49
readout_competition_trace: 0.71
stepwise_rollout_trace: 0.34
causal_closure: 0.14
general_language_mechanism_confidence: 0.62
```

### 阶段结论

Phase253 完成了三个新拼图：

```text
1. 控制轴干预在中后层更有效；
2. 控制轴 + 读出轴联合抑制对 32 token closure proxy 最强；
3. 已筛出 15 个 closure validation 候选。
```

当前结论仍然不是闭合：

```text
我们已经能影响 closure proxy；
但还没有证明 ModelClose 真实执行。
```

下一阶段 Phase254 建议：

```text
闭合候选的真实停止验证与候选表修复。
```

具体任务：

```text
1. 修复 closure candidate 表中的 delta 字段回填；
2. 对 15 个候选做 64 token rollout；
3. 区分 EOS、句号、换行、语义结束和客户端停止；
4. 判断 closure proxy 改善是否真的减少 over-generation；
5. 选出少数进入 ModelClose 机制验证的候选。
```

## Phase 254: 闭合候选真实停止验证与候选表修复 [2026-07-08 01:16]

### 任务来源

Phase253 已经证明：

```text
control axis（控制轴）和 readout axis（读出轴）联合抑制，
可以明显改善 32 token rollout 的 closure proxy（闭合代理）。
```

但 Phase253 仍不是闭合验证，因为：

```text
closure proxy 不等于真实停止；
final closure proxy 为正，不代表模型产生 EOS；
候选表中的 closure_proxy_delta / target_margin_delta 字段没有正确回填。
```

因此 Phase254 的目标是：

```text
修复候选表字段；
对 closure candidates 做 64 token rollout；
区分 EOS stop、句号边界、换行边界、语义完成、客户端截断、答案后继续；
判断 closure proxy 是否真的转化为更真实的停止行为。
```

### 脚本与结果文件

新增脚本：

```text
tests/gpt5/phase254_closure_candidate_stop_validation.py
tests/gpt5/run_phase254_closure_candidate_stop_validation.sh
```

结果目录：

```text
tests/result/phase254_closure_candidate_stop_validation/closure_candidate_stop_validation/
```

核心输出：

```text
phase254_closure_candidate_fixed_rows.jsonl
phase254_64token_rollout_rows.jsonl
phase254_stop_type_rows.jsonl
phase254_weighted_combined_suppression_rows.jsonl
phase254_modelclose_candidate_rows.jsonl
phase254_closure_validation_report.md
phase254_cross_model_summary.json
```

并已同步：

```text
tests/result/pattern_family_atlas/v1/
frontend/public/vis_data/pattern_family_atlas/v1/
```

前端构建通过，仍只有既有大 chunk（大代码块）警告。

### 实验设计

Phase254 使用 Phase253 的 closure validation candidates（闭合验证候选）。

每个候选做 64 token rollout：

```text
no_intervention
tokenbank_suppression
natural_raw_suppression
combined_suppression
weighted_combined_suppression
```

权重网格：

```text
lambda_c ∈ {0.25, 0.5, 1.0}
lambda_r ∈ {0.25, 0.5, 1.0}
```

联合抑制公式：

$$
\boxed{
h'
=
h
-
\lambda_c v_{\mathrm{control}}
-
\lambda_r v_{\mathrm{readout}}
}
$$

停止类型分类：

```text
eos_stop
period_boundary_stop
newline_boundary_stop
semantic_done_no_continue
continued_after_answer
client_truncation
other_stop
```

其中最关键的是区分：

```text
model stop（模型停止）
vs
client truncation（客户端截断）
```

### 客观结果

三模型顺序完成：

```text
qwen3 -> GLM4 -> DS7B
```

跨模型统计：

```text
candidate_count: 15
fixed_candidate_rows: 15
rollout_rows: 195
stop_type_rows: 195
weighted_combined_rows: 135
modelclose_candidate_rows: 1
observation_rows: 195
metric_rows: 29
graph_edges: 180
missing_rows: 0
```

停止类型：

```text
client_truncation: 186
eos_stop: 9
```

这说明：

```text
大多数 closure proxy 改善没有转化为真实停止；
只有 GLM4 出现少量 EOS stop。
```

### 模型分解

qwen3：

```text
candidate_count: 8
rollout_rows: 104
stop_type_counts:
  client_truncation: 104
modelclose_candidate_rows: 0
```

qwen3 是重要负结果：

```text
即使 closure proxy 提高，
64 token 内仍全部客户端截断，
没有真实 EOS stop。
```

GLM4：

```text
candidate_count: 7
rollout_rows: 91
stop_type_counts:
  client_truncation: 82
  eos_stop: 9
modelclose_candidate_rows: 1（去重后唯一候选）
```

DS7B：

```text
candidate_count: 0
```

DS7B 没有 Phase253 高置信闭合候选，因此本阶段不能判断 DS7B 的闭合机制。

### 权重校准结果

跨模型 final closure proxy 均值最高的条件：

```text
weighted_combined_c0.25_r1.0: +3.376302
weighted_combined_c1.0_r0.25: +3.012500
tokenbank_suppression: +1.723437
weighted_combined_c0.25_r0.5: +0.965104
weighted_combined_c0.5_r1.0: +0.965104
```

过生成长度较低的条件：

```text
weighted_combined_c0.25_r1.0: 236.2
natural_raw_suppression: 238.0
tokenbank_suppression: 245.466667
```

最重要结果：

```text
weighted_combined_c0.25_r1.0
同时提高 final closure proxy，
并明显降低 over-generation length。
```

这说明简单 1:1 combined suppression（联合抑制）不是最优；控制轴和读出轴需要权重校准。

### 唯一 ModelClose 候选

去重后唯一候选：

```text
model: GLM4
case_id: phase241_output_protocol_explain_answer_0000
variant_id: no_answer_anchor
condition: weighted_combined_c0.25_r1.0
stop_type: eos_stop
semantic_answer_seen: true
client_truncation: false
final_closure_proxy_margin: +7.15625
over_generation_length: 47
base_over_generation_length: 324
```

这是真正进入下一阶段 ModelClose validation（模型闭合验证）的候选。

但必须谨慎：

```text
它目前只是候选；
还没有证明内部 ModelClose 机制；
也没有证明跨模型通用。
```

### 候选表修复

Phase254 修复了：

```text
closure candidate 去重；
condition name mapping；
fixed candidate rows；
stop type rows；
weighted combined rows；
modelclose candidate rows。
```

但仍有一个局限：

```text
候选表中的 closure_proxy_delta / target_margin_delta 只对同名 rollout condition 稳定；
更复杂的 weighted condition 与 projection effect 的回连仍需要单独定义。
```

### 正确性分析

Phase254 是一个正负混合结果。

正结果：

```text
GLM4 中出现了 EOS stop；
weighted_combined_c0.25_r1.0 是当前最强权重；
closure proxy 可以筛出至少 1 个更接近 ModelClose 的候选。
```

负结果：

```text
qwen3 全部 client truncation；
大多数候选即使 closure proxy 提高也没有真实停止；
closure proxy 不能直接等同于 ModelClose。
```

因此当前结论必须写成：

```text
Phase254 完成了 closure proxy 到真实停止行为的第一轮校准；
只得到 1 个唯一 ModelClose 候选；
整体闭合仍未完成。
```

### 对语言机制的反思

Phase254 进一步说明：

```text
提高停止/边界压力
不等于
执行停止。
```

也就是说语言生成至少分成三层：

```text
readout preference（读出偏好）
rollout trajectory（生成轨迹）
stop execution（停止执行）
```

前面 Phase253 改善的是：

```text
readout preference + rollout closure proxy
```

Phase254 检查的是：

```text
stop execution 是否真的发生
```

结果说明：

```text
二者之间仍有缺口。
```

这对破解语言编码机制非常重要：

```text
闭合不是一个 logit 方向；
闭合是状态、读出、生成步和停止执行共同形成的过程。
```

### 当前机制公式更新

Phase253 公式：

$$
\boxed{
M_{\mathrm{closure}}^{(t)}
=
B
(
h_t,
v_{\mathrm{control}},
v_{\mathrm{readout}}
)
}
$$

Phase254 后需要区分：

$$
\boxed{
\mathrm{ClosureProxy}
\neq
\mathrm{ModelClose}
}
$$

更具体地：

$$
\boxed{
\mathrm{ModelClose}
=
E
(
S_{\mathrm{done}},
R_{\mathrm{stop}},
G_{\mathrm{rollout}},
C_{\mathrm{client}}
)
}
$$

其中：

```text
S_done = 语义完成状态；
R_stop = 停止 / 边界读出；
G_rollout = 生成展开轨迹；
C_client = 客户端截断 / 外部停止规则。
```

当前只验证到：

```text
R_stop 和 G_rollout 的部分代理；
尚未破解 S_done 到 ModelClose 的完整链条。
```

### 图谱进度

本阶段后图谱进度更新为：

```text
pattern_family_atlas: 0.82
high_value_trace_selection: 0.67
trace_signature_validation: 0.38
focused_causal_validation: 0.25
regime_field_direction_bank: 0.35
natural_regime_direction_bank: 0.30
regime_level_causal_validation: 0.26
shared_subspace_analysis: 0.20
coupled_regime_field_analysis: 0.23
control_readout_coupling: 0.20
stop_type_validation: 0.18
residual_state_signature: 0.49
readout_competition_trace: 0.72
stepwise_rollout_trace: 0.38
causal_closure: 0.16
general_language_mechanism_confidence: 0.63
```

### 阶段结论

Phase254 完成了三个关键校准：

```text
1. closure proxy 可以改善，但多数不会变成真实停止；
2. 权重校准比简单 combined suppression 更重要；
3. 当前只筛出 1 个唯一 ModelClose candidate。
```

下一阶段 Phase255 建议：

```text
唯一 ModelClose 候选的内部停止机制追踪。
```

具体任务：

```text
1. 对 GLM4 / output_protocol_explain_answer_0000 / no_answer_anchor / weighted_combined_c0.25_r1.0 做重复验证；
2. 记录 EOS step 前后的 hidden trajectory；
3. 比较 no_intervention 与 weighted condition 的 stop-logit trajectory；
4. 检查 semantic done 是否先于 EOS stop；
5. 判断这是偶然采样 / 读出偏置，还是内部停止机制迹象。
```

## Phase 255: 唯一闭合候选的内部停止轨迹追踪 [2026-07-08 01:30]

### 任务来源

本阶段分析 Phase254 的复盘内容是否正确，并继续同一阶段任务。Phase254 的核心判断是正确的：

```text
closure proxy（闭合代理）改善
≠
ModelClose（模型真实闭合）
```

Phase254 的 186 个 client_truncation（客户端截断）和 9 个 eos_stop（结束符停止）说明，读出边界改善多数不能自动转化为停止执行。因此 Phase255 不再扩大普通候选，而是对 Phase254 产生的唯一 GLM4 ModelClose candidate（模型闭合候选）做内部轨迹追踪。

测试脚本：

```text
tests/gpt5/phase255_modelclose_internal_stop_trace.py
tests/gpt5/run_phase255_modelclose_internal_stop_trace.sh
```

测试结果：

```text
tests/result/phase255_modelclose_internal_stop_trace/modelclose_internal_stop_trace/
```

前端固定格式图谱数据已同步，并通过：

```text
npm run sync:pattern-atlas
npm run build
```

构建通过，仅保留既有大 chunk warning（大包警告）。

### 测试原理

Phase255 针对唯一候选：

```text
model: GLM4
case_id: phase241_output_protocol_explain_answer_0000
variant_id: no_answer_anchor
source condition: weighted_combined_c0.25_r1.0
```

比较 5 个条件：

```text
no_intervention（无干预）
tokenbank_suppression（词元库读出轴抑制）
natural_raw_suppression（自然控制轴抑制）
combined_suppression（控制轴+读出轴 1:1 抑制）
weighted_combined_c0.25_r1.0（控制轴0.25+读出轴1.0加权抑制）
```

干预公式：

$$
\boxed{
h'_t
=
h_t
-
\lambda_c v_{\mathrm{control}}
-
\lambda_r v_{\mathrm{readout}}
}
$$

其中：

```text
v_control（控制轴）来自 natural_continuation_explain（自然续写/解释对比方向）；
v_readout（读出轴）来自 continuation_regime（续写词元库方向）；
lambda_c / lambda_r 是控制轴和读出轴权重。
```

本阶段同时记录三类数据：

```text
1. actual stop trace（真实停止轨迹）：是否 EOS、是否客户端截断、答案出现位置、过生成长度；
2. generation step trace（生成步轨迹）：每一步 closure proxy、EOS logit、读出指标；
3. prefix hidden projection（前缀隐藏投影）：每个生成前缀在多层 residual hidden state 上对 control/readout 方向的投影。
```

机制判定公式从 Phase254 继续收紧为：

$$
\boxed{
\mathrm{ModelClose}
=
E(
S_{\mathrm{done}},
R_{\mathrm{stop}},
G_{\mathrm{rollout}},
C_{\mathrm{client}}
)
}
$$

Phase255 主要检查：

$$
\boxed{
R_{\mathrm{stop}}
\ \mathrm{and}\
G_{\mathrm{rollout}}
\rightarrow
\mathrm{EOS\ execution}
}
$$

是否存在可观察轨迹，而不是只看最终 margin（边界）。

### 客观结果

跨模型摘要：

```text
qwen3: 无 Phase254 ModelClose 候选，生成空结果，未加载模型；
GLM4: 1 个候选，完成内部停止轨迹追踪；
DS7B: 无 Phase254 ModelClose 候选，生成空结果，未加载模型。
```

GLM4 结果：

```text
stop_trace_rows: 5
generation_step_rows: 245
prefix_projection_rows: 1225
stop_type_counts:
  eos_stop: 3
  client_truncation: 2
missing_rows: 0
```

各条件结果：

```text
no_intervention:
  stop_type = client_truncation
  tokens = 96
  answer_first_step = 7
  eos_pos = None
  final_closure_proxy_margin = +1.3125
  over_generation_length = 507

tokenbank_suppression:
  stop_type = client_truncation
  tokens = 96
  answer_first_step = None
  eos_pos = None
  final_closure_proxy_margin = -1.0625
  over_generation_length = 499

natural_raw_suppression:
  stop_type = eos_stop
  tokens = 12
  answer_first_step = 1
  eos_pos = 12
  final_closure_proxy_margin = +2.90625
  over_generation_length = 56

combined_suppression:
  stop_type = eos_stop
  tokens = 12
  answer_first_step = 1
  eos_pos = 12
  final_closure_proxy_margin = +7.382812
  over_generation_length = 56

weighted_combined_c0.25_r1.0:
  stop_type = eos_stop
  tokens = 29
  answer_first_step = 3
  eos_pos = 29
  final_closure_proxy_margin = +11.03125
  over_generation_length = 131
```

最终层前缀投影均值：

```text
no_intervention:
  readout_projection_mean = -4.449800
  control_projection_mean = -11.366206
  closure_proxy_mean = -1.157878

tokenbank_suppression:
  readout_projection_mean = -6.094681
  control_projection_mean = -14.623580
  closure_proxy_mean = -0.183431

natural_raw_suppression:
  readout_projection_mean = -3.546949
  control_projection_mean = -9.140234
  closure_proxy_mean = -0.814779

combined_suppression:
  readout_projection_mean = -3.546949
  control_projection_mean = -9.140234
  closure_proxy_mean = +1.399089

weighted_combined_c0.25_r1.0:
  readout_projection_mean = -5.060608
  control_projection_mean = -10.590529
  closure_proxy_mean = -0.123072
```

EOS 前关键轨迹：

```text
natural_raw_suppression:
  step 11 输出句号，closure_proxy = +4.75，eos_logit = +6.25
  step 12 输出 EOS，closure_proxy = +3.50，eos_logit = +7.5625

combined_suppression:
  step 11 输出句号，closure_proxy = +10.234375，eos_logit = +7.21875
  step 12 输出 EOS，closure_proxy = +6.40625，eos_logit = +8.125

weighted_combined_c0.25_r1.0:
  step 28 输出句号，closure_proxy = +17.4375，eos_logit = +4.09375
  step 29 输出 EOS，closure_proxy = +10.730469，eos_logit = +9.9375
```

### 正确性分析

Phase254 复盘内容总体正确，而且 Phase255 进一步支持其中最关键的判断：

```text
真实停止执行不是单纯读出轴问题。
```

原因是：

```text
tokenbank_suppression 是纯 readout axis（读出轴）抑制；
它没有触发 EOS；
甚至没有稳定产生目标答案。
```

而：

```text
natural_raw_suppression 只使用 control axis（控制轴）；
却直接触发 EOS stop；
combined 和 weighted combined 也触发 EOS stop。
```

这说明当前唯一闭合候选中，停止执行更依赖 control axis（控制轴）参与，而不是只靠 readout preference（读出偏好）增强。

Phase255 的正结果：

```text
1. 唯一 GLM4 ModelClose 候选可复现 EOS stop；
2. EOS 不只出现在 Phase254 的 weighted condition，也出现在 natural_raw 和 combined；
3. EOS 前通常先出现句号/语义完成，再出现 closure proxy 和 EOS logit 上升；
4. 纯读出轴不能完成真实停止，反而保持 client truncation。
```

Phase255 的负结果：

```text
1. qwen3 和 DS7B 没有 Phase254 ModelClose 候选，本阶段无法跨模型验证；
2. GLM4 只有 1 个样本，不能证明通用机制；
3. natural_raw 和 combined 的输出更短，但语义格式不如 weighted condition 完整；
4. weighted condition 虽然 EOS 最强，但过生成长度不一定最短；
5. 当前仍未证明 S_done（语义完成状态）在内部作为独立状态被稳定编码。
```

### 关键进展

此前 Phase253/254 的核心结构是：

```text
control axis（控制轴）
readout axis（读出轴）
closure proxy（闭合代理）
actual stop（真实停止）
```

Phase255 后可以进一步拆成：

```text
readout axis: 改变边界压力，但不能单独执行停止；
control axis: 更可能改变生成状态，使模型进入可停止轨迹；
combined axis: 同时增强边界和轨迹；
weighted combined: 可以形成更强 EOS logit，但格式和过生成仍需校准。
```

当前机制图谱应更新为：

$$
\boxed{
S_{\mathrm{done}}
\xrightarrow{\ v_{\mathrm{control}}\ }
G_{\mathrm{rollout}}
\xrightarrow{\ v_{\mathrm{readout}}\ }
R_{\mathrm{stop}}
\xrightarrow{}
\mathrm{EOS}
}
$$

但必须注意，这只是当前小模型 GLM4 的单样本机制迹象，不是已闭合公式。

### 图谱进度

本阶段后固定格式图谱数据已更新：

```text
pattern_family_atlas: 0.82
high_value_trace_selection: 0.68
trace_signature_validation: 0.40
focused_causal_validation: 0.25
regime_field_direction_bank: 0.35
natural_regime_direction_bank: 0.30
regime_level_causal_validation: 0.26
shared_subspace_analysis: 0.20
coupled_regime_field_analysis: 0.23
control_readout_coupling: 0.21
stop_type_validation: 0.20
residual_state_signature: 0.50
readout_competition_trace: 0.73
stepwise_rollout_trace: 0.40
causal_closure: 0.17
general_language_mechanism_confidence: 0.63
```

整体语言模式图谱进度评估：

```text
行为图谱: 约 82%
读出竞争图谱: 约 73%
残差状态签名: 约 50%
停止类型验证: 约 20%
因果闭合: 约 17%
总体语言机制信心: 约 63%
```

### 问题、硬伤和瓶颈

1. 单样本硬伤仍然存在。GLM4 的这个样本非常有价值，但不能证明跨样本、跨任务、跨模型机制。

2. 小模型偏差必须保留。当前测试模型内部结构可能比大模型更粗糙，控制轴/读出轴分离程度、EOS 执行机制、短答协议都可能和真实大模型有 30% 到 50% 偏差。

3. 语义完成状态仍未被直接定位。现在观察到 answer_step 和 EOS step 的关系，但还没有找到稳定的 hidden state done signature（完成状态签名）。

4. 过生成长度指标需要更精细。当前 over_generation_length（过生成长度）按文本长度近似，适合粗筛，但对短答案和解释答案的评价不够精准。

5. 读出轴不是停止轴。tokenbank_suppression 的失败说明，继续沿“只修读出边界”路线会进入边际收益递减。

### 结论

Phase255 是 Phase254 后的实质推进，但仍不是闭合成功。

本阶段最重要的客观结论是：

```text
在唯一 GLM4 ModelClose 候选中，
真实 EOS stop 更依赖 control axis（控制轴）参与，
纯 readout axis（读出轴）不能触发停止执行。
```

因此当前研究路线应从：

```text
寻找更强 closure proxy
```

转为：

```text
寻找语义完成状态 S_done
以及 S_done 如何通过 control axis 进入 rollout trajectory，
再通过 readout axis 执行 EOS stop。
```

### 下一阶段任务

Phase256 仍属于当前阶段，应继续自动推进。建议任务：

```text
Phase256: 语义完成状态 done signature 的反事实定位
```

核心方案：

```text
1. 以 Phase255 的 GLM4 EOS 候选为种子；
2. 构造 answer-before-done、answer-after-done、explain-continue、short-answer-stop 四类前缀；
3. 对 EOS 前后 3 到 5 个 token 做 residual hidden state 差分；
4. 检查 done signature 是否早于 EOS logit 上升；
5. 如果存在 done signature，再进行小规模因果注入/抑制；
6. 输出固定 Pattern Atlas 格式，继续更新语言模式图谱。
```

阶段性目标不是立刻闭合，而是先确认：

```text
模型内部是否存在独立的 S_done（语义完成状态）；
它是否是 ModelClose 的上游原因。
```

## Phase 256: 语义完成状态签名的反事实定位 [2026-07-08 01:35]

### 任务来源

Phase255 发现唯一 GLM4 ModelClose candidate（模型闭合候选）中，真实 EOS stop（结束符停止）并不是纯 readout axis（读出轴）触发，而更依赖 control axis（控制轴）参与。因此 Phase256 继续同一阶段任务：不急于闭合，而是检查是否存在更上游的语义完成状态：

```text
S_done（语义完成状态）
```

测试脚本：

```text
tests/gpt5/phase256_done_signature_counterfactual_localization.py
tests/gpt5/run_phase256_done_signature_counterfactual_localization.sh
```

测试结果：

```text
tests/result/phase256_done_signature_counterfactual_localization/done_signature_counterfactual_localization/
```

图谱数据已同步到前端固定格式，并通过：

```text
npm run sync:pattern-atlas
npm run build
```

构建通过，仅有既有大 chunk warning（大包警告）。

### 测试原理

本阶段不重新采样，而是使用 Phase255 已生成的 step trace（生成步轨迹）重建前缀，减少变量。

对每个 EOS 条件，取最终层 residual hidden state（残差隐藏状态）：

```text
h_answer = 答案首次出现前缀的 hidden state；
h_eos = EOS 输出前缀的 hidden state。
```

构造候选 done direction（完成方向）：

$$
\boxed{
v_{\mathrm{done}}
=
\operatorname{unit}
\left(
\frac{1}{N}
\sum_i
\left(
h_{\mathrm{eos}}^{(i)}
-
h_{\mathrm{answer}}^{(i)}
\right)
\right)
}
$$

然后对每个条件、每个生成前缀、多个层位计算：

$$
\boxed{
p_{\mathrm{done}}(t,l)
=
\left\langle
h_{t,l},
v_{\mathrm{done}}
\right\rangle
}
$$

这个设计只做基础差分和投影，不引入复杂统计模型。目标是观察：

```text
answer step（答案步）到 EOS step（结束符步）
是否存在稳定 done_projection（完成投影）增长。
```

### 客观结果

跨模型情况：

```text
qwen3: 无 Phase255 stop seed，生成空结果；
GLM4: 完成 1 个候选的 done signature 定位；
DS7B: 无 Phase255 stop seed，生成空结果。
```

GLM4 结果：

```text
seed_stop_rows: 5
done_vector_component_rows: 3
done_signature_rows: 1225
counterfactual_rows: 5
missing_rows: 0
interpretation_counts:
  eos_aligned_done_growth: 3
  no_eos_or_no_growth: 2
```

三条 EOS 条件的 done gain（完成投影增益）：

```text
natural_raw_suppression:
  answer_done_projection = -37.017658
  pre_eos_done_projection = +3.829839
  eos_done_projection = +21.242704
  done_gain_answer_to_eos = +58.260362

combined_suppression:
  answer_done_projection = -37.017658
  pre_eos_done_projection = +3.829839
  eos_done_projection = +21.242704
  done_gain_answer_to_eos = +58.260362

weighted_combined_c0.25_r1.0:
  answer_done_projection = -50.142101
  pre_eos_done_projection = +7.158975
  eos_done_projection = +21.224991
  done_gain_answer_to_eos = +71.367092
```

两个非 EOS 条件：

```text
no_intervention:
  stop_type = client_truncation
  answer_done_projection = -42.577522
  late_done_projection = -4.214076
  done_gain_answer_to_eos = None

tokenbank_suppression:
  stop_type = client_truncation
  answer_done_projection = 0.0
  late_done_projection = -5.510945
  done_gain_answer_to_eos = None
```

关键步轨迹：

```text
natural_raw_suppression:
  answer step 1: done_projection = -37.017658
  pre-EOS step 11: done_projection = +3.829839
  EOS step 12: done_projection = +21.242704

combined_suppression:
  answer step 1: done_projection = -37.017658
  pre-EOS step 11: done_projection = +3.829839
  EOS step 12: done_projection = +21.242704

weighted_combined_c0.25_r1.0:
  answer step 3: done_projection = -50.142101
  pre-EOS step 28: done_projection = +7.158975
  EOS step 29: done_projection = +21.224991
```

### 正确性分析

Phase256 得到的是一个局部正结果：

```text
在 GLM4 唯一闭合候选中，
EOS 条件都存在 answer step 到 EOS step 的 done_projection 大幅上升。
```

这说明 Phase255 的判断进一步得到支持：

```text
停止执行不是单纯的 EOS logit 或 readout preference；
它可能有一个更上游的 residual state transition（残差状态转换）。
```

但这个结果不能过度解释。因为：

```text
1. done direction 是从同一个候选样本构造的，存在自解释风险；
2. natural_raw 和 combined 的前缀轨迹高度相同，不能当成独立样本；
3. no_intervention 后期也出现 late_done_projection 上升，但仍未 EOS；
4. 因此 done_projection 可能是必要迹象，但不是充分条件。
```

更谨慎的结论是：

```text
Phase256 找到了一个局部 done signature candidate（完成状态签名候选），
它与 EOS 执行同向，
但还没有证明它是通用因果机制。
```

### 机制进展

Phase253 到 Phase256 的机制链条可以暂时写成：

$$
\boxed{
S_{\mathrm{answer}}
\rightarrow
S_{\mathrm{done}}
\rightarrow
G_{\mathrm{rollout}}
\rightarrow
R_{\mathrm{stop}}
\rightarrow
\mathrm{EOS}
}
$$

其中当前证据强度：

```text
S_answer（答案出现）: 已可行为定位；
S_done（语义完成）: 发现局部候选签名；
G_rollout（生成轨迹）: 已有 step trace 证据；
R_stop（停止读出）: closure proxy 和 EOS logit 可观测；
EOS（结束符执行）: GLM4 单样本可复现。
```

但是尚未证明：

```text
S_done 是否可跨样本复用；
S_done 是否可因果注入；
S_done 是否早于并驱动 R_stop；
S_done 是否在 qwen3 / DS7B 中存在同构结构。
```

### 图谱进度

Phase256 后图谱进度：

```text
pattern_family_atlas: 0.82
high_value_trace_selection: 0.68
trace_signature_validation: 0.42
focused_causal_validation: 0.25
regime_field_direction_bank: 0.35
natural_regime_direction_bank: 0.30
regime_level_causal_validation: 0.26
shared_subspace_analysis: 0.20
coupled_regime_field_analysis: 0.23
control_readout_coupling: 0.21
stop_type_validation: 0.20
semantic_done_signature: 0.12
residual_state_signature: 0.51
readout_competition_trace: 0.73
stepwise_rollout_trace: 0.41
causal_closure: 0.17
general_language_mechanism_confidence: 0.63
```

整体判断：

```text
语言模式图谱主体仍在推进；
闭合验证仍处于早期；
semantic_done_signature 是新开拼图，当前只有 0.12；
causal_closure 仍只有 0.17，不能宣布闭合。
```

### 问题、硬伤和瓶颈

1. 单样本构造 done direction 有循环解释风险。当前 v_done 来自 EOS 条件本身，因此只能算局部签名，不能算通用公式。

2. 非 EOS 条件也出现 late_done_projection 上升，说明 done_projection 单独不足以触发停止。

3. tokenbank_suppression 没有目标答案，导致 answer_done_projection 无法有效比较，这暴露出纯读出轴干预破坏语义轨迹的问题。

4. qwen3 和 DS7B 没有同类 ModelClose seed，跨模型验证仍缺失。

5. 当前小模型可能存在停止机制粗糙、EOS 训练偏置强、短答协议不稳定等问题，不能直接外推到大模型语言机制。

### 阶段结论

Phase256 是一个必要的局部机制定位阶段。它没有闭合语言机制，但把问题从：

```text
怎样提高 EOS / closure proxy
```

推进到：

```text
是否存在 EOS 上游的 S_done residual signature
```

本阶段最重要的发现是：

```text
GLM4 的三个 EOS 条件都出现 answer → EOS 的 done_projection 大幅增长；
这支持“停止执行需要语义完成状态参与”的路线。
```

但必须保持谨慎：

```text
done signature 目前是局部候选，不是通用编码机制；
下一步必须做跨样本反事实验证。
```

### 下一阶段任务

Phase257 仍属于当前阶段，应继续推进：

```text
Phase257: done signature 的跨样本反事实复用测试
```

建议方案：

```text
1. 用 Phase256 的 v_done 作为固定方向，不重新拟合；
2. 在 GLM4 的多个 output_protocol / short_answer / explain_answer 样本上投影；
3. 检查 answer_done、pre_period_done、post_period_done、eos_done 的相对顺序；
4. 对少量高匹配样本做 v_done 注入/抑制；
5. 判断 v_done 是单样本局部方向，还是可复用的完成状态方向。
```

如果 Phase257 失败，也很有价值，因为它会说明：

```text
done state 不是单一方向，
而可能是任务族、输出协议、语义类型共同决定的局部状态簇。
```

## Phase 257: done signature 的跨样本复用测试 [2026-07-08 01:40]

### 任务来源

Phase256 找到了局部 done signature candidate（完成状态签名候选），但它来自唯一 GLM4 ModelClose candidate（模型闭合候选），存在自解释风险。因此 Phase257 继续同一阶段任务：固定 Phase256 的 v_done（完成方向），不重新拟合，在更多 GLM4 输出协议样本上做跨样本复用测试。

测试脚本：

```text
tests/gpt5/phase257_done_signature_cross_sample_reuse.py
tests/gpt5/run_phase257_done_signature_cross_sample_reuse.sh
```

测试结果：

```text
tests/result/phase257_done_signature_cross_sample_reuse/done_signature_cross_sample_reuse/
```

图谱数据已同步并通过前端构建：

```text
npm run sync:pattern-atlas
npm run build
```

构建通过，仅保留既有大 chunk warning（大包警告）。

### 测试原理

Phase257 的关键约束是：

```text
v_done 固定；
复用样本不重新拟合 done direction。
```

对 40 个 GLM4 output_protocol（输出协议）样本，构造 5 类前缀：

```text
prompt_only（只有提示词）
answer_only（提示词 + 答案）
answer_period（提示词 + 答案 + 句号）
answer_explain_stub（提示词 + 答案 + because）
answer_done_template（提示词 + Answer/Reason 完成模板）
```

计算最终层：

$$
\boxed{
p_{\mathrm{done}}(x)
=
\left\langle
h_{\mathrm{last}}(x),
v_{\mathrm{done}}
\right\rangle
}
$$

并检查：

$$
\boxed{
p_{\mathrm{done}}(\mathrm{answer\_period})
>
p_{\mathrm{done}}(\mathrm{answer\_only})
}
$$

以及：

$$
\boxed{
p_{\mathrm{done}}(\mathrm{answer\_done\_template})
>
p_{\mathrm{done}}(\mathrm{prompt\_only})
}
$$

如果两者同时成立，标记为 reuse_match（复用匹配）。

### 客观结果

跨模型：

```text
qwen3: 无 done seed，生成空结果；
GLM4: 完成 40 个样本复用测试；
DS7B: 无 done seed，生成空结果。
```

GLM4 结果：

```text
seed_eos_rows: 3
done_vector_component_rows: 3
reuse_rows: 1000
case_summary_rows: 40
missing_rows: 0
reuse_match_count: 18
reuse_match_rate: 0.45
```

不同前缀的平均 done_projection：

```text
prompt_only: -17.489933
answer_only: -7.151982
answer_period: -5.547511
answer_explain_stub: -10.986023
answer_done_template: -2.426071
```

差分结果：

```text
mean_period_minus_answer: +1.604471
mean_done_template_minus_prompt: +15.063862
```

按样本明细观察：

```text
output_protocol 总样本: 40
reuse_match: 18
reuse_match_rate: 45%
```

解释型 answer 样本上 done_template 增益较强：

```text
phase241_output_protocol_explain_answer_0000 one_word_strict:
  done_template_minus_prompt = +45.466290

phase241_output_protocol_explain_answer_0000 short_answer_instruction:
  done_template_minus_prompt = +41.770173

phase241_output_protocol_explain_answer_0000 full:
  done_template_minus_prompt = +37.035934
```

JSON answer 相关样本上增益弱或为负：

```text
phase241_output_protocol_json_answer_0001 no_answer_anchor:
  done_template_minus_prompt = -2.977939

phase241_output_protocol_json_answer_0000 no_answer_anchor:
  done_template_minus_prompt = -2.364160

phase241_output_protocol_json_answer_0000 target_seeded:
  done_template_minus_prompt = -1.713798
```

### 正确性分析

Phase257 是一个弱正结果，同时也是重要校准。

正结果：

```text
固定 v_done 在 40 个样本上不是完全失效；
answer_done_template 相对 prompt_only 平均提升明显；
answer_period 相对 answer_only 平均也有小幅提升；
说明 Phase256 的 done direction 至少捕捉到一部分“答案完成/格式完成”的状态变化。
```

负结果：

```text
reuse_match_rate 只有 45%；
JSON answer 样本上增益弱或负；
说明 v_done 不是全局通用完成方向。
```

因此不能说：

```text
已经找到语言完成状态的统一方向。
```

更准确的说法是：

```text
找到一个与 explain_answer 输出协议强相关的 done-like direction（类完成方向）；
它可以弱复用，但不是全局闭合机制。
```

### 机制进展

Phase257 把 Phase256 的局部 done signature 从单样本扩展到小批量样本，得到一个更谨慎的结构：

$$
\boxed{
S_{\mathrm{done}}
\not\approx
v_{\mathrm{done}}^{\mathrm{global}}
}
$$

更可能的形式是：

$$
\boxed{
S_{\mathrm{done}}
\approx
\mathcal{C}
\left(
v_{\mathrm{semantic}},
v_{\mathrm{protocol}},
v_{\mathrm{boundary}},
v_{\mathrm{rollout}}
\right)
}
$$

其中：

```text
v_semantic（语义完成方向）
v_protocol（输出协议方向）
v_boundary（边界/句号方向）
v_rollout（生成轨迹方向）
```

共同形成局部 done state cluster（完成状态簇）。

这与前面图谱路线一致：

```text
语言不是单一向量机制；
语言更像模式网络；
完成状态也不是一个方向，而是模式族中的局部状态簇。
```

### 图谱进度

Phase257 后图谱进度：

```text
pattern_family_atlas: 0.82
high_value_trace_selection: 0.68
trace_signature_validation: 0.43
focused_causal_validation: 0.25
regime_field_direction_bank: 0.35
natural_regime_direction_bank: 0.30
regime_level_causal_validation: 0.26
shared_subspace_analysis: 0.20
coupled_regime_field_analysis: 0.23
control_readout_coupling: 0.21
stop_type_validation: 0.20
semantic_done_signature: 0.15
residual_state_signature: 0.52
readout_competition_trace: 0.73
stepwise_rollout_trace: 0.41
causal_closure: 0.17
general_language_mechanism_confidence: 0.63
```

### 问题、硬伤和瓶颈

1. 复用率只有 45%，不能支持全局单向量 done mechanism（完成机制）。

2. 样本主要来自 GLM4 output_protocol，仍不能跨模型泛化。

3. 前缀模板是人工构造，虽然有利于控制变量，但不等同于模型自然生成轨迹。

4. JSON answer 和 explain answer 差异明显，说明输出协议本身强烈影响 done signature。

5. 当前只是投影复用，不是因果注入/抑制验证，因此 causal_closure（因果闭合）没有明显提高。

### 阶段结论

Phase257 完成了 Phase256 之后必须做的跨样本校准。结论是：

```text
Phase256 的 v_done 不是纯偶然方向，
但也不是全局通用完成方向。
```

更可能的真实结构是：

```text
完成状态 = 语义完成 + 输出协议完成 + 边界符号 + 生成轨迹
共同形成的局部状态簇。
```

因此当前研究应避免继续寻找一个单一 done vector（完成向量），而应进入：

```text
done state cluster（完成状态簇）
```

的图谱分析。

### 下一阶段任务

Phase258 可以继续同一大阶段，但已经从“单方向验证”转入“状态簇图谱”。建议任务：

```text
Phase258: done state cluster 的模式族分解
```

具体方案：

```text
1. 按 explain_answer / short_answer / json_answer / one_word_answer 分组；
2. 每组单独构造局部 done direction；
3. 比较组内复用率和组间迁移率；
4. 判断 done state 是协议族分裂，还是共享核心 + 协议外壳；
5. 只在复用率高的组内做小规模因果注入。
```

阶段性目标：

```text
从“寻找单一完成方向”
转为
绘制“完成状态簇图谱”。
```

## Phase 258: done state cluster 的模式族分解 [2026-07-08 02:24]

### 任务来源

本阶段分析 Phase255-257 的复盘内容是否正确，并继续当前同一阶段任务。附件判断总体正确：Phase255 证明 GLM4 单样本中真实 EOS stop（结束符停止）与 control axis（控制轴）参与有关；Phase256 找到局部 done-like direction（类完成方向）；Phase257 证明该方向可以弱复用，但不是全局 done vector（完成向量）。

因此 Phase258 从：

```text
寻找单一完成方向
```

转入：

```text
done state cluster（完成状态簇）图谱分解
```

测试脚本：

```text
tests/gpt5/phase258_done_state_cluster_mode_decomposition.py
tests/gpt5/run_phase258_done_state_cluster_mode_decomposition.sh
```

测试结果：

```text
tests/result/phase258_done_state_cluster_mode_decomposition/done_state_cluster_mode_decomposition/
```

固定格式图谱数据已同步到前端，并通过：

```text
npm run sync:pattern-atlas
npm run build
```

构建通过，仅保留既有大 chunk warning（大包警告）。

### 测试原理

Phase258 不再只使用 Phase256 的单一样本 v_done，而是在每个模型中按 output_protocol（输出协议）模式族分组：

```text
short_answer
one_word
explain_answer
repeat_answer
list_answer
json_answer
table_answer
stop_after_answer
```

每个模型、每个模式族最多取 8 个样本，共：

```text
每模型 64 个样本；
三模型 192 个样本。
```

对每个样本构造 5 类前缀：

```text
prompt_only（只有提示）
answer_only（提示 + 答案）
answer_period（提示 + 答案 + 句号）
answer_explain_stub（提示 + 答案 + because）
answer_done_template（提示 + Answer/Reason 完成模板）
```

每个模式族构造局部完成方向：

$$
\boxed{
v_{\mathrm{done}}^{(m)}
=
\operatorname{unit}
\left(
\frac{1}{N_m}
\sum_i
\left(
h_{\mathrm{done\_template}}^{(i,m)}
-
h_{\mathrm{prompt}}^{(i,m)}
\right)
\right)
}
$$

然后计算所有源模式方向到所有目标模式样本的投影迁移：

$$
\boxed{
p_{\mathrm{done}}^{(s \rightarrow t)}(x)
=
\left\langle
h(x),
v_{\mathrm{done}}^{(s)}
\right\rangle
}
$$

并用两个基础条件判断 reuse_match（复用匹配）：

$$
\boxed{
p(\mathrm{answer\_period})
>
p(\mathrm{answer\_only})
}
$$

$$
\boxed{
p(\mathrm{answer\_done\_template})
>
p(\mathrm{prompt\_only})
}
$$

注意：这个测试仍然是投影图谱，不是因果闭合。

### 客观结果

三模型均完成：

```text
done_cluster_vectors: 24
projection_rows: 7680
transfer_rows: 1536
observation_rows: 9216
metric_rows: 282
graph_edges: 192
missing_rows: 0
```

跨模型复用率：

```text
qwen3:
  within_mode_reuse_rate = 0.953125
  cross_mode_reuse_rate = 0.984375

GLM4:
  within_mode_reuse_rate = 0.843750
  cross_mode_reuse_rate = 0.944196

DS7B:
  within_mode_reuse_rate = 0.937500
  cross_mode_reuse_rate = 0.937500

overall:
  within_mode_reuse_rate = 0.911458
  cross_mode_reuse_rate = 0.955357
```

按模型总体复用率：

```text
qwen3: 0.980469
GLM4: 0.931641
DS7B: 0.937500
```

按源模式复用率：

```text
short_answer: 0.937500
one_word: 0.963542
explain_answer: 0.942708
repeat_answer: 0.984375
list_answer: 0.958333
json_answer: 0.911458
table_answer: 0.947917
stop_after_answer: 0.953125
```

按目标模式复用率：

```text
explain_answer: 0.979167
json_answer: 0.822917
list_answer: 0.937500
one_word: 0.947917
repeat_answer: 0.984375
short_answer: 0.968750
stop_after_answer: 0.989583
table_answer: 0.968750
```

方向相似度：

```text
qwen3 mode-direction cosine:
  mean = 0.629099
  min = 0.518082
  max = 0.885819

GLM4 mode-direction cosine:
  mean = 0.509306
  min = 0.302626
  max = 0.741393

DS7B mode-direction cosine:
  mean = 0.793704
  min = 0.656966
  max = 0.926663
```

明显薄弱迁移：

```text
GLM4:
  json_answer -> json_answer = 0.125
  json_answer -> table_answer = 0.500
  table_answer -> json_answer = 0.500
  one_word -> json_answer = 0.625
  list_answer -> json_answer = 0.625

qwen3:
  list_answer -> list_answer = 0.625
  json_answer -> list_answer = 0.625
  stop_after_answer -> list_answer = 0.625

DS7B:
  short_answer -> json_answer = 0.750
  short_answer -> short_answer = 0.750
  one_word -> json_answer = 0.750
  explain_answer -> json_answer = 0.750
```

### 正确性分析

Phase258 是一个重要校准阶段。表面上复用率很高，尤其 cross-mode（跨模式）复用率甚至高于 within-mode（组内复用率），但这不能解释为“找到了统一完成方向”。

更谨慎的解释是：

```text
当前 done direction 很大一部分捕捉到的是 answer_done_template（完成模板）相对 prompt_only（纯提示）的格式/协议轴；
该轴跨模式共享较强；
因此 cross-mode 高不等于真实语义完成机制闭合。
```

这也是为什么 GLM4 最有价值：它的模式方向相似度均值最低：

```text
GLM4 cosine mean = 0.509306
```

说明 GLM4 中 output_protocol 的不同模式族确实有一定分裂，而 qwen3 和 DS7B 的方向更相似，可能反映小模型结构更粗，或者模板轴更强。

JSON answer 是最关键异类：

```text
overall target json_answer reuse_rate = 0.822917
GLM4 json_answer -> json_answer = 0.125
```

这说明 json_answer 的完成状态很可能不是普通“答案+句号+解释模板”方向，而是更接近结构化协议闭合：

```text
括号/引号/冒号/字段完整性/结构结束
```

因此 Phase258 没有证明全局 done cluster 已完成；它证明：

```text
完成状态簇至少包含强共享模板轴 + 协议族分裂轴。
```

### 机制进展

Phase257 的公式是：

$$
\boxed{
S_{\mathrm{done}}
\not\approx
v_{\mathrm{done}}^{\mathrm{global}}
}
$$

Phase258 后更精确地改为：

$$
\boxed{
S_{\mathrm{done}}^{(m)}
=
S_{\mathrm{template}}
+
S_{\mathrm{protocol}}^{(m)}
+
S_{\mathrm{boundary}}^{(m)}
+
S_{\mathrm{semantic}}^{(m)}
}
$$

其中：

```text
S_template（完成模板状态）是强共享轴；
S_protocol（协议状态）决定 json/table/list 等结构化差异；
S_boundary（边界状态）决定句号、换行、EOS 等边界行为；
S_semantic（语义状态）决定答案是否真的完成。
```

当前测试主要验证了：

```text
S_template 强；
S_protocol 存在，尤其 GLM4/json_answer 明显；
S_semantic 仍未被充分分离。
```

### 图谱进度

Phase258 后图谱进度：

```text
pattern_family_atlas: 0.83
trace_signature_validation: 0.44
semantic_done_signature: 0.20
done_state_cluster_map: 0.16
residual_state_signature: 0.53
readout_competition_trace: 0.73
stepwise_rollout_trace: 0.41
causal_closure: 0.17
general_language_mechanism_confidence: 0.64
```

总体判断：

```text
语言模式图谱继续推进；
完成状态簇图谱刚打开；
闭合验证仍没有明显进展；
当前更接近“全局图谱拼图”，而不是“机制闭合”。
```

### 问题、硬伤和瓶颈

1. 测试指标可能被模板轴主导。answer_done_template 与 prompt_only 的差异很大，容易让跨模式复用率虚高。

2. 复用率高不等于闭合。当前只是投影顺序满足条件，没有验证真实 EOS stop，也没有因果注入。

3. 组内复用低于跨组复用是警告信号。它说明当前方向可能不是细粒度模式簇，而是一个更粗的完成模板轴。

4. JSON answer 暴露结构化语言机制缺口。它可能需要独立的结构闭合方向，不能用普通答案完成方向解释。

5. 小模型偏差仍然重要。qwen3 和 DS7B 的高相似度可能是结构粗糙，也可能是模板偏置更强，不能直接外推。

### 阶段结论

Phase258 的结论是：

```text
done state cluster 路线正确；
但当前分解出的第一主轴更像完成模板轴，而不是纯语义完成轴。
```

换句话说：

```text
完成状态不是单一方向；
也不是简单按模式族完全分裂；
而是共享模板轴 + 协议族局部轴 + 语义完成轴共同构成。
```

这比 Phase257 更进一步，因为它解释了为什么固定 v_done 可以弱复用，也解释了为什么 JSON answer 不稳定。

### 下一阶段任务

Phase259 仍属于当前大阶段。建议下一步不要继续扩大同类模板投影，而要把模板轴和语义轴拆开：

```text
Phase259: template-done 与 semantic-done 的解耦测试
```

具体方案：

```text
1. 构造四类前缀：
   A. 模板完整但语义错误；
   B. 语义正确但模板未完成；
   C. 模板完整且语义正确；
   D. 模板未完成且语义错误；

2. 分别测：
   template_projection（模板投影）
   semantic_projection（语义投影）
   boundary_projection（边界投影）

3. 检查 EOS / closure proxy 更依赖哪一类状态。
```

阶段目标：

```text
把 S_template 和 S_semantic 分开；
否则 done state cluster 会被模板轴污染，无法靠近真实语言闭合机制。
```

## Phase 259: template-done 与 semantic-done 的解耦测试 [2026-07-08 02:59]

### 任务来源

本阶段分析 Phase258 复盘内容是否正确，并继续同一阶段任务。Phase258 的判断基本正确：done state cluster（完成状态簇）路线是必要推进，但高复用率主要说明模板/协议轴很强，不能证明已经找到纯 semantic-done（语义完成）机制。

因此 Phase259 不再扩大同类模板投影，而是直接拆分：

```text
S_template（模板完成状态）
S_semantic（语义完成状态）
S_boundary（边界状态）
```

测试脚本：

```text
tests/gpt5/phase259_template_semantic_done_disentanglement.py
tests/gpt5/run_phase259_template_semantic_done_disentanglement.sh
```

测试结果：

```text
tests/result/phase259_template_semantic_done_disentanglement/template_semantic_done_disentanglement/
```

固定格式图谱数据已同步到前端，并通过：

```text
npm run sync:pattern-atlas
npm run build
```

构建通过，仅有既有大 chunk warning（大包警告）。

### 测试原理

本阶段构造四象限前缀：

```text
template_complete_semantic_correct（模板完整 + 语义正确）
template_complete_semantic_wrong（模板完整 + 语义错误）
template_incomplete_semantic_correct（模板不完整 + 语义正确）
template_incomplete_semantic_wrong（模板不完整 + 语义错误）
```

并额外加入边界前缀：

```text
boundary_complete_semantic_correct（边界完整 + 语义正确）
boundary_complete_semantic_wrong（边界完整 + 语义错误）
```

在每个模型中构造三个方向：

$$
\boxed{
v_{\mathrm{template}}
=
\operatorname{unit}
\left(
\frac{
(h_{TC}-h_{IC})+(h_{TW}-h_{IW})
}{2}
\right)
}
$$

$$
\boxed{
v_{\mathrm{semantic}}
=
\operatorname{unit}
\left(
\frac{
(h_{TC}-h_{TW})+(h_{IC}-h_{IW})+(h_{BC}-h_{BW})
}{3}
\right)
}
$$

$$
\boxed{
v_{\mathrm{boundary}}
=
\operatorname{unit}
\left(
\frac{
(h_{BC}-h_{IC})+(h_{BW}-h_{IW})
}{2}
\right)
}
$$

其中：

```text
TC = template complete + semantic correct；
TW = template complete + semantic wrong；
IC = incomplete template + semantic correct；
IW = incomplete template + semantic wrong；
BC = boundary complete + semantic correct；
BW = boundary complete + semantic wrong。
```

然后检查：

```text
template_axis 是否只响应模板完整；
semantic_axis 是否只响应语义正确；
boundary_axis 是否响应句号/边界；
closure proxy 是否跟这些轴同步变化。
```

### 客观结果

三模型均完成：

```text
vector_rows: 9
prefix_rows: 720
case_summary_rows: 120
observation_rows: 720
metric_rows: 24
graph_edges: 120
missing_rows: 0
```

跨模型解耦率：

```text
qwen3: 1.0
GLM4: 1.0
DS7B: 1.0
overall: 1.0
```

平均投影效应：

```text
template_axis_effect_correct: +120.060837
template_axis_effect_wrong: +166.050752
semantic_axis_effect_template: +13.505275
semantic_axis_effect_incomplete: +96.979530
```

平均 closure proxy 效应：

```text
closure_template_effect_correct: -6.834180
closure_semantic_effect_template: -0.049740
closure_boundary_effect_correct: -5.406445
```

分模型：

```text
qwen3:
  template_axis_effect_correct = +179.763319
  semantic_axis_effect_template = +20.767336
  closure_template_effect_correct = -8.278125
  closure_semantic_effect_template = -0.376563

GLM4:
  template_axis_effect_correct = +31.547640
  semantic_axis_effect_template = +2.800993
  closure_template_effect_correct = -4.681250
  closure_semantic_effect_template = -0.018750

DS7B:
  template_axis_effect_correct = +148.871551
  semantic_axis_effect_template = +16.947497
  closure_template_effect_correct = -7.543164
  closure_semantic_effect_template = +0.246094
```

方向相似度：

```text
qwen3:
  semantic_done vs template_done cosine = 0.306191
  boundary_done vs template_done cosine = 0.878367
  boundary_done vs semantic_done cosine = 0.378845

GLM4:
  semantic_done vs template_done cosine = 0.341111
  boundary_done vs template_done cosine = 0.777494
  boundary_done vs semantic_done cosine = 0.384409

DS7B:
  semantic_done vs template_done cosine = 0.326304
  boundary_done vs template_done cosine = 0.874964
  boundary_done vs semantic_done cosine = 0.172210
```

### 正确性分析

Phase259 是一个强校准结果。

正结果：

```text
template axis 和 semantic axis 可以在投影层面明显拆开；
三模型 semantic/template cosine 约 0.31 到 0.34；
boundary axis 与 template axis 高相似，约 0.78 到 0.88；
说明边界更接近输出模板/协议结构，而不是纯语义。
```

负结果更关键：

```text
template 投影增强没有提高 closure proxy，反而平均下降；
semantic 正确性几乎不提高 closure proxy；
boundary 完整也平均降低 closure proxy。
```

这说明：

```text
投影上的完成状态
≠
停止读出压力
≠
真实 EOS 执行。
```

因此 Phase259 没有提高 causal_closure（因果闭合），但它把机制缺口定位得更清楚：

```text
S_template 和 S_semantic 可以分离；
但它们没有自动接入 R_stop（停止读出）。
```

### 机制进展

Phase258 的公式：

$$
\boxed{
S_{\mathrm{done}}^{(m)}
=
S_{\mathrm{template}}
+
S_{\mathrm{protocol}}^{(m)}
+
S_{\mathrm{boundary}}^{(m)}
+
S_{\mathrm{semantic}}^{(m)}
}
$$

Phase259 后应进一步改为：

$$
\boxed{
S_{\mathrm{done}}
=
\left(
S_{\mathrm{template}},
S_{\mathrm{semantic}},
S_{\mathrm{boundary}}
\right)
}
$$

但：

$$
\boxed{
S_{\mathrm{done}}
\nRightarrow
R_{\mathrm{stop}}
}
$$

当前更完整的链条应写成：

$$
\boxed{
S_{\mathrm{semantic}}
\oplus
S_{\mathrm{template}}
\oplus
S_{\mathrm{boundary}}
\rightarrow
G_{\mathrm{rollout}}
\rightarrow
R_{\mathrm{stop}}
\rightarrow
\mathrm{EOS}
}
$$

其中：

```text
S_semantic / S_template / S_boundary 已能在投影上区分；
G_rollout 到 R_stop 的连接仍未破解；
R_stop 到 EOS 只在 GLM4 单候选中看到局部迹象。
```

### 图谱进度

Phase259 后图谱进度：

```text
pattern_family_atlas: 0.83
trace_signature_validation: 0.45
semantic_done_signature: 0.23
done_state_cluster_map: 0.20
template_semantic_disentanglement: 0.18
residual_state_signature: 0.54
readout_competition_trace: 0.73
stepwise_rollout_trace: 0.41
causal_closure: 0.17
general_language_mechanism_confidence: 0.64
```

整体判断：

```text
图谱拼图继续推进；
完成状态内部结构更清楚；
但闭合链路仍未贯通。
```

### 问题、硬伤和瓶颈

1. 四象限前缀是人工构造，不等于自然生成轨迹。它适合分离状态，但不能直接证明模型自然使用这些状态。

2. wrong_answer（错误答案）是通用替代词，语义错误强度可能不完全一致。

3. closure proxy 平均下降说明当前前缀可能改变了模型继续生成策略，而不是触发停止策略。

4. boundary axis 与 template axis 高相似，说明句号/边界在小模型中可能被协议模板吞并。

5. 仍没有因果注入。当前只是投影解耦，不是机制闭合。

### 阶段结论

Phase259 的结论是：

```text
template-done 与 semantic-done 可以在 hidden state 投影上分离；
boundary-done 更接近 template-done；
但三者都没有直接转化为 closure proxy 或 EOS stop。
```

这非常关键，因为它说明此前“完成状态簇”的主要瓶颈不在于无法分离状态，而在于：

```text
完成状态如何驱动 rollout trajectory 和 stop readout。
```

因此研究路线应从：

```text
继续找 done state
```

转向：

```text
寻找 S_done 到 R_stop 的桥接机制。
```

### 下一阶段任务

Phase260 仍属于当前大阶段，建议任务：

```text
Phase260: S_done 到 R_stop 的桥接层定位
```

具体方案：

```text
1. 使用 Phase259 的 template/semantic/boundary 三个方向；
2. 在多层 residual hidden state 上投影，而不是只看最终层；
3. 检查哪些层出现：
   semantic/template projection 上升；
   closure proxy 随后上升；
   EOS logit 随后上升；
4. 重点看 GLM4，因为它在前面 Phase255 出现真实 EOS 候选；
5. 只在桥接迹象强的层做小规模干预。
```

阶段目标：

```text
找到 S_done → R_stop 的中间层或桥接状态；
否则完成状态图谱仍然无法闭合到真实停止。
```

## Phase 260: S_done 到 R_stop 的桥接层定位 [2026-07-08 03:29]

### 任务来源

本阶段分析 Phase259 复盘内容是否正确，并继续同一阶段任务。Phase259 的判断是正确的：它不是闭合阶段，而是把完成状态拆成：

```text
S_template（模板完成状态）
S_semantic（语义完成状态）
S_boundary（边界状态）
```

并证明三者在 hidden state（隐藏状态）投影上可以区分。但 Phase259 同时证明：

```text
完成状态投影
≠
closure proxy（闭合代理）
≠
EOS stop（结束符停止）
```

因此 Phase260 的任务是定位：

```text
S_done → R_stop
```

的桥接层或桥接迹象。

测试脚本：

```text
tests/gpt5/phase260_sdone_rstop_bridge_layer_localization.py
tests/gpt5/run_phase260_sdone_rstop_bridge_layer_localization.sh
```

测试结果：

```text
tests/result/phase260_sdone_rstop_bridge_layer_localization/sdone_rstop_bridge_layer_localization/
```

固定格式图谱数据已同步到前端，并通过：

```text
npm run sync:pattern-atlas
npm run build
```

构建通过，仅有既有大 chunk warning（大包警告）。

### 测试原理

本阶段沿用 Phase259 的六类前缀：

```text
TC = template_complete_semantic_correct
TW = template_complete_semantic_wrong
IC = template_incomplete_semantic_correct
IW = template_incomplete_semantic_wrong
BC = boundary_complete_semantic_correct
BW = boundary_complete_semantic_wrong
```

但不再只看 final layer（最终层），而是在多个 residual layer（残差层）上分别构造：

```text
template_done
semantic_done
boundary_done
```

方向。

对每个层位 l：

$$
\boxed{
v_{\mathrm{template}}^{(l)}
=
\operatorname{unit}
\left(
\frac{
(h_{TC}^{(l)}-h_{IC}^{(l)})+
(h_{TW}^{(l)}-h_{IW}^{(l)})
}{2}
\right)
}
$$

$$
\boxed{
v_{\mathrm{semantic}}^{(l)}
=
\operatorname{unit}
\left(
\frac{
(h_{TC}^{(l)}-h_{TW}^{(l)})+
(h_{IC}^{(l)}-h_{IW}^{(l)})+
(h_{BC}^{(l)}-h_{BW}^{(l)})
}{3}
\right)
}
$$

$$
\boxed{
v_{\mathrm{boundary}}^{(l)}
=
\operatorname{unit}
\left(
\frac{
(h_{BC}^{(l)}-h_{IC}^{(l)})+
(h_{BW}^{(l)}-h_{IW}^{(l)})
}{2}
\right)
}
$$

然后比较：

```text
projection_effect（投影变化）
closure_proxy_effect（闭合代理变化）
eos_logit_effect（结束符 logit 变化）
```

桥接候选定义：

```text
projection_effect > 0
closure_proxy_effect > 0
并且多数样本同向
```

另设 EOS bridge（结束符桥接）：

```text
projection_effect > 0
eos_logit_effect > 0
并且多数样本同向
```

注意：EOS bridge 不等于 closure bridge，因为 EOS logit 上升不一定赢得整体停止读出竞争。

### 客观结果

三模型均完成：

```text
vector_rows: 54
case_layer_rows: 2160
layer_summary_rows: 54
observation_rows: 2160
metric_rows: 54
graph_edges: 54
missing_rows: 0
```

跨模型总结果：

```text
bridge_candidate_count: 6
eos_bridge_candidate_count: 42
```

按轴的 closure bridge rate（闭合桥接率）：

```text
template_done: 0.108333
semantic_done: 0.491667
boundary_done: 0.091667
```

按轴的 EOS bridge rate（结束符桥接率）：

```text
template_done: 0.700000
semantic_done: 0.488889
boundary_done: 0.691667
```

按轴的平均 closure proxy effect：

```text
template_done: -6.834180
semantic_done: -0.049740
boundary_done: -5.406445
```

按轴的平均 eos logit effect：

```text
template_done: +2.022526
semantic_done: +0.021875
boundary_done: +1.333532
```

分模型：

```text
qwen3:
  bridge_candidate_count = 0
  eos_bridge_candidate_count = 12
  template_done eos effect = +3.660693
  boundary_done eos effect = +2.516147
  closure effects 全部为负

GLM4:
  bridge_candidate_count = 0
  eos_bridge_candidate_count = 12
  template_done eos effect = +0.972510
  boundary_done eos effect = +0.101636
  closure effects 全部为负或近零

DS7B:
  bridge_candidate_count = 6
  eos_bridge_candidate_count = 18
  semantic_done closure effect = +0.246094
  semantic_done eos effect = +0.135547
```

关键候选：

```text
DS7B 的 semantic_done 在所有观察层都出现 weak closure bridge：
L10, L16, L22, L24, L26, L27

但 qwen3 和 GLM4 没有 closure bridge candidate。
```

### 正确性分析

Phase260 是一个强负结果 + 弱正迹象阶段。

强负结果：

```text
qwen3 和 GLM4 都没有 S_done → closure proxy 的桥接候选；
template_done 和 boundary_done 虽然强烈提高 EOS logit，
但 closure proxy 平均为负。
```

这说明：

```text
提高 EOS logit
≠
赢得停止读出竞争
```

也就是说，模型可能同时提高了 EOS，但 continuation、格式继续、解释继续等竞争项仍然更强，导致 closure proxy 不升反降。

弱正迹象：

```text
DS7B 的 semantic_done 出现 6 个 closure bridge candidate。
```

但必须谨慎，因为：

```text
1. 只出现在 DS7B；
2. 平均 closure effect 只有 +0.246094；
3. 当前模型是小模型，内部结构可能粗糙；
4. 每层候选重复性较强，可能来自同一组前缀差异，而不是真正层级传播。
```

因此不能说已经找到桥接机制，只能说：

```text
S_done → R_stop 的桥接在 DS7B 上有弱迹象；
qwen3 和 GLM4 当前不支持该桥接。
```

### 机制进展

Phase259 后的结构是：

$$
\boxed{
S_{\mathrm{semantic}}
\oplus
S_{\mathrm{template}}
\oplus
S_{\mathrm{boundary}}
\rightarrow
G_{\mathrm{rollout}}
\rightarrow
R_{\mathrm{stop}}
\rightarrow
\mathrm{EOS}
}
$$

Phase260 后必须再拆：

$$
\boxed{
S_{\mathrm{template/boundary}}
\rightarrow
z_{\mathrm{EOS}}
\not\Rightarrow
R_{\mathrm{stop}}
}
$$

以及：

$$
\boxed{
S_{\mathrm{semantic}}
\rightarrow
R_{\mathrm{stop}}
\quad
\text{only weakly in DS7B}
}
$$

其中：

```text
z_EOS = EOS logit；
R_stop = 停止读出竞争整体，包括 EOS、句号、换行、续写 token 之间的相对竞争。
```

这说明真实停止不是单个 EOS logit，而是：

$$
\boxed{
R_{\mathrm{stop}}
=
\max(z_{\mathrm{EOS}}, z_{\mathrm{period}}, z_{\mathrm{newline}})
-
z_{\mathrm{continuation}}
}
$$

当前问题是：

```text
template/boundary 可以推高 z_EOS，
但同时没有压低 continuation regime（续写机制），
所以 closure proxy 不提高。
```

### 图谱进度

Phase260 后图谱进度：

```text
pattern_family_atlas: 0.83
trace_signature_validation: 0.46
semantic_done_signature: 0.24
done_state_cluster_map: 0.21
template_semantic_disentanglement: 0.19
sdone_rstop_bridge: 0.08
residual_state_signature: 0.55
readout_competition_trace: 0.73
stepwise_rollout_trace: 0.41
causal_closure: 0.17
general_language_mechanism_confidence: 0.64
```

整体判断：

```text
完成状态图谱更清楚；
EOS logit 桥接有大量迹象；
closure bridge 很弱；
causal_closure 没有提升。
```

### 问题、硬伤和瓶颈

1. 当前桥接层测试仍是前缀静态测试，不是自然 rollout 动态测试。

2. closure effect 在同一前缀条件下与层无关，因此层位差异主要来自 projection 侧，不代表完整层级因果传播。

3. EOS bridge 很强但 closure bridge 很弱，说明测试必须继续加入 continuation suppression（续写抑制）或竞争读出分析。

4. DS7B 的弱正结果不能外推，可能是小模型粗糙结构或偶然模板响应。

5. 没有做因果注入，因此仍不能证明 bridge layer 是机制原因。

### 阶段结论

Phase260 的关键结论是：

```text
S_done 到 EOS logit 的桥接比较容易出现；
S_done 到整体 stop readout 的桥接很弱。
```

换句话说，当前真正缺口不是：

```text
模型是否知道“可以结束”
```

而是：

```text
模型如何让“结束”赢过“继续生成”。
```

这把下一步任务明确指向：

```text
stop-vs-continuation competition（停止-续写竞争）
```

### 下一阶段任务

Phase261 仍属于当前大阶段。建议任务：

```text
Phase261: stop-vs-continuation competition 的桥接验证
```

具体方案：

```text
1. 继续使用 Phase259/260 的 template、semantic、boundary 方向；
2. 同时构造 continuation_regime（续写机制）方向；
3. 测试：
   S_done 是否提高 EOS logit；
   S_done 是否压低 continuation logit；
   两者是否共同提高 closure proxy；
4. 对 GLM4 的真实 EOS 候选重点验证；
5. 如果存在组合方向，再做小规模 causal intervention。
```

阶段目标：

```text
从“完成状态是否存在”
推进到
“完成状态如何赢得停止-续写竞争”。
```

## Phase 261: 停止-续写竞争图谱的系统验证 [2026-07-08 03:48]

### 任务来源和总判断

本阶段综合了 Phase260 的强负结果和最新系统工程方案。两个判断基本正确：

```text
Phase260 证明的不是闭合失败本身，
而是把闭合链条中的主要瓶颈定位到 stop-vs-continuation competition（停止-续写竞争）。
```

Phase260 已经说明：

```text
S_done（完成状态） -> EOS logit（结束符读出）相对容易出现；
S_done（完成状态） -> 完整 R_stop（停止读出）很弱；
因此真正困难的是让“停止”赢过“继续生成”。
```

最新系统工程方案也正确：当前不应继续用一个小测试追一个局部闭合，而应进入语言闭合机制的系统图谱阶段。本阶段因此不追求 closure validation（闭合验证），而是先构建 stop-vs-continuation competition（停止-续写竞争）的固定格式图谱数据。

### 测试脚本和结果文件

测试脚本：

```text
tests/gpt5/phase261_stop_continuation_competition_atlas.py
tests/gpt5/run_phase261_stop_continuation_competition_atlas.sh
```

结果目录：

```text
tests/result/phase261_stop_continuation_competition_atlas/stop_continuation_competition_atlas/
```

核心输出：

```text
phase261_cross_model_summary.json
phase261_competition_rows.jsonl
phase261_effect_rows.jsonl
phase261_observations.jsonl
phase261_metrics.jsonl
phase261_graph_edges.jsonl
phase261_vector_rows.jsonl
phase261_stop_continuation_competition_atlas_report.md
```

本阶段已按固定 Pattern Atlas（模式图谱）格式生成数据，并同步到可视化客户端：

```text
npm run sync:pattern-atlas
npm run build
```

构建结果通过，但 Vite（前端构建工具）仍提示部分 chunk（代码块）较大，这是前端性能问题，不影响本阶段数据正确性。

### 测试原理

本阶段把 Phase259/260 的完成状态条件转化为三类竞争读出：

```text
1. template（模板完成）
2. semantic（语义正确）
3. boundary（边界/终止提示）
```

并在每个模型中同时记录停止候选和续写候选的最大读出值。

核心公式：

$$
R_{\text{stop}} =
\max(
z_{\text{EOS}},
z_{\text{period}},
z_{\text{newline}},
z_{\text{end\_boundary}}
)
$$

$$
R_{\text{continue}} =
\max(
z_{\text{the}},
z_{\text{because}},
z_{\text{and}},
z_{\text{comma}},
z_{\text{is}},
z_{\text{for}},
z_{\text{next\_sentence}}
)
$$

停止-续写竞争边界：

$$
M_{\text{close}} =
R_{\text{stop}} - R_{\text{continue}}
$$

判断规则：

```text
M_close > 0：停止读出赢；
M_close < 0：续写读出赢。
```

本阶段测试了 qwen3、GLM4、DS7B 三个模型，每个模型 40 个 case（案例），每个模型 240 条 competition rows（竞争记录），跨模型总计 720 条 competition rows（竞争记录）。

### 客观结果

跨模型总结果：

```text
vector_rows: 9
competition_rows: 720
effect_rows: 720
observation_rows: 720
metric_rows: 36
graph_edges: 18
missing_rows: 0
competition_winner_counts:
  continue: 504
  stop: 216
stop_win_rate: 0.300000
```

按模型看：

```text
qwen3:
  continue: 182
  stop: 58
  stop_win_rate: 0.241667

GLM4:
  continue: 156
  stop: 84
  stop_win_rate: 0.350000

DS7B:
  continue: 166
  stop: 74
  stop_win_rate: 0.308333
```

跨模型不同条件下的平均停止边界：

```text
template_complete_semantic_correct: -6.179688
template_complete_semantic_wrong:   -6.313021
template_incomplete_semantic_correct: 1.860417
template_incomplete_semantic_wrong:   2.039062
boundary_complete_semantic_correct: -7.292318
boundary_complete_semantic_wrong:   -7.016927
```

最重要的客观现象：

```text
template_complete 和 boundary_complete 条件下，M_close 多数为负；
template_incomplete 条件下，M_close 反而多数为正。
```

换句话说，在这些小模型上，“看起来更完整”的模板和边界提示没有稳定增强停止竞争，反而经常触发更强续写竞争。

跨模型 effect（效应）均值：

```text
template_effect_correct: -8.040104
template_effect_wrong:   -8.352083
semantic_effect_template: 0.133333
semantic_effect_incomplete: -0.178646
boundary_effect_correct: -9.152734
boundary_effect_wrong:   -9.055990
```

读出拆分：

```text
mean_r_stop_delta_by_effect:
  template_effect_correct: -9.134375
  template_effect_wrong:   -7.592448
  semantic_effect_template: -0.058594
  semantic_effect_incomplete: 1.483333
  boundary_effect_correct: -8.631380
  boundary_effect_wrong:   -7.257812

mean_r_continue_delta_by_effect:
  template_effect_correct: -1.094271
  template_effect_wrong:    0.759635
  semantic_effect_template: -0.191927
  semantic_effect_incomplete: 1.661979
  boundary_effect_correct:  0.521354
  boundary_effect_wrong:    1.798177
```

这说明 template（模板）和 boundary（边界）并不是单纯提高停止信号，而是会同时改变停止读出和续写读出；其中很多条件下，停止读出下降更明显，续写读出没有被充分压制。

### 结果分析

本阶段最重要结论不是“找到闭合方向”，而是确认 Phase260 的瓶颈判断：

```text
停止机制不是单一 EOS 激活问题，
而是停止候选和续写候选之间的竞争问题。
```

三个模型共同显示：

```text
continue winner（续写获胜）明显多于 stop winner（停止获胜）。
```

这意味着：

```text
模型可能已经有“答案完成”的局部状态，
但输出层仍然可以选择 because、the、逗号、下一句 等续写通道。
```

因此，后续不能只寻找 S_done（完成状态）或 EOS neuron（结束符神经元），而要分解：

```text
1. 哪些 continuation channel（续写通道）在赢；
2. 它们来自模板、语义、边界、任务格式还是预训练惯性；
3. stop readout（停止读出）是否需要增强；
4. continuation readout（续写读出）是否需要抑制；
5. 两者是否需要组合机制。
```

### 理论进展

本阶段对“语言是动态模式网络”的图谱路线提供了一个重要校准：

```text
语言模式不是只激活目标模式；
它还会激活相邻的延展模式、解释模式、补充模式、下一句模式。
```

从智能理论角度看，输出不是单一完成状态的投影，而是多个模式族在最后读出层竞争：

$$
O_t =
\operatorname{Readout}
\left(
S_{\text{answer}},
S_{\text{done}},
S_{\text{continue}},
S_{\text{format}},
S_{\text{task}}
\right)
$$

更适合当前证据的机制图是：

$$
P(y_{t+1})
=
\operatorname{softmax}
\left(
W_U h_t
{}+ \Delta_{\text{stop}}
{}+ \Delta_{\text{continue}}
{}+ \Delta_{\text{format}}
{}+ \Delta_{\text{task}}
{}+ \epsilon
\right)
$$

其中闭合不是：

$$
\Delta_{\text{stop}} > 0
$$

而是：

$$
\Delta_{\text{stop}}
-
\Delta_{\text{continue}}
>
\tau
$$

这解释了为什么 Phase260 中 EOS logit（结束符读出）可被提高，但 closure proxy（闭合代理）仍然很弱：因为续写通道没有被压下去。

### 问题、硬伤和瓶颈

1. 当前仍是静态前缀读出测试，不是自然 rollout（逐步生成）过程，不能证明真实生成时的动态闭合。

2. stop token（停止词元）和 continuation token（续写词元）集合是近似定义，不同模型 tokenizer（分词器）可能造成偏差。

3. template_complete（模板完成）和 boundary_complete（边界完成）在测试中可能引入了新的续写暗示，因此不能把它们等同于真实完成状态。

4. 本阶段没有做因果注入，只能说明读出竞争结构，不能证明具体层或方向是因果机制。

5. 当前模型都是小模型，内部编码结构可能粗糙，结论至少需要保留 30%-50% 的外推不确定性。

6. stop_win_rate 只有 0.30，说明当前图谱仍处于机制定位阶段，不能声称闭合机制已经破解。

### 当前图谱进度

```text
pattern_family_atlas: 0.84
trace_signature_validation: 0.46
semantic_done_signature: 0.24
done_state_cluster_map: 0.21
template_semantic_disentanglement: 0.19
sdone_rstop_bridge: 0.08
stop_continuation_competition: 0.12
residual_state_signature: 0.55
readout_competition_trace: 0.75
stepwise_rollout_trace: 0.41
causal_closure: 0.17
general_language_mechanism_confidence: 0.64
```

总体判断：

```text
语言模式图谱主干继续变清楚；
读出竞争图谱已有较高进展；
停止-续写竞争刚开始系统化；
真正闭合仍然较远。
```

### 阶段结论

Phase261 完成了 Phase260 之后最重要的一步：把“为什么 EOS 增强不等于闭合”的问题转化为可测的竞争图谱。

核心结论：

```text
当前瓶颈不是模型完全没有完成状态，
而是完成状态没有稳定赢过续写状态。
```

更直接地说：

```text
破解停止机制 = 找到停止增强 + 续写抑制的组合机制。
```

### 下一阶段任务

下一阶段仍属于当前系统工程大阶段，但应作为 Phase262 单独推进：

```text
Phase262: continuation regime decomposition atlas
```

任务目标：

```text
分解 continuation winner（续写获胜者）的来源。
```

具体方案：

```text
1. 按 continue_the、continue_because、continue_next_sentence、continue_comma 等通道分组；
2. 对不同任务模式、答案长度、标点、边界符号、解释诱导语进行矩阵测试；
3. 先建立 continuation source map（续写来源图谱）；
4. 再测试哪些状态能够压低 continuation channel（续写通道）；
5. 只有在图谱稳定后，再进入小规模 causal intervention（因果干预）。
```

阶段性目标：

```text
从“停止为什么输”
推进到
“续写具体从哪里赢”。
```

## Phase 262: 续写机制分解图谱 [2026-07-08 04:06]

### 任务来源和总判断

本阶段分析了最新附件中对 Phase261 的判断。总体结论正确：

```text
Phase261 的价值不是闭合成功，
而是把闭合瓶颈明确转化为 stop-vs-continuation competition（停止-续写竞争）。
```

附件中提出的下一步也正确：

```text
不要继续追单个 EOS 或单个 done direction（完成方向）；
应该系统分解 continuation regime（续写机制/续写场景）。
```

因此 Phase262 继续处于同一个系统工程大阶段，目标不是 closure validation（闭合验证），而是回答：

```text
续写具体从哪里赢？
```

### 测试脚本和结果文件

测试脚本：

```text
tests/gpt5/phase262_continuation_regime_decomposition_atlas.py
tests/gpt5/run_phase262_continuation_regime_decomposition_atlas.sh
```

结果目录：

```text
tests/result/phase262_continuation_regime_decomposition_atlas/continuation_regime_decomposition_atlas/
```

核心输出：

```text
phase262_cross_model_summary.json
phase262_continuation_channel_rows.jsonl
phase262_continuation_source_map_rows.jsonl
phase262_stop_continue_matrix_rows.jsonl
phase262_protocol_continuation_rows.jsonl
phase262_structured_continuation_rows.jsonl
phase262_token_coverage_rows.jsonl
phase262_observations.jsonl
phase262_metrics.jsonl
phase262_graph_edges.jsonl
phase262_continuation_decomposition_report.md
```

本阶段已按固定 Pattern Atlas（模式图谱）格式写入：

```text
tests/result/pattern_family_atlas/v1/
frontend/public/vis_data/pattern_family_atlas/v1/
```

并完成前端同步和构建：

```text
npm run sync:pattern-atlas
npm run build
```

构建通过，仍有 Vite（前端构建工具）chunk（代码块）较大的性能提示，不影响图谱数据读取。

### 测试设计

本阶段复用 Phase259/260/261 的 40 个基础 case（案例），每个模型执行：

```text
40 base cases
× 6 done conditions（完成条件）
× 9 continuation regimes（续写场景）
= 2160 matrix rows（矩阵记录）
```

三模型总计：

```text
qwen3 -> GLM4 -> DS7B
```

按顺序加载和释放，避免 GPU 显存叠加。

续写场景包括：

```text
plain
period_boundary
newline_boundary
comma_stub
because_stub
answer_anchor
json_structure
list_item
next_sentence
```

续写通道包括：

```text
continue_the
continue_because
continue_and
continue_comma
continue_is
continue_for
continue_next_sentence
continue_format
continue_json_structure
continue_list_item
```

### 核心公式

停止读出：

$$
R_{\mathrm{stop}}
=
\max_{t \in C_{\mathrm{stop}}} z_t
$$

第 k 个续写通道读出：

$$
R_{\mathrm{continue}}^{(k)}
=
\max_{t \in C_k} z_t
$$

总续写读出：

$$
R_{\mathrm{continue}}
=
\max_k R_{\mathrm{continue}}^{(k)}
$$

停止-续写边界：

$$
M_{\mathrm{close}}
=
R_{\mathrm{stop}}
-
R_{\mathrm{continue}}
$$

单通道压制需求：

$$
\Delta_k
=
R_{\mathrm{continue}}^{(k)}
-
R_{\mathrm{stop}}
$$

如果：

$$
\Delta_k > 0
$$

说明第 k 个续写通道压过停止通道，是后续 continuation suppression（续写抑制）的候选。

### 客观结果

跨模型总结果：

```text
matrix_rows: 6480
channel_rows: 64800
source_map_rows: 6480
protocol_rows: 1440
structured_rows: 1440
observation_rows: 6480
metric_rows: 60
graph_edges: 57
token_coverage_rows: 42
missing_rows: 0
```

停止-续写胜负：

```text
continue: 5661
stop: 819
stop_win_rate: 0.126389
```

按模型：

```text
qwen3:
  continue: 1859
  stop: 301
  stop_win_rate: 0.139352

GLM4:
  continue: 1982
  stop: 178
  stop_win_rate: 0.082407

DS7B:
  continue: 1820
  stop: 340
  stop_win_rate: 0.157407
```

注意：Phase262 的 stop_win_rate 低于 Phase261，不应解释为机制退步，而是因为本阶段主动加入了更多续写诱导场景、协议场景和结构化场景，因此更容易暴露续写通道。

### 续写通道排名

跨模型 continuation winner（续写获胜者）计数：

```text
continue_the: 2035
continue_list_item: 961
continue_because: 943
continue_next_sentence: 614
continue_format: 594
continue_json_structure: 171
continue_comma: 120
continue_is: 78
continue_for: 73
continue_and: 72
```

最重要的前三类：

```text
1. continue_the：自然语言延展续写；
2. continue_list_item：列表/结构化协议续写；
3. continue_because：解释型续写。
```

这说明续写不是一个统一方向，而是至少包含：

```text
自然语言续写；
解释续写；
结构化协议续写；
下一句续写；
格式继续。
```

### 续写来源图谱

source_hypothesis（来源假设）计数：

```text
structured_protocol_continuation: 1437
explanation_continuation: 1378
boundary_aftereffect_or_stop_failure: 1187
answer_protocol_continuation: 705
natural_language_continuation: 368
next_sentence_continuation: 338
template_induced_continuation: 248
```

最强来源不是单纯语义，而是：

```text
结构化协议；
解释续写；
边界后效应；
回答协议。
```

这对前面 Phase261 的异常现象给出了解释：

```text
template_complete / boundary_complete 可能不是“停止信号”，
而是会触发结构化协议、解释、下一句或格式继续。
```

### 场景结果

不同 regime（场景）的平均续写优势：

```text
plain: 4.118728
period_boundary: 6.523069
newline_boundary: 7.785395
comma_stub: 4.456923
because_stub: 2.666667
answer_anchor: 6.616504
json_structure: 6.546723
list_item: 7.565533
next_sentence: -0.432617
```

关键现象：

```text
newline_boundary、list_item、answer_anchor、json_structure、period_boundary
都会显著增强续写优势。
```

这说明：

```text
换行、列表、答案锚点、JSON 结构、句号边界
在小模型里经常不是停止边界，
而是协议继续或结构继续的触发源。
```

`next_sentence` 场景反而平均为负，这一点需要谨慎解释：它可能把下一句提示提前消耗掉，导致后续位置停止读出相对增强；也可能是 token bank（词元库）定义不完整导致的读出偏差。不能把它直接解释为“下一句提示有利于停止”。

### 通道边界结果

mean_channel_vs_stop_margin（通道相对停止均值）：

```text
continue_the: 1.822756
continue_list_item: 1.685750
continue_because: -0.376319
continue_format: -0.409276
continue_next_sentence: -0.604745
continue_is: -1.458539
continue_for: -1.805679
continue_comma: -1.976703
continue_and: -1.993179
continue_json_structure: -3.147655
```

这里有一个重要差异：

```text
winner count（获胜次数）高
不等于 average channel margin（平均通道边界）高。
```

`continue_because` 虽然平均边界不是最高，但在高价值 suppression candidates（抑制候选）中反复出现，说明它更像“少数场景极强”的解释续写触发源。

### 高价值抑制候选

最高候选主要集中在：

```text
qwen3
comma_stub / period_boundary / plain
continue_because
explanation_continuation
```

最大观测值：

```text
top_continue_vs_stop_margin: 34.5625
source_hypothesis: explanation_continuation
```

这说明下一阶段不应盲目抑制所有续写，而应优先测试：

```text
because / explanation continuation（解释续写）
list_item / structured protocol continuation（结构化协议续写）
the / natural language continuation（自然语言延展续写）
next_sentence / boundary aftereffect（边界后效应）
```

### 理论进展

Phase262 对语言动态模式网络理论的推进是：

```text
续写不是一个单通道变量，
而是多个 continuation regime（续写机制）构成的场。
```

更合适的机制公式是：

$$
R_{\mathrm{continue}}
=
\max
\left(
R_{\mathrm{the}},
R_{\mathrm{because}},
R_{\mathrm{list}},
R_{\mathrm{format}},
R_{\mathrm{next}},
R_{\mathrm{json}},
\cdots
\right)
$$

闭合条件因此从：

$$
R_{\mathrm{stop}} > R_{\mathrm{continue}}
$$

细化为：

$$
R_{\mathrm{stop}}
>
\max_k R_{\mathrm{continue}}^{(k)}
$$

也就是：

```text
停止必须赢过所有主要续写通道，
而不是只赢过一个平均续写方向。
```

这解释了为什么前面多个“平均方向”“正交方向”“done direction（完成方向）”难以闭合：它们可能只压住了部分续写通道，没有压住结构化协议续写或解释续写。

### 问题、硬伤和瓶颈

1. 本阶段仍是静态前缀读出测试，不是自然 rollout（生成展开）测试，因此不能证明这些通道在真实生成中按同样顺序激活。

2. continuation token bank（续写词元库）仍然是人工近似，虽然记录了 token coverage（词元覆盖），但不同 tokenizer（分词器）仍可能带来偏差。

3. 场景构造有诱导性：comma_stub、because_stub、json_structure、list_item 本来就是续写触发源，因此本阶段适合做来源分解，不适合估计自然闭合概率。

4. source_hypothesis（来源假设）是规则归因，不是因果归因。它只能作为下一阶段干预候选，不能直接当作机制结论。

5. 小模型偏差仍然需要保留 30%-50%。小模型可能更容易受模板和结构化符号牵引，真实大模型的续写场可能更平滑或更可控。

6. 本阶段没有做 continuation suppression（续写抑制）因果注入，因此 causal_closure（因果闭合）不应提高。

### 当前图谱进度

```text
pattern_family_atlas: 0.85
trace_signature_validation: 0.46
semantic_done_signature: 0.24
done_state_cluster_map: 0.21
template_semantic_disentanglement: 0.19
sdone_rstop_bridge: 0.08
stop_continuation_competition: 0.18
continuation_regime_decomposition: 0.16
residual_state_signature: 0.55
readout_competition_trace: 0.76
stepwise_rollout_trace: 0.41
causal_closure: 0.17
general_language_mechanism_confidence: 0.65
```

总体判断：

```text
语言模式图谱继续推进；
续写机制图谱已经打开；
停止-续写竞争从“是否续写赢”推进到“哪类续写赢”；
闭合因果仍未突破。
```

### 阶段结论

Phase262 完成了 Phase261 后最关键的分解任务：

```text
续写胜出不是一个整体黑箱，
而是由自然语言延展、解释续写、结构化协议续写、边界后效应和格式继续共同组成。
```

当前最关键的客观结论：

```text
continue_the、continue_list_item、continue_because
是最优先研究的三类续写通道。
```

从破解语言编码机制的角度看，这一步把“停止为什么输”进一步拆成：

```text
停止输给了哪一种语言模式。
```

### 下一阶段任务

下一阶段仍属于当前系统工程大阶段，应作为 Phase263 推进：

```text
Phase263: continuation suppression candidate causal audit
```

任务目标：

```text
验证能否压低高价值续写通道，
以及压低续写通道是否真的改善 M_close。
```

建议优先测试三类候选：

```text
1. explanation continuation suppression:
   目标通道 continue_because；

2. structured protocol continuation suppression:
   目标通道 continue_list_item / continue_format / continue_json_structure；

3. natural language extension suppression:
   目标通道 continue_the。
```

阶段成功标准：

```text
1. 至少找到 1 到 2 类可稳定压低的续写通道；
2. M_close 有可重复改善；
3. 不显著破坏 answer readout（答案读出）；
4. 仍不宣布闭合，直到 rollout 和真实 ModelStopExecuted（模型停止执行）验证完成。
```

## Phase 263: 续写抑制候选因果审计 [2026-07-08 04:24]

### 任务来源和总判断

本阶段分析了最新附件对 Phase262 的复盘。附件判断基本正确：

```text
Phase262 的价值不是闭合成功，
而是把 continuation（续写）拆成多个 continuation regime（续写机制）。
```

附件提出的下一步也正确：

```text
不能继续只做续写来源统计；
必须测试能否压低高价值续写通道，
以及压低后 M_close 是否改善。
```

因此 Phase263 进入 continuation suppression candidate causal audit（续写抑制候选因果审计）。但本阶段必须严格限定结论：

```text
这是 final-hidden/readout-level intervention（最终隐状态/读出层级干预），
不是完整内部机制闭合，
也不是自然生成闭合成功。
```

### 测试脚本和结果文件

测试脚本：

```text
tests/gpt5/phase263_continuation_suppression_candidate_causal_audit.py
tests/gpt5/run_phase263_continuation_suppression_candidate_causal_audit.sh
```

结果目录：

```text
tests/result/phase263_continuation_suppression_candidate_causal_audit/continuation_suppression_candidate_causal_audit/
```

核心输出：

```text
phase263_cross_model_summary.json
phase263_continuation_suppression_rows.jsonl
phase263_channel_causal_effect_rows.jsonl
phase263_stop_plus_suppression_rows.jsonl
phase263_answer_preservation_rows.jsonl
phase263_rollout_probe_rows.jsonl
phase263_observations.jsonl
phase263_metrics.jsonl
phase263_graph_edges.jsonl
phase263_continuation_suppression_report.md
```

本阶段已同步固定 Pattern Atlas（模式图谱）格式数据到前端，并完成构建：

```text
npm run sync:pattern-atlas
npm run build
```

构建通过，仍有 Vite（前端构建工具）chunk（代码块）较大的性能提示，不影响本阶段数据。

### 测试原理

Phase263 从 Phase262 的高价值 suppression candidates（抑制候选）中，每个模型选择：

```text
explanation continuation（解释续写）
structured protocol continuation（结构化协议续写）
natural language extension（自然语言延展续写）
boundary aftereffect（边界后效应）
```

每类 8 个候选，三模型各 32 个候选。

干预策略：

```text
suppress_explanation
suppress_structured
suppress_natural
suppress_boundary_aftereffect
suppress_top
stop_plus_top
```

干预强度：

```text
lambda: 2, 4, 8, 12
alpha_stop: 4
```

核心思想是构造续写通道的 unembedding direction（输出嵌入方向），在最终隐状态上做读出层级干预。

单通道抑制：

$$
h'
=
h
-
\lambda v_{\mathrm{continue}}^{(k)}
$$

停止增强 + top continuation 抑制：

$$
h'
=
h
+
\alpha v_{\mathrm{stop}}
-
\lambda v_{\mathrm{continue}}^{(\mathrm{top})}
$$

观察：

$$
\Delta M_{\mathrm{close}}
=
M_{\mathrm{close}}(h')
-
M_{\mathrm{close}}(h)
$$

同时记录：

```text
top_channel_logit_delta（目标续写通道变化）
target_logit_delta（答案读出变化）
winner_flip_to_stop（是否翻转为停止胜出）
rollout_stop_rate（小规模生成是否真实停止）
```

### 客观结果

跨模型总量：

```text
suppression_rows: 2304
channel_causal_effect_rows: 2304
stop_plus_rows: 384
answer_preservation_rows: 2304
rollout_probe_rows: 45
observation_rows: 2304
metric_rows: 27
graph_edges: 18
missing_rows: 0
```

跨模型平均 M_close 改变量：

```text
suppress_explanation: 1.121358
suppress_structured: -1.291441
suppress_natural: -0.035718
suppress_boundary_aftereffect: 1.704183
suppress_top: 2.304891
stop_plus_top: 4.339233
```

目标续写通道 logit 变化：

```text
suppress_explanation: -1.217122
suppress_structured: -0.780436
suppress_natural: -1.270020
suppress_boundary_aftereffect: -1.342122
suppress_top: -2.988770
stop_plus_top: -2.733236
```

答案保持率：

```text
suppress_explanation: 1.000000
suppress_structured: 1.000000
suppress_natural: 1.000000
suppress_boundary_aftereffect: 0.973958
suppress_top: 0.989583
stop_plus_top: 0.908854
```

winner flip rate（翻转为停止胜出率）：

```text
所有策略均为 0.000000
```

小规模 rollout（生成展开）：

```text
no_patch:
  mean_generated_tokens: 24.000000
  model_stop_rate: 0.000000

suppress_top:
  mean_generated_tokens: 24.000000
  model_stop_rate: 0.000000

stop_plus_top:
  mean_generated_tokens: 21.866667
  model_stop_rate: 0.200000
```

### 分模型结果

qwen3：

```text
stop_plus_top mean_stop_margin_delta: 5.833496
suppress_top mean_stop_margin_delta: 3.039307
suppress_boundary_aftereffect: 2.394531
suppress_explanation: 1.628418
rollout_stop_rate(stop_plus_top): 0.000000
target_preserved_rate(stop_plus_top): 0.914062
```

GLM4：

```text
stop_plus_top mean_stop_margin_delta: 3.015137
suppress_top mean_stop_margin_delta: 1.594971
suppress_boundary_aftereffect: 1.082275
suppress_explanation: 0.796265
rollout_stop_rate(stop_plus_top): 0.200000
target_preserved_rate(stop_plus_top): 1.000000
```

DS7B：

```text
stop_plus_top mean_stop_margin_delta: 4.169067
suppress_top mean_stop_margin_delta: 2.280396
suppress_boundary_aftereffect: 1.635742
suppress_explanation: 0.939392
rollout_stop_rate(stop_plus_top): 0.400000
target_preserved_rate(stop_plus_top): 0.812500
```

### 结果分析

本阶段是弱正结果 + 强校准结果。

弱正结果：

```text
1. 续写抑制确实能压低目标续写通道；
2. suppress_top 和 stop_plus_top 能稳定改善 M_close；
3. explanation suppression 和 boundary-aftereffect suppression 在三模型上平均为正；
4. stop_plus_top 在 GLM4 和 DS7B 的小规模 rollout 中出现真实停止。
```

强校准结果：

```text
1. 所有策略 winner_flip_rate 都是 0；
2. 单纯 suppress_top 不产生真实停止；
3. stop_plus_top 只产生 0.20 的 rollout_stop_rate；
4. qwen3 即使 M_close 改善最大，也没有 rollout stop；
5. stop_plus_top 对答案读出有一定损伤风险。
```

这说明：

```text
读出层级抑制可以改善边界，
但还不足以完成真实闭合。
```

### 关键洞察

Phase263 证明了一个重要事实：

```text
续写通道不是不可压；
但压低续写通道不等于停止胜出，
更不等于模型真实停止。
```

这与 Phase254 以来的路线一致：

```text
readout improvement（读出改善）
≠
ModelClose（模型闭合）。
```

当前更准确的闭合链条应写成：

$$
S_{\mathrm{done}}
\rightarrow
R_{\mathrm{stop}}
\uparrow
\land
R_{\mathrm{continue}}^{(k)}
\downarrow
\rightarrow
M_{\mathrm{close}}
\uparrow
\rightarrow
\mathrm{WinnerFlip}
\rightarrow
\mathrm{RolloutStop}
\rightarrow
\mathrm{NoDrift}
$$

Phase263 只完成到：

```text
R_continue^(k) 下降；
M_close 上升；
少量 rollout stop。
```

还没有完成：

```text
winner flip 稳定发生；
自然生成稳定停止；
无漂移。
```

### 问题、硬伤和瓶颈

1. 本阶段是 final-hidden/readout-level intervention（最终隐状态/读出层级干预），不是模型内部自然机制定位。

2. winner_flip_rate 为 0，说明当前候选虽然改善 M_close，但还没跨过停止胜出的阈值。

3. stop_plus_top 有更强效果，但会带来答案读出损伤风险，尤其 DS7B 的 target_preserved_rate 只有 0.812500。

4. suppress_structured 平均为负，说明结构化协议续写不是简单沿一个方向抑制就能改善；它可能需要协议完成检测，而不是粗暴压制。

5. suppress_natural 基本无效，说明 continue_the 可能是背景语言流，不是一个容易单独压制的局部机制。

6. rollout_probe 规模较小，只能作为迹象，不能作为闭合验证。

7. 当前仍是小模型测试，需要保留 30%-50% 外推偏差。

### 图谱进度

```text
pattern_family_atlas: 0.85
trace_signature_validation: 0.46
semantic_done_signature: 0.24
done_state_cluster_map: 0.21
template_semantic_disentanglement: 0.19
sdone_rstop_bridge: 0.08
stop_continuation_competition: 0.20
continuation_regime_decomposition: 0.18
continuation_suppression_causal_audit: 0.10
residual_state_signature: 0.55
readout_competition_trace: 0.77
stepwise_rollout_trace: 0.42
causal_closure: 0.18
general_language_mechanism_confidence: 0.65
```

总体判断：

```text
图谱继续推进；
读出层级因果链有弱正结果；
真实闭合仍未突破；
下一步必须从读出层级推进到 rollout 稳定性和 winner flip 阈值。
```

### 阶段结论

Phase263 的核心结论：

```text
续写抑制是有效方向，
但单独续写抑制不够；
必须结合停止增强、阈值跨越和生成展开稳定。
```

更具体地说：

```text
suppress_top 可以压低续写并改善 M_close；
stop_plus_top 效果最强，并产生少量真实停止；
但 winner_flip 仍未稳定发生。
```

因此，当前不能宣布闭合，但可以确认：

```text
停止增强 + top continuation suppression
是下一阶段最值得追的组合机制。
```

### 下一阶段任务

下一阶段仍属于当前系统工程大阶段，应作为 Phase264 推进：

```text
Phase264: stop-plus-continuation suppression threshold sweep with rollout validation
```

任务目标：

```text
系统扫描 alpha_stop 和 lambda_suppression，
找到能稳定跨过 winner flip 阈值的组合，
并验证 32/64 token rollout 是否减少过生成。
```

建议方案：

```text
1. 只保留 Phase263 中有效的 suppress_top 和 stop_plus_top；
2. 对 alpha_stop 与 lambda_suppression 做二维网格；
3. 分别统计 M_close、winner_flip、target_preservation、rollout_stop_rate；
4. 单独处理 structured protocol，因为 suppress_structured 平均为负；
5. 不再盲目扩大通道，而是寻找阈值曲线。
```

阶段成功标准：

```text
1. winner_flip_rate 明显大于 0；
2. rollout_stop_rate 明显高于 baseline；
3. target_preserved_rate 不低于 0.90；
4. 至少在两个模型上复现；
5. 仍不宣布最终闭合，直到 NoDrift 和长程 rollout 验证完成。
```

## Phase 264: 语言模式族物理路径图谱第一版 [2026-07-08 04:55]

### 任务来源和总判断

本阶段综合两个附件。

第一个附件对 Phase263 的判断正确：

```text
Phase263 比 Phase262 更接近因果层；
它证明续写通道可以被压低，M_close 可以改善；
但 winner flip、稳定 ModelStopExecuted、NoDrift 和完整闭合都没有完成。
```

第二个附件提出了更重要的方法论修正，也基本正确：

```text
当前不应继续把主目标放在闭合、单点 patch、单方向修复、单个 EOS 机制；
应该转向语言模式族物理编码路径图谱。
```

因此，本阶段对上一阶段建议做了修正：

```text
Phase264 不继续做 stop_plus_top 阈值扫描；
而是先把 Phase261-263 的数据整理为第一版 physical path atlas（物理路径图谱）。
```

阈值扫描不是放弃，而是降级为后续稳定路径候选的因果审计工具。

### 脚本和结果文件

脚本：

```text
tests/gpt5/phase264_language_pattern_physical_path_atlas.py
tests/gpt5/run_phase264_language_pattern_physical_path_atlas.sh
```

结果目录：

```text
tests/result/phase264_language_pattern_physical_path_atlas/language_pattern_physical_path_atlas/
```

核心输出：

```text
phase264_cross_model_summary.json
phase264_mode_family_case_bank_v3.jsonl
phase264_internal_path_trace_rows.jsonl
phase264_state_factor_projection_rows.jsonl
phase264_readout_competition_rows.jsonl
phase264_rollout_trace_rows.jsonl
phase264_path_cluster_rows.jsonl
phase264_mechanism_candidate_rows.jsonl
phase264_observations.jsonl
phase264_metrics.jsonl
phase264_graph_edges.jsonl
phase264_language_pattern_physical_path_atlas_report.md
```

本阶段没有重新加载模型，没有新增 CUDA 测试，而是复用 Phase261、Phase262、Phase263 的固定格式结果，生成第一版路径图谱。数据已同步到可视化客户端，并完成构建：

```text
npm run sync:pattern-atlas
npm run build
```

构建通过，仍有前端 chunk（代码块）较大的提示，不影响图谱数据。

### 算法原理

本阶段把已有测试结果从“指标表”改造成“路径表”。

核心路径定义：

$$
\mathcal{M}(P_i)
=
(
\mathcal{L}_i,
\mathcal{C}_i,
\mathcal{S}_i,
\mathcal{R}_i,
\mathcal{G}_i
)
$$

其中：

```text
L_i = layer path（层路径），当前阶段暂用 final hidden/readout-level 近似；
C_i = component path（组件路径），当前记录 final_hidden 和 lm_head_readout；
S_i = state path（状态路径）；
R_i = readout competition path（读出竞争路径）；
G_i = rollout trajectory（生成轨迹路径）。
```

路径签名：

$$
\Sigma_i
=
[
\mathrm{family},
\mathrm{state\_path},
\mathrm{component\_path},
\mathrm{readout\_winner},
\mathrm{rollout\_class},
\mathrm{closure\_class}
]
$$

状态因子坐标：

$$
s_{i}
=
[
S_{\mathrm{template}},
S_{\mathrm{semantic}},
S_{\mathrm{boundary}},
S_{\mathrm{protocol}},
S_{\mathrm{structure}},
S_{\mathrm{continue}},
S_{\mathrm{stop}},
S_{\mathrm{done}}
]
$$

需要注意：这些状态因子只是观测坐标和标签，不是假设为真实正交方向。

### 数据规模

Phase264 汇总了 Phase261-263 的结果，生成：

```text
case_bank_rows: 6480
path_signature_rows: 6480
state_factor_rows: 6480
readout_rows: 6480
rollout_rows: 45
path_cluster_rows: 191
mechanism_candidate_rows: 928
observation_rows: 6480
metric_rows: 4
graph_edges: 191
```

这一步的核心价值不是新增模型行为结果，而是把已有客观现象组织成可继续扩展的路径图谱。

### 关键路径簇

最强跨模型稳定簇：

```text
cluster:
  output_protocol / explanation_continuation / continue_because / continuation_dominant

path_count: 568
model_counts:
  qwen3: 232
  GLM4: 119
  DS7B: 217
cross_model_coverage: 3
mean_stop_continue_margin: -9.331784
mean_top_continue_vs_stop_margin: 9.331784
mean_best_stop_margin_delta: 0.292804
rollout_stop_count: 0
```

解释：

```text
解释续写 because 是跨模型稳定的强失败路径，
但 Phase263 的读出层级补丁对这个大簇平均修复很弱。
```

第二类稳定簇：

```text
output_protocol / structured_protocol_continuation / continue_list_item / continuation_dominant

path_count: 448
cross_model_coverage: 3
mean_stop_continue_margin: -6.614816
mean_best_stop_margin_delta: 0.000000
```

解释：

```text
结构化协议续写是稳定失败路径，
但不能用简单续写抑制来修复。
```

第三类稳定簇：

```text
output_protocol / structured_protocol_continuation / continue_the / continuation_dominant

path_count: 410
cross_model_coverage: 3
mean_stop_continue_margin: -7.204764
```

解释：

```text
结构化协议不仅会触发 list_item，也会触发自然语言延展。
```

另一个重要簇：

```text
output_protocol / next_sentence_continuation / continue_list_item / static_stop_winner

path_count: 261
cross_model_coverage: 3
mean_stop_continue_margin: 1.229885
```

解释：

```text
某些 next_sentence 条件在静态读出上反而接近 stop winner，
但还没有 rollout stop，因此不能解释为闭合。
```

### 机制候选

共生成：

```text
mechanism_candidate_rows: 928
```

高优先级候选主要集中在：

```text
qwen3
output_protocol
explanation_continuation
continue_because
period_boundary / comma_stub / plain
```

最高候选示例：

```text
model: qwen3
mode_id: explain_answer
condition: template_incomplete_semantic_correct
regime_id: period_boundary
source_hypothesis: explanation_continuation
top_continue_channel: continue_because
best_stop_margin_delta: 13.125
priority_score: 11.871875
status: candidate_not_closure
```

另一个重要候选：

```text
model: DS7B
mode_id: stop_after_answer
regime_id: comma_stub
source_hypothesis: explanation_continuation
top_continue_channel: continue_because
best_stop_margin_delta: 7.25
rollout_stop_seen: true
```

这说明：

```text
解释续写路径既是强失败路径，
也是当前最值得做局部因果审计的路径。
```

### 当前图谱进度

```text
pattern_family_atlas: 0.86
physical_path_atlas: 0.24
state_factor_atlas: 0.34
path_cluster_mining: 0.12
trace_signature_validation: 0.47
readout_competition_trace: 0.78
stepwise_rollout_trace: 0.43
causal_closure: 0.18
general_language_mechanism_confidence: 0.66
```

与附件给出的判断一致：

```text
全局模式图谱已经较清楚；
但物理路径图谱仍很早；
闭合不应作为当前主驱动。
```

### 理论进展

本阶段的理论改进不是提出新名词，而是修正研究顺序。

旧顺序：

```text
公式 -> patch -> 闭合
```

新顺序：

```text
大样本图谱
-> 路径签名
-> 路径聚类
-> 稳定路径
-> 少量因果审计
-> 闭合候选
```

统一机制公式应暂时作为图谱组织工具，而不是最终机制：

$$
\mathrm{LanguageMechanism}
=
\sum_i
\alpha_i(x,t)
P_i(x,t)
$$

其中每个模式路径：

$$
P_i
=
[
T_i,
G_i,
A_i,
M_i,
R_i,
O_i,
L_i,
K_i
]
$$

当前真正要破解的是：

```text
某个语言模式族如何在层、组件、状态、读出和生成轨迹中形成稳定路径。
```

### 问题和硬伤

1. Phase264 是路径图谱聚合，不是新模型测试，因此不能提供新的内部激活证据。

2. 当前 component_path（组件路径）仍粗，只记录到 final_hidden 和 lm_head_readout，还没有完整 attention、MLP、residual 层级追踪。

3. state_factor 是标签化坐标，不是真实独立状态方向。

4. path_cluster 是规则聚类，不是深层机制聚类，后续需要加入层位、组件、生成步数据。

5. 结果目前主要覆盖 output_protocol 和 readout_competition，九大模式族尚未完整覆盖。

6. rollout_rows 只有 45 条，生成轨迹图谱仍明显不足。

7. 当前仍基于小模型，需要保留 30%-50% 外推偏差。

### 阶段结论

Phase264 完成了一个重要路线校准：

```text
从闭合/patch 主线，
转向语言模式族物理编码路径图谱主线。
```

本阶段已经把 Phase261-263 的结果组织成：

```text
case bank；
state factor；
readout path；
rollout trace；
path cluster；
mechanism candidate。
```

这说明研究方向已经从“某个指标能不能修好”推进到：

```text
哪些语言模式族在模型中形成稳定失败路径和候选路径。
```

### 下一阶段任务

下一阶段仍属于 Phase264-300 大阶段，应作为 Phase265 推进：

```text
Phase265: multi-family case bank and path schema expansion
```

任务目标：

```text
把当前主要集中在 output_protocol 的路径图谱扩展到九大语言模式族。
```

建议方案：

```text
1. 建立 mode_family_case_bank_v3；
2. 覆盖 content_knowledge、reasoning_constraint、syntax_structure、language_action、cross_lingual、state_drift、closure 等模式族；
3. 每个样本固定 target、protocol、boundary、done、continuation、scoring risk 标签；
4. 不急于跑因果 patch；
5. 先让路径图谱覆盖面变完整。
```

阶段成功标准：

```text
1. 九大模式族都有固定格式 case bank；
2. 每个模式族至少有基础样本、对照样本、扰动样本、边界变体；
3. 可视化客户端能读取并筛选模式族、路径簇、候选机制；
4. 后续内部追踪脚本可以直接消费这些样本。
```

## Phase 265: 九大语言模式族样本库与路径 Schema 扩展 [2026-07-08 05:14]

### 任务来源和总判断

本阶段分析了 Phase264 复盘附件。附件判断正确：

```text
Phase264 是关键路线升级；
它不是新增模型行为，也不是闭合证明；
它的价值是把 Phase261-263 的结果重组成 language pattern physical path atlas（语言模式族物理路径图谱）。
```

附件指出的主要问题也正确：

```text
当前路径图谱主要集中在 output_protocol（输出协议）和 readout_competition（读出竞争）；
九大语言模式族还没有完整覆盖；
因此第一优先级应是补全语言族物理路径样本库。
```

所以 Phase265 不做模型 forward（前向计算），不做闭合，也不做 patch（补丁）。本阶段目标是：

```text
建立九大语言模式族统一 case bank（样本库）和 path schema（路径模式），
让后续内部追踪脚本可以直接消费这些样本。
```

### 脚本和结果文件

脚本：

```text
tests/gpt5/phase265_multi_family_case_bank_path_schema_expansion.py
tests/gpt5/run_phase265_multi_family_case_bank_path_schema_expansion.sh
```

结果目录：

```text
tests/result/phase265_multi_family_case_bank_path_schema_expansion/multi_family_case_bank_path_schema_expansion/
```

核心输出：

```text
phase265_cross_model_summary.json
phase265_mode_family_case_bank_v3.jsonl
phase265_mode_variant_matrix_rows.jsonl
phase265_case_quality_audit_rows.jsonl
phase265_path_schema_rows.jsonl
phase265_state_factor_design_rows.jsonl
phase265_observations.jsonl
phase265_metrics.jsonl
phase265_graph_edges.jsonl
phase265_multi_family_case_bank_report.md
```

同时同步到全局 Pattern Atlas（模式图谱）：

```text
tests/result/pattern_family_atlas/v1/test_cases.jsonl
tests/result/pattern_family_atlas/v1/mode_family_case_bank_v3.jsonl
tests/result/pattern_family_atlas/v1/mode_variant_matrix_rows.jsonl
tests/result/pattern_family_atlas/v1/case_quality_audit_rows.jsonl
tests/result/pattern_family_atlas/v1/path_schema_rows.jsonl
tests/result/pattern_family_atlas/v1/state_factor_design_rows.jsonl
```

前端同步和构建已完成：

```text
npm run sync:pattern-atlas
npm run build
```

构建通过，仍有前端 chunk（代码块）较大的性能提示，不影响数据读取。

### 数据规模

本阶段生成：

```text
case_rows: 1296
matrix_rows: 1296
quality_rows: 1296
path_schema_rows: 216
state_factor_rows: 1296
observation_rows: 1296
metric_rows: 11
graph_edges: 216
```

九大模式族每族样本数：

```text
content_knowledge: 144
output_protocol: 144
reasoning_constraint: 144
syntax_structure: 144
language_action: 144
cross_lingual: 144
readout_competition: 144
state_drift: 144
closure: 144
```

变体覆盖：

```text
base: 216
protocol: 216
continuation: 216
boundary: 216
structure: 432
```

评分风险：

```text
low: 504
medium: 792
high: 0
```

全局 test_cases.jsonl 当前总量：

```text
1620
```

### 路径 Schema

每条样本固定记录：

```text
target
expected_pattern
output_protocol
boundary_type
done_label
continuation_trigger
scoring_risk
state factor labels
path_schema_id
```

每个 path schema（路径模式）包含：

```text
trigger_trace
component_trace_targets
state_trace_targets
readout_trace_targets
rollout_trace_targets
closure_trace_targets
```

组件追踪目标：

```text
embedding
attention_out
mlp_gate
mlp_up
mlp_product
mlp_down
residual_stream
lm_head
```

状态追踪目标：

```text
S_content
S_target
S_protocol
S_boundary
S_done
S_continue
S_stop
S_structure
```

读出竞争目标：

```text
target_vs_wrong
stop_vs_continuation
protocol_vs_drift
structure_completion_vs_structure_continuation
```

生成轨迹目标：

```text
step_1
step_4
step_8
step_16
step_32
answer_step
drift_step
eos_step
```

闭合追踪目标：

```text
AnswerCorrect
PatternMatched
BoundaryStable
DoneStateStable
ModelStopExecuted
NoDrift
```

### 理论意义

Phase265 的意义是把“语言族物理路径”从口头方案变成可执行数据结构。

当前路径对象可以写成：

$$
\mathcal{M}(P_i)
=
(
\mathcal{L}_i,
\mathcal{C}_i,
\mathcal{S}_i,
\mathcal{R}_i,
\mathcal{G}_i
)
$$

Phase265 主要补齐的是：

```text
P_i 的族覆盖；
S_i 的状态标签；
R_i 的读出竞争任务；
G_i 的 rollout 追踪入口；
C_i 的组件追踪目标。
```

它还没有真正测出：

```text
L_i 的层路径；
C_i 的组件激活；
S_i 的真实内部状态；
G_i 的真实生成轨迹。
```

但它让后续测试可以不再临时造样本，而是按统一 schema 执行。

### 问题和硬伤

1. 本阶段是 case-bank/schema expansion（样本库/路径模式扩展），不是模型测试，因此没有新增模型行为证据。

2. 样本是设计样本，不是自动从真实失败分布中采样，可能存在人工模板偏差。

3. 每个模式族 144 条样本只是第一版覆盖，距离每族 200-500 条高质量样本还有距离。

4. state factor labels（状态因子标签）仍是设计标签，不是测量出来的内部变量。

5. scoring risk（评分风险）目前只有 low/medium，没有经过真实模型输出校准。

6. 结构化、跨语言、语法样本需要后续用真实输出做质量校验。

7. 当前仍没有完成内部组件路径追踪。

### 当前图谱进度

```text
pattern_family_atlas: 0.87
physical_path_atlas: 0.27
multi_family_case_bank: 0.42
state_factor_atlas: 0.36
path_cluster_mining: 0.12
trace_signature_validation: 0.47
readout_competition_trace: 0.78
stepwise_rollout_trace: 0.43
causal_closure: 0.18
general_language_mechanism_confidence: 0.66
```

总体判断：

```text
语言族覆盖明显改善；
物理路径图谱仍处于早期；
第一优先级仍是补全语言族路径；
闭合仍是第二优先级之后的终检目标。
```

### 阶段结论

Phase265 完成了 Phase264 之后最需要补的一块：

```text
把物理路径图谱从 output_protocol 局部样本，
扩展为九大语言模式族统一样本矩阵。
```

当前最重要进展：

```text
后续内部追踪不再需要临时构造样本；
可以直接使用 mode_family_case_bank_v3 和 path_schema_rows。
```

### 下一阶段任务

下一阶段仍属于 Phase264-300 大阶段，应作为 Phase266 推进：

```text
Phase266: multi-family baseline behavior and readout scan
```

任务目标：

```text
对 Phase265 的九大模式族样本库进行三模型基线测试，
记录行为结果、读出竞争、续写胜出、评分风险和初始 rollout 迹象。
```

建议方案：

```text
1. 使用 qwen3、GLM4、DS7B 顺序测试；
2. 优先每族抽取均衡子集，避免一次性过大；
3. 输出固定格式 behavior_rows、readout_rows、rollout_probe_rows、quality_calibration_rows；
4. 校准 Phase265 的 scoring_risk；
5. 找出每个模式族的主要失败路径。
```

阶段成功标准：

```text
1. 九大模式族都有三模型基线结果；
2. 每个模式族能初步定位 top failure path；
3. 可视化客户端能按 family/mode/variant 查看结果；
4. 不追闭合，只为后续内部路径追踪选择高价值样本。
```

## Phase 266: 多语言族基线行为与读出竞争扫描 [2026-07-08 06:31]

### 本阶段判断

Phase266 继续了 Phase265 的正确方向，而且属于同一大阶段：

```text
Phase264-265: 建立语言族物理路径样本与路径 schema（路径格式）
Phase266: 对这些样本做三模型行为层和读出层 baseline（基线）扫描
```

本阶段没有尝试闭合，也不应把结果解释为机制闭合。它完成的是更基础的一步：

```text
从“统一样本库已经存在”
推进到
“九大语言族在三模型上的行为表现、读出竞争、初始 rollout（生成展开）和风险校准已经有固定格式数据”。
```

附件中 Phase265 的判断基本正确：先完成语言族物理路径，再考虑闭合，是当前最稳妥的路线。继续单点 patch（补丁）或继续追局部闭合，会进入边际收益递减区；现在更需要把真实现象拼图铺开。

### 测试脚本和结果位置

脚本：

```text
tests/gpt5/phase266_multi_family_baseline_behavior_readout_scan.py
tests/gpt5/run_phase266_multi_family_baseline_behavior_readout_scan.sh
```

结果：

```text
tests/result/phase266_multi_family_baseline_behavior_readout_scan/multi_family_baseline_behavior_readout_scan/
```

同步到可视化客户端：

```text
tests/result/pattern_family_atlas/v1/phase266_behavior_rows.jsonl
tests/result/pattern_family_atlas/v1/phase266_readout_rows.jsonl
tests/result/pattern_family_atlas/v1/phase266_rollout_probe_rows.jsonl
tests/result/pattern_family_atlas/v1/phase266_quality_calibration_rows.jsonl
```

前端同步和构建已经通过：

```text
npm run sync:pattern-atlas
npm run build
```

客户端当前同步了 26 个 pattern atlas（模式图谱）文件。

### 测试原理

Phase266 使用 Phase265 的 `mode_family_case_bank_v3.jsonl`，从九大模式族中均衡抽样：

```text
每个 family（语言族）36 条；
每个 model（模型）324 条；
三个模型合计 972 条。
```

测试模型顺序：

```text
qwen3 -> GLM4 -> DS7B
```

每条样本记录四类数据：

```text
behavior_rows: 行为输出是否命中目标、是否符合输出模式；
readout_rows: 下一词元读出竞争；
rollout_probe_rows: 12 token（词元）短生成展开；
quality_calibration_rows: 根据真实输出和读出结果校准 scoring_risk（评分风险）。
```

核心观测公式：

$$
R(x)
=
\arg\max_{c \in C_{stop} \cup C_{continue} \cup C_{target}}
\text{logit}_c(x)
$$

其中：

```text
C_stop: 停止候选通道；
C_continue: 继续候选通道；
C_target: 目标答案候选通道。
```

停止-继续差值：

$$
\Delta_{stop,continue}(x)
=
\max_{c \in C_{stop}}\text{logit}_c(x)
-
\max_{c \in C_{continue}}\text{logit}_c(x)
$$

目标-胜出差值：

$$
\Delta_{target,winner}(x)
=
\max_{c \in C_{target}}\text{logit}_c(x)
-
\max_{c \in C_{all}}\text{logit}_c(x)
$$

风险校准规则是基础规则，不引入复杂统计：

```text
如果目标未命中，或 target_margin_vs_winner < -5，则 high；
如果模式未匹配，或 stop_continue_margin < -6，或有 drift marker（漂移标记），则 medium；
如果模型自然停止且模式匹配，则 low；
否则保留 Phase265 设计风险。
```

### 客观结果

总数据量：

```text
behavior_rows: 972
readout_rows: 972
rollout_rows: 972
quality_rows: 972
observation_rows: 972
metric_rows: 63
graph_edges: 86
missing_rows: 0
```

最重要的客观结果：

```text
competition_winner_counts:
continue = 972
```

也就是说，在本阶段的三模型、九语言族、972 条样本中，下一词元读出竞争全部由 continue（继续）通道胜出。这个结果不能证明“模型一定会继续输出”，但它强烈说明：

```text
语言族样本的首步物理入口不是 stop（停止）；
更稳定的入口是 continuation regime（继续状态）；
后续物理路径追踪应从继续通道分叉开始，而不是从终止符号开始。
```

Top continue channel（最高频继续通道）：

```text
continue_list_item: 334
continue_the: 330
continue_next_sentence: 172
continue_json_structure: 70
continue_format: 66
```

模型自然停止率：

```text
model_stop_rate: 0.041152
```

这说明短 rollout（生成展开）里自然停止很少。停止不是主导路径，继续态才是主导路径。

各语言族 answer_correct_proxy_rate（答案命中代理率）：

```text
closure: 0.685185
content_knowledge: 0.592593
cross_lingual: 0.833333
language_action: 0.944444
output_protocol: 0.925926
readout_competition: 0.722222
reasoning_constraint: 0.944444
state_drift: 0.925926
syntax_structure: 0.851852
```

各语言族 pattern_matched_proxy_rate（模式匹配代理率）：

```text
closure: 0.407407
content_knowledge: 0.166667
cross_lingual: 0.5
language_action: 0.5
output_protocol: 0.518519
readout_competition: 0.259259
reasoning_constraint: 0.462963
state_drift: 0.5
syntax_structure: 0.222222
```

这个结果非常关键：答案命中率整体高于模式匹配率，说明模型经常“知道答案”，但没有稳定执行目标输出模式。语言能力的物理路径不能只看语义答案，还要拆开：

```text
答案内容路径；
输出协议路径；
格式结构路径；
停止/继续路径。
```

各语言族 mean_stop_continue_margin（平均停止-继续差值）：

```text
closure: -5.988426
content_knowledge: -7.708912
cross_lingual: -7.296875
language_action: -10.900752
output_protocol: -8.175058
readout_competition: -6.095197
reasoning_constraint: -8.155382
state_drift: -8.385127
syntax_structure: -10.691551
```

所有语言族都是负值，说明 continue（继续）相对 stop（停止）有系统优势。其中：

```text
language_action: -10.900752
syntax_structure: -10.691551
```

继续态优势最强。后续追踪语言动作和语法结构时，不应先从停止机制入手，而应先看继续态如何组织结构展开。

风险校准：

```text
medium: 610
high: 322
low: 40
```

Phase265 的低/中风险设计在真实模型输出后被明显上调，说明样本库的评分难度低估了真实模式执行难度。这个校准是必要进展。

### 分模型现象

qwen3：

```text
behavior_rows: 324
competition_winner: continue 324
model_stop_rate: 0.0
top_continue_channel:
continue_the 136
continue_list_item 134
continue_format 38
continue_json_structure 12
continue_next_sentence 4
```

GLM4：

```text
behavior_rows: 324
competition_winner: continue 324
model_stop_rate: 0.030864
top_continue_channel:
continue_list_item 122
continue_next_sentence 116
continue_json_structure 58
continue_format 26
continue_the 2
```

DS7B：

```text
behavior_rows: 324
competition_winner: continue 324
model_stop_rate: 0.092593
top_continue_channel:
continue_the 192
continue_list_item 78
continue_next_sentence 52
continue_format 2
```

跨模型共同点：

```text
continue winner（继续胜出）稳定；
停止路径弱；
列表、普通英文延续、下一句、JSON 结构、格式延续构成主要继续通道。
```

跨模型差异：

```text
qwen3 更偏 continue_the 和 continue_list_item；
GLM4 更偏 continue_list_item、continue_next_sentence、continue_json_structure；
DS7B 更偏 continue_the。
```

这说明“继续态主导”可能是跨模型稳定现象，但继续态内部的通道分叉有明显模型差异。由于当前模型都是小模型，内部编码机制可能比大模型粗糙，差异中可能有 30%-50% 是模型规模和训练数据带来的偏差，不能直接上升为通用语言机制。

### 已完成的语言模式图谱内容

到 Phase266，已经完成：

```text
1. 九大语言族统一 case bank（样本库）；
2. 每族 144 条设计样本；
3. 每族 36 条三模型基线扫描；
4. 行为输出、读出竞争、短 rollout、风险校准四类固定格式数据；
5. family -> top_continue_channel 的第一版 failure path（失败路径）边；
6. 可视化客户端数据同步。
```

当前图谱进度：

```text
pattern_family_atlas: 0.88
physical_path_atlas: 0.30
multi_family_case_bank: 0.45
multi_family_baseline_scan: 0.16
state_factor_atlas: 0.37
path_cluster_mining: 0.14
trace_signature_validation: 0.48
readout_competition_trace: 0.79
stepwise_rollout_trace: 0.44
causal_closure: 0.18
general_language_mechanism_confidence: 0.67
```

总体进度评估：

```text
语言模式图谱整体约 35%-40%；
语言族物理路径约 30%；
闭合约 18%。
```

这个比例说明当前第一优先级仍然不是闭合，而是补语言族物理路径。

### 问题、硬伤和瓶颈

1. answer_correct_proxy（答案正确代理）和 pattern_matched_proxy（模式匹配代理）仍是字符串规则，不是真正语义判分。

2. 每族只跑 36 条，是均衡 baseline（基线）而不是全量 1296 条穷尽测试。

3. rollout 只有 12 token（词元），只能观察初始展开，不能观察长程漂移和长程停止。

4. readout（读出）只看下一词元竞争，还没有记录层路径、注意力头、MLP channel（多层感知机通道）和 residual stream（残差流）中的具体物理传播。

5. continue（继续）通道全部胜出是强现象，但这可能部分来自 prompt（提示词）结构、模板格式和小模型输出习惯，不能直接视为语言机制闭合。

6. syntax_structure（语法结构）答案命中高但模式匹配低，说明当前代理评分对语法样本可能过粗，需要更细的结构判分。

7. content_knowledge（知识内容）模式匹配最低，可能是模板目标过短，也可能是知识输出天然倾向解释扩展，需要后续拆成事实召回、实体关系、定义解释、条件知识几个子族。

8. 当前没有 causal intervention（因果干预），因此只能说“路径候选”，不能说“机制已证明”。

### 智能理论角度的进展

Phase266 支持一个更清楚的分解：

$$
Language(x)
\neq
Answer(x)
$$

更合理的机制图是：

$$
Language(x)
=
Content(x)
\oplus
Protocol(x)
\oplus
Structure(x)
\oplus
Continuation(x)
\oplus
Stop(x)
$$

其中本阶段最强证据落在：

$$
Continuation(x) > Stop(x)
$$

并且：

$$
AnswerHit(x) > PatternMatch(x)
$$

这意味着语言智能不是单纯“生成正确语义”，而是多个模式系统同时竞争：

```text
知识网络负责内容；
推理能力负责约束和步骤；
语法系统负责结构；
输出协议负责格式；
继续/停止系统负责生成边界；
读出竞争把这些路径投影到下一词元。
```

破解语言编码机制的第一性原理应继续保持：

```text
不要先找一个闭合公式；
先找全局物理路径图谱；
当路径图谱足够完整后，公式应从图谱中自然浮现。
```

### 阶段结论

Phase266 完成了 Phase265 之后必须做的三模型基线扫描。它证明的不是闭合，而是：

```text
九大语言族在首步读出层面都被 continuation regime（继续状态）主导；
答案命中和模式执行明显分离；
后续物理路径追踪应从 continue channel（继续通道）分叉开始。
```

当前最值得追踪的物理路径入口：

```text
continue_list_item
continue_the
continue_next_sentence
continue_json_structure
continue_format
```

### 下一阶段任务

下一阶段仍属于当前大阶段，应作为 Phase267 推进：

```text
Phase267: multi-family continuation channel physical path tracing
```

目标：

```text
从 Phase266 的 top failure path 中选择高价值样本，
追踪 continue channel 在不同语言族中的层路径、组件路径和状态变化。
```

建议任务：

```text
1. 每个语言族选择 3-5 条高风险样本；
2. 对 qwen3、GLM4、DS7B 分别记录 layer-wise readout（逐层读出）；
3. 记录 residual stream（残差流）、attention output（注意力输出）、MLP output（多层感知机输出）对 continue channel 的贡献；
4. 对 continue_list_item、continue_the、continue_next_sentence、continue_json_structure、continue_format 分别建路径签名；
5. 输出固定格式 physical_path_rows、component_contribution_rows、channel_trace_rows；
6. 只在路径稳定后再考虑局部因果干预。
```

阶段成功标准：

```text
1. 每个语言族至少得到一个可视化的 continue physical path（继续物理路径）；
2. 能区分“答案内容路径”和“输出模式路径”；
3. 能判断 top_continue_channel 是在哪些层开始稳定胜出的；
4. 不追闭合，只补语言族物理路径图谱。
```

## Phase 267: 多语言族继续通道逐层物理路径追踪 [2026-07-08 07:08]

### 本阶段判断

Phase267 继续 Phase264-300 的同一大阶段，方向正确：

```text
第一优先级：完成语言族物理路径；
第二优先级：在物理路径稳定后再尝试闭合。
```

附件对 Phase266 的分析基本正确。Phase266 证明了九大语言族的首步读出入口几乎全部由 continuation regime（继续机制）主导，但它还没有追踪内部层路径。Phase267 补上的正是这一块：

```text
从 behavior/readout baseline（行为/读出基线）
推进到
layer-wise residual readout trace（逐层残差读出追踪）。
```

本阶段仍不是闭合验证，也不是因果干预。它是物理路径图谱的第一版逐层 tracing（追踪）。

### 脚本和结果位置

脚本：

```text
tests/gpt5/phase267_multifamily_continuation_physical_path_trace.py
tests/gpt5/run_phase267_multifamily_continuation_physical_path_trace.sh
```

结果：

```text
tests/result/phase267_multifamily_continuation_physical_path_trace/multifamily_continuation_physical_path_trace/
```

同步到可视化客户端：

```text
tests/result/pattern_family_atlas/v1/phase267_physical_path_rows.jsonl
tests/result/pattern_family_atlas/v1/phase267_layerwise_readout_rows.jsonl
tests/result/pattern_family_atlas/v1/phase267_component_contribution_rows.jsonl
tests/result/pattern_family_atlas/v1/phase267_continue_channel_trace_rows.jsonl
tests/result/pattern_family_atlas/v1/phase267_family_path_signature_rows.jsonl
```

前端同步和构建已经通过：

```text
npm run sync:pattern-atlas
npm run build
```

客户端当前同步了 31 个 pattern atlas（模式图谱）文件。

### 测试原理

Phase267 从 Phase266 的结果中，每个模型、每个语言族选择 3 条高价值样本：

```text
优先 high / medium risk（高/中风险）；
优先答案命中但模式失败；
优先 stop_continue_margin（停止-继续差值）很负；
优先 Phase266 中出现 top continue channel（最高继续通道）的样本。
```

样本量：

```text
9 个语言族 × 3 条 × 3 个模型 = 81 条 physical path（物理路径）
```

对每条样本，记录所有 hidden_states（隐藏状态），把每一层的 residual state（残差状态）投影到 lm_head（输出头），计算 stop / continue / target 的读出竞争。

逐层继续优势公式：

$$
M_{\mathrm{continue}}^{(l)}
=
R_{\mathrm{continue}}^{(l)}
-
R_{\mathrm{stop}}^{(l)}
$$

其中：

```text
R_continue^(l): 第 l 层继续通道最高 logit；
R_stop^(l): 第 l 层停止通道最高 logit。
```

稳定层定义：

$$
L_{\mathrm{stable}}
=
\min l
\quad
\text{s.t.}
\quad
M_{\mathrm{continue}}^{(l)},
M_{\mathrm{continue}}^{(l+1)},
M_{\mathrm{continue}}^{(l+2)}
> 0
$$

层级 residual delta（残差层增量）：

$$
\Delta M_{\mathrm{continue}}^{(l)}
=
M_{\mathrm{continue}}^{(l)}
-
M_{\mathrm{continue}}^{(l-1)}
$$

需要注意：本阶段的 component_contribution_rows（组件贡献行）是 layer-level residual delta（层级残差增量），还没有拆成 attention（注意力）和 MLP（多层感知机）的独立贡献。

### 客观结果

总输出：

```text
physical_path_rows: 81
layerwise_readout_rows: 2889
component_contribution_rows: 2808
continue_channel_trace_rows: 81
family_path_signature_rows: 81
observation_rows: 81
metric_rows: 31
graph_edges: 32
missing_rows: 0
```

九大语言族覆盖：

```text
closure: 9
content_knowledge: 9
cross_lingual: 9
language_action: 9
output_protocol: 9
readout_competition: 9
reasoning_constraint: 9
state_drift: 9
syntax_structure: 9
```

本阶段追踪到的 continue channel（继续通道）分布：

```text
continue_json_structure: 29
continue_next_sentence: 21
continue_format: 20
continue_the: 8
continue_list_item: 3
```

最终层竞争：

```text
continue: 78
target: 3
```

这说明，在高风险样本子集中，continue（继续）仍然是最终层主导路径；但 GLM4 有 3 条样本 target（目标答案）强于 continue。这个结果不推翻 Phase266，因为 Phase266 的 competition_winner 主要是 stop / continue 竞争，本阶段额外把 target 纳入了最终竞争。

关键层指标：

```text
mean_first_continue_win_layer: 0
mean_stable_continue_from_layer: 0.296296
mean_final_continue_stop_margin: 8.247106
```

逐层稳定分布：

```text
first_continue_win_layer:
L0 = 81 / 81

stable_continue_from_layer:
L0 = 75 / 81
L2 = 3 / 81
L6 = 3 / 81
```

这个结果非常强，但必须谨慎解释：

```text
L0 读出可解码出 continue 优势，
不等于 L0 已经因果地产生完整继续机制。
```

更稳妥的解释是：

```text
continuation bias（继续偏置）在输入嵌入/词元先验层面已经非常强，
后续层更多是在重塑、放大或改写这个继续路径，
而不是从零产生继续状态。
```

### 分模型结果

qwen3：

```text
physical_path_rows: 27
layerwise_readout_rows: 999
component_contribution_rows: 972
final_winner_counts: continue 27
channel_counts:
continue_format 16
continue_json_structure 8
continue_the 3
mean_stable_continue_from_layer: 0.666667
mean_final_continue_stop_margin: 10.412037
```

GLM4：

```text
physical_path_rows: 27
layerwise_readout_rows: 1107
component_contribution_rows: 1080
final_winner_counts:
continue 24
target 3
channel_counts:
continue_json_structure 21
continue_format 4
continue_next_sentence 2
mean_stable_continue_from_layer: 0.222222
mean_final_continue_stop_margin: 4.688079
```

DS7B：

```text
physical_path_rows: 27
layerwise_readout_rows: 783
component_contribution_rows: 756
final_winner_counts: continue 27
channel_counts:
continue_next_sentence 19
continue_the 5
continue_list_item 3
mean_stable_continue_from_layer: 0
mean_final_continue_stop_margin: 9.641204
```

跨模型共同点：

```text
continue 路径极早可读出；
最终层多数仍由 continue 胜出；
九大语言族都有可视化逐层路径；
高风险样本中结构化继续通道占比更高。
```

跨模型差异：

```text
qwen3 高风险路径偏 continue_format / continue_json_structure；
GLM4 高风险路径强烈偏 continue_json_structure；
DS7B 高风险路径偏 continue_next_sentence。
```

这说明 continue regime（继续机制）是跨模型稳定现象，但具体通道分叉明显受模型架构、词表和训练分布影响。当前模型较小，必须继续保留 30%-50% 的偏差空间。

### 各语言族物理路径观察

按 family（语言族）汇总的最终 continue-stop margin（继续-停止差值）：

```text
closure: 7.388889
content_knowledge: 8.34375
cross_lingual: 7.958333
language_action: 7.53125
output_protocol: 8.034722
readout_competition: 7.887153
reasoning_constraint: 6.875
state_drift: 8.416667
syntax_structure: 11.788194
```

syntax_structure（语法结构）最高：

```text
syntax_structure final_continue_stop_margin = 11.788194
```

这支持 Phase266 的判断：语法结构不是静态规则外壳，而是强烈依赖继续展开的生成控制路径。

reasoning_constraint（推理约束）最低：

```text
reasoning_constraint final_continue_stop_margin = 6.875
```

但它仍然是正值，说明推理约束也没有转向 stop-first（停止优先），只是继续优势相对弱一些。

### 进展

Phase267 完成了当前阶段最需要的一步：

```text
把 Phase266 的 continue winner（继续胜出）
进一步拆成逐层 physical path（物理路径）。
```

已经完成的拼图：

```text
1. 九大语言族每族都有三模型逐层路径；
2. continue 路径在 L0 就可读出；
3. 75/81 样本从 L0 开始连续稳定；
4. 结构化继续通道成为高风险路径主入口；
5. residual layer delta（残差层增量）已经有第一版记录；
6. 可视化客户端已经能读取 Phase267 路径数据。
```

### 问题和硬伤

1. L0 可读出 continue 优势，可能包含词元频率、词表先验和 prompt 模板偏置，不能直接解释为完整内部机制。

2. 本阶段没有拆分 attention output（注意力输出）和 MLP output（多层感知机输出），component_contribution_rows 只是残差层级差分。

3. 每族每模型只有 3 条高风险样本，适合追路径，不适合估计总体分布。

4. 样本是从 Phase266 高风险子集中选出的，因此 channel_counts 会偏向结构化失败路径，不能代表全量语言族分布。

5. 逐层投影是 readout probing（读出探针），不是 causal intervention（因果干预）。它能说明“某层可读出什么”，不能直接说明“某层导致什么”。

6. 当前使用小模型，继续路径很可能比大模型更粗糙，尤其是 EOS（结束符）和结构化输出控制可能偏弱。

7. GLM4 的 target 胜出样本说明 target/content path（目标/内容路径）在部分情况下能压过 continue path（继续路径），后续必须拆分内容路径和继续路径的相互作用。

### 理论更新

Phase267 不需要改理论名词，但可以改进机制图：

$$
\mathcal{M}(P_i)
=
(
\mathcal{L}_i,
\mathcal{C}_i,
\mathcal{S}_i,
\mathcal{R}_i,
\mathcal{G}_i
)
$$

本阶段补强的是：

```text
L_i: 逐层路径开始有数据；
R_i: 继续/停止/目标读出竞争更完整；
S_i: continuation state（继续状态）在早层已可读出；
C_i: 只有 residual delta（残差增量），还未拆组件；
G_i: 本阶段没有扩展长程生成轨迹。
```

更具体的当前公式：

$$
LanguagePath_i
=
ContentPath_i
\oplus
ProtocolPath_i
\oplus
StructurePath_i
\oplus
ContinuationPath_i
\oplus
StopPath_i
$$

其中 Phase267 支持：

$$
ContinuationPath_i^{(L0)}
>
StopPath_i^{(L0)}
$$

但这应解释为“早层可读出优势”，不是完整因果生成机制。

### 当前图谱进度

```text
pattern_family_atlas: 0.89
physical_path_atlas: 0.34
multi_family_case_bank: 0.45
multi_family_baseline_scan: 0.18
state_factor_atlas: 0.38
path_cluster_mining: 0.16
trace_signature_validation: 0.50
readout_competition_trace: 0.80
stepwise_rollout_trace: 0.44
causal_closure: 0.18
general_language_mechanism_confidence: 0.68
```

总体评估：

```text
语言模式图谱整体约 38%-42%；
语言族物理路径约 34%；
闭合约 18%。
```

第一优先级仍然是补全语言族物理路径，特别是 attention / MLP 组件级路径。

### 阶段结论

Phase267 的结论：

```text
继续机制不是只在最终输出层突然出现；
在高风险语言族样本中，continue path 通常从极早层就可读出，并在多数样本中保持稳定。
```

最重要的新拼图：

```text
语言族物理路径的入口不是“后期闭合失败”，
而是“早层 continuation bias + 后续层结构化重塑”。
```

这解释了为什么单点闭合 patch（补丁）反复失败：

```text
如果继续路径从早层就已占优，
只在末端压 EOS 或修正完成向量，
通常无法稳定改变整个生成动力学。
```

### 下一阶段任务

下一阶段仍属于当前大阶段，应作为 Phase268：

```text
Phase268: attention / MLP separated continuation path attribution
```

目标：

```text
把 Phase267 的 residual layer delta（残差层增量）
拆成 attention output（注意力输出）和 MLP output（多层感知机输出）的贡献。
```

建议：

```text
1. 只选 Phase267 中最稳定的 18 条样本，避免显存和数据噪声；
2. 每个模型每个主要 channel 至少 1-2 条；
3. hook 每层 attention output 和 MLP output；
4. 分别计算加入/移除组件输出后 continue_stop_margin 的变化；
5. 输出 component_physical_path_rows、attention_contribution_rows、mlp_contribution_rows；
6. 继续不做闭合，只定位组件级物理路径。
```

成功标准：

```text
1. 能判断 continue path 主要由 attention、MLP 还是 residual accumulation 推动；
2. 能为九大语言族至少给出粗粒度组件路径；
3. 能识别不同模型的组件分叉差异；
4. 为后续少量因果干预选择组件目标。
```

## Phase 268: 注意力/MLP 分离的继续路径组件归因 [2026-07-08 16:30]

### 本阶段判断

Phase268 继续 Phase264-300 大阶段，仍属于“语言模式图谱物理路径”优先级，不属于闭合阶段。

附件对 Phase267 的判断基本正确：

```text
Phase267 已经证明 continue path（继续路径）在高风险样本中极早可读出；
但 residual delta（残差增量）还没有拆成 attention（注意力）和 MLP（多层感知机）贡献；
因此 Phase268 应补组件级路径，而不是立刻做闭合。
```

本阶段完成的是：

```text
layer_input
-> layer_input + attention_out
-> layer_input + attention_out + MLP_out
-> layer_out
```

的观测分解，计算每一步对 continue-stop margin（继续-停止差值）的影响。

### 脚本和结果位置

脚本：

```text
tests/gpt5/phase268_attention_mlp_continuation_path_attribution.py
tests/gpt5/run_phase268_attention_mlp_continuation_path_attribution.sh
```

结果：

```text
tests/result/phase268_attention_mlp_continuation_path_attribution/attention_mlp_continuation_path_attribution/
```

同步到可视化客户端：

```text
tests/result/pattern_family_atlas/v1/phase268_component_physical_path_rows.jsonl
tests/result/pattern_family_atlas/v1/phase268_attention_contribution_rows.jsonl
tests/result/pattern_family_atlas/v1/phase268_mlp_contribution_rows.jsonl
tests/result/pattern_family_atlas/v1/phase268_residual_accumulation_rows.jsonl
tests/result/pattern_family_atlas/v1/phase268_component_summary_rows.jsonl
```

前端同步和构建已通过：

```text
npm run sync:pattern-atlas
npm run build
```

客户端当前同步了 36 个 pattern atlas（模式图谱）文件。

### 测试原理

Phase268 从 Phase267 的稳定高价值样本中选择：

```text
每个模型 6 条；
三个模型合计 18 条；
覆盖 continue_json_structure、continue_format、continue_next_sentence、continue_the、continue_list_item；
覆盖 syntax_structure、content_knowledge、output_protocol、state_drift、reasoning_constraint、readout_competition 等核心语言族。
```

每层捕获：

```text
h_l: layer_input（层输入残差）
a_l: attention_out（注意力输出）
m_l: MLP_out（多层感知机输出）
o_l: layer_out（层输出）
```

逐层继续优势：

$$
M(h)
=
R_{\mathrm{continue}}(h)
-
R_{\mathrm{stop}}(h)
$$

注意力贡献：

$$
\Delta M_{\mathrm{attn}}^{(l)}
=
M(h_l + a_l)
-
M(h_l)
$$

MLP 贡献：

$$
\Delta M_{\mathrm{mlp}}^{(l)}
=
M(h_l + a_l + m_l)
-
M(h_l + a_l)
$$

残差 carry（残差携带/结构差）：

$$
\Delta M_{\mathrm{resid}}^{(l)}
=
M(o_l)
-
M(h_l + a_l + m_l)
$$

注意：这是 observational attribution（观测归因），不是 causal ablation（因果消融）。它说明自然前向中哪类组件更强地增加 continue-stop margin，但不能单独证明组件因果必要性。

### 客观结果

总输出：

```text
component_physical_path_rows: 624
attention_contribution_rows: 624
mlp_contribution_rows: 624
residual_accumulation_rows: 624
component_summary_rows: 18
observation_rows: 18
metric_rows: 18
graph_edges: 18
missing_rows: 0
```

语言族覆盖：

```text
syntax_structure: 3
state_drift: 2
language_action: 1
reasoning_constraint: 2
content_knowledge: 3
output_protocol: 3
readout_competition: 2
closure: 1
cross_lingual: 1
```

继续通道覆盖：

```text
continue_json_structure: 6
continue_format: 4
continue_next_sentence: 4
continue_the: 3
continue_list_item: 1
```

最终层结果：

```text
final_winner_counts:
continue = 18
```

组件主导结果：

```text
dominant_positive_component_counts:
mlp = 18
```

平均正向贡献：

```text
mean_sum_positive_attn_delta: 9.446689
mean_sum_positive_mlp_delta: 26.821832
mean_sum_positive_residual_delta: 0.207329
mean_final_continue_stop_margin: 9.039062
```

这是一条强结果：

```text
在本阶段 18 条高价值继续路径样本中，
MLP 对 continue-stop margin 的正向增强显著大于 attention；
attention 也有明显正贡献；
residual carry 贡献很小。
```

但必须限制解释：

```text
MLP 是最大自然正向写入/增强者；
还不能说 MLP 是唯一因果必要组件。
```

### 分模型结果

qwen3：

```text
selected_cases: 6
dominant_positive_component: mlp 6/6
mean_final_continue_stop_margin: 11.166667
mean_sum_positive_attn_delta: 11.328186
mean_sum_positive_mlp_delta: 30.655925
mean_sum_positive_residual_delta: 0.291911
strongest_attn_layers: L25(4), L15(1), L9(1)
strongest_mlp_layers: L34(5), L35(1)
```

GLM4：

```text
selected_cases: 6
dominant_positive_component: mlp 6/6
mean_final_continue_stop_margin: 5.648438
mean_sum_positive_attn_delta: 8.131673
mean_sum_positive_mlp_delta: 25.885092
mean_sum_positive_residual_delta: 0.151692
strongest_attn_layers: L36(4), L22(1), L33(1)
strongest_mlp_layers: L38(4), L22(1), L39(1)
```

DS7B：

```text
selected_cases: 6
dominant_positive_component: mlp 6/6
mean_final_continue_stop_margin: 10.302083
mean_sum_positive_attn_delta: 8.880208
mean_sum_positive_mlp_delta: 23.924479
mean_sum_positive_residual_delta: 0.178385
strongest_attn_layers: L9(3), L0(2), L12(1)
strongest_mlp_layers: L27(4), L26(1), L25(1)
```

跨模型共同点：

```text
MLP 是 18/18 样本的最大正向增强组件；
最终层 continue 仍为 18/18；
attention 有稳定正贡献，但弱于 MLP；
residual carry 很小。
```

跨模型差异：

```text
qwen3 的 strongest MLP 多在 L34/L35；
GLM4 的 strongest MLP 多在 L38/L39；
DS7B 的 strongest MLP 多在 L25-L27；
attention 的强层分布更分散。
```

这说明继续路径的组件图谱出现一个新模式：

```text
attention 更像路由/上下文搬运；
MLP 更像 continue advantage（继续优势）的主写入/放大器；
residual stream 负责携带，但本阶段观测到的额外 carry 增量很小。
```

### 核心拼图进展

Phase268 补上了 Phase267 最大缺口：

```text
从 residual layer delta（残差层增量）
推进到
attention / MLP separated attribution（注意力 / MLP 分离归因）。
```

当前新增拼图：

```text
1. 继续路径不是只有残差层级曲线；
2. MLP 在高价值继续路径中是最强正向增强组件；
3. attention 有稳定贡献，但多数低于 MLP；
4. 继续路径的强 MLP 层多在模型后段；
5. 结构化继续通道和语法/协议路径具有明显 MLP 写入特征；
6. 三模型都显示同一方向，但层号和通道分叉模型特异。
```

这使当前语言模式物理路径图谱变成：

```text
早层 continuation bias（继续偏置）
-> 中后层 attention 路由和上下文搬运
-> 后段 MLP 写入/放大 continue advantage
-> 最终读出层 continue / target / stop 竞争
```

### 问题和硬伤

1. 本阶段仍是 observational attribution（观测归因），不是 causal intervention（因果干预）。

2. 样本只有 18 条，适合定位组件路径，不适合估计全量分布。

3. 样本来自 Phase267 高风险稳定样本，因此偏向继续路径强、结构化风险高的样本。

4. `h_l + attention_out + MLP_out` 是近似分解。不同架构存在 RMSNorm、残差缩放、并行结构、post-norm / pre-norm 差异，可能带来测量偏差。

5. MLP 正向贡献大，不等于 MLP 是唯一原因。attention 可能提供了 MLP 所需上下文，二者存在依赖关系。

6. 当前没有对 MLP 做消融、替换、抑制或跨样本 patch，因此不能证明必要性。

7. 小模型可能更依赖 MLP 进行格式和继续写入，大模型中 attention / MLP 分工可能更复杂，仍需保留 30%-50% 外推偏差。

### 理论更新

不改理论名词，但机制公式可以更具体：

$$
M_{\mathrm{continue}}^{(l)}
=
B_{\mathrm{embed}}
+
\sum_{j=1}^{l}
\left(
\Delta M_{\mathrm{attn}}^{(j)}
+
\Delta M_{\mathrm{mlp}}^{(j)}
+
\Delta M_{\mathrm{resid}}^{(j)}
\right)
$$

Phase268 的经验关系：

$$
\sum \Delta M_{\mathrm{mlp}}^{+}
>
\sum \Delta M_{\mathrm{attn}}^{+}
>>
\sum \Delta M_{\mathrm{resid}}^{+}
$$

对应当前结果：

```text
MLP positive sum: 26.821832
Attention positive sum: 9.446689
Residual positive sum: 0.207329
```

智能理论中的语言路径可以进一步写成：

$$
\mathrm{LanguagePath}
=
B_{\mathrm{embed}}
\oplus
\mathrm{AttentionRoute}
\oplus
\mathrm{MLPWrite}
\oplus
\mathrm{ReadoutCompetition}
\oplus
\mathrm{Rollout}
$$

其中 Phase268 强化的是：

```text
MLPWrite 是 continue path 的主要正向增强环节。
```

### 当前图谱进度

```text
pattern_family_atlas: 0.90
physical_path_atlas: 0.37
multi_family_case_bank: 0.45
multi_family_baseline_scan: 0.18
state_factor_atlas: 0.39
path_cluster_mining: 0.17
trace_signature_validation: 0.52
readout_competition_trace: 0.80
component_path_atlas: 0.16
stepwise_rollout_trace: 0.44
causal_closure: 0.18
general_language_mechanism_confidence: 0.69
```

总体评估：

```text
语言模式图谱整体约 40%-43%；
语言族物理路径约 37%；
组件路径图谱约 16%；
闭合约 18%。
```

第一优先级仍是补物理路径。闭合还不应前置。

### 阶段结论

Phase268 的阶段结论：

```text
在高价值继续路径样本中，
MLP 是 continuation path（继续路径）最强的自然正向增强组件；
attention 提供稳定但较弱的正向贡献；
residual carry 的额外增量很小。
```

这说明语言模式图谱的关键瓶颈已经从：

```text
继续路径在哪里出现？
```

推进到：

```text
哪些 MLP 层在写入或放大继续路径？
这些 MLP 层是否因果必要？
它们写入的是通用继续偏置，还是具体结构/协议通道？
```

### 下一阶段任务

下一阶段仍属于当前大阶段，应作为 Phase269：

```text
Phase269: MLP continuation writer necessity audit
```

目标：

```text
对 Phase268 定位出的 strongest MLP layers 做小规模因果必要性审计。
```

建议：

```text
1. 每个模型选 2 条样本；
2. 优先 qwen3 L34/L35、GLM4 L38/L39、DS7B L25-L27；
3. 对 strongest MLP output 做 suppression（抑制）、zero ablation（置零）或 mean replacement（均值替换）；
4. 观察 continue_stop_margin、target_margin、生成前 8 token 是否变化；
5. 输出 mlp_necessity_rows、causal_effect_rows、rollout_effect_rows；
6. 只做小规模因果审计，不宣称闭合。
```

成功标准：

```text
1. 判断 strongest MLP 是否对 continue path 必要；
2. 区分“MLP 写入继续通道”和“MLP 只是相关放大”；
3. 找到后续大规模组件图谱的最小因果入口；
4. 继续保持闭合后置。
```

## Phase 269: MLP 继续路径写入器必要性审计 [2026-07-08 17:05]

### 本阶段判断

Phase269 继续 Phase264-300 大阶段，仍属于“语言模式图谱物理路径”优先级。

附件对 Phase268 的判断基本正确：

```text
Phase268 证明 MLP 在高价值 continuation path（继续路径）样本中是最强自然正向增强组件；
但它仍是 observational attribution（观测归因）；
必须通过小规模 causal necessity audit（因果必要性审计）检验 strongest MLP 是否真的必要。
```

本阶段完成的不是闭合，而是第一版小规模 MLP necessity audit（MLP 必要性审计）。

### 脚本和结果位置

脚本：

```text
tests/gpt5/phase269_mlp_continuation_writer_necessity_audit.py
tests/gpt5/run_phase269_mlp_continuation_writer_necessity_audit.sh
```

结果：

```text
tests/result/phase269_mlp_continuation_writer_necessity_audit/mlp_continuation_writer_necessity_audit/
```

同步到可视化客户端：

```text
tests/result/pattern_family_atlas/v1/phase269_mlp_necessity_rows.jsonl
tests/result/pattern_family_atlas/v1/phase269_causal_effect_rows.jsonl
tests/result/pattern_family_atlas/v1/phase269_rollout_effect_rows.jsonl
```

前端同步和构建已通过：

```text
npm run sync:pattern-atlas
npm run build
```

客户端当前同步了 39 个 pattern atlas（模式图谱）文件。

### 测试原理

从 Phase268 的 strongest MLP layers（最强 MLP 层）中选样：

```text
qwen3: 2 条
GLM4: 2 条
DS7B: 2 条
总计 6 条样本
```

选择的核心层：

```text
qwen3: L35 / L34
GLM4: L38
DS7B: L25 / L26
```

对每条样本做两个干预：

```text
mlp_zero_last_token: 把该层 MLP output 在最后 token 位置置零；
mlp_half_last_token: 把该层 MLP output 在最后 token 位置缩放到 0.5。
```

核心指标：

$$
\Delta M_{\mathrm{continue}}
=
M_{\mathrm{patched}}
-
M_{\mathrm{base}}
$$

其中：

$$
M
=
R_{\mathrm{continue}}
-
R_{\mathrm{stop}}
$$

必要性支持规则：

```text
如果 delta_continue_stop_margin < -1.0，
或 winner 从 continue 翻转为 stop/target，
则记为 necessity_supported = True。
```

另外记录 8 token（8 词元）短 rollout（生成展开），检查输出是否改变。

### 客观结果

总输出：

```text
mlp_necessity_rows: 12
causal_effect_rows: 12
rollout_effect_rows: 12
observation_rows: 12
metric_rows: 6
graph_edges: 12
missing_rows: 0
```

干预分布：

```text
mlp_zero_last_token: 6
mlp_half_last_token: 6
```

必要性结果：

```text
necessity_supported:
True = 8
False = 4
```

winner 翻转：

```text
winner_changed:
True = 2
False = 10
```

rollout 改变：

```text
rollout_changed:
True = 4
False = 8
```

平均效果：

```text
mean_delta_continue_stop_margin: -2.873698
mean_delta_target_logit: -0.858337
```

总体结论：

```text
MLP 抑制总体削弱 continue-stop margin；
但效果不是跨模型一致；
qwen3 和 DS7B 支持 MLP 必要性；
GLM4 在本轮两条样本上不支持，甚至出现 continue margin 上升。
```

### 分模型结果

qwen3：

```text
selected_cases: 2
necessity_supported: 4 / 4
winner_changed: 1 / 4
mean_delta_continue_stop_margin: -6.535156
mean_delta_target_logit: -2.841064
rollout_changed: 1 / 4
```

关键样本：

```text
syntax_structure L35:
zero: 16.75 -> 10.765625, delta = -5.984375
half: 16.75 -> 13.78125, delta = -2.96875

reasoning_constraint L34:
zero: 10.4375 -> -3.125, delta = -13.5625, continue -> stop
half: 10.4375 -> 6.8125, delta = -3.625
```

qwen3 中 strongest MLP 对 continue path 具有强必要性迹象，尤其 reasoning_constraint（推理约束）出现 winner 翻转。

GLM4：

```text
selected_cases: 2
necessity_supported: 0 / 4
winner_changed: 0 / 4
mean_delta_continue_stop_margin: +1.5
mean_delta_target_logit: +0.055115
rollout_changed: 2 / 4
```

关键样本：

```text
output_protocol L38:
zero: 4.5625 -> 6.5, delta = +1.9375
half: 4.5625 -> 5.3125, delta = +0.75

content_knowledge L38:
zero: 5.0625 -> 7.4375, delta = +2.375
half: 5.0625 -> 6.0, delta = +0.9375
```

这是一个重要负/校准结果。它说明：

```text
GLM4 的 Phase268 MLP 正向贡献大，
但直接抑制该层 MLP 不一定削弱最终 continue margin；
该层可能同时写入 continue 和 stop/target 相关成分，
或者后续层存在补偿。
```

DS7B：

```text
selected_cases: 2
necessity_supported: 4 / 4
winner_changed: 1 / 4
mean_delta_continue_stop_margin: -3.585938
mean_delta_target_logit: +0.210938
rollout_changed: 1 / 4
```

关键样本：

```text
content_knowledge L25:
zero: 11.34375 -> 7.75, delta = -3.59375
half: 11.34375 -> 10.21875, delta = -1.125

reasoning_constraint L26:
zero: 7.125 -> -0.25, delta = -7.375, continue -> stop
half: 7.125 -> 4.875, delta = -2.25
```

DS7B 中 strongest MLP 也具有必要性迹象，尤其 reasoning_constraint 出现 winner 翻转。

### 进展

Phase269 完成了 Phase268 之后必须补的一块：

```text
从“MLP 是最大自然正向增强组件”
推进到
“MLP 在 qwen3 和 DS7B 中具有小规模因果必要性迹象，但 GLM4 不一致”。
```

这是比单纯正结果更有价值的校准：

```text
MLPWrite 是重要候选；
但 MLPWrite 不是单一、跨模型、无条件必要机制；
必须进入模型特异和路径特异的组件图谱。
```

当前核心拼图更新：

```text
1. qwen3/DS7B: 后段 MLP 抑制会显著削弱 continue path；
2. qwen3/DS7B: reasoning_constraint 样本出现 continue -> stop 翻转；
3. GLM4: strongest MLP 抑制不削弱 continue，反而增强；
4. MLP 自然正向贡献和 MLP 因果必要性不能等同；
5. 组件路径必须区分模型、语言族、通道和后续补偿。
```

### 问题和硬伤

1. 样本只有 6 条，干预 12 次，仍是小规模 necessity pilot（必要性试点）。

2. 只干预最后 token 的 MLP output，不能代表整个生成轨迹中的 MLP 作用。

3. 干预方式是 scale/zero，可能改变分布自然性，尤其对 GLM4 的补偿效应解释仍不确定。

4. 没有做 mean replacement（均值替换）、random same norm（同范数随机替换）或跨样本替换，无法排除范数/分布偏移问题。

5. rollout 只有 8 token，不能判断长程结构和停止效果。

6. GLM4 负结果说明当前线性组件公式仍不足以模拟真实运行机制。单层 MLP output 不是完整机制。

7. 当前模型为小模型，组件分工可能粗糙，对大模型外推仍需保留 30%-50% 偏差。

### 理论更新

Phase269 不改理论名词，但要修正 Phase268 的过强解释。

Phase268 支持：

$$
\sum \Delta M_{\mathrm{mlp}}^{+}
>
\sum \Delta M_{\mathrm{attn}}^{+}
$$

Phase269 进一步说明：

$$
\mathrm{ObservedWrite}_{\mathrm{MLP}}
\nRightarrow
\mathrm{CausalNecessary}_{\mathrm{MLP}}
$$

更准确公式：

$$
\mathrm{ContinuePath}
=
B_{\mathrm{embed}}
\oplus
\mathrm{AttentionRoute}
\oplus
\mathrm{MLPWrite}
\oplus
\mathrm{CompensationPath}
\oplus
\mathrm{ReadoutCompetition}
$$

其中 Phase269 新增的是：

```text
CompensationPath（补偿路径）必须进入公式；
否则无法解释 GLM4 中 MLP 抑制后 continue margin 反而增强。
```

这也支持用户当前提醒：

```text
当前线性公式很可能无法模拟真实运行机制。
```

正确路线不是继续 patch 线性公式，而是继续扩展物理路径图谱。

### 当前图谱进度

```text
pattern_family_atlas: 0.90
physical_path_atlas: 0.39
multi_family_case_bank: 0.45
multi_family_baseline_scan: 0.18
state_factor_atlas: 0.39
path_cluster_mining: 0.18
trace_signature_validation: 0.54
readout_competition_trace: 0.80
component_path_atlas: 0.20
stepwise_rollout_trace: 0.45
causal_closure: 0.18
general_language_mechanism_confidence: 0.69
```

总体评估：

```text
语言模式图谱整体约 41%-44%；
语言族物理路径约 39%；
组件路径图谱约 20%；
闭合约 18%。
```

闭合仍不能前置。当前最重要的是补：

```text
补偿路径；
跨层 MLP 组合路径；
attention -> MLP 的条件依赖路径；
长程 rollout 路径。
```

### 阶段结论

Phase269 的结论是“混合结果 + 关键校准”：

```text
qwen3 和 DS7B 支持 strongest MLP 对 continue path 的必要性；
GLM4 不支持，反而出现抑制 MLP 后 continue margin 上升；
因此 MLP 是重要候选写入器，但不是单层、跨模型、无条件必要机制。
```

这对当前路线非常重要：

```text
不能把 Phase268 的 MLP 主导观测结果直接升级为统一闭合公式；
必须继续完成组件级物理路径图谱，特别是补偿路径和跨层组合路径。
```

### 下一阶段任务

下一阶段仍属于当前大阶段，应作为 Phase270：

```text
Phase270: MLP compensation and cross-layer writer set audit
```

目标：

```text
解释 GLM4 的反向结果，并测试 continue path 是否由单层 MLP 变成跨层 writer set（写入器集合）。
```

建议：

```text
1. 保留 Phase269 的 6 条样本；
2. 对 strongest MLP 前后 2 层组成 window；
3. 测试 single-layer zero、multi-layer window zero、attention+MLP combined zero；
4. 增加 random same norm control（同范数随机控制）；
5. 对 GLM4 重点观察是否存在后续层补偿；
6. 输出 compensation_rows、writer_set_rows、control_rows、rollout_effect_rows；
7. 仍然不做闭合，只做组件路径图谱。
```

成功标准：

```text
1. 判断 GLM4 反向结果是否来自补偿路径；
2. 判断 continue path 是否由跨层 MLP writer set 支撑；
3. 区分真实必要性和范数/分布扰动；
4. 为更大规模组件图谱选择稳定干预方法。
```

## Phase 270: MLP 补偿路径与跨层写入器集合审计 [2026-07-08 17:26]

### 任务来源

本阶段继续 Phase269。附件中对 Phase269 的判断基本正确：Phase269 是混合结果和关键校准，不是闭合。它证明 MLP（多层感知机）是 continuation path（继续路径）的高价值候选，但不能被简化为“单层、跨模型、无条件必要”的统一机制。尤其 GLM4（模型名）出现抑制 strongest MLP（最强 MLP）后 continue-stop margin（继续-停止差值）反而上升，说明必须检查 compensation path（补偿路径）和 cross-layer writer set（跨层写入器集合）。

本阶段第一优先级仍是语言模式图谱的物理路径，不追求闭合。

### 测试脚本与结果文件

新增脚本：

```text
tests/gpt5/phase270_mlp_compensation_writer_set_audit.py
tests/gpt5/run_phase270_mlp_compensation_writer_set_audit.sh
```

结果目录：

```text
tests/result/phase270_mlp_compensation_writer_set_audit/mlp_compensation_writer_set_audit/
```

固定图谱数据已同步到：

```text
tests/result/pattern_family_atlas/v1/
frontend/public/vis_data/pattern_family_atlas/v1/
```

前端验证：

```text
npm run sync:pattern-atlas
npm run build
```

构建通过。Vite（前端构建工具）仍提示部分 chunk（代码块）超过 500KB，这是已有体积警告，不影响本阶段数据同步。

### 算法原理

Phase270 保留 Phase269 的 6 条样本，每个模型 2 条：

```text
qwen3
GLM4
DS7B
```

对每个样本，以 Phase268 找到的 strongest MLP layer（最强 MLP 层）为中心，构造前后 2 层窗口：

```text
W(L)=\{L-2,L-1,L,L+1,L+2\}
```

实际边界处自动截断。例如 qwen3 的 L35 只形成：

```text
[33, 34, 35]
```

本阶段测试 5 类干预：

```text
single_mlp_zero：只置零中心层 MLP 输出；
window_mlp_zero：置零窗口内所有 MLP 输出；
window_mlp_half：窗口内所有 MLP 输出缩放为 0.5；
attn_mlp_window_zero：窗口内 attention（注意力）和 MLP 同时置零；
random_same_norm_control：中心层 MLP 输出替换为同范数随机向量。
```

核心读出指标仍然是：

$$
M_{\mathrm{cont-stop}}
=
R_{\mathrm{continue}}
-
R_{\mathrm{stop}}
$$

干预效应：

$$
\Delta M
=
M_{\mathrm{patched}}
-
M_{\mathrm{base}}
$$

如果：

$$
\Delta M < -1.0
$$

或 winner（胜出项）发生变化，则记为 effect_supported（支持干预有效）。

跨层写入器集合判据：

$$
\Delta M_{\mathrm{window}}
<
\Delta M_{\mathrm{single}}
-1.0
$$

或 single-layer（单层）没有翻转而 window（窗口）翻转，则记为 writer_set_supported（支持跨层写入器集合）。

补偿迹象判据：

$$
\Delta M_{\mathrm{single}} > 1.0
$$

或单层效应很弱但窗口显著变弱，记为 compensation_suspected（疑似补偿）。

### 客观结果

总量：

```text
compensation_rows: 30
writer_set_rows: 6
control_rows: 6
rollout_effect_rows: 30
observation_rows: 30
metric_rows: 21
graph_edges: 36
missing_rows: 0
```

跨模型总体：

```text
effect_supported: True=24, False=6
reverse_effect: True=4, False=26
winner_changed: True=8, False=22
writer_set_supported: True=5, False=1
compensation_suspected: True=2, False=4
rollout_changed: True=25, False=5
mean_delta_continue_stop_margin: -5.739063
mean_control_delta_continue_stop_margin: -5.578125
```

qwen3：

```text
selected_cases: 2
effect_supported: 10 / 10
winner_changed: 3 / 10
writer_set_supported: 1 / 2
compensation_suspected: 0 / 2
mean_delta_continue_stop_margin: -9.192187
mean_control_delta_continue_stop_margin: -13.015625
```

关键样本：

```text
syntax_structure:
single_mlp_zero: -5.984375
window_mlp_zero: -13.4375
window_minus_single: -7.453125
writer_set_supported: True

reasoning_constraint:
single_mlp_zero: -13.5625
window_mlp_zero: -13.6875
writer_set_supported: False
```

解释：qwen3 中 syntax_structure（语法结构）更像跨层 MLP 写入器集合；reasoning_constraint（推理约束）中单层 L34 已经足够强，窗口没有显著增加必要性。

GLM4：

```text
selected_cases: 2
effect_supported: 4 / 10
reverse_effect: 4 / 10
winner_changed: 0 / 10
writer_set_supported: 2 / 2
compensation_suspected: 2 / 2
mean_delta_continue_stop_margin: 0.09375
mean_control_delta_continue_stop_margin: 0.53125
```

关键样本：

```text
output_protocol:
single_mlp_zero: +1.9375
window_mlp_zero: -1.84375
window_minus_single: -3.78125
compensation_suspected: True

content_knowledge:
single_mlp_zero: +2.375
window_mlp_zero: -2.4375
window_minus_single: -4.8125
compensation_suspected: True
```

解释：GLM4 的 Phase269 反向结果在 Phase270 得到更清楚的校准。单层 L38 MLP 抑制会增强 continue margin（继续差值），但抑制 L36-L39 的 MLP 窗口后，continue margin 反而下降。这支持“GLM4 的 continuation path 不是单层 L38 必要，而更像跨层 writer set 或补偿路径”。

DS7B：

```text
selected_cases: 2
effect_supported: 10 / 10
winner_changed: 5 / 10
writer_set_supported: 2 / 2
compensation_suspected: 0 / 2
mean_delta_continue_stop_margin: -8.11875
mean_control_delta_continue_stop_margin: -4.25
```

关键样本：

```text
content_knowledge:
single_mlp_zero: -3.59375
window_mlp_zero: -16.84375
window_minus_single: -13.25
writer_set_supported: True

reasoning_constraint:
single_mlp_zero: -7.375
window_mlp_zero: -15.5
window_minus_single: -8.125
writer_set_supported: True
```

解释：DS7B 的结果最支持跨层 MLP writer set。中心层有效，但窗口置零远强于单层置零。

### 关键校准

本阶段最重要的正结果：

```text
5 / 6 样本支持跨层 MLP writer set；
GLM4 的 2 / 2 样本支持补偿路径解释；
qwen3 和 DS7B 的窗口 MLP 抑制整体比单层抑制更强。
```

但本阶段最重要的风险也很明显：

```text
random_same_norm_control 不是弱效应。
```

同范数随机控制的总体均值：

```text
mean_control_delta_continue_stop_margin: -5.578125
```

这说明，简单替换 MLP 输出方向本身就会强烈扰动读出竞争。也就是说：

```text
window_mlp_zero 的强效应不能直接等价为“语义机制必要”；
其中混有方向扰动、分布扰动、读出层敏感性和局部状态破坏。
```

因此 Phase270 不能被解释为闭合，也不能直接推出统一公式。它更适合被解释为：

```text
跨层 MLP 区域是 continuation path 的高价值物理路径区域；
但具体方向、子空间和自然运行机制还没有分离出来。
```

### 对 Phase269 判断的修正

Phase269 的判断“MLP 是重要继续路径写入候选，但不是单层统一机制”继续成立。

Phase270 的新增修正是：

```text
MLPWrite（MLP 写入）更可能不是一个点，而是一个局部跨层写入器集合；
GLM4 的反向结果更可能来自单层抑制后其他层或相邻 MLP 路径的补偿；
attention+MLP 同时置零并不总是比 MLP window 更强，说明 attention route（注意力路由）不是简单正向叠加项。
```

当前更合适的公式不是线性闭合公式，而是机制谱图公式：

$$
\mathrm{ContinuePath}
=
B_{\mathrm{early}}
\oplus
R_{\mathrm{attn}}
\oplus
\mathcal{W}_{\mathrm{MLP}}(L-r:L+r)
\oplus
C_{\mathrm{comp}}
\oplus
\Gamma_{\mathrm{readout}}
$$

其中：

```text
B_early：早层继续偏置；
R_attn：注意力路由；
W_MLP：跨层 MLP 写入器集合；
C_comp：补偿路径；
Gamma_readout：最终读出竞争。
```

但这个公式仍然只是图谱组织公式，不是闭合公式。

### 当前硬伤

1. 样本量仍小。每个模型只有 2 条样本，适合路径校准，不适合总体结论。

2. 同范数随机控制效应太大，说明当前干预不是纯机制选择，而是会破坏局部状态方向。

3. window_mlp_zero 是强干预，可能同时破坏目标路径、格式路径和继续路径，不能直接解释成 continuation-only（仅继续）机制。

4. attention+MLP window zero 在 GLM4 中反而仍有正向/弱正向效应，说明 attention 与 MLP 不是简单加和关系。

5. 当前模型为小模型，内部编码机制可能较粗糙，对真实语言编码机制可能有 30%-50% 偏差。

### 当前图谱进度

```text
pattern_family_atlas: 0.90
physical_path_atlas: 0.41
multi_family_case_bank: 0.45
multi_family_baseline_scan: 0.18
state_factor_atlas: 0.39
path_cluster_mining: 0.18
trace_signature_validation: 0.55
readout_competition_trace: 0.80
component_path_atlas: 0.24
stepwise_rollout_trace: 0.45
causal_closure: 0.18
general_language_mechanism_confidence: 0.70
```

总体评估：

```text
语言模式图谱整体约 43%-45%；
语言模式图谱物理路径约 41%；
组件路径图谱约 24%；
闭合约 18%。
```

进度提升主要来自：

```text
从单层 MLP 候选推进到跨层 MLP writer set 候选；
初步定位 GLM4 补偿路径；
明确同范数方向控制是下一阶段必须处理的关键瓶颈。
```

### 下一阶段任务

下一阶段仍属于 Phase264-300 大阶段，应继续自动推进为：

```text
Phase271: MLP writer set direction and natural-subspace control audit
```

目标不是闭合，而是把 Phase270 的强干预拆开：

```text
1. 在 window MLP 中分离自然方向和随机方向；
2. 对比 zero、scale、mean-replacement、same-norm-random、phase268-positive-direction-ablation；
3. 检查 continue margin、target logit、format channel 是否被共同破坏；
4. 判断 writer set 的真实机制是幅度、方向、子空间，还是局部状态完整性；
5. 将样本从 6 条扩展到每模型 6-12 条，优先覆盖 syntax_structure、reasoning_constraint、output_protocol、content_knowledge。
```

阶段成功标准：

```text
如果自然方向抑制强于随机方向控制，说明 writer set 具有可分离机制方向；
如果随机方向控制同样强，说明当前 hook 干预主要破坏状态完整性，需要转向更细粒度的子空间/通道分析；
如果 GLM4 window 继续稳定压低 continue margin，则补偿路径假说增强。
```

### 阶段结论

Phase270 是正确推进，但不是闭合。它把 Phase269 的混合结果推进为更清楚的物理路径判断：

```text
continuation path（继续路径）更像跨层 MLP writer set（写入器集合）支撑，
GLM4 的反向结果更像补偿路径或跨层分布效应，
但同范数随机控制强效应说明当前干预仍然过粗。
```

当前第一优先级仍然是完成语言模式图谱的物理路径，第二优先级才是在物理路径足够完整后尝试闭合。

## Phase 271: MLP 写入器方向控制与 Closure-Fiber 质量审计 [2026-07-08 17:44]

### 任务来源

本阶段综合两个输入：

```text
1. Phase270 的结果分析；
2. AGI_GLM5_MEMO_SUMMARY.md 中 closure-fiber score、blocker field、boundary margin、gear-field-gate、answer-class/span/alias closure 等历史机制公式。
```

判断：这些 GLM5 历史公式具有很高参考价值，但不能把当前路线重新拉回“闭合优先”。正确用法是：

```text
把 GLM5 的闭合判据降级为 Pattern Atlas（模式图谱）和 Trace（链路追踪）的质量控制字段；
用它们筛掉强扰动但副作用大的伪机制边；
继续优先完成语言模式图谱的物理路径。
```

Phase270 的判断也基本正确：跨层 MLP writer set（MLP 写入器集合）是高价值物理路径区域，但 random_same_norm_control（同范数随机控制）强效应说明当前干预仍然过粗，不能直接解释成自然机制方向。

### 测试脚本与结果文件

新增脚本：

```text
tests/gpt5/phase271_mlp_writer_direction_closure_fiber_audit.py
tests/gpt5/run_phase271_mlp_writer_direction_closure_fiber_audit.sh
```

结果目录：

```text
tests/result/phase271_mlp_writer_direction_closure_fiber_audit/mlp_writer_direction_closure_fiber_audit/
```

固定图谱数据已同步到：

```text
tests/result/pattern_family_atlas/v1/
frontend/public/vis_data/pattern_family_atlas/v1/
```

前端验证：

```text
npm run sync:pattern-atlas
npm run build
```

构建通过。Vite（前端构建工具）仍提示部分 chunk（代码块）超过 500KB，这是已有体积警告，不影响本阶段数据同步。

### 算法原理

Phase271 将样本从 Phase270 的每模型 2 条扩大到每模型 6 条，总计：

```text
qwen3: 6
GLM4: 6
DS7B: 6
总计: 18
```

优先选择 Phase268 中 MLP 正向增强最强、且覆盖不同语言模式族的样本。

每个样本仍以 strongest MLP layer（最强 MLP 层）为中心，构造窗口：

$$
W(L)=\{L-2,L-1,L,L+1,L+2\}
$$

本阶段测试 4 类窗口控制：

```text
window_mlp_zero：窗口 MLP 输出置零；
window_mlp_half：窗口 MLP 输出缩放到 0.5；
window_mlp_mean_replace：窗口 MLP 输出替换为该模型/该层样本均值；
window_mlp_random_same_norm：窗口 MLP 输出替换为同范数随机方向。
```

这四类控制的目的不是闭合，而是区分：

```text
自然方向强度；
幅度扰动；
均值分布替换；
随机方向破坏；
局部状态完整性破坏。
```

继续-停止边界仍定义为：

$$
M_{\mathrm{cont-stop}}
=
R_{\mathrm{continue}}
-
R_{\mathrm{stop}}
$$

干预效应：

$$
\Delta M
=
M_{\mathrm{patched}}
-
M_{\mathrm{base}}
$$

本阶段引入 GLM5 风格 closure-fiber（闭合纤维）质量控制字段：

$$
\mathrm{FiberScore}
=
\mathrm{TargetLift}
+
\mathrm{RankMarginDelta}
+
\mathrm{BlockerSuppression}
-
\mathrm{SideEffect}
$$

其中：

```text
TargetLift：答案类 logit 变化；
RankMarginDelta：答案类边界间隔变化；
BlockerSuppression：阻塞场最大 logit 被压低的程度；
SideEffect：目标/边界被扰动的副作用；
```

clean_edge_candidate（干净边候选）必须同时满足：

```text
field_admissible：基线状态处于 continue winner；
blocker_suppressed：阻塞项被压低；
low_side_effect：副作用较低；
continue_margin_delta < 0：确实压低 continue margin；
patch_type 不是 random_same_norm。
```

这不是闭合判据，只是机制候选筛选器。

### 客观结果

总量：

```text
direction_control_rows: 72
closure_fiber_rows: 72
control_rows: 36
rollout_effect_rows: 72
observation_rows: 72
metric_rows: 15
graph_edges: 144
missing_rows: 0
```

跨模型总体：

```text
direction_effect_supported: True=54, False=18
state_integrity_risk: True=17, False=55
clean_edge_candidate: True=5, False=67
winner_changed: True=13, False=59
rollout_changed: True=62, False=10
mean_delta_continue_stop_margin: -4.831814
mean_closure_fiber_score: -5.402267
```

最重要的客观结果：

```text
72 条方向控制中，54 条能影响 continue-stop margin；
但只有 5 条通过 clean-edge candidate 质量筛选。
```

这说明：

```text
能扰动路径 ≠ 找到干净机制边；
Phase270 的窗口强效应大多仍混有状态破坏、副作用或阻塞场迁移。
```

### 分模型结果

qwen3：

```text
direction_control_rows: 24
direction_effect_supported: True=19, False=5
state_integrity_risk: True=6, False=18
clean_edge_candidate: True=0, False=24
winner_changed: True=3, False=21
mean_delta_continue_stop_margin: -6.044271
mean_closure_fiber_score: -10.793696
```

分干预均值：

```text
window_mlp_zero: -10.604167
window_mlp_random_same_norm: -10.302083
window_mlp_half: -2.0625
window_mlp_mean_replace: -1.208333
```

解释：qwen3 的 zero 和 random 效应几乎同级，说明窗口 MLP 区域很重要，但当前干预主要可能破坏局部状态完整性。没有 clean-edge 候选，不能把 qwen3 的结果解释成干净自然机制方向。

GLM4：

```text
direction_control_rows: 24
direction_effect_supported: True=12, False=12
state_integrity_risk: True=5, False=19
clean_edge_candidate: True=4, False=20
winner_changed: True=2, False=22
mean_delta_continue_stop_margin: -1.099609
mean_closure_fiber_score: -1.619487
```

分干预均值：

```text
window_mlp_zero: -1.742188
window_mlp_random_same_norm: -2.4375
window_mlp_half: -0.445312
window_mlp_mean_replace: +0.226562
```

clean-edge 候选主要来自：

```text
output_protocol / window_mlp_half
content_knowledge / window_mlp_half
closure / window_mlp_half
cross_lingual / window_mlp_mean_replace
```

解释：GLM4 的窗口强干预效应不算最强，但低幅度干预和均值替换更容易通过质量筛选。这与 Phase269/270 的 GLM4 补偿路径判断一致：GLM4 可能更依赖细微的跨层平衡，而不是粗暴置零。

DS7B：

```text
direction_control_rows: 24
direction_effect_supported: True=23, False=1
state_integrity_risk: True=6, False=18
clean_edge_candidate: True=1, False=23
winner_changed: True=8, False=16
mean_delta_continue_stop_margin: -7.351562
mean_closure_fiber_score: -3.79362
```

分干预均值：

```text
window_mlp_zero: -12.78125
window_mlp_random_same_norm: -9.895833
window_mlp_half: -4.520833
window_mlp_mean_replace: -2.208333
```

唯一 clean-edge 候选：

```text
reasoning_constraint / window_mlp_mean_replace
delta_continue_stop_margin: -3.0
blocker_suppression: 0.125
side_effect_score: 0.375
winner_changed: False
```

解释：DS7B 的 MLP window 物理路径很强，但 zero/random 仍然副作用大；更干净的候选出现在 mean_replace，这提示“自然分布内替换”可能比粗暴置零更适合下一阶段。

### 对附件判断的校准

附件中关于 Phase270 的判断正确：

```text
Phase270 不是闭合；
跨层 MLP 区域是高价值物理区域；
random_same_norm_control 强效应说明机制方向没有分离。
```

本阶段新增校准：

```text
GLM5 的 closure-fiber、blocker field、boundary margin、gear-field-gate 很有参考价值；
但它们应作为质量控制字段，而不是当前主目标；
用这些字段筛选后，大多数强扰动都不能算 clean edge。
```

因此当前路线应从：

```text
窗口干预有效
```

升级为：

```text
窗口干预有效 + 阻塞场被合理压低 + 副作用低 + rollout 不异常 + answer boundary 不崩坏
```

只有后者才接近高质量机制候选。

### 核心进展

Phase271 的核心进展有 6 点：

```text
1. 样本从 6 条扩大到 18 条；
2. 固定输出 72 条方向控制和 72 条 closure-fiber 行；
3. 首次把 GLM5 后期成熟的 blocker/boundary/side-effect 判据并入 GPT5 图谱；
4. 证明强 MLP window 干预大量存在，但多数不是干净机制边；
5. 发现 zero/random 强效应常常伴随状态完整性风险；
6. 发现 half/mean_replace 虽然效应较弱，但更可能成为干净候选。
```

这说明物理路径图谱已经从：

```text
哪里有强效应
```

推进到：

```text
哪些强效应可能是干净机制候选，哪些只是粗暴状态破坏。
```

### 当前硬伤

1. closure-fiber score 仍是初版。当前 SideEffect（副作用）只用了答案类和边界变化，尚未完整覆盖格式、协议、EOS（结束符）、多 token span（多词元片段）。

2. answer-class 仍主要基于首 token 别名，未完成 span/alias/protocol-compatible answer 的完整闭合。

3. mean_replace 的均值来自本轮 6 条样本，分布估计较粗，需要更大样本库。

4. random_same_norm 强效应仍说明 MLP window hook 很容易破坏状态完整性。

5. 当前测试模型为小模型，内部编码机制可能较粗糙，对真实语言编码可能有 30%-50% 偏差。

6. clean-edge candidate 只有 5 / 72，说明距离闭合仍远。

### 当前图谱进度

```text
pattern_family_atlas: 0.91
physical_path_atlas: 0.43
multi_family_case_bank: 0.46
multi_family_baseline_scan: 0.18
state_factor_atlas: 0.40
path_cluster_mining: 0.19
trace_signature_validation: 0.56
readout_competition_trace: 0.82
component_path_atlas: 0.27
closure_fiber_quality_control: 0.18
stepwise_rollout_trace: 0.46
causal_closure: 0.18
general_language_mechanism_confidence: 0.70
```

总体评估：

```text
语言模式图谱整体约 45%-47%；
语言模式图谱物理路径约 43%；
组件路径图谱约 27%；
closure-fiber 质量控制约 18%；
闭合约 18%。
```

### 智能理论反思

当前结果继续支持“语言是动态模式网络”，但需要更严格表述：

```text
语言模式不是单一向量；
也不是单一组件；
而是由物理路径、竞争边界、阻塞场、协议门控和 rollout 控制共同构成。
```

更合适的机制谱图公式是：

$$
\mathrm{PatternPath}_i
=
\left[
B_{\mathrm{early}},
R_{\mathrm{attn}},
\mathcal{W}_{\mathrm{MLP}},
C_{\mathrm{comp}},
D_{\mathrm{state}},
\Gamma_{\mathrm{boundary}},
\Phi_{\mathrm{blocker}},
G_{\mathrm{rollout}}
\right]
$$

机制候选质量控制公式是：

$$
\mathrm{CandidateQuality}
=
\mathrm{PathEffect}
\land
\mathrm{BlockerSuppression}
\land
\mathrm{LowSideEffect}
\land
\mathrm{BoundaryStable}
\land
\mathrm{RolloutStable}
$$

当前线性公式仍然很可能无法模拟真实运行机制，因此不能靠继续 patch 线性公式闭合。更可靠路线是：

```text
先完成物理路径图谱；
再为每条路径补 blocker/boundary/side-effect/rollout 字段；
最后只在高质量候选上尝试闭合。
```

### 下一阶段任务

下一阶段仍属于 Phase264-300 大阶段，应继续为：

```text
Phase272: Span/Alias/Protocol Closure-Fiber Atlas Expansion
```

目标：

```text
1. 把 answer-class 从首 token 扩展到 alias set（别名集合）；
2. 增加 span score（多词元片段分数）；
3. 增加 protocol-compatible forms（协议兼容答案形式）；
4. 增加 period/EOS/protocol stop gate（句号/结束符/协议停止门）；
5. 对 Phase271 的 18 条样本补完整 closure-fiber 字段；
6. 再筛选一批真正低副作用、高稳定性的机制候选。
```

阶段成功标准：

```text
如果 clean-edge candidate 数量仍很少，说明当前组件干预还不够细；
如果 half/mean_replace 在 span/alias/protocol 层仍保持低副作用，说明它们是下一阶段因果审计优先对象；
如果 zero/random 在 span/alias 层大面积破坏答案或协议，说明强干预应降级为定位工具，而非机制验证工具。
```

### 阶段结论

Phase271 是正确推进。它证明：

```text
GLM5 后期公式对当前研究非常有参考价值；
但最合理的使用方式是图谱质量控制，而不是闭合优先。
```

客观结果显示：

```text
MLP window 确实是高价值物理路径区域；
但大多数强效应不是 clean edge；
mean_replace 和 half 干预虽然较弱，却更可能保留自然机制结构；
下一步必须把 answer-class/span/alias/protocol stop gate 纳入图谱。
```

当前第一优先级仍然是语言模式图谱的物理路径，第二优先级才是在高质量候选上尝试闭合。

## Phase 272: Span/Alias/Protocol Closure-Fiber 图谱扩展 [2026-07-08 17:56]

### 任务来源

本阶段继续 Phase271。附件对 Phase271 的判断正确：

```text
Phase271 是质量升级；
它没有把研究重新拉回闭合优先；
它证明“能扰动 continue path（继续路径）≠ 找到 clean edge（干净机制边）”。
```

Phase271 的核心瓶颈是：

```text
closure-fiber 仍主要停留在首 token / 简化 answer-class；
SideEffect（副作用）尚未覆盖 span（多词元片段）、alias（别名）、protocol-compatible answer（协议兼容答案）和 stop gate（停止门）。
```

因此 Phase272 的目标不是闭合，而是把 closure-fiber quality control（闭合纤维质量控制）扩展到：

```text
answer alias；
span logprob；
protocol-compatible forms；
period/EOS/protocol stop gate。
```

### 测试脚本与结果文件

新增脚本：

```text
tests/gpt5/phase272_span_alias_protocol_closure_fiber_atlas.py
tests/gpt5/run_phase272_span_alias_protocol_closure_fiber_atlas.sh
```

结果目录：

```text
tests/result/phase272_span_alias_protocol_closure_fiber_atlas/span_alias_protocol_closure_fiber_atlas/
```

固定图谱数据已同步到：

```text
tests/result/pattern_family_atlas/v1/
frontend/public/vis_data/pattern_family_atlas/v1/
```

前端验证：

```text
npm run sync:pattern-atlas
npm run build
```

构建通过。Vite（前端构建工具）仍提示部分 chunk（代码块）超过 500KB，这是已有体积警告，不影响本阶段数据同步。

### 算法原理

Phase272 复用 Phase271 的 18 条样本：

```text
qwen3: 6
GLM4: 6
DS7B: 6
```

并复用 Phase271 的 4 类窗口控制：

```text
window_mlp_zero
window_mlp_half
window_mlp_mean_replace
window_mlp_random_same_norm
```

新增三类评分：

第一，alias/span score（别名/片段分数）：

```text
alias_plain
alias_leading_space
alias_period
protocol_json
protocol_list
protocol_explain
```

对每个 completion（补全文本）计算：

$$
S_{\mathrm{span}}(y_{1:n})
=
\sum_{k=1}^{n}
\log p(y_k \mid x,y_{<k})
$$

并记录：

```text
best_alias_mean_logprob
best_span_mean_logprob
best_protocol_mean_logprob
```

第二，protocol stop gate（协议停止门）：

在 prompt + best_protocol_completion 后，计算：

```text
period_logit
eos_logit
newline_logit
continue_after_answer_logit
protocol_stop_margin
```

其中：

$$
M_{\mathrm{stop-gate}}
=
\max(z_{\mathrm{period}},z_{\mathrm{EOS}},z_{\mathrm{newline}})
-
z_{\mathrm{continue}}
$$

第三，span-protocol fiber（片段-协议纤维）质量：

$$
\mathrm{SpanProtocolFiber}
=
\Delta S_{\mathrm{span}}
+
\Delta S_{\mathrm{protocol}}
+
\Delta M_{\mathrm{stop-gate}}
-
\mathrm{SideEffect}_{\mathrm{span/protocol}}
$$

其中：

```text
SideEffect_span/protocol =
|span_delta| + |protocol_delta| + max(0, -stop_gate_delta)
```

strict_protocol_clean（严格协议干净候选）判据：

```text
continue_margin_delta < 0；
span_protocol_side_effect < 2.5；
protocol_stop_margin_delta > -1.0；
patch_type 不是 window_mlp_random_same_norm。
```

注意：这是质量筛选，不是闭合。

### 客观结果

总量：

```text
span_alias_rows: 90
protocol_gate_rows: 90
span_protocol_fiber_rows: 72
observation_rows: 72
metric_rows: 15
graph_edges: 72
missing_rows: 0
```

跨模型总体：

```text
strict_protocol_clean: True=25, False=47
mean_span_protocol_fiber_score: -1.488281
mean_continue_margin_delta: -4.831814
mean_protocol_stop_margin_delta: -0.482205
```

分模型：

```text
qwen3:
strict_protocol_clean: True=5, False=19
mean_span_protocol_fiber_score: -3.018229
mean_continue_margin_delta: -6.044271
mean_protocol_stop_margin_delta: -1.454427

GLM4:
strict_protocol_clean: True=11, False=13
mean_span_protocol_fiber_score: 0.933594
mean_continue_margin_delta: -1.099609
mean_protocol_stop_margin_delta: 1.152344

DS7B:
strict_protocol_clean: True=9, False=15
mean_span_protocol_fiber_score: -2.380208
mean_continue_margin_delta: -7.351562
mean_protocol_stop_margin_delta: -1.144531
```

分干预的关键结果：

```text
qwen3:
window_mlp_mean_replace clean=3/6
window_mlp_half clean=1/6
window_mlp_zero clean=1/6
window_mlp_random_same_norm clean=0/6

GLM4:
window_mlp_zero clean=5/6
window_mlp_half clean=5/6
window_mlp_mean_replace clean=1/6
window_mlp_random_same_norm clean=0/6

DS7B:
window_mlp_half clean=5/6
window_mlp_zero clean=2/6
window_mlp_mean_replace clean=2/6
window_mlp_random_same_norm clean=0/6
```

### 关键发现

第一，加入 span/alias/protocol stop gate 后，严格候选从 Phase271 的 5 条变成 25 条：

```text
Phase271 clean_edge_candidate: 5 / 72
Phase272 strict_protocol_clean: 25 / 72
```

这不是闭合增加，而是质量控制字段更完整后，部分干预被证明在协议层副作用较低。

第二，random_same_norm 仍然全部失败：

```text
window_mlp_random_same_norm strict_protocol_clean = 0 / 18
```

这强烈支持 Phase270/271 的校准：

```text
同范数随机方向主要是状态完整性破坏，不是自然机制控制。
```

第三，half 和 mean_replace 更像可用机制审计工具：

```text
GLM4 window_mlp_half clean=5/6
DS7B window_mlp_half clean=5/6
qwen3 window_mlp_mean_replace clean=3/6
```

这说明低幅度、分布内替换比粗暴随机方向更适合继续筛选高质量机制候选。

第四，GLM4 继续表现最好：

```text
GLM4 strict_protocol_clean = 11 / 24
mean_span_protocol_fiber_score = +0.933594
mean_protocol_stop_margin_delta = +1.152344
```

GLM4 的结果说明：

```text
某些窗口 MLP 干预不仅压低 continue margin，
还可能改善 protocol stop gate；
这与前几阶段“GLM4 依赖细微跨层平衡”的判断一致。
```

### 对 Phase271 的校准

Phase271 的判断继续成立：

```text
强效应大多不干净；
必须引入 blocker/boundary/side-effect/rollout 等质量字段。
```

Phase272 的新增校准是：

```text
当副作用字段从首 token 扩展到 span/alias/protocol stop gate 后，
部分 half/mean_replace/zero 干预的质量比 Phase271 估计更好；
但 random_same_norm 仍然失败，说明状态完整性风险是真实的。
```

当前不能说：

```text
25 条 strict_protocol_clean 已经闭合。
```

只能说：

```text
这 25 条是更值得下一步做机制因果审计的高质量候选。
```

### 当前硬伤

1. protocol-compatible forms 仍由程序规则生成，覆盖不完整。

2. alias set 仍依赖 case bank 中已有 target_aliases，很多任务只有一个别名。

3. span score 只覆盖短 completion，没有完整自然回答的长 span。

4. stop gate 只看 period/EOS/newline 与少量 continuation token，仍然是近似。

5. strict_protocol_clean 是质量筛选，不是闭合判据。

6. 当前模型为小模型，内部机制可能较粗糙，对真实语言编码机制可能有 30%-50% 偏差。

### 当前图谱进度

```text
pattern_family_atlas: 0.92
physical_path_atlas: 0.44
multi_family_case_bank: 0.46
multi_family_baseline_scan: 0.18
state_factor_atlas: 0.40
path_cluster_mining: 0.19
trace_signature_validation: 0.57
readout_competition_trace: 0.82
component_path_atlas: 0.28
closure_fiber_quality_control: 0.25
span_alias_protocol_gate: 0.20
stepwise_rollout_trace: 0.46
causal_closure: 0.18
general_language_mechanism_confidence: 0.70
```

总体评估：

```text
语言模式图谱整体约 47%-49%；
语言模式图谱物理路径约 44%；
组件路径图谱约 28%；
closure-fiber 质量控制约 25%；
span/alias/protocol gate 约 20%；
闭合约 18%。
```

### 智能理论反思

Phase272 进一步说明语言模式不是一个单点输出，而是：

```text
答案类；
表面形式；
协议格式；
终止门；
继续路径；
阻塞场；
物理组件路径；
rollout 控制
```

共同组成的动态模式网络。

因此更合适的图谱公式是：

$$
\mathrm{PatternClosureCandidate}_i
=
\left[
P_{\mathrm{physical}},
A_{\mathrm{alias}},
S_{\mathrm{span}},
G_{\mathrm{protocol}},
M_{\mathrm{stop}},
\Phi_{\mathrm{blocker}},
\Delta_{\mathrm{side}}
\right]
$$

其中：

```text
P_physical：物理路径；
A_alias：答案别名集合；
S_span：片段概率；
G_protocol：协议格式门；
M_stop：停止门；
Phi_blocker：阻塞场；
Delta_side：副作用。
```

当前线性公式仍然无法模拟真实运行机制。更稳妥的路线是：

```text
继续补全图谱字段；
对高质量候选做小规模因果审计；
把粗干预降级为定位工具；
把 half/mean_replace 作为下一阶段主要机制审计工具。
```

### 下一阶段任务

下一阶段仍属于 Phase264-300 大阶段，应继续为：

```text
Phase273: High-quality candidate causal audit with span/protocol guards
```

目标：

```text
1. 只选择 Phase272 中 strict_protocol_clean=True 的候选；
2. 优先选择 GLM4 half/zero、DS7B half、qwen3 mean_replace；
3. 对这些候选做更细粒度 layer-window shrink（窗口收缩）；
4. 比较 L±2、L±1、single layer、layer pair；
5. 保留 span/alias/protocol stop gate 作为硬约束；
6. 判断高质量候选是否能收缩到更明确的物理路径。
```

阶段成功标准：

```text
如果窗口收缩后仍保持 strict_protocol_clean，说明路径更接近真实机制边；
如果收缩后失效，说明机制依赖跨层 writer set；
如果只在 GLM4 保持稳定，说明模型特异路径很强；
如果所有候选收缩后都失败，说明当前分辨率仍不够，需要转向通道/子空间级图谱。
```

### 阶段结论

Phase272 是正确推进，但不是闭合。它把 Phase271 的质量控制从首 token 扩展到 span/alias/protocol stop gate，并得到一个关键结果：

```text
random_same_norm 仍然失败；
half/mean_replace 更可能保留自然结构；
GLM4 出现最多协议干净候选；
strict_protocol_clean 候选是下一阶段机制审计的优先样本。
```

当前第一优先级仍然是完成语言模式图谱的物理路径和质量字段，第二优先级才是在高质量候选上尝试闭合。

## Phase 273: Pattern Family Atlas v2 系统构建 [2026-07-08 18:25]

### 任务来源

本阶段综合两个附件：

```text
1. Phase272 的判断校准；
2. 语言模式族图谱系统方案建议。
```

判断：附件对 Phase272 的分析正确。Phase272 不是闭合，而是把质量控制从强干预扩展到 answer alias、span score、protocol-compatible answer 和 stop gate。它证明：

```text
random_same_norm 仍然失败；
half / mean_replace 更适合机制审计；
GLM4 出现最多协议干净候选；
strict_protocol_clean 是下一阶段候选筛选器，不是闭合判据。
```

第二个附件的系统判断也正确：当前路线方向合理，但执行方式需要从“小 Phase 试探”升级为“图谱工程驱动”。因此本阶段不继续跑一个局部模型测试，而是完成 Pattern Family Atlas v2 的系统骨架。

### 本阶段是否进行模型测试

本阶段没有重新加载 qwen3、GLM4、DS7B 做模型推理。

原因：

```text
当前任务优先级是完成语言模式图谱的物理分布拼图；
Phase266-272 已经产生大量固定格式数据；
当前瓶颈不是再跑一个局部 patch，而是统一 schema、主表、评分、详情索引和客户端加载方式。
```

因此 Phase273 是系统工程阶段，不是模型测试阶段。

### 新增脚本与文档

新增脚本：

```text
tests/gpt5/phase273_pattern_family_atlas_v2_system_build.py
tests/gpt5/run_phase273_pattern_family_atlas_v2_system_build.sh
```

新增前端同步脚本：

```text
frontend/scripts/sync_pattern_atlas_v2.mjs
```

新增客户端方案：

```text
frontend/PATTERN_ATLAS_V2_CLIENT_SPEC.md
```

新增系统方案文档：

```text
research/MainAnalysis/20260709_03_Pattern_Family_Atlas_v2_系统方案.md
```

修改：

```text
frontend/package.json
```

新增命令：

```text
npm run sync:pattern-atlas:v2
```

### 结果文件

v2 输出目录：

```text
tests/result/pattern_family_atlas/v2/
```

前端同步目录：

```text
frontend/public/vis_data/pattern_family_atlas/v2/
```

核心文件：

```text
manifest.json
schema.json
client_index.json
families.jsonl
modes.jsonl
cases.jsonl
path_signature_rows.jsonl
atlas_scores.jsonl
graph_nodes.jsonl
graph_edges.jsonl
summary.md
case_details/*.json
```

### 客观结果

Phase273 构建结果：

```text
path_signatures: 972
atlas_scores: 36
case_details: 972
```

前端同步：

```text
Synced 983 pattern atlas v2 files
```

前端构建：

```text
npm run build: passed
```

Vite（前端构建工具）仍提示部分 chunk 超过 500KB，这是已有体积警告，不影响 v2 数据同步。

### v2 主表设计

v2 的主表是：

```text
path_signature_rows.jsonl
```

每一行对应：

```text
model + case_id
```

核心结构：

```json
{
  "schema_version": "2.0.0",
  "signature_id": "phase273:signature:glm4:case_id",
  "case_id": "...",
  "model": "glm4",
  "family_id": "output_protocol",
  "mode_id": "explain_answer",
  "variant_id": "structured_json",
  "path_signature": {
    "trigger": "json_structure",
    "state": ["S_content", "S_protocol", "S_continue"],
    "dominant_layers": [36, 37, 38, 39],
    "attention_route_score": 0.0,
    "mlp_write_score": 30.648437,
    "compensation_score": 3.78125,
    "readout_winner": "continue",
    "top_competitor": "answer_boundary",
    "strict_protocol_clean_count": 2
  },
  "scores": {
    "behavior": 1.0,
    "readout": 0.201302,
    "layer_path": 0.828472,
    "component_path": 0.541396,
    "causal": 0.583333,
    "rollout": 0.75,
    "closure": 0.75,
    "overall": 0.664929
  },
  "status": "high_quality_candidate_not_closed",
  "detail_ref": "case_details/glm4__case_id.json"
}
```

### v2 评分公式

v2 使用工程评分，不是闭合公式：

$$
Score(x)
=
\frac{
B(x)+R(x)+L(x)+C(x)+I(x)+G(x)+K(x)
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

图谱矩阵：

$$
AtlasScore(f,m)
=
\frac{1}{|X_{f,m}|}
\sum_{x \in X_{f,m}}
Score(x)
$$

注意：这是图谱完成度和候选排序公式，不是智能理论统一公式。

### v2 当前分布

v2 主表状态：

```text
mapped_partial: 963
path_candidate_not_closed: 7
high_quality_candidate_not_closed: 2
```

当前最强候选：

```text
glm4 / output_protocol / explain_answer / structured_json
overall: 0.664929
status: high_quality_candidate_not_closed

glm4 / closure / answer_correct / structured_json
overall: 0.659777
status: high_quality_candidate_not_closed
```

这说明：

```text
v2 图谱已经能识别少数高价值候选；
但绝大多数样本仍是 mapped_partial；
component_path、causal、closure_quality 覆盖仍不足。
```

### family x model 矩阵结果

cross-model overall 分数：

```text
content_knowledge: 0.276110
output_protocol: 0.376285
reasoning_constraint: 0.364217
syntax_structure: 0.299738
language_action: 0.340660
cross_lingual: 0.356006
readout_competition: 0.297222
state_drift: 0.362152
closure: 0.334316
```

当前最强模式族：

```text
output_protocol
reasoning_constraint
state_drift
cross_lingual
```

当前最弱图谱区：

```text
content_knowledge
readout_competition
syntax_structure
```

需要注意：这些分数受 v1 数据覆盖影响，不能直接解释为模型真实能力强弱。

### 客户端方案

新增客户端设计文档：

```text
frontend/PATTERN_ATLAS_V2_CLIENT_SPEC.md
```

v2 客户端加载策略：

```text
初始加载：
manifest.json
client_index.json
atlas_scores.jsonl
families.jsonl

按需加载：
case_details/{model}__{case_id}.json
```

建议视图：

```text
Overview
Family Matrix
Path Explorer
Component View
Causal Audit
Case Detail
```

这样客户端不再一次性加载 observations 大表，而是先显示总览和矩阵，点击 case 后再加载详情。

### 对当前路线的判断

当前路线合理，但执行方式必须改变。

过去路线：

```text
做一个测试
写一个 Phase
临时决定下一步
```

Phase273 后应改为：

```text
统一 schema
批量补字段
自动评分
前端查看缺口
只对缺口开专项 Phase
```

这可以避免效率低、字段分散、结论反复被小样本推翻的问题。

### 当前问题和硬伤

1. v2 分数是工程评分，不是理论闭合。

2. 972 条 path signature 中，963 条仍是 mapped_partial，说明内部物理路径字段覆盖还不足。

3. component_path 和 causal 字段主要来自 Phase268-272 的高价值样本，不是全量覆盖。

4. v2 当前是从 v1 汇总生成，没有重新验证所有 case 的内部轨迹。

5. 小模型内部结构可能较粗糙，v2 图谱不能直接外推真实语言机制。

6. 前端目前只是数据同步和规范完成，正式 React v2 驾驶舱还没有实现。

### 当前图谱进度

```text
pattern_family_atlas: 0.94
atlas_v2_system: 0.35
physical_path_atlas: 0.45
multi_family_case_bank: 0.46
readout_competition_trace: 0.82
component_path_atlas: 0.29
closure_fiber_quality_control: 0.26
span_alias_protocol_gate: 0.21
causal_closure: 0.18
general_language_mechanism_confidence: 0.70
```

总体评估：

```text
语言模式图谱整体约 49%-51%；
语言模式图谱物理分布拼图约 45%；
v2 系统骨架约 35%；
组件路径图谱约 29%；
闭合约 18%。
```

### 下一阶段任务

下一阶段仍属于当前大阶段，但不应再做散点小实验。建议：

```text
Phase274: Pattern Family Atlas v2 full-path gap fill batch
```

目标：

```text
1. 读取 v2 path_signature_rows；
2. 自动找出 component_path / causal / closure_quality 缺口；
3. 选择每个 family-model 的 top 缺口样本；
4. 批量补 component path 和 low-side-effect causal audit；
5. 写回 v2 主表和 case_details；
6. 更新 atlas_scores。
```

优先级：

```text
1. high_quality_candidate_not_closed；
2. path_candidate_not_closed；
3. family-model 矩阵中 overall 高但 component/causal 低的样本；
4. 低分模式族的代表样本。
```

### 阶段结论

Phase273 是一次系统工程升级，不是闭合阶段。它完成了：

```text
Pattern Family Atlas v2 schema；
972 条 path signature 主表；
36 条 family-model 分数；
972 个 case detail；
前端 v2 同步脚本；
客户端 v2 方案；
系统方案文档。
```

这一步把研究从实验日志推进到图谱数据库。下一步应围绕 v2 缺口批量补字段，而不是继续每次只做一个小测试。

## Phase 274: Pattern Family Atlas v2 缺口队列与批量补图谱计划 [2026-07-08 19:11]

### 任务判断

附件中对 Phase273 的判断基本正确：当前路线已经从局部闭合尝试，升级为语言模式图谱系统工程。Phase273 的核心价值不是新增模型测试，而是把 Phase266-272 的分散结果统一成 v2 图谱主表。Phase274 继续这个阶段，目标不是立刻宣布闭合，而是把 v2 中的物理路径、组件路径、因果验证和闭合质量缺口显式抽出来，形成可执行的批量补图谱队列。

因此本阶段没有重新跑 qwen3、GLM4、DS7B 模型测试，原因是当前任务是图谱缺口定位和队列生成。模型测试应在下一阶段按队列顺序执行，避免继续随机挑单点。

### 新增文件

```text
tests/gpt5/phase274_pattern_family_atlas_v2_gap_queue.py
tests/gpt5/run_phase274_pattern_family_atlas_v2_gap_queue.sh
tests/result/phase274_pattern_family_atlas_v2_gap_queue/
tests/result/pattern_family_atlas/v2/phase274_gap_rows.jsonl
tests/result/pattern_family_atlas/v2/phase274_selected_batch_rows.jsonl
tests/result/pattern_family_atlas/v2/phase274_coverage_matrix_rows.jsonl
tests/result/pattern_family_atlas/v2/phase274_gap_summary.json
tests/result/pattern_family_atlas/v2/phase274_gap_report.md
frontend/public/vis_data/pattern_family_atlas/v2/
```

同时更新：

```text
tests/result/pattern_family_atlas/v2/manifest.json
tests/result/pattern_family_atlas/v2/client_index.json
tests/result/pattern_family_atlas/v2/schema.json
frontend/PATTERN_ATLAS_V2_CLIENT_SPEC.md
```

### 算法原理

Phase274 读取 v2 主表：

```text
path_signature_rows.jsonl
```

对每条 model-case 记录计算缺口标记：

```text
need_component_path
need_causal_audit
need_closure_quality
need_layer_path
need_readout_competition
candidate_not_closed
good_behavior_low_path
good_readout_low_causal
```

核心思想不是用复杂统计模型，而是用直接阈值和基础加权规则，把“哪里缺数据”变成可排序队列。

缺口压力公式：

```text
family_model_pressure
= 0.45 * (1 - component_path)
+ 0.45 * (1 - causal)
+ 0.10 * (1 - closure)
```

优先级公式：

```text
priority_score
= candidate_bonus
+ 2.0 * overall
+ 1.4 * behavior
+ 1.2 * readout
+ 0.8 * rollout
+ 0.8 * closure
+ 0.7 * family_model_pressure
+ gap_count_bonus
+ high_signal_gap_bonus
```

其中：

```text
candidate_bonus:
  high_quality_candidate_not_closed = 6.0
  path_candidate_not_closed = 4.0

high_signal_gap_bonus:
  good_behavior_low_path = 1.0
  good_readout_low_causal = 1.0
```

然后按 family-model 均衡抽样，形成首批 54 条补图谱队列。

### 客观结果

输入：

```text
source_path_signatures = 972
```

状态分布：

```text
mapped_partial = 963
path_candidate_not_closed = 7
high_quality_candidate_not_closed = 2
```

缺口统计：

```text
need_component_path = 954
need_causal_audit = 959
need_layer_path = 891
need_closure_quality = 587
need_readout_competition = 376
candidate_not_closed = 9
good_behavior_low_path = 675
good_readout_low_causal = 376
```

批次类型：

```text
candidate_closure_path_fill = 9
high_signal_missing_mechanism = 753
component_and_causal_gap = 201
causal_audit_gap = 5
coverage_balance = 4
```

首批队列：

```text
selected_batch_rows = 54
qwen3 = 18
glm4 = 18
deepseek7b = 18
```

每个模式族覆盖：

```text
content_knowledge = 6
output_protocol = 6
reasoning_constraint = 6
syntax_structure = 6
language_action = 6
cross_lingual = 6
readout_competition = 6
state_drift = 6
closure = 6
```

最高压力单元：

```text
cross_lingual / deepseek7b:
  pressure = 0.994444
  physical_distribution_progress = 0.126093
  closure_readiness = 0.018519

syntax_structure / glm4:
  pressure = 0.987605
  physical_distribution_progress = 0.095341
  closure_readiness = 0.023585

content_knowledge / deepseek7b:
  pressure = 0.984665
  physical_distribution_progress = 0.076952
  closure_readiness = 0.014060

closure / deepseek7b:
  pressure = 0.977778
  physical_distribution_progress = 0.097354
  closure_readiness = 0.074074

language_action / deepseek7b:
  pressure = 0.977778
  physical_distribution_progress = 0.123576
  closure_readiness = 0.074074
```

前端同步结果：

```text
npm run sync:pattern-atlas:v2
Synced 988 pattern atlas v2 files

npm run build
build passed
```

仍存在 Vite 大 chunk 警告，但不影响图谱数据发布。

### 结果分析

Phase274 证明了一个非常关键的客观事实：

```text
当前 v2 图谱不是缺少行为样本，
而是严重缺少 component_path 和 causal_audit。
```

972 条记录中：

```text
component_path 缺口 954 条；
causal_audit 缺口 959 条；
```

说明当前研究已经不应该继续追加更多普通行为样本。真正瓶颈是：

```text
行为结果 -> 读出竞争 -> 层级路径 -> 组件路径 -> 低副作用因果验证
```

这条链条没有在大多数模式族中连起来。

另一个重要结果是：

```text
good_behavior_low_path = 675
good_readout_low_causal = 376
```

这说明很多样本表面行为较好，或读出信号较强，但物理路径和因果链条没有补齐。也就是说，不能用“回答对了”或“读出方向看起来强”替代机制解释。

### 问题和硬伤

1. Phase274 是缺口队列，不是新的模型测试结果。

2. priority_score 是工程排序指标，不是智能理论公式，也不是闭合证据。

3. component_path 和 causal_audit 的阈值是保守工程阈值，后续可能需要根据更多真实补图谱结果校准。

4. selected_batch_rows 虽然三模型和九个模式族均衡，但每个 family-model 首批只有 2 条，仍然不能用于理论总结。

5. 当前小模型内部结构可能较粗糙，路径缺口可能部分来自模型能力不足或结构不稳定，不能直接外推为真实语言编码机制。

6. 当前 v2 客户端已经能读取 gap 文件，但正式驾驶舱 UI 还需要实现 Gap Matrix、Gap Queue、Batch Planner 三个视图。

### 理论进展

Phase274 对理论的贡献不是提出新名词，而是把“语言模式图谱”的完成目标变得更客观：

```text
模式族图谱完成度
不应由样本数量决定，
而应由物理路径链条覆盖度决定。
```

当前应优先补齐：

```text
Layer path
Component path
Readout competition
Causal audit
Closure quality
```

也就是把每个模式族从“行为标签”推进到“物理路径记录”。

统一机制公式仍应保持谨慎：

```text
Output
= Readout(
    ResidualState
    + AttentionRoute
    + MLPWrite
    + ComponentCompensation
    + ProtocolGate
  )
```

但 Phase274 的结果再次提示：当前线性公式可能只适合作为索引框架，不能当作真实运行机制。真正需要完成的是机制谱图：

```text
PatternFamily
-> StateTrigger
-> LayerPath
-> ComponentPath
-> ReadoutCompetition
-> CausalEffect
-> ClosureQuality
```

### 当前进度评估

只根据当前进展估计：

```text
语言模式图谱整体进度: 50%-52%
物理分布拼图进度: 45%-47%
v2 图谱系统骨架: 42%-45%
组件路径覆盖: 29%-31%
因果审计覆盖: 18%-20%
闭合进度: 18%-20%
```

Phase274 把系统骨架推进了一步，但没有显著提高闭合进度。

### 下一阶段任务

下一阶段仍属于同一大阶段，应继续自动推进，不应回到散点实验。

```text
Phase275: selected_batch_rows 三模型顺序补图谱测试
```

任务：

```text
1. 读取 phase274_selected_batch_rows.jsonl；
2. 按 qwen3 -> GLM4 -> DS7B 顺序运行；
3. 对每条样本补 layerwise component path；
4. 对高优先级样本做 low-side-effect causal audit；
5. 写回 phase275_component_fill_rows.jsonl；
6. 写回 phase275_causal_fill_rows.jsonl；
7. 更新 v2 case_details；
8. 重新生成 atlas_scores 和 gap_summary；
9. 客户端同步查看缺口是否实际减少。
```

### 阶段结论

Phase274 是正确推进。它没有新增模型因果证据，但完成了从“v2 图谱主表”到“可执行补图谱队列”的关键转换。

最重要的结论是：

```text
当前第一瓶颈不是样本数量，
也不是闭合公式，
而是 component_path 与 causal_audit 的大面积缺口。
```

下一步必须按队列补真实物理路径和低副作用因果审计，然后再讨论闭合。

## Phase 275: Phase274 队列驱动的三模型物理路径补图谱 [2026-07-08 19:48]

### 任务判断

附件对 Phase274 的判断正确：Phase274 的价值是把 v2 图谱从“数据主表”推进到“可执行补图谱队列”。因此 Phase275 应继续当前大阶段，不能回到随机散点实验，也不能直接追闭合。正确任务是读取 Phase274 的 selected_batch_rows，按 qwen3、GLM4、DS7B 顺序执行组件路径补全和低副作用因果审计。

本阶段实际完成了两步：

```text
1. 小批链路验证：每模型 3 条，共 9 条；
2. 放大批次：每模型最多 18 条，实际完成 45 条非候选补路径样本。
```

### 新增文件

```text
tests/gpt5/phase275_selected_gap_batch_physical_path_fill.py
tests/gpt5/run_phase275_selected_gap_batch_physical_path_fill.sh
tests/result/phase275_selected_gap_batch_physical_path_fill/
tests/result/pattern_family_atlas/v2/phase275_component_physical_path_rows.jsonl
tests/result/pattern_family_atlas/v2/phase275_component_summary_rows.jsonl
tests/result/pattern_family_atlas/v2/phase275_causal_fill_rows.jsonl
tests/result/pattern_family_atlas/v2/phase275_rollout_fill_rows.jsonl
tests/result/pattern_family_atlas/v2/phase275_missing_rows.jsonl
tests/result/pattern_family_atlas/v2/phase275_cross_model_summary.json
tests/result/pattern_family_atlas/v2/phase275_report.md
```

同时更新：

```text
tests/result/pattern_family_atlas/v2/manifest.json
tests/result/pattern_family_atlas/v2/client_index.json
tests/result/pattern_family_atlas/v2/schema.json
frontend/PATTERN_ATLAS_V2_CLIENT_SPEC.md
frontend/public/vis_data/pattern_family_atlas/v2/
```

### 测试原理

Phase275 消费 Phase274 的队列：

```text
phase274_selected_batch_rows.jsonl
```

筛选条件：

```text
优先补 need_component_path / need_causal_audit / need_layer_path；
跳过只需要 candidate_closure_verification 的候选闭合复核样本；
按 qwen3 -> GLM4 -> DS7B 顺序加载模型；
每个模型结束后释放 GPU。
```

组件路径补图谱使用逐层分解：

```text
layer_input
after_attention
after_mlp
layer_output
```

在每层计算：

```text
delta_attention
= margin(after_attention) - margin(layer_input)

delta_mlp
= margin(after_mlp) - margin(after_attention)

delta_residual
= margin(layer_output) - margin(after_mlp)
```

其中 margin 使用 continue 和 stop 的读出竞争：

```text
continue_stop_margin
= r_continue - r_stop
```

低副作用因果审计使用 strongest_mlp_layer：

```text
MLP_half:
  last_token_mlp_output *= 0.5

MLP_zero:
  last_token_mlp_output *= 0.0
```

判定：

```text
causal_effect_supported
= delta_continue_stop_margin < -0.75
   or winner_changed

side_effect_risk
= winner_changed or rollout_changed
```

注意：MLP_half 是较低副作用候选，MLP_zero 是诊断性强干预，不作为闭合证明。

### 执行结果

小批链路验证：

```text
selected_gap_rows = 9
component_summary_rows = 9
causal_fill_rows = 18
missing_rows = 0
```

放大批次：

```text
selected_gap_rows = 45
component_physical_path_rows = 1552
component_summary_rows = 45
causal_fill_rows = 90
rollout_fill_rows = 90
missing_rows = 0
```

模型分布：

```text
qwen3 = 17
glm4 = 13
deepseek7b = 15
```

模式族分布：

```text
closure = 5
cross_lingual = 5
state_drift = 6
reasoning_constraint = 5
output_protocol = 4
language_action = 6
readout_competition = 5
content_knowledge = 4
syntax_structure = 5
```

主导组件：

```text
mlp = 42
attention = 3
```

最终读出赢家：

```text
continue = 45
```

因果审计：

```text
causal_effect_supported True = 59
causal_effect_supported False = 31

side_effect_risk False = 50
side_effect_risk True = 40

low_side_effect_supported_rate = 0.622222
low_side_effect_risk_rate = 0.266667
```

平均正向贡献：

```text
mean_sum_positive_attn_delta = 10.215375
mean_sum_positive_mlp_delta = 24.487299
mean_sum_positive_residual_delta = 0.236155
```

分模型结果：

```text
qwen3:
  selected_gap_rows = 17
  dominant_positive_component = mlp 17/17
  causal_effect_supported = 30/34
  low_side_effect_supported_rate = 0.882353
  low_side_effect_risk_rate = 0.117647
  strongest_mlp_layers: L35=11, L34=4, L16=2

GLM4:
  selected_gap_rows = 13
  dominant_positive_component = mlp 10/13, attention 3/13
  causal_effect_supported = 8/26
  low_side_effect_supported_rate = 0.307692
  low_side_effect_risk_rate = 0.615385
  strongest_mlp_layers: L22=4, L38=3, L5=2, L39=2, L8=1, L14=1

DS7B:
  selected_gap_rows = 15
  dominant_positive_component = mlp 15/15
  causal_effect_supported = 21/30
  low_side_effect_supported_rate = 0.600000
  low_side_effect_risk_rate = 0.133333
  strongest_mlp_layers: L26=12, L0=2, L25=1
```

前端同步：

```text
npm run sync:pattern-atlas:v2
Synced 995 pattern atlas v2 files

npm run build
build passed
```

仍有 Vite 大 chunk 警告，不影响图谱数据发布。

### 客观分析

Phase275 的第一条客观结果：

```text
Phase274 队列可以真实驱动三模型补图谱；
45 条样本补图谱无缺失。
```

这说明 Phase274 不是单纯计划，而是可执行调度系统。

第二条客观结果：

```text
MLP 在本批物理路径中占明显主导：
42/45 样本 dominant_positive_component = mlp。
```

这支持前面 Phase268-272 的路线：很多语言模式的 continue/stop/readout 竞争，不只是表层读出，而存在明显的 MLP 写入通道。

第三条客观结果：

```text
qwen3 和 DS7B 的低副作用 MLP_half 支持率较高，
GLM4 的低副作用支持率低且 side_effect_risk 高。
```

这说明不同小模型的内部机制稳定性不同。GLM4 的结果不能简单解释为“机制不存在”，更可能说明：

```text
1. GLM4 的相关路径更分散；
2. 单层 MLP half 干预副作用较大；
3. 需要更细的 channel / neuron / source-restricted intervention；
4. 小模型结构粗糙导致干预污染更明显。
```

### 问题和硬伤

1. 本阶段仍不是闭合验证。

2. 所有样本来自 Phase274 队列中的非候选补路径部分，候选闭合复核样本还没有专项处理。

3. MLP_zero 是强干预，副作用较大，不能作为低副作用闭合证据。

4. MLP_half 虽然更温和，但仍可能影响多个并行状态，不等于精准机制干预。

5. 当前只补 45 条，不是 972 条全量补图谱。

6. GLM4 的高 side_effect_risk 是重要硬伤，说明简单单层 MLP 缩放不足以稳定解释所有模型。

7. 当前模型是小模型，内部结构可能粗糙，路径和因果结果可能和真实语言编码机制有 30%-50% 偏差。

### 理论进展

Phase275 给当前机制谱图补上了一块重要拼图：

```text
PatternFamily
-> StateTrigger
-> LayerPath
-> ComponentPath
-> MLPWriteDominance
-> LowSideEffectCausalProbe
```

但它也说明不能把机制公式简化为单一线性项：

```text
Output = Readout(Residual + MLP)
```

更谨慎的表达应是：

```text
ReadoutCompetition
= f(
    LayerState,
    AttentionRoute,
    MLPWrite,
    ResidualCarry,
    ProtocolGate,
    SideEffectCoupling
  )
```

当前最确定的是：

```text
MLPWrite 是重要物理路径；
但 MLPWrite 不等于完整语言机制；
低副作用因果链仍未闭合。
```

### 当前进度评估

只根据当前结果估计：

```text
语言模式图谱整体进度: 52%-54%
物理分布拼图进度: 48%-50%
v2 图谱系统骨架: 45%-48%
组件路径覆盖: 32%-34%
因果审计覆盖: 21%-23%
闭合进度: 19%-20%
```

Phase275 明显推进物理分布拼图，但闭合只小幅推进。

### 下一阶段任务

下一阶段仍属于同一大阶段，应继续完成：

```text
Phase276: Phase275 结果回灌后的 v2 缺口重算
```

任务：

```text
1. 读取 phase274_gap_rows；
2. 读取 phase275_component_summary_rows；
3. 读取 phase275_causal_fill_rows；
4. 标记哪些缺口已经被真实补图谱覆盖；
5. 重新计算 remaining gaps；
6. 生成下一批 Phase277 队列；
7. 客户端同步显示补图谱进度。
```

### 阶段结论

Phase275 是 Phase274 之后的正确推进。它第一次把 v2 缺口队列转成三模型真实补图谱结果，证明当前路线可以系统推进。

最重要结论：

```text
当前语言模式图谱的物理路径中，
MLP 写入通道是高频主导路径，
但低副作用因果闭合仍不稳定，
尤其 GLM4 存在明显副作用风险。
```

因此下一步不是总结理论，而是回灌结果、重算缺口、继续补下一批物理路径。

## Phase 276: Phase275 回灌后的 v2 缺口重算与下一批队列 [2026-07-08 19:51]

### 任务判断

Phase275 已经把 Phase274 队列中的一批真实样本转化为组件路径和低副作用因果审计结果。按照当前大阶段目标，下一步不应该立刻总结理论，也不应该直接追闭合，而是要把 Phase275 的结果回灌到 v2 缺口系统，重新计算还有哪些物理路径缺口。

因此 Phase276 属于同一阶段，是 Phase274-Phase275 的必要闭环。

### 新增文件

```text
tests/gpt5/phase276_gap_recalibration_after_phase275.py
tests/gpt5/run_phase276_gap_recalibration_after_phase275.sh
tests/result/phase276_gap_recalibration_after_phase275/
tests/result/pattern_family_atlas/v2/phase276_recalibrated_gap_rows.jsonl
tests/result/pattern_family_atlas/v2/phase276_next_batch_rows.jsonl
tests/result/pattern_family_atlas/v2/phase276_coverage_rows.jsonl
tests/result/pattern_family_atlas/v2/phase276_summary.json
tests/result/pattern_family_atlas/v2/phase276_report.md
```

同时更新：

```text
tests/result/pattern_family_atlas/v2/manifest.json
tests/result/pattern_family_atlas/v2/client_index.json
tests/result/pattern_family_atlas/v2/schema.json
frontend/public/vis_data/pattern_family_atlas/v2/
```

### 回算原理

输入：

```text
phase274_gap_rows.jsonl
phase275_component_summary_rows.jsonl
phase275_causal_fill_rows.jsonl
```

对每条 Phase274 gap row 进行回算：

```text
if phase275_component_summary exists:
  need_component_path = false
  need_layer_path = false

if phase275_causal_fill_rows exists:
  need_causal_audit = false

if low-side-effect causal supported and no side-effect risk:
  good_readout_low_causal = false
```

状态定义：

```text
filled_by_phase275:
  当前缺口维度全部被 Phase275 覆盖

partially_filled_by_phase275:
  有组件路径或因果审计补入，但仍有其他缺口

still_open:
  Phase275 尚未覆盖
```

优先级回算：

```text
priority_after_phase275
= priority_before
- 2.0 * component_path_filled
- 2.0 * causal_audit_filled
- 1.0 * low_side_effect_supported
```

这不是理论公式，只是图谱调度公式，用于下一批队列排序。

### 客观结果

输入：

```text
source_gap_rows = 972
phase275_component_summary_rows = 45
phase275_causal_fill_rows = 90
```

Phase276 状态：

```text
filled_by_phase275 = 33
partially_filled_by_phase275 = 12
still_open = 927
```

剩余缺口：

```text
need_component_path = 909
need_causal_audit = 914
need_layer_path = 846
need_closure_quality = 587
need_readout_competition = 376
candidate_not_closed = 9
good_behavior_low_path = 675
good_readout_low_causal = 356
```

下一批队列：

```text
next_batch_rows = 54
qwen3 = 18
glm4 = 18
deepseek7b = 18
```

模式族分布：

```text
closure = 6
output_protocol = 6
cross_lingual = 6
content_knowledge = 6
state_drift = 6
language_action = 6
reasoning_constraint = 6
readout_competition = 6
syntax_structure = 6
```

前端同步：

```text
npm run sync:pattern-atlas:v2
Synced 1000 pattern atlas v2 files

npm run build
build passed
```

### 结果分析

Phase276 给出一个非常客观的进度信号：

```text
Phase275 真实补图谱确实减少了缺口，
但缺口规模仍然很大。
```

具体来看：

```text
component_path:
  954 -> 909
  减少 45

causal_audit:
  959 -> 914
  减少 45

layer_path:
  891 -> 846
  减少 45
```

这与 Phase275 的 45 条补图谱样本完全一致，说明回灌逻辑没有虚增结果。

另一个关键结果：

```text
filled_by_phase275 = 33
partially_filled_by_phase275 = 12
```

这说明 45 条中，有 33 条在当前定义下已经补齐该阶段主要缺口，但仍有 12 条因为 readout、closure 或候选验证等维度没有完全关闭。

### 问题和硬伤

1. Phase276 只是回灌和重算，不是新模型测试。

2. remaining_gap_counts 仍然很大，说明当前图谱远未完成。

3. good_behavior_low_path 没有减少，因为本字段表达“行为好但原始路径弱”的历史性质，后续可能需要拆成 original_gap 和 current_gap 两个字段。

4. need_closure_quality 没有减少，因为 Phase275 没做严格闭合质量复核。

5. candidate_not_closed 仍为 9，因为 Phase275 刻意跳过候选闭合复核样本，优先补组件路径和因果路径。

6. 当前回算仍基于工程规则，不能替代真实机制闭合。

### 理论进展

Phase276 的理论意义不是发现新机制，而是把研究推进方式固定成循环：

```text
Gap Queue
-> Physical Path Fill
-> Causal Audit
-> Gap Recalibration
-> Next Queue
```

这比过去单点实验更可靠，因为它能客观显示：

```text
哪些拼图已补；
哪些拼图仍缺；
下一批该补什么；
闭合是否具备前提条件。
```

当前机制谱图继续保持：

```text
PatternFamily
-> StateTrigger
-> LayerPath
-> ComponentPath
-> ReadoutCompetition
-> CausalEffect
-> ClosureQuality
```

Phase276 证明这条谱图路线可以被工程化推进。

### 当前进度评估

只根据当前结果估计：

```text
语言模式图谱整体进度: 53%-55%
物理分布拼图进度: 50%-52%
v2 图谱系统骨架: 48%-50%
组件路径覆盖: 34%-36%
因果审计覆盖: 23%-25%
闭合进度: 19%-20%
```

闭合进度基本没有变化，因为 Phase276 不做闭合复核。

### 下一阶段任务

下一阶段仍属于同一大阶段，但应分两类执行：

```text
Phase277A:
  继续执行 phase276_next_batch_rows 的组件路径和因果审计补图谱。

Phase277B:
  对 9 条 candidate_not_closed 单独做 closure quality 复核。
```

优先级建议：

```text
1. 继续补 phase276_next_batch_rows；
2. 单独开 candidate closure verification；
3. 对 GLM4 高 side-effect risk 做更细粒度 channel / source-restricted 审计；
4. 更新客户端 Gap Matrix 和 Fill Results 视图。
```

### 阶段结论

Phase276 是正确推进。它证明 Phase275 的补图谱结果可以被 v2 图谱系统吸收，并能客观减少缺口。

最重要的结果是：

```text
当前已经形成稳定循环：
队列生成 -> 三模型补图谱 -> 缺口回算 -> 下一批队列。
```

这说明当前研究已经从散点实验进入系统化图谱工程阶段。下一步应继续扩大物理路径补图谱，同时单独处理候选闭合复核。

## Phase 277: Phase276 下一批队列的三模型物理路径补图谱 [2026-07-08 20:02]

### 任务判断

附件对 Phase275-276 的判断正确：当前研究已经进入“缺口队列 -> 补图谱 -> 回灌重算”的工程闭环。Phase277 继续同一阶段，读取 Phase276 生成的下一批队列，继续补组件路径和低副作用因果审计。

本阶段继续遵守第一优先级：

```text
先补语言模式图谱的物理分布拼图；
暂不把候选闭合复核混入普通补图谱批次。
```

因此 Phase277A 排除了只需要 candidate_closure_verification 的样本，只执行仍缺：

```text
need_component_path
need_causal_audit
need_layer_path
```

的样本。

### 新增文件

```text
tests/gpt5/phase277_next_gap_batch_physical_path_fill.py
tests/gpt5/run_phase277_next_gap_batch_physical_path_fill.sh
tests/result/phase277_next_gap_batch_physical_path_fill/
tests/result/pattern_family_atlas/v2/phase277_component_physical_path_rows.jsonl
tests/result/pattern_family_atlas/v2/phase277_component_summary_rows.jsonl
tests/result/pattern_family_atlas/v2/phase277_causal_fill_rows.jsonl
tests/result/pattern_family_atlas/v2/phase277_rollout_fill_rows.jsonl
tests/result/pattern_family_atlas/v2/phase277_missing_rows.jsonl
tests/result/pattern_family_atlas/v2/phase277_cross_model_summary.json
tests/result/pattern_family_atlas/v2/phase277_report.md
```

### 测试原理

输入：

```text
phase276_next_batch_rows.jsonl
```

筛选：

```text
排除 candidate_closure_verification；
保留物理路径缺口样本；
按 qwen3 -> GLM4 -> DS7B 顺序运行；
每个模型结束后释放 GPU。
```

组件路径分解仍使用：

```text
delta_attention = margin(after_attention) - margin(layer_input)
delta_mlp = margin(after_mlp) - margin(after_attention)
delta_residual = margin(layer_output) - margin(after_mlp)
```

因果审计仍使用：

```text
MLP_half
MLP_zero
```

并记录：

```text
causal_effect_supported
side_effect_risk
rollout_changed
```

### 客观结果

总结果：

```text
selected_gap_rows = 49
component_summary_rows = 49
component_physical_path_rows = 1688
causal_fill_rows = 98
rollout_fill_rows = 98
missing_rows = 0
```

模型分布：

```text
qwen3 = 17
GLM4 = 15
DS7B = 17
```

模式族分布：

```text
closure = 5
state_drift = 6
cross_lingual = 6
reasoning_constraint = 6
output_protocol = 5
language_action = 6
readout_competition = 6
content_knowledge = 4
syntax_structure = 5
```

主导组件：

```text
mlp = 45
attention = 4
```

最终读出赢家：

```text
continue = 49
```

因果审计：

```text
causal_effect_supported True = 55
causal_effect_supported False = 43

side_effect_risk True = 45
side_effect_risk False = 53

low_side_effect_supported_rate = 0.489796
low_side_effect_risk_rate = 0.367347
```

平均正向贡献：

```text
mean_sum_positive_attn_delta = 10.384138
mean_sum_positive_mlp_delta = 23.292037
mean_sum_positive_residual_delta = 0.245605
```

分模型：

```text
qwen3:
  selected_gap_rows = 17
  dominant_positive_component = mlp 17/17
  causal_effect_supported = 30/34
  low_side_effect_supported_rate = 0.882353
  low_side_effect_risk_rate = 0.352941

GLM4:
  selected_gap_rows = 15
  dominant_positive_component = mlp 11/15, attention 4/15
  causal_effect_supported = 2/30
  low_side_effect_supported_rate = 0.000000
  low_side_effect_risk_rate = 0.466667

DS7B:
  selected_gap_rows = 17
  dominant_positive_component = mlp 17/17
  causal_effect_supported = 23/34
  low_side_effect_supported_rate = 0.529412
  low_side_effect_risk_rate = 0.294118
```

### 结果分析

Phase277 继续支持一个稳定现象：

```text
在当前补图谱批次中，MLP 仍是高频主导组件。
```

Phase275：

```text
mlp = 42/45
```

Phase277：

```text
mlp = 45/49
```

两批合计后，MLP 写入通道仍然是最强候选物理路径。

但 Phase277 也暴露出更强的硬伤：

```text
low_side_effect_supported_rate 从 Phase275 的 0.622222 降到 0.489796；
low_side_effect_risk_rate 从 Phase275 的 0.266667 升到 0.367347。
```

说明随着样本扩展，低副作用因果稳定性变差。特别是 GLM4：

```text
low_side_effect_supported_rate = 0.000000
```

这说明普通单层 MLP_half 干预对 GLM4 当前批次不够稳定，不能用来支持闭合。

### 问题和硬伤

1. Phase277 仍不是闭合验证。

2. MLP 主导是路径现象，不等于 MLP 是完整语言机制。

3. GLM4 的结果是强警告：单层 MLP 缩放可能存在严重干预污染或路径错位。

4. qwen3 和 DS7B 虽然支持率较高，但 side_effect_risk 仍不可忽视。

5. 当前样本仍来自小模型，内部结构可能粗糙，不能直接外推到真实语言编码机制。

### 理论进展

Phase277 强化了机制谱图中这一段：

```text
LayerPath
-> ComponentPath
-> MLPWriteDominance
```

但也削弱了简单闭合想法：

```text
MLPWriteDominance
不等于
LowSideEffectCausalClosure
```

这意味着当前线性公式仍只能作为索引框架，真实机制更像多组件耦合路径：

```text
ReadoutCompetition
= f(
    LayerState,
    AttentionRoute,
    MLPWrite,
    ResidualCarry,
    ProtocolGate,
    SideEffectCoupling
  )
```

### 阶段结论

Phase277 是正确推进。它继续扩大了物理路径拼图，证明 MLP 主导不是单批偶然现象，但也证明低副作用因果审计仍不稳定，尤其 GLM4 需要更细粒度机制审计。

## Phase 278: Phase275+Phase277 双批次回灌后的缺口重算 [2026-07-08 20:02]

### 任务判断

Phase277 完成第二批真实补图谱后，必须再次回灌 v2 图谱，重新计算剩余缺口。Phase278 因此继续同一工程闭环：

```text
队列 -> 补图谱 -> 回灌 -> 下一批队列
```

本阶段不跑模型，只进行数据回算和下一批调度。

### 新增文件

```text
tests/gpt5/phase278_gap_recalibration_after_phase277.py
tests/gpt5/run_phase278_gap_recalibration_after_phase277.sh
tests/result/phase278_gap_recalibration_after_phase277/
tests/result/pattern_family_atlas/v2/phase278_recalibrated_gap_rows.jsonl
tests/result/pattern_family_atlas/v2/phase278_next_batch_rows.jsonl
tests/result/pattern_family_atlas/v2/phase278_summary.json
tests/result/pattern_family_atlas/v2/phase278_report.md
```

### 回算原理

Phase278 合并读取：

```text
phase275_component_summary_rows
phase277_component_summary_rows
phase275_causal_fill_rows
phase277_causal_fill_rows
```

对 Phase274 原始 gap rows 重新回算：

```text
如果任一补图谱批次存在 component_summary:
  need_component_path = false
  need_layer_path = false

如果任一补图谱批次存在 causal_fill_rows:
  need_causal_audit = false

如果存在 low-side-effect causal supported 且无 side-effect risk:
  good_readout_low_causal = false
```

### 客观结果

输入：

```text
source_gap_rows = 972
total_component_summary_rows = 94
total_causal_fill_rows = 188
```

回算状态：

```text
filled_by_phase275_277 = 67
partially_filled_by_phase275_277 = 27
still_open = 878
```

剩余缺口：

```text
need_component_path = 860
need_causal_audit = 865
need_layer_path = 797
need_closure_quality = 587
need_readout_competition = 376
candidate_not_closed = 9
good_behavior_low_path = 675
good_readout_low_causal = 345
```

下一批队列：

```text
next_batch_rows = 54
qwen3 = 18
GLM4 = 18
DS7B = 18
```

模式族仍均衡：

```text
closure = 6
output_protocol = 6
cross_lingual = 6
state_drift = 6
reasoning_constraint = 6
language_action = 6
content_knowledge = 6
syntax_structure = 6
readout_competition = 6
```

前端同步：

```text
npm run sync:pattern-atlas:v2
Synced 1014 pattern atlas v2 files

npm run build
build passed
```

### 结果分析

两轮补图谱合计减少：

```text
component_path:
  954 -> 860
  减少 94

causal_audit:
  959 -> 865
  减少 94

layer_path:
  891 -> 797
  减少 94
```

这与两轮补图谱合计样本数完全一致：

```text
Phase275 = 45
Phase277 = 49
Total = 94
```

说明回灌机制仍然没有虚增进度。

但剩余缺口仍然巨大：

```text
still_open = 878
```

因此当前还远未进入闭合阶段。

### 当前进度评估

只根据当前结果估计：

```text
语言模式图谱整体进度: 55%-57%
物理分布拼图进度: 53%-55%
v2 图谱系统骨架: 50%-52%
组件路径覆盖: 37%-39%
因果审计覆盖: 26%-28%
闭合进度: 20%-21%
```

### 下一阶段任务

下一阶段仍属于同一大阶段，但需要分支：

```text
Phase279A:
  执行 phase278_next_batch_rows 的第三批物理路径补图谱。

Phase279B:
  单独处理 9 条 candidate_not_closed，做 closure quality verification。

Phase279C:
  对 GLM4 高风险样本做 source-restricted / channel-level 因果审计。
```

优先级：

```text
1. 继续补物理路径；
2. 同步启动候选闭合复核；
3. 暂不进行理论总结；
4. 对 GLM4 建立单独风险队列。
```

### 阶段结论

Phase278 是正确推进。它证明两轮补图谱都能被 v2 图谱系统稳定吸收，且缺口减少严格等于真实补图谱样本数。

当前最重要结论是：

```text
系统闭环已经稳定，
但物理路径覆盖仍不足，
闭合仍然不是第一优先级。
```

## Phase 279: 扩大样本后的第三批语言模式图谱物理路径测试 [2026-07-09 05:21]

### 任务判断

本阶段根据“语言编码机制 - 语言模式图谱”的路线要求，继续加大样本数量，不再停留在单点实验或少量候选样本。当前目标仍然是完成语言模式图谱的物理分布拼图，而不是直接追求闭合。

Phase279 读取 Phase278 生成的下一批队列：

```text
phase278_next_batch_rows.jsonl
```

并排除只需要 candidate_closure_verification 的样本，继续执行第三批三模型物理路径补图谱。

### 新增文件

```text
tests/gpt5/phase279_third_gap_batch_physical_path_fill.py
tests/gpt5/run_phase279_third_gap_batch_physical_path_fill.sh
tests/result/phase279_third_gap_batch_physical_path_fill/
tests/result/pattern_family_atlas/v2/phase279_component_physical_path_rows.jsonl
tests/result/pattern_family_atlas/v2/phase279_component_summary_rows.jsonl
tests/result/pattern_family_atlas/v2/phase279_causal_fill_rows.jsonl
tests/result/pattern_family_atlas/v2/phase279_rollout_fill_rows.jsonl
tests/result/pattern_family_atlas/v2/phase279_missing_rows.jsonl
tests/result/pattern_family_atlas/v2/phase279_cross_model_summary.json
tests/result/pattern_family_atlas/v2/phase279_report.md
```

### 测试原理

测试仍使用当前稳定闭环：

```text
Gap Queue
-> Physical Path Fill
-> Causal Audit
-> Gap Recalibration
```

每个样本补：

```text
layer_input
after_attention
after_mlp
layer_output
```

并计算：

```text
delta_attention = margin(after_attention) - margin(layer_input)
delta_mlp = margin(after_mlp) - margin(after_attention)
delta_residual = margin(layer_output) - margin(after_mlp)
```

因果审计继续使用：

```text
MLP_half
MLP_zero
```

其中 MLP_half 是低副作用候选，MLP_zero 只是诊断性强干预。

### 客观结果

第三批总结果：

```text
component_summary_rows = 48
component_physical_path_rows = 1660
causal_fill_rows = 96
rollout_fill_rows = 96
missing_rows = 0
```

模型分布：

```text
qwen3 = 17
GLM4 = 15
DS7B = 16
```

模式族分布：

```text
closure = 5
state_drift = 6
cross_lingual = 6
output_protocol = 4
reasoning_constraint = 6
language_action = 6
readout_competition = 6
content_knowledge = 4
syntax_structure = 5
```

主导组件：

```text
mlp = 46
attention = 2
```

因果审计：

```text
causal_effect_supported True = 71
causal_effect_supported False = 25

side_effect_risk True = 55
side_effect_risk False = 41

low_side_effect_supported_rate = 0.645833
low_side_effect_risk_rate = 0.500000
```

平均正向贡献：

```text
mean_sum_positive_attn_delta = 9.014667
mean_sum_positive_mlp_delta = 25.441825
mean_sum_positive_residual_delta = 0.204419
```

分模型：

```text
qwen3:
  selected_gap_rows = 17
  dominant_positive_component = mlp 17/17
  causal_effect_supported = 32/34
  low_side_effect_supported_rate = 0.882353
  low_side_effect_risk_rate = 0.588235

GLM4:
  selected_gap_rows = 15
  dominant_positive_component = mlp 13/15, attention 2/15
  causal_effect_supported = 9/30
  low_side_effect_supported_rate = 0.133333
  low_side_effect_risk_rate = 0.466667

DS7B:
  selected_gap_rows = 16
  dominant_positive_component = mlp 16/16
  causal_effect_supported = 30/32
  low_side_effect_supported_rate = 0.875000
  low_side_effect_risk_rate = 0.437500
```

### 结果分析

Phase279 进一步扩大样本后，MLP 主导现象仍然稳定：

```text
Phase275: mlp = 42/45
Phase277: mlp = 45/49
Phase279: mlp = 46/48
```

这说明 MLPWrite 是当前语言模式图谱中非常强的物理路径候选。

但同时必须注意一个更重要的负面校准：

```text
low_side_effect_risk_rate = 0.500000
```

也就是说，第三批中低副作用候选干预虽然支持率不低，但副作用风险已经非常高。尤其 qwen3 和 DS7B 的支持率高，但 side_effect_risk 也同步升高，这说明简单 MLP_half 不能直接当作闭合证据。

GLM4 仍然是特殊困难模型：

```text
low_side_effect_supported_rate = 0.133333
```

说明 GLM4 可能需要更细粒度的 channel / source-restricted 因果审计。

### 问题和硬伤

1. 本阶段仍不是闭合验证。

2. MLP 主导稳定，但 MLP_half 副作用风险升高。

3. 当前测试仍基于小模型，可能有 30%-50% 的内部结构偏差。

4. 候选闭合复核样本仍未单独处理。

5. need_closure_quality 没有减少，因为本阶段继续优先补物理路径。

### 阶段结论

Phase279 完成了第三批扩大样本测试。它强化了 MLPWrite 在语言模式图谱中的高频主导地位，但也进一步证明低副作用因果闭合仍不稳定。

核心结论：

```text
物理路径拼图继续前进；
闭合仍不能提前宣布；
下一步必须合并三批结果重新回算缺口。
```

## Phase 280: 三批物理路径测试后的语言模式图谱缺口重算 [2026-07-09 05:21]

### 任务判断

Phase279 完成第三批补图谱后，需要把 Phase275、Phase277、Phase279 三批结果合并回灌，重新计算语言模式图谱的总缺口。这一步是判断“加大样本数量是否真实推进图谱”的关键。

### 新增文件

```text
tests/gpt5/phase280_gap_recalibration_after_phase279.py
tests/gpt5/run_phase280_gap_recalibration_after_phase279.sh
tests/result/phase280_gap_recalibration_after_phase279/
tests/result/pattern_family_atlas/v2/phase280_recalibrated_gap_rows.jsonl
tests/result/pattern_family_atlas/v2/phase280_next_batch_rows.jsonl
tests/result/pattern_family_atlas/v2/phase280_summary.json
tests/result/pattern_family_atlas/v2/phase280_report.md
```

### 回算原理

Phase280 合并：

```text
phase275_component_summary_rows
phase277_component_summary_rows
phase279_component_summary_rows
phase275_causal_fill_rows
phase277_causal_fill_rows
phase279_causal_fill_rows
```

然后对原始 972 条 gap rows 回算：

```text
如果任一批次已补 component_summary：
  need_component_path = false
  need_layer_path = false

如果任一批次已补 causal_fill_rows：
  need_causal_audit = false

如果存在低副作用因果支持且无副作用：
  good_readout_low_causal = false
```

### 客观结果

三批合计：

```text
total_component_summary_rows = 142
total_causal_fill_rows = 284
```

回算状态：

```text
filled_by_phase275_277_279 = 98
partially_filled_by_phase275_277_279 = 44
still_open = 830
```

剩余缺口：

```text
need_component_path = 812
need_causal_audit = 817
need_layer_path = 749
need_closure_quality = 587
need_readout_competition = 376
candidate_not_closed = 9
good_behavior_low_path = 675
good_readout_low_causal = 337
```

与 Phase274 初始缺口相比：

```text
component_path:
  954 -> 812
  减少 142

causal_audit:
  959 -> 817
  减少 142

layer_path:
  891 -> 749
  减少 142
```

这与三批真实补图谱样本数完全一致：

```text
Phase275 = 45
Phase277 = 49
Phase279 = 48
Total = 142
```

说明回灌机制仍没有虚增。

下一批队列：

```text
next_batch_rows = 54
qwen3 = 18
GLM4 = 18
DS7B = 18
```

### 前端同步

```text
npm run sync:pattern-atlas:v2
Synced 1028 pattern atlas v2 files

npm run build
build passed
```

仍有 Vite 大 chunk 警告，不影响图谱数据发布。

### 当前图谱进度

只根据当前进展估计：

```text
语言模式图谱整体进度: 57%-59%
物理分布拼图进度: 56%-58%
v2 图谱系统骨架: 53%-55%
组件路径覆盖: 40%-42%
因果审计覆盖: 29%-31%
闭合进度: 20%-21%
```

### 关键结论

三批扩大样本测试证明：

```text
语言模式图谱测试可以系统扩展；
每一批真实补图谱都会等量减少 component_path / causal_audit / layer_path 缺口；
MLPWrite 是高频主导物理路径；
但低副作用因果稳定性不足，不能进入闭合阶段。
```

当前最大瓶颈已经更清楚：

```text
1. 剩余物理路径缺口仍大；
2. closure_quality 完全没有推进；
3. GLM4 需要专门的低副作用细粒度审计；
4. candidate_not_closed 需要单独闭合质量复核。
```

### 下一阶段任务

下一阶段仍属于语言模式图谱测试大阶段，但应拆为两条：

```text
Phase281A:
  继续执行 phase280_next_batch_rows，进一步加大物理路径样本。

Phase281B:
  对 9 条 candidate_not_closed 做 closure_quality verification。
```

为了避免只做无尽补表，建议 Phase281B 必须启动，因为 closure_quality 当前仍为：

```text
need_closure_quality = 587
candidate_not_closed = 9
```

### 阶段结论

Phase280 证明第三批样本扩展有效，语言模式图谱测试已经进入可扩展工程阶段。但距离“完成语言模式图谱”仍有明显距离，目前完成的是阶段性大样本补图谱测试，不是全图谱闭合。

## Phase 281: 候选闭合点四条件复核 [2026-07-09 06:02]

### 任务判断

本轮附件对 Phase279/280 的判断基本正确：当前研究已经从单点机制追踪进入语言模式图谱的系统补图谱阶段。需要补充的是，继续扩大物理路径样本之前，必须先把 9 条 `candidate_not_closed` 复核掉，否则后续图谱会把高分候选误认为闭合。

### 测试原理

Phase281 对 Phase280 中全部 9 条候选闭合行做严格四条件复核。闭合不再只看行为正确，也不只看整体分数，而是要求：

```text
ClosureCandidate =
  SemanticDone
  ∧ StopWins
  ∧ ContinueSuppressed
  ∧ RolloutStable
```

其中：

```text
SemanticDone = answer_correct_proxy
StopWins = r_stop > r_continue
ContinueSuppressed = top_continue_vs_stop_margin <= -0.5
RolloutStable =
  pattern_matched_proxy
  ∧ no_drift_marker
  ∧ no_repeated_protocol_marker
  ∧ (model_stop_executed ∨ generated_token_count < max_rollout_tokens)
```

这里的关键是把“语义完成”和“停止机制完成”拆开：语义正确不能自动推出模型内部已经停止。

### 执行

新增脚本：

```text
tests/gpt5/phase281_candidate_closure_quality_verification.py
tests/gpt5/run_phase281_candidate_closure_quality_verification.sh
```

按顺序使用本地 CUDA 模型运行：

```text
qwen3 -> GLM4 -> DS7B
```

输出固定图谱格式：

```text
tests/result/pattern_family_atlas/v2/phase281_closure_quality_rows.jsonl
tests/result/pattern_family_atlas/v2/phase281_cross_model_summary.json
frontend/public/vis_data/pattern_family_atlas/v2/phase281_closure_quality_rows.jsonl
```

### 客观结果

```text
closure_quality_rows = 9
missing_rows = 0
four_condition_closed_count = 0
weak_candidate_survived_count = 0
semantic_done_rate = 0.666667
stop_wins_rate = 0.0
continue_suppressed_rate = 0.0
rollout_stable_rate = 0.111111
mean_stop_continue_margin = -7.838542
```

分模型：

```text
qwen3:
  candidate_rows = 1
  semantic_done = 1
  stop_wins = 0
  rollout_stable = 0

GLM4:
  candidate_rows = 5
  semantic_done = 3
  stop_wins = 0
  rollout_stable = 0

DS7B:
  candidate_rows = 3
  semantic_done = 2
  stop_wins = 0
  rollout_stable = 1
```

主要阻塞项：

```text
stop_not_winner = 9
continue_not_suppressed = 9
rollout_not_stable = 8
semantic_not_done = 3
```

### 结果分析

这是一个重要负结果。9 条候选里有 6 条语义完成，但没有任何一条通过停止胜出和继续压制。因此当前所谓 candidate 不是机制闭合，只是行为或路径分数较高。

最关键的客观现象是：

```text
语义完成 ≠ 停止完成
行为正确 ≠ 读出竞争完成
候选高分 ≠ 四条件闭合
```

这与 Phase279/280 的路线判断一致：第一优先级仍应是语言模式图谱的物理分布拼图，闭合只能在更完整的物理路径基础上尝试。

### 硬伤

当前复核仍有三个限制：

```text
1. semantic evaluator 仍是代理指标，不能替代人工语义判断；
2. rollout_stable 使用短生成，无法证明长程稳定；
3. 当前模型是小模型，停止/继续竞争机制可能比大模型粗糙。
```

所以 Phase281 不能证明大模型也没有闭合，只能证明当前小模型图谱中的这 9 条候选不能算闭合。

## Phase 282: Phase281 后图谱缺口重算 [2026-07-09 06:02]

### 任务

Phase282 不做新模型测试，只把 Phase281 的严格闭合复核结果回写到语言模式图谱缺口表。

新增脚本：

```text
tests/gpt5/phase282_gap_recalibration_after_phase281.py
tests/gpt5/run_phase282_gap_recalibration_after_phase281.sh
```

### 客观结果

```text
source_gap_rows = 972
closure_quality_rows = 9
four_condition_closed_count = 0
weak_candidate_survived_count = 0
```

状态：

```text
filled_by_phase275_277_279_281 = 98
partially_filled_by_phase275_277_279 = 44
closure_quality_rechecked_phase281 = 9
still_open = 821
```

剩余缺口：

```text
candidate_not_closed = 9
need_closure_quality = 593
need_readout_competition = 376
good_behavior_low_path = 675
good_readout_low_causal = 337
need_causal_audit = 817
need_component_path = 812
need_layer_path = 749
```

### 分析

Phase282 说明：严格闭合复核没有减少候选闭合缺口，反而把 `need_closure_quality` 从 587 校准到 593。这不是倒退，而是纠正了高分候选带来的虚假乐观。

当前闭合链条可写为：

```text
BehaviorCorrect
  ↛ StopWins
  ↛ ContinueSuppressed
  ↛ RolloutStable
```

即行为正确没有自然推出停止机制完成。

### 阶段结论

Phase282 完成的是图谱校准，不是闭合推进。它证明当前阶段不能把 closure_quality 当作普通分数补齐，而必须继续追踪读出竞争和停止/继续通道的物理来源。

## Phase 283: 第四批大样本物理路径填充 [2026-07-09 06:02]

### 任务

根据用户要求“加大测试样本数量，完成语言模式图谱测试”，Phase283 在 Phase282 的基础上继续扩大样本。为了避免重复候选闭合行，本阶段排除纯 `candidate_closure_verification`，只选择仍需要组件路径、层路径或因果审计的开放缺口。

新增脚本：

```text
tests/gpt5/phase283_fourth_gap_batch_physical_path_fill.py
tests/gpt5/run_phase283_fourth_gap_batch_physical_path_fill.sh
```

测试规模：

```text
selected_gap_rows = 54
qwen3 = 18
GLM4 = 18
DS7B = 18
families = 9
每个模型每个模式族 2 行
missing_rows = 0
```

### 测试公式

本阶段仍使用物理路径分解：

```text
ΔR_total(layer)
  = ΔR_attention(layer)
  + ΔR_mlp(layer)
  + ΔR_residual(layer)
```

并记录主导组件：

```text
DominantComponent =
  argmax_component Σ max(ΔR_component(layer), 0)
```

低副作用因果审计：

```text
LowSideEffectSupported =
  causal_effect_supported
  ∧ not side_effect_risk
```

### 客观结果

总体：

```text
component_summary_rows = 54
causal_fill_rows = 108
missing_rows = 0
dominant_positive_component_counts:
  mlp = 52
  attention = 2
final_winner_counts:
  continue = 54
low_side_effect_supported_rate = 0.666667
low_side_effect_risk_rate = 0.388889
mean_sum_positive_attn_delta = 9.105415
mean_sum_positive_mlp_delta = 25.700721
mean_sum_positive_residual_delta = 0.222938
```

分模型：

```text
qwen3:
  component_summary_rows = 18
  dominant_positive_component = mlp 18
  low_side_effect_supported_rate = 0.888889
  low_side_effect_risk_rate = 0.333333

GLM4:
  component_summary_rows = 18
  dominant_positive_component = mlp 17, attention 1
  low_side_effect_supported_rate = 0.166667
  low_side_effect_risk_rate = 0.611111

DS7B:
  component_summary_rows = 18
  dominant_positive_component = mlp 17, attention 1
  low_side_effect_supported_rate = 0.944444
  low_side_effect_risk_rate = 0.222222
```

### 进展

Phase283 是正结果，但不是闭合。它进一步强化了以下拼图：

```text
1. MLPWrite 是当前小模型语言模式图谱中最稳定的主导物理路径；
2. attention 不是不存在，但在当前样本中更多像辅助或局部路径；
3. continue winner 在 54/54 中保持，说明停止问题仍是系统性问题；
4. GLM4 是低副作用因果审计的主要困难模型。
```

### 硬伤

最大问题是 GLM4：

```text
GLM4 low_side_effect_supported_rate = 0.166667
GLM4 low_side_effect_risk_rate = 0.611111
```

这说明“压制 MLP 继续写入”虽然可能改变读出，但副作用很高，不能当作稳定机制公式。

因此当前线性补丁公式仍不能模拟真实运行机制。它更适合做物理路径探针，而不是闭合控制公式。

## Phase 284: 第四批样本后图谱总校准 [2026-07-09 06:02]

### 任务

Phase284 合并 Phase275、Phase277、Phase279、Phase283 四轮物理路径填充和 Phase281 闭合复核，重算整个语言模式图谱缺口。

新增脚本：

```text
tests/gpt5/phase284_gap_recalibration_after_phase283.py
tests/gpt5/run_phase284_gap_recalibration_after_phase283.sh
```

### 客观结果

```text
source_gap_rows = 972
total_component_summary_rows = 196
total_causal_fill_rows = 392
closure_quality_rows = 9
```

状态：

```text
filled_by_phase275_277_279_281_283 = 124
partially_filled_by_phase275_277_279_281_283 = 81
still_open = 767
```

剩余缺口：

```text
candidate_not_closed = 9
need_readout_competition = 376
good_behavior_low_path = 675
good_readout_low_causal = 324
need_causal_audit = 763
need_component_path = 758
need_layer_path = 695
need_closure_quality = 587
```

累计主导组件：

```text
dominant_component_counts_all_fills:
  mlp = 185
  attention = 11

mean_component_mlp_delta_all_fills = 24.756555
```

### 当前进度估计

只根据当前进展估计：

```text
语言模式图谱整体进度: 60%
物理分布拼图进度: 60%
组件路径覆盖: 50%
因果审计覆盖: 40%
闭合进度: 20%
```

### 图谱核心拼图

当前已经积累的核心拼图是：

```text
1. 语言模式族可以稳定拆成九个大族；
2. 每个族都能映射到固定 case / mode / variant / path_signature；
3. 大多数开放缺口的正向物理写入来自 MLP；
4. attention 在少数样本中成为主导，说明路由/选择机制仍不能忽略；
5. residual_delta 通常较小，更像累计载体而不是主要写入源；
6. continue winner 在多批样本中持续占优；
7. 高质量行为候选无法自然通过停止闭合；
8. GLM4 的低副作用因果稳定性明显弱于 qwen3 和 DS7B；
9. 当前线性干预适合定位物理路径，但不适合直接作为闭合公式。
```

### 智能理论角度

当前更合理的第一性原理表述不是：

```text
找到一个线性方向 -> patch -> 闭合
```

而是：

```text
语言能力 = 多模式族在网络内部形成的动态物理路径图谱
```

更接近当前客观结果的统一公式是：

```text
PatternState(t)
  = Route(t)
  + Write_MLP(t)
  + Select_Attention(t)
  + ResidualCarry(t)
  + ReadoutCompetition(t)
```

输出是否闭合取决于：

```text
ClosedOutput
  = SemanticDone
  ∧ ProtocolSatisfied
  ∧ StopWins
  ∧ ContinueSuppressed
  ∧ RolloutStable
```

当前已经看到，前两个条件较容易出现，后三个条件才是瓶颈。

### 接下来的阶段性任务

下一阶段仍属于“完成语言模式图谱物理分布拼图”的同一大阶段，应继续系统推进，不应退回单点 patch：

```text
Phase285A:
  继续执行 phase284_next_batch_rows，进一步扩大物理路径样本。

Phase285B:
  对 GLM4 高风险样本做 source-restricted 低副作用审计。

Phase285C:
  把 need_readout_competition = 376 单独拆成 stop/continue/channel 竞争来源表。

Phase285D:
  将 closure_quality 的四条件复核推广到非候选高行为样本，避免只复核高分候选。
```

阶段目标不是马上闭合，而是完成：

```text
语言模式族 -> 物理路径 -> 组件分布 -> 读出竞争 -> 因果审计
```

这一整条图谱链。只有这条链足够完整，闭合公式才有可能自然浮现。

### 前端同步

已同步到可视化客户端：

```text
npm run sync:pattern-atlas:v2
Synced 1050 pattern atlas v2 files

npm run build
build passed
```

仍有 Vite 大 chunk 警告，不影响图谱数据读取。

### 阶段结论

Phase281-284 完成了两件事：

```text
1. 严格否定了当前 9 条候选闭合点；
2. 把语言模式图谱物理路径样本从 142 扩展到 196。
```

当前研究方向正确，但还没有完成语言模式图谱，更没有闭合。最可靠的结论是：语言模式图谱已经进入可扩展测试阶段，MLPWrite 是目前最稳定的物理路径拼图，停止/继续读出竞争仍是闭合瓶颈。

## Phase 285: 闭合质量扩展扫描 [2026-07-09 06:33]

### 任务判断

附件对 Phase281-284 的判断总体正确：当前路线成熟点在于把“高分候选”与“机制闭合”拆开，并把图谱测试扩展到工程化批处理。但附件指出的一个问题也成立：如果继续只补 `component_path` 和 `causal_audit`，`closure_quality` 会长期停滞。

因此 Phase285 不再继续单纯扩大物理路径批次，而是启动闭合质量扩展扫描：

```text
输入:
  high behavior score
  need_closure_quality = true
  非 candidate_not_closed 旧候选

输出:
  SemanticDone
  StopWins
  ContinueSuppressed
  RolloutStable
  closure_reclassification
```

### 测试脚本

新增：

```text
tests/gpt5/phase285_closure_quality_expansion_scan.py
tests/gpt5/run_phase285_closure_quality_expansion_scan.sh
```

测试按顺序执行：

```text
qwen3 -> GLM4 -> DS7B
```

每模型 9 条，共 27 条，使用 32 token rollout。

输出固定图谱格式：

```text
tests/result/pattern_family_atlas/v2/phase285_closure_quality_rows.jsonl
tests/result/pattern_family_atlas/v2/phase285_four_condition_rows.jsonl
tests/result/pattern_family_atlas/v2/phase285_rollout_stability_rows.jsonl
tests/result/pattern_family_atlas/v2/phase285_cross_model_summary.json
```

### 公式

闭合质量继续使用硬条件：

```text
ClosedOutput =
  SemanticDone
  ∧ ProtocolMatched
  ∧ StopWins
  ∧ ContinueSuppressed
  ∧ RolloutStable
```

其中：

```text
StopWins = r_stop > r_continue
ContinueSuppressed = top_continue_vs_stop_margin <= -0.5
```

### 客观结果

总体：

```text
closure_quality_rows = 27
missing_rows = 0
semantic_done_rate = 1.0
stop_wins_rate = 0.0
continue_suppressed_rate = 0.0
rollout_stable_rate = 0.0
four_condition_closed_count = 0
```

重分类：

```text
closure_rejected = 25
semantic_protocol_ok_but_not_closed = 2
```

阻塞项：

```text
stop_not_winner = 27
continue_not_suppressed = 27
rollout_not_stable = 27
```

分模型：

```text
qwen3:
  selected_rows = 9
  semantic_done = 9
  stop_wins = 0
  four_condition_closed = 0

GLM4:
  selected_rows = 9
  semantic_done = 9
  stop_wins = 0
  four_condition_closed = 0

DS7B:
  selected_rows = 9
  semantic_done = 9
  stop_wins = 0
  four_condition_closed = 0
```

### 分析

这是比 Phase281 更强的负结果。Phase281 只复核 9 条候选闭合点；Phase285 扩展到 27 条高行为、缺闭合质量样本。结果全部语义完成，但全部无法停止胜出和继续压制。

这说明：

```text
语义完成不是瓶颈；
停止/继续竞争才是瓶颈；
rollout 稳定不能从短答正确自然推出。
```

换句话说，当前模型可以“答对”，但没有进入机制意义上的“完成态”。

### 硬伤

1. 语义完成仍然是代理指标，虽然本批样本目标命中非常干净，但不能替代人工语义评估。
2. 32 token rollout 比此前更长，但仍不是长程稳定证明。
3. 当前小模型可能有粗糙停止机制，不能直接外推到大模型。

### 阶段结论

Phase285 证明：`need_closure_quality` 不能只靠高行为样本自然补齐。闭合质量扩展扫描应该继续做，但它目前主要产生的是客观负结果，真正突破点仍在 `ReadoutCompetition`、`StopGate` 和 `ContinueSuppression`。

## Phase 286: Phase285 后缺口回灌 [2026-07-09 06:33]

### 任务

Phase286 将 Phase281 的 9 条候选复核和 Phase285 的 27 条扩展闭合质量扫描合并回灌到 Pattern Family Atlas v2。

新增：

```text
tests/gpt5/phase286_gap_recalibration_after_phase285.py
tests/gpt5/run_phase286_gap_recalibration_after_phase285.sh
```

### 客观结果

```text
source_gap_rows = 972
closure_quality_checked_rows = 36
closure_rejected_rows = 36
```

状态：

```text
filled_by_phase275_277_279_281_283_285 = 134
partially_filled_by_phase275_277_279_281_283_285 = 88
still_open = 750
```

剩余缺口：

```text
candidate_not_closed = 9
need_readout_competition = 376
good_behavior_low_path = 675
good_readout_low_causal = 324
need_causal_audit = 763
need_component_path = 758
need_layer_path = 695
need_closure_quality = 557
```

进度估计：

```text
语言模式图谱整体进度: 62%
物理分布拼图进度: 60%
组件路径覆盖: 50%
因果审计覆盖: 40%
闭合质量测量进度: 25%
闭合进度: 20%
```

### 分析

Phase286 的关键不是“闭合增加”，而是把 36 条样本从“未测闭合质量”转成“已测但闭合失败”。因此：

```text
need_closure_quality: 587 -> 557
closure: 不增加
closure_rejected: 36
```

这是必要校准。它让图谱区分：

```text
没有做闭合质量测试
≠
做了闭合质量测试但失败
```

### 阶段结论

Phase286 后，闭合质量测量开始真正进入图谱循环。但所有已测样本均失败，说明下一阶段不能再期待“高行为样本自动闭合”，必须进入读出竞争、停止门和继续压制机制的物理来源分析。

## Phase 287: GLM4 高副作用风险队列 [2026-07-09 06:33]

### 任务

附件指出 GLM4 需要单独机制分支，这个判断正确。GLM4 在 Phase283 中表现为：

```text
low_side_effect_supported_rate = 0.166667
low_side_effect_risk_rate = 0.611111
```

Phase287 不加载模型，而是从已有因果审计结果中整理 GLM4 高副作用风险队列，为后续 source-restricted / channel-level 审计建立固定输入。

新增：

```text
tests/gpt5/phase287_glm4_side_effect_risk_queue.py
tests/gpt5/run_phase287_glm4_side_effect_risk_queue.sh
```

输出：

```text
tests/result/pattern_family_atlas/v2/phase287_glm4_side_effect_risk_rows.jsonl
tests/result/pattern_family_atlas/v2/phase287_glm4_next_audit_rows.jsonl
tests/result/pattern_family_atlas/v2/phase287_summary.json
```

### 客观结果

```text
source_glm4_causal_rows = 122
glm4_side_effect_risk_rows = 74
next_audit_rows = 36
```

风险分桶：

```text
generic_side_effect_risk = 30
coupled_target_continue_risk = 22
attention_mlp_joint_risk = 18
readout_competition_risk = 4
```

推荐审计：

```text
random_same_norm_control = 74
subspace_or_mean_replace_audit = 41
source_restricted_low_side_effect_audit = 33
attention_mlp_joint_audit = 20
channel_level_stop_continue_audit = 8
```

平均变化：

```text
mean_delta_continue_stop_margin = -0.552365
mean_delta_target_logit = -0.636824
```

### 分析

GLM4 的问题不是简单“MLP 半缩放不够强”，而是：

```text
target path 和 continue path 强耦合；
降低 continue 往往也伤害 target；
attention 和 MLP 在部分样本中存在联合风险；
读出竞争仍有开放缺口。
```

这解释了为什么普通 `mlp_half_last_token` 难以稳定解决 GLM4：它不是干净的继续通道开关，而是可能同时碰到答案、协议、结构和继续路径。

### 智能理论洞察

当前更合理的机制图不是：

```text
MLPWrite -> output
```

而是：

```text
MLPWrite
  -> ReadoutCompetition(target, continue, stop, protocol)
  -> SideEffectCoupling
  -> RolloutTrajectory
```

GLM4 说明语言编码机制中存在“耦合路径”，即同一个物理写入区域可能同时承载目标答案和继续倾向。破解语言机制不能只找正向写入源，还必须找到如何分离耦合路径。

### 前端同步

已同步：

```text
npm run sync:pattern-atlas:v2
Synced 1064 pattern atlas v2 files

npm run build
build passed
```

仍有 Vite 大 chunk 警告，不影响图谱读取。

### 下一阶段任务

下一阶段仍属于“语言模式图谱物理分布拼图”同一大阶段，应继续自动推进：

```text
Phase288A:
  对 phase287_glm4_next_audit_rows.jsonl 做 GLM4 source-restricted low-side-effect audit。

Phase288B:
  对 readout_competition_gap_open 样本做 stop / continue / target / protocol channel 分解。

Phase288C:
  继续小规模 phase286_next_batch_rows 物理路径补图谱，避免物理分布增长停滞。
```

阶段目标仍不是闭合，而是补齐：

```text
物理路径分布
闭合质量测量
GLM4 高风险耦合路径
读出竞争来源
```

只有这些拼图足够完整，才有可能进入真正闭合。

## Phase 288: 模式图谱特征挖掘 [2026-07-09 07:37]

### 任务判断

本轮附件的核心判断正确：Phase285-287 已经证明继续局部追闭合或继续只做 MLP 机制审计会进入边际收益递减区。当前应明显增加“大数据图谱特征分析”的权重，把主线从：

```text
机制优先
```

调整为：

```text
图谱特征优先，机制审计后置
```

因此 Phase288 不跑新模型，不做 patch，而是读取当前 Pattern Family Atlas v2 的全量数据，抽取语言族、模型、组件、通道、层轨迹、副作用、闭合瓶颈和缺口热力图。

### 输入数据

```text
path_signature_rows = 972
component_summary_rows = 196
causal_rows = 392
closure_quality_rows = 36
gap_rows = 972
```

### 新增脚本

```text
tests/gpt5/phase288_pattern_atlas_feature_mining.py
tests/gpt5/run_phase288_pattern_atlas_feature_mining.sh
```

### 输出表

```text
phase288_family_feature_matrix.jsonl
phase288_model_feature_matrix.jsonl
phase288_component_distribution_rows.jsonl
phase288_continue_channel_distribution_rows.jsonl
phase288_layer_curve_cluster_rows.jsonl
phase288_side_effect_distribution_rows.jsonl
phase288_closure_bottleneck_rows.jsonl
phase288_gap_heatmap_rows.jsonl
phase288_feature_mining_summary.json
phase288_feature_mining_report.md
```

### 特征公式

每个样本被视为图谱路径对象：

```text
v_i =
[
  family,
  model,
  component,
  continue_channel,
  layer_cluster,
  side_effect,
  closure_failure,
  gap_state
]
```

语言机制不再写成单个线性公式，而写成图谱对象集合：

```text
LanguageMechanism =
  Atlas({M+(P_i)}_{i=1..N})
```

其中：

```text
M+(P_i) =
(
  LayerPath,
  ComponentPath,
  StatePath,
  ReadoutCompetition,
  BlockerBoundaryField,
  AnswerAliasSpanField,
  ProtocolField,
  RolloutTrajectory,
  StopClosureGate,
  SideEffectAudit
)
```

### 客观结果

全局：

```text
global_mlp_dominance_rate = 0.943878
global_attention_dominance_rate = 0.056122
global_continue_win_rate = 1.0
global_side_effect_risk_rate = 0.507653
global_closure_closed_count = 0
global_closure_rejected_count = 36
```

层轨迹粗聚类：

```text
late_mlp_strong_continue = 80
middle_mlp_strong_continue = 67
late_mlp_continue = 15
middle_mlp_continue = 15
early_attention_routed_continue = 6
middle_attention_routed_continue = 5
early_mlp_continue = 4
early_mlp_strong_continue = 4
```

继续通道分布：

```text
continue_list_item = 347
continue_the = 344
continue_next_sentence = 176
continue_json_structure = 74
continue_format = 67
```

模型矩阵：

```text
qwen3:
  mlp_dominance_rate = 1.0
  side_effect_risk_rate = 0.507246
  low_side_effect_supported_rate = 0.884058

GLM4:
  mlp_dominance_rate = 0.836066
  attention_dominance_rate = 0.163934
  side_effect_risk_rate = 0.606557
  low_side_effect_supported_rate = 0.147541

DS7B:
  mlp_dominance_rate = 0.984848
  side_effect_risk_rate = 0.416667
  low_side_effect_supported_rate = 0.742424
```

### 关键发现

1. 当前全图谱中 `continue` 胜出率是 1.0，这说明停止/继续竞争是全局性瓶颈，不是候选样本偶然问题。
2. MLP 主导率 0.943878，说明 MLPWrite 是稳定共享主干，但不是闭合机制。
3. GLM4 的 attention 主导率和副作用风险显著更高，是模型特异风险。
4. 继续通道主要集中在 `continue_list_item`、`continue_the`、`continue_next_sentence`，说明“继续”不是单一方向，而是一组通道族。
5. 闭合通过数仍为 0，说明图谱特征挖掘不能被误读为闭合推进。

### 硬伤

Phase288 是特征挖掘，不是因果验证。它只能说明分布结构：

```text
什么现象稳定出现；
什么区域缺口最大；
什么模型风险最高；
什么通道最常胜出。
```

它不能证明这些特征就是真实机制原因。因此后续机制审计必须由这些特征驱动，而不是完全取消机制审计。

## Phase 289: 特征复用-差分分析 [2026-07-09 07:37]

### 任务

Phase289 基于 Phase288 的特征矩阵，进一步做：

```text
共享主干估计；
语言族差分；
模型差分；
特征驱动的机制审计候选选择。
```

新增脚本：

```text
tests/gpt5/phase289_feature_reuse_delta_analysis.py
tests/gpt5/run_phase289_feature_reuse_delta_analysis.sh
```

### 复用-差分公式

族均值：

```text
P_bar(f) =
  mean_{x in family f} P(x)
```

共享主干：

```text
P_shared =
  mean_f P_bar(f)
```

族差分：

```text
Delta_f =
  P_bar(f) - P_shared
```

模型差分：

```text
Delta_model =
  P_model - mean_model(P_model)
```

### 客观结果

共享主干：

```text
shared_continue_win_rate = 1.0
shared_mlp_dominance_rate = 0.946745
shared_attention_dominance_rate = 0.053255
shared_mean_positive_mlp_delta = 24.749139
shared_mean_positive_attn_delta = 9.632531
shared_side_effect_risk_rate = 0.513343
shared_closure_rejected_rate = 0.888889
shared_stop_not_winner_rate = 0.888889
```

族差分标签：

```text
high_side_effect_family = 2
mlp_reuse_strong_family = 4
attention_delta_family = 2
high_behavior_family = 1
```

模型差分标签：

```text
low_side_effect_strong_model = 2
glm4_high_risk_delta = 1
```

特征驱动审计候选：

```text
feature_driven_audit_candidates = 83
```

候选类型：

```text
side_effect_distribution = 44
closure_bottleneck = 24
gap_heatmap_hotspot = 15
```

推荐下一步：

```text
source_restricted_or_subspace_audit = 32
readout_competition_channel_decomposition = 24
queue_driven_physical_path_fill = 15
closure_quality_probe = 12
```

### 关键结论

Phase289 把下一步机制审计从“人工选择样本”改成“图谱特征驱动选择样本”。这很重要，因为当前研究的第一优先级是完成物理分布拼图，不是凭局部直觉追闭合。

当前共享主干可以客观写成：

```text
SharedBackbone =
  continue_wins_all
  + MLP_dominant_write
  + high_side_effect_risk
  + closure_rejected
```

族差分说明：

```text
不同语言族不是完全不同机制；
它们共享 continue + MLPWrite 主干；
差异主要体现在副作用、attention 比例、行为分和缺口结构上。
```

模型差分说明：

```text
GLM4 是高风险差分模型；
qwen3 和 DS7B 更像低副作用强模型；
但三者都处于 continue 优先状态。
```

### 当前进度

只根据当前进展估计：

```text
语言模式图谱整体进度: 65%
物理分布拼图进度: 61%
大数据特征挖掘进度: 38%
复用-差分分析进度: 25%
机制审计进度: 40%
闭合进度: 20%
```

### 前端同步

```text
npm run sync:pattern-atlas:v2
Synced 1080 pattern atlas v2 files

npm run build
build passed
```

仍有 Vite 大 chunk 警告，不影响图谱数据读取。

### 下一阶段任务

下一阶段仍属于同一阶段：语言模式图谱物理分布拼图。

但执行顺序应调整为：

```text
Phase290:
  使用 phase289_feature_driven_audit_candidates.jsonl 选择样本，
  优先做 readout_competition_channel_decomposition。

Phase291:
  对 source_restricted_or_subspace_audit 候选做 GLM4 风险解耦。

Phase292:
  对 closure_quality_probe 候选做小规模闭合质量复核。
```

阶段目标不是马上闭合，而是：

```text
先画地图；
再找共享主干；
再找族差分；
再由图谱选择机制审计样本；
最后才进入闭合复核。
```

### 阶段结论

Phase288-289 完成了当前路线的关键升级：从“补图谱数据”推进到“从图谱中抽取稳定结构”。当前最可靠的客观结果是：

```text
continue 全局胜出；
MLPWrite 是共享主干；
GLM4 是高风险模型差分；
闭合仍为 0；
下一步机制审计必须由图谱特征驱动。
```

## Phase 290: 读出竞争通道分解与下一轮审计队列 [2026-07-09 08:13]

### 任务判断

本轮附件对 Phase288-289 的判断基本正确。当前研究已经不适合继续只做局部 patch 或单点闭合，而应该把 Pattern Atlas（模式图谱）中的大数据分布转化为可审计的机制对象。

Phase288-289 已经说明：

```text
continue 全局胜出；
MLPWrite 是经验共享主干；
GLM4 是高副作用风险差分模型；
闭合候选仍然没有通过四条件复核；
下一步应由图谱特征驱动机制审计。
```

但不同字段覆盖率不一致。972 条 path_signature_rows 是全图谱签名，而 component_summary_rows、causal_rows、closure_quality_rows 都是子集。因此本阶段不把统计结果解释成完整因果结论，而是先补一个固定格式的读出竞争通道分解层。

### 算法原理

本阶段没有重新运行 qwen3、GLM4、DS7B 模型，而是对现有 972 条 Pattern Atlas v2 样本做离线结构分析。每条样本被转换为：

```text
case -> readout competition -> continue channel family -> bottleneck -> audit priority
```

核心对象为：

```text
R_i = (model_i, family_i, case_i, w_i, c_i, m_i, b_i, g_i)
```

其中：

```text
w_i：读出胜出者；
c_i：top continue channel；
m_i：top_continue_vs_stop_margin；
b_i：stop / continue / target / protocol bottleneck；
g_i：缺口队列和 Phase289 特征优先级。
```

通道族映射：

```text
continue_the / continue_next_sentence -> natural_language_continue
continue_list_item -> list_structure_continue
continue_json_structure -> protocol_json_continue
continue_format -> protocol_format_continue
because / for / is -> explanation_relation_continue
comma / and -> local_syntax_continue
```

读出瓶颈判定：

```text
competition_winner != stop -> stop_not_winner
top_continue_vs_stop_margin >= -0.25 -> continue_not_suppressed
target_rank > 100 -> target_readout_weak
json / format / list 通道 -> protocol_or_structure_continue
```

### 生成的数据

新增脚本：

```text
tests/gpt5/phase290_readout_competition_channel_decomposition.py
tests/gpt5/run_phase290_readout_competition_channel_decomposition.sh
```

新增固定格式结果：

```text
tests/result/pattern_family_atlas/v2/phase290_readout_channel_rows.jsonl
tests/result/pattern_family_atlas/v2/phase290_channel_family_model_matrix.jsonl
tests/result/pattern_family_atlas/v2/phase290_stop_continue_bottleneck_rows.jsonl
tests/result/pattern_family_atlas/v2/phase290_readout_competition_audit_queue.jsonl
tests/result/pattern_family_atlas/v2/phase290_family_model_readout_summary.jsonl
tests/result/pattern_family_atlas/v2/phase290_summary.json
tests/result/pattern_family_atlas/v2/phase290_report.md
```

前端同步：

```text
npm run sync:pattern-atlas:v2
Synced 1087 pattern atlas v2 files

npm run build
build passed
```

仍有 Vite chunk size warning，不影响图谱数据读取。

### 客观结果

Phase290 统计结果：

```text
source_signature_rows = 972
readout_channel_rows = 972
channel_family_model_matrix_rows = 86
stop_continue_bottleneck_rows = 340
readout_competition_audit_queue_rows = 144
family_model_readout_summary_rows = 27
```

全图谱读出结果：

```text
global_continue_winner_rate = 1.0
global_stop_winner_rate = 0.0
global_mean_top_continue_vs_stop_margin = 8.155253
```

这说明在当前 Pattern Atlas v2 的 972 条签名中，读出竞争没有任何一条自然进入 stop winner。这个结果比前面 36 条 closure_quality 的负结果更强，因为它覆盖了完整签名层。

continue 通道族分布：

```text
natural_language_continue = 502
list_structure_continue = 334
protocol_json_continue = 70
protocol_format_continue = 66
```

top continue channel：

```text
continue_list_item = 334
continue_the = 330
continue_next_sentence = 172
continue_json_structure = 70
continue_format = 66
```

读出瓶颈计数：

```text
continue_not_suppressed = 972
stop_not_winner = 972
protocol_or_structure_continue = 470
gap_need_readout_competition = 376
target_readout_weak = 186
closure_continue_not_suppressed = 36
closure_stop_not_winner = 36
closure_rollout_not_stable = 35
closure_semantic_not_done = 3
```

下一轮审计队列：

```text
total = 144
qwen3 = 55
GLM4 = 46
DS7B = 43
```

审计类型：

```text
protocol_continue_suppression = 87
readout_channel_decomposition = 42
target_vs_continue_competition = 15
```

### 结果分析

最重要的新结果不是“闭合成功”，而是读出瓶颈被分解成了三层结构：

```text
第一层：stop 几乎完全不胜出；
第二层：continue 不是单一方向，而是多个通道族；
第三层：协议 / 列表 / JSON / 格式通道在 470 条样本中形成结构性继续压力。
```

这解释了为什么前面很多 intervention 能提高目标答案，却无法稳定停止：

```text
目标答案路径和继续通道不是同一个问题；
提高 target 并不等价于压制 continue；
压制 continue 也不能只压一个方向，必须按通道族拆解。
```

### 问题和硬伤

1. 本阶段仍然是离线图谱分析，不是新的因果干预。

```text
它证明了读出竞争分布；
没有证明哪个内部组件可以低副作用压制某个通道族。
```

2. 当前通道族分类仍是工程分类。

```text
continue_the / continue_next_sentence / continue_list_item 等分类来自 readout detail 字段；
它们很有用，但还不是真实神经元或真实子空间。
```

3. 小模型偏差必须继续保留。

```text
qwen3、GLM4、DS7B 是当前测试小模型；
内部编码可能比强模型更粗糙；
读出瓶颈可能被放大 30% 到 50%。
```

4. 线性公式仍不能闭合。

当前结果继续支持一个判断：

```text
简单线性方向无法模拟真实语言停止机制。
```

因为停止不是单个 scalar gate，而更像：

```text
target completion
+ protocol completion
+ continue channel suppression
+ stop boundary elevation
+ rollout stability
```

的联合结构。

### 理论进展

当前统一公式应继续从线性闭合公式转为机制谱图公式：

```text
LanguageMechanism = Atlas({M+(P_i)}_{i=1}^N)
```

其中 Phase290 补上的是读出竞争层：

```text
R_i =
(
W_stop,
W_continue_channel,
M_continue-stop,
TargetRank,
ProtocolPressure,
ClosureBlocker
)
```

更准确的闭合条件暂时应写成：

```text
Closed(i) =
SemanticDone(i)
AND StopWins(i)
AND ContinueSuppressed(i)
AND RolloutStable(i)
```

Phase290 说明当前最大的缺口是：

```text
StopWins(i) = false
ContinueSuppressed(i) = false
```

而不是：

```text
SemanticDone(i) = false
```

### 当前进度

只根据当前进展估计：

```text
语言模式图谱整体进度: 66%
物理分布拼图进度: 63%
大数据特征挖掘进度: 42%
读出竞争通道分解进度: 35%
机制审计进度: 41%
闭合进度: 20%
```

### 下一阶段任务

下一阶段仍然属于同一阶段：语言模式图谱的物理分布拼图。

Phase291 应按 Phase290 生成的 144 条队列继续做，但不要马上追求闭合。优先顺序：

```text
1. protocol_continue_suppression：
   分别测试 list / json / format 通道是否有不同源头。

2. target_vs_continue_competition：
   区分 target 弱和 continue 强两类失败。

3. readout_channel_decomposition：
   对 qwen3 -> GLM4 -> DS7B 顺序运行局部模型测试，
   每次只加载一个模型，避免 GPU 内存溢出。
```

### 阶段结论

Phase290 完成了一个关键拼图：把“continue 全局胜出”拆解为可观测的通道族和审计队列。

当前最可靠结论是：

```text
当前 Pattern Atlas v2 中 stop 自然胜出率为 0；
continue 不是单一机制，而是自然语言、列表、JSON、格式四类主要通道；
协议和结构继续压力是闭合失败的重要来源；
下一步不能盲目 patch，应按通道族做源头定位和低副作用压制审计。
```

## Phase 291: 扩大样本类型与大批量测试队列 [2026-07-09 08:22]

### 任务判断

用户要求继续加大样本类型和数量，改进特征分析算法，完成语言模式族图谱。这个方向正确，而且是 Phase290 之后必须做的事情。

但需要严格区分两类数据：

```text
已测图谱数据：来自模型实际行为、读出、路径、因果和闭合扫描；
待测扩样数据：只是样本和测试计划，不能当成模型结果。
```

因此本阶段先完成样本类型扩展和跨模型大队列，不把未测样本混入 path_signature_rows。

### 样本扩展原则

原始 v2 case bank：

```text
existing_cases = 1296
root_cases = 216
families = 9
old_variant_types = 6
```

原有变体：

```text
base
answer_only
explain_pressure
boundary_period
structured_json
list_pressure
```

新增 12 类变体：

```text
stop_word_hard
newline_stop
double_newline_stop
comma_continue_pressure
because_suppression
for_relation_pressure
json_closed_brace
json_no_markdown
numbered_list_one
markdown_bullet_one
quote_close
bilingual_answer_only
```

这些变体覆盖：

```text
stop_boundary
natural_language_continue
local_syntax_continue
explanation_relation_continue
protocol_json_continue
list_structure_continue
answer_boundary_continue
```

### 生成的数据

新增脚本：

```text
tests/gpt5/phase291_expanded_sample_type_and_large_batch_queue.py
tests/gpt5/run_phase291_expanded_sample_type_and_large_batch_queue.sh
```

新增固定格式数据：

```text
tests/result/pattern_family_atlas/v2/phase291_expanded_case_candidates.jsonl
tests/result/pattern_family_atlas/v2/phase291_full_model_test_plan_rows.jsonl
tests/result/pattern_family_atlas/v2/phase291_selected_large_batch_queue.jsonl
tests/result/pattern_family_atlas/v2/phase291_sample_type_coverage_rows.jsonl
tests/result/pattern_family_atlas/v2/phase291_summary.json
tests/result/pattern_family_atlas/v2/phase291_report.md
```

### 客观结果

```text
source_existing_cases = 1296
source_root_cases = 216
new_variant_types = 12
expanded_case_candidates = 2592
full_model_test_plan_rows = 7776
selected_large_batch_queue_rows = 972
sample_type_coverage_rows = 108
```

第一批大队列保持三模型均衡：

```text
qwen3 = 324
GLM4 = 324
DS7B = 324
```

九大语言族均衡：

```text
每个 family = 108
```

通道焦点覆盖：

```text
protocol_json_continue = 162
list_structure_continue = 162
explanation_relation_continue = 162
natural_language_continue = 162
local_syntax_continue = 81
stop_boundary = 162
answer_boundary_continue = 81
```

### 阶段意义

Phase291 的价值不是产生模型行为结果，而是把图谱测试从 6 类变体扩展到 18 类变体，并形成 7776 条跨模型完整测试计划。

这解决了前面一个硬伤：

```text
旧样本类型太少，容易把局部通道误认为全局规律。
```

现在新增样本能更系统地区分：

```text
停止边界；
自然语言继续；
解释继续；
列表继续；
JSON 继续；
格式继续；
引号边界；
跨语言协议。
```

### 当前限制

Phase291 是 sample expansion and queue only，没有运行模型，所以：

```text
不能把 2592 条候选样本当成已测证据；
不能更新 closure 结论；
不能说明新增变体的真实模型行为。
```

这些样本必须进入 Phase293 之后才成为真实图谱数据。

## Phase 292: 特征分析算法 v2 与覆盖率归一图谱完成度 [2026-07-09 08:22]

### 算法改进

Phase288 的特征分析以计数为主，Phase289 做复用-差分，Phase290 做读出通道分解。Phase292 在此基础上改进为 coverage-aware feature analysis（覆盖率感知特征分析）。

每个 family-model cell 现在包含：

```text
behavior_score
readout_score
rollout_score
component_coverage
causal_coverage
closure_quality_coverage
measurement_coverage
expanded_selection_rate
channel_entropy
structure_continue_rate
bottleneck_pressure
target_weak_rate
side_effect_risk_rate
gap_pressure_norm
atlas_completion_v2
```

通道熵：

```text
H = - sum(p_c log(p_c)) / log(K)
```

含义：

```text
H 高：多个 continue 通道共同竞争；
H 低：单一 continue 通道主导。
```

新版完成度不是简单平均，而是：

```text
AtlasCompletionV2 =
behavior
+ readout
+ rollout
+ measurement_coverage
+ expanded_selection_rate
+ channel_entropy
+ clean_structure_score
+ target_strength_score
+ gap_clean_score
+ side_effect_clean_score
- bottleneck_pressure
```

该公式仍是工程评分，不是最终机制公式。它的作用是排序图谱缺口，而不是证明智能理论闭合。

### 生成的数据

新增脚本：

```text
tests/gpt5/phase292_feature_analysis_algorithm_v2.py
tests/gpt5/run_phase292_feature_analysis_algorithm_v2.sh
```

新增固定格式数据：

```text
tests/result/pattern_family_atlas/v2/phase292_feature_matrix_v2_rows.jsonl
tests/result/pattern_family_atlas/v2/phase292_channel_entropy_rows.jsonl
tests/result/pattern_family_atlas/v2/phase292_coverage_normalized_gap_rows.jsonl
tests/result/pattern_family_atlas/v2/phase292_feature_priority_queue_rows.jsonl
tests/result/pattern_family_atlas/v2/phase292_global_atlas_completion.json
tests/result/pattern_family_atlas/v2/phase292_report.md
```

### 客观结果

```text
feature_matrix_rows = 27
mean_atlas_completion_v2 = 0.410425
min_atlas_completion_v2 = 0.331615
max_atlas_completion_v2 = 0.476055
mean_bottleneck_pressure = 1.0
mean_channel_entropy = 0.69222
mean_measurement_coverage = 0.410494
```

按语言族：

```text
closure = 0.415058
content_knowledge = 0.387764
cross_lingual = 0.449626
language_action = 0.397709
output_protocol = 0.419891
readout_competition = 0.395063
reasoning_constraint = 0.447451
state_drift = 0.409878
syntax_structure = 0.371386
```

按模型：

```text
DS7B = 0.417307
GLM4 = 0.405683
qwen3 = 0.408285
```

下一优先级：

```text
urgent_readout_bottleneck = 23
large_gap_physical_path_fill = 2
balanced_large_batch_measurement = 2
```

### 结果分析

新版算法给出一个更严格的判断：

```text
图谱结构框架正在变完整；
但真实机制测量覆盖率仍然低；
读出瓶颈压力仍然满格；
闭合进度不能提高。
```

mean_atlas_completion_v2 只有 0.410425，说明此前主观估计的“图谱进度 60%+”需要拆开看：

```text
样本框架和数据格式进度高；
真实机制证据进度中等偏低；
闭合证据仍然低。
```

这比单纯说“整体进度 70%”更准确。

### 当前问题和硬伤

1. 瓶颈压力仍为 1.0。

```text
说明 stop_not_winner / continue_not_suppressed 不是少数异常，而是当前图谱核心瓶颈。
```

2. measurement_coverage 只有 0.410494。

```text
说明 component、causal、closure 三类内部证据覆盖还不够。
```

3. 新增 2592 条样本尚未测量。

```text
它们提高了图谱设计覆盖率；
但没有提高机制证据覆盖率。
```

4. 当前模型仍是小模型。

```text
qwen3、GLM4、DS7B 的内部机制可能比强模型粗糙；
读出瓶颈和协议漂移可能被放大 30% 到 50%。
```

## Phase 293: 大队列跨模型执行脚本准备 [2026-07-09 08:22]

### 目的

为了避免下一步又临时写脚本，本轮补充了跨模型顺序执行脚本。它读取 Phase292 的优先队列，并按：

```text
qwen3 -> GLM4 -> DS7B
```

顺序执行，避免 GPU 内存溢出。

新增脚本：

```text
tests/gpt5/phase293_expanded_queue_behavior_readout_runner.py
tests/gpt5/run_phase293_expanded_queue_behavior_readout_runner.sh
```

脚本已通过 py_compile 语法检查。

默认完整运行：

```text
bash tests/gpt5/run_phase293_expanded_queue_behavior_readout_runner.sh 0
```

小规模烟测：

```text
bash tests/gpt5/run_phase293_expanded_queue_behavior_readout_runner.sh 5
```

### 本轮为什么没有直接跑 972 条模型测试

原因是当前请求同时包含扩样、改算法和完成图谱结构。直接启动 972 条跨模型 CUDA 测试会长时间占用 GPU，而且如果样本队列或算法字段有问题，会浪费大量时间。

本轮先完成：

```text
样本扩展；
大队列生成；
特征算法升级；
可视化数据同步；
跨模型执行脚本准备。
```

下一步才应运行 Phase293 完整大队列。

### 前端同步

```text
npm run sync:pattern-atlas:v2
Synced 1099 pattern atlas v2 files

npm run build
build passed
```

仍有 Vite chunk size warning，不影响数据读取。

### 当前进度

只根据当前进展估计：

```text
语言模式图谱整体进度: 71%
样本类型覆盖进度: 62%
大数据特征挖掘进度: 50%
物理分布拼图进度: 64%
机制审计进度: 42%
闭合进度: 20%
```

### 下一阶段任务

下一阶段仍属于同一阶段：语言模式族图谱物理分布拼图。

最优先任务：

```text
运行 Phase293 大队列：
qwen3 -> GLM4 -> DS7B
每个模型 324 条
总计 972 条
```

执行后需要生成：

```text
expanded_behavior_rows
expanded_readout_rows
expanded_path_signature_rows
expanded_gap_rows
expanded_feature_matrix_v3
```

然后再回到：

```text
component path
causal audit
closure quality
```

不能直接追闭合。

### 阶段结论

本轮完成了“加大样本类型和数量 + 改进特征分析算法”的系统升级。

最可靠结论：

```text
图谱样本设计已经从 6 类变体扩展到 18 类变体；
新增 2592 条候选样本和 7776 条跨模型完整测试计划；
第一批大测试队列为 972 条，三模型和九族均衡；
新版特征算法显示 atlas_completion_v2 均值只有 0.410425；
主要瓶颈仍然是读出竞争，不是语义完成；
下一步必须运行 Phase293 大队列，才能把扩样从计划转为真实证据。
```

## Phase 293: 扩样大队列三模型真实测量 [2026-07-09 08:38]

### 任务判断

Phase291 和 Phase292 只是完成扩样计划和特征算法升级，还没有把新增样本转为真实模型证据。因此本阶段继续执行 Phase293 大队列，按项目要求依次使用：

```text
qwen3 -> GLM4 -> DS7B
```

每次只加载一个模型，完成后释放 GPU，避免显存溢出。

### 执行情况

执行脚本：

```text
tests/gpt5/phase293_expanded_queue_behavior_readout_runner.py
tests/gpt5/run_phase293_expanded_queue_behavior_readout_runner.sh
```

先做 5 条烟测，发现一个脚本问题：底层模型加载器需要逗号字符串形式的 attn_implementations，原脚本传入 list，已修复。

随后运行完整队列：

```text
qwen3 = 324
GLM4 = 324
DS7B = 324
total = 972
```

FlashAttention2 不可用，自动回退到 sdpa。三模型均完成，无 GPU 内存溢出。

### 客观结果

qwen3：

```text
rows = 324
answer_correct_proxy_rate = 0.962963
pattern_matched_proxy_rate = 0.194444
model_stop_executed_rate = 0.0
continue_winner = 321
stop_winner = 3
mean_top_continue_vs_stop_margin = 9.308353
```

GLM4：

```text
rows = 324
answer_correct_proxy_rate = 0.953704
pattern_matched_proxy_rate = 0.367284
model_stop_executed_rate = 0.067901
continue_winner = 300
stop_winner = 24
mean_top_continue_vs_stop_margin = 4.147111
```

DS7B：

```text
rows = 324
answer_correct_proxy_rate = 0.771605
pattern_matched_proxy_rate = 0.296296
model_stop_executed_rate = 0.518519
continue_winner = 324
stop_winner = 0
mean_top_continue_vs_stop_margin = 8.31115
```

### 关键观察

扩样后最重要的现象是：

```text
语义答案正确率很高；
协议匹配率明显偏低；
模型生成层停止和读出层 stop winner 不是一回事；
continue 仍然是压倒性读出优势。
```

尤其 DS7B：

```text
model_stop_executed_rate = 0.518519
continue_winner_rate = 1.0
```

这说明一个重要分离：

```text
生成最终停止可以发生；
但最后一步读出竞争仍然可能是 continue 胜出。
```

这支持前面关于三层停止结构的判断：

```text
模型内部停止
!= 任务层完成
!= 客户端停止
!= 读出层 stop winner
```

## Phase 294: 扩样测量结果写入图谱 [2026-07-09 08:38]

### 任务目的

Phase293 的结果原始保存在测试目录中，本阶段把 972 条真实测量转换为 Pattern Atlas v2 固定格式。

新增脚本：

```text
tests/gpt5/phase294_expanded_measurement_atlas_update.py
tests/gpt5/run_phase294_expanded_measurement_atlas_update.sh
```

新增图谱数据：

```text
phase294_expanded_behavior_rows.jsonl
phase294_expanded_readout_rows.jsonl
phase294_expanded_path_signature_rows.jsonl
phase294_expanded_gap_rows.jsonl
phase294_expanded_family_model_update_rows.jsonl
phase294_cross_model_summary.json
```

### 汇总结果

```text
expanded_behavior_rows = 972
expanded_readout_rows = 972
expanded_path_signature_rows = 972
expanded_gap_rows = 972
family_model_update_rows = 27
```

全局：

```text
global_answer_correct_proxy_rate = 0.896091
global_pattern_matched_proxy_rate = 0.286008
global_model_stop_executed_rate = 0.195473
global_continue_winner_rate = 0.972222
global_stop_winner_rate = 0.027778
global_mean_top_continue_vs_stop_margin = 7.255538
```

这说明扩样后继续压倒性胜出的结论仍成立，但不再是 100%。新增边界和协议样本让少量 stop winner 出现，尤其 GLM4。

### 图谱缺口变化

虽然行为和读出层已扩展，但每条扩样样本仍然缺少：

```text
layer_path
component_path
causal_audit
closure_quality
```

因此 Phase294 把新增样本标记为：

```text
expanded_measured_partial
```

这很重要，因为它避免把 behavior/readout 层的测量误读成完整物理路径。

## Phase 295: 加入实测扩样后的特征算法 v3 [2026-07-09 08:38]

### 算法改进

Phase292 的完成度使用的是扩样计划；Phase295 改为使用 Phase294 的实测 behavior/readout。

新增脚本：

```text
tests/gpt5/phase295_feature_algorithm_v3_after_expansion.py
tests/gpt5/run_phase295_feature_algorithm_v3_after_expansion.sh
```

新增数据：

```text
phase295_feature_matrix_v3_rows.jsonl
phase295_summary.json
phase295_report.md
```

v3 评分使用：

```text
expanded_answer_correct_proxy_rate
expanded_pattern_matched_proxy_rate
expanded_model_stop_executed_rate
expanded_continue_winner_rate
expanded_stop_winner_rate
expanded_mean_top_continue_vs_stop_margin
```

并加入读出惩罚：

```text
readout_penalty =
continue_winner_rate * 0.65
+ normalized_continue_stop_margin * 0.35
```

### 客观结果

```text
feature_v3_rows = 27
mean_atlas_completion_v3 = 0.361124
mean_completion_delta_v3_minus_v2 = -0.049301
mean_expanded_answer_correct_proxy_rate = 0.896091
mean_expanded_pattern_matched_proxy_rate = 0.286008
mean_expanded_model_stop_executed_rate = 0.195473
mean_expanded_continue_winner_rate = 0.972222
```

按模型：

```text
DS7B = 0.369678
GLM4 = 0.37625
qwen3 = 0.337444
```

按语言族：

```text
closure = 0.349814
content_knowledge = 0.345609
cross_lingual = 0.384346
language_action = 0.351537
output_protocol = 0.367011
readout_competition = 0.362091
reasoning_constraint = 0.386572
state_drift = 0.366638
syntax_structure = 0.336499
```

下一优先级：

```text
hard_readout_stop_failure = 12
generation_stop_without_readout_stop = 4
protocol_pattern_failure = 6
measured_expansion_followup = 5
```

### 严格分析

完成度从 v2 的 0.410425 降到 v3 的 0.361124，不是研究倒退，而是更严格的实测校准。

原因：

```text
扩样后发现：
答案正确率高；
协议匹配率低；
continue winner 仍然高达 0.972222；
读出层停止非常少；
新增样本没有 layer/component/causal 路径证据。
```

因此图谱更真实，但闭合更难。

### 当前硬伤

1. 扩样只完成 behavior/readout。

```text
还没有完成新增样本的层路径、组件路径、因果路径。
```

2. 读出停止和生成停止分离。

```text
DS7B 有较高 model_stop_executed_rate；
但 readout competition 仍然 100% continue winner。
```

3. 协议匹配是新增大瓶颈。

```text
global_pattern_matched_proxy_rate = 0.286008
```

说明模型常常知道答案，但不能稳定遵守输出模式。

4. 小模型偏差仍然存在。

```text
当前小模型可能放大协议漂移和停止失败；
强模型可能更稳，但不应假设机制完全不同。
```

### 当前进度

只根据当前进展估计：

```text
语言模式图谱整体进度: 75%
样本类型覆盖进度: 68%
大数据特征挖掘进度: 58%
物理分布拼图进度: 67%
机制审计进度: 44%
闭合进度: 21%
```

### 前端同步

```text
npm run sync:pattern-atlas:v2
Synced 1109 pattern atlas v2 files

npm run build
build passed
```

仍有 Vite chunk size warning，不影响图谱数据读取。

### 下一阶段任务

下一阶段仍属于同一大阶段：完成语言模式族图谱的物理分布拼图。

优先级应为：

```text
Phase296:
  从 Phase294 expanded_gap_rows 中选择高价值样本，
  补 layer_path 和 component_path。

Phase297:
  对 hard_readout_stop_failure 和 protocol_pattern_failure 做组件路径审计。

Phase298:
  对 generation_stop_without_readout_stop 做停止机制分离分析。
```

不要直接追闭合。当前闭合瓶颈已经更清楚：

```text
语义完成不是主瓶颈；
协议模式不稳；
读出 stop winner 稀少；
continue 通道仍强；
生成停止与读出停止可分离。
```

### 阶段结论

本轮真正完成了“加大样本类型和数量”的实测推进：

```text
新增 972 条真实跨模型测量；
扩样样本不再只是计划；
continue winner 从 1.0 降到 0.972222，但仍压倒性占优；
GLM4 出现最多 stop winner；
DS7B 出现生成停止与读出继续分离；
新版完成度降到 0.361124，说明图谱更严格、更接近真实难度。
```

下一步要补的是新增样本的内部物理路径，而不是继续扩大行为样本或直接闭合。

## Phase 296: 扩样样本组件物理路径补全 [2026-07-09 08:43]

### 任务判断

上一阶段已经完成 972 条扩样样本的 behavior/readout 实测，但这些样本仍然缺少内部物理路径。因此本阶段不继续盲目扩大行为样本，而是把新增样本推进到 layer / attention / MLP / residual 组件路径层。

旧的 Phase275 组件路径脚本绑定 v1 case bank，不能直接处理 Phase291-294 的新增样本。因此新写 Phase296 适配版，直接读取 Phase292 优先队列中的 prompt，并对 Phase294 高缺口样本做内部组件路径追踪。

### 新增脚本

```text
tests/gpt5/phase296_expanded_component_path_probe.py
tests/gpt5/run_phase296_expanded_component_path_probe.sh
```

运行：

```text
bash tests/gpt5/run_phase296_expanded_component_path_probe.sh 9
```

即每个模型 9 条，三模型共 27 条。样本按九大语言族均衡选择。

### 输出数据

```text
phase296_component_physical_path_rows.jsonl
phase296_attention_contribution_rows.jsonl
phase296_mlp_contribution_rows.jsonl
phase296_residual_accumulation_rows.jsonl
phase296_component_summary_rows.jsonl
phase296_missing_rows.jsonl
phase296_summary.json
```

### 客观结果

```text
component_physical_path_rows = 936
attention_contribution_rows = 936
mlp_contribution_rows = 936
residual_accumulation_rows = 936
component_summary_rows = 27
missing_rows = 0
```

三模型：

```text
qwen3 = 9
GLM4 = 9
DS7B = 9
```

九大语言族：

```text
每个 family = 3
```

主导组件：

```text
MLP = 24
attention = 3
```

最终读出：

```text
continue = 27
stop = 0
```

平均正向贡献：

```text
mean_sum_positive_attn_delta = 13.229134
mean_sum_positive_mlp_delta = 27.735397
mean_sum_positive_residual_delta = 0.233892
```

模型细节：

```text
qwen3:
  MLP = 7
  attention = 2
  final_continue = 9

GLM4:
  MLP = 8
  attention = 1
  final_continue = 9

DS7B:
  MLP = 9
  final_continue = 9
```

### 关键进展

这是扩样样本第一次进入内部组件路径层。结果与前面 Phase288-290 的主结论一致：

```text
新增样本中 MLP 仍然是 continue path 的主要写入主干；
attention 也存在，但比例较小；
最终读出仍全部是 continue；
residual carry 的独立正向贡献很小。
```

这说明前面的 MLPWrite 共享主干不是只在旧样本中出现，新扩展样本也复现了这一结构。

### 严格限制

本阶段只有 27 条内部路径样本，覆盖率仍然很低：

```text
27 / 972 = 2.7778%
```

所以结论应表述为：

```text
扩样高优先级子集中，MLP 继续路径主干继续复现。
```

不能表述为：

```text
所有扩样样本都已经完成内部路径。
```

## Phase 297: 加入组件路径后的特征算法 v4 [2026-07-09 08:43]

### 算法改进

Phase295 的 v3 完成度只纳入扩样 behavior/readout。Phase297 把 Phase296 的 component path evidence（组件路径证据）加入图谱完成度。

新增脚本：

```text
tests/gpt5/phase297_feature_algorithm_v4_with_component_paths.py
tests/gpt5/run_phase297_feature_algorithm_v4_with_component_paths.sh
```

新增数据：

```text
phase297_feature_matrix_v4_rows.jsonl
phase297_summary.json
phase297_report.md
```

v4 增加字段：

```text
expanded_component_summary_rows
expanded_component_coverage
expanded_mlp_dominance_rate
expanded_attention_dominance_rate
expanded_component_continue_winner_rate
expanded_mean_sum_positive_mlp_delta
expanded_mean_sum_positive_attn_delta
component_path_score
atlas_completion_v4
```

### 客观结果

```text
feature_v4_rows = 27
mean_atlas_completion_v4 = 0.361956
mean_completion_delta_v4_minus_v3 = 0.000832
mean_expanded_component_coverage = 0.027778
mean_expanded_mlp_dominance_rate = 0.888889
mean_expanded_component_continue_winner_rate = 1.0
```

下一优先级：

```text
mlp_continue_path_causal_audit = 24
attention_route_followup = 3
```

按模型：

```text
DS7B = 0.371278
GLM4 = 0.370302
qwen3 = 0.344287
```

按语言族：

```text
closure = 0.35488
content_knowledge = 0.335059
cross_lingual = 0.382693
language_action = 0.359632
output_protocol = 0.373225
readout_competition = 0.356862
reasoning_constraint = 0.380825
state_drift = 0.371747
syntax_structure = 0.342677
```

### 结果分析

v4 完成度只比 v3 增加 0.000832，原因不是组件路径无价值，而是覆盖率太低：

```text
expanded_component_coverage = 0.027778
```

但机制方向很明确：

```text
88.8889% 的扩样内部路径由 MLP 主导；
100% 的扩样组件路径最终仍是 continue winner；
因此下一步应进入 MLP continue path causal audit，而不是继续扩大行为样本。
```

### 当前硬伤

1. 内部路径覆盖率太低。

```text
只覆盖 27 条扩样样本。
```

2. 没有新增 causal audit。

```text
Phase296 是组件路径观察，不是因果干预。
```

3. stop winner 仍没有被解释。

```text
当前组件路径主要解释 continue path；
还没有找到 stop path 的自然来源。
```

4. 小模型偏差仍然存在。

```text
当前结果可能放大 MLP 主导和协议漂移；
但跨三模型一致出现 MLP continue 主干，说明它不是单模型偶然。
```

### 当前进度

只根据当前进展估计：

```text
语言模式图谱整体进度: 76%
样本类型覆盖进度: 68%
大数据特征挖掘进度: 62%
物理分布拼图进度: 70%
机制审计进度: 47%
闭合进度: 21%
```

### 前端同步

```text
npm run sync:pattern-atlas:v2
Synced 1120 pattern atlas v2 files

npm run build
build passed
```

仍有 Vite chunk size warning，不影响图谱数据读取。

### 下一阶段任务

下一阶段仍属于语言模式族图谱的物理分布拼图。

最优先：

```text
Phase298:
  对 Phase296 中 24 条 MLP 主导样本做低副作用 causal audit。

Phase299:
  对 3 条 attention 主导样本做 attention route followup。

Phase300:
  汇总 behavior/readout/component/causal 四层图谱，形成 Pattern Atlas v3 候选。
```

### 阶段结论

本轮完成了从“扩样行为/读出证据”到“扩样内部组件路径证据”的关键推进。

最可靠结论：

```text
扩样样本中 MLP continue path 继续强复现；
27 条内部路径样本全部最终 continue；
MLP 主导占 24/27；
下一步应做 MLP continue path 的因果审计；
继续单纯扩大行为样本的边际收益已经降低。
```

## Phase 298: 扩样 MLP 续写路径低副作用因果审计 [2026-07-09 08:51]

### 任务背景

用户要求继续加大样本类型和数量，改进特征分析算法，完成语言模式族图谱。

根据 Phase296/297 的客观结果，继续单纯扩大行为样本的边际收益已经下降；当前最关键缺口不是更多输出样本，而是对已经定位出的 24 条 MLP 主导 continuation path（续写路径）进行因果审计。

因此本阶段没有盲目扩样，而是把样本数量加到关键瓶颈上：

```text
Phase296 MLP 主导样本: 24 条
每条样本 patch 类型: 3 个
总因果审计行: 72 行
模型顺序: qwen3 -> GLM4 -> DS7B
```

### 测试脚本

```text
tests/gpt5/phase298_expanded_mlp_continue_causal_audit.py
tests/gpt5/run_phase298_expanded_mlp_continue_causal_audit.sh
```

### 输出文件

```text
tests/result/phase298_expanded_mlp_continue_causal_audit/expanded_mlp_continue_causal_audit/
tests/result/pattern_family_atlas/v2/phase298_mlp_causal_audit_rows.jsonl
tests/result/pattern_family_atlas/v2/phase298_causal_effect_rows.jsonl
tests/result/pattern_family_atlas/v2/phase298_rollout_rows.jsonl
tests/result/pattern_family_atlas/v2/phase298_cross_model_summary.json
```

### 算法原理

对 Phase296 中 MLP 主导样本，读取其 strongest_mlp_layer（最强 MLP 层），在该层最后 token 的 MLP 输出上做缩放干预：

```text
mlp_zero_last_token: scale = 0.0
mlp_quarter_last_token: scale = 0.25
mlp_half_last_token: scale = 0.5
```

基础公式：

$$
h_l = h_{l-1} + Attn_l(h_{l-1}) + MLP_l(h_{l-1})
$$

干预公式：

$$
h_l^{patch} = h_{l-1} + Attn_l(h_{l-1}) + \alpha \cdot MLP_l(h_{l-1})
$$

其中：

$$
\alpha \in \{0, 0.25, 0.5\}
$$

读出差分：

$$
\Delta M_{continue-stop}
= M_{continue-stop}^{patch} - M_{continue-stop}^{base}
$$

判定标准：

```text
如果 Delta continue-stop margin < -1.0:
  说明 MLP 干预明显削弱 continuation readout，记为 weak causal support。

如果 winner 发生翻转:
  说明 MLP 干预改变读出胜者，记为 strong causal support。
```

### 客观结果

全局结果：

```text
selected_mlp_dominant_cases: 24
audit_rows: 72
causal_effect_rows: 72
rollout_rows: 72
missing_rows: 0
patch_counts:
  mlp_zero_last_token: 24
  mlp_quarter_last_token: 24
  mlp_half_last_token: 24
necessity_supported:
  True: 48
  False: 24
winner_changed:
  False: 72
causal_support_level:
  weak: 48
  not_supported: 24
rollout_changed:
  True: 46
  False: 26
mean_delta_continue_stop_margin: -2.700304
mean_delta_target_logit: -1.579861
```

分模型结果：

```text
qwen3:
  MLP dominant cases: 7
  audit_rows: 21
  necessity_supported: 20/21
  winner_changed: 0/21
  rollout_changed: 15/21
  mean_delta_continue_stop_margin: -5.222470

GLM4:
  MLP dominant cases: 8
  audit_rows: 24
  necessity_supported: 8/24
  winner_changed: 0/24
  rollout_changed: 20/24
  mean_delta_continue_stop_margin: -0.682292

DS7B:
  MLP dominant cases: 9
  audit_rows: 27
  necessity_supported: 20/27
  winner_changed: 0/27
  rollout_changed: 11/27
  mean_delta_continue_stop_margin: -2.532407
```

分语言模式族平均 Delta：

```text
closure: -3.774306
content_knowledge: -2.520833
cross_lingual: -4.435764
language_action: -1.494792
output_protocol: -1.809028
readout_competition: -2.177083
reasoning_constraint: -2.843750
state_drift: -2.984375
syntax_structure: -1.968750
```

### 结果分析

本阶段最重要的正结果：

```text
MLP patch 可以系统性降低 continuation margin。
```

48/72 个干预行支持 MLP 对 continuation readout 有必要性，而且三模型均出现负向 Delta。

但最重要的负结果同样清楚：

```text
没有任何一个 patch 导致 winner 翻转。
```

这说明当前 MLP 路径是 continuation pressure（续写压力）的重要组成部分，但不是完整控制开关。它更像底层连续写入通道，而不是最终停止/继续决策器。

### 硬伤和瓶颈

1. 仍不是闭合。

```text
winner_changed = 0/72
```

说明 MLP 缩放只能削弱 continuation margin，不能直接控制最终读出胜者。

2. GLM4 的读出效应弱。

GLM4 mean_delta_continue_stop_margin 只有 -0.682292，但 rollout_changed 达到 20/24，说明：

```text
读出 margin 变化和实际生成轨迹变化不是同一层证据。
```

3. stop source 仍未定位。

当前因果审计主要围绕 MLP continuation path，仍没有找到自然 stop path 的物理来源。

4. 当前 patch 仍是线性缩放。

线性缩放可能不符合真实机制，所以不能用它反复 patch 追求闭合。

### 阶段结论

Phase298 正确推进了语言模式图谱的物理分布拼图：

```text
MLP continuation path:
  已有跨模型、跨语言模式族、弱因果支持。

完整停止/继续闭合:
  未完成。
```

## Phase 299: 特征分析算法 v5 引入因果审计证据 [2026-07-09 08:51]

### 任务背景

Phase297 的特征算法 v4 已经加入组件路径证据，但缺少因果证据。

Phase299 将 Phase298 的 MLP causal audit（因果审计）加入 cell-level（单元级）特征矩阵。

### 测试脚本

```text
tests/gpt5/phase299_feature_algorithm_v5_with_causal_audit.py
tests/gpt5/run_phase299_feature_algorithm_v5_with_causal_audit.sh
```

### 输出文件

```text
tests/result/pattern_family_atlas/v2/phase299_feature_matrix_v5_rows.jsonl
tests/result/pattern_family_atlas/v2/phase299_summary.json
tests/result/pattern_family_atlas/v2/phase299_feature_algorithm_v5_report.md
```

### 算法改进

v5 不再只看行为和组件观察，而加入：

```text
causal_case_count
causal_support_rate
causal_strong_rate
mean_delta_continue_stop_margin
```

特征合成公式：

$$
CausalScore =
0.45 \cdot Coverage
+ 0.30 \cdot SupportRate
+ 0.15 \cdot StrongRate
+ 0.10 \cdot NormDelta
$$

其中：

$$
NormDelta = \min \left( \frac{|\Delta M_{continue-stop}|}{8}, 1 \right)
$$

图谱完成度更新：

$$
AtlasCompletion_{v5}
= clamp(AtlasCompletion_{v4} + 0.07 \cdot CausalScore)
$$

### 客观结果

```text
input_cells: 27
feature_cells: 27
phase298_audit_rows: 72
phase298_selected_mlp_dominant_cases: 24
mean_atlas_completion_v5: 0.386058
mean_completion_delta_v5_minus_v4: 0.024103
mean_expanded_mlp_causal_support_rate: 0.592593
mean_expanded_mlp_causal_strong_rate: 0.0
mean_expanded_mlp_causal_score: 0.344326
```

下一优先级分布：

```text
weak_causal_path_expand: 18
mlp_continue_path_causal_audit: 3
side_effect_or_noncausal_path_recheck: 6
```

分模型完成度：

```text
deepseek7b: 0.400050
GLM4: 0.386562
qwen3: 0.371563
```

### 结果分析

v5 的提升很小：

```text
mean delta: +0.024103
```

这是合理的，因为 Phase298 只提供 weak causal support，没有 strong causal support。

这一点非常重要：

```text
当前算法没有把弱证据包装成强闭合。
```

### 当前进度

```text
language_pattern_family_atlas: 78%
sample_type_coverage: 70%
large_data_feature_mining: 66%
physical_distribution_puzzle: 72%
mechanism_causal_audit: 50%
closure: 21%
```

### 阶段结论

Phase299 改进了特征分析算法，但没有虚增闭合。

最可靠结论：

```text
MLP continuation path 已经有弱因果证据；
但强因果证据仍为 0；
下一步不能继续只做线性 MLP patch。
```

## Phase 300: 语言模式族图谱 v3 候选合成 [2026-07-09 08:51]

### 任务背景

用户要求系统完成语言模式族图谱，不要每次只做小功能。

因此 Phase300 将已有四层证据合成为 Pattern Atlas v3 candidate（模式图谱第三版候选）：

```text
行为层: Phase294
读出竞争层: Phase294
组件路径层: Phase296
因果审计层: Phase298
特征矩阵层: Phase299
```

### 脚本

```text
tests/gpt5/phase300_pattern_atlas_v3_candidate_synthesis.py
tests/gpt5/run_phase300_pattern_atlas_v3_candidate_synthesis.sh
```

### 输出文件

```text
tests/result/pattern_family_atlas/v2/phase300_pattern_atlas_v3_cell_rows.jsonl
tests/result/pattern_family_atlas/v2/phase300_pattern_atlas_v3_node_rows.jsonl
tests/result/pattern_family_atlas/v2/phase300_pattern_atlas_v3_edge_rows.jsonl
tests/result/pattern_family_atlas/v2/phase300_pattern_atlas_v3_summary.json
```

### 合成公式

单元证据完整度：

$$
EvidenceCompleteness =
\frac{
I_{behavior}+I_{readout}+I_{component}+I_{causal}
}{4}
$$

物理路径置信度：

$$
PhysicalPathConfidence =
0.30 \cdot EvidenceCompleteness
+ 0.25 \cdot MLPDominance
+ 0.15 \cdot AttentionDominance
+ 0.20 \cdot CausalSupport
+ 0.10 \cdot (1 - WinnerFlip)
$$

闭合缺口：

$$
ClosureGap =
1 -
\left(
0.25 \cdot PatternScore
+ 0.20 \cdot StopRate
+ 0.25 \cdot WinnerFlip
+ 0.30 \cdot EvidenceCompleteness
\right)
$$

### 客观结果

```text
cell_rows: 27
node_rows: 12
edge_rows: 27
behavior_rows: 972
readout_rows: 972
component_rows: 27
causal_rows: 72
mean_atlas_completion_v5: 0.386058
mean_physical_path_confidence: 0.749074
mean_closure_gap: 0.597737
```

图谱状态：

```text
partial_physical_path: 27/27
```

下一优先级：

```text
search_stop_source_path: 15
fill_component_or_causal_evidence: 3
stronger_causal_intervention_design: 7
protocol_pattern_failure_analysis: 2
```

### 结果分析

本阶段完成了一个重要系统升级：

```text
语言模式族图谱已经从散乱阶段结果，进入 cell/node/edge 可视化图谱结构。
```

但它没有完成闭合：

```text
mean_closure_gap: 0.597737
winner flip: 0
stop source path: 未定位
```

所以当前最准确说法是：

```text
语言模式族图谱 v3 candidate 已形成；
物理路径拼图已经有较强轮廓；
机制闭合仍远未完成。
```

### 前端同步

```text
npm run sync:pattern-atlas:v2
Synced 1134 pattern atlas v2 files

npm run build
build passed
```

仍有 Vite chunk size warning，不影响数据读取。

### 当前进度

只根据当前进展估计：

```text
语言模式族图谱整体进度: 80%
样本类型覆盖进度: 70%
大数据特征挖掘进度: 68%
物理分布拼图进度: 74%
机制因果审计进度: 52%
闭合进度: 21%
```

### 严格审视

硬伤：

1. 当前因果证据只有 weak support。

```text
winner_changed = 0/72
```

2. 当前图谱主要解释 continuation path。

```text
stop source path 仍没有自然物理来源。
```

3. 组件路径样本仍少。

```text
behavior/readout: 972 rows
component: 27 rows
causal: 72 rows
```

4. 当前 patch 是线性缩放，可能不符合真实非线性运行机制。

5. 当前测试模型是小模型，内部结构粗糙可能导致 30%-50% 偏差。

### 智能理论角度的关键洞察

语言模式族图谱目前显示：

```text
语言不是单一语义向量；
也不是单一停止符号；
而是多个模式族在读出层竞争，
并由 MLP/Attention/Residual 等物理路径共同塑造输出轨迹。
```

当前最重要的第一性原理不是闭合公式，而是：

```text
先找到模式族在网络中的物理分布；
再找到模式族之间如何竞争、复用、漂移；
最后才尝试闭合。
```

### 下一阶段大任务

下一阶段仍属于同一大阶段：完成语言模式图谱的物理分布拼图。

不要继续只做单点 patch，应进入系统化任务：

```text
Phase301-305:
  stop source path 搜索。
  目标：找到与停止、边界、答案完成、协议结束相关的自然物理路径。

Phase306-310:
  扩大 component path coverage。
  目标：把 27 条组件路径扩展到至少 81 条，覆盖 9 个模式族 x 3 模型 x 多 variant。

Phase311-315:
  非线性因果干预设计。
  目标：不要只缩放 MLP，而是测试层间组合、attention+MLP 联合路径、窗口位置路径。

Phase316-320:
  Pattern Atlas v4。
  目标：把 behavior/readout/component/causal/stop-source 五层证据合成新版图谱。
```

### 阶段结论

本轮完成了用户要求中的核心部分：

```text
加大了关键样本数量；
完成了跨 qwen3、GLM4、DS7B 的顺序 CUDA 测试；
改进了特征分析算法；
生成了固定格式图谱数据；
合成了语言模式族图谱 v3 candidate；
前端数据已同步并构建通过。
```

但还不能说完成语言编码机制破解。

当前最可靠结论：

```text
语言模式族图谱已经形成可视化候选；
MLP continuation path 是稳定物理主干之一；
stop path 和非线性闭合机制仍是最大瓶颈；
下一阶段第一优先级是 stop source path，而不是继续线性 patch。
```

## Phase 301: 语义复用-差分图谱样本库 [2026-07-09 14:45]

### 对附件判断的分析

附件中关于 Phase290-300 的总体判断基本正确。

正确部分：

```text
当前路线已经从机制优先，转向大数据图谱特征优先；
Pattern Atlas v3 candidate 已形成，但不是闭合；
MLP continuation path 只有弱因果支持，不能当作完整控制机制；
stop source path、非线性机制和跨模型泛化仍是最大瓶颈；
语义复用-差分机制应该成为独立重点方向。
```

需要修正的部分：

```text
语义复用-差分不能只靠理论特征表；
必须做跨模型行为/读出实测；
并且要防止“两个对象都答不好，所以表现相似”被误判为高复用。
```

因此本轮把附件建议落成一个新子图谱方向：

```text
Semantic Reuse-Delta Atlas（语义复用-差分图谱）
```

它仍属于当前大阶段：

```text
第一优先级：完成语言模式图谱的物理分布拼图；
第二优先级：在高质量图谱基础上尝试闭合。
```

### 脚本

```text
tests/gpt5/phase301_semantic_reuse_delta_case_bank.py
tests/gpt5/run_phase301_semantic_reuse_delta_case_bank.sh
```

### 输出文件

```text
tests/result/pattern_family_atlas/v2/phase301_semantic_object_rows.jsonl
tests/result/pattern_family_atlas/v2/phase301_semantic_attribute_case_rows.jsonl
tests/result/pattern_family_atlas/v2/phase301_semantic_contrast_case_rows.jsonl
tests/result/pattern_family_atlas/v2/phase301_semantic_full_test_plan_rows.jsonl
tests/result/pattern_family_atlas/v2/phase301_semantic_reuse_delta_case_bank_summary.json
```

### 样本设计

对象库：

```text
20 objects:
  15 个水果；
  2 个非水果食物/植物对象；
  3 个非食物控制对象。
```

属性任务：

```text
category
subclass
color
shape
taste
texture
part
use
```

对比任务：

```text
shared backbone:
  orange-lemon, lemon-lime, apple-pear, strawberry-blueberry, banana-mango

difference:
  apple-banana, orange-lemon, banana-apple, fruit-chair
```

### 客观结果

```text
object_rows: 20
attribute_case_rows: 160
contrast_case_rows: 9
full_test_plan_rows: 507
models: qwen3, GLM4, DS7B
```

### 核心公式

对象语义复用-差分公式：

$$
\mathrm{Sem}(x \mid c)
=
S_{\mathrm{shared}}(x)
+ \Delta_{\mathrm{id}}(x)
+ \Delta_{\mathrm{attr}}(x \mid c)
+ \Delta_{\mathrm{rel}}(x \mid c)
+ \Delta_{\mathrm{use}}(x \mid c)
$$

水果共享主干：

$$
S_{\mathrm{fruit}}
=
S_{\mathrm{entity}}
+ S_{\mathrm{plant}}
+ S_{\mathrm{food}}
+ S_{\mathrm{fruit-category}}
$$

水果差分：

$$
\Delta_{\mathrm{fruit}}(x)
=
\Delta_{\mathrm{color}}(x)
+ \Delta_{\mathrm{shape}}(x)
+ \Delta_{\mathrm{taste}}(x)
+ \Delta_{\mathrm{texture}}(x)
+ \Delta_{\mathrm{subclass}}(x)
+ \Delta_{\mathrm{use}}(x)
$$

### 阶段结论

Phase301 完成了语义复用-差分子图谱的固定格式样本库。

它不是模型测试，也不是理论闭合，而是为 Phase302/303 的实测和矩阵图谱提供对象、属性、对比三类基础数据。

## Phase 302: 语义复用-差分跨模型行为/读出实测 [2026-07-09 14:45]

### 脚本

```text
tests/gpt5/phase302_semantic_reuse_delta_behavior_readout_runner.py
tests/gpt5/run_phase302_semantic_reuse_delta_behavior_readout.sh
```

### 输出文件

```text
tests/result/pattern_family_atlas/v2/phase302_semantic_behavior_rows.jsonl
tests/result/pattern_family_atlas/v2/phase302_semantic_readout_rows.jsonl
tests/result/pattern_family_atlas/v2/phase302_semantic_rollout_rows.jsonl
tests/result/pattern_family_atlas/v2/phase302_cross_model_summary.json
```

### 测试设置

```text
qwen3: 169 rows
GLM4: 169 rows
DS7B: 169 rows
total: 507 rows
missing_rows: 0
```

三个模型按顺序在 CUDA 上运行，避免显存叠加。

### 客观结果

全局：

```text
behavior_rows: 507
readout_rows: 507
rollout_rows: 507
missing_rows: 0
answer_correct_proxy_rate: 0.325444
pattern_matched_proxy_rate: 0.307692
model_stop_executed_rate: 0.001972
competition_winner:
  continue: 418
  stop: 89
```

分模型：

```text
qwen3:
  answer_correct_proxy_rate: 0.325444
  pattern_matched_proxy_rate: 0.301775
  model_stop_executed_rate: 0.0
  competition_winner:
    continue: 169

GLM4:
  answer_correct_proxy_rate: 0.473373
  pattern_matched_proxy_rate: 0.443787
  model_stop_executed_rate: 0.0
  competition_winner:
    stop: 74
    continue: 95

DS7B:
  answer_correct_proxy_rate: 0.177515
  pattern_matched_proxy_rate: 0.177515
  model_stop_executed_rate: 0.005917
  competition_winner:
    continue: 154
    stop: 15
```

属性成功率：

```text
category: 0.766667
color: 0.516667
taste: 0.500000
part: 0.333333
subclass: 0.200000
shared contrast: 0.200000
texture: 0.183333
use: 0.100000
shape: 0.083333
difference contrast: 0.083333
```

### 结果分析

最可靠正结果：

```text
共享类别主干最强；
颜色和味道属性相对稳定；
用途、形状、对比差分明显更弱。
```

这支持附件中的核心判断：

```text
语义不是单一对象向量；
而是共享主干 + 属性差分 + 上下文路由。
```

最重要的负结果：

```text
difference contrast 成功率只有 0.083333；
use 成功率只有 0.100000；
shape 成功率只有 0.083333。
```

这说明对象差分和关系/用途差分比类别主干更难稳定激活。

另一个重要现象：

```text
GLM4 出现 74/169 stop winner；
但 model_stop_executed_rate = 0。
```

这再次说明：

```text
读出停止 != 生成停止。
```

### 硬伤

1. 别名表仍偏窄。

例如 shape、use、texture 任务可能有多个合理答案，但当前评分只覆盖少量目标词。

2. 小模型偏差明显。

DS7B 在语义属性任务上 target rank 很差，可能是模型能力和提示格式共同造成。

3. 当前仍是行为/读出层。

没有定位语义属性在 attention、MLP、residual 中的物理路径。

## Phase 303: 语义复用-差分矩阵图谱 [2026-07-09 14:45]

### 脚本

```text
tests/gpt5/phase303_semantic_reuse_delta_atlas.py
tests/gpt5/run_phase303_semantic_reuse_delta_atlas.sh
```

### 输出文件

```text
tests/result/pattern_family_atlas/v2/phase303_semantic_object_summary_rows.jsonl
tests/result/pattern_family_atlas/v2/phase303_semantic_reuse_matrix_rows.jsonl
tests/result/pattern_family_atlas/v2/phase303_semantic_delta_matrix_rows.jsonl
tests/result/pattern_family_atlas/v2/phase303_semantic_attribute_path_rows.jsonl
tests/result/pattern_family_atlas/v2/phase303_semantic_family_cluster_rows.jsonl
tests/result/pattern_family_atlas/v2/phase303_semantic_reuse_delta_atlas_summary.json
```

### 结果

```text
object_summary_rows: 20
reuse_matrix_rows: 190
delta_matrix_rows: 190
attribute_path_rows: 30
cluster_rows: 10
behavior_rows: 507
readout_rows: 507
mean_attribute_success_rate: 0.321667
mean_measured_reuse_score: 0.697758
mean_theoretical_reuse_score: 0.247345
mean_delta_score: 0.549969
high_reuse_pair_count: 25
high_delta_pair_count: 32
```

自然出现的高复用对：

```text
lemon-lime
orange-lemon
orange-lime
strawberry-blueberry
banana-mango
banana-pineapple
apple-pear
```

自然出现的高差分对：

```text
fruit / non-fruit controls:
  blueberry-stone
  lemon-chair
  lemon-knife
  strawberry-stone
  grape-chair
  grape-knife
  orange-stone
```

### 问题发现

Phase303 暴露了一个算法硬伤：

```text
measured_reuse_score 太粗；
如果两个对象都答不好，它们的答题表现也可能相似；
这会造成假高复用。
```

例如：

```text
pear-watermelon
banana-blueberry
```

这类 pair 不应该因为答题表现相似就被过度解释为共享语义主干。

因此需要立刻进入 Phase304 修正算法。

## Phase 304: 语义复用-差分算法 v2 修正 [2026-07-09 14:45]

### 脚本

```text
tests/gpt5/phase304_semantic_reuse_delta_algorithm_v2.py
tests/gpt5/run_phase304_semantic_reuse_delta_algorithm_v2.sh
```

### 输出文件

```text
tests/result/pattern_family_atlas/v2/phase304_semantic_reuse_matrix_v2_rows.jsonl
tests/result/pattern_family_atlas/v2/phase304_semantic_reuse_false_high_audit_rows.jsonl
tests/result/pattern_family_atlas/v2/phase304_semantic_reuse_delta_algorithm_v2_summary.json
```

### 算法修正

v1 的粗公式：

$$
Reuse_{v1}
=
0.55 \cdot Reuse_{theory}
+ 0.45 \cdot Reuse_{measured}
$$

问题是 measured reuse 会受低质量答题共同失败影响。

v2 引入 evidence quality（证据质量）：

$$
Q(x,y)
=
\frac{
Success(x)+Success(y)
}{2}
$$

动态权重：

$$
w_{measured}
=
0.20 + 0.25 \cdot Q(x,y)
$$

$$
w_{theory}
=
1 - w_{measured}
$$

类别/子类约束：

$$
Reuse_{v2}
=
w_{theory} \cdot Reuse_{theory}
+ w_{measured} \cdot Reuse_{measured}
+ B_{category}
+ B_{subclass}
$$

其中：

$$
B_{category} =
\begin{cases}
0.10, & same category \\
-0.10, & different category
\end{cases}
$$

$$
B_{subclass} =
\begin{cases}
0.12, & same subclass \\
0, & otherwise
\end{cases}
$$

差分：

$$
Delta_{v2}
=
1 - Reuse_{v2}
$$

### 客观结果

```text
input_reuse_rows: 190
corrected_reuse_rows: 190
audit_rows: 5
mean_reuse_v1: 0.450031
mean_corrected_reuse_v2: 0.391870
mean_corrected_delta_v2: 0.608130
likely_shared_backbone_count: 41
semantic_relation_counts:
  category_shared_backbone: 87
  subclass_shared_backbone: 14
  contrast_control: 84
  ambiguous_or_needs_more_evidence: 5
```

修正后的 top reuse 更合理：

```text
lemon-lime: 0.915689
orange-lime: 0.840676
mango-pineapple: 0.838937
peach-cherry: 0.818398
orange-lemon: 0.804938
strawberry-blueberry: 0.799011
banana-mango: 0.763360
banana-pineapple: 0.757697
apple-pear: 0.726202
mango-kiwi: 0.723466
```

### 分析

Phase304 说明：

```text
语义复用不能只用行为相似度；
必须加入对象特征、类别/子类约束和证据质量。
```

这和当前大路线一致：

```text
不要轻易用高级统计包装结果；
先把客观拼图和明显错误校准做好。
```

### 当前进度

只根据当前进展估计：

```text
语言模式族图谱整体进度: 81%
语义复用-差分子图谱进度: 38%
样本类型覆盖进度: 72%
大数据特征挖掘进度: 71%
物理分布拼图进度: 75%
机制因果审计进度: 52%
闭合进度: 21%
```

### 前端同步

```text
npm run sync:pattern-atlas:v2
Synced 1155 pattern atlas v2 files

npm run build
build passed
```

仍有 Vite chunk size warning，不影响数据读取。

### 严格结论

本轮完成了一个新的语义子图谱雏形：

```text
水果 / 非水果对象库；
属性任务库；
对比任务库；
三模型行为/读出实测；
语义复用矩阵；
语义差分矩阵；
语义簇；
复用算法 v2 修正。
```

最可靠结论：

```text
category 主干最稳定；
color/taste 属性较稳定；
shape/use/contrast difference 差分很弱；
柑橘、浆果、热带、树果等子类主干能在修正算法后自然显现；
语义复用-差分方向值得继续推进。
```

不能过度解释的地方：

```text
当前是 behavior/readout semantic atlas；
还不是 internal semantic component path atlas；
不能说已经破解水果语义在网络内部的物理路径。
```

### 下一阶段任务

下一阶段仍属于同一大阶段。

建议任务：

```text
Phase305:
  扩大语义别名表和属性答案集合，降低 shape/use/texture 的假阴性。

Phase306:
  对 high shared backbone pairs 做内部组件路径探针。
  重点对象：lemon-lime, orange-lemon, apple-pear, banana-mango。

Phase307:
  对 high delta control pairs 做内部组件路径探针。
  重点对象：fruit/non-fruit, citrus/tool, berry/mineral。

Phase308:
  把语义子图谱接回 Pattern Atlas v4。
```

### 阶段结论

附件提出的“水果之间如何复用和差分”方向是正确的。

但它应该按当前研究原则推进：

```text
先建语义图谱；
再做实测校准；
再做内部路径；
最后才谈机制闭合。
```

本轮已经完成前两步，并完成了第一版算法校准。

## Phase 305: 内部语义物理路径图谱定位 [2026-07-09 16:11]

### 对附件判断的分析

附件关于 Phase301-304 的判断总体正确：

```text
Phase301-304 已经把研究从输出路径、继续通道、停止瓶颈，
推进到语义知识网络的复用-差分结构。
```

最关键的正确点是：

```text
当前已经不能只看对象行为相似、答案读出相似、复用矩阵和差分矩阵；
必须进入内部语义物理路径定位。
```

但需要谨慎修正：

```text
附件提出 object token / query token / answer readout 三位置追踪是正确目标；
本阶段先完成 answer readout position（答案读出位置）的层/组件语义路径定位；
object token 和 query token 路径还没有完成。
```

因此本阶段是：

```text
internal semantic physical path probe v1
内部语义物理路径探针第一版
```

不是完整语义编码闭合。

### 脚本

```text
tests/gpt5/phase305_internal_semantic_physical_path_probe.py
tests/gpt5/run_phase305_internal_semantic_physical_path_probe.sh
```

### 输出文件

```text
tests/result/pattern_family_atlas/v2/phase305_semantic_component_rows.jsonl
tests/result/pattern_family_atlas/v2/phase305_semantic_component_summary_rows.jsonl
tests/result/pattern_family_atlas/v2/phase305_cross_model_summary.json
```

### 测试设置

从 Phase304 的语义复用-差分图谱中选择 high shared backbone（高共享主干）和 contrast control（对比控制）相关样本。

每个模型：

```text
12 semantic cases
```

三模型：

```text
qwen3 -> GLM4 -> DS7B
```

总计：

```text
semantic_component_summary_rows: 36
semantic_component_rows: 1248
missing_rows: 0
```

### 算法原理

对每条语义样本构造 target semantic group（目标语义组）和 distractor semantic group（干扰语义组）。

例如：

```text
category:
  target = fruit
  distractor = vegetable / tool / furniture / mineral

color:
  target = yellow
  distractor = red / green / orange / blue / purple

taste:
  target = sour
  distractor = sweet / bitter / starchy
```

语义读出边界：

$$
M_{\mathrm{semantic}}
=
\max z(\mathrm{target\ group})
-
\max z(\mathrm{distractor\ group})
$$

每层分解：

$$
h_l
=
h_{l-1}
+ A_l
+ M_l
+ R_l
$$

其中：

```text
A_l: attention output
M_l: MLP output
R_l: residual carry / unmodeled carry
```

组件贡献：

$$
\Delta_{\mathrm{attn}}^{(l)}
=
M_{\mathrm{semantic}}(h_{l-1}+A_l)
-
M_{\mathrm{semantic}}(h_{l-1})
$$

$$
\Delta_{\mathrm{mlp}}^{(l)}
=
M_{\mathrm{semantic}}(h_{l-1}+A_l+M_l)
-
M_{\mathrm{semantic}}(h_{l-1}+A_l)
$$

$$
\Delta_{\mathrm{residual}}^{(l)}
=
M_{\mathrm{semantic}}(h_l)
-
M_{\mathrm{semantic}}(h_{l-1}+A_l+M_l)
$$

### 客观结果

全局：

```text
semantic_component_rows: 1248
semantic_component_summary_rows: 36
missing_rows: 0
final_semantic_winner:
  target: 21
  distractor: 15
dominant_positive_semantic_component:
  MLP: 23
  attention: 13
mean_final_semantic_margin: 1.243056
mean_sum_positive_attn_semantic_delta: 22.174643
mean_sum_positive_mlp_semantic_delta: 18.608308
mean_sum_positive_residual_semantic_delta: 0.189424
```

分模型：

```text
qwen3:
  target: 8
  distractor: 4
  dominant component:
    attention: 10
    MLP: 2
  mean_final_semantic_margin: 1.583333
  mean_attn_delta: 44.893168
  mean_mlp_delta: 23.318015

GLM4:
  target: 9
  distractor: 3
  dominant component:
    MLP: 10
    attention: 2
  mean_final_semantic_margin: 2.411458
  mean_attn_delta: 17.396070
  mean_mlp_delta: 21.280244

DS7B:
  target: 4
  distractor: 8
  dominant component:
    MLP: 11
    attention: 1
  mean_final_semantic_margin: -0.265625
  mean_attn_delta: 4.234691
  mean_mlp_delta: 11.226664
```

### 结果分析

最重要的进展：

```text
语义路径和 continuation path 不完全相同。
```

此前 continuation path 中 MLP 是更稳定的主干。但 Phase305 显示：

```text
qwen3 的语义路径更偏 attention；
GLM4 / DS7B 的语义路径更偏 MLP；
residual 正向贡献很小。
```

这说明：

```text
语义知识网络可能更依赖 attention route + MLP write 的组合；
不能直接把 MLP continuation path 当作语义编码主干。
```

### 硬伤

1. 只追踪 last position。

```text
object token position 未追踪；
query token position 未追踪。
```

2. 没有因果审计。

```text
当前是 observational component attribution；
不是 causal patch。
```

3. distractor set 仍手工构造。

语义候选集合仍需要更完整的 alias / distractor 表。

4. 样本量仍小。

```text
内部语义路径样本: 36
行为/读出语义样本: 507
```

### 阶段结论

Phase305 完成了内部语义物理路径第一版定位。

当前可靠结论：

```text
语义路径已经出现可测的 attention/MLP 分解；
类别、颜色、味道等语义边界可在层组件上追踪；
但还没有完成对象 token / 查询 token / 答案读出三位置路径闭合。
```

## Phase 306: 语义物理路径子图谱更新 [2026-07-09 16:11]

### 脚本

```text
tests/gpt5/phase306_semantic_physical_path_atlas_update.py
tests/gpt5/run_phase306_semantic_physical_path_atlas_update.sh
```

### 输出文件

```text
tests/result/pattern_family_atlas/v2/phase306_semantic_physical_path_cell_rows.jsonl
tests/result/pattern_family_atlas/v2/phase306_semantic_attribute_atlas_rows.jsonl
tests/result/pattern_family_atlas/v2/phase306_semantic_physical_path_atlas_summary.json
```

### 合成目标

把 Phase305 的 36 条内部路径样本合成：

```text
model-attribute path cell
attribute-level semantic atlas
```

### 评分公式

语义物理路径分数：

$$
Score_{\mathrm{semantic\ path}}
=
0.30 \cdot TargetWinnerRate
+ 0.25 \cdot NormAttnDelta
+ 0.25 \cdot NormMLPDelta
+ 0.20 \cdot BehaviorSuccess
$$

其中：

$$
NormAttnDelta = \min \left( \frac{\Delta_{\mathrm{attn}}^+}{25}, 1 \right)
$$

$$
NormMLPDelta = \min \left( \frac{\Delta_{\mathrm{mlp}}^+}{25}, 1 \right)
$$

### 客观结果

```text
semantic_component_rows: 1248
semantic_component_summary_rows: 36
semantic_path_cell_rows: 21
semantic_attribute_atlas_rows: 7
mean_semantic_physical_path_score: 0.563390
mean_final_target_winner_rate: 0.547619
dominant_component:
  MLP: 15
  attention: 6
```

属性级下一步：

```text
expand_object_query_position_trace: 4
contrast_delta_path_followup: 1
expand_alias_and_distractor_calibration: 2
```

模型-属性局部结果：

```text
category:
  qwen3 score 0.970000, dominant attention
  GLM4 score 0.948372, dominant MLP
  DS7B score 0.409645, dominant MLP

color:
  qwen3 score 0.837062, dominant MLP
  GLM4 score 0.825491, dominant MLP
  DS7B score 0.520230, dominant MLP

taste:
  qwen3 score 0.900000, dominant attention
  GLM4 score 0.816100, dominant attention
  DS7B score 0.270120, dominant MLP

use:
  qwen3 score 0.510000, final target winner 0.0
  GLM4 score 0.309976, final target winner 0.0
  DS7B score 0.128306, final target winner 0.0
```

### 结果分析

最可靠的语义路径拼图：

```text
category / color / taste 是当前最强语义物理路径候选；
use / shared / subclass 在部分模型上仍弱；
difference 对比差分仍不稳定；
DS7B 的语义 target winner 明显弱于 qwen3 和 GLM4。
```

一个关键洞察：

```text
语义共享主干和对象差分并不等价。
```

category 是共享主干，路径最稳定；
color/taste 是属性差分，部分稳定；
use/difference 是关系和对比差分，明显更弱。

这和 Phase302 行为层结果一致，说明不是单纯读出噪声。

### 当前进度

只根据当前进展估计：

```text
语言模式族图谱整体进度: 82%
语义复用-差分子图谱进度: 43%
语义内部物理路径进度: 22%
样本类型覆盖进度: 72%
大数据特征挖掘进度: 72%
物理分布拼图进度: 76%
机制因果审计进度: 52%
闭合进度: 21%
```

### 前端同步

```text
npm run sync:pattern-atlas:v2
Synced 1163 pattern atlas v2 files

npm run build
build passed
```

仍有 Vite chunk size warning，不影响图谱数据读取。

### 严格结论

本轮完成了附件提出的下一步核心任务的一部分：

```text
从语义行为/读出图谱
推进到内部语义组件路径图谱。
```

但还没有完成附件中的完整目标：

```text
object token path: 未完成
query token path: 未完成
answer readout path: 已完成第一版
shared backbone subspace: 未完成
attribute delta subspace: 未完成
causal validation: 未完成
```

### 下一阶段任务

下一阶段仍属于同一大阶段，应继续自动推进：

```text
Phase307:
  object/query/last 三位置语义路径追踪。

Phase308:
  high shared backbone pairs 的对象 token 复用路径定位。
  重点: lemon-lime, orange-lemon, apple-pear, banana-mango。

Phase309:
  high delta control pairs 的差分路径定位。
  重点: fruit/non-fruit, citrus/tool, berry/mineral。

Phase310:
  语义子图谱接回 Pattern Atlas v4。
```

### 阶段结论

附件方向正确，但必须谨慎推进。

当前最可靠结论：

```text
语义物理路径已能在 attention/MLP 层组件中观察；
category/color/taste 的路径最清楚；
use/difference/shared/subclass 仍需要更强别名表和三位置追踪；
语义编码机制还没有闭合，但物理分布拼图已经开始进入内部层级。
```

## Phase 307: 三位置语义路径追踪 [2026-07-09 16:28]

### 任务判断

附件中对 Phase305/306 的判断基本正确。Phase305/306 已经把语义复用-差分方向推进到内部物理路径层，但它主要观察 answer readout position（答案读出位置），还没有回答一个关键问题：

```text
语义关系是在 object token（对象词元）写入，
在 query token（查询词元）重组，
还是只在 last token（最后词元）读出？
```

因此继续做三位置追踪是合理的，不应直接进入闭合。当前线性闭合公式仍不足以模拟真实机制，第一优先级仍是完成语言模式图谱的物理分布拼图。

### 测试脚本和数据

新增脚本：

```text
tests/gpt5/phase307_three_position_semantic_path_trace.py
tests/gpt5/run_phase307_three_position_semantic_path_trace.sh
```

测试模型按顺序执行：

```text
qwen3 -> GLM4 -> DS7B
```

输出文件：

```text
tests/result/pattern_family_atlas/v2/phase307_three_position_component_rows.jsonl
tests/result/pattern_family_atlas/v2/phase307_three_position_summary_rows.jsonl
tests/result/pattern_family_atlas/v2/phase307_cross_model_summary.json
```

### 测试原理

本阶段保持 Phase305/306 的 target（目标对象）和 distractor（干扰对象）语义边界，但把位置拆成三类：

```text
object position: 对象词元位置
query position: 查询词元位置
last position: 答案读出前最后位置
```

核心观测量为组件投影边际：

$$
M_{c,p,l}^{attn}
=
\left\langle A_{c,p,l}, v_{target} - v_{distractor} \right\rangle
$$

$$
M_{c,p,l}^{mlp}
=
\left\langle F_{c,p,l}, v_{target} - v_{distractor} \right\rangle
$$

$$
M_{c,p,l}^{resid}
=
\left\langle R_{c,p,l}, v_{target} - v_{distractor} \right\rangle
$$

其中：

```text
c: case（测试样本）
p: position（位置）
l: layer（层）
A: attention 输出
F: MLP 输出
R: residual hidden state（残差隐藏态）
v_target - v_distractor: 目标-干扰读出差分方向
```

三位置路径可写成：

$$
\mathcal{P}_{semantic}(x)
=
\left[
T_{object}(x),
T_{query}(x),
T_{last}(x)
\right]
$$

其中每个位置的局部路径为：

$$
T_p(x)
=
\left\{
M_{p,l}^{attn},
M_{p,l}^{mlp},
M_{p,l}^{resid}
\right\}_{l=1}^{L}
$$

### 客观结果

总量：

```text
three_position_component_rows: 3744
three_position_summary_rows: 108
missing_rows: 0
position_counts:
  object: 36
  query: 36
  last: 36
attribute_counts:
  category: 18
  color: 18
  difference: 18
  shared: 18
  subclass: 18
  taste: 9
  use: 9
dominant_component_counts:
  MLP: 90
  attention: 18
```

三位置平均组件强度：

```text
position_mean_attn_delta:
  last: 22.174643
  object: 9.083975
  query: 9.057919

position_mean_mlp_delta:
  query: 19.453018
  object: 19.224558
  last: 18.608308

position_mean_residual_delta:
  last: 0.189424
  object: 0.189018
  query: 0.168942
```

跨模型摘要：

```text
qwen3:
  rows: 36
  final_target_winner_count: 18
  final_distractor_winner_count: 18
  dominant_component_counts: MLP 24 / attention 12
  mean_attn_delta: 21.796431
  mean_mlp_delta: 22.802338

GLM4:
  rows: 36
  final_target_winner_count: 20
  final_distractor_winner_count: 16
  dominant_component_counts: MLP 33 / attention 3
  mean_attn_delta: 11.983748
  mean_mlp_delta: 18.663619

DS7B:
  rows: 36
  final_target_winner_count: 12
  final_distractor_winner_count: 24
  dominant_component_counts: MLP 33 / attention 3
  mean_attn_delta: 6.536358
  mean_mlp_delta: 15.819926
```

### 结果分析

最稳定的客观现象：

```text
object/query 位置明显偏 MLP-dominant（MLP 主导）；
last 位置 attention 强度明显上升；
qwen3 的 last position attention 路径最突出；
GLM4 和 DS7B 更偏 MLP 全路径；
DS7B 的 final winner 明显偏 distractor，说明小模型语义边界更粗糙。
```

这说明语义路径不是简单的单点读出，而更像：

```text
对象/查询阶段: MLP 写入或重组语义特征；
最后读出阶段: attention 可能参与答案路由或关系选择；
最终输出: residual/logit readout 汇合。
```

但这仍然不是闭合，只是把语义模式图谱从 answer readout position 扩展到 object/query/last 三位置物理分布。

### 问题和硬伤

1. object/query 位置通过 tokenizer subsequence（词元子序列）定位，复杂分词情况下可能有误差。
2. 当前只做观测追踪，没有做三位置 causal patch（因果补丁）。
3. 样本仍集中于 Phase301 的语义对象库，数量比行为图谱小。
4. target-distractor 差分方向仍是线性投影，可能无法模拟真实非线性运行机制。
5. 小模型内部编码可能粗糙，尤其 DS7B 的 distractor 偏置不能直接外推到大模型语言机制。

### 阶段结论

Phase307 正确推进了语义物理路径图谱：

```text
语义关系不是只在最后词元出现；
object/query/last 三位置都有可测组件差异；
object/query 更像语义写入或局部重组；
last 更像答案路由或读出聚合；
MLP 是主干，attention 在最后位置可能承担路由角色。
```

## Phase 308: 三位置语义图谱合成 [2026-07-09 16:28]

### 任务判断

Phase307 产生的是逐层、逐组件、逐位置的原始观测数据。为了让语言模式图谱客户端和后续研究使用，需要把结果合成为固定格式的 atlas cell（图谱单元）和 route row（路径行）。因此 Phase308 属于同一阶段的必要自动延续。

### 新增脚本和输出

新增脚本：

```text
tests/gpt5/phase308_three_position_semantic_atlas_update.py
tests/gpt5/run_phase308_three_position_semantic_atlas_update.sh
```

输出文件：

```text
tests/result/pattern_family_atlas/v2/phase308_three_position_semantic_position_cell_rows.jsonl
tests/result/pattern_family_atlas/v2/phase308_three_position_semantic_attribute_cell_rows.jsonl
tests/result/pattern_family_atlas/v2/phase308_three_position_semantic_route_rows.jsonl
tests/result/pattern_family_atlas/v2/phase308_three_position_semantic_atlas_summary.json
```

并同步到可视化客户端：

```text
frontend/public/vis_data/pattern_family_atlas/v2/
```

客户端构建结果：

```text
npm run sync:pattern-atlas:v2: success
npm run build: success
```

### 合成公式

位置路径得分：

$$
S_{model,position}
=
\alpha \cdot W_{target}
+ \beta \cdot D_{component}
+ \gamma \cdot R_{confidence}
+ \delta \cdot C_{coverage}
$$

其中：

```text
W_target: target winner rate（目标胜出率）
D_component: dominant component strength（主导组件强度）
R_confidence: readout confidence（读出置信）
C_coverage: coverage completeness（覆盖完整度）
```

模型路径类型：

$$
Route(model)
=
\arg\max_{component}
\left[
S_{object},
S_{query},
S_{last}
\right]
$$

本阶段只作为图谱合成指标，不作为闭合公式。

### 客观结果

总体结果：

```text
three_position_component_rows: 3744
three_position_summary_rows: 108
position_cell_rows: 9
attribute_position_cell_rows: 21
route_rows: 3
dominant_component_counts:
  MLP: 8
  attention: 1
mean_position_path_score: 0.599125
route_type_counts:
  mlp->mlp->mlp: 2
  mlp->mlp->attention: 1
```

模型级位置图谱：

```text
qwen3:
  object: MLP, score 0.635106
  query: MLP, score 0.593230
  last: attention, score 0.874949
  route_type: mlp->mlp->attention

GLM4:
  object: MLP, score 0.547169
  query: MLP, score 0.559889
  last: MLP, score 0.785116
  route_type: mlp->mlp->mlp

DS7B:
  object: MLP, score 0.550492
  query: MLP, score 0.463139
  last: MLP, score 0.383038
  route_type: mlp->mlp->mlp
```

属性级现象：

```text
category:
  last attention target_rate: 0.833333
  object MLP target_rate: 1.000000
  query MLP target_rate: 0.333333

color:
  last attention target_rate: 1.000000
  object MLP target_rate: 0.833333
  query MLP target_rate: 0.666667

difference:
  last attention target_rate: 0.500000
  object MLP target_rate: 0.666667
  query MLP target_rate: 0.000000
```

### 图谱进展

更新后的进度估计：

```text
language_pattern_family_atlas: 0.83
semantic_reuse_delta_subatlas: 0.46
semantic_internal_physical_path: 0.30
sample_type_coverage: 0.72
feature_mining_coverage: 0.73
physical_distribution_coverage: 0.77
causal_audit_coverage: 0.52
closure_validation: 0.21
```

这说明当前真正完成的是：

```text
语言模式图谱的物理分布拼图继续推进；
语义子图谱从行为/读出进入三位置内部路径；
闭合验证仍然较低，不应提前宣称完成机制公式。
```

### 理论进展

当前更合理的语言模式族物理路径表述为：

$$
\mathcal{G}_{language}
=
\left(
\mathcal{P}_{syntax},
\mathcal{P}_{semantic},
\mathcal{P}_{reasoning},
\mathcal{P}_{stop},
\mathcal{E}_{reuse},
\mathcal{E}_{delta}
\right)
$$

其中语义路径暂时应写成分布式路径，而不是单线性公式：

$$
\mathcal{P}_{semantic}
=
\left[
\mathcal{W}_{object}^{MLP},
\mathcal{W}_{query}^{MLP},
\mathcal{R}_{last}^{Attention/MLP},
\mathcal{O}_{logit}
\right]
$$

解释：

```text
object MLP: 对象语义写入
query MLP: 查询条件重组
last attention/MLP: 答案路由或最终语义聚合
logit output: 输出读出
```

这比“语义向量”或“单一方向”更接近真实现象，但仍然只是物理分布图谱，不是完整运行机制。

### 严格审视

当前结果的可信部分：

```text
三位置差异稳定存在；
MLP 是 object/query 的主干；
last position 的 attention 作用在 qwen3 中明显；
不同小模型的路径形态不同，说明机制有模型结构依赖。
```

当前不能得出的结论：

```text
不能说 MLP 就是语义本体；
不能说 attention 就是语义路由的唯一机制；
不能说三位置路径已经闭合；
不能用当前线性投影公式代替真实运行机制。
```

### 下一阶段任务

下一步仍属于同一阶段，应继续自动推进，但目标应从“三位置定位”转向“语义复用-差分对子路径”：

```text
Phase309:
  high shared backbone pairs 的 object/query/last 三位置复用路径定位。
  重点对象:
    lemon-lime
    orange-lemon
    apple-pear
    banana-mango

Phase310:
  high delta control pairs 的 object/query/last 差分路径定位。
  重点对象:
    fruit/non-fruit
    citrus/tool
    berry/mineral

Phase311:
  把 shared backbone path 和 delta path 合并为语义子图谱 v3。
```

### 阶段结论

Phase308 完成了 Phase307 的图谱化合成。当前语言模式图谱不再只是行为层图谱，也不只是最后读出层图谱，而开始形成：

```text
行为结果
-> 读出竞争
-> 内部组件
-> token position 物理路径
-> 语义复用/差分子图谱
```

总体判断：

```text
方向正确；
结果不是闭合；
物理分布拼图继续推进；
下一步应继续做 shared backbone 和 delta control 的三位置路径，而不是回到单点 patch。
```

## Phase 309: 共享主干与差分对象对的三位置路径图谱 [2026-07-09 16:54]

### 任务判断

附件对 Phase307/308 的判断基本正确。Phase307/308 已经证明语义路径不是只在 last token（最后词元）读出，而是存在 object/query/last 三位置结构。但它仍有一个关键缺口：

```text
单对象三位置路径已经完成；
对象对之间的 shared backbone（共享主干）和 delta control（差分控制）是否真的在内部路径上可区分，还没有完成。
```

因此本阶段继续推进到对象对路径矩阵，而不是追闭合。这个选择符合当前总路线：

```text
第一优先级: 完成各种语言模式族的物理分布拼图；
第二优先级: 在高质量物理路径基础上尝试闭合。
```

### 新增脚本

新增测试脚本：

```text
tests/gpt5/phase309_pair_three_position_reuse_delta_path_atlas.py
tests/gpt5/run_phase309_pair_three_position_reuse_delta_path_atlas.sh
```

结果保存到新规则目录：

```text
tests/gpt5/result/phase309_pair_three_position_reuse_delta_path_atlas/
tests/gpt5/result/pattern_family_atlas/v2/
```

同时为了兼容当前可视化客户端，也同步了一份到：

```text
tests/result/pattern_family_atlas/v2/
frontend/public/vis_data/pattern_family_atlas/v2/
```

前端同步与构建：

```text
npm run sync:pattern-atlas:v2: success
npm run build: success
```

构建只有 chunk size warning（包体积警告），不影响图谱数据读取。

### 测试设计

测试对象对分两类。

高复用对象对：

```text
lemon-lime
orange-lemon
apple-pear
banana-mango
strawberry-blueberry
```

高差分对象对：

```text
apple-banana
fruit-chair
lemon-knife
blueberry-stone
orange-stone
```

属性类型：

```text
category
subclass
color
taste
use
```

测试规模：

```text
models: qwen3, GLM4, DS7B
cases_per_model: 100
total_object_attribute_traces: 300
three_position_summary_rows: 900
component_rows: 31200
pair_path_rows: 150
pair_matrix_rows: 1350
missing_rows: 0
```

模型按顺序执行：

```text
qwen3 -> GLM4 -> DS7B
```

没有并行加载多个模型，避免 GPU 内存溢出。

### 算法原理

本阶段把 Phase307 的单对象路径：

$$
\mathcal{P}_{semantic}(x)
=
\left[
T_{object}(x),
T_{query}(x),
T_{last}(x)
\right]
$$

升级为对象对路径比较：

$$
\mathcal{P}_{pair}(x,y)
=
\left[
Sim(T_{object}(x),T_{object}(y)),
Sim(T_{query}(x),T_{query}(y)),
Sim(T_{last}(x),T_{last}(y))
\right]
$$

其中每个位置继续拆成 attention/MLP/residual 三类组件：

$$
T_p(x)
=
\left\{
M_{p,l}^{attn},
M_{p,l}^{mlp},
M_{p,l}^{resid}
\right\}_{l=1}^{L}
$$

路径复用指标：

$$
Reuse_{path}(x,y,p,c)
=
\frac{
\cos(T_{p,c}(x),T_{p,c}(y)) + 1
}{2}
$$

路径差分指标：

$$
Delta_{path}(x,y,p,c)
=
1 - Reuse_{path}(x,y,p,c)
$$

这里使用的是逐层组件 margin profile（边际轨迹）的相似度，不是最终机制公式。它的意义是基础物理分布探针：

```text
如果高复用对象对的路径相似度显著高于高差分对象对，
说明 shared backbone / delta control 不只是行为标签，
而是在内部三位置路径上也有可观察差异。
```

### Token Span 改进

本阶段补充了 Phase307/308 的一个硬伤：token 位置定位信息不足。

新增字段：

```text
token_start
token_end
token_match_confidence
match_surface
multi_token_pooling_method
```

正式批次结果：

```text
token_match_confidence_mean: 0.8
```

这说明大部分 object/query/last 定位可用，但仍不是完美，需要后续继续改进多词元 span pooling。

### 客观结果

总体结果：

```text
component_rows: 31200
summary_rows: 900
pair_path_rows: 150
pair_matrix_rows: 1350
component_similarity_rows: 1350
missing_rows: 0
```

核心复用/差分结果：

```text
shared_backbone mean reuse: 0.702623
delta_control mean reuse: 0.563324

shared_backbone mean delta: 0.297377
delta_control mean delta: 0.436676
```

分模型结果：

```text
qwen3:
  shared_backbone reuse: 0.901764
  delta_control reuse: 0.755106

GLM4:
  shared_backbone reuse: 0.838703
  delta_control reuse: 0.680402

DS7B:
  shared_backbone reuse: 0.830350
  delta_control reuse: 0.690642
```

三个模型都出现同向结果：

```text
shared_backbone path reuse > delta_control path reuse
```

这是本阶段最重要的正结果。

### 位置结构

按位置统计：

```text
overall:
  object reuse: 0.551451
  query reuse: 0.666259
  last reuse: 0.681211

shared_backbone:
  object reuse: 0.609287
  query reuse: 0.739696
  last reuse: 0.758886

delta_control:
  object reuse: 0.493614
  query reuse: 0.592822
  last reuse: 0.603535
```

这说明：

```text
shared/delta 的可区分性不是只在 object 入口；
query 和 last 位置反而更清楚；
对象对语义复用更可能在查询路由和最终读出聚合阶段表现出来。
```

### 属性结构

按属性统计：

```text
shared_backbone:
  category reuse: 0.751802
  subclass reuse: 0.768946
  color reuse: 0.590974
  taste reuse: 0.745104
  use reuse: 0.656290

delta_control:
  category reuse: 0.622357
  subclass reuse: 0.558913
  color reuse: 0.587033
  taste reuse: 0.497986
  use reuse: 0.550330
```

最清楚的区分在：

```text
subclass: shared 0.768946 vs delta 0.558913
taste: shared 0.745104 vs delta 0.497986
category: shared 0.751802 vs delta 0.622357
```

color 区分较弱：

```text
shared color reuse: 0.590974
delta color reuse: 0.587033
```

这提示颜色可能更像表层属性或对象具体属性，不一定稳定承载共享主干。

### 组件结构

按组件统计：

```text
attention reuse: 0.707727
MLP reuse: 0.688073
residual reuse: 0.503120
```

本阶段与 Phase307/308 相比出现一个重要校准：

```text
单对象路径中 object/query 偏 MLP；
对象对复用矩阵中 attention 与 MLP 都强，attention 略高；
residual 明显更弱。
```

这说明对象对复用不是简单的“MLP 写入强度相同”，而可能还包含 attention 对查询路由、对象-属性绑定、上下文聚合的复用。

### 路径类型

高频 route type：

```text
attention->attention->attention: 33
attention->mlp->attention: 22
attention->attention->mlp: 12
mlp->mlp->attention: 10
mlp->attention->attention: 9
mlp->attention->mlp: 8
residual->mlp->mlp: 8
```

这进一步说明：

```text
对象对路径复用的主导组件与单对象目标读出路径不同；
复用/差分图谱不能直接套用 Phase307 的单对象 MLP 主干结论；
attention 在 shared/delta pair path 中更重要。
```

### 图谱进展

更新后的进度估计：

```text
language_pattern_family_atlas: 0.84
semantic_reuse_delta_subatlas: 0.52
semantic_internal_physical_path: 0.36
pair_path_distribution_coverage: 0.34
sample_type_coverage: 0.74
feature_mining_coverage: 0.76
physical_distribution_coverage: 0.79
causal_audit_coverage: 0.52
closure_validation: 0.21
```

解释：

```text
语义复用-差分子图谱从 0.46 提升到 0.52；
语义内部物理路径从 0.30 提升到 0.36；
闭合没有明显提升，因为本阶段仍是观测路径图谱，不是因果闭合。
```

### 正确结论

本阶段可以谨慎得出的结论：

```text
shared_backbone 对象对在三模型中均表现出更高路径复用；
delta_control 对象对表现出更高路径差分；
query/last 位置比 object 位置更能体现对象对复用结构；
subclass/category/taste 比 color/use 更能区分共享主干；
attention 与 MLP 都参与对象对路径复用，attention 在本阶段略强；
DS7B 虽然语义边界粗糙，但仍保留 shared > delta 的方向。
```

不能得出的结论：

```text
不能说 shared backbone 已经闭合；
不能说 attention 就是共享主干本体；
不能说当前 cosine profile 就是真实机制；
不能把小模型路径直接外推为大模型或大脑机制。
```

### 问题和硬伤

1. 当前相似度比较的是逐层 margin profile，不是隐藏态完整几何结构。
2. target-distractor 仍是线性读出探针，真实机制可能是非线性、多子空间、门控竞争结构。
3. token_match_confidence_mean 为 0.8，仍有 20% 左右定位不够理想。
4. 本阶段对象库仍偏水果/非水果控制，知识网络覆盖不足。
5. 本阶段主要用单对象属性 prompt 做 pair path 比较，还没有系统测试 contrast prompt 本身。
6. 没有做 causal patch，因此只能说“路径复用可观察”，不能说“路径复用因果必要”。

### 智能理论洞察

本阶段补上了一个重要拼图：

```text
知识网络不是对象静态向量集合，
而是对象之间在 object/query/last 三位置和 attention/MLP 组件上的复用-差分路径网络。
```

更接近当前观测的表达是：

$$
KnowledgeNetwork
=
ObjectEntry
+ QueryRoute
+ SharedBackbonePath
+ DeltaPath
+ ReadoutCompetition
$$

其中：

```text
ObjectEntry: 对象入口；
QueryRoute: 查询路由；
SharedBackbonePath: 共享主干路径；
DeltaPath: 差分路径；
ReadoutCompetition: 读出竞争。
```

这说明语言的语义系统可能不是“概念向量表”，而是：

```text
对象-属性-关系在不同位置和组件中的动态路径复用网络。
```

### 下一阶段任务

下一步仍属于同一大阶段，应该继续自动推进，但不应马上闭合。

Phase310 建议：

```text
Phase310: contrast prompt shared/difference path atlas
```

核心目标：

```text
把 Phase309 的单对象属性 path comparison，
推进到 contrast prompt 本身：

An lemon and a lime are both ___.
Compared with a knife, a lemon is more associated with ___.
```

需要回答：

```text
shared prompt 是否显式激活 shared backbone path？
difference prompt 是否显式激活 delta path？
query position 是否是 shared/difference 路由分叉点？
last position 是否是答案聚合点？
```

Phase311 建议：

```text
把 single-object pair path、contrast prompt path、Phase303/304 reuse-delta matrix 合并为 Semantic Reuse-Delta Atlas v3。
```

### 阶段结论

Phase309 是一个重要正结果，但不是闭合。

最重要的客观发现是：

```text
三个模型中，shared_backbone 对象对的三位置路径复用均高于 delta_control 对象对。
```

这说明当前“语言是模式网络 / 语义是复用-差分路径网络”的路线继续得到支持，但仍必须谨慎：

```text
当前完成的是物理分布拼图；
不是因果证明；
不是最终机制公式；
更不是 AGI 理论闭合。
```

## Phase 310: 当前研究总审计与语言物理图谱完成路线 [2026-07-09 17:04]

### 一、任务范围

本阶段对以下三类材料进行交叉审计：

```text
1. GPT 路线 Phase195-309；
2. GLM 路线最新的语义坐标、边界齿轮和通道因果结果；
3. IntelligentTheory.md 中的语言编码机制、语言模式图谱和智能理论框架。
```

目标不是再提出一个新理论名词，也不是运行一轮局部模型测试，而是回答：

```text
当前真正完成了哪些拼图？
哪些只是工程覆盖或观测相关？
哪些已经有因果证据？
距离语言模式族物理分布图谱、语言编码机制和智能理论分别还有多远？
下一阶段怎样系统推进，而不是继续逐个小补丁？
```

本阶段没有重新运行 qwen3、GLM4、DS7B。原因是当前任务是证据审计、理论校准和阶段设计，没有新的实验假设需要立即加载模型。后续模型测试仍必须按 qwen3 -> GLM4 -> DS7B 顺序单模型执行。

### 二、总判断

当前研究方向总体正确，最重要的路线升级已经完成：

```text
概念神经元/单一语义方向
-> 动态语言模式
-> 状态写入和读出竞争
-> 层与组件路径
-> 因果审计
-> 语言模式族物理路径图谱
-> 语义复用-差分路径图谱
```

当前最可靠的总判断是：

```text
语言编码不是静态概念向量表；
也不是一个神经元对应一个语义或语法规则；
更接近上下文条件下的相对状态、路径路由、组件更新、候选竞争和生成门共同构成的动态路径网络。
```

但是，当前还不能说已经破解语言编码机制，更不能说智能理论已经完成。现阶段已经形成了较成熟的研究基础设施和一批重要物理分布拼图，但完整因果链、自然门控、跨模型统一性和可预测性仍明显不足。

### 三、已经完成的核心成果

#### 1. 研究对象完成了正确升级

早期研究主要寻找：

```text
语义向量；
概念神经元；
停止方向；
某一层或某一通道的最大贡献。
```

大量正负结果已经证明，这些对象不足以描述真实运行机制。当前研究对象已经升级为完整路径：

$$
PatternPath(x,m)
=
[
Trigger,
State,
Route,
ComponentUpdate,
ReadoutCompetition,
ProtocolGate,
Rollout,
Closure
]
$$

这个升级是目前最重要的方法论成果。它避免把“可读出”误认为“因果节点”，也避免把“首词元获胜”误认为“自然生成闭合”。

#### 2. 九大语言模式族和统一样本框架已经建立

当前工作分类包括：

```text
content_knowledge（内容知识）
output_protocol（输出协议）
reasoning_constraint（推理约束）
syntax_structure（语法结构）
language_action（语言动作）
cross_lingual（跨语言）
readout_competition（读出竞争）
state_drift（状态漂移）
closure（闭合）
```

Phase265 建立了 1296 条统一设计样本，每个模式族 144 条；Phase266、Phase293/294 等阶段完成了九族三模型的行为和读出扫描。Phase273 以后又建立了固定 schema（数据模式）、case detail（样本详情）、cell/node/edge（单元/节点/边）和前端数据同步系统。

因此可以确认：

```text
语言模式族图谱已经不再只是研究设想；
它已经具备样本库、执行脚本、结果格式、缺口队列和可视化数据结构。
```

#### 3. 行为结果和读出竞争已经形成较大规模基线

九族扩样批次的客观规模：

```text
behavior rows: 972
readout rows: 972
family-model cells: 27
global answer-correct proxy: 0.896091
global pattern-matched proxy: 0.286008
global continue winner rate: 0.972222
global stop winner rate: 0.027778
```

这组结果说明一个稳定事实：

```text
模型经常已经知道答案，但输出模式、协议完成和停止执行仍然失败。
```

因此下列区分已经得到充分支持：

```text
答案正确 != 输出模式正确；
语义完成 != 停止获胜；
句号出现 != EOS 动作；
首词元闭合 != 自然生成闭合；
模型内部停止 != 客户端停止。
```

#### 4. 继续路径已经出现稳定的组件轮廓

Phase288 汇总了 196 条组件摘要和 392 条历史因果记录，观察到：

```text
global MLP dominance rate: 0.943878
global attention dominance rate: 0.056122
global closure closed count: 0
```

Phase296 的均衡九族代表样本进一步得到：

```text
27 个 family-model 组件样本；
24 个 MLP 主导；
3 个 attention 主导；
27 个最终仍由 continue 获胜。
```

可以谨慎确认：

```text
MLP 是当前小模型 continue path（继续路径）的稳定写入主干之一；
attention 更可能参与检索、路由、格式和边界聚合；
但不同层存在补偿路径，单层最大贡献不等于最小因果机制。
```

#### 5. 因果审计已经否定了“单点闭合”的简单解释

Phase298 对 24 个 MLP 主导样本进行了 72 次干预：

```text
weak necessity support: 48 / 72
not supported: 24 / 72
winner changed: 0 / 72
strong causal support rate: 0
```

这不是“实验失败”，而是一个重要负结果：

```text
压低 MLP continue margin 可以改变轨迹；
但不足以让 stop winner 翻转；
说明完整机制包含补偿、竞争者替代、协议场和自然生成门。
```

GLM 路线补充了更细的局部证据：

```text
GLM4 L39 存在 EOS-vs-a 的有符号边界齿轮候选；
标点阻塞更像公共齿轮骨架 + case residual（样本残差）；
qwen3 的部分 color/function MLP 通道组通过了激活加权和随机组控制；
GLM4 存在明显通用边界敏感性；
DS7B 有坐标集中，但不稳定转化为正向边界因果效应。
```

这些结果支持“局部齿轮存在”，但还没有找到自然门控，也没有完成 strict-clean rollout（严格干净自然生成）。

#### 6. 语义知识网络已经从行为图谱进入内部路径图谱

Phase301-304 建立了 20 个对象、8 类属性和对比关系的语义样本库，并完成 507 条三模型行为/读出测试。属性成功率显示：

```text
category: 0.766667
color: 0.516667
taste: 0.500000
part: 0.333333
shared: 0.200000
subclass: 0.200000
use: 0.100000
difference: 0.083333
shape: 0.083333
```

这说明类别、颜色、味道比功能、差异和形状更容易被当前小模型与当前模板稳定读出。

Phase305-308 又把语义路径扩展到 object/query/last（对象/查询/最后读出）三位置：

```text
object/query 多数为 MLP 主导；
last 位置 attention 贡献明显增加；
qwen3 路径为 MLP -> MLP -> attention；
GLM4、DS7B 为 MLP -> MLP -> MLP。
```

Phase309 的规模为：

```text
10 个对象对；
5 类属性；
3 个模型；
150 个独立 pair-attribute-model 路径单元；
900 条三位置摘要；
31200 条逐层组件记录；
missing rows: 0。
```

核心结果：

```text
shared-backbone reuse: 0.702623
delta-control reuse: 0.563324

qwen3: 0.901764 > 0.755106
GLM4: 0.838703 > 0.680402
DS7B: 0.830350 > 0.690642
```

三个模型都出现：

```text
共享主干对象对的路径复用 > 差分控制对象对的路径复用。
```

这支持“知识网络包含复用路径和差分路径”的研究方向。query/last 的区分比 object 更清楚，attention 与 MLP 都参与对象对路径复用。

#### 7. 数据系统和可视化基础已经形成

当前已经具备：

```text
统一 JSON/JSONL 输出；
模型、语言族、模式、变体、路径、证据和缺口字段；
case detail 按需读取；
固定前端同步流程；
图谱 summary、cell、node、edge 和 route 数据；
自动构建和前端查看入口。
```

这是后续大规模研究能够积累而不是反复推翻的必要条件。

### 四、当前最合理的语言编码机制框架

当前理论中最可信的主体不应再改名。可以保持为：

```text
语言是动态模式网络；
编码的基本单位是条件化物理路径，而不是孤立向量或神经元。
```

模型的精确前向过程仍是：

$$
H_{l+1}=T_l(H_l)
$$

$$
P(y_{t+1}\mid x,y_{\le t})
=
softmax\left(W_U N(h_{L,t})\right)
$$

但要解释语言机制，需要在这个前向过程中识别因果子图。当前最合适的机制分解为：

$$
\mathcal{G}_{language}
=
(V_{state},V_{route},V_{component},V_{boundary},V_{gate},E)
$$

其中：

```text
V_state: 对象、角色、关系、构式、操作符、作用域和完成状态；
V_route: 检索、绑定、语法、推理、格式、继续、停止等路线；
V_component: attention、MLP、residual、norm、W_U 等组件事件；
V_boundary: target、wrong、echo、prose、format、continue、stop 的竞争边界；
V_gate: 协议门、自然触发门、生成门和停止门；
E: 自然运行或受控干预下可验证的路径边。
```

当前语言输出机制可以暂时写成组织公式：

$$
LanguageOutput
=
Readout\Big(
GenerationGate\big(
BoundaryCompetition\big(
ComponentUpdate\big(
Route(StateEncode(x))
\big)\big)\big)\Big)
$$

必须强调：这只是待验证的机制图组织公式，不是已经闭合的数学定律。

当前语义知识网络的局部拼图可以写成：

$$
\mathcal{P}_{semantic}(o,r)
=
[
ObjectEntry,
QueryRoute,
SharedPath,
DeltaPath,
ReadoutCompetition
]
$$

Phase309 支持 SharedPath 和 DeltaPath 在内部轨迹上可区分，但尚未证明它们是因果必要子空间。

### 五、智能理论当前进展

当前智能理论中较可信的第一性原理候选是：

```text
智能不是存储孤立知识点；
智能是在持续输入中构造相对状态网络，
根据目标选择条件化路径，
让候选知识或行动竞争，
再根据结果和反馈更新内部路径。
```

可以用以下状态更新框架组织：

$$
S_{t+1}
=
F(S_t,x_t,g_t,feedback_t;\mathcal{G}_t)
$$

$$
a_t
=
Select(Candidates(S_{t+1}),g_t)
$$

$$
\mathcal{G}_{t+1}
=
Update(\mathcal{G}_t,S_t,a_t,feedback_t)
$$

其中：

```text
S: 相对状态网络；
G: 可复用机制图谱；
g: 当前目标；
a: 输出词元、回答或行动；
Update: 不破坏已有知识的局部可塑性更新。
```

当前 DNN 已经显示出前两式的部分能力：相对状态、路径路由、候选竞争和复杂读出。但第三式仍主要依赖离线训练，尚未证明可控实时学习、低副作用更新和稳定系统思维。因此这仍是智能理论框架，不是完成的智能数学理论。

### 六、当前问题和硬伤

#### 1. 图谱覆盖率和机制完成度被混用

历史记录中的：

```text
language_pattern_family_atlas: 0.84
physical_distribution_coverage: 0.79
causal_audit_coverage: 0.52
closure_validation: 0.21
```

属于项目管理估计，不是科学测量。Phase300 同时给出：

```text
mean atlas evidence completion: 0.386058
mean physical path confidence: 0.749074
mean closure gap: 0.597737
```

两组数字口径不同。前者偏向“有没有建立框架和数据”，后者才更接近“单元证据是否完整”。因此不能用 84% 或 79% 表示“语言机制已经完成约八成”。

#### 2. 逐层记录数不能当作独立样本数

Phase309 的 31200 条 component rows（组件记录）来自 150 个独立 pair-attribute-model 单元。逐层、逐组件、逐位置记录增加了分辨率，但没有等比例增加独立证据。

后续所有报告必须同时列出：

```text
independent_case_count；
pair_count；
prompt_template_count；
object_count；
layer_component_row_count。
```

#### 3. 九族覆盖仍不等于九族物理机制已完成

行为/读出已有 972 条，但 Phase296 的均衡完整组件样本只有：

```text
9 families x 3 models x 1 representative case = 27 cases。
```

当前深入图谱主要集中在：

```text
输出协议/继续-停止；
水果对象的类别、颜色、味道；
少量语义方向和边界齿轮。
```

语法、推理、多跳绑定、跨语言、语言动作仍没有达到同等内部路径深度。

#### 4. 语义样本域过窄

Phase309 主要是水果及少量工具、家具、矿物控制。当前 shared > delta 可能同时包含：

```text
类别相似；
词频相似；
词元长度相似；
模板相似；
目标/干扰词表相似；
全模型共有的层形状。
```

因此需要动物、材料、地点、动作、情绪、抽象概念、关系和规则等更多领域，并加入词频、词元数和模板匹配控制。

#### 5. 当前复用算法仍可能把公共层形状当成共享机制

Phase309 使用逐层 margin profile（边际轨迹）的余弦相似度。它保留路径形状，但可能忽略：

```text
绝对幅度；
正负号；
局部峰值是否来自同一层事件；
同模板和同候选词表带来的全局基线。
```

高差分控制组仍有 0.563324 的平均复用，说明必须先扣除匹配随机对象对和模板基线，再解释共享主干。

#### 6. 功能标签仍主要是解释，不是因果事实

当前把 object MLP 解释为“对象写入”、query MLP 解释为“查询重组”、last attention 解释为“答案路由”。这些解释与现象一致，但尚未通过源位置到目标位置的因果传递验证。

正确标记应是：

```text
observed component event（观测组件事件）；
functional hypothesis（功能假设）；
causal edge candidate（因果边候选）。
```

不能直接标记为已确认 writer/router/gate（写入器/路由器/门）。

#### 7. 因果干预仍主要是人工线性补丁

zero、half scaling、direction add/remove 可以定位候选，但常常产生分布外状态。Phase298 的 0/72 winner flip 和 GLM4 的强随机敏感性都说明：

```text
干预有效 != 找到自然机制；
投影很强 != 因果节点；
候选组优于随机 != 找到最小齿轮；
首词边界获胜 != 自然输出闭合。
```

#### 8. 没有完成自然触发门

目前最大的机制缺口是：

```text
谁在自然输入下启动 reader/router/writer？
哪些上游状态决定通道的符号和强度？
为什么同一组件在不同模型或模板上出现相反效应？
```

没有 natural gate（自然门控），图谱仍是组件地图，不是运行机制图。

#### 9. 缺少留出预测

当前图谱主要解释已经观察过的数据。还没有系统证明：给定未见样本，图谱能提前预测：

```text
主导层；
主导组件；
路径类型；
主要竞争者；
失败模式；
干预效果；
自然闭合结果。
```

没有前向预测能力，图谱仍可能只是精细的实验日志。

#### 10. 跨模型结果不能简单平均

qwen3、GLM4、DS7B 在层数、维度、词元化、训练方式和协议行为上不同。已有结果显示：

```text
qwen3 的语义路径 attention 更突出；
GLM4 的边界场对通用扰动敏感；
DS7B 的语义读出较粗，但保留部分 shared > delta 结构。
```

跨模型一致只应建立在功能事件和归一化路径上，不能按绝对层号或通道号直接对齐。

#### 11. 小模型外推是根本限制

当前小模型的编码可能比大模型粗糙，甚至存在 30%-50% 的结构偏差。三小模型一致只能说明现象不是单一模型偶然，不能证明大模型、人脑或一般智能必然使用相同机制。

#### 12. 数据和文档版本存在复现风险

当前结果同时存在于：

```text
tests/result/...
tests/gpt5/result/...
frontend/public/vis_data/...
```

`progress.json` 仍停留在 Phase308，而 Phase309 已经完成。GPT 和 GLM 两条路线使用不同 Phase 编号空间，`IntelligentTheory.md` 又吸收了其他历史阶段。若不增加 branch_id（研究分支编号）、source_phase（来源阶段）、model_hash（模型哈希）、tokenizer_hash（词元器哈希）和 git commit（代码提交），后续很容易把不同版本结果混合为同一证据链。

### 七、进度重新评估

在没有冻结全图谱分母前，不应给出单一精确百分比。当前更合理的是分层区间：

| 层级 | 当前完成度 | 判断依据 |
|---|---:|---|
| 图谱工程与数据基础设施 | 85%-90% | schema、样本库、结果格式、缺口队列、前端同步已形成 |
| 九族行为/读出基线 | 65%-75% | 三模型已有 972 条均衡结果，但真实任务域和开放集不足 |
| 层/组件观测分布 | 30%-40% | 历史组件记录较多，但九族均衡深测只有少量代表样本 |
| 语义复用-差分子图谱 | 35%-45% | 已有三位置和对象对结果，但领域窄、无对比提示因果链 |
| 因果必要性图谱 | 15%-25% | 有局部弱正结果，强因果、低副作用和最小性不足 |
| 自然门控与因果充分性 | 5%-10% | 自然触发源尚未定位，跨样本迁移不稳定 |
| 严格自然闭合 | 5%-10% | 首词边界候选存在，但严格干净和自然 EOS 动作未闭合 |
| 语言编码机制整体 | 20%-30% | 机制框架清楚，完整可预测因果图尚未完成 |
| 可验证智能理论 | 15%-25% | 有统一框架，但实时学习、跨模态、可塑性和系统级预测未验证 |

如果只评价“语言模式族物理分布拼图”，当前更谨慎的整体估计是：

```text
约 40%-50%，而不是 79%。
```

79% 更接近工程字段覆盖和候选图谱轮廓；40%-50% 更接近独立样本、族覆盖、组件深度、因果等级和留出验证综合后的科学完成度。

### 八、如何改进图谱和特征分析算法

#### 1. 冻结科学分母

图谱基本单元应固定为：

$$
Cell
=
(family,task,model,language,template,variant,position,component,evidence)
$$

每个 Cell 必须记录：

```text
planned independent cases；
valid independent cases；
missing cases；
heldout cases；
negative controls；
evidence level；
source files；
reproduction status。
```

以后只允许用有效独立单元数计算覆盖率，逐层行数只表示测量分辨率。

#### 2. 使用简单但严格的匹配差分

对复用路径，不应只看原始余弦值。至少增加：

$$
ReuseAdjusted
=
Reuse(shared\ pair)
-
Reuse(matched\ random\ pair)
$$

匹配随机对必须控制：

```text
词元数；
词频档；
模板；
候选集合；
答案长度；
对象类别距离。
```

路径特征同时保留：

```text
符号；
绝对幅度；
首次出现层；
峰值层；
持续层数；
反转层；
object -> query -> last 的转移顺序。
```

#### 3. 把路径角色拆成可验证边

不要直接写：

```text
object MLP = writer。
```

应测试：

```text
object state 改变
-> query state 是否按预测改变
-> last boundary 是否按预测改变
-> rollout 是否按预测改变。
```

每条边依次获得：

```text
L2 readout；
L3 layer path；
L4 component attribution；
L5 low-side-effect necessity；
L6 controlled sufficiency；
L7 rollout stability；
L8 clean closure。
```

#### 4. 用四条件检查最基本的非线性耦合

对候选组件 A、B，固定四种条件：

```text
baseline；
A only；
B only；
A + B。
```

基本交互量：

$$
Interaction(A,B)
=
\Delta_{A+B}
-
\Delta_A
-
\Delta_B
$$

若不接近 0，就不能继续用单组件贡献相加解释机制。这个测试只需要基础差分，不依赖复杂数学模型。

#### 5. 把自然触发源加入图谱

每个 writer/gear（写入器/齿轮）候选必须追踪：

```text
上游来源词元；
上游 attention head；
进入 MLP 前的 residual state；
gate/up/product/down 的自然激活；
下游 boundary 和 rollout。
```

目标是从“人工拨齿轮”推进到：

```text
什么输入条件自然拨动齿轮，
齿轮如何改变下游状态，
为什么在错误样本上没有被正确启动。
```

#### 6. 引入留出预测和反例优先

每轮样本固定拆为：

```text
60% 路径发现；
20% 参数和阈值校准；
20% 完全留出验证。
```

先冻结预测，再打开留出结果。必须记录预测失败样本，并优先研究反例，不允许只报告均值正结果。

### 九、下一阶段大任务

下一阶段应定义为一个连续大阶段：

```text
Phase311-330: 语言模式族物理机制图谱完成阶段
```

该阶段不是追求一次性“最终闭合”，而是完成从观测图谱到可预测因果图谱的关键升级。

#### Batch A: 图谱口径冻结和证据清洗

```text
1. 合并 tests/result 与 tests/gpt5/result 的正式来源；
2. 给所有记录增加 branch_id、run_id、model_hash、tokenizer_hash、git_commit；
3. 更新 progress.json 到 Phase310；
4. 区分 independent cases 和 layer/component rows；
5. 建立 claim registry（机制主张登记）和正/负/反例文件索引；
6. 重新计算证据封顶后的真实完成度。
```

#### Batch B: 三个语言核心能力的均衡物理分布扩展

知识网络：

```text
从水果扩展到动物、材料、工具、地点、动作、情绪、抽象概念和社会关系；
测试 category、attribute、function、part、relation、comparison、negation；
补 Phase309 的 contrast prompt（对比提示）shared/difference 路径。
```

语法系统：

```text
主谓一致；
词性和角色；
语序；
依存距离；
嵌套从句；
否定与作用域；
时态和指代；
跨语言同构语法。
```

推理能力：

```text
单规则检索；
对象-关系绑定；
两跳组合；
变量替换；
否定条件；
反事实；
顺序约束；
错误中间状态和错误关系控制。
```

每个 family-model 至少需要 100 个独立高质量样本后，才允许形成稳定机制主张；内部全链路深测可先对每格 20-30 个均衡样本执行，再根据缺口扩展。

#### Batch C: 全链路物理路径追踪

每条高价值样本统一追踪：

```text
source token
-> object/relation/query positions
-> attention route
-> MLP gate/up/product/down
-> residual/norm
-> full-vocabulary boundary
-> phrase likelihood
-> natural rollout
-> stop/protocol gate。
```

输出固定的 event rows（事件行）和 edge rows（边行），不再只输出单个总分。

#### Batch D: 低副作用因果链和非线性审计

对稳定候选执行：

```text
自然状态匹配替换；
half/mean replacement；
same-norm random；
permutation；
negative family control；
source-to-target patch；
attention + MLP 联合干预；
四条件非线性交互；
对象、模板、语言和模型留出。
```

目标不是找到“有用补丁”，而是确认：

```text
触发源必要；
中间边必要；
目标组件具有受控充分性；
全词表边界按预测移动；
自然生成按预测改变；
副作用在预设阈值内。
```

#### Batch E: 前向预测、闭合和智能理论验证

图谱必须对未见样本预测：

```text
主导层和组件；
shared/delta 路径；
主要竞争者；
错误路径；
干预方向；
自然闭合结果。
```

闭合仍坚持硬判据：

$$
Closure
=
SemanticDone
\land StopWins
\land ContinueSuppressed
\land RolloutStable
$$

只有图谱在留出样本上能稳定预测这些结果，才可以把机制主张提升为语言编码理论的一部分。

智能理论验证则至少还要增加：

```text
在线学习新关系；
低副作用局部编辑；
跨任务复用；
多步状态传递；
错误路径自诊断；
新知识更新后的系统一致性。
```

### 十、阶段成功标准

下一大阶段完成的最低标准：

```text
1. 九族三模型的独立样本分母冻结；
2. 知识、语法、推理三条核心能力都有三位置/多位置组件图谱；
3. 每个高价值机制主张都有正证据、负证据和反例；
4. 至少形成若干条 L5 以上的完整低副作用因果边；
5. 至少一个机制链在对象、模板和语言留出上复现；
6. 图谱对未见样本的路径和失败类型有明确高于简单基线的预测能力；
7. 闭合继续独立计分，不因图谱覆盖增加而虚增；
8. 所有数据可由固定脚本、固定模型版本和固定提交重新生成。
```

### 十一、最终结论

当前研究已经完成的不是“语言编码机制破解”，而是破解所需的核心实验框架和第一批高价值拼图：

```text
语言模式族分类；
行为和读出竞争基线；
继续/停止/协议场分离；
attention/MLP/residual 路径拆解；
补偿与副作用认识；
语义 object/query/last 三位置路径；
shared backbone 与 delta control 的跨模型路径差异；
局部语义坐标和边界齿轮候选；
固定格式图谱和可视化基础。
```

当前最关键的缺口是：

```text
自然触发门；
源到目标的完整因果链；
语法和推理的同等深度物理图谱；
跨领域、跨模板和跨模型留出；
完整短语与自然生成闭合；
图谱对未知样本的前向预测。
```

所以接下来的第一优先级仍然正确：

```text
完成各种语言模式族的物理分布拼图图谱。
```

但“完成”的标准必须从“有很多记录”升级为：

```text
分母明确、路径完整、证据分级、反例保留、因果可复现、留出可预测。
```

第二优先级才是在这些稳定因果路径上尝试闭合。智能理论也不应继续靠增加抽象公式推进，而应等待知识网络、语法、推理、协议闭合和可塑性更新这几条机制链共同提供约束，让统一结构从拼图中自然浮现。

## Phase 311: 知识、语法、推理三核心族的冻结分母物理路径测试 [2026-07-09 17:48]

### 任务目标

Phase310 指出当前九族图谱存在一个核心硬伤：行为/读出样本较多，但知识、语法、推理三条核心能力缺少同一口径、同一深度的内部路径数据。本阶段不再随机选择局部案例，而是冻结一个明确分母：

```text
3 个核心模式族；
每族 8 种机制；
每种机制 5 个独立词汇或规则变体；
3 个模型；

3 x 8 x 5 x 3 = 360 个独立模型样本。
```

样本拆分：

```text
discovery: 72 个基础样本 / 216 个模型样本；
calibration: 24 个基础样本 / 72 个模型样本；
heldout: 24 个基础样本 / 72 个模型样本。
```

三模型按顺序运行：

```text
qwen3 -> GLM4 -> DS7B
```

没有同时加载多个模型，没有发生显存溢出。

### 脚本和固定格式数据

```text
tests/gpt5/phase311_core_language_physical_atlas.py
tests/gpt5/run_phase311_314_core_language_physical_mechanism_atlas.sh
```

核心输出：

```text
tests/gpt5/result/pattern_family_atlas/v2/phase311_core_language_case_bank.jsonl
tests/gpt5/result/pattern_family_atlas/v2/phase311_core_language_model_plan_rows.jsonl
tests/gpt5/result/pattern_family_atlas/v2/phase311_core_language_case_result_rows.jsonl
tests/gpt5/result/pattern_family_atlas/v2/phase311_core_language_component_rows.jsonl
tests/gpt5/result/pattern_family_atlas/v2/phase311_core_language_position_summary_rows.jsonl
tests/gpt5/result/pattern_family_atlas/v2/phase311_core_language_physical_atlas_summary.json
```

每条记录新增或固定：

```text
branch_id；
run_id；
git_commit；
model_hash；
tokenizer_hash；
independent_case；
split；
source/query/last token span；
token_match_confidence。
```

这一步修复了 Phase310 指出的部分复现风险，也明确区分 independent case（独立样本）和 layer/component row（逐层组件记录）。

### 三个核心族

知识网络：

```text
category binding；
color binding；
function binding；
part binding；
habitat binding；
material binding；
comparison binding；
negated attribute。
```

语法系统：

```text
subject role；
object role；
singular agreement；
plural agreement；
past tense；
pronoun number；
adjective attachment；
relative-clause role。
```

推理能力：

```text
direct entailment；
direct contradiction；
two-hop entailment；
two-hop blocked control；
transitive order；
reversed-order control；
conjunction rule；
missing-conjunct control。
```

### 测试原理

每个样本追踪三个位置：

$$
\mathcal{P}(x)
=
[T_{source}(x),T_{query}(x),T_{last}(x)]
$$

每个位置记录 attention、MLP、residual 的逐层候选边际变化：

$$
\Delta A_l=M(h_l+A_l)-M(h_l)
$$

$$
\Delta F_l=M(h_l+A_l+F_l)-M(h_l+A_l)
$$

$$
\Delta R_l=M(h_{l+1})-M(h_l+A_l+F_l)
$$

其中：

$$
M(h)=z_{target}(h)-z_{distractor}(h)
$$

这仍是目标-干扰读出探针，不是最终机制公式。

### 客观结果

```text
planned independent model cases: 360
valid independent model cases: 360
missing independent cases: 0
layer/component rows: 37440
position summary rows: 1080
token match confidence mean: 1.0
```

模型级目标相对干扰项胜率：

```text
qwen3: 0.966667
GLM4: 0.991667
DS7B: 0.958333
overall: 0.972222
```

模式族级胜率：

```text
content_knowledge: 1.000000
reasoning_constraint: 0.983333
syntax_structure: 0.933333
```

需要严格解释：这只是受控 target-vs-distractor 候选边界，不是全词表自然回答率，也不是自然停止或闭合率。

### 初步结果分析

1. 三核心族都可以在 source/query/last 三位置形成可测组件轨迹。
2. 语法的目标边界最弱，错误主要集中于 adjective attachment、relative-clause role、object role 和部分 agreement。
3. 两跳 blocked control 在 qwen3、DS7B 各出现失败，说明“缺少推理链”控制比正向 entailment 更困难。
4. 高目标胜率说明测试候选集合可用，但不能据此声称已经找到三族各自的独立物理主干。

## Phase 312: 匹配基线、路径事件和留出预测算法 [2026-07-09 17:48]

### 脚本

```text
tests/gpt5/phase312_matched_path_feature_analysis.py
```

### 算法改进

本阶段不再只使用原始余弦相似度，而同时记录：

```text
符号；
绝对幅度；
首次显著层；
峰值层；
归一化峰值深度；
持续层数；
正负贡献；
符号翻转次数。
```

路径事件：

$$
Event(x,p,c)
=
[onset,peak,depth,persistence,positive,negative,flips]
$$

匹配复用分数：

$$
ReuseAdjusted
=
Reuse(within\ mechanism)
-
Reuse(matched\ control)
$$

推理样本的控制进一步收紧为：

```text
相同模型；
相同模式族；
相同 item index；
相同 yes/no 目标标签；
不同推理机制。
```

这一步避免把 yes/no 答案路线误判为推理机制复用。

留出预测使用 discovery + calibration 生成简单路径原型：

$$
\hat m(x)
=
\arg\max_m
cos(P(x),Prototype_m)
$$

预测阈值和原型在打开 heldout 前冻结。

### 客观结果

```text
path event rows: 3240
matched similarity rows: 3240
aggregate rows: 81
heldout prediction rows: 72
```

匹配控制前后：

```text
mean within-mechanism reuse: 0.637247
mean matched-control reuse: 0.594349
mean adjusted reuse: 0.042899
```

按模式族：

```text
content_knowledge adjusted reuse: 0.011873
reasoning_constraint adjusted reuse: 0.100799
syntax_structure adjusted reuse: 0.016024
```

这个结果大幅收紧了原始复用结论：

```text
三族都有很高的公共路径相似度；
但扣除匹配控制后，知识和语法的族内机制净复用非常弱；
推理净复用较强，但仍可能包含 yes/no、规则格式和显式 Question 路线。
```

最强净复用出现在推理 source 位置：

```text
source attention: 0.260146
source MLP: 0.258210
source residual: 0.236718
```

这提示受控推理任务的共享结构更早出现在规则/事实入口，但当前只能称为观测路径结构。

### 同模板词汇/规则留出预测

```text
family accuracy: 0.847222
mechanism accuracy: 0.736111
family random baseline: 0.333333
mechanism unconditioned baseline: 0.125000
mechanism target-conditioned baseline: 0.239583
```

结果高于简单基线，说明路径轨迹包含可预测信息。但同模板留出仍不能排除固定提示结构泄漏。

### 组件事件的整体形状

当前三族都呈现：

```text
attention 峰值总体早于 MLP；
MLP 的持续率高于 attention；
last 位置 MLP 峰值总体晚于 source/query；
attention 和 MLP 都存在大量正负交替，不是单向累积；
residual 未建模增量绝对值很小。
```

例如平均归一化峰值深度：

```text
content knowledge:
  source attention 0.1648 -> source MLP 0.3004
  query attention 0.3756 -> query MLP 0.4139
  last attention 0.5515 -> last MLP 0.7151

reasoning:
  source attention 0.0614 -> source MLP 0.2645
  query attention 0.1213 -> query MLP 0.3357
  last attention 0.1682 -> last MLP 0.6692

syntax:
  source attention 0.1398 -> source MLP 0.3413
  query attention 0.1750 -> query MLP 0.2702
  last attention 0.3131 -> last MLP 0.4812
```

最谨慎的解释是：

```text
attention 更早改变候选边界；
MLP 在更长层区间持续改写；
最后位置的 MLP 更像后期候选整合之一；
但不能仅凭时间顺序把 attention 命名为 reader、MLP 命名为 writer。
```

## Phase 313: 留出样本 attention-MLP 联合干预审计 [2026-07-09 17:48]

### 脚本

```text
tests/gpt5/phase313_heldout_component_interaction_audit.py
```

### 选择和审计隔离

候选层、位置和机制只使用 discovery + calibration 数据选择；干预只在 heldout 词汇/规则样本执行。

每模型每核心族选择 2 个机制：

```text
3 models x 3 families x 2 mechanisms = 18 heldout causal cases。
```

每个样本执行：

```text
baseline；
attention half；
MLP half；
attention + MLP half；
attention same-norm feature permutation；
MLP same-norm feature permutation。
```

总计：

```text
108 intervention rows；
18 interaction rows；
18 rollout comparison rows；
missing rows: 0。
```

基本交互量：

$$
Interaction(A,F)
=
\Delta_{A+F}
-
\Delta_A
-
\Delta_F
$$

预设强交互条件：

$$
|Interaction|>1
$$

### 客观结果

```text
target-vs-distractor winner changes: 0 / 18
strong nonlinear interactions: 0 / 18
full-vocabulary top1 changes: 2 / 18
rollout text changes: 4 / 18
mean absolute interaction: 0.217882
```

分模型：

```text
qwen3:
  winner changes 0
  strong interactions 0
  top1 changes 2
  rollout changes 2

GLM4:
  winner changes 0
  strong interactions 0
  top1 changes 0
  rollout changes 0

DS7B:
  winner changes 0
  strong interactions 0
  top1 changes 0
  rollout changes 2
```

这是一个明确负结果：

```text
观测路径峰值没有在 heldout 上转化为强因果边；
attention/MLP 半缩放的联合效应很小；
当前简单组件强度选择算法不足以定位最小因果单元。
```

qwen3 两例 full-vocabulary top1 变化也不能升级为强正结果：一个只是 capitalization（大小写）变化，另一个把 adjective attachment 从正确名词路线推到 color 路线，属于副作用迹象。

因此本阶段证据等级全部保持为：

```text
L4 intervention effect；
没有 L5 因果必要性边。
```

## Phase 314: 核心语言物理机制图谱合成 [2026-07-09 17:48]

### 脚本和输出

```text
tests/gpt5/phase314_core_mechanism_atlas_synthesis.py

phase314_core_mechanism_atlas_summary.json
phase314_mechanism_claim_rows.jsonl
phase314_graph_nodes.jsonl
phase314_graph_edges.jsonl
phase314_evidence_progress.json
```

合成结果：

```text
graph nodes: 106
graph edges: 99
mechanism claims: 5
```

本阶段第一次把覆盖率改成冻结分母下的精确指标：

```text
controlled denominator coverage: 1.0
three-position observational coverage: 1.0
matched baseline coverage: 1.0
same-template heldout prediction coverage: 1.0
heldout causal case coverage: 0.05
heldout causal quality proxy: 0.0
natural gate coverage: 0.0
strict clean closure: 0.0
```

这组指标不能解释为整个语言图谱已经完成，而是：

```text
本轮冻结的受控三核心族观测分母已完成；
因果分母只覆盖 5%；
因果质量、自然门控和严格闭合仍为 0。
```

## Phase 315: 模板改写后的完全留出路径验证 [2026-07-09 17:48]

### 任务原因

Phase312 的 heldout 只替换了词汇或规则对象，模板保持相同。为了检查路径预测是否主要来自模板，本阶段在原型冻结后改写提示：

```text
Fact -> Statement；
Complete -> Fill the blank；
Answer -> Response；
Question -> Decide whether；
语法指令改写为另一套表达。
```

每个机制使用此前完全未参与原型训练的 item index=4：

```text
24 mechanisms x 3 models = 72 template-and-item heldout cases。
```

### 脚本和结果

```text
tests/gpt5/phase315_template_heldout_path_validation.py
tests/gpt5/run_phase315_template_heldout_path_validation.sh
```

```text
valid independent cases: 72 / 72
component rows: 7488
position summary rows: 216
missing rows: 0
target-vs-distractor winner rate: 0.944444
token match confidence mean: 1.0
```

冻结原型预测：

```text
family accuracy: 0.736111
mechanism accuracy: 0.486111
family random baseline: 0.333333
mechanism random baseline: 0.125000
```

与同模板留出相比：

```text
family accuracy: 0.847222 -> 0.736111, delta -0.111111
mechanism accuracy: 0.736111 -> 0.486111, delta -0.250000
```

分模型：

```text
qwen3: family 0.791667, mechanism 0.458333
GLM4: family 0.708333, mechanism 0.458333
DS7B: family 0.708333, mechanism 0.541667
```

分族最重要现象：

```text
reasoning 的模板留出仍相对较强；
knowledge 族级识别较稳定，但具体关系机制混淆较多；
syntax 对模板改写最敏感：
  qwen3 family 0.625
  GLM4 family 0.375
  DS7B family 0.500。
```

严格结论：

```text
路径原型中存在可跨模板保留的信息；
但原同模板结果包含显著模板成分；
当前机制级路径签名远未达到模板不变表示。
```

规则改写仍由程序生成，不是独立人工模板，也不是 open-set（开放集）任务，因此只能算第一版模板压力测试。

## Phase 316: 核心物理图谱连续阶段收束 [2026-07-09 17:48]

### 阶段输出

```text
tests/gpt5/phase316_core_atlas_stage_completion.py

tests/gpt5/result/pattern_family_atlas/v2/phase316_core_atlas_stage_summary.json
tests/gpt5/result/pattern_family_atlas/v2/phase316_mechanism_claim_rows.jsonl
tests/gpt5/result/pattern_family_atlas/v2/phase316_evidence_progress.json
tests/gpt5/result/pattern_family_atlas/v2/phase316_report.md
```

本连续阶段新增：

```text
controlled independent model cases: 360
template-and-item heldout model cases: 72
total new independent model cases: 432
layer/component rows: 44928
path event rows: 3240
matched similarity rows: 3240
heldout causal cases: 18
```

数据已经同步到：

```text
frontend/public/vis_data/pattern_family_atlas/v2/
frontend/dist/vis_data/pattern_family_atlas/v2/
```

前端结果：

```text
Synced 1216 pattern atlas v2 files
npm run build: passed
```

只有既有 chunk size warning，不影响图谱数据读取。

### 本阶段最重要的正结果

1. 第一次对知识、语法、推理建立同一冻结分母、同一三位置、同一组件粒度的三模型路径图谱。
2. 第一次严格区分 432 个独立模型样本和 44928 条逐层记录。
3. 路径原型在词汇/规则留出和模板改写留出上都高于简单基线，说明内部路径包含部分可迁移结构。
4. attention 峰值总体早于 MLP、MLP 持续更久、last MLP 更晚的时间轮廓在三族中均可观察。
5. progress.json 已改为工程进度和科学进度双口径，不再用旧管理百分比代表机制完成度。

### 本阶段最重要的负结果

1. 扣除匹配控制后，总净复用只有 0.042899。
2. 知识净复用 0.011873、语法净复用 0.016024，不能支持强“族级共享物理主干”结论。
3. 推理净复用 0.100799 较强，但仍可能部分来自 yes/no 候选路线、规则格式和问题位置。
4. 模板改写使机制预测下降 25 个百分点，证明模板不是噪声，而是路径组成的一部分。
5. 18 个 heldout 联合干预中没有 winner flip，也没有强非线性交互。
6. 当前没有自然门控，没有全词表边界闭合，没有严格自然生成闭合。

### 理论进展

本阶段支持对“语言是模式网络”作一次收紧：

旧的过强表达：

```text
每个模式族存在一个稳定共享主干，再叠加差分路径。
```

当前更符合证据的表达：

```text
语言任务在 source/query/last 和 attention/MLP 上形成可预测的条件化路径；
其中既有共享的候选/格式/位置结构，也有机制特异更新；
共享程度依赖模式族、提示模板、目标词表和模型；
不能预设每个模式族都有一个模板不变、线性可移植的共享主干。
```

因此当前机制组织公式应继续写成条件化图，而不是固定路径：

$$
\mathcal{P}(x)
=
\mathcal{P}
(family,mechanism,template,target,position,model)
$$

更严格地说：

$$
SharedPath
\neq
RawSimilarity
$$

而应至少满足：

$$
SharedPathCandidate
=
WithinMechanismSimilarity
-MatchedControl
-TemplateEffect
-TargetEffect
$$

并通过留出预测和因果边验证。

### 智能理论角度的关键洞察

1. 语言路径具有可预测结构，但不是简单模板不变代码。
2. 模板、候选词表和任务指令不是外部噪声，而是条件化路由的一部分。
3. 推理 source 位置出现最强净复用，支持“规则入口组织后续候选路线”的假设，但没有证明这是推理本体。
4. 知识和语法在当前 profile 相似度下净复用很弱，说明复用可能发生在更细通道、绑定关系或非线性门上，而不是完整逐层 margin 轮廓。
5. 可预测不等于因果；本阶段预测结果为正、干预结果为负，恰好证明“描述图谱”和“运行机制图谱”必须分开计分。

### 严格进度更新

本轮冻结局部范围：

```text
controlled core case coverage: 100%
controlled three-position event coverage: 100%
matched-control analysis coverage: 100%
same-template heldout prediction coverage: 100%
template-heldout prediction coverage: 100%
heldout causal case coverage: 5%
heldout causal quality proxy: 0%
natural gate coverage: 0%
strict clean closure: 0%
```

全局研究估计只做小幅更新：

```text
语言模式族物理分布拼图: 约 45%-55%
语言编码机制: 约 20%-30%
严格自然闭合: 约 5%-10%
可验证智能理论: 约 15%-25%
```

物理分布拼图提高的原因是三核心族获得了均衡数据、匹配控制和两种留出验证；机制和闭合没有显著提高，因为因果审计是负结果。

### 当前硬伤

1. 受控任务全部为英文，小模型结果不能直接外推。
2. 每种机制只有一个模板改写留出样本，尚不足以形成强模板不变结论。
3. 推理使用 yes/no 词表，尽管匹配控制已同标签化，仍可能保留共享答案路线。
4. knowledge/syntax 的净复用接近 0，现有完整 profile 算法可能太粗，也可能说明族级共享主干假设过强。
5. Phase313 使用 half scaling 和 feature permutation，仍不是自然状态匹配干预。
6. 候选层经常落在 L0，可能反映表面模板/词汇入口，而非高级机制节点。
7. 没有追踪 source token attention 到 MLP gate/up/product/down 的自然传递。
8. 没有开放集 unknown family 测试。
9. 没有完整短语似然、全词表阻塞者和 strict-clean rollout 联合审计。

### 下一阶段

Phase311-316 已完成一个独立阶段目标：

```text
把知识、语法、推理从零散代表样本，推进到冻结分母的观测图谱、匹配控制、留出预测和小规模干预审计。
```

下一阶段与当前阶段不再是同一个任务层级，应转向：

```text
Phase317-325:
Natural Source-to-Boundary Causal Edge Mapping
自然来源到输出边界的因果边图谱。
```

核心任务：

```text
1. 独立编写新模板、新领域和开放集样本；
2. 追踪 source token -> attention head -> MLP gate/up/product/down；
3. 使用自然状态匹配替换，不再只做 half scaling；
4. 验证 source/query/last 的方向传递，而不是只看局部峰值；
5. 同时测 target/distractor、全词表 blocker、完整短语和自然 rollout；
6. 只有跨对象、跨模板、跨语言留出成立，才升级为 L5/L6 因果边。
```

### 阶段结论

Phase311-316 完成了用户要求的本阶段系统任务，没有在单个局部测试后停止。

最可靠的新结论是：

```text
知识、语法、推理的内部路径具有部分可预测结构；
但匹配控制和模板留出会显著削弱表面高复用；
当前简单 attention/MLP 路径峰值不能稳定转化为 heldout 因果边；
语言模式族物理图谱已经更严格、更完整，语言编码机制仍未闭合。
```

## Phase 317: Phase309-316 转述审计与自然因果分母冻结 [2026-07-09 18:30]

### 附件判断审计

附件的主方向基本正确：

```text
Phase309 是对象对内部路径的观测正结果；
Phase310-316 对原始复用、模板效应、独立样本数和因果证据作了必要降温；
当前已经形成语言模式族物理图谱基础设施，但没有破解完整语言编码机制。
```

但附件存在一个需要修正的统计口径问题。

Phase309 原始汇总同时包含两类复用均值：

```text
逐组件相似度记录总体均值：
  shared_backbone（共享主干） = 0.702623
  delta_control（差分控制） = 0.563324

对象对主路径记录的模型级均值：
  qwen3: 0.901764 vs 0.755106
  GLM4: 0.838703 vs 0.680402
  DS7B: 0.830350 vs 0.690642
```

这两组数值都来自原始文件，但不是同一个统计单位：前者来自 1350 条位置-组件相似度记录，后者来自 150 条对象对主路径记录。附件把它们并列时没有标明分母，容易被误读成简单平均不一致。

因此 Phase309 的严格结论只能是：

```text
在两种观测汇总口径下，共享主干对象对均高于差分控制对象对；
但这个结果仍受对象类别、提示模板、候选词表和属性构造影响；
它是观测路径正结果，不是因果边或共享机制闭合。
```

### 冻结科学分母

脚本：

```text
tests/gpt5/phase317_natural_source_boundary_case_bank.py
```

本阶段建立全新独立分母：

```text
3 个核心模式族；
每族 4 个机制；
每机制 12 个基础案例；
发现、校准、留出各 4 个案例；
每机制 3 套模板；
144 个基础案例；
144 个供体-受体反事实配对；
432 个模型配对。
```

分族：

```text
知识族 48；
语法族 48；
推理族 48。
```

分割：

```text
发现 48；
校准 48；
开放模板留出 48。
```

加入四类控制：

```text
同功能标签自然状态控制；
无关机制自然状态控制；
错误位置替换；
同范数特征置换。
```

时间戳已经从科学分母哈希中排除：

```text
case_bank_hash = fdabbf60a606c54d5b10
pair_bank_hash = 93b7b526cca4d01411c9
```

这一步保证重复生成文件不会因为 created_at（创建时间）字段改变科学分母指纹。

## Phase 318: 自然来源状态替换和来源到查询/边界传播 [2026-07-09 18:30]

### 脚本和运行

```text
tests/gpt5/phase318_natural_source_state_transfer.py
tests/gpt5/run_phase317_319_natural_source_boundary_atlas.sh
```

按以下顺序完成 CUDA（英伟达并行计算平台）测试：

```text
qwen3 -> GLM4 -> DS7B
```

每个模型结束后释放显存，再加载下一个模型。正式运行没有缺失案例和显存溢出。

### 干预原理

对供体样本和受体样本，先提取自然运行时来源位置的逐层状态。然后在受体指定层和来源位置执行完整自然状态替换：

$$
h'_{l,s}(x_r)=h_{l,s}(x_d)
$$

其中：

```text
x_d 为供体；
x_r 为受体；
l 为来源候选层；
s 为来源词元位置。
```

供体相对受体的边界：

$$
B(x)=L_{donor}(x)-L_{recipient}(x)
$$

来源替换效应：

$$
\Delta_{source}=B(do(h_{l,s}=h^d_{l,s}))-B(baseline)
$$

控制校正效应：

$$
\Delta_{corrected}
=
\Delta_{source}
-
\max(\Delta_{unrelated},\Delta_{wrong\ position})
$$

发现集只用于扫描 6 个归一化深度候选层；校准集不参与选层，只检验冻结层和控制项；留出集在 Phase319 才打开。

来源变化在查询或最后位置朝供体方向传播的投影比例：

$$
\rho_{p,k}
=
\frac{
(h'_{k,p}-h^r_{k,p})^\top(h^d_{k,p}-h^r_{k,p})
}{
\|h^d_{k,p}-h^r_{k,p}\|^2
}
$$

### 客观结果

```text
discovery scan rows（发现扫描记录）: 1728
calibration control rows（校准控制记录）: 864
source/query/last propagation rows（传播记录）: 37096
source layer selections（来源层选择）: 36
missing pairs（缺失配对）: 0
```

三模型校准结果：

```text
qwen3:
  source replace = 8.072286
  unrelated replace = 3.011598
  source - unrelated = 5.060688
  donor win rate = 0.604167

GLM4:
  source replace = 3.444587
  unrelated replace = 0.923412
  source - unrelated = 2.521175
  donor win rate = 0.395833

DS7B:
  source replace = 4.911377
  unrelated replace = 1.979026
  source - unrelated = 2.932351
  donor win rate = 0.562500
```

总体：

```text
source transfer mean = 5.476083
unrelated transfer mean = 1.971345
control-corrected transfer = 3.504738
donor win rate = 0.520833
full-vocabulary top1 change rate = 0.291667
```

查询和最后位置出现正供体方向投影的比例：

```text
qwen3: query 0.660636, last 0.819226
GLM4: query 0.701480, last 0.827714
DS7B: query 0.686070, last 0.805760
```

### 结果边界

这是比 half scaling（半缩放）更自然的状态干预，但仍然只是整条高维残差状态替换。它可能同时携带：

```text
词元身份；
显式答案值；
模板位置；
局部语法；
多个注意力头和多层感知机通道的混合信息。
```

因此本阶段只能升级为 L4 自然状态干预效应，不能称为最小因果边。

## Phase 319: 开放模板留出的注意力头-多层感知机通道中介审计 [2026-07-09 18:30]

### 脚本

```text
tests/gpt5/phase319_heldout_component_mediation.py
```

### 选择隔离

只使用发现集测量来源替换自然引发的组件响应，并为每个模型-模式族-机制冻结：

```text
1 个来源层；
1 个注意力头输入候选；
1 个多层感知机乘积通道组候选；
候选作用位置为 query（查询）或 last（最后位置）。
```

开放模板留出样本完全不参与来源层、注意力头或通道选择。

### 中介恢复原理

在来源状态替换保持有效时，把被选注意力头或多层感知机乘积通道恢复到受体自然基线值：

$$
M_A
=
\Delta_{source}
-
\Delta_{source+A\ restore}
$$

$$
M_F
=
\Delta_{source}
-
\Delta_{source+F\ restore}
$$

$$
M_{AF}
=
\Delta_{source}
-
\Delta_{source+A,F\ restore}
$$

若某个头或通道真实中介来源效应，恢复后应稳定削弱供体边界移动，而不是只产生无方向扰动。

同时测量：

```text
供体-受体首词元边界；
全词表首选词元；
完整目标短语对数似然；
8 词元自然生成展开；
无关机制、错误位置和特征置换控制。
```

### 数据量

```text
discovery component rows（发现组件记录）: 58624
component selections（组件选择）: 72
open-template heldout cases（开放模板留出案例）: 144
heldout condition rows（留出条件记录）: 1152
natural component event rows（自然组件事件）: 288
phrase/rollout cases（短语/生成案例）: 144
missing cases（缺失案例）: 0
```

### 总体结果

```text
heldout source transfer mean = 7.674791
heldout control-corrected transfer = 3.216129
heldout donor win rate = 0.590278
phrase transfer shift mean = 6.453154
rollout change rate = 0.715278
patched rollout starts with donor = 0.166667
joint mediation loss mean = 0.018746
```

这组结果必须拆开解释：

```text
整来源状态替换在开放模板上仍然有效；
短语边界也明显朝供体移动；
但自然生成只有 16.67% 直接以供体答案开始；
单注意力头加单多层感知机通道平均只解释 0.018746；
局部边界变化没有变成稳定完整生成机制。
```

### 分族结果

```text
知识族:
  source transfer = 18.654491
  control-corrected = 8.526307
  joint mediation = -0.015669

语法族:
  source transfer = 4.071312
  control-corrected = 1.178225
  joint mediation = 0.059169

推理族:
  source transfer = 0.298571
  control-corrected = -0.056145
  joint mediation = 0.012739
```

这是本阶段最重要的分化：

```text
知识族测试的 source（来源）就是事实中显式出现的答案值词元；
它证明的是显式记录值的搬运路径，不是参数记忆中的潜在知识检索；
语法来源词元有中等条件效应；
推理不能由单个“关键词元”状态替换恢复，说明推理来源很可能是多语句、多词元绑定结构。
```

### 初筛候选

DS7B 出现 2 个个体筛选候选：

```text
category_binding（类别绑定）；
material_binding（材料绑定）。
```

但两者主要来自最后位置注意力头恢复，多层感知机通道恢复分别很弱或为负；同时无关机制和错误位置控制也很大。因此它们没有直接升级为 L5，而是进入 Phase320 注册复核。

## Phase 320: 全新对象和第四模板的注册式因果边复核 [2026-07-09 18:30]

### 脚本和注册分母

```text
tests/gpt5/phase320_registered_edge_replication.py
```

在查看复核结果之前冻结：

```text
2 个候选知识机制；
8 个全新对象；
第四套独立提示模板；
8 个新供体-受体配对；
24 个模型复核配对；
每配对 7 个条件。
```

qwen3 首次运行发生一次底层段错误；在确认显存回落至 561 MiB 后单独完整重跑，8/8 案例成功。GLM4 和 DS7B 首次均完成。最终三模型数据完整，缺失为 0。

### 更严格的个体通过标准

个体必须同时满足：

$$
\Delta_{source}>0.5
$$

$$
\Delta_{corrected}>0.5
$$

$$
DonorWins=1
$$

$$
M_A>\max(0.5,0.2|\Delta_{source}|)
$$

$$
M_F>\max(0.5,0.1|\Delta_{source}|)
$$

$$
M_{AF}>\max(M_A,M_F)
$$

即使个体通过，机制仍需在平行对象和至少两个模型中复现才可升级。

### 客观结果

```text
registered model cases（注册模型案例）: 24
condition rows（条件记录）: 168
missing cases（缺失案例）: 0
source transfer mean = 21.009809
control-corrected transfer mean = 10.187535
donor win rate = 1.000000
attention mediation loss mean = 0.748984
MLP product mediation loss mean = -0.388312
joint mediation loss mean = 0.409402
registered pass count = 0
registered pass rate = 0.0
promoted L5 edge count = 0
```

分模型：

```text
qwen3:
  source transfer 25.339998
  corrected 12.767922
  attention mediation 0.484290
  MLP mediation -1.258609
  pass 0/8

GLM4:
  source transfer 23.219342
  corrected 12.126297
  attention mediation 0.274372
  MLP mediation 0.020497
  pass 0/8

DS7B:
  source transfer 14.470087
  corrected 5.668385
  attention mediation 1.488292
  MLP mediation 0.073177
  pass 0/8
```

严格结论：

```text
全新对象和模板重复证明整来源值状态替换很强；
但没有证明一个注意力头和一个多层感知机通道构成稳定完整中介链；
Phase319 的 2 个筛选候选被注册复核否定升级；
当前仍为 L4 干预效应，没有 L5 因果边。
```

## Phase 321: 自然来源因果候选图谱合成 [2026-07-09 18:30]

### 脚本和图谱

```text
tests/gpt5/phase321_natural_causal_edge_atlas_synthesis.py
```

生成：

```text
graph nodes（图谱节点）: 131
graph edges（图谱边）: 252
evidence claims（证据声明）: 5
```

固定格式文件包括：

```text
phase321_natural_causal_edge_atlas_summary.json
phase321_natural_causal_edge_progress.json
phase321_natural_causal_edge_claim_rows.jsonl
phase321_natural_causal_edge_graph_nodes.jsonl
phase321_natural_causal_edge_graph_edges.jsonl
phase321_family_aggregate_rows.jsonl
phase321_model_family_aggregate_rows.jsonl
phase321_mechanism_aggregate_rows.jsonl
phase321_calibration_condition_aggregate_rows.jsonl
phase321_rollout_family_aggregate_rows.jsonl
```

### 图谱边的证据分级

```text
模式族 -> 机制：L1 注册关系；
机制 -> 来源层：L3 发现候选；
来源层 -> 输出边界：L4 自然状态干预效应；
来源层 -> 查询/最后位置：L3 观测传播；
来源层 -> 注意力头/乘积通道：L3 自然响应；
单头/单通道 -> 输出边界：L4 负因果审计；
L5 已复现因果边：0。
```

### 机制公式的改进

旧的线性写法隐含：

$$
\Delta B
\approx
\Delta A+
\Delta F
$$

本阶段数据不支持这种单头、单通道相加解释。更符合当前证据的组织形式是条件化多节点图：

$$
\Delta B
=
\mathcal{F}
(S,
\mathcal{H},
\mathcal{C},
R,
Q,
T,
M)
$$

其中：

```text
S：来源状态组；
H：注意力头集合；
C：多层感知机通道集合；
R：残差携带状态；
Q：查询和角色绑定条件；
T：模板、目标和候选集合；
M：具体模型。
```

这仍然是待验证的组织公式，不是闭合定律。

当前最可靠的机制分解应写成：

$$
SourceGroup
\rightarrow
DistributedCarrierSet
\rightarrow
ConditionalBoundary
\rightarrow
Phrase
\rightarrow
Rollout
$$

而不是：

$$
SingleToken
\rightarrow
SingleHead
\rightarrow
SingleMLPChannel
\rightarrow
Answer
$$

## Phase 322: 自然来源到边界阶段收束 [2026-07-09 18:30]

### 脚本和客户端

```text
tests/gpt5/phase322_natural_source_boundary_stage_completion.py
```

阶段数据已同步到：

```text
tests/gpt5/result/pattern_family_atlas/v2/
tests/result/pattern_family_atlas/v2/
frontend/public/vis_data/pattern_family_atlas/v2/
frontend/dist/vis_data/pattern_family_atlas/v2/
```

客户端同步和构建：

```text
Synced 1255 pattern atlas v2 files
npm run build: passed
```

仅有既有打包体积警告，不影响数据读取。

### 阶段客观总量

```text
基础独立案例: 144
基础反事实配对: 144
模型配对: 432
开放模板留出模型配对: 144
注册复核模型配对: 24
发现扫描记录: 1728
校准控制记录: 864
来源到查询/最后传播记录: 37096
发现组件记录: 58624
留出条件记录: 1152
短语/生成案例: 144
缺失案例: 0
```

### 阶段最重要正结果

1. 自然供体完整来源状态在开放模板和全新对象上都能稳定改变显式值边界。
2. 来源变化在查询和最后位置出现可测的供体方向传播。
3. 语法族部分机制存在弱到中等来源条件效应。
4. 测试第一次同时覆盖首词元边界、完整短语、自然生成、错误位置、无关机制和注册复核。
5. 描述图谱与因果图谱继续分开计分，没有用工程覆盖率替代机制完成度。

### 阶段最重要负结果

1. 推理族校正来源效应为 -0.056145，不支持单关键词元承载通用推理路径。
2. 知识族强效应主要来自显式答案值词元搬运，不代表参数记忆检索。
3. 单注意力头和单多层感知机通道联合中介均值只有 0.018746。
4. 自然生成直接以供体答案开始的比例只有 0.166667。
5. 2 个个体筛选候选在 24 个注册复核案例中通过 0 个。
6. 没有 L5 因果边，没有自然门控，没有严格闭合。

### 当前硬伤

1. 知识族来源位置直接出现答案值，存在强复制/搬运捷径。
2. 推理来源应是规则、事实和变量绑定的多词元组，本阶段单词元来源定义过粗。
3. 候选组件按自然响应范数选择，响应大不等于中介作用强。
4. 每个机制只选择一个注意力头和一个固定宽度通道组，无法覆盖分布式冗余集合。
5. 多层感知机只在 product（乘积）位置做通道恢复，gate/up/down（门控/上投影/下投影）尚未形成完整因果分解。
6. 所有任务仍为英语受控提示，没有跨语言复现。
7. 小模型可能合并或粗化真实语言机制，三模型一致仍不能外推大型模型和人脑。
8. 整残差状态替换副作用仍较大，必须加入状态完整性和内容保持审计。

### 理论进展

本阶段没有改名，而是进一步收紧“语言是动态模式网络”理论：

```text
语言模式不是固定线性方向；
也不是一个模式对应一个注意力头或一个多层感知机通道；
显式值搬运、语法约束和推理绑定使用不同粒度的来源结构；
同一个表面任务可能由多节点、冗余、条件门控的载体集合实现；
完整语言机制必须同时解释来源形成、跨层传播、边界竞争、短语概率和自然生成。
```

当前工作理论公式：

$$
\mathcal{P}_{language}(x)
=
\mathcal{G}
(SourceGroup,
BindingState,
CarrierSet,
Boundary,
Gate,
Template,
Model)
$$

其中任何一条边只有在以下条件成立后才可升级：

$$
L5Edge
=
InterventionEffect
\land
MatchedControl
\land
HeldoutPrediction
\land
ParallelObjectReplication
\land
CrossModelReplication
\land
LowSideEffect
$$

当前满足上述全部条件的边数为 0。

### 严格进度

局部冻结范围：

```text
自然来源状态干预覆盖: 100%
来源到查询/最后传播覆盖: 100%
开放模板组件中介审计覆盖: 100%
已选择候选的注册复核覆盖: 100%
已升级 L5 边质量: 0%
潜在参数记忆检索路径覆盖: 0%
分布式多节点中介覆盖: 0%
自然门控覆盖: 0%
严格闭合: 0%
```

全局谨慎估计：

```text
语言模式族物理分布图谱: 约 48%-56%
语言编码机制: 约 22%-30%
严格自然闭合: 约 5%-10%
可验证智能理论: 约 17%-25%
```

物理图谱小幅提升，是因为新增了自然状态替换、传播、开放模板和注册复核；机制进度只做极小调整，因为没有 L5 边。

### 是否继续自动进入下一步

Phase317-322 已完成一个完整独立阶段：

```text
从观测路径峰值推进到自然来源状态干预、传播、组件中介和注册复核，
并严格确认没有可升级的单头-单通道 L5 因果边。
```

下一阶段不再是同一任务层级，应设为：

```text
Phase323-330:
分布式多节点载体集合与潜在参数记忆检索图谱。
```

优先任务：

```text
1. 用模型参数记忆问题替换显式答案值记录，消除复制捷径；
2. 用多词元规则子句、事实链和变量绑定组定义推理来源；
3. 按因果消融选择稀疏注意力头集合和多层感知机通道集合，而不是按响应范数选单点；
4. 分别追踪 gate/up/product/down；
5. 加入跨语言、跨领域、完整短语、自然生成和副作用复核；
6. 只有注册复现后才升级 L5，仍不以闭合为第一优先级。
```

### 通俗总结

这轮实验说明，把一个句子里“真正提供信息的词”的完整内部状态换成另一个样本的状态，模型的答案确实会跟着移动。这证明来源信息存在可追踪的物理传递。

但这种传递不是一根简单电线。恢复一个最活跃的注意力头和一个最活跃的多层感知机通道，几乎不能稳定撤销效果；推理问题甚至不能靠替换一个关键词来转移。更符合现象的图景是：语言依赖多个位置、多个头、多个通道和边界条件共同形成的动态网络。

所以本阶段完成的是一块重要拼图，并且排除了“单词元 -> 单头 -> 单通道 -> 答案”的简单机制。它没有破解完整语言编码，也没有完成智能理论或闭合。

## Phase 323: Phase288 全量单神经元 CUDA 干预未运行根因审计 [2026-07-09 18:39]

### 审计对象

用户指出：

```text
Phase288 全量单神经元 CUDA 干预尚未运行，
因此当前不能宣称全图谱已实现真实神经元级机制闭合。
```

这个判断正确。本阶段只审计代码、候选依赖、执行记录和结果目录，没有运行新的模型测试。

### 直接证据

仓库中存在第二套 Phase288：

```text
tests/gpt5/phase288_color_single_unit_heldout.py
tests/gpt5/run_phase288_color_single_unit_heldout.py
tests/gpt5/test_phase288_color_single_unit_heldout.py
```

但正式结果目录在审计前不存在：

```text
tests/result/phase288_color_single_unit_heldout/
```

也不存在以下完成证据：

```text
intervention_rows.jsonl；
clean_control_rows.jsonl；
summary.json；
manifest.json。
```

因此只能确认脚本和候选数据已经准备，不能确认 CUDA 干预已经执行。

### 根因一：Phase288 编号冲突

现有备忘录中的原 Phase288 是：

```text
模式图谱特征挖掘；
明确规定不跑新模型，不做干预，只读取全量图谱数据。
```

提交 `ec64f322` 又新增了另一套同名 Phase288：

```text
颜色单神经元留出干预。
```

新脚本提交时间位于 Phase317-322 正式执行前几分钟，但没有接入当时的阶段计划、统一运行器、备忘录阶段索引或图谱缺口队列。上一轮执行沿备忘录中的 Phase309-316 -> Phase317-322 主线推进，错误地把低编号 Phase288 视为已经完成的历史阶段，没有发现同编号新分支。

这是阶段编号和调度审计缺失，不是科学结论。

### 根因二：候选分母并不完整

Phase288 声明颜色集合为 12 类：

```text
red, blue, green, yellow, orange, purple,
brown, black, white, gray, silver, pink。
```

但 Phase286 的三个模型候选文件都只有 11 类，每类 50 条候选，缺少：

```text
pink（粉色）。
```

因此设计文件的理论分母是：

$$
12\times20\times6=1440\ cases/model
$$

当前实际可运行分母是：

$$
11\times20\times6=1320\ cases/model
$$

三模型合计：

$$
1320\times3=3960\ independent\ model\ cases
$$

如果不先补齐 pink 候选，就不能称为设计定义下的全量运行。

### 根因三：测试没有接入默认仓库导入路径

直接从仓库根目录运行：

```text
python -m unittest tests.gpt5.test_phase288_color_single_unit_heldout
```

会因为：

```text
ModuleNotFoundError: hf_probe_env
```

而失败。加入：

```text
PYTHONPATH=tests/gpt5
```

后三项设计测试可以通过。

这说明算法单元本身可导入，但尚未形成不依赖临时环境变量的持续集成入口。

### 根因四：“全量单神经元”名称过强

当前脚本不是遍历模型中全部多层感知机神经元。它实际执行的是：

```text
每个颜色从 Phase286 候选中选择 1 个最高分神经元；
在 20 个对象、6 个模板上重复测试该候选；
执行 zero、half 和 matched-control 三类干预。
```

因此更准确的名称是：

```text
颜色候选单神经元的大样本留出必要性测试。
```

不是：

```text
全模型所有神经元穷举干预；
全语言模式族神经元图谱；
真实神经元级机制闭合。
```

### 候选证据本身仍为 L4

Phase286 三模型候选来源：

```text
qwen3: 550 个已发布 MLP product（多层感知机乘积）神经元地址；
GLM4: 550；
DS7B: 550。
```

但 Phase286 报告明确写明：

```text
候选来自 channel-group（通道组）干预和真实激活归一化；
仍是 L4 组件归因证据；
没有任何记录被标记为单神经元必要性。
```

Phase288 正是用来补这个缺口，而不是已经完成这个缺口。

### 是否受 GPU 资源阻塞

不是主要阻塞。

当前环境：

```text
GPU: NVIDIA GeForce RTX 4090 D
总显存: 24564 MiB
审计时占用约 654 MiB
```

运行器已经设计为：

```text
qwen3 -> GLM4 -> DS7B 顺序执行；
每个模型结束后释放；
GLM4 限制 11GiB；
DS7B 限制 12GiB；
必要时使用 device_map=auto（自动设备映射）。
```

因此没有运行的核心原因是分支未调度和分母/测试入口未完全就绪，而不是当前硬件无法运行。

### 即使运行成功，也不能宣称什么

脚本自身的科学边界正确写明：

```text
通过只代表某个预注册颜色候选神经元，
对下一词元颜色边界具有留出必要性候选证据；
不代表充分性；
不代表自然生成稳定；
不代表低副作用闭合；
不代表其他语言模式族；
不代表全图谱神经元机制闭合。
```

所以当前图谱不能宣称真实神经元级机制闭合；Phase288 将来跑完后仍然不能单独支持这个宣称。

### 正确补全顺序

在正式运行前应先完成：

```text
1. 为第二套 Phase288 分配唯一阶段号或分支标识，消除编号冲突；
2. 补齐 pink 候选，或明确把冻结分母修改为 11 类颜色；
3. 修复 hf_probe_env 的默认导入路径；
4. 先依次运行三模型 smoke（冒烟测试）；
5. 冻结候选、分母和验收阈值；
6. 再按 qwen3 -> GLM4 -> DS7B 完成正式 CUDA 干预；
7. 输出固定 JSON/JSONL、缺失记录、结果清单和哈希；
8. 将结果接入 Pattern Atlas（模式图谱）并进行注册式复现。
```

### 审计结论

```text
Phase288 单神经元正式 CUDA 结果确实缺失；
缺失原因主要是同编号新分支未接入执行链；
其次是 pink 候选缺失和默认测试入口不完整；
GPU 不是主要阻塞；
当前脚本测试的是颜色候选单神经元，不是全部神经元；
即使正式运行，也只能补充颜色局部必要性证据，不能完成全图谱机制闭合。
```

## Phase 324: 三维图谱是否下钻到神经元级的客户端架构审计 [2026-07-09 19:28]

### 任务边界

本阶段不运行新模型实验，也不声称发现新编码机制。目标是审计当前三维可视化是否应显示神经元级对象，以及在不夸大证据、不牺牲性能的前提下，给出可实施的客户端和数据链改造方案。

### 总判断

```text
三维图谱需要支持神经元级下钻；
但不应该默认同时显示全部神经元；
更不能把“神经元被画出来”解释为“神经元机制已经被证明”。
```

神经元级显示对本项目是必要能力，因为研究目标不是只观察行为相关性，而是把语言模式族继续定位到：

```text
模型 -> 层 -> 组件 -> 头/通道组 -> 单元 -> 上下文位置 -> 输出影响
```

如果客户端只能停留在模式族、层或组件级，它可以展示研究目录，却不能承载最终的物理分布拼图。但是，模型内单元数量巨大，且单元通常具有多义性和上下文依赖性，所以正确方案是多尺度按需下钻，而不是一次加载和渲染全模型所有单元与所有连线。

### 当前客户端已经具备的基础

当前 `RealUnitTraceRenderer`（真实单元追踪渲染器）已经能够显示四类真实地址对象：

```text
mlp_product_neuron（多层感知器乘积神经元）；
attention_head_channel（注意力头通道）；
residual_dimension（残差维度）；
unembedding_token（反嵌入词元）。
```

它按层放置单元，能够读取候选分数、证据等级、因果范围和来源文件，并将 Phase286 的稳定候选与 Phase287 的真实事件结合。这说明客户端不是从零开始，已经具有有限的真实单元级覆盖层。

但当前覆盖仍有明显限制：

```text
每层主要展示最多 64 个稳定候选和当前激活单元；
没有形成完整的神经元库存索引；
没有接入 Phase288 单神经元正式干预结果；
没有单元点击、路径追踪、基线/干预/对照联动；
不能把候选单元组织成可验证的因果路径。
```

当前 `AtlasGraphRenderer`（图谱图渲染器）支持模型、阶段、层、头、通道、簇、干预、概念、任务和失败等节点，但没有明确的神经元、来源词元、门控、乘积神经元、读出和证据声明节点类型。它使用逐节点网格和逐边线段，适合小图，不适合数万到数百万单元。

当前 `DNNAnalysis3DVisualization`（深度神经网络三维分析可视化）中的激活、坐标和连线强度使用随机数生成，并且每层最多生成 50 个演示神经元和相邻层全连接线。这只能作为界面演示，不能进入科研证据视图，也不能作为真实模型结构或机制的代理。

### 必须澄清的“神经元”定义

Transformer（变换器）中的“神经元”不能与生物神经元混为一谈。客户端必须把不同物理对象分开：

```text
MLP gate/up（多层感知器门控/上投影）坐标；
MLP product（多层感知器乘积）坐标；
MLP down（多层感知器下投影）贡献；
attention head（注意力头）；
attention head channel（注意力头通道）；
residual dimension（残差维度）；
unembedding readout（反嵌入读出）；
token-position event（词元位置事件）。
```

这些对象的索引、数值含义和可干预位置不同，不能统一叫作一个模糊的 `neuron`（神经元）后直接比较。

### 正确的多尺度三维结构

建议把三维图谱固定为六级下钻：

```text
L0 语言模式族/任务；
L1 模型与阶段；
L2 层与组件；
L3 注意力头、通道组、候选簇；
L4 单神经元/单通道/单残差维度；
L5 词元位置、生成时间和因果路径事件。
```

远距离视图只显示族、层、组件和密度；接近某层时再展开候选簇；用户选择簇或证据节点后才加载单元；选择单元后才绘制其自然激活和干预路径。这样既保留全局形状，也避免把屏幕变成无意义的点云。

### 三种必须分离的视图

```text
结构视图：显示模型中实际存在的对象和索引，不代表机制证据；
自然运行视图：显示特定提示词、词元位置和时间步中的真实激活；
因果证据视图：只显示具有干预、对照、留出和复现记录的节点与边。
```

可增加比较视图，用于基线与干预、来源与查询、成功与失败、模型与模型之间同步对照。结构、激活和因果不能混用同一颜色含义。

### 神经元级固定数据身份

每个可定位对象至少需要以下不可歧义的身份：

```text
model_id；
model_hash；
tokenizer_hash；
layer_index；
component；
unit_kind；
unit_index；
head_index（如适用）；
token_position；
case_id；
run_id。
```

只保存“第 123 个神经元”是不够的，因为模型版本、组件位置、词元位置或挂钩位置一旦变化，该地址就失去可复现性。

### 建议增加的数据格式

在现有 `atlas_graph_v1`（图谱图第一版）之外，增加 `neuron_atlas_v1`（神经元图谱第一版），并按模型、层、组件和证据等级分区：

```text
neuron_nodes.jsonl：物理对象、索引、所属族和证据状态；
neuron_edges.jsonl：对象之间被观测或被干预支持的关系；
neuron_events.jsonl：提示词、位置、时间步下的自然激活事件；
neuron_interventions.jsonl：消融、替换、放大及匹配对照；
neuron_runs.jsonl：模型哈希、数据集切分、随机种子和运行环境；
neuron_index.json：分区、范围、数量、哈希和缺失记录。
```

节点核心字段建议为：

```text
node_id, node_type, model_id, model_hash, layer_index,
component, unit_kind, unit_index, head_index, position_role,
activation, candidate_score, evidence_level, causal_scope,
family_id, mechanism_id, case_id, split, source_artifact
```

因果边核心字段建议为：

```text
edge_id, source_id, target_id, relation,
baseline_value, intervention_value, raw_delta,
matched_control_delta, corrected_effect, effect_sign,
side_effect, replication_count, evidence_level, source_artifact
```

必须同时记录负结果和缺失结果，不能只把成功干预画进图谱。

### 证据视觉语义

节点大小只能表达激活量或效应量，不能单独表达证据可靠性。证据可靠性应使用稳定的边框、透明度和标记：

```text
结构已知；
自然激活已观测；
相关候选；
L4 局部干预支持；
L5 留出复现支持；
负结果；
缺失/运行失败。
```

点击任一单元必须能看到：自然激活、基线、干预值、匹配对照、超对照效应、短语滚动结果、副作用、样本数、数据切分、来源文件和模型哈希。当前 Phase286 候选必须标记为组件候选，Phase288 未正式运行前不能标记为单神经元必要机制。

### 渲染层改造

单元数量较大时，不能为每个单元创建独立 React（反应式界面框架）组件。建议：

```text
数百个选中单元：实例化球体；
数千至数万个单元：InstancedMesh（实例化网格）；
更大库存：点精灵或密度体；
边：只绘制当前选择、当前路径和聚合边；
拾取：颜色编号缓冲或分区拾取；
数据：TypedArray（类型化数组）与 Web Worker（网页工作线程）解析；
加载：按模型/层/组件/证据等级懒加载；
坐标：由层、组件和单元索引确定，禁止随机坐标。
```

任何“显示全部连接”的方案都会接近平方级边数，既无法渲染，也没有解释价值。远景只应显示聚合流量或证据密度，选择路径后再展开真实边。

### 客户端具体改动顺序

```text
1. 保留并扩展 RealUnitTraceRenderer，加入单元选择、精确地址搜索、证据筛选和路径高亮；
2. 将 AtlasGraphRenderer 分成机制图和神经元下钻图，增加明确的单元节点与因果边类型；
3. 扩展 useVisData，支持 neuron_atlas_v1 和 single_unit_intervention.v1；
4. 扩展 useResearchKernel，读取 Phase288 及以后按层分区的单元干预记录；
5. 将随机 DNNAnalysis3DVisualization 移出科研证据入口，或永久标记为 simulation_only（仅模拟）；
6. 更新 PATTERN_ATLAS_CLIENT_SPEC，冻结神经元身份、事件、干预和证据格式；
7. 接入统一的模型/族/机制/层/组件/证据过滤器和生成时间轴；
8. 在详情面板中显示原始证据和负结果，不只显示可视化摘要。
```

### 与 Phase288 的关系

客户端可以先完成神经元级数据契约和多尺度渲染，但不能预填不存在的因果结果。当前 Phase288 正式 CUDA 干预没有运行，因此：

```text
可以显示模型结构中的全部单元库存；
可以显示 Phase286 的候选单元；
可以显示 Phase287 的自然事件；
不能把这些单元标成已验证的单神经元必要机制；
只有正式干预、匹配对照、留出复现和副作用检查通过后，才能升级证据状态。
```

### 进度判断

仅针对客户端神经元级能力，当前可粗略判断为：

```text
真实单元覆盖层与基础数据读取：约 35%-45%；
完整单元库存与多尺度渲染：约 15%-25%；
单元级因果图谱：约 5%-15%；
跨模型、跨族、跨样本的神经元机制闭合：尚未完成。
```

百分比只描述工程和证据覆盖，不代表语言编码机制已经破解。

### 阶段结论

```text
神经元级显示是必要终点能力，但不是默认全量视图；
正确形式是“全局聚合 + 证据候选 + 按需单元 + 事件时间轴 + 因果路径”；
当前已有真实单元追踪基础，但主图谱、数据格式和干预证据尚未闭合；
第一优先级应是冻结 neuron_atlas_v1 数据契约并接入真实结果；
第二优先级是实现实例化、多尺度和按需加载；
第三优先级才是依据 Phase288 及后续复现结果升级单元和边的证据等级。
```

## Phase 325: 语言模式族神经元关键脉络图谱与客户端一致化 [2026-07-09 19:59]

### 阶段目标

Phase324 确认三维图谱需要支持神经元级下钻，但本阶段进一步收紧目标：

```text
不是大范围记录和显示无关神经元状态；
而是以语言模式族为入口，只显示具有研究证据的关键物理单元、组件事件和读出脉络；
实验结果、证据边界和客户端显示必须使用同一份固定数据。
```

本阶段不运行新模型，不产生新机制证据。使用 Phase286、Phase287 和模式族图谱已有结果，建立可重复生成的数据桥和聚焦三维客户端。

### 数据构造原则

每个物理单元使用以下复合身份，不允许仅用单元编号跨模型比较：

```text
U = (family, model, model_revision, layer, component, unit_kind, unit_index)
```

Phase286 中同一物理地址可能对应多个样本记录。本阶段按复合身份聚合，保留：

```text
最大与平均候选分数；
样本数、目标标签和模板/对象覆盖；
读出贡献；
是否在 Phase287 单样本自然轨迹中再次出现；
是否属于通道组干预支持集合；
原始来源文件和证据边界。
```

显示优先级只用于决定客户端首先显示哪些已有候选，不用于升级证据：

```text
DisplayPriority(u)
= MaxCandidateScore(u)
+ 0.12 * NaturalOverlap(u)
+ 0.08 * GroupSupport(u)
+ 0.004 * min(CaseCount(u), 20)
```

这只是基础排序规则。它没有被解释为语言机制公式，也没有被用于统计显著性判断。

### 边的严格语义

本阶段只生成两类边：

```text
observed_component_sequence：同一次真实前向轨迹中的组件事件顺序；
contains_localized_candidate：关键层锚点包含某个真实地址候选。
```

两类边全部固定为：

```text
causal = false
```

通道组干预记录统一标记：

```text
causal_scope = channel_group_not_single_unit
single_unit_causal = false
```

因此三维连线表示“观察到的组件顺序和候选归属”，不是神经元到神经元的因果连接。

### 固定数据包

新增生成器：

```text
tests/gpt5/phase325_pattern_family_neuron_atlas.py
```

规范结果目录：

```text
tests/gpt5/result/pattern_family_neuron_atlas/v1/
```

客户端公开镜像：

```text
frontend/public/vis_data/pattern_family_neuron_atlas/v1/
```

固定入口和文件：

```text
manifest.json
families.json
neuron_index.json
neuron_nodes.jsonl
neuron_edges.jsonl
neuron_events.jsonl
neuron_interventions.jsonl
neuron_runs.jsonl
checksums.json
partitions/{family_id}/{model}.json
```

客户端只初始读取清单和模式族索引，再按当前模式族和模型读取一个分区。原始大文件不进入初始加载链。

### 客观数据结果

当前九个模式族中，只有：

```text
content_knowledge（内容知识模式族）/ color（颜色关系）
```

具备三模型真实单元候选映射。其余八个模式族明确标记：

```text
not_mapped_to_real_units
```

没有生成占位神经元或推测路径。

全包结果：

```text
模式族总数：9；
已映射模式族：1；
模型数：3；
唯一物理候选单元：833；
非因果图谱边：848；
自然组件事件：318；
通道组干预记录：2012；
单神经元因果记录：0。
```

分模型结果：

```text
qwen3：5 个关键层，288 个唯一候选，23 个自然交叉，32 个组级支持候选；
GLM4：4 个关键层，297 个唯一候选，21 个自然交叉，44 个组级支持候选；
DS7B：3 个关键层，248 个唯一候选，19 个自然交叉，27 个组级支持候选。
```

### 客户端改造

新增：

```text
usePatternFamilyNeuronAtlas：按模式族和模型懒加载证据分区；
PatternFamilyNeuronAtlasRenderer：实例化渲染关键物理单元和路径锚点；
PatternFamilyAtlasControls：模式族、模型、证据交叉和显示数量控制；
NEURON_PATTERN_ATLAS_FORMAT.md：冻结数据身份和证据语义。
```

语言机制研究路线现在默认进入模式族关键脉络视图。该视图开启时：

```text
隐藏苹果/果实等旧演示节点；
隐藏旧 RealUnitTrace 和手动图谱叠层；
只显示所选模式族、所选模型和所选证据筛选下的关键单元；
未映射模式族只显示明确空状态；
点击层锚点或真实单元进入证据详情；
可随时返回旧工作台。
```

三类单元颜色固定为：

```text
黄色：L4 真实地址候选；
青色：同时在自然轨迹中再次观测；
橙色：属于组级干预支持集合，但不是单神经元因果。
```

单元点击详情显示模型版本、层、组件、单元类型、单元索引、激活、候选分数、样本覆盖、目标标签、自然交叉、组级范围、来源文件和证据边界。

### 渲染与交互实现

只对当前筛选后的最多 12 到 96 个关键单元生成显示对象。候选按关键层分层抽样，避免末层候选数量压倒其他层。

单元位置使用单元索引的确定性黄金角映射：

```text
theta(u) = 2 * pi * frac(0.618033988749895 * unit_index)
```

层位置由真实层号归一到固定纵轴。该坐标只是稳定布局，不代表模型内部真实欧氏距离。

视觉节点使用三批实例化网格，分别对应候选、自然交叉和组级支持。为屏幕上最多 96 个已筛选节点增加透明拾取壳，解决不同 WebGL 环境中实例拾取不稳定的问题；未显示单元不创建拾取对象。

### 验证结果

数据验证：

```text
python tests/gpt5/phase325_pattern_family_neuron_atlas.py --validate-only
结果：9 families / 833 nodes / 848 edges；
节点 ID 无重复；
所有分区模式族一致；
所有当前边 causal=false；
single_unit_causal_count=0。
```

自动测试：

```text
python tests/gpt5/test_phase325_pattern_family_neuron_atlas.py
结果：2 tests passed。
```

客户端验证：

```text
新增/修改文件定向 ESLint：通过；
npm run build：通过；
桌面 1440x1000：画布非空、路径和三类颜色正常；
窄屏 390x844：旧固定面板自动收起、无横向溢出、控制和画布可见；
九族选择：未映射族不生成神经元；
qwen3 -> GLM4 -> DS7B 切换：分区计数与数据一致；
关键候选/自然交叉/组级支持筛选：正常；
层锚点点击：正常；
真实实例单元点击与证据详情：正常。
```

生产构建仍有既有主包超过 500KB 警告，构建成功。本阶段新增模式族分区采用选择后加载，不进入初始 JavaScript 主包。

### 进展判断

客户端能力与研究覆盖必须分开：

```text
模式族关键脉络数据契约：已建立；
三模型切换、筛选、下钻和证据详情：已建立第一版；
真实物理单元候选映射：1/9 模式族，约 11.1%；
具有自然轨迹交叉的候选：三模型共 63 个唯一候选；
单神经元因果闭合：0；
全语言模式族物理脉络图谱：未完成。
```

Phase324 对客户端神经元级能力的粗略进度可以上调，但只能解释为工程能力提高。语言模式族真实物理覆盖仍然很低。

### 硬伤与边界

```text
1. 目前只有颜色关系进入真实单元图谱，不能代表整个内容知识模式族；
2. Phase286 单元是组件归因候选，通道组干预不能分配到单个神经元；
3. Phase287 自然交叉主要是一个 red cube 单样本，不能证明跨模板稳定；
4. 组件事件顺序不是因果传播路径；
5. 三个小模型内部结构可能比更大模型粗糙，不能直接外推普遍语言编码；
6. 当前布局表示索引和层次，不表示真实几何距离；
7. Phase288 正式单神经元 CUDA 干预仍未完成。
```

### 下一阶段任务

下一阶段仍属于“语言模式族物理分布拼图”同一大阶段。最合理的 Phase326 不是继续美化客户端，而是按当前固定契约扩展真实证据：

```text
第一批：output_protocol（输出协议）和 readout_competition（竞争读出）；
第二批：syntax_structure（语法结构）；
每族先完成自然组件轨迹和真实单元候选映射；
再进行通道组、匹配随机组和留出样本干预；
最后才选择少量高复现候选进入单神经元必要性测试。
```

每次新研究只需生成对应分区并更新 `manifest.json`，客户端即可在同一三维空间中查看，不再新增孤立可视化页面。

### 阶段结论

```text
Phase325 完成了“研究结果 -> 固定图谱数据 -> 聚焦三维脉络 -> 原始证据详情”的一致链；
客户端现在围绕语言模式族显示关键神经元脉络，不再默认展示无关神经元状态；
但客观研究覆盖仍只有 1/9 模式族，且单神经元因果计数为 0；
因此这是图谱基础设施和已有证据可视化完成，不是语言编码机制破解或机制闭合。
```

## Phase 326: 分布式载体集合、隐式知识检索与物理脉络图谱扩大确认 [2026-07-09 20:56]

### 阶段目标与附件判断校准

本阶段首先审计了最新附件对 Phase309-325 的收紧判断。附件的核心方向正确：当前成果应被描述为实验框架、图谱系统和第一批物理拼图，不能描述为完整语言编码机制、自然机制闭合或智能理论验证。

需要进一步修正的是旧口径中的“物理图谱完成 48% 到 56%”。该数值混合了组件级覆盖、工程能力和真实单元级覆盖。Phase325 的严格真实单元映射仍只有内容知识模式族中的颜色关系，Phase326 后可以直接核验的覆盖是：

```text
具有真实物理候选分区的模式族：2/9 = 22.2%；
具有跨模型严格扩大确认集合证据的模式族：1/9 = 11.1%；
单神经元因果模式族：0/9；
自然门控因果模式族：0/9。
```

因此附件提出的 Phase326 方向被保留，但实验被收束成同一个冻结方案：隐式知识检索、多词元推理源组、分布式载体集合、注册留出和扩大确认一次完成，不再拆成多个缺少共同分母的小实验。

### 固定样本分母

主批次：

```text
模式族：content_knowledge、reasoning_constraint；
每族机制数：4；
每机制独立对象：12；
每对象模板：3；
每模型提示数：288；
三模型提示数：864；
发现、校准、留出：各 4 个独立对象；
知识题目标答案在提示中显式出现：0。
```

内容知识机制：

```text
color_retrieval；
material_retrieval；
habitat_retrieval；
category_retrieval。
```

推理约束机制：

```text
transitive_order；
implication_chain；
conjunction_rule；
spatial_composition。
```

知识题输入只给对象和关系问题，不给答案词。推理题不再把单个关键词当作来源，而把完整规则和事实片段注册为多词元源组。

扩大确认批次不重新选择组件，只使用主批次发现集冻结的载体集合：

```text
内容知识 4 个机制；
每机制新增 16 个独立对象；
每对象 2 个全新模板；
每模型 128 个提示；
三模型 384 个提示；
独立对象-模型案例：192；
确认干预记录：2304。
```

主批次和确认批次合计：

```text
模型-提示案例：1248；
独立对象-模型案例：480；
组件集合干预记录：3840；
自然能量对照：96。
```

### 测试脚本与固定产物

新增正式脚本：

```text
tests/gpt5/phase326_distributed_carrier_case_bank.py
tests/gpt5/phase326_distributed_carrier_atlas.py
tests/gpt5/phase326_validate_token_spans.py
tests/gpt5/phase326_publish_physical_path_atlas.py
tests/gpt5/run_phase326_distributed_carrier_atlas.sh
tests/gpt5/run_phase326_expanded_confirmation.sh
```

新增自动测试：

```text
tests/gpt5/test_phase326_distributed_carrier_case_bank.py
tests/gpt5/test_phase326_publish_physical_path_atlas.py
```

结果目录：

```text
tests/gpt5/result/phase326_distributed_carrier_atlas/distributed_carrier_atlas/
```

固定图谱镜像：

```text
tests/gpt5/result/pattern_family_atlas/v2/
frontend/public/vis_data/pattern_family_atlas/v2/
tests/gpt5/result/pattern_family_neuron_atlas/v1/
frontend/public/vis_data/pattern_family_neuron_atlas/v1/
```

主要固定文件：

```text
phase326_registered_cases.jsonl；
phase326_protocol.json；
phase326_component_summary_rows.jsonl；
phase326_carrier_sets.jsonl；
phase326_intervention_rows.jsonl；
phase326_registered_heldout.jsonl；
phase326_natural_gate_rows.jsonl；
phase326_expanded_confirmation_rows.jsonl；
phase326_expanded_confirmation_audits.jsonl；
phase326_cross_model_summary.json；
phase326_expanded_confirmation_cross_model.json；
phase326_token_span_validation.json；
phase326_atlas_nodes.jsonl；
phase326_atlas_edges.jsonl；
phase326_report.md。
```

### 特征分析算法

目标与干扰答案的近似读出方向：

$$
d = \frac{W_U(t)-\frac{1}{|D|}\sum_{j\in D}W_U(j)}{\left\|W_U(t)-\frac{1}{|D|}\sum_{j\in D}W_U(j)\right\|}
$$

注意力头输入候选的近似目标贡献：

$$
s_h = \left\langle x_h, W_{O,h}^{\mathsf T}d \right\rangle
$$

多层感知机乘积组候选的近似目标贡献：

$$
s_g = \left\langle z_g, W_{down,g}^{\mathsf T}d \right\rangle
$$

初版为了压缩结果，只保存每例每层前三个候选，再用稀疏记录做发现集聚合。审计发现部分入选候选只覆盖 1 到 2 个独立对象，而且正向率被选择过程推到 100%。该结果被废弃，没有进入最终结论。

修正后：

```text
发现集保存并聚合全部注意力头和全部 32 个多层感知机乘积组；
校准和留出只保存前三名用于压缩文件；
每个最终冻结成员必须覆盖 4 个独立对象、3 个模板、12 次自然观测；
288/288 个冻结成员满足该覆盖；
发现集正向率恢复为 0.416667 到 1.0 的实际分布。
```

每个模型、每个机制冻结：

```text
source、query、last 三个位置角色；
每个角色 2 个注意力头输入候选；
每个角色 2 个多层感知机乘积组候选；
合计 12 个分布式集合成员。
```

注意：注意力与多层感知机、三个位置角色数量相等是实验设计，不是自然分布发现。

### 词元跨度错误与修复

第一次正式运行后进行了逐例解码审计，发现慢速 BPE/SentencePiece 分词器在上下文中会把对象前的空格与首词元合并。原前缀长度算法把部分知识对象的 source 位置落在后续问号或句号上。

该问题意味着末位置证据仍有效，但 source/query 路径证据不可信。前两轮结果因此没有直接保留。

修复方法：上下文前缀长度回退一个合并词元，并对三个分词器逐例解码校验。最终结果：

```text
每个模型案例：416；
每模型 source/query 跨度：832；
三模型跨度总数：2496；
词元词面重叠低于 0.5：0；
知识答案显式泄漏：0。
```

修复后重新按 qwen3 -> GLM4 -> DS7B 顺序完整运行主批次和扩大确认批次。GLM4 确认批次首次加载后出现一次底层段错误；检查无残留显存进程后单独重跑成功，未跳过模型，未混用半成品结果。

### 干预与严格通过标准

主批次条件：

```text
baseline；
single_attention_zero；
attention_set_zero；
single_mlp_zero；
mlp_set_zero；
joint_set_zero；
matched_random_joint_zero；
wrong_layer_joint_zero。
```

集合必要性效应：

$$
\Delta_{set}=m_{base}-m_{zero(set)}
$$

其中 $m$ 为目标首词元相对最强干扰首词元的差值。

匹配对照特异性：

$$
S=\Delta_{set}-\max(\Delta_{random},\Delta_{wrong\_layer})
$$

分布式增益：

$$
G=\Delta_{set}-\max(\Delta_{single\_attn},\Delta_{single\_mlp})
$$

主批次每模型每机制至少需要 3 个基线正确留出案例，并同时满足正向效应、匹配对照特异性、超过单组件和方向一致性。

扩大确认进一步要求：

```text
至少 12 个基线正确独立对象；
两个新模板分别通过；
集合效应、匹配特异性和分布增益均为正；
方向一致率至少 0.70；
初始留出必须已经通过；
扩大确认不能覆盖初始留出失败。
```

### 三模型自然基线

```text
qwen3：0.833333；
GLM4：0.784722；
DS7B：0.840278。
```

全部 144 个推理提示/模型均使用多词元规则事实源组。三个模型的知识答案泄漏均为 0。

### 主批次严格留出结果

跨模型组级必要性候选：

```text
category_retrieval：GLM4 + DS7B；
color_retrieval：qwen3 + GLM4；
habitat_retrieval：GLM4 + DS7B。
```

单模型通过但未跨模型：

```text
transitive_order：qwen3；
implication_chain：GLM4。
```

其余机制未通过。推理四机制没有一个跨模型复现。

自然能量门控观测只有 GLM4/material 和 DS7B/habitat、spatial 等零散单模型结果，没有形成同一机制的跨模型因果门控。该字段维持 L3 观测，不升级证据。

### 扩大确认结果

严格标准要求同一模型同时通过初始留出和扩大确认。最终跨模型结果：

```text
类别检索：GLM4 + DS7B；
颜色检索：qwen3 + GLM4；
栖息地检索：GLM4 + DS7B；
材料检索：没有模型同时满足两个阶段，未确认。
```

部分关键数值：

```text
qwen3/color：joint=0.574951，specificity=0.519775，gain=0.119629，consistency=0.96875；
GLM4/color：joint=0.863542，specificity=0.833724，gain=0.476953，consistency=0.866667；
GLM4/category：joint=0.482918，specificity=0.348263，gain=0.304308，consistency=0.766667；
DS7B/category：joint=0.385417，specificity=0.306135，gain=0.328704，consistency=0.740741；
GLM4/habitat：joint=0.721670，specificity=0.667739，gain=0.624238，consistency=0.88；
DS7B/habitat：joint=1.125000，specificity=0.835227，gain=0.464489，consistency=0.772727。
```

负面校准同样重要：

```text
DS7B/color 基线只有 0.53125，集合效应 -0.150735；
qwen3/habitat 的模板 D 一致率只有 0.615385；
qwen3/material 的模板 D 匹配特异性为负；
DS7B/material 集合效应 -0.512228；
因此不能把某一模型中的知识关系载体直接外推到其他模型。
```

### 物理分布结果

最终冻结候选集中在后段：

```text
qwen3：L35 80 个，L28 16 个；
GLM4：L39 58 个，L31 33 个，L23 5 个；
DS7B：L27 84 个，L22 12 个。
```

这个分布不能解释为完整知识检索路径。直接读出贡献评分天然偏向末层，因此 Phase326 得到的是“晚层分布式读出载体集合”，不是“对象触发 -> 参数知识检索 -> 读出”的完整物理路径。

### 图谱数据与客户端同步

Phase325 的 833 个真实单元候选被保留。Phase326 新增的对象不被伪装成神经元，而明确标记：

```text
node_type = component_set_member；
unit_kind = attention_head 或 mlp_product_group；
causal_scope = distributed_component_set_not_single_unit；
single_unit_causal = false。
```

客户端固定包：

```text
模式族总数：9；
具有物理候选分区：2；
物理候选节点：1121；
Phase286/287 单元候选：833；
Phase326 组件集合成员：288；
同时通过初始留出与扩大确认的集合成员：72；
非因果边：1207；
自然组件事件：318；
图谱内组级干预/审计记录：3560；
单神经元因果计数：0。
```

内容知识模式族分区包含既有颜色真实单元候选和四类知识关系集合成员。推理约束模式族首次获得三模型物理组件候选分区，但由于没有跨模型必要性复现，全部保持 L3 或单模型 L4 边界。

客户端改造：

```text
标题由“神经元脉络”改为“物理脉络”；
区分 unit_candidate 与 component_set_member；
增加“扩大确认”证据筛选；
绿色只表示同时通过初始留出和扩大确认，仍不是单组件因果；
多机制同层锚点按层聚合，避免标签重叠；
详情显示组件集合身份、证据边界和是否扩大确认；
未映射模式族继续显示明确空状态。
```

### 验证

案例与数据测试：

```text
phase326 case-bank tests：3 passed；
Phase325 compatibility tests：2 passed；
Phase326 published-atlas tests：2 passed；
图谱验证：2 mapped families / 1121 nodes / 1207 edges；
所有图谱边 causal=false；
single_unit_causal_count=0。
```

客户端验证：

```text
定向 ESLint：通过；
npm run build：通过；
Playwright 内容知识/推理约束切换和扩大确认筛选：通过；
Playwright 390x844 边界与无横向页面溢出：通过；
桌面与窄屏截图：画布非空，层锚点无重复标签，控制面板可见；
生产构建仍有既有主包超过 500KB 警告，但构建成功。
```

本地开发服务继续运行：

```text
http://127.0.0.1:5173/
```

### 主要硬伤

```text
1. 直接读出评分偏向末层，只能作为读出载体发现器；
2. 集合同时包含多个层、三个位置角色和两类组件，整体置零可能产生广泛功能损伤；
3. 匹配随机组和错层组降低但不能消除分布外干预问题；
4. 当前只比较候选答案首词元，没有验证完整自然生成和多词元答案；
5. MLP product group 不是单神经元，集合通过不能分配因果给每个成员；
6. 自然能量差不是因果门控，尚无自然充分性移植；
7. 推理四机制无跨模型复现，当前方法没有恢复推理物理路径；
8. 三个小模型的精确地址和效应差异很大，可能存在 30% 到 50% 的结构偏差；
9. 本地模型修订仍为 local_unknown，需要增加权重稳定哈希；
10. 图谱层间边只是候选顺序，不是已证明的神经元传播边。
```

### 智能理论与第一性原理校准

本阶段支持但没有证明以下工作假设：

```text
知识回答不依赖单一概念方向或单神经元开关；
正确读出依赖跨注意力与多层感知机的晚层分布式载体集合；
同一知识机制的精确载体具有明显模型条件性；
不同知识关系可能复用晚层读出骨架，再由对象、关系和上下文条件形成差分选择。
```

当前线性公式只是候选排序和干预测量工具，不是真实语言运行机制公式。不能因为集合必要性通过，就把语言编码理论总结为线性方向叠加。

更接近当前拼图的工作结构仍是：

```text
SourceGroup
-> DistributedCarrierSet
-> ConditionalBoundary
-> PhraseReadout
-> Rollout
```

但 Phase326 只较强地补充了 `DistributedCarrierSet -> PhraseReadout` 的晚层必要性候选。SourceGroup 到载体集合的自然检索路径、ConditionalBoundary 的自然门控和 Rollout 的完整充分性仍缺失。

### 当前进度

优先使用可核验比例：

```text
物理候选模式族覆盖：2/9 = 22.2%；
严格扩大确认模式族覆盖：1/9 = 11.1%；
Phase326 测试机制严格跨模型通过：3/8 = 37.5%；
单神经元因果闭合：0%；
自然门控因果闭合：0%；
L5 机制闭合：0。
```

若只用于项目排期而必须给出粗略总值：

```text
语言模式族物理分布图谱：约 20% 到 26%；
语言编码机制：约 12% 到 18%；
严格自然闭合：约 0% 到 5%；
智能理论实验验证：约 15% 到 22%。
```

这些范围是项目管理估计，不是实验测量，不能替代分项计数。

### 下一阶段注册方向

Phase326 的直接阶段目标已经完成，并自动完成了必要的扩大确认。下一任务仍属于“语言模式族物理分布拼图”大阶段，但已经形成新的因果链目标，应注册为 Phase327：

```text
自然对象触发
-> 早中层检索变化
-> 晚层冻结载体集合
-> 完整答案生成
```

执行顺序：

```text
1. 只聚焦严格跨模型确认的颜色、类别、栖息地；
2. 用正确对象与匹配错误对象的自然差分定位早中层变化，不再用直接读出分数寻找整条路径；
3. 分开测试 source、query、last 的必要性，不允许一次整体置零后直接解释传播；
4. 对冻结集合做自然状态移植，并使用错误供体、随机供体和错层供体测试充分性；
5. 验证完整自然生成，不限于答案首词元；
6. 只有集合同时通过必要性、充分性、自然门控和跨模板后，才展开到单神经元 CUDA 干预；
7. 只有上游干预稳定改变下游冻结集合和最终答案时，才生成 causal=true 的路径边；
8. 推理模式族另行改进源组和任务设计，不复用知识检索的末层读出补丁。
```

### 阶段结论

```text
Phase326 将物理候选覆盖从 1/9 模式族扩展到 2/9，并建立了隐式知识与多词元推理的统一冻结分母；
三项知识机制出现跨两个模型、跨初始留出、16 个新对象和两个新模板的分布式集合必要性候选；
推理机制、自然门控、自然充分性和单神经元因果均未闭合；
客户端已经准确区分单元候选与组件集合成员，研究结果和三维显示保持一致；
因此这是“晚层分布式读出载体图谱”的实质进展，不是完整语言编码机制或智能理论完成。
```

## Phase 327: 自然对象到冻结载体集合及完整生成的注册链验证 [2026-07-09 23:40]

### 附件判断审计

附件对 Phase326 的核心判断基本正确：研究已经从单头、单通道和单点读出推进到分布式载体集合、隐式知识检索、扩大对象和模板确认，但仍然只是晚层读出载体图谱，不是完整语言编码机制。

需要收紧两点：

```text
附件列出的 98 项“核心拼图”只能作为历史研究目录，不能解释为 98 个已经闭合的机制；
Phase326 的自然门控、自然充分性、完整生成、上游传播和单神经元因果都仍为 0。
```

因此保留附件提出的直接任务，并一次完成：

```text
自然对象触发
-> source/query/last 残差身份变化
-> Phase326 冻结晚层载体集合
-> 完整目标词串概率
-> 全词表贪心自然生成。
```

### 冻结分母

只测试 Phase326 已经跨模型扩大确认的三个知识机制：

```text
color_retrieval；
category_retrieval；
habitat_retrieval。
```

独立样本：

```text
每机制 18 个新对象；
三机制共 54 个对象；
每对象 2 个新完形模板；
每模型 108 个注册提示；
三模型共 324 个提示-模型案例。
```

每个提示构造五个自然条件：

```text
正确对象；
同答案对象；
同语义类错误答案对象；
长度匹配错误答案对象；
无关错误答案对象。
```

固定数据量：

```text
自然条件行：1620；
逐层残差行：134784；
位置干预行：2916；
自然状态移植行：1944；
完整生成行：648；
机制审计：9；
图谱自然路径：9。
```

自动验证：

```text
目标答案泄漏：0；
与 Phase326 同机制对象重叠：0；
重复案例编号：0；
自然控制目标错误：0；
source 位于 query 之后的因果顺序错误：0；
载体集合重新选择：0。
```

### 模板顺序校准

第一次完形烟雾运行发现 query 词位于对象 source 之前。由于这是因果语言模型，query 状态不可能包含后出现对象的信息，导致 query 差分结构性为 0。

这批中间结果被废弃，不进入最终图谱。最终注册模板强制：

```text
For the {subject}, the common {relation} is
Regarding the {subject}, its typical {relation} is
```

因此 source 在 query 之前，query 允许接收对象信息。验证器新增顺序硬约束，最终 108 个提示全部通过。

### 基础测量公式

自然载体身份特异性：

$$
C
=
\overline{\cos(z_{correct},z_{same\ target})}
-
\overline{\cos(z_{correct},z_{wrong\ target})}
$$

逐层残差身份特异性：

$$
R_{l,r}
=
\overline{\operatorname{RMS}(h^{correct}_{l,r}-h^{wrong}_{l,r})}
-
\overline{\operatorname{RMS}(h^{correct}_{l,r}-h^{same}_{l,r})}
$$

完整目标词串必要性下降：

$$
D_c
=
\log P(y\mid x)_{base}
-
\log P(y\mid x)_{c}
$$

位置集合匹配控制特异性：

$$
S_{pos}
=
D_{joint}
-
\max(D_{random},D_{wrong\ layer})
$$

自然供体移植增益：

$$
G_d
=
\log P(y\mid x,do(z\leftarrow z_d))
-
\log P(y\mid x)
$$

自然供体特异性：

$$
S_{donor}
=
\min(G_{correct},G_{same\ target})
-
\max(G_{wrong},G_{unrelated})
$$

这些公式只是基础操作量，不是语言编码统一公式。

### 四批实验

批次 A：

```text
比较正确对象、同答案对象和三类错误对象；
保存全部层 source/query/last 残差；
读取冻结载体集合自然状态；
不使用直接反嵌入方向选择早中层。
```

批次 B：

```text
baseline；
source/query/last 分别置零；
source+query、query+last、三角色联合置零；
匹配随机集合；
错层集合；
同时测候选边距、完整词串概率和全词表副作用。
```

批次 C：

```text
把正确对象、同答案对象、同语义错误对象和无关对象的自然载体状态移植到错误对象接收者；
加入正确供体错层移植；
不使用人工目标方向作为供体。
```

批次 D：

```text
对正确基线、联合置零、错误接收者基线、正确供体移植做全词表贪心生成；
目标是一词答案，但保留后续自然续写，不把候选首词元胜出当作生成成功。
```

### 最终跨模型结果

```text
颜色：
自然身份通过模型 0；
位置必要性通过 Qwen3、GLM4；
自然状态移植通过 Qwen3、GLM4；
完整生成只通过 GLM4；
完整链 0。

类别：
自然身份通过 Qwen3、GLM4；
位置必要性只通过 Qwen3；
自然状态移植三个模型全部通过；
完整生成 0；
完整链 0。

栖息地：
自然身份只通过 Qwen3；
位置必要性只通过 DS7B；
自然状态移植三个模型全部通过；
完整生成 0；
完整链 0。
```

严格跨模型计数：

```text
自然身份：1/3 机制；
位置必要性：1/3 机制；
自然状态移植：3/3 机制；
完整生成：0/3 机制；
完整自然链：0/3 机制；
单神经元因果：0。
```

### 物理分布拼图

类别机制在三个模型中都出现较清楚的 query 身份差分。第一批 12 个对象、前 70% 层范围内的最大值分别位于：

```text
Qwen3：L24/35；
GLM4：L27/39；
DS7B：L19/27。
```

三者都位于搜索上边界附近。因此只能记录为“前 70% 范围最大值”，不能宣称这些层是真实局部峰值。

Phase326 冻结集合的位置必要性主要来自 last 位置。source 和 query 的集合置零效应大多接近 0。这说明 Phase326 集合更像晚层读出状态，而不是已经证明的 source 到 query 传播通道。

颜色机制在三个模型中自然身份特异性都为负，但 Qwen3/GLM4 的晚层置零和自然移植仍有效。最谨慎解释是：当前颜色集合可能承载通用颜色关系或读出支持，不是稳定的具体颜色身份编码。

### CUDA 运行校准

三个模型严格按 Qwen3、GLM4、DS7B 顺序运行并逐个释放。

DS7B 第一次长进程在自然移植 90/108 后由 `libcudart.so.12` 通用保护错误退出，显存并未溢出。脚本增加正式可恢复接口：

```text
A/B 单独完成；
C/D 每 36 个提示一个 CUDA 进程；
三块分别产出 216 条移植和 72 条生成记录；
合并前强制验证 648/216 行；
没有跳过案例或改变顺序。
```

### 图谱和客户端

固定物理图谱保持原库存：

```text
映射模式族：2/9；
物理候选：1121；
单元候选：833；
组件集合成员：288；
扩大确认成员：72；
新增虚假神经元：0；
单神经元因果：0。
```

Phase327 只增加：

```text
9 条自然检索路径；
0 条严格自然闭合；
所有路径 causal=false。
```

客户端继续使用原 36 层 DNN 形状。新增状态附着在既有集合成员上，未测试的 material 机制明确标记 `phase327_not_in_registered_scope`，避免把未测试误写成失败。

### 核心硬伤

```text
1. Phase326 候选由直接反嵌入归因选择，天然偏向晚层读出；
2. 最后位置主导效应不能解释为完整自然检索传播；
3. 多组件自然状态移植可能覆盖宽泛状态，不是单组件或单神经元充分性；
4. 候选答案内准确率与全词表生成严重分离；
5. 长度匹配控制不等于训练频率匹配；
6. 当前三个模型较小，内部机制可能与更大模型有显著偏差；
7. 没有任何单神经元 CUDA 因果结果。
```

### 阶段判断

Phase327 完成了 Phase326 之后注册的自然链验证，得到自然身份、位置必要性和自然状态移植的客观拼图，但完整生成和完整链为 0。不能升级语言编码机制闭合，也不能开始全量单神经元干预。

由于类别自然身份已经跨 Qwen3 和 GLM4 出现，直接下一步仍属于同一物理路径阶段，自动继续 Phase328，验证上游 query 残差是否真正改变下游冻结集合和全词表答案。

## Phase 328: 类别 query 残差到冻结载体集合的独立留出中介验证 [2026-07-09 23:46]

### 注册原因

Phase327 证明类别机制的自然身份在 Qwen3 和 GLM4 复现，但没有因果证明：

```text
自然对象差分
-> query 残差差分
-> Phase326 冻结载体集合
-> 全词表答案。
```

Phase328 不扩展到颜色、栖息地，也不重新搜索单元，只测试类别机制。

### 发现和验证严格分离

发现分母：

```text
Phase327 前 12 个类别对象；
每对象两个模板；
每模型 24 个提示；
只使用 query 角色；
只搜索前 70% 层；
选择更新被禁止。
```

独立验证分母：

```text
Phase327 后 6 个类别对象；
每对象两个模板；
每模型 12 个提示；
三模型 36 个验证提示；
每提示 6 个条件；
总干预行 216。
```

冻结 query 层：

```text
Qwen3：L24；
GLM4：L27；
DS7B：L19。
```

这些都是前 70% 搜索边界附近最大值，不解释为局部峰值。

### 干预条件

```text
错误对象接收者基线；
正确对象 query 残差移植；
同答案对象 query 残差移植；
同语义错误对象 query 残差移植；
无关对象 query 残差移植；
正确对象错层 query 残差移植。
```

移植发生在 Transformer 层输入残差。供体使用角色跨度平均向量并广播到接收者 query 位置。随后同时读取：

```text
完整目标词串概率；
候选目标边距；
全词表目标排名；
全词表 top-1；
下游 Phase326 冻结载体状态与正确供体的相似度；
错供体、无关供体和错层控制。
```

### 客观结果

Qwen3：

```text
正确供体词串增益：+0.212760；
同答案供体词串增益：+0.212167；
错误供体：+0.094682；
无关供体：+0.079584；
目标全词表排名平均提高：93.25；
正确供体载体相似度增益：-0.005534；
错层供体载体相似度增益：+0.041246；
中介判据：失败；
top-1 解锁：失败。
```

GLM4：

```text
正确供体词串增益：+0.159158；
同答案供体词串增益：+0.175318；
错误供体：-0.001283；
无关供体：+0.047404；
错层供体：+0.086561；
正确供体载体相似度增益：+0.029307；
错误供体载体相似度增益：-0.008009；
无关供体载体相似度增益：-0.016939；
错层供体载体相似度增益：+0.018191；
目标全词表排名平均提高：425.166667；
中介判据：通过；
top-1 解锁：失败。
```

DS7B：

```text
正确供体词串增益：+0.351028；
同答案供体词串增益：+0.180319；
错误供体：+0.285869；
无关供体：+0.311998；
目标全词表排名平均提高：1687.833333；
正确供体载体相似度增益：-0.003540；
中介判据：失败；
top-1 解锁：失败。
```

跨模型结果：

```text
上游残差中介通过模型：仅 GLM4；
自然 top-1 解锁模型：0；
模型级因果边候选：0；
跨模型因果边复现：false；
L5：0；
单神经元因果：0。
```

### 关键解释

三个模型的正确 query 残差都能提高目标全词表排名，说明 Phase327 的类别 query 差分不是完全无关的观测量。

但是排名提高不等于生成。三个模型移植后目标 top-1 仍全部为 0，说明目标前方还存在强全词表竞争者。Phase326 的候选目标边距无法描述这一瓶颈。

只有 GLM4 同时满足：

```text
正确/同答案供体优于错误和无关供体；
正确层优于错层；
目标词串概率稳定提高；
下游冻结载体更接近正确供体；
目标全词表排名提高。
```

因此 GLM4 形成单模型 pooled query residual state（池化查询残差状态）到 distributed carrier set（分布式载体集合）的中介候选。由于没有跨模型复现且没有自然 top-1 解锁，图谱边必须保持 `causal=false`。

### 物理图谱和客户端同步

最新图谱 manifest 升级到 Phase328，并保留原 DNN 形状和全部物理数量：

```text
物理候选：1121；
组件集合成员：288；
自然检索路径：9；
上游残差中介候选边：3；
中介通过模型：1；
跨模型因果路径边：0；
严格自然链：0；
单神经元因果：0。
```

客户端节点详情新增：

```text
Phase327 自然身份、位置必要性、自然移植和生成状态；
Phase328 冻结 query 层、中介通过状态、自然 top-1 解锁状态和因果边状态。
```

控制面板显示自然路径、严格自然闭合、上游中介候选和因果路径边数量。没有新增神经元球，没有改变基础 36 层 DNN 几何。

### 工程验证

```text
Python 合并测试：17 passed；
Phase327/328 定向单元测试：通过；
固定图谱发布验证：通过；
前端生产构建：3668 modules transformed，通过；
定向 ESLint：通过；
git diff --check：通过；
桌面 Playwright 截图：1440x960，358KB 以上，画布非空；
移动 Playwright 截图：390x844，100KB 以上，画布非空；
人工检查：原 DNN 层栈、候选叠层和移动控制面板均可见，无几何替换和明显重叠。
```

完整 `@playwright/test` 套件未执行，因为当前项目依赖中没有安装该包；使用已可用的 Playwright Chromium 命令完成桌面和移动真实页面截图。前端构建和数据契约测试均已通过。

本地客户端：

```text
http://127.0.0.1:5173/
```

### 严格硬伤

```text
1. query 残差使用角色跨度平均并广播，不是真实逐词元状态移植；
2. 三个选择层都接近 70% 搜索边界，真实峰值可能更晚；
3. 下游载体相似度使用平均余弦，可能漏掉符号、子组件和逐词元中介；
4. GLM4 只是单模型通过，不能解释为架构不变机制；
5. 巨大排名增益可能部分来自错误接收者基线排名很低；
6. top-1 全部未解锁，完整自然答案仍未形成；
7. 当前小模型编码较粗糙，结论可能与更大模型有 30% 到 50% 的结构偏差；
8. 单神经元 CUDA 因果仍未开始，因为上游路径边没有跨模型闭合。
```

### 对智能理论的约束

当前结果继续支持“语言是动态模式网络”的工作框架，但对机制表述增加重要约束：

```text
对象身份可以在约 70% 深度附近形成 query 残差差异；
残差状态可以提高目标排名；
晚层冻结载体可以改变目标词串概率；
但残差身份、载体状态和自然生成不是一个可互换的线性方向。
```

因此不能把当前统一机制写成单一线性加法：

$$
h + \alpha d \Rightarrow y
$$

更准确的待验证结构仍是分阶段条件系统：

$$
O
\rightarrow
H^{query}_{mid}
\rightarrow
Z^{carrier}_{late}
\rightarrow
\operatorname{Rank}_{V}(y)
\rightarrow
Y_{generated}
$$

其中每条箭头都必须分别通过自然身份、匹配控制、干预中介和自然生成。Phase328 只在 GLM4 对中间两步形成单模型迹象，整体链没有闭合。

### 进度口径

只按可直接计数的当前结果：

```text
物理候选模式族覆盖：2/9 = 22.2%；
三个知识机制严格自然链：0/3；
跨模型上游因果路径边：0；
单神经元因果闭合：0；
完整语言编码机制：未完成；
智能理论实验闭合：未完成。
```

Phase327/328 没有增加模式族覆盖或神经元因果覆盖，因此不应上调总体完成百分比。它们提高的是证据分层和瓶颈定位质量。

### 下一阶段注册方向

当前阶段目标已完成。继续工作需要改变研究分母，应注册为 Phase329，而不是继续给 Phase328 增加补丁：

```text
1. 保存每例全词表 top-50，区分语义竞争者、续写词、标点和协议词；
2. 把候选答案边距升级为全词表目标排名和竞争者族图谱；
3. 用逐词元 query 残差移植替代跨度平均广播；
4. 加入范数匹配残差、错误词元排列和同层随机方向控制；
5. 要求上游干预同时改变逐组件载体身份、目标排名和自然生成；
6. 只有残差到载体的中介跨至少两个模型复现后，才定位上游 attention/MLP 组件；
7. 只有跨模型因果路径边形成后，才启动全量单神经元 CUDA 干预。
```

### 阶段结论

```text
Phase327 完成了自然对象到晚层冻结载体及自然生成的全链验证；
Phase328 自动完成了类别 query 残差到冻结载体的独立留出中介测试；
自然状态移植比自然身份、位置必要性和完整生成更容易通过；
GLM4 出现单模型上游中介候选，但三个模型都没有自然 top-1 解锁；
当前语言模式族物理图谱新增了真实研究状态，没有新增虚假神经元或因果边；
完整语言编码机制、单神经元因果和智能理论均未闭合。
```

## Phase 329: 全词表竞争者、逐词元查询中介与物理图谱同步 [2026-07-10 00:41]

### 阶段原因与输入判断

Phase327/328 被判断为“重要负结果加局部正结果”总体正确：类别 query 残差可以提高目标排名，但没有解锁自然 top-1，也没有形成查询残差、冻结载体和完整生成之间的闭合链。

本阶段接受并系统执行以下正确方向：

```text
保存全词表 top-50 竞争词元；
用全词表目标排名替代候选集合边距；
用逐词元 query 状态替代查询跨度平均广播；
加入同目标、错目标、无关、范数匹配、词元反转和错层控制；
逐成员读取冻结载体；
同时验证自然生成；
未跨模型闭合前不启动全量单神经元 CUDA 干预。
```

同时收紧三点：

```text
1. Phase328 的残差观测层输出被注入同编号层输入，存在一层接口偏移；
2. 功能词、标点和格式词是表面协议竞争者，不能自动解释为语义阻挡机制；
3. 拼图目录数量不是机制闭合数量。
```

### 注册样本与固定合同

新增完全独立对象：

```text
颜色：12 个；
类别：12 个；
栖息地：12 个；
合计：36 个；
每对象 2 个新模板；
每模型 72 个提示；
三个模型合计 216 个提示模型实例；
对象模型实例 108；
自然变体 288 个；
与 Phase326/327 同机制对象重叠 0；
目标词泄漏 0；
因果顺序错误 0。
```

所有选择都来自 Phase327 注册主样本，并禁止在 Phase329 独立样本上更新。

模型按以下顺序独立加载、测试和释放：

```text
qwen3 -> GLM4 -> DS7B
```

### 残差层算法和接口修正

对每层查询角色计算：

$$
S_k=overline{D}_{k,\mathrm{wrong}}-overline{D}_{k,\mathrm{same}}
$$

其中：

$$
D_k(a,b)=\sqrt{\frac{1}{d}\sum_{j=1}^{d}(h_{k,j}^{a}-h_{k,j}^{b})^2}
$$

只在前 70% 层中选择最大值。若最大值不大于零，则把该机制保留为注册负分支，不升级为身份机制。

Phase329 按计算接口把观测层输出注入下一层输入：

$$
H^{query}_{k,\mathrm{out}}
\longrightarrow
H^{query}_{k+1,\mathrm{in}}
$$

这意味着 Phase328 的 GLM4 单模型中介迹象不能原样并入 Phase329 对齐证据，必须降级为旧接口候选。

冻结结果：

```text
颜色：三个模型最大身份特异性均为负；

类别：
qwen3 观测 L24 -> 注入 L25，S=0.301019；
GLM4 观测 L27 -> 注入 L28，S=0.110123；
DS7B 观测 L19 -> 注入 L20，S=0.227848；

栖息地：
qwen3 观测 L24 -> 注入 L25，S=0.098135；
GLM4 观测 L27 -> 注入 L28，S=0.054001；
DS7B 观测 L19 -> 注入 L20，S=0.087887。
```

### 逐词元和均值广播算法

逐词元移植保留 query 跨度顺序：

$$
H^{recipient}_{q,1:m}\leftarrow H^{donor}_{q,1:m}
$$

均值广播为：

$$
H^{recipient}_{q,i}\leftarrow
\frac{1}{m}\sum_{j=1}^{m}H^{donor}_{q,j},
\quad i=1,\ldots,m
$$

同一矩阵中的条件：

```text
正确自然基线；
正确载体联合置零；
错误接收者基线；
正确逐词元 query；
正确均值广播 query；
同目标逐词元 query；
同语义错目标逐词元 query；
无关逐词元 query；
范数匹配无关 query；
正确 query 词元反转；
正确 query 错层移植；
正确自然载体移植。
```

严格“逐词元优于均值”要求：

```text
词串概率增益大于零；
全词表排名增益大于零；
两项都严格优于均值广播。
```

因此颜色中“两种操作都变差但逐词元变差较少”不被记为正结果。

### 全词表阻挡者算法

目标首词元阻挡者定义：

$$
\mathcal{B}(y)=\{v\in V:z_v>z_y\}
$$

所以：

$$
|\mathcal{B}(y)|=\operatorname{Rank}_V(y)-1
$$

阻挡者下降：

$$
\Delta B=B_{recipient}-B_{condition}
$$

每条件保存前 50 词元，并使用运行前冻结的类型：

```text
目标；
目标别名；
注册错误答案；
标点或空白；
功能续写词；
协议或格式词；
对象复制；
其他内容词。
```

前 50 只负责竞争者类型样本。精确全词表阻挡者总数来自目标排名，不能声称已经分类全部词表阻挡者。

### 载体成员算法

每个冻结成员独立计算向正确自然状态靠近的变化：

$$
G_m=cos(c_m^{correct},c_m^{condition})-cos(c_m^{correct},c_m^{recipient})
$$

只有以下条件同时满足才记为载体成员中介：

```text
正确和同目标供体均优于全部控制；
成员多数向正确状态变化；
成员变化和排名变化在样本上同向；
全词表排名正向改善。
```

### 固定产物计数

全部输出通过精确计数：

```text
条件与排名行：2592；
前 50 竞争词元行：129600；
载体成员行：31104；
自然生成行：648；
模型机制审计：9；
冻结残差选择：9。
```

核心产物：

```text
tests/gpt5/phase329_full_vocabulary_case_bank.py
tests/gpt5/phase329_full_vocabulary_mediation.py
tests/gpt5/phase329_publish_full_vocabulary_atlas.py
tests/gpt5/run_phase329_full_vocabulary_mediation.sh
tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329A_rank_transition_rows.jsonl
tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329A_top50_competitors.jsonl
tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329A_blocker_types.jsonl
tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329A_condition_summaries.jsonl
tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329A_rank_band_summaries.jsonl
tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329B_tokenwise_query_transplant_rows.jsonl
tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329C_carrier_member_mediation_rows.jsonl
tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329D_full_generation_rows.jsonl
tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329_cross_model_summary.json
tests/gpt5/result/phase329_full_vocabulary_mediation/full_vocabulary_mediation/phase329_report.md
```

### 客观结果：逐词元与排名

```text
qwen3 颜色：逐词元词串 -0.031453，排名 -0.166667；失败；
qwen3 类别：逐词元词串 +0.241043，均值 +0.189577；排名 +243.125，均值 +232.541667；模型内通过；
qwen3 栖息地：逐词元词串 +0.129987，均值 +0.343174；失败；

GLM4 颜色：逐词元词串 +0.029920，排名 0；失败；
GLM4 类别：逐词元词串 +0.184978，均值 +0.193244；失败；
GLM4 栖息地：逐词元词串 +0.200149，均值 +0.157048；排名 +490.75，均值 +160.166667；模型内通过；

DS7B 颜色：逐词元词串 -0.019662，排名 -0.75；失败；
DS7B 类别：逐词元词串 +0.050749，排名 +962.916667，但均值排名 +2911.416667；失败；
DS7B 栖息地：逐词元词串 -0.028601，排名 +210.416667；失败。
```

跨模型逐词元优于均值机制：

```text
0/3。
```

### 客观结果：阻挡者

```text
类别阻挡者下降：qwen3、GLM4 通过；
栖息地阻挡者下降：qwen3、GLM4 通过；
颜色：0 模型通过；
DS7B：0 机制通过完整一致性和副作用条件。
```

类别排名高于 1000 的提示：

```text
qwen3：20/24 -> 15/24；
GLM4：16/24 -> 13/24；
DS7B：24/24 -> 24/24。
```

因此“平均排名大幅提高”只是部分竞争者减少，不是答案形成。

### 客观结果：成员、第一候选和生成

```text
载体成员中介：0/9 模型机制；
跨模型载体成员中介：0/3 机制；
跨模型 top-1 解锁：0/3；
跨模型自然生成改善：0/3；
跨模型完整链：0/3；
单神经元干预门：关闭；
单神经元因果：0。
```

类别与栖息地的 top-1 多为表面协议：

```text
qwen3 类别：24/24 功能续写词；
qwen3 栖息地：24/24 功能续写词；
GLM4 栖息地：24/24 功能续写词；
DS7B 类别：主要为格式词、标点或空白。
```

正确对象基线的首词元命中：

```text
颜色：qwen3 66.7%，GLM4 70.8%，DS7B 12.5%；
类别：三个模型均为 0；
栖息地：三个模型均为 0。
```

但正确栖息地别名在四词元生成任意位置出现：

```text
qwen3 41.7%；
GLM4 66.7%；
DS7B 33.3%。
```

这说明当前首词元接口混合了：

```text
语义内容竞争；
冠词和连接词；
引号和标点；
模型回答协议。
```

Phase329 可靠证明首词元未解锁，但不能把全部表面协议词解释为语义抑制器。

### 跨模型总结果

```text
跨模型逐词元优于均值：0/3；
跨模型阻挡者下降：2/3（类别、栖息地）；
跨模型成员中介：0/3；
跨模型 top-1 解锁：0/3；
跨模型生成改善：0/3；
跨模型完整链：0/3；
closure_claim=false；
single_unit_causal_count=0。
```

最可靠的新拼图：

```text
类别和栖息地的中层 query 状态能够在 qwen3/GLM4 中减少目标前方的全词表竞争者；
逐词元相对均值的优势不跨模型；
冻结载体成员不能中介该变化；
第一候选和完整生成没有解锁。
```

### 物理图谱和客户端同步

图谱 manifest 升级到 Phase329，保留原 DNN 几何和全部物理地址：

```text
物理候选：1121；
真实单元候选：833；
冻结组件集合成员：288；
扩大确认候选：72；
自然检索路径：9；
全词表中介路径：9；
跨模型阻挡者下降机制：2；
跨模型成员中介：0；
跨模型 top-1 解锁：0；
跨模型完整链：0；
单神经元因果：0。
```

客户端改动：

```text
保留原 36/40/28 层 DNN 形状；
新增“竞争路径”筛选；
在既有组件成员上显示观测层、实际注入层、阻挡者下降、成员中介、top-1、生成和单神经元门；
没有新增虚构神经元球；
没有新增因果边；
底部控制和指标允许稳定换行，桌面与移动端无横向滚动。
```

### 工程验证

```text
Phase327-329 定向 Python 测试：21 passed；
Phase329 原始行精确计数：通过；
固定图谱发布验证：通过；
研究目录与客户端公开目录逐文件一致：通过；
git diff --check：通过；
前端 ESLint：通过；
前端生产构建：3668 modules transformed，通过；
桌面真实交互画布：1440x857；
移动真实交互画布：500x789；
演示数据自动加载：通过；
Phase329 竞争路径筛选激活：通过；
原 36 层 Qwen3 三维形状和物理候选叠层可见。
```

前端仍有既有的大体积构建块警告，但不影响本阶段构建成功和功能验证。

本地客户端：

```text
http://127.0.0.1:5173/
```

### 严格硬伤

```text
1. Phase328 一层接口偏移使旧 GLM4 中介迹象必须降级；
2. 类别和栖息地的正确基线首词元接口本身不合格；
3. 前 50 不是全词表竞争者的完整类型分类；
4. 冻结成员仍是头输入块和 MLP 分组，不是单神经元；
5. 选择层处于 70% 搜索边界或颜色早层边界，不是已证明的局部峰值；
6. DS7B 排名移动大但控制不特异，平均 JS 散度在三个机制中约 0.0527、0.0612、0.0551；
7. 英文模板结果尚未证明跨语言和跨协议不变；
8. 小模型内部结构可能比大型模型粗糙，可能存在 30% 到 50% 的结构偏差；
9. 结果不支持单一线性方向，也不支持把冻结载体称为完整知识存储位置。
```

### 智能理论约束

当前“语言是动态模式网络”仍只能作为工作框架。机制至少要包含：

$$
(O,T)
\rightarrow H^{query}_{k,1:m}
\rightarrow C^{carrier}_{late}
\rightarrow Z_V
\rightarrow P^{surface}_{1:r}
\rightarrow Y_{content}
$$

其中：

```text
O：对象和上下文；
T：任务和模板；
H：逐词元 query 状态；
C：分布式载体成员；
Z_V：全词表竞争；
P：冠词、标点、格式等表面协议；
Y：内容答案。
```

Phase329 只对“query 状态改变全词表竞争者数量”提供类别和栖息地的跨 qwen3/GLM4 结果。其余箭头均未闭合。

### 进度口径

只按直接计数：

```text
Phase329 注册任务：100%；
物理映射模式族覆盖：2/9 = 22.2%；
本轮知识机制全词表路径测试：3/3；
跨模型阻挡者下降：2/3；
严格完整自然链：0/3；
单神经元因果闭合：0；
完整语言编码机制：未完成；
智能理论实验闭合：未完成。
```

本阶段提高了竞争者图谱和读出瓶颈的分辨率，但没有增加模式族覆盖、因果边或单神经元闭合，因此不能上调完整机制完成度。

### 下一阶段边界

Phase329 的注册目标和客户端同步已经完成。继续模型研究需要改变分母，应整体注册为 Phase330，而不是在 Phase329 结果上增加后验补丁：

```text
Phase330：多词元答案起点与表面协议竞争图谱。

1. 用旧主样本冻结自然答案协议，不在新样本上选模板；
2. 把直接目标、冠词加目标、引号加目标等固定成多路径答案起点；
3. 分开记录协议前缀概率、目标条件概率和完整路径概率；
4. 把正确基线能产生目标内容和不能产生目标内容的样本分开；
5. 在基线合格样本上重新验证逐词元 query、冻结载体和自然生成；
6. 只有至少一个机制跨两个模型通过身份、成员中介、内容起点解锁和生成改善，才重新打开单神经元 CUDA 门。
```

当前不能自动进入全量单神经元 CUDA 干预。门槛失败是客观结果，不是工程中断。

### 阶段结论

```text
Phase329 已系统完成附件提出的全词表、逐词元、成员中介和自然生成矩阵；
类别和栖息地出现跨 qwen3/GLM4 的阻挡者下降；
没有机制跨模型证明逐词元优于均值；
没有载体成员中介、top-1 解锁、生成改善或完整链；
表面协议是首词元竞争中的重要混杂因素；
固定图谱和可视化客户端已经同步真实结果；
单神经元 CUDA 门保持关闭；
语言编码机制和智能理论均未闭合。
```

## Phase 330: 九族全局物理图谱、平衡留出集合审计与客户端同步 [2026-07-10 02:56]

### 阶段输入与方案判断

本阶段综合审查了两份输入：

```text
第一份：Phase329 是接口校准与瓶颈分层，建议继续多词元答案起点和表面协议竞争；
第二份：冻结九族总分母，一次完成统一行为、读出、组件路径、留出因果、分析和发布。
```

总体判断：第二份系统方案应作为 Phase330 主合同，第一份窄方案应并入 `readout_competition`（读出竞争）和 `output_protocol`（输出协议）两个模式族，不能继续拆成小阶段。

接受的正确部分：

```text
1. 阶段中只检查完整性，不根据中间效果更新理论；
2. 九族、机制、题项、模板和模型顺序一次冻结；
3. qwen3 -> GLM4 -> DS7B 严格顺序加载并释放显存；
4. 同时记录行为、全词表读出、生成、全层组件路径和留出因果；
5. 发现/校准/留出按独立题项划分，模板 C 不参与选择；
6. 单神经元门未满足前，组件块不得标记为因果神经元。
```

执行前校准：`24 independent items x 3 templates`（24 个独立题项乘 3 模板）表示每机制 24 个题项，每个题项生成 3 个模板，而不是总共 24 条提示。因此固定总量为：

```text
9 families x 8 mechanisms x 24 items x 3 templates = 5184 prompts；
5184 prompts x 3 models = 15552 prompt-model cases。
```

### 固定样本合同

九族和每族八机制：

```text
content_knowledge（内容知识）：category、attribute、function、part、material、habitat、comparison_relation、negated_attribute；
output_protocol（输出协议）：answer_only、single_sentence、single_item_list、json、quote_closure、newline_closure、format_template、no_explanation；
reasoning_constraint（推理约束）：direct_entailment、direct_contradiction、two_hop_entailment、two_hop_blocked、transitive_order、reversed_order_control、conjunction_rule、missing_condition_control；
syntax_structure（语法结构）：subject_role、object_role、singular_agreement、plural_agreement、past_tense、pronoun_number、adjective_attachment、relative_clause_role；
language_action（语言动作）：answer、classify、extract、transform、translate、rewrite、summarize、refuse_or_comply；
cross_lingual（跨语言）：semantic_equivalence、translation、negation、question、role_binding、number_agreement、protocol_preservation、mixed_language_routing；
readout_competition（读出竞争）：target_vs_wrong、target_vs_continue、target_vs_echo、target_vs_protocol、target_vs_punctuation、answer_alias、multi_token_answer、full_vocabulary_blockers；
state_drift（状态漂移）：entity_drift、attribute_drift、role_drift、language_drift、format_drift、reasoning_drift、repetition_drift、long_context_drift；
closure（闭合）：semantic_completion、protocol_completion、stop_wins、continue_suppression、multi_token_completion、alias_completion、generation_stability、client_visible_closure。
```

划分合同：

```text
每机制 item 0-11：discovery（发现），12 个；
每机制 item 12-17：calibration（校准），6 个；
每机制 item 18-23：heldout（留出），6 个；
template A/B：选择样式；
template C：留出样式，不参与层位和成员选择。
```

样本表校验：

```text
语言族：9；
机制：72；
机制内独立题项：1728；
提示：5184；
提示模型实例：15552；
重复 case_id：0；
缺失机制：0；
来源出现在查询之后：0；
空目标或空干扰项：0；
目标文本出现在提示中：1347。
```

1347 条目标可见样本主要来自抽取、改写、格式保持和显式事实任务，不能作为错误全部删除；分析和图谱保留 `target_absent_from_prompt` 字段。但“1728 个独立题项”只在每个机制内部成立，包装型模式族之间复用了 `BASE_QA`（基础问答）语义骨架，不能解释成 1728 个完全独立语义对象。

### 测量原理与公式边界

对目标第一词元和注册干扰词元构造统一读数坐标：

$$
\hat d =
\frac{
W_U(t)-\frac{1}{J}\sum_{j=1}^{J}W_U(d_j)
}{
\left\|W_U(t)-\frac{1}{J}\sum_{j=1}^{J}W_U(d_j)\right\|
}
$$

其中 $W_U$ 是输出嵌入，$t$ 是目标第一词元，$d_j$ 是注册干扰词元。该方向仅是跨模型统一测量坐标，不是语言运行机制公式。

全层、组件和位置角色事件：

$$
s_{l,c,r}=\left\langle
\operatorname{Pool}_{r}(o_{l,c}),\hat d
\right\rangle,
\quad
c\in\{Attention,MLP,Residual\},
\quad
r\in\{source,query,last\}
$$

路径冻结只使用发现集：

$$
S_{select}
=
Support_{discovery}
\times Amplitude_{discovery}
\times Persistence_{discovery}
$$

校准集只检查支持率，不改变已选层和角色。留出层位预测记录：

$$
e_{depth}
=
\frac{|l^{pred}_{peak}-l^{heldout}_{peak}|}{L-1}
$$

集合因果净差：

$$
\Delta_{joint-random}
=M_{joint\_zero}-M_{matched\_random}
$$

$$
\Delta_{joint-wronglayer}
=M_{joint\_zero}-M_{wrong\_layer}
$$

两者都小于零才记为该模型中的集合级读出特异结果。自然身份差为：

$$
\Delta_{identity}
=M_{matched\_natural}-M_{wrong\_donor}
$$

### 三模型 CUDA 执行量

使用 RTX 4090，严格依次执行 qwen3、GLM4、DS7B；每个模型族分区独立落盘，模型切换前释放显存。

最终有效数据：

```text
行为行：15552；
读出行：15552；
生成行：15552；
全词表 top-50（前 50）行：777600；
全层组件事件：4852224；
路径签名：139968；
注意力头/MLP 组候选观测：487296；
冻结载体成员：864；
平衡注册留出案例：432；
因果条件：10；
正式因果行：4320；
留出峰层预测：3888。
```

组件事件按模型精确对应层数：

```text
Qwen3：36 层，每族 186624；
GLM4：40 层，每族 207360；
DS7B：28 层，每族 145152；
九族三模型合计：4852224。
```

### 两次工具和分母校准

第一次校准：生成缓存接口。

```text
观测执行最初使用 use_cache=true；
因果钩子执行使用 use_cache=false；
GLM4 在缓存观测中大量输出连续感叹号；
三模型全部 15552 条生成统一改为无缓存并完整重跑；
读出和组件事件不受该校准影响。
```

统一无缓存后 GLM4 的原始补全模板 A/C 生成命中仍接近零，说明剩余主要问题是 `completion interface`（补全接口）和 `chat-trained interface`（对话训练接口）不匹配，而不是缓存可以完全解释。GLM4 行为率不能用于模型能力排名。

第二次校准：初始因果目标集中。

```text
初始留出索引 18/19 在六个包装族反复落到 7/moon 目标；
旧 4320 行保留为失效注册试跑，不进入最终汇总；
正式注册改为留出区间分离端点 18/23；
错误供体固定为 20/22；
三模型全部 4320 行重新执行；
最终 `causal_rows.parquet` 只汇总 balanced（平衡）目录。
```

### 客观行为和读出结果

统一运行时后的模型总表：

```text
Qwen3：候选集合目标胜出 0.9101；全词表前 50 为 0.8661；生成目标命中 0.7612；综合行为成功 0.7292；
GLM4：候选集合目标胜出 0.9010；全词表前 50 为 0.7936；生成目标命中 0.1757；综合行为成功 0.1734；
DS7B：候选集合目标胜出 0.7838；全词表前 50 为 0.5992；生成目标命中 0.5019；综合行为成功 0.4819。
```

GLM4 的高读出、低自然生成进一步证明：

```text
候选集合正确 != 全词表正确；
全词表目标可见 != 表面协议正确；
第一词元读出 != 完整自然生成；
模型推理接口本身也是实验变量。
```

### 物理路径结果

三模型中，注意力、MLP 和残差的目标方向峰值总体位于相对后段。残差具有更高后段持续性和更少符号翻转；注意力和 MLP 输出更振荡。

工作图景收紧为：

```text
Attention（注意力）：条件化路由和搬运事件；
MLP（多层感知机）：局部写入、转换和竞争事件；
Residual（残差）：跨层累积主干；
Unembedding（输出嵌入）：全词表读出接口；
Rollout/Closure（生成/闭合）：把读出变成客户端可见行为。
```

这不证明残差是完整知识存储位置，也不证明单一线性方向可模拟真实运行。

留出峰层预测：

```text
预测行：3888；
精确层命中率：0.6119；
10% 相对深度容差命中率：0.8315；
平均归一化层误差：0.1156。
```

预测是在已冻结位置角色条件下预测峰层，尚未独立预测 `source/query/last`（来源/查询/末位）角色本身。

### 平衡留出集合因果结果

跨 qwen3、GLM4、DS7B 同时满足联合集合比随机同规模和错层对照都更强的机制：

```text
5/72：
content_knowledge/negated_attribute；
language_action/summarize；
language_action/transform；
reasoning_constraint/missing_condition_control；
syntax_structure/singular_agreement。
```

正确自然供体优于错误供体，且三个模型同向：

```text
14/72：
content_knowledge/material；
content_knowledge/negated_attribute；
content_knowledge/part；
cross_lingual/negation；
cross_lingual/number_agreement；
cross_lingual/question；
cross_lingual/semantic_equivalence；
language_action/classify；
language_action/summarize；
language_action/translate；
reasoning_constraint/direct_contradiction；
reasoning_constraint/two_hop_entailment；
state_drift/repetition_drift；
syntax_structure/past_tense。
```

三个模型同时出现可见行为必要性的机制：

```text
0/72。
```

因此 5 个正机制只能标记为：

```text
L4 registered heldout component-set readout candidate
（四级：注册留出组件集合读出候选）
```

不能标记为：

```text
完整行为因果；
单成员因果；
单神经元因果；
完整自然闭合；
语言编码机制公式闭合。
```

### 图谱数据与客户端同步

固定物理图谱升级到 Phase330：

```text
语言族：9；
模型族分区：27；
历史节点保留：1121；
Phase330 新组件成员：864；
全部显示节点：1985；
全部边：2214；
组件集合成员总数：1152；
历史单神经元候选：833；
单神经元因果：0。
```

新增发布文件：

```text
phase330_paths.jsonl；
phase330_carrier_sets.jsonl；
phase330_claim_registry.jsonl；
27 个 partitions/{family}/{model}.json 分区。
```

客户端保持 Phase324 之后恢复的原 DNN 几何，不改变模型形状。新功能：

```text
九族均可选择；
Qwen3、GLM4、DS7B 均可切换；
新增“注册集合”和“跨模型”筛选；
跨模型集合读出节点使用独立颜色；
详情显示 Phase330 两类对照差、自然供体差和证据边界；
组件组不显示为单神经元。
```

验证：

```text
Phase330 单元测试：12/12 通过；
前端构建：3668 modules transformed（3668 个模块转换）并成功；
桌面 1440x900：画布非空、9 个族、无控制台错误、无文字溢出；
移动 390x844：侧栏隐藏、原模型几何可见、控制区在视口内、无文字溢出；
本地地址：http://127.0.0.1:5173/。
```

### 严格硬伤

```text
1. 当前三个模型都是小模型，相对大型语言模型可能有 30%-50% 的结构偏差；
2. 九族是工作分类，未通过开放集 unknown/mixed 分类完备性检验；
3. 跨机制复用了语义骨架，1728 不能当成完全独立语义对象；
4. 第一词元线性方向只是读数坐标，不能模拟多词元、非线性、条件化真实机制；
5. 注意力头和 MLP 分组是组件块，不是单神经元；
6. 每机制因果筛选只有 2 个留出项，只适合筛选；
7. GLM4 原始补全接口与对话训练接口不匹配；
8. 自然迁移只替换冻结成员切片，不是完整自然状态；
9. 干预后补偿路径没有完整追踪；
10. 跨模型行为必要性为 0/72，完整自然链仍为 0；
11. Phase288 全量单神经元 CUDA 干预仍未运行；
12. 当前结果不支持给线性机制公式继续增加 patch（补丁）并宣称闭合。
```

### 智能理论更新边界

理论名称不变：“语言是动态模式网络”。当前复合机制工作式为：

$$
Y=
\operatorname{Closure}
\circ\operatorname{Rollout}
\circ\operatorname{Competition}
\circ\operatorname{Readout}
\circ\operatorname{ComponentPath}
\circ\operatorname{StateWrite}
\circ\operatorname{Route}
\circ\operatorname{Trigger}(X,T)
$$

Phase330 新增的客观约束：

```text
1. 九族都能在统一读数坐标下形成可重复的后段物理路径；
2. 层峰可在新模板上部分预测，说明路径并非纯样本噪声；
3. 少量机制的冻结集合对第一词元读出有位置和成员特异性；
4. 集合级读出改变通常不转化为跨模型可见行为改变；
5. 自然状态身份、全词表竞争、表面协议、生成稳定和客户端停止必须分层建模；
6. 真实机制很可能包含上下文门控、非线性耦合、冗余成员和补偿旁路。
```

不能推出：

```text
语言只由后层编码；
残差本身就是知识库；
5 个正机制已经闭合；
单一线性方向是统一智能公式；
小模型图谱可以直接等同于人类语言或大型模型编码结构。
```

### 直接进度口径

```text
Phase330 工程合同：100%；
九族行为、读出、全层路径覆盖：9/9；
72 机制细组件候选覆盖：72/72 x 3 模型；
72 机制注册集合干预覆盖：72/72 x 3 模型；
跨三模型集合级读出特异正结果：5/72；
跨三模型自然身份正结果：14/72；
跨三模型行为必要性：0/72；
全量单神经元 CUDA 干预：0；
完整自然语言链：0；
完整语言编码机制：未完成；
智能理论实验闭合：未完成。
```

不能再用单一总体百分比把“覆盖率 100%”和“机制闭合 0”平均掉；后续图谱必须同时显示覆盖向量和证据向量。

### 下一阶段边界

Phase330 的冻结合同、执行、分析、图谱和客户端同步已完整完成。下一步应新注册为 Phase331，而不是继续向 Phase330 后验添加样本：

```text
Phase331：五个跨模型集合读出正机制的扩大留出、成员细化和补偿路径审计。

1. 冻结 5 个正机制，不从其余 67 个机制后验增加候选；
2. 使用每机制剩余 4 个留出题项扩大确认；
3. 同时运行原始补全和正确 chat template（对话模板）双接口；
4. 对集合执行 leave-one-member-out（逐成员移除）和递归二分；
5. 把 MLP 组逐步缩小到神经元候选，同时保留随机同规模、错层和错供体控制；
6. 逐层记录干预后的注意力、MLP、残差补偿流；
7. 只有至少一个机制跨三个模型通过读出、完整生成、低副作用和单元级扩大确认，才打开全量单神经元 CUDA 门。
```

Phase331 会改变机制分母、留出用法和干预粒度，属于新的注册阶段，不应自动混入 Phase330 当前结论。本阶段目标已经闭合为“九族全局物理图谱覆盖完成，但编码机制未闭合”。

## Phase 331: 五条集合读出候选的扩大留出、双接口、成员细化与补偿审计 [2026-07-10 05:54]

### 本阶段问题与对前置分析的判断

Phase330 完成了九族、72 机制、三模型的统一观察分母，并筛出 5 个跨模型集合级首词元读出候选：

```text
content_knowledge / negated_attribute（内容知识 / 否定属性）；
language_action / summarize（语言动作 / 摘要）；
language_action / transform（语言动作 / 变换）；
reasoning_constraint / missing_condition_control（推理约束 / 缺失条件控制）；
syntax_structure / singular_agreement（语法结构 / 单数一致）。
```

附件对 Phase330 的总体分析基本正确：

```text
九族观察覆盖完成，不等于九族机制闭合；
五条正结果只是冻结组件集合对首词元读出的候选；
两个留出对象不足以支持稳定机制结论；
必须补充正确对话接口、成员细化、自然状态身份和补偿路径；
在完整生成必要性出现前，不能打开全量单神经元 CUDA 门。
```

需要补充两点校正：

```text
1. 附件原建议的 360 个接口案例只计算五条正机制。本阶段给五个匹配负对照完全相同的
   对象、模板、接口和模型权重，因此总注册分母是 720 个接口案例；
2. “行为改变”必须拆成行为损失和行为增益。只有“基线成功 -> 联合干预失败”可进入
   行为必要性门，不能把“基线失败 -> 干预后成功”也算成必要性证据。
```

### 冻结注册分母

Phase331 在任何新模型执行前冻结以下正负机制对：

```text
negated_attribute（否定属性） <-> attribute（属性）；
summarize（摘要） <-> rewrite（改写）；
transform（变换） <-> extract（抽取）；
missing_condition_control（缺失条件控制） <-> two_hop_blocked（两跳阻断）；
singular_agreement（单数一致） <-> plural_agreement（复数一致）。
```

负对照按同模式族、相近任务结构和相同目标类型冻结，不使用 Phase331 结果挑选。

正式分母：

```text
正机制：5；
匹配负对照：5；
未使用留出对象：item 19、20、21、22，共 4 个；
模板：template_a、template_b、template_c，共 3 个；
接口：raw_completion（原始续写）、chat_template（模型原生对话模板），共 2 个；
模型：Qwen3-4B、GLM4-9B、DS7B，共 3 个；
正机制接口案例：5 x 4 x 3 x 2 x 3 = 360；
负对照接口案例：5 x 4 x 3 x 2 x 3 = 360；
总接口案例：720。
```

目标状态按数据显式拆分：

```text
present（目标已在提示中）：252；
absent（目标不在提示中）：252；
transformed（目标需变换得到）：216。
```

### 干预条件与运行规模

每个正机制沿用 Phase330 冻结的四成员集合：

```text
2 个 attention head input（注意力头输入切片）；
2 个 MLP product group（多层感知机乘积组）。
```

正机制预注册 21 个条件：

```text
baseline（基线）；
joint_set_zero（四成员联合清零）；
attention_set_zero（两个注意力成员清零）；
mlp_set_zero（两个 MLP 成员清零）；
single_member_0..3_zero（四个逐成员清零）；
set_without_member_0..3_zero（四个逐成员留一/去一审计）；
matched_random_joint_zero（同规模随机索引控制）；
wrong_layer_joint_zero（错层控制）；
paired_control_joint_zero（配对负机制集合控制）；
correct_donor_transplant（正确供体迁移）；
wrong_donor_transplant（错误目标供体迁移）；
same_target_donor_transplant（同目标、脱离原任务机制供体迁移）；
matched_random_donor_transplant（同桶随机供体迁移）；
wrong_layer_donor_transplant（错层供体迁移）；
correct_donor_restoration（清零后正确供体恢复）。
```

每个负对照预注册 5 个条件：

```text
baseline；
joint_set_zero；
matched_random_joint_zero；
wrong_layer_joint_zero；
paired_positive_joint_zero（正机制集合施加到负对照案例）。
```

正式完成：

```text
接口案例：720/720；
条件结果：9360/9360；
自然生成条件结果：5760；
执行全层补偿追踪的条件：3120；
全层路径事件：1,288,560；
候选层全部头/MLP 组响应事件：195,520。
```

三模型按以下顺序独立加载、执行和释放：

```text
Qwen3 -> GLM4 -> DS7B。
```

单模型质量结果：

```text
Qwen3：240 接口案例，3120 条件，446160 路径事件，66560 组件响应；
GLM4：240 接口案例，3120 条件，496080 路径事件，66560 组件响应；
DS7B：240 接口案例，3120 条件，346320 路径事件，62400 组件响应；
全部有效，无显存溢出，无缺失条件，无后验成员更新。
```

### 测量公式

第一词元目标边际仍作为局部读数，但不再单独承担机制结论：

$$
m(x,I)=z_{t}(x,I)-\max_{d\in D}z_d(x,I)
$$

其中：

```text
I 是 raw_completion 或 chat_template 接口；
t 是目标第一词元；
D 是注册错误候选集合。
```

集合干预效应：

$$
\Delta m_c=m\!\left(x,I;\operatorname{do}(S=c)\right)-m(x,I;\text{baseline})
$$

集合读出特异门要求：

$$
\overline{\Delta m}_{joint}<-0.05
$$

并且：

$$
\overline{\Delta m}_{joint}
<\min\left(
\overline{\Delta m}_{random},
\overline{\Delta m}_{wrong\ layer},
\overline{\Delta m}_{paired\ mechanism}
\right)
$$

四个留出对象中至少三个对象的三模板均值方向必须为负：

$$
\frac{1}{4}\sum_{i=19}^{22}
\mathbf{1}\!\left[
\operatorname{Mean}_{template}(\Delta m_i)<0
\right]
\ge 0.75
$$

完整答案串采用教师强制条件概率：

$$
\log P(y\mid x,I,c)
=\sum_{k=1}^{|y|}
\log P(y_k\mid x,I,c,y_{<k})
$$

行为必要性只计算损失，不把增益混入：

$$
N_{beh}(c)=
\frac{1}{n}\sum_i
\mathbf{1}
\left[
B_i^{base}=1\land B_i^c=0
\right]
$$

预注册行为门为：

$$
N_{beh}(joint)\ge 0.10
$$

累计残差状态和层间增量分开记录：

$$
r_l=\operatorname{Proj}_{v_t}(h_l)
$$

$$
\Delta r_l=
\operatorname{Proj}_{v_t}(h_{l+1}-h_l)
$$

不能再把 $r_l$ 直接称为第 $l$ 层写入量。

未选择组件补偿比：

$$
C_{unit}=
\frac{
\sum_{u\notin S}|q_u^{joint}|
}{
\max\left(\sum_{u\notin S}|q_u^{base}|,\epsilon\right)
}
$$

晚层残差恢复比例：

$$
C_{late}=1-
\min\left(
1,
\frac{|r_{final}^{base}-r_{final}^{joint}|}
{\max_l|r_l^{base}-r_l^{joint}|}
\right)
$$

完整门仍冻结为：

$$
G=
G_{readout}
\land G_{heldout}
\land G_{interface}
\land G_{model}
\land G_{member}
\land G_{compensation}
\land G_{generation}
\land G_{sideeffect}
$$

### 直接客观结果

30 个正机制“模型 x 接口”单元格中：

```text
局部集合读出通过：13/30；
其中原始续写接口：9/15；
其中原生对话接口：4/15；
局部成员定位通过：15/30；
局部行为损失达到 10% 门槛：2/30；
自然供体身份特异：0/30。
```

30 个负对照“模型 x 接口”单元格中：

```text
局部集合读出通过：2/30。
```

两个负对照局部阳性是：

```text
Qwen3 / raw_completion / attribute（属性）；
GLM4 / chat_template / two_hop_blocked（两跳阻断）。
```

这说明 Phase330 的五条正候选并没有形成完全干净的正负边界。

五条候选的跨模型、跨接口结果：

| 机制 | 平均联合边际变化 | 平均完整串变化 | 平均行为损失率 | 平均补偿比 | 平均晚层恢复 | 完整门 |
|---|---:|---:|---:|---:|---:|---:|
| negated_attribute（否定属性） | +0.1139730 | +0.0059150 | 0 | 1.0016618 | 0.0953840 | 失败 |
| summarize（摘要） | -0.2515793 | +0.3335264 | 0.0277778 | 0.9610723 | 0.1876142 | 失败 |
| transform（变换） | -0.6706062 | +0.0499747 | 0 | 1.0149794 | 0.1747307 | 失败 |
| missing_condition_control（缺失条件控制） | -1.2186415 | -0.3043506 | 0.0277778 | 0.9781841 | 0.0398012 | 失败 |
| singular_agreement（单数一致） | -0.3208991 | -0.5663145 | 0 | 1.0431122 | 0.1695490 | 失败 |

最重要的分解结果：

```text
1. summarize 在三个模型的 raw_completion 接口均通过局部集合读出，但三个 chat_template
   单元格均失败，因此是“原始续写接口候选”，不是跨接口机制；
2. missing_condition_control 在三个模型的 raw_completion 均通过，Qwen3 与 DS7B 的
   chat_template 也通过，但 GLM4 对话接口没有越过 -0.05 门槛；同时 GLM4 对应负对照
   two_hop_blocked 在对话接口局部通过，所以仍不能形成干净正负边界；
3. transform 在 Qwen3 两接口和 GLM4 对话接口有局部读出效应，但 DS7B 没有复现；
4. singular_agreement 只在 Qwen3、GLM4 的原始接口通过，DS7B 未复现；
5. negated_attribute 没有保留 Phase330 的跨模型集合读出阳性。
```

完整串与第一词元也出现方向分离：

```text
summarize 的平均第一词元边际下降 -0.2515793，
但完整目标串对数概率平均上升 +0.3335264；

transform 的平均第一词元边际下降 -0.6706062，
但完整目标串对数概率平均上升 +0.0499747。
```

因此“第一词元边际下降”不能自动解释成“完整答案机制被破坏”。

自然生成中只有两个局部单元格达到至少 2/12 的行为损失：

```text
Qwen3 / raw_completion / summarize：2/12；
GLM4 / raw_completion / missing_condition_control：2/12。
```

没有任何一个机制在两个接口、三个模型上同时产生注册阈值以上的行为损失。

最终严格计数：

```text
跨三个模型且跨两个接口的扩大集合读出：0/5；
跨模型、跨接口稳定小成员集：0/5；
补偿测量完整且低于预注册门：1/5；
跨模型、跨接口完整生成必要性：0/5；
完整八门通过：0/5；
行为机制闭合：0/5；
单神经元因果：0；
全 72 机制行为闭合：0/72。
```

### 结论与证据降级

Phase331 是一个强负结果，同时包含两个接口受限的局部正迹象。

Phase330 的五条候选不能继续统一标记为：

```text
L4 expanded set-readout mechanism（四级扩大集合读出机制）。
```

当前严格标记应降为：

```text
L3 frozen component-set candidate not expanded cross-interface
（三级冻结组件集合候选，未通过扩大跨接口验证）。
```

允许保留的局部现象只有：

```text
summarize 与 missing_condition_control 在原始续写接口上保留三模型集合读出迹象；
部分模型/接口出现 1-2 个局部成员候选；
少数局部单元格出现自然生成行为损失；
这些现象都没有形成同一条跨模型、跨接口、自然身份、成员定位和完整生成链。
```

不能宣称：

```text
五条机制已复现；
四成员集合是语言模式的稳定物理载体；
某个注意力头或 MLP 组是机制核心；
自然状态迁移已证明状态身份；
第一词元线性方向已解释完整语言生成；
可以启动 Phase288 式全量单神经元 CUDA 闭合扫描；
语言编码机制或智能理论已闭合。
```

### 关键硬伤

```text
1. 当前仍是 4B-9B 小模型，内部机制可能比大型模型粗糙，不能把跨模型失败直接外推为
   所有语言模型都没有共享机制；
2. 每机制只有 4 个真正新留出对象，三个模板共享语义核心，统计独立对象仍是 4 而非 12；
3. DS7B 对话模板自动进入思考前缀，12 个生成词元可能没有到达最终答案；其对话接口
   行为失败同时包含协议与长度因素；
4. 对话端点的第一词元在思考模型上不一定是答案起点，端点读出和答案读出必须继续区分；
5. 正机制与负对照只是同族、同目标类型近邻，并非严格同难度、同长度、同自然频率；
6. same_target_donor（同目标供体）使用直接目标身份控制，不是完全自然的同任务供体；
7. 清零注意力头输入和 MLP 乘积组仍可能产生分布外状态；
8. 成员响应使用输出嵌入方向下的近似贡献，仍是线性读数，不是组件完整非线性功能；
9. 全层追踪记录所有层的注意力、MLP、残差，但神经元组细分只在冻结候选层展开，
   尚未完成全层全部神经元的因果扫描；
10. 完整串概率采用教师强制，不能替代自由生成轨迹；
11. 预注册阈值是基础工程门槛，不是由独立功效分析得到的自然常数；
12. 13/30 局部读出阳性与 2/30 负对照阳性提示模型、接口、对象交互较强，
    不能继续用一个固定四成员集合描述真实运行机制；
13. 自然身份特异为 0/30，说明冻结切片移植没有抓住稳定自然状态对象；
14. 当前没有全字符串梯度、别名子空间因果干预和答案起点对齐的非线性状态追踪。
```

### 语言模式图谱与客户端同步

图谱不改变 Phase324 恢复后的 DNN 三维几何，只更新证据叠层：

```text
Phase331 更新组件集合成员：60；
新增“扩展审计”筛选；
新增原始/对话接口局部读出状态；
新增第一词元边际和完整串概率变化；
新增未选择组件补偿比和晚层残差恢复比例；
新增成员定位、完整生成和完整门状态；
所有 60 个节点 single_unit_causal（单单元因果）仍为 false（否）；
完整门通过：0；
行为机制闭合：0；
单神经元门：0。
```

可视化状态文字：

```text
Phase331 扩展审计，非单元因果。
```

验证：

```text
Phase331 单元测试：6/6 通过；
前端构建：3668 个模块转换并成功；
桌面 1440x900：九族、扩展审计筛选、新指标和非空 WebGL 画布通过；
移动 390x844：原模型几何、扩展审计控制、非空画布和文字边界通过；
桌面截图 RGB 像素标准差：(21.72, 27.93, 32.18)；
移动截图 RGB 像素标准差：(22.39, 26.45, 28.16)；
本地地址：http://127.0.0.1:5173/。
```

主要产物：

```text
tests/gpt5/phase331_refined_mechanism_case_bank.py；
tests/gpt5/phase331_refined_mechanism_audit.py；
tests/gpt5/phase331_refined_mechanism_analysis.py；
tests/gpt5/phase331_publish_refined_atlas.py；
tests/gpt5/run_phase331_refined_mechanism_audit.sh；
tests/gpt5/result/phase331_refined_mechanism_audit/refined_mechanism_audit/phase331_report.md；
tests/gpt5/result/phase331_refined_mechanism_audit/refined_mechanism_audit/phase331_global_summary.json；
tests/gpt5/result/phase331_refined_mechanism_audit/refined_mechanism_audit/phase331_cross_model_summary.jsonl；
tests/gpt5/result/phase331_refined_mechanism_audit/refined_mechanism_audit/phase331_compensation_path_rows.parquet；
tests/gpt5/result/phase331_refined_mechanism_audit/refined_mechanism_audit/phase331_component_response_rows.parquet。
```

### 对智能理论的约束

理论名称仍保持“语言是动态模式网络”，不因本轮局部结果修改理论名词。

Phase330 的复合工作式仍可作为任务分层描述：

$$
Y=
\operatorname{Closure}
\circ\operatorname{Rollout}
\circ\operatorname{Competition}
\circ\operatorname{Readout}
\circ\operatorname{ComponentPath}
\circ\operatorname{StateWrite}
\circ\operatorname{Route}
\circ\operatorname{Trigger}(X,T)
$$

但 Phase331 证明其中物理运行不能再省略接口状态、条件门控和补偿旁路。更贴近当前拼图的状态更新约束是：

$$
h_{l+1}=
h_l+
A_l(h_l,X,T,I)+
M_l(h_l,X,T,I)
$$

以及：

$$
P(Y\mid X,T,I)=
\operatorname{Decode}
\left(
\{h_l\}_{l=0}^{L},
\mathcal{C}_{comp},
I
\right)
$$

其中：

```text
I 是模型接口和协议状态；
A_l 是注意力增量；
M_l 是 MLP 增量；
mathcal{C}_{comp} 是未选择成员与后续层形成的补偿结构。
```

Phase331 的核心理论约束不是新增统一公式，而是否定以下简化：

$$
\text{固定四成员线性集合}
\Rightarrow
\text{跨接口稳定语言机制}
$$

当前数据明确不支持这个蕴含。

更符合现象的工作假设是：

```text
语言模式不是固定少量单元的静态向量；
它更可能是接口状态、位置角色、上下文目标、候选竞争、分布式组件和补偿路径共同限定的
条件化运行子图；
同一语义任务在原始续写与对话协议中可以进入不同的物理运行轨迹；
小模型中的路径复用可能比大型模型更粗糙，也可能更依赖协议包装。
```

这仍然只是工作假设，不是理论闭合。

### 直接进度口径

```text
九族注册与观察覆盖：9/9；
72 机制三模型行为、读出和全层普查：72/72；
Phase331 五条候选扩大执行：5/5 已测试；
跨模型且跨接口扩大集合读出：0/5；
跨模型且跨接口稳定成员定位：0/5；
跨模型且跨接口行为必要性：0/5；
语言机制行为闭合：0/72；
单神经元因果闭合：0/72；
完整语言编码机制：未完成；
智能理论实验闭合：未完成。
```

仍然禁止把“九族工程覆盖 100%”与“机制闭合 0/72”平均成单一总体百分比。

### 下一阶段：Phase332 大任务边界

Phase331 已完整闭合其注册目标，下一步不应继续给原四成员线性集合增加小补丁，也不应启动全量单神经元 CUDA 扫描。

最可行的下一阶段是：

```text
Phase332：接口状态分叉与答案起点对齐的条件化物理路径图谱。
```

它应作为新注册阶段，目标不是再次证明五条旧候选，而是解释：

```text
为什么 summarize 与 missing_condition_control 在 raw_completion 三模型复现，
却没有形成跨 chat_template 的同一条机制链。
```

建议一次性完成以下四个工作包：

```text
工作包 A：接口与答案起点
1. 冻结 summarize、missing_condition_control 及其两个负对照；
2. 分开 raw、原生 chat、关闭思考的 chat、答案前缀对齐 chat；
3. DS7B 使用足以到达 </think> 后答案的长生成，但把思考区与可见答案区分开评分；
4. 记录助手起点、思考起点、答案起点三个位置角色。

工作包 B：条件化路径而非固定集合
1. 不重新从结果中挑四成员集合；
2. 对同一语义对象比较不同接口下自然激活的头/MLP 组交集与差集；
3. 只保留跨对象稳定、跨模板稳定的接口共享骨架和接口特异分支；
4. 用路径交换检验“共享骨架 + 接口分支”是否比固定集合解释力更强。

工作包 C：非线性读出
1. 保留第一词元边际作为参考；
2. 加入完整字符串概率、别名集合、答案起点状态和自由生成轨迹；
3. 比较累计状态、层间增量和后层恢复，不再用单一线性方向代表真实运行；
4. 把行为损失与行为增益严格分开。

工作包 D：门槛
1. 只有接口共享骨架在三模型、四对象、三模板上稳定，才升级为 L4；
2. 只有路径交换改变完整答案且低副作用，才升级为 L5 分布式机制候选；
3. 只有稳定小成员集通过自然身份、必要性、充分性和完整生成，才打开单神经元 CUDA 门；
4. 任何一个前置门失败，都继续记录物理分叉图谱，不进入闭合扫描。
```

Phase332 会改变接口分母、位置角色、生成长度和路径对象，属于新阶段。Phase331 当前阶段已经完整完成，不应在其结果中后验追加答案起点样本或重新选择成员。

## Phase 332: 接口状态分叉、答案起点与保留集路径交换图谱 [2026-07-10 07:29]

### 一、对输入分析的审计结论

输入对 Phase331（阶段331）的总判断基本正确：它确实是高价值强负结果，而不是五条机制扩大复现成功。以下判断得到 Phase331 原始结果支持：

```text
1. 固定四成员集合不能解释跨模型、跨接口、跨对象和跨模板的稳定语言机制；
2. raw_completion（原始续写）和 chat（对话）不能被视为同一物理运行接口；
3. 第一词元读出、完整字符串概率和自由生成行为可以分离；
4. 自然状态身份 0/30、完整行为闭合 0/5，禁止启动全量单神经元扫描；
5. summarize（摘要）和 missing_condition_control（缺失条件控制）只能保留为局部现象，
   不能称为跨接口机制。
```

输入提出 Phase332（阶段332）研究接口共享骨架、接口分支、答案起点和路径交换，也是 Phase331 之后合理且必要的下一步。本阶段已经按这一方向完整执行。

但输入中有三处需要收紧：

```text
1. “总体完成 30%-35%”没有固定分母，不能作为客观进度；本记录改用进度向量；
2. 层级统计模型、多重检验等方法可以作为将来辅助工具，但目前不能代替物理路径、
   保留集复现和因果对照，因此没有成为本阶段核心；
3. GLM4 的 native_chat（原生对话）与 chat_no_think（无思考对话）提示完全相同，
   只能算一个独立接口证据，不能把重复提示计成两次跨接口复现。
```

### 二、阶段问题与固定分母

本阶段只追踪 Phase331 中仍有局部迹象的两个正机制及其同族对照：

```text
language_action/summarize（语言动作/摘要）
  对照：language_action/rewrite（语言动作/改写）

reasoning_constraint/missing_condition_control（推理约束/缺失条件控制）
  对照：reasoning_constraint/two_hop_blocked（推理约束/两跳阻断）
```

每个机制使用 8 个全新对象、3 个模板、4 个工程接口、3 个模型：

```text
对象 0-3：只允许发现自然路径成员；
对象 4-7：只允许验证成员稳定性和路径交换；
接口：raw_completion、native_chat、chat_no_think、answer_aligned_chat；
模型顺序：qwen3 -> GLM4 -> DS7B；
每次只加载一个 CUDA 模型；
完整生成长度：64 个词元。
```

答案位置角色被拆成：

```text
answer_start（答案起点）；
assistant_start（助手起点）；
think_start（思考起点）；
visible_answer_start（可见答案起点）。
```

固定分母为：

```text
注册接口案例：1152；
真正不同的提示案例：1056；
GLM4 重复接口案例：96；
发现集接口案例：576；
保留集接口案例：576；
注册交换案例：288；
每个交换案例 6 个条件；
交换条件结果：1728；
交换完整生成：1728。
```

交换六条件为：

```text
baseline（基线）；
shared_skeleton_correct（正确共享骨架）；
interface_branch_correct（正确接口分支）；
shared_plus_branch_correct（正确共享骨架加接口分支）；
shared_plus_branch_wrong_item（错误对象联合交换）；
matched_random_units_correct（匹配随机组件对照）。
```

### 三、算法原理与基础公式

对模型、机制和接口分别记录自然全层状态：

$$
h_{l+1}^{m,k,I}
=
h_l^{m,k,I}
+
A_l\left(h_l^{m,k,I},X,T,I\right)
+
M_l\left(h_l^{m,k,I},X,T,I\right)
$$

其中同时记录累计残差、层间增量、注意力头输入和多层感知机乘积组，不把它们合并成一个线性方向。

发现集组件必须满足跨对象符号一致性：

$$
C(u)=
\frac{1}{|O_d|}
\sum_{o\in O_d}
\mathbf{1}
\left[
\operatorname{sign}\left(c_{u,o}\right)
=
\operatorname{sign}\left(\bar c_u\right)
\right]
\ge 0.75
$$

每个组件类型和位置角色只保留前 10%，并限制为最多 12 个成员。接口共享骨架定义为一个模型内真正不同接口的精确物理成员交集：

$$
S_{m,k}
=
\bigcap_{I\in\mathcal I_m^{unique}}
P_{m,k,I}
$$

接口分支定义为该接口稳定集合扣除其他接口集合：

$$
B_{m,k,I}
=
P_{m,k,I}
\setminus
\bigcup_{J\ne I}P_{m,k,J}
$$

保留集不允许更新成员。共享成员必须在每个真正不同接口上保持至少 0.75 的对象方向一致性；接口分支还必须满足所属接口的绝对贡献大于所有其他接口。

路径交换不是清零，而是把供体条件下冻结组件的自然状态写入受体接口：

$$
\tilde h_{l+1}
=
F_l
\left(
\tilde h_l;
do\left(S_{m,k}=S_{donor}\right),
do\left(B_{m,k,I}=B_{donor,I}\right)
\right)
$$

联合交换的非加性交互参考量为：

$$
\Delta_{int}
=
\Delta_{S+B}
-
\Delta_S
-
\Delta_B
$$

但本阶段不把这个参考量直接解释成真实非线性机制，只用于观察联合交换是否超过两个局部交换的简单相加。

完整门为：

$$
G_{332}
=
G_{shared}
\land
G_{branch}
\land
G_{exchange}
\land
G_{string}
\land
G_{generation}
\land
G_{sideeffect}
\land
G_{crossmodel}
$$

固定阈值为：

```text
保留对象方向一致性 >= 0.75；
完整字符串对数概率改善 >= 0.10；
行为增益率 >= 0.10；
对照行为副作用 <= 0.10；
协议副作用 <= 0.10。
```

### 四、执行规模与数据完整性

三模型顺序执行完成：

```text
自然全层路径事件：475776；
自然细组件事件：7538688；
发现集冻结成员记录：4493；
保留集自然完整生成：576；
交换路径事件：158592；
交换组件响应：244096；
交换完整生成：1728；
三模型自然执行有效：3/3；
三模型交换执行有效：3/3；
选择规则运行中更新：否；
单神经元干预门预先打开：否。
```

模型级自然扫描规模：

| 模型 | 接口案例 | 自然路径 | 自然组件 | 共享候选 | 接口分支候选 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 384 | 164736 | 2654208 | 85 | 263 |
| GLM4 | 384 | 183168 | 2949120 | 42 | 316 |
| DS7B | 384 | 127872 | 1935360 | 108 | 223 |

这些候选数量不能直接横向解释为某个模型“机制更多”，因为模型层数、注意力头和中间层尺寸不同。

### 五、客观结果

#### 1. summarize（摘要）

保留集稳定共享成员：

```text
qwen3：13；
GLM4：0；
DS7B：7。
```

保留集接口分支：

```text
raw_completion：qwen3 12，GLM4 46，DS7B 10；
answer_aligned_chat：qwen3 0，GLM4 6，DS7B 4。
```

因此：

```text
跨模型共享骨架：失败；
跨模型接口分支：失败；
raw 到答案对齐的联合交换平均完整串变化：-2.0416563；
反向交换平均完整串变化：+0.5672488，但不满足对象一致性、对照和跨模型门；
跨模型自由生成增益：0；
低副作用门：失败；
完整门：失败。
```

GLM4 的 24 个摘要交换案例没有共享成员，正反方向各 12 个。这不是“共享交换无效”，而是更早一层的结果：严格精确交集下没有可交换的共享集合。

#### 2. missing_condition_control（缺失条件控制）

保留集稳定共享成员：

```text
qwen3：18；
GLM4：16；
DS7B：43。
```

保留集接口分支：

```text
raw_completion：qwen3 35，GLM4 41，DS7B 16；
answer_aligned_chat：qwen3 8，GLM4 5，DS7B 6。
```

因此：

```text
跨模型共享骨架：通过观察门；
跨模型接口分支：通过观察门；
raw 到答案对齐的联合交换平均完整串变化：-0.2012257；
反向交换平均完整串变化：-0.1546992；
跨模型路径交换有效：失败；
跨模型自由生成增益：0；
低副作用门：通过；
完整门：失败。
```

这是本阶段最重要的新拼图：缺失条件控制存在跨模型、跨对象、跨模板的共享骨架和接口分支物理签名，但移植这些冻结端点状态不能按预测改善完整答案或自由生成。

#### 3. 总门结果

```text
两个正机制：2；
跨模型稳定共享骨架：1/2；
跨模型接口特异分支：1/2；
跨模型路径交换有效：0/2；
跨模型完整字符串改善：0/2；
跨模型自由生成改善：0/2；
完整门：0/2；
行为机制闭合：0；
单神经元因果：0；
单神经元扫描门：0。
```

### 六、非有限值异常与失败关闭

GLM4 出现 4 个自然案例的非有限读出值，其中一个保留集 `rewrite（改写）` 案例输出连续 64 个感叹号，并派生出非有限交换差值。

审计计数：

```text
基线非有限指标：8；
交换及派生非有限指标：38；
不完整模型-机制-方向单元：2。
```

分析器已经修改为：

```text
只对有限数值求平均；
同时登记有限值数量和总行数；
任一条件指标不完整时 metrics_complete（指标完整）为 false（否）；
正机制和负对照都按失败关闭，不能因缺失值通过门槛；
固定 JSON 输出不允许 NaN 或 Infinity（无穷值）。
```

这不会改变两个正机制 0/2 路径交换闭合的结论，但避免错误地把异常对照解释为低副作用。

### 七、结论的严格证据边界

本阶段支持：

```text
1. 接口状态确实对应不同的自然物理成员分布；
2. 缺失条件控制具有可跨三模型复现的共享组件骨架和 raw/答案对齐分支；
3. 同一模式的物理图谱应包含接口、位置角色和答案阶段；
4. 固定四成员集合应被可变条件路径替代；
5. 可观察稳定路径不等于可交换的因果运行状态。
```

本阶段不支持：

```text
1. summarize 已找到跨模型共享路径；
2. missing_condition_control 的共享骨架是行为机制；
3. 注意力头或 MLP 组是单神经元因果单元；
4. 端点状态移植已经复原完整动态过程；
5. 接口分支解释了自然生成；
6. 可以启动全量单神经元 CUDA 扫描；
7. 语言编码机制或智能理论已经闭合。
```

证据等级保持：

```text
summarize：L3 接口条件路径图谱，未因果闭合；
missing_condition_control：L3 接口条件路径图谱，未因果闭合；
所有 286 个可视化成员：非单神经元因果。
```

### 八、主要问题、硬伤和瓶颈

```text
1. 每机制只有 4 个发现对象和 4 个保留对象；三个模板共享同一语义对象，独立对象仍是 4；
2. 任务是短格式英文合成任务，不能代表开放语言、长上下文和多轮交互；
3. 当前只扩展两个正机制和两个对照，不能外推到全部 72 个机制；
4. 共享骨架使用精确物理成员交集，可能对小模型结构噪声过严，也可能漏掉功能等价但编号不同的成员；
5. 成员发现仍依赖输出嵌入方向下的近似贡献，尚未测量组件完整非线性功能；
6. 注意力头和 MLP 乘积组仍是组件级，不是单神经元级；
7. 冻结端点状态交换可能破坏时间顺序、归一化关系和上下文协同，属于部分分布外状态；
8. 共享骨架和接口分支可能是结果签名，而不是产生结果的控制变量；
9. 路径交换只连接 raw 与答案对齐接口，尚未覆盖全部接口之间的动态转换；
10. 当前补偿分析记录全层响应，但没有形成可验证的有向补偿图；
11. 自由生成成败仍是较粗的答案段评分，没有逐词元路径身份和错误首次出现位置；
12. GLM4 出现非有限读出和退化生成，说明小模型数值稳定性会污染部分图谱；
13. qwen3、GLM4、DS7B 都是 4B-9B 小模型，真实大模型可能有 30%-50% 的结构差异；
14. 路径稳定和因果闭合之间仍有核心缺口，继续给线性集合加成员会进入边际收益递减。
```

### 九、对“语言是动态模式网络”理论的更新约束

理论名称保持不变。当前结果不需要重新命名理论，只需要收紧运行机制描述。

更符合现象的工作式为：

$$
Y
=
\operatorname{Decode}
\left(
\operatorname{Rollout}
\left(
\mathcal{G}_{k}
\left(X,T,I,p,t\right)
\right)
\right)
$$

其中：

```text
k 是语言模式；
I 是接口和协议；
p 是来源、查询、思考起点或答案起点等位置角色；
t 是生成时间；
G_k 是随上下文和时间变化的条件化运行子图。
```

Phase332 的新约束是：

$$
\operatorname{StableSignature}
\left(\mathcal G_k\right)
\not\Rightarrow
\operatorname{PortableCausalState}
\left(\mathcal G_k\right)
$$

也就是：稳定共享骨架和接口分支可以是可靠的物理签名，但它们仍可能只是动态过程的投影。真正机制可能依赖：

```text
跨层顺序；
组件之间的同时关系；
归一化前后的相对尺度；
逐词元时间演化；
全词表竞争状态；
后层补偿与旁路；
接口触发的动态门控。
```

因此，当前不能继续假设一个静态线性集合能够完整模拟真实语言运行机制。

### 十、语言模式图谱与客户端同步

本阶段没有改变 Phase324 后恢复的三维深度神经网络形状，只增加证据叠层和关键路径连线。

发布规则：

```text
原有节点：1985，全部保留；
新增保留集稳定路径成员：286；
更新分区：语言动作和推理约束在三个模型中的 6 个分区；
只显示保留集仍稳定的共享骨架或接口分支；
不显示全部 4493 条发现成员；
共享骨架：青色；
接口分支：橙色；
同一机制、路径角色、接口和位置角色按层连接；
新增“接口脉络”筛选；
所有新增节点 single_unit_causal（单单元因果）为 false（否）。
```

客户端指标增加：

```text
接口路径成员数；
稳定共享成员数；
接口分支成员数；
接口路径完整门数；
Phase332 路径闭合边界。
```

可视化验证：

```text
Phase332 单元测试：8/8 通过；
前端构建：3668 个模块转换并成功；
桌面 1440x900：九族、接口脉络筛选、新指标、原网络形状、非空画布通过；
移动 390x844：接口脉络筛选、原网络形状、文字边界、非空画布通过；
控制台错误：0；
溢出标签：0；
桌面画布 RGB 标准差：(26.018758, 29.486293, 32.643711)；
桌面非背景像素比例：0.490187；
移动画布 RGB 标准差：(28.142007, 29.723510, 29.436007)；
移动非背景像素比例：0.340938；
本地地址：http://127.0.0.1:5173/。
```

### 十一、直接进度向量

不使用没有固定分母的单一总体百分比：

```text
九族注册与观察覆盖：9/9；
72 机制三模型行为、读出和全层普查：72/72；
Phase332 正机制和对照执行：2/2 正机制 + 2/2 对照；
跨模型稳定共享骨架：1/2；
跨模型接口特异分支：1/2；
跨模型路径交换有效：0/2；
Phase332 完整门：0/2；
语言机制行为闭合：0/72；
单神经元因果闭合：0/72；
完整语言编码机制：未完成；
智能理论实验闭合：未完成。
```

“九族工程覆盖 9/9”只表示登记和观察分母完整，不能与“行为闭合 0/72”平均成一个进度百分比。

### 十二、是否自动继续及 Phase333 边界

Phase332 已完整完成注册目标，不能在本阶段后验改变成员、对象、接口或阈值。下一步仍属于“语言模式族物理分布拼图”大阶段，但必须新注册为 Phase333，而不是继续给 Phase332 打补丁。

Phase333 建议一次性完成四个工作包：

```text
工作包 A：动态时序路径
1. 以 missing_condition_control 为主正机制，two_hop_blocked 为严格对照；
2. 在每个生成词元记录 source、query、answer_start 和当前生成位的组件路径；
3. 定位正确答案压力首次形成、第一次被竞争者反超和最终读出三个时间事件；
4. 比较稳定成员的出现顺序，而不是只比较端点集合。

工作包 B：组合状态而非独立成员
1. 同时保存残差、归一化输入、注意力输出和 MLP 输出；
2. 交换连续两到四层的完整局部状态块；
3. 比较单层、连续层块、只共享骨架、只接口分支和联合动态块；
4. 错误时间、错误接口、错误对象和随机块必须配平。

工作包 C：全词表与补偿图
1. 每个时间点记录目标、主要竞争者和继续/协议词元；
2. 建立“组件增量 -> 残差竞争 -> 后层恢复 -> 最终生成”的有向事件表；
3. 要求交换后的竞争变化能沿后续层按预测传播；
4. 若后层恢复抵消早期交换，明确登记补偿路径而不是继续增大干预强度。

工作包 D：升级门
1. 动态路径必须在新对象和新模板保持顺序稳定；
2. 连续状态块交换必须优于错误时间、错误对象和随机块；
3. 完整字符串和自由生成必须同时改善；
4. 对照和协议副作用必须低；
5. 只有以上全部跨三模型通过，才重新打开小规模单神经元扫描门。
```

Phase333 不应直接扩大到九族全量动态扫描。应先用本阶段唯一通过观察双门的 `missing_condition_control（缺失条件控制）` 验证“静态签名与动态机制分离”是否真实存在；若连续状态块仍不能产生预测效应，应停止路径移植路线，转向自然必要性和训练中形成过程的研究。

### 十三、主要产物

```text
tests/gpt5/phase332_interface_branch_case_bank.py；
tests/gpt5/phase332_interface_branch_survey.py；
tests/gpt5/phase332_interface_path_exchange.py；
tests/gpt5/phase332_interface_branch_analysis.py；
tests/gpt5/phase332_publish_interface_atlas.py；
tests/gpt5/run_phase332_interface_branch_atlas.sh；
tests/gpt5/test_phase332_interface_branch_case_bank.py；
tests/gpt5/test_phase332_interface_branch_analysis.py；
tests/gpt5/test_phase332_publish_interface_atlas.py；
tests/gpt5/result/phase332_interface_branch_atlas/interface_branch_atlas/phase332_global_summary.json；
tests/gpt5/result/phase332_interface_branch_atlas/interface_branch_atlas/phase332_report.md；
tests/gpt5/result/phase332_interface_branch_atlas/interface_branch_atlas/phase332_cross_model_summary.jsonl；
tests/gpt5/result/phase332_interface_branch_atlas/interface_branch_atlas/phase332_natural_path_rows.parquet；
tests/gpt5/result/phase332_interface_branch_atlas/interface_branch_atlas/phase332_natural_unit_rows.parquet；
tests/gpt5/result/phase332_interface_branch_atlas/interface_branch_atlas/phase332_exchange_path_rows.parquet；
tests/gpt5/result/phase332_interface_branch_atlas/interface_branch_atlas/phase332_exchange_unit_rows.parquet；
frontend/public/vis_data/pattern_family_neuron_atlas/v1/phase332_interface_path_nodes.jsonl。
```

### 十四、通俗总结

这次没有找到“把一组神经元状态搬过去，模型就按预测答对”的机制。找到的是更基础、也更可靠的一块拼图：同一个语言任务在不同输入接口里，确实会经过不同的内部组件分支；缺失条件推理还存在三模型都能看到的共享骨架。

但这些组件更像一条运行路线留下的路标，不像可以单独搬运的发动机。把路标对应的状态移植到另一接口后，完整答案没有稳定改善。因此当前最重要的改进不是继续扩大静态线性集合，而是追踪这些路标在逐层、逐词元时间里的出现顺序、相互配合和后层补偿。只有动态状态块能够跨新对象、负对照和三模型稳定改变完整生成，才有资格进入更细的单神经元因果扫描。

### 十五、最终工程审计补充

最终严格 JSON 审计还发现 Phase330 发布物中的 4 个历史文件含非有限值：1 个载体集合文件和 3 个 GLM4 分区。发布器已经统一执行失败关闭转换，将所有 `NaN（非数值）` 写为 `null（空值）`，没有改变节点身份、模型层号、组件编号或三维几何。

```text
结果图谱严格 JSON 文件：49/49 可解析；
客户端图谱严格 JSON 文件：49/49 可解析；
非有限字面量：0；
结果目录与客户端目录核心文件一致：7/7；
GPU 模型进程：0；
客户端状态：http://127.0.0.1:5173/，HTTP 200。
```

## Phase 333: 动态时序路径、连续残差块与补偿图谱审计 [2026-07-10 09:07]

### 一、对 Phase332 后续判断的严格审计

附件的核心判断总体正确，但需要收紧为以下口径：

```text
可以保留：
1. Phase332 已部分确认接口条件化的物理签名；
2. 静态端点集合没有表现为可跨接口移植的因果状态；
3. 下一步应观察生成时间、层深和组件类型共同构成的动态轨迹；
4. 连续状态块、错误时间、错误对象、错误接口和配平扰动必须在同一冻结分母比较。

不能推出：
1. 静态端点交换失败，不等于动态状态块必然是机制本体；
2. 接口签名存在，不等于接口分支具有行为必要性或充分性；
3. 观察到后层恢复或持续，不等于已经识别补偿因果边；
4. 工程图谱已覆盖九族，不等于九族物理机制已经闭合；
5. 当前证据不能给出无固定分母的 30%-35% 单一总体进度。
```

另外，不能同时把残差输入、归一化输入、注意力输出、MLP 输出和残差输出全部强制替换。它们在同一个计算图中存在先后依赖，同时替换会把自然计算过度约束，无法判断哪一步产生效应。本阶段观察五类组件，但因果交换只作用于连续层的残差输出。

### 二、冻结问题与样本分母

本阶段只验证 Phase332 唯一通过观察双门的正机制 `missing_condition_control（缺失条件控制）`，并使用 `two_hop_blocked（两跳阻断）` 作为配平对照，没有扩大到九族后再挑选正例。

```text
模型顺序：Qwen3 -> GLM4 -> DS7B；
执行方式：本地 CUDA，逐模型加载和释放；
新对象：12 个，均未用于 Phase332；
模板：3 个；
接口：raw_completion、native_chat、answer_aligned_chat；
机制：1 个正机制 + 1 个配平对照；
自然动态案例：648；
发现集：324；
校准集：162；
持出集：162；
逐词元记录：35,772；
五组件动态路径：6,310,960；
冻结块计划：18；
持出交换案例：108；
每案例条件：9；
条件生成：972；
干预后动态响应：134,784。
```

九种条件固定为：

```text
baseline；
correct_block_1；
correct_block_2；
correct_block_4；
wrong_object_block_4；
wrong_interface_block_4；
wrong_time_block_4；
moment_matched_permutation_block_4；
matched_control_block_4。
```

层块只由发现集冻结。最初的“最大投影层”会机械偏向最后读出层，因此在查看持出效果前，正式选择规则改为“相邻层目标残差写入增量最大的正增量层”。旧 Qwen3 最后层试运行结果已隔离在 `exchange_pilot_late_readout_qwen3`，不进入正式汇总。

### 三、基础物理记录公式

对模型、机制、接口、对象、生成时刻、层和组件，记录目标词与主要竞争词之间的基础读出差：

$$
D_{m,k,i,o,\tau,l,c}
=
\left\langle
C_{l,c}(h_{\tau}),
e_{target}-e_{competitor}
\right\rangle
$$

其中 $C_{l,c}$ 只表示残差输入、归一化输入、注意力输出、MLP 输出或残差输出中的一个被观察组件。该线性读出是定位工具，不是完整运行机制公式。

目标压力形成时刻定义为：

$$
\tau_{form}
=
\min\left\{
\tau:\ D_{m,k,i,o,\tau,L,residual}>0
\right\}
$$

冻结候选层使用正残差写入增量：

$$
l^{*}
=
\arg\max_l
\left[
D_{\tau,l,residual\_output}
-
D_{\tau,l-1,residual\_output}
\right]_{+}
$$

连续块交换只在功能时刻对齐后执行：

$$
h^{R}_{\tau,l+1}
\leftarrow
Align_{D\rightarrow R}
\left(h^{D}_{\tau',l'+1}\right),
\quad l\in B_w,\quad w\in\{1,2,4\}
$$

这里 $D$ 是供体接口，$R$ 是接收接口，$Align$ 只按冻结的功能层顺序和功能生成时刻对齐，不要求不同模型或不同接口具有相同绝对层号。

补偿只登记为干预后的下游差分：

$$
\delta_l
=
D^{patched}_{l,residual\_output}
-
D^{baseline}_{l,residual\_output}
$$

若块结束层为 $l_b$，候选恢复层为：

$$
l_{recover}
=
\min\left\{
l>l_b:
|\delta_l|\leq 0.5|\delta_{l_b}|
\right\}
$$

只有 $l_b$ 后存在真实下游层，才允许登记 `recovered（恢复）` 或 `persisted（持续）`。块覆盖到最后一层时，末层与块尾是同一测量点，必须登记为 `unresolved（未解析）`，不能把它当作补偿已解释。

### 四、客观结果：动态时序

```text
18 个“模型 × 机制 × 接口”时序单元中，稳定单元为 12/18；
正机制稳定单元为 8/9；
Qwen3：6/6 稳定；
GLM4：3/6 稳定；
DS7B：3/6 稳定。
```

正机制的三接口功能对齐结果：

```text
Qwen3：通过；持出峰层跨度 0.0285715；
GLM4：失败；仅 2/3 接口稳定，持出峰层跨度 0.5128205；
DS7B：通过；持出峰层跨度 0.0370370；
跨三模型门：失败，2/3 模型通过。
```

GLM4 的原始补全接口在中层形成候选写入，而聊天接口在最后几层形成候选写入。这支持“接口会改变物理运行路线”的局部观察，但也直接否定“一个统一相对深度可以跨接口描述全部路径”。

Qwen3 和 DS7B 的候选层大多位于相对深度 0.86-1.00。这个现象可能是答案读出形成，不应解释为语言机制源头已经定位。

### 五、客观结果：连续状态块

```text
自由生成阶段干预命中：864/864；
教师强制短语阶段干预命中：664/864；
Qwen3 短语命中：282/288；
GLM4 短语命中：184/288；
DS7B 短语命中：198/288；
无效条件指标：116，全部来自 GLM4 的缺失值；
非有限浮点字面量：0；
缺失值全部按失败关闭，没有补零。
```

12 个“模型 × 机制 × 交换方向”单元中：

```text
correct_block_specific：0/12；
正机制特异块：0/6；
四层块长度单调：局部存在，但不能跨方向和三模型稳定；
正机制六单元平均短语对数概率变化：-1.6286055；
正机制六单元平均目标排名改善：-1813.9259259；
正机制自由生成平均增益率：0.0925926；
完整字符串竞争门：失败；
配平对照干净门：失败。
```

局部正向现象不能越过对照：

```text
Qwen3 raw_to_answer_aligned 的四层块短语增量为 +16.7988065，
但最大配平对照同样为 +16.7988065，自由生成增益为 0；

DS7B raw_to_answer_aligned 的四层块短语增量为 +0.5516048，
但最大配平对照为 +0.5679655，目标排名反而下降 128.6666667；

GLM4 两个正机制方向的平均短语增量均为负，且存在缺失指标。
```

因此不能声称连续残差块优于静态端点，也不能声称它是可移植因果状态。

### 六、客观结果：补偿候选

收紧“必须存在块后下游层”的定义后：

```text
具有真实下游轨迹的候选：33/108；
Qwen3：18/36；
GLM4：15/36；
DS7B：0/36；
正机制候选：18/54；
对照机制候选：15/54；
跨三模型补偿解释门：失败。
```

Qwen3 和 GLM4 的部分 answer_aligned_to_raw 方向出现滞后恢复，但反向和 DS7B 不稳定；对照机制也出现相近候选。因此这些边只能标为 `L3_lagged_compensation_candidate（三级滞后补偿候选）`，不能标为机制因果边。

### 七、七道严格门

```text
dynamic_sequence_stable：失败；
state_block_effective：失败；
competition_consistent：失败；
compensation_explained：失败；
free_generation_improved：失败；
matched_controls_clean：失败；
cross_model_execution：通过；
full_gate_pass：失败；
行为机制闭合：0；
单神经元因果：0；
单神经元扫描门：关闭。
```

阶段333 是“完整执行 + 强负结果 + 局部动态观察”，不是闭合阶段。

### 八、对当前理论拼图的更新

当前可以继续保留、但不能升级证据等级的拼图：

```text
1. 语言能力不是一个固定概念向量或单神经元地址；
2. 同一任务在不同接口下会出现不同的层深和组件路径；
3. 正确目标压力具有生成时序，不能只看最终端点；
4. 残差、注意力和 MLP 的局部写入会被后层继续变换；
5. 部分后层恢复现象真实存在，但目前既不跨模型，也不区别正机制和对照；
6. 当前冻结的线性目标差主要定位接近输出端的读出形成，不能代表完整非线性机制。
```

“语言是动态模式网络”仍可作为统一研究假设。更谨慎的数据对象是条件轨迹：

$$
\Gamma(P\mid x,i,m)
=
\left\{
(\tau,l,c,D,\Delta h,rank,token)
\right\}
$$

其中路径依赖输入 $x$、接口 $i$ 和模型 $m$。这只是图谱记录结构，不是已经闭合的智能理论公式。

Transformer 的真实递推仍至少包含：

$$
h_{l+1,\tau}
=
h_{l,\tau}
+A_l\left(N_l(h_{\leq\tau,l})\right)
+M_l\left(N'_l(h_{\tau,l}+A_l)\right)
$$

$$
y_{\tau}
=
Decode\left(N_f(h_{L,\tau})W_U\right)
$$

当前实验只测到了该递推中的少量条件投影和干预响应。线性方向可以帮助寻找拼图，但不能替代 $A_l$、$M_l$、归一化、跨位置注意力和自回归反馈共同构成的非线性运行机制。

### 九、问题、硬伤和小模型边界

```text
1. 候选层选择仍依赖目标与竞争词的线性输出方向，容易抓到晚期读出；
2. 只验证了推理约束族中的一个正机制和一个对照，不能外推九族；
3. 连续块交换检验的是接口间充分性，不等于自然运行中的必要性；
4. 教师强制短语干预未在全部条件命中，GLM4 还有 116 个缺失指标；
5. 补偿分类只有层级滞后，没有证明具体注意力头、MLP 通道或神经元承担恢复；
6. 自由生成变化容易混入协议、思考模板和接口格式变化；
7. 三个模型规模较小、架构和对齐方式不同，粗糙路径可能放大晚期读出和接口差异；
8. 不能主观指定“小模型与真实语言编码偏差 30% 或 50%”，只能把模型规模作为外推风险；
9. 语言机制行为闭合与单神经元因果闭合仍为 0/72；
10. 当前结果不支持继续扩大同类 donor-state transplant（供体状态移植）。
```

### 十、语言模式图谱与客户端同步

图谱保持此前的逐层矩形网络形状，没有改成大范围无关神经元云。阶段333只新增 18 个动态事件锚点：

```text
3 模型 × 2 机制 × 3 接口 = 18；
正机制使用绿色；
对照机制使用红色；
接口事件按冻结功能顺序连线；
节点明确标记 dynamic_path_event；
single_unit_causal 全部为 false；
full_gate_pass 全部为 false。
```

客户端新增 `动态时序` 筛选和以下指标：动态事件锚点、稳定时序接口、特异状态块单元、补偿候选、动态路径完整门。选择 Qwen3 时显示 6 个动态锚点、6 个稳定接口、0 个特异块、18 条候选补偿轨迹、0 个完整门。

```text
发布前节点：2,271；
新增动态锚点：18；
更新分区：3；
后端测试：7/7 通过；
Vite 生产构建：通过；
桌面 1440x900：通过；
移动 390x844：通过；
浏览器错误：0；
控件溢出：0；
桌面画布非背景比例：0.475750；
移动画布非背景比例：0.346000；
结果与客户端严格 JSON：通过；
本地地址：http://127.0.0.1:5173/。
```

### 十一、直接进度向量

```text
九族注册和基础观察覆盖：9/9；
72 机制三模型普查：72/72；
Phase333 自然动态执行：648/648；
Phase333 条件干预执行：972/972；
正机制稳定时序单元：8/9；
正机制跨模型三接口对齐：2/3 模型，严格门 0/1；
正机制特异连续块：0/6；
具有真实下游层的补偿候选：33/108；
Phase333 完整门：0/1；
已完成同等级动态深审机制：1/72；
语言机制行为闭合：0/72；
单神经元因果闭合：0/72；
完整语言编码机制：未完成；
智能理论实验闭合：未完成。
```

不能把 9/9 工程登记覆盖、1/72 动态深审和 0/72 因果闭合平均成一个总体百分比。

### 十二、是否继续及 Phase334 系统任务

Phase333 已完成冻结目标。按照 Phase332 预注册的分支规则，连续状态块没有产生跨三模型、优于配平对照的预测效应，因此不应继续自动扩大状态移植、提高干预强度或进入全量单神经元 CUDA 扫描。

下一阶段仍服务于语言模式族物理分布拼图，但研究问题需要从“跨接口状态充分性”切换为“接收者自然路径必要性”，注册为 Phase334：

```text
工作包 A：自然必要性
1. 不再使用另一接口作为供体；
2. 只消融接收者自身在自然运行中形成的注意力写入、MLP 写入或连续残差增量；
3. 错误时间、错误对象、同范数置换和配平机制作为对照；
4. 要求目标排名、完整字符串和自由生成同步下降，才能标记必要性候选。

工作包 B：源头而非末端读出
1. 发现集同时冻结早期、中期和晚期候选，不再只取最大线性写入；
2. 记录 source、query、answer_start 和生成位之间的跨位置传递；
3. 比较注意力写入先发生、MLP 写入后放大、残差竞争最终读出的实际顺序；
4. 若只有末端读出有作用，明确登记为 readout bottleneck，而不是模式源头。

工作包 C：非线性基础分析
1. 保留真实 top-k 排名、词元转移和组件增量，不用单一余弦或复杂统计代替物理轨迹；
2. 对每个候选做“删除后是否缺失”的必要性检查；
3. 不把投影相关性当因果，不把同一最终层测量当传播；
4. 正机制必须明显区别配平对照和协议副作用。

工作包 D：形成过程
1. 先审计本地是否存在同架构训练检查点序列；
2. 若存在，冻结同一案例追踪路径从早期到成熟检查点的形成顺序；
3. 若不存在，不伪造训练形成结论，只完成成熟模型自然必要性图谱。
```

Phase334 属于新的因果问题，不应在 Phase333 内后验追加。当前可以自动完成的是 Phase334 的固定格式注册和工具准备；在新分母冻结前，不应继续产生理论结论。

### 十三、主要产物

```text
tests/gpt5/phase333_dynamic_case_bank.py；
tests/gpt5/phase333_dynamic_survey.py；
tests/gpt5/phase333_residual_block_exchange.py；
tests/gpt5/phase333_dynamic_analysis.py；
tests/gpt5/phase333_publish_dynamic_atlas.py；
tests/gpt5/run_phase333_dynamic_path_atlas.sh；
tests/gpt5/test_phase333_dynamic_case_bank.py；
tests/gpt5/test_phase333_dynamic_analysis.py；
tests/gpt5/test_phase333_publish_dynamic_atlas.py；
tests/gpt5/result/phase333_dynamic_path_atlas/dynamic_path_atlas/phase333_global_summary.json；
tests/gpt5/result/phase333_dynamic_path_atlas/dynamic_path_atlas/phase333_report.md；
tests/gpt5/result/phase333_dynamic_path_atlas/dynamic_path_atlas/phase333_sequence_summary.jsonl；
tests/gpt5/result/phase333_dynamic_path_atlas/dynamic_path_atlas/phase333_block_local_summary.jsonl；
tests/gpt5/result/phase333_dynamic_path_atlas/dynamic_path_atlas/phase333_compensation_candidates.jsonl；
tests/gpt5/result/phase333_dynamic_path_atlas/dynamic_path_atlas/phase333_dynamic_path_rows.parquet；
tests/gpt5/result/phase333_dynamic_path_atlas/dynamic_path_atlas/phase333_dynamic_response_rows.parquet；
tests/gpt5_temp/phase333_client_visual_check.cjs；
tests/gpt5_temp/phase333_canvas_pixel_check.py；
frontend/public/vis_data/pattern_family_neuron_atlas/v1/phase333_dynamic_event_nodes.jsonl。
```

### 十四、通俗总结

这次把“只看一张神经网络快照”改成了“逐词元看一段内部运行录像”。录像里确实能看到：同一个问题换成不同接口后，正确答案压力出现的层和时间会变化；部分模型还会在后层抵消前面的扰动。

但把连续四层状态从一个接口搬到另一个接口，并没有稳定地让模型更正确。很多表面正向变化也会在错误对象、错误接口或配平对照中同样出现。因此这些动态脉络现在只能作为可视化研究路标，不能叫作已经破解的语言编码线路。

最重要的路线调整是停止继续搬运外部状态，改问一个更基础的问题：模型自然运行时，删除哪一段自身路径会让对应语言能力消失，而且这种消失是否只发生在正确时间、正确对象和正确机制。只有自然必要性跨新样本和三模型成立，才值得继续下钻到具体注意力头、MLP 通道和单神经元。

## Phase 334: 接收者自然路径必要性图谱与停止局部删除分支 [2026-07-10 11:08]

### 一、对 Phase333 外部审计文本的判断

审计文本的主方向正确：Phase333 只部分确认了接口分叉的动态物理签名，没有证明静态端点集合是可移植因果状态；研究应当从供体状态移植转向接收者自然路径必要性，而不是继续提高移植强度。

但其中有五处必须收紧：

1. 不能把工程字段、分析工具、局部观察和机制闭合都计作同一种“拼图”；72 个机制才是机制分母，脚本数量不增加机制完成度。
2. `attribute`、`two_hop_blocked` 和 `plural_agreement` 是独立的配平机制，不是无语言结构的空白对照。
3. 过去材料、过去时和复数协议的基线并非天然合格，必须逐模型、逐接口先通过基线资格门，再进入必要性统计。
4. “正机制 8/9、对照 4/9”只能描述观察频率，不能证明机制特异性，更不能证明因果闭合。
5. 本轮知识案例在提示中显式提供事实，所以测到的是关系绑定与检索，不是参数知识从训练中形成的来源。

因此，本阶段不接受单一总体进度百分比，也不接受“72 个拼图大部分已完成”的表述，改用彼此不可互换的进度向量。

### 二、冻结问题与样本结构

本阶段冻结三个模式族、六个机制、三个模型和三个接口：

```text
内容知识族：material、attribute；
推理约束族：missing_condition_control、two_hop_blocked；
语法结构族：past_tense、plural_agreement；
模型：Qwen3-4B、GLM4-9B、DeepSeek-R1-Distill-Qwen-7B；
接口：raw_completion、native_chat、answer_aligned_chat。
```

每个机制使用 12 个新对象，扩展为 3 个模板、3 个接口和 3 个模型，共登记 1944 个案例；发现集 972，校准集 486，留出集 486。发现、校准、留出之间不允许更新选择规则。

本地检查点审计结果为三个模型同一次训练的中间检查点数量均为 0。因此本阶段只研究成熟模型的自然运行，不声称观察到语言结构的训练形成过程。

### 三、基础算法

对自然运行中候选组件的差分，不直接比较绝对范数，而使用相对于两侧自然状态尺度的基础比值：

$$
R_{c,l,p}=\frac{\left\|v_{c,l,p}(x)-v_{c,l,p}(x')\right\|_2}
{\frac{1}{2}\left(\left\|v_{c,l,p}(x)\right\|_2+\left\|v_{c,l,p}(x')\right\|_2\right)+\epsilon}
$$

其中 $c$ 是注意力写入、MLP 写入或残差增量，$l$ 是层，$p$ 是 source、query 或 answer_start 位置。早期试跑曾按绝对范数选择，导致 Qwen3 的 54 个候选全部落到残差增量；该试跑已隔离到 `calibration_pilot_raw_scale_qwen3`，没有进入正式结论。

正式干预只删除接收者自然形成的组件增量：

$$
h'_{l,p}=h_{l,p}-\alpha v_{c,l,p}(x),\qquad \alpha=1
$$

它与供体完整状态移植不同。对照包括错误时间、错误对象增量、配平机制增量、同矩置换和错误层删除；留出集同时执行正确组件删除、注意力删除、MLP 删除、残差删除与联合删除。

目标字符串损失、目标排名损失和行为损失分别为：

$$
L_{phrase}=\log P(y\mid x)-\log P(y\mid do(-v_c),x)
$$

$$
L_{rank}=rank(y\mid do(-v_c),x)-rank(y\mid x)
$$

$$
L_{behavior}=\mathbb{1}[\text{基线完成目标且干预后未完成目标}]
$$

只有同一个案例在基线正确、目标首词元排名不超过 50、完整目标概率有限、11 个干预条件全部命中时，才进入共同有效集：

$$
V=V_{base}\cap V_{rank}\cap V_{phrase}\cap\bigcap_{j=1}^{11}V_{patch,j}
$$

必要性候选必须同时满足足够的共同有效案例、正确删除造成目标概率或排名损失、行为受损，并且正确删除明显强于所有配平对照。下游传播只作独立的物理迹象，不能替代必要性门。

### 四、CUDA 执行与固定数据量

三个模型严格按 Qwen3、GLM4、DeepSeek7B 的顺序加载、执行和释放显存，没有并行驻留。

```text
登记案例：1944；
基线生成：1944；
自然差分记录：808704；
发现候选计划：162；
校准案例：486；
校准条件：1458；
冻结必要性计划：54；
留出案例：486；
留出条件与生成：5346；
下游响应记录：84834；
调查、校准、留出有效性：全部通过；
后验选择更新：禁止；
训练形成轨迹：0/3 模型。
```

正式冻结的组件分布不是单一残差方向：Qwen3 为 3 个注意力、15 个 MLP；GLM4 为 3 个注意力、15 个 MLP；DeepSeek7B 为 16 个 MLP、2 个残差增量。这说明相对尺度校正改变了候选排序，但不等于这些组件具有因果必要性。

### 五、客观结果

54 个“模型 × 机制 × 接口”单元中，只有 24 个达到至少 6 个基线合格案例：Qwen3 为 9/18，GLM4 为 12/18，DeepSeek7B 为 3/18。

```text
局部自然必要性候选：0/54；
局部下游传播通过：2/54；
局部完整门：0/54；
跨模型自然必要性：0/6；
小规模单神经元扫描门：0；
无效条件指标：0；
行为机制闭合：0/72；
单神经元因果闭合：0/72。
```

两个传播弱正迹象都来自 GLM4 的 `past_tense`：

```text
answer_aligned_chat：基线/共同有效 9/9，传播率 8/9；
正确删除目标字符串损失 0.254，目标排名损失 6.22，行为损失 0；
最大对照字符串损失 1.634。

native_chat：基线/共同有效 7/7，传播率 6/7；
正确删除目标字符串损失 2.199，目标排名损失 248.57，行为损失 0；
最大对照字符串损失 3.662。
```

两者都因正确删除不强于对照且没有自由生成行为损失而失败。它们只能说明扰动可沿后续层传播，不能说明被删组件是过去时机制的必要源头。

### 六、最重要的负结果

自然状态中差异最大的注意力写入、MLP 写入或残差增量，在新对象和新模板上均没有成为跨模型、跨接口、优于配平对照的必要组件。

这个结果排除了一个过强解释：

```text
“每个语言机制都由一个可按最大自然差分找到、删除后能力消失的局部组件承载。”
```

它没有排除语言模式存在物理路径。当前仍有至少四种基础解释：

1. 功能由多个组件冗余实现，删除单个最大差分组件会被同层或后层补偿。
2. 最大差分是读出结果而不是上游生成原因。
3. 真正关键变量是注意力、MLP 与残差之间的非线性联合状态，不能用一个线性增量表示。
4. 当前任务协议和小模型粗糙结构降低了可观测性，尤其自由生成与单词目标并不完全一致。

### 七、硬伤与边界

第一，基线资格只有 24/54。Qwen3 在复数案例中经常生成语义和语法正确的完整句子，但协议要求单词式目标，导致首目标词元排名门失败；这暴露的是任务协议与自然生成耦合，不是“模型没有复数语法”。

第二，Qwen3 的原生聊天在 24 个生成词元内常停留在思考段，没有进入最终答案，也造成资格损失。后续必须把“自然能力失败”和“答案尚未到达”分开。

第三，本轮每个机制深审的留出对象只有 3 个模板下的 9 个模型接口案例；它足以否定当前强门候选，不足以证明所有可能分布式机制不存在。

第四，三个测试模型规模较小且架构相关。跨模型失败不能直接推广到更大模型；小模型内部路径可能更粗糙，也可能更依赖接口协议。

第五，当前干预仍以组件输出为单位，不是全量单神经元 CUDA 扫描；因为自然必要性门为 0，继续下钻神经元会扩大伪候选而不是提高证据等级。

第六，线性删除只回答局部增量是否必要，不能模拟真实非线性门控、归一化、注意力竞争和后层补偿。因此不得把本阶段写成“线性公式已经描述真实运行机制”。

### 八、图谱与客户端同步

固定格式图谱新增 54 个自然必要性节点，更新 9 个族-模型分区。旧的层叠网络三维形状保持不变，在其上叠加黄色自然主候选、青色配平路径，并把候选组件、传播迹象、必要性门和闭合状态分开显示。

桌面 1440×900 与移动端 390×844 均通过浏览器检查：9 个模式族选项存在，Phase334 指标可见，自然必要性焦点可切换，控件未越界，画布非空比例分别为 0.4903 和 0.3642。可视化显示的是研究证据，不把未通过门的节点绘制为已破解神经元。

### 九、当前真实进度向量

```text
九族工程登记覆盖：9/9；
72 机制名录覆盖：72/72；
自然必要性同等级深审：6/72；
跨模型自然必要性候选：0/6；
行为机制闭合：0/72；
单神经元因果闭合：0/72；
完整语言编码机制：未完成；
智能理论实验闭合：未完成。
```

这组向量不能平均成“总体完成百分比”。登记覆盖回答有没有位置，深审覆盖回答有没有完整测量，闭合回答有没有跨样本因果证据，它们是不同问题。

### 十、理论进展与第一性原理判断

“语言是动态模式网络”仍可作为工作假设，但本阶段要求把它写得更严格：语言能力不是静态向量、固定端点集合或最大差分组件，而可能是输入条件触发后，多个组件在不同位置和时间上共同维持的受约束状态转移。

当前最小机制表达应保留非线性和补偿项：

$$
h_{l+1,t}=h_{l,t}+A_{l,t}(H_{\leq t})+M_{l,t}(h_{l,t})+C_{l,t}(H_{\leq t},q)+\varepsilon_{l,t}
$$

其中 $A$ 是注意力写入，$M$ 是 MLP 写入，$C$ 代表归一化、竞争、门控及后层补偿的合成影响。该式只是测量坐标，不是已经闭合的统一理论。真正需要寻找的是：删除一个由多个组件组成的最小充分路径集合后，目标能力是否在正确对象、正确时间和正确机制上稳定消失，并且能够由同机制自然路径恢复。

### 十一、阶段停止规则与下一步

Phase334 已完整执行预注册分母，属于 Phase333 之后的同一自然因果审计阶段，现已达到阶段性停止条件。由于 0/54 单元通过自然必要性门，不应自动继续以下工作：

```text
不扩大同类单组件删除；
不提高线性删除强度；
不启动全量单神经元 CUDA 扫描；
不把两个传播迹象升级为必要机制；
不继续用小补丁修正统一线性公式。
```

下一阶段应注册为新的系统分支，而不是 Phase334 的后验追加：

```text
Phase335：分布式功能模块与形成来源审计

工作包 A：重做协议资格层
- 将答案到达、自然生成正确、目标字符串正确和首词元排名拆开；
- 为语法族使用完整句最小对，而不是强制单词回答；
- 先提高各模型接口的可审计基线覆盖，再做因果比较。

工作包 B：从单组件转向冻结的小集合
- 仅在发现集构建注意力+MLP+残差的小型候选集合；
- 在校准集冻结集合大小和删除顺序；
- 留出集要求集合删除强于每个单组件和所有配平对照；
- 记录补偿组件是否按固定时间顺序接管。

工作包 C：参数与形成来源
- 优先寻找同架构、同训练运行的真实检查点序列；
- 若没有检查点，不声称破解训练形成，只登记证据缺口；
- 有检查点后追踪候选集合何时出现、何时获得可泛化功能。

工作包 D：神经元下钻门
- 只有跨模型或至少两模型同构架构的小集合必要性成立，才在集合内部做 CUDA 单神经元扫描；
- 神经元证据必须同时包含自然激活、删除必要性、配平对照和下游中介。
```

Phase335 需要先修订任务协议，并需要真实检查点或更合适模型资源才能完整回答形成问题，因此本轮不盲目自动运行另一个同类局部干预。暂停的是低收益实验分支，不是语言模式图谱工程。

### 十二、主要产物

```text
tests/gpt5/phase334_natural_necessity_case_bank.py；
tests/gpt5/phase334_natural_contrast_survey.py；
tests/gpt5/phase334_natural_necessity_intervention.py；
tests/gpt5/phase334_natural_necessity_analysis.py；
tests/gpt5/phase334_publish_natural_necessity_atlas.py；
tests/gpt5/run_phase334_natural_necessity_atlas.sh；
tests/gpt5/result/phase334_natural_necessity_atlas/natural_necessity_atlas/phase334_global_summary.json；
tests/gpt5/result/phase334_natural_necessity_atlas/natural_necessity_atlas/phase334_report.md；
tests/gpt5/result/phase334_natural_necessity_atlas/natural_necessity_atlas/phase334_local_necessity_summary.jsonl；
tests/gpt5/result/phase334_natural_necessity_atlas/natural_necessity_atlas/phase334_cross_model_summary.jsonl；
tests/gpt5/result/phase334_natural_necessity_atlas/natural_necessity_atlas/phase334_natural_contrast_rows.parquet；
tests/gpt5/result/phase334_natural_necessity_atlas/natural_necessity_atlas/phase334_downstream_response_rows.parquet；
frontend/public/vis_data/pattern_family_neuron_atlas/v1/phase334_natural_necessity_nodes.jsonl；
tests/gpt5_temp/phase334_client_visual_check.cjs；
tests/gpt5_temp/phase334_canvas_pixel_check.py。
```

### 十三、通俗总结

这次不再把另一种提示方式的内部状态搬进来，而是直接删除模型自己正常运行时产生的注意力、MLP 或残差片段。三个模型、六类语言机制和三种接口全部按同一规则测试后，没有一个候选能稳定满足“删对地方能力下降，而且删错地方不下降”。

GLM4 的过去时有两条扰动会继续传到后层，但对照扰动更强，最终回答也没有因此失败，所以它们只是可视化路标，不是过去时的关键开关。

目前最可靠的新认识不是“找到了某个神经元”，而是“最大自然差分不等于必要机制”。下一步必须先把自然回答协议修好，再研究多个组件共同组成的最小功能集合；只有集合必要性成立，才值得下钻到集合内部的单神经元。这样图谱会增长得慢一些，但每个亮起来的节点都有明确证据等级，不会把相关性误画成破解结果。

## Phase 335: 三核心问题审计与分层因果规则提取算法 [2026-07-10 13:28]

### 问题重新定义

本阶段不运行新模型，综合 `research/IntelligentTheory.md` 与 Phase323-330 的研究记录，审计三个核心问题：语言理论如何组织、如何从自回归网络提取有效神经元与规则、如何在小模型和局部近似条件下形成严格闭合。

这三个问题不是并列问题，而是一条依赖链：

```text
可操作的语言变量
-> 成对对照样本
-> 条件路径发现
-> 分层因果缩小
-> 上下游中介
-> 自然生成闭合
-> 可预测规则。
```

### 项目现有证据给出的约束

Phase326 的初版每层 Top-3 候选只覆盖少量对象，并因选择过程出现接近 100% 的正向率，该结果已被废弃；直接 $W_U$ 读出评分还天然偏向末层。Phase330 虽完成九族、72 机制、三模型的全层覆盖，但严格结果是：

```text
留出峰层精确预测：0.6119；
10% 深度容差命中：0.8315；
跨三模型集合级读出特异：5/72；
跨三模型自然身份：14/72；
跨三模型可见行为必要性：0/72；
单神经元因果：0；
完整自然语言链：0。
```

因此，当前已经证明“路径具有部分可重复结构”，但没有证明最大激活、最大读出贡献、单个神经元或某个线性方向就是语言规则。

### 问题一：语言理论应如何组织

九大模式族应保留，但定位必须从“语言本体分类”降为 benchmark coverage taxonomy（基准覆盖分类）。更适合作为理论核心的是三层结构：

```text
语义变量层：对象、关系、角色、属性、作用域、目标、候选、协议；
操作基元层：读取、绑定、比较、否定、路由、写入、抑制、停止、继续；
物理实现层：token position、attention head、MLP neuron/group、residual、norm、W_U。
```

每条机制规则登记为：

$$
Rule_r=
[Trigger_r,StatePre_r,Route_r,PhysicalPath_r,StateUpdate_r,
Competition_r,Closure_r]
$$

其中九族回答“测试覆盖了什么任务”，规则回答“模型在什么条件下执行什么状态变换”。理论对象不再是模式族名称，而是能够跨任务复用的操作基元和条件转移规则。

### 问题二：有效神经元和内部规则如何提取

#### 有效神经元定义

有效神经元不能按激活值或投影值定义。对给定机制 $r$ 和上下文集合 $X$，真实有效单元 $u$ 至少需要满足：

```text
自然参与：机制触发时处于对应计算路径；
必要性：低副作用移除后目标竞争或完整生成按预测下降；
受控充分性：在损坏状态中恢复该单元能够恢复部分功能；
中介性：上游变化能够改变该单元，恢复该单元能中介下游结果；
留出稳定：跨对象、模板和私有留出保持方向；
特异性：随机同规模、错层、错供体和负模式族不能复制效果。
```

若单个神经元不满足，但最小集合满足，则科学对象应登记为 `minimal causal set`，不能为了获得单神经元结论而拆散真实分布式机制。

#### 统一读数

对目标完整短语 $y^+$ 和最强竞争短语集合 $Y^-$，定义：

$$
M(x)=
\operatorname{MeanLogP}(y^+|x)
-
\max_{y^-\in Y^-}\operatorname{MeanLogP}(y^-|x)
$$

竞争集合至少包含 wrong、prose、echo、protocol、continue 和 stop。首词元边距只能作为辅助读数，不能作为最终目标。

#### 分层因果路径提取算法 HCPE

算法名称：Hierarchical Causal Path Extraction（分层因果路径提取，HCPE）。

第一步，冻结 discovery、calibration、heldout、private-heldout，并按基线状态分成 capable、unstable、incapable。机制提取只在 capable 上判断必要性，模型失败单独建图，不能混入机制均值。

第二步，从层和组件块开始真实干预，不按激活 Top-K 筛选。依次测试 attention、MLP、residual 和位置角色的 zero、half、mean replacement，并与随机同规模和错层对照比较。

第三步，对通过的 MLP 组执行递归二分：

```text
父组 U -> U_left + U_right；
分别移除 left、right 和 joint；
保留有特异效应的分支；
若两支单独弱而联合强，则保留交互集合；
直到单神经元或不可再分的最小交互集合。
```

为避免神经元编号顺序造成偏差，同一父组使用至少两种固定平衡划分复核，而不是只按连续索引切半。

第四步，计算必要性、充分性和特异性：

$$
Nec(U)=M_{base}-M_{remove(U)}
$$

$$
Suf(U)=M_{restore(U)}-M_{corrupt}
$$

$$
Spec(U)=Nec(U)-\max(Nec(U_{random}),Nec(U_{wronglayer}))
$$

第五步，检查非线性交互：

$$
I(U_1,U_2)=
\Delta M_{U_1\cup U_2}
-\Delta M_{U_1}
-\Delta M_{U_2}
$$

若 $|I|$ 超过预注册阈值，机制节点应是交互集合或条件边，而不是把联合效应分配给最高值神经元。

第六步，建立因果中介边。只有当上游单元 $a$ 的干预改变下游单元 $b$ 的自然状态，并且在干预 $a$ 后恢复 $b$ 可以恢复输出时，才登记 $a\rightarrow b$：

$$
Med(a\rightarrow b)=
\frac{M_{a\ removed,b\ restored}-M_{a\ removed}}
{M_{base}-M_{a\ removed}}
$$

第七步，逐层记录干预后的注意力、MLP、残差和全词表变化，显式发现 compensation path。若移除 $U$ 后其他路径接管，$U$ 只能标记为可替代成员，不能标记为规则本体。

第八步，在 private-heldout 上让图谱提前预测 dominant layer、物理集合、作用符号、主要 blocker、干预幅度区间和 rollout 结果。实验完成后再揭示真实值；不能用同一留出集反复调阈值。

### 从物理路径提取规则

规则不是从权重矩阵直接读取的一条自然语言公式，而是从重复因果路径中归纳出的条件转移：

$$
\hat z_{t,l+1}=F_r(z_{t,l},Condition(x_t))
$$

$$
\hat M_{t+1}=G_r(\hat z_{t,L},CompetitorSet_t)
$$

一个候选规则只有在未见样本上同时预测“何时触发、走哪条路径、改变哪个竞争者、干预后如何变化”时才成立。不同模型可以使用不同神经元地址，只要在操作基元、因果顺序和行为作用上同构，就可登记为同一功能规则的不同物理实现。

### 问题三：如何形成闭合

不应寻找一个覆盖全部语言的全局拟合公式。Transformer 本身是条件化、分段和非线性的，闭合应分为三个层次：

#### 局部机制闭合

在一个自回归时间步内，触发、状态、路径、竞争和读出具有可预测的必要性、充分性与中介链。

#### 序列闭合

局部机制在后续 token 中没有被 prose、echo、continue、格式或补偿路径夺走，完整生成保持语义与协议一致。

#### 理论闭合

规则在独立对象、模板和 private-heldout 上可预测；跨模型不要求物理地址相同，只要求功能结构可比较；反例能够明确缩小规则作用域。

闭合向量应保留各维度，不能压成一个平均百分比：

$$
ClosureVector=
[Predictive,Necessary,Sufficient,Mediated,Specific,
LowSideEffect,RolloutStable,Heldout]
$$

八项全部通过才称 clean closure。小模型只能先完成 model-specific closure；两个以上模型出现功能同构后才能称 family-level mechanism；更大模型和更多架构复现后才讨论通用语言原理。

### 可行性判断

```text
完整破解所有语言规则：当前不可直接完成；
恢复一个模式机制的最小因果状态转移：可行；
找到少量真实有效单神经元：可能，但不能预设一定存在；
找到最小分布式因果集合：比单神经元目标更可行；
从多个局部闭合规则归纳复用骨架：是当前最合理的理论路线；
由三个小模型直接推出人脑或通用智能公式：证据不足。
```

### 下一阶段大任务

冻结 Phase330 的 5 个跨模型集合级读出正机制，统一执行扩大留出、双接口、递归二分、逐成员移除、随机平衡划分、充分性恢复、上游中介、补偿追踪和完整 rollout。目标不是再增加九族覆盖，而是验证 HCPE 是否能把至少一个组件集合缩小为可复现的单神经元或最小交互集合，并恢复一条 private-heldout 可预测的条件规则。

### 严格边界

```text
本阶段没有运行 CUDA；
没有新增模型结果；
HCPE 是根据既有负结果和证据缺口设计的待验证算法；
现有 5/72 只属于集合级读出候选，不是行为闭合；
现有单神经元因果和完整自然链仍均为 0。
```

## Phase 336: 同类工具旁路验证与分阶段工程改进方案 [2026-07-10 20:19]

### 一、两个方案文件的总体判断

`20260711_01_同类项目比较.md` 的正确部分是：本项目不应重复实现所有底层机制解释工具；MIB 的独立留出、因果定位和电路忠实性评测值得吸收；项目真正的差异应保留在三模型语言模式图谱、统一证据格式和三维物理脉络。

`20260711_02_三个工程难点.md` 的正确部分是：九族更适合作为测试分类，不是最终理论本体；高激活、高投影、高贡献、因果必要和机制核心必须分开；最小分布式因果集合比预设单概念神经元更符合当前证据；闭合必须保存为向量，不能压成一个总分。

但两个文件合并后不能直接执行“大工具栈 + HCPE 全流程”，原因如下：

1. Phase335 仍建议从 Phase330 的五个集合级读出候选继续递归下钻，但 Phase331-334 已经证明它们没有通过扩大留出行为必要性，也没有通过自然必要性，不能再作为已确认起点。
2. `TransformerLens + NNsight + pyvene + Circuit Tracer + SAELens + MIB` 不是可直接拼接的同一种坐标系。组件命名、Hook 位置、归一化时点、干预语义和模型支持范围不同。
3. HCPE 的递归二分隐含局部单调性；若两个子集单独无效、联合才有效，按“弱子集即剪枝”会删除真实交互集合。
4. Phase334 只有 24/54 单元达到基线资格。协议资格不修复，任何更细神经元搜索都会主要拟合“答案是否到达”和格式副作用。
5. SAE 或转码器特征是学习得到的特征坐标，不是真实单神经元地址，不能直接写入当前神经元图层并宣称完成物理定位。

### 二、外部工具的采用边界

当前仓库已经包含 TransformerLens 代码，同时大量研究脚本直接使用 Hugging Face 模块的 `register_forward_hook`。因此不应替换现有执行底座，先建立旁路对照。

官方资料显示，Circuit Tracer 已提供 Qwen3-4B 的预训练转码器，可以作为 Qwen3 候选电路发现器；其 NNsight 后端可覆盖更多 Hugging Face 模型，但官方明确说明该后端仍属实验性，速度和内存效率较差。MIB 直接支持的标准模型、任务和本项目三模型并不相同，因此应借用其评测原则，而不是把 MIB 分数当成本项目机制结论。

工具优先级冻结为：

```text
第一：借用 MIB 的分集、忠实性和私有留出原则，不立即迁移代码；
第二：Circuit Tracer 仅对 Qwen3-4B 做旁路候选图试验；
第三：pyvene 只在统一干预适配器通过等价性测试后考虑；
第四：NNsight 只作为现有 Hook 无法覆盖架构时的备选；
第五：SAELens 推迟到有明确数据预算和特征重构误差门之后。
```

不一次安装全部工具，不修改主 Python 环境，不重写现有 Hook 脚本，不把外部图谱直接合并为机制闭合节点。

### 三、阶段一：Phase336 协议资格修复

本阶段只改案例契约和分析字段，不改模型 Hook，不改三维客户端，不做神经元扫描。

#### 最小修改

在新版本案例中增加四个互不替代的字段：

```text
answer_reached：生成是否进入可判定答案区；
semantic_correct：完整生成在语义上是否正确；
target_phrase_valid：注册目标短语是否适合该模型和接口；
baseline_capability：该案例是否允许进入必要性计算。
```

九族分类保持不变，另建机制规则契约：

$$
RuleContract_r=
[Trigger,StatePre,Operation,ExpectedPath,Competitor,Readout,Rollout]
$$

第一试点机制使用 Phase334 基线覆盖最高的 `material`，但必须改名并限定为 `material_relation_binding`，避免把显式事实绑定误写成参数知识来源。

#### 数据与门

三个模型仍按 Qwen3、GLM4、DS7B 顺序测试。先使用每模型、每接口 12 个对象，只运行自然生成和完整短语评分，不做内部干预。

只有至少两个模型在同一接口达到预注册的 capable 门，且错误主要不是 `answer_reached=false`，才进入阶段二。若失败，只修一次协议；第二次仍失败则停止该机制，不用提示补丁反复追高基线。

#### 验收产物

```text
一个规则契约文件；
一份协议资格报告；
一个固定案例清单；
不产生机制候选节点；
不改变旧图谱证据等级。
```

### 四、阶段二：Phase337 外部工具旁路等价性

本阶段是工程兼容性试验，不是新的语言机制结论。

#### 最小范围

只对 Qwen3-4B、`material_relation_binding` 和阶段一冻结的少量 discovery 案例启用 Circuit Tracer。外部依赖放入独立虚拟环境，结果写入独立目录，不加入主 requirements，不改三模型执行脚本。

#### 必须比较

```text
同一提示的词元序列；
基线目标短语边距；
层和位置坐标映射；
一次块级删除的目标边距方向；
显存峰值、运行时间和失败案例；
外部特征图与现有组件路径是否只在候选层面相交。
```

只有基线输出和同语义干预在预注册容差内一致，才允许编写统一 `InterventionBackend` 适配器。若不一致，保留 Circuit Tracer 为独立观察工具，不进入正式图谱。

Circuit Tracer 的转码器节点必须标为 `learned_feature`，与 `physical_neuron`、`attention_head`、`mlp_channel` 分层存储。

### 五、阶段三：Phase338 块级因果筛选

只有 Phase336 资格门通过后才运行；Phase337 是否成功不构成必要前提，正式结论仍可使用现有本地 Hook。

#### 搜索单位

不按 Top-K 激活筛选。先冻结如下粗粒度组合：

```text
组件：attention_output、mlp_output、residual_increment；
深度：early、middle、late；
位置：source、query、answer_start；
共 27 个物理块。
```

每个块执行 `zero`、`half`、`mean replacement`，并与随机同规模、错误深度和错误位置比较。主读数使用完整短语竞争边距和自然生成行为；首词元排名只作诊断。

$$
M(x)=MeanLogP(y^+|x)-\max_{y^-\in Y^-}MeanLogP(y^-|x)
$$

发现、校准、留出分开冻结。三个模型按顺序执行和释放显存。只有正确块在留出集造成稳定损失、明显强于全部配平对照、且至少一个自然生成指标同步变化，才进入阶段四。

如果 27 个块全部失败，应停止神经元下钻，转向任务变量或时间位置错误审计；不能提高干预强度直到出现正结果。

### 六、阶段四：Phase339 最小交互集合提取

本阶段只处理 Phase338 通过的少量物理块，不处理 Phase330 的五个旧候选，也不全模型全层扫描。

#### 改进后的 HCPE

每个父集合 $U$ 使用至少两种固定平衡划分，并始终同时测试左右子集和联合集合：

$$
I(U_L,U_R)=\Delta M_{U_L\cup U_R}-\Delta M_{U_L}-\Delta M_{U_R}
$$

若左右单独弱但联合强，不剪枝，登记为不可拆交互集合。只有作用在两种划分、留出对象和配平对照上稳定时才继续缩小。终点可以是单神经元，也可以是最小神经元集合；不以“必须找到单神经元”为成功条件。

必要性成立后才做损坏后恢复：

$$
Nec(U)=M_{base}-M_{remove(U)}
$$

$$
Suf(U)=M_{restore(U)}-M_{corrupt}
$$

只有恢复下游集合能够中介上游删除造成的损失，才登记有方向的因果边。分母接近零的中介比例不计算，直接标记不可判定，避免产生虚高比值。

### 七、阶段五：Phase340 图谱与三维客户端增量同步

客户端最后修改，不在算法尚未通过时先做大规模界面工程。

#### 只增加一个新焦点层

在原有层叠网络形状上增加“因果集合”焦点，不重做场景。节点证据等级分为：

```text
L1：外部或自然观察候选；
L2：块级受控干预候选；
L3：留出必要性；
L4：必要性 + 充分性 + 中介；
L5：跨模型功能顺序同构。
```

默认只显示当前机制有关的块、集合和因果边；其余神经元不记录、不渲染。SAE 或转码器特征使用不同节点形状，并明确显示重构误差和来源工具，不能伪装成真实神经元。

每一阶段只发布本阶段新增文件，不迁移历史大数据。旧图谱继续可读，清单增加可选字段，客户端对缺失字段保持兼容。

### 八、最终扩展规则

只有一个机制完成 Phase336-340 后，才扩展到第二机制；只有两个不同模式族各有一个局部闭合规则后，才讨论操作基元复用；只有至少两个模型出现必要性、充分性和因果顺序同构后，才讨论语言族级机制。

推荐扩展顺序：

```text
material_relation_binding：协议最简单、现有基线最好；
missing_condition_control：验证推理约束；
past_tense：验证语法并复核 Phase334 的 GLM4 传播迹象；
plural_agreement：必须等完整句协议修复后再启动。
```

### 九、停止与回滚原则

```text
Phase336 失败：只保留协议报告，不进入内部干预；
Phase337 失败：删除隔离环境即可，现有研究底座不受影响；
Phase338 失败：停止神经元搜索，不修改客户端；
Phase339 失败：保留块级节点，不宣称最小集合；
Phase340 失败：回退新增焦点，固定格式研究数据仍有效。
```

每个阶段只改变一个主变量，并且有独立验收和退出点。这比一次性引入所有工具、重写测试框架、扩大三模型扫描和改造三维客户端更符合当前证据状态。

### 十、当前结论边界

本阶段只完成方案审计，没有运行 CUDA，没有新增模型结果，没有安装外部工具，也没有修改客户端。Phase335 的 HCPE 仍是待验证算法；Phase336-340 是把它变成可逐步证伪的工程路线。

现阶段最应该自动执行的下一步仅为 Phase336 协议资格修复。Phase337 需要独立环境和外部模型特征文件，Phase338 以后必须由上一阶段证据门触发，不应一次性连续启动。

## Phase 337: 显式关系绑定协议资格修复与三模型基线门 [2026-07-10 20:35]

### 一、阶段目标与编号校准

Phase336 已作为分阶段工程规划存在，因此本次实际执行登记为 Phase337。它只回答一个问题：在进入外部电路工具或内部块级干预前，哪些模型接口能够稳定完成冻结的语言任务，旧资格门的失败有多少来自答案未到达而不是语义能力缺失。

本阶段没有采集内部激活，没有执行消融、替换或注入，没有改三维客户端，也没有发布机制或神经元节点。

### 二、规则契约

试点机制从旧 `material` 收紧为：

```text
material_relation_binding（材料关系绑定）
```

它只测试提示中显式给出的对象-材料关系，不测试参数知识来源。规则契约为：

$$
RuleContract=
[Trigger,StatePre,Operation,ExpectedPath,Competitor,Readout,Rollout]
$$

本轮各项具体定义：

```text
Trigger：上下文同时给出对象材料和对象属性，查询材料；
StatePre：目标材料出现在源上下文，查询明确指出对象；
Operation：读取关系、绑定对象和值、路由到答案；
ExpectedPath：source -> query -> answer_start，仅作为后续假设；
Competitor：已陈述属性、unknown；
Readout：注册材料短语；
Rollout：自然生成到达答案，并在答案首行给出材料。
```

### 三、资格算法修正

Phase334 的资格门为：

$$
Eligible_{old}=BehaviorSuccess\land Rank_{first}\leq 50\land FiniteLogP
$$

它把自然语义、答案到达、目标词元切分和首词元排名混在一起。Phase337 拆成四个观测：

```text
answer_reached：模型是否进入可判定答案阶段；
semantic_correct：整个生成中是否已经形成正确材料；
target_phrase_valid：注册目标和竞争短语能否稳定分词计分；
answer_head_semantic_correct：答案第一个非空行是否含正确材料。
```

正式能力定义为：

$$
BaselineCapability=
AnswerReached\land AnswerHeadCorrect\land TargetPhraseValid
$$

首词元排名、格式是否只含短答案、思考是否耗尽词元预算只作诊断，不再决定能力资格。

之所以要求答案首行正确，而不是只要求完整生成中出现目标，是为了避免把后文复述上下文或先答错后纠正误计为基线能力。

### 四、冻结分母与 CUDA 执行

只使用 `template_a`，避免同时增加模板变量。冻结 12 个对象、3 个接口、3 个模型：

```text
12 对象 × 3 接口 × 3 模型 = 108 案例；
discovery：54；
calibration：27；
heldout：27；
每个模型-接口单元：12 案例；
最大自然生成：128 词元；
内部干预允许：false；
后验选择更新：false。
```

三个模型严格按 Qwen3、GLM4、DeepSeek7B 顺序加载、执行和释放显存，分别完成 36/36，总计 108/108；无效指标 0，执行结束后 GPU 计算进程 0。

### 五、客观结果

严格答案首行资格结果：

| 模型 | 原始补全 | 原生聊天 | 答案对齐聊天 |
|---|---:|---:|---:|
| Qwen3 | 12/12 | 2/12 | 12/12 |
| GLM4 | 12/12 | 12/12 | 12/12 |
| DeepSeek7B | 11/12 | 0/12 | 12/12 |

```text
合格模型-接口单元：7/9；
通过三模型共同接口：raw_completion、answer_aligned_chat；
下一阶段首选接口：answer_aligned_chat；
协议资格门：通过；
机制因果结论：0；
单神经元因果结论：0。
```

答案对齐接口三模型均为 12/12，因此优于原始补全。原始补全中 DeepSeek7B 唯一失败案例先在首行生成 `maples`，随后才纠正为 `maple`，严格门将其保留为失败。

### 六、最重要的协议现象

Qwen3 和 DeepSeek7B 的原生聊天不是关系绑定失败：

```text
Qwen3 native_chat：全文语义正确 12/12，答案到达 2/12；
DeepSeek7B native_chat：全文语义正确 12/12，答案到达 1/12，首行答案正确 0/12；
Qwen3 有 10/12 在 128 词元仍未结束思考；
DeepSeek7B 有 11/12 在 128 词元仍未结束思考。
```

失败生成中已经明确推导出正确材料，但没有在冻结预算内关闭思考并输出最终答案。因此 Phase334 把这些案例直接计为行为失败，会混入接口执行状态。

这项结果支持：

$$
SemanticState\neq AnswerArrival\neq ProtocolCompletion
$$

但它不证明三者在网络内部由独立电路实现，只证明评测层必须分开记录。

### 七、读出诊断边界

答案对齐接口上 Qwen3 的平均初始目标排名很差，但完整目标短语相对两个注册竞争短语的边距为正，且自然答案首行 12/12 正确。这再次说明不同接口的第一词元切分和前导空格会严重影响单一排名指标。

因此下一阶段不得按初始目标排名选择接口或组件；答案对齐接口被选中，是因为它在三个模型、所有 12 个对象和三个冻结分集上均达到完整行为资格。

### 八、问题与硬伤

1. 当前只有一种显式关系绑定任务，不能推广到知识网络、推理或语法。
2. 每个接口只有一个模板；这适合资格校准，但尚未验证模板鲁棒性。
3. `semantic_correct` 仍是字符串别名匹配，不是通用语义判分器；本任务目标是一词材料，因此暂时可控。
4. 128 词元仍不足以让多数 Qwen3、DeepSeek7B 原生聊天结束思考；继续加预算只能研究接口时长，不能自动改善机制定位。
5. 答案对齐接口人为跳过思考段，适合建立共同可审计分母，但不代表它等价于模型完整自然聊天路径。
6. 目标短语竞争集合只有属性和 unknown，尚未覆盖复述、解释、继续和停止竞争。
7. 三个小模型的高基线只说明简单显式绑定任务可完成，不代表真实知识检索机制已被破解。

### 九、阶段门与下一步

Phase337 的资格门已通过，因此允许进入下一个独立阶段，但不自动进入神经元递归拆分。

下一阶段应执行隔离的外部工具旁路兼容性试验：

```text
Phase338：Qwen3 Circuit Tracer 旁路等价性

1. 使用独立虚拟环境，不修改主依赖；
2. 只读 Phase337 discovery 案例和答案对齐接口；
3. 比较词元序列、目标短语边距、层位置映射和一个块级删除方向；
4. 转码器节点标为 learned_feature，不写成 physical_neuron；
5. 若与现有 Hook 不等价，停止工具集成，Phase337 结果仍保留；
6. 不在该阶段生成语言机制、行为闭合或单神经元结论。
```

Phase338 涉及独立环境和外部转码器资源，属于新的工程变量。按照分阶段修改原则，本轮在 Phase337 门后停止，不同时安装工具、执行块级干预和修改客户端。

### 十、主要产物

```text
tests/gpt5/phase337_protocol_qualification_case_bank.py；
tests/gpt5/phase337_protocol_qualification.py；
tests/gpt5/phase337_protocol_qualification_analysis.py；
tests/gpt5/run_phase337_protocol_qualification.sh；
tests/gpt5/test_phase337_protocol_qualification.py；
tests/gpt5/result/phase337_protocol_qualification/material_relation_binding/phase337_rule_contract.json；
tests/gpt5/result/phase337_protocol_qualification/material_relation_binding/phase337_registered_protocol.json；
tests/gpt5/result/phase337_protocol_qualification/material_relation_binding/phase337_registered_cases.jsonl；
tests/gpt5/result/phase337_protocol_qualification/material_relation_binding/phase337_qualified_rows.jsonl；
tests/gpt5/result/phase337_protocol_qualification/material_relation_binding/phase337_cell_summary.jsonl；
tests/gpt5/result/phase337_protocol_qualification/material_relation_binding/phase337_interface_gate_summary.jsonl；
tests/gpt5/result/phase337_protocol_qualification/material_relation_binding/phase337_global_summary.json；
tests/gpt5/result/phase337_protocol_qualification/material_relation_binding/phase337_report.md。
```

### 十一、通俗总结

这一步没有去找神经元，而是先检查尺子是否可靠。旧尺子会把“模型还在思考”“目标词第一个词元排名不高”和“模型真的不知道答案”混成同一种失败。

新结果表明：Qwen3 和 DeepSeek7B 在原生聊天中几乎都已经在思考文本里找到了正确材料，只是 128 个词元内没有结束思考；把答案起点对齐后，三个模型都是 12/12 正确。因此后续可以在同一个稳定接口上比较内部路径，而不必用提示补丁反复修复基线。

这只是把实验分母修好了，尚未找到任何语言电路。下一步应先验证外部电路工具与现有 Hook 是否测量同一个物理位置；只有工具等价，才值得继续块级因果筛选。

## Phase 338: 材料关系绑定的三模型分层粗块因果筛选 [2026-07-10 21:52]

### 一、对外部审计文本的判断

审计文本对 Phase334-337 的主结论基本正确：最大自然差分不等于自然必要性；九族是基准分类而不是理论本体；Phase337 只修复了可审计分母；下一科学对象应从单组件转向粗块和最小交互集合。

但必须收紧五点：

1. “九族图谱较完整”只能指登记和描述性观测，不能指自然必要性或因果图谱完整。
2. 72 项清单混合了工程框架、测量坐标、负结果和机制主张，不能将其统一计作 72 个已完成机制拼图。
3. 工程基础设施 96%-98%、描述图谱 85%-92%、总体 24%-28% 都没有统一可复算分母，本阶段不接受这些百分比。
4. Phase337 只验证了 `material_relation_binding`，不能反推 Phase334 其余五个机制已经获得合格协议分母。
5. 2 个对象乘 3 模板得到的 6 个 private-heldout 记录，只是密封式小留出，不是真正足量、外部不可见的私有基准。

外部审计建议把 Circuit Tracer 作为 Qwen3 旁路，同时以本地 Hook 完成正式三模型块筛选。考虑到不应同时改变工具底座和因果算法，本阶段优先完成本地块筛选；外部工具集成继续保留为独立工程分支。

### 二、为何采用分阶段计算

直接执行：

```text
24 对象 × 3 模板 × 3 模型 × 27 块 × 全部干预条件 × 自然生成
```

会把候选发现、阈值校准和留出验证混在一起，也会产生大量无必要生成。因此 Phase338 冻结为三级：

```text
发现：27 块全部置零，只测完整短语竞争边距；
校准：每模型只保留发现前三块，测试置零、半缩放和同块置换；
留出：每模型只保留一个冻结块，增加错误深度、错误位置和自然生成；
私有留出：使用同一冻结块和阈值，不允许更新选择。
```

### 三、冻结分母

使用 24 个未出现在 Phase334/337 中的新对象、材料和属性，三个模板、答案对齐接口和三个模型：

```text
注册案例：216；
每模型：72；
discovery：108，即每模型 36；
calibration：54，即每模型 18；
heldout：36，即每模型 12；
private_heldout：18，即每模型 6；
注册粗块：27；
单神经元干预允许：false。
```

27 个粗块为：

$$
3\ component\ types\times3\ depth\ bins\times3\ position\ roles=27
$$

其中一个粗块覆盖某个组件在一个深度三分区内的全部层，并只作用于 source、query 或 answer_start 中的一个位置。

### 四、块干预与读数

对注意力或 MLP 输出，置零和半缩放为：

$$
v'_{c,l,p}=\alpha v_{c,l,p},\qquad \alpha\in\{0,0.5\}
$$

对残差增量：

$$
h'_{l+1,p}=h_{l,p}+\alpha(h_{l+1,p}-h_{l,p})
$$

同块置换保持向量数值集合与粗略尺度，但打乱坐标结构：

$$
v'_{c,l,p}=\Pi(v_{c,l,p})
$$

主读数不使用第一词元排名，而使用目标短语和最强注册竞争短语的平均词元对数概率边距：

$$
M(x)=MeanLogP(y^+|x)-\max_{y^-\in Y^-}MeanLogP(y^-|x)
$$

块损失为：

$$
L_B=M_{base}-M_{do(B)}
$$

留出门要求基线首行正确、正确块短语损失和正向案例率达门、自然生成实际失败，并且错误深度和错误位置的短语及行为损伤明显更低。

### 五、控制角色的关键校正

初版统一分析一度把同块置换与错误深度、错误位置一起作为低副作用空白对照。这会错误否定真实结构化块：如果块内容的坐标关系重要，置换同一块导致失败本来就是预期现象。

因此正式定义改为：

```text
wrong_depth_zero、wrong_position_zero：位置特异性和低副作用控制；
correct_permutation：同块内容结构敏感性，不是空白位置控制；
attribute binding：真实关系绑定机制，不是 null control。
```

该校正改变了 Qwen3 的冻结候选，因而只重跑受影响的 Qwen3 留出；GLM4 和 DeepSeek7B 的冻结块没有变化。门定义和重跑过程均保留在固定结果中。

### 六、执行规模与质量

三个模型严格按 Qwen3、GLM4、DeepSeek7B 顺序执行和释放显存。

```text
发现块汇总：81，即 27 × 3 模型；
发现条件记录：3024，即 1008 × 3；
校准块汇总：9，即 3 × 3 模型；
校准条件记录：540，即 180 × 3；
留出与私有留出条件记录：324，即 108 × 3；
总分阶段条件记录：3888；
全部阶段完成有效：true；
无 GPU 残留进程。
```

发现门较宽，三模型分别有 19、16、17 个块通过，共 52/81。这不是 52 个机制，而是说明整段深度区间置零会造成大范围读出变化。

校准后，三模型的前三块都表现为置零和置换损伤，共 9/9 通过结构敏感性门；每模型只冻结联合结构分数最高的一块进入留出。

### 七、冻结块

```text
Qwen3：residual_increment__early__source；
GLM4：mlp_output__early__source；
DeepSeek7B：residual_increment__early__source。
```

三个模型都定位到早期 source 区域，Qwen3/DeepSeek7B 落在残差增量，GLM4 落在 MLP 输出。这是功能位置上的弱收敛，但组件类型并不相同，且只有严格留出通过后才可升级。

### 八、留出客观结果

#### Qwen3

```text
heldout 正确置零：短语损失 17.695，行为损失 0.75；
heldout 错误深度：短语损失 6.540，行为损失 0.25；
private 正确置零：短语损失 23.874，行为损失 1.00；
private 错误深度：短语损失 7.140，行为损失 0；
完整模型门：失败。
```

Qwen3 在小私有留出上表现特异，但正式留出中错误深度仍造成 25% 行为失败，超过低副作用上限。

#### GLM4

```text
heldout 正确置零：短语损失 7.169，行为损失 1.00；
heldout 错误深度：短语损失 1.145，行为损失 0；
heldout 错误位置：短语损失 0.414，行为损失 0；
private 正确置零：短语损失 6.981，行为损失 1.00；
private 错误深度/位置：行为损失均为 0；
完整模型门：通过。
```

GLM4 的早期 source MLP 深度块可以登记为模型特异粗块候选。置换同块在 heldout/private 的行为损失为 0.917/1.00，说明该区域不仅需要非零写入，还依赖内部坐标结构。

#### DeepSeek7B

```text
heldout 正确置零：短语损失 10.084，行为损失 1.00；
heldout 错误深度：短语损失 2.475，行为损失 0.333；
private 正确置零：短语损失 9.819，行为损失 1.00；
private 错误深度：短语损失 4.055，行为损失 0；
完整模型门：失败。
```

DeepSeek7B 与 Qwen3 一样，在正式留出中未排除深度区域的广泛损伤。

### 九、严格总结果

```text
发现门块：52/81；
校准结构敏感块：9/9 被审计块；
heldout 模型门：1/3；
private 模型门：3/3；
heldout + private 完整模型门：1/3；
跨模型粗块门：0/1；
最小因果集合入口：关闭；
行为机制闭合：0/72；
单神经元因果闭合：0/72。
```

私有门 3/3 而正式留出只有 1/3，说明 2 个私有对象不足以稳定估计错误深度副作用，不能用它覆盖较大的留出结果。

### 十、固定图谱数据与客户端边界

本阶段生成 81 个固定格式粗块节点，其中：

```text
模型特异 L4 粗块候选：1；
跨模型因果块：0；
单神经元因果节点：0。
```

这些节点保存在独立结果目录，带有 discovery、calibration、selected、local gate 和 cross-model gate 字段。由于跨模型门失败，客户端主清单继续保持 Phase334，不把 Phase338 观测块绘制成已确认语言路径。

### 十一、进展向量

```text
九族工程登记覆盖：9/9；
72 机制名录普查：72/72；
协议资格试点：1 个机制；
粗块同等级深审：1/72；
跨模型粗块候选：0/1；
最小因果集合：0/72；
行为机制闭合：0/72；
单神经元因果闭合：0/72；
训练形成轨迹：0/3 模型。
```

不将这些不同分母平均为总体百分比，也不采用外部文本中的 24%-28%。

### 十二、问题、硬伤和解释边界

1. 一个粗块覆盖约三分之一网络深度的全部层，干预非常宽，容易造成一般计算损伤。
2. 错误深度只是相邻深度三分区，不能完全配平不同层数和组件范数。
3. 当前只用错误深度、错误位置控制，没有验证 GLM4 候选对一般复制、属性关系、推理和语法任务的副作用。
4. 半缩放在三个冻结块上几乎不改变自然行为，而置零和置换很强，提示非线性阈值或结构完整性，但也可能来自粗块破坏。
5. 同块置换使用确定性坐标滚动，不等于多种随机同矩变换。
6. private-heldout 只有 2 个对象和 3 模板，不能独立支撑强复现。
7. 三模型选择同为早期 source，但组件不同，尚不能称为功能同构路径。
8. 小模型可能使用更粗糙、更冗余或更易受协议影响的路径，不能直接外推大模型。
9. 本阶段没有测试均值替换、充分性恢复、上游中介或补偿接管。
10. 没有训练检查点，仍不能研究关系绑定路径如何形成。

### 十三、理论进展

本阶段不支持“某个最大块就是语言规则”。更严格的客观现象是：

```text
早期 source 区域对显式关系绑定具有高敏感性；
GLM4 的早期 source MLP 区域出现位置特异的模型内必要性候选；
Qwen3/DeepSeek7B 的粗块效应仍混有错误深度副作用；
三模型尚未形成同等级块级因果同构。
```

这与动态模式网络假设相容，但不能验证该理论。当前公式仍只是测量框架：

$$
h_{l+1,t}=h_{l,t}+A_{l,t}(H,I,T)+M_{l,t}(h,I,T)+C_{l,t}(H,q,I)+\varepsilon
$$

Phase338 只给出某些大区域删除后的结果，没有识别 $A$、$M$、$C$ 的最小相互作用规律。

### 十四、是否自动继续

按预注册规则，至少两个模型通过完整粗块门，才能进入递归最小集合。当前只有 GLM4 通过，故：

```text
不自动递归二分；
不启动全量单神经元 CUDA 扫描；
不执行均值替换、充分性恢复和中介链；
不更新客户端因果图层；
不把早期 source 收敛写成跨模型机制。
```

下一阶段若启动，应是新的 Phase339 模型特异边界审计，而不是最小集合提取：

```text
1. 只审计 GLM4 早期 source MLP 粗块，同时保留 Qwen3/DS7B 作为同协议参照；
2. 增加 attribute relation binding、identity copy、非关系语法和简单推理任务；
3. 判断该块是材料专用、关系绑定复用、一般 source 处理，还是广泛损伤；
4. 只有跨任务低副作用成立，才把早期深度三分区缩小到层组；
5. 若一般任务同样失败，关闭该候选，不做神经元下钻。
```

Phase339 会同时改变任务类型和候选作用域，属于新的阶段。Phase338 已达到阶段目标并触发停止门，本轮不后验追加。

### 十五、主要产物

```text
tests/gpt5/phase338_block_causal_case_bank.py；
tests/gpt5/phase338_block_causal_screen.py；
tests/gpt5/phase338_block_causal_analysis.py；
tests/gpt5/run_phase338_block_causal_screen.sh；
tests/gpt5/test_phase338_block_causal_screen.py；
tests/gpt5/result/phase338_block_causal_screen/material_relation_block_screen/phase338_registered_protocol.json；
tests/gpt5/result/phase338_block_causal_screen/material_relation_block_screen/phase338_registered_cases.jsonl；
tests/gpt5/result/phase338_block_causal_screen/material_relation_block_screen/phase338_registered_blocks.jsonl；
tests/gpt5/result/phase338_block_causal_screen/material_relation_block_screen/phase338_physical_block_nodes.jsonl；
tests/gpt5/result/phase338_block_causal_screen/material_relation_block_screen/phase338_model_gate_summary.jsonl；
tests/gpt5/result/phase338_block_causal_screen/material_relation_block_screen/phase338_claim_registry.jsonl；
tests/gpt5/result/phase338_block_causal_screen/material_relation_block_screen/phase338_global_summary.json；
tests/gpt5/result/phase338_block_causal_screen/material_relation_block_screen/phase338_report.md。
```

### 十六、通俗总结

这次没有从最高值神经元开始，而是先把网络按“组件、深度区间、词元位置”切成 27 个大块。大多数大块置零都会让答案概率下降，所以发现阶段看起来有很多候选；加入置换和新对象后，每个模型只留下一个最强块。

三个模型都指向较早的 source 区域，这是值得保留的路标。但只有 GLM4 的早期 source MLP 大块在新对象上表现为“删正确块答案全错，删错误深度和位置答案不受影响”。Qwen3 和 DeepSeek7B 删除错误深度也会损伤答案，说明块太粗或模型路径更分散。

因此现在不能进入神经元级拆分。最严谨的下一步，是先用其他任务检查 GLM4 这块到底负责关系绑定，还是负责所有 source 信息。只有它对关系任务重要、对无关任务副作用低，才值得继续缩小到具体层组和神经元集合。

## Phase 339: 冻结早期来源粗块的九任务功能边界审计 [2026-07-10 22:42]

### 一、对输入分析的审计

Phase338 后续分析中，以下判断正确：

1. 搜索单位应从单组件提升到“组件类型 × 深度区间 × 位置角色”的粗物理块；
2. GLM4 的早期来源 MLP 粗块必须先做跨任务副作用审计；
3. Qwen3 和 DeepSeek7B 在 Phase338 没有通过完整门，只能作为冻结参照；
4. 九任务、十八题项、三模板、三模型形成 1458 个提示-模型案例，分母计算正确；
5. 未证明任务选择性之前，不允许层组、通道或单神经元下钻。

需要收紧的部分：

1. 不能把工程框架、脚本字段和机制证据混合计成大量“核心拼图”；
2. 不能在同一阶段同时加入跨任务审计、均值替换、多置换、全词表散度、层组收缩和通道交互；
3. 不能给出缺少固定分母的 18%-22% 总体进度；
4. 同块置换测量的是块内结构敏感性，不是错误位置空对照；
5. 线性差分只能作为读数，不能充当真实前向机制公式。

因此本阶段只执行九任务边界审计。均值替换、全词表散度、层组收缩和通道交互继续关闭。

### 二、固定分母与算法

任务矩阵包括：

```text
关系绑定：材料、属性、部件、位置；
来源操作：身份复制、来源片段抽取；
跨族控制：单数一致、直接蕴含、仅答案协议。
```

每任务十八个新题项、三模板、三模型，共：

$$
N_{339}=9\times18\times3\times3=1458
$$

冻结 Phase338 的三个模型粗块，不重新选择。词组条件为：

```text
基线；正确块置零；正确块半缩放；正确块置换；
错误深度置零；错误位置置零。
```

行为生成只在公开留出和私有留出执行：

```text
基线；正确块置零；错误深度置零；错误位置置零。
```

对模型 $m$、任务 $t$、划分 $s$，先定义分母资格：

$$
E_{m,t,s}
=
\mathbf{1}[Acc^{base}_{m,t,s}\ge 0.8]
\land
\mathbf{1}[FiniteRate_{m,t,s}=1]
$$

任务因果门为：

$$
G_{m,t,s}
=E_{m,t,s}
\land
\mathbf{1}[Loss^{behavior}_{correct}\ge0.5]
\land
\mathbf{1}[Loss^{behavior}_{control}\le0.1]
\land
\mathbf{1}[Loss^{phrase}_{correct}-Loss^{phrase}_{control}\ge0.05]
$$

公开留出和私有留出必须同时通过：

$$
G_{m,t}=G_{m,t,heldout}\land G_{m,t,private}
$$

### 三、执行规模与数据完整性

```text
注册案例：1458；
词组条件行：8748；
单样本/批量行为行：1620；
模型 × 任务单元：27；
完整执行：3/3；
通过完整分母资格：15/27；
通过完整任务门：3/27。
```

GLM4 有 53 条非有限词组评分，来自 12 个来源片段抽取案例，全部集中在模板 A：

```text
发现：27 行；
校准：13 行；
公开留出：11 行；
私有留出：2 行。
```

这些值没有改成零，也没有被均值忽略后继续过门，而是写成 `null` 并将对应划分判为分母不合格。

### 四、客观结果

Qwen3：

```text
Phase338 原候选资格：失败；
合格任务分母：7/9；
通过任务：材料关系、仅答案协议；
证据身份：冻结块描述性现象，不能升级候选。
```

GLM4：

```text
Phase338 原候选资格：通过；
合格任务分母：0/9；
通过任务：0/9；
状态：跨任务范围不可判定，不是候选被否定。
```

DeepSeek7B：

```text
Phase338 原候选资格：失败；
合格任务分母：8/9；
通过任务：部件关系；
证据身份：冻结块描述性现象，不能升级候选。
```

### 五、结论与硬伤

Phase339 没有证明 GLM4 粗块是材料模块、关系绑定模块或一般来源模块。主要原因不是因果门本身，而是 GLM4 新任务分母没有获得基线资格。

硬伤包括：

1. 模板 A 在 GLM4 上频繁产生连续感叹号；
2. GLM4 来源片段抽取出现非有限词组分数；
3. Qwen3 和 DeepSeek7B 的局部通过项不能覆盖 Phase338 原门失败；
4. 九任务虽覆盖功能类别，但身份复制和仅答案协议仍可能共享相同的显式词元复制结构；
5. 粗块置零仍可能造成阈值越界或广泛数值损伤。

因此：

```text
层组收缩门：关闭；
通道集合门：关闭；
单神经元 CUDA 门：关闭；
行为机制闭合：0/72；
单神经元因果闭合：0/72。
```

### 六、阶段产物

```text
tests/gpt5/phase339_cross_task_boundary_case_bank.py；
tests/gpt5/phase339_cross_task_boundary_audit.py；
tests/gpt5/phase339_cross_task_boundary_analysis.py；
tests/gpt5/run_phase339_cross_task_boundary.sh；
tests/gpt5/test_phase339_cross_task_boundary.py；
tests/gpt5/result/phase339_cross_task_boundary/early_source_cross_task_boundary/。
```

## Phase 340: 全新跨任务协议修复与 GLM4 批处理不变性校准 [2026-07-10 22:56]

### 一、阶段目的

Phase339 无法回答 GLM4 候选范围，直接继续干预会把协议故障当成机制负结果。本阶段只修复和验证基线协议，不进行任何内部干预。

使用九任务、十八个全新题项、两套在上一阶段相对稳定的答案对齐模板和三个模型：

$$
N_{340}=9\times18\times2\times3=972
$$

每个模型任务必须在发现、校准、公开留出、私有留出四个划分同时达到：

$$
Acc^{base}\ge0.8,
\qquad
FiniteRate=1
$$

### 二、首次批量结果

三模型共生成：

```text
基线词组行：972；
基线行为行：972；
非有限词组行：0；
完整执行：3/3。
```

但 GLM4 批量 6 生成时再次出现连续感叹号。若直接采用该结果，GLM4 为 0/9 合格任务。

### 三、批处理不变性诊断

对同一批 324 个 GLM4 案例逐条重新生成，不改变模型、提示、阈值或解码长度。

定义文本不变率：

$$
I_{text}
=
\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[y_i^{batch6}=y_i^{batch1}]
$$

定义正确性不变率：

$$
I_{correct}
=
\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[c_i^{batch6}=c_i^{batch1}]
$$

结果：

```text
批量 6 准确率：56.79012%；
批量 1 准确率：80.55556%；
文本不变率：70.06173%；
正确性不变率：76.23457%；
失败恢复：77 个；
成功转失败：0 个；
批处理不变门：失败。
```

这是重要的工具负结果。GLM4 的行为生成不能继续使用批量 6 作为语义判据。两套数据均保留，协议资格采用逐条生成结果，因为批量结果已被同输入复核证伪。

### 四、修复后资格结果

修复后共有 18/27 个模型任务单元通过四划分资格。每个模型通过 6/9。

GLM4 通过：

```text
材料关系；
部件关系；
位置关系；
身份复制；
直接蕴含；
仅答案协议。
```

GLM4 未通过：

```text
属性关系；
来源片段抽取；
单数一致。
```

因此已获得：

```text
目标任务：1；
关系近邻：2；
来源控制：1；
跨族控制：2。
```

Phase341 的新因果边界入口打开，但这只是 L2 基线协议资格，不是内部因果证据。

### 五、工程和理论校准

1. 高性能批处理不是天然语义等价，必须加入模型级批处理不变性测试；
2. 客户端连续感叹号不能直接解释成模型不会任务；
3. 模型行为判据和词组概率判据必须分别记录；
4. 后续 GLM4 关键因果行为统一采用单样本生成；
5. 本阶段没有增加任何语言机制闭合进度。

### 六、阶段产物

```text
tests/gpt5/phase340_cross_task_protocol_case_bank.py；
tests/gpt5/phase340_cross_task_protocol_qualification.py；
tests/gpt5/phase340_batch_invariance_diagnostic.py；
tests/gpt5/phase340_cross_task_protocol_analysis.py；
tests/gpt5/run_phase340_cross_task_protocol.sh；
tests/gpt5/test_phase340_cross_task_protocol.py；
tests/gpt5/result/phase340_cross_task_protocol/fresh_cross_task_protocol_repair/。
```

## Phase 341: 六任务单样本因果边界复核与材料候选重分类 [2026-07-10 23:05]

### 一、阶段设计

冻结 Phase340 中 GLM4 已通过基线资格的六个任务：

```text
材料关系；部件关系；位置关系；
身份复制；直接蕴含；仅答案协议。
```

三模型都使用 Phase338 已冻结的粗块，不重新选择。共有：

$$
N_{341}=6\times18\times2\times3=648
$$

词组条件为六种，行为条件为四种。所有行为干预均逐条生成，避免 Phase340 发现的 GLM4 批处理伪影。

### 二、执行规模

```text
注册案例：648；
词组条件行：3888；
单样本行为行：720；
非有限词组行：0；
完整模型执行：3/3；
模型 × 任务单元：18；
完整任务门通过：3/18。
```

### 三、模型结果

Qwen3：

```text
通过：仅答案协议；
材料关系：未通过；
Phase338 原候选：未通过；
身份：描述性冻结块效应，不能升级。
```

DeepSeek7B：

```text
通过：0/6；
Phase338 原候选：未通过；
身份：无可升级结果。
```

GLM4：

```text
通过：身份复制、仅答案协议；
未通过：材料关系、部件关系、位置关系、直接蕴含；
材料关系公开/私有正确块行为损失率：0/0；
材料关系公开/私有词组对照优势：-0.8191624/-2.4154837；
材料关系候选复现：失败。
```

GLM4 身份复制：

```text
公开/私有正确块行为损失率：1/1；
公开/私有错误位置或深度最大行为损失率：0/0；
公开/私有词组对照优势：7.2653592/8.8613458。
```

GLM4 仅答案协议：

```text
公开/私有正确块行为损失率：1/1；
公开/私有错误位置或深度最大行为损失率：0/0；
公开/私有词组对照优势：5.7366315/9.3830947。
```

半缩放几乎没有损失，而置零和置换损失很大：

```text
身份复制公开：半缩放 0.0578455，置零 11.7741435，置换 11.7101201；
身份复制私有：半缩放 -0.2151465，置零 14.1838887，置换 14.8917419；
仅答案公开：半缩放 0.0089783，置零 6.9802479，置换 9.6042983；
仅答案私有：半缩放 -0.1149854，置零 10.2946607，置换 11.0136032。
```

该现象更像非线性阈值或块内结构依赖，不能用简单线性剂量公式解释。

### 四、严格结论

Phase338 的 GLM4 早期来源 MLP 粗块不能继续称为材料关系绑定候选。它在全新合格分母上没有影响三类关系读取，却稳定影响两个高度相似的显式来源词元回送任务。

当前最窄、最谨慎的登记为：

```text
GLM4 模型特异的显式来源词元复制/回送粗块候选；
原材料关系候选被否定；
不是关系绑定复用模块；
不是跨模型统一机制；
不是单神经元机制。
```

身份复制和仅答案协议使用相似代码词及直接回送结构，因此二者不能被算成两个独立功能族的闭合证据。下一阶段必须用不重叠词表、不同词元长度、改写提示、非复制控制和参数知识任务复核。

### 五、图谱状态和停止门

固定图谱新增 18 个任务边界节点，保存在：

```text
tests/gpt5/result/phase341_fresh_causal_boundary/
qualified_six_task_causal_boundary/phase341_task_boundary_nodes.jsonl
```

状态为：

```text
九族登记覆盖：9/9；
机制目录覆盖：72/72；
深因果审计：1/72；
行为机制闭合：0/72；
单神经元因果闭合：0/72；
任务选择性层收缩门：0 个模型；
语言编码机制闭合：否；
智能理论实验闭合：否。
```

不报告单一总体百分比，因为“登记覆盖、粗块审计、行为闭合、神经元闭合”不是同一分母，强行加权会制造虚假进度。

### 六、小模型边界

三个模型规模较小，内部机制可能比大模型更粗糙、更分散或更脆弱，结论不能直接外推到完整语言编码。模型间组件类型不同也可能是真实架构差异。但本轮的协议失败、批处理伪影和跨模型不一致不能全部用“小模型偏差”解释，它们首先是当前证据不足。

### 七、智能理论更新

本阶段没有改名或重写智能理论。保留原主体：语言能力来自输入、任务、接口、位置、时间和模型条件共同形成的动态运行子图。

新增的客观约束只有两条：

1. 显式来源词元回送与关系读取可以在同一早期来源区域中分离；
2. 粗块效应可能呈“半缩放稳定、置零/置换崩溃”的非线性阈值结构。

这支持用条件化动态子图而不是固定线性方向描述运行过程，但还不足以给出新的统一数学定律。

### 八、下一大阶段

当前关系绑定阶段已经触发停止门，不自动执行层组或神经元下钻。下一阶段会改变目标机制，应单独冻结：

```text
显式复制：标签复制、任意符号回送、跨句指针回送；
非复制控制：关系读取、语法补全、直接推理；
词元控制：单词/多词元、常见/罕见、不同词表；
模板控制：至少三种改写和独立私有模板；
工具控制：GLM4 单样本生成及批处理不变性持续检查。
```

只有该候选在新任务、私有模板和低副作用门上复现，才允许对 GLM4 早期 MLP 做四层、两层、单层收缩。否则关闭该候选，回到九族物理分布图谱的下一未审计机制。

### 九、阶段产物

```text
tests/gpt5/phase341_fresh_causal_boundary_case_bank.py；
tests/gpt5/phase341_fresh_causal_boundary_audit.py；
tests/gpt5/phase341_fresh_causal_boundary_analysis.py；
tests/gpt5/run_phase341_fresh_causal_boundary.sh；
tests/gpt5/test_phase341_fresh_causal_boundary.py；
tests/gpt5/result/phase341_fresh_causal_boundary/qualified_six_task_causal_boundary/。
```

### 十、通俗总结

Phase338 看起来像找到了一块“材料关系区域”。这次把同一块放到更多合格任务里重新测试，并修复了 GLM4 批量生成会异常输出感叹号的问题。修复后，这块区域并不负责从句子里读出材料、部件或位置；它更像在帮助模型把提示中明确给出的一个词直接送到答案位置。

所以本轮真正的进展不是“找到了关系神经元”，而是排除了一个错误解释，并发现了一个更窄的复制候选。现在继续缩小关系块会走错方向，必须先独立验证这个复制候选是否真实、是否跨模板、是否只是某些代码词造成的。

## Phase 342: 复制候选前置执行不变性与正式测量路径冻结 [2026-07-11 00:00]

### 一、对 Phase339-341 系统审计的判断

附件对 Phase339-341 的主结论基本正确：

1. GLM4 早期来源 MLP 粗块不能继续解释为材料关系或一般关系绑定模块；
2. 身份复制和仅答案协议只能合并为一个显式来源词元传输候选；
3. GLM4 批处理异常是最高优先级工程风险；
4. 置零与单一置换不足以证明块内状态携带复制内容；
5. 没有跨模型复现、自然状态替换、充分性恢复或最小因果集合；
6. 未通过复制选择性之前，不能缩小层组或扫描神经元。

需要收紧的内容：

1. “88 项核心拼图”仍含人为归并，不能作为科学完成分母；
2. “总体约 20%、17%-23%”来自主观权重，不是当前证据可推导的客观进度；
3. 建议的 Phase342 同时混入执行不变性、十六任务、十种干预、内容替换和层组收缩，范围过大；
4. 对象级置信区间和随机置换零分布可以作为辅助，但不能替代逐对象、逐模板、公开/私有双留出的基础事实；
5. `CopyRelay` 仍是候选标签，不应提前写入统一理论主体。

因此本阶段只完成工作包零：自然执行不变性。没有运行任何激活干预。

### 二、固定案例与执行变量

从 Phase340 的九任务中按四个题项、两模板、三模型分层抽取：

$$
N_{342}=9\times4\times2\times3=216
$$

每模型 72 个自然案例，比较 11 种执行模式：

```text
批量：1、2、4、6；
缓存：关闭、开启；
填充：左填充，以及批量 2/4/6 的右填充对照。
```

参考路径冻结为：

```text
batch_size=1；
padding_side=left；
use_cache=false。
```

同时记录：

```text
完整生成文本；
语义正确性；
下一目标首词元逻辑值；
下一词元最高候选；
早期来源边界隐藏状态；
所有逻辑值和隐藏状态有限性。
```

执行资格门为：

$$
G_{exec}
=G_{finite}
\land G_{text}
\land G_{correct}
\land G_{top1}
\land G_{hidden}
\land G_{logit}
$$

固定工程阈值：

```text
文本不变率 >= 0.99；
正确性不变率 = 1；
最高词元不变率 = 1；
来源隐藏状态最小余弦 >= 0.999；
目标首词元最大逻辑值差 <= 0.05；
有限率 = 1。
```

这些是后续因果证据的执行资格阈值，不是语言规律。

### 三、执行规模

```text
注册自然案例：216；
执行模式：11；
固定结果行：2376；
三模型完整执行：3/3；
非有限结果行：108，全部来自 GLM4；
内部激活干预：0。
```

### 四、客观结果

三个模型只有以下两条路径通过完整门：

```text
b1_left_cache0；
b1_left_cache1。
```

因此缓存开关在单样本路径下没有改变结果，正式因果路径继续选择更简单的：

```text
b1_left_cache0。
```

Qwen3：

```text
批量 2/4/6 左填充的文本、正确性和最高词元均可保持；
但最大目标逻辑值差为 0.09375-0.109375，超过 0.05；
右填充批量 4 文本不变率 0.7361111；
右填充批量 6 文本不变率 0.5694444；
所有批量路径均不进入正式因果证据。
```

GLM4：

```text
批量 2 左填充有限率 1、文本不变率 1，但最大逻辑值差 0.0703125；
批量 4 左填充有限率 0.7222222；
批量 6 左填充有限率 0.5277778；
批量 4/6 在缓存开关下重复相同非有限模式；
累计非有限行：108；
右填充虽然前向有限，但批量 4/6 文本不变率仅 0.7222222/0.5277778。
```

DeepSeek7B：

```text
批量 2/4/6 左填充文本不变率 0.9722222/0.9305556/0.9861111；
来源隐藏状态最小余弦最低到 0.9924228；
目标首词元最大逻辑值差达到 0.75；
右填充批量 6 文本不变率 0.5138889；
所有批量路径均失败。
```

### 五、严格解释

该结果证明的是当前本地模型和执行后端的测量路径不具备普遍批处理不变性，不证明模型内部语言机制随“批量”变化。

可能来源包括：

```text
半精度矩阵乘法顺序；
批量形状对应的不同 CUDA 内核；
填充和位置处理；
模型远程实现；
接近决策边界的微小数值差；
GLM4 特有的批量数值失稳。
```

无论根因是哪一种，后续关键行为、完整短语和隐藏状态证据都必须使用单样本左填充路径。高性能并行只能用于已经通过不变性验证的非关键预处理，不能用于正式因果判据。

### 六、图谱和闭合状态

固定新增 33 个“模型 × 执行模式”测量节点。它们属于测量执行层，不属于语言神经机制。

```text
语言机制闭合新增：0；
单神经元因果新增：0；
复制候选任务矩阵入口：打开；
自然状态内容替换入口：尚未测试；
层组收缩入口：关闭。
```

### 七、阶段产物

```text
tests/gpt5/phase342_copy_relay_execution_case_bank.py；
tests/gpt5/phase342_copy_relay_execution_invariance.py；
tests/gpt5/phase342_copy_relay_execution_analysis.py；
tests/gpt5/run_phase342_copy_relay_execution.sh；
tests/gpt5/test_phase342_copy_relay_execution.py；
tests/gpt5/result/phase342_copy_relay_execution/copy_relay_execution_invariance/。
```

## Phase 343: 十六任务复制边界的全新基线资格图谱 [2026-07-11 00:20]

### 一、阶段目标

Phase342 只允许单样本正式路径。本阶段不做粗块干预，而是先建立复制、复制近邻和非复制控制的全新大样本分母。

任务矩阵：

```text
显式复制 6 类：随机标签、数字、任意符号、跨句指针、多词元短语、延迟复制；
复制近邻 3 类：键值读取、对象名称回送、指定字段抽取；
非复制控制 7 类：材料关系、属性关系、语义分类、单数一致、直接蕴含、词元转换、无来源答案。
```

每类 18 个全新题项、3 套模板、3 个模型：

$$
N_{343}=16\times18\times3\times3=2592
$$

发现、校准、公开留出、私有留出分别使用 9、4、3、2 个对象。每个模型任务必须在四个划分同时满足：

$$
Acc^{base}\ge0.8,
\qquad
FiniteRate=1
$$

### 二、执行规模

```text
注册案例：2592；
单样本词组行：2592；
单样本行为行：2592；
非有限词组行：0；
完整模型执行：3/3；
合格模型任务单元：38/48；
内部干预：0。
```

### 三、模型资格结果

Qwen3：13/16 合格。

```text
显式复制：6/6；
复制近邻：3/3；
非复制控制：4/7；
失败：属性关系、语义分类、无来源答案。
```

GLM4：13/16 合格。

```text
显式复制：6/6；
复制近邻：3/3；
非复制控制：4/7；
失败：属性关系、语义分类、无来源答案。
```

DeepSeek7B：12/16 合格。

```text
显式复制：6/6；
复制近邻：3/3；
非复制控制：3/7；
失败：属性关系、语义分类、词元转换、无来源答案。
```

GLM4 六类显式复制和三类复制近邻几乎全部划分达到 1.0。该结果只说明测试分母足够稳定，不能说明 Phase338 粗块参与这些任务。

### 四、因果入口

预注册入口要求：

```text
GLM4 显式复制合格 >= 3；
复制近邻合格 >= 1；
非复制控制合格 >= 4。
```

实际为：

```text
6、3、4。
```

因此复制粗块因果边界入口打开。Phase344 冻结 GLM4 合格的 13 类任务，并让 Qwen3、DeepSeek7B 使用同一矩阵作为功能参照。

### 五、图谱和理论边界

固定新增 48 个协议资格节点。当前仍没有：

```text
复制粗块必要性；
复制任务特异性；
内容可转移性；
层组定位；
单神经元因果。
```

该阶段只改善了操作边界图谱的可测量分母，没有提升 0/72 的行为机制闭合数。

### 六、阶段产物

```text
tests/gpt5/phase343_copy_boundary_protocol_case_bank.py；
tests/gpt5/phase343_copy_boundary_protocol_qualification.py；
tests/gpt5/phase343_copy_boundary_protocol_analysis.py；
tests/gpt5/run_phase343_copy_boundary_protocol.sh；
tests/gpt5/test_phase343_copy_boundary_protocol.py；
tests/gpt5/result/phase343_copy_boundary_protocol/copy_boundary_protocol_qualification/。
```

## Phase 344: 十三任务单样本粗块因果边界与一般复制候选关闭 [2026-07-11 00:40]

### 一、固定分母和算法

只使用 Phase343 中 GLM4 合格任务的公开和私有留出，不使用发现和校准对象：

```text
显式复制：6 类；
复制近邻：3 类；
非复制控制：材料关系、单数一致、直接蕴含、词元转换。
```

每类 5 个留出对象、3 模板、3 模型：

$$
N_{344}=13\times5\times3\times3=585
$$

Phase338 粗块完全冻结，不在复制任务上重新选择。每案例执行：

```text
词组：基线、正确块置零、半缩放、同块置换、错误深度、错误位置；
行为：基线、正确块置零、错误深度、错误位置。
```

所有评分和生成均为单样本左填充、关闭缓存。

任务门沿用：

$$
G_{task}
=G_{baseline}
\land G_{finite}
\land G_{correct\ loss}
\land G_{spatial\ control}
\land G_{phrase\ superiority}
\land G_{heldout}
\land G_{private}
$$

一般复制特异门额外要求：

$$
G_{copy}
=
\mathbf{1}[N_{explicit}\ge4]
\land
\mathbf{1}[N_{noncopy}=0]
\land
G_{label}
\land G_{digit}
\land G_{symbol}
\land G_{multitoken}
$$

### 二、执行规模

```text
注册留出案例：585；
词组条件行：3510；
单样本行为行：2340；
完整执行：3/3；
非有限词组行：1；
模型 × 任务单元：39；
完整任务门通过：2/39。
```

唯一非有限行来自：

```text
GLM4；
单数一致；
私有留出；
format_c；
正确块置换。
```

该行写为 `null`，使对应划分有限率门失败。

### 三、三模型结果

Qwen3：

```text
通过：数字复制；
显式复制：1/6；
复制近邻：0/3；
非复制控制：0/4；
词汇泛化门：失败；
Phase338 原粗块门：失败；
身份：描述性局部任务效应。
```

DeepSeek7B：

```text
通过：0/13；
Phase338 原粗块门：失败；
身份：无可升级结果。
```

GLM4：

```text
通过：多词元短语复制；
显式复制：1/6；
复制近邻：0/3；
非复制控制：0/4；
词汇泛化门：失败；
一般复制特异门：失败。
```

GLM4 多词元短语复制：

```text
公开/私有基线能力：1/1；
正确块行为损失率：0.6666667/0.6666667；
错误深度或位置最大行为损失率：0/0；
词组对照优势：3.3191275/2.3906239；
公开/私有双留出：通过。
```

但其他显式复制任务未通过：

```text
随机标签：正确块损伤强，但公开错误块行为损失 0.4444444；
数字复制：公开错误块行为损失 0.2222222；
任意符号：公开/私有错误块行为损失 0.1111111/0.5；
跨句指针：词组对照优势 -0.1658257/-0.4652912；
延迟复制：正确块行为损失 0/0，词组优势为负；
键值读取：正确块行为损失 0/0，词组优势为负；
对象名称回送：错误块行为损失超过门；
字段抽取：正确块行为损失 0/0。
```

### 四、关键负结果

Phase341 的“显式来源词元复制/回送粗块候选”没有在更广词表、更长短语、跨句、延迟、键值和字段任务上形成统一边界。

因此应当关闭：

```text
GLM4 一般显式复制模块；
GLM4 词汇泛化复制粗块；
复制自然状态内容替换入口；
复制层组收缩入口；
复制通道或单神经元扫描入口。
```

保留的最窄客观现象只有：

```text
GLM4 早期来源 MLP 粗块对多词元短语复制存在模型特异、任务特异的必要性效应；
Qwen3 对数字复制存在描述性冻结块效应。
```

单任务效应不能升级为操作基元，更不能进入智能理论主体。

### 五、为何不能继续自然状态替换

自然错误词元状态替换是用来回答“块状态是否携带可转移内容”的充分性问题。它的前提是同一块已经在多类复制任务上满足必要性和低副作用。

当前：

$$
N_{explicit}^{GLM4}=1<4,
\qquad
G_{lexical}=0
$$

前提不成立。继续状态替换只会围绕一个单任务异常优化干预，属于后验追逐，不是系统图谱推进。

### 六、全局图谱进度

```text
九族工程登记：9/9；
机制目录普查：72/72；
深因果审计：1/72；
复制操作边界审计：13 任务 × 3 模型；
跨模型复制候选：0；
行为机制闭合：0/72；
单神经元因果闭合：0/72；
语言编码机制闭合：否；
智能理论实验闭合：否。
```

仍不报告单一总体百分比。Phase342-344 新增的是测量可靠性和一个候选的否定，不等于完成了新的语言机制。

### 七、小模型边界

小模型可能把复制功能分散在多个区域，或因冗余不足对错误深度块同样敏感。GLM4、Qwen3、DeepSeek7B 的冻结组件类型也不同。因此当前结果不能证明大模型不存在复制操作。

但本轮能够严格证明：

```text
在当前三个小模型、当前冻结粗块和当前合格执行路径下，
没有一个粗块满足跨词表、跨任务、低副作用的一般复制边界。
```

### 八、智能理论与第一性原理反思

本轮不修改智能理论名称。对“操作基元优先”增加一个负约束：

```text
操作基元不能仅由任务行为相似性命名；
必须由跨包装必要性、低副作用、内容可转移性和物理路径共同定义。
```

复制任务的共同外观不意味着它们共享同一粗块。多词元复制、数字复制、标签复制、指针读取可能依赖不同的来源保持、位置路由、候选竞争和输出序列机制。

这更符合动态运行子图主体，但仍只是对理论对象的约束，不是新的统一公式。

### 九、接下来的阶段性任务

Phase342-344 已完成并触发停止门。下一任务不应继续缩小当前复制粗块，而应回到全局语言模式图谱：

1. 将“多词元序列保持”登记为单任务现象，不立即下钻；
2. 从 72 个机制中选择尚未深审、协议已稳定且跨三模型基线合格的下一机制；
3. 先完成自然运行的组件 × 深度 × 位置分布，再做冻结粗块因果筛选；
4. 所有正式行为使用 Phase342 合格的单样本路径；
5. 只有重新出现多任务功能同构，才允许最小集合、内容替换或层组收缩。

这已经改变研究对象，因此不属于当前复制候选阶段，不在本轮自动执行。

### 十、阶段产物

```text
tests/gpt5/phase344_copy_block_boundary_case_bank.py；
tests/gpt5/phase344_copy_block_boundary_audit.py；
tests/gpt5/phase344_copy_block_boundary_analysis.py；
tests/gpt5/run_phase344_copy_block_boundary.sh；
tests/gpt5/test_phase344_copy_block_boundary.py；
tests/gpt5/result/phase344_copy_block_boundary/copy_block_heldout_boundary/。
```

### 十一、通俗总结

前一阶段看起来像发现了一块“把提示里的词送到答案”的区域。这次把复制拆成标签、数字、符号、跨句、多词短语、延迟复制、键值和字段读取，再用关系、语法、推理和转换做控制。

结果只有 GLM4 的多词短语复制通过，其他复制形式要么正确块没有作用，要么错误深度或错误位置也会让答案失败。也就是说，这不是一个通用复制区域，只是一个局部任务效应。

因此最重要的进展是及时关闭了一个仍然过宽的解释。现在继续找它的层和神经元没有科学意义，应回到全局图谱选择下一个独立机制。

## Phase 345: 三核心正交基线资格与附件方案审计 [2026-07-11 01:42]

### 一、对 Phase342-344 总结的审计

附件对主要证据链的判断基本正确：Phase342 冻结了当前可信的单样本测量路径，Phase343 建立了较大的复制任务边界，Phase344 否定了冻结粗块作为一般复制操作基元。以下内容可以保留：

```text
批量执行不能默认等价于单样本执行；
任务外观相似不能推出物理机制相同；
GLM4 多词元复制只是一条模型特异、任务特异局部效应；
一般复制粗块、自然状态内容替换和神经元下钻入口均应关闭；
自然轨迹只能提供搜索范围，不能形成因果结论；
有效单元必须同时满足自然参与、必要性、充分性、中介性、特异性、留出稳定和低副作用。
```

需要收紧的部分：

```text
“96 项拼图”只是整理目录，不是稳定的机制分母；
约 20% 或 17%-22% 的总体完成率没有可审计分母，不能报告；
MCUE 只是待验证算法框架，不是已经有效的神经元提取方法；
原建议把 2592 基线、全轨迹、27 粗块、神经元集合、充分性和中介性放入同一阶段，范围过大；
若不设置逐级停止门，后续结论会受到候选选择、留出泄漏和多重后验筛选影响。
```

因此将原 Phase345 拆成四个顺序门：

```text
Phase345：基线资格；
Phase346：协议失败修复；
Phase347：自然物理轨迹；
Phase348：最小粗块因果筛选与留出停止门。
```

### 二、固定分母

三核心与协议控制共 12 个任务：

```text
知识网络：上下文关系绑定、参数知识检索、显式复制；
推理：缺失条件检查、两跳蕴含、直接事实控制；
语法：完整句数一致、过去时、无形态变化；
协议：仅答案、多词元自然答案、无来源答案。
```

每任务 24 个独立题项、3 个模板、3 个模型：

$$
N_{345}=12\times24\times3\times3=2592
$$

划分固定为：

```text
发现 12；校准 5；公开留出 4；私有留出 3。
```

所有测试使用 Phase342 冻结的单样本、左侧自然序列、关闭缓存路径。本阶段没有内部干预。

### 三、客观结果

```text
注册案例：2592；
词组评分行：2592；
自然生成行：2592；
模型 × 任务单元：36；
四划分全部通过：27/36；
内部干预：0；
行为机制闭合：0；
单神经元因果闭合：0。
```

按模型的合格任务数：

```text
Qwen3：知识 3，推理 3，语法 3，协议 1；
GLM4：知识 3，推理 2，语法 2，协议 1；
DeepSeek7B：知识 3，推理 2，语法 3，协议 1。
```

失败任务：

```text
Qwen3：多词元自然答案、无来源答案；
GLM4：缺失条件检查、完整句数一致、多词元自然答案、无来源答案；
DeepSeek7B：缺失条件检查、多词元自然答案、无来源答案。
```

GLM4 出现 24 条无效词组评分，全部集中在完整句数一致的 `format_c` 模板。由于三个模型都只有一个协议任务通过，原定“协议至少两个任务”的自然轨迹入口没有打开。

### 四、阶段结论

原 12 任务矩阵适合暴露协议问题，但不能直接进入内部机制测试。任务分类覆盖不等于任务契约已经稳定。需要先修复协议任务，而不是降低入口门槛。

### 五、产物

```text
tests/gpt5/phase345_three_core_protocol_case_bank.py；
tests/gpt5/phase345_three_core_protocol_qualification.py；
tests/gpt5/phase345_three_core_protocol_analysis.py；
tests/gpt5/run_phase345_three_core_protocol.sh；
tests/gpt5/test_phase345_three_core_protocol.py；
tests/gpt5/result/phase345_three_core_protocol/three_core_protocol_qualification/。
```

## Phase 346: 协议控制修复与自然轨迹入口复核 [2026-07-11 01:43]

### 一、修复原则

Phase345 的多词元和无来源任务同时改变了答案连续性、答案格式和任务难度，不能确定失败来自语言机制还是测试契约。Phase346 使用两类更小的协议控制：

```text
连续多词元答案：目标短语在上下文中完整连续出现；
简单无来源答案：小整数加法，答案不在上下文出现。
```

仍使用 24 题项、3 模板、3 模型：

$$
N_{346}=2\times24\times3\times3=432
$$

### 二、结果

```text
注册案例：432；
词组评分行：432；
自然生成行：432；
无效行：0；
两个任务在三个模型的发现、校准、公开留出、私有留出全部通过；
自然轨迹入口：三个模型均打开；
内部干预：0。
```

该结果证明 Phase345 的协议失败不能直接解释为模型缺少多词元或无来源回答能力。任务构造本身是重要混杂变量。

### 三、边界

这只是协议分母修复，不证明多词元生成或无来源推理共享任何物理机制，也不增加 72 个机制的闭合数量。

### 四、产物

```text
tests/gpt5/phase346_protocol_repair_case_bank.py；
tests/gpt5/phase346_protocol_repair_qualification.py；
tests/gpt5/phase346_protocol_repair_analysis.py；
tests/gpt5/run_phase346_protocol_repair.sh；
tests/gpt5/test_phase346_protocol_repair.py；
tests/gpt5/result/phase346_protocol_repair/three_core_protocol_repair/。
```

## Phase 347: 十任务三模型自然物理轨迹图谱与尺度校准 [2026-07-11 01:44]

### 一、固定任务与测量对象

从 Phase345/346 选择十个三模型共同合格任务，每个划分固定抽取 2 个题项，保留 3 模板：

$$
N_{347}=10\times8\times3\times3=720
$$

记录三个组件：

```text
attention_output（注意力输出）；
mlp_output（多层感知机输出）；
residual_increment（残差增量）。
```

记录三个深度区间和三个位置角色：

```text
early / middle / late（早/中/晚）；
source / query / answer_start（来源/查询/答案起点）。
```

因此固定图谱节点为：

$$
N_{node}=3\text{ 模型}\times10\text{ 任务}\times3\text{ 组件}\times3\text{ 深度}\times3\text{ 位置}=810
$$

### 二、特征分析算法校准

初始算法直接比较目标首词元在组件向量上的绝对投影：

$$
p=|v\cdot \hat{u}_{target}|
$$

第一次聚合中 4/10 任务出现“晚层残差 × 答案起点”的跨模型完全一致，但这主要受到组件范数和通用晚层读出的影响，不能作为模式特异脉络。

因此改为逐向量归一化：

$$
a(v,t)=\frac{|v\cdot u_t|}{\|v\|\|u_t\|}
$$

再扣除同模型、同组件、同深度、同位置上其他任务的公共基线：

$$
s_{r,n}=a_{r,n}-\frac{1}{|R|-1}\sum_{q\ne r}a_{q,n}
$$

这里的 $s_{r,n}$ 只是任务相对对齐超额，不是因果贡献。

### 三、执行结果

```text
注册/完成案例：720/720；
Qwen3 原始轨迹：38880；
GLM4 原始轨迹：43200；
DeepSeek7B 原始轨迹：30240；
原始轨迹总数：112320；
固定物理节点：810/810；
非有限案例：0；
任务 × 模型自然主导区域：30。
```

经过范数和跨任务公共基线校准后：

```text
跨模型深度-位置一致：4/10；
跨模型组件-深度-位置完全一致：2/10。
```

完全一致的两项：

```text
连续多词元答案：晚层注意力输出 × 答案起点；
无形态变化控制：晚层注意力输出 × 答案起点。
```

阶段和位置一致、组件分叉的两项：

```text
直接事实控制：晚层 × 答案起点，MLP/残差分叉；
两跳蕴含：晚层 × 答案起点，MLP/残差分叉。
```

其余 6 项在模型间不一致。

### 四、硬伤与边界

```text
自然轨迹记录的是相关性，不是必要性；
首词元读出不能代表完整短语和生成时间链；
跨任务基线校准仍会受到不同目标词元、词频和模型尺度影响；
Phase347 使用了四个划分的自然特征，后续因果留出不是完全不可见的物理留出；
810 个节点是固定工程节点，不是 810 个已发现机制；
没有记录单神经元，也没有执行任何神经元干预。
```

### 五、图谱固定数据

```text
phase347_natural_physical_nodes.jsonl：810 个固定物理节点；
phase347_dominant_natural_regions.jsonl：30 个任务-模型描述性区域；
phase347_cross_model_convergence.jsonl：10 个任务跨模型一致性；
phase347_global_summary.json：固定分母和声明边界。
```

数据目录：

```text
tests/gpt5/result/phase347_three_core_natural_trace/three_core_natural_physical_trace/。
```

### 六、结论

Phase347 完成的是三核心十任务的自然物理分布子图，不是语言模式族全图谱。绝对投影峰值已被证明容易产生通用晚层读出伪候选；只有经过尺度和跨任务公共基线校准的节点可以进入粗筛，而且仍不得计作机制。

## Phase 348: 校准候选粗块因果筛选与留出否定 [2026-07-11 01:45]

### 一、预注册设计

只冻结 Phase347 中两个跨模型完全一致的候选：

```text
连续多词元答案；
无形态变化控制。
```

两者的冻结块均为：

```text
attention_output × late × answer_start
（注意力输出 × 晚层 × 答案起点）。
```

加入四个匹配控制：

```text
显式复制；
简单无来源答案；
过去时；
直接事实控制。
```

完整注册分母：

$$
N_{348}=6\times24\times3\times3=1296
$$

其中发现和校准 918 个案例，公开与私有留出 378 个案例预先封存。每个案例执行：

```text
基线；
正确块置零；
正确块半缩放；
错误深度置零；
错误位置置零。
```

目标和两个干扰项均拆成真实批量 1 前向，关闭缓存。候选门要求发现和校准同时满足：

$$
G_{screen}
=G_{baseline}\land G_{loss}\land G_{positive}
\land G_{spatial}\land G_{taskcontrol}\land G_{half}
$$

### 二、发现与校准结果

```text
筛选案例：918；
条件记录：4590；
内部干预记录：3672；
无效记录：0；
实际模型批量：1；
候选模型单元：6；
发现和校准双门通过：1/6。
```

失败：

```text
Qwen3 连续多词元、无形态变化；
GLM4 连续多词元、无形态变化；
DeepSeek7B 连续多词元。
```

唯一进入留出的候选：

```text
DeepSeek7B 无形态变化控制。
```

多词元候选在三模型均出现明显正确块损伤，但匹配控制任务损伤相当或更大，因此是通用晚层读出敏感，不是多词元特异机制。

### 三、选择性留出结果

只解封 DS7B 无形态变化及其过去时、直接事实两个控制：

```text
留出案例：63；
词组条件记录：315；
自然生成记录：252；
实际模型批量：1。
```

公开留出：

```text
目标正确块词组损失：8.8780815；
最大空间控制词组损失：4.5852377；
最大匹配任务词组损失：4.7836506；
目标行为损失率：0.5；
最大空间控制行为损失率：0.75；
最大匹配任务行为损失率：0.8333333；
词组门：通过；
行为特异门：失败。
```

私有留出：

```text
目标正确块词组损失：2.2153614；
最大空间控制词组损失：2.409657；
最大匹配任务词组损失：4.5112303；
目标行为损失率：0.5555556；
最大空间控制行为损失率：0.8888889；
最大匹配任务行为损失率：0.5555556；
词组门：失败；
行为特异门：失败。
```

最终：

$$
G_{heldout}=0,\qquad G_{private}=0
$$

### 四、严格结论

Phase347 的跨模型自然对齐峰值没有形成可留出复制的任务特异粗块。当前证据支持：

```text
晚层答案起点是多个任务的通用输出敏感区；
绝对或校准后的自然投影可以缩小搜索空间；
自然峰值仍不能可靠预测低副作用因果块；
当前两个候选全部关闭；
MCUE 入口关闭；
单神经元 CUDA 扫描入口关闭。
```

不能支持：

```text
多词元操作基元；
无形态变化专用块；
跨模型同构粗块；
有效神经元或最小神经元集合；
行为机制闭合。
```

### 五、图谱进度

```text
九族工程登记：9/9；
机制目录普查：72/72；
本轮三核心自然轨迹：10 任务 × 3 模型；
本轮固定自然物理节点：810/810；
本轮粗块候选因果筛选：2 任务 × 3 模型；
本轮留出深审：1 任务 × 1 模型；
本轮新增跨模型粗块候选：0；
行为机制闭合：0/72；
单神经元因果闭合：0/72；
语言编码机制闭合：否；
智能理论实验闭合：否。
```

仍不报告总体百分比。九族登记、72 项普查、810 个工程节点和机制闭合是不同分母，不能相加。

### 六、小模型边界

三个模型的规模和结构可能使内部机制更粗糙、冗余更少，错误区域删除也更容易造成广泛损伤。当前结果不能推出大模型一定使用相同或不同结构。

本轮可以严格复现的范围只有：

```text
在 Qwen3、GLM4、DeepSeek7B，当前十任务、当前冻结执行路径和当前粗块定义下，
自然任务对齐峰值没有产生跨模型、任务特异、低副作用的粗块因果机制。
```

### 七、智能理论约束

不修改理论名称。新增的实验约束是：

$$
\text{模式运行脉络}
\ne
\arg\max_n \text{Activation}(n)
$$

$$
\text{模式运行脉络}
\ne
\arg\max_n \text{ReadoutProjection}(n)
$$

更合理的研究对象仍是条件化动态运行子图：

$$
G_r(x,t,m)
=
\big(V_r,E_r,\tau_r,\omega_r\big)
\mid(x,t,m)
$$

但该表达目前只是实验对象定义，不是已闭合的智能统一公式。

### 八、下一阶段大任务

Phase348 已触发停止门，不能自动进入神经元搜索。下一阶段应改变自然特征算法和样本设计，而不是修补当前块：

1. 为九族建立全新因子正交样本，不复用 Phase347 私有自然特征；
2. 使用任务对照差分而非绝对激活或绝对投影；
3. 增加完整生成时间轴，区分来源读取、查询整合、首词元读出和多词元维持；
4. 先比较同任务正确/错误、同词汇不同操作、同操作不同词汇；
5. 对每个模型单独建立组件尺度基线，再寻找功能事件顺序，不强求相同层号或相同神经元；
6. 只有自然差分在发现、校准和全新物理留出中稳定，才重新开放 27 粗块因果筛选；
7. 只有粗块在短语、自然生成、任务控制、错位置和错深度中同时通过，才开放 MCUE；
8. 单神经元只是最小集合可能收缩到 1 的终点，不作为预设目标。

这属于新的全局差分图谱阶段，不属于当前候选块阶段。Phase345-348 的阶段性目标已经完成，当前停止是预注册规则执行结果，不是任务中断。

### 九、产物

```text
tests/gpt5/phase347_three_core_natural_trace_case_bank.py；
tests/gpt5/phase347_three_core_natural_trace.py；
tests/gpt5/phase347_three_core_natural_trace_analysis.py；
tests/gpt5/run_phase347_three_core_natural_trace.sh；
tests/gpt5/test_phase347_three_core_natural_trace.py；
tests/gpt5/phase348_adjusted_block_screen_case_bank.py；
tests/gpt5/phase348_adjusted_block_screen.py；
tests/gpt5/phase348_adjusted_block_screen_analysis.py；
tests/gpt5/phase348_adjusted_block_heldout.py；
tests/gpt5/phase348_adjusted_block_heldout_analysis.py；
tests/gpt5/run_phase348_adjusted_block_screen.sh；
tests/gpt5/test_phase348_adjusted_block_screen.py；
tests/gpt5/result/phase347_three_core_natural_trace/；
tests/gpt5/result/phase348_adjusted_block_screen/。
```

### 十、通俗总结

这轮先把知识、推理、语法和回答协议中的十个稳定任务放进模型，观察信息在哪些层、哪些组件和哪些位置最明显。两个任务在三个模型里都指向“晚层、答案开始位置的注意力输出”。

但真正把这块删除后，它也会严重破坏别的任务，错位置和错深度有时破坏得更厉害；唯一进入留出的 DS7B 候选也没有通过自然生成和私有留出。因此它不是语言规则所在的专用区域，而是一个普遍脆弱的输出区域。

最重要的成果不是找到神经元，而是用大样本、三模型和严格停止门证明：自然运行中最亮的地方仍可能只是公共输出瓶颈。下一步必须研究“相对于严格对照，哪些路径发生了任务特异变化”，而不是继续追最亮的层和神经元。

## Phase 349: 七十二机制四联合同审计与统一分母否定 [2026-07-11 03:26]

### 一、对最新附件的审计

附件对 Phase345-348 的主要证据判断正确：

```text
自然高对齐区域只能缩小搜索空间；
自然峰值不能预测任务特异、低副作用的因果块；
绝对余弦丢失增强/抑制符号；
跨任务均值不是严格匹配对照；
候选选择使用全部自然划分，使后续因果留出不是完全独立的物理留出；
下一路线应转向成对、有符号、动态、独立物理留出的差分图谱。
```

需要收紧：

```text
“100 项拼图”仍是知识目录，不是机制完成分母；
24% 科学进度来自人为权重，不是实验统计量，不能作为总体完成度；
十二个操作标签只是候选分类，不是已验证的语言本体；
统一 A/B/C/D 四联定义不能直接覆盖知识、协议、语法、动作、漂移和闭合等不同对象；
在合同资格未知时直接执行 20736 个全层、全时间案例，会放大样本构造错误。
```

### 二、合同审计算法

从 Phase330 冻结的 9 族、72 机制、24 题项、2 个选择模板中，为每个机制登记一个候选主操作，并在同族寻找不同候选操作的匹配控制。

不读取任何模型效应，只检查：

```text
配对案例数；
基础词汇重合；
语言一致；
协议一致；
答案空间一致；
候选操作标签不同。
```

严格门：

$$
G_{contract}
=G_{pair}\land G_{lexical}\land G_{language}
\land G_{protocol}\land G_{targetspace}\land G_{operationlabel}
$$

门槛只是执行资格，不是语言规律。

### 三、结果

```text
登记语言族：9；
登记机制：72；
候选操作标签：12；
审计源案例：3456；
原计划模型提示案例：20736；
严格四联就绪：18/72；
需要重构：54/72；
三模型前向：未启动；
内部干预：0。
```

按族：

```text
闭合：0/8；
内容知识：0/8；
跨语言：0/8；
语言动作：0/8；
输出协议：0/8；
读出竞争：7/8；
推理约束：4/8；
状态漂移：7/8；
语法结构：0/8。
```

输出协议和闭合族的平均词汇重合为 1，但严格门仍为 0。原因是这些族的目标变量本身就是协议或停止规则，要求 A/B 协议完全相同会把被研究变量一并消除。这证明统一合同门在逻辑上不能无修改地覆盖九族。

### 四、结论

附件提出的 20736 全量分母当前不具备执行资格。直接运行不会产生全局正交图谱，只会生成大量名义上成对、实际上混杂的数据。

固定格式产物：

```text
phase349_contrast_contract_registry.jsonl：72 个合同节点；
phase349_family_contract_summary.jsonl：9 个族合同统计；
phase349_global_summary.json：固定分母、门槛和声明边界。
```

目录：

```text
tests/gpt5/result/phase349_contrast_contract_audit/orthogonal_contrast_contract_audit/。
```

## Phase 350: 九族代表性最小四联合同与三模型资格 [2026-07-11 03:26]

### 一、修复设计

为避免一次修改 72 个机制，先为每个语言族人工构造一个代表合同：

```text
内容知识：否定属性；
输出协议：仅答案；
推理约束：缺失条件；
语法结构：过去时；
语言动作：内容转换；
跨语言：翻译；
读出竞争：目标对错误候选；
状态漂移：实体保持；
闭合：多词元完成。
```

每个题项构造：

```text
A：目标操作 + 词汇 X；
B：显式捷径或放宽协议 + 词汇 X；
C：目标操作 + 词汇 Y；
D：显式捷径或放宽协议 + 词汇 Y。
```

A/B 与 C/D 的目标完全一致，但 B/D 可能显式给出答案，因此它不是纯粹的“操作关闭”，只能作为操作需求对显式捷径的工程控制。

固定分母：

$$
N_{350}=9\times12\times2\times3\times4=2592
$$

划分：

```text
物理发现：6 个四联题项；
物理校准：2；
物理留出：2；
因果封存：2。
```

短语目标和两个干扰项均使用真实批量 1 独立评分，同时执行批量 1 自由生成，关闭缓存。

### 二、结果

```text
注册案例：2592；
短语记录：2592；
自然生成记录：2592；
模型 × 族单元：27；
四划分完整合同通过：14/27；
三模型四划分完整通过族：2/9；
三模型物理发现/校准入口族：3/9；
内部干预：0。
```

三模型完整通过：

```text
闭合；
状态漂移。
```

发现/校准三模型通过、但后续划分不完整：

```text
语言动作。
```

因此只允许闭合、状态漂移和语言动作进入发现/校准自然轨迹，不解封任何物理留出内部状态。

GLM4 有 3 条无效短语评分：

```text
内容知识物理留出 B 控制，item 08，两个模板；
推理约束物理发现 A 操作，item 05，format_a。
```

这些行保持无效并令对应门失败，没有从均值中删除。

### 三、边界

Phase350 验证的是九族代表性对照合同的行为可执行性，不是 72 机制差分图谱，也不是物理或因果结论。显式捷径控制仍混入答案可见性变化。

固定数据：

```text
phase350_registered_cases.jsonl：2592 个案例；
phase350_model_family_contract_summary.jsonl：27 个模型族单元；
phase350_cross_model_family_summary.jsonl：9 个跨模型族单元；
phase350_atlas_nodes.jsonl：27 个资格图谱节点。
```

## Phase 351: 三族静态有符号成对差分物理图谱 [2026-07-11 03:26]

### 一、测量范围

只读取三族的物理发现和物理校准：

$$
N_{351}=3\times8\times4\times2\times3=576
$$

不读取物理留出和因果封存。记录：

```text
注意力输出、MLP 输出、残差增量；
早、中、晚深度；
来源、查询、答案起点；
目标首词元有符号余弦；
最佳干扰项有符号余弦；
目标对竞争者的有符号边距。
```

对每个匹配节点：

$$
\Delta_x=M(A)-M(B)
$$

$$
\Delta_y=M(C)-M(D)
$$

并记录：

$$
\bar\Delta=\frac{\Delta_x+\Delta_y}{2},
\qquad
I_{lex}=|\Delta_x-\Delta_y|
$$

这里的差分仍是自然描述量，不是因果效应。

### 二、结果

```text
案例：576；
原始逐案例轨迹：179712；
完整四联物理事件：44928；
不完整事件：0；
固定静态有符号节点：243/243；
非有限轨迹：0；
静态发现/校准节点门：1；
具有静态门的模型族单元：1；
跨模型深度-位置-符号一致族：0/3。
```

唯一静态门：

```text
GLM4；
状态漂移；
中层注意力输出；
答案起点；
正向差分。
```

其他模型在状态漂移上分别偏向晚层来源注意力和早层来源 MLP，功能位置不一致。

### 三、结论

有符号配对算法显著收紧了 Phase347 的自然峰值：从多个晚层绝对投影候选缩减为一个单模型静态节点，且跨模型一致为零。该节点不能进入物理留出，因为静态提示末端不是完整动态路径。

固定数据：

```text
phase351_signed_physical_nodes.jsonl：243 个节点；
phase351_dominant_static_regions.jsonl：9 个模型族区域；
phase351_cross_model_convergence.jsonl：3 个族一致性记录。
```

## Phase 352: 教师强制生成时间轴与动态入口关闭 [2026-07-11 03:26]

### 一、算法

对 Phase351 的 576 个案例，沿正确目标序列最多展开 4 个教师强制时间步。新增位置角色：

```text
current_generation（当前生成位置）。
```

时间相位：

```text
first（首步）；
middle（中间）；
final（末步）；
first_final（单词元首末合一）。
```

固定动态图节点：

$$
N_{node}=3\times3\times4\times3\times3\times4=1296
$$

### 二、结果

```text
案例：576；
目标时间步：1128；
原始时间轨迹：469248；
完整四联时间事件：94848；
相位不完整事件：44928；
完整动态配对率：0.6785714；
固定动态节点：1296/1296；
非有限轨迹：0；
动态节点门：8；
具有动态门的模型族单元：5；
跨模型动态入口族：0/3。
```

相位不完整不是文件缺行，而是 X/Y 两组目标经过不同分词后具有不同词元长度，导致 first/middle/final 无法一一配对。这证明跨词汇生成时间不能只按离散相位名称对齐。

三族跨模型功能值：

```text
闭合：末步，但深度、位置和符号分叉；
语言动作：末步负向，但早/中/晚和查询/答案起点分叉；
状态漂移：首步/末步、来源/答案起点和符号均分叉。
```

因此：

$$
G_{dynamic}^{cross-model}=0/3
$$

### 三、严格结论

当前可以支持：

```text
统一四联合同不能覆盖九族；
九族代表性人工合同中只有三族具备跨模型发现/校准轨迹入口；
有符号成对差分比绝对投影明显更严格；
教师强制动态差分在单模型出现局部节点，但没有跨模型功能一致；
词元长度是生成时间对齐的关键物理变量。
```

不能支持：

```text
三族共享同构动态路径；
显式捷径差分等于纯操作差分；
教师强制路径等于自由生成路径；
任何物理留出候选；
任何粗块、神经元或机制闭合。
```

当前入口：

```text
物理留出：关闭；
因果封存：关闭；
27 粗块：关闭；
MCUE：关闭；
单神经元 CUDA：关闭。
```

### 四、图谱进度

```text
九族工程登记：9/9；
72 机制描述目录：72/72；
严格统一四联合同：18/72；
九族代表合同：9/9；
三模型完整行为合同：2/9 族；
三模型发现/校准轨迹入口：3/9 族；
静态有符号节点：243；
动态时间节点：1296；
跨模型动态路径候选：0；
行为机制闭合：0/72；
单神经元闭合：0/72；
语言编码机制闭合：否；
智能理论实验闭合：否。
```

仍不报告单一总体百分比。目录覆盖、合同资格、描述节点和因果闭合不能相加。

### 五、小模型边界

三个小模型可能在组件、深度和词元分解上差异很大，因此跨模型物理编号不一致并不证明不存在共同功能。但本轮连归一化后的时间相位、位置角色和符号都不一致，所以当前还没有功能顺序同构证据。

### 六、下一阶段大任务

Phase349-352 已完整触发停止门。不能自动解封物理留出。下一阶段必须先修复合同和时间对齐：

1. 为各族分别定义被研究变量，取消不适用的统一协议一致门；
2. 用“同答案、不同自然计算路径”替换显式答案泄漏控制；
3. 对 X/Y 词汇集合进行目标词元长度配平；
4. 使用目标词元相对进度和语义子词边界，而不是仅用 first/middle/final；
5. 优先修复内容知识、推理、语法、跨语言等未进入三模型轨迹的族；
6. 重新运行发现/校准，只有跨模型功能事件顺序稳定才解封物理留出；
7. 物理留出通过后，才允许因果封存和粗块干预。

这属于新的合同重构阶段。当前不继续运行留出不是中断，而是执行预注册独立留出规则。

### 七、阶段产物

```text
tests/gpt5/phase349_contrast_contract_audit.py；
tests/gpt5/test_phase349_contrast_contract_audit.py；
tests/gpt5/phase350_nine_family_minimal_contrast_case_bank.py；
tests/gpt5/phase350_nine_family_minimal_contrast_qualification.py；
tests/gpt5/phase350_nine_family_minimal_contrast_analysis.py；
tests/gpt5/phase351_signed_paired_trace_case_bank.py；
tests/gpt5/phase351_signed_paired_trace.py；
tests/gpt5/phase351_signed_paired_trace_analysis.py；
tests/gpt5/phase352_generated_time_trace.py；
tests/gpt5/phase352_generated_time_trace_analysis.py；
tests/gpt5/result/phase349_contrast_contract_audit/；
tests/gpt5/result/phase350_nine_family_minimal_contrast/；
tests/gpt5/result/phase351_signed_paired_trace/；
tests/gpt5/result/phase352_generated_time_trace/。
```

### 八、通俗总结

原计划想给 72 种语言现象都配四道题，然后直接比较内部差分。检查后发现，大多数题并没有真正做到“词汇相同、只改变操作”，尤其协议和停止本身就是变量，不能又要求协议完全相同。

这次先手工做了九族各一个小合同。只有闭合、状态漂移和语言动作能进入三模型内部观察。使用保留正负方向的成对差分后，模型内确实能看到少量变化，但三个模型在发生时间、位置和方向上都没有形成相同路径。

所以当前最重要的拼图是：差分路线比找最亮神经元可靠，但真正困难的核心已经转移到“如何构造只改变一个语言操作的自然对照”和“如何跨不同分词对齐生成时间”。这两个问题解决前，不应该打开留出和神经元搜索。

## Phase 353: 九族族特异合同编译与三模型行为资格 [2026-07-11 04:50]

### 一、审计结论

Phase349-352 审计文本的核心判断正确：统一四联合同不能直接覆盖九个语言模式族，教师强制离散相位也不能替代自然生成时间。需要收紧的是：附件提出的 Phase353 仍是方案，不是证据；“112 个拼图”和单一总体百分比属于管理目录，不能与行为、物理、因果闭合相加；族特异合同必须先通过可执行资格门。

### 二、固定合同和分母

冻结九族、每族两个代表机制，共 18 份合同：

$$
C_f=(Z_f,I_f,M_f,Y_f)
$$

其中 $Z_f$ 是匹配上下文，$I_f$ 是目标操作，$M_f$ 是测量变量，$Y_f$ 是预期输出。

固定注册分母：

$$
N_{353}=9\times2\times12\times2\times3\times4=5184
$$

机械资格检查目标可见性、语言一致性、目标词元数、对照词汇重叠和格式可执行性。结果：

```text
严格机械合同：11/18；
需要重构：7/18；
每模型执行：1056；
三模型执行：3168；
候选评分行：3168；
自由生成行：3168。
```

推理约束、语法结构、语言动作及跨语言翻译中的 7 份合同因机械混杂被挡在模型执行之前。

### 三、三模型结果

严格按 `qwen3 -> GLM4 -> DS7B` 串行执行，批量大小为 1，每个模型释放后才加载下一个模型。

```text
模型-合同完整行为单元：11/33；
跨模型完整行为合同：2/11；
跨模型发现/校准轨迹入口：3/11；
轨迹入口：关系绑定、目标对错误候选竞争、实体新近性；
行为机制闭合：0；
单神经元因果：0。
```

GLM4 在关系绑定物理留出 item 08、format_a 上产生 1 条稳定非有限候选评分。使用相同样本连续重试三次，并用带前导空格的自然上下文词元完整重跑后仍然无效，因此保留异常并令 GLM4 完成清单严格无效，没有删除该行或替换样本。

初始候选评分使用无前导空格的小写目标词元，而自然生成通常使用带前导空格、且可能首字母大写的词元。修正为：

$$
v_{ctx}=\operatorname{Tok}(\text{" "}+v)
$$

修正没有改变题目、目标、分割、行为成功判定和三份轨迹入口名单。它证明词元边界是物理测量协议的一部分。

### 四、严格边界

Phase353 只完成合同编译和行为资格。跨模型完整行为合同 2/11 不能解释为 2 个语言机制闭合；3 个轨迹入口也不能解释为已找到物理路径。物理留出内部状态、因果封存、粗块干预和单神经元 CUDA 干预全部保持关闭。

## Phase 354: 合格合同的教师强制与自由生成语义时间图谱 [2026-07-11 04:50]

### 一、范围和算法

只追踪 Phase353 通过跨模型发现/校准门的三份合同：内容知识/关系绑定、读出竞争/目标对错误候选、状态漂移/实体新近性。

固定案例：

$$
N_{354}=3\times3\times8\times4\times2=576
$$

其中物理发现 432，物理校准 144，物理留出和因果封存均为 0。

并行记录教师强制轨迹和自由生成轨迹：

$$
H^{TF}_{t}=F(x,y^*_{<t})
$$

$$
H^{FR}_{t}=F(x,\hat y_{<t})
$$

语义时间为：

$$
\rho_t=\frac{t}{\max(T-1,1)}
$$

每步记录注意力输出、MLP 输出和残差增量，在来源、查询、答案起点、当前生成位置四个角色上计算目标对最佳竞争候选的有符号边距。

四条件成对差分：

$$
\Delta_x=M(A)-M(B)
$$

$$
\Delta_y=M(C)-M(D)
$$

$$
\bar\Delta=\frac{\Delta_x+\Delta_y}{2},\qquad I_{lex}=|\Delta_x-\Delta_y|
$$

### 二、表面词元校准

第一轮发现三个模型 576/576 个自由生成案例都在首词元被标记为偏离，但自由生成语义正确。原因是目标使用无空格小写词元，自由生成使用带空格或首字母大写词元。

修正为各模型自然回答表面形式后：

```text
qwen3 自由回答前缀匹配：191/192；
GLM4 自由回答前缀匹配：192/192；
DS7B 自由回答前缀匹配：140/192；
DS7B 自由生成语义正确：188/192。
```

qwen3 和 GLM4 的旧首词元偏离几乎完全是测量错误；DS7B 仍有较强的真实表面路径分叉。

### 三、修正版全量结果

```text
注册案例：576；
原始自然轨迹：1192896；
完整四条件事件：239376；
不完整事件：49632；
完整配对率：0.8282677；
固定物理节点：2592；
语义时间边：756；
非有限内部轨迹：0；
严格动态节点：0；
模型-轨迹主区域通过：0；
教师强制跨模型合同：0/3；
自由生成跨模型合同：0/3；
教师强制/自由生成跨模型一致：0/3。
```

因此：

$$
G_{354}^{strict}=0
$$

物理留出入口继续关闭。

### 四、结果解释

该负结果不是“没有任何内部信号”。发现集和校准集存在同号、模板同号的局部区域，但同一操作在不同词汇实例之间频繁翻转，且操作差异经常小于词汇差异。当前对照不能提供可跨实例复用的物理路径。

教师强制与自由生成轨迹长度明显不同，当前数据也不支持二者共享同一功能位置序列。所有节点仍是自然相关轨迹，不是因果效应。

## Phase 355: 语义时间失败分解与保守图谱同步 [2026-07-11 04:50]

### 一、失败分解

不修改 Phase354 的预注册门槛，只对 2592 个固定节点分解失败原因。严格门要求发现和校准均同时满足：绝对操作差分不小于 0.005、词汇实例符号一致率不小于 0.75、操作差异幅度大于词汇不稳定度、两个模板差分符号一致且非零。

```text
严格动态候选：0；
发现/校准同号且模板同号的降级近候选：67；
仅描述节点：2525；
物理留出入口：0；
因果入口：0；
单神经元因果：0。
```

失败计数表明第一瓶颈是词汇符号不稳定，第二瓶颈是操作差异未超过词汇差异，第三才是绝对信号不足。67 个近候选全部标记：

```text
heldout_eligible=false；
causal_eligible=false；
causal_status=not_tested。
```

### 二、固定图谱同步

研究图谱和前端图谱同步写入：

```text
phase354_semantic_time_nodes.jsonl：2592 个节点；
phase354_semantic_time_edges.jsonl：756 条边；
phase354_cross_model_convergence.jsonl：3 份合同；
phase355_near_candidates.jsonl：67 个降级近候选；
phase355_failure_summary.json：严格负结果和失败原因。
```

两个目录的 manifest 均登记：

```text
status=strict_negative_with_near_candidates；
physical_heldout_revealed=false；
single_unit_causal_count=0。
```

可视化客户端可以查看自然轨迹、语义时间边和失败原因，但不能把近候选绘制为已确认因果路径。

### 三、当前进展与硬伤

```text
九族目录：9/9；
72 机制管理目录：72/72；
Phase353 机械合同：11/18；
跨模型完整行为合同：2/11；
跨模型发现/校准轨迹入口：3/11；
Phase354 严格动态候选：0/2592；
行为机制闭合：0/72；
单神经元闭合：0/72；
语言编码机制闭合：否；
智能理论实验闭合：否。
```

仍不报告单一总体百分比。目录完成、样本运行、描述节点、物理复现和因果闭合是不同分母。

主要硬伤：

1. A/B 与 C/D 虽保持答案相同，但完整句式、答案位置和上下文路径仍可能同时变化；
2. 每个词汇集合的复本仍少，0.75 符号门容易被单个实例翻转，但当前不能事后降低门槛；
3. 自由生成在不同模型中包含不同标点、结束符和继续文本，语义时间尚未达到真正的功能事件对齐；
4. 当前只覆盖三份合格合同，不能代表九族或 72 机制；
5. 三个模型都是小模型，组件分工可能粗糙且模型间偏差较大；
6. 所有 Phase354 节点均为自然相关轨迹，不是必要性、充分性或单神经元证据；
7. 线性读出方向和有符号余弦只是局部测量，不应升级为真实运行公式。

### 四、智能理论边界

本阶段支持的最低限度判断是：语言行为更像条件化、分布式、随生成历史变化的模式网络，而不是一个静态神经元集合。但当前还没有跨模型、跨词汇、跨轨迹模式稳定的物理对象，因此不能更新智能理论主体，更不能宣称找到了语言背后的统一数学结构。

### 五、下一阶段大任务与停止门

下一阶段应是“位置配平、多词汇复本的族特异合同重构”，不是立即做物理留出或神经元搜索：

1. 每个操作对至少冻结 4 组独立词汇复本；
2. 目标与控制的答案表面形式、词元数、来源位置和查询位置严格配平；
3. 将操作变化与句式变化拆为独立维度，避免用完整句子替换近似操作干预；
4. 在发现和校准上重新验证词汇符号一致性；
5. 只有严格动态节点在两个分割、两个模板和三模型上复现，才冻结候选并打开物理留出；
6. 物理留出通过后，才进入粗块因果、回滚和单神经元 CUDA 干预。

本轮不自动打开下一次大规模模型测试。原因不是任务中断，而是 Phase355 已完成当前阶段目标并触发预注册停止门；下一轮需要先生成新的位置配平合同和独立词汇材料，否则继续扩大同一种样本只会重复已经确认的词汇不稳定。

### 六、阶段产物

```text
tests/gpt5/phase353_family_contract_case_bank.py；
tests/gpt5/phase353_family_contract_qualification.py；
tests/gpt5/phase353_family_contract_analysis.py；
tests/gpt5/run_phase353_family_contracts.sh；
tests/gpt5/phase354_semantic_time_case_bank.py；
tests/gpt5/phase354_semantic_time_contract_trace.py；
tests/gpt5/phase354_semantic_time_contract_analysis.py；
tests/gpt5/run_phase354_semantic_time_contract_trace.sh；
tests/gpt5/phase355_semantic_time_failure_audit.py；
tests/gpt5/result/phase353_family_contracts/；
tests/gpt5/result/phase354_semantic_time_contract_trace/；
tests/gpt5/result/phase355_semantic_time_failure_audit/。
```

### 七、通俗总结

这轮先给九类语言现象分别设计更合适的考试合同，18 份里只有 11 份在结构上合格。三模型真正同时稳定完成的只有 2 份，只有 3 份有资格观察内部过程。

随后把这 3 份任务的“按正确答案继续运行”和“模型自己生成”两条内部轨迹都画出来。修正空格和大小写造成的假词元偏离后，内部确实有一些重复方向，但换一组词，方向经常翻转；发现集里看到的区域也无法在校准集同时通过严格门槛。

因此当前最可靠的结论不是“找到了 67 条语言通路”，而是“找到了 67 个值得用于改进对照设计的近候选，同时确认它们还没有资格进入留出和因果实验”。下一步必须先把词汇、位置和词元边界真正配平，再谈神经元级闭合。

## Phase 356: 标签盲化粗脉络临摹可行性验证 [2026-07-11 06:21]

### 一、材料审计

“先临摹物理脉络、后解释功能、最后做因果”的方向正确，并且比继续围绕目标方向和高分组件打补丁更符合 Phase353-355 的负结果。但需要收紧：

1. 完全无预设不可能，只能保留层、生成步、组件、位置等最小原生预设；
2. 只保存范数和投影不能称为可恢复全迹；
3. 无标签重复结构可能只是公共架构骨架，不一定是语言模式；
4. 盲发现仍需独立校准、物理留出和因果验证；
5. 当前不能一次性记录所有案例的全部神经元，应采用完整骨架、平衡神经元分片、预注册全量锚点和因果深审四种分辨率。

数学材料的总体边界也正确：叠加、稀疏编码、概念格、张量绑定、预测状态和范畴组合分别解释候选部件，但没有统一解释真实大语言模型的知识、推理、语法和生成机制。范畴论可用于组织已发现对象和组合映射，不提供从权重自动提取神经函子的算法。

### 二、盲轨迹格式

流式读取 Phase354 的三模型轨迹。发现阶段只使用自由生成，教师强制轨迹不参与，避免正确答案注入。

盲骨架只保留：

```text
匿名案例编号；
发现/校准分割；
自然生成步和相对进度；
注意力输出、MLP 输出、残差增量；
早中晚相对深度；
来源、查询、答案起点、当前生成位置；
组件 L2 范数。
```

发现数据删除：

```text
模型名；
语言族；
机制名；
正确答案和干扰项；
操作条件；
词汇和模板标签；
目标与竞争方向；
实际词元身份；
历史候选。
```

原标签密钥单独保存，盲发现脚本不读取该目录。

### 三、基础脉络算法

对同一案例、组件和位置，保留所有早中晚深度槽，不做 Top-K。相邻值的基础趋势定义为：

$$
q(a,b)=
\begin{cases}
increase,&(b-a)/\max(|a|,\epsilon)>0.1\\
decrease,&(b-a)/\max(|a|,\epsilon)<-0.1\\
stable,&\text{其他}
\end{cases}
$$

构造两类匿名脉络：

```text
深度形状：早层 -> 中层 -> 晚层的趋势序列；
时间形状：上一生成步 -> 当前生成步在三个深度上的趋势序列。
```

发现门在读取数据前固定：发现集至少 24 个案例；校准集至少 8 个案例；校准出现率不低于发现出现率的一半。

### 四、结果

```text
注册案例：576；
原始轨迹：1192896；
自由生成轨迹：828480；
教师强制轨迹（未用于盲发现）：364416；
盲骨架：70704；
原始自由轨迹守恒：828480/828480；
非有限记录：0；
标签字段泄漏：0；
基础脉络分配：40224；
唯一脉络：117；
发现集冻结脉络：76；
盲校准重复脉络：68；
三模型发现/校准重复脉络：47；
图谱边：24。
```

事后揭示标签发现：47 种三模型重复脉络全部同时覆盖关系绑定、目标对错误候选和实体新近性，单机制脉络为 0。最高支持脉络主要是：

```text
注意力或 MLP 范数随深度普遍增加；
组件范数在相邻生成步保持稳定；
来源、查询和答案位置共享同类深度形状。
```

因此，盲算法首先恢复的是公共架构和尺度骨架，不是语言操作特异脉络。

### 五、严格结论

当前可以支持：

```text
标签隔离工程可行；
粗物理骨架可以在不读取功能标签时发现重复结构；
发现和校准分割可以按先冻结、后验证执行；
公共架构脉络在三个模型中大量复现。
```

不能支持：

```text
Phase356 已完成全迹；
47 种脉络是语言不变量；
盲重复等于机制；
任何物理留出、因果或单神经元结论。
```

Phase354 没有保存原始残差向量、注意力连接矩阵、归一化状态、头、通道和神经元，因此：

$$
FullTrace_{356}=False
$$

物理留出、因果封存和神经元搜索继续关闭。

## Phase 357: 三模型块级残差重构与数值精度校准 [2026-07-11 06:21]

### 一、目标和分母

在继续设计全向量轨迹前，先验证通用挂钩能否重构真实 Transformer 块更新。每个模型按与标签无关的固定哈希预注册 12 个发现案例：

$$
N_{anchor}=3\times12=36
$$

逐层记录：

```text
层输入；
注意力模块输出；
MLP 模块输出；
层输出。
```

理论块更新：

$$
h_{l+1}=h_l+\Delta A_l+\Delta M_l
$$

### 二、第一轮负结果和原因

第一轮先把半精度张量转换为单精度，再计算：

$$
(h_{l+1}-h_l)-(\Delta A_l+\Delta M_l)
$$

结果：

```text
qwen3 增量门：395/432；
GLM4 增量门：468/480；
DS7B 增量门：84/336；
跨模型重构：失败。
```

失败层在所有案例中固定重复。代码和模型前向过程审计发现，模型实际执行两次原生半精度残差加法；先转单精度会跳过两次真实舍入，DS7B 的 bfloat16 误差最明显。

### 三、原生精度重放

修正为：

$$
\widehat h_{l+1}
=
\operatorname{Add}_{dtype}
\left(
\operatorname{Add}_{dtype}(h_l,\Delta A_l),
\Delta M_l
\right)
$$

阈值没有放宽，案例没有更换。重新按 `qwen3 -> GLM4 -> DS7B` 串行执行。

结果：

```text
qwen3：432/432；
GLM4：480/480；
DS7B：336/336；
总重构：1248/1248；
形状不匹配：0；
非有限：0；
三个模型重构门：3/3。
```

所有原生精度重构误差为 0。块级挂钩位置和组件加法顺序得到确认。

### 四、边界和数学启示

Phase357 只证明：在三个当前小模型中，捕获的注意力输出和 MLP 输出按真实数据类型顺序相加，可以精确恢复块输出。它不证明：

```text
注意力来源边已经记录；
归一化状态已经记录；
头、通道或神经元已完整覆盖；
公共块更新就是语言脉络；
任何脉络具有因果作用。
```

本阶段对理论最重要的提醒是：真实运行机制不仅包含抽象代数结构，还包含执行数据类型、舍入顺序和接口时点。若理论公式忽略这些物理细节，数学等式正确也可能与真实运行轨迹不一致。

### 五、数学理论证据层级

当前数学材料应分为：

1. 已有定义或受控定理：稀疏组合容量、叠加玩具模型、概念格、张量绑定、范畴和函子；
2. 受控模型或表示几何证据：稀疏自编码器、WordNet 层级几何、共现谱、隐马尔可夫信念状态；
3. 当前项目假说：动态概念格、分布式编码、绑定、状态转移、读出和机制等价关系的统一对象。

$$
\mathfrak K=(L,\mathcal R,E,B,F,U,\sim)
$$

该式目前只是研究接口，不是理论闭合。范畴论只有在实验先发现稳定状态和可组合事件后，才适合表达跨模型映射：

$$
\Phi(g\circ f)\simeq\Phi(g)\circ\Phi(f)
$$

$\simeq$ 必须由跨模型复现、未来轨迹预测和因果干预共同定义，不能由余弦相似度或可视化形状定义。

### 六、下一阶段大任务

下一阶段不是扩大粗范数脉络数量，而是设计重构有效的多分辨率全迹格式：

1. 所有案例保存可验证的块级残差骨架和数值精度元数据；
2. 保存注意力来源—接收边和归一化时点，并分别做守恒审计；
3. 用与任务无关的固定分片覆盖全部注意力头、MLP 通道和神经元；
4. 预注册少量全向量锚点，验证分片轨迹没有系统遗漏；
5. 再运行盲发现，先分离公共架构脉络，之后寻找剩余的操作特异脉络；
6. 只有独立物理留出复现后才进入因果验证。

当前应该停止自动扩大模型样本。Phase357 已完成本阶段可行性目标，下一步涉及新的高容量数据格式和存储预算；在格式、精度和重构门冻结前运行大样本会产生无法恢复的数据。

### 七、阶段产物

```text
tests/gpt5/phase356_blind_trace_schema.py；
tests/gpt5/phase356_blind_motif_discovery.py；
tests/gpt5/phase356_posthoc_motif_validation.py；
tests/gpt5/run_phase356_blind_neural_path_cartography.sh；
tests/gpt5/phase357_residual_reconstruction_audit.py；
tests/gpt5/phase357_residual_reconstruction_analysis.py；
tests/gpt5/run_phase357_residual_reconstruction_audit.sh；
tests/gpt5/result/phase356_blind_neural_path_cartography/；
tests/gpt5/result/phase357_residual_reconstruction_audit/。
```

### 八、通俗总结

这轮先把任务名字、答案和研究者标签全部遮住，只让算法看模型自然运行时各组件强弱怎样随层和生成步变化。算法确实找到了 47 种三个模型都重复的形状，但它们在三类任务中同样常见，主要说明网络共有“越到后层组件范数越大”等公共骨架，还不是语言规律。

然后检查记录工具是否真的把每层计算记全。第一次看起来 DS7B 大量失败，最后发现不是模型多了未知组件，而是审计器把半精度数据提前转成单精度，改变了真实加法。按模型原精度重放后，三个模型所有 1248 个层更新都能精确恢复。

所以新路线已经证明可以实施，但真正全迹还没有完成。现在有了可信的块级地基，下一步才是增加注意力连接、归一化、头、通道和神经元分片，而不是把 47 种公共形状包装成语言编码机制。

## Phase 358: 三模型多分辨率组件守恒账本 [2026-07-11 07:14]

### 一、对 Phase356-357 后续方案的审计

“先临摹、后归纳、再因果”的方向正确，Phase356 证明了标签盲化流程可执行，Phase357 证明了块级记录接口可重构。但原方案中以下说法需要收紧：

```text
盲流程可执行，不等于已经发现语言脉络；
块级守恒，不等于注意力头、MLP 通道和神经元已经记全；
三模型格式通过，不等于九族均已具备测试资格；
没有统一证据分母时，不能报告单一“总体进度 24%”。
```

因此本阶段没有开启物理留出和因果封存，只验证全迹格式所需的五类组件账本。

### 二、组件记录与重构公式

每层捕获：层输入、输入归一化状态、注意力输出投影输入、注意力概率、注意力模块输出、后注意力归一化状态、MLP 降维投影输入、MLP 输出和层输出。

注意力头输出分解为：

$$
\widehat{\Delta A}_l
=
\sum_{h=1}^{H_l}
W^{O}_{l,h}z_{l,h}+b^O_l
$$

MLP 全通道按与任务无关的固定哈希分成 16 片：

$$
\widehat{\Delta M}_l
=
\sum_{k=1}^{16}
W^{down}_{l,S_k}a_{l,S_k}+b^{down}_l,
\qquad
\bigcup_{k=1}^{16}S_k=\{1,\ldots,I_l\}
$$

块输出按原生数据类型和真实顺序重放：

$$
\widehat h_{l+1}
=
\operatorname{Add}_{dtype}
\left(
\operatorname{Add}_{dtype}(h_l,\Delta A_l),
\Delta M_l
\right)
$$

注意力概率归一化门为：

$$
\max_{l,h,q}
\left|
\sum_s P_{l,h,q,s}-1
\right|
\le 10^{-2}
$$

所有组件相对重构误差门保持为 $10^{-2}$，没有因模型不同放宽。

### 三、执行与结果

按 `qwen3 -> GLM4 -> DS7B` 串行执行。

格式开发集：

```text
三模型案例：6；
层账本：208；
注意力头账本：6432；
MLP 分片账本：3328；
块、注意力、注意力概率、MLP、归一化五门：全部通过。
```

随后自动扩展到盲发现和盲校准格式子集：

```text
盲发现案例：9；
盲校准案例：3；
扩展层账本：416；
扩展匿名多视图行：416；
五类守恒门：三个模型全部通过；
物理留出：未开启；
因果干预：0；
单神经元因果：0。
```

这里的“匿名多视图行”仍然只含层、范数和重构误差等摘要，不是盲模式发现结果。Phase358 证明的是记录格式和组件守恒可以扩展，不是语言操作脉络已经出现。

## Phase 359: 存储预算冻结与三模型全向量锚点回放 [2026-07-11 07:14]

### 一、存储预算负结果

在扩大样本前，按每模型 18 个机制、每机制 8 例、教师强制与自由生成共 8 次轨迹、规划序列长度 256 估算。直接保存九类状态、MLP 通道、注意力矩阵、逐头投影贡献和 16 个 MLP 分片贡献：

$$
B_{naive}
=
\sum_m CP
\left(
18L_mTH_m
+2L_mTI_m
+2L_mA_mT^2
+4L_mA_mTH_m
+4L_mKTH_m
\right)
$$

其中 $C=144$，$P=8$，$K=16$。三模型估算结果：

```text
直接全量全向量保存：22924052398080 字节，约 22.9 TB；
运行时可用磁盘：1379268964352 字节，约 1.38 TB；
保留 200 GiB 安全空间后：不可执行。
```

分层方案冻结为：

```text
R0：全案例只保存 4 个预注册位置的组件状态与头/分片标量账本；
R1：按固定哈希平衡分配，每案例保存一个原始 MLP 分片；
R2：每模型先保存一个密封全向量锚点；
R3：物理留出和因果封存继续关闭。
```

保守估算约 50.37 GB，可执行。原始全向量文件不复制到可视化客户端。

### 二、密封锚点与离线回放

每个模型选取一个与语言标签无关的预注册格式锚点。每层单独保存原生组件张量、全部头投影贡献、全部 MLP 分片贡献、通道编号、概率和归一化状态；文件保存后释放模型，再在不加载模型权重的条件下逐层回放。

```text
Qwen3：36 层，1038726252 字节；
GLM4：40 层，1578007160 字节；
DS7B：28 层，933673300 字节；
合计：104 层文件，3550406712 字节。
```

文件哈希、注意力头求和、MLP 分片求和、块重构、概率归一化和归一化有限性六门全部通过。最大相对误差：

```text
Qwen3：注意力 0.0003241，MLP 0.0003252，块 0；
GLM4：注意力 0.0003293，MLP 0.0003380，块 0；
DS7B：注意力 0.0024703，MLP 0.0024991，块 0。
```

DS7B 误差较大但仍低于预注册的 $10^{-2}$ 门，可能与 bfloat16 量化更粗有关。三者都是当前小模型的格式校准，不能外推为大型模型拥有相同细粒度编码。

## Phase 360: 九族十八机制总分母冻结与准入审计 [2026-07-11 07:14]

### 一、为什么不能直接运行九族全迹

Phase353 已注册九族十八机制，但注册不等于对照合同合格，更不等于三模型行为合格。统一审计得到：

```text
注册语言族：9/9；
注册机制：18/18；
结构合同通过：11/18；
完成三模型行为测量：11/18；
三模型盲轨迹准入：3/18；
三模型完整行为门通过：2/18；
物理留出：0/18；
因果封存：0/18。
```

当前获准进入盲发现的只有：

```text
content_knowledge / relation_binding；
readout_competition / target_vs_wrong；
state_drift / entity_recency。
```

Phase358 的 12 个扩展案例也只来自这三族三机制。因此不能把 Phase358 表述成九族全图谱测试，也不能用三族的守恒结果填充其余六族。

### 二、七个结构合同负结果

以下合同在读取模型效应之前就未通过固定结构门：

```text
reasoning_constraint：missing_condition、two_hop_entailment；
syntax_structure：past_tense_sentence、number_agreement；
language_action：case_transform、field_extract；
cross_lingual：translation。
```

主要可见问题是目标/控制提示词汇重叠过低，`number_agreement` 和 `translation` 还存在目标可见性不匹配。此结果不能用降低门槛修补；需要重新设计最小对照，再从合同资格开始独立运行。

另外 8 个合同虽然结构合格，但没有获得三模型共同轨迹准入，应保留为图谱中的负单元，除非存在独立于现有结果的重设计理由。

### 三、严格结论与理论边界

当前新增的客观拼图是：

$$
\text{完整块更新}
=
\text{全部注意力头投影之和}
+
\text{全部 MLP 通道分片投影之和}
$$

该等式已在三个模型的密封锚点上实现离线可回放，但它仍是计算守恒式，不是语言编码公式。它没有给出：

```text
哪些头、通道或神经元形成知识、推理、语法脉络；
脉络如何跨时间组合；
同一操作如何跨模型对应；
干预后为何产生特定语言行为。
```

所以统一智能对象

$$
\mathfrak K=(L,\mathcal R,E,B,F,U,\sim)
$$

仍只能作为待填充的研究接口。现阶段不应引入新的高阶数学名称，也不应把线性加和守恒误认为真实语言机制是线性的。

### 四、下一阶段大任务

下一阶段应整体完成“九族合同修复与准入冻结”，而不是继续追加一个候选神经元：

1. 在不读取模型效应的条件下重写 7 个被拒合同，保持目标、模板、可见性和词元长度配平；
2. 依次在 Qwen3、GLM4、DS7B 上完成修复合同的全分割行为资格测试；
3. 对原 8 个行为负合同保持负状态，不因希望扩大图谱而降低标准；
4. 冻结 18 个机制的“准入、拒绝、待修复”矩阵；
5. 只对准入单元运行 R0/R1 平衡盲记录，再做独立校准；
6. 发现操作特异且可预测未来轨迹的脉络后，才开启物理留出和因果封存。

### 五、阶段产物

```text
tests/gpt5/phase358_multiresolution_component_conservation.py；
tests/gpt5/phase358_multiresolution_component_analysis.py；
tests/gpt5/phase358_expanded_ledger_analysis.py；
tests/gpt5/phase359_storage_budget.py；
tests/gpt5/phase359_full_vector_anchor.py；
tests/gpt5/phase359_full_vector_replay.py；
tests/gpt5/phase360_denominator_freeze.py；
tests/gpt5/result/phase358_multiresolution_full_trace/；
tests/gpt5/result/phase359_full_vector_anchor/；
tests/gpt5/result/phase360_denominator_freeze/。
```

### 六、通俗总结

这轮把“全迹”从口号变成了可检查的文件：模型每一层的输入、归一化、注意力头、MLP 通道分片和输出都能保存下来，模型卸载后仍能重新相加并还原原结果。这个工具地基是真进展。

但九族图谱还不能全量开跑。现在只有 3 个机制具备三模型共同准入，7 个机制的对照合同本身需要重做，其余多个机制是行为负结果。继续盲目扩大记录会得到昂贵而无法解释的数据。正确的下一步是先把九族对照资格修好并冻结，再让临摹算法只看合格数据。

## Phase 361: 七合同修复、四机制分层临摹与盲预测校准 [2026-07-11 08:22]

### 一、对 Phase358-360 审计文本的判断

审计文本对工程地基和科学边界的判断总体正确：组件守恒、密封回放和存储预算不等于九族语言图谱已经开始，更不等于语言机制闭合。提出“合同修复 -> 行为资格 -> 准入冻结 -> 分层临摹 -> 公共骨架 -> 预测”的顺序也正确。

需要修正两点：

1. 用人为权重把不同证据层加成约 18%，只能作为项目管理偏好，不能作为客观科学进度；本阶段继续使用多轴分母；
2. 合同修复、盲预测和机制发现不能合并成一个成功门。每一门必须串行通过，失败时不得自动进入下一层。

### 二、七个合同的离线重构

只处理 Phase360 中结构拒绝的七个机制：缺失条件、两跳蕴含、过去时句、数一致、大小写转换、字段抽取和翻译。没有读取模型激活、历史候选或模型行为来设计合同。

固定合同门为：

$$
G_{contract}
=
T\land V\land L\land P\land Q\land X
$$

其中：

```text
T：目标相同；
V：目标可见性相同；
L：目标词元长度相同；
P：答案位置角色相同；
Q：提示词元长度差不超过 24；
X：上下文与问题的词汇重叠不低于 0.65。
```

第一轮结果为 5/7。字段抽取重叠 0.6452，翻译重叠 0.5839，模型执行没有启动。随后只统一这两个合同的问题表面框架，目标、案例、阈值和操作差异不变。第二轮七个合同全部通过，最低重叠为翻译的 0.6631。

固定案例分母：

$$
N=7\times 12\times2\times2\times2\times3=2016
$$

对应每模型 672 例，覆盖发现、校准、行为留出和行为封存；内部轨迹留出和因果封存继续未开启。

### 三、三模型行为资格

按 `Qwen3 -> GLM4 -> DS7B` 串行运行，全部采用单样本、左填充、关闭缓存和相同生成上限。

```text
Qwen3：672/672 有效，自然输出正确 502；
GLM4：672/672 有效，自然输出正确 514；
DS7B：672/672 有效，自然输出正确 434；
非有限短语分数：0；
换题或降低阈值：0。
```

七个修复机制中：

```text
新增三模型发现/校准准入：number_agreement（数一致）1/7；
新增三模型完整行为门：0/7；
两跳、过去时、大小写和字段抽取仅有 2/3 模型通过轨迹门；
缺失条件和翻译为 0/3；
其余六个机制冻结为行为负结果。
```

合并 Phase360 后，十八机制状态为：盲轨迹准入 4/18，完整行为门 2/18。新增准入不表示数一致机制已被解释，只表示它可以进入自然轨迹比较。

### 四、R0/R1 分层临摹

只对四个准入机制运行。每模型每机制固定 8 例，共 96 例；A/B/C/D 四类条件各 24 例，发现 72 例、校准 24 例。16 个 R1 MLP 分片各分配 6 例，没有用功能标签选择分片。

R0 保存四个角色位置上的九类状态、全部注意力头标量、注意力来源概率行、全部 MLP 分片标量和全词表竞争向量。R1 每例保存一个固定分配的原始 MLP 分片。

首次质量审计发现：来源和查询角色精确匹配均为 0/3328，旧算法使用了回退位置。因此首次记录只保留为工具负结果，未进入脉络分析。原因是聊天模板中的换行与前一标点合并成同一个词元，独立编码完整片段无法匹配。

修正为：先在完整聊天文本中定位字符片段，再对“片段末尾加一个边界字符”的完整前缀编码，从前缀词元后缀反查真实位置。三模型 96 例离线位置检查全部通过后，重新运行全部记录。

正式结果：

```text
案例：96；
层账本：3328；
密封原始数据：1289962528 字节；
来源、查询、答案起点、当前生成位置：各 3328/3328 精确；
标签泄漏行：0；
块重构最大误差：0；
注意力最大相对误差：0.003623889；
MLP 最大相对误差：0.005293302；
输入归一化最大误差：0.002559967；
后注意力归一化最大误差：0.002582695；
注意力概率最大误差：0.003417969。
```

所有误差低于冻结的 $10^{-2}$ 门。与 Phase358 相比，本阶段第一次对 RMSNorm/LayerNorm 的数值输出做了显式重放，而不是只检查有限性。

### 五、公共骨架和下一层预测

盲阶段不读取模型名、语言族、机制、答案和对照条件。对每个匿名模型、深度区间和状态特征，发现集建立当前层中位值、下一层中位值及其变化：

$$
\Delta B_{m,d,f}
=
\operatorname{Median}(x_{l+1,f})
-
\operatorname{Median}(x_{l,f})
$$

公共基线预测为：

$$
\widehat x^{base}_{l+1,f}
=
\operatorname{Median}(x_{l+1,f})
$$

状态连续性预测为：

$$
\widehat x^{state}_{l+1,f}
=x_{l,f}+\Delta B_{m,d,f}
$$

只在盲校准集比较归一化绝对误差。结果：

```text
公共骨架单元：324；
校准转移特征：29088；
公共基线误差：0.516760962；
状态连续性误差：0.217729627；
预测增益：0.299031335；
三模型均为正的冻结候选：93/108。
```

93/108 过于普遍，首先说明相邻层状态具有强连续性，而不是发现了 93 个语言操作。

### 六、候选冻结后的事后功能审计

候选完全冻结后才读取私有标签。24 个校准案例中，每模型每机制只有 2 个独立案例。

```text
四机制普遍为正：77；
只在部分机制为正：16；
无支持：0；
操作特异机制：0。
```

16 个选择性候选只能登记为关联：独立案例太少，且同一案例的多层转移不能作为独立样本。当前也只测试了下一层，没有测试下一生成步、候选词变化或自然输出分叉。

### 七、严格结论和硬伤

本阶段新增的可信拼图：

1. 七个结构合同可以在不看模型效应时完成配平，但行为资格仍会淘汰 6/7；
2. 四个准入机制的 R0/R1 数据可以在三模型上守恒记录；
3. 来源和查询词元跨度必须在完整聊天模板内定位，旧的片段独立编码回退会污染位置图谱；
4. 相邻层状态连续性显著优于只用公共层基线；
5. 大多数预测候选跨四机制普遍存在，主要是公共骨架；少量选择性关联尚不能解释为语言机制。

仍存在的硬伤：

```text
每模型每机制只有 2 个独立校准案例；
只记录生成前时间，未记录自然生成时间链；
注意力只保存来源概率，没有保存来源值向量经输出投影后的边贡献；
每例一个 MLP 原始分片不能观察跨分片共现；
当前模型较小，编码粗糙性可能放大跨模型失败；
物理留出、因果封存、神经元因果和机制闭合仍为 0。
```

附件给出的单一约 18% 进度不采用。当前客观轴为：合同结构 18/18 已获得接受或拒绝结论，三模型盲轨迹准入 4/18，完整行为 2/18，四准入机制首批 R0/R1 记录完成，操作特异预测机制 0，物理留出 0，因果闭合 0/72。

### 八、下一阶段大任务

下一阶段应为 Phase362：独立校准扩容与生成时间全迹。必须整体完成：

1. 从未使用的校准题项中扩大每模型每机制独立案例，不复用本阶段 24 例；
2. 为短序列、长序列和多步生成各建立至少一个密封锚点/模型；
3. 记录自然生成前、首词元后和第二词元后的状态，检验下一生成步而不只下一层；
4. 增加来源值向量经头输出投影后的边贡献守恒；
5. 增加少量双分片和四分片共现锚点，估计 MLP 跨分片联合结构；
6. 用冻结的 93 个预测候选做独立校准，不重新选择；
7. 只有跨模型、跨独立案例、跨生成时间仍有选择性的候选，才能进入物理留出。

### 九、通俗总结

这一轮先把七类不合格题目重做成真正可比较的对照，三个模型完整回答了 2016 个案例。最终只有“数一致”新增了三模型共同准入，说明合同看起来工整不代表模型行为足够稳定。

随后记录四个合格机制的内部状态。第一次发现来源和问题位置其实都找错了，工具只是悄悄用了估计位置；修正聊天模板内的词元定位并重跑后，四个位置才全部精确。盲算法确实能用当前层更好地预测下一层，但 93 个正候选里 77 个在四种任务中都存在，主要是模型共有的连续运行骨架。剩下 16 个目前只是小样本关联，还不是语言编码规律。

## Phase 362: 独立校准、注意力来源边与生成时间可识别性审计 [2026-07-11 09:03]

### 一、对 Phase361 审计文本的判断

审计文本对 Phase361 的核心定位正确：真正进展是合同准入冻结、聊天模板内词元位置纠错和下一层公共连续性基线，而不是 93 个候选本身。下一阶段扩大独立案例、增加来源边守恒和记录生成时间的方向也正确。

需要补充一个决定性限制：Phase361 候选实际只冻结了：

```text
深度区间；
状态特征；
下一层公共增量公式；
三个匿名模型上的下一层预测增益。
```

没有冻结从候选到下一生成步、竞争变化或自然分叉的预测公式。因此 Phase362 可以检验旧候选的下一层外推，不能在不新增规则的条件下检验旧候选的生成时间预测。审计文本假定“作用方向和全部预测公式已经冻结”，与真实候选文件不一致。

审计文本再次给出的管理百分比不采用。人为权重没有从证据分母自然导出，不能作为科学进度。

### 二、独立案例分母冻结

目标分母为：

$$
N=3\times4\times8\times4=384
$$

但跨三个模型共同完全未使用的语义组不足 8 个：关系绑定只有 4 个，目标竞争只有 3 个。为避免伪造共同分母，改为在每个“模型 × 机制”单元内排除任何 Phase361 已用提示，再固定选择 8 个完整 A/B/C/D 组。

```text
独立校准：72 组，288 个提示；
物理确认封存：24 组，96 个提示；
与 Phase361 精确案例重叠：0；
Phase361 候选数：93；
候选文件 SHA256：dfecaea04bda525177f2c4c6cc7a9b009a0fa9b8d8698df5fad7ac0d8698e6c0。
```

物理确认提示本阶段没有运行。

### 三、九锚点与注意力来源边守恒

每模型固定短序列、长序列和多步生成三个锚点。短、长锚点记录一个时刻，多步锚点记录生成前、首词元后和第二词元后，共 9 个锚点、15 个锚点时刻。

对每个头、接收位置和来源位置记录：

$$
e^A_{l,h,q\leftarrow s}
=
W^O_{l,h}\left(P_{l,h,q,s}V_{l,h,s}\right)
$$

验证：

$$
\sum_s e^A_{l,h,q\leftarrow s}
\approx o_{l,h,q}
$$

以及：

$$
\sum_h\sum_s e^A_{l,h,q\leftarrow s}+b_l^O
\approx\Delta A_{l,q}
$$

短锚点保存两个 MLP 共现分片，长锚点保存四个，多步锚点保存全部十六个。按 `Qwen3 -> GLM4 -> DS7B` 串行运行并卸载模型后离线回放。

```text
锚点：9；
锚点时刻：15；
层文件：520；
密封数据：6928904360 字节；
全部在线门：通过；
全部离线门：通过；
来源边最大相对误差：0.001738097；
注意力最大相对误差：0.003180101；
MLP 最大相对误差：0.004180492；
块误差：0；
输入归一化最大误差：0.002922072；
后归一化最大误差：0.003006373；
概率最大误差：0.003417969。
```

来源边格式第一次实现可离线重构，但边守恒只证明账本完整，不证明该边承载某种语言功能。

### 四、三模型生成时间轨迹

只运行 288 个独立校准提示，每例记录：生成前 $t_0$、首词元后 $t_1$、第二词元后 $t_2$。每个时刻保存四个精确角色位置的九类状态、注意力熵、一个平衡 MLP 原始分片和完整词表 logits（对数几率前值）。

```text
Qwen3：96 例，10368 层时刻行，2855794656 字节；
GLM4：96 例，11520 层时刻行，4565462880 字节；
DS7B：96 例，8064 层时刻行，3338992704 字节；
合计：288 例，29952 层时刻行，10760250240 字节；
三模型质量门：全部通过。
```

这些数据证明生成时间格式可执行，不自动产生时间机制。

### 五、冻结候选与四类强基线

Phase361 发现集重新拟合四类基线参数，Phase362 全新 72 组只做测试：

$$
B_0=\operatorname{Median}(x_{l+1})
$$

$$
B_1=x_l
$$

$$
B_2=ax_l+b
$$

$$
B_3=x_l+\operatorname{Median}(x_{l+1}-x_l)
$$

Phase361 候选对应 $B_3$。只有 $B_3$ 在三个模型上都严格优于 $B_0$、$B_1$、$B_2$ 时，才保留为独立下一层幸存者。

结果：

```text
冻结候选：93；
三模型独立强基线幸存者：7；
关闭：86。
```

7 个幸存者包括早层查询位置后注意力残差、中层答案起点/当前生成位置注意力输出、中层来源归一化、晚层来源/查询注意力输出和晚层来源注意力投影输入。部分增益非常小，例如 GLM4 中层来源归一化仅比最强基线好 0.000026983，因此不能按“严格大于零”直接解释为强机制。

### 六、冻结后机制审计

7 个幸存者冻结后才读取机制标签。每模型每机制有 6 个独立组：

```text
四机制普遍为正：0；
部分机制三模型为正：6；
四机制均不稳定：1；
生成时间预测幸存者：0；
竞争预测幸存者：0；
操作特异机制：0。
```

例如中层答案起点注意力输出只在实体新近性中三模型均为正；中层当前生成位置注意力输出在目标竞争和数一致中为正。它们是下一层公式的机制选择性关联，不是时间或行为机制。

### 七、不可识别性负结果

当前可以客观计算候选特征在 $t_0,t_1,t_2$ 的变化，但不能声称旧候选预测这些变化，因为旧候选没有冻结：

```text
时间转移公式；
竞争变量定义；
竞争预测映射；
分叉预测映射。
```

如果根据 Phase362 时间结果事后为 7 个幸存者补充最有利公式，再用同一 288 例报告成功，会构成二次选择偏差。故物理确认 96 例继续密封。

### 八、严格结论和硬伤

本阶段新增的可信拼图：

1. 来源概率和值向量经过头输出投影后，可以逐来源重构头输出；
2. 三模型能够在三个自然生成时刻稳定记录相同组件接口；
3. Phase361 的 93 个宽松候选在强基线和全新案例下只剩 7 个；
4. 其中 6 个有下一层机制选择性，但没有任何候选具备预注册时间或竞争预测资格；
5. “数据已经记录”与“旧候选可以回答该数据上的新问题”是两个不同证据门。

仍存在的硬伤：

```text
7 个幸存者的部分强基线增益接近数值噪声尺度；
时间公式和竞争公式缺失；
当前来源边只在 9 个锚点上完整保存；
批量时间轨迹仍是每例一个 MLP 原始分片；
跨分片联合结构只在少量锚点出现；
物理确认、因果干预和单神经元验证仍为 0；
小模型结果不能外推到更大模型。
```

### 九、下一阶段任务

下一阶段应作为新的 Phase363，而不是给 Phase361 候选打补丁：

1. 只把本阶段 288 例作为时间假设发现集；
2. 预先定义时间创新、完整词表竞争和自然分叉变量；
3. 为候选特征到这些目标冻结明确预测公式和方向；
4. 冻结候选数量、阈值及模型一致性门；
5. 再运行当前未开启的 96 个物理确认提示；
6. 未通过者关闭，不返回本阶段重新选公式；
7. 只有跨三模型、跨机制对照和跨生成时间均通过者，才登记为物理状态候选；仍不进行因果干预。

### 十、通俗总结

这轮把 93 个候选放到更多没见过的题目上，并加入了比原来更强的比较方法。结果只剩 7 个还能稍好地预测下一层，其中 6 个在部分任务中表现不同。与此同时，项目第一次把注意力“从哪个位置搬运过来”的贡献按来源加回去，并在模型卸载后验证账目正确。

但 93 个旧候选从来没有规定怎样预测下一个生成词元或竞争变化。现在虽然已经记录了三个生成时刻，也不能事后挑一个最有效的时间公式然后宣称旧候选成功。正确做法是先用这些新数据形成一组新的、写清公式的时间假设，再用尚未打开的 96 个提示做真正验证。

## Phase 363: 生成时间创新与词元竞争公式严格冻结，关闭七候选路线 [2026-07-11 09:35]

### 一、阶段目标与 Phase362 审计

本阶段先审计 Phase362 的解释，再使用 Phase362 已完成的 288 个独立发现提示检验七个下一层幸存候选。没有运行新的 CUDA（统一计算设备架构）模型推理，也没有读取或运行 96 个物理确认提示。

Phase362 的总体方向正确，尤其是以下四点：

```text
组件账本守恒、下一层预测和下一生成时刻预测必须分门验证；
旧候选不能因新增时间数据而自动获得时间解释；
独立组是分析单位，层、位置和 A/B/C/D 条件不是独立样本；
发现公式未通过以前，物理确认集必须保持密封。
```

但原分析需要收紧四处：

```text
Phase361 的 93 个候选只冻结了下一层公共增量关系，没有冻结时间、竞争或分叉映射；
当前封存日志只能严格恢复逐词元竞争，不能恢复完整短语联合边距和语义分叉；
预测增益大于零不够，必须超过自然条件与模板波动；
工程完成度、行为准入率、物理留出率和因果闭合率不能合成一个总体百分比。
```

### 二、数据分母与密封边界

正式发现分母为：

```text
模型：Qwen3、GLM4、DeepSeek7B，共 3 个；
旧候选输入：7 个；
发现提示：288 个；
独立组：72 个，每组 4 个 A/B/C/D 条件；
公式训练组：48 个；
公式验证组：24 个；
密封物理确认提示：96 个，读取数和运行数均为 0；
时间创新公式：14 个；
词元竞争变化公式：6 个；
总公式：20 个。
```

Phase361 候选文件的 SHA-256（安全散列算法第二代 256 位）摘要继续固定为：

```text
dfecaea04bda525177f2c4c6cc7a9b009a0fa9b8d8698df5fad7ac0d8698e6c0
```

每个模型和机制的 6 个独立组按固定哈希切分为 4 个公式训练组和 2 个公式验证组。条件、层和位置只在组内聚合，不进入独立分母。

### 三、严格时间创新目标

第一版可执行草稿直接预测下一时刻原始状态，并使用跨组原始状态波动作为效应地板。运行后审计发现这不完全符合预注册的“时间创新”定义，而且会把不同机制的状态尺度差异混入噪声。该草稿结果保留在 `frozen_temporal_formulas` 目录作为审计记录，但不能作为正式结论。

严格版本先在公式训练组拟合单变量时间持续基线：

$$
\widehat{x}^{\mathrm{time}}_{t+1,m,j}
=
a_{m,j}x_{t,m,j}+b_{m,j}
$$

然后定义候选状态的时间创新：

$$
\xi_{t,m,j}
=
x_{t+1,m,j}-\widehat{x}^{\mathrm{time}}_{t+1,m,j}
$$

七候选的组级输入为：

$$
z_{g,t,j}
=
\frac{1}{4}
\sum_{c\in\{A,B,C,D\}}
\frac{1}{|L_j|}
\sum_{l\in L_j}
\lVert h_{g,c,t,l,j}\rVert_2
$$

其中，候选冻结了组件类型、相对深度区间和位置角色。正式低容量联合公式为固定正则线性映射：

$$
\widehat{Y}_{g,t+1}
=
w^\top
\left(
\frac{z_{g,t}-\mu}{s}
\right)
+b
$$

正则目标为：

$$
\min_{w,b}
\sum_g
\left(
Y_g-\widehat{Y}_g
\right)^2
+
10^{-3}\lVert w\rVert_2^2
$$

时间创新比较的强基线包括零创新、全局中位创新和同机制中位创新。联合公式必须优于其中误差最低者。

### 四、可识别的词元竞争目标

当前每个时点保存了完整词表 logits（未归一化词元分数），因此可以严格计算：

$$
C_t
=
\left[
M_t^{\mathrm{token}},
\log(1+R_t^{\mathrm{target}}),
H_t^{\mathrm{vocab}}
\right]
$$

其中：

$$
M_t^{\mathrm{token}}
=
\ell_t(y_t^*)-
\max_{d\in\mathcal D}\ell_t(d_t)
$$

$$
H_t^{\mathrm{vocab}}
=
-\sum_v p_t(v)\log p_t(v)
$$

竞争预测目标为：

$$
\Delta C_t=C_{t+1}-C_t
$$

竞争基线包括零变化、全局中位变化、同机制中位变化、当前目标排名单变量公式和当前词表熵单变量公式。

现有轨迹不能严格识别完整短语联合边距、继续与停止协议边距及语义分叉，因此这些字段被明确登记为不可识别，没有用近似字段冒充。

### 五、实际效应地板

没有重复运行数据，故本轮不能估计重跑噪声。可识别的两个自然波动项为：

$$
\sigma_{\mathrm{case}}
=
\operatorname{median}_{g,c}
\left|
Y_{g,c}-\operatorname{median}_{c'}Y_{g,c'}
\right|
$$

$$
\sigma_{\mathrm{template}}
=
\operatorname{median}_{k,g}
\left|
\overline Y_{k,g}-
\operatorname{median}_{g'}\overline Y_{k,g'}
\right|
$$

本轮效应地板固定为：

$$
\sigma_{\mathrm{floor}}
=
\max
\left(
\sigma_{\mathrm{case}},
\sigma_{\mathrm{template}}
\right)
$$

联合公式相对最强基线的增益为：

$$
PG
=
\operatorname{MAE}_{\mathrm{best\ baseline}}
-
\operatorname{MAE}_{\mathrm{joint}}
$$

严格发现门为：

$$
G_{363}^{\mathrm{discovery}}
=
\bigwedge_{m\in\{Q,G,D\}}
\left[
PG_m>\sigma_{\mathrm{floor},m}
\right]
$$

三个模型全部通过后才允许冻结公式并开启 96 个物理确认提示。

### 六、客观结果

20 个公式的结果为：

```text
三模型均为正增益：1/20；
三模型均超过实际效应地板：0/20；
Qwen3 单模型超过效应地板：0/20；
GLM4 单模型超过效应地板：0/20；
DeepSeek7B 单模型超过效应地板：2/20；
冻结公式：0；
物理确认运行：0/96；
因果干预：0。
```

唯一在三个模型中都出现正增益的公式是：

```text
t0 到 t1；
中层 attention_output（注意力输出）；
current_generation（当前生成位置）；
预测时间创新。
```

但它在三个模型中的增益与效应地板分别为：

```text
Qwen3：增益 0.047757285，地板 0.230964215；
GLM4：增益 0.018559133，地板 0.038911450；
DeepSeek7B：增益 0.000683141，地板 0.607623327。
```

三个增益都没有越过自然波动，因此不能登记为生成时间物理状态候选。

DeepSeek7B 的两个局部通过项分别涉及中层当前生成位置注意力输出和晚层来源位置注意力输出。它们在另外两个模型中没有通过，故只能保留为模型特异局部信号，不能进入统一机制或智能理论主体。

### 七、阶段结论与路线关闭

Phase361/362 七候选路线的严格结果是强负结果：

$$
7\ \text{个下一层幸存候选}
\not\Rightarrow
\text{可复用的时间创新或竞争状态}
$$

这并不证明网络没有动态语言状态，只证明当前七个“组件范数 × 深度区间 × 位置角色”静态汇总量不足以承担该状态。按照预注册停止规则：

```text
关闭围绕七候选继续增加线性参数、窗口或阈值的路线；
把它们登记为公共连续性、弱选择性关联或模型特异局部信号；
96 个物理确认提示继续封存；
不进行针对七候选的必要性干预；
不把本轮结果写成语言编码机制或智能理论闭合。
```

### 八、图谱同步与客观进度

结果已同步到研究图谱和可视化客户端，前端只导出摘要与公式行，没有导出 `.pt` 张量。

当前可比较的分母进度为：

```text
语言模式族登记：9/9；
机制登记：18/18；
盲发现准入：4/18，约 22.2%；
严格时间与竞争公式：0/20；
物理留出机制：0/18；
因果封存机制：0/18；
严格机制闭合：0/72。
```

`9/9` 和 `18/18` 只表示目录与合同分母已经建立，不表示物理路径已经完成。项目不存在科学有效的单一总体百分比。

### 九、问题、硬伤与小模型边界

```text
每个模型每个机制只有 6 个独立组，公式训练 4 组、验证 2 组，独立组数量仍偏少；
没有重跑样本，效应地板缺少 sigma_rerun；
当前候选是深度区间内组件范数均值，不保留方向、相位、神经元组合和跨层联合结构；
词元竞争只覆盖三个可识别连续量，不是完整短语、协议和语义竞争；
四个准入机制不能代表全部九族和 18 个机制；
Qwen3、GLM4 和 DeepSeek7B 均为小模型，内部路径可能更粗糙或模型特异；
跨模型负结果不能直接推出更大模型中不存在对应机制；
没有自然必要性、充分性和单神经元 CUDA 干预证据；
语言编码机制和智能理论均未闭合。
```

### 十、对智能理论的有限更新

本轮只允许增加一个负约束，不修改理论名称：

$$
\text{局部层间连续性}
\neq
\text{生成时间功能状态}
$$

更可能需要研究的对象不是单个静态组件范数，而是随生成时间变化、具有来源、转换和汇合关系的动态路径对象。不过这仍是下一阶段待检验的研究方向，不是已证实理论。

### 十一、下一阶段与自动执行判断

当前 Phase363 阶段目标已经完成，预注册停止规则要求不运行 96 个确认提示，因此没有继续自动执行模型测试。下一阶段属于新的假设阶段，应命名为“动态轨迹对象预注册”，而不是 Phase363 的同阶段补丁。

下一阶段必须先完成：

1. 从已可回放的组件账本和注意力来源边中定义动态路径对象；
2. 在不读取行为标签和确认集的条件下冻结路径匹配规则；
3. 先验证路径对象是否跨提示、跨机制和跨模型复现；
4. 只有复现后才注册新的竞争预测和因果干预；
5. 不允许重新使用本轮七候选的最有利参数包装成新候选。

该阶段需要新的预注册对象和独立分母，不能在看到 Phase363 负结果后立即运行模型，否则会形成新的事后选择。

### 十二、工程验证

```text
Phase350-355 回归测试：22/22 通过；
Phase360-363 回归测试：11/11 通过；
Phase363 专项测试：4/4 通过，包含在上述 11 项中；
客户端与研究图谱 Phase363 状态一致；
客户端公开目录没有 .pt 张量；
96 个物理确认提示未读取、未运行。
```

### 十三、通俗总结

前一阶段找到了 7 个看起来能帮助预测“下一层会怎样”的信号。这一阶段问得更严格：它们能不能预测“模型生成下一个词时，新发生了什么”，以及正确词与错误词的竞争会怎样变化。

答案是否定的。虽然有少量局部改善，其中一条在三个模型里方向都为正，但改善幅度都小于题目和模板本身造成的自然波动。两个只在 DeepSeek7B 中较强的结果也没有跨模型复现。因此没有理由打开最后 96 道密封题，更没有理由做神经元干预。

这次失败很有价值：它关闭了“把下一层相关信号直接当成语言动态状态”的路线。下一步应该重新临摹动态路径，研究信息从哪里来、经过哪些组件、怎样汇合和转移，而不是继续给这 7 个静态数值增加补丁。

## Phase 364: 正确生成与无法闭合的矛盾审计及动态流束算法改进方案 [2026-07-11 09:42]

### 一、问题重述

本阶段分析如下矛盾：

```text
Qwen3、GLM4 和 DeepSeek7B 能在部分合格合同上生成正确内容；
当前项目却无法确认可跨模型复用的语言规则，也没有任何严格机制闭合。
```

直觉上，这说明研究算法可能存在根本错误或局限。这个方向高度值得重视，但“只有算法错误一种可能”仍然是过强判断。

严格逻辑只能推出：

$$
\text{正确行为存在}
\land
\text{当前观测和干预下机制不可识别}
$$

不能直接推出：

$$
\text{当前算法一定完全错误}
$$

也不能推出：

$$
\text{模型内部不存在规则}
$$

### 二、还存在的五种解释

#### 1. 观测对象丢失了真实状态

当前 Phase361-363（阶段361-363）的主要候选是：

```text
组件范数；
粗深度区间；
位置角色；
跨层均值；
少量生成时刻。
```

真实机制可能存在于：

```text
向量方向；
多个向量之间的相对关系；
注意力来源到接收者的有向写入；
MLP 门值与上投影值的乘积；
多个神经元的协同集合；
跨层和跨生成时间的条件分叉；
补偿路径之间的替代关系。
```

此时模型行为正确，而范数候选无法闭合，是完全一致的结果。

#### 2. 行为正确不等于使用统一抽象规则

小模型可能通过以下方式答对部分案例：

```text
模板记忆；
词元共现；
局部启发式；
多条不同路径的混合；
任务特异捷径；
偶然正确的候选竞争。
```

当前只有 4/18 机制通过盲发现准入，完整行为门更少。因此“模型会生成正确内容”尚不能提升为“九族语言规则已经稳定存在”。

#### 3. 真实机制可能是分布式和条件化的

同一功能可能由多个可替代子网络实现。删除其中一个组件后，其他路径补偿；把单个状态移植到另一案例时，又可能因上下文不匹配而失效。

#### 4. 跨模型闭合对象选错

不同模型不需要使用相同神经元、相同层号或相同线性方向。真正可复用的对象可能是：

```text
有类型的事件顺序；
来源和接收位置关系；
条件分叉结构；
守恒关系；
干预后的响应模式；
机制等价类。
```

要求固定坐标相同，可能把真实的结构同构错误判为失败。

#### 5. 闭合标准正确，但当前证据层级不足

严格闭合要求未来预测、功能特异、必要性、充分性、中介性、独立留出和跨模型复现。模型能答对只是行为层证据，不自动满足这些条件。

### 三、当前算法最可能的根本局限：非充分投影

设模型真实状态为：

$$
s_{k+1}=F(s_k,x_k)
$$

当前算法并不观察完整状态，而是观察压缩投影：

$$
o_k=P(s_k)
$$

例如：

$$
P(h)=\lVert h\rVert_2
$$

范数是多对一映射。对许多不同向量，可能有：

$$
P(h_1)=P(h_2)
$$

但下游权重对方向敏感，因此：

$$
Wh_1\neq Wh_2
$$

更一般地，如果存在：

$$
P(s_1)=P(s_2)
$$

同时：

$$
P(F(s_1,x))\neq P(F(s_2,x))
$$

或：

$$
U(F(s_1,x))\neq U(F(s_2,x))
$$

那么不存在只依赖当前压缩量的闭合公式：

$$
\nexists G
\quad
\text{使得}
\quad
P\circ F=G\circ P
$$

这不是回归模型容量不足，而是输入已经丢失决定未来所需的信息。增加线性项、窗口、阈值或更复杂拟合器都无法从不可逆压缩中恢复丢失状态。

Phase363（阶段363）的结果与这一诊断一致：七个组件范数候选有一定下一层连续性，但 20 个时间创新和词元竞争公式没有一个跨三模型越过自然波动地板。

### 四、哪些部分不是错误

当前项目并非整体推倒重来。以下工程基础是有效的：

```text
单样本和接口合同；
模型顺序执行和显存边界；
半精度原始加法顺序回放；
块级残差守恒；
全部注意力头和哈希平衡 MLP 分片格式；
注意力来源边精确分解；
生成时间轴；
完整词表 logits；
标签盲化发现原则；
发现集、确认集和因果集分离；
大量负结果和停止规则。
```

需要根本修改的是：

```text
从原始账本到候选对象的压缩方式；
把静态标量当成状态变量的假设；
以最高分组件替代动态路径的搜索单位；
以固定坐标相似替代跨模型结构等价；
在状态充分性未知时直接进入干预。
```

### 五、算法改进一：建立可逆的分层观测体系

不再从全迹直接跳到组件范数。建立逐级观测：

$$
P_0(s)
=
\text{组件范数骨架}
$$

$$
P_1(s)
=
\text{组件完整向量}
$$

$$
P_2(s)
=
\text{来源边和神经元写入账本}
$$

$$
P_3(s)
=
\text{跨层、跨位置、跨生成时间的动态流束}
$$

每次从高分辨率压缩到低分辨率以前，都必须进行“投影碰撞审计”。定义：

$$
\operatorname{Collision}(P)
=
\left\{
(i,j):
d(P(s_i),P(s_j))\leq\epsilon_m,
d(Y_i,Y_j)>\delta_m
\right\}
$$

其中，阈值必须来自同一测量的重复误差和自然条件波动，不能事后选择。

如果两个观测几乎相同但未来轨迹或输出竞争明显不同，该投影不能作为闭合状态。

### 六、算法改进二：从组件节点改为精确有类型写入边

注意力来源边继续采用已经通过回放的定义：

$$
e^A_{l,h,q\leftarrow s}
=
W^O_{l,h}
\left(
P_{l,h,q,s}V_{l,h,s}
\right)
$$

要求：

$$
\sum_s e^A_{l,h,q\leftarrow s}
=
\Delta A_{l,h,q}
$$

对于常见门控 MLP，可按实际架构记录每个通道的自然写入：

$$
g_{l,u,p}
=
\operatorname{act}
\left(
(W_g h)_{u}
\right)
\cdot
(W_u h)_{u}
$$

$$
e^M_{l,u,p}
=
W_d[:,u]g_{l,u,p}
$$

要求：

$$
\sum_u e^M_{l,u,p}
=
\Delta M_{l,p}
$$

最终逐块账本必须满足模型原始数据类型和加法顺序下的守恒：

$$
h_{l+1,p}
=
\operatorname{Add}_{dtype}
\left(
\operatorname{Add}_{dtype}
\left(
h_{l,p},
\sum_{h,s}e^A_{l,h,p\leftarrow s}
\right),
\sum_u e^M_{l,u,p}
\right)
$$

不同模型的门控结构必须分别适配，不能把一个模型的 MLP 分解公式直接复制到另外两个模型。

### 七、算法改进三：寻找动态流束，而不是假设唯一信息路径

残差叠加和特征复用意味着“某条语义只有一条唯一物理路径”通常不可识别。新算法的发现对象应是动态流束：

$$
\Pi
=
\left(
V,
E^A,
E^M,
E^{res},
E^{time},
E^{readout}
\right)
$$

其中保留：

```text
事件类型；
相对层深；
生成时间；
来源和接收位置角色；
完整写入向量或可恢复分片；
分叉与汇合结构；
竞争变化；
补偿支路。
```

盲发现不读取九族、机制、答案和历史候选，只枚举并复核重复的有类型局部流束。发现后才揭示功能标签。

公共架构主干必须单独登记。只有相对匹配控制出现稳定差分、且在独立组复现的流束，才能成为语言功能候选。

### 八、算法改进四：先检验状态充分性，再拟合规则

候选状态不再由“分数最高”决定，而由是否保留未来所需信息决定。

对候选状态：

$$
z_k=\Phi(\mathcal T_{\leq k})
$$

至少检验三件事：

#### 1. 未来轨迹预测

$$
\widehat{\Pi}_{k+1}
=
F_{\Pi}(z_k)
$$

#### 2. 竞争预测

$$
\widehat{C}_{k+1}
=
F_C(z_k)
$$

#### 3. 历史剩余信息

如果加入更早完整轨迹后，未来预测仍大幅改善，说明当前状态不是充分摘要。

预测目标首先是下一段真实流束和竞争状态，而不是同一个组件范数的持续性。

### 九、算法改进五：跨模型比较结构等价，而不是坐标相等

每个模型建立自己的完整物理图，再根据不含任务标签的结构签名寻找候选对应：

$$
I_m(\Pi)
=
\left[
\text{边类型顺序},
\text{位置角色变换},
\text{相对深度},
\text{分叉汇合},
\text{守恒比例},
\text{时间响应}
\right]
$$

跨模型闭合不要求：

$$
\Pi_Q=\Pi_G=\Pi_D
$$

而要求存在冻结映射，使结构转移一致：

$$
A_m
\left(
F_m(\Pi_m)
\right)
\approx
F_*
\left(
A_m(\Pi_m)
\right)
$$

该映射必须在发现集冻结，在独立留出和因果响应上确认，不能只依赖余弦相似度或可视化形状。

### 十、算法改进六：因果对象从单点升级为最小动态集合

重复流束通过物理留出后，才进入因果阶段。

必要性：

$$
Nec(\Pi)
=
M_{base}-M_{remove(\Pi)}
$$

充分性：

$$
Suf(\Pi)
=
M_{restore(\Pi)}-M_{corrupt}
$$

中介性：

$$
Med(a\rightarrow\Pi)
=
\frac{
M_{a\ removed,\Pi\ restored}-M_{a\ removed}
}{
M_{base}-M_{a\ removed}
}
$$

干预必须包含：

```text
错误时间；
错误位置；
错误来源；
同规模随机流束；
相同写入能量控制；
机制控制；
自然匹配状态；
补偿路径记录。
```

然后采用固定二分缩小策略寻找最小集合。只有最小集合最终大小为 1，才登记单神经元机制。

### 十一、闭合标准需要分层，不需要降低

建议把闭合拆成五层：

$$
G_{measure}
=
\text{完整账本可回放}
$$

$$
G_{predict}
=
\text{动态状态预测独立未来}
$$

$$
G_{function}
=
\text{流束与语言操作具有特异关系}
$$

$$
G_{causal}
=
Nec\land Suf\land Med
$$

$$
G_{transfer}
=
\text{跨提示、跨机制与跨模型结构复现}
$$

严格机制闭合仍要求：

$$
G_{closure}
=
G_{measure}
\land
G_{predict}
\land
G_{function}
\land
G_{causal}
\land
G_{transfer}
$$

这不是降低标准，而是让失败能够定位到具体层级。

### 十二、分阶段修改方案

#### Phase364-A（阶段364-A）：离线投影充分性与碰撞审计

不运行模型，复用当前 9 个全来源边锚点、全向量锚点和 288 个生成时间骨架。

任务：

```text
实现 P0 到 P3 的统一对象接口；
证明每级数据能否从上一级恢复；
测量组件范数投影的碰撞；
区分测量误差、自然条件波动和真实未来分叉；
冻结动态流束数据格式和存储预算。
```

当前 9 个锚点只能验证工程可行性，不能得出语言机制结论。

#### Phase365（阶段365）：三模型小规模仪器验证

只有 Phase364-A 通过后才运行。依次执行 Qwen3、GLM4 和 DeepSeek7B，不能同时装载。

固定工程分母建议为：

```text
4 个已准入机制；
每机制每模型 2 个独立组；
每组 4 个条件；
3 个模型；
总计 96 个案例。
```

这 96 例只验证：

```text
注意力来源边守恒；
MLP 神经元写入守恒；
生成时间对齐；
哈希分片均衡；
动态流束可离线重建。
```

不得据此总结语言机制。

#### Phase366（阶段366）：大样本盲流束发现

仪器通过后，使用：

```text
发现：4 机制 × 6 组 × 4 条件 × 3 模型 = 288 案例；
密封留出：4 机制 × 2 组 × 4 条件 × 3 模型 = 96 案例。
```

采用全骨架、平衡神经元分片和固定全量锚点三种分辨率，不进行结果驱动的前 K 名筛选。

#### Phase367（阶段367）：动态状态充分性和结构等价验证

只允许使用 Phase366 冻结的流束，检验：

```text
未来流束预测；
竞争变化预测；
历史剩余信息；
跨模型结构映射；
独立留出复现。
```

#### Phase368（阶段368）：最小动态集合因果验证

只有通过 Phase367 的流束才进入 CUDA 干预。依次运行三个模型，先必要性，再充分性，再中介性，最后固定二分缩小到神经元或最小分布式集合。

### 十三、停止规则

```text
如果 P0 失败而 P1/P2 成功，确认旧压缩算法是主要瓶颈；
如果 P2 仍无法回放，先修复挂钩和组件分解，不进入发现；
如果 P2 可回放但动态流束不能复现，只登记模型自然轨迹，不命名语言规则；
如果流束只预测公共主干，不预测竞争和行为，登记架构脉络；
如果只在单模型成立，登记模型特异结构，不进入统一理论；
如果跨模型预测通过但因果失败，登记预测状态，不登记机制；
任何阶段失败都不得打开下一阶段密封数据寻找补丁。
```

### 十四、对当前判断的最终回答

“模型能正确生成，但研究无法闭合”确实强烈提示当前算法存在根本局限，尤其是候选构造前的不可逆压缩和错误搜索单位。不过它不是唯一可能性；行为样本可能依赖捷径，机制可能分布式、条件化或只在模型内部以不同坐标实现。

当前最应优先验证的不是某个更复杂公式，而是：

$$
\boxed{
当前观测投影是否保留了决定未来和行为所需的信息
}
$$

如果这个门不通过，任何规则拟合和因果补丁都没有闭合基础。

### 十五、理论边界与本阶段执行情况

本阶段没有修改智能理论名称，没有新增语言机制结论，也没有运行模型测试。客观进度分母保持 Phase363（阶段363）结果：

```text
盲发现准入：4/18；
严格时间和竞争公式：0/20；
物理留出机制：0/18；
因果封存机制：0/18；
严格机制闭合：0/72。
```

下一步与当前问题属于同一算法校准阶段，应自动进入 Phase364-A（阶段364-A）的离线投影充分性和碰撞审计；在该审计完成前，不应运行新的三模型数据。

### 十六、通俗总结

模型能够答对，说明机器内部确实发生了足以产生答案的计算。但我们当前把这段计算压缩成了几个“这一层有多强”的数字。两个方向完全不同、作用也不同的向量，可以拥有相同长度；多个神经元如何配合、信息从哪里搬到哪里，也会在求平均后消失。

因此，现在最可能的问题不是公式太简单，而是给公式看的东西已经不够。改进办法是先保存能够加回原模型的来源边和神经元写入，再寻找跨层、跨位置和跨生成时间重复出现的动态流束。只有流束能预测未来、通过留出，并在干预时产生预期变化，才能继续缩小到真正有效的神经元。

## Phase 364-A: 七候选投影退化与高分辨率仪器资格离线审计 [2026-07-11 09:48]

### 一、执行范围

根据 Phase364（阶段364）的判断，继续自动完成离线投影充分性资格审计。本阶段：

```text
没有加载或运行任何模型；
没有使用 CUDA；
没有读取 96 个物理确认提示；
复用 288 个发现案例的组件范数账本；
复用 9 个全来源边锚点和 520 个层文件；
复用 Phase363 的 20 个严格公式结果。
```

### 二、七候选投影秩审计

对每个模型和生成时刻构造：

$$
X_{m,t}
\in
\mathbb R^{96\times 7}
$$

七列严格使用 Phase362 冻结候选，没有重新选择特征。结果为：

```text
Qwen3：t0 秩 6，t1 秩 7，t2 秩 7；
GLM4：t0 秩 6，t1 秩 7，t2 秩 7；
DeepSeek7B：t0 秩 6，t1 秩 7，t2 秩 7。
```

三个模型在 t0 都出现相同的全案例精确重复列：

```text
middle / attention_output / answer_start
等于
middle / attention_output / current_generation
```

每个模型均为：

```text
96/96 案例完全相等；
相等比例 1.0；
七维输入的实际秩降为 6。
```

原因是第一个生成时刻中，答案起点和当前生成位置发生位置别名。该结果证明七候选在关键起始时刻不是七个独立状态量。

这只能证明投影退化，不能证明所有可能的非线性映射都必然失败。

### 三、P0 到 P3 的现实可用性

#### P0：组件范数骨架

```text
覆盖：288 个发现案例、三个生成时刻；
优点：分母大、格式统一；
缺点：范数和深度均值不可逆；
结论：只能作为粗骨架，不能假定为充分状态。
```

#### P1：组件完整向量

```text
覆盖：9 个锚点；
三模型隐藏维分别为 2560、4096、3584；
层输入、归一化状态、注意力输出、MLP 输出和层输出均保存；
结论：格式可回放，但分母只够工程验证。
```

#### P2：有类型来源和通道写入

```text
三模型均可离线回放注意力来源边：3/3；
三模型均可离线回放 MLP 哈希分片写入：3/3；
逐神经元 MLP 写入自包含：0/3。
```

现有锚点保存了 MLP 分片激活和通道编号，但没有保存下投影权重列或冻结权重引用，因此模型卸载后不能只依赖结果文件恢复每个神经元的残差写入。

#### P3：动态流束

```text
三模型固定格式：0/3；
跨层连接：未实现；
跨生成时间连接：未实现；
分叉、汇合和补偿支路：未实现。
```

### 四、与 Phase363 负结果的关系

现有证据链为：

$$
\text{P0 结构上多对一}
\land
\text{t0 存在精确列退化}
\land
\text{严格公式通过为 }0/20
$$

因此可以得出：

$$
\boxed{
P0\ \text{不能继续被默认视为语言动态的充分状态}
}
$$

但不能得出：

$$
\boxed{
P0\ \text{对所有可能非线性映射均已被数学否定}
}
$$

后一个结论需要重复噪声、明确碰撞阈值和更广的映射类别。本阶段保持该边界。

### 五、冻结的下一仪器合同

已冻结 Phase365 的工程分母：

```text
模型：3；
已准入机制：4；
每模型每机制独立组：2；
每组条件：4；
总案例：96；
生成时刻：3；
执行顺序：Qwen3 → GLM4 → DeepSeek7B。
```

但当前 `new_model_execution_authorized` 为假。运行前必须完成：

1. 三模型各自的 MLP 门值、上投影值、乘积和下投影写入适配器；
2. 能离线恢复单神经元写入的权重引用合同；
3. 跨层、跨位置、跨生成时间的动态流束数据格式；
4. 重复运行噪声分母，用于冻结碰撞阈值；
5. 组件、来源、分片和原始数据类型加法顺序的守恒单元测试。

### 六、决策

本阶段决策为：

```text
永久把 P0 降级为有损粗骨架；
不再使用七候选增加时间公式补丁；
保留 P1/P2 锚点作为新仪器开发基准；
在 P2/P3 仪器完成前禁止新模型执行；
下一任务是实现并单元测试模型特异 MLP 写入适配器和动态流束格式。
```

### 七、图谱同步与验证

Phase364-A 摘要和 Phase365 仪器合同已经同步到研究图谱与可视化客户端。前端没有导出原始张量。

```text
Phase364-A 专项测试：4/4 通过；
Phase360-364 回归测试：15/15 通过；
新模型运行：0；
物理确认读取：0；
严格机制闭合：0/72。
```

### 八、通俗总结

这次离线检查发现，原来的七个数字在模型刚开始生成答案时其实只有六份独立信息，因为“答案起点”和“当前生成位置”指向同一位置。更重要的是，这些数字只保存强度，没有保存方向和来源。

现有九个高分辨率锚点已经足够证明注意力来源边和 MLP 分片能够记账，但还不能在模型卸载后恢复每个神经元到底写入了什么，也没有把这些写入连成跨层、跨时间的动态流束。因此下一步不是增加样本，而是先把测量仪器补完整；否则样本越多，只会积累更多有损摘要。

## Phase 365: 三模型动态流束仪器、逐神经元写入恢复与自由生成全量采集 [2026-07-11 18:03]

### 一、对 Phase363 审计判断的校准

Phase363（阶段363）的核心判断正确：`0/20` 关闭的是“七个静态范数汇总量作为生成时间状态变量”的具体路线，不是否定网络中存在动态状态，也不是否定动态路径。

但“模型能够正确生成，而规则和闭合没有确认，因此只有算法错误一种可能”需要收紧。正确生成至少说明模型内部存在足以产生该次输出的计算；仍无法闭合还可能来自：

1. 观测投影丢失来源、方向、顺序和汇合关系；
2. 语言行为由分布式、条件化、可补偿的路径实现，不存在单一静态规则位置；
3. 当前四个准入机制和小模型不是完整语言分母；
4. 三个模型可能实现同一功能，但内部坐标、深度节奏和分工不同；
5. 行为正确可能来自局部捷径，不能自动推出统一规则已被实现；
6. 测量、候选发现、跨模型对齐或因果操作仍可能不充分。

因此，本阶段把“算法存在根本局限”作为强候选解释，而不是唯一逻辑可能性。

### 二、逐神经元 MLP 写入合同

三种真实 MLP（多层感知机）布局均实现适配：

$$
g=\phi(W_g h),\qquad u=W_u h,\qquad p=g\odot u
$$

$$
\Delta h_{\mathrm{MLP}}=W_d p=\sum_i p_i W_{d,:,i}
$$

其中 Qwen3（通义千问3）和 DeepSeek7B（深度求索7B）使用分离门投影与上投影，GLM4（智谱语言模型4）使用融合门—上投影。结果文件保存实际前向中的门值、上投影值、乘积和冻结权重引用，因此模型卸载后可以分块恢复任意神经元写入：

$$
w_i=p_i W_{d,:,i}
$$

没有保存所有神经元的完整输出张量；这避免了约 `3.871 TB` 的显式写入存储。角色聚焦原始账本预算约 `10.389 GB`，实际全量采集约 `18.85 GB`，差异来自扩展到完整 288 案例及文件开销。

### 三、注意力来源写入合同

对接收位置 $r$、来源位置 $s$、层 $l$、生成时刻 $t$，来源写入定义为：

$$
e_{t,l,r\leftarrow s}
=
W_O\left[\alpha_{t,l,h,r,s}v_{t,l,h,s}\right]_h
$$

角色来源集合之外的全部来源不被删除，而是保留守恒桶：

$$
e_{t,l,r\leftarrow \mathrm{other}}
=
\Delta h^{\mathrm{attn}}_{t,l,r}
-
\sum_{s\in R}e_{t,l,r\leftarrow s}
$$

因此角色来源边与其他来源桶之和可以恢复完整注意力写入。答案起点与当前生成位置发生位置重合时，显式保存别名组，不重复伪造两个物理位置。

### 四、六次重复与格式门

按 Qwen3（通义千问3）→ GLM4（智谱语言模型4）→ DeepSeek7B（深度求索7B）顺序运行，每次只加载一个模型：

```text
总前向：6；
每模型：同一无标签提示重复 2 次；
重复完全一致：3/3；
MLP 写入回放通过：3/3；
最大直接相对误差：0.0006166791；
最大逐神经元回放相对误差：0.002198834；
固定执行重复误差：0。
```

重复为 0 只说明当前固定 CUDA 执行路径可精确复现，不等于自然模板噪声为 0。

### 五、288 案例全量自由生成采集

冻结分母：

```text
模型：3；
独立组：72；
匿名条件：每组 4；
案例：288，每模型 96；
自由生成时刻：3；
层文件：29,952；
时间文件：864；
原始账本：18,848,575,296 字节。
```

每个时刻使用模型自然首选词元继续生成，不使用教师强制目标词元。采集器不读取语言族、机制、正确答案或目标竞争标签。

最终三模型均通过守恒门，最大误差为：

```text
注意力来源回放：0.004213266；
MLP 直接回放：0.006121746；
MLP 逐神经元回放：0.005713898；
MLP 乘积捕获：0.001408686；
整块残差：0；
注意力概率：0.003462791。
```

DeepSeek7B（深度求索7B）初次检查的乘积误差为 `0.0163587`，超过冻结门 `0.01`。没有降低门槛；改为直接挂钩真实完整序列的门投影和上投影输出后，只重跑 DeepSeek7B（深度求索7B），乘积误差降为 0。该失败来自局部重算与实际整序列矩阵乘法数值顺序不等价。

### 六、工程故障与修复

全量动态束提取在 Qwen3（通义千问3）和 GLM4（智谱语言模型4）完成后，DeepSeek7B（深度求索7B）进程以状态 `139` 退出。原因不是模型证据失败，而是提取器缺少单模型断点续跑，并遗留一个零字节派生文件。

修复内容：

1. 增加模型级、设备级和断点续跑参数；
2. 续跑资格从“文件数量存在”收紧为“数量完整且非零”；
3. 单独补齐 DeepSeek7B（深度求索7B）缺失的 58 个案例；
4. 重建受损案例；
5. 对全部派生引用执行可读性和 SHA-256 哈希审计。

最终：

```text
派生来源边文件：29,952；
可读且哈希一致：29,952；
缺失、零字节、不可读或哈希不一致：0。
```

### 七、本阶段结论

Phase365（阶段365）完成了动态路径研究的测量仪器和全量自由生成原始账本。它证明来源边和逐神经元 MLP 写入可以按守恒公式记录与恢复，但没有发现语言路径，没有运行因果干预，也没有打开物理确认集。

## Phase 366: 288 例有类型动态束、盲分组、方向描述量与阈值保管 [2026-07-11 18:03]

### 一、动态图对象

每个案例建立有类型动态图：

$$
G_c=(V_c,E_c)
$$

节点包含来源写入、注意力汇合、残差状态、MLP 写入、层输出、词表状态；边包含路由、写入、汇合、残差连续、生成时间和词表转移。

结果：

```text
动态束：288/288 有效；
有类型事件：1,069,152；
有类型边：1,068,864；
角色来源派生文件：29,952；
派生数据：3,616,404,480 字节左右；
目标竞争字段：0；
直接图相减：0；
语言路径候选：0。
```

这些动态束覆盖来源、查询、答案起点和当前生成位置四类角色别名，不是所有词元位置的完整图。

### 二、盲发现与盲校准冻结

72 个独立组严格冻结为：

```text
盲发现：48 组、192 案例；
盲校准：24 组、96 案例；
每模型：64 个发现案例、32 个校准案例；
每组：4 个匿名条件槽；
与密封物理确认 96 案例交集：0。
```

四条件保持独立，不在发现前取平均。类型事件必须先对齐，未匹配事件必须保留，禁止把两张图直接相减。

### 三、方向敏感描述量

对每条“来源写入→注意力汇合→残差汇合与 MLP 写入→层输出”局部路径，保留全部来源路由并计算 10 个基础描述量，包括：

$$
\frac{\lVert e_{r\leftarrow s}\rVert}{\lVert \Delta h_{\mathrm{attn}}\rVert},
\qquad
\cos(e_{r\leftarrow s},\Delta h_{\mathrm{attn}}),
\qquad
\frac{e_{r\leftarrow s}^{\mathsf T}\Delta h_{\mathrm{attn}}}
{\lVert\Delta h_{\mathrm{attn}}\rVert^2}
$$

以及来源写入与层输出变化的对齐、注意力与 MLP 写入平衡、输入与输出对齐等。

```text
方向路径描述：519,168；
每条特征：10；
Qwen3：179,712；
GLM4：199,680；
DeepSeek7B：139,776；
前 K 名筛选：未使用；
原始向量：仍由哈希引用保留。
```

描述量不是原始向量的替代物，也不能直接命名为语言路径。

### 四、独立阈值保管

阈值保管器只使用发现组中的同操作跨词汇模板对。A/C 与 B/D 分开配对，不把操作条件差异计入噪声；校准组完全未参与阈值形成。

冻结地板：

$$
\tau_{m,t,b,s,r,f}
=
\max\left(
e_{\mathrm{repeat}},
4e_{\mathrm{reconstruct}},
Q_{0.75}\left(\left|d_x-d_y\right|_{\mathrm{same\ operation}}\right)
\right)
$$

客观分母：

```text
同操作案例对：96；
配对路由：173,056；
特征差值：1,730,560；
冻结地板行：4,680；
校准描述行未使用：173,056。
```

`4 × 最大重构误差` 和上四分位模板差值都是保守工程规则，不是已经证明的自然噪声定律，更不是语言数学定律。

## Phase 367: 标签盲化多尺度动态脉络发现与候选冻结 [2026-07-11 18:03]

### 一、路径离散化

每个模型按相对深度选取 10 个固定锚点。对每个生成时刻、来源角色、接收角色和特征，计算相邻锚点变化：

$$
q_i=
\begin{cases}
+1,&d_{a_{i+1}}-d_{a_i}>\tau_i\\
-1,&d_{a_{i+1}}-d_{a_i}< -\tau_i\\
0,&\text{其他}
\end{cases}
$$

枚举长度为 `2、3、4、6、8` 的连续路径。全零路径只作为公共基线，不作为动态候选。没有进行结果驱动的前 K 名筛选。

### 二、冻结预测门

独立组是分析单位。每模型发现组为 16 个，候选必须满足：

```text
独立组支持：至少 8；
下一变化组等权准确率：至少 0.75；
超过最强基线：至少 0.10。
```

基线包括：

1. 模型—时间—相对位置—角色的公共转移；
2. 当前状态持续；
3. 生成时间循环打乱，角色不对应时使用深度顺序反转；
4. 同模型、同规模、不同独立组的确定性随机案例。

这些门是预注册工程门，不是显著性结论。

### 三、发现结果

```text
枚举窗口：2,196,480；
全零窗口：365,950；
非零脉络形状：236,442；
达到独立组支持门：30,880；
同时越过未来预测和四基线门：49；
Qwen3：3；
GLM4：4；
DeepSeek7B：42；
三模型完全同构签名：0。
```

49 条只是在发现集冻结的候选形状，未读取语言族、机制、条件含义、正确答案或目标词元，不能解释为语言规则。

## Phase 368: 冻结动态脉络独立校准与跨模型失败审计 [2026-07-11 18:03]

### 一、独立校准

49 条候选的路径形状、阈值、下一状态预测和全部门槛在校准前冻结。24 个校准独立组只运行一次，不允许重拟合。

结果：

```text
冻结候选：49；
校准通过：4；
Qwen3：2；
GLM4：0；
DeepSeek7B：2；
三模型同构校准签名：0；
语义标签读取：0；
物理确认读取：0；
CUDA 因果干预：0。
```

四条局部校准形状均出现在第三个自由生成时刻，涉及来源到当前生成位置的注意力对齐、其他来源桶到当前生成位置的输出变化对齐，以及当前生成位置自路由强度变化。它们只是模型特异的预测形状；由于三模型同构为 0，禁止进行语言族标签解释。

### 二、对“算法存在根本错误或局限”的回答

旧算法的根本局限已得到直接证据支持：

$$
P_0(H)=
\left[
\operatorname{mean}_{l\in B}\lVert c_{t,l,r}\rVert
\right]
$$

会不可逆地删除方向、来源、层内顺序、分叉、汇合和神经元协作，而且首生成时刻七列实际秩为 6。因此，旧投影不能继续被默认视为充分状态。

但新算法尚未解决充分性：

1. 原始流束又被压成 10 个标量描述量；
2. 连续层轨迹被压成 10 个深度锚点和三值符号；
3. 注意力头在描述阶段经过输出投影后合并；
4. MLP 神经元写入虽可恢复，但没有进入脉络枚举；
5. 跨模型要求相同角色、特征、锚点和离散序列，可能过于刚性；
6. 顺序打乱控制在角色不对应时采用深度反转替代，不是完美生成时间置换；
7. 49 个候选经过同一组工程门，仍存在多候选筛选风险；
8. 当前只覆盖 4/18 个准入机制和四类位置角色；
9. 三个小模型的内部机制可能比大模型粗糙，结果不能直接外推。

因此当前证据支持：

$$
\boxed{
旧静态算法存在根本信息损失，
新动态算法完成了测量升级但仍未证明充分
}
$$

不能支持：

$$
\boxed{
网络不存在可复用语言规则
}
$$

也不能支持：

$$
\boxed{
四条局部校准形状就是语言编码机制
}
$$

### 三、当前物理图谱的客观完成项

```text
三模型动态采集：288/288；
动态束格式：288/288；
派生来源边完整性：29,952/29,952；
方向描述账本：519,168 条；
盲发现与校准：48/24 独立组；
局部校准形状：4；
跨模型语言路径：0；
物理确认机制：0/18；
因果密封机制：0/18；
严格机制闭合：0/72。
```

工程账本在当前四角色、四机制范围内完成；九族语言模式物理图谱没有完成。单一总体进度百分比仍不成立，因为工程覆盖、语言族覆盖、跨模型预测和因果闭合分母不同。

### 四、下一阶段 Phase369（阶段369）

下一阶段不能直接复用 Phase368（阶段368）的校准组调参，否则校准组会变成发现组。也不能打开 96 个物理确认案例修补跨模型对齐。

应分四个大任务执行：

1. **原始向量与拓扑保持的跨模型等价合同**：允许相对深度轻微偏移和模型内部坐标变换，但必须保持来源—接收角色、分叉—汇合顺序、未来转移和守恒关系；不得只匹配相同离散符号串。
2. **头与神经元写入进入路径**：在所有角色来源边上按守恒分块展开注意力头和 MLP 神经元写入，不以激活前 K 名代替分母；先做小分母回放和存储门，再扩展。
3. **全新独立发现—校准数据**：重新冻结新的发现组和校准组。Phase368 校准结果只作为算法失败审计，不参与新门槛选择。
4. **因果顺序**：只有新数据上出现三模型同构、未来可预测路径，才允许打开物理确认；确认通过后再依次对 Qwen3、GLM4、DeepSeek7B 运行必要性、充分性和中介性 CUDA 干预。

当前不自动运行 Phase369（阶段369）模型测试，因为它已经进入新的数据周期；在跨模型等价合同和新分母冻结前继续使用旧校准数据，会破坏本轮独立性。

### 五、图谱和客户端同步

研究图谱与可视化客户端均更新到 Phase368（阶段368）：

```text
允许展示：288 个动态测量束、事件/边分母、完整性、发现/校准计数；
禁止展示：把四条局部形状标为语言模式族路径；
原始张量导出前端：否；
语义标签导出前端：否；
客户端最新阶段：Phase368。
```

### 六、通俗总结

以前的算法把模型内部运行压成几个“强不强”的数字，像只记录每条道路的总车流，却不记录车从哪里来、经过哪里、在哪里汇合。现在已经能记录来源边、MLP 神经元写入、跨层和跨生成时刻关系，并把三模型 288 次自由生成临摹成可回放动态图。

新算法确实比旧算法更接近真实运行，但仍把复杂轨迹压成十个特征和十个深度点。独立校准后只剩四条单模型局部形状，没有任何三模型共同路径。因此本轮最重要的成果不是“找到四条语言规则”，而是客观定位了下一瓶颈：必须解决原始向量级、拓扑保持的跨模型功能等价，同时把头和神经元写入真正接入路径搜索。

## Phase 369: 全新分母上的原始向量关系、未来预测与跨模型拓扑严格检验 [2026-07-11 19:33]

### 一、对 Phase364-368 统一审计文本的判断

输入审计的主判断基本正确：旧 P0 范数骨架是多对一有损投影，Phase365-368 虽已建立高分辨率物理账本，但实际发现对象又被压缩成十个描述量、十个深度锚点和三值序列，不能据此宣布语言路径。

本阶段在执行前收紧了六处口径：

1. `18%-22%` 不是科学天然进度，继续保持 `single_global_progress_percentage_valid=false`。
2. `0/72` 只能称为七十二个注册闭合单元中的零个闭合，不能称为已经存在七十二个真实神经元机制。
3. 盲事件中的 `rank` 不能包含目标词元排名或边际；只允许标签无关的完整词表状态引用。
4. Qwen3 与 DS7B 共享 Qwen 架构家族，二者共同结果不能单独构成异构模型复现；Level 2 必须包含 GLM4。
5. 不允许拟合任意跨模型坐标旋转，因为它容易把模型差异过拟合成对齐。
6. 不再使用可互相抵消的加权拓扑总分；每个预测、角色、顺序、控制和跨模型分量必须独立过门。

### 二、全新数据合同与行为资格

首先冻结 576 例全新并行合同：

```text
模型：3；
准入机制：4；
每模型每机制独立组：12；
每组条件：4；
发现案例：288；
校准案例：144；
物理留出案例：144；
与 Phase353/354/361/362 提示重合：0；
盲注册表语义字段：0。
```

首次行为测试暴露出测量伪影：变长样本在同一批次左填充时，GLM4 大量输出 token 0 对应的 `!`；8 token 预算又截断了 Qwen3 与 DS7B 的完整答案。该结果不能解释为模型能力失败。

资格脚本随后冻结为：

```text
相同 token 长度分桶；
批内不再混合不同长度；
最大生成 24 token；
目标命中且干扰项不命中；
四条件和三模型全部自然正确才保留并行组。
```

修复后初始案例正确数：

```text
Qwen3：126/144；
GLM4：123/144；
DS7B：125/144。
```

初始数一致案例把复数写成 `casket-p369-...-xs` 等非自然形式，三模型共同完整组为零。该合同被登记为无效，不用内部轨迹修补。第一版自然词形替换仍只有 1 个发现组、0 个校准组，因此再冻结一个常用名词大分母扩展：

```text
扩展并行组：30；
发现：18；
校准：9；
物理留出：3；
扩展行为案例：360；
旧提示重合：0。
```

扩展非物理案例结果：

```text
Qwen3：108/108；
GLM4：101/108；
DS7B：102/108。
```

最终行为资格分母：

```text
共同合格并行组：43；
合格案例：516；
发现并行组：28，案例 336；
校准并行组：15，案例 180；
关系绑定发现/校准：5/3 组；
目标竞争发现/校准：6/2 组；
实体新近性发现/校准：6/3 组；
数一致发现/校准：11/7 组；
物理留出读取：0。
```

### 三、三模型 CUDA 原始物理账本

只对 336 个合格发现案例按 `qwen3 -> GLM4 -> DS7B` 串行采集。校准和物理留出保持封存。

```text
Qwen3：112 案例，36 层，12,432 文件，6,900,480,096 字节；
GLM4：112 案例，40 层，13,776 文件，8,438,442,720 字节；
DS7B：112 案例，28 层，9,744 文件，7,235,792,480 字节；
合计：336 案例，35,952 文件，22,574,715,296 字节。
```

每个案例记录三个自然生成时刻，并保留：

```text
四角色原始组件向量；
全部来源位置值向量；
全部注意力头概率；
MLP gate/up/product；
完整词表 logits；
模型权重引用；
文件 SHA-256。
```

六类守恒门全部通过。三模型最大相对误差：

```text
注意力来源重构：0.00392816；
MLP 直接重构：0.00733444；
MLP 逐神经元重构：0.00619484；
MLP 乘积：0；
块残差：0；
概率归一：0.00357008。
```

动态束初次验证统一缺少根字段 `anonymous_group_id`。审计确认每束只有这一项错误，事件、边和向量引用未损坏。用原始采集清单补齐匿名组 ID，未重算向量、未修改事件和边，重新验证结果：

```text
有效动态束：336/336；
有类型事件：1,247,344；
有类型边：1,247,008；
角色来源派生文件：34,944；
派生字节：4,219,884,032。
```

### 四、原始向量关系对象

为避免任意跨模型旋转，六个同路径原始向量使用坐标不变关系：

$$
K_{ij}
=
\frac{v_i^{\mathsf T}v_j}
{\lVert v_i\rVert\lVert v_j\rVert}
$$

$$
r_i
=
\frac{\lVert v_i\rVert}
{\sum_j\lVert v_j\rVert}
$$

其中：

```text
v = [来源路由、层输入、注意力写入、注意力后状态、MLP 写入、层输出变化]。
```

每条路径形成 15 个有符号 Gram 关系与 6 个范数份额，共 21 维；同一路径同时计算旧 10 描述量，保证比较分母完全相同。

完整词表只转换为：

```text
top1 概率；
top2 概率；
top1-top2 概率差；
top5 质量；
归一化熵。
```

没有读取正确答案 token 的排名或边际。

结果：

```text
案例：336；
对齐来源路径：605,696；
旧描述量：10；
原始关系：21；
词表状态：5；
语义标签：0；
校准或物理案例：0。
```

### 五、盲未来预测

以独立匿名组为重复单位，禁止同组四条件互相充当邻居。第一个生成时刻的全层路径作为状态，第二个生成时刻的原始关系流和标签无关词表状态作为未来。跨模型比较只使用 32 点连续线性重采样视图；原始逐层数据仍完整保留。

发现前冻结的控制包括：

```text
旧 10 描述量；
确定性随机流；
案例特异时间打乱；
案例特异角色置换；
等能量错误流；
公共主干未来。
```

未来流平均误差：

| 模型 | 原始关系 | 旧 10 描述量 | 随机流 | 等能量错误流 | 公共主干 |
|---|---:|---:|---:|---:|---:|
| Qwen3 | 0.02360991 | 0.02377236 | 0.05694381 | 0.02353663 | 0.04036510 |
| GLM4 | 0.02972003 | 0.03074608 | 0.06427462 | 0.03120360 | 0.04788062 |
| DS7B | 0.03413307 | 0.03405182 | 0.08831753 | 0.03602289 | 0.05869458 |

原始关系逐案胜过旧描述量的比例：

```text
Qwen3：0.17857143；
GLM4：0.09821428；
DS7B：0.09821428。
```

因此 Qwen3 与 GLM4 虽有轻微平均改善，但改善集中在少数案例，逐案多数未改善；DS7B 的流和词表未来均略差。三模型完整未来门全部失败，Level 1 模型数为 0。

### 六、跨模型结构检索

跨模型仅减去各模型标签盲公共主干，不拟合坐标旋转。以同一匿名并行提示作为正确检索对象。

| 模型对 | 原始关系匹配/错误距离比 | 原始 top5 | 旧描述 top5 | 随机 top5 | 四分量全过 |
|---|---:|---:|---:|---:|---:|
| Qwen3 -> GLM4 | 0.95709914 | 0.11607143 | 0.16964285 | 0.04464286 | 否 |
| GLM4 -> DS7B | 0.96193761 | 0.15178572 | 0.07142857 | 0.04464286 | 是 |
| Qwen3 -> DS7B | 0.97466880 | 0.08035714 | 0.11607143 | 0.04464286 | 否 |

GLM4 -> DS7B 的原始关系跨模型四分量局部通过，但这两个模型的未来充分性门都未通过，所以不能形成 Level 2 联合证据。

最终：

```text
Level 1 模型：0/3；
Level 2 异构模型对：0/2；
Level 3 三模型：否；
语义标签揭示：0；
校准内部轨迹：0/180；
物理留出：0；
语言路径：0。
```

### 七、Phase369 严格结论

Phase369 是强负结果，不是物理账本失败：

$$
\boxed{
\text{精确物理账本成立}
\quad\land\quad
\text{Gram/范数份额状态不充分}
}
$$

它不能支持“模型没有语言规则”。它只否定了以下具体假设：

$$
\boxed{
\text{六类原始向量的二阶关系}
\text{足以形成稳定跨模型未来状态}
}
$$

失败可能来自两类原因：

1. 路径对象仍丢失所有 token 位置、Q/K 关系、头与神经元的精确向量协同和分片内交叉项。
2. 案例最近邻不是模型真实状态转移律，即使特征含有信息，也可能无法用这种读出方式恢复未来。

因此按预注册停止规则，不采集校准内部轨迹，不打开物理留出，不运行 CUDA 因果干预。

## Phase 370: 固定哈希头与神经元能量拓扑探索性诊断 [2026-07-11 19:33]

### 一、诊断资格与边界

Phase369 未授权校准，但失败直接指向头和神经元信息缺失。因此只在同一发现账本上进行探索性诊断，用于判断是否值得开启新的独立数据周期；该结果不能反向挽救 Phase369。

冻结合同：

```text
头/神经元分片：8、32、128；
固定哈希种子：17、29、43；
任务分数 Top-K：禁止；
多个种子和分辨率：敏感性检查，不是独立重复；
组合规则：最坏分量候选排名最小，再比较排名和；
加权距离：未使用。
```

注意力头贡献代理为：

$$
a_h
=
p_{h,r,s}
\lVert v_{h,s}\rVert
\lVert W^O_h\rVert_F
$$

MLP 单神经元自然写入幅度是精确量：

$$
m_j
=
|p_j|
\lVert W^D_{:,j}\rVert
$$

这些量按固定哈希聚合并排序。它们保留单位贡献分布，但不保留分片内向量方向和神经元间交叉项，所以只能称为能量拓扑诊断。

### 二、客观分母

```text
案例：336；
头路由记录：605,696；
MLP 角色记录：128,128；
哈希分辨率：3；
种子：3；
Top-K：0；
语义标签：0；
单神经元因果确认：0。
```

### 三、未来诊断结果

组合状态逐案胜过 Phase369 原始关系的比例：

| 模型 | 8 分片 | 32 分片 | 128 分片 |
|---|---:|---:|---:|
| Qwen3 | 0.08928572 | 0.11607143 | 0.08928572 |
| GLM4 | 0.07142857 | 0.10714286 | 0.07142857 |
| DS7B | 0.13392857 | 0.15178572 | 0.11607143 |

所有模型、所有分辨率都未超过 0.5。Qwen3 的词表误差在三种分辨率均变差；GLM4 词表误差略好但逐案门失败；DS7B 的流和词表均变差。

跨模型 top5 也没有稳定提升：

```text
Qwen3 -> GLM4：原始 0.1161，组合最好 0.1250，但最坏分量距离比失败；
GLM4 -> DS7B：原始 0.1518，组合只有 0.0536-0.0625；
Qwen3 -> DS7B：原始 0.0804，组合 0.0357-0.0804。
```

最终：

```text
未来全分量通过：0；
异构跨模型全分量通过：0；
可开启新独立哈希拓扑周期：否；
Phase369 校准：否；
物理留出：否；
单神经元机制：0。
```

### 四、硬伤与路线关闭

固定哈希避免了任务监督 Top-K，但它仍然把真正的向量协同压成能量份额：

$$
\boxed{
\sum_j \lVert w_j\rVert
\neq
\left\lVert\sum_j w_j\right\rVert
}
$$

尤其当神经元互相抵消、补偿或形成方向协同时，排序后的分片能量无法恢复：

```text
方向；
相位或符号协同；
分片内交叉项；
分叉与补偿身份；
真实下游响应。
```

因此关闭“继续扩大固定哈希能量分片”路线。不能因为 32 分片某个平均量略好就打开校准。

## Phase 371: 精确向量协同与守恒树路径对象协议冻结 [2026-07-11 19:33]

### 一、第一性原理修正

确定性 Transformer 的完整输入序列、权重和精确机器状态必然决定下一步输出。当前没有找到闭合，不代表状态不存在，而是尚未找到保持未来信息的较小商状态。

下一对象不再是新的标量特征表，而是守恒保持的精确向量协同树：

$$
\boxed{
v_{parent}
=
\sum_{c\in children(parent)}v_c
}
$$

每个节点必须记录：

```text
生成时间；
层；
全部 token 位置；
事件类型；
来源和接收角色；
精确向量引用；
父子关系；
分叉和汇合；
守恒残差。
```

必须新增 Q/K 分数、逐头精确写入、逐神经元精确写入、残差汇合、生成反馈和标签无关词表状态。四角色只保留为可视化投影视图，不再作为物理分母。

### 二、守恒细化规则

所有头和神经元先进入确定性索引树，不用任务标签或目标分数选择。只有出现下列情况才细分父节点：

```text
子节点抵消率高；
子节点方向多样性高；
下游重放误差超过重复噪声门。
```

必须保留精确向量交叉项，哈希能量不能作为终态。

### 三、未来和跨模型验证

未来测试从案例最近邻改为同一前向图重放：

```text
下一层向量重放；
下一生成时刻角色状态重放；
标签无关词表分布重放；
加入历史后剩余信息是否下降。
```

跨模型不要求相同层号、头号或神经元号，也不拟合任意旋转；只比较来源-接收角色、部分顺序、分叉汇合、守恒以及标准化干预响应。

### 四、阶段执行门

```text
Phase371A：每模型一个现有案例，只做工程重放，不产生科学主张；
Phase371B：冻结存储、数值和重放门；
Phase371C：371A/371B 全过后才建立新的独立发现-校准数据。
```

当前授权：

```text
现有账本小规模工程可行性：是；
新模型生成：否；
复用 Phase369 密封校准调参：否；
物理留出：否；
语言机制因果主张：否。
```

### 五、图谱、客户端与客观进度

研究图谱和客户端更新到 Phase370：

```text
允许显示：336 个验证动态束、事件/边分母、原始关系负结果、头/神经元诊断负结果；
禁止显示：把 Phase369/370 任何对象画成语言模式族路径；
新增 3D 神经元路径节点：0；
原始私有张量导出前端：否；
综合单一进度：N/A。
```

客观向量：

```text
注册语言族：9/9；
注册代表机制：18/18；
当前动态准入机制：4/18；
Phase369 测量账本：336/336；
原始关系未来门模型：0/3；
异构 Level 2 模型对：0/2；
物理确认机制：0/18；
因果密封机制：0/18；
严格注册闭合单元：0/72；
单神经元因果机制：0。
```

九族目录完成不等于九族物理路径完成；336 个有效动态束也不等于 336 条语言机制。科学上继续禁止用一个百分比合并这些不同分母。

### 六、验证

```text
Phase365 动态仪器测试：17/17 通过；
Phase369 协议、分母、账本、封存和客户端一致性测试：7/7 通过；
前端生产构建：通过；
Vite 大 chunk 警告：存在，但不是本阶段功能失败。
```

### 七、通俗总结

本轮先把三模型的内部运行完整临摹下来，数据是完整且能守恒重放的。随后测试“向量之间的夹角和大小比例能不能代表模型正在做什么”。答案是：少数平均数变好，但多数案例没有变好，三个模型也没有形成联合证据。

再把所有注意力头和神经元按固定规则分组，测试“每组用了多少能量”是否补足信息。答案仍然是否定的，因为模型可能依靠多个向量互相加强、抵消和补偿，只有能量多少看不到这些方向关系。

因此成果不是破解了语言编码，而是用大样本关闭了两条看似合理但不充分的路线，并把下一步缩小到更具体的问题：必须保留精确向量之间如何共同写入、在哪里分叉汇合，以及这些写入在同一前向图里怎样真实改变后续状态。

## Phase 372: 全词元精确 Q/K 守恒树与无损充分状态格式 [2026-07-11 20:33]

### 一、对 Phase364-368 审计的校正结论

附件的主判断正确：旧范数、粗深度和静态描述量是非充分投影的直接候选，不能再默认代表生成时间状态；新动态流束更接近真实计算，但 Phase369/370 已证明 Gram/范数关系和哈希能量仍然有损。

必须同时收紧三点：

1. 现有证据只否定已测试投影的充分性，不能否定所有 P0 非线性函数。
2. “允许模型特异坐标旋转”不能扩展成事后拟合任意旋转，否则任何两个有限样本子空间都可能被强行对齐；跨模型只能使用架构对称性或预注册功能响应。
3. 动态流束存在不等于语言规则已定位，更不等于因果闭合。

### 二、Phase371A：现有账本可行性审计

每模型选择一个既有 Phase369 发现案例，在早/中/晚三层和三个生成时刻重放，共：

```text
模型：3；
案例：3；
锚点行：27；
树分区：8；
新模型执行：0。
```

注意力与 MLP 的精确树为：

$$
\Delta h_{attn}=\sum_h W_{O,h}\left(\sum_s A_{h,r,s}V_{h,s}\right)
$$

$$
\Delta h_{mlp}=\sum_i p_i W_{D,:,i}
$$

连续索引分区只改变求和顺序：

$$
Parent=\sum_b Child_b
$$

结果：四角色位置上的精确树 27/27 通过；注意力树最大误差约为 $1.18\times10^{-7}$，MLP 树最大误差约为 $5.14\times10^{-7}$。但现有账本没有全词元接收状态，也没有实际进入注意力核的旋转后 Q/K，因此完整路径门失败。

### 三、Phase371B：实际 Q/K 与全词元三锚点采集

冻结后依次执行 qwen3、GLM4、DS7B，每模型一个既有案例、三个锚点、三个生成时刻。采集的是实际送入 eager attention 的旋转后 Q/K/V，而不是从投影输出近似猜测。

注意力概率重放：

$$
\widehat A=Softmax(QK^T\alpha+M)
$$

第一次 DS7B 审计出现 0.377 的概率误差，原因不是模型机制，而是脚本在矩阵乘法前把 BF16 Q/K 强制转换为 FP32，改变了真实数值路径。修复为保持模型原始 dtype 后，三模型 Q/K 概率重放最大误差均为 0；27/27 行全部通过。

三模型其他最大误差：

```text
Qwen3：attention 0.000410，MLP 0.000533，block 0.000238；
GLM4：attention 0.000394，MLP 0.000413，block 0.000370；
DS7B：attention 0.002325，MLP 0.003378，block 0.002464。
```

### 四、存储门与无损去冗余

直接物化逐头写入和八叉分区写入的三个工程案例占 809,648,661 字节，外推 336 个案例为：

$$
90,680,650,032\ bytes>64\ GiB
$$

因此原始物化格式严格失败。字段审计发现，约 77.7% 是可由 Q/K/V、概率、MLP 通道乘积和本地权重确定性重建的缓存，不是独立状态。

保留充分状态：

```text
七类全词元组件向量；
实际旋转后 Q/K/V；
全接收者到全来源注意力概率；
MLP 通道乘积；
确定性头/通道分区；
权重引用和质量门。
```

删除重复物化：

```text
逐头输出向量缓存；
注意力分区写入缓存；
MLP 分区写入缓存。
```

压缩后的 27 个工程文件为 180,748,902 字节，按实际文件外推 336 个案例为 20,243,877,024 字节。按需重建 27/27 通过，存储门通过。

### 五、阶段边界

本阶段确认：全词元精确状态、Q/K 分数、逐头写入树和逐神经元写入树在工程上可采集、可守恒、可回放。

本阶段没有确认：语言模式路径、未来状态充分性、跨模型功能等价、因果机制或闭合。校准和物理留出均未打开。

## Phase 373: 独立精确流束账本、相邻层连续性与懒路径图 [2026-07-11 20:33]

### 一、独立案例总分母

Phase371C 在模型执行前冻结 88 个并行组：

```text
机制：关系绑定、目标竞争、实体近因、数一致；
发现组：48；
校准组：24；
物理组：16；
每组：四条件 × 三模型；
总行为案例：1056；
发现案例：576；
校准案例：288；
物理案例：192；
既往提示重叠：0。
```

物理执行文件与非物理执行文件被物理分离，模型脚本只读取 864 个发现和行为校准案例。内部校准状态和物理案例保持密封。

### 二、三模型行为资格

严格按 qwen3、GLM4、DS7B 顺序执行，24 词元生成预算，完全相同词元长度分桶。GLM4 首次批量 8 在首批生成前退出码 139，未写入结果；显存释放后仅把工程批量降为 2，提示、解码和科学分母均未改变。

结果：

```text
Qwen3：255/288；
GLM4：276/288；
DS7B：268/288；
合计：799/864；
物理案例加载：0。
```

三模型共同四条件合格组：

```text
关系绑定：发现 11/12，校准 6/6；
实体近因：发现 11/12，校准 5/6；
数一致：发现 3/12，校准 3/6；
目标竞争：发现 1/12，校准 0/6。
```

冻结门要求发现至少 8 组、校准至少 4 组，因此只有关系绑定和实体近因进入内部发现。失败组没有被替换，四机制完整周期明确失败。

### 三、精确内部账本

两个合格机制形成 22 个并行发现组，共 264 个案例、每模型 88 例。基础三锚点账本：

```text
文件：3168；
字节：17,412,761,184；
锚点行：2376；
语义标签进入采集器：否；
完整无标签词表 logits：是。
```

三模型最大块误差：

```text
Qwen3：0.000255；
GLM4：0.000370；
DS7B：0.002500。
```

全部通过 0.01 数值门。

### 四、相邻层合同修复

三锚点只能证明单块守恒，不能验证“下一层向量重放”。在尚未进行语义候选搜索前，冻结并补采：

```text
早层：0 -> 1；
中层：floor(L/2) -> floor(L/2)+1；
晚层：L-2 -> L-1。
```

只增加每模型三层，不扩张到全层。新增 2376 文件、16,930,502,352 字节；基础与相邻账本合计 34,343,263,536 字节，低于 64 GiB。

独立重跑的贪心生成词元 264/264 与基础账本一致；2376 条“上一层输出 = 下一层输入”连续性最大相对误差在三个模型中均为 0。

这证明局部三段连续测量有效，不代表全层全局路径已经完成。

### 五、懒加载精确路径对象

为避免枚举数十亿事件造成重复存储，路径对象只保存原始张量引用、切片范围、权重引用和确定性推导规则。

结果：

```text
案例：264；
显式节点：38,808；
显式边：46,464；
隐式 Q/K 分数事件：913,988,976；
隐式逐头写入：11,455,344；
隐式逐神经元写入：5,248,129,536；
残差汇合事件：747,432。
```

没有使用 Top-K、哈希能量终态或语义标签，也没有把精确向量重复导出到客户端。

### 六、客观边界

当前完成的是两个机制、三个模型、三段相邻层和三个生成时刻的精确物理测量拼图。全层连续路径、其他 16 个代表机制、因果干预、校准复制和物理留出仍未完成。

## Phase 374: 盲六对发现、历史剩余信息强负结果与路线收束 [2026-07-11 20:33]

### 一、盲六对精确对比

在 A/B/C/D 含义不可见时，对每个模型组保留全部 6 个无序条件对和 3 种完美匹配。冻结并流式生成：

```text
模型组：66；
匿名条件对：396；
路线对比行：299,376；
完整词表对比行：1,188；
每匿名对路线行：756；
Top-K：否；
候选选择：否；
语义标签：否。
```

导航索引包括差向量范数、与父差向量的有符号余弦、内积份额、相邻输出方向保持，以及错误深度、错误角色和时间打乱控制。它们只用于定位，不能替代原始向量。

盲行 300,564/300,564 审计通过并密封哈希后，才冻结语义门并只打开新发现集的 A/B/C/D 映射；校准和物理条件键未打开。

### 二、发现门结果

冻结门要求：

1. 差向量非零，且对输出差的方向和内积贡献为正。
2. 正确相邻层关系分别胜过错误深度、错误角色和时间打乱。
3. A-B 与 C-D 两个词汇复制同时通过。
4. 两个正确配对的平均保持度高于其他四个配对。
5. 同一路线在每模型、每机制至少 8/11 个独立组复现。

语义映射后：

```text
语义路线组：49,896；
单模型临时候选：210；
异构 Level 2 临时候选：39；
三模型 Level 3 临时候选：3；
完整语言路径：0。
```

这些临时候选没有进入校准，因为历史剩余信息和因果重放尚未通过。

### 三、历史剩余信息门

只有生成时刻 t1 同时拥有过去 t0 和未来 t2，因此 210 条临时候选中只有 62 条具备历史门资格。t0 和 t2 不补算为通过。

对当前路线差 $x_1$、过去路线差 $x_0$ 和未来相邻输出差 $y_2$，使用基础几何投影：

$$
e_{current}=\frac{\|y_2-Proj_{span(x_1)}y_2\|}{\|y_2\|}
$$

$$
e_{history}=\frac{\|y_2-Proj_{span(x_1,x_0)}y_2\|}{\|y_2\|}
$$

$$
HistoryGain=e_{current}-e_{history}
$$

每个 A-B 和 C-D 复制都要求：

```text
当前误差 < 过去误差；
HistoryGain <= 0.01；
未来差非零；
至少 8 个独立组通过。
```

结果从 62 条 t1 单模型临时候选下降为 3 条模型特异路线：

```text
DS7B：实体近因，中层，当前生成位置，attention_partition_0，8 组；
GLM4：关系绑定，中层，当前生成位置，attention_merge，10 组；
Qwen3：关系绑定，早层，当前生成位置，mlp_partition_2，8 组。
```

三条路线的机制、深度或组件不同：

```text
异构 Level 2：0；
Level 3：0；
完整语言路径：0；
校准授权：否；
物理留出授权：否。
```

历史投影仍只是局部线性诊断，不是因果重放；即使三条单模型路线通过，也不能称为机制。

### 四、算法结论

本轮支持 Phase364-368 的核心诊断：旧标量和能量投影确实有严重信息损失，精确向量账本可以修复测量层。但结果同时否定了更强的新假设：单一路线差向量不是可跨模型复用的历史充分状态。

严格逻辑为：

$$
\text{单路线跨模型历史门失败}
\not\Rightarrow
\text{模型内部没有动态语言结构}
$$

更可能的下一候选对象是多个并行分支、补偿支路和汇合关系组成的精确子图状态，而不是某个头、某个神经元分区或某个组件向量。

### 五、当前客观进度向量

```text
注册语言族：9/9；
注册代表机制：18/18；
Phase371C 行为合格机制：2/4；
精确局部账本机制：2/18（仅表示已测量，不表示路径完成）；
严格跨模型语言路径：0/18；
物理确认机制：0/18；
因果密封机制：0/18；
严格闭合单元：0/72；
单神经元因果机制：0。
```

不同分母不能合并成单一百分比。此前管理性全局估计已在图谱进度文件中标记为科学完成度无效。

### 六、图谱与客户端

研究图谱、公共客户端和神经元图谱均同步到 `Phase371C-History`：

```text
允许显示：264 案例精确账本摘要、9 条模型层对、连续性、事件分母、发现门和历史门；
禁止显示：原始私有张量、盲六对私有行、把临时候选标为语言路径；
新增语言路径节点：0；
新增单神经元因果节点：0；
校准与物理显示：密封；
单一总体进度：N/A。
```

### 七、主要硬伤

1. 四个预注册机制只有两个通过跨三模型行为资格。
2. 当前只有三段相邻层，不是全层连续脉络。
3. 30 万导航索引仍是标量视图，虽可回到精确向量，但本身不充分。
4. 历史门是基础线性投影，不覆盖非线性状态转移。
5. 尚未执行同图因果干预，不能区分携带、相关和必要机制。
6. 三条历史路线模型特异，不能后验放宽对齐制造共同结论。
7. 当前模型较小，Qwen3 与 DS7B 具有架构亲缘，外推大模型和人脑均不成立。

### 八、下一阶段大任务

下一阶段应定义为：

```text
Phase375：多路线精确子图状态与预注册因果重放
```

阶段任务：

1. 停止继续枚举单路线。
2. 以完整注意力分支集、MLP 分支集、残差汇合和补偿关系组成子图，不允许事后任意组合。
3. 用守恒树的自然父节点和分叉边界定义有限子图集合，避免组合爆炸。
4. 在现有 22 个发现组上先完成子图历史门；加入历史后仍显著改善的子图直接淘汰。
5. 只有出现包含 GLM4 的异构 Level 2 子图，才允许进行小规模发现集同图干预。
6. 因果重放通过后才一次性打开密封校准；物理留出仍需校准复制后授权。
7. 在关系绑定和实体近因尚未出现跨模型子图前，不扩张到剩余语言族。

### 九、验证

```text
Phase371 新增一致性测试：8/8 通过；
Phase365 动态仪器回归：17/17 通过；
Phase369 原始拓扑回归：7/7 通过；
前端生产构建：通过；
图谱研究目录与前端镜像：一致；
神经元图谱校验和：通过；
已知 Vite 大 chunk 警告：仍存在，不影响本阶段数据一致性。
```

### 十、通俗总结

这轮先把“只看能量和平均数”改成“保存模型实际使用的 Q、K、V、每个注意力头和每个 MLP 神经元怎样写入”。工程上这件事已经做通，而且数据能一笔一笔加回原输出。

随后用全新题目测试。最初有 39 条看起来能跨模型对应的路线，3 条甚至三个模型都出现；但加入一个更严格的问题后，它们都没有留下：如果只看当前路线，过去状态还会不会额外解释未来？最后三个模型各剩一条不同路线，没有共同路线。

所以现在不能说已经破解语言编码。真正的收获是把问题进一步缩小：语言状态很可能不是一条线，而是一组同时分叉、补偿再汇合的路径。下一轮应研究这种有限精确子图，而不是继续从几十亿个节点里寻找一个“关键神经元”。

## Phase 375: 有限精确子图历史门与读出算法强负结果 [2026-07-11 22:44]

### 一、对 Phase369-374 统一分析的审计

附件的主线判断基本正确：范数、能量、Gram 关系、固定哈希拓扑和单路线都不足以充当跨模型语言状态；精确守恒账本应当成为后续研究的基础。

但必须收紧六个表述：

1. 当前只测量三个局部相邻层段，不是全模型连续重放。
2. “无损”只表示已测操作可由 Q/K/V、概率、MLP 乘积和权重引用重建，不表示保存了完整模型状态。
3. Gram 谱、有效秩和抵消率只能用于导航，不能替代精确向量状态。
4. 局部守恒重放只验证账本构造，不等于预测下一层或下一生成时刻。
5. 局部语言机制不必成为整个自回归模型的 Markov 充分状态。
6. 15%-18% 等单一总进度没有自然科学分母，继续判定为无效管理数字。

### 二、形成图和状态图分离

Phase375 首次严格区分：

```text
形成图：注意力头和 MLP 子写入如何形成父向量；
状态图：下游计算实际能够读取的边界残差和块内过渡向量。
```

注意力或 MLP 子节点在线性汇合后不能继续被下游分别读取，因此不能仅因为子节点集合能拟合未来，就把它登记成下游状态。

冻结四个状态模板：

```text
receiver_transition；
source_query_receiver_outputs；
binding_transition；
four_role_transition。
```

冻结三个形成模板：

```text
attention_children；
mlp_children；
joint_attention_mlp_children。
```

不允许任意组合头和神经元，不使用 Top-K，不训练探针。

盲清单结果：

```text
案例：264；
状态图对象：9504；
形成图对象：7128；
总对象：16632；
语义字段：0；
复制精确张量：0。
```

### 三、多向量历史门

设当前模板的精确差向量集合为：

$$
S_1=\{v_{1,1},\ldots,v_{1,k}\}
$$

过去集合为：

$$
S_0=\{v_{0,1},\ldots,v_{0,k}\}
$$

未来目标为下一生成时刻相邻接收层当前位置差向量 $y_2$。

使用基础 Gram-Schmidt 正交化，不拟合参数：

$$
E(S,y)=\frac{\|y-\operatorname{Proj}_{\operatorname{span}(S)}y\|}{\|y\|}
$$

冻结门同时要求：

```text
当前误差 <= 0.75；
至少两个独立基方向；
优于模板内最佳单向量 0.01；
优于过去、错误深度、错误角色和错误组各 0.02；
加入过去后的额外改善 <= 0.01；
A-B 与 C-D 两个词汇对同时通过；
每模型每机制至少 8 个独立组。
```

### 四、结果

三模型依次读取已有 CUDA 精确账本，未运行新提示：

```text
组级候选：792；
词汇对评估：1584；
绝对误差门通过：0/1584；
模型候选：0；
异构 Level 2：0；
Level 3：0；
因果重放授权：否。
```

当前误差分布：

```text
最小值：0.828719；
中位数：0.984209；
均值：0.969161；
最大值：0.999969。
```

多向量相对最佳单向量的改善：

```text
中位数：0.003603；
均值：0.008640。
```

这不是“差一点过门”。即使加入 3-6 个精确局部向量，未来向量仍几乎不在当前向量子空间内。

### 五、严格结论

Phase375 能够否定：

> 这些固定有限模板通过同坐标线性投影形成跨层、跨生成时刻的未来充分状态。

不能否定：

> 有限物理子图存在；这些子图对任务行为具有因果作用；非线性确定性转移能够携带状态。

算法瓶颈已经从“子图太小”上移到：

> 用同一坐标系中的几何投影，直接跨越非线性层变换和自回归词元插入，本身不是可靠的主因果读出。

## Phase 376: 答案决策时刻对齐与发现集直接因果转移 [2026-07-11 22:44]

### 一、固定生成时刻的语义错位

对 264 个内部发现案例重新读取自然生成记录，定位目标答案第一次完整出现的词元步。

结果：

```text
Qwen3：63/88 在 t0-t2 内；
GLM4：81/88 在 t0-t2 内；
DS7B：16/88 在 t0-t2 内；

三模型共同语义条件：88；
三个模型都在 t0-t2 内：16/88；
三个模型决策步完全一致：16/88。
```

DS7B 大多数目标在第 15-17 个生成词元才出现。原 `t0/t1/t2` 是词元偏移，不是跨模型共同语义事件。

因此 Phase374-375 的张量测量仍然真实，但结论必须改写为：

> 它们研究的是早期输出前缀动力学，不是答案决策机制。

### 二、决策对齐干预

每个案例使用实际自然生成轨迹，构造：

```text
answer_entry：尚未生成任何答案词元；
target_decision：恰好位于完成目标答案词元之前。
```

主要内容转移对：

```text
A -> C；C -> A。
```

直接路线控制：

```text
B -> D；D -> B。
```

冻结三个自然模板：

```text
当前位置残差输出；
来源+查询+当前位置残差输出；
当前位置注意力+MLP 联合写入。
```

对每个模板同时执行：

```text
正确深度/角色/时间；
错误深度；
错误角色；
错误时间。
```

设供体答案词元为 $d$，接收者答案词元为 $r$：

$$
M(x)=z_x(d)-z_x(r)
$$

转移增益为：

$$
G=M(\text{patch})-M(\text{baseline})
$$

冻结要求：

```text
正确转移增益 >= 0.10；
分别超过错误深度、错误角色、错误时间 0.05；
A->C 与 C->A 同时通过；
至少 8 个发现组。
```

### 三、发现结果

```text
案例：264；
模板×深度×转移对象：2376；
补丁前向条件：9504；
所有 hook 到达：是；

Qwen3 候选：5；
GLM4 候选：6；
DS7B 候选：6；

异构 Level 2 转移对象：6；
Level 3 转移对象：5；
异构 Level 2 重复赢家翻转：4。
```

关系绑定晚段当前位置残差：

```text
Qwen3：11/11 组通过，10/11 双向赢家翻转；
GLM4：10/11 组通过，10/11 双向赢家翻转；
DS7B：11/11 组通过，11/11 双向赢家翻转。
```

实体新近性晚段当前位置残差：

```text
Qwen3：11/11，赢家 11/11；
GLM4：11/11，赢家 11/11；
DS7B：10/11，赢家 6/11。
```

当前位置注意力+MLP 联合写入产生稳定增益，但没有达到重复赢家翻转。这说明答案内容在晚段已主要累积于残差状态，而不是只由最后一个块的新增写入决定。

质量异常完整保留：GLM4 有 1/88 个自然决策全前向重放词元与缓存生成记录不一致，该案例没有被删除，相关方向不能通过组门。

### 四、结论边界

这是直接因果内容转移的正结果，但对象紧邻输出端。因此只能登记为：

> 晚段当前位置残差内容载体候选。

不能登记为：

```text
上游语言规则；
自然必要路径；
完整生成充分状态；
单神经元机制；
语言编码闭合。
```

## Phase 377: 独立校准复制 [2026-07-11 22:44]

### 一、分母

只打开 Phase371 已完成行为资格审计的密封校准组：

```text
关系绑定：6 个三模型共同组；
实体新近性：5 个三模型共同组；
每模型案例：44；
总案例：132。
```

只复测发现阶段胜出的两个模板：

```text
晚段当前位置残差；
晚段来源+查询+当前位置残差。
```

阈值保持不变，独立组门按预注册校准要求设为至少 4 组。

### 二、结果

```text
模板×转移对象：264；
补丁前向条件：1056；
所有 hook 到达：是；

Qwen3：4/4 候选复制；
GLM4：4/4 候选复制；
DS7B：关系绑定 2 个候选复制，实体新近性未复制；

异构 Level 2 校准对象：4；
Level 3 校准对象：2。
```

关系绑定两个模板在三模型复制；实体新近性两个模板在 Qwen3+GLM4 复制。

质量异常：DS7B 有 2/44 个自然决策重放词元不一致，未删除案例。

校准复制只授权狭窄物理确认，不授权机制闭合。

## Phase 378: 物理留出确认、最小性审计与图谱校准 [2026-07-11 22:44]

### 一、物理留出

只打开行为严格正确的 96 个物理留出案例，依次运行 qwen3、GLM4 和 DS7B。关系绑定在三个模型确认晚段当前位置残差内容转移；实体新近性在 qwen3 和 GLM4 确认，DS7B 未达到冻结门。

### 二、最小性审计

对来源、查询和当前位置残差组合进行最小性比较后，来源和查询增量没有形成独立必要路径。可保留的最小对象是：

```text
晚段；
当前位置；
残差输出；
答案内容载体。
```

它是输出端点，不是上游规则、完整语言路径或神经元级闭合。图谱只显示终端内容载体，神经元图谱新增路径节点为 0。

## Phase 379: 全局复用差异轮廓与共有骨架混淆审计 [2026-07-11 23:25]

### 一、研究目标升级

本阶段接受以下优先级调整：

> 第一优先级不再是单个语言模式族闭合，而是不同功能在模型内部的复用、差异、分叉和汇合布局。

局部闭合保留为检验全局布局解释是否正确的工具。

### 二、旧分母审计

对 Phase330 的 15552 个案例按“三模型行为严格正确且全词表目标词元 rank=1”重新联合审计，只有 1 个案例满足。旧 485 万级工程事件不能作为跨模型科学布局证据，只保留工程覆盖意义。

冻结新的决策对齐分母：

```text
机制：relation_binding、entity_recency、number_agreement、target_vs_wrong；
平行组：43；
模型案例：516；
发现案例：336；
校准案例：180；
所有案例行为严格正确；
所有目标决策时刻已定位。
```

### 三、精确测量

每个目标决策时刻记录：

```text
所有层；
layer_input、attention_output、mlp_output、layer_output；
source、query、current 三个角色；
完整词表 logits；
完整向量；
不使用 Top-K。
```

原始事件权重：

$$
w(e)=\min\left(1,\frac{\lVert\Delta e\rVert}{\lVert\Delta T\rVert}\right)
\left|\cos(\Delta e,\Delta T)\right|
$$

共有骨架与功能残差：

$$
B_{m,a}=\frac{1}{K}\sum_k W_{m,a,k},\qquad R_{m,a,k}=W_{m,a,k}-B_{m,a}
$$

### 四、结果

```text
原始发现/校准余弦：最小 0.977818，中位 0.997318，均值 0.995017；
残差发现/校准余弦：最小 0.392541，中位 0.950896，均值 0.881508；
残差范数比例：中位 0.179592；
异构跨模型残差余弦：最小 -0.618150，中位 0.365694，均值 0.277852。
```

原指标使 36/36 个模型轮廓和 12/12 个机制轴对象全部通过。这个过强结果主要来自共有架构和残差骨架，不是功能机制复用。

### 五、严格结论

```text
原始轮廓复制是功能特异证据：否；
共有骨架混淆存在：是；
使用已消费校准集修正结论：否；
因果扫描授权：否；
单神经元扫描授权：否；
语言编码闭合：否。
```

下一阶段必须在全新分母上预先冻结骨架残差指标。

## Phase 380: 独立残差复制与终端接口因果定位 [2026-07-12 00:08]

### 一、全新行为分母

初始冻结 4 个机制、每机制 24 个平行组、每组四条件、三个模型，共 1152 个行为案例。number_agreement 初始共同合格组不足，在任何内部追踪前增加 864 个行为样本；原组全部退役，失败组没有替换，门槛没有降低。

最终：

```text
三模型共同合格平行组：65；
精确追踪案例：780；
relation_binding：23 组；
entity_recency：19 组；
number_agreement：9 组；
target_vs_wrong：14 组。
```

### 二、精确追踪质量

```text
qwen3：260/260 重放一致；
GLM4：260/260 重放一致；
DS7B：250/260 重放一致；
总精确事件向量：324480；
全配对事件行：486720；
重放合格平行组：57；
异常平行组：8，保留质量账本但排除轮廓结论。
```

### 三、独立残差复制

冻结指标未调参，得到 5 个异构跨模型稳定对象：

```text
entity_recency × content_change；
entity_recency × joint_change；
relation_binding × content_change；
relation_binding × joint_change；
target_vs_wrong × joint_change。
```

这些对象只是重复性残差轮廓，尚未建立因果复用。

### 四、组件级因果扫描

冻结 5 个相对深度、4 个组件边界、3 个位置角色，自然替换与等能量排列成对执行，并加入循环错误深度和错误角色控制。

```text
每模型条件：19200；
总条件：57600；
单方向门通过：2477；
模型单元通过：34；
异构 Level 2 单元：10；
Level 3 单元：10。
```

严格重新分类后：

```text
late/layer_input/current：终端接口前边界；
late/layer_output/current：终端接口后边界；
异构终端接口单元：10；
异构上游单元：0；
跨机制上游领地：0；
完整上游路径：0；
单神经元因果：0。
```

### 五、关键校准

`late/layer_input/current` 紧邻终端输出，不能称为上游规则。三个机制共享归一化终端位置，也不能称为共享相同物理神经元。

可视化数据分成三种证据：

```text
重复性残差轮廓：描述性；
late/current 输入和输出边界：组件级终端因果接口；
上游形成路径：未解决。
```

普通图谱和神经元图谱已同步；新增神经元路径节点为 0，单一全局完成百分比无效。

## Phase 381: 联合语义位置状态因果检验 [2026-07-12 00:43]

### 一、测试原理

Phase380 的单位置上游负结果可能来自状态同时分布在 source、query 和 current。为检验这一具体解释，在全新数据上同时比较：

```text
source；
query；
current；
source+query+current。
```

联合状态必须超过等能量排列、最佳单位置、循环错误深度、循环错误组件和非目标 logits 副作用门。每模型每单元要求至少 6 个组的四个转移方向全部通过。

### 二、全新分母与行为门

```text
基础行为案例：864；
与 Phase380 提示词重叠：0；
原始共同合格组：relation 21、entity 23、target 6；
target 低于 8 组门。
```

在内部追踪前冻结 288 个 target 行为扩展案例。三个模型扩展正确数：

```text
qwen3：95/96；
GLM4：93/96；
DS7B：68/96；
扩展共同合格组：7。
```

最终选择每机制 8 组，共 24 组、288 个追踪案例。只使用行为结果选组，失败组不替换，阈值不调整。

### 三、精确追踪与重放

```text
qwen3：96/96；
GLM4：96/96；
DS7B：93/96；
精确事件向量：119808；
重放合格组：22；
关系绑定：7；
实体新近性：7；
读出竞争：8。
```

2 个包含异常案例的平行组进入质量账本，不进入因果分母。

### 四、CUDA 因果结果

三模型依次运行并释放显存：

```text
qwen3：23040 条，全部 hook 到达；
GLM4：23040 条，全部 hook 到达；
DS7B：23040 条，全部 hook 到达；
总条件：69120；
单神经元扫描：否；
Top-K：否。
```

结果：

```text
联合方向门行：8640；
联合方向通过：563；
联合转移增益中位数：2.304688；
联合相对最佳单位置协同中位数：-0.014648；
模型单元：300；
模型单元通过：0；
任一模型单元最大完整重复组数：1；
异构跨模型单元：0；
上游联合状态单元：0。
```

八个单项门通过数：

```text
基础增益：6249；
超过等能量：6418；
终端份额：5644；
超过等能量终端份额：5887；
超过最佳单位置：1998；
超过错误深度：4128；
超过错误组件：4204；
副作用比：6679。
```

### 五、严格结论

联合全向量能够产生大原始内容转移，但没有稳定超过最佳单位置，也没有跨组重复。因此关闭：

> source+query+current 全向量联合替换是可复用上游状态算子。

不能由此否定所有分布式动态状态，只能否定这一具体静态联合算子。图谱将其显示为强负结果，不显示组件路径或神经元。

## Phase 382: 总层更新算子离线可辨认性审计 [2026-07-12 00:49]

### 一、动机

Phase381 说明扩大静态状态替换不能解决问题。为避免盲目继续消耗 CUDA，先使用 Phase381 精确轨迹检验最基本动态对象：

$$
U_{l,r}(x)=H^{out}_{l,r}(x)-H^{in}_{l,r}(x)
$$

四条件效应：

$$
E_{content}=\frac{(C-A)+(D-B)}{2}
$$

$$
E_{operation}=\frac{(A-B)+(C-D)}{2}
$$

$$
E_{interaction}=\frac{(A-B)-(C-D)}{2}
$$

每个机制冻结 4 个离线发现组和至少 3 个离线留出组。指标不拟合阈值，比较总层更新和静态层输入的功能残差轮廓。

### 二、结果

```text
重放合格平行组：22；
模型组：66；
转换事件行：20592；
轮廓：108；
发现/留出比较：54；
跨模型比较：54。
```

总层更新：

```text
自身机制胜出：12/27；
发现/留出中位余弦：0.004605；
异构跨模型中位余弦：0.163368。
```

静态层输入：

```text
自身机制胜出：13/27；
发现/留出中位余弦：0.327942；
异构跨模型中位余弦：0.234705。
```

三个无阈值改进判据全部失败：

```text
自身胜出数提高：否；
发现/留出中位数提高：否；
异构跨模型中位数提高：否。
```

### 三、算法结论

`layer_output-layer_input` 不是比静态状态更可辨认的候选算子。它把注意力路由、MLP 写入、残差继承和相互补偿合并成一个总向量，发生了过度压缩。

这与 Phase381 一起确认两个算法硬伤：

```text
全状态替换过宽，会注入通用骨架和内容；
总层更新过粗，会抵消和混合内部事件。
```

### 四、当前图谱进展向量

```text
九族工程注册：9/9，但不等于物理机制完成；
独立复制功能残差对象：5；
跨模型终端接口机制：3；
跨模型上游机制：0；
完整语言路径：0/18；
单神经元因果路径：0/18；
严格闭合单元：0/72；
单一全局完成百分比：无效。
```

### 五、问题、硬伤与智能理论边界

1. 当前只能看到答案内容进入终端接口，尚未看到知识、关系、语法和读出竞争如何在上游形成。
2. 归一化位置跨模型一致不等于相同物理层、头、通道或神经元。
3. 三模型均为小模型，Qwen3 和 DS7B 还有架构亲缘，路径可能粗糙或模型特有。
4. 尚未证明自然必要性、完整生成充分性、跨生成步稳定性和训练形成过程。
5. Phase288 全量单神经元 CUDA 干预没有上游路径授权，仍未运行。
6. 当前结果不能推出完整语言数学结构，更不能推出完整智能理论。

### 六、下一阶段大任务

Phase383 不再继续静态补丁或总层向量，应建立精确组件事件守恒账本：

$$
R_{l+1,p}=R_{l,p}+\sum_{s,h}C^{attn}_{l,h,s\rightarrow p}+C^{mlp}_{l,p}+\epsilon_{l,p}
$$

执行顺序：

1. 在无语义烟雾样本上分别为 qwen3、GLM4、DS7B 验证注意力来源词元贡献、MLP 写入与残差更新能否逐层重建。
2. 数值容差只依据 dtype 和无语义重建误差冻结，不依据语言结果调参。
3. 保留来源词元、目标词元、组件、层和生成时刻，不先池化为整层范数。
4. 守恒后再构造新的四条件事件算子，并先预测全新留出终端接口变化。
5. 只有预测独立复制，才授权新 CUDA 因果干预。
6. 只有上游组件事件路径复制，才授权稀疏头、MLP 通道和单神经元定位。

Phase382 已完成本轮阶段目标。Phase383 属于新的仪器和数据契约阶段；在组件事件守恒成立前，不应继续自动扩大模型干预。
