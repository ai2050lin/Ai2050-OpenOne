# AGI GPT5 Memo

## Phase 330-UI1: 恢复左侧语言模式族物理叠层控制器 [2026-07-10 08:49]

### 任务定位

本次是 Phase330 客户端同步工作的界面补记，不占用已经注册的 Phase331 机制实验编号。任务是把“语言模式族物理叠层”恢复到左侧控制面板，不再作为主 3D 画布底部浮层。

本次只调整控制器的界面归属和响应式布局，不回退真实物理单元投影、组件专属形状、RMSNorm/GQA/SwiGLU 修正和 Trace 数据。

### 修改

```text
1. 在左侧“研究驾驶舱”的语言机制路线中挂载 PatternFamilyAtlasControls；
2. 控制器位于研究路线下拉框之后；
3. 删除 Canvas 上方原有的底部绝对定位控制器实例；
4. 左侧不重复显示模型选择，模型仍由运行配置统一控制；
5. 左侧隐藏长统计 footer，详细数据继续由右侧说明与数据窗口承担；
6. 保留模式族、证据范围、候选上限和叠层显隐控制；
7. 禁用叠层后，恢复按钮仍保留在左侧面板；
8. 删除移动端“启用叠层后隐藏工作面板”的旧规则。
```

`PatternFamilyAtlasControls` 新增可复用参数：

```text
variant="floating" | "panel"
showModel=true | false
showDetails=true | false
```

左侧使用：

```text
variant="panel"
showModel=false
showDetails=false
```

### 响应式修正

第一次移动后，DOM 归属正确，但 `max-width:1280px` 的旧浮层规则仍给控制器施加 `left:330px`、`right:330px` 和 `min-width:460px`，导致横向溢出。

panel 变体现在固定使用：

```text
left/right/bottom: auto;
width: 100%;
min-width: 0;
```

并恢复左侧字段标签，使 panel 与 floating 分别使用自己的响应式规则。

### 浏览器验证

1280×720 视口实测：

```text
控制器总数：1；
floating 控制器：0；
属于 workspace-panel--input：true；
左侧输入面板范围：x=20 到 x=414；
叠层控制器范围：x=37 到 x=395；
叠层控制器宽度：358px；
fitsInputWidth：true。
```

这说明控制器不仅在 DOM 中属于左侧，而且视觉边界也完整落在左侧面板内。

### 工程验证

```text
PatternFamilyAtlasControls / ResearchPlaybackPanel 定向 ESLint：通过；
git diff --check：通过；
Vite production build：3668 modules transformed，构建通过；
客户端地址：http://127.0.0.1:5176/。
```

`App.jsx` 全文件 ESLint 仍有历史遗留的未使用变量、空块、旧函数和未定义 `getColor` 等问题。这些不是本次移动产生，生产构建可通过；为避免扩大修改范围，本次未清理。

### 严格边界

```text
没有运行 CUDA 模型；
没有修改图谱数据或候选筛选算法；
没有改变证据等级；
没有增加或删除 H#/N#/G# 物理候选；
这是客户端布局恢复，不是新的机制研究结果。
```

### 结论

“语言模式族物理叠层”已经恢复为左侧研究驾驶舱中的主控制模块，画布底部不再存在独立浮层。左侧负责叠层配置，右侧负责说明和详细数据，主 3D 空间继续只承担模型、动画和物理单元显示。

## Phase 330-UI2: 证据范围分段按钮改为下拉框 [2026-07-10 10:12]

### 修改

左侧“语言模式族物理叠层”中，原位于模式族下方的五个证据筛选按钮改为单个“证据范围”下拉框。

下拉选项保持原值和原筛选语义：

```text
key -> 关键候选；
natural -> 自然交叉；
registered -> 注册集合；
cross_model -> 跨模型；
competition -> 竞争路径。
```

只改变控件形态：

$$
evidenceFocus_{new}=evidenceFocus_{old}
$$

图谱节点筛选、物理地址、候选上限和证据数据均未变化。下拉框在当前模式族尚未完成物理映射时继续禁用，行为与原五个按钮一致。

### 工程清理与验证

```text
删除 pattern-atlas-segment 及五按钮相关 CSS；
新增 aria-label="证据范围" 的原生 select；
PatternFamilyAtlasControls 定向 ESLint：通过；
git diff --check：通过；
Vite production build：3668 modules transformed，构建通过。
```

### 严格边界

```text
没有运行 CUDA 模型；
没有修改图谱数据；
没有改变五种证据范围的定义；
没有改变机制结论；
本次仅为客户端控件整理。
```

## Phase 331: IntelligentTheory 语言原理审计与可证伪破解路线 [2026-07-10 12:49]

### 任务

审计 `research/IntelligentTheory.md` 中关于语言原理、语言编码机制、语言模式图谱和非线性体系的分析，区分已经进入证据链的结论、仍待验证的工作假设与仅用于组织实验的概念，并给出可执行的破解路线。

### 当前理论中最有价值的主线

文件最重要的进步，是把语言编码从“固定语义轴或 token 向量存储”修正为如下动态计算：

$$
x_{\le t}
\rightarrow S_t
\rightarrow R_t
\rightarrow C_t
\rightarrow Q_t
\rightarrow G_t
\rightarrow y_{t+1}
$$

其中：

```text
S_t：由身份、角色、关系、构式、操作符、作用域和绑定构成的条件状态；
R_t：当前上下文选择的物理路线；
C_t：attention、MLP、residual、norm 和 W_U 组成的组件计算；
Q_t：target、wrong、prose、echo、continue、stop 等候选竞争；
G_t：格式协议、读出与自然生成闭合门；
y_{t+1}：下一 token，并继续进入后续自回归状态。
```

这条主线与已有实验中“固定方向跨层失效、路径依赖、候选竞争、读出门控、自然生成不等于首 token margin”相容，适合作为后续统一研究骨架。

### 需要严格降级的内容

1. 隐藏状态加法分解

$$
h_l=I_l+R_l+F_l+C_l+O_l+S_l+K_l+B_l+Q_l+N_l+\varepsilon_l
$$

目前主要是概念记账式分解。各项未满足唯一性、可识别性和独立干预条件，不能解释为模型内部真实存在的十个可分离模块。

2. 九大语言模式族

它们是实验覆盖分类，不是语言的自然完备基元。必须经过 open-set、mixed-family 和未知族检验后才能冻结。

3. 低维流形、语法纤维和非交换代数

这些结果可以提供测试假设，但当前受样本覆盖、句式混杂、数值估计和模型特异性限制，不能直接提升为语言的最终数学结构。

4. 图谱完成度加权分数

$$
Score_{base}=0.10B+0.10R+0.15L+0.20C+0.25I+0.10G+0.10K
$$

权重是工程优先级，不是从模型机制推导出的自然常数。它适合项目管理，不应作为理论成立的数学证据。

5. ClosureGate

$$
SemanticDone\land StopWins\land ContinueSuppressed\land RolloutStable
$$

这是严格的结果验收条件，但只说明输出闭合，不自动说明已找到造成闭合的最小内部机制。

### 可行的破解对象

不再以“单神经元是否表示颜色”作为最终问题，而把最小破解对象定义为条件化因果状态机：

$$
z_{l+1}=F_l(z_l,x_{\le t};\theta_l),
\qquad
q_{t+1}=W_U\,Norm(z_L)
$$

要破解的是能够预测下列三件事的局部机制：

```text
1. 在给定上下文下，哪组真实组件和神经元会被选择；
2. 它们如何改变候选词或候选短语之间的竞争；
3. 对它们进行最小干预后，完整自然生成是否按预测改变。
```

### 系统破解方案

#### 第一阶段：冻结问题和数据

先选择颜色属性作为标定任务，但不能只使用“物体是什么颜色”一种模板。建立对象、关系、值、格式和干扰因素的平衡因子设计；固定 discovery、validation、private heldout 三个集合；三个模型分别执行，禁止用同一批 heldout 反复调参。

#### 第二阶段：记录完整前向计算

对每个样本、每个生成步保存真实物理地址：

```text
model/layer/component/head-or-neuron/token-position/generation-step；
residual input/output；
QK attention logits 和 softmax 权重；
V 与 W_O 写入；
MLP gate/up 激活和 down projection；
RMSNorm 输入、缩放和输出；
全词表 logits、候选短语似然与自然 rollout。
```

原始张量不全部常驻主表；主表保存索引、摘要和来源，原始数据采用分片文件。

#### 第三阶段：发现可重复的条件路径

使用基础差分与一致性分析，寻找在对象变化、模板变化和候选变化后仍能重复出现的物理单元集合。候选发现必须同时满足层时序、组件方向和读出结果一致，不能只按激活幅度取 Top-K。

#### 第四阶段：因果拆解

对候选执行 zero、half scaling、mean replacement、random-same-norm、permutation、cross-sample patch、方向移除和组合干预。必要性和充分性分开计算：

$$
Nec(U)=M_{base}-M_{ablate(U)}
$$

$$
Suf(U)=M_{repair(U)}-M_{corrupt}
$$

其中 $M$ 至少同时包含正确短语与最强错误短语的 log-likelihood margin、自然生成正确率和副作用。只有相关、必要、受控充分三类证据一致，单元集合才进入机制候选。

#### 第五阶段：恢复中层状态机

把通过因果验证的单元按功能归纳为 reader、router、carrier、writer、suppressor、gate，但这些名称必须由干预行为定义。对每条边学习最简单的可预测局部转移：

$$
\widehat{\Delta z}_{l,U}=f_U(z_{l-1},\,prompt\ features)
$$

然后在 heldout 样本上预测激活单元、作用符号、主要竞争者、干预效果和失败类型。预测失败就否定或缩小 claim，不用新术语解释失败。

#### 第六阶段：形成最小闭合电路

用逐步删减和组合审计寻找满足以下条件的最小集合 $U^*$：

$$
U^*=\arg\min_U |U|
$$

约束为：

$$
Nec(U)\ge\tau_n,
\quad Suf(U)\ge\tau_s,
\quad SideEffect(U)\le\tau_e,
\quad RolloutStable(U)=1
$$

同时检查补偿路径和非线性缺口。若组合效果不能由单元效果预测，就把交互边而不是单个节点登记为机制对象。

#### 第七阶段：从颜色推广到语言基元

颜色闭合后，依次复制到类别、关系、否定/作用域、句法角色、输出协议和多步组合。跨任务复现的 reader/router/gate 才可能是语言通用结构；只在单任务成立的部分保留为 family-specific mechanism。

### 破解成功的最低判据

```text
能从未见 prompt 预测真实物理路径；
能预测主要候选竞争和失败类型；
最小干预按预测改变完整自然生成；
随机、同范数、负族和模板对照不产生同样效果；
在独立留出数据上复现；
能够区分跨模型共性与模型特异实现；
机制模型比“层均值、激活 Top-K、线性 probe”等简单基线有明确增益。
```

### 严格结论

`IntelligentTheory.md` 已经具备较成熟的实验组织框架，尤其是 PatternPath、Claim Registry、证据等级、闭合门、heldout 预测和非线性审计。但它目前主要回答“应该记录和验证什么”，还没有回答“语言的最小生成规则究竟是什么”。最可行的突破不是继续扩大总图谱或增加抽象公式，而是完成一个模式族的条件化最小因果电路，并证明该电路能够跨样本预测、受控干预和自然生成闭合。

### 下一阶段大任务

建立“颜色条件化因果状态机基准”：冻结大样本因子数据、完整记录三个模型的真实计算路径、发现并因果删减物理单元集合、拟合最简单的局部状态转移、在 private heldout 上预测路径和干预结果，最后将证据同步到图谱与 3D 客户端。该任务应作为一个完整里程碑推进，不再拆成只增加单个统计字段或单个界面控件的小 Phase。

### 本轮边界

```text
未运行 CUDA 模型；
未产生新的模型实验数据；
未修改 IntelligentTheory.md 的理论正文；
本轮结论来自对现有理论结构、公式、证据边界和可证伪性的审计。
```


## Phase 332: 主三维研究空间的证据驱动改进方案 [2026-07-11 02:25]

### 任务与输入

本阶段不修改客户端，不运行 CUDA。综合以下材料形成下一版 3D 主空间的实施合同：

```text
research/IntelligentTheory.md；
research/MainAnalysis/20260711_01_同类项目比较.md；
research/MainAnalysis/20260711_02_三个工程难点.md；
Phase323-330 的真实物理图谱、证据边界和客户端记录；
当前 frontend React Three Fiber 场景、图谱 Renderer、播放面板和数据契约；
Transformer Debugger、Neuronpedia、circuit-tracer、LLM Transparency Tool、CircuitsVis。
```

### 当前客户端已有能力

```text
1. 模型快照包含真实层数、hidden size、intermediate size、Head 数和模型哈希；
2. 图谱节点能绑定 model/layer/component/unit_kind/unit_index；
3. H#、N#、G# 已区分 Attention Head、MLP product neuron 和组件组；
4. PatternFamilyNeuronAtlasRenderer 使用 InstancedMesh 批量渲染；
5. 证据范围、模式族、模型、候选上限和回放来源已有控制入口；
6. 主模型、Layer 内部模型、组件模型已经形成三级下钻；
7. 详情能够显示 evidence level、causal scope、source artifact 和 Phase327-330 边界。
```

这些能力应保留，下一版不重新制作一套独立 3D 客户端。

### 当前关键问题

#### 真实地址与显示位置混淆

现有节点具有真实张量地址，但层内位置由 unit index、黄金角和固定半径生成。该坐标是可重复的 UI layout，不是模型中神经元的物理欧氏坐标。Transformer 权重本身也没有大脑式三维空间位置。

下一版必须同时保存：

```text
physical_address：真实 model/layer/component/unit_kind/unit_index；
display_coordinate：客户端为了可读性生成的逻辑三维坐标；
coordinate_semantics：logical_tensor_layout，不得标记为 biological/physical_xyz。
```

#### 动画时间轴过粗

当前统一播放主要按 layer 推进，不能准确表达一个自回归生成步中的 token position、QK、softmax、V/O、MLP gate/up/product/down、residual 和 readout 顺序，也不能表达多个生成 token。

#### 聚合图谱与单次运行混合

Phase330 分区适合展示跨样本聚合路径，但 `neuron_events` 中的运行和事件不足以完整恢复每个 case、每个 generation step、每个干预条件的真实动画。聚合候选不能在播放时伪装成当前样本正在激活。

#### 组件详情仍有演示值

`ComponentDetailPanel3D` 中存在按层比例生成的功能标签和模拟运行参数；`LayerExplodedView3D` 仍只显示少量 `top_neurons`。当真实 Trace 可用时，这些演示值必须退出；数据缺失时应显示 unavailable，而不是生成近似数值。

#### 3D 承载过多文本和精确矩阵

注意力矩阵、全词表竞争、干预前后数值和来源审计在 2D 表格/矩阵中更准确。3D 应负责空间定位、路径导航和时间回放，不应替代所有分析面板。

### 开源项目可直接参考的设计原则

```text
Transformer Debugger：把 React neuron viewer、activation server 和单组件页面分开；
Neuronpedia：把 inference、graph、search/filter、dashboard、export 和 API schema 分层；
circuit-tracer：对图进行阈值剪枝，支持 pin、group、annotation 和 intervention；
LLM Transparency Tool：从选中 token 建贡献图，可点击 edge/head/FFN/neuron，并查看词表促进与抑制；
CircuitsVis：用可复用 2D React 组件表达 token、Attention 和精确数值。
```

本项目的差异化不应是“把所有对象都画成立体球”，而应是：真实模型结构、真实物理地址、自回归时钟、因果 A/B 回放和模式族图谱在同一主空间中联动。

### 目标架构：一个主空间，四级语义缩放

#### Level 0：全模型空间

显示真实模型层数和层结构。每层只显示聚合状态：当前 token、Attention/MLP/Residual 活动量、目标排名、主要 blocker、证据数量。适合观察路径在哪些层开始、增强、反转和结束。

#### Level 1：Layer 内部空间

点击 Layer 后原位展开真实计算顺序：

```text
residual input
-> pre-attention norm
-> Q/K/V
-> attention logits
-> softmax
-> V aggregation/W_O
-> residual add
-> pre-MLP norm
-> gate/up
-> product neurons
-> down projection
-> residual add/output。
```

不同架构按模型快照显示 GQA/MHA、RMSNorm/LayerNorm、SwiGLU/其他 MLP，不使用统一假模型覆盖架构差异。

#### Level 2：组件空间

Attention 显示真实 H# 地址、query token 到 source token 的边、pre-softmax logit、softmax weight、V/O 写入和目标/阻挡者贡献。MLP 显示真实 N#/G#、gate、up、product、down write 和干预差值。Norm 与 Residual 只显示维度/向量流，不伪装成神经元。

#### Level 3：机制子图空间

只显示当前 case 和当前 claim 的最小子图：trigger、reader/router/carrier/writer/suppressor/gate、目标候选、blocker、stop/continue 和 closure。节点仍锚定真实模型结构；抽象功能标签只是 annotation，不生成虚构物理节点。

相机使用 semantic zoom：双击 Layer 进入 Level 1，双击组件进入 Level 2，Pin 后形成 Level 3 子图。返回时保持原模型、case、token 和选择状态。

### 五类叠层必须分开

```text
Structure：模型真实静态结构，始终存在；
Trace：单次自然运行的真实事件；
Causal Delta：baseline 与 intervention 的差值；
Atlas：跨样本聚合候选和 Claim；
Evidence Boundary：证据等级、对照、反例和未完成状态。
```

Atlas 节点不能因当前播放经过同层就自动发光。只有 Trace 事件明确包含该地址时才显示自然激活；只有干预记录明确包含该地址时才显示因果差值。

### 统一时钟

动画事件键升级为：

$$
t=(generation\_step,token\_position,layer,component\_phase,event\_order)
$$

播放顺序：

```text
输入 token 编码；
每个 Layer 内部组件顺序；
final norm 与全词表读出；
候选竞争与 winner；
生成下一 token；
进入下一 generation step。
```

时间轴需要支持：播放、暂停、单事件步进、单层步进、单 token 步进、跳到读出、跳到干预点、速度调整和 A/B 同步。

### 基线与干预双轨回放

同一 case 可选择 baseline、zero、half、mean replacement、random same norm、wrong layer、wrong donor、restore。3D 中不要复制两套完整模型，使用同一结构上的双轨编码：

```text
细实线：baseline 路径；
粗实线：intervention 后增强；
衰减虚线：intervention 后减弱；
分叉线：补偿路径；
双线边：通过中介恢复；
灰色点线：仅观测关系。
```

颜色职责固定：色相表示目标促进/目标抑制/中性，亮度表示效应大小，轮廓和线型表示证据等级，透明度表示跨样本支持率。禁止用同一种颜色同时表达激活大小、因果等级和模式族。

### 数据契约 v2

建议拆成六类对象：

```text
model_snapshot.v2；
trace_run.v2；
trace_event.v2；
physical_unit.v2；
causal_intervention.v2 / causal_edge.v2；
claim_registry.v2。
```

`trace_run.v2` 至少包含：

```json
{
  "run_id": "...",
  "case_id": "...",
  "model_revision": "...",
  "tokenizer_revision": "...",
  "prompt": "...",
  "chat_template": "...",
  "target_sequence": [0],
  "competitor_sequences": [],
  "generation_config": {},
  "condition_id": "baseline",
  "source_artifacts": []
}
```

`trace_event.v2` 至少包含：

```json
{
  "run_id": "...",
  "case_id": "...",
  "generation_step": 0,
  "token_position": 0,
  "token_id": 0,
  "layer": 0,
  "component_phase": "mlp_product",
  "event_order": 0,
  "physical_addresses": [],
  "metrics": {},
  "top_competitors": [],
  "tensor_chunk_ref": "...",
  "evidence_level": "L2",
  "source_artifact": "..."
}
```

精确 QK 矩阵、Attention pattern、神经元数组和全词表 logits 不内嵌进巨大 JSON；使用按 `run/case/generation_step/layer/component` 分片的 Arrow/typed-array 二进制块，JSON 只保存索引、摘要和校验值。客户端按下钻层级延迟加载。

### 前端模块改造

```text
PatternFamilyNeuronAtlasRenderer
  -> ArchitectureRenderer
  + TraceEventRenderer
  + CausalDeltaRenderer
  + AtlasEvidenceRenderer
  + SelectionRenderer；

nodePosition
  -> PhysicalLayoutRegistry，明确 address 与 display coordinate；

ResearchPlaybackPanel
  -> 增加 case、generation token、component event、intervention condition 和 A/B 控制；

LayerExplodedView3D
  -> 只读取真实 model snapshot 与 trace event，移除固定 Top-8 展示逻辑；

ComponentDetailPanel3D
  -> 删除按层比例生成的模拟语义标签，改为真实值或 unavailable；

usePatternFamilyNeuronAtlas
  -> 保留聚合图谱职责；新增 useTraceRun、useTraceChunk、useCausalComparison；

SceneState
  -> 统一 model/run/case/generationStep/layer/component/address/claim/condition。
```

右上角只放二级操作：Pin、隔离、A/B、聚焦路径、重置相机。右下角详情窗口显示物理地址、当前值、baseline delta、intervention delta、证据边界、样本数、模型哈希和来源文件。3D 内只保留必要层号、token 和选中地址，不堆放解释段落。

### 大规模渲染策略

```text
Level 0 只画层聚合，不画全部神经元；
Level 1 只画当前 Layer 的组件和候选集合；
Level 2 使用 InstancedMesh、instanceId picking 和按证据筛选；
边按阈值剪枝，Pinned 节点永不被剪枝；
使用 Web Worker 解码数据分片；
缓存当前 run 相邻 Layer 和相邻 generation step；
大量节点使用 LOD，标签只对 hover/selected/pinned 显示；
跨模型比较使用归一化深度，但详情始终显示真实 Layer 地址。
```

### 与开源工具的集成边界

circuit-tracer 的 Qwen3 transcoder feature 可作为独立 `feature_inference` 叠层，用于发现候选子图；它不是本项目的真实 MLP neuron，因此不得合并成 N#。SAE/transcoder feature、真实 Head、真实 product neuron 和组件组必须使用不同 node_kind 和证据命名空间。

CircuitsVis 类二维视图可用于右侧详情中的 colored tokens、Attention matrix 和 token-to-token edges。3D 负责回答“发生在哪里、按什么顺序传播”，二维详情负责回答“精确数值是多少”。

### 实施顺序

```text
M1 Truth Contract：冻结地址、坐标、事件、证据和颜色/线型语义；
M2 Event Clock：完成逐 token/逐组件回放，禁止演示数据混入真实 Trace；
M3 Semantic Zoom：完成全模型 -> Layer -> 组件 -> 机制子图下钻；
M4 Causal A/B：接入 baseline/intervention、补偿路径和中介边；
M5 Detail Workbench：接入词表竞争、Attention 矩阵、神经元详情和来源审计；
M6 Scale and Compare：分片加载、LOD、跨模型归一化对照；
M7 HCPE Integration：把未来最小因果集合和 private-heldout 预测写入同一空间。
```

### 验收标准

```text
任一发光节点都能追溯到真实 trace event 或 intervention row；
任一 H#/N#/G# 都能解析到模型哈希和张量地址；
显示坐标明确标记为逻辑布局；
causal=false 的边永远不使用因果实线样式；
动画事件数量、顺序和 token 与源文件完全一致；
baseline/intervention 数值可以复算备忘录中的 margin 与 rollout；
聚合 Atlas 与单次 Trace 不混淆；
无真实数据时显示 unavailable，不生成演示结论；
桌面和移动视口无文本遮挡；
Playwright 截图和 canvas pixel 检查通过；
典型研究场景保持稳定交互，Pinned 节点和来源详情可访问。
```

### 严格结论

当前 3D 客户端不需要推倒重建。它最有价值的升级不是增加更多造型和粒子，而是建立“真实地址、真实事件、真实证据、真实时间”的统一合同。3D 主空间应成为模型计算与机制证据的导航器，精确矩阵和数字交给联动详情面板。只有当 HCPE 或后续干预生成了单元级因果结果，节点才能从候选样式升级为因果样式。

### 本轮边界

```text
未修改 frontend；
未启动开发服务器；
未运行浏览器视觉验证；
未运行 CUDA；
未修改现有图谱数据或证据等级；
本阶段产物是可实施的 3D 架构方案，不是已经完成的客户端功能。
```
