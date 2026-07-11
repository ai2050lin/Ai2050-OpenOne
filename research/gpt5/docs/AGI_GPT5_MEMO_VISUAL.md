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
