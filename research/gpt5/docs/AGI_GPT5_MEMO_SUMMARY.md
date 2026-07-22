我按 `research/gpt5/docs/AGI_GPT5_MEMO.md` 中 Phase 195 到 Phase 278 的主线整理。整体上，这一段不是单一测试，而是从“记录颜色特征神经元”升级到“构建语言模式族谱 Pattern Family Atlas”的过程。

**总路线**
Phase 195-278 的推进逻辑可以概括为：

```text
单点神经元记录
-> 全链路 Trace
-> 输出闭合与停止闭合
-> 语言模式 Pattern 建模
-> StateWrite / Readout / Competitor 机制拆解
-> Pattern Family Atlas 数据化
-> Done / Stop / Continue 竞争机制
-> 物理路径图谱 v2
```

核心变化是：研究对象从“哪些神经元激活”变成了“一个语言模式如何被触发、路由、写入、竞争、读出、终止”。

---

**阶段一：Phase 195-203，全链路闭合 Trace**
核心问题：  
最初只找“最大的几个神经元”，但这不足以解释模型为什么输出某个答案。所以 Phase 195 开始把目标改成记录完整计算链路：

```text
上文状态
-> 条件化路由
-> 候选空间打开
-> 知识路径激活
-> 输出边界竞争
-> 下一 token
```

测试原理：  
在每一层保存 residual state（残差状态）、attention output（注意力输出）、MLP output（前馈输出），然后把这些中间状态投影到输出矩阵 `W_U` 上，看它们对目标 token 和竞争 token 的贡献。

核心公式：

$$
score_l(token)=h_l \cdot W_U[token]
$$

含义：第 `l` 层残差状态 `h_l` 对某个 token 的输出支持度。

$$
p_l(c)=softmax(\{h_l \cdot W_U[c] \mid c \in Colors\})
$$

含义：在颜色候选集合中，每个颜色 token 的相对概率。

$$
K_l=score_l(target)-\max_{c \neq target}score_l(c)
$$

含义：目标颜色相对于最强竞争颜色的 margin（边界优势）。

$$
M_l=1-\frac{H(p_l)}{\log |Colors|}
$$

含义：候选空间是否集中。越接近 1，说明模型越明确地偏向某个候选。

这一阶段的关键发现是：  
“颜色候选空间闭合”不等于“全词表输出闭合”。也就是说，目标颜色可能在颜色集合里赢了，但在全词表里仍然输给解释词、标点、格式词或其他 token。

---

**阶段二：Phase 204-208，停止机制与输出协议**
核心问题：  
模型不只是要答对，还要“正确结束”。Phase 204 之后开始区分：

```text
输出了句号
≠ 模型真的停止
```

很多模型会输出 `.`，但后面继续解释、重复、漂移。

测试原理：  
记录生成轨迹中 stop token、continue token、prose token、echo token 的竞争情况，看模型是否真正进入终止状态。

核心公式：

$$
m_{stop}(t)=\max_{v \in V_{stop}}z_t(v)-\max_{v \in V_{prose}}z_t(v)
$$

含义：停止 token 相对解释性文本 token 的优势。

$$
m_{prose}(t)=\max_{v \in V_{prose}}z_t(v)-\max_{v \in V_{stop}}z_t(v)
$$

含义：继续解释的倾向是否压过停止倾向。

$$
StopExecuted(x)=\exists t[EOS_t=1 \lor (Period_t=1 \land \neg Continue_{t+1:T})]
$$

含义：真正的停止不是出现句号，而是句号后没有继续生成。

阶段结论：  
模型的语言能力里存在一个独立的 stop execution（停止执行）机制。它和语义答案机制不是一回事。

---

**阶段三：Phase 209-218，语言模式 Pattern 建模**
核心问题：  
为什么同一个问题，有时模型回答一个词，有时解释一长段，有时重复题目？  
Phase 209 开始提出：模型内部不是只有“答案机制”，而是多个语言模式在竞争。

例如：

```text
answer-only 模式
explain 模式
repeat 模式
list 模式
chatty 模式
protocol-follow 模式
```

测试原理：  
把生成过程表示成多个 Pattern（语言模式）的动态混合，每个 Pattern 有自己的触发条件、状态变量、输出约束和失败模式。

核心公式：

$$
Pattern=(Trigger,StateVariables,FeatureTrajectory,PriorityProxy,OutputConstraint,FailureModes)
$$

含义：一个语言模式不是单个神经元，而是一套动态结构。

$$
h_{t+1}=\sum_{k \in P}\alpha_k(x,t)T_k(h_t)+\varepsilon_t
$$

含义：下一步状态由多个模式共同作用产生，`α_k` 是第 `k` 个模式的激活权重。

$$
P_k(x,t)=[\alpha_k(x,t),\phi_k(x,t),\Delta h_k(x,t),b_k(x,t),o_k(x,t)]
$$

含义：每个模式包含激活强度、内部特征、状态写入、边界竞争和输出倾向。

这一阶段的重要转折：  
研究对象从“答案 token”升级成“语言模式族”。

---

**阶段四：Phase 219-233，StateWrite / Readout / Competitor 机制拆解**
核心问题：  
Pattern 不是抽象概念，必须落到模型内部的物理路径上。于是开始拆：

```text
谁写入状态？
写入到哪里？
怎么进入 readout？
竞争 token 从哪里来？
```

测试原理：  
把 success case（成功样本）和 drift case（漂移样本）做差，提取内部方向，然后干预这些方向，观察行为是否改变。

核心公式：

$$
v_{l,t}^{S-D}=E[h_{l,t}\mid success]-E[h_{l,t}\mid drift]
$$

含义：成功状态和漂移状态之间的残差方向差。

$$
u_{m,l,t}^{S-D}=E[O_{m,l,t}\mid success]-E[O_{m,l,t}\mid drift]
$$

含义：某个模块输出在成功和漂移之间的差异方向。

$$
SourceAlign_{m,l,t}=cos(u_{m,l,t}^{S-D},v_{l,t}^{S-D})
$$

含义：模块输出方向和目标残差方向是否一致。

干预公式：

$$
h'=h+\lambda \hat v
$$

含义：向残差状态中加入目标机制方向。

$$
h'=h-\lambda \hat v
$$

含义：移除或抑制目标机制方向。

MLP 拆解公式：

$$
p=W_{gate}(x)\odot W_{up}(x)
$$

$$
MLP(x)=W_{down}(p)
$$

含义：MLP 写入不是整体黑箱，而可以拆成 gate、up、product、down 四个部分。

阶段结论：  
很多关键写入来自 MLP，而不是单个 attention head。但 attention 仍可能负责路由、定位和条件化。

---

**阶段五：Phase 234-245，Pattern Family Atlas v1**
核心问题：  
前面已经有很多测试，但缺少统一数据格式。Phase 234 之后开始把结果变成可复用图谱。

测试原理：  
每个语言模式都记录固定字段，包括行为基线、内部 Trace、因果证据、读出竞争、rollout 稳定性和失败模式。

核心公式：

$$
LanguageMechanism=\sum_i \alpha_i(x,t)P_i(x,t)
$$

含义：语言机制是多个 Pattern 的加权组合。

$$
P_i=
TriggerTrace_i
\circ GateProductTrace_i
\circ ResidualWriteTrace_i
\circ ReadoutTrace_i
\circ CompetitorTrace_i
\circ RolloutTrace_i
\circ ClosureTrace_i
$$

含义：一个 Pattern 必须有完整链路，而不是只记录激活神经元。

阶段结论：  
这是从“实验记录”转向“图谱工程”的关键阶段。研究开始具备积累性。

---

**阶段六：Phase 246-253，机制方向库与共享子空间**
核心问题：  
单个模式的方向能不能复用？不同语言模式是不是共享某些内部子空间？

测试原理：  
从自然样本中提取 mechanism direction（机制方向），再做正交干预、方向移除、方向增强，看是否能控制输出行为。

核心公式：

$$
v_{mech}=\mu_{positive}-\mu_{negative}
$$

含义：目标机制方向由正样本和负样本的内部状态差得到。

$$
s_{mech}(h)=h \cdot \hat v_{mech}
$$

含义：当前状态在机制方向上的投影强度。

$$
h'=h+\lambda \hat v_{mech}
$$

含义：增强某个机制。

$$
h'=h-(h\cdot \hat v_{mech})\hat v_{mech}
$$

含义：移除某个机制方向的投影。

阶段结论：  
语言模式不是完全独立的。answer、explain、repeat、continue、stop 等模式之间可能共享底层子空间，只是在不同边界条件下被读出为不同模式。

---

**阶段七：Phase 254-263，Done / Stop / Continue 竞争机制**
核心问题：  
模型为什么知道“答案已经完成”？为什么有时继续解释？  
这一阶段把闭合问题进一步拆成：

```text
semantic done 语义完成
template done 模板完成
stop readout 停止读出
continue readout 继续读出
```

测试原理：  
比较 close candidate 和 non-close candidate 的内部轨迹，找 done state（完成状态）方向，再追踪它如何影响 stop/continue 竞争。

核心公式：

$$
v_{done}=\mu_{closed}-\mu_{open}
$$

含义：完成状态方向。

$$
S_{done}(h)=h\cdot \hat v_{done}
$$

含义：当前状态是否进入完成状态。

$$
M_{stop/continue}=R_{stop}(h)-R_{continue}(h)
$$

含义：停止读出相对继续读出的优势。

$$
Closure=SemanticDone \land StopWins \land ContinueSuppressed \land RolloutStable
$$

含义：真正闭合必须同时满足语义完成、停止获胜、继续被抑制、后续生成稳定。

阶段结论：  
“答对”只是闭合的一部分。真正的语言闭合还需要压制 continuation path（继续路径）。

---

**阶段八：Phase 264-278，物理路径图谱 v2**
核心问题：  
前面证明了很多机制存在，但还不够系统。Phase 264 以后开始构建 physical path atlas（物理路径图谱），目标是把每个模式拆成可验证路径。

测试原理：  
对每个语言模式记录：

```text
embedding bias
attention route
MLP writer set
compensation path
readout competition
rollout effect
side effect
cross-model consistency
```

核心公式：

$$
M(h)=R_{continue}(h)-R_{stop}(h)
$$

含义：继续路径相对停止路径的优势。

组件贡献拆解：

$$
\Delta M_{attn}^{(l)}=M(h_l+a_l)-M(h_l)
$$

$$
\Delta M_{mlp}^{(l)}=M(h_l+a_l+m_l)-M(h_l+a_l)
$$

$$
\Delta M_{resid}^{(l)}=M(o_l)-M(h_l+a_l+m_l)
$$

含义：分别计算 attention、MLP、residual 对 continue-stop margin 的贡献。

继续路径总公式：

$$
ContinuePath=
B_{embed}
\oplus AttentionRoute
\oplus MLPWriterSet
\oplus CompensationPath
\oplus ReadoutCompetition
\oplus RolloutEffect
$$

含义：继续生成不是单点机制，而是多组件路径。

Phase 273 之后进入 Atlas v2，开始用工程评分管理图谱进度：

$$
Score(x)=w_1B(x)+w_2R(x)+w_3L(x)+w_4C(x)+w_5I(x)+w_6G(x)+w_7K(x)
$$

含义：对一个样本或模式的图谱价值进行综合评分，包括行为、读出、层路径、组件、干预、泛化和知识路径等证据。

缺口队列公式可以概括为：

$$
Gap(f,m,s)=TargetCoverage(f,m,s)-ObservedCoverage(f,m,s)
$$

$$
Priority=Gap \times EvidenceValue \times CrossModelValue \times RiskControl
$$

含义：优先补最缺、最有价值、跨模型最能验证的部分。

阶段结论：  
Phase 264-278 的重点已经不是提出单个公式，而是把机制研究变成“可预测、可验证、可复用”的图谱系统。

---

**最终判断**
Phase 195-278 的主线非常清晰：

```text
Phase 195-203：从颜色特征走向全链路 Trace
Phase 204-208：发现停止执行是独立机制
Phase 209-218：提出语言模式 Pattern 动力学
Phase 219-233：拆解 StateWrite、MLP 写入、Readout 和竞争源
Phase 234-245：建立 Pattern Family Atlas v1 数据结构
Phase 246-253：提取机制方向库和共享子空间
Phase 254-263：深入 Done / Stop / Continue 闭合机制
Phase 264-278：升级为物理路径图谱 v2 和缺口驱动系统
```

最核心的理论公式不是某一个单独公式，而是这条综合机制链：

$$
LanguageMechanism(x,t)
=
\sum_i \alpha_i(x,t)
[
Trigger_i
\circ Route_i
\circ StateWrite_i
\circ Readout_i
\circ Competition_i
\circ Rollout_i
\circ Closure_i
]
$$

通俗讲：  
语言能力不是某个“语言神经元”产生的，而是多个语言模式在上下文条件下被触发，通过注意力路由和 MLP 写入改变内部状态，再在输出层竞争，最后由停止机制决定是否闭合。Phase 195-278 的价值，就是把这个过程一步步从现象、Trace、因果干预，推进到了图谱化系统。








## 总体判断

Phase 278–427 共包含 **151 条记录**，其中 Phase 293 重复编号；Phase 422、427 属于 Git 大文件处理，不是科学实验。

整个推进逻辑不是围绕一个公式不断拟合，而是不断发现旧观测对象不充分，并逐步升级：

> 闭合分数 → 模式族图谱 → 三位置语义路径 → 分布式载体 → 动态流束 → 精确计算账本 → 多位置事件图 → 条件响应核 → 形成—运输—竞争全局图谱。

截至 Phase 426，最严格结果仍然是：

\[
\text{跨模型完整语言机制}=0/72
\]

\[
\text{单神经元因果闭合}=0/72
\]

但这段研究并非没有进展。最大的成果是建立了较严格的测量系统，并连续排除了大量看似合理、实际上不能闭合的机制对象。

---

## 一、Phase 278–300：图谱扩样与候选闭合否定

**核心问题：** 前期高分候选是否已经实现语义完成、停止竞争和自然生成闭合？

Phase 281 将闭合拆成四项：

\[
\mathrm{Closed}
=
\mathrm{SemanticDone}
\land
\mathrm{StopWins}
\land
\mathrm{ContinueSuppressed}
\land
\mathrm{RolloutStable}
\]

其中：

\[
\mathrm{StopWins}:r_{\mathrm{stop}}>r_{\mathrm{continue}}
\]

\[
\mathrm{ContinueSuppressed}:
M_{\mathrm{continue-stop}}\le -0.5
\]

结果是：9 条高分候选中，6 条语义完成，但停止胜出为 `0/9`，继续压制为 `0/9`，四条件闭合为 `0/9`。

Phase 288–290 扩展到 972 条图谱签名：

- MLP 主导率：`0.943878`
- continue 胜出率：`1.0`
- stop 胜出率：`0`
- 平均 continue-stop margin：`8.155253`
- 36 条闭合候选全部被拒绝
- Phase 298 的 72 次 MLP 干预 winner flip：`0/72`

Phase 300 使用了图谱评分公式：

\[
E_{\mathrm{complete}}
=
\frac{
I_{\mathrm{behavior}}+
I_{\mathrm{readout}}+
I_{\mathrm{component}}+
I_{\mathrm{causal}}
}{4}
\]

\[
C_{\mathrm{path}}
=
0.30E_{\mathrm{complete}}
+0.25D_{\mathrm{MLP}}
+0.15D_{\mathrm{Attention}}
+0.20S_{\mathrm{causal}}
+0.10(1-\mathrm{WinnerFlip})
\]

但这是候选排序公式，不是语言机制公式。该阶段得到的核心结论是：

> 行为正确、组件主导、图谱高分都不等于自然生成闭合。

---

## 二、Phase 301–322：语义复用与三位置物理路径

**核心问题：** 语义信息究竟只在答案位置读出，还是从对象位置经过查询位置传播到输出？

研究对象从单个答案位置升级为：

\[
\mathcal P(x)
=
[T_{\mathrm{source}}(x),T_{\mathrm{query}}(x),T_{\mathrm{last}}(x)]
\]

语义边界定义为：

\[
M_{\mathrm{semantic}}
=
\max z(\mathrm{target\ group})
-
\max z(\mathrm{distractor\ group})
\]

逐层分解 Attention（注意力）和 MLP（多层感知机）的边际写入：

\[
\Delta A_l
=
M(h_l+A_l)-M(h_l)
\]

\[
\Delta F_l
=
M(h_l+A_l+F_l)-M(h_l+A_l)
\]

Phase 305–309 发现 object/query/last 三个位置都存在可重复的语义差分，shared path（共享路径）和 delta path（差异路径）也能在内部轨迹上区分。

但是 Phase 313 的联合干预结果为：

\[
\mathrm{winner\ change}=0/18
\]

Phase 320 对筛选候选做注册复核，三个模型全部是 `0/8`。

所以这个阶段证明的是：

> 语义差异具有内部空间分布，但尚未证明这些差异是答案生成所必需的因果边。

---

## 三、Phase 323–355：九族72机制与分布式载体审计

**核心问题：** 单神经元是不是错误对象？语言机制是否由组件集合、接口分支和动态路径共同承载？

Phase 323 首先发现，所谓“全量单神经元测试”实际上没有完成正式 CUDA 执行，候选还缺失，算法也只是每种颜色选一个最高分神经元。因此不能宣称真实神经元级图谱已经形成。

Phase 326–330 将研究扩展为：

- 9 个语言模式族
- 72 个注册机制
- 15,552 个 prompt-model 案例
- 4,852,224 个组件事件

Phase 330 的结果：

- 跨模型集合读出候选：`5/72`
- 自然身份候选：`14/72`
- 跨模型行为必要性：`0/72`
- 单神经元因果闭合：`0/72`
- 完整自然链：`0/72`

Phase 331 扩大留出、双接口和成员细化后，五条集合候选没有一条同时通过跨模型、跨接口、行为必要性和自然身份门。

Phase 332–334 又检验接口分叉、动态状态块和自然删除：

- 跨模型路径交换：`0/2`
- 局部自然必要性：`0/54`
- 跨模型自然必要性：`0/6`
- 完整门：`0/54`

Phase 337 修复了一个重要协议问题：Qwen3 和 DS7B 经常已经在思考文本中得到正确答案，只是没有在预算内进入最终答案。答案对齐接口后三模型均为 `12/12`。

Phase 338 的粗块因果筛选只留下 GLM4 的早层 source MLP 模型特异候选，跨模型粗块候选仍为 0。

这一阶段的结论是：

> 机制更可能是条件化分布式载体，不是最高值神经元；但集合读出和粗块损伤仍没有形成跨模型自然因果链。

---

## 四、Phase 356–364：从标签找峰值转向盲化物理临摹

**核心问题：** 为什么模型可以正确生成，而候选机制始终无法预测未来或形成闭合？

Phase 356 改为标签盲化：发现阶段不读取任务名称、正确答案和机制标签，只保留层、位置、组件、生成时间等物理坐标。

Phase 357–358 建立了原生精度守恒：

\[
h_{l+1}=h_l+\Delta A_l+\Delta M_l
\]

按模型真实 BF16/FP16 加法顺序重放后，三个模型 `1248` 个层更新全部恢复。此前 DS7B 的重构失败来自提前转成 FP32，并非隐藏未知组件。

Phase 363 检验 20 个时间创新和词元竞争公式，冻结公式为 `0/20`。

Phase 364 给出关键诊断。设真实状态为：

\[
s_{k+1}=F(s_k,x_k)
\]

研究者观察到的是压缩量：

\[
o_k=P(s_k)
\]

如果存在：

\[
P(s_1)=P(s_2)
\]

但：

\[
P(F(s_1,x))\neq P(F(s_2,x))
\]

那么不存在只依赖当前压缩量的闭合公式：

\[
P\circ F=G\circ P
\]

这意味着组件范数、少数方向和几个深度锚点可能把未来不同的状态压成同一个观测值。

这是 Phase 278–427 最重要的理论转折：

> 失败的不一定是模型规则，而可能是研究者选择的状态投影不充分。

---

## 五、Phase 365–378：精确计算账本与终端内容载体

**核心问题：** 如果保存真实 Q/K、注意力来源边和逐神经元写入，能否恢复有效路径？

MLP 的逐神经元写入被定义为：

\[
g=\phi(W_gh),\qquad
u=W_uh,\qquad
p=g\odot u
\]

\[
\Delta h_{\mathrm{MLP}}
=
W_dp
=
\sum_i p_iW_{d,:,i}
\]

单神经元写入为：

\[
w_i=p_iW_{d,:,i}
\]

注意力来源写入为：

\[
e_{t,l,r\leftarrow s}
=
W_O
\left[
\alpha_{t,l,h,r,s}v_{t,l,h,s}
\right]_h
\]

Phase 372 又建立了精确 Q/K 概率重放和守恒树：

\[
\widehat A=\operatorname{Softmax}(QK^\top\alpha+M)
\]

\[
\mathrm{Parent}=\sum_b\mathrm{Child}_b
\]

测量工具通过，但候选路径没有随之闭合：

- Phase 367 发现候选：49
- Phase 368 独立校准通过：4
- 跨模型路径：0
- Phase 374 单路线历史充分状态：失败
- Phase 375 有限子图绝对误差门：`0/1584`

Phase 376 将测量对齐到真正的答案决策时刻，并进行 A→C、C→A 内容转移：

\[
M(x)=z_x(d)-z_x(r)
\]

\[
G=M(\mathrm{patch})-M(\mathrm{baseline})
\]

结果在发现、校准和物理留出中得到较强答案翻转。Phase 378 确认关系绑定在三个模型上存在晚段当前位置残差内容载体。

但它靠近输出端，所以只能说明：

> 晚段残差状态已经携带答案内容，不能说明上游语言规则在哪里形成。

---

## 六、Phase 379–397：从预测关系到计算图因果方向

**核心问题：** 观测到的来源—查询关系是否真的是合法传播边？

Phase 386 使用真实增量向量：

\[
\Delta x=A_x-B_x,\qquad
\Delta y=A_y-B_y
\]

\[
R_{xy}=\cos(\Delta x,\Delta y)
\]

发现 135 个跨模型描述候选，最终 10 个通过预测对照。但 Phase 387 检查计算图方向后，10 条都不能解释为直接因果边。

Phase 388 对来源 K/V 做真实替换：

- 84 个干预场景
- donor 答案切换：`0/96`
- 查询门通过：`0/3`
- margin 门通过：`0/3`
- 行为门通过：`0/3`

Phase 391 找到跨模型局部父节点布局，但 Phase 392 的 139/144 答案切换主要由属性内容整体搬运解释，联合多来源特异性为 `0/3`。

Phase 393–396 因而只保留范围有限的结论：

> 属性位置或字段值位置的上下文状态可以被后续网络使用。

Phase 397 进一步固定词元身份和绝对位置：

- 关系签名观测复现：`27/27`
- 因果关系载体：`0/9`
- 答案切换：`0/144`

也就是说：

\[
\text{关系签名可读}
\not\Rightarrow
\text{存在独立可搬运的关系变量}
\]

---

## 七、Phase 398–402：多位置动态事件图

**核心问题：** 单点状态失败后，多位置、多组件、分阶段事件链能否解释语言运行？

Phase 398 用三因素析因效应：

\[
E_S=
\frac18
\sum_{r,o,q\in\{-1,+1\}}
\left(\prod_{f\in S}f\right)h(r,o,q)
\]

ROQ（关系—顺序—查询）联合轨迹在 9/9 单元中复现，但单查询位置因果为 `0/9`。

Phase 399 定义来源写入、查询整合、终端写入三类事件：

\[
l_{\mathrm{source\to query}}
\leq
l_{\mathrm{query}}
\leq
l_{\mathrm{terminal}}
\]

结果：

- 三类事件存在：`27/27`
- 完整有序链：`3/27`
- 跨模型共同任务：`0/3`

Phase 400 使用部分序事件图，发现图在 `5/6` 单元通过，但答案预测准确率只有 `0.4844–0.5938`，预测门 `0/6`。

Phase 402 再检验多父状态，只有 `8/13728` 局部组合通过，模型级候选 `0/12`。

所以稳定活动区域仍然只是：

> “哪里同时活跃”的图，不是“如何计算答案”的状态转移图。

---

## 八、Phase 403–414：条件响应核、状态商与观察者

**核心问题：** 是否应该放弃寻找唯一内部路径，改为研究“给定状态和条件，模型产生什么响应”？

自回归终端核定义为：

\[
K_\theta(x)(v)
=
P_\theta(v\mid x)
=
\operatorname{softmax}(z_L(x))_v
\]

完整未来概率为：

\[
P_\theta(y_{1:H}\mid x)
=
\prod_{t=1}^{H}
K_\theta(x,y_{<t})(y_t)
\]

Phase 403–407 发现：

- 有限候选正确不等于全词表 top-1 正确
- 首词元正确不等于短序列正确
- 给定答案序列得分高不等于自然生成会选择它
- 表面、接口、历史和生成地平线都会改变响应
- 跨模型稳定条件响应族仍为 `0/3`

Phase 411–412 在人工有限世界中定义状态操作：

\[
T_o:S_f\rightarrow S_f
\]

以及组合闭包：

\[
T_{o_2}(T_{o_1}(s))
=
T_{o_3}(s)
\]

类型化观察者协变可以成立：

\[
(s,q,y)
\xrightarrow{a}
(a(s),T_a(q),R_a(y))
\]

但这只证明外部协议自洽，不证明模型内部存在同样的代数。

Phase 414 定义完整计算状态：

\[
\Sigma_{t,l}
=
(H^{\mathrm{all\ positions}}_{t,l},
KV_{t,l},
\mathrm{position},
\mathrm{history},
\mathrm{mask},
\mathrm{execution\ contract})
\]

完整状态续跑满足：

\[
K^{\mathrm{replay}}_{t,l}
=
\operatorname{softmax}
\left(F_{l\to L}(\Sigma_{t,l})\right)
=
K_{\theta,t}
\]

结果：

- 完整状态续跑失败：`0/60`
- 不完整局部状态续跑失败：`59/60`

这证明完整状态足以恢复输出，但完整状态几乎等于保存整个计算现场，还不是所需的紧凑语言机制。

---

## 九、Phase 416–421：接口、历史与自然来源写入

**核心问题：** 在自然生成中，历史答案如何写入当前查询并参与竞争？

Phase 416–418 建立提示前向、原生生成和接口—历史配对账本，并发现绝对范数严重偏向晚层，因此改用层内标准化和等词元历史差分。

Phase 419 确认：即使完整提示词元数相同，历史答案身份仍会改变内部状态，但跨模型相同区域只有 `3/12`。

Phase 420 定义历史交互项：

\[
C(X)
=
\frac12
\left[
(X_{a,b}-X_{a,a})
+
(X_{b,a}-X_{b,b})
\right]
\]

并记录来源写入：

\[
e_{l,h,q\leftarrow r}
=
W^{(h)}_{O,l}
\left(
\sum_{s\in S_r}
\alpha_{l,h,q,s}V_{l,h,s}
\right)
\]

来源写入复现通过，但未见行为预测失败。

Phase 421 平衡正、负和近零竞争边界后：

- 行为边界通过
- 来源写入复现通过
- 历史/当前来源坐标分离通过
- 校准与行为留出预测同时通过：`0/3`

这说明历史信息确实进入查询端，但现有物理特征仍无法预测它最终如何改变行为。

---

## 十、Phase 423–427：全局工作空间与九族全局物理图谱

Phase 423 审计 J-lens（雅可比透镜）：

- 三模型平均雅可比矩阵都稳定
- Qwen3 留出失败
- GLM4 留出通过
- DS7B 校准通过、留出失败
- 完整工作空间模型：`0/3`

因此“平均雅可比稳定”不等于统一语义工作空间。

Phase 424 回到九族72机制，构造形成—运输—竞争图谱。

基础计算为：

\[
a_l=A_l(h_l;K_l,V_l)
\]

\[
m_l=M_l(h_l+a_l)
\]

\[
h_{l+1}=h_l+a_l+m_l
\]

来源集合到查询位置的合法写入：

\[
w_l(S\rightarrow q)
=
W_{O,l}
\operatorname{concat}_h
\left(
\sum_{p\in S}
\alpha_{l,h,q,p}v_{l,h,p}
\right)
\]

标准化差异：

\[
D(x,y)
=
\frac{\|x-y\|}
{\frac12(\|x\|+\|y\|)+\varepsilon}
\]

形成量：

\[
F_l
=
D(s_l^a,s_l^b)
-
D(c_l^a,c_l^b)
\]

运输量：

\[
T_l
=
D(w_l^a(S\to q),w_l^b(S\to q))
-
D(w_l^a(C\to q),w_l^b(C\to q))
\]

结果：

- 没有跨模型完整三阶段拓扑
- 严格双盲：`0/72`
- 因果中介：`0/72`
- 神经元闭合：`0/72`

Phase 425 进一步进行同词元角色交换。严格形成量为：

\[
F_l^{\mathrm{strict}}
=
\min
\left(
F_l^{\mathrm{func}},
F_l^{\mathrm{dom}}
\right)
\]

12 个模型—机制块：

- 形成失败：`12/12`
- 运输失败：`12/12`
- 预测失败：`12/12`
- 跨模型候选：`0/2`

Phase 426 修复来源位置不匹配，比较角色标签在来源前后时的二阶差分：

\[
\Delta_T\Delta_R X_l
=
\Delta_R X_l^{\mathrm{early}}
-
\Delta_R X_l^{\mathrm{late}}
\]

得到局部正结果：

- Qwen3 翻译块通过形成、运输、竞争和部分顺序
- DS7B 翻译块通过形成与运输
- GLM4 只通过形成
- 跨模型完整候选：0
- 密封物理留出未打开
- 因果、头、通道和神经元扫描均未运行

Phase 422、427 只是处理超过 GitHub 100 MB 的结果文件，不能计入科学进展。

---

## 最终理论总结

Phase 278–427 最值得保留的统一框架是：

\[
\mathcal G_m
=
(V_m,E_m,\rho_m,\omega_m)
\]

其中：

- \(V_m\)：状态、组件、位置和生成事件；
- \(E_m\)：符合 Transformer 计算图方向的合法边；
- \(\rho_m\)：来源、查询、角色、接口、历史和生成时间条件；
- \(\omega_m\)：观察、预测、物理留出、因果和跨模型证据。

当前证据支持：

\[
\text{语言生成}
\approx
\text{条件化动态模式网络}
\]

但还不支持：

\[
\text{语言机制}
=
\text{固定神经元集合}
\]

也不支持：

\[
\text{语言机制}
=
\text{单个静态方向或单条线性路径}
\]

更严格的闭合条件应该是：

\[
\mathrm{Closure}
=
\mathrm{Behavior}
\land
\mathrm{Prediction}
\land
\mathrm{Necessity}
\land
\mathrm{Sufficiency}
\land
\mathrm{Mediation}
\land
\mathrm{Recovery}
\land
\mathrm{LowSideEffect}
\land
\mathrm{CrossModelReplication}
\]

当前真正缺少的，是一种比局部向量更充分、又比完整模型状态更紧凑的 **conditional mesoscopic state（条件化中尺度状态）**。

下一阶段不应继续扩充静态候选，而应在少量三模型都稳定掌握的任务上，冻结原生执行合同，记录完整的“来源形成 → 合法运输 → 查询整合 → MLP 重写 → 候选竞争 → 自然生成”事件核；先通过未见预测和密封物理留出，再进行成组损伤、恢复和中介交换，最后才下钻到真实头、通道和单神经元。







**总判断**

Phase 428–570 的主线不是围绕某个预设机制公式持续拆分，而是不断提高证据门槛：

```text
行为是否真实稳定
→ 测量接口是否可靠
→ 内部状态能否独立预测
→ 干预后候选分数是否变化
→ 自然行为是否变化
→ 是否必要、可恢复、排他、跨模型、密封复现
```

整个过程最大的进步，是逐步把“可读出”“可预测”“可搬运”“局部因果贡献”和“完整机制闭合”严格分开。

**十二个阶段**

1. **Phase 428–438：行为与测量资格重建**  
   用来源、查询二因素差分和完整序列裕量检查双路线机制。30,720 个条件没有跨模型行为门通过；后续修正提示终端、记录位置和表面条件后，候选观察器降为 0/9。说明旧信号主要受接口和位置影响。

2. **Phase 439–447：自然任务与抗捷径审计**  
   Qwen3 最初在知识、推理、语法任务上均为 80/80，但简单的首项、末项规则也能解题，因此撤销物理授权。20,736 个抗字符串捷径变体中，没有模型通过完整门。

3. **Phase 448–476：模板、顺序与标签映射分解**  
   GLM4 在原模板达到 944/960，换独立生成器后降到 669/768。标签映射实验发现模型常使用“默认肯定”策略，而不是先判断真假再执行 A/B 映射。

4. **Phase 477–493：语义、标签与序列化三分账**  
   目标盲重算得到语义正确 4,067/4,608，但严格完整输出只有 6/4,608。Qwen3、GLM4 的原生关系语义分别为 256/256、253/256；晚层观察器在独立集达到 512/512、510/512，但这只是可读方向。

5. **Phase 494–517：撤销抽象真值方向，建立 R/B/J 合同**  
   固定断言、只交换证据连接后，旧晚层方向不再具有跨关系族有效性。进一步拆成关系求值 \(R\)、标签编译 \(B\) 和联合输出 \(J\)：只有 GLM4 的 \(R\) 独立确认；三个模型的 \(B\) 全部失败。预注册物理窗口四联预测仅 71/96，不能用后验高分窗口替换。

6. **Phase 518–525：世界状态与查询求值平台分离**  
   Qwen3、GLM4 的关系首语义事件均在新数据上达到 384/384。Qwen3 的断言实体和断言末端连续平台达到 768/768，128 次完整置换的经验值为 \(1/129\)。但两个模型的世界拓扑平台均为零，因此找到的是“查询结果站”，不是上游世界结构和运输路径。

7. **Phase 526–543：角色极性与实体配对绑定反证**  
   GLM4 可以 100% 区分来源、目标方向，但对不存在的来源—目标组合假阳性约 99.52%。这说明观察器识别的是角色极性，不是“哪个实体和哪个实体相连”。Qwen3 的答案对数几率在旧开放集很强，却在全新词汇上降到世界全对 51.76%，因此不能视为可迁移配对状态。

8. **Phase 544–553：九族统一入口与三因素去混杂**  
   18 个自然入口中，Qwen3 通过 4 个、GLM4 通过 5 个、DS7B 为 0。生成前观察器有 7/9 在独立样本复现，但加入表面形式、语义路线和答案身份控制后，Phase 551 的 1,053 个候选全部撤销，1,824 个坐标中通过数为零。

9. **Phase 554–555：从表示相似转向功能响应等价**  
   研究单位改为：“删除哪里会坏、恢复哪里会好、错误供体为什么无效”。同时停止全网高分扫描，把研究收缩到水果类别的对象、类别、属性和绑定。

10. **Phase 556–557：恢复对象身份运行路径**  
    晚层类别和绑定状态能够控制答案，但注意力或 MLP 单项父贡献均不能独立恢复，只有完整父组有效。路径继续追到 L0 后确认：替换完整对象身份会启动供体对象的知识路径，但这不是颜色专属编码。

11. **Phase 558–566：固定对象身份的颜色绑定**  
    扩大样本后只有 Qwen3 获得内部资格。来源颜色位置 L3、L12、L25 的完整状态能稳定改变答案；单位置注意力、多位置注意力块和来源值贡献边分别为 0/3、0/4、0/4。七角色联合残差块则 6/6 具有强供体充分性，说明状态是多位置分布式的，但仍没有自然必要性和最小计算算子。

12. **Phase 567–570：双关系竞争和局部因果贡献**  
    多关系任务的主要错误是返回同一对象的另一关系值。三个模型都出现晚层答案边界竞争区。移除该处注意力输出中的目标方向分量，会按预期改变目标—错误关系裕量，并超过随机方向和错误层控制；但自然生成行为几乎不变，完整因果门为 0/3。

**核心公式**

最初的二因素路线差分是：

\[
\Delta_SX=X_{1,0}-X_{0,0}
\]

\[
\Delta_QX=X_{0,1}-X_{0,0}
\]

\[
\Delta_{SQ}X=X_{1,1}-X_{1,0}-X_{0,1}+X_{0,0}
\]

它分别测量来源影响、查询影响及二者是否发生非加性交互。

观察器与连续平台为：

\[
z=Ph,\qquad d=\mu^+-\mu^-
\]

\[
\widehat y=
\mathbf 1\!\left[
\left\langle z-\frac{\mu^++\mu^-}{2},d\right\rangle>0
\right]
\]

\[
\Pi=(r,[l_a,l_b]),\qquad l_b-l_a+1\ge4
\]

这里的 \(d\) 是研究者构造的读出方向，连续多层可预测也只构成观察平台，不自动形成层间计算边。

关系表示至少需要：

\[
R_l(a,b\mid W)
=
G_l\!\left(U_l(a),V_l(b),B_l(a,b\mid W)\right)
\]

其中 \(U_l,V_l\) 是来源和目标角色，\(B_l\) 才是具体实体对绑定。Phase 533 只确认了前两项，尚未恢复 \(B_l\)。

Phase 554 的关键改进是功能响应指纹：

\[
\Phi(x)=
\left\{
O(\operatorname{Run}(x,c,a)):
c\in\mathcal C,\ a\in\mathcal A
\right\}
\]

\[
x_1\sim_\Phi x_2
\iff
\Phi(x_1)\approx\Phi(x_2)
\]

两个状态只有在多种未来条件和干预下产生相似后果，才视为功能等价，而不是因为隐藏向量距离接近。

状态供体干预为：

\[
h_l'=h_l^{(R)}+\left(h_l^{(D)}-h_l^{(R)}\right)
\]

它检验供体状态是否足以让接收案例转向供体答案；但完整状态替换成功，只证明有限充分性，不能证明其中存在纯净的颜色、关系或绑定变量。

Phase 569–570 的候选竞争为：

\[
d_x=W_{t(x)}-W_{o(x)}
\]

\[
m_l(x)=
\left\langle
\operatorname{Norm}(h_{l,p_{\mathrm{ans}}}(x)),d_x
\right\rangle
\]

\[
a'=a-\langle a,\hat d\rangle\hat d
\]

最后一个操作删除注意力写入中沿目标—错误答案轴的分量。它证明局部裕量贡献，但行为没有显著变化，因此尚不是行为必要机制。

**最终结论**

截至 Phase 570，最可信的拼图是：

```text
对象身份存在可追踪的残差运输路径；
固定身份颜色任务存在多位置联合残差充分状态；
关系选择最终表现为候选间竞争；
晚层注意力对竞争裕量有真实但不足的局部因果贡献。
```

仍缺少上游关系或属性绑定的自然形成、连续合法计算路径、自然必要性、删除—恢复中介、跨模型功能复制、真实神经元定位、多 token 时间程序和密封确认。因此严格机制闭合仍为：

\[
\boxed{0/72=0\%}
\]

Phase 570 给出的图谱约 35%、科学成熟度约 32%只是项目管理估计，不是统计意义上的机制完成率；72 也只是历史分类分母，不是 72 个已经建立自然合同的机制。

详细复盘已追加到 [AGI_GLM5_MEMO.md](/D:/AI2050/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md:30919) 的 Phase 985。本次仅进行文档证据审计，没有运行新的 CUDA 模型测试。