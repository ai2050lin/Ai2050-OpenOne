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