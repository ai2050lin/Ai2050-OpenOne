不是 301-594 每个 Phase 都在做同一种测试。它是一条逐步收紧的路线：

**先拆语言状态变量 → 再找组件/层/通道 → 再验证能否生成 → 最后进入闭合微世界和候选排序图谱。**

我按阶段压缩如下，参考文件为 [AGI_GLM5_MEMO_20260625.md](D:/AI2050/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO_20260625.md:5)。

**1. Phase 301-312：功能子空间分解**
核心问题：语言 hidden state（隐藏状态）里到底有哪些可分离成分？  
解决对象：I/R/C/O/S，即 identity（词元身份）、role（角色）、construction/frame（构式）、operator（操作符）、scope（作用域）。  
测试原理：构造最小对 prompt，提取差分方向，做正交化、因果 patch、归一化注入，判断方向是否真的改变输出。

核心公式：

$$
h_l(x)
=
I_l(x)
+
\alpha_R R_l
+
\alpha_C C_l
+
\alpha_O O_l
+
\alpha_S S_l
+
U_l
$$

$$
O_{\perp R}
=
O
-
\operatorname{Proj}_R(O)
=
O
-
\frac{\langle O,R\rangle}{\|R\|^2}R
$$

这一阶段最大结论：Qwen3/GLM4 更像正交子空间编码；DS7B 更像共享主轴加差分读取，很多方向在激活空间共线，但输出因果效果不同。

**2. Phase 313-320：读出层与关系方向验证**
核心问题：差分方向为什么有效？是 W_U（输出层读出）直接放大，还是中间层传播放大？  
测试原理：比较 W_U gain（输出层增益）和 Jacobian gain（传播增益），再做关系级 causal patch（因果修补）和随机/模板/否定控制。

核心公式：

$$
G_{W_U}(v)
=
\frac{\|W_U v\|}{\|v\|}
$$

$$
G_J(v)
=
\frac{\|J_{l\to out}v\|}{\|v\|}
$$

$$
\text{amplification}(v)
=
\frac{G_J(v)}{G_{W_U}(v)}
$$

结论：W_U 本身没有选择性差分放大；真正的放大发生在中间层传播过程。

**3. Phase 321-347：对象-属性 Binding（绑定）与 MLP 机制**
核心问题：模型如何把 object（对象）和 attribute/value（属性/值）绑定起来？  
测试原理：slot（槽位）验证、对象-属性替换、层归因、Attention vs MLP 分解、gate/up 交互分解、W_down 通道结构分析。

层归因公式：

$$
h_N
=
h_0
+
\sum_l
\left(
a_l + m_l
\right)
$$

$$
\Delta_{\text{binding},l}
=
\left(
W_U[y]-W_U[c]
\right)^\top
\left(
h_{l+1}-h_l
\right)
$$

MLP 交互公式：

$$
\operatorname{MLP}(h)
=
W_{\text{down}}
\left(
\operatorname{SiLU}(W_g h)
\odot
W_u h
\right)
$$

$$
\text{interaction}
=
CC - CR - RC + RR
$$

结论：binding 主要不是 embedding 预编码，而是 transformer 层计算出来的；MLP 的 gate × up 交互贡献很大，W_down 本身呈正负通道近似 50/50 对称，微偏置来自激活差异。

**4. Phase 348-386：类别因子、RMSNorm 与真实因果方向**
核心问题：category（类别）到底是不是独立方向？还是 object identity（对象身份）的一部分？  
测试原理：PCA、SVD、ANOVA（方差分解）、RMSNorm Jacobian、category centroid（类别质心）因果注入。

ANOVA 分解：

$$
\Delta h
=
\mu
+
I_{\text{object}}
+
A_{\text{category}}
+
\epsilon
$$

方差占比：

$$
R_A^2
=
\frac{\|A_{\text{category}}\|^2}
{\|\Delta h-\mu\|^2}
$$

因果注入：

$$
h' = h + \beta A_{\text{category}}
$$

$$
\Delta D
=
D(h') - D(h)
$$

结论：纯 category 方差很小，但 category centroid 的因果效力显著；也就是说，类别信号不一定“大”，但方向非常准。

**5. Phase 387-400：兼容性、值偏好与上下文交互**
核心问题：模型输出正确属性，是因为有 compatibility gradient（兼容性梯度），还是因为 value bias（值偏好）？  
测试原理：correct/incorrect mirror test（正确/错误镜像测试）、neutral prompt（中性提示）、per-object 分解。

核心判据：

$$
\cos
\left(
\Delta h_{\text{correct}},
\Delta h_{\text{incorrect}}
\right)
$$

如果是纯兼容性梯度，应该满足：

$$
\Delta T_{\text{correct}}>0,\quad
\Delta C_{\text{correct}}<0
$$

$$
\Delta T_{\text{incorrect}}>0,\quad
\Delta C_{\text{incorrect}}<0
$$

结论：兼容性不是静态方向，而是对象编码与当前 value context（值上下文）交互后的结果。

**6. Phase 401-414：连续属性与规则重编码**
核心问题：speed、type 等连续/多属性语义是否也是同一套机制？  
测试原理：多候选分布、动态规则、路径级中介、词频控制、W_U 候选方向结构。

核心公式：

$$
D
=
z_{\text{target}}
-
z_{\text{competitor}}
$$

$$
\Delta D_{\text{path}}
=
D_{\text{patched}}
-
D_{\text{base}}
$$

结论：语义不是一个单独概念向量，而是候选竞争场里的动态重排。

**7. Phase 415-460：自然运输方向、共享/私有通道与读出接口**
核心问题：知识能不能沿自然方向 transport（运输）？对象-属性绑定是否通过 shared/private（共享/私有）通道实现？  
测试原理：虚构对象、规则反转、attention head 消融、对象解锁门控、候选族边际动力学、多跳路径验证。

核心公式：

$$
d_{\text{transport}}
=
h_{\text{target context}}
-
h_{\text{source context}}
$$

$$
h' = h + \alpha d_{\text{transport}}
$$

$$
\Delta \ell_y
=
\ell_y(h')-\ell_y(h)
$$

结论：读出接口比单个语义方向更重要；候选族整体会重新分布，不能只看目标 token。

**8. Phase 461-499：参数级语义码、DCF 与 RMSNorm/Gain 闭环**
核心问题：语义码是否能追到参数结构、神经元写入器、RMSNorm gain（增益门）？  
测试原理：W_down 行结构、neuron writer（神经元写入者）、跨语言语义/语言码正交分解、DCF、刹车-释放机制、final RMSNorm/gain 分解。

核心公式：

$$
q_y
=
g \odot W_U[y]
$$

$$
D
=
\langle h, q_y-q_c\rangle
$$

$$
\Delta D
=
\langle \Delta h, q_y-q_c\rangle
$$

结论：语义显化不是 hidden state 单独决定，而是 hidden state 与 RMSNorm gain、读出方向共同决定。

**9. Phase 500-523：Gain-Support Alignment 与正交语义场**
核心问题：GLM5 路线的 gain gate（增益门）和 GPT5 路线的 support path（支持路径）能否统一？  
测试原理：比较类别语义方向 \(v_c\) 与普通读出方向 \(w_D\)、gain 加权读出方向 \(g\odot w_D\) 的对齐；再分解 parallel/perpendicular（平行/正交）成分。

核心公式：

$$
v_c
=
h_{\text{rich}}
-
h_{\text{neutral}}
$$

$$
q_c
=
g \odot
\left(
W_U[y]-W_U[c]
\right)
$$

$$
v_c
=
v_{\parallel}
+
v_{\perp}
$$

$$
v_{\parallel}
=
\operatorname{Proj}_{q_c}(v_c)
$$

结论：可读语义只占高维语义场的极低维投影；大量语义能量在正交空间，但不是噪声。

**10. Phase 524-550：语义选择性、接口矩阵与生成闭合**
核心问题：rank improvement（排名提升）能不能变成真实 generation hit（生成命中）？  
测试原理：category interface response matrix（类别接口响应矩阵）、top-k competition trajectory（前 K 竞争轨迹）、generation closure audit（生成闭合审计）、label gate vs paraphrase gate（标签门/改写门）分离。

核心公式：

$$
\text{hit}
=
\mathbf{1}
[
\operatorname{Generate}(h') \in Y_{\text{target}}
]
$$

$$
\Delta \text{hit}
=
\text{hit}_{\text{patched}}
-
\text{hit}_{\text{base}}
$$

结论：hidden geometry 可以移动 margin，readout competition 可以移动 rank，但真正是否生成目标，还取决于 generation policy gate（生成策略门）。

**11. Phase 551-574：路径恢复、供体特异性、token fork 与格式/回声瓶颈**
核心问题：为什么语义方向明明有效，生成却跑到 object echo（对象回声）、format token（格式词）或 generic output（泛化输出）？  
测试原理：route restore（路径恢复）、wrong donor control（错误供体控制）、prototype injection（原型注入）、prefix fork（前缀分叉）、step0 logit field（第 0 步词表场）、format gate/echo path 审计。

核心公式：

$$
B_{\text{echo}}
=
z_{\text{object}}
-
z_{\text{target}}
$$

$$
B_{\text{format}}
=
z_{\text{format}}
-
z_{\text{target}}
$$

$$
h' = h + \alpha d_{\text{semantic}} - \lambda d_{\text{echo}}
$$

结论：压低 object echo 后，概率质量经常流向 other/generic，而不是 clean synonym；说明路径瓶颈不是单纯读出问题。

**12. Phase 575-594：闭合语义微世界与候选排序图谱**
核心问题：在可控 micro-world（微世界）里，模型能否完成对象-类别、对象-关系-值检索，并进一步组合推理？  
测试原理：人工对象/关系/值表、全字符串 logprob、注意力边消融、ORV 检索、两跳组合、candidate ranking audit（候选排序审计）、atlas-guided patch（图谱引导修补）、conditional transformation atlas（条件化状态变换图谱）。

全字符串评分：

$$
\log P(a_{1:m}\mid x)
=
\sum_{t=1}^{m}
\log P(a_t\mid x,a_{<t})
$$

候选排序分解：

$$
D_{i,j}
=
score_{\text{after}}(i,j)
-
score_{\text{base}}(i,j)
$$

$$
common_i
=
\frac{1}{|V|}
\sum_j D_{i,j}
$$

$$
specific_{i,j}
=
D_{i,j}
-
common_i
$$

$$
margin\_gain
=
D_{i,\text{correct}}
-
D_{i,\text{old-top-wrong}}
$$

Phase 594 的最终收紧公式：

$$
\Delta h_{\text{update}}
=
h^{repair}_{l+1,pos}
-
h^{base}_{l+1,pos}
-
\left(
h^{repair}_{l,pos}
-
h^{base}_{l,pos}
\right)
$$

$$
specific\_margin
=
specific_{\text{correct}}
-
specific_{\text{old-top-wrong}}
$$

结论：真正缺口不是候选集合没被激活，而是 correct-specific ranking（正确候选特异排序）不足。Phase 593 证明 projection node（投影节点）不等于 causal node（因果节点）；Phase 594 把问题升级为条件化状态变换，而不是静态向量 patch。

**总路线一句话**
Phase 301-594 的推进逻辑是：

```text
功能方向分解
→ 组件/层/通道定位
→ 类别/属性/关系绑定
→ RMSNorm/gain 读出接口
→ 高维语义场
→ 生成策略门
→ 路径瓶颈
→ 闭合微世界
→ 候选特异排序与条件化状态变换
```

最重要的理论变化是：  
早期以为“找到一个语义方向就能解释语言编码”，后期逐步证明不够。更接近真实机制的是：

$$
\text{Language Mechanism}
=
\text{Semantic Field}
+
\text{Readout Interface}
+
\text{Policy Gate}
+
\text{Candidate Ranking}
+
\text{Conditional State Transformation}
$$

也就是说，语言编码不是单个概念向量，而是一套高维状态场、读出门、生成门、候选竞争和条件化更新共同闭合的系统。





下面按“推进逻辑”来讲。Phase 600 到 Phase 700 的主线不是零散测试，而是在追问同一个问题：

模型为什么有时已经“知道答案”，却不能稳定输出正确答案？

它一路从 final readout（最终读出）往前追，最后追到 answer-last top-head source contribution（答案末位稀疏注意力头的源 token 贡献）。参考位置主要在 [AGI_GLM5_MEMO.md](D:/AI2050/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md:2198) 到 [Phase 700](D:/AI2050/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md:43245)。

**总路线**

可以概括成这条链：

```text
规则/样例信息
-> value token selection（值 token 选择）
-> Q/K routing（查询-键路由）
-> result carrier state（结果承载状态）
-> format/protocol gate（格式/协议门）
-> intent gate（任务意图门）
-> final readout competition（最终读出竞争）
-> continuation controller（续写控制器）
-> answer-last source contribution（答案末位源 token 贡献）
```

也就是说，Phase 600-700 的推进是：先看“最后一层为什么没接受正确答案”，再发现“答案不是一个单点 logit，而是一条生成轨迹”，然后把轨迹拆成 semantic value（语义值）、format protocol（格式协议）、task intent（任务意图）、readout competitor（读出竞争者）、continuation（续写）几个机制，最后在 Phase 699-700 追到具体 source token contribution（源 token 贡献）的可组合因果结构。

**阶段一：Phase 600-605，最终层接受规则和多 token 生成**

核心问题：  
为什么正确答案信息进入模型后，final layer（最终层）不一定输出正确 token？

测试原理：  
比较 base（原始失败）、natural correct（自然正确）、natural wrong（自然错误）、artificial repair（人工修复）等条件，在 final layer input/output、final norm（最终归一化）、LM head（语言模型头）处看正确 token 的 margin（优势差）。

早期假设是：

```text
correct candidate vector
-> final residual
-> final norm
-> lm_head
-> correct token wins
```

但 Phase 604-605 发现这太简单。真实生成不是单 token，而是：

```text
prefix_format_step（前缀/格式步）
-> value_digit_decision_step（值数字决策步）
-> weak_tail_confirmation_step（尾部确认步）
```

对应公式可以写成：

```text
S_c = R_c(N(h_t))
```

其中：

- `h_t` 是当前位置 residual hidden state（残差隐藏状态）
- `N` 是 final norm（最终归一化）
- `R_c` 是 candidate token readout（候选 token 读出）
- `S_c` 是候选 token 分数

这个阶段的结论是：  
只看最后一个 token 的 logit 不够。正确答案必须通过一整条 generation trajectory（生成轨迹）闭合。

**阶段二：Phase 606-620，Digit1 选择、Q/K/V/O 分解和注意力路由**

核心问题：  
第一个真正区分答案的 digit token（数字 token）是怎么被选中的？

测试原理：  
从 digit1 的 final readout 往前追，拆 attention（注意力）内部结构：

```text
Q（Query，查询）
K（Key，键）
V（Value，值）
O projection（输出投影）
attention pattern（注意力模式）
```

关键公式：

```text
z_h = sum_s alpha_{h,s} V_{h,s}
```

中文解释：

- `z_h`：第 `h` 个 attention head（注意力头）的输出
- `alpha_{h,s}`：这个头对 source token `s` 的注意力权重
- `V_{h,s}`：source token `s` 在这个头里的 value vector（值向量）

Phase 608-613 证明：  
不是单纯改 `V` 就行，核心是 answer-position Q state（答案位置查询状态）改变了路由，让注意力头去读正确 value token。

更精确的链条是：

```text
answer-position residual state
-> Q projection
-> attention routing pattern
-> correct value token selected
-> head mixture
-> digit1 margin
```

所以这里的核心公式从“读出公式”推进到“注意力选择公式”：

```text
M_t^l = concat(z^l_{1,t}, ..., z^l_{H,t})
```

`M_t^l` 是第 `l` 层、位置 `t` 上所有 head output（头输出）的拼接。Phase 610 的 cumulative mixture（累积混合）测试说明：少数 top heads（关键头）比全头平均更重要。

**阶段三：Phase 621-627，selection state 和 result state 分离**

核心问题：  
模型内部到底是“选择了正确值”，还是“把正确值写成可读出的结果状态”？

测试原理：  
把 residual delta（残差差分）分成两类：

1. selection state（选择状态）：控制 Q 怎么选 source token  
2. result state（结果状态）：把被选中的值变成最终可读出的 residual direction（残差方向）

关键公式：

```text
delta_h = h_repair - h_base
```

然后把它投影到 Q 相关方向：

```text
delta_aligned = Proj_{u_Q}(delta_h)

delta_orthogonal = delta_h - delta_aligned
```

中文解释：

- `delta_h`：修复状态和失败状态的隐藏层差值
- `u_Q`：和 Q state（查询状态）相关的方向
- `delta_aligned`：推动正确选择的部分
- `delta_orthogonal`：不直接负责选择、但可能负责结果承载的部分

结论：  
模型里有两个不同机制：

```text
selection state -> 让模型看向正确 value token
result state -> 把正确 value 写进后续 residual/readout
```

Phase 626-627 进一步说明：  
即使 discriminative token（区分性 token）对了，自然生成的完整字符串也可能不闭合，因为 prefix/format（前缀/格式）还没解决。

**阶段四：Phase 628-645，Prefix/Format Gate 和 Protocol State**

核心问题：  
为什么语义值已经对了，模型仍然输出换行、解释、空格或错误格式？

测试原理：  
把 semantic value patch（语义值补丁）和 prefix/format patch（前缀/格式补丁）分开测试。

关键结论来自 Phase 628：

```text
format/prefix gate + semantic value gate -> natural generation closure
```

意思是：

- 只修 semantic value（语义值）不够
- 只强迫 prefix token（前缀 token）也不够
- 两者一起才接近闭合

Phase 639-642 发现 separator boundary（分隔符边界）非常关键，例如 `"\nAnswer:"` 和 `" Answer:"` 会触发不同 protocol state（协议状态）。

机制公式可以写成：

```text
H_t = F_theta(Tokens_{<=t}, C_semantic, C_protocol)
```

其中：

- `H_t`：当前位置隐藏状态
- `Tokens_{<=t}`：当前位置之前的上下文 token
- `C_semantic`：语义值条件
- `C_protocol`：格式/协议条件

Protocol state（协议状态）可以再写成：

```text
R_l = R_l^semantic + R_l^protocol + R_l^syntax + R_l^noise
```

中文解释：

- `R_l^semantic`：语义答案部分
- `R_l^protocol`：短答、换行、Answer 标签等协议部分
- `R_l^syntax`：语法/格式部分
- `R_l^noise`：其它残余因素

这个阶段的结论：  
生成失败不是“不知道答案”，而是 semantic route（语义路线）和 protocol route（协议路线）没有同时对齐。

**阶段五：Phase 646-653，协议图谱和任务意图门**

核心问题：  
protocol field（协议场）虽然能让模型短答，但会不会把本来需要解释、判断、造句的任务也错误压成短答？

测试原理：  
做 global atlas（全局图谱），把不同任务分为：

```text
short value answer
reasoning answer
yes/no answer
sentence answer
non-value answer
```

然后测试 protocol field 在不同任务上是否有 side effect（副作用）。

Phase 650 的机制公式是：

```text
value_short_answer_protocol
= field_strength
  * template_compatibility
  * task_intent_gate
  * model_polarity
```

中文解释：

- `field_strength`：Answer 标签、separator、relation tail 等协议信号强度
- `template_compatibility`：当前模板是否适合短答
- `task_intent_gate`：任务到底是不是要短答
- `model_polarity`：不同模型对这个机制的偏好

这个阶段得到一个重要结论：  
不能只增强短答协议。必须有 task intent gate（任务意图门），否则会把需要解释的任务错误压成 value-only output（只输出值）。

**阶段六：Phase 654-668，最终策略门、投影屏障和续写控制**

核心问题：  
即使第一 token 对了，为什么完整答案仍然不对？

测试原理：  
把 first-token readout（首 token 读出）和 continuation transition（后续 token 转换）拆开。

Phase 662 的 readout 公式很关键：

```text
logit_i = W_i · Norm(h) + b_i
```

进一步拆成：

```text
logit_i = ||W_i|| ||Norm(h)|| cos(Norm(h), W_i) + b_i
```

中文解释：

- `W_i`：第 `i` 个 token 的 LM head 向量
- `Norm(h)`：归一化后的隐藏状态
- `cos(...)`：方向相似度
- `b_i`：bias（偏置）

这里发现：  
有时 correct token（正确 token）的方向更对，但 competitor token（竞争 token）因为 norm/projection advantage（范数/投影优势）赢了。

Phase 664 再引入 multi-competitor margin（多竞争者优势）：

```text
multi_margin
= logit(correct_prefix)
  - max(logit(space), logit(newline), logit(word), logit(explanation))
```

结论：  
输出失败不只是“正确 vs 错误”二选一，而是正确 token 要同时打败 space（空格）、newline（换行）、word（普通词）、explanation（解释开头）等多个竞争路线。

**阶段七：Phase 669-673，机制图谱、反事实控制和自然轨迹验证**

核心问题：  
前面拆出的机制是不是只在个别样例成立，还是可以变成通用机制图？

Phase 669 把整条语言输出机制写成：

```text
language output
= F(S_semantic, G_intent, P_protocol, R_readout, C_continuation)
```

中文解释：

- `S_semantic`：语义值支持
- `G_intent`：任务意图门
- `P_protocol`：格式协议场
- `R_readout`：最终读出竞争
- `C_continuation`：续写控制器

展开成路线：

```text
S_value
-> G_task
-> P_format
-> R_multi-competitor
-> C_format
-> C_value-token
-> W_writer/topology
```

Phase 672-673 做 natural trajectory audit（自然轨迹审计）和 failure taxonomy（失败分类），把失败分成：

```text
readout_competitor_failure
protocol_route_failure
value_binding_failure
format_surface_failure
continuation_transition_failure
```

这个阶段的意义是：  
把局部因果测试整理成可复用的 failure diagnosis system（失败诊断系统）。

**阶段八：Phase 674-684，DS7B 的读出竞争和 prose bias**

核心问题：  
DS7B 为什么特别容易输出解释性 prose（散文/解释路线），而不是短值答案？

测试原理：  
构造 value direction（值方向）和 prose direction（解释方向），测试 remove prose（移除解释）和 add value（增强值）哪个更有效。

关键方向：

```text
d = mean(W_U[value]) - mean(W_U[prose])
```

中文解释：

- `W_U[value]`：value token 的 unembedding/readout 向量
- `W_U[prose]`：解释性 token 的 unembedding/readout 向量
- `d`：value-minus-prose direction（值减解释方向）

隐藏状态干预：

```text
h' = h + rho * ||h|| * d / ||d||
```

结论：  
对 DS7B 来说，失败主要不是“prose 太强”，而是“value readout 不够强”。Phase 684 里 add_value（增强值）比 remove_prose（移除解释）有效得多。

**阶段九：Phase 685-692，值支持状态和残差轨迹边界**

核心问题：  
value-minus-prose direction（值减解释方向）是在哪里被写入 residual stream（残差流）的？

测试原理：  
比较 short_only fail（短答失败）和 terse_no_explain success（简短无解释成功），看哪些层的 residual delta 可以修复失败。

Phase 689-690 的路线：

```text
instruction wording
-> early/mid residual route bifurcation
-> L13-L18 visible residual trajectory bifurcation
-> L18-L25 carry
-> L26 layer_input value-support state
-> L26/L27 residual propagation
-> final readout
```

Phase 691 把 layer delta（层差分）拆成：

```text
delta_layer = terse_layer_out - short_layer_out

delta_attn = terse_attn_out - short_attn_out

delta_mlp = terse_mlp_out - short_mlp_out

delta_carry_est = delta_layer - delta_attn - delta_mlp
```

中文解释：

- `delta_attn`：注意力模块写入的差异
- `delta_mlp`：MLP 模块写入的差异
- `delta_carry_est`：不是当前层直接写入，而是从前层 residual carry（残差携带）来的部分

结论：  
单个 attention 或 MLP 不是全部原因；真正机制是 multi-layer residual trajectory（多层残差轨迹）加上 carry（携带）累积。

**阶段十：Phase 693-700，答案末位注意力头、source token 贡献和组合闭合**

核心问题：  
value-support state 最终是从哪些 source tokens（源 token）来的？能不能用少数 source contribution 复现完整修复？

Phase 698-699 追到具体路径：

```text
record-line target_value
-> selected answer_last heads
-> top head-slot mixture
-> o_proj
-> L23-L27 residual/carry
-> final readout
```

注意力源贡献的核心公式是：

```text
source_contribution
= attention_weight(answer_last -> source_tokens)
  * source_value_vectors
  * selected_head_slots
```

也就是：

```text
C_source = sum_{h in H} sum_{s in S}
alpha_{h,s} V_{h,s} O_h
```

中文解释：

- `H`：选中的 top heads（关键注意力头）
- `S`：选中的 source tokens（源 token 集合）
- `alpha_{h,s}`：答案末位对源 token 的注意力权重
- `V_{h,s}`：源 token 在 head 里的值向量
- `O_h`：该 head 经过 output projection（输出投影）的贡献

Phase 699 的干预方式：

```text
restore:
short_only + (terse source contribution - short source contribution)

degradation:
terse_no_explain + (short source contribution - terse source contribution)

erase:
terse_no_explain - terse source contribution
```

意思是：

- restore（修复）：把成功样例里的源贡献差分加到失败样例
- degradation（降级）：把失败样例的源贡献写回成功样例
- erase（擦除）：直接从成功样例中移除该源贡献

Phase 700 进一步做 composition and scaling（组合和缩放）：

```text
C_combo = C_target_value + C_answer_line + C_self_last
```

修复公式：

```text
short_only
+ [terse combo contribution - short combo contribution]
```

缩放公式：

```text
short_only
+ alpha * [terse target_value contribution - short target_value contribution]
```

Phase 700 的理论近似：

```text
R_answer^{L23:L27}
= R_base
  + A_top(C_value + C_answer_line + C_self_last)
  + epsilon
```

中文解释：

- `R_answer^{L23:L27}`：L23 到 L27 的答案位置残差状态
- `R_base`：基础残差状态
- `A_top`：关键 attention heads 的变换
- `C_value`：目标值 token 的贡献
- `C_answer_line`：答案行 token 的贡献
- `C_self_last`：答案末位自身 token 的贡献
- `epsilon`：剩余未解释因素，比如 layernorm、非线性、其它 token 交互

Phase 700 的结果很关键：  
`target_value + answer_line + self_last` 的组合几乎可以接近甚至达到 full_top32（完整 top32 头）的修复效果。说明答案输出的关键因果贡献已经从“模糊 residual 状态”压缩到了少数 source contribution（源贡献）的组合。

**整体结论**

Phase 600 到 Phase 700 的推进逻辑是：

```text
Phase 600-605:
最终层接受规则不够，要看多 token 生成轨迹。

Phase 606-620:
正确值来自 answer-position Q routing 对 value token 的选择。

Phase 621-627:
选择正确值和把正确值写成可读出结果，是两个不同状态。

Phase 628-645:
语义值正确还不够，必须同时闭合 prefix/format/protocol state。

Phase 646-653:
协议场有副作用，因此需要 task intent gate 控制什么时候短答。

Phase 654-668:
首 token 读出还不够，要处理多竞争者和续写控制。

Phase 669-673:
把机制整理成全局图谱和失败分类系统。

Phase 674-684:
DS7B 的主要失败是 value readout 不足，而不仅是 prose route 太强。

Phase 685-692:
value-support state 是一条跨层 residual trajectory，不是单层单模块。

Phase 693-700:
最终追到 answer-last top-head source contribution，并证明少数源 token 贡献可以组合近似闭合。
```

最通俗地说：  
这 100 个 Phase 是从“模型最后为什么没说对”一路反推到“哪个位置、哪个注意力头、从哪个源 token 读了多少信息”。Phase 700 的意义就是：它不再只是说“某层 residual 有用”，而是能说“target_value、answer_line、self_last 这些源贡献组合起来，构成了答案末位读出的主要因果材料”。




下面这段从 [Phase 700](D:/AI2050/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md:43245) 到 [Phase 755](D:/AI2050/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md:56981)，整体不是散乱测试，而是一条很清楚的推进线：

> 从“某些 head/channel 能提高答案”  
> 推进到“source(源) → writer(写入器) → rewriter(重写器) → suppressor(抑制器) → readout competition(读出竞争) → cross-domain route invariant(跨域路线不变量)”。

中间最大的认识变化是：

```text
语义答案存在 ≠ 自然生成闭合
target boost(目标增强) ≠ competitor suppression(竞争者抑制)
单 token 竞争 ≠ route-level competition field(路线级竞争场)
局部 fruit/apple 图谱 ≠ 跨语义域机制图谱
```

**阶段一：Phase 700-704，source contribution 到 channel ensemble**

核心问题：  
前面已经知道 `target_value source contribution(目标值源贡献)` 有用，但它是不是完整机制？还是需要 `answer_line(答案行)`、`self_last(当前位置)` 一起配合？

Phase 700 的关键公式：

```text
C_combo = C_target_value + C_answer_line + C_self_last
```

也就是把三类 source contribution(源贡献)加起来，看它们是否能近似完整 head patch(注意力头补丁)。

channel(通道)打分公式：

```text
score(l,h,c)
=
[C_combo^terse(l,h,c) - C_combo^short(l,h,c)]
*
< W_O(l,h,c), d_value-prose >
```

解释：

- `l,h,c`：第几层、第几个 head、第几个 channel。
- `C_combo^terse - C_combo^short`：短答格式和解释格式之间的源贡献差异。
- `W_O`：attention output projection(注意力输出投影)。
- `d_value-prose`：value answer(短值答案) 相对 prose answer(解释性回答) 的读出方向。

测试原理：

```text
restore:
short_only + selected_channel(C_terse - C_short)

degradation:
terse_no_explain + selected_channel(C_short - C_terse)
```

Phase 703 做 holdout(留出验证)，用一半样本排序 channel，在另一半样本上测试，避免“同案过拟合”。

Phase 704 做 cross-case donor(跨样本供体)验证，测试 same_value donor(同值供体) 和 unrelated donor(无关供体)。

结论：

```text
DS7B 中存在 source-restricted positive channel ensemble(源限制正向通道集合)。
但它不是纯 semantic value code(语义值代码)，而是混合了 route gain(路线增益)、format state(格式状态)、case residual(样本残差)。
```

所以公式从：

```text
source_channel_ensemble ≈ V_identity
```

被修正为：

```text
source_channel_ensemble
=
G_route + V_identity + P_format + E_case
```

---

**阶段二：Phase 705-710，identity、phrase、natural generation 分离**

核心问题：  
cross-case patch 成功时，到底是恢复了 target value(目标值)，还是注入了 donor value(供体值)，还是只是把模型推向 value-answer route(短值答案路线)？

Phase 705 比较：

```text
target_minus_donor = logit(target_value) - logit(donor_value)
target_minus_prose = logit(target_value) - logit(prose_route)
```

Phase 706 修正 first-token identity(首词元身份)混淆：

```text
canonical_first_token(target)
∩
canonical_first_token(donor)
=
empty
```

Phase 707 进入 full-value phrase likelihood(完整值短语似然)：

```text
L(y|x)
=
(1/m) * Σ_i log P(y_i | x, y_<i)
```

比较：

```text
L(target_value_phrase)
L(donor_value_phrase)
L(target_prose_phrase)
```

Phase 709 进入 natural generation(自然生成)：patch 后直接 greedy generate(贪心生成)。

Phase 710 比较注入位置：

```text
pre_o_input
post_o_output
post_layer_output
```

结论：

```text
source_channel_ensemble 主要改变 value/prose route competition(短值/解释路线竞争)。
donor identity(供体身份)并没有稳定迁移。
target identity(目标身份)更多由 target prompt residual(目标提示残差)锁定。
```

公式进一步收紧为：

```text
source_channel_ensemble
=
G_route
+ P_format
+ E_target_context
+ V_identity_local
+ GenerationGate
```

---

**阶段三：Phase 711-720，从 patch 转向 atlas(图谱)**

核心问题：  
继续堆 patch 收益变小，必须把机制整理成可查询、可反证的 mechanism atlas(机制图谱)。

Phase 711 建立 atlas unit(图谱单元)：

```text
unit =
model + layer + head/channel + source_group + target_position + role_scores + status
```

Phase 712 做 QK/V factor decomposition(QK/V 因子分解)。

attention contribution(注意力贡献)：

```text
C = Σ_s a_s v_s
```

其中：

- `a_s`：attention weight(注意力权重)，回答位置看向源 token 的强度。
- `v_s`：value vector(值向量)，从源 token 搬来的内容。

两种状态差异：

```text
ΔC = C_terse - C_short
```

拆成三项：

```text
ΔC
=
Σ_s (a_terse_s - a_short_s) v_short_s
+
Σ_s a_short_s (v_terse_s - v_short_s)
+
Σ_s (a_terse_s - a_short_s)(v_terse_s - v_short_s)
```

对应：

```text
ΔC_QK       = 寻址变化，看哪里变了
ΔC_V        = 内容变化，搬什么变了
ΔC_QKxV     = 寻址和内容的耦合变化
```

结论：

```text
DS7B 更偏 QK addressing(查询/键寻址)。
qwen3 更 mixed coupled(混合耦合)。
GLM4 不够稳定。
```

这一步把机制从“某个 head 有用”推进为：

```text
这个 head/channel 的作用主要来自 QK、V，还是 QK×V 耦合？
```

---

**阶段四：Phase 721-727，functional atlas 到 apple-fruit micro-atlas**

核心问题：  
图谱不能只测一个 object-relation-value 任务，需要扩展功能族，但扩展后又发现太散，于是收缩到 apple-fruit-attribute(苹果-水果-属性)微世界。

Phase 721 计算 source focus score(源聚焦分数)：

```text
source_focus_score
=
target_value_mass
+ 0.5*object_name_mass
+ 0.5*relation_name_mass
+ 0.5*target_language_mass
+ 0.5*grammar_marker_mass
- 0.5*instruction_line_mass
- 0.25*answer_line_mass
```

Phase 722 做 causal ablation(因果消融)：

```text
target_logprob_delta
target_rank_delta
margin_delta
top1_drop_rate
```

关键认识：

```text
high attention(高注意力) ≠ causal necessity(因果必要性)
likelihood support(似然支撑) ≠ generation closure(生成闭合)
```

Phase 725-727 发现：

```text
channel cluster(通道簇) 能影响 category likelihood(类别似然)，
但不能稳定改变 natural generation(自然生成)。
full head(完整头) 对生成更敏感。
```

---

**阶段五：Phase 728-732，full-path functional atlas**

核心问题：  
差分 patch 只能看到某一条边，不能看到完整功能骨架。所以要记录 absolute trajectory(绝对轨迹) 和 differential trajectory(差分轨迹)。

Phase 729 的传播公式：

```text
δh_l = h_l(intervention) - h_l(baseline)
```

放大率：

```text
amplification_vs_source
=
||δh_l|| / ||δh_source||
```

Phase 731 的 factor effect(因素效应)：

```text
effect_norm
=
|| mean(h | factor = level) - mean(h) ||
```

Phase 732 做 prompt transfer(提示类型迁移)：

```text
h_v^commonsense ← h_v^explicit
```

结论：

```text
prompt_type / knowledge_source skeleton(提示类型/知识来源骨架)
是 apple-fruit-attribute 微世界中非常强的全路径因素。
```

也就是说，模型不是先处理 apple/fruit，再处理格式；而是先形成：

```text
explicit_profile / conflict_profile / commonsense
```

这种任务骨架，然后在骨架里绑定 object、relation、value。

---

**阶段六：Phase 733-738，writer → rewriter → readout competition**

核心问题：  
既然 prompt-type skeleton 有效，谁把它写进去？谁重写？为什么最后还不生成正确答案？

Phase 734 定义 skeleton direction(骨架方向)：

```text
d = h_target(explicit) - h_target(commonsense)
```

消融某组件后的 skeleton loss(骨架损失)：

```text
explicit_skeleton_loss
=
- < Δh, d / ||d|| >
```

Phase 735 做 source-restricted writer validation(源限制写入器验证)：

```text
C_G(l,h)
=
Σ_{t∈G} α_{l,h}(a,t) V_{l,h}(t)
```

只擦除某个 head 从某个 source group 带来的贡献。

Phase 736 做 source replacement(源替换)：

```text
head_input'
=
head_input
- C_G(recipient)
+ C_G(donor)
```

Phase 737 加入 MLP rewriter(MLP 重写器)：

```text
MLP_l[start:end]' = MLP_l[start:end]^donor
```

读出边际：

```text
donor_vs_recipient_margin
=
logit(donor_answer) - logit(recipient_answer)
```

结论：

```text
DS7B L22H24 是很强的 source writer(源写入器)候选。
L27/L22 MLP group 是 rewriter/amplifier(重写器/放大器)候选。
但 writer + rewriter 只能提高 margin，不能让 donor answer 自然闭合。
```

Phase 738 证明瓶颈已经移动到：

```text
token0 competition(首词元竞争)
+
token1 continuation gate(续写门)
```

---

**阶段七：Phase 739-742，readout threshold 和 natural threshold components**

核心问题：  
如果路径已经把答案推近输出端，那离真正 top1 还差多少？

读出公式：

```text
logit(y) = W_U(y)^T h_final
```

构造 donor-vs-top 方向：

```text
d = normalize(W_U(y_donor) - W_U(c_top))
```

人工读出增强：

```text
h_final' = h_final + αd
```

闭合条件：

```text
logit(y_donor) + Δreadout(y_donor) > max_c logit(c)
```

Phase 740 用 threshold fraction(阈值比例)：

```text
fraction = projected_delta / threshold
```

Phase 741 验证自然候选组件是否因果有效：

```text
effect_fraction
=
projection(condition_final - base_final, d) / threshold
```

Phase 742 做 topK 组合：

```text
h_combo
=
h_joint + Σ_{u∈topK} Δu_donor-recipient
```

结论：

```text
人工跨过 readout threshold 后，三模型都能短答闭合。
自然 donor path 里确实有足够答案方向。
但当前已定位的 writer/rewriter patch 只传递了很小一部分。
```

这说明失败不是“答案方向不存在”，而是：

```text
答案方向没有被自然路径充分传到最终读出阈值。
```

---

**阶段八：Phase 743-745，从单竞争者到 route-level competition field**

核心问题：  
就算 donor answer 被增强了，为什么还是不 top1？因为它不是只和一个 token 竞争，而是和多个 route class(路线类)竞争。

Phase 743 抑制当前 top competitor(最高竞争者)：

```text
d_c = normalize(W_U(c) - W_U(y_donor))
```

竞争缺口：

```text
gap_c = logit(c) - logit(y_donor)
```

最小抑制量：

```text
α_c
=
gap_c / dot(W_U(c)-W_U(y_donor), d_c)
```

Phase 745 把闭合条件升级为：

```text
Closure
⇔
logit(y_donor)
-
max_R max_{c∈R} logit(c)
>
0
```

其中：

```text
R ∈ {recipient, format, echo, punctuation, prose, other}
```

所以不是：

```text
donor > current_top
```

而是：

```text
donor > every dominant route
```

更完整的力学公式：

```text
ClosureForce
=
DonorBoost
+ MultiRouteSuppression
- RouteTakeoverRisk
```

结论：

```text
qwen3:
压掉 recipient 后 format 接管。

GLM4:
已经接近闭合，route-level suppression 可到 0.9。

DS7B:
format + echo 是主竞争场，单 top suppression 不够。
```

这一步是非常关键的理论跃迁：  
闭合不是单点 logit 增强，而是 readout field reordering(读出场重排)。

---

**阶段九：Phase 746-747，理论整合：训练、预测充分状态、生成闭合**

这两阶段主要是理论整合，不是新模型测试。

自回归训练公式：

```text
Pθ(x_{t+1}|x_{≤t})
=
softmax(W_U · LN(h_t^L))
```

loss：

```text
L_t = -log Pθ(x_{t+1}|x_{≤t})
```

logit 梯度：

```text
∂L_t / ∂logit(y)
=
Pθ(y|x_{≤t}) - 1[y = x_{t+1}]
```

意思是：

```text
正确 token 被推高；
高概率错误 token 被压低；
这种压力沿 lm_head、residual、MLP、attention、Q/K/V 反传。
```

Phase 747 引入 predictive sufficient state(预测充分状态)：

```text
H_suf(x_{≤t})
=
{
  h :
  D(Pθ(.|h), Pθ(.|x_{≤t})) < ε
}
```

意思是：  
不同 hidden state(隐藏态)只要读出的 next-token distribution(下一词元分布)足够接近，就属于同一个预测充分等价类。

这给前面所有 patch 一个更高层目标：

```text
不是只改变 hidden state，
而是把状态推入目标预测充分等价类，
并最终完成 generation closure。
```

---

**阶段十：Phase 748-752，自然 route suppressor matrix 和自然 writer 稳定性**

核心问题：  
Phase 745 证明了需要 route-level suppression，但那还是人工读出几何干预。自然模型里有没有 suppressor(抑制器)？

Phase 748 定义 route score(路线分数)：

```text
S_R(h)
=
max_{y∈V_R} W_U(y)^T Norm(h)
```

组件对路线的抑制：

```text
Suppress_u(R)
=
S_R(h_base) - S_R(h_do(u))
```

目标增强：

```text
Boost_u
=
logit_target(h_do(u)) - logit_target(h_base)
```

route suppressor matrix(路线抑制矩阵)：

```text
M_{u,R} = Suppress_u(R)
```

Phase 749 下钻到 headset/channelset(头集合/通道集合)。  
Phase 750 做 natural necessity test(自然必要性测试)：

```text
target_logit_drop_after_erase
total_positive_route_release_after_erase
mean_margin_drop_target_vs_routes
top1_loss_rate
```

Phase 751 回到 attention mechanism(注意力机制)：

```text
source contribution
=
Σ_{t∈G} α_{l,h}(a,t) V_{l,h}(t)
```

这一步明确拆成：

```text
Q/K pattern(看哪里)
+
V/O content(搬什么并写入哪里)
```

Phase 752 做跨 relation/object 稳定性验证。

结论：

```text
DS7B L22:H24 和 L22:H1 是稳定 mixed writer/guard(混合写入器/守卫器)候选。
它们既支持 target logit，也会影响 route release。
qwen3/GLM4 没有同等稳定复现。
```

---

**阶段十一：Phase 755，跨语义域路线不变量**

注意：memo 里 Phase 752 后直接跳到 Phase 755，中间没有完整 Phase 753/754 记录。

核心问题：  
前面 apple-fruit 局部图谱成立，但它是不是只适用于水果域？还是跨 fruit、animal、plant、object、tool、abstract 都复用？

Phase 755 测试 domain(语义域)：

```text
fruit, animal, plant, object, tool, abstract
```

关系：

```text
category, color, taste, shape, edible, grows_on_tree
```

route profile divergence(路线轮廓差异)用 JS divergence(JS 散度)：

```text
JS(P || Q)
=
1/2 KL(P || M) + 1/2 KL(Q || M)

M = (P + Q) / 2
```

如果不同 domain 的 route profile JS 很低，说明它们的输出竞争路线结构相似。

source removal 指标：

```text
target_logit_drop
=
logit_base(y_target) - logit_after(y_target)
```

```text
route_release
=
max_route logit_after - max_route logit_base
```

结果：

```text
DS7B L22:H24 records_all:
support rate = 0.862
mean target drop = 0.528
route guard rate = 0.310
top1 loss = 0.121

DS7B L22:H1 records_all:
support rate = 0.810
mean target drop = 0.554
route guard rate = 0.328
top1 loss = 0.121
```

结论：

```text
DS7B L22:H24 / L22:H1 不只是 fruit-domain writer。
它们是 DS7B-local cross-domain writer/guard candidate(DS7B 局部跨域写入器/守卫器候选)。
```

但不能说是跨模型 universal invariant(通用不变量)，因为：

```text
qwen3 弱；
GLM4 弱；
DS7B 强。
```

---

**总收束**

Phase 700-755 的推进逻辑可以浓缩成这条链：

```text
source contribution
→ source-restricted channel ensemble
→ identity/prose/generation gate split
→ QK/V factor atlas
→ functional atlas
→ full-path causal edge
→ writer / rewriter path
→ readout threshold
→ route-level competitor suppression
→ natural suppressor / writer localization
→ cross-domain route invariance
```

最重要的理论结论是：

```text
语言模型不是把一个“语义向量”直接读出来。
它是在条件化残差流中形成 source-conditioned route，
经过 writer、rewriter、suppressor、carrier，
最后在全词表 readout competition 中完成或失败于闭合。
```

而 Phase 755 是这段的阶段性终点：  
它把前面 apple-fruit 局部机制第一次扩展到多个 semantic domain(语义域)，证明 DS7B 至少存在一组跨域复用的 writer/guard 路线候选。





下面按“推进路线”来讲。这里的主线不是单个测试孤立推进，而是从 **局部路线(local route，本地路线)** 一步步推进到 **全词表闭合(full-vocabulary closure，全词表闭合)**，最后发现：真正的闭合不是“把目标 logit 抬高”这么简单，而是要同时定位 **target booster(目标增强器)**、**suppressor(抑制器)**、**identity anchor(身份锚点)** 和 **new blocker(新阻塞项)**。

参考文件：  
[AGI_GLM5_MEMO.md](D:/AI2050/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md:56981)

**总推理路线**

Phase 755 到 Phase 800 的整体逻辑可以压缩成一句话：

> 先证明模型里存在跨任务稳定的“语义路线”，再追问这条路线由哪些组件承载，为什么内部语义对了但输出 token 不闭合，最后把失败原因推进到全词表 blocker field(阻塞场) 和 true suppressor(真实抑制器) 的区分。

也就是说，研究路径大概是：

`跨域路线不变量 → 下游承载/重写 → 语义因果纤维 → 输出闭合失败 → 表面形式/词元身份 → Q/K/V/O 机制拆解 → 多组件闭合 → 全词表阻塞场 → 真实抑制器定位`

---

**阶段一：Phase 755-761，寻找跨域稳定路线**

核心问题：  
模型在不同语义域里，是不是用同一类内部路线处理“对象 → 属性/答案”？

例如 fruit(水果)、animal(动物)、tool(工具) 等不同 domain(语义域)，如果都出现类似的 attention head(注意力头) 或 source token(源词元)贡献，就说明不是偶然 prompt trick(提示技巧)，而可能是模型内部的通用路线。

测试原理：  
对候选 head/source group(注意力头/源组)做 removal(移除) 或 restore(恢复)，观察 target logit(目标 logit) 和 route competitor(路线竞争项) 的变化。

主要公式：

```text
target_logit_drop = z_base(y+) - z_after(y+)
```

意思是：  
原本目标答案 `y+` 的 logit 是 `z_base(y+)`，干预后变成 `z_after(y+)`。如果下降很多，说明被移除的组件原本在支持目标答案。

```text
route_release = max z_after(route) - max z_base(route)
```

意思是：  
某些非目标路线在干预后被“释放”出来。如果 competitor(竞争项)升高，说明原组件可能还承担了抑制错误路线的功能。

这几个 Phase 的推进：

- Phase 755：发现跨域候选 writer(写入器) 和 guard(守卫器)。
- Phase 756：加入 control group(控制组)，验证不是同层随机组件导致。
- Phase 757：测试 downstream carrier(下游承载器)，看后层是否能恢复被破坏的目标。
- Phase 758：发现 L25-L26 更像 late rewrite(晚期重写)，不是简单承载。
- Phase 759：把 target recovery(目标恢复) 和 route suppression(路线抑制) 分开。
- Phase 760：做 route suppression matrix(路线抑制矩阵)，发现抑制不是单点完成。
- Phase 761：拆 source contribution(源贡献)，发现“目标增强”和“路线抑制”混在一起。

这一阶段的结论：  
早期想找一个“统一抑制器”，但结果显示：模型内部至少有两种作用混在一起：

```text
目标增强 target boost ≠ 竞争路线抑制 route suppression
```

---

**阶段二：Phase 762-770，提出 causal fiber(因果纤维)**

核心问题：  
如果不同 prompt(提示)里同一个 object(对象)有相似内部因果模式，这种模式能不能叫一条 fiber(纤维)？

这里的 fiber(纤维)不是物理纤维，而是“同一个语义对象在不同上下文中的一组稳定因果轨迹”。

测试原理：  
把一个样本经过多个干预后得到的 effect vector(效应向量)看成它的 causal fiber profile(因果纤维画像)。

一个样本的 fiber 可以抽象成：

```text
F(x) = [Δz_1(x), Δz_2(x), ..., Δz_n(x)]
```

其中每个 `Δz_i` 是一次组件干预带来的 logit/margin/top-k 变化。

比较两个样本的 fiber 是否相似，用 cosine similarity(余弦相似度)：

```text
cos(F_a, F_b) = (F_a · F_b) / (||F_a|| ||F_b||)
```

如果同一个 object(对象)在不同 context(语境)下 `cos` 很高，说明内部机制稳定。

Phase 762-770 的推进：

- Phase 762：确认 semantic-numeric interface(语义-数值接口)与 causal fiber(因果纤维)。
- Phase 763：做 feature ablation(特征消融)，看 fiber 是由哪些特征撑起来。
- Phase 764：换 record format(记录格式)和 natural context(自然语境)，看 fiber 是否还在。
- Phase 765：进入 commonsense context(常识语境)。
- Phase 766：审计 prediction-sufficient state(预测充分状态)是否可靠。
- Phase 767：分析闭合失败类型。
- Phase 768：构造 semantic clean subset(语义干净子集)。
- Phase 769-770：做 balanced reanalysis(平衡重分析)，避免 domain/relation/context 混淆。

关键闭合公式：

```text
R_atlas(x) = R_output(x) ∧ R_fiber^balanced(x) ∧ R_paired(x)
```

通俗解释：

- `R_output(x)`：输出层面是否闭合。
- `R_fiber^balanced(x)`：内部因果纤维是否稳定。
- `R_paired(x)`：配对语境下是否一致。

这一阶段的结论：  
“语义正确”不等于“输出 token 正确”。模型内部可能已经选对语义，但最终读出时输成了大小写、别名、同义词或错误 token。

---

**阶段三：Phase 771-778，从 fiber 走向 component validation(组件验证)**

核心问题：  
fiber profile(纤维画像)稳定，不代表已经知道是哪一个组件真的在起因果作用。要从“相关画像”推进到“因果组件”。

测试原理：  
对候选 component(组件)做 matched causal intervention(配对因果干预)，看目标答案、margin(间隔)、top1 是否真的变化。

典型 source contribution(源贡献)公式：

```text
C_g(l,h|x) = Σ_{t∈g} α_{l,h}(p,t|x) V_{l,h}(t|x)
```

解释：

- `l`：layer(层)
- `h`：head(注意力头)
- `g`：source token group(源词元组)
- `α`：attention weight(注意力权重)
- `V`：value vector(值向量)
- `p`：answer position(答案位置)

也就是：某个 head 从一组源 token 搬运了多少 value 信息到答案位置。

干预后看：

```text
Δz_y = z_y(x) - z_y(x')
```

```text
ΔMargin = [z_y(x)-z_c(x)] - [z_y(x')-z_c(x')]
```

如果移除组件后目标 logit 或 margin 明显下降，说明组件有因果作用。

Phase 771-778 的推进：

- Phase 771：验证旧候选组件是否真的 causally sufficient(因果充分)。
- Phase 772：重新 discovery scan(发现扫描)，不再只依赖 Phase 755 旧候选。
- Phase 773-775：拆 instruction source(指令来源)、candidate list(候选列表)、free semantic transfer(自由语义迁移)。
- Phase 776：发现 readout bridge competition(读出桥竞争)。
- Phase 777：修正 strict token closure(严格词元闭合)口径。
- Phase 778：测试 surface-form normalization(表面形式归一化)。

关键公式：

```text
C_strict = 1[argmax_{v∈V_all} z_v = y*]
```

严格闭合：全词表 top1 必须正好是目标 token。

```text
C_equiv = 1[top1_class ∈ equivalent_target_family]
```

语义等价闭合：top1 虽然不是完全同一个 token，但属于目标等价类。

```text
G_surface = C_equiv - C_strict
```

如果 `G_surface=1`，说明语义对了，但表面 token 没对。

这一阶段的结论：  
失败不一定是 semantic failure(语义失败)，很多是 surface form failure(表面形式失败) 或 token identity failure(词元身份失败)。

---

**阶段四：Phase 779-785，定位 surface-form route(表面形式路线)**

核心问题：  
既然语义对但 token 不对，那么模型在哪里把“语义答案”变成“具体词元形式”？

测试原理：  
测试候选组件是否负责 lowercase(小写)、punctuation(标点)、token identity(词元身份)、answer-site readout(答案位读出)。

这一阶段关注的是：

```text
semantic latent state
→ relation value pool
→ surface-form family
→ tokenizer identity anchor
→ final token
```

中文解释：  
模型先有一个语义潜状态，然后在候选答案池里选值，再决定表面形式，最后锚定到 tokenizer(分词器)里的某个具体 token。

相关公式可以写成：

```text
ΔC_strict = C_strict(after patch) - C_strict(before patch)
```

如果 patch(补丁干预)后严格 token 闭合从 0 变 1，说明这个组件可能参与 token identity closure(词元身份闭合)。

Phase 779-785 的推进：

- Phase 779：理论收束，提出表面形式组件定位方案。
- Phase 780：找 surface-form candidate component(表面形式候选组件)。
- Phase 781：做 causal patch/ablation(因果补丁/消融)。
- Phase 782：多组件组合 patch。
- Phase 783：跨 token position(词元位置)测试 fiber 边界。
- Phase 784：分析 answer-site route channel budget(答案位路线通道预算)。
- Phase 785：拆 positive/negative subspace(正/负子空间)。

这一阶段的结论：  
输出失败不只是“目标不够高”，而是目标 token 的 identity anchor(身份锚点)不够稳定，读出空间里还有很多竞争项。

---

**阶段五：Phase 786-794，把机制拆成 Q/K/V/O**

核心问题：  
如果要真正解释机制，不能只说某个 head 有用，而要拆成 Q/K/V/O：

- Q(query，查询)：答案位置在问什么？
- K(key，键)：源 token 是否被匹配到？
- V(value，值)：源 token 搬运了什么内容？
- O(output，输出投影)：head 的输出如何写回残差流？

测试原理：  
分别替换或干预 Q/K/V/O，看哪一部分对闭合有贡献。

简化机制公式：

```text
Attn(Q,K,V) = softmax(QK^T / √d) V
```

具体到一个 head：

```text
O_head = Attn(Q_h, K_h, V_h) W_O
```

如果只替换 `V` 有效，说明 value content(值内容)关键。  
如果只替换 `K/Q` 有效，说明 routing/matching(路由匹配)关键。  
如果替换 `O` 有效，说明 output projection(输出投影)关键。

Phase 786-794 的推进：

- Phase 786：审计 head projection(头投影)和 MLP activation channel(MLP 激活通道)。
- Phase 787：验证 signed subspace source unit(带符号子空间源单元)。
- Phase 788：matched source unit causal fiber validation(配对源单元因果纤维验证)。
- Phase 789：复盘 Phase 755 的条件化相对状态分解公式。
- Phase 790：把目标拆成 formula(公式)、atlas(图谱)、closure(闭合)。
- Phase 791：追踪 upstream Q/K/V source-token causal fiber(上游 Q/K/V 源词元因果纤维)。
- Phase 792：审计 formula-atlas isomorphism(公式-图谱同构)。
- Phase 793：Q/K/V/O 独立因果拆解和 closure gate(闭合门)。
- Phase 794：source-to-answer Q/K/V/O replacement closure validation(源到答案替换闭合验证)。

这一阶段的结论：  
机制图谱不能只停留在“哪个 head 重要”，而要能对应到公式变量：

```text
组件节点 component node
↔ Q/K/V/O 变量
↔ 因果干预结果
↔ 输出闭合变化
```

---

**阶段六：Phase 795-800，全词表 blocker field(阻塞场) 与 true suppressor(真实抑制器)**

核心问题：  
为什么多组件 patch 后，margin(间隔)改善了，rank(排名)改善了，但 token closure(词元闭合)还是没有过？

答案是：因为目标不是只和一个 contrast answer(对照答案)竞争，而是和全词表里所有高于它的 token 竞争。

Phase 798 给出 full blocker(全阻塞项)定义：

```text
B_full(x) = {v ∈ V, v ≠ y+ | z_v(x) > z_y+(x)}
```

解释：  
所有 logit 高于目标 token `y+` 的词元，都是 blocker(阻塞项)。

真正闭合要求：

```text
z_y+ + Δz_target + Δz_anchor > z_v - Δz_suppress(v), for all v ∈ B_full
```

也就是：  
目标要升高，身份锚点要稳定，所有 blocker 还要被压下去。

Phase 799 的 closure-fiber score(闭合纤维分数)公式：

```text
S_blocker = (1/|B_rec|) Σ_{v∈B_rec} [z_rec(v) - z_after(v)]
```

这是平均 blocker suppression(阻塞项抑制)。正数表示 blocker 被压低。

```text
S_lift = (1/|B_rec|) Σ_{v∈B_rec}
[(z_after(y+) - z_after(v)) - (z_rec(y+) - z_rec(v))]
```

这是目标相对 blocker 的平均 margin lift(间隔提升)。

```text
R_new = |B_after \ B_rec| / max(|B_after|, 1)
```

这是 new blocker ratio(新阻塞项比例)。如果旧 blocker 被压下去了，但新 blocker 又冒出来，闭合仍然失败。

```text
S_closure-fiber =
αΔz_target + βΔz_anchor + γS_blocker - λR_new
```

中文解释：

- `Δz_target`：目标答案有没有被抬高。
- `Δz_anchor`：目标 token 身份有没有被锚定。
- `S_blocker`：旧 blocker 有没有被压低。
- `R_new`：有没有产生新 blocker，要扣分。

Phase 800 继续拆分 target booster(目标增强器) 和 true suppressor(真实抑制器)：

```text
S_true-suppressor =
max(Δz_target,0) · max(S_blocker,0) · R_resolved · (1-R_new)
```

真实抑制器需要同时满足：目标升高、blocker 下降、阻塞被解决、新 blocker 少。

```text
S_threshold-shift =
max(Δz_target,0) · max(-S_blocker,0) · R_resolved · (1+R_new)
```

如果目标升高了，但 blocker 平均也升高了，这更像 threshold shift(阈值平移)，不是抑制器。

最关键结论：

```text
ΔT ≠ ΔS
```

目标增强不等于阻塞抑制。

```text
ΔBlockerCount < 0 does not imply ΔS > 0
```

blocker 数量减少，也不等于 blocker 场真的被压低；可能只是目标线升高，导致一部分 blocker 低于目标阈值。

这一阶段的结论：  
Phase 799/800 把问题最终说清楚了：闭合失败的关键不是“目标不够强”一个变量，而是一个场问题：

```text
closure = target boost + identity anchor + blocker suppression - new blocker emergence
```

---

**最终总结**

Phase 755-800 的推进可以分成 6 个大阶段：

1. **Phase 755-761**：证明跨域路线存在，并发现 target boost(目标增强) 和 route suppression(路线抑制)不能混为一谈。  
2. **Phase 762-770**：提出 causal fiber(因果纤维)，说明同一语义对象在不同上下文中有稳定内部因果画像。  
3. **Phase 771-778**：从 fiber profile(纤维画像)推进到 component causality(组件因果性)，并发现 strict token closure(严格词元闭合)经常失败。  
4. **Phase 779-785**：定位 surface-form/token identity route(表面形式/词元身份路线)，说明语义正确还要经过读出桥。  
5. **Phase 786-794**：把机制拆成 Q/K/V/O 和 source-unit fiber(源单元纤维)，让公式、图谱、闭合三者对齐。  
6. **Phase 795-800**：进入 full-vocabulary blocker field(全词表阻塞场)，最终区分 target booster(目标增强器) 和 true suppressor(真实抑制器)。

所以整个研究的核心转变是：

```text
从“模型有没有选对语义”
推进到
“模型能不能在全词表竞争中闭合到唯一正确 token”
```

而 Phase 799/800 的价值就在于：它们把“闭合失败”从模糊现象变成了可计算的机制分解。