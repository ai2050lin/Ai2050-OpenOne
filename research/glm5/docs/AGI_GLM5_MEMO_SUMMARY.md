# Phase 20-300

我读的是 [AGI_GLM5_MEMO_20260601.md](D:/AI2050/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO_20260601.md:5)。Phase 20-300 的主线可以概括为：

```text
语法方向
→ 语义属性
→ 局部算子/Jacobian
→ 语法因果回路
→ 约束动力学
→ 关系/算子伪影审计
→ 子空间拓扑
→ Attention/MLP 契约
→ Identity-Role-Frame-Operator 分解
```

**阶段 1：Phase 20-25，语法角色流形与纤维结构**
核心问题：语法角色是不是近正交方向？  
测试原理：比较 nsubj/dobj/amod 等语法角色的 hidden state（隐藏状态）差分、PCA（主成分分析）、Gram-Schmidt（正交化）、跨语言/跨角色方差分解。

早期误判是“语法方向近正交”，Phase 21 修正为：很多正交性来自人为正交化。更合理模型是：

$$
h_l(x,r)
=
M_l(x)
+
A_l(r)
+
B_l(x,r)
+
\epsilon_l
$$

其中：

```text
M_l(x) = 词/语义共享底座
A_l(r) = 角色平均偏移
B_l(x,r) = 词与角色的交互项
```

这个阶段的价值是把“语法=几个方向”改成“语法=共享语义流形上的角色纤维”。

**阶段 2：Phase 26-33，语义属性不是独立方向**
核心问题：edible（可食）、color（颜色）、size（大小）等属性是不是可分离坐标？  
测试原理：属性方向、条件解耦、回归、PCA/CCA/ICA、非线性模型、probe（探针）和 logit 解码。

核心发现：属性不是互相正交的独立轴，而是条件依赖的非线性结构。

$$
h
=
f(v_{\text{attribute}}, c_{\text{context}})
+
\epsilon
$$

局部线性化为：

$$
\Delta h
\approx
J_f(v,c)\Delta v
$$

但：

$$
J_f(v,c_1)
\neq
J_f(v,c_2)
$$

意思是：同一个 edible 方向，在 red context（红色上下文）和 green context（绿色上下文）里并不完全相同。

**阶段 3：Phase 34-43，从方向转向算子和传播**
核心问题：属性/语法不是固定方向，那它是不是一个局部算子？  
测试原理：Jacobian chain（雅可比链）、扰动传播、谱分析、routing vs spectral selection（路由 vs 谱选择）、RMSNorm 校准。

核心公式：

$$
h_{l+1}
=
h_l
+
F_l(h_l)
$$

$$
J_l(h_l)
=
\frac{\partial h_{l+1}}{\partial h_l}
=
I
+
\frac{\partial F_l}{\partial h_l}
$$

扰动传播：

$$
\delta h_{l+1}
\approx
J_l(h_l)\delta h_l
$$

这个阶段开始认识到：真正重要的不是“方向在哪里”，而是“模型如何传播一个扰动”。

**阶段 4：Phase 44-58，语法图、Attention Head 与强 baseline**
核心问题：语法结构能不能从 attention graph（注意力图）恢复？  
测试原理：SLT（结构语言拓扑）、Attention → Syntax Graph、head localization（注意力头定位）、复杂句、强 baseline 对照。

核心思想：

$$
G_{\text{syntax}}
\approx
\operatorname{Graph}
\left(
A^{(l,h)}
\right)
$$

其中 \(A^{(l,h)}\) 是第 \(l\) 层第 \(h\) 个 attention head（注意力头）的注意力矩阵。

这个阶段的结论比较谨慎：attention 里有语法信息，但“能恢复语法图”不等于“attention 就是语法机制”。

**阶段 5：Phase 59-64，语法信息的因果验证**
这是早期最重要的正结果之一。  
核心问题：语法信息是可读出来，还是被模型真正使用？  
测试原理：activation patching（激活修补）、number direction（数方向）、position-controlled analysis（位置控制分析）、LayerNorm 控制失败定理、路由状态离散性。

数方向：

$$
d_{\text{num}}
=
h_{\text{plural}}
-
h_{\text{singular}}
$$

位置无关性判据：

$$
\cos(d_{\text{num}}, d_{\text{pos}})
\approx
0
$$

因果 patch：

$$
h'_l
=
h_l
+
\alpha d_{\text{num}}
$$

输出变化：

$$
\Delta z
=
z_{\text{plural-verb}}(h')
-
z_{\text{singular-verb}}(h')
$$

核心结论：number（数）是相对位置不变的隐变量，而且激活修补能改变语法输出，因此它不仅可解码，还有因果作用。

**阶段 6：Phase 65-89，微分几何、Jacobian 与计算原语**
核心问题：Transformer 是不是局部线性流形系统？语言计算有没有稳定算子？  
测试原理：局部 Jacobian、梯度流、eigenspace（特征空间）、full layer operator decomposition（完整层算子分解）、computation graph dynamics（计算图动力学）。

核心公式：

$$
\delta h_{l+k}
\approx
J_{l+k-1}
J_{l+k-2}
\cdots
J_l
\delta h_l
$$

如果某方向稳定传播，应满足：

$$
J_l v
\approx
\lambda v
$$

但这个阶段反复发现：谱相似不等于算子相似，局部几何可解释一部分，但不足以描述完整算法。

**阶段 7：Phase 90-119，多模型语义动力学与 spike/不变量**
核心问题：跨模型是否有共同语义结构？语义不变量是真实信号还是统计假象？  
测试原理：多模型对比、语义速度场、概率轨迹、causal semantic intervention（因果语义干预）、spike 传播、MLP/Attention 路由机制。

核心对象：

$$
\Delta p_t
=
p_{t+1}
-
p_t
$$

$$
\Delta h_l
=
h_l^{\text{condition A}}
-
h_l^{\text{condition B}}
$$

阶段结论：可观测语义变化常常只是输出层投影的一小部分，大量信号在 \(W_U\) 不容易直接读出的空间里。

**阶段 8：Phase 120-152，从表示转向动力学流形**
核心问题：非线性放大、子空间输运、Fisher 几何、局部 Jacobian，能否统一？  
测试原理：ablation（消融）、random control（随机对照）、subspace transport（子空间输运）、Fisher information（Fisher 信息）、Marchenko-Pastur 检验。

核心公式：

$$
\mathcal{I}(h)
=
\mathbb{E}
\left[
\nabla_h \log p(y|h)
\nabla_h \log p(y|h)^\top
\right]
$$

子空间输运：

$$
S_{l+1}
\approx
J_l S_l
$$

阶段结论：很多“低维结构”是真的，但必须严防 PCA、范数、采样不足、探针伪相关造成的假结构。

**阶段 9：Phase 153-181，约束场与语言状态变量**
核心问题：语言是不是约束传播系统？真正的状态变量是什么？  
测试原理：constraint dynamics（约束动力学）、Koopman 特征函数、logits manifold（logit 流形）、未来分布压缩、概念预测变形。

语义被重新定义为对未来可行空间的收缩：

$$
\operatorname{Meaning}(x)
=
\Delta \Omega
$$

可操作化为熵变化：

$$
\Delta H
=
H(Y_{\text{future}}|c+x)
-
H(Y_{\text{future}}|c)
$$

如果 \(\Delta H < 0\)，说明 token 压缩了未来空间，是强约束 token。

**阶段 10：Phase 182-212，Logit Lens 伪象、关系算子伪影、MLP 条件判断器**
这是一个关键纠错阶段。  
核心问题：可解码关系是否真的被模型使用？  
测试原理：Tuned Lens 对照、causal tracing、bilinear relation operator（双线性关系算子）、关系破坏实验、MLP/Attention 归因。

关系算子假设：

$$
s(i,j)
=
h_i^\top A h_j
$$

Phase 208 发现它可高精度解码关系，但 Phase 209 证明破坏它对输出几乎没影响。因此：

```text
可解码性 ≠ 因果使用
```

Phase 210-212 收紧为：

$$
h_{l+1}
=
h_l
+
\Delta h_{\text{attn},l}
+
\Delta h_{\text{mlp},l}
$$

$$
\Delta h_{\text{mlp}}
=
W_{\text{down}}
\left[
\sigma(W_g h)
\odot
W_u h
\right]
$$

结论：Attention 更像 where to read（在哪里读），MLP 更像 how to update（如何更新状态）。

**阶段 11：Phase 213-228，Transformer 作为状态依赖动力系统**
核心问题：语言机制是不是吸引子动力学、Jacobian 场或三阶段动力系统？  
测试原理：Jacobian-vector product、谱结构、协方差传输、残差可预测性 \(R^2(\Delta h|h_l)\)、Jacobian 子空间三体对齐。

核心动力学：

$$
h_{l+1}
=
h_l
+
F_l(h_l)
$$

$$
\delta h_{l+1}
=
J_l(h_l)\delta h_l
+
\epsilon_l
$$

Phase 225 的关键几何等式：

$$
\operatorname{Top5}(J_l)
\equiv
\operatorname{Top5}(\operatorname{PCA}(\Delta h_l))
\perp
\operatorname{Row}(W_U)
$$

意思是：约束传播主方向与 Jacobian 主方向重合，但几乎在输出层 \(W_U\) 的盲区中。这就是“暗物质动力系统”思想。

**阶段 12：Phase 229-239，关系约束力学与路线重置**
核心问题：hidden state 不是最终对象，是否应转向关系闭合性？  
测试原理：future entropy compression（未来熵压缩）、非交换操作、关系评分传播、系统路线审计。

非交换性测试：

$$
D(A,B)
=
1
-
\cos
\left(
h(AB),
h(BA)
\right)
$$

若：

$$
D(A,B) > 0
$$

则：

$$
AB \neq BA
$$

结论：语言操作确实有非交换性，尤其在中层最明显。但 Phase 239 又做了重要路线修正：不要过早建大理论，先彻底搞清编码机制。

**阶段 13：Phase 240-254，子空间拓扑与跨位置语义核心**
核心问题：hidden state 的真实结构是什么？有没有双信号、语义瓶颈、跨位置共享核心？  
测试原理：全层 SVD、去偏置 ID、LayerNorm 前后对比、跨位置 ID、注意力子空间翻译、单轴多概念定位、因果验证。

双信号模型：

$$
h_l
=
b_l
+
s_l
$$

其中：

```text
b_l = rank-1 bias direction（偏置方向）
s_l = semantic residual（语义残差）
```

语义维度应在去偏置后估计：

$$
\operatorname{ID}_{\text{semantic}}
=
\operatorname{ID}
\left(
h_l
-
\operatorname{Proj}_{b_l}(h_l)
\right)
$$

LayerNorm 的作用：

$$
\operatorname{LN}(h)
=
\frac{h-\mu(h)}{\sigma(h)}
\odot
g
+
\beta
$$

结论：所谓低维坍缩很多时候是偏置/范数假象；LayerNorm 是把微弱语义信号重新均衡到可计算范围的关键机制。

**阶段 14：Phase 255-284，直接破解编码机制、W_U、Attention/MLP 路径**
核心问题：编码基本单元是神经元、方向、子空间、W_U 轴，还是组件契约？  
测试原理：superposition（叠加）检测、W_U SVD、反义词/近义词几何、logit attribution、grammar head decoding、Jacobian/MLP 干预、attention graph dynamics。

W_U 低秩读出：

$$
W_U
=
U\Sigma V^\top
$$

有效秩：

$$
r_{\text{eff}}
=
\exp
\left(
-
\sum_i p_i \log p_i
\right),
\quad
p_i
=
\frac{\sigma_i}{\sum_j \sigma_j}
$$

隐藏态投影到读出空间：

$$
\rho_{WU}(h)
=
\frac{\|\operatorname{Proj}_{\operatorname{Row}(W_U)}h\|^2}
{\|h\|^2}
$$

结论：单神经元解释失败，superposition 明显；W_U 不是简单语义相似空间，而更像语义对比/候选竞争读出接口。

**阶段 15：Phase 285-293，真实 Forward Patching 与 Attention-MLP 契约**
核心问题：手工 patch 和真实 forward patch 是否一致？Attention 和 MLP 是配合还是冲突？  
测试原理：real forward activation patching（真实前向激活修补）、head-level patching、route/content separation（路由/内容分离）、recomputed contract（重算式契约）。

契约测试核心：

$$
h_{l+1}^{A\leftarrow B}
=
h_l^A
+
\Delta h_{\text{attn}}^B
+
\operatorname{MLP}_l
\left(
h_l^A+\Delta h_{\text{attn}}^B
\right)
$$

衡量 progress：

$$
\operatorname{PROG}
=
\frac{
\|h_{\text{patched}}-h_A\|
-
\|h_{\text{patched}}-h_B\|
}
{
\|h_A-h_B\|
}
$$

结论：Qwen3 比较能吸收外来 attention，GLM4 强烈拒绝，DS7B 某些情况下甚至敌对。机制不是统一的，存在模型特异的 module contract（模块契约）。

**阶段 16：Phase 294-300，Identity-Role-Frame-Operator 严格分解**
核心问题：之前的语法/角色方向是不是被 token identity（词元身份）和 frame（句框）混淆了？否定 operator（操作符）又如何编码？  
测试原理：token alignment fix、identity-role decoupling、ANOVA、orthogonal decomposition、role subspace causal test、operator-scope decomposition。

Identity-role 分解：

$$
h_l(t,r)
=
\mu_l
+
I_l(t)
+
R_l(r)
+
\operatorname{Interaction}_l(t,r)
+
\epsilon_l
$$

Phase 297 加入 frame：

$$
h_l
\approx
\mu
+
I(\text{token})
+
R(\text{role})
+
F(\text{frame})
+
I\times F
+
\epsilon
$$

Phase 300 否定操作符分解：

$$
h
=
\mu
+
I(\text{operand})
+
O(\text{operator})
+
S(\text{scope})
+
F(\text{frame})
+
O\times S
+
I\times O
+
O\times F
+
\epsilon
$$

关键结论：role（角色）方向比最初想的更依赖 token/frame；但 negation operator（否定操作符）比 role 更稳定、更低维、更跨 operand 一致。也就是说，“not” 更像真正的操作符，而普通语法角色更像 identity/frame 条件下的增量编码。

**总判断**
Phase 20-300 的推进不是线性累加，而是多次自我推翻后的收紧：

```text
1. 从“语法方向”收紧到“角色纤维”
2. 从“属性轴”收紧到“条件非线性属性”
3. 从“可解码关系”收紧到“因果使用”
4. 从“hidden state 几何”收紧到“Jacobian/残差更新动力学”
5. 从“大理论约束场”收紧到“先破解编码机制”
6. 从“神经元/方向”收紧到“子空间、读出接口、模块契约”
7. 从“角色方向”收紧到“identity-role-frame-operator 分解”
```

最核心的理论收获是这条公式链：

$$
h_{l+1}
=
h_l
+
\Delta h_{\text{attn},l}
+
\Delta h_{\text{mlp},l}
$$

$$
\Delta h_{\text{attn}}
=
W_O
\operatorname{softmax}
\left(
QK^\top
\right)
V
$$

$$
\Delta h_{\text{mlp}}
=
W_{\text{down}}
\left[
\sigma(W_g h)
\odot
W_u h
\right]
$$

$$
\operatorname{logits}
=
W_U h_{\text{final}}
$$

但这只是 Transformer 计算图。Phase 20-300 真正推进出来的语言机制版本是：

$$
\text{Language Encoding}
=
\text{Identity Basis}
+
\text{Role/Frame Increment}
+
\text{Operator Transform}
+
\text{Attention Routing}
+
\text{MLP Conditional Update}
+
\text{Low-rank Readout}
$$

也就是说，到 Phase 300 为止，研究已经从“找一个语义方向”推进到“拆出身份、角色、句框、操作符、路由、条件更新和读出接口之间的完整机制分工”。


# Phase 301-594


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


# Phase 595-826

下面这段 Phase 595-826 的主线，可以概括成一句话：

**研究从“某个答案 token 为什么能出来”，逐步推进到“全词表竞争中，哪一条因果路径能稳定压制所有 blocker（阻断词），让正确答案在自然生成里闭合”。**

参考文件：[AGI_GLM5_MEMO.md](D:/AI2050/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md:5)

**总路线**
Phase 595-826 大致分成 10 个阶段：

1. Phase 595-604：发现“加一个方向向量”不能解释答案闭合，必须看完整轨迹。
2. Phase 605-643：从 final token readout（最终读出）转向 sequence-level trajectory（序列级轨迹）。
3. Phase 649-668：拆出 protocol field（协议场）、format gate（格式门）、token1 transition writer（第一 token 转换写入器）。
4. Phase 669-712：建立 graph atlas（机制图谱），把 head、QK、V、readout 拆成路径。
5. Phase 720-736：从 head atlas 进入 full-path functional atlas（全路径功能图谱）。
6. Phase 755-790：从局部路径转向 closure fiber（闭合纤维）与全词表竞争。
7. Phase 799-811：提出 closure-fiber score（闭合纤维分数）和 blocker field（阻断场）。
8. Phase 812-820：发现 token 级闭合不够，改成 answer-class / span / alias 闭合。
9. Phase 821-824：用 boundary class（边界类别）做因果定位和稀疏子空间搜索。
10. Phase 826：进入 exact-natural consistency（精确样本与自然样本一致性）验证。

---

**阶段 1：Phase 595-604，否定“静态向量闭合”**

核心问题：  
早期假设是，正确答案可能对应一个固定方向，只要把这个方向加到 hidden state（隐藏状态）里，答案就会出来。但 Phase 595-604 发现不是这样。模型不是靠一个孤立向量完成答案，而是靠多层、多位置、多 token 的轨迹共同完成。

测试原理：  
比较自然正确轨迹、自然错误轨迹、人工投影轨迹，看人工加向量是否能复现自然正确状态。

核心公式：

$$
\Delta h = h_{\text{correct}} - h_{\text{wrong}}
$$

$$
h' = h_{\text{wrong}} + \alpha \Delta h
$$

如果静态向量成立，那么：

$$
P(y_{\text{correct}} \mid h') > P(y_{\text{wrong}} \mid h')
$$

但测试结果说明：  
人工投影可以提升一部分 logit（输出分数），却不能稳定复现自然正确路径。因此结论是：

$$
\text{answer closure} \ne \text{single vector addition}
$$

中文理解：  
不是“答案向量一插就通”，而是“整条生成路径要同时对齐”。

---

**阶段 2：Phase 605-643，进入序列级轨迹闭合**

核心问题：  
如果单个 final token 不够，那么答案闭合是不是发生在完整答案序列上？

测试原理：  
把模型生成答案时每个 token 的 hidden state、final norm、readout 都拿出来，比较自然正确轨迹和干预轨迹。

核心公式：

$$
S_c = \sum_{k=1}^{K} R_{c,k}(N(h_{t+k}))
$$

其中：

- $h_{t+k}$：第 $k$ 个答案 token 的隐藏状态。
- $N(\cdot)$：final norm（最终归一化）。
- $R_{c,k}$：第 $k$ 位 token 对候选答案 $c$ 的读出贡献。
- $S_c$：整个答案串的总分。

候选答案 margin（间隔）：

$$
M_c =
\sum_{k=1}^{K} \log P(c_k \mid x,c_{<k})
-
\max_{j \ne c}
\sum_{k=1}^{K} \log P(j_k \mid x,j_{<k})
$$

闭合目标：

$$
M_{\text{correct}} > 0
$$

中文理解：  
不是只看第一个字是否对，而是看完整答案串的累计概率是否超过所有竞争答案。

---

**阶段 3：Phase 649-668，拆出协议场和 token 转换写入器**

核心问题：  
模型经常“知道答案”，但输出格式不对，例如不进入 short answer（短答案）协议，或者先输出解释性文本。这说明答案生成前还有 protocol gate（协议门）。

测试原理：  
定位 answer_label_aligned、answer_colon、separator 等位置，观察这些位置是否控制模型进入短答案模式。

协议场公式：

$$
F_{\text{protocol}}
=
A_{\text{answer-label}}
+
A_{\text{colon}}
+
A_{\text{separator}}
+
A_{\text{format-prior}}
$$

第一 token 转换状态：

$$
h_{\text{value-trans}}
=
G_{\text{model}}
(
\{W_i(x)\},
R(x),
N(x)
)
$$

其中：

- $W_i(x)$：不同 writer head（写入头）的贡献。
- $R(x)$：residual stream（残差流）。
- $N(x)$：normalization（归一化）状态。
- $h_{\text{value-trans}}$：从“知道答案”转换到“开始写答案”的状态。

中文理解：  
这一阶段发现，答案不是直接从语义区读出来，还要经过“现在该短答了”“现在该写第一个 token 了”的协议转换。

---

**阶段 4：Phase 669-712，建立机制图谱与 QK/V 拆解**

核心问题：  
前面发现很多局部机制，但还没有统一图谱。Phase 669 开始把机制组织成 graph atlas（机制图谱）：谁提供语义、谁写入、谁控制格式、谁最终读出。

基本 Transformer 更新公式：

$$
h_{l+1,p}
=
h_{l,p}
+
A_{l,p}(QK_{l,p},V_{l,p})
+
M_{l,p}(h_{l,p})
$$

输出公式：

$$
P(x_{t+1}\mid x_{\le t})
=
\text{softmax}(W_U h_{L,t})
$$

QK/V 拆解公式：

$$
C = \sum_s a_s v_s
$$

$$
\Delta C
=
\sum_s (a_s^T-a_s^S)v_s^S
+
\sum_s a_s^S(v_s^T-v_s^S)
+
\sum_s (a_s^T-a_s^S)(v_s^T-v_s^S)
$$

解释：

- 第一项：QK/addressing（寻址）差异。
- 第二项：V/content（内容）差异。
- 第三项：寻址和内容共同变化的交互项。

中文理解：  
注意力头不是一个黑箱。它至少分成“看哪里”和“搬什么内容”两部分。

---

**阶段 5：Phase 720-736，从 head 图谱进入 full-path atlas**

核心问题：  
只看单个 head 不够，需要把 source（来源）、writer（写入）、carrier（承载）、rewriter（重写）、readout（读出）串成完整路径。

全路径状态分解：

$$
h_l(o,r,f,k)
=
S_l
+
K_l(k)
+
P_l(f)
+
R_l(r)
+
O_l(o)
+
V_l(v)
+
B_l(o,r,v)
+
M_l
+
I_l
+
\epsilon_l
$$

其中：

- $S_l$：共享语义骨架。
- $K_l(k)$：知识源。
- $P_l(f)$：格式协议。
- $R_l(r)$：推理路线。
- $O_l(o)$：对象身份。
- $V_l(v)$：答案值。
- $B_l(o,r,v)$：对象、路线、答案之间的绑定项。
- $\epsilon_l$：噪声或未解释残差。

source erasure（来源擦除）测试：

$$
C_G(l,h)
=
\sum_{t\in G}
\alpha_{l,h}(a,t)V_{l,h}(t)
$$

$$
\text{Loss}_K(h,G)
=
-
\langle
h_T^{\text{erase}(h,G)} - h_T,
d_K
\rangle
$$

中文理解：  
把某一类 source token 的贡献擦掉，看答案知识方向掉多少，就知道这组 token/head 是否真的在传递知识。

---

**阶段 6：Phase 755-790，转向 closure fiber**

核心问题：  
前面已经能定位很多组件，但仍然不能保证自然生成稳定闭合。于是问题变成：  
正确答案所在的“因果纤维”到底是哪一条？

closure fiber（闭合纤维）可以理解为：  
在高维 hidden space（隐藏空间）里，有一束状态路径最终都会导向同一个正确答案闭合结果。这一束路径不是单点，而是一个等价路径族。

抽象公式：

$$
\mathcal{F}_y
=
\{
h
\mid
\operatorname{Decode}(h)=y,
\quad
B(h)=\varnothing,
\quad
\text{protocol}(h)=\text{valid}
\}
$$

其中：

- $\mathcal{F}_y$：答案 $y$ 的闭合纤维。
- $B(h)$：在状态 $h$ 下压过正确答案的 blocker 集合。
- protocol valid：输出协议合法。

中文理解：  
不是找“一个正确点”，而是找“所有能自然通向正确答案的状态通道”。

---

**阶段 7：Phase 799-811，closure-fiber score 与 blocker field**

核心问题：  
即使 target logit（目标输出分数）升高，也可能有别的 token 同时升得更高。于是 Phase 799 之后把目标改成“全词表竞争胜利”。

blocker field（阻断场）：

$$
B_{\text{full}}(x)
=
\{
v \in V,\ v\ne y
\mid
z_v(x) > z_y(x)
\}
$$

意思是：  
所有 logit 分数超过正确答案 $y$ 的 token，都是 blocker。

target-neutral decomposition（目标中性拆解）：

$$
\Delta h = h_{\text{donor}} - h_{\text{recipient}}
$$

$$
\Delta h_{\text{target}}
=
\langle \Delta h,u_y\rangle u_y
$$

$$
\Delta h_{\text{neutral}}
=
\Delta h - \Delta h_{\text{target}}
$$

closure-fiber score 可以抽象成：

$$
C_{\text{fiber}}
=
\alpha \Delta z_y
-
\beta |B_{\text{new}}|
+
\gamma |B_{\text{resolved}}|
-
\lambda \Omega(\Delta h)
$$

其中：

- $\Delta z_y$：正确答案 logit 提升。
- $B_{\text{new}}$：新增 blocker。
- $B_{\text{resolved}}$：被消除的 blocker。
- $\Omega(\Delta h)$：干预幅度或副作用惩罚。

中文理解：  
一个好干预不是“把正确答案推高”这么简单，而是要同时做到：推高正确答案、压下旧 blocker、不制造新 blocker、干预尽量小。

---

**阶段 8：Phase 812-820，闭合目标从 token 改成 answer-class/span/alias**

核心问题：  
token 级别太严格也太脆弱。比如正确答案可能有别名、大小写、多个 token 形式。Phase 812-820 把目标改成答案类闭合。

answer class（答案类别）：

$$
E(y)
=
\{
v
\mid
\operatorname{norm}(\operatorname{text}(v))
\in V(y)
\}
$$

类别分数：

$$
z_{E(y)}
=
\max_{v\in E(y)} z_v
$$

答案类闭合：

$$
C_{\text{answer-class}}
=
\mathbf{1}
[
z_{E(y)}
>
\max_{u\notin E(y)} z_u
]
$$

span score（多 token 答案分数）：

$$
Z(S)
=
\sum_{i=1}^{k}
\log P(v_i \mid x,v_{<i})
$$

$$
\bar{Z}(S)
=
\frac{1}{k}
\sum_{i=1}^{k}
\log P(v_i \mid x,v_{<i})
$$

span 闭合：

$$
C_{\text{span-score}}
=
\mathbf{1}
[
\bar{Z}(S_y)
>
\max_{S\notin \mathcal{S}(y)}
\bar{Z}(S)
]
$$

中文理解：  
如果答案是 “New York”，不能只看 “New” 这个 token；也不能因为模型输出 “NYC” 就简单判错。闭合目标必须和真实答案语义对齐。

---

**阶段 9：Phase 821-824，用 boundary class 做因果定位**

核心问题：  
需要判断一次干预后，输出边界有没有变好。于是引入 boundary class（边界类别），给不同输出状态排序。

边界等级：

$$
r(T)=5,\quad
r(C)=4,\quad
r(B_r)=3,\quad
r(FT)=2,\quad
r(G)=1
$$

$$
r(F)=r(O)=r(W)=r(U)=0
$$

边界改善：

$$
\Delta r
=
r(B_{\text{after}})
-
r(B_{\text{before}})
$$

稀疏子空间候选：

$$
\Delta h^{(m)}
=
h_D^{(m)}
-
h_R^{(m)}
$$

$$
s_i^{(m)}
=
\Delta h_i^{(m)} g_i^{(m)}
$$

搜索目标：

$$
J(G_j)
=
[
\mathbf{1}(B_{\text{after}}=T),
\Delta r,
r(B_{\text{after}}),
\mathbf{1}(\text{protocol repaired})
]
$$

中文理解：  
这一阶段不是盲目 patch 整个 hidden state，而是在找少数关键维度：哪些维度一改，输出边界就从错误类跳到正确类。

---

**阶段 10：Phase 826，exact-natural consistency**

核心问题：  
一个干预在 exact prompt（精确构造样本）上有效，不代表在 natural prompt（自然样本）上也有效。Phase 826 开始要求二者一致。

自然 donor 转移：

$$
\hat{y}_{N,S}
=
F
(
h_R^{(m)}
+
P_S(h_N^{(m)}-h_R^{(m)})
)
$$

一致性目标：

$$
C_{\text{exact-natural-consistency}}
=
\sum_N
\mathbf{1}
[
B_{\text{exact},S}=T
\land
B_{N,S}=T
]
$$

最终目标函数：

$$
E(S)
=
\alpha C_{\text{exact-target}}
+
\beta C_{\text{natural-target}}
+
\gamma C_{\text{exact-natural-consistency}}
-
\mu C_{\text{degraded}}
-
\rho C_{\text{route-divergence}}
$$

中文理解：  
这一步是在防止“只在实验样本里闭合”的假阳性。真正有价值的机制，必须在人工可控样本和自然生成样本中都走同一条因果路线。

---

**最终判断**

Phase 595-826 的推进不是简单重复，而是目标层级不断升级：

$$
\text{single vector}
\rightarrow
\text{sequence trajectory}
\rightarrow
\text{protocol field}
\rightarrow
\text{mechanism graph}
\rightarrow
\text{full-path atlas}
\rightarrow
\text{closure fiber}
\rightarrow
\text{full-vocabulary blocker field}
\rightarrow
\text{answer-class/span closure}
\rightarrow
\text{exact-natural consistency}
$$

它的核心研究路线是：

**先证明“答案不是一个静态向量”，再拆出生成答案所需的协议、路径、读出和 blocker 竞争，最后把闭合目标改成可预测、可验证、可复用的全词表因果纤维。**


# Phase 827-874


我读的是 [AGI_GLM5_MEMO.md](D:/AI2050/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md:91697) 中 Phase 827-874。总体看，这一段不是简单堆实验，而是从“找可闭合的 causal fiber（因果纤维）”逐步推进到“gear（齿轮）必须在 blocker field（阻塞场）和 output gate（输出门）同时允许时才构成干净因果边”。

**总推进线**

Phase 827-874 的主线可以压缩成一句话：

> 从“哪些组件能让答案闭合”推进到“在什么竞争场、什么输出门状态下，某个齿轮才是真正可预测、可复用、无副作用的因果机制”。

也就是从单纯找 patch / component，升级为：

```text
gear 本身
+ blocker field 条件
+ output gate 状态
+ route purity 副作用约束
= clean causal edge
```

**阶段一：827-834，闭合纤维与路线门控**

核心问题：  
前面已经能找到一些 patch 让目标答案上升，但 exact answer（精确答案）和 natural answer（自然生成答案）经常互相牺牲。所以 Phase 827 开始把目标改成“exact + multi-natural 同时闭合”。

测试原理：  
对候选组件集合 \(S\) 做干预，观察它是否同时满足：

1. exact prompt 能输出目标；
2. 多个 natural prompt 也能输出目标；
3. 不破坏其他目标；
4. 不引入 degraded case（退化样本）。

核心公式是 Phase 827 的闭合纤维目标：

$$
C_{\text{exact+multi-natural}}(S)
=
\mathbf{1}
\left[
B_{\text{exact},S}=T
\land
\sum_{N\in\mathcal{N}}
\mathbf{1}(B_{N,S}=T)
\ge k
\right]
$$

对应的 causal fiber（因果纤维）可以写成：

$$
\mathcal{F}_{T,k}
=
\left\{
S:
B_{\text{exact},S}=T
\land
\sum_{N\in\mathcal{N}}\mathbf{1}(B_{N,S}=T)\ge k
\land
D_S=0
\right\}
$$

Phase 828-830 尝试把不同 component 合成：

$$
h_R'
=
h_R
+
P_{S_1}(h_D^{(1)}-h_R^{(1)})
+
P_{S_2}(h_D^{(2)}-h_R^{(2)})
$$

但结果说明：简单相加不能稳定带来新闭合，会产生 interference（干扰）。所以 Phase 831-834 转向 internal gate predictor（内部门控预测器），判断“这个 component 在当前 prompt / route 下该不该启用”。

门控公式大致变成：

$$
g_i(d,c)
=
\Omega
\left(
C_i,
Q_d,
P_d,
B_i,
R_i
\right)
$$

其中 \(C_i\) 是组件因果效果，\(Q_d\) 是 prompt 类型，\(P_d\) 是 protocol 状态，\(B_i\) 是 blocker 画像，\(R_i\) 是 route 类型。

这一阶段的结论：  
真正决定闭合的不是“某个组件强不强”，而是“这个组件在当前路线边界下是否被允许工作”。

**阶段二：835-836，token blocker 到 span blocker**

核心问题：  
只看 first token rank（第一个 token 排名）不够。模型可能第一个 token 好了，但完整答案 span 仍失败。

测试原理：  
把目标从单 token 扩展到完整 answer span（答案片段），使用 teacher-forced scoring（教师强制评分）计算整个答案序列的平均 log probability。

核心公式：

$$
S_{\text{span}}(a|x)
=
\frac{1}{|a|}
\sum_{t=1}^{|a|}
\log p(a_t|x,a_{<t})
$$

然后比较 target answer 与 competitor answer：

$$
M_{\text{contrast}}
=
S_{\text{target}}
-
S_{\text{contrast}}
$$

$$
M_{\text{generic}}
=
S_{\text{target}}
-
S_{\text{generic blocker}}
$$

结果很关键：  
Qwen3 更像 first-token rank gate；DS7B 不完全由 span margin 解释，更多依赖 contrast blocker 是否被清除。于是 blocker 不再只是一个 token，而是一个竞争场。

**阶段三：837-843，全局齿轮响应图谱**

核心问题：  
前面是在找局部 component，但需要统一记录每个 gear 对输出系统的多维影响。

测试原理：  
对每个 gear 做干预，不只看 target 是否成功，还记录 rank、logit、blocker、echo、protocol、rollout 等响应向量。

核心响应向量：

$$
\Delta\Phi_m
=
(
\Delta B,
\Delta R_{\text{target}},
\Delta S_{\text{target}},
\Delta S_{\text{contrast}},
\Delta S_{\text{generic}},
\Delta E_{\text{echo}},
\Delta P_{\text{protocol}},
\Delta C_{\text{rollout}}
)
$$

Phase 839 开始测 gear interaction（齿轮交互）：

$$
I_Q(C)
=
Q(C)
-
\max_{m_i\in C} Q(m_i)
$$

如果组合 \(C\) 明显强于任何单个 gear，就说明不是独立加法，而是存在 interaction edge（交互边）。

Phase 840-843 的关键发现是：  
成功组合里有些 gear 是正向 answer writer，有些却是负向 MLP gear。负向 gear 不是“坏东西”，而可能是在削弱 blocker。

负向齿轮的作用可写成：

$$
Role(g)
=
\begin{cases}
answer\ lift, & \Delta z_{\text{answer}}>0 \\
blocker\ weakening, & \Delta z_{\text{blocker}}<0 \\
mixed, & \Delta z_{\text{answer}}>0 \land \Delta z_{\text{blocker}}<0
\end{cases}
$$

**阶段四：844-850，几何边界与强交互边**

核心问题：  
能不能把 gear 的作用写成一个边界方程？也就是答案类和干扰类之间的 logit boundary 如何移动。

测试原理：  
定义 target 与 object / competitor 的边界：

$$
B(x)
=
z_{\text{target}}(x)
-
z_{\text{object}}(x)
$$

单个 gear 的边界增量：

$$
\Delta_g(x)
=
B_g(x)-B_0(x)
$$

组合 gear 的实际增量：

$$
\Delta_S(x)
=
B_S(x)-B_0(x)
$$

如果完全可加，应满足：

$$
\widehat{\Delta}_S(x)
=
\sum_{g\in S}\Delta_g(x)
$$

残差为：

$$
R_S(x)
=
\Delta_S(x)
-
\sum_{g\in S}\Delta_g(x)
$$

若 \(R_S>0\)，是 synergy（协同）；若 \(R_S<0\)，是 antagonism（拮抗）。

这一阶段的结论：  
边界方程不是简单线性加法，必须加入 context gate（上下文门）和 route gate（路线门）。于是 Phase 850 形成全局齿轮图谱结构：

$$
\mathcal{G}_{gear}
=
(
V_{state},
V_{route},
V_{gear},
V_{gate},
V_{boundary},
E_{causal},
E_{interaction},
E_{transport},
E_{closure},
\Omega_{evidence}
)
$$

**阶段五：851-857，从强边到全词表阻塞场**

核心问题：  
前面很多成功只是局部 token 成功，不一定是全词表答案类别闭合。Phase 854 开始显式看 full-vocabulary blocker field（全词表阻塞场）。

测试原理：  
不是只问目标 token 是否上升，而是问：所有非答案类 token 是否仍压过答案类。

答案类闭合：

$$
C_{\text{answer-class}}(x)
=
\mathbf{1}
\left[
\max_{t\in A(y)} z_t(x)
>
\max_{u\notin A(y)} z_u(x)
\right]
$$

全词表 blocker 集合：

$$
B_{\text{full}}(x)
=
\left\{
u\notin A(y):
z_u(x)>
\max_{t\in A(y)}z_t(x)
\right\}
$$

全词表 blocker 清除：

$$
C_{\text{full-blocker}}
=
\mathbf{1}
[
B_{\text{full}}(x)=\emptyset
]
$$

Phase 855-857 加入 short rollout（短生成验证）和 prompt gate（提示门控）：

$$
C_{\text{short-closure}}
=
G_{\text{prompt}}
\cdot
C_{\text{answer-class}}
\cdot
C_{\text{clear-rollout}}
\cdot
(1-C_{\text{object-echo}})
$$

结论：  
first-token answer-class closure 对 rollout 有很强预测力，但不是所有模型都稳定。DS7B 尤其容易 object echo（对象回声）。

**阶段六：858-861，跨域齿轮发现与证据阶梯**

核心问题：  
这些 gear 是 universal（跨域通用）的吗？还是 domain-local（领域局部）的？

测试原理：  
在 color、animal、material、geometry 等 domain 中独立发现 gear，再看是否共享 exact channel、layer band、sign pattern。

gear support 公式：

$$
s_{d,l,c}(x,p)
=
a_{l,c}(x,p)
\cdot
\left[
(W_U[t_d(x)]-W_U[o(x)])W_{down,l}
\right]_c
$$

跨域同构：

$$
I_{\text{exact}}(d_i,d_j)
=
|G^*_{d_i}\cap G^*_{d_j}|
$$

$$
I_{\text{layer}}(d_i,d_j)
=
|L(G^*_{d_i})\cap L(G^*_{d_j})|
$$

Phase 860 建立证据阶梯：

$$
G_{\text{high}}
=
\{
g:
\Delta C_{\text{rep}}>0,
\Delta C_{\text{loss}}=0,
\Delta C_{\text{control}}=0,
H_{\text{split}}\ge2,
H_{\text{prompt}}\ge2
\}
$$

结论：  
没有得到强 universal exact gear，但得到稳定的结构指纹：late-layer、two-channel、negative-blocker、domain-local。

**阶段七：862-866，符号机制与干净路线公式**

核心问题：  
这些 late-layer two-channel gear 到底是在提升答案，还是削弱 blocker，还是两者都有？

测试原理：  
对 gear 做 zero、flip、half、scale-up 等不同干预，分别测 answer delta、blocker reduction、object delta、format side effect。

核心变量：

$$
A=\Delta z_{\text{answer}}
$$

$$
B=
blocker\_count_{\text{original}}
-
blocker\_count_{\text{intervened}}
$$

$$
Z_b
=
z_{\text{intervened}}(t_b)
-
z_{\text{original}}(t_b)
$$

干净混合路线定义为：

$$
CleanMixedRoute(g,d,m)
=
[A>0]
\land
[B>0]
\land
[Z_b<0]
\land
[\Delta z_{\text{object}}\le0.25]
\land
[\text{no echo}]
\land
[\text{no format side effect}]
$$

这一阶段很重要：  
它把“齿轮有效”拆成了 answer lift（答案提升）、blocker weakening（阻塞削弱）、side effect filter（副作用过滤）三部分。

但 Phase 866 只是样本内规则，还没有跨对象、跨 prompt 可靠泛化。

**阶段八：867-874，条件化失败、阻塞场准入与输出门分解**

核心问题：  
Phase 866 的 clean route 规则在 holdout 上失败。说明路线纯净性不是 gear 固有属性，而是依赖当前输入的 blocker field。

于是公式从：

$$
CleanMixedRoute(g,d,m)
$$

改成：

$$
CleanMixedRoute(g,d,m|x)
$$

其中：

$$
x = object + prompt + blocker\ field
$$

Phase 869-870 建立 blocker field admissibility（阻塞场准入）：

$$
FieldAdmissible(B_x)
=
\neg TooManyBlockers
\land
\neg ObjectDominatesClass
\land
\neg FormatDominates
\land
ReducibleOriginalBlockers
$$

经验规则：

$$
field\_base
=
[blocker\_count<20]
\land
[class\_minus\_object\_logit\ge0]
\land
[format\_pressure<3]
$$

再加 gear effect：

$$
field\_strict\_plus\_effect
=
field\_strict
\land
[answer\_delta>0]
\land
[blocker\_reduction>0]
\land
[original\_blocker\_delta<0]
\land
[object\_echo=0]
$$

Phase 871 外部验证后发现：  
FieldAdmissible 能显著减少 false positive，但仍不是充分条件。还需要 output gate（输出门）。

Phase 872-874 最终把输出拆成三层：

第一层，输出状态是否打开：

$$
OutputStateOpen(x')
=
TargetTop1(x')
\land
TargetBeatsNonTarget(x')
$$

第二层，是否发生输出状态转移：

$$
ObservedOutputTransition(x,x')
=
\neg RolloutClear(x)
\land
OutputStateOpen(x')
$$

第三层，是否是干净因果边：

$$
CleanCausalEdge(g,d,m,x)
=
ObservedOutputTransition(x,T_g(x))
\land
FieldAdmissible(B_x)
\land
GearEffect(g,d,m,x)
\land
NoSideEffect(x,T_g(x))
$$

Phase 874 的最终结论非常关键：

```text
CleanCausalEdge 是 OutputTransition 的子集；
OutputTransition 不一定是 CleanCausalEdge。
```

也就是说，模型输出正确不等于我们找到了干净机制；只有同时满足 blocker field 准入、gear effect、output transition、无副作用，才算真正闭合了一条 clean causal edge。

**一句话总结**

Phase 827-874 的实质进步是：  
研究从“找能让答案出现的 patch”推进到“建立 gear-field-gate 三层机制图谱”。最重要的理论转折是：语言生成闭合不是单个齿轮决定，而是由当前竞争场、路线门控、输出门状态共同决定。

最终机制公式可以浓缩为：

$$
CleanCausalEdge
=
GearEffect
\land
FieldAdmissible
\land
OutputTransition
\land
NoSideEffect
$$

这比早期 closure-fiber score 更进一步：closure-fiber 主要回答“哪些组件组合能闭合”，而 Phase 827-874 进一步回答“为什么同一个组件组合有时闭合、有时失败”。



# Phase 875-907

你写的 `Phase 8745` 在文件里不存在，我按文件实际编号理解为 **Phase 875 到 Phase 907**。这一段位于 [AGI_GLM5_MEMO.md](D:/AI2050/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md:115093) 后半部分。

总体一句话：

> Phase 875-907 的推进逻辑，是从“干净因果闭合”扩展到“有效输出转移图谱”，再从语义答案边界一路追到协议续写场、句号边界和 EOS 终止动作缺口。

也就是研究对象从：

```text
哪个 gear 让答案出来
```

升级成：

```text
gear 如何改变 full-vocab boundary
-> answer class 如何出现
-> protocol continuation 为什么接管
-> period 为什么不是 EOS
-> EOS 为什么不能自然胜出
```

**阶段一：Phase 875-879，非干净输出转移与观察边界割**

核心问题：  
Phase 874 已经发现：

```text
OutputStateOpen != ObservedOutputTransition != CleanCausalEdge
```

也就是说，输出变正确不一定是 clean causal edge（干净因果边）。Phase 875-879 专门研究那些“不满足 clean 条件，但确实把输出推到正确答案”的路线。

关键定义：

$$
NoncleanOutputTransition(g,d,m,x)
=
ObservedOutputTransition(x,T_g(x))
\land
\neg CleanCausalEdge(g,d,m,x)
$$

有效输出转移扩展为：

$$
EffectiveOutputTransition
=
CleanCausalEdge
\lor
NoncleanOutputTransition
$$

Phase 878 进一步发现，nonclean route 不是 blocker field 完全没变，而是：

$$
NoncleanEffectiveTransition
=
ObservedOutputTransition
\land
TargetRankTakeover
\land
BlockerFieldDisplacement
\land
\neg OriginalBlockerWeakening
$$

Phase 879 把 blocker displacement 收紧成 observed boundary cut proxy（观察边界割代理）：

$$
C_{obs}(x,g)
=
B_{base}(x)-B_{int}(x,g)
$$

$$
ObservedBoundaryClosed(x,g)
=
[B_{int}(x,g)=\emptyset]
\land
[rank_{int}(target)=1]
\land
[|C_{obs}|=|B_{base}|]
$$

结论：  
nonclean route 不是噪声。它是有效输出路线的一部分，只是它不通过“压低原始 blocker”获胜，而是通过 target rank takeover（目标排名接管）和 blocker field 位移获胜。

**阶段二：Phase 880-884，从 pair 归因到主导齿轮与图谱评分**

核心问题：  
Phase 879 只证明了 observed boundary closed，不证明 pair 是最小割。所以 Phase 880 重新测试 full gear set 和 proper subset。

边界闭合：

$$
BoundaryClosed(G,x)
=
[class\_blocker\_count(G,x)=0]
\land
[class\_target\_rank(G,x)=1]
$$

齿轮集合最小割：

$$
GearSetMinimalCut(G,x)
=
BoundaryClosed(G,x)
\land
\neg BoundaryClosed(\emptyset,x)
\land
\forall G'\subset G,\neg BoundaryClosed(G',x)
$$

结果很关键：DS7B 的 nonclean pair `L27C16651 + L24C3875` 不是二齿轮最小割，真正主导的是单齿轮 `L27C16651`。

于是引入：

$$
DominantGearClosure(g,x)
=
BoundaryClosed(\{g\},x)
$$

Phase 882-884 开始转向 atlas-first（图谱优先），不再只追求 closure，而是给每条边打分：

$$
S_{edge}
=
5D
+3C
+2A
+B
+M
-3X
-F
-1.5R
-4N
$$

其中可理解为：

```text
D: domain-specific rate，领域特异率
C: closure rate，闭合率
A: answer rate，答案率
B: blocker reduction，阻塞减少
M: minimality bonus，最小性奖励
X: cross-domain side effect，跨域副作用
F: false closed，假闭合
R: random control，随机控制泄漏
N: non-minimal penalty，非最小惩罚
```

结论：  
这几步把研究从“某个 pair 有效”推进到“证据校准全局齿轮图谱”。最重要的收紧是：有效组合不等于最小机制，pair 里可能有 dominant gear、companion gear、redundant gear。

**阶段三：Phase 885-890，稳定边界、局部子空间与 target-lift 主导**

核心问题：  
Phase 884 找到了 stable boundary candidate（稳定边界候选），但还不知道它是不是 signed minimal gear（带符号最小齿轮）。

Phase 885 测试 holdout、random control、neighbor control、opposite mode。结果是：很多候选在 holdout 上有效，但 opposite mode 也能复现，所以它们不是简单的“正负号齿轮”，更像 local subspace boundary gear（局部子空间边界齿轮）。

带符号最小边界可写成：

$$
SignedMinimalBoundary(e)
=
StableBoundaryCandidate(e)
-
OppositeModeReproduction(e)
-
SameLayerRandomReproduction(e)
-
NeighborChannelReproduction(e)
$$

Phase 886-887 检查 candidate 和 opposite 是否移除同一批 blocker：

$$
Removed_{candidate}
=
BaseBlockers-IntervenedBlockers_{candidate}
$$

$$
Removed_{opposite}
=
BaseBlockers-IntervenedBlockers_{opposite}
$$

共享割：

$$
SharedRemoved
=
Removed_{candidate}\cap Removed_{opposite}
$$

单 blocker 精确割：

$$
ExactSingleBlockerCut(t)
=
SameBoundaryClosure
\land
[base\_blocker\_count=1]
\land
[SharedRemoved=\{t\}]
$$

Phase 888-890 做 direction-set intervention（方向集合干预）和 restore 测试：

$$
h'_c=
\begin{cases}
0, & mode=zero \\
-h_c, & mode=flip \\
0.5h_c, & mode=half \\
\alpha h_c, & mode=scale\_up
\end{cases}
$$

恢复重开：

$$
RestoreReopen(m,C)
=
BoundaryClosed(logits_m)
\land
\neg BoundaryClosed(Restore(logits_m,logits_0,C))
$$

结果：  
DS7B 大量 closure 不是单个 cut-token 的内部必要因果；distributed restore 不能重开边界。更准确解释是：

```text
target-lift dominated boundary migration
目标提升主导的边界迁移
```

也就是目标类被抬高，整体边界迁移，而不是某个 blocker token 被单点切断。

**阶段四：Phase 891-896，目标提升来源、多轴互补与 color pair**

核心问题：  
如果不是 blocker restore，那么 target lift 从哪里来？多个通道之间是否存在互补？

目标提升：

$$
TargetLift(S,m)
=
logit_{S,m}(target\_class)-logit_0(target\_class)
$$

互补增益：

$$
Complementarity(S,m)
=
TargetLift(S,m)
-
\max_{i\in S}TargetLift(\{i\},m)
$$

组合独有闭合：

$$
ClosureWithoutSingle(S,m)
=
BoundaryClosed(S,m)
\land
\neg \exists i\in S:BoundaryClosed(\{i\},m)
$$

Phase 892-896 的最重要发现：

```text
DS7B color route:
L26C8587 + L27C15369
```

是一个 known-axis minimal pair candidate（当前已知坐标轴集合内的最小 pair 候选）。

已知坐标轴最小 pair：

$$
KnownAxisMinimalPair(a,b,U,x)
=
NoSinglePair(a,b,x)
\land
\forall \{c,d\}\subset U,\{c,d\}\ne\{a,b\}:
\neg BoundaryClosed(\{c,d\},x)
$$

长 rollout 稳定性：

$$
LongRolloutStable(S,x,T)
=
ClassHit(S,x,T)
\land
ClearAnswer(S,x,T)
\land
\neg ObjectEcho(S,x,T)
\land
\neg ProtocolDrift(S,x,T)
$$

结果：  
DS7B color pair 在 color 域复现强，但不能跨 domain 复用。说明它不是 universal language pair，而是 domain-specific route pair。

**阶段五：Phase 897-899，领域坐标轴图谱与协议缺口**

核心问题：  
既然 color pair 不能跨域复用，就要为每个 domain 发现自己的坐标轴。

候选轴分数：

$$
AxisScore(g,d)
=
MeanAbsActivation(g|domain=d)
-
MeanAbsActivation(g|domain\ne d)
$$

领域候选集合：

$$
U_d
=
TopK_g AxisScore(g,d)
\cup
HistoryAxes(d)
$$

领域内 no-single pair：

$$
DomainNoSinglePair(a,b,x,d)
=
BoundaryClosed(\{a,b\},x,d)
\land
\neg BoundaryClosed(\{a\},x,d)
\land
\neg BoundaryClosed(\{b\},x,d)
$$

Phase 897-898 发现：非颜色 domain 可以建立 candidate_U，但多数 pair 很弱；更稳定的是 single-axis route，例如 DS7B animal 的 `L27C16651`、qwen3 material 的 `L31C2257`。

Phase 899 是巨大转折：first-token boundary closure 不等于 clean natural answer。

首词闭合：

$$
FirstTokenClosed(S,x)
=
\mathbf{1}[ClassRank(S,x)=1]
\cdot
\mathbf{1}[FullClassBlockerCount(S,x)=0]
$$

干净协议 rollout：

$$
CleanProtocolRollout(S,x)
=
AnswerClassRollout(S,x)
\cdot
(1-ObjectEcho(S,x))
\cdot
(1-ProtocolDrift(S,x))
$$

Phase 899 的结论：

$$
FirstTokenClosed(S,x)
\centernot\Rightarrow
CleanProtocolRollout(S,x)
$$

数据上是：

```text
answer_class = 68 / 77
clean_answer_no_protocol = 0 / 77
protocol_drift = 77 / 77
```

也就是说，语义答案类已经被拉出来，但输出继续进入解释、列表、字段、长短语。研究目标因此从 semantic axis 转向 protocol stop gate。

**阶段六：Phase 900-903，协议停止门与协议续写场**

核心问题：  
答案类已经出现，为什么模型不停？有没有简单的 protocol stop gate？

协议缺口样本：

$$
ProtocolGap(S,x)
=
AnswerClassRollout(S,x)
\cdot
ProtocolDrift(S,x)
$$

控制后干净闭合：

$$
ControlledClean(C,S,x)
=
AnswerClassRollout(C,S,x)
\cdot
(1-ObjectEcho(C,S,x))
\cdot
(1-ProtocolDrift(C,S,x))
$$

Phase 900 测 step=1/2 的 repeat、zero、flip、head zero，结果：

```text
clean_answer_no_protocol = 0
```

Phase 901 不再先控制，而是审计 stop token 竞争力：

$$
StopRank(x)
=
Rank\left(
\max_{z\in \mathcal{T}_{stop}}
logit(z|x,y_{\le t^*})
\right)
$$

$$
ProtocolRank(x)
=
Rank\left(
\max_{z\in \mathcal{T}_{protocol}}
logit(z|x,y_{\le t^*})
\right)
$$

结果：

```text
stop_top10 = 61 / 68
period_top50 = 68 / 68
median_protocol_rank = 1
median_eos_rank = 147
```

解释：stop 不是完全缺失，soft stop/period 接近，但 protocol continuation 仍然排第一，EOS 较弱。

Phase 902 搜索 protocol suppressor：

$$
\Delta_{protocol}(C,x)
=
logit_C(P_0(x))-logit_0(P_0(x))
$$

$$
Removed(C,x)
=
\mathbf{1}
[
ProtocolRank_0(x)=1
\land
ProtocolRank_C(x)>1
]
$$

结果：能压低 protocol token，但 clean 仍为 0。

Phase 903 把 protocol continuation field 图谱化：

$$
\mathcal{T}_{protocol}
=
\mathcal{T}_{newline}
\cup
\mathcal{T}_{comma}
\cup
\mathcal{T}_{field}
\cup
\mathcal{T}_{explanation}
\cup
\mathcal{T}_{list}
$$

协议替代边：

$$
Substitution(C,x)
=
c_{base}(x)\rightarrow c_C(x)
$$

阶段结论：  
协议漂移不是某个 token 的孤立问题，而是一个 protocol continuation field。典型替代关系是：

```text
comma <-> newline
period -> newline
```

**阶段七：Phase 904-907，终止动作、period 边界与 EOS 缺口**

核心问题：  
既然 period 接近、protocol 可被压低，为什么仍然没有 clean answer？Phase 904-907 把 stop 拆成 period boundary 和 EOS action。

严格 clean：

$$
Clean_{strict}(C,x)
=
AnswerClass(C,x)
\land
\neg ObjectEcho(C,x)
\land
\neg ProtocolDrift(C,x)
\land
\neg StrictProtocolDrift(C,x)
$$

Phase 904 发现：

```text
nominal clean = 6
strict clean = 0
```

并提出链式闭合标准：

$$
\Delta z_{protocol}<0
\Rightarrow
rank(protocol)\downarrow
\Rightarrow
rank(stop)\uparrow
\Rightarrow
StopAction
\Rightarrow
Clean_{strict}
$$

当前只达到前三步，没有达到 StopAction。

Phase 905 拆开 period 和 EOS：

$$
StopTop1(x)
=
EOSTop1(x)
\lor
PeriodTop1(x)
$$

真正停止应是：

$$
ClosureStop(x)
=
EOSAction(x)
\lor
[
PeriodBoundary(x)
\land
\neg ContinuationAfterPeriod(x)
]
$$

结果：

```text
stop_top1 = 55
stop_top1_period_best = 55
stop_top1_eos_best = 0
period_then_continuation = 55
strict_clean = 0
```

也就是说，所谓 stop_top1 全部是句号，不是 EOS。句号胜出后仍继续生成。

Phase 906 强制 EOS 对照：

$$
EOSAvailable(x)=1
$$

但：

$$
EOSActionNatural(x)=0
$$

数据：

```text
forced EOS strict clean = 68 / 68
natural EOS top1 = 0 / 68
after-period EOS top50 = 0 / 68
```

这证明：EOS 不是 tokenizer/config 不可用，而是自然竞争场中 EOS 极弱。

Phase 907 进一步找 period 后 continuation source：

$$
\Delta rank_{EOS}^{\ell,kind}(x)
=
rank_{patched}(EOS|x,Period)
-
rank_{base}(EOS|x,Period)
$$

EOS 近邻：

$$
EOSProximity(x)
=
\mathbf{1}[rank(EOS|x)\le K],
\quad K\in\{10,50\}
$$

真正 EOS 动作：

$$
EOSAction(x)
=
\mathbf{1}[rank(EOS|x)=1]
\land
\neg ContinuationAfterEOS(x)
$$

结果：

```text
eos_rank_improved = 2105 / 4504
patched_eos_top1 = 0
patched_eos_top10 = 13
patched_eos_top50 = 17
```

只有 GLM4 出现 EOS top10/top50 近邻，三模型都没有 EOS top1。

**最终总公式**

Phase 875-907 后，语言输出机制被拆成：

$$
Output(x)
=
F(
S_{answer}(x),
S_{protocol}(x),
S_{period}(x),
S_{EOS}(x),
A_{substitution}(x),
T_{termination}(x)
)
$$

其中：

```text
S_answer: 语义答案场
S_protocol: 协议续写场
S_period: 句号/标点边界场
S_EOS: EOS 竞争场
A_substitution: 协议替代图
T_termination: 真实终止动作场
```

当前已经比较清楚的是：

```text
S_answer
S_protocol
S_period
A_substitution
部分 S_EOS proximity
```

仍未闭合的是：

```text
T_termination
EOS action dominance
strict clean natural answer
```

**总判断**

Phase 875-907 最大的推进不是“完成闭合”，而是把失败位置定位得非常清楚：

```text
语义答案场已经能把答案类别推出来；
full-vocab boundary 可以被 domain axes / pair axes 改变；
但 clean natural answer 失败在 protocol continuation 和 EOS termination action。
```

所以这一段的研究路线可以概括为：

```text
Clean-only atlas
-> Effective-transition atlas
-> Dominant/subspace boundary atlas
-> Domain-specific axis atlas
-> Answer-class rollout atlas
-> Protocol continuation field atlas
-> Period/EOS termination gap atlas
```

最核心的结论是：

> 当前已经比较接近“全局齿轮图谱”，但还没有破解完整语言编码机制。真正缺口不在“模型知不知道答案”，而在“模型如何自然选择停止、避免协议续写、形成 strict clean answer”。


















# Phase 908-940





我按 [AGI_GLM5_MEMO.md](D:/AI2050/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md:3333) 里的 Phase 908-940 读下来，主线可以概括为一句话：

从“为什么 EOS（结束符）不能自然闭合”一路缩小到“后层 L39 MLP（第39层多层感知机）里有可复用边界齿轮”，再发现“不同 blocker（阻塞者）需要不同齿轮”，最后转向“语义方向是否能桥接到输出边界竞争”。

**1. Phase 908-917：终止路线、阻塞带、理论框架确认**

核心问题：模型明明知道答案，为什么最后不能干净停止？

测试原理：先看 EOS 是否只是“离得近但赢不了”。通过 L0 attention（第0层注意力）、prompt span（提示片段）、route delta（路线差分）、L4 MLP（第4层 MLP）、L39 MLP 等干预，观察 EOS rank（排名）、EOS logit（对数几率）和 blocker margin（阻塞边界差）。

核心公式：

$$
M_{EOS}(x)=z(EOS|x)-\max_{v\ne EOS}z(v|x)
$$

$$
\tilde h=h+\alpha d_{route}
$$

$$
\mathcal B_k(x)=TopK_{v\ne EOS}(z_v(x))
$$

$$
\text{closure}=1[M_{EOS}(x)\ge 0 \land rank(EOS)=1 \land strict\_clean=1]
$$

结果：L0 attention 能把 EOS 拉近，L4 MLP 能做弱边界调节，L39 MLP 能把 EOS 推到 rank 2 附近，但仍没有自然 top1 / strict-clean（严格干净闭合）。这阶段的结论是：问题不是“没有 EOS 信号”，而是 EOS 被 full-vocabulary blocker band（全词表阻塞带）压住。

**2. Phase 918-920：L39 有符号边界齿轮被定位**

核心问题：Phase 915 看到 L39 MLP 整体放大会让 EOS 接近胜出，那么到底是哪批通道在起作用？

测试原理：捕获 L39 MLP down_proj 输入通道，计算每个 channel（通道）对 EOS 和 `"a"` blocker 的读出贡献，再放大或抑制通道组。

核心公式：

$$
C_j(v|x)=a_j^{39}(x)\cdot \left(W_U(v)^T W_{down}^{39}[:,j]\right)
$$

$$
C_j(EOS-a|x)=a_j^{39}(x)\cdot \left((W_U(EOS)-W_U(a))^T W_{down}^{39}[:,j]\right)
$$

$$
G_{margin+}^{64}=Top64_j\,C_j(EOS-a|x)
$$

$$
a_j'=
\begin{cases}
f a_j,& j\in G\\
a_j,& j\notin G
\end{cases}
$$

结果：Phase 918 证明 GLM4 的 L39 中确实有 EOS-vs-a signed margin subspace（有符号边界子空间）；Phase 919 冻结来源通道组后跨样本仍有效；Phase 920 把它压缩为 consensus gear（共识齿轮），正向组 top1/margin>=0 大量成功，random / rotated / a-logit-only 负控制为 0。这里可以说：L39 的 EOS-vs-a 边界齿轮存在，而且不是普通随机 patch。

**3. Phase 921-924：从“齿轮存在”转向“自然门控在哪里”**

核心问题：既然拨 L39 共识齿轮能让 EOS 赢，为什么模型自然状态下没有自己拨？

测试原理：固定 L39 consensus gear，扫描低因子，看哪些状态更容易闭合；再测试 route_alpha（路线强度）和 protocol pressure（协议压力）的响应曲线。

核心公式：

$$
G_{cons}=Top64_j\sum_s 1[j\in G_s]
$$

$$
Gap_{boundary}(x)=-M_B(x)
$$

$$
Support_{consensus}(x)=\sum_{j\in G_{cons}}C_j(EOS-a|x)
$$

$$
Pressure_{gate}(x)=Gap_{boundary}(x)-Support_{consensus}(x)
$$

$$
M_{\alpha,p}(x,f)=z_{EOS}(x;\alpha\Delta r,pP,G_{39}f)-z_{blocker}(x;\alpha\Delta r,pP,G_{39}f)
$$

结果：1.375 是明显转折点，1.75/2.0 基本能闭合 12/12；但候选上游正向干预没有稳定新增闭合。route_alpha 不是越大越好，二维 route-protocol surface（路线-协议响应曲面）存在，但新增闭合仍只集中在 fish 个例。结论：自然门控不是单调旋钮，而是状态匹配曲面。

**4. Phase 925-927：响应曲面泛化，但 blocker 必须分裂**

核心问题：fish 个例是不是偶然？曲面结构能不能扩到更多 case？

测试原理：从旧实验里扩展候选 seed（种子状态），按 route_alpha × protocol_factor 画更大响应曲面，再按 blocker class（阻塞类别）拆分。

候选筛选思想：

$$
C(x)=1[usable(x)\land(near\_margin(x)\lor top10(x)\lor weak\_candidate(x))]
$$

结果：Phase 925 选出 GLM4 的 96 个 surface seeds；Phase 926 验证 30 个 seeds / 60 张曲面，发现曲面结构泛化了，但新增闭合仍只有 2 个 fish 坐标；Phase 927 拆分后发现 article_a（冠词 a 阻塞）和 punctuation_period（句号阻塞）完全不是同一类锁。当前 L39 EOS-vs-a 齿轮对 `"a"` 有用，对 `"."` 基本打不开。

**5. Phase 928-936：标点 blocker 的专属齿轮、公共骨架和 case residual**

核心问题：句号阻塞者是不是有自己的齿轮？

测试原理：对 punctuation_period seeds 单独搜索 L39 margin_support_pos_64，再验证 opening threshold（打开阈值）、fixed gear（固定齿轮）和 case residual（样本残差齿形）。

核心公式：

$$
f_{open}(s)=\min\{f:M_{candidate}(s,\alpha,p,f)\ge 0 \lor rank(EOS)=1\}
$$

$$
G_{punct}(x)=G_{common}+G_{case}(x)+G_{residual}(x)
$$

$$
R_{case}=G_{case}-G_{common}
$$

$$
R_{case}^{-i}=\left(\bigcap_{s\in case,s\ne i}G_s\right)-G_{common}
$$

结果：Phase 929 发现 factor 2.25 时 30/30 punctuation states 都能被推到 EOS top1/margin>=0，但 strict_clean 仍为 0。Phase 931 的 fixed_topfreq_64 只能覆盖 20/30；Phase 932-933 加上 case residual 后覆盖 30/30，LOSO（留一状态）仍成立；Phase 934 排除“只是通道更多”的混杂；Phase 935 发现 residual gate candidate 但完全和 chair case 纠缠；Phase 936 同 case 未见状态 90/90 可迁移。结论：标点边界像“公共齿轮骨架 + case-specific 齿形补偿”，但自然选择齿形的 gate 仍没找到，strict-clean 仍没解决。

**6. Phase 937-939：从输出边界转向语义复用机制**

核心问题：如果要破解语言编码机制，不能只看 EOS，还要看 fruit/color/function（水果/颜色/功能）这类语义因子如何复用。

测试原理：先做 hidden state atlas（隐藏状态图谱），看同 target（目标标签）的状态是否更相似；再构造 semantic direction（语义方向）做 causal transfer（因果迁移）；最后用中文模板和 specificity control（特异性控制）排除通用扰动。

核心公式：

$$
gap=\mathbb E[\cos(h_i,h_j)|same\ target]-\mathbb E[\cos(h_i,h_j)|diff\ target]
$$

$$
d_{r,y,l,t}=\mathbb E[h_l|r,y,t]-\mathbb E[h_l|r,\neg y,t]
$$

$$
h_l'=h_l+\alpha d
$$

$$
d_{specific}=d_{target}-Proj_{span(d_{wrong},d_{template})}(d_{target})
$$

结果：qwen3 的 color/function 方向较干净；DS7B 有弱正结果；GLM4 英文模板下控制项污染强，但中文 color 明显改善。category（类别）方向整体不稳。结论：语义方向确实存在一段短因果链，但不是完整语义编码闭合。

**7. Phase 940：语义方向到输出边界竞争的桥接**

核心问题：通过 Phase 939 筛过的语义方向，是否不仅改变目标 label，还能改变 target-vs-boundary（目标词对边界词）的竞争？

测试原理：选 specific_direction，在 first-token output boundary（首词输出边界）上比较 target 与 period / punctuation / protocol / EOS 的 margin。

核心公式：

$$
M_{boundary}=z_{target}-\max_{b\in Boundary}z_b
$$

$$
\Delta M_{boundary}=M_{after}-M_{before}
$$

$$
BridgeGain=\Delta M_{boundary}(specific)-\max_{control}\Delta M_{boundary}(control)
$$

结果：qwen3 最干净，color/function 多个语言对都有正 bridge gain；GLM4 边界移动很强，但 random/template/wrong controls 也强，所以只能算正但不干净；DS7B 只有 function en->en 弱正。结论：Phase 940 建立了 semantic-to-boundary bridge evidence（语义到边界桥接证据），还不是 channel-level boundary gear closure（通道级边界齿轮闭合）。

**总判断**

Phase 908-940 的推进不是重复做同一个实验，而是在不断收缩问题边界：

知道答案 ≠ 能停止；  
EOS 靠近 ≠ EOS 赢；  
EOS top1 ≠ strict-clean；  
人工拨齿轮 ≠ 自然门控；  
语义方向有效 ≠ 完整语言生成闭合。

当前最硬的成果是：GLM4 中已经找到 L39 EOS-vs-a 共识边界齿轮，以及 punctuation blocker 的 common gear + case residual 结构。最大缺口仍是 natural gate（自然门控）、strict-clean transition（严格干净转移）、跨模型复现，以及语义方向到具体通道齿轮的闭合桥。