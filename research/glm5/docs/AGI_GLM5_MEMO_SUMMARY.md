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