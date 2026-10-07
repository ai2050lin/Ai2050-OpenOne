"""Deliver the claim audit; append only to the unified GLM memo."""
import hashlib,json,re,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
from phase2755_framework_audit import ROOT,OUT,MEMO,sha,write

CLAIMS=[
("F01","彻底白盒化、完整HDMCC、解释无限智能","未证实",[3076,3081,3093,3100],"已有局部观测与干预；没有完整语言编码算法、未见组合全链预测或通用控制证据。"),
("F02","Attention头是纯线性搬运工","整体表述错误",[3072],"固定Q/K及attention权重的V路径线性；完整头包含状态依赖路由。"),
("F03","Attention只路由，MLP独占知识与加工","绝对分工未证实",[3067,3072,3098],"可区分跨位置聚合与位置内更新，不能推出知识和计算的独占分工。"),
("F04","L8–20主动免疫、误差统一衰减27倍","局部现象过度解释",[3018,3019,3035,3039],"保留特定扰动的负叉积与概率效应衰减；未证明自动识别并修复语义错误。"),
("F05","全部KV写入98.3%词身份、1.7%情景","范围及信息比例解释错误",[3040,3041],"实际是L3 kv7、92次出现、26种token在样本内均值子空间的投影能量。"),
("F06","KV五分量完整解码、早层全局引力场","候选描述，未闭合",[3040,3041,3042,3043,3044],"不同统计构造不等于独立分量；前缀相关轴的因果作用需另证。"),
("F07","头作用不由参数或上游状态决定","排除推理错误",[3078],"所测投影不相关不等于全状态无信息；推理仍由参数、状态、位置等共同决定。"),
("F08","TT与头子空间角度共振决定路由","测量对象错误，共振未测",[3079,3080,3081,3094],"实际比较两组最终logit变化向量；没有头子空间角度或频率共振实验。"),
("F09","γ白化并压低废话、放大高级逻辑","白化已反证，词频故事未测",[3056,3057],"保留学习到的坐标重加权；γ不是逐词/逐任务的动态开关。"),
("F10","多词竞争与词表差方向","精确定义后保留",[3035,3039],"softmax全词表竞争及log-odds差方向成立；不能推广为所有扰动下的二词曲率律。"),
("F11","Hill容量与小集合竞争、大集合补全是普遍定律","普适主张已反证",[3074,3075,3076],"A族预算拟合可留；B/C失效，正高阶Möbius不自动证明补全机制。"),
("F12","低PR必迁移、高PR不复用","充分性被反证，必要性未证",[3082,3083,3093,3094],"14B为trunk却未满足原迁移合取门；不等于14B没有其他迁移能力。"),
("F13","物体—水果—苹果嵌套且彼此正交","按字面数学不相容",[2806,2807],"可研究共享背景与正交增量；类别聚合不等于真实MLP嵌套存储。"),
("F14","同一神经元池在两条件直接写相反符号","复用可留，符号叙事须纠正",[3067],"所选神经元集两条件都写负对齐差分；总符号翻转来自输入与其余系统的竞争。"),
("F15","末块MLP摧毁现代词、装配所有复杂能力","局部重写可留，语义扩张未证",[3095,3096,3097,3098,3099,3100],"保留末位TT跨族对齐变化；没有逐词知识摧毁或能力全部集中装配证据。"),
("F16","莎翁苹果案例是已验证的完整内部运行","待测示例",[3042,3076,3098],"关联材料是英文连接词续写和风格前缀，未找到该中文案例全链数据。"),
("F17","只剩四大深水区、流形/流体机制已确立","研究问题可留，前提未证",[3078,3093,3100],"研究连续生成、长程约束、跨任务和训练影响，不预设坍缩、虫洞或拓扑手术。"),
("F18","写读绝对解耦、只有γ放大","数学与计算解释错误",[3055,3057,3067,3098],"读出梯度依赖状态与共同RMS分母；Attention/MLP等也改变增益。"),
]

def build():
    idx=json.loads((OUT/"memo_index.json").read_text(encoding="utf-8"))
    assert sha(MEMO)==idx["sha256"],"Memo changed; reconcile before sealing."
    records=json.loads((OUT/"saved_result_inventory.json").read_text(encoding="utf-8"))
    stamp=datetime.now().strftime("%Y-%m-%d %H:%M")
    write("claim_ledger.json",dict(phase=2755,created_utc=datetime.now(timezone.utc).isoformat(),
        claims=[dict(id=i,claim=c,status=s,source_phases=p,correction=r) for i,c,s,p,r in CLAIMS]))
    text=f"## Phase 2755: HDMCC总结的逐项核对、明确错误与后续研究基础 [{stamp}]\n\n"
    text+=r"""### 1. 审查范围与总判断

**状态：完成文档、实现、保存结果与CPU数组核对；没有加载模型，没有新增GPU实验。** 用户要求核对五大系统、三大编码机制、四大空白及莎翁苹果案例，本轮不把文档审查扩大为模型测试。

结论：总结包含值得保留的局部线索，但把条件恒等式、局部干预、描述性相关、拟合曲线、比喻和未做过的案例拼成了完整白盒理论。**不能原样作为后续研究的已证实基础；可以拆成有范围、有反例、有预测合同的候选机制清单。** 这是对测量对象和推断强度的修正，不是否定研究积累，也不是因为小模型没有达到完美才否定规律。

实际文件为research/gpt5/docs/AGI_GPT5_MEMO.md。全文检索主张并追查后续反证，不声称独立重跑了“3000多个Phase”。编号不等于独立证据数，多个Phase重用同一批材料和数组。项目理论名称沿用RDC；HDMCC可作候选视角，不能用新名称提升证据等级。

"""
    text+=f"当前快照匹配到{idx['headings']}个标准Phase标题，编号{idx['first']}–{idx['last']}；抽取{len(idx['selected'])}个相关段落，核对{len(records)}份保存的result.json，独立复算3组既有数组统计。快照SHA256：{idx['sha256']}。这是本轮脚本匹配口径，不与其他快照的标题计数混用。\n\n"
    text+=r"""### 2. 逐项判定与证据定位

“错误”限定为所写强命题不成立；“未证实”不等于已经证明不存在。

| ID | 原主张 | 判定 | 可保留部分或修正 | 来源 |
| --- | --- | --- | --- | --- |
"""
    for i,c,s,ph,r in CLAIMS:
        links=[]
        for n in ph:
            v=idx["selected"].get(str(n))
            links.append(f"[GPT {n}]({MEMO.as_posix()}:{v['line']})" if v else f"GPT {n}")
        text+=f"| {i} | {c} | {s} | {r} | "+ "、".join(links)+" |\n"
    text+=r"""
### 3. 可以明确纠正的数学与机制错误

#### 3.1 固定V路径线性，不等于完整Attention线性

3072明确固定Q/K与attention权重，只改变V，线性重构首先验证该条件下的实现与采集。一般计算为：

$$
O(X)=A(X)V(X)W_O,\qquad
A(X)=\operatorname{softmax}(Q(X)K(X)^\top/\sqrt{d_k}+M).
$$

M为因果掩码；实际Qwen还包含Q/K归一化与位置处理。两个状态的完整变化为：

$$
\Delta O=[A_0\Delta V+\Delta A\,V_0+\Delta A\,\Delta V]W_O.
$$

只有ΔA=0才简化成A₀ΔV W_O。权重矩阵线性，不代表包含乘积、softmax和归一化的整个头对输入线性；3072还记录下游L35注意力权重最大变化0.762。这是运算对象不同，不是模型大小问题。标准公式参见[原Transformer论文](https://arxiv.org/abs/1706.03762)。

#### 3.2 γ并非已证实的白化器，也不是高级词汇开关

3056由γ与unembedding列标准差负相关猜测白化，3057明确修正为反方差重加权。保存数据中列标准差变异系数由0.037430升至0.118364，逆幂拟合指数约3.3076、R²约0.4158；GLM2752直接检查权重也复现离散度变化。

这些是词表矩阵列统计，不等于HiddenState协方差白化。白化需要BΣBᵀ≈I；即使精确按标准差倒数作对角缩放，也不能一般地消除坐标相关。完整RMSNorm的样本归一化与单独γ作用也须分开。

γ按隐藏坐标缩放，不是每个词的独立开关，固定权重推理时不按词频、逻辑难度或古典程度更新。压低gravity/fall、放大doth的解释没有逐词证据。RMSNorm正式定义见[原论文](https://arxiv.org/abs/1910.07467)。

#### 3.3 写读可以分解，但并不绝对独立

令h为最终残差、s(h)=√(‖h‖²/d+ε)、Dγ为γ对角阵，u_a/u_b为词表读出行：

$$
\log\frac{p_a}{p_b}=z_a-z_b
=(u_a-u_b)^\top D_\gamma h/s(h).
$$

这是可保留的精确log-odds关系，不是新发现的语言定律。其敏感度为：

$$
\nabla_h(z_a-z_b)=
\left[D_\gamma\left(\frac{I}{s}-\frac{hh^\top}{d\,s^3}\right)\right]^\top(u_a-u_b).
$$

上游写入改变h、共同分母及局部读出梯度；W_V、W_O、MLP和后续路由都能改变增益。模块位置可分，不推出统计、因果或训练独立，也不推出只有γ放大。词表差方向针对同一前缀下的token；完整概率还依赖全词表分母。

#### 3.4 “角度共振”比较对象错误，“上游无信息”没有证明

3079–3080的f₂=cos(TT_f,TT_g)，TT是两种Prompt的最终logit差。它不是目标对头权重子空间的夹角，没有频率、振荡或共振实验。TT需输出前向才能得到，不能直接冒充语言条件提供的事前路由信号。

3078只检验部分层、末位置、特定投影与范数对因果谱的预测，不代表完整上游状态、所有位置和非线性读出都没有信息。尤其基座未包含完整新注入时，其单独预测失败不能证明信息不在上游。固定计算仍由参数、状态/历史、位置和执行配置共同决定。

3080重用3079已观察的数据，不是独立确认；3081及3093–3094已有跨模型限制。“测过的特征不成功”不能推出“唯一现场点名机制已破解”。

#### 3.5 完整嵌套空间彼此正交不相容

非零U⊆V且U⊥V会推出任意u∈U满足uᵀu=0，矛盾。可以研究共享背景加正交新增分量，不能把嵌套的完整概念空间说成彼此正交。

2806高层方向由类别均值聚合，落在类别均值差张成空间内很大程度上是构造结果，不是直接发现MLP的内部知识本体。2807确有99新词类别读出推广（全维0.838、十个方向特征0.869），这是可留的类别几何线索；它没有证明嵌套存储算法。知识的类属、部分、来源等关系还须分别建模。

本轮代码追踪还确认两项口径问题：2806的类差向量按构造和为零，十个方向最多秩9；2807逐次增加相关方向特征会改变最近质心的距离度量，不能由第十个特征改善分类推出“十个方向全部独立、无冗余”。2806 Arm E直接求若干单位方向内积平方之和，没有将这些方向正交化；因此它一般不是子空间投影能量，不能自动作可加的域/类/身份能量份额。独立新词分类成绩可作为既有描述保留，相关能量解释需要修复后再用。

有限维空间不支持无限个严格正交的非零方向；近似叠加需要容差和干扰预算。有限参数可以实现可扩展组合规则，但从这一点到无限智能仍缺少算法、范围和正确性保证。

### 4. 数字可复现，但含义与总结不同

#### 4.1 98.3%是投影能量，不是整个模型的信息分配

本轮从3040数组重建26种token的均值矩阵、SVD基底B，再计算：

$$
e_i=\|B^\top v_i\|^2/\|v_i\|^2.
$$

92个128维L3 kv7 V向量的中位数为0.9833626311036598，与保存数组逐值一致；L20原值约0.9036。B由同一批出现构造，均值可以包含位置、共享语境和偏置。落在这个子空间内不等于只编码词身份；1.7%残差也不等于1.7%任务信息，低能量成分仍可能影响答案。

所谓五分量跨越公共方向、均值投影、残差、位置梯度、前缀位移等不同构造，没有证明互斥、正交、穷尽和百分比可相加。3041还撤回了普遍秩1情景轴。3044的L3相关场轴注入效率约为随机的0.98倍，L20约3.42倍；可测相关轴不自动等于可用控制轴。

#### 4.2 27倍是早晚注入的概率效应比，不是所有错误统一衰减

本轮用3035的11个tag重算：

$$
r_i=\frac{\operatorname{mean}_m|\Delta p_{A,\mathrm{early}}(m)|}
{\operatorname{mean}_m|\Delta p_{A,\mathrm{late}}(m)|}.
$$

中位r=0.03731274829112813，倒数26.8005，重算差0。它依赖指定方向、剂量、位置和读出协议；不是任意误差向量经过中层后的范数缩小27倍。

3018–3019的负内积及单元分解支持特定扰动被分布式抵消。若e为两运行的状态差，ΔM为MLP写入差：

$$
\|e+\Delta M\|^2=\|e\|^2+2\langle e,\Delta M\rangle+\|\Delta M\|^2.
$$

负叉积与其他项共同决定差异变化。“删掉负叉积”的记账反事实不等于一个实际可实现的去免疫模型。自动识别语义错误、保护苹果/重力、仅抵消坏风格和全域稳定性都未测。小规模神经元集不集中也不证明所有专门电路不存在。

3039自然直读协议仅复现部分方向特异、多词概率变化等现象，早先logistic曲率符号律变为4/9；不能沿用“唯一非线性源是读出”的强叙事。

#### 4.3 Hill是局部预算拟合，不是一般容量定律

3074对选定8头的255个子集定义干预恢复量R(S)，以x(S)=Σ|r₁(h)|作预算，拟合：

$$
\widehat R(x)=a x^\nu/(b^\nu+x^\nu).
$$

用单/双头点拟合后，A族8头/全32头外推误差约0.01294/0.03706。这是有价值的组合规模外推，未引入独立新语境。指数模型第二门误差0.05365也接近0.05阈值，不能将Hill视为唯一形式。

3076相同协议跨族时B/C结构判定失败、参数漂移，C族部分对子竞争消失且子集可超过全8头。它反驳普适版本，仍保留条件非加性与局部拟合。

GLM2750已确认3075误把“所有高阶Möbius系数非正”当作次模性的充要条件。正确条件针对二阶差分的和；正高阶系数不自动证明大集合补全或协作机制。实际次模违反可以保留，错误定理必须撤回。

#### 4.4 PR是响应谱描述，不是能力迁移开关

PR=(Σλ)²/Σλ²描述指定矩阵的能量集中度。3082的CS来自干预响应；后处理免前向不等于原始数据免干预。

3093中14B三族PR约1.16/1.37/2.34，top3约0.986/0.942/0.944，仍属trunk，原迁移合取门却只满足3/6。低PR不充分保证该规律；这也不意味着14B没有任何举一反三能力，其头级相关仍在。3083退化出口代表不可检验，不应作阴性成功复现；高PR必不复用没有证明。

独立模型很少，族对共享权重和材料；旧谱—迁移统计的并列秩、伪重复和层位敏感性已经GLM2750复核。PR可作候选协变量，与方向、信号幅度、层位、条件及真实推广一起测量。

#### 4.5 末块MLP是较实在的局部拼图，“摧毁”须改为指标下降

本轮从3098子步数组复算，活动组MLP绝对步幅占两子步绝对步幅和的中位数为4B 0.94719、14B 0.94853。分子分母都是跨族TT余弦变化；不是95%的知识、风格或语言计算集中在末块。

消融MLP比消融attention引起更大的KL/top1变化，支持该材料末块MLP对输出有实质作用。清零attention还会改变MLP输入，不能证明attention无用。formal的“摧毁”是跨族logit差方向对齐下降；原文同时记录TT范数放大，没有现代术语被删除或逻辑知识消失的测量。

3067的同一神经元集在两种条件都写负对齐差分；总响应变号来自输入方向与其余计算的竞争。同池复用值得保留，不能直接说该池自身写相反语义。3100导数、能量份额、头坐标对象错误已在GLM2750–2751修复，不得把旧“通道级完全闭合”重新装回框架。

### 5. 莎翁苹果案例只能作为待测示例

在主张关联的日志与实际材料中，未找到该中文完整Prompt及所给回答的对应全链运行。3076/3098是“The weather was cold, so”等英文连接词续写，配“In Shakespearean style,”前缀。

| 案例描述 | 核对 |
| --- | --- |
| L1–L3对苹果盖98.3%词身份章 | 数字来自其他词、L3单KV组，不能移植到这个案例。 |
| 中层保护重力语义、防止风格破坏知识 | 未做对应语义保持/选择性抵消实验；风格也是合法任务条件。 |
| h14负责因果逻辑并共振 | 它是特定4B注入三族top8的交集，不是通用逻辑头。 |
| 多头按Hill饱和产生推导 | 本例未测，Hill普适性也失败。 |
| L39摧毁现代术语、放大古典词 | L39是40块14B的零基末块，4B末块为35；指定词方向未测。 |
| doth fall与gravity决斗 | 短语不默认等于一个token，二者不必在同一步竞争；须给分词、前缀、logits及完整生成。 |

可用的教学描述是：知识、关系、风格共同影响各层状态，经attention与MLP接续，产生下一token分布，再加入自回归前缀。具体的单元、位置和作用仍待解释。这不是完整机制演示。

若后续用此例，应冻结内容×风格×因果反事实和新改写，先确认模型自然生成能力，分开评估知识、风格、语法及停止，再预测未见组合的内部响应。不能先写完整故事再挑轨迹印证。

### 6. 后续研究可采用的修订基础

保留五个对象：①类型化语言关系/角色；②共享参数下的状态依赖计算；③跨位置/层的传播与交互；④归一化及全词表读出；⑤完整前缀和KV历史驱动的自回归接续。各对象记录可执行提取、预测输入、有效范围及反例，不把免疫、引力、虫洞当作已测变量。

RDC主接口与三图谱公式不变，本轮无新增普遍定理：

$$
\mathcal T^D_{\ell,\tau}:
(\mathcal L(p),\mathbf W_\ell(p),\mathcal X_\ell(p))
\rightharpoonup\mathbf W_{\ell+1}(p),
\qquad
\operatorname{Atlas}_D=(G_{\mathrm{external}},G_{\mathrm{internal}},E^D_{\mathrm{association}}).
$$

𝓛为外部条件，𝓧为注明边界的观测场，𝐖为类型化响应束，D限定模型/输入/变换范围。这是研究接口，不是已求得的万能转移函数。真实计算底座仍为：

$$
r_\ell=H_\ell+A_\ell(N_\ell(H_\ell);KV_\ell,\mathrm{position},\mathrm{mask}),\qquad
x_\ell=N'_\ell(r_\ell),
$$
$$
H_{\ell+1}=r_\ell+W_{d,\ell}
[\operatorname{SiLU}(W_{g,\ell}x_\ell)\odot W_{u,\ell}x_\ell],\qquad
p_{t+1}=\operatorname{softmax}(W_U N_f(H_L)_t).
$$

已知前向结构不自动构成语言解释；单末位向量也不等于下一步完整状态。

核心拼图清单：固定路由的V读取恒等式保留；特定扰动抵消、方向特异读出、条件化MLP与末块重写限域保留；Hill/PR/角度关系作为条件候选；绝对解耦、完整头线性、普适白化、无限正交撤回；完整组合与自回归闭合待验证。GLM2754的角色位置读出及近似变号规律是新的可信入口，其删除可读方向而原输出不变的阴性证据一并保留。

三图谱增量属于证据治理：外部图分离实体、关系、风格和任务；内部图区分V、头拼接空间、残差坐标、MLP单元和词表方向；关联图将18项主张连接原Phase、保存数据、证据等级及后续反证。没有新增原场采集或UI，claim_ledger.json提供稳定查询ID。

### 7. 小模型因素与下一整体阶段

认可用户核心思路：规律可以近似、条件化、有例外；发现结构不以完美答题或单坐标翻转为门槛。但模型粗糙性不能修复错误导数、测量对象、数据泄漏或数学矛盾，也不能成为所有失败的解释。14B结果提醒我们同时控制语境、训练和输出协议。

稳定性同时报告变换符号/方向、幅值误差、独立材料表现、模型行为、可读性与因果性，以及失败域。高余弦不代替幅值误差；答错不证明关系信息不存在；一次删除失败不自动否定表征线索。

接续为四项大任务，均为计划、本轮未执行：

1. **图与条件运算的推广。** 沿GLM2754加入断边、分支、同端点异路径、角色交换、否定/双重否定，排除端点捷径；冻结预测输入，不读未来答案或目标状态。
2. **真实跨层接续。** 对通过推广的规律追踪事实/查询位置、QK/V与MLP路径，定位保留、改写、重建；保留全坐标/全单元背景。
3. **连续生成与长程约束。** 研究错误首次出现、完整回答和停止，不预设错误必是吸引子或流形坍缩。长程attention连接本身是架构事实，是否承载前提约束还须验证。
4. **跨模型与训练来源。** 顺序比较本地4B/14B/GLM4，区分表示、条件合成和格式；RLHF需匹配base/对齐检查点及训练来源控制，未测前不称拓扑手术。

若新图失败，先判断原规则编码的是端点、模板、关系还是其他统计量；保留可重复部分，再修订范围。能提前预测新组合响应和输出，并与实际计算路径对应，才逐步升级为机制解释。

### 8. 交付与限制

本轮CPU复算验证旧数组到统计量的一致性，不是独立模型复现，也不能排除原采集的一切系统误差。未执行新模型测试或该中文案例完整生成。既有GLM2750–2754结果保留各自来源与证据等级。

脚本：tests/glm5/phase2755_framework_audit.py、phase2755_framework_delivery.py。结果：tests/glm5/result/rdc_framework_audit_20260923/，包括原文快照、Phase摘录、保存结果身份、三组重算、18项主张账本、证据清单、报告和追加回执。无历史数据清理，无旧Phase改写。

仅追加统一GLM2755。当前GPT文件尾部再次与旧3104追加回执不同；本轮登记实际读取快照，不覆盖或重建GPT历史，不推断变更原因。完整先前审查以GLM对应Phase及各自产物为依据。
"""
    assert not [(i,ord(c)) for i,c in enumerate(text) if ord(c)<32 and c not in "\n\t"]
    (OUT/"phase_report.md").write_text(text,encoding="utf-8")
    shutil.copyfile(MEMO,OUT/"memo_source_snapshot.md")
    evidence=[ROOT/r["path"] for r in records]
    for r in records:
        evidence.extend(p for p in (ROOT/r["path"]).parent.glob("*") if p.suffix==".npz" or p.name in ("seal.json","execution.json"))
    for n in [2806,2807,3018,3019,3035,3040,3041,3057,3067,3072,3074,3076,3078,3079,3080,3082,3093,3094,3098,3100]:
        evidence.extend((ROOT/"tests/glm5").glob(f"phase{n}_*.py"))
    evidence.extend((ROOT/"tests/glm5/result/rdc_context_interaction_20260923").glob("reference_*.json"))
    evidence.append(ROOT/"tests/glm5/result/rdc_relation_stability_20260923/phase_report.md")
    write("evidence_manifest.json",dict(scope="Evidence identity, not proof of scientific correctness",
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(evidence))]))
    write("review_scope.json",dict(source_sha256=idx["sha256"],raw_results=len(records),
        selected_sections=len(idx["selected"]),claims=len(CLAIMS),array_rechecks=3,new_model_forwards=0,stamp_local=stamp))
    return text

def run(append=False):
    assert not (OUT/"append_receipt.json").exists(),"Already delivered"
    text=build()
    if not append:
        print("Draft prepared; no memo append.")
        return
    memo=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"
    before=memo.read_bytes()
    assert int(re.findall(r"^## Phase (\d+):",before.decode("utf-8-sig"),re.M)[-1])==2754
    assert b"## Phase 2755:" not in before
    payload=("\n\n"+text).encode("utf-8")
    with memo.open("ab") as f:
        f.write(payload)
    after=memo.read_bytes()
    assert after[:len(before)]==before and after[len(before):]==payload
    write("append_receipt.json",dict(phase=2755,path=str(memo.relative_to(ROOT)),
        heading_line=before.count(b"\n")+3,prefix_sha256=hashlib.sha256(before).hexdigest(),
        prefix_bytes=len(before),append_sha256=hashlib.sha256(payload).hexdigest(),
        full_sha256=sha(memo),append_only_verified=True))
    for script in ["phase2755_framework_audit.py","phase2755_framework_delivery.py"]:
        shutil.copyfile(ROOT/"tests/glm5"/script,OUT/script)
    files=[p for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ("artifact_manifest.json","manifest_verification.json")]
    entries=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    write("artifact_manifest.json",dict(files=entries))
    assert all(sha(OUT/r["path"])==r["sha256"] for r in entries)
    write("manifest_verification.json",dict(count=len(entries),all_hashes_match=True))
    print(json.dumps(dict(phase=2755,line=before.count(b"\n")+3,artifacts=len(entries))))

if __name__=="__main__":
    run("--append" in sys.argv)
