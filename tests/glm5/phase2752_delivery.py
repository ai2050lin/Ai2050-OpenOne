"""Assemble verifiable Phase2752 results; append memo only when all arms finish."""
import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
import numpy as np
from phase2752_context_interaction import ROOT,OUT,now,sha,write,FAMILIES

LABELS=dict(entity='新实体',wording='新表述／原实体',joint_wording='新实体＋新表述',role_order='未见角色×顺序',depth='三层关系链')
METHOD_LABELS=dict(family_mean='固定族均值',surface='长度／位置',graph_only='关系角色摘要',source_product='单条件状态＋乘积项',hybrid='摘要＋状态混合')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def table(summary):
    methods=tuple(METHOD_LABELS)
    lines=['| 测试范围 | 独立世界／四条件组 | '+' | '.join(METHOD_LABELS[m] for m in methods)+' |',
           '| --- | --- | '+' | '.join(['---:']*len(methods))+' |']
    for s,label in LABELS.items():
        item=summary['splits'][s]
        lines.append(f'| {label} | {item["worlds"]}／{item["groups"]} | '+' | '.join(f'{item["methods"][m]["interaction_mean"]:.4f}' for m in methods)+' |')
    return '\n'.join(lines)


def external_graphs():
    mat=read(OUT/'material.json')
    graphs=[]
    for w in mat['worlds']:
        p=w['entities'][:w['depth']+1]
        if w['family']=='category':
            edges=[dict(source=p[0],relation='member_of',target=p[1])]+[dict(source=p[i],relation='subclass_of',target=p[i+1]) for i in range(1,w['depth'])]+[dict(source=p[1],relation='disjoint_from',target=w['entities'][4])]
        else:
            relation={'role':'gives_key_to','spatial':'left_of','containment':'inside'}[w['family']]
            edges=[dict(source=p[i],relation=relation,target=p[i+1],event_index=i+1 if w['family']=='role' else None) for i in range(w['depth'])]
        graphs.append(dict(world=w['id'],family=w['family'],nodes=w['entities'],edges=edges,
                           semantics='Generator-provided facts; not a model-inferred circuit',role=w['role'],surface_order=w['order']))
    write(OUT/'external_graphs.json',dict(created_utc=now(),graphs=graphs))


def index(completed=False):
    arms={}
    for label,path in [('4B',OUT/'4B'),('14B',OUT/'14B'),('assertion4B',OUT/'assertion_control/4B')]:
        arms[label]=dict(status='analyzed' if (path/'prediction_summary.json').exists() else 'captured' if (path/'capture_done.json').exists() else 'pending',
                         path=str(path.relative_to(ROOT)))
    write(OUT/'index.json',dict(phase=2752,created_utc=now(),status='completed_bounded_phase' if completed else 'in_progress',
        research_question='Predict condition interactions from known relation/role/context and allowed source states; critically audit universal-head claims',
        models=arms,claims='reference_claims.json',external_graphs='external_graphs.json',
        report='phase_report.md' if completed else None,
        evidence_scope='Native last-position fields; fixed synthetic families; input scopes and diagnostic status explicit',
        prior_evidence='tests/glm5/result/rdc_trusted_rebuild_20260923/index.json',
        api='/api/rdc-construction/context-interaction',ui_scope='Read-only route implemented and tested; live server restart/UI deployment not claimed'))


def report():
    summaries={s:read(OUT/s/'prediction_summary.json') for s in ('4B','14B')}
    control=read(OUT/'assertion_control/4B/prediction_summary.json')
    norms=read(OUT/'reference_readout_statistics.json')
    text=r'''本阶段回答两个问题：能否在看到双条件结果之前，根据输入中已知的关系、角色、语境以及明确允许的单条件状态预测交互；参考资料一是否可以据此宣布“头权重万能复用已彻底破解”。

**结论边界先行：已经交付可复算的条件交互预测器，但没有完成语言机制理论。参考资料一保留了若干真实局部观察，却把条件恒等式、描述性词表投影和有限相关性扩张成全局机制，关键结论不成立。** 下列结果区分预先冻结的主分析与看到4B结果后追加的诊断，不用后者冒充独立确认。

本轮共7056次正式前向：原问句4B3328、14B400，明确陈述句4B3328；这些材料共享352个基础世界，不能加总为7056个独立样本。陈述句对照中，验证选定的状态预测器把五类外推的交互误差从零交互基线1降至0.617–0.735；固定族均值为0.842–0.927。关系／角色摘要单独只有有限收益。**因此可信的是“给定三个真实源状态后，部分交互可被稳定预测”；尚不能声称“只根据语言关系就完成了理论”。**

### 1. 材料、测量和可用输入

四个合成英语任务族：类别归属、钥匙转交中的初始施事／最终受事、左右空间关系、嵌套容器。352个世界，各有5个唯一合成实体字符串；832个“世界×表述”组，每组普通／正式语气与中性／否定条件的2×2四格，共3328次4B正式前向。它们不是3328个独立样本。模板来自同一有限生成语法；实体字符串隔离不等于子词词表隔离或预训练中未知。

4B训练128世界×2表述，验证32世界×2表述；新实体、新实体＋新表述、未见角色×事实顺序、三层关系链各48世界；另在128个训练世界上保留第三种表述，专测表述变化。训练覆盖深度1、2，测试深度3。训练角色×顺序仅00、01、10，组合测试11。四条件组内token长度完全匹配；跨表述长度／位置仍变化，以单独基线控制，不能宣称已消除所有混杂。

14B按运行前冻结的成本子集完成400次前向：每族8训练世界、2验证世界、每类测试3世界，且仅一种训练表述。其实体真值／深度覆盖较窄，不能把模型差异全部归因于参数规模。两个CUDA模型顺序运行，14B使用10GiB GPU预算＋CPU卸载，BF16、eager、batch1、无缓存、无截断。首个14B加载进程异常退出且无Python traceback；原日志保留，重试单独加载成功，具体根因未被证明，不能补写成确定OOM。

保存末token的全部原生坐标、每个返回HiddenState边界和额外的末层归一化前残差。4B为38×2560、14B为42×5120；它们不是全token场。每条样本核对最后残差相加、最终RMSNorm输出；独立试运行与正式重复样本核对。数组FP32保存是原BF16值的无损嵌入，不能视为FP32推理精度验证。

关系预测器实际使用**人工生成图的摘要特征**：任务族、查询角色、事实顺序、路径深度、句式／语境描述及指定交叉项。它不是自动语义解析器，也不是完整关系图网络。长度／位置为独立对照；答案、关系真值、世界ID、测试双条件状态和未来token均不进入预测输入。状态预测器额外使用同一层的三个已观测源状态，需要三次原模型前向；不称为纯文本预测器、早层推导整个网络或完整机制提取。

### 2. 数学对象与可执行算法

对世界及表述x，令h^{sn}_{ℓ}(x)为语气s∈{0,1}、否定n∈{0,1}下同一边界的末位状态。交互是有限差分的**定义**：

$$I_\ell(x)=h^{11}_\ell-h^{10}_\ell-h^{01}_\ell+h^{00}_\ell,$$
$$\widehat h^{11}_\ell=h^{10}_\ell+h^{01}_\ell-h^{00}_\ell+\widehat I_\ell.$$

不能把四次真实前向得到的I本身当作事先预测。主算法比较零交互、固定族均值、长度／位置岭回归、关系角色摘要岭回归、摘要＋长度回归、基态单坐标回归、单条件状态多项式回归、摘要残差＋状态混合回归。

对坐标j，状态特征为
$$z_{\ell,j}=(h^{00}_{\ell,j},\delta^S_{\ell,j},\delta^N_{\ell,j},\delta^S_{\ell,j}\delta^N_{\ell,j},(\delta^S_{\ell,j})^2,(\delta^N_{\ell,j})^2),$$
其中δ^S=h^{10}−h^{00}、δ^N=h^{01}−h^{00}。每个坐标独立岭回归；摘要路线为多输出岭回归，所有原生坐标保留，不用PCA或Top-K定义主干。

$$\widehat I_{\ell,j}=b_{\ell,j}+a_{\ell,j}^{\mathsf T}\operatorname{standardize}_{train}(z_{\ell,j}),\qquad
\min_{a,b}\frac1{N_{train}}\sum_{x\in train}(I_{\ell,j}-\widehat I_{\ell,j})^2+\lambda\|a\|^2.$$

这是**拟合模型**，不是新数学定理。训练集拟合均值、尺度和系数；λ仅在验证集从0.01、0.1、1、10中选择；各方法固定λ后评估所有测试，不再重拟合。主评估为final-norm边界；其他层和raw-final单列。主误差E_I=||Ihat−I||/||I||；完全不预测交互恒为1。次误差E_H=||hhat11−h11||/||h11−h00||，分母是条件变化，不是巨大原始状态范数。零分母记缺失并计数，embedding交互为零，不参与“高精度”宣称。

不确定性按整个世界重采样2000次，同世界表述和四条件保持在一块。区间只描述这些固定模板下的源世界差异，不表示从自然语言所有模板或模型总体抽样，也不是经过多重选择校正的普适显著性。

### 3. 冻结主分析的实际结果

以下是交互相对L2误差的组均值，越小越好；零交互基线均为1。不能把1−E_I直接称为解释方差。
'''
    for side,summary in summaries.items():
        text+=f'\n**{side}：验证集选中 `{summary["selected_method"]}`。**\n\n'+table(summary)+'\n'
        winner=summary['selected_method']
        text+='\n该预先选定方法的组合变化误差E_H：'+ '；'.join(f'{label} {summary["splits"][s]["methods"][winner]["combined_change_mean"]:.4f}' for s,label in LABELS.items())+'。\n'
    text+=r'''
4B里，状态信息提供了比粗粒度关系摘要更大的增量；验证集最好的混合方法并不在每种外推上最好。因此保留全部候选成绩，不根据测试结果另换“主冠军”。尤其关系摘要换表述后接近零交互误差1，不能把同模板实体推广写成抽象语义推广；它的失败也不证明完整关系图或上游状态不含相关信息。

### 4. 追加诊断与测量修正

**D1：区分单条件信息与乘积项。** 在看到4B主结果后加入线性源状态对照[h00,δS,δN]、仅沿用h01／h10，以及族内按世界同步置乱训练目标的对照；不改原主分析。源状态线性回归已经取得主要收益，二次／乘积项在原实体测试有小增益、跨表述反而下降。它不能支持“发现了普适非线性交互单元”。14B诊断说明在正式模型结果前已固定；陈述句对照按相同诊断执行。
'''
    for side in ('4B','14B'):
        ab=read(OUT/side/'source_ablation_summary.json')
        text+=f'\n{side}：'
        for s,label in LABELS.items():
            d=ab['splits'][s]
            delta=d['product_minus_linear']
            text+=f'{label}线性 {d["methods"]["source_linear"]:.4f}，乘积−线性 {delta["mean"]:+.4f}（95%世界区间 {delta["ci95"][0]:+.4f}, {delta["ci95"][1]:+.4f}）；'
        text+='\n'
    text+=r'''
**D2：最终归一化对照。** 额外用已知架构公式预测RMSNorm(r10+r01−r00)，输入仅为三个raw残差和γ，不用r11。4B各测试交互误差约0.95–0.99，明显弱于源状态回归。故本批交互不能全部归因于末端RMSNorm。该对照使用FP64解析归一化，保存了与真实BF16输出的参考差异；它是已知公式基线，不是新提取理论。

**D3：否定问句评分存在语用歧义。** 原材料“Is it not true that…”一类问句的yes/no回答，未必服从本实验预设的命题否定标签。原4B肯定条件多数正确，否定条件明显较差；因此不能以原标签准确率声称已识别正确否定推理。原始结果不删除、不按模型答案改标签。另冻结明确陈述句对照：`Statement: It is [really/not] the case that ...`，并指示只有**整句**为真才回答yes，否则no。使用同一352世界、相同划分和四条件，在4B再采集3328次，重新训练／验证既定算法；这是同世界控制，不是独立材料复现。
'''
    text+='\n**明确陈述句对照4B，验证集选择 `'+control['selected_method']+'`：**\n\n'+table(control)+'\n'
    text+='\n首token准确率（原问句 → 陈述句，非自由生成完整正确率）：\n\n| 范围 | 原问句 | 陈述句 |\n| --- | ---: | ---: |\n'
    for s,label in LABELS.items():
        text+=f'| {label} | {summaries["4B"]["first_token_behavior"][s]["exact_accuracy"]:.1%} | {control["first_token_behavior"][s]["exact_accuracy"]:.1%} |\n'
    text+=r'''
陈述句对照五种外推的首token准确率为75.8%–86.5%，比原问句提高，但仍非全部正确；832组中仅482组的四条件首token全部正确。交互预测的全部总体统计保留了这些失败。该对照验证选定方法相对固定族均值的误差下降约0.188–0.234，五项世界重采样95%区间均在零以下；区间仅适用于固定构式下的材料，不是语义普遍性证明。组合变化E_H约0.228–0.245，不能把这个百分比称为正确生成率。完整序列、首次分叉、停止行为及自身历史没有在本阶段测量。

### 5. 对参考资料一的逐项审查

**①“头是纯线性，内部没有任务相关非线性”——把协议限定删除了。** Phase3072明确只改V，Q/K及同层注意力权重不变；其谱系相对误差0.00173658支持固定路由下的线性搬运。自然输入中，注意力整体为
$$Y(X)=A(X)V(X)W_O,\quad A(X)=\operatorname{softmax}(Q(X)K(X)^T/\sqrt{d_h}+M).$$
$$\Delta Y=[A_0\Delta V+\Delta A\,V_0+\Delta A\,\Delta V]W_O.$$
3072协议只留下第一项；不能因此否定后两项。Qwen3本地实现还包含Q/K RMSNorm与RoPE。该数学形式见[Transformer原论文§3.2](https://arxiv.org/html/1706.03762v7)，也可直接核对本地`modeling_qwen3.py`。本轮toy例子的整头加性误差0.94795，而固定A线性误差约6.25e−17；toy是数学反例，不能冒充真实语言实验。

**②“头只学语义族、不学具体词，能处理一切能力”——证据不足。** 3072只做少数头在24对旧材料上的W_U直读Top10，未含最终norm。h1列表同时含theatre变体与Twe、Sche、Trom、Sle、Jeg，不能称“完美语义族投影”。相关词方向是可保留的描述性线索，但缺少词频、分词／拼写、随机方向、独立族纯度和任务覆盖控制。单头可处理一切任务的前提也未经证明。

**③“写读绝对解耦，头不放大，全部放大由γ完成”——错误。** 写入和最终词表读出是不同操作，不等于因果独立。W_V、W_O的尺度／方向、注意力路由、下游MLP以及归一化共同决定效果。注意力本身还读取Q/K/V。3055的γ打乱使特定替换协议的对齐下降，是局部因果证据；3059的16坐标读预像能量0.5184与写方向能量0.1321，是集中程度不同，不能推导“绝对解耦”或“独立训练”。

**④“γ就是反方差白化”——应改为各向异性的对角重标度。**
$$R(x)=D_\gamma x/r(x),\quad r(x)=\sqrt{\|x\|^2/d+\epsilon},$$
$$J_R(x)=D_\gamma\left(I/r-xx^T/(d r^3)\right).$$
共享分母产生坐标耦合；γ不是全部动态增益。完整白化要求变换后协方差为单位阵（或声明的单位比例），对角重标度通常不能去除非零交叉相关。原3057的反比拟合指数约3.3076、R²约0.4158，也不是精确逆标准差定律。参见[RMSNorm原论文§4](https://arxiv.org/html/1910.07467v1)。
'''
    text+=f'\n本轮读取实际4B检查点、以词表行为样本重算：γ与读出列标准差相关 {norms["gamma_std_correlation"]:.6f}；各列标准差变异系数从 {norms["column_std_cv_before"]:.6f} 变为 {norms["column_std_cv_after"]:.6f}，并未变平；抽样128列的最大绝对相关 {norms["sampled128_max_abs_correlation"]:.6f}，非零对角重标度不消除其绝对相关。这只涉及unembedding列分布，不是对激活场做白化测试。\n'
    text+=r'''
**⑤“3079–3080证明TT与头子空间频率共振”——测量对象都被换了。** 原量是两个语境的输出logit变化向量之间的cos(TT_f,TT_g)，不是TT对某个头固有子空间的夹角。没有测频率、振荡、共振方程或能量共振；“音叉”至多是比喻。相近方向cos接近1，不是“cos极小”。TT来自两个实际输出，若在生成前使用它预测对应结果，还需审查oracle循环。

**⑥“角度规律已独立确认且普适”——遗漏反例。** 3080复用了3079已经看过的冻结数据，不能称新的独立确认；3081独立模型仅1/6相关检验显著，最小相关为−0.267，未复现主门。早先2750／2751还修正了并列秩、相关检验合并及模型层面伪重复问题。现阶段保留4B有限范式上的相关性假说，不把相关称作因果装配规则。

**⑦“高维存在无限无干扰正交任务空间”——数学表述错误。** d维空间最多容纳d个两两正交的非零向量；单头写入矩阵的秩不超过head_dim（本地4B为128）。大量近似正交方向可以存在，但容量、误差和干扰必须定量限定。有限程序可通过规则实现开放组合，这与在有限维空间中储存无限独立、无干扰任务码不是同一命题。

**⑧“10^32组合证明覆盖一切语言、已彻底破解”——没有证明链。** 十种模式如何识别、能否独立调用、组合是否可达、是否对应不同能力、组合后是否正确，均未验证。非线性不妨碍复用，线性也不保证无干扰。三个局部观察不能推出无限智能、AGI或完整语言理论。

**⑨“作用不由静态权重决定，也不在上游表示中”——排除过强。** 固定模型的响应由参数、输入及完整历史／位置共同决定。参数单独不足以指定当前响应，但不意味着参数不参与决定；有限线性探针失败也不证明信息不在上游状态中。更准确说法是“共享参数的有效作用随输入条件改变”。

### 6. 核心可信拼图、RDC接口和实际理论增量

| 拼图／来源Phase | 保留什么 | 修正或尚缺什么 |
| --- | --- | --- |
| 2750–2751／GPT3100 | 正确SwiGLU乘积导数、RMSNorm导数、能量交叉项和同基底头比较，已复算 | 旧错误导数、无交叉项能量比例及统计夸大不再作为理论支持 |
| GPT3055–3059 | γ通道分配对特定干预读出有因果作用，读写方向集中度可不同 | 不支持严格白化、绝对独立、全部放大定位在γ |
| GPT3072 | 固定Q/K和注意力时V读取线性，数值谱系成立 | 自然注意力整体非线性；描述性Top词不足以认定语义模块 |
| GPT3079–3081 | 小范式中方向相似与干预谱相似有关联；存在独立模型反例 | 无普适路由律、频率共振或TT—头子空间测量 |
| 2751组合检验 | 部分同族组合规律可复用，固定均值跨表述失效 | 需要条件依赖交互；当时旧实体泄漏／抽样单位已有追加修正 |
| 本Phase2752 | 完整坐标上的三状态交互回归可在限定新世界／组合上评估；关系摘要、表面特征和源状态分账 | 输入仍含三个原模型源状态；语义概括、因果路径及自回归闭合未完成 |
| 本Phase2752诊断 | 线性源状态、norm-only、明确陈述句等对照定位收益与评分问题 | 后验诊断不充当独立确认；控制共享世界／模板生成器 |
| 既有2745–2749接口与历史42公式／46拼图 | 保留已有索引与原证据等级 | 本轮未逐一重新认证，不用新增前向数量自动升级全部历史结果 |

RDC原主接口沿用，**没有新增普遍闭合公式或新数学定理**。外部描述为𝓛(p)，观测场为𝓧_ℓ(p)，历史符号𝐖_ℓ专指类型化析因响应束，不随意改成任意H：
$$\mathcal T^D_{\ell,\tau}:(\mathcal L(p),\mathbf W_\ell(p),\mathcal X_\ell(p))\rightharpoonup\mathbf W_{\ell+1}(p).$$
$$\operatorname{Atlas}_D=(G_{external},G_{internal},E^D_{association}).$$
内部节点携带(model,run,prefix,step,layer,position,coordinate/unit/parameter,boundary)；关联边携带(condition,algorithm,inputscope,fit,heldoutprediction,evidence,status)。本轮只实例化有限域的交互预测关联边，未证明上述跨层部分映射普遍存在。

可精确核对的统一计算底座仍是已知架构，而非新发现：
$$r_\ell=H_\ell+A_\ell(N_\ell(H_\ell);KV_\ell,position,mask),\quad x_\ell=N'_\ell(r_\ell),$$
$$H_{\ell+1}=r_\ell+W_{d,\ell}[\operatorname{SiLU}(W_{g,\ell}x_\ell)\odot W_{u,\ell}x_\ell],\quad
p_{t+1}=\operatorname{softmax}(W_U N_f(H_L)_t).$$
$$J_{MLP}(x)=W_d[\operatorname{diag}(u\odot\operatorname{SiLU}'(g))W_g+\operatorname{diag}(\operatorname{SiLU}(g))W_u],\quad g=W_gx,\ u=W_ux.$$
状态须包含所需前缀、KV及位置，不能把一个末位h当作完整自回归状态；逐层展开这些定义不等于破解语言。

三图谱本轮增量：外部图新增352个可追踪的类型化关系世界与隔离划分；内部图新增原生坐标响应、边界锚及行为失败；关联图新增可执行预测算法、允许输入、验证选择与各外推失败。没有定位新增单神经元或标量参数的语义因果机制，不能把回归坐标权重混称为模型真实连接。

### 7. 硬伤、第一性问题和下一大阶段

当前最大的缺口是：**一个算法可能预测模型如何响应，却没有解释模型为什么正确理解某个关系。** 本轮明确保留行为错误及否定语用问题；看起来可预测的交互也可能包含格式、词形、归一化和共同背景。

其他限制：固定英语合成语法、已知四族、少量表述构式、只考查语气×否定；14B源组较少且训练表述与真值覆盖不同；仅末位而非全token场；BF16无整网FP32敏感性复刻；未把改进的状态误差接到完整下一token分布及自由生成；没有独立多语种／多关系族检验；关系摘要省略了节点完整内容，不能以摘要失败排除语言图结构；回归包含大量坐标系数，尚未压缩成可解释共享操作。world bootstrap不消除这些限制。

下一阶段应围绕“从可用早态和类型化关系，提取可接续的共享操作”组织，而不是继续换名称或重复同模板：

1. **语义与材料门。** 先以明确陈述／角色绑定任务验证行为、答案token、首次分叉和完整答案；扩大独立构式、平衡真值／角色／深度／实体，冻结发现、选择、确认三批材料。
2. **减少源状态依赖。** 比较节点embedding＋类型化边的组合算法、早层全坐标场条件预测、简单词序／位置／线性基线；预测时禁止读取待预测层的单条件状态与输出TT。分别检验实体、表述、边组合和深度，不能用随机切分代替结构外推。
3. **由预测追到真实计算。** 在成功且行为正确的子域拆开Q/K路由、V内容、OV写入、MLP交互和norm分母；保存原生坐标、单元、参数来源，针对明确候选做自然前向、删除／救援与替代解释对照。局部干预既不证明唯一机制，也不以单坐标切换答案作为唯一成功标准。
4. **跨层和生成闭合。** 把有效规则推广到真实已消费前缀及完整KV状态，区分teacher forcing和自由生成；检验输出分布、首次分叉、后续内容和停止。只有多轴独立推广、优于简单基线且因果边界明确后，才升级为理论规律。

通俗说：现在已有一把可信的量尺和一个能预测部分组合变化的工具，也知道了哪些收益只是简单状态信息。还没有找到支配全部语言能力的通用规则。参考资料一把“在固定条件下可复用的一段计算”写成“整个智能的终极密码”，跨越了实验尚未支持的多个环节。
'''
    text+='\n### 8. 实例、产物、时间与恢复入口\n\n'
    mat=read(OUT/'assertion_control/material.json')
    bygroup={r['group']:r for r in mat['rows'] if r['cond']==3}
    groups=read(OUT/'assertion_control/4B/prediction_groups.json')
    with np.load(OUT/'assertion_control/4B/prediction_metrics.npz') as z:
        err=z[control['selected_method']+'_interaction'][:,36]
    test=[i for i,g in enumerate(groups) if g['split'] not in ('train','validation')]
    chosen=[min(test,key=lambda i:err[i]),max(test,key=lambda i:err[i])]
    observed={r['id']:r for p in (OUT/'assertion_control/4B').glob('chunk_*.json') for r in read(p)}
    cases=[]
    for tag,i in zip(('较好预测实例','失败实例'),chosen):
        row=bygroup[groups[i]['group']]
        pred=observed[row['id']]
        cases.append(dict(kind=tag,group=row['group'],split=row['split'],text=row['text'],expected=row['expected'],first_token=pred['prediction_text'],interaction_error=float(err[i])))
        text+=f'{tag}（按控制测试误差选出的极端实例，不代替总体统计）：`{row["group"]}`，范围`{row["split"]}`，交互误差{err[i]:.4f}，目标首词`{row["expected"]}`，实际首token`{pred["prediction_text"]}`。\n\n```text\n{row["text"]}\n```\n\n'
    write(OUT/'representative_cases.json',dict(created_utc=now(),selection='Smallest/largest validation-selected method error on assertion4B test groups; extremes explicitly labeled',cases=cases))
    for label,directory in [('4B',OUT/'4B'),('14B',OUT/'14B'),('陈述句4B',OUT/'assertion_control/4B')]:
        done=read(directory/'capture_done.json')
        text+=f'- {label}正式采集：{done["count"]}条、{done["elapsed"]:.1f}秒（采集进程口径，含其加载；不等于整个任务耗时）。\n'
    text+='\n执行材料、design、pre_capture_seal、predictor_pre_fit_seal、执行代码内容快照、chunk数组／输入身份、验证集选择、逐世界／逐层指标、原始与row-RMS全坐标图、质量核查、参考资料命题账本及数学反例均保存在 `tests/glm5/result/rdc_context_interaction_20260923/`。所有原始场保留，无清理。初版仅校准16条后发现训练深度恒定，在正式采集前改为深度1／2并封存旧设计于`pre_main_design_v1/`；未把初版16条并入正式样本。\n\n'
    text+='核心入口为`phase2752_context_interaction.py`、`phase2752_predict_interaction.py`、`phase2752_assertion_control.py`、`phase2752_source_ablation.py`、`phase2752_norm_control.py`、`phase2752_reference_audit.py`、`phase2752_quality_and_plots.py`。只读接口`/api/rdc-construction/context-interaction`提供研究索引、审查命题、模型指标、完整原生坐标行及世界材料；接口契约测试通过不表示已重启用户服务器。\n'
    return text


def complete(append=False):
    for path in (OUT/'4B',OUT/'14B',OUT/'assertion_control/4B'):
        for name in ('capture_done.json','prediction_summary.json','source_ablation_summary.json','quality_audit.json'):
            assert (path/name).exists(),str(path/name)
    external_graphs()
    body=report()
    (OUT/'phase_report.md').write_text(body,encoding='utf-8')
    index(True)
    if append:
        memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
        gpt=ROOT/'research/gpt5/docs/AGI_GPT5_MEMO.md'
        before=memo.read_bytes()
        gbefore=gpt.read_bytes()
        assert not re.search(r'^## Phase 2752:',before.decode('utf-8-sig'),re.M)
        assert max(map(int,re.findall(r'^## Phase (\d+):',before.decode('utf-8-sig'),re.M)))==2751
        assert max(map(int,re.findall(r'^## Phase (\d+):',gbefore.decode('utf-8-sig'),re.M)))==3101
        stamp=datetime.now().strftime('%Y-%m-%d %H:%M')
        with memo.open('ab') as f:
            f.write((f'\n\n## Phase 2752: 条件交互的事先预测、语义评分对照与万能头主张审查 [{stamp}]\n\n'+body).encode('utf-8'))
        note=f'\n\n## Phase 3102: 条件交互预测与“万能线性头”说明的证据边界修正 [{stamp}]\n\n'
        note+='本次完整实验与审查追加在`research/glm5/docs/AGI_GLM5_MEMO.md`的Phase2752，产物为`tests/glm5/result/rdc_context_interaction_20260923/phase_report.md`。原记录保留，不把修正插入历史中间。\n\n'
        note+='4B3328条＋14B400条主采集，以及明确陈述句4B3328条控制，比较关系／角色摘要、长度位置、单条件状态和组合项。具有条件输入边界的预测器支持局部可预测性；它需要三次源前向，不等于完整语言机制或AGI。否定问句评分有语用歧义，控制结果与主结果分开报告。\n\n'
        note+='针对参考资料一：3072只证明固定Q/K与注意力的V线性读取；3079–3080的cos比较两个输出变化方向，不是头子空间或频率共振；3081未独立复现普适角度规则；γ通道因果作用不支持写读绝对解耦或全部增益由γ产生。实际检查点列标准差变异系数经γ缩放由0.03743增至0.11836，不能称白化。有限秩不允许无限无干扰正交任务空间，10^32组合计数不构成通用语言能力证明。因此撤回这些段落被扩张为“彻底破解万能复用”的解读，保留有明确条件的局部观察。没有新增普适RDC定理。\n'
        with gpt.open('ab') as f:
            f.write(note.encode('utf-8'))
        assert memo.read_bytes()[:len(before)]==before and gpt.read_bytes()[:len(gbefore)]==gbefore
        receipt=dict(created_utc=now(),phase=2752,gpt_index_phase=3102,
            memo_prefix_bytes=len(before),memo_prefix_sha256=hashlib.sha256(before).hexdigest(),memo_line=before.count(b'\n')+3,
            gpt_prefix_bytes=len(gbefore),gpt_prefix_sha256=hashlib.sha256(gbefore).hexdigest(),gpt_line=gbefore.count(b'\n')+3,
            report_sha256=sha(OUT/'phase_report.md'),append_only_verified=True)
        write(OUT/'completion_receipt.json',receipt)
        print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['index','report','append'])
    a=p.parse_args()
    if a.mode=='index':
        external_graphs();index()
    else:
        complete(a.mode=='append')
