"""Consolidate approximate regularities and append one substantial Phase."""
import hashlib,json,re,subprocess,sys
from datetime import datetime
from pathlib import Path
import numpy as np
from phase2754_relation_stability import ROOT,OUT,write,sha,now,snapshot

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def pct(x):return f'{100*x:.2f}%'
def interval(x):return '['+', '.join(f'{v:.4f}' for v in x)+']'
LABELS={'entity':'新实体','surface':'新表达格式','depth':'更深关系'}

def build():
    p=read('probe_confirmation.json');a=read('4B/regularity_summary.json');b=read('14B/regularity_summary.json');m=read('mechanism_summary.json')
    challenge=read('fact_order_challenge/summary.json');profile=read('source_profile_diagnostic.json');eq=read('equivariance_diagnostic.json');compare=read('matched_model_comparison.json')
    gen=read('parsed_generation_diagnostic.json');quality=read('quality_audit.json');stamp=datetime.now().strftime('%Y-%m-%d %H:%M')
    text=f'## Phase 2754: 粗糙实现中的稳定关系信号、角色位置编码与真实计算来源 [{stamp}]\n\n'
    text+=r'''### 1. 研究重心与本轮结论

**已完成本轮有界研究；没有声称完整编码理论或AGI已经解决。** 按用户最新要求，把稳定的近似规律作为核心，不再把“所有题答对”“状态完全相等”“一次单方向删除就翻转答案”作为保留线索的必要条件。小模型实现可能粗糙，是需要控制和检验的因素；不能因此预先断言所有失败都来自模型太小。

本轮保留了三个具体线索：

1. **端点角色关系的可复用读出。** 在已知、无分叉的有向链材料中，同一个冻结解码器从第16层末位及两个查询角色位置读出基础谓词方向，跨新实体、新表达、深度3和新的事实倒序材料均保持正确符号；连续幅度并不完全不变。这是有范围的角色关系编码线索。
2. **关系信号与输出是否答对可以分开。** 模型总体有较稳定的真／假相对响应，即使否定语境、偏置或输出格式造成错误；14B匹配样本也保留正向响应。不能只筛选答对题，也不能把错误输出全部当作无信息。
3. **自然计算来源的形状比幅值稳定。** 沿真实attention、MLP、残差舍入和最终读出逐项对账，4B跨测试轴的全层来源形状相似，但幅值仍变化。该图谱可定位后续机制研究，不等于证明某条投影就是网络唯一使用的算法。

重要反证同时保留：抹去第16层所学探针的可读方向，探针分数显著消失，而原模型输出几乎不变。因此“稳定可读编码”还没有升级为“已识别必要因果电路”。

### 2. 材料、输入合同与证据顺序

主实验4B共320个世界、5120条输入：4类严格有向关系（严格子类别、钥匙传递链、左右次序、严格包含）×世界×2表述×8条件。训练96世界1536条，验证32世界512条；正式确认192世界3072条，三个轴各64世界。世界内两个表述与8个条件都是重复测量，不能当作16个独立世界。

三个因素为事实链方向θ∈{−1,+1}、查询源／目标角色绑定ρ∈{−1,+1}、肯定／否定η∈{−1,+1}。基础谓词真值r=θρ，整句真值y=θρη。所有8格真／假精确平衡。正负样本来自同一世界，固定η时token多重集合完全一致；4B和14B分词器均逐组检查通过。标签另用已知有向图可达性／传递链始末角色检查，未从模型答案倒推。

独立后续对照：主确认后新建64个世界、1024条4B输入，深度均为3，保持同一图、同一查询、同一词袋、同一答案，只将事实呈现顺序倒转。钥匙事件的数值序号保留，故没有改变事件发生语义。此对照使用已冻结的原解码器，不重训。共计384个不同世界，其中真正用于确认的世界为192+64=256；14B和干预没有再增加独立世界。

14B有限复核：从主确认中预定24世界192条、每世界一表述，与4B逐输入匹配。使用本地Qwen3-4B、Qwen3-14B，顺序加载；BF16、eager attention、batch1、无截断／padding，主采集无cache。14B受16GB显存约束使用10GiB GPU上限及CPU卸载；未量化，未训练模型权重。本轮没有GLM4复核，不能把结果写成所有模型通用。

材料→采集发现集→训练／验证选定探针→正式确认。`material_seal.json`、`probe_algorithm_seal.json`、`selection.json`、代码内容快照和拟合hash记录这个顺序。机制候选的两个MLP层仅按训练集正向贡献选择；干预协议在查看主确认指标前冻结。事实顺序对照是主确认后的新设计，不能倒写为最初就预注册的实验。

采集全体原生坐标：4B为38个边界×2560坐标，14B为42个边界×5120坐标；包含最终归一化及额外原始末层残差。每块attention、MLP的末位实际写入也完整保存；第4/8/12/16层另存两个查询角色末subtoken位置。BF16按uint16位模式无损保存，读取时左移并转FP32；既不是量化，也不是全token／全KV采集。多token实体的语义角色位置来自材料注释，不把一个token自动当成一个概念。

**预测输入限制。** 主解码器只用已知查询角色位置上的第16层状态及末位状态，不读未来层、模型最终答案、当前真值标签或θ／ρ标签。训练可以用已知真值作监督。角色位置是外部注释，尚无自动语义解析器。对照包括只看末位、静态表面条件和最终层描述性解码器；最终层结果不能冒充早层预测。

### 3. 数学对象与提取算法

对完整八格输入，任一状态或标量响应F都可按析因基底展开：
$$
C_S=\frac{1}{8}\sum_{\theta,\rho,\eta}\chi_S(\theta,\rho,\eta)F(\theta,\rho,\eta),
\qquad
\chi_S\in\{1,\theta,\rho,\eta,\theta\rho,\theta\eta,\rho\eta,\theta\rho\eta\}.
$$
这是定义和可核对恒等式，不是“发现了语言服从某个新定律”。r项表示关系方向与查询绑定的耦合，y项表示再与否定条件结合的耦合。

令m=z_yes−z_no为指定单token“ yes”与“ no”的logit差，d=C_{θρη}(m)。d>0表示整个八格的平均响应沿真值方向；它不保证每个条件都答对。另固定ρ、η，改变θ，比较真命题和假命题的m：正确排序率能检测相对关系响应，较少受共同yes偏置影响。canonical差、yes/no多种单token写法的总概率质量、实际首token、生成前缀与停止分开记录。

冻结的角色解码器使用所有原生坐标，无PCA或Top-K：
$$
z_\ell=\left[
\frac{h_{\ell,\mathrm{end}}-\mu_0}{s_0},
\frac{h_{\ell,\mathrm{src}}-h_{\ell,\mathrm{dst}}-\mu_1}{s_1},
\frac{h_{\ell,\mathrm{src}}+h_{\ell,\mathrm{dst}}-\mu_2}{s_2}
\right],
\qquad
(\widehat r,\widehat y,\widehat m)=z_\ell B+b.
$$
μ、s只由训练集确定；每块s是一个RMS欧氏长度。三列分别监督基础谓词、整句真值及模型真实最终margin；分类取符号，margin保留连续值。岭参数在{0.001,0.01,0.1,1,10}中分别用验证准确率／MSE选择，不在测试上重选。候选层为4、8、12、16，另设最终层描述性上限。验证选择slots16；验证整句准确率94.92%，相同层只看末位为74.41%。这不是原模型自带的分类头，而是从其状态提取信息的外部算法。

### 4. 主要独立结果：保留稳定部分，记录失效范围

下表均为4B主确认，基础谓词是不含否定操作的链式关系r。整句列包含否定，实际模型列为原始输入协议下的首token判定。每轴64世界，区间按世界、族内分层bootstrap2000次；不包含重新训练或模板总体的不确定性。

| 确认轴 | 基础谓词解码 | 第16层角色整句解码 | 其95%世界区间 | 原模型首token正确率 | 真／假相对排序正确率 |
| --- | ---: | ---: | --- | ---: | ---: |
'''
    for s,label in LABELS.items():
        pr=p['splits'][s]['slots16'];rg=a['splits'][s]
        text+=f"| {label} | {pct(pr['predicate_accuracy'])} | {pct(pr['truth_accuracy'])} | {interval(pr['truth_world_ci95'])} | {pct(rg['native_accuracy']['mean'])} | {pct(rg['paired_ranking']['mean'])} |\n"
    text+=r'''
所有192个主确认世界的两个表述都有d>0；这是384组上的实际结果，不能外推为自然语言中的100%规律。全成功时bootstrap会退化为[1,1]；另报告每轴64世界“两个模板均正向”的描述性Wilson区间下界约0.943，仍不能覆盖未知模板总体。

基础谓词信号在事实倒序对照中仍为100%符号正确，但整句真值解码由93.75%降为68.16%，变化−25.59个百分点，95%世界区间约[−29.10,−22.07]个百分点；原模型首token则由75.59%到74.80%，差异区间跨0。由此保留的是**简单有向链的端点角色对齐信号**，不能把它扩张为任意图上的推理编码。

连续信号并不完美。对冻结谓词分数做事后析因分析，主确认的关系增益约0.93–0.98，非关系分量与关系增益之比约0.096–0.141。事实倒序后增益由0.976降至0.452，杂项比升至0.213，但符号仍正确。因此“幅度变化”与“关系结构消失”应分别判断。交换事实方向或查询方向对应分数近似反号，同时交换对应近似保持；这是抽取分数的经验等变性，不是整个HiddenState已经证明服从某个群作用，也不是新数学定理。

在深度3材料，平均肯定关系增益为5.171、否定关系增益0.838；相对方向尚存，否定响应却明显减弱。这支持把关系表示和条件运算／读出分开追踪，不能简单用总体答错率否定基础表示。

**事后组合候选。** 保持已冻结的基础谓词解码器，再用文字中明确的否定标记η执行ŷ=sign(r̂)η，在当前主确认及顺序对照上均正确。该组合是在看过结果后提出，尚未完成它自己的独立确认；逻辑否定规则由外部提供，不是已破解模型内部的否定算法。保存为`factorized_readout_diagnostic.json`，作为下一阶段的可执行基线。

### 5. 模型规模、粗糙实现与输出协议

14B匹配样本在三个轴的真／假相对排序率均为93.75%，24世界与4B的d符号全一致；支持该响应不只出现在一个4B检查点。14B相对4B的匹配排序提升为4.17个百分点，95%区间[−2.08,10.42]个百分点；yes/no总概率质量判定提升4.69个百分点，区间[−1.04,9.90]个百分点。这个样本不能确认“规模扩大改善了规律”，更不能外推出无限大模型会完美。

14B直接首token分数只有28%–38%，但不是对应的语义判断能力只有这个水平。它经常先输出格式符号、空行、“Let”等开头；其yes/no总概率质量判定各轴约84%–91%。主实验是原始续写协议，不是完整的标准聊天能力评测。两个检查点训练、规模、运行映射和呈现方式不同，无法把所有差异归结为参数规模。

为定位协议影响，补做固定匹配子集的greedy续写：4B192条最多8个新token；14B24条最多4个新token，再分别以原文和本地chat template（enable_thinking=False）运行，内容不变。不同样本规模与截断长度不允许作完整生成能力比较。原始严格首词／完整短答分数保留；另用与标签无关的开头解析器识别显式Answer:、普通markup或boxed答案，不从解释中挑选有利词。未出现答案的短前缀属于未决／截断，不能认定最终答案错误。

| 诊断协议 | 条数 | 前缀中识别到答案的比例 | 全部条目中已观察到正确答案的比例 | 已识别子集正确率* |
| --- | ---: | ---: | ---: | ---: |
'''
    for key,label in [('generation','4B raw≤8token'),('generation_chat','4B chat≤8token'),('generation14B','14B raw≤4token'),('generation14B_chat','14B chat≤4token')]:
        v=gen['results'][key]
        text+=f"| {label} | {v['prompts']} | {pct(v['coverage'])} | {pct(v['correct_observed'])} | {pct(v['accuracy_among_recognized']) if v['accuracy_among_recognized'] is not None else '未识别'} |\n"
    text+=r'''
*已识别子集有选择偏差，不能作为整体能力估计。4B raw常在yes/no之后继续解释，8token内没有EOS；原始指令没有严格禁止解释，所以不能把“未止于一个词”直接写成理解失败。所有token、实际输入、前缀、EOS／截断记录可回查。完整解释及最终停止尚未测完；本轮主要结论来自冻结首步状态／相对响应，而非未经完成的长生成。

因此，用户提出的小模型粗糙性需要保留，但至少应拆成：表示本身、条件合成、输出选择／偏置、输入协议与数值测量。当前证据支持前几项之间存在差异，不支持把它们全部统一解释为“小模型参数不够”。

### 6. 从稳定现象追到实际计算

取最终RMSNorm参数γ及真实unembedding行，固定读出方向v=γ⊙(W_yes−W_no)。设r_L为最终原始残差，s_L为其RMS分母，则理想高精度投影满足m=vᵀr_L/s_L。真实BF16运行另保留最终norm及logit舍入差。逐层残差写入给出可对账来源：
$$
m=\frac{v^\top h_0+\sum_\ell v^\top A_\ell+\sum_\ell v^\top M_\ell+
\sum_\ell v^\top\epsilon_\ell}{s_L}
+\epsilon_{\mathrm{norm}}+\epsilon_{\mathrm{readout}}.
$$
ε_ℓ是BF16残差相加的实际舍入差，不被忽略。该式是原生计算的归因恒等式；除以每条输入的最终s_L用了未来量，所以它是事后来源分析，绝不能当作早层预测器输入。

在4B保留全部36层attention和36层MLP的真值条件贡献，与训练族均值比较，事后诊断的平均余弦相似度：
'''
    text+='、'.join(f"{LABELS[s]} {profile['splits'][s]['cos']['mean']:.4f}" for s in LABELS)+'；对应相对幅值误差分别为'+ '、'.join(f"{profile['splits'][s]['relative_error']['mean']:.4f}" for s in LABELS)+'。全部层原序保留，没有用Top-K或降维图定义主干。这个形状稳定性值得续研，但它是看过主结果后的描述性指标，不能冒充一项新预注册预测成功。共同yes/no读出也可能促成共同形状。\n\n'
    text+=r'''4B后段attention沿真值方向的净直接贡献为正，MLP存在明显的正负抵消。MLP净投影为负不等于MLP损害推理：后续attention本身已依赖前面MLP改写的状态，这种投影分账不是隔离每个模块的全部因果作用。

训练集选择了第30、34块（零基索引29、33）的正向MLP来源，另在预定24世界192条中采集其全部9728个SwiGLU单元。对真实down-projection输入u：
$$
v^\top M_\ell\approx\sum_{j=0}^{9727}c_{\ell j}u_{\ell j},
\qquad
c_{\ell j}=v^\top W_{\mathrm{down},\ell}[:,j].
$$
保存全部u、c与原生投影误差，而不是只展示几个大单元。将其除以各样本最终RMS后，第34块的最大对账差约0.00372 logit，第30块约0.00210；近似号包含原生矩阵乘法与投影精度差异。现在可查询到真实参数列、MLP单元及条件贡献，但没有证明某个单元是一个概念或是必要的语义组件。

**因果方向对照没有通过。** 从第16层冻结探针系数构造三个位置的联合向量V，p=VᵀX+b；删除δX=−pV/∥V∥²，同时以相同范数、与V正交的固定随机方向作对照。干预不读当前标签。BF16实测探针绝对分数均值从0.528降至0.00924，改变量约占联合状态范数0.45%；删除确实发生。

但八格真值响应系数均值删除前后都是2.9967，变化95%世界区间约[−0.0280,0.0247]；与随机对照的差异也跨0。局部可读方向可能不是原模型采用的方向，可能有冗余信息，也可能经其他位置重建；本轮尚未区分。不能因一次干预失败抹掉独立解码规律，也不能把高解码率当成已经定位了内部算法。

### 7. 当前可信拼图与理论接口

| 拼图 | 当前可保留内容 | 尚不能推出 |
| --- | --- | --- |
| Phase2750–2751／GPT3100测量修复 | 正确导数、能量交叉项、同基底比较和数值对账 | 原来过强的完整机制判决 |
| Phase2752–2753预测边界 | 部分组合可预测，输入范围和输出指标必须明确 | 低状态误差代表理解正确，或早层信息不存在 |
| 本轮角色位置读出 | 简单链端点绑定可由同一冻结规则跨词汇／形式／深度提取 | 任意图推理、自动角色解析、原模型必用该探针 |
| 本轮条件差异 | 基础谓词较稳，否定合成／输出呈现更易变 | 将所有差异都归因于规模，或要求不存在任何例外 |
| 本轮实际来源与单位 | 全层原生写入形状、真实W_down列和全部单元贡献 | 独立语义原子、唯一必要因果电路、严格写读解耦 |
| 本轮跨检查点 | 24匹配世界中真值响应方向一致 | 纯规模因果效应，或大模型必然完美 |
| 历史接口与其余拼图 | 保留各自已有证据等级 | 本轮未逐项重新认证，不能自动全部升级 |

RDC名称和主接口不变，本轮没有新增普遍定理：
$$
\mathcal T^D_{\ell,\tau}:
(\mathcal L(p),\mathbf W_\ell(p),\mathcal X_\ell(p))
\rightharpoonup\mathbf W_{\ell+1}(p),
\qquad
\operatorname{Atlas}_D=(G_{\mathrm{external}},G_{\mathrm{internal}},
E^D_{\mathrm{association}}).
$$
𝓛是类型化语言条件，𝓧是指定边界的观测场，𝐖仍专指类型化析因响应束，D限定模型与输入域。新关联边加入query-role positions、预测时可用信息、变换等变性、正／负例、幅值误差和因果检查状态。不能把一个末位向量当作完整自回归状态。

统一实际计算底座仍是已知结构：
$$
r_\ell=H_\ell+A_\ell(N_\ell(H_\ell);KV_\ell,\mathrm{position},\mathrm{mask}),
\qquad x_\ell=N'_\ell(r_\ell),
$$
$$
H_{\ell+1}=r_\ell+W_{d,\ell}
[\operatorname{SiLU}(W_{g,\ell}x_\ell)\odot W_{u,\ell}x_\ell],
\qquad
p_{t+1}=\operatorname{softmax}(W_U N_f(H_L)_t).
$$
本轮提取出的候选接口是“角色位置响应→基础关系符号→条件合成与读出”，而不是重新命名这个前向恒等式为完整理论。基础谓词读出和外部否定乘法可以执行，但模型内部如何完成相同合成仍需定位。

三图谱增量：外部图新增类型化八格世界、精确词袋对照及事实顺序变换；内部图连接角色位置、全层原生写入、真实参数列及MLP单元；关联图增加冻结探针、近似等变关系、读出协议差异和失败的因果方向检查。原“万能线性头、无限无干扰正交、γ严格白化、彻底破解复用”仍没有获得支持。

### 8. 第一性洞察、硬伤与下一整体阶段

**应寻找对关系变换稳定的计算关系，而不是要求每个数值、每道题或每个模型都完美。** 本轮最有用的对象不是单个高亮坐标，而是角色绑定改变时，多个位置联合响应如何变号、保持或被条件调制。稳定的符号／变换关系可以和幅值变化共存；它适合成为后续编码研究的约束。

必须正视的硬伤：材料都是结构完整、无分叉的简单链，端点角色匹配可能绕过中间关系推理。倒序排除了一个表面呈现捷径，却未排除编号提示、端点检索或图生成器特性。当前真值监督、人工角色注释和固定yes/no读出都提供了强先验。16层已经做了大量原模型计算；本轮不是从embedding直接恢复全部算法。14B样本24世界，GLM4尚未运行；长生成的正确性和停止仍未闭合。所有“100%”只针对已测固定范围。

下一大阶段是“关系变换算子的图推广与因果接续”，计划而未执行：

1. **破除链端点捷径。** 冻结新材料，加入中间断边、同端点异路径、分支、无关边与不同事实顺序；明确可达性／真／假／未知的语义，保持词袋和位置对照。先检查基础关系解码究竟跟随真实路径还是端点匹配。
2. **提取条件运算。** 固定已发现的关系读出，比较外部显式逻辑基线与从上下文状态学习的否定、双重否定、角色交换等运算；使用新组合独立确认。不把后验组合100%当作模型内部否定已破解。
3. **追真实路径与冗余。** 围绕通过图推广的子域，增加相关事实／查询位置的跨层记录，区分被删除信息经何处重建；针对真实QK、V与MLP参数路径设计匹配控制。保留全坐标／全单元背景，不以单坐标切换答案作为唯一成功标准。
4. **规模与协议分账。** 在相同图、相同评分和经核验的输入协议下顺序比较4B、14B、GLM4，再扩大独立关系族与生成步。结构稳定性、幅值、行为正确性和格式／停止分别报告，不假设规模必然单调改善。

这些任务服务同一个问题：哪些相对关系被共享参数稳定编码，条件怎样改变其作用，计算如何跨层和生成接续。当前已有可执行、可复核的局部入口，尚没有完整语言编码定律。

### 9. 测量核查、产物、失败记录与恢复

日志连续性异常单独登记：追加前检查发现当前GPT日志末尾为3100，旧Phase2752／2753交付回执曾记录追加到3102／3103；原因未确认。GLM主记录2750–2753仍在。本轮保留当前GPT文件全部字节，不猜测恢复可能被其他操作移除的文字，以3104继续已使用的编号，并指向完整GLM研究。此前2754追加尝试在两文件预检阶段停止，未发生部分追加。该差异及当前文件hash存于memo_continuity_observation.json，不把旧回执当作当前文件仍含该内容的证明。

4B主采集5120条、14B192条和4B顺序对照1024条合计6336次正式首步前向；不同模型的相同输入、pilot、干预及生成不被重复计算为新世界。主采集全部层残差锚为0，4B／14B各8条pilot与正式原场逐值一致；来源加和、RMS和读出舍入逐项保存。`checkpoint_environment.json`保存两个完整检查点分片及tokenizer/config/index的hash，以及Python、torch、transformers、numpy版本。

原生场、全部拟合系数、预测、词袋与图注释、数值对账、反例、原始／标准化全坐标图、全部9728单元场均保留。原值与显示归一化分开；标准化热力图±6只限制显示，原值不裁剪。坐标和单元不排序、不用PCA／Top-K定义主干。

实现问题如实保留：第一次单位投影在样本前向前因Tensor转numpy缺少detach退出，修正后记录执行版本；首次chat模板tokenize返回编码对象而非列表，改为显式渲染后编码；14B同进程第二次加载退出且无诊断traceback，原因未确认，改用独立进程重跑该未完成条件。失败前的设计／代码快照保留，没有用它们伪造成功样本，也没有据此修改模型或目标标签。

只读接口`/api/rdc-construction/relation-stability`提供模型指标、预测器、机制、顺序对照、全坐标、全部单元与世界材料；新增及相关API共12项测试通过，另开头解析器8项断言通过。存在Starlette httpx弃用警告，未影响测试。没有声称已重启服务或部署新UI。

主入口：`phase2754_relation_stability.py`、`phase2754_probes.py`、`phase2754_mechanism.py`、`phase2754_analysis.py`、`phase2754_order_challenge.py`、`phase2754_generation.py`及交付／诊断脚本。主结果目录`tests/glm5/result/rdc_relation_stability_20260923/`。已经可见的确认集不能再次作为下一算法的未见测试；重做独立验证须新运行身份及新材料。无原场清理，无历史Phase改写。
'''
    # Real example: positive predicate under negation, with an incorrect raw answer.
    material=read('material.json');src={r['id']:r for r in material['rows']}
    meta={r['id']:r for path in (OUT/'4B/confirmation').glob('chunk_*.json') for r in json.loads(path.read_text(encoding='utf-8'))}
    example=next(r for r in material['rows'] if r['split']=='depth' and r['predicate_sign']==1 and r['eta']==-1 and meta[r['id']]['prediction_text'].strip().lower()!=r['expected'])
    example_text=f"\n\n实际反例（描述性选例，未用于算法选择）：`{example['id']}`。基础链关系成立，否定后的正确答案为`{example['expected']}`，原模型首token为`{meta[example['id']]['prediction_text']}`。\n\n```text\n{example['text']}\n```\n"
    text+=example_text
    timing={name:read(name) for name in ('4B/discovery/done.json','4B/confirmation/done.json','14B/confirmation/done.json','fact_order_challenge/4B/confirmation/done.json','mechanism_done.json')}
    text+='\n实际耗时（各采集／实验进程口径，不是整轮任务耗时）：'+ '；'.join(f"{name} {info['elapsed_seconds']:.1f}秒" for name,info in timing.items())+'。\n'
    assert not [(i,ord(c)) for i,c in enumerate(text) if ord(c)<32 and c not in '\n\t'], 'Invalid report control characters'
    assert len(re.findall(r'\$\$.*?\$\$',text,re.S))==7
    (OUT/'phase_report.md').write_text(text,encoding='utf-8')
    write(OUT/'index.json',dict(created_utc=now(),phase=2754,status='completed_bounded_research',theory_status='stable_approximate_relational_signals_with_unresolved_causal_mechanism',
        primary_worlds=320,independent_followup_worlds=64,formal_forwards=6336,matched14B_worlds=24,models=['Qwen3-4B','Qwen3-14B'],
        report=str((OUT/'phase_report.md').relative_to(ROOT)),artifacts=['4B/regularity_summary.json','14B/regularity_summary.json','probe_confirmation.json','mechanism_summary.json','fact_order_challenge/summary.json','parsed_generation_diagnostic.json','equivariance_diagnostic.json'],
        api='/api/rdc-construction/relation-stability',next='Graph topology generalization, conditional operations and redundant causal paths; fresh confirmation required',
        no_claims=['No universal perfect code','No unique necessary probe circuit','No pure model-scale causal conclusion','No complete free-generation benchmark']))
    return text,stamp

def run(append=False):
    assert not (OUT/'append_receipt.json').exists(),'Delivery already sealed.'
    text,stamp=build()
    write(OUT/'delivery_verification.json',dict(created_utc=now(),source=snapshot(Path(__file__)),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        preexisting_changes_preserved=True,api_tests_passed=12,answer_parser_assertions=8,figures_visually_inspected=True,
        exact_test_command='.venv/Scripts/python.exe -X utf8 -m unittest tests/glm5/test_relation_stability_service.py tests/glm5/test_early_interaction_service.py tests/glm5/test_context_interaction_service.py tests/glm5/test_trusted_rebuild_service.py'))
    if not append:print('Report prepared, memos not appended.',flush=True);return
    current_gpt=ROOT/'research/gpt5/docs/AGI_GPT5_MEMO.md'
    current_bytes=current_gpt.read_bytes()
    old_receipt=ROOT/'tests/glm5/result/rdc_early_interaction_20260923/append_receipt.json'
    old_record=next(x for x in json.loads(old_receipt.read_text(encoding='utf-8'))['appends'] if 'gpt5' in x['path'])
    write(OUT/'memo_continuity_observation.json',dict(created_utc=now(),
        observed_current_gpt_sha256=sha(current_gpt),observed_current_bytes=len(current_bytes),
        observed_last_phase=int(re.findall(r'^## Phase (\d+):',current_bytes.decode('utf-8-sig'),re.M)[-1]),
        prior_receipt=str(old_receipt.relative_to(ROOT)),prior_receipt_sha256=sha(old_receipt),
        prior_record=old_record,cause='unknown',
        action='Preserve current bytes, append index 3104, retain complete GLM narrative; do not reconstruct or overwrite missing GPT notes.'))
    pointer=f'''\n\n## Phase 3104: 小模型粗糙实现中的稳定关系信号与机制边界 [{stamp}]

索引连续性说明：当前文件在追加前止于3100；旧交付回执记录过3101–3103相关追加，但这些段落当前缺失，原因未确认。本轮不覆盖或重建历史，保留当前前缀并继续已使用编号3104。相应完整研究见GLM主记录2751–2753；连续性差异已登记在本轮memo_continuity_observation.json。

完整记录追加至GLM Phase2754。本轮按“寻找稳定近似规律”推进，4B主实验320世界5120条，另64新世界1024条事实顺序对照；14B在24匹配世界192条复核。角色位置的冻结第16层解码器在简单有向链中稳定读出基础谓词方向，整句真值解码对新实体／新格式／深度为93.36%／73.14%／91.70%；事实倒序后基础谓词符号保持，整句解码明显下降。实际模型的真／假相对响应与最终答案、输出协议分账。

保留全层真实attention／MLP来源及第30、34块全部9728单元贡献；来源形状稳定但幅度并不完美。删除可读探针方向几乎不改变输出，尚不能把解码规律认定为必要电路。14B正向关系响应复现，但样本与协议限制不允许推出纯规模效应。后验“关系解码＋外部否定”只是可执行候选，不是已破解内部条件运算。

详细公式、统计范围、反例、模型协议诊断、完整证据与下一阶段见`tests/glm5/result/rdc_relation_stability_20260923/phase_report.md`。原万能线性头／绝对读写解耦／无限正交结论仍不成立；本轮将可信线索收紧为有界的角色关系编码与条件化计算来源。
'''
    appends=[]
    destinations=[(ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md',2753,'\n\n'+text),(ROOT/'research/gpt5/docs/AGI_GPT5_MEMO.md',3100,pointer)]
    for path,previous,_ in destinations:
        matches=re.findall(r'^## Phase (\d+):',path.read_text(encoding='utf-8-sig'),re.M)
        assert int(matches[-1])==previous, f'Unexpected phase in {path}'
    for path,previous,payload in destinations:
        before=path.read_bytes();matches=re.findall(r'^## Phase (\d+):',before.decode('utf-8-sig'),re.M);assert int(matches[-1])==previous
        encoded=payload.encode('utf-8')
        with path.open('ab') as f:f.write(encoded)
        after=path.read_bytes();assert after[:len(before)]==before and after[len(before):]==encoded
        appends.append(dict(path=str(path.relative_to(ROOT)),prefix_bytes=len(before),prefix_sha256=hashlib.sha256(before).hexdigest(),append_sha256=hashlib.sha256(encoded).hexdigest(),heading_line=before.count(b'\n')+3,full_sha256=sha(path)))
    write(OUT/'append_receipt.json',dict(created_utc=now(),appends=appends,report_sha256=sha(OUT/'phase_report.md')))
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('artifact_manifest.json','manifest_verification.json')]
    manifest=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files];write(OUT/'artifact_manifest.json',dict(created_utc=now(),files=manifest))
    for record in manifest:assert sha(OUT/record['path'])==record['sha256']
    write(OUT/'manifest_verification.json',dict(created_utc=now(),count=len(files),all_hashes_match=True,manifest_sha256=sha(OUT/'artifact_manifest.json')))
    print(json.dumps(dict(appends=appends,verified_artifacts=len(files)),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':run('--append' in sys.argv)
