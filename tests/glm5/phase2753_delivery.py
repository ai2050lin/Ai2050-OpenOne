"""Seal evidence, generate the phase report and append it to research memos."""
import hashlib,json,re,subprocess,sys
from datetime import datetime
from pathlib import Path
import numpy as np
from phase2753_early_scope import ROOT,OUT,OLD,FAMILIES,write,sha,now,snapshot

LABELS={'fresh_entity':'新实体','fresh_surface':'新表达格式','fresh_role_order':'新角色×顺序','fresh_depth':'更深关系'}
ORDER=list(LABELS)
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def fmt(x):return f'{x:.4f}'

def report():
    s=read('confirmation_summary.json');sel=read('selection.json');q=read('quality_audit.json');r=read('readout_summary.json');d=read('post_confirmation_diagnostics.json');cap=read('4B/capture_done.json')
    stamp=datetime.now().strftime('%Y-%m-%d %H:%M')
    text=r'''## Phase 2753: 从关系与早层预测晚层交互——输入依赖削减及独立推广的失败边界 [__STAMP__]

### 1. 本阶段结论与执行状态

**状态：已完成有界实验；理论整体未完成。** 本轮将Phase2752“读取目标层三个单条件状态”的预测，推进为只使用文字、已知关系注释或单个前缀的早层状态。冻结384个新世界、3072条正式输入后，用旧训练／验证集选择14条候选路线，再采集新材料并一次性确认。结论是：**可以事先预测部分同构材料的交互，但没有获得跨表述、跨深度稳定的规则，也没有恢复可靠的答案读出。**

这是一项有信息价值的失败：它划清了“背景状态可回归”“交互可预测”和“正确关系判断”之间的证据边界，不能将三者互相代替。主结果目录：`tests/glm5/result/rdc_early_interaction_20260923/`；验证选择、拟合权重、逐组指标、原场与执行代码快照全部保存。

### 2. 材料、冻结顺序、输入合同与测量

- 模型：本地Qwen3-4B；CUDA、BF16、eager attention、batch1、无KV缓存、无padding或截断、无模型训练／量化。完整3个模型分片和tokenizer/config的SHA256见`model_identity.json`。本轮没有14B或GLM4复制实验，不能声称跨模型规律。
- 发现材料：仅加载Phase2752陈述句控制的train及validation，训练128个世界×2表述=256组，验证32个世界×2表述=64组；旧测试集响应不进入本次拟合／选择。旧训练真值并非精确平衡：每族17真／15假；旧验证每族5真／3假。本轮确认每族每轴12真／12假，不能将旧材料一概写成完全平衡。
- 新确认：4族（类别、钥匙传递角色、空间次序、包含）×4轴×24世界×2表述×4条件=3072条。1920个实体字符串在新世界间唯一，并与旧352世界中的全部实体字符串不重叠；这不是分词token或预训练语料的新颖性证明。
- 条件仍是语气ordinary/formal与陈述really/not的2×2交互，不是任意关系交互。新实体、角色顺序组合11、深度3各自使用已知两种表述；新表述轴使用项目此前未测的bullet事实与slash记录两种协议，同时改变格式／指令，不能宣称只操纵了纯语义改写。每个四格组token长度一致。
- 采集末token的38个边界、每个2560原生坐标：0为embedding；1..35为中间残差；36为最终RMSNorm输出；37另存最终原始残差。保存的FP32数组是原生BF16状态的无损数值嵌入；不等于整网FP32实验，也不是全token／全KV场。
- 封存顺序：`material_seal.json`→`algorithm_seal.json`→只用旧训练／验证产生`selection.json`→16条pilot→3072条正式采集→冻结预测确认→读出与明确标注的后验诊断。材料和代码均保留内容快照；正式结果未用于重新选alpha或重训。
- 测量门：3072条残差拼接与最终norm锚最大误差均为0，16条pilot与正式复采逐值一致；另实际执行16条截断前向，运行到第12块即中断，第4/8/12层状态与完整前向逐值一致，后24块及输出头未执行。该检查证明输入可在早层取得，不代表完整模型权重不占显存。
- 修改所有第13层以后状态及答案标签不改变预测特征；此断言检查了两条样本，并结合特征函数的明确访问范围，不冒充穷尽的软件形式验证。

**允许输入按成本分账。** static路线只读已知类型化边、角色／顺序／深度、token长度／位置以及输入embedding；target4/8/12读一个双条件目标前缀对应早层的末位向量；base8读基准前缀；four8读四个早层前缀。目标前缀文字在推理前已知，读取其早层状态合法；读取其目标层状态不合法。不同成本路线不混称为相同输入机制。

节点embedding取实际输入中首次出现的token跨度均值，多token名称不当成单token。static_binding还用查询中先后两个实体／角色短语、正向关系图的入邻居／出邻居均值、边差、类别排斥边差及查询槽乘积。图注释来自已知生成器与实际文字，不含truth/expected；它既不是自动语义解析器，也没有保留完整事件次序、全部token位置或高阶图路径。候选失败不能否定所有图组合算法。

### 3. 算法与可计算公式

令hᵃᵇ_ℓ为两个条件a,b∈{{0,1}}下的末位状态，主要预测对象在边界L=36：
$$I_\ell=h^{11}_\ell-h^{10}_\ell-h^{01}_\ell+h^{00}_\ell,\qquad E_I=\frac{{\|\widehat I_L-I_L\|_2}}{{\|I_L\|_2}}.$$
这是析因交互的定义，不是语言机制定律。零交互预测的E_I为1；结果是逐组相对误差的均值，不是“解释百分比”，更不能直接用1−E_I声称解释了多少智能。

特征按块保留全部原生坐标，每块只使用训练均值μ及一个训练RMS欧氏长度s标准化，拼成z。这样既不做PCA／Top-K，也不让较大维数的块仅因维数多而自动主导。对中心化训练特征矩阵Z与训练目标Y，拟合带截距岭回归；实现为等价的对偶形式：
$$\widehat Y_* = \mathbf1\bar Y+ Z_*Z^\top(ZZ^\top+n\lambda I_n)^{{-1}}(Y-\mathbf1\bar Y),\quad n=256.$$
每条路线的λ在{{0.001,0.01,0.1,1,10}}中用旧验证集E_I选定，无选择后重训。对偶实现通过独立原始矩阵岭回归核对。另设每坐标[h₈,h₈²]的native8路线，它与跨坐标岭回归是不同假设类。全部14路线及数值见`selection.json`与`confirmation_summary.json`。

验证集选择target8，λ=0.01，E_I=0.7904；固定族均值为0.8446；static最优graph_surface为0.8015。static_binding没有赢得验证选择。不能见到新测试结果后把某一轴较好的其他路线改称“已选中的主算法”。

独立拟合完整目标状态h¹¹_L，输入范围与对应交互预测器相同，λ单独按旧验证集的中心化误差选择：
$$E_{{H,c}}=\frac{{\|\widehat h^{11}_L-h^{11}_L\|_2}}{{\|h^{11}_L-\bar h^{{11}}_{{L,train}}\|_2}},\qquad E_H=\frac{{\|\widehat h^{11}_L-h^{11}_L\|_2}}{{\|h^{11}_L\|_2}}.$$
这样不借用三个真实晚层状态补回基线。全词表读出用真实绑定unembedding：q̂=softmax(W_U ĥ)，参考q=softmax(W_U h)；共同FP32乘法、关闭TF32。KL(q∥q̂)、模型top1一致性和真值首token正确率分别报告。

### 4. 独立确认结果

交互误差越低越好。每轴96个独立世界、192组；95%区间是族内分层、按世界配对bootstrap2000次，两个固定表述随同世界保留。区间不覆盖重新训练、模型选择、模板总体及自然语言总体的不确定性，也未进行多重检验校正。

| 确认轴 | 固定族均值 | 关系＋长度 | 早层target8 | target8减族均值的95%区间 | 允许晚层三源状态的参考* |
| --- | ---: | ---: | ---: | --- | ---: |
'''.replace('__STAMP__',stamp).replace('{{','{').replace('}}','}')
    for split in ORDER:
        mm=s['splits'][split]['methods'];ci=mm['target8']['paired_world_ci95']
        text+=f"| {LABELS[split]} | {fmt(mm['family_mean']['interaction'])} | {fmt(mm['graph_surface']['interaction'])} | {fmt(mm['target8']['interaction'])} | [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(d['splits'][split]['late_source_reference'])} |\n"
    text+=r'''
*晚层参考是确认之后补做的诊断：复用Phase2752已选source_product及λ=0.01，仅用同一旧训练集拟合，再输入本轮h00/h10/h01的真实晚层状态。它不参与本轮候选选择，不满足早层预测合同，不作为本轮新发现算法。

保留的有限结论：早层路线对新实体和新角色组合优于固定族均值。不能升级的结论：新实体上，它相对关系＋长度基线仅改善0.0029，后验配对区间跨0；在更深关系上没有可靠胜过族均值；新格式E_I>1，连零交互预测都未胜过。更强的晚层三源状态参考在新格式上也只到0.9696，说明此前推广到两种既定句式并不保证更广的格式推广。

静态绑定路线的新实体／新格式／新角色／更深关系误差分别为0.8153／1.0509／0.8382／0.8877；对应不绑定的bag路线为0.8184／1.0432／0.8104／0.8958。后验对照没有稳定的绑定收益，角色新组合反而变差。target8与base8在新实体、角色组合、深度三个轴差值都接近0，区间跨0；新格式target8略优于base8但两者均失败。不能据此宣称读出了独特的“现场点名”机制。

完整状态与读出结果如下。真值正确率只评价目标11条件的第一token；原模型全部四条件的总体正确率另列于结构化文件，不能拿不同条件的总体数值作为直接比较。

| 确认轴 | 完整状态相对误差E_H | 去除共同均值后的误差E_H,c | 全词表KL | 预测首token真值正确率 | 原模型同一11条件正确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
    for split in ORDER:
        mm=s['splits'][split]['methods']['target8'];rr=r['splits'][split]['target8'];beh=s['behavior'][split]
        text+=f"| {LABELS[split]} | {fmt(mm['full_absolute'])} | {fmt(mm['full_centered'])} | {fmt(rr['kl'])} | {rr['accuracy']:.2%} | {beh['target11_accuracy']:.2%} |\n"
    text+=rf'''
**低状态误差并不等于保留了语义决策。** 新实体E_H只有9.1%，但相对于输入间变化，误差仍是65.5%；首token真值正确率只有{r['splits']['fresh_entity']['target8']['accuracy']:.2%}。更深关系中，固定训练均值总回答“yes”，与原模型原生BF16 top1一致率达到92.71%，但真值正确率只有50%。这直接展示了输出一致性可能被模型偏好支配，不能把它当作理解能力。

所有四格答案均正确的组：新实体112/192、新格式119/192、新角色85/192、深度91/192。只看这些后验子组仍可见格式和深度瓶颈；该子组按输出筛选，会改变样本分布，不作为独立确认。保留全部行为错误，未用错误模型输出冒充成功推理。

线性读出方向的交互也未稳定推广。对单token“ yes”“ no”，(W_U[yes]−W_U[no])ᵀI是对应logit差的析因交互恒等式。target8该投影的符号一致率大约0.52–0.78，新格式只有0.526；它不是概率，也不是头子空间的夹角。FP32读出与原生BF16参考top1一致率{r['fp32_vs_native']['top1_agreement']:.4%}；原生已保存top20 logits的最大差{r['fp32_vs_native']['sampled_top20_max_logit_error']:.6f}，不能据此界定未保存全词表logits误差。KL以共同FP32读出为参考，不声称精确等于原生BF16整分布KL。

### 5. 失败原因的可证部分与未证部分

**可证的算法容量限制。** 对偶岭预测必在256个训练目标的仿射张成空间内。训练交互中心化数值秩为{d['dual_ridge_capacity']['centered_training_output_rank']}，而原生维数为2560。这是算法约束，不是发现了一个255维“语言本体”。将真实测试I投影到这个训练仿射空间所得不可部署的oracle下界为：
'''
    text+='、'.join(f"{LABELS[x]} {d['splits'][x]['oracle_train_affine_span_floor']:.4f}" for x in ORDER)+'。该后验下界读取了待预测目标，只用于量化表示能力限制；不作为预测指标、PCA主干或新机制。实际误差仍显著高于该下界，说明容量限制也不能独自解释全部失败。\n'
    text+=r'''
还没有区分的因素包括：早层末位向量省略其他位置与KV；图特征省略绑定／次序细节；全局岭映射不擅长输入条件下的运算切换；用隐藏状态L2选择的目标与输出决策方向不一致；固定构式覆盖不足及训练世界数有限。因此本轮否定的是这些具体预测路线已经实现普遍推广，不是证明“早层没有信息”“语义图无用”或“必须存在某个神奇非线性零件”。

图像见`native_coordinate_field.png`（原值、共享对称色标）、`native_coordinate_row_rms.png`（用观测行RMS同时归一化观测／预测／残差，显示限±6、原值保留）及`layer_error.png`。全部2560坐标保持原序；行是每轴每族48组平均，均值会抵消个体差异。层曲线1–8的灰区不是早层方法的未来预测；λ是按最终层选定，其他层为诊断，不能挑某个更好层当独立成功。所有逐组预测仍保留，展示平均图不取代其误差。

### 6. 可信拼图、参考资料一及RDC更新

| 拼图与来源 | 保留 | 本轮收紧的边界 |
| --- | --- | --- |
| 2750–2751／GPT3100：测量修复 | 正确SwiGLU乘积导数、RMSNorm导数、能量交叉项、同基底头比较 | 不恢复旧错误统计及过度结论 |
| GPT3055–3059：γ相关干预 | 特定读出对γ通道分配敏感 | 不支持严格白化、绝对读写独立或全部放大都由γ完成 |
| GPT3072：固定注意力V路径 | 固定Q/K/attention时，V到头输出线性 | 不等于自然注意力整体线性；Top词族不是已证语义原子 |
| GPT3079–3081：方向与干预谱 | 受限场景中的相关性及独立模型反例 | 没有TT—头响应子空间的普适路由律或物理共振证据 |
| 2751–2752：已知源状态组合 | 修复测量后部分同族交互可预测，三源状态有用 | 本轮新格式把源状态路线推到接近零预测，原范围必须保留 |
| 本Phase2753：早层合同 | 不用晚层源状态，部分新实体／角色组合仍有小幅预测收益 | 与简单图特征／基准早态差距有限，语义输出未闭合 |
| 本Phase2753：输出核查 | 状态误差、全词表分布、模型一致性、真值正确率分账 | 模型偏好可制造高一致性；状态背景可制造小相对误差 |
| 既有2745–2749接口及历史42公式／46拼图 | 保留原索引与原证据等级 | 本轮未逐项重新认证，新增前向不自动升级历史结论 |

参考资料一的最终判断保持不变：**“彻底破解万能复用”“头纯线性所以能处理一切”“高维无限正交复用”“写读绝对解耦”“角度匹配即共振”都超出证据或存在数学／对象混淆。** 本轮没有测量头路由，更不能用早层预测结果补证这些主张。可保留的弱版本是：共享参数在不同输入状态下表现不同，部分条件响应可以组合描述。先前详细逐条审查和原始引文定位见Phase2752及其`reference_claims.json`，本轮不将那些旧定位伪装成新测量。

本轮**没有新普遍闭合公式，也没有新数学定理**。沿用RDC的有限域接口与全局三图谱框架，历史𝐖专指类型化析因响应束，不能偷偷改成任意隐藏向量：
$$\mathcal T^D_{\ell,\tau}:(\mathcal L(p),\mathbf W_\ell(p),\mathcal X_\ell(p))\rightharpoonup\mathbf W_{\ell+1}(p),\qquad
\operatorname{Atlas}_D=(G_{external},G_{internal},E^D_{association}).$$
其中𝓛是已知语言条件与关系描述，𝓧为指定边界的观测场，τ为变换／条件类型，D限定材料及模型。关联边现在必须同时记录inputscope、forecast horizon、选择方式、状态误差、行为误差和失败域。本轮的z₈→Î₃₆是受限回归实例，不证明上式可作通用逐层接续。

可精确核对的统一计算底座仍是已知模型架构：
$$r_\ell=H_\ell+A_\ell(N_\ell(H_\ell);KV_\ell,position,mask),\quad x_\ell=N'_\ell(r_\ell),$$
$$H_{\ell+1}=r_\ell+W_{d,\ell}[\operatorname{SiLU}(W_{g,\ell}x_\ell)\odot W_{u,\ell}x_\ell],\qquad
p_{t+1}=\operatorname{softmax}(W_U N_f(H_L)_t).$$
这里H是所需位置的状态场，不是本轮单独记录的末位h；完整前缀、KV和位置依赖不能因观测简化而消失。写下架构恒等式不是提取出了语言算法。

三图谱实际新增：外部图增加384个独立确认世界、显式角色绑定／排斥边与新协议；内部图增加3072条原生全坐标末位轨迹及真实截断执行锚；关联图增加从静态图／embedding／早层输入到晚层交互的可重算映射、容量下界和读出失败边。没有定位新的单神经元或标量参数语义机制，回归系数不是原模型真实连接。

### 7. 第一性问题与下一大阶段

决定输出的不是隐藏向量“整体看起来相近”，而是与竞争词相关的方向。对最终归一化状态误差e，两个token u,v之间的logit差误差严格为：
$$\delta(z_u-z_v)=(W_U[u]-W_U[v])^\top e.$$
这个线性恒等式解释了为何总体范数误差不保证答案正确；它不是新发现的语言定律。第一性问题因此收紧为：**已知关系如何在输入条件下形成、维持并跨层传递那些真正改变候选竞争的信号，同时保持角色绑定与关系方向？** 必须同时控制表面位置变化，不能只拟合共同背景或用输出TT反过来解释自己。

下一阶段为一个整体任务“关系保持的跨层转移与决策充分性”，计划而未执行：

1. 冻结新的发现／验证／确认材料，真实平衡真值、角色、深度与表述；加入有相同bag统计而角色／边方向不同的最小对，以及保持语义的格式变换。先验证分词、首次答案分叉、完整输出与停止；解析注释仍注明人工／生成器来源。
2. 以全坐标、相关位置及所需历史为主体，比较关系边聚合／条件化转移算子、简单线性与位置基线；联合检查状态误差、输出分布和真值判断。输出约束仅用于训练目标，预测时不能读未来logits或答案。学习一个复用操作后再重复应用到更深组合，而不是仅对最终答案做整体回归。
3. 对同时通过语义和格式推广的子域，追到真实QK、V、MLP和norm计算连接，提出可区分的因果假设与替代解释，再做有定位的删除／救援。若仍失败，按输入不足、假设类不足或材料语义问题分流，不靠换术语持续报小进展。
4. 通过上述门后再顺序运行14B／GLM4和后续生成步；用独立关系族与更深路径检验跨域复用。单次拟合或若干好看的热力图不升级为“完整理论”。

本轮结束表示这个有界任务已经交付，不表示继续研究没有价值或科学问题已解决。可信资产是修复的量尺、明确的预测输入合同、局部可预测性及其失败边界；下一步应围绕决策信号与关系保持组织实验，而不是继续为“万能线性头”寻找佐证。

### 8. 资源、测试、保存与恢复

'''
    text+=f"正式采集{cap['elapsed']:.1f}秒，CUDA峰值{cap['peak_cuda_bytes']/2**30:.3f}GiB；旧数据发现拟合{sel['elapsed_seconds']:.2f}秒，新确认分析{s['elapsed_seconds']:.2f}秒，全词表读出{r['elapsed_seconds']:.2f}秒（各自进程阶段计时，不等于整轮任务耗时）。另有16条pilot及16条截断核查，均不是新增独立世界；只运行一个4B模型，600秒采集上限未触发。原场、旧结果及模型未清理。\n\n"
    text+='''代码入口为`phase2753_early_scope.py`、`phase2753_forecast.py`、`phase2753_validate.py`、`phase2753_diagnostics.py`和本交付脚本。只读接口`/api/rdc-construction/early-interaction`提供索引、完整指标、读出、原生坐标行与世界材料。新增及已有相关API共9项契约测试通过；存在Starlette关于httpx的弃用警告，未影响测试。没有声称重启服务或已部署UI。

冻结选择后不允许静默覆写发现结果；重算确认使用`phase2753_forecast.py confirm`，核查使用`phase2753_validate.py early/readout/quality`。独立再验证必须生成新运行目录和新确认材料，当前测试已经可见，不能继续把它称为未见测试。`artifact_manifest.json`登记交付文件hash，`append_receipt.json`登记memo追加前缀hash及追加位置。所有历史叙述采用追加更正，不改写旧Phase。
'''
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    groups=read('confirmation_groups.json');read_rows=read('readout_rows.json')
    native={r['id']:r for p in (OUT/'4B').glob('chunk_*.json') for r in json.loads(p.read_text(encoding='utf-8'))}
    with np.load(OUT/'confirmation_metrics.npz') as z:errors=z['target8_interaction'][:,36]
    examples=[];paragraph='### 4.1 可回查实例\n\n以下各取该轴交互误差最接近中位数的一组，为结果后的描述性实例，未用于选择算法。\n\n'
    for split in ('fresh_entity','fresh_surface'):
        ix=np.array([i for i,x in enumerate(groups) if x['split']==split]);j=int(ix[np.argmin(abs(errors[ix]-np.median(errors[ix])))])
        row=groups[j];prediction=tok.decode([read_rows['methods']['target8']['predicted_ids'][j]])
        examples.append(dict(group=row['group'],split=split,interaction_error=float(errors[j]),text=row['text'],expected=row['expected'],native=native[row['id']]['prediction_text'],forecast=prediction))
        paragraph+=f"**{LABELS[split]}：`{row['group']}`**，交互误差{errors[j]:.4f}；目标真值`{row['expected']}`，原模型首token`{native[row['id']]['prediction_text']}`，早层预测状态读出`{prediction}`。\n\n```text\n{row['text']}\n```\n\n"
    write(OUT/'representative_examples.json',examples)
    text=text.replace('### 5. 失败原因的可证部分与未证部分',paragraph+'### 5. 失败原因的可证部分与未证部分')
    (OUT/'phase_report.md').write_text(text,encoding='utf-8')
    write(OUT/'index.json',dict(phase=2753,status='completed_bounded_experiment',theory_status='partial_prediction_with_failed_format_depth_and_readout_generalization',created_utc=now(),
        worlds=384,prompts=3072,model='Qwen3-4B',selected=sel['selected'],report=str((OUT/'phase_report.md').relative_to(ROOT)),
        primary='confirmation_summary.json',readout='readout_summary.json',diagnostics='post_confirmation_diagnostics.json',quality='quality_audit.json',
        field='full_coordinate_fields.npz',figures=['native_coordinate_field.png','native_coordinate_row_rms.png','layer_error.png'],
        api='/api/rdc-construction/early-interaction',ui_scope='Read-only API implemented and tested; server restart/deployment not claimed',
        next_phase='Relation-preserving transitions and decision sufficiency; independent fresh confirmation required'))
    return text,stamp

def run(append=False):
    assert not (OUT/'append_receipt.json').exists(),'Final delivery already sealed.'
    text,stamp=report()
    write(OUT/'delivery_verification.json',dict(created_utc=now(),source=snapshot(Path(__file__)),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        working_tree='Pre-existing dirty tree preserved; execution sources stored by content hash.',tests=dict(command='.venv/Scripts/python.exe -X utf8 -m unittest tests/glm5/test_early_interaction_service.py tests/glm5/test_context_interaction_service.py tests/glm5/test_trusted_rebuild_service.py',
            observed_exit_code=0,tests=9,warning='Starlette TestClient httpx deprecation; no test failure'),figures_visually_inspected=True))
    if not append:
        print('Report generated for review; memos not appended.',flush=True);return
    records=[]
    gpt=f'''\n\n## Phase 3103: 早层交互预测的独立确认与读出失败边界 [{stamp}]

接续3102／GLM2752，本轮GLM Phase2753已完成：旧训练／验证选定预测器后，使用384新世界、3072条CUDA Qwen3-4B正式输入确认；只读文字／已知关系或单前缀早层状态，不读取目标晚层源状态。target8新实体／新角色组合交互误差0.8061／0.8076，小幅优于族均值0.8417／0.8309；更深关系0.9246没有可靠改善，新格式1.1380失败。完整预测状态的首token真值正确率约50%–54%，不能将背景状态可预测写成语义机制闭合。允许真实晚层三源输入的旧算法在新格式也降至0.9696，仅为后验诊断参考。

参考资料一“万能线性头／绝对读写解耦／无限正交／频率共振／彻底破解”仍不成立；本轮没有测量头路由，不能拿早层回归补证这些命题。低隐藏状态误差与高模型top1一致性都可能遗漏真正决定答案的差异。

完整材料、公式、各轴区间、原生全坐标图、代码／权重封存、容量oracle下界、真值评分、硬伤及下一整体阶段见`research/glm5/docs/AGI_GLM5_MEMO.md`尾部Phase2753，以及`tests/glm5/result/rdc_early_interaction_20260923/phase_report.md`。当前仍是有限域候选规律，没有新普遍数学定理；后续聚焦关系保持的跨层转移与决策充分性，须使用新的独立确认材料。
'''
    for path,prev,payload in [(ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md',2752,'\n\n'+text),(ROOT/'research/gpt5/docs/AGI_GPT5_MEMO.md',3102,gpt)]:
        before=path.read_bytes();phases=re.findall(r'^## Phase (\d+):',before.decode('utf-8-sig'),re.M);assert int(phases[-1])==prev
        encoded=payload.encode('utf-8')
        with path.open('ab') as f:f.write(encoded)
        after=path.read_bytes();assert after[:len(before)]==before and after[len(before):]==encoded
        records.append(dict(path=str(path.relative_to(ROOT)),prefix_bytes=len(before),prefix_sha256=hashlib.sha256(before).hexdigest(),
            append_sha256=hashlib.sha256(encoded).hexdigest(),heading_line=before.count(b'\n')+3,full_sha256=sha(path)))
    write(OUT/'append_receipt.json',dict(created_utc=now(),appends=records,report_sha256=sha(OUT/'phase_report.md')))
    excluded={'artifact_manifest.json','manifest_verification.json'}
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in excluded]
    manifest=dict(created_utc=now(),files=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    write(OUT/'artifact_manifest.json',manifest)
    for x in manifest['files']:assert sha(OUT/x['path'])==x['sha256']
    write(OUT/'manifest_verification.json',dict(created_utc=now(),files_verified=len(files),all_hashes_match=True,manifest_sha256=sha(OUT/'artifact_manifest.json')))
    print(json.dumps(dict(appends=records,artifacts=len(files)),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':run('--append' in sys.argv)
