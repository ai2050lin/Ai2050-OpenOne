"""Render evidence-derived append patches; never opens MEMO for writing."""
import argparse,re
from rdc_update_common import *


def n(value):
    if value is None:return '不可估计'
    if isinstance(value,(int,np.integer)):return str(value)
    if isinstance(value,(float,np.floating)):return f'{value:.7g}'
    return str(value).replace('|','／').replace('\n','<br>')


def ci(r):
    if not r:return '无可用匹配对照，不估计'
    return n(r['mean'])+' ['+', '.join(map(n,r['interval95']))+']'


def table(headers,rows):
    return '\n\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(n(v) for v in row)+' |' for row in rows)+'\n\n'


def reference(names):
    return '\n'.join('- `tests/glm5/result/'+BASE.name+'/'+p+'`。' for p in names)+'\n'


def phase38():
    required=['language_capture/result.json','language_analysis/result.json','language_identity/result.json','causal_anchor/result.json','causal_replay/result.json',
      'language_prediction/result.json','native_paths/result.json','same_history/result.json',
      'scale/qwen4/result.json','scale/qwen14/result.json','scale/glm4/result.json','scale_analysis/result.json']
    assert all((BASE/p).exists() for p in required)
    lang=read(BASE/'language_analysis/result.json');identity0=read(BASE/'language_identity/result.json');causal=read(BASE/'causal_anchor/result.json')
    replay=read(BASE/'causal_replay/result.json')
    paths=read(BASE/'native_paths/result.json');same=read(BASE/'same_history/result.json')
    pred=read(BASE/'language_prediction/result.json');scale=read(BASE/'scale_analysis/result.json')
    behavior=read(BASE/'behavior_analysis'/('result.json' if (BASE/'behavior_analysis/result.json').exists() else 'preliminary.json'))
    assert all(behavior['complete_collections'][m] for m in ('own_history','same_history','qwen4','qwen14','glm4'))
    txt=r'''
### C001：目标、材料与已经执行的范围

承接 Phase2736 的关系提取和 Phase2737 的真实参数继续训练，检查同一候选规则能否跨语言模式、回答方式、层和生成历史复用。本 Phase 已执行五类语言全坐标图谱、冻结预测器迁移、24例原生计算来源账本、1056条自身历史、108条同历史诊断，以及三个本地原始 BF16 模型的顺序复查。先观察和记录，再区分预测增量与机制证据；不以单坐标删救是否切换答案作为统一成功门槛。

五族是属性绑定、类型知识链、长距离角色、否定辖域和词义语境；每族32个语义组，每组中英文×直接回答/解释后回答4表达，共160组640表达。各族16/8/8组划为训练/验证/测试；同组4表达不跨划分。测试共160表达、320个锚点。英语主体用独立规则检查160组及4表达标签一致性；中文标签与镜像材料一致，不构成独立中文自然语义证明。词义组包含Apple/apple大小写与水果/公司问题的明确约定，不能把这种约定当普遍词义编码。与之联用的混合程序是 alias/mapping/conditional/addition 的组合，不把所有文本任务混称逻辑。自然主体另有真实 EWT/GUM 非标点内容窗口；本批语言材料是受控补充，不代表完整自然语言分布。

Qwen3-4B 保留原生2560坐标、36个block、每个已选MLP全部9728单元。640表达全部37层×2锚点，共121241600个H标量；block6/16/35全部激活共37355520个标量。另扫描全部token×全部层，保留逐层数组身份、全H12来源、预定全场fixture和可重算输入。这里的“全坐标”不意味着每条表达所有token的所有层原场全部永久归档。

下列是真实测试材料，不是教学虚构；完整模型输入、offset、token ID、target和语义组在同名material中。
'''
    rows=gzread(BASE/'language_material.json.gz')
    for family in sorted({r['family'] for r in rows}):
        r=next(r for r in rows if r['family']==family and r['split']=='language_test' and r['language']=='en' and r['answer_style']=='direct')
        txt+=f"\n- `{r['sample_id']}`，{family}，组`{r['source_group']}`，目标`{r['target']}`：\n\n```text\n{r['text']}\n```\n"
    txt+=r'''
### C002：全坐标条件关系、词汇身份及锚点纠错

同一模型内保持原坐标顺序，分别保存原值与行RMS视图；60个family/language/style/split条件剖面、H0/6/12/17/24/36×2锚点的12张完整640×640余弦矩阵均可查询。没有PCA、Top-K或删除低幅值背景。以同语义组跨表达相似减去同族、同真假但不同语义组对照，并按语义组计算区间：

$$
c_{ij}^{(\ell,a)}=\frac{\langle h_{i,\ell,a},h_{j,\ell,a}\rangle}{\|h_{i,\ell,a}\|\|h_{j,\ell,a}\|},\qquad
\delta_i=c_{i,j(i)}-\frac1{|\mathcal C_i|}\sum_{k\in\mathcal C_i}c_{ik}.
$$

这里h使用全部2560个原生坐标，a是已登记token锚点，j(i)是同组另一表达，C_i是声明匹配集合。它是条件统计定义，不是语义空间公理。下表只取英语direct→中文direct、正文锚点H12；完整层/位置/风格比较在原始文件中，不以这一个截面替代全场。
'''
    selected=[r for r in lang['comparisons'] if (r['layer'],r['anchor'],r['target_language'],r['target_style'])==(12,0,'zh','direct')]
    txt+=table(['族','原同族同真假对照优势与95%区间','再匹配当前token/精确位置可用组','加强对照优势与95%区间'],[
      [r['family'],ci(r['same_semantic_group_over_same_family_truth']),
       next(c for c in identity0['comparisons'] if (c['layer'],c['anchor'],c['target_language'],c['target_style'],c['family'],c['control'])==(12,0,'zh','direct',r['family'],'plus_current_target_token_and_exact_position'))['available_pairs'],
       ci(next(c for c in identity0['comparisons'] if (c['layer'],c['anchor'],c['target_language'],c['target_style'],c['family'],c['control'])==(12,0,'zh','direct',r['family'],'plus_current_target_token_and_exact_position'))['same_group_advantage'])] for r in selected])
    txt+=r'''
必须修正“正文末端等于句号位置”的潜在误读：640个正文锚点实际都是正文内最后一个完整token，没有一个恰好落在声明的正文字符终点。英语320例未纳入最后`.`，中文320例未纳入`。`，因为标点与换行可合为跨越边界的token。因此锚点常是名字或普通词，不是标点；数组、模型输入和原统计没有改动，新增了完整alignment表。

更关键的是，词义和长距离角色两族在严格当前token匹配后均为0/8可用对照。不能说其优势被“消除”，也不能说它们证实了独立于词汇身份的语义对齐；正确结论是这批材料对此不可识别。词义/角色在H0已经有优势，词汇身份、大小写和语境是具体替代解释。其他三族加强对照区间跨0。540组补充比较复用原矩阵且保存其原SHA，不是额外独立确认，不据此重选模型。

还有一个因果时间位置限制：正文锚点在问题和回答风格指令之前，尚未读到它们。已核对本地Qwen3配置：rope_scaling为空、rope_theta=1000000、max_position_embeddings=40960、未启用sliding window，本批长度均在范围内。对于相同前缀，这个固定位置规则下的理想精确因果计算满足：

$$
H^{\rm exact}_{\ell,t}(x_{\le T})=H^{\rm exact}_{\ell,t}(x_{\le t}),\qquad T\ge t.
$$

但BF16实现的不同完整执行长度可选择不同数值路径，不能强行要求相同bit。新增的只读归档审计核对全部320对direct/explain同正文前缀，token IDs全相同、H0全逐bit相同；H12仅110/320对逐bit相同，全坐标相对RMS差平均0.01053635，最大0.03129908。全部37层都核对，而非只看H12。这里不能把较晚风格指令解释为较早正文的语义原因；该审计也没有独立隔离具体舍入/执行长度/采集因素的贡献。

原采集是B1、无padding、完整prompt的因果mask前向，两种风格总长度不同。后来的问题/风格作用只能在已经读入它们的最终指令锚点研究，不把整任务标签提前当成正文已知语义。原数组、标签与结果未修改；`causal_anchor`保存320对身份、全部37层差异和640档案原SHA。

为检查严格可部署的早期预测，本轮另外对全部未见40语义组×两种语言＝80个唯一正文前缀执行4次原生前向：仅前缀、同前缀原样重复、direct后缀右padding到配对共同长度、explain后缀右padding到同长度。两个完整输入使用同B1总长度和显式mask，仅未来后缀改变；早期token ID与位置不变。保存全部37层×完整2560坐标、全部前缀H12来源及block6/16/35全部MLP单元。它覆盖160个风格表达的正文别名，但不算160个独立前缀。没有在这里生成回答。
'''
    txt+=table(['前缀执行审计','实际结果'],[
      ['唯一前缀／语义组／原生前向',f"{replay['unique_prefixes']}／{replay['semantic_groups']}／{replay['native_forward_calls']}"],
      ['相同前缀原样重复全部37层逐bit相同',replay['all_prefix_repeats_exact']],
      ['等长度不同未来后缀全部37层逐bit相同的配对',f"{replay['same_length_future_suffix_all37_exact_pairs']}/80"],
      ['H12仅前缀与旧完整direct前向的相对RMS均值',np.mean([r['H12_prefix_vs_old_direct_relative_RMS'] for r in replay['records']])],
      ['H12仅前缀与旧完整explain前向的相对RMS均值',np.mean([r['H12_prefix_vs_old_explain_relative_RMS'] for r in replay['records']])]])
    txt+=table(['block','冻结核','原冻结解码器','80个仅前缀按族语言分组相对MSE最小','最大'],[
      [r['block'],r['kernel'],r['decoder'],min(q['relative_mse'] for q in r['reports']),max(q['relative_mse'] for q in r['reports'])] for r in replay['fixed_predictor_reports']])
    txt+='\n冻结的旧320英语自然训练bank、head映射和解码选择全部保留；测试输入和真实目标都改用本次仅前缀原生前向，不输入未来问题、风格或答案。误差分母仍是旧训练均值预测误差。逐组directed−query比较及新旧协议误差差异保存在causal_replay；两种协议同时改变输入和真实目标，所以不能把差值命名为语义预测提升。它是已有未见组的数值／因果位置恢复检查，不是新的语义独立确认；旧训练bank仍保留其原完整prompt数值协议。本结果不外推为所有输入长度、padding或精度都逐bit一致。\n'
    txt+=table(['block','语言族／语言，仅正文前缀','相对query的误差差：组均值与95%区间（负数较好）'],[
      [r['block'],r['cohort'],ci(r['directed_minus_query'])] for r in replay['directed_minus_query']])
    txt+='\n值得保留的条件性线索：否定辖域中英文在两个block都比query核误差更低；知识链中英文在block16较低、block35较高，属性绑定则出现相反的优劣交叉。这是具体可复查的条件性预测结构，不能被笼统的“全部失败”抹去。但block16和35分别使用direct与native_joint两种冻结解码器，故优劣交叉不能孤立归因为层功能。尤其block16的否定材料相对旧训练均值误差仍大于1，优于query不等于已胜均值基线。每格只有8组，区间未作为多重检验校正后的新普遍规律；仍需身份控制、统一解码比较与独立自然材料。\n'
    txt+=table(['block','冻结核','仅前缀减旧完整prompt相对误差：40组均值及95%区间'],[
      [r['block'],r['kernel'],ci(r['relative_error_prefix_minus_old_full_prompt'])] for r in replay['cross_protocol_error_shift']])
    txt+='\n四组跨执行协议误差变化区间都跨0。因此当前冻结预测器的平均测试误差没有因仅前缀复核出现可区分的整体变化；这保留了有限预测结果，同时否定把原较早正文差异解释成未来风格指令作用的说法。它不保证每例数值都不变，也不修复尚未重采的旧训练协议。\n'
    txt+=r'''

### C003：从原生来源写入到全部MLP单元的条件账本

在材料选定时冻结24例：20例语言材料各两个锚点、4例程序材料各一个锚点。在block16和35记录全部可见来源、32个attention head、8个KV head、2560坐标及9728单元。先核对同形状重复前向，再展开原生参数收缩。不是把一个高亮单元命名为某种概念。

$$
C_{s,d}=\sum_{h,f}A_h(q,s)V_{s,\mathrm{kv}(h),f}(W_O)_{d,hf},\quad
x_s=\frac{\gamma}{\rho_{\rm observed}}\odot C_s,\quad
g_s=W_gx_s,\quad u_s=W_ux_s.
$$

$$
a_s=\tfrac12\sigma(g)\odot(g_s\odot u+u_s\odot g),\qquad m_s=W_da_s.
$$

q是当前查询token，s是因果可见来源，h/f是实际head与head坐标，W是原生权重。分母rho和sigmoid门使用本次已观察状态；残差、舍入与其他来源合并为显式other项，与所有来源共同重构g、u及激活。在完整来源+other的实数代数中，和式等于SiLU(g)⊙u；实际BF16与FP32收缩差异单列，不能消失在等号内。

这是已知线性收缩与对称双线性分配下的条件记账。它不唯一，不是把删去来源后的归一化和门保持不变的因果断言，也不能使用已观察后层门值冒充早层预测器。
'''
    audits=[a for r in paths['reports'] for a in r['audits']]
    txt+=table(['核对项','实际结果'],[
      ['原生样例／可见来源×锚点×block路径',str(paths['rows'])+'／'+str(paths['visible_source_anchor_block_paths'])],
      ['归档槽位（含因果未来零项）',paths['stored_source_slots_including_causally_masked_future_zeros']],
      ['24例同形状重复全部bit一致',all(r['same_shape_repeat_bitwise'] for r in paths['reports'])],
      ['FP32 attention收缩相对原生RMS误差最大',max(a['attention_relative_RMS_error'] for a in audits)],
      ['FP32 MLP相对原生RMS误差最大',max(a['MLP_FP32_vs_native_relative_RMS'] for a in audits)],
      ['分配恒等式相对RMS误差最大',max(a['activation_allocation_identity_relative_RMS'] for a in audits)]])
    txt+=r'''
### C004：未见表达预测与自身历史部署

Phase2736英语自然H12拟合的读出/核/解码器在640语言材料出现前已冻结；此处只用可见前缀H12，不输入测试gold依存边、答案或待预测后层状态。block16使用原选中direct解码、block35使用原选中native_joint解码，各自比较directed_rms与query核。报告的是后层MLP写回相对MSE，各细分为8个语义组；分母是仅预测原训练均值的总平方误差，接近1表示与该均值基线相近，不是误差接近0。
'''
    txt+=table(['block','固定核','原冻结解码器','各族/语言/风格/锚点相对MSE最小','最大'],[
      [r['block'],r['kernel'],r['decoder'],min(q['relative_mse'] for q in r['reports']),max(q['relative_mse'] for q in r['reports'])] for r in pred['records']])
    txt+=r'''
这些值没有形成跨族稳定预测优势。它们收窄的是“旧自然来源核已足以解释新双语模式”的主张，不能否定已记录的响应结构；也不能以跨语余弦较高替代后层预测。

主轨迹采用12分支×88材料＝1056条：16自然、32程序、40语言表达；各分支自身argmax推进，cap128。包含原生、末层冻结全参数方向及Phase2737四条32步中层训练的实际参数。完整token ID、首次分叉、答案/停止和原参数恢复独立记录。语言测试40表达来自10语义组，非40独立机制样本。自然续写没有唯一正确完整答案。
'''
    txt+=table(['自身历史分支','语言表达数','终止答案正确且EOS','可解析但答错且EOS','截断'],[
      [r['branch'],r['rows'],r['correct_and_stopped'],r['wrong_parsed_and_stopped'],r['censored']]
      for r in behavior['summaries'] if (r['mode'],r['granularity'],r['cohort'])==('own_history','kind','controlled_language')])
    txt+=r'''
所有12分支的32条程序表达在128token内都未完整结束；这叫截断，不能记为32条都答错。初始leading Yes/No检测可能把解释正文开头误当最终答案，现正式改用保守终止答案解析，原commits和生成ID不改。新增1024token诊断归入Phase2739。训练前后语言结果未出现可靠普遍改善；监督loss和有限导数改善不等于自然语言推理能力改善。

另在18材料×6已执行更新＝108条主轨迹上，用未修改模型独立重放完全相同的主分支历史，比较同历史的输出和KV。参照从不回注主分支。所有主轨迹ID完全一致，原参数恢复检查通过。
'''
    txt+=table(['同历史更新分支','来源组平均KL与95%区间','KV相等检查／总检查','改变的KV block'],[
      [r['branch'],ci(r['mean_source_KL']),f"{r['all_KV_equal_checks']}/{r['total_KV_checks']}",str(r['changed_KV_blocks'])]
      for r in same['summaries']])
    txt+=r'''
末MLP更新不改变该步已生成的所有层KV，三条末层方向均156/156相等；中间block16更新会使后续17..35的KV不同，三条中层分支均0/156全部KV相等。此处定位的是实际架构接续边界；不是证明末层更新不会改变未来生成历史——一旦其选出的token不同，下一步输入也可不同。

### C005：三个原始BF16模型、执行形状恢复与边界

三个模型各128条同源材料（32自然、56程序、40语言）与36条自由生成（8自然、8程序、20语言），保持各自tokenizer、chat格式、层数和原生坐标。逐个加载，CUDA+CPU/磁盘device_map，不量化，不同时运行两个模型。

Qwen14在B1下发生严重CPU/磁盘换页慢，三个被中止尝试均保留运行时间与源码/运行元数据；未把未完成轨迹写成成功。恢复为固定左padding、显式mask和从0起的有效position IDs，prefill最大batch8，生成自然8/程序8/direct语言10/explain语言10。没有改材料ID、cap、权重或答案，没有回注参照KV。Q4原B1数据不动，另采同批batch8全坐标shadow用于三模型关系比较。每模型另外6例独立B1全前向，保存完整H和全词表对数概率数值对照。较大模型的hf_device_map保留最初派发映射，原BF16 lm_head随后单独驻留GPU；实际驻留依据residency/load_audit.json的独立字段，不能只看映射中lm_head的初始disk标记。
'''
    model_rows=[];shape_rows=[]
    for m in ('qwen4','qwen14','glm4'):
        r=read(BASE/'scale'/m/'result.json');s=r['runtime']
        model_rows.append([m,s['width'],s['depth'],s['units'],r['rows'],r['generation_rows'],r['seconds']])
        folder=BASE/('scale_batch' if m=='qwen4' else 'scale')/m
        a=read(folder/'shape_audit/result.json')['rows']
        shape_rows.append([m,len(a),sum(v['batch_H_bit_exact'] for v in a),max(v['all_H_relative_RMS'] for v in a),
          max(max(v['full_vocab_KL_B1_to_batch']) for v in a),np.mean([v['first_logits_argmax_agreement'] for v in a])])
    txt+=table(['模型','坐标','block数','MLP单元','材料','生成','主执行秒（不含失败尝试）'],model_rows)
    txt+=table(['模型','B1对照例数','全H bit一致例数','全H相对RMS差最大','全词表KL最大','各例锚点argmax一致率均值'],shape_rows)
    txt+=r'''
BF16 batch8不能当成与B1逐bit相同；这个差异已实测保存，不能用“同模型”掩盖执行形状混杂。三模型关系核使用匹配batch8字段，但Q4行为是原B1、较大模型行为是batch8/10，行为差异不能孤立解释为规模效应。六个prefill控制也不等于对所有生成历史已完成B1等价验证。
'''
    txt+=table(['模型','材料种类','生成数','正确且停止','可解析错误且停止','EOS数','截断数'],[
      [r['mode'],r['cohort'],r['rows'],'不适用' if r['cohort']=='natural' else r['correct_and_stopped'],'不适用' if r['cohort']=='natural' else r['wrong_parsed_and_stopped'],r['EOS'],r['censored']]
      for r in behavior['summaries'] if r['mode'] in ('qwen4','qwen14','glm4') and r['granularity']=='kind'])
    txt+=r'''
内部跨模型比较使用同材料关系矩阵，不对应不同模型的坐标索引。分别计算embedding、早层query、末层query、来源均值、postnorm、MLP激活的完整128×128矩阵；三模型每个单元的gate/up均值、协方差、相关及未定义mask均保留。不是所有单元两两协方差。进一步在cohort/split对之间中心化，再做200次族内行置换；这是结果后描述诊断，不是新的独立p值或语义同构证明。
'''
    txt+=table(['模型对','全坐标特征','全部材料对相关','cohort-pair中心化相关','族内打乱均值','打乱描述95%范围'],[
      ['/'.join(r['models']),r['feature'],r['all_pairs_correlation'],r['cohort_pair_centered_correlation'],r['within_cohort_shuffle_mean'],str(r['within_cohort_shuffle_interval95'])]
      for r in scale['cross_model']])
    txt+=r'''
### C006：三图谱、理论增量与问题硬伤

外部图谱新增五族双语关系与角色/词义/风格的显式身份，内部图谱新增全坐标条件矩阵、原生来源账本和自身历史，关联图谱新增冻结迁移预测及实际训练参数的部署结果。共同现象是条件响应和历史作用随层、来源和表达改变；候选规律是可用前缀来源关系与后层门控写回；原生参数结构由真实全矩阵计算核对；未见预测尚不稳定；训练形成证据是受限继续训练，不是还原原始预训练。

当前限制是：语言受控模板/词汇身份混杂；每细分仅8组；正文不能因果地读入较晚问题/风格指令；完整prompt采集的早期状态有执行长度数值限制；较大模型只做原生复查没有独立重复训练；执行形状和分词不可忽略；固定门/归一化的来源账本不是唯一因果分工；旧核迁移几乎没有足够稳定增量；首token监督目标与完整自主回答之间仍有断层。不能把出现了原生参数等号称为找到了语言的“万有引力”。RDC主体公式本Phase没有修改，也没有新增全局闭合定理。

第一性原理启发是：表示覆盖、可识别性和预测充分性是三个问题。即使所有坐标都保留，若语义组与词汇身份锁定，就无法识别关系作用；即使来源账本精确，也仍需从早期可用量预测它。继续推进自然有序事件来源、身份/位置强控制和可查询未来，不退回“换一句话—单坐标删救”的循环。

下一个阶段仍服务同一目标，已按冻结大方案自动进入Phase2739：新自然来源确认、有限矩反例、可达前缀查询、KV成本已经执行，1024token答案诊断正在按资源门顺序完成；全部最终状态见随后追加的Phase2739。数据均用于客户端或后续研究，保留而不执行条件清理；未删除用户或旧研究材料。

### C007：相关文件与复算入口

脚本位于`tests/glm5/phase2738_rdc_update_*.py`及`phase2739_rdc_update_scale_batch.py`、`phase2739_rdc_update_language_identity.py`、`phase2739_rdc_update_scale_analysis.py`；原始运行源码快照在本批sources下。已中止尝试见`scale_recovery`与`runtime_recovery`，不抹去失败。客户端`/rdc-update`提供原坐标、来源图、标量、梯度和轨迹查询。

'''
    txt+=reference(required+['scale_batch/qwen4/result.json','scale/qwen14/shape_audit/result.json','scale/glm4/shape_audit/result.json','behavior_analysis/preliminary.json'])
    return txt


def phase39():
    required=['fresh_graph/result.json','moment_boundary/result.json','predictive_state/result.json','long_answers/result.json',
      'behavior_analysis/result.json','terminal_format_audit/result.json','manual_terminal_audit/result.json','theory_snapshot.json','verification/scientific_integrity.json','verification/model_checkpoint_fingerprints.json',
      'client/api_final.json','client/browser_final.json','client/code_checks.json','client/visual_review.json','next_stage_admission.json']
    assert all((BASE/p).exists() for p in required)
    fresh=read(BASE/'fresh_graph/result.json');mom=read(BASE/'moment_boundary/result.json');psr=read(BASE/'predictive_state/result.json')
    long=read(BASE/'long_answers/result.json');behavior=read(BASE/'behavior_analysis/result.json');theory=read(BASE/'theory_snapshot.json')
    science=read(BASE/'verification/scientific_integrity.json');admission=read(BASE/'next_stage_admission.json');terminal=read(BASE/'terminal_format_audit/result.json')
    manual=read(BASE/'manual_terminal_audit/result.json')
    txt=r'''
### C001：同目标自动续研与本Phase状态

本Phase不是只保存下一步计划：在Phase2736—2738整合方案之外，实际完成同目标续研：128新自然窗口的强匹配对照、三/四阶全坐标矩的严格适用边界、自然可达前缀的固定查询实验、真实KV资源核算，以及192条最长1024token答案诊断。它们共同回答：已有全坐标结构是否提取到了不能被更简单统计解释、且足以支持接续的机制信息。

延续观察→候选结构→未见预测→局部原生核对，不把因果闭合失败当作终止研究的唯一理由。未执行100K无损KV或“消除遗忘”：有限低阶矩天然无碰撞这一所需前提被明确反例否定，不能在错误前提上扩大GPU规模。这个结论只限制所测摘要，不是否定所有可压缩记忆。

### C002：新自然来源、同RMS强基线与算子方向边界

冻结5个同RMS、同df128、同原拟合材料的query/square/position/shuffled/directed核，沿用block16的direct解码与block35的native_joint解码，不在新结果上重新选择胜者。新增128真实自然窗口、384个非标点内容锚点、96个来源文档。与已有2048条自然材料清单核对source component不重复，并排除旧拟合训练文档。正式材料来自官方dev92/test36；GUM为dev38/test26，EWT为dev54/test10。最初仅从剩余EWT test不足以构成冻结覆盖，在读取新模型结果之前扩大到已缓存官方dev；这不是查看结果后的抽样补救。

相对MSE按来源组计算区间。下表值为“directed误差－对照误差”，负数才有利于directed；不是绝对语义准确率。fresh_connected与fresh_matched是声明自然连接条件划分，不是全语言组合深度证明。
'''
    txt+=table(['block','对照核','自然划分','语料','directed－对照：来源均值与95%区间'],[
      [r['block'],r['control'],r['split'],r['cohort'],ci(r['source_cluster_advantage'])] for r in fresh['matched_controls']])
    txt+=r'''
结果细化了Phase2736观察：中层有向核胜query、同幅值打乱和位置基线，但没有胜过更简单的square核；因此不能把它的全部增益解释为正确的语义来源绑定。末层比较更依赖语料和划分，对打乱的优势并非全面复现。已有全坐标观察保留，但“提取到了普遍条件齿轮”的解释应降级。

另一个数学硬伤需要单独写明。以完整来源矩阵H_i和软关系A_i构造：

$$
T_i=\frac{H_i^\top A_iH_i}{n_iD},\qquad S_{ij}=\langle T_i,T_j\rangle_F,\qquad
K_{ij}=1+b_{ij}+S_{ij}+b_{ij}S_{ij}.
$$

$$
b_{ij}=\frac{q_i^\top q_j+e_i^\top e_j}{2D},\qquad
\langle T_i^\top,T_j^\top\rangle_F=\langle T_i,T_j\rangle_F.
$$

q/e是原可用query/H0，D=2560，n_i是可见token数。即使T_i不对称，全体共同转置后Gram仍相同；因此全局这个核本身不能识别因果箭头方向。显式T_iq与T_i^Tq是不同候选，不能把内积核的名字“有向”当成方向识别证据。这是已知线性代数的适用性审查，不是新的基本数学定理。

### C003：低阶有符号矩仍不能唯一保存有序历史

测试对象明确为32个位置、全部2560坐标的FP64合成历史。构造epsilon_s=(-1)^popcount(s)，取位于已冻结粗角色线性读出零空间、单位RMS的完整向量v：

$$
\epsilon_s=(-1)^{\operatorname{popcount}(s)},\quad s=0,\ldots,31,\quad
\sum_s\epsilon_s s^j=0\ (j=0,\ldots,4),\qquad H_{a,s}=\epsilon_sv,\quad H_{b,s}=-\epsilon_sv.
$$

$$
\sum_s s^j H_{a,s}^{\otimes k}=\sum_s s^j H_{b,s}^{\otimes k},
\quad k=1,\ldots,4,\ j=0,\ldots,4,
\qquad \operatorname{softmax}(qs)^\top H_a\ne\operatorname{softmax}(qs)^\top H_b\ \text{可成立}.
$$

偶数阶逐位置相等，奇数阶由精确整数Prouhet恒等式抵消，因此推导覆盖所有张量坐标，不需要把D^4个元素同时物化。角色读出也几乎相同。查询exp(qs)不是声明的4次以下位置多项式，能区分两段历史。实际全部20组阶数/位置阶数检查和4种查询均保存。
'''
    txt+=table(['核对项','实际数值'],[['角色零空间误差',mom['role_nullspace_error']],['角色读出最大差',mom['role_scores_max_difference']],
      ['精确整数0..4阶和',str(mom['exact_integer_sums_powers0through4'])],['20组数值系数差最大绝对值',max(abs(r['coefficient_difference']) for r in mom['all_state_position_moment_checks'])]])
    txt+=table(['查询q','attention输出差RMS'],[[r['query'],r['attention_output_difference_rms']] for r in mom['attention']])
    txt+=r'''
边界必须保留：这没有证明两段合成状态都能由自然Transformer前缀到达；不是所有压缩算法、完整KV、逐位置one-hot身份存储或适当条件下全阶矩的反例；更没有证明人脑必然依赖这些矩。真正进展是识别了“全坐标”与“有序信息充分”之间的逻辑空缺：不删坐标，也可因有限汇总运算而丢失查询信息。

### C004：可达自然前缀与有限查询预测状态

选32中心自然前缀，按可见H12 query/H0 embedding/全部来源均值的完整坐标描述，在长度差不超过8的候选中各取近/远前缀，共64对、63个不同实际前缀、27个来源组。它们描述相近而非完全相等，不叫精确碰撞。对每个原token前缀追加固定probe token IDs：空串、` the`、` because`、` However,`、换行后`The`，不把追加probe与前缀整体重新分词。每例真实原生模型前向，比较完整词表：

$$
\Psi_{\mathcal Q}(p)=\left[P_\theta(\cdot\mid p\oplus q)\right]_{q\in\mathcal Q},\qquad
\Delta_q(p,p')=\tfrac12\left[D_{KL}(P_{pq}\Vert P_{p'q})+D_{KL}(P_{p'q}\Vert P_{pq})\right].
$$

P是全词表概率分布，oplus是已登记token ID拼接，Q是固定可获得查询集合而不是未来gold答案。有限查询下接近只给出操作性等价范围，不能由此证明所有未来字符串下等价或马尔可夫闭合。
'''
    txt+=table(['固定probe','近对平均对称KL','远对平均对称KL','来源组近－远与95%区间'],[
      [repr(r['probe']),r['near_mean_KL'],r['far_mean_KL'],ci(r['paired_near_minus_far'])] for r in psr['summary']])
    txt+=r'''
空probe的近对更接近输出，追加probe后四种比较区间均跨0，且近对本身KL并非0。这支持当前描述对立即读出有局部预测关联、但不能保证未来查询作用下保留相同关系。它不能升级为自然状态严格不可压缩证明。

真实Qwen4全部36层KV逐块计数、数组哈希并与架构成本核对：

$$
\operatorname{bytes}(KV)=2L\,n_{KV}\,d_{head}\,T\,b
=2\times36\times8\times128\times T\times2.
$$

这里首个2是K与V，末个b=2是BF16字节；不是把完整残差宽度误当KV宽度。512/1024/2048token分别实际72/144/288MiB，所有层都计算并扫描；KV工作张量在核算后释放，仅保留身份、配置和输入作为可重算临时数据，没有删除已归档证据。

### C005：192条1024token诊断、停止与答案分离

原128token的程序轨迹全部截断，所以保留原32表达、8语义组，比较native、末层format-constrained1e-6及四条真实中层32步训练分支，共6×32=192条。先2例原生pilot再扩展；最长1024且不继续追长。每条都核对其前128token与旧分支自身历史完全一致，没有成功选择样例，没有给模型塞答案或参照历史。

正式评分分别为仅数字的严格输出、完整终止答案解析、EOS、截断和“正确且停止”。解释开头出现Yes/No不是自动终止答案；未完整解析的截断也不能自动判成语义错误。

$$
Y=\frac1N\sum_i\mathbf1\{\widehat a_i=a_i^{gold}\ \land\ EOS_i\}.
$$

Y是操作性成功产出率，不是宣称未解析输出都答错后的准确率。完整思维链的每句话不在该评分覆盖内；自然续写没有唯一gold完整答案。
'''
    txt+='\n先完整保留运行时冻结的较窄原解析器结果：\n'
    txt+=table(['分支','表达','语义组','平均token','原解析正确且停止比例及8组区间','EOS率','截断率','原解析覆盖'],[
      [r['branch'],r['expressions'],r['semantic_groups'],r['mean_generated_tokens'],ci(r['stopped_correct_cluster']),r['EOS'],r['censored'],r['parse_coverage']]
      for r in long['summary'] if r['representation']=='all'])
    txt+=table(['训练/更新分支','相对原生正确且停止产出差：来源组均值及95%区间','新增成功数','失去成功数'],[
      [r['branch'],ci(r['operational_yield_delta']),len(r['new_correct_stopped_ids']),len(r['lost_correct_stopped_ids'])]
      for r in behavior['paired'] if (r['mode'],r['kind'])==('long_answers','controlled_program')])
    txt+=r'''
原生长答案完成后发现明确的评分覆盖缺口：`Final Answer`后的数字可能在代码块、显示公式或boxed中，也可能是题目指定变量的完整末行值陈述。较窄原解析器把这些保持为“未解析”，不能误称为答错。新增**次级、事后**格式审计：只在EOS且未截断的程序输出上，补认完整末尾答案标记的Markdown/LaTeX字面量、末尾boxed字面量、或精确匹配题目所问变量的完整末行值。变量名来自原题关系记录，gold值只用于之后判断正确性；不执行生成代码，不抓取任意最后数字。原评分器和全部原commits/生成ID保持不变，两套统计同时保留。

格式审计开发时已有部分长答案，故不是预注册新确认；`terminal_format_audit/protocol.json`保存实际冻结时间、当时存在的分支记录数、首次协议和修正链。原10例与新增18例回归都实际执行，包括错误答案、错误变量、未完成文本、歧义多数字及截断拒绝。一版bold分隔符处理被回归测试拒绝，修正前源码和拒绝记录保留；不能把调试写成始终成功。
'''
    txt+=table(['分支','相同32表达中的格式复核正确且停止','错误且停止','仍未解析且停止','格式复核成功率及8组区间','严格仅数字输出数'],[
      [r['branch'],r['correct_and_stopped'],r['wrong_parsed_and_stopped'],r['EOS']-r['parsed_and_stopped'],ci(r['success_source_cluster']),
       sum(bool(v['format_aware_scoring']['strict_answer_only']) for v in gzread(BASE/'behavior_analysis'/behavior['records_file']) if v['mode']=='long_answers' and v['branch']==r['branch'])]
      for r in behavior['format_aware_summaries'] if r['mode']=='long_answers' and r['granularity']=='kind'])
    txt+=table(['训练/更新分支','格式复核后相对原生成功率差：8组均值与95%区间','新增成功数','失去成功数'],[
      [r['branch'],ci(r['operational_yield_delta']),len(r['new_correct_stopped_ids']),len(r['lost_correct_stopped_ids'])]
      for r in behavior['format_aware_paired'] if (r['mode'],r['kind'])==('long_answers','controlled_program')])
    txt+=r'''
逐条阅读剩余未解析终止输出后，又发现明确的覆盖边界：中文“只输出一个数字：4”、英文“The digit printed is 7”和最终输出箭头等完整终值陈述仍可能不在冻结次级语法内。不能由某训练分支自动成功计数较低就推论语义能力退化。本轮**不再修改两版解析器**，对次级解析器下全部未解析、EOS且未截断的长输出，执行第三层人工终止答案复核：主研究代理读完整输出，逐项填写明确答案或弃权及原文末尾证据，保留原commit SHA。清单全覆盖、引用原文与SHA由独立脚本核对。

这是主代理非盲、事后判读，不能冒充预注册解析器、独立评审或新的模型运行。它只评判明确末答，不给推理链逐句评分，也不把未解析或截断自动判错；未来确认需预先冻结充分的语言包装规则或另设盲评。
'''
    txt+=f"\n残余清单共{manual['reviewed_residual_outputs']}条，明确终值复核解决{manual['resolved_outputs']}条。两版自动评分、人工注释及原始生成三者分别保留，人工复核不改写behavior_analysis/result.json。\n"
    txt+=table(['分支','人工终值复核后正确且停止','错误且停止','仍未解析','成功率及8组区间'],[
      [r['branch'],r['correct_and_stopped'],r['wrong_parsed_and_stopped'],r['EOS']-r['parsed_and_stopped'],ci(r['success_source_cluster'])]
      for r in manual['summaries'] if r['granularity']=='kind'])
    txt+=table(['训练/更新分支','人工终值复核后相对原生成功率差及8组区间','新增成功数','失去成功数'],[
      [r['branch'],ci(r['operational_yield_delta']),len(r['new_correct_stopped_ids']),len(r['lost_correct_stopped_ids'])]
      for r in manual['paired']])
    txt+='\n因此，训练的候选内容损失、自动解析覆盖和明确终值正确性必须分别解释。主结果不能用较窄解析器的低覆盖制造语义退化，也不能把同一次生成经过复核后的计数上升叫作训练进步。下述比较仅支持本批8组范围内的判断；重复表达、非盲复核和有限组数均限制推广。\n'
    txt+='\n实际结果为原生及末层约束更新31/32；两条seed2737中层分支30/32，两条seed2738分支31/32。seed2737两分支各新增1个正确末答、同时失去2个；seed2738内容分支新增1个、失去1个，约束分支与原生终值正确性完全同列。全部配对差区间均包含0：不能声称稳定提升，也不能把较低自动解析计数写成显著语义退化。192/192均EOS，严格仅数字输出0/192；原128token截断主要限制了该批程序终值的测量，不能将它当作模型已整体崩溃的证据。这一边界只针对本批程序，不覆盖自然长篇质量、推理链逐句有效性或1024之外的行为。\n'
    txt+=f"\n全部1464条既有轨迹中，本次格式审计补解析{terminal['recovered_records']}条；这是同一次生成的测量修正，不是新增模型运行或学习收益。主解析器不变检查、逐条原commit SHA和次级复算均通过。若Phase2738某个大模型程序EOS样例被补认，以下同批短生成两套数字可直接对照，原Phase2738表保留其当时测量版本。\n"
    txt+=table(['模型','原较窄解析正确且停止','格式复核正确且停止','格式复核错误且停止'],[
      [r['mode'],next(q['correct_and_stopped'] for q in behavior['summaries'] if (q['mode'],q['granularity'],q['cohort'])==(r['mode'],'kind','controlled_program')),
       r['correct_and_stopped'],r['wrong_parsed_and_stopped']]
      for r in behavior['format_aware_summaries'] if r['mode'] in ('qwen4','qwen14','glm4') and (r['granularity'],r['cohort'])==('kind','controlled_program')])
    txt+='\n真实输出示例仅展示最后800字符，完整生成和每步logprob均在commits。列出原先未解析但有明确正确末答的例子，以及实际错误/仍未解析（如有）；不把例子选择作为胜者选择或独立证据。\n'
    records=[read(p) for p in sorted((BASE/'long_answers/commits/native').glob('*.json'))]
    formal={r['sample_id']:r for r in gzread(BASE/'behavior_analysis'/behavior['records_file']) if r['mode']=='long_answers' and r['branch']=='native'}
    cases=[]
    for kind in ('recovered_correct','wrong','unparsed'):
        rr=[r for r in records if ((kind=='recovered_correct' and not r['answer_scoring']['parsed_and_stopped_correct'] and formal[r['sample_id']]['format_aware_scoring']['parsed_and_stopped_correct'])
          or (kind=='wrong' and formal[r['sample_id']]['format_aware_scoring']['conservative_final_correct'] is False)
          or (kind=='unparsed' and formal[r['sample_id']]['format_aware_scoring']['conservative_final_answer'] is None))]
        if rr:cases.append(rr[0])
    for r in cases:
        txt+=f"\n- `{r['sample_id']}`；目标`{r['target']}`，生成{len(r['generated_ids'])}token；原解析：`{json.dumps(r['answer_scoring'],ensure_ascii=False)}`；格式复核：`{json.dumps(formal[r['sample_id']]['format_aware_scoring'],ensure_ascii=False)}`。\n\n```text\n{r['generated_text'][-800:]}\n```\n"
    txt+=r'''
这批长答案仍只是原8组的延长诊断，不能把192重复分支当192个独立新语义任务；跨分支的成功增减需与8组区间一起看。四条训练仅改变block16三矩阵；其内容/格式目标、两种批序以及原生suffix反传早已在2737记录，不把本Phase重放当作新训练。长答案没有扩展高斯等范数分支，且末层和中层实际更新范数不同，不能将结果分离为纯粹的更新方向效果；格式审计也不评判推理文本每一步是否正确。

### C006：完整核心拼图清单、RDC统一接口和全局图谱公式

累计拼图保留旧34项并新增本轮4项，共38项。旧项的原适用域和未重新执行状态不能被本表抹去；“保留”不意味着全部升级为闭合规律。下面给出全部索引与边界，细粒度实例、源码、产物引用在`theory_snapshot.json`及相应Phase。
'''
    txt+=table(['对应Phase','保留的拼图','限制／纠正','本轮状态'],[
      [p['phase'],p['retained_puzzle'],p['boundary'],p['evidence_status']] for p in theory['puzzles']])
    txt+=r'''
RDC名称保持“条件化输出场闭合理论”；本轮主体公式未改，没有新增全局闭合定理。特别保留原历史粗体W为类型化析因响应束，不擅自替换成任意HiddenState：

$$
X_\ell(p)=\{H_{\ell,t,j}(p)\},\qquad
\mathcal T^{\mathcal D}_{\ell,\tau}:\big(\mathcal L(p),\mathbf W_\ell(p),X_\ell(p)\big)
\rightharpoonup\mathbf W_{\ell+1}(p).
$$

p是声明材料，L是外部描述，X是观察全场，W是原析因响应束，tau是条件/角色，D是已测试域；部分箭头表示未全域闭合。它是统一问题接口，不是已经找到一个无需原模型便能精确执行全部语言的定理。

原生Transformer计算核对对象仍是架构定义：

$$
r_\ell=H_\ell+\operatorname{Attn}_\ell(N_\ell(H_\ell),KV_\ell,\mathrm{position}),\quad
x_\ell=N'_\ell(r_\ell),\quad
H_{\ell+1}=r_\ell+W_d^\ell[\operatorname{SiLU}(W_g^\ell x_\ell)\odot W_u^\ell x_\ell],
$$

$$
P_{next}=\operatorname{softmax}(W_UN_{final}(H_L)_{query}).
$$

实际mask、RoPE、bias和有限精度顺序随本地架构记录；这些等号不能单独算新语义理论。三图谱以外部类型关系与内部响应的稳定ID关联，关联对象是可执行、限定域的预测和来源计算，而不是把二者图形画得相像。

为避免仅列新式而丢失旧理论，下面逐项保留当前31条公式/算法账本的完整机器可计算表达、变量与证据限制。旧23条为继承范围；新增8条均用已知线性代数、微积分、概率和组合恒等式表达，没有证据要求诉诸新数学。
'''
    for i,f in enumerate(theory['formulas'],1):
        txt+=f"\n#### 公式账本{i:02d}：{f['id']}\n\n类型：`{f['kind']}`。\n\n```text\n{f['expression']}\n```\n\n变量：{f['variables']}\n\n证据与边界：{f['evidence']}\n"
    txt+=r'''
### C007：三图谱实际增量、硬伤与第一性原理

外部图谱拥有2176个稳定ID：512旧严格自然、256新自然、768混合程序、640双语表达；这不是2176独立自然文档。内部图谱包括完整锚点坐标/早层来源、全部指定MLP单元、可见原生来源链、74711040参数的全因子梯度、实际继续训练增量和生成历史。关联图谱包含冻结核、强控制、真实参数来源恒等式和学习效应，但尚无从语言操作到唯一条件齿轮再到跨层组合的完整预测编译器。

关键排除项有四个：第一，同RMS强控制显示有向核收益不能排除简单平方结构；第二，正文词汇身份匹配缺少对照使高余弦的语义解释不可识别；第三，低阶全坐标矩不能保证保存有序查询信息；第四，受限真实训练和首token监督收益尚不能推出完整语言能力改善。这些负结果限制明确主张，不抹去可复用响应、原生计算账本和真实学习数据。

从智能理论看，至少要同时刻画“什么关系发生”“保留了哪些可被未来查询的有序历史”“条件如何把历史编译为输出”。单个压缩状态、一个余弦不变量或一个标量参数不能自动承担三者。下一候选理论应先明确对象、带方向/角色的运算和组合条件，再证明它对未见前缀的增量预测优于身份、位置、范数和平方强基线；若连这个增量都没有，就不该只增加图谱数量。

下一大任务是自然事件流中的可识别有序来源更新，而非再孤立测试一个模板。以下为待执行计划，不冒充本轮成果：

1. 以跨句事件、施受角色、否定和指代建立来源对，预先冻结关系组合与身份/位置/长度控制，增加独立文档；保持完整坐标及原生token跨度，避免语义组与token唯一绑定。
2. 同批比较有序来源更新、查询条件Tq/T^Tq、简单平方和有限矩；显式保存来源身份并报告内存、时间和可查询范围。先问哪些关系信息必要且有用，不预设精确小状态压缩。
3. 训练输入仅用当前可见前缀，同时预测多后层全坐标及固定未来probe分布；后层state/未来token/gold标签只评分。冻结完整自然确认，检验跨事件而非仅随机样本泛化。
4. 将胜过强基线的同一规则与真实多步原生参数继续训练连接，至少两种批次顺序，对自然首token、完整答案、停止、格式分别报告。只有该链共同成立，才提升为新的机制拼图，而非单靠训练loss下降。

本轮未提出超出现有数学工具的新运算或可证明的普遍命题；有限Prouhet反例和算子Gram转置审查是具体边界进展，不是“AI万有引力已经发现”。不从本模型结果直接推导脑可塑性、意识或AGI。

### C008：证据审计、客户端、资源与保留

所有脚本在tests/glm5，临时环境在tests/glm5_temp，全部新结果在本批目录。原MEMO前缀SHA核对、旧冻结证据/材料/训练方向/核及所有模型原始shard和tokenizer配置核对通过；所有保存NPZ逐数组检查数值有限和哈希，客户端注册覆盖全部归档。下表为实际审计值，不以“全量”掩盖未归档的全层全token范围。
'''
    txt+=table(['审计对象','数量或状态'],[[k,science[k]] for k in ('client_material_ids','new_native_Q4_materials','alltoken_alllayer_coordinate_values_scanned_at_capture',
      'committed_native_archive_hash_checks','all_npz_files','all_arrays','all_array_elements_checked','independent_native_B1_shape_controls','language_identity_position_controls','formal_trajectory_records')])
    txt+=f"\n保留的完整全部层×全部token fixture数：{len(science['full_layer_all_token_fixtures'])}。其余全部层全token只保留扫描身份/汇总与重算输入，不声称永久保存全原场。所有已归档数组均可在`/rdc-update`按area/file/array分页查原数值，或用于后续研究，因此用户的条件清理要求不触发；没有删除旧数据、用户文件或模型权重。\n"
    txt+=r'''
客户端新增全坐标热图、来源图、标量路径、全参数梯度、12组语言相似矩阵、同历史和自身历史、最长1024token轨迹及理论账本；图坐标与汇总方式明确标识。静态图不使用Top-K删轴，原值与RMS/asinh显示分开，asinh只是色标不是改动分析值。生产构建、只读API、独立headless Edge桌面/窄屏回归与逐图视觉检查单独保存验证回执。用户现有浏览器CUA初始化失败，不能把独立测试浏览器的成功写成已控制用户浏览器。

预算延续冻结资源门：结果上限12GiB、剩余磁盘保留12GiB、计时脚本总上限21600秒、单模型进程7200秒；实际计时与失败恢复在compute_ledger，最终bytes/耗时与完整交付状态由delivery_manifest和verification/final记录。它不是精确GPU忙时，也不包含实现/浏览器全部墙钟时间。未以删除原始证据腾挪预算。

### C009：相关文件、复算与结论

脚本`phase2739_rdc_update_fresh.py`、`phase2739_rdc_update_moment_boundary.py`、`phase2739_rdc_update_predictive_state.py`、`phase2739_rdc_update_long_answers.py`、`phase2739_rdc_update_behavior_analysis.py`、`phase2739_rdc_update_manual_terminal.py`、`phase2739_rdc_update_integrity.py`、`phase2739_rdc_update_fingerprints.py`、`phase2739_rdc_update_theory.py`、`phase2739_rdc_update_delivery.py`均位于tests/glm5。执行前应先读取已完成commits和资源门，避免重复模型加载和覆盖运行身份。

'''
    txt+='\n下一阶段目标仍相同，2739已实际自动续研；再下一完整自然事件阶段的准入评估如下。参考本轮已执行的128新自然窗口、四条32步训练、三模型各128采集/36生成及Q4批量对照，按实际时间和保留体积加20%工程余量估计。它不是未来运行的精确预测或物理下界，不声称一点小试运行也做不了；也不以剩余少量预算再重复同类模板。\n'
    txt+=table(['下一完整阶段资源门','实际测量或参考估计'],[
      ['目标相同',admission['same_long_term_goal']],
      ['已执行同目标自动续研',str(admission['executed_automatic_same_goal_followup'])],
      ['已计时脚本秒',admission['current_booked_seconds']],
      ['预留最终审计后可用秒',admission['remaining_seconds_after_finalization_reserve']],
      ['相当规模下一完整阶段参考秒',admission['estimated_full_stage_seconds']],
      ['预留最终交付后结果可用GiB',admission['remaining_bytes_after_finalization_reserve']/2**30],
      ['相当规模下一完整阶段参考新增GiB',admission['estimated_full_stage_additional_bytes']/2**30],
      ['时间门／存储门／完整阶段获准',str([admission['compute_admitted'],admission['storage_admitted'],admission['full_next_stage_admitted']])]])
    txt+='\n当前完整下一阶段未通过本轮资源门，因此保存新自然事件／身份位置控制／仅前缀预测／实际训练及三模型确认的大协议，未把它写成已执行。不是语言研究已解决，也不是预算每一秒都已耗尽。当前字段是可查询证据，不为强行追加一轮删除它们；后续需明确新阶段可用预算与保留方案。\n'
    txt+=reference(required+['fresh_graph/material_audit.json','language_identity/result.json','scale_analysis/result.json','figures/index.json','compute_ledger.json'])
    txt+='\n本Phase结论：获得新的未见自然来源控制、历史信息边界、可查询未来诊断和长答案评分，以及可追溯全坐标/参数/训练/生成的统一证据入口；仍未破解任一种语言关系的唯一、可组合且完整跨层接续机制。应保留真实结构，停止过度命名，把下一阶段解释空间压缩到“身份控制下仍有增量的有序来源更新与学习形成”。\n'
    return txt


def main(phase):
    memo=MEMO.read_text(encoding='utf-8');numbers=[int(x) for x in re.findall(r'^## Phase (\d+):',memo,re.M)]
    assert numbers[-1]==phase-1 and phase not in numbers,(numbers[-1],phase)
    title={2738:'多语言模式族、原生来源账本与自身历史的非量化三模型复查',2739:'同目标自动续研：强控制、预测状态信息边界与长答案证据整合'}[phase]
    section=f"\n\n## Phase {phase}: {title} [{stamp()[:16].replace('T',' ')}]\n"+(phase38() if phase==2738 else phase39())
    tail=memo.splitlines()[-4:]
    patch='*** Begin Patch\n*** Update File: '+MEMO.as_posix()+'\n@@\n'+'\n'.join(' '+line for line in tail)+'\n'+'\n'.join('+'+line for line in section.splitlines())+'\n*** End Patch'
    print(json.dumps({'phase':phase,'memo_bytes_before':MEMO.stat().st_size,'memo_sha_before':sha(MEMO),'patch':patch},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',type=int,choices=[2738,2739],required=True);main(p.parse_args().phase)
