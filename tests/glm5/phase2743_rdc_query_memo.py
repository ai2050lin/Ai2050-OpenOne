"""Render append text to stdout; the caller applies it with apply_patch only."""
import argparse,re
from rdc_query_common import *


def number(v):
    if isinstance(v,float):return f'{v:.6g}'
    return str(v)


def ci(v):return f"{v['mean']:.6g} [{v['interval95'][0]:.6g}, {v['interval95'][1]:.6g}]"


def paragraph(value):
    return value.replace(r'\n\n','\n\n').replace(r'$$\n','$$\n').replace(r'\n$$','\n$$')


def table(headers,rows):
    clean=lambda x:str(x).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(clean(number(x)) for x in row)+' |' for row in rows])+'\n'


def phase2742():
    a=read(BASE/'analysis/phase2742.json');assert a['all_passed'];f=a['formation'];parts=[]
    parts.append(paragraph(r"""### C001：共同问题、原理和真实学习形成\n\n状态：已执行。运行根目录仍为tests/glm5/result/rdc_query_campaign_20260913；本Phase合并原生中层继续训练、跨表达注入、五路晚期类别偏置和三模型原始精度查询核对。共同问题是：可观察的响应与来源关系，是否在真实参数更新及自身历史生成中带来可区分的效果，而不把格式、概率校准或执行形状当成语义机制。\n\n训练池为288条train自然前缀的两个实际内容位置，共576条；输入在内容位置截断，紧接的真实token只作监督，不作为输入。两个种子2742/2743，每次32步、每步4个样本的梯度均值，真实标签与同cohort内置乱标签使用相同抽样序列。每条训练抽128次，不是把576条都训练一遍。全部74711040个block16 gate/up/down标量参与更新，其他权重固定；真实17—35块和完整词表仍参与反传。\n\n$$\ng_t=\frac14\sum_{i\in B_t}\nabla_\theta[-\log p_\theta(x_{n_i+1}\mid x_{0:n_i})],\qquad\theta_{t+1}=\theta_t-0.02\frac{g_t}{\|g_t\|_2}.\n$$\n\nθ仅指该中层MLP真实权重；trainable块为FP32、边界转BF16，原始BF16基线、FP32桥接基线及最终实际BF16部署分别测量。更新后保留完整FP32参数增量，并复原桥接前4例精确核对。不是修改原始checkpoint，不是恢复预训练历史，也不是只拟合外部回归器。"""))
    parts.append(table(['条件','抽样种子','128次中不同样本数'],[[r['condition'],r['seed'],r['examples']] for r in f['distinct_drawn_examples_per_run']]))
    parts.append(table(['条件','抽样种子','累计FP32位移范数','实际BF16部署位移范数'],[[r['condition'],r['seed'],r['delta_FP32_norm'],r['delta_native_BF16_norm']] for r in f['actual_parameter_displacements']]))
    parts.append('重要强对照边界：每步都归一化到0.02，不代表32步向量和或BF16部署后的总位移相同。置乱与真实标签组的梯度方向相干性、累计位移和舍入结果均可不同；本轮没有做累计位移范数匹配插值。因此真实−置乱差不能唯一归因于标签语义，也不能据此宣称随机监督本质更优。')
    parts.append('测试面板为384个自然留出内容位置（192前缀、170文档），另有30个旧真实生成的终答数字位置。后者取最后显式终答标记后的首个数字，不按是否答对选；2条无合格标记的旧轨迹显式排除。30个位置来自8个语义组，不作为30次新独立推理确认。1/8/32步与最终BF16均评价，24条预定自然例保存全部gate/up/activation单元及末端坐标。')
    parts.append('### C002：学习结果与目标打乱这一关键强对照\n\n下表自然NLL均以来源簇为统计单位；区间为条件于当前模型、面板、抽样顺序的95%簇bootstrap。1/8/32步对应FP32中层桥接基线，deployed_BF16对应原始BF16基线；不能跨数值基线混读。real−permuted为正，表示真实标签更新在此指标上反而较差。')
    rr=[r for r in f['paired_reports'] if r['kind']=='natural_content' and r['cohort']=='all']
    parts.append(table(['种子','检查点','真实−对应数值基线NLL','置乱−对应数值基线NLL','真实−置乱NLL','真实/置乱argmax'],[[r['seed'],r['checkpoint'],ci(r['natural_minus_baseline_NLL']),ci(r['permuted_minus_baseline_NLL']),ci(r['natural_minus_permuted_NLL']),f"{r['natural_argmax_accuracy']:.6f}/{r['permuted_argmax_accuracy']:.6f}"] for r in rr]))
    parts.append('真实标签训练的自然NLL下降值得保留，但置乱标签下降更大，不能把这种下降唯一解释成关系齿轮的学习形成。下一阶段应将温度、词频先验、置信度与数值分辨率等简单解释纳入同批对照。该结果也不能外推为置乱监督一般优于真实监督。终答位置的逐语种指标保存在analysis/phase2742.json：高基线、样本少、历史暴露，且并没有让训练后的模型重新自由生成整条推理链，所以不据此声称推理链形成。')
    parts.append('### C003：跨表达响应的一次性输出注入\n\n从Phase2741的32个mixed-holdout英文程序中，各运行native、code_identity、mapped_code三分支。固定query为编号40的Answer后缀；后两者读取对应Python表达的已观测查询响应，将全2560维向量一次性送入unembedding，之后完全按各自新token历史生成，最长1024token。不是移植完整KV/推理状态，也不是给定正确答案。Python→EN映射按预先指定的test×unseen-query词表门未通过，因此明示为失败映射压力测试，不称验证修复。')
    rr=[r for r in a['injection'] if r['representation']=='all']
    parts.append(table(['分支','表达','正确且EOS','已解析','未解析EOS','截断','明确解析错误','平均token','相对native成功率变化'],[[r['branch'],r['expressions'],r['correct_and_stopped'],r['parsed'],r['unparsed_EOS'],r['censored'],r['parsed_wrong'],r['mean_tokens'],ci(r['paired_success_delta_vs_native'])] for r in rr]))
    matches=0
    for p in (BASE/'transfer/injection/commits/native').glob('*.json'):
        if read(p)['generated_ids']==read(BASE/'transfer/injection/commits/mapped_code'/p.name)['generated_ids']:matches+=1
    parts.append(f'映射分支有{matches}/32条完整生成ID序列与native相同；相同评分汇总不自动当作序列相同，本项已逐ID核对。三个分支在初始替换前后当前完整KV均不变。无干预收益只限制当前全向量映射与一次性读出，不否定所有可能的跨表达关系。')
    parts.append(paragraph(r"""### C004：晚期偏置、格式与未来历史\n\n在未参与旧32条长生成的24个语义组、96个四表达程序上，固定native、step128_digit、entropy_digit、entropy_letter、terminal_marker_digit五分支，每条最长1024token。旧轨迹校准的高熵阈值只作为冻结触发器，不命名为“废话检测”。数字偏置一次性给全部1—8 token加8；字母对照给全部A—H加8，不输入gold，不强制EOS，不改变当前数字内部argmax。\n\n$$\nz_v^{\prime}=z_v+b\mathbf1[v\in D],\quad p_D^{\prime}=\frac{e^bp_D}{1-p_D+e^bp_D},\quad p^{\prime}(v\mid D)=p(v\mid D)\ (v\in D).\n$$\n\n此为实数概率恒等式。D是声明的8个数字类别，不是通用内容子空间；高数字总质量与数字条件竞争分开。修改已计算logits不会回溯改变当前KV，但下一个被选token不同后未来KV可以分叉。不能把“同历史读出隔离”升级成“任意历史语义绝对隔离”。\n\n先导B1实际计时后，正式统一使用固定B8分组、左padding、显式attention mask和各行position IDs。结束行使用掩码dummy而不再记录token；没有混入其他行的历史。正式五分支的相同shape对照保持一致，同时另报2条B1/B8差异。"""))
    rr=[r for r in a['late'] if r['representation']=='all']
    parts.append(table(['分支','正确且EOS/96','未解析EOS','截断','明确解析错','平均token','触发数','下步KV比较/改变','成功率差95%区间'],[[r['branch'],r['correct_and_stopped'],r['unparsed_EOS'],r['censored'],r['parsed_wrong'],r['mean_tokens'],r['triggered'],f"{r['next_KV_compared']}/{r['next_KV_changed']}",ci(r['paired_success_delta_vs_native'])] for r in rr]))
    parts.append(table(['B1/B8独立执行形状核对样本','首次生成分叉位置','B1/B8 token数','首postnorm MSE'],[[r['sample_id'],r['B1_B8_generated_prefix_first_divergence'],f"{r['B1_steps']}/{r['B8_steps']}",r['first_postnorm_MSE']] for r in a['native_batch_shape_controls']]))
    parts.append('所有实际轨迹保留原始ID、文本、原评分、前轮格式评分和本轮生成前冻结的扩展评分。评分覆盖提升另记，不算模型能力提升；未解析、上限截断和明确解析错误不合并。完整推理文本未逐步判真。未触发分支必须逐ID等于相同B8 native；触发前不得提前分叉；当前有效KV逐字节核对，padding槽不当作真实来源。next-KV只在双方均有相应实际下一步记录时比较，不能把缺失计为未改变。')
    example=gzread(BASE/'late/material.json.gz')[0]
    parts.append('按冻结材料顺序的首条实际用例（不是按成功选择）：'+example['sample_id']+'，语义组'+example['source_group']+'。\n\n```text\n'+example['original_text']+'\n```\n\n程序规则给定终值：'+str(example['target'])+'。五个实际分支的结果如下：')
    actual=[read(BASE/'late'/b/'commits'/f"{example['sample_id']}.json") for b in ['native','fixed128_digit','entropy_digit','entropy_letter','terminal_marker_digit']]
    parts.append(table(['实际分支','生成token','保守终答','正确且EOS','首次相对native分叉'],[[r['branch'],len(r['generated_ids']),r['answer_scoring']['conservative_final_answer'],r['answer_scoring']['parsed_and_stopped_correct'],r.get('first_token_divergence_from_native')] for r in actual]))
    parts.append('该例原生实际完整输出（忠实记录，文本推理未另作逐步正确性证明）：\n\n```text\n'+actual[0]['generated_text']+'\n```')
    parts.append('### C005：原始精度的三模型自身坐标核对\n\n顺序运行本地Qwen3-4B、Qwen3-14B、GLM4，均BF16无量化。大模型按实际host/Windows commit余量使用device_map=auto及CPU/原checkpoint引用卸载，不复制或改写权重。前3个预定平衡来源仅按耗时选择64/32/16完整100-query面板；跨模型统计只用共同文本交集。每个模型独立分词、独立坐标、独立Q/K位置处理，GLM保留其实际部分/交错RoPE。')
    scale_rows=[]
    for m in ['qwen4','qwen14','glm4']:
        r=read(BASE/'scale'/m/'result.json');scale_rows.append([m,r['sources'],r['queries'],r['width'],r['depth'],max(x['max_abs_error'] for x in r['native_QK_RoPE_attention_checks']),r['seconds']])
    parts.append(table(['原生模型','来源数','查询端点','原生宽度','块数','Q/K/RoPE注意力最大误差','实测秒'],scale_rows))
    parts.append(f"共同来源交集为{a['scale']['matched_documents']}个文档。全部100查询和每个模型全部坐标构造各自中心化Gram，以下是描述性比较，不进行跨模型原生坐标逐一对齐：")
    parts.append(table(['模型对','中心化Gram非对角条目相关','单位Frobenius矩阵距离'],[[str(r['models']),r['centered_Gram_entry_correlation'],r['unit_Frobenius_Gram_difference']] for r in a['scale']['descriptive_between_model_Gram_comparisons']]))
    parts.append('矩阵条目相互依赖，不是4950次独立试验；不论相关高低，都不证明所有条件同构成立或不存在，也不能将架构、tokenizer、训练、规模与执行驻留差异单独归因于模型大小。Q4的重复采集与主图谱逐位核对，另外两个模型没有借用Q4坐标或token ID当作原生结果。')
    qc=read(BASE/'analysis/query_identity_control.json')
    parts.append(paragraph(r'''观察到高相关后，追加了同16文档的探索性query身份对照；这是结果后的分析选择，不是新独立确认。令H_s∈R^{100×D}为历史s的查询响应、H_0为独立查询响应，C=I−11^T/100为查询中心化矩阵：

$$
K_{\mathrm{native}}=\frac1{ND}\sum_s(CH_s)(CH_s)^T,\quad K_{\mathrm{alone}}=\frac1D(CH_0)(CH_0)^T,
$$
$$
K_{\mathrm{change}}=\frac1{ND}\sum_s C(H_s-H_0)[C(H_s-H_0)]^T,\quad K_{\mathrm{between}}=\frac1{ND}\sum_s C(H_s-\overline H)[C(H_s-\overline H)]^T.
$$

N=16，D为各自原生宽度，平均H仅对这16个历史取平均。它们是基本统计定义：独立查询、接入历史的差异，以及这批历史之间的差异，不是因果语义分解，也没有把差异向量搬运到另一条历史。全部坐标和100查询保留，三个原主Gram被重新计算且误差均为0。'''))
    parts.append(table(['统计对象','模型对','非对角条目相关','单位Frobenius距离'],[[kind,str(r['models']),r['entry_correlation'],r['unit_Frobenius_difference']] for kind,rr in qc['between_model_comparisons'].items() for r in rr]))
    parts.append('移除独立查询向量后，相关仍为0.791—0.890；移除跨历史共同查询profile后，历史间变化的相关为0.852—0.941。因此不接受“原高相关已经证明语义同构”，也不能反向声称“所有相关都只是独立查询文字”。共有任务/语言形式、窗口选择和共享外部统计仍可解释部分一致性。实际完整矩阵在analysis/full_query_identity_control_grams.npz，方法及数值在analysis/query_identity_control.json，脚本phase2742_rdc_query_query_identity_control.py。')
    parts.append('### C006：五环对应、理论、硬伤和可恢复下一步\n\n共同现象是实际参数更新与读出偏置都能改变指标，但内容、格式、停止、概率置信度和未来历史不是同一对象；候选规律是原生下一token学习及有明确输入边界的无gold类别/映射操作；原生结构为全部中层参数、真实上层计算与有效KV；未见检查由自然测试文档、24个新长生成语义组和三模型共同文本限定；训练形成来自四条真正执行的32步权重轨迹，不从局部成功还原预训练历史。\n\n三图谱新增真实参数训练与自然内容/真实终答位置的对应、五类自身历史策略和独立原生模型空间。RDC主公式没有变更；没有得到新普遍数学定律。最重要的新限制是训练NLL收益必须与目标打乱和概率校准竞争，局部读出改变还必须按自身历史而非固定历史验证。小样本或某个干预门失败不终止观察路线，但也不能越过这些强对照宣布已闭合。\n\n主要脚本为phase2742_rdc_query_formation.py、phase2742_rdc_query_injection.py、phase2742_rdc_query_late.py、phase2742_rdc_query_late_batch.py、phase2742_rdc_query_scale.py、phase2742_rdc_query_analysis.py及rdc_query_scoring.py，均在tests/glm5。实际产物为formation/result.json及四组完整参数增量、transfer/injection/result.json、late/*/result.json、scale/*/result.json、analysis/phase2742.json与figures/index.json。逐模型/逐分支用时与失败恢复记录在compute_ledger和science_queue；数据可从/rdc-query回查，无原场或原模型删除。\n\n按同一已授权目标，2743继续使用预留的96个自然文档，保持五个预测器不变，并以明确增加“实际查询H12”的诊断来检验遗漏查询构造信息的影响。之后再按实际剩余资源决定完整的严格身份/概率校准阶段；计划与实际执行分开登记。')
    return '\n\n'.join(parts)


def phase2743():
    a=read(BASE/'followup/result.json');t=read(BASE/'theory_snapshot.json');assert a['all_passed'] and len(t['puzzles'])==42
    r=next(r for r in a['summary'] if r['query_split']=='unseen_query' and r['cohort']=='all');parts=[]
    parts.append(paragraph(r"""### C001：同目标自动续研与明确的信息边界\n\n状态：独立确认已执行。96个预留来源文档（GUM/EWT/CMRC各32）与本轮10000条主前缀的2777文档不重叠；共9600固定查询端点。材料、代码和主预测器SHA在新采集前冻结，五个原预测器未改系数。此为相对本轮主库的文档独立，不宣称无历史项目暴露或无预训练接触。\n\n同时拟合一个单独的四列诊断[1,前缀H12,独立query H13,实际接入前缀后的query H12]，仅使用原train-source×train-query拟合、validation选λ。它输入了更晚实际观察到的query H12，因此只能诊断后层预测的信息缺口，不能冒充此前缀可用信息的同输入胜出者。目标H24/H36/postnorm仍不作为该诊断输入。\n\n$$\n\widehat h_{\ell,d}=\beta_{0,\ell,d}+\beta_{1,\ell,d}H_{12,d}(S)+\beta_{2,\ell,d}H_{13,d}(q\text{ alone})+\beta_{3,\ell,d}H_{12,d}(S,q).\n$$\n\n最后一项是额外观察信息。这一公式是受限拟合，不是从外部语言规则直接生成HiddenState的闭合方程。全部原生坐标、选定多层状态、前缀KV和输入保留，候选特征除3条固定fixture外流式计算，完整原权重及重算入口保留。"""))
    parts[0]=parts[0].replace('（GUM/EWT/CMRC各32）','（GUM19、EWT39、CMRC38）')
    parts.append('材料计划修正：初始冻结检查失败，因400个GUM预留窗口只来自19个文档，不能当作32个独立文档。失败发生在任何本Phase模型计算之前，没有产生确认输出。保留同一预留池、文档不重叠和总数96，将缺少13个名额按固定EWT/CMRC顺序分配为39/38；没有用新结果决定名额。原失败源码、父进程实测7.395918秒与回执保存在followup_recovery/material_freeze_failure.json及science_queue历史，失败时间已入账。以下总体均值对应实际19/39/38文档混合，不再称三类各32；分语料结果同时报告。')
    parts.append(table(['候选/信息条件','未见查询postnorm MSE [95%]','完整词表KL [95%]'],[[name,ci(r['postnorm_MSE'][name]),ci(r['full_vocab_KL'][name])] for name in r['postnorm_MSE']]))
    parts.append(table(['相同输入控制−有序的MSE','文档簇均值 [95%]'],[[k,ci(v)] for k,v in r['same_input_control_minus_ordered_MSE'].items()]))
    parts.append('增加实际query H12信息后相对有序规则的MSE减少：'+ci(r['additional_query_information_MSE_reduction'])+'。无论正负，只在这一明确的信息条件下解释；不能把更晚观察量暗中提前提供。各cohort结果在followup/result.json完整保留，不用总体均值宣称每一类成立。')
    parts.append('独立确认的关键限制：未见查询上，有序规则相对query-only、uniform、quadratic的配对MSE优势区间为正，但相对shuffled-values的区间跨过零。故不能把主实验的所有状态优势视为已经独立复现，更不能据此识别独有的语言关系机制。完整词表KL与状态MSE仍是分别衡量的目标；额外query H12诊断的改善揭示信息条件的重要性，而不唯一确定剩余误差的内部原因。')
    example=gzread(BASE/'followup/material.json.gz')[0];probe=read(BASE/'probes/protocol.json')['probes'][99]
    parts.append('预定顺序首条确认用例：'+example['sample_id']+'，来源文档'+example['source_group']+'；完整实际前缀与第99号查询分别为：\n\n```text\n'+example['text']+'\n```\n\n```text\n'+probe['text']+'\n```\n\n这是已知查询接入真实前缀后的状态/分布预测测试，不是要求query文字本身构成某个标准答案。前缀原生ID、查询ID、全部100响应和同来源六候选误差均在该sample的followup/capture文件中，可以按第99行和全部坐标原序回查。')
    parts.append(paragraph(r"""### C002：当前RDC统一接口与全局图谱核心公式\n\nRDC仍称“条件化输出场闭合理论”，但闭合是待验证方向，不是既成全局定理。历史主体对象是特定域的析因响应束，不把它无声替换为任意HiddenState：\n\n$$\nX_\ell(p)=\{H_{\ell,t,j}(p)\},\qquad\mathcal T^{\mathcal D}_{\ell,\tau}:(\mathcal L(p),\mathbf W_\ell(p),X_\ell(p))\dashrightarrow\mathbf W_{\ell+1}(p).\n$$\n\np为材料，L为外部描述，W为历史类型化响应束，τ为条件/角色，D限定已测试域，虚箭头表示部分接口而非全局可执行映射。原生Transformer计算作为核对对象：\n\n$$\nr_\ell=H_\ell+A_\ell(N_\ell(H_\ell),KV_\ell,\mathrm{pos}),\quad x_\ell=N'_\ell(r_\ell),\quad H_{\ell+1}=r_\ell+W_{d,\ell}[\mathrm{SiLU}(W_{g,\ell}x_\ell)\odot W_{u,\ell}x_\ell],\quad p_{\mathrm{next}}=\mathrm{softmax}(W_U N_f(H_L)_{\mathrm{query}}).\n$$\n\n这是架构定义，不是新发现的语言定律。三图谱的已有统一登记对象保持：\n\n$$\n\mathrm{Atlas}_{\mathcal D}=(G_{\mathrm{external}},G_{\mathrm{internal}},E_{\mathrm{association}}^{\mathcal D}).\n$$\n\n外图保存来源、token跨度、语言族及类型化外部关系；内图索引(model,run,prefix,step,layer,position,coordinate/unit/parameter,boundary)；关联边记录(condition,algorithm,inputscope,fit,heldoutprediction,evidence,status)。这是可追溯的类型化账本，不是已经确定了图布局距离对应真实神经几何。本轮主公式没有普遍性改进；新增的有序账本、有限查询与学习/概率控制使用既有代数、微积分和统计工具。"""))
    lc=read(BASE/'analysis/query_language_control.json')
    parts.append(paragraph(r'''### C002补充：对2742中英文分块图的语言内部核对

在实际查看2742三模型图后，发现中英文50/50查询存在明显分块，因此追加同16文档、各语言内重新中心化的探索性对照。仍保留每个模型全部坐标与每种语言全部50查询；不是新独立确认。令g(q)为查询语言：

$$
B_{s,q}=\overline H_{s,g(q)}-\overline H_s,\qquad W_{s,q}=H_{s,q}-\overline H_{s,g(q)},\qquad \sum_{s,q}\|H_{s,q}-\overline H_s\|^2=\sum_{s,q}\|B_{s,q}\|^2+\sum_{s,q}\|W_{s,q}\|^2.
$$

这仅是标量平方能量分解；完整Gram还含交叉项，不能声称两个Gram直接相加就是原Gram。能量比例也不等于语言的因果贡献。'''))
    parts.append(table(['模型','语言组间平方能量比例','语言内部比例'],[[r['model'],r['between_language_fraction_of_query_centered_coordinate_energy'],r['within_language_fraction']] for r in lc['energy']]))
    parts.append(table(['重新中心化对象','模型对','非对角相关','单位Frobenius距离'],[[kind,str(r['models']),r['entry_correlation'],r['unit_Frobenius_difference']] for kind,rr in lc['within_language_comparisons'].items() for r in rr]))
    parts.append('原生响应的语言内相关：英文0.642—0.855、中文0.845—0.894，明显不能直接以混合100查询约0.97概括。历史间变化的语言内相关仍为英文0.571—0.856、中文0.824—0.882。保留有限、条件化的相似结构，不把它消解为“全是语言标签”，也不命名为语义同构。材料只有16共同文档，查询共享形式和意图、语言与tokenizer、通用任务模式等混杂仍未完全分离。脚本phase2742_rdc_query_language_control.py在本Phase作补充审阅执行，原始2742结果没有改写；实际矩阵为analysis/full_within_language_grams.npz，指标和设计状态为analysis/query_language_control.json。')
    parts.append('### C003：完整核心机制拼图清单\n\n下表继承38项既有拼图并增加本轮4项，共42项。历史条目没有全部重跑；保留原适用范围、来源与修正边界，不把不同证据等级并称“已破解”。')
    parts.append(table(['Phase','保留拼图','边界','当前证据状态'],[[p['phase'],p['retained_puzzle'],p['boundary'],p.get('evidence_status','inherited')] for p in t['puzzles']]))
    parts.append('### C004：全部核心公式账本\n\n以下40项包括31项继承公式与9项本轮新增可执行表述。原表达、变量、证据类型和范围逐条保留；代码式表达用于可重算性，不临时改名为新数学定理。关键RDC/原生/图谱式已在上节以数学形式列出。')
    for i,f in enumerate(t['formulas'],1):
        parts.append(f"#### 公式账本{i}：{f['id']}\n\n类型：`{f['kind']}`。\n\n```text\n{f['expression']}\n```\n\n变量与可计算对象：{f['variables']}\n\n证据及边界：{f['evidence']}")
    parts.append('### C005：三图谱与第一性原理更新\n\n'+t['first_principles_insight']+'\n\n'+t['new_mathematical_increment']+'\n\n完整图谱中，外部关系注释、原生计算分解、预测增量、真实参数变化、自然生成成功是相互关联但不能互相替代的证据。有限100query不是全未来量词；拥有全部坐标也不等于已经恢复了来源顺序、关系身份和上下文如何构造query。当前仍缺一个在严格身份/位置控制下、能跨语言族解释并预测完整输出与学习效应的条件更新规律。')
    parts.append('### C006：实际交付、资源和下一完整阶段\n\n本Phase新增96文档/9600端点、原五预测器不变核对、额外信息诊断及客户端理论账本。公共入口/rdc-query以稳定model/sample/query/layer/step ID回查完整字段、真实来源参数、原生轨迹和所有NPZ。原始值与RMS/asinh显示分离，不用Top-K/PCA定义主干；只将预定9个主前缀保存为全部token全部层fixture，其他范围明确保留锚点或重算依据。\n\n用户浏览器CUA初始化失败的回执保留；成功的是独立headless Edge中的本项目页面，不冒充已控制用户浏览器。只读API、代码构建、完整数组审计、原checkpoint/既有证据SHA和逐图视觉检查分别记录；最终跨阶段回执将统一汇总，不能把脚本returncode当作科学假设全部通过。\n\n脚本phase2743_rdc_query_followup.py、phase2743_rdc_query_integrity.py、phase2743_rdc_query_fingerprints.py、phase2743_rdc_query_theory.py、phase2743_rdc_query_figures.py及客户端回归脚本均在tests/glm5。产物为followup/protocol.json、followup/result.json、followup/oracle/result.json、theory_snapshot.json、verification、client和figures。\n\n下一完整阶段仍服务同一目标，具体针对已暴露的两个可识别性缺口：严格相同token多重集但关系相反的五族对照，以及真实/置乱训练收益与简单概率校准的竞争。材料预检与实现不冒充模型执行；实际是否准入、320表达/160配对/80组的完整测试及余量由next_stage_admission.json记录。只有完整阶段通过既定21600计时秒、12GiB结果和12GiB磁盘保留门才自动继续，不删除本轮仍可查询的关键证据来强行循环。\n\n通俗结论：本轮把问题从“图很像、损失降低”推进到“相同信息下究竟多预测了什么”。更早状态、实际query构造、输出概率和学习形成必须分开核对；已有规律被保留，过度结论被压缩，但仍没有破解完整语言编码机制。')
    return '\n\n'.join(parts)


def main(phase):
    text=MEMO.read_text(encoding='utf-8');phases=[int(v) for v in re.findall(r'^## Phase (\d+):',text,re.M)]
    assert phases[-1]==phase-1,(phase,phases[-1]);anchor=text.rstrip('\r\n').splitlines()[-1]
    title={2742:'实际自然内容学习、无gold自身历史与三个原始模型',2743:'独立查询确认、完整机制拼图与同目标续研边界'}[phase]
    body={2742:phase2742,2743:phase2743}[phase]()
    append=f"\n\n## Phase {phase}: {title} [{stamp()[:16].replace('T',' ')}]\n\n{body}\n"
    print(json.dumps({'phase':phase,'anchor':anchor,'append':append},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',type=int,choices=[2742,2743]);a=p.parse_args();main(a.phase)
