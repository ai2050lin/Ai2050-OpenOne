"""Render the completed continuation narrative; caller appends with apply_patch."""
import re
from phase2744_rdc_query_identifiability import *
from phase2743_rdc_query_memo import table,ci,paragraph


def main():
    assert read(OUT/'queue/status.json')['all_passed'];a=read(OUT/'analysis/result.json');t=read(BASE/'theory_snapshot.json')
    assert len(t['puzzles'])==43 and len(t['formulas'])==42
    text=MEMO.read_text(encoding='utf-8');assert int(re.findall(r'^## Phase (\d+):',text,re.M)[-1])==2743
    parts=[paragraph(r'''### C001：同目标自动准入、共同问题与材料

状态：已执行。2743结束后按实际计时和结果大小检查完整阶段余量，准入记录为next_stage_admission.json；本Phase不是预先把计划写成结果。目标是区分三个容易混在一起的现象：词汇身份引起的响应、顺序/关系变化引起的响应，以及训练后概率置信度改变引起的NLL收益。

冻结320个实际表达、160个严格配对、80个语义组，覆盖属性绑定、否定范围、双语境词义、长距离角色和虚构知识链，中英文各半。每对完整实际prompt的token多重集完全相同，问题相同，外部规则推导的答案相反。并非自然语料，且使用了旧阶段暴露过的姓名/词义语境成分，不声称未见词汇。配对是新的关系组合；每族16个case，每个族内case的两个语言、两个世界共享组ID，共5×16=80组。bootstrap以80个语义组而非320条表达为单位。

例如属性绑定只交换两个人对红/青苹果的所有关系；长距离角色交换首次包裹交接角色，但保留后面的杯子交接；词义材料同时保留水果和公司语境，交换A/B并固定询问A。知识链将item→fruit倒转为fruit→item，并明确“否”指不能仅凭规则推出，不把缺乏蕴含偷换成现实中为假。所有160对实际token计数逐项通过，无不匹配后的弱化回退。

完整原生Qwen3-4B保留每条100query×2560坐标、37层末端锚点；四个已训练的BF16参数变体只测预定6query，原生也另做相同6query执行形状作为比较。全部320×5条件保存block16/35各9728个gate、up和activation单元，不做Top-K或PCA。实际旧训练的74711040标量增量重新部署，每个变体在两个采集阶段都核对原形成面板的前4个NLL；本阶段没有新增训练，也没有写回原始checkpoint。''')]
    parts.append(paragraph(r'''### C002：严格词袋身份下的关系响应

$$
S_{AB}=(2y_A-1)\left[p(\mathrm{Yes}\mid A,\{\mathrm{Yes},\mathrm{No}\})-p(\mathrm{Yes}\mid B,\{\mathrm{Yes},\mathrm{No}\})\right].
$$

A、B为外部答案相反的成对prompt，y_A为配方给定的真值；中文对应“是/否”。这是答案方向一致的条件概率分离度定义。另记二候选在完整词表的总质量与自由生成，不能把两候选条件概率冒充完整输出概率。下表只取未更新的原生模型，区间按语义组计算；所有四个部署变体的完整同表在identifiability/analysis/result.json。'''))
    rr=[r for r in a['relations']['matched_pair_reports'] if r['variant']=='native']
    parts.append(table(['关系族','严格配对数','答案方向分离度 [95%]','两边首argmax均正确','6query全坐标位移MSE [95%]'],[[r['family'],r['token_matched_pairs'],ci(r['answer_aligned_yes_probability_separation']),r['both_first_argmax_answers_correct'],ci(r['mean_query_displacement_MSE'])] for r in rr]))
    parts.append('正分离可以反对在这批配对上完全不随顺序变化的纯词袋解释；它不排除位置、局部序列、任务模板或训练词汇的作用。非零响应位移本身也不证明位移编码了正确关系。每个case同时保存反向世界，不能只挑答对的一边。')
    parts.append('实际差异很不均匀：属性绑定分离度约0.999986、首argmax双边正确32/32；否定、词义和长角色分离度分别约0.579、0.544和0.300。知识链虽有0.137344的六query全坐标位移MSE，答案方向分离却仅约1.39×10⁻⁸，首argmax双边正确0/32。这一极小正数不是精确零，但也不是有实际意义的正确关系分离；不能凭其bootstrap区间为正就宣布机制通过。')
    parts.append('### C003：冻结全坐标规则的跨域未见查询预测\n\nPhase2741五个解码器保持不变，不在身份对照上重新拟合。每个预测器只使用其冻结允许的前缀H12、query-alone原生特征和来源KV；不把实际目标层、目标答案或未来token作为输入。下表为20个未见查询的全表达簇均值，完整分族和配对变化预测误差也保留。')
    r=next(r for r in a['relations']['frozen_rule_unseen_query_reports'] if r['family']=='all')
    parts.append(table(['冻结候选','postnorm MSE [95%]','完整词表KL [95%]'],[[n,ci(v),ci(r['KL'][n])] for n,v in r['MSE'].items()]))
    parts.append(table(['控制−有序','MSE差 [95%]','KL差 [95%]'],[[n,ci(v),ci(r['control_minus_ordered_KL'][n])] for n,v in r['control_minus_ordered_MSE'].items()]))
    parts.append('这既是新的受控表达域，也含已暴露材料成分；不能把全部表达独立、全部语言普遍成立或原生自然生成正确视为同一结论。所有坐标纳入比较仍不保证简单的逐坐标四列回归已描述充分的查询构造过程。')
    parts.append('实际总体结果：有序规则MSE8.91764、KL6.74642，相对四个控制的配对差区间均为正；但绝对误差比自然主域更大，且相对query-only的MSE改善只有约0.047。它是这个受控域上的有限预测增量，不等于正确预测了关系差异。')
    control=read(OUT/'analysis/pair_change_control.json');cr=next(r for r in control['reports'] if r['query_split']=='unseen_query' and r['family']=='all')
    parts.append(paragraph(r'''#### C003补充：整体响应拟合与配对变化预测分开

原采集已经保存160个配对的冻结预测变化误差；查看总体结果后，追加不改预测器的零变化基线汇总，设计状态为未盲补充审阅，不冒充新独立实验。全部2560坐标和20个未见query保留：

$$
\Delta h=h_B-h_A,\qquad E_c=\frac1{20D}\sum_{q,d}\left[(\widehat h_{c,B}-\widehat h_{c,A})-\Delta h\right]_{q,d}^{2},\qquad E_0=\frac1{20D}\sum_{q,d}\Delta h_{q,d}^{2}.
$$

D=2560，E₀对应直接预测两边响应不变。差值仅为统计评价对象，不向模型搬运任何激活。'''))
    parts.append('零变化基线MSE：'+ci(cr['zero_change_MSE'])+'。')
    parts.append(table(['候选','配对变化预测MSE [95%]','零变化−候选MSE [95%]'],[[n,ci(v),ci(cr['zero_minus_candidate_change_MSE'][n])] for n,v in cr['candidate_change_MSE'].items()]))
    parts.append('关键结果：有序规则0.394897高于零变化0.394067，零变化−有序为−0.000830815，区间[−0.00106025,−0.000609566]。没有一个候选在总体未见query上稳定胜过零变化。这将上一表的阳性限定为整体状态/分布拟合，未证明提取器抓住同词袋下的关系变化。不能反推原模型没有关系机制；失败的是这一已冻结提取器在明确配对目标上的预测。完整分族、分query和每对结果位于analysis/pair_change_control.json；原analysis/result.json与全部原生输出保持不变。')
    parts.append(paragraph(r'''### C004：真实训练收益与简单概率校准的竞争

在192个原validation内容位置选择标量控制，在96个预留文档的192个新内容位置评价；这些文档在2743用作另一项query任务，因此称新内容位置而非又一套全新文档。内容位置截断后，紧接真实token只用于NLL目标。先验只使用原576个训练位置的token计数，同一先验用于全部变体。

$$
p_{T,\alpha}(v\mid x)=(1-\alpha)\operatorname{softmax}(z(x)/T)_v+\alpha\pi_{\mathrm{train}}(v),\qquad \pi_{\mathrm{train}}(v)=\frac{c_{\mathrm{train}}(v)+1}{576+151936}.
$$

T∈{0.5,0.75,1,1.25,1.5,2,3}，α∈{0,0.01,0.05,0.1,0.2,0.5}；按validation文档等权NLL选择，不查看确认目标决定参数。分别报告温度单独、先验单独和联合控制。以下表中raw差为该变体原NLL减原生原NLL，联合差为两者各自validation校准后的NLL之差。校准是评分端的统计控制，没有用于下节自由生成。'''))
    parts.append(table(['实际参数变体','选定T','选定α'],[[v,r['joint_temperature'],r['joint_prior_mixture']] for v,r in a['calibration']['selections'].items()]))
    rr=[r for r in a['calibration']['prospective_reports'] if r['cohort']=='all']
    parts.append(table(['实际参数变体','原NLL [95%]','校准NLL [95%]','raw差 [95%]','联合差 [95%]','原argmax准确率'],[[r['variant'],ci(r['raw_NLL']),ci(r['joint_calibrated_NLL']),ci(r['raw_minus_original_raw']),ci(r['joint_minus_original_joint']),r['raw_argmax_accuracy']] for r in rr]))
    r=next(r for r in rr if r['variant']=='native')
    parts.append(table(['原生校准−原生raw','NLL变化 [95%]'],[[k,ci(r[k])] for k in ['original_temperature_only_minus_original_raw','original_prior_only_minus_original_raw','original_joint_minus_original_raw']]))
    parts.append(table(['真实−置乱训练','评分条件','NLL差 [95%]'],[[r['seed'],r['loss'],ci(r['natural_minus_permuted'])] for r in a['calibration']['paired_training_target_comparisons']]))
    parts.append('校准控制若解释了部分NLL变化，只能缩小解释空间，不能证明它是参数更新的唯一原因；若仍有残差，也不能立即命名为语义结构。温度网格有限、训练样本数小、监督目标常见度和两种种子均有限。真实学习形成的证据来自2742实际参数轨迹，本阶段新增的是它在新关系/新内容位置上的迁移和强对照。')
    parts.append('实际选择：五个参数变体的联合校准均选T=1.25、α=0，因此联合控制实际为温度控制，词频混合没有被选择。原生NLL3.21793→3.00591，减少0.212019[0.127199,0.297980]。两种种子的真实−置乱raw差约0.0573/0.0577，校准后缩至0.00978/0.00928且区间均跨零，削弱了仅以置乱loss更低解释学习性质的依据。真实训练相对各自校准后的原生仍有约0.0168/0.0180的有限改善，但不等于所有效应已被温度解释，也不唯一支持语义学习。自然argmax约42.2%—42.7%，并无普遍准确率改善。')
    parts.append('另外，2742真实/置乱训练虽具有同一步长范数，累计FP32及BF16位移却不同；本Phase部署这些实际终点而未进行位移范数匹配。概率校准不能替代参数方向/累计幅度的匹配对照，剩余效应不能唯一归因于标签语义。这也是后续学习形成路线需要补齐的识别条件。')
    parts.append('### C005：1600条实际自身历史生成与预定真实例\n\n每个参数变体独立运行全部320表达，固定B8分组、左padding、mask及position IDs，最长128token，无gold、无读出偏置、无强制EOS。首状态另与同变体B1采集比较。评分在输出前冻结，只增加完整yes/no Markdown或LaTeX外包语法，保留旧严格评分；不能把解析覆盖收益算作模型学会更多。')
    rr=[r for r in a['behavior']['summary'] if r['family']=='all']
    parts.append(table(['实际参数变体','正确且EOS/320','两边均正确配对/160','未解析EOS','截断','解析错误','平均token','相对native成功率差 [95%]','B1/B8首token不同'],[[r['variant'],r['correct_and_stopped'],r['paired_token_matched_both_correct'],r['unparsed_EOS'],r['censored'],r['parsed_wrong'],r['mean_tokens'],ci(r['paired_success_change_vs_native']),r['first_token_B1_B8_disagreements']] for r in rr]))
    parts.append('实际所有输出均3token、均有EOS、均可按原严格规则解析，新增外包评分没有贡献成功数。原生238/320，真实训练两种种子均239/320，置乱分别237/320与238/320；变化至多1例，所有成功率差区间均含零。原生双边均正确78/160：属性32/32、否定19/32、词义17/32、长角色10/32、知识链0/32。B1与B8首token有3—4例不同，因此正式参数比较固定B8；不得拿B1首argmax与B8生成的差别冒充训练提升。')
    parts.append('这里的“变化至多1例”只指正确总数的净差，并非宣称最多只有1条输出轨迹发生改变；成对成功变化、输出ID和首次分叉均分别保留。')
    for e in a['behavior']['predeclared_real_examples']:
        parts.append('#### 预定case0英文实例：'+e['family'])
        for j,row in enumerate(e['material']):
            output=e['actual_outputs']['native'][j]
            parts.append('实际用户输入（不含已另存的chat包装）：\n\n```text\n'+row['original_text']+'\n```\n\n外部规则答案：'+row['target']+'。原生实际输出：\n\n```text\n'+output['generated_text']+'\n```\n\n实际保守评分：'+str(output['answer_scoring']['parsed_and_stopped_correct'])+'；生成token数：'+str(len(output['generated_ids']))+'。')
    parts.append('这些实例按族内case0和英文预先固定，不按成败挑选；其他语言、全部变体与失败文本均可从相同sample/pair ID回查。完整中间推理未逐步判真，最终正确且停止不自动证明推理文本忠实。')
    parts.append('### C006：共同现象—规律—参数—推广—形成五环与完整理论状态\n\n'+a['common_phenomenon']+'\n\n候选规律：'+a['candidate_rules']+'\n\n原生结构：'+a['native_parameter_structure']+'\n\n未见检查：'+a['unseen_composition_prediction']+'\n\n训练形成：'+a['training_formation'])
    parts.append('累计核心拼图为43项，公式42项；2743已逐条保留前40个公式，本节新增的严格配对测量和留出校准公式在C002/C004完整定义。RDC主体公式与三图谱接口不变，不把测量定义或拟合控制升级成新数学定律。完整核心拼图及边界如下：')
    parts.append(table(['Phase','保留拼图','证据边界','状态'],[[p['phase'],p['retained_puzzle'],p['boundary'],p.get('evidence_status','inherited')] for p in t['puzzles']]))
    parts.append(t['first_principles_insight']+'\n\n'+t['new_mathematical_increment'])
    parts.append('### C007：文件、全场保留与下一大阶段资源边界\n\n脚本均为tests/glm5/phase2744_rdc_query_*.py，队列为rdc_query_identifiability_queue.py。实际产物位于tests/glm5/result/rdc_query_campaign_20260913/identifiability：protocol.json、material.json.gz、relations、calibration、behavior、analysis/result.json、verification.json，以及主目录theory_snapshot.json和figures/index.json。源码版本、完整参数增量、分词/材料身份、所有选定全坐标/全单元字段和原始输出保留，客户端/rdc-query新增严格身份配对与五个实际参数变体查询。所有NPZ还可按原始轴分页，没有删除原始模型或仍可查询的HiddenState场。')
    admission=read(BASE/'continuation_after_2744.json')
    parts.append('下一大阶段问题仍相同：在这套严格身份和校准控制上，跨三个原生模型研究早期query如何被语境构造，而不再只比较末端向量或模型大小。具体完整方案是320表达×100query×3模型的全部Q坐标、全部来源attention、后层响应与前缀可用规则对照。现有跨模型采集仅前3例保留完整Q/attention；成本估计取三模型这3例的真实时间/文件体积并乘320与1.2余量，不是物理下界，也没有宣称所有小先导都不可能。')
    parts.append('同一完整阶段还需在Q4的自然内容和严格关系决策上加入累计参数位移匹配的方向控制，分别测量FP32目标位移与BF16实际位移，比较真实目标、置乱目标和原生基线。下表成本是上述三模型查询采集的参考估计，尚未包含新增方向控制和提取算法的精确成本，不应把它误读为已完成全部预算审计。')
    parts.append(table(['实际剩余与完整阶段估计','数值'],[[k,admission[k]] for k in ['reference_estimated_capture_seconds','remaining_seconds_after_audit_reserve','reference_estimated_capture_bytes','remaining_result_bytes_after_delivery_reserve','time_gate','storage_gate','complete_stage_admitted']]))
    parts.append('完整下一阶段准入为'+str(admission['complete_stage_admitted'])+'。若任一资源门未过，当前有限研究交付结束，保留该明确问题、完整材料及可恢复入口，不删除关键证据强行继续；这不是宣称总磁盘已满、整个时间预算已用尽或语言机制已破解。若所有门均通过，则需继续执行，不能把此记录当作停止理由。\n\n通俗结论：本阶段让“同样的词是否表达不同关系”和“损失变好是否只是概率变温和”成为同批可核对问题。全坐标图谱、真实参数学习、自由生成与强对照继续被联到同一套身份索引；仍没有一个能解释知识、推理和语法无限组合能力的普遍闭合结构。')
    body='\n\n'.join(parts)
    append=f"\n\n## Phase 2744: 严格词袋身份、真实训练迁移与概率校准的可识别性 [{stamp()[:16].replace('T',' ')}]\n\n{body}\n"
    assert not any(ord(c)<32 and c not in '\n\r\t' for c in append)
    print(json.dumps({'phase':2744,'anchor':text.rstrip('\r\n').splitlines()[-1],'append':append},ensure_ascii=False))


if __name__=='__main__':main()
