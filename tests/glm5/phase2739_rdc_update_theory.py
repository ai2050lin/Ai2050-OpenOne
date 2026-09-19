"""Evidence-limited cumulative RDC ledger, not a new universal closure theorem."""
from copy import deepcopy
from rdc_update_common import *


def main():
    required=['graph/confirmation.json','learning/finite_result.json','middle_training/result.json',
      'native_paths/result.json','same_history/result.json','language_prediction/result.json',
      'fresh_graph/result.json','predictive_state/result.json','long_answers/result.json',
      'behavior_analysis/result.json','terminal_format_audit/result.json','manual_terminal_audit/result.json','scale_analysis/result.json','language_identity/result.json',
      'scale_batch/qwen4/result.json','scale/qwen14/shape_audit/result.json','scale/glm4/shape_audit/result.json','causal_anchor/result.json','causal_replay/result.json','next_stage_admission.json']
    assert all((BASE/p).exists() for p in required)
    old=read(PRIOR/'theory_snapshot.json');puzzles=deepcopy(old['puzzles']);assert len(puzzles)==34
    for p in puzzles:
        p['evidence_status']='inherited_scoped_evidence_not_rerun_in2736_2739'
        if 'artifacts' in p:p['artifacts']=['../'+PRIOR.name+'/'+a for a in p['artifacts']]
    updates=[
      {'phase':'2736','retained_puzzle':'全原生坐标的软head读出、有向来源算子、原幅值/RMS/方向/打乱控制；128新自然窗口及混合操作材料形成统一入口。',
       'boundary':'54候选核/容量/解码组合的选择不构成语义识别；全局算子Gram共同转置不变，方向敏感必须另作查询。自然确认的主增益伴随归一化与表达分布影响。',
       'evidence_status':'full_coordinate_observation_and_frozen_limited_prediction',
       'artifacts':['graph/frozen.json','graph/confirmation.json','graph/pilot.json','analysis/phase2736.json'],
       'common_phenomenon':'不同自然来源呈现可测的早层来源组织和后层条件写回。',
       'candidate_rule':'由可见H12预测软head，用全部来源和坐标构造关系核并预测MLP。',
       'native_parameter_structure':'用实际gate/up/down编译预测的全输入或全激活，另与直接写回比较。',
       'unseen_composition_prediction':'旧严格320拟合/64验证；128新自然窗口与混合操作独立评估，不使用新gold边或未来token作输入。',
       'training_formation_evidence':'本Phase的head/核拟合是外部提取器学习；真实原生参数继续训练在2737单独检验，不能把两者混称。'},
      {'phase':'2737','retained_puzzle':'74711040个真实参数的内容/格式非正交梯度、192约束方向全因子ridge与实际BF16范数；四条中层32步训练和无标签H12效应预测。',
       'boundary':'格式约束不是纯语义，局部小导数不保证有限更新格式不变；效应预测未稳定优于表达类别均值。混合数字loss的进展未转化为普遍自然语言能力进展。',
       'evidence_status':'complete_parameter_calculus_and_restricted_continuation_training',
       'artifacts':['learning/frozen.json','learning/autograd_audit.json','learning/finite_result.json','learning/forecast_audit.json','middle_training/result.json'],
       'common_phenomenon':'同一更新在不同表达、监督目标与自然语句上的作用不一致。',
       'candidate_rule':'连续ridge削弱声明格式梯度响应，再测试有限更新和训练前效应预测。',
       'native_parameter_structure':'末MLP全部三矩阵；中层block16通过真实17..35后缀反传，参数仅在内存改变。',
       'unseen_composition_prediction':'混合操作家族、EN重排/Python/ZH及自然内容锚点分别报告；保留强类别基线和失败。',
       'training_formation_evidence':'两种目标×两种固定批次顺序×32步的实际继续训练；不是原预训练形成过程的恢复。'},
      {'phase':'2738','retained_puzzle':'五类双语/两种回答风格640表达；24例4796条可见来源—坐标—单元—写回账本；1056自身历史与108同历史参照；三原始非量化模型匹配复查。',
       'boundary':'来源分配不是唯一因果分工；正文锚点为最后完整token而非句号，且尚未读问题/风格指令。320对同正文前缀H0全相同但H12仅110对bit相同，不能把提前的数值差异叫风格因果效应。词义/长角色匹配当前token后无对照，不能识别独立语义优势。三模型图谱用匹配batch8及各6例B1复查；Q4原行为B1、较大模型batch8/10，未隔离执行形状/规模/训练/架构。',
       'evidence_status':'language_family_atlas_native_provenance_and_history_diagnostics',
       'artifacts':['language_analysis/result.json','language_analysis/logic_audit.json','native_paths/result.json','language_prediction/result.json',
         'own_history/result.json','same_history/result.json','scale/qwen4/result.json','scale/qwen14/result.json','scale/glm4/result.json','scale_analysis/result.json',
         'language_identity/result.json','causal_anchor/result.json','causal_replay/result.json','scale_batch/qwen4/result.json','scale/qwen14/shape_audit/result.json','scale/glm4/shape_audit/result.json'],
       'common_phenomenon':'正文末端与回答指令末端的响应关系不同；最后MLP与中间MLP修改对历史缓存的作用不同。',
       'candidate_rule':'固定锚点的全坐标条件关系；来源attention写入经真实归一化和全部MLP单元的对称分配。',
       'native_parameter_structure':'完整32head/8KVhead、W_O、gate/up/down原生标量因子；中间层修改后的KV变化从后续block出现。',
       'unseen_composition_prediction':'冻结英语自然H12预测器迁往未见语言表达，整体收益不稳定；三模型保留自己的tokenizer/深度/坐标。',
       'training_formation_evidence':'2737保存的实际末层方向与中层训练参数在本Phase部署；主分支自身选择token，原生参照仅独立诊断不回注。'},
      {'phase':'2739','retained_puzzle':'自动同目标续研：新128自然窗口强匹配控制、四阶全坐标矩的明确反例、64可达近远前缀对与固定probe、真实KV成本及1024token长答案核对。',
       'boundary':'中间层有向核未胜平方基线；近摘要不等于等价状态。合成矩碰撞不能证明自然状态可达或所有压缩不可能。长生成仍是原8个语义组诊断，不是额外独立确认。',
       'evidence_status':'automatic_same_goal_confirmation_and_predictive_information_boundary',
       'artifacts':['fresh_graph/material_audit.json','fresh_graph/frozen.json','fresh_graph/result.json','moment_boundary/result.json',
         'predictive_state/result.json','long_answers/result.json','behavior_analysis/result.json'],
       'common_phenomenon':'保留所有坐标仍可能在聚合时丢失有序历史；截断内没有答案不能直接判为答错。',
       'candidate_rule':'以固定可用查询区分历史摘要，按完整答案/停止/截断分开评估；同RMS/同df比较来源关系候选。',
       'native_parameter_structure':'真实全部KV张量计数/哈希、全词表比较和已冻结原生参数更新；矩反例明确为FP64合成对象。',
       'unseen_composition_prediction':'128新自然窗口/384内容锚点，来源组置信区间；预测规则未依据新结果重选。',
       'training_formation_evidence':'1024token复用同批实际训练增量；不添加未经执行的新训练，也不宣称无损100K记忆。'}]
    # The frozen main grid consists of 9kernels×2df×3decoders×2blocks=108 records.
    updates[0]['boundary']=updates[0]['boundary'].replace('54候选','每个block54候选')
    puzzles.extend(updates);assert len(puzzles)==38
    formulas=deepcopy(old['formulas']);assert len(formulas)==23
    for f in formulas:f['evidence']+=' Inherited scoped formula; not newly discovered or universally validated in2736..2739.'
    formulas.extend([
      {'id':'available_prefix_directed_operator_kernel','kind':'fitted_all_coordinate_finite_feature_rule',
       'expression':'u_s=h_s/RMS(h_s); v_s=RMSnormalize([u_s,1]B); A_st=softmax_(t!=s)(v_s dot u_t/(0.15D)); T_i=H_i^T A_i H_i/(n_i D); S_ij=<T_i,T_j>_F; K_ij=1+b_ij+S_ij+b_ij*S_ij.',
       'variables':'H contains every visible source in native coordinate order (raw or rowRMS according to candidate); B is the frozen2561x2560 English head ridge readout; b=(q_i dot q_j+e_i dot e_j)/(2D). D2560, n is visible token count. Native causal H12 is available, gold test head and target layer are not inputs.',
       'evidence':'Full-source contractions audited, original320train/64validation selection retained and two128window confirmations. Earlier states are slices of causal full-prompt BF16 forwards: strict prefix-only execution-shape equivalence is not established for every case; no future gold labels/tokens are explicitly fed to the predictor. Gram(T_i^T,T_j^T)=Gram(T_i,T_j): common direction reversal is unidentifiable from this kernel alone; query-conditioned Tq/reversed candidates are separate.'},
      {'id':'all_factor_format_constraint_ridge','kind':'regularized_linear_algebra_not_semantic_orthogonality',
       'expression':'F_i=grad L_format,i / ||grad L_format,i||; c=(F^T F+lambda I)^(-1)F^T g_content_mean; d=g_content_mean-Fc; theta_new=BF16(theta-alpha*d/||d||).',
       'variables':'Columns F contain all192 declared training EN/Python format gradients over74711040 parameters. lambda is relative1e-6 or1e-3 times mean Gram diagonal. No hard eigenvector deletion. Alpha calibration uses actual BF16 parameter displacement only; heldout outcomes are not calibration inputs.',
       'evidence':'All192factor Gram entries, full coefficient vectors, all parameter direction arrays, supervised finite comparisons and source-heldout effect forecasts. This is a ridge residual, not an exact null-space projection; finite/later format changes remain.'},
      {'id':'middle_batch_format_constraint','kind':'executed_supervised_training_rule',
       'expression':'c_t=mean_batch grad_theta L_content; f_t=mean_batch grad_theta L_format; d_t=c_t-(<c_t,f_t>/max(||f_t||²,1e-30))*f_t; theta_(t+1)=theta_t-0.02*d_t/||d_t||.',
       'variables':'Theta is actual block16gate/up/down; content-only comparator uses d_t=c_t. Four32-step runs, batch4, paired draw seeds2737/2738; fixed0.02parameter norm per step, FP32MLP/BF16 native suffix. Orthogonality refers only to current batch mean f_t.',
       'evidence':'Saved per-step actual gradients, parameters, bridge/original checkpoints and source-heldout evaluation. Small instantaneous <d,f> does not force finite mean format loss unchanged, does not constrain every sample, and does not reconstruct pretraining.'},
      {'id':'native_source_to_unit_symmetric_allocation','kind':'known_linear_contraction_and_conditioned_accounting_identity',
       'expression':'C_s,d=sum_(h,f) A_h(q,s)V_s,kv(h),f W_O,d,hf; x_s=(gamma/rho_observed)*C_s; g_s=Wg*x_s; u_s=Wu*x_s; a_s=0.5*sigmoid(g)*(g_s*u+u_s*g); m_s=Wd*a_s.',
       'variables':'Every visible source,32attention heads,8KV heads,all2560coordinates and9728MLP units atblocks16/35. rho_observed is the actual RMS denominator, not a counterfactual one. Residual/other and rounding terms are kept so source-plus-other gates/ups reconstruct native factors.',
       'evidence':'24predeclared cases/4796visible paths; native-repeat bit equality and real-arithmetic contraction/remainder checks. Fixed-denominator/sigmoid allocation is not a uniquely identified causal decomposition, nor an earlier-layer predictor.'},
      {'id':'finite_set_gram_isometry_necessary_condition','kind':'known_linear_algebra_condition_with_rejection_in_tested_domain',
       'expression':'If V=UQ and Q Q^T=I, then V V^T=U U^T. Nonzero normalized Frobenius Gram difference rules out that exact shared Euclidean isometry on the specified finite paired set.',
       'variables':'U,V rows are corresponding full-parameter full/content/format gradients across EN/reordered/Python/ZH expressions. This tests a specified Gram equality, not all maps or approximate notions of semantic invariance.',
       'evidence':'36measured Gram conditions; target-dependent gradients include language/format/identity differences. The failure of exact isometry cannot establish the absence of every cross-expression mechanism.'},
      {'id':'finite_degree_ordered_history_counterexample','kind':'known_Prouhet_identity_applied_to_declared_summary',
       'expression':'epsilon_s=(-1)^popcount(s), s=0..31; sum_s epsilon_s*s^j=0 for j=0..4. H_a,s=epsilon_s*v; H_b,s=-epsilon_s*v. sum_s s^j H_a,s^(tensor k)=sum_s s^j H_b,s^(tensor k), k=1..4,j=0..4. softmax(q*s)^T H_a can differ from softmax(q*s)^T H_b.',
       'variables':'v is full2560-dimensional FP64, unitRMS and in the actual frozen coarse-role probe null space. Even tensor orders match positionwise; odd orders use the exact integer sums. This identity represents all tensor coordinates exactly without materializing D^4.',
       'evidence':'All32x2560 histories retained; role values and20order pairs audited; nonzero scalar query attention outputs differ. Ambient construction, not demonstrated natural Transformer reachability. No claim against injective positional storage, arbitrary compression, fullKV or all-order moment conditions.'},
      {'id':'finite_probe_predictive_state_and_native_cache_size','kind':'operational_definition_and_architectural_cost_identity',
       'expression':'Psi_Q(p)=[P_theta(.|p concatenated with q)]_(q in Q); Delta_Q(p,pprime)=[0.5*(KL(P_theta(.|pq)||P_theta(.|pprime q))+KL(P_theta(.|pprime q)||P_theta(.|pq)))]_(q in Q); bytes(KV)=2*L*n_KVhead*d_head*T*bytes_per_scalar.',
       'variables':'Q is the saved fixed probe set empty/the/because/However/newlineThe, not future gold; separately tokenized probe IDs are appended to actual prefix IDs. Delta is symmetric KL. Natural prefixes are selected by full qH12/embedding/source-mean proximity; near/far descriptors are never asserted exactly equal. KV count here is actual Qwen4 L36,8KVheads,128head dimension,BF16.',
       'evidence':'64near/far pairs from32centers,63distinct prefixes and27source groups; fullvocabulary KL and512/1024/2048token actual all-layer cache audit. Finite probe agreement is not sufficient to prove equality under all future continuations or a Markov closure.'},
      {'id':'conservative_terminal_success_yield','kind':'measurement_definition_not_semantic_theorem',
       'expression':'Y=(1/N)sum_i 1[explicit_terminal_answer_i=gold_i AND EOS_i]; report separately unparsed_i, capped_i, wrong_parsed_i and strict_answer_only_i.',
       'variables':'Terminal grammar requires a complete digit-only answer or explicit terminal answer marker for programs; Yes/No or 是/否 obey whole-answer/terminal-marker rules. Censored output is not silently treated as a complete wrong answer. Natural continuation has no unique gold.',
       'evidence':'Original trajectory commits/IDs and narrower primary scorer retained;10primary plus18secondary parser regression cases. Secondary post-outcome audit recognizes complete Markdown/LaTeX/boxed or exact requested-variable terminal values; both measurement versions and paired results reported. Bounded1024token continuation of exactly the original32program expressions. Parse+stop is operational success, not a grading of every reasoning sentence.'}
    ])
    behavior=read(BASE/'behavior_analysis/result.json');long=read(BASE/'long_answers/result.json')
    manual=read(BASE/'manual_terminal_audit/result.json')
    formulas[-1]['evidence']+=' The entire residual unparsed EOS long-answer set is separately reviewed by the unblinded main agent with exact terminal quotes. This third post-outcome adjudication does not change either grammar, the original outputs or the reasoning-chain boundary.'
    replay=read(BASE/'causal_replay/result.json')
    updates[2]['boundary']+=f" 另对全部80个未见组正文唯一前缀执行320次仅前缀/重复/等长度未来后缀前向；同前缀重复全相同，等长度未来后缀全部37层相同的对数为{replay['same_length_future_suffix_all37_exact_pairs']}/80。冻结预测器用仅前缀实际输入和目标重查，仍不构成新语义独立确认。"
    result={'timestamp':stamp(),'source':snapshot(__file__),'theory_name':'条件化输出场闭合理论（RDC）',
      'inherited_snapshot_sha256':sha(PRIOR/'theory_snapshot.json'),'puzzles':puzzles,'formulas':formulas,
      'RDC_primary_formula_changed':False,'global_closed_theorem_added':False,'new_mathematics_claimed':False,
      'new_mathematical_increment':'可执行的全坐标关系算子/约束学习/来源分配候选及明确有限矩反例；均由已知线性代数、微积分、概率和组合恒等式表达，尚无必须诉诸新数学的证据。',
      'five_correspondence_columns':['common_phenomenon','candidate_rule','native_parameter_structure','unseen_composition_prediction','training_formation_evidence'],
      'global_atlas_interface':{'external':'2176stable material IDs:512reused strict natural+256new natural+768mixedprogram+640five-family bilingual expressions; source groups and operation/UD/word-sense types remain explicit.',
        'internal':'Full-coordinate anchor fields, all causal early sources, all native MLP units,24source-path fixtures, complete74711040gradient factors, actual training deltas and own histories.',
        'association':'Frozen predictive banks, matched controls, known native contractions and finite learned effects linked through IDs. Neither graph similarity nor a parameter identity substitutes for an extracted semantic program.',
        'unclosed':'No sufficient finite semantic state, unique gear decomposition, universal composition operator or original pretraining reconstruction.'},
      'actual_long_answer_summary':long['summary'],'formal_trajectory_records':behavior['trajectories'],
      'actual_format_aware_long_answer_summary':[r for r in behavior['format_aware_summaries'] if r['mode']=='long_answers' and r['granularity']=='kind'],
      'actual_manually_audited_long_answer_summary':[r for r in manual['summaries'] if r['granularity']=='kind'],
      'manual_terminal_review_limits':manual['limits'],
      'prefix_only_recovery':{k:replay[k] for k in ('unique_prefixes','semantic_groups','all_prefix_repeats_exact','same_length_future_suffix_all37_exact_pairs','fixed_predictor_reports','cross_protocol_error_shift','scope')},
      'conditional_prefix_prediction_observation':'仅前缀下否定辖域两语/两block相对query误差均更低；知识链在16较低/35较高，属性反向交叉。保留为具体条件性预测线索。两个block使用不同冻结解码器，不能孤立解释为层功能；block16否定仍未胜旧训练均值，8组区间不是多重校正后的普遍语义规律。四组跨数值协议平均误差差区间均跨0，未抹去已有有限预测。',
      'evidence_hashes':{p:sha(BASE/p) for p in required},
      'next_stage_admission':read(BASE/'next_stage_admission.json'),
      'first_principles_insight':'区分三个对象：表达的关系结构、模型保留的有序可查询历史、在条件下把历史编译成输出的运算。全坐标只是避免删轴，不能保证聚合摘要充分；结构相似、监督导数与原生自主回答是不同证据层级。词汇身份若与语义组一一绑定，即使相似很强也缺少可识别性；坐标覆盖和矩阵规模不能替代这种设计条件。',
      'next_big_question':'在自然事件流中，什么可获得的有序来源更新能超过身份/位置/平方特征强基线，并用同一条规则预报后层全坐标、实际学习方向以及后续可查询输出？',
      'next_stage_not_executed':[
        '建立事件/角色有方向的自然跨句来源对；在采集前冻结关系组合和身份/位置/长度强对照，扩大独立文档而非重复同模板。',
        '比较保留来源身份的有序更新、查询条件算子Tq和有限矩；明确新增内存成本与可查询范围，不预设精确压缩。',
        '训练输入只用可用前缀；同时预测多个后层完整坐标及未来probe分布，独立保留标签仅作评分。',
        '把胜过强基线的同一规则接到实际多步参数继续训练，做至少两种顺序和冻结自然确认；独立报告回答、停止及格式。'],
      'automatic_continuation_status':'2739 is the executed finite same-goal continuation of2736..2738. The next proposal is saved, not mislabelled executed; further expansive collection must follow the remaining resource ledger and information-value gate.'}
    save(BASE/'theory_snapshot.json',result);print('UPDATE_THEORY',len(puzzles),len(formulas),flush=True)


if __name__=='__main__':main()
