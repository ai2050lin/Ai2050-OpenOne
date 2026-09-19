"""Cumulative scoped puzzles and executable equations; no renamed closure claim."""
from copy import deepcopy
from rdc_query_common import *


def formulas():
    old=deepcopy(read(PRIOR/'theory_snapshot.json')['formulas']);assert len(old)==31
    for r in old:r['evidence']+=' Inherited equation; not newly discovered in2740..2743. Scope remains tied to the cited prior experiment.'
    additions=[
      ('transpose_relative_orientation','known_linear_algebra_identity_with_full_coordinate_audit',
       'T_i=S_i+A_i, S_i=(T_i+T_i^T)/2, A_i=(T_i-T_i^T)/2; <T_i,T_j>_F=<S_i,S_j>_F+<A_i,A_j>_F; <T_i^T,T_j>_F=<S_i,S_j>_F-<A_i,A_j>_F.',
       'T is the prior full2560by2560ordered-source operator. Transposing both members preserves the Gram; transposing only one generally does not.',
       'algebra/result.json;16actual native-source operators plus2independent dense contractions. Direction-sensitive terms are not automatically calibrated semantic arrows.'),
      ('native_ordered_gate_up_source_pair','known_conditioned_accounting_identity',
       'a_(s,r,k)=sigmoid(g_k)*g_(s,k)*u_(r,k); sum_(s,r) a_(s,r,k)=SiLU(g_k)*u_k; a_(s,k)^symmetric=0.5 sum_r[a_(s,r,k)+a_(r,s,k)].',
       'g_(s,k),u_(s,k) are complete native gate/up reads of source attention contributions under the observed RMS denominator. Explicit other and BF16 rounding terms retained; k spans all9728units.',
       'algebra/result.json and events/paths_index.json. Ordered factors distinguish branches, but their antisymmetric part cancels in complete double-source summation. Allocation is not unique causal semantics.'),
      ('all_unit_antisymmetric_pair_energy','known_exact_factorization_not_semantic_percentage',
       '||A_k||_F^2=0.5*sigmoid(g_k)^2*(||g_.k||^2 ||u_.k||^2-<g_.k,u_.k>^2), A_k=(a_..k-a_..k^T)/2.',
       'All visible source positions plus explicit other are included. Every gate/up unit is retained; factorization evaluates the full SbySbyM quantity without dropping axes.',
       'analysis/phase2740.json; no proportion of linguistic causation is inferred from this numerical energy fraction.'),
      ('native_query_conditioned_source_readout','known_attention_computation_with_native_architecture_checks',
       's_hs=<R_pos(q_h),K_hs>/sqrt(d_h); w_hs=exp(s_hs)/sum_r exp(s_hr); r_h=sum_s w_hs V_hs; attention_write=W_O concat_h(r_h).',
       'Q/K are projected and normalized only as required by the specific native architecture; R_pos rotates native Q/K, not the entire residual field. Prefix keys already include their native positional transform. Qwen and GLM use their own implementations.',
       'rules/capture_result.json; scale/*/result.json. Actual nativeattention checks and finiteprecision tolerance are separate from context-free query-prototype approximation.'),
      ('fixed_query_stable_accumulator','known_associative_exponential_sum_merge_at_fixed_query',
       'm=max(m1,m2); Z=exp(m1-m)Z1+exp(m2-m)Z2; N=exp(m1-m)N1+exp(m2-m)N2; readout=N/Z.',
       'Each chunk has m=max_s score_s, Z=sum_s exp(score_s-m), N=sum_s exp(score_s-m)V_s. Query q and positional scores stay fixed during merging. Empty chunk handled separately.',
       'rules/capture_result.json. Exact real-arithmetic identity independently checked inFP64; it does not show that a fixed finite accumulator remains sufficient after arbitrary query changes.'),
      ('frozen_prefix_query_full_coordinate_decoder','fitted_available_input_rule',
       'hhat_(c,l,d)=beta_(c,l,d,0)+beta_(c,l,d,1)h_prefix12,d+beta_(c,l,d,2)h_queryalone13,d+beta_(c,l,d,3)f_(c,d).',
       'c:query_only/uniform/quadratic/shuffled_values/ordered_softmax. f is the declared nativeblock12feature using known-query-alone Q/K/V and actual prefixKV. d includes all2560coordinates; l israwH24/rawH36/postnorm. Native nonlinear feature has cross-coordinate mixing; learned decoder is coordinatewise4columns. Mean/scales and lambda chosen only from frozen train/validation.',
       'rules/fit_result.json; rules/vocabulary_result.json; followup/result.json. Actual queriedfutureH12/24/36/postnorm are excluded from these five candidate inputs. Equal nominal coefficient counts do not equal effective rank.'),
      ('paired_text_query_response_mapping','fitted_paired_observed_representation_transfer',
       'hhat_target,d=alpha_d+beta_d*h_source,d+gamma_d*h_queryalone,d.',
       'Observed source expression after the known query is an input; observed target-expression response is only a target. Identity, query-only, two-column affine and shuffled-pair training are controls. EN/ZH/Python are all text.',
       'transfer/fit_result.json and transfer/vocabulary_result.json. Group/query holdout prediction, not invertibility or manifold isomorphism. One-shot readout injection uses this disclosed additional source-response information.'),
      ('authentic_natural_content_native_training','executed_restricted_continued_training_protocol',
       'g_t=(1/4)sum_(i in batch_t) grad_theta[-log P_theta(x_(pos_i+1)|x_(0:pos_i))]; theta_(t+1)=theta_t-0.02*g_t/||g_t||.',
       'Theta comprises all74711040block16gate/up/down parameters; other native weights fixed and all native upperlayers/fullvocabulary participate inbackpropagation. Two draw orders,32steps, matched within-cohort permuted-label controls. FP32trainableMLP/BF16bridge and finalnativeBF16deployment have separate baselines.',
       'formation/result.json and analysis/phase2742.json. A576example pool does not mean everyexample was used in each128draw run. Restricted continuation learning does not recover historical pretraining or semantic emergence.'),
      ('gold_free_uniform_category_logit_bias','known_probability_identity_with_own_history_behavior_test',
       'zprime_v=z_v+b*1[v in D]; Pprime_D=exp(b)P_D/(1-P_D+exp(b)P_D); Pprime(v|D)=P(v|D), v in D.',
       'D is all8digit tokens1..8, b=8 at a single declared own-history trigger; matched8letters are control. No correct label and no forcedEOS. Current computedKV unchanged by external logit addition; selectednexttoken may change futureKV.',
       'late/*/result.json and analysis/phase2742.json. High fullvocabulary entropy is not defined as nonsense. Preserving current conditional digit ranking is not a guarantee of future content invariance.')]
    for i,k,e,v,s in additions:old.append({'id':i,'kind':k,'expression':e,'variables':v,'evidence':s})
    return old


def main():
    required=['analysis/phase2740.json','analysis/phase2741.json','analysis/phase2742.json','analysis/query_identity_control.json','analysis/query_language_control.json','followup/result.json']
    for p in required:assert read(BASE/p)['all_passed']
    old=read(PRIOR/'theory_snapshot.json');puzzles=deepcopy(old['puzzles']);assert len(puzzles)==38
    for p in puzzles:
        p['evidence_status']='inherited_scoped_evidence_not_rerun_in2740_2743'
        if 'artifacts' in p:p['artifacts']=['../'+PRIOR.name+'/'+a for a in p['artifacts']]
    a=read(BASE/'analysis/phase2740.json');b=read(BASE/'analysis/phase2741.json');c=read(BASE/'analysis/phase2742.json');d=read(BASE/'followup/result.json')
    entries=[
      {'phase':'2740','retained_puzzle':'万条自然前缀、百万固定查询的完整末端坐标图谱；共同转置边界纠错、双分支有序来源对与真实生成时间锚点。',
       'boundary':'2777文档不是10000独立样本；全token全层被扫描但只在9个预定样例完整持久化。固定查询不是自然自由生成。反对称分配在完整求和中抵消，文字事件不等于内部符号程序。',
       'evidence_status':'full_coordinate_observation_and_exact_conditioned_accounting','artifacts':['atlas/result.json','algebra/result.json','events/result.json','analysis/phase2740.json'],
       'common_phenomenon':'不同自然历史改变同一已知查询的完整响应；门控与up来源标签的单项可不同。',
       'candidate_rule':'全来源双分支配对和固定查询的有序原生读出，保留其他项及舍入项。',
       'native_parameter_structure':'实际Q/K/RoPE、W_O与block16/35全部gate/up/down坐标和单元。',
       'unseen_composition_prediction':'本Phase为观察/核算；冻结预测、未见文档/查询和预留来源确认分别在2741/2743给出，不能把重建记为提前预测。',
       'training_formation_evidence':'既有2737真实中层训练证据保留且本轮未重写。新自然内容形成实验与同一材料身份关联，在2742单独报告；本Phase未恢复预训练。'},
      {'phase':'2741','retained_puzzle':'576详细前缀上五种同名义容量的全坐标候选，双重未见误差与全词表检验；768配对文字/代码查询映射及可达近远历史对。',
       'boundary':'总体小收益不等于各语料普遍成立；query-only共同解码器仍看到前缀H12。配对映射输入含已观察来源响应，不是更早预测。有限查询与近状态不能证明未来等价。',
       'evidence_status':'frozen_source_and_query_heldout_prediction','artifacts':['rules/fit_result.json','rules/vocabulary_result.json','transfer/fit_result.json','transfer/vocabulary_result.json','pairs/result.json','analysis/phase2741.json'],
       **{k:b[k] for k in ['common_phenomenon','candidate_rule','native_parameter_structure','unseen_composition_prediction','training_formation_evidence']}},
      {'phase':'2742','retained_puzzle':'真实自然内容目标的四条32步中层训练、晚期无gold类别偏置、文字代码一次性读出注入，以及三个原始精度模型自身坐标的查询核对。',
       'boundary':'576是训练池，各次只抽128次；形成实验不是预训练史。相同逐步梯度范数不代表累计参数位移匹配。终值且停止不评分推理链；高熵不是废话。批形状差异、失败映射压力测试、当前缓存与未来历史分别报告。',
       'evidence_status':'actual_restricted_parameter_learning_and_own_history_tests','artifacts':['formation/result.json','late/native/result.json','transfer/injection/result.json','analysis/phase2742.json'],
       'common_phenomenon':'相同原生参数更新、表达条件和外部读出规则的效果不能只由初始数字loss概括。',
       'candidate_rule':'真实下一token监督及目标打乱对照；无gold全类别偏置；已知配对来源向量的一次性映射。',
       'native_parameter_structure':'全部74711040中层MLP参数、完整原生后缀、真实Q/K位置处理与所有有效KV字节。',
       'unseen_composition_prediction':'自然测试文档、旧真实答案形成位置、24未参与旧长生成的语义组和三模型匹配材料分别限定。',
       'training_formation_evidence':'四条实际32步训练和FP32完整参数增量；BF16实际部署、两种抽样顺序、原始/桥接基线及目标打乱对照均保留。'},
      {'phase':'2743','retained_puzzle':'同目标自动续研：96预留文档的冻结规则独立确认，以及明确增加实际查询H12信息的后层诊断；全数据客户端与累计理论审计。',
       'boundary':'额外观察到查询H12的诊断属于更晚信息条件，不能冒充原来的前缀预测。预留文档相对本轮主库独立，不宣称没有历史接触或预训练接触。',
       'evidence_status':'executed_independent_source_confirmation_and_information_boundary','artifacts':['followup/protocol.json','followup/result.json'],
       'common_phenomenon':'冻结规则在真正预留来源上的稳定性与查询构造信息的作用需要分别实测。',
       'candidate_rule':'保持五个原预测器不变，另训练等4列但具有实际查询H12输入的透明诊断。',
       'native_parameter_structure':'全坐标、全部原生prefixKV和原始Q/K/MLP参数；不写回checkpoint。',
       'unseen_composition_prediction':'96不同预留文档×100queries；20未见查询目标单独报告。',
       'training_formation_evidence':'2742真实参数学习证据不被改称本Phase新训练；这里仅额外拟合训练集诊断解码器，原生参数未更新。'}]
    puzzles.extend(entries)
    result={'timestamp':stamp(),'source':snapshot(__file__),'theory_name':'条件化输出场闭合理论（RDC）',
      'inherited_snapshot_sha256':sha(PRIOR/'theory_snapshot.json'),'puzzles':puzzles,'formulas':formulas(),
      'RDC_primary_formula_changed':False,'global_closed_theorem_added':False,'new_mathematics_claimed':False,
      'new_mathematical_increment':'细化了可执行的分支有序来源账本、固定查询合并、同输入候选和全概率行为边界；这些使用已知代数、概率、微积分和监督拟合，没有得到新的普遍语言闭合定理。',
      'attachment_claim_corrections':read(BASE/'contract.json')['claim_audit'],
      'supplemental_attachment_corrections':a['supplemental_attachment_corrections'],
      'main_coordinate_and_vocabulary_qualification':b['prespecified_candidate_qualification'],'actual_formation_and_behavior':c,
      'independent_followup':d,'query_identity_control':read(BASE/'analysis/query_identity_control.json'),'query_language_control':read(BASE/'analysis/query_language_control.json'),'global_atlas_interface':{
        'external':'10000natural windows,2777source documents,21genres; explicit query strings, UDtyped edges, Chinese context metadata,768paired programs and actual generated event annotations.',
        'internal':'Every1millionendpoint full2560coordinate vector; all37prefixlastanchors;576detailed all-source/multilayertarget fields;9fullprefixalltokenalllayerfixtures;352nativegeneratedanchors with100queries each; native ordered factor paths, full parameter deltas,3own-model coordinate spaces.',
        'association':'Stable material/run/model/query/step/layer IDs link frozen candidate predictions, actual source/coordinate/unit/parameter contractions, native continuedlearning and own-history behavior. No single edge label denotes complete cracking.',
        'unclosed':'No unique semantic gear, all-future sufficient state, universal cross-layer reuse rule, cross-expression isomorphism or original pretraining formation mechanism.'},
      'first_principles_insight':'先区分历史中保留什么、当前查询如何在条件下形成、查询如何读取来源、来源如何通过不同门控分支写回、下一token如何反过来改变历史。保留全坐标避免删轴，却不会自动恢复聚合掉的来源次序或尚未知的查询。观测、恒等式、预测、真实学习与自身历史行为必须通过同一数据身份关联，而不能互相代替。',
      'required_evidence_sha256':{p:sha(BASE/p) for p in required}}
    save(BASE/'theory_snapshot.json',result);print('QUERY_THEORY',len(puzzles),len(result['formulas']),flush=True)


if __name__=='__main__':main()
