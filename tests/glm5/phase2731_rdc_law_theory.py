"""Append-compatible, queryable evidence inventory; identities are not new laws."""
from rdc_law_common import *


def main():
    required=['atlas/result.json','formation/initial_stable/result.json','formation/gradient_controls/result.json',
        'formation/trajectories/result.json','prediction/result.json','confirmation/result.json','confirmation/combination_visibility/result.json','deployment/result.json',
        'deployment/scope/result.json','own_history/result.json','scale/qwen4/result.json','scale/qwen14/result.json','scale/glm4/result.json']
    for p in required:assert (BASE/p).exists(),p
    previous=read(OPERATOR/'theory_snapshot.json');puzzles=[]
    for item in previous['puzzles']:
        r=dict(item,evidence_status='inherited_scoped_evidence_not_rerun_in2728_2731')
        if 'artifacts' in r:r['artifacts']=['../rdc_operator_atlas_20260911/'+p for p in r['artifacts']]
        puzzles.append(r)
    additions=[
        (2728,'六语料/任务族自然全坐标图谱，原生门乘积完整交叉项，以及74711040参数的精确梯度外积与真实单步训练预测。',
         '事后关系标签不是内部语义事实；局部训练影响预测给定目标标签，不能冒充下一token无标签预测；小步规律不保证大步。',
         'observed_structure_known_identities_and_actual_parameter_updates',['atlas/result.json','formation/initial_stable/result.json','formation/gradient_controls/result.json']),
        (2729,'只用较早H12、embedding、可用历史与任务信息预测后层写回；同线性平滑器的联合乘积与直接写回交换校准；四条真实64步原生参数训练轨迹。',
         '全坐标历史摘要不等于完整可预测状态。真实参数编译未统一胜过直接写回。连贯训练优于乱序对照不等于优于原模型。',
         'frozen_full_coordinate_prediction_and_restricted_continuation_training',['prediction/result.json','prediction/frozen.json','formation/trajectories/result.json']),
        (2730,'新来源和被排除依存共现确认；七分支真实自身历史部署、实际BF16训练参数，以及三个本地非量化模型的同材料原生宽度复查。',
         '状态预测收益、输出概率收益和语言能力分别核查；192组合窗口锚点仅126已出现所需端点，细分核查为后验口径收紧，有限共现不是语义组合代数。三模型差异不是规模因果结论。',
         'prospective_confirmation_and_native_deployment_with_matched_limited_replication',['confirmation/result.json','confirmation/combination_visibility/result.json','deployment/result.json','deployment/scope/result.json','scale/qwen4/result.json','scale/qwen14/result.json','scale/glm4/result.json']),
        (2731,'在近似模型自己选出的历史上，独立原生参照逐步核对完整词表和全部KV；同历史直接缓存变化与输出规则误差被分开。',
         '参照是事后核对，不喂回原生KV/答案；重复材料诊断不是新独立确认。末MLP位于所有attention之后的结果只适用于该作用位置。',
         'automatic_same_goal_own_history_diagnostic',['own_history/protocol.json','own_history/result.json'])]
    for phase,retained,boundary,status,paths in additions:
        puzzles.append({'phase':str(phase),'retained_puzzle':retained,'boundary':boundary,'evidence_status':status,'artifacts':paths})
    formulas=[dict(r,evidence=r['evidence']+' Inherited evidence and scope from Phase2727; not rerun here unless explicitly linked below.') for r in previous['formulas']]
    formulas.extend([
        {'id':'early_full_coordinate_condition_kernel','kind':'fitted_candidate_not_complete_predictive_state',
         'expression':'b(i,j)=(q_i dot q_j + e_i dot e_j)/(2D)+rho_i*rho_j; h(i,j)=mu_i dot mu_j/D; K_add=1+b+h; K_mul=1+b+h+b*h; K_task=1+b+b*1[c_i=c_j]. yhat=k_*^T alpha+ybar; alpha=S^(1/2)(S^(1/2) K S^(1/2)+lambda I)^(-1) S^(1/2)(Y-ybar).',
         'variables':'q/e: complete H12/embedding vectors, per-vectorRMS and train-only vector/global-scalar normalization. mu: all-causal-source normalized H12 mean; routed variant uses allhead attentionweights. rho=log(1+position)/log2048. c:knownlanguage xnatural/QA inmain4B, languageONLYin natural-onlythree-model scale fitting; not goldsemantic label. S:positive source/cohort weights. df=tr[K_S(K_S+lambda I)^(-1)], full spectrum retained.',
         'evidence':'192 main candidates and frozen new-source/new-cooccurrence checks. Prediction does not consume target-layerx or futuretoken; later native network remains outside the extracted predictor.'},
        {'id':'joint_product_and_linear_commutation','kind':'known_probability_and_linear_algebra_identities',
         'expression':'E[phi*u | I] = E[phi | I]*E[u | I] + Cov(phi,u | I); T(A Wd^T)=T(A) Wd^T for a shared linear smoother T.',
         'variables':'phi=SiLU(g), u=up activation, elementwise product and covariance. I: available conditions. A: full native unit product rows. Conditional expectation identity is not automatically an identity of separately fitted ridge models; linear commutation requires identical weights, centering and regularization.',
         'evidence':'Measured all-unit joint covariance, native finite-product ledgers and linear-commutation calibration. DirectMLP/joint-product forecasts are not two independent discoveries; BF16 rounding is separately bounded.'},
        {'id':'complete_native_parameter_gradient','kind':'known_chain_rule_exact_outer_product_representation',
         'expression':'a=SiLU(g)*u; t=Wd^T s; bg=t*u*SiLUprime(g); bu=t*SiLU(g); grad_Wd loss=s a^T; grad_Wg loss=bg x^T; grad_Wu loss=bu x^T. Gram(i,j)=(s_i dot s_j)(a_i dot a_j)+(x_i dot x_j)[(bg_i dot bg_j)+(bu_i dot bu_j)].',
         'variables':'s=d(loss)/d(MLPwriteback) through native finalRMS and fullvocabularyCE. x/g/u are current native factors. Gram sums all74711040 native scalarparameters. Factorized storage is an exact outerproduct identity, not a Top-K parameter approximation.',
         'evidence':'Independentfloat64 calculus checks plus native autograd/shape controls, complete288x288 Gram and36 actual single-example updates. Observed pair similarities are dependent and semantically confounded.'},
        {'id':'formation_operation_local_bridge','kind':'known_local_expansion_with_empirical_step_size_limits',
         'expression':'theta_new=theta-eta grad_theta loss_i; loss_j(theta_new)-loss_j(theta)=-eta Gram(i,j)+O(eta^2); theta_(t+1)=theta_t-0.01 min(1,10/||g_t||) g_t.',
         'variables':'theta only the three native lastMLP matrices; eta is the actual single-example step coefficient, while g_t is a16example minibatch gradient for64step continuation. Originalcheckpoints are read-only. GiventargetCE gradients are training inputs, not future answers inferred without labels.',
         'evidence':'Small actual steps are predictable; larger finite displacements depart from initial-gradient forecasts. Two paired minibatch-orderseeds, coherent vs matchedtarget prefixorder control, plus unseen-source evaluation and BF16deployment. No reconstruction of originalpretraining.'},
        {'id':'last_MLP_same_history_cache_scope','kind':'known_architecture_dependency_prediction_with_native_audit',
         'expression':'For fixed identical tokenhistory x_(<=t), if theta and theta_prime differ only in finalMLP: KV_theta(x_(<=t))=KV_theta_prime(x_(<=t)); p_theta(.|x_(<=t)) need not equal p_theta_prime(.|x_(<=t)). Once chosen histories diverge, their KV equality is no longer implied.',
         'variables':'All layers compute attention/KV before the finalMLP. Assumes the tested causal decoder architecture, unchanged earlier parameters and numericalshape; includes finalquery-only vs allprefill action. An earlierMLP can change upperlayerKV.',
         'evidence':'Predeclared in main deployment, tested with all36layers/allKVentries on common known prefixes and independent native references on each autonomous branch ownhistory. No nativecache reinjection.'},
        {'id':'three_atlas_evidence_schema','kind':'typed_bookkeeping_definition_not_new_geometric_law',
         'expression':'Atlas_D=(G_external,G_internal,E_association^D). G_internal indexes (model,run,prefix,step,layer,position,coordinate/unit/parameter,boundary). E_association^D records (condition,algorithm,inputscope,fit,heldoutprediction,evidence,status).',
         'variables':'G_external contains original language/source/family IDs, typed retrospective relationships and actualquestion targets; these are observations/annotations, not assumed neuralobjects. Internal graphs retain native indices; equalindices acrossmodels or layers do not imply equalfunctions. D explicitly limits testedconditions.',
         'evidence':'All1152 materials linked to retained fields, source matrices, frozenpredictions, gradients, trainedparameters and actual outputs in the client. Layout distance is not physicalgeometry; no universal transition law is identified.'}])
    save(BASE/'theory_snapshot.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),
        'inherited_snapshot_sha256':sha(OPERATOR/'theory_snapshot.json'),'theory_name':previous['theory_name'],
        'puzzles':puzzles,'formulas':formulas,'global_closed_theorem_added':False,'new_mathematics_claimed':False,
        'global_atlas_interface':{'external':'Original multi-genre natural language, source families, token/role/typeddependency annotations and humanquestion/answer/support data; retrospective labels never supplied as onlinegold.',
            'internal':'Embedding and fullnativecoordinate field -> actualMLPunit/parameter sums -> entirevocabulary readout -> explicitly recorded native or selfselected histories.',
            'association':'Full-coordinate statistical structure and earlycondition prediction linked to realparameter formation, independent combination checks and local/cached/native deployment boundaries.',
            'unclosed':'No identified universal semanticoperator, no originaltraining reconstruction, no sufficient compact state proven, and no complete knowledge/reasoning/syntax algorithm.'},
        'scope':'26历史条目继承原有边界；4个新Phase的实际证据按运行文件核验。已知数学恒等式、拟合算法和经验适用域不是新数学定理；RDC历史响应束没有被随意改名为任意HiddenState。',
        'next_big_question':'Learn source/role-resolved crosslayer update rules and their formation under controlled continuation, using available prefix inputs, matched complexity and separately frozen novel composition/depth materials; optimize and compare at native full-output and own-history boundaries.'})
    print('LAW_THEORY_INVENTORY_COMPLETE',len(puzzles),len(formulas),flush=True)


if __name__=='__main__':main()
