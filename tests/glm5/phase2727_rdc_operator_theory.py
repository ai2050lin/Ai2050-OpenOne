"""Query-ready theory/puzzle inventory with inherited evidence separated from new results."""
from rdc_operator_common import *


def main():
    for path in ('compiled/result.json','behavior/result.json','operations/result.json','metric_followup/result.json','metric_followup/paired_audit.json','metric_followup/population_geometry/result.json'):
        assert (BASE/path).exists(),path
    memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';text=memo.read_text(encoding='utf-8')
    inherited_section=text.split('## Phase 2723:',1)[1].split('### C007 核心拼图总账：保留、修正与待验证',1)[1].split('### C008',1)[0]
    puzzles=[]
    for line in inherited_section.splitlines():
        if not line.startswith('| '):continue
        parts=[p.strip() for p in line.strip('|').split('|')]
        if not parts or not parts[0][0].isdigit():continue
        puzzles.append({'phase':parts[0],'retained_puzzle':parts[1],'boundary':parts[2],
            'evidence_status':'inherited_scoped_evidence_not_rerun_this_campaign','source':'AGI_GLM5_MEMO.md Phase2723 C007'})
    assert len(puzzles)>=20
    puzzles.extend([
        {'phase':'2724','retained_puzzle':'2048个自然窗口、346920token的全层全原生坐标观察，连接真实问答与有类型的支持关系。',
         'boundary':'材料以百科阅读和前缀词汇线索为主，不是完整金标句法/语义图；稀少数值事件不等于原生硬门。',
         'evidence_status':'new_observation_and_native_behavior','artifacts':['material_audit.json','capture/main/result.json','capture/confirmation/result.json','qa_atlas/result.json']},
        {'phase':'2725','retained_puzzle':'完整原生门控乘积分解、K/J微分核对及冻结全坐标算子在新来源上的局部预测。',
         'boundary':'预测仍接收真实当前归一化输入x；已知架构/微分恒等式用于校准，局部近似不等于完整语言程序。',
         'evidence_status':'known_identity_plus_scoped_local_prediction','artifacts':['operators/result.json','calculus/result.json','structure/result.json','confirmation/result.json']},
        {'phase':'2726','retained_puzzle':'真实剩余网络的全词表编译、三模型匹配复查、自然问题换序，以及BF16舍入账本；同答案可伴随不同连续响应。',
         'boundary':'原确认概率比较只有block34的文章区间稳定为正；局部/联合排名不证明普遍收益，未识别语义重置或重复根因。',
         'evidence_status':'new_compilation_behavior_and_numeric_audit','artifacts':['compiled/paired_audit.json','behavior/result.json','operations/result.json','precision/result.json','identity_audit/corrected_cue_composition.json']},
        {'phase':'2727','retained_puzzle':'输出标准选择的组合、同形状核对、全词表误差度量/总体全坐标耦合，以及同一已选历史上的独立原生诊断。',
         'boundary':'自然材料为复分析；问答为已有行为材料上的任务入口迁移。已知log-partition恒等式不是新数学，未提取整个LLM。',
         'evidence_status':'same_goal_followup_with_explicit_reanalysis_scope','artifacts':['metric_followup/result.json','metric_followup/paired_audit.json','metric_followup/population_geometry/result.json']}])
    formulas=[
        {'id':'historical_RDC_interface','kind':'domain_limited_partial_interface_not_new_global_theorem',
         'expression':r'X_l(p)={H_{l,t,j}(p)}; T^D_{l,tau}:(L(p), boldW_l(p), X_l(p)) partially maps to boldW_{l+1}(p).',
         'variables':'p: declared material/program; L: external description; X: observed full field; historical boldW: factorial response bundle, NOT an arbitrary hidden state; tau: condition/role; D: tested domain.',
         'evidence':'Inherited interface. This campaign does not replace the historical response-bundle object with arbitrary H or prove global closure.'},
        {'id':'native_transformer_step','kind':'known_architectural_definition',
         'expression':'r = H_l + A_l(N_l(H_l), KV_l, position); x = Nprime_l(r); H_(l+1) = r + Wd_l[SiLU(Wg_l x) * Wu_l x]; p_next = softmax(W_U N_final(H_L)_query).',
         'variables':'* is elementwise product. Actual masks, Q/K/RoPE, bias, cache and finite-precision order follow each native architecture; W weights remain fixed.',
         'evidence':'Native module traces/parameters and numerical audits, not a newly discovered semantic theory.'},
        {'id':'native_conditional_operator','kind':'known_real_arithmetic_identity',
         'expression':'K(x) = Wd diag(SiLU(Wg x)) Wu; m(x) = K(x)x.',
         'variables':'x is actual post-attention-normalized input; complete native residual coordinates and all MLP units retained.',
         'evidence':'K(x)x reconstruction checks do not by themselves constitute predictive mechanism extraction.'},
        {'id':'full_jacobian','kind':'known_calculus_identity',
         'expression':'J_m(x) = Wd[diag(Wu x) diag(SiLUprime(Wg x)) Wg + diag(SiLU(Wg x)) Wu]. Residual-to-MLP derivative also includes J_Nprime.',
         'variables':'Both gate and value derivatives are present; K(x) is generally not J_m(x).',
         'evidence':'Complete2560x2560 K/J matrices at3representatives and complete-coordinate derivative actions, plus independent synthetic checks.'},
        {'id':'finite_product_structure','kind':'known_finite_identity_with_measured_nonorthogonal_terms',
         'expression':'m = Wd[phi_bar*u_bar + phi_bar*du + dphi*u_bar + dphi*du]; E(m)=sum_(a,b) <term_a,term_b>/D.',
         'variables':'phi=SiLU(g); du=u-u_bar; dphi=phi-phi_bar; centers fitted on training data. All16 signed Gram terms retained.',
         'evidence':'Center-dependent shared/value/gate/interaction account, not unique semantic components or additive contribution percentages.'},
        {'id':'frozen_gate_predictor','kind':'fitted_local_prediction_rule',
         'expression':'mhat_c(x)=Wd diag(phi_bar_c) Wu x; tangent/quadratic candidates and capacity-matched condition controls are separately evaluated.',
         'variables':'Condition c must be available in known token prefix; fitted center uses train sources only. Current x is supplied, not predicted from language labels.',
         'evidence':'New-article512anchor confirmation and independent native-model matched cohorts; not a full-language decoder.'},
        {'id':'output_metric','kind':'known_identity_with_empirical_full_vocab_audit',
         'expression':'p_s = softmax(z+s*delta); KL(p_0 || p_1) = integral_0^1 (1-s) Var_(p_s)(delta) ds.',
         'variables':'z and z+delta are full-vocabulary logits. Straight z+s*delta interpolation is an analytic path, not a native layer/generation trajectory. Constant shifts do not change softmax. For arbitrary q, KL(q||p_1)=KL(q||p_0)+(p_0-q) dot delta+the integral.',
         'evidence':'All151936categories for640queries x3configurations; adaptive quadrature checked against exact log partitions. Endpoint half-Fisher variance is a local approximation only.'},
        {'id':'complete_output_coordinate_coupling','kind':'known_real_valued_sensitivity_with_two_full_native_matrices',
         'expression':'G_h = W_U^T[diag(p)-p p^T]W_U; dh^T G_h dh = Var_p(W_U dh). Diagonal-only terms can differ because signed cross-coordinate terms remain.',
         'variables':'h is the final post-RMS query vector. p is native full-vocabulary probability; W_U is the actual head weight interpreted in real arithmetic. All2560x2560matrix entries and all151936readout rows are used.',
         'evidence':'Two predetermined natural/QA queries. Direct full-vocabulary variance versus complete matrix action checked; actual BF16 finite logit delta is reported separately and need not equal real W_U dh.'},
        {'id':'population_output_geometry','kind':'known_total_covariance_identity_plus_source_heldout_metric_estimation',
         'expression':'mu_q=W_U^T p_q; mean(G_q)=W_U^T diag(mean(p_q)) W_U - mean(mu_q mu_q^T). G(mean(p_q))-mean(G_q)=Cov(mu_q).',
         'variables':'Complete original readout rows/coordinates. Three train-only groups: pooled, natural, QA. Globally disjoint article groups fit/evaluate metric estimators; no coordinate projection or pruning.',
         'evidence':'640original queries; six full matrices plus complete dh/mu vectors. Predicts local output sensitivity given already observed state errors, NOT future tokens or semantic answers. Parent behaviors/materials were already observed; the held-out split applies only to this metric estimator.'},
        {'id':'native_rounding','kind':'known_exact_accounting_identity_at_a_measured_counterexample',
         'expression':'s=r+a+m in FP64; q=y_native-s; E(y_native)=E(s)+2<s,q>/D+E(q).',
         'variables':'y_native is actual sequential BF16 residual output. D is full native width, E(v)=sum_j v_j^2/D; this is squared amplitude, not physical energy.',
         'evidence':'Full2560coordinate replay at the preselected largest discrepancy; no discrete reset follows from cancellation or finite rounding.'}]
    graph={'external':'Original document/family IDs, actual text/token prefixes, human questions and retrospective answer/support spans; typed relations retain their annotation scope.',
        'internal':'Embedding identity -> complete native HiddenState boundaries -> actual normalized inputs/all MLP units/parameter indices -> native output distribution and explicitly available history.',
        'association':'Condition statistics and fitted full-coordinate operator candidates -> new-source checks -> true remaining-network probability compilation -> own-history diagnostics.',
        'unclosed':'The external labels have not yielded a universal transferable update algorithm for knowledge, reasoning and syntax. Layout adjacency and similar responses are not proof of causal semantic identity.'}
    save(BASE/'theory_snapshot.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'theory_name':'条件化输出场闭合理论（RDC）',
        'global_closed_theorem_added':False,'new_mathematics_claimed':False,'puzzles':puzzles,'formulas':formulas,'global_atlas_interface':graph,
        'scope':'历史条目继承已审计的既有拼图账本，本轮未重跑；新增条目链接实际完成的产物。该总账不证明AGI或唯一的人脑机制。'})
    print('OPERATOR_THEORY_INVENTORY_COMPLETE',len(puzzles),len(formulas),flush=True)


if __name__=='__main__':main()
