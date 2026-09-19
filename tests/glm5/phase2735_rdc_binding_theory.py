"""Preserve scoped RDC history and append evidence-linked candidate interfaces."""
from rdc_binding_common import *

def main():
    for p in ['binding_live/result.json','scale/suite_result.json','format_content/decomposition_result.json','format_content/projection_condition_audit.json','format_content/autonomous/result.json','analysis/behavior.json','signed_source/result.json','signed_source/math.json']:
        assert (BASE/p).exists(),p
    prior=read(LAW/'theory_snapshot.json');puzzles=[]
    for p in prior['puzzles']:
        r=dict(p,evidence_status='inherited_scoped_evidence_not_rerun_in2732_2735')
        if 'artifacts' in r:r['artifacts']=['../rdc_law_campaign_20260911/'+s for s in r['artifacts']]
        if r['phase'] in ('2729','2730'):
            r['boundary']+=' Phase2732追加审计：前轮组合缺席证书只覆盖有UD解析的英语自然训练窗口；256条CMRC/QA训练材料未解析，不证明泄漏但不能宣称全拟合缺席。'
        puzzles.append(r)
    additions=[
      (2732,'完整来源—位置—粗角色核、严格英语拟合集及独立共享head句法组合确认；全坐标/全部来源保留。',
       '相比来源均值收益小，尚未稳定胜过角色打乱；有限二阶矩和归一化可能损失必要信息，句法连接不是语义组合。',
       'full_coordinate_candidate_and_scoped_connected_confirmation',['prediction/frozen.json','confirmation/result.json','analysis/result.json']),
      (2733,'三个block全部单元双线性参数Gram、当前全74711040参数梯度span、四条真实中层32步训练及Alpha/Beta/Gamma实际更新。',
       '连贯续训NLL优势不等于准确率普遍提高，乱序也可学习；gold更新是监督oracle，代理抑制重复可损害合法答案；不恢复原预训练。',
       'exact_native_relations_and_restricted_continuation_training',['native_bilinear/result.json','middle_training/result.json','alpha_natural/result.json','gradient_span/result.json','beta_updates/result.json']),
      (2734,'冻结规则与中层增量的自然自身历史部署、三本地非量化模型原生响应、完整只读图谱与来源端点审计。',
       '各分支保留自己的token/KV；自然续写没有唯一gold，8token限制造成大量截断；不同架构/训练/tokenizer混杂，不能因规模归因或对应同坐标。',
       'own_history_deployment_and_limited_native_replication',['binding_live/result.json','scale/qwen4/result.json','scale/qwen14/result.json','scale/glm4/result.json']),
      (2735,'针对解释开头/数字格式断点分离loss与全梯度，冻结128新六步表达并核对长输出及自身历史；另由来源矩碰撞反例发展有符号对照，在128新自然窗口检验。',
       '内容梯度仍依赖目标与数字候选集，非正交项不是独立语义比例；内容投影的原尺度数值秩75/96，不能把弱方向称为精确零或语义无关。合成来源矩反例不证明自然状态可达；有符号对照未胜旧验证集，仍按原冻结选择报告。',
       'automatic_same_goal_loss_decomposition_and_source_information_diagnostics',['format_content/decomposition_result.json','format_content/native_capture_result.json','format_content/autonomous/result.json','analysis/behavior.json','verification/source_moment_collision/result.json','signed_source/result.json'])]
    for phase,puzzle,boundary,status,paths in additions:
        puzzles.append({'phase':str(phase),'retained_puzzle':puzzle,'boundary':boundary,'evidence_status':status,'artifacts':paths})
    formulas=[dict(f,evidence=f['evidence']+' Inherited with its original domain/precision limits; no new historical rerun claimed.') for f in prior['formulas']]
    formulas.extend([
      {'id':'full_source_position_role_pair_kernel','kind':'fitted_finite_moment_candidate',
       'expression':'b_ij=(q_i·q_j+e_i·e_j)/(2D); S_ij=mean_st[(u_is·u_jt/D)^2 (p_is·p_jt)(1+r_is·r_jt)]; K_ij=1+b_ij+S_ij+b_ij S_ij.',
       'variables':'u/q/e are full-coordinate per-vectorRMS normalized causal H12sources/currentH12/currentembedding; p=(1,z,z²) for source position z in[-1,0]; r=frozen six-role ridge scores, not currentgold roles. All source/coordinate pairs are included before aggregation.',
       'evidence':'84capacity/decoder candidates selected byvalidation, new128connected/matchedwindow evaluation and position/role shuffles. Correct role binding did not robustly beat role-shuffled control.'},
      {'id':'native_same_unit_bilinear_parameter_relation','kind':'known_exact_factorization_evaluated_on_all_native_units',
       'expression':'C[k,i,j]=Wg[k,i]*Wu[k,j]; a_k(x)=sigmoid(Wg[k,:]x) x^T C_k x; <C_k,C_l>_F=(Wg[k,:]·Wg[l,:])(Wu[k,:]·Wu[l,:]).',
       'variables':'One C per actualMLPunit; D2560 and M9728 inQwen4. Exact full factors avoid a255GBdense same-unit tensor without discarding any coordinate. Sigmoid and input remain required.',
       'evidence':'All94,633,984unit pairs inblocks6/16/35; factors and fullGram persisted, identitynumerics checked. Parameter similarity is not semantic equivalence.'},
      {'id':'complete_declared_gradient_span_projection','kind':'regularized_finite_linear_algebra_with_scope_limited_transfer_tests',
       'expression':'G_ij=<grad_theta L_i,grad_theta L_j>; G_epsilon^+=sum_(lambda_k>epsilon*lambda_max) v_k v_k^T/lambda_k; c=(G_TT)_epsilon^+ G_Tv; P_(T,epsilon) v=sum_i c_i grad_theta L_i; theta_new=theta-eta P_(T,epsilon) v/||P_(T,epsilon) v||.',
       'variables':'T includes every declared trainingquery; all74711040parameter innerproducts and allGram entries are retained. Epsilon=1e-9 regularizes the inverse, not an exact-rank or semantic-zero claim. Phase2733 retained96/96; Phase2735 content rawGram retained75/96 while row-normalized diagnostic retained96/96. Eta=.02/.10 is the FP32 targetnorm; BF16 realizednorms differ.',
       'evidence':'Natural coherent/order96query spans and EN/ZH/Python96query spans; actual positive/reverse/random controls, depth/probability/autonomous evaluation. The content inverse excludes21 weak eigen-directions at its declared cutoff; nineteen have positive computed eigenvalues. All underlying fields/gradients remain saved, and the normalized diagnostic was not used to retune the predictor.'},
      {'id':'middle_block_restricted_training_update','kind':'actual_training_protocol_not_historical_formation_law',
       'expression':'g_t=(1/4)sum_(i in batch_t) grad_theta mean_(q in row_i) CE_q; theta_(t+1)=theta_t-0.02 min(1,1/||g_t||)g_t.',
       'variables':'Theta consists only ofblock16gate/up/down. Eight target positions per row;32steps; two fixed draws seeds; coherent/order controls share suffix,labels,length anddraws. Native17..35suffix participates inautograd; FP32MLP/BF16bridge is explicit.',
       'evidence':'All4trajectories and original/bridgebaselines; coherent NLL decreases but argmax not uniformly better. Autograd through dtype casts follows library training convention, not derivative of a discrete finite-precision map or originalpretraining identification.'},
      {'id':'full_content_format_loss_gradient_identity','kind':'known_probability_chain_rule_with_new_experimental_diagnostic',
       'expression':'P_D=sum_(v in D)p_v; L_full=-log p_y=-log(p_y/P_D)-log P_D=L_content+L_format; g_full=g_content+g_format; ||g_full||²=||g_content||²+||g_format||²+2<g_content,g_format>.',
       'variables':'D is the eight declared single-digit targettokens; y inD is a supervisedtarget. Content scoring conditions onD, not native generation. Completeparameter gradient factors and signedcross terms use all declared nativeparameters.',
       'evidence':'768existingprogram diagnostic, all-parameter Gramidentity/nativeautograd checks, directions fit only onoldtrain and evaluated on128frozennewdepth6expressions plus independent own-token histories. Direct conditionalCE and FP64 probability normalization repair documented FP32 cancellation; MLP/logits remain FP32. Does not uniquely separate all possible language content/style.'},
      {'id':'source_moment_ambient_collision_certificate','kind':'known_sign_invariance_applied_to_actual_frozen_probe_not_native_reachability',
       'expression':'Let v in ker(C_role^T), RMS(v)=1. H_a=(v,-v,h_2,...,q), H_b=(-v,v,h_2,...,q). Then mean(H_a)=mean(H_b), Phi_square(H_a)=Phi_square(H_b), K_a,*=K_b,* although H_a != H_b.',
       'variables':'C_role is the actual2560by6 linear part of the frozen affine role probe, of rank6. Phi_square retains h outer h paired with source position and affine-probe-derived role. Query/embedding and positions are unchanged; allcoordinates included.',
       'evidence':'All2560squared outer-product entries and actualprobe null vector checked;16realreference prefix kernels agree. These synthesized histories are not asserted reachable in the LLM. This refutes ambient injectivity only: it does not itself prove downstream outputs differ, nor refute predictive sufficiency on every reachable language domain.'},
      {'id':'signed_source_position_role_moments','kind':'known_exact_feature_kernel_with_frozen_limited_forecast_test',
       'expression':'T_i=(1/n_i)sum_s h_is tensor p_is tensor [1,r_is]; S_ij=<T_i,T_j>/D; K_signed=1+b_ij+S_ij+b_ij S_ij. K_mix=0.5 K_original/mean_train diag(K_original)+0.5 K_signed/mean_train diag(K_signed).',
       'variables':'Allsource tokens and nativecoordinates contribute before aggregation; position basis(1,z,z²), six frozen role scores and scalarintercept. T has2560x3x7entries without coordinate/rank selection. DirectMLP2560coordinate ridge, effective df128.',
       'evidence':'Independent allsourcepair versus feature-innerproduct checks about1e-15, synthetic sign collision separated; original rule still wins oldvalidation inbothblocks. New128natural windows and128newdepth6programs evaluated with selection unchanged, withposition-only androleshuffled controls.'}])
    assert len(puzzles)==34 and len(formulas)==23
    save(BASE/'theory_snapshot.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'theory_name':prior['theory_name'],
      'inherited_snapshot_sha256':sha(LAW/'theory_snapshot.json'),'puzzles':puzzles,'formulas':formulas,
      'RDC_primary_formula_changed':False,'global_closed_theorem_added':False,'new_mathematics_claimed':False,
      'numerical_projection_scope':read(BASE/'format_content/projection_condition_audit.json'),
      'signed_material_identity_scope':read(BASE/'signed_source/identity_recovery/result.json'),
      'five_correspondence_columns':['common_phenomenon','candidate_rule','native_parameter_structure','unseen_composition_prediction','training_formation_evidence'],
      'global_atlas_interface':{'external':'StrictUDnatural window/document families, connected edges with prefix endpoint visibility, separately interpreted programs and prospective depth6 cases.',
       'internal':'AllH12sources/fullanchorcoordinates, allMLPunits/fullscalarfactor identities, actualmiddle trainingdeltas and eachbranch ownhistory.',
       'association':'Frozen source kernels, present-loss parameter spans, local finite steps and behavior connected by stable IDs and explicit input/label/precision scope.',
       'unclosed':'No sufficient language state, no unique semantic gear or universal composition operator, no original training reconstruction.'},
      'next_big_question':'Separate identity/norm/format from source-head-dependent semantic binding, and learn an available-prefix relation update whose all-coordinate predictions and actual parameter learning survive unseen relation operations and autonomous history. Do not repeat the same role-shuffle non-result by mere template substitution.'})
    print('BINDING_THEORY_COMPLETE',len(puzzles),len(formulas),flush=True)

if __name__=='__main__':main()
