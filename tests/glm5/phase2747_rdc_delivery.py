"""Evidence-gated formation handoff; preserve every inherited scoped puzzle/formula."""
from copy import deepcopy
from rdc_formation_common import *


TASKS=[
    ('actual_training_formation',
     ['material/protocol.json','training/result.json','training_analysis/result.json','gradient/result.json','verification/material_trace_identities.json'],
     'All2048training examples consumed exactly once in each of six128step runs; all74711040block16parameters, fourcheckpoints/run;256example fullparameter gradient panel.'),
    ('precision_and_actual_parameter_radius',
     ['radius/result.json','radius_analysis/result.json'],
     'Forty predeclared radius/direction controls, FP32 bridge and nativeBF16 separately;896positions and full151936vocabulary per deployment.'),
    ('probability_calibration_alternatives',
     ['calibration/result.json','calibration_analysis/result.json'],
     'All54variants,504candidate grid with validation-only selection. Frequency priors from training corpus; all54argmax arrays unchanged.'),
    ('complete_prefix_parameter_propagation',
     ['parameter_propagation/native/result.json','parameter_propagation/smooth/result.json','parameter_propagation/analysis/result.json'],
     'All64prefixes, sixactual parameterdirections, threefinite scales, fullprefix/lastposition/nochange controls;20downstream blocks,768adjoint checks and1152smooth finite endpoints.'),
    ('paired_text_program_readout_and_own_history',
     ['transfer/readout_result.json','transfer/analysis/result.json','program_own_history/result.json','program_own_history/analysis/result.json','program_own_history/terminal_review/result.json'],
     'Fourtextdirections,64paired semanticgroups,100diagnosticqueries,fivepaths;128000fullVcandidate readouts. Separately32heldoutprogramgroups x6one-shot own-history branches.'),
    ('native_and_learned_own_history',
     ['own_history/analysis/result.json','own_history/terminal_review/result.json','engineering/microbatch/qwen4_qualification/result.json',
      'engineering/microbatch/current_qualification.json','engineering/microbatch/slice_reader_current.json',
      'engineering/microbatch/unit_current.json',
      'own_history/qwen14/native/wave_pilot_replay.json','own_history/glm4/native/wave_pilot_replay.json'],
     'Three original unquantized models and sixactual trainedQ4deployments,512expressions each. Everyownstep fullpostnorm;44predeclared expressions/run everyH. Native arithmetic/fullpilot replay qualification.'),
    ('three_atlas_client_and_integrity',
     ['figures/training_v2/visual_review.json','figures/propagation_v1/visual_review.json','figures/probability_v2/visual_review.json','figures/history_v1/visual_review.json',
      'client/code_checks.json','verification/result.json'],
     'Actual reviewed scientific images, exact originalcoordinate API arrays, native/program trajectory branch selectors and all original checkpoint/archive/MEMO integrity checks.')]


def main():
    finish=OUT/'delivery_manifest.json'
    if finish.exists():
        result=read(finish);assert result['all_passed'] and result['phase2747_complete'];return result
    prior_path=OUT/'inherited_phase2746_theory.json';prior=read(prior_path)
    assert len(prior['puzzles'])==45 and len(prior['formulas'])==42
    assert sha(prior_path)==read(BASE/'phase2746/delivery_manifest.json')['theory_snapshot_sha256']
    assert sha(BASE/'theory_snapshot.json')==sha(prior_path),'Unexpected concurrent theory change; preserve and review before delivery'
    evidence={};tasks=[]
    for name,files,scope in TASKS:
        for file in files:
            path=OUT/file;value=read(path)
            if 'all_passed' in value:assert value['all_passed'],file
            evidence[file]={'path':path.relative_to(BASE).as_posix(),'sha256':sha(path)}
        tasks.append({'task':name,'state':'executed_verified_with_explicit_boundaries','actual_scope':scope,
            'evidence':[evidence[file] for file in files]})
    for pointer in ['current_regression.json','current_evidence_regression.json','current_history_regression.json']:
        current=read(OUT/'client'/pointer);path=ROOT/current['result']
        assert sha(path)==current['result_sha256'] and read(path)['all_passed']
        review=ROOT/current['image_directory']/'visual_review.json';assert read(review)['all_passed']
        for p in [path,review]:evidence[p.relative_to(OUT).as_posix()]={'path':p.relative_to(BASE).as_posix(),'sha256':sha(p)}
    training=read(OUT/'training/result.json');own=read(OUT/'own_history/analysis/result.json')
    program=read(OUT/'program_own_history/analysis/result.json')
    assert own['complete_runs']==9 and not own['partial'] and len(program['checks'])==192
    own_runs=[]
    for report in own['reports']:
        path=BASE/report['run_result']['path'];assert sha(path)==report['run_result']['sha256']
        r=read(path);assert r['all_passed'] and r['trajectories']==512
        own_runs.append({'model':report['model'],'variant':report['variant'],'trajectories':512,
            'generated_tokens':r['actual_generated_tokens'],'summary':report['summary'],
            'frozen_scoring_audit':report['scoring_audit'],
            'evidence':report['run_result']})
    summary={'actual_backward_training_examples':training['actual_backward_training_examples'],
        'training_runs':len(training['runs']),'own_runs':own_runs,
        'own_trajectories':sum(r['trajectories'] for r in own_runs),
        'own_generated_tokens':sum(r['generated_tokens'] for r in own_runs),
        'common_native_ability':own['common_ability'],
        'language_supplemental_unblinded_terminal_review':read(OUT/'own_history/terminal_review/result.json')['summary'],
        'program_summary':[r for r in program['summary'] if r['depth']=='all'],
        'program_generated_tokens':program['actual_generated_tokens'],
        'program_supplemental_unblinded_terminal_review':read(OUT/'program_own_history/terminal_review/result.json')['summary']}
    puzzle={'phase':'2747',
        'retained_puzzle':'真实局部参数训练形成、全坐标参数半径/精度/概率对照、完整前缀跨层传播，以及完整词表和自身历史的同来源证据链。',
        'evidence_status':'verified_restricted_training_and_conditional_propagation_with_prediction_and_behavior_limits',
        'common_phenomenon':'自然NLL、关系排序、格式、停止和自行生成的历史可分离；完整前缀传播的平均收益伴有方向、语言族和有限幅度的反例。',
        'candidate_rule':'用真实完整参数方向和完整前缀计算跨层变分预测，并与末位置、不变化、实际有限部署和校准控制逐项比较；已知微分工具非自主语言规律。',
        'native_parameter_structure':'Q4block16全部74711040gate/up/down参数；实际训练轨迹、全部9728单元与block16至35原生后续计算接续，非单神经元独立语义。',
        'unseen_composition_prediction':'192新中文来源及冻结关系/程序保留组；真实训练概率收益经温度校准缩小，关系自身历史未随自然NLL同步改善；不把暴露过的诊断材料改称新确认。',
        'training_formation_evidence':'3监督条件x2种子，每次完整2048例、128次真实梯度更新；FP32训练桥与BF16部署分开；只恢复本次受限继续训练，不恢复原始预训练。',
        'boundary':'正确配对映射未稳定胜错误配对，类别质量不确定数字内部排序；一次替换仅首步生效。三模型和全部失败轨迹按原冻结评分保留，无无限组合或唯一语义齿轮结论。'}
    theory=deepcopy(prior)
    theory.update(timestamp=stamp(),source=snapshot(__file__),inherited_snapshot_sha256=sha(prior_path),
        puzzles=prior['puzzles']+[puzzle],phase2747_evidence=evidence,phase2747_tasks=tasks,
        phase2747_actual_summary=summary,
        inheritance_audit={'all45prior_puzzles_preserved_exactly':True,'all42prior_formulas_preserved_exactly':True,
            'historical_results_not_rerun':True,'no_reinterpretation_of_historical_response_bundle_as_arbitrary_H':True},
        RDC_primary_formula_changed=False,global_closed_theorem_added=False,new_mathematics_claimed=False,
        first_principles_insight='需要提取的是条件如何改变共享参数的有效读写关系，以及这些关系如何在自行生成的历史中保持可预测性。概率尺度、关系选择和终止是不同检验，不能互相代替。',
        remaining='未形成普遍语言闭合结构。继续同一授权目标，优先研究自然轨迹中可推广的条件依赖与原生读写组织；新阶段需在实际执行前冻结材料、输入权限和强对照。')
    theory['current_algorithm_definitions']=prior['current_algorithm_definitions']+[
        {'id':'complete_parameter_direction_variation','kind':'known_multivariate_chain_rule_tested_in_restricted_domain',
         'expression':'dotH_(l+1)=D_H F_l[dotH_l]+D_theta F_l[DeltaTheta]; dotz=W_U D_N[dotH_L]; H includes allvisibleprefixpositions.',
         'variables':'Allsixactual learned block16parameterdirections; remaining original weights fixed; fullprefix carried through attention/MLP/norm. SmoothsamevaluedFP32reference is separate from nativeBF16finite endpoints.'},
        {'id':'precision_specific_centered_first_order_output','kind':'known_first_order_approximation_with_observed_native_endpoint',
         'expression':'phat_s^F=softmax(z0^F+s*dotz); phat_s^B=softmax(z0^B+s*dotz); E_s=KL(p_s||phat_s).',
         'variables':'z0^B is extra observed originalnative endpoint; this centering is not early-only extraction. Lastposition and nochange references included; smallnativeBF16effects need not be predicted by the smoothderivative.'},
        {'id':'validation_only_total_concentration_calibration','kind':'fitted_probability_control_not_identified_semantic_residual',
         'expression':'pi_k=(c_k+beta/V)/(sum_j c_j+beta); pcal_k=(1-alpha)*softmax(z/T)_k+alpha*pi_k.',
         'variables':'Twoactual trainingcorpus priors,7temperatures,6concentrations,6mixture coefficients;192source-separated validationpositions select; all54actual variants evaluated separately.'},
        {'id':'one_shot_readout_same_token_continuation','kind':'deterministic_execution_implication_verified_by_complete_array_replay',
         'expression':'If C0prime=C0, thetaPrime=theta and argmax(z0prime)=argmax(z0), then the subsequent greedy native token/cache/state trajectories coincide under identical execution.',
         'variables':'Only initial postnormreadout or goldfree categorybias changed; initialKV/nativehidden and everylaterparameter untouched. Not a general claim for state/cache/parameter interventions or stochastic decoding.'}]
    theory['global_atlas_interface']={**prior['global_atlas_interface'],
        'phase2747_external_increment':'2048training examples with authentic nexttoken identity and fivecontrolled families,192validation,512diagnostic,192newChinesedocuments;64textprogram groups/100fixedqueries.',
        'phase2747_internal_increment':'All74711040localparameters at fourcheckpoints/run,128step scalartraining traces and completegradient panels; every20block current-coordinate propagation and declared allprefixcomputation;9ownruns everypostnorm with44fullHexpressions/run.',
        'phase2747_association_increment':'Stable source/model/deployment/query/step IDs join actual training, frozen prediction, fullVcalibration, one-shot readout and native self-generated trajectories. Outcome-conditioned commonability explicitly marked descriptive.'}
    assert theory['puzzles'][:45]==prior['puzzles'] and theory['formulas']==prior['formulas']
    save(OUT/'theory_snapshot.json',theory)
    save(BASE/'theory_snapshot.json',theory)
    manifest={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'phase2747_complete':True,'goal_complete':False,
        'tasks':tasks,'evidence':evidence,'actual_summary':summary,'theory_snapshot_sha256':sha(OUT/'theory_snapshot.json'),
        'puzzles':len(theory['puzzles']),'inherited_formulas':42,'new_unified_formula':False,
        'science_boundary':'Alldeclared tests executed and audited; informative failures remain failures. No universal language mechanism completion claim.',
        'archive_audit':read(OUT/'verification/result.json')['archive_coverage'],
        'retention':'All checkpoints/displayed fields preserved.3146task metadata files losslessly relocated with explicit junction and fullSHAaudit; no material deletion.',
        'next_same_goal':True,'next':'Phase2748 sameauthorized encoding research; actual plan/data contracts must be frozen before new test execution.'}
    save(finish,manifest)
    save(BASE/'status.json',{'timestamp':stamp(),'phase2745':'verified_complete','phase2746':'verified_complete',
        'phase2747':'verified_complete','phase2748':'next_same_goal_not_yet_executed','goal_complete':False,
        'phase2745_manifest':'delivery_manifest.json','phase2746_manifest':'phase2746/delivery_manifest.json',
        'phase2747_manifest':'phase2747/delivery_manifest.json'})
    print('PHASE2747_DELIVERED',len(tasks),len(theory['puzzles']),len(theory['formulas']),summary['own_trajectories'],flush=True)
    return manifest


if __name__=='__main__':main()
