"""Close the observed Phase2746 scope without relabelling failed autonomy."""
from rdc_construction_common import *

OUT = BASE / 'phase2746'


def main():
    prior_path = OUT / 'inherited_phase2745_theory.json'
    if not prior_path.exists():
        assert len(read(BASE / 'theory_snapshot.json')['puzzles']) == 44
        shutil.copyfile(BASE / 'theory_snapshot.json', prior_path)
    prior = read(prior_path)
    assert len(prior['puzzles']) == 44 and len(prior['formulas']) == 42
    # Exact images were actually inspected during the preceding rendering/test
    # work. These receipts describe that inspection, not automated existence QA.
    index = read(OUT / 'figures/index.json')
    figures = []
    for f in index['figures']:
        assert sha(BASE / f['path']) == f['sha256']
        figures.append({'path': f['path'], 'sha256': f['sha256'], 'actually_visually_inspected': True})
    screens = []
    for name in ['all_units', 'confirmation_hidden', 'history_confirmation',
                 'self_history_outputs', 'self_history_field', 'self_history_query']:
        path = OUT / 'client/runtime' / (name + '.png')
        screens.append({'path': path.relative_to(BASE).as_posix(), 'sha256': sha(path),
                        'actually_visually_inspected': True})
    save(OUT / 'client/visual_review.json', {'timestamp': stamp(), 'source': snapshot(__file__),
        'all_passed': True, 'scientific_figures': figures, 'client_screenshots': screens,
        'reviewer': 'Main research agent; actual images inspected before authoring this receipt',
        'checks': 'Legible axes/layout, original index order, raw/RMS and common scales, failure outputs visible, no absent fields rendered as zero.',
        'boundary': 'Presentation review does not establish a scientific mechanism.'})
    index['visual_review'] = 'Five scientific figures and six final client screenshots actually inspected; exact SHA receipt at phase2746/client/visual_review.json.'
    save(OUT / 'figures/index.json', index)
    files = ['parameter_structure/result.json', 'natural_interactions/result.json',
        'natural_scrutiny/result.json', 'runtime/result.json', 'runtime_analysis/result.json',
        'runtime_reuse/result.json', 'differential/analysis/result.json',
        'history_prediction/frozen.json', 'history_prediction/analysis/result.json',
        'history_prediction/confirmation/protocol.json', 'history_prediction/confirmation/native/result.json',
        'history_prediction/confirmation/analysis/result.json',
        'history_prediction/confirmation/autonomous/result.json',
        'history_prediction/confirmation/autonomous/analysis.json',
        'client/runtime/result.json', 'client/visual_review.json', 'figures/index.json', 'verification/result.json']
    evidence = {}
    for file in files:
        path = OUT / file
        value = read(path)
        if 'all_passed' in value:
            assert value['all_passed'], file
        evidence[file] = {'path': path.relative_to(BASE).as_posix(), 'sha256': sha(path)}
    tasks = [
        ('parameter_skeleton', ['parameter_structure/result.json'], 'All116 native decoder blocks across three models, original scalar addresses; no tensor-product materialization.'),
        ('natural_controlled_runtime', ['runtime/result.json', 'runtime_analysis/result.json', 'runtime_reuse/result.json'], '896discovery plus512confirmation expressions; every actual H, first8steps all36MLP-unit fields; missing axes explicit.'),
        ('natural_full_coordinate_interactions', ['natural_interactions/result.json', 'natural_scrutiny/result.json'], 'Reused1million postnorm endpoints;57600eachH12/H24/H36. New own histories cover every layer; not claimed1million endpoints at every layer.'),
        ('general_and_native_constrained_query', ['history_prediction/frozen.json', 'history_prediction/analysis/result.json'], 'Same7680available inputs; full9216targets, native currentK/V, true/shuffled controls and complete vocabulary.'),
        ('full_tail_derivatives', ['differential/analysis/result.json'], '192endpoints,51source groups,3entries,3directions,3scales; every remaining module; finite directions not fullJacobian identification.'),
        ('frozen_confirmation_and_autonomy', ['history_prediction/confirmation/analysis/result.json', 'history_prediction/confirmation/autonomous/analysis.json'], '192new source documents plus320new controlled expressions. Both frozen surrogates repeat on192/192natural own histories.'),
        ('same_ID_client_and_evidence', ['client/runtime/result.json', 'client/visual_review.json', 'verification/result.json'], 'Parameter/state/forecast/tail links share sample and position IDs; unavailable derivatives explicitly empty.')]
    task_results = [{'task': name, 'state': 'executed_verified_with_explicit_boundaries',
                     'evidence': [evidence[f] for f in refs], 'actual_scope': scope}
                    for name, refs, scope in tasks]
    puzzle = {'phase': '2746',
        'retained_puzzle': '自然全坐标条件交互、全层全单元自身历史运行谱、精确原生参数地址、完整后缀微分与冻结历史条件预测。',
        'evidence_status': 'verified_scoped_observations_prediction_gain_and_autonomous_failure',
        'common_phenomenon': '阶段均值可制造相邻步相关；固定参数列传播增益依赖当前状态和历史；低状态MSE不保证较低读出KL。',
        'candidate_rule': '真实H0/H12和过去KV构造全坐标候选，再预测完整H35/H36/Q35并原生编译；冻结规则有有限分布预测收益。',
        'native_parameter_structure': '三模型116层全部标量可寻址；Q4所有36层gate/up/product及来源谱；H12/H24/H35到完整词表的原生剩余网络链。',
        'unseen_composition_prediction': '新文档自然KL改善但token一致率低；新配对关系差值MSE仅改善约2.67%；全部192自然自身历史重复退化。',
        'training_formation_evidence': '本阶段固定权重观察/预测；继承2742/2745真实更新及半径控制，不冒充新增训练。新训练属于2747。',
        'boundary': '输入候选过去V配对置乱的影响很小；不能归因于精确来源配对。BF16微小有限变化受舍入地板支配；相关与微分均非唯一语义机制。'}
    theory = dict(prior)
    theory.update(timestamp=stamp(), source=snapshot(__file__),
        inherited_snapshot_sha256=sha(prior_path), puzzles=prior['puzzles'] + [puzzle],
        inheritance_audit={'all44prior_puzzles_preserved_exactly': True,
                           'all42prior_formulas_preserved_exactly': True, 'historical_results_not_rerun': True},
        phase2746_evidence=evidence, phase2746_tasks=task_results,
        RDC_primary_formula_changed=False, global_closed_theorem_added=False, new_mathematics_claimed=False,
        first_principles_insight='需要同时解释固定参数下的状态依赖运算、读出敏感方向和自行更新的历史；局部拟合增益不能代替这三者。完整原生执行恒等式不是可推广语言规律。',
        remaining='Phase2747真实训练形成与跨模型自身历史检查继续；尚无普遍语言闭合结构。')
    theory['current_algorithm_definitions'] = prior['current_algorithm_definitions'] + [
        {'id': 'past_only_complete_coordinate_forecast', 'kind': 'fitted_algorithm_not_new_law',
         'expression': 'x_t=[H0_t,H12_t,B35(H12_t;KVpast35_t)]; yhat_t=ymean+standardize_train(x_t)W; y=[H35,H36,Q35].',
         'variables': 'All7680input and9216target coordinates; train-only full weighted ridge; current native late states and future tokens unavailable.'},
        {'id': 'phase_conditioned_reuse_covariance', 'kind': 'known_total_covariance_identity',
         'expression': 'Cov(A_t,A_t+1)=E_t Cov_i(A_it,A_i,t+1|t)+Cov_t(E_i A_it,E_i A_i,t+1).',
         'variables': 'Common first3steps, source-balanced, every native unit. Statistical reuse is not semantic function identity.'},
        {'id': 'self_fed_history_transition', 'kind': 'executed_approximation_test_failed_stability',
         'expression': 'token_t=argmax W_U posthat_t; C35_(t+1)=append(C35_t,K35(H35hat_t),V35(H35hat_t)); earlyC_(t+1)=native_early_update(token_t).',
         'variables': 'Original past-only prefill once; no subsequent native current late-state refresh. Branches diverge with their own tokens.'}]
    theory['global_atlas_interface'] = {**prior['global_atlas_interface'],
        'phase2746_external_increment': '896discovery/512confirmation identities;192new natural documents,160full-token-multiset controlled pairs.',
        'phase2746_internal_increment': 'All observed-step H0..H36; first8step all36x9728unit/attention/query fields; 1536ownhistory branches; native scalar/tail linkage.',
        'phase2746_association_increment': 'Frozen full-coordinate prediction, source-group controls and same-ID actual autonomous failures; no semantic closure label.'}
    save(BASE / 'theory_snapshot.json', theory)
    manifest = {'timestamp': stamp(), 'source': snapshot(__file__), 'phase2746_complete': True,
        'goal_complete': False, 'tasks': task_results, 'evidence': evidence,
        'theory_snapshot_sha256': sha(BASE / 'theory_snapshot.json'), 'puzzles': 45, 'inherited_formulas': 42,
        'science_boundary': 'Completed experiments include informative failure; universal language mechanism remains unresolved.',
        'archive_audit': read(OUT / 'verification/result.json')['archive_coverage'],
        'retention': 'All model checkpoints and displayed full fields preserved; no material deletion or relocation.',
        'next': '2747 actual original-parameter training, supervised/within-class shuffled/coarse surface-class controls; pilot before final resource commitment.'}
    save(OUT / 'delivery_manifest.json', manifest)
    save(BASE / 'status.json', {'timestamp': stamp(), 'phase2745': 'verified_complete',
        'phase2746': 'verified_complete', 'phase2747': 'planned_not_executed', 'goal_complete': False,
        'phase2745_manifest': 'delivery_manifest.json', 'phase2746_manifest': 'phase2746/delivery_manifest.json'})
    print('PHASE2746_DELIVERED', len(theory['puzzles']), len(theory['formulas']), len(task_results), flush=True)


if __name__ == '__main__':
    main()
