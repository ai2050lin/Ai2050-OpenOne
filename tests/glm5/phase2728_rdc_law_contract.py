"""Audit attachments against retained primary artifacts; freeze falsifiable research scope."""
import re
from collections import Counter
from rdc_law_common import *

ATTACHMENTS = [
    Path('C:/Users/Admin/.codex/attachments/803e79d5-93b1-4ef4-85d3-2a66170a95d2/pasted-text.txt'),
    Path('C:/Users/Admin/.codex/attachments/5169a840-0649-4896-82c3-cb28eeaa7a73/pasted-text.txt'),
    Path('C:/Users/Admin/.codex/attachments/7bbcb217-e916-4643-9e44-69549257a7e3/pasted-text.txt'),
]


def main():
    if (BASE / 'plan.json').exists():
        print('LAW_CONTRACT_ALREADY_FROZEN', flush=True)
        return
    memo_bytes = MEMO.read_bytes()
    phases = re.findall(r'^## Phase (\d+):', memo_bytes.decode('utf-8'), re.M)
    assert phases[-1] == '2727', phases[-4:]
    save(BASE / 'memo_prefix.json', {'bytes': len(memo_bytes), 'sha256': hashlib.sha256(memo_bytes).hexdigest(), 'last_phase': 2727})
    evidence_names = ['material_audit.json', 'capture/main/result.json', 'capture/confirmation/result.json',
        'qa_extension.json', 'qa_atlas/result.json', 'operators/result.json', 'operators/frozen.json',
        'structure/result.json', 'compiled/paired_audit.json', 'operations/result.json',
        'metric_followup/result.json', 'metric_followup/paired_audit.json',
        'metric_followup/population_geometry/result.json', 'theory_snapshot.json',
        'verification/model_checkpoint_fingerprints.json', 'delivery_manifest.json']
    evidence = [{'path': str((OPERATOR / p).relative_to(ROOT)), 'sha256': sha(OPERATOR / p)} for p in evidence_names]
    material = gzread(OPERATOR / 'material.json.gz')
    assert len(material) == 2048 and sum(len(r['prompt_ids']) for r in material) == 346920
    inv = read(OPERATOR / 'metric_followup/paired_audit.json')['same_prefix_rank_inversions']
    assert [(r['block'], r['locally_better_queries'], r['local_better_but_probability_worse_queries']) for r in inv] == [(6,458,92),(16,308,135),(34,488,107)]
    grams = [r for r in read(OPERATOR / 'structure/result.json')['signed_Gram_reports'] if r['stratum'] == 'ordinary']
    for r in grams:
        assert abs(np.asarray(r['mean_energy_normalized_signed_Gram']).sum() - 1) < 1e-5
    # Audit ALL Hotpot context titles, not only question IDs or support documents.
    from phase2724_rdc_operator_material import normalize
    old_titles = {normalize(r['title']).replace('_', '') for r in material if r['language'] == 'en'}
    hp = gzread(OPERATOR / 'qa_multihop_material.json.gz')
    import pyarrow.parquet as pq
    original_hp = {r['id']: r for r in pq.read_table(OPERATOR / 'sources/hotpot_distractor_validation_hf.parquet').to_pylist()}
    hp_titles = set()
    for r in hp:
        titles = {normalize(t).replace('_', '') for t, _ in r['context_paragraphs']}
        assert not titles & (old_titles | hp_titles), ('Prior multi-document overlap', r['sample_id'])
        raw_context = original_hp[r['source_id']]['context']
        assert r['context_paragraphs'] == [[t, s] for t, s in zip(raw_context['title'], raw_context['sentences'])]
        hp_titles.update(titles)
    corrections = [
        ['A numeric summary', 'retain', '2048/346920, full signed Gram sums and three same-prefix inversion counts checked against raw result files. Other historical numbers retain original scope, not all historical phases rerun.'],
        ['A Hotpot source separation concern', 'resolved_for_this_selected_cohort', 'All ten context title sets checked pairwise disjoint and disjoint from prior English encyclopedia titles. Question-ID source groups are safe for THIS selected set, not a general grouping rule.'],
        ['A/B/old MEMO every Hotpot question has10 paragraphs', 'corrected_from_original_source', '124of128 have10; other four have7,8,5,4 respectively. Every saved paragraph array equals its original downloaded Parquet source. Correct claim: all AVAILABLE original distractor paragraphs retained. No omitted contexts or changed behavioral result.'],
        ['A original parameter operators and output covariance', 'retain_with_boundary', 'Known identities and conditional empirical prediction; actual input x is provided, not inferred solely from semantic relations.'],
        ['B 99.9 percent is smooth/stable', 'reject', 'Complement of one amplitude-event threshold does not establish continuity, stability or semantic homogeneity.'],
        ['B no sparse gears exist', 'reject', 'A finite event census cannot exclude sparse conditional mechanisms or establish that every coordinate is functionally required.'],
        ['B 86 percent of language energy and energy conservation', 'correct', '0.860238 is one nonorthogonal, center-dependent squared-amplitude term divided by native output energy; cross terms and rounding remain. No physical conservation law or semantic attribution percentage.'],
        ['B K eigenvectors rotate violently / low resistance channels', 'unsupported', 'K is input-conditioned and generally nonsymmetric. No such spectral rotation/channel measurements were performed; K is not its full Jacobian.'],
        ['B Fisher proves a Riemannian language mechanism', 'reject', 'G is a positive-semidefinite post-final-normalization pullback; rank/nondegeneracy and any semantic manifold were not established. Raw/pre-normalization states require norm/tail derivatives.'],
        ['B Var(delta)=delta^T G delta', 'dimensionally_corrected', 'For residual-coordinate dn, Var_p(W_U dn)=dn^T G dn. Logit delta and state dn must not share an ambiguous symbol. Finite BF16 changes additionally contain numerical effects.'],
        ['B cross terms explain every ranking inversion', 'reject', 'A real output metric accounts for chosen endpoint directions; propagation through the tail and finite path also matter, and cross terms need not dominate every query.'],
        ['B KV drift uniquely causes repeat collapse', 'unsupported', 'Prefill and decode intervention scope, altered current computation and self-selected history were not independently isolated.'],
        ['B Top-K eigenvectors are semantic gears', 'exclude_from_primary_plan', 'Violates requested full-coordinate discovery and presupposes unproved semantics. Keep complete-coordinate/factor actions and simple baselines.'],
        ['B Apple example, syntax high frequency and knowledge low resistance', 'not_experimental_evidence', 'No supporting semantic measurements. Native coordinate order has no defined spatial frequency.'],
        ['B Alpha/Beta/Gamma guaranteed cures/isomorphism', 'replace_with_tests', 'No promise of arbitrary-length prediction, hallucination cure or AGI theorem. Text code/math comparisons are domains, not proof of sensory cross-modality.'],
        ['C formation-operation-composition hypothesis', 'retain_as_competing_hypothesis', 'Compare fixed features, conditional read/write operators and heterogeneous mechanisms. None chosen by rhetoric.'],
        ['C gradient inner products and training', 'retain_with_boundary', 'Current gradients are not past training history; use actual controlled parameter updates and checkpoints. Local linearization needs a step-size check and actual update direction.'],
        ['C infinite combinations', 'correct', 'Finite rules can process many combinations, but neither model capability nor this finite study establishes exact infinite-language competence.'],
    ]
    save(BASE / 'review.json', {'timestamp': stamp(), 'attachments': [{'path': str(p), 'bytes': p.stat().st_size,
        'sha256': sha(p), 'lines': len(p.read_text(encoding='utf-8').splitlines())} for p in ATTACHMENTS],
        'source': snapshot(Path(__file__)), 'evidence': evidence, 'checked_numeric_inversions': inv,
        'prior_Hotpot_all_context_title_audit': {'questions': len(hp), 'distinct_titles': len(hp_titles), 'pairwise_overlap': 0, 'overlap_with_English_natural_titles': 0,
            'paragraph_count_histogram': dict(Counter(len(r['context_paragraphs']) for r in hp)), 'all_original_source_contexts_exactly_retained': True,
            'short_original_examples': [{'sample_id':r['sample_id'], 'paragraphs':len(r['context_paragraphs'])} for r in hp if len(r['context_paragraphs']) != 10]},
        'corrections': corrections, 'new_theorem_claimed': False,
        'references_checked': [
            {'url': 'https://arxiv.org/abs/2002.05202', 'scope': 'GLU/SwiGLU architecture, not new language law.'},
            {'url': 'https://arxiv.org/abs/1806.07572', 'scope': 'Training/function kernel connection; infinite-width conclusions not assumed for finite local checkpoints.'},
            {'url': 'https://arxiv.org/abs/2301.05217', 'scope': 'Modular-addition training mechanism case study, not a natural-language result.'},
            {'url': 'https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html', 'scope': 'Predictive-state motivation, no assumed sufficient one-vector LLM state.'}]})
    save(BASE / 'resources.json', {'timestamp': stamp(), 'kind': 'engineering_envelope_not_user_supplied_budget',
        'result_ceiling_bytes': 12*1024**3, 'disk_floor_bytes': 12*1024**3, 'host_floor_bytes': 2*1024**3,
        'compute_ceiling_seconds': 21600, 'per_process_ceiling_seconds': 7200,
        'max_simultaneous_CUDA_models': 1, 'quantization': False,
        'pilot_required': True, 'pilot_policy': 'Estimate actual capture, training, transfer and artifact costs before expansion; revise scope openly if infeasible. No infinite launch loop.',
        'checkpoint_policy': 'Original model files read-only. Controlled training uses in-memory native parameter copies/deltas; no save_pretrained into original model folders.',
        'timing': 'Recorded script elapsed time, not human analysis or GPU kernel-only time.'})
    save(BASE / 'plan.json', {'timestamp': stamp(), 'status': 'design_frozen_before_new_model_outputs',
        'main_question': 'Do source/role/history conditions predict reusable native joint gate-value computation, and does controlled language training produce the predicted parameter cooperation and composition behavior?',
        'phases': [
            {'phase': 2728, 'question': 'Common phenomena and relational/native-state atlas, with initial training-transfer evidence.',
             'tasks': ['Attachment and primary-source audit; common theory-constraint ledger.', 'Natural multi-genre English/Chinese texts and original human QA with typed source relations; group/family identities and prefix-validity controls.',
                       'Embedding/all-layer/full-coordinate streaming plus complete anchors and all-unit factors; raw/standardized visualizations.', 'Initial controlled native final-MLP parameter-update test: actual CE gradients and predicted versus measured held-out loss change; same-target/length/identity controls.']},
            {'phase': 2729, 'question': 'Candidate organization rules compete on the same inputs and target native parameter structure.',
             'tasks': ['Earlier-state versus source/history-conditioned full-coordinate kernels, matched effective capacity and shuffled-condition baselines.', 'Predict complete native gate/up joint products and writebacks, not target-gate oracle reconstruction.',
                       'Known native read/write factors and full-coordinate output sensitivity/actions connect conditional differences to parameters.', 'Multiple real training checkpoints and coherent-versus-prefix-order controls test formation/cooperation, explicitly bounded continuation not original pretraining history.']},
            {'phase': 2730, 'question': 'Frozen new combinations, genuine new sources, self-selected continuation and model-scale boundaries.',
             'tasks': ['Freeze before confirmation; distinguish new document, new relation combination and new expression claims.', 'Single-query versus prefill/decode scope controls, full QA answer and natural own-history generation with no teacher-forcing claim.',
                       'Sequential Qwen3-4B, Qwen3-14B, GLM4 native-coordinate replication; different models not coordinate-isomorphic.', 'Training-created parameter changes tested on unseen combinations and autonomous behavior; complete theory mapping and actual client/integrity verification.']},
        ],
        'initial_material_targets': {'natural_main': 768, 'natural_sources': ['GUM multi-genre English', 'SQuAD English encyclopedia', 'CMRC Chinese encyclopedia'],
            'human_QA_main': 192, 'QA_sources': ['SQuAD', 'CMRC', 'HotpotQA all10paragraphs'],
            'natural_confirmation_target': 192, 'counts_are_targets_until_eligibility_audit': True},
        'training_scope': 'Native final MLP with frozen prefix/attention/norm/readout. Cached pre-MLP state suffices for this restricted teacher-forced update because no updated final-MLP output is consumed by attention upstream. Native rollout separately requires full model. FP32 audit replica and BF16 deployment effects kept distinct.',
        'hypotheses': [
            {'id': 'fixed_feature', 'prediction': 'A full-coordinate shared early-state rule suffices; source-conditioned interaction adds no reliable capacity-matched held-out gain.'},
            {'id': 'conditional_operator', 'prediction': 'Available earlier query/source history predicts gate-value joint effects and output beyond matched-capacity fixed features, with transferable parameter-level cooperative updates.'},
            {'id': 'heterogeneous_composition', 'prediction': 'Rules specialize by measurable family/domain, and a validation-frozen mixture can predict those boundaries better than a single uniform rule.'}],
        'per_phase_delivery': ['common phenomena', 'candidate laws strengthened/weakened', 'native coordinates/units/scalar parameters', 'prospective unseen-combination predictions and actual outcomes', 'controlled training-formation evidence and its scope'],
        'negative_controls': ['No future tokens/gold answer/gold parse as online feature.', 'Document grouping includes every QA context document, not only question ID.', 'No old material called prospective unseen.', 'Full coordinates and units; no PCA/Top-K discovery.', 'No exact native recomputation called extracted prediction.', 'Actual model errors retained; task labels do not establish successful computation.'],
        'automatic_continuation': 'After integrated phases, execute an information-bearing same-goal follow-up if measurements identify it and resources allow; finish finite deliverables, do not claim AGI solved or invent resource exhaustion.',
        'retention': 'Register display/research/recompute purpose. Preserve checkpoint copies/deltas and key evidence. Delete only audited reproducible transient fields unused by both client and next analysis; report exact deletion list.'})
    print('LAW_AUDIT_AND_PLAN_FROZEN', len(corrections), 'Hotpot', len(hp), 'all_titles', len(hp_titles), flush=True)


if __name__ == '__main__':
    main()
