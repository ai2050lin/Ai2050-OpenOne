"""Audit new attachments and freeze the integrated ordinary-language operator campaign."""
from rdc_operator_common import *

ATTACHMENTS = [Path('C:/Users/Admin/.codex/attachments/73620099-157d-48bd-a69d-4859e20f1e8f/pasted-text.txt'),
               Path('C:/Users/Admin/.codex/attachments/eeaa2370-34b6-4ad3-8db9-92447b3f4f1c/pasted-text.txt')]


def main():
    import psutil
    if (BASE / 'plan.json').exists():
        print('OPERATOR_CONTRACT_ALREADY_FROZEN', flush=True)
        return
    memo = ROOT / 'research/glm5/docs/AGI_GLM5_MEMO.md'
    immutable(BASE / 'memo_prefix.json', {'timestamp': stamp(), 'bytes': memo.stat().st_size, 'sha256': sha(memo)})
    resources = {'timestamp': stamp(), 'engineer_selected_not_user_numeric_budget': True,
        'result_ceiling_bytes': 6 * 1024**3, 'disk_floor_bytes': 8 * 1024**3,
        'compute_ceiling_seconds': 21600, 'per_process_ceiling_seconds': 7200,
        'host_floor_bytes': 2 * 1024**3, 'maximum_concurrent_CUDA_models': 1,
        'initial_disk_free_bytes': shutil.disk_usage(ROOT).free, 'initial_host_available_bytes': psutil.virtual_memory().available,
        'pilot': 'Eight outcome-blind train paragraphs first; estimate full capture bytes/time before expanding. No unbounded all-layer archive.',
        'retention': 'Every requested native coordinate and unit processed. Full representative token-by-layer fields and all-source anchor factors retained for client/research; remaining streamed raw values have hashes, exact sources/config and full-coordinate aggregates. No deletion of prior runs.'}
    assert resources['initial_disk_free_bytes'] - resources['result_ceiling_bytes'] > resources['disk_floor_bytes']
    immutable(BASE / 'resources.json', resources)
    evidence_paths = ['verification/final.json', 'verification/model_checkpoint_fingerprints.json', 'material_audit.json',
        'extension/amplification.json', 'extension/event_forecast.json', 'extension/event_threshold.json',
        'extension/native_regimes/result.json', 'extension/native_regimes/frozen.json',
        'extension/tail_confirmation/result.json', 'generation/result.json', 'native_factors/result.json']
    evidence, missing = [], []
    for rel in evidence_paths:
        path = PRIOR / rel
        if path.exists():
            evidence.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
        else:
            missing.append(rel)
    native = read(PRIOR / 'extension/native_regimes/result.json')
    assert native['blocks'] == [6, 16, 34] and native['sources'] == 44 and native['probes'] == 90
    assert native['condition_summaries']['new_event']['tokens'] == 19
    rr = gzread(PRIOR / 'extension/native_regimes/rows.json.gz')
    event = [r for r in rr if r['stage'] == 'new' and r['role'] == 'event']
    recomputed = {}
    for block in (6, 16, 34):
        for name in ('MLP_energy', 'output_energy', 'pre_MLP_MLP_cross_term', 'sum_unit_diagonal_energies'):
            v = float(np.mean([r['blocks'][str(block)][name] for r in event]))
            assert abs(v - native['condition_summaries']['new_event']['blocks'][str(block)][name]) < 1e-8
            recomputed[f'L{block}_{name}'] = v
    # Independently recompute means from original bit-preserved fields, not just their prose summary.
    checks = []
    for row in event:
        with np.load(PRIOR / 'extension/native_regimes/fields' / (row['sample_id'] + '.npz')) as z:
            i = z['positions'].tolist().index(row['position'])
            h = unbits(z['layers'][:, i]).astype(np.float64)
            error = float(np.max(np.abs(np.mean(h*h, 1) - np.array(row['energy_by_layer']))))
            assert error < 1e-8
            checks.append({'sample_id': row['sample_id'], 'position': row['position'], 'energy_max_error': error})
    corrections = [
        ['A experimental numbers', 'Retain scoped results; attachment is not an independent model/array replication. This audit recomputes19 original energy trajectories and selected means.'],
        ['A K(x)=Wd diag(SiLU(Wg x)) Wu', 'Valid real-arithmetic gated-MLP identity without biases, not its full Jacobian, not independently a discovered semantic law. x is the actual post-attention-normalized residual.'],
        ['A quadratic coordinate expansion', 'Coefficients contain sigmoid(Wg x) and are context-dependent; not a fixed tensor or globally degree-two polynomial.'],
        ['A radius/direction and shared vector/residual', 'Valid descriptive decomposition provided the complete residual and conditioning scope remain. Not a low-rank replacement for the field.'],
        ['A Jacobian', 'Both gate and up branches required, plus actual norm derivative for residual-to-MLP derivatives.'],
        ['B continuous manifolds disproved by pulses', 'Unsupported. Large finite responses in a continuous nonlinear network do not prove discontinuity or disprove any manifold model.'],
        ['B explicit syntax absent / H12 efficiently compresses all history', 'Weak performance of selected relation/history extractors proves neither universal absence nor sufficient state.'],
        ['B sixth layer / energy rises15000fold', 'block6 is the seventh block. Absolute squared-amplitude values and growth ratios must not be interchanged.'],
        ['B events are discrete native gears', 'The energy>10 AND growth>=10 threshold is an analyst convention, not a native hard gate or a demonstrated semantic partition.'],
        ['B direction erased / deep reset', 'Late writeback is approximately opposing, residual remains nonzero, and no causal semantic erasure was measured.'],
        ['B repetition caused by missing pulse reset', 'Unrefreshed approximate trajectories repeated within32 tokens; no unique cause, infinite-time claim or demonstrated native reset mechanism.'],
        ['B eat-to-future-apple path', 'Unexecuted teaching story; a causal prefix cannot attend to the not-yet-input future token.'],
        ['B scalar radius is logit temperature', 'False around actual RMS normalization; positive residual scaling is largely normalized away, unlike scaling final logits.'],
        ['B linear predictors should be discarded', 'Contradicted by the retained all-coordinate H12 ranking result and full-vocabulary linear-fit results.'],
        ['B crossmodel ranking disproves isomorphism', 'Finite model/predictor differences are not a proof about all possible coordinate alignments.'],
        ['B language/code/math are different sensory modalities', 'They are text domains here. No sensory-modality transfer or AGI theorem follows.'],
        ['B injection will repair reasoning/hallucination', 'Unsupported intervention promise; no donor pulse injection is a core work package. Observation, conditional structure and native behavior take priority.'],
        ['Historical puzzle tables', 'Retain traceable scope from2719--2723 and earlier MEMO; no claim to rerun thousands of phases. Phase2619 target-output counts are not all answer flips.']]
    plan = {'timestamp': stamp(), 'status': 'new integrated design; model work not yet executed',
        'scientific_question': 'Which ordinary-position relations have reusable native conditional-coordinate operators, after identity/position/amplitude controls, and what do they predict about full-vocabulary competition and continuation?',
        'phases': [
            {'phase': 2724, 'title': '统一语言关系与普通状态图谱', 'work': [
                'Audit attachments against artifacts and original energy arrays; preserve correct historical puzzle boundaries.',
                'Reuse frozen UD/GUM relation library as historical discovery context; new English/Chinese natural reading passages with human answer spans, article/content-group splits and old-text overlap audit.',
                'Eight-source cost/collection pilot, then2048 natural windows (1024 per language); process embedding and every block boundary at every token, ordinary and event positions together.',
                'Preserve raw, RMS, train-z and conditional coordinate moments/cross terms; original model natural-next-token NLL and separate256 human-gold QA prompts for native-behavior eligibility.']},
            {'phase': 2725, 'title': '原生条件坐标算子提取', 'work': [
                'On identical frozen material, measure complete-coordinate input/output relations and all gate/up/activation units at three prior-supported4B blocks6/16/34.',
                'Compare constant output, globally frozen gate K, prefix-condition frozen gate K, full two-branch tangent J and native arithmetic oracle; no target-layer future state or gold answer as forecast input.',
                'Extract shared/read/write/interaction decomposition, complete-unit conditional covariance and full-coordinate signed contributions; compare identity/position/cue conditions with matched-size controls.',
                'Full matrix/Jacobian checks on declared representatives and all-coordinate derivative action checks; local MLP approximation vs whole-model next-token probability kept separate.']},
            {'phase': 2726, 'title': '组合、自回归接续与三模型边界', 'work': [
                'Freeze candidate selection on validation before confirmation; report unseen document, unseen token identity and cue-combination strata without equating cues with full semantics.',
                'Real block compilation and full-vocabulary KL/temperature controls at heldout positions; natural-prefix and own-generated-prefix branches, no hidden-state refresh in approximate branch.',
                'Native QA with original questions and controlled question/paragraph reordering; content, full answer, repeated sequences, output order and stopping reported separately.',
                'Serial nonquantized Qwen4, Qwen14 device_map=auto CPU/GPU, GLM; comparable natural coordinates and behavior-qualified/failed cases separately. Model indices not aligned by assumption.',
                'Client and scalar/array traceability, static full-coordinate figures, complete append-only MEMO and final integrity check.']}],
        'counts_before_material_pilot': {'natural_windows': 2048, 'per_language_split': {'train': 640, 'validation': 128, 'test': 128, 'confirmation': 128},
            'natural_window_token_range': [64, 192], 'native_QA': 256, 'full_layer_all_token_representatives': 16,
            'retained_anchor_positions_per_window': 2, 'crossmodel_natural_sources_per_model': 128, 'crossmodel_QA_per_model': 64,
            'autonomous_sources': 64, 'autonomous_new_tokens': 48},
        'counts_scope': 'Engineering design, not executed sample counts. Source eligibility and measured pilot may require explicit outcome-blind revision; no10000-source claim.',
        'full_coordinate_policy': 'No Top-K, PCA, neuron pruning or donor displacement defines the primary object. Streaming and exact factored matrix actions retain complete native coordinate domains.',
        'condition_scope': 'Causal token-prefix features and supplied current pre-MLP state are online. Human gold answer spans, retrospective UD and success labels are analysis labels only.',
        'continuation': 'After initial integrated delivery, automatically pursue a bounded same-goal follow-up if it adds new structure/prediction evidence and fits remaining resources; do not repeat event counts indefinitely.',
        'not_promised': ['Closed general-language/AGI theory', 'Universal pulse reset', 'Hallucination cure', 'Infinite-time stability', 'All scalar parameters causally identified']}
    save(BASE / 'review.json', {'timestamp': stamp(), 'source': snapshot(Path(__file__)),
        'attachments': [{'path': str(p), 'sha256': sha(p), 'lines': len(p.read_text(encoding='utf-8').splitlines())} for p in ATTACHMENTS],
        'evidence': evidence, 'missing_optional_paths': missing, 'recomputed19_raw_trajectories': checks,
        'recomputed_summary_means': recomputed, 'corrections': corrections,
        'literature': [{'url': 'https://arxiv.org/abs/' + aid, 'scope': scope} for aid, scope in [
            ('2402.17762', 'Prior massive-activation observations; not evidence for our particular semantic hypothesis.'),
            ('2309.17453', 'Attention sinks and streaming history; not a proof of absent syntax or complete history compression.'),
            ('2503.22329', 'Refined model-specific massive-activation analysis; no universal reset theorem.'),
            ('2002.05202', 'GLU/SwiGLU is existing architecture, not a newly invented operator identity.'),
            ('2104.09864', 'Rotary position embedding acts on architecture-specific query/key paths.')] ]})
    save(BASE / 'plan.json', plan)
    print('OPERATOR_CONTRACT_FROZEN', len(checks), 'raw trajectories;', len(corrections), 'audit decisions; missing optional', missing, flush=True)


if __name__ == '__main__':
    main()
