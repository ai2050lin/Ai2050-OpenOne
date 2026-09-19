"""Freeze a joint natural-question research stage before source/model sampling."""
from rdc_construction_common import *
from rdc_question_material import SOURCE_REFERENCES


OUT = BASE/'phase2748'
PHYSICAL = Path('C:/AI2050-RDC-Archive/rdc_query_construction_20260913/phase2748')


def main():
    prior_path = BASE/'phase2747/delivery_manifest.json'
    prior = read(prior_path)
    assert prior['phase2747_complete'] and prior['all_passed'] and len(prior['tasks']) == 7
    assert not prior['goal_complete']
    assert OUT.is_dir() and OUT.resolve() == PHYSICAL.resolve(), 'Register only the explicit new task junction'
    fields = OUT/'field_store'
    assert fields.is_dir() and fields.resolve() == (PHYSICAL/'field_store').resolve()
    guard()
    assert shutil.disk_usage(fields).free > 4*1024**3 + 512*1024**2
    protocol_path = OUT/'protocol.json'
    if protocol_path.exists():
        value = read(protocol_path)
        assert value['previous_delivery_sha256'] == sha(prior_path)
        return value
    memo = MEMO.read_bytes()
    assert '### C011：七项联合交付、完整机制拼图与下一大阶段判断'.encode('utf-8') in memo
    original = read(BASE/'protocol.json')
    assert hashlib.sha256(memo[:original['memo_original_bytes']]).hexdigest() == original['memo_original_sha256']
    immutable(OUT/'storage.json', {
        'timestamp': stamp(), 'source': snapshot(__file__),
        'logical_result_entry': str(fields), 'physical_directory': str(PHYSICAL/'field_store'),
        'logical_phase_entry': str(OUT), 'physical_phase_directory': str(PHYSICAL),
        'entry_type': 'New task-only phase directory junction; no existing files moved, overwritten or removed.',
        'initial_free_bytes_C': shutil.disk_usage(fields).free,
        'initial_free_bytes_D': shutil.disk_usage(ROOT).free,
        'retention': 'Keep actual inputs, native coordinate fields, fit coefficients, complete-answer histories and native parameter checkpoints while displayed or needed for analysis.',
        'minimum_free_bytes_each_drive': 4*1024**3,
    })
    tasks = [
        {'id': 'natural_external_graph', 'question': 'What varies across different authentic questions about the same unchanged passage?',
         'work': ['Acquire exact original Quoref/DROP data bytes and preserve annotations, including joint spans versus complete alternatives.',
                  'Audit historical article/context overlap, near duplicates, actual tokenizer boundaries and all selection exclusions.',
                  'Build stable context/question/answer/source-span IDs; corpus membership is not a mechanistic label.']},
        {'id': 'causal_native_response_atlas', 'question': 'Which complete-coordinate responses change with the actual known question?',
         'work': ['Native unquantized Q4 then Q14 then GLM, one loaded model at a time.',
                  'Compute each shared visible context prefix separately and branch real question tokens from its native cache; prohibit numerical dependence on unconsumed future suffix shape.',
                  'Retain all coordinates of every declared boundary. Audit sample/position/layer axes and full-prefix fixtures before expanding.']},
        {'id': 'conditional_selectivity_prediction', 'question': 'Can known early information predict question-dependent response changes beyond question/context prototypes?',
         'work': ['Direct full-coordinate additive and full-Cartesian-interaction ridge kernels with ordinary and within-context-selectivity objectives.',
                  'Compare question-only, context-only, vocabulary/position, native source readout and context-stratified pairing controls.',
                  'Training-only normalization; validation-only regularization/objective selection; both overall and within-context error, zero-change, full-vocabulary and source-weighted uncertainty.']},
        {'id': 'native_parameter_paths', 'question': 'How do repeatable response patterns correspond to actual native reads and writes?',
         'work': ['Trace declared natural examples through original Q/K/V, source positions, normalization and full gate/up/product/write vectors at predeclared layers.',
                  'Keep raw and normalized full-coordinate views, low-amplitude background and explicit rounding/other terms.',
                  'Treat contraction identities as accounting, not uniquely identified causal semantics; arbitrary native coordinate/unit/parameter addresses remain queryable.']},
        {'id': 'complete_answer_learning', 'question': 'Does authentic complete-answer supervision form more selective behavior than matched alternatives?',
         'work': ['Q4 block16 all original gate/up/down parameters, other original weights fixed; complete answer including native EOS as training target.',
                  'Three conditions x two paired seeds: true answer, within-context whole-answer permutation, and true-teacher-history decoded-surface-class mass.',
                  'Preserve every actual training example draw, token exposure and FP32 checkpoint with exact BF16 deployment; class supervision still sees task and teacher-history content.',
                  'Report parameter displacement, old-native/FP32-bridge baselines, held-context response specificity, full-answer content/format/EOS and natural forward histories separately.']},
        {'id': 'prospective_history_and_delivery', 'question': 'What survives genuinely reserved contexts and self-generated history?',
         'work': ['Only after fitting/training/selection is frozen, observe the reserved confirmation set.',
                  'Original models, each frozen selected predictor under its disclosed early-input permissions, and learned native deployments receive their own histories; native diagnostics are never fed back as target states.',
                  'Full-vocabulary teacher scores and free generation are distinct. Report whole-answer/EOS, nonmatching terminal text, cap censorship, branch divergence and numerical shape controls.',
                  'Update the three atlases, current client, inherited scoped puzzle/formula registry, append-only MEMO and complete data/checkpoint integrity receipts.']},
    ]
    value = {
        'timestamp': stamp(), 'source': snapshot(__file__), 'phase': 2748,
        'status': 'frozen_before_model_observation', 'previous_delivery_sha256': sha(prior_path),
        'inherited_theory_sha256': sha(BASE/'phase2747/theory_snapshot.json'),
        'goal': 'Same authorized language-encoding research: natural question-conditioned coordinate organization, native parameter relations and unseen-context/history prediction.',
        'science_boundary': 'The full-coordinate interaction kernel and weighted least squares are known tools already present historically. New value must be in natural conditional coverage, contrast-sensitive fitting and prospectively tested native behavior, not renamed algebra.',
        'tasks': tasks, 'sources': SOURCE_REFERENCES,
        'prospective_material': {
            'cohorts': ['quoref', 'drop'], 'questions_per_context': 4,
            'requested_contexts_per_cohort': {'train': 96, 'validation': 24, 'diagnostic': 48, 'confirmation': 32},
            'requested_contexts_total': 400, 'requested_questions_total': 1600,
            'count_status': 'Requested coverage, not acquired or scientifically observed counts.',
            'qualification': ['Keep original human passage/question text; one uniform disclosed task wrapper, no synthesized evidence facts or target leakage.',
                              'Choose four distinct questions with pairwise nonoverlapping accepted complete-answer signatures in each context, using stable source-ID hashes, not native model accuracy.',
                              'Primary cap: actual model prompt at most1024tokens and complete teacher answer at most64tokens for every included tokenizer; do not silently truncate.',
                              'At least full-context separation across splits and exclusion of registered old exact/normalized contexts. Quoref additionally uses original article identities; DROP article identity remains unknown unless established from actual primary metadata.',
                              'Group near duplicates conservatively before assigning splits. Incomplete historical article exclusion restricts the claim; pretraining exclusion is never claimed.'],
            'freeze_before_native_run': 'Save exact selected rows, raw source hashes, all token IDs, span offsets, assignment/eligibility counts and protocol input permissions. If requested sizes are unavailable, record the actual inventory and revise counts before any model observation; never silently replace source families.',
            'confirmation_seal': 'Metadata/length screening is allowed; no hidden-state collection, predictions scored against native target, free generation or checkpoint choice on confirmation until the separate freeze certificate is committed.',
        },
        'prediction_contract': {
            'target': 'Actual later full-coordinate responses and native full-vocabulary readout, with actual question contrasts evaluated separately from overall fit.',
            'permitted': ['Known original passage/question/prefix token IDs, positions, native input embedding and declared early H12.',
                          'Causal earlier-prefix H12 and native block12 keys/values, and deterministic native source readouts explicitly declared as input features.',
                          'Training-fitted coefficients and train-only means/scales; known own-generated history when deployed.'],
            'forbidden': ['Unseen target-layer H/MLP, future teacher tokens, correct answer labels, future-state or target-context means at prediction time.',
                          'Fitting or selecting rules from diagnostic/confirmation performance, importing a contrasting target response by delta transplant.',
                          'Calling full original-model reruns or copying native output a completed extracted mechanism.'],
            'initial_candidate_grid': {'contrast_strength': [0., 4., 16.], 'interaction': [0., 1., 3.],
                                       'effective_df_targets': [32, 64, 128, 256]},
            'selection': 'Equal-context then equal-cohort validation objective; normalized overall and within-context errors both reported. Exact target boundaries, feature variants and selection expression are registered with the selected material before native execution.',
            'all_coordinates': True, 'top_k_or_PCA': False,
        },
        'learning_contract': {
            'model': 'qwen4', 'block': 16, 'parameters': 74711040,
            'conditions': ['true_complete_answer', 'within_context_permuted_complete_answer', 'surface_class_mass_on_true_teacher_history'],
            'seeds': [2748, 2749], 'all_training_questions_consumed_once_per_run': True,
            'target_token_scope': 'Every token of the complete declared answer and native EOS, mean loss within answer then mean over examples; question/passage tokens are conditioning, not supervised output targets.',
            'precision': 'Original BF16 frozen weights; FP32 trainableMLP bridge and exact final BF16 deployment separately evaluated; no quantization.',
            'format_caveat': 'Class-mass loss sees true teacher-answer history and is not semantic removal. Whole-answer permutation changes intermediate teacher histories and length when answers differ; preserve exact exposures and frequencies.',
            'update': 'Reuse normalized full-gradient SGD and preservation checks from2747. Exact batches, steps/checkpoints, class membership and output grammar frozen with materials before training.',
        },
        'generation_contract': {'greedy': True, 'maximum_new_tokens': 128,
            'whole_response_grammar': 'Original multi-span answer as nonempty JSON array of strings; conservative normalized whole-response equality is primary, strict JSON and actual native EOS separate.',
            'semantic_scope': 'Not the official DROP/Quoref benchmark score and not a general semantic oracle. Nonmatching or capped text is not automatically a reasoning error; any post-outcome adjudication is separate and complete-set audited.',
            'no_forced_answer_or_EOS': True},
        'resources': {'one_CUDA_model_at_a_time': True, 'quantization': False,
            'bounded_pilot_before_full_capture': True, 'minimum_free_bytes_each_drive': 4*1024**3,
            'chunked_lossless_coordinate_storage': True,
            'scope': 'No user time/token cap imposed. Estimate actual elapsed, RAM, CUDA, commit and storage before expansion; resource guards protect consistency and existing user data, not a scientific success gate.'},
        'current_execution': 'Protocol/storage registration only. New algebra/parser tests, original data downloads and all language experiments still unexecuted at this timestamp.',
    }
    immutable(protocol_path, value)
    save(BASE/'status.json', {'timestamp': stamp(), 'phase2745': 'verified_complete', 'phase2746': 'verified_complete',
        'phase2747': 'verified_complete', 'phase2748': 'protocol_frozen_source_qualification_pending',
        'goal_complete': False, 'phase2747_manifest': 'phase2747/delivery_manifest.json',
        'phase2748_protocol': 'phase2748/protocol.json'})
    print('PHASE2748_PROTOCOL_FROZEN', len(tasks), flush=True)
    return value


if __name__ == '__main__':
    main()
