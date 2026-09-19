"""Exact capture, prediction and learning boundaries before any model observation."""
from rdc_construction_common import *
from phase2748_rdc_context_inventory import ranked

OUT = BASE/'phase2748'


def freeze():
    path = OUT/'experiment_contract.json'
    if path.exists():
        value = read(path)
        assert value['material_manifest_sha256'] == sha(OUT/'material/manifest.json')
        return value
    audit = read(OUT/'material/audit_current.json')
    assert audit['all_passed'] and audit['material_manifest_sha256'] == sha(OUT/'material/manifest.json')
    material = read(OUT/'material/manifest.json')
    groups = gzread(ROOT/material['context_groups']['path'])
    ordered = sorted(groups, key=lambda g: ranked('pilot/'+g['group_id']))
    pilot = [next(g['group_id'] for g in ordered if g['cohort'] == c and g['split'] == 'train')
             for c in ['quoref', 'drop']]
    full_history = [g['group_id'] for c in ['quoref', 'drop'] for split in ['validation', 'diagnostic', 'confirmation']
                    for g in sorted((g for g in groups if g['cohort'] == c and g['split'] == split),
                                    key=lambda g: ranked('allH_history/'+g['group_id']))[:2]]
    value = {'timestamp': stamp(), 'source': snapshot(__file__),
        'status': 'frozen_before_any_Phase2748_native_model_observation',
        'protocol_sha256': sha(OUT/'protocol.json'), 'material_manifest_sha256': sha(OUT/'material/manifest.json'),
        'material_audit_sha256': sha(OUT/'material/audit_current.json'),
        'models': ['qwen4', 'qwen14', 'glm4'],
        'capture': {
            'formal_first_prefix': 'Native B1 common context prefix alone, then each of four actual question suffixes from an independent cloned causal cache; original position/mask/native RoPE and all weights unchanged.',
            'scheduling': 'Layer-major independent B1 requests, initially two contexts/eight questions per wave; temporary original-layer weight caching only. Any safe width adjustment changes scheduling, never per-request B1 arithmetic shape.',
            'initial_prefix_all_H': 'Every H0..HL residual boundary at final consumed prompt token, plus final postnorm separately, all native coordinates.',
            'context_fields': 'H12 at every shared prefix token; mean H0 and H12 over actual passage-token positions; complete block12 causal prefix keys/values; source token IDs/positions.',
            'question_fields': 'H0 and H12 means over actual human-question token positions; H12 final prompt coordinate; all native block12 source attention probabilities, final pre-RoPE Q and appended native K/V; complete selected-layer gate/up/product and attention-input/write/preMLP/MLP-input/write.',
            'selected_MLP_blocks': {'qwen4': [6, 12, 24, 35], 'qwen14': [6, 12, 24, 39], 'glm4': [6, 12, 24, 39]},
            'full_source_H_context_ids': pilot,
            'full_source_H_scope': 'For these two training contexts per model only, retain every shared prefix token at every H0..HL boundary; remaining contexts have fullH at final question and H12 at every source position, not every position at every layer.',
            'value_pair_control': 'Keep actual block12 attention probabilities and question/format-source values fixed; permute every passage-source value position with one stable per-model/context permutation shared by all four questions. Evaluate using original O projection; original model computation is never replaced.',
            'native_accounting': 'Replay complete native attention-probability times V then O with the same B1/query-length shape; verify true replay versus actual native attention write. Gate/up/product identities checked against actual down-projection input. Accounting is not semantic identification.',
            'pilot_context_ids': pilot, 'pilot_questions': 8,
            'admission': ['Compare new layer-major engine with installed native complete-B1 forward on all eight prompts at every H and postnorm; exact equality required before formal data.',
                          'Repeat common-prefix branch execution in reversed independent request order; every saved field, readout, teacher first-token statistic and source cache identity must agree.',
                          'Compare formal segmented versus whole-B1 allH/postnorm/full-vocabulary quantitatively; equality is not assumed and this difference is never attributed to semantics.',
                          'Measure actual host/CUDA/commit/storage/time before expanding. Confirmation data never used for qualification.'],
            'native_first_field_questions': 1600,
            'native_free_generation_splits': ['validation', 'diagnostic', 'confirmation'],
            'native_free_histories_per_model': 832,
            'own_history_fields': 'All steps: emitted IDs, postnorm, final H12, native block12 attention write, complete-vocabulary entropy/chosen probability and exact prefix position. FullH all layers for all four questions in predeclared full_history_context_ids only.',
            'full_history_context_ids': full_history,
            'uncollected_axes': 'No full all-layer/all-position tensor for every context or every generation step. No all-layer attention or MLP-unit trajectories at every generation step. All coordinates of each declared saved boundary are preserved.',
        },
        'prediction': {
            'training_samples': '768 first-question-prefix observations, one row per real question; groups are192training contexts. No history-state fit is added after diagnostic outcomes.',
            'targets': ['postnorm', 'H24', 'H32', 'H_last', 'MLP24_product'],
            'primary_target': 'postnorm',
            'query_definition': 'Native H12 at the final consumed prompt token, known before all prediction targets. This contains context; it is not called a context-free question representation.',
            'context_definition': 'Mean full-coordinate native H12 over frozen passage-token positions in the separately consumed common prefix; no question/target-group mean.',
            'variants': [
                {'id': 'early_query_context', 'q': 'H12_last', 'c': 'context_H12_mean'},
                {'id': 'early_query_only', 'q': 'H12_last', 'c': 'zero_scalar'},
                {'id': 'context_only', 'q': 'zero_scalar', 'c': 'context_H12_mean'},
                {'id': 'question_embedding_only', 'q': 'question_H0_mean', 'c': 'zero_scalar'},
                {'id': 'embedding_question_context', 'q': 'question_H0_mean', 'c': 'context_H0_mean'},
                {'id': 'native_source_read', 'q': 'concat(H12_last,native_block12_attention_write)', 'c': 'context_H12_mean'},
                {'id': 'source_value_pair_shuffle', 'q': 'concat(H12_last,value_permuted_block12_attention_write)', 'c': 'context_H12_mean'},
                {'id': 'within_context_target_pair_shuffle', 'q': 'H12_last', 'c': 'context_H12_mean',
                 'control': 'Only training target assignment cyclically shifts question index by one within each context. Actual target evaluations are never shuffled.'},
                {'id': 'lexical_position', 'q': 'All-vocabulary normalized question token counts plus known normalized length/position features',
                 'c': 'All-vocabulary normalized passage token counts',
                 'normalization': 'Counts divided by corresponding token count, vocabulary width fixed by native tokenizer; all coordinates, sparse exact arithmetic, train-only centering. Continuous auxiliary fields standardized from training only.'},
            ],
            'grid': {'contrast_strength': [0., 4., 16.], 'interaction': [0., 1., 3.], 'effective_df_target': [32, 64, 128, 256]},
            'regularization': 'For each full eigenspectrum and DF target solve sum(e/(e+lambda))=target by monotone positive-lambda bisection. If maximum attainable DF is smaller, report saturation and smallest positive registered numerical floor; do not truncate eigenvectors.',
            'selection': 'For each variant, minimize validation mean over cohorts of 0.5*(absolute_MSE/train_total_variance + within_context_MSE/train_within_variance), both denominators from unshuffled native training postnorm with positive1e-12floor; equal question counts imply equal context weighting. Tie-break strength, interaction, DF lexicographically. Secondary targets use that variant\'s selected postnorm hyperparameters, not separate favorable selection.',
            'matched_baseline': 'For each variant also report its best validation-selected alpha0 model using the same DF/rho grid.',
            'uncertainty': 'Whole-context paired bootstrap,2000resamples within each cohort; report per-cohort and equal-cohort means, absolute errors, within-context errors, zero-change, full-vocabulary readout KL/top1 and whole-answer outcomes separately.',
            'prospective_primary_rule': 'On each model, select among early_query_context and native_source_read by the same primary validation objective; independent control is the selected within_context_target_pair_shuffle rule. Freeze both before confirmation.',
            'self_history': 'The two frozen first-prefix rules predict each new output from their own known generated history and native layers0..12 only. Fits have first-prefix training support; later-history use is explicitly an out-of-support stress test, not silently claimed within-distribution prediction. No native later target is fed back.',
            'self_history_splits': ['diagnostic', 'confirmation'], 'self_histories_per_model_per_rule': 640,
            'readout': 'Cast predicted postnorm to original BF16, call original complete-vocabulary unembedding and greedy decoder; record casting effects separately from FP64/FP32 fitting error.',
            'no_future_target_input': True, 'no_delta_transplant': True, 'no_PCA_or_TopK': True,
        },
        'learning': {
            'model': 'qwen4', 'block': 16, 'all_MLP_parameters': 74711040,
            'conditions': ['true_complete_answer', 'within_context_permuted_complete_answer', 'surface_class_mass_on_true_teacher_history'],
            'seeds': [2748, 2749], 'epochs': 1, 'questions_per_run': 768, 'batch_size': 8, 'steps': 96,
            'batches': 'Seeded complete permutation of the same768questions, contiguous8-example groups; same example order for all three paired conditions of a seed. Per-example teacher histories/lengths recorded; no dropped draws.',
            'permutation': 'Within each training context, answer teacher text at question index(i+1)mod4; fixed across seeds and complete answer plus EOS moved together.',
            'objective': 'Average over every complete teacher-answer token including native EOS, then average eight examples; no passage/question token is a supervised output target.',
            'optimizer': 'Full-gradient normalized SGD with relative step0.02: theta <- theta -0.02*norm(theta0)*gradient/(norm(gradient)+1e-12), no momentum or weight decay. FP32 bridge, original other BF16 weights fixed.',
            'checkpoints': [1, 8, 32, 96],
            'checkpoint_retention': 'Final96stepFP32 parameters and exactBF16 deployment persist. At1/8/32save losses, full-parameter norms/update summaries and complete validation teacher scores; no intermediate behavioral/checkpoint selection. Seeded draws and exact source config permit recomputation.',
            'deployment_choice': 'Final96stepcheckpoint fixed before training, not chosen from diagnostic or confirmation scores.',
            'surface_classes': 'Use the existing audited decoded-token surface partition for Q4 with its exact class-ID artifact and SHA registered before first update; contains true teacher-history content, not semantic erasure.',
            'all_other_parameters': 'Original native parameters remain bit-identical; training bridge and BF16deployment separately audited against unchanged original model and local native checkpoints.',
            'final_free_generation_splits': ['validation', 'diagnostic', 'confirmation'],
            'final_free_histories_per_condition_seed': 832,
        },
        'scoring': {'greedy_maximum_new_tokens': 128, 'whole_answer_rule': 'Frozen conservative_complete_answer; native stops terminate, no forced output. Complete accepted alternatives versus joint spans preserved.',
            'teacher_scores': 'Full-vocabulary likelihood of separately tokenized complete JSON answer plus native EOS, conditioning on its preceding given tokens; separate from actual free generation and never substituted for it.',
            'uncertain_nonmatches': 'Report literal outputs, format, EOS and cap independently. Any later human/model adjudication is a separate, complete declared subset; primary scores immutable.'},
        'confirmation_gate': 'All three native pilots and nonconfirmation acquisition, all first-prefix fits/primary selections and six final trained parameter checkpoints must be saved and SHA-frozen before any confirmation native target, scoring or free generation.',
        'resource_policy': 'New data only inside registered phase directory/field_store on C; code and small source snapshots on D. Minimum4GiB free each drive. Per-wave cost and full-scope pilot forecast, sequential CUDA models, lossless coordinate storage; no user/system-process or model-file mutation.',
        'research_boundary': 'Known kernel/regression, native contraction and gradient tools. Mechanism advances require actual held-context/condition evidence; no new universal law or AGI completion inferred from algebra or corpus membership.'}
    immutable(path, value)
    print('NATURAL_EXPERIMENT_BOUNDARIES_FROZEN', len(value['prediction']['variants']), flush=True)
    return value


def effective_contract():
    """Retain the first registration and a pre-observation arithmetic correction."""
    import copy
    original = freeze()
    path = OUT/'experiment_amendments/0001_absolute_gradient_step.json'
    if not path.exists():
        immutable(path, {'timestamp': stamp(), 'source': snapshot(__file__),
            'original_contract_sha256': sha(OUT/'experiment_contract.json'),
            'before_any_native_observation_or_parameter_update': True,
            'reason': 'Direct inspection of Phase2747 training shows the normalized gradient step had absolute FP32 L2 length0.02, not0.02times the original parameter norm. The first2748registration mistakenly introduced relative scaling while intending reuse. Preserve it and explicitly correct before any model execution.',
            'previous_training_source': snapshot(Path(__file__).with_name('phase2747_rdc_training.py')),
            'changed_field': 'learning.optimizer',
            'old_value': original['learning']['optimizer'],
            'new_value': 'Full-gradient normalized SGD with absolute FP32 L2 step0.02: theta <- theta -0.02*gradient/(norm(gradient)+1e-12), no momentum or weight decay. FP32 bridge, original other BF16 weights fixed.',
            'unchanged': 'All samples, targets, groups, grid, loss averaging,96steps,6runs, selected checkpoints and prospective split policy.'})
    amendment = read(path)
    value = copy.deepcopy(original)
    value['learning']['optimizer'] = amendment['new_value']
    value['base_contract_sha256'] = sha(OUT/'experiment_contract.json')
    value['preobservation_amendment_sha256'] = sha(path)
    immutable(OUT/'effective_experiment_contract.json', value)
    return value


if __name__ == '__main__':
    effective_contract()
