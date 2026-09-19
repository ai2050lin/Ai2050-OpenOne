"""Explicit full-coordinate retention axes before fitting or training outcomes."""
from rdc_question_common import *


def freeze():
    path = OUT/'retention_contract.json'
    if path.exists():
        return read(path)
    contract = effective_contract()
    codec = read(OUT/'unit/checkpoint_codec_current.json')
    assert codec['all_passed']
    value = {'timestamp': stamp(), 'source': snapshot(__file__),
        'effective_experiment_contract_sha256': sha(OUT/'effective_experiment_contract.json'),
        'before_any2748_fit_or_optimizer_update': True,
        'unchanged_native_original_fields': contract['capture'],
        'parameter_storage': {'codec': codec['codec'], 'codec_unit_sha256': sha(OUT/'unit/checkpoint_codec_current.json'),
            'format': 'FP32uint32 words XOR immutable native BF16expanded-to-FP32words; native deployedBF16uint16 words XOR immutable nativeBF16words. Deflate every word, without pruning. Exact inverse requires original checkpoint SHA and shape/name metadata.',
            'scientific_effect': 'None: reversible bit-level storage, not quantization, approximate subtraction, hidden-state transplantation or a scientific feature extraction.',
            'all_trainable_parameters_per_checkpoint': 74711040,
            'both_actual_FP32_and_deployed_BF16_words_preserved': True},
        'learned_deployment_histories': {
            'all_832_questions_per_run': 'Keep complete generated IDs/text, all-step complete-vocabulary entropy/chosen probability, positions, content/JSON/EOS/cap scores; first-prefix postnorm and H12/readout metadata; complete teacher per-token scores/postnorm separately.',
            'full_coordinate_trajectory_questions': 'All four questions in the same12predeclared full_history_context_ids:48questions/run, everystep allH0..HL plus postnorm/H12/native block12read. Full selected-layer MLP and coordinate fields at first prefix on this fixed subset.',
            'full_history_context_ids': contract['capture']['full_history_context_ids'],
            'uncollected_axes': 'No persistent every-layer/every-step field for the other784learned histories. Their exact original inputs, generated histories, full parameter checkpoints, runtime sources/settings and numerical readout statistics are retained as recomputation inputs.',
            'selection': 'Fixed source IDs from the pre-native-observation contract, identical across all6learning conditions/seeds, not chosen by answer correctness or effect size.'},
        'extracted_predictor_histories': {
            'all_640_questions_per_rule_model': 'Keep every generated ID/text, complete-vocabulary entropy/chosen probability, positions, all casts/numerical status and complete content/JSON/EOS/cap scores.',
            'raw_full_coordinate_rows': 'All four questions in the predeclared full_history_context_ids for diagnostic/confirmation:32questions per rule/model. Keep every generated-step actual permitted H12, source read, unrounded predicted postnorm and deployedBF16postnorm, all their coordinates.',
            'uncollected_axes': 'For the remaining608questions, intermediate predictor/internal vectors are recomputable from original native early-layer weights, saved exact selected coefficients and complete own-history IDs, but are not all persisted. No original later native state is supplied to the predictor.'},
        'fit_storage': {
            'all_candidates': 'Keep every actual grid result, training/validation identity, means/scales, target/source boundaries and fitting recipe; no selection on diagnostic/confirmation.',
            'selected_variants_and_alpha0_controls': 'Retain complete selected768x768kernel solution operators, normalization/centering metadata and exact training identities. Any secondary-target full coefficient matrix can be rebuilt by applying its frozen operator to retained native training targets. This is not spectral truncation.',
            'two_deployed_rules_per_model': 'Persist exact full-coordinate postnorm coefficients and full necessary training-feature/centering state for causal own-history execution.',
            'native_target_retention': 'All original native target coordinates/units remain in the original acquisition fields; source data are not discarded to save coefficient duplicates.'},
        'recomputation_scope': 'Kept precise recipes and inputs do not mean an unexecuted recomputation has passed a bit-level replay. Stored fixtures establish the stated qualification; additional query-time recomputation must verify runtime compatibility and disclose numeric differences.',
        'no_original_model_or_previous_field_removed': True,
        'status': 'Retention rules only; no fit/training/deployment claimed executed.'}
    immutable(path, value)
    print('NATURAL_RETENTION_AXES_FROZEN', flush=True)
    return value


if __name__ == '__main__':
    freeze()
