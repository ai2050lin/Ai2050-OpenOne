"""Freeze a complete same-goal stage before any new native measurements."""
from collections import Counter
from rdc_construction_common import *


def freeze():
    path = BASE / 'protocol.json'
    if path.exists():
        return read(path), gzread(BASE / 'material.json.gz')
    start = time.monotonic()
    guard(5 * 1024**3)
    prior = read(OLD / 'delivery_manifest.json')
    assert [r['phase'] for r in prior['phases']] == list(range(2740, 2745))
    panel = gzread(OLD / 'identifiability/material.json.gz')
    probes = read(OLD / 'probes/protocol.json')['probes']
    from transformers import AutoTokenizer
    materials, token_audits = {}, {}
    order = sorted(range(16), key=lambda c: rank('case_split/' + str(c)))
    case_split = {c: 'train' if j < 8 else 'validation' if j < 12 else 'test' for j, c in enumerate(order)}
    for key in ['qwen4', 'qwen14', 'glm4']:
        tok = AutoTokenizer.from_pretrained(ROOT / 'models/hf' / MODELS[key], local_files_only=True, trust_remote_code=True)
        rows, audits = [], []
        for old in panel['controlled']:
            prompt = tok.apply_chat_template([{'role': 'user', 'content': old['original_text']}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
            choices = [tok(t, add_special_tokens=False)['input_ids'] for t in old['candidate_texts']]
            assert all(len(c) == 1 for c in choices), (key, old['sample_id'], choices)
            r = {**old, 'model': key, 'text': prompt, 'prompt_ids': enc['input_ids'],
                 'token_offsets': enc['offset_mapping'], 'candidate_ids': [c[0] for c in choices],
                 'target_ids': choices[0 if old['truth'] else 1], 'split': case_split[old['case']],
                 'novelty': 'Same controlled panel as2744; newly measured query-construction objects and prospective extraction splits. Not globally unexposed language or natural-corpus confirmation.'}
            if key == 'qwen4':
                assert r['prompt_ids'] == old['prompt_ids'], 'Original Q4 chat/input contract changed'
            rows.append(r)
        for i in range(0, len(rows), 2):
            a, b = rows[i:i+2]
            matched = Counter(a['prompt_ids']) == Counter(b['prompt_ids'])
            assert matched and a['pair_id'] == b['pair_id'] and a['target'] != b['target']
            audits.append({'pair_id': a['pair_id'], 'same_native_full_token_multiset': matched,
                           'prefix_tokens': len(a['prompt_ids']), 'split': a['split']})
        materials[key] = {'rows': rows, 'probes': [{**p, 'token_ids': tok(p['text'], add_special_tokens=False)['input_ids']} for p in probes]}
        token_audits[key] = audits
    assert all(len(m['rows']) == 320 for m in materials.values())
    evidence = ['delivery_manifest.json', 'analysis/phase2741.json', 'analysis/phase2742.json',
        'followup/result.json', 'identifiability/protocol.json', 'identifiability/material.json.gz',
        'identifiability/analysis/result.json', 'identifiability/analysis/pair_change_control.json',
        'formation/result.json', 'formation/material.json.gz', 'probes/protocol.json', 'rules/decoder.npz',
        'theory_snapshot.json', 'continuation_after_2744.json']
    memo = MEMO.read_bytes()
    protocol = {
        'timestamp': stamp(), 'phase': 2745, 'source': snapshot(__file__),
        'authorization': 'User explicitly requested continued goal-led mechanism research after explanation of the assistant-selected6hour/12GiB budget. Those arbitrary ceilings are not applied to this continuation.',
        'goal': 'Identify context-conditioned query construction and relational response rules, separately from query identity, word bags, generic output calibration and parameter displacement magnitude.',
        'models_serial': ['qwen4', 'qwen14', 'glm4'], 'quantization': False,
        'native_rows_per_model': 320, 'semantic_groups': 80, 'pairs_per_model': 160, 'queries_per_row': 100,
        'native_templates': 'Each tokenizer applies its own chat template to the exact same original user text; all160 native token-multiset pairs must pass independently.',
        'native_query_batches': 'Original exact-length groups, maximum16queries, no query padding; source prefill B1. No execution-shape enlargement.',
        'streaming_execution': 'Evaluate one original decoder layer on the prefix and all independent query microbatches before moving to the next layer. Reuse exact original scalar weights. Only admit after all-coordinate bit equality to standard whole-model cached calls; optimization is not an extracted mechanism.',
        'retention': 'All native prefix final-position boundaries; all query final-position coordinates at H0 through H(early+1), Hmid and Hdepth, plus postnorm. Complete early Q/projection inputs, all-source attention and early native K/V. Additional first-block Q/attention and complete all-boundary query endpoints in20predeclared fixtures. No PCA, TopK or deleted low-value axes.',
        'retention_fixture': 'Every family case0, both languages and both worlds:20rows permodel. Other unretained query layer/token interiors have frozen input/config and recomputation path.',
        'early_mid': 'early=floor(depth/3),mid=floor(2depth/3); indices are native boundaries, not cross-model equal functions.',
        'candidate_rules': ['query_only', 'uniform', 'quadratic', 'shuffled_values', 'ordered_softmax'],
        'candidate_scope': 'Known standalone-query native early Q/K/V plus actual prefix early KV and original one-block parameters. All candidate coordinates retained. Contextual queried early states are diagnostic targets, never silently predictor inputs.',
        'new_extraction_split': {'case_order': order, 'train_cases': order[:8], 'validation_cases': order[8:12], 'test_cases': order[12:]},
        'query_split': 'Reuse original60train/20validation/20unseen query IDs. Group all worlds/languages of one family/case; report paired-change errors and zero-change baseline separately from absolute state error.',
        'primary_observation': 'Full-coordinate pair-displacement trajectory and native Q/attention construction at fixed token bag and matched query; query-alone and prefix-order controls remain explicit.',
        'primary_prediction': 'Validation-selected native-coordinate ridge/operator rules must improve heldout pair-response-change prediction over zero-change and shuffled controls. Absolute response fit alone is insufficient.',
        'training_direction_control': {'learned_directions': ['natural_target_2742', 'within_cohort_permuted_target_2742', 'natural_target_2743', 'within_cohort_permuted_target_2743'],
            'target_actual_BF16_displacement_norms': [.05, .10, .18],
            'primary_radius': .10, 'matching_relative_tolerance': .001,
            'additional_controls': 'Parameter-index-permuted natural directions at .10/.18 and sign-reversed natural directions at .10; native and four original deployments retained.',
            'meaning': 'Rescale actual learned directions in memory; match measured deployed BF16 displacement by monotone search. This is a prospective direction/magnitude control, not new training or reconstruction of pretraining.',
            'evaluation': 'Same384natural calibration/test positions and320strict relation expressions, calibrated full-vocabulary scores and matched own-history behavior. Prior exposure disclosed.'},
        'resource_policy': {'arbitrary_compute_ceiling_seconds': None, 'arbitrary_result_ceiling_bytes': None,
            'minimum_actual_free_disk_bytes': 4 * 1024**3, 'maximum_simultaneous_CUDA_models': 1,
            'physical_safety_not_scientific_completion': True, 'old_resource_contract_not_modified': True},
        'memo_original_bytes': len(memo), 'memo_original_sha256': hashlib.sha256(memo).hexdigest(),
        'required_prior_artifacts': [{'path': p, 'sha256': sha(OLD / p)} for p in evidence],
        'independent_followup_plan': 'Use results to select an information-bearing next whole phase, with natural-corpus confirmation separated from exposed controlled-panel extraction. No automatic universal-closure claim.'}
    compressed(BASE / 'material.json.gz', {'models': materials, 'token_audits': token_audits, 'natural': panel['natural']})
    immutable(path, protocol)
    save(BASE / 'status.json', {'timestamp': stamp(), 'phase': 2745, 'state': 'frozen_not_yet_native_executed', 'goal_complete': False})
    ledger('phase2745_contract', time.monotonic() - start)
    return protocol, gzread(BASE / 'material.json.gz')


def freeze_extraction():
    path = BASE / 'extraction_protocol.json'
    if path.exists():
        return read(path)
    protocol, _ = freeze()
    value = {
        'timestamp': stamp(), 'source': snapshot(__file__), 'phase': 2745,
        'frozen_before_formal_native_capture_and_target_inspection': not any((BASE / 'capture').glob('*/result.json')),
        'motivation_known_before_this_stage': 'The prior five diagonal absolute-state decoders did not beat zero on relation-change prediction; observed actualH12 was later information, so localize the information gap with an actual first-query-layer input and full cross-coordinate operators.',
        'primary_targets': ['Hearly', 'postnorm'],
        'primary_input': 'Actual queriedH1 after only original block0, known at prediction time. It does not contain the later queried target. The prefix and known query string are available; one native query block is explicitly paid for.',
        'primary_operator': 'Fit a full DbyD native-coordinate linear operator on paired natural response changes: minimize sum||deltaH1 A-deltaY||^2 plus ridge. Apply the same A separately to the two H1 inputs; their prediction difference is deltaH1 A. This is statistical prediction, not injected or transported hidden-state differences.',
        'operator_inputs': ['actual_query_H1', 'available_prefix_ordered_candidate_early_output'],
        'target_pairing_controls': ['true_correspondence', 'training_pair_correspondence_shuffled_within_family_language'],
        'shuffle': 'Fixed hash-seeded permutation of training family/case pairs within each family/language; all query IDs remain aligned. No validation/test targets are used.',
        'scaling': 'Each input coordinate divided by its training-pair RMS, floor1e-8; no coordinate is removed. Through-origin paired map; a global intercept cancels in a difference.',
        'ridge_lambdas': [.001, .01, .1, 1., 10.],
        'solve': 'All-coordinate covariance/kernel ridge. An eigendecomposition may be used only as a full-rank numerical linear-system solver; every eigencomponent is retained, no PCA or spectral truncation defines the scientific representation.',
        'selection': 'For each input/control/target/model choose ridge using validation cases AND validation queries only. Test cases and unseen queries are never used to fit, select, rescale, or choose a coordinate.',
        'primary_comparison': 'On test-case/unseen-query pairs, actualH1 true-correspondence prediction must beat both zero change and its same-input shuffled-correspondence operator with positive paired group-cluster intervals. Q4 primary, other two models separate native replications, never pooled equal coordinates.',
        'secondary_baseline': 'Five all-coordinate four-column diagonal decoders, fit both absolute responses and paired responses, compare the earlier available-prefix alternatives at equal declared input availability.',
        'reporting': 'All-coordinate error maps, full DbyD operators, regularization/rank diagnostics, all heldout/family/language metrics, reconstruction tests and raw pair outputs. Family test groups are only4 each; family-specific general laws are not inferred from this panel.',
        'limitations': 'The panel and previous outputs have been exposed in earlier phases. New predictor fitting splits do not retrospectively make the material globally blind. A useful full-coordinate statistical transition is not a unique causal semantic mechanism or a complete autoregressive state.',
        'next_confirmation': 'If a relation-change rule survives these controls, freeze it before evaluating larger independent natural-source/combination materials in a separate whole phase.'}
    assert value['frozen_before_formal_native_capture_and_target_inspection']
    immutable(path, value)
    return value


if __name__ == '__main__':
    p, m = freeze()
    freeze_extraction()
    print('CONSTRUCTION_FROZEN', p['phase'], {k: len(v['rows']) for k, v in m['models'].items()}, flush=True)
