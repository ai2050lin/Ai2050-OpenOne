"""Predeclare remaining formation controls without selecting on new outcomes."""
from collections import Counter, defaultdict
from rdc_formation_common import *
from phase2747_rdc_material import freeze as material_freeze


def freeze():
    path = OUT / 'followup/protocol.json'
    if path.exists():
        return read(path), gzread(OUT / 'followup/material.json.gz')
    p, data = material_freeze()
    # Proportional larger gradient panel: 72per natural cohort +8per relation
    # family =256, exactly the original1728:320 training mixture.
    gradient_rows = []
    for family in sorted({r['family'] for r in data['train']}):
        pool = [r for r in data['train'] if r['family'] == family]
        if family.startswith('natural_'):
            groups = defaultdict(list)
            for r in pool:
                groups[r['source_group']].append(r)
            for rr in groups.values():
                rr.sort(key=lambda r: rank('large_gradient/'+r['sample_id']))
            order = sorted(groups, key=lambda g: rank('large_gradient/'+g))
            selected = []
            depth = 0
            while len(selected) < 72:
                for g in order:
                    if len(groups[g]) > depth:
                        selected.append(groups[g][depth])
                    if len(selected) == 72:
                        break
                depth += 1
            gradient_rows += selected
        else:
            groups = sorted({r['source_group'] for r in pool}, key=lambda g: rank('large_gradient/'+g))[:2]
            gradient_rows += [r for r in pool if r['source_group'] in groups]
    assert len(gradient_rows) == 256
    own = []
    for r in data['fresh']:
        own.append({**r, 'prompt_ids': r['ids'], 'kind': 'natural', 'max_new_tokens': 96,
                    'input_kind': 'raw corpus prefix at declared authentic nexttoken boundary'})
    for r in data['diagnostic']:
        if r['kind'] == 'controlled_relation':
            own.append({**r, 'target_token': r['target'], 'target': r['target_text'],
                        'prompt_ids': r['ids'], 'kind': 'controlled', 'max_new_tokens': 128})
    assert len(own) == 512
    # Program pair objects and response mapping were frozen before this design.
    transfer_rows = gzread(OLD / 'transfer/material.json.gz')
    probes = read(OLD / 'probes/protocol.json')['probes']
    groups = sorted({r['source_group'] for r in transfer_rows if r['split'] in ['test', 'mixed_holdout']})
    assert len(groups) == 64
    code_own = [r for r in transfer_rows if r['representation'] == 'en' and r['split'] == 'mixed_holdout']
    assert len(code_own) == 32
    # Fullprefix parameter propagation samples selected without response size,
    # correctness or learned outcomes. All cases are outside parameter training.
    differential = sorted(data['fresh'], key=lambda r: rank('parameter_tangent/'+r['sample_id']))[:24]
    for family in sorted({r['family'] for r in data['diagnostic'] if r['kind'] == 'controlled_relation'}):
        rr = [r for r in data['diagnostic'] if r['family'] == family]
        chosen = sorted({r['source_group'] for r in rr}, key=lambda g: rank('parameter_tangent/'+g))[:2]
        differential += [r for r in rr if r['source_group'] in chosen]
    assert len(differential) == 64
    value = {'gradient': gradient_rows, 'own_history': own, 'parameter_differential': differential,
             'program_transfer_groups': groups, 'program_own_targets': code_own}
    compressed(OUT / 'followup/material.json.gz', value)
    protocol = {'timestamp': stamp(), 'source': snapshot(__file__),
        'material_sha256': sha(OUT / 'followup/material.json.gz'),
        'status_at_freeze': 'Maintraining active; no formal learned outcome examined for this design.',
        'gradient_panel': {'examples': 256, 'source_groups': len({r['source_group'] for r in gradient_rows}),
            'families': dict(Counter(r['family'] for r in gradient_rows)), 'parameters': 74711040,
            'conditions': ['true_token', 'within_surface_class_permuted_token', 'surface_class_mass'],
            'scope': 'Complete original-parameter gradient on a proportional enlarged hash-fixed training subset, not fullpopulation covariance.'},
        'direction_radius_controls': {
            'learned_directions': 'All3conditions x2seeds final checkpoint, reconstruct exactFP32weights with residual.',
            'radii': '0.5 and1.0 times the actual true_token final displacement of the matching seed, separately in FP32 bridge and nativeBF16 deployment.',
            'additional_directions': 'True_token direction reversed and complete within-matrix parameter-index permutation, same two radii and two precisions.',
            'BF16_match': 'Binary search scale using actual originalBF16 parameter differences; relative radius tolerance0.001; no assumed FP32-to-BF16 equivalence.',
            'selection': 'All40new controls plus existing12actual final precision variants and2baselines; no heldout ranking selection.',
            'scoring': 'Every896heldout/validation/diagnostic position, fullvocabulary; unchanged B1 shape. Parameter equality/null and restoration checks.'},
        'calibration': {'temperatures': [.5, .75, 1., 1.25, 1.5, 2., 3.],
            'mixture_alpha': [0., .01, .05, .1, .2, .5],
            'total_concentration_beta': [.1, 1., 10., 100., 1000., 10000.],
            'priors': ['raw_training_corpus_counts', 'document_weighted_training_corpus_counts'],
            'definition': 'pi_beta(v)=(count_v+beta/V)/(sum_count+beta); pcal=(1-alpha)softmax(z/T)+alpha*pi_beta.',
            'selection': 'Source-group equal validation naturalNLL only. Report all uncalibrated versus calibrated test/freshNLL, uncertainty and controlled-pair ranking changes. No calibrated score treated as a unique semantic residual.'},
        'parameter_propagation': {'examples': 64, 'fresh_Chinese_natural': 24, 'controlled_expressions': 40,
            'directions': 'All6actual final complete parameter directions, no coordinate selection.',
            'scales': [.1, .3, 1.],
            'smooth_map': 'Original BF16 complete-prefix block16preMLP residual and normalized input promotedFP32; parameter-varying originalMLP16 then ALL original same-valuedFP32blocks17..35, finalnorm and fullvocabulary.',
            'history_control': 'Compare whole-prefix variational propagation with current-last-position-only initial perturbation. Fixed earlier-prefix tangent is a control, not the full parameter effect.',
            'precision': 'Separate nativeBF16 full-parameter finite-deployment results. Smooth reference is not the derivative of BF16rounding.',
            'information': 'Native parameter delta, current complete tokenprefix, known original weights. Gold/futuretokens excluded from state/logit prediction; labels enter scoring only.'},
        'program_transfer': {'directions': ['python_to_en', 'en_to_python', 'zh_to_en', 'en_reordered_to_en'],
            'heldout_groups': 64, 'queries': 100, 'paths': ['identity', 'query_only', 'affine', 'query_conditioned', 'shuffled_pair'],
            'mapping': 'Reuse exact old train/validation-frozen coefficients and source/target query arrays; compare complete-coordinate and fullvocabulary results, not a new cross-modal isomorphism claim.',
            'gold_free_readout': 'Uniform+8bias to all8digit tokens1..8; matched8single-token letters are alternative, no answer-dependent token choice.',
            'ownhistories': {'groups': 32, 'direction': 'python_to_en', 'cap': 256,
                'branches': ['native', 'code_identity', 'mapped_code', 'shuffled_map', 'mapped_digit_bias', 'mapped_letter_bias'],
                'timing': 'One initial readout replacement at the same old fixed Answer query; original target prefixKV stays unchanged, later steps use each branch ownhistory. Explicit extra observed source-response information, not early-only prediction.'}},
        'native_and_trained_ownhistories': {'expressions': 512, 'natural_cap': 96, 'controlled_cap': 128,
            'models': ['qwen4', 'qwen14', 'glm4'], 'trained_variants': 'All6finaloriginalBF16parameterdeployments ofQ4',
            'concurrency': 'Exactly one CUDA model at a time; original unquantized model paths; larger models may use already-qualified CPU/offload readers.',
            'comparison': 'Identical text/caps, each native tokenizer and originalchat wrapper recorded. Common-ability/failure strata based on actual complete answers, not a model-size excuse.',
            'retention': 'Every actual generated token and full postnorm; everyH boundary for a predeclared source-balanced subset. All coordinates kept on collected axes, explicit noncoverage elsewhere.',
            'no_gold': 'Never constrain decoding to the known correct answer. Naturalfirsttoken target, exactcontrolledcontent, formatting, firstdivergence, repetition and stopping separately.'},
        'full_program_remaining': 'No task marked complete merely because this specification exists; execute, analyze, visualize, audit and append actual outcomes.'}
    immutable(path, protocol)
    return protocol, value


if __name__ == '__main__':
    p, data = freeze()
    print('FORMATION2747_FOLLOWUP_FROZEN', len(data['gradient']), len(data['own_history']), len(data['parameter_differential']), flush=True)
