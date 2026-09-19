"""Whole-panel matched-direction analysis; no new labels or outcome-selected radius."""
from collections import defaultdict
from rdc_construction_common import *
from rdc_construction_direction import variants, radius_name
from phase2744_rdc_query_identifiability import TEMPERATURES, MIXTURES

OUT = BASE / 'norm_controls'
FIELDS = ['H16', 'H17', 'H36', 'postnorm'] + [f'L{l}_{k}' for l in [16, 35] for k in ['gate_proj', 'up_proj', 'activation']]


def records(variant, stage, rows):
    if variant['kind'] != 'reuse':
        value = gzread(OUT / variant['name'] / (stage+'.json.gz'))
    elif stage == 'natural':
        value = gzread(OLD / 'identifiability/calibration' / variant['old_variant'] / 'observations.json.gz')
    else:
        folder = OLD / 'identifiability' / stage / variant['old_variant'] / 'commits'
        value = [read(folder / (r['sample_id']+'.json')) for r in rows]
        if stage == 'relations':
            value = [r | {'conditional_yes_probability': r['actual_current']['binary_conditional_yes_probability'],
                'conditional_answer_NLL': r['actual_current']['binary_conditional_NLL'],
                'full_vocabulary_NLL': r['actual_current']['full_vocab_NLL'],
                'argmax_correct': r['actual_current']['first_argmax_correct']} for r in value]
        if stage == 'behavior':
            value = [r | {'B1_first_token': r['initial_shape_control']['B1_initial_token_id'],
                'B8_first_token': r['initial_shape_control']['B8_initial_token_id']} for r in value]
    assert [r['sample_id'] for r in value] == [r['sample_id'] for r in rows]
    return value


def relation_arrays(variant, rows):
    if variant['kind'] != 'reuse':
        with np.load(OUT / variant['name'] / 'relation_fields.npz') as z:
            return {k: unbits(z[k]).astype(float) for k in FIELDS}
    value = defaultdict(list)
    for row in rows:
        with np.load(OLD / 'identifiability/relations' / variant['old_variant'] / 'fields' / (row['sample_id']+'.npz')) as z:
            for k in FIELDS:
                if k.startswith('H'):
                    a = z['prefix_layers'][int(k[1:])] if 'prefix_layers' in z.files else z['prefix_selected_layers_H16_H17_H36'][[16, 17, 36].index(int(k[1:]))]
                else:
                    a = z['postnorm_original_prompt'] if k == 'postnorm' else z[k]
                value[k].append(unbits(a).astype(float))
    return {k: np.stack(a) for k, a in value.items()}


def calibrate(material, controls):
    rows = material['natural']
    ci = [i for i, r in enumerate(rows) if r['split'] == 'calibration']
    ti = [i for i, r in enumerate(rows) if r['split'] == 'prospective_natural']
    cgroups = sorted({rows[i]['source_group'] for i in ci})
    assert not set(cgroups) & {rows[i]['source_group'] for i in ti}
    selections, losses, obs = {}, {}, {}
    for v in controls:
        name = v['name']
        path = OLD / 'identifiability/calibration' / v['old_variant'] / 'all_fields.npz' if v['kind'] == 'reuse' else OUT / name / 'natural_fields.npz'
        with np.load(path) as z:
            grid = z['temperature_prior_mixture_NLL'].copy()
        assert grid.shape == (384, len(TEMPERATURES), len(MIXTURES)) and np.isfinite(grid).all()
        obs[name] = records(v, 'natural', rows)
        raw = np.array([r.get('raw_NLL', r.get('native_NLL')) for r in obs[name]])
        assert np.array_equal(raw, grid[:, 2, 0])
        objective = np.mean([grid[[i for i in ci if rows[i]['source_group'] == group]].mean(0) for group in cgroups], 0)
        ix = np.unravel_index(objective.argmin(), objective.shape)
        it, ia = int(objective[:, 0].argmin()), int(objective[2].argmin())
        selections[name] = {'joint_temperature': TEMPERATURES[ix[0]], 'joint_prior_mixture': MIXTURES[ix[1]],
            'joint_index': list(map(int, ix)), 'temperature_only': TEMPERATURES[it], 'prior_only_alpha': MIXTURES[ia],
            'calibration_documents': len(cgroups), 'selection_objective': 'Equal document mean calibration NLL, no test selection.'}
        losses[name] = {'raw': raw, 'joint_calibrated': grid[:, ix[0], ix[1]], 'temperature_only': grid[:, it, 0], 'prior_only': grid[:, 2, ia]}
    # Persist selections before any test comparisons are calculated.
    save(OUT / 'calibration_selection.json', selections)
    reports = []
    for cohort in ['all']+sorted({r['cohort'] for r in rows}):
        ix = [i for i in ti if cohort == 'all' or rows[i]['cohort'] == cohort]
        gs = [rows[i]['source_group'] for i in ix]
        for v in controls:
            n = v['name']
            reports.append({'variant': n, 'cohort': cohort, 'positions': len(ix), 'documents': len(set(gs)),
                'raw_NLL': clustered(losses[n]['raw'][ix], gs), 'joint_calibrated_NLL': clustered(losses[n]['joint_calibrated'][ix], gs),
                'raw_minus_native_raw': clustered((losses[n]['raw']-losses['native']['raw'])[ix], gs),
                'joint_minus_native_joint': clustered((losses[n]['joint_calibrated']-losses['native']['joint_calibrated'])[ix], gs),
                'raw_entropy': clustered([obs[n][i]['entropy'] for i in ix], gs),
                'raw_argmax_accuracy': clustered([int(obs[n][i]['argmax_correct']) for i in ix], gs)})
    npz(OUT / 'calibrated_losses.npz', **{v+'__'+k: a for v, entry in losses.items() for k, a in entry.items()})
    return {'selections': selections, 'prospective_reports': reports,
        'scope': '96 test documents were exposed in2744; excluded from new calibrator selection, not globally blind. Calibration changes scoring, not reported generation.'}, losses, ti


def panel(material, controls):
    rows = material['models']['qwen4']['rows']
    families = ['all']+sorted({r['family'] for r in rows})
    groups = [r['source_group'] for r in rows]
    indices = defaultdict(list)
    for i, r in enumerate(rows):
        indices[r['pair_id']].append(i)
    assert len(indices) == 160 and all(len(ix) == 2 for ix in indices.values())
    native = relation_arrays(controls[0], rows)
    base_behavior = records(controls[0], 'behavior', rows)
    reports, pair_reports, coordinate_reports, examples, outcomes = [], [], [], [], {}
    pair_details = []
    for v in controls:
        name = v['name']
        rr, bb = records(v, 'relations', rows), records(v, 'behavior', rows)
        arr = native if name == 'native' else relation_arrays(v, rows)
        assert all(np.isfinite(a).all() for a in arr.values())
        assert np.array_equal(arr['H16'], native['H16']), ('Update leaked before its block', name)
        coordinate = {}
        pairs = []
        for pair_id, ix in indices.items():
            a, b = sorted(ix, key=lambda i: rows[i]['world'])
            assert rows[a]['truth'] != rows[b]['truth'] and rows[a]['world'] == 0
            pairs.append({'variant': name, 'pair_id': pair_id, 'source_group': rows[a]['source_group'],
                'family': rows[a]['family'], 'language': rows[a]['language'],
                'answer_aligned_separation': (rr[a]['conditional_yes_probability']-rr[b]['conditional_yes_probability'])*(2*int(rows[a]['truth'])-1),
                'both_B1_first_answers_correct': bool(rr[a]['argmax_correct'] and rr[b]['argmax_correct']),
                'both_B8_complete_answers_correct': all(bb[i]['answer_scoring']['parsed_and_stopped_correct'] for i in ix)})
        pair_details.extend(pairs)
        outcomes[name] = {'conditional_NLL': np.array([r['conditional_answer_NLL'] for r in rr]),
            'full_vocab_NLL': np.array([r['full_vocabulary_NLL'] for r in rr]),
            'complete_success': np.array([int(r['answer_scoring']['parsed_and_stopped_correct']) for r in bb], dtype=float),
            'pair_separation': np.array([r['answer_aligned_separation'] for r in pairs])}
        for family in families:
            ix = [i for i, r in enumerate(rows) if family == 'all' or r['family'] == family]
            gs = [groups[i] for i in ix]
            scoring = [bb[i]['answer_scoring'] for i in ix]
            reports.append({'variant': name, 'family': family, 'expressions': len(ix), 'groups': len(set(gs)),
                'correct_and_stopped': sum(r['parsed_and_stopped_correct'] for r in scoring),
                'strict_answer_only': sum(r['strict_answer_only'] for r in scoring),
                'parsed_wrong': sum(r['conservative_final_answer'] is not None and not r['conservative_final_correct'] for r in scoring),
                'unparsed_EOS': sum(r['EOS'] and r['conservative_final_answer'] is None for r in scoring),
                'censored': sum(r['censored'] for r in scoring),
                'B1_argmax_correct': sum(rr[i]['argmax_correct'] for i in ix),
                'B1_B8_first_disagreements': sum(bb[i]['B1_first_token'] != bb[i]['B8_first_token'] for i in ix),
                'changed_sequences_vs_native': sum(bb[i]['generated_ids'] != base_behavior[i]['generated_ids'] for i in ix),
                'first_token_divergences': sum(bb[i]['first_divergence_from_native_B8'] == 0 for i in ix),
                'later_divergences': sum(bb[i]['first_divergence_from_native_B8'] is not None and bb[i]['first_divergence_from_native_B8'] > 0 for i in ix),
                'mean_tokens': float(np.mean([len(bb[i]['generated_ids']) for i in ix])),
                'success_change_vs_native': clustered([int(bb[i]['answer_scoring']['parsed_and_stopped_correct'])-int(base_behavior[i]['answer_scoring']['parsed_and_stopped_correct']) for i in ix], gs),
                'full_vocabulary_NLL': clustered(outcomes[name]['full_vocab_NLL'][ix], gs),
                'conditional_answer_NLL': clustered(outcomes[name]['conditional_NLL'][ix], gs)})
            pr = [p for p in pairs if family == 'all' or p['family'] == family]
            pair_reports.append({'variant': name, 'family': family, 'pairs': len(pr),
                'both_B1_first_answers_correct': sum(r['both_B1_first_answers_correct'] for r in pr),
                'both_B8_complete_answers_correct': sum(r['both_B8_complete_answers_correct'] for r in pr),
                'answer_aligned_separation': clustered([r['answer_aligned_separation'] for r in pr], [r['source_group'] for r in pr])})
            for field in FIELDS:
                delta = arr[field][ix]-native[field][ix]
                coordinate[family+'__'+field+'__mean_delta'] = delta.mean(0)
                coordinate[family+'__'+field+'__MSE'] = (delta**2).mean(0)
                coordinate_reports.append({'variant': name, 'family': family, 'field': field,
                    'all_axes': delta.shape[-1], 'mean_squared_change': float(np.mean(delta**2)),
                    'changed_scalars': int(np.count_nonzero(delta)), 'total_scalars': delta.size})
        npz(OUT / 'coordinate_changes' / (name+'.npz'), **coordinate)
        examples.append({'variant': name, 'selection': 'Predeclared case0 English, both worlds, every family.',
            'rows': [{'material': r, 'B1_relation': rr[i], 'B8_behavior': bb[i]} for i, r in enumerate(rows) if r['case'] == 0 and r['language'] == 'en']})
        print('NORM_ANALYSIS_PANEL', name, flush=True)
        if name != 'native':
            del arr
    compressed(OUT / 'paired_observations.json.gz', pair_details)
    compressed(OUT / 'predeclared_examples.json.gz', examples)
    npz(OUT / 'panel_outcomes.npz', **{v+'__'+k: a for v, entry in outcomes.items() for k, a in entry.items()})
    return {'behavior': reports, 'relations': pair_reports, 'coordinate_summary': coordinate_reports,
        'H16_bit_identical_for_all23conditions': True, 'field_order': FIELDS}, outcomes, [rows[ix[0]]['source_group'] for ix in indices.values()]


def comparisons(material, losses, test_ix, outcomes, pair_groups):
    natural_groups = [material['natural'][i]['source_group'] for i in test_ix]
    groups = [r['source_group'] for r in material['models']['qwen4']['rows']]
    reports = []
    for radius in [.05, .10, .18]:
        for kind in ['permuted', 'coordinate_shuffle', 'reverse']:
            if (kind == 'coordinate_shuffle' and radius == .05) or (kind == 'reverse' and radius != .10):
                continue
            differences = defaultdict(list)
            for seed in [2742, 2743]:
                a = f'matched_natural_target_{seed}_r'+radius_name(radius)
                b = (f'matched_within_cohort_permuted_target_{seed}_r' if kind == 'permuted' else f'{kind}_{seed}_r')+radius_name(radius)
                for metric in ['raw', 'joint_calibrated']:
                    differences['natural_'+metric].append((losses[a][metric]-losses[b][metric])[test_ix])
                for metric in outcomes[a]:
                    differences[metric].append(outcomes[a][metric]-outcomes[b][metric])
                reports.append({'radius': radius, 'control': kind, 'seed': seed, 'primary_radius': radius == .10,
                    'natural_minus_control': {k: clustered(v[-1], natural_groups if k.startswith('natural_') else pair_groups if k == 'pair_separation' else groups) for k, v in differences.items()}})
            reports.append({'radius': radius, 'control': kind, 'seed': 'paired_mean_of_two_fixed_seeds', 'primary_radius': radius == .10,
                'natural_minus_control': {k: clustered(np.mean(v, 0), natural_groups if k.startswith('natural_') else pair_groups if k == 'pair_separation' else groups) for k, v in differences.items()},
                'scope': 'Cluster uncertainty is over documents/semantic groups; two seeds do not establish a training-seed population interval.'})
    return reports


def main():
    if (OUT / 'analysis.json').exists():
        assert read(OUT / 'analysis.json')['all_passed']
        return
    start = time.monotonic()
    assert read(OUT / 'result.json')['all_passed']
    material, controls = gzread(BASE / 'material.json.gz'), variants()
    version = snapshot(__file__)
    cal, losses, ti = calibrate(material, controls)
    panels, outcomes, pair_groups = panel(material, controls)
    result = {'timestamp': stamp(), 'source': version, 'all_passed': True, 'phase': 2745,
        'variants': len(controls), 'calibration': cal, **panels,
        'direction_comparisons': comparisons(material, losses, ti, outcomes, pair_groups),
        'matching': [read(OUT / v['name'] / 'parameter_matching.json') for v in controls if v['kind'] != 'reuse'],
        'scope': 'Actual BF16 global displacement matched; matrix norms and changed-coordinate supports are measured, not equated. Reuses four learned directions, adds no new gradient-training trajectories. All coordinates and units retained; observed changes are not uniquely semantic or causal mediation.',
        'seconds': time.monotonic()-start}
    save(OUT / 'analysis.json', result)
    ledger('construction_norm_analysis', result['seconds'])
    print('NORM_ANALYSIS_DONE', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
