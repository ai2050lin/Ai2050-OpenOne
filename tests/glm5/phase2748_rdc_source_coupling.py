"""Post-selection passage-only permutation strength, all heads/coordinates.

Natural BF16 source values/probabilities are decoded and contracted in FP64.
This is not a new native GPU replay, intervention, fit, or semantic label.
"""
import argparse
from rdc_question_common import *
import rdc_question_data as data

HEAD_COLUMNS = ['raw_attention_mass', 'passage_mass_fraction', 'question_mass_fraction', 'other_mass_fraction',
    'passage_weight_deviation_L2', 'passage_conditional_effective_source_fraction',
    'passage_weighted_read_L2', 'passage_uniform_weight_read_L2', 'passage_coupling_L2',
    'passage_permutation_delta_L2', 'Cauchy_Schwarz_permutation_bound', 'full_FP64_read_L2',
    'FP64_read_minus_native_BF16_preO_L2', 'passage_mean_value_L2', 'passage_centered_values_Frobenius']
QUESTION_COLUMNS = ['native_read_mean_square', 'permuted_read_mean_square', 'native_permuted_read_MSE',
                    'native_permuted_read_relative_squared_L2', 'maximum_native_read_coordinate_change']


def contraction(probability, values, passage, question, order):
    a = np.asarray(probability, dtype=np.float64); v = np.asarray(values, dtype=np.float64)
    h, n = a.shape; kv, nv, d = v.shape
    assert n == nv and h % kv == 0 and np.isfinite(a).all() and np.isfinite(v).all() and (a >= 0).all()
    passage = np.asarray(passage, dtype=np.int64); question = np.asarray(question, dtype=np.int64)
    order = np.asarray(order, dtype=np.int64)
    assert len(passage) and len(question) and len(np.unique(passage)) == len(passage) and len(np.unique(question)) == len(question)
    assert set(passage).isdisjoint(question) and min(passage) >= 0 and max(passage) < n and min(question) >= 0 and max(question) < n
    assert order.shape == (n,) and np.array_equal(np.sort(order), np.arange(n))
    mask = np.ones(n, bool); mask[passage] = False
    assert np.array_equal(order[mask], np.arange(n)[mask]) and set(order[passage]) == set(passage)
    other = mask.copy(); other[question] = False
    repeated = np.repeat(v, h//kv, axis=0)
    av = a[:, passage]; vv = repeated[:, passage]
    mass = av.sum(1); total = a.sum(1); assert (total > 0).all()
    mean = vv.mean(1); deviation = av-mass[:, None]/len(passage)
    centered = vv-mean[:, None]
    weighted = np.einsum('hn,hnd->hd', av, vv)
    uniform = mass[:, None]*mean
    coupled = np.einsum('hn,hnd->hd', deviation, centered)
    permuted = repeated[:, order[passage]]
    delta = np.einsum('hn,hnd->hd', av, permuted-vv)
    centered_delta = np.einsum('hn,hnd->hd', deviation, permuted-vv)
    bound = np.linalg.norm(deviation, axis=1)*np.linalg.norm((permuted-vv).reshape(h, -1), axis=1)
    full = np.einsum('hn,hnd->hd', a, repeated)
    # Error tolerances scale with summation length and source magnitude. They
    # audit FP64 algebra only; native BF16 rounding is reported separately.
    scale = max(1., float(np.max(np.abs(weighted))), float(np.max(np.abs(full))), float(bound.max()))
    tolerance = 128*np.finfo(np.float64).eps*n*scale
    first_error = float(np.max(np.abs(weighted-uniform-coupled)))
    second_error = float(np.max(np.abs(delta-centered_delta)))
    assert max(first_error, second_error) <= tolerance, (first_error, second_error, tolerance)
    assert np.all(np.linalg.norm(delta, axis=1) <= bound+tolerance)
    square = np.sum(av*av, axis=1); effective_valid = square > 0
    effective = np.divide(mass*mass, square*len(passage), out=np.zeros(h), where=effective_valid)
    masses = np.stack([mass, a[:, question].sum(1), a[:, other].sum(1)], axis=1)/total[:, None]
    assert np.allclose(masses.sum(1), 1., atol=1e-14, rtol=1e-14)
    scalars = np.stack([total, *masses.T, np.linalg.norm(deviation, axis=1), effective,
        np.linalg.norm(weighted, axis=1), np.linalg.norm(uniform, axis=1), np.linalg.norm(coupled, axis=1),
        np.linalg.norm(delta, axis=1), bound, np.linalg.norm(full, axis=1), np.zeros(h),
        np.linalg.norm(mean, axis=1), np.linalg.norm(centered.reshape(h, -1), axis=1)], axis=1)
    return {'statistics': scalars, 'effective_fraction_valid': effective_valid,
            'full_read': full, 'weighted': weighted, 'uniform': uniform, 'coupling': coupled, 'delta': delta,
            'decomposition_error': first_error, 'permutation_identity_error': second_error, 'algebra_tolerance': tolerance}


def freeze():
    path = OUT/'source_coupling/execution.json'
    revision = {'source': snapshot(__file__), 'data': snapshot(Path(__file__).with_name('rdc_question_data.py')),
        'native_capture': snapshot(Path(__file__).with_name('rdc_question_native.py')),
        'material_manifest_sha256': sha(OUT/'material/manifest.json')}
    if path.exists():
        result = read(path); assert result['execution'] == revision; return result
    unit = read(OUT/'unit/source_coupling_current.json')
    assert unit['all_passed'] and unit['analysis_sha256'] == sha(__file__)
    result = {'timestamp': stamp(), 'execution': revision, 'unit_sha256': sha(OUT/'unit/source_coupling_current.json'),
        'design_status': 'New diagnostic after Q4/Q14 first-prefix fits and Q4 readout outcomes. Not original preregistration. No new fitting, model/layer/head selection or confirmation access.',
        'population': 'Every1344nonconfirmation question from336wholecontexts for each completed original model; train/validation/diagnostic and bothcohorts separately.',
        'scope': 'Block12, every native attention head, all source positions and every head-value coordinate. The fixed permutation changes passage values ONLY; original query/key/attention, question and instruction values are unchanged.',
        'identity': 'S=sum_i a_i v_i = m mean(v)+sum_i(a_i-m/n)(v_i-mean(v)); delta=sum_i(a_i-m/n)(v_perm(i)-v_i). These are finite-sum identities, not new language laws.',
        'bound': '||delta||2 <= ||a-m/n||2 ||V_perm-V||F. No claim bound is tight or proves downstream irrelevance.',
        'precision': 'FP64 real arithmetic on actual BF16-decoded weights and values. Raw probability mass may differ from1 by BF16rounding; mass fractions use actual total. Effective-source fraction normalizes conditional passage weights only. Native BF16preO and postO read perturbation retained separately; no new Oprojection or GPU execution.',
        'aggregation': 'Retain all per-question/per-head metrics and validmasks; means by split/cohort/head, and head means by question. All postOcoordinate means and change-square maps preserve native indices. No TopK, PCA, reordering or effect filtering.',
        'limits': 'A weak passage-only shuffle cannot establish source irrelevance. A strong one with similar downstream fit also cannot identify redundancy uniquely. Attention mass/norms are not semantic or causal attribution; instructions/question values carry previous context processing too. No scalar-coordinate concept assignment.',
        'no_confirmation': True}
    immutable(path, result); return result


def main(key):
    start = time.monotonic(); protocol = freeze(); folder = Path('source_coupling')/key; final = OUT/folder/'result.json'
    if final.exists():
        assert read(final)['execution_sha256'] == sha(OUT/'source_coupling/execution.json')
        print('NATURAL_SOURCE_COUPLING_ALREADY_COMPLETE', key, flush=True); return
    ids = []; refs = {}; heads = []; valid = []; qstats = []; read_values = []; shuffled_values = []; errors = []
    for split in ['train', 'validation', 'diagnostic']:
        rows, groups, questions = data.index(key, {split})
        loaded_gid = None; prefix_values = None
        for row in rows:
            gid = row['group_id']; record = questions[row['question_id']]; ref = record['field']; refs[ref['path']] = ref
            if loaded_gid != gid:
                cref = groups[gid]['context_field']; refs[cref['path']] = cref
                prefix_values = data.field(cref, ['block12_prefix_values_BF16'])['block12_prefix_values_BF16']; loaded_gid = gid
            fields = data.field(ref, ['block12_attention_BF16', 'block12_appended_values_BF16',
                'source_value_permutation', 'block12_pre_O_BF16', 'native_source_read_BF16', 'source_value_pair_shuffle_BF16'])
            values = np.concatenate([prefix_values, fields['block12_appended_values_BF16']], axis=1)
            token = row['tokens']; assert fields['block12_attention_BF16'].shape[1] == len(token['input_ids'])
            result = contraction(fields['block12_attention_BF16'], values, token['context_token_positions'],
                                 token['question_token_positions'], fields['source_value_permutation'])
            shape = result['full_read'].shape; native = fields['block12_pre_O_BF16'].reshape(shape)
            result['statistics'][:, HEAD_COLUMNS.index('FP64_read_minus_native_BF16_preO_L2')] = np.linalg.norm(result['full_read']-native, axis=1)
            heads.append(result['statistics']); valid.append(result['effective_fraction_valid'])
            r = fields['native_source_read_BF16']; s = fields['source_value_pair_shuffle_BF16']; rr = float(np.sum(r*r)); assert rr > 0
            delta = s-r
            qstats.append([float(np.mean(r*r)), float(np.mean(s*s)), float(np.mean(delta*delta)), float(np.sum(delta*delta)/rr), float(np.max(np.abs(delta)))])
            read_values.append(r); shuffled_values.append(s)
            errors.append([result['decomposition_error'], result['permutation_identity_error'], result['algebra_tolerance']])
            ids.append({k: row[k] for k in ['question_id', 'group_id', 'within_context_index', 'cohort', 'split']})
        print('NATURAL_SOURCE_COUPLING', key, split, len(rows), round(time.monotonic()-start, 1), flush=True)
    arrays = {'per_question_all_heads': np.stack(heads), 'effective_source_fraction_valid': np.stack(valid),
              'per_question_postO_statistics': np.asarray(qstats), 'FP64_identity_errors_and_tolerances': np.asarray(errors)}
    native_reads = np.stack(read_values); permuted_reads = np.stack(shuffled_values)
    assert len(ids) == 1344 and len({r['group_id'] for r in ids}) == 336
    summaries = []
    for split in ['train', 'validation', 'diagnostic']:
        for cohort in ['drop', 'quoref']:
            take = np.array([r['split'] == split and r['cohort'] == cohort for r in ids]); stem = split+'__'+cohort
            all_heads = arrays['per_question_all_heads'][take]; vv = arrays['effective_source_fraction_valid'][take]
            means = all_heads.mean(0); eff = HEAD_COLUMNS.index('passage_conditional_effective_source_fraction')
            counts = vv.sum(0); means[:, eff] = np.divide((all_heads[:, :, eff]*vv).sum(0), counts, out=np.zeros_like(counts, dtype=float), where=counts > 0)
            arrays[stem+'__head_means'] = means; arrays[stem+'__effective_source_valid_questions'] = counts
            for name, values in [('native_read_mean', native_reads[take]), ('permuted_read_mean', permuted_reads[take]),
                                 ('permuted_minus_native_read_mean_square', (permuted_reads[take]-native_reads[take])**2)]:
                arrays[stem+'__'+name] = values.mean(0)
            scalars = {c: float(means[:, i].mean()) if c != HEAD_COLUMNS[eff] or (counts > 0).all() else None for i,c in enumerate(HEAD_COLUMNS)}
            post = {c: float(arrays['per_question_postO_statistics'][take, i].mean()) for i,c in enumerate(QUESTION_COLUMNS)}
            summaries.append({'split': split, 'cohort': cohort, 'questions': int(take.sum()), 'contexts': int(take.sum())//4,
                'field_prefix': stem, 'head_equal_means': scalars, 'postO_question_means': post,
                'passage_mass_fraction_all_question_head_minmax': [float(all_heads[:, :, 1].min()), float(all_heads[:, :, 1].max())],
                'relative_postO_change_all_question_minmax': [float(arrays['per_question_postO_statistics'][take, 3].min()), float(arrays['per_question_postO_statistics'][take, 3].max())]})
    reference = commit_arrays(folder, 'all_heads_and_coordinates', arrays)
    value = {'timestamp': stamp(), 'all_passed': True, 'model': key, 'contexts': 336, 'questions': 1344,
        'execution_sha256': sha(OUT/'source_coupling/execution.json'),
        'native_result_sha256': sha(OUT/f'native/{key}/nonconfirmation/result.json'),
        'head_count': arrays['per_question_all_heads'].shape[1], 'native_width': native_reads.shape[1],
        'head_columns': HEAD_COLUMNS, 'question_columns': QUESTION_COLUMNS, 'identities': ids,
        'source_fields': list(refs.values()), 'field': reference, 'summaries': summaries,
        'maximum_FP64_decomposition_error': float(arrays['FP64_identity_errors_and_tolerances'][:, 0].max()),
        'maximum_FP64_permutation_identity_error': float(arrays['FP64_identity_errors_and_tolerances'][:, 1].max()),
        'seconds': time.monotonic()-start, 'limits': protocol['limits'], 'no_new_fitting_or_selection': True}
    immutable(final, value)
    print('NATURAL_SOURCE_COUPLING_COMPLETE', key, value['head_count'], round(value['seconds'], 1), flush=True)
    for record in summaries:
        if record['split'] == 'diagnostic':
            print('NATURAL_SOURCE_COUPLING_DIAGNOSTIC', key, record['cohort'], record['head_equal_means'], record['postO_question_means'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--model', choices=['qwen4', 'qwen14', 'glm4']); parser.add_argument('--freeze-only', action='store_true'); args = parser.parse_args()
    if args.freeze_only: freeze()
    else:
        assert args.model; main(args.model)
