"""Frozen nine-route full-coordinate fits, validation selection then diagnostics."""
import argparse
from rdc_question_common import *
from rdc_question_fit import *
import rdc_question_data as data


def freeze():
    path = OUT/'fit/implementation_contract.json'
    rev = {'fitter': snapshot(__file__), 'solver': snapshot(Path(__file__).with_name('rdc_question_fit.py')),
        'data': snapshot(Path(__file__).with_name('rdc_question_data.py')),
        'effective_contract_sha256': sha(OUT/'effective_experiment_contract.json'),
        'retention_contract_sha256': sha(OUT/'retention_contract.json')}
    if path.exists():
        old = read(path)
        assert old['execution'] == rev, 'Frozen fitter changed; versioned repair required'
        return old
    unit = read(OUT/'unit/fit_solver_current.json')
    assert unit['all_passed'] and unit['solver']['sha256'] == rev['solver']['sha256']
    result = {'timestamp': stamp(), 'execution': rev, 'unit_sha256': sha(OUT/'unit/fit_solver_current.json'),
        'before_first_native_response_fit': True,
        'arithmetic': 'CPU float64, original target coordinate order, exact sample-space kernel; every eigenvector retained, negative eigenvalues clipped only within1e-10*max(1,maxabs spectrum).',
        'positive_lambda_floor': 'max(1e-12,1e-10*largest nonnegative eigenvalue);100bisection steps, unattainable DF explicitly saturated at floor. No spectral truncation.',
        'weights': 'Every training question1/768, four questions/context; each cohort384questions. Uniform train mean/centering, no target normalization; target variance denominators per cohort from unshuffled training.',
        'lexical_auxiliary_order': ['human_question_token_count/1024','passage_token_count/1024','prompt_token_count/1024','common_prefix_token_count/1024','first_human_question_position/1024','last_human_question_position/1024'],
        'lexical': 'Full native output-vocabulary width; token counts divided by corresponding original aligned token count, then train-only centering, no variance scaling of counts. Six continuous auxiliary fields standardized using training alone; query dot divided by vocab+6, context dot by vocab. Exact sparse centered dot product before bilinear kernel; no answer type/cohort/source answer labels used as features.',
        'target_shuffle': 'Only training assignment i receives complete native target of within-context(i+1)mod4, applied consistently to all five targets. Validation and later targets never shuffled.',
        'selection_order': 'All nine validation grids and chosen rules saved before diagnostic field read. Primary vs target-pair-null selected only on validation. Confirmation remains gated.',
        'retained_operator': 'O=B^T U diag(1/(e+lambda)) U^T B, all768x768 entries in float64. Reconstructed coefficients O(Y-trainmean) audited against spectrum prediction to roundoff, not claimed bit-identical between changed multiplication parenthesizations.',
        'coordinate_scores': 'Preserve all-coordinate absolute/within error, native/predicted within amplitude and response inner product, plus per-question values. Retained operators/native targets recreate any full predicted target; no parameter or coordinate chosen by amplitude.',
        'pending_separate_stage': 'Complete-vocabulary original-BF16head readout and own-history generation are separate GPU runs; this script does not claim them executed.'}
    immutable(path, result)
    return result


def score(predicted, actual, rows, den):
    groups = np.array([r['group_id'] for r in rows])
    cohorts = np.array([r['cohort'] for r in rows])
    arrays = evaluation_arrays(predicted, actual, groups)
    return summarize(arrays, cohorts, den), arrays


def identity_rows(rows):
    return [{k: r[k] for k in ['question_id','group_id','within_context_index','cohort','split']} for r in rows]


def main(key):
    start = time.monotonic()
    freeze()
    contract = effective_contract()
    folder = Path('fit')/key
    finalpath = OUT/folder/'result.json'
    if finalpath.exists():
        old = read(finalpath)
        assert old['implementation_sha256'] == sha(OUT/'fit/implementation_contract.json')
        print('NATURAL_FIT_ALREADY_COMPLETE', key, flush=True)
        return
    storage_guard()
    try:
        train, tg, tq = data.index(key, {'train'})
        valid, vg, vq = data.index(key, {'validation'})
        assert len(train) == 768 and len(valid) == 192
        groups = np.array([r['group_id'] for r in train])
        cohorts = np.array([r['cohort'] for r in train])
        weights = np.ones(len(train))/len(train)
        permutation = data.target_permutation(train)
        tr_features = data.native_features(key, train, tg, tq)
        va_features = data.native_features(key, valid, vg, vq)
        ytr = data.targets(train, tq, 'postnorm')
        yva = data.targets(valid, vq, 'postnorm')
        den = denominators(ytr, groups, cohorts)
        variants = [v['id'] for v in contract['prediction']['variants']]
        selected = {}
        selectionpath = OUT/folder/'validation_selection.json'
        for variant in variants:
            gridpath = OUT/folder/variant/'grid.json'
            if gridpath.exists():
                grid_record = read(gridpath)
                assert grid_record['implementation_sha256'] == sha(OUT/'fit/implementation_contract.json')
                selected[variant] = grid_record['selected']
                continue
            query, context = data.variant_features(key, train, tr_features, variant)
            vquery, vcontext = data.variant_features(key, valid, va_features, variant)
            feature = FeatureKernel(query, context, weights, variant == 'lexical_position')
            vparts = feature.parts(vquery, vcontext)
            yy = ytr[permutation] if variant == 'within_context_target_pair_shuffle' else ytr
            grid = []
            for rho in contract['prediction']['grid']['interaction']:
                gram, _, _ = feature.kernel(feature.train_parts, rho)
                cross, _, _ = feature.kernel(vparts, rho)
                for alpha in contract['prediction']['grid']['contrast_strength']:
                    spectrum = Spectrum(gram, groups, weights, alpha)
                    for target in contract['prediction']['grid']['effective_df_target']:
                        ridge, numerical = spectrum.ridge(target)
                        pred = spectrum.predictions(cross, yy, weights, ridge)
                        summary, _ = score(pred, yva, valid, den)
                        grid.append({'alpha': alpha, 'rho': rho, 'DF': target, 'lambda': ridge,
                            'numerical': numerical, 'validation': summary,
                            'objective': summary['equal_cohort']['normalized_selection_objective']})
                    print('NATURAL_FIT_GRID', key, variant, 'rho', rho, 'alpha', alpha, round(time.monotonic()-start, 1), flush=True)
            def rank_candidate(candidate):
                return (candidate['objective'], candidate['alpha'], candidate['rho'], candidate['DF'])
            best = min(grid, key=rank_candidate)
            alpha0 = min((v for v in grid if v['alpha'] == 0), key=rank_candidate)
            selected[variant] = {'selected': best, 'alpha0': alpha0}
            immutable(gridpath, {'timestamp': stamp(), 'variant': variant, 'grid': grid,
                'selected': selected[variant], 'training_denominators': den,
                'implementation_sha256': sha(OUT/'fit/implementation_contract.json'),
                'training_rows': identity_rows(train), 'validation_rows': identity_rows(valid),
                'diagnostic_or_confirmation_not_used': True})
        primary = min(['early_query_context', 'native_source_read'], key=lambda v: (selected[v]['selected']['objective'], v))
        selection = {'model': key, 'implementation_sha256': sha(OUT/'fit/implementation_contract.json'),
            'native_nonconfirmation_sha256': sha(OUT/f'native/{key}/nonconfirmation/result.json'),
            'variants': selected, 'primary_rule': primary, 'control_rule': 'within_context_target_pair_shuffle',
            'training': identity_rows(train), 'validation': identity_rows(valid),
            'target_denominators': den, 'selection_uses_validation_only': True}
        immutable(selectionpath, selection)
        print('NATURAL_FIT_SELECTION_FROZEN', key, primary, flush=True)
        # First read of diagnostic native fields occurs only after selection.
        diag, dg, dq = data.index(key, {'diagnostic'})
        assert len(diag) == 384
        di_features = data.native_features(key, diag, dg, dq)
        results = []
        for variant in variants:
            recordpath = OUT/folder/variant/'evaluation.json'
            if recordpath.exists():
                results.append(read(recordpath))
                continue
            query, context = data.variant_features(key, train, tr_features, variant)
            feature = FeatureKernel(query, context, weights, variant == 'lexical_position')
            evalparts = {split: feature.parts(*data.variant_features(key, rows, ff, variant))
                for split, rows, ff in [('validation', valid, va_features), ('diagnostic', diag, di_features)]}
            operators = {}
            operator_refs = {}
            for kind, choice in selected[variant].items():
                gram, column, grand = feature.kernel(feature.train_parts, choice['rho'])
                spectrum = Spectrum(gram, groups, weights, choice['alpha'])
                op = spectrum.solution(choice['lambda'])
                operators[kind] = op
                operator_refs[kind] = commit_arrays(folder/variant/'operators', kind,
                    {'solution_operator': op, 'eigenvalues': spectrum.eigenvalues, 'weights': weights,
                     'kernel_column_mean': column, 'kernel_grand_mean': np.array(grand), **feature.state})
                if kind == 'selected' and variant in [primary, 'within_context_target_pair_shuffle']:
                    yy = ytr[permutation] if variant == 'within_context_target_pair_shuffle' else ytr
                    mean = weights @ yy
                    coef = op @ (yy-mean)
                    assert not feature.lexical
                    deploy = {'coefficients': coef, 'target_mean': mean, 'weights': weights,
                        'train_query': feature.training['query'], 'train_context': feature.training['context'],
                        'kernel_column_mean': column, 'kernel_grand_mean': np.array(grand), **feature.state}
                    check_cross, _, _ = feature.kernel(evalparts['validation'], choice['rho'])
                    direct = spectrum.predictions(check_cross, yy, weights, choice['lambda'])
                    rebuilt = check_cross @ coef+mean
                    maxerror = float(np.abs(direct-rebuilt).max())
                    assert np.allclose(direct, rebuilt, atol=1e-7, rtol=1e-7), maxerror
                    ref = commit_arrays(folder/'deployed', variant, deploy)
                    immutable(OUT/folder/'deployed'/(variant+'.json'), {'variant': variant, 'choice': choice,
                        'field': ref, 'selection_sha256': sha(selectionpath), 'operator_reconstruction_maximum_error': maxerror,
                        'training': identity_rows(train), 'fitted_on_first_prefix_only': True,
                        'later_own_history_out_of_training_support': True})
            evaluations = []
            for target in contract['prediction']['targets']:
                yy_native = ytr if target == 'postnorm' else data.targets(train, tq, target)
                yy = yy_native[permutation] if variant == 'within_context_target_pair_shuffle' else yy_native
                mean = weights @ yy
                targetden = denominators(yy_native, groups, cohorts)
                for kind, choice in selected[variant].items():
                    coef = operators[kind] @ (yy-mean)
                    for split, rows, qq in [('validation', valid, vq), ('diagnostic', diag, dq)]:
                        actual = yva if target == 'postnorm' and split == 'validation' else data.targets(rows, qq, target)
                        cross, _, _ = feature.kernel(evalparts[split], choice['rho'])
                        predicted = cross @ coef+mean
                        summary, arrays = score(predicted, actual, rows, targetden)
                        reference = commit_arrays(folder/variant/'scores', kind+'_'+target+'_'+split, arrays)
                        evaluations.append({'kind': kind, 'target': target, 'split': split,
                            'summary': summary, 'field': reference, 'target_training_denominators': targetden})
                    del coef
            result = {'timestamp': stamp(), 'variant': variant, 'selection_sha256': sha(selectionpath),
                'operator_fields': operator_refs, 'evaluations': evaluations,
                'diagnostic': identity_rows(diag), 'validation': identity_rows(valid),
                'readout_and_bootstrap_pending_separate_analysis': True}
            immutable(recordpath, result)
            results.append(result)
            print('NATURAL_FIT_VARIANT_EVALUATED', key, variant, round(time.monotonic()-start, 1), flush=True)
        result = {'timestamp': stamp(), 'all_passed': True, 'model': key,
            'implementation_sha256': sha(OUT/'fit/implementation_contract.json'),
            'selection_sha256': sha(selectionpath), 'variants': variants,
            'grid_candidates': 36*len(variants), 'full_targets': contract['prediction']['targets'],
            'primary_rule': primary, 'control_rule': 'within_context_target_pair_shuffle',
            'training_questions': 768, 'validation_questions': 192, 'diagnostic_questions': 384,
            'confirmation_observed': False, 'seconds': time.monotonic()-start,
            'scope': 'Full-coordinate fitted predictions and retained operators completed; paired uncertainty, complete-vocabulary readout and own histories remain separate stages. Not a completed mechanism/Phase.'}
        immutable(finalpath, result)
        print('NATURAL_FIT_COMPLETE', key, round(result['seconds'], 1), flush=True)
    except Exception as exc:
        failure(OUT/folder, start, exc)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen4','qwen14','glm4'], required=True)
    args = parser.parse_args()
    main(args.model)
