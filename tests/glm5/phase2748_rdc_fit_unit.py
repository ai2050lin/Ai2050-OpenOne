"""Data-independent numerical tests before native-response fits."""
from rdc_question_common import *
from rdc_question_fit import *
from rdc_question_selectivity import SelectivityKernel


def main():
    rng = np.random.default_rng(2748003)
    n = 24
    groups = np.repeat(np.arange(n//4), 4)
    weights = rng.uniform(.2, 2., n)
    weights /= weights.sum()
    q, c, y = rng.normal(size=(n, 7)), rng.normal(size=(n, 5)), rng.normal(size=(n, 11))
    qnew, cnew = rng.normal(size=(8, 7)), rng.normal(size=(8, 5))
    checks = []
    feature = FeatureKernel(q, c, weights)
    for strength in [0., 4., 16.]:
        for interaction in [0., 1., 3.]:
            train, _, _ = feature.kernel(feature.train_parts, interaction)
            cross, _, _ = feature.kernel(feature.parts(qnew, cnew), interaction)
            spectrum = Spectrum(train, groups, weights, strength)
            ridge, numeric = spectrum.ridge(8)
            actual = spectrum.predictions(cross, y, weights, ridge)
            reference = SelectivityKernel(strength, ridge, interaction).fit(q, c, y, groups, weights).predict(qnew, cnew)
            delta = float(np.abs(actual-reference).max())
            assert delta < 1e-9 and abs(numeric['DF_actual']-8) < 1e-8
            op = spectrum.solution(ridge)
            rebuilt = cross @ (op @ (y-weights @ y))+weights @ y
            assert np.allclose(actual, rebuilt, atol=1e-9, rtol=1e-9)
            checks.append({'case': 'full_spectrum_vs_existing_kernel_and_retained_operator',
                'alpha': strength, 'rho': interaction, 'maximum_error': delta})
    # Exact sparse centering with all count coordinates plus six continuous
    # fields equals explicitly materialized dense features, including zeros.
    lq = sparse.csr_matrix(np.concatenate([rng.poisson(.2, (n, 31))/12, rng.normal(size=(n, 6))], axis=1))
    lc = sparse.csr_matrix(rng.poisson(.2, (n, 31))/28)
    lexical = FeatureKernel(lq, lc, weights, lexical=True)
    for role, values, index in [('query', lq, 0), ('context', lc, 1)]:
        xx = values.toarray()/lexical.state[role+'_scale']-lexical.state[role+'_mean']
        delta = float(np.abs(lexical.train_parts[index]-xx @ xx.T/xx.shape[1]).max())
        assert delta < 1e-12
        checks.append({'case': 'full_vocabulary_sparse_exact_centering', 'role': role, 'maximum_error': delta})
    context = np.repeat(rng.normal(size=(6, 5)), 4, axis=0)
    cf = FeatureKernel(np.zeros((n, 1)), context, weights)
    train, _, _ = cf.kernel(cf.train_parts, 3.)
    spectrum = Spectrum(train, groups, weights, 16.)
    ridge, numeric = spectrum.ridge(16)
    assert numeric['saturated'] and numeric['DF_actual'] < 6 and len(spectrum.eigenvalues) == n
    checks.append({'case': 'context_only_DF_saturation_all_eigenvectors_retained', **numeric})
    result = {'timestamp': stamp(), 'all_passed': True, 'checks': checks,
        'source': snapshot(__file__), 'solver': snapshot(Path(__file__).with_name('rdc_question_fit.py')),
        'existing_reference': snapshot(Path(__file__).with_name('rdc_question_selectivity.py')),
        'scope': 'Synthetic CPU algebra only; no language result or confirmation observation.'}
    path = OUT/'unit'/('fit_solver_'+str(time.time_ns())+'.json')
    immutable(path, result)
    save(OUT/'unit/fit_solver_current.json', result)
    print('NATURAL_FIT_UNIT_PASS', len(checks), flush=True)


if __name__ == '__main__':
    main()
