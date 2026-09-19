"""CPU-only synthetic implementation checks, not language evidence."""
from phase2745_rdc_construction_fit import eigensystem, solve_projected, correspondence
from rdc_construction_common import *


def main():
    import torch
    from phase2745_rdc_construction_diagonal import fit_small, paired_features
    torch.set_num_threads(2)
    rng = np.random.default_rng(2745)
    x = rng.normal(size=(53, 17))
    x[:, -1] = x[:, 3]  # Rank deficient; ridge still uses every coordinate.
    a = rng.normal(size=(17, 9))
    y = x@a + .001*rng.normal(size=(53, 9))
    t = torch.from_numpy(x)
    normalized, scale, covariance, eigenvalues, vectors, residual = eigensystem(t)
    cross = normalized.T@torch.from_numpy(y)/len(x)
    projected = vectors.T@cross
    errors = []
    for lam in [.001, .01, .1, 1., 10.]:
        observed = solve_projected(vectors, eigenvalues, projected, lam).numpy()/scale.numpy()[:, None]
        xn = x/scale.numpy()
        expected = np.linalg.solve(xn.T@xn/len(x)+lam*np.eye(17), xn.T@y/len(x))/scale.numpy()[:, None]
        error = float(np.max(abs(observed-expected)))
        assert error < 1e-9
        errors.append({'lambda': lam, 'max_coefficient_error': error})
    pairs = [{'family': f'f{i//16}', 'language': 'en' if i%2 == 0 else 'zh', 'case': i%16} for i in range(64)]
    train = np.arange(64)
    perm = correspondence(pairs, train)
    assert sorted(perm) == train.tolist()
    assert all((pairs[i]['family'], pairs[i]['language']) == (pairs[perm[i]]['family'], pairs[perm[i]]['language']) for i in train)
    assert np.array_equal(perm, correspondence(pairs, train))
    pq = rng.normal(size=(64, 5, 17))
    flat = (perm[:, None]*5+np.arange(5)[None]).reshape(-1)
    assert np.array_equal(pq[perm].reshape(-1, 17), pq.reshape(-1, 17)[flat])
    # An operator is applied to each native state; evaluating a pair is not patching.
    ca, cb = rng.normal(size=(2, 5, 17))
    pair_error = float(np.max(abs((cb-ca)@a - (cb@a-ca@a))))
    assert pair_error < 1e-12
    xx = rng.normal(size=(53, 17, 4))
    xx[..., 0] = 1
    yy = rng.normal(size=(53, 17, 2))
    cov = np.einsum('qdf,qdg->dfg', xx, xx)
    cross = np.einsum('qdf,qdt->dft', xx, yy)
    diagonal_error = 0.
    for absolute in [False, True]:
        coef = fit_small(cov, cross, len(xx), .1, absolute)
        for d in range(17):
            scale = np.sqrt((xx[:, d]**2).mean(0))
            xn = xx[:, d]/scale
            penalty = np.eye(4)*.1
            if absolute:
                penalty[0, 0] = 0
            ref = np.linalg.solve(xn.T@xn/len(xx)+penalty, xn.T@yy[:, d]/len(xx))/scale[:, None]
            diagonal_error = max(diagonal_error, float(abs(coef[d]-ref).max()))
    assert diagonal_error < 1e-12
    aa = [rng.normal(size=(5, 17, 4)) for _ in range(5)]+[rng.normal(size=(5, 17, 2))]
    bb = [rng.normal(size=a.shape) for a in aa]
    for j, delta in enumerate(paired_features(aa, bb)):
        assert delta.shape == (5, 17, 2 if j < 5 else 1)
    result = {'timestamp': stamp(), 'source': snapshot(__file__),
        'fit_source': snapshot(Path(__file__).with_name('phase2745_rdc_construction_fit.py')),
        'all_passed': True, 'device': 'CPU', 'synthetic': True,
        'ridge_checks': errors, 'full_covariance_reconstruction_relative_error': residual,
        'native_coordinates_removed': 0, 'pair_operator_algebra_error': pair_error,
        'diagonal_source': snapshot(Path(__file__).with_name('phase2745_rdc_construction_diagonal.py')),
        'batched_diagonal_ridge_max_error': diagonal_error,
        'training_pair_shuffle_group_safe': True, 'query_alignment_preserved': True,
        'scope': 'Tests actual shared solver functions and pair permutation indexing, including a rank-deficient synthetic input. Does not establish language generalization.'}
    save(BASE / 'fit_preflight/result.json', result)
    print('CONSTRUCTION_FIT_PREFLIGHT_PASS', max(r['max_coefficient_error'] for r in errors), flush=True)


if __name__ == '__main__':
    main()
