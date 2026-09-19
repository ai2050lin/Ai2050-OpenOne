"""Independent small source contractions, identities and rejected scopes."""
from itertools import permutations
from rdc_question_common import *
import phase2748_rdc_source_coupling as analysis


def main():
    rng = np.random.default_rng(2748016); a = rng.random((4, 6)); a /= a.sum(1)[:, None]
    v = rng.normal(size=(2, 6, 3)); passage = [1, 2, 3]; question = [4]; order = np.array([0, 2, 3, 1, 4, 5])
    r = analysis.contraction(a, v, passage, question, order)
    expected = np.array([[sum(a[h,i]*v[h//2,i,d] for i in range(6)) for d in range(3)] for h in range(4)])
    assert np.allclose(r['full_read'], expected, rtol=0, atol=1e-14)
    expected_delta = np.array([[sum(a[h,i]*(v[h//2,order[i],d]-v[h//2,i,d]) for i in passage) for d in range(3)] for h in range(4)])
    assert np.allclose(r['delta'], expected_delta, rtol=0, atol=1e-14)
    all_reads = []
    for p in permutations(passage):
        o = np.arange(6); o[passage] = p
        rr = analysis.contraction(a, v, passage, question, o)
        all_reads.append(rr['weighted']+rr['delta'])
    assert np.allclose(np.mean(all_reads, axis=0), r['uniform'], rtol=0, atol=1e-14)
    flat = a.copy(); flat[:, passage] = flat[:, passage].mean(1)[:, None]
    rr = analysis.contraction(flat, v, passage, question, order)
    assert np.max(np.abs(rr['coupling'])) < 1e-14 and np.max(np.abs(rr['delta'])) < 1e-14
    fixed = v.copy(); fixed[:, passage] = fixed[:, 1:2]
    rr = analysis.contraction(a, fixed, passage, question, order)
    assert np.max(np.abs(rr['coupling'])) < 1e-14 and np.max(np.abs(rr['delta'])) < 1e-14
    zero = a.copy(); zero[:, passage] = 0
    rr = analysis.contraction(zero, v, passage, question, order)
    assert not rr['effective_fraction_valid'].any() and not rr['weighted'].any()
    rr = analysis.contraction(a*.999, v, passage, question, order)
    assert np.allclose(rr['statistics'][:, 1:4], r['statistics'][:, 1:4], atol=1e-14)
    failures = 0
    for bad_order, bad_question in [(np.array([1,0,2,3,4,5]), question), (np.array([0,1,1,3,4,5]), question), (order, [1,4])]:
        try: analysis.contraction(a, v, passage, bad_question, bad_order)
        except AssertionError: failures += 1
        else: raise AssertionError('Invalid source partition/permutation accepted')
    assert failures == 3
    result = {'timestamp': stamp(), 'all_passed': True, 'checks': 9,
        'check_names': ['independent_all_GQA_heads_and_scalar_sums', 'all_passage_permutation_mean_equals_uniform',
            'uniform_weights_remove_pairing', 'constant_values_remove_pairing', 'zero_passage_mass_explicit_undefined_effective_fraction',
            'raw_BF16_mass_normalization', 'reject_nonpassage_permutation', 'reject_nonbijective_permutation', 'reject_overlapping_roles'],
        'analysis_sha256': sha(analysis.__file__), 'source': snapshot(__file__),
        'scope': 'CPU synthetic arithmetic only, no language field or model output used.'}
    immutable(OUT/'unit'/('source_coupling_'+str(time.time_ns())+'.json'), result)
    save(OUT/'unit/source_coupling_current.json', result)
    print('NATURAL_SOURCE_COUPLING_UNIT_PASS', result['checks'], flush=True)


if __name__ == '__main__': main()
