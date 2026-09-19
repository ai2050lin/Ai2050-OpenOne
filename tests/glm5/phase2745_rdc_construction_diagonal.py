"""Streamed full-coordinate, same-input diagonal controls for query prediction."""
import argparse
from phase2745_rdc_construction_contract import freeze, freeze_extraction
from phase2745_rdc_construction_fit import correspondence, SPLITS, QUERY_SPLITS
from rdc_construction_common import *

NAMES = ['query_only', 'uniform', 'quadratic', 'shuffled_values', 'ordered_softmax', 'actual_query_H1']
OBJECTIVES = ['absolute', 'paired']
CONTROLS = ['true_correspondence', 'training_pair_shuffled']


def freeze_diagonal():
    path = BASE / 'diagonal_protocol.json'
    if path.exists():
        return read(path)
    freeze_extraction()
    value = {'timestamp': stamp(), 'source': snapshot(__file__),
        'timing': 'Secondary specification during partial native collection, before any fitted2745target result inspection; primary extraction protocol unchanged.',
        'candidate_inputs': 'Five earlier available-prefix rules use [1,prefix_HElast,standalone_query_H(E+1),candidate_H(E+1)] per coordinate; actualH1 control uses [1,actualH1] and nothing later.',
        'paired_inputs': 'Bias and query-alone terms cancel: five candidates use [deltaPrefix_HElast,deltaCandidate]; same-input actualH1 diagonal uses only deltaH1.',
        'targets': ['Hearly', 'postnorm'], 'objectives': OBJECTIVES, 'controls': CONTROLS,
        'scaling': 'All input features divided by training RMS coordinatewise, floor1e-8; paired maps through origin. Absolute intercept not penalized.',
        'lambdas': [.001, .01, .1, 1., 10.],
        'selection': 'One lambda per candidate/objective/control/target, selected on validation cases and validation queries; all test strata follow freeze.',
        'shuffle': 'Identical hash-seeded training pair permutation as the full operator, within family/language; keep world and query alignment.',
        'scope': 'Every native coordinate retained. Paired actualH1 is the same-input no-cross-coordinate baseline, whereas prefix candidates use different declared information. No activation injection.'}
    immutable(path, value)
    return value


def packet(key, row, early, proto, features=True):
    path = BASE / 'capture' / key / 'fields' / (row['sample_id']+'.npz')
    commit = read(BASE / 'capture' / key / 'commits' / (row['sample_id']+'.json'))
    assert sha(path) == commit['sha256']
    with np.load(path) as z:
        ix = z['query_layer_indices'].tolist()
        states = z['query_selected_states']
        he = unbits(states[ix.index(early)]).astype(float)
        post = unbits(z['postnorm']).astype(float)
        target = np.stack([he, post], -1)
        if not features:
            return target
        h1 = unbits(states[ix.index(1)]).astype(float)
        prefix = unbits(z['prefix_layers'][early]).astype(float)
        candidates = unbits(z['candidate_early_output']).astype(float)
    prefix = np.broadcast_to(prefix, he.shape)
    values = [np.stack([np.ones_like(he), prefix, proto, candidates[j]], -1) for j in range(5)]
    values.append(np.stack([np.ones_like(he), h1], -1))
    return values, target


def pair_packet(key, rows, pair_index, early, proto, features=True):
    return [packet(key, r, early, proto, features) for r in rows[2*pair_index:2*pair_index+2]]


def paired_features(a, b):
    return [(b[j]-a[j])[..., [1, 3]] if j < 5 else (b[j]-a[j])[..., [1]] for j in range(6)]


def fit_small(cov, cross, count, lam, absolute):
    covariance = cov/count
    scale = np.sqrt(np.maximum(np.diagonal(covariance, axis1=-2, axis2=-1), 1e-16))
    norm = covariance/scale[..., :, None]/scale[..., None, :]
    rhs = cross/count/scale[..., :, None]
    penalty = np.eye(norm.shape[-1])*lam
    if absolute:
        penalty[0, 0] = 0
    coef = np.linalg.solve(norm+penalty, rhs)/scale[..., :, None]
    assert np.isfinite(coef).all()
    return coef


def main(key):
    specification = freeze_diagonal()
    out = BASE / 'diagonal' / key
    if (out / 'result.json').exists():
        assert read(out / 'result.json')['all_passed']
        return
    start = time.monotonic()
    _, material = freeze()
    native = read(BASE / 'capture' / key / 'result.json')
    assert native['all_passed']
    width, early = native['width'], native['early']
    rows, probes = material['models'][key]['rows'], material['models'][key]['probes']
    pairs = [{k: r[k] for k in ['pair_id', 'source_group', 'family', 'language', 'split']} for r in rows[::2]]
    pi = {s: np.array([i for i, r in enumerate(pairs) if r['split'] == s]) for s in SPLITS}
    qi = {s: np.array([i for i, q in enumerate(probes) if q['split'] == s]) for s in QUERY_SPLITS}
    permutation = correspondence(pairs, pi['train'])
    with np.load(BASE / 'capture' / key / 'prototypes.npz') as z:
        proto = unbits(z['query_all_states'][early+1]).astype(float)
    stats = {}
    for v in range(6):
      for ob in range(2):
        f = (4 if v < 5 else 2) if ob == 0 else (2 if v < 5 else 1)
        for control in range(2):
            stats[v, ob, control] = [np.zeros((width, f, f)), np.zeros((width, f, 2)), 0]
    for slot, p in enumerate(pi['train']):
        (xa, ya), (xb, yb) = pair_packet(key, rows, p, early, proto)
        sy = pair_packet(key, rows, pi['train'][permutation[slot]], early, proto, False)
        dx = paired_features(xa, xb)
        for control, target in enumerate([[ya, yb], sy]):
          for ob in range(2):
            xs = [np.concatenate([xa[v], xb[v]], 0) for v in range(6)] if ob == 0 else dx
            yy = np.concatenate(target, 0) if ob == 0 else target[1]-target[0]
            qq = np.r_[qi['train_query'], qi['train_query']+100] if ob == 0 else qi['train_query']
            y = yy[qq]
            for v in range(6):
                x = xs[v][qq]
                st = stats[v, ob, control]
                st[0] += np.einsum('qdf,qdg->dfg', x, x, optimize=True)
                st[1] += np.einsum('qdf,qdt->dft', x, y, optimize=True)
                st[2] += len(qq)
        if (slot+1)%16 == 0:
            print('CONSTRUCTION_DIAG_STATS', key, slot+1, len(pi['train']), flush=True)
    candidates = {k: np.stack([fit_small(*st, lam, k[1] == 0) for lam in specification['lambdas']]) for k, st in stats.items()}
    val = {k: np.zeros((5, 2)) for k in candidates}
    for p in pi['validation']:
        (xa, ya), (xb, yb) = pair_packet(key, rows, p, early, proto)
        dx = paired_features(xa, xb)
        for k, coefs in candidates.items():
            v, ob, control = k
            x = np.concatenate([xa[v], xb[v]], 0) if ob == 0 else dx[v]
            y = np.concatenate([ya, yb], 0) if ob == 0 else yb-ya
            qq = np.r_[qi['validation_query'], qi['validation_query']+100] if ob == 0 else qi['validation_query']
            pred = np.einsum('qdf,ldft->lqdt', x[qq], coefs, optimize=True)
            val[k] += ((pred-y[qq])**2).mean((1, 2))
    selected, selections = {}, []
    for k, coefs in candidates.items():
        chosen = val[k].argmin(0)
        c = np.stack([coefs[chosen[t], :, :, t] for t in range(2)], -1)
        selected[k] = c
        name = NAMES[k[0]]+'__'+OBJECTIVES[k[1]]+'__'+CONTROLS[k[2]]
        npz(out / 'coefficients' / (name+'.npz'), coefficients=c)
        selections.append({'name': name, 'validation_MSE': (val[k]/len(pi['validation'])).tolist(),
            'selected_indices': chosen.tolist(), 'selected_lambdas': [specification['lambdas'][i] for i in chosen],
            'coefficient_count': int(c.size), 'all_native_coordinates': width})
    save(out / 'validation_selection.json', {'timestamp': stamp(), 'selections': selections,
        'test_evaluation_not_started': True, 'training_pair_permutation': permutation.tolist()})
    del candidates, stats
    # Store both absolute objective errors and pair-change errors for every fit.
    absolute = np.zeros((6, 2, 2, 2, len(pairs), 100))
    paired = np.zeros_like(absolute)
    zero = np.zeros((2, len(pairs), 100))
    coordinates = np.zeros((6, 2, 2, 2, 3, 3, width))
    counts = np.zeros((3, 3), int)
    for p, pair in enumerate(pairs):
        (xa, ya), (xb, yb) = pair_packet(key, rows, p, early, proto)
        dx, dy = paired_features(xa, xb), yb-ya
        zero[:, p] = (dy**2).mean(1).T
        si = SPLITS.index(pair['split'])
        for qj, qs in enumerate(QUERY_SPLITS):
            counts[si, qj] += len(qi[qs])
        for k, coef in selected.items():
            v, ob, control = k
            if ob == 0:
                pa = np.einsum('qdf,dft->qdt', xa[v], coef, optimize=True)
                pb = np.einsum('qdf,dft->qdt', xb[v], coef, optimize=True)
                err = ((pa-ya)**2+(pb-yb)**2)/2
                absolute[v, ob, control, :, p] = err.mean(1).T
                prediction = pb-pa
            else:
                prediction = np.einsum('qdf,dft->qdt', dx[v], coef, optimize=True)
                # No absolute anchoring is claimed for a pair-only map.
                absolute[v, ob, control, :, p] = 0
            square = (prediction-dy)**2
            paired[v, ob, control, :, p] = square.mean(1).T
            for qj, qs in enumerate(QUERY_SPLITS):
                coordinates[v, ob, control, :, si, qj] += square[qi[qs]].sum(0).T
        if (p+1)%32 == 0:
            print('CONSTRUCTION_DIAG_EVAL', key, p+1, len(pairs), flush=True)
    coordinates /= counts[None, None, None, None, :, :, None]
    npz(out / 'all_errors.npz', paired_change_MSE=paired, absolute_MSE_for_absolute_objective_only=absolute,
        zero_change_MSE=zero, all_coordinate_pair_error=coordinates, stratum_counts=counts)
    summaries = []
    for ss in SPLITS:
      for qs in QUERY_SPLITS:
        ii, qq = pi[ss], qi[qs]
        groups = [pairs[p]['source_group'] for p in ii]
        for v in range(6):
          for ob in range(2):
            for t, target in enumerate(['Hearly', 'postnorm']):
                true = paired[v, ob, 0, t, ii][:, qq].mean(-1)
                shuffled = paired[v, ob, 1, t, ii][:, qq].mean(-1)
                z = zero[t, ii][:, qq].mean(-1)
                summaries.append({'source_split': ss, 'query_split': qs, 'input': NAMES[v],
                    'objective': OBJECTIVES[ob], 'target': target, 'pairs': len(ii),
                    'true_pair_change_MSE': clustered(true, groups),
                    'zero_minus_true_MSE': clustered(z-true, groups),
                    'shuffled_minus_true_MSE': clustered(shuffled-true, groups)})
    result = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True, 'model': key,
        'variants': NAMES, 'objectives': OBJECTIVES, 'controls': CONTROLS,
        'summaries': summaries, 'selections': selections, 'seconds': time.monotonic()-start,
        'scope': specification, 'global_encoding_mechanism_closed': False}
    save(out / 'pair_index.json', pairs)
    save(out / 'result.json', result)
    ledger('construction_diagonal_same_input_'+key, result['seconds'])
    print('CONSTRUCTION_DIAGONAL_DONE', key, result['seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', nargs='?', choices=['qwen4', 'qwen14', 'glm4'])
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze_diagonal()
    else:
        assert args.model
        main(args.model)
