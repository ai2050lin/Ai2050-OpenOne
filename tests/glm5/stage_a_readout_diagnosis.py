"""Stage A (Phase 384) diagnosis of an old result, CPU only, no model forward.

Question: Phase 2887 found that a language readout fitted under one colour fails
(~50 %) on the other colour at hs[33..35].  Is that a threshold (centre) failure,
a direction change, or an artefact of feature scale?

Material: the 128 discovery / 512 evaluation families of MLP-NATIVE-PUZZLES-20261109-v1.
All of it was exposed before; this is exploratory, not a confirmation.
The decision rules in CRITERIA were written before any number below was computed.
"""
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import tools

ROOT = Path(tools.__file__).resolve().parents[1]
WORK = Path(os.environ['AI2050_TEST_WORK_DIR'])
SPEC = json.loads(Path(os.environ['AI2050_TEST_QUESTIONS']).read_text(encoding='utf-8'))
SOURCE, PRIOR = ROOT / SPEC['source'], ROOT / SPEC['prior']
LAYERS, POS = SPEC['report_layers'], 1  # answer_prefix
CRITERIA = {
    'centre_explains': 'M1 <= 0.60 and M2 >= 0.90 and paired >= 0.90',
    'direction_matters': 'M2 < 0.90 and M3 >= 0.90',
    'scale_sensitive': 'verdict differs between variance-weighted and unweighted readout, '
                       'or the 10 largest coordinates carry > 50 % of the absolute centre shift',
    'otherwise': 'unresolved',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def unbits(a):
    return (np.asarray(a, dtype=np.uint32) << 16).view(np.float32)


def design(rows):
    return np.array([[1, r['query'], r['record'], r['language'], r['query'] * r['language'],
                      r['record'] * r['language'], r['query'] * r['record'],
                      r['query'] * r['record'] * r['language']] for r in rows], dtype=np.float64)


def readout(beta, var, colour, weighted):
    """Language readout fitted on discovery families of one colour (columns: 3 language, 6 colour, 7 both)."""
    centre = beta[:, 0] + colour * beta[:, 6]
    direction = beta[:, 3] + colour * beta[:, 7]
    weight = direction / np.maximum(var, 1e-6) if weighted else direction.copy()
    weight /= (direction * weight).sum(-1, keepdims=True)
    return centre, direction, weight


def main():
    start = time.monotonic()
    estimator = read(PRIOR / 'estimator.json')
    assert sha(PRIOR / 'frozen-coefficients.npz') == estimator['coefficients_sha256']
    with np.load(PRIOR / 'frozen-coefficients.npz') as z:
        beta = z['beta'][:, POS].astype(np.float64)       # [37, 8, 2560], discovery only
        var = z['variance'][:, POS].astype(np.float64)    # [37, 2560], discovery only
    manifest = {r['path']: r['sha256'] for r in read(PRIOR / 'plan.json')['native_files']}
    rows = read(SOURCE / 'holdout-materials.json')
    x = design(rows[:8])
    assert np.array_equal(x.T @ x, 8 * np.eye(8))
    families = len(rows) // 8

    fits = {(c, w): readout(beta, var, c, w) for c in (-1, 1) for w in (True, False)}
    pairs = [(i, j) for i in range(8) for j in range(8)
             if x[i, 3] == 1 and x[j, 3] == -1 and np.array_equal(x[i, [1, 2]], x[j, [1, 2]])]
    acc = {k: np.zeros(37) for k in
           [(m, c, w) for m in ('M1', 'M2', 'M3', 'paired') for c in (-1, 1) for w in (True, False)]}
    class_mean = np.zeros((37, 2))  # M1, train red, weighted: mean score of English / Chinese on blue
    pred = {'four_state': np.einsum('vf,lfd->lvd', x[:, [0, 3, 6, 7]], beta[:, [0, 3, 6, 7]]),
            'eight_cell': np.einsum('vf,lfd->lvd', x, beta)}
    mse = {(name, kind): np.zeros(37) for name in pred for kind in ('raw', 'standardised')}
    live = var > 1e-6

    for f in range(families):
        path = SOURCE / f'holdout-{f:04d}.npz'
        assert sha(path) == manifest[path.name], path.name
        assert np.array_equal(design(rows[f * 8:f * 8 + 8]), x)
        with np.load(path) as z:
            h = unbits(z['hidden'][:, :, POS]).transpose(1, 0, 2).astype(np.float64)  # [37, 8, 2560]
        for (c, w), (centre, _, weight) in fits.items():
            keep = x[:, 6] == -c                       # evaluate on the other colour
            labels = x[keep, 3] > 0
            other = fits[(-c, w)]
            for name, cen, wt in (('M1', centre, weight), ('M2', other[0], weight), ('M3', other[0], other[2])):
                s = np.einsum('lvd,ld->lv', h[:, keep] - cen[:, None], wt)
                acc[(name, c, w)] += ((s >= 0) == labels).mean(-1)
                if name == 'M1' and c == 1 and w:
                    class_mean[:, 0] += s[:, labels].mean(-1)
                    class_mean[:, 1] += s[:, ~labels].mean(-1)
            diffs = np.stack([h[:, i] - h[:, j] for i, j in pairs if x[i, 6] == -c], 1)
            acc[('paired', c, w)] += (np.einsum('lpd,ld->lp', diffs, weight) > 0).mean(-1)
        for name, p in pred.items():
            err = (h - p) ** 2
            mse[(name, 'raw')] += err.mean((1, 2))
            mse[(name, 'standardised')] += np.where(live[:, None], err / np.maximum(var, 1e-6)[:, None], 0).sum((1, 2)) \
                / (8 * np.maximum(live.sum(-1), 1))
        if (f + 1) % 64 == 0 or f + 1 == families:
            used = time.monotonic() - start
            print(f'progress step=evaluate done={f + 1}/{families} elapsed={used:.0f}s '
                  f'eta={used / (f + 1) * (families - f - 1):.0f}s', flush=True)

    for v in list(acc.values()) + list(mse.values()):
        v /= families
    class_mean /= families

    def verdict(layer, w):
        m1, m2, m3, pr = (min(acc[(m, c, w)][layer] for c in (-1, 1)) if m != 'M1'
                          else max(acc[(m, c, w)][layer] for c in (-1, 1)) for m in ('M1', 'M2', 'M3', 'paired'))
        if m1 <= .60 and m2 >= .90 and pr >= .90:
            return 'centre_explains'
        if m2 < .90 and m3 >= .90:
            return 'direction_matters'
        return 'unresolved'

    report = {}
    for layer in LAYERS:
        centre, direction, weight = fits[(1, True)]
        shift = weight[layer] * (fits[(-1, True)][0][layer] - centre[layer])   # per-coordinate centre shift in score units
        order = np.argsort(-np.abs(shift))
        vorder = np.argsort(-var[layer])
        a, b = direction[layer], fits[(-1, True)][1][layer]
        cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
        sigma = np.sqrt(np.maximum(var[layer], 1e-6))
        rest = np.ones(2560, bool); rest[vorder[:10]] = False
        verdicts = {('weighted' if w else 'unweighted'): verdict(layer, w) for w in (True, False)}
        top10 = float(np.abs(shift[order[:10]]).sum() / np.abs(shift).sum())
        report[f'hs{layer}'] = {
            'accuracy': {f"{m}_train_{'red' if c == 1 else 'blue'}_{'weighted' if w else 'unweighted'}":
                         round(float(v[layer]), 6) for (m, c, w), v in acc.items()},
            'M1_train_red_weighted_mean_score': {'english': round(float(class_mean[layer, 0]), 6),
                                                 'chinese': round(float(class_mean[layer, 1]), 6)},
            'centre_shift': {'signed_total': round(float(shift.sum()), 6),
                             'half_gap_on_target': round(float(weight[layer] @ b), 6),
                             'top1_abs_share': round(float(np.abs(shift[order[0]]) / np.abs(shift).sum()), 6),
                             'top10_abs_share': round(top10, 6), 'top10_coordinates': order[:10].tolist()},
            'variance': {'top1_share': round(float(var[layer, vorder[0]] / var[layer].sum()), 6),
                         'top10_share': round(float(var[layer, vorder[:10]].sum() / var[layer].sum()), 6),
                         'top10_coordinates': vorder[:10].tolist()},
            'language_direction_cosine_red_vs_blue': {'raw': round(cos(a, b), 6),
                                                      'per_coordinate_standardised': round(cos(a / sigma, b / sigma), 6),
                                                      'raw_without_top10_variance_coordinates': round(cos(a[rest], b[rest]), 6)},
            'four_state_over_eight_cell_mse': {k: round(float(mse[('four_state', k)][layer] / mse[('eight_cell', k)][layer]), 6)
                                               for k in ('raw', 'standardised')},
            'verdict': verdicts,
            'scale_sensitive': bool(verdicts['weighted'] != verdicts['unweighted'] or top10 > .5),
        }
    conclusion = {
        'schema': 'stage_a_readout_diagnosis.v1', 'status': 'exploratory; all material previously exposed',
        'criteria': CRITERIA, 'families': {'fit_discovery': 128, 'evaluated': families}, 'position': 'answer_prefix',
        'definitions': {'M1': 'centre and direction from the training colour (reproduces Phase 2887)',
                        'M2': 'centre of the target colour (discovery, needs the colour label), direction of the training colour',
                        'M3': 'centre and direction of the target colour (reference)',
                        'paired': 'centre-free: same family, same query and record, English minus Chinese, training-colour direction',
                        'note': 'M2/M3 use the colour label to choose a centre; they describe geometry, not a text-only predictor'},
        'inputs': {'coefficients_sha256': estimator['coefficients_sha256'], 'model_snapshot': estimator.get('model_snapshot')},
        'layers': report, 'elapsed_seconds': round(time.monotonic() - start, 1),
    }
    (WORK / 'conclusion.json').write_text(json.dumps(conclusion, ensure_ascii=False, indent=1), encoding='utf-8')
    print('done', flush=True)


if __name__ == '__main__':
    main()
