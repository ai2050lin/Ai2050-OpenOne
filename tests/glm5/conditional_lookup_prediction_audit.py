"""Recompute frozen confirmation predictions directly, no refitting or selection."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
import numpy as np
from conditional_lookup_stage_audit import SOURCE, OUTPUT, read, sha, bits_to_float
from deploy.storage_policy import check_write


def main():
    seal = read(SOURCE / 'candidate.json')
    report = read(SOURCE / 'confirm-analysis.json')
    for name, digest in seal['files'].items():
        assert sha(SOURCE / name) == digest
    collection = read(SOURCE / 'confirm-collection.json')
    rows = read(SOURCE / 'confirm-materials.json')
    keep = [i for i, r in enumerate(rows) if r['late'] and r['b'] == 0]
    x, z, a, masses = [], [], [], []
    b = seal['parameter']['layer']
    layer, head = seal['head']['layer'], seal['head']['head']
    for name, digest in sorted(collection['shards'].items()):
        assert sha(SOURCE / name) == digest
        with np.load(SOURCE / name) as data:
            x.append(bits_to_float(data[seal['row_key']]))
            z.append(bits_to_float(data[f'z{b}']))
            a.append(bits_to_float(data[f'a{b}']))
            masses.append(data[f'mass{layer}'][:, -1, head, :4])
    x, z, a = [np.concatenate(chunk)[keep].astype(np.float64) for chunk in (x, z, a)]
    with np.load(SOURCE / 'readout-models.npz') as models:
        idx = models['row_idx']
        pred = ((x[:, idx] - models['row_mu']) / models['row_sd']) @ models['row_w'] + models['row_ym']
    truth = np.array([[rows[i]['r0'], rows[i]['r1']] for i in keep])
    with np.load(SOURCE / 'confirm-predictions.npz') as frozen:
        assert np.array_equal(truth, frozen['truth'])
        assert np.array_equal(np.array([rows[i]['id'] for i in keep]), frozen['ids'])
        error = float(np.abs(pred - frozen['row_scores']).max())
        assert error < 1e-10
    correct = (pred >= .5) == truth
    encoding = dict(bit0=float(correct[:, 0].mean()), bit1=float(correct[:, 1].mean()), joint=float(correct.all(1).mean()))
    assert encoding == report['encoding']
    errors = {}
    with np.load(SOURCE / 'native-parameters.npz') as params:
        target = a[:, params['units']]
        for name, key in [('native32', 'input_idx'), ('random32', 'random_idx')]:
            idx = params[key]
            center = params['input_mean'].astype(np.float64)
            gate, up = params['gate'].astype(np.float64), params['up'].astype(np.float64)
            g = (z[:, idx] - center[idx]) @ gate[:, idx].T + center @ gate.T
            u = (z[:, idx] - center[idx]) @ up[:, idx].T + center @ up.T
            estimate = g / (1 + np.exp(-g)) * u
            errors[name] = float(np.square(estimate - target).mean())
        errors['mean'] = float(np.square(target - params['mean_a']).mean())
    for key, value in errors.items():
        assert np.isclose(value, report['parameter'][key], rtol=1e-5, atol=1e-8)
    mass = np.concatenate(masses)
    predicted = mass.argmax(-1)
    gold = np.array([2 * r['b'] + (r['r0'], r['r1'])[r['b']] for r in rows])
    late = np.array([bool(r['late']) for r in rows])
    routing = dict(late_correct=int((predicted[late] == gold[late]).sum()), late_n=int(late.sum()),
                   early_correct=int((predicted[~late] == gold[~late]).sum()), early_n=int((~late).sum()))
    assert routing['late_correct'] / routing['late_n'] == report['routing']['chosen']['late_slot_acc']
    assert routing['early_correct'] / routing['early_n'] == report['routing']['chosen']['early_slot_acc']
    result = dict(schema='clt_frozen_prediction_arithmetic_audit.v1', kind='real',
                  provenance=seal['provenance'], candidate_sha256=sha(SOURCE / 'candidate.json'),
                  analysis_sha256=sha(SOURCE / 'confirm-analysis.json'), auditor_sha256=sha(Path(__file__)),
                  prediction_max_abs_difference=error, encoding=encoding, parameter_mse_fp64=errors,
                  measured_routing=routing, all_arithmetic_checks_pass=True,
                  causal_tested=False, mechanism_accepted=False,
                  scope='frozen-model arithmetic reproduction from saved native confirmation states; no new model forward or causal certification')
    target = OUTPUT / 'frozen-prediction-arithmetic-audit.json'
    check_write(target, 1 << 20, operation='CLT frozen prediction audit')
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
