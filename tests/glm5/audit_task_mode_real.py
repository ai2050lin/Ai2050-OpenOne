"""Verify the real trace manifests and recompute all task-layer comparisons on CPU."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.dist_protocol.mechanism_trace import TraceStore, unbits


def main():
    out = ROOT / 'tests/glm5/result/task_mode_real'
    store = TraceStore()
    before = json.loads((out / 'baseline-browser.json').read_text(encoding='utf-8'))
    after = json.loads((out / 'real-browser.json').read_text(encoding='utf-8'))
    assert before['passed'] and after['passed'] and after['unloaded']
    assert len(before['runs']) == len(after['runs']) == 6
    audit = {'scope': 'engineering smoke test; repeated visible templates', 'inputs': [], 'pairs': {}, 'retained_native_bytes': 0}
    vectors = {}
    for first, final in zip(before['runs'], after['runs']):
        assert first['trace'] != final['trace']
        assert (first['kind'], first['condition'], first['text'], first['snapshot']) == (
            final['kind'], final['condition'], final['text'], final['snapshot'])
        repetitions = []
        for run in (first, final):
            manifest, folder = store.manifest(run['trace'])
            assert manifest['kind'] == 'real' and manifest['split'] == 'unregistered_live'
            assert manifest['text'] == run['text'] and manifest['model_snapshot'] == run['snapshot']
            assert manifest['n_tokens'] == run['tokens'] and manifest['readout_positions'][-1] == run['tokens'] - 1
            for record in manifest['files']:
                store.native_path(run['trace'], record['path'])
                audit['retained_native_bytes'] += record['bytes']
            audit['retained_native_bytes'] += (folder / 'manifest.json').stat().st_size
            native = np.load(store.native_path(run['trace'], 'hidden_full.npy'), mmap_mode='r')
            assert native.shape == (37, run['tokens'], 2560) and native.dtype == np.uint16
            final_position = np.array(native[:, -1, :])
            selected = np.load(store.native_path(run['trace'], 'hidden_readouts.npy'), mmap_mode='r')
            np.testing.assert_array_equal(final_position, selected[:, -1, :])
            assert np.isfinite(unbits(final_position)).all()
            repetitions.append(final_position)
        key = (final['kind'], final['condition'])
        vectors[key] = unbits(repetitions[-1]).astype(np.float64)
        audit['inputs'].append({'kind': key[0], 'condition': key[1], 'trace': final['trace'],
                                'baseline_trace': first['trace'], 'n_chars': len(final['text']), 'n_tokens': final['tokens'],
                                'top': final['top'][0], 'baseline_top': first['top'][0],
                                'replicated_final_position_bf16_bits': bool(np.array_equal(*repetitions))})
    for kind in ('binding', 'negation', 'order'):
        a, b = vectors[kind, 'A'], vectors[kind, 'B']
        a_norm, b_norm = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        relative = np.linalg.norm(b - a, axis=1) / a_norm
        cosine = np.clip(np.einsum('ld,ld->l', a, b) / (a_norm * b_norm), -1, 1)
        np.testing.assert_allclose(relative[12], after[f'{kind}_L12']['relativeL2'], atol=1e-12, rtol=0)
        assert relative[0] == 0
        audit['pairs'][kind] = [{'layer': i, 'relativeL2': float(relative[i]), 'cosine': float(cosine[i])} for i in range(37)]
    (out / 'native-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'verified_traces':12,'pairs':3,'coordinates_per_condition':37*2560,
                      'bit_exact_repetitions':sum(row['replicated_final_position_bf16_bits'] for row in audit['inputs']),
                      'retained_native_bytes':audit['retained_native_bytes']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
