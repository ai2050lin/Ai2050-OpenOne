"""Verify source interleaving against previously byte-verified native fixtures."""
import argparse
import faulthandler
from phase2745_rdc_construction_contract import freeze
from rdc_construction_batch import BatchedEngine
from rdc_construction_common import *


def main(key):
    import torch
    faulthandler.enable()
    _, material = freeze()
    out = BASE / 'batch_pilot' / key
    if (out / 'result.json').exists():
        prior = read(out / 'result.json')
        assert prior['all_passed']
        required = ['rdc_construction_batch.py', 'rdc_construction_stream.py'] + (['rdc_construction_common.py'] if key == 'glm4' else [])
        if all(prior['sources'][n]['sha256'] == sha(Path(__file__).with_name(n)) for n in required):
            return
        save(out / 'history' / ('result_' + prior['sources']['rdc_construction_batch.py']['sha256'][:16] + '.json'), prior)
    old = read(BASE / 'pilot' / key / 'result.json')
    assert old['all_passed']
    if key == 'glm4':
        assert old['common_source']['sha256'] == sha(Path(__file__).with_name('rdc_construction_common.py'))
    start, model = time.monotonic(), None
    versions = {p.name: snapshot(p) for p in [Path(__file__), Path(__file__).with_name('rdc_construction_batch.py'),
                Path(__file__).with_name('rdc_construction_stream.py'), Path(__file__).with_name('rdc_construction_common.py')]}
    assert versions['rdc_construction_stream.py']['sha256'] == old['scheduler_source']['sha256']
    indices = [0, 66, 129, 195, 258, 0, 192, 193, 194, 196, 197, 198, 199, 256, 257, 259]
    try:
        model, tok = load(key, out)
        print('CONSTRUCTION_BATCH_MODEL_READY', key, flush=True)
        rows, probes = material['models'][key]['rows'], material['models'][key]['probes']
        class ProgressEngine(BatchedEngine):
            def advance(self, s, index, layer, standalone, prototypes, feature_fn):
                # Logging only; all model operations remain in the verified engine.
                if index != getattr(self, '_logged_index', None):
                    print('CONSTRUCTION_BATCH_LAYER', key, index, flush=True)
                    self._logged_index = index
                return super().advance(s, index, layer, standalone, prototypes, feature_fn)
        engine = ProgressEngine(model, key, probes)
        with torch.inference_mode():
            tick = time.monotonic()
            actual = engine.run_many([rows[i]['prompt_ids'] for i in indices])
            elapsed = time.monotonic()-tick
        checks = []
        expected = {r['row_index']: r for r in old['checks']}
        for j, index in enumerate(indices[:5]):
            refs = expected[index]['reference_identities']
            hashes = {k: identity(actual[j][k]) for k in refs}
            check = {'row_index': index, 'sample_id': rows[index]['sample_id'],
                     'all_native_array_identities_equal': hashes == refs, 'actual_identities': hashes}
            assert check['all_native_array_identities_equal'], check
            checks.append(check)
        duplicate = all(np.array_equal(actual[0][k], actual[5][k]) for k in
                        ['query_all_states', 'postnorm', 'prefix_layers', 'prefix_postnorm'])
        assert duplicate, 'Independent identical-prefix branches contaminated each other'
        result = {'timestamp': stamp(), 'sources': versions, 'all_passed': True, 'model': key,
            'source_indices': indices, 'source_ids': [rows[i]['sample_id'] for i in indices], 'sources_interleaved': 16,
            'checks_against_standard_native_reference': checks, 'independent_duplicate_bit_equal': duplicate,
            'attention_max_error': max(c['max_abs_error'] for a in actual for c in a['attention_checks']),
            'capture_seconds_for_16sources': elapsed, 'mean_seconds_per_source': elapsed/16,
            'seconds': time.monotonic()-start, 'peak_cuda_allocated': torch.cuda.max_memory_allocated(),
            'scope': 'Five complete per-query/per-layer native-array hashes from independent standard references verified in a16source-interleaved schedule. Duplicate branch also compared elementwise. No source stacking, query padding, dtype change, parameter learning or mechanism-extraction claim.'}
        save(out / 'result.json', result)
        ledger('construction_source_interleaving_pilot_' + key, result['seconds'])
        print('CONSTRUCTION_BATCH_PILOT_DONE', key, 'native_equal', True, '16sources_seconds', elapsed, flush=True)
    except Exception as exc:
        failure(out, start, exc)
        raise
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['qwen4', 'qwen14', 'glm4'])
    main(parser.parse_args().model)
