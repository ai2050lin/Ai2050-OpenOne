"""All-coordinate numerical admission of an execution scheduler, not a law."""
import argparse
from phase2745_rdc_construction_contract import freeze
from rdc_construction_stream import Engine, original_reference, compare
from rdc_construction_common import *


def main(key):
    import torch
    protocol, material = freeze()
    out = BASE / 'pilot' / key
    if (out / 'result.json').exists():
        prior = read(out / 'result.json')
        assert prior['all_passed']
        same_common = key != 'glm4' or prior['common_source']['sha256'] == sha(Path(__file__).with_name('rdc_construction_common.py'))
        if same_common and prior['scheduler_source']['sha256'] == sha(Path(__file__).with_name('rdc_construction_stream.py')):
            return
        save(out / 'history' / ('result_' + prior['scheduler_source']['sha256'][:16] + '_' + prior['common_source']['sha256'][:16] + '.json'), prior)
    start = time.monotonic()
    source_version = snapshot(__file__)
    scheduler_version = snapshot(Path(__file__).with_name('rdc_construction_stream.py'))
    model = None
    try:
        guard(300 * 1024**2)
        model, tok = load(key, out)
        rows = material['models'][key]['rows']
        probes = material['models'][key]['probes']
        engine = Engine(model, key, probes)
        checks = []
        with torch.inference_mode():
            for index in [None, 0, 66, 129, 195, 258]:
                ids = None if index is None else rows[index]['prompt_ids']
                tick = time.monotonic()
                ref = original_reference(model, ids, probes, key)
                native_seconds = time.monotonic() - tick
                save(out / 'progress.json', {'timestamp': stamp(), 'checks': checks,
                     'currently_verified_reference': index, 'native_seconds': native_seconds})
                print('CONSTRUCTION_NATIVE_REFERENCE', key, index, round(native_seconds, 3), flush=True)
                tick = time.monotonic()
                streamed = engine.run(ids, standalone=index is None)
                stream_seconds = time.monotonic() - tick
                check = compare(ref, streamed)
                check.update({'row_index': index, 'sample_id': None if index is None else rows[index]['sample_id'],
                    'prefix_tokens': 0 if ids is None else len(ids), 'native_seconds': native_seconds,
                    'streamed_seconds': stream_seconds, 'reference_identities': {k: identity(v) for k, v in ref.items() if v is not None},
                    'streamed_identities': {k: identity(streamed[k]) for k in ref if ref[k] is not None},
                    'native_attention_max_error': max(r['max_abs_error'] for r in streamed['attention_checks'])})
                if key == 'qwen4' and index is not None:
                    with np.load(OLD / 'identifiability/relations/native/fields' / (rows[index]['sample_id'] + '.npz')) as old:
                        check['previous2744_postnorm_bit_equal'] = bool(np.array_equal(ref['postnorm'], old['postnorm']))
                        check['previous2744_prefix_bit_equal'] = bool(np.array_equal(ref['prefix_layers'], old['prefix_layers']))
                    assert check['previous2744_postnorm_bit_equal'] and check['previous2744_prefix_bit_equal']
                checks.append(check)
                save(out / 'progress.json', {'timestamp': stamp(), 'checks': checks})
                print('CONSTRUCTION_PILOT', key, index, 'all_bit_equal', check['all_bit_equal'],
                      'native', round(native_seconds, 3), 'streamed', round(stream_seconds, 3), flush=True)
                assert check['all_bit_equal'], ('Scheduler is not admitted; preserve exact-shape native execution', key, index, check)
                del ref, streamed
                gc.collect()
                torch.cuda.empty_cache()
        result = {'timestamp': stamp(), 'source': source_version, 'scheduler_source': scheduler_version,
            'common_source': snapshot(Path(__file__).with_name('rdc_construction_common.py')),
            'all_passed': True, 'model': key, 'checks': checks, 'staged_original_blocks': sorted(engine.copied_blocks),
            'seconds': time.monotonic() - start, 'peak_cuda_allocated': torch.cuda.max_memory_allocated(),
            'scope': 'Complete per-query last-position coordinates across every native boundary match original full-model cached calls. This only qualifies scheduling equivalence on these fixed fixtures; no mechanism or universal numeric equivalence claim.'}
        save(out / 'result.json', result)
        ledger('construction_scheduler_pilot_' + key, result['seconds'])
        print('CONSTRUCTION_PILOT_DONE', key, result['seconds'], flush=True)
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
