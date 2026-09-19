"""Numerical admission for full-scalar norm matching and own-history scoring."""
from phase2745_rdc_construction_norm import freeze_norm, calibration, relation_fields, own_history
from phase2745_rdc_construction_contract import freeze
from rdc_construction_direction import Direction, variants
from rdc_construction_common import *


def main():
    import torch
    from phase2742_rdc_query_formation import evaluate
    out = BASE / 'norm_pilot'
    names = ['phase2745_rdc_construction_norm.py', 'rdc_construction_direction.py']
    versions = {n: snapshot(Path(__file__).with_name(n)) for n in names}
    if (out / 'result.json').exists():
        old = read(out / 'result.json')
        if all(old['sources'][n]['sha256'] == versions[n]['sha256'] for n in names):
            assert old['all_passed']
            return
        save(out / 'history' / ('result_'+old['sources'][names[0]]['sha256'][:16]+'.json'), old)
    start, model, manager = time.monotonic(), None, None
    try:
        freeze_norm()
        _, material = freeze()
        model, tok = load('qwen4', out)
        rows = material['models']['qwen4']['rows']
        chosen = [r for begin in [0, 64, 128, 192, 256] for r in rows[begin:begin+8]]
        controls = gzread(OLD / 'formation/material.json.gz')['panel'][:4]
        with np.load(OLD / 'formation/native_baseline.npz') as z:
            expected = z['loss'][:4]
        with torch.inference_mode():
            assert np.array_equal(evaluate(model, controls)['loss'], expected)
            native_arrays, native_records = relation_fields(model, chosen)
            for i, row in enumerate(chosen):
                with np.load(OLD / 'identifiability/relations/native/fields' / (row['sample_id']+'.npz')) as z:
                    for n in native_arrays:
                        ref = z['postnorm_original_prompt'] if n == 'postnorm' else (
                            z['prefix_layers'][int(n[1:])] if n.startswith('H') else z[n])
                        assert np.array_equal(native_arrays[n][i], ref), (row['sample_id'], n)
                old = read(OLD / 'identifiability/relations/native/commits' / (row['sample_id']+'.json'))['actual_current']
                for a, b in [('full_vocabulary_NLL', 'full_vocab_NLL'),
                             ('conditional_answer_NLL', 'binary_conditional_NLL'),
                             ('conditional_yes_probability', 'binary_conditional_yes_probability')]:
                    assert abs(native_records[i][a]-old[b]) < 1e-12, (row['sample_id'], a)
            native_behavior_arrays, native_behavior = own_history(model, tok, chosen, native_arrays, native_records)
            for i, row in enumerate(chosen):
                old = read(OLD / 'identifiability/behavior/native/commits' / (row['sample_id']+'.json'))
                assert old['generated_ids'] == native_behavior[i]['generated_ids']
                with np.load(OLD / 'identifiability/behavior/native/fields' / (row['sample_id']+'.npz')) as z:
                    assert np.array_equal(z['first_and_final_postnorm'], native_behavior_arrays['first_final_postnorm'][i])
            manager = Direction(model)
            selected = [v for v in variants() if v['kind'] == 'matched' and v['radius'] == .10
                        and (v['direction'].endswith('2742'))]
            results = []
            for variant in selected:
                tick = time.monotonic()
                match = manager.match(variant)
                losses = evaluate(model, controls)['loss']
                assert np.isfinite(losses).all() and match['relative_norm_error'] <= .001
                # Same complete scalar direction, not a small TopK stand-in.
                assert match['all_scalars_used'] == 74711040
                manager.restore()
                reset = evaluate(model, controls)['loss']
                assert np.array_equal(reset, expected)
                results.append({'variant': variant, 'matching': match, 'pilot_losses': losses.tolist(),
                                'restored_native_bit_equal': True, 'seconds': time.monotonic()-tick})
                print('CONSTRUCTION_NORM_PILOT', variant['name'], match['actual_BF16_norm'], flush=True)
            result = {'timestamp': stamp(), 'source': snapshot(__file__), 'sources': versions,
                'all_passed': True, 'variants': results, 'native_field_and_score_fixtures': len(chosen),
                'native_B8_token_and_endpoint_fixtures': len(chosen),
                'source_ids': [r['sample_id'] for r in chosen], 'seconds': time.monotonic()-start,
                'scope': 'Full74711040scalar matching and restore, original all-coordinate fields and exactly matched B8 own-history fixture regression. No pilot language improvement claim.'}
            save(out / 'result.json', result)
            ledger('construction_norm_matching_pilot', result['seconds'])
            print('CONSTRUCTION_NORM_PILOT_DONE', result['seconds'], flush=True)
    except Exception as exc:
        failure(out, start, exc)
        raise
    finally:
        if manager is not None:
            manager.restore()
        del manager, model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
