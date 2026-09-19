"""Serial three native models and six actual BF16 learning endpoints, full caps."""
import argparse
from rdc_formation_common import *
from rdc_formation_history import freeze, HiddenObserver, generate_batch, record, summarize
from rdc_formation_direction import ExactDirection
from rdc_formation_readout import CUDA_FORMATION, checked_arrays


def native_admission(model, key, observer):
    import torch
    if key == 'qwen4': return []
    rows = gzread(BASE/'material.json.gz')['models'][key]['rows']
    checks = []
    with torch.inference_mode():
        observer.active = True
        for index in [0, 66, 129, 195, 258, 319]:
            row = rows[index]
            value = model.model(input_ids=torch.tensor([row['prompt_ids']], device='cuda'), use_cache=True)
            with np.load(BASE/'capture'/key/'fields'/(row['sample_id']+'.npz')) as z:
                equal = np.array_equal(np.stack([observer.state[i][0] for i in range(len(model.model.layers)+1)]), z['prefix_layers'])
                post_equal = np.array_equal(bits(value.last_hidden_state[0, -1]), z['prefix_postnorm'])
            assert equal and post_equal, (key, index)
            checks.append({'row_index': index, 'all_hidden_boundaries_bit_equal': equal, 'postnorm_bit_equal': post_equal})
            del value
        observer.active = False; observer.state.clear()
    return checks


def main(key, variant='native', pilot=False):
    import torch
    from rdc_native_tail import cuda_singleton
    from phase2744_rdc_query_identifiability import language_checks
    cuda_singleton(CUDA_FORMATION)
    protocol, material = freeze()
    rows = material[key]
    assert key == 'qwen4' or variant == 'native'
    if variant != 'native': assert read(OUT/'training/result.json')['all_passed']
    folder = OUT/'own_history'/key/variant
    result_path = folder/('pilot.json' if pilot else 'result.json')
    if result_path.exists(): return
    if not pilot: assert read(OUT/'own_history'/key/'native/pilot.json')['all_passed']
    start = time.monotonic(); model = manager = observer = None
    source = snapshot(__file__)
    try:
        model, tok = load(key, folder/('pilot_load' if pilot else 'main_load'))
        observer = HiddenObserver(model)
        admission = native_admission(model, key, observer)
        original_eval = None
        if key == 'qwen4':
            if variant == 'native':
                original_eval = checked_arrays(read(OUT/'training/baseline/commits/native.json'))
                parameter_audit = {'original_native_parameters': True}
            else:
                trained = read(OUT/'training'/variant/'result.json')
                manager = ExactDirection(model)
                manager.activate({'run': variant, 'transform': 'identity'})
                manager.precision('native_BF16')
                actual_norm, per_matrix = manager.measure(1., 'native_BF16')
                assert abs(actual_norm-trained['deployed_BF16_delta_L2']) < 1e-10
                with torch.no_grad():
                    named = dict(manager.target.named_parameters())
                    for name, value in manager.values(1., 'native_BF16'):
                        named[name].copy_(value); assert torch.equal(named[name], value)
                parameter_audit = {'source_training_result': str((OUT/'training'/variant/'result.json').relative_to(ROOT)),
                    'actual_BF16_parameter_norm': actual_norm, 'per_matrix': per_matrix, 'direction': manager.audit,
                    'no_calibration_or_label_in_decode': True}
                original_eval = checked_arrays(trained['deployment'])
            panel_ids = read(OUT/'training/evaluation_rows.json')['panel']
            panel_index = {sid: i for i, sid in enumerate(panel_ids)}
        else: parameter_audit = {'original_native_parameters': True}
        save(folder/('pilot_admission.json' if pilot else 'admission.json'), {'timestamp': stamp(),
            'all_passed': True, 'native_checks': admission, 'parameter_deployment': parameter_audit, 'parser_checks': language_checks()})
        batches = [rows[:8], rows[192:200]] if pilot else [rows[i:i+8] for i in range(0, len(rows), 8)]
        records = []; first_checks = []; batch_times = []
        for batch in batches:
            paths = [folder/('pilot_records' if pilot else 'records')/(r['sample_id']+'.json') for r in batch]
            previous = [read(path) if path.exists() else None for path in paths]
            if all(r is not None for r in previous):
                for r in previous: checked_arrays(r['field'])
                records.extend(previous); continue
            tick = time.monotonic()
            packets, stop = generate_batch(model, tok, batch, observer)
            elapsed = time.monotonic()-tick; batch_times.append(elapsed)
            for row, packet, path, prior in zip(batch, packets, paths, previous):
                if original_eval is not None:
                    index = panel_index[row['sample_id']]
                    assert np.array_equal(packet['first_B1_postnorm_BF16'], original_eval['postnorm_BF16'][index]), (variant, row['sample_id'])
                    assert int(packet['first_B1_statistics'][0]) == original_eval['argmax'][index]
                    assert abs(packet['first_B1_statistics'][2]-original_eval['NLL'][index]) < 1e-10
                first_checks.append({'sample_id': row['sample_id'], 'training_B1_endpoint_bit_equal': original_eval is not None})
                if prior is not None:
                    stored = checked_arrays(prior['field'])
                    assert set(stored) == set(packet) and all(np.array_equal(stored[k], value, equal_nan=True) for k, value in packet.items())
                    records.append(prior); continue
                r = record(row, packet, tok, stop, variant)
                field = commit_array('own_history/'+key+'/'+variant+('/pilot' if pilot else ''), row['sample_id'], **packet)
                r.update(timestamp=stamp(), source=source, field=field, batch_ids=[r['sample_id'] for r in batch], allocated_batch_seconds=elapsed/len(batch))
                if variant != 'native':
                    native_path = OUT/'own_history'/key/'native/records'/(row['sample_id']+'.json')
                    native = read(native_path)
                    a, b = r['generated_ids'], native['generated_ids']
                    r['first_divergence_from_native'] = next((i for i in range(max(len(a), len(b))) if i >= len(a) or i >= len(b) or a[i] != b[i]), None)
                if not pilot:
                    earlier = folder/'pilot_records'/(row['sample_id']+'.json')
                    if earlier.exists():
                        pilot_record = read(earlier)
                        assert pilot_record['generated_ids'] == r['generated_ids']
                        pilot_packet = checked_arrays(pilot_record['field'])
                        assert set(pilot_packet) == set(packet) and all(np.array_equal(pilot_packet[k], v, equal_nan=True) for k, v in packet.items())
                        r['pilot_full_array_replay_exact'] = True
                save(path, r); records.append(r)
            save(folder/'progress.json', {'timestamp': stamp(), 'pilot': pilot, 'expressions': len(records), 'total': 16 if pilot else 512,
                'generated_tokens': sum(len(r['generated_ids']) for r in records), 'seconds': time.monotonic()-start})
            print('FORMATION_OWN_HISTORY', key, variant, pilot, len(records), 'tokens', sum(len(r['generated_ids']) for r in records), round(time.monotonic()-start, 2), flush=True)
            storage_guard()
        result = {'timestamp': stamp(), 'source': source, 'all_passed': True, 'model': key, 'variant': variant,
            'pilot': pilot, 'trajectories': len(records), 'summary': summarize(records), 'numerical_admission': admission,
            'training_B1_checks': first_checks, 'parameter_audit': parameter_audit, 'batch_seconds': batch_times,
            'history_engine': snapshot(Path(__file__).with_name('rdc_formation_history.py')),
            'missing_target_storage': 'A first_target_available boolean masks the unscored zero placeholder. Every natural trajectory remains present; no artificial target score is reported.',
            'actual_generated_tokens': sum(len(r['generated_ids']) for r in records),
            'peak_CUDA_allocated': torch.cuda.max_memory_allocated(), 'seconds': time.monotonic()-start,
            'scope': 'Actual native own history at a declared finite cap; no unique gold natural continuation or unbounded-composition claim. B1/B8 execution-shape effects separately measured.'}
        if pilot:
            result['measured_batch_cost_projection_64_batches_seconds'] = float(np.mean(batch_times)*64) if batch_times else None
            result['projection_boundary'] = 'Pilot lengths/completion rates and parameter residency may not represent all materials. Projection is not an elapsed-time limit.'
        save(result_path, result); ledger('phase2747_own_history_'+key+'_'+variant+('_pilot' if pilot else ''), result['seconds'])
        print('FORMATION_OWN_HISTORY_DONE', key, variant, pilot, result['seconds'], flush=True)
    except Exception as exc:
        failure(folder, start, exc); raise
    finally:
        if observer is not None: observer.close()
        if manager is not None: manager.close()
        if model is not None: del model
        del observer, manager
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('model', choices=list(MODELS))
    parser.add_argument('--variant', default='native'); parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args(); main(args.model, args.variant, args.pilot)
