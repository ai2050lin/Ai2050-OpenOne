"""Native full-prefix fixtures and finite original-parameter deployments."""
import argparse
from rdc_formation_propagation import *
from rdc_formation_readout import CUDA_FORMATION, checked_arrays
from rdc_formation_direction import ExactDirection
from phase2747_rdc_training import FormationObserver


def main(pilot=False):
    import torch
    from rdc_native_tail import cuda_singleton
    cuda_singleton(CUDA_FORMATION|{'phase2747_rdc_parameter_capture.py'})
    freeze()
    folder = PROP/('native_pilot' if pilot else 'native')
    result_path = folder/'result.json'
    if result_path.exists(): return
    if not pilot: assert read(PROP/'native_pilot/result.json')['all_passed']
    assert read(OUT/'training/result.json')['all_passed']
    rows = material(pilot); directions = runs()[:2] if pilot else runs()
    start = time.monotonic(); model = observer = manager = None; handles = []
    try:
        model, tok = load('qwen4', folder/'load'); observer = FormationObserver(model)
        prefix = {}
        def capture_prefix(name, value):
            prefix[name] = bits(value[0])
        layer = model.model.layers[16]
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda m, a: capture_prefix('prefix_residual_BF16', a[0])))
        handles.append(layer.post_attention_layernorm.register_forward_hook(lambda m, a, o: capture_prefix('prefix_MLP_input_BF16', o)))
        handles.append(model.model.rotary_emb.register_forward_hook(lambda m, a, o: (capture_prefix('cos_BF16', o[0]), capture_prefix('sin_BF16', o[1])) and None))
        # The rotary hook's return is explicitly None; it records, never replaces.
        def capture(row, full=True):
            observer.active = True; prefix.clear()
            value = model.model(input_ids=torch.tensor([row['ids']], device='cuda'), use_cache=False)
            h = value.last_hidden_state[0, -1]; lp = model.lm_head(h[None]).float()[0].double().log_softmax(-1)
            result = observer.collect(); observer.active = False
            result = {k+'_BF16' if k != 'postnorm_BF16' else k: bf16bits(v) for k, v in result.items() if full or k == 'hidden'}
            result.update(postnorm_BF16=bits(h), statistics=np.array([float(-lp[row['target']]), int(lp.argmax()), float(-(lp.exp()*lp).sum())]))
            return result, {k: v.copy() for k, v in prefix.items()}
        records = []; baseline = {}
        with torch.no_grad():
            for row in rows:
                path = folder/'baseline'/ (row['sample_id']+'.json')
                if path.exists():
                    r = read(path); checked_arrays(r['field']); baseline[row['sample_id']] = r; continue
                packet, before = capture(row)
                packet.update(before)
                receipt = commit_array('parameter_propagation/'+folder.name+'/baseline', row['sample_id'], **packet)
                record = {'timestamp': stamp(), 'sample_id': row['sample_id'], 'field': receipt,
                    'full_prefix_tokens': len(row['ids']), 'all_passed': True}
                save(path, record); baseline[row['sample_id']] = record
            manager = ExactDirection(model)
            for run in directions:
                manager.activate({'run': run, 'transform': 'identity'})
                manager.precision('native_BF16')
                named = dict(manager.target.named_parameters())
                for scale in SCALES:
                    for name, value in manager.values(scale, 'native_BF16'): named[name].copy_(value)
                    norm, per = manager.measure(scale, 'native_BF16')
                    for row in rows:
                        tag = run+'/s'+str(scale).replace('.', 'p')
                        path = folder/'records'/tag/(row['sample_id']+'.json')
                        if path.exists():
                            r = read(path); checked_arrays(r['field']); records.append(r); continue
                        base = checked_arrays(baseline[row['sample_id']]['field'])
                        packet, before = capture(row, scale == 1.)
                        assert all(np.array_equal(before[k], base[k]) for k in before), ('PreMLP16prefix unexpectedly changed', run, scale, row['sample_id'])
                        assert np.array_equal(packet['hidden_BF16'][:17], base['hidden_BF16'][:17])
                        receipt = commit_array('parameter_propagation/'+folder.name+'/'+tag, row['sample_id'], **packet)
                        record = {'timestamp': stamp(), 'sample_id': row['sample_id'], 'run': run, 'scale': scale,
                            'all_passed': True, 'all_prefix_preMLP_input_bit_invariant': True, 'earlier_H0_to_H16_bit_invariant': True,
                            'actual_BF16_parameter_delta_L2': norm, 'per_parameter_L2': per, 'field': receipt,
                            'full_unit_fields_collected': scale == 1.}
                        save(path, record); records.append(record)
                    print('FORMATION_PARAMETER_NATIVE', pilot, run, scale, len(records), round(time.monotonic()-start, 2), flush=True)
            manager.restore()
            reset, _ = capture(rows[0])
            base = checked_arrays(baseline[rows[0]['sample_id']]['field'])
            assert all(np.array_equal(v, base[k]) for k, v in reset.items())
        result = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True, 'pilot': pilot,
            'rows': len(rows), 'directions': directions, 'scales': SCALES, 'records': len(records),
            'baseline': baseline, 'full_prefix_input_invariance_checked_every_finite_expression': True,
            'original_native_restored_bit_equal': True, 'seconds': time.monotonic()-start,
            'peak_CUDA_bytes': torch.cuda.max_memory_allocated(),
            'scope': 'Actual native finite parameter effects including changed upper-prefix KV. No transported hidden-state difference enters original-model inference.'}
        save(result_path, result); ledger('phase2747_parameter_native'+('_pilot' if pilot else ''), result['seconds'])
    except Exception as exc:
        failure(folder, start, exc); raise
    finally:
        for handle in handles: handle.remove()
        if observer is not None: observer.close()
        if manager is not None: manager.close()
        if model is not None: del model
        del manager, observer
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--pilot', action='store_true'); main(parser.parse_args().pilot)
