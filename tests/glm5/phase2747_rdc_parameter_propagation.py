"""Full-prefix variational chain, finite changes, and complete-parameter adjoint."""
import argparse
from rdc_formation_propagation import *
from rdc_formation_readout import CUDA_FORMATION, checked_arrays, head_parameter_name
from rdc_formation_direction import exact_direction
from rdc_native_tail import loaded_block, checkpoint_tensor, final_norm, config, cuda_singleton

FIELD_NAMES = ['hidden', 'Q', 'gate', 'up', 'product', 'MLP_input', 'MLP_write']


def persist(category, name, **arrays):
    path = OUT/category/'commits'/(name+'.json')
    if path.exists():
        receipt = read(path); previous = checked_arrays(receipt)
        assert set(previous) == set(arrays) and all(np.array_equal(previous[k], value) for k, value in arrays.items()), ('Resume changed a committed field', name)
        return receipt
    return commit_array(category, name, **arrays)


def layer_packet(base, tangents, finite, native, ndirection):
    arrays = {}
    for j, name in enumerate(FIELD_NAMES):
        arrays['base_'+name] = np.stack([v[j] for v in base])
        arrays['tangent_'+name] = np.stack([[v[j] for v in row] for row in tangents])
        arrays['scale1_finite_'+name] = np.stack([[row[i*3+2][j] for i in range(ndirection)] for row in finite])
        arrays['native_base_'+name] = np.stack([v[j] for v in native])
    # Every finite scale keeps every residual coordinate; unit fields at .1/.3
    # are used in full-coordinate metrics but are recomputable, not retained.
    arrays['all_scale_finite_hidden'] = np.stack([[v[0] for v in row] for row in finite])
    error = []
    for b, tt, ff in zip(base, tangents, finite):
        per_run = []
        for ri in range(ndirection):
            per_scale = []
            for si, scale in enumerate(SCALES):
                per_control = []
                for control in range(2):
                    vals = []
                    for j in range(len(FIELD_NAMES)):
                        observed = ff[ri*3+si][j].astype(float)-b[j].astype(float)
                        predicted = scale*tt[ri*2+control][j].astype(float)
                        vals.append([float(np.mean(observed**2)), float(np.mean((observed-predicted)**2)), float(np.mean(predicted**2))])
                    per_control.append(vals)
                per_scale.append(per_control)
            per_run.append(per_scale)
        error.append(per_run)
    arrays['full_coordinate_finite_error'] = np.array(error)
    return arrays


def native_last(arrays, block):
    result = [unbits(arrays['hidden_BF16'][block+1]).astype(np.float32)]
    for name in FIELD_NAMES[1:]: result.append(unbits(arrays[name+'_BF16'][block]).astype(np.float32).reshape(-1))
    return result


def prepare(rows, direction_names, native_folder):
    import torch
    originals = tuple(checkpoint_tensor('model.layers.16.mlp.'+n, torch.float32) for n in NAMES)
    original_dict = dict(zip(NAMES, originals))
    directions = []; audit = []
    for name in direction_names:
        exact, receipt = exact_direction(name, original_dict)
        dd = tuple(torch.tensor(exact[n].astype(np.float32), device='cuda') for n in NAMES)
        error = np.sqrt(sum(np.sum((exact[n]-exact[n].astype(np.float32).astype(np.float64))**2) for n in NAMES))
        norm = np.sqrt(sum(np.sum(exact[n]**2) for n in NAMES))
        audit.append({'run': name, 'exact_FP64_direction_L2': float(norm), 'FP32_tangent_rounding_L2': float(error), 'source': receipt})
        directions.append(dd); del exact
    states = []; native = []; base_values = []; tangent_values = []; finite_values = []
    for row in rows:
        receipt = read(native_folder/'baseline'/(row['sample_id']+'.json'))['field']
        arrays, residual, x, mask, cos, sin = causal_inputs(receipt)
        q = tensor(arrays['Q_BF16'][16]).reshape(-1)
        local = mlp_function(x, residual)
        def call(*weights):
            value = local(*weights)
            return value[0], q, value[1], value[2], value[3], x[0, -1], value[4]
        base = call(*originals)
        tt = []; tv = []
        for direction in directions:
            primal, tangent = torch.func.jvp(call, originals, direction)
            assert all(torch.equal(a, b) for a, b in zip(primal, base))
            full = tangent[0].detach().cpu()
            last = torch.zeros_like(full); last[:, -1] = full[:, -1]
            tt += [full, last]
            current = cpu_tuple(tangent); tv += [current, [v.copy() for v in current]]
        ff = []; fv = []
        for direction in directions:
            for scale in SCALES:
                weights = tuple(w+scale*d for w, d in zip(originals, direction))
                finite = call(*weights)
                ff.append(finite[0].detach().cpu()); fv.append(cpu_tuple(finite))
                del weights, finite
        states.append({'row': row, 'initial_receipt': receipt, 'base': base[0].detach().cpu(),
            'tangents': tt, 'initial_tangents': tt.copy(), 'finite': ff, 'base_inputs': [],
            'mask': mask.cpu(), 'cos': cos.cpu(), 'sin': sin.cpu()})
        native.append(native_last(arrays, 16)); base_values.append(cpu_tuple(base)); tangent_values.append(tv); finite_values.append(fv)
        del arrays, residual, x, mask, cos, sin, base, primal, tangent, q
    return originals, directions, states, audit, layer_packet(base_values, tangent_values, finite_values, native, len(directions))


def forward_layers(states, ndirection, category):
    import torch
    receipts = []
    for block in range(17, 36):
        tick = time.monotonic(); layer = loaded_block(block, torch.float32); operator = FullBlock(layer)
        base_values = []; tangent_values = []; finite_values = []; native_values = []
        for state in states:
            h = state['base'].to('cuda'); mask, cos, sin = (state[k].to('cuda') for k in ['mask', 'cos', 'sin'])
            state['base_inputs'].append(state['base'])
            fn = lambda value: operator.call(value, mask, cos, sin)
            base = fn(h); tv = []; new_tangents = []
            for tangent in state['tangents']:
                primal, derivative = torch.func.jvp(fn, (h,), (tangent.to('cuda'),))
                assert all(torch.equal(a, b) for a, b in zip(primal, base)), ('JVP primal changed', block, state['row']['sample_id'])
                new_tangents.append(derivative[0].detach().cpu()); tv.append(cpu_tuple(derivative))
            fv = []; new_finite = []
            for finite in state['finite']:
                value = fn(finite.to('cuda')); new_finite.append(value[0].detach().cpu()); fv.append(cpu_tuple(value))
            state['base'] = base[0].detach().cpu(); state['tangents'] = new_tangents; state['finite'] = new_finite
            base_values.append(cpu_tuple(base)); tangent_values.append(tv); finite_values.append(fv)
            native_values.append(native_last(checked_arrays(state['initial_receipt']), block))
            del h, mask, cos, sin, base, primal, derivative, value
        arrays = layer_packet(base_values, tangent_values, finite_values, native_values, ndirection)
        receipt = persist(category+'/layers', f'block_{block:02d}', **arrays); receipts.append(receipt)
        operator.close(); del layer, operator, arrays, base_values, tangent_values, finite_values, native_values
        gc.collect(); torch.cuda.empty_cache()
        print('FORMATION_PARAMETER_JVP_LAYER', category, block, 'seconds', round(time.monotonic()-tick, 2), flush=True)
        save(PROP/category.split('/')[-1]/'progress.json', {'timestamp': stamp(), 'completed_block': block, 'rows': len(states), 'directions': ndirection})
    return receipts


def endpoint(states, directions, direction_names, category, native_folder):
    import torch
    import torch.nn.functional as F
    weight = checkpoint_tensor('model.norm.weight', torch.float32)
    head = checkpoint_tensor(head_parameter_name(), torch.float32); native_head = head.bfloat16()
    epsilon = config().rms_norm_eps; records = []; adjoints = []; checks = []; fields = []
    for state in states:
        row = state['row']; sid = row['sample_id']; last = state['base'][0, -1].to('cuda')
        norm = lambda h: final_norm(h[None], weight, epsilon)[0]
        base_post = norm(last); base_logits = F.linear(base_post[None], head)[0]
        base_lp = base_logits.double().log_softmax(-1)
        native_arrays = checked_arrays(state['initial_receipt'])
        native_post = torch.tensor(unbits(native_arrays['postnorm_BF16']), device='cuda', dtype=torch.bfloat16)
        native_logits = F.linear(native_post[None], native_head).float()[0]
        native_lp = native_logits.double().log_softmax(-1)
        assert abs(float(-native_lp[row['target']])-native_arrays['statistics'][0]) < 1e-10
        generator = torch.Generator(device='cuda'); generator.manual_seed(int(rank('adjoint/'+sid)[:8], 16))
        covector = torch.randn(head.shape[0], device='cuda', generator=generator, dtype=torch.float32)
        covector /= covector.norm()
        logit_tangents = []; post_tangents = []; dual_dots = []; last_hidden_directions = []
        for tangent in state['tangents']:
            _, dt = torch.func.jvp(norm, (last,), (tangent[0, -1].to('cuda'),))
            dz = F.linear(dt[None], head)[0]
            post_tangents.append(dt.detach().cpu().numpy()); logit_tangents.append(dz)
            dual_dots.append(float((covector.double()*dz.double()).sum()))
            last_hidden_directions.append(float(dz.double().norm()))
        # Compute adjoint of full V readout without retaining a full Jacobian.
        norm_value, pullback = torch.func.vjp(norm, last)
        readout_covector = F.linear(covector[None], head.T)[0]
        incoming = pullback(readout_covector)[0].detach()
        prefix_adjoint = torch.zeros_like(state['base'], device='cuda'); prefix_adjoint[0, -1] = incoming
        adjoints.append(prefix_adjoint.cpu()); del pullback
        state['readout_dual_dots'] = dual_dots; state['logit_tangent_norms'] = last_hidden_directions
        finite_posts = []; finite_metrics = []
        for ri, run in enumerate(direction_names):
            for si, scale in enumerate(SCALES):
                current = norm(state['finite'][ri*3+si][0, -1].to('cuda'))
                zz = F.linear(current[None], head)[0]; finite_lp = zz.double().log_softmax(-1)
                finite_posts.append(current.detach().cpu().numpy())
                actual = checked_arrays(read(native_folder/'records'/run/('s'+str(scale).replace('.', 'p'))/(sid+'.json'))['field'])
                actual_logits = F.linear(torch.tensor(unbits(actual['postnorm_BF16']), device='cuda', dtype=torch.bfloat16)[None], native_head).float()[0]
                actual_lp = actual_logits.double().log_softmax(-1)
                assert abs(float(-actual_lp[row['target']])-actual['statistics'][0]) < 1e-10
                record = {'sample_id': sid, 'source_group': row['source_group'], 'family': row['family'], 'kind': row['kind'],
                    'run': run, 'scale': scale, 'native_baseline_NLL': float(-native_lp[row['target']]),
                    'smooth_baseline_NLL': float(-base_lp[row['target']]), 'native_finite_NLL': float(-actual_lp[row['target']]),
                    'smooth_finite_NLL': float(-finite_lp[row['target']]),
                    'native_finite_to_native_baseline_KL': float((actual_lp.exp()*(actual_lp-native_lp)).sum()),
                    'smooth_finite_to_smooth_baseline_KL': float((finite_lp.exp()*(finite_lp-base_lp)).sum()),
                    'native_finite_to_raw_smooth_finite_KL': float((actual_lp.exp()*(actual_lp-finite_lp)).sum()),
                    'smooth_baseline_to_native_baseline_KL': float((base_lp.exp()*(base_lp-native_lp)).sum())}
                for ci, control in enumerate(['full_prefix', 'last_position_only']):
                    derivative = logit_tangents[ri*2+ci]
                    predicted = base_logits+scale*derivative; plp = predicted.double().log_softmax(-1)
                    centered = native_logits+scale*derivative; clp = centered.double().log_softmax(-1)
                    record[control] = {'smooth_finite_to_linear_KL': float((finite_lp.exp()*(finite_lp-plp)).sum()),
                        'native_finite_to_baseline_centered_linear_KL': float((actual_lp.exp()*(actual_lp-clp)).sum()),
                        'linear_smooth_NLL': float(-plp[row['target']]), 'linear_native_centered_NLL': float(-clp[row['target']]),
                        'all_V_finite_logit_delta_MSE': float((zz.double()-base_logits.double()).square().mean()),
                        'all_V_linear_logit_error_MSE': float((zz.double()-base_logits.double()-scale*derivative.double()).square().mean()),
                        'native_centered_argmax': int(centered.argmax()), 'native_finite_argmax': int(actual_logits.argmax())}
                records.append(record)
                finite_metrics.append([record['full_prefix']['smooth_finite_to_linear_KL'], record['last_position_only']['smooth_finite_to_linear_KL']])
        receipt = persist(category+'/endpoints', sid, baseline_postnorm=base_post.detach().cpu().numpy(),
            tangent_postnorm=np.stack(post_tangents), all_scale_finite_postnorm=np.stack(finite_posts),
            dense_V_covector=covector.detach().cpu().numpy(), dual_dots=np.array(dual_dots),
            logit_tangent_L2=np.array(last_hidden_directions), baseline_native_postnorm_BF16=native_arrays['postnorm_BF16'],
            smooth_linear_KL_full_last=np.array(finite_metrics))
        fields.append(receipt); checks.append({'sample_id': sid, 'native_B1_baseline_and_finite_readouts_reproduced': True})
        del logit_tangents, native_arrays, covector, incoming, prefix_adjoint, native_logits, native_lp, norm_value
    del head, native_head, weight
    gc.collect(); torch.cuda.empty_cache()
    return records, adjoints, fields, checks


def adjoint(states, adjoints, originals, directions, category, pilot):
    import torch
    for block in range(35, 16, -1):
        tick = time.monotonic(); layer = loaded_block(block, torch.float32)
        for i, state in enumerate(states):
            h = state['base_inputs'][block-17].to('cuda')
            mask, cos, sin = (state[k].to('cuda') for k in ['mask', 'cos', 'sin'])
            fn = lambda value: layer(hidden_states=value, attention_mask=mask, position_embeddings=(cos, sin), use_cache=False)
            value, pullback = torch.func.vjp(fn, h)
            back = pullback(adjoints[i].to('cuda'))[0]
            adjoints[i] = back.detach().cpu()
            del value, pullback, back, h, mask, cos, sin
        del layer; gc.collect(); torch.cuda.empty_cache()
        print('FORMATION_PARAMETER_VJP_LAYER', category, block, 'seconds', round(time.monotonic()-tick, 2), flush=True)
    checks = []; full_gradient = None
    for i, state in enumerate(states):
        arrays, residual, x, mask, cos, sin = causal_inputs(state['initial_receipt'])
        incoming = adjoints[i].to('cuda')
        local = mlp_function(x, residual)
        fn = lambda *weights: local(*weights)[0]
        value, pullback = torch.func.vjp(fn, *originals)
        gradient = pullback(incoming)
        assert all(bool(torch.isfinite(v).all()) for v in gradient)
        for ri, direction in enumerate(directions):
            parameter_dot = float(sum((g.double()*d.double()).sum() for g, d in zip(gradient, direction)))
            initial_full = float((incoming.double()*state['initial_tangents'][ri*2].to('cuda').double()).sum())
            initial_last = float((incoming.double()*state['initial_tangents'][ri*2+1].to('cuda').double()).sum())
            for control, measured in [('full_prefix', parameter_dot), ('last_position_only', initial_last)]:
                ci = 0 if control == 'full_prefix' else 1
                observed = state['readout_dual_dots'][ri*2+ci]
                error = abs(observed-measured)
                tolerance = 1e-6+2e-5*state['logit_tangent_norms'][ri*2+ci]
                assert error <= tolerance, ('Fullprefix adjoint failed', state['row']['sample_id'], ri, control, error, tolerance)
                checks.append({'sample_id': state['row']['sample_id'], 'run_index': ri, 'control': control,
                    'dense_V_JVP_pairing': observed, 'reverse_pairing': measured, 'absolute_error': error,
                    'tolerance': tolerance, 'full_initial_state_pairing': initial_full,
                    'complete_parameter_VJP_L2': float(sum(g.double().square().sum() for g in gradient).sqrt())})
        if i == 0 and not pilot:
            full_gradient = persist(category+'/adjoint', 'first_declared_sample_complete_parameter_VJP',
                **{n: g.detach().cpu().numpy() for n, g in zip(NAMES, gradient)})
        state_receipt = persist(category+'/adjoint', state['row']['sample_id'],
            complete_prefix_H17_adjoint=incoming.detach().cpu().numpy()[0])
        state['adjoint_field'] = state_receipt
        del arrays, residual, x, mask, cos, sin, incoming, value, pullback, gradient
    return checks, full_gradient


def main(pilot=False):
    import torch
    cuda_singleton(CUDA_FORMATION|{'phase2747_rdc_parameter_capture.py'})
    freeze(); folder = PROP/('smooth_pilot' if pilot else 'smooth'); finish = folder/'result.json'
    if finish.exists(): return
    native_folder = PROP/('native_pilot' if pilot else 'native')
    assert read(native_folder/'result.json')['all_passed']
    if not pilot: assert read(PROP/'smooth_pilot/result.json')['all_passed']
    rows = material(pilot); direction_names = runs()[:2] if pilot else runs()
    category = 'parameter_propagation/'+folder.name
    start = time.monotonic(); originals = directions = states = None
    try:
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
        originals, directions, states, directions_audit, first = prepare(rows, direction_names, native_folder)
        first_receipt = persist(category+'/layers', 'block_16', **first); del first
        layer_receipts = [first_receipt]+forward_layers(states, len(directions), category)
        endpoint_records, adjoints, endpoint_fields, readout_checks = endpoint(states, directions, direction_names, category, native_folder)
        compressed(folder/'endpoint_records.json.gz', endpoint_records)
        adjoint_checks, parameter_vjp = adjoint(states, adjoints, originals, directions, category, pilot)
        from phase2747_rdc_training_analysis import source_stat
        summary = []
        for run in direction_names:
            for scale in SCALES:
                selected = [r for r in endpoint_records if r['run'] == run and r['scale'] == scale]
                for family in ['all']+sorted({r['family'] for r in selected}):
                    rr = [r for r in selected if family == 'all' or r['family'] == family]
                    summary.append({'run': run, 'scale': scale, 'family': family,
                        'full_prefix_smooth_KL': source_stat([r['full_prefix']['smooth_finite_to_linear_KL'] for r in rr], rr),
                        'last_minus_full_smooth_KL': source_stat([r['last_position_only']['smooth_finite_to_linear_KL']-r['full_prefix']['smooth_finite_to_linear_KL'] for r in rr], rr),
                        'full_prefix_native_centered_KL': source_stat([r['full_prefix']['native_finite_to_baseline_centered_linear_KL'] for r in rr], rr),
                        'last_minus_full_native_centered_KL': source_stat([r['last_position_only']['native_finite_to_baseline_centered_linear_KL']-r['full_prefix']['native_finite_to_baseline_centered_linear_KL'] for r in rr], rr)})
        result = {'timestamp': stamp(), 'source': snapshot(__file__), 'helper_source': snapshot(Path(__file__).with_name('rdc_formation_propagation.py')),
            'all_passed': True, 'pilot': pilot, 'rows': len(rows), 'direction_names': direction_names, 'directions': directions_audit,
            'actual_output_weight_checkpoint_key': head_parameter_name(),
            'scales': SCALES, 'finite_endpoints': len(endpoint_records), 'summary': summary,
            'complete_field_receipts': layer_receipts, 'endpoint_fields': endpoint_fields, 'readout_checks': readout_checks,
            'adjoint_checks': adjoint_checks, 'first_sample_parameter_VJP': parameter_vjp,
            'maximum_adjoint_absolute_error': max(r['absolute_error'] for r in adjoint_checks),
            'seconds': time.monotonic()-start, 'peak_CUDA_bytes': torch.cuda.max_memory_allocated(),
            'scope': 'Complete-prefix known-calculus tracing of actual learned directions. Distinguish smoothFP32 reference, nativeBF16 finite effect and baseline-observed centered prediction; no unique semantic mechanism, early-only extraction or new general mathematical theorem claimed.'}
        save(finish, result); ledger('phase2747_parameter_propagation'+('_pilot' if pilot else ''), result['seconds'])
        print('FORMATION_PARAMETER_PROPAGATION_DONE', pilot, result['seconds'], result['maximum_adjoint_absolute_error'], flush=True)
    except Exception as exc:
        failure(folder, start, exc); raise
    finally:
        del originals, directions, states
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--pilot', action='store_true'); main(parser.parse_args().pilot)
