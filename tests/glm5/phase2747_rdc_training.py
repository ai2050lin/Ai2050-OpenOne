"""Actual original-MLP learning with true, matched-shuffled and surface-mass loss."""
import argparse
from collections import defaultdict
from rdc_formation_common import *
from phase2747_rdc_material import freeze as material_freeze

TRAIN = OUT / 'training'
CONDITIONS = ['true_token', 'within_surface_class_permuted_token', 'surface_class_mass']


class FormationObserver:
    """All native last-position coordinates/units, never amplitude-selected."""
    def __init__(self, model):
        self.active = False
        self.handles = []
        self.values = {}
        self.depth = len(model.model.layers)

        def put(key, value):
            if self.active:
                self.values[key] = value[0, -1].detach().float().cpu().numpy().copy()

        self.handles.append(model.model.embed_tokens.register_forward_hook(lambda m, a, o: put(('H', 0), o)))
        for i, layer in enumerate(model.model.layers):
            self.handles.append(layer.register_forward_hook(lambda m, a, o, i=i: put(('H', i+1), o)))
            self.handles.append(layer.self_attn.q_norm.register_forward_hook(lambda m, a, o, i=i: put(('Q', i), o)))
            self.handles.append(layer.post_attention_layernorm.register_forward_hook(lambda m, a, o, i=i: put(('MLP_input', i), o)))
            self.handles.append(layer.mlp.gate_proj.register_forward_hook(lambda m, a, o, i=i: put(('gate', i), o)))
            self.handles.append(layer.mlp.up_proj.register_forward_hook(lambda m, a, o, i=i: put(('up', i), o)))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: put(('product', i), a[0])))
            self.handles.append(layer.mlp.register_forward_hook(lambda m, a, o, i=i: put(('MLP_write', i), o)))

    def collect(self):
        result = {'hidden': np.stack([self.values[('H', i)] for i in range(self.depth + 1)])}
        for name in ['Q', 'MLP_input', 'gate', 'up', 'product', 'MLP_write']:
            result[name] = np.stack([self.values[(name, i)] for i in range(self.depth)])
        self.values.clear()
        return result

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def packet(model, row):
    import torch
    post = model.model(input_ids=torch.tensor([row['ids']], device='cuda'), use_cache=False).last_hidden_state[0, -1]
    logits = model.lm_head(post[None]).float()[0]
    return post, logits


def objective(model, row, condition, classes):
    import torch
    post, logits = packet(model, row)
    lp = logits.double().log_softmax(-1)
    if condition == 'surface_class_mass':
        value = -torch.logsumexp(lp[classes == int(row['surface_class'])], dim=0)
    else:
        target = row['target'] if condition == 'true_token' else row['permuted_target']
        value = -lp[target]
    return value


def evaluate(model, rows, classes):
    import torch
    values = defaultdict(list)
    with torch.no_grad():
        for row in rows:
            post, z = packet(model, row)
            lp = z.double().log_softmax(-1)
            p = lp.exp()
            values['postnorm_BF16'].append(bits(post))
            values['NLL'].append(float(-lp[row['target']]))
            values['argmax'].append(int(z.argmax()))
            values['entropy'].append(float(-(lp*p).sum()))
            values['surface_mass'].append([float(p[classes == i].sum()) for i in range(10)])
    return {k: np.stack(v) for k, v in values.items()}


def observe(model, rows, observer):
    import torch
    result = defaultdict(list)
    with torch.no_grad():
        for r in rows:
            observer.active = True
            post, logits = packet(model, r)
            observer.active = False
            for k, v in observer.collect().items():
                result[k].append(v)
            result['postnorm_BF16'].append(bits(post))
    return {k: np.stack(v) for k, v in result.items()}


def bridge(target):
    import torch
    target.float()
    return [target.register_forward_pre_hook(lambda m, a: (a[0].float(),) + a[1:]),
            target.register_forward_hook(lambda m, a, o: o.to(torch.bfloat16))]


def prepare_gradients(model):
    from torch.utils.checkpoint import checkpoint
    for p in model.parameters():
        p.requires_grad_(False)
    target = model.model.layers[16].mlp
    original = {n: p.detach().float().cpu().clone() for n, p in target.named_parameters()}
    handles = bridge(target)
    for p in target.parameters():
        p.requires_grad_(True)
    for layer in model.model.layers[17:]:
        original_forward = layer.forward

        def forward(*args, _forward=original_forward, **kwargs):
            import torch
            return checkpoint(_forward, *args, use_reentrant=False, **kwargs) if torch.is_grad_enabled() else _forward(*args, **kwargs)

        layer.forward = forward
    return target, original, handles


def mean_gradient(model, rows, condition, classes, params):
    import torch
    acc = [torch.zeros_like(p) for p in params]
    losses = []
    for r in rows:
        ll = objective(model, r, condition, classes)
        grads = torch.autograd.grad(ll, params)
        losses.append(float(ll.detach()))
        for a, g in zip(acc, grads):
            a.add_(g, alpha=1/len(rows))
        del grads, ll
    assert all(bool(torch.isfinite(g).all()) for g in acc)
    return acc, float(np.mean(losses))


def frozen_protocol():
    material, data = material_freeze()
    path = TRAIN / 'protocol.json'
    if path.exists():
        p = read(path)
        assert p['engine']['sha256'] == sha(__file__), 'Versioned engine changed; preserve old run and audit explicitly'
        return p, data
    pilot = read(TRAIN / 'pilot/result.json')
    assert pilot['all_passed']
    p = {'timestamp': stamp(), 'engine': snapshot(__file__), 'material_sha256': material['material_sha256'],
        'question': 'Which conditional response changes are formed by actual pretrained-parameter continuation, beyond target-frequency/surface-class alternatives?',
        'parameters': 'All74711040 original block16 gate/up/down scalars; no LoRA/compression, all other native parameters fixed.',
        'seeds': [2747, 2748], 'conditions': CONDITIONS, 'steps': 128, 'batch_examples': 16,
        'checkpoints': [1, 8, 32, 128], 'step_FP32_norm': .02,
        'consumption': 'One full permutation of2048real training records in each run. Same seed means identical example order across all3conditions.',
        'precision': 'Trainable MLP FP32 with BF16 input/output boundary; all remaining layers and LMhead originalBF16. Smooth gradient passes through boundary cast, not claimed an exact derivative of discreteBF16 rounding. Native and bridge baselines separately measured.',
        'objectives': {'true_token': '-log p(y|x)', 'within_surface_class_permuted_token': '-log p(y_perm|x)',
                       'surface_class_mass': '-log sum_(v:class(v)=class(y)) p(v|x)'},
        'format_scope': material['format_boundary'], 'selection': 'No checkpoint/seed selected on heldout outcomes; all4checkpoints, all6runs reported.',
        'evaluation': 'All192validation +512exposed diagnostic +192newsource positions at each checkpoint; complete vocabulary probability from B1 actual readout. Full finalBF16 deployment evaluated separately.',
        'fields': 'All37H, all36x9728gate/up/product, all36Q and MLP inputs/write for24natural+20controlled predeclared heldout expressions, every checkpoint, both baselines and finalnativeBF16 deployments. The trainableMLP write hook records its FP32 output BEFORE the explicit BF16 bridge cast; raw H boundaries record the actual BF16 residual stream. OtherMLP writes are originalBF16 values promoted for storage.',
        'initial_gradients': 'Same hash-fixed16training examples for all3full-parameter gradients; exact complete-vector dot products, no selectedcoordinates.',
        'resource_estimate': {'pilot_backward_seconds_per_example': pilot['backward_seconds_per_example'],
            'planned_backward_examples': 12288, 'estimated_backward_seconds': pilot['backward_seconds_per_example']*12288,
            'pilot_CUDA_peak_bytes': pilot['peak_cuda_bytes'], 'nominal_parameter_delta_bytes': 6*4*74711040*4,
            'uncompressed_fullfield_estimate_bytes': 32*44*(37*2560+36*(3*9728+4096+2*2560))*4,
            'arbitrary_elapsed_time_ceiling': None, 'reserve_each_drive_bytes': 4*1024**3},
        'remaining_integrated_tasks': [
            'FP32/BF16 direction/radius matched deployments with permutation and reversal controls',
            'Full training-corpus prior count plus validation-only temperature/total-concentration calibration',
            'Full-parameter change propagation to query/gate/up/layers/output with finite-change/gradient compatibility checks',
            'Existing paired natural-language/program representations: matched mapping/shuffle/gold-free readout',
            'Longer ownhistories and three original unquantized models in sequence; common-ability and failure strata',
            'Theory/formula/client/integrity audit; then same-goal information-value continuation'],
        'scientific_boundary': 'A restricted continuation learning test, not reconstruction of original pretraining or proof of purely semantic coordinates.'}
    immutable(path, p)
    status = read(BASE / 'status.json')
    status.update(timestamp=stamp(), phase2747='active_actual_training_after_passed_pilot',
                  phase2747_protocol='phase2747/training/protocol.json')
    save(BASE / 'status.json', status)
    return p, data


def pilot():
    import torch
    if (TRAIN / 'pilot/result.json').exists():
        print('FORMATION2747_PILOT_ALREADY_COMPLETE', flush=True)
        return
    material, data = material_freeze()
    start = time.monotonic()
    model = None
    handles = []
    storage_guard()
    try:
        model, tok = load('qwen4', TRAIN / 'pilot')
        assert all(p.device.type == 'cuda' for p in model.parameters())
        with np.load(ROOT / material['vocabulary_receipt']['field_path']) as z:
            classes = torch.tensor(z['classes'].astype(np.int64), device='cuda')
        selected = []
        for family in sorted({r['family'] for r in data['train']}):
            selected.append(min((r for r in data['train'] if r['family'] == family), key=lambda r: rank(r['sample_id'])))
        small = data['validation'][:4]
        native = evaluate(model, small, classes)
        target, original, handles = prepare_gradients(model)
        params = list(target.parameters())
        bridge_baseline = evaluate(model, small, classes)
        tick = time.monotonic()
        gradients = []
        losses = []
        for condition in CONDITIONS:
            grads, loss = mean_gradient(model, selected, condition, classes, params)
            gradients.append(grads)
            losses.append(loss)
        torch.cuda.synchronize()
        seconds = time.monotonic()-tick
        norms = [float(torch.stack([g.double().square().sum() for g in gg]).sum().sqrt()) for gg in gradients]
        cos = [[float(sum((x.double()*y.double()).sum() for x, y in zip(a, b)))/(norms[i]*norms[j])
                for j, b in enumerate(gradients)] for i, a in enumerate(gradients)]
        with torch.no_grad():
            for p, g in zip(params, gradients[0]):
                p.add_(g, alpha=-.02/norms[0])
        updated = evaluate(model, small, classes)
        delta_norm = float(torch.stack([(p.detach().double()-original[n].to(p.device).double()).square().sum()
                                       for n, p in target.named_parameters()]).sum().sqrt())
        with torch.no_grad():
            for n, p in target.named_parameters():
                p.copy_(original[n].to(p.device))
        restored = evaluate(model, small, classes)
        assert all(np.array_equal(restored[k], bridge_baseline[k]) for k in restored)
        assert abs(delta_norm-.02) < 1e-5 and all(n > 0 for n in norms)
        result = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True,
            'training_examples': len(data['train']), 'pilot_distinct_examples': len(selected),
            'pilot_examples': [r['sample_id'] for r in selected], 'conditions': CONDITIONS,
            'backward_seconds_per_example': seconds/(3*len(selected)), 'gradient_norms': norms,
            'full_parameter_gradient_cosine': cos, 'objective_values': losses,
            'bridge_minus_native_NLL': float((bridge_baseline['NLL']-native['NLL']).mean()),
            'updated_minus_bridge_NLL': float((updated['NLL']-bridge_baseline['NLL']).mean()),
            'actual_FP32_step_norm': delta_norm, 'restored_all_evaluation_arrays_bit_equal': True,
            'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.monotonic()-start,
            'scope': 'Engineering/numerical pilot on training examples; not a final training-effect conclusion.'}
        save(TRAIN / 'pilot/result.json', result)
        ledger('phase2747_training_pilot', result['seconds'])
        print('FORMATION2747_PILOT', result, flush=True)
    except Exception as exc:
        failure(TRAIN / 'pilot', start, exc)
        raise
    finally:
        for h in handles:
            h.remove()
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def run():
    import torch
    protocol, data = frozen_protocol()
    if (TRAIN / 'result.json').exists():
        print('FORMATION2747_TRAINING_ALREADY_COMPLETE', flush=True)
        return
    start = time.monotonic()
    model = None
    handles = []
    observer = None
    storage_guard()
    try:
        material = read(OUT / 'material/protocol.json')
        model, tok = load('qwen4', TRAIN)
        with np.load(ROOT / material['vocabulary_receipt']['field_path']) as z:
            classes = torch.tensor(z['classes'].astype(np.int64), device='cuda')
        panel = data['validation']+data['diagnostic']+data['fresh']
        fields = []
        for family in sorted({r['family'] for r in data['diagnostic']}):
            rr = sorted((r for r in data['diagnostic'] if r['family'] == family), key=lambda r: rank(r['sample_id']))
            if family.startswith('natural_'):
                fields += rr[:8]
            else:
                group = min({r['source_group'] for r in rr}, key=rank)
                fields += [r for r in rr if r['source_group'] == group]
        assert len(fields) == 44
        immutable(TRAIN / 'evaluation_rows.json', {'panel': [r['sample_id'] for r in panel], 'fields': [r['sample_id'] for r in fields]})
        observer = FormationObserver(model)
        native = evaluate(model, panel, classes)
        if not (TRAIN / 'baseline/commits/native.json').exists():
            commit_array('training/baseline', 'native', **native)
            commit_array('training/baseline', 'native_fields', **observe(model, fields, observer))
        target, original, handles = prepare_gradients(model)
        named = dict(target.named_parameters())
        params = list(named.values())
        assert sum(p.numel() for p in params) == 74711040
        bridge_baseline = evaluate(model, panel, classes)
        if not (TRAIN / 'baseline/commits/bridge.json').exists():
            commit_array('training/baseline', 'bridge', **bridge_baseline)
            commit_array('training/baseline', 'bridge_fields', **observe(model, fields, observer))
        initial_rows = sorted(data['train'], key=lambda r: rank('initial_gradient/'+r['sample_id']))[:16]
        initial_gradients = []
        for condition in CONDITIONS:
            gradients, loss = mean_gradient(model, initial_rows, condition, classes, params)
            initial_gradients.append([g.detach().cpu() for g in gradients])
            path = TRAIN / 'initial_gradients/commits' / (condition+'.json')
            if not path.exists():
                commit_array('training/initial_gradients', condition, **{n: g.detach().cpu().numpy() for n, g in zip(named, gradients)})
            del gradients
        immutable(TRAIN / 'initial_gradient_rows.json', [r['sample_id'] for r in initial_rows])
        runs = []
        for seed in protocol['seeds']:
            draws = np.random.default_rng(seed).permutation(len(data['train'])).reshape(128, 16)
            if not (TRAIN / 'draws/commits' / (str(seed)+'.json')).exists():
                commit_array('training/draws', str(seed), indices=draws)
            for condition in CONDITIONS:
                name = condition + '_' + str(seed)
                folder = TRAIN / name
                if (folder / 'result.json').exists():
                    runs.append(read(folder / 'result.json'))
                    continue
                with torch.no_grad():
                    for n, p in named.items():
                        p.copy_(original[n].to(p.device))
                trace = []
                checkpoints = []
                completed_step = 0
                if (folder / 'progress.json').exists():
                    progress = read(folder / 'progress.json')
                    trace, checkpoints = progress['trace'], progress['checkpoints']
                    completed_step = checkpoints[-1]['step']
                    rec = read(folder / 'commits' / f'delta_{completed_step:03d}.json')
                    assert sha(ROOT / rec['field_path']) == rec['field_sha256']
                    with np.load(ROOT / rec['field_path']) as z, torch.no_grad():
                        for n, p in named.items():
                            rebuilt = original[n].numpy()+z[n]
                            rebuilt = rebuilt+z['reconstruction_residual__'+n]
                            p.copy_(torch.tensor(rebuilt, device=p.device))
                runstart = time.monotonic()
                for step, ix in enumerate(draws, 1):
                    if step <= completed_step:
                        continue
                    gradients, loss = mean_gradient(model, [data['train'][i] for i in ix], condition, classes, params)
                    norm = float(torch.stack([g.double().square().sum() for g in gradients]).sum().sqrt())
                    assert norm > 0 and np.isfinite(norm)
                    with torch.no_grad():
                        for p, g in zip(params, gradients):
                            p.add_(g, alpha=-.02/norm)
                    trace.append({'step': step, 'objective': loss, 'full_gradient_norm': norm,
                                  'nominal_FP32_step_norm': .02, 'examples': ix.tolist()})
                    del gradients
                    if step % 8 == 0:
                        print('FORMATION2747_PROGRESS', name, step, round(time.monotonic()-runstart, 2), flush=True)
                    if step in protocol['checkpoints']:
                        delta = {}
                        reconstruction_audit = {}
                        norm_square = 0.
                        compatibility = np.zeros(3)
                        for param_index, (n, p) in enumerate(named.items()):
                            actual = p.detach().cpu().numpy()
                            value = actual-original[n].numpy()
                            approximate = original[n].numpy()+value
                            residual = actual-approximate
                            rebuilt = approximate+residual
                            assert np.array_equal(rebuilt.view(np.uint32), actual.view(np.uint32))
                            delta[n] = value
                            delta['reconstruction_residual__'+n] = residual
                            reconstruction_audit[n] = {
                                'nonzero_residual_scalars': int(np.count_nonzero(residual)),
                                'maximum_absolute_residual': float(np.abs(residual).max()),
                                'all_reconstructed_parameter_bits_equal': True}
                            exact_difference = actual.astype(np.float64)-original[n].numpy().astype(np.float64)
                            norm_square += float(np.sum(exact_difference**2))
                            for j, gg in enumerate(initial_gradients):
                                compatibility[j] += float(np.sum(exact_difference*gg[param_index].numpy().astype(np.float64)))
                        norm_delta = float(np.sqrt(norm_square))
                        delta_receipt = commit_array('training/'+name, f'delta_{step:03d}', **delta)
                        measured = evaluate(model, panel, classes)
                        evaluation_receipt = commit_array('training/'+name, f'evaluation_{step:03d}', **measured)
                        field_receipt = commit_array('training/'+name, f'fields_{step:03d}', **observe(model, fields, observer))
                        checkpoints.append({'step': step, 'delta_FP32_L2': norm_delta,
                            'initial_full_gradient_dot_delta': dict(zip(CONDITIONS, compatibility.tolist())),
                            'exact_FP32_parameter_reconstruction_from_original_plus_delta_and_residual': True,
                            'parameter_reconstruction_audit': reconstruction_audit,
                            'delta': delta_receipt, 'evaluation': evaluation_receipt, 'fields': field_receipt})
                        save(folder / 'progress.json', {'timestamp': stamp(), 'trace': trace, 'checkpoints': checkpoints})
                        print('FORMATION2747_CHECKPOINT', name, step, norm_delta, round(time.monotonic()-runstart, 2), flush=True)
                        del delta, measured
                        storage_guard()
                for h in handles:
                    h.remove()
                handles = []
                target.bfloat16()
                deployed_norm = float(torch.stack([(p.detach().float().cpu()-original[n]).double().square().sum()
                                                   for n, p in named.items()]).sum().sqrt())
                deployment = commit_array('training/'+name, 'deployed_BF16', **evaluate(model, panel, classes))
                deployment_fields = commit_array('training/'+name, 'deployed_BF16_fields', **observe(model, fields, observer))
                handles = bridge(target)
                run_result = {'timestamp': stamp(), 'condition': condition, 'seed': seed, 'all_passed': True,
                    'trace': trace, 'checkpoints': checkpoints, 'distinct_drawn_examples': len(set(draws.ravel().tolist())),
                    'deployed_BF16_delta_L2': deployed_norm, 'deployment': deployment, 'deployment_fields': deployment_fields,
                    'seconds_this_execution': time.monotonic()-runstart,
                    'complete_training_pool_consumed_once': np.array_equal(np.sort(draws.ravel()), np.arange(2048))}
                assert run_result['complete_training_pool_consumed_once']
                save(folder / 'result.json', run_result)
                runs.append(run_result)
        with torch.no_grad():
            for n, p in named.items():
                p.copy_(original[n].to(p.device))
        restored = evaluate(model, panel[:4], classes)
        assert all(np.array_equal(restored[k], bridge_baseline[k][:4]) for k in restored)
        result = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True,
            'runs': runs, 'conditions': CONDITIONS, 'actual_backward_training_examples': 12288,
            'evaluation_positions_per_checkpoint': len(panel), 'full_field_positions_per_checkpoint': len(fields),
            'original_FP32_bridge_restored_exactly': True, 'original_checkpoint_unmodified_by_implementation': True,
            'seconds': time.monotonic()-start, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
            'scope': 'Actual original-MLP restricted continuation; interpret learning effects only after matched controls and source-group analysis.'}
        save(TRAIN / 'result.json', result)
        ledger('phase2747_training_complete', result['seconds'])
        print('FORMATION2747_TRAINING_COMPLETE', len(runs), result['seconds'], flush=True)
    except Exception as exc:
        failure(TRAIN, start, exc)
        raise
    finally:
        for h in handles:
            h.remove()
        if observer is not None:
            observer.close()
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.pilot:
        pilot()
    elif args.freeze:
        print('FORMATION2747_PROTOCOL', frozen_protocol()[0], flush=True)
    else:
        run()
