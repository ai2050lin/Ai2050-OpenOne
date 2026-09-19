"""Frozen pair maps under an explicit absolute anchor and native output readout."""
import argparse
from collections import defaultdict
from rdc_construction_common import *
from phase2745_rdc_construction_fit import INPUTS, TARGETS, CONTROLS

OUT = BASE / 'compilation'


def freeze_compile():
    path = OUT / 'protocol.json'
    if path.exists():
        return read(path)
    value = {'timestamp': stamp(), 'source': snapshot(__file__),
        'timing': 'Frozen before any2745full-matrix fitted outcome inspection. Native capture and direction-control outcomes already partly observed; this is not an original-campaign preregistration.',
        'question': 'Does a relation-change predictor also predict absolute states, actual nativeQ, and full-vocabulary distributions?',
        'anchor_formula': 'Yhat(p,q)=Yalone(q)+(X(p,q)-Xalone(q))A. This is an additional extrapolation assumption, not implied by accurate pair changes.',
        'inputs': {'actual_query_H1': 'Actual query after nativeblock0 and no later observed query state.',
            'available_prefix_ordered_candidate_early_output': 'Original native-parameter one-block ordered rule, using available prefixKV and standalone query template, no actual target or queryKV.'},
        'baselines': 'Standalone target; same-input through-origin paired diagonal with the same standalone anchor; separately fitted absolute diagonal from[1,H1]. True and same-input training-pair shuffled controls retained.',
        'selection': 'No new fit, calibration or test-driven selection. All previously validation-selected operators are frozen.',
        'evaluation': 'Every80test expression and20unseen query;80expressions form20shared semantic groups. Other query rows computed only to preserve original exact-length B<=16 lm_head shapes.',
        'native_projection': 'ForecastHearly through original input_layernorm, q_proj and q_norm if present; compare all pre-RoPEQ coordinates. No actual futureK/V used as predictor input. ActualHearly through last-token-only projection quantifies GEMM-shape numerical floor.',
        'native_head': 'Forecast postnorm rounded BF16, then original BF16 lm_head and full-vocabulary FP64 log_softmax. Actual reference reproduced in the original capture query batch groups and checked against saved entropy/argmax.',
        'metrics': 'Full-coordinate absolute FP64 and deployed BF16 MSE, fullQ MSE, KL(native||forecast), reverseKL, entropy and argmax agreement. Group paired intervals; no generation or semantic-neuron claim.',
        'retention': 'Original input/target fields and full frozen matrices retained. Forecast tensors exactly reconstructable; retain all-coordinate error sums and every evaluated expression/query scalar metric, not redundant full forecast copies.'}
    immutable(path, value)
    return value


def native_modules(key, early):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from safetensors import safe_open
    path = ROOT / 'models/hf' / MODELS[key]
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    with torch.device('meta'):
        shell = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation='eager')
    layer = shell.model.layers[early]
    modules = {'head': (shell.lm_head, 'lm_head'), 'input_norm': (layer.input_layernorm, f'model.layers.{early}.input_layernorm'),
               'q_proj': (layer.self_attn.q_proj, f'model.layers.{early}.self_attn.q_proj')}
    if hasattr(layer.self_attn, 'q_norm'):
        modules['q_norm'] = (layer.self_attn.q_norm, f'model.layers.{early}.self_attn.q_norm')
    mapping = read(path / 'model.safetensors.index.json')['weight_map']
    result, receipt = {}, []
    for name, (module, prefix) in modules.items():
        state = {}
        for suffix in module.state_dict():
            full = prefix+'.'+suffix
            if full not in mapping and name == 'head' and config.tie_word_embeddings:
                full = 'model.embed_tokens.weight'
            with safe_open(str(path / mapping[full]), framework='pt', device='cpu', backend='pread') as f:
                t = f.get_tensor(full)
                assert t.dtype == torch.bfloat16
                state[suffix] = t.to('cuda')
                receipt.append({'tensor': full, 'shape': list(t.shape), 'dtype': str(t.dtype), 'all_scalars': t.numel()})
        module.load_state_dict(state, strict=True, assign=True)
        module.eval()
        result[name] = module
    result['head_dim'] = layer.self_attn.head_dim
    result['heads'] = config.num_attention_heads
    del shell, layer, modules
    return result, receipt


def projected_q(h, modules):
    shape = (*h.shape[:-1], modules['heads'], modules['head_dim'])
    q = modules['q_proj'](modules['input_norm'](h)).view(shape)
    if 'q_norm' in modules:
        q = modules['q_norm'](q)
    return q.flatten(-2)


def query_groups(probes):
    grouped = defaultdict(list)
    for q, p in enumerate(probes):
        grouped[len(p['token_ids'])].append(q)
    return [ix[i:i+16] for _, ix in sorted(grouped.items()) for i in range(0, len(ix), 16)]


def forecasts(x, ysolo, xsolo, operators, diagonal, ti):
    import torch
    result = {'standalone': ysolo[ti]}
    for input_name in INPUTS:
        dx = x[input_name]-xsolo[input_name]
        for ci, control in enumerate(CONTROLS):
            result['full__'+input_name+'__'+str(ci)] = ysolo[ti]+dx@operators[input_name, ci, ti]
    for ci in range(2):
        delta = x['actual_query_H1']-xsolo['actual_query_H1']
        result['H1_diagonal_paired__'+str(ci)] = ysolo[ti]+delta*diagonal['paired', ci][:, 0, ti]
        coef = diagonal['absolute', ci][:, :, ti]
        result['H1_diagonal_absolute__'+str(ci)] = coef[:, 0]+x['actual_query_H1']*coef[:, 1]
    assert all(torch.isfinite(v).all() for v in result.values())
    return result


def same_input_test(key, material):
    pairs = read(BASE / 'fit' / key / 'pair_index.json')
    probes = material['models'][key]['probes']
    pi = [i for i, p in enumerate(pairs) if p['split'] == 'test']
    qi = [i for i, q in enumerate(probes) if q['split'] == 'unseen_query']
    groups = [pairs[i]['source_group'] for i in pi]
    with np.load(BASE / 'fit' / key / 'all_coordinate_pair_errors.npz') as z:
        full = z['pair_query_MSE'][0, 0]
    with np.load(BASE / 'diagonal' / key / 'all_errors.npz') as z:
        diagonal = z['paired_change_MSE'][5, 1, 0]
    reports = []
    for ti, target in enumerate(TARGETS):
        a, b = full[ti, pi][:, qi].mean(-1), diagonal[ti, pi][:, qi].mean(-1)
        reports.append({'target': target, 'groups': len(set(groups)), 'pairs': len(pi),
            'full_matrix_MSE': clustered(a, groups), 'same_input_diagonal_MSE': clustered(b, groups),
            'diagonal_minus_full_MSE': clustered(b-a, groups),
            'scope': 'Both use only actualH1; cross-coordinate coupling has higher parameter capacity, not uniquely mechanistic attribution.'})
    return reports


def forecast_preflight():
    import torch
    generator=torch.Generator().manual_seed(2745007)
    random=lambda *s:torch.randn(s,generator=generator,dtype=torch.float64)
    solo=[random(7,5),random(7,5)]
    xs={n:random(7,5) for n in INPUTS}
    x={n:random(7,5) for n in INPUTS}
    operators={(n,c,t):random(5,5) for n in INPUTS for c in range(2) for t in range(2)}
    diagonal={(o,c):random(5,1 if o=='paired' else 2,2) for o in ['paired','absolute'] for c in range(2)}
    for ti in range(2):
        y=forecasts(x,solo,xs,operators,diagonal,ti)
        z=forecasts(xs,solo,xs,operators,diagonal,ti)
        assert len(y)==9 and all(tuple(a.shape)==(7,5) for a in y.values())
        for n in INPUTS:
            for ci in range(2):
                name='full__'+n+'__'+str(ci)
                assert torch.equal(z[name],solo[ti])
                assert torch.allclose(y[name]-z[name],(x[n]-xs[n])@operators[n,ci,ti],atol=1e-12,rtol=1e-12)
    return {'synthetic':True,'all_passed':True,'nine_forecast_names_and_shapes_checked':True,'standalone_anchor_and_full_pair_algebra_checked':True}


def main(key):
    import torch, psutil
    protocol = freeze_compile()
    out = OUT / key
    if (out / 'result.json').exists():
        assert read(out / 'result.json')['all_passed']
        return
    own_process_chain = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.info['pid'] in own_process_chain:
            continue
        command = process.info['cmdline'] or []
        if any(Path(arg).name in {'phase2745_rdc_construction_capture.py', 'phase2745_rdc_construction_pilot.py',
            'phase2745_rdc_construction_batch_pilot.py', 'phase2745_rdc_construction_fit.py', 'phase2745_rdc_construction_norm.py',
            'phase2745_rdc_construction_compile.py'} for arg in command):
            raise RuntimeError('Native/solver CUDA process still active: '+str(process.info['pid']))
    assert all(read(BASE / stage / key / 'result.json')['all_passed'] for stage in ['capture', 'fit', 'diagonal'])
    start = time.monotonic()
    version = snapshot(__file__)
    numerical_preflight = forecast_preflight()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    material = gzread(BASE / 'material.json.gz')
    rows = [r for r in material['models'][key]['rows'] if r['split'] == 'test']
    probes = material['models'][key]['probes']
    unseen = [q for q, p in enumerate(probes) if p['split'] == 'unseen_query']
    qslot = {q: i for i, q in enumerate(unseen)}
    groups = query_groups(probes)
    native = read(BASE / 'capture' / key / 'result.json')
    width, early = native['width'], native['early']
    assert len(rows) == 80 and len(unseen) == 20
    operator, diagonal = {}, {}
    for input_name in INPUTS:
        for ci, control in enumerate(CONTROLS):
            for ti, target in enumerate(TARGETS):
                file = BASE / 'fit' / key / 'operators' / (input_name+'__'+control+'__'+target+'.npz')
                assert sha(file) == read(file.with_suffix('.json'))['operator_sha256']
                with np.load(file) as z:
                    operator[input_name, ci, ti] = torch.from_numpy(z['operator']).to('cuda')
    for objective in ['absolute', 'paired']:
        for ci, control in enumerate(['true_correspondence', 'training_pair_shuffled']):
            with np.load(BASE / 'diagonal' / key / 'coefficients' / ('actual_query_H1__'+objective+'__'+control+'.npz')) as z:
                diagonal[objective, ci] = torch.from_numpy(z['coefficients']).to('cuda')
    with np.load(BASE / 'capture' / key / 'prototypes.npz') as z:
        hs = unbits(z['query_all_states']).astype(float)
        solo = [torch.from_numpy(hs[early]).to('cuda'), torch.from_numpy(unbits(z['postnorm']).astype(float)).to('cuda')]
        xsolo = {INPUTS[0]: torch.from_numpy(hs[1]).to('cuda'), INPUTS[1]: torch.from_numpy(hs[early+1]).to('cuda')}
    modules, loaded_parameters = native_modules(key, early)
    names = ['standalone']+['full__'+i+'__'+str(c) for i in INPUTS for c in range(2)]+[f'H1_diagonal_{o}__{c}' for o in ['paired', 'absolute'] for c in range(2)]
    metrics = {k: np.empty((len(names), len(rows), len(unseen))) for k in
               ['Hearly_MSE', 'postnorm_MSE', 'Hearly_BF16_MSE', 'postnorm_BF16_MSE', 'Q_MSE', 'KL_native_to_prediction', 'KL_prediction_to_native', 'predicted_entropy', 'argmax_agreement']}
    coords = {k: np.zeros((len(names), width)) for k in ['Hearly_MSE', 'postnorm_MSE']}
    qcoords = np.zeros((len(names), modules['heads']*modules['head_dim']))
    floor = np.zeros((len(rows), len(unseen)))
    head_checks = []
    try:
      with torch.inference_mode():
        for ri, row in enumerate(rows):
            path = BASE / 'capture' / key / 'fields' / (row['sample_id']+'.npz')
            assert sha(path) == read(BASE / 'capture' / key / 'commits' / (row['sample_id']+'.json'))['sha256']
            with np.load(path) as z:
                ix = z['query_layer_indices'].tolist()
                h = z['query_selected_states']
                x = {INPUTS[0]: torch.from_numpy(unbits(h[ix.index(1)]).astype(float)).to('cuda'),
                     INPUTS[1]: torch.from_numpy(unbits(z['candidate_early_output'][4]).astype(float)).to('cuda')}
                targets = [torch.from_numpy(unbits(h[ix.index(early)]).astype(float)).to('cuda'),
                           torch.from_numpy(unbits(z['postnorm']).astype(float)).to('cuda')]
                actual_q = torch.from_numpy(np.stack([unbits(z[f'p{q}_L{early}_q_before_rope']).reshape(-1) for q in range(100)]).astype(float)).to('cuda')
                original_stats = z['full_vocabulary_statistics']
            predictions = [forecasts(x, solo, xsolo, operator, diagonal, ti) for ti in range(2)]
            assert all(set(d) == set(names) for d in predictions)
            for ti, target in enumerate(TARGETS):
                for ni, name in enumerate(names):
                    delta = predictions[ti][name][unseen]-targets[ti][unseen]
                    metrics[target+'_MSE'][ni, ri] = delta.square().mean(-1).cpu().numpy()
                    metrics[target+'_BF16_MSE'][ni, ri] = (predictions[ti][name][unseen].bfloat16().double()-targets[ti][unseen]).square().mean(-1).cpu().numpy()
                    coords[target+'_MSE'][ni] += delta.square().sum(0).cpu().numpy()
            entropy_error, argmax_diff = 0., 0
            for ix in groups:
                active = [j for j, q in enumerate(ix) if q in qslot]
                slots = [qslot[ix[j]] for j in active]
                lp = modules['head'](targets[1][ix].bfloat16()).float().double().log_softmax(-1)
                entropy = -(lp.exp()*lp).sum(-1)
                entropy_error = max(entropy_error, float(np.max(abs(entropy.cpu().numpy()-original_stats[ix, 0]))))
                argmax_diff += int(np.count_nonzero(lp.argmax(-1).cpu().numpy() != original_stats[ix, 3]))
                if not active:
                    continue
                ref = lp[active]
                ref_probability = ref.exp()
                actual_q_sub = actual_q[ix][active]
                q_floor = projected_q(targets[0][ix, None].bfloat16(), modules)[:, 0].double()[active]
                floor[ri, slots] = (q_floor-actual_q_sub).square().mean(-1).cpu().numpy()
                for ni, name in enumerate(names):
                    pq = projected_q(predictions[0][name][ix, None].bfloat16(), modules)[:, 0].double()[active]
                    qdelta = (pq-actual_q_sub).square()
                    metrics['Q_MSE'][ni, ri, slots] = qdelta.mean(-1).cpu().numpy()
                    qcoords[ni] += qdelta.sum(0).cpu().numpy()
                    pp = modules['head'](predictions[1][name][ix].bfloat16()).float().double().log_softmax(-1)[active]
                    prob = pp.exp()
                    metrics['KL_native_to_prediction'][ni, ri, slots] = (ref_probability*(ref-pp)).sum(-1).cpu().numpy()
                    metrics['KL_prediction_to_native'][ni, ri, slots] = (prob*(pp-ref)).sum(-1).cpu().numpy()
                    metrics['predicted_entropy'][ni, ri, slots] = -(prob*pp).sum(-1).cpu().numpy()
                    metrics['argmax_agreement'][ni, ri, slots] = (pp.argmax(-1) == ref.argmax(-1)).double().cpu().numpy()
                del lp, ref, pp, prob, ref_probability, pq, qdelta
            assert entropy_error < 1e-12 and argmax_diff == 0, ('Native head shape replay failed', row['sample_id'], entropy_error, argmax_diff)
            head_checks.append({'sample_id': row['sample_id'], 'all100_query_entropy_max_error': entropy_error, 'all100_argmax_disagreements': argmax_diff})
            if (ri+1)%8 == 0:
                print('CONSTRUCTION_COMPILE', key, ri+1, len(rows), round(time.monotonic()-start, 1), flush=True)
            del x, targets, predictions, actual_q
        assert all(np.isfinite(a).all() for a in metrics.values())
        assert all(float(metrics[k].min()) > -1e-9 for k in ['KL_native_to_prediction', 'KL_prediction_to_native'])
        npz(out / 'every_test_expression_query.npz', **metrics, Q_native_last_token_projection_floor_MSE=floor)
        npz(out / 'all_coordinate_errors.npz', **{k: a/(len(rows)*len(unseen)) for k, a in coords.items()}, Q_MSE=qcoords/(len(rows)*len(unseen)))
        save(out / 'metric_identity.json', {'variants': names, 'rows': rows, 'query_indices': unseen, 'query_ids': [probes[q]['probe_id'] for q in unseen], 'metrics': list(metrics)})
        summaries = []
        for family in ['all']+sorted({r['family'] for r in rows}):
            ix = [i for i, r in enumerate(rows) if family == 'all' or r['family'] == family]
            gs = [rows[i]['source_group'] for i in ix]
            for ni, name in enumerate(names):
                summaries.append({'family': family, 'variant': name, 'expressions': len(ix),
                    'metrics': {k: clustered(a[ni, ix].mean(-1), gs) for k, a in metrics.items()},
                    'standalone_minus_variant': {k: clustered((metrics[k][0, ix]-metrics[k][ni, ix]).mean(-1), gs) for k in ['Hearly_MSE', 'postnorm_MSE', 'Q_MSE', 'KL_native_to_prediction']}})
        primary_index = names.index('full__actual_query_H1__0')
        controls = ['full__actual_query_H1__1', 'H1_diagonal_paired__0', 'H1_diagonal_absolute__0', 'standalone']
        comparison = []
        gs = [r['source_group'] for r in rows]
        for control in controls:
            ci = names.index(control)
            comparison.append({'control': control, 'control_minus_primary': {k: clustered((a[ci]-a[primary_index]).mean(-1), gs) for k, a in metrics.items()}})
        result = {'timestamp': stamp(), 'source': version, 'all_passed': True, 'model': key, 'protocol': protocol,
            'expressions': len(rows), 'queries': len(unseen), 'semantic_groups': len(set(gs)), 'variants': names,
            'numerical_forecast_preflight': numerical_preflight,
            'same_input_pair_prediction': same_input_test(key, material), 'summaries': summaries, 'primary_comparisons': comparison,
            'native_head_reconstruction': head_checks, 'loaded_original_parameter_tensors': loaded_parameters,
            'Q_native_last_token_projection_floor_MSE': clustered(floor.mean(-1), gs),
            'seconds': time.monotonic()-start, 'peak_cuda_allocated': torch.cuda.max_memory_allocated(),
            'scope': 'Original native head/Q modules only; no original decoder forward, no future target supplied to forecast, no rollout or full mechanism closure claim.'}
        save(out / 'result.json', result)
        ledger('construction_native_compilation_'+key, result['seconds'])
        print('CONSTRUCTION_COMPILE_DONE', key, result['seconds'], flush=True)
    except Exception as exc:
        failure(out, start, exc)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', nargs='?', choices=['qwen4', 'qwen14', 'glm4'])
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze_compile()
        print('COMPILATION_PROTOCOL_FROZEN', flush=True)
    else:
        assert args.model
        main(args.model)
