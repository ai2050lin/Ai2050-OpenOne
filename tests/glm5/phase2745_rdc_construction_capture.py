"""Full-coordinate contextual query construction in three native models."""
import argparse, sys
from phase2745_rdc_construction_contract import freeze, freeze_extraction
from rdc_construction_batch import BatchedEngine
from rdc_construction_common import *

NAMES = ['query_only', 'uniform', 'quadratic', 'shuffled_values', 'ordered_softmax']


def tensor(a):
    import torch
    return torch.from_numpy(unbits(a).copy()).to(device='cuda', dtype=torch.bfloat16)


def native_features(model, layer, prototypes, probes, keys, values):
    """No actual queried state or answer is read by any candidate."""
    import torch
    att = layer.self_attn
    apply_rope = sys.modules[att.__class__.__module__].apply_rotary_pos_emb
    n, width = keys.shape[-2], model.config.hidden_size
    kp = keys.repeat_interleave(att.num_key_value_groups, 0)
    vp = values.repeat_interleave(att.num_key_value_groups, 0)
    result = np.empty((5, len(probes), width), np.uint16)
    for q, probe in enumerate(probes):
        qq = tensor(prototypes[f'p{q}_q_before_rope_all'])
        kk = tensor(prototypes[f'p{q}_k_before_rope_all'])
        vv = tensor(prototypes[f'p{q}_values_all'])
        h = tensor(prototypes[f'p{q}_Hearly'][-1])
        length = len(probe['token_ids'])
        pos = torch.arange(n, n+length, device='cuda')[None]
        cos, sin = model.model.rotary_emb(h[None, None], pos)
        qr, kr = apply_rope(qq[None], kk[None], cos, sin)
        qr = qr[0, :, -1]
        kr = kr[0].repeat_interleave(att.num_key_value_groups, 0)
        vr = vv.repeat_interleave(att.num_key_value_groups, 0)
        k, v = torch.cat([kp, kr], -2), torch.cat([vp, vr], -2)
        scores = (qr.float().unsqueeze(-2) @ k.float().transpose(-2, -1)).squeeze(-2) * att.scaling
        soft = scores.softmax(-1)
        poly = 1 + scores + .5*scores.square()
        poly = poly / poly.sum(-1, keepdim=True)
        shifted = torch.cat([vp.roll(1, dims=-2), vr], -2)
        readouts = []
        for j, weight in enumerate([torch.ones_like(soft)/soft.shape[-1], poly, soft, soft]):
            val = shifted if j == 2 else v
            mixed = (weight.to(torch.bfloat16).unsqueeze(-2) @ val).squeeze(-2).reshape(1, 1, -1)
            readouts.append(att.o_proj(mixed)[0, 0])
        after_attn = torch.stack(readouts) + h[None]
        after_mlp = after_attn + layer.mlp(layer.post_attention_layernorm(after_attn))
        result[0, q] = prototypes['query_all_states'][layer.self_attn.layer_idx+1, q]
        result[1:, q] = bits(after_mlp)
    assert np.isfinite(unbits(result)).all()
    return result


def distribution_statistics(model, engine, post, reference=None):
    import torch
    rows = np.zeros((len(engine.probes), 4))
    logprobs = [] if reference is None else None
    for _, ix in engine.groups:
        lp = model.lm_head(tensor(post[ix])).float().double().log_softmax(-1)
        prob = lp.exp()
        rows[ix, 0] = (-(prob*lp).sum(-1)).cpu().numpy()
        rows[ix, 3] = lp.argmax(-1).double().cpu().numpy()
        if reference is not None:
            ref = reference[ix].to('cuda')
            rows[ix, 1] = (prob*(lp-ref)).sum(-1).cpu().numpy()
            rows[ix, 2] = (ref.exp()*(ref-lp)).sum(-1).cpu().numpy()
        else:
            logprobs.append((ix, lp.detach().cpu()))
    if reference is None:
        # Keep the complete vocabulary reference on CPU between native layers;
        # only an exact-shape query microbatch is copied to CUDA for scoring.
        reference = torch.empty((len(engine.probes), model.config.vocab_size), dtype=torch.float64)
        for ix, lp in logprobs:
            reference[ix] = lp
    return rows, reference


def pair_arrays(a, b):
    depth, nq, width = a.shape
    coordinate = np.empty((depth, width))
    query = np.empty((depth, nq))
    signed = np.empty((depth, width))
    for layer in range(depth):
        delta = unbits(b[layer]).astype(float) - unbits(a[layer]).astype(float)
        coordinate[layer] = np.mean(delta*delta, axis=0)
        query[layer] = np.mean(delta*delta, axis=1)
        signed[layer] = np.mean(delta, axis=0)
    return {'all_coordinate_query_mean_squared_difference': coordinate,
            'all_query_coordinate_mean_squared_difference': query,
            'all_coordinate_query_mean_signed_difference': signed}


def main(key):
    import torch
    protocol, material = freeze()
    freeze_extraction()
    out = BASE / 'capture' / key
    if (out / 'result.json').exists():
        assert read(out / 'result.json')['all_passed']
        return
    pilot = read(BASE / 'pilot' / key / 'result.json')
    assert pilot['all_passed']
    batch_pilot = read(BASE / 'batch_pilot' / key / 'result.json')
    assert batch_pilot['all_passed']
    scheduler = Path(__file__).with_name('rdc_construction_stream.py')
    assert sha(scheduler) == pilot['scheduler_source']['sha256'], 'Changed scheduler requires fresh numerical admission'
    batched = Path(__file__).with_name('rdc_construction_batch.py')
    assert sha(batched) == batch_pilot['sources'][batched.name]['sha256']
    start, model = time.monotonic(), None
    sources = {p.name: snapshot(p) for p in [Path(__file__), scheduler, batched,
               Path(__file__).with_name('rdc_construction_common.py')]}
    try:
        guard(1024**3)
        model, tok = load(key, out)
        rows, probes = material['models'][key]['rows'], material['models'][key]['probes']
        engine = BatchedEngine(model, key, probes)
        commits, pairs = [], []
        with torch.inference_mode():
            proto = engine.run(standalone=True)
            standalone_identity = identity(proto['query_all_states'])
            assert standalone_identity == pilot['checks'][0]['streamed_identities']['query_all_states']
            proto_arrays = {**proto['detail'], 'query_all_states': proto['query_all_states'], 'postnorm': proto['postnorm']}
            if (out / 'prototypes.npz').exists():
                with np.load(out / 'prototypes.npz') as old:
                    assert set(old.files) == set(proto_arrays) and all(np.array_equal(old[k], v) for k, v in proto_arrays.items())
            else:
                npz(out / 'prototypes.npz', **proto_arrays)
            _, reference = distribution_statistics(model, engine, proto['postnorm'])
            del proto
            pending_results = {}
            pending_metadata = {}
            for begin in range(0, len(rows), 2):
                pair = rows[begin:begin+2]
                pair_name = 'pair_' + rank(pair[0]['pair_id'])[:20]
                pp, pf = out / 'pair_commits' / (pair_name+'.json'), out / 'pairs' / (pair_name+'.npz')
                if pp.exists():
                    record = read(pp)
                    assert sha(pf) == record['sha256']
                    pairs.append(record)
                    for r in pair:
                        c = read(out / 'commits' / (r['sample_id']+'.json'))
                        assert sha(out / 'fields' / (r['sample_id']+'.npz')) == c['sha256']
                        commits.append(c)
                    continue
                if not pending_results:
                    upcoming = []
                    for offset in range(begin, min(begin+16, len(rows)), 2):
                        name = 'pair_' + rank(rows[offset]['pair_id'])[:20]
                        if not (out / 'pair_commits' / (name+'.json')).exists():
                            upcoming.extend(rows[offset:offset+2])
                    tick = time.monotonic()
                    captured = engine.run_many([r['prompt_ids'] for r in upcoming], prototypes=proto_arrays, feature_fn=native_features)
                    batch_seconds = time.monotonic()-tick
                    pending_results = {r['sample_id']: a for r, a in zip(upcoming, captured)}
                    pending_metadata = {r['sample_id']: {'source_batch_ids': [x['sample_id'] for x in upcoming],
                        'source_batch_forward_seconds': batch_seconds,
                        'allocated_forward_seconds': batch_seconds/len(upcoming)} for r in upcoming}
                    del captured
                both = []
                for row in pair:
                    tick = time.monotonic()
                    sid = row['sample_id']
                    result = pending_results.pop(sid)
                    timing = pending_metadata.pop(sid)
                    stats, _ = distribution_statistics(model, engine, result['postnorm'], reference)
                    fixture = row['case'] == 0
                    indices = list(range(engine.depth+1)) if fixture else engine.selected
                    arrays = {**result['detail'], 'prefix_layers': result['prefix_layers'],
                        'prefix_postnorm': result['prefix_postnorm'],
                        'query_layer_indices': np.array(indices), 'query_selected_states': result['query_all_states'][indices],
                        'postnorm': result['postnorm'], 'full_vocabulary_statistics': stats,
                        'candidate_early_output': result['candidate_early_output']}
                    lp = model.lm_head(tensor(result['prefix_postnorm'])).float().double().log_softmax(-1)
                    choice = row['candidate_ids']
                    binary = lp[choice].log_softmax(-1)
                    current = {'full_vocabulary_NLL': float(-lp[row['target_ids'][0]]),
                        'conditional_yes_probability': float(binary[0].exp()),
                        'conditional_answer_NLL': float(-binary[0 if row['truth'] else 1]),
                        'candidate_total_probability': float(lp[choice].exp().sum()),
                        'argmax_id': int(lp.argmax()), 'argmax_correct': int(lp.argmax()) == row['target_ids'][0]}
                    same_old = None
                    if key == 'qwen4':
                        with np.load(OLD / 'identifiability/relations/native/fields' / (sid+'.npz')) as old:
                            same_old = {'postnorm_bit_equal': bool(np.array_equal(result['postnorm'], old['postnorm'])),
                                        'prefix_bit_equal': bool(np.array_equal(result['prefix_layers'], old['prefix_layers']))}
                        assert all(same_old.values()), (sid, same_old)
                    path, cp = out / 'fields' / (sid+'.npz'), out / 'commits' / (sid+'.json')
                    record = {k: row[k] for k in ['sample_id', 'source_group', 'pair_id', 'world', 'family', 'language', 'case', 'split']}
                    record.update(timestamp=stamp(), model=key, prefix_tokens=len(row['prompt_ids']), queries=len(probes),
                        execution_sources=sources,
                        full_layer_fixture=fixture, query_layer_indices=indices, actual_current=current,
                        previous2744=same_old, attention_checks=result['attention_checks'],
                        all_query_state_identity=identity(result['query_all_states']),
                        seconds=time.monotonic()-tick+timing['allocated_forward_seconds'], **timing,
                        seconds_scope='Equal allocation of measured interleaved native forward plus separately timed row readout; not an independently timed native source.')
                    if cp.exists():
                        with np.load(path) as old:
                            assert set(old.files) == set(arrays) and all(np.array_equal(old[k], v) for k, v in arrays.items())
                        assert read(cp)['actual_current'] == current
                        record = read(cp)
                    else:
                        npz(path, **arrays)
                        record['sha256'] = sha(path)
                        save(cp, record)
                    commits.append(record)
                    both.append(result['query_all_states'])
                    del result, arrays, lp, binary
                pa = pair_arrays(*both)
                npz(pf, **pa)
                pr = {k: pair[0][k] for k in ['source_group', 'pair_id', 'family', 'language', 'case', 'split']}
                pr.update(timestamp=stamp(), sample_ids=[r['sample_id'] for r in pair], sha256=sha(pf),
                    all_layer_mean_squared_difference=pa['all_query_coordinate_mean_squared_difference'].mean(-1).tolist(),
                    primary_unit='Pair, clustered by shared family/case across languages; coordinates and queries are not independent samples.')
                save(pp, pr)
                pairs.append(pr)
                del both, pa
                if (begin+2) % 16 == 0:
                    guard(256*1024**2)
                    print('CONSTRUCTION_CAPTURE', key, begin+2, len(rows), round(time.monotonic()-start, 1), flush=True)
                    save(out / 'progress.json', {'timestamp': stamp(), 'rows': len(commits), 'pairs': len(pairs), 'total_rows': len(rows), 'seconds': time.monotonic()-start})
            result = {'timestamp': stamp(), 'sources': sources, 'all_passed': True, 'model': key,
                'rows': len(commits), 'pairs': len(pairs), 'queries': len(commits)*len(probes),
                'depth': engine.depth, 'width': engine.width, 'early': engine.early, 'mid': engine.mid,
                'query_native_head_dim': model.model.layers[engine.early].self_attn.head_dim,
                'query_native_heads': model.config.num_attention_heads, 'selected_query_boundaries': engine.selected,
                'full_layer_fixtures': sum(r['full_layer_fixture'] for r in commits),
                'native_attention_max_error': max(x['max_abs_error'] for r in commits for x in r['attention_checks']),
                'all_previous2744_Q4_bit_equal': all(all(r['previous2744'].values()) for r in commits) if key == 'qwen4' else None,
                'staged_original_blocks': sorted(engine.copied_blocks), 'prototype_sha256': sha(out / 'prototypes.npz'),
                'seconds': time.monotonic()-start, 'peak_cuda_allocated': torch.cuda.max_memory_allocated(),
                'scope': 'Native measurements, exact scheduling and available-prefix one-block candidates. No heldout relationship labels enter predictions; no cross-model equal-axis or extracted full-model claim.'}
            assert len(commits) == 320 and len(pairs) == 160
            assert not pending_results and not pending_metadata
            save(out / 'result.json', result)
            ledger('construction_native_capture_' + key, result['seconds'])
            print('CONSTRUCTION_CAPTURE_DONE', key, result['seconds'], flush=True)
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
