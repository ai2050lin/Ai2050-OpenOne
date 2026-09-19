"""Stream every token and native layer, retaining complete anchor fields and native MLP factors."""
import argparse
import gc
import re
import sys
from collections import Counter, defaultdict
from rdc_operator_common import *
from phase2724_rdc_operator_material import lexical_features

BLOCKS = (6, 16, 34)
CLASSES = ['initial', 'whitespace', 'punctuation', 'number_piece', 'latin_piece', 'cjk_piece', 'other_piece']


def persist_arrays(path, packet):
    if path.exists():
        with np.load(path) as old:
            assert set(old.files) == set(packet)
            for key, value in packet.items():
                assert np.array_equal(old[key], value, equal_nan=True), ('Native replay mismatch', path, key)
    else:
        npz(path, **packet)


def token_conditions(row, tok):
    cats, cues = [], []
    for i, tid in enumerate(row['prompt_ids']):
        piece = tok.decode([tid], clean_up_tokenization_spaces=False)
        s = piece.strip()
        cat = 0 if i == 0 else 1 if not s else 2 if all(not c.isalnum() for c in s) else 3 if s.isdecimal() else 4 if re.fullmatch(r'[A-Za-z]+', s) else 5 if any('\u4e00' <= c <= '\u9fff' for c in s) else 6
        cats.append(cat)
        cue = lexical_features(row['text'][:row['token_offsets'][i][1]])
        cues.append(sum((1 << j) * int(cue[k]) for j, k in enumerate(('cause', 'contrast', 'negation', 'reference'))))
    return np.asarray(cats), np.asarray(cues)


class AllFieldObserver:
    def __init__(self, model):
        self.enabled = False
        self.model = model
        self.handles = []
        self.reset([], False)
        def emb(m, a, o):
            if self.enabled:
                self.keep_hidden(0, o[0])
        self.handles.append(model.get_input_embeddings().register_forward_hook(emb))
        for i, layer in enumerate(model.model.layers):
            def before(m, a, i=i):
                if self.enabled:
                    self.active[i] = {'input': a[0]}
            self.handles.append(layer.register_forward_pre_hook(before))
            def attention(m, a, o, i=i):
                if self.enabled:
                    self.active[i]['attention'] = o[0]
                    if i in BLOCKS:
                        self.active[i]['attention_probability'] = o[1]
            self.handles.append(layer.self_attn.register_forward_hook(attention))
            def mlp(m, a, o, i=i):
                if self.enabled:
                    self.active[i]['mlp'] = o
            self.handles.append(layer.mlp.register_forward_hook(mlp))
            if i in BLOCKS:
                for key, module in [('x', layer.post_attention_layernorm), ('gate', layer.mlp.gate_proj), ('up', layer.mlp.up_proj)]:
                    def factor(m, a, o, i=i, key=key):
                        if self.enabled:
                            self.active[i][key] = o
                    self.handles.append(module.register_forward_hook(factor))
                def activation(m, a, i=i):
                    if self.enabled:
                        self.active[i]['activation'] = a[0]
                self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(activation))
            def after(m, a, o, i=i):
                if not self.enabled:
                    return
                import torch
                o = o[0] if isinstance(o, tuple) else o
                self.keep_hidden(i+1, o[0])
                d = self.active.pop(i)
                r, att, mlp = (d[k][0].float() for k in ('input', 'attention', 'mlp'))
                y = o[0].float()
                stats = torch.stack([r.square().mean(-1), att.square().mean(-1), mlp.square().mean(-1),
                    2*(r*att).mean(-1), 2*(r*mlp).mean(-1), 2*(att*mlp).mean(-1),
                    y.square().mean(-1), (y-r).square().mean(-1), (y-r-att-mlp).square().mean(-1)])
                self.block_energy[i] = stats.cpu().numpy()
                if i in BLOCKS:
                    d['output'] = o
                    self.factors[i] = d
            self.handles.append(layer.register_forward_hook(after))

    def reset(self, positions, full, indicators=None):
        self.positions, self.full = positions, full
        self.indicators = indicators
        self.hidden, self.full_hidden, self.hidden_hashes, self.energies = {}, {}, {}, {}
        self.block_energy, self.active, self.factors, self.moments = {}, {}, {}, {}

    def keep_hidden(self, index, value):
        import torch
        raw = bits(value)
        self.hidden[index] = raw[self.positions]
        if self.full:
            self.full_hidden[index] = raw
        self.hidden_hashes[index] = identity(raw)
        f = value.float()
        ee = f.square().mean(-1)
        self.energies[index] = ee.cpu().numpy()
        if self.indicators is not None:
            ind = self.indicators
            u = f / ee.sqrt().clamp_min(1e-12)[:, None]
            self.moments[index] = torch.stack([ind.T @ f, ind.T @ f.square(), ind.T @ u, ind.T @ u.square()]).cpu().numpy()

    def close(self):
        for handle in self.handles:
            handle.remove()


def main(pilot=False, confirmation=False):
    import torch
    import psutil
    from phase2662_symmetric_mapping_contract import load_native
    material = rows()
    if confirmation:
        assert (BASE / 'operators/frozen.json').exists(), 'No confirmation capture before operator selection freeze'
        selected = [r for r in material if r['split'] == 'confirmation']
        mode = 'confirmation'
    elif pilot:
        selected = [r for lang in ('en', 'zh') for r in [s for s in material if s['split'] == 'train' and s['language'] == lang][:4]]
        mode = 'pilot'
    else:
        assert read(BASE / 'capture_pilot.json')['passed']
        selected = [r for r in material if r['split'] != 'confirmation']
        mode = 'main'
    out = BASE / 'capture' / mode
    if (out / 'result.json').exists():
        print('OPERATOR_CAPTURE_ALREADY_COMPLETE', mode, flush=True)
        return
    guard(128 * 1024**2)
    start = time.monotonic()
    source = snapshot(Path(__file__))
    config = read(BASE / 'resources.json')
    model, tok = load_native('qwen4')
    assert len(model.model.layers) == 36 and model.config.hidden_size == 2560 and not getattr(model, 'is_quantized', False)
    observer = AllFieldObserver(model)
    device = model.get_input_embeddings().weight.device
    keep = set(read(BASE / 'material_audit.json')['representative_full_fields'])
    runtime = {'timestamp': stamp(), 'model': 'qwen4', 'dtype': str(model.dtype), 'quantized': False, 'torch': torch.__version__,
        'device_map': getattr(model, 'hf_device_map', {'actual_first_parameter': str(next(model.parameters()).device)}), 'config': model.config.to_dict(), 'source': source,
        'model_code_sha': sha(Path(sys.modules[model.model.__class__.__module__].__file__)),
        'model_fingerprint_reference': str((PRIOR / 'verification/model_checkpoint_fingerprints.json').relative_to(ROOT)),
        'material_sha': sha(BASE / 'material.json.gz'), 'execution': 'CUDA native BF16 eager; unpadded batch1; no chat; no truncation of frozen window; use_cache=False.',
        'coverage': 'Every actual input token, embedding plus36 post-block residual boundaries; post-final-RMSnorm separate; all2560 residual coordinates and all9728 units at blocks6,16,34.',
        'streamed_statistics': 'FP32 reductions on actual BF16 values, accumulated in host FP64. Raw field identity uses original16bit payload. Float reductions are not exact BF16 architectural identities.',
        'conditions': 'Split/language plus seven observable token-piece classes and16 causal-prefix lexical cue masks. Not gold syntax/meaning. No future question/answer label supplied.',
        'storage': 'Every source:37-layer2-anchor raw fields, three-block all-unit2-anchor factors, every-token all-layer energy and full-vocabulary output statistics;16 predeclared full all-token fixtures across complete campaign.'}
    if (out / 'runtime.json').exists():
        immutable(out / 'runtime_history' / (sha(out / 'runtime.json')[:16]+'.json'), read(out / 'runtime.json'))
    save(out / 'runtime.json', runtime)
    accum, counts, records = {}, {}, []
    try:
      with torch.inference_mode():
        for index, row in enumerate(selected):
            tick = time.monotonic()
            sid, n, positions = row['sample_id'], len(row['prompt_ids']), row['anchors']
            assert tok(row['text'], add_special_tokens=False)['input_ids'] == row['prompt_ids']
            cats, cues = token_conditions(row, tok)
            # Each row contributes to one class and one cue combination, keeping all ordinary positions.
            ind = torch.zeros((n, 23), device=device)
            arange = torch.arange(n, device=device)
            ind[arange, torch.tensor(cats, device=device)] = 1
            ind[arange, torch.tensor(cues+7, device=device)] = 1
            observer.reset(positions, sid in keep or pilot, ind)
            observer.enabled = True
            ids = torch.tensor([row['prompt_ids']], device=device)
            post = model.model(input_ids=ids, use_cache=False).last_hidden_state
            observer.enabled = False
            assert len(observer.hidden) == 37 and len(observer.block_energy) == 36
            packet = {'H': np.stack([observer.hidden[l] for l in range(37)]), 'positions': np.array(positions),
                      'postnorm': bits(post[0, positions])}
            factors = {}
            unit_moments = {}
            checks = {}
            for b in BLOCKS:
                d = observer.factors[b]
                layer = model.model.layers[b]
                for name in ('input', 'attention', 'x', 'gate', 'up', 'activation', 'mlp', 'output'):
                    factors[f'L{b}_{name}'] = bits(d[name][0, positions])
                ap = d['attention_probability'][0, :, positions]
                assert ap is not None
                factors[f'L{b}_attention_probability'] = bits(ap)
                # All tokens and all units contribute to complete conditional moments.
                g, u, a = (d[k][0].float() for k in ('gate', 'up', 'activation'))
                phi = layer.mlp.act_fn(d['gate'])[0].float()
                unit_moments[b] = torch.stack([ind.T @ g, ind.T @ u, ind.T @ a,
                    ind.T @ phi, ind.T @ g.square(), ind.T @ u.square(), ind.T @ a.square(),
                    ind.T @ phi.square(), ind.T @ (phi*u)]).cpu().numpy()
                if pilot or index < 2:
                    residual = d['input'] + d['attention']
                    assert torch.equal(layer.post_attention_layernorm(residual), d['x'])
                    assert torch.equal(layer.mlp.act_fn(d['gate']) * d['up'], d['activation'])
                    assert torch.equal(layer.mlp.down_proj(d['activation']), d['mlp'])
                    assert torch.equal(residual + d['mlp'], d['output'])
                    checks[f'L{b}_norm_gate_up_down_residual_bitwise'] = True
            factors['positions'] = np.array(positions)
            if pilot and index < 2:
                plain = model.model(input_ids=ids, use_cache=False).last_hidden_state
                assert torch.equal(plain, post)
                changed_ids = ids.clone()
                q = positions[0]
                changed_ids[:, q+1:] = tok.eos_token_id
                changed = model.model(input_ids=changed_ids, use_cache=False).last_hidden_state
                assert torch.equal(changed[:, :q+1], post[:, :q+1])
                checks['hook_noop_and_same_shape_future_suffix_invariance'] = True
                emb = bits(model.get_input_embeddings().weight[ids[0, positions]])
                assert np.array_equal(emb, packet['H'][0])
                checks['embedding_table_identity'] = True
                del plain, changed, changed_ids, emb
            logits_meta = []
            for at in range(0, n, 32):
                logits = model.lm_head(post[0, at:at+32]).float()
                lp = logits.log_softmax(-1)
                valid = min(32, n-at-1)
                nll = -lp[torch.arange(valid, device=lp.device), ids[0, at+1:at+1+valid]]
                values = {'native_argmax': logits.argmax(-1).cpu().numpy(),
                          'entropy': (-(lp.exp()*lp).sum(-1)).cpu().numpy(),
                          'next_NLL': np.r_[nll.cpu().numpy(), [np.nan] if at+32 >= n else []]}
                assert len(values['next_NLL']) == len(values['entropy'])
                logits_meta.append(values)
                del logits, lp
            energy = np.stack([observer.energies[l] for l in range(37)])
            epacket = {'H_energy': energy, 'block_terms': np.stack([observer.block_energy[l] for l in range(36)]),
                      'token_class': cats.astype(np.int8), 'prefix_cue_mask': cues.astype(np.int8),
                      **{k: np.concatenate([d[k] for d in logits_meta]) for k in logits_meta[0]}}
            assert all(np.isfinite(unbits(a)).all() for k, a in packet.items() if k != 'positions')
            assert np.isfinite(energy).all() and np.isfinite(epacket['block_terms']).all()
            # Pilot does not contribute again to main moments: main replays the same source independently.
            key = row['split'] + '_' + row['language']
            hstats = np.stack([observer.moments[l] for l in range(37)])
            ustat = np.stack([unit_moments[b] for b in BLOCKS])
            if key not in accum:
                accum[key] = [np.zeros_like(hstats, dtype=np.float64), np.zeros_like(ustat, dtype=np.float64)]
                counts[key] = np.zeros(23, np.int64)
            accum[key][0] += hstats
            accum[key][1] += ustat
            counts[key] += ind.sum(0).cpu().numpy().astype(np.int64)
            estbytes = sum(a.nbytes for a in packet.values()) + sum(a.nbytes for a in factors.values())
            guard(estbytes + 64 * 1024**2)
            persist_arrays(out / 'fields' / (sid + '.npz'), packet)
            persist_arrays(out / 'factors' / (sid + '.npz'), factors)
            persist_arrays(out / 'energies' / (sid + '.npz'), epacket)
            if observer.full:
                persist_arrays(out / 'full_fields' / (sid + '.npz'), {'H': np.stack([observer.full_hidden[l] for l in range(37)]), 'postnorm': bits(post[0])})
            # Native full-layer values are independently hashed before nonfixture raw buffers are released.
            record = {'sample_id': sid, 'source_group': row['source_group'], 'split': row['split'], 'language': row['language'],
                'tokens': n, 'positions': positions, 'seconds': time.monotonic()-tick, 'full_H_identities': observer.hidden_hashes,
                'anchor_arrays': {k: identity(a) for k, a in packet.items()}, 'factor_arrays': {k: identity(a) for k, a in factors.items()},
                'checks': checks, 'all_layer_all_token_processed': True, 'anchor_raw_bytes': estbytes,
                'full_field_retained': observer.full, 'next_NLL_mean': float(np.mean(epacket['next_NLL'][:-1])),
                'native_argmax_next_match': float(np.mean(epacket['native_argmax'][:-1] == np.asarray(row['prompt_ids'][1:])))}
            commit = out / 'commits' / (sid + '.json.gz')
            if commit.exists():
                previous = gzread(commit)
                for name in ('full_H_identities', 'anchor_arrays', 'factor_arrays'):
                    # JSON makes integer dictionary keys strings.
                    assert json.loads(json.dumps(record[name])) == previous[name], ('Replay identity', sid, name)
            else:
                compressed(commit, record)
            pilot_path = BASE / 'capture/pilot/commits' / (sid + '.json.gz')
            if not pilot and pilot_path.exists():
                previous = gzread(pilot_path)
                for name in ('full_H_identities', 'anchor_arrays', 'factor_arrays'):
                    assert json.loads(json.dumps(record[name])) == previous[name], ('Pilot/main identity', sid, name)
            records.append(record)
            observer.reset([], False)
            del ids, post, packet, factors, d, g, u, a, phi, ap, ind, hstats, ustat, epacket, energy, unit_moments
            assert psutil.virtual_memory().available > config['host_floor_bytes']
            assert time.monotonic()-start < config['per_process_ceiling_seconds']
            if index < 2 or (index+1) % 32 == 0:
                print('OPERATOR_CAPTURE', mode, index+1, len(selected), 'elapsed', round(time.monotonic()-start, 1), 'bytes', usage(), flush=True)
        for key, (h, u) in accum.items():
            npz(out / 'moments' / (key + '.npz'), H_sums=h, unit_sums=u, counts=counts[key])
    finally:
        observer.close()
        del model, observer
        gc.collect()
        torch.cuda.empty_cache()
    seconds = time.monotonic()-start
    result = {'timestamp': stamp(), 'mode': mode, 'sources': len(records), 'tokens': sum(r['tokens'] for r in records),
        'seconds': seconds, 'retained_full_fields': [r['sample_id'] for r in records if r['full_field_retained']],
        'all_layer_every_token': True, 'all_coordinates_and_units': True, 'blocks_zero_index': BLOCKS,
        'class_names': CLASSES, 'condition_columns': '0..6 token-piece classes;7..22 four-bit causal-prefix lexical-cue masks',
        'H_moment_axes': ['layer0..36', 'sum_raw/sum_raw_square/sum_RMS/sum_RMS_square', 'condition0..22', 'coordinate0..2559'],
        'unit_moment_axes': ['block6,16,34', 'g/u/activation/phi/g2/u2/activation2/phi2/phi_times_u', 'condition0..22', 'unit0..9727'],
        'block_energy_row_order': ['input_energy', 'attention_energy', 'MLP_energy', '2input_attention', '2input_MLP', '2attention_MLP', 'output_energy', 'native_delta_energy', 'rounding_residual_energy'],
        'retention_scope': runtime['storage'], 'source': source}
    save(out / 'result.json', result)
    ledger('qwen4_all_token_all_layer_' + mode, seconds, sources=len(records), tokens=result['tokens'])
    if pilot:
        rate = sum(r['seconds'] for r in records) / sum(r['tokens'] for r in records)
        estimate_seconds = rate * sum(len(r['prompt_ids']) for r in material)
        # Deliberately conservative uncompressed estimates; client and operator outputs have a separate reserve.
        anchorbytes = sum(r['anchor_raw_bytes'] for r in records) / len(records) * len(material)
        fullbytes = sum(len(r['prompt_ids']) * 38 * 2560 * 2 for r in material if r['sample_id'] in keep)
        energybytes = sum(len(r['prompt_ids']) * (37+36*9+5) * 8 for r in material)
        total = anchorbytes + fullbytes + energybytes + 2 * 1024**3
        passed = estimate_seconds < config['per_process_ceiling_seconds'] and total < config['result_ceiling_bytes']
        report = {'timestamp': stamp(), 'passed': passed, 'pilot_sources': len(records), 'pilot_seconds': seconds,
            'estimated2048_native_capture_seconds': estimate_seconds, 'estimated_uncompressed_anchor_bytes': anchorbytes,
            'estimated_fixture_bytes': fullbytes, 'estimated_energy_bytes': energybytes, 'additional_outputs_reserve_bytes': 2 * 1024**3,
            'estimated_total_result_bytes': total, 'checks': [r['checks'] for r in records],
            'scope': 'Pilot includes repeated no-op/future checks; rate is conservative, not a completed full-run time.'}
        save(BASE / 'capture_pilot.json', report)
        assert passed, report
        print('OPERATOR_PILOT_PASSED', report, flush=True)
    else:
        print('OPERATOR_CAPTURE_COMPLETE', result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--confirmation', action='store_true')
    args = p.parse_args()
    main(args.pilot, args.confirmation)
