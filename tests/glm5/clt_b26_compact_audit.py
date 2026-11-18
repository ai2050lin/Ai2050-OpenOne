"""Audit frozen B26/H36 evidence and replay one example per shard on CPU."""
import json
import argparse
from pathlib import Path
import itertools
import time

import numpy as np
import torch
from torch.nn import functional as F

from clt_b26_compact import ARMS, BLOCK, CAP, COUNT, HEAD_SLICE, KV_SLICE, OUT, SEED, behavior, candidate, checked, commit
from source_binding_design import read, sha
from source_binding_runtime import Runtime


def truth_bits(x):
    assert x.dtype == np.uint16
    return torch.from_numpy(x.copy()).view(torch.bfloat16)


def audit_causal(report, names, old_names):
    path = OUT / 'causal-causal-summary.json'
    if not path.exists():
        return None
    assert report['causal_admitted'] and read(OUT/'calibration-causal-summary.json')['technical_pass']
    result = read(path)
    freeze = read(OUT/'causal-freeze.json')
    assert result['causal_freeze_sha'] == sha(OUT/'causal-freeze.json')
    assert freeze['prediction_sha'] == sha(OUT/'evaluation.json')
    assert freeze['causal_material_sha'] == sha(OUT/'causal-materials.json')
    materials = read(OUT/'causal-materials.json')
    rows = [r for r in materials if r['cell'] == (r['family']//2) % 16]
    control_names = {n for r in rows for n in r['names']}
    assert len(rows) == 128 and len(control_names) == 256 and not control_names & (names | old_names)
    assert set(r['family'] for r in rows) == set(range(128))
    assert np.all(np.bincount([r['cell'] for r in rows]) == 8)
    trace = OUT/'causal-causal-traces.npz'
    assert sha(trace) == result['traces_sha']
    with np.load(trace) as x:
        gold = np.array([r['target_record'] for r in rows])
        assert np.array_equal(x['ids'], [r['id'] for r in rows]) and np.array_equal(x['gold_record'], gold)
        dose = x['q_dose']
        ratios = dose[:, 2]/np.maximum(dose[:, 1],1e-12)
        matches = (ratios>=.9) & (ratios<=1.1)
        assert np.array_equal(matches,x['matched_q_dose']) and matches.mean() == result['dose_match_fraction']
        keys = ('q_actual', 'actual_head', 'actual_gua')
        measured_invariance = {}
        for key in keys:
            for arm, i in (('sham', 3), ('restore', 4)):
                condition = arm+'_'+key
                measured_invariance[condition] = np.array_equal(x[key][:,0].view(np.uint32), x[key][:,i].view(np.uint32))
                assert measured_invariance[condition] == result['bitwise'][condition]
        assert np.array_equal(x['applied_q_pre_norm'][:,1],x['applied_q_pre_norm'][:,4]) == result['touched_and_restored']
        assert (dose[:, 1] > 0).all() and float(x['sdpa_error'].max()) == result['max_sdpa_error']
        rr = np.arange(len(rows))

        def margin(key):
            a = x[key]
            return np.log(np.maximum(a[rr,:,gold],1e-20)) - np.log(np.maximum(a[rr,:,gold^1],1e-20))

        measured, predicted = margin('actual_mass'), margin('compact_mass')
        effect = measured[:,2]-measured[:,1]
        boot = np.random.default_rng(SEED+90).integers(0,len(rows),(10000,len(rows)))
        interval = np.quantile(effect[boot].mean(1),[.0125,.9875])
        assert float(effect.mean()) == result['matched_route_effect']
        assert np.array_equal(interval,result['matched_route_interval'])
        signs = np.sign(measured[:,1]-measured[:,0]) == np.sign(predicted[:,1]-predicted[:,0])
        assert float(signs.mean()) == result['margin_change_direction_agreement']
    return dict(analysis_sha=sha(path), trace_sha=sha(trace), all_arithmetic_checks_pass=True,
                independently_reproduced=['family selection', 'dose ratios', 'Q/head/GUA exact restoration', 'matched margin effect and interval', 'predicted effect sign'],
                scope='Saved measurements and arithmetic; final-logit bitwise identity was checked on CUDA during collection')


def audit():
    start = time.monotonic()
    torch.set_num_threads(4)
    prereg, seal = checked(), candidate()
    report = read(OUT / 'evaluation.json')
    assert report['candidate_sha'] == sha(OUT / 'candidate.json')
    assert report['collection_sha'] == sha(OUT / 'confirm-collection.json')
    assert seal['utc'] >= prereg['utc']
    rt = Runtime('qwen14')
    rotary = rt.Rotary(rt.cfg, device='cpu')
    parameters = np.load(OUT / 'predictor.npz')
    units = np.array(seal['units'])
    prefix = f'model.layers.{BLOCK}.'
    specs = {
        'q': ('self_attn.q_proj.weight', HEAD_SLICE, slice(None)),
        'k': ('self_attn.k_proj.weight', KV_SLICE, slice(None)),
        'v': ('self_attn.v_proj.weight', KV_SLICE, slice(None)),
        'o': ('self_attn.o_proj.weight', slice(None), HEAD_SLICE),
        'gate': ('mlp.gate_proj.weight', units, slice(None)),
        'up': ('mlp.up_proj.weight', units, slice(None)),
        'down': ('mlp.down_proj.weight', slice(None), units),
        'q_norm': ('self_attn.q_norm.weight', slice(None), None),
        'k_norm': ('self_attn.k_norm.weight', slice(None), None),
        'post_norm': ('post_attention_layernorm.weight', slice(None), None),
    }
    identity = {}
    for key, (name, row, column) in specs.items():
        weight = rt.weight(prefix + name, 'cpu')
        part = weight[row] if column is None else weight[row, column]
        bits = parameters['param_' + key]
        assert np.array_equal(bits, part.contiguous().view(torch.uint16).numpy()), key
        identity[name] = int(bits.size)

    def norm(weight):
        module = rt.Norm(len(weight), eps=rt.cfg.rms_norm_eps)
        module.weight = torch.nn.Parameter(weight, requires_grad=False)
        return module

    qnorm = norm(truth_bits(parameters['param_q_norm']))
    knorm = norm(truth_bits(parameters['param_k_norm']))
    post = norm(truth_bits(parameters['param_post_norm']))
    w = {key: truth_bits(parameters['param_' + key]).float() for key in ('q', 'k', 'v', 'o')}
    gate, up = [truth_bits(parameters['param_' + key]) for key in ('gate', 'up')]

    def replay(arrays, row):
        z = truth_bits(arrays['replay_z_example'])
        resid, other = truth_bits(arrays['replay_background_example'])[0]
        width = z.shape[1]
        pad = width - len(row['tokens'])
        pos = (torch.arange(width)[None, :] - pad).clamp_min(0)
        cos, sin = rotary(z, pos)
        def forward_local(projection):
            q, k = rt.mod.apply_rotary_pos_emb(qnorm(projection['q']).unsqueeze(1),
                                               knorm(projection['k']).unsqueeze(1), cos, sin)
            score = (q[0, 0, -1].float() * k[0, 0].float()).sum(-1) * (q.shape[-1] ** -.5)
            score[:pad] = float('-inf')
            mass = score.softmax(-1)
            record = torch.stack([mass[[pad+i for i in loc]].sum() for loc in row['records']])
            context = (mass[:, None] * projection['v'][0].float()).sum(0).bfloat16().float()
            predicted = w['o'] @ context
            x = post(resid + (other.float() + predicted).bfloat16())
            g, u = F.linear(x, gate), F.linear(x, up)
            act = F.silu(g) * u
            return record, context, predicted, torch.stack([g, u, act]).float()

        peak, compact = {}, {}
        branches = ('q', 'k', 'v')
        for arm in ARMS[:5]:
            projection = {}
            for key in branches:
                mean = torch.tensor(parameters['mu_' + key])
                x = z[:, -1:] if key == 'q' else z
                if arm == 'mean':
                    estimate = (w[key] @ mean).expand(1, x.shape[1], -1)
                else:
                    idx = parameters[f'{arm}_idx_{key}']
                    estimate = F.linear(x[..., idx].float()-mean[idx], w[key][:, idx]) + w[key] @ mean
                projection[key] = estimate.bfloat16()
            if arm == 'native128':
                compact = projection
            record, context, _, gua = forward_local(projection)
            gpu = torch.tensor(arrays['mlp_' + arm][0])
            peak[arm] = dict(mass_abs=float((record - torch.tensor(arrays['mass_' + arm][0])).abs().max()),
                             head_abs=float((context - torch.tensor(arrays['head_' + arm][0])).abs().max()),
                             GUA_abs=(gua-gpu).abs().amax(-1).tolist())
        full = {key: F.linear(z[:, -1:] if key == 'q' else z, w[key].bfloat16()) for key in branches}
        actual_mass = arrays['mass_actual'][0]
        actual_w = torch.tensor(arrays['write_actual'][0])
        actual_u = torch.tensor(arrays['mlp_actual'][0, 2])
        mixed = {}
        for mode in itertools.product((False, True), repeat=3):
            label = ''.join(k.upper() for k, restored in zip(branches, mode) if restored) or 'none'
            variant = {key: (full if exact else compact)[key] for key, exact in zip(branches, mode)}
            mass, _, write, gua = forward_local(variant)
            mixed[label] = dict(mass_l1=float((mass-torch.tensor(actual_mass)).abs().sum()),
                                route_agree=bool(mass.argmax().item() == actual_mass.argmax()),
                                write_mse=float((write-actual_w).square().mean()),
                                activation_mse=float((gua[2]-actual_u).square().mean()))
        return peak, mixed

    splits, names, replay_results = {}, {}, []
    arm_squared = {arm: np.zeros(3) for arm in ARMS}
    route_by_cell = np.zeros(16, dtype=np.int64)
    earlier = set()
    for folder, partitions in [('conditional_lookup_native_v1', ('pilot', 'discovery', 'confirm')),
                               ('clt_b26_native_v1', ('discovery', 'confirm'))]:
        for split in partitions:
            earlier.update(n for row in read(OUT.parent / folder / f'{split}-materials.json') for n in row['names'])
    for split in ('discovery', 'confirm'):
        path = OUT / f'{split}-materials.json'
        assert sha(path) == prereg['materials'][split]['sha']
        rows = read(path)
        assert len(rows) == COUNT[split] * 16 == len({r['id'] for r in rows})
        names[split] = {n for row in rows for n in row['names']}
        assert len(names[split]) == COUNT[split] * 2
        assert not names[split] & earlier
        collection = read(OUT / f'{split}-collection.json')
        scores, prefix_count = [], 0
        route_n, selected_n = 0, 0
        for file, digest in sorted(collection['shards'].items()):
            assert sha(OUT / file) == digest
            shard = read(OUT / file.replace('.npz', '.json'))
            assert shard['array_sha'] == digest
            assert shard['prereg_sha'] == sha(OUT / 'prereg.json')
            assert shard['seal_sha'] == (sha(OUT / 'candidate.json') if split == 'confirm' else None)
            assert shard['head_max'] <= CAP
            batch = rows[len(scores):len(scores)+len(shard['scores'])]
            for row, score in zip(batch, shard['scores'], strict=True):
                cell = row['cell']
                r0, r1, identity_bit, cue = ((cell >> bit) & 1 for bit in (3, 2, 1, 0))
                gold = cue * 2 + ((r0, r1)[cue] ^ identity_bit)
                values = score['scores']
                assert row['target_record'] == gold and row['id'] == score['id']
                assert score['gold_margin'] == values[gold] - max(v for i, v in enumerate(values) if i != gold)
                assert score['top1_correct'] == (score['top1'] == row['gold_id'])
            scores.extend(shard['scores'])
            prefix_count += shard['prefix_equal']
            if split == 'confirm':
                with np.load(OUT / file) as data:
                    masses = data['mass_actual']
                    observed = masses.argmax(-1)
                    route_n += sum(a == b['target_record'] for a, b in zip(observed, batch, strict=True))
                    agreement = observed == data['mass_native128'].argmax(-1)
                    selected_n += int(agreement.sum())
                    route_by_cell += agreement.reshape(-1, 16).sum(0)
                    write = data['write_actual'].astype(np.float64)
                    actual_a = data['mlp_actual'].astype(np.float64)
                    for arm in ARMS:
                        arm_squared[arm] += np.square(data['mlp_' + arm] - actual_a).sum(axis=(0, 2))
                    for arm in ARMS[:5]:
                        estimate = data['head_' + arm].astype(np.float64) @ w['o'].numpy().astype(np.float64).T
                        errors = np.square(estimate - write).mean(-1)
                        assert np.allclose(errors, data['write_se_' + arm], atol=1e-8, rtol=3e-5)
                    with torch.inference_mode():
                        differences, mixed = replay(data, batch[0])
                        replay_results.append(dict(id=batch[0]['id'], cell=batch[0]['cell'], differences=differences,
                                                   offline_restore_full_input=mixed))
            print(json.dumps(dict(stage='audit', split=split, file=file, completed=len(scores), total=len(rows),
                                  elapsed=time.monotonic()-start), ensure_ascii=False), flush=True)
        assert prefix_count == len(rows) // 2
        actual_cells, passes = behavior(rows, scores)
        assert collection['behavior_pass'] == passes and actual_cells == collection['cells']
        if split == 'confirm':
            assert route_n / len(rows) == report['natural_gold_route']
            assert selected_n / len(rows) == report['arms']['native128']['route_agreement']
            assert np.array_equal(route_by_cell / COUNT[split], report['arms']['native128']['by_cell_route_agreement'])
            for arm in ARMS:
                assert np.allclose(arm_squared[arm]/(len(rows)*len(units)), report['arms'][arm]['mlp_mse_by_GUA'], atol=1e-12, rtol=1e-8)
        splits[split] = dict(n=len(rows), pairs=prefix_count, behavior_pass=passes,
                             strict=sum(r['gold_margin'] > 0 for r in scores), top1=sum(r['top1_correct'] for r in scores))
    assert not names['discovery'] & names['confirm']
    causal = audit_causal(report, names['discovery'] | names['confirm'], earlier)
    commit('audit.json', dict(provenance=prereg['provenance'], candidate_sha=sha(OUT / 'candidate.json'),
                              eval_sha=sha(OUT / 'evaluation.json'), splits=splits, native_scalar_identity=identity,
                              scalars=sum(identity.values()), cpu_replay=replay_results,
                              auditor_sha=sha(Path(__file__)), disjoint_from_previous_families=True,
                              causal=causal,
                              offline_branch_scope='Exploratory CPU replay using full inputs in selected branches on the first input in each shard. This is surrogate arithmetic, not a causal manipulation of the model. Each example is cell 0.',
                              note='CPU replay errors describe cross-device arithmetic, not independent model forwards'))
    print('Completed independent evidence audit.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--causal-only', action='store_true')
    if parser.parse_args().causal_only:
        baseline = read(OUT/'audit.json')
        assert baseline['eval_sha'] == sha(OUT/'evaluation.json')
        current = {n for split in ('discovery','confirm') for row in read(OUT/f'{split}-materials.json') for n in row['names']}
        earlier = {n for directory, splits in [('conditional_lookup_native_v1', ('pilot','discovery','confirm')),
                                                ('clt_b26_native_v1', ('discovery','confirm'))]
                   for split in splits for row in read(OUT.parent/directory/f'{split}-materials.json') for n in row['names']}
        verified = audit_causal(read(OUT/'evaluation.json'), current, earlier)
        assert verified
        commit('causal-audit.json', dict(**verified, provenance=baseline['provenance'], auditor_sha=sha(Path(__file__))))
        print('Completed independent intervention evidence audit.', flush=True)
    else:
        audit()
