"""Freeze discoveries and analyze prespecified paired endpoints and recursion failure."""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest

import mlp_b34_confirm_design as D
from shared.dist_protocol.mechanism_trace import unbits
from tools.power_analysis import check_phase

OUT, READ, SAVE, HASH = D.OUT, D.READ, D.SAVE, D.HASH


def chunks(plan, split):
    path = OUT / f'{split}-materials.json'
    if HASH(path) != plan['inputs'][split]['sha256']:
        raise ValueError('Material digest changed')
    items = READ(path)['items']
    for start in range(0, len(items), 16):
        name = f'{split}-{start:04d}'
        meta = READ(OUT/f'{name}.json')
        path = OUT/f'{name}.npz'
        if meta['raw_sha256'] != HASH(path) or meta['plan_sha256'] != HASH(OUT/'prereg.json'):
            raise ValueError(f'Checkpoint identity mismatch: {name}')
        if split == 'holdout' and meta['selection_sha256'] != HASH(OUT/'selection.json'):
            raise ValueError(f'Checkpoint selection mismatch: {name}')
        sample = items[start:start+16]
        assert [r['id'] for r in sample] == [r['id'] for r in meta['rows']]
        yield sample, meta['rows'], path


def select(plan):
    path = OUT/'selection.json'
    if path.exists() or list(OUT.glob('holdout-0*.npz')):
        raise RuntimeError('Selection frozen or holdout already opened')
    rng = np.random.default_rng(D.SEED)
    deltas = {b: [] for b in (34, 35)}
    correct = []
    raw_index = []
    for items, rows, raw in chunks(plan, 'discovery'):
        with np.load(raw) as bank:
            for b in deltas:
                a = unbits(bank[f'A{b}'])
                deltas[b].append(a[:, 1]-a[:, 0])
        correct.extend(np.all(np.asarray([r['base_scores'] for r in rows]).argmax(-1) == [s['correct'] for s in items], axis=1).tolist())
        raw_index.extend({'path': p.name, 'sha256': HASH(p)} for p in (raw, raw.with_suffix('.json')))
    rate = float(np.mean(correct))
    selected = {'provenance': plan['provenance'], 'plan_sha256': HASH(OUT/'prereg.json'), 'discovery_chunks': raw_index,
                'discovery_both_correct': rate, 'baseline_gate': rate >= plan['calibration']['pair_base_accuracy_min'],
                'frozen_at': datetime.now().astimezone().isoformat()}
    for b, values in deltas.items():
        diff = np.concatenate(values)
        mean = diff.mean(axis=0, dtype=np.float64)
        rms = np.maximum(np.sqrt(np.square(diff, dtype=np.float64).mean(axis=0)), 1e-6)
        units = np.argsort(-abs(mean)/rms, kind='stable')[:D.GROUP]
        control = rng.choice(np.setdiff1d(np.arange(D.atlas.WIDTH), units), D.GROUP, replace=False)
        selected[f'B{b}'] = {'units': units.tolist(), 'write': mean[units].tolist(),
            'orientation': np.sign(mean[units]).tolist(), 'scale': rms[units].tolist(),
            'control': control.tolist(), 'control_orientation': np.sign(mean[control]).tolist(), 'control_scale': rms[control].tolist()}
    SAVE(path, selected)
    print({'selection_sha256': HASH(path), 'both_correct': rate, 'baseline_gate': selected['baseline_gate']}, flush=True)


def paired(edited, control, alpha=D.ALPHA_PRIMARY):
    edited, control = np.asarray(edited, dtype=bool), np.asarray(control, dtype=bool)
    better, worse = int((edited & ~control).sum()), int((~edited & control).sum())
    n = len(edited)
    return {'n': n, 'edited_rate': float(edited.mean()) if n else None, 'control_rate': float(control.mean()) if n else None,
            'gain': (better-worse)/n if n else None, 'edit_only': better, 'control_only': worse,
            'p': binomtest(better, better+worse).pvalue if better+worse else 1., 'alpha': alpha}


def semantic_test(diff, items, selection, plan):
    output = {}
    reversed_codes = np.array([s['reverse_code'] for s in items])
    frame = np.array([s['frame'] for s in items])
    for b in (34, 35):
        card = selection[f'B{b}']
        z = diff[b][:, card['units']] * card['orientation'] / card['scale']
        random = diff[b][:, card['control']] * card['control_orientation'] / card['control_scale']
        group_hit = z.mean(1) > 0
        group = paired(group_hit, random.mean(1) > 0)
        group['by_frame'] = {name: float(group_hit[frame == name].mean()) for name in sorted(set(frame))}
        group['by_code'] = [float(group_hit[reversed_codes == value].mean()) for value in (False, True)]
        group['passed'] = (group['p'] <= group['alpha'] and group['gain'] >= .05 and group['edited_rate'] >= .62
                           and min(group['by_code']) >= .55)
        units = []
        for i, unit in enumerate(card['units']):
            hit = z[:, i] > 0
            n, count = len(hit), int(hit.sum())
            # Ties are kept as non-hits, making a positive directional claim conservative.
            p = binomtest(count, n, .5, alternative='greater').pvalue
            by_code = [float(hit[reversed_codes == value].mean()) for value in (False, True)]
            units.append({'unit': unit, 'orientation': card['orientation'][i], 'n': n, 'hits': count, 'rate': count/n,
                          'ties': int((z[:, i] == 0).sum()), 'p': p, 'by_code': by_code,
                          'by_frame': {name: float(hit[frame == name].mean()) for name in sorted(set(frame))},
                          'passed': count/n >= .62 and min(by_code) >= .55 and p <= plan['alpha_unit']})
        output[f'B{b}'] = {'group': group, 'units': units, 'failed_group_ids': [s['id'] for s, hit in zip(items, group_hit) if not hit]}
    return output


def causal_test(items, rows, selection, baseline_gate):
    results, counter = [], []
    target = selection['B35']
    for i, dose in enumerate(D.DOSES):
        edit_ok, null_ok, margin, null_margin, rescue_ok, sham_ok = [], [], [], [], [], []
        downstream, downstream_null, target_deltas, control_deltas, rescue_margin, sham_margin = [], [], [], [], [], []
        for s, r in zip(items, rows):
            record = r['doses'][i]
            rec = s['recipient_index']
            gold, desired = s['correct'][rec], s['correct'][1-rec]
            base = np.asarray(r['base_scores'][rec]); edit = np.asarray(record['edit']['scores']); control = np.asarray(record['control']['scores'])
            hit, null_hit = (int(x.argmax()) == desired and int(base.argmax()) == gold for x in (edit, control))
            if dose == D.PRIMARY and (not hit or null_hit or not record['norm_valid']):
                counter.append({'sample_id': s['id'], 'frame': s['frame'], 'recipient': rec, 'gold_code': 'AB'[gold],
                    'desired_code': 'AB'[desired], 'base_scores': base.tolist(), 'edit_scores': edit.tolist(), 'control_scores': control.tolist(),
                    'reason': 'norm_invalid' if not record['norm_valid'] else 'control_also_switches' if null_hit else 'no_desired_switch'})
            if not record['norm_valid']:
                continue
            edit_ok.append(hit); null_ok.append(null_hit)
            margin.append(float((edit[desired]-edit[gold])-(base[desired]-base[gold])))
            null_margin.append(float((control[desired]-control[gold])-(base[desired]-base[gold])))
            orientation = (1-2*rec) * np.asarray(target['orientation']) / target['scale']
            downstream.append(float(np.dot(record['edit']['target_delta'], orientation)/D.GROUP))
            downstream_null.append(float(np.dot(record['control']['target_delta'], orientation)/D.GROUP))
            target_deltas.append(np.asarray(record['edit']['target_delta'])*(1-2*rec))
            control_deltas.append(np.asarray(record['control']['target_delta'])*(1-2*rec))
            if dose == D.PRIMARY and record['rescue']['norm_valid']:
                restore = record['rescue']
                rescued, sham = (np.asarray(restore[key]) for key in ('scores', 'sham_scores'))
                valid_switch = bool(hit)
                rescue_ok.append(valid_switch and int(rescued.argmax()) == gold)
                sham_ok.append(valid_switch and int(sham.argmax()) == gold)
                rescue_margin.append(float((rescued[gold]-rescued[desired])-(edit[gold]-edit[desired])))
                sham_margin.append(float((sham[gold]-sham[desired])-(edit[gold]-edit[desired])))
        result = {'dose': dose, 'norm_invalid': len(items)-len(edit_ok), 'switch': paired(edit_ok, null_ok),
                  'edit_margin': average(margin), 'control_margin': average(null_margin),
                  'target_shift': average(downstream), 'control_target_shift': average(downstream_null),
                  'target_unit_mean_deltas': np.mean(target_deltas, axis=0).tolist() if target_deltas else [],
                  'control_unit_mean_deltas': np.mean(control_deltas, axis=0).tolist() if control_deltas else []}
        if dose == D.PRIMARY:
            result['rescue'] = paired(rescue_ok, sham_ok)
            result['rescue_margin'] = average(rescue_margin)
            result['sham_margin'] = average(sham_margin)
            for name in ('switch', 'rescue'):
                stat = result[name]
                stat['realized_power'] = check_phase(.10, stat['n'], .35, D.ALPHA_PRIMARY)
                stat['passed'] = (baseline_gate and stat['realized_power']['pass'] and stat['p'] <= stat['alpha']
                                  and stat['gain'] is not None and stat['gain'] >= .05)
        results.append(result)
    return results, counter


def average(values):
    return float(np.mean(values)) if len(values) else None


def capability_test(plan):
    doc = READ(OUT/'capability.json')
    rows = doc['rows']
    assert len(rows) == plan['capability']['n'] and doc['provenance'] == plan['provenance']
    assert doc['selection_sha256'] == HASH(OUT/'selection.json') and doc['materials_sha256'] == HASH(OUT/'capability-materials.json')
    base = np.array([np.argmax(r['scores']['base']) == r['gold'] for r in rows])
    result = {}
    for sign in ('positive', 'negative'):
        edited = np.array([np.argmax(r['scores'][sign]) == r['gold'] for r in rows])
        comparison = paired(edited, base, .025)
        loss = comparison['control_only']
        upper = float(beta.ppf(.975, loss+1, len(rows)-loss)) if loss < len(rows) else 1.
        comparison.update(loss_upper=upper, margin=.02, passed=upper <= .02,
            by_source={s: paired(edited[ix], base[ix], .025) for s in sorted({r['source'] for r in rows})
                       if (ix := np.array([r['source'] == s for r in rows])).any()})
        result[sign] = comparison
    return result


def silu(x):
    return x * np.exp(-np.logaddexp(0, -x))


def predict(x, par, field):
    z = (x - par['xmean']) / par['xstd']
    return np.einsum('nuk,uk->nu', z[:, par['parents']], par[field+'_coef']) + par[field+'_mean']


def squared_rows(pred, truth):
    return np.square(pred-truth, dtype=np.float64).sum(axis=1)


def replay_setup():
    import torch
    from safetensors import safe_open

    torch.set_num_threads(4)
    weights = D.atlas.weights(35)
    index = READ(D.MODEL/'model.safetensors.index.json')['weight_map']
    key = 'model.layers.35.post_attention_layernorm.weight'
    with safe_open(D.MODEL/index[key], framework='pt', device='cpu') as f:
        scale = f.get_tensor(key)
    eps = READ(D.MODEL/'config.json')['rms_norm_eps']

    def run(fields):
        import torch.nn.functional as F

        tensors = {k: torch.from_numpy(fields[k].copy()).view(torch.bfloat16) for k in ('R35','H34','W34','T35')}
        resid, h34, w34, attn = (tensors[k] for k in ('R35','H34','W34','T35'))
        reconstructed = h34+attn
        if not torch.equal(reconstructed, resid):
            raise RuntimeError('B35 residual + attention identity failed')
        output = {}
        with torch.inference_mode():
            for key, value in [('native_replay', reconstructed), ('without_attention', h34),
                               ('without_carried_residual', w34+attn), ('write_only', w34)]:
                f = value.float()
                z = (f * torch.rsqrt(f.square().mean(-1, keepdim=True)+eps)).to(torch.bfloat16) * scale
                output[key] = (F.silu(F.linear(z, weights['gate'])) * F.linear(z, weights['up'])).float().numpy()
        return output
    return run


def recursion(bank_chunks, plan):
    params = {}
    for b in (32, 33, 34):
        name = f'B{b}-B{b+1}-predictor.npz';path = D.atlas.OUT/name
        if HASH(path) != plan['historical_predictors'][name]:
            raise ValueError('Historical predictor changed')
        with np.load(path) as loaded:
            params[b] = {k: loaded[k] for k in ('parents','xmean','xstd','A_coef','A_mean','G_coef','G_mean','V_coef','V_mean')}
    replay = replay_setup()
    metrics = {b: {} for b in (33, 34, 35)}
    ids, denominator = [], {b: [] for b in (33, 34, 35)}
    z_stats = {'true_B34': [], 'recursive_composed_B34': []}
    for items, _, path in bank_chunks:
        ids.extend(s['id'] for s in items)
        with np.load(path) as raw:
            fields = {k: raw[k].reshape(len(items)*2, -1) for k in raw.files}
        a = {b: unbits(fields[f'A{b}']) for b in (32, 33, 34, 35)}
        direct = composed = a[32]
        for b, par in params.items():
            y = a[b+1]
            denominator[b+1].extend(squared_rows(par['A_mean'], y).reshape(-1, 2).sum(axis=1))
            one_g, one_v = predict(a[b], par, 'G'), predict(a[b], par, 'V')
            recursive_g, recursive_v = predict(composed, par, 'G'), predict(composed, par, 'V')
            if b == 34:
                for k, arr in [('true_B34', a[b]), ('recursive_composed_B34', composed)]:
                    z_stats[k].extend((np.abs((arr-par['xmean'])/par['xstd']) > 10).mean(axis=1).tolist())
            direct = predict(direct, par, 'A')
            composed = silu(recursive_g)*recursive_v
            predictions = {'one_step_direct': predict(a[b], par, 'A'), 'one_step_composed': silu(one_g)*one_v,
                           'recursive_direct': direct, 'recursive_composed': composed}
            if b == 34:
                g, v = unbits(fields['G35']), unbits(fields['V35'])
                predictions.update({'true_gate_one_step': silu(g)*one_v, 'true_value_one_step': silu(one_g)*v,
                    'true_gate_recursive': silu(g)*recursive_v, 'true_value_recursive': silu(recursive_g)*v,
                    'true_gate_and_value_fp32': silu(g)*v, **replay(fields)})
            for k, x in predictions.items():
                metrics[b+1].setdefault(k, []).extend(squared_rows(x, y).reshape(-1, 2).sum(axis=1))
    result = {}
    for b, methods in metrics.items():
        den = sum(denominator[b]); result[f'B{b}'] = {}
        for name, sse in methods.items():
            errors = np.asarray(sse)
            order = np.argsort(errors)[::-1]
            total = errors.sum()
            result[f'B{b}'][name] = {'fvu': float(total/den), 'top_1pct_error_share': float(errors[order[:max(1,len(ids)//100)]].sum()/total) if total else 0.,
                'worst': [{'id': ids[i], 'sse': float(errors[i])} for i in order[:5]]}
    return {'metrics': result, 'n_pairs': len(ids), 'standardized_B34_fraction_above_10': {k: average(v) for k, v in z_stats.items()},
            'native_replay_gate': result['B35']['native_replay']['fvu'] <= plan['calibration']['native_boundary_rebuild_fvu_max']}


def report(plan):
    selection = READ(OUT/'selection.json')
    bank = list(chunks(plan, 'holdout'))
    items = [x for ss, _, _ in bank for x in ss]
    rows = [r for _, rs, _ in bank for r in rs]
    assert len(items) == len(rows) == D.N_HOLDOUT
    diff = {34: [], 35: []}
    for _, _, path in bank:
        with np.load(path) as data:
            for b in diff:
                arr = unbits(data[f'A{b}']);diff[b].append(arr[:, 1]-arr[:, 0])
    base = np.asarray([r['base_scores'] for r in rows]).argmax(-1) == [s['correct'] for s in items]
    baseline = float(base.all(axis=1).mean())
    baseline_gate = baseline >= plan['calibration']['pair_base_accuracy_min']
    semantics = semantic_test({b: np.concatenate(v) for b, v in diff.items()}, items, selection, plan)
    doses, counter = causal_test(items, rows, selection, baseline_gate)
    cap = capability_test(plan)
    print('analysis: paired tests complete; replaying B35 and historical predictors', flush=True)
    diagnostic = recursion(bank, plan)
    primary = next(r for r in doses if r['dose'] == D.PRIMARY)
    behavior = primary['switch']['passed']
    path = (baseline_gate and behavior and primary['rescue']['passed'] and diagnostic['native_replay_gate']
            and all(s['group']['passed'] for s in semantics.values()) and all(c['passed'] for c in cap.values()))
    worst = {d['id'] for d in diagnostic['metrics']['B35']['recursive_composed']['worst']}
    snippets = {s['id']: {'pair_prompts': s['prompts'], 'frame': s['frame'], 'gold_codes': ['AB'[k] for k in s['correct']]}
                for s in items if s['id'] in worst | {c['sample_id'] for c in counter[:6]}}
    raw = [{'path': p.name, 'sha256': HASH(p), 'bytes': p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'report.json']
    result = {'schema': 'mlp_b34_confirm_report.v1', 'kind': 'real', 'run_id': D.RUN, 'model_snapshot': D.SNAPSHOT,
        'provenance': plan['provenance'], 'plan_sha256': HASH(OUT/'prereg.json'), 'selection_sha256': HASH(OUT/'selection.json'),
        'discovery_pairs': D.N_DISCOVERY, 'holdout_pairs': len(items), 'holdout_both_correct': baseline, 'baseline_gate': baseline_gate,
        'semantic': semantics, 'dose_curve': doses, 'capability': cap, 'diagnosis': diagnostic, 'controlled_behavior_passed': behavior,
        'mediation_passed': path, 'counterexamples': counter, 'counterexample_materials': snippets, 'raw_manifest': raw,
        'scope': 'Fresh authored English role-swap records, fixed A/B scoring at one token, and historical keyed capability monitor. Single-node study.'}
    SAVE(OUT/'report.json', result)
    print({'baseline': baseline, 'semantic_units': {k: sum(x['passed'] for x in v['units']) for k, v in semantics.items()},
           'groups': {k: v['group'] for k, v in semantics.items()}, 'primary': {k: primary[k] for k in ('switch','rescue')},
           'capability': {k: (v['control_only'],v['loss_upper'],v['passed']) for k,v in cap.items()},
           'B35': {k: v['fvu'] for k,v in diagnostic['metrics']['B35'].items()},
           'path': path}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['select', 'report'])
    args = parser.parse_args()
    plan = D.checked_plan()
    (select if args.stage == 'select' else report)(plan)
