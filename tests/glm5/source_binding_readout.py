"""An edit-free predictor of the block-24 transition from the three block-25 heads (Claude Phase 418).  Qwen3-14B BF16.

Phase 417: the score x after a narrow exchange in heads 25.18, 25.24, 25.15 predicts whether the broad name-token
exchange at block 24 moves the answer (AUC 0.905).  x still needs an edited forward pass.  Here x is replaced by a
linear read-out of quantities computed from the two UNEDITED runs (base and donor) at the input of block 25:

  z_h      output of head h at the answer position before the output projection (128 numbers per head)
  obs      z(donor run) - z(base run): what the heads write in the donor prompt minus what they write in the base prompt
  recomb   z recomputed with the base query and the donor keys / values of the name tokens and the holder positions,
           minus z(base): what the heads would write, with no later block run
  scalars  attention mass of the heads on names / holders (base, recombined), norms of recomb, the unedited score

A ridge regression of x on a feature set is fitted on the opened panel holdout_per (Phase 417), checked on the opened
panel holdout_h25 (Phase 415), frozen, and then scored on a new panel: x_hat > T (the Phase 417 threshold) predicts
that the broad exchange moves the answer.
Stages: collect_open, fit (opened panels) -> prereg -> gate -> run.
"""
import argparse, gc, hashlib, json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
import source_binding_relay as RL
import source_binding_pointer as PT
import source_binding_nameloc as NL
import source_binding_fidelity3 as F3
import source_binding_arbitration as AR
import source_binding_h2518 as H25
import source_binding_peredit as PE
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
H25_OUT, PE_OUT = H25.OUT, PE.OUT
OUT = D.ROOT / 'tests/glm5/result/source_binding_readout_v1'
RUN, SEED = 'SOURCE-BINDING-READOUT-20261112-v1', 202611120418
SPLIT, FAMILIES, REPS, CHUNK, PB = 'holdout_ro', 64, 4, 192, 25
C, T0 = H25.C, -7.75
SETS = ['recomb', 'obs', 'scalars', 'recomb+obs', 'all']
LAMBDAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
PRIMARY = 'all'       # feature set, chosen on the opened panels before pre-registration
OTHER_PANELS = PE.OTHER_PANELS + ['source_binding_peredit_v1/holdout_per']
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'layout-factorial families as in Phase 405. holdout_ro: 64 families (1,024 prompts; 64 so that the contrast with the class rule has power >= 0.8 with one edit counted per item). Entity names occur in no earlier '
                'panel of this line; the 16 discovery code words, 4 seen frames.',
    'features': 'computed at the input of block 25 from the unedited runs of the base and the donor prompt, heads 25.18, 25.24, 25.15, '
                'attention row of the answer position; no block after the attention of block 25 is run for them. obs = z(donor) - '
                'z(base); recomb = z(base query, donor keys / values of name tokens and holders) - z(base); z is the head output '
                'before the output projection (3 x 128 numbers).',
    'read_out': 'x_hat = unedited score(P) - score(B) plus a ridge regression of the shift (x minus that unedited score) on the '
                'standardised features, fitted on all conflict edits of the opened '
                'panel holdout_per; lambda chosen by 6-fold family cross-validation there; weights frozen in readout.npz. The broad '
                'exchange is predicted to move the answer iff x_hat > -7.75 (the Phase 417 threshold, not refitted).',
    'broad_edit': 'as in Phases 412-417: both name tokens exchanged as residual states at the input of block 24; 4 seeded donors per item.',
    'comparisons': {'x': 'the Phase 417 predictor (needs the narrow edit), run on the same donors', 'class rule': 'P iff P = I',
                    'none': 'the answer never moves', 'natural score': 'score(P) - score(B) of the unedited base run'},
    'outcome': 'on conflict edits: AUC of x_hat for follow (overall and inside each class), accuracy of the rule, correlation of x_hat '
               'with x; family-bootstrap 95 % intervals.',
    'controls': 'as in Phase 417 for the edited arms. The base forward of the feature pass is the unedited run.',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed.',
    'predictions_on_holdout': {
        'U1_edit_free_ranking': 'primary read-out (all features): AUC >= 0.72 and AUC minus the AUC of the unedited score has a lower '
                                'bound > 0. not_supported if AUC < 0.68 or the lower bound <= 0.',
        'U2_edit_free_rule': 'accuracy of x_hat > -7.75 is >= 0.68 and exceeds the class rule with a lower bound > 0. not_supported if '
                             'the lower bound <= 0.',
        'U3_linear_read_out_falls_short': 'AUC of x (the narrow edit) minus AUC of x_hat has a lower bound > 0.05. not_supported if '
                                          'the lower bound <= 0 (the read-out would then be as good as the edit).'},
    'disclosure': 'feature sets, the target (shift of the score, added to the unedited score) and the primary set were chosen on the '
                  'opened panels: the first fit regressed x itself and reached AUC 0.73-0.76; refitting on the shift gave 0.74-0.78. '
                  'On holdout_per the read-out is fitted (cross-validated AUC 0.773 for the primary set); holdout_h25 was used as a '
                  'check (0.782) and for nothing else. The expected result is therefore a partial success: clearly above the unedited '
                  'score (0.64-0.65) and clearly below the narrow edit (0.89-0.90); U3 states the shortfall as a prediction.',
    'not_covered': 'why the head outputs differ between items (name-token states, attention); a read-out without a fitted map; single '
                   'name sites, blocks 23 and 25; other tasks and models. A fitted linear read-out shows that the information is '
                   'linearly present in what the heads write, not how later blocks use it.',
}


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定免编辑读出产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def build_materials():
    taken = {n for rows in D.read(SRC / f'{MODEL}-materials.json')['splits'].values() for r in rows for n in r['names']}
    for p in PT.PANELS + [PT.OUT / 'pilot_layout-materials.json', PT.OUT / 'holdout_layout-materials.json',
                          NL.OUT / 'pilot_name-materials.json',
                          NL.OUT / 'holdout_name-materials.json'] + [D.ROOT / 'tests/glm5/result' / f'{x}-materials.json' for x in OTHER_PANELS]:
        taken |= {n for r in D.read(p) for n in r['names']}
    PT.SPLITS[SPLIT] = FAMILIES
    return PT.build_materials(SPLIT, taken)


def final_arms(rows):
    H25.RUN, H25.SEED = RUN, SEED
    dons, rels = H25.draws(rows)
    both = [['names', C], ['holders', C]]
    arms = {'sole': {'hide': H25.SOLE}, 'self:NH:C': {'hide': [], 'patch': both, 'donor': list(range(len(rows)))}}
    for k, don in enumerate(dons):
        arms[f'swap#{k}'] = {'hide': [], 'swap': True, 'donor': don, 'rel': rels[k]}
        arms[f'x|NH:C#{k}'] = {'hide': [], 'patch': both, 'donor': don, 'rel': rels[k]}
    return arms


def collect(rows, donors, stage):
    """Features of every (item, donor) pair from the unedited runs; the forward stops inside block 25."""
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    t = torch
    assert snapshot(D.MODELS[MODEL]) == D.read(SRC / f'{MODEL}-identity.json')['model_snapshot']
    RL.OUT = OUT
    start, total = time.monotonic(), len(rows)
    feats = {k: [[] for _ in donors] for k in ('recomb', 'obs', 'scalars')}
    with GpuLease(), t.inference_mode():
        rt = Runtime(MODEL)
        hd, kv, grp = rt.hd, rt.kv, rt.heads // rt.kv
        rot = lambda x, co, si: x * co + rt.mod.rotate_half(x) * si
        RL.log(stage, 0, total * (PB + 1), start, rows=total)
        for c0 in range(0, total, CHUNK):
            rs = rows[c0:c0 + CHUNK]
            N = len(rs)
            ids, pos, pe, mask = rt.context(rs)
            h, length = rt.embed(ids), ids.shape[1]
            off = [length - len(r['tokens']) for r in rs]
            cos, sin = pe
            for b in range(PB):
                layer = rt.block(b)
                h = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin), use_cache=False)
                del layer
                gc.collect()
                t.cuda.empty_cache()
                if (b + 1) % 5 == 0:
                    free, tot = t.cuda.mem_get_info()
                    RL.log(stage, c0 * (PB + 1) + (b + 1) * N, total * (PB + 1), start, block=b, device_used_MiB=round((tot - free) / 2 ** 20))
            layer = rt.block(PB)
            a = layer.self_attn
            hn = layer.input_layernorm(h)
            q = a.q_norm(a.q_proj(hn[:, -1:]).view(N, 1, rt.heads, hd)).transpose(1, 2)[:, C]
            q = rot(q, cos[:, None, -1:], sin[:, None, -1:]).float()
            k = rt.mod.repeat_kv(a.k_norm(a.k_proj(hn).view(N, length, kv, hd)).transpose(1, 2), grp)[:, C]
            v = rt.mod.repeat_kv(a.v_proj(hn).view(N, length, kv, hd).transpose(1, 2), grp)[:, C].float()
            ok = mask[:, 0, -1, :][:, None, None]

            def flags(fn):
                m = t.zeros((N, length), dtype=t.bool, device='cuda')
                for i, r in enumerate(rs):
                    m[i, [off[i] + j for j in fn(r)]] = True
                return m
            mn, mh = flags(AR.names_of), flags(AR.holders_of)

            def heads(kk, vv):
                w = (q @ rot(kk, cos[:, None], sin[:, None]).float().transpose(2, 3) / hd ** .5).masked_fill(~ok, float('-inf')).softmax(-1)
                return (w @ vv)[:, :, 0], (w[:, :, 0] * mn[:, None]).sum(-1), (w[:, :, 0] * mh[:, None]).sum(-1)
            z, an, ah = heads(k, v)
            hs = t.arange(len(C), device='cuda')[None]
            for rep, don in enumerate(donors):
                local = [j - c0 for j in don[c0:c0 + N]]
                assert all(0 <= j < N and rs[j]['family'] == r['family'] for j, r in zip(local, rs))
                k2, v2 = k.clone(), v.clone()
                for fn in (AR.names_of, AR.holders_of):
                    bi, bp, di, dp = (t.tensor(x, device='cuda') for x in (
                        [i for i, r in enumerate(rs) for _ in fn(r)], [off[i] + x for i, r in enumerate(rs) for x in fn(r)],
                        [j for j in local for _ in fn(rs[j])], [off[j] + y for j in local for y in fn(rs[j])]))
                    assert len(bi) == len(di)
                    k2[bi[:, None], hs, bp[:, None]] = k[di[:, None], hs, dp[:, None]]
                    v2[bi[:, None], hs, bp[:, None]] = v[di[:, None], hs, dp[:, None]]
                z2, an2, ah2 = heads(k2, v2)
                j = t.tensor(local, device='cuda')
                rec, obs = z2 - z, z[j] - z
                sc = t.cat([an, ah, an2, ah2, an2 - an, ah2 - ah, rec.norm(dim=-1), obs.norm(dim=-1)], 1)
                feats['recomb'][rep].append(rec.reshape(N, -1).cpu().numpy())
                feats['obs'][rep].append(obs.reshape(N, -1).cpu().numpy())
                feats['scalars'][rep].append(sc.cpu().numpy())
                del k2, v2
            del layer, a, hn, q, k, v, h, ids, pos, pe, mask
            gc.collect()
            t.cuda.empty_cache()
            RL.log(stage, min(total, c0 + N) * (PB + 1), total * (PB + 1), start, block=PB)
        peak = round(t.cuda.max_memory_allocated() / 2 ** 20)
    return {k: np.stack([np.concatenate(x) for x in v]).astype(np.float32) for k, v in feats.items()}, \
        {'feature_peak_MiB': peak, 'feature_seconds': round(time.monotonic() - start, 1)}


def dataset(rows, arms, items, F, xarm='x|NH:C'):
    """Conflict edits: features, x, follow, class, family."""
    fam_id = {f: i for i, f in enumerate(sorted({r['family'] for r in rows}))}
    X = {s: [] for s in ('recomb', 'obs', 'scalars')}
    out = {k: [] for k in ('x', 'follow', 'cls', 'fam', 'nat')}
    for k in range(REPS):
        a = arms[f'swap#{k}']
        for i, (r, j) in enumerate(zip(rows, a['donor'])):
            B, P, I = F3.records(r, rows[j], 'names', 24)
            c = F3.klass(B, P, I)
            if not c:
                continue
            s, b = items[f'{xarm}#{k}'][i]['scores'], items['base'][i]['scores']
            out['x'].append(s[P] - s[B]); out['nat'].append(b[P] - b[B]); out['cls'].append(c); out['fam'].append(fam_id[r['family']])
            out['follow'].append(items[f'swap#{k}'][i]['pred_record'] == P)
            for name in X:
                X[name].append(np.concatenate([F[name][k][i], [b[P] - b[B]]]) if name == 'scalars' else F[name][k][i])
    X = {k: np.array(v, np.float64) for k, v in X.items()}
    X['recomb+obs'] = np.concatenate([X['recomb'], X['obs']], 1)
    X['all'] = np.concatenate([X['recomb'], X['obs'], X['scalars']], 1)
    return X, {k: np.array(v) for k, v in out.items()}, len(fam_id)


def ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return {'mu': mu, 'sd': sd, 'w': w, 'b': float(y.mean())}


def apply(m, X):
    return ((X - m['mu']) / m['sd']) @ m['w'] + m['b']


def load_open(out, split, xarm):
    rows, arms, items = (D.read(out / x) for x in (f'{split}-materials.json', 'design.json', 'items.json'))
    return rows, arms, items, [arms[f'swap#{k}']['donor'] for k in range(REPS)], xarm


def collect_open():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定免编辑读出产物')
    for tag, (out, split, xarm) in {'per': (PE_OUT, PE.SPLIT, 'x|NH:C'), 'h25': (H25_OUT, H25.SPLIT, 'nat|NH:C')}.items():
        rows, arms, items, dons, _ = load_open(out, split, xarm)
        F, meta = collect(rows, dons, f'collect_{tag}')
        np.savez_compressed(OUT / f'features-{tag}.npz', **F)
        print(json.dumps({tag: meta, 'shapes': {k: list(v.shape) for k, v in F.items()}}), flush=True)


def fit():
    """Opened panels only: fit on holdout_per, check on holdout_h25; writes readout.npz for every feature set."""
    data = {}
    for tag, (out, split, xarm) in {'per': (PE_OUT, PE.SPLIT, 'x|NH:C'), 'h25': (H25_OUT, H25.SPLIT, 'nat|NH:C')}.items():
        rows, arms, items, dons, _ = load_open(out, split, xarm)
        data[tag] = dataset(rows, arms, items, dict(np.load(OUT / f'features-{tag}.npz')), xarm)
    Xp, yp, nf = data['per']
    Xh, yh, _ = data['h25']
    folds = np.random.default_rng(SEED).permutation(nf) % 6
    res, models = {}, {}
    for s in SETS:
        cv = {}
        for lam in LAMBDAS:
            pred = np.zeros(len(yp['x']))
            for f in range(6):
                te = folds[yp['fam']] == f
                pred[te] = yp['nat'][te] + apply(ridge(Xp[s][~te], (yp['x'] - yp['nat'])[~te], lam), Xp[s][te])
            cv[lam] = {'r': float(np.corrcoef(pred, yp['x'])[0, 1]), 'auc': PE.auc(pred, yp['follow']),
                       'acc': float(np.mean((pred > T0) == yp['follow']))}
        lam = max(LAMBDAS, key=lambda l: cv[l]['r'])
        m = ridge(Xp[s], yp['x'] - yp['nat'], lam)
        ph = yh['nat'] + apply(m, Xh[s])
        res[s] = {'dim': int(Xp[s].shape[1]), 'lambda': lam, 'cv_on_per': {k: round(v, 4) for k, v in cv[lam].items()},
                  'check_on_h25': {'r': round(float(np.corrcoef(ph, yh['x'])[0, 1]), 4), 'auc': round(PE.auc(ph, yh['follow']), 4),
                                   'acc': round(float(np.mean((ph > T0) == yh['follow'])), 4),
                                   'auc_by_class': {c: round(PE.auc(ph[yh['cls'] == c], yh['follow'][yh['cls'] == c]), 4) for c in PE.CLASSES}}}
        models[s] = m
    res['reference'] = {tag: {'x_auc': round(PE.auc(y['x'], y['follow']), 4), 'x_acc': round(float(np.mean((y['x'] > T0) == y['follow'])), 4),
                              'nat_auc': round(PE.auc(y['nat'], y['follow']), 4), 'n': int(len(y['x']))} for tag, (_, y, _) in data.items()}
    np.savez(OUT / 'readout.npz', **{f'{s}|{k}': np.asarray(v) for s, m in models.items() for k, v in m.items()})
    save('fit.json', {'code_sha256': D.sha(__file__), 'results': res, 'readout_sha256': D.sha(OUT / 'readout.npz'),
                      'features_sha256': {t: D.sha(OUT / f'features-{t}.npz') for t in ('per', 'h25')}})
    print(json.dumps(res), flush=True)


def model(s):
    z = np.load(OUT / 'readout.npz')
    return {k: (float(z[f'{s}|{k}']) if k == 'b' else z[f'{s}|{k}']) for k in ('mu', 'sd', 'w', 'b')}


def summarise(results, arms, rows, F):
    X, y, nf = dataset(rows, arms, results, F)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, nf, (1000, nf))
    fam, fol = y['fam'], y['follow']
    members = [np.flatnonzero(fam == f) for f in range(nf)]

    def rate(v):
        v = np.asarray(v, float)
        per, cnt = np.bincount(fam, weights=v, minlength=nf), np.bincount(fam, minlength=nf)
        bs = per[draws].sum(1) / np.maximum(cnt[draws].sum(1), 1)
        return {'rate': round(float(v.mean()), 4), 'ci95': [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]}

    def boot(fn):
        vals = [fn(np.concatenate([members[f] for f in d])) for d in draws[:300]]
        return [round(float(np.nanquantile(vals, .025)), 4), round(float(np.nanquantile(vals, .975)), 4)]
    pred = {s: y['nat'] + apply(model(s), X[s]) for s in SETS}
    pred.update({'x': y['x'], 'nat': y['nat']})
    out = {'conflict_edits': int(len(fol)), 'follow': rate(fol), 'predictors': {}}
    cls, none = (y['cls'] == 'P=I') == fol, ~fol
    for name, p in pred.items():
        e = {'auc': round(PE.auc(p, fol), 4), 'auc_ci95': boot(lambda idx, p=p: PE.auc(p[idx], fol[idx])),
             'auc_by_class': {c: round(PE.auc(p[y['cls'] == c], fol[y['cls'] == c]), 4) for c in PE.CLASSES}}
        if name != 'nat':
            ok = (p > T0) == fol
            e.update({'accuracy': rate(ok), 'minus_class_rule': rate(ok.astype(float) - cls.astype(float)),
                      'minus_none': rate(ok.astype(float) - none.astype(float)), 'r_with_x': round(float(np.corrcoef(p, y['x'])[0, 1]), 4)})
        out['predictors'][name] = e
    out['accuracy_class_rule'], out['accuracy_none'] = rate(cls), rate(none)
    p = pred[PRIMARY]
    out['primary'] = PRIMARY
    out['primary_auc_minus_nat'] = {'rate': round(PE.auc(p, fol) - PE.auc(y['nat'], fol), 4),
                                    'ci95': boot(lambda idx: PE.auc(p[idx], fol[idx]) - PE.auc(y['nat'][idx], fol[idx]))}
    out['x_auc_minus_primary'] = {'rate': round(PE.auc(y['x'], fol) - PE.auc(p, fol), 4),
                                  'ci95': boot(lambda idx: PE.auc(y['x'][idx], fol[idx]) - PE.auc(p[idx], fol[idx]))}
    q = np.quantile(p, np.linspace(0, 1, 11))
    out['follow_by_decile_of_x_hat'] = [round(float(fol[(p >= q[i]) & ((p < q[i + 1]) | (i == 9))].mean()), 4) for i in range(10)]
    return out


def verdicts(s, meta):
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {'controls_valid': bool(meta['controls_ok'])}
    names = list(CRITERIA['predictions_on_holdout'])
    if not v['controls_valid']:
        return {**v, **{k: 'controls_invalid' for k in names}}
    p = s['predictors'][s['primary']]
    d = s['primary_auc_minus_nat']
    v['U1_edit_free_ranking'] = tri(p['auc'] >= .72 and d['ci95'][0] > 0, p['auc'] < .68 or d['ci95'][0] <= 0)
    c = p['minus_class_rule']
    v['U2_edit_free_rule'] = tri(p['accuracy']['rate'] >= .68 and c['ci95'][0] > 0, c['ci95'][0] <= 0)
    g = s['x_auc_minus_primary']
    v['U3_linear_read_out_falls_short'] = tri(g['ci95'][0] > .05, g['ci95'][0] <= 0)
    return v


DEPS = (RL, PT, NL, F3, AR, H25, PE)


def prereg():
    from tools import provenance
    from tools.power_analysis import check_phase
    assert PRIMARY and not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    rows = build_materials()
    assert len(rows) == FAMILIES * 16 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    arms = final_arms(rows)
    save(f'{SPLIT}-materials.json', rows)
    save('design.json', arms)
    fitj = D.read(OUT / 'fit.json')
    assert fitj['readout_sha256'] == D.sha(OUT / 'readout.npz')
    body = {'schema': 'source_binding_readout_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'primary': PRIMARY, 'threshold': T0, 'readout_sha256': fitj['readout_sha256'],
            'fit_sha256': D.sha(OUT / 'fit.json'), 'fit_results': fitj['results'],
            'materials': {'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': len(rows)},
            'design': {'sha256': D.sha(OUT / 'design.json'), 'arms': len(arms)},
            'power': {'unit': 'paired binary outcome per item (read-out rule right / class rule right); alpha = 0.05',
                      'as_if_one_edit_per_item': {'expected_delta': 0.06, 'p_d': 0.40, **check_phase(0.06, len(rows), p_d=0.40)},
                      'caveat': 'about 3,400 conflict edits clustered in 1,024 items and 64 families; intervals are family bootstraps'},
            'custody': 'holdout_ro is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 8, 'disk_GiB_estimate': 0.1, 'models_concurrent': 1,
                          'shape': 'chunks of 192 prompts in one batch; edited arms with the Phase 415 engine, feature pass up to block 25'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'design': body['design'], 'power': body['power']}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and all(p['dependency_sha256'][m.__name__] == D.sha(m.__file__) for m in DEPS), \
        'source changed after pre-registration'
    assert p['materials']['sha256'] == D.sha(OUT / f'{SPLIT}-materials.json') and p['design']['sha256'] == D.sha(OUT / 'design.json')
    assert p['readout_sha256'] == D.sha(OUT / 'readout.npz')
    D.checked(MODEL)
    return p


def gate():
    checked()
    assert not (OUT / 'gate.json').exists(), 'holdout behaviour already read'
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    H25.OUT = OUT
    results, meta = H25.engine(rows, None, 'gate')
    save('gate-items.json', results['base'])
    g = D.behavior(rows, results['base'])
    save('gate.json', {'schema': 'source_binding_readout_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre = checked()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows, arms = D.read(OUT / f'{SPLIT}-materials.json'), D.read(OUT / 'design.json')
    if g['passed']:
        H25.OUT = OUT
        results, meta = H25.engine(rows, arms, 'run')
        meta.pop('attention_mass', None)
        F, fm = collect(rows, [arms[f'swap#{k}']['donor'] for k in range(REPS)], 'features')
        meta.update(fm)
        np.savez_compressed(OUT / 'features-holdout.npz', **F)
        save('items.json', {k: [{'id': x['id'], 'pred_record': x['pred_record'], 'scores': x['scores']} for x in v] for k, v in results.items()})
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, arms, rows, F)
        v = verdicts(s, meta)
        print(json.dumps({'meta': meta, 'verdicts': v, 'summary': s}), flush=True)
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in CRITERIA['predictions_on_holdout']}
    report = {'schema': 'source_binding_readout_report.v1', 'run_id': RUN, 'split': SPLIT, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'primary': pre['primary'],
              'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_ro read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-readout-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                    'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


STAGES = {f.__name__: f for f in (collect_open, fit, prereg, gate, run)}
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=list(STAGES))
    STAGES[ap.parse_args().stage]()
