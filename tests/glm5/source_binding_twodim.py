"""Do 'which archive' and 'which entity' enter the relay position separately?  Qwen3-14B BF16, source binding.

Pre-registered test of the post-hoc observation of Claude Phase 403.  Configuration iso throughout: in blocks 24-27
the relay position R sees only the four records (plus itself and the chat-template head), and from block 24 on the
final position reads only R, so the request acts only through the state R has at block 24.
  knock-outs   attention of R in blocks 20-23: of the three question groups of Phase 403 (entity name, archive name,
               other question tokens) only the archive name / only the entity name stays readable; single blocks
  swaps        the residual state of R at the input of block 24 is replaced by that of a donor prompt of the same
               family and record order that asks for the other archive (src), the other entity (ent) or both
  addition     R + (R_src - R) + (R_ent - R): if the two dimensions are carried separately and additively, the sum of
               the two single shifts should give the request of the both-donor
The choice is classified against the base prompt: correct / other entity / other archive / neither.
Stages: prereg -> pilot (discovery, exposed; nothing selected) -> gate -> run.
"""
import argparse, gc, hashlib, json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
import source_binding_relay as RL
import source_binding_upstream as UP
import source_binding_heads as HD
import source_binding_qkov as QK
import source_binding_query as QY
import source_binding_request as RQ
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
PANELS = RQ.PANELS + [RQ.OUT / 'holdout_request-materials.json']
OUT = D.ROOT / 'tests/glm5/result/source_binding_twodim_v1'
RUN, SEED, SPLIT = 'SOURCE-BINDING-TWODIM-20261111-v1', 202611110404, 'holdout_twodim'
FINAL_LAYER, WIN, ISO_END, FAMILIES, BATCH, NB, NH, R_FROM_END, TOL = 24, 20, 28, 96, 128, 40, 40, UP.R_FROM_END, 1.0
HEAD, QGROUPS, UNIONS = (30, 4), ['qent', 'qrest', 'qsrc'], QY.UNIONS
MASKS = {'block:N': QGROUPS, 'leave:qsrc': ['qent', 'qrest'], 'leave:qent': ['qsrc', 'qrest'], 'block:qent': ['qent'], 'block:qsrc': ['qsrc']}
STATES = ['swap:self', 'swap:src', 'swap:ent', 'swap:both', 'swap:same', 'add:src+ent', 'add:rand', 'xfam:src', 'half:src+ent']
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'configuration': 'iso in every arm and in the reference: blocks 24-27 R attends only to the four records, itself and the template head; '
                     'from block 24 on the final position attends only to R (and itself)',
    'knock_outs': 'attention of R (all heads) in blocks 20-23 cannot reach the listed groups. block:N = entity name, archive name and other '
                  'question tokens (necessary set of Phase 403); leave:qsrc = N without the archive name; leave:qent = N without the entity '
                  'name; block:qent, block:qsrc = single groups',
    'state_edits': {'swap:<d>': 'residual state of R at the input of block 24 replaced by that of the donor (same family, same record order; '
                                'src = other archive, ent = other entity, both; same = same record, archives listed in the other order; '
                                'self = identity check)',
                    'add:src+ent': 'R + (R_src - R) + (R_ent - R), computed in float32',
                    'half:src+ent': 'R + 0.5 (R_src - R) + 0.5 (R_ent - R) (descriptive)',
                    'add:rand': 'R + a random vector with the norm of (R_src - R) + (R_ent - R), seeded per item',
                    'xfam:src': 'R + (R_src - R) taken from the same item of another family with the same frame (family + 4) (descriptive)'},
    'outcome': 'the chosen record among four (chance 0.25), classified against the base prompt: correct, other_entity (same archive), '
               'other_source (same entity), neither. archive_ok = correct + other_entity; entity_ok = correct + other_source. '
               'All items; family-bootstrap 95 % intervals.',
    'tail_recomputation': 'as in Phase 403: arms recompute the 11 positions >= R. Control: the no-edit arm recomputed from block 0 gives the '
                          'same forced choice as the base forward on every item, maximal score difference <= 1.0',
    'material': 'holdout_twodim: 96 new families x 8 items; entity names that occur in no earlier panel of this line; the 16 discovery code '
                'words, 4 seen frames. Pilot: the 512 discovery items (exposed); nothing is selected on it.',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed; otherwise behavior_ineligible.',
    'predictions_on_holdout': {
        'controls_valid': 'tail recomputation control holds; ref_iso correct >= 0.80; swap:self gives the choices of ref_iso on every item; '
                          'block:N correct <= 0.50',
        'K1_archive_name_gives_archive_only': {'supported': 'leave:qsrc: archive_ok >= 0.90 and entity_ok between 0.35 and 0.65',
                                               'not_supported': 'archive_ok < 0.70 or entity_ok >= 0.80'},
        'K2_entity_name_gives_entity_only': {'supported': 'leave:qent: entity_ok >= 0.90 and archive_ok <= 0.75',
                                             'not_supported': 'entity_ok < 0.70 or archive_ok >= 0.90'},
        'X1_state_carries_archive': {'supported': 'swap:src: other_source >= 0.70', 'not_supported': '<= 0.30'},
        'X2_state_carries_entity': {'supported': 'swap:ent: other_entity >= 0.70', 'not_supported': '<= 0.30'},
        'X3_additive': {'supported': 'add:src+ent: neither (the record of the both-donor) >= 0.70 and add:rand correct >= 0.70',
                        'not_supported': 'add:src+ent: neither <= 0.30'}},
    'descriptive': 'swap:both, swap:same, half, xfam, single knock-outs, attention mass of head 30.4 on the correct record, top-1',
    'not_covered': 'which heads write the two dimensions; a basis for the two subspaces; the natural configuration where R can also read '
                   'the question in blocks 24-27; other models, unseen answer words, unseen frames',
    'disclosure': 'the code was smoke-tested on 32 discovery items (exposed) before pre-registration; thresholds were written first. '
                  'K1 and K2 restate an observation made after holdout_request was opened (Phase 403); that panel is not reused here.',
}


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定两维分读产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def log(*a, **kw):
    RL.OUT = OUT
    RL.log(*a, **kw)


def build_materials():
    tok = RL.tokenizer()
    used = {n for rows in D.read(SRC / f'{MODEL}-materials.json')['splits'].values() for r in rows for n in r['names']}
    for p in PANELS:
        used |= {n for r in D.read(p) for n in r['names']}
    rows = []
    for family in range(FAMILIES):
        rnd = stream('names', family)
        names = []
        while len(names) < 2:
            name = ''.join(rnd.choice(list('bdfgklmnprstvz')) + rnd.choice(list('aeiou')) for _ in range(3)).capitalize()
            if name not in used:
                used.add(name)
                names.append(name)
        values = rnd.choice(D.WORDS[:16], 4, replace=False).tolist()
        for s in (0, 1):
            for e in (0, 1):
                for o in (0, 1):
                    r = D.material(family, SPLIT, s, e, o, family % 4, names, values)
                    prompt = tok.apply_chat_template([{'role': 'user', 'content': r['content']}], tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=False) + 'Answer:\n'
                    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
                    ids, off = enc['input_ids'], prompt.index(r['content'])
                    span = lambda a, b: [i for i, (lo, hi) in enumerate(enc['offset_mapping']) if hi > off + a and lo < off + b]
                    recs = {str(k): {key: span(a, b) for key, (a, b) in info.items()} for k, info in r['records'].items()}
                    choice = []
                    for val in r['values']:
                        suffix = tok.encode(prompt + val, add_special_tokens=False)
                        assert suffix[:-1] == ids
                        choice.append(suffix[-1])
                    r = {**r, 'records': recs, 'headers': {str(k): span(a, b) for k, (a, b) in r['headers'].items()}, 'prompt': prompt,
                         'tokens': ids, 'choice_ids': choice, 'gold_id': choice[r['target_record']]}
                    rows.append(QY.add_groups(tok, r))
    return rows


DEPS = (RL, UP, HD, QK, QY, RQ)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定两维分读产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    rows = build_materials()
    assert len(rows) == FAMILIES * 8 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    save(f'{SPLIT}-materials.json', rows)
    n = len(rows)
    body = {'schema': 'source_binding_twodim_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'request_report_sha256': D.sha(RQ.OUT / 'report.json'),
            'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'materials': {'split': SPLIT, 'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': n},
            'power': {'unit': 'paired binary outcome per item (McNemar); alpha = 0.05',
                      'K_X': {'expected_delta': 0.30, 'p_d': 0.50, **check_phase(0.30, n, p_d=0.50)},
                      'caveat': 'items are clustered in 96 families and 4 frames; K1 states an interval for entity_ok, whose 95 % '
                                'family-bootstrap interval is reported'},
            'custody': 'window, groups and head come from Phases 400-403 (opened panels). holdout_twodim is generated here, read first by '
                       'the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 10, 'disk_GiB_estimate': 0.2, 'models_concurrent': 1,
                          'shape': 'one chunk per panel, inner batch 128, arms recompute 11 positions; fixed before pre-registration'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'power': body['power']['K_X']}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and all(p['dependency_sha256'][m.__name__] == D.sha(m.__file__) for m in DEPS), \
        'source changed after pre-registration'
    assert p['materials']['sha256'] == D.sha(OUT / f'{SPLIT}-materials.json')
    D.checked(MODEL)
    return p


def donors(rows):
    index = {r['id']: i for i, r in enumerate(rows)}
    fams = sorted({r['family'] for r in rows})
    key = lambda fam, s, e, o: index[f'{fam}-S{s}E{e}O{o}']
    out = {'self': list(range(len(rows)))}
    out['src'] = [key(r['family'], 1 - r['source'], r['entity'], r['order']) for r in rows]
    out['ent'] = [key(r['family'], r['source'], 1 - r['entity'], r['order']) for r in rows]
    out['both'] = [key(r['family'], 1 - r['source'], 1 - r['entity'], r['order']) for r in rows]
    out['same'] = [key(r['family'], r['source'], r['entity'], 1 - r['order']) for r in rows]
    other = lambda r: fams[(fams.index(r['family']) + 4) % len(fams)]
    out['xbase'] = [key(other(r), r['source'], r['entity'], r['order']) for r in rows]
    out['xsrc'] = [key(other(r), 1 - r['source'], r['entity'], r['order']) for r in rows]
    return out


def engine(rows, edits=True, stage='run'):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    t, F = torch, torch.nn.functional
    assert snapshot(D.MODELS[MODEL]) == D.read(SRC / f'{MODEL}-identity.json')['model_snapshot']
    # name -> (fork block, mask groups or None, iso)
    arms = {}
    if edits:
        arms = {'noop': (0, None, False), 'ref_iso': (FINAL_LAYER, None, True)}
        arms.update({k: (WIN, g, True) for k, g in MASKS.items()})
        arms.update({k: (FINAL_LAYER, None, True) for k in STATES})
    start, N = time.monotonic(), len(rows)
    results, mass = {}, {}
    with GpuLease(), t.inference_mode():
        rt = Runtime(MODEL)
        hd, kv, grp = rt.hd, rt.kv, rt.heads // rt.kv
        ids, pos, pe, mask = rt.context(rows)
        h, length = rt.embed(ids), ids.shape[1]
        R = length - R_FROM_END
        T = length - R
        log(stage, 0, NB, start, rows=N, length=length, arms=len(arms))
        frow = t.zeros((N, length), dtype=t.bool, device='cuda')
        frow[:, R] = True
        frow[:, -1] = True
        cor = t.zeros((N, length), dtype=t.bool, device='cuda')
        base_row = mask[:, 0, R]

        def row_for(kind, groups):
            row = base_row.clone()
            for i, r in enumerate(rows):
                off = length - len(r['tokens'])
                sel = [off + j for g in groups for j in r['up'][g]]
                if kind == 'block':
                    row[i, sel] = False
                else:
                    keep = row[i].clone()
                    row[i] = False
                    row[i, [R] + [off + j for j in r['up']['tpl_head']]] = True
                    row[i, sel] = keep[sel]
                    row[i] &= keep
            return row
        for i, r in enumerate(rows):
            assert len(r['tokens']) - R_FROM_END == r['up']['tail0'][-1] + 1
            cor[i, [length - len(r['tokens']) + j for j in r['up']['rec_correct']]] = True
        isorow = row_for('only', UNIONS['records4'])
        rrow = {name: row_for('block', g) for name, (_, g, _) in arms.items() if g}
        don = {k: t.tensor(v, device='cuda') for k, v in donors(rows).items()} if edits else {}
        rot = lambda x, co, si: x * co + rt.mod.rotate_half(x) * si
        state, cap = {}, {}
        for b in range(NB):
            for name, (l, g, _) in arms.items():
                if l != b:
                    continue
                state[name] = h[:, R:].clone()
                if name in STATES:
                    r0 = h[:, R].float()
                    ds, de = r0[don['src']] - r0, r0[don['ent']] - r0
                    if name.startswith('swap:'):
                        new = r0[don[name[5:]]]
                    elif name == 'add:src+ent':
                        new = r0 + ds + de
                    elif name == 'half:src+ent':
                        new = r0 + .5 * ds + .5 * de
                    elif name == 'xfam:src':
                        new = r0 + r0[don['xsrc']] - r0[don['xbase']]
                    else:
                        g_ = t.Generator(device='cpu')
                        noise = t.stack([t.randn(r0.shape[1], generator=g_.manual_seed(
                            int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|rand|{r["id"]}'.encode()).digest()[:7], 'little'))) for r in rows]).cuda()
                        new = r0 + noise / noise.norm(dim=-1, keepdim=True) * (ds + de).norm(dim=-1, keepdim=True)
                    state[name][:, 0] = h[:, R] if name == 'swap:self' else new.to(h.dtype)
            layer = rt.block(b)
            a = layer.self_attn
            handles = []
            if state:
                handles = [a.k_proj.register_forward_hook(lambda m, i, o: cap.__setitem__('k', o[:, :R])),
                           a.v_proj.register_forward_hook(lambda m, i, o: cap.__setitem__('v', o[:, :R]))]
            y = t.empty_like(h)
            for lo in range(0, N, BATCH):
                hi = min(lo + BATCH, N)
                B = hi - lo
                cos, sin = pe[0][lo:hi], pe[1][lo:hi]
                y[lo:hi] = layer(h[lo:hi], attention_mask=mask[lo:hi], position_ids=pos[lo:hi], position_embeddings=(cos, sin), use_cache=False)
                if not state:
                    continue
                kpre, vpre = cap['k'], cap['v']
                tm0 = mask[lo:hi][:, :, R:, :]
                for name in state:
                    l, g, iso = arms[name]
                    tm = tm0.clone()
                    if name != 'noop' and b >= FINAL_LAYER:
                        tm[:, 0, -1] = frow[lo:hi]
                    if g and l <= b < FINAL_LAYER:
                        tm[:, 0, 0] = rrow[name][lo:hi]
                    if iso and FINAL_LAYER <= b < ISO_END:
                        tm[:, 0, 0] = isorow[lo:hi]
                    x = state[name][lo:hi]
                    hn = layer.input_layernorm(x)
                    q = a.q_norm(a.q_proj(hn).view(B, T, NH, hd)).transpose(1, 2)
                    k = a.k_norm(t.cat([kpre, a.k_proj(hn)], 1).view(B, length, kv, hd)).transpose(1, 2)
                    v = t.cat([vpre, a.v_proj(hn)], 1).view(B, length, kv, hd).transpose(1, 2)
                    q, k = rot(q, cos[:, None, R:], sin[:, None, R:]), rot(k, cos[:, None], sin[:, None])
                    k, v = rt.mod.repeat_kv(k, grp), rt.mod.repeat_kv(v, grp)
                    if b == HEAD[0]:
                        w = ((q[:, HEAD[1], 0, None, :] * k[:, HEAD[1]]).sum(-1) * a.scaling).float().masked_fill(~tm[:, 0, 0], float('-inf'))
                        mass.setdefault(name, []).extend((w.softmax(-1) * cor[lo:hi]).sum(-1).cpu().tolist())
                    o = F.scaled_dot_product_attention(q, k, v, attn_mask=tm)
                    x = x + a.o_proj(o.transpose(1, 2).reshape(B, T, -1))
                    state[name][lo:hi] = x + layer.mlp(layer.post_attention_layernorm(x))
            h = y
            for x in handles:
                x.remove()
            cap.clear()
            del layer, y, a
            gc.collect()
            t.cuda.empty_cache()
            if (b + 1) % 4 == 0:
                log(stage, b + 1, NB, start, block=b, arms=len(state))
        norm = rt.Norm(rt.dim, eps=rt.cfg.rms_norm_eps).to(device='cuda', dtype=t.bfloat16)
        norm.weight = t.nn.Parameter(rt.weight('model.norm.weight'), requires_grad=False)
        W = rt.weight('lm_head.weight' if 'lm_head.weight' in rt.index else 'model.embed_tokens.weight')
        logits = lambda x: t.cat([(norm(x[lo:lo + BATCH]) @ W.T).float() for lo in range(0, N, BATCH)])
        results['base'] = rt.score(logits(h[:, -1]), rows)
        for key, s in state.items():
            results[key] = rt.score(logits(s[:, -1]), rows)
        peak = round(t.cuda.max_memory_allocated() / 2 ** 20)
    meta = {'peak_MiB': peak, 'seconds': round(time.monotonic() - start, 1), 'padded_length': length}
    if edits:
        same = lambda a, b: all(x['pred_record'] == y['pred_record'] for x, y in zip(results[a], results[b]))
        meta['noop_same_choice'] = same('noop', 'base')
        meta['noop_max_score_difference'] = max(abs(p - q) for x, y in zip(results['noop'], results['base']) for p, q in zip(x['scores'], y['scores']))
        meta['tail_control_ok'] = bool(meta['noop_same_choice'] and meta['noop_max_score_difference'] <= TOL)
        meta['swap_self_same_choice'] = same('swap:self', 'ref_iso')
    return results, mass, meta


def summarise(results, mass, rows):
    fam = np.array([r['family'] for r in rows])
    members = [np.flatnonzero(fam == f) for f in sorted(set(fam))]
    draws = np.random.default_rng(SEED).integers(0, len(members), (2000, len(members)))
    cnt = np.array([len(m) for m in members])

    def ci(v):
        v = np.asarray(v, float)
        per = np.array([v[m].sum() for m in members])
        bs = per[draws].sum(1) / cnt[draws].sum(1)
        return {'rate': round(float(v.mean()), 4), 'ci95': [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]}
    s, e = np.array([r['source'] for r in rows]), np.array([r['entity'] for r in rows])
    out = {'n': len(rows), 'arms': {}}
    for k, res in results.items():
        pred = np.array([x['pred_record'] for x in res])
        ps, pe = pred // 2, pred % 2
        out['arms'][k] = {'correct': ci((ps == s) & (pe == e)), 'other_entity': ci((ps == s) & (pe != e)),
                          'other_source': ci((ps != s) & (pe == e)), 'neither': ci((ps != s) & (pe != e)),
                          'archive_ok': ci(ps == s), 'entity_ok': ci(pe == e), 'top1': ci([x['top1_correct'] for x in res]),
                          'mass': ci(mass[k]) if k in mass else None}
    return out


def verdicts(s, meta):
    r = lambda k, f: s['arms'][k][f]['rate']
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    names = [k for k in CRITERIA['predictions_on_holdout'] if k != 'controls_valid']
    ok = meta['tail_control_ok'] and meta['swap_self_same_choice'] and r('ref_iso', 'correct') >= .80 and r('block:N', 'correct') <= .50
    v = {'controls_valid': bool(ok)}
    if not ok:
        v.update({k: 'controls_invalid' for k in names})
        return v
    a, e = r('leave:qsrc', 'archive_ok'), r('leave:qsrc', 'entity_ok')
    v[names[0]] = tri(a >= .90 and .35 <= e <= .65, a < .70 or e >= .80)
    e, a = r('leave:qent', 'entity_ok'), r('leave:qent', 'archive_ok')
    v[names[1]] = tri(e >= .90 and a <= .75, e < .70 or a >= .90)
    v[names[2]] = tri(r('swap:src', 'other_source') >= .70, r('swap:src', 'other_source') <= .30)
    v[names[3]] = tri(r('swap:ent', 'other_entity') >= .70, r('swap:ent', 'other_entity') <= .30)
    v[names[4]] = tri(r('add:src+ent', 'neither') >= .70 and r('add:rand', 'correct') >= .70, r('add:src+ent', 'neither') <= .30)
    return v


def show(s, meta, v, smoke=None):
    table = {k: [a[f]['rate'] for f in ('correct', 'other_entity', 'other_source', 'neither')] + [a['mass']['rate'] if a['mass'] else None]
             for k, a in s['arms'].items()}
    print(json.dumps({'smoke': smoke, 'meta': meta, 'verdicts': v, 'correct_otherEntity_otherSource_neither_mass': table}), flush=True)


def pilot(smoke=False):
    tok = RL.tokenizer()
    rows = [QY.add_groups(tok, r) for r in D.read(SRC / f'{MODEL}-materials.json')['splits']['discovery']][:64 if smoke else None]
    if not smoke:
        checked()
        assert not (OUT / 'pilot.json').exists()
    results, mass, meta = engine(rows, stage='pilot')
    s = summarise(results, mass, rows)
    v = verdicts(s, meta)
    if not smoke:
        save('pilot.json', {'schema': 'source_binding_twodim_pilot.v1', 'run_id': RUN, 'note': 'discovery (exposed); exploratory, nothing selected',
                            'verdict_rules_applied_for_description': v, 'summary': s, **meta})
    show(s, meta, v, smoke)


def gate():
    checked()
    assert (OUT / 'pilot.json').exists() and not (OUT / 'gate.json').exists(), 'holdout behaviour already read'
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    results, _, meta = engine(rows, edits=False, stage='gate')
    save('gate-items.json', results['base'])
    g = D.behavior(rows, results['base'])
    save('gate.json', {'schema': 'source_binding_twodim_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre = checked()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    if g['passed']:
        results, mass, meta = engine(rows, stage='run')
        save('items.json', {'results': results, 'mass': mass})
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, mass, rows)
        v = verdicts(s, meta)
        show(s, meta, v)
    else:
        s, meta = {'note': 'behaviour gate failed; no edit was run'}, {}
        v = {k: 'behavior_ineligible' for k in CRITERIA['predictions_on_holdout'] if k != 'controls_valid'}
    report = {'schema': 'source_binding_twodim_report.v1', 'run_id': RUN, 'split': SPLIT, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'pilot_sha256': D.sha(OUT / 'pilot.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'verdicts': v, 'summary': s,
              **meta, 'formal_evidence_promotion': 0, 'custody': 'holdout_twodim read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-twodim-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                     'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['prereg', 'pilot', 'gate', 'run'])
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    pilot(a.smoke) if a.stage == 'pilot' else {'prereg': prereg, 'gate': gate, 'run': run}[a.stage]()
