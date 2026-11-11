"""What does head 30.4 select by (QK) and what does it write (OV)?  Qwen3-14B BF16, source binding.

Follows Claude Phase 400: at the relay position R (line break after the user turn) head 30.4 carries the read from the
correct record; head 28.21 cooperates.  Readout as before: from block 24 on the final position may attend only to R.
  attention  attention of the head from R, recomputed from the captured q / k / v of the original forward
  q:<donor>  the head's query vector at R (q_proj output, before q_norm and RoPE) is replaced by that of a donor prompt
  z:<donor>  the head's output at R (its slice of the o_proj input) is replaced by that of a donor prompt
  spans      head 30.4 may not attend from R to the value token / to the other tokens / to any token of the correct record
Donors are prompts of the same family and record order that ask for the other archive (src), the other entity (ent) or
both; 'same' asks for the same record with the archives listed in the other order; 'self' is the identity check.
The whole panel is one chunk (each block is read from disk once); inner batch 64.
Stages: prereg -> pilot (discovery, exposed) -> gate -> run.
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
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
PANELS = [RL.ABL / 'holdout_abl-materials.json', RL.OUT / 'holdout_relay-materials.json', UP.OUT / 'holdout_up-materials.json',
          HD.OUT / 'holdout_heads-materials.json']
OUT = D.ROOT / 'tests/glm5/result/source_binding_qkov_v1'
RUN, SEED, SPLIT = 'SOURCE-BINDING-QKOV-20261111-v1', 202611110401, 'holdout_qkov'
FINAL_LAYER, FAMILIES, BATCH, NB, NH, R_FROM_END = 24, 96, 64, 40, 40, UP.R_FROM_END
HEADS = {'h30': (30, 4), 'h28': (28, 21)}
DONORS = ['src', 'ent', 'both', 'same']
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'heads': 'primary: block 30 head 4 (necessary set of Phase 400); descriptive: block 28 head 21. Position: R, 10 tokens before the end.',
    'readout': 'in every edited arm and in the reference, from block 24 on the final position may attend only to R (and itself)',
    'donors': {'src': 'same family and record order, question names the other archive', 'ent': 'the other entity', 'both': 'both differ',
               'same': 'same archive and entity, archives listed in the other order', 'self': 'the prompt itself (identity check)'},
    'edits': {'q': 'slice of the q_proj output of the head at R replaced by the donor slice (the model then applies q_norm and RoPE at the '
                   'position of the base prompt); keys, values and all other heads untouched',
              'z': 'slice of the o_proj input of the head at R replaced by the donor slice; the attention pattern is untouched',
              'spans': 'per-head attention mask of head 30.4 at R: val_only = cannot attend to tokens of the correct record other than the '
                       'value token; noval = cannot attend to the value token; block = cannot attend to the correct record'},
    'outcome': '4-way forced choice (chance 0.25). keep = the answer of the base prompt is chosen; follow = the answer of the donor prompt '
               'is chosen. All items analysed; family-bootstrap 95 % intervals.',
    'material': 'holdout_qkov: 96 new families x 8 items; entity names that occur in no earlier panel of this line; the 16 discovery code '
                'words, 4 seen frames. Pilot: the 512 discovery items (exposed); nothing is selected on it.',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed; otherwise behavior_ineligible.',
    'predictions_on_holdout': {
        'controls_valid': 'noop reproduces the base logits bit for bit; q:self and z:self reproduce the reference logits bit for bit; '
                          'reference keep >= 0.90; block keep <= 0.50; recomputed attention reproduces the captured head output '
                          '(median relative error <= 0.05)',
        'P1_attends_correct_record': {'supported': 'mean attention mass of 30.4 from R on the correct record >= 0.50 and >= 3 x the mean mass '
                                                   'on each other record', 'not_supported': 'mass < 0.25'},
        'P2a_query_redirects_attention': {'supported': 'with the both-donor query, mean mass on the record the donor asks for >= 0.50',
                                          'not_supported': '< 0.25'},
        'P2b_query_redirects_answer': {'supported': 'q:both@h30 follow >= 0.60', 'not_supported': 'follow <= 0.20'},
        'P3_output_carries_value': {'supported': 'z:both@h30 follow >= 0.60', 'not_supported': 'follow <= 0.20'},
        'P4_reads_value_token': {'supported': 'val_only keep >= 0.80 and noval keep <= 0.50', 'not_supported': 'val_only keep <= 0.50'}},
    'descriptive': 'src / ent / same donors, head 28.21 alone, both heads together, attention mass by record and on the value token, '
                   'top-1, margins',
    'not_covered': 'weight-level decomposition of W_Q, W_K, W_O W_V; which upstream components write the query into R; routes that do not '
                   'pass through R; other models, unseen answer words, unseen frames',
    'disclosure': 'the code was smoke-tested on 32 discovery items (exposed) before pre-registration; thresholds were written first',
}


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定头分解产物')
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
                    recs = {str(k): {key: [i for i, (lo, hi) in enumerate(enc['offset_mapping']) if hi > off + a and lo < off + b]
                                     for key, (a, b) in info.items()} for k, info in r['records'].items()}
                    choice = []
                    for val in r['values']:
                        suffix = tok.encode(prompt + val, add_special_tokens=False)
                        assert suffix[:-1] == ids
                        choice.append(suffix[-1])
                    r = {**r, 'records': recs, 'headers': {str(k): v for k, v in r['headers'].items()}, 'prompt': prompt, 'tokens': ids,
                         'choice_ids': choice, 'gold_id': choice[r['target_record']]}
                    rows.append(UP.add_groups(tok, r))
    return rows


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定头分解产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    rows = build_materials()
    assert len(rows) == FAMILIES * 8 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    save(f'{SPLIT}-materials.json', rows)
    n = len(rows)
    body = {'schema': 'source_binding_qkov_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'heads_report_sha256': D.sha(HD.OUT / 'report.json'),
            'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in (RL, UP, HD)},
            'criteria': CRITERIA, 'materials': {'split': SPLIT, 'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': n},
            'power': {'unit': 'paired binary outcome per item (McNemar); alpha = 0.05',
                      'P2b_P3_P4': {'expected_delta': 0.40, 'p_d': 0.50, **check_phase(0.40, n, p_d=0.50)},
                      'P1_P2a': 'continuous attention mass; stated on means with family-bootstrap intervals, n = 768',
                      'caveat': 'items are clustered in 96 families and 4 frames'},
            'custody': 'the heads come from Phase 400 (holdout_heads, opened). holdout_qkov is generated here, read first by the gate stage '
                       '(behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 15, 'disk_GiB_estimate': 0.2, 'models_concurrent': 1,
                          'shape': 'one chunk per panel, inner batch 64, fixed before pre-registration'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'power': body['power']['P2b_P3_P4']}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and all(p['dependency_sha256'][m.__name__] == D.sha(m.__file__) for m in (RL, UP, HD)), \
        'source changed after pre-registration'
    assert p['materials']['sha256'] == D.sha(OUT / f'{SPLIT}-materials.json')
    D.checked(MODEL)
    return p


def donors(rows):
    index = {r['id']: i for i, r in enumerate(rows)}
    key = lambda r, s, e, o: index[f"{r['family']}-S{s}E{e}O{o}"]
    out = {'self': list(range(len(rows)))}
    out['src'] = [key(r, 1 - r['source'], r['entity'], r['order']) for r in rows]
    out['ent'] = [key(r, r['source'], 1 - r['entity'], r['order']) for r in rows]
    out['both'] = [key(r, 1 - r['source'], 1 - r['entity'], r['order']) for r in rows]
    out['same'] = [key(r, r['source'], r['entity'], 1 - r['order']) for r in rows]
    return out


def arm_list():
    arms = []
    for hk in HEADS:
        for d in ['self'] + DONORS:
            arms += [(f'q:{d}@{hk}', [('q', hk, d)], None), (f'z:{d}@{hk}', [('z', hk, d)], None)]
    arms += [('q:both@h28+h30', [('q', 'h28', 'both'), ('q', 'h30', 'both')], None),
             ('z:both@h28+h30', [('z', 'h28', 'both'), ('z', 'h30', 'both')], None)]
    arms += [(s, [], s) for s in ('val_only', 'noval', 'block')]
    return arms


def engine(rows, edits=True, stage='run'):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    t = torch
    assert snapshot(D.MODELS[MODEL]) == D.read(SRC / f'{MODEL}-identity.json')['model_snapshot']
    arms = arm_list() if edits else []
    don = donors(rows) if edits else {}
    fork = {name: min([HEADS[hk][0] for _, hk, _ in patches] + ([HEADS['h30'][0]] if span else [])) for name, patches, span in arms}
    start, N = time.monotonic(), len(rows)
    results, att = {}, {}
    with GpuLease(), t.inference_mode():
        rt = Runtime(MODEL)
        assert rt.heads == NH and rt.cfg.num_hidden_layers == NB
        hd, grp = rt.hd, rt.heads // rt.kv
        ids, pos, pe, mask = rt.context(rows)
        h, length = rt.embed(ids), ids.shape[1]
        R = length - R_FROM_END
        log(stage, 0, NB, start, rows=N, length=length)
        frow = t.zeros((N, length), dtype=t.bool, device='cuda')
        frow[:, R] = True
        frow[:, -1] = True
        recm = t.zeros((N, 4, length), dtype=t.bool, device='cuda')
        valm = t.zeros((N, 4, length), dtype=t.bool, device='cuda')
        for i, r in enumerate(rows):
            off = length - len(r['tokens'])
            assert len(r['tokens']) - R_FROM_END == r['up']['tail0'][-1] + 1
            for k in range(4):
                recm[i, k, [off + j for j in r['records'][str(k)]['span']]] = True
                valm[i, k, [off + j for j in r['records'][str(k)]['value_span']]] = True
        target = t.tensor([r['target_record'] for r in rows], device='cuda')
        ar = t.arange(N, device='cuda')
        didx = {d: t.tensor(v, device='cuda') for d, v in don.items()}
        blocks = {b: hk for hk, (b, _) in HEADS.items()}
        cap, state, hook = {}, {}, {'mode': None}
        for b in range(NB):
            if edits and b == FINAL_LAYER:
                state['noop'], state['ref'] = h[:, R:].clone(), h[:, R:].clone()
            for name in fork:
                if fork[name] == b:
                    state[name] = state['ref'].clone()
            layer = rt.block(b)
            handles = []
            if edits and b in blocks:
                hk = blocks[b]
                head = HEADS[hk][1]
                qs, ks = slice(head * hd, (head + 1) * hd), slice((head // grp) * hd, (head // grp + 1) * hd)
                cap[hk] = {x: [] for x in 'qkvz'}

                def fq(mod, args, out, hk=hk, qs=qs):
                    if hook['mode'] == 'capture':
                        cap[hk]['q'].append(out[:, R, qs].clone())
                    elif hook['mode'] and ('q', hk) in hook['mode']:
                        out[:, R, qs] = hook['mode'][('q', hk)]
                    return out

                def fk(mod, args, out, hk=hk, ks=ks):
                    if hook['mode'] == 'capture':
                        cap[hk]['k'].append(out[..., ks].clone())

                def fv(mod, args, out, hk=hk, ks=ks):
                    if hook['mode'] == 'capture':
                        cap[hk]['v'].append(out[..., ks].clone())

                def fz(mod, args, hk=hk, qs=qs):
                    if hook['mode'] == 'capture':
                        cap[hk]['z'].append(args[0][:, R, qs].clone())
                    elif hook['mode'] and ('z', hk) in hook['mode']:
                        args[0][:, R, qs] = hook['mode'][('z', hk)]
                a = layer.self_attn
                handles = [a.q_proj.register_forward_hook(fq), a.k_proj.register_forward_hook(fk), a.v_proj.register_forward_hook(fv),
                           a.o_proj.register_forward_pre_hook(fz)]
            y = t.empty_like(h)
            for lo in range(0, N, BATCH):
                hi = min(lo + BATCH, N)
                kw = {'position_ids': pos[lo:hi], 'position_embeddings': tuple(p[lo:hi] for p in pe), 'use_cache': False}
                hook['mode'] = 'capture'
                y[lo:hi] = layer(h[lo:hi], attention_mask=mask[lo:hi], **kw)
                hook['mode'] = None
            if edits and b in blocks:
                hk = blocks[b]
                c = {x: t.cat(v) for x, v in cap[hk].items()}
                cap[hk] = c
                a, (cos, sin) = layer.self_attn, pe
                rot = lambda x, co, si: x * co + rt.mod.rotate_half(x) * si
                kr = rot(a.k_norm(c['k']), cos, sin)

                def attention(q):
                    qr = rot(a.q_norm(q), cos[:, R], sin[:, R])
                    sc = ((qr[:, None, :] * kr).sum(-1) * a.scaling).float().masked_fill(~mask[:, 0, R], float('-inf'))
                    return sc.softmax(-1)
                w = attention(c['q'])
                zm = (w.to(c['v'].dtype)[:, :, None] * c['v']).sum(1).float()
                err = ((zm - c['z'].float()).norm(dim=-1) / c['z'].float().norm(dim=-1)).cpu().numpy()
                rec = (w[:, None, :] * recm).sum(-1)
                val = (w[:, None, :] * valm).sum(-1)
                stat = {'relative_error': err.tolist(), 'mass_correct': rec[ar, target].cpu().tolist(),
                        'mass_value_token': val[ar, target].cpu().tolist(),
                        'mass_other_max': rec.masked_fill(t.nn.functional.one_hot(target, 4).bool(), 0).max(1).values.cpu().tolist(),
                        'mass_all_records': rec.sum(1).cpu().tolist()}
                for d in DONORS:
                    wd = attention(c['q'][didx[d]])
                    rd = (wd[:, None, :] * recm).sum(-1)
                    tgt = target[didx[d]]
                    stat[f'{d}_mass_donor_target'] = rd[ar, tgt].cpu().tolist()
                    stat[f'{d}_mass_base_target'] = rd[ar, target].cpu().tolist()
                att[hk] = stat
                del kr, w, zm, rec, val
            if state:
                for lo in range(0, N, BATCH):
                    hi = min(lo + BATCH, N)
                    kw = {'position_ids': pos[lo:hi], 'position_embeddings': tuple(p[lo:hi] for p in pe), 'use_cache': False}
                    step = lambda key, m: layer(t.cat([h[lo:hi, :R], state[key][lo:hi]], 1), attention_mask=m, **kw)[:, R:]
                    new = {'noop': step('noop', mask[lo:hi])}
                    mref = mask[lo:hi].clone()
                    mref[:, 0, -1] = frow[lo:hi]
                    new['ref'] = step('ref', mref)
                    for name, patches, span in arms:
                        if name not in state:
                            continue
                        m = mref
                        if span and b == HEADS['h30'][0]:
                            m = mref.expand(-1, NH, -1, -1).clone()
                            cor = recm[lo:hi][t.arange(hi - lo), target[lo:hi]]
                            v = valm[lo:hi][t.arange(hi - lo), target[lo:hi]]
                            drop = {'val_only': cor & ~v, 'noval': v, 'block': cor}[span]
                            m[:, HEADS['h30'][1], R] = m[:, HEADS['h30'][1], R] & ~drop
                        hook['mode'] = {(kind, hk): cap[hk][kind][didx[d][lo:hi]] for kind, hk, d in patches if HEADS[hk][0] == b} or None
                        new[name] = step(name, m)
                        hook['mode'] = None
                    for key, v in new.items():
                        state[key][lo:hi] = v
            h = y
            for x in handles:
                x.remove()
            del layer, y
            gc.collect()
            t.cuda.empty_cache()
            if (b + 1) % 4 == 0:
                log(stage, b + 1, NB, start, block=b, arms=len(state))
        norm = rt.Norm(rt.dim, eps=rt.cfg.rms_norm_eps).to(device='cuda', dtype=t.bfloat16)
        norm.weight = t.nn.Parameter(rt.weight('model.norm.weight'), requires_grad=False)
        W = rt.weight('lm_head.weight' if 'lm_head.weight' in rt.index else 'model.embed_tokens.weight')
        logits = lambda x: t.cat([(norm(x[lo:lo + BATCH]) @ W.T).float() for lo in range(0, N, BATCH)])
        lg0 = logits(h[:, -1])
        results['base'] = rt.score(lg0, rows)
        equal = {}
        if edits:
            lgref = logits(state['ref'][:, -1])
            for key, s in state.items():
                lg = logits(s[:, -1])
                if key == 'noop':
                    equal['noop'] = bool(t.equal(lg, lg0))
                if ':self@' in key:
                    equal[key] = bool(t.equal(lg, lgref))
                results[key] = rt.score(lg, rows)
                del lg
        peak = round(t.cuda.max_memory_allocated() / 2 ** 20)
    return results, att, {'bitwise_equal': equal, 'peak_MiB': peak, 'seconds': round(time.monotonic() - start, 1), 'padded_length': length}


def summarise(results, att, rows):
    fam = np.array([r['family'] for r in rows])
    members = [np.flatnonzero(fam == f) for f in sorted(set(fam))]
    draws = np.random.default_rng(SEED).integers(0, len(members), (2000, len(members)))
    cnt = np.array([len(m) for m in members])

    def ci(v):
        v = np.asarray(v, float)
        per = np.array([v[m].sum() for m in members])
        bs = per[draws].sum(1) / cnt[draws].sum(1)
        return {'rate': round(float(v.mean()), 4), 'ci95': [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]}
    don = donors(rows)
    tgt = np.array([r['target_record'] for r in rows])
    out = {'n': len(rows), 'arms': {}, 'attention': {}}
    for k, res in results.items():
        pred = np.array([x['pred_record'] for x in res])
        body = {'keep': ci(pred == tgt), 'top1': ci([x['top1_correct'] for x in res]),
                'gold_margin': round(float(np.mean([x['gold_margin'] for x in res])), 3)}
        d = next((d for d in DONORS if f':{d}@' in k), None)
        if d:
            body['follow'] = ci(pred == tgt[don[d]])
        out['arms'][k] = body
    for hk, stat in att.items():
        out['attention'][hk] = {key: (ci(v) if key != 'relative_error' else
                                      {'median': round(float(np.median(v)), 5), 'max': round(float(np.max(v)), 5)}) for key, v in stat.items()}
    return out


def verdicts(s, meta):
    keep = lambda k: s['arms'][k]['keep']['rate']
    follow = lambda k: s['arms'][k]['follow']['rate']
    a = s['attention']['h30']
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    ok = all(meta['bitwise_equal'].values()) and len(meta['bitwise_equal']) == 5 and keep('ref') >= .90 and keep('block') <= .50 \
        and a['relative_error']['median'] <= .05
    v = {'controls_valid': bool(ok)}
    names = ('P1_attends_correct_record', 'P2a_query_redirects_attention', 'P2b_query_redirects_answer', 'P3_output_carries_value',
             'P4_reads_value_token')
    if not ok:
        v.update({k: 'controls_invalid' for k in names})
        return v
    mc, mo = a['mass_correct']['rate'], a['mass_other_max']['rate']
    v['P1_attends_correct_record'] = tri(mc >= .50 and mc >= 3 * mo, mc < .25)
    md = a['both_mass_donor_target']['rate']
    v['P2a_query_redirects_attention'] = tri(md >= .50, md < .25)
    v['P2b_query_redirects_answer'] = tri(follow('q:both@h30') >= .60, follow('q:both@h30') <= .20)
    v['P3_output_carries_value'] = tri(follow('z:both@h30') >= .60, follow('z:both@h30') <= .20)
    v['P4_reads_value_token'] = tri(keep('val_only') >= .80 and keep('noval') <= .50, keep('val_only') <= .50)
    return v


def pilot(smoke=False):
    rows = D.read(SRC / f'{MODEL}-materials.json')['splits']['discovery']
    tok = RL.tokenizer()
    rows = [UP.add_groups(tok, r) for r in rows][:32 if smoke else None]
    if not smoke:
        checked()
        assert not (OUT / 'pilot.json').exists()
    results, att, meta = engine(rows, stage='pilot')
    s = summarise(results, att, rows)
    v = verdicts(s, meta)
    if not smoke:
        save('pilot.json', {'schema': 'source_binding_qkov_pilot.v1', 'run_id': RUN, 'note': 'discovery (exposed); exploratory, nothing selected',
                            'verdict_rules_applied_for_description': v, 'summary': s, **meta})
    print(json.dumps({'smoke': smoke, 'meta': meta, 'verdicts': v, 'attention': s['attention'],
                      'arms': {k: {x: (y['rate'] if isinstance(y, dict) else y) for x, y in a.items()} for k, a in s['arms'].items()}}), flush=True)


def gate():
    checked()
    assert (OUT / 'pilot.json').exists() and not (OUT / 'gate.json').exists(), 'holdout behaviour already read'
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    results, _, meta = engine(rows, edits=False, stage='gate')
    save('gate-items.json', results['base'])
    g = D.behavior(rows, results['base'])
    save('gate.json', {'schema': 'source_binding_qkov_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre = checked()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    if g['passed']:
        results, att, meta = engine(rows, stage='run')
        save('items.json', {'results': results, 'attention': att})
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, att, rows)
        v = verdicts(s, meta)
    else:
        s, meta = {'note': 'behaviour gate failed; no edit was run'}, {}
        v = {k: 'behavior_ineligible' for k in CRITERIA['predictions_on_holdout'] if k != 'controls_valid'}
    report = {'schema': 'source_binding_qkov_report.v1', 'run_id': RUN, 'split': SPLIT, 'heads': {k: list(x) for k, x in HEADS.items()},
              'gate_passed': g['passed'], 'prereg_sha256': D.sha(OUT / 'prereg.json'), 'pilot_sha256': D.sha(OUT / 'pilot.json'),
              'gate_sha256': D.sha(OUT / 'gate.json'), 'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None,
              'criteria': pre['criteria'], 'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_qkov read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-qkov-holdout', {'verdicts': v, 'gate_passed': g['passed'], 'heads': report['heads'],
                                                                   'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['prereg', 'pilot', 'gate', 'run'])
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    pilot(a.smoke) if a.stage == 'pilot' else {'prereg': prereg, 'gate': gate, 'run': run}[a.stage]()
