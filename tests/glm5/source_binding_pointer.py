"""Is the request held at the relay position a pointer to the position of the record?  Qwen3-14B BF16, source binding.

Pre-registered test of the post-hoc observation of Claude Phase 404.  The layout of the four records is varied on two
independent binary factors: o (which archive block comes first) and x (the order of the two entities inside both
blocks).  Configuration iso throughout (blocks 24-27 the relay position R sees only the four records; from block 24
on the final position reads only R).  The residual state of R at the input of block 24 is replaced by that of a donor
prompt of the same family:
  layout donors   same request, other layout (o, x or both flipped): the content account predicts the correct record,
                  the position account predicts the record that sits where the donor's target sat (three different ones)
  crossed donors  other request and other layout, chosen so that the donor's target sits exactly where the base target
                  sits: content predicts a wrong record, position predicts the correct one
  same-layout donors (other request, same layout): both accounts agree (replication of Phase 404)
Stages: prereg -> pilot (own exposed panel; nothing selected) -> gate -> run.
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
import source_binding_twodim as TD
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
PANELS = TD.PANELS + [TD.OUT / 'holdout_twodim-materials.json']
OUT = D.ROOT / 'tests/glm5/result/source_binding_pointer_v1'
RUN, SEED = 'SOURCE-BINDING-POINTER-20261111-v1', 202611110405
SPLITS = {'pilot_layout': 32, 'holdout_layout': 96}
FINAL_LAYER, ISO_END, BATCH, NB, NH, R_FROM_END, TOL, HEAD = 24, 28, 128, 40, 40, UP.R_FROM_END, 1.0, (30, 4)
# donor = (source, entity, order, x) with 1 meaning 'flipped relative to the base prompt'
DONORS = {'self': (0, 0, 0, 0),
          'lay:o': (0, 0, 1, 0), 'lay:x': (0, 0, 0, 1), 'lay:o+x': (0, 0, 1, 1),
          'cross:src': (1, 0, 1, 1), 'cross:ent': (0, 1, 0, 1), 'cross:both': (1, 1, 1, 0),
          'req:src': (1, 0, 0, 0), 'req:ent': (0, 1, 0, 0), 'req:both': (1, 1, 0, 0)}
LAYOUT, CROSSED = ['lay:o', 'lay:x', 'lay:o+x'], ['cross:src', 'cross:ent', 'cross:both']
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'each family: 2 archives x 2 entities x 2 archive orders (o) x 2 entity orders inside the blocks (x) = 16 prompts with the '
                'same four records. With x = 0 a prompt is identical to the frozen v1 design. pilot_layout: 32 families (exposed after the '
                'pilot); holdout_layout: 96 families. Entity names occur in no earlier panel of this line; the 16 discovery code words, '
                '4 seen frames.',
    'configuration': 'iso in every arm and in the reference: blocks 24-27 R attends only to the four records, itself and the template head; '
                     'from block 24 on the final position attends only to R (and itself)',
    'edit': 'residual state of R at the input of block 24 replaced by that of the donor prompt (same family)',
    'donors': {'lay:o / lay:x / lay:o+x': 'same archive and entity; archive order, entity order or both flipped',
               'cross:src / cross:ent / cross:both': 'other archive (with o and x flipped), other entity (x flipped), both (o flipped): the '
                                                     'donor target has the ordinal position of the base target',
               'req:src / req:ent / req:both': 'other request, same layout', 'self': 'identity check'},
    'outcome': 'the chosen record among four (chance 0.25). follow_content = the record the donor prompt asks for; follow_position = the '
               'record of the base prompt whose ordinal position (1st to 4th record) equals that of the donor target in the donor prompt. '
               'All items; family-bootstrap 95 % intervals.',
    'tail_recomputation': 'as in Phase 403: arms recompute the 11 positions >= R. Control: the no-edit arm recomputed from block 0 gives the '
                          'same forced choice as the base forward on every item, maximal score difference <= 1.0',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate on all 1,536 prompts. Edits run only if the gate passed; '
                'otherwise behavior_ineligible.',
    'predictions_on_holdout': {
        'controls_valid': 'tail recomputation control holds; ref_iso correct >= 0.80; swap self gives the choices of ref_iso on every item',
        'L1_archive_order': {'supported': 'lay:o: follow_position >= 0.60 and follow_content <= 0.30', 'not_supported': 'follow_content >= 0.60'},
        'L2_entity_order': {'supported': 'lay:x: follow_position >= 0.60 and follow_content <= 0.30', 'not_supported': 'follow_content >= 0.60'},
        'L3_both_orders': {'supported': 'lay:o+x: follow_position >= 0.60 and follow_content <= 0.30', 'not_supported': 'follow_content >= 0.60'},
        'L4_crossed': {'supported': 'for each of cross:src, cross:ent, cross:both: follow_position (the correct record) >= 0.60 and '
                                    'follow_content <= 0.30', 'not_supported': 'for some crossed donor follow_content >= 0.60'}},
    'descriptive': 'same-layout donors; split by frame; attention mass of head 30.4 on the correct record; top-1',
    'not_covered': 'where names are turned into positions; whether the pointer is ordinal or absolute token position (the two are nearly '
                   'the same here: records differ by at most one token in length); the natural configuration; other models',
    'disclosure': 'the code was smoke-tested on 4 pilot families (exposed) before pre-registration; thresholds were written first. '
                  'The prediction restates an observation made after holdout_twodim was opened (Phase 404); that panel is not reused.',
}


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定位置指针产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def log(*a, **kw):
    RL.OUT = OUT
    RL.log(*a, **kw)


def material2(family, split, s, e, o, x, frame, names, values):
    """The v1 material with one more factor: x = 1 flips the entity order inside both archive blocks."""
    pieces, records, heads, offset = [], {}, {}, 0

    def add(text):
        nonlocal offset
        span = (offset, offset + len(text))
        pieces.append(text)
        offset += len(text)
        return span
    add('Two independent archives assign codes to the same entities. Use only the archive named in the question.\n')
    for src in ([0, 1] if o == 0 else [1, 0]):
        heads[src] = add(f'Archive {D.SOURCES[src]}:\n')
        entorder = [0, 1] if (family + src + x) % 2 == 0 else [1, 0]
        for ent in entorder:
            text = D.sentence(frame, names[ent], values[src * 2 + ent])
            a, b = add(text + '\n')
            vs = a + text.index(values[src * 2 + ent])
            records[src * 2 + ent] = {'span': [a, b - 1], 'value_span': [vs, vs + len(values[src * 2 + ent])],
                                      'name_span': [a + text.index(names[ent]), a + text.index(names[ent]) + len(names[ent])]}
    base = D.material(family, split, s, e, o, frame, names, values)
    q = base['content'][base['query_span'][0]:base['query_span'][1]]
    qa, qb = add(q)
    shift = qa - base['query_span'][0]
    assert shift == 0
    r = {**base, 'id': base['id'] + f'X{x}', 'x': x, 'content': ''.join(pieces), 'records': records, 'headers': heads}
    if x == 0:
        assert r['content'] == base['content'] and r['records'] == base['records']
    return r


def build_materials(split, taken):
    tok = RL.tokenizer()
    rows = []
    for family in range(SPLITS[split]):
        rnd = stream(f'names|{split}', family)
        names = []
        while len(names) < 2:
            name = ''.join(rnd.choice(list('bdfgklmnprstvz')) + rnd.choice(list('aeiou')) for _ in range(3)).capitalize()
            if name not in taken:
                taken.add(name)
                names.append(name)
        values = rnd.choice(D.WORDS[:16], 4, replace=False).tolist()
        for s in (0, 1):
            for e in (0, 1):
                for o in (0, 1):
                    for x in (0, 1):
                        r = material2(family, split, s, e, o, x, family % 4, names, values)
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


def all_materials():
    taken = {n for rows in D.read(SRC / f'{MODEL}-materials.json')['splits'].values() for r in rows for n in r['names']}
    for p in PANELS:
        taken |= {n for r in D.read(p) for n in r['names']}
    return {split: build_materials(split, taken) for split in SPLITS}


def expectations(rows):
    """Donor index, content-predicted record and position-predicted record for every donor type."""
    index = {r['id']: i for i, r in enumerate(rows)}
    ordinal = lambda r: {k: i for i, k in enumerate(sorted(range(4), key=lambda k: r['records'][str(k)]['span'][0]))}
    out = {}
    for name, (fs, fe, fo, fx) in DONORS.items():
        don, content, position = [], [], []
        for r in rows:
            j = index[f"{r['family']}-S{r['source'] ^ fs}E{r['entity'] ^ fe}O{r['order'] ^ fo}X{r['x'] ^ fx}"]
            d = rows[j]
            inv = {v: k for k, v in ordinal(r).items()}
            don.append(j)
            content.append(d['target_record'])
            position.append(inv[ordinal(d)[d['target_record']]])
        out[name] = {'donor': don, 'content': content, 'position': position}
    tgt = [r['target_record'] for r in rows]
    assert all(out[k]['content'] == tgt and all(a != b for a, b in zip(out[k]['position'], tgt)) for k in LAYOUT)
    assert all(out[k]['position'] == tgt and all(a != b for a, b in zip(out[k]['content'], tgt)) for k in CROSSED)
    assert all(out[k]['position'] == out[k]['content'] for k in ('self', 'req:src', 'req:ent', 'req:both'))
    assert len({tuple(out[k]['position']) for k in LAYOUT}) == 3
    return out


DEPS = (RL, UP, HD, QK, QY, RQ, TD)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定位置指针产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    mats = all_materials()
    for split, rows in mats.items():
        assert len(rows) == SPLITS[split] * 16 and len({tuple(r['tokens']) for r in rows}) == len(rows)
        expectations(rows)
        save(f'{split}-materials.json', rows)
    n = len(mats['holdout_layout'])
    body = {'schema': 'source_binding_pointer_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'twodim_report_sha256': D.sha(TD.OUT / 'report.json'),
            'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'materials': {split: {'sha256': D.sha(OUT / f'{split}-materials.json'), 'families': SPLITS[split],
                                                        'rows': len(rows)} for split, rows in mats.items()},
            'power': {'unit': 'paired binary outcome per item (McNemar); alpha = 0.05',
                      'L1_L4': {'expected_delta': 0.30, 'p_d': 0.50, **check_phase(0.30, n, p_d=0.50)},
                      'caveat': 'items are clustered in 96 families and 4 frames'},
            'custody': 'pilot_layout is exposed by the pilot stage. holdout_layout is read first by the gate stage (behaviour only) and '
                       'then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 10, 'disk_GiB_estimate': 0.3, 'models_concurrent': 1,
                          'shape': 'one chunk per panel, inner batch 128, arms recompute 11 positions; fixed before pre-registration'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'power': body['power']['L1_L4']}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and all(p['dependency_sha256'][m.__name__] == D.sha(m.__file__) for m in DEPS), \
        'source changed after pre-registration'
    assert all(v['sha256'] == D.sha(OUT / f'{k}-materials.json') for k, v in p['materials'].items())
    D.checked(MODEL)
    return p


def engine(rows, edits=True, stage='run'):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    t, F = torch, torch.nn.functional
    assert snapshot(D.MODELS[MODEL]) == D.read(SRC / f'{MODEL}-identity.json')['model_snapshot']
    exp = expectations(rows) if edits else {}
    forks = {'noop': 0, 'ref_iso': FINAL_LAYER, **{k: FINAL_LAYER for k in exp}} if edits else {}
    start, N = time.monotonic(), len(rows)
    results, mass = {}, {}
    with GpuLease(), t.inference_mode():
        rt = Runtime(MODEL)
        hd, kv, grp = rt.hd, rt.kv, rt.heads // rt.kv
        ids, pos, pe, mask = rt.context(rows)
        h, length = rt.embed(ids), ids.shape[1]
        R = length - R_FROM_END
        T = length - R
        log(stage, 0, NB, start, rows=N, length=length, arms=len(forks))
        frow = t.zeros((N, length), dtype=t.bool, device='cuda')
        frow[:, R] = True
        frow[:, -1] = True
        cor = t.zeros((N, length), dtype=t.bool, device='cuda')
        isorow = t.zeros((N, length), dtype=t.bool, device='cuda')
        for i, r in enumerate(rows):
            off = length - len(r['tokens'])
            assert len(r['tokens']) - R_FROM_END == r['up']['tail0'][-1] + 1
            cor[i, [off + j for j in r['up']['rec_correct']]] = True
            isorow[i, [R] + [off + j for g in ['tpl_head'] + QY.UNIONS['records4'] for j in r['up'][g]]] = True
        isorow &= mask[:, 0, R]
        rot = lambda x, co, si: x * co + rt.mod.rotate_half(x) * si
        state, cap = {}, {}
        for b in range(NB):
            for name, l in forks.items():
                if l == b:
                    state[name] = h[:, R:].clone()
                    if name in exp:
                        state[name][:, 0] = h[t.tensor(exp[name]['donor'], device='cuda'), R]
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
                    tm = tm0.clone()
                    if name != 'noop' and b >= FINAL_LAYER:
                        tm[:, 0, -1] = frow[lo:hi]
                        if b < ISO_END:
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
        meta['swap_self_same_choice'] = same('self', 'ref_iso')
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
    exp = expectations(rows)
    tgt = np.array([r['target_record'] for r in rows])
    frame = np.array([r['frame'] for r in rows])
    out = {'n': len(rows), 'arms': {}}
    for k, res in results.items():
        pred = np.array([x['pred_record'] for x in res])
        body = {'correct': ci(pred == tgt), 'top1': ci([x['top1_correct'] for x in res]), 'mass': ci(mass[k]) if k in mass else None}
        if k in exp:
            fp, fc = pred == np.array(exp[k]['position']), pred == np.array(exp[k]['content'])
            body.update(follow_position=ci(fp), follow_content=ci(fc),
                        by_frame={str(f): [round(float(fp[frame == f].mean()), 4), round(float(fc[frame == f].mean()), 4)] for f in sorted(set(frame))})
        out['arms'][k] = body
    return out


def verdicts(s, meta):
    fp = lambda k: s['arms'][k]['follow_position']['rate']
    fc = lambda k: s['arms'][k]['follow_content']['rate']
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    names = [k for k in CRITERIA['predictions_on_holdout'] if k != 'controls_valid']
    ok = meta['tail_control_ok'] and meta['swap_self_same_choice'] and s['arms']['ref_iso']['correct']['rate'] >= .80
    v = {'controls_valid': bool(ok)}
    if not ok:
        v.update({k: 'controls_invalid' for k in names})
        return v
    for name, k in zip(names[:3], LAYOUT):
        v[name] = tri(fp(k) >= .60 and fc(k) <= .30, fc(k) >= .60)
    v[names[3]] = tri(all(fp(k) >= .60 and fc(k) <= .30 for k in CROSSED), any(fc(k) >= .60 for k in CROSSED))
    return v


def show(s, meta, v, smoke=None):
    table = {k: [a['correct']['rate'], a.get('follow_position', {}).get('rate'), a.get('follow_content', {}).get('rate'),
                 a['mass']['rate'] if a['mass'] else None] for k, a in s['arms'].items()}
    print(json.dumps({'smoke': smoke, 'meta': meta, 'verdicts': v, 'correct_followPosition_followContent_mass': table}), flush=True)


def pilot(smoke=False):
    if smoke:
        rows = [r for r in all_materials()['pilot_layout'] if int(r['family'][-4:]) < 4]
    else:
        checked()
        assert not (OUT / 'pilot.json').exists()
        rows = D.read(OUT / 'pilot_layout-materials.json')
    results, mass, meta = engine(rows, stage='pilot')
    s = summarise(results, mass, rows)
    v = verdicts(s, meta)
    g = D.behavior(rows, results['base'])
    meta['pilot_behaviour'] = {k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}
    if not smoke:
        save('pilot.json', {'schema': 'source_binding_pointer_pilot.v1', 'run_id': RUN, 'note': 'pilot_layout (exposed); exploratory, nothing selected',
                            'verdict_rules_applied_for_description': v, 'summary': s, **meta})
    show(s, meta, v, smoke)


def gate():
    checked()
    assert (OUT / 'pilot.json').exists() and not (OUT / 'gate.json').exists(), 'holdout behaviour already read'
    rows = D.read(OUT / 'holdout_layout-materials.json')
    results, _, meta = engine(rows, edits=False, stage='gate')
    save('gate-items.json', results['base'])
    g = D.behavior(rows, results['base'])
    xcells = {f'X{x}': float(np.mean([it['forced_correct'] for r, it in zip(rows, results['base']) if r['x'] == x])) for x in (0, 1)}
    save('gate.json', {'schema': 'source_binding_pointer_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g,
                       'forced_accuracy_by_x': xcells, **meta})
    print(json.dumps({**{k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}, 'by_x': xcells}), flush=True)


def run():
    from tools import provenance
    pre = checked()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows = D.read(OUT / 'holdout_layout-materials.json')
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
    report = {'schema': 'source_binding_pointer_report.v1', 'run_id': RUN, 'split': 'holdout_layout', 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'pilot_sha256': D.sha(OUT / 'pilot.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'verdicts': v, 'summary': s,
              **meta, 'formal_evidence_promotion': 0, 'custody': 'holdout_layout read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-pointer-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                      'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['prereg', 'pilot', 'gate', 'run'])
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    pilot(a.smoke) if a.stage == 'pilot' else {'prereg': prereg, 'gate': gate, 'run': run}[a.stage]()
