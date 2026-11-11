"""Where do the names of the question turn into positions?  Qwen3-14B BF16, source binding.

Follows Claude Phase 405: at block 24 the relay position R holds the request mostly as a pointer to the position of
the record (archive block, line).  Here the residual state of a group of question-side tokens is replaced, at the
input of one block l, by the state the same tokens have in a prompt of the same family with the same request but
another record layout.  Tokens and names are identical in base and donor, so a pure content carrier is unaffected by
the swap; if the choice moves to the record that sits where the donor's target sat, the swapped tokens carried the
position at block l and passed it on.
Natural configuration: nothing else is edited (no restriction on R or on the final position); every later position
is recomputed.  Chunks of 256 prompts, inner batch 128; per arm only the positions from the first swapped token on.
Stages: prereg -> pilot (own exposed panel: carriers are selected here) -> freeze -> gate -> run.
"""
import argparse, gc, json, sys, time
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
import source_binding_pointer as PT
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_nameloc_v1'
RUN, SEED = 'SOURCE-BINDING-NAMELOC-20261111-v1', 202611110406
SPLITS = {'pilot_name': 16, 'holdout_name': 48}
CHUNK, BATCH, NB, NH, R_FROM_END, TOL = 256, 128, 40, 40, UP.R_FROM_END, 1.0
LAYERS = [0, 8, 12, 14, 16, 18, 20, 22, 24, 26]
# series: tokens swapped, donor type (names of source_binding_pointer.DONORS)
SERIES = {'qent<lay:x': (['qent'], 'lay:x'), 'qsrc<lay:o': (['qsrc'], 'lay:o'), 'names<lay:o+x': (['qsrc', 'qent'], 'lay:o+x'),
          'qrest<lay:o+x': (['qrest'], 'lay:o+x'), 'instr<lay:o+x': (['instr'], 'lay:o+x'), 'R<lay:o+x': (['R'], 'lay:o+x'),
          'tail0<lay:o+x': (['tail0'], 'lay:o+x'), 'qsrc<req:src': (['qsrc'], 'req:src'), 'qent<req:ent': (['qent'], 'req:ent')}
LAY = [k for k in SERIES if '<lay' in k]
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'layout-factorial families as in Phase 405 (2 archives x 2 entities x 2 archive orders x 2 entity orders = 16 prompts). '
                'pilot_name: 16 families (exposed by the pilot); holdout_name: 48 families (768 prompts). Entity names occur in no earlier '
                'panel of this line; the 16 discovery code words, 4 seen frames.',
    'edit': 'at the input of block l the residual state of the listed tokens is replaced by the state of the same tokens in the donor '
            'prompt (same family). Natural configuration: no other edit, all later positions recomputed.',
    'series': {'qent<lay:x': 'entity-name tokens of the question; donor: same request, entity order inside the blocks flipped',
               'qsrc<lay:o': 'archive-name token of the question; donor: same request, archive order flipped',
               'names<lay:o+x': 'both names; donor: both orders flipped', 'qrest<lay:o+x': 'the other question tokens',
               'instr<lay:o+x': 'the instruction sentence', 'tail0<lay:o+x': '<|im_end|> of the user turn', 'R<lay:o+x': 'the relay position',
               'qsrc<req:src': 'control: archive-name token from a prompt that asks for the other archive (same layout)',
               'qent<req:ent': 'control: entity-name tokens from a prompt that asks for the other entity (same layout); only items whose '
                               'two names have the same number of tokens'},
    'layers': [0, 8, 12, 14, 16, 18, 20, 22, 24, 26],
    'outcome': 'the chosen record among four (chance 0.25). follow_position = the record of the base prompt at the ordinal position of the '
               'donor target; follow_content = the record the donor prompt asks for (for layout donors: the correct record). '
               'All items; family-bootstrap 95 % intervals.',
    'tail_recomputation': 'arms recompute the positions from the first swapped token on with their own code path. Control: the no-edit arm '
                          'recomputed from block 0 (longest tail) gives the same forced choice as the base forward on every item, maximal '
                          'score difference <= 1.0',
    'selection_on_pilot': 'for each of the 7 layout series: peak = the largest follow_position over the 10 layers, L* = its layer (smallest '
                          'on ties). carrier if peak >= 0.50; non-carrier if peak <= 0.25; otherwise undetermined (no prediction).',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed; otherwise behavior_ineligible.',
    'predictions_on_holdout': {
        'controls_valid': 'tail recomputation control holds; qsrc<req:src at layer 0 follow_content >= 0.80',
        'N1_carriers': {'supported': 'every carrier series: follow_position at its L* >= 0.40 and above follow_content',
                        'not_supported': 'some carrier series: follow_position at its L* < 0.20', 'not_applicable': 'no carrier in the pilot'},
        'N2_non_carriers': {'supported': 'every non-carrier series: follow_position <= 0.35 at all 10 layers',
                            'not_supported': 'some non-carrier series reaches >= 0.50', 'not_applicable': 'no non-carrier in the pilot'}},
    'descriptive': 'the full layer profile of every series; chosen-record classes; onset layer (first layer with at least half of the peak)',
    'not_covered': 'which heads write the position into the carrier tokens; ordinal versus absolute token position; a swap at one block can '
                   'be overwritten by later blocks, so a low value does not prove that the tokens never carry the position; other models',
    'disclosure': 'the code was smoke-tested on 2 pilot families (exposed) before pre-registration; thresholds were written first. '
                  'The smoke test (layers 0, 6, 10, 14, 16, 18, 20, 22) showed changes only from block 16 on, so the grid was made denser '
                  'there and extended to blocks 24 and 26; thresholds unchanged.',
}


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定名字定位产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def log(*a, **kw):
    RL.OUT = OUT
    RL.log(*a, **kw)


def all_materials():
    taken = {n for rows in D.read(SRC / f'{MODEL}-materials.json')['splits'].values() for r in rows for n in r['names']}
    for p in PT.PANELS + [PT.OUT / 'pilot_layout-materials.json', PT.OUT / 'holdout_layout-materials.json']:
        taken |= {n for r in D.read(p) for n in r['names']}
    PT.SPLITS.update(SPLITS)
    return {split: PT.build_materials(split, taken) for split in SPLITS}


DEPS = (RL, UP, HD, QK, QY, RQ, TD, PT)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定名字定位产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    mats = all_materials()
    for split, rows in mats.items():
        assert len(rows) == SPLITS[split] * 16 and len({tuple(r['tokens']) for r in rows}) == len(rows)
        PT.expectations(rows)
        save(f'{split}-materials.json', rows)
    n = len(mats['holdout_name'])
    body = {'schema': 'source_binding_nameloc_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'pointer_report_sha256': D.sha(PT.OUT / 'report.json'),
            'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'materials': {split: {'sha256': D.sha(OUT / f'{split}-materials.json'), 'families': SPLITS[split],
                                                        'rows': len(rows)} for split, rows in mats.items()},
            'power': {'unit': 'paired binary outcome per item (McNemar); alpha = 0.05',
                      'N1_N2': {'expected_delta': 0.30, 'p_d': 0.50, **check_phase(0.30, n, p_d=0.50)},
                      'caveat': 'items are clustered in 48 families and 4 frames; 90 arms are reported, verdicts use only the frozen cells'},
            'custody': 'pilot_name is exposed by the pilot stage, which fixes carriers and layers. holdout_name is read first by the gate '
                       'stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 30, 'disk_GiB_estimate': 0.3, 'models_concurrent': 1,
                          'shape': 'chunks of 256 prompts, inner batch 128, per-arm tails; fixed before pre-registration'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'power': body['power']['N1_N2']}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and all(p['dependency_sha256'][m.__name__] == D.sha(m.__file__) for m in DEPS), \
        'source changed after pre-registration'
    assert all(v['sha256'] == D.sha(OUT / f'{k}-materials.json') for k, v in p['materials'].items())
    D.checked(MODEL)
    return p


def tokens_of(r, groups):
    return [j for g in groups for j in ([len(r['tokens']) - R_FROM_END] if g == 'R' else r['up'][g])]


def engine(rows, edits=True, stage='run'):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    t, F = torch, torch.nn.functional
    assert snapshot(D.MODELS[MODEL]) == D.read(SRC / f'{MODEL}-identity.json')['model_snapshot']
    start, total = time.monotonic(), len(rows)
    results, valid = {}, {}
    with GpuLease(), t.inference_mode():
        rt = Runtime(MODEL)
        hd, kv, grp = rt.hd, rt.kv, rt.heads // rt.kv
        rot = lambda x, co, si: x * co + rt.mod.rotate_half(x) * si
        log(stage, 0, total * NB, start, rows=total)
        for c0 in range(0, total, CHUNK):
            rs = rows[c0:c0 + CHUNK]
            N = len(rs)
            exp = PT.expectations(rs) if edits else {}
            ids, pos, pe, mask = rt.context(rs)
            h, length = rt.embed(ids), ids.shape[1]
            off = [length - len(r['tokens']) for r in rs]
            plan = {}
            if edits:
                for sname, (groups, dname) in SERIES.items():
                    bi, bp, di, dp, ok = [], [], [], [], []
                    for i, r in enumerate(rs):
                        j = exp[dname]['donor'][i]
                        a, b = tokens_of(r, groups), tokens_of(rs[j], groups)
                        good = len(a) == len(b)
                        ok.append(good)
                        if good:
                            if '<lay' in sname:
                                assert [r['tokens'][x] for x in a] == [rs[j]['tokens'][x] for x in b]
                            bi += [i] * len(a)
                            bp += [off[i] + x for x in a]
                            di += [j] * len(b)
                            dp += [off[j] + x for x in b]
                    assert all(ok) or sname == 'qent<req:ent'
                    P = min(off[i] + min(tokens_of(r, groups)) for i, r in enumerate(rs))
                    idx = tuple(t.tensor(v, device='cuda') for v in (bi, bp, di, dp))
                    for l in LAYERS:
                        plan[f'{sname}@{l}'] = (l, P, idx)
                    valid.setdefault(sname, []).extend(ok)
                plan['noop'] = (0, min(p[1] for p in plan.values()), None)
            state, cap = {}, {}
            for b in range(NB):
                for name, (l, P, idx) in plan.items():
                    if l == b:
                        state[name] = h[:, P:].clone()
                        if idx is not None:
                            state[name][idx[0], idx[1] - P] = h[idx[2], idx[3]]
                layer = rt.block(b)
                a = layer.self_attn
                handles = []
                if state:
                    handles = [a.k_proj.register_forward_hook(lambda m, i, o: cap.__setitem__('k', o)),
                               a.v_proj.register_forward_hook(lambda m, i, o: cap.__setitem__('v', o))]
                y = t.empty_like(h)
                for lo in range(0, N, BATCH):
                    hi = min(lo + BATCH, N)
                    B = hi - lo
                    cos, sin = pe[0][lo:hi], pe[1][lo:hi]
                    y[lo:hi] = layer(h[lo:hi], attention_mask=mask[lo:hi], position_ids=pos[lo:hi], position_embeddings=(cos, sin), use_cache=False)
                    kb, vb = cap.get('k'), cap.get('v')
                    for name in state:
                        P = plan[name][1]
                        T = length - P
                        x = state[name][lo:hi]
                        hn = layer.input_layernorm(x)
                        q = a.q_norm(a.q_proj(hn).view(B, T, NH, hd)).transpose(1, 2)
                        k = a.k_norm(t.cat([kb[:, :P], a.k_proj(hn)], 1).view(B, length, kv, hd)).transpose(1, 2)
                        v = t.cat([vb[:, :P], a.v_proj(hn)], 1).view(B, length, kv, hd).transpose(1, 2)
                        q, k = rot(q, cos[:, None, P:], sin[:, None, P:]), rot(k, cos[:, None], sin[:, None])
                        o = F.scaled_dot_product_attention(q, rt.mod.repeat_kv(k, grp), rt.mod.repeat_kv(v, grp), attn_mask=mask[lo:hi][:, :, P:, :])
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
                    log(stage, c0 * NB + (b + 1) * N, total * NB, start, block=b, arms=len(state))
            norm = rt.Norm(rt.dim, eps=rt.cfg.rms_norm_eps).to(device='cuda', dtype=t.bfloat16)
            norm.weight = t.nn.Parameter(rt.weight('model.norm.weight'), requires_grad=False)
            W = rt.weight('lm_head.weight' if 'lm_head.weight' in rt.index else 'model.embed_tokens.weight')
            logits = lambda x: t.cat([(norm(x[lo:lo + BATCH]) @ W.T).float() for lo in range(0, N, BATCH)])
            results.setdefault('base', []).extend(rt.score(logits(h[:, -1]), rs))
            for key, s in state.items():
                results.setdefault(key, []).extend(rt.score(logits(s[:, -1]), rs))
            del W, norm, h, state, ids, pos, pe, mask, plan
            gc.collect()
            t.cuda.empty_cache()
        peak = round(t.cuda.max_memory_allocated() / 2 ** 20)
    meta = {'peak_MiB': peak, 'seconds': round(time.monotonic() - start, 1)}
    if edits:
        meta['noop_same_choice'] = all(x['pred_record'] == y['pred_record'] for x, y in zip(results['noop'], results['base']))
        meta['noop_max_score_difference'] = max(abs(p - q) for x, y in zip(results['noop'], results['base']) for p, q in zip(x['scores'], y['scores']))
        meta['tail_control_ok'] = bool(meta['noop_same_choice'] and meta['noop_max_score_difference'] <= TOL)
    return results, valid, meta


def summarise(results, valid, rows):
    exp = PT.expectations(rows)
    fam = np.array([r['family'] for r in rows])
    s0, e0 = np.array([r['source'] for r in rows]), np.array([r['entity'] for r in rows])
    rng = np.random.default_rng(SEED)

    def ci(v, keep):
        v = np.asarray(v, float)[keep]
        f = fam[keep]
        members = [np.flatnonzero(f == x) for x in sorted(set(f))]
        draws = rng.integers(0, len(members), (1000, len(members)))
        per, cnt = np.array([v[m].sum() for m in members]), np.array([len(m) for m in members])
        bs = per[draws].sum(1) / cnt[draws].sum(1)
        return {'rate': round(float(v.mean()), 4), 'ci95': [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]}
    out = {'n': len(rows), 'base_correct': round(float(np.mean([x['forced_correct'] for x in results['base']])), 4), 'series': {}}
    for sname, (_, dname) in SERIES.items():
        keep = np.array(valid[sname], bool)
        body = {'n': int(keep.sum()), 'layers': {}}
        for l in LAYERS:
            pred = np.array([x['pred_record'] for x in results[f'{sname}@{l}']])
            ps, pe = pred // 2, pred % 2
            cls = {'correct': (ps == s0) & (pe == e0), 'other_entity': (ps == s0) & (pe != e0), 'other_source': (ps != s0) & (pe == e0),
                   'neither': (ps != s0) & (pe != e0)}
            body['layers'][str(l)] = {'follow_position': ci(pred == np.array(exp[dname]['position']), keep),
                                      'follow_content': ci(pred == np.array(exp[dname]['content']), keep),
                                      'classes': {k: round(float(v[keep].mean()), 4) for k, v in cls.items()}}
        fp = [body['layers'][str(l)]['follow_position']['rate'] for l in LAYERS]
        body.update(peak=max(fp), peak_layer=LAYERS[int(np.argmax(fp))],
                    onset_layer=next(l for l, v in zip(LAYERS, fp) if v >= max(fp) / 2))
        out['series'][sname] = body
    return out


def show(s, meta, extra=None):
    table = {k: [[v['layers'][str(l)]['follow_position']['rate'], v['layers'][str(l)]['follow_content']['rate']] for l in LAYERS]
             for k, v in s['series'].items()}
    print(json.dumps({'meta': meta, 'layers': LAYERS, 'followPosition_followContent': table, **(extra or {})}), flush=True)


def pilot(smoke=False):
    if smoke:
        rows = [r for r in all_materials()['pilot_name'] if int(r['family'][-4:]) < 2]
    else:
        checked()
        assert not (OUT / 'pilot.json').exists()
        rows = D.read(OUT / 'pilot_name-materials.json')
    results, valid, meta = engine(rows, stage='pilot')
    s = summarise(results, valid, rows)
    carriers = {k: s['series'][k]['peak_layer'] for k in LAY if s['series'][k]['peak'] >= .50}
    non = [k for k in LAY if s['series'][k]['peak'] <= .25]
    if not smoke:
        assert meta['tail_control_ok']
        save('pilot.json', {'schema': 'source_binding_nameloc_pilot.v1', 'run_id': RUN, 'summary': s, 'carriers': carriers, 'non_carriers': non, **meta})
        save('frozen.json', {'schema': 'source_binding_nameloc_frozen.v1', 'run_id': RUN, 'carriers': carriers, 'non_carriers': non,
                             'prereg_sha256': D.sha(OUT / 'prereg.json'), 'pilot_sha256': D.sha(OUT / 'pilot.json'),
                             'created_at': datetime.now().astimezone().isoformat()})
    show(s, meta, {'carriers': carriers, 'non_carriers': non})


def gate():
    checked()
    assert (OUT / 'frozen.json').exists() and not (OUT / 'gate.json').exists(), 'holdout behaviour already read'
    rows = D.read(OUT / 'holdout_name-materials.json')
    results, _, meta = engine(rows, edits=False, stage='gate')
    save('gate-items.json', results['base'])
    g = D.behavior(rows, results['base'])
    save('gate.json', {'schema': 'source_binding_nameloc_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre, fz = checked(), D.read(OUT / 'frozen.json')
    assert fz['prereg_sha256'] == D.sha(OUT / 'prereg.json') and fz['pilot_sha256'] == D.sha(OUT / 'pilot.json')
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows = D.read(OUT / 'holdout_name-materials.json')
    names = ('N1_carriers', 'N2_non_carriers')
    if g['passed']:
        results, valid, meta = engine(rows, stage='run')
        save('items.json', {'results': {k: [{'id': x['id'], 'pred_record': x['pred_record']} for x in v] for k, v in results.items()},
                            'valid': valid})
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, valid, rows)
        cell = lambda k, l, f: s['series'][k]['layers'][str(l)][f]['rate']
        tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
        ok = meta['tail_control_ok'] and cell('qsrc<req:src', 0, 'follow_content') >= .80
        v = {'controls_valid': bool(ok)}
        if ok:
            c, n = fz['carriers'], fz['non_carriers']
            v['N1_carriers'] = 'not_applicable' if not c else tri(
                all(cell(k, l, 'follow_position') >= .40 and cell(k, l, 'follow_position') > cell(k, l, 'follow_content') for k, l in c.items()),
                any(cell(k, l, 'follow_position') < .20 for k, l in c.items()))
            mx = {k: max(cell(k, l, 'follow_position') for l in LAYERS) for k in n}
            v['N2_non_carriers'] = 'not_applicable' if not n else tri(all(x <= .35 for x in mx.values()), any(x >= .50 for x in mx.values()))
        else:
            v.update({k: 'controls_invalid' for k in names})
        show(s, meta, {'verdicts': v})
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in names}
    report = {'schema': 'source_binding_nameloc_report.v1', 'run_id': RUN, 'split': 'holdout_name', 'gate_passed': g['passed'],
              'carriers': fz['carriers'], 'non_carriers': fz['non_carriers'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'frozen_sha256': D.sha(OUT / 'frozen.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'verdicts': v, 'summary': s,
              **meta, 'formal_evidence_promotion': 0, 'custody': 'holdout_name read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-nameloc-holdout', {'verdicts': v, 'gate_passed': g['passed'], 'carriers': fz['carriers'],
                                                                      'non_carriers': fz['non_carriers'],
                                                                      'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'carriers': fz['carriers'], 'non_carriers': fz['non_carriers'], 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['prereg', 'pilot', 'gate', 'run'])
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    pilot(a.smoke) if a.stage == 'pilot' else {'prereg': prereg, 'gate': gate, 'run': run}[a.stage]()
