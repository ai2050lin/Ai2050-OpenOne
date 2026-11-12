"""How well does the source-binding graph predict the model on held-out edits?  Qwen3-14B BF16.

Claude Phase 408 summarised Phases 396-407 as feature cards and edges.  Here the part of that graph that concerns the
two name tokens of the question is turned into a predictor with no free parameter and scored on a new panel:

  archive-name token   carries the archive name up to block 14, the index of the archive block (1st / 2nd) at blocks
                       16-22, and nothing the answer still depends on from block 24 on
  entity-name tokens   carry the entity name up to block 20, at block 22 the line of the entity in each archive block
                       (a map block -> line), and nothing the answer depends on from block 24 on
  read-out             the model answers with the record at (block index, map[block index])

Edit: at the input of block l the residual state of a token group is replaced by the state of the same tokens in
another prompt of the same family (natural configuration, everything later recomputed).  The donor differs from the
base prompt in a random non-empty subset of four binary factors (archive asked, entity asked, archive order, entity
order): 15 donor relations, of which Phases 405-406 ran five.  For every (item, site, layer) four relations are
drawn without replacement.  The predictor names one of the four records; the outcome is whether the model's forced
choice is that record.  Baselines use the same code with one regime at every layer: no effect, name only, position
only.  The predictor rules are in this file and are committed before any edit of the new panel is run; there is no
pilot panel and nothing is fitted.
Stages: prereg -> smoke (exposed rows, engine control only) -> gate -> run.
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
import source_binding_pointer as PT
import source_binding_nameloc as NL
import source_binding_writers as WR
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_fidelity_v1'
RUN, SEED = 'SOURCE-BINDING-FIDELITY-20261111-v1', 202611110409
SPLIT, FAMILIES = 'holdout_fid', 48
CHUNK, BATCH, NB, NH, R_FROM_END, TOL, REPS = 256, 128, 40, 40, UP.R_FROM_END, 1.0, 4
SITES = {'qsrc': ['qsrc'], 'qent': ['qent'], 'names': ['qsrc', 'qent'], 'R': ['R'], 'qrest': ['qrest']}
LAYERS = {'qsrc': [8, 14, 18, 22, 26], 'qent': [8, 14, 18, 22, 26], 'names': [8, 14, 18, 22, 26], 'R': [22, 28, 32], 'qrest': [22]}
RELATIONS = [(a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)][1:]   # (archive, entity, order, x) flipped
SEEN = {('qsrc', (0, 0, 1, 0)), ('qent', (0, 0, 0, 1)), ('names', (0, 0, 1, 1)), ('qsrc', (1, 0, 0, 0)), ('qent', (0, 1, 0, 0)),
        ('R', (0, 0, 1, 1)), ('qrest', (0, 0, 1, 1))}
MODES = ['graph', 'none', 'identity', 'position', 'graph_scalar']
BASELINES = ['none', 'identity', 'position']
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'layout-factorial families as in Phase 405 (2 archives x 2 entities x 2 archive orders x 2 entity orders = 16 prompts). '
                'holdout_fid: 48 families (768 prompts). Entity names occur in no earlier panel of this line; the 16 discovery code words, '
                '4 seen frames. There is no pilot panel.',
    'edit': 'at the input of block l the residual state of the tokens of a site is replaced by the state of the same tokens in a donor '
            'prompt of the same family. Natural configuration: no other edit, all later positions recomputed.',
    'sites_and_layers': {'qsrc': 'archive-name token of the question, blocks 8, 14, 18, 22, 26', 'qent': 'entity-name tokens, same blocks',
                         'names': 'both, same blocks', 'R': 'relay position, blocks 22, 28, 32', 'qrest': 'other question tokens, block 22'},
    'donors': 'the donor differs from the base prompt in a non-empty subset of (archive asked, entity asked, archive order, entity order): '
              '15 relations. For every (item, site, layer) 4 relations are drawn without replacement, seeded, among those for which base '
              'and donor have the same number of tokens in the site.',
    'predictor': {
        'graph': 'archive-name token: name if l <= 14, block index if 16 <= l <= 22, no effect if l >= 24. entity-name tokens: name if '
                 'l <= 20, map block -> line if l = 22, no effect if l >= 24. A name is resolved in the base prompt; a block index or a '
                 'map is taken from the donor prompt as it is. Answer = record at (block index, map[block index]) of the base prompt. '
                 'R: no effect up to block 26; at block 28 the ordinal position of the donor target; at block 32 the record holding the '
                 'donor answer word. qrest: no effect.',
        'none': 'the edit never changes the choice', 'identity': 'the name regime at every layer (R, qrest: the record the donor asks for)',
        'position': 'the position regime at every layer (R, qrest: the ordinal position of the donor target)',
        'graph_scalar': 'as graph, but at block 22 the entity-name tokens carry one line (that of the donor target) instead of a map; '
                        'reported as an alternative reading of the graph, not used in a verdict',
        'why_map': 'Phase 406: with the archive-name token swapped at block 22 the model chose the other archive and the SAME entity '
                   '(0.98). The entity sits on opposite lines in the two blocks, so the line had to follow the block: a map, not a '
                   'single line. The map reading was fixed from that opened result before this panel was built.'},
    'tiers': {'seen': 'the (site, relation) pairs of Phase 406 at its layers: replication',
              'scope': 'name sites with the other relations: predictions obtained by composing the cards; the primary tier',
              'frontier': 'R and qrest with relations or layers never run: the graph has gaps here (redundant routes); descriptive'},
    'outcome': 'agreement = the forced choice of the edited run is the predicted record (four records, chance 0.25). All drawn edits; '
               'family-bootstrap 95 % intervals. Also the mean four-way softmax probability the model puts on the predicted record.',
    'tail_recomputation': 'arms recompute the positions from the first swapped token on. Control: the no-edit arm recomputed from block 0 '
                          'gives the same forced choice as the base forward on every item, maximal score difference <= 1.0',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed; otherwise behavior_ineligible.',
    'predictions_on_holdout': {
        'controls_valid': 'tail recomputation control holds',
        'F1_fidelity_in_scope': {'supported': 'scope tier: agreement of graph >= 0.80', 'not_supported': '< 0.65'},
        'F2_beats_baselines': {'supported': 'scope tier: agreement of graph minus that of the best of none / identity / position >= 0.10 '
                                            'and the family-bootstrap lower bound of that difference > 0', 'not_supported': 'difference <= 0'},
        'F3_replication': {'supported': 'seen tier: agreement of graph >= 0.85', 'not_supported': '< 0.70'}},
    'descriptive': 'frontier tier; every (site, layer) and (site, layer, relation) cell; edits on which graph and a baseline disagree; '
                   'graph_scalar; mean probability on the predicted record',
    'not_covered': 'block 24 (transition, no clear prediction); unedited prompts (the model is at ceiling, every predictor agrees); '
                   'predicted activations of features; next-token distribution beyond the four candidate words; heads and MLPs; other '
                   'tasks, answer words, frames and models. Agreement measures this graph on this family of edits, not the share of '
                   'the model that is understood.',
    'disclosure': 'the rules restate results of panels that are already open (Phases 404-407); those panels are not reused. The engine '
                  'was smoke-tested on 2 exposed families of pilot_name (Phase 406) before pre-registration, printing only the '
                  'tail-recomputation control and counts, no agreement.',
}


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定预测保真度产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def log(*a, **kw):
    RL.OUT = OUT
    RL.log(*a, **kw)


def build_materials():
    taken = {n for rows in D.read(SRC / f'{MODEL}-materials.json')['splits'].values() for r in rows for n in r['names']}
    for p in PT.PANELS + [PT.OUT / 'pilot_layout-materials.json', PT.OUT / 'holdout_layout-materials.json',
                          NL.OUT / 'pilot_name-materials.json', NL.OUT / 'holdout_name-materials.json',
                          WR.OUT / 'pilot_head-materials.json', WR.OUT / 'holdout_head-materials.json']:
        taken |= {n for r in D.read(p) for n in r['names']}
    PT.SPLITS[SPLIT] = FAMILIES
    return PT.build_materials(SPLIT, taken)


# ---------------------------------------------------------------- the predictor
def geom(r):
    """Block index of each archive, archive at each block, line of (archive, entity)."""
    order = sorted(range(4), key=lambda k: r['records'][str(k)]['span'][0])
    rank = {k: i for i, k in enumerate(order)}
    block = {s: rank[2 * s] // 2 for s in (0, 1)}
    assert all(rank[2 * s + 1] // 2 == block[s] for s in (0, 1)) and block[0] != block[1]
    return {'order': order, 'block': block, 'src_at': {b: s for s, b in block.items()},
            'line': {(s, e): rank[2 * s + e] % 2 for s in (0, 1) for e in (0, 1)}}


def regime(mode, token, l):
    if mode == 'none':
        return 'none'
    if mode in ('identity', 'position'):
        return {'identity': 'id', 'position': 'pos'}[mode]
    if l >= 24:
        return 'none'
    if token == 'qsrc':
        return 'id' if l <= 14 else 'pos'
    return 'id' if l <= 20 else 'pos'


def predict(r, d, site, l, mode):
    """The record (0-3) the model is predicted to choose when `site` of base r takes the state of donor d at block l."""
    g, gd = geom(r), geom(d)
    s, e, sd, ed = r['source'], r['entity'], d['source'], d['entity']
    ordinal_of_donor_target = gd['block'][sd] * 2 + gd['line'][(sd, ed)]
    if site in ('R', 'qrest'):
        if mode == 'none':
            return r['target_record']
        if mode == 'identity':
            return d['target_record']
        if mode == 'position':
            return g['order'][ordinal_of_donor_target]
        if site == 'qrest' or l <= 26:
            return r['target_record']
        return g['order'][ordinal_of_donor_target] if l < 32 else d['target_record']
    B = g['block'][s]
    M = lambda b: g['line'][(g['src_at'][b], e)]
    if 'qsrc' in SITES[site]:
        k = regime(mode, 'qsrc', l)
        B = g['block'][sd] if k == 'id' else gd['block'][sd] if k == 'pos' else B
    if 'qent' in SITES[site]:
        k = regime(mode, 'qent', l)
        if k == 'id':
            M = lambda b: g['line'][(g['src_at'][b], ed)]
        elif k == 'pos' and mode == 'graph_scalar':
            M = lambda b: gd['line'][(sd, ed)]
        elif k == 'pos':
            M = lambda b: gd['line'][(gd['src_at'][b], ed)]
    return g['order'][B * 2 + M(B)]


def tier(site, l, rel):
    if (site, tuple(rel)) in SEEN and l in NL.LAYERS:
        return 'seen'
    return 'scope' if site in ('qsrc', 'qent', 'names') else 'frontier'


def tokens_of(r, groups):
    return [j for g in groups for j in ([len(r['tokens']) - R_FROM_END] if g == 'R' else r['up'][g])]


def design(rows):
    """For every arm (site, layer, replicate): per item the donor row and the relation; seeded, independent of any model output."""
    index = {r['id']: i for i, r in enumerate(rows)}
    arms = {}
    for site, groups in SITES.items():
        donors = []
        for r in rows:
            row = {}
            for rel in RELATIONS:
                j = index[f"{r['family']}-S{r['source'] ^ rel[0]}E{r['entity'] ^ rel[1]}O{r['order'] ^ rel[2]}X{r['x'] ^ rel[3]}"]
                if len(tokens_of(r, groups)) == len(tokens_of(rows[j], groups)):
                    row[rel] = j
            assert len(row) >= REPS
            donors.append(row)
        for l in LAYERS[site]:
            picks = []
            for r, row in zip(rows, donors):
                rels = sorted(row)
                picks.append([rels[i] for i in stream(f'draw|{site}|{l}', r['id']).permutation(len(rels))[:REPS]])
            for k in range(REPS):
                arms[f'{site}@{l}#{k}'] = {'site': site, 'layer': l, 'rel': [list(p[k]) for p in picks],
                                           'donor': [row[p[k]] for row, p in zip(donors, picks)]}
    return arms


def self_check(rows):
    """The predictor reproduces the published outcomes of Phase 406 on the layout donors, and the regimes are as stated."""
    index = {r['id']: i for i, r in enumerate(rows)}
    don = lambda r, rel: rows[index[f"{r['family']}-S{r['source'] ^ rel[0]}E{r['entity'] ^ rel[1]}O{r['order'] ^ rel[2]}X{r['x'] ^ rel[3]}"]]
    cls = lambda r, k: ('correct', 'other_entity', 'other_source', 'neither')[2 * (k // 2 != r['source']) + (k % 2 != r['entity'])]
    for r in rows:
        assert cls(r, predict(r, don(r, (0, 0, 1, 0)), 'qsrc', 18, 'graph')) == 'other_source'
        assert cls(r, predict(r, don(r, (0, 0, 0, 1)), 'qent', 22, 'graph')) == 'other_entity'
        assert cls(r, predict(r, don(r, (0, 0, 1, 1)), 'names', 22, 'graph')) == 'other_source'
        assert cls(r, predict(r, don(r, (1, 0, 0, 0)), 'qsrc', 8, 'graph')) == 'other_source'
        assert cls(r, predict(r, don(r, (0, 1, 0, 0)), 'qent', 8, 'graph')) == 'other_entity'
        for rel in RELATIONS:
            d = don(r, rel)
            assert all(predict(r, d, s, 26, 'graph') == r['target_record'] for s in ('qsrc', 'qent', 'names'))
            assert predict(r, d, 'names', 8, 'graph') == d['target_record']
            assert all(predict(r, r, s, l, m) == r['target_record'] for s in SITES for l in LAYERS[s] for m in MODES)


DEPS = (RL, UP, HD, QK, QY, RQ, TD, PT, NL, WR)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定预测保真度产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    rows = build_materials()
    assert len(rows) == FAMILIES * 16 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    self_check(rows)
    arms = design(rows)
    save(f'{SPLIT}-materials.json', rows)
    save('design.json', arms)
    count = {t: 0 for t in ('seen', 'scope', 'frontier')}
    differ = {b: 0 for b in BASELINES}
    for a in arms.values():
        for r, j, rel in zip(rows, a['donor'], a['rel']):
            t = tier(a['site'], a['layer'], rel)
            count[t] += 1
            if t == 'scope':
                g = predict(r, rows[j], a['site'], a['layer'], 'graph')
                for b in BASELINES:
                    differ[b] += g != predict(r, rows[j], a['site'], a['layer'], b)
    body = {'schema': 'source_binding_fidelity_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'chain_summary': 'results/2026-11-11/202044_SOURCE-BINDING-chain-summary.json',
            'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'materials': {'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': len(rows)},
            'design': {'sha256': D.sha(OUT / 'design.json'), 'arms': len(arms), 'edits_per_tier': count,
                       'scope_edits_where_graph_differs_from': differ},
            'power': {'unit': 'paired binary outcome per edit (graph right / baseline right); alpha = 0.05',
                      'F2': {'expected_delta': 0.10, 'p_d': 0.30, **check_phase(0.10, count['scope'], p_d=0.30)},
                      'F2_as_if_one_edit_per_item': {'expected_delta': 0.10, 'p_d': 0.30, **check_phase(0.10, len(rows), p_d=0.30)},
                      'caveat': 'edits are clustered in 768 items and 48 families; intervals are family bootstraps'},
            'custody': 'holdout_fid is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 15, 'disk_GiB_estimate': 0.3, 'models_concurrent': 1,
                          'shape': 'chunks of 256 prompts, inner batch 128, 76 arms with per-arm tails; fixed before pre-registration'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'design': body['design'], 'power': body['power']['F2'],
                      'power_item_level': body['power']['F2_as_if_one_edit_per_item']}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and all(p['dependency_sha256'][m.__name__] == D.sha(m.__file__) for m in DEPS), \
        'source changed after pre-registration'
    assert p['materials']['sha256'] == D.sha(OUT / f'{SPLIT}-materials.json') and p['design']['sha256'] == D.sha(OUT / 'design.json')
    D.checked(MODEL)
    return p


def engine(rows, arms=None, stage='run'):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    t, F = torch, torch.nn.functional
    assert snapshot(D.MODELS[MODEL]) == D.read(SRC / f'{MODEL}-identity.json')['model_snapshot']
    start, total = time.monotonic(), len(rows)
    results = {}
    with GpuLease(), t.inference_mode():
        rt = Runtime(MODEL)
        hd, kv, grp = rt.hd, rt.kv, rt.heads // rt.kv
        rot = lambda x, co, si: x * co + rt.mod.rotate_half(x) * si
        log(stage, 0, total * NB, start, rows=total)
        for c0 in range(0, total, CHUNK):
            rs = rows[c0:c0 + CHUNK]
            N = len(rs)
            ids, pos, pe, mask = rt.context(rs)
            h, length = rt.embed(ids), ids.shape[1]
            off = [length - len(r['tokens']) for r in rs]
            plan = {}
            if arms:
                for name, a in arms.items():
                    groups = SITES[a['site']]
                    bi, bp, di, dp = [], [], [], []
                    for i, r in enumerate(rs):
                        j = a['donor'][c0 + i] - c0
                        assert 0 <= j < N and rs[j]['family'] == r['family']
                        x, y = tokens_of(r, groups), tokens_of(rs[j], groups)
                        assert len(x) == len(y)
                        bi += [i] * len(x)
                        bp += [off[i] + v for v in x]
                        di += [j] * len(y)
                        dp += [off[j] + v for v in y]
                    P = min(off[i] + min(tokens_of(r, groups)) for i, r in enumerate(rs))
                    plan[name] = (a['layer'], P, tuple(t.tensor(v, device='cuda') for v in (bi, bp, di, dp)))
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
                    free, tot = t.cuda.mem_get_info()
                    log(stage, c0 * NB + (b + 1) * N, total * NB, start, block=b, arms=len(state), device_used_MiB=round((tot - free) / 2 ** 20))
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
    if arms:
        meta['noop_same_choice'] = all(x['pred_record'] == y['pred_record'] for x, y in zip(results['noop'], results['base']))
        meta['noop_max_score_difference'] = max(abs(p - q) for x, y in zip(results['noop'], results['base']) for p, q in zip(x['scores'], y['scores']))
        meta['tail_control_ok'] = bool(meta['noop_same_choice'] and meta['noop_max_score_difference'] <= TOL)
    return results, meta


def summarise(results, arms, rows):
    fam_id = {f: i for i, f in enumerate(sorted({r['family'] for r in rows}))}
    rec = {k: [] for k in ('fam', 'tier', 'site', 'layer', 'rel', 'changed', 'prob') + tuple(MODES)}
    for name, a in arms.items():
        for i, (r, j, rel) in enumerate(zip(rows, a['donor'], a['rel'])):
            out = results[name][i]
            sc = np.array(out['scores'], float)
            p = np.exp(sc - sc.max())
            p /= p.sum()
            rec['fam'].append(fam_id[r['family']])
            rec['tier'].append(tier(a['site'], a['layer'], rel))
            rec['site'].append(a['site'])
            rec['layer'].append(a['layer'])
            rec['rel'].append(''.join(map(str, rel)))
            rec['changed'].append(out['pred_record'] != r['target_record'])
            preds = {m: predict(r, rows[j], a['site'], a['layer'], m) for m in MODES}
            for m in MODES:
                rec[m].append(preds[m] == out['pred_record'])
            rec['prob'].append([float(p[preds[m]]) for m in MODES])
    A = {k: np.array(v) for k, v in rec.items()}
    rng = np.random.default_rng(SEED)
    nf = len(fam_id)
    draws = rng.integers(0, nf, (1000, nf))

    def rate(v, keep):
        v = np.asarray(v, float)
        per = np.bincount(A['fam'][keep], weights=v[keep], minlength=nf)
        cnt = np.bincount(A['fam'][keep], minlength=nf)
        bs = per[draws].sum(1) / np.maximum(cnt[draws].sum(1), 1)
        return {'rate': round(float(v[keep].mean()), 4), 'ci95': [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]}

    def block(keep):
        body = {'n': int(keep.sum()), 'model_choice_changed': round(float(A['changed'][keep].mean()), 4),
                'agreement': {m: rate(A[m], keep) for m in MODES},
                'mean_probability_on_predicted': {m: round(float(A['prob'][keep][:, i].mean()), 4) for i, m in enumerate(MODES)}}
        best = max(BASELINES, key=lambda m: body['agreement'][m]['rate'])
        body['best_baseline'] = best
        body['graph_minus_best_baseline'] = rate(A['graph'].astype(float) - A[best].astype(float), keep)
        body['where_graph_and_baseline_disagree'] = {
            b: {'n': int((keep & (A['graph'] != A[b])).sum()),
                'graph_right': round(float(A['graph'][keep & (A['graph'] != A[b])].mean()), 4) if (keep & (A['graph'] != A[b])).any() else None}
            for b in BASELINES}
        return body
    out = {'edits': len(A['fam']), 'base_correct': round(float(np.mean([x['forced_correct'] for x in results['base']])), 4),
           'tiers': {t: block(A['tier'] == t) for t in ('seen', 'scope', 'frontier')}, 'cells': {}, 'relations': {}}
    for site in SITES:
        for l in LAYERS[site]:
            keep = (A['site'] == site) & (A['layer'] == l)
            out['cells'][f'{site}@{l}'] = {'n': int(keep.sum()), 'changed': round(float(A['changed'][keep].mean()), 4),
                                           **{m: round(float(A[m][keep].mean()), 4) for m in MODES}}
            for rel in sorted(set(A['rel'][keep])):
                k2 = keep & (A['rel'] == rel)
                out['relations'][f'{site}@{l}<{rel}'] = {'n': int(k2.sum()), 'tier': str(A['tier'][k2][0]),
                                                         'changed': round(float(A['changed'][k2].mean()), 4),
                                                         **{m: round(float(A[m][k2].mean()), 4) for m in MODES}}
    return out


def smoke():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定预测保真度产物')
    rows = [r for r in D.read(NL.OUT / 'pilot_name-materials.json') if int(r['family'][-4:]) < 2]
    self_check(rows)
    arms = design(rows)
    results, meta = engine(rows, arms, stage='smoke')
    s = summarise(results, arms, rows)       # exercised for code coverage; agreement is not printed
    print(json.dumps({'meta': meta, 'arms': len(arms), 'edits': s['edits'], 'tier_n': {t: v['n'] for t, v in s['tiers'].items()}}), flush=True)


def gate():
    checked()
    assert not (OUT / 'gate.json').exists(), 'holdout behaviour already read'
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    results, meta = engine(rows, stage='gate')
    save('gate-items.json', results['base'])
    g = D.behavior(rows, results['base'])
    save('gate.json', {'schema': 'source_binding_fidelity_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre = checked()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows, arms = D.read(OUT / f'{SPLIT}-materials.json'), D.read(OUT / 'design.json')
    names = ('F1_fidelity_in_scope', 'F2_beats_baselines', 'F3_replication')
    if g['passed']:
        results, meta = engine(rows, arms, stage='run')
        save('items.json', {k: [{'id': x['id'], 'pred_record': x['pred_record'], 'scores': x['scores']} for x in v] for k, v in results.items()})
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, arms, rows)
        tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
        v = {'controls_valid': bool(meta['tail_control_ok'])}
        if v['controls_valid']:
            sc, sn = s['tiers']['scope'], s['tiers']['seen']
            a, d = sc['agreement']['graph']['rate'], sc['graph_minus_best_baseline']
            v['F1_fidelity_in_scope'] = tri(a >= .80, a < .65)
            v['F2_beats_baselines'] = tri(d['rate'] >= .10 and d['ci95'][0] > 0, d['rate'] <= 0)
            v['F3_replication'] = tri(sn['agreement']['graph']['rate'] >= .85, sn['agreement']['graph']['rate'] < .70)
        else:
            v.update({k: 'controls_invalid' for k in names})
        print(json.dumps({'meta': meta, 'verdicts': v, 'tiers': s['tiers'], 'cells': s['cells']}), flush=True)
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in names}
    report = {'schema': 'source_binding_fidelity_report.v1', 'run_id': RUN, 'split': SPLIT, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'verdicts': v, 'summary': s,
              **meta, 'formal_evidence_promotion': 0, 'custody': 'holdout_fid read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-fidelity-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                       'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['prereg', 'smoke', 'gate', 'run'])
    {'prereg': prereg, 'smoke': smoke, 'gate': gate, 'run': run}[ap.parse_args().stage]()
