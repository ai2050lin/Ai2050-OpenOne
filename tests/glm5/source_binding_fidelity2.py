"""Fidelity of the revised source-binding graph (revision 2, Claude Phase 412) on a new panel.  Qwen3-14B BF16.

Phase 409 scored the Phase 408 graph on held-out state exchanges: the name-token part predicted well, the part through
the relay position R failed in the natural run.  Phases 410-411 found the natural short chain (the answer position reads
the name tokens in blocks 24-25 and fetches the code word in block 28) and three redundant answer sources (direct
fetch, R, the think positions).  The revised graph is turned into a predictor with no free parameter:

  name tokens          as in Phase 409 up to block 22; NEW: still the position regime at block 24 (the answer position
                       reads them in blocks 24-25 and the direct route wins over the backups); nothing from block 26 on
  R, think block, R + think, other question tokens
                       NEW: exchanging their state does not change the answer (they are backups; the direct fetch decides)

Edits, donors, outcome and engine are those of Phase 409.  Tiers: repl (the Phase 409 cells on a new panel), new (name
sites at block 24, never run), backup (R, think, R + think, other question tokens).  The Phase 409 predictor is scored
as graph_v1.  The rules are committed before any edit of the new panel is run; there is no pilot panel.
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
OUT = D.ROOT / 'tests/glm5/result/source_binding_fidelity_v2'
RUN, SEED = 'SOURCE-BINDING-FIDELITY-20261112-v2b', 202611120412
SPLIT, FAMILIES = 'holdout_fid2', 48
CHUNK, BATCH, NB, NH, R_FROM_END, TOL, REPS = 256, 128, 40, 40, UP.R_FROM_END, 1.0, 4
NAME_SITES, BACKUP_SITES = ('qsrc', 'qent', 'names'), ('R', 'think', 'Rthink', 'qrest')
SITES = {'qsrc': ['qsrc'], 'qent': ['qent'], 'names': ['qsrc', 'qent'], 'R': ['R'], 'think': ['think'], 'Rthink': ['R', 'think'],
         'qrest': ['qrest']}
LAYERS = {'qsrc': [8, 14, 18, 22, 24, 26], 'qent': [8, 14, 18, 22, 24, 26], 'names': [8, 14, 18, 22, 24, 26], 'R': [22, 28, 32],
          'think': [28, 32], 'Rthink': [28, 32], 'qrest': [22]}
RELATIONS = [(a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)][1:]   # (archive, entity, order, x) flipped
OTHER_PANELS = ['source_binding_writers_v1/pilot_head', 'source_binding_writers_v1/holdout_head', 'source_binding_fidelity_v1/holdout_fid',
                'source_binding_readers_v1/pilot_read', 'source_binding_readers_v1/holdout_read',
                'source_binding_fetch_v1/pilot_fetch', 'source_binding_fetch_v1/holdout_fetch']
MODES = ['graph', 'none', 'identity', 'position', 'graph_v1']
BASELINES = ['none', 'identity', 'position']
TIERS = ['repl', 'new', 'backup']
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'layout-factorial families as in Phase 405. holdout_fid2: 48 families (768 prompts). Entity names occur in no earlier '
                'panel of this line; the 16 discovery code words, 4 seen frames. There is no pilot panel.',
    'edit': 'at the input of block l the residual state of the tokens of a site is replaced by the state of the same tokens in a donor '
            'prompt of the same family. Natural configuration: no other edit, all later positions recomputed.',
    'sites_and_layers': {'qsrc / qent / names': 'name tokens of the question, blocks 8, 14, 18, 22, 24, 26', 'R': 'relay position, blocks '
                         '22, 28, 32', 'think': 'the 4 tokens of the empty think block, blocks 28, 32', 'Rthink': 'both, blocks 28, 32',
                         'qrest': 'other question tokens, block 22'},
    'donors': 'as in Phase 409: the donor differs in a non-empty subset of (archive asked, entity asked, archive order, entity order); '
              'for every (item, site, layer) 4 of the 15 relations are drawn without replacement, seeded.',
    'predictor': {
        'graph': 'revision 2. archive-name token: name if l <= 14, block index if 16 <= l <= 24, no effect if l >= 26. entity-name tokens: '
                 'name if l <= 20, map block -> line if l = 22 or 24, no effect if l >= 26. Answer = record at (block index, map[block '
                 'index]) of the base prompt. R, think, Rthink, qrest: no effect at any layer.',
        'graph_v1': 'the Phase 409 predictor: name tokens have no effect from block 24 on; R: no effect up to block 26, the ordinal '
                    'position of the donor target at block 28, the record holding the donor answer word at block 32 (Rthink scored like '
                    'R; think and qrest: no effect). Reported for comparison, not used in a verdict.',
        'none': 'the edit never changes the choice', 'identity': 'the name regime at every layer (backup sites: the record the donor asks for)',
        'position': 'the position regime at every layer (backup sites: the ordinal position of the donor target)',
        'why_block_24': 'Phase 410: the answer position reads the name tokens in blocks 24-25 and this alone suffices (1.000 / 0.997); '
                        'Phase 411: when the direct route and the backups disagree the direct route wins (0.035 with only the correct '
                        'code word hidden). So a name token exchanged at the input of block 24 should still move the answer to the '
                        'donor position although every backup holds the base answer. Fixed from those opened results.',
        'why_backups': 'Phase 409: exchanging R never changed the answer (0 of 9,216); Phase 411: R and the think block are each '
                       'sufficient but none is necessary. Exchanging R and think together has never been run.'},
    'tiers': {'repl': 'name sites at blocks 8, 14, 18, 22, 26: the cells of Phase 409 on a new panel',
              'new': 'name sites at block 24: excluded in Phase 409 as a transition without prediction; the primary tier',
              'backup': 'R, think, Rthink, qrest: the revised graph predicts no effect, which coincides with the none baseline; R and '
                        'qrest replicate Phase 409, think and Rthink have never been run'},
    'outcome': 'agreement = the forced choice of the edited run is the predicted record (four records, chance 0.25). All drawn edits; '
               'family-bootstrap 95 % intervals. Also the mean four-way softmax probability the model puts on the predicted record.',
    'tail_recomputation': 'arms recompute the positions from the first swapped token on. Control: the no-edit arm recomputed from block 0 '
                          'gives the same forced choice as the base forward on every item, maximal score difference <= 1.0',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed; otherwise behavior_ineligible.',
    'predictions_on_holdout': {
        'controls_valid': 'tail recomputation control holds',
        'G1_replication': {'supported': 'repl tier: agreement of graph >= 0.95', 'not_supported': '< 0.85'},
        'G2_block_24': {'supported': 'new tier: agreement of graph >= 0.80, and graph minus the better of none / identity >= 0.10 with a '
                                     'family-bootstrap lower bound > 0', 'not_supported': 'agreement < 0.65 or difference <= 0',
                        'note': 'at a single layer the position baseline is the same rule as the graph (0 of 9,216 edits differ), so it '
                                'cannot be a competitor here; the competitors are no effect (which is also graph_v1 at this layer) and '
                                'name only'},
        'G3_backups': {'supported': 'backup tier: agreement of graph >= 0.95', 'not_supported': '< 0.85',
                       'note': 'not discriminating against the none baseline by construction; it tests that exchanging R and think, '
                               'separately and together, leaves the answer unchanged'}},
    'descriptive': 'every (site, layer) and (site, layer, relation) cell; edits on which graph and a baseline disagree; graph_v1; mean '
                   'probability on the predicted record; all tiers pooled',
    'not_covered': 'block 25; attention edits (Phases 410-411 are interventions of that kind, not predictions scored here); unedited '
                   'prompts; predicted activations of features; next-token distribution beyond the four candidate words; other tasks, '
                   'answer words, frames and models. Agreement measures this graph on this family of edits, not the share of the model '
                   'that is understood.',
    'disclosure': 'the rules restate results of panels that are already open (Phases 404-411); those panels are not reused. The rule for '
                  'R and qrest was changed after Phase 409 showed the revision-1 rule wrong, so on those sites this run is a '
                  'confirmation on a new panel, not an independent prediction. The engine was smoke-tested on 2 exposed families of '
                  'pilot_name (Phase 406) before pre-registration, printing only the tail-recomputation control and counts. '
                  'Version v2 of this pre-registration (kept as prereg-superseded.json, sha256 0f9a4169...) compared the graph at '
                  'block 24 with the best of none / identity / position; its own design count showed that position coincides with the '
                  'graph on every edit of that tier, so G2 could not be supported by construction. This was noticed after the '
                  'behaviour gate (unedited prompts only) and before any edit was run; v2b changes only the competitor set of G2. '
                  'Materials, thresholds and all other rules are unchanged; the donor draws were redrawn by the same rule because '
                  'the seed stream contains the run id (design.json of v2: sha256 bdf3a087...).',
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
                          NL.OUT / 'pilot_name-materials.json',
                          NL.OUT / 'holdout_name-materials.json'] + [D.ROOT / 'tests/glm5/result' / f'{x}-materials.json' for x in OTHER_PANELS]:
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
    if l >= (24 if mode == 'graph_v1' else 26):
        return 'none'
    if token == 'qsrc':
        return 'id' if l <= 14 else 'pos'
    return 'id' if l <= 20 else 'pos'


def predict(r, d, site, l, mode):
    """The record (0-3) the model is predicted to choose when `site` of base r takes the state of donor d at block l."""
    g, gd = geom(r), geom(d)
    s, e, sd, ed = r['source'], r['entity'], d['source'], d['entity']
    ordinal_of_donor_target = gd['block'][sd] * 2 + gd['line'][(sd, ed)]
    if site in BACKUP_SITES:
        if mode == 'identity':
            return d['target_record']
        if mode == 'position':
            return g['order'][ordinal_of_donor_target]
        if mode == 'graph_v1' and site in ('R', 'Rthink') and l > 26:
            return g['order'][ordinal_of_donor_target] if l < 32 else d['target_record']
        return r['target_record']
    B = g['block'][s]
    M = lambda b: g['line'][(g['src_at'][b], e)]
    if 'qsrc' in SITES[site]:
        k = regime(mode, 'qsrc', l)
        B = g['block'][sd] if k == 'id' else gd['block'][sd] if k == 'pos' else B
    if 'qent' in SITES[site]:
        k = regime(mode, 'qent', l)
        if k == 'id':
            M = lambda b: g['line'][(g['src_at'][b], ed)]
        elif k == 'pos':
            M = lambda b: gd['line'][(gd['src_at'][b], ed)]
    return g['order'][B * 2 + M(B)]


def tier(site, l, rel):
    return 'backup' if site in BACKUP_SITES else 'new' if l == 24 else 'repl'


def tokens_of(r, groups):
    n = len(r['tokens'])
    return [j for g in groups for j in ([n - R_FROM_END] if g == 'R' else [n - 6, n - 5, n - 4, n - 3] if g == 'think' else r['up'][g])]


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
            assert all(predict(r, d, s, 26, m) == r['target_record'] for s in NAME_SITES for m in ('graph', 'graph_v1'))
            assert all(predict(r, d, s, 24, 'graph') == predict(r, d, s, 22, 'graph') == predict(r, d, s, 22, 'graph_v1') for s in NAME_SITES)
            assert all(predict(r, d, s, 24, 'graph_v1') == r['target_record'] for s in NAME_SITES)
            assert all(predict(r, d, s, l, 'graph') == r['target_record'] for s in BACKUP_SITES for l in LAYERS[s])
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
    count = {t: 0 for t in TIERS}
    differ = {b: 0 for b in BASELINES}
    for a in arms.values():
        for r, j, rel in zip(rows, a['donor'], a['rel']):
            t = tier(a['site'], a['layer'], rel)
            count[t] += 1
            if t == 'new':
                g = predict(r, rows[j], a['site'], a['layer'], 'graph')
                for b in BASELINES:
                    differ[b] += g != predict(r, rows[j], a['site'], a['layer'], b)
    body = {'schema': 'source_binding_fidelity_prereg.v2', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'chain_summary': 'results/2026-11-12/010635_SOURCE-BINDING-chain-summary-v2.json',
            'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'materials': {'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': len(rows)},
            'design': {'sha256': D.sha(OUT / 'design.json'), 'arms': len(arms), 'edits_per_tier': count,
                       'new_tier_edits_where_graph_differs_from': differ},
            'power': {'unit': 'paired binary outcome per edit (graph right / baseline right); alpha = 0.05',
                      'G2': {'expected_delta': 0.10, 'p_d': 0.30, **check_phase(0.10, count['new'], p_d=0.30)},
                      'G2_as_if_one_edit_per_item': {'expected_delta': 0.10, 'p_d': 0.30, **check_phase(0.10, len(rows), p_d=0.30)},
                      'caveat': 'edits are clustered in 768 items and 48 families; intervals are family bootstraps'},
            'custody': 'holdout_fid2 is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 20, 'disk_GiB_estimate': 0.3, 'models_concurrent': 1,
                          'shape': 'chunks of 256 prompts, inner batch 128, 104 arms with per-arm tails; fixed before pre-registration'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'design': body['design'], 'power': body['power']['G2'],
                      'power_item_level': body['power']['G2_as_if_one_edit_per_item']}), flush=True)


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
        alt = max(('none', 'identity'), key=lambda m: body['agreement'][m]['rate'])
        body['better_of_none_identity'] = alt
        body['graph_minus_better_of_none_identity'] = rate(A['graph'].astype(float) - A[alt].astype(float), keep)
        body['where_graph_and_baseline_disagree'] = {
            b: {'n': int((keep & (A['graph'] != A[b])).sum()),
                'graph_right': round(float(A['graph'][keep & (A['graph'] != A[b])].mean()), 4) if (keep & (A['graph'] != A[b])).any() else None}
            for b in BASELINES}
        return body
    out = {'edits': len(A['fam']), 'base_correct': round(float(np.mean([x['forced_correct'] for x in results['base']])), 4),
           'tiers': {t: block(A['tier'] == t) for t in TIERS}, 'all_edits': block(A['tier'] != ''), 'cells': {}, 'relations': {}}
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
    save('gate.json', {'schema': 'source_binding_fidelity_gate.v2', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre = checked()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows, arms = D.read(OUT / f'{SPLIT}-materials.json'), D.read(OUT / 'design.json')
    names = ('G1_replication', 'G2_block_24', 'G3_backups')
    if g['passed']:
        results, meta = engine(rows, arms, stage='run')
        save('items.json', {k: [{'id': x['id'], 'pred_record': x['pred_record'], 'scores': x['scores']} for x in v] for k, v in results.items()})
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, arms, rows)
        tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
        v = {'controls_valid': bool(meta['tail_control_ok'])}
        if v['controls_valid']:
            T = s['tiers']
            ag = lambda t: T[t]['agreement']['graph']['rate']
            d = T['new']['graph_minus_better_of_none_identity']
            v['G1_replication'] = tri(ag('repl') >= .95, ag('repl') < .85)
            v['G2_block_24'] = tri(ag('new') >= .80 and d['rate'] >= .10 and d['ci95'][0] > 0, ag('new') < .65 or d['rate'] <= 0)
            v['G3_backups'] = tri(ag('backup') >= .95, ag('backup') < .85)
        else:
            v.update({k: 'controls_invalid' for k in names})
        print(json.dumps({'meta': meta, 'verdicts': v, 'tiers': s['tiers'], 'cells': s['cells']}), flush=True)
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in names}
    report = {'schema': 'source_binding_fidelity_report.v2', 'run_id': RUN, 'split': SPLIT, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'verdicts': v, 'summary': s,
              **meta, 'formal_evidence_promotion': 0, 'custody': 'holdout_fid2 read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-fidelity-v2-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                       'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['prereg', 'smoke', 'gate', 'run'])
    {'prereg': prereg, 'smoke': smoke, 'gate': gate, 'run': run}[ap.parse_args().stage]()
