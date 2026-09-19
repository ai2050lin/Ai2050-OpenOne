"""Phase 2768: category knowledge atlas, corrected design — knowledge vs
frame/lexical response separated; category state geometry.

Motivation (2767 follow-up): with the category predicate at the final token,
last-position divergence was dominated by the lexical identity response of the
swapped predicate (shallow peak nd~0.34 at L9-10; P1-P4 failed).  This phase
moves the predicate away from the sentence end with a fixed tail and adds a
pseudo-word control that has frame+lexical response but no knowledge.

Design (frozen before any forward):
  Frame: "{Art} {subj} is {art} {cat}, as everyone knows."  Fixed tail ->
  token length identical within a pair; embedding-layer dh at the last
  position is exactly 0.
  Real pairs: per category c, 16 entities, true c vs preregistered wrong
  c' = DERANGE[c] (fruit->animal, plant->liquid, animal->solid, solid->fruit,
  liquid->plant).  5 x 16 = 80 pairs.
  Pseudo control: 16 single-token pseudo subjects (first 16 single-token
  words from a fixed candidate list), same 5 category pairs -> 80 pairs.
  Attributes: colour (16 items, cyclic next in pool) and size (16 items,
  huge<->tiny), each with real subjects and the same pseudo-subject control.
  32 + 32 real pairs, 128 pseudo pairs.
Measurements at last prompt position, all 37 hidden states:
  1. nd curves per panel (real/pseudo x category/attribute).
  2. Candidate coordinates (top-32 |dh|, stable >= 60% of 16 pairs) at all
     layers; primary layer L28.  Knowledge-specific signature = real
     candidates whose overlap with pseudo candidates is within the random
     null (Jaccard <= null q95) and real count > pseudo count.
  3. Category state geometry: mean hidden state per category over its 16
     true statements at L28 -> 5x5 cosine matrix (biological cluster
     {fruit,plant,animal} vs physical {solid,liquid}).
Preregistered predictions:
  Q1 (deep amplification, corrected design): pooled real category nd,
      median(L24..35)/max(L0..19) >= 10.
  Q2 knowledge-specific candidates: >= 4/5 categories have >= 3 real
      candidates at L28 whose Jaccard with the pseudo set <= null q95.
  Q3 knowledge amplification: nd_real/nd_pseudo at L28 > 1 for >= 4/5
      categories.
  Q4 cluster geometry: mean of intra-biological cosines > mean of
      bio-vs-physical cosines at L28; solid-liquid cosine > 0.
  Q5 attribute separation: Jaccard(att_color_real, att_size_real) at L28
      <= null q95.
Status: descriptive + preregistered predictions; NOT mechanism closure.
Outputs: execution.json (frozen pre-forward), result.json, panel_stats.npz.
"""
import json
import time
from pathlib import Path

import numpy as np

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2768' / 'qwen4_category_atlas_v2'
TOPK = 32
STABILITY = 0.6
PRIMARY_LAYER = 28
TAIL = ', as everyone knows.'

CATEGORIES = ['fruit', 'plant', 'animal', 'solid', 'liquid']
ENTITIES = {
    'fruit': ['apple', 'banana', 'cherry', 'grape', 'lemon', 'mango', 'peach',
              'pear', 'plum', 'melon', 'orange', 'kiwi', 'fig', 'papaya',
              'apricot', 'coconut'],
    'plant': ['oak', 'pine', 'rose', 'fern', 'cactus', 'bamboo', 'tulip',
              'maple', 'cedar', 'orchid', 'ivy', 'moss', 'birch', 'willow',
              'daisy', 'lotus'],
    'animal': ['dog', 'cat', 'horse', 'eagle', 'salmon', 'tiger', 'rabbit',
               'dolphin', 'sparrow', 'wolf', 'bear', 'frog', 'snake', 'owl',
               'deer', 'whale'],
    'solid': ['iron', 'granite', 'marble', 'copper', 'silver', 'ice', 'glass',
              'wood', 'stone', 'brick', 'salt', 'diamond', 'aluminum',
              'quartz', 'chalk', 'slate'],
    'liquid': ['water', 'milk', 'oil', 'honey', 'wine', 'juice', 'ink',
               'rain', 'mercury', 'gasoline', 'vinegar', 'syrup', 'blood',
               'tea', 'beer', 'soup'],
}
DERANGE = {'fruit': 'animal', 'plant': 'liquid', 'animal': 'solid',
           'solid': 'fruit', 'liquid': 'plant'}
PSEUDO_CANDIDATES = ['wug', 'blicket', 'dax', 'zorple', 'fep', 'kiki',
                     'toma', 'nirp', 'splet', 'cruv', 'plap', 'snid',
                     'glore', 'vint', 'quap', 'melf', 'tarn', 'blork',
                     'yib', 'skem', 'pwick', 'lome', 'frast', 'nupe']
COLOR_ITEMS = [
    ['crow', 'black'], ['coal', 'black'], ['snow', 'white'], ['milk', 'white'],
    ['chalk', 'white'], ['blood', 'red'], ['cherry', 'red'],
    ['strawberry', 'red'], ['grass', 'green'], ['leaf', 'green'],
    ['moss', 'green'], ['banana', 'yellow'], ['lemon', 'yellow'],
    ['cheese', 'yellow'], ['sky', 'blue'], ['ocean', 'blue'],
]
COLOR_POOL = ['black', 'white', 'red', 'green', 'yellow', 'blue']
SIZE_ITEMS = [
    ['elephant', 'huge'], ['whale', 'huge'], ['mountain', 'huge'],
    ['skyscraper', 'huge'], ['glacier', 'huge'], ['cathedral', 'huge'],
    ['boulder', 'huge'], ['canyon', 'huge'], ['mouse', 'tiny'], ['ant', 'tiny'],
    ['pebble', 'tiny'], ['grain', 'tiny'], ['coin', 'tiny'], ['bee', 'tiny'],
    ['snowflake', 'tiny'], ['kitten', 'tiny'],
]
SIZE_POOL = ['huge', 'tiny']


def art(word, capital=True):
    a = 'an' if word[0].lower() in 'aeiou' else 'a'
    return a.capitalize() if capital else a


def cat_stmt(subj, cat):
    return '%s %s is %s %s%s' % (art(subj), subj, art(cat, capital=False),
                                 cat, TAIL)


def att_stmt(subj, val):
    return 'The %s is %s%s' % (subj, val, TAIL)


def build_material(tok):
    pool_ids = {}
    pool_words = CATEGORIES + COLOR_POOL + SIZE_POOL
    for w in pool_words:
        ids = tok(' ' + w, add_special_tokens=False)['input_ids']
        assert len(ids) == 1, ('predicate must be single token', w, ids)
        pool_ids[w] = ids[0]
    pseudo = list(PSEUDO_CANDIDATES)[:16]
    for w in pseudo:
        assert tok(' ' + w, add_special_tokens=False)['input_ids'], w

    prompts = []
    def add(text, kind, meta):
        ids = tok(text, add_special_tokens=False)['input_ids']
        prompts.append({'text': text, 'ids': ids, 'kind': kind, 'meta': meta})
    real_key = {}
    for c in CATEGORIES:
        for i, e in enumerate(ENTITIES[c]):
            real_key[(c, i)] = len(prompts)
            add(cat_stmt(e, c), 'real_true', {'cat': c, 'entity': e})
    real_pairs = []
    for c in CATEGORIES:
        cw = DERANGE[c]
        for i in range(16):
            real_pairs.append((real_key[(c, i)], real_key[(cw, i)], c, cw))
    pseudo_key = {}
    for p in pseudo:
        for c in CATEGORIES:
            pseudo_key[(p, c)] = len(prompts)
            add(cat_stmt(p, c), 'pseudo_true', {'cat': c, 'entity': p})
    pseudo_pairs = []
    for c in CATEGORIES:
        cw = DERANGE[c]
        for p in pseudo:
            pseudo_pairs.append((pseudo_key[(p, c)], pseudo_key[(p, cw)], c, cw))
    att_defs = []
    for e, col in COLOR_ITEMS:
        nxt = COLOR_POOL[(COLOR_POOL.index(col) + 1) % len(COLOR_POOL)]
        att_defs.append(('color', e, col, nxt))
    for e, size in SIZE_ITEMS:
        nxt = SIZE_POOL[(SIZE_POOL.index(size) + 1) % len(SIZE_POOL)]
        att_defs.append(('size', e, size, nxt))
    att_real_pairs, att_pseudo_pairs = [], []
    for kind, e, val, nxt in att_defs:
        att_real_pairs.append((len(prompts), 0, kind, val))
        add(att_stmt(e, val), 'att_real_true',
            {'kind': kind, 'entity': e, 'value': val})
        att_real_pairs[-1] = (att_real_pairs[-1][0], len(prompts), kind, val)
        add(att_stmt(e, nxt), 'att_real_wrong',
            {'kind': kind, 'entity': e, 'value': nxt})
    for kind, e, val, nxt in att_defs:
        att_pseudo_pairs.append((len(prompts), 0, kind, val))
        add(att_stmt(pseudo[0], val), 'att_pseudo_true',
            {'kind': kind, 'entity': pseudo[0], 'value': val})
        att_pseudo_pairs[-1] = (att_pseudo_pairs[-1][0], len(prompts), kind, val)
        add(att_stmt(pseudo[0], nxt), 'att_pseudo_wrong',
            {'kind': kind, 'entity': pseudo[0], 'value': nxt})
    att_real_pairs = [(a, b, k, v) for a, b, k, v in att_real_pairs]
    return {'prompts': prompts, 'real_pairs': real_pairs,
            'pseudo_pairs': pseudo_pairs, 'att_real_pairs': att_real_pairs,
            'att_pseudo_pairs': att_pseudo_pairs, 'pool_ids': pool_ids,
            'pseudo': pseudo, 'att_defs': att_defs}


def nd_curve(dh, hsq_a, hsq_b):
    L = dh.shape[1]
    nd = np.empty(L)
    for l in range(L):
        num = (dh[:, l, :].astype(np.float64) ** 2).sum(axis=1)
        den = 0.5 * (hsq_a[:, l] + hsq_b[:, l])
        nd[l] = float(num.mean() / max(den.mean(), 1e-30))
    return nd


def cand_coords(dh_l, n_pairs):
    top = np.argsort(-np.abs(dh_l), axis=1)[:, :TOPK]
    count = np.zeros(dh_l.shape[1], dtype=np.int64)
    for k in range(n_pairs):
        count[top[k]] += 1
    return np.where(count >= STABILITY * n_pairs)[0]


def jaccard(a, b):
    s1, s2 = set(a.tolist()), set(b.tolist())
    if not s1 and not s2:
        return 0.0
    return len(s1 & s2) / max(len(s1 | s2), 1)


def main():
    t0 = time.time()
    cc.guard(0)
    assert not (OUT / 'result.json').exists(), 'result.json immutable; delete before rerun'
    OUT.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    from rdc_query_common import MODELS
    tok = AutoTokenizer.from_pretrained(ROOT / 'models/hf' / MODELS['qwen4'],
                                        local_files_only=True,
                                        trust_remote_code=True, use_fast=True)
    material = build_material(tok)
    prompts = material['prompts']
    n_prompts = len(prompts)

    execution = {
        'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
        'phase': 2768,
        'prereg': {
            'question': 'After removing the final-token lexical floor (fixed '
                        'tail) and adding a pseudo-word control, where does '
                        'category knowledge live and are its coordinates '
                        'knowledge-specific rather than frame artifacts?',
            'design': {'frame_tail': TAIL, 'derange': DERANGE,
                       'pseudo_words': material['pseudo'],
                       'real_pairs': len(material['real_pairs']),
                       'pseudo_pairs': len(material['pseudo_pairs']),
                       'att_real_pairs': len(material['att_real_pairs']),
                       'att_pseudo_pairs': len(material['att_pseudo_pairs'])},
            'predictions': {
                'Q1': 'pooled real category nd median(L24..35)/max(L0..19) >= 10',
                'Q2': '>= 4/5 categories have >= 3 real candidates at L28 with '
                      'Jaccard(real, pseudo) <= null q95',
                'Q3': 'nd_real/nd_pseudo at L28 > 1 for >= 4/5 categories',
                'Q4': 'mean intra-biological cosine > mean bio-physical cosine '
                      'of category mean states at L28; solid-liquid cos > 0',
                'Q5': 'Jaccard(att_color_real, att_size_real) at L28 <= null q95'},
            'status': 'descriptive + preregistered predictions; not mechanism '
                      'closure',
            'frozen_before_any_forward': True},
        'material': {'categories': CATEGORIES, 'entities': ENTITIES,
                     'pseudo': material['pseudo'], 'tail': TAIL,
                     'n_prompts': n_prompts,
                     'prompts': [{'text': p['text'], 'kind': p['kind'],
                                  'meta': p['meta']} for p in prompts],
                     'real_pairs': material['real_pairs'],
                     'pseudo_pairs': material['pseudo_pairs'],
                     'att_real_pairs': material['att_real_pairs'],
                     'att_pseudo_pairs': material['att_pseudo_pairs']},
    }
    fc.save(OUT / 'execution.json', execution)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok2 = load_native('qwen4')
    model.eval()
    n_layers = model.config.num_hidden_layers
    assert n_layers == 36
    d_model = model.config.hidden_size
    device = next(model.parameters()).device
    L = n_layers + 1

    h_all = np.empty((n_prompts, L, d_model), dtype=np.float32)
    hsq_all = np.empty((n_prompts, L), dtype=np.float64)
    pool_names = CATEGORIES + COLOR_POOL + SIZE_POOL
    pool_vec = np.array([material['pool_ids'][w] for w in pool_names])
    slot = {w: k for k, w in enumerate(pool_names)}
    probe = np.empty((n_prompts, len(pool_names)), dtype=np.float64)
    with torch.inference_mode():
        for pi, p in enumerate(prompts):
            ids = torch.tensor([p['ids']], device=device)
            out = model(ids, output_hidden_states=True)
            h_all[pi] = np.stack([h[0, -1].float().cpu().numpy()
                                  for h in out.hidden_states])
            hsq_all[pi] = (h_all[pi].astype(np.float64) ** 2).sum(axis=1)
            pid = None
            if p['kind'] in ('real_true', 'pseudo_true'):
                pid = material['pool_ids'][p['meta']['cat']]
            elif p['kind'] in ('att_real_true', 'att_pseudo_true'):
                pid = material['pool_ids'][p['meta']['value']]
            if pid is None:
                probe[pi] = 0.0
            else:
                pos = list(p['ids']).index(pid)   # predicts token at pos
                probe[pi] = out.logits[0, pos - 1].float().cpu().numpy()[pool_vec]
            if pi % 100 == 0:
                print('P2768 PROMPT %d/%d' % (pi, n_prompts), flush=True)

    def dh_stack(pairs):
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        return h_all[b] - h_all[a], hsq_all[a], hsq_all[b]

    results = {'phase': 2768, 'n_prompts': n_prompts,
               'primary_layer': PRIMARY_LAYER, 'pseudo_words': material['pseudo']}
    nd_curves = {}
    dh_store = {}
    for name, pairs in (('real', material['real_pairs']),
                        ('pseudo', material['pseudo_pairs'])):
        dh, ha, hb = dh_stack(pairs)
        dh_store[name] = (dh, np.array([p[2] for p in pairs]))
        nd_curves[name + '_pooled'] = nd_curve(dh, ha, hb)
        for c in CATEGORIES:
            sel = np.array([p[2] for p in pairs]) == c
            nd_curves['%s_%s' % (name, c)] = nd_curve(dh[sel], ha[sel], hb[sel])
    for name, pairs in (('att_real', material['att_real_pairs']),
                        ('att_pseudo', material['att_pseudo_pairs'])):
        dh, ha, hb = dh_stack([(p[0], p[1]) for p in pairs])
        kinds = np.array([p[2] for p in pairs])
        dh_store[name] = (dh, kinds)
        for kind in ('color', 'size'):
            sel = kinds == kind
            nd_curves['%s_%s' % (name, kind)] = nd_curve(dh[sel], ha[sel],
                                                         hb[sel])
    results['nd'] = {k: v.tolist() for k, v in nd_curves.items()}

    # Q1
    rp = nd_curves['real_pooled']
    ratio = float(np.median(rp[24:36]) / max(rp[0:20].max(), 1e-30))
    results['Q1_ratio'] = ratio
    results['Q1'] = bool(ratio >= 10.0)

    # candidates at L28 (and per-layer counts)
    groups = {}
    for c in CATEGORIES:
        dh, cats = dh_store['real']
        groups['real_' + c] = dh[cats == c]
        dh, cats = dh_store['pseudo']
        groups['pseudo_' + c] = dh[cats == c]
    for name in ('att_real', 'att_pseudo'):
        dh, kinds = dh_store[name]
        for kind in ('color', 'size'):
            groups['%s_%s' % (name, kind)] = dh[kinds == kind]
    gnames = sorted(groups)
    cand28 = {}
    n_cand = np.empty((len(gnames), L), dtype=np.int64)
    for gi, g in enumerate(gnames):
        dhg = groups[g]
        n = dhg.shape[0]
        for l in range(L):
            cset = cand_coords(dhg[:, l, :].astype(np.float64), n)
            n_cand[gi, l] = len(cset)
            if l == PRIMARY_LAYER:
                cand28[g] = cset
    results['candidates_at_L28'] = {g: cand28[g].tolist() for g in gnames}
    results['n_candidates_per_layer'] = {g: n_cand[gi].tolist()
                                         for gi, g in enumerate(gnames)}

    rng = np.random.default_rng(2768001)
    d = d_model
    null = np.array([jaccard(rng.choice(d, TOPK, replace=False),
                             rng.choice(d, TOPK, replace=False))
                     for _ in range(20000)])
    q95 = float(np.quantile(null, 0.95))
    results['null'] = {'mean': float(null.mean()), 'q95': q95}

    # Q2 knowledge-specific candidates
    q2 = {}
    for c in CATEGORIES:
        real_set, pseudo_set = cand28['real_' + c], cand28['pseudo_' + c]
        jac = jaccard(real_set, pseudo_set)
        q2['cat_' + c] = {
            'n_real': len(real_set), 'n_pseudo': len(pseudo_set),
            'jaccard': jac,
            'knowledge_specific': bool(len(real_set) >= 3 and jac <= q95
                                       and len(real_set) > len(pseudo_set))}
    results['Q2'] = {'per_category': q2,
                     'supported': bool(sum(v['knowledge_specific']
                                           for v in q2.values()) >= 4)}

    # Q3 knowledge amplification at L28
    amp = {}
    for c in CATEGORIES:
        r = float(nd_curves['real_' + c][PRIMARY_LAYER])
        p = float(nd_curves['pseudo_' + c][PRIMARY_LAYER])
        amp['cat_' + c] = {'real': r, 'pseudo': p, 'ratio': r / max(p, 1e-30)}
    results['Q3'] = {'per_category': amp,
                     'supported': bool(sum(v['ratio'] > 1
                                           for v in amp.values()) >= 4)}

    # Q4 category state geometry at L28 (mean state per category, true stmts)
    real_key_meta = [(p['meta']['cat'], pi) for pi, p in enumerate(prompts)
                     if p['kind'] == 'real_true']
    means = {}
    for c in CATEGORIES:
        idxs = [pi for cc_, pi in real_key_meta if cc_ == c]
        m = h_all[idxs, PRIMARY_LAYER, :].astype(np.float64).mean(axis=0)
        means[c] = m / np.linalg.norm(m)
    C = CATEGORIES
    cos = np.array([[float(means[a] @ means[b]) for b in C] for a in C])
    bio = ['fruit', 'plant', 'animal']
    phys = ['solid', 'liquid']
    intra_bio = [float(cos[C.index(a), C.index(b)]) for a in bio for b in bio
                 if a < b]
    cross = [float(cos[C.index(a), C.index(b)]) for a in bio for b in phys]
    results['category_state_cosine_L28'] = {'categories': C,
                                            'matrix': cos.tolist(),
                                            'intra_bio': intra_bio,
                                            'cross': cross}
    results['Q4'] = {'intra_bio_mean': float(np.mean(intra_bio)),
                     'cross_mean': float(np.mean(cross)),
                     'cos_solid_liquid': float(cos[C.index('solid'),
                                                   C.index('liquid')]),
                     'supported': bool(np.mean(intra_bio) > np.mean(cross)
                                       and cos[C.index('solid'),
                                               C.index('liquid')] > 0)}

    # Q5 attribute separation
    jac_att = jaccard(cand28['att_real_color'], cand28['att_real_size'])
    results['Q5'] = {'jaccard_color_size': jac_att,
                     'supported': bool(jac_att <= q95)}

    # knowledge probe (same as 2767, panel pools)
    def probe_stats(idxs, true_names, pool):
        margins = []
        for pi, tn in zip(idxs, true_names):
            vals = {w: probe[pi, slot[w]] for w in pool}
            margins.append(vals[tn] - max(v for w, v in vals.items()
                                          if w != tn))
        margins = np.array(margins)
        return {'n': len(margins), 'rank1_rate': float((margins > 0).mean()),
                'margin_median': float(np.median(margins))}
    kp = {}
    for c in CATEGORIES:
        idxs = [pi for pi, p in enumerate(prompts) if p['kind'] == 'real_true'
                and p['meta']['cat'] == c]
        kp['cat_' + c] = probe_stats(idxs, [c] * len(idxs), CATEGORIES)
    for kind, pool in (('color', COLOR_POOL), ('size', SIZE_POOL)):
        idxs = [pi for pi, p in enumerate(prompts) if p['kind'] == 'att_real_true'
                and p['meta']['kind'] == kind]
        tns = [p['meta']['value'] for pi, p in enumerate(prompts)
               if p['kind'] == 'att_real_true' and p['meta']['kind'] == kind]
        kp['att_' + kind] = probe_stats(idxs, tns, pool)
    results['knowledge_probe'] = kp

    fc.npz(OUT / 'panel_stats.npz', h=h_all, hsq=hsq_all, probe=probe,
           n_cand=n_cand,
           group_names=np.array(gnames, dtype=np.str_),
           **{'nd_%s' % k: v for k, v in nd_curves.items()},
           **{'cand28_%s' % g: cand28[g] for g in gnames})
    results['Q_all'] = {'Q1': results['Q1'], 'Q2': results['Q2']['supported'],
                        'Q3': results['Q3']['supported'],
                        'Q4': results['Q4']['supported'],
                        'Q5': results['Q5']['supported']}
    results['seconds'] = time.time() - t0
    fc.save(OUT / 'result.json', results)
    print('PHASE2768_DONE seconds=%.1f' % results['seconds'], flush=True)


if __name__ == '__main__':
    import torch
    try:
        main()
    except Exception:
        import traceback
        (OUT / 'crash.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
