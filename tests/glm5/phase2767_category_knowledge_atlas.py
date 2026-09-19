"""Phase 2767: category knowledge atlas in qwen3-4b — layer distribution,
per-category deep-band coordinate signatures, and cross-family separation.

User request (2026-09-15): crack the CATEGORY knowledge atlas (fruit / plant /
animal / solid / liquid): where does it live across depth, do different
categories use distinct coordinate groups, then extrapolate to attribute
regions (colour / size).

Design (frozen before any forward; this docstring is part of the prereg):

C001 category atlas.
  5 categories x 16 entities, statement prompts "{Art} {e} is {art} {c}."
  World A = true category; World B = wrong category c' != c (all 4 wrongs
  per entity -> 320 divergence pairs).  Entity-swap control: same category,
  entity i paired with entity (i+1) mod 16 (both true statements, 80 pairs).
C002 attribute extrapolation.
  Colour: 16 canonical-colour items "The {e} is {col}." true vs cyclic-next
  colour (pool black,white,red,green,yellow,blue).  Size: 16 items true
  huge/tiny vs swapped.  32 divergence pairs.
C003 cross-family separation.
  At preregistered primary layer L28 (deep-band platform middle): candidate
  coordinate sets (per-pair top-32 |dh|, stable in >= 60% of pairs) for
  {5 categories, pooled entity-swap, colour, size} + the 2760 relation
  families recomputed from phase2760 layer_stats.npz dh store.  Jaccard
  matrix vs analytic+empirical random null.  Category-mean divergence
  directions: cosine matrix (ontological distance prediction: fruit-plant
  closer than fruit-animal).

Preregistered predictions (evaluated honestly, descriptive phase):
  P1 deep amplification: pooled category nd, median(L24..L35) / max(L0..L19)
     >= 10.
  P2 each category has >= 5 stable candidate coordinates at L28.
  P3 orthogonality: all 10 pairwise category-set Jaccards at L28 <= 95th
     percentile of the random 32-subset null.
  P4 semantic distance: cos(fruit,plant) > cos(fruit,animal) of mean
     divergence directions at L28.
  P5 knowledge probe: last-position next-token margin (true predicate vs
     predicate pool) rank-1 rate >= 0.5 per category / colour / size.
Status: descriptive + preregistered predictions; NOT a mechanism-closure
claim (project discipline: 条件齿轮候选, not 条件齿轮闭合).
Outputs: execution.json (frozen pre-forward), result.json, panel_stats.npz.
"""
import json
import time
from pathlib import Path

import numpy as np

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2767' / 'qwen4_category_atlas'
TOPK = 32
STABILITY = 0.6
PRIMARY_LAYER = 28

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


def build_material(tok):
    """Deterministic prompts + pair definitions. Returns dict of lists."""
    cat_pool_ids = {}
    for w in CATEGORIES + COLOR_POOL + SIZE_POOL:
        ids = tok(' ' + w, add_special_tokens=False)['input_ids']
        assert len(ids) == 1, ('predicate must be single token', w, ids)
        cat_pool_ids[w] = ids[0]
    prompts = []          # list of dicts: text, kind, meta
    def add(text, kind, meta):
        ids = tok(text, add_special_tokens=False)['input_ids']
        prompts.append({'text': text, 'ids': ids, 'kind': kind, 'meta': meta})
    true_key = {}
    for c in CATEGORIES:
        for i, e in enumerate(ENTITIES[c]):
            text = '%s %s is %s %s.' % (art(e), e, art(c, capital=False), c)
            true_key[(c, i)] = len(prompts)
            add(text, 'cat_true', {'cat': c, 'entity': e, 'idx': i})
    cat_pairs = []        # (true_idx, wrong_idx, true_cat, wrong_cat, entity)
    for c in CATEGORIES:
        for i, e in enumerate(ENTITIES[c]):
            for cw in CATEGORIES:
                if cw == c:
                    continue
                widx = true_key[(cw, i)]  # same entity asserted in wrong cat
                cat_pairs.append((true_key[(c, i)], widx, c, cw, e))
    ent_pairs = []        # entity-swap control within category
    for c in CATEGORIES:
        for i in range(len(ENTITIES[c])):
            j = (i + 1) % len(ENTITIES[c])
            ent_pairs.append((true_key[(c, i)], true_key[(c, j)], c))
    att_defs = []
    for e, col in COLOR_ITEMS:
        nxt = COLOR_POOL[(COLOR_POOL.index(col) + 1) % len(COLOR_POOL)]
        att_defs.append(('color', e, col, nxt))
    for e, size in SIZE_ITEMS:
        nxt = SIZE_POOL[(SIZE_POOL.index(size) + 1) % len(SIZE_POOL)]
        att_defs.append(('size', e, size, nxt))
    att_true_idx, att_wrong_idx = [], []
    for kind, e, val, nxt in att_defs:
        t = 'The %s is %s.' % (e, val)
        w = 'The %s is %s.' % (e, nxt)
        att_true_idx.append(len(prompts))
        add(t, 'att_true', {'kind': kind, 'entity': e, 'value': val})
        att_wrong_idx.append(len(prompts))
        add(w, 'att_wrong', {'kind': kind, 'entity': e, 'value': nxt})
    return {'prompts': prompts, 'cat_pairs': cat_pairs, 'ent_pairs': ent_pairs,
            'att_defs': att_defs, 'att_true_idx': att_true_idx,
            'att_wrong_idx': att_wrong_idx, 'pool_ids': cat_pool_ids}


def nd_curve(dh, hsq_a, hsq_b):
    L = dh.shape[1]
    nd = np.empty(L)
    en = np.empty(L)
    for l in range(L):
        num = (dh[:, l, :].astype(np.float64) ** 2).sum(axis=1)
        den = 0.5 * (hsq_a[:, l] + hsq_b[:, l])
        en[l] = float(num.mean())
        nd[l] = float(num.mean() / max(den.mean(), 1e-30))
    return nd, en


def cand_coords(dh_l, n_pairs):
    top = np.argsort(-np.abs(dh_l), axis=1)[:, :TOPK]
    count = np.zeros(dh_l.shape[1], dtype=np.int64)
    for k in range(n_pairs):
        count[top[k]] += 1
    cand = np.where(count >= STABILITY * n_pairs)[0]
    return cand, count, top


def jaccard(a, b):
    s1, s2 = set(a.tolist()), set(b.tolist())
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
        'phase': 2767,
        'prereg': {
            'question': 'Where does category knowledge (fruit/plant/animal/'
                        'solid/liquid) live across the 36 layers of qwen3-4b, '
                        'do categories use distinct coordinate groups, and do '
                        'attributes (colour/size) occupy separate groups?',
            'panels': {'C001_category': '5 cat x 16 entities x (1 true + 4 '
                        'wrong category) = 320 divergence pairs + 80 '
                        'entity-swap control pairs',
                       'C002_attribute': '16 colour + 16 size true-vs-permuted '
                       'pairs',
                       'C003_separation': 'Jaccard of candidate coordinate '
                       'sets at L28 across categories/attributes/entity-swap/'
                       '2760 relation families'},
            'definitions': {
                'nd': 'mean_pairs ||hA-hB||^2 / mean_pairs '
                      '0.5*(||hA||^2+||hB||^2), last prompt position, all 37 '
                      'hidden states',
                'candidate': 'coordinate in per-pair top-32 |dh| for >= 60% '
                             'of pairs',
                'primary_layer': PRIMARY_LAYER,
                'knowledge_probe': 'logit of true predicate first token minus '
                                   'max over predicate pool at the position '
                                   'before the predicate token'},
            'predictions': {
                'P1': 'pooled category nd ratio median(L24..35)/max(L0..19) '
                      '>= 10',
                'P2': 'each category >= 5 candidates at L28',
                'P3': 'all 10 category-pair Jaccards at L28 <= null q95',
                'P4': 'cos(fruit,plant) > cos(fruit,animal) of mean divergence '
                      'directions at L28',
                'P5': 'knowledge rank-1 rate >= 0.5 per category/colour/size'},
            'status': 'descriptive + preregistered predictions; gear '
                      'candidates are NOT a closed mechanism claim',
            'frozen_before_any_forward': True},
        'material': {
            'categories': CATEGORIES,
            'entities': ENTITIES,
            'n_prompts': n_prompts,
            'color_items': COLOR_ITEMS, 'color_pool': COLOR_POOL,
            'size_items': SIZE_ITEMS, 'size_pool': SIZE_POOL,
            'prompts': [{'text': p['text'], 'kind': p['kind'],
                         'meta': p['meta']} for p in prompts],
            'cat_pairs': material['cat_pairs'],
            'ent_pairs': material['ent_pairs'],
            'att_defs': material['att_defs']},
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
    pool_ids = material['pool_ids']
    pool_names = CATEGORIES + COLOR_POOL + SIZE_POOL
    probe = np.empty((n_prompts, len(pool_names)), dtype=np.float64)
    pool_vec = np.array([pool_ids[w] for w in pool_names])
    name_to_slot = {w: k for k, w in enumerate(pool_names)}
    with torch.inference_mode():
        for pi, p in enumerate(prompts):
            ids = torch.tensor([p['ids']], device=device)
            out = model(ids, output_hidden_states=True)
            h_all[pi] = np.stack([h[0, -1].float().cpu().numpy()
                                  for h in out.hidden_states])
            hsq_all[pi] = (h_all[pi].astype(np.float64) ** 2).sum(axis=1)
            lg = out.logits[0, -3].float().cpu().numpy()  # predicts ids[-2]
            probe[pi] = lg[pool_vec]
            if pi % 80 == 0:
                print('P2767 PROMPT %d/%d' % (pi, n_prompts), flush=True)

    results = {'phase': 2767, 'n_prompts': n_prompts,
               'primary_layer': PRIMARY_LAYER}

    # ---- knowledge probe (P5) ----
    def probe_stats(idxs, true_names):
        rows = []
        for pi, tn in zip(idxs, true_names):
            vals = {w: probe[pi, name_to_slot[w]] for w in
                    set(pool_names) if probe[pi, name_to_slot[w]] != 0 or True}
            rows.append((vals[tn], max(v for w, v in vals.items() if w != tn)))
        margins = np.array([a - b for a, b in rows])
        return {'n': len(rows), 'rank1_rate': float((margins > 0).mean()),
                'margin_median': float(np.median(margins)),
                'margin_mean': float(margins.mean())}
    kp = {}
    for c in CATEGORIES:
        idxs = [material['cat_pairs'][k][0] for k in range(len(material['cat_pairs']))
                if material['cat_pairs'][k][2] == c]
        kp['cat_' + c] = probe_stats(idxs, [c] * len(idxs))
    att_true_names = [d[2] for d in material['att_defs']]
    att_kinds = [d[0] for d in material['att_defs']]
    for kind in ('color', 'size'):
        idxs = [material['att_true_idx'][k]
                for k in range(len(material['att_defs']))
                if att_kinds[k] == kind]
        tns = [att_true_names[k] for k in range(len(material['att_defs']))
               if att_kinds[k] == kind]
        kp['att_' + kind] = probe_stats(idxs, tns)
    results['knowledge_probe'] = kp

    # ---- divergence pairs ----
    def dh_stack(pairs):
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        return h_all[b] - h_all[a], hsq_all[a], hsq_all[b]

    cat_dh, cat_hsq_a, cat_hsq_b = dh_stack(material['cat_pairs'])
    ent_dh, ent_hsq_a, ent_hsq_b = dh_stack(material['ent_pairs'])
    att_pairs = list(zip(material['att_true_idx'], material['att_wrong_idx']))
    att_dh, att_hsq_a, att_hsq_b = dh_stack(att_pairs)
    cat_true_cats = np.array([p[2] for p in material['cat_pairs']])
    cat_wrong_cats = np.array([p[3] for p in material['cat_pairs']])
    att_kinds_arr = np.array(att_kinds)

    results['nd'] = {}
    results['energy'] = {}
    nd_pool, en_pool = nd_curve(cat_dh, cat_hsq_a, cat_hsq_b)
    results['nd']['cat_pooled'] = nd_pool.tolist()
    results['energy']['cat_pooled'] = en_pool.tolist()
    nd_per = {}
    for c in CATEGORIES:
        sel = cat_true_cats == c
        ndc, enc = nd_curve(cat_dh[sel], cat_hsq_a[sel], cat_hsq_b[sel])
        nd_per[c] = ndc
        results['nd']['cat_' + c] = ndc.tolist()
        results['energy']['cat_' + c] = enc.tolist()
    nd_ent, en_ent = nd_curve(ent_dh, ent_hsq_a, ent_hsq_b)
    results['nd']['ent_pooled'] = nd_ent.tolist()
    results['energy']['ent_pooled'] = en_ent.tolist()
    for kind in ('color', 'size'):
        sel = att_kinds_arr == kind
        nda, ena = nd_curve(att_dh[sel], att_hsq_a[sel], att_hsq_b[sel])
        results['nd']['att_' + kind] = nda.tolist()
        results['energy']['att_' + kind] = ena.tolist()

    # P1 deep amplification
    ratio = float(np.median(nd_pool[24:36]) / max(nd_pool[0:20].max(), 1e-30))
    results['P1_deep_amplification_ratio'] = ratio
    results['P1'] = bool(ratio >= 10.0)

    # ---- candidate coordinates ----
    groups = {}
    for c in CATEGORIES:
        groups['cat_' + c] = cat_dh[cat_true_cats == c]
    groups['ent_pooled'] = ent_dh
    for kind in ('color', 'size'):
        groups['att_' + kind] = att_dh[att_kinds_arr == kind]
    # 2760 relation families from archived dh store
    z60 = np.load(BASE / 'phase2760' / 'qwen4_gear_layers' / 'layer_stats.npz',
                  allow_pickle=True)
    dh60, fam60 = z60['dh'], z60['fam_of_pair']
    for f in sorted(set(fam60.tolist())):
        groups['rel2760_' + str(f)] = dh60[fam60 == f]

    n_cand = np.empty((len(groups), L), dtype=np.int64)
    cand_at_primary = {}
    for gi, (g, dhg) in enumerate(sorted(groups.items())):
        n = dhg.shape[0]
        for l in range(L):
            cand, _, _ = cand_coords(dhg[:, l, :].astype(np.float64), n)
            n_cand[gi, l] = len(cand)
            if l == PRIMARY_LAYER:
                cand_at_primary[g] = cand
    results['n_candidates_per_layer'] = {g: n_cand[gi].tolist()
                                         for gi, g in enumerate(sorted(groups))}
    results['candidates_at_L28'] = {g: cand_at_primary[g].tolist()
                                    for g in sorted(cand_at_primary)}
    results['P2'] = {'cat_' + c: bool(len(cand_at_primary['cat_' + c]) >= 5)
                     for c in CATEGORIES}
    results['P2_all'] = bool(all(results['P2'].values()))

    # ---- P3 orthogonality: Jaccard matrix + null ----
    gnames = sorted(groups)
    J = np.zeros((len(gnames), len(gnames)))
    for i in range(len(gnames)):
        for j in range(len(gnames)):
            J[i, j] = jaccard(cand_at_primary[gnames[i]],
                              cand_at_primary[gnames[j]])
    rng = np.random.default_rng(2767001)
    d = d_model
    null = []
    for _ in range(20000):
        a = rng.choice(d, TOPK, replace=False)
        b = rng.choice(d, TOPK, replace=False)
        null.append(len(set(a) & set(b)) / len(set(a) | set(b)))
    null = np.array(null)
    results['jaccard_L28'] = {'groups': gnames, 'matrix': J.tolist(),
                              'null_mean': float(null.mean()),
                              'null_q95': float(np.quantile(null, 0.95))}
    cat_pairs_J = [(gnames[i], gnames[j], J[i, j])
                   for i in range(5) for j in range(i + 1, 5)]
    results['P3'] = {'category_pair_jaccards': cat_pairs_J,
                     'all_within_null_q95': bool(
                         all(j <= np.quantile(null, 0.95)
                             for _, _, j in cat_pairs_J))}

    # ---- P4 semantic distance of mean divergence directions ----
    cos = np.zeros((5, 5))
    means = {}
    for k, c in enumerate(CATEGORIES):
        m = cat_dh[cat_true_cats == c][:, PRIMARY_LAYER, :].astype(np.float64)\
            .mean(axis=0)
        means[c] = m / np.linalg.norm(m)
    for i, a in enumerate(CATEGORIES):
        for j, b in enumerate(CATEGORIES):
            cos[i, j] = float(means[a] @ means[b])
    fp = cos[CATEGORIES.index('fruit'), CATEGORIES.index('plant')]
    fa = cos[CATEGORIES.index('fruit'), CATEGORIES.index('animal')]
    results['category_direction_cosine_L28'] = {
        'categories': CATEGORIES, 'matrix': cos.tolist()}
    results['P4'] = {'cos_fruit_plant': float(fp), 'cos_fruit_animal': float(fa),
                     'supported': bool(fp > fa)}

    # ---- npz ----
    fc.npz(OUT / 'panel_stats.npz',
           h=h_all, hsq=hsq_all, probe=probe,
           cat_dh=cat_dh, cat_true=cat_true_cats.astype(np.str_),
           cat_wrong=cat_wrong_cats.astype(np.str_),
           ent_dh=ent_dh, att_dh=att_dh,
           att_kind=att_kinds_arr.astype(np.str_),
           n_cand=n_cand,
           group_names=np.array(sorted(groups), dtype=np.str_),
           jaccard=J, null=null,
           cosine=cos, nd_pooled=nd_pool, energy_pooled=en_pool,
           **{'nd_cat_%s' % c: nd_per[c] for c in CATEGORIES})
    results['P_all'] = {'P1': results['P1'], 'P2': results['P2_all'],
                        'P3': results['P3']['all_within_null_q95'],
                        'P4': results['P4']['supported'],
                        'P5': {k: v['rank1_rate'] >= 0.5
                               for k, v in kp.items()}}
    results['seconds'] = time.time() - t0
    fc.save(OUT / 'result.json', results)
    print('PHASE2767_DONE seconds=%.1f' % results['seconds'], flush=True)


if __name__ == '__main__':
    import torch
    try:
        main()
    except Exception:
        import traceback
        (OUT / 'crash.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
