"""Phase 2811 (LPF-24): NOUN EMBEDDING SUPERPOSITION FORMAT — the shape
of the gear.

User anchor (2026-09-17): crack the SHAPE of the gear — why is apple's
embedding what it is, how does one embedding express fruit/food/company?
Hypothesis: a systematic superposition format makes noun embeddings both
expressive (multiple class memberships) and extremely efficient.

Zero-forward, 2808 safetensors protocol (embed_tokens + lm_head +
model.norm + config rms_norm_eps).  Class directions/templates built
ONLY from the 2806 atlas (template-eval separation, 2807 protocol).

Batteries:
  atlas  100 words (2806, template source)
  held    99 words (2807)
  NEW    ~170 candidates (this script, tokenizer-filtered) -> eval

Arms:
  A  census: 10-dim class-feature profile per noun; argmax-secondary
     map M[c1][c2] on atlas / eval / all                  -> P-L1
  B  cross-word prediction: atlas-only secondary profile R[c1][c2]
     predicts eval words' secondary class                 -> P-L2
  C  efficiency: acc_k on enlarged battery; class-subspace energy
     share; residual effective rank
  E  exploratory (no prereg): capitalization sense pairs
     apple/Apple, turkey/Turkey, china/China, japan/Japan

Nulls (2809 lesson institutionalized):
  null-A1  eval label shuffle (class sizes preserved) x1000
           -> per-cell secondary-rate q95               (P-L1)
  null-A2  atlas label shuffle x1000, rebuild profile R,
           predict eval observed secondaries            (P-L2 p-value)
  null-B   250 random vocab tokens -> participation-ratio and
           secondary-strength baseline

Prereg (frozen before any readout):
  P-L1  superposition_map_real iff exists cell (c1,c2), c2!=c1, with
        eval-subset rate >= 0.50 (n_c1_eval >= 8) AND atlas-subset
        rate >= 0.40 AND eval rate > null-A1 q95 for that cell
  P-L2  secondary_systematic iff profile prediction hit rate >= 2/9
        AND null-A2 p < 0.001
  P-L3  battery_generalization iff nearest-centroid accuracy (10-dim
        feats, atlas templates) on eval battery >= 0.75 (chance 0.10)
  verdict: gear_shape_superposition iff P-L1 AND P-L2
        (P-L3 = battery validity gate)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2811' / 'noun_superposition'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SEED = 2811
N_NULL = 1000
N_RAND = 250

NEW = {
    'fruit': ['nectarine', 'tangerine', 'cranberry', 'gooseberry',
              'plantain', 'persimmon', 'cantaloupe', 'honeydew',
              'kumquat', 'clementine', 'sultana', 'loganberry'],
    'animal': ['camel', 'donkey', 'eagle', 'falcon', 'owl', 'shark',
               'seal', 'otter', 'beaver', 'squirrel', 'hedgehog',
               'raccoon', 'moose', 'bison', 'panther', 'leopard',
               'hyena', 'badger', 'hamster', 'tortoise', 'lizard',
               'python', 'cobra', 'sparrow', 'crow', 'crane', 'swan',
               'pigeon', 'gorilla', 'baboon'],
    'metal': ['alloy', 'pewter', 'solder', 'ingot', 'ore', 'bauxite',
              'graphite'],
    'vehicle': ['van', 'jeep', 'sled', 'cart', 'buggy', 'moped',
                'tricycle', 'rickshaw', 'chariot', 'sailboat',
                'rowboat', 'steamboat', 'airship', 'blimp', 'trolley',
                'cab'],
    'country': ['Spain', 'Qatar', 'Oman', 'Yemen', 'Nepal', 'Ghana',
                'Mali', 'Chad', 'Cuba', 'Laos', 'Fiji', 'Malta',
                'Korea', 'Mongolia', 'Thailand', 'Vietnam', 'Nigeria',
                'Angola', 'Bolivia', 'Ecuador', 'Uruguay', 'Panama',
                'Jamaica'],
    'food': ['donut', 'bagel', 'muffin', 'cereal', 'oatmeal',
             'porridge', 'broth', 'gravy', 'salsa', 'hummus',
             'falafel', 'sushi', 'kimchi', 'pretzel', 'waffle',
             'pastry', 'custard', 'pudding', 'yogurt', 'kebab'],
    'nature': ['cliff', 'dune', 'creek', 'brook', 'glacier', 'volcano',
               'tornado', 'breeze', 'hail', 'sunset', 'sunrise',
               'marsh', 'swamp', 'lagoon', 'reef', 'cavern', 'boulder',
               'pebble', 'geyser', 'avalanche'],
    'furniture': ['futon', 'vanity', 'rocker', 'settee', 'bureau',
                  'armoire', 'recliner', 'loveseat', 'daybed',
                  'headboard', 'footstool', 'hassock', 'credenza'],
    'tool': ['awl', 'rasp', 'shears', 'tongs', 'funnel', 'gauge',
             'winch', 'hoist', 'jack', 'mill', 'loom', 'level'],
    'clothing': ['blazer', 'hoodie', 'tuxedo', 'parka', 'sandal',
                 'slipper', 'loafer', 'mitten', 'beanie', 'turban',
                 'poncho', 'tunic', 'kilt', 'corset', 'suit', 'garter',
                 'sarong', 'bikini'],
}
PAIRS = [('apple', 'Apple'), ('turkey', 'Turkey'),
         ('china', 'China'), ('japan', 'Japan')]

PREREG = {
    'P-L1': 'superposition_map_real iff exists cell (c1,c2), c2!=c1, '
            'with eval-subset rate >= 0.50 (n_c1_eval >= 8) AND '
            'atlas-subset rate >= 0.40 AND eval rate > null-A1 q95',
    'P-L2': 'secondary_systematic iff atlas-profile secondary '
            'prediction hit >= 2/9 AND null-A2 p < 0.001',
    'P-L3': 'battery_generalization iff nearest-centroid acc (10-dim '
            'feats, atlas templates) on eval >= 0.75 (chance 0.10)',
    'verdict': 'gear_shape_superposition iff P-L1 AND P-L2; '
               'P-L3 = battery validity gate',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)

    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())
    exec2807 = json.loads((SRC_2807 / 'execution.json').read_text(
        encoding='utf-8'))
    HELD = exec2807['held']
    assert list(HELD.keys()) == CAT_WORDS

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED, 'n_null': N_NULL,
                 'n_rand': N_RAND, 'new_candidates': NEW,
                 'sense_pairs': PAIRS,
                 'note': 'templates from 2806 atlas only; eval = 2807 '
                         'held-out + NEW battery'}
    fc.save(OUT / 'execution.json', execution)

    # ---------- tensors (2808 safetensors protocol) ----------
    from safetensors import safe_open
    mdir = ROOT / 'models' / 'hf' / 'qwen3-4b'
    index = json.loads((mdir / 'model.safetensors.index.json')
                       .read_text(encoding='utf-8'))['weight_map']

    def read_tensor(name):
        with safe_open(str(mdir / index[name]), framework='pt') as f:
            return f.get_tensor(name).float().numpy()

    Etab = read_tensor('model.embed_tokens.weight')
    g = read_tensor('model.norm.weight').astype(np.float64)
    try:
        Wu = read_tensor('lm_head.weight')
        tie = False
    except KeyError:
        Wu = Etab
        tie = True
    cfg = json.loads((mdir / 'config.json').read_text(encoding='utf-8'))
    eps = float(cfg.get('rms_norm_eps', 1e-6))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(mdir), local_files_only=True, trust_remote_code=True,
        use_fast=True)

    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, t
            tc[t] = int(ids[0])
        return tc[t]

    atlas_words = [w for v in CATS.values() for w in v]
    held_words = [w for v in HELD.values() for w in v]
    for c in CAT_WORDS:
        tid(c)
    for w in atlas_words + held_words:
        tid(w)

    # ---------- battery assembly ----------
    lower_known = set(w.lower() for w in atlas_words + held_words
                      + CAT_WORDS)
    new_kept, new_rejected = {}, {}
    seen_new = set()
    for c in CAT_WORDS:
        kept, rej = [], []
        for w in NEW[c]:
            if w.lower() in lower_known or w.lower() in seen_new:
                rej.append(w + ' dup')
                continue
            try:
                tid(w)
            except AssertionError:
                rej.append(w + ' multi-tok')
                continue
            kept.append(w)
            seen_new.add(w.lower())
        new_kept[c] = kept
        new_rejected[c] = rej
    n_new = sum(len(v) for v in new_kept.values())
    print('P2811 new battery survivors %d (candidates %d)'
          % (n_new, sum(len(v) for v in NEW.values())), flush=True)

    eval_pairs = [(w, c) for c in CAT_WORDS for w in HELD[c]] \
        + [(w, c) for c in CAT_WORDS for w in new_kept[c]]
    eval_words = [w for w, _ in eval_pairs]
    ytrue = np.array([CAT_WORDS.index(c) for _, c in eval_pairs])
    n_held = len(held_words)
    atlas_labels = np.array(
        [ci for ci, c in enumerate(CAT_WORDS) for _ in CATS[c]])

    all_tids = set(tc.values())

    def zrow(t):
        e = Etab[tid(t)].astype(np.float64)
        return e / np.sqrt((e ** 2).mean() + eps) * g

    def zrow_id(i):
        e = Etab[i].astype(np.float64)
        return e / np.sqrt((e ** 2).mean() + eps) * g

    Z_atlas = np.stack([zrow(w) for w in atlas_words])
    Z_eval = np.stack([zrow(w) for w in eval_words])

    # ---------- gates ----------
    h7 = np.load(SRC_2807 / 'heldout.npz')
    cent = {c: np.stack([Wu[tid(w)].astype(np.float64)
                         for w in CATS[c]]).mean(0) for c in CAT_WORDS}
    Cm = np.stack([cent[c] for c in CAT_WORDS])
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    gate_dW = float(np.abs(dW - h7['dW_class'].astype(np.float64)).max())
    gate_Z = float(np.abs(Z_eval[:n_held]
                          - h7['Znew'].astype(np.float64)).max())
    overlap = sorted(set(w.lower() for w in eval_words)
                     & set(w.lower() for w in atlas_words))
    print('P2811 gates: dW_vs_2807=%.2e Z_vs_2807=%.2e overlap=%d tie=%s'
          % (gate_dW, gate_Z, len(overlap), tie), flush=True)
    assert gate_dW < 1e-6 and gate_Z < 1e-4 and not overlap

    unitD = np.stack([unit(dW[i]) for i in range(10)])
    F_atlas = Z_atlas @ unitD.T
    F_eval = Z_eval @ unitD.T

    # ---------- templates (atlas only, 2807 protocol) ----------
    tmpl = {}
    for ci, c in enumerate(CAT_WORDS):
        idx = [i for i in range(len(atlas_words))
               if atlas_labels[i] == ci]
        tmpl[c] = F_atlas[idx].mean(0)
    Tm = np.stack([tmpl[c] for c in CAT_WORDS])
    Tm_unit = np.stack([unit(Tm[i]) for i in range(10)])
    tmpl_cos = Tm_unit @ Tm_unit.T

    def nca(F, cents):
        out = []
        for i in range(F.shape[0]):
            dd = [np.linalg.norm(F[i] - cc_) for cc_ in cents]
            out.append(int(np.argmin(dd)))
        return out

    # ---------- Arm C: accuracy on enlarged battery ----------
    pred10 = nca(F_eval, [tmpl[c] for c in CAT_WORDS])
    acc10 = float(np.mean(np.array(pred10) == ytrue))
    accs_k = {}
    Qlist = [unitD[i] for i in range(10)]
    for k in range(1, 11):
        Qk = np.stack(Qlist[:k])
        Fak = Z_atlas @ Qk.T
        Fev = Z_eval @ Qk.T
        cfk = np.stack([Fak[atlas_labels == ci].mean(0)
                        for ci in range(10)])
        pr = nca(Fev, [cfk[c] for c in range(10)])
        accs_k[k] = float(np.mean(np.array(pr) == ytrue))
    p_l3 = bool(acc10 >= 0.75)
    print('P2811 Arm C acc10=%.3f acc_k=%s P-L3=%s'
          % (acc10, {k: round(v, 3) for k, v in accs_k.items()}, p_l3),
          flush=True)

    # ---------- profiles ----------
    def profiles(F):
        Fu = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True),
                            1e-30)
        return Fu @ Tm_unit.T

    P_atlas = profiles(F_atlas)
    P_eval = profiles(F_eval)

    def secondary(P, y):
        rows = []
        for i in range(P.shape[0]):
            pc = P[i].copy()
            pc[y[i]] = -np.inf
            j = int(np.argmax(pc))
            rows.append((j, float(pc[j])))
        return rows

    sec_atlas = secondary(P_atlas, atlas_labels)
    sec_eval = secondary(P_eval, ytrue)

    def mmap(rows, y):
        cnt = np.zeros((10, 10))
        n = np.zeros(10)
        for (j, _), c1 in zip(rows, y):
            cnt[c1, j] += 1
            n[c1] += 1
        rates = np.divide(cnt, n[:, None],
                          out=np.zeros((10, 10)), where=n[:, None] > 0)
        return cnt, n, rates

    cnt_atlas, n_atlas, M_atlas = mmap(sec_atlas, atlas_labels)
    cnt_eval, n_eval, M_eval = mmap(sec_eval, ytrue)
    cnt_all = cnt_atlas + cnt_eval
    n_all = n_atlas + n_eval
    M_all = np.divide(cnt_all, n_all[:, None],
                      out=np.zeros((10, 10)), where=n_all[:, None] > 0)

    # ---------- null-A1: eval label shuffle (class sizes kept) ----------
    rng = np.random.default_rng(SEED)
    sizes = [int((ytrue == c).sum()) for c in range(10)]
    null_cell = np.zeros((N_NULL, 10, 10))
    for it in range(N_NULL):
        perm = rng.permutation(len(eval_words))
        lab_n = np.empty(len(eval_words), dtype=int)
        pos = 0
        for c in range(10):
            lab_n[perm[pos:pos + sizes[c]]] = c
            pos += sizes[c]
        rows_n = secondary(P_eval, lab_n)
        _, _, rates_n = mmap(rows_n, lab_n)
        null_cell[it] = rates_n
    q95_cell = np.quantile(null_cell, 0.95, axis=0)

    # ---------- Arm B: atlas profile -> eval secondary ----------
    R = np.stack([P_atlas[atlas_labels == c].mean(0) for c in range(10)])
    hits = 0
    pred_sec = []
    for i in range(len(eval_words)):
        row = R[ytrue[i]].copy()
        row[ytrue[i]] = -np.inf
        j = int(np.argmax(row))
        pred_sec.append(j)
        hits += int(j == sec_eval[i][0])
    hit_rate = hits / len(eval_words)
    null_hits = []
    for it in range(N_NULL):
        perm = rng.permutation(len(atlas_words))
        lab_n = np.empty(len(atlas_words), dtype=int)
        for pos, wi in enumerate(perm):
            lab_n[wi] = pos // 10
        Rn = np.stack([P_atlas[lab_n == c].mean(0) for c in range(10)])
        h = 0
        for i in range(len(eval_words)):
            row = Rn[ytrue[i]].copy()
            row[ytrue[i]] = -np.inf
            h += int(np.argmax(row) == sec_eval[i][0])
        null_hits.append(h / len(eval_words))
    null_hits = np.array(null_hits)
    p_l2_null = float((1 + int((null_hits >= hit_rate).sum()))
                      / (N_NULL + 1))
    p_l2 = bool(hit_rate >= 2.0 / 9.0 and p_l2_null < 0.001)
    print('P2811 Arm B hit=%.3f (chance 0.111) null median=%.3f '
          'q95=%.3f p=%.4f P-L2=%s'
          % (hit_rate, float(np.median(null_hits)),
             float(np.quantile(null_hits, 0.95)), p_l2_null, p_l2),
          flush=True)

    # ---------- P-L1 ----------
    found = []
    for c1 in range(10):
        if n_eval[c1] < 8:
            continue
        for c2 in range(10):
            if c2 == c1:
                continue
            r_e, r_a = M_eval[c1, c2], M_atlas[c1, c2]
            if r_e >= 0.50 and r_a >= 0.40 and r_e > q95_cell[c1, c2]:
                found.append({'from': CAT_WORDS[c1], 'to': CAT_WORDS[c2],
                              'rate_eval': round(float(r_e), 3),
                              'n_eval': int(n_eval[c1]),
                              'rate_atlas': round(float(r_a), 3),
                              'null_q95': round(float(q95_cell[c1, c2]),
                                                3)})
    p_l1 = bool(found)
    print('P2811 P-L1 cells=%s' % found, flush=True)

    # ---------- null-B: random tokens ----------
    rand_ids = []
    while len(rand_ids) < N_RAND:
        r = int(rng.integers(0, Etab.shape[0]))
        if r > 0 and r not in all_tids and r not in rand_ids:
            rand_ids.append(r)
    Z_rand = np.stack([zrow_id(i) for i in rand_ids])
    P_rand = profiles(Z_rand @ unitD.T)
    e_rand = P_rand ** 2
    pr_rand = (e_rand.sum(1)) ** 2 / np.maximum((e_rand ** 2).sum(1),
                                                1e-30)
    sec_rand = secondary(P_rand, np.argmax(P_rand, axis=1))
    rand_sec_q95 = float(np.quantile([s for _, s in sec_rand], 0.95))

    def part_ratio(P):
        e = P ** 2
        return (e.sum(1)) ** 2 / np.maximum((e ** 2).sum(1), 1e-30)

    pr_atlas = part_ratio(P_atlas)
    pr_eval = part_ratio(P_eval)
    print('P2811 null-B PR rand mean=%.2f | atlas %.2f | eval %.2f | '
          'rand sec q95=%.3f'
          % (float(pr_rand.mean()), float(pr_atlas.mean()),
             float(pr_eval.mean()), rand_sec_q95), flush=True)

    # ---------- Arm C: energy + residual rank ----------
    Z_all = np.vstack([Z_atlas, Z_eval])
    Qb, _ = np.linalg.qr(unitD.T)
    shares = (np.linalg.norm(Z_all @ Qb, axis=1) ** 2
              / np.maximum(np.linalg.norm(Z_all, axis=1) ** 2, 1e-30))
    Zc = Z_all - Z_all.mean(0, keepdims=True)
    resid = Zc - (Zc @ Qb) @ Qb.T
    sv = np.linalg.svd(resid, compute_uv=False)
    eff_rank = int((sv > sv[0] * 0.01).sum())

    # ---------- Arm E: sense pairs (exploratory) ----------
    pair_rows = []
    for lo, up in PAIRS:
        try:
            zl, zu = zrow(lo), zrow(up)
        except AssertionError:
            pair_rows.append({'pair': '%s/%s' % (lo, up),
                              'error': 'multi-token'})
            continue
        cos = float(unit(zl) @ unit(zu))
        pl = profiles((zl @ unitD.T)[None, :])[0]
        pu = profiles((zu @ unitD.T)[None, :])[0]
        pair_rows.append({
            'pair': '%s/%s' % (lo, up),
            'cos_lower_upper': round(cos, 4),
            'profile_lower': [round(float(x), 3) for x in pl],
            'profile_upper': [round(float(x), 3) for x in pu],
            'argmax_lower': CAT_WORDS[int(np.argmax(pl))],
            'argmax_upper': CAT_WORDS[int(np.argmax(pu))],
        })
    print('P2811 Arm E %s' % json.dumps(pair_rows), flush=True)

    # ---------- verdict ----------
    verdict = {
        'n_atlas': len(atlas_words), 'n_held': n_held, 'n_new': n_new,
        'n_eval': len(eval_words),
        'new_survivors': new_kept, 'new_rejected': new_rejected,
        'acc10_eval': round(acc10, 3),
        'acc_k_eval': {str(k): round(v, 3) for k, v in accs_k.items()},
        'battery_generalization': p_l3,
        'secondary_hit_rate': round(hit_rate, 4),
        'secondary_chance': round(1 / 9, 4),
        'null_hit_median': round(float(np.median(null_hits)), 4),
        'null_hit_q95': round(float(np.quantile(null_hits, 0.95)), 4),
        'null_hit_p': round(p_l2_null, 4),
        'secondary_systematic': p_l2,
        'pl1_cells': found,
        'superposition_map_real': p_l1,
        'pr_random_mean': round(float(pr_rand.mean()), 2),
        'pr_atlas_mean': round(float(pr_atlas.mean()), 2),
        'pr_eval_mean': round(float(pr_eval.mean()), 2),
        'rand_secondary_q95': round(rand_sec_q95, 3),
        'class_subspace_energy_share_mean': round(float(shares.mean()),
                                                  4),
        'residual_effective_rank': eff_rank,
        'sense_pairs': pair_rows,
        'gear_shape_superposition': bool(p_l1 and p_l2),
    }
    result = {'phase': 2811, 'prereg': PREREG, 'verdict': verdict,
              'M_eval': np.round(M_eval, 3).tolist(),
              'M_atlas': np.round(M_atlas, 3).tolist(),
              'M_all': np.round(M_all, 3).tolist(),
              'M_all_counts': cnt_all.astype(int).tolist(),
              'n_per_class_all': n_all.astype(int).tolist(),
              'R_profile': np.round(R, 3).tolist(),
              'q95_cell': np.round(q95_cell, 3).tolist(),
              'template_cos': np.round(tmpl_cos, 3).tolist(),
              'eval_words': eval_words,
              'rand_ids': rand_ids}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'battery.npz',
           Z_atlas=Z_atlas.astype(np.float32),
           Z_eval=Z_eval.astype(np.float32),
           unitD=unitD.astype(np.float32),
           ytrue=ytrue, atlas_labels=atlas_labels,
           M_all=M_all.astype(np.float32),
           R=R.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2811', elapsed)
    print('P2811 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2811 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
