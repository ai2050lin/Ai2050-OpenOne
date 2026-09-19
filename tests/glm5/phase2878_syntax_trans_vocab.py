# -*- coding: utf-8 -*-
"""Phase 2878: syntax-axis + translation-axis vocabulary (zero forward).

Goal: extend the mechanism-base growth curve to two NEW axis families
beyond class (2861/2867) and attribute (2877):
  - syntax axes  : morphological polarity pairs (same root, two forms)
  - translation axes : cross-lingual same-concept pairs (en vs L)
Axis direction = unit(emb[+pole] - emb[-pole]) exactly as 2874.

Gen1 -> Gen2 amendment (preregistered, gates unchanged):
  Gen1 verdicts: syntax family VG1 3/4 pass (comparative 3/5 pairs),
  translation family VG1 2/3 (lang_fr only 4/8 valid pairs), and VG2b
  killed every translation axis (0.064-0.123, all < 0.2) while syntax
  number 0.2021 / gerund 0.2352 survived.  Diagnosed Gen1 defect:
  HOMOGRAPH CONTAMINATION - single_token_id prefers ' '+t, so 'chat'
  resolved to the English "chat" token, not French "cat"; likewise
  sol/table-class collisions distort pair directions.  Per the 2874
  expansion precedent (pool power, gates frozen), Gen2 (a) expands the
  translation pools, (b) adds a frozen homograph exclusion list
  applied to both poles before any tokenization, (c) expands the
  comparative pool.  Judgement is DUAL-TRACK: syn_legal (syntax
  family) and trans_legal (translation family) are registered
  separately; a translation VG2b failure after decontamination is a
  REAL result (no unified cross-lingual axis direction in the tied
  unembed), not a vocabulary failure.

Pools (polarity-ordered pairs (+pole, -pole)):
  syntax:
    number      cats/cat dogs/dog books/book cars/car ideas/idea rivers/river
    tense       walked/walk jumped/jump cooked/cook painted/paint talked/talk
    gerund      walking/walk sleeping/sleep reading/read singing/sing eating/eat
    comparative sweeter/sweet softer/soft louder/loud richer/rich
                braver/brave cleaner/clean cheaper/cheap closer/close
                longer/long harder/hard darker/dark warmer/warm
                cooler/cool neater/neat kinder/kind
  translation (en vs L, same concept):
    lang_fr     cat/chat house/maison book/livre night/nuit roi/king
                femme/woman homme/man arbre/tree vin/wine lune/moon
                mer/sea ciel/sky fleur/flower ville/city guerre/war
                pied/foot water/eau sun/soleil milk/lait dog/chien
    lang_de     cat/Katze dog/Hund water/Wasser house/Haus book/Buch
                night/Nacht sun/Sonne milk/Milch Vogel/bird Fisch/fish
                Brot/bread Wein/wine Mond/moon Stern/star Baum/tree
                Feuer/fire Berg/mountain Schnee/snow Frau/woman
                Wolf/wolf Koenig/king Sommer/summer
    lang_es     water/agua house/casa book/libro night/noche cat/gato
                dog/perro milk/leche mesa/table luna/moon mar/sea
                cielo/sky vino/wine flor/flower rey/king mujer/woman
                hombre/man madre/mother padre/father guerra/war
                luz/light fuego/fire tierra/earth puerta/door
                caballo/horse queso/cheese huevo/egg lobo/wolf
                nieve/snow cama/bed estrella/star
Homograph exclusion (frozen; applied to BOTH poles, then L-pole
residuals screened): en-pole words are by construction English; L-pole
exclusions = words whose form is a common English word:
  fr: chat sol pain chef table main train route lion grand mine
  de: Mann Gold Winter Hand Bank Wind Kind Sommer Morgen Boot Arm
      Haus-parallel null; (Katze Hund Wasser Buch Nacht Sonne Milch
      Vogel Fisch Brot Wein Mond Stern Baum Feuer Berg Schnee Frau
      Wolf Koenig all clear)
  es: sol pan plaza real color metal hotel piano radio rosa mano

Prereg (frozen before any unembed statistic is computed):
  VG0  homograph exclusion list applied before tokenization (Gen2).
  VG1  coverage: per axis, >= 5 pairs with BOTH poles single-token.
       Syntax family passes iff >= 3/4 axes; translation family passes
       iff >= 3/3 axes.
  VG2a cross-axis orthogonality (over all surviving axes):
       max |cos(dir_a,dir_b)| < 0.488 (2858 G2 gate, frozen unchanged).
  VG2b within-axis pair reproducibility: per axis with >= 2 valid pairs,
       mean pair-pair direction cos > 0.2.  An axis SURVIVES to the
       spectrum phase iff VG1 pass AND its VG2b > 0.2.  (Family means
       reported descriptively; survival is per-axis, so a weak axis is
       quarantined rather than dragging its family below the gate.)
  VG3  norm health (surviving axes): max per-axis word norm ratio < 3.0.
  verdict (dual track): syn_legal iff >= 2 syntax axes survive AND
       VG2a AND VG3; trans_legal iff >= 2 translation axes survive
       (VG2a/VG3 are global over the union and re-reported per track).
"""
import hashlib
import io
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2878', 'syntax_trans_vocab')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2878

AXES = {
    # ---- syntax family: morphological pairs ----
    'number': [('num1', 'cats', 'cat'), ('num2', 'dogs', 'dog'),
               ('num3', 'books', 'book'), ('num4', 'cars', 'car'),
               ('num5', 'ideas', 'idea'), ('num6', 'rivers', 'river')],
    'tense': [('t1', 'walked', 'walk'), ('t2', 'jumped', 'jump'),
              ('t3', 'cooked', 'cook'), ('t4', 'painted', 'paint'),
              ('t5', 'talked', 'talk')],
    'gerund': [('g1', 'walking', 'walk'), ('g2', 'sleeping', 'sleep'),
               ('g3', 'reading', 'read'), ('g4', 'singing', 'sing'),
               ('g5', 'eating', 'eat')],
    'comparative': [('c1', 'sweeter', 'sweet'), ('c2', 'softer', 'soft'),
                    ('c3', 'louder', 'loud'), ('c4', 'richer', 'rich'),
                    ('c5', 'braver', 'brave'),
                    ('c6', 'cleaner', 'clean'), ('c7', 'cheaper', 'cheap'),
                    ('c8', 'closer', 'close'), ('c9', 'longer', 'long'),
                    ('c10', 'harder', 'hard'), ('c11', 'darker', 'dark'),
                    ('c12', 'warmer', 'warm'), ('c13', 'cooler', 'cool'),
                    ('c14', 'neater', 'neat'), ('c15', 'kinder', 'kind')],
    # ---- translation family: en vs L, same concept ----
    'lang_fr': [('f1', 'cat', 'chat'), ('f2', 'house', 'maison'),
                ('f3', 'book', 'livre'), ('f4', 'night', 'nuit'),
                ('f5', 'roi', 'king'), ('f6', 'femme', 'woman'),
                ('f7', 'homme', 'man'), ('f8', 'arbre', 'tree'),
                ('f9', 'vin', 'wine'), ('f10', 'lune', 'moon'),
                ('f11', 'mer', 'sea'), ('f12', 'ciel', 'sky'),
                ('f13', 'fleur', 'flower'), ('f14', 'ville', 'city'),
                ('f15', 'guerre', 'war'), ('f16', 'pied', 'foot'),
                ('f17', 'water', 'eau'), ('f18', 'sun', 'soleil'),
                ('f19', 'milk', 'lait'), ('f20', 'dog', 'chien')],
    'lang_de': [('d1', 'cat', 'Katze'), ('d2', 'dog', 'Hund'),
                ('d3', 'water', 'Wasser'), ('d4', 'house', 'Haus'),
                ('d5', 'book', 'Buch'), ('d6', 'night', 'Nacht'),
                ('d7', 'sun', 'Sonne'), ('d8', 'milk', 'Milch'),
                ('d9', 'Vogel', 'bird'), ('d10', 'Fisch', 'fish'),
                ('d11', 'Brot', 'bread'), ('d12', 'Wein', 'wine'),
                ('d13', 'Mond', 'moon'), ('d14', 'Stern', 'star'),
                ('d15', 'Baum', 'tree'), ('d16', 'Feuer', 'fire'),
                ('d17', 'Berg', 'mountain'), ('d18', 'Schnee', 'snow'),
                ('d19', 'Frau', 'woman'), ('d20', 'Wolf', 'wolf'),
                ('d21', 'Koenig', 'king'), ('d22', 'Sommer', 'summer')],
    'lang_es': [('e1', 'water', 'agua'), ('e2', 'house', 'casa'),
                ('e3', 'book', 'libro'), ('e4', 'night', 'noche'),
                ('e5', 'cat', 'gato'), ('e6', 'dog', 'perro'),
                ('e7', 'milk', 'leche'), ('e8', 'mesa', 'table'),
                ('e9', 'luna', 'moon'), ('e10', 'mar', 'sea'),
                ('e11', 'cielo', 'sky'), ('e12', 'vino', 'wine'),
                ('e13', 'flor', 'flower'), ('e14', 'rey', 'king'),
                ('e15', 'mujer', 'woman'), ('e16', 'hombre', 'man'),
                ('e17', 'madre', 'mother'), ('e18', 'padre', 'father'),
                ('e19', 'guerra', 'war'), ('e20', 'luz', 'light'),
                ('e21', 'fuego', 'fire'), ('e22', 'tierra', 'earth'),
                ('e23', 'puerta', 'door'), ('e24', 'caballo', 'horse'),
                ('e25', 'queso', 'cheese'), ('e26', 'huevo', 'egg'),
                ('e27', 'lobo', 'wolf'), ('e28', 'nieve', 'snow'),
                ('e29', 'cama', 'bed'), ('e30', 'estrella', 'star')],
}
SYNTAX_FAMILY = ['number', 'tense', 'gerund', 'comparative']
TRANS_FAMILY = ['lang_fr', 'lang_de', 'lang_es']

# VG0 (Gen2, frozen): homograph exclusion - L-pole words whose form is
# a common English word (the ' chat' -> English "chat" defect); also
# dropped from the en pole for symmetric hygiene.
HOMOGRAPH_EXCLUDE = {
    'chat', 'sol', 'pain', 'chef', 'table', 'main', 'train', 'route',
    'lion', 'grand', 'mine', 'Mann', 'Gold', 'Winter', 'Hand', 'Bank',
    'Wind', 'Kind', 'Sommer', 'Morgen', 'Boot', 'Arm', 'pan', 'plaza',
    'real', 'color', 'metal', 'hotel', 'piano', 'radio', 'rosa',
}

PREREG = {
    'VG0': 'Gen2 homograph exclusion list (frozen in script) applied to '
           'both poles before tokenization',
    'VG1': 'per axis >= 5 pairs both poles single-token; syntax >= 3/4, '
           'translation >= 3/3',
    'VG2a': 'max cross-axis |cos(dir_a,dir_b)| < 0.488 (2858 G2 gate)',
    'VG2b': 'per-axis mean pair-pair dir cos > 0.2 for axis survival '
            '(axes with >= 2 valid pairs); weak axes quarantined',
    'VG3': 'max per-axis word norm ratio max/min < 3.0 (surviving axes)',
    'verdict': 'dual track: syn_legal iff >=2 syntax axes survive and '
               'VG2a and VG3; trans_legal iff >=2 translation axes '
               'survive (trans VG2b failure = real negative: no unified '
               'cross-lingual axis direction)',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def load_embed(model_dir):
    idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(idx_path):
        idx = json.load(io.open(idx_path, encoding='utf-8'))
        shard = idx['weight_map']['model.embed_tokens.weight']
    else:
        shard = 'model.safetensors'
    from safetensors import safe_open
    with safe_open(os.path.join(model_dir, shard), framework='pt') as f:
        emb = f.get_tensor('model.embed_tokens.weight').float().numpy()
    return emb


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.monotonic()
    os.makedirs(OUT, exist_ok=True)

    script = os.path.abspath(__file__)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2878, 'name': 'syntax_trans_vocab',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(script),
                   'prereg': PREREG, 'seed': SEED,
                   'n_pool_pairs': sum(len(v) for v in AXES.values())},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=True,
        use_fast=True)

    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from rdc_atlas_census import single_token_id

    emb = load_embed(MODEL_DIR)
    log('embed loaded %s' % (emb.shape,))

    # ---------- VG0: homograph exclusion ----------
    n_excl = 0
    eff_axes = {}
    for axis, pairs in AXES.items():
        keep = []
        for pid, wp, wm in pairs:
            if wp in HOMOGRAPH_EXCLUDE or wm in HOMOGRAPH_EXCLUDE:
                n_excl += 1
                continue
            keep.append((pid, wp, wm))
        eff_axes[axis] = keep
    log('VG0 excluded %d homograph pairs' % n_excl)

    # ---------- VG1: single-token filtering (both poles) ----------
    tc = {}
    axis_words = {}       # all single-token words per axis (both poles)
    axis_pairs = {}       # valid pairs per axis
    for axis, pairs in eff_axes.items():
        words = []
        valid = []
        for _, wp, wm in pairs:
            ok_p = ok_m = False
            try:
                single_token_id(tok, wp, tc)
                ok_p = True
            except AssertionError:
                pass
            try:
                single_token_id(tok, wm, tc)
                ok_m = True
            except AssertionError:
                pass
            if ok_p and ok_m:
                valid.append((wp, wm))
            for w, ok in ((wp, ok_p), (wm, ok_m)):
                if ok and w not in words:
                    words.append(w)
        axis_words[axis] = words
        axis_pairs[axis] = valid
        log('VG1 %-11s %d/%d pairs: %s'
            % (axis, len(valid), len(pairs),
               '; '.join('%s/%s' % p for p in valid)))

    g1 = {a: len(axis_pairs[a]) >= 5 for a in AXES}
    syn_ok = sum(g1[a] for a in SYNTAX_FAMILY)
    tr_ok = sum(g1[a] for a in TRANS_FAMILY)
    vg1 = bool(syn_ok >= 3 and tr_ok >= 3)
    log('VG1 syntax %d/4 translation %d/3 -> %s' % (syn_ok, tr_ok, vg1))

    # ---------- axis directions from valid pairs ----------
    axis_dirs = {}
    pair_dirs = {}
    for a in AXES:
        dirs = []
        for wp, wm in axis_pairs[a]:
            d = unit(emb[tc[wp]] - emb[tc[wm]])
            dirs.append(d)
            pair_dirs.setdefault(a, []).append(d)
        if dirs:
            axis_dirs[a] = unit(np.stack(dirs).mean(axis=0))

    # ---------- VG2b: per-axis pair reproducibility -> survival ----------
    vg2b = {}
    for a, dirs in pair_dirs.items():
        if len(dirs) >= 2:
            D = np.stack(dirs)
            S = D @ D.T
            iu = np.triu_indices(len(dirs), 1)
            vg2b[a] = float(S[iu].mean())
        else:
            vg2b[a] = float('nan')
    surviving = [a for a in axis_dirs
                 if g1[a] and len(pair_dirs[a]) >= 2
                 and vg2b[a] > 0.2]
    syn_surv = [a for a in SYNTAX_FAMILY if a in surviving]
    tr_surv = [a for a in TRANS_FAMILY if a in surviving]
    for a in axis_dirs:
        log('VG2b %-11s %.4f -> %s' % (a, vg2b[a],
                                       'survive' if a in surviving
                                       else 'quarantined'))

    # ---------- VG2a: cross-axis orthogonality (surviving axes) ----------
    dW = np.stack([axis_dirs[a] for a in surviving])
    C = dW @ dW.T
    off = C[np.triu_indices(len(surviving), 1)]
    max_off = float(np.abs(off).max())
    vg2a = bool(max_off < 0.488)

    # ---------- VG3: norm health ----------
    worst_ratio = 0.0
    worst_axis = ''
    for a in surviving:
        norms = [float(np.linalg.norm(emb[tc[w]]))
                 for w in axis_words[a]]
        r = max(norms) / max(min(norms), 1e-30)
        if r > worst_ratio:
            worst_ratio, worst_axis = r, a
    vg3 = bool(worst_ratio < 3.0)

    n_syn, n_tr = len(syn_surv), len(tr_surv)
    syn_legal = bool(n_syn >= 2 and vg2a and vg3)
    trans_legal = bool(n_tr >= 2 and vg2a and vg3)
    verdict = bool(syn_legal or trans_legal)

    # ---------- save census-ready vocab (2874 schema) ----------
    target_list = [(a, w) for a in surviving for w in axis_words[a]]
    n_words = len(target_list)
    rng = np.random.default_rng(SEED)
    word_tids = set(tc.values())
    null_tids = {}
    while len(null_tids) < n_words:
        r = int(rng.integers(0, emb.shape[0]))
        if r not in word_tids and r > 0:
            null_tids[len(null_tids)] = r
    func_tid = single_token_id(tok, 'the', tc)

    res = {
        'phase': 2878,
        'gen': 2,
        'prereg': PREREG,
        'n_homograph_excluded': n_excl,
        'axis_pair_counts': {a: len(eff_axes[a]) for a in AXES},
        'VG1': {'syntax_pass': syn_ok, 'translation_pass': tr_ok,
                'verdict': vg1},
        'VG2b_per_axis': {a: round(v, 4) for a, v in vg2b.items()},
        'surviving_axes': surviving,
        'syntax_surviving': syn_surv,
        'translation_surviving': tr_surv,
        'VG2a': {'max_cross_axis_cos': round(max_off, 4),
                 'gate': 0.488, 'verdict': vg2a},
        'VG3': {'worst_norm_ratio': round(worst_ratio, 4),
                'worst_axis': worst_axis, 'gate': 3.0, 'verdict': vg3},
        'syn_legal': syn_legal,
        'trans_legal': trans_legal,
        'vocab_legal': verdict,
        'n_words_v2': n_words,
        'n_pairs_v2': sum(len(pair_dirs[a]) for a in surviving),
        'final_verdict': 'syn_legal=%s trans_legal=%s (%d syntax + %d '
                         'translation axes, %d words, %d pairs)'
                         % (syn_legal, trans_legal, n_syn, n_tr,
                            n_words,
                            sum(len(pair_dirs[a]) for a in surviving)),
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    axis_words_slim = {a: axis_words[a] for a in surviving}
    np.savez_compressed(
        os.path.join(OUT, 'syntax_trans_vocab.npz'),
        targets=np.array(json.dumps(axis_words_slim), dtype=object),
        pairs_json=np.array(json.dumps(
            {a: AXES[a] for a in surviving}), dtype=object),
        axes=np.array(surviving, dtype=object),
        target_list=np.array(['%s:%s' % t for t in target_list],
                             dtype=object),
        dW_unit=dW.astype(np.float32),
        null_tids=np.array([null_tids[i] for i in range(n_words)],
                           dtype=np.int64),
        func_tid=np.array(func_tid, dtype=np.int64),
        tid_map=np.array(json.dumps(tc), dtype=object),
        n_words=np.array(n_words, dtype=np.int64))

    log('==== VERDICT ====')
    log(json.dumps({k: res[k] for k in
                    ('VG1', 'VG2b_per_axis', 'surviving_axes', 'VG2a',
                     'VG3', 'syn_legal', 'trans_legal',
                     'final_verdict')},
                   ensure_ascii=False))
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
