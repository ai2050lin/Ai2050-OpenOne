# -*- coding: utf-8 -*-
"""Phase 2919: multi-axis direction families (speed/size/moisture)
under the 2886 caliber + axis-collinearity audit.

Why (roadmap 2919, lpf_multiaxis_gating_roadmap_v1.md): the
multi-axis event atlas (2920) needs per-layer direction families
for non-language axes. The language family exists (2886 dirs,
consumed by 2913-2918). This phase builds speed/size/moisture
families with the SAME protocol and audits collinearity BEFORE
any multi-axis atlas is attempted.

Mode: forward, qwen3-4b (load_native full GPU), 200 frozen
sentences, last-token hidden state per layer (37 = embed + 36
blocks, 2886 S2 caliber), rows [0,36) used (input-to-block
convention, 2917 caliber).

Axis construction (frozen):
  lang   - the 40 en/fr parallel pairs of 2886 VERBATIM (80
           sentences, class 0=en, 1=fr). Anchors only; no new
           construction.
  speed  - 20 same-subject pairs "The {s} is fast." (cls 0) /
           "The {s} is slow." (cls 1).
  size   - 20 same-subject pairs "The {s} is huge." / "tiny.".
  moist  - 20 same-subject pairs "The {s} is wet." / "dry.".
  Same-subject design: each subject appears once per class, so
  the subject contribution CANCELS in the class-diff of means;
  the residual axis direction is the attribute pole contrast
  (template interaction included by design).
  Pole convention (registered): dirs = unit_per_layer of
  mean(cls 0 = HIGH pole: fast/huge/wet) - mean(cls 1 = LOW
  pole: slow/tiny/dry); ties to the 2918 polarity pi(e).

Anchors (frozen):
  a1 fresh S[0:80] vs 2886 npz S_last: max rel < 1e-5
     (2914/2917 cross-run scale ~3e-8).
  a2 dirs_lang rows [0,36) recomputed from fresh S vs the
     2917-consumed derivation from the 2886 npz: max rel < 1e-5.

Probes (frozen):
  P1 quality: per axis, per layer li in [0,36): LOO
     nearest-centroid (cosine, 2886 S4 caliber) on the axis's
     own sentences; axis_ready iff best-layer acc >= 0.70
     (chance 0.5; lang reference only).
  P2 collinearity: |cos| between unit dirs of all 6 axis pairs
     per layer; non_collinear iff global max < 0.90; argmax
     pair/layer reported. Lang-attr vs attr-attr split
     descriptive.
  P3 descriptive: raw class-diff norm per layer per axis.
  P4 descriptive: probe curves stored for the atlas design.

Adjudication (frozen):
  anchor fail => anchor_fail_all_void;
  n_ready == 3 AND non_collinear => multiaxis_families_ready;
  n_ready == 3 AND not non_collinear => multiaxis_families_collinear;
  1 <= n_ready <= 2 => multiaxis_families_partial;
  n_ready == 0 => multiaxis_families_failed.

Output: phase2919/multiaxis_direction_families/.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2886 = os.path.join(BASE, 'phase2886', 'hourglass_cka',
                        'hourglass_cka.npz')
OUT = os.path.join(BASE, 'phase2919',
                   'multiaxis_direction_families')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2919_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
AXIS_NAMES = ['lang', 'speed', 'size', 'moist']
READY_ACC = 0.70
COS_MAX = 0.90

PAIRS_LANG = [
    ('The cat is black.', 'Le chat est noir.'),
    ('The dog is big.', 'Le chien est grand.'),
    ('The house is small.', 'La maison est petite.'),
    ('The book is new.', 'Le livre est nouveau.'),
    ('The night is dark.', 'La nuit est sombre.'),
    ('The sun is bright.', 'Le soleil est brillant.'),
    ('The milk is cold.', 'Le lait est froid.'),
    ('The water is clear.', "L'eau est claire."),
    ('The tree is tall.', "L'arbre est grand."),
    ('The moon is white.', 'La lune est blanche.'),
    ('The sea is deep.', 'La mer est profonde.'),
    ('The sky is blue.', 'Le ciel est bleu.'),
    ('The flower is red.', 'La fleur est rouge.'),
    ('The city is loud.', 'La ville est bruyante.'),
    ('The war was long.', 'La guerre était longue.'),
    ('The king is old.', 'Le roi est vieux.'),
    ('The woman is kind.', 'La femme est gentille.'),
    ('The man is tall.', "L'homme est grand."),
    ('The mother is here.', 'La mère est ici.'),
    ('The father is strong.', 'Le père est fort.'),
    ('The bird sings.', "L'oiseau chante."),
    ('The fish swims.', 'Le poisson nage.'),
    ('The bread is warm.', 'Le pain est chaud.'),
    ('The wine is good.', 'Le vin est bon.'),
    ('The star shines.', "L'étoile brille."),
    ('The fire is hot.', 'Le feu est chaud.'),
    ('The mountain is high.', 'La montagne est haute.'),
    ('The snow is cold.', 'La neige est froide.'),
    ('The wolf howls.', 'Le loup hurle.'),
    ('The summer is short.', "L'été est court."),
    ('The table is round.', 'La table est ronde.'),
    ('The door is open.', 'La porte est ouverte.'),
    ('The horse runs.', 'Le cheval court.'),
    ('The cheese is French.', 'Le fromage est français.'),
    ('The egg is fresh.', "L'œuf est frais."),
    ('The bed is soft.', 'Le lit est doux.'),
    ('The light is warm.', 'La lumière est chaude.'),
    ('The earth is round.', 'La terre est ronde.'),
    ('The moon rises.', 'La lune se lève.'),
    ('The child sleeps.', "L'enfant dort."),
]

SPEED_SUBJ = ['cheetah', 'rabbit', 'horse', 'train', 'car',
              'wind', 'river', 'stream', 'eagle', 'runner',
              'dog', 'cyclist', 'clock', 'glacier', 'turtle',
              'snail', 'sloth', 'traffic', 'lava', 'cloud']
SIZE_SUBJ = ['mountain', 'elephant', 'whale', 'building',
             'tree', 'hill', 'tower', 'bridge', 'planet',
             'door', 'car', 'cat', 'mouse', 'book', 'coin',
             'pebble', 'ant', 'grain', 'atom', 'cell']
MOIST_SUBJ = ['towel', 'sponge', 'soil', 'cloth', 'hair',
              'sand', 'leaf', 'wood', 'paper', 'stone',
              'skin', 'grass', 'clay', 'brick', 'road',
              'wall', 'rope', 'field', 'boots', 'jacket']
POLES = {'speed': ('fast', 'slow'), 'size': ('huge', 'tiny'),
         'moist': ('wet', 'dry')}

PREREG = {
    'mode': 'forward 200 frozen sentences, 2886 last-token '
            'caliber, rows [0,36) input-to-block convention',
    'question': 'roadmap 2919: build per-layer direction '
                'families for speed/size/moisture under the 2886 '
                'caliber and audit axis collinearity BEFORE the '
                'multi-axis atlas (2920)',
    'axis_design': 'lang = 2886 pairs verbatim (anchor only); '
                   'speed/size/moist = 20 same-subject pairs '
                   'each, template "The {s} is {attr}." with '
                   'pole pairs fast/slow, huge/tiny, wet/dry; '
                   'same-subject => subject contribution cancels '
                   'in class-diff of means; pole convention: '
                   'dirs = unit(mean HIGH - mean LOW) per layer',
    'anchors': {
        'a1': 'fresh S[0:80] vs 2886 npz S_last max rel < 1e-5',
        'a2': 'dirs_lang rows [0,36) vs 2917-consumed derivation '
              'from 2886 npz max rel < 1e-5',
    },
    'probes': {
        'P1': 'LOO nearest-centroid (cosine, 2886 S4) per axis '
              'per layer; axis_ready iff best acc >= 0.70',
        'P2': 'axis-pair |cos| per layer (6 pairs x 36); '
              'non_collinear iff global max < 0.90',
        'P3': 'raw diff norm per layer per axis (descriptive)',
        'P4': 'probe curves stored (descriptive)',
    },
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'n_ready==3 AND non_collinear => '
               'multiaxis_families_ready; n_ready==3 AND not '
               'non_collinear => multiaxis_families_collinear; '
               '1<=n_ready<=2 => multiaxis_families_partial; '
               'n_ready==0 => multiaxis_families_failed',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def unit(v):
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-30)


def loo_nc_acc(X, lab):
    n = len(lab)
    ok = 0
    for i in range(n):
        m = np.arange(n) != i
        c0 = X[m & (lab == 0)].mean(0)
        c1 = X[m & (lab == 1)].mean(0)
        c0 = c0 / max(float(np.linalg.norm(c0)), 1e-30)
        c1 = c1 / max(float(np.linalg.norm(c1)), 1e-30)
        x = X[i] / max(float(np.linalg.norm(X[i])), 1e-30)
        ok += int(float(x @ c1) > float(x @ c0)) if lab[i] == 1 \
            else int(float(x @ c0) > float(x @ c1))
    return ok / n


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    sents = []
    axis_id = []
    cls = []
    for en, fr in PAIRS_LANG:
        sents.append(en)
        axis_id.append(0)
        cls.append(0)
        sents.append(fr)
        axis_id.append(0)
        cls.append(1)
    for ai, subj in ((1, SPEED_SUBJ), (2, SIZE_SUBJ),
                     (3, MOIST_SUBJ)):
        hi, lo = POLES[AXIS_NAMES[ai]]
        for s in subj:
            sents.append('The %s is %s.' % (s, hi))
            axis_id.append(ai)
            cls.append(0)
            sents.append('The %s is %s.' % (s, lo))
            axis_id.append(ai)
            cls.append(1)
    axis_id = np.array(axis_id)
    cls = np.array(cls)
    n_sent = len(sents)
    assert n_sent == 200

    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2919,
                   'name': 'multiaxis_direction_families',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2886': sha8(SRC_2886)},
                   'model': 'qwen3-4b', 'n_sentences': n_sent,
                   'n_layers': NL, 'ready_acc': READY_ACC,
                   'cos_max': COS_MAX,
                   'sentences': sents, 'axis_id':
                       axis_id.tolist(), 'cls': cls.tolist(),
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log = lambda m: (lines.append(m), print(m, flush=True))
    log('execution.json frozen (%d sentences)' % n_sent)

    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import load_native

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    model, _ = load_native('qwen4')
    model.eval()
    log('model loaded (load_native full GPU)')

    S = np.zeros((n_sent, NL + 1, 2560), dtype=np.float32)
    with torch.no_grad():
        for si, s in enumerate(sents):
            ids = tok(s, return_tensors='pt')['input_ids'] \
                .to('cuda')
            out = model(ids, output_hidden_states=True)
            hs = out.hidden_states
            S[si] = np.stack(
                [h[0, -1].float().cpu().numpy() for h in hs]) \
                .astype(np.float32)
            if (si + 1) % 40 == 0:
                log('sentences [%d/%d]' % (si + 1, n_sent))
    log('S %s' % (S.shape,))

    # ---------- anchors ----------
    z86 = np.load(SRC_2886, allow_pickle=True)
    S86 = z86['S_last'].astype(np.float64)
    lab86 = np.asarray(z86['labels']).astype(int)
    Sa = S[:80].astype(np.float64)
    aid80 = axis_id[:80]
    cls80 = cls[:80]
    rel_a1 = float(np.abs(Sa - S86).max()
                   / max(float(np.abs(S86).max()), 1e-30))
    a1_ok = bool(rel_a1 < 1e-5)
    diffs_ref = (S86[lab86 == 0].mean(0)
                 - S86[lab86 == 1].mean(0))
    lang_hi = Sa[(aid80 == 0) & (cls80 == 0)]
    lang_lo = Sa[(aid80 == 0) & (cls80 == 1)]
    dirs_mine = np.zeros((NL, 2560))
    for li in range(NL):
        dirs_mine[li] = unit(lang_hi.mean(0)[li]
                             - lang_lo.mean(0)[li])
    dirs_ref = np.stack([unit(diffs_ref[li])
                         for li in range(NL)])
    rel_a2 = float(np.abs(dirs_mine - dirs_ref).max())
    a2_ok = bool(rel_a2 < 1e-5)
    anchor_ok = bool(a1_ok and a2_ok)
    log('a1 rel vs 2886 S_last %.2e ok=%s | a2 dirs rel %.2e '
        'ok=%s' % (rel_a1, a1_ok, rel_a2, a2_ok))

    verdict = None
    p1 = p2 = p3 = p4 = None
    save = {}
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- dirs for all axes ----------
        dirs_all = np.zeros((4, NL, 2560))
        dirs_all[0] = dirs_mine
        for ai in (1, 2, 3):
            hi = S[(axis_id == ai) & (cls == 0)].astype(
                np.float64)
            lo = S[(axis_id == ai) & (cls == 1)].astype(
                np.float64)
            for li in range(NL):
                dirs_all[ai, li] = unit(
                    hi.mean(0)[li] - lo.mean(0)[li])

        # ---------- P1 quality probe ----------
        probe = np.zeros((4, NL))
        for ai in range(4):
            m = axis_id == ai
            X = S[m].astype(np.float64)
            lab_a = cls[m]
            for li in range(NL):
                probe[ai, li] = loo_nc_acc(X[:, li, :], lab_a)
        best = [(AXIS_NAMES[ai], int(np.argmax(probe[ai])),
                 round(float(probe[ai].max()), 4),
                 bool(probe[ai].max() >= READY_ACC))
                for ai in range(4)]
        # verdict counts the 3 ATTRIBUTE axes only (lang is the
        # anchor family, ready by construction via a1/a2)
        n_ready = sum(1 for b in best[1:] if b[3])
        p1 = {'ready_acc_threshold': READY_ACC,
              'per_axis_best': best,
              'lang_reference': best[0],
              'n_ready_attr': n_ready}
        log('P1 per-axis best LOO acc: %s | n_ready_attr %d'
            % (best, n_ready))

        # ---------- P2 collinearity ----------
        pair_ids = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3),
                    (2, 3)]
        cos_m = np.zeros((len(pair_ids), NL))
        for pi, (a, b) in enumerate(pair_ids):
            for li in range(NL):
                cos_m[pi, li] = abs(float(
                    dirs_all[a, li] @ dirs_all[b, li]))
        gmax = float(cos_m.max())
        pi_max, li_max = np.unravel_index(np.argmax(cos_m),
                                          cos_m.shape)
        non_collinear = bool(gmax < COS_MAX)
        attr_attr = [i for i, (a, b) in enumerate(pair_ids)
                     if a >= 1 and b >= 1]
        p2 = {'pair_names': ['%s-%s' % (AXIS_NAMES[a],
                                        AXIS_NAMES[b])
                             for a, b in pair_ids],
              'global_max': round(gmax, 4),
              'argmax_pair': p2_name(pair_ids, pi_max),
              'argmax_layer': int(li_max),
              'attr_attr_max': round(
                  float(cos_m[attr_attr].max()), 4),
              'lang_attr_max': round(
                  float(cos_m[[i for i in range(3)]].max()),
                  4),
              'non_collinear': non_collinear,
              'cos_max_threshold': COS_MAX,
              'cos_matrix': [[round(float(x), 4)
                              for x in row] for row in cos_m]}
        log('P2 collinearity: global max %.4f at %s L%d | '
            'attr-attr %.4f lang-attr %.4f | non_collinear %s'
            % (gmax, p2['argmax_pair'], li_max,
               p2['attr_attr_max'], p2['lang_attr_max'],
               non_collinear))

        # ---------- P3/P4 descriptive ----------
        diff_norm = np.zeros((4, NL))
        for ai in range(4):
            m = axis_id == ai
            hi = S[m & (cls == 0)].astype(np.float64)
            lo = S[m & (cls == 1)].astype(np.float64)
            for li in range(NL):
                diff_norm[ai, li] = float(np.linalg.norm(
                    hi.mean(0)[li] - lo.mean(0)[li]))
        p3 = {'diff_norm': {AXIS_NAMES[ai]:
                            [round(float(x), 3)
                             for x in diff_norm[ai]]
                            for ai in range(4)}}
        p4 = {'probe_curves': {AXIS_NAMES[ai]:
                               [round(float(x), 4)
                                for x in probe[ai]]
                               for ai in range(4)}}
        log('P3 diff norms: %s'
            % {AXIS_NAMES[ai]:
               (round(float(diff_norm[ai].max()), 2),
                int(np.argmax(diff_norm[ai])))
               for ai in range(4)})

        # ---------- verdict ----------
        if n_ready == 3 and non_collinear:
            verdict = 'multiaxis_families_ready'
        elif n_ready == 3:
            verdict = 'multiaxis_families_collinear'
        elif n_ready >= 1:
            verdict = 'multiaxis_families_partial'
        else:
            verdict = 'multiaxis_families_failed'
        save = {'dirs_all': dirs_all.astype(np.float32),
                'probe_curves': probe.astype(np.float32),
                'cos_matrix': cos_m.astype(np.float32),
                'pair_names': np.array(
                    p2['pair_names'], dtype=object),
                'diff_norm': diff_norm.astype(np.float32),
                'axis_id': axis_id, 'cls': cls,
                'sentences': np.array(sents, dtype=object),
                'axis_names': np.array(AXIS_NAMES,
                                       dtype=object)}

    log('==== VERDICT: %s ====' % verdict)
    res = {'phase': 2919, 'model': 'qwen3-4b', 'prereg': PREREG,
           'anchors': {'a1_rel_vs_2886': float('%.3e' % rel_a1),
                       'a1_ok': a1_ok,
                       'a2_dirs_rel': float('%.3e' % rel_a2),
                       'a2_ok': a2_ok, 'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(
            os.path.join(OUT, 'multiaxis_families.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2919 verdict=%s' % verdict, flush=True)


def p2_name(pair_ids, idx):
    a, b = pair_ids[int(idx)]
    return '%s-%s' % (AXIS_NAMES[a], AXIS_NAMES[b])


if __name__ == '__main__':
    main()
