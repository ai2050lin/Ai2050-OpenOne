# -*- coding: utf-8 -*-
"""Phase 2900: glm4 mlp eigen-cell multi-draw margin-first
adjudication (last un-multi-drawn positive cell in the L14
spectrum).

M2893 (SEED=2893) and M2897 (SEED=2897) both found glm4 mlp
eigen acc-positive (0.6923 / 0.6795 > p95 ~0.63-0.64) but
margin-negative (0.0313 < 0.0349 / 0.0198 < 0.0402) - a
threshold-adjacent cell never adjudicated under margin-first
multi-draw criteria (G_multi_draw_lesson).  This phase runs the
glm4 mlp channel with three conditions x 8 null-draws:

  stale: inject lang_dir (li=20), project lang_dir
         (EXPECTED-NEGATIVE internal control: M2890/M2897
         margin-absent; M2898 showed glm4 attn stale is
         null-draw fragile)
  eigen: inject dir_q(li), project dir_q(li)  (PRIMARY cell)
  orth:  inject unit(dir_q(li)-(dir_q(li).lang_dir)lang_dir),
         project it (M2897 borderline: acc 0.6282 vs p95 0.6288)

Window W=[28,40); words/conds verbatim 2890 (78 words);
directions zero-forward from 2890 npz.  Per draw s in SEEDS:
null_tids from rng(s) (2893/2897 identical exclusion rules ->
seeds 2893/2897 reproduce M2893/M2897 null sets exactly),
200 label perms from rng2(s).

v1 (per draw): mlp recompute rel err < 1e-6, else that draw
void.  No positive anchor gate exists for glm4 mlp (no known
robust-positive condition on this channel); procedure stability
is guarded by v1 determinism plus exact reproduction of
M2893/M2897 eigen accs at their seeds (recorded as
reproduction_check).

Frozen decision rule (on valid draws, eigen):
  acc_flag_rate >= 0.8 AND margin_flag_rate >= 0.5
      => glm4_mlp_eigen_robust
  margin_flag_rate == 0
      => glm4_mlp_eigen_acc_only_fragile (M2893/M2897
         downgraded: acc-flag artifact-prone, margin never
         carries)
  0 < margin_flag_rate < 0.5, or acc_flag_rate < 0.8 with
  margin_flag_rate >= 0.5
      => glm4_mlp_eigen_mixed (rates recorded)
stale/orth: descriptive rates only (control conditions).

SEEDS = [2893, 2897, 2901, 2902, 2903, 2904, 2905, 2906]
(includes both previously-run draws for within-run
comparability).
Output: phase2900/glm4_mlp_eigen_robustness/.
"""
import hashlib
import io
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2890 = os.path.join(BASE, 'phase2890', 'language_axis_glm4',
                        'language_axis_glm4.npz')
OUT = os.path.join(BASE, 'phase2900', 'glm4_mlp_eigen_robustness')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\glm4-9b-chat-hf'
SEEDS = [2893, 2897, 2901, 2902, 2903, 2904, 2905, 2906]
N_NULL = 200
EPS = 1.0
W_LO, W_HI = 28, 40

PAIRS = [
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

PREREG = {
    'window': 'W=[28,40) frozen (same as 2890-2893/2897-2898)',
    'sources': 'dirs/lang_dir/words/labels zero-forward from 2890 '
               'npz (sha recorded); conds same/func(the)/null '
               'verbatim 2890 protocol; null-draw rules identical '
               'to 2893/2897 (seeds 2893/2897 reproduce their '
               'null sets)',
    'design': 'N_SEEDS=%d draws; per draw: null_tids from rng(s) '
              'with word-tid exclusion, 200 label perms from '
              'rng2(s); mlp channel only; conditions '
              'stale(lang_dir, EXPECTED-NEGATIVE control) + '
              'eigen(dir_q(li), PRIMARY) + orth(unit(dir_q-'
              '(dir_q.lang_dir)lang_dir), descriptive)'
              % len(SEEDS),
    'v1': 'mlp recompute rel err < 1e-6 per draw, else that draw '
          'void; no positive anchor gate exists for glm4 mlp - '
          'procedure stability guarded by v1 + exact '
          'reproduction_check of M2893/M2897 eigen accs at their '
          'seeds',
    'decision': 'eigen: acc_flag_rate >= 0.8 AND margin_flag_rate '
                '>= 0.5 => glm4_mlp_eigen_robust; '
                'margin_flag_rate == 0 => '
                'glm4_mlp_eigen_acc_only_fragile; else '
                'glm4_mlp_eigen_mixed (rates recorded). '
                'stale/orth: descriptive rates only',
    'seeds': SEEDS,
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def zmat(M):
    mu = M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, keepdims=True)
    Z = (M - mu) / np.maximum(sd, 1e-30)
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(n, 1e-30)


def loo_acc(C, lab):
    S = C @ C.T
    np.fill_diagonal(S, -2.0)
    nn = S.argmax(axis=1)
    return float(np.mean(lab[nn] == lab))


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.monotonic()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2900, 'name': 'glm4_mlp_eigen_robustness',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s_last_2890': sha8(SRC_2890)},
                   'model': 'glm4-9b-chat-hf',
                   'n_pairs': len(PAIRS),
                   'prereg': PREREG, 'seeds': SEEDS, 'eps': EPS,
                   'window': [W_LO, W_HI]},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    # ---------- zero-forward directions ----------
    z90 = np.load(SRC_2890, allow_pickle=True)
    S_last = z90['S_last'].astype(np.float64)
    lang_dir = z90['lang_dir'].astype(np.float64)
    words = [tuple(str(w).split(':')) for w in z90['words']]
    lab_lang = z90['labels_lang']
    lab_concept = z90['labels_concept']
    n_words = len(words)
    lab_sent = np.array([i % 2 for i in range(2 * len(PAIRS))])
    assert S_last.shape[0] == 2 * len(PAIRS)

    mean_en = S_last[lab_sent == 0].mean(0)
    mean_fr = S_last[lab_sent == 1].mean(0)
    dirs, orth_dirs = {}, {}
    for li in range(W_LO, W_HI):
        d = unit(mean_en[li] - mean_fr[li])
        dirs[li] = d
        orth_dirs[li] = unit(d - float(d @ lang_dir) * lang_dir)

    # ---------- model + hooks (2888/2898 glm4 loading) ----------
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            tc[t] = int(ids[0])
        return tc[t]

    tid_map = {}
    for lang, key, w in words:
        tid_map[w] = tid(w)
    func_tid = tid('the')

    cfg = json.load(io.open(os.path.join(MD, 'config.json'),
                            encoding='utf-8'))
    dm = {'model.embed_tokens': 0, 'model.norm': 0, 'lm_head': 0}
    for li_ in range(W_HI):
        dm['model.layers.%d' % li_] = 0 if W_LO <= li_ < W_HI \
            else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(
        MD, local_files_only=True, device_map=dm)
    model.eval()
    layers = model.model.layers
    log('model loaded: window [%d,%d) on cuda' % (W_LO, W_HI))
    for li_ in range(W_LO, W_HI):
        for p in layers[li_].mlp.parameters():
            assert p.device.type != 'meta'

    cap = {'mlpin': {}, 'mlpout': {}}
    handles = []

    def pre_hook(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            cap['mlpin'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())   # [1, seq, d]
        return h

    def out_hook(li):
        def h(module, args, output):
            o = output[0] if isinstance(output, tuple) else output
            if o.dim() < 2:
                return
            cap['mlpout'].setdefault(li, []).append(
                o.detach().float().cpu().numpy())   # [1, seq, d]
        return h

    for li in range(W_LO, W_HI):
        handles.append(layers[li].mlp.register_forward_pre_hook(
            pre_hook(li), with_kwargs=True))
        handles.append(layers[li].mlp.register_forward_hook(
            out_hook(li)))

    def clear_cap():
        for dd in cap:
            for li in cap[dd]:
                del cap[dd][li][:]

    def forward2(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return ({li: cap['mlpin'][li][0] for li in cap['mlpin']},
                {li: cap['mlpout'][li][0] for li in cap['mlpout']})

    def mlp_call(li, x):
        out = layers[li].mlp(torch.tensor(
            x, device='cuda', dtype=torch.bfloat16))
        if isinstance(out, tuple):
            out = out[0]
        return out[0, 1].detach().float().cpu().numpy()

    def same_ctx(i):
        lang = words[i][0]
        cands = [j for j in range(n_words)
                 if words[j][0] == lang and j != i]
        return min(cands, key=lambda j: tid_map[words[j][2]])

    n_win = W_HI - W_LO
    vocab_size = int(cfg.get('vocab_size'))
    word_tids = set(tid_map.values())

    def draw_null_tids(seed):
        rng = np.random.default_rng(seed)
        out = []
        while len(out) < n_words:
            r = int(rng.integers(0, vocab_size))
            if r not in word_tids and r > 0:
                out.append(r)
        return out

    def margin_of(Sm, pl):
        n = len(pl)
        sm = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                sm[i, j] = (pl[i] == pl[j]) and i != j
        df = (~np.eye(n, dtype=bool)) & (~sm)
        return float(Sm[sm].mean() - Sm[df].mean())

    def stats_of(B, seed):
        C = zmat(B)
        acc = loo_acc(C, lab_lang)
        U = B / np.maximum(
            np.linalg.norm(B, axis=1, keepdims=True), 1e-30)
        Sm = U @ U.T
        n = len(lab_lang)
        same_m = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                same_m[i, j] = (lab_lang[i] == lab_lang[j]) and \
                    i != j
        off = ~np.eye(n, dtype=bool) & ~same_m
        m_obs = float(Sm[same_m].mean() - Sm[off].mean())
        rng2 = np.random.default_rng(seed)
        perms = [rng2.permutation(lab_lang) for _ in range(N_NULL)]
        null_acc = [loo_acc(C, pl) for pl in perms]
        null_m = [margin_of(Sm, pl) for pl in perms]
        return {'acc': round(acc, 4),
                'p95': round(float(np.percentile(null_acc, 95)), 4),
                'flag_acc': bool(acc > np.percentile(null_acc, 95)),
                'margin': round(m_obs, 4),
                'margin_p95': round(
                    float(np.percentile(null_m, 95)), 4),
                'flag_margin': bool(
                    m_obs > np.percentile(null_m, 95)),
                'concept_acc': round(loo_acc(C, lab_concept), 4)}

    # capture same/func once (seed-independent)
    log('capturing same/func contexts once...')
    base_caps = {}
    for cn in ('same', 'func'):
        per_word = {}
        for i, (_, _, w) in enumerate(words):
            w_tid = tid_map[w]
            if cn == 'same':
                ctx_tid = tid_map[words[same_ctx(i)][2]]
            else:
                ctx_tid = func_tid
            per_word[i] = forward2([ctx_tid, w_tid])
        base_caps[cn] = per_word
    log('base captures done')

    RTYPE_DIRS = ('stale', 'eigen', 'orth')
    results = {rt: [] for rt in RTYPE_DIRS}
    v1_by_draw = {}
    B_store = {rt: {} for rt in RTYPE_DIRS}
    for s in SEEDS:
        td = time.monotonic()
        null_tids = draw_null_tids(s)
        g = {rt: {cn: np.zeros((n_words, n_win)) for cn in
                  ('same', 'func', 'null')} for rt in RTYPE_DIRS}
        v1_err = 0.0
        for i, (_, _, w) in enumerate(words):
            w_tid = tid_map[w]
            conds = {'same': base_caps['same'][i],
                     'func': base_caps['func'][i],
                     'null': forward2([null_tids[i], w_tid])}
            for cn, (mlpin, mlpout) in conds.items():
                for q, li in enumerate(range(W_LO, W_HI)):
                    x0 = mlpin[li]          # [1, seq, d]
                    ref = mlp_call(li, x0)  # pos-1 output
                    if cn == 'same' and q == 0:
                        ref2 = mlp_call(li, x0)
                        v1_err = max(v1_err, float(np.linalg.norm(
                            ref - ref2))
                            / max(float(np.linalg.norm(ref)), 1e-30))
                    for rt in RTYPE_DIRS:
                        d_rt = lang_dir if rt == 'stale' \
                            else (dirs[li] if rt == 'eigen'
                                  else orth_dirs[li])
                        xp = x0.copy()
                        xp[0, 1] = xp[0, 1] + EPS * d_rt
                        g[rt][cn][i, q] = float(
                            (mlp_call(li, xp) - ref) @ d_rt) / EPS
        v1_by_draw[s] = float('%.3e' % v1_err)
        if v1_err >= 1e-6:
            log('draw seed=%d v1=%.3e VOID' % (s, v1_err))
            continue
        for rt in RTYPE_DIRS:
            B = g[rt]['same'] - 0.5 * (g[rt]['func'] + g[rt]['null'])
            B_store[rt][s] = B.astype(np.float32)
            st = stats_of(B, s)
            results[rt].append(dict(st, seed=s))
            log('seed=%d %s: acc=%.4f p95=%.4f flag=%s margin=%.4f '
                'mp95=%.4f mflag=%s (%.0fs)'
                % (s, rt, st['acc'], st['p95'], st['flag_acc'],
                   st['margin'], st['margin_p95'],
                   st['flag_margin'], time.monotonic() - td))

    n_valid = len(results['eigen'])
    rates = {}
    for rt in RTYPE_DIRS:
        rs = results[rt]
        rates[rt] = {
            'n_valid': n_valid,
            'acc_flag_rate': round(
                sum(r['flag_acc'] for r in rs) / max(n_valid, 1), 4),
            'margin_flag_rate': round(
                sum(r['flag_margin'] for r in rs) / max(n_valid, 1),
                4),
            'acc_values': [r['acc'] for r in rs],
            'margin_values': [r['margin'] for r in rs]}

    # reproduction check: eigen accs at historical seeds
    repro = {}
    for r in results['eigen']:
        if r['seed'] == 2893:
            repro['s2893_vs_M2893'] = bool(
                abs(r['acc'] - 0.6923) < 5e-4)
        if r['seed'] == 2897:
            repro['s2897_vs_M2897'] = bool(
                abs(r['acc'] - 0.6795) < 5e-4)
    log('reproduction_check: %s' % repro)

    er = rates['eigen']
    if er['acc_flag_rate'] >= 0.8 and er['margin_flag_rate'] >= 0.5:
        verdict = 'glm4_mlp_eigen_robust'
    elif er['margin_flag_rate'] == 0.0:
        verdict = 'glm4_mlp_eigen_acc_only_fragile'
    else:
        verdict = 'glm4_mlp_eigen_mixed'

    res = {
        'phase': 2900, 'model': 'glm4-9b-chat-hf', 'prereg': PREREG,
        'window': [W_LO, W_HI],
        'v1_by_draw': v1_by_draw,
        'reproduction_check': repro,
        'rates': rates, 'per_draw': results,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'glm4_mlp_eigen_robustness.npz'),
        **{('B_%s_s%d' % (rt, s)): B_store[rt][s]
           for rt in RTYPE_DIRS for s in B_store[rt]},
        labels_lang=lab_lang, labels_concept=lab_concept,
        words=np.array(['%s:%s:%s' % w for w in words], dtype=object))
    log('==== VERDICT: %s ====' % verdict)
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
