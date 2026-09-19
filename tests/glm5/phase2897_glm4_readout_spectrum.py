# -*- coding: utf-8 -*-
"""Phase 2897: GLM4 readout-type spectrum (three-condition dual
channel injection, closes the 2896 cross-model loop).

2896 (qwen): subspace_readout_tolerant - stale/eigen/orth all
positive.  GLM4 side so far: stale (lang_dir li=20) negative in
both channels (M2890 mlp / M2891 attn); eigen (dir_q(li)) positive
in both (M2893).  Missing cell: ORTH.  If orth is negative, GLM4
is direction_matched_strict; if positive, the readout types form
a partial-tolerance spectrum.

Stage 1 (zero forward, 2890 npz sha 74933303):
  dir_q(li) = unit(mean_en S_last[:,li] - mean_fr S_last[:,li])
  for li in W=[28,40); lang_dir from same npz (2890 frozen).
  orth(li) = unit(dir_q(li) - (dir_q(li).lang_dir) lang_dir)
  cos curves recorded (descriptive).

Stage 2 (three-condition dual-channel injection, 2893 machinery
verbatim, 78 words/conds verbatim 2890):
  For rt in {stale, eigen, orth} with direction d_rt(li):
    g[ch][rt][cn][i,q] = [(module(x+eps*d_rt(li)) - ref) . d_rt(li)]/eps
    module in {mlp, self_attn}; pos 1; eps=1.0.
    B_rt = g_rt(same) - 0.5(g_rt(func)+g_rt(null)).

  v1: module recompute rel err < 1e-6 per channel (shape-matched
  capture, 2890-2893 convention), else all void.
  A_m/A_a per rt: acc(loo-NN zmat B_rt, lab_lang) > null p95
  (200 perms, SEED=2897).
  Concept control per rt/channel (expect ~null).

  Verdict (frozen):
    orth positive (either channel) & stale negative =>
        partial_tolerance_spectrum_glm4
    orth positive & stale positive =>
        broad_tolerance_glm4 (conflict with M2890/M2891 - errata)
    orth negative & eigen positive & stale negative =>
        direction_matched_strict_glm4
    nothing positive => readout_absent_window (conflict with
        M2893 - errata)

SEED=2897.  Output: phase2897/glm4_readout_spectrum/.
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
OUT = os.path.join(BASE, 'phase2897', 'glm4_readout_spectrum')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\glm4-9b-chat-hf'
SEED = 2897
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
    'window': 'W=[28,40) frozen (same deep window as 2890-2893)',
    'sources': 'dirs/lang_dir/words/labels zero-forward from 2890 '
               'npz (sha recorded); conds same/func(the)/null '
               'verbatim 2890 protocol',
    'dirs': 'dir_q(li)=unit(mean_en S_last[:,li]-mean_fr '
            'S_last[:,li]); orth(li)=unit(dir_q(li)-'
            '(dir_q(li).lang_dir)lang_dir); cos curves recorded '
            '(descriptive)',
    'injection': 'three per-layer injections at pos 1 into mlp and '
                 'self_attn: stale(lang_dir, proj lang_dir), '
                 'eigen(dir_q(li), proj dir_q(li)), orth(orth(li), '
                 'proj orth(li)); eps=1.0; B = g(same)-0.5(g(func)+'
                 'g(null)) per response type and channel',
    'v1': 'module recompute rel err < 1e-6 per channel '
          '(shape-matched capture, 2890-2893 convention), else '
          'all void',
    'A': 'acc(loo-NN zmat B_rt, lab_lang) > null p95 (200 perms, '
         'SEED=2897) per channel; concept control recorded',
    'verdict': 'orth+ & stale- => partial_tolerance_spectrum_glm4; '
               'orth+ & stale+ => broad_tolerance_glm4 (errata '
               'vs M2890/M2891); orth- & eigen+ & stale- => '
               'direction_matched_strict_glm4; none => '
               'readout_absent_window (errata vs M2893)',
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
        json.dump({'phase': 2897, 'name': 'glm4_readout_spectrum',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s_last_2890': sha8(SRC_2890)},
                   'model': 'glm4-9b-chat-hf',
                   'n_pairs': len(PAIRS),
                   'prereg': PREREG, 'seed': SEED, 'eps': EPS,
                   'window': [W_LO, W_HI]},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    # ---------- Stage 1: zero-forward directions ----------
    z90 = np.load(SRC_2890, allow_pickle=True)
    S_last = z90['S_last'].astype(np.float64)   # (80, 41, 4096)
    lang_dir = z90['lang_dir'].astype(np.float64)
    words = [tuple(str(w).split(':')) for w in z90['words']]
    lab_lang = z90['labels_lang']
    lab_concept = z90['labels_concept']
    n_words = len(words)
    lab_sent = np.array([i % 2 for i in range(2 * len(PAIRS))])
    assert S_last.shape[0] == 2 * len(PAIRS)
    assert W_HI <= S_last.shape[1] - 1

    mean_en = S_last[lab_sent == 0].mean(0)
    mean_fr = S_last[lab_sent == 1].mean(0)
    dirs, orth_dirs, cos_lang, cos_orth = {}, {}, {}, {}
    for li in range(W_LO, W_HI):
        d = unit(mean_en[li] - mean_fr[li])
        dirs[li] = d
        c = float(d @ lang_dir)
        cos_lang[li] = c
        o = d - c * lang_dir
        orth_dirs[li] = unit(o)
        cos_orth[li] = float(orth_dirs[li] @ lang_dir)
    log('Stage1: cos(dir_q,lang_dir)=%s' %
        {k: round(v, 4) for k, v in cos_lang.items()})
    log('Stage1: cos(orth,lang_dir)=%s' %
        {k: round(v, 4) for k, v in cos_orth.items()})

    # ---------- Stage 2: three-condition dual-channel injection ----
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
        for p in layers[li_].self_attn.parameters():
            assert p.device.type != 'meta'
        for p in layers[li_].mlp.parameters():
            assert p.device.type != 'meta'

    cap = {'mlpin': {}, 'attnin': {}}
    handles = []

    def pre_mlp(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            cap['mlpin'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())
        return h

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            cap['attnin'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())
        return h

    for li in range(W_LO, W_HI):
        handles.append(layers[li].mlp.register_forward_pre_hook(
            pre_mlp(li), with_kwargs=True))
        handles.append(layers[li].self_attn.register_forward_pre_hook(
            pre_attn(li), with_kwargs=True))

    def clear_cap():
        for dd in cap:
            for li in cap[dd]:
                del cap[dd][li][:]

    def forward2(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap['mlpin'][li][0] for li in cap['mlpin']}, \
            {li: cap['attnin'][li][0] for li in cap['attnin']}

    vdt = next(layers[W_LO].mlp.parameters()).dtype

    def layer_dev(li):
        return next(layers[li].mlp.parameters()).device

    rotary = model.model.rotary_emb

    def mlp_call(li, X):
        t = torch.tensor(X, device=layer_dev(li), dtype=vdt)
        with torch.no_grad():
            o = layers[li].mlp(t)
        if isinstance(o, tuple):
            o = o[0]
        return o[0, 1].detach().float().cpu().numpy().astype(np.float64)

    def attn_call(li, X):
        t = torch.tensor(X, device=layer_dev(li), dtype=vdt)
        position_ids = torch.arange(t.shape[1],
                                    device=layer_dev(li)).unsqueeze(0)
        with torch.no_grad():
            pos_emb = rotary(t, position_ids)
            o = layers[li].self_attn(
                t, position_embeddings=pos_emb, attention_mask=None,
                past_key_values=None)
        if isinstance(o, tuple):
            o = o[0]
        return o[0, 1].detach().float().cpu().numpy().astype(np.float64)

    def same_ctx(i):
        lang = words[i][0]
        cands = [j for j in range(n_words)
                 if words[j][0] == lang and j != i]
        return min(cands, key=lambda j: tid_map[words[j][2]])

    n_win = W_HI - W_LO
    RTS = ('stale', 'eigen', 'orth')
    g = {ch: {rt: {cn: np.zeros((n_words, n_win)) for cn in
                   ('same', 'func', 'null')}
              for rt in RTS} for ch in ('mlp', 'attn')}
    v1_err = {'mlp': 0.0, 'attn': 0.0}
    rng = np.random.default_rng(SEED)
    vocab_size = int(cfg.get('vocab_size'))
    word_tids = set(tid_map.values())
    null_tids = []
    while len(null_tids) < n_words:
        r = int(rng.integers(0, vocab_size))
        if r not in word_tids and r > 0:
            null_tids.append(r)

    def dir_of(rt, li):
        if rt == 'stale':
            return lang_dir
        if rt == 'eigen':
            return dirs[li]
        return orth_dirs[li]

    for i, (_, _, w) in enumerate(words):
        w_tid = tid_map[w]
        conds = {'same': [tid_map[words[same_ctx(i)][2]], w_tid],
                 'func': [func_tid, w_tid],
                 'null': [null_tids[i], w_tid]}
        for cn, toks in conds.items():
            mlpin_all, attnin_all = forward2(toks)
            for q, li in enumerate(range(W_LO, W_HI)):
                x_m = mlpin_all[li]
                x_a = attnin_all[li]
                ref_m = mlp_call(li, x_m)
                ref_a = attn_call(li, x_a)
                if cn == 'same' and q == 0:
                    v1_err['mlp'] = max(
                        v1_err['mlp'],
                        float(np.linalg.norm(
                            mlp_call(li, x_m) - ref_m))
                        / max(float(np.linalg.norm(ref_m)), 1e-30))
                    v1_err['attn'] = max(
                        v1_err['attn'],
                        float(np.linalg.norm(
                            attn_call(li, x_a) - ref_a))
                        / max(float(np.linalg.norm(ref_a)), 1e-30))
                for rt in RTS:
                    d_rt = dir_of(rt, li)
                    xm_p = x_m.copy()
                    xm_p[0, 1] = xm_p[0, 1] + EPS * d_rt
                    g['mlp'][rt][cn][i, q] = float(
                        (mlp_call(li, xm_p) - ref_m) @ d_rt) / EPS
                    xa_p = x_a.copy()
                    xa_p[0, 1] = xa_p[0, 1] + EPS * d_rt
                    g['attn'][rt][cn][i, q] = float(
                        (attn_call(li, xa_p) - ref_a) @ d_rt) / EPS
        if (i + 1) % 10 == 0:
            log('words [%d/%d] v1 mlp=%.2e attn=%.2e'
                % (i + 1, n_words, v1_err['mlp'], v1_err['attn']))

    v1 = bool(v1_err['mlp'] < 1e-6 and v1_err['attn'] < 1e-6)
    log('v1: mlp=%.3e attn=%.3e pass=%s'
        % (v1_err['mlp'], v1_err['attn'], v1))

    B = {ch: {rt: g[ch][rt]['same']
              - 0.5 * (g[ch][rt]['func'] + g[ch][rt]['null'])
              for rt in RTS} for ch in ('mlp', 'attn')}

    def margin_of(Sm, pl):
        n = len(pl)
        sm = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                sm[i, j] = (pl[i] == pl[j]) and i != j
        df = (~np.eye(n, dtype=bool)) & (~sm)
        return float(Sm[sm].mean() - Sm[df].mean())

    rng2 = np.random.default_rng(SEED)
    perms = [rng2.permutation(lab_lang) for _ in range(N_NULL)]
    out_c = {ch: {} for ch in ('mlp', 'attn')}
    for ch in ('mlp', 'attn'):
        for rt in RTS:
            C = zmat(B[ch][rt])
            acc = loo_acc(C, lab_lang)
            U = B[ch][rt] / np.maximum(
                np.linalg.norm(B[ch][rt], axis=1, keepdims=True),
                1e-30)
            Sm = U @ U.T
            n = len(lab_lang)
            same_m = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(n):
                    same_m[i, j] = (lab_lang[i] == lab_lang[j]) \
                        and i != j
            off = ~np.eye(n, dtype=bool) & ~same_m
            m_obs = float(Sm[same_m].mean() - Sm[off].mean())
            null_acc = [loo_acc(C, pl) for pl in perms]
            null_m = [margin_of(Sm, pl) for pl in perms]
            acc_p95 = float(np.percentile(null_acc, 95))
            m_p95 = float(np.percentile(null_m, 95))
            acc_c = loo_acc(C, lab_concept)
            per_layer = [round(loo_acc(zmat(B[ch][rt][:, [q]]),
                                       lab_lang), 4)
                         for q in range(n_win)]
            out_c[ch][rt] = {
                'acc': round(acc, 4), 'null_p95': round(acc_p95, 4),
                'null_mean': round(float(np.mean(null_acc)), 4),
                'margin': round(m_obs, 4),
                'margin_p95': round(m_p95, 4),
                'concept_acc': round(acc_c, 4),
                'per_layer_acc_descriptive': per_layer,
                'layer_profile': [round(float(x), 4)
                                  for x in np.abs(
                                      B[ch][rt]).mean(axis=0)]}
            log('%s/%s: acc=%.4f p95=%.4f margin=%.4f p95=%.4f '
                'concept=%.4f'
                % (ch, rt, acc, acc_p95, m_obs, m_p95, acc_c))

    a = {ch: {rt: bool(out_c[ch][rt]['acc'] > out_c[ch][rt]['null_p95'])
              for rt in RTS} for ch in ('mlp', 'attn')}
    orth_pos = a['mlp']['orth'] or a['attn']['orth']
    stale_pos = a['mlp']['stale'] or a['attn']['stale']
    eig_pos = a['mlp']['eigen'] or a['attn']['eigen']
    if orth_pos and stale_pos:
        verdict = 'broad_tolerance_glm4'
    elif orth_pos:
        verdict = 'partial_tolerance_spectrum_glm4'
    elif eig_pos:
        verdict = 'direction_matched_strict_glm4'
    else:
        verdict = 'readout_absent_window'

    res = {
        'phase': 2897, 'model': 'glm4-9b-chat-hf', 'prereg': PREREG,
        'window': [W_LO, W_HI],
        'cos_dirq_langdir': {str(k): round(v, 4)
                             for k, v in cos_lang.items()},
        'cos_orth_langdir': {str(k): round(v, 4)
                             for k, v in cos_orth.items()},
        'v1': v1, 'v1_err': {k: float('%.3e' % v)
                             for k, v in v1_err.items()},
        'responses': out_c,
        'A_flags': {'%s_%s' % (ch, rt): a[ch][rt]
                    for ch in ('mlp', 'attn') for rt in RTS},
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'glm4_readout_spectrum.npz'),
        B_mlp_stale=B['mlp']['stale'].astype(np.float32),
        B_mlp_eigen=B['mlp']['eigen'].astype(np.float32),
        B_mlp_orth=B['mlp']['orth'].astype(np.float32),
        B_attn_stale=B['attn']['stale'].astype(np.float32),
        B_attn_eigen=B['attn']['eigen'].astype(np.float32),
        B_attn_orth=B['attn']['orth'].astype(np.float32),
        cos_dirq_langdir=np.array([cos_lang[li] for li in
                                   range(W_LO, W_HI)],
                                  dtype=np.float32),
        labels_lang=lab_lang, labels_concept=lab_concept,
        words=np.array(['%s:%s:%s' % w for w in words], dtype=object))
    log('==== VERDICT: %s ====' % verdict)
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
