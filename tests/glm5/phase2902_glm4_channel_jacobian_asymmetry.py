# -*- coding: utf-8 -*-
"""Phase 2902: GLM4 attn/mlp channel Jacobian asymmetry (structural
root of the margin hierarchy gap).

2896-2901 established the readout-tolerance spectrum: qwen mlp
(~0.2) >> glm4 attn (~0.06-0.12) >> glm4 mlp (~0.02-0.03).  In
GLM4, same direction family (dir_q(li), eigen condition) yields
attn-channel margins ~4-5x the mlp-channel margins.  Both channels
inject eps*d at the MODULE INPUT at pos 1 (2897 machinery), so the
landed residual-stream direction is the channel's LOCAL JACOBIAN
response:
  r_attn(li) = (Attn_li(x+eps d) - Attn_li(x))/eps   (attn Jacobian)
  r_mlp(li)  = (Mlp_li(x+eps d) - Mlp_li(x))/eps     (mlp Jacobian)
The B matrix measured in 2897/2900/2901 is the d-projection of
this response: B = r_ch . d.  The margin gap must therefore be
explained by how each channel Jacobian transforms d:
  gamma_ch = ||r_ch||           (norm gain)
  rho_ch   = cos(r_ch, d)       (direction retention; B ~ gamma*rho)
  A_ch     = unembed alignment of r_ch (language readout of the
             LANDED direction, not just its d-component)

Stage 1 (low forward, 2897 machinery verbatim): for li in [28,40),
78 words x 3 conds (same/func/null, contexts reproduced bit-exact
with SEED=2897 rng order), capture module inputs via forward2, then
  r_ch = (module_call(li, x + eps*d) - ref)/eps  at pos 1.
v2 anchor: B' = [r.same - 0.5(r.func + r.null)] . d must equal the
stored B_attn_eigen / B_mlp_eigen (2897 npz) to rel err < 1e-3,
else all void.

Stage 2 (decomposition, zero forward on captured r):
  gamma, rho, A (78 word unembed rows; 200 null rows SEED=2897
  continuation), per li.  Ratios R_gamma/R_rho/R_A (median over
  li), B-scale ratio R_B (median of mean|B'_attn|/mean|B'_mlp|).

Stage 3 (weights-only descriptive): o_proj vs down_proj spectra
per li: top1/top2, participation ratio, energy gain on d.

Null control: 20 random unit dirs (SEED=2902) at layers
{28,31,34,37}, both channels, same-cond: gamma_null/rho_null/A_null.

Verdict (frozen):
  anchor fail                     => v2_anchor_fail_all_void
  R_B >= 2.0 and max(R_gamma,R_rho) >= 1.3
      => jacobian_asymmetry_confirmed_dominant_{norm_gain|
         direction_retention}   (dominant = larger of the two)
  else R_A >= 1.3                => language_alignment_asymmetry
  else                           => asymmetry_unresolved
Also reports margin_ratio(B'_attn/B'_mlp, margin_of SEED=2897
perms) against the observed ~4-5x hierarchy gap.

SEED=2897 (context replication + perms), SEED=2902 (null dirs).
Output: phase2902/glm4_channel_jacobian_asymmetry/.
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
SRC_2893 = os.path.join(BASE, 'phase2893', 'language_perlayer_glm4',
                        'language_perlayer_glm4.npz')
SRC_2897 = os.path.join(BASE, 'phase2897', 'glm4_readout_spectrum',
                        'glm4_readout_spectrum.npz')
OUT = os.path.join(BASE, 'phase2902', 'glm4_channel_jacobian_asymmetry')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\glm4-9b-chat-hf'
SEED = 2897
SEED_NULL = 2902
N_NULL = 200
EPS = 1.0
W_LO, W_HI = 28, 40
N_RND_DIRS = 20
RND_LAYERS = (28, 31, 34, 37)

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
    'window': 'W=[28,40) frozen (same as 2890-2893/2897)',
    'sources': 'dirs from 2893 npz (cross-checked vs S_last 2890 '
               'recompute, allclose); words/labels/contexts '
               'reproduced bit-exact from 2897 machinery (SEED=2897 '
               'rng order: null_tids first, then perms)',
    'response': 'r_ch(li,i,cn) = (module_call(li, x+eps*dir_q(li)) '
                '- ref)/eps at pos 1, module in {mlp, self_attn}, '
                'eps=1.0, 78 words x 3 conds; machinery verbatim '
                '2897',
    'v2_anchor': "B'[ch] = (r.same - 0.5(r.func + r.null)) . d must "
                 'match stored B_%s_eigen (2897 npz) rel err < 1e-3 '
                 'else v2_anchor_fail_all_void',
    'metrics': 'per (ch, li): gamma=mean_i||r||, '
               '  rho=mean_i cos(r,d), A=mean_i mean_w |cos(r,u_w)| '
               '(78 word rows; 200 null rows, fresh rng '
               'SEED=2897); ratios R_gamma/R_rho/'
               'R_A = median over li of attn/mlp; R_B = median of '
               'mean|B_attn|/mean|B_mlp|',
    'null_control': '20 random unit dirs SEED=2902, layers '
                    '{28,31,34,37}, same-cond: gamma_null/rho_null/'
                    'A_null descriptive',
    'weights_descriptive': 'down_proj / o_proj spectra (top1/top2, '
                           'participation ratio) + zero-forward '
                           'response gain and cos(response,d): mlp '
                           'via SwiGLU on gain*d, attn via GQA-'
                           'expanded W_VO composite on gain*d; '
                           'W_VO composite spectra recorded; all '
                           'descriptive',
    'verdict': 'anchor fail => v2_anchor_fail_all_void; '
               'R_B>=2.0 and max(R_gamma,R_rho)>=1.3 => '
               'jacobian_asymmetry_confirmed_dominant_{norm_gain|'
               'direction_retention}; else R_A>=1.3 => '
               'language_alignment_asymmetry; else '
               'asymmetry_unresolved',
    'margin_ratio': 'margin_of(B_attn)/margin_of(B_mlp) with '
                    'SEED=2897 perms (200), compared to observed '
                    '~4-5x hierarchy gap (2897/2900/2901)',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.monotonic()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2902,
                   'name': 'glm4_channel_jacobian_asymmetry',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s2890': sha8(SRC_2890),
                               's2893': sha8(SRC_2893),
                               's2897': sha8(SRC_2897)},
                   'model': 'glm4-9b-chat-hf',
                   'n_pairs': len(PAIRS),
                   'prereg': PREREG, 'seed': SEED,
                   'seed_null_dirs': SEED_NULL, 'eps': EPS,
                   'window': [W_LO, W_HI]},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    # ---------- Stage 0: sources + dir cross-check ----------
    z90 = np.load(SRC_2890, allow_pickle=True)
    z93 = np.load(SRC_2893, allow_pickle=True)
    z97 = np.load(SRC_2897, allow_pickle=True)
    S_last = z90['S_last'].astype(np.float64)   # (80, 41, 4096)
    words = [tuple(str(w).split(':')) for w in z90['words']]
    lab_lang = z90['labels_lang']
    lab_concept = z90['labels_concept']
    n_words = len(words)
    lab_sent = np.array([i % 2 for i in range(2 * len(PAIRS))])
    assert S_last.shape[0] == 2 * len(PAIRS)
    assert z93['dirs'].shape == (W_HI - W_LO, 4096)
    dirs_npz = z93['dirs'].astype(np.float64)
    mean_en = S_last[lab_sent == 0].mean(0)
    mean_fr = S_last[lab_sent == 1].mean(0)
    d_ck = np.stack([unit(mean_en[li] - mean_fr[li])
                     for li in range(W_LO, W_HI)])
    dir_check_max = float(np.abs(d_ck - dirs_npz).max())
    log('dir cross-check 2893-vs-2890-recompute max=%.3e'
        % dir_check_max)
    assert dir_check_max < 1e-6

    # ---------- Stage 1: low-forward channel Jacobians ----------
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
    rng = np.random.default_rng(SEED)
    vocab_size = int(cfg.get('vocab_size'))
    word_tids = set(tid_map.values())
    null_tids = []
    while len(null_tids) < n_words:
        r = int(rng.integers(0, vocab_size))
        if r not in word_tids and r > 0:
            null_tids.append(r)

    # r[ch][cn] : (n_words, n_win, 4096) float32
    r_store = {ch: {cn: np.zeros((n_words, n_win, 4096),
                                  dtype=np.float32)
                    for cn in ('same', 'func', 'null')}
               for ch in ('mlp', 'attn')}
    v1_err = {'mlp': 0.0, 'attn': 0.0}

    for i, (_, _, w) in enumerate(words):
        w_tid = tid_map[w]
        conds = {'same': [tid_map[words[same_ctx(i)][2]], w_tid],
                 'func': [func_tid, w_tid],
                 'null': [null_tids[i], w_tid]}
        for cn, toks in conds.items():
            mlpin_all, attnin_all = forward2(toks)
            for q, li in enumerate(range(W_LO, W_HI)):
                d = dirs_npz[q]
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
                xm_p = x_m.copy()
                xm_p[0, 1] = xm_p[0, 1] + EPS * d
                r_m = (mlp_call(li, xm_p) - ref_m) / EPS
                xa_p = x_a.copy()
                xa_p[0, 1] = xa_p[0, 1] + EPS * d
                r_a = (attn_call(li, xa_p) - ref_a) / EPS
                r_store['mlp'][cn][i, q] = r_m.astype(np.float32)
                r_store['attn'][cn][i, q] = r_a.astype(np.float32)
        if (i + 1) % 10 == 0:
            log('words [%d/%d] v1 mlp=%.2e attn=%.2e'
                % (i + 1, n_words, v1_err['mlp'], v1_err['attn']))

    v1 = bool(v1_err['mlp'] < 1e-6 and v1_err['attn'] < 1e-6)
    log('v1: mlp=%.3e attn=%.3e pass=%s'
        % (v1_err['mlp'], v1_err['attn'], v1))

    # B'[ch] = (r_same - 0.5 r_func - 0.5 r_null) . d  (2897 contrast)
    d_stack = np.stack([dirs_npz[q] for q in range(n_win)])
    B_proj = {}
    for ch in ('mlp', 'attn'):
        r_c = (r_store[ch]['same'].astype(np.float64)
               - 0.5 * r_store[ch]['func'].astype(np.float64)
               - 0.5 * r_store[ch]['null'].astype(np.float64))
        B_proj[ch] = np.einsum('nqd,qd->nq', r_c, d_stack)

    # ---------- v2 anchor vs stored 2897 B ----------
    anchor = {}
    for ch in ('mlp', 'attn'):
        stored = z97['B_%s_eigen' % ch].astype(np.float64)
        diff = np.abs(B_proj[ch] - stored)
        anchor[ch] = float(diff.max()
                           / max(float(np.abs(stored).max()), 1e-30))
    log('v2 anchor rel err: mlp=%.3e attn=%.3e'
        % (anchor['mlp'], anchor['attn']))
    anchor_ok = bool(anchor['mlp'] < 1e-3 and anchor['attn'] < 1e-3)

    # ---------- unembed rows ----------
    Wu = model.lm_head.weight.detach().float().cpu().numpy()
    n_null_rows = 0
    rng_rows = np.random.default_rng(SEED)   # continuation: fresh
    null_rows = []
    while len(null_rows) < N_NULL:
        r = int(rng_rows.integers(0, vocab_size))
        if r not in word_tids and r > 0:
            null_rows.append(r)
    word_rows = np.stack([unit(Wu[tid_map[w]])
                          for _, _, w in words])
    nullU = np.stack([unit(Wu[r]) for r in null_rows])
    del Wu

    def A_of(R):
        # R: (n, 4096) -> mean |cos| with word rows, mean with null
        Un = R / np.maximum(np.linalg.norm(
            R, axis=1, keepdims=True), 1e-30)
        a_w = float(np.abs(Un @ word_rows.T).mean())
        a_n = float(np.abs(Un @ nullU.T).mean())
        return a_w, a_n

    # ---------- Stage 2: decomposition ----------
    gam, rho, aw, an = {}, {}, {}, {}
    for ch in ('mlp', 'attn'):
        gam[ch] = np.linalg.norm(
            r_store[ch]['same'].astype(np.float64), axis=2).mean(0)
        dcos = np.einsum('nqd,qd->nq',
                         r_store[ch]['same'].astype(np.float64),
                         np.stack([dirs_npz[q]
                                   for q in range(n_win)]))
        rho[ch] = (dcos / np.maximum(np.linalg.norm(
            r_store[ch]['same'].astype(np.float64), axis=2),
            1e-30)).mean(0)
        a_w_l, a_n_l = [], []
        for q in range(n_win):
            aw_, an_ = A_of(r_store[ch]['same'][:, q, :]
                            .astype(np.float64))
            a_w_l.append(aw_)
            a_n_l.append(an_)
        aw[ch] = np.array(a_w_l)
        an[ch] = np.array(a_n_l)

    Babs = {ch: np.abs(B_proj[ch]).mean(0) for ch in ('mlp', 'attn')}
    R_gamma = float(np.median(gam['attn'] / gam['mlp']))
    R_rho = float(np.median(rho['attn'] / rho['mlp']))
    R_A = float(np.median(aw['attn'] / aw['mlp']))
    R_B = float(np.median(Babs['attn'] / np.maximum(Babs['mlp'],
                                                    1e-30)))
    log('R_gamma=%.3f R_rho=%.3f R_A=%.3f R_B=%.3f'
        % (R_gamma, R_rho, R_A, R_B))

    # ---------- margin ratio (SEED=2897 perms continuation) ------
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
    marg = {}
    accs = {}
    for ch in ('mlp', 'attn'):
        C = zmat(B_proj[ch])
        accs[ch] = loo_acc(C, lab_lang)
        U = B_proj[ch] / np.maximum(np.linalg.norm(
            B_proj[ch], axis=1, keepdims=True), 1e-30)
        Sm = U @ U.T
        n = len(lab_lang)
        same_m = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                same_m[i, j] = (lab_lang[i] == lab_lang[j]) and i != j
        off = ~np.eye(n, dtype=bool) & ~same_m
        m_obs = float(Sm[same_m].mean() - Sm[off].mean())
        null_m = [margin_of(Sm, pl) for pl in perms]
        marg[ch] = {'margin': m_obs,
                    'margin_p95': float(np.percentile(null_m, 95))}
        log('%s: margin=%.4f p95=%.4f acc=%.4f'
            % (ch, m_obs, marg[ch]['margin_p95'], accs[ch]))
    margin_ratio = (marg['attn']['margin']
                    / max(abs(marg['mlp']['margin']), 1e-30))
    log('margin_ratio attn/mlp = %.2f' % margin_ratio)

    # ---------- null random directions ----------
    rng3 = np.random.default_rng(SEED_NULL)
    null_stats = []
    for k in range(N_RND_DIRS):
        q = int(rng3.integers(0, n_win))
        li = W_LO + q
        d_r = unit(rng3.normal(size=4096))
        w_i = int(rng3.integers(0, n_words))
        w_tid = tid_map[words[w_i][2]]
        toks = [tid_map[words[same_ctx(w_i)][2]], w_tid]
        mlpin_all, attnin_all = forward2(toks)
        g_r, r_r = {}, {}
        x_m = mlpin_all[li]
        x_a = attnin_all[li]
        ref_m = mlp_call(li, x_m)
        ref_a = attn_call(li, x_a)
        xm_p = x_m.copy()
        xm_p[0, 1] = xm_p[0, 1] + EPS * d_r
        r_m = (mlp_call(li, xm_p) - ref_m) / EPS
        xa_p = x_a.copy()
        xa_p[0, 1] = xa_p[0, 1] + EPS * d_r
        r_a = (attn_call(li, xa_p) - ref_a) / EPS
        entry = {'k': k, 'li': li}
        for ch, rv in (('mlp', r_m), ('attn', r_a)):
            aw_, an_ = A_of(rv[None, :])
            entry[ch] = {'gamma': float(np.linalg.norm(rv)),
                         'rho': float(rv @ d_r),
                         'A_word': aw_, 'A_null': an_}
        null_stats.append(entry)
    gam_n = {ch: float(np.median([e[ch]['gamma']
                                  for e in null_stats]))
             for ch in ('mlp', 'attn')}
    rho_n = {ch: float(np.median([e[ch]['rho']
                                  for e in null_stats]))
             for ch in ('mlp', 'attn')}
    log('null dirs: gamma_n mlp=%.3f attn=%.3f ; rho_n mlp=%.3f '
        'attn=%.3f' % (gam_n['mlp'], gam_n['attn'],
                       rho_n['mlp'], rho_n['attn']))

    # ---------- Stage 3: weights-only descriptive ----------
    def spectra(W):
        sv = np.linalg.svd(W, compute_uv=False)
        return {'top1_over_top2': float(sv[0] / max(sv[1], 1e-30)),
                'participation_ratio': float(
                    sv.sum() ** 2 / max(float((sv ** 2).sum()),
                                        1e-30))}

    spec = {}
    for q, li in enumerate(range(W_LO, W_HI)):
        d = unit(dirs_npz[q])
        ly = layers[li]
        g_mlp = ly.post_attention_layernorm.weight.detach() \
            .float().cpu().numpy()
        g_attn = ly.input_layernorm.weight.detach() \
            .float().cpu().numpy()
        entry = {}
        # mlp: down_proj spectra + SwiGLU zero-forward response on d
        Wd = ly.mlp.down_proj.weight.detach().float().cpu().numpy()
        Wgu = ly.mlp.gate_up_proj.weight.detach().float() \
            .cpu().numpy()          # fused [2*inter, 4096]
        inter = Wgu.shape[0] // 2
        Wg, Wu_ = Wgu[:inter], Wgu[inter:]
        dg = g_mlp * d
        h = (torch.nn.functional.silu(
            torch.tensor(Wg @ dg)) * torch.tensor(Wu_ @ dg)).numpy()
        v_m = Wd @ h
        entry['mlp'] = {**spectra(Wd),
                        'zf_response_gain': float(
                            np.linalg.norm(v_m) ** 2),
                        'zf_cos_resp_d': float(unit(v_m) @ d)}
        del Wd, Wg, Wu_
        # attn: o_proj spectra + W_VO composite (GQA-expanded) spectra
        Wo = ly.self_attn.o_proj.weight.detach().float() \
            .cpu().numpy()          # [4096, 4096]
        Wv = ly.self_attn.v_proj.weight.detach().float() \
            .cpu().numpy()          # [256, 4096]
        nh = int(cfg['num_attention_heads'])
        nkv = int(cfg['num_key_value_heads'])
        hd = 4096 // nh
        rep = nh // nkv
        M = np.zeros((4096, 4096))
        for i in range(nh):
            M[:, i * hd:(i + 1) * hd] = \
                Wv[(i // rep) * hd:(i // rep + 1) * hd, :].T
        W_VO = Wo @ M
        va = W_VO @ (g_attn * d)
        entry['attn'] = {**spectra(Wo),
                         'spectra_W_VO_composite': spectra(W_VO),
                         'zf_response_gain': float(
                             np.linalg.norm(va) ** 2),
                         'zf_cos_resp_d': float(unit(va) @ d)}
        del Wo, Wv, M, W_VO
        spec[li] = entry
    log('weights spectra done')

    # ---------- verdict ----------
    if not anchor_ok:
        verdict = 'v2_anchor_fail_all_void'
    elif R_B >= 2.0 and max(R_gamma, R_rho) >= 1.3:
        dom = 'norm_gain' if R_gamma >= R_rho \
            else 'direction_retention'
        verdict = 'jacobian_asymmetry_confirmed_dominant_' + dom
    elif R_A >= 1.3:
        verdict = 'language_alignment_asymmetry'
    else:
        verdict = 'asymmetry_unresolved'

    res = {
        'phase': 2902, 'model': 'glm4-9b-chat-hf',
        'prereg': PREREG, 'window': [W_LO, W_HI],
        'dir_check_max': dir_check_max,
        'v1': v1, 'v1_err': {k: float('%.3e' % v)
                             for k, v in v1_err.items()},
        'v2_anchor_rel_err': anchor, 'v2_anchor_ok': anchor_ok,
        'per_layer': {
            'gamma': {ch: [round(float(x), 4) for x in gam[ch]]
                      for ch in ('mlp', 'attn')},
            'rho': {ch: [round(float(x), 4) for x in rho[ch]]
                    for ch in ('mlp', 'attn')},
            'A_word': {ch: [round(float(x), 5) for x in aw[ch]]
                       for ch in ('mlp', 'attn')},
            'A_nullrow': {ch: [round(float(x), 5) for x in an[ch]]
                          for ch in ('mlp', 'attn')},
            'B_absmean': {ch: [round(float(x), 4)
                               for x in Babs[ch]]
                          for ch in ('mlp', 'attn')},
        },
        'ratios': {'R_gamma': round(R_gamma, 4),
                   'R_rho': round(R_rho, 4),
                   'R_A': round(R_A, 4), 'R_B': round(R_B, 4)},
        'margins': marg, 'accs': accs,
        'margin_ratio_attn_over_mlp': round(margin_ratio, 3),
        'null_dirs': {'gamma_null_med': gam_n,
                      'rho_null_med': rho_n,
                      'entries': null_stats},
        'weights_descriptive': {str(li): {
            ch: {kk: (round(vv, 5) if not isinstance(vv, dict)
                      else {k2: round(v2, 5)
                            for k2, v2 in vv.items()})
                  for kk, vv in e[ch].items()}
            for ch in ('mlp', 'attn')} for li, e in spec.items()},
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'glm4_channel_jacobian_asymmetry.npz'),
        B_mlp=B_proj['mlp'].astype(np.float32),
        B_attn=B_proj['attn'].astype(np.float32),
        r_mlp_same=r_store['mlp']['same'],
        r_attn_same=r_store['attn']['same'],
        gamma_mlp=gam['mlp'].astype(np.float32),
        gamma_attn=gam['attn'].astype(np.float32),
        rho_mlp=rho['mlp'].astype(np.float32),
        rho_attn=rho['attn'].astype(np.float32),
        A_word_mlp=aw['mlp'].astype(np.float32),
        A_word_attn=aw['attn'].astype(np.float32),
        labels_lang=lab_lang, labels_concept=lab_concept,
        words=np.array(['%s:%s:%s' % w for w in words],
                       dtype=object))
    log('==== VERDICT: %s ====' % verdict)
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
