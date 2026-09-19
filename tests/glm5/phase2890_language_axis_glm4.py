# -*- coding: utf-8 -*-
"""Phase 2890: GLM4 language-axis discrimination (2887 protocol port).

Candidate A from 2889: port the 2886+2887 language-axis protocol to
glm4-9b-chat-hf to test whether the L10 finding (mid-stream language
direction acts as an mlp carrier direction; language/concept channel
separation E3=0 on qwen3-4b) replicates cross-architecture.

Stage 1 (sentence states, 2886 protocol verbatim): the same 40 frozen
en/fr parallel pairs (script constant copied verbatim from 2886);
GLM4 forward with output_hidden_states=True -> S_last (80, 41, 4096).
Descriptive D0: CKA curve + language probe per layer with scaled
mid-third (L=40: mid = li 13..27 scaled from 12..24 at L=36; ends =
first 6 + last 6) - GLM4 hourglass check, NOT gated (2886 negative
already settled the stripping story on the primary model).

Direction (zero additional forward): lang_dir = unit(mean
S_last[en, LI_LANG] - mean S_last[fr, LI_LANG]), LI_LANG = 20 (scaled
rule round(18/36*L); new per-model provenance registered - sentence
states extracted on GLM4 itself, not transplanted across models).

Stage 2 (mlp carrier, 2887 protocol verbatim):
  vocab: 2878 TRANS_PAIRS + HOMOGRAPH_EXCLUDE verbatim (canonical en,L
  form), GLM4 single-token filter; tid-level dedup across languages;
  labels en=0, L=1 (fr+de+es pooled).
  conds same/func('the')/null, pos-1 two-token forward; same-context
  word = another word of the SAME language (min tid);
  cdir = lang_dir for every window layer (WIN=[28,40), 2888 rule);
  g[cn][i,q] = [mlp(mlpin + eps*cdir) - mlp(mlpin)] . cdir / eps;
  B3_lang = g(same) - 0.5*(g(func)+g(null)).

Prereg (frozen before any forward):
  v1  determinism: mlp recompute twice identical input, max rel err
      < 1e-6 else all void.
  E1  language retrieval: acc(loo-NN on zmat(B3_lang), lab_lang) >
      null p95 (200 label permutations, SEED=2890)
      => mlp_carries_language_axis_glm4.
  E2  same-language cos margin > null p95 => language_margin_in_mlp_glm4.
  E3  descriptive: concept-pair retrieval on same spectrum + layer
      mean|B3_lang| profile (L10 separation check: concept acc should
      stay near null if the channel is language-pure).
  E4  descriptive (zero forward): per-layer cos(lang_dir_q, lang_dir)
      for window q and mid-third layers.
  WC  descriptive band (frozen): |acc_glm4 - 0.7719| <= 0.15 =>
      language_carrier_replicates_within_band else
      language_carrier_deviation (2887 qwen3-4b acc = 0.7719 anchor,
      mirroring the class W2 banding).
  verdict language_axis_in_mlp_glm4 iff v1 and E1 and E2.
SEED=2890.  Output: phase2890/language_axis_glm4/.
"""
import hashlib
import io
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2890', 'language_axis_glm4')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\glm4-9b-chat-hf'
SEED = 2890
N_NULL = 200
EPS = 1.0
LI_LANG = 20
QWEN_2887_ACC = 0.7719

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

TRANS_PAIRS = [
    ('cat', 'chat'), ('house', 'maison'), ('book', 'livre'),
    ('night', 'nuit'), ('king', 'roi'), ('woman', 'femme'),
    ('man', 'homme'), ('tree', 'arbre'), ('wine', 'vin'),
    ('moon', 'lune'), ('sea', 'mer'), ('sky', 'ciel'),
    ('flower', 'fleur'), ('city', 'ville'), ('war', 'guerre'),
    ('foot', 'pied'), ('water', 'eau'), ('sun', 'soleil'),
    ('milk', 'lait'), ('dog', 'chien'),
    ('cat', 'Katze'), ('dog', 'Hund'), ('water', 'Wasser'),
    ('house', 'Haus'), ('book', 'Buch'), ('night', 'Nacht'),
    ('sun', 'Sonne'), ('milk', 'Milch'), ('bird', 'Vogel'),
    ('fish', 'Fisch'), ('bread', 'Brot'), ('wine', 'Wein'),
    ('moon', 'Mond'), ('star', 'Stern'), ('tree', 'Baum'),
    ('fire', 'Feuer'), ('mountain', 'Berg'), ('snow', 'Schnee'),
    ('woman', 'Frau'), ('wolf', 'Wolf'), ('king', 'Koenig'),
    ('summer', 'Sommer'),
    ('water', 'agua'), ('house', 'casa'), ('book', 'libro'),
    ('night', 'noche'), ('cat', 'gato'), ('dog', 'perro'),
    ('milk', 'leche'), ('table', 'mesa'), ('moon', 'luna'),
    ('sea', 'mar'), ('sky', 'cielo'), ('wine', 'vino'),
    ('flower', 'flor'), ('king', 'rey'), ('woman', 'mujer'),
    ('man', 'hombre'), ('mother', 'madre'), ('father', 'padre'),
    ('war', 'guerra'), ('light', 'luz'), ('fire', 'fuego'),
    ('earth', 'tierra'), ('door', 'puerta'), ('horse', 'caballo'),
    ('cheese', 'queso'), ('egg', 'huevo'), ('wolf', 'lobo'),
    ('snow', 'nieve'), ('bed', 'cama'), ('star', 'estrella'),
]
HOMOGRAPH_EXCLUDE = {
    'chat', 'sol', 'pain', 'chef', 'table', 'main', 'train',
    'route', 'lion', 'grand', 'mine', 'Mann', 'Gold', 'Winter',
    'Hand', 'Bank', 'Wind', 'Kind', 'Sommer', 'Morgen', 'Boot',
    'Arm', 'pan', 'plaza', 'real', 'color', 'metal', 'hotel',
    'piano', 'radio', 'rosa',
}

PREREG = {
    'D0': 'descriptive GLM4 hourglass check (2886 protocol, scaled '
          'mid third li 13..27); NOT gated (N5 settled on primary)',
    'direction': 'lang_dir = unit(mean S_last[en,20] - mean '
                 'S_last[fr,20]) from GLM4 own sentence states (80 '
                 'pairs verbatim 2886); LI scaled rule round(18/36*L); '
                 'per-model provenance registered',
    'vocab': '2878 TRANS_PAIRS + HOMOGRAPH_EXCLUDE verbatim, GLM4 '
             'single-token filter, tid dedup; labels en=0 L=1 pooled',
    'window': 'WIN=[28,40) (2888 frozen rule floor(26/36*L))',
    'v1': 'max relative mlp recompute error < 1e-6, else all void',
    'E1': 'acc(loo-NN zmat B3_lang, lab_lang) > null p95 (200 perms, '
          'SEED=2890) => mlp_carries_language_axis_glm4',
    'E2': 'same-language cos margin > null p95 => '
          'language_margin_in_mlp_glm4',
    'E3': 'descriptive: concept-pair retrieval acc + layer profile',
    'E4': 'descriptive: cos(lang_dir_q, lang_dir) profile, zero '
          'forward, window + mid third',
    'WC': 'descriptive band: |acc - 0.7719| <= 0.15 => '
          'language_carrier_replicates_within_band else '
          'language_carrier_deviation',
    'verdict': 'language_axis_in_mlp_glm4 iff v1 and E1 and E2',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def linear_cka(X, Y):
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    num = float(np.linalg.norm(Yc.T @ Xc, 'fro')) ** 2
    den = (float(np.linalg.norm(Xc.T @ Xc, 'fro'))
           * float(np.linalg.norm(Yc.T @ Yc, 'fro')))
    return num / max(den, 1e-30)


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
        json.dump({'phase': 2890, 'name': 'language_axis_glm4',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'model': 'glm4-9b-chat-hf',
                   'n_pairs': len(PAIRS),
                   'prereg': PREREG, 'seed': SEED, 'eps': EPS,
                   'li_lang': LI_LANG, 'qwen_2887_acc': QWEN_2887_ACC},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    cfg = json.load(io.open(os.path.join(MD, 'config.json'),
                            encoding='utf-8'))
    L = int(cfg['num_hidden_layers'])
    WIN_LO = int(np.floor(26 / 36.0 * L))
    WIN_HI = L
    MID_LO = int(round(12 / 36.0 * L))
    MID_HI = int(round(24 / 36.0 * L)) + 1

    dm = {'model.embed_tokens': 0, 'model.norm': 0, 'lm_head': 0}
    for li_ in range(L):
        dm['model.layers.%d' % li_] = 0 if WIN_LO <= li_ < WIN_HI \
            else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(
        MD, local_files_only=True, device_map=dm)
    model.eval()
    layers = model.model.layers
    log('model loaded: %d layers (window %d-%d on cuda)'
        % (L, WIN_LO, WIN_HI - 1))
    for li_ in range(WIN_LO, WIN_HI):
        for p in layers[li_].mlp.parameters():
            assert p.device.type != 'meta', \
                'meta tensor in window layer %d' % li_

    # ---------- stage 1: sentence states ----------
    sents = []
    for en, fr in PAIRS:
        sents.append(en)
        sents.append(fr)
    lab = np.array([i % 2 for i in range(2 * len(PAIRS))])
    states_last = []
    with torch.no_grad():
        for si, s in enumerate(sents):
            ids = tok(s, return_tensors='pt')['input_ids'].to('cuda')
            out = model(ids, output_hidden_states=True)
            hs = out.hidden_states
            last = torch.stack([h[0, -1] for h in hs])
            states_last.append(last.float().cpu().numpy())
            if (si + 1) % 20 == 0:
                log('sentences [%d/%d]' % (si + 1, len(sents)))
    S_last = np.stack(states_last)
    n_li = L + 1

    mid = list(range(MID_LO, MID_HI))
    ends = list(range(0, 6)) + list(range(L - 5, L + 1))
    cka = np.zeros(n_li)
    probe = np.zeros(n_li)
    for li in range(n_li):
        cka[li] = linear_cka(S_last[lab == 0, li, :],
                             S_last[lab == 1, li, :])
        probe[li] = loo_nc_acc(S_last[:, li, :], lab)
    log('D0 descriptive: cka peak li=%d val=%.4f probe mid=%.4f '
        'ends=%.4f' % (int(np.argmax(cka)), float(cka.max()),
                       float(probe[mid].mean()),
                       float(probe[ends].mean())))

    lang_dir = unit(S_last[lab == 0, LI_LANG].mean(0)
                    - S_last[lab == 1, LI_LANG].mean(0))
    log('lang_dir extracted from GLM4 S_last li=%d' % LI_LANG)

    # ---------- stage 2: vocab ----------
    from rdc_atlas_census import single_token_id
    tc = {}
    n_excl = 0
    eff_pairs = []
    for en_w, l_w in TRANS_PAIRS:
        if en_w in HOMOGRAPH_EXCLUDE or l_w in HOMOGRAPH_EXCLUDE:
            n_excl += 1
            continue
        try:
            single_token_id(tok, en_w, tc)
            single_token_id(tok, l_w, tc)
        except AssertionError:
            continue
        eff_pairs.append((en_w, l_w))
    log('vocab: %d/%d pairs after VG0(%d excl) + single-token filter'
        % (len(eff_pairs), len(TRANS_PAIRS), n_excl))
    tid_map = {'the': single_token_id(tok, 'the', tc)}

    words = []
    seen_tid = {}
    for en_w, l_w in eff_pairs:
        t_en = single_token_id(tok, en_w, tc)
        t_l = single_token_id(tok, l_w, tc)
        tid_map[en_w] = t_en
        tid_map[l_w] = t_l
        if t_en not in seen_tid:
            seen_tid[t_en] = len(words)
            words.append(('en', t_en, en_w))
        if t_l not in seen_tid:
            seen_tid[t_l] = len(words)
            words.append(('L', t_en, l_w))
    n_words = len(words)
    n_en = sum(1 for x in words if x[0] == 'en')
    log('words=%d (en=%d L=%d)' % (n_words, n_en, n_words - n_en))

    lab_lang = np.array([0 if w[0] == 'en' else 1 for w in words])
    lab_concept = np.array([w[1] for w in words])

    # ---------- hooks ----------
    cap = {'mlpin': {}, 'mlpout': {}}
    handles = []

    def pre_hook(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            cap['mlpin'].setdefault(li, []).append(
                x.detach()[0, 1].float().cpu().numpy())
        return h

    def out_hook(li):
        def h(module, args, output):
            o = output[0] if isinstance(output, tuple) else output
            if o.dim() < 2:
                return
            cap['mlpout'].setdefault(li, []).append(
                o.detach()[0, 1].float().cpu().numpy())
        return h

    for li in range(WIN_LO, WIN_HI):
        handles.append(layers[li].mlp.register_forward_pre_hook(
            pre_hook(li), with_kwargs=True))
        handles.append(layers[li].mlp.register_forward_hook(
            out_hook(li)))

    def clear_cap():
        for d in cap:
            for li in cap[d]:
                del cap[d][li][:]

    def forward2(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return ({li: cap['mlpin'][li][0] for li in cap['mlpin']},
                {li: cap['mlpout'][li][0] for li in cap['mlpout']})

    vdt = next(layers[WIN_LO].mlp.parameters()).dtype

    def layer_dev(li):
        return next(layers[li].mlp.parameters()).device

    def mlp_call(li, x):
        out = layers[li].mlp(torch.tensor(
            x, device=layer_dev(li), dtype=vdt))
        if isinstance(out, tuple):
            out = out[0]
        return out.detach().float().cpu().numpy()

    def same_ctx(i):
        lang = words[i][0]
        cands = [j for j in range(n_words)
                 if words[j][0] == lang and j != i]
        return min(cands, key=lambda j: tid_map[words[j][2]])

    g = {cn: np.zeros((n_words, WIN_HI - WIN_LO)) for cn in
         ('same', 'func', 'null')}
    v1a_err = 0.0
    hook_noise = 0.0
    rng = np.random.default_rng(SEED)
    vocab_size = int(cfg.get('vocab_size'))
    word_tids = set(tid_map.values())
    null_tids = []
    while len(null_tids) < n_words:
        r = int(rng.integers(0, vocab_size))
        if r not in word_tids and r > 0:
            null_tids.append(r)
    func_tid = tid_map['the']

    for i, (_, _, w) in enumerate(words):
        w_tid = tid_map[w]
        conds = {'same': [tid_map[words[same_ctx(i)][2]], w_tid],
                 'func': [func_tid, w_tid],
                 'null': [null_tids[i], w_tid]}
        for cn, toks in conds.items():
            mlpin, mlpout = forward2(toks)
            for q, li in enumerate(range(WIN_LO, WIN_HI)):
                x0 = mlpin[li]
                ref = mlp_call(li, x0)
                if cn == 'same' and q == 0:
                    ref2 = mlp_call(li, x0)
                    denom = max(float(np.linalg.norm(ref)), 1e-30)
                    v1a_err = max(v1a_err, float(np.linalg.norm(
                        ref - ref2)) / denom)
                    hook_ref = mlpout[li]
                    hook_noise = max(hook_noise, float(np.linalg.norm(
                        ref - hook_ref))
                        / max(float(np.linalg.norm(hook_ref)), 1e-30))
                g[cn][i, q] = float((mlp_call(li, x0 + EPS * lang_dir)
                                     - ref) @ lang_dir) / EPS
        if (i + 1) % 10 == 0:
            log('words [%d/%d] v1a=%.2e' % (i + 1, n_words, v1a_err))

    log('v1a determinism max rel err = %.3e' % v1a_err)
    v1 = bool(v1a_err < 1e-6)

    B3_lang = g['same'] - 0.5 * (g['func'] + g['null'])

    # ---------- E1 / E2 ----------
    C = zmat(B3_lang)
    acc = loo_acc(C, lab_lang)
    U = B3_lang / np.maximum(
        np.linalg.norm(B3_lang, axis=1, keepdims=True), 1e-30)
    S = U @ U.T
    n = len(lab_lang)
    same_m = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            same_m[i, j] = (lab_lang[i] == lab_lang[j]) and i != j
    off = ~np.eye(n, dtype=bool) & ~same_m

    def margin(pl):
        sm = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                sm[i, j] = (pl[i] == pl[j]) and i != j
        df = (~np.eye(n, dtype=bool)) & (~sm)
        return float(S[sm].mean() - S[df].mean())

    m_obs = float(S[same_m].mean() - S[off].mean())
    rng2 = np.random.default_rng(SEED)
    null_acc, null_m = [], []
    for _ in range(N_NULL):
        pl = rng2.permutation(lab_lang)
        null_acc.append(loo_acc(C, pl))
        null_m.append(margin(pl))
    na, nm = np.array(null_acc), np.array(null_m)
    e1 = bool(acc > float(np.percentile(na, 95)))
    e2 = bool(m_obs > float(np.percentile(nm, 95)))

    # ---------- E3 ----------
    acc_concept = loo_acc(C, lab_concept)
    rng3 = np.random.default_rng(SEED + 1)
    null_c = []
    for _ in range(N_NULL):
        null_c.append(loo_acc(C, rng3.permutation(lab_concept)))
    acc_concept_p95 = float(np.percentile(null_c, 95))
    layer_profile = np.abs(B3_lang).mean(axis=0)

    # ---------- E4 ----------
    e4_win = []
    for li in range(WIN_LO, WIN_HI):
        d_q = unit(S_last[lab == 0, li].mean(0)
                   - S_last[lab == 1, li].mean(0))
        e4_win.append(round(float(d_q @ lang_dir), 4))
    e4_mid = []
    for li in range(MID_LO, MID_HI):
        d_q = unit(S_last[lab == 0, li].mean(0)
                   - S_last[lab == 1, li].mean(0))
        e4_mid.append(round(float(d_q @ lang_dir), 4))

    wc = 'language_carrier_replicates_within_band' \
        if abs(acc - QWEN_2887_ACC) <= 0.15 \
        else 'language_carrier_deviation'
    e1_verdict = 'mlp_carries_language_axis_glm4' if e1 \
        else 'mlp_language_signal_absent_glm4'
    e2_verdict = 'language_margin_in_mlp_glm4' if e2 \
        else 'margin_absent_glm4'
    verdict = bool(v1 and e1 and e2)

    res = {
        'phase': 2890, 'model': 'glm4-9b-chat-hf', 'prereg': PREREG,
        'n_words': n_words, 'n_en': n_en, 'window': [WIN_LO, WIN_HI],
        'li_lang': LI_LANG,
        'v1a_max_rel_err': float('%.3e' % v1a_err), 'v1': v1,
        'kernel_context_noise': float('%.3e' % hook_noise),
        'D0': {'cka_peak_li': int(np.argmax(cka)),
               'cka_peak': round(float(cka.max()), 4),
               'probe_mid': round(float(probe[mid].mean()), 4),
               'probe_ends': round(float(probe[ends].mean()), 4),
               'cka_per_layer': [round(float(x), 4) for x in cka],
               'probe_per_layer': [round(float(x), 4) for x in probe]},
        'E1': {'acc': round(acc, 4),
               'null_p95': round(float(np.percentile(na, 95)), 4),
               'null_mean': round(float(na.mean()), 4),
               'verdict': e1_verdict},
        'E2': {'margin': round(m_obs, 4),
               'null_p95': round(float(np.percentile(nm, 95)), 4),
               'null_mean': round(float(nm.mean()), 4),
               'verdict': e2_verdict},
        'E3': {'concept_pair_acc': round(acc_concept, 4),
               'concept_null_p95': round(acc_concept_p95, 4),
               'layer_mean_abs_profile':
                   [round(float(x), 4) for x in layer_profile]},
        'E4': {'cos_langdir_window_dirs': e4_win,
               'cos_langdir_mid_dirs': e4_mid},
        'WC': {'qwen_2887_reference': QWEN_2887_ACC,
               'delta': round(acc - QWEN_2887_ACC, 4), 'verdict': wc},
        'final_verdict': 'v1=%s/E1=%s/E2=%s => '
                         'language_axis_in_mlp_glm4=%s (%s)'
                         % (v1, e1_verdict, e2_verdict, verdict, wc),
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'language_axis_glm4.npz'),
        B3_lang=B3_lang.astype(np.float32),
        g_same=g['same'].astype(np.float32),
        g_func=g['func'].astype(np.float32),
        g_null=g['null'].astype(np.float32),
        labels_lang=lab_lang, labels_concept=lab_concept,
        lang_dir=lang_dir.astype(np.float32),
        S_last=S_last.astype(np.float32),
        words=np.array(['%s:%s:%s' % w for w in words], dtype=object))

    log('==== VERDICTS ====')
    log('acc=%.4f null_p95=%.4f | margin=%.4f null_p95=%.4f'
        % (acc, np.percentile(na, 95), m_obs, np.percentile(nm, 95)))
    log('concept acc=%.4f p95=%.4f | WC: %s'
        % (acc_concept, acc_concept_p95, wc))
    log('v1=%s E1=%s E2=%s' % (v1, e1, e2))
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
