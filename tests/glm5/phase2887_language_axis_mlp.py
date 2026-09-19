# -*- coding: utf-8 -*-
"""Phase 2887: language-identity axis enters the mlp atlas (4th family).

Background chain: 2878 (qwen tied) + 2885 (DS7B untied) established
that NO unified cross-lingual axis exists at the endpoint unembed rows
(VG2b ~ 0, model-general negative N1/N4).  2886 killed the mid-layer
stripping story (language probe = 1.0 at EVERY layer) and left the
language direction extractable mid-stream: the en-vs-fr sentence
centroid difference is perfectly language-readable at li 12-24.
Question: does a language-identity DIRECTION extracted from the
mid-stream sentence states act as an mlp carrier direction like the
class/attr/syntax axes (2861/2877/2879)?  I.e. is "language" a 4th
axis family living in the same mlp response channel?

Direction source (frozen, zero forward): 2886 npz S_last (80, 37, 2560),
labels 0=en 1=fr.  lang_dir = unit(mean(S_last[en, LI_LANG])
- mean(S_last[fr, LI_LANG])) with LI_LANG = 18 (mid third centre,
frozen).  This is a NEW direction provenance (sentence-state mid-stream,
not unembed difference) - registered as such.

Vocab: 2878 qwen translation pairs, transcribed to canonical
(en, L) form (2878 stored mixed polarity): lang_fr 20, lang_de 22,
lang_es 30 pairs; VG0 homograph exclusion (2878 frozen list) then
single-token filtering, both verbatim 2878 logic.  Dedup at tid level
across languages (an English word paired into fr/de/es appears once;
its L partners are distinct) to prevent same-word-copy leakage in the
LOO retrieval.  Language labels: en=0, L=1 (fr+de+es pooled: the
direction is en-vs-fr, the test is en vs non-English).

Protocol (2879 verbatim transplant): conds same/func('the')/null,
pos 1 2-token forward; same-context word = another word of the SAME
language (min tid); cdir = lang_dir for every window layer;
g_direct[w, q] = [mlp(mlpin + eps*cdir) - mlp(mlpin)] . cdir / eps,
eps = 1.0, layers L26-35.  B3_lang = g(same) - 0.5*(g(func)+g(null)).

Prereg (frozen before any readout; execution.json written first):
  v1  determinism: mlp recompute twice identical input, max rel err
      < 1e-6 else all void.
  E1  language retrieval: acc(loo-NN on zmat(B3_lang), lab_lang) >
      null p95 (200 label permutations, SEED=2887)
      => mlp_carries_language_axis.
  E2  same-language cos margin > null p95 => language_margin_in_mlp.
  E3  descriptive: concept-pair retrieval on same spectrum (labels =
      concept index of the 35 pairs; if concept acc is also high the
      language-direction spectrum partly encodes concept too);
      per-layer mean |B3_lang| profile.
  E4  descriptive (zero forward): per-layer en-fr centroid contrast
      of lang_dir on 2886 sentence states = cos(lang_dir_q, cdir)
      for q in window, where lang_dir_q uses layer li = window layer
      mapping (last-token states li are residuals entering layer li;
      we compare cdir against directions from li 26..35 and 12..24
      mid third).
  verdict language_axis_in_mlp iff v1 and E1 and E2.
SEED=2887.  Output: phase2887/language_axis_mlp/.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2886 = os.path.join(BASE, 'phase2886', 'hourglass_cka',
                        'hourglass_cka.npz')
OUT = os.path.join(BASE, 'phase2887', 'language_axis_mlp')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2887
N_NULL = 200
EPS = 1.0
WIN_LO, WIN_HI = 26, 36
LI_LANG = 18

PREREG = {
    'direction': 'lang_dir = unit(mean S_last[en,18] - mean '
                 'S_last[fr,18]) from 2886 npz (mid-stream sentence '
                 'states; new provenance, registered)',
    'vocab': '2878 qwen translation pairs canonical (en,L): fr20/de22/'
             'es30, VG0 exclusion + single-token filter (verbatim 2878); '
             'tid-level dedup across languages; labels en=0, L=1 pooled',
    'v1': 'max relative mlp recompute error < 1e-6, else all void',
    'E1': 'acc(loo-NN zmat B3_lang, lab_lang) > null p95 (200 perms, '
          'SEED=2887) => mlp_carries_language_axis',
    'E2': 'same-language cos margin > null p95 => language_margin_in_mlp',
    'E3': 'descriptive: concept-pair retrieval acc + layer profile',
    'E4': 'descriptive: cos(lang_dir, layer-q sentence direction) '
          'profile, zero forward',
    'verdict': 'language_axis_in_mlp iff v1 and E1 and E2',
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
        json.dump({'phase': 2887, 'name': 'language_axis_mlp',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'direction_2886': sha8(SRC_2886),
                               'vocab': 'transcribed from 2878 script '
                                        'AXES (canonical en,L form), '
                                        'VG0+VG1 logic verbatim'},
                   'prereg': PREREG, 'seed': SEED, 'eps': EPS,
                   'window': [WIN_LO, WIN_HI - 1], 'li_lang': LI_LANG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    # ---------- direction (zero forward, from 2886) ----------
    z86 = np.load(SRC_2886, allow_pickle=True)
    S_last = z86['S_last'].astype(np.float64)      # (80, 37, 2560)
    lab86 = z86['labels']
    lang_dir = unit(S_last[lab86 == 0, LI_LANG].mean(0)
                    - S_last[lab86 == 1, LI_LANG].mean(0))
    log('lang_dir extracted from 2886 li=%d' % LI_LANG)

    # ---------- vocabulary: canonical (en, L) pairs, 2878 data ----------
    TRANS_PAIRS = [
        # lang_fr (20)
        ('cat', 'chat'), ('house', 'maison'), ('book', 'livre'),
        ('night', 'nuit'), ('king', 'roi'), ('woman', 'femme'),
        ('man', 'homme'), ('tree', 'arbre'), ('wine', 'vin'),
        ('moon', 'lune'), ('sea', 'mer'), ('sky', 'ciel'),
        ('flower', 'fleur'), ('city', 'ville'), ('war', 'guerre'),
        ('foot', 'pied'), ('water', 'eau'), ('sun', 'soleil'),
        ('milk', 'lait'), ('dog', 'chien'),
        # lang_de (22)
        ('cat', 'Katze'), ('dog', 'Hund'), ('water', 'Wasser'),
        ('house', 'Haus'), ('book', 'Buch'), ('night', 'Nacht'),
        ('sun', 'Sonne'), ('milk', 'Milch'), ('bird', 'Vogel'),
        ('fish', 'Fisch'), ('bread', 'Brot'), ('wine', 'Wein'),
        ('moon', 'Mond'), ('star', 'Stern'), ('tree', 'Baum'),
        ('fire', 'Feuer'), ('mountain', 'Berg'), ('snow', 'Schnee'),
        ('woman', 'Frau'), ('wolf', 'Wolf'), ('king', 'Koenig'),
        ('summer', 'Sommer'),
        # lang_es (30)
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
    from rdc_atlas_census import single_token_id
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=True,
        use_fast=True)
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

    # tid-level dedup across languages; concept id = en word tid
    words = []          # (lang, concept_key, word)
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
            words.append(('L', t_en, l_w))   # concept key = en tid
    n_words = len(words)
    n_en = sum(1 for x in words if x[0] == 'en')
    log('words=%d (en=%d L=%d)' % (n_words, n_en, n_words - n_en))

    lab_lang = np.array([0 if w[0] == 'en' else 1 for w in words])
    lab_concept = np.array([w[1] for w in words])
    concept_ok = True

    import torch
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers

    cap = {'mlpin': {}, 'mlpout': {}}
    handles = []

    def pre_hook(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x.dim() < 2:
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
        handles.append(layers[li].mlp.register_forward_hook(out_hook(li)))

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

    def mlp_call(li, x):
        out = layers[li].mlp(torch.tensor(
            x, device='cuda', dtype=torch.bfloat16))
        if isinstance(out, tuple):
            out = out[0]
        return out.detach().float().cpu().numpy()

    def same_ctx(i):
        # another word of the SAME language, min tid
        lang = words[i][0]
        cands = [j for j in range(n_words)
                 if words[j][0] == lang and j != i]
        return min(cands, key=lambda j: tid_map[words[j][2]])

    # ---------- measurement ----------
    g = {cn: np.zeros((n_words, WIN_HI - WIN_LO)) for cn in
         ('same', 'func', 'null')}
    v1a_err = 0.0
    hook_noise = 0.0
    rng = np.random.default_rng(SEED)
    vocab_size = 151936
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
        if (i + 1) % 20 == 0:
            log('words [%d/%d] v1a=%.2e' % (i + 1, n_words, v1a_err))

    log('v1a determinism max rel err = %.3e' % v1a_err)
    v1 = bool(v1a_err < 1e-6)

    B3_lang = g['same'] - 0.5 * (g['func'] + g['null'])

    # ---------- E1 / E2 with label-permutation null ----------
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

    # ---------- E3 descriptive: concept retrieval + layer profile ----------
    acc_concept = loo_acc(C, lab_concept) if concept_ok else float('nan')
    rng3 = np.random.default_rng(SEED + 1)
    null_c = []
    if concept_ok:
        for _ in range(N_NULL):
            null_c.append(loo_acc(C, rng3.permutation(lab_concept)))
        acc_concept_p95 = float(np.percentile(null_c, 95))
    else:
        acc_concept_p95 = float('nan')
    layer_profile = np.abs(B3_lang).mean(axis=0)

    # ---------- E4 descriptive: layer-direction contrast (zero fwd) ----
    e4 = []
    for li in range(WIN_LO, WIN_HI):
        d_q = unit(S_last[lab86 == 0, li].mean(0)
                   - S_last[lab86 == 1, li].mean(0))
        e4.append(round(float(d_q @ lang_dir), 4))
    mid_dirs = []
    for li in range(12, 25):
        d_q = unit(S_last[lab86 == 0, li].mean(0)
                   - S_last[lab86 == 1, li].mean(0))
        mid_dirs.append(round(float(d_q @ lang_dir), 4))

    e1_verdict = 'mlp_carries_language_axis' if e1 \
        else 'mlp_language_signal_absent'
    e2_verdict = 'language_margin_in_mlp' if e2 else 'margin_absent'
    verdict = bool(v1 and e1 and e2)

    res = {
        'phase': 2887, 'prereg': PREREG,
        'v1a_max_rel_err': float('%.3e' % v1a_err), 'v1': v1,
        'kernel_context_noise': float('%.3e' % hook_noise),
        'n_words': n_words, 'n_en': n_en,
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
        'E4': {'cos_langdir_window_dirs': e4,
               'cos_langdir_mid_dirs': mid_dirs},
        'final_verdict': 'v1=%s/E1=%s/E2=%s => language_axis_in_mlp=%s'
                         % (v1, e1_verdict, e2_verdict, verdict),
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'language_axis_mlp.npz'),
        B3_lang=B3_lang.astype(np.float32),
        g_same=g['same'].astype(np.float32),
        g_func=g['func'].astype(np.float32),
        g_null=g['null'].astype(np.float32),
        labels_lang=lab_lang, labels_concept=lab_concept,
        lang_dir=lang_dir.astype(np.float32),
        words=np.array(['%s:%s:%s' % w for w in words], dtype=object))

    log('==== VERDICTS ====')
    log('acc=%.4f null_p95=%.4f | margin=%.4f null_p95=%.4f'
        % (acc, np.percentile(na, 95), m_obs, np.percentile(nm, 95)))
    log('concept acc=%.4f p95=%.4f' % (acc_concept, acc_concept_p95))
    log('v1=%s E1=%s E2=%s' % (v1, e1, e2))
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
