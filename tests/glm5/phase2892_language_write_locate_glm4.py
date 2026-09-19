# -*- coding: utf-8 -*-
"""Phase 2892: GLM4 language write-location (early vs deep vs passive).

N7 (2891, settled): glm4's mid-stream language direction is written by
NEITHER mlp NOR attention at the deep window [28,40).  Remaining
candidates (N7 implication): (a) written by EARLY layers (< window),
or (b) never actively written after embeddings (passive preservation).

Stage 1 (zero forward, S_last reused from 2890 npz sha 74933303):
  sentence labels reconstructed as i%2 (2886 PAIRS order, frozen
  constant copied verbatim into this script and 2890).
  sep(li)   = (mean_en S_last[:,li] - mean_fr S_last[:,li]) . lang_dir
  d(li)     = ||mean_en S_last[:,li] - mean_fr S_last[:,li]||
  Delta(li) = sep(li+1) - sep(li)  (layer-li modules attn+mlp net
              contribution along lang_dir), li = 0..39.
  Segments (frozen): EARLY = Delta(0..12), MID = Delta(13..27),
  DEEP = Delta(28..39).
  L1  attribution rule: share_s = sum |Delta| over segment s /
      sum |Delta(0..39)|; early_write_dominant if EARLY share > 0.5;
      deep_write_dominant if DEEP share > 0.5; else distributed.
  L2  passive check: if sum |Delta(28..39)| < 0.10 * sep(41 end) =>
      deep_passive_confirmed (deep window adds <10% of final
      separation).

Stage 2 (frozen rule, observed parameter): li* = argmax_{0<=li<=39}
  |Delta(li)|; W2 = [max(0, li*-2), min(L, li*+4)) (<= 6 layers).
  Dual-channel direct-injection discrimination over W2 with cdir =
  lang_dir (same direction as 2890/2891 - internal validity):
    g_mlp[i,q]  = [(mlp(mlpin + eps*cdir) - mlp(mlpin)) . cdir]/eps
    g_attn[i,q] = [(self_attn(attnin(+pos1 cdir)) -
                    self_attn(attnin)) . cdir]/eps  (pos 1 only)
    B_mlp = g(same) - 0.5(g(func)+g(null)), same for attn.
  vocab/conds identical to 2890/2891 (78 words reused, zero forward).
  A1m: acc(loo-NN zmat B_mlp, lab_lang) > null p95 (200 perms,
       SEED=2892) => mlp_carries_language_axis_in_W2
  A1a: same for attn => attn_carries_language_axis_in_W2
  A2m/A2a: same-language cos margin > null p95.
  Both-negative => no_aligned_write_in_W2 (sep growth there is not
  channel direct-write along lang_dir; passive/non-aligned).
  One positive => the language carrier exists at W2 (2887 finding
  refined: carrier present at li* region, absent in deep window).

Stage 2 executes whenever W2 is non-empty (li*=0 -> W2=[0,4)).
SEED=2892.  Output: phase2892/language_write_locate_glm4/.
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
OUT = os.path.join(BASE, 'phase2892', 'language_write_locate_glm4')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\glm4-9b-chat-hf'
SEED = 2892
N_NULL = 200
EPS = 1.0

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
    'stage1': 'zero forward: sep/d/Delta curves from 2890 S_last; '
              'sentence labels i%2 (2886 PAIRS order frozen)',
    'L1': 'segment shares: EARLY Delta(0..12), MID Delta(13..27), '
          'DEEP Delta(28..39); early_write_dominant if EARLY>0.5, '
          'deep_write_dominant if DEEP>0.5, else distributed',
    'L2': 'deep_passive_confirmed if sum|Delta(28..39)| < 0.10*sep_end',
    'stage2_rule': 'li*=argmax|Delta(li)|, li in 0..39; W2=[max(0,'
                   'li*-2), min(L, li*+4)); rule frozen before Stage 1 '
                   'observation (observed parameter, not choice)',
    'injection': 'dual-channel direct injection, cdir=lang_dir (same '
                 'as 2890/2891); g = [(module(x + eps*cdir) - '
                 'module(x)) . cdir]/eps at target pos; B = g(same) - '
                 '0.5(g(func)+g(null)); conds/vocab verbatim 2890/2891',
    'v1': 'module recompute rel err < 1e-6, else all void',
    'A1m': 'acc(loo-NN zmat B_mlp, lab_lang) > null p95 (200 perms, '
           'SEED=2892) => mlp_carries_language_axis_in_W2',
    'A1a': 'same for attn => attn_carries_language_axis_in_W2',
    'A2': 'same-language cos margin > null p95 (per channel)',
    'verdict': 'both negative => no_aligned_write_in_W2; any positive '
               '=> language carrier located at W2 channel(s)',
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
        json.dump({'phase': 2892, 'name': 'language_write_locate_glm4',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s_last_2890': sha8(SRC_2890)},
                   'model': 'glm4-9b-chat-hf',
                   'n_pairs': len(PAIRS),
                   'prereg': PREREG, 'seed': SEED, 'eps': EPS},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    # ---------- Stage 1: zero-forward write-location ----------
    z90 = np.load(SRC_2890, allow_pickle=True)
    S_last = z90['S_last'].astype(np.float64)   # (80, 41, 4096)
    lang_dir = z90['lang_dir'].astype(np.float64)
    words = [tuple(str(w).split(':')) for w in z90['words']]
    lab_lang = z90['labels_lang']
    lab_concept = z90['labels_concept']
    n_words = len(words)
    n_li = S_last.shape[1]                       # 41
    lab_sent = np.array([i % 2 for i in range(2 * len(PAIRS))])
    assert S_last.shape[0] == 2 * len(PAIRS)

    mean_en = S_last[lab_sent == 0].mean(0)      # (41, d)
    mean_fr = S_last[lab_sent == 1].mean(0)
    diff = mean_en - mean_fr
    sep = diff @ lang_dir                        # (41,)
    dnorm = np.linalg.norm(diff, axis=1)         # (41,)
    delta_sep = np.diff(sep)                     # (40,) Delta(li)
    total_abs = float(np.abs(delta_sep).sum())
    early_share = float(np.abs(delta_sep[0:13]).sum()) / total_abs
    mid_share = float(np.abs(delta_sep[13:28]).sum()) / total_abs
    deep_share = float(np.abs(delta_sep[28:40]).sum()) / total_abs
    if early_share > 0.5:
        l1 = 'early_write_dominant'
    elif deep_share > 0.5:
        l1 = 'deep_write_dominant'
    else:
        l1 = 'distributed'
    sep_end = float(sep[-1])
    deep_abs = float(np.abs(delta_sep[28:40]).sum())
    l2 = bool(deep_abs < 0.10 * abs(sep_end))
    li_star = int(np.argmax(np.abs(delta_sep)))
    L = n_li - 1
    W2_LO = max(0, li_star - 2)
    W2_HI = min(L, li_star + 4)
    log('Stage1: sep_end=%.4f shares early=%.3f mid=%.3f deep=%.3f'
        % (sep_end, early_share, mid_share, deep_share))
    log('L1=%s L2(deep_passive)=%s li*=%d W2=[%d,%d)'
        % (l1, l2, li_star, W2_LO, W2_HI))
    log('sep curve: %s' % [round(float(x), 3) for x in sep])
    log('Delta curve: %s'
        % [round(float(x), 3) for x in delta_sep])

    # ---------- Stage 2: dual-channel injection over W2 ----------
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
    for li_ in range(L):
        dm['model.layers.%d' % li_] = 0 if W2_LO <= li_ < W2_HI \
            else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(
        MD, local_files_only=True, device_map=dm)
    model.eval()
    layers = model.model.layers
    log('model loaded: window W2 %d-%d on cuda'
        % (W2_LO, W2_HI - 1))
    for li_ in range(W2_LO, W2_HI):
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

    for li in range(W2_LO, W2_HI):
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

    vdt = next(layers[W2_LO].mlp.parameters()).dtype

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

    n_win = W2_HI - W2_LO
    g = {ch: {cn: np.zeros((n_words, n_win)) for cn in
              ('same', 'func', 'null')} for ch in ('mlp', 'attn')}
    v1_err = {'mlp': 0.0, 'attn': 0.0}
    rng = np.random.default_rng(SEED)
    vocab_size = int(cfg.get('vocab_size'))
    word_tids = set(tid_map.values())
    null_tids = []
    while len(null_tids) < n_words:
        r = int(rng.integers(0, vocab_size))
        if r not in word_tids and r > 0:
            null_tids.append(r)

    for i, (_, _, w) in enumerate(words):
        w_tid = tid_map[w]
        conds = {'same': [tid_map[words[same_ctx(i)][2]], w_tid],
                 'func': [func_tid, w_tid],
                 'null': [null_tids[i], w_tid]}
        for cn, toks in conds.items():
            mlpin_all, attnin_all = forward2(toks)
            for q, li in enumerate(range(W2_LO, W2_HI)):
                x_m = mlpin_all[li]        # [1, seq, d]
                x_a = attnin_all[li]       # [1, seq, d]
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
                xm_p[0, 1] = xm_p[0, 1] + EPS * lang_dir
                g['mlp'][cn][i, q] = float(
                    (mlp_call(li, xm_p) - ref_m) @ lang_dir) / EPS
                xa_p = x_a.copy()
                xa_p[0, 1] = xa_p[0, 1] + EPS * lang_dir
                g['attn'][cn][i, q] = float(
                    (attn_call(li, xa_p) - ref_a) @ lang_dir) / EPS
        if (i + 1) % 10 == 0:
            log('words [%d/%d] v1 mlp=%.2e attn=%.2e'
                % (i + 1, n_words, v1_err['mlp'], v1_err['attn']))

    v1 = bool(v1_err['mlp'] < 1e-6 and v1_err['attn'] < 1e-6)
    log('v1: mlp=%.3e attn=%.3e pass=%s'
        % (v1_err['mlp'], v1_err['attn'], v1))

    B = {ch: g[ch]['same'] - 0.5 * (g[ch]['func'] + g[ch]['null'])
         for ch in ('mlp', 'attn')}

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
    out_c = {}
    for ch in ('mlp', 'attn'):
        C = zmat(B[ch])
        acc = loo_acc(C, lab_lang)
        U = B[ch] / np.maximum(
            np.linalg.norm(B[ch], axis=1, keepdims=True), 1e-30)
        Sm = U @ U.T
        n = len(lab_lang)
        same_m = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                same_m[i, j] = (lab_lang[i] == lab_lang[j]) and i != j
        off = ~np.eye(n, dtype=bool) & ~same_m
        m_obs = float(Sm[same_m].mean() - Sm[off].mean())
        null_acc = [loo_acc(C, pl) for pl in perms]
        null_m = [margin_of(Sm, pl) for pl in perms]
        acc_p95 = float(np.percentile(null_acc, 95))
        m_p95 = float(np.percentile(null_m, 95))
        a1 = bool(acc > acc_p95)
        a2 = bool(m_obs > m_p95)
        acc_c = loo_acc(C, lab_concept)
        out_c[ch] = {
            'acc': round(acc, 4), 'null_p95': round(acc_p95, 4),
            'null_mean': round(float(np.mean(null_acc)), 4),
            'margin': round(m_obs, 4), 'margin_p95': round(m_p95, 4),
            'concept_acc': round(acc_c, 4),
            'A1': 'carries' if a1 else 'absent',
            'A2': 'margin' if a2 else 'margin_absent',
            'layer_profile': [round(float(x), 4)
                              for x in np.abs(B[ch]).mean(axis=0)]}
        log('%s: acc=%.4f p95=%.4f margin=%.4f p95=%.4f A1=%s A2=%s'
            % (ch, acc, acc_p95, m_obs, m_p95, out_c[ch]['A1'],
               out_c[ch]['A2']))

    a1m = out_c['mlp']['A1'] == 'carries'
    a1a = out_c['attn']['A1'] == 'carries'
    if a1m and a1a:
        s2_verdict = 'language_carrier_in_W2_both_channels'
    elif a1m:
        s2_verdict = 'language_carrier_in_W2_mlp_only'
    elif a1a:
        s2_verdict = 'language_carrier_in_W2_attn_only'
    else:
        s2_verdict = 'no_aligned_write_in_W2'

    res = {
        'phase': 2892, 'model': 'glm4-9b-chat-hf', 'prereg': PREREG,
        'Stage1': {'sep_curve': [round(float(x), 4) for x in sep],
                   'delta_curve': [round(float(x), 4)
                                   for x in delta_sep],
                   'dnorm_curve': [round(float(x), 4)
                                   for x in dnorm],
                   'shares': {'early': round(early_share, 4),
                              'mid': round(mid_share, 4),
                              'deep': round(deep_share, 4)},
                   'L1': l1, 'L2_deep_passive': l2,
                   'sep_end': round(sep_end, 4),
                   'li_star': li_star, 'W2': [W2_LO, W2_HI]},
        'v1': v1, 'v1_err': {k: float('%.3e' % v)
                             for k, v in v1_err.items()},
        'Stage2': out_c,
        'final_verdict': 'L1=%s L2=%s li*=%d W2=%s | Stage2: %s'
                         % (l1, l2, li_star, [W2_LO, W2_HI],
                            s2_verdict),
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'language_write_locate_glm4.npz'),
        B_mlp=B['mlp'].astype(np.float32),
        B_attn=B['attn'].astype(np.float32),
        sep=sep.astype(np.float32),
        delta_sep=delta_sep.astype(np.float32),
        labels_lang=lab_lang, labels_concept=lab_concept,
        lang_dir=lang_dir.astype(np.float32),
        words=np.array(['%s:%s:%s' % w for w in words], dtype=object))
    log('==== VERDICT: %s ====' % res['final_verdict'])
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
