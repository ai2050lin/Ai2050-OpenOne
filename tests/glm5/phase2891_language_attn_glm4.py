# -*- coding: utf-8 -*-
"""Phase 2891: GLM4 language attention-route discrimination (N6 followup).

N6 (2890, settled): glm4's mid-stream language direction produces NO
mlp direct-write response (acc at null mean), while language identity
persists in the residual stream (probe >= 0.96 all layers).  Open
candidate tested here: on glm4 the language direction routes through
the ATTENTION channel instead.

Protocol (strict parallel to 2890, injection site mlp -> attention):
  direction: lang_dir reused verbatim from 2890 npz (GLM4 S_last
  li=20, sha frozen) - zero forward, same direction so the mlp/attn
  contrast is internally valid.
  vocab: 78 words (29 en / 49 L) reused verbatim from 2890 npz (words,
  labels_lang, labels_concept), re-tokenized with the glm4 tokenizer.
  window: WIN=[28,40) (2888 frozen rule).
  conds same/func('the')/null, pos-1 two-token forward; same-context
  word = another word of the SAME language (min tid).
  g_attn[i,q] = [(self_attn(attnin + eps*cdir) - self_attn(attnin))
                 . cdir] / eps, injection at the TARGET position (1)
  only; cdir = lang_dir; attnin = self_attn input captured at pos
  [1,seq,d] by pre_hook; direct call path probed OK (rel err 0.0 vs
  hook, probe_2891_attn.txt): pos_emb = model.model.rotary_emb(attnin,
  position_ids); layer.self_attn(attnin, position_embeddings=pos_emb,
  attention_mask=None, past_key_values=None).
  B3_attn = g(same) - 0.5*(g(func)+g(null)).

Prereg (frozen before any forward):
  v1  determinism: attn direct-call recompute twice identical input,
      max rel err < 1e-6 else all void; plus kernel-vs-hook baseline
      noise recorded (descriptive).
  A1  language retrieval: acc(loo-NN on zmat(B3_attn), lab_lang) >
      null p95 (200 label permutations, SEED=2891)
      => attn_carries_language_axis_glm4.
  A2  same-language cos margin > null p95 => language_margin_in_attn_glm4.
  A3  descriptive: concept-pair retrieval on same spectrum.
  AC  descriptive route contrast: acc_attn vs M2890 mlp acc 0.5256
      and qwen mlp acc 0.7719 (reference only, no gate).
  verdict attn_language_route_glm4 iff v1 and A1 and A2.
SEED=2891.  Output: phase2891/language_attn_glm4/.
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
OUT = os.path.join(BASE, 'phase2891', 'language_attn_glm4')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\glm4-9b-chat-hf'
SEED = 2891
N_NULL = 200
EPS = 1.0
MLP_2890_ACC = 0.5256
QWEN_MLP_ACC = 0.7719

PREREG = {
    'direction': 'lang_dir reused verbatim from 2890 npz (GLM4 S_last '
                 'li=20; sha74933303); zero forward',
    'vocab': '78 words / labels reused verbatim from 2890 npz, '
             're-tokenized with glm4 tokenizer',
    'injection': 'g_attn[i,q] = [(self_attn(attnin + eps*cdir) - '
                 'self_attn(attnin)) . cdir] / eps at target pos 1 '
                 'only; direct-call path (probe_2891 rel err 0.0); '
                 'strict parallel to 2890 mlp g_direct',
    'window': 'WIN=[28,40) (2888 frozen rule)',
    'v1': 'attn recompute rel err < 1e-6, else all void',
    'A1': 'acc(loo-NN zmat B3_attn, lab_lang) > null p95 (200 perms, '
          'SEED=2891) => attn_carries_language_axis_glm4',
    'A2': 'same-language cos margin > null p95 => '
          'language_margin_in_attn_glm4',
    'A3': 'descriptive: concept-pair retrieval acc',
    'AC': 'descriptive: acc_attn vs mlp 0.5256 (M2890) vs qwen mlp '
          '0.7719; route contrast, no gate',
    'verdict': 'attn_language_route_glm4 iff v1 and A1 and A2',
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
        json.dump({'phase': 2891, 'name': 'language_attn_glm4',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'direction_vocab_2890': sha8(SRC_2890)},
                   'model': 'glm4-9b-chat-hf',
                   'prereg': PREREG, 'seed': SEED, 'eps': EPS},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # ---------- reuse direction + vocab from 2890 (zero forward) ----
    z90 = np.load(SRC_2890, allow_pickle=True)
    lang_dir = z90['lang_dir'].astype(np.float64)
    words = [tuple(str(w).split(':')) for w in z90['words']]
    lab_lang = z90['labels_lang']
    lab_concept = z90['labels_concept']
    n_words = len(words)
    n_en = int((lab_lang == 0).sum())
    log('lang_dir + %d words (en=%d L=%d) reused from 2890'
        % (n_words, n_en, n_words - n_en))

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
    log('vocab re-tokenized: %d unique words, func=%d'
        % (len(tid_map), func_tid))

    # ---------- model load (2888 Gen2 device_map) ----------
    cfg = json.load(io.open(os.path.join(MD, 'config.json'),
                            encoding='utf-8'))
    L = int(cfg['num_hidden_layers'])
    WIN_LO = int(np.floor(26 / 36.0 * L))
    WIN_HI = L
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
        for p in layers[li_].self_attn.parameters():
            assert p.device.type != 'meta', \
                'meta tensor in window layer %d' % li_

    # ---------- hooks: ln-independent attnin + attnout ----------
    cap = {'attnin': {}, 'attnout': {}}
    handles = []

    def pre_hook(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            cap['attnin'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())
        return h

    def out_hook(li):
        def h(module, args, output):
            o = output[0] if isinstance(output, tuple) else output
            if o.dim() < 2:
                return
            cap['attnout'].setdefault(li, []).append(
                o.detach()[0, 1].float().cpu().numpy())
        return h

    for li in range(WIN_LO, WIN_HI):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_hook(li), with_kwargs=True))
        handles.append(layers[li].self_attn
                       .register_forward_hook(out_hook(li)))

    def clear_cap():
        for dd in cap:
            for li in cap[dd]:
                del cap[dd][li][:]

    def forward2(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap['attnin'][li][0] for li in cap['attnin']}, \
            {li: cap['attnout'][li][0] for li in cap['attnout']}

    vdt = next(layers[WIN_LO].self_attn.parameters()).dtype

    def layer_dev(li):
        return next(layers[li].self_attn.parameters()).device

    rotary = model.model.rotary_emb

    def attn_call(li, X, probe_pos1=False):
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
        out_full = o[0].detach().float().cpu().numpy().astype(
            np.float64)
        return (out_full[1], out_full) if probe_pos1 else out_full[1]

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

    ldt_all = torch.tensor(lang_dir, dtype=torch.float64)
    for i, (_, _, w) in enumerate(words):
        w_tid = tid_map[w]
        conds = {'same': [tid_map[words[same_ctx(i)][2]], w_tid],
                 'func': [func_tid, w_tid],
                 'null': [null_tids[i], w_tid]}
        for cn, toks in conds.items():
            attnin_all, attnout_all = forward2(toks)
            for q, li in enumerate(range(WIN_LO, WIN_HI)):
                Xin = attnin_all[li]           # [seq, d]
                ref_pos1, ref_full = attn_call(li, Xin,
                                               probe_pos1=True)
                if cn == 'same' and q == 0:
                    ref2 = attn_call(li, Xin)
                    denom = max(float(np.linalg.norm(ref_pos1)), 1e-30)
                    v1a_err = max(v1a_err, float(np.linalg.norm(
                        ref_pos1 - ref2)) / denom)
                    hook_ref = attnout_all[li]
                    hook_noise = max(hook_noise, float(np.linalg.norm(
                        ref_pos1 - hook_ref))
                        / max(float(np.linalg.norm(hook_ref)), 1e-30))
                Xp = Xin.copy()
                Xp[0, 1] = Xp[0, 1] + EPS * lang_dir
                out_p = attn_call(li, Xp)
                g[cn][i, q] = float((out_p - ref_pos1) @ lang_dir) / EPS
        if (i + 1) % 10 == 0:
            log('words [%d/%d] v1a=%.2e' % (i + 1, n_words, v1a_err))

    log('v1 determinism max rel err = %.3e' % v1a_err)
    v1 = bool(v1a_err < 1e-6)

    B3_attn = g['same'] - 0.5 * (g['func'] + g['null'])

    # ---------- A1 / A2 ----------
    C = zmat(B3_attn)
    acc = loo_acc(C, lab_lang)
    U = B3_attn / np.maximum(
        np.linalg.norm(B3_attn, axis=1, keepdims=True), 1e-30)
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
    a1 = bool(acc > float(np.percentile(na, 95)))
    a2 = bool(m_obs > float(np.percentile(nm, 95)))

    # ---------- A3 ----------
    acc_concept = loo_acc(C, lab_concept)
    rng3 = np.random.default_rng(SEED + 1)
    null_c = []
    for _ in range(N_NULL):
        null_c.append(loo_acc(C, rng3.permutation(lab_concept)))
    acc_concept_p95 = float(np.percentile(null_c, 95))
    layer_profile = np.abs(B3_attn).mean(axis=0)

    ac = ('attn_route_above_mlp' if acc > MLP_2890_ACC
          else 'attn_route_not_above_mlp')
    a1_verdict = 'attn_carries_language_axis_glm4' if a1 \
        else 'attn_language_signal_absent_glm4'
    a2_verdict = 'language_margin_in_attn_glm4' if a2 \
        else 'margin_absent_attn_glm4'
    verdict = bool(v1 and a1 and a2)

    res = {
        'phase': 2891, 'model': 'glm4-9b-chat-hf', 'prereg': PREREG,
        'n_words': n_words, 'n_en': n_en, 'window': [WIN_LO, WIN_HI],
        'v1a_max_rel_err': float('%.3e' % v1a_err), 'v1': v1,
        'kernel_context_noise': float('%.3e' % hook_noise),
        'A1': {'acc': round(acc, 4),
               'null_p95': round(float(np.percentile(na, 95)), 4),
               'null_mean': round(float(na.mean()), 4),
               'verdict': a1_verdict},
        'A2': {'margin': round(m_obs, 4),
               'null_p95': round(float(np.percentile(nm, 95)), 4),
               'null_mean': round(float(nm.mean()), 4),
               'verdict': a2_verdict},
        'A3': {'concept_pair_acc': round(acc_concept, 4),
               'concept_null_p95': round(acc_concept_p95, 4),
               'layer_mean_abs_profile':
                   [round(float(x), 4) for x in layer_profile]},
        'AC': {'glm4_mlp_2890': MLP_2890_ACC,
               'qwen_mlp_2887': QWEN_MLP_ACC,
               'acc_attn': round(acc, 4),
               'delta_vs_mlp': round(acc - MLP_2890_ACC, 4),
               'verdict': ac},
        'final_verdict': 'v1=%s/A1=%s/A2=%s => '
                         'attn_language_route_glm4=%s (%s)'
                         % (v1, a1_verdict, a2_verdict, verdict, ac),
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'language_attn_glm4.npz'),
        B3_attn=B3_attn.astype(np.float32),
        g_same=g['same'].astype(np.float32),
        g_func=g['func'].astype(np.float32),
        g_null=g['null'].astype(np.float32),
        labels_lang=lab_lang, labels_concept=lab_concept,
        lang_dir=lang_dir.astype(np.float32),
        words=np.array(['%s:%s:%s' % w for w in words], dtype=object))

    log('==== VERDICTS ====')
    log('acc=%.4f null_p95=%.4f | margin=%.4f null_p95=%.4f'
        % (acc, np.percentile(na, 95), m_obs, np.percentile(nm, 95)))
    log('concept acc=%.4f p95=%.4f | AC: %s'
        % (acc_concept, acc_concept_p95, ac))
    log('v1=%s A1=%s A2=%s' % (v1, a1, a2))
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
