# -*- coding: utf-8 -*-
"""Phase 2920: multi-axis word-level (head,layer) event atlas.

Why (roadmap 2920 + 2919 candidate A): the lang-axis event atlas
(2917: 24 events, maxT family-wise) established the (head,layer)
event language for ONE axis. The multi-axis theory needs the same
atlas for the attribute axes (speed/size/moisture, 2919 dirs now
ready and non-collinear) and for the concept axis (2887
labels_concept: 22 cross-language concept classes over the same
57 words). Plus the cross-protocol transfer check (2919 candidate
B): do sentence-level axis directions (2919 dirs_all) align with
word-level axis directions built from single-token presentation?

Mode: forward per-head jacobian (2917 verbatim), 110 single-token
words = 57 lang (2887 verbatim) + 53 attribute adjectives
(tokenizer-screened pools), conds same/func/null per word
(lang words: 2917 same-language min-tid context verbatim; attr
words: same-axis min-tid context), SEED=2896, eps=1.0, pos 1,
o_proj-input capture all 36 layers.

Jacobian families (4): family 0 = lang/concept (injection dirs =
2886 lang class-diff, 2917 verbatim float64 derivation; BOTH
partitions are applied to the SAME B - same probe, two
partitions); family 1/2/3 = speed/size/moist (injection dirs =
2919 dirs_all[1..3]).

Axes (5): lang (labels_lang 0=en,1=L), concept (labels_concept,
2887, 22 classes), speed/size/moist (pole labels, 1=HIGH
fast/huge/wet per the 2919 pole convention).

Event statistic per axis: sign-Gram margin (2910 caliber) over
the axis's own words; null = 200 label permutations (fresh
default_rng(2896) per axis; lang perms therefore 2917-identical);
per-axis maxT family (1152 cells, Westfall-Young single step,
q=0.05) PRIMARY; joint 5x1152 maxT (independent per-axis perms)
secondary descriptive.

Anchors (frozen):
  a0 dirs_all2919[lang] vs 2886-derived dirs_ref max rel < 1e-6
  a1 B_lang fresh vs 2917 npz B_heads max rel < 1e-5
  a2 |m78 - 0.28036| < 5e-3 (2917 verbatim)
  a3 sign_M_lang vs 2917 sign_M max abs diff < 1e-4 AND
     significant set (p_maxT <= 0.05) == 2917 24-cell set
  any fail => anchor_fail_all_void

Probes (frozen):
  P1 per-axis significant sets (top-15 events each)
  P2 cross-axis co-occurrence: 5x5 overlap + Jaccard + shared
     cells
  P3 transfer (4 binary axes with 2919 dirs): cos(dirs_sent,
     dirs_word) per layer; dirs_word = unit(mean HIGH - mean LOW)
     of func-cond input-to-block states at the word position
     (rows [0,36) convention); strength = median |cos| over
     layers 1..35; ok >= 0.5 / weak 0.3-0.5 / failed < 0.3
  P4 descriptive: per-axis per-layer max margin, per-head sig
     counts

Adjudication (frozen):
  anchor fail => anchor_fail_all_void;
  n_sig_attr_axes == 0 AND concept n_sig == 0 =>
     nonlang_events_absent;
  else => multi_axis_word_atlas_ready (transfer classification
     reported per axis in P3; it does not gate the word-internal
     atlas verdict).

Output: phase2920/multiaxis_word_atlas/.
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
SRC_2887 = os.path.join(BASE, 'phase2887', 'language_axis_mlp',
                        'language_axis_mlp.npz')
SRC_2917 = os.path.join(BASE, 'phase2917', 'event_atlas',
                        'event_atlas.npz')
SRC_2919 = os.path.join(BASE, 'phase2919',
                        'multiaxis_direction_families',
                        'multiaxis_families.npz')
OUT = os.path.join(BASE, 'phase2920', 'multiaxis_word_atlas')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2920_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
EPS = 1.0
VOCAB = 151936
N_PERM = 200
NH, HD, NKV = 32, 128, 8
NL = 36
SIG_Q = 0.05
RECON_REF = 0.28036
OK_COS = 0.5
WEAK_COS = 0.3
AXES = ['lang', 'concept', 'speed', 'size', 'moist']
POOLS = {
    'speed': {'hi': ['fast', 'quick', 'rapid', 'swift', 'speedy',
                     'brisk', 'fleet', 'speeding', 'flying',
                     'racing'],
              'lo': ['slow', 'sluggish', 'creeping', 'crawling',
                     'stagnant', 'idle', 'static', 'sleepy',
                     'passive', 'frozen', 'drifting']},
    'size': {'hi': ['huge', 'enormous', 'giant', 'massive',
                    'immense', 'colossal', 'gigantic', 'vast',
                    'towering'],
             'lo': ['tiny', 'small', 'little', 'miniature',
                    'minute', 'petite', 'microscopic', 'mini']},
    'moist': {'hi': ['wet', 'damp', 'humid', 'moist', 'soaked',
                     'dank', 'rainy', 'dripping'],
              'lo': ['dry', 'dusty', 'baked', 'dried', 'barren',
                     'crisp', 'thirsty']},
}
DROPPED_DESIGN = ['hasty', 'leisurely', 'plodding', 'unhurried',
                  'diminutive', 'soggy', 'drenched', 'arid',
                  'parched', 'dehydrated', 'withered', 'waterless']

PREREG = {
    'mode': 'forward per-head jacobian (2917 verbatim), 110 '
            'single-token words, 4 jacobian families (lang/concept '
            'shared probe = 2886 lang dirs; speed/size/moist '
            'probes = 2919 dirs_all[1..3]), 5 axis partitions',
    'question': 'roadmap 2920: does the (head,layer) event atlas '
                'generalize beyond the language axis - per-axis '
                'event sets, cross-axis co-occurrence (channel '
                'sharing), and word-vs-sentence direction transfer',
    'axis_design': 'lang = 2887 57 words verbatim, labels_lang '
                   '(0=en,1=L); concept = same 57 words, '
                   'labels_concept (2887, 22 classes, == ck), SAME '
                   'B as lang (same probe, second partition); '
                   'speed/size/moist = tokenizer-screened adjective '
                   'pools (1=HIGH fast/huge/wet per 2919 pole '
                   'convention), conds same-axis min-tid context; '
                   'words excluded at design time for multi-token '
                   'spaced AND bare forms: ' + ','.join(DROPPED_DESIGN),
    'null_scheme': 'per-axis fresh default_rng(2896) label '
                   'permutations x200 (lang therefore 2917-exact); '
                   'null tids: stage1 rng(2896) 57 draws excluding '
                   'lang tids (2917 verbatim), stage2 fresh '
                   'rng(2896) draws excluding all word tids',
    'family_correction': 'per-axis maxT (Westfall-Young single '
                         'step, 1152-cell family, q=0.05) PRIMARY; '
                         'joint 5x1152 maxT over independent '
                         'per-axis perms SECONDARY descriptive; '
                         'p floor 1/201 = 0.004975 << 0.05 '
                         '(granularity precheck, discipline 7)',
    'anchors': {
        'a0': 'dirs_all2919[lang] vs 2886-derived dirs_ref '
              'max rel < 1e-6 (float32 storage scale)',
        'a1': 'B_lang fresh vs 2917 npz B_heads max rel < 1e-5',
        'a2': '|m78 - 0.28036| < 5e-3',
        'a3': 'sign_M_lang vs 2917 max abs diff < 1e-4 AND '
              'sig set equality with the 2917 24 cells',
    },
    'probes': {
        'P1': 'per-axis sig sets (p_maxT <= 0.05), top-15 events',
        'P2': '5x5 sig-set overlap + Jaccard + shared cells',
        'P3': 'transfer per 4 binary axes: cos(dirs_sent, '
              'dirs_word) per layer, dirs_word from func-cond '
              'input-to-block pos-1 states, HIGH-LOW pooling '
              '(lang: en-L to match 2886 pole); median |cos| '
              'over layers 1..35; ok >= 0.5 / weak >= 0.3 / '
              'failed < 0.3',
        'P4': 'per-axis per-layer max margin + per-head counts '
              '(descriptive)',
    },
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'n_sig_attr==0 AND concept n_sig==0 => '
               'nonlang_events_absent; else => '
               'multi_axis_word_atlas_ready (P3 transfer is a '
               'preregistered secondary classification, not a '
               'verdict gate)',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def log(msg, lines):
    lines.append(msg)
    print(msg, flush=True)


def rownorm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-30)


def masks(lab):
    n = len(lab)
    eye = np.eye(n, dtype=bool)
    same = (lab[:, None] == lab[None, :]) & (~eye)
    diff = (~eye) & (~same)
    return same, diff


def margin_of(Sm, same, diff):
    return float(Sm[same].mean() - Sm[diff].mean())


def gram_margin(B, same, diff):
    U = rownorm(B)
    return margin_of(U @ U.T, same, diff)


def axis_stats(B, lab, seed):
    """2917 verbatim statistic + maxT for one axis."""
    nh = B.shape[0]
    same_m, diff_m = masks(lab)
    sign_M = np.zeros((nh, NL))
    Gs = {}
    for h in range(nh):
        for li in range(NL):
            s = np.sign(B[h, :, li])
            s[s == 0] = 1.0
            Gm = np.outer(s, s)
            Gs[(h, li)] = Gm
            sign_M[h, li] = Gm[same_m].mean() \
                - Gm[diff_m].mean()
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(lab) for _ in range(N_PERM)]
    perm_masks = [masks(pl) for pl in perms]
    p_M = np.zeros((nh, NL))
    max_perm = np.zeros(N_PERM)
    for h in range(nh):
        for li in range(NL):
            Gm = Gs[(h, li)]
            obs = sign_M[h, li]
            cnt = 0
            for pi, (sm_p, df_p) in enumerate(perm_masks):
                v = Gm[sm_p].mean() - Gm[df_p].mean()
                if v >= obs:
                    cnt += 1
                if v > max_perm[pi]:
                    max_perm[pi] = v
            p_M[h, li] = float(cnt + 1) / (N_PERM + 1)
    del Gs
    p_maxT = np.zeros((nh, NL))
    for h in range(nh):
        for li in range(NL):
            p_maxT[h, li] = float(
                np.sum(max_perm >= sign_M[h, li]) + 1) \
                / (N_PERM + 1)
    return sign_M, p_M, p_maxT, max_perm


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2920,
                   'name': 'multiaxis_word_atlas',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2886': sha8(SRC_2886),
                               's2887': sha8(SRC_2887),
                               's2917': sha8(SRC_2917),
                               's2919': sha8(SRC_2919)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'eps': EPS, 'seed': SEED, 'n_perm': N_PERM,
                   'sig_q': SIG_Q, 'axes': AXES,
                   'pools': POOLS,
                   'dropped_design': DROPPED_DESIGN,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z87 = np.load(SRC_2887, allow_pickle=True)
    words87 = [tuple(str(w).split(':')) for w in z87['words']]
    lab_lang = np.asarray(z87['labels_lang']).astype(int)
    lab_concept = np.asarray(z87['labels_concept']).astype(int)
    n_lang = len(words87)
    assert n_lang == 57
    assert len(set(lab_concept.tolist())) == 22

    z86 = np.load(SRC_2886, allow_pickle=True)
    S_last = z86['S_last'].astype(np.float64)
    lab_sent = np.asarray(z86['labels']).astype(int)
    assert all(int(lab_sent[i]) == i % 2 for i in range(80))
    diffs = S_last[lab_sent == 0].mean(0) \
        - S_last[lab_sent == 1].mean(0)
    dirs_ref = np.stack([unit(diffs[li]) for li in range(NL)])

    z17 = np.load(SRC_2917, allow_pickle=True)
    B17 = z17['B_heads'].astype(np.float64)
    sign17 = z17['sign_M'].astype(np.float64)
    pmax17 = z17['p_maxT'].astype(np.float64)
    sig17 = set((int(h), int(li))
                for h, li in zip(*np.where(pmax17 <= SIG_Q)))
    assert len(sig17) == 24

    z19 = np.load(SRC_2919, allow_pickle=True)
    dirs19 = z19['dirs_all'].astype(np.float64)
    assert [str(x) for x in z19['axis_names']] \
        == ['lang', 'speed', 'size', 'moist']
    log('sources ok', lines)

    # a0 (pre-forward)
    rel_a0 = float(np.abs(dirs19[0] - dirs_ref).max()
                   / max(float(np.abs(dirs_ref).max()), 1e-30))
    a0_ok = bool(rel_a0 < 1e-6)
    log('a0 dirs2919[lang] vs 2886-derived rel %.2e ok=%s'
        % (rel_a0, a0_ok), lines)

    # ---------- model ----------
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import load_native

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t,
                      add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t,
                          add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                raise AssertionError(
                    'multi-token pool word: %s' % t)
            tc[t] = int(ids[0])
        return tc[t]

    tid_map = {}
    for (_, ck, w) in words87:
        tid_map[w] = tid(w)
        wlang = [x[0] for x in words87
                 if x[2] == w][0]
        if wlang == 'en':
            assert tid_map[w] == int(ck), 'key mismatch %s' % w
    func_tid = tid('the')

    # attr registry: family 1=speed 2=size 3=moist; pole 1=HIGH
    fam_words = {0: [(x[0], -1, x[2]) for x in words87]}
    for fi, ax in ((1, 'speed'), (2, 'size'), (3, 'moist')):
        lst = []
        for pole, tag in ((1, 'hi'), (0, 'lo')):
            for t in POOLS[ax][tag]:
                tid_map[t] = tid(t)
                lst.append((ax, pole, t))
        fam_words[fi] = lst
    fam_sizes = {f: len(v) for f, v in fam_words.items()}
    assert fam_sizes[0] == 57
    for f in (1, 2, 3):
        hi_n = sum(1 for (_, p, _) in fam_words[f] if p == 1)
        lo_n = fam_sizes[f] - hi_n
        assert hi_n >= 5 and lo_n >= 5, (f, hi_n, lo_n)
    n_all = sum(fam_sizes.values())
    log('registry: %s (total %d)'
        % (fam_sizes, n_all), lines)

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded (load_native full GPU)', lines)

    cap = {'attnin': {}, 'oprin': {}}
    state = {'capture': False}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            cap['attnin'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())
        return h

    def pre_opro(li):
        def h(module, args, kwargs):
            if not state['capture']:
                return
            x = args[0]
            cap['oprin'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())
        return h

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li), with_kwargs=True))
        handles.append(
            layers[li].self_attn.o_proj
            .register_forward_pre_hook(
                pre_opro(li), with_kwargs=True))

    def clear_cap():
        for dd in cap:
            for li in cap[dd]:
                del cap[dd][li][:]

    def forward2(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap['attnin'][li][0]
                for li in cap['attnin']}

    vdt = next(layers[0].mlp.parameters()).dtype

    def layer_dev(li):
        return next(layers[li].mlp.parameters()).device

    rotary = model.model.rotary_emb

    def attn_call(li, X):
        t = torch.tensor(X, device=layer_dev(li), dtype=vdt)
        position_ids = torch.arange(
            t.shape[1], device=layer_dev(li)).unsqueeze(0)
        cap['oprin'].pop(li, None)
        state['capture'] = True
        with torch.no_grad():
            pos_emb = rotary(t, position_ids)
            o = layers[li].self_attn(
                t, position_embeddings=pos_emb,
                attention_mask=None, past_key_values=None)
        state['capture'] = False
        if isinstance(o, tuple):
            o = o[0]
        c = np.asarray(cap['oprin'][li][0][0, 1]) \
            .astype(np.float64)
        return c

    # ---------- null tids ----------
    lang_tids = set(tid_map[x[2]] for x in words87)
    rng = np.random.default_rng(SEED)
    null_lang = []
    while len(null_lang) < n_lang:
        r = int(rng.integers(0, VOCAB))
        if r not in lang_tids and r > 0:
            null_lang.append(r)
    all_tids = set()
    for f in range(4):
        for (_, _, w) in fam_words[f]:
            all_tids.add(tid_map[w])
    rng = np.random.default_rng(SEED)
    null_attr = []
    while len(null_attr) < n_all - n_lang:
        r = int(rng.integers(0, VOCAB))
        if r not in all_tids and r > 0:
            null_attr.append(r)
    log('null tids ready (%d + %d)'
        % (len(null_lang), len(null_attr)), lines)

    # ---------- context choices ----------
    def same_ctx_fam(f, i):
        pool = fam_words[f]
        wi = pool[i][2]
        cands = [j for j in range(len(pool))
                 if pool[j][2] != wi]
        if f == 0:
            wlang = pool[i][0]
            cands = [j for j in cands if pool[j][0] == wlang]
        return min(cands, key=lambda j: tid_map[pool[j][2]])

    # ---------- injection dirs per family ----------
    fam_dirs = {0: dirs_ref}
    for fi, ax_idx in ((1, 1), (2, 2), (3, 3)):
        fam_dirs[fi] = dirs19[ax_idx].copy()

    # ---------- forward capture ----------
    r_c = {f: {cn: np.zeros((fam_sizes[f], NL, NH, HD),
                            dtype=np.float32)
               for cn in ('same', 'func', 'null')}
           for f in range(4)}
    H_func = np.zeros((n_all, NL, 2560), dtype=np.float32)
    gi = 0
    for f in range(4):
        pool = fam_words[f]
        base = sum(fam_sizes[g] for g in range(f))
        for i, (_, _, w) in enumerate(pool):
            w_tid = tid_map[w]
            if f == 0:
                ntid = null_lang[i]
            else:
                ntid = null_attr[gi]
                gi += 1
            ctx = tid_map[pool[same_ctx_fam(f, i)][2]]
            conds = {'same': [ctx, w_tid],
                     'func': [func_tid, w_tid],
                     'null': [ntid, w_tid]}
            for cn, toks in conds.items():
                attnin_all = forward2(toks)
                if cn == 'func':
                    for li in range(NL):
                        H_func[base + i, li] = \
                            attnin_all[li][0, 1]
                for li in range(NL):
                    x = attnin_all[li]
                    ref = attn_call(li, x)
                    xp = x.copy()
                    xp[0, 1] = xp[0, 1] + EPS * fam_dirs[f][li]
                    pert = attn_call(li, xp)
                    r_c[f][cn][i, li] = (
                        (pert - ref) / EPS) \
                        .reshape(NH, HD).astype(np.float32)
        log('family %d done (%d words)' % (f, fam_sizes[f]),
            lines)
    log('forward capture complete', lines)

    # ---------- B per family ----------
    G3 = {}
    B = {}
    for f in range(4):
        r_comb = (r_c[f]['same'].astype(np.float64)
                  - 0.5 * r_c[f]['func'].astype(np.float64)
                  - 0.5 * r_c[f]['null'].astype(np.float64))
        Gf = np.zeros((NL, NH * HD))
        for li in range(NL):
            Wo_li = layers[li].self_attn.o_proj.weight \
                .detach().float().cpu().numpy() \
                .astype(np.float64)
            Gf[li] = Wo_li.T @ fam_dirs[f][li]
        G3[f] = Gf.reshape(NL, NH, HD)
        B[f] = np.einsum('nlhk,lhk->hnl', r_comb, G3[f])
        del r_comb
    log('B families: %s'
        % {f: B[f].shape for f in range(4)}, lines)

    # ---------- anchors a1/a2 ----------
    blk = B[0][:, :, 26:36]
    rel_a1 = float(np.abs(blk - B17[:, :, 26:36]).max()
                   / max(float(np.abs(B17).max()), 1e-30))
    a1_ok = bool(rel_a1 < 1e-5)
    same_m, diff_m = masks(lab_lang)
    m78 = gram_margin(blk[7] + blk[8], same_m, diff_m)
    a2_ok = bool(abs(m78 - RECON_REF) < 5e-3)
    log('a1 rel vs 2917 %.2e ok=%s | a2 m78 %.6f ok=%s'
        % (rel_a1, a1_ok, m78, a2_ok), lines)

    # ---------- per-axis statistics ----------
    axis_def = [
        ('lang', 0, lab_lang),
        ('concept', 0, lab_concept),
        ('speed', 1, np.array([p for (_, p, _)
                               in fam_words[1]])),
        ('size', 2, np.array([p for (_, p, _)
                              in fam_words[2]])),
        ('moist', 3, np.array([p for (_, p, _)
                               in fam_words[3]])),
    ]
    sign_M = {}
    p_M = {}
    p_maxT = {}
    max_perms = {}
    for ai, (ax, f, lab) in enumerate(axis_def):
        sm, pm, pt, mp = axis_stats(B[f], lab, SEED)
        sign_M[ax] = sm
        p_M[ax] = pm
        p_maxT[ax] = pt
        max_perms[ax] = mp
        log('axis %s: n_sig(maxT q=0.05) = %d'
            % (ax, int(np.sum(pt <= SIG_Q))), lines)
    global_max = np.max(np.stack(
        [max_perms[ax] for ax in AXES], axis=0), axis=0)
    p_joint = {ax: (np.sum(global_max[:, None, None]
                           >= sign_M[ax], axis=0)
                    + 1) / (N_PERM + 1)
               for ax in AXES}

    # ---------- anchor a3 ----------
    d_sign = float(np.abs(sign_M['lang'] - sign17).max())
    sig_fresh = set((int(h), int(li))
                    for h, li in zip(*np.where(
                        p_maxT['lang'] <= SIG_Q)))
    a3_ok = bool(d_sign < 1e-4 and sig_fresh == sig17)
    log('a3 sign_M diff %.2e | fresh sig %d/24 set_eq %s ok=%s'
        % (d_sign, len(sig_fresh), sig_fresh == sig17, a3_ok),
        lines)
    anchor_ok = bool(a0_ok and a1_ok and a2_ok and a3_ok)

    # ---------- verdict ----------
    verdict = None
    p1 = p2 = p3 = p4 = None
    save = {}
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # P1 per-axis sig sets
        p1 = {}
        for ax in AXES:
            pt = p_maxT[ax]
            sm = sign_M[ax]
            cells = [(int(h), int(li), float(sm[h, li]),
                      float(pt[h, li]),
                      float(p_joint[ax][h, li]))
                     for h, li in zip(*np.where(pt <= SIG_Q))]
            cells.sort(key=lambda t: -t[2])
            p1[ax] = {
                'n_sig': len(cells),
                'top15': [{'head': h, 'layer': li,
                           'sign_margin': round(v, 5),
                           'p_maxT': round(pt_, 6),
                           'p_joint': round(pj, 6)}
                          for (h, li, v, pt_, pj)
                          in cells[:15]]}
            log('P1 %s n_sig=%d top=%s'
                % (ax, len(cells),
                   [(e['head'], e['layer'],
                     e['sign_margin'])
                    for e in p1[ax]['top15'][:8]]), lines)
        n_sig_attr = sum(1 for ax in ('speed', 'size', 'moist')
                         if p1[ax]['n_sig'] > 0)
        n_sig_concept = p1['concept']['n_sig']

        # P2 co-occurrence
        sig_sets = {ax: set((h, li) for h, li in
                            zip(*np.where(
                                p_maxT[ax] <= SIG_Q)))
                    for ax in AXES}
        ov = np.zeros((5, 5), dtype=int)
        jac = np.zeros((5, 5))
        shared = {}
        for i, a in enumerate(AXES):
            for j, b in enumerate(AXES):
                inter = sig_sets[a] & sig_sets[b]
                ov[i, j] = len(inter)
                union = sig_sets[a] | sig_sets[b]
                jac[i, j] = len(inter) / len(union) \
                    if union else 0.0
                if j > i and inter:
                    shared['%s-%s' % (a, b)] = \
                        sorted(inter)
        p2 = {'axes': AXES,
              'overlap': ov.tolist(),
              'jaccard': [[round(float(x), 4) for x in row]
                          for row in jac],
              'shared': {k: [list(c) for c in v]
                         for k, v in shared.items()}}
        log('P2 overlap diag=%s | offdiag total shared=%d'
            % ([p1[ax]['n_sig'] for ax in AXES],
               sum(len(v) for v in shared.values())), lines)

        # P3 transfer
        Hf = H_func.astype(np.float64)
        row0 = 0
        trans = {}
        cos_curve = np.zeros((4, NL))
        dirs_word = np.zeros((4, NL, 2560))
        for ai, ax in enumerate(('lang', 'speed', 'size',
                                 'moist')):
            f, lab = (0, lab_lang) if ax == 'lang' \
                else ( {'speed': 1, 'size': 2, 'moist': 3}[ax],
                       np.array([p for (_, p, _)
                                 in fam_words[
                                     {'speed': 1, 'size': 2,
                                      'moist': 3}[ax]]]) )
            base = sum(fam_sizes[g] for g in range(f))
            hi_m = lab == 1
            lo_m = lab == 0
            if ax == 'lang':
                hi_m = lab_lang == 0
                lo_m = lab_lang == 1
            for li in range(NL):
                dv = unit(Hf[base:base + fam_sizes[f]][hi_m]
                          .mean(0)[li]
                          - Hf[base:base + fam_sizes[f]][lo_m]
                          .mean(0)[li])
                dirs_word[ai, li] = dv
                cos_curve[ai, li] = float(
                    dirs19[{'lang': 0, 'speed': 1,
                            'size': 2, 'moist': 3}[ax], li]
                    @ dv)
            med = float(np.median(
                np.abs(cos_curve[ai, 1:36])))
            cls = ('ok' if med >= OK_COS
                   else 'weak' if med >= WEAK_COS
                   else 'failed')
            trans[ax] = {'median_abs_cos_l1_35': round(med, 4),
                         'class': cls,
                         'argmax_layer':
                             int(np.argmax(
                                 np.abs(cos_curve[ai])))}
            log('P3 %s: median |cos| L1-35 = %.4f (%s)'
                % (ax, med, cls), lines)
        p3 = {'transfer': trans,
              'cos_curve': [[round(float(x), 4)
                             for x in cos_curve[ai]]
                            for ai in range(4)],
              'thresholds': {'ok': OK_COS, 'weak': WEAK_COS}}

        # P4 descriptive
        p4 = {'per_axis_per_layer_max': {
                  ax: [round(float(sign_M[ax][:, li].max()), 4)
                       for li in range(NL)]
                  for ax in AXES},
              'per_axis_sig_head_counts': {
                  ax: {str(h): int(c) for h, c in
                       sorted(zip(*np.unique(
                           [h for (h, li) in sig_sets[ax]],
                           return_counts=True)),
                           key=lambda t: -t[1])}
                  for ax in AXES}}

        # verdict
        if n_sig_attr == 0 and n_sig_concept == 0:
            verdict = 'nonlang_events_absent'
        else:
            verdict = 'multi_axis_word_atlas_ready'

        save = {
            'B_lang': B[0].astype(np.float32),
            'B_speed': B[1].astype(np.float32),
            'B_size': B[2].astype(np.float32),
            'B_moist': B[3].astype(np.float32),
            'sign_M': np.stack([sign_M[ax] for ax in AXES])
                .astype(np.float32),
            'p_M': np.stack([p_M[ax] for ax in AXES])
                .astype(np.float32),
            'p_maxT': np.stack([p_maxT[ax] for ax in AXES])
                .astype(np.float32),
            'p_joint': np.stack([p_joint[ax] for ax in AXES])
                .astype(np.float32),
            'max_perm': np.stack([max_perms[ax]
                                  for ax in AXES])
                .astype(np.float32),
            'global_max_perm': global_max.astype(np.float32),
            'dirs_used': np.stack(
                [dirs_ref, dirs_ref, dirs19[1], dirs19[2],
                 dirs19[3]]).astype(np.float32),
            'dirs_word': dirs_word.astype(np.float32),
            'cos_curve': cos_curve.astype(np.float32),
            'labels_lang': lab_lang,
            'labels_concept': lab_concept,
            'labels_speed': axis_def[2][2],
            'labels_size': axis_def[3][2],
            'labels_moist': axis_def[4][2],
            'words_lang': np.array(
                ['%s:%s:%s' % x for x in words87],
                dtype=object),
            'words_speed': np.array(
                ['pole%d:%s' % (p, w) for (_, p, w)
                 in fam_words[1]], dtype=object),
            'words_size': np.array(
                ['pole%d:%s' % (p, w) for (_, p, w)
                 in fam_words[2]], dtype=object),
            'words_moist': np.array(
                ['pole%d:%s' % (p, w) for (_, p, w)
                 in fam_words[3]], dtype=object),
            'axis_names': np.array(AXES, dtype=object),
        }

    log('==== VERDICT: %s ====' % verdict, lines)
    res = {
        'phase': 2920, 'model': 'qwen3-4b', 'prereg': PREREG,
        'anchors': {'a0_dirs_rel': float('%.3e' % rel_a0),
                    'a0_ok': a0_ok,
                    'a1_rel_vs_2917': float('%.3e' % rel_a1),
                    'a1_ok': a1_ok,
                    'a2_m78': round(m78, 6),
                    'a2_ref': RECON_REF, 'a2_ok': a2_ok,
                    'a3_sign_diff': float('%.3e' % d_sign),
                    'a3_sig_set_eq': bool(sig_fresh == sig17),
                    'a3_ok': a3_ok, 'ok': anchor_ok},
        'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(
            os.path.join(OUT, 'multiaxis_word_atlas.npz'),
            **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2920 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
