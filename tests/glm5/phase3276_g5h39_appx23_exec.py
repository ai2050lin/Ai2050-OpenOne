# -*- coding: utf-8 -*-
"""Phase 3276 (G5-H39): APPX-2/3 execution phase (owner-named 2026-10-21).

Protocol: appx23_protocol_v2.json (ce73984d, phase3267) — executed verbatim;
recipe operationalizations frozen in DESIGN before any observation (H1/H2);
per-model independence (H4, no pooling); landing only via a separate registry
phase (G5/H5 -> 3277); this phase writes zero gap/registry bytes and zero
ledger measurement rows (closeout receipt only).

Arm a (APPX-2, 5 candidate models, per-model):
  S1 rebuild S_attr (2874 rules verbatim, float32, VG1 >=5 words per axis,
     >=8/10 family pass) and S_syntax (2878 rules verbatim incl homograph
     exclusion, per-axis survival: >=5 valid pairs AND VG2b > 0.2) on W_U
     rows (tied -> embed_tokens else lm_head); VG2a/VG3 family gates
     verbatim; per-model legality, never pooled.
  S2  K_entity profile: 3157-style collect (SHARED[:16], isa/hasa +/-, ctx16,
     128 rows, last-token fp16) -> K_entity = center-SVD top8 of H[:,NL,:];
     kross(K_entity, K_readout) -> eff_ge05/top1 registered (descriptive).
  S3  K_readout = 3158-verbatim static (G = W^T W fp32 chunked -> eigh fp64,
     top64 desc); S_class = 2881 recipe per-model (up-to-8/cat single-token,
     soft material gate); kross verdicts, thresholds verbatim (>=30
     separable / <15 collinear / else weakly_separated).
  S4  synthesis vs qwen3-4b incumbents (3165: 67.0/68.094/58.488, all
     separable); conservation/divergence table.

Arm b (APPX-3, qwen3-4b + qwen3-14b + glm4, sealed 3155 collects reused):
  T1  K2 direct: frozen operationalization of the conditional component
      phi_l(c) = relation-conditional structure QC u QEC (balanced orthogonal
      ANOVA bases at KOUT, 3155 verbatim construction); separation projection
      P = I - QQ^T; response cosine drop := 1 - mean_row cos(h, Ph).
  T2  K2 behavioral half: analytic logits on the readout interface
      (logits = W_U @ h, h = post-final-norm slot NL — 3158 bitwise-verified
      protocol); B1 first-token agreement := P(argmax_patched == argmax_base)
      over the shared-panel rows; FP_GATE 0.8 (3155/3157 precedent).
      K2 deathline fires iff drop > 0.5 AND agreement >= 0.8.
  T3  K3 coverage: per family (isa/hasa/mof), per-coordinate between-entity
      signal eta2_d = between-SS_d / total-SS_d at KOUT on the 72 family
      rows; coverage := share of total between-entity SS held by the top-50
      coordinates ranked by eta2; fires iff all 3 families < 0.30.
  T4  per-model verdict registration; proxy-vs-direct documented.
Cross-checks (machinery validation, hard-fail):
  X1 4b S1 rebuild == sealed 2874/2878 dW_unit bitwise + axes list equal.
  X2 4b K_readout top64 allclose p3158 sealed top64 (atol 1e-6).
  X3 4b S_class x K_readout top1 within 0.5 deg of 3165 incumbent 67.0.
  X4 4b recomputed ExC share == sealed 3155 value (atol 1e-9).
"""
import hashlib
import io
import json
import os
import sys
import time
import traceback

import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

PHASE = 3276
NAME = 'g5h39_appx23_exec'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result',
                    'rdc_query_construction_20260913')
BASE = os.path.join(RDIR, 'phase3276', NAME)
SMOKE = os.environ.get('P3276_SMOKE', '') == '1'
LOGP = os.path.join(BASE, 'run_log.txt')
os.makedirs(BASE, exist_ok=True)


def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    with open(LOGP, 'a', encoding='utf-8') as f:
        f.write(ln + '\n')
    try:
        print(ln, flush=True)
    except Exception:
        pass


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def unit(v):
    n = float(np.linalg.norm(v))
    assert n > 0
    return v / n


# ---------------- frozen anchors ----------------
P2874 = os.path.join(RDIR, 'phase2874', 'attr_vocab_v2', 'attr_vocab_v2.npz')
P2878 = os.path.join(RDIR, 'phase2878', 'syntax_trans_vocab',
                     'syntax_trans_vocab.npz')
P3158_4B = os.path.join(RDIR, 'phase3158', 'g4p1_output_equivalence_class',
                        'qwen3-4b', 'collect.npz')
P3165_RES = os.path.join(RDIR, 'phase3165', 'g5a3_family_alignment',
                         'result.json')
P2806_EXEC = os.path.join(RDIR, 'phase2806', 'qwen4_hierarchy',
                          'execution.json')
P3155_DIR = os.path.join(RDIR, 'phase3155',
                         'g2p1_relation_family_operator_separability')
ANCHOR_SHA = {
    'protocol_v2': 'ce73984d',
    'p2874_npz': sha8(P2874),
    'p2878_npz': sha8(P2878),
    'p3158_4b_npz': sha8(P3158_4B),
    'p3165_result': sha8(P3165_RES),
    'p2806_exec': sha8(P2806_EXEC),
}
assert ANCHOR_SHA['protocol_v2'] == 'ce73984d'
assert ANCHOR_SHA['p2874_npz'] == '5f7796e1', ANCHOR_SHA['p2874_npz']
assert ANCHOR_SHA['p2878_npz'] == '0bb3ec1e', ANCHOR_SHA['p2878_npz']

MODELS_A = ['qwen2.5-3b-instruct', 'qwen2-7b', 'deepseek-r1-distill-qwen-7b',
            'glm4-9b-chat-hf', 'gemma-3-4b-it']
MODELS_B = ['qwen3-4b', 'qwen3-14b', 'glm4']
MDIR_ALIAS = {'qwen3-14b': 'Qwen3-14B', 'glm4': 'glm4-9b-chat-hf'}
MDIR = {m: os.path.join(ROOT, 'models', 'hf', MDIR_ALIAS.get(m, m))
        for m in set(MODELS_A + MODELS_B)}

# ---------------- 2874/2878 frozen material (verbatim) ----------------
AXES_ATTR = {
    'size': [('size1', 'big', 'small'), ('size2', 'large', 'little'),
             ('size3', 'huge', 'tiny')],
    'speed': [('speed1', 'fast', 'slow'), ('speed2', 'quick', 'slow'),
              ('speed3', 'rapid', 'slow'), ('speed4', 'swift', 'slow')],
    'temp': [('temp1', 'hot', 'cold'), ('temp2', 'warm', 'cool'),
             ('temp3', 'boiling', 'freezing')],
    'age': [('age1', 'old', 'new'), ('age2', 'young', 'old'),
            ('age3', 'ancient', 'modern')],
    'weight': [('weight1', 'heavy', 'light'),
               ('weight2', 'hefty', 'light'),
               ('weight3', 'heavy', 'lightweight'),
               ('weight4', 'ponderous', 'light')],
    'strength': [('strength1', 'strong', 'weak'),
                 ('strength2', 'powerful', 'weak'),
                 ('strength3', 'sturdy', 'weak'),
                 ('strength4', 'mighty', 'weak')],
    'brightness': [('brightness1', 'bright', 'dark'),
                   ('brightness2', 'brilliant', 'dim'),
                   ('brightness3', 'radiant', 'dark')],
    'moisture': [('moisture1', 'wet', 'dry'),
                 ('moisture2', 'damp', 'dry'),
                 ('moisture3', 'soggy', 'dry')],
    'height': [('height1', 'tall', 'short'), ('height2', 'high', 'low'),
               ('height3', 'towering', 'short')],
    'fullness': [('fullness1', 'full', 'empty'),
                 ('fullness2', 'packed', 'hollow'),
                 ('fullness3', 'vacant', 'empty')],
}
AXES_ST = {
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
}
HOMOGRAPH_EXCLUDE = {
    'chat', 'sol', 'pain', 'chef', 'table', 'main', 'train', 'route',
    'lion', 'grand', 'mine', 'Mann', 'Gold', 'Winter', 'Hand', 'Bank',
    'Wind', 'Kind', 'Sommer', 'Morgen', 'Boot', 'Arm', 'pan', 'plaza',
    'real', 'color', 'metal', 'hotel', 'piano', 'radio', 'rosa',
}
VG2A_GATE, VG2B_GATE, VG3_GATE = 0.488, 0.2, 3.0

# ---------------- 3157 panel (verbatim, S2 collects) ----------------
SHARED_3157 = [
    ('桌子', '家具', '桌腿'), ('椅子', '座椅', '椅背'), ('窗户', '建筑构件', '窗框'),
    ('门', '建筑构件', '门把手'), ('硬币', '货币', '图案'), ('轮胎', '橡胶配件', '花纹'),
    ('纸', '材料', '纤维'), ('书', '出版物', '书页'), ('刀', '工具', '刀刃'),
    ('碗', '餐具', '碗底'), ('衣服', '服饰', '袖子'), ('鞋', '日用品', '鞋底'),
    ('桥', '建筑', '桥墩'), ('房子', '建筑', '屋顶'), ('汽车', '交通工具', '车轮'),
    ('钟表', '计时工具', '表盘'),
]
TPL_3157 = {
    ('isa', '+'): '{E}是一种{C}。', ('isa', '-'): '{E}不是一种{C}。',
    ('hasa', '+'): '{E}有{P}。', ('hasa', '-'): '{E}没有{P}。',
}
PREFIX_ZH = '今天天气很好，我们在讨论语言模型如何表示语言。'
K_CTX = 16

# ---------------- 3155 machinery (verbatim) ----------------
RELS_3155 = ['isa', 'hasa', 'mof']


def orthonorm(M, drop_tol=1e-8):
    U, s, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64),
                             full_matrices=False)
    keep = s > (s[0] * drop_tol if s.size and s[0] > 0 else 1e-12)
    return U[:, keep]


def orth_against(M, *qlist):
    R = np.asarray(M, dtype=np.float64).copy()
    for Q in qlist:
        R = R - Q @ (Q.T @ R)
    return R


def build_k2_bases(n_ent, rows):
    n = len(rows)
    E = np.zeros((n, n_ent)); C = np.zeros((n, 3)); T = np.zeros((n, 3))
    EC = np.zeros((n, n_ent * 3)); ET = np.zeros((n, n_ent * 3))
    CT = np.zeros((n, 9))
    for j, r in enumerate(rows):
        E[j, r['ent_id']] = 1.0; C[j, r['rel']] = 1.0; T[j, r['tmpl']] = 1.0
        EC[j, r['ent_id'] * 3 + r['rel']] = 1.0
        ET[j, r['ent_id'] * 3 + r['tmpl']] = 1.0
        CT[j, r['rel'] * 3 + r['tmpl']] = 1.0
    QE = orthonorm(E)
    QC = orthonorm(orth_against(orthonorm(C), QE))
    QT = orthonorm(orth_against(orthonorm(T), QE, QC))
    QEC = orthonorm(orth_against(orthonorm(EC), QE, QC, QT))
    QET = orthonorm(orth_against(orthonorm(ET), QE, QC, QT))
    QCT = orthonorm(orth_against(orthonorm(CT), QE, QC, QT))
    return dict(QE=QE, QC=QC, QT=QT, QEC=QEC, QET=QET, QCT=QCT)


def k2_shares_at(Hk, sh_idx, rows, B):
    X = Hk[sh_idx].astype(np.float64)
    R0 = X - X.mean(0)
    SS = float((R0 ** 2).sum())
    if SS < 1e-10:
        return {k: 0.0 for k in ('E', 'C', 'T', 'EintC', 'EintT', 'CintT',
                                 'ECT')}
    SS += 1e-18
    e = {}
    resid = R0
    for nm, Q in (('E', B['QE']), ('C', B['QC']), ('T', B['QT']),
                  ('EintC', B['QEC']), ('EintT', B['QET']),
                  ('CintT', B['QCT'])):
        P = Q @ (Q.T @ resid)
        e[nm] = float((P ** 2).sum())
        resid = resid - P
    e['ECT'] = float((resid ** 2).sum())
    sh = {k: v / SS for k, v in e.items()}
    assert abs(sum(sh.values()) - 1.0) < 1e-6, 'k2 anova closure'
    return sh


def kross(A, Bd):
    Qa, _ = np.linalg.qr(np.asarray(A, dtype=np.float64).T)
    Qb, _ = np.linalg.qr(np.asarray(Bd, dtype=np.float64).T)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    top1 = float(np.degrees(np.arccos(s[0])))
    return s, top1


def effdim(s):
    c2 = s ** 2
    cnt = int((c2 >= 0.5).sum())
    pr = float(c2.sum() ** 2 / ((c2 ** 2).sum() + 1e-300))
    return cnt, pr


def load_WU_t(mdir):
    import torch
    from safetensors import safe_open
    cfgm = json.load(io.open(os.path.join(mdir, 'config.json'),
                             encoding='utf-8'))
    tc = cfgm.get('text_config', {})
    tied = bool(cfgm.get('tie_word_embeddings',
                         tc.get('tie_word_embeddings', False)))
    want = 'model.embed_tokens.weight' if tied else 'lm_head.weight'
    W_t = None
    used = None
    for sh in sorted(os.listdir(mdir)):
        if not sh.endswith('.safetensors'):
            continue
        with safe_open(os.path.join(mdir, sh), framework='pt') as f:
            keys = set(f.keys())
            if want in keys:
                W_t = f.get_tensor(want)
                used = want
                break
    if W_t is None:
        for sh in sorted(os.listdir(mdir)):
            if not sh.endswith('.safetensors'):
                continue
            with safe_open(os.path.join(mdir, sh), framework='pt') as f:
                for k in f.keys():
                    if k.endswith('embed_tokens.weight') or \
                            k.endswith('lm_head.weight'):
                        W_t = f.get_tensor(k)
                        used = k
                        break
            if W_t is not None:
                break
    assert W_t is not None, 'unembed not found in ' + mdir
    return W_t.float(), tied, used, cfgm


def gram_top64(W_t, dev='cuda'):
    """3158 verbatim: G = W^T W fp32 chunked on dev -> eigh fp64 -> top64."""
    import torch
    V, D = W_t.shape
    W = W_t.to(dev, torch.float32)
    G = torch.zeros((D, D), dtype=torch.float32, device=dev)
    CH = 8192
    for s0 in range(0, V, CH):
        blk = W[s0:s0 + CH]
        G += blk.T @ blk
    Gc = G.cpu().numpy().astype(np.float64)
    del G, W
    if dev == 'cuda':
        torch.cuda.empty_cache()
    w0, U0 = np.linalg.eigh(Gc)
    top64 = U0[:, ::-1][:, :64].copy()
    return top64.astype(np.float64)


def build_family(tok, W_rows, axes_spec, exclude=None):
    """2874/2878 pair construction verbatim on given rows (dtype preserved)."""
    sys.path.insert(0, os.path.join(ROOT, 'tests', 'glm5'))
    from rdc_atlas_census import single_token_id
    tc = {}
    eff = {}
    for axis, pairs in axes_spec.items():
        keep = []
        for pid, wp, wm in pairs:
            if exclude and (wp in exclude or wm in exclude):
                continue
            keep.append((pid, wp, wm))
        eff[axis] = keep
    pair_dirs, axis_words, axis_meta = {}, {}, {}
    for axis, pairs in eff.items():
        words, valid = [], []
        for _, wp, wm in pairs:
            okp = okm = False
            try:
                single_token_id(tok, wp, tc); okp = True
            except AssertionError:
                pass
            try:
                single_token_id(tok, wm, tc); okm = True
            except AssertionError:
                pass
            if okp and okm:
                valid.append((wp, wm))
            for w, ok in ((wp, okp), (wm, okm)):
                if ok and w not in words:
                    words.append(w)
        axis_words[axis] = words
        axis_meta[axis] = dict(n_pairs_pool=len(pairs),
                               n_pairs_valid=len(valid))
        dirs = []
        for wp, wm in valid:
            d = W_rows[tc[wp]] - W_rows[tc[wm]]
            dirs.append(unit(d))
        pair_dirs[axis] = dirs
    return tc, axis_words, axis_meta, pair_dirs


def land_attr(pair_dirs, axis_words):
    """2874 verbatim: VG1 (>=5 words/axis, >=8/10 pass) -> dW over passing."""
    g1 = {a: len(axis_words[a]) >= 5 for a in axis_words}
    n_pass = sum(g1.values())
    vg1 = bool(n_pass >= 8)
    axes_ok = [a for a in axis_words if g1[a]]
    dirs_all = {a: pair_dirs[a] for a in axes_ok}
    pp = []
    for a, dirs in dirs_all.items():
        if len(dirs) >= 2:
            Dm = np.stack(dirs)
            S = Dm @ Dm.T
            iu = np.triu_indices(len(dirs), 1)
            pp.append(float(S[iu].mean()))
    mean_pp = float(np.mean(pp)) if pp else float('nan')
    vg2b = bool(mean_pp > VG2B_GATE)
    if axes_ok:
        dW = np.stack([unit(np.stack(dirs_all[a]).mean(axis=0))
                       for a in axes_ok])
        C = dW @ dW.T
        off = C[np.triu_indices(len(axes_ok), 1)]
        max_off = float(np.abs(off).max()) if off.size else 0.0
    else:
        dW = np.zeros((0, 0), np.float32)
        max_off = float('nan')
    vg2a = bool(max_off < VG2A_GATE) if np.isfinite(max_off) else False
    return dict(dW=dW.astype(np.float32), axes_ok=axes_ok,
                vg1=dict(pass_axes=n_pass, verdict=vg1),
                vg2b=dict(mean_pair_pair_cos=(round(mean_pp, 4)
                                              if np.isfinite(mean_pp)
                                              else None),
                          gate=VG2B_GATE, verdict=vg2b),
                vg2a=dict(max_cross_axis_cos=(round(max_off, 4)
                                              if np.isfinite(max_off) else None),
                          gate=VG2A_GATE, verdict=vg2a),
                words={a: axis_words[a] for a in axes_ok})


def land_syntax(pair_dirs, axis_words):
    """2878 verbatim: per-axis survival (>=5 valid pairs AND VG2b>0.2)."""
    vg2b = {}
    for a, dirs in pair_dirs.items():
        if len(dirs) >= 2:
            Dm = np.stack(dirs)
            S = Dm @ Dm.T
            iu = np.triu_indices(len(dirs), 1)
            vg2b[a] = float(S[iu].mean())
        else:
            vg2b[a] = float('nan')
    surviving = [a for a in pair_dirs
                 if len(pair_dirs[a]) >= 5
                 and np.isfinite(vg2b[a]) and vg2b[a] > VG2B_GATE]
    if surviving:
        dW = np.stack([unit(np.stack(pair_dirs[a]).mean(axis=0))
                       for a in surviving])
        C = dW @ dW.T
        off = C[np.triu_indices(len(surviving), 1)]
        max_off = float(np.abs(off).max()) if off.size else 0.0
    else:
        dW = np.zeros((0, 0), np.float32)
        max_off = float('nan')
    vg2a = bool(max_off < VG2A_GATE) if np.isfinite(max_off) else False
    return dict(dW=dW.astype(np.float32), surviving=surviving,
                vg2b={k: (round(v, 4) if np.isfinite(v) else None)
                      for k, v in vg2b.items()},
                vg2a=dict(max_cross_axis_cos=(round(max_off, 4)
                                              if np.isfinite(max_off) else None),
                          gate=VG2A_GATE, verdict=vg2a),
                words={a: axis_words[a] for a in surviving})


# ---------------- DESIGN freeze ----------------
DESIGN = {
    'phase': PHASE, 'name': NAME,
    'prereg_id': 'G5-H39 (owner adjudication 2026-10-21: 点名 APPX-2/3 执行相)',
    'protocol': 'appx23_protocol_v2.json (ce73984d, phase3267)',
    'anchor_sha8': ANCHOR_SHA,
    'mode': 'smoke' if SMOKE else 'formal',
    'arm_a': dict(
        models=MODELS_A,
        s1='2874 (float32, VG1>=5 words, >=8/10) / 2878 (homograph excl, '
           'per-axis survival >=5 pairs & VG2b>0.2) rules verbatim on W_U '
           'rows; VG2a<0.488 / VG3 context recorded; per-model, never pooled',
        s2='3157-style 128-row collect -> K_entity top8 -> kross vs K_readout '
           '(descriptive registration, G3 = present & consistent)',
        s3='K_readout = 3158-verbatim static Gram top64; S_class = 2881 '
           'recipe per-model (up-to-8/cat single-token, soft material gate '
           '>=8 cats with >=4 words); kross verdicts, thresholds verbatim '
           '30/15',
        s4='synthesis vs 4b incumbents 67.0/68.094/58.488 (all separable); '
           'no pooling (H4)',
    ),
    'arm_b': dict(
        models=MODELS_B,
        collect_source='sealed phase3155 collect.npz (formal rows)',
        t1='phi_l(c) frozen := QC u QEC balanced bases at KOUT (3155 '
           'verbatim construction); drop := 1 - mean_row cos(h, P_perp h)',
        t2='analytic readout-interface logits (logits = W_U @ h, h = '
           'post-final-norm slot NL, 3158 bitwise protocol); B1 agreement := '
           'P(argmax equal); FP_GATE 0.8; K2 fires iff drop > 0.5 AND '
           'agreement >= 0.8',
        t3='per-family between-entity eta2 ranking, top-50 coverage of '
           'between-entity SS at KOUT; fires iff all 3 families < 0.30',
        t4='per-model verdict registration; proxy-vs-direct documented',
    ),
    'cross_checks': {
        'X1': '4b S1 rebuild == sealed 2874/2878 dW_unit bitwise + axes list',
        'X2': '4b K_readout allclose p3158 sealed top64 (atol 1e-6)',
        'X3': '4b S_class x K_readout top1 within 0.5 deg of 67.0 incumbent',
        'X4': '4b recomputed ExC share == sealed 3155 value (atol 1e-9)',
    },
    'honesty': [
        'phi_l(c) operationalization (QC u QEC) frozen here before '
        'observation; thresholds verbatim from p35 deathlines (no drift)',
        'T2 separation applied at the readout interface (post-final-norm '
        'slot NL), which is the exact linear logit interface (3158); this is '
        'the frozen reading of "same separation at inference"',
        'S_class soft material gate: per-cat cap 8 single-token words as '
        '2881; candidates failing the 4b material assert get per-cat '
        'shortfall recorded and S_class requires >=8 cats with >=4 words',
        'per-model independence (H4/F4): no pooling; failures recorded '
        'per-model without blocking others',
    ],
    'landing': 'G5/H5: gap/registry landing deferred to separate registry '
               'phase (3277); this phase zero gap/registry/ledger-measurement '
               'writes',
    'created': time.strftime('%Y-%m-%d %H:%M:%S'),
}
_canon = json.dumps({k: v for k, v in DESIGN.items() if k != 'created'},
                    ensure_ascii=False, sort_keys=True,
                    indent=1).encode('utf-8')
DESIGN_SHA = hashlib.sha256(_canon).hexdigest()[:8]
with io.open(os.path.join(BASE, 'design_sha8.txt'), 'w', encoding='utf-8',
             newline='') as f:
    f.write(DESIGN_SHA)
log('freeze %s mode=%s' % (DESIGN_SHA, DESIGN['mode']))

gates = {}
per_model_a = {}
per_model_b = {}

# ================= cross-checks on qwen3-4b (always run) =================
log('==== X-gates on qwen3-4b ====')
import torch  # noqa: E402
torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
gates['cuda'] = bool(torch.cuda.is_available())
sys.path.insert(0, os.path.join(ROOT, 'tests', 'glm5'))

from transformers import AutoTokenizer  # noqa: E402

W_t, tied4, used4, cfg4 = load_WU_t(MDIR['qwen3-4b'])
W4_32 = W_t.cpu().numpy()  # float32 (2874/2878 dtype path)
log('4b W_U tied=%s via %s shape=%s' % (tied4, used4, W4_32.shape))

tok4 = AutoTokenizer.from_pretrained(MDIR['qwen3-4b'], local_files_only=True,
                                     trust_remote_code=True, use_fast=True)
tc4, aw_a4, meta_a4, pd_a4 = build_family(tok4, W4_32, AXES_ATTR)
la4 = land_attr(pd_a4, aw_a4)
z74 = np.load(P2874, allow_pickle=True)
gates['X1_attr_bitwise'] = bool(
    la4['axes_ok'] == [str(x) for x in z74['axes']]
    and la4['dW'].shape == z74['dW_unit'].shape
    and np.array_equal(la4['dW'], z74['dW_unit'].astype(np.float32)))
tcs4, aw_s4, meta_s4, pd_s4 = build_family(tok4, W4_32, AXES_ST,
                                           exclude=HOMOGRAPH_EXCLUDE)
ls4 = land_syntax(pd_s4, aw_s4)
z78 = np.load(P2878, allow_pickle=True)
gates['X1_syn_bitwise'] = bool(
    ls4['surviving'] == [str(x) for x in z78['axes']]
    and ls4['dW'].shape == z78['dW_unit'].shape
    and np.array_equal(ls4['dW'], z78['dW_unit'].astype(np.float32)))
log('X1 attr=%s syn=%s' % (gates['X1_attr_bitwise'], gates['X1_syn_bitwise']))

top64_4b = gram_top64(W_t)
z58 = np.load(P3158_4B)
gates['X2_kreadout_allclose'] = bool(
    top64_4b.shape == z58['top64'].shape
    and float(np.abs(top64_4b
                     - z58['top64'].astype(np.float64)).max()) < 1e-6)
log('X2 kreadout allclose: %s' % gates['X2_kreadout_allclose'])

e2806 = json.load(io.open(P2806_EXEC, encoding='utf-8'))
CATS = e2806['cats']
CAT_WORDS = list(CATS.keys())
tc_c = {}


def tid_c(tok, t):
    if t not in tc_c:
        ids = tok(' ' + t, add_special_tokens=False)['input_ids']
        if len(ids) != 1:
            ids = tok(t, add_special_tokens=False)['input_ids']
        assert len(ids) == 1, '%s -> %s' % (t, ids)
        tc_c[t] = int(ids[0])
    return tc_c[t]


def build_s_class(tok, W_rows, cat_cap=8):
    """2881/3165 recipe per-model: centroids over ALL single-token words per
    cat (3165 build_S_class verbatim); capped-8 list only for the material
    assert (3165: sum(class_targets)==80). Soft material gate for candidates:
    >=8 cats with >=8 single-token words."""
    per_cat, capped = {}, {}
    for cat in CAT_WORDS:
        ws = []
        for w in CATS[cat]:
            try:
                tid_c(tok, w)
                ws.append(w)
            except AssertionError:
                pass
        per_cat[cat] = ws
        capped[cat] = ws[:cat_cap]
    n_full = sum(1 for c in CAT_WORDS if len(capped[c]) >= cat_cap)
    if not (n_full >= 8):
        return None, dict(per_cat={k: len(v) for k, v in per_cat.items()},
                          n_full_cats=n_full, usable=False)
    cents = [np.stack([W_rows[tid_c(tok, w)] for w in per_cat[c]]).mean(0)
             for c in CAT_WORDS]
    Cm = np.stack(cents)
    dW_c = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    dW = np.stack([unit(dW_c[i]) for i in range(len(CAT_WORDS))])
    return dW.astype(np.float64), dict(
        per_cat={k: len(v) for k, v in per_cat.items()},
        n_full_cats=n_full, usable=True)


S_class4, sc4_meta = build_s_class(tok4, W4_32.astype(np.float64))
assert S_class4 is not None and sc4_meta['n_full_cats'] == 10, sc4_meta
s_sc, t_sc = kross(S_class4, top64_4b.T)
gates['X3_sclass_incumbent'] = bool(abs(t_sc - 67.0) < 0.5)
log('X3 s_class x k_readout top1=%.3f (incumbent 67.0): %s'
    % (t_sc, gates['X3_sclass_incumbent']))

# X4: 4b sealed 3155 collect recompute
r3155_4b = json.load(io.open(os.path.join(P3155_DIR, 'qwen3-4b',
                                           'result.json'),
                             encoding='utf-8'))
z = np.load(os.path.join(P3155_DIR, 'qwen3-4b', 'collect.npz'))
H4b = z['H']; REL4 = z['rel'].astype(int); EID4 = z['ent_id'].astype(int)
TMP4 = z['tmpl'].astype(int); SRC4 = z['src'].astype(int)
SH4 = np.where(SRC4 == 1)[0]
NL4 = H4b.shape[1] - 1
KOUT4 = NL4 - 1
rows4 = [dict(rel=int(REL4[i]), ent_id=int(EID4[i]), tmpl=int(TMP4[i]))
         for i in SH4]
B4 = build_k2_bases(24, rows4)
sh4 = k2_shares_at(H4b[:, KOUT4, :], SH4, rows4, B4)
gates['X4_exc_exact'] = bool(
    abs(sh4['EintC'] - r3155_4b['k2_shares_kout']['EintC']) < 1e-9)
log('X4 4b ExC recompute=%.6f sealed=%.6f: %s'
    % (sh4['EintC'], r3155_4b['k2_shares_kout']['EintC'],
       gates['X4_exc_exact']))
del W4_32, W_t, z, H4b
torch.cuda.empty_cache()

xc_ok = all(gates[k] for k in ('X1_attr_bitwise', 'X1_syn_bitwise',
                               'X2_kreadout_allclose', 'X3_sclass_incumbent',
                               'X4_exc_exact'))
gates['xgates_all'] = xc_ok
log('X-GATES ALL: %s' % xc_ok)

# ================= Arm a: per candidate model =================
for m in MODELS_A:
    if SMOKE and m != 'qwen2.5-3b-instruct':
        continue
    rec = dict(model=m)
    try:
        mdir = MDIR[m]
        tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True,
                                            trust_remote_code=True,
                                            use_fast=True)
        W_t, tied, used, cfgm = load_WU_t(mdir)
        W32 = W_t.cpu().numpy()
        Wr = W32.astype(np.float64)
        D = Wr.shape[1]
        rec['D'] = int(D); rec['tied'] = tied; rec['W_U_key'] = used
        rec['vocab_rows'] = int(Wr.shape[0])
        # ---- S1 ----
        tca, aw_a, meta_a, pd_a = build_family(tok, W32, AXES_ATTR)
        la = land_attr(pd_a, aw_a)
        tcs, aw_s, meta_s, pd_s = build_family(tok, W32, AXES_ST,
                                               exclude=HOMOGRAPH_EXCLUDE)
        ls = land_syntax(pd_s, aw_s)
        rec['s1'] = dict(
            attr=dict(n_axes_pool=len(AXES_ATTR),
                      vg1=la['vg1'], axes_ok=la['axes_ok'],
                      vg2b=la['vg2b'], vg2a=la['vg2a'],
                      axis_words=la['words']),
            syntax=dict(n_axes_pool=len(AXES_ST),
                        surviving=ls['surviving'], vg2b=ls['vg2b'],
                        vg2a=ls['vg2a'], axis_words=ls['words']))
        np.savez_compressed(
            os.path.join(BASE, 'appx2_%s_axes.npz' % m),
            dW_attr=la['dW'], dW_syntax=ls['dW'],
            attr_axes=np.array(la['axes_ok'], dtype=object),
            syntax_axes=np.array(ls['surviving'], dtype=object),
            tid_map=np.array(json.dumps({**tca, **tcs}), dtype=object),
            meta=np.array(json.dumps(dict(attr=meta_a, syntax=meta_s)),
                          dtype=object))
        # ---- S3 static (K_readout + S_class) ----
        top64 = gram_top64(W_t)
        K_read = top64.T
        sclass, sc_meta = build_s_class(tok, Wr)
        s3 = dict(s_class_meta=sc_meta, pairs={})
        if sclass is not None:
            s, t1 = kross(sclass, K_read)
            cnt, pr = effdim(s)
            s3['pairs']['K_readout__S_class'] = dict(
                top1_deg=round(t1, 3), eff_ge05=cnt, pr=round(pr, 4),
                verdict=('separable' if t1 >= 30.0 else
                         ('collinear' if t1 < 15.0 else
                          'weakly_separated')))
        if la['dW'].shape[0] >= 1:
            s, t1 = kross(la['dW'].astype(np.float64), K_read)
            cnt, pr = effdim(s)
            s3['pairs']['K_readout__S_attr'] = dict(
                top1_deg=round(t1, 3), eff_ge05=cnt, pr=round(pr, 4),
                n_axes=int(la['dW'].shape[0]),
                verdict=('separable' if t1 >= 30.0 else
                         ('collinear' if t1 < 15.0 else
                          'weakly_separated')))
        if ls['dW'].shape[0] >= 1:
            s, t1 = kross(ls['dW'].astype(np.float64), K_read)
            cnt, pr = effdim(s)
            s3['pairs']['K_readout__S_syntax'] = dict(
                top1_deg=round(t1, 3), eff_ge05=cnt, pr=round(pr, 4),
                n_axes=int(ls['dW'].shape[0]),
                verdict=('separable' if t1 >= 30.0 else
                         ('collinear' if t1 < 15.0 else
                          'weakly_separated')))
        rec['s3'] = s3
        np.savez_compressed(
            os.path.join(BASE, 'appx2_%s_kreadout.npz' % m),
            top64=top64.astype(np.float32))
        # ---- S2 (K_entity profile; forwards) ----
        try:
            from transformers import AutoModelForCausalLM
            cfg = json.load(io.open(os.path.join(mdir, 'config.json'),
                                    encoding='utf-8'))
            tcfg = cfg.get('text_config', cfg)
            NL = int(tcfg['num_hidden_layers'])
            HID = int(tcfg['hidden_size'])
            rows2 = []
            for ei, ent in enumerate(SHARED_3157):
                for rel in ('isa', 'hasa'):
                    for pol in ('+', '-'):
                        for cx in (0, 1):
                            rows2.append(dict(
                                ei=ei, rel=rel, pol=pol, ctx=cx,
                                prompt=TPL_3157[(rel, pol)]
                                .replace('{E}', ent[0])
                                .replace('{C}', ent[1])
                                .replace('{P}', ent[2])))
            pre_ids = tok(PREFIX_ZH,
                          add_special_tokens=False)['input_ids']
            ctx_ids = (pre_ids * (K_CTX // len(pre_ids) + 1))[:K_CTX]
            model = AutoModelForCausalLM.from_pretrained(
                mdir, dtype=torch.bfloat16,
                trust_remote_code=True).to('cuda').eval()
            NH = NL + 1
            # fp32 storage: some families (gemma-3) overflow fp16 at slot NL
            H32 = np.zeros((len(rows2), NH, HID), np.float32)
            with torch.no_grad():
                for i, r in enumerate(rows2):
                    ids = (ctx_ids if r['ctx'] == 1 else []) + \
                          tok(r['prompt'],
                              add_special_tokens=False)['input_ids']
                    ii = torch.tensor([ids], dtype=torch.int64,
                                      device='cuda')
                    o = model(input_ids=ii, output_hidden_states=True)
                    hs = o.hidden_states
                    H32[i] = np.stack(
                        [h[0, -1].float().detach().cpu().numpy()
                         for h in hs], 0).astype(np.float32)
                    del o, hs
                    if (i + 1) % 64 == 0:
                        log('S2 collect %s %d/%d' % (m, i + 1, len(rows2)))
            del model
            torch.cuda.empty_cache()
            X = H32[:, NL, :].astype(np.float64)
            slot_norm = float(np.linalg.norm(X, axis=1).mean())
            Xc = X - X.mean(0, keepdims=True)
            _, _, Vh = np.linalg.svd(Xc, full_matrices=False)
            K_ent = Vh[:8]
            s_ke, t_ke = kross(K_ent, K_read)
            cnt_ke, pr_ke = effdim(s_ke)
            np.savez_compressed(
                os.path.join(BASE, 'appx2_%s_kentity.npz' % m),
                k_entity=K_ent.astype(np.float32), H=H32)
            rec['s2'] = dict(NL=NL, HID=HID, n_rows=len(rows2),
                             slot_norm_mean=round(slot_norm, 4),
                             slot_norm_ref_sqrtD=round(float(HID) ** 0.5, 4),
                             k_entity_vs_readout=dict(
                                 top1_deg=round(t_ke, 3),
                                 eff_ge05=cnt_ke, pr=round(pr_ke, 4)),
                             gate='registered_descriptive')
        except Exception as e:  # per-model independence (H4)
            rec['s2'] = dict(error='%s: %s' % (type(e).__name__, e),
                             tb=traceback.format_exc()[-800:])
            log('S2 FAIL %s: %s' % (m, e))
        rec['status'] = 'ok'
    except Exception as e:
        rec['status'] = 'failed'
        rec['error'] = '%s: %s' % (type(e).__name__, e)
        rec['tb'] = traceback.format_exc()[-800:]
        log('ARM A FAIL %s: %s' % (m, e))
    per_model_a[m] = rec
    json.dump(rec, io.open(os.path.join(BASE, 'appx2_%s_record.json' % m),
                           'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    try:
        del Wr, W32, W_t
    except Exception:
        pass
    torch.cuda.empty_cache()
    log('ARM A done %s status=%s' % (m, rec['status']))

# ================= Arm b: K2/K3 direct on sealed 3155 collects ============
for m in MODELS_B:
    if SMOKE and m != 'qwen3-4b':
        continue
    rec = dict(model=m)
    try:
        pdir = os.path.join(P3155_DIR, m)
        z = np.load(os.path.join(pdir, 'collect.npz'))
        H = z['H']
        REL = z['rel'].astype(int); EID = z['ent_id'].astype(int)
        TMP = z['tmpl'].astype(int); SRC = z['src'].astype(int)
        SH = np.where(SRC == 1)[0]
        NL = H.shape[1] - 1
        KOUT = NL - 1
        HID = H.shape[2]
        rows = [dict(rel=int(REL[i]), ent_id=int(EID[i]), tmpl=int(TMP[i]))
                for i in SH]
        r_sealed = json.load(io.open(os.path.join(pdir, 'result.json'),
                                     encoding='utf-8'))
        B = build_k2_bases(24, rows)
        Hk = H[:, KOUT, :].astype(np.float32)
        sh = k2_shares_at(Hk, SH, rows, B)
        rec['proxy_recheck'] = dict(
            EintC=round(sh['EintC'], 6),
            sealed=round(r_sealed['k2_shares_kout']['EintC'], 6),
            delta=float(abs(sh['EintC']
                            - r_sealed['k2_shares_kout']['EintC'])),
            proxy_note='ANOVA ExC share <50% suggests separability but is '
                       'not the direct criterion (protocol T4)')
        # ---- T1: K2 direct (QC u QEC separation) ----
        # NOTE: 3155 bases are SAMPLE-space (n, k); projection of the response
        # matrix is Q @ (Q.T @ X) (3155 k2_shares_at verbatim operand order).
        X = Hk[SH].astype(np.float64)
        Q = np.concatenate([B['QC'], B['QEC']], axis=1)
        Q, _ = np.linalg.qr(Q)
        Qc = B['QC']; Qe = B['QEC']
        sep = X - Q @ (Q.T @ X)
        denom = np.linalg.norm(X, axis=1)
        cos_full = np.array([
            float(np.dot(X[i], sep[i]) /
                  max(denom[i] * np.linalg.norm(sep[i]), 1e-30))
            for i in range(len(SH))])
        sep_c = X - Qc @ (Qc.T @ X)
        sep_e = X - Qe @ (Qe.T @ X)
        cos_c = np.array([float(np.dot(X[i], sep_c[i]) /
                                max(denom[i] * np.linalg.norm(sep_c[i]),
                                    1e-30)) for i in range(len(SH))])
        cos_e = np.array([float(np.dot(X[i], sep_e[i]) /
                                max(denom[i] * np.linalg.norm(sep_e[i]),
                                    1e-30)) for i in range(len(SH))])
        drop_full = float(1.0 - cos_full.mean())
        drop_c = float(1.0 - cos_c.mean())
        drop_e = float(1.0 - cos_e.mean())
        rec['t1'] = dict(
            basis='QC u QEC (dims=%d)' % Q.shape[1],
            cos_drop_full=round(drop_full, 6),
            cos_drop_QC_only=round(drop_c, 6),
            cos_drop_QEC_only=round(drop_e, 6),
            cos_full_mean=round(float(cos_full.mean()), 6),
            threshold='drop > 0.5')
        np.savez_compressed(os.path.join(BASE, 'appx3_%s_bases.npz' % m),
                            QC=Qc, QEC=Qe, Q=Q)
        # ---- T2: behavioral half (analytic readout logits) ----
        W_t, tied, used, cfgm = load_WU_t(MDIR[m])
        Wg = W_t.to(dev, torch.float32)
        Xg = torch.from_numpy(X.astype(np.float32)).to(dev)
        Sg = torch.from_numpy(sep.astype(np.float32)).to(dev)
        base_am, sep_am = [], []
        CH = 32
        with torch.no_grad():
            for s0 in range(0, Xg.shape[0], CH):
                lb = (Xg[s0:s0 + CH] @ Wg.T).argmax(1)
                lsp = (Sg[s0:s0 + CH] @ Wg.T).argmax(1)
                base_am.append(lb.cpu().numpy())
                sep_am.append(lsp.cpu().numpy())
        base_am = np.concatenate(base_am)
        sep_am = np.concatenate(sep_am)
        agree = float((base_am == sep_am).mean())
        del Wg, Xg, Sg, W_t
        torch.cuda.empty_cache()
        k2_fires = bool(drop_full > 0.5 and agree >= 0.8)
        rec['t2'] = dict(
            protocol='logits = W_U @ h (post-final-norm slot NL; 3158 '
                     'bitwise-verified readout interface)',
            b1_first_token_agreement=round(agree, 6),
            fp_gate=0.8, behavior_intact=bool(agree >= 0.8),
            n_rows=int(len(SH)))
        rec['k2_verdict'] = dict(
            fires=k2_fires,
            rule='fires iff cos_drop > 0.5 AND agreement >= 0.8',
            reading=('conditional gate NOT independent structure -> abandon'
                     if k2_fires else
                     'deathline does not fire -> conditional gate not '
                     'abandoned by this direct measurement'))
        # ---- T3: K3 coverage ----
        fam_cov = {}
        rel_arr = np.array([r['rel'] for r in rows])
        for c, fam in enumerate(RELS_3155):
            idx = SH[rel_arr == c]
            ents = EID[idx]
            Xf = Hk[idx].astype(np.float64)
            Xf = Xf - Xf.mean(0, keepdims=True)
            tot = (Xf ** 2).sum(0)
            ue = np.unique(ents)
            gmeans = np.stack([Xf[ents == e].mean(0) for e in ue])
            gmc = gmeans - gmeans.mean(0)
            between = (gmc ** 2).sum(0) * len(ue)
            with np.errstate(divide='ignore', invalid='ignore'):
                eta2 = between / np.maximum(tot, 1e-300)
            order = np.argsort(-eta2)
            top50 = order[:50]
            cov = float(between[top50].sum() / max(between.sum(), 1e-300))
            fam_cov[fam] = dict(coverage=round(cov, 6),
                                n_rows=int(len(idx)),
                                top1_eta2=round(float(eta2[order[0]]), 6))
        k3_fires = bool(all(v['coverage'] < 0.30
                            for v in fam_cov.values()))
        rec['t3'] = dict(per_family=fam_cov,
                         threshold='<0.30 all families', fires=k3_fires)
        rec['k3_verdict'] = dict(
            fires=k3_fires,
            reading=('single-coordinate mechanism target abandoned '
                     '(<=10-dim joint + port class)' if k3_fires else
                     'deathline does not fire -> single-coordinate target '
                     'not abandoned by this direct measurement'))
        rec['status'] = 'ok'
        del z, H, Hk
    except Exception as e:
        rec['status'] = 'failed'
        rec['error'] = '%s: %s' % (type(e).__name__, e)
        rec['tb'] = traceback.format_exc()[-800:]
        log('ARM B FAIL %s: %s' % (m, e))
    per_model_b[m] = rec
    json.dump(rec, io.open(os.path.join(BASE, 'appx3_%s_record.json' % m),
                           'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    log('ARM B done %s status=%s' % (m, rec['status']))

# ================= verdict =================
if not SMOKE:
    gates['arm_a_all_registered'] = all(
        per_model_a[m].get('status') == 'ok' for m in MODELS_A)
    gates['arm_b_all_registered'] = all(
        per_model_b[m].get('status') == 'ok' for m in MODELS_B)
k2_fires_any = any(per_model_b[m].get('k2_verdict', {}).get('fires')
                   for m in per_model_b)
k3_fires_any = any(per_model_b[m].get('k3_verdict', {}).get('fires')
                   for m in per_model_b)
res = {
    'phase': PHASE, 'name': NAME, 'line': 'G',
    'date': time.strftime('%Y-%m-%d'),
    'design_sha8': DESIGN_SHA, 'mode': DESIGN['mode'],
    'gates': gates,
    'arm_a_appx2': per_model_a,
    'arm_b_appx3': per_model_b,
    'synthesis': dict(
        k2_deathline_fires_any_model=k2_fires_any,
        k3_deathline_fires_any_model=k3_fires_any,
        note='per-model verdicts independent (H4); landing via 3277 (G5/H5)'),
    'protocol': 'ce73984d (PREREGISTERED_DRAFT_V2 executed)',
}
if not SMOKE:
    res['res_sha8'] = hashlib.sha256(json.dumps(
        {k: v for k, v in res.items() if k != 'res_sha8'},
        sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
    with io.open(os.path.join(BASE, 'result.json'), 'w', encoding='utf-8',
                 newline='') as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=1))
    print('RES_SHA8', res['res_sha8'])
else:
    os.makedirs(os.path.join(BASE, 'smoke'), exist_ok=True)
    with io.open(os.path.join(BASE, 'smoke', 'result_smoke.json'), 'w',
                 encoding='utf-8', newline='') as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=1))
print('DONE mode=%s elapsed=%.1fs' % (DESIGN['mode'], time.time() - T0))
