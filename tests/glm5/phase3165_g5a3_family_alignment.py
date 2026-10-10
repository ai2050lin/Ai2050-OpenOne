# -*- coding: utf-8 -*-
# Phase 3165 (G5-A3): 跨族连接 v0 —— 族轴子空间对齐普查（图谱缺口③首步）
# 零 GPU（全部用已封存 npz + safetensors 行读取）。
#
# 子空间（统一 D=2560 残差流/读出空间，4b 主普查）：
#   K_readout : 3158 top64（W_U Gram top64 特征向量, (D,64) 列空间）
#   K_entity  : 3157 H[:, NL, :] 128 行（16 锚实体）中心化 SVD top8
#   S_class   : 2881 类质心差分 10 方向（公式+CATS+tid 规则 verbatim 重建）
#   S_attr    : 2874 dW_unit (8, D)
#   S_syntax  : 2878 dW_unit (3, D)
#   S_joint   : 上述 21 拼接
# 门（预注册）：族间 top-1 主角 >=30deg -> separable / <15deg -> collinear /
#              之间 -> weakly_separated；判决对 = K_readout x S_class/S_attr/S_syntax。
# 推理族 R：无已封存方向材料 -> pending_material（预注册(iv) 缺者待补）。
# 跨模型 (iv)：K_readout/K_entity 于 14b/glm4 同构读数（descriptive；D 不可跨模型求角）。
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

PHASE = 3165
NAME = 'g5a3_family_alignment'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
B = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
BASE = os.path.join(B, 'phase3165', NAME)
MDIR4 = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
LOG = []
SMOKE = os.environ.get('P3165_SMOKE', '') == '1'


def log(s):
    line = '[3165] %s' % s
    LOG.append(line)
    print(line, flush=True)


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def unit(x):
    n = float(np.linalg.norm(x))
    assert n > 0
    return x / n


# ---------------- 源锚（冻结前实测，见 tests/gpt5_temp/p3165_shas.txt） ----------------
SRC = {
    'p3158_4b': B + r'\phase3158\g4p1_output_equivalence_class\qwen3-4b\collect.npz',
    'p3158_14b': B + r'\phase3158\g4p1_output_equivalence_class\qwen3-14b\collect.npz',
    'p3158_glm4': B + r'\phase3158\g4p1_output_equivalence_class\glm4\collect.npz',
    'p3157_4b': B + r'\phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz',
    'p3157_14b': B + r'\phase3157\g2p2_transform_algebra_commutator\qwen3-14b\collect.npz',
    'p3157_glm4': B + r'\phase3157\g2p2_transform_algebra_commutator\glm4\collect.npz',
    'p2874': B + r'\phase2874\attr_vocab_v2\attr_vocab_v2.npz',
    'p2878': B + r'\phase2878\syntax_trans_vocab\syntax_trans_vocab.npz',
    'p2881': B + r'\phase2881\joint_word_coords\joint_word_coords.npz',
    'p2806_exec': B + r'\phase2806\qwen4_hierarchy\execution.json',
}
SHA_ANCHOR = {k: sha8(p) for k, p in SRC.items()}

DESIGN = dict(
    phase=PHASE, name=NAME,
    subspace_defs=dict(
        K_readout='3158 collect.npz top64 (D,64) column space (W_U Gram top eigvecs)',
        K_entity='3157 collect.npz H[:,NL,:] 128 rows center-SVD top8 right-eigvecs',
        S_class='2881 recipe verbatim: CATS(phase2806 exec) + MAX_WORDS=8 + single_tok '
                "filter + tid(' '+w else w) + centroid diff dW_c = Cm-(sum-Cm)/9, unit",
        S_attr='2874 attr_vocab_v2 dW_unit (8,D)',
        S_syntax='2878 syntax_trans_vocab dW_unit (3,D)',
        S_joint='concat(S_class,S_attr,S_syntax) 21 dirs'),
    source_sha8=SHA_ANCHOR,
    angle_method='QR orthonormalize each subspace; principal cos = svd(Qa.T@Qb) singular values desc',
    eff_dim=dict(count='#{cos^2>=0.5}', pr='(sum s^2)^2 / sum s^4'),
    gates=dict(
        pairs=['K_readout|xS_class', 'K_readout|xS_attr', 'K_readout|xS_syntax'],
        rule='top1 principal angle: >=30deg separable / <15deg collinear / else weakly_separated',
        aggregate='all three agree -> single cls; else mixed(min pair reported)'),
    descriptive=['S internal pairs', 'K_entity x S pairs', 'K_readout x K_entity per model',
                 'cross-model K family stats (D differs, no cross-model angles)'],
    reason_family='pending_material: no sealed logic-tag direction npz (3151/3152 H are '
                  'true-proposition panels; no contrastive direction constructible without '
                  'new design freedom) -> deferred per prereg (iv)',
    smoke=SMOKE,
)


# ---------------- freeze / drift ----------------
CANON_KEYS = ('phase', 'name', 'subspace_defs', 'source_sha8', 'angle_method',
              'eff_dim', 'gates', 'descriptive', 'reason_family')


def canon_sha(d):
    blob = json.dumps({k: d[k] for k in CANON_KEYS}, ensure_ascii=False,
                      indent=1, sort_keys=True).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()[:8]


def freeze():
    os.makedirs(BASE, exist_ok=True)
    p = os.path.join(BASE, 'execution.json')
    if os.path.exists(p):
        old = json.load(io.open(p, encoding='utf-8'))
        for k in CANON_KEYS:
            assert old.get(k) == DESIGN[k], ('design drift', k)
        assert old.get('design_sha8') == canon_sha(old), 'execution self-sha drift'
        return old['design_sha8']
    d8 = canon_sha(DESIGN)
    with io.open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(dict(DESIGN, design_sha8=d8), ensure_ascii=False,
                           indent=1, sort_keys=True))
    log('execution.json frozen design_sha8=%s' % d8)
    return d8


# ---------------- subspace builders ----------------
def load_WU(mdir):
    from safetensors import safe_open
    cfgm = json.load(io.open(os.path.join(mdir, 'config.json'), encoding='utf-8'))
    tied = bool(cfgm.get('tie_word_embeddings', False))
    want = 'model.embed_tokens.weight' if tied else 'lm_head.weight'
    for sh in sorted(os.listdir(mdir)):
        if not sh.endswith('.safetensors'):
            continue
        with safe_open(os.path.join(mdir, sh), framework='pt') as f:
            if want in set(f.keys()):
                W = f.get_tensor(want).float().numpy()
                break
    else:
        raise AssertionError('unembed not found in ' + mdir)
    return W.astype(np.float64), cfgm


def build_S_class(W4):
    """2881 L143-190 verbatim（word list 与 target_list 对拍）。"""
    e2806 = json.load(io.open(SRC['p2806_exec'], encoding='utf-8'))
    CATS = e2806['cats']
    CAT_WORDS = list(CATS.keys())
    assert CAT_WORDS == ['fruit', 'animal', 'metal', 'vehicle', 'country',
                         'food', 'nature', 'furniture', 'tool', 'clothing']
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MDIR4, local_files_only=True,
                                        trust_remote_code=True, use_fast=True)
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            tc[t] = int(ids[0])
        return tc[t]

    all_words = [w for v in CATS.values() for w in v]
    single_tok = []
    for w in all_words:
        try:
            tid(w)
            single_tok.append(w)
        except AssertionError:
            pass
    class_targets = {}
    for cat in CAT_WORDS:
        class_targets[cat] = [w for w in CATS[cat] if w in single_tok][:8]
    assert sum(len(v) for v in class_targets.values()) == 80

    Erows = {w: W4[tid(w)] for w in single_tok}
    cents = []
    for cat in CAT_WORDS:
        ws = [w for w in CATS[cat] if w in single_tok]
        cents.append(np.stack([Erows[w] for w in ws]).mean(0))
    Cm = np.stack(cents)
    dW_c = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    dW_class = np.stack([unit(dW_c[i]) for i in range(10)])
    class_words = [w for cat in CAT_WORDS for w in class_targets[cat]]
    return dW_class.astype(np.float32), class_words, single_tok


def kross(A, Bd):
    """principal angles from row-form (k, D) subspaces; returns cos desc, top1 deg."""
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(Bd.T)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    top1 = float(np.degrees(np.arccos(s[0])))
    return s, top1


def effdim(s):
    c2 = s ** 2
    cnt = int((c2 >= 0.5).sum())
    pr = float(c2.sum() ** 2 / (c2 ** 2).sum() + 1e-300)
    return cnt, pr


def seal_result(result, fname):
    raw = json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True)
    res8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
    mid = json.dumps(dict(result, res_sha8=res8), ensure_ascii=False,
                     indent=1, sort_keys=True)
    seal8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
    result['res_sha8'] = res8
    result['seal_sha8'] = seal8
    out = os.path.join(BASE, fname)
    with io.open(out, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True))
    log('sealed %s res=%s seal=%s' % (fname, res8, seal8))
    return res8, seal8


# ---------------- main ----------------
def main():
    t0 = time.monotonic()
    dsha = freeze()
    for k, p in SRC.items():
        assert sha8(p) == SHA_ANCHOR[k], ('source drift', k)

    # ---- 4b materials ----
    z58 = np.load(SRC['p3158_4b'])
    top64 = z58['top64'].astype(np.float64)              # (2560, 64)
    D = top64.shape[0]
    z57 = np.load(SRC['p3157_4b'])
    H = z57['H'].astype(np.float64)                      # (128, NL+1, D)
    NL = H.shape[1] - 1
    ei = z57['ei'].astype(int)
    assert len(np.unique(ei)) == 16, 'entity anchors'
    X = H[:, NL, :]
    Xc = X - X.mean(0, keepdims=True)
    _, _, Vh = np.linalg.svd(Xc, full_matrices=False)
    K_ent = Vh[:8]                                        # (8, D) row form

    W4, cfg4 = load_WU(MDIR4)
    assert int(cfg4['hidden_size']) == D
    dW_class, class_words, single_tok = build_S_class(W4)

    z74 = np.load(SRC['p2874'], allow_pickle=True)
    dW_attr = z74['dW_unit'].astype(np.float32)           # (8, 2560)
    z78 = np.load(SRC['p2878'], allow_pickle=True)
    dW_syn = z78['dW_unit'].astype(np.float32)            # (3, 2560)
    z81 = np.load(SRC['p2881'], allow_pickle=True)
    tl81 = [str(x) for x in z81['target_list']]
    lf81 = z81['labels_fam'].astype(int)
    assert z81['B3_joint'].shape == (170, 210)
    assert z81['B3_joint'].shape[1] == 21 * 10   # 21 dirs x 10 layers (L26-35)
    # 对拍（target_list 为 'fam:cat:word' 标签；word = 最后一段）
    wpart = lambda s: s.split(':')[-1]
    w81 = [wpart(s) for s in tl81]
    assert (lf81[:80] == 0).all() and (lf81[80:122] == 1).all() and (lf81[122:] == 2).all()
    assert w81[:80] == class_words, 'class word order drift'
    tl74 = [wpart(str(x)) for x in z74['target_list']]
    tl78 = [wpart(str(x)) for x in z78['target_list']]
    assert w81[80:122] == tl74, 'attr word drift'
    assert w81[122:] == tl78, 'syntax word drift'
    assert len(tl74) == 42 and len(tl78) == 48 and len(tl81) == 170
    log('crosscheck words: class80 + attr42 + syntax48 = 170 OK')

    S_class = dW_class.astype(np.float64)                 # (10, D)
    S_attr = dW_attr.astype(np.float64)                   # (8, D)
    S_syn = dW_syn.astype(np.float64)                     # (3, D)
    S_joint = np.concatenate([S_class, S_attr, S_syn], 0)  # (21, D)
    K_read = top64.T                                       # (64, D) row form

    subs = {'K_readout': K_read, 'K_entity': K_ent, 'S_class': S_class,
            'S_attr': S_attr, 'S_syntax': S_syn, 'S_joint': S_joint}

    # ---- pairwise principal angles (4b) ----
    pair_res = {}
    names = list(subs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            s, top1 = kross(subs[a], subs[b])
            cnt, pr = effdim(s)
            pair_res['%s__%s' % (a, b)] = dict(
                k=int(min(subs[a].shape[0], subs[b].shape[0])),
                top1_deg=round(top1, 3), cos=s.round(6).tolist(),
                eff_ge05=cnt, pr=round(pr, 4))
    gate_pairs = ['K_readout__S_class', 'K_readout__S_attr', 'K_readout__S_syntax']
    clsmap = []
    for gp in gate_pairs:
        t1 = pair_res[gp]['top1_deg']
        clsmap.append('separable' if t1 >= 30.0 else ('collinear' if t1 < 15.0 else 'weakly_separated'))
    agree = len(set(clsmap)) == 1
    cls = clsmap[0] if agree else 'mixed'
    log('4b gate pairs: %s -> %s' % (dict(zip(gate_pairs, clsmap)), cls))

    # ---- cross-model K family (14b, glm4) ----
    cross = {}
    for m, mk in (('qwen3-14b', 'p3158_14b'), ('glm4', 'p3158_glm4')):
        zm = np.load(SRC[mk])
        t64m = zm['top64'].astype(np.float64)
        Dm = t64m.shape[0]
        zm7 = np.load(SRC[mk.replace('p3158', 'p3157')])
        Hm = zm7['H'].astype(np.float64)
        NLm = Hm.shape[1] - 1
        Xm = Hm[:, NLm, :]
        Xm = Xm - Xm.mean(0, keepdims=True)
        _, _, Vhm = np.linalg.svd(Xm, full_matrices=False)
        K_entm = Vhm[:8]
        s_ke, t_ke = kross(t64m.T, K_entm)
        cnt_ke, pr_ke = effdim(s_ke)
        c = json.load(io.open(os.path.join(ROOT, 'models', 'hf',
                                           {'qwen3-14b': 'Qwen3-14B',
                                            'glm4': 'glm4-9b-chat-hf'}[m],
                                           'config.json'), encoding='utf-8'))
        cross[m] = dict(D=Dm, NL=NLm,
                        K_readout_dim=int(t64m.shape[1]),
                        K_entity_top1_vs_readout_deg=round(t_ke, 3),
                        K_entity_eff_ge05=cnt_ke, K_entity_pr=round(pr_ke, 4),
                        V=int(c['vocab_size']))
        log('cross %s: D=%d K_ent-vs-readout top1=%.2fdeg' % (m, Dm, t_ke))

    # 4b K_readout x K_entity same-space angle
    s_kk, t_kk = kross(K_read, K_ent)
    cnt_kk, pr_kk = effdim(s_kk)

    elapsed = round(time.monotonic() - t0, 1)
    result = dict(
        phase=PHASE, name=NAME, model_4b='qwen3-4b', D=D, NL=NL,
        design_sha8=dsha, smoke=SMOKE,
        subspace_dims={k: int(v.shape[0]) for k, v in subs.items()},
        pairwise_4b=pair_res,
        verdict=dict(cls=cls, per_pair=dict(zip(gate_pairs, clsmap)),
                     agree=agree,
                     note='family R (logic tag) = pending_material; gates only K x S'),
        k_entity_vs_readout_4b=dict(top1_deg=round(t_kk, 3), eff_ge05=cnt_kk,
                                    pr=round(pr_kk, 4)),
        cross_model_K=cross,
        source_sha8=SHA_ANCHOR, elapsed_s=elapsed,
    )
    if SMOKE:
        log('SMOKE RESULT cls=%s top1s=%s' % (cls, {g: pair_res[g]['top1_deg'] for g in gate_pairs}))
        log('SMOKE S-internal: %s' % {k: v['top1_deg'] for k, v in pair_res.items() if k.startswith('S_')})
        log('SMOKE K_entity pairs: %s' % {k: v['top1_deg'] for k, v in pair_res.items() if 'K_entity' in k and not k.startswith('K_readout__K_entity')})
        seal_result(result, 'smoke_result.json')
        return

    seal_result(result, 'result.json')
    with io.open(os.path.join(BASE, 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    log('DONE elapsed=%.1fs' % elapsed)


if __name__ == '__main__':
    main()
