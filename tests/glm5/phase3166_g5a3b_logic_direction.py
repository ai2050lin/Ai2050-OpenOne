# -*- coding: utf-8 -*-
# Phase 3166 (G5-A3b): R 族逻辑对比方向补采 + 三族普查 v1（图谱缺口③第二步）
# 预注册：ledger n=317 detail（3165 条目）+ MEMO 3165 节「下一步 3166=G5-A3b」。
# 设计细化（本文件 DESIGN）于任何观测前冻结（execution.json drift 断言）。
#
# 关键认识（区别于 ledger 3165 的 pending 理由）：
#   ledger 记「3151/3152 H are true-proposition panels」——实为不精确：面板 PAIRS =
#   41 实体 x 6 类全组合（738 行），天然含真臂 (i, CLS_OF[i]) 与假臂 (i, c != CLS_OF[i])。
#   => 零 GPU 复用已封存 collect.npz 即可构造反事实逻辑对比方向（比预注册的 GPU 采集更强：
#      材料已封存、sha 已锚定、面板 verbatim）。
#
# 构造（主口径）：
#   真臂 = (i, true_c(i)), 假臂 = (i, c != true_c(i))；模板 t=0..2 平均。
#   per-entity 逻辑方向: D_i = mean_t H[t,(i,true),k] - mean_{t, c!=true} H[t,(i,c),k]
#   d_i = unit(D_i); 中心化; SVD -> Vh[:8] = R_logic (8, D) row-form。
#   主槽位 k_final = NL（最后槽 = final-norm 前末层输出，与 3157 K_entity 槽位 H[:,NL,:] 一致）。
#   辅助槽位（descriptive）：k_kout = NL-1（3152 readout 口径 = E_read 复现槽）、
#                           k_kstar = 3（3152 K1 门层，冻结 KSTAR_FRAC=0.075）。
# 装置门（观测前冻结）：
#   G0 行为对照: mean(MARG_true true-class margin) > mean(MARG_false claimed-class margin)
#   G1 尺度:     mean_i |D_i| >= 0.01 * mean(true rows |H_row,k|)
#   G2 区分力:   严格 LOEO AUC >= 0.6（fold=实体；fold 内用其余实体重建 top8+判别方向）
# 普查门（同 3165）：gate pairs = R_logic x {K_readout, K_entity, S_class};
#   top-1 主角 >=30deg separable / <15deg collinear / else weakly_separated；
#   三对一致 -> 单一 cls，否则 mixed。混淆检验：R x S_class <15deg -> confounded_classword。
# 跨模型（descriptive）：每模型 R x {K_readout, K_entity, S_class 同配方重建} 并排（D 不可跨模型求角）。
# 对拍锚（装置自证，容差 0.05deg）：3165 result 4b K_read x K_ent 69.823 / x S_class 67.000 /
#   x S_attr 68.094 / x S_syntax 58.488；cross 14b 70.235 / glm4 79.01。
# dtype 链：H16(float16) -> float64 计算域；S_class 重建 verbatim（float32 round-trip，3165 主链）。
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

PHASE = 3166
NAME = 'g5a3b_logic_direction'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
B = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
BASE = os.path.join(B, 'phase3166', NAME)
SMOKE = os.environ.get('P3166_SMOKE', '') == '1'
LOG = []

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def log(s):
    line = '[3166] %s' % s
    LOG.append(line)
    try:
        print(line, flush=True)
    except Exception:
        pass


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def unit(x):
    n = float(np.linalg.norm(x))
    assert n > 0, 'zero vector'
    return x / n


# ---------------- 源锚（probe 实测 2026-10-09，见 tests/gpt5_temp/p3166_shas.txt） ----------------
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
    'p3151_collect': B + r'\phase3151\g1p1_combo_additive_vs_interaction\collect.npz',
    'p3152_4b_collect': B + r'\phase3152\g1p2_tri_model_k1\qwen3-4b\collect.npz',
    'p3152_14b_collect': B + r'\phase3152\g1p2_tri_model_k1\qwen3-14b\collect.npz',
    'p3152_4b_result': B + r'\phase3152\g1p2_tri_model_k1\qwen3-4b\result.json',
    'p3152_14b_result': B + r'\phase3152\g1p2_tri_model_k1\qwen3-14b\result.json',
    'p3152_glm4k1_result': B + r'\phase3152\g1p2_tri_model_k1\glm4k1\result.json',
    'p3165_result': B + r'\phase3165\g5a3_family_alignment\result.json',
}
SHA_ANCHOR = {
    'p3158_4b': 'c4d9b9fb', 'p3158_14b': '7f53f3a0', 'p3158_glm4': '2ea72cfe',
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p2874': '5f7796e1', 'p2878': '0bb3ec1e', 'p2881': 'abe74030',
    'p2806_exec': 'b291ea35',
    'p3151_collect': 'c711946c', 'p3152_4b_collect': '4ef190b7',
    'p3152_14b_collect': '733182c1',
    'p3152_4b_result': 'c38b97ff', 'p3152_14b_result': '23dfb646',
    'p3152_glm4k1_result': '111a1152',
    'p3165_result': '511d9b13',
}
# 3152 summary inputs_used 内嵌 res_sha8（对 result.json 内容的第二重锚）
RES_SHA_EMBED = {'p3152_4b_result': '199b145d', 'p3152_14b_result': '8a29eace',
                 'p3152_glm4k1_result': '52d05025'}

# 3151 冻结面板 verbatim（phase3151/3152 DESIGN 逐字）
CLASSES = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
TPL = {0: '{e}是一种{c}。',
       1: '{e}属于{c}这一类。',
       2: '{e}，一种常见的{c}。'}
ENTS = [e for cl in CLASSES for e in ENT[cl]]
CLS_OF = [CLASSES.index(cl) for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS)
NC = len(CLASSES)
NT = len(TPL)
PAIRS_FULL = [(i, c) for i in range(NE) for c in range(NC)]

MDIR_MAP = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'Qwen3-14B', 'glm4': 'glm4-9b-chat-hf'}

DESIGN = dict(
    phase=PHASE, name=NAME,
    panel=dict(classes=CLASSES, ents=ENTS, tpl={str(k): v for k, v in TPL.items()},
               pairs='41x6 full combo (3151 verbatim)',
               tpl_mean=True, panel_rows_full=738),
    arms=dict(true='(i, CLS_OF[i])', false='(i, c != CLS_OF[i])',
              note='counterfactual false arm already inside sealed 3151/3152 panel '
                   '(ledger 3165 pending reason was imprecise) -> zero-GPU reuse'),
    direction_construct=dict(
        per_entity='D_i = mean_t H[t,(i,true),k] - mean_{t,c!=true} H[t,(i,c),k]; '
                   'd_i = unit(D_i); center over 41; SVD Vh[:8] = R_logic (8,D) row-form',
        k_final='NL (last slot, final-norm pre, aligned with 3157 K_entity slot H[:,NL,:])',
        aux_slots=dict(k_kout='NL-1 (3152 readout slot, E_read replicate)',
                       k_kstar='3 (3152 K1 gate layer, KSTAR_FRAC=0.075 frozen)'),
        dtype_chain='H float16 -> float64 domain; S_class rebuild verbatim '
                    '(float32 round-trip -> float64, 3165 main chain)'),
    device_gates=dict(
        G0='mean(MARG_true true-class margin) > mean(MARG_false claimed-class margin)',
        G1='mean_i |D_i| >= 0.01 * mean(true rows |H_row,k|)',
        G2='strict leave-one-entity-out AUC >= 0.6 (fold rebuilds top8+disc from other 40)'),
    census_gates=dict(
        pairs=['R_logic__K_readout', 'R_logic__K_entity', 'R_logic__S_class'],
        rule='top1 principal angle: >=30deg separable / <15deg collinear / else weakly_separated',
        aggregate='all three agree -> single cls; else mixed; '
                  'R x S_class <15deg -> confounded_classword flag'),
    cross_model='descriptive per model R x {K_readout, K_entity, S_class rebuild}; '
                'S_attr/S_syntax stay 4b-only (2874/2878 vocab)',
    crosscheck_anchors=dict(
        deg_4b_Kread_Kent=69.823, deg_4b_Kread_Sclass=67.000, deg_4b_Kread_Sattr=68.094,
        deg_4b_Kread_Ssyntax=58.488, deg_14b_Kent_readout=70.235, deg_glm4_Kent_readout=79.01,
        tol_deg=0.05),
    source_sha8=SHA_ANCHOR,
    res_sha_embed=RES_SHA_EMBED,
    smoke=SMOKE,
)

CANON_KEYS = ('phase', 'name', 'panel', 'arms', 'direction_construct', 'device_gates',
              'census_gates', 'cross_model', 'crosscheck_anchors', 'source_sha8',
              'res_sha_embed')


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


# ---------------- S_class rebuild (3165 recipe verbatim, per-model W_U) ----------------
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


def build_S_class(W, mdir):
    """2881/3165 recipe verbatim: CATS(phase2806 exec) + single_tok + tid + centroid diff."""
    e2806 = json.load(io.open(SRC['p2806_exec'], encoding='utf-8'))
    CATS = e2806['cats']
    CAT_WORDS = list(CATS.keys())
    assert CAT_WORDS == ['fruit', 'animal', 'metal', 'vehicle', 'country',
                         'food', 'nature', 'furniture', 'tool', 'clothing']
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True,
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
    assert sum(len(v) for v in class_targets.values()) == 80, 'class_targets != 80'

    Erows = {w: W[tid(w)] for w in single_tok}
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
    """principal angles from row-form (k, D); cos desc, top1 deg."""
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


def build_R(Hk, ent_rows, y, n_ent):
    """per-entity contrastive directions -> unit -> center -> SVD top8.
    returns Dm (n_ent, D) raw, scales, sv, V8 (8, D)."""
    ds = []
    ents = sorted(ent_rows.keys())
    for i in ents:
        tr = [r for r in ent_rows[i] if y[r]]
        fa = [r for r in ent_rows[i] if not y[r]]
        assert tr and fa, ('empty arm', i)
        ds.append(Hk[tr].mean(0) - Hk[fa].mean(0))
    Dm = np.stack(ds)
    scales = np.linalg.norm(Dm, axis=1)
    DmU = np.stack([unit(v) for v in Dm])
    Dc = DmU - DmU.mean(0, keepdims=True)
    _, sv, Vh = np.linalg.svd(Dc, full_matrices=False)
    return Dm, scales, sv, Vh[:8]


def loeo_auc(Hk, ent_rows, y, n_ent):
    """strict LOEO: fold e rebuilds top8 + disc from other entities only."""
    aucs = []
    ents = sorted(ent_rows.keys())
    for e in ents:
        other = [i for i in ents if i != e]
        ds = []
        for i in other:
            tr = [r for r in ent_rows[i] if y[r]]
            fa = [r for r in ent_rows[i] if not y[r]]
            ds.append(unit(Hk[tr].mean(0) - Hk[fa].mean(0)))
        Dm = np.stack(ds)
        Dc = Dm - Dm.mean(0, keepdims=True)
        _, _, Vh = np.linalg.svd(Dc, full_matrices=False)
        V8 = Vh[:8]
        tr_rows = [r for i in other for r in ent_rows[i]]
        P = Hk[tr_rows] @ V8.T
        yt = y[tr_rows]
        w = P[yt].mean(0) - P[~yt].mean(0)
        he = ent_rows[e]
        sh = (Hk[he] @ V8.T) @ w
        yh = y[he]
        pos, neg = sh[yh], sh[~yh]
        n1, n0 = len(pos), len(neg)
        m = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        aucs.append(float(m) / (n1 * n0))
    return float(np.mean(aucs)), aucs


def cls_of_top1(t1):
    return 'separable' if t1 >= 30.0 else ('collinear' if t1 < 15.0 else 'weakly_separated')


def main():
    t0 = time.monotonic()
    dsha = freeze()
    for k, p in SRC.items():
        got = sha8(p)
        assert got == SHA_ANCHOR[k], ('source drift', k, got)
    log('source sha8 anchored: %d files OK' % len(SRC))
    for tag, want in RES_SHA_EMBED.items():
        r = json.load(io.open(SRC[tag], encoding='utf-8'))
        got = r.get('res_sha8')
        assert got == want, ('embedded res_sha8 drift', tag, got)
    log('embedded res_sha8 crosscheck OK (3152 x3 vs summary inputs_used)')

    z165 = json.load(io.open(SRC['p3165_result'], encoding='utf-8'))
    cc = DESIGN['crosscheck_anchors']
    tol = cc['tol_deg']

    def chk(tag, got, want):
        assert abs(got - want) <= tol, ('crosscheck fail', tag, got, want)
        log('crosscheck %s: %.3f == %.3f OK' % (tag, got, want))

    MODELS = [('qwen3-4b', SRC['p3152_4b_collect']),
              ('qwen3-14b', SRC['p3152_14b_collect']),
              ('glm4', SRC['p3151_collect'])]
    if SMOKE:
        MODELS = [('qwen3-4b', B + r'\phase3152\g1p2_tri_model_k1\qwen3-4b\smoke\collect_smoke.npz'),
                  ('glm4', B + r'\phase3151\g1p1_combo_additive_vs_interaction\smoke\collect_smoke.npz')]

    per_model = {}
    overall_cls = []
    overall_dev = []
    overall_confound = []

    for mname, hpath in MODELS:
        log('==== model %s (%s) ====' % (mname, os.path.basename(hpath)))
        mdir = os.path.join(ROOT, 'models', 'hf', MDIR_MAP[mname])
        cfg = json.load(io.open(os.path.join(mdir, 'config.json'), encoding='utf-8'))
        NL = int(cfg['num_hidden_layers'])
        Dm_cfg = int(cfg['hidden_size'])
        zm = np.load(hpath)
        H16 = zm['H']
        MARG = zm['marg']
        NTn, NPn, NHn, Dn = H16.shape
        assert NTn == NT and NHn == NL + 1 and Dn == Dm_cfg, ('panel shape', H16.shape)
        if SMOKE:
            keep_e = []
            cnt = {}
            for i, cl in enumerate(CLS_OF):
                if cnt.get(cl, 0) < 2:
                    keep_e.append(i)
                    cnt[cl] = cnt.get(cl, 0) + 1
            pairs = [(i, c) for (i, c) in PAIRS_FULL if i in set(keep_e)]
        else:
            pairs = list(PAIRS_FULL)
        assert NPn == len(pairs), ('PAIRS drift', NPn, len(pairs))
        n_ent = len(set(i for i, c in pairs))
        keep_ents = sorted(set(ei for ei, ci in pairs))
        ent_rows = {i: [] for i in keep_ents}
        idx_of_pair = {p: pi for pi, p in enumerate(pairs)}
        for i in keep_ents:
            for t in range(NTn):
                for pi, (ei, ci) in enumerate(pairs):
                    if ei == i:
                        ent_rows[i].append(t * NPn + pi)
        y = np.zeros(NTn * NPn, bool)
        for t in range(NTn):
            for pi, (ei, ci) in enumerate(pairs):
                if ci == CLS_OF[ei]:
                    y[t * NPn + pi] = True
        n_true = int(y.sum())
        n_false = int((~y).sum())
        log('panel rows=%d true=%d false=%d ents=%d NL=%d D=%d' %
            (NTn * NPn, n_true, n_false, n_ent, NL, Dn))

        # ---- K materials (3165 verbatim rebuild, same dtype chain) ----
        mk = {'qwen3-4b': '4b', 'qwen3-14b': '14b', 'glm4': 'glm4'}[mname]
        z58 = np.load(SRC['p3158_' + mk])
        top64 = z58['top64'].astype(np.float64)
        K_read = top64.T
        z57 = np.load(SRC['p3157_' + mk])
        H57 = z57['H'].astype(np.float64)
        NL57 = H57.shape[1] - 1
        X = H57[:, NL57, :]
        Xc = X - X.mean(0, keepdims=True)
        _, _, Vh57 = np.linalg.svd(Xc, full_matrices=False)
        K_ent = Vh57[:8]

        # ---- S_class rebuild (4b must match 3165; others descriptive try) ----
        scls_state = 'skipped'
        S_class = None
        try:
            W_m, _ = load_WU(mdir)
            dWc, class_words, single_tok = build_S_class(W_m, mdir)
            S_class = dWc.astype(np.float64)
            scls_state = 'ok(n=%d words=%d)' % (S_class.shape[0], len(single_tok))
            if mname == 'qwen3-4b':
                z81 = np.load(SRC['p2881'], allow_pickle=True)
                tl81 = [str(x) for x in z81['target_list']]
                w81 = [s.split(':')[-1] for s in tl81]
                assert w81[:80] == class_words, 'class word order drift'
                log('4b S_class word order vs 2881 target_list OK')
        except Exception as ex:
            scls_state = 'failed: %s' % ex
            log('S_class rebuild failed for %s: %s' % (mname, ex))

        # ---- R_logic at k_final = NL ----
        Hk = H16[:, :, NL, :].reshape(NTn * NPn, Dn).astype(np.float64)
        Dm, scales, sv, V8 = build_R(Hk, ent_rows, y, n_ent)
        R_logic = V8
        g1_ratio = float(scales.mean() / np.linalg.norm(Hk[y], axis=1).mean())
        g1_pass = bool(g1_ratio >= 0.01)

        # ---- G0 behavior contrast from MARG ----
        mt, mf = [], []
        for t in range(NTn):
            for pi, (ei, ci) in enumerate(pairs):
                mg = MARG[t, pi]
                if ci == CLS_OF[ei]:
                    mt.append(float(mg[ci]))
                else:
                    mf.append(float(mg[ci]))
        g0_diff = float(np.mean(mt) - np.mean(mf))
        g0_pass = bool(g0_diff > 0)

        # ---- G2 strict LOEO AUC ----
        auc_mean, aucs = loeo_auc(Hk, ent_rows, y, n_ent)
        g2_pass = bool(auc_mean >= 0.6)
        dev_ok = bool(g0_pass and g1_pass and g2_pass)
        log('device: G0 diff=%.4f pass=%s | G1 ratio=%.5f pass=%s | G2 AUC=%.4f pass=%s'
            % (g0_diff, g0_pass, g1_ratio, g1_pass, auc_mean, g2_pass))

        # ---- census (same gates as 3165) ----
        subs = {'R_logic': R_logic, 'K_readout': K_read, 'K_entity': K_ent}
        if S_class is not None:
            subs['S_class'] = S_class
        pair_res = {}
        names = list(subs.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, bn = names[i], names[j]
                s, t1 = kross(subs[a], subs[bn])
                cnt_e, pr = effdim(s)
                pair_res['%s__%s' % (a, bn)] = dict(
                    k=int(min(subs[a].shape[0], subs[bn].shape[0])),
                    top1_deg=round(t1, 3), cos=s.round(6).tolist(),
                    eff_ge05=cnt_e, pr=round(pr, 4))
        gate_pairs = ['R_logic__K_readout', 'R_logic__K_entity', 'R_logic__S_class']
        have_gp = [g for g in gate_pairs if g in pair_res]
        clsmap = [cls_of_top1(pair_res[g]['top1_deg']) for g in have_gp]
        agree = len(set(clsmap)) == 1
        census_cls = clsmap[0] if agree else 'mixed'
        confound = ('confounded_classword'
                    if ('R_logic__S_class' in pair_res and
                        pair_res['R_logic__S_class']['top1_deg'] < 15.0)
                    else 'not_confounded')
        log('census gate pairs: %s -> cls=%s confound=%s'
            % ({g: pair_res[g]['top1_deg'] for g in have_gp}, census_cls, confound))

        # ---- crosschecks vs 3165 (K family must reproduce) ----
        s_kk, t_kk = kross(K_read, K_ent)
        if mname == 'qwen3-4b':
            chk('4b Kread_Kent', t_kk, cc['deg_4b_Kread_Kent'])
            chk('4b Kread_Sclass', pair_res['K_readout__S_class']['top1_deg'],
                cc['deg_4b_Kread_Sclass'])
            z74 = np.load(SRC['p2874'], allow_pickle=True)
            S_attr = z74['dW_unit'].astype(np.float64)
            z78 = np.load(SRC['p2878'], allow_pickle=True)
            S_syn = z78['dW_unit'].astype(np.float64)
            s_a, t_a = kross(K_read, S_attr)
            s_y, t_y = kross(K_read, S_syn)
            chk('4b Kread_Sattr', t_a, cc['deg_4b_Kread_Sattr'])
            chk('4b Kread_Ssyntax', t_y, cc['deg_4b_Kread_Ssyntax'])
            s_ra, t_ra = kross(R_logic, S_attr)
            s_ry, t_ry = kross(R_logic, S_syn)
            pair_res['R_logic__S_attr'] = dict(
                k=8, top1_deg=round(t_ra, 3), cos=s_ra.round(6).tolist(),
                eff_ge05=int((s_ra ** 2 >= 0.5).sum()),
                pr=round(float(s_ra.sum() ** 2 / (s_ra ** 2).sum() + 1e-300), 4))
            pair_res['R_logic__S_syntax'] = dict(
                k=3, top1_deg=round(t_ry, 3), cos=s_ry.round(6).tolist(),
                eff_ge05=int((s_ry ** 2 >= 0.5).sum()),
                pr=round(float(s_ry.sum() ** 2 / (s_ry ** 2).sum() + 1e-300), 4))
        else:
            want = cc['deg_14b_Kent_readout'] if mname == 'qwen3-14b' else cc['deg_glm4_Kent_readout']
            chk('%s Kent_readout' % mname, t_kk, want)

        # ---- aux slots (descriptive) ----
        aux = {}
        for tag, kk in (('k_kout', NL - 1), ('k_kstar', 3)):
            Hka = H16[:, :, kk, :].reshape(NTn * NPn, Dn).astype(np.float64)
            _, sca, _, V8a = build_R(Hka, ent_rows, y, n_ent)
            s1, t1r = kross(V8a, K_read)
            s2, t1e = kross(V8a, K_ent)
            aux[tag] = dict(k_slot=int(kk), scale_mean=float(sca.mean()),
                            R_vs_Kreadout_top1=round(t1r, 3),
                            R_vs_Kentity_top1=round(t1e, 3))

        # ---- per-class 6 directions (descriptive) ----
        pc = {}
        if not SMOKE:
            Gc = []
            for c in range(NC):
                ents_c = [i for i in keep_ents if CLS_OF[i] == c]
                tr = [t * NPn + idx_of_pair[(i, c)] for i in ents_c for t in range(NTn)]
                fa = [t * NPn + idx_of_pair[(i, c2)]
                      for i in ents_c for t in range(NTn)
                      for c2 in range(NC) if c2 != c and (i, c2) in idx_of_pair]
                if tr and fa:
                    Gc.append(unit(Hk[tr].mean(0) - Hk[fa].mean(0)))
            if len(Gc) == NC:
                Gm = np.stack(Gc)
                s_pc, t_pc = kross(Gm, K_read)
                s_pce, t_pce = kross(Gm, K_ent)
                pc = dict(n=len(Gc), vs_Kreadout_top1=round(t_pc, 3),
                          vs_Kentity_top1=round(t_pce, 3))

        rec = dict(D=Dn, NL=NL, k_final=NL, panel_rows=int(NTn * NPn),
                   n_true_rows=n_true, n_false_rows=n_false, n_entities=n_ent,
                   S_class_rebuild=scls_state,
                   device=dict(G0=dict(margin_true=float(np.mean(mt)),
                                       margin_false=float(np.mean(mf)),
                                       diff=g0_diff, **{'pass': g0_pass}),
                               G1=dict(ratio=g1_ratio, **{'pass': g1_pass}),
                               G2=dict(auc_mean=round(auc_mean, 4),
                                       aucs=[round(a, 4) for a in aucs],
                                       **{'pass': g2_pass}),
                               device_cls='device_ok' if dev_ok else 'device_ineffective'),
                   R_logic=dict(singular_values=[round(float(v), 6) for v in sv],
                                top1_share=round(float(sv[0] ** 2 / (sv ** 2).sum()), 4)),
                   census=pair_res, census_cls=census_cls, confound=confound,
                   crosscheck_Kread_Kent_top1=round(t_kk, 3),
                   aux_slots=aux, perclass=pc)
        per_model[mname] = rec
        overall_cls.append(census_cls)
        overall_dev.append(rec['device']['device_cls'])
        overall_confound.append(confound)

    agree_cls = len(set(overall_cls)) == 1
    final_cls = overall_cls[0] if agree_cls else 'mixed'
    dev_all = all(d == 'device_ok' for d in overall_dev)
    confound_any = any(c == 'confounded_classword' for c in overall_confound)
    final_confound = 'confounded_classword' if confound_any else 'not_confounded'
    elapsed = round(time.monotonic() - t0, 1)
    verdict = 'g5a3b_%s|%s|%s|sha8_pending' % (
        ('device_ok_x%d' % len(overall_dev)) if dev_all else 'device_' + '/'.join(overall_dev),
        ('%s_x%d' % (final_cls, len(overall_cls))) if agree_cls else 'mixed:' + '/'.join(overall_cls),
        final_confound)
    result = dict(
        phase=PHASE, name=NAME, design_sha8=dsha, smoke=SMOKE,
        models=[m for m, _ in MODELS], per_model=per_model,
        overall=dict(census_cls=final_cls, cls_agree=agree_cls,
                     census_per_model=overall_cls,
                     device='device_ok_all' if dev_all else '/'.join(overall_dev),
                     confound=final_confound),
        verdict=verdict, elapsed_s=elapsed,
        source_sha8=SHA_ANCHOR)
    log('VERDICT %s' % verdict)

    if SMOKE:
        seal_result(result, 'smoke_result.json')
        with io.open(os.path.join(BASE, 'smoke_run_log.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(LOG) + '\n')
        return

    res8, seal8 = seal_result(result, 'result.json')
    result['verdict'] = verdict.replace('sha8_pending', 'sha8_' + res8)
    with io.open(os.path.join(BASE, 'result.json'), 'w', encoding='utf-8',
                 newline='\r\n') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True))
    with io.open(os.path.join(BASE, 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    log('DONE elapsed=%.1fs res=%s seal=%s' % (elapsed, res8, seal8))


if __name__ == '__main__':
    main()
