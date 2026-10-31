# -*- coding: utf-8 -*-
"""Phase 3281 (G5-H44): consumer attribution, share-normalized (preregistered
in #432 as the correction to 3280's K-A/K-A2 FAIL).

Root cause established in 3280: dynamic D(l,h)=mean_rows||proj_B(c_h)|| is an
ABSOLUTE norm — confounded by per-head output magnitude ||c_h||. Correction
(preregistered verbatim): D_norm = mean_rows( ||proj_B(c_h)|| / ||c_h|| ),
i.e. the SHARE of the head's output that lands in the payload subspace.

Guards (frozen in DESIGN before any run):
  rows with ||c_h|| < 1e-6 are EXCLUDED from the mean (silent row carries no
  share information); a head silent in ALL rows gets D_norm = 0.
Arms:
  a  static OV write analysis — IDENTICAL to 3280 (S_rel = S_abs/||W_OV||_F,
     Gram-trace norm, null = 20 random 8-dim orthonormal bases)
  b  dynamic share-normalized verification: one bf16 forward over the p3157
     128-row panel, per-head share projected on B at the last token
  c  report-only confound evidence: spearman(D_abs, mean||c||) and
     spearman(S_rel, mean||c||) over all 1152 heads — quantifies the magnitude
     confound that broke 3280
Gates (SAME thresholds as 3280, per preregistration):
  gA  p3166 census anchors reproduced (tol 0.05deg)
  gB  static table sane + top1 S_rel > null p95
  K-A overlap(top32 S_rel, top32 D_norm) >= 10 BOTH subspaces (null ~0.9)
  K-A2 spearman(S_rel, D_norm) >= 0.25 BOTH subspaces
  K-B report-only: layer distribution of dynamic top-64 by D_norm
Zero write to atlas_ledger.json (closeout separate).
"""
import hashlib
import json
import os
import time

import numpy as np

PHASE = 3281
NAME = 'g5h44_consumer_attribution_norm'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RB = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
BASE = os.path.join(RB, 'phase3281', NAME)
SMOKE = os.environ.get('P3281_SMOKE', '') == '1'
SMOKE_DIR = os.path.join(BASE, 'smoke')

L = []


def log(s):
    t = time.strftime('%H:%M:%S')
    L.append('[%s] %s' % (t, s))
    print('[%s] %s' % (t, s), flush=True)


def s8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def today():
    import datetime
    return datetime.date.today().isoformat()


MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
NL, HID, NQ, NKV, HD = 36, 2560, 32, 8, 128

# ---------------- panel material (3151/3152 + 3157 verbatim, == 3280) ------
CLASSES = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
ENTS = [e for cl in CLASSES for e in ENT[cl]]
CLS_OF = [CLASSES.index(cl) for cl in CLASSES for e in ENT[cl]]
NE, NC, NT = len(ENTS), 6, 3
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)
ENTS3157 = [["桌子", "家具", "桌腿"], ["椅子", "座椅", "椅背"], ["窗户", "建筑构件", "窗框"],
            ["门", "建筑构件", "门把手"], ["硬币", "货币", "图案"], ["轮胎", "橡胶配件", "花纹"],
            ["纸", "材料", "纤维"], ["书", "出版物", "书页"], ["刀", "工具", "刀刃"],
            ["碗", "餐具", "碗底"], ["衣服", "服饰", "袖子"], ["鞋", "日用品", "鞋底"],
            ["桥", "建筑", "桥墩"], ["房子", "建筑", "屋顶"], ["汽车", "交通工具", "车轮"],
            ["钟表", "计时工具", "表盘"]]
PREFIX_ZH = '今天天气很好，我们在讨论语言模型如何表示语言。'
K_CTX = 16
TPL3157 = {('isa', '+'): '{E}是一种{C}。', ('isa', '-'): '{E}不是一种{C}。',
           ('hasa', '+'): '{E}有{P}。', ('hasa', '-'): '{E}没有{P}。'}
RELS, POLS = ['isa', 'hasa'], ['+', '-']
ROWS3157 = []
for ei, ent in enumerate(ENTS3157):
    for ri, rel in enumerate(RELS):
        for pi, pol in enumerate(POLS):
            for ci in (0, 1):
                ROWS3157.append(dict(ei=ei, rel=rel, pol=pol, ctx=ci,
                                     prompt=TPL3157[(rel, pol)]
                                     .replace('{E}', ent[0])
                                     .replace('{C}', ent[1])
                                     .replace('{P}', ent[2])))
assert len(ROWS3157) == 128

SRC_3152 = os.path.join(RB, 'phase3152', 'g1p2_tri_model_k1', 'qwen3-4b', 'collect.npz')
SRC_3157 = os.path.join(RB, 'phase3157', 'g2p2_transform_algebra_commutator',
                        'qwen3-4b', 'collect.npz')
P3166 = os.path.join(RB, 'phase3166', 'g5a3b_logic_direction', 'result.json')


def unit(x):
    n = np.linalg.norm(x)
    assert n > 0
    return x / n


def kross(A, Bd):
    """3165/3166 kross: principal angles between row spaces."""
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(Bd.T)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return s, float(np.degrees(np.arccos(s[0])))


def build_r_logic(H16, nl):
    """3166 build_R verbatim (from 3278/3280)."""
    NTn, NPn, NHn, Dn = H16.shape
    ent_rows = {i: [] for i in range(NE)}
    for i in range(NE):
        for t in range(NTn):
            for pi, (ei, ci) in enumerate(PAIRS):
                if ei == i:
                    ent_rows[i].append(t * NPn + pi)
    y = np.zeros(NTn * NPn, bool)
    for t in range(NTn):
        for pi, (ei, ci) in enumerate(PAIRS):
            if ci == CLS_OF[ei]:
                y[t * NPn + pi] = True
    Hk = H16[:, :, nl, :].reshape(NTn * NPn, Dn).astype(np.float64)
    ds = []
    for i in range(NE):
        tr = [r for r in ent_rows[i] if y[r]]
        fa = [r for r in ent_rows[i] if not y[r]]
        assert tr and fa
        ds.append(Hk[tr].mean(0) - Hk[fa].mean(0))
    Dm = np.stack(ds)
    DmU = np.stack([unit(v) for v in Dm])
    Dc = DmU - DmU.mean(0, keepdims=True)
    _, sv, Vh = np.linalg.svd(Dc, full_matrices=False)
    return Vh[:8], sv


def build_k_entity(H57):
    X = H57[:, -1, :]
    Xc = X - X.mean(0, keepdims=True)
    _, sv, Vh = np.linalg.svd(Xc, full_matrices=False)
    return Vh[:8], sv


# ---------------------------------------------------------------- DESIGN
DESIGN = dict(
    phase=PHASE, name=NAME,
    prereg='G5-H44 (share-normalized consumer attribution; preregistered in '
           'ledger #432 as the correction to 3280 K-A/K-A2 FAIL: dynamic '
           'absolute norm confounded by per-head output magnitude). SAME gate '
           'thresholds as 3280.',
    arms=dict(
        a='static OV write analysis — IDENTICAL to 3280: S_abs=||B^T W_OV B||_F, '
          'S_rel=S_abs/||W_OV||_F (Gram-trace norm); null = 20 random 8-dim '
          'orthonormal bases',
        b='dynamic SHARE-normalized verification: one bf16 forward over p3157 '
          '128-row panel; per-row share = ||proj_B(c_h)||/||c_h||; D_norm = '
          'mean_rows(share) with rows ||c_h||<1e-6 EXCLUDED (all-silent head '
          '-> 0). Guards frozen here, before any run.',
        c='report-only confound evidence: spearman(D_abs, mean||c||) and '
          'spearman(S_rel, mean||c||) over 1152 heads (quantifies the 3280 '
          'magnitude confound); 3280-style absolute gates recomputed '
          'report-only for continuity'),
    subspaces=['R_logic(8,末层基,3152 collect 重建)', 'K_entity(8,末层基,3157 collect 重建)'],
    gates=dict(
        gA='p3166 qwen3-4b census anchors reproduced (tol 0.05deg)',
        gB='static table sane + top1 S_rel > null p95 (real structure)',
        KA='overlap(top32 S_rel, top32 D_norm) >= 10 BOTH subspaces (null ~0.9)',
        KA2='spearman(S_rel, D_norm) >= 0.25 BOTH subspaces',
        KB='report-only layer distribution of dynamic top-64 by D_norm'),
    limits='末层基跨层复用（坐标系漂移已知）；份额口径排除静默行（<1e-6）；'
           '若 K-A/K-A2 仍 FAIL，结论=静态写入强度不预测动态份额（注意力活跃度'
           '门控主导），如实 BLOCKED 入账——这也是发现不是失败',
    source_sha8=None, smoke=SMOKE)
DESIGN['source_sha8'] = {'p3152_4b': s8(SRC_3152), 'p3157_4b': s8(SRC_3157),
                         'p3166': s8(P3166),
                         'p3280': s8(os.path.join(
                             RB, 'phase3280', 'g5h43_consumer_attribution',
                             'result.json'))}
CANON = ('phase', 'name', 'prereg', 'arms', 'subspaces', 'gates', 'limits',
         'source_sha8', 'smoke')


def canon_sha(d):
    return hashlib.sha256(json.dumps(
        {k: d[k] for k in CANON}, sort_keys=True, ensure_ascii=False,
    ).encode('utf-8')).hexdigest()[:8]


def freeze():
    os.makedirs(BASE, exist_ok=True)
    if SMOKE:
        os.makedirs(SMOKE_DIR, exist_ok=True)
    fp = os.path.join(SMOKE_DIR, 'execution_smoke.json') if SMOKE \
        else os.path.join(BASE, 'execution.json')
    with open(fp, 'w', encoding='utf-8', newline='') as f:
        json.dump(DESIGN, f, ensure_ascii=False, indent=1)
    with open(os.path.join(BASE, 'design_sha8.txt'), 'w',
              encoding='utf-8', newline='') as f:
        f.write(canon_sha(DESIGN))
    log(('SMOKE ' if SMOKE else '') + 'freeze ' + canon_sha(DESIGN))


def main():
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    torch.manual_seed(0)
    freeze()
    t0 = time.time()
    gates = {}

    # ---------------- gA: rebuild payload bases + reproduce 3166 census ----
    z52 = np.load(SRC_3152)
    H52 = z52['H']
    assert H52.shape == (NT, NP_, NL + 1, HID), H52.shape
    z57 = np.load(SRC_3157)
    H57 = z57['H']
    assert H57.shape == (128, NL + 1, HID), H57.shape
    R_logic, sv_r = build_r_logic(H52, NL)
    K_ent, sv_e = build_k_entity(H57.astype(np.float64))
    r16 = json.load(open(P3166, encoding='utf-8'))
    cen16 = r16['per_model']['qwen3-4b']['census']
    Bases = {'R_logic': R_logic, 'K_entity': K_ent}
    cen_now = {'R_logic__K_entity': round(kross(R_logic, K_ent)[1], 3)}
    idx = json.load(open(os.path.join(MDIR, 'model.safetensors.index.json'),
                         encoding='utf-8'))['weight_map']
    lm_key = 'lm_head.weight' if 'lm_head.weight' in idx \
        else 'model.embed_tokens.weight'   # qwen3-4b: tied embeddings
    with safe_open(os.path.join(MDIR, idx[lm_key]), framework='pt') as f:
        lm = f.get_tensor(lm_key).float().numpy().astype(np.float64)
    assert lm.shape == (151936, HID), lm.shape
    G_D = lm.T @ lm
    evals, evecs = np.linalg.eigh(G_D)
    K_read = evecs[:, ::-1][:, :64].T.copy().astype(np.float64)
    Bases['K_readout'] = K_read
    cen_now['R_logic__K_readout'] = round(kross(R_logic, K_read)[1], 3)
    cen_now['K_readout__K_entity'] = round(kross(K_read, K_ent)[1], 3)
    ok_a = all(abs(cen_now[k] - cen16[k]['top1_deg']) < 0.05
               for k in ('R_logic__K_entity', 'R_logic__K_readout',
                         'K_readout__K_entity'))
    gates['gA_anchors'] = bool(ok_a)
    log('gA census now=%s anchors=%s' % (
        cen_now, {k: cen16[k]['top1_deg'] for k in cen_now}))
    del G_D, evals, evecs

    # ---------------- load attention weights (36 layers x v/o proj) --------
    log('loading attention weights')
    V_all = np.zeros((NL, NKV * HD, HID), np.float32)
    O_all = np.zeros((NL, HID, NQ * HD), np.float32)
    for sh in sorted(set(idx.values())):
        with safe_open(os.path.join(MDIR, sh), framework='pt') as f:
            for k in f.keys():
                if '.self_attn.v_proj.weight' in k:
                    l = int(k.split('.')[2])
                    V_all[l] = f.get_tensor(k).float().numpy()
                elif '.self_attn.o_proj.weight' in k:
                    l = int(k.split('.')[2])
                    O_all[l] = f.get_tensor(k).float().numpy()
    assert np.isfinite(V_all).all() and np.isfinite(O_all).all()

    # ---------------- Arm a: static OV write analysis (GPU, == 3280) -------
    dev = 'cuda'
    Bt = {nm: torch.tensor(B, dtype=torch.float32, device=dev)
          for nm, B in Bases.items() if nm != 'K_readout'}
    S_rel = {nm: np.zeros((NL, NQ), np.float64) for nm in Bt}
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(20):
        Mr = rng.standard_normal((HID, 8))
        nulls.append(np.linalg.qr(Mr)[0])
    for l in range(NL):
        Ot = torch.tensor(O_all[l], dtype=torch.float32, device=dev)
        Vt = torch.tensor(V_all[l], dtype=torch.float32, device=dev)
        Go = Ot.T @ Ot
        Gv = Vt @ Vt.T
        Go3 = Go.view(NQ, HD, NQ, HD)
        Gv3 = Gv.view(NKV, HD, NKV, HD)
        for h in range(NQ):
            kv = h // (NQ // NKV)
            wov_f2 = max(float((Go3[h, :, h, :] * Gv3[kv, :, kv, :]).sum()), 1e-30)
            Oh = Ot[:, h * HD:(h + 1) * HD]
            Vh = Vt[kv * HD:(kv + 1) * HD, :]
            for nm, Bm in Bt.items():
                M = Bm @ Oh @ Vh @ Bm.T
                S_rel[nm][l, h] = float(M.norm()) / np.sqrt(wov_f2)
    null_layers = [2, 12, 22, 32]
    null_all = {nm: [] for nm in Bt}
    for bi in range(20):
        Bn = nulls[bi]
        Bn_t = torch.tensor(Bn.T, dtype=torch.float32, device=dev)
        for l in null_layers:
            Ot = torch.tensor(O_all[l], dtype=torch.float32, device=dev)
            Vt = torch.tensor(V_all[l], dtype=torch.float32, device=dev)
            Go = Ot.T @ Ot
            Gv = Vt @ Vt.T
            Go3 = Go.view(NQ, HD, NQ, HD)
            Gv3 = Gv.view(NKV, HD, NKV, HD)
            for nm in Bt:
                Bm = Bt[nm]
                for h in range(NQ):
                    kv = h // (NQ // NKV)
                    M = Bm @ Ot[:, h * HD:(h + 1) * HD] @ \
                        Vt[kv * HD:(kv + 1) * HD, :] @ Bm.T
                    f2 = max(float((Go3[h, :, h, :] * Gv3[kv, :, kv, :]).sum()), 1e-30)
                    null_all[nm].append(float(M.norm()) / np.sqrt(f2))
    null_mean = {nm: float(np.mean(v)) for nm, v in null_all.items()}
    null_p95 = {nm: float(np.percentile(v, 95)) for nm, v in null_all.items()}
    del V_all
    gates['gB_static_sane'] = bool(
        np.isfinite(S_rel['R_logic']).all() and np.isfinite(S_rel['K_entity']).all()
        and null_mean['R_logic'] > 0)
    top_static = {}
    for nm in Bt:
        flat = S_rel[nm].flatten()
        order = np.argsort(flat)[::-1]
        top_static[nm] = [(int(i // NQ), int(i % NQ), round(float(flat[i]), 4))
                          for i in order[:16]]
        log('static top8 %s: %s null_mean=%.5f p95=%.5f' % (
            nm, top_static[nm][:8], null_mean[nm], null_p95[nm]))

    # ---------------- Arm b: dynamic SHARE-normalized (one forward) --------
    from transformers import AutoModelForCausalLM
    log('bf16 model load begin')
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, dtype=torch.bfloat16, trust_remote_code=True,
        device_map='cuda').eval()
    tok = AutoTokenizer.from_pretrained(MDIR, local_files_only=True,
                                        trust_remote_code=True, use_fast=True)
    pre_ids = tok(PREFIX_ZH, add_special_tokens=False)['input_ids']
    ctx_ids = (pre_ids * (K_CTX // len(pre_ids) + 1))[:K_CTX]
    assert len(ctx_ids) == K_CTX

    rows = ROWS3157 if not SMOKE else ROWS3157[:8]
    # per row we store BOTH proj norms and output norms; shares computed after
    rec = {nm: np.zeros((NL, len(rows), NQ), np.float32) for nm in Bt}
    cnorm = np.zeros((NL, len(rows), NQ), np.float32)
    hooks = []
    per_layer = {}

    def make_hook(l, Ot):
        Ot_t = torch.tensor(Ot, dtype=torch.float32, device=dev)
        Oh_all = Ot_t.view(HID, NQ, HD).permute(1, 0, 2).contiguous()

        def hook(module, args):
            x = args[0][0, -1, :].float()    # (NQ*HD,) last token
            X = x.view(NQ, HD)
            c_all = torch.einsum('hd,hed->he', X, Oh_all)   # (NQ, HID)
            outs = {'c': torch.linalg.norm(c_all, dim=1)}   # (NQ,)
            for nm, Bm in Bt.items():
                proj = c_all @ Bm.T          # (NQ, 8)
                outs[nm] = torch.linalg.norm(proj, dim=1)
            per_layer.setdefault(l, []).append(outs)
        return hook

    for l in range(NL):
        hndl = model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
            make_hook(l, O_all[l]))
        hooks.append(hndl)
    log('hooks registered; forward over %d rows' % len(rows))
    with torch.no_grad():
        for i, r in enumerate(rows):
            ids = (list(ctx_ids) if r['ctx'] == 1 else []) + \
                  tok(r['prompt'], add_special_tokens=False)['input_ids']
            ii = torch.tensor([ids], dtype=torch.int64, device='cuda')
            model(input_ids=ii)
            if (i + 1) % 32 == 0:
                log('arm b forward %d/%d' % (i + 1, len(rows)))
    for hndl in hooks:
        hndl.remove()
    del model
    del O_all
    torch.cuda.empty_cache()

    for l in range(NL):
        acc = per_layer[l]
        assert len(acc) == len(rows)
        cnorm[l] = torch.stack([a['c'] for a in acc]).cpu().numpy()
        for nm in Bt:
            rec[nm][l] = torch.stack([a[nm] for a in acc]).cpu().numpy()

    # shares: D_norm = mean over rows with ||c||>=1e-6 of ||proj||/||c||
    GUARD = 1e-6
    mask = cnorm >= GUARD                                   # (NL, R, NQ)
    D_norm = {nm: np.zeros((NL, NQ), np.float64) for nm in Bt}
    D_abs = {nm: np.zeros((NL, NQ), np.float64) for nm in Bt}
    mean_c = cnorm.mean(axis=1)                             # (NL, NQ)
    silent_all = 0
    for nm in Bt:
        share = np.where(mask, rec[nm] / np.maximum(cnorm, GUARD), 0.0)
        denom = np.maximum(mask.sum(axis=1), 1)
        D_norm[nm] = share.sum(axis=1) / denom
        D_abs[nm] = rec[nm].mean(axis=1)
        silent_all += int(((~mask).all(axis=1)).sum())
    log('silent-all heads (excluded, D_norm=0): %d' % silent_all)

    # ---------------- gates K-A / K-A2 / K-B (SAME thresholds) -------------
    from scipy.stats import spearmanr
    gates_k = {}
    KB_report = {}
    for nm in ('R_logic', 'K_entity'):
        s_flat = S_rel[nm].flatten()
        d_flat = D_norm[nm].flatten()
        top_s = set(np.argsort(s_flat)[::-1][:32].tolist())
        top_d = set(np.argsort(d_flat)[::-1][:32].tolist())
        ov = len(top_s & top_d)
        rho = float(spearmanr(s_flat, d_flat).statistic)
        gates_k[nm] = dict(overlap=ov, spearman=round(rho, 4),
                           pass_A=ov >= 10, pass_A2=rho >= 0.25)
        order_d = np.argsort(d_flat)[::-1][:64]
        late = sum(1 for i in order_d if i // NQ >= 28)
        KB_report[nm] = dict(
            top64_late_share=round(late / 64, 4),
            top8=[(int(i // NQ), int(i % NQ), round(float(d_flat[i]), 5))
                  for i in order_d[:8]])
        log('K %s overlap=%d spearman=%.4f late_share=%.3f' % (
            nm, ov, rho, late / 64))
    gates['KA_overlap'] = all(v['pass_A'] for v in gates_k.values())
    gates['KA2_spearman'] = all(v['pass_A2'] for v in gates_k.values())

    # ---------------- Arm c (report-only): confound evidence ---------------
    arm_c = {}
    for nm in ('R_logic', 'K_entity'):
        s_flat = S_rel[nm].flatten()
        a_flat = D_abs[nm].flatten()
        c_flat = mean_c.flatten()
        rho_abs_c = float(spearmanr(a_flat, c_flat).statistic)
        rho_s_c = float(spearmanr(s_flat, c_flat).statistic)
        # 3280-style absolute gates, recomputed report-only
        top_a = set(np.argsort(a_flat)[::-1][:32].tolist())
        top_s = set(np.argsort(s_flat)[::-1][:32].tolist())
        arm_c[nm] = dict(
            spearman_Dabs_meanC=round(rho_abs_c, 4),
            spearman_Srel_meanC=round(rho_s_c, 4),
            abs_overlap_3280style=len(top_s & top_a),
            abs_spearman_3280style=round(float(spearmanr(s_flat, a_flat).statistic), 4))
        log('armC %s %s' % (nm, arm_c[nm]))
    spec_ok = {}
    for nm in ('R_logic', 'K_entity'):
        spec_ok[nm] = bool(float(S_rel[nm].flatten().max()) > null_p95[nm])
    gates['gB_specificity'] = all(spec_ok.values())
    log('specificity top1 vs null p95: %s' % spec_ok)

    # ---------------- verdict + result --------------------------------------
    core = ['gA_anchors', 'gB_static_sane', 'gB_specificity',
            'KA_overlap', 'KA2_spearman']
    gate_all = all(gates[k] for k in core)
    res = {
        'phase': PHASE, 'name': NAME, 'line': 'G', 'date': today(),
        'design_sha8': canon_sha(DESIGN), 'mode': 'smoke' if SMOKE else 'formal',
        'gates': gates, 'gates_k': gates_k, 'KB_report': KB_report,
        'arm_c_confound': arm_c,
        'silent_all_heads': silent_all,
        'guard': GUARD,
        'census_now': cen_now,
        'census_anchor': {k: cen16[k]['top1_deg'] for k in cen_now},
        'null': {'mean': null_mean, 'p95': null_p95},
        'top_static': top_static,
        'elapsed_s': round(time.time() - t0, 1),
        'gate_all': gate_all,
        'verdict': 'consumer_attribution_norm_registered' if gate_all else 'BLOCKED',
    }
    if not SMOKE:
        res['res_sha8'] = hashlib.sha256(json.dumps(
            {k: v for k, v in res.items() if k != 'res_sha8'},
            sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
        with open(os.path.join(BASE, 'result.json'), 'w',
                  encoding='utf-8', newline='') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        np.savez_compressed(
            os.path.join(BASE, 'head_write_map.npz'),
            S_rel_R_logic=S_rel['R_logic'], S_rel_K_entity=S_rel['K_entity'],
            Dnorm_R_logic=D_norm['R_logic'], Dnorm_K_entity=D_norm['K_entity'],
            Dabs_R_logic=D_abs['R_logic'], Dabs_K_entity=D_abs['K_entity'],
            meanC=mean_c)
    else:
        with open(os.path.join(SMOKE_DIR, 'result_smoke.json'), 'w',
                  encoding='utf-8', newline='') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    with open(os.path.join(BASE, 'run_log.txt' if not SMOKE
                           else os.path.join('smoke', 'run_log.txt')), 'w',
              encoding='utf-8', newline='') as f:
        f.write('\n'.join(L))
    print('verdict', res['verdict'])
    for k, v in gates.items():
        print(' ', k, v)
    print('SMOKE DONE' if SMOKE else 'FORMAL DONE')


if __name__ == '__main__':
    main()
