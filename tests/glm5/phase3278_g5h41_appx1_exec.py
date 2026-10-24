# -*- coding: utf-8 -*-
"""Phase 3278 (G5-H41): APPX-1 execution — locate the two shared dims of
14b R_logic x K_entity (eff_ge05=2, unique to 14b; p3166 anchor 448af595 /
census 26.641deg) and explain their origin, with owner-approved nf4 baseline
(Qwen3-14B-bnb-nf4, bnb 0.49.2, lm_head fp16 unquantized).

Arm a (zero-GPU, sealed anchors): rebuild R_logic (p3152 14b collect, k_final
=NL) + K_entity (p3157 14b) + K_readout (p3158 14b top64); reproduce 3166
census (26.641/2, 71.796/0, 70.235/0, tol 0.05deg); localize shared dims via
cross-gram SVD (R_logic coords x K_ent coords), layer profile (last-slot
emergence), vocab projection (W_U nf4-dir fp16).

Arm b (nf4 GPU): re-collect 3152 panel (738 rows) + 3157 panel (128 rows)
under bnb-nf4; K_readout_nf4 from nf4-dir lm_head fp16 (3158 recipe);
fidelity gate G-B (top64 top1 angle diff <= 1.0deg vs fp anchor); conserva-
tion gate G-C (R_logic_nf4 x K_ent_nf4: eff_ge05==2, top1 within 3.0deg of
26.641, shared-dim cos diff <= 0.05); mixed pairs descriptive.
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

PHASE = 3278
NAME = 'g5h41_appx1_exec'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
B = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
BASE = os.path.join(B, 'phase3278', NAME)
SMOKE = os.environ.get('P3278_SMOKE', '') == '1'
SMOKE_DIR = os.path.join(BASE, 'smoke')

L = []


def log(s):
    line = '[3278] %s' % s
    L.append(str(line))
    try:
        print(line, flush=True)
    except Exception:
        pass


def s8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def unit(x):
    n = float(np.linalg.norm(x))
    return x / n if n > 0 else x


def kross(A, Bd):
    Qa, _ = np.linalg.qr(np.asarray(A, dtype=np.float64).T)
    Qb, _ = np.linalg.qr(np.asarray(Bd, dtype=np.float64).T)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return s, float(np.degrees(np.arccos(s[0])))


# ---------------- panel material (3151/3152 + 3157 verbatim) ----------------
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
TPL3152 = {0: '{e}是一种{c}。',
           1: '{e}属于{c}这一类。',
           2: '{e}，一种常见的{c}。'}
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

SRC_3152 = B + r'\phase3152\g1p2_tri_model_k1\qwen3-14b\collect.npz'
SRC_3157 = B + r'\phase3157\g2p2_transform_algebra_commutator\qwen3-14b\collect.npz'
SRC_3158 = B + r'\phase3158\g4p1_output_equivalence_class\qwen3-14b\collect.npz'
MDIR_FP = os.path.join(ROOT, 'models', 'hf', 'Qwen3-14B')
MDIR_NF4 = os.path.join(ROOT, 'models', 'hf', 'Qwen3-14B-bnb-nf4')

ANCHORS = {'R_logic__K_entity': (26.641, 2), 'R_logic__K_readout': (71.796, 0),
           'K_readout__K_entity': (70.235, 0)}
TOL_ANCHOR = 0.05
TOL_FID = 1.0      # G-B deg
TOL_CONS = 3.0     # G-C deg
TOL_COS = 0.05     # G-C shared-dim cos

DESIGN = dict(
    phase=PHASE, name=NAME,
    prereg='G5-H41 (owner approved nf4 baseline 2026-10-21; APPX-1 gap '
           'v1.15 appendix_open, anchor p3166 448af595)',
    arms=dict(
        a='zero-GPU: rebuild 14b R_logic/K_entity/K_readout from sealed '
          'collects; reproduce 3166 census anchors (tol 0.05deg); localize '
          '2 shared dims (cross-gram SVD); layer profile + vocab projection',
        b='nf4 GPU: re-collect 3152 panel (738 rows) + 3157 panel (128 rows) '
          'under Qwen3-14B-bnb-nf4; K_readout_nf4 from fp16 lm_head; G-B '
          'fidelity (top1 diff <=1.0deg); G-C conservation (eff_ge05==2, '
          'top1 within 3.0deg, shared-dim cos diff <=0.05); mixed pairs '
          'descriptive'),
    panels=dict(p3152='41x6 x3 tpl =738 rows, k_final=NL=40',
                p3157='16 ents x2rel x2pol x2ctx =128 rows, last slot'),
    nf4=dict(dir='Qwen3-14B-bnb-nf4', bnb='0.49.2',
             lm_head='fp16 unquantized (probe verified)'),
    gates=dict(G_A='anchors reproduced (tol 0.05deg)',
               G_B='K_readout_nf4 vs fp top1 angle diff <= %.1fdeg' % TOL_FID,
               G_C='conservation: eff_ge05==2, top1 within %.1fdeg, '
                   'shared cos diff <= %.2f' % (TOL_CONS, TOL_COS)),
    source_sha8={k: None for k in ('p3152_14b', 'p3157_14b', 'p3158_14b')},
    smoke=SMOKE,
)
DESIGN['source_sha8'] = {
    'p3152_14b': '733182c1', 'p3157_14b': '9552086d', 'p3158_14b': '7f53f3a0'}

CANON = ('phase', 'name', 'prereg', 'arms', 'panels', 'nf4', 'gates',
         'source_sha8', 'smoke')


def canon_sha(d):
    blob = json.dumps({k: d[k] for k in CANON}, ensure_ascii=False,
                      indent=1, sort_keys=True).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()[:8]


def freeze():
    os.makedirs(BASE, exist_ok=True)
    if SMOKE:
        os.makedirs(SMOKE_DIR, exist_ok=True)
    # 3263 lesson: SMOKE and formal freeze in SEPARATE files
    p = os.path.join(SMOKE_DIR if SMOKE else BASE, 'execution.json')
    if os.path.exists(p):
        old = json.load(io.open(p, encoding='utf-8'))
        for k in CANON:
            assert old.get(k) == DESIGN[k], ('design drift', k)
        return old['design_sha8']
    d8 = canon_sha(DESIGN)
    with io.open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(dict(DESIGN, design_sha8=d8), ensure_ascii=False,
                           indent=1, sort_keys=True))
    log('freeze design_sha8=%s' % d8)
    return d8


def build_r_logic(H16, nl):
    """3166 build_R verbatim: k_final=NL contrastive dirs -> unit -> center
    -> SVD top8. H16 (NT, NP, NL+1, D) fp16."""
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
    return Vh[:8], sv, sv


def build_k_entity(H57):
    """3166/3157 recipe: H (rows, slots, D); X = H[:, last, :]; center; SVD."""
    X = H57[:, -1, :]
    Xc = X - X.mean(0, keepdims=True)
    _, sv, Vh = np.linalg.svd(Xc, full_matrices=False)
    return Vh[:8], sv


def load_wu(mdir):
    from safetensors import safe_open
    want = 'lm_head.weight'
    for sh in sorted(os.listdir(mdir)):
        if sh.endswith('.safetensors'):
            with safe_open(os.path.join(mdir, sh), framework='pt') as f:
                if want in set(f.keys()):
                    return f.get_tensor(want).float().numpy().astype(np.float64)
    raise AssertionError('lm_head not found in ' + mdir)


def main():
    t0 = time.monotonic()
    dsha = freeze()
    for k, want in DESIGN['source_sha8'].items():
        p = {'p3152_14b': SRC_3152, 'p3157_14b': SRC_3157,
             'p3158_14b': SRC_3158}[k]
        got = s8(p)
        assert got == want, ('source drift', k, got)
    log('source sha8 anchored: 3 files OK')

    gates = {}

    # ---------------- Arm a: sealed-anchor rebuild ----------------
    z58 = np.load(SRC_3158)
    K_read_fp = z58['top64'].astype(np.float64).T          # (64, D)
    z57 = np.load(SRC_3157)
    K_ent_fp, sv_ent = build_k_entity(z57['H'].astype(np.float64))
    zm52 = np.load(SRC_3152)
    H52_fp = zm52['H']
    cfg = json.load(io.open(os.path.join(MDIR_FP, 'config.json'), encoding='utf-8'))
    NL, HID = int(cfg['num_hidden_layers']), int(cfg['hidden_size'])
    assert H52_fp.shape[2] == NL + 1 and H52_fp.shape[3] == HID
    R_fp, sv_r, _ = build_r_logic(H52_fp, NL)
    cens_fp = {}
    subs = {'R_logic': R_fp, 'K_entity': K_ent_fp, 'K_readout': K_read_fp}
    for a, bn in (('R_logic', 'K_entity'), ('R_logic', 'K_readout'),
                  ('K_readout', 'K_entity')):
        s, t1 = kross(subs[a], subs[bn])
        cens_fp['%s__%s' % (a, bn)] = (round(t1, 3), int((s ** 2 >= 0.5).sum()), s)
    gA = all(abs(cens_fp[k][0] - v[0]) <= TOL_ANCHOR and cens_fp[k][1] == v[1]
             for k, v in ANCHORS.items())
    gates['gA_anchors'] = gA
    log('Arm a census: ' + json.dumps({k: v[:2] for k, v in cens_fp.items()}))

    # shared-dim localization (cross-gram SVD)
    QR_fp, _ = np.linalg.qr(R_fp.T)
    QK_fp, _ = np.linalg.qr(K_ent_fp.T)
    U, s_c, Wt = np.linalg.svd(QR_fp.T @ QK_fp)
    shared_fp = [float(s_c[j]) for j in range(4)]
    log('cross-gram singulars: %s' % [round(v, 6) for v in shared_fp])

    # layer profile (per-layer rowspace alignment with shared dirs)
    prof = {}
    for k in range(1, NL + 1):        # skip slot0 embed (degenerate QR)
        Hkk = H52_fp[:, :, k, :].reshape(NT * NP_, HID).astype(np.float64)
        dsk = []
        ent_rows = {i: [] for i in range(NE)}
        for i in range(NE):
            for t in range(NT):
                for pi, (ei, ci) in enumerate(PAIRS):
                    if ei == i:
                        ent_rows[i].append(t * NP_ + pi)
        yv = np.array([ci == CLS_OF[ei] for (ei, ci) in PAIRS])
        yb = np.concatenate([yv] * NT)
        for i in range(NE):
            tr = [r for r in ent_rows[i] if yb[r]]
            fa = [r for r in ent_rows[i] if not yb[r]]
            dsk.append(unit(Hkk[tr].mean(0) - Hkk[fa].mean(0)))
        Dck = np.stack(dsk)
        Dck = Dck - Dck.mean(0, keepdims=True)
        Qk, _ = np.linalg.qr(Dck.T)
        Mj = U.T @ (QR_fp.T @ Qk)
        prof[k] = [float(np.linalg.norm(Mj[j])) for j in range(2)]
    for j in range(2):
        top = sorted(range(1, NL + 1), key=lambda k: -prof[k][j])[:4]
        log('layer profile dim%d top: %s' % (j, [(k, round(prof[k][j], 3)) for k in sorted(top)]))

    # vocab projection (W_U from nf4 dir fp16)
    W_nf4 = load_wu(MDIR_NF4)
    tok_lat = None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MDIR_FP, local_files_only=True,
                                        trust_remote_code=True, use_fast=True)
    vocab_proj = []
    for j in range(2):
        v = unit(QR_fp @ U[:, j])
        cw = W_nf4 @ v
        top = np.argsort(-np.abs(cw))[:8]
        vocab_proj.append([('%s:%.3f' % (tok.decode([int(t)]), cw[t])) for t in top])
        log('shared dim%d vocab top8: %s' % (j, vocab_proj[-1]))

    arm_a = dict(census={k: v[:2] for k, v in cens_fp.items()},
                 shared_cos=shared_fp,
                 coeff=[dict(R_logic=U[:, j].round(4).tolist(),
                             K_ent=Wt[j].round(4).tolist()) for j in range(2)],
                 layer_profile={str(k): [round(x, 4) for x in prof[k]]
                                for k in sorted(prof)},
                 vocab_top8=vocab_proj)

    if SMOKE:
        # pipeline check: load nf4, 2 rows each panel, kross math on partial
        import torch
        from transformers import AutoModelForCausalLM
        torch.manual_seed(0)
        log('SMOKE nf4 load begin')
        model = AutoModelForCausalLM.from_pretrained(
            MDIR_NF4, dtype=torch.float16, trust_remote_code=True,
            device_map='cuda').eval()
        assert model.config.num_hidden_layers == NL
        tok_nf4 = AutoTokenizer.from_pretrained(MDIR_NF4, local_files_only=True,
                                                trust_remote_code=True, use_fast=True)
        with torch.no_grad():
            ids = tok_nf4(ENTS[0] + '是一种水果。', add_special_tokens=False,
                          return_tensors='pt')['input_ids'].to('cuda')
            o = model(input_ids=ids, output_hidden_states=True)
            h_last = o.hidden_states[-1][0, -1].float().cpu().numpy()
            del o
        assert h_last.shape == (HID,) and np.isfinite(h_last).all()
        log('SMOKE nf4 forward OK, h_last norm=%.3f' % float(np.linalg.norm(h_last)))
        # small-pipeline: 8 rows of p3152
        with torch.no_grad():
            for pi in range(4):
                for t in range(2):
                    pr = TPL3152[t].format(e=ENTS[pi], c=CLASSES[CLS_OF[pi]])
                    ii = tok_nf4(pr, add_special_tokens=False,
                                 return_tensors='pt')['input_ids'].to('cuda')
                    o = model(input_ids=ii, output_hidden_states=True)
                    _ = o.hidden_states[-1][0, -1].float().cpu().numpy()
                    del o
        # p3157 rows
        with torch.no_grad():
            for r in ROWS3157[:4]:
                ids = (tok_nf4(PREFIX_ZH, add_special_tokens=False)
                       ['input_ids'] if r['ctx'] == 1 else []) + \
                      tok_nf4(r['prompt'], add_special_tokens=False)['input_ids']
                ii = torch.tensor([ids], device='cuda')
                o = model(input_ids=ii, output_hidden_states=True)
                _ = o.hidden_states[-1][0, -1].float().cpu().numpy()
                del o
        torch.cuda.empty_cache()
        del model
        log('SMOKE pipeline OK')
        res = dict(phase=PHASE, name=NAME, smoke=True, design_sha8=dsha,
                   arm_a=arm_a, gates={'gA_anchors': gates['gA_anchors']},
                   verdict='smoke_ok' if gates['gA_anchors'] else 'smoke_anchor_fail')
        with io.open(os.path.join(SMOKE_DIR, 'result_smoke.json'), 'w',
                     encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True))
        with io.open(os.path.join(SMOKE_DIR, 'run_log.txt'), 'w',
                     encoding='utf-8') as f:
            f.write('\n'.join(L) + '\n')
        print('SMOKE DONE', res['verdict'])
        return

    # ---------------- Arm b: nf4 GPU re-collection ----------------
    import torch
    from transformers import AutoModelForCausalLM
    torch.manual_seed(0)
    NH = NL + 1
    # b2: p3152 panel (738 rows) — cache-hit on rerun (3152/3157 precedent)
    cache_nf4 = os.path.join(BASE, 'nf4_collect.npz')
    if os.path.exists(cache_nf4):
        zc = np.load(cache_nf4)
        H52_nf4, H57_nf4 = zc['H3152'], zc['H3157']
        assert H52_nf4.shape == (NT, NP_, NH, HID) and H57_nf4.shape == (128, NH, HID)
        log('nf4 collect cache hit %s' % os.path.basename(cache_nf4))
    else:
        log('nf4 load begin (dtype=bfloat16, aligned with sealed 3152/3157 '
            'collection dtype and bnb_4bit_compute_dtype)')
        model = AutoModelForCausalLM.from_pretrained(
            MDIR_NF4, dtype=torch.bfloat16, trust_remote_code=True,
            device_map='cuda').eval()
        assert model.config.num_hidden_layers == NL
        tok_nf4 = AutoTokenizer.from_pretrained(MDIR_NF4, local_files_only=True,
                                                trust_remote_code=True, use_fast=True)
        H52_nf4 = np.zeros((NT, NP_, NH, HID), np.float16)
        with torch.no_grad():
            for t in range(NT):
                for pi, (ei, ci) in enumerate(PAIRS):
                    pr = TPL3152[t].format(e=ENTS[ei], c=CLASSES[ci])
                    ii = tok_nf4(pr, add_special_tokens=False,
                                 return_tensors='pt')['input_ids'].to('cuda')
                    o = model(input_ids=ii, output_hidden_states=True)
                    H52_nf4[t, pi] = np.stack(
                        [h[0, -1].float().detach().cpu().numpy()
                         for h in o.hidden_states], 0).astype(np.float16)
                    del o
                log('b2 collect tpl%d done' % t)
        # b3: p3157 panel (128 rows)
        pre_ids = tok_nf4(PREFIX_ZH, add_special_tokens=False)['input_ids']
        ctx_ids = (pre_ids * (K_CTX // len(pre_ids) + 1))[:K_CTX]
        assert len(ctx_ids) == K_CTX
        H57_nf4 = np.zeros((128, NH, HID), np.float16)
        with torch.no_grad():
            for i, r in enumerate(ROWS3157):
                ids = (list(ctx_ids) if r['ctx'] == 1 else []) + \
                      tok_nf4(r['prompt'], add_special_tokens=False)['input_ids']
                ii = torch.tensor([ids], device='cuda')
                o = model(input_ids=ii, output_hidden_states=True)
                H57_nf4[i] = np.stack(
                    [h[0, -1].float().detach().cpu().numpy()
                     for h in o.hidden_states], 0).astype(np.float16)
                del o
                if (i + 1) % 32 == 0:
                    log('b3 collect %d/128' % (i + 1))
        torch.cuda.empty_cache()
        del model
        np.savez_compressed(cache_nf4, H3152=H52_nf4, H3157=H57_nf4)
        log('nf4 collects saved (%.1f MB)' % (os.path.getsize(cache_nf4) / 1e6))

    # b4: K_readout_nf4 (3158 recipe: W_U Gram top64) — D-space Gram via GPU
    # fp32 (V x V Gram needs 172 GiB fp64 host RAM; 3158 uses chunked GPU)
    Wt_t = torch.tensor(W_nf4, dtype=torch.float32, device='cuda')
    with torch.no_grad():
        G = Wt_t.T @ Wt_t                              # (D, D) fp32 GPU
        evals, evecs = torch.linalg.eigh(G)            # ascending
        top64 = evecs.flip(1)[:, :64].T.contiguous().cpu().numpy()
    del Wt_t, G
    torch.cuda.empty_cache()
    K_read_nf4 = top64.astype(np.float64)              # (64, D) rows

    # rebuild under nf4
    R_nf4, sv_r4, _ = build_r_logic(H52_nf4, NL)
    K_ent_nf4, sv_e4 = build_k_entity(H57_nf4.astype(np.float64))

    # fidelity G-B: K_readout_nf4 vs fp top64
    sB, tB = kross(K_read_fp, K_read_nf4)
    gB = bool(abs(tB - 0.0) <= TOL_FID)   # rowspace angle ~0 means faithful
    # stricter: per-anchor reproduces if we re-run census with nf4 readout?
    # G-B per design: top1 angle diff vs fp anchor 70.235 using fp K_ent
    s_cmp, t_kent_nf4read = kross(K_ent_fp, K_read_nf4)
    gB2 = bool(abs(t_kent_nf4read - ANCHORS['K_readout__K_entity'][0]) <= TOL_FID)
    gates['gB_fidelity'] = gB and gB2
    log('G-B: rowspace angle=%.4f deg | K_ent_fp x K_read_nf4=%.3f (anchor 70.235)'
        % (tB, t_kent_nf4read))

    # conservation G-C
    cens_nf4 = {}
    subs4 = {'R_logic': R_nf4, 'K_entity': K_ent_nf4}
    for a, bn in (('R_logic', 'K_entity'),):
        s, t1 = kross(subs4[a], subs4[bn])
        cens_nf4['%s__%s' % (a, bn)] = (round(t1, 3), int((s ** 2 >= 0.5).sum()), s)
    mixed = {}
    for tag, (A, Bd) in dict(Rnf4__Kent_fp=(R_nf4, K_ent_fp),
                             Rfp__Kent_nf4=(R_fp, K_ent_nf4)).items():
        s, t1 = kross(A, Bd)
        mixed[tag] = (round(t1, 3), int((s ** 2 >= 0.5).sum()))
    t1_nf4 = cens_nf4['R_logic__K_entity'][0]
    eff_nf4 = cens_nf4['R_logic__K_entity'][1]
    # shared-dim cos diff
    QR4, _ = np.linalg.qr(R_nf4.T)
    QK4, _ = np.linalg.qr(K_ent_nf4.T)
    _, s4, _ = np.linalg.svd(QR4.T @ QK4)
    cos_diff = [abs(float(s4[j]) - shared_fp[j]) for j in range(2)]
    gC = bool(eff_nf4 == 2 and abs(t1_nf4 - ANCHORS['R_logic__K_entity'][0]) <= TOL_CONS
              and max(cos_diff) <= TOL_COS)
    gates['gC_conservation'] = gC
    log('G-C: R_nf4 x K_nf4 = %.3f deg eff=%d | shared cos nf4=%s fp=%s diff=%s'
        % (t1_nf4, eff_nf4, [round(float(v), 4) for v in s4[:4]],
           [round(v, 4) for v in shared_fp[:2]], [round(v, 4) for v in cos_diff]))
    log('mixed pairs: ' + json.dumps(mixed))

    gate_all = all(gates.values())
    verdict = 'appx1_resolved' if gate_all else 'appx1_blocked'
    elapsed = round(time.monotonic() - t0, 1)
    res = dict(
        phase=PHASE, name=NAME, design_sha8=dsha, smoke=False,
        arm_a=arm_a, arm_b=dict(
            fidelity=dict(top64_rowspace_deg=round(tB, 4),
                          kent_fp_x_kread_nf4_deg=round(t_kent_nf4read, 3),
                          gB=gates['gB_fidelity']),
            conservation=dict(top1_deg=t1_nf4, eff_ge05=eff_nf4,
                              shared_cos_nf4=[round(float(v), 6) for v in s4[:4]],
                              shared_cos_fp=[round(v, 6) for v in shared_fp[:2]],
                              cos_diff=[round(v, 4) for v in cos_diff],
                              mixed_pairs=mixed,
                              gC=gates['gC_conservation']),
            k_ent_nf4_singulars=[round(float(v), 3) for v in sv_e4[:8]],
            r_nf4_singulars=[round(float(v), 4) for v in sv_r4[:8]]),
        gates=gates, gate_all=gate_all, verdict=verdict,
        elapsed_s=elapsed, source_sha8=DESIGN['source_sha8'])
    res8 = hashlib.sha256(json.dumps(
        {k: v for k, v in res.items() if k != 'res_sha8'},
        sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
    res['res_sha8'] = res8
    with io.open(os.path.join(BASE, 'result.json'), 'w',
                 encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True))
    with io.open(os.path.join(BASE, 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    log('VERDICT %s res=%s elapsed=%.1fs' % (verdict, res8, elapsed))
    print('FORMAL DONE', verdict)


if __name__ == '__main__':
    main()
