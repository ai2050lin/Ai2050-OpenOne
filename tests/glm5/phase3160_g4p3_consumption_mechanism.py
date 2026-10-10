# -*- coding: utf-8 -*-
# Phase 3160 (G4-P3): 读出方向消耗的机制判别 —— MLP 压缩 vs attention 再分配
# 预注册: AGI_GPT5_MEMO Phase 3160 (3159 closeout 冻结, 观测前):
#   假设: 3159 证 top-sigma 注入在剩余第 1-2 块被旋转消耗(share_top 0.999->0.27->0.11
#         口径为逐层曲线; 与 massive 无关); P3 问消耗载体: MLP 压缩 vs attention 再分配。
#   设计: (1) zero 模式(零 GPU): 3159 SHARE top 臂逐层 q10/q50/q90 分位曲线 + 消耗层位谱
#         (每层降幅 = q50(L-1)-q50(L), big-drop 层) + 跨模型消耗曲线指纹(截 min NH,
#         Pearson >= 0.8);
#         (2) model 模式(GPU): 4 锚(3159 anchor_idx 16 取 linspace 4) x top64 方向
#             linspace(0,63,6)(= batch 行, 与 3159 batch=6 kernel 路径一致) x alpha=0.1
#             相对 ||h_mid||; 4 消融配置 ABLS = none / mlp_mid / mlp_mid1 / mlp_mid1_2
#             (hook 置零相应块 mlp 输出, 残差流恒等替换); 每配置 base+注入各一前向,
#             dh = pert - base(同配置同 kernel 路径); share_top 逐层;
#             recover = (share_abl(NL) - share_none(NL)) / (share_none(L_mid) - share_none(NL));
#             判决(主配置 = mlp_mid1, 预注册字面; mlp_mid1_2 组合复核; mlp_mid 第一段辅助):
#             recover >= 0.5 -> mlp_compression_primary; < 0.1 -> attention_reallocation_primary;
#             之间 -> mixed; 两配置类别不一致 -> mixed_inconsistent;
#         (3) delta_h 去向: none 配置 dh(槽 L_mid+2) 在锚态 massive 维度 e_d1 上的份额
#             (d1 = argmax_d mean_anchor |H3157[., L_mid, d]|; 设计偏差记录: 预注册的
#             3156 rank-1 轴不可从 3156 npz 复现 —— en last-token L7 逐位恒等, 逐行位移
#             SVD 最高 share 0.685 远低于当时报告 0.9999, 判定当时轴为现场计算未落盘;
#             改用 3157 锚态 massive 维度(4b d1=0, 与 3156 塌缩维度一致), 三模型统一可用)
#             + top-8 massive 维度集份额 + cos(dh(L_mid+2), h(L_mid));
#         (4) summary 模式: 三模型判决汇总 + none 配置 q50 消耗曲线指纹(Pearson >= 0.8)。
#   装置锚: 锚态 batch=1 bitwise vs 3157; 每 dh 在注入槽 L_mid 份额 = 1(top 臂);
#           槽 < L_mid dh = 0; none 配置与 3159 SHARE top 臂逐层均值一致(容差 bf16 噪声
#           0.03, 方向子集不同仅统计抖动)。
# 教训内置: SMOKE 目录分离; design 全 str 键 JSON; fail-fast 断言; 数字一律 result 现场渲染;
#           mlp hook 兼容 tuple/Tensor; 消融块号 <= NL-1 断言; 无百分号字面陷阱。
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
MODEL = os.environ.get('P3160_MODEL') or (sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b')
SMOKE = os.environ.get('P3160_SMOKE') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g4p3_consumption_mechanism'
BASE = os.path.join(RDIR, 'phase3160', NAME, MODEL)
if SMOKE:
    BASE = os.path.join(BASE, 'smoke')
os.makedirs(BASE, exist_ok=True)
LOGP = os.path.join(BASE, 'run_log.txt')

def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    with open(LOGP, 'a', encoding='utf-8') as f:
        f.write(ln + '\n')
    try:
        print(ln, flush=True)
    except Exception:
        pass

def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]

def freeze_design(phase_name, design):
    eblob = json.dumps(design, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    sha = hashlib.sha256(eblob).hexdigest()
    exe_p = os.path.join(BASE, 'execution.json')
    if os.path.exists(exe_p):
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha'] == sha, 'DESIGN DRIFT: delete execution.json+result.json after script change'
        log('execution.json match (sha %s)' % sha[:8])
    else:
        json.dump({'phase': 3160, 'name': phase_name, 'design_sha': sha,
                   'design': design, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('execution.json FROZEN (sha %s)' % sha[:8])
    return sha

def seal_result(result, out_name):
    blob = json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    res_sha8 = sha8(blob)
    result['res_sha8'] = res_sha8
    result['verdict'] = result['verdict'] + '|sha8_' + res_sha8
    rp = os.path.join(BASE, out_name)
    json.dump(result, open(rp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    seal = sha8(open(rp, 'rb').read())
    result['seal_sha8'] = seal
    json.dump(result, open(rp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log('RESULT %s res_sha8=%s seal=%s verdict=%s' % (out_name, res_sha8, seal, result['verdict']))
    return res_sha8, seal

MODELS3 = ('qwen3-4b', 'qwen3-14b', 'glm4')
FP_GATE = 0.8
RECOVER_MLP = 0.5
RECOVER_ATTN = 0.1

# ================= zero 模式(零 GPU): 3159 SHARE 分位曲线 + 层位谱 + 指纹 =================
if MODEL == 'zero':
    L_MIDS, QCURVE, DROPS, TOPS = {}, {}, {}, {}
    for m in MODELS3:
        z9 = np.load(os.path.join(RDIR, 'phase3159', 'g4p2_equivalence_dynamics', m, 'collect.npz'))
        SHARE = z9['SHARE'].astype(np.float64)          # (3, NA, ND, NALPHA, NH)
        L_MID = int(z9['l_mid'])
        L_MIDS[m] = L_MID
        top = SHARE[2].reshape(-1, SHARE.shape[4])       # (NA*ND*NALPHA, NH)
        q = np.percentile(top, [10, 50, 90], axis=0)     # (3, NH)
        QCURVE[m] = q
        drop = q[1, :-1] - q[1, 1:]                      # 每层中位降幅 (NH-1,)
        DROPS[m] = drop
        big = int(np.argmax(drop[L_MID:]) + L_MID)       # 注入后最大降幅层(槽号)
        TOPS[m] = dict(l_mid=L_MID, big_drop_slot=big,
                       q50_lm=float(q[1, L_MID]), q50_big=float(q[1, big]),
                       q50_nl=float(q[1, -1]),
                       drop_big=float(drop[big]),
                       gain_top=float(q[1, -1] - q[1, L_MID]))
        log('%s: L_mid=%d big_drop=%d q50 %.4f->%.4f->%.4f drop=%.4f gain=%.4f' % (
            m, L_MID, big, q[1, L_MID], q[1, big], q[1, -1], drop[big], TOPS[m]['gain_top']))
    NH = min(QCURVE[m].shape[1] for m in MODELS3)
    W = min(QCURVE[m].shape[1] - L_MIDS[m] for m in MODELS3)   # 消耗段窗口(相对 L_mid 对齐)
    pairs, pairs_raw = {}, {}
    ms = list(MODELS3)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = ms[i], ms[j]
            ca = QCURVE[a][1][L_MIDS[a]:L_MIDS[a] + W]          # 消耗段(主口径)
            cb = QCURVE[b][1][L_MIDS[b]:L_MIDS[b] + W]
            pairs['%s_vs_%s' % (a, b)] = float(np.corrcoef(ca, cb)[0, 1])
            ra = QCURVE[a][1][:NH]                              # 直接槽号对齐(对照)
            rb = QCURVE[b][1][:NH]
            pairs_raw['%s_vs_%s' % (a, b)] = float(np.corrcoef(ra, rb)[0, 1])
    fpmin = float(min(pairs.values()))
    fpmin_raw = float(min(pairs_raw.values()))
    fp_ok = fpmin >= FP_GATE
    verdict = ('g4p3_zero_fingerprint_ok' if fp_ok else 'g4p3_zero_fingerprint_fail') + \
              '|fpmin_%.4f' % fpmin
    npz_p = os.path.join(BASE, 'collect_zero.npz')
    # 变长曲线逐数组存
    arrs = dict(l_mids=np.array([L_MIDS[m] for m in ms], np.int64),
                fp_pairs=np.array([pairs['%s_vs_%s' % (ms[i], ms[j])]
                                   for i in range(3) for j in range(i + 1, 3)], np.float64),
                fp_pairs_raw=np.array([pairs_raw['%s_vs_%s' % (ms[i], ms[j])]
                                       for i in range(3) for j in range(i + 1, 3)], np.float64),
                nh_min=np.int64(NH), window=np.int64(W))
    for mi, m in enumerate(ms):
        arrs['q10_%d' % mi] = QCURVE[m][0]
        arrs['q50_%d' % mi] = QCURVE[m][1]
        arrs['q90_%d' % mi] = QCURVE[m][2]
        arrs['drop_%d' % mi] = DROPS[m]
    np.savez_compressed(npz_p, **arrs)
    npz_sha = sha8(open(npz_p, 'rb').read())
    result = {
        'phase': 3160, 'name': NAME, 'mode': 'zero', 'smoke': bool(SMOKE),
        'models': list(ms), 'l_mids': L_MIDS, 'layer_stats': TOPS,
        'big_drop_slots': {m: TOPS[m]['big_drop_slot'] for m in ms},
        'fp_pairs_q50': pairs, 'fp_pairs_q50_raw_slotalign': pairs_raw,
        'fpmin_q50': fpmin, 'fpmin_q50_raw': fpmin_raw,
        'fp_gate': FP_GATE, 'consumption_window': W, 'align_note':
        'main fingerprint aligned by L_mid offset (consumption window W=%d); raw slot '
        'alignment recorded as control (L_mid differs 18 vs 20, misalignment dilutes)' % W,
        'npz_sha8': npz_sha,
        'verdict': verdict,
        'runtime_s': round(time.time() - T0, 1),
    }
    seal_result(result, 'result_zero.json')
    log('ZERO DONE')
    sys.exit(0)

# ================= model / summary 模式 =================
design = {
    'phase': '3160', 'name': NAME, 'model': MODEL, 'smoke': bool(SMOKE),
    'prereg': 'AGI_GPT5_MEMO Phase 3160 (frozen at 3159 closeout, before any observation)',
    'anchors': '3159 anchor_idx (16) -> linspace 4 (smoke 2); anchor state batch=1 bitwise vs 3157 H slot L_mid',
    'dirs': '3158 collect.npz top64, pick=linspace(0,63,6); 6 dirs = batch rows '
            '(same batch=6 kernel path as 3159); alpha=0.1 rel ||h_mid||',
    'ablations': 'none / mlp_mid (block L_mid) / mlp_mid1 (block L_mid+1, prereg literal) / '
                 'mlp_mid1_2 (blocks L_mid+1 and L_mid+2); hook zeroes mlp output '
                 '(residual-stream identity replacement); base and injection per config '
                 'share the same ablation state and batch=6 kernel path',
    'recover_def': '(share_abl(NL) - share_none(NL)) / (share_none(L_mid) - share_none(NL)), '
                   'share = mean over anchors x dirs',
    'verdict_gate': 'main config = mlp_mid1 (prereg literal): recover >= 0.5 -> '
                    'mlp_compression_primary; < 0.1 -> attention_reallocation_primary; '
                    'else mixed; mlp_mid1_2 as consistency check; mismatch -> mixed_inconsistent',
    'destination_def': 'none-config dh at slot L_mid+2: share on massive dim e_d1 '
                       '(d1 = argmax_d mean_anchor |H3157[., L_mid, d]|), share on top-8 '
                       'massive dims set, cos(dh(L_mid+2), h(L_mid)); DESIGN DEVIATION: '
                       'prereg 3156 rank-1 axis not reproducible from 3156 npz (en last-token '
                       'L7 bitwise identical across k; best row-wise SVD share 0.685 vs '
                       'reported 0.9999 -> axis was computed live, not persisted); replaced by '
                       '3157 anchor massive dim (4b d1=0, same as 3156 collapse dim), '
                       'uniform across models; decided before any 3160 observation',
    'fp_gate': FP_GATE, 'recover_mlp': RECOVER_MLP, 'recover_attn': RECOVER_ATTN,
    'n_anchors': 4, 'n_dirs': 6, 'alpha': 0.1,
    'summary_gates': 'none-config q50 consumption curve fingerprint 3 pairs >= 0.8 AND '
                     'mechanism class agreement across models (descriptive)',
}
DESIGN_SHA = freeze_design(NAME, design)

if MODEL == 'summary':
    log('summary mode')
    res, dest, curves, lmids = {}, {}, {}, {}
    for m in MODELS3:
        rp = os.path.join(RDIR, 'phase3160', NAME, m, 'result.json')
        r = json.load(open(rp, encoding='utf-8'))
        res[m] = r
        dest[m] = r['destination']
        z = np.load(os.path.join(RDIR, 'phase3160', NAME, m, 'collect.npz'))
        curves[m] = z['q50_none'].astype(np.float64)
        lmids[m] = int(z['l_mid'])
        log('%s: verdict=%s recover_mid1=%.4f recover_mid1_2=%.4f' % (
            m, r['verdict'], r['recover']['mlp_mid1'], r['recover']['mlp_mid1_2']))
    W = min(curves[m].shape[0] - lmids[m] for m in curves)
    pairs, pairs_raw = {}, {}
    ms = list(MODELS3)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = ms[i], ms[j]
            ca = curves[a][lmids[a]:lmids[a] + W]            # 消耗段对齐(主口径, 同 zero)
            cb = curves[b][lmids[b]:lmids[b] + W]
            pairs['%s_vs_%s' % (a, b)] = float(np.corrcoef(ca, cb)[0, 1])
            ra = curves[a]
            rb = curves[b]
            nmin = min(len(ra), len(rb))
            pairs_raw['%s_vs_%s' % (a, b)] = float(np.corrcoef(ra[:nmin], rb[:nmin])[0, 1])
    fpmin = float(min(pairs.values()))
    fp_ok = fpmin >= FP_GATE
    cls = [res[m]['mech_class'] for m in ms]
    agree = len(set(cls)) == 1
    if agree and cls[0] == 'mlp_compression_primary' and fp_ok:
        verdict = 'g4p3_mlp_compression_primary|fp_ok'
    elif agree and cls[0] == 'attention_reallocation_primary' and fp_ok:
        verdict = 'g4p3_attention_reallocation_primary|fp_ok'
    elif agree and fp_ok:
        verdict = 'g4p3_mixed|fp_ok'
    else:
        verdict = 'g4p3_divergent|fpmin_%.4f|classes_%s' % (fpmin, '/'.join(cls))
    # 去向跨模型(描述)
    e1_sh = [dest[m]['share_e1_lm2'] for m in ms]
    t8_sh = [dest[m]['share_top8_lm2'] for m in ms]
    result = {
        'phase': 3160, 'name': NAME, 'mode': 'summary', 'smoke': bool(SMOKE),
        'models': ms, 'mech_classes': {m: res[m]['mech_class'] for m in ms},
        'recover': {m: res[m]['recover'] for m in ms},
        'share_nl': {m: res[m]['share_nl_none'] for m in ms},
        'destination': dest, 'e1_shares': e1_sh, 'top8_shares': t8_sh,
        'fp_pairs_q50_none': pairs, 'fp_pairs_raw_slotalign': pairs_raw,
        'fpmin_q50_none': fpmin, 'fp_gate': FP_GATE, 'consumption_window': W,
        'gates': {'fingerprint': bool(fp_ok), 'class_agreement': bool(agree)},
        'verdict': verdict, 'design_sha': DESIGN_SHA,
        'runtime_s': round(time.time() - T0, 1),
    }
    seal_result(result, 'result_summary.json')
    log('SUMMARY DONE')
    sys.exit(0)

# ---- model 模式 ----
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MDIR_MAP = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'Qwen3-14B', 'glm4': 'glm4-9b-chat-hf'}
MDIR = os.path.join(ROOT, 'models', 'hf', MDIR_MAP[MODEL])
cfgm = json.load(open(os.path.join(MDIR, 'config.json'), encoding='utf-8'))
V, D = int(cfgm['vocab_size']), int(cfgm['hidden_size'])
NL = int(cfgm['num_hidden_layers'])
L_MID = int(round(0.5 * NL))
ALPHA = 0.1
N_ANCH = 2 if SMOKE else 4
N_DIR = 6
ABLS = ['none', 'mlp_mid', 'mlp_mid1', 'mlp_mid1_2']
assert L_MID + 2 <= NL - 1, ('ablation blocks exceed range', L_MID, NL)
log('model=%s V=%d D=%d NL=%d L_mid=%d smoke=%s' % (MODEL, V, D, NL, L_MID, SMOKE))

R7 = os.path.join(RDIR, 'phase3157', 'g2p2_transform_algebra_commutator')
R8 = os.path.join(RDIR, 'phase3158', 'g4p1_output_equivalence_class')
z8 = np.load(os.path.join(R8, MODEL, 'collect.npz'))
top64 = z8['top64'].astype(np.float64)
anchor_idx = z8['anchor_idx'].astype(int)
pick = np.linspace(0, 63, N_DIR).round().astype(int)
assert len(set(pick)) == N_DIR
U = top64[:, pick]
nn = np.linalg.norm(U, axis=0)
assert np.abs(nn - 1.0).max() < 1e-5, ('unit norm', nn)
full_idx = np.linspace(0, 15, 4).round().astype(int)
if SMOKE:
    full_idx = np.array([0, 15])
idx = anchor_idx[full_idx]
assert len(set(idx.tolist())) == len(idx)
log('anchors from 3159 anchor_idx: %s (full_idx %s)' % (idx.tolist(), full_idx.tolist()))

# 3157 行重建(零抄写)
exe7 = json.load(open(os.path.join(R7, MODEL, 'execution.json'), encoding='utf-8'))
d7 = exe7['design']
z7 = np.load(os.path.join(R7, MODEL, 'collect.npz'))
H7 = z7['H'].astype(np.float64)
assert H7.shape[0] == int(d7['n_rows']) and H7.shape[1] == NL + 1 and H7.shape[2] == D
TPL7 = {tuple(kk.split('|')): v for kk, v in d7['tpl'].items()}
ENTS7 = [tuple(e) for e in d7['ents']]
RELS7 = list(d7['rels']); POLS7 = list(d7['pols']); CTXS7 = list(d7['ctx'])
rows7 = []
for ei in range(len(ENTS7)):
    for rel in RELS7:
        for pol in POLS7:
            for cx in CTXS7:
                e, c, p = ENTS7[ei]
                rows7.append(dict(ei=ei, rel=rel, pol=pol, ctx=cx,
                                  prompt=TPL7[(rel, pol)].replace('{E}', e).replace('{C}', c).replace('{P}', p)))
assert np.array_equal(np.array([r['ei'] for r in rows7], np.int16), z7['ei'])
idx7 = np.unique(np.linspace(0, len(rows7) - 1, 16).round().astype(int))
assert np.array_equal(idx7, anchor_idx), 'anchor_idx mismatch vs 3158'

# massive 维度(设计冻结: 3157 锚槽 L_mid mean|h| argmax)
am = np.abs(H7[idx7, L_MID, :]).mean(0)
d1 = int(am.argmax())
top8_dims = np.argsort(-am)[:8]
log('massive dim d1=%d (mean|h|=%.1f, 2nd=%d %.1f); top8=%s' % (
    d1, am[d1], int(np.argsort(-am)[1]), np.sort(am)[-2], top8_dims.tolist()))

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
log('model load begin')
model = AutoModelForCausalLM.from_pretrained(
    MDIR, dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()
log('model loaded: %s' % type(model).__name__)
assert model.config.num_hidden_layers == NL
dev = 'cuda'
NH = NL + 1
pre_ids = tok(d7['prefix'], add_special_tokens=False)['input_ids']
ctx_ids = (pre_ids * (int(d7['k_ctx']) // len(pre_ids) + 1))[:int(d7['k_ctx'])]

# --- hooks ---
INJ = {'on': False, 'delta': None}
ABL = {'blocks': set()}

def _inj_hook(module, args, output):
    if INJ['on'] and INJ['delta'] is not None:
        out0 = output[0] if isinstance(output, tuple) else output
        new0 = out0.clone()
        new0[:, -1, :] = new0[:, -1, :] + INJ['delta'].to(new0.dtype)
        if isinstance(output, tuple):
            return (new0,) + tuple(output[1:])
        return new0
    return None

def _make_mlp_hook(block_no):
    def _mlp_hook(module, args, output):
        if block_no in ABL['blocks']:
            out0 = output[0] if isinstance(output, tuple) else output
            z = torch.zeros_like(out0)
            if isinstance(output, tuple):
                return (z,) + tuple(output[1:])
            return z
        return None
    return _mlp_hook

model.model.layers[L_MID - 1].register_forward_hook(_inj_hook)
for b in (L_MID, L_MID + 1, L_MID + 2):
    model.model.layers[b].mlp.register_forward_hook(_make_mlp_hook(b))

def last_token_states(o):
    return np.stack([h[:, -1].float().detach().cpu().numpy() for h in o.hidden_states], 1)

BATCH_REP = 6

def fwd(ids, delta_rows):
    ii = torch.tensor([ids] * BATCH_REP, dtype=torch.int64, device=dev)
    INJ['delta'] = delta_rows
    INJ['on'] = True
    with torch.no_grad():
        o = model(input_ids=ii, output_hidden_states=True)
    INJ['on'] = False
    INJ['delta'] = None
    hs = last_token_states(o)
    lg = o.logits[:, -1, :].float().detach().cpu().numpy().astype(np.float64)
    del o
    return lg, hs

# --- 逐锚 ---
SHARE = np.zeros((len(ABLS), N_ANCH, N_DIR, NH), np.float64)
DEST = np.zeros((N_ANCH, N_DIR, 3), np.float64)   # share_e1 / share_top8 / cos_h at slot L_mid+2
anchor_meta = []
base_rel_max = 0.0
for ai0, r0 in enumerate(idx):
    row = rows7[int(r0)]
    ids = (ctx_ids if row['ctx'] == 1 else []) + tok(row['prompt'], add_special_tokens=False)['input_ids']
    # batch=1 锚态 bitwise vs 3157
    ii1 = torch.tensor([ids], dtype=torch.int64, device=dev)
    with torch.no_grad():
        o1 = model(input_ids=ii1, output_hidden_states=True)
    hs1 = np.stack([h[0, -1].float().detach().cpu().numpy() for h in o1.hidden_states], 0)
    del o1
    assert np.array_equal(hs1[L_MID].astype(np.float16), H7[int(r0), L_MID].astype(np.float32).astype(np.float16)), \
        ('anchor bitwise vs 3157', int(r0))
    h_mid_b6 = None
    for cfg_i, cfg in enumerate(ABLS):
        ABL['blocks'] = set() if cfg == 'none' else set(
            [L_MID] if cfg == 'mlp_mid' else ([L_MID + 1] if cfg == 'mlp_mid1' else [L_MID + 1, L_MID + 2]))
        # base(无注入)
        _, hs_base = fwd(ids, None)
        if cfg == 'none':
            h_mid_b6 = hs_base[:, L_MID, :]
        pnorm_rows = np.linalg.norm(hs_base[:, L_MID, :].astype(np.float64), axis=1)
        eps = ALPHA * pnorm_rows
        delta_rows = torch.from_numpy((eps[:, None] * U.T).astype(np.float32)).to(dev)
        _, hs_pert = fwd(ids, delta_rows)
        dh = hs_pert - hs_base
        pre_rel = float(np.abs(dh[:, :L_MID, :]).max() / (pnorm_rows.max() + 1e-18))
        base_rel_max = max(base_rel_max, pre_rel)
        assert pre_rel < 1e-5, ('pre-slot dh too large', cfg, ai0, pre_rel)
        num = (dh[:, L_MID:, :] @ top64.astype(np.float64)) ** 2
        den = (dh[:, L_MID:, :] ** 2).sum(2) + 1e-18
        SHARE[cfg_i, ai0, :, L_MID:] = (num.sum(2) / den)
        if cfg == 'none':
            dh2 = dh[:, L_MID + 2, :]                     # (6, D)
            dnorm = (dh2 ** 2).sum(1) + 1e-18
            DEST[ai0, :, 0] = dh2[:, d1] ** 2 / dnorm
            DEST[ai0, :, 1] = (dh2[:, top8_dims] ** 2).sum(1) / dnorm
            hmid = hs_base[:, L_MID, :].astype(np.float64)
            hmid = hmid / (np.linalg.norm(hmid, axis=1, keepdims=True) + 1e-18)
            dh2n = dh2 / np.sqrt(dnorm)[:, None]
            DEST[ai0, :, 2] = (dh2n * hmid).sum(1)
        del hs_base, hs_pert, dh, num, den
    ABL['blocks'] = set()
    anchor_meta.append(dict(row=int(r0), ei=int(row['ei']), rel=row['rel'], pol=row['pol'],
                            ctx=int(row['ctx'])))
    sh_none_lm = float(SHARE[0, ai0, :, L_MID].mean())
    assert sh_none_lm > 0.95, ('none share at L_mid too low', sh_none_lm)
    log('anchor %d/%d row=%d done; none share(L_mid)=%.4f share(NL)=%.4f' % (
        ai0 + 1, N_ANCH, int(r0), sh_none_lm, float(SHARE[0, ai0, :, NL].mean())))

# --- 汇总 + 判决 ---
def q50(x):
    return float(np.median(x))

share_nl = {cfg: float(SHARE[ci, :, :, NL].mean()) for ci, cfg in enumerate(ABLS)}
share_lm = {cfg: float(SHARE[ci, :, :, L_MID].mean()) for ci, cfg in enumerate(ABLS)}
recover = {}
for cfg in ('mlp_mid', 'mlp_mid1', 'mlp_mid1_2'):
    ci = ABLS.index(cfg)
    rec = (share_nl[cfg] - share_nl['none']) / (share_lm['none'] - share_nl['none'] + 1e-18)
    recover[cfg] = float(rec)
c1 = recover['mlp_mid1']
c2 = recover['mlp_mid1_2']
if c1 >= RECOVER_MLP:
    cls1 = 'mlp_compression_primary'
elif c1 < RECOVER_ATTN:
    cls1 = 'attention_reallocation_primary'
else:
    cls1 = 'mixed'
if c2 >= RECOVER_MLP:
    cls2 = 'mlp_compression_primary'
elif c2 < RECOVER_ATTN:
    cls2 = 'attention_reallocation_primary'
else:
    cls2 = 'mixed'
mech_class = cls1 if cls1 == cls2 else 'mixed_inconsistent'
log('share_nl: %s; recover: %s -> %s' % (
    {k: round(v, 4) for k, v in share_nl.items()},
    {k: round(v, 4) for k, v in recover.items()}, mech_class))

dest_mean = dict(share_e1_lm2=float(DEST[:, :, 0].mean()),
                 share_top8_lm2=float(DEST[:, :, 1].mean()),
                 cos_h_lm2=float(DEST[:, :, 2].mean()))
log('destination(L_mid+2): %s' % {k: round(v, 4) for k, v in dest_mean.items()})

q50_none = np.median(SHARE[0].reshape(-1, NH), axis=0)
q50_mid1 = np.median(SHARE[2].reshape(-1, NH), axis=0)
gain_none = float(q50_none[NL] - q50_none[L_MID])

npz_p = os.path.join(BASE, 'collect.npz')
np.savez_compressed(npz_p,
                    SHARE=SHARE.astype(np.float32), DEST=DEST.astype(np.float32),
                    q50_none=q50_none.astype(np.float32), q50_mid1=q50_mid1.astype(np.float32),
                    anchor_idx=idx.astype(np.int64), l_mid=np.int64(L_MID),
                    d1=np.int64(d1), top8_dims=top8_dims.astype(np.int64),
                    anchor_meta=json.dumps(anchor_meta, ensure_ascii=False))
npz_sha = sha8(open(npz_p, 'rb').read())
log('npz saved sha8=%s' % npz_sha)

result = {
    'phase': 3160, 'name': NAME, 'mode': MODEL, 'smoke': bool(SMOKE),
    'prereg': design['prereg'], 'deviation': design['destination_def'].split('DESIGN DEVIATION: ')[1],
    'model': {'V': V, 'D': D, 'NL': NL, 'L_mid': L_MID, 'd1': d1,
              'top8_dims': top8_dims.tolist()},
    'anchor_meta': anchor_meta,
    'share_nl_none': share_nl['none'], 'share_lm_none': share_lm['none'],
    'share_nl': share_nl, 'share_lm': share_lm,
    'recover': recover,
    'mech_class': mech_class, 'mech_class_mid1': cls1, 'mech_class_mid1_2': cls2,
    'gates': {'recover_mlp': RECOVER_MLP, 'recover_attn': RECOVER_ATTN},
    'destination': dest_mean,
    'gain_none_q50': gain_none,
    'det': {'anchor_bitwise_3157': True, 'pre_slot_rel_max': base_rel_max,
            'dirs': 'top64 pick=%s' % pick.tolist(), 'batch': BATCH_REP},
    'verdict': 'g4p3_%s|rec_mid1_%.4f|rec_mid12_%.4f|e1_%.4f' % (
        mech_class, c1, c2, dest_mean['share_e1_lm2']),
    'npz_sha8': npz_sha, 'design_sha': DESIGN_SHA,
    'runtime_s': round(time.time() - T0, 1),
}
seal_result(result, 'result.json')
log('DONE model=%s smoke=%s' % (MODEL, SMOKE))
