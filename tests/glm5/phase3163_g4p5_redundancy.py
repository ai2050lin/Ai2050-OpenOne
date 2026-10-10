# -*- coding: utf-8 -*-
# Phase 3163 (G4-P5): 消耗冗余性判别 —— 真恒等块 + 扩展窗 (机制链 3159->3160->3161 收官)
# 预注册(逐字): AGI_GPT5_MEMO Phase 3161 closeout 预注册 Phase 3163 (观测前冻结):
#   假设: 3161 证单组件置零不恢复 -> 冗余假说: 任一组件被移除后, 下游流补完旋转。
#         P5 用「真恒等块」与「扩展窗」判别补完的位置与载体。
#   设计(4 锚 x 6 top64 方向 x alpha=0.1 同口径):
#     配置 A=联合置零(全头+MLP)块 {L_mid,+1,+2}(块变恒等映射) -> 装置门: 槽 L_mid+3 的
#           share >= 0.95; 读 share(NL);
#     配置 B=扩展窗全头置零(块 L_mid..NL-1 全部注意力头);
#     配置 C=扩展窗联合置零(残差恒等装置门: NL share=1±0.05)。
#   门: |share_A(NL) - share_none(NL)| < 0.05 -> redundant_closing(3 块内消耗完全冗余) /
#       >= 0.1 -> joint_localized(3 块联合承担不可替代消耗);
#       B: share_B(NL) >= 0.5 -> attention_primary_extended / <= 0.15 -> mlp_or_residual_primary;
#       C: share >= 0.95 装置恒等。GPU ~3min/模型。执行后回图谱主线 G5-A2(缺口②)。
# 物理澄清(观测前冻结):
#   (1) 「全头置零」= 3161 封存口径: o_proj/dense 输入 (B,T,H*HD) 全头切片置零
#       (= attention 输出贡献为 0, GQA KV 不切分); 「MLP 置零」= 3160 封存口径: mlp 模块
#       forward hook 输出替换 zeros。联合 => 块输出 = in + 0 + 0 = in(残差恒等);
#       o_proj bias 存在性记录于 det, 装置门兜底。
#   (2) 中间带 [0.05,0.1) 与 (0.15,0.5) 预注册未定义 -> band_undecided(不发明新类别)。
#   (3) 装置门失败 -> verdict 后缀 _device_gate_fail, 诚实记录不关链。
# 协议逐字继承 3161: 4 锚(3159 anchor_idx linspace4) x 6 top64 方向(linspace(0,63,6)) x
#   alpha=0.1 rel ||h_mid||; 注入 hook 在 block L_mid-1 输出末 token; share = dh 在 3158
#   top64 子空间能量份额(槽 >= L_mid); aggregate-first 均值; base/pert 成对同 chunk
#   (行独立性 -> dh 前 L_mid 槽逐位 0); CHUNK 自适应 + OOM 对半兜底; eps=batch=1 锚态槽
#   L_mid; 4b=bf16 / 14b=NF4(pre-quantized checkpoint) / glm4=NF4(现场量化)。
# SMOKE 重冻结(2026-10-09, 任何正式观测前; 4b SMOKE 2 锚暴露, 装置语义澄清非门放松):
#   (R1) 索引修正: SPECS 顺序 none/A/B/C -> curves 行 0/1/2/3; SHARE_A3 应取 curves[1]
#        (首版误取 curves[2]=B; 主判决列 s_A/s_B 取值一直正确, 仅 A3 装置门读错行)。
#   (R2) C 装置门口径澄清: 实现语义中槽 NL = final RMSNorm 之后(hidden_states 末槽),
#        norm 的 Jacobian 不保 top64 子空间 -> 预注册字面「NL share=1±0.05」物理不可达
#        (SMOKE 实测 norm 效应 share_C(NL-1)=0.9998 -> share_C(NL)=0.9353, 降幅 0.065;
#        none 曲线同槽位降幅同量级)。重冻结 C 装置门 = 三条合取:
#        (i) 恒等窗逐位传播: dh_C 槽 L_mid..NL-1 逐位恒定(ident_c < 1e-4; norm 前口径);
#        (ii) share_C(NL) >= 0.90(norm 后下限, norm 效应 0.065 给 3.5% 余量);
#        (iii) norm 效应对照: drop_norm_C 与 drop_norm_none 同量级(det 记录)。
#        逐位传播把窗内泄漏钉死在 <1e-4, 门整体仍是紧的。
# 装置锚: batch=1 锚态 vs 3157(4b bitwise / NF4 cos>=0.96 且 rel<=0.30); share_none(L_mid)>0.95;
#   A 恒等窗: share_A(L_mid+3)>=0.95 且 ident_a<1e-4; efficacy: A-base vs none-base 槽
#   L_mid+1 maxabs>1e-3; B-base vs none-base 槽 NL rel>0.01(hook 活性检查)。
# 教训内置: SMOKE 目录分离; design 全 JSON; fail-fast 断言; 数字一律 result 现场渲染;
#           hook 兼容 tuple/Tensor; 无百分号字面陷阱; summary 须显式 P3163_MODEL=summary。
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
MODEL = os.environ.get('P3163_MODEL') or (sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b')
SMOKE = os.environ.get('P3163_SMOKE') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g4p5_redundancy'
BASE = os.path.join(RDIR, 'phase3163', NAME, MODEL)
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
        json.dump({'phase': 3163, 'name': phase_name, 'design_sha': sha,
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
G_RED = 0.05
G_JOINT = 0.1
B_ATTN = 0.5
B_MLP = 0.15
DEV_A3 = 0.95
DEV_C_NORM = 0.90
IDENT_MAX = 1e-4

design = {
    'phase': '3163', 'name': NAME, 'model': MODEL, 'smoke': bool(SMOKE),
    'prereg': 'AGI_GPT5_MEMO Phase 3161 closeout prereg of Phase 3163 (frozen before any '
              'observation): redundancy discrimination via true identity blocks and extended '
              'window. A=joint zeroing (all heads + MLP) on blocks {L_mid,+1,+2} (block becomes '
              'identity map) -> device gate share(L_mid+3) >= 0.95, read share(NL); B=extended '
              'window all-head zeroing (blocks L_mid..NL-1 all attention heads); C=extended '
              'window joint zeroing (residual identity device gate)',
    'gates': 'A: |share_A(NL)-share_none(NL)| < 0.05 -> redundant_closing (consumption inside '
             '3 blocks fully redundant) / >= 0.1 -> joint_localized (3 blocks jointly carry '
             'irreplaceable consumption); B: share_B(NL) >= 0.5 -> attention_primary_extended / '
             '<= 0.15 -> mlp_or_residual_primary; middle band [0.05,0.1) and (0.15,0.5) '
             'prereg-undefined -> band_undecided (no invented class); device-gate fail -> '
             'suffix _device_gate_fail (honest, no chain closure)',
    'smoke_refreeze': 'R1: index fix, SHARE_A3 takes curves[1] (A); first draft took curves[2]'
                      '(=B); main verdict columns s_A/s_B were always correct. R2: C device '
                      'gate slot-semantics clarification: slot NL is AFTER final RMSNorm, whose '
                      'Jacobian does not preserve the top64 subspace -> prereg literal '
                      '"NL share = 1 +/- 0.05" physically unreachable (SMOKE: share_C(NL-1)'
                      '=0.9998 -> share_C(NL)=0.9353, norm effect 0.065; none curve shows the '
                      'same-magnitude drop at the same slots). Refrozen C gate = conjunction: '
                      '(i) bitwise identity propagation of dh_C over slots L_mid..NL-1 '
                      '(ident_c < 1e-4, pre-norm); (ii) share_C(NL) >= 0.90 post-norm floor; '
                      '(iii) norm-effect parity: drop_norm_C vs drop_norm_none recorded in det. '
                      'Both refreezes before any formal observation (4b SMOKE 2-anchor only).',
    'protocol_inherit': '3161 verbatim: anchors 3159 anchor_idx linspace4; dirs 3158 top64 '
                        'linspace(0,63,6); inj hook at block L_mid-1 output last token; share = '
                        'energy fraction of dh on top64 subspace (slots >= L_mid); aggregate-first '
                        'means over anchors x dirs; base+pert paired in same chunk (row '
                        'independence => dh slots < L_mid bitwise 0); CHUNK adaptive; eps from '
                        'batch=1 anchor state at slot L_mid',
    'zeroing_impl': 'CLARIFICATION (frozen): "all-head zeroing" = 3161 sealed convention '
                    '(o_proj/dense INPUT (B,T,H*HD) zeroed for all heads = attention output '
                    'contribution is zero; GQA KV untouched); "MLP zeroing" = 3160 sealed '
                    'convention (mlp module forward hook output replaced by zeros). Joint => '
                    'block out = in + 0 + 0 = in (residual identity)',
    'windows': 'A blocks {L_mid, L_mid+1, L_mid+2}; B/C blocks L_mid..NL-1 (full extension); '
               'injection block L_mid-1 untouched (outside both windows)',
    'n_anchors': 4, 'n_dirs': 6, 'alpha': 0.1,
    'precision': 'same as 3161 frozen: 4b=bf16; 14b=NF4 via pre-quantized checkpoint '
                 'Qwen3-14B-bnb-nf4 (5.14 on-the-fly bnb materializes 29.5GB bf16 in RAM -> '
                 'segfault, see 3161 addendum); glm4=NF4 on-the-fly; NF4 anchor tolerance '
                 'cos>=0.96 and rel<=0.30 vs 3157 bf16',
    'execution': 'per-anchor process isolation for 14b/glm4 formal runs (P3163_ANCHOR=k + '
                 'P3163_COLLECT=1), single-process for 4b; same rationale as 3161 (co-tenant '
                 'memory pressure on this machine)',
    'summary_gates': 'class agreement on A-verdict + none-config q50 consumption curve '
                     'fingerprint (L_mid offset aligned, Pearson >= 0.8, same as 3160/3161) + '
                     'A/B/C share table cross-model',
    'rand_seed': 3163,
}
DESIGN_SHA = freeze_design(NAME, design)

if MODEL == 'summary':
    log('summary mode')
    clsA_all, clsB_all, cok_all = {}, {}, {}
    shs = {}
    q50s, lmids = {}, {}
    for m in MODELS3:
        rp = os.path.join(RDIR, 'phase3163', NAME, m, 'result.json')
        r = json.load(open(rp, encoding='utf-8'))
        clsA_all[m] = r['clsA']; clsB_all[m] = r['clsB']; cok_all[m] = r['c_ok']
        shs[m] = (r['share_none_nl'], r['share_A_nl'], r['share_B_nl'], r['share_C_nl'])
        z = np.load(os.path.join(RDIR, 'phase3163', NAME, m, 'collect.npz'))
        q50s[m] = z['q50_none'].astype(np.float64)
        lmids[m] = int(z['l_mid'])
        log('%s: clsA=%s clsB=%s c_ok=%s s_none=%.4f sA=%.4f sB=%.4f sC=%.4f' % (
            m, clsA_all[m], clsB_all[m], cok_all[m], *shs[m]))
    ms = list(MODELS3)
    NHmin = min(q50s[m].shape[0] for m in ms)
    W = min(NHmin - lmids[m] for m in ms)
    pairs = {}
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = ms[i], ms[j]
            ca = q50s[a][lmids[a]:lmids[a] + W]
            cb = q50s[b][lmids[b]:lmids[b] + W]
            pairs['%s_vs_%s' % (a, b)] = float(np.corrcoef(ca, cb)[0, 1])
    fpmin = float(min(pairs.values()))
    fp_ok = fpmin >= FP_GATE
    agree = len(set(clsA_all.values())) == 1
    cok = all(cok_all.values())
    if agree and fp_ok and cok:
        verdict = 'g4p5_%s|agree|fp_ok|c_ok' % clsA_all[ms[0]]
    elif agree and fp_ok:
        verdict = 'g4p5_%s|agree|fp_ok|c_device_%s' % (clsA_all[ms[0]], 'ok' if cok else 'fail')
    elif agree:
        verdict = 'g4p5_%s|agree|fp_low_%.4f' % (clsA_all[ms[0]], fpmin)
    else:
        verdict = 'g4p5_divergent|fpmin_%.4f|classes_%s' % (
            fpmin, '/'.join(clsA_all[m] for m in ms))
    result = {
        'phase': 3163, 'name': NAME, 'mode': 'summary', 'smoke': bool(SMOKE),
        'models': ms, 'clsA': clsA_all, 'clsB': clsB_all, 'c_ok': cok_all,
        'shares': shs,
        'fp_pairs_q50_none': pairs, 'fpmin_q50_none': fpmin, 'fp_gate': FP_GATE,
        'consumption_window': W,
        'gates': {'fingerprint': bool(fp_ok), 'class_agreement': bool(agree), 'c_device': bool(cok)},
        'verdict': verdict, 'design_sha': DESIGN_SHA,
        'runtime_s': round(time.time() - T0, 1),
    }
    seal_result(result, 'result_summary.json')
    log('SUMMARY DONE')
    sys.exit(0)

ANCHOR_K = os.environ.get('P3163_ANCHOR')
COLLECT = os.environ.get('P3163_COLLECT') == '1'
if (ANCHOR_K is not None or COLLECT) and SMOKE:
    raise RuntimeError('anchor/collect modes are formal-only (no SMOKE)')
if ANCHOR_K is not None:
    ANCHOR_K = int(ANCHOR_K)

# ---- CPU 侧公共设置(model / anchor / collect 三模式共用) ----
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MDIR_MAP = {'qwen3-4b': 'qwen3-4b', 'glm4': 'glm4-9b-chat-hf'}
# 14b: pre-quantized NF4 checkpoint (converted once under tf4.57 streaming loader;
# 5.14 on-the-fly bnb load materializes 29.5GB bf16 in RAM -> segfault, issue #43032
# family; see 3161 addendum). Same bnb 0.50.2 NF4 kernel; 3161 diag5 verified slot
# semantics = stock 5.14.
MDIR_MAP['qwen3-14b'] = 'Qwen3-14B-bnb-nf4'
MDIR = os.path.join(ROOT, 'models', 'hf', MDIR_MAP[MODEL])
cfgm = json.load(open(os.path.join(MDIR, 'config.json'), encoding='utf-8'))
V, D = int(cfgm['vocab_size']), int(cfgm['hidden_size'])
NL = int(cfgm['num_hidden_layers'])
L_MID = int(round(0.5 * NL))
H = int(cfgm.get('num_attention_heads') or cfgm.get('n_head'))
HD = int(cfgm['head_dim']) if 'head_dim' in cfgm else D // H
ALPHA = 0.1
N_ANCH = 2 if SMOKE else 4
N_DIR = 6
CHUNK = 128 if D < 4096 else (48 if D < 4608 else 16)
BLKS3 = (L_MID, L_MID + 1, L_MID + 2)
EXT = tuple(range(L_MID, NL))
NH = NL + 1
assert L_MID + 3 <= NL, ('A device gate needs slot L_mid+3 within curve', L_MID, NL)
assert L_MID >= 1, 'injection block L_mid-1 out of range'
log('model=%s V=%d D=%d NL=%d L_mid=%d H=%d HD=%d smoke=%s' % (
    MODEL, V, D, NL, L_MID, H, HD, SMOKE))

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

# 3157 行重建(零抄写, 同 3160/3161)
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

am_ = np.abs(H7[idx7, L_MID, :]).mean(0)
d1 = int(am_.argmax())
log('massive dim d1=%d (continuity, unused in gates)' % d1)

top64f = top64.astype(np.float64)
PRM = {'v': 0.0}
ANC = {'note': 'bitwise'}
PREC = 'bf16' if MODEL == 'qwen3-4b' else 'nf4'
NPR = 4  # 行 0/1/2/3 = none / A / B / C
# 每方向 8 行 = 4 配置 x (base, pert); specs = (attn_mode, mlp_mode, pert)
#   attn/mlp mode: 0=none, 1=blocks3({L_mid..L_mid+2}), 2=extended(L_mid..NL-1)
SPECS = [(0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 0, 1), (2, 2, 0), (2, 2, 1)]
log('configs: none / A(3blk joint) / B(ext attn) / C(ext joint); rows/dir=%d, chunks/dir=%d' % (
    len(SPECS), max(1, int(np.ceil(len(SPECS) / CHUNK)))))

def aggregate_and_seal(SHARE_NL, SHARE_LM, SHARE_A3, CURVES, anchor_meta,
                       prm_v, eff_a, eff_b, ida_v, idc_v, bias_note, exec_note):
    s_mean = SHARE_NL.mean(axis=(0, 1))
    s_none, s_A, s_B, s_C = [float(x) for x in s_mean]
    none_lm = float(SHARE_LM[..., 0].mean())
    assert none_lm > 0.95, ('none share at L_mid too low', none_lm)
    a3 = float(SHARE_A3.mean())
    dA = abs(s_A - s_none)
    if dA < G_RED:
        clsA = 'redundant_closing'
    elif dA >= G_JOINT:
        clsA = 'joint_localized'
    else:
        clsA = 'band_undecided'
    if s_B >= B_ATTN:
        clsB = 'attention_primary_extended'
    elif s_B <= B_MLP:
        clsB = 'mlp_or_residual_primary'
    else:
        clsB = 'band_undecided'
    ida_ok = ida_v < IDENT_MAX
    idc_ok = idc_v < IDENT_MAX
    c_floor_ok = s_C >= DEV_C_NORM
    # norm 效应对照: none 与 C 在最后槽(NL = final RMSNorm 之后)的降幅
    drop_none = float(CURVES[..., 0, NL - 1].mean()) - s_none
    scp_prev = float(CURVES[..., 3, NL - 1].mean())
    drop_C = scp_prev - s_C
    c_ok = idc_ok and c_floor_ok
    a3_ok = a3 >= DEV_A3
    suffix = '' if (c_ok and a3_ok and ida_ok) else '_device_gate_fail'
    log('shares: none(NL)=%.4f A(NL)=%.4f B(NL)=%.4f C(NL)=%.4f; device: A3=%.4f ident_a=%.3g ident_c=%.3g '
        'norm_drop none=%.4f C=%.4f (C pre-norm share=%.4f) -> A=%s B=%s' % (
        s_none, s_A, s_B, s_C, a3, ida_v, idc_v, drop_none, drop_C, scp_prev, clsA, clsB))
    npz_p = os.path.join(BASE, 'collect.npz')
    q50_none = np.median(CURVES[..., 0, :].reshape(-1, NH), axis=0)
    np.savez_compressed(npz_p,
                        SHARE_NL=SHARE_NL.astype(np.float32), SHARE_LM=SHARE_LM.astype(np.float32),
                        SHARE_A3=SHARE_A3.astype(np.float32), CURVES=CURVES.astype(np.float32),
                        q50_none=q50_none.astype(np.float32),
                        anchor_idx=idx.astype(np.int64), l_mid=np.int64(L_MID),
                        heads=np.array([H, HD], np.int64),
                        anchor_meta=json.dumps(anchor_meta, ensure_ascii=False))
    npz_sha = sha8(open(npz_p, 'rb').read())
    log('npz saved sha8=%s' % npz_sha)
    result = {
        'phase': 3163, 'name': NAME, 'mode': MODEL, 'smoke': bool(SMOKE),
        'prereg': design['prereg'],
        'clarifications': [design['zeroing_impl'], design['windows'], design['smoke_refreeze']],
        'model': {'V': V, 'D': D, 'NL': NL, 'L_mid': L_MID, 'H': H, 'HD': HD,
                  'oproj': OPROJ_NAME, 'd1': d1},
        'anchor_meta': anchor_meta,
        'share_none_lm': none_lm,
        'share_none_nl': s_none, 'share_A_nl': s_A, 'share_B_nl': s_B, 'share_C_nl': s_C,
        'share_A_lmid3': a3, 'dA': dA,
        'clsA': clsA, 'clsB': clsB,
        'c_ok': c_ok,
        'device_gates': {'A_identity_window': bool(a3_ok and ida_ok),
                         'C_identity_propagation_bitwise': bool(idc_ok),
                         'C_post_norm_floor': bool(c_floor_ok)},
        'gates': {'redundant': G_RED, 'joint': G_JOINT, 'B_attn': B_ATTN, 'B_mlp': B_MLP,
                  'dev_A3': DEV_A3, 'dev_C_norm_floor': DEV_C_NORM, 'ident_max': IDENT_MAX},
        'det': {'anchor_check': ANC['note'], 'pre_slot_rel_max': prm_v,
                'efficacy_A_maxabs_Lmid1': eff_a, 'efficacy_B_rel_NL': eff_b,
                'identity_window_rel_max_A': ida_v, 'identity_window_rel_max_C': idc_v,
                'norm_effect': {'drop_none': drop_none, 'drop_C': drop_C,
                                'share_C_pre_norm': scp_prev},
                'oproj_bias': bias_note,
                'dirs': 'top64 pick=%s' % pick.tolist(), 'chunk_rows': CHUNK,
                'precision': PREC, 'execution': exec_note, 'rand_seed': 3163,
                'n_pairs_per_dir': NPR},
        'verdict': 'g4p5_%s|B_%s|C_%s|dA_%.4f|sB_%.4f|sC_%.4f%s' % (
            clsA, clsB, 'pass' if c_ok else 'fail', dA, s_B, s_C, suffix),
        'npz_sha8': npz_sha, 'design_sha': DESIGN_SHA,
        'runtime_s': round(time.time() - T0, 1),
    }
    seal_result(result, 'result.json')
    return result

if COLLECT:
    pdir_parts = os.path.join(BASE, '_parts')
    parts = []
    for k in range(N_ANCH):
        pp = os.path.join(pdir_parts, 'collect_anchor%d.npz' % k)
        z = np.load(pp, allow_pickle=False)
        assert int(z['k']) == k and int(z['row']) == int(idx[k]), ('part mismatch', k)
        parts.append(z)
    SHARE_NLc = np.stack([np.asarray(p['share_nl'], np.float64) for p in parts])
    SHARE_LMc = np.stack([np.asarray(p['share_lm'], np.float64) for p in parts])
    SHARE_A3c = np.stack([np.asarray(p['share_a3'], np.float64) for p in parts])
    CURVESc = np.stack([np.asarray(p['curves'], np.float64) for p in parts])
    metac = [json.loads(str(p['anchor_meta_json'])) for p in parts]
    prmc = max(float(p['prm']) for p in parts)
    eff_ac = float(parts[0]['eff_a']); eff_bc = float(parts[0]['eff_b'])
    idac = max(float(p['ident_a']) for p in parts)
    idcc = max(float(p['ident_c']) for p in parts)
    OPROJ_NAME = str(parts[0]['oproj'])
    bias_note = str(parts[0]['attn_bias'])
    log('collect: 4 parts merged (prm=%.3g eff_a=%.4g eff_b=%.4g ida=%.3g idc=%.3g oproj=%s)' % (
        prmc, eff_ac, eff_bc, idac, idcc, OPROJ_NAME))
    aggregate_and_seal(SHARE_NLc, SHARE_LMc, SHARE_A3c, CURVESc, metac,
                       prmc, eff_ac, eff_bc, idac, idcc, bias_note,
                       'per-anchor process isolation (4 fresh processes) + collect merge '
                       '(same rationale as 3161: co-tenant memory pressure)')
    log('COLLECT DONE model=%s' % MODEL)
    sys.exit(0)

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
log('model load begin: prec=%s' % PREC)
if PREC == 'bf16':
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()
else:
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    free_v, tot_v = torch.cuda.mem_get_info()
    if free_v > 11.0 * (1 << 30):
        model = AutoModelForCausalLM.from_pretrained(
            MDIR, quantization_config=bnb, device_map={'': 0}, trust_remote_code=True).eval()
        log('device_map: all-GPU NF4 (free_vram=%.1f GB)' % (free_v / (1 << 30)))
    else:
        # Mixed placement fallback (3161 frozen): layers 0..GPU_L-1 NF4 GPU, rest CPU bf16.
        # Not expected for pre-quantized 14b (~9.9GB) unless co-tenant VRAM spikes.
        GPU_L = 20
        assert GPU_L > L_MID - 1, 'injection layer L_mid-1 must stay on GPU'
        assert free_v > 3.8 * (1 << 30), ('vram below floor for GPU_L=20', free_v / (1 << 30))
        bnb2 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_use_double_quant=True,
                                  llm_int8_enable_fp32_cpu_offload=True)
        dmap = {'model.embed_tokens': 'cpu', 'model.norm': 0, 'lm_head': 'cpu'}
        for _li in range(NL):
            dmap['model.layers.%d' % _li] = 0 if _li < GPU_L else 'cpu'
        model = AutoModelForCausalLM.from_pretrained(
            MDIR, quantization_config=bnb2, device_map=dmap, dtype=torch.bfloat16,
            trust_remote_code=True).eval()
        model.config.use_cache = False
        log('device_map: MIXED GPU_L=%d (free_vram=%.1f GB)' % (GPU_L, free_v / (1 << 30)))
log('model loaded: %s vram_alloc=%.2f GB' % (type(model).__name__, torch.cuda.memory_allocated() / 1e9))
assert model.config.num_hidden_layers == NL
dev = 'cuda'
BASEM = model.model
assert hasattr(BASEM, 'layers'), 'base transformer without layers'
pre_ids = tok(d7['prefix'], add_special_tokens=False)['input_ids']
ctx_ids = (pre_ids * (int(d7['k_ctx']) // len(pre_ids) + 1))[:int(d7['k_ctx'])]

# --- hooks ---
INJ = {'on': False, 'delta': None}
ABL = {'attn': None, 'mlp': None}  # per-row mode tensors: 0=none 1=blk3 2=extended

def _inj_hook(module, args, output):
    if INJ['on'] and INJ['delta'] is not None:
        out0 = output[0] if isinstance(output, tuple) else output
        new0 = out0.clone()
        new0[:, -1, :] = new0[:, -1, :] + INJ['delta'].to(new0.dtype)
        if isinstance(output, tuple):
            return (new0,) + tuple(output[1:])
        return new0
    return None

def _make_attn_hook(blk_no):
    # 3161 convention: zero o_proj INPUT rows (B,T,H*HD) entirely => attn output contribution 0
    def _attn_pre(module, args):
        at = ABL['attn']
        if at is None:
            return None
        in3 = blk_no in BLKS3
        if in3:
            sel = ((at == 1) | (at == 2)).nonzero(as_tuple=True)[0]
        else:
            sel = (at == 2).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            return None
        x = args[0]
        mask = torch.ones_like(x)
        mask[sel] = 0.0
        return (x * mask,) + tuple(args[1:])
    return _attn_pre

def _make_mlp_hook(blk_no):
    # 3160 convention: replace mlp module OUTPUT with zeros for selected rows
    def _mlp_hook(module, args, output):
        mm = ABL['mlp']
        if mm is None:
            return None
        in3 = blk_no in BLKS3
        if in3:
            sel = ((mm == 1) | (mm == 2)).nonzero(as_tuple=True)[0]
        else:
            sel = (mm == 2).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            return None
        out0 = output[0] if isinstance(output, tuple) else output
        z = out0.clone()
        z[sel] = 0.0
        if isinstance(output, tuple):
            return (z,) + tuple(output[1:])
        return z
    return _mlp_hook

def resolve_oproj(layer):
    cands = [('self_attn', 'o_proj'), ('self_attention', 'dense'),
             ('self_attn', 'out_proj'), ('self_attention', 'projection')]
    for a, b in cands:
        if hasattr(layer, a):
            m = getattr(layer, a)
            if hasattr(m, b):
                lin = getattr(m, b)
                if isinstance(lin, torch.nn.Linear) and lin.in_features == H * HD:
                    return a + '.' + b, lin
    best = None
    for nm, mod in layer.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.in_features == H * HD and \
           any(k in nm for k in ('o_proj', 'dense', 'out_proj', 'projection')):
            if best is None or len(nm) < len(best[0]):
                best = (nm, mod)
    if best is None:
        raise RuntimeError('attn out-proj not found in layer')
    return best

OPROJ_NAME = None
BIAS_NOTE = None
for b in EXT:
    nm, mod = resolve_oproj(BASEM.layers[b])
    if OPROJ_NAME is None:
        OPROJ_NAME = nm
        BIAS_NOTE = 'none' if mod.bias is None else 'present'
    assert nm.split('.')[-1] == OPROJ_NAME.split('.')[-1], ('oproj inconsistent', nm, OPROJ_NAME)
    mod.register_forward_pre_hook(_make_attn_hook(b))
    assert hasattr(BASEM.layers[b], 'mlp'), ('mlp module missing', b)
    BASEM.layers[b].mlp.register_forward_hook(_make_mlp_hook(b))
BASEM.layers[L_MID - 1].register_forward_hook(_inj_hook)
log('hooks: inj@L%d; attn-zero+mlp-zero on L%d..L%d (%d blocks); oproj=%s bias=%s' % (
    L_MID - 1, L_MID, NL - 1, len(EXT), OPROJ_NAME, BIAS_NOTE))

def _forward_rows(ids, specs, dvec, eps0):
    n = len(specs)
    am = np.array([s[0] for s in specs], np.int64)
    mm = np.array([s[1] for s in specs], np.int64)
    delta = np.zeros((n, D), np.float32)
    for i, s in enumerate(specs):
        if s[2]:
            delta[i] = (eps0 * dvec).astype(np.float32)
    ii = torch.tensor([ids] * n, dtype=torch.int64, device=dev)
    INJ['delta'] = torch.from_numpy(delta).to(dev)
    INJ['on'] = True
    ABL['attn'] = torch.from_numpy(am).to(dev)
    ABL['mlp'] = torch.from_numpy(mm).to(dev)
    try:
        with torch.no_grad():
            o = BASEM(input_ids=ii, output_hidden_states=True)
    finally:
        INJ['on'] = False
        INJ['delta'] = None
        ABL['attn'] = None
        ABL['mlp'] = None
    hs = np.stack([h[:, -1].float().detach().cpu().numpy() for h in o.hidden_states], 1)
    del o
    return hs   # (n, NH, D) float32

def run_dir(ids, dvec, eps0):
    n = len(SPECS)
    assert n <= CHUNK, ('specs exceed chunk', n, CHUNK)
    curves = np.zeros((NPR, NH), np.float64)
    try:
        hs = _forward_rows(ids, SPECS, dvec, eps0)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        half = (n // 4) * 2
        hs = np.concatenate([_forward_rows(ids, SPECS[:half], dvec, eps0),
                             _forward_rows(ids, SPECS[half:], dvec, eps0)], 0)
    for j in range(NPR):
        base = hs[2 * j].astype(np.float64)
        pert = hs[2 * j + 1].astype(np.float64)
        dh = pert - base
        pre_rel = float(np.abs(dh[:L_MID]).max() / (np.linalg.norm(base[L_MID]) + 1e-18))
        assert pre_rel < 1e-5, ('pre-slot dh too large', j, pre_rel)
        if pre_rel > PRM['v']:
            PRM['v'] = pre_rel
        num = (dh[L_MID:] @ top64f) ** 2
        den = (dh[L_MID:] ** 2).sum(1) + 1e-18
        sh = num.sum(1) / den
        curves[j, L_MID:] = sh
    # identity-window bitwise propagation:
    #   A (rows 2/3): dh slots L_mid..L_mid+3 constant; C (rows 6/7): slots L_mid..NL-1 constant
    dha = (hs[3] - hs[2]).astype(np.float64)
    da0 = dha[L_MID]
    dna = np.linalg.norm(da0) + 1e-18
    ident_a = max(float(np.linalg.norm(dha[L_MID + k] - da0) / dna) for k in (1, 2, 3))
    dhc = (hs[7] - hs[6]).astype(np.float64)
    dc0 = dhc[L_MID]
    dnc = np.linalg.norm(dc0) + 1e-18
    ident_c = max(float(np.linalg.norm(dhc[s] - dc0) / dnc) for s in range(L_MID + 1, NL))
    del hs
    return curves, ident_a, ident_c

SHARE_NL = np.zeros((N_ANCH, N_DIR, NPR), np.float64)
SHARE_LM = np.zeros((N_ANCH, N_DIR, NPR), np.float64)
SHARE_A3 = np.zeros((N_ANCH, N_DIR), np.float64)
CURVES = np.zeros((N_ANCH, N_DIR, NPR, NH), np.float64)
anchor_meta = []
pre_rel_max = 0.0
eff_a = -1.0
eff_b = -1.0
ida_max = 0.0
idc_max = 0.0
for ai0, r0 in enumerate(idx):
    if ANCHOR_K is not None and ai0 != ANCHOR_K:
        continue
    row = rows7[int(r0)]
    ids = (ctx_ids if row['ctx'] == 1 else []) + tok(row['prompt'], add_special_tokens=False)['input_ids']
    ii1 = torch.tensor([ids], dtype=torch.int64, device=dev)
    with torch.no_grad():
        o1 = BASEM(input_ids=ii1, output_hidden_states=True)
    hs1 = np.stack([h[0, -1].float().detach().cpu().numpy() for h in o1.hidden_states], 0)
    del o1
    if PREC == 'bf16':
        assert np.array_equal(hs1[L_MID].astype(np.float16), H7[int(r0), L_MID].astype(np.float32).astype(np.float16)), \
            ('anchor bitwise vs 3157', int(r0))
        ANC['note'] = 'bitwise'
    else:
        h_ref = H7[int(r0), L_MID].astype(np.float64)
        h_new = hs1[L_MID].astype(np.float64)
        c_anc = float((h_new * h_ref).sum() / (np.linalg.norm(h_new) * np.linalg.norm(h_ref) + 1e-18))
        r_anc = float(np.linalg.norm(h_new - h_ref) / (np.linalg.norm(h_ref) + 1e-18))
        assert c_anc >= 0.96 and r_anc <= 0.30, ('anchor NF4 tolerance vs 3157 bf16', c_anc, r_anc)
        ANC['note'] = 'nf4_tolerance cos=%.4f rel=%.4f' % (c_anc, r_anc)
        log('anchor tolerance check (NF4 vs 3157 bf16): cos=%.4f rel=%.4f' % (c_anc, r_anc))
    eps0 = ALPHA * float(np.linalg.norm(hs1[L_MID].astype(np.float64)))
    if ai0 == 0:
        # hook 活性检查: A-base vs none-base 槽 L_mid+1 maxabs; B-base vs none-base 槽 NL rel
        hs_chk = _forward_rows(ids, [(0, 0, 0), (1, 1, 0), (2, 0, 0)], np.zeros(D, np.float64), 0.0)
        eff_a = float(np.abs(hs_chk[1, L_MID + 1] - hs_chk[0, L_MID + 1]).max())
        eff_b = float(np.linalg.norm(hs_chk[2, NL] - hs_chk[0, NL]) /
                      (np.linalg.norm(hs_chk[0, NL]) + 1e-18))
        assert eff_a > 1e-3, ('A ablation efficacy too small (hook dead?)', eff_a)
        assert eff_b > 0.01, ('B extended-window efficacy too small (hook dead?)', eff_b)
        log('efficacy: A maxabs@L_mid+1=%.4g; B rel@NL=%.4g (both vs none base)' % (eff_a, eff_b))
        del hs_chk
    for di in range(N_DIR):
        curves, ida, idc = run_dir(ids, U[:, di], eps0)
        if ida > ida_max:
            ida_max = ida
        if idc > idc_max:
            idc_max = idc
        SHARE_NL[ai0, di] = curves[:, NL]
        SHARE_LM[ai0, di] = curves[:, L_MID]
        SHARE_A3[ai0, di] = curves[1, L_MID + 3]  # R1 fix: row 1 = A (row 2 = B)
        CURVES[ai0, di] = curves
    assert ida_max < IDENT_MAX, ('A identity window propagation broken', ida_max)
    assert idc_max < IDENT_MAX, ('C identity window propagation broken', idc_max)
    sh_none_lm = float(SHARE_LM[ai0, :, 0].mean())
    assert sh_none_lm > 0.95, ('none share at L_mid too low', sh_none_lm)
    anchor_meta.append(dict(row=int(r0), ei=int(row['ei']), rel=row['rel'], pol=row['pol'],
                            ctx=int(row['ctx'])))
    log('anchor %d/%d row=%d done; none: s(L_mid)=%.4f s(NL)=%.4f | A s(NL)=%.4f B s(NL)=%.4f C s(NL)=%.4f' % (
        ai0 + 1, N_ANCH, int(r0), sh_none_lm,
        float(SHARE_NL[ai0, :, 0].mean()), float(SHARE_NL[ai0, :, 1].mean()),
        float(SHARE_NL[ai0, :, 2].mean()), float(SHARE_NL[ai0, :, 3].mean())))

if ANCHOR_K is not None:
    pdir_parts = os.path.join(BASE, '_parts')
    os.makedirs(pdir_parts, exist_ok=True)
    np.savez_compressed(os.path.join(pdir_parts, 'collect_anchor%d.npz' % ANCHOR_K),
                        k=np.int64(ANCHOR_K), row=np.int64(idx[ANCHOR_K]),
                        share_nl=SHARE_NL[ANCHOR_K].astype(np.float32),
                        share_lm=SHARE_LM[ANCHOR_K].astype(np.float32),
                        share_a3=SHARE_A3[ANCHOR_K].astype(np.float32),
                        curves=CURVES[ANCHOR_K].astype(np.float32),
                        anchor_meta_json=json.dumps(anchor_meta, ensure_ascii=False),
                        prm=np.float64(PRM['v']), eff_a=np.float64(eff_a),
                        eff_b=np.float64(eff_b), ident_a=np.float64(ida_max),
                        ident_c=np.float64(idc_max),
                        oproj=str(OPROJ_NAME), attn_bias=str(BIAS_NOTE))
    log('ANCHOR %d DONE (part saved, fresh-process isolation)' % ANCHOR_K)
    sys.exit(0)

aggregate_and_seal(SHARE_NL, SHARE_LM, SHARE_A3, CURVES, anchor_meta,
                   PRM['v'], eff_a, eff_b, ida_max, idc_max, BIAS_NOTE,
                   'single-process (all anchors in one process)')
log('DONE model=%s smoke=%s' % (MODEL, SMOKE))
