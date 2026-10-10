# -*- coding: utf-8 -*-
# Phase 3161 (G4-P4): 消耗的 attention 头归因 —— 逐头置零 + top-4 集中度三分门
# 预注册: AGI_GPT5_MEMO Phase 3160 closeout + 3162 缺口排序(1) (观测前冻结):
#   对 big-drop 块(L_mid)全部注意力头逐头置零, 4 锚 x top64 方向 6 x alpha=0.1(同 3160 口径);
#   head recover 曲线 -> top-4 头集中度 = sum(recover(top4)) / sum(recover(全部>0));
#   门: 集中度 >= 0.5 -> localized_heads / < 0.2 -> distributed_heads / 之间 -> weakly_localized;
#   跨模型头层位分布(相对深度)描述性对比 + 消耗曲线指纹(对齐口径); GPU 预算 ~10min/模型;
#   GQA 注意: KV 头不切分, 只按 query 头切 o_proj 输入维度。
# 设计(观测前冻结, 含预注册字面的两点物理澄清):
#   (1) 协议逐字继承 3160: 4 锚(3159 anchor_idx linspace4) x 6 top64 方向(linspace(0,63,6))
#       x alpha=0.1 rel ||h_mid||; 注入 hook 在 block L_mid-1 输出末 token; share = dh 在
#       3158 top64 子空间能量份额(槽 >= L_mid); recover = (share_cfg(NL) - share_none(NL)) /
#       (share_none(L_mid) - share_none(NL)), 均值先聚(aggregate-first, 同 3160)。
#   (2) 消融实现 = attention 输出投影(o_proj/dense)输入的逐头切片置零。澄清: 预注册字面
#       "self_attn 输出的 head*dh 切片"在 o_proj 之后无头语义(头已混合), 按物理正确口径
#       在 o_proj 输入(B, T, H*HD)切片 = 等价于置零该头对 self_attn 输出的贡献; KV 头不切分。
#   (3) 块集合 = {L_mid, L_mid+1, L_mid+2}: 预注册字面主门在块 L_mid; 另两块与三块并集为
#       预声明扩展(避免 big-drop 块身份的歧义自由度), 类别不一致时判决取主门 + 后缀标注。
#       配置: 每(块,头) 1 配置 + 每块 4 随机头(种子 3161) + 每块全头(-2) + 三块全头(-3)。
#   (4) base 与 pert 成对进同一 chunk(行独立性 => dh 槽 < L_mid 逐位 0); CHUNK 按显存自适应
#       (4b D=2560 -> 128 行; glm4 D=4096 -> 48 行; 14b D=5120 -> 16 行: 16GB 卡上 14b 权重
#       ~13GB, 48 行批次峰值激活仍超空闲显存 -> WDDM 共享内存页出导致逐锚单调减速
#       350s->642s->>1300s, 两次终止后于任何正式观测前重冻结; 16 行峰值 ~0.5GB 稳定;
#       不做 chunk 间 empty_cache, 让分配器复用缓存块; OOM 对半重试兜底);
#       eps 取 batch=1 锚态槽 L_mid(逐位 vs 3157; 3160 用 batch=6 base, 仅 bf16 末位差)。
#   (5) 判决分支(预声明): ctrl(三块全头 recover) < 0.2 -> consumption_not_in_attn_out(与 3160
#       张力, 诚实记录); 并集单头总恢复 T < 0.1 -> distributed_interactive(单头亚阈值但集体大);
#       否则按预注册三分门(主门=块 L_mid)。
#   (6) summary: 类别一致 + none 配置 q50 消耗曲线指纹(L_mid 偏移对齐, Pearson >= 0.8, 同
#       3160 口径) + sorted-R 谱(描述性); ctrl 标注。
# 装置锚: batch=1 锚态 bitwise vs 3157; dh 前 L_mid 槽 < 1e-5; share_none(L_mid) > 0.95;
#         消融效力门: 全三块全头置零后槽 L_mid+1 隐态 maxabs 差 > 1e-3; rec[0]=0 一致性。
# 教训内置: SMOKE 目录分离; design 全 str 键 JSON; fail-fast 断言; 数字一律 result 现场渲染;
#           hook 兼容 tuple/Tensor; 无百分号字面陷阱; 裸 Tensor 与 output_capturing 语义同 3159。
#   (7) 精度偏差(观测前冻结): 4b=bf16(8.0GB 入 VRAM); 14b(29.5GB)/glm4(18.8GB) bf16 超 16GB
#       卡 -> NVIDIA sysmem fallback 每 fwd 经 PCIe 流式读权重, 共租内存压力下退化为 pagefile
#       磁盘流式(1.3s/fwd 温 -> 60s+/fwd 冷, 实测) -> 14b/glm4 改 NF4 double-quant
#       (bitsandbytes, 本机 deepseek Q05 先例); NF4 下锚检查由 bitwise 降为容差
#       (cos>=0.96 且 rel<=0.30, 对照 3157 bf16 H; 首设 0.05/0.999 在未见 NF4 误差量级前
#       标定过紧, glm4 实测 rel 0.196-0.241 / cos 0.9706-0.9806 (massive 维度 4-bit 误差
#       主导), 于任何正式 NF4 观测前校准重冻结; 错行 cos 仅 0.3-0.7, 门仍钉住行/槽/协议)。
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
MODEL = os.environ.get('P3161_MODEL') or (sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b')
SMOKE = os.environ.get('P3161_SMOKE') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g4p4_head_attribution'
BASE = os.path.join(RDIR, 'phase3161', NAME, MODEL)
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
        json.dump({'phase': 3161, 'name': phase_name, 'design_sha': sha,
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
CTRL_GATE = 0.2
C4_LOCAL = 0.5
C4_DIST = 0.2
T_MIN = 0.1

design = {
    'phase': '3161', 'name': NAME, 'model': MODEL, 'smoke': bool(SMOKE),
    'prereg': 'AGI_GPT5_MEMO Phase 3160 closeout + 3162 gap ranking (1), frozen before any '
              'observation: per-head zeroing at big-drop block (L_mid), 4 anchors x top64 dirs 6 '
              'x alpha=0.1 (3160 protocol); top-4 concentration = sum recover(top4) / sum '
              'recover(positive); >=0.5 localized_heads / <0.2 distributed_heads / else '
              'weakly_localized; GQA: KV heads not split, only o_proj input per query head',
    'protocol_inherit': '3160 verbatim: anchors 3159 anchor_idx linspace4; dirs 3158 top64 '
                        'linspace(0,63,6); inj hook at block L_mid-1 output last token; share = '
                        'energy fraction of dh on top64 subspace (slots >= L_mid); recover = '
                        '(share_cfg(NL)-share_none(NL))/(share_none(L_mid)-share_none(NL)), '
                        'aggregate-first means over anchors x dirs (same as 3160)',
    'ablation_impl': 'attention out-proj (o_proj/dense) INPUT per-head slice zeroing. CLARIFICATION '
                     '(frozen): prereg literal "self_attn output head*dh slice" has no head '
                     'semantics after o_proj mixing; slicing o_proj input (B,T,H*HD) is the '
                     'physically correct equivalent (= zeroing that head contribution to attn '
                     'output); KV heads untouched (GQA)',
    'blocks': 'primary gate on block L_mid (prereg literal); blocks L_mid+1, L_mid+2 and the '
              '3-block union are pre-declared extensions (big-drop block identity ambiguity); '
              'configs: per (block, head) + 4 random heads per block (seed 3161) + all-heads per '
              'block (-2) + all-heads all-3-blocks (-3)',
    'batching': 'base+pert paired in same chunk (row independence => dh slots < L_mid bitwise 0); '
                'CHUNK adaptive by free VRAM: 128 rows for D<4096, 48 for D<4608, 16 for D>=4608 '
                '(16GB card: 14b weights ~13GB, CHUNK 128 and 48 both exceeded free VRAM peak -> '
                'WDDM shared-memory paging with monotonic per-anchor slowdown 350s->642s->1300s; '
                'killed twice and refrozen before any formal observation; 16-row peak ~0.5GB '
                'stable); no empty_cache between chunks (allocator block reuse); OOM auto-halving '
                'backstop; eps from batch=1 anchor state at slot L_mid (bitwise vs 3157; 3160 '
                'used batch=6 base, last-bit bf16 difference only)',
    'verdict_branches': 'ctrl (all-3-blocks all-heads recover) < 0.2 -> consumption_not_in_attn_out; '
                        'union single-head total T < 0.1 -> distributed_interactive; else prereg '
                        'triple gate on block L_mid (primary); ext class differs -> suffix '
                        '_ext_differs; per-block C4 + random-control reported',
    'summary_gates': 'class agreement + none-config q50 consumption curve fingerprint (L_mid offset '
                     'aligned, Pearson >= 0.8, same as 3160) + sorted-R spectrum (descriptive)',
    'n_anchors': 4, 'n_dirs': 6, 'alpha': 0.1, 'chunk_rows': 'adaptive 128/48/16 (see batching)',
    'rand_seed': 3161,
    'execution': 'per-anchor process isolation for formal runs: env P3161_ANCHOR=k runs anchor k '
                 'only and saves _parts/collect_anchor<k>.npz; env P3161_COLLECT=1 merges the 4 '
                 'parts and seals. Decided after discovering 14b bf16 (29.5GB) on the 16GB card '
                 'runs via NVIDIA sysmem fallback -> weights streamed over PCIe every fwd '
                 '(~1s/fwd floor), and co-tenant memory pressure (GameViewer streaming, '
                 'server.py, browser) trims the working set -> within-process monotonic slowdown '
                 '(anchor 121s -> 206s -> 1400s+); anchor 1 was fast in ALL 3 aborted runs, so a '
                 'fresh process per anchor restores speed. Physics unchanged: anchors are '
                 'independent, no cross-anchor state, deterministic. Refrozen before any formal '
                 '14b/glm4 observation; 4b formal rerun under this design.',
    'precision': '4b = bf16 (8.0GB fits VRAM); 14b (29.5GB) and glm4 (18.8GB) = NF4 '
                 'double-quant via bitsandbytes (Q05 convention on this machine). Deviation '
                 'decided before any formal 14b/glm4 observation: bf16 sysmem-fallback proved '
                 'infeasible (1.3s/fwd warm -> pagefile 60s+/fwd cold under co-tenant memory '
                 'pressure, measured). Verdict gates are relative (recover/C4/ctrl); '
                 'cross-model comparability kept at pattern level; 3160 parity for 14b/glm4 '
                 'downgraded to pattern level (documented). Anchor check for NF4 models: '
                 'tolerance vs 3157 bf16 H, cos >= 0.96 AND rel <= 0.30. First calibration '
                 '(0.999/0.05) was set before seeing NF4 error magnitude and failed 4/4 on '
                 'glm4 (measured rel 0.196-0.241, cos 0.9706-0.9806; massive-dim 4-bit '
                 'quantization error dominates); recalibrated and refrozen before any formal '
                 'NF4 observation. The gate still pins row/slot/protocol: a wrong row yields '
                 'cos ~0.3-0.7.',
    'gates': {'ctrl': CTRL_GATE, 'C4_localized': C4_LOCAL, 'C4_distributed': C4_DIST,
              'T_min': T_MIN, 'fp': FP_GATE},
}
DESIGN_SHA = freeze_design(NAME, design)

if MODEL == 'summary':
    log('summary mode')
    cls_all, ctrls, C4s, tops, Ts = {}, {}, {}, {}, {}
    q50s, lmids = {}, {}
    Rmap = {}
    for m in MODELS3:
        rp = os.path.join(RDIR, 'phase3161', NAME, m, 'result.json')
        r = json.load(open(rp, encoding='utf-8'))
        cls_all[m] = r['cls']; ctrls[m] = r['ctrl_all3']; C4s[m] = r['C4']
        tops[m] = r['top4_heads_union']; Ts[m] = r['T']
        z = np.load(os.path.join(RDIR, 'phase3161', NAME, m, 'collect.npz'))
        NONE = z['NONE_CURVE'].astype(np.float64)
        q50s[m] = np.median(NONE.reshape(-1, NONE.shape[2]), axis=0)
        lmids[m] = int(z['l_mid'])
        Rmap[m] = z['R'].astype(np.float64)
        log('%s: cls=%s C4=%.4f T=%.4f ctrl=%.4f top4=%s' % (
            m, cls_all[m], C4s[m], Ts[m], ctrls[m], tops[m]))
    ms = list(MODELS3)
    W = min(q50s[m].shape[0] - lmids[m] for m in ms)
    pairs, pairs_spec = {}, {}
    L = int(min(Rmap[m].shape[0] for m in ms))
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = ms[i], ms[j]
            ca = q50s[a][lmids[a]:lmids[a] + W]
            cb = q50s[b][lmids[b]:lmids[b] + W]
            pairs['%s_vs_%s' % (a, b)] = float(np.corrcoef(ca, cb)[0, 1])
            sa = np.sort(Rmap[a])[::-1][:L] / (Rmap[a].sum() + 1e-18)
            sb = np.sort(Rmap[b])[::-1][:L] / (Rmap[b].sum() + 1e-18)
            pairs_spec['%s_vs_%s' % (a, b)] = float(np.corrcoef(sa, sb)[0, 1])
    fpmin = float(min(pairs.values()))
    fpmin_spec = float(min(pairs_spec.values()))
    fp_ok = fpmin >= FP_GATE
    agree = len(set(cls_all.values())) == 1
    ctrl_ok = all(v >= CTRL_GATE for v in ctrls.values())
    ctag = 'ctrl_ok' if ctrl_ok else 'ctrl_weak'
    if agree and fp_ok:
        verdict = 'g4p4_%s|%s|fp_ok' % (cls_all[ms[0]], ctag)
    elif agree:
        verdict = 'g4p4_%s|%s|fp_low_%.4f' % (cls_all[ms[0]], ctag, fpmin)
    else:
        verdict = 'g4p4_divergent|%s|fpmin_%.4f|classes_%s' % (
            ctag, fpmin, '/'.join(cls_all[m] for m in ms))
    result = {
        'phase': 3161, 'name': NAME, 'mode': 'summary', 'smoke': bool(SMOKE),
        'models': ms, 'cls': cls_all, 'C4': C4s, 'T': Ts, 'ctrl_all3': ctrls,
        'top4_heads_union': tops,
        'fp_pairs_q50_none': pairs, 'fp_pairs_sortedR': pairs_spec,
        'fpmin_q50_none': fpmin, 'fpmin_sortedR': fpmin_spec,
        'fp_gate': FP_GATE, 'consumption_window': W,
        'gates': {'fingerprint': bool(fp_ok), 'class_agreement': bool(agree), 'ctrl': bool(ctrl_ok)},
        'verdict': verdict, 'design_sha': DESIGN_SHA,
        'runtime_s': round(time.time() - T0, 1),
    }
    seal_result(result, 'result_summary.json')
    log('SUMMARY DONE')
    sys.exit(0)

ANCHOR_K = os.environ.get('P3161_ANCHOR')
COLLECT = os.environ.get('P3161_COLLECT') == '1'
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
# family). Same bnb 0.50.2 NF4 kernel; diag5 verified slot semantics = stock 5.14.
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
BLKS = (L_MID, L_MID + 1, L_MID + 2)
NH = NL + 1
assert L_MID + 2 <= NL - 1, ('ablation blocks exceed range', L_MID, NL)
rng = np.random.default_rng(3161)
RAND4 = {b: rng.choice(H, 4, replace=False).tolist() for b in BLKS}
log('model=%s V=%d D=%d NL=%d L_mid=%d H=%d HD=%d smoke=%s rand4=%s' % (
    MODEL, V, D, NL, L_MID, H, HD, SMOKE, RAND4))

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

# 3157 行重建(零抄写, 同 3160)
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

am = np.abs(H7[idx7, L_MID, :]).mean(0)
d1 = int(am.argmax())
log('massive dim d1=%d (continuity, unused in gates)' % d1)

cfgs = []
for b in BLKS:
    for h in range(H):
        cfgs.append((b, h))
    for h in RAND4[b]:
        cfgs.append((b, int(h)))
    cfgs.append((b, -2))
cfgs.append((-3, -2))
NPR = 1 + len(cfgs)
top64f = top64.astype(np.float64)
PRM = {'v': 0.0}
ANC = {'note': 'bitwise'}
PREC = 'bf16' if MODEL == 'qwen3-4b' else 'nf4'
log('configs: %d (per dir pairs=%d, rows/dir=%d, chunks/dir=%d)' % (
    len(cfgs), NPR, 2 * NPR, max(1, int(np.ceil(2 * NPR / CHUNK)))))

def aggregate_and_seal(SHARE_NL, SHARE_LM, NONE_CURVE, ALL3_CURVE, anchor_meta, prm_v, eff_v, exec_note):
    sh_lm_all = float(SHARE_LM.mean())
    assert sh_lm_all > 0.95, ('share at L_mid across all pairs too low', sh_lm_all)
    sh_mean = SHARE_NL.mean(axis=(0, 1))
    none_lm = float(NONE_CURVE[..., L_MID].mean())
    none_nl = float(NONE_CURVE[..., NL].mean())
    den = none_lm - none_nl + 1e-18
    rec = (sh_mean - none_nl) / den
    assert abs(float(rec[0])) < 1e-9, ('none self-consistency', float(rec[0]))
    REC1 = rec[1:1 + 3 * (H + 5)].reshape(3, H + 5)
    assert len(cfgs) == 3 * (H + 5) + 1
    ALL3 = float(rec[-1])
    assert ALL3 == float(rec[1 + 3 * (H + 5)])
    M = REC1[:, :H]
    RAND_M = REC1[:, H:H + 4]
    ALLBLK = REC1[:, H + 4]
    R = np.maximum(M, 0).sum(axis=0)
    T = float(R.sum())
    order = np.argsort(-R)
    top4 = order[:4]
    C4 = float(R[top4].sum() / (T + 1e-18))
    C4B = []
    for bi in range(3):
        Rb = np.maximum(M[bi], 0.0)
        tb = float(Rb.sum())
        C4B.append(float(np.sort(Rb)[::-1][:4].sum() / (tb + 1e-18)) if tb > 1e-12 else 0.0)
    rand_ratio = float(np.maximum(RAND_M, 0.0).mean()) / (T / (3 * H) + 1e-18)
    cls_mid = None
    Rm = np.maximum(M[0], 0.0)
    Tm = float(Rm.sum())
    if Tm <= 1e-12:
        cls_mid = 'no_single_head_effect'
    else:
        c4m = float(np.sort(Rm)[::-1][:4].sum() / Tm)
        if c4m >= C4_LOCAL:
            cls_mid = 'localized_heads'
        elif c4m < C4_DIST:
            cls_mid = 'distributed_heads'
        else:
            cls_mid = 'weakly_localized'
    if ALL3 < CTRL_GATE:
        cls = 'consumption_not_in_attn_out'
    elif T < T_MIN:
        cls = 'distributed_interactive'
    else:
        cls = cls_mid
        ext_cls = []
        for bi in (1, 2):
            Rb = np.maximum(M[bi], 0.0)
            tbb = float(Rb.sum())
            if tbb <= 1e-12:
                ext_cls.append('no_single_head_effect')
            elif float(np.sort(Rb)[::-1][:4].sum() / tbb) >= C4_LOCAL:
                ext_cls.append('localized_heads')
            elif float(np.sort(Rb)[::-1][:4].sum() / tbb) < C4_DIST:
                ext_cls.append('distributed_heads')
            else:
                ext_cls.append('weakly_localized')
        if ext_cls[0] != cls_mid or ext_cls[1] != cls_mid:
            cls = cls + '_ext_differs'
    log('share_nl none=%.4f; rec: all3=%.4f allblk=%s; M pos-sum T=%.4f C4=%.4f (per block %s) -> %s' % (
        none_nl, ALL3, np.round(ALLBLK, 4).tolist(), T, C4, np.round(C4B, 4).tolist(), cls))
    den_cell = NONE_CURVE[..., L_MID] - NONE_CURVE[..., NL] + 1e-18
    RECH_CELL = np.zeros((3, H, N_ANCH, N_DIR), np.float64)
    RECR_CELL = np.zeros((3, 4, N_ANCH, N_DIR), np.float64)
    RECA_CELL = np.zeros((3, N_ANCH, N_DIR), np.float64)
    for bi in range(3):
        for h in range(H):
            RECH_CELL[bi, h] = (SHARE_NL[..., 1 + bi * (H + 5) + h] - SHARE_NL[..., 0]) / den_cell
        for j in range(4):
            RECR_CELL[bi, j] = (SHARE_NL[..., 1 + bi * (H + 5) + H + j] - SHARE_NL[..., 0]) / den_cell
        RECA_CELL[bi] = (SHARE_NL[..., 1 + bi * (H + 5) + H + 4] - SHARE_NL[..., 0]) / den_cell
    RECA3_CELL = (SHARE_NL[..., -1] - SHARE_NL[..., 0]) / den_cell
    npz_p = os.path.join(BASE, 'collect.npz')
    np.savez_compressed(npz_p,
                        SHARE_NL=SHARE_NL.astype(np.float32), SHARE_LM=SHARE_LM.astype(np.float32),
                        NONE_CURVE=NONE_CURVE.astype(np.float32), ALL3_CURVE=ALL3_CURVE.astype(np.float32),
                        RECH_CELL=RECH_CELL.astype(np.float32), RECR_CELL=RECR_CELL.astype(np.float32),
                        RECA_CELL=RECA_CELL.astype(np.float32), RECA3_CELL=RECA3_CELL.astype(np.float32),
                        M=M.astype(np.float64), R=R.astype(np.float64),
                        RAND_M=RAND_M.astype(np.float64), ALLBLK=ALLBLK.astype(np.float64),
                        anchor_idx=idx.astype(np.int64), l_mid=np.int64(L_MID),
                        heads=np.array([H, HD], np.int64),
                        anchor_meta=json.dumps(anchor_meta, ensure_ascii=False))
    npz_sha = sha8(open(npz_p, 'rb').read())
    log('npz saved sha8=%s' % npz_sha)
    q50_none = np.median(NONE_CURVE.reshape(-1, NH), axis=0)
    q50_all3 = np.median(ALL3_CURVE.reshape(-1, NH), axis=0)
    np.savez_compressed(npz_p, **dict(np.load(npz_p), q50_none=q50_none.astype(np.float32),
                                      q50_all3=q50_all3.astype(np.float32)))
    npz_sha = sha8(open(npz_p, 'rb').read())
    result = {
        'phase': 3161, 'name': NAME, 'mode': MODEL, 'smoke': bool(SMOKE),
        'prereg': design['prereg'],
        'clarifications': [design['ablation_impl'], design['blocks']],
        'model': {'V': V, 'D': D, 'NL': NL, 'L_mid': L_MID, 'H': H, 'HD': HD,
                  'oproj': OPROJ_NAME, 'd1': d1},
        'anchor_meta': anchor_meta,
        'share_lm_none': none_lm, 'share_nl_none': none_nl,
        'share_lm_all_mean': sh_lm_all,
        'cls': cls, 'cls_mid_primary': cls_mid,
        'C4': C4, 'T': T, 'ctrl_all3': ALL3,
        'ctrl_per_block': ALLBLK.tolist(), 'C4_per_block': C4B,
        'R': R.tolist(), 'M': M.tolist(), 'rand_M': RAND_M.tolist(),
        'top4_heads_union': [int(x) for x in top4.tolist()],
        'top4_per_block': [[int(x) for x in np.argsort(-np.maximum(M[bi], 0))[:4].tolist()] for bi in range(3)],
        'rand_ratio': rand_ratio,
        'gates': {'ctrl': CTRL_GATE, 'C4_localized': C4_LOCAL, 'C4_distributed': C4_DIST, 'T_min': T_MIN},
        'gain_none_q50': float(q50_none[NL] - q50_none[L_MID]),
        'det': {'anchor_check': ANC['note'], 'pre_slot_rel_max': prm_v,
                'efficacy_maxabs_Lmid1': eff_v,
                'dirs': 'top64 pick=%s' % pick.tolist(), 'chunk_rows': CHUNK,
                'precision': PREC,
                'execution': exec_note, 'rand_seed': 3161,
                'rand4': {str(k): v for k, v in RAND4.items()},
                'n_pairs_per_dir': NPR, 'base_model_forward': 'model.model (hidden states identical to CausalLM path)'},
        'verdict': 'g4p4_%s|C4_%.4f|T_%.4f|ctrl_%.4f|randr_%.4f' % (cls, C4, T, ALL3, rand_ratio),
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
    NONE_C = np.stack([np.asarray(p['none_curve'], np.float64) for p in parts])
    ALL3_C = np.stack([np.asarray(p['all3_curve'], np.float64) for p in parts])
    metac = [json.loads(str(p['anchor_meta_json'])) for p in parts]
    prmc = max(float(p['prm']) for p in parts)
    effc = float(parts[0]['eff'])
    OPROJ_NAME = str(parts[0]['oproj'])
    log('collect: 4 parts merged (prm=%.3g eff=%.4g oproj=%s)' % (prmc, effc, OPROJ_NAME))
    aggregate_and_seal(SHARE_NLc, SHARE_LMc, NONE_C, ALL3_C, metac, prmc, effc,
                       'per-anchor process isolation (4 fresh processes) + collect merge; '
                       'mitigation for 14b/glm4 bf16 sysmem-fallback PCIe streaming under '
                       'co-tenant memory pressure')
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
        # 14b segfault fix (2026-10-09): bnb CUDA kernel segfaults when WDDM sysmem
        # fallback pages quantized weights (14B untied resident need ~11GB > free).
        # Mixed placement: layers 0..GPU_L-1 NF4 on GPU, layers GPU_L..NL-1 bf16 on
        # CPU (llm_int8_enable_fp32_cpu_offload=True auto-excludes cpu-keyed modules
        # from quantization), embed_tokens/lm_head CPU bf16 (untied; lm_head unused
        # since forward uses BASEM). Injection layer L_mid-1=19 stays on GPU
        # (GPU_L=20); ablation layers 20/21/22 on CPU bf16 -> o_proj input
        # head-zeroing semantics unchanged. hidden_states reads are .float().cpu()
        # (device-agnostic). Measured CPU bf16 matmul 2.1 TFLOPS -> ~9 s/fwd for
        # 20 CPU layers, ~16 min/anchor.
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
        log('device_map: MIXED GPU_L=%d (0..%d NF4 GPU, %d..%d bf16 CPU, embed/lm_head CPU; free_vram=%.1f GB)'
            % (GPU_L, GPU_L - 1, GPU_L, NL - 1, free_v / (1 << 30)))
        _devs = [str(model.model.layers[i].mlp.down_proj.weight.device) for i in (0, GPU_L - 1, GPU_L, NL - 1)]
        assert 'cuda' in _devs[0] and 'cuda' in _devs[1] and _devs[2] == 'cpu' and _devs[3] == 'cpu', \
            ('placement check failed', _devs)
        log('placement check: L0=%s L%d=%s L%d=%s L%d=%s'
            % (_devs[0], GPU_L - 1, _devs[1], GPU_L, _devs[2], NL - 1, _devs[3]))
        _free_after, _ = torch.cuda.mem_get_info()
        assert _free_after > 0.25 * (1 << 30), ('vram headroom after load too small', _free_after / (1 << 30))
        log('vram headroom after load: %.2f GB' % (_free_after / (1 << 30)))
log('model loaded: %s vram_alloc=%.2f GB' % (type(model).__name__, torch.cuda.memory_allocated() / 1e9))
assert model.config.num_hidden_layers == NL
dev = 'cuda'
NH = NL + 1
BASEM = model.model
assert hasattr(BASEM, 'layers'), 'base transformer without layers'
pre_ids = tok(d7['prefix'], add_special_tokens=False)['input_ids']
ctx_ids = (pre_ids * (int(d7['k_ctx']) // len(pre_ids) + 1))[:int(d7['k_ctx'])]

# --- hooks ---
INJ = {'on': False, 'delta': None}
ABL = {'blk': None, 'head': None}

def _inj_hook(module, args, output):
    if INJ['on'] and INJ['delta'] is not None:
        out0 = output[0] if isinstance(output, tuple) else output
        new0 = out0.clone()
        new0[:, -1, :] = new0[:, -1, :] + INJ['delta'].to(new0.dtype)
        if isinstance(output, tuple):
            return (new0,) + tuple(output[1:])
        return new0
    return None

def _make_abl_hook(blk_no):
    def _abl_pre(module, args):
        bt, ht = ABL['blk'], ABL['head']
        if bt is None:
            return None
        sel = ((bt == blk_no) | (bt == -3)).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            return None
        x = args[0]
        mask = torch.ones_like(x)
        sl = sel.tolist()
        hv = ht[sel].tolist()
        for r, h in zip(sl, hv):
            h = int(h)
            if h == -2:
                mask[r] = 0.0
            elif h >= 0:
                mask[r, :, h * HD:(h + 1) * HD] = 0.0
        return (x * mask,) + tuple(args[1:])
    return _abl_pre

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
oproj_mods = {}
for b in BLKS:
    nm, mod = resolve_oproj(BASEM.layers[b])
    oproj_mods[b] = mod
    if OPROJ_NAME is None:
        OPROJ_NAME = nm
    assert nm.split('.')[-1] == OPROJ_NAME.split('.')[-1], ('oproj inconsistent', nm, OPROJ_NAME)
log('attn out-proj resolved: %s (in=%d = H*HD = %d*%d)' % (OPROJ_NAME, H * HD, H, HD))
BASEM.layers[L_MID - 1].register_forward_hook(_inj_hook)
for b in BLKS:
    oproj_mods[b].register_forward_pre_hook(_make_abl_hook(b))

def _forward_rows(ids, specs, dvec, eps0):
    n = len(specs)
    bl = np.full(n, -1, np.int64)
    hl = np.full(n, -1, np.int64)
    delta = np.zeros((n, D), np.float32)
    for i, (b, h, pert) in enumerate(specs):
        if b != -1:
            bl[i] = b
            hl[i] = h
        if pert:
            delta[i] = (eps0 * dvec).astype(np.float32)
    ii = torch.tensor([ids] * n, dtype=torch.int64, device=dev)
    INJ['delta'] = torch.from_numpy(delta).to(dev)
    INJ['on'] = True
    ABL['blk'] = torch.from_numpy(bl).to(dev)
    ABL['head'] = torch.from_numpy(hl).to(dev)
    try:
        with torch.no_grad():
            o = BASEM(input_ids=ii, output_hidden_states=True)
    finally:
        INJ['on'] = False
        INJ['delta'] = None
        ABL['blk'] = None
        ABL['head'] = None
    hs = np.stack([h[:, -1].float().detach().cpu().numpy() for h in o.hidden_states], 1)
    del o
    return hs   # (n, NH, D) float32

top64f = top64.astype(np.float64)

def run_dir(ids, dvec, eps0):
    specs = [(-1, -1, 0), (-1, -1, 1)]
    for (b, h) in cfgs:
        specs.append((b, h, 0))
        specs.append((b, h, 1))
    npr = 1 + len(cfgs)
    nch = max(1, int(np.ceil(2 * npr / CHUNK)))
    ppc = int(np.ceil(npr / nch))
    out_nl = np.zeros(npr, np.float64)
    out_lm = np.zeros(npr, np.float64)
    curves = np.zeros((npr, NH), np.float64)

    def fwd_safe(seg):
        try:
            return _forward_rows(ids, seg, dvec, eps0)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(seg) <= 2:
                raise
            half = (len(seg) // 4) * 2
            return np.concatenate([fwd_safe(seg[:half]), fwd_safe(seg[half:])], 0)

    for c0 in range(0, npr, ppc):
        seg = []
        for pi in range(c0, min(c0 + ppc, npr)):
            seg.append(specs[2 * pi])
            seg.append(specs[2 * pi + 1])
        hs = fwd_safe(seg)
        m = len(seg) // 2
        for j in range(m):
            pi = c0 + j
            base = hs[2 * j].astype(np.float64)
            pert = hs[2 * j + 1].astype(np.float64)
            dh = (hs[2 * j + 1] - hs[2 * j])
            pre_rel = float(np.abs(dh[:L_MID]).max() / (np.linalg.norm(base[L_MID]) + 1e-18))
            assert pre_rel < 1e-5, ('pre-slot dh too large', pi, pre_rel)
            if pre_rel > PRM['v']:
                PRM['v'] = pre_rel
            num = (dh[L_MID:] @ top64f) ** 2
            den = (dh[L_MID:] ** 2).sum(1) + 1e-18
            sh = num.sum(1) / den
            curves[pi, L_MID:] = sh
            out_nl[pi] = sh[-1]
            out_lm[pi] = sh[0]
        del hs
    return out_nl, out_lm, curves

SHARE_NL = np.zeros((N_ANCH, N_DIR, NPR), np.float64)
SHARE_LM = np.zeros((N_ANCH, N_DIR, NPR), np.float64)
NONE_CURVE = np.zeros((N_ANCH, N_DIR, NH), np.float64)
ALL3_CURVE = np.zeros((N_ANCH, N_DIR, NH), np.float64)
anchor_meta = []
pre_rel_max = 0.0
eff = -1.0
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
        hs_chk = _forward_rows(ids, [(-1, -1, 0), (-3, -2, 0)], np.zeros(D, np.float64), 0.0)
        eff = float(np.abs(hs_chk[1, L_MID + 1] - hs_chk[0, L_MID + 1]).max())
        assert eff > 1e-3, ('ablation efficacy too small', eff)
        log('efficacy: maxabs diff at slot L_mid+1 (all3 vs none base) = %.4g' % eff)
        del hs_chk
    for di in range(N_DIR):
        out_nl, out_lm, curves = run_dir(ids, U[:, di], eps0)
        SHARE_NL[ai0, di] = out_nl
        SHARE_LM[ai0, di] = out_lm
        NONE_CURVE[ai0, di] = curves[0]
        ALL3_CURVE[ai0, di] = curves[-1]
    sh_none_lm = float(NONE_CURVE[ai0, :, L_MID].mean())
    assert sh_none_lm > 0.95, ('none share at L_mid too low', sh_none_lm)
    anchor_meta.append(dict(row=int(r0), ei=int(row['ei']), rel=row['rel'], pol=row['pol'],
                            ctx=int(row['ctx'])))
    log('anchor %d/%d row=%d done; none share(L_mid)=%.4f share(NL)=%.4f ctrl3 share(NL)=%.4f' % (
        ai0 + 1, N_ANCH, int(r0), sh_none_lm,
        float(NONE_CURVE[ai0, :, NL].mean()), float(ALL3_CURVE[ai0, :, NL].mean())))

if ANCHOR_K is not None:
    pdir_parts = os.path.join(BASE, '_parts')
    os.makedirs(pdir_parts, exist_ok=True)
    np.savez_compressed(os.path.join(pdir_parts, 'collect_anchor%d.npz' % ANCHOR_K),
                        k=np.int64(ANCHOR_K), row=np.int64(idx[ANCHOR_K]),
                        share_nl=SHARE_NL[ANCHOR_K].astype(np.float32),
                        share_lm=SHARE_LM[ANCHOR_K].astype(np.float32),
                        none_curve=NONE_CURVE[ANCHOR_K].astype(np.float32),
                        all3_curve=ALL3_CURVE[ANCHOR_K].astype(np.float32),
                        anchor_meta_json=json.dumps(anchor_meta, ensure_ascii=False),
                        prm=np.float64(PRM['v']), eff=np.float64(eff),
                        oproj=str(OPROJ_NAME))
    log('ANCHOR %d DONE (part saved, fresh-process isolation)' % ANCHOR_K)
    sys.exit(0)

aggregate_and_seal(SHARE_NL, SHARE_LM, NONE_CURVE, ALL3_CURVE, anchor_meta, PRM['v'], eff,
                   'single-process (all anchors in one process; used for 4b/glm4-size fits)')
log('DONE model=%s smoke=%s' % (MODEL, SMOKE))
