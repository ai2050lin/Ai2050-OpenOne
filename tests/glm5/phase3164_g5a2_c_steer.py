# -*- coding: utf-8 -*-
"""Phase 3164 轴(a)：G5-A2 图谱缺口② —— C_steer 跨模型同口径复测。

预注册：AGI_GPT5_MEMO L15175-15178（3163 closeout 冻结）。
  (a) C_steer 跨模型：Q06 承重轴装置（qwen3-4b L29 WR 主 PC 同构移植）到 14b/glm4
      对应层 + x 端口替换，held-out cells x 10 steer 配置同口径；
      门 = steered 成功率 Wilson 上界（Q06=1.0% 对照）+ collateral（Q06 frac0=0.933 对照）。
4b 不重跑：引用 Q06 已封存 result（tests/deepseek/result/q06_result.json，Phase 40）。

同构移植定义（execution.json 冻结）：
  R1 层移植：LAY' = round(29/36 * NL')；NL=40 -> L32（两模型同值）。
  R2 精度：14b = NF4 pre-quantized checkpoint（Qwen3-14B-bnb-nf4，3161 补记）；
      glm4 = NF4 现场量化（3161/3163 同路径）；Q06 原版 bf16 -> 精度差异登记
      known deviation（NF4 模型上轴抽取与读出自洽，同模型同精度）。
  R3 面板逐字一致（panel_sha8=be17ef8a 断言）；轴抽取同 seed7 train fold /
      ridge(1e-3) / SVD 第一右奇异向量 / rand 种子 20261007 / 探针 13 同规则；
      t 规则 = Q06 annex v2（t = s + sgn*alpha*sigma'，sigma' 为本模型 sigma）。
  R4 执行：per-anchor 进程隔离（4 fresh processes，3161 理由）+ collect merge；
      anchor0 计算轴（591 forwards）落盘 axis npz，anchor1-3 读回。
判据（跨模型，冻结于 execution.json）：
  cls = zero_like_q06 (C_main<=0.02) / weak (<=0.10) / substantial (>0.10)
  collat_clean = frac0 >= 0.80（Q06 对照 0.933；差分口径）
  装置门：F1 identity 逐位==0、hook_hits==1、G2 computability、G2b maxd>0、
         F6 |cos(v1,vr)|<0.2、F7 sigma>0、CLS_TOK 不碰撞。
"""
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
PHASE = 3164
ROOT = r'D:\AI2050\Ai2050-OpenOne'
BASE = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3164', 'g5a2_c_steer')
TEMP = os.path.join(ROOT, 'tests', 'gpt5_temp')
MODEL = os.environ.get('P3164_MODEL', 'qwen3-14b')
SMOKE = os.environ.get('P3164_SMOKE', '0') == '1'
COLLECT = os.environ.get('P3164_COLLECT', '0') == '1'
SUMMARY = (MODEL == 'summary')
ANCHOR_K = int(os.environ.get('P3164_ANCHOR', '-1'))
N_ANCH = 4
assert MODEL in ('qwen3-14b', 'glm4', 'summary'), MODEL
LOGP = os.path.join(TEMP, 'p3164_%s%s.log' % (MODEL, '_smoke' if SMOKE else ''))
LOG = []

def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    LOG.append(ln)
    with open(LOGP, 'a', encoding='utf-8') as f:
        f.write(ln + '\n')
    try:
        print(ln, flush=True)
    except Exception:
        pass

def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]

# ---------------- 面板冻结（与 Q06 逐字一致） ----------------
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
TPL_P0 = {0: '{e}是一种',
          1: '{e}属于',
          2: '{e}，一种常见的'}
SEEDS_S1 = [7, 8, 9]
FRAC_S1 = 0.2
ENTS = [e for cl in CLASSES for e in ENT[cl]]
CLS_OF = [CLASSES.index(cl) for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS); NC = len(CLASSES)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS); NT = len(TPL)
assert (NE, NC, NP_, NT) == (41, 6, 246, 3), (NE, NC, NP_, NT)
ALLP = set(PAIRS)

def split_s1(seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(NP_)
    n_test = int(round(FRAC_S1 * NP_))
    test = set([PAIRS[j] for j in idx[:n_test]])
    return ALLP - test, test

def rows_of(pair_set):
    return [t * NP_ + pi for t in range(NT)
            for pi, p in enumerate(PAIRS) if p in pair_set]

PANEL_SHA = hashlib.sha256(
    json.dumps([CLASSES, ENTS, [TPL[k] for k in sorted(TPL)], SEEDS_S1, FRAC_S1],
               ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
assert PANEL_SHA == 'be17ef8a', 'PANEL DRIFT: %s' % PANEL_SHA

Q06_PARAMS = dict(layer_rule='LAY = round(29/36 * NL)', alphas=[0.05, 0.10, 0.15, 0.25, 0.50],
                  sgns=[+1, -1], axis_seed=7, rand_seed=20261007, n_probe=13, ridge_lam=1e-3)
ALPHAS = Q06_PARAMS['alphas']; SGNS = Q06_PARAMS['sgns']
AXIS_SEED = 7; RAND_SEED = 20261007; N_PROBE = 13; LAM = 1e-3

MDIR_MAP = {'qwen3-14b': 'Qwen3-14B-bnb-nf4', 'glm4': 'glm4-9b-chat-hf'}
PREC = {'qwen3-14b': 'nf4-pre', 'glm4': 'nf4'}

# ---------------- design freeze ----------------
def freeze_design():
    phase_name = 'g5a2a_c_steer_cross_model'
    design = {
        'phase': PHASE, 'name': phase_name,
        'prereg': 'AGI_GPT5_MEMO L15175-15178 (3163 closeout); axis (a) C_steer cross-model',
        'base_ref': {'phase': 'Q06/Phase40 (deepseek line)', 'result': 'tests/deepseek/result/q06_result.json',
                     'C_steer_main': 0.0, 'wilson_upper': 0.010, 'collat_frac0': 0.933,
                     'model': 'qwen3-4b', 'prec': 'bf16', 'layer': 29, 'NL': 36},
        'panel_sha8': PANEL_SHA,
        'isomorphic_map': {
            'layer_rule': Q06_PARAMS['layer_rule'],
            'precision': {'qwen3-14b': 'NF4 pre-quantized (Qwen3-14B-bnb-nf4)',
                          'glm4': 'NF4 on-the-fly (3161/3163 path)'},
            'known_deviation': 'Q06 was bf16; 14b/glm4 run NF4 with on-model axis extraction '
                               '(axis and readout self-consistent at same precision)'},
        'operator': 'readout-substitution (port replacement) h <- h + (t - h.v)v, '
                    't = s + sgn*alpha*sigma_model (annex v2 t-rule), no direction subtraction',
        'arms': ['base', 'identity', 'steer(sgn x alpha, 10)', 'rand(sgn x alpha, 10)'],
        'cells': 'S1 held-out (seeds 7/8/9, frac 0.2) x 3 templates = 441; smoke = seed7 rows[:12]',
        'gates': {
            'cls': 'zero_like_q06 (C_main<=0.02) / weak (<=0.10) / substantial (>0.10)',
            'collat_clean': 'frac0 >= 0.80 (Q06 ref 0.933, differential collateral)',
            'device': 'F1 identity bitwise==0; hook_hits==1; G2 computability; G2b maxd>0; '
                      'F6 |cos(v1,vr)|<0.2; F7 sigma>0; CLS_TOK unique'},
        'execution': 'per-anchor process isolation (4 fresh processes) + collect merge '
                     '(3161 rationale: co-tenant memory pressure); anchor0 extracts axis',
        'frozen_before': 'any observation',
    }
    blob = json.dumps(design, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    sha = hashlib.sha256(blob).hexdigest()
    os.makedirs(BASE, exist_ok=True)
    exe_p = os.path.join(BASE, 'execution.json')
    if os.path.exists(exe_p):
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha'] == sha, 'DESIGN DRIFT: delete execution.json+result.json after script change'
        log('execution.json match (sha %s)' % sha[:8])
    else:
        json.dump({'phase': PHASE, 'name': phase_name, 'design_sha': sha,
                   'design': design, 'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('execution.json FROZEN (sha %s)' % sha[:8])
    return sha

DESIGN_SHA = freeze_design()

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

def pool_success(RES, cells_keys, kind, sg, al):
    key = '%s|%+d|%.2f' % (kind, sg, al)
    per_seed = {}
    for s_ in SEEDS_S1:
        cells_s = [k for k in cells_keys if k.startswith('%d|' % s_)]
        elig = [k for k in cells_s if RES[k]['eligible']]
        if not elig:
            continue
        hits = sum(1 for k in elig
                   if RES[k][key]['argmax'] == RES[k]['true_class'] and RES[k][key]['collat'] == 0)
        per_seed[str(s_)] = dict(num=hits, den=len(elig), rate=hits / len(elig))
    tot_n = sum(v['den'] for v in per_seed.values())
    tot_h = sum(v['num'] for v in per_seed.values())
    return dict(per_seed=per_seed, pool_rate=(tot_h / tot_n if tot_n else None),
                pool_num=tot_h, pool_den=tot_n)

def wilson(h, n, z=1.96):
    if not n:
        return None
    p = h / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - hw), float(c + hw)]

def aggregate_and_seal(RES, exec_note, axis_meta):
    cells_keys = sorted(RES.keys())
    STEER_CFG = [('steer', sg, a) for sg in SGNS for a in ALPHAS] + \
                [('rand', sg, a) for sg in SGNS for a in ALPHAS]
    steer_curves = {('steer|%+d|%.2f' % (sg, al)): pool_success(RES, cells_keys, 'steer', sg, al)
                    for sg in SGNS for al in ALPHAS}
    rand_curves = {('rand|%+d|%.2f' % (sg, al)): pool_success(RES, cells_keys, 'rand', sg, al)
                   for sg in SGNS for al in ALPHAS}
    best_steer = max(steer_curves.items(), key=lambda kv: (kv[1]['pool_rate'] or 0))
    best_rand = max(rand_curves.items(), key=lambda kv: (kv[1]['pool_rate'] or 0))
    C_MAIN = best_steer[1]['pool_rate']
    C_RAND = best_rand[1]['pool_rate']
    F1 = max(max(r['identity_maxd'], r['identity_maxd_concat']) for r in RES.values())
    F1_ok = (F1 == 0.0)
    hook_ok = all(r[k]['hook_hits'] == 1 for r in RES.values()
                  for k in r if isinstance(r[k], dict) and 'hook_hits' in r[k])
    all_collat = [r[k]['collat'] for r in RES.values() for k in r
                  if isinstance(r[k], dict) and 'collat' in r[k]]
    collat_frac0 = float(np.mean([c == 0 for c in all_collat])) if all_collat else None
    err_base_all = [r['err_base'] for r in RES.values()]
    n_elig = sum(1 for r in RES.values() if r['eligible'])
    maxd_all = [r[k]['maxd'] for r in RES.values() for k in r
                if isinstance(r[k], dict) and 'maxd' in r[k] and k.startswith('steer')]
    argmax_moved = sum(1 for r in RES.values() for k in r
                       if isinstance(r[k], dict) and 'argmax' in r[k] and k.startswith('steer')
                       and r[k]['argmax'] != r['beh_argmax_base'])
    argmax_tot = len(maxd_all)
    n_per_seed = {str(s_): sum(1 for k in cells_keys if k.startswith('%d|' % s_)) for s_ in SEEDS_S1}
    n_eff = max((v['den'] for v in best_steer[1]['per_seed'].values()), default=0)
    p0 = C_MAIN if C_MAIN is not None else 0.0
    MDE80 = float(2.80 * np.sqrt(max(p0 * (1 - p0), 1e-9) / max(n_eff, 1)))
    # ---- 判决（预注册门）----
    if C_MAIN is None:
        cls = 'no_eligible_cells'
    elif C_MAIN <= 0.02:
        cls = 'zero_like_q06'
    elif C_MAIN <= 0.10:
        cls = 'weak'
    else:
        cls = 'substantial'
    collat_clean = (collat_frac0 is not None and collat_frac0 >= 0.80)
    G1 = F1_ok
    G2 = all(v['pool_rate'] is not None or v['pool_den'] == 0 for v in steer_curves.values())
    G2b = (min(maxd_all) > 0.0) if maxd_all else False
    G3 = all(isinstance(c, int) and -N_PROBE <= c <= N_PROBE for c in all_collat) and len(all_collat) > 0
    G4 = (axis_meta['sigma'] > 0) and (axis_meta['cos_vr'] < 0.2)
    dev_ok = all([G1, hook_ok, G2, G3, G4])
    suffix = '' if dev_ok else '_device_gate_fail'
    result = {
        'phase': PHASE, 'name': 'g5a2a_c_steer_cross_model', 'mode': MODEL,
        'smoke': bool(SMOKE), 'prec': PREC[MODEL],
        'design_sha': DESIGN_SHA,
        'model_meta': axis_meta,
        'cells': dict(per_seed=n_per_seed, total=len(RES), eligible=n_elig,
                      base_argmax_eq_true=float(np.mean([not r['eligible'] for r in RES.values()]))),
        'floors': dict(F1_identity_maxd=F1, F1_ok=F1_ok, hook_hits_all_1=hook_ok,
                       F2_panel=PANEL_SHA == 'be17ef8a', F4_probes=N_PROBE,
                       F5_unit=True, F6_cos_lt02=axis_meta['cos_vr'] < 0.2,
                       F7_sigma_pos=axis_meta['sigma'] > 0),
        'steer_curves': steer_curves, 'rand_curves': rand_curves,
        'sensitivity': dict(argmax_moved=argmax_moved, argmax_total=argmax_tot,
                            maxd_min=float(min(maxd_all)) if maxd_all else None,
                            maxd_max=float(max(maxd_all)) if maxd_all else None),
        'C_steer_main': dict(rule='max over 10 steer configs pool_rate (eligible only)',
                             config=best_steer[0], value=C_MAIN,
                             wilson=wilson(best_steer[1]['pool_num'], best_steer[1]['pool_den']),
                             rand_config=best_rand[0], rand_value=C_RAND,
                             spec_diff=(None if (C_MAIN is None or C_RAND is None) else C_MAIN - C_RAND),
                             MDE80=MDE80),
        'collateral': dict(mean=float(np.mean(all_collat)) if all_collat else None,
                           max=int(np.max(all_collat)) if all_collat else None,
                           frac_zero=collat_frac0, clean_ge_080=collat_clean,
                           err_base_mean=float(np.mean(err_base_all)) if err_base_all else None),
        'gates': dict(G1_identity=G1, G2_computability=G2, G2b_sensitivity=[G2b, (max(maxd_all) >= 0.05) if maxd_all else False],
                      G3_collat_int=G3, G4_nondegen=G4),
        'cls': cls,
        'verdict': 'g5a2a_%s|C_%s|rand_%s|frac0_%.4f%s' % (
            cls, C_MAIN, C_RAND, collat_frac0 if collat_frac0 is not None else -1.0, suffix),
        'runtime_s': round(time.time() - T0, 1),
        'execution': exec_note,
    }
    if SMOKE:
        seal_result(result, 'smoke_result_%s.json' % MODEL)
        with open(os.path.join(BASE, 'smoke_cells_detail_%s.json' % MODEL), 'w', encoding='utf-8') as f:
            json.dump(RES, f, ensure_ascii=False, indent=1)
    else:
        seal_result(result, 'result_%s.json' % MODEL)
        with open(os.path.join(BASE, 'cells_detail_%s.json' % MODEL), 'w', encoding='utf-8') as f:
            json.dump(RES, f, ensure_ascii=False, indent=1)
    return result

# ================= summary 模式（跨模型汇总，零 GPU） =================
if SUMMARY:
    q06_p = os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q06_result.json')
    q06 = json.load(open(q06_p, encoding='utf-8'))
    models = {}
    models['qwen3-4b'] = dict(
        ref='Q06 Phase40 (deepseek line, q06_result.json)', prec='bf16', layer=29, NL=36,
        C_main=q06['C_steer_main']['value'], rand=q06['C_steer_main']['rand_value'],
        frac0=q06['collateral']['frac_zero'], res_sha8=q06['res_sha8'])
    cls_vals = {}
    for m in ('qwen3-14b', 'glm4'):
        rp = os.path.join(BASE, 'result_%s.json' % m)
        r = json.load(open(rp, encoding='utf-8'))
        models[m] = dict(ref='3164 axis(a) this phase', prec=r['prec'], layer=r['model_meta']['layer'],
                         NL=r['model_meta']['NL'], C_main=r['C_steer_main']['value'],
                         rand=r['C_steer_main']['rand_value'], frac0=r['collateral']['frac_zero'],
                         cls=r['cls'], res_sha8=r['res_sha8'], seal_sha8=r['seal_sha8'],
                         device_ok=r['floors']['F1_ok'] and r['floors']['hook_hits_all_1'])
        cls_vals[m] = r['cls']
    ref_cls = 'zero_like_q06' if (q06['C_steer_main']['value'] or 0) <= 0.02 else 'ref_nonzero'
    agree = all(v == cls_vals['qwen3-14b'] == cls_vals['glm4'] for v in cls_vals.values())
    summary = {
        'phase': PHASE, 'name': 'g5a2a_summary', 'axis': '(a) C_steer cross-model',
        'models': models,
        'ref_cls_4b': ref_cls,
        'class_agreement_14b_glm4': agree,
        'panel_sha8': PANEL_SHA, 'design_sha': DESIGN_SHA,
        'gates_note': 'gate = cls consistency across models + collateral clean flag per model; '
                      'zero_like_q06 across all three closes graph gap-2 axis (a)',
        'verdict': 'g5a2a_summary|agree_%s|cls14b_%s|clsglm4_%s' % (
            agree, cls_vals['qwen3-14b'], cls_vals['glm4']),
        'runtime_s': round(time.time() - T0, 1),
    }
    seal_result(summary, 'result_summary.json')
    log('SUMMARY DONE')
    sys.exit(0)

# ================= collect 模式 =================
if COLLECT:
    parts = []
    for k in range(N_ANCH):
        pp = os.path.join(BASE, '_parts_%s' % MODEL, 'anchor%d.json' % k)
        parts.append(json.load(open(pp, encoding='utf-8')))
    RES = {}
    for p in parts:
        for ck, row in p['rows'].items():
            assert ck not in RES, ('cell dup', ck)
            RES[ck] = row
    metas = [p['axis_meta'] for p in parts]
    for m_ in metas[1:]:
        for kk in ('layer', 'sigma', 'cos_vr'):
            assert abs(float(m_[kk]) - float(metas[0][kk])) < 1e-9, ('axis meta mismatch', kk)
    log('collect: %d parts -> %d cells merged' % (len(parts), len(RES)))
    aggregate_and_seal(RES, 'per-anchor process isolation (4 fresh processes) + collect merge',
                       metas[0])
    log('COLLECT DONE model=%s' % MODEL)
    sys.exit(0)

# ================= torch 路径 =================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
MDIR = os.path.join(ROOT, 'models', 'hf', MDIR_MAP[MODEL])
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
log('model load begin: %s prec=%s' % (MDIR, PREC[MODEL]))
if PREC[MODEL] == 'nf4-pre':
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, dtype=torch.bfloat16, trust_remote_code=True).eval()
else:
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, quantization_config=bnb, device_map={'': 0}, trust_remote_code=True).eval()
NL = int(model.config.num_hidden_layers)
LAY = int(round(29.0 / 36.0 * NL))
assert NL == 40 and LAY == 32 and LAY < NL - 1, (NL, LAY)
log('model loaded NL=%d LAY=%d device_mem=%.1f GB' % (
    NL, LAY, torch.cuda.memory_allocated() / 2**30))

def ids_of(text):
    return tok(text, add_special_tokens=False)['input_ids']

CLS_TOK = [ids_of(c)[0] for c in CLASSES]
assert len(set(CLS_TOK)) == NC, 'F1b class first tokens collide: %s' % CLS_TOK
log('CLS_TOK=%s' % CLS_TOK)
CLS_T = torch.tensor(CLS_TOK, device='cuda')

# ---------------- LAY hook（逐字同 Q06 语义） ----------------
STATE = {'mode': 'off', 'pos': 0, 'dt': 0.0, 'v': None, 'hits': 0}

def lay_hook(module, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if STATE['mode'] == 'off':
        return out
    pos = STATE['pos']
    hf = h[0, pos].float()
    if STATE['mode'] == 'capture':
        if STATE['cap'] is not None:
            STATE['cap'].append(hf.detach().cpu().numpy())
        return out
    v = STATE['v']
    if STATE['mode'] == 'identity':
        hf2 = hf
    else:
        s = float(hf @ v)
        hf2 = hf + (s + STATE['dt'] - s) * v
    h2 = h.clone()
    h2[0, pos] = hf2.to(h.dtype)
    STATE['hits'] += 1
    return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

model.model.layers[LAY].register_forward_hook(lay_hook)

def six_cls_logits(ids, steer=None):
    STATE['mode'] = 'off'
    if steer is not None:
        mode, pos, dt, v = steer
        STATE['mode'] = mode; STATE['pos'] = pos; STATE['dt'] = dt; STATE['v'] = v; STATE['hits'] = 0
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device='cuda'))
    lg = out.logits[0, -1 if steer is None else steer[1], :].float()
    cl = lg[CLS_T].detach().cpu().numpy()
    STATE['mode'] = 'off'
    return cl

# ---------------- 轴抽取（anchor0 / SMOKE 执行；其余读回） ----------------
axis_npz_p = os.path.join(BASE, '_parts_%s' % MODEL, 'axis.npz')
axis_json_p = os.path.join(BASE, '_parts_%s' % MODEL, 'axis.json')
IS_EXTRACTOR = SMOKE or (ANCHOR_K == 0)

train7, test7 = split_s1(AXIS_SEED)
tr_rows = rows_of(train7)
vr_expected = None

if IS_EXTRACTOR:
    os.makedirs(os.path.dirname(axis_npz_p), exist_ok=True)
    log('v1 axis extract: seed%d train rows=%d layer=%d' % (AXIS_SEED, len(tr_rows), LAY))
    CAP = []
    STATE['mode'] = 'capture'; STATE['cap'] = CAP
    with torch.no_grad():
        for r in tr_rows:
            t_, pi_ = r // NP_, r % NP_
            ids = ids_of(TPL_P0[t_].format(e=ENTS[PAIRS[pi_][0]]))
            STATE['pos'] = len(ids) - 1
            model(input_ids=torch.tensor([ids], device='cuda'))
    STATE['mode'] = 'off'; STATE['cap'] = None
    H = np.stack(CAP).astype(np.float64)
    assert H.shape == (len(tr_rows), model.config.hidden_size), H.shape
    log('H captured: %s' % (H.shape,))
    ncol = NE + NC + NT + 1
    def rowvec(t_, pi_):
        i, c = PAIRS[pi_]
        v = np.zeros(ncol, np.float64)
        v[i] = 1.0; v[NE + c] = 1.0; v[NE + NC + t_] = 1.0; v[-1] = 1.0
        return v
    X = np.stack([rowvec(r // NP_, r % NP_) for r in tr_rows])
    Beta = np.linalg.solve(X.T @ X + LAM * np.eye(ncol), X.T @ H)
    Rres = H - X @ Beta
    U_, S_, Vt = np.linalg.svd(Rres, full_matrices=False)
    v1 = Vt[0].copy()
    if float(np.sum(Rres @ v1)) < 0:
        v1 = -v1
    sv_share = float(S_[0] ** 2 / np.sum(S_ ** 2))
    s_all = H @ v1
    MU, SIG = float(s_all.mean()), float(s_all.std())
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-9, 'v1 not unit'
    assert SIG > 0, 'F7 sigma degenerate'
    rng = np.random.default_rng(RAND_SEED)
    vr = rng.normal(size=(model.config.hidden_size,))
    vr = vr / np.linalg.norm(vr)
    s_r = H @ vr
    MU_R, SIG_R = float(s_r.mean()), float(s_r.std())
    cos_vr = float(abs(np.dot(v1, vr)))
    assert cos_vr < 0.2, 'F6 rand dir too aligned: %.3f' % cos_vr
    log('v1 axis: sv_share=%.4f mu=%.4f sigma=%.4f |cos(v1,vr)|=%.4f' % (
        sv_share, MU, SIG, cos_vr))
    # 探针 13（同 Q06 规则；rng 在 vr 之后继续 -> 与 Q06 同序）
    probe_pool = [p for p in PAIRS if p in train7]
    pidx = rng.choice(len(probe_pool), N_PROBE, replace=False)
    PROBES = [probe_pool[int(j)] for j in pidx]
    np.savez_compressed(axis_npz_p, v1=v1.astype(np.float64), vr=vr.astype(np.float64),
                        mu=np.float64(MU), sigma=np.float64(SIG),
                        mu_r=np.float64(MU_R), sigma_r=np.float64(SIG_R),
                        sv_share=np.float64(sv_share))
    json.dump({'probes': [[int(a), int(b)] for a, b in PROBES],
               'layer': LAY, 'NL': NL, 'H': int(model.config.hidden_size),
               'sv_share': sv_share, 'mu': MU, 'sigma': SIG,
               'mu_r': MU_R, 'sigma_r': SIG_R, 'cos_vr': cos_vr,
               'train_rows': len(tr_rows), 'panel_sha8': PANEL_SHA},
              open(axis_json_p, 'w', encoding='utf-8'), indent=1)
    log('axis saved -> %s' % axis_npz_p)
else:
    z = np.load(axis_npz_p)
    v1 = z['v1']; vr = z['vr']
    MU, SIG = float(z['mu']), float(z['sigma'])
    MU_R, SIG_R = float(z['mu_r']), float(z['sigma_r'])
    axj = json.load(open(axis_json_p, encoding='utf-8'))
    cos_vr = axj['cos_vr']
    assert axj['layer'] == LAY and axj['panel_sha8'] == PANEL_SHA
    log('axis loaded: sigma=%.4f cos_vr=%.4f' % (SIG, cos_vr))

V1_T = torch.tensor(v1, dtype=torch.float32, device='cuda')
VR_T = torch.tensor(vr, dtype=torch.float32, device='cuda')
AXIS_META = dict(layer=LAY, NL=NL, H=int(model.config.hidden_size), prec=PREC[MODEL],
                 mu=MU, sigma=SIG, mu_r=MU_R, sigma_r=SIG_R, cos_vr=cos_vr)

# 探针段构建（各进程同规则重建；PROBES 从 axis json 读取保证一致）
PROBES = [tuple(p) for p in json.load(open(axis_json_p, encoding='utf-8'))['probes']]
PROBE_SEG = []
for (pi_k, c_k) in PROBES:
    p0k = ids_of(TPL_P0[0].format(e=ENTS[pi_k]))
    PROBE_SEG.append((p0k + ids_of(CLASSES[c_k]) + ids_of('。'), len(p0k) - 1, c_k))

def concat_prompt(t_, pi_):
    main = ids_of(TPL_P0[t_].format(e=ENTS[PAIRS[pi_][0]]))
    ids = list(main)
    probe_pos, probe_c = [], []
    for seg, plen, ck in PROBE_SEG:
        probe_pos.append(len(ids) + plen)
        probe_c.append(ck)
        ids = ids + seg
    return ids, len(main) - 1, probe_pos, probe_c

# ---------------- cells ----------------
STEER_CFG = [('steer', sg, a) for sg in SGNS for a in ALPHAS] + \
            [('rand', sg, a) for sg in SGNS for a in ALPHAS]
if SMOKE:
    rows7 = rows_of(test7)[:12]
    CELLS_ALL = [(AXIS_SEED, r // NP_, r % NP_) for r in rows7]
    CELLS = CELLS_ALL
else:
    CELLS_ALL = []
    for s_ in SEEDS_S1:
        _, te = split_s1(s_)
        for r in rows_of(te):
            CELLS_ALL.append((s_, r // NP_, r % NP_))
    CELLS = CELLS_ALL[ANCHOR_K::N_ANCH]
log('cells: total=%d this_proc=%d arms=22 forwards/cell=44' % (len(CELLS_ALL), len(CELLS)))

RES = {}
t_last = time.time()
for n_, (s_, t_, pi_) in enumerate(CELLS):
    main_ids = ids_of(TPL_P0[t_].format(e=ENTS[PAIRS[pi_][0]]))
    pos_last = len(main_ids) - 1
    true_cls = int(PAIRS[pi_][1])
    c_ids, cm_pos, probe_pos, probe_c = concat_prompt(t_, pi_)
    beh_base = six_cls_logits(main_ids)
    am_base = int(np.argmax(beh_base))
    eligible = (am_base != true_cls)
    col_base = six_cls_logits(c_ids, steer=None)
    STATE['mode'] = 'off'
    with torch.no_grad():
        out_c = model(input_ids=torch.tensor([c_ids], device='cuda'))
    lgc = out_c.logits[0].float()
    probe_base = [lgc[p, CLS_T].detach().cpu().numpy() for p in probe_pos]
    err_base = int(sum(int(np.argmax(pb)) != ck for pb, ck in zip(probe_base, probe_c)))
    row = dict(true_class=true_cls, err_base=err_base,
               beh_base=[float(x) for x in beh_base],
               beh_argmax_base=am_base, eligible=bool(eligible))
    for (kind, sg, al) in STEER_CFG:
        v = V1_T if kind == 'steer' else VR_T
        sig = SIG if kind == 'steer' else SIG_R
        dtv = float(sg) * al * sig
        beh = six_cls_logits(main_ids, steer=('replace', pos_last, dtv, v))
        STATE['mode'] = 'replace'; STATE['pos'] = cm_pos; STATE['dt'] = dtv; STATE['v'] = v; STATE['hits'] = 0
        with torch.no_grad():
            out_ci = model(input_ids=torch.tensor([c_ids], device='cuda'))
        lgc2 = out_ci.logits[0].float()
        probe_after = [lgc2[p, CLS_T].detach().cpu().numpy() for p in probe_pos]
        STATE['mode'] = 'off'
        err_after = int(sum(int(np.argmax(pb)) != ck for pb, ck in zip(probe_after, probe_c)))
        key = '%s|%+d|%.2f' % (kind, sg, al)
        row[key] = dict(beh=[float(x) for x in beh],
                        argmax=int(np.argmax(beh)),
                        maxd=float(np.max(np.abs(beh - beh_base))),
                        collat=err_after - err_base,
                        hook_hits=STATE['hits'])
        assert STATE['hits'] == 1, ('hook hits', key, STATE['hits'])
    beh_id = six_cls_logits(main_ids, steer=('identity', pos_last, 0.0, V1_T))
    row['identity_maxd'] = float(np.max(np.abs(beh_id - beh_base)))
    STATE['mode'] = 'identity'; STATE['pos'] = cm_pos; STATE['v'] = V1_T; STATE['hits'] = 0
    with torch.no_grad():
        out_id = model(input_ids=torch.tensor([c_ids], device='cuda'))
    STATE['mode'] = 'off'
    lgi = out_id.logits[0].float()
    probe_id = [lgi[p, CLS_T].detach().cpu().numpy() for p in probe_pos]
    row['identity_maxd_concat'] = float(max(
        float(np.max(np.abs(p2 - pb))) for p2, pb in zip(probe_id, probe_base)))
    RES['%d|%d|%d' % (s_, t_, pi_)] = row
    if (n_ + 1) % 25 == 0:
        log('cells %d/%d (%.2f s/cell)' % (n_ + 1, len(CELLS), (time.time() - t_last) / (n_ + 1)))

log('forward loop done: %.1f min' % ((time.time() - t_last) / 60.0))

if SMOKE:
    aggregate_and_seal(RES, 'single-process smoke (axis extracted in-process)', AXIS_META)
    log('SMOKE DONE model=%s' % MODEL)
else:
    pdir = os.path.join(BASE, '_parts_%s' % MODEL)
    os.makedirs(pdir, exist_ok=True)
    json.dump({'k': ANCHOR_K, 'rows': RES, 'axis_meta': AXIS_META},
              open(os.path.join(pdir, 'anchor%d.json' % ANCHOR_K), 'w', encoding='utf-8'),
              ensure_ascii=False)
    log('part saved: anchor%d (%d cells)' % (ANCHOR_K, len(RES)))
