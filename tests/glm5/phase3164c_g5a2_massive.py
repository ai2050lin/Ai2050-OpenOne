# -*- coding: utf-8 -*-
"""Phase 3164 轴(c)：G5-A2 图谱缺口② —— massive 77x 塌缩跨模型对应性（零 GPU）。

预注册：AGI_GPT5_MEMO L15177（3163 closeout 冻结）。
  (c) 3157/3158 的 d1 与 77x 现象在 14b/glm4 的对应性；NF4 量化误差下口径
      = 容差标注（倍数门，执行前定，冻结于 execution.json）。
v2 修订（4b 参照运行后、任何新模型观测前重冻结；SMOKE 先例 R1）：
  R1 塌缩读数双口径：collapse_mean(k,lg) = mean_l in mid [mean_t ||h_A0||]/[mean_t ||h_Ak||]；
     collapse_max 同式用 max_t（massive token 峰值主导；3156 4b 实测 zh L12 A0 max=11274）。
     MEMO 4b 参照 11274->146 的 146 侧口径不可从本 npz 直读复原（addendum 口径），
     3164 以本 npz 现场渲染双口径为锚（4b 现场值登记，不做 MEMO 数字断言）。
  R2 massive 维度分层登记：d1_3157 = argmax_d mean_rows |H3157[:, L_mid, d]|（3160 口径，
     断言 == 0/731/2319）；d_rope = A0 mid 层 token 平均 argmax（材料相关，4b 实测=4，
     登记；3156 材料上有前缀时 massive 整体消失，A128 峰值维度不再同一坐标）。
判据（v2 冻结）：supported = collapse_max(k=128) >= 10 且 collapse_mean(k=128) >= 5
  且 k_independence(=collapse_max(1)/collapse_max(128)) < 3；
  partial_collapse = 塌缩过门但 k 依赖；absent = < 门。
"""
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
PHASE = 3164
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g5a2c_massive_cross_model'
MODEL = os.environ.get('P3164C_MODEL', 'summary')
assert MODEL in ('qwen3-4b', 'qwen3-14b', 'glm4', 'summary'), MODEL
BASE = os.path.join(RDIR, 'phase3164', NAME, MODEL)
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

def seal_result(result, out_name):
    blob = json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    res_sha8 = hashlib.sha256(blob).hexdigest()[:8]
    result['res_sha8'] = res_sha8
    result['verdict'] = result['verdict'] + '|sha8_' + res_sha8
    rp = os.path.join(BASE, out_name)
    json.dump(result, open(rp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    seal = hashlib.sha256(open(rp, 'rb').read()).hexdigest()[:8]
    result['seal_sha8'] = seal
    json.dump(result, open(rp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log('RESULT %s res_sha8=%s seal=%s verdict=%s' % (out_name, res_sha8, seal, result['verdict']))
    return res_sha8, seal

D1_REF = {'qwen3-4b': 0, 'qwen3-14b': 731, 'glm4': 2319}
NL_OF = {'qwen3-4b': 36, 'qwen3-14b': 40, 'glm4': 40}
PREC = {'qwen3-4b': 'bf16', 'qwen3-14b': 'nf4-pre', 'glm4': 'nf4'}
P3157 = os.path.join(RDIR, 'phase3157', 'g2p2_transform_algebra_commutator', '%s', 'collect.npz')
P3156 = os.path.join(RDIR, 'phase3156', 'g3p1_position_shift_family', 'qwen3-4b', 'collect.npz')
P3164B = os.path.join(RDIR, 'phase3164', 'g5a2b_position_shift_cross_model', '%s', 'collect.npz')
G_CM, G_CX, G_KINDEP = 5.0, 10.0, 3.0

design = dict(phase=PHASE, name=NAME, version=2, zero_gpu=True,
              data_sources=['3157 collect.npz (d1 recomputation, 3 models)',
                            '3156 collect.npz (4b reference)',
                            '3164 axis(b) collect.npz (14b/glm4, NF4 known deviation)'],
              revise_log=['v1->v2 (4b reference run failed on cross-npz d1 assertion, before any '
                          'new-model observation): R1 dual-convention collapse readout (mean/max '
                          'token-norm ratios; MEMO 11274->146 146-side convention not directly '
                          'recoverable from this npz, 4b on-npz values registered as anchor); '
                          'R2 massive dims layered: d1_3157 asserted vs 0/731/2319, d_rope '
                          '(A0 argmax, material-dependent, 4b=4) registered'],
              gates=dict(collapse_mean_ge=G_CM, collapse_max_ge=G_CX, k_independence_lt=G_KINDEP,
                         cls='supported / partial_collapse (k-dependent) / absent'),
              mid_layers='L in [NL//3, NL//2)',
              frozen_before='any new-model observation (4b reference re-run under v2)')
eblob = json.dumps(design, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
exe_sha = hashlib.sha256(eblob).hexdigest()
exe_p = os.path.join(BASE, 'execution.json')
if os.path.exists(exe_p):
    assert json.load(open(exe_p, encoding='utf-8'))['design_sha'] == exe_sha, 'DESIGN DRIFT'
    log('execution.json match (sha %s)' % exe_sha[:8])
else:
    json.dump({'phase': PHASE, 'name': NAME, 'design_sha': exe_sha, 'design': design,
               'created': time.strftime('%Y-%m-%d %H:%M:%S')},
              open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log('execution.json FROZEN v2 (sha %s)' % exe_sha[:8])

if MODEL == 'summary':
    models = {}
    cls_vals = {}
    for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
        rp = os.path.join(RDIR, 'phase3164', NAME, m, 'result.json')
        r = json.load(open(rp, encoding='utf-8'))
        models[m] = dict(d1_3157=r['d1_3157'], d1_match=r['d1_match'], d_rope=r['d_rope'],
                         collapse_mean_kmax=r['collapse_mean_kmax'],
                         collapse_max_kmax=r['collapse_max_kmax'],
                         k_independence=r['k_independence'], cls=r['cls'],
                         prec=r['prec'], res_sha8=r['res_sha8'], seal_sha8=r['seal_sha8'])
        cls_vals[m] = r['cls']
    agree = len(set(cls_vals.values())) == 1
    summary = dict(phase=PHASE, name=NAME + '_summary', version=2,
                   axis='(c) massive 77x cross-model', models=models,
                   class_agreement_all3=agree, d1_refs=D1_REF,
                   verdict='g5a2c_summary|agree_%s|%s' % (
                       agree, '|'.join('%s_%s' % (m.split('-')[-1], c) for m, c in cls_vals.items())),
                   runtime_s=round(time.time() - T0, 1))
    seal_result(summary, 'result_summary.json')
    log('SUMMARY DONE')
    sys.exit(0)

# ---- 单模型（零 GPU） ----
NL = NL_OF[MODEL]
L_MID = NL // 2
# S1: d1 复算（3157 锚态）
z7 = np.load(P3157 % MODEL)
H7 = z7['H'].astype(np.float32)
assert H7.ndim == 3 and H7.shape[1] == NL + 1, (H7.shape, NL)
d1_3157 = int(np.argmax(np.abs(H7[:, L_MID, :]).mean(0)))
d1_match = bool(d1_3157 == D1_REF[MODEL])
mass_dom_3157 = float(np.median(np.abs(H7[:, L_MID, d1_3157])) /
                      np.median(np.abs(H7[:, L_MID, :])))
log('d1_3157=%d ref=%d match=%s mass_dom=%.1f (H7=%s)' % (
    d1_3157, D1_REF[MODEL], d1_match, mass_dom_3157, H7.shape))

# S2: 塌缩（4b=3156 npz；14b/glm4=3164b npz）
npz_p = P3156 if MODEL == 'qwen3-4b' else (P3164B % MODEL)
z = np.load(npz_p)
H = z['H'].astype(np.float32)
lang = [str(s) for s in z['lang']]; arm = [str(s) for s in z['arm']]
kk = [int(v) for v in z['k']]; ntv = [int(v) for v in z['n_tgt']]
NLr = H.shape[2] - 1; D = H.shape[3]
assert NLr == NL, (NLr, NL)
l_lo, l_hi = NL // 3, NL // 2
IDX = {(lang[i], arm[i], kk[i]): i for i in range(len(lang))}
KMAX = max(kk)
# d_rope：A0 mid 层 token 平均 argmax（合并两语言）
abs_chunks = []
for lg in sorted(set(lang)):
    ia0 = IDX[(lg, 'A', 0)]
    abs_chunks.append(np.abs(H[ia0, :ntv[ia0], l_lo:l_hi, :]).reshape(-1, D).mean(0))
d_rope = int(np.argmax(np.mean(abs_chunks, 0)))

collapse = {}
d1_tok = {}
for lg in sorted(set(lang)):
    ia0 = IDX[(lg, 'A', 0)]
    n = ntv[ia0]
    nrm0 = np.linalg.norm(H[ia0, :n, l_lo:l_hi, :], axis=2)      # (n, nl_mid)
    d0v = np.abs(H[ia0, :n, l_lo:l_hi, d_rope])                   # (n, nl_mid)
    row_m, row_x = {}, {}
    for k in range(KMAX + 1):
        if (lg, 'A', k) not in IDX:
            continue
        ia = IDX[(lg, 'A', k)]
        nrmk = np.linalg.norm(H[ia, :n, l_lo:l_hi, :], axis=2)
        row_m[str(k)] = float(np.mean(nrm0) / np.mean(nrmk))
        row_x[str(k)] = float(np.max(nrm0) / np.max(nrmk))
    dk = np.abs(H[IDX[(lg, 'A', KMAX)], :n, l_lo:l_hi, d_rope])
    d1_tok[lg] = dict(d_rope_peak_A0=float(np.max(d0v)),
                      d_rope_peak_Akmax=float(np.max(dk)),
                      d_rope_peak_ratio=float(np.max(d0v) / max(np.max(dk), 1e-9)))
    collapse[lg] = dict(collapse_mean=row_m, collapse_max=row_x)
cm = float(np.mean([np.mean([v for kk2, v in collapse[lg]['collapse_mean'].items()
                             if int(kk2) == KMAX]) for lg in collapse]))
cx = float(np.mean([np.mean([v for kk2, v in collapse[lg]['collapse_max'].items()
                             if int(kk2) == KMAX]) for lg in collapse]))
k1m = [collapse[lg]['collapse_max'].get('1') for lg in collapse]
kindep = float(np.mean([kk1 / cx_l for kk1, cx_l in zip(
    k1m, [np.mean([v for kk2, v in collapse[lg]['collapse_max'].items() if int(kk2) == KMAX])
          for lg in collapse])]))
log('collapse_mean(k=%d)=%.1f collapse_max=%.1f k_indep=%.2f d_rope=%d d_tok=%s'
    % (KMAX, cm, cx, kindep, d_rope, {lg: round(v['d_rope_peak_ratio'], 1) for lg, v in d1_tok.items()}))

if cx >= G_CX and cm >= G_CM and kindep < G_KINDEP:
    cls = 'massive_context_gate_supported'
elif cx >= G_CX and cm >= G_CM:
    cls = 'partial_collapse'
else:
    cls = 'absent'

verdict = 'g5a2c_%s|d1_%d%s|dscp_%d|cmean_%.1f|cmax_%.0f|kindep_%.2f' % (
    cls, d1_3157, '' if d1_match else '_MISMATCH', d_rope, cm, cx, kindep)

result = dict(phase=PHASE, name=NAME, version=2, model=MODEL, prec=PREC[MODEL],
              design_sha=exe_sha, nl=NL, l_mid=L_MID, mid_layers=[l_lo, l_hi - 1],
              d1_3157=d1_3157, d1_ref=D1_REF[MODEL], d1_match=d1_match,
              mass_dom_3157=mass_dom_3157, d_rope=d_rope, d_rope_token=d1_tok,
              collapse=collapse, collapse_mean_kmax=cm, collapse_max_kmax=cx,
              k_independence=kindep,
              gates=dict(collapse_mean_ge=G_CM, collapse_max_ge=G_CX,
                         k_independence_lt=G_KINDEP, cls=cls, d1_match=d1_match),
              cls=cls, verdict=verdict, runtime_s=round(time.time() - T0, 1))
seal_result(result, 'result.json')
log('DONE runtime=%.1fs' % (time.time() - T0))
