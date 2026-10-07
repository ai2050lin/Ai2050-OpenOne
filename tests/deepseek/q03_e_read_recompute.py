# -*- coding: utf-8 -*-
"""
Q03: E_read 统一基线复算（零 GPU；recompute-only）
================================================================
依据 : RDC_RESEARCH_CONSTITUTION_v1 §1 (I1)
       research/deepseek/atlas/metric_dict.json -> global_kpis.E_read
方法 : 逐字复用 tests/glm5/phase3152_g1p2_tri_model_k1.py 的
       split_s1 / rows_of / phi_main / ridge_primal / b4_fit
产物 : tests/deepseek/result/q03_execution.json  （预注册，观测前冻结）
       tests/deepseek/result/q03_result.json     （复算结果 + bootstrap CI）
       tests/deepseek/result/q03_report.txt      （人类可读）
纪律 : 复算，不产生新模型观测；数字一律现场渲染；不改任何既有判定
"""
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
B3152 = os.path.join(RDIR, 'phase3152', 'g1p2_tri_model_k1')
B3151 = os.path.join(RDIR, 'phase3151', 'g1p1_combo_additive_vs_interaction')
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'result')
os.makedirs(OUT, exist_ok=True)
LOG = []

def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    LOG.append(ln)
    try:
        print(ln, flush=True)
    except Exception:
        pass

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

# ---------------- 冻结面板材料（与 3151/3152 逐字一致） ----------------
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
SEEDS_S1 = [7, 8, 9]
FRAC_S1 = 0.2
ENTS = [e for cl in CLASSES for e in ENT[cl]]
CLS_OF = [CLASSES.index(cl) for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS)
NC = len(CLASSES)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)
NT = len(TPL)
PANEL = NT * NP_
keep_e = list(range(NE))
NE_KEEP = NE
assert (NP_, PANEL, NE, NC, NT) == (246, 738, 41, 6, 3), (NP_, PANEL, NE, NC, NT)

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

def ridge_primal(Xtr, Ytr, lam=1e-3):
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    return W

def phi_main(train_set):
    cols = NE_KEEP + NC + NT + 1
    tr_rows = rows_of(train_set)
    def rowvec(t, pi):
        i, c = PAIRS[pi]
        v = np.zeros(cols, np.float32)
        v[keep_e.index(i)] = 1.0
        v[NE_KEEP + c] = 1.0
        v[NE_KEEP + NC + t] = 1.0
        v[-1] = 1.0
        return v
    Xtr = np.stack([rowvec(r // NP_, r % NP_) for r in tr_rows])
    return Xtr, tr_rows, rowvec, cols

# ---------------- 三模型载体 ----------------
MODELS = [
    dict(name='qwen3-4b',
         npz=os.path.join(B3152, 'qwen3-4b', 'collect.npz'),
         res=os.path.join(B3152, 'qwen3-4b', 'result.json'), npz_sha8='4ef190b7'),
    dict(name='qwen3-14b',
         npz=os.path.join(B3152, 'qwen3-14b', 'collect.npz'),
         res=os.path.join(B3152, 'qwen3-14b', 'result.json'), npz_sha8='733182c1'),
    dict(name='glm4-9b',
         npz=os.path.join(B3151, 'collect.npz'),
         res=os.path.join(B3152, 'glm4k1', 'result.json'), npz_sha8='c711946c'),
]

# ---------------- 预注册（观测前冻结） ----------------
design = dict(
    query='Q03', title='E_read 统一基线复算',
    mode='recompute-only (no new model observation)',
    method=dict(
        predictor='B4 = ridge_primal(one-hot[entity] + one-hot[class] + one-hot[template] + bias)',
        lam=1e-3, feature_cols=NE_KEEP + NC + NT + 1,
        readout_layer='per-model, from phase3152 result.json k1_model_report.readout',
        normalization='Dk = mean train-row variance (NOT hidden dim) => E_read = normalized MSE',
        split='S1 s7 test fold; frac 0.2 -> 49 test pairs x 3 templates = 147 test rows',
        seeds=SEEDS_S1,
        aggregation='mean over 147 test rows; then mean over 3 seeds',
        gate=dict(threshold=0.05, sense='error <= 0.05 pass', name='5% gate'),
        bootstrap=dict(kind='percentile', unit='row-level within each seed',
                       n_boot=10000, rng_seed=20261003),
    ),
    carriers={m['name']: dict(npz=m['npz'], npz_sha8_expected=m['npz_sha8']) for m in MODELS},
    frozen_before='any (re)observation',
    created=time.strftime('%Y-%m-%d %H:%M:%S'),
)
ejson = json.dumps(design, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
design_sha = hashlib.sha256(ejson).hexdigest()
design['design_sha'] = design_sha
with open(os.path.join(OUT, 'q03_execution.json'), 'w', encoding='utf-8') as f:
    json.dump(design, f, ensure_ascii=False, indent=1)
log('execution.json FROZEN design_sha=%s' % design_sha[:8])

# ---------------- held-out 指纹（统一性验证） ----------------
fp = {}
for s in SEEDS_S1:
    tr, te = split_s1(s)
    fp[str(s)] = dict(n_train_pairs=len(tr), n_test_pairs=len(te),
                      test_pairs_sha8=hashlib.sha256(
                          json.dumps(sorted(map(list, te)), ensure_ascii=False).encode('utf-8')
                      ).hexdigest()[:8])
log('held-out fingerprint = %s' % json.dumps(fp, ensure_ascii=False))

# ---------------- 复算 ----------------
NB = 10000
result = dict(query='Q03', design_sha=design_sha, carriers={}, per_model={},
              fingerprint=fp, bootstrap=dict(n_boot=NB))
per_seed_pool = []

for m in MODELS:
    name = m['name']
    car_sha = sha8(m['npz'])
    car_ok = (car_sha == m['npz_sha8'])
    car_bytes = os.path.getsize(m['npz'])
    rp = json.loads(open(m['res'], 'rb').read().decode('utf-8-sig'))['k1_model_report']
    kout = int(rp['readout']); kstar = int(rp['kstar'])
    anchor_o = [float(x) for x in rp['b4_rel_readout_per_seed']]
    anchor_o_mean = float(rp['b4_rel_readout_mean3seed'])
    anchor_k = [float(x) for x in rp['b4_rel_kstar_per_seed']]
    anchor_k_mean = float(rp['b4_rel_kstar_mean3seed'])
    log('%s carrier sha8=%s ok=%s bytes=%d | readout=%d kstar=%d' %
        (name, car_sha, car_ok, car_bytes, kout, kstar))

    z = np.load(m['npz'])
    H16 = z['H']
    NTz, NPz, NHz, Dz = H16.shape
    assert NPz == NP_ and NTz == NT, ('H shape mismatch', H16.shape)
    log('%s H.shape=%s' % (name, (H16.shape,)))

    def Y_at(k):
        return H16[:, :, k, :].reshape(NT * NP_, Dz).astype(np.float32)

    def b4_fit(seed, k):
        train_set, test_set = split_s1(seed)
        tr_rows = rows_of(train_set)
        te_rows = rows_of(test_set)
        Xtr, _, rowvec, cols = phi_main(train_set)
        Y = Y_at(k)
        W = ridge_primal(Xtr, Y[tr_rows], lam=1e-3)
        Xte = np.stack([rowvec(r // NP_, r % NP_) for r in te_rows])
        B4te = Xte @ W
        ref = Y[tr_rows].mean(0)
        Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
        Yte = Y[te_rows]
        e = ((B4te - Yte) ** 2).sum(1) / Dk
        return e

    rec_o, rec_k, boot = [], [], {}
    for s in SEEDS_S1:
        eo = b4_fit(s, kout)
        ek = b4_fit(s, kstar)
        rec_o.append(float(eo.mean()))
        rec_k.append(float(ek.mean()))
        rng = np.random.RandomState(20261003)
        idx = rng.randint(0, len(eo), size=(NB, len(eo)))
        bs = eo[idx].mean(1)
        boot[str(s)] = dict(mean=float(eo.mean()),
                            ci_lo=float(np.percentile(bs, 2.5)),
                            ci_hi=float(np.percentile(bs, 97.5)),
                            n_rows=int(len(eo)))
    mean_o = float(np.mean(rec_o))
    drift_o = [abs(a - b) for a, b in zip(anchor_o, rec_o)]
    mean_k = float(np.mean(rec_k))
    drift_k = [abs(a - b) for a, b in zip(anchor_k, rec_k)]
    log('%s B4@readout recompute=%s mean=%.6f (anchor %.6f drift=%.2e)' %
        (name, ['%.5f' % x for x in rec_o], mean_o, anchor_o_mean,
         abs(mean_o - anchor_o_mean)))
    log('%s B4@kstar   recompute=%s mean=%.6f (anchor %.6f drift=%.2e)' %
        (name, ['%.5f' % x for x in rec_k], mean_k, anchor_k_mean,
         abs(mean_k - anchor_k_mean)))

    per_model = dict(
        readout_layer=kout, kstar=kstar, H_shape=list(H16.shape),
        b4_rel_readout_per_seed_recompute=rec_o,
        b4_rel_readout_per_seed_anchor=anchor_o,
        b4_rel_readout_per_seed_drift=drift_o,
        b4_rel_readout_mean3seed_recompute=mean_o,
        b4_rel_readout_mean3seed_anchor=anchor_o_mean,
        b4_rel_readout_mean3seed_drift=abs(mean_o - anchor_o_mean),
        b4_rel_kstar_per_seed_recompute=rec_k,
        b4_rel_kstar_mean3seed_recompute=mean_k,
        b4_rel_kstar_mean3seed_anchor=anchor_k_mean,
        bootstrap_per_seed=boot,
        gate_pass=bool(mean_o <= 0.05),
        anchor_match=bool(max(drift_o) < 1e-4 and abs(mean_o - anchor_o_mean) < 1e-4),
    )
    result['per_model'][name] = per_model
    result['carriers'][name] = dict(npz_sha8=car_sha, npz_sha8_expected=m['npz_sha8'],
                                    ok=car_ok, bytes=car_bytes)
    per_seed_pool.extend(rec_o)
    del H16, z

# ---------------- 汇总 ----------------
E = {k: v['b4_rel_readout_mean3seed_recompute'] for k, v in result['per_model'].items()}
pool_mean = float(np.mean(list(E.values())))
pool_sd = float(np.std(list(E.values()), ddof=1))
all_anchor_ok = all(v['anchor_match'] for v in result['per_model'].values())
all_car_ok = all(v['ok'] for v in result['carriers'].values())
gate_pass_n = sum(1 for v in result['per_model'].values() if v['gate_pass'])
result['summary'] = dict(
    E_read=dict(E),
    pooled_mean=pool_mean, pooled_sd_3models=pool_sd,
    gate_threshold=0.05, gate_pass_models=gate_pass_n, gate_pass_frac='%d/3' % gate_pass_n,
    gate_ratio_1x='%s x threshold' % ['%.2f' % (E[k] / 0.05) for k in E],
    min_E=float(min(E.values())), min_E_x=float(min(E.values()) / 0.05),
    all_anchor_match=bool(all_anchor_ok), all_carrier_sha_ok=bool(all_car_ok),
    per_seed_pool_n=int(len(per_seed_pool)),
)
result['sealed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
blob = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
result['res_sha8'] = hashlib.sha256(blob).hexdigest()[:8]
with open(os.path.join(OUT, 'q03_result.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=1)
log('result.json res_sha8=%s' % result['res_sha8'])

# ---------------- 报告 ----------------
R = []
R.append('Q03: E_read 统一基线复算 — 报告（recompute-only，零 GPU）')
R.append('=' * 68)
R.append('design_sha = %s' % design_sha[:8])
R.append('res_sha8   = %s' % result['res_sha8'])
R.append('')
R.append('统一 held-out 指纹（三模型共用同一构造）:')
for s, v in fp.items():
    R.append('  seed %s : %d train pairs / %d test pairs (x3 tpl = %d test rows)  test_sha8=%s'
             % (s, v['n_train_pairs'], v['n_test_pairs'], v['n_test_pairs'] * NT, v['test_pairs_sha8']))
R.append('')
R.append('逐模型复算 vs 锚（3161 口径：B4 加性 one-hot，λ=1e-3，Dk=训练方差）:')
R.append('  %-10s %-8s %-10s %-10s %-10s %s' % ('model', 'layer', 'recompute', 'anchor', 'drift', 'gate(<=0.05)'))
for k, v in result['per_model'].items():
    R.append('  %-10s %-8d %-10.6f %-10.6f %-10.2e %s'
             % (k, v['readout_layer'], v['b4_rel_readout_mean3seed_recompute'],
                v['b4_rel_readout_mean3seed_anchor'], v['b4_rel_readout_mean3seed_drift'],
                'PASS' if v['gate_pass'] else 'FAIL'))
R.append('')
R.append('per-seed（读出层）复算:')
for k, v in result['per_model'].items():
    R.append('  %-10s %s' % (k, ' '.join('%.6f' % x for x in v['b4_rel_readout_per_seed_recompute'])))
R.append('')
R.append('bootstrap 95%% CI（row-level, n_boot=%d, rng_seed=20261003）:' % NB)
for k, v in result['per_model'].items():
    for s, b in v['bootstrap_per_seed'].items():
        R.append('  %-10s seed %s : %.6f  [%.6f, %.6f]  (n_rows=%d)'
                 % (k, s, b['mean'], b['ci_lo'], b['ci_hi'], b['n_rows']))
R.append('')
R.append('汇总:')
S = result['summary']
R.append('  E_read 池化(3模型均值) = %.6f  (sd=%.6f, n=3)' % (S['pooled_mean'], S['pooled_sd_3models']))
R.append('  5%% 门 (<=0.05) 过门模型数 = %s' % S['gate_pass_frac'])
R.append('  最小 E_read = %.6f = 门的 %.2fx' % (S['min_E'], S['min_E_x']))
R.append('  carrier sha8 全部匹配 = %s' % S['all_carrier_sha_ok'])
R.append('  锚全部匹配 (drift<1e-4) = %s' % S['all_anchor_match'])
R.append('')
R.append('注: E_read 为归一化 MSE（除以训练行方差），非原始 L2。0.05 门 = 加性模型须解释 95% 方差。')
txt = '\n'.join(R)
with open(os.path.join(OUT, 'q03_report.txt'), 'w', encoding='utf-8') as f:
    f.write(txt + '\n\n--- run log ---\n' + '\n'.join(LOG) + '\n')
print(txt)
print()
print('ALL_ANCHOR_MATCH =', all_anchor_ok)
print('ALL_CARRIER_OK   =', all_car_ok)
print('GATE_PASS        =', '%d/3' % gate_pass_n)
print('EXIT_OK')
