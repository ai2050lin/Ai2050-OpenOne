# -*- coding: utf-8 -*-
# Phase 3181 (G5-B6): FTR-04 glm4 弱特异性归因（零 GPU，几何臂 vs 读出臂二选一定判）
# 预注册: AGI_GPT5_MEMO Phase 3180 tail（观测前）。FTR-04 弱特异性证据 = 3176 arm(a) LOEO
#   acc 0.6019/0.5981/0.4800（4b/14b/glm4，glm4 低于 0.5 门）。
# 数据源: 3169 collect npz ×3（H 面板 seen rows）+ 2806 CATS + per-model W_U（safetensors 只读）
#   + per-model tokenizer（CPU）。S_class 重建 = 3166 配方 verbatim。
# 协议:
#   几何臂: S_class(10,D) 行空间奇异谱 —— kappa=s1/s10、有效维 e90（累计能量>=0.9 的奇异值计数）、
#           PR（participation ratio）。
#   读出臂: 3169 seen rows（3176 arm_a 常量 verbatim，35 usable 实体 x 5 mappable 类）——
#           (i) 类均值差方向在 S_class 子空间的能量占比 proj_ratio（10 个 pairwise 均值）;
#           (ii) LOEO margin 分布重跑（3176 arm_a 判决式 verbatim，逐 test 行记录 top1-top2 归一化
#               cos margin），mean_margin。
#   门（freeze 钉定）:
#     G-a 几何确证: kappa_glm4 >= 1.25 x max(qwen)  OR  PR_glm4 <= 0.8 x min(qwen PR)
#         OR  e90_glm4 <= e90_qwen_min - 1;
#     G-b 读出确证: NOT G-a AND ( proj_glm4 <= 0.75 x min(qwen proj)
#         OR  margin_glm4 <= 0.75 x min(qwen margin) );
#     否则: 噪声级（弱特异性不作为独立机制挂账；E 级不变，held_out 路线维持 3176 判闭）。
#   registry v1.7 -> v1.8: 所有分支 counter_evidence 量化追加（append only）+ atlas_v1_9.html。
import os, sys, json, time, hashlib, copy, datetime
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3179_g5b4_ftr23_registration as p79

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g5b6_ftr04_glm4_attribution'
OUTDIR = os.path.join(SRC_DIR, 'phase3181', NAME)
SMOKE = os.environ.get('P3181_SMOKE') == '1'
LOG = []

def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    LOG.append(ln)
    try:
        print(ln, flush=True)
    except Exception:
        pass

def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]

def sha8_file(p):
    return sha8(open(p, 'rb').read())

SRC = {
    'p3169_4b': os.path.join(SRC_DIR, 'phase3169', 'g5a6_oov_panel', 'collect_qwen3-4b.npz'),
    'p3169_14b': os.path.join(SRC_DIR, 'phase3169', 'g5a6_oov_panel', 'collect_qwen3-14b.npz'),
    'p3169_glm4': os.path.join(SRC_DIR, 'phase3169', 'g5a6_oov_panel', 'collect_glm4-9b.npz'),
    'p3169_result': os.path.join(SRC_DIR, 'phase3169', 'g5a6_oov_panel', 'result.json'),
    'p2806_exec': os.path.join(SRC_DIR, 'phase2806', 'qwen4_hierarchy', 'execution.json'),
    'registry_v17': os.path.join(SRC_DIR, 'phase3180', 'g5b5_ftr15_heldout', 'atlas_registry_v1_7.json'),
    'gap_v15': os.path.join(SRC_DIR, 'phase3175', 'g5a12_residual_arms', 'gap_ledger_v1_5.json'),
    'registry_3162': os.path.join(SRC_DIR, 'phase3162', 'g5a1_atlas_foundation', 'atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575', 'p3169_glm4': '9cc3f8b4',
    'p3169_result': '5b51c2c1', 'p2806_exec': 'b291ea35',
    'registry_v17': 'f3ad070f', 'gap_v15': '4fb3f37d', 'registry_3162': '00f15e98',
}
CONTENT_SHA_3169 = ('49430a39', '018c6024')

# 3176 面板常量 verbatim
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
NE = len(ENTS)
CLASSES_3169 = CLASSES + ['乐器', '天气', '运动', '电器']
NE69, NC69 = 73, 10
NP69 = NE69 * NC69
SEEN_CLS_MAP = {'水果': 'fruit', '动物': 'animal', '交通工具': 'vehicle',
                '家具': 'furniture', '金属': 'metal'}
CLS69_IDX = {cn: i for i, cn in enumerate(CLASSES_3169)}
MODEL_NPZ = {'qwen3-4b': 'p3169_4b', 'qwen3-14b': 'p3169_14b', 'glm4-9b': 'p3169_glm4'}
MDIR_MAP = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'Qwen3-14B', 'glm4-9b': 'glm4-9b-chat-hf'}
G_KAPPA_X = 1.25
G_PR_X = 0.8
G_PROJ_X = 0.75
G_MARGIN_X = 0.75

def unit(v):
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-30)

def freeze():
    design = {
        'phase': '3181', 'name': NAME,
        'prereg': 'AGI_GPT5_MEMO Phase 3180 tail G5-B6 (frozen before any observation)',
        'data': 'sealed 3169 collect npz x3 (H panel seen rows) + 2806 CATS + per-model W_U '
                '(safetensors read-only) + per-model tokenizer (CPU); zero GPU',
        'protocol_geo': 'S_class rebuild per model (3166 verbatim); row-space SVD: kappa=s1/s10, '
                        'e90 = count of cumulative energy >=0.9, PR = (sum s^2)^2/sum s^4',
        'protocol_readout': '3169 seen rows (3176 constants verbatim): (i) pairwise class-mean '
                            'diff directions energy fraction in S_class subspace (proj_ratio, '
                            'mean over 10 pairs); (ii) LOEO margin re-run (3176 arm_a verdict '
                            'formula verbatim), mean_margin over test rows',
        'gates': {'Ga_kappa_x': G_KAPPA_X, 'Ga_pr_x': G_PR_X, 'Ga_e90_drop': 1,
                  'Gb_proj_x': G_PROJ_X, 'Gb_margin_x': G_MARGIN_X},
        'actions': 'G-a: geometry confirmed; G-b: readout confirmed; else: noise-level '
                   '(E-level unchanged in ALL branches; counter_evidence quantified append; '
                   'registry v1.7->v1.8 + atlas_v1_9.html)',
        'sources': SHA_ANCHOR,
    }
    eblob = json.dumps(design, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    dsha = hashlib.sha256(eblob).hexdigest()
    os.makedirs(OUTDIR, exist_ok=True)
    exe_p = os.path.join(OUTDIR, 'execution.json')
    if os.path.exists(exe_p) and not SMOKE:
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha8'] == dsha[:8], 'DESIGN DRIFT'
        log('freeze: existing execution.json OK (%s)' % dsha[:8])
    else:
        json.dump({'phase': 3181, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': design, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]

# ---------------- S_class rebuild（3166 verbatim） ----------------
def load_WU(mdir):
    from safetensors import safe_open
    cfgm = json.load(open(os.path.join(mdir, 'config.json'), encoding='utf-8'))
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
    e2806 = json.load(open(SRC['p2806_exec'], encoding='utf-8'))
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
    return dW_class.astype(np.float32)

# ---------------- 几何臂 ----------------
def geo_arm(S_class):
    S = S_class.astype(np.float64)
    s = np.linalg.svd(S, compute_uv=False)
    e2 = s ** 2
    cum = np.cumsum(e2) / e2.sum()
    e90 = int((cum < 0.9).sum()) + 1
    pr = float(e2.sum() ** 2 / (e2 ** 2).sum())
    kappa = float(s[0] / s[-1])
    return {'sv': [float(x) for x in s], 'kappa': kappa, 'e90': e90, 'pr': pr}

# ---------------- 读出臂 ----------------
def seen_rows_H(zpath):
    z = np.load(zpath)
    H = z['H']
    assert H.shape[0] == 3 and H.shape[2] >= 1
    NL = H.shape[2] - 1
    D = H.shape[3]
    Hk = H[:, :, NL, :].reshape(3 * NP69, D).astype(np.float64)
    return Hk

def readout_arm(Hk, S_class):
    S = S_class.astype(np.float64)
    Q, _ = np.linalg.qr(S.T)                      # (D,10) 子空间正交基
    usable = [i for i in range(NE) if CLASSES[CLS_OF[i]] in SEEN_CLS_MAP]
    assert len(usable) == 35
    cat_c69 = {cn: CLS69_IDX[cn_zh] for cn_zh, cn in SEEN_CLS_MAP.items()}
    # (i) proj_ratio: 类均值差方向在 S_class 子空间能量占比
    rows_by_cls = {}
    for cn in SEEN_CLS_MAP.values():
        rows_by_cls[cn] = []
    for i in usable:
        cn = SEEN_CLS_MAP[CLASSES[CLS_OF[i]]]
        r0 = i * NC69 + CLS69_IDX[CLASSES[CLS_OF[i]]]
        for t in range(3):
            rows_by_cls[cn].append(t * NP69 + r0)
    cents = {cn: Hk[rs].mean(0) for cn, rs in rows_by_cls.items()}
    cn_list = sorted(cents)
    projs = []
    for a in range(len(cn_list)):
        for b in range(a + 1, len(cn_list)):
            dp = cents[cn_list[a]] - cents[cn_list[b]]
            pe = float(np.sum((dp @ Q) ** 2) / max(float(dp @ dp), 1e-300))
            projs.append(pe)
    proj_ratio = float(np.mean(projs))
    # (ii) LOEO margin 重跑（3176 arm_a 判决式 verbatim）
    margins, fold_acc = [], []
    for e in usable:
        true_cn = SEEN_CLS_MAP[CLASSES[CLS_OF[e]]]
        train_rows, labels = [], []
        for i in usable:
            if i == e:
                continue
            r0 = i * NC69 + CLS69_IDX[CLASSES[CLS_OF[i]]]
            cn = SEEN_CLS_MAP[CLASSES[CLS_OF[i]]]
            for t in range(3):
                train_rows.append(t * NP69 + r0)
                labels.append(cn)
        test_rows = [t * NP69 + e * NC69 + c
                     for t in range(3) for c in sorted(cat_c69.values())]
        P = Hk[train_rows] @ S.T
        mus = {cn: np.stack([P[j] for j in range(len(train_rows)) if labels[j] == cn]).mean(0)
               for cn in sorted(set(labels))}
        Pt = Hk[test_rows] @ S.T
        ok, mrg = [], []
        norms = {cn: max(float(np.linalg.norm(mus[cn])), 1e-30) for cn in mus}
        for j in range(len(test_rows)):
            v = Pt[j]
            scores = sorted(((float(v @ mus[cn]) / norms[cn]), cn) for cn in mus)
            best = scores[-1]
            second = scores[-2]
            ok.append(1.0 if best[1] == true_cn else 0.0)
            mrg.append(best[0] - second[0])
        fold_acc.append(float(np.mean(ok)))
        margins.extend(mrg)
    return {'proj_ratio': proj_ratio, 'mean_margin': float(np.mean(margins)),
            'loeo_acc': float(np.mean(fold_acc)), 'n_pairs_proj': len(projs),
            'n_test_rows': len(margins), 'fold_accs': fold_acc}

def main():
    design_sha = freeze()
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    r69 = json.load(open(SRC['p3169_result'], encoding='utf-8'))
    assert r69['res_sha8'] == CONTENT_SHA_3169[0] and r69['seal_sha8'] == CONTENT_SHA_3169[1]
    assert_val = r69['gate']['ratio_B']
    assert abs(assert_val - 2.5388114997805062) < 1e-12, 'r69 ratio_B anchor'
    log('content double-lock: p3169 %s/%s, ratio_B anchor ok' % CONTENT_SHA_3169)

    MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
    out = {}
    for m in MODELS:
        W, _ = load_WU(os.path.join(ROOT, 'models', 'hf', MDIR_MAP[m]))
        S_class = build_S_class(W, os.path.join(ROOT, 'models', 'hf', MDIR_MAP[m]))
        g = geo_arm(S_class)
        Hk = seen_rows_H(SRC[MODEL_NPZ[m]])
        r = readout_arm(Hk, S_class)
        out[m] = dict(g, **r)
        log('%s: kappa=%.3f e90=%d pr=%.4f | proj=%.4f margin=%.4f loeo=%.4f'
            % (m, g['kappa'], g['e90'], g['pr'], r['proj_ratio'], r['mean_margin'],
               r['loeo_acc']))
        del W
    qwen = ['qwen3-4b', 'qwen3-14b']
    kmax_q = max(out[m]['kappa'] for m in qwen)
    prmin_q = min(out[m]['pr'] for m in qwen)
    e90min_q = min(out[m]['e90'] for m in qwen)
    projmin_q = min(out[m]['proj_ratio'] for m in qwen)
    mrgmin_q = min(out[m]['mean_margin'] for m in qwen)
    ga = (out['glm4-9b']['kappa'] >= G_KAPPA_X * kmax_q or
          out['glm4-9b']['pr'] <= G_PR_X * prmin_q or
          out['glm4-9b']['e90'] <= e90min_q - 1)
    gb = (not ga) and (out['glm4-9b']['proj_ratio'] <= G_PROJ_X * projmin_q or
                       out['glm4-9b']['mean_margin'] <= G_MARGIN_X * mrgmin_q)
    if ga:
        verdict = 'geometry_confirmed (S_class row-space ill-conditioning carries the glm4 weakness)'
    elif gb:
        verdict = 'readout_confirmed (S_class geometry comparable; readout-side dispersion carries it)'
    else:
        verdict = 'noise_level (weak specificity not attributable to geometry or readout; E-level unchanged)'
    log('gates: G-a=%s G-b=%s -> %s' % (ga, gb, verdict))
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3181, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'per_model': {m: {k: v for k, v in out[m].items() if k != 'sv' and k != 'fold_accs'}
                      for m in MODELS},
        'geo_sv': {m: out[m]['sv'] for m in MODELS},
        'gates': {'Ga_geometry': bool(ga), 'Gb_readout': bool(gb),
                  'Ga_kappa_x': G_KAPPA_X, 'Ga_pr_x': G_PR_X,
                  'Gb_proj_x': G_PROJ_X, 'Gb_margin_x': G_MARGIN_X},
        'verdict': verdict,
    }
    raw = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
    res8 = sha8(raw.encode('utf-8'))
    mid = json.dumps(dict(summary, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
    seal8 = sha8(mid.encode('utf-8'))
    rp = os.path.join(OUTDIR, suffix + 'result.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(dict(summary, res_sha8=res8, seal_sha8=seal8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('result written res=%s seal=%s' % (res8, seal8))
    if not SMOKE:
        finish_registry(summary, verdict, ga, gb, res8, seal8, design_sha)
    with open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    log('verdict: %s' % verdict)

# ---------------- registry v1.8 + atlas_v1_9 ----------------
CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
span = p79.span

def finish_registry(summary, verdict, ga, gb, res8, seal8, design_sha):
    reg17 = json.loads(open(SRC['registry_v17'], 'rb').read().decode('utf-8'))
    gl15 = json.loads(open(SRC['gap_v15'], encoding='utf-8').read())
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg17['version'] == '1.7' and len(reg17['features']) == 23
    reg18 = copy.deepcopy(reg17)
    reg18['version'] = '1.8'
    reg18['supersedes'] = 'atlas_registry_v1_7.json (f3ad070f, Phase 3180)'
    f04 = [f for f in reg18['features'] if f['id'] == 'FTR-04'][0]
    pm = summary['per_model']
    ce = ('3181 归因定判 %s：几何臂 kappa %.3f/%.3f/%.3f（e90 %d/%d/%d，PR %.4f/%.4f/%.4f）；'
          '读出臂 proj_ratio %.4f/%.4f/%.4f、LOEO mean_margin %.4f/%.4f/%.4f、'
          'acc 重跑 %.4f/%.4f/%.4f（3176 arm_a verbatim，与封存 0.6019/0.5981/0.4800 同量级）。'
          % ('G-a 几何确证' if ga else ('G-b 读出确证' if gb else '噪声级（不作为独立机制挂账）'),
             pm['qwen3-4b']['kappa'], pm['qwen3-14b']['kappa'], pm['glm4-9b']['kappa'],
             pm['qwen3-4b']['e90'], pm['qwen3-14b']['e90'], pm['glm4-9b']['e90'],
             pm['qwen3-4b']['pr'], pm['qwen3-14b']['pr'], pm['glm4-9b']['pr'],
             pm['qwen3-4b']['proj_ratio'], pm['qwen3-14b']['proj_ratio'], pm['glm4-9b']['proj_ratio'],
             pm['qwen3-4b']['mean_margin'], pm['qwen3-14b']['mean_margin'], pm['glm4-9b']['mean_margin'],
             pm['qwen3-4b']['loeo_acc'], pm['qwen3-14b']['loeo_acc'], pm['glm4-9b']['loeo_acc']))
    f04['counter_evidence'] = f04['counter_evidence'] + [ce]
    held_anchor = {
        'src': 'p3181', 'phase': 3181, 'role': 'weak_specificity_attribution',
        'tags': ['cross_model'],
        'asserts': {
            'kappa.qwen3-4b': pm['qwen3-4b']['kappa'], 'kappa.qwen3-14b': pm['qwen3-14b']['kappa'],
            'kappa.glm4-9b': pm['glm4-9b']['kappa'],
            'proj_ratio.glm4-9b': pm['glm4-9b']['proj_ratio'],
            'mean_margin.glm4-9b': pm['glm4-9b']['mean_margin'],
            'gate_Ga_geometry': bool(ga), 'gate_Gb_readout': bool(gb),
            'res_sha8': res8, 'seal_sha8': seal8}}
    f04['anchors'] = f04['anchors'] + [held_anchor]
    for f in reg17['features']:
        if f['id'] == 'FTR-04':
            continue
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg18['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        assert a == b, ('feature not preserved', f['id'])
    for k in ('failures', 'upgrade_log', 'evidence_level_taxonomy', 'principles',
              'schema', 'models_mainline', 'models_pending', 'provenance',
              'created', 'phase', 'name', 'design_sha8'):
        assert json.dumps(reg17.get(k), ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg18.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    rp = os.path.join(OUTDIR, 'atlas_registry_v1_8.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg18, ensure_ascii=False, indent=1))
    reg18_sha = sha8_file(rp)
    log('registry v1.8 written sha8 %s (FTR-04 counter_evidence append + anchor)' % reg18_sha)

    meta_sha = {'reg18': reg18_sha}
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.9 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.9 &mdash; 语义机制图谱（registry v1.8 渲染）</h1>')
    H.append('<div class="sub">Phase 3181 G5-B6 &middot; FTR-04 glm4 弱特异性归因（%s） &middot; '
             'registry v1.8 sha8=%s &middot; v1.7 sha8=%s &middot; v1.6 sha8=%s &middot; v1.5 sha8=%s'
             ' &middot; gap ledger v1.5 sha8=%s（3175 定稿，原样渲染）'
             ' &middot; 源 p3169 collect %s/%s/%s + result %s + p2806 %s &middot; 渲染于 %s</div>' % (
                 '几何确证' if ga else ('读出确证' if gb else '噪声级'),
                 span('meta.registry_v18_sha8', reg18_sha),
                 span('meta.registry_v17_sha8', 'f3ad070f'),
                 span('meta.registry_v16_sha8', 'bebdf38d'),
                 span('meta.registry_v15_sha8', p79.SHA_ANCHOR['registry_v15']),
                 span('meta.gap_v15_sha8', SHA_ANCHOR['gap_v15']),
                 span('meta.p3169_4b', SHA_ANCHOR['p3169_4b']),
                 span('meta.p3169_14b', SHA_ANCHOR['p3169_14b']),
                 span('meta.p3169_glm4', SHA_ANCHOR['p3169_glm4']),
                 span('meta.p3169_result', SHA_ANCHOR['p3169_result']),
                 span('meta.p2806_exec', SHA_ANCHOR['p2806_exec']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg18['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg18['principles']):
        H.append('<div class="kv"><b>原则%d</b>%s</div>' % (i + 1, p79.esc(pr)))
    H.append('</div>')
    H.append('<h2>图谱节点基座（16 节点，3162 audit disk_verified）</h2>')
    H.append('<table><tr><th>id</th><th>title</th><th>evidence</th><th>checks</th></tr>')
    for n in nodes:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td><span class="badge b-%s">%s</span>%s</td>'
                 '<td class="mono">%s</td></tr>' % (
                     n['id'], span(n['id'] + '.title', n['title']),
                     n['evidence_level'], span(n['id'] + '.elevel', n['evidence_level']),
                     span(n['id'] + '.status', n['status']),
                     span(n['id'] + '.checks', '%d/%d' % (n['n_pass'], n['n_checks']))))
    H.append('</table>')
    fams = {}
    for f in reg18['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.8，%d 条）</h2>' % len(reg18['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg18['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg18['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg18['failures']:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td class="mono">%s</td><td>%s</td><td class="mono">%s</td></tr>' % (
            fl['id'], span(fl['id'] + '.kind', fl['kind']), span(fl['id'] + '.phase', fl['phase']),
            span(fl['id'] + '.text', fl['text']), span(fl['id'] + '.evidence', fl['evidence'])))
    H.append('</table>')
    H.append('<h2>缺口账本 v1.5（GAP-4 quantified_collapse 定稿）</h2>')
    for g in gl15['gaps']:
        closed_at = str(g['closed_at_phase']) if g.get('closed_at_phase') is not None else '-'
        badge_closed = '<span class="badge">closed@%s</span>' % span(g['id'] + '.closed_at', closed_at) \
            if g['status'].startswith('closed') else ''
        H.append('<div class="card"><h3>%s &middot; %s <span class="badge b-%s">%s</span>%s</h3>' % (
            g['id'], span(g['id'] + '.title', g['title']), g['status'], span(g['id'] + '.status', g['status']), badge_closed))
        H.append('<div class="kv"><b>statement</b>%s</div>' % span(g['id'] + '.statement', g['statement']))
        H.append('<div class="evid">evidence:</div>')
        H.append('<div class="kv">%s</div>' % span(g['id'] + '.evidence', ' || '.join(g['evidence'])))
        H.append('<div class="anchorbox"><b>anchors</b> %s</div>' % span(
            g['id'] + '.anchors', ', '.join('%s=%s' % (k, v) for k, v in sorted(g['anchor_sha8'].items()))))
        if g['id'] == 'GAP-4':
            pr = g['prereg']
            H.append('<div class="prereg"><div class="kv"><b>prereg</b>Phase %s %s（%s）</div>' % (
                span('GAP-4.prereg.phase', pr['phase']), span('GAP-4.prereg.name', pr['name']),
                span('GAP-4.prereg.status', pr['status'])))
            H.append('<div class="kv"><b>hypothesis</b>%s</div>' % span('GAP-4.prereg.hypothesis', pr['hypothesis']))
            H.append('<div class="kv"><b>protocol</b>%s</div>' % span('GAP-4.prereg.protocol', pr['protocol']))
            H.append('<div class="kv"><b>gate</b>%s</div></div>' % span('GAP-4.prereg.gate', pr['gate']))
            H.append('<div class="mech"><div class="kv"><b>mechanism_note</b>%s</div></div>' % span(
                'GAP-4.mechanism_note', g['mechanism_note']))
        H.append('</div>')
    H.append('<h2>附录挂账（%d 项，逐条证据锚）</h2><div class="card">' % len(gl15['appendix']))
    for a in gl15['appendix']:
        H.append('<div class="kv"><b>%s</b>%s <span class="badge b-%s">%s</span>'
                 '<span class="mono">anchor=%s(%s)</span><div class="evid">%s</div></div>' % (
                     a['id'], span(a['id'] + '.title', a['title']), a['status'],
                     span(a['id'] + '.status', a['status']),
                     span(a['id'] + '.anchor', a['anchor']),
                     span(a['id'] + '.anchor_sha8', a['anchor_sha8']),
                     span(a['id'] + '.note', a['note'])))
    H.append('</div>')
    H.append('<div class="footer">RDC/LPF atlas v1.9 &middot; 渲染自 atlas_registry_v1_8.json (%s)'
             ' + gap ledger v1.5 (%s) &middot; 增量来源: Phase 3181 G5-B6 FTR-04 glm4 弱特异性归因'
             '（数据源 p3169 collect %s / %s / %s + p2806 %s）&middot; 本文件为 Phase 3181 封存产物，'
             '字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg18_sha, SHA_ANCHOR['gap_v15'],
                 SHA_ANCHOR['p3169_4b'], SHA_ANCHOR['p3169_14b'], SHA_ANCHOR['p3169_glm4'],
                 SHA_ANCHOR['p2806_exec']))
    H.append('</body></html>')
    html_text = '\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_9.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v19(reg18, nodes, gl15, meta_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_9.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3181, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True, 'arm': {'verdict': verdict, 'per_model': summary['per_model']},
        'counts': {'features': 23, 'failures': len(reg18['failures']),
                   'upgrades': len(reg18['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': 'v1.7_to_v1.8_ftr04_counter_evidence_append',
        'registry_v18_file': 'atlas_registry_v1_8.json', 'registry_v18_sha8': reg18_sha,
        'html_file': 'atlas_v1_9.html', 'html_sha8': html_sha,
        'field_check': {'n_expected': v['n_expected'], 'n_found': v['n_found'],
                        'missing': len(v['missing']), 'extra': len(v['extra']),
                        'mismatch': len(v['mismatch']), 'dups': len(v['dups'])},
        'sources': SHA_ANCHOR, 'res_sha8': res8, 'seal_sha8': seal8,
    }
    raw = json.dumps(final, ensure_ascii=False, indent=1, sort_keys=True)
    f_res8 = sha8(raw.encode('utf-8'))
    mid = json.dumps(dict(final, res_sha8=f_res8), ensure_ascii=False, indent=1, sort_keys=True)
    f_seal8 = sha8(mid.encode('utf-8'))
    with open(os.path.join(OUTDIR, 'final_result.json'), 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(dict(final, res_sha8=f_res8, seal_sha8=f_seal8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('final result res=%s seal=%s' % (f_res8, f_seal8))

def flatten_v19(reg, nodes, gl, meta_sha):
    d = {}
    for f in reg['features']:
        p = f['id']
        d[p + '.statement'] = f['statement']
        d[p + '.family'] = f['family']
        d[p + '.evidence_level'] = f['evidence_level']
        d[p + '.model_scope'] = ', '.join(f['model_scope'])
        d[p + '.n_anchors'] = str(len(f['anchors']))
        d[p + '.counter_evidence'] = ' | '.join(f['counter_evidence']) if f['counter_evidence'] else '(none)'
        d[p + '.replication'] = f['replication']
        d[p + '.scope_limits'] = f['scope_limits'] if f['scope_limits'] else '(none)'
        for i, a in enumerate(f['anchors']):
            d['%s.anchors.%d.src' % (p, i)] = str(a['src'])
            d['%s.anchors.%d.phase' % (p, i)] = str(a['phase'])
            d['%s.anchors.%d.role' % (p, i)] = a['role']
            d['%s.anchors.%d.asserts' % (p, i)] = json.dumps(a['asserts'], ensure_ascii=False, sort_keys=True)
        for k, v in (f.get('values') or {}).items():
            d['%s.values.%s' % (p, k)] = p79.fmt(v) if hasattr(p79, 'fmt') else str(v)
    for n in nodes:
        d[n['id'] + '.title'] = n['title']
        d[n['id'] + '.elevel'] = n['evidence_level']
        d[n['id'] + '.status'] = n['status']
        d[n['id'] + '.checks'] = '%d/%d' % (n['n_pass'], n['n_checks'])
    for i, u in enumerate(reg['upgrade_log']):
        d['UPG.%d.node' % i] = u['node']
        d['UPG.%d.change' % i] = u['change']
        d['UPG.%d.reason' % i] = u['reason']
    for fl in reg['failures']:
        d[fl['id'] + '.kind'] = fl['kind']
        d[fl['id'] + '.phase'] = str(fl['phase'])
        d[fl['id'] + '.text'] = fl['text']
        d[fl['id'] + '.evidence'] = str(fl['evidence'])
    for g in gl['gaps']:
        p = g['id']
        d[p + '.title'] = g['title']
        d[p + '.status'] = g['status']
        if g['status'].startswith('closed'):
            d[p + '.closed_at'] = str(g['closed_at_phase'])
        d[p + '.statement'] = g['statement']
        d[p + '.evidence'] = ' || '.join(g['evidence'])
        d[p + '.anchors'] = ', '.join('%s=%s' % (k, v) for k, v in sorted(g['anchor_sha8'].items()))
        if p == 'GAP-4':
            pr = g['prereg']
            d[p + '.prereg.phase'] = str(pr['phase'])
            d[p + '.prereg.name'] = pr['name']
            d[p + '.prereg.hypothesis'] = pr['hypothesis']
            d[p + '.prereg.protocol'] = pr['protocol']
            d[p + '.prereg.gate'] = pr['gate']
            d[p + '.prereg.status'] = pr['status']
            d[p + '.mechanism_note'] = g['mechanism_note']
    for a in gl['appendix']:
        p = a['id']
        d[p + '.title'] = a['title']
        d[p + '.status'] = a['status']
        d[p + '.note'] = a['note']
        d[p + '.anchor'] = a['anchor']
        d[p + '.anchor_sha8'] = a['anchor_sha8']
    d['meta.registry_v18_sha8'] = meta_sha['reg18']
    d['meta.registry_v17_sha8'] = 'f3ad070f'
    d['meta.registry_v16_sha8'] = 'bebdf38d'
    d['meta.registry_v15_sha8'] = p79.SHA_ANCHOR['registry_v15']
    d['meta.gap_v15_sha8'] = SHA_ANCHOR['gap_v15']
    d['meta.p3169_4b'] = SHA_ANCHOR['p3169_4b']
    d['meta.p3169_14b'] = SHA_ANCHOR['p3169_14b']
    d['meta.p3169_glm4'] = SHA_ANCHOR['p3169_glm4']
    d['meta.p3169_result'] = SHA_ANCHOR['p3169_result']
    d['meta.p2806_exec'] = SHA_ANCHOR['p2806_exec']
    d['meta.created'] = CREATED_STR
    return d

if __name__ == '__main__':
    main()
