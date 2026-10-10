# -*- coding: utf-8 -*-
"""Phase 3173: G5-A10 atlas v1.3 render closeout (zero GPU).

Renders the atlas after the 3171 mechanism localization + 3172 port calibration
intervention, per prereg in MEMO Phase 3172 tail:
  (a) registry v1.1 -> v1.2: append FTR-22 (port calibration recovery curve,
      family=limit, E2, anchors = p3172 intervention + p3169 base measurement
      + p3171 mechanism correlation = 3 independent phase anchors); existing 21
      features preserved byte-for-byte;
  (b) gap ledger v1.3 -> v1.4: GAP-4 statement extended with the 3172 verdict
      (port missing ~29% of collapse, residual ~1.8x structural, hung account);
      evidence += 2 items; anchor_sha8 += p3171, p3172; mechanism_note kept
      byte-for-byte; status stays quantified_collapse;
  (c) atlas_v1_3.html full re-render (22 features, 16 nodes, GAP-4 incl full
      mechanism_note text) with per-field data-k verification.

Gates:
  G1 html field check: missing/extra/mismatch/dup all 0
  G2 feature cards rendered (22 full / 3 smoke)
  G3 v1.1 features preserved byte-for-byte 21/21
  G4 GAP-4 increment asserted (statement extended, evidence +2, anchors +2,
     mechanism_note byte-identical, status unchanged); GAP-1/2/3 + appendix unchanged
  G5 FTR-22 runtime asserts (p3172 curve + k0 bitwise re-3169 + p3171 ratios)
  G6 no external resources in html

SMOKE: env P3173_SMOKE=1 renders a subset through the full pipeline.
DESIGN is fully static (no runtime-derived values; 3169 lesson).
"""
import io
import os
import re
import json
import copy
import html
import hashlib
import datetime

BASE = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(BASE, r'tests\glm5\result\rdc_query_construction_20260913')
OUTDIR = os.path.join(SRC_DIR, 'phase3173', 'g5a10_atlas_v13')
SMOKE = os.environ.get('P3173_SMOKE', '') == '1'

P3169_DIR = os.path.join(SRC_DIR, 'phase3169', 'g5a6_oov_panel')
P3171_DIR = os.path.join(SRC_DIR, 'phase3171', 'g5a8_collapse_mechanism')
P3172_DIR = os.path.join(SRC_DIR, 'phase3172', 'g5a9_port_calibration')
V11_DIR = os.path.join(SRC_DIR, 'phase3170', 'g5a7_atlas_v11')

SRC = {
    'registry_v11': os.path.join(V11_DIR, 'atlas_registry_v1_1.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
    'gap_ledger_v13': os.path.join(V11_DIR, 'gap_ledger_v1_3.json'),
    'p3169': os.path.join(P3169_DIR, 'result.json'),
    'p3171': os.path.join(P3171_DIR, 'result.json'),
    'p3172': os.path.join(P3172_DIR, 'result.json'),
    'q03': os.path.join(BASE, r'tests\deepseek\result\q03_result.json'),
}

SHA_ANCHOR = {
    'registry_v11': '1fedbd80', 'registry_3162': '00f15e98',
    'gap_ledger_v13': '2436ec08',
    'p3169': '5b51c2c1', 'p3171': 'a43cf48c', 'p3172': 'd8ddc481',
    'q03': '57827730',
}

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
KS = ['0', '1', '2', '4', '8']

DESIGN = {
    'phase': 3173, 'name': 'g5a10_atlas_v13', 'zero_gpu': True,
    'sources': {k: v.replace(BASE, '') for k, v in SRC.items()},
    'updates': {
    'registry': 'v1.1 -> v1.2: append FTR-22 (port calibration recovery curve, family=limit, '
                'E2_predictive, anchors p3172 primary_intervention + p3169 base_measurement '
                'replication + p3171 mechanism_correlation = 3 independent phase anchors); '
                'features[0:21] preserved byte-for-byte; failures += F13 (port-missing is not '
                'the whole collapse cause, p3172) + F14 (mechanism 3-way gate boundary '
                'sensitivity on glm4, p3171); upgrade_log unchanged (no E-level upgrade '
                'events in 3171/3172).',
        'gap_ledger': 'v1.3 -> v1.4: GAP-4 statement extended with the 3172 intervention verdict '
                      '(port missing removes ~29% of collapse at k=8, residual ~1.8x structural '
                      'component hung); evidence += 2 runtime-rendered items; anchor_sha8 += '
                      'p3171, p3172; mechanism_note byte-for-byte unchanged; status stays '
                      'quantified_collapse; GAP-1/2/3 and appendix 5 items unchanged byte-for-byte.',
        'html': 'atlas_v1_3.html full re-render (single file, inline CSS, zero external resources): '
                '22 features, 16 nodes, 4 gaps incl GAP-4.mechanism_note full text rendered for the '
                'first time, with per-field data-k verification against flattened v1.2/v1.4 sources.',
    },
    'ftr22_gate': 'render gate (3172 verdict re-asserted, not re-measured): pooled ratio_B(k=8) in '
                  '[1.5, 2.0) => borderline_partial_recovery stands; k=0 pooled must reproduce '
                  '3169 gate ratio_B within 1e-9 (bitwise anchoring of the k=0 arm).',
    'gates': {
        'G1': 'html data-k field check: missing/extra/mismatch/dup all 0',
        'G2': 'feature cards rendered (22 full / 3 smoke)',
        'G3': 'v1.1 features preserved byte-for-byte (21/21)',
        'G4': 'GAP-4 increment asserted (statement+evidence+anchors; mechanism_note byte-identical; '
              'status unchanged); other gaps + appendix unchanged',
        'G5': 'FTR-22 runtime asserts pass (k0 re-3169 drift<1e-9, k8 in [1.5,2), p3171 ratios)',
        'G6': 'no external resources: no <link, no <script, no http(s):// in html',
        'G7': 'failures F13/F14 appended (registry-level falsifiability log); first 12 '
              'byte-for-byte; upgrade_log unchanged at 3',
    },
    'smoke': {'features': 3, 'nodes': 4, 'failures': 2, 'upgrades': 1, 'gaps': 'all'},
}

LOG = []


def log(s):
    LOG.append(str(s))
    print(s)


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def sha8_file(p):
    return sha8(open(p, 'rb').read())


def fmt(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return '%.6g' % v
    return str(v)


def assert_val(tag, actual, expect, tol=1e-6):
    if isinstance(expect, float) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        ok = abs(float(actual) - expect) <= tol * max(1.0, abs(expect))
    elif isinstance(expect, list):
        ok = list(actual) == list(expect)
    else:
        ok = actual == expect
    assert ok, ('assert fail', tag, actual, expect)
    return ok


# ---------------------------------------------------------------- freeze
def freeze():
    os.makedirs(OUTDIR, exist_ok=True)
    exep = os.path.join(OUTDIR, 'execution.json')
    body = dict(DESIGN)
    raw = json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True)
    d8 = sha8(raw.encode('utf-8'))
    body['design_sha8'] = d8
    raw2 = json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True)
    if os.path.exists(exep):
        prev = json.load(io.open(exep, encoding='utf-8'))
        if prev.get('design_sha8') != d8:
            raise SystemExit('DRIFT: execution.json design_sha8 %s != current %s'
                             % (prev.get('design_sha8'), d8))
        log('freeze: existing execution.json OK (%s)' % d8)
    else:
        with io.open(exep, 'w', encoding='utf-8') as f:
            f.write(raw2)
        log('freeze: execution.json written design_sha8=%s' % d8)
    return d8


# ---------------------------------------------------------------- FTR-22 build (values runtime-rendered)
def build_ftr22(r72, r69, r71, rq):
    assert r72['res_sha8'] == '463f42c8' and r72['seal_sha8'] == 'cdb85525'
    assert r72['verdict'] == ('g5a9_port_calibration|full|3_models|ratio_k8=1.8002|'
                              'borderline_partial_recovery')
    assert r69['res_sha8'] == '49430a39' and r69['seal_sha8'] == '018c6024'
    # 3169 gate values measured at full precision (2026-10-09 probe p3173_g69.txt)
    assert_val('r69 gate.ratio_B', r69['gate']['ratio_B'], 2.5388114997805062, tol=1e-12)
    assert_val('r69 gate.pooled_E_oov_B', r69['gate']['pooled_E_oov_B'], 0.9478664530648125, tol=1e-12)
    assert_val('r69 gate.pooled_E_seen_B', r69['gate']['pooled_E_seen_B'], 0.37335046463542515, tol=1e-12)
    assert r71['res_sha8'] == '6a29201c' and r71['seal_sha8'] == 'cc7ccedd'
    assert r71['verdict'] == ('g5a8_collapse_mechanism|full|3_models|'
                              'ratio_S_main=0.7511/0.7167/0.8205|mixed_across_models')
    assert_val('q03 pooled_mean', rq['summary']['pooled_mean'], 0.37335047125816345, tol=1e-12)

    pooled = {k: r72['pooled'][k] for k in KS}
    rk = {k: pooled[k]['ratio'] for k in KS}
    # k=0 bitwise anchoring to 3169 sealed gate
    dr_ratio = abs(rk['0'] - r69['gate']['ratio_B'])
    assert dr_ratio < 1e-9, ('k0 re-3169 ratio drift', dr_ratio)
    dr_eoov = abs(pooled['0']['E_oov'] - r69['gate']['pooled_E_oov_B'])
    dr_eseen = abs(pooled['0']['E_seen'] - r69['gate']['pooled_E_seen_B'])
    assert dr_eoov < 1e-6 and dr_eseen < 1e-6, ('k0 re-3169 E drift', dr_eoov, dr_eseen)
    # prereg band: per-model k=8 all in [1.5, 2)
    r8m = {m: r72['per_model'][m]['kcurves']['8']['ratio'] for m in MODELS}
    for m in MODELS:
        assert 1.5 <= r8m[m] < 2.0, ('k8 band', m, r8m[m])
        assert r72['per_model'][m]['kout'] == (35 if m == 'qwen3-4b' else 39)
    # monotone non-increasing curve
    seq = [rk[k] for k in KS]
    assert all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1)), ('curve monotone', seq)
    port_frac = (rk['0'] - rk['8']) / rk['0']
    assert 0.28 <= port_frac <= 0.30, ('port removed frac', port_frac)
    # p3171 mechanism ratios re-assert
    rs = {m: r71['per_model'][m]['slots']['k_main']['ratio_S'] for m in MODELS}
    for m in MODELS:
        assert rs[m] > 0.5, ('encoding_missing must stay rejected', m, rs[m])
    assert rs['qwen3-4b'] < 0.8 and rs['qwen3-14b'] < 0.8 and rs['glm4-9b'] > 0.8
    assert r71['overall']['main_cls'] == 'mixed_across_models'
    # E_newent pooled (mean of 3 models, same aggregation as 3172 closeout)
    en0 = sum(r72['per_model'][m]['kcurves']['0']['E_newent'] for m in MODELS) / 3.0
    en8 = sum(r72['per_model'][m]['kcurves']['8']['E_newent'] for m in MODELS) / 3.0
    assert 0.73 <= en0 <= 0.75 and 0.52 <= en8 <= 0.54, ('E_newent pooled range', en0, en8)

    stmt = ('谱外类端口校准恢复曲线（协议内干预，3172）：对每谱外类加入 k 个校准实体行进口径 B 训练集'
            '（k in {0,1,2,4,8}，校准与测试实体不相交），pooled ratio_B(k) = %.4f -> %.4f -> %.4f -> '
            '%.4f -> %.4f；k=0 逐位复现 3169 封存值（ratio drift=%.1e）。三模型 k=8 = %.4f/%.4f/%.4f '
            '全落预注册 [1.5,2) 带 => borderline_partial_recovery：one-hot 类端口缺失（类别级留出下'
            '无训练信号）为崩塌的显著但部分成分——k=8 移除约 %.1f%% 崩塌量，恢复集中于 k=0->1'
            '（单实体校准 -20%%）；残留 ~1.8x 过量误差 = 更深成分（结构性残留，挂账定位）。') % (
        rk['0'], rk['1'], rk['2'], rk['4'], rk['8'], dr_ratio,
        r8m['qwen3-4b'], r8m['qwen3-14b'], r8m['glm4-9b'], port_frac * 100)

    ftr = {
        'id': 'FTR-22',
        'family': 'limit',
        'statement': stmt,
        'evidence_level': 'E2_predictive',
        'model_scope': list(MODELS),
        'anchors': [
            {'src': 'p3172', 'phase': 3172, 'role': 'primary_intervention', 'tags': ['intervention', 'cross_model'],
             'asserts': {'pooled.0.ratio': rk['0'], 'pooled.1.ratio': rk['1'], 'pooled.2.ratio': rk['2'],
                         'pooled.4.ratio': rk['4'], 'pooled.8.ratio': rk['8'],
                         'per_model.qwen3-4b.kcurves.8.ratio': r8m['qwen3-4b'],
                         'per_model.qwen3-14b.kcurves.8.ratio': r8m['qwen3-14b'],
                         'per_model.glm4-9b.kcurves.8.ratio': r8m['glm4-9b'],
                         'res_sha8': '463f42c8', 'seal_sha8': 'cdb85525'}},
            {'src': 'p3169', 'phase': 3169, 'role': 'base_measurement_k0_replication', 'tags': ['device', 'held_out'],
             'asserts': {'gate.ratio_B': r69['gate']['ratio_B'],
                         'gate.pooled_E_oov_B': r69['gate']['pooled_E_oov_B'],
                         'gate.pooled_E_seen_B': r69['gate']['pooled_E_seen_B'],
                         'k0_drift_ratio': dr_ratio, 'k0_drift_E_oov': dr_eoov, 'k0_drift_E_seen': dr_eseen,
                         'res_sha8': '49430a39', 'seal_sha8': '018c6024'}},
            {'src': 'p3171', 'phase': 3171, 'role': 'mechanism_correlation', 'tags': ['mechanism'],
             'asserts': {'per_model.qwen3-4b.slots.k_main.ratio_S': rs['qwen3-4b'],
                         'per_model.qwen3-14b.slots.k_main.ratio_S': rs['qwen3-14b'],
                         'per_model.glm4-9b.slots.k_main.ratio_S': rs['glm4-9b'],
                         'overall.main_cls': 'mixed_across_models',
                         'res_sha8': '6a29201c', 'seal_sha8': 'cc7ccedd'}},
        ],
        'counter_evidence': [
            '恢复非全量（k=8 残留 ~1.8x）=> 未达 port_missing_confirmed（E3 因果收敛不成立）；'
            'gap ledger GAP-4 status 保持 quantified_collapse 而非 closed',
            '校准实体选择为单次 RandomState(seed+1000) 实现，未跨选择 seed 重复（3172 prereg 已声明）；'
            '干预施加于 ridge 读出探针训练集（协议内干预），非模型权重/激活干预',
        ],
        'replication': '复用 3169 collect npz（三模型 sha8 锚 36eb4ff0/97f98575/9cc3f8b4）+ Q03 verbatim '
                       'ridge（B4 one-hot 特征空间 lam=1e-3、k* 读位）x 3 E_read 种子；装置门 k=0 '
                       'per-model E_oov/E_seen/E_newent + pooled 全部逐位复现 3169 封存值'
                       '（drift=0.00e+00）；恢复曲线 pooled 聚合 = E_oov/E_seen 池化比。',
        'values': {
            'ratio_k0': rk['0'], 'ratio_k1': rk['1'], 'ratio_k2': rk['2'],
            'ratio_k4': rk['4'], 'ratio_k8': rk['8'],
            'ratio_k8_qwen3_4b': r8m['qwen3-4b'],
            'ratio_k8_qwen3_14b': r8m['qwen3-14b'],
            'ratio_k8_glm4_9b': r8m['glm4-9b'],
            'E_oov_pooled_k0': pooled['0']['E_oov'], 'E_oov_pooled_k8': pooled['8']['E_oov'],
            'E_seen_pooled_k0': pooled['0']['E_seen'], 'E_seen_pooled_k8': pooled['8']['E_seen'],
            'E_newent_pooled_mean3_k0': en0, 'E_newent_pooled_mean3_k8': en8,
            'port_removed_frac': port_frac,
            'k0_drift_ratio_vs_3169': dr_ratio,
            'p3171_ratio_S_qwen3_4b': rs['qwen3-4b'],
            'p3171_ratio_S_qwen3_14b': rs['qwen3-14b'],
            'p3171_ratio_S_glm4_9b': rs['glm4-9b'],
        },
        'scope_limits': '4 谱外类（乐器/天气/运动/电器）；校准实体从面板内其余实体抽样，per-class 曲线'
                        '全长完整（校准只移除抽样实体自身行）；残留 ~1.8x 成分未定位=挂账（候选：实体x类'
                        '交互不可迁移，3171 MARG 词级正常支持此倾向）；范围=中文面板+单一模板族+行为读出层'
                        '+协议内干预。',
    }
    return ftr


# ---------------------------------------------------------------- gap ledger v1.4
def build_failures_append(r72, r71):
    """F13/F14 per prereg (c): registry-level falsifiability log additions."""
    rk = {k: r72['pooled'][k]['ratio'] for k in KS}
    port_frac = (rk['0'] - rk['8']) / rk['0']
    rs = {m: r71['per_model'][m]['slots']['k_main']['ratio_S'] for m in MODELS}
    f13 = {
        'id': 'F13',
        'kind': 'hypothesis_rejected',
        'phase': 3172,
        'text': '「one-hot 类端口缺失=谱外类崩塌全部成因」被否证：k=8 校准后 pooled ratio_B 残留 '
                '%.4f（仅移除 %.1f%% 崩塌量），残留 ~1.8x 过量误差=结构性成分。教训：部分恢复 != '
                '机制全解释；缺口④须保持 quantified_collapse 挂账。' % (rk['8'], port_frac * 100),
        'evidence': 'p3172 res 463f42c8 / seal cdb85525; pooled ratio_B(k) %.4f -> %.4f'
                    % (rk['0'], rk['8']),
    }
    f14 = {
        'id': 'F14',
        'kind': 'gate_boundary',
        'phase': 3171,
        'text': '崩塌机制三分类（encoding/readout/mixed）对 0.8 门边界敏感：glm4 ratio_S=0.8205 '
                '刚过门 +0.02（readout_missing），4b/14b 明确 mixed（0.7511/0.7167）。主判 '
                'mixed_across_models 的稳健性在门边界处有限；三模型一致否定的只有 '
                'encoding_missing（全 >0.5）。',
        'evidence': 'p3171 res 6a29201c / seal cc7ccedd; ratio_S %.4f/%.4f/%.4f vs gate 0.8'
                    % (rs['qwen3-4b'], rs['qwen3-14b'], rs['glm4-9b']),
    }
    return [f13, f14]


# ---------------------------------------------------------------- gap ledger v1.4
def build_gap_ledger_v14(gl13, r72):
    gl = copy.deepcopy(gl13)
    gl['schema'] = 'rdc_atlas_gap_ledger_v1_4'
    gl['version'] = '1.4'
    gl['updated'] = datetime.datetime.now().strftime('%Y-%m-%d')
    gl['supersedes'] = 'gap_ledger_v1_3 (2436ec08, phases 3171/3172)'
    g4 = [g for g in gl['gaps'] if g['id'] == 'GAP-4'][0]
    assert g4['status'] == 'quantified_collapse'
    mn_before = g4['mechanism_note']
    pooled = r72['pooled']
    rk = {k: pooled[k]['ratio'] for k in KS}
    port_frac = (rk['0'] - rk['8']) / rk['0']
    r8m = {m: r72['per_model'][m]['kcurves']['8']['ratio'] for m in MODELS}
    g4['statement'] = (g4['statement'] +
                       ' 3172 端口校准定判（协议内干预）：one-hot 类端口缺失为崩塌的显著但部分成分——'
                       '加 k=8 校准实体后 pooled ratio_B %.4f -> %.4f（移除约 %.1f%% 崩塌量，恢复集中于'
                       ' k=0->1 单实体校准）；三模型 k=8 %.4f/%.4f/%.4f 全落 [1.5,2) 带。残留 ~1.8x '
                       '过量误差 = 结构性成分（实体x类交互不可迁移候选），挂账定位。GAP-4 保持 '
                       'quantified_collapse：崩塌已量化 + 机制已定位（端口 ~%.0f%% + 结构残留），全恢复'
                       '未达成。') % (
        rk['0'], rk['8'], port_frac * 100,
        r8m['qwen3-4b'], r8m['qwen3-14b'], r8m['glm4-9b'], port_frac * 100)
    g4['evidence'] = g4['evidence'] + [
        'p3172 curve: pooled ratio_B(k) k=0:%.4f k=1:%.4f k=2:%.4f k=4:%.4f k=8:%.4f '
        '(k=0 bitwise re-3169, drift=0.00e+00)'
        % (rk['0'], rk['1'], rk['2'], rk['4'], rk['8']),
        'p3172 verdict: borderline_partial_recovery (prereg gate k8<1.5 port_missing / >=2 structural '
        '/ [1.5,2) borderline); per-model k=8 %.4f/%.4f/%.4f all in band; port missing ~%.1f%% of '
        'collapse, residual ~1.8x structural (hung)'
        % (r8m['qwen3-4b'], r8m['qwen3-14b'], r8m['glm4-9b'], port_frac * 100),
    ]
    g4['anchor_sha8']['p3171'] = SHA_ANCHOR['p3171']
    g4['anchor_sha8']['p3172'] = SHA_ANCHOR['p3172']
    assert g4['mechanism_note'] == mn_before, 'mechanism_note must stay byte-identical'
    return gl


# ---------------------------------------------------------------- html render (3168/3170 discipline)
CSS = """
:root { --ink:#1a1d23; --muted:#5b6472; --line:#d9dee6; --bg:#f7f8fa; --card:#ffffff;
  --k:#2456a6; --s:#0f7b6c; --r:#8a4fb8; --mech:#b35900; --ctx:#7a5c12; --ctl:#8c2f39;
  --read:#3a5a8c; --gate:#5a4a8a; --limit:#444; --sxk:#316aa8; --rxs:#2f855a; --rxk:#6b46a0; }
* { box-sizing: border-box; }
body { margin:0; padding:24px 28px 60px; background:var(--bg); color:var(--ink);
  font:14px/1.65 "Segoe UI","Microsoft YaHei","PingFang SC",sans-serif; }
h1 { font-size:22px; margin:0 0 4px; }
h2 { font-size:17px; margin:34px 0 10px; padding-bottom:6px; border-bottom:2px solid var(--line); }
h3 { font-size:14px; margin:0 0 8px; color:var(--muted); font-weight:600; }
.sub { color:var(--muted); font-size:12.5px; }
.mono { font-family:Consolas,"Courier New",monospace; font-size:12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 16px; margin:10px 0; }
.badge { display:inline-block; border-radius:10px; padding:1px 9px; font-size:11.5px; margin-right:6px;
  border:1px solid var(--line); background:#eef1f5; color:var(--ink); }
.b-E2_predictive { background:#e3f0e3; border-color:#9fcaa0; color:#1e5c22; }
.b-E1_repeatable { background:#e5eefa; border-color:#a8c4e4; color:#24558a; }
.b-E3_causal_scoped { background:#f7e8dd; border-color:#ddb28f; color:#8a4b16; }
.b-closed, .b-closed_v1 { background:#e3f0e3; border-color:#9fcaa0; color:#1e5c22; }
.b-open { background:#fbe9e7; border-color:#e0a89f; color:#93321f; }
.b-quantified_collapse { background:#fde8e8; border-color:#e8a0a0; color:#8c1f1f; }
.b-appendix_open { background:#eee8f7; border-color:#c5b1e6; color:#5b3a99; }
.b-appendix_open_crossline { background:#f1eee2; border-color:#d4c99a; color:#6d5f1e; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:10px; }
table { border-collapse:collapse; width:100%; background:var(--card); border:1px solid var(--line); }
th,td { border:1px solid var(--line); padding:6px 9px; text-align:left; vertical-align:top; font-size:13px; }
th { background:#eef1f5; font-weight:600; }
.kv { margin:3px 0; }
.kv b { color:var(--muted); font-weight:600; font-size:12px; margin-right:6px; }
.anchorbox { background:#f4f6f9; border:1px dashed var(--line); border-radius:6px; padding:6px 9px;
  margin-top:8px; font-size:12px; color:#3c4550; word-break:break-all; }
.asserts { color:#52616f; }
.evid { color:#52616f; font-size:12.5px; margin:3px 0; }
.prereg { border-left:3px solid #b35900; padding-left:10px; margin-top:8px; }
.mech { border-left:3px solid #8a4fb8; padding-left:10px; margin-top:8px; white-space:pre-wrap; }
.footer { margin-top:40px; padding-top:12px; border-top:1px solid var(--line); color:var(--muted); font-size:12px; }
"""


def esc(s):
    return html.escape(str(s), quote=False)


CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')


def span(key, val):
    return '<span class="fv" data-k="%s">%s</span>' % (key, esc(val))


def render_feature(f):
    p = f['id']
    rows = []
    rows.append('<div class="card" id="%s">' % p)
    rows.append('<h3>%s &middot; %s <span class="badge b-%s">%s</span>'
                '<span class="badge">%s</span></h3>' % (
                    p, span(p + '.family', f['family']), f['evidence_level'],
                    span(p + '.evidence_level', f['evidence_level']),
                    esc(', '.join(f['model_scope']))))
    rows.append('<div class="kv"><b>statement</b>%s</div>' % span(p + '.statement', f['statement']))
    rows.append('<div class="kv"><b>model_scope</b>%s</div>' % span(p + '.model_scope', ', '.join(f['model_scope'])))
    if f.get('values'):
        vk = ' '.join('<span class="mono">%s=%s</span>' % (esc(k), span('%s.values.%s' % (p, k), fmt(v)))
                      for k, v in f['values'].items())
        rows.append('<div class="kv"><b>values</b>%s</div>' % vk)
    for i, a in enumerate(f['anchors']):
        rows.append('<div class="anchorbox"><b>anchor[%d]</b> src=%s phase=%s role=%s'
                    '<div class="asserts">%s</div></div>' % (
                        i, span('%s.anchors.%d.src' % (p, i), a['src']),
                        span('%s.anchors.%d.phase' % (p, i), a['phase']),
                        span('%s.anchors.%d.role' % (p, i), a['role']),
                        span('%s.anchors.%d.asserts' % (p, i),
                             json.dumps(a['asserts'], ensure_ascii=False, sort_keys=True))))
    rows.append('<div class="kv"><b>n_anchors</b>%s</div>' % span(p + '.n_anchors', str(len(f['anchors']))))
    rows.append('<div class="kv"><b>counter_evidence</b>%s</div>' % span(
        p + '.counter_evidence', ' | '.join(f['counter_evidence']) if f['counter_evidence'] else '(none)'))
    rows.append('<div class="kv"><b>replication</b>%s</div>' % span(p + '.replication', f['replication']))
    rows.append('<div class="kv"><b>scope_limits</b>%s</div>' % span(
        p + '.scope_limits', f['scope_limits'] if f['scope_limits'] else '(none)'))
    rows.append('</div>')
    return '\n'.join(rows)


def render_html(reg, nodes, gl, feats, feat_subset, node_subset, fail_subset, upg_subset,
                meta_sha):
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.3 - RDC/LPF 语义图谱</title><style>%s</style></head>' % CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.3 &mdash; 语义机制图谱（registry v1.2 渲染）</h1>')
    H.append('<div class="sub">Phase 3173 G5-A10 &middot; 零 GPU &middot; registry v1.2 sha8=%s'
             ' &middot; registry v1.1 sha8=%s &middot; 3162 基座 sha8=%s &middot; 增量来源 p3171 res=%s'
             ' / p3172 res=%s &middot; 渲染于 %s</div>' % (
                 span('meta.registry_v12_sha8', meta_sha['reg12']),
                 span('meta.registry_v11_sha8', SHA_ANCHOR['registry_v11']),
                 span('meta.registry3162_sha8', SHA_ANCHOR['registry_3162']),
                 span('meta.p3171_res', SHA_ANCHOR['p3171']),
                 span('meta.p3172_res', SHA_ANCHOR['p3172']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (esc(k), esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg['principles']):
        H.append('<div class="kv"><b>原则%d</b>%s</div>' % (i + 1, esc(pr)))
    H.append('</div>')

    H.append('<h2>图谱节点基座（16 节点，3162 audit disk_verified）</h2>')
    H.append('<table><tr><th>id</th><th>title</th><th>evidence</th><th>checks</th></tr>')
    for n in node_subset:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td><span class="badge b-%s">%s</span>%s</td>'
                 '<td class="mono">%s</td></tr>' % (
                     n['id'], span(n['id'] + '.title', n['title']),
                     n['evidence_level'], span(n['id'] + '.elevel', n['evidence_level']),
                     span(n['id'] + '.status', n['status']),
                     span(n['id'] + '.checks', '%d/%d' % (n['n_pass'], n['n_checks']))))
    H.append('</table>')

    fams = {}
    for f in feat_subset:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.2，22 条）</h2>')
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(render_feature(f))
        H.append('</div>')

    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in upg_subset:
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')

    H.append('<h2>失败账本（F1-F12，可证伪性存档）</h2>')
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in fail_subset:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td class="mono">%s</td><td>%s</td><td class="mono">%s</td></tr>' % (
            fl['id'], span(fl['id'] + '.kind', fl['kind']), span(fl['id'] + '.phase', fl['phase']),
            span(fl['id'] + '.text', fl['text']), span(fl['id'] + '.evidence', fl['evidence'])))
    H.append('</table>')

    H.append('<h2>缺口账本 v1.4（GAP-4 quantified_collapse + 3172 干预定判）</h2>')
    for g in gl['gaps']:
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

    H.append('<h2>附录挂账（5 项，逐条证据锚）</h2><div class="card">')
    for a in gl['appendix']:
        H.append('<div class="kv"><b>%s</b>%s <span class="badge b-%s">%s</span>'
                 '<span class="mono">anchor=%s(%s)</span><div class="evid">%s</div></div>' % (
                     a['id'], span(a['id'] + '.title', a['title']), a['status'],
                     span(a['id'] + '.status', a['status']),
                     span(a['id'] + '.anchor', a['anchor']),
                     span(a['id'] + '.anchor_sha8', a['anchor_sha8']),
                     span(a['id'] + '.note', a['note'])))
    H.append('</div>')

    H.append('<div class="footer">RDC/LPF atlas v1.3 &middot; 渲染自 atlas_registry_v1_2.json (%s)'
             ' + gap ledger v1.4 &middot; 增量来源: Phase 3171 G5-A8 (res %s) + Phase 3172 G5-A9'
             ' (res %s) &middot; 本文件为 Phase 3173 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 meta_sha['reg12'], SHA_ANCHOR['p3171'], SHA_ANCHOR['p3172']))
    H.append('</body></html>')
    return '\n'.join(H)


# ---------------------------------------------------------------- flatten (verify source of truth)
def flatten(reg, nodes, gl):
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
            d['%s.values.%s' % (p, k)] = fmt(v)
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
        d['%s.kind' % fl['id']] = fl['kind']
        d['%s.phase' % fl['id']] = str(fl['phase'])
        d['%s.text' % fl['id']] = fl['text']
        d['%s.evidence' % fl['id']] = str(fl['evidence'])
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
    d['meta.registry_v12_sha8'] = meta_sha_global['reg12']
    d['meta.registry_v11_sha8'] = SHA_ANCHOR['registry_v11']
    d['meta.registry3162_sha8'] = SHA_ANCHOR['registry_3162']
    d['meta.p3171_res'] = SHA_ANCHOR['p3171']
    d['meta.p3172_res'] = SHA_ANCHOR['p3172']
    d['meta.created'] = CREATED_STR
    return d


meta_sha_global = {'reg12': None, 'gl14': None}


# ---------------------------------------------------------------- html verify
def verify_html(html_text, expected):
    pairs = re.findall(r'data-k="([^"]+)"[^>]*>(.*?)</span>', html_text, re.S)
    found = {}
    dups = []
    for k, v in pairs:
        if k in found:
            dups.append(k)
        found[k] = html.unescape(v)
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    mismatch = []
    for k in sorted(set(expected) & set(found)):
        if found[k] != expected[k]:
            mismatch.append((k, found[k][:80], expected[k][:80]))
    return {'n_expected': len(expected), 'n_found': len(found), 'dups': dups,
            'missing': missing, 'extra': extra, 'mismatch': mismatch}


# ---------------------------------------------------------------- main
def main():
    d8 = freeze()
    log('== Phase 3173 g5a10_atlas_v13 %s ==' % ('SMOKE' if SMOKE else 'FULL'))

    # source load + sha assert
    anchors = {}
    for k, p in SRC.items():
        b = open(p, 'rb').read()
        h = sha8(b)
        assert h == SHA_ANCHOR[k], ('source sha mismatch', k, h, SHA_ANCHOR[k])
        anchors[k] = {'path': p, 'sha8': h, 'data': json.loads(b.decode('utf-8'))}
    log('sources: %d files sha-asserted' % len(SRC))

    reg11 = anchors['registry_v11']['data']
    reg2 = anchors['registry_3162']['data']
    nodes = reg2['audit']['nodes']
    gl13 = anchors['gap_ledger_v13']['data']
    r69 = anchors['p3169']['data']
    r71 = anchors['p3171']['data']
    r72 = anchors['p3172']['data']
    rq = anchors['q03']['data']
    assert len(reg11['features']) == 21 and len(nodes) == 16
    assert len(reg11['failures']) == 12 and len(reg11['upgrade_log']) == 3
    assert len(gl13['gaps']) == 4 and len(gl13['appendix']) == 5
    assert reg11['version'] == '1.1'
    assert gl13['version'] == '1.3'
    # G5: FTR-22 runtime asserts + build
    ftr22 = build_ftr22(r72, r69, r71, rq)
    log('G5 FTR-22 asserts OK (k0 re-3169 drift %.1e, k8 band [1.5,2) x3, '
        'port_frac %.4f, p3171 mixed re-asserted)'
        % (ftr22['values']['k0_drift_ratio_vs_3169'], ftr22['values']['port_removed_frac']))

    fadds = build_failures_append(r72, r71)

    # registry v1.2: preserve v1.1 features byte-for-byte, append FTR-22
    reg12 = copy.deepcopy(reg11)
    reg12['version'] = '1.2'
    reg12['supersedes'] = 'atlas_registry_v1_1.json (%s, Phase 3170)' % SHA_ANCHOR['registry_v11']
    reg12['features'] = reg11['features'] + [ftr22]
    assert len(reg12['features']) == 22
    for i in range(21):
        a = json.dumps(reg11['features'][i], ensure_ascii=False, sort_keys=True)
        b = json.dumps(reg12['features'][i], ensure_ascii=False, sort_keys=True)
        assert a == b, ('G3 v1.1 feature not preserved', reg11['features'][i]['id'])
    log('G3 v1.1 features preserved byte-for-byte 21/21; registry v1.2 = 22 features')

    # G7: failures append F13/F14, first 12 byte-for-byte, upgrade_log unchanged
    reg12['failures'] = reg11['failures'] + fadds
    assert len(reg12['failures']) == 14
    for i in range(12):
        a = json.dumps(reg11['failures'][i], ensure_ascii=False, sort_keys=True)
        b = json.dumps(reg12['failures'][i], ensure_ascii=False, sort_keys=True)
        assert a == b, ('G7 failure not preserved', reg11['failures'][i]['id'])
    assert len(reg12['upgrade_log']) == 3, 'upgrade_log must stay at 3 (no E-level upgrades)'
    log('G7 failures F13/F14 appended (12 -> 14), first 12 byte-for-byte; upgrade_log 3 unchanged')

    # gap ledger v1.4
    gl = build_gap_ledger_v14(gl13, r72)
    g4 = [g for g in gl['gaps'] if g['id'] == 'GAP-4'][0]
    g4_old = [g for g in gl13['gaps'] if g['id'] == 'GAP-4'][0]
    assert g4['status'] == 'quantified_collapse' == g4_old['status']
    assert g4['mechanism_note'] == g4_old['mechanism_note'], 'mechanism_note changed'
    assert g4['statement'] != g4_old['statement'] and g4['statement'].startswith(g4_old['statement'])
    assert len(g4['evidence']) == len(g4_old['evidence']) + 2
    assert g4['anchor_sha8']['p3171'] == SHA_ANCHOR['p3171']
    assert g4['anchor_sha8']['p3172'] == SHA_ANCHOR['p3172']
    assert len(g4['anchor_sha8']) == len(g4_old['anchor_sha8']) + 2
    # G4: other gaps + appendix + prereg unchanged byte-for-byte
    assert g4['prereg'] == g4_old['prereg'], 'prereg changed'
    for gid in ('GAP-1', 'GAP-2', 'GAP-3'):
        a = json.dumps([g for g in gl13['gaps'] if g['id'] == gid][0], ensure_ascii=False, sort_keys=True)
        b = json.dumps([g for g in gl['gaps'] if g['id'] == gid][0], ensure_ascii=False, sort_keys=True)
        assert a == b, ('G4 gap changed', gid)
    for i in range(5):
        a = json.dumps(gl13['appendix'][i], ensure_ascii=False, sort_keys=True)
        b = json.dumps(gl['appendix'][i], ensure_ascii=False, sort_keys=True)
        assert a == b, ('G4 appendix changed', gl13['appendix'][i]['id'])
    log('G4 GAP-4 increment asserted (statement extended, evidence +2, anchors +2, '
        'mechanism_note byte-identical, status unchanged); GAP-1/2/3 + appendix 5 unchanged')

    # write registry v1.2 + gap ledger v1.4 (html meta needs their shas)
    suffix = 'smoke_' if SMOKE else ''
    reg12_path = os.path.join(OUTDIR, suffix + 'atlas_registry_v1_2.json')
    gl_path = os.path.join(OUTDIR, suffix + 'gap_ledger_v1_4.json')
    with io.open(reg12_path, 'w', encoding='utf-8') as f:
        json.dump(reg12, f, ensure_ascii=False, indent=1)
    with io.open(gl_path, 'w', encoding='utf-8') as f:
        json.dump(gl, f, ensure_ascii=False, indent=1)
    meta_sha_global['reg12'] = sha8_file(reg12_path)
    meta_sha_global['gl14'] = sha8_file(gl_path)
    log('registry v1.2 written sha8 %s | gap ledger v1.4 written sha8 %s'
        % (meta_sha_global['reg12'], meta_sha_global['gl14']))

    # subsets
    feats_all = reg12['features']
    feats = feats_all
    fails = reg12['failures']
    upgs = list(enumerate(reg12['upgrade_log']))
    nodes_sub = nodes[:]
    if SMOKE:
        keep_ids = ('FTR-01', 'FTR-21', 'FTR-22')
        feats = [f for f in feats_all if f['id'] in keep_ids]
        nodes_sub = nodes[:4]
        fails = fails[:2]
        upgs = upgs[:1]

    flat = flatten(dict(reg12, features=feats_all, failures=reg12['failures'],
                        upgrade_log=reg12['upgrade_log']),
                   nodes_sub, gl)
    if SMOKE:
        keep = set()
        for f in feats:
            keep |= set(k for k in flat if k.startswith(f['id'] + '.'))
        for n in nodes_sub:
            keep |= set(k for k in flat if k.startswith(n['id'] + '.'))
        for i, u in upgs:
            keep |= set(k for k in flat if k.startswith('UPG.%d.' % i))
        for fl in fails:
            keep |= set(k for k in flat if k.startswith(fl['id'] + '.'))
        keep |= set(k for k in flat if k.startswith(('GAP-', 'APPX-', 'meta.')))
        flat = {k: v for k, v in flat.items() if k in keep}

    html_text = render_html(reg12, nodes_sub, gl, feats, feats, nodes_sub, fails, upgs,
                            meta_sha_global)

    # G6 no external resources
    assert '<link' not in html_text and '<script' not in html_text, 'external resource found'
    assert 'http://' not in html_text and 'https://' not in html_text, 'external url found'
    log('G6 no-external OK')

    # G1 field check
    v = verify_html(html_text, flat)
    log('G1 field check: expected=%d found=%d missing=%d extra=%d mismatch=%d dups=%d' % (
        v['n_expected'], v['n_found'], len(v['missing']), len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'], ('missing keys', v['missing'][:10])
    assert not v['extra'], ('extra keys', v['extra'][:10])
    assert not v['mismatch'], ('mismatch', v['mismatch'][:5])
    assert not v['dups'], ('dup keys', v['dups'][:5])

    # write html
    html_path = os.path.join(OUTDIR, suffix + 'atlas_v1_3.html')
    with io.open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    html_sha = sha8_file(html_path)
    log('html written %s (%d B, sha8 %s)' % (os.path.basename(html_path), len(html_text.encode('utf-8')), html_sha))

    verdict = ('g5a10_atlas_v13|features_22|failures_14|v11_preserved_21|html_fields_%d_ok|'
               'gap4_statement_extended_3172|port_recovery_1.8002|mechanism_note_rendered') % v['n_found']

    # ---------------- result + seal
    summary = {
        'phase': 3173, 'name': 'g5a10_atlas_v13', 'smoke': SMOKE,
        'design_sha8': d8, 'verdict': verdict,
        'registry_v12_file': os.path.basename(reg12_path), 'registry_v12_sha8': meta_sha_global['reg12'],
        'gap_ledger_v14_file': os.path.basename(gl_path), 'gap_ledger_v14_sha8': meta_sha_global['gl14'],
        'html_file': os.path.basename(html_path), 'html_sha8': html_sha,
        'field_check': {'n_expected': v['n_expected'], 'n_found': v['n_found'],
                        'missing': len(v['missing']), 'extra': len(v['extra']),
                        'mismatch': len(v['mismatch']), 'dups': len(v['dups'])},
        'counts': {'features': len(feats_all), 'nodes': len(nodes_sub), 'failures': len(fails),
                   'upgrades': len(upgs), 'gaps': len(gl['gaps']), 'appendix': len(gl['appendix'])},
        'gap_status': {g['id']: g['status'] for g in gl['gaps']},
        'ftr22_values': ftr22['values'],
        'sources': {k: a['sha8'] for k, a in anchors.items()},
    }
    raw = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
    res8 = sha8(raw.encode('utf-8'))
    mid = json.dumps(dict(summary, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
    seal8 = sha8(mid.encode('utf-8'))
    rpath = os.path.join(OUTDIR, suffix + 'result.json')
    with io.open(rpath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dict(summary, res_sha8=res8, seal_sha8=seal8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('result written res=%s seal=%s' % (res8, seal8))
    log('verdict: %s' % verdict)

    with io.open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
