# -*- coding: utf-8 -*-
"""Phase 3170: G5-A7 atlas v1.1 incremental update (zero GPU).

Updates the atlas with the Phase 3169 GAP-4 verdict (collapse_confirmed,
ratio_B=2.5388):
  (a) registry v1 -> v1.1: append FTR-21 (OOV-class readout collapse), the
      existing 20 features preserved byte-for-byte;
  (b) gap ledger v1 -> v1.1: GAP-4 open -> quantified_collapse with 3169
      evidence anchors, prereg status -> executed_collapse_confirmed;
  (c) three-tier gradient (entity/combination generalizes, class axis does
      not) recorded in FTR-21 scope_limits;
  (d) atlas_v1.html -> atlas_v1_1.html re-render with per-field data-k
      verification (same discipline as 3168).

Gates:
  G1 html field check: every data-k span equals flattened json (0 missing/extra/mismatch/dup)
  G2 feature cards rendered (21 full / 3 smoke)
  G3 v1 features preserved byte-for-byte 20/20
  G4 GAP-4 flip asserted (open -> quantified_collapse); other gaps + appendix unchanged
  G5 FTR-21 runtime asserts: 3169 gate values + D3 bitwise + D4 zero drift + q03 cross-line drift < 1e-6
  G6 no external resources in html

SMOKE: env P3170_SMOKE=1 renders a subset through the full pipeline.
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
OUTDIR = os.path.join(SRC_DIR, 'phase3170', 'g5a7_atlas_v11')
SMOKE = os.environ.get('P3170_SMOKE', '') == '1'

P3169_DIR = os.path.join(SRC_DIR, 'phase3169', 'g5a6_oov_panel')

SRC = {
    'p3169': os.path.join(P3169_DIR, 'result.json'),
    'npz_4b': os.path.join(P3169_DIR, 'collect_qwen3-4b.npz'),
    'npz_14b': os.path.join(P3169_DIR, 'collect_qwen3-14b.npz'),
    'npz_glm4': os.path.join(P3169_DIR, 'collect_glm4-9b.npz'),
    'registry_v1': os.path.join(SRC_DIR, r'phase3167\g5a4_feature_registry\atlas_registry_v1.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
    'gap_ledger_v1': os.path.join(SRC_DIR, r'phase3168\g5a5_atlas_render\gap_ledger_v1.json'),
    'q03': os.path.join(BASE, r'tests\deepseek\result\q03_result.json'),
    'p3168_res': os.path.join(SRC_DIR, r'phase3168\g5a5_atlas_render\result.json'),
}

SHA_ANCHOR = {
    'p3169': '5b51c2c1',
    'npz_4b': '36eb4ff0', 'npz_14b': '97f98575', 'npz_glm4': '9cc3f8b4',
    'registry_v1': 'f207aa8d', 'registry_3162': '00f15e98',
    'gap_ledger_v1': 'f92ed0cd',
    'q03': '57827730', 'p3168_res': 'd229a6e6',
}

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']

DESIGN = {
    'phase': 3170, 'name': 'g5a7_atlas_v11', 'zero_gpu': True,
    'sources': {k: v.replace(BASE, '') for k, v in SRC.items()},
    'updates': {
        'registry': 'v1 -> v1.1: append FTR-21 (OOV-class readout collapse, family=limit, '
                    'E2_predictive, anchors p3169 primary + p3169 device D3/D4 + q03 cross-line '
                    'pooled replication); features[0:20] preserved byte-for-byte; failures 12 and '
                    'upgrade_log 3 unchanged.',
        'gap_ledger': 'v1 -> v1.1: GAP-4 status open -> quantified_collapse; statement extended with '
                      'the 3169 quantification; evidence += 3 runtime-rendered items from p3169 '
                      '(pooled ratio_B, per-model B>2 & A<1, device D3/D4); anchor_sha8 += p3169; '
                      'prereg.status -> executed_collapse_confirmed; GAP-1/2/3 and appendix 5 items '
                      'unchanged byte-for-byte.',
        'html': 'atlas_v1.html -> atlas_v1_1.html re-render (single file, inline CSS, zero external '
                'resources) with per-field data-k verification against flattened v1.1 sources.',
        'ftr21_scope_limits': 'three-tier gradient: E_newent (new entity, seen class) 0.74/0.69/0.78 '
                              'mid -> entity axis generalizes; A (combo held-out) ratio<1 -> '
                              'combination axis generalizes; B (class leave-out) ratio>2 -> class '
                              'axis does not generalize (readout hard boundary).',
    },
    'ftr21_gate': 'collapse_confirmed iff pooled ratio_B >= 2.0 (prereg 3169 gate); '
                  'borderline 1.5-2.0; generalization holds if <= 2.0 would close GAP-4.',
    'gates': {
        'G1': 'html data-k field check: missing/extra/mismatch/dup all 0',
        'G2': 'feature cards rendered (21 full / 3 smoke)',
        'G3': 'v1 features preserved byte-for-byte (20/20)',
        'G4': 'GAP-4 flip asserted; other gaps + appendix unchanged',
        'G5': 'FTR-21 runtime asserts pass (gate values, D3/D4, q03 drift<1e-6)',
        'G6': 'no external resources: no <link, no <script, no http(s):// in html',
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


def get_path(data, dotted):
    cur = data
    for part in dotted.split('.'):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


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


# ---------------------------------------------------------------- FTR-21 build (values runtime-rendered)
def build_ftr21(r69, rq):
    g = r69['gate']
    assert_val('gate.ratio_B', g['ratio_B'], 2.5388115)
    assert_val('gate.pooled_E_oov_B', g['pooled_E_oov_B'], 0.947866453)
    assert_val('gate.pooled_E_seen_B', g['pooled_E_seen_B'], 0.373350465)
    assert r69['res_sha8'] == '49430a39' and r69['seal_sha8'] == '018c6024'
    assert r69['verdict'] == 'g5a6_oov_panel|full|3_models|ratio_B=2.5388|collapse_confirmed_gap4_open'
    rb = {m: r69['per_model'][m]['B']['ratio'] for m in MODELS}
    ra = {m: r69['per_model'][m]['A']['ratio'] for m in MODELS}
    en = {m: r69['per_model'][m]['B']['E_newent'] for m in MODELS}
    for m in MODELS:
        assert rb[m] > 2.0, ('ratio_B must exceed 2', m, rb[m])
        assert ra[m] < 1.0, ('ratio_A must be < 1', m, ra[m])
        d3 = r69['per_model'][m]
        assert d3['d3_bitwise_rows'] == 738 and d3['d3_mismatch'] == 0, ('D3', m)
        assert d3['anchor']['ok'] is True, ('D4 ok', m)
        assert list(d3['anchor']['drift']) == [0.0, 0.0, 0.0], ('D4 drift', m, d3['anchor']['drift'])
    # cross-line pooled replication: q03 pooled_mean vs 3169 pooled_E_seen_B
    drift = abs(rq['summary']['pooled_mean'] - g['pooled_E_seen_B'])
    assert drift < 1e-6, ('q03 pooled cross-line drift', drift)
    assert_val('q03.min_E_x', rq['summary']['min_E_x'], 6.632306178410848)

    stmt = ('谱外类别读出崩塌量化：类别级留出（口径 B，谱外类别行完全缺席训练）下 pooled '
            'E_oov=%.6f vs pooled E_seen=%.6f，ratio_B=%.4f >= 2 门；三模型单模型 ratio_B '
            '%.2f/%.2f/%.2f 全 > 2。组合留出口径（A，训练含谱外类部分组合）ratio %.2f/%.2f/%.2f '
            '全 < 1——「未见组合」与「未见类别」是两个 regime（双口径结构性对照）。') % (
        g['pooled_E_oov_B'], g['pooled_E_seen_B'], g['ratio_B'],
        rb['qwen3-4b'], rb['qwen3-14b'], rb['glm4-9b'],
        ra['qwen3-4b'], ra['qwen3-14b'], ra['glm4-9b'])

    ftr = {
        'id': 'FTR-21',
        'family': 'limit',
        'statement': stmt,
        'evidence_level': 'E2_predictive',
        'model_scope': list(MODELS),
        'anchors': [
            {'src': 'p3169', 'phase': 3169, 'role': 'primary_measurement', 'tags': ['cross_model', 'held_out'],
             'asserts': {'gate.ratio_B': 2.5388115, 'gate.pooled_E_oov_B': 0.947866453,
                         'gate.pooled_E_seen_B': 0.373350465,
                         'per_model.qwen3-4b.B.ratio': rb['qwen3-4b'],
                         'per_model.qwen3-14b.B.ratio': rb['qwen3-14b'],
                         'per_model.glm4-9b.B.ratio': rb['glm4-9b'],
                         'res_sha8': '49430a39', 'seal_sha8': '018c6024'}},
            {'src': 'p3169', 'phase': 3169, 'role': 'device_bitwise_and_q03_anchor', 'tags': ['device'],
             'asserts': {'per_model.qwen3-4b.d3_mismatch': 0, 'per_model.qwen3-14b.d3_mismatch': 0,
                         'per_model.glm4-9b.d3_mismatch': 0,
                         'per_model.qwen3-4b.anchor.ok': True, 'per_model.qwen3-14b.anchor.ok': True,
                         'per_model.glm4-9b.anchor.ok': True,
                         'per_model.qwen3-4b.anchor.drift': [0.0, 0.0, 0.0],
                         'per_model.qwen3-14b.anchor.drift': [0.0, 0.0, 0.0],
                         'per_model.glm4-9b.anchor.drift': [0.0, 0.0, 0.0]}},
            {'src': 'q03', 'phase': 'N-Q03', 'role': 'cross_line_pooled_replication', 'tags': ['held_out'],
             'asserts': {'summary.pooled_mean': rq['summary']['pooled_mean'],
                         'summary.min_E_x': rq['summary']['min_E_x']}},
        ],
        'counter_evidence': [
            'prereg 词表笔误更正：3168 prereg 所列「水果/金属」实为已见类——谱外性以双重排除冻结'
            '（乐器/天气/运动/电器，避开 3152 六类与 2881 十族），判定门不变',
            '行为读出层证据（E_read 归一化 MSE），非权重级/激活级干预证据',
        ],
        'replication': 'Q03 协议 verbatim（B4 ridge one-hot 特征空间 lam=1e-3、k* 读位、batch=1 bf16 '
                       '采集 H fp16）+ 类别级留出切分 B（谱外类别行完全缺席训练）x 3 种子；装置门 '
                       'D3=已见行 H 与 3152/3151 封存 npz 逐位（738/738 行 x3）、D4=已见子面板 '
                       'cols=51 复算 q03 锚漂移 0.000e+00 x3。',
        'values': {
            'ratio_B_pooled': g['ratio_B'],
            'pooled_E_oov_B': g['pooled_E_oov_B'],
            'pooled_E_seen_B': g['pooled_E_seen_B'],
            'ratio_B_qwen3_4b': rb['qwen3-4b'],
            'ratio_B_qwen3_14b': rb['qwen3-14b'],
            'ratio_B_glm4_9b': rb['glm4-9b'],
            'ratio_A_qwen3_4b': ra['qwen3-4b'],
            'ratio_A_qwen3_14b': ra['qwen3-14b'],
            'ratio_A_glm4_9b': ra['glm4-9b'],
            'E_newent_qwen3_4b': en['qwen3-4b'],
            'E_newent_qwen3_14b': en['qwen3-14b'],
            'E_newent_glm4_9b': en['glm4-9b'],
            'd3_bitwise_rows_total': 3 * 738,
            'd3_mismatch_total': 0,
            'q03_pooled_crossline_drift': drift,
        },
        'scope_limits': '三档梯度：E_newent（新实体旧类别）%.2f/%.2f/%.2f 居中——实体轴可泛化；'
                        'A 口径 ratio %.2f/%.2f/%.2f 全 <1——组合轴可泛化；B 口径 ratio %.2f/%.2f/%.2f '
                        '全 >2——类别轴不可泛化（读出编码硬边界）。范围=中文面板+单一模板族+行为读出层。' % (
                            en['qwen3-4b'], en['qwen3-14b'], en['glm4-9b'],
                            ra['qwen3-4b'], ra['qwen3-14b'], ra['glm4-9b'],
                            rb['qwen3-4b'], rb['qwen3-14b'], rb['glm4-9b']),
    }
    return ftr


# ---------------------------------------------------------------- gap ledger v1.1
def build_gap_ledger_v11(gl1, r69, q03_pooled_drift):
    gl = copy.deepcopy(gl1)
    gl['schema'] = 'rdc_atlas_gap_ledger_v1_1'
    gl['version'] = '1.1'
    gl['updated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    gl['supersedes'] = 'gap_ledger_v1.json (f92ed0cd, Phase 3168)'
    g4 = [g for g in gl['gaps'] if g['id'] == 'GAP-4'][0]
    assert g4['status'] == 'open', ('GAP-4 must be open in v1', g4['status'])
    pr = g4['prereg']
    # unchanged-by-design fields asserted, then flip only the allowed ones
    assert pr['phase'] == 3169 and pr['status'] == 'preregistered_not_executed'
    g4['status'] = 'quantified_collapse'
    g4['statement'] = (g4['statement'] +
                       ' 3169 定判：谱外类别读出崩塌确认且量化——pooled ratio_B=2.5388 '
                       '（E_oov=0.9479 vs E_seen=0.3733），三模型 2.83/2.13/2.71 全 >2；'
                       '组合口径 A ratio 0.91/0.85/0.95 全 <1（未见组合不等于未见类别）；'
                       '三档梯度=实体轴/组合轴可泛化、类别轴不可泛化（读出编码硬边界）。')
    g4['evidence'] = g4['evidence'] + [
        'p3169 gate: pooled ratio_B=%.9f pooled E_oov=%.9f E_seen=%.9f (prereg 门 >=2 => collapse)'
        % (r69['gate']['ratio_B'], r69['gate']['pooled_E_oov_B'], r69['gate']['pooled_E_seen_B']),
        'p3169 per-model: ratio_B %.4f/%.4f/%.4f all>2; ratio_A %.4f/%.4f/%.4f all<1; '
        'E_newent %.4f/%.4f/%.4f mid-tier'
        % tuple([r69['per_model'][m]['B']['ratio'] for m in MODELS] +
                [r69['per_model'][m]['A']['ratio'] for m in MODELS] +
                [r69['per_model'][m]['B']['E_newent'] for m in MODELS]),
        'p3169 device: D3 已见行 H 与 3152/3151 封存 npz 逐位 738/738 x3 (0 mismatch); '
        'D4 已见子面板复算 q03 锚漂移 0.000e+00 x3; q03 pooled cross-line drift=%.3e'
        % q03_pooled_drift,
    ]
    g4['anchor_sha8']['p3169'] = SHA_ANCHOR['p3169']
    pr['status'] = 'executed_collapse_confirmed'
    return gl


# ---------------------------------------------------------------- html render (3168 discipline)
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


def render_html(reg, nodes, gl, feats, feat_subset, node_subset, fail_subset, upg_subset):
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.1 - RDC/LPF 语义图谱</title><style>%s</style></head>' % CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.1 &mdash; 语义机制图谱（registry v1.1 渲染）</h1>')
    H.append('<div class="sub">Phase 3170 G5-A7 &middot; 零 GPU &middot; registry v1.1 sha8=%s'
             ' &middot; v1 sha8=%s &middot; 3162 基座 sha8=%s &middot; GAP-4 定判来源 p3169 res=%s'
             ' &middot; 渲染于 %s</div>' % (
                 span('meta.registry_v11_sha8', REG_V11_SHA8),
                 span('meta.registry_v1_sha8', SHA_ANCHOR['registry_v1']),
                 span('meta.registry3162_sha8', SHA_ANCHOR['registry_3162']),
                 span('meta.p3169_res', SHA_ANCHOR['p3169']),
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
    H.append('<h2>跨模型稳定特征登记表（registry v1.1，21 条）</h2>')
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

    H.append('<h2>缺口账本 v1.1（GAP-4 quantified_collapse@3169）</h2>')
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

    H.append('<div class="footer">RDC/LPF atlas v1.1 &middot; 渲染自 atlas_registry_v1_1.json (%s)'
             ' + gap ledger v1.1 &middot; 增量来源: Phase 3169 G5-A6 (res %s) &middot; '
             '本文件为 Phase 3170 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 REG_V11_SHA8, SHA_ANCHOR['p3169']))
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
    for a in gl['appendix']:
        p = a['id']
        d[p + '.title'] = a['title']
        d[p + '.status'] = a['status']
        d[p + '.note'] = a['note']
        d[p + '.anchor'] = a['anchor']
        d[p + '.anchor_sha8'] = a['anchor_sha8']
    d['meta.registry_v11_sha8'] = REG_V11_SHA8
    d['meta.registry_v1_sha8'] = SHA_ANCHOR['registry_v1']
    d['meta.registry3162_sha8'] = SHA_ANCHOR['registry_3162']
    d['meta.p3169_res'] = SHA_ANCHOR['p3169']
    d['meta.created'] = CREATED_STR
    return d


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
    log('== Phase 3170 g5a7_atlas_v11 %s ==' % ('SMOKE' if SMOKE else 'FULL'))

    # source load + sha assert (includes 1.6GB npz set)
    anchors = {}
    for k, p in SRC.items():
        b = open(p, 'rb').read()
        h = sha8(b)
        assert h == SHA_ANCHOR[k], ('source sha mismatch', k, h, SHA_ANCHOR[k])
        anchors[k] = {'path': p, 'sha8': h, 'data': json.loads(b.decode('utf-8')) if not k.startswith('npz_') else None}
    log('sources: %d files sha-asserted (incl 3 collect npz)' % len(SRC))

    reg1 = anchors['registry_v1']['data']
    reg2 = anchors['registry_3162']['data']
    nodes = reg2['audit']['nodes']
    gl1 = anchors['gap_ledger_v1']['data']
    r69 = anchors['p3169']['data']
    rq = anchors['q03']['data']
    assert len(reg1['features']) == 20 and len(nodes) == 16
    assert len(reg1['failures']) == 12 and len(reg1['upgrade_log']) == 3
    assert len(gl1['gaps']) == 4 and len(gl1['appendix']) == 5

    # G5: FTR-21 runtime asserts + build
    ftr21 = build_ftr21(r69, rq)
    log('G5 FTR-21 asserts OK (gate values, D3 0-mismatch x3, D4 drift 0 x3, q03 drift %.3e)'
        % ftr21['values']['q03_pooled_crossline_drift'])

    # registry v1.1: preserve v1 features byte-for-byte, append FTR-21
    reg11 = copy.deepcopy(reg1)
    reg11['version'] = '1.1'
    reg11['supersedes'] = 'atlas_registry_v1.json (%s, Phase 3167)' % SHA_ANCHOR['registry_v1']
    reg11['features'] = reg1['features'] + [ftr21]
    assert len(reg11['features']) == 21
    for i in range(20):
        a = json.dumps(reg1['features'][i], ensure_ascii=False, sort_keys=True)
        b = json.dumps(reg11['features'][i], ensure_ascii=False, sort_keys=True)
        assert a == b, ('G3 v1 feature not preserved', reg1['features'][i]['id'])
    log('G3 v1 features preserved byte-for-byte 20/20; registry v1.1 = 21 features')

    # gap ledger v1.1
    gl = build_gap_ledger_v11(gl1, r69, ftr21['values']['q03_pooled_crossline_drift'])
    g4 = [g for g in gl['gaps'] if g['id'] == 'GAP-4'][0]
    assert g4['status'] == 'quantified_collapse'
    assert g4['prereg']['status'] == 'executed_collapse_confirmed'
    assert g4['anchor_sha8']['p3169'] == SHA_ANCHOR['p3169']
    # G4: other gaps + appendix unchanged byte-for-byte
    for gid in ('GAP-1', 'GAP-2', 'GAP-3'):
        a = json.dumps([g for g in gl1['gaps'] if g['id'] == gid][0], ensure_ascii=False, sort_keys=True)
        b = json.dumps([g for g in gl['gaps'] if g['id'] == gid][0], ensure_ascii=False, sort_keys=True)
        assert a == b, ('G4 gap changed', gid)
    for i in range(5):
        a = json.dumps(gl1['appendix'][i], ensure_ascii=False, sort_keys=True)
        b = json.dumps(gl['appendix'][i], ensure_ascii=False, sort_keys=True)
        assert a == b, ('G4 appendix changed', gl1['appendix'][i]['id'])
    log('G4 GAP-4 flip asserted (open -> quantified_collapse); GAP-1/2/3 + appendix 5 unchanged')

    # write registry v1.1 + gap ledger v1.1 (html needs their shas in meta)
    suffix = 'smoke_' if SMOKE else ''
    reg11_path = os.path.join(OUTDIR, suffix + 'atlas_registry_v1_1.json')
    gl_path = os.path.join(OUTDIR, suffix + 'gap_ledger_v1_1.json')
    with io.open(reg11_path, 'w', encoding='utf-8') as f:
        json.dump(reg11, f, ensure_ascii=False, indent=1)
    with io.open(gl_path, 'w', encoding='utf-8') as f:
        json.dump(gl, f, ensure_ascii=False, indent=1)
    global REG_V11_SHA8
    REG_V11_SHA8 = sha8_file(reg11_path)
    gl_sha = sha8_file(gl_path)
    log('registry v1.1 written sha8 %s | gap ledger v1.1 written sha8 %s' % (REG_V11_SHA8, gl_sha))

    # subsets
    feats_all = reg11['features']
    feats = feats_all
    fails = reg11['failures']
    upgs = list(enumerate(reg11['upgrade_log']))
    nodes_sub = nodes[:]
    if SMOKE:
        keep_ids = ('FTR-01', 'FTR-20', 'FTR-21')
        feats = [f for f in feats_all if f['id'] in keep_ids]
        nodes_sub = nodes[:4]
        fails = fails[:2]
        upgs = upgs[:1]

    flat = flatten(dict(reg11, features=feats_all, failures=reg11['failures'],
                        upgrade_log=reg11['upgrade_log']),
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

    html_text = render_html(reg11, nodes_sub, gl, feats_all, feats, nodes_sub, fails, upgs)

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
    html_path = os.path.join(OUTDIR, suffix + 'atlas_v1_1.html')
    with io.open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    html_sha = sha8_file(html_path)
    log('html written %s (%d B, sha8 %s)' % (os.path.basename(html_path), len(html_text.encode('utf-8')), html_sha))

    n_closed = sum(1 for g in gl['gaps'] if g['status'].startswith('closed'))
    verdict = 'g5a7_atlas_v11|features_%d|v1_preserved_20|html_fields_%d_ok|gap4_quantified_collapse|ratio_B=2.5388|other_gaps_unchanged' % (
        len(feats_all), v['n_found'])

    # ---------------- result + seal
    summary = {
        'phase': 3170, 'name': 'g5a7_atlas_v11', 'smoke': SMOKE,
        'design_sha8': d8, 'verdict': verdict,
        'registry_v11_file': os.path.basename(reg11_path), 'registry_v11_sha8': REG_V11_SHA8,
        'gap_ledger_v11_file': os.path.basename(gl_path), 'gap_ledger_v11_sha8': gl_sha,
        'html_file': os.path.basename(html_path), 'html_sha8': html_sha,
        'field_check': {'n_expected': v['n_expected'], 'n_found': v['n_found'],
                        'missing': len(v['missing']), 'extra': len(v['extra']),
                        'mismatch': len(v['mismatch']), 'dups': len(v['dups'])},
        'counts': {'features': len(feats_all), 'nodes': len(nodes_sub), 'failures': len(fails),
                   'upgrades': len(upgs), 'gaps': len(gl['gaps']), 'appendix': len(gl['appendix'])},
        'gap_status': {g['id']: g['status'] for g in gl['gaps']},
        'ftr21_values': ftr21['values'],
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
