# -*- coding: utf-8 -*-
"""Phase 3168: G5-A5 atlas v1 render + gap ledger rewrite (zero GPU).

Renders atlas_registry_v1.json (3167, f207aa8d) -> atlas_v1.html single-file
with per-field data-k markers; rewrites the atlas gap ledger (gaps 1-3 closed,
gap 4 open with preregistered experiment design; 5 appendix items).

Gates:
  G1 html field check: every data-k span equals flattened json value (mismatch 0)
  G2 feature cards 20/20 (smoke subset)
  G3 nodes 16/16 (smoke subset)
  G4 failures 12/12 (smoke subset)
  G5 upgrades 3/3 (smoke subset)
  G6 gaps 4 + appendix 5, closed items carry evidence anchors, gap4 carries prereg
  G7 no external resources in html

SMOKE: env P3168_SMOKE=1 renders a subset (FTR-01..03, N00..N03, F1-F2, UPG.0)
through the full pipeline into smoke_atlas_v1.html.
"""
import io
import os
import re
import json
import html
import hashlib
import datetime

BASE = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(BASE, r'tests\glm5\result\rdc_query_construction_20260913')
OUTDIR = os.path.join(SRC_DIR, 'phase3168', 'g5a5_atlas_render')
SMOKE = os.environ.get('P3168_SMOKE', '') == '1'

SRC = {
    'registry_v1': os.path.join(SRC_DIR, r'phase3167\g5a4_feature_registry\atlas_registry_v1.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
    # gap evidence anchors (values extracted at runtime, files asserted below)
    'p3164a': os.path.join(SRC_DIR, r'phase3164\g5a2_c_steer\result_summary.json'),
    'p3155': os.path.join(SRC_DIR, r'phase3155\g2p1_relation_family_operator_separability\summary\result_summary.json'),
    'p3164b': os.path.join(SRC_DIR, r'phase3164\g5a2b_position_shift_cross_model\summary\result_summary.json'),
    'p3164c': os.path.join(SRC_DIR, r'phase3164\g5a2c_massive_cross_model\summary\result_summary.json'),
    'p3159': os.path.join(SRC_DIR, r'phase3159\g4p2_equivalence_dynamics\summary\result_summary.json'),
    'p3160': os.path.join(SRC_DIR, r'phase3160\g4p3_consumption_mechanism\summary\result_summary.json'),
    'p3161': os.path.join(SRC_DIR, r'phase3161\g4p4_head_attribution\summary\result_summary.json'),
    'p3163': os.path.join(SRC_DIR, r'phase3163\g4p5_redundancy\summary\result_summary.json'),
    'p3165': os.path.join(SRC_DIR, r'phase3165\g5a3_family_alignment\result.json'),
    'p3166': os.path.join(SRC_DIR, r'phase3166\g5a3b_logic_direction\result.json'),
    'q03': os.path.join(BASE, r'tests\deepseek\result\q03_result.json'),
    'p3151': os.path.join(SRC_DIR, r'phase3151\g1p1_combo_additive_vs_interaction\result_rev3151b.json'),
    'agate': os.path.join(BASE, r'research\deepseek\atlas\a_gate_closure_v1.json'),
}

SHA_ANCHOR = {
    'registry_v1': 'f207aa8d', 'registry_3162': '00f15e98',
    'p3164a': 'e7a1f67d', 'p3155': '46a5034d', 'p3164b': '2d5329dc', 'p3164c': 'b5aaf29b',
    'p3159': 'a552f590', 'p3160': '2d97d24b', 'p3161': 'ec0e4488', 'p3163': 'db151a50',
    'p3165': '511d9b13', 'p3166': '448af595',
    'q03': '57827730', 'p3151': 'dd0cc176', 'agate': '24c60160',
}

DESIGN = {
    'phase': 3168, 'name': 'g5a5_atlas_render', 'zero_gpu': True,
    'sources': {k: v.replace(BASE, '') for k, v in SRC.items()},
    'render': {
        'output': 'atlas_v1.html', 'single_file': True, 'inline_css': True,
        'field_marker': 'data-k spans; verify extracts and compares with flattened json',
        'sections': ['header', 'taxonomy_principles', 'nodes16', 'features20',
                     'upgrades3', 'failures12', 'gap_ledger', 'footer'],
    },
    'gap_ledger': {
        'schema': 'rdc_atlas_gap_ledger_v1',
        'gaps': [
            {'id': 'GAP-1', 'title': '跨模型同口径', 'status': 'closed', 'closed_at_phase': 3164,
             'evidence_anchors': ['p3164a', 'p3164b', 'p3164c'],
             'statement': '机制链核心读数（C_steer 零结果 / RoPE 输出相对性 / massive 上下文门控）'
                          '在主线三模型同口径复现，缺口①关闭。'},
            {'id': 'GAP-2', 'title': '机制链闭环', 'status': 'closed', 'closed_at_phase': 3163,
             'evidence_anchors': ['p3159', 'p3160', 'p3161', 'p3163'],
             'statement': '3159 动力学破坏 → 3160 attention 再分配 → 3161 头归因排除单点执行者'
                          ' → 3163 整块恒等化不改变消耗：消耗=残差流全流分布式性质，机制链判闭。'},
            {'id': 'GAP-3', 'title': '跨族连接', 'status': 'closed_v1', 'closed_at_phase': '3165-3166',
             'evidence_anchors': ['p3165', 'p3166'],
             'statement': '三族编码子空间几何普查 v1：K×S separable×3（67.0-74.0 度）、'
                          'R×S_class not_confounded×3、R×K_readout separable×3；'
                          '唯一弱混合=14b R×K_entity 26.6 度（2 共享维）已登记为真信号。'},
            {'id': 'GAP-4', 'title': '谱外迁移证伪', 'status': 'open',
             'evidence_anchors': ['q03', 'p3151'],
             'statement': '未见类别读出泛化未证：Q03 E_read 门 0/3 过（min_E_x=6.63x 门）；'
                          '3151 v3 谱外类别崩塌（b4 最差折=shuiguo 1.2226，v4_s3_b4_mean=0.6114）。'
                          '待谱外类别对照实验定崩塌边界。',
             'prereg': {
                 'phase': 3169, 'name': 'G5-A6 谱外类别面板',
                 'hypothesis': '模型对训练词表面板外的类别（谱外类别）不具组合读出泛化：'
                               '同协议下谱外类别 E_read 显著高于已见类别。',
                 'protocol': '复用 Q03 协议 verbatim（同构模板/同 k* 读位/同判定层=行为读出），'
                             '仅替换类别词集：4 个谱外类别 x 每类 8-10 实体（水果/金属/乐器/天气，'
                             '均不在 2881 词表）；对照臂=已见类别重测作 paired 基线；三模型。',
                 'gate': '谱外 pooled E_read <= 2x 已见 pooled E_read => 泛化成立（缺口④可关）；'
                         '>= 2x => 崩塌确认（量化崩塌比，缺口④保持 open）；'
                         '1.5-2x => borderline（换种子重测一次定判）。',
                 'status': 'preregistered_not_executed'},
             },
        ],
        'appendix': [
            {'id': 'APPX-1', 'title': '14b R x K_entity 2 共享维定位',
             'anchor': 'p3166', 'status': 'appendix_open',
             'note': 'eff_ge05=2 仅 14b；定位共享方向并解释其来源。'},
            {'id': 'APPX-2', 'title': 'S_attr/S_syntax 跨模型移植',
             'anchor': 'p3165', 'status': 'appendix_open',
             'note': '词表方向目前 4b 专属（tokenizer 面依赖）；移植需每模型重建词表坐标。'},
            {'id': 'APPX-3', 'title': 'K2/K3 直接测量',
             'anchor': 'p3155', 'status': 'appendix_open',
             'note': 'A 闸门诊断：K2/K3 从未被直接测量；p3155 给出 KOUT 交互份额读数（15.2-18.3%）。'},
            {'id': 'APPX-4', 'title': 'N 线 P3-P7 补 Ledger',
             'anchor': 'agate', 'status': 'appendix_open_crossline',
             'note': 'N 线 P3-P7（主轴三段）登记补全至 atlas_ledger（跨线账本，只补不改）。'},
            {'id': 'APPX-5', 'title': '跨线账本补丁施加确认',
             'anchor': 'agate', 'status': 'appendix_open_crossline',
             'note': 'A 闸门 C4/C6 补丁 ledger_corrections_v1.json 已落盘未施加；施加需跨线确认。'},
        ],
    },
    'gates': {
        'G1': 'html data-k field check: mismatch == 0 vs flattened json',
        'G2': 'feature cards rendered (20 full / 3 smoke)',
        'G3': 'nodes rendered (16 full / 4 smoke)',
        'G4': 'failures rendered (12 full / 2 smoke)',
        'G5': 'upgrades rendered (3 full / 1 smoke)',
        'G6': 'gap ledger: 4 gaps + 5 appendix rendered; closed items carry anchors; gap4 carries prereg',
        'G7': 'no external resources: no <link, no <script, no http(s):// in html',
    },
    'smoke': {'features': 3, 'nodes': 4, 'failures': 2, 'upgrades': 1},
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


# ---------------------------------------------------------------- freeze
def freeze():
    os.makedirs(OUTDIR, exist_ok=True)
    exep = os.path.join(OUTDIR, 'execution.json')
    body = {k: v for k, v in DESIGN.items()}
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


# ---------------------------------------------------------------- gap ledger build
def build_gap_ledger(reg, anchors):
    """Gather gap evidence values from anchor files at runtime (no hardcoding)."""
    gl = {'schema': 'rdc_atlas_gap_ledger_v1',
          'created': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
          'provenance': {k: anchors[k]['sha8'] for k in
                         ('p3164a', 'p3164b', 'p3164c', 'p3159', 'p3160', 'p3161',
                          'p3163', 'p3165', 'p3166', 'q03', 'p3151', 'agate')},
          'gaps': [], 'appendix': []}
    for g in DESIGN['gap_ledger']['gaps']:
        rec = dict(g)
        ev = []
        if g['id'] == 'GAP-1':
            a = anchors['p3164a']['data']
            b = anchors['p3164b']['data']
            c = anchors['p3164c']['data']
            ev.append('p3164a verdict=%s class_agreement=%s' % (
                a.get('verdict'), json.dumps(a.get('class_agreement_14b_glm4'), ensure_ascii=False)))
            ev.append('p3164b verdict=%s class_agreement=%s' % (
                b.get('verdict'), json.dumps(b.get('class_agreement_14b_glm4'), ensure_ascii=False)))
            ev.append('p3164c verdict=%s class_agreement_all3=%s' % (
                c.get('verdict'), json.dumps(c.get('class_agreement_all3'), ensure_ascii=False)))
        elif g['id'] == 'GAP-2':
            m = anchors['p3160']['data']
            h = anchors['p3161']['data']
            rd = anchors['p3163']['data']
            ev.append('p3159 res=%s' % anchors['p3159']['sha8'])
            ev.append('p3160 verdict=%s mech=%s' % (m.get('verdict'), json.dumps(m.get('mech_classes'), ensure_ascii=False)))
            ev.append('p3161 verdict=%s cls=%s' % (h.get('verdict'), json.dumps(h.get('cls'), ensure_ascii=False)))
            ev.append('p3163 verdict=%s clsA=%s clsB=%s' % (
                rd.get('verdict'), json.dumps(rd.get('clsA'), ensure_ascii=False),
                json.dumps(rd.get('clsB'), ensure_ascii=False)))
        elif g['id'] == 'GAP-3':
            r65 = anchors['p3165']['data']
            r66 = anchors['p3166']['data']
            ev.append('p3165 verdict=%s' % r65.get('verdict'))
            ev.append('p3166 overall=%s' % json.dumps(r66.get('overall'), ensure_ascii=False))
        elif g['id'] == 'GAP-4':
            q = anchors['q03']['data']['summary']
            e51 = anchors['p3151']['data']['evidence']
            ev.append('q03 gate_pass_frac=%s min_E_x=%.2f pooled_mean=%.4f' % (
                q['gate_pass_frac'], q['min_E_x'], q['pooled_mean']))
            ev.append('p3151 v3 b4_worst_fold_k39=%s v4_s3_b4_mean=%s' % (
                e51['v3']['b4_worst_fold_k39'], e51['v4_s3_b4_mean']))
        rec['evidence'] = ev
        rec['anchor_sha8'] = {k: anchors[k]['sha8'] for k in g['evidence_anchors']}
        gl['gaps'].append(rec)
    for a in DESIGN['gap_ledger']['appendix']:
        rec = dict(a)
        rec['anchor_sha8'] = anchors[a['anchor']]['sha8']
        gl['appendix'].append(rec)
    return gl


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
    for i, fl in enumerate(reg['failures']):
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
    d['meta.registry_sha8'] = SHA_ANCHOR['registry_v1']
    d['meta.registry3162_sha8'] = SHA_ANCHOR['registry_3162']
    d['meta.created'] = CREATED_STR
    return d


# ---------------------------------------------------------------- html render
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


def render_html(reg, nodes, gl, feat_subset, node_subset, fail_subset, upg_subset):
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1 - RDC/LPF 语义图谱</title><style>%s</style></head>' % CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1 &mdash; 语义机制图谱（registry v1 渲染）</h1>')
    H.append('<div class="sub">Phase 3168 G5-A5 &middot; 零 GPU &middot; 源 registry sha8=%s'
             ' &middot; 3162 基座 sha8=%s &middot; 渲染于 %s</div>' % (
                 span('meta.registry_sha8', SHA_ANCHOR['registry_v1']),
                 span('meta.registry3162_sha8', SHA_ANCHOR['registry_3162']),
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
    H.append('<h2>跨模型稳定特征登记表（registry v1）</h2>')
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

    H.append('<h2>缺口账本 v1（重写）</h2>')
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
                     span(a['id'] + '.anchor_sha8', gl_appendix_sha8(gl, a['id'])),
                     span(a['id'] + '.note', a['note'])))
    H.append('</div>')

    H.append('<div class="footer">RDC/LPF atlas v1 &middot; 渲染自 atlas_registry_v1.json (%s)'
             ' + gap ledger v1 &middot; 本文件为 Phase 3168 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 SHA_ANCHOR['registry_v1'],))
    H.append('</body></html>')
    return '\n'.join(H)


GL_APPX = {}


def gl_appendix_sha8(gl, appx_id):
    return GL_APPX[appx_id]


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
    log('== Phase 3168 g5a5_atlas_render %s ==' % ('SMOKE' if SMOKE else 'FULL'))

    # source load + sha assert
    anchors = {}
    for k, p in SRC.items():
        b = open(p, 'rb').read()
        h = sha8(b)
        assert h == SHA_ANCHOR[k], ('source sha mismatch', k, h, SHA_ANCHOR[k])
        anchors[k] = {'path': p, 'sha8': h, 'data': json.loads(b.decode('utf-8'))}
    log('sources: %d files sha-asserted' % len(SRC))

    reg = anchors['registry_v1']['data']
    reg2 = anchors['registry_3162']['data']
    nodes = reg2['audit']['nodes']
    assert len(reg['features']) == 20 and len(nodes) == 16
    assert len(reg['failures']) == 12 and len(reg['upgrade_log']) == 3

    gl = build_gap_ledger(reg, anchors)
    for a in gl['appendix']:
        GL_APPX[a['id']] = a['anchor_sha8']

    # subsets
    feats = reg['features']
    fails = reg['failures']
    upgs = list(enumerate(reg['upgrade_log']))
    if SMOKE:
        feats = feats[:3]
        nodes = nodes[:4]
        fails = fails[:2]
        upgs = upgs[:1]
    else:
        nodes = nodes[:]

    flat = flatten(reg if not SMOKE else dict(reg, features=feats, failures=fails,
                                              upgrade_log=[u for _, u in upgs]),
                   nodes, gl)
    # when smoke, flatten uses full reg for features not rendered: restrict expected to rendered keys
    if SMOKE:
        keep = set()
        for f in feats:
            keep |= set(k for k in flat if k.startswith(f['id'] + '.'))
        for n in nodes:
            keep |= set(k for k in flat if k.startswith(n['id'] + '.'))
        for i, u in upgs:
            keep |= set(k for k in flat if k.startswith('UPG.%d.' % i))
        for fl in fails:
            keep |= set(k for k in flat if k.startswith(fl['id'] + '.'))
        keep |= set(k for k in flat if k.startswith(('GAP-', 'APPX-', 'meta.')))
        flat = {k: v for k, v in flat.items() if k in keep}

    html_text = render_html(reg, nodes, gl, feats, nodes, fails, upgs)

    # G7 no external resources
    assert '<link' not in html_text and '<script' not in html_text, 'external resource found'
    assert 'http://' not in html_text and 'https://' not in html_text, 'external url found'
    log('G7 no-external OK')

    v = verify_html(html_text, flat)
    log('G1 field check: expected=%d found=%d missing=%d extra=%d mismatch=%d dups=%d' % (
        v['n_expected'], v['n_found'], len(v['missing']), len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'], ('missing keys', v['missing'][:10])
    assert not v['extra'], ('extra keys', v['extra'][:10])
    assert not v['mismatch'], ('mismatch', v['mismatch'][:5])
    assert not v['dups'], ('dup keys', v['dups'][:5])

    # write outputs
    html_path = os.path.join(OUTDIR, 'smoke_atlas_v1.html' if SMOKE else 'atlas_v1.html')
    with io.open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    glp = os.path.join(OUTDIR, 'smoke_gap_ledger_v1.json' if SMOKE else 'gap_ledger_v1.json')
    with io.open(glp, 'w', encoding='utf-8') as f:
        json.dump(gl, f, ensure_ascii=False, indent=1)
    html_sha = sha8_file(html_path)
    gl_sha = sha8_file(glp)
    log('html written %s (%d B, sha8 %s)' % (os.path.basename(html_path), len(html_text.encode('utf-8')), html_sha))
    log('gap ledger written sha8 %s' % gl_sha)

    n_closed = sum(1 for g in gl['gaps'] if g['status'].startswith('closed'))
    n_open = sum(1 for g in gl['gaps'] if g['status'] == 'open')
    verdict = 'g5a5_atlas_v1|features_%d|nodes_%d|failures_%d|upgrades_%d|html_fields_%d_ok|gaps_closed_%d_open_%d|appendix_%d' % (
        len(feats), len(nodes), len(fails), len(upgs), v['n_found'], n_closed, n_open, len(gl['appendix']))

    # ---------------- result + seal
    summary = {
        'phase': 3168, 'name': 'g5a5_atlas_render', 'smoke': SMOKE,
        'design_sha8': d8, 'verdict': verdict,
        'html_file': os.path.basename(html_path), 'html_sha8': html_sha,
        'gap_ledger_file': os.path.basename(glp), 'gap_ledger_sha8': gl_sha,
        'field_check': {'n_expected': v['n_expected'], 'n_found': v['n_found'],
                        'missing': len(v['missing']), 'extra': len(v['extra']),
                        'mismatch': len(v['mismatch']), 'dups': len(v['dups'])},
        'counts': {'features': len(feats), 'nodes': len(nodes), 'failures': len(fails),
                   'upgrades': len(upgs), 'gaps': len(gl['gaps']), 'appendix': len(gl['appendix'])},
        'gap_status': {g['id']: g['status'] for g in gl['gaps']},
        'sources': {k: a['sha8'] for k, a in anchors.items()},
    }
    raw = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
    res8 = sha8(raw.encode('utf-8'))
    mid = json.dumps(dict(summary, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
    seal8 = sha8(mid.encode('utf-8'))
    rpath = os.path.join(OUTDIR, 'smoke_result.json' if SMOKE else 'result.json')
    with io.open(rpath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dict(summary, res_sha8=res8, seal_sha8=seal8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('result written res=%s seal=%s' % (res8, seal8))
    log('verdict: %s' % verdict)

    with io.open(os.path.join(OUTDIR, 'smoke_run_log.txt' if SMOKE else 'run_log.txt'),
                 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
