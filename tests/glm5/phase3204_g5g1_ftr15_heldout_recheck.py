# -*- coding: utf-8 -*-
"""Phase 3204 (G5-G1): FTR-15 held_out 补强复读臂 (registry v1.25 -> v1.26,
zero GPU, sealed 3158/3180 chain re-read + E2_held_out 口径注记 append).

Per prereg in MEMO Phase 3203 tail G5-G1 (frozen before any observation;
default arm (a) confirmed viable by pre-freeze probe: sealed p3158 summary +
p3180 held-out result + registry v1.25 FTR-15 anchors all available):
  RECHECK (pinned, probed BEFORE freeze; honest registration):
  H1 - sealed p3158 ratios bitwise == pinned full-precision constants
  (qwen3-4b 1.0283477199188433 / qwen3-14b 1.3316768638966026 /
  glm4 1.2472595295618603) + p3158 content double lock (res fa5cca12 /
  seal e0c60629).
  H2 - sealed p3180 held-out scalars: ratio_full_recompute bitwise ==
  p3158 ratios x3 (device drift 0 x3); max_band 4dp 0.0300/0.0278/0.0182
  < 0.25; folds 16 x3 all same_class; n_c_gt1 10/16/15; gates
  G1_device/G2_heldout True, G2_class 3.0, G2_log_band 0.25; content
  double lock (res f1689e94 / seal a3c1b53e).
  H3 - registry v1.25 FTR-15 anchors bitwise (p3158 ratios + p3180
  ratio_full_recompute/max_log_band/all_same_class + res/seal shas);
  upgrade_log FTR-15 E1->E2_predictive entry present; evidence_level
  E2_predictive with held_out tag.
  Registration (H1-H3 clean): FTR-15 scope_limits append ONLY (E2_held_out
  口径注记: held_out = 实体 LOO fold 链 3204 复读 CLEAN，E2_held_out 为
  E2_predictive 强化档口径), registry v1.25 -> v1.26, atlas_v1_28.html;
  upgrade_log / failures / all other features UNCHANGED.
  Dirty => contradictions registered, report-only (v1.25 unchanged).
  Control anchors: sealed registry v1.25 sha 3bb33a24 + p3158 summary file
  sha + p3180 result file sha + gap v1.5 4fb3f37d + registry_3162 00f15e98.
  SMOKE (env P3204_SMOKE=1): registry/html writes skipped; checks only.
"""

import copy
import datetime
import hashlib
import json
import os
import sys
import time

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3179_g5b4_ftr23_registration as p79

span = p79.span

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g5g1_ftr15_heldout_recheck'
OUTDIR = os.path.join(SRC_DIR, 'phase3204', NAME)
SMOKE = os.environ.get('P3204_SMOKE') == '1'
LOG = []

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
KEY = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4-9b': 'glm4'}

GATE = {'reread_bitwise': True, 'anchor_consistent': True, 'annotation_append': True}

SRC = {
    'p3158_summary': os.path.join(SRC_DIR, r'phase3158\g4p1_output_equivalence_class\summary\result_summary.json'),
    'p3180_result': os.path.join(SRC_DIR, r'phase3180\g5b5_ftr15_heldout\result.json'),
    'registry_v125': os.path.join(SRC_DIR, r'phase3202\g5f1_ftr04_recheck\atlas_registry_v1_25.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
}
SHA_ANCHOR = {
    'p3158_summary': '7ed85b36', 'p3180_result': 'cb811310',
    'registry_v125': '3bb33a24', 'registry_3162': '00f15e98',
    'gap_v15': '4fb3f37d',
}
CONTENT_3158 = ('fa5cca12', 'e0c60629')
CONTENT_3180 = ('f1689e94', 'a3c1b53e')

# pinned full-precision 3158 ratios (probed BEFORE freeze; bitwise compare)
PIN_3158 = {
    'qwen3-4b': 1.0283477199188433,
    'qwen3-14b': 1.3316768638966026,
    'glm4-9b': 1.2472595295618603,
}
# pinned 4dp p3180 held-out scalars (probed BEFORE freeze)
PIN_3180 = {
    'qwen3-4b': dict(max_band=0.0300, drift=0.0, all_same=True, n_c_gt1=10),
    'qwen3-14b': dict(max_band=0.0278, drift=0.0, all_same=True, n_c_gt1=16),
    'glm4-9b': dict(max_band=0.0182, drift=0.0, all_same=True, n_c_gt1=15),
}
N_FOLDS = 16
BAND_BAR = 0.25
GATES_3180 = {'G1_device': True, 'G2_heldout': True, 'G2_class': 3.0, 'G2_log_band': 0.25}

SCOPE_APPEND_F15 = ('；3204 G5-G1 FTR-15 held_out 补强复读（封存 3158/3180 链读数复用，'
                    '零 GPU）：H1 3158 预算比逐位回读（1.0283477199188433 / '
                    '1.3316768638966026 / 1.2472595295618603）+ 内容双锁；H2 3180 '
                    '实体 LOO 链复读——ratio_full_recompute 逐位 == 3158 ×3（device '
                    'drift 0 ×3）、max_log_band 0.0300/0.0278/0.0182 <0.25、16 折 ×3 '
                    '全 same_class、n_c_gt1 10/16/15、G1_device/G2_heldout 门 '
                    'True；H3 registry v1.25 FTR-15 锚逐位 + 升级链条目在册。'
                    '口径注记：本特征 held_out 证据为实体 LOO fold（16 锚=16 实体）'
                    '口径，E2_held_out 为 E2_predictive 强化档注记，不新增独立 '
                    'evidence_level；「局部等价」仍不外推为全局商结构。')

DESIGN = {
    'phase': '3204', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3203 tail G5-G1 (frozen before any observation; '
              'default arm (a) FTR-15 held_out recheck; pre-freeze probe confirmed '
              'sealed p3158 summary + p3180 result + registry v1.25 anchors)',
    'data': 'sealed p3158 summary (ratios bitwise) + sealed p3180 held-out result '
            '(per_model + folds_full re-read) + sealed registry v1.25 (3bb33a24) + '
            'gap v1.5 (4fb3f37d) + registry_3162 (00f15e98); zero GPU',
    'control_anchor': 'p3158 ratios bitwise vs pinned full-precision constants + '
                      'p3180 ratio_full_recompute bitwise == p3158 x3 (drift 0) + '
                      '4dp max_band pins + content double locks (3158 fa5cca12/'
                      'e0c60629, 3180 f1689e94/a3c1b53e) + registry FTR-15 anchor '
                      'bitwise',
    'measurements': 'H1 3158 ratios bitwise x3; H2 3180 held-out chain (folds '
                    '16x3 same_class, max_band 4dp, drift 0, gates); H3 registry '
                    'anchor bitwise + upgrade_log entry + evidence_level',
    'gate': 'REGISTERED iff H1-H3 clean (0 contradictions) => FTR-15 scope_limits '
            'append ONLY (E2_held_out 口径注记), registry v1.25->v1.26, '
            'atlas_v1_28.html; else contradictions registered, report-only',
    'sources': SHA_ANCHOR,
    'gates': GATE,
}


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


def freeze():
    eblob = json.dumps(DESIGN, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    dsha = hashlib.sha256(eblob).hexdigest()
    os.makedirs(OUTDIR, exist_ok=True)
    exe_p = os.path.join(OUTDIR, 'execution.json')
    if os.path.exists(exe_p) and not SMOKE:
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha8'] == dsha[:8], 'DESIGN DRIFT'
        log('freeze: existing execution.json OK (%s)' % dsha[:8])
    else:
        json.dump({'phase': 3204, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def d4(x):
    return round(float(x), 4)


def run_recheck(sealed_3158, sealed_3180, reg_v125):
    """Pinned recheck H1-H3 over sealed 3158/3180 chain + registry v1.25 FTR-15."""
    out = {'contradictions': []}

    # H1: 3158 ratios bitwise vs pinned full-precision constants + double lock
    assert sealed_3158['res_sha8'] == CONTENT_3158[0] and \
        sealed_3158['seal_sha8'] == CONTENT_3158[1]
    h1_ok = all(sealed_3158['ratios'][KEY[m]] == PIN_3158[m] for m in MODELS)
    if not h1_ok:
        out['contradictions'].append(('H1_3158_ratios_bitwise', 'mismatch'))
    log('H1: sealed 3158 ratios bitwise == pinned constants x3: %s'
        % ('OK' if h1_ok else 'FAIL'))

    # H2: 3180 held-out chain re-read
    assert sealed_3180['res_sha8'] == CONTENT_3180[0] and \
        sealed_3180['seal_sha8'] == CONTENT_3180[1]
    pm80 = sealed_3180['per_model']
    h2_bad = []
    for m in MODELS:
        E = pm80[KEY[m]]
        P = PIN_3180[m]
        if E['ratio_full_recompute'] != PIN_3158[m]:
            h2_bad.append(('ratio_bitwise', m))
        if E['ratio_sealed'] != E['ratio_full_recompute']:
            h2_bad.append(('ratio_sealed_ne', m))
        if d4(E['max_band']) != P['max_band']:
            h2_bad.append(('max_band_4dp', m))
        if float(E['drift']) != 0.0:
            h2_bad.append(('drift_nonzero', m))
        if E['all_same_class'] is not True:
            h2_bad.append(('all_same', m))
        if int(E['n_c_gt1']) != P['n_c_gt1']:
            h2_bad.append(('n_c_gt1', m))
        folds = sealed_3180['folds_full'][KEY[m]]
        if len(folds) != N_FOLDS or not all(f['same_class'] for f in folds):
            h2_bad.append(('folds', m))
        if d4(max(f['log_band'] for f in folds)) != P['max_band']:
            h2_bad.append(('folds_max_band', m))
        if max(f['ratio_train'] for f in folds) >= 3.0:
            h2_bad.append(('fold_class_ge3', m))
    for k, v in GATES_3180.items():
        if sealed_3180['gates'].get(k) != v:
            h2_bad.append(('gate', k))
    if h2_bad:
        out['contradictions'].append(('H2_3180_chain', h2_bad[:6]))
    log('H2: 3180 held-out chain (ratio bitwise x3, max_band %s, folds 16x3 '
        'same_class, gates): %s' % (
            '/'.join('%.4f' % PIN_3180[m]['max_band'] for m in MODELS),
            'OK' if not h2_bad else 'FAIL %s' % h2_bad[:3]))

    # H3: registry v1.25 FTR-15 anchors bitwise + upgrade entry + level
    f15 = [f for f in reg_v125['features'] if f['id'] == 'FTR-15'][0]
    A = {a['src']: a for a in f15['anchors']}
    a58 = A['p3158']['asserts']
    a80 = A['p3180']['asserts']
    h3_ok = (
        all(a58['ratios.%s' % KEY[m]] == PIN_3158[m] for m in MODELS)
        and a58['res_sha8'] == CONTENT_3158[0] and a58['seal_sha8'] == CONTENT_3158[1]
        and all(a80['ratio_full_recompute.%s' % KEY[m]] == PIN_3158[m] for m in MODELS)
        and all(a80['max_log_band.%s' % KEY[m]] == pm80[KEY[m]]['max_band'] for m in MODELS)
        and a80['all_same_class'] is True
        and a80['res_sha8'] == CONTENT_3180[0] and a80['seal_sha8'] == CONTENT_3180[1]
        and 'held_out' in A['p3180']['tags']
        and f15['evidence_level'] == 'E2_predictive'
        and any(u.get('node') == 'FTR-15' for u in reg_v125['upgrade_log']))
    if not h3_ok:
        out['contradictions'].append(('H3_registry_anchors', 'FTR-15'))
    log('H3: registry v1.25 FTR-15 anchors bitwise (3158+3180) + upgrade entry '
        '+ E2_predictive/held_out: %s' % ('OK' if h3_ok else 'FAIL'))

    out['recheck_clean'] = not out['contradictions']
    log('RECHECK: %s (%d contradictions)'
        % ('CLEAN' if out['recheck_clean'] else 'DIRTY', len(out['contradictions'])))
    return out


# ---------------- registry v1.25 -> v1.26 + atlas_v1_28 ----------------
CREATED_STR = None
REG_V125_SHA = '3bb33a24'
SCOPE_F15_PREFIX_LEN = None


def finish_registry(control, res8, seal8, design_sha):
    global CREATED_STR, SCOPE_F15_PREFIX_LEN
    CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    reg25 = json.loads(open(SRC['registry_v125'], 'rb').read().decode('utf-8'))
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg25['version'] == '1.25' and len(reg25['features']) == 26
    reg26 = copy.deepcopy(reg25)
    reg26['version'] = '1.26'
    reg26['supersedes'] = 'atlas_registry_v1_25.json (3bb33a24, Phase 3202)'
    audit = control
    f15_old = [f for f in reg25['features'] if f['id'] == 'FTR-15'][0]
    f15 = copy.deepcopy(f15_old)
    gate_all = bool(audit['recheck_clean'])
    if gate_all:
        SCOPE_F15_PREFIX_LEN = len(f15_old['scope_limits'])
        f15['scope_limits'] = f15_old['scope_limits'] + SCOPE_APPEND_F15
        reg26['features'] = [f15 if f['id'] == 'FTR-15' else f
                             for f in reg25['features']]
        disposition = 'v1_25_to_v1_26_ftr15_heldout_annotation_registered'
    else:
        reg26 = copy.deepcopy(reg25)
        disposition = 'v1_25_unchanged_recheck_dirty_report_only'
    # preservation asserts
    for f in reg25['features']:
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg26['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        if f['id'] == 'FTR-15':
            fa = json.loads(a)
            if gate_all:
                assert b != a
                assert f15['scope_limits'].startswith(fa['scope_limits'])
                assert f15['scope_limits'][:-len(SCOPE_APPEND_F15)] == fa['scope_limits']
                assert f15['evidence_level'] == fa['evidence_level'] == 'E2_predictive'
                assert f15['statement'] == fa['statement']
                assert f15['anchors'] == fa['anchors']
                assert f15['counter_evidence'] == fa['counter_evidence']
                assert f15['values'] == fa['values']
            else:
                assert a == b
        else:
            assert a == b, ('feature not preserved', f['id'])
    assert json.dumps(reg26['upgrade_log'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg25['upgrade_log'], ensure_ascii=False, sort_keys=True)
    for k in ('evidence_level_taxonomy', 'principles', 'schema',
              'models_mainline', 'models_pending', 'provenance', 'created',
              'phase', 'name', 'design_sha8'):
        assert json.dumps(reg26.get(k), ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg25.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    assert json.dumps(reg26['failures'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg25['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'
    rp = os.path.join(OUTDIR, 'atlas_registry_v1_26.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg26, ensure_ascii=False, indent=1))
    reg26_sha = sha8_file(rp)
    log('registry v1.26 written sha8 %s (%s)' % (reg26_sha, disposition))

    meta_sha = {'reg26': reg26_sha}
    gl15 = json.loads(open(SRC['gap_v15'], encoding='utf-8').read())
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.28 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.28 &mdash; 语义机制图谱（registry v1.26 渲染）</h1>')
    H.append('<div class="sub">Phase 3204 G5-G1 &middot; FTR-15 held_out 补强复读臂（%s） &middot; '
             'registry v1.26 sha8=%s &middot; v1.25 sha8=%s &middot; v1.24 sha8=%s &middot; v1.23 sha8=%s &middot; v1.22 sha8=%s &middot; v1.21 sha8=%s &middot; gap v1.6 sha8=%s'
             ' &middot; 源 p3158 %s + p3180 %s &middot; 渲染于 %s</div>' % (
                 '复读 CLEAN + REGISTERED（FTR-15 scope_limits E2_held_out 口径注记 append）' if gate_all else '复读矛盾登记，report-only',
                 span('meta.registry_v126_sha8', reg26_sha),
                 span('meta.registry_v125_sha8', REG_V125_SHA),
                 span('meta.registry_v124_sha8', '0d74b628'),
                 span('meta.registry_v123_sha8', '5f5b9e73'),
                 span('meta.registry_v122_sha8', '9fd87433'),
                 span('meta.registry_v121_sha8', '47cf01e0'),
                 span('meta.gap_v16_sha8', '0f6423e0'),
                 span('meta.p3158_summary', SHA_ANCHOR['p3158_summary']),
                 span('meta.p3180_result', SHA_ANCHOR['p3180_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg26['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg26['principles']):
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
    for f in reg26['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.26，%d 条）</h2>' % len(reg26['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg26['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg26['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg26['failures']:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td class="mono">%s</td><td>%s</td><td class="mono">%s</td></tr>' % (
            fl['id'], span(fl['id'] + '.kind', fl['kind']), span(fl['id'] + '.phase', fl['phase']),
            span(fl['id'] + '.text', fl['text']), span(fl['id'] + '.evidence', fl['evidence'])))
    H.append('</table>')
    H.append('<h2>缺口账本 v1.5（GAP-4 quantified_collapse 定稿；v1.6 mechanism_note 见 3192 封存）</h2>')
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
    H.append('<div class="footer">RDC/LPF atlas v1.28 &middot; 渲染自 atlas_registry_v1_26.json (%s)'
             ' + gap ledger v1.5 (4fb3f37d) &middot; 增量来源: Phase 3204 G5-G1 FTR-15 held_out 补强复读臂'
             '（数据源 p3158 %s + p3180 %s）'
             '&middot; 本文件为 Phase 3204 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg26_sha, SHA_ANCHOR['p3158_summary'], SHA_ANCHOR['p3180_result']))
    H.append('</body></html>')
    html_text = '\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_28.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v126(reg26, nodes, gl15, meta_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_28.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3204, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'recheck_clean': bool(gate_all), 'recheck': audit},
        'counts': {'features': len(reg26['features']), 'failures': len(reg26['failures']),
                   'upgrades': len(reg26['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v126_file': 'atlas_registry_v1_26.json', 'registry_v126_sha8': reg26_sha,
        'registry_v125_sha8_ref': REG_V125_SHA,
        'html_file': 'atlas_v1_28.html', 'html_sha8': html_sha,
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


def flatten_v126(reg, nodes, gl, meta_sha):
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
    d['meta.registry_v126_sha8'] = meta_sha['reg26']
    d['meta.registry_v125_sha8'] = REG_V125_SHA
    d['meta.registry_v124_sha8'] = '0d74b628'
    d['meta.registry_v123_sha8'] = '5f5b9e73'
    d['meta.registry_v122_sha8'] = '9fd87433'
    d['meta.registry_v121_sha8'] = '47cf01e0'
    d['meta.gap_v16_sha8'] = '0f6423e0'
    d['meta.p3158_summary'] = SHA_ANCHOR['p3158_summary']
    d['meta.p3180_result'] = SHA_ANCHOR['p3180_result']
    d['meta.created'] = CREATED_STR
    return d


def main():
    design_sha = freeze()
    log('== Phase 3204 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    sealed_3158 = json.load(open(SRC['p3158_summary'], encoding='utf-8'))
    sealed_3180 = json.load(open(SRC['p3180_result'], encoding='utf-8'))
    log('sealed 3158/3180 loaded (content locks asserted inside run_recheck)')
    reg_v125 = json.load(open(SRC['registry_v125'], encoding='utf-8'))
    assert reg_v125['version'] == '1.25' and len(reg_v125['features']) == 26
    log('sealed registry v1.25 loaded (%s)' % SRC['registry_v125'])

    control = run_recheck(sealed_3158, sealed_3180, reg_v125)
    gate_all = bool(control['recheck_clean'])
    if gate_all:
        verdict = ('ftr15_heldout_recheck_registered (3158 ratios bitwise x3; 3180 '
                   'held-out chain clean: ratio_full_recompute bitwise == 3158 x3, '
                   'max_band 0.0300/0.0278/0.0182 <0.25, folds 16x3 same_class, '
                   'drift 0; FTR-15 registry anchors bitwise; scope_limits '
                   'E2_held_out 口径注记 append; registry v1.25 -> v1.26, '
                   'atlas_v1_28)')
    else:
        verdict = ('ftr15_heldout_recheck_dirty (%d contradictions; report-only, '
                   'registry v1.25 unchanged)' % len(control['contradictions']))
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3204, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'gates': dict(GATE, recheck_clean=bool(gate_all),
                      n_contradictions=len(control['contradictions'])),
        'recheck': control,
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
        finish_registry(control, res8, seal8, design_sha)
    with open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
