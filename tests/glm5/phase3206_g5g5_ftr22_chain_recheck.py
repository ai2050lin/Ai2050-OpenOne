# -*- coding: utf-8 -*-
"""Phase 3206 (G5-G5): FTR-22 upgrade chain re-read arm (registry v1.27 ->
v1.28, zero GPU, sealed 3169/3171/3172 chain re-read + 残留挂账口径 scope_limits
append).

Per prereg in MEMO Phase 3205 tail G5-G5 (frozen before any observation;
default arm (a) confirmed viable by pre-freeze probe: sealed p3169/p3171/p3172
results + registry v1.27 FTR-22 three anchors all available):
  RECHECK (pinned, probed BEFORE freeze; honest registration):
  L1 - sealed 3172 pooled curve bitwise x5 (2.5388114997805062 /
  2.031459140015479 / 1.9429485491548533 / 1.8766676450829491 /
  1.800176217907229) + per_model kcurves.8.ratio bitwise x3
  (1.9485993218753288 / 1.7100461949026464 / 1.767394689531845) +
  overall ratio_kmax 1.800176217907229 / main_cls
  borderline_partial_recovery + content double lock (res 463f42c8 /
  seal cdb85525).
  L2 - sealed 3169 gate bitwise == 3172 pooled.0 x3 (k0 drift 0:
  ratio_B 2.5388114997805062 / E_oov_B 0.9478664530648125 / E_seen_B
  0.37335046463542515) + 3169 content double lock (res 49430a39 / seal
  018c6024); sealed 3171 slots.k_main.ratio_S bitwise x3 (0.7511424513908266
  / 0.7166531333803954 / 0.8205301477290912) + overall.main_cls
  mixed_across_models + content double lock (res 6a29201c / seal
  cc7ccedd).
  L3 - registry v1.27 FTR-22 three anchors (p3172/p3169/p3171) asserts
  bitwise vs sealed values (incl k0_drift==0.0 x3) + values dict +
  evidence_level E2_predictive.
  Registration (L1-L3 clean): FTR-22 scope_limits append ONLY (残留 1.8x
  成分挂账口径更新: 3175 残留归因 delta 0.0660 port_residual_dominant
  (in 1.8002 / out 1.8662) + 3203 G5-F2 双链复读 CLEAN 闭环), registry
  v1.27 -> v1.28, atlas_v1_30.html; upgrade_log / failures / other
  features UNCHANGED.
  Dirty => contradictions registered, report-only (v1.27 unchanged).
  Control anchors: sealed registry v1.27 sha d5ce1c19 + p3169 file sha
  5b51c2c1 + p3171 file sha a43cf48c + p3172 file sha d8ddc481 + gap
  v1.5 4fb3f37d + 3162 00f15e98.
  SMOKE (env P3206_SMOKE=1): registry/html writes skipped; checks only.
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

ROOT = r'D://AI2050//Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g5g5_ftr22_chain_recheck'
OUTDIR = os.path.join(SRC_DIR, 'phase3206', NAME)
SMOKE = os.environ.get('P3206_SMOKE') == '1'
LOG = []

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']

GATE = {'reread_bitwise': True, 'anchor_consistent': True, 'annotation_append': True}

SRC = {
    'p3169_result': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\result.json'),
    'p3171_result': os.path.join(SRC_DIR, r'phase3171\g5a8_collapse_mechanism\result.json'),
    'p3172_result': os.path.join(SRC_DIR, r'phase3172\g5a9_port_calibration\result.json'),
    'registry_v127': os.path.join(SRC_DIR, r'phase3205\g5g3_ftr14_attribution_recheck\atlas_registry_v1_27.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
}
SHA_ANCHOR = {
    'p3169_result': '5b51c2c1', 'p3171_result': 'a43cf48c', 'p3172_result': 'd8ddc481',
    'registry_v127': 'd5ce1c19', 'registry_3162': '00f15e98',
    'gap_v15': '4fb3f37d',
}
CONTENT_3169 = ('49430a39', '018c6024')
CONTENT_3171 = ('6a29201c', 'cc7ccedd')
CONTENT_3172 = ('463f42c8', 'cdb85525')

# pinned bitwise constants (probed BEFORE freeze)
PIN_POOLED = {  # k -> (ratio, E_oov, E_seen)
    '0': (2.5388114997805062, 0.9478664530648125, 0.37335046463542515),
    '1': (2.031459140015479, 0.7525900138749017, 0.3704677091704474),
    '2': (1.9429485491548533, 0.7152450746960111, 0.36812352803018356),
    '4': (1.8766676450829491, 0.6834135187996759, 0.3641633192698161),
    '8': (1.800176217907229, 0.6452026499642267, 0.358410828643375),
}
PIN_K8 = {'qwen3-4b': 1.9485993218753288, 'qwen3-14b': 1.7100461949026464,
          'glm4-9b': 1.767394689531845}
PIN_RATIO_S = {'qwen3-4b': 0.7511424513908266, 'qwen3-14b': 0.7166531333803954,
               'glm4-9b': 0.8205301477290912}
KGRID = ['0', '1', '2', '4', '8']

SCOPE_APPEND_F22 = ('；3206 G5-G5 FTR-22 升级链复读（封存 3169/3171/3172 链读数复用，'
                    '零 GPU）：L1 3172 pooled 曲线逐位 ×5（2.5388114997805062 → '
                    '2.031459140015479 → 1.9429485491548533 → 1.8766676450829491 → '
                    '1.800176217907229）+ per_model k=8 逐位 ×3 + overall '
                    'borderline_partial_recovery；L2 k0 逐位 == 3169 gate ×3（k0 '
                    'drift 0）+ 3171 ratio_S 逐位 ×3（mixed_across_models）；L3 '
                    'v1.27 FTR-22 三锚（p3172/p3169/p3171）逐位 + values 字典。'
                    '挂账口径更新：残留 ~1.8x 成分已在 3175 完成残留归因定判（delta '
                    '0.0660，ratio_in 1.8002 / ratio_out 1.8662，'
                    'port_residual_dominant）并于 3203 G5-F2 双链复读 CLEAN 闭环'
                    '（机制链 3169→3172→3175 + C 线五连 3188→3192）——「残留成分未'
                    '定位」表述由本注记起更新为「已定位为端口残留主导（'
                    'port_residual_dominant），因果收敛（E3）仍不成立但机制归属 '
                    'closed@3175」。范围口径不变：中文面板+单一模板族+行为读出层+'
                    '协议内干预。')

DESIGN = {
    'phase': '3206', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3205 tail G5-G5 (frozen before any observation; '
              'default arm (a) FTR-22 upgrade chain re-read; pre-freeze probe '
              'confirmed sealed p3169/p3171/p3172 + registry v1.27 anchors)',
    'data': 'sealed p3169 + p3171 + p3172 results (FTR-22 upgrade chain re-read) + '
            'sealed registry v1.27 (d5ce1c19) + gap v1.5 (4fb3f37d) + registry_3162 '
            '(00f15e98); zero GPU',
    'control_anchor': '3172 pooled curve bitwise x5 + kcurves.8 bitwise x3 + 3169 '
                      'gate bitwise (k0 drift 0) + 3171 ratio_S bitwise x3 + '
                      'content double locks (3169 49430a39/018c6024, 3171 '
                      '6a29201c/cc7ccedd, 3172 463f42c8/cdb85525) + registry FTR-22 '
                      'three-anchor bitwise',
    'measurements': 'L1 3172 pooled curve + k=8 per model + overall; L2 3169 k0 '
                    'replication bitwise + 3171 mechanism correlation; L3 registry '
                    'FTR-22 anchors bitwise + values',
    'gate': 'REGISTERED iff L1-L3 clean (0 contradictions) => FTR-22 scope_limits '
            'append ONLY (残留挂账口径更新), registry v1.27->v1.28, '
            'atlas_v1_30.html; else contradictions registered, report-only',
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
        json.dump({'phase': 3206, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def d4(x):
    return round(float(x), 4)


def run_recheck(sealed_3172, sealed_3169, sealed_3171, reg_v127):
    """Pinned recheck L1-L3 over sealed 3169/3171/3172 chain + registry v1.27."""
    out = {'contradictions': []}

    # L1: sealed 3172 pooled curve bitwise x5 + kcurves.8 x3 + overall
    assert sealed_3172['res_sha8'] == CONTENT_3172[0] and \
        sealed_3172['seal_sha8'] == CONTENT_3172[1]
    l1_bad = []
    for k in KGRID:
        P = sealed_3172['pooled'][k]
        if (P['ratio'], P['E_oov'], P['E_seen']) != PIN_POOLED[k]:
            l1_bad.append(('pooled', k))
    pm72 = sealed_3172['per_model']
    for m in MODELS:
        if pm72[m]['kcurves']['8']['ratio'] != PIN_K8[m]:
            l1_bad.append(('kcurves.8.ratio', m))
    ov72 = sealed_3172['overall']
    if ov72['ratio_kmax'] != PIN_POOLED['8'][0] or \
            ov72['main_cls'] != 'borderline_partial_recovery':
        l1_bad.append(('overall',))
    if l1_bad:
        out['contradictions'].append(('L1_3172_curve', l1_bad[:6]))
    log('L1: 3172 pooled bitwise x5 (2.5388->2.0315->1.9429->1.8767->1.8002) '
        '+ kcurves.8 x3 + overall borderline_partial_recovery: %s'
        % ('OK' if not l1_bad else 'FAIL %s' % l1_bad[:3]))

    # L2: sealed 3169 gate bitwise == pooled.0 (k0 drift 0) + 3171 ratio_S x3
    assert sealed_3169['res_sha8'] == CONTENT_3169[0] and \
        sealed_3169['seal_sha8'] == CONTENT_3169[1]
    assert sealed_3171['res_sha8'] == CONTENT_3171[0] and \
        sealed_3171['seal_sha8'] == CONTENT_3171[1]
    l2_bad = []
    g69 = sealed_3169['gate']
    if (g69['ratio_B'], g69['pooled_E_oov_B'], g69['pooled_E_seen_B']) != \
            PIN_POOLED['0']:
        l2_bad.append(('k0_replication_3169',))
    if g69['verdict'] != 'collapse_confirmed_gap4_open':
        l2_bad.append(('gate_verdict_3169',))
    pm71 = sealed_3171['per_model']
    for m in MODELS:
        if pm71[m]['slots']['k_main']['ratio_S'] != PIN_RATIO_S[m]:
            l2_bad.append(('ratio_S', m))
    ov71 = sealed_3171['overall']
    if ov71['main_cls'] != 'mixed_across_models':
        l2_bad.append(('overall_3171',))
    if l2_bad:
        out['contradictions'].append(('L2_3169_3171_chain', l2_bad[:6]))
    log('L2: 3169 gate bitwise == 3172 pooled.0 (k0 drift 0) x3 + 3171 '
        'ratio_S bitwise x3 + mixed_across_models: %s'
        % ('OK' if not l2_bad else 'FAIL %s' % l2_bad[:3]))

    # L3: registry v1.27 FTR-22 three anchors bitwise + values + level
    f22 = [f for f in reg_v127['features'] if f['id'] == 'FTR-22'][0]
    a72 = [a for a in f22['anchors'] if a['src'] == 'p3172'][0]['asserts']
    a69 = [a for a in f22['anchors'] if a['src'] == 'p3169'][0]['asserts']
    a71 = [a for a in f22['anchors'] if a['src'] == 'p3171'][0]['asserts']
    V = f22['values']
    l3_ok = (
        all(a72['pooled.%s.ratio' % k] == PIN_POOLED[k][0] for k in KGRID)
        and all(a72['per_model.%s.kcurves.8.ratio' % m] == PIN_K8[m] for m in MODELS)
        and a72['res_sha8'] == CONTENT_3172[0] and a72['seal_sha8'] == CONTENT_3172[1]
        and a69['gate.ratio_B'] == PIN_POOLED['0'][0]
        and a69['gate.pooled_E_oov_B'] == PIN_POOLED['0'][1]
        and a69['gate.pooled_E_seen_B'] == PIN_POOLED['0'][2]
        and a69['k0_drift_ratio'] == 0.0 and a69['k0_drift_E_oov'] == 0.0
        and a69['k0_drift_E_seen'] == 0.0
        and a69['res_sha8'] == CONTENT_3169[0] and a69['seal_sha8'] == CONTENT_3169[1]
        and all(a71['per_model.%s.slots.k_main.ratio_S' % m] == PIN_RATIO_S[m]
                for m in MODELS)
        and a71['overall.main_cls'] == 'mixed_across_models'
        and a71['res_sha8'] == CONTENT_3171[0] and a71['seal_sha8'] == CONTENT_3171[1]
        and V['ratio_k0'] == PIN_POOLED['0'][0] and V['ratio_k8'] == PIN_POOLED['8'][0]
        and V['E_oov_pooled_k0'] == PIN_POOLED['0'][1]
        and V['E_seen_pooled_k0'] == PIN_POOLED['0'][2]
        and V['k0_drift_ratio_vs_3169'] == 0.0
        and V['p3171_ratio_S_qwen3_4b'] == PIN_RATIO_S['qwen3-4b']
        and V['p3171_ratio_S_qwen3_14b'] == PIN_RATIO_S['qwen3-14b']
        and V['p3171_ratio_S_glm4_9b'] == PIN_RATIO_S['glm4-9b']
        and f22['evidence_level'] == 'E2_predictive')
    if not l3_ok:
        out['contradictions'].append(('L3_registry_cross', 'FTR-22'))
    log('L3: v1.27 FTR-22 p3172/p3169/p3171 anchors bitwise (incl '
        'k0_drift==0.0 x3) + values dict + E2_predictive: %s'
        % ('OK' if l3_ok else 'FAIL'))

    out['recheck_clean'] = not out['contradictions']
    log('RECHECK: %s (%d contradictions)'
        % ('CLEAN' if out['recheck_clean'] else 'DIRTY', len(out['contradictions'])))
    return out


# ---------------- registry v1.27 -> v1.28 + atlas_v1_30 ----------------
CREATED_STR = None
REG_V127_SHA = 'd5ce1c19'


def finish_registry(control, res8, seal8, design_sha):
    global CREATED_STR
    CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    reg27 = json.loads(open(SRC['registry_v127'], 'rb').read().decode('utf-8'))
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg27['version'] == '1.27' and len(reg27['features']) == 26
    reg28 = copy.deepcopy(reg27)
    reg28['version'] = '1.28'
    reg28['supersedes'] = 'atlas_registry_v1_27.json (d5ce1c19, Phase 3205)'
    audit = control
    f22_old = [f for f in reg27['features'] if f['id'] == 'FTR-22'][0]
    f22 = copy.deepcopy(f22_old)
    gate_all = bool(audit['recheck_clean'])
    if gate_all:
        f22['scope_limits'] = f22_old['scope_limits'] + SCOPE_APPEND_F22
        reg28['features'] = [f22 if f['id'] == 'FTR-22' else f
                             for f in reg27['features']]
        disposition = 'v1_27_to_v1_28_ftr22_chain_reread_annotation_registered'
    else:
        reg28 = copy.deepcopy(reg27)
        disposition = 'v1_27_unchanged_recheck_dirty_report_only'
    # preservation asserts
    for f in reg27['features']:
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg28['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        if f['id'] == 'FTR-22':
            fa = json.loads(a)
            if gate_all:
                assert b != a
                assert f22['scope_limits'].startswith(fa['scope_limits'])
                assert f22['scope_limits'][:-len(SCOPE_APPEND_F22)] == fa['scope_limits']
                assert f22['evidence_level'] == fa['evidence_level'] == 'E2_predictive'
                assert f22['statement'] == fa['statement']
                assert f22['anchors'] == fa['anchors']
                assert f22['counter_evidence'] == fa['counter_evidence']
                assert f22['values'] == fa['values']
            else:
                assert a == b
        else:
            assert a == b, ('feature not preserved', f['id'])
    assert json.dumps(reg28['upgrade_log'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg27['upgrade_log'], ensure_ascii=False, sort_keys=True)
    for k in ('evidence_level_taxonomy', 'principles', 'schema',
              'models_mainline', 'models_pending', 'provenance', 'created',
              'phase', 'name', 'design_sha8'):
        assert json.dumps(reg28.get(k), ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg27.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    assert json.dumps(reg28['failures'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg27['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'
    rp = os.path.join(OUTDIR, 'atlas_registry_v1_28.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg28, ensure_ascii=False, indent=1))
    reg28_sha = sha8_file(rp)
    log('registry v1.28 written sha8 %s (%s)' % (reg28_sha, disposition))

    meta_sha = {'reg28': reg28_sha}
    gl15 = json.loads(open(SRC['gap_v15'], encoding='utf-8').read())
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.30 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.30 &mdash; 语义机制图谱（registry v1.28 渲染）</h1>')
    H.append('<div class="sub">Phase 3206 G5-G5 &middot; FTR-22 升级链复读臂（%s） &middot; '
             'registry v1.28 sha8=%s &middot; v1.27 sha8=%s &middot; v1.26 sha8=%s &middot; v1.25 sha8=%s &middot; v1.24 sha8=%s &middot; v1.23 sha8=%s &middot; v1.22 sha8=%s &middot; v1.21 sha8=%s &middot; gap v1.6 sha8=%s'
             ' &middot; 源 p3169 %s + p3171 %s + p3172 %s &middot; 渲染于 %s</div>' % (
                 '复读 CLEAN + REGISTERED（FTR-22 scope_limits 残留挂账口径更新 append）' if gate_all else '复读矛盾登记，report-only',
                 span('meta.registry_v128_sha8', reg28_sha),
                 span('meta.registry_v127_sha8', REG_V127_SHA),
                 span('meta.registry_v126_sha8', '112a46ee'),
                 span('meta.registry_v125_sha8', '3bb33a24'),
                 span('meta.registry_v124_sha8', '0d74b628'),
                 span('meta.registry_v123_sha8', '5f5b9e73'),
                 span('meta.registry_v122_sha8', '9fd87433'),
                 span('meta.registry_v121_sha8', '47cf01e0'),
                 span('meta.gap_v16_sha8', '0f6423e0'),
                 span('meta.p3169_result', SHA_ANCHOR['p3169_result']),
                 span('meta.p3171_result', SHA_ANCHOR['p3171_result']),
                 span('meta.p3172_result', SHA_ANCHOR['p3172_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg28['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg28['principles']):
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
    for f in reg28['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.28，%d 条）</h2>' % len(reg28['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg28['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg28['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg28['failures']:
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
    H.append('<div class="footer">RDC/LPF atlas v1.30 &middot; 渲染自 atlas_registry_v1_28.json (%s)'
             ' + gap ledger v1.5 (4fb3f37d) &middot; 增量来源: Phase 3206 G5-G5 FTR-22 升级链复读臂'
             '（数据源 p3169 %s + p3171 %s + p3172 %s）'
             '&middot; 本文件为 Phase 3206 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg28_sha, SHA_ANCHOR['p3169_result'], SHA_ANCHOR['p3171_result'],
                 SHA_ANCHOR['p3172_result']))
    H.append('</body></html>')
    html_text = '\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_30.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v128(reg28, nodes, gl15, meta_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_30.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3206, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'recheck_clean': bool(gate_all), 'recheck': audit},
        'counts': {'features': len(reg28['features']), 'failures': len(reg28['failures']),
                   'upgrades': len(reg28['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v128_file': 'atlas_registry_v1_28.json', 'registry_v128_sha8': reg28_sha,
        'registry_v127_sha8_ref': REG_V127_SHA,
        'html_file': 'atlas_v1_30.html', 'html_sha8': html_sha,
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


def flatten_v128(reg, nodes, gl, meta_sha):
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
    d['meta.registry_v128_sha8'] = meta_sha['reg28']
    d['meta.registry_v127_sha8'] = REG_V127_SHA
    d['meta.registry_v126_sha8'] = '112a46ee'
    d['meta.registry_v125_sha8'] = '3bb33a24'
    d['meta.registry_v124_sha8'] = '0d74b628'
    d['meta.registry_v123_sha8'] = '5f5b9e73'
    d['meta.registry_v122_sha8'] = '9fd87433'
    d['meta.registry_v121_sha8'] = '47cf01e0'
    d['meta.gap_v16_sha8'] = '0f6423e0'
    d['meta.p3169_result'] = SHA_ANCHOR['p3169_result']
    d['meta.p3171_result'] = SHA_ANCHOR['p3171_result']
    d['meta.p3172_result'] = SHA_ANCHOR['p3172_result']
    d['meta.created'] = CREATED_STR
    return d


def main():
    design_sha = freeze()
    log('== Phase 3206 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    sealed_3169 = json.load(open(SRC['p3169_result'], encoding='utf-8'))
    sealed_3171 = json.load(open(SRC['p3171_result'], encoding='utf-8'))
    sealed_3172 = json.load(open(SRC['p3172_result'], encoding='utf-8'))
    log('sealed 3169/3171/3172 loaded (content locks asserted inside run_recheck)')
    reg_v127 = json.load(open(SRC['registry_v127'], encoding='utf-8'))
    assert reg_v127['version'] == '1.27' and len(reg_v127['features']) == 26
    log('sealed registry v1.27 loaded (%s)' % SRC['registry_v127'])

    control = run_recheck(sealed_3172, sealed_3169, sealed_3171, reg_v127)
    gate_all = bool(control['recheck_clean'])
    if gate_all:
        verdict = ('ftr22_chain_recheck_registered (3172 pooled bitwise x5 '
                   '2.5388->2.0315->1.9429->1.8767->1.8002 + kcurves.8 x3 + '
                   'overall borderline_partial_recovery; 3169 gate bitwise == '
                   'pooled.0, k0 drift 0; 3171 ratio_S x3 '
                   'mixed_across_models; FTR-22 three anchors bitwise + '
                   'values + E2_predictive; FTR-22 scope_limits 残留挂账口径 '
                   '更新 append; registry v1.27 -> v1.28, atlas_v1_30)')
    else:
        verdict = ('ftr22_chain_recheck_dirty (%d contradictions; '
                   'report-only, registry v1.27 unchanged)'
                   % len(control['contradictions']))
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3206, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
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
