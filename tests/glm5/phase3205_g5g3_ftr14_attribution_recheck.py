# -*- coding: utf-8 -*-
"""Phase 3205 (G5-G3): FTR-14 A2 归因后复读臂 (registry v1.26 -> v1.27,
zero GPU, sealed 3177/3178 chain re-read + gate-fail 后续口径 scope_limits append).

Per prereg in MEMO Phase 3204 tail G5-G3 (frozen before any observation;
default arm (a) confirmed viable by pre-freeze probe: sealed p3177/p3178
results + registry v1.26 FTR-14/22/23 anchors all available):
  RECHECK (pinned, probed BEFORE freeze; honest registration):
  K1 - sealed 3178 A2 attribution scalars: r_m 4dp 0.0918/0.0951/0.1289
  (>= 0.05 bar x3); resid (g,p,r) 4dp per (model, rel); p_m aggregate
  bitwise == registry FTR-23 asserts; tc_full_drift 0 + exch_drift 0 x3;
  gates calibration_bar False / device_3157 True / replication_3177 True;
  exch_mean 1.2728174525072664 bitwise; content double lock (res
  4f521d9b / seal 1f166645).
  K2 - sealed 3177 directional folds: gate_a1 True / gate_a2 False /
  gate_exch_band True (gate False => F16, 归因后口径); per-model
  cos_full_a2 + tc_full + exchange_obs bitwise == 3178 sealed values;
  tc_full bitwise == 3157 device constants; content double lock (res
  d451030f / seal 7b746eb2).
  K3 - registry v1.26 cross-bitwise: FTR-23 p3178/p3177/p3157 asserts vs
  sealed values; FTR-14 values.exch_mean bitwise + counter_evidence 3178
  quantified text (r=0.0918/0.0951/0.1289) present.
  Registration (K1-K3 clean): FTR-14 scope_limits append ONLY (gate-fail
  后续口径: 3177 A2 fail 归因 = 关系特异残差 FTR-23 非读出缺陷; A1 6/6
  过; exch band 保持; FTR-14 维持 E1_repeatable), registry v1.26 ->
  v1.27, atlas_v1_29.html; upgrade_log / failures / other features
  UNCHANGED.
  Dirty => contradictions registered, report-only (v1.26 unchanged).
  Control anchors: sealed registry v1.26 sha 112a46ee + p3177 file sha
  b2ad52ef + p3178 file sha abae296e + gap v1.5 4fb3f37d + 3162
  00f15e98.
  SMOKE (env P3205_SMOKE=1): registry/html writes skipped; checks only.
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
NAME = 'g5g3_ftr14_attribution_recheck'
OUTDIR = os.path.join(SRC_DIR, 'phase3205', NAME)
SMOKE = os.environ.get('P3205_SMOKE') == '1'
LOG = []

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
ANCH_KEY_14 = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4-9b': 'glm4'}

GATE = {'reread_bitwise': True, 'anchor_consistent': True, 'annotation_append': True}

ANCH_3157 = {
    'qwen3-4b': dict(tc=0.5629908442497253),
    'qwen3-14b': dict(tc=0.5614360570907593),
    'glm4': dict(tc=0.5640953183174133),
}

SRC = {
    'p3177_result': os.path.join(SRC_DIR, r'phase3177\g5b2_ftr14_heldout\result.json'),
    'p3178_result': os.path.join(SRC_DIR, r'phase3178\g5b3_ftr14_a2_attribution\result.json'),
    'registry_v126': os.path.join(SRC_DIR, r'phase3204\g5g1_ftr15_heldout_recheck\atlas_registry_v1_26.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
}
SHA_ANCHOR = {
    'p3177_result': 'b2ad52ef', 'p3178_result': 'abae296e',
    'registry_v126': '112a46ee', 'registry_3162': '00f15e98',
    'gap_v15': '4fb3f37d',
}
CONTENT_3177 = ('d451030f', '7b746eb2')
CONTENT_3178 = ('4f521d9b', '1f166645')

# pinned scalars (probed BEFORE freeze; measured-anchor discipline)
PIN_R_M = {'qwen3-4b': 0.0918, 'qwen3-14b': 0.0951, 'glm4-9b': 0.1289}   # 4dp
BAR_R = 0.05
PIN_RESID = {
    'qwen3-4b': {'isa': (0.1145, 0.0236, 0.0909), 'hasa': (0.1190, 0.0264, 0.0926)},
    'qwen3-14b': {'isa': (0.1412, 0.0231, 0.1181), 'hasa': (0.0936, 0.0216, 0.0720)},
    'glm4-9b': {'isa': (0.1500, 0.0193, 0.1307), 'hasa': (0.1486, 0.0214, 0.1272)},
}
PIN_P_M = {'qwen3-4b': 0.025004226714372635, 'qwen3-14b': 0.02230689488351345,
           'glm4-9b': 0.02035186067223549}
PIN_COS_FULL_A2 = {'qwen3-4b': 0.7548636198043823, 'qwen3-14b': 0.7538491487503052,
                   'glm4-9b': 0.7555834054946899}
PIN_EXCHANGE = {'qwen3-4b': 1.1928509676419494, 'qwen3-14b': 1.294480635663979,
                'glm4-9b': 1.3311207542158712}
EXCH_MEAN = 1.2728174525072664
PIN_TC_HELD = {
    'qwen3-4b': {'isa': 0.6052, 'hasa': 0.6012},
    'qwen3-14b': {'isa': 0.5812, 'hasa': 0.6228},
    'glm4-9b': {'isa': 0.6176, 'hasa': 0.6189},
}
GATES_3177 = {'gate': False, 'gate_a1': True, 'gate_a2': False, 'gate_exch_band': True}
GATES_3178 = {'calibration_bar': False, 'device_3157': True, 'r_tol': 0.05,
              'replication_3177': True}

SCOPE_APPEND_F14 = ('；3205 G5-G3 FTR-14 A2 归因后复读（封存 3177/3178 链读数复用，'
                    '零 GPU）：3177 gate fail（F16，A2 留出-全量差超门）的归因定判 '
                    '经复读确证——K1 3178 r_m 4dp 0.0918/0.0951/0.1289 ≥0.05 bar ×3'
                    '（resid g/p/r 4dp 逐对复核、p_m 聚合逐位 == FTR-23 锚）；K2 3177 '
                    '门态 gate_a1 True / gate_a2 False / gate_exch_band True 复现，'
                    'cos_full_a2 / tc_full / exchange_obs 逐位 == 3178 封存值，'
                    'tc_full 逐位 == 3157 装置常量；K3 v1.26 FTR-14/23 锚逐位 + '
                    'FTR-14 counter_evidence 3178 量化文本在册。口径注记：A2 门失败'
                    '为关系几何事实（关系特异残差，见 FTR-23）而非读出缺陷，A1 局部'
                    '薄邻域 6/6 过、exch 均值 1.2728 带保持；FTR-14 维持 '
                    'E1_repeatable，A2 held_out 升级不再重试（关系留出构造对重建型'
                    '读数不可达，见 3182 LORO cross-null）。')

DESIGN = {
    'phase': '3205', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3204 tail G5-G3 (frozen before any observation; '
              'default arm (a) FTR-14 attribution recheck; pre-freeze probe confirmed '
              'sealed p3177/p3178 + registry v1.26 FTR-14/22/23 anchors)',
    'data': 'sealed p3177 result + p3178 result (A2 attribution chain re-read) + '
            'sealed registry v1.26 (112a46ee) + gap v1.5 (4fb3f37d) + registry_3162 '
            '(00f15e98); zero GPU',
    'control_anchor': '3178 r_m/p_m/resid 4dp pins + p_m bitwise == FTR-23 asserts + '
                      '3177 gates + cross-bitwise cos_full_a2/tc_full/exchange_obs '
                      '3177<->3178<->3157 + content double locks (3177 d451030f/'
                      '7b746eb2, 3178 4f521d9b/1f166645)',
    'measurements': 'K1 3178 attribution scalars (r_m 4dp, resid gpr 4dp, p_m '
                    'bitwise, drift 0, gates); K2 3177 gates + cross-bitwise vs '
                    '3178/3157; K3 registry FTR-14/23 anchors bitwise + '
                    'counter_evidence text',
    'gate': 'REGISTERED iff K1-K3 clean (0 contradictions) => FTR-14 scope_limits '
            'append ONLY (gate-fail 后续口径注记), registry v1.26->v1.27, '
            'atlas_v1_29.html; else contradictions registered, report-only',
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
        json.dump({'phase': 3205, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def d4(x):
    return round(float(x), 4)


def run_recheck(sealed_3177, sealed_3178, reg_v126):
    """Pinned recheck K1-K3 over sealed 3177/3178 chain + registry v1.26."""
    out = {'contradictions': []}

    # K1: 3178 attribution scalars
    assert sealed_3178['res_sha8'] == CONTENT_3178[0] and \
        sealed_3178['seal_sha8'] == CONTENT_3178[1]
    pm78 = sealed_3178['arm']['per_model']
    k1_bad = []
    for m in MODELS:
        E = pm78[m]
        if d4(E['r_m']) != PIN_R_M[m] or E['r_m'] < BAR_R:
            k1_bad.append(('r_m', m))
        for rel in ('isa', 'hasa'):
            R = E['resid'][rel]
            g, p, r = PIN_RESID[m][rel]
            if (d4(R['gap']), d4(R['p']), d4(R['r'])) != (g, p, r):
                k1_bad.append(('resid', m, rel))
        pm_agg = (E['resid']['isa']['p'] + E['resid']['hasa']['p']) / 2.0
        if pm_agg != PIN_P_M[m]:
            k1_bad.append(('p_m_agg', m))
        if E['tc_full_drift'] != 0.0 or E['exch_drift'] != 0.0:
            k1_bad.append(('drift', m))
        if E['cos_full_a2'] != PIN_COS_FULL_A2[m] or E['exchange_obs'] != PIN_EXCHANGE[m]:
            k1_bad.append(('bitwise_scalars', m))
        if E['tc_full'] != ANCH_3157[ANCH_KEY_14[m]]['tc']:
            k1_bad.append(('tc_full_3157', m))
    if sealed_3178['arm']['exch_mean'] != EXCH_MEAN:
        k1_bad.append(('exch_mean',))
    for k, v in GATES_3178.items():
        if sealed_3178['gates'].get(k) != v:
            k1_bad.append(('gate', k))
    if sealed_3178['arm']['calibration_pass'] is not False:
        k1_bad.append(('calibration_pass',))
    if k1_bad:
        out['contradictions'].append(('K1_3178_attribution', k1_bad[:6]))
    log('K1: 3178 r_m %s (bar %.2f) + resid gpr 4dp x6 + p_m bitwise x3 + '
        'drift 0 + gates: %s' % (
            '/'.join('%.4f' % PIN_R_M[m] for m in MODELS), BAR_R,
            'OK' if not k1_bad else 'FAIL %s' % k1_bad[:3]))

    # K2: 3177 gates + cross-bitwise vs 3178 / 3157
    assert sealed_3177['res_sha8'] == CONTENT_3177[0] and \
        sealed_3177['seal_sha8'] == CONTENT_3177[1]
    arm77 = sealed_3177['arm']
    pm77 = arm77['per_model']
    k2_bad = []
    for k, v in GATES_3177.items():
        if arm77.get(k) is not v:
            k2_bad.append(('gate', k))
    if arm77['exch_mean'] != EXCH_MEAN:
        k2_bad.append(('exch_mean',))
    for m in MODELS:
        E = pm77[m]
        E8 = pm78[m]
        if E['cos_full_a2'] != E8['cos_full_a2']:
            k2_bad.append(('cos_a2_x', m))
        if E['tc_full'] != E8['tc_full'] or E['tc_full'] != ANCH_3157[ANCH_KEY_14[m]]['tc']:
            k2_bad.append(('tc_x', m))
        if E['exchange_obs'] != E8['exchange_obs']:
            k2_bad.append(('exch_x', m))
        if E['exch_drift'] != 0.0 or E['tc_full_drift'] != 0.0:
            k2_bad.append(('drift', m))
        for rel in ('isa', 'hasa'):
            if d4(E['folds'][rel]['tc_held_a1']) != PIN_TC_HELD[m][rel]:
                k2_bad.append(('tc_held_a1', m, rel))
    if k2_bad:
        out['contradictions'].append(('K2_3177_cross', k2_bad[:6]))
    log('K2: 3177 gates (A1 pass / A2 fail / exch band) + cos_a2/tc/exch '
        'bitwise 3177==3178==3157 x3 + tc_held_a1 4dp: %s'
        % ('OK' if not k2_bad else 'FAIL %s' % k2_bad[:3]))

    # K3: registry v1.26 FTR-14/23 anchors bitwise + counter_evidence text
    f14 = [f for f in reg_v126['features'] if f['id'] == 'FTR-14'][0]
    f23 = [f for f in reg_v126['features'] if f['id'] == 'FTR-23'][0]
    a78 = [a for a in f23['anchors'] if a['src'] == 'p3178'][0]['asserts']
    a77 = [a for a in f23['anchors'] if a['src'] == 'p3177'][0]['asserts']
    a57 = [a for a in f23['anchors'] if a['src'] == 'p3157'][0]['asserts']
    ce14 = ' | '.join(f14['counter_evidence'])
    k3_ok = (
        all(a78['r_m.%s' % m] == pm78[m]['r_m'] for m in MODELS)
        and all(a78['p_m.%s' % m] == PIN_P_M[m] for m in MODELS)
        and a78['exch_mean'] == EXCH_MEAN and a78['calibration_pass'] is False
        and a78['res_sha8'] == CONTENT_3178[0] and a78['seal_sha8'] == CONTENT_3178[1]
        and all(a77['tc_full.%s' % m] == pm77[m]['tc_full'] for m in MODELS)
        and all(a77['exchange_obs.%s' % m] == pm77[m]['exchange_obs'] for m in MODELS)
        and all(a77['tc_held_a1.%s.%s' % (m, rel)] == pm77[m]['folds'][rel]['tc_held_a1']
                for m in MODELS for rel in ('isa', 'hasa'))
        and a57['exch_mean_obs'] == EXCH_MEAN
        and a57['res_sha8'] == '0fe043bf' and a57['seal_sha8'] == 'cfa5c3ed'
        and f14['values']['exch_mean'] == EXCH_MEAN
        and 'r=0.0918' in ce14 and 'r=0.0951' in ce14 and 'r=0.1289' in ce14
        and f14['evidence_level'] == 'E1_repeatable'
        and f23['evidence_level'] == 'E2_held_out')
    if not k3_ok:
        out['contradictions'].append(('K3_registry_cross', 'FTR-14/23'))
    log('K3: v1.26 FTR-23 p3178/p3177/p3157 anchors bitwise + FTR-14 '
        'values/counter_evidence + levels: %s' % ('OK' if k3_ok else 'FAIL'))

    out['recheck_clean'] = not out['contradictions']
    log('RECHECK: %s (%d contradictions)'
        % ('CLEAN' if out['recheck_clean'] else 'DIRTY', len(out['contradictions'])))
    return out


# ---------------- registry v1.26 -> v1.27 + atlas_v1_29 ----------------
CREATED_STR = None
REG_V126_SHA = '112a46ee'


def finish_registry(control, res8, seal8, design_sha):
    global CREATED_STR
    CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    reg26 = json.loads(open(SRC['registry_v126'], 'rb').read().decode('utf-8'))
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg26['version'] == '1.26' and len(reg26['features']) == 26
    reg27 = copy.deepcopy(reg26)
    reg27['version'] = '1.27'
    reg27['supersedes'] = 'atlas_registry_v1_26.json (112a46ee, Phase 3204)'
    audit = control
    f14_old = [f for f in reg26['features'] if f['id'] == 'FTR-14'][0]
    f14 = copy.deepcopy(f14_old)
    gate_all = bool(audit['recheck_clean'])
    if gate_all:
        f14['scope_limits'] = f14_old['scope_limits'] + SCOPE_APPEND_F14
        reg27['features'] = [f14 if f['id'] == 'FTR-14' else f
                             for f in reg26['features']]
        disposition = 'v1_26_to_v1_27_ftr14_gatefail_annotation_registered'
    else:
        reg27 = copy.deepcopy(reg26)
        disposition = 'v1_26_unchanged_recheck_dirty_report_only'
    # preservation asserts
    for f in reg26['features']:
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg27['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        if f['id'] == 'FTR-14':
            fa = json.loads(a)
            if gate_all:
                assert b != a
                assert f14['scope_limits'].startswith(fa['scope_limits'])
                assert f14['scope_limits'][:-len(SCOPE_APPEND_F14)] == fa['scope_limits']
                assert f14['evidence_level'] == fa['evidence_level'] == 'E1_repeatable'
                assert f14['statement'] == fa['statement']
                assert f14['anchors'] == fa['anchors']
                assert f14['counter_evidence'] == fa['counter_evidence']
                assert f14['values'] == fa['values']
            else:
                assert a == b
        else:
            assert a == b, ('feature not preserved', f['id'])
    assert json.dumps(reg27['upgrade_log'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg26['upgrade_log'], ensure_ascii=False, sort_keys=True)
    for k in ('evidence_level_taxonomy', 'principles', 'schema',
              'models_mainline', 'models_pending', 'provenance', 'created',
              'phase', 'name', 'design_sha8'):
        assert json.dumps(reg27.get(k), ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg26.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    assert json.dumps(reg27['failures'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg26['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'
    rp = os.path.join(OUTDIR, 'atlas_registry_v1_27.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg27, ensure_ascii=False, indent=1))
    reg27_sha = sha8_file(rp)
    log('registry v1.27 written sha8 %s (%s)' % (reg27_sha, disposition))

    meta_sha = {'reg27': reg27_sha}
    gl15 = json.loads(open(SRC['gap_v15'], encoding='utf-8').read())
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.29 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.29 &mdash; 语义机制图谱（registry v1.27 渲染）</h1>')
    H.append('<div class="sub">Phase 3205 G5-G3 &middot; FTR-14 A2 归因后复读臂（%s） &middot; '
             'registry v1.27 sha8=%s &middot; v1.26 sha8=%s &middot; v1.25 sha8=%s &middot; v1.24 sha8=%s &middot; v1.23 sha8=%s &middot; v1.22 sha8=%s &middot; v1.21 sha8=%s &middot; gap v1.6 sha8=%s'
             ' &middot; 源 p3177 %s + p3178 %s &middot; 渲染于 %s</div>' % (
                 '复读 CLEAN + REGISTERED（FTR-14 scope_limits gate-fail 后续口径 append）' if gate_all else '复读矛盾登记，report-only',
                 span('meta.registry_v127_sha8', reg27_sha),
                 span('meta.registry_v126_sha8', REG_V126_SHA),
                 span('meta.registry_v125_sha8', '3bb33a24'),
                 span('meta.registry_v124_sha8', '0d74b628'),
                 span('meta.registry_v123_sha8', '5f5b9e73'),
                 span('meta.registry_v122_sha8', '9fd87433'),
                 span('meta.registry_v121_sha8', '47cf01e0'),
                 span('meta.gap_v16_sha8', '0f6423e0'),
                 span('meta.p3177_result', SHA_ANCHOR['p3177_result']),
                 span('meta.p3178_result', SHA_ANCHOR['p3178_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg27['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg27['principles']):
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
    for f in reg27['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.27，%d 条）</h2>' % len(reg27['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg27['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg27['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg27['failures']:
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
    H.append('<div class="footer">RDC/LPF atlas v1.29 &middot; 渲染自 atlas_registry_v1_27.json (%s)'
             ' + gap ledger v1.5 (4fb3f37d) &middot; 增量来源: Phase 3205 G5-G3 FTR-14 A2 归因后复读臂'
             '（数据源 p3177 %s + p3178 %s）'
             '&middot; 本文件为 Phase 3205 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg27_sha, SHA_ANCHOR['p3177_result'], SHA_ANCHOR['p3178_result']))
    H.append('</body></html>')
    html_text = '\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_29.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v127(reg27, nodes, gl15, meta_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_29.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3205, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'recheck_clean': bool(gate_all), 'recheck': audit},
        'counts': {'features': len(reg27['features']), 'failures': len(reg27['failures']),
                   'upgrades': len(reg27['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v127_file': 'atlas_registry_v1_27.json', 'registry_v127_sha8': reg27_sha,
        'registry_v126_sha8_ref': REG_V126_SHA,
        'html_file': 'atlas_v1_29.html', 'html_sha8': html_sha,
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


def flatten_v127(reg, nodes, gl, meta_sha):
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
    d['meta.registry_v127_sha8'] = meta_sha['reg27']
    d['meta.registry_v126_sha8'] = REG_V126_SHA
    d['meta.registry_v125_sha8'] = '3bb33a24'
    d['meta.registry_v124_sha8'] = '0d74b628'
    d['meta.registry_v123_sha8'] = '5f5b9e73'
    d['meta.registry_v122_sha8'] = '9fd87433'
    d['meta.registry_v121_sha8'] = '47cf01e0'
    d['meta.gap_v16_sha8'] = '0f6423e0'
    d['meta.p3177_result'] = SHA_ANCHOR['p3177_result']
    d['meta.p3178_result'] = SHA_ANCHOR['p3178_result']
    d['meta.created'] = CREATED_STR
    return d


def main():
    design_sha = freeze()
    log('== Phase 3205 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    sealed_3177 = json.load(open(SRC['p3177_result'], encoding='utf-8'))
    sealed_3178 = json.load(open(SRC['p3178_result'], encoding='utf-8'))
    log('sealed 3177/3178 loaded (content locks asserted inside run_recheck)')
    reg_v126 = json.load(open(SRC['registry_v126'], encoding='utf-8'))
    assert reg_v126['version'] == '1.26' and len(reg_v126['features']) == 26
    log('sealed registry v1.26 loaded (%s)' % SRC['registry_v126'])

    control = run_recheck(sealed_3177, sealed_3178, reg_v126)
    gate_all = bool(control['recheck_clean'])
    if gate_all:
        verdict = ('ftr14_attribution_recheck_registered (3178 r_m 4dp '
                   '0.0918/0.0951/0.1289 >= 0.05 x3 + resid gpr 4dp x6 + p_m '
                   'bitwise == FTR-23 anchors; 3177 gates A1 pass / A2 fail / '
                   'exch band + cross-bitwise 3177==3178==3157 x3; FTR-14 '
                   'scope_limits gate-fail 后续口径 append; registry v1.26 -> '
                   'v1.27, atlas_v1_29)')
    else:
        verdict = ('ftr14_attribution_recheck_dirty (%d contradictions; '
                   'report-only, registry v1.26 unchanged)'
                   % len(control['contradictions']))
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3205, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
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
