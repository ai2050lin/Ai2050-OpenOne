# -*- coding: utf-8 -*-
"""Phase 3200 (G5-E2): FTR-23 held_out upgrade arm (registry v1.22 -> v1.23,
zero GPU, sealed 3178 null-fold scalars reuse; 3176 batch-upgrade precedent).

Per prereg in MEMO Phase 3199 tail G5-E2 (frozen before any observation;
protocol refined pre-freeze: FTR-23 has NO sealed direction vector in the
3169 OOV row space - it is a scalar phenomenon on the 3157 context-transform
panel - so the held-out arm evaluates the SEALED 3178 entity-LOO null folds
cos_out (held entity cells vs within-relation mean direction, 16 folds x 2
relations x 3 models), the panel-internal held-out reading, instead of an
OOV-row projection; refinement registered honestly):
  GATE (pinned pre-observation): HOLDOUT_PREDICTIVE_FTR23 iff median cos_out
  >= 0.5 in ALL 6 (model, relation) units => FTR-23 upgraded E1_repeatable ->
  E2_held_out: evidence_level change + anchors append (role=held_out) +
  statement append + upgrade_log #9, registry v1.22 -> v1.23,
  atlas_v1_25.html. Not passed => counter_evidence append, E level
  UNCHANGED E1_repeatable.
  Control anchors: sealed 3178 scalars re-read vs pinned 4dp constants
  (medians probed from sealed p3178 result BEFORE freeze, measured-anchor
  discipline; exch_mean bitwise) + sealed 3178 content double lock +
  3183 dv scalars bitwise (build_dvu) + 3157 tc drift < 1e-9.
  SMOKE (env P3200_SMOKE=1): registry/html writes skipped; checks only.
"""
import copy
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np

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
NAME = 'g5e2_ftr23_heldout_upgrade'
OUTDIR = os.path.join(SRC_DIR, 'phase3200', NAME)
SMOKE = os.environ.get('P3200_SMOKE') == '1'
LOG = []

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
NPZ57_KEY = {'qwen3-4b': 'p3157_4b', 'qwen3-14b': 'p3157_14b', 'glm4-9b': 'p3157_glm4'}
ANCH_KEY = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4-9b': 'glm4'}

G_MED = 0.5
DEV_TOL = 1e-9
REPL_TOL = 1e-9
GATE = {'heldout_med_cos_out_ftr23': G_MED, 'units_needed': 6,
        'dev_tol': DEV_TOL, 'repl_tol': REPL_TOL}

ANCH_3157 = {
    'qwen3-4b': dict(nl=36, kout=35, tc_mean_cos=0.5629908442497253),
    'qwen3-14b': dict(nl=40, kout=39, tc_mean_cos=0.5614360570907593),
    'glm4': dict(nl=40, kout=39, tc_mean_cos=0.5640953183174133),
}

SRC = {
    'p3157_4b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz'),
    'p3157_14b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-14b\collect.npz'),
    'p3157_glm4': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\glm4\collect.npz'),
    'p3183_result': os.path.join(SRC_DIR, r'phase3183\g5b8_dv_lowrank\result.json'),
    'p3178_result': os.path.join(SRC_DIR, r'phase3178\g5b3_ftr14_a2_attribution\result.json'),
    'registry_v122': os.path.join(SRC_DIR, r'phase3199\g5e1_ftr24_heldout_upgrade\atlas_registry_v1_22.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3183_result': '0ab70579', 'p3178_result': 'abae296e',
    'registry_v122': '9fd87433', 'registry_3162': '00f15e98',
}
CONTENT_3183 = ('c55e4e5e', '7196fa8a')
CONTENT_3178 = ('4f521d9b', '1f166645')

# sealed 3178 scalars (4dp medians probed from sealed p3178 result BEFORE
# freeze; exch_mean bitwise; cos_a2 4dp) - measured-anchor discipline
ANCH_3178 = {
    'qwen3-4b': dict(exch=1.2728174525072664, cos_train_a2=0.7834,
                     cos_held_isa=0.6689, cos_held_hasa=0.6668,
                     med_isa=0.7696, med_hasa=0.7429),
    'qwen3-14b': dict(exch=1.2728174525072664, cos_train_a2=0.7966,
                      cos_held_isa=0.6554, cos_held_hasa=0.6773,
                      med_isa=0.7564, med_hasa=0.7855),
    'glm4-9b': dict(exch=1.2728174525072664, cos_train_a2=0.7942,
                    cos_held_isa=0.6442, cos_held_hasa=0.6449,
                    med_isa=0.7762, med_hasa=0.7885),
}

DESIGN = {
    'phase': '3200', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3199 tail G5-E2 (frozen before any observation; '
              'protocol refined pre-freeze: sealed-3178-null-fold evaluation '
              'instead of OOV-row projection - FTR-23 has no OOV-panel direction)',
    'data': 'sealed p3178 result (null folds cos_out, 16 x 2 x 3) + sealed 3157 '
            'collect npz x3 + sealed 3183 result + sealed registry v1.22; zero GPU',
    'control_anchor': 'sealed 3178 scalars re-read vs pinned 4dp constants '
                      '(ANCH_3178: med cos_out x6 units, cos_train_a2, '
                      'cos_held_a2 x6, exch_mean bitwise) drift == 0; sealed '
                      '3183 dv scalars (cos_rl/norm_dv) bitwise via build_dvu; '
                      '3157 tc drift < 1e-9',
    'measurements': 'per (model, relation): median/min/max of sealed null '
                    'cos_out over 16 entity-LOO folds (held entity cells vs '
                    'within-relation mean direction, 3178 protocol verbatim)',
    'gate': 'HOLDOUT_PREDICTIVE_FTR23 iff median cos_out >= 0.5 in ALL 6 '
            '(model, relation) units => FTR-23 E1_repeatable -> E2_held_out '
            '(evidence_level change + anchors append role=held_out + statement '
            'append + upgrade_log #9), registry v1.22->v1.23, atlas_v1_25.html; '
            'else counter_evidence append, E level UNCHANGED E1_repeatable',
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
        json.dump({'phase': 3200, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def unit(v):
    return v / (float(np.linalg.norm(v)) + 1e-18)


def unit_mean(rows_dn):
    v = rows_dn.mean(0)
    return v / (float(np.linalg.norm(v)) + 1e-18)


def build_dvu(m, sealed_3183):
    """dvu per model, 3157/3183 verbatim (3188 recipe) - device anchor."""
    A = ANCH_3157[ANCH_KEY[m]]
    z = np.load(SRC[NPZ57_KEY[m]])
    H = z['H'].astype(np.float32)
    NROWS, NH, D = H.shape
    NL = NH - 1
    KOUT = NL - 1
    assert NROWS == 128 and (NL, KOUT) == (A['nl'], A['kout']), ('layer', m)
    EI = z['ei'].astype(int)
    RL = z['rel'].astype(int)
    PL = z['pol'].astype(int)
    CX = z['ctx'].astype(int)
    IDX = {(int(EI[i]), int(RL[i]), int(PL[i]), int(CX[i])): i for i in range(NROWS)}
    assert len(IDX) == 128
    dC = []
    for ei in range(16):
        for rl in range(2):
            for pl in range(2):
                dC.append(H[IDX[(ei, rl, pl, 1)], KOUT] - H[IDX[(ei, rl, pl, 0)], KOUT])
    dC = np.array(dC)
    assert np.isfinite(dC).all()
    dCn = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-18)
    cosM = dCn @ dCn.T
    tri = cosM[np.triu_indices(len(dC), 1)]
    tc_full = float(tri.mean())
    assert abs(tc_full - A['tc_mean_cos']) < DEV_TOL, ('tc drift', m)
    RL_rows = np.array([(i // 4, (i % 4) // 2, i % 2) for i in range(64)])
    v_isa = unit_mean(dCn[RL_rows[:, 1] == 0])
    v_hasa = unit_mean(dCn[RL_rows[:, 1] == 1])
    cos_rl = float(v_isa @ v_hasa)
    dv = v_isa - v_hasa
    norm_dv = float(np.linalg.norm(dv))
    E83 = sealed_3183['per_model'][m]
    d83 = max(abs(cos_rl - E83['cos_rl']), abs(norm_dv - E83['norm_dv']))
    assert d83 < REPL_TOL, ('3183 dv replication', m, d83)
    del z, H
    return unit(dv), tc_full, d83


def run_control(sealed_3178):
    """Control anchor: sealed 3178 scalars vs pinned 4dp constants."""
    out = {}
    for m in MODELS:
        pm = sealed_3178['arm']['per_model'][m]
        A = ANCH_3178[m]
        drift = abs(sealed_3178['arm']['exch_mean'] - A['exch'])
        drift = max(drift, abs(round(pm['folds']['isa']['cos_train_a2'], 4) - A['cos_train_a2']))
        drift = max(drift, abs(round(pm['folds']['isa']['cos_held_a2'], 4) - A['cos_held_isa']))
        drift = max(drift, abs(round(pm['folds']['hasa']['cos_held_a2'], 4) - A['cos_held_hasa']))
        meds = {}
        mins = {}
        maxs = {}
        for rel, key in (('isa', 'med_isa'), ('hasa', 'med_hasa')):
            folds = pm['null'][rel]['folds'] if isinstance(pm['null'][rel], dict) else pm['null'][rel]
            co = sorted(f['cos_out'] for f in folds)
            n = len(co)
            assert n == 16, ('fold count', m, rel, n)
            med = (co[n // 2 - 1] + co[n // 2]) / 2 if n % 2 == 0 else co[n // 2]
            meds[rel] = med
            mins[rel] = co[0]
            maxs[rel] = co[-1]
            drift = max(drift, abs(round(med, 4) - A[key]))
        assert drift == 0.0, ('3178 control drift', m, drift)
        out[m] = dict(drift=drift, med_isa=meds['isa'], med_hasa=meds['hasa'],
                      min_isa=mins['isa'], min_hasa=mins['hasa'],
                      max_isa=maxs['isa'], max_hasa=maxs['hasa'])
        log('  control 3178 %s: drift=0 (med isa %.4f hasa %.4f, min %.4f/%.4f)'
            % (m, meds['isa'], meds['hasa'], mins['isa'], mins['hasa']))
    log('CONTROL: sealed 3178 scalars == pinned 4dp constants x3/3 (+exch bitwise)')
    return out


# ---------------- registry v1.22 -> v1.23 + atlas_v1_25 ----------------
CREATED_STR = None


def finish_registry(control, res8, seal8, design_sha):
    global CREATED_STR
    CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    reg22 = json.loads(open(SRC['registry_v122'], 'rb').read().decode('utf-8'))
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg22['version'] == '1.22' and len(reg22['features']) == 26
    reg23 = copy.deepcopy(reg22)
    reg23['version'] = '1.23'
    reg23['supersedes'] = 'atlas_registry_v1_22.json (9fd87433, Phase 3199)'
    ctl = control
    f23_old = [f for f in reg22['features'] if f['id'] == 'FTR-23'][0]
    f23 = copy.deepcopy(f23_old)
    gate_all = all(min(ctl[m]['med_isa'], ctl[m]['med_hasa']) >= G_MED for m in MODELS)
    if gate_all:
        anchor = {
            'src': 'p3178', 'phase': 3200,
            'role': 'held_out',
            'tags': ['zero_gpu', 'sealed_reuse', 'e_level_upgrade'],
            'asserts': {
                'med_cos_out.isa.qwen3-4b': ctl['qwen3-4b']['med_isa'],
                'med_cos_out.isa.qwen3-14b': ctl['qwen3-14b']['med_isa'],
                'med_cos_out.isa.glm4-9b': ctl['glm4-9b']['med_isa'],
                'med_cos_out.hasa.qwen3-4b': ctl['qwen3-4b']['med_hasa'],
                'med_cos_out.hasa.qwen3-14b': ctl['qwen3-14b']['med_hasa'],
                'med_cos_out.hasa.glm4-9b': ctl['glm4-9b']['med_hasa'],
                'gate_heldout_med_cos_out_ftr23': G_MED,
                'e_level_before': 'E1_repeatable', 'e_level_after': 'E2_held_out',
                'res_sha8': res8, 'seal_sha8': seal8}}
        f23['anchors'] = f23_old['anchors'] + [anchor]
        f23['evidence_level'] = 'E2_held_out'
        f23['statement'] = f23_old['statement'] + (
            '；3200 G5-E2 held_out 升格臂：3178 实体留出 null 封存读数复用——关系内'
            '均值方向（留出实体重建）对留出实体 2 细胞 cos_out 的 16-fold 中位数 '
            'isa ' + '/'.join('%.4f' % ctl[m]['med_isa'] for m in MODELS) +
            ' / hasa ' + '/'.join('%.4f' % ctl[m]['med_hasa'] for m in MODELS) +
            '（门 ≥0.5 全 6 单元；min fold ≥0.61）——关系内几何对实体留出可预测，'
            'E1_repeatable 升 E2_held_out（升级链 #9）；对照锚：3178 标量 4dp 逐位'
            '回读 + 3183 dv 标量逐位复现。')
        reg23['features'] = [f23 if f['id'] == 'FTR-23' else f
                             for f in reg22['features']]
        reg23['upgrade_log'] = reg22['upgrade_log'] + [{
            'node': 'FTR-23', 'change': 'E1_repeatable -> E2_held_out',
            'reason': 'G5-E2 held_out arm (Phase 3200): sealed 3178 entity-LOO '
                      'null folds cos_out (held entity cells vs within-relation '
                      'mean direction) median >= 0.5 in all 6 (model, relation) '
                      'units, min fold >= 0.61; panel-internal held-out '
                      'predictiveness established (zero GPU, sealed reuse; '
                      '3178 scalar 4dp re-read + 3183 dv anchors bitwise)'}]
        disposition = 'v1_22_to_v1_23_ftr23_e2_held_out'
    else:
        ce = ('3200 G5-E2 held_out 升格臂未过：3178 封存 cos_out 16-fold 中位数 '
              '（门 ≥0.5 全 6 单元未过）——关系内均值方向对留出实体细胞预测力不足，'
              'E1_repeatable 维持。')
        f23['counter_evidence'] = f23_old['counter_evidence'] + [ce]
        reg23['features'] = [f23 if f['id'] == 'FTR-23' else f
                             for f in reg22['features']]
        disposition = 'v1_22_to_v1_23_ftr23_ce_not_predictive'
    # preservation asserts
    for f in reg22['features']:
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg23['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        if f['id'] == 'FTR-23':
            fa = json.loads(a)
            assert b != a
            if gate_all:
                assert f23['statement'].startswith(fa['statement'])
                assert f23['counter_evidence'] == fa['counter_evidence']
                assert json.dumps(f23['anchors'][:-1], ensure_ascii=False, sort_keys=True) == \
                    json.dumps(fa['anchors'], ensure_ascii=False, sort_keys=True)
                assert f23['evidence_level'] == 'E2_held_out' and fa['evidence_level'] == 'E1_repeatable'
            else:
                assert f23['counter_evidence'][:-1] == fa['counter_evidence']
                assert f23['statement'] == fa['statement']
                assert f23['anchors'] == fa['anchors']
                assert f23['evidence_level'] == fa['evidence_level'] == 'E1_repeatable'
        else:
            assert a == b, ('feature not preserved', f['id'])
    if gate_all:
        assert json.dumps(reg23['upgrade_log'][:-1], ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg22['upgrade_log'], ensure_ascii=False, sort_keys=True)
    else:
        assert json.dumps(reg23['upgrade_log'], ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg22['upgrade_log'], ensure_ascii=False, sort_keys=True)
    for k in ('evidence_level_taxonomy', 'principles', 'schema',
              'models_mainline', 'models_pending', 'provenance', 'created',
              'phase', 'name', 'design_sha8'):
        assert json.dumps(reg22.get(k), ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg23.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    assert json.dumps(reg22['failures'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg23['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'
    rp = os.path.join(OUTDIR, 'atlas_registry_v1_23.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg23, ensure_ascii=False, indent=1))
    reg23_sha = sha8_file(rp)
    log('registry v1.23 written sha8 %s (%s)' % (reg23_sha, disposition))

    meta_sha = {'reg23': reg23_sha}
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.25 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.25 &mdash; 语义机制图谱（registry v1.23 渲染）</h1>')
    H.append('<div class="sub">Phase 3200 G5-E2 &middot; FTR-23 held_out 升格臂（%s） &middot; '
             'registry v1.23 sha8=%s &middot; v1.22 sha8=%s &middot; v1.21 sha8=%s &middot; v1.20 sha8=%s &middot; gap v1.6 sha8=%s'
             ' &middot; 源 p3178 %s + p3183 %s &middot; 渲染于 %s</div>' % (
                 'FTR-23 E1->E2_held_out 升格' if gate_all else 'FTR-23 counter_evidence 未过升格门',
                 span('meta.registry_v123_sha8', reg23_sha),
                 span('meta.registry_v122_sha8', '9fd87433'),
                 span('meta.registry_v121_sha8', '47cf01e0'),
                 span('meta.registry_v120_sha8', '2aac7e5e'),
                 span('meta.gap_v16_sha8', '0f6423e0'),
                 span('meta.p3178_result', SHA_ANCHOR['p3178_result']),
                 span('meta.p3183_result', SHA_ANCHOR['p3183_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg23['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg23['principles']):
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
    for f in reg23['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.23，%d 条）</h2>' % len(reg23['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg23['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg23['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg23['failures']:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td class="mono">%s</td><td>%s</td><td class="mono">%s</td></tr>' % (
            fl['id'], span(fl['id'] + '.kind', fl['kind']), span(fl['id'] + '.phase', fl['phase']),
            span(fl['id'] + '.text', fl['text']), span(fl['id'] + '.evidence', fl['evidence'])))
    H.append('</table>')
    gl15 = json.loads(open(os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'), encoding='utf-8').read())
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
    H.append('<div class="footer">RDC/LPF atlas v1.25 &middot; 渲染自 atlas_registry_v1_23.json (%s)'
             ' + gap ledger v1.5 (4fb3f37d) &middot; 增量来源: Phase 3200 G5-E2 FTR-23 held_out 升格臂'
             '（数据源 p3178 %s + p3183 %s）'
             '&middot; 本文件为 Phase 3200 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg23_sha, SHA_ANCHOR['p3178_result'], SHA_ANCHOR['p3183_result']))
    H.append('</body></html>')
    html_text = '\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_25.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v122(reg23, nodes, gl15, meta_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_25.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3200, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'gate_upgraded': bool(gate_all), 'control': control,
                'per_model': {m: ANCH_3178[m] for m in MODELS}},
        'counts': {'features': len(reg23['features']), 'failures': len(reg23['failures']),
                   'upgrades': len(reg23['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v123_file': 'atlas_registry_v1_23.json', 'registry_v123_sha8': reg23_sha,
        'registry_v122_sha8_ref': '9fd87433',
        'html_file': 'atlas_v1_25.html', 'html_sha8': html_sha,
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


def flatten_v122(reg, nodes, gl, meta_sha):
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
    d['meta.registry_v123_sha8'] = meta_sha['reg23']
    d['meta.registry_v122_sha8'] = '9fd87433'
    d['meta.registry_v121_sha8'] = '47cf01e0'
    d['meta.registry_v120_sha8'] = '2aac7e5e'
    d['meta.gap_v16_sha8'] = '0f6423e0'
    d['meta.p3178_result'] = SHA_ANCHOR['p3178_result']
    d['meta.p3183_result'] = SHA_ANCHOR['p3183_result']
    d['meta.created'] = CREATED_STR
    return d


def main():
    design_sha = freeze()
    log('== Phase 3200 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    sealed_3183 = json.load(open(SRC['p3183_result'], encoding='utf-8'))
    assert sealed_3183['res_sha8'] == CONTENT_3183[0] and sealed_3183['seal_sha8'] == CONTENT_3183[1]
    sealed_3178 = json.load(open(SRC['p3178_result'], encoding='utf-8'))
    assert sealed_3178['res_sha8'] == CONTENT_3178[0] and sealed_3178['seal_sha8'] == CONTENT_3178[1]
    log('sealed results double-locked: 3183/3178')
    for m in MODELS:
        build_dvu(m, sealed_3183)
    log('device anchors: 3157 tc + 3183 dv bitwise x3')

    control = run_control(sealed_3178)
    n_units = sum(1 for m in MODELS for rel in ('isa', 'hasa')
                  if control[m]['med_%s' % rel] >= G_MED)
    gate_all = bool(n_units >= 6)
    if gate_all:
        verdict = ('ftr23_heldout_predictive (med cos_out isa %s / hasa %s >= 0.5 '
                   'in 6/6 units; FTR-23 E1_repeatable -> E2_held_out, '
                   'upgrade_log #9, registry v1.22 -> v1.23)'
                   % ('/'.join('%.4f' % control[m]['med_isa'] for m in MODELS),
                      '/'.join('%.4f' % control[m]['med_hasa'] for m in MODELS)))
    else:
        verdict = ('ftr23_heldout_not_predictive (med cos_out gate fail in %d/6 '
                   'units; FTR-23 counter_evidence append, E level unchanged '
                   'E1_repeatable, registry v1.22 -> v1.23)' % n_units)
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3200, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'gates': dict(GATE, upgraded=bool(gate_all), n_units_pass=n_units),
        'control': control,
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
