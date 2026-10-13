# -*- coding: utf-8 -*-
"""Phase 3201 (G5-E3): E-line closing audit arm (registry v1.23 -> v1.24,
zero GPU, sealed registry re-read; audit + methodology closing).

Per prereg in MEMO Phase 3200 tail G5-E3 (frozen before any observation;
audit criteria refined pre-freeze after a measured probe of sealed registry
v1.23 anchor roles - probe registered honestly):
  AUDIT (pinned): A1 - every feature evidence_level in {E0_candidate,
  E1_repeatable, E2_predictive, E2_held_out, E3_causal_scoped} and counts
  E2_held_out==3 / E3==3 / E1==3 / E2_predictive==17; A2a - FTR-23/25/26
  each carry a role=held_out anchor; A2b - FTR-10/11/16 each carry
  role=primary_intervention; A2c - FTR-04/14/24 each have >=1
  counter_evidence; A3 - upgrade_log node sequence == 9 pinned entries.
  CLOSING (A4): FTR-24 scope_limits append ONLY (E-line methodology
  verdict: held-out predictability geometric divide - class-mean /
  within-relation structure carries it (3199+3200 evidence), between-
  relation offsets and relation directions do not; plus taxonomy note:
  E2_held_out is the held-out strengthened tier of E2_predictive, legacy
  values byte-preserved); upgrade_log / failures / all other features
  UNCHANGED.
  GATE (pinned pre-observation): E_LINE_CLOSED iff A1-A3 audit clean (0
  contradictions) AND A4 append verified => registry v1.23 -> v1.24,
  atlas_v1_26.html; else contradictions registered, report-only (registry
  v1.23 unchanged).
  Appendix inventory: all 5 items GPU/crossline dependent (zero-GPU
  candidate set empty), registered in MEMO + final_result.
  Control anchors: sealed 3178 scalars 4dp re-read drift == 0 (ANCH_3178
  carried from 3200) + content double lock + 3183 dv bitwise (build_dvu).
  SMOKE (env P3201_SMOKE=1): registry/html writes skipped; checks only.
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
NAME = 'g5e3_eline_audit'
OUTDIR = os.path.join(SRC_DIR, 'phase3201', NAME)
SMOKE = os.environ.get('P3201_SMOKE') == '1'
LOG = []

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
NPZ57_KEY = {'qwen3-4b': 'p3157_4b', 'qwen3-14b': 'p3157_14b', 'glm4-9b': 'p3157_glm4'}
ANCH_KEY = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4-9b': 'glm4'}

DEV_TOL = 1e-9
REPL_TOL = 1e-9
GATE = {'audit_clean': True, 'closing_append': True,
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
    'registry_v123': os.path.join(SRC_DIR, r'phase3200\g5e2_ftr23_heldout_upgrade\atlas_registry_v1_23.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3183_result': '0ab70579', 'p3178_result': 'abae296e',
    'registry_v123': '5f5b9e73', 'registry_3162': '00f15e98',
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

UPG_SEQ_PINNED = ['N04->FTR-13', 'N05->FTR-12', 'N13->FTR-16', 'FTR-06',
                  'FTR-08', 'FTR-15', 'FTR-25', 'FTR-26', 'FTR-23']
E2Held_IDS = ('FTR-23', 'FTR-25', 'FTR-26')
E3_IDS = ('FTR-10', 'FTR-11', 'FTR-16')
E1_IDS = ('FTR-04', 'FTR-14', 'FTR-24')
E_LEVEL_ALLOWED = ('E0_candidate', 'E1_repeatable', 'E2_predictive',
                   'E2_held_out', 'E3_causal_scoped')
SCOPE_APPEND = ('；3201 G5-E3 E 线收官定论（3199+3200 双臂证据）：held-out 可预测性的'
                '几何分界——类均值/关系内结构承载（FTR-25/26 OOV 面板 med|cos|≥0.3、'
                'FTR-23 关系内实体留出 med cos 0.74–0.79），关系间偏移方向与关系方向'
                '不承载跨面板预测（本特征 dvu med|cos_held| 0.035–0.060 ≪0.3）；E1 残余'
                '可证伪记录完备（FTR-04 noise_level 已闭 3181、FTR-14 F16 gate fail '
                '3177+3178 归因、本特征 3199 counter_evidence #4）；口径注记：'
                'E2_held_out 为 E2_predictive 的 held-out 强化档（FTR-23/25/26），'
                'legacy E2_predictive 值字节保全不动。')

DESIGN = {
    'phase': '3201', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3200 tail G5-E3 (frozen before any observation; '
              'audit criteria refined pre-freeze after measured probe of sealed '
              'registry v1.23 anchor roles)',
    'data': 'sealed registry v1.23 (5f5b9e73) + sealed p3178 result (4dp control '
            're-read) + sealed 3157 npz x3 + sealed 3183 result + gap ledger '
            'v1.5; zero GPU',
    'control_anchor': 'sealed 3178 scalars re-read vs pinned 4dp constants '
                      '(ANCH_3178 carried from 3200) drift == 0 x3; sealed 3183 '
                      'dv scalars bitwise via build_dvu; 3157 tc drift < 1e-9',
    'measurements': 'A1 evidence_level domain + counts (E2_held_out 3 / E3 3 / '
                    'E1 3 / E2_predictive 17); A2a held_out anchors on '
                    'FTR-23/25/26; A2b primary_intervention on FTR-10/11/16; '
                    'A2c counter_evidence >=1 on FTR-04/14/24; A3 upgrade_log '
                    'node sequence == 9 pinned entries',
    'gate': 'E_LINE_CLOSED iff A1-A3 audit clean (0 contradictions) AND A4 '
            'FTR-24 scope_limits append verified (startswith + 25 legacy '
            'features byte-preserved + evidence_level/upgrade_log/failures '
            'unchanged) => registry v1.23->v1.24, atlas_v1_26.html; else '
            'contradictions registered, report-only',
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
        json.dump({'phase': 3201, 'name': NAME, 'design_sha8': dsha[:8],
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


def run_audit(reg):
    """Pinned audit A1-A3 over sealed registry v1.23 (0 contradictions)."""
    feats = reg['features']
    out = {'contradictions': []}
    bad = [f['id'] for f in feats if f['evidence_level'] not in E_LEVEL_ALLOWED]
    if bad:
        out['contradictions'].append(('A1_domain', bad))
    counts = {}
    for f in feats:
        counts[f['evidence_level']] = counts.get(f['evidence_level'], 0) + 1
    out['counts'] = counts
    if counts.get('E2_held_out') != 3 or counts.get('E3_causal_scoped') != 3             or counts.get('E1_repeatable') != 3 or counts.get('E2_predictive') != 17:
        out['contradictions'].append(('A1_counts', counts))
    for fid in E2Held_IDS:
        f = [x for x in feats if x['id'] == fid][0]
        if not any(a['role'] == 'held_out' for a in f['anchors']):
            out['contradictions'].append(('A2a_missing_held_out', fid))
    for fid in E3_IDS:
        f = [x for x in feats if x['id'] == fid][0]
        if not any(a['role'] == 'primary_intervention' for a in f['anchors']):
            out['contradictions'].append(('A2b_missing_intervention', fid))
    for fid in E1_IDS:
        f = [x for x in feats if x['id'] == fid][0]
        if len(f['counter_evidence']) < 1:
            out['contradictions'].append(('A2c_missing_ce', fid))
    nodes = [u['node'] for u in reg['upgrade_log']]
    if nodes != UPG_SEQ_PINNED:
        out['contradictions'].append(('A3_upgrade_seq', nodes))
    out['audit_clean'] = not out['contradictions']
    log('  A1 domain+counts: %s' % ('OK' if not any(c[0].startswith('A1') for c in out['contradictions']) else 'FAIL'))
    log('  A2a held_out anchors x3: %s | A2b interventions x3: %s | A2c ce x3: %s'
        % ('OK' if not any(c[0] == 'A2a_missing_held_out' for c in out['contradictions']) else 'FAIL',
           'OK' if not any(c[0] == 'A2b_missing_intervention' for c in out['contradictions']) else 'FAIL',
           'OK' if not any(c[0] == 'A2c_missing_ce' for c in out['contradictions']) else 'FAIL'))
    log('  A3 upgrade_log sequence (9 entries): %s' % ('OK' if not any(c[0] == 'A3_upgrade_seq' for c in out['contradictions']) else 'FAIL'))
    log('AUDIT: %s (%d contradictions)' % ('CLEAN' if out['audit_clean'] else 'DIRTY', len(out['contradictions'])))
    return out


# ---------------- registry v1.22 -> v1.23 + atlas_v1_25 ----------------
CREATED_STR = None


def finish_registry(control, res8, seal8, design_sha):
    global CREATED_STR
    CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    reg23 = json.loads(open(SRC['registry_v123'], 'rb').read().decode('utf-8'))
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg23['version'] == '1.23' and len(reg23['features']) == 26
    reg24 = copy.deepcopy(reg23)
    reg24['version'] = '1.24'
    reg24['supersedes'] = 'atlas_registry_v1_23.json (5f5b9e73, Phase 3200)'
    audit = control
    f24_old = [f for f in reg23['features'] if f['id'] == 'FTR-24'][0]
    f24 = copy.deepcopy(f24_old)
    gate_all = bool(audit['audit_clean'])
    if gate_all:
        f24['scope_limits'] = f24_old['scope_limits'] + SCOPE_APPEND
        reg24['features'] = [f24 if f['id'] == 'FTR-24' else f
                             for f in reg23['features']]
        disposition = 'v1_23_to_v1_24_eline_closed'
    else:
        reg24 = copy.deepcopy(reg23)
        disposition = 'v1_23_unchanged_audit_dirty_report_only'
    # preservation asserts
    for f in reg23['features']:
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg24['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        if f['id'] == 'FTR-24':
            fa = json.loads(a)
            if gate_all:
                assert b != a
                assert f24['scope_limits'].startswith(fa['scope_limits'])
                assert f24['scope_limits'][:-len(SCOPE_APPEND)] == fa['scope_limits']
                assert f24['evidence_level'] == fa['evidence_level'] == 'E1_repeatable'
                assert f24['statement'] == fa['statement']
                assert f24['anchors'] == fa['anchors']
                assert f24['counter_evidence'] == fa['counter_evidence']
            else:
                assert a == b
        else:
            assert a == b, ('feature not preserved', f['id'])
    assert json.dumps(reg24['upgrade_log'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg23['upgrade_log'], ensure_ascii=False, sort_keys=True)
    for k in ('evidence_level_taxonomy', 'principles', 'schema',
              'models_mainline', 'models_pending', 'provenance', 'created',
              'phase', 'name', 'design_sha8'):
        assert json.dumps(reg24.get(k), ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg23.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    assert json.dumps(reg24['failures'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg23['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'
    rp = os.path.join(OUTDIR, 'atlas_registry_v1_24.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg24, ensure_ascii=False, indent=1))
    reg24_sha = sha8_file(rp)
    log('registry v1.24 written sha8 %s (%s)' % (reg24_sha, disposition))

    meta_sha = {'reg24': reg24_sha}
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.26 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.26 &mdash; 语义机制图谱（registry v1.24 渲染）</h1>')
    H.append('<div class="sub">Phase 3201 G5-E3 &middot; E 线收官审计臂（%s） &middot; '
             'registry v1.24 sha8=%s &middot; v1.23 sha8=%s &middot; v1.22 sha8=%s &middot; v1.21 sha8=%s &middot; v1.20 sha8=%s &middot; gap v1.6 sha8=%s'
             ' &middot; 源 p3178 %s + p3183 %s &middot; 渲染于 %s</div>' % (
                 '审计 CLEAN + E_LINE_CLOSED（FTR-24 scope_limits 定论 append）' if gate_all else '审计矛盾登记，report-only',
                 span('meta.registry_v124_sha8', reg24_sha),
                 span('meta.registry_v123_sha8', '5f5b9e73'),
                 span('meta.registry_v122_sha8', '9fd87433'),
                 span('meta.registry_v121_sha8', '47cf01e0'),
                 span('meta.registry_v120_sha8', '2aac7e5e'),
                 span('meta.gap_v16_sha8', '0f6423e0'),
                 span('meta.p3178_result', SHA_ANCHOR['p3178_result']),
                 span('meta.p3183_result', SHA_ANCHOR['p3183_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg24['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg24['principles']):
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
    for f in reg24['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.24，%d 条）</h2>' % len(reg24['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg24['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg24['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg24['failures']:
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
    H.append('<div class="footer">RDC/LPF atlas v1.26 &middot; 渲染自 atlas_registry_v1_24.json (%s)'
             ' + gap ledger v1.5 (4fb3f37d) &middot; 增量来源: Phase 3201 G5-E3 E 线收官审计臂'
             '（数据源 p3178 %s + p3183 %s）'
             '&middot; 本文件为 Phase 3201 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg24_sha, SHA_ANCHOR['p3178_result'], SHA_ANCHOR['p3183_result']))
    H.append('</body></html>')
    html_text = '\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_26.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v122(reg24, nodes, gl15, meta_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_26.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3201, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'eline_closed': bool(gate_all), 'audit': audit,
                'appendix_inventory': {a['id']: a['status'] for a in gl15['appendix']}},
        'counts': {'features': len(reg24['features']), 'failures': len(reg24['failures']),
                   'upgrades': len(reg24['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v124_file': 'atlas_registry_v1_24.json', 'registry_v124_sha8': reg24_sha,
        'registry_v123_sha8_ref': '5f5b9e73',
        'html_file': 'atlas_v1_26.html', 'html_sha8': html_sha,
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
    d['meta.registry_v124_sha8'] = meta_sha['reg24']
    d['meta.registry_v123_sha8'] = '5f5b9e73'
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
    log('== Phase 3201 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
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
    reg_v123 = json.load(open(SRC['registry_v123'], encoding='utf-8'))
    assert reg_v123['version'] == '1.23' and len(reg_v123['features']) == 26
    log('sealed registry v1.23 loaded (%s)' % SRC['registry_v123'])

    control = run_audit(reg_v123)
    gate_all = bool(control['audit_clean'])
    if gate_all:
        verdict = ('eline_closed (audit A1-A3 clean 0 contradictions; FTR-24 '
                   'scope_limits methodology verdict append; registry v1.23 -> '
                   'v1.24, atlas_v1_26)')
    else:
        verdict = ('eline_audit_dirty (%d contradictions registered; report-only, '
                   'registry v1.23 unchanged)' % len(control['contradictions']))
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3201, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'gates': dict(GATE, eline_closed=bool(gate_all),
                      n_contradictions=len(control['contradictions'])),
        'audit': control,
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
