# -*- coding: utf-8 -*-
"""Create phase3201_g5e3_eline_audit.py from 3200 (ordered patches)."""
import io
import hashlib

SRC = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3200_g5e2_ftr23_heldout_upgrade.py'
DST = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3201_g5e3_eline_audit.py'
t = io.open(SRC, encoding='utf-8').read()

def rep(old, new, n=1):
    global t
    c = t.count(old)
    assert c == n, ('patch count', c, n, old[:90])
    t = t.replace(old, new)

# ---- P1 docstring ----
rep('''"""Phase 3200 (G5-E2): FTR-23 held_out upgrade arm (registry v1.22 -> v1.23,
zero GPU, sealed 3178 null-fold scalars reuse; 3176 batch-upgrade precedent).''',
'''"""Phase 3201 (G5-E3): E-line closing audit arm (registry v1.23 -> v1.24,
zero GPU, sealed registry re-read; audit + methodology closing).''')

rep('''Per prereg in MEMO Phase 3199 tail G5-E2 (frozen before any observation;
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
"""''',
'''Per prereg in MEMO Phase 3200 tail G5-E3 (frozen before any observation;
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
"""''')

# ---- P2 name/outdir/smoke ----
rep("""NAME = 'g5e2_ftr23_heldout_upgrade'
OUTDIR = os.path.join(SRC_DIR, 'phase3200', NAME)
SMOKE = os.environ.get('P3200_SMOKE') == '1'""",
"""NAME = 'g5e3_eline_audit'
OUTDIR = os.path.join(SRC_DIR, 'phase3201', NAME)
SMOKE = os.environ.get('P3201_SMOKE') == '1'""")

# ---- P3 GATE ----
rep("""GATE = {'heldout_med_cos_out_ftr23': G_MED, 'units_needed': 6,
        'dev_tol': DEV_TOL, 'repl_tol': REPL_TOL}""",
"""GATE = {'audit_clean': True, 'closing_append': True,
        'dev_tol': DEV_TOL, 'repl_tol': REPL_TOL}""")

# ---- P4 SRC: registry source -> v1.23; drop 3157 npz? keep for build_dvu ----
rep("""    'registry_v122': os.path.join(SRC_DIR, r'phase3199\\g5e1_ftr24_heldout_upgrade\\atlas_registry_v1_22.json'),""",
"""    'registry_v123': os.path.join(SRC_DIR, r'phase3200\\g5e2_ftr23_heldout_upgrade\\atlas_registry_v1_23.json'),""")

rep("""    'registry_v122': '9fd87433', 'registry_3162': '00f15e98',""",
"""    'registry_v123': '5f5b9e73', 'registry_3162': '00f15e98',""")

# ---- P5 pinned audit constants (probed pre-freeze from sealed v1.23) ----
rep("""DESIGN = {
    'phase': '3200', 'name': NAME,""",
"""UPG_SEQ_PINNED = ['N04->FTR-13', 'N05->FTR-12', 'N13->FTR-16', 'FTR-06',
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
    'phase': '3201', 'name': NAME,""")

rep("""    'prereg': 'AGI_GPT5_MEMO Phase 3199 tail G5-E2 (frozen before any observation; '
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
            'else counter_evidence append, E level UNCHANGED E1_repeatable',""",
"""    'prereg': 'AGI_GPT5_MEMO Phase 3200 tail G5-E3 (frozen before any observation; '
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
            'contradictions registered, report-only',""")

# ---- P6 freeze phase ----
rep("""        json.dump({'phase': 3200, 'name': NAME, 'design_sha8': dsha[:8],""",
"""        json.dump({'phase': 3201, 'name': NAME, 'design_sha8': dsha[:8],""")

# ---- P7 run_control -> run_audit ----
rep('''def run_control(sealed_3178):
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
    return out''',
'''def run_audit(reg):
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
    if counts.get('E2_held_out') != 3 or counts.get('E3_causal_scoped') != 3 \
            or counts.get('E1_repeatable') != 3 or counts.get('E2_predictive') != 17:
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
    return out''')

# ---- P8 finish_registry header ----
rep('''    reg22 = json.loads(open(SRC['registry_v122'], 'rb').read().decode('utf-8'))
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg22['version'] == '1.22' and len(reg22['features']) == 26
    reg23 = copy.deepcopy(reg22)
    reg23['version'] = '1.23'
    reg23['supersedes'] = 'atlas_registry_v1_22.json (9fd87433, Phase 3199)'
    ctl = control
    f23_old = [f for f in reg22['features'] if f['id'] == 'FTR-23'][0]
    f23 = copy.deepcopy(f23_old)
    gate_all = all(min(ctl[m]['med_isa'], ctl[m]['med_hasa']) >= G_MED for m in MODELS)''',
'''    reg23 = json.loads(open(SRC['registry_v123'], 'rb').read().decode('utf-8'))
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg23['version'] == '1.23' and len(reg23['features']) == 26
    reg24 = copy.deepcopy(reg23)
    reg24['version'] = '1.24'
    reg24['supersedes'] = 'atlas_registry_v1_23.json (5f5b9e73, Phase 3200)'
    audit = control
    f24_old = [f for f in reg23['features'] if f['id'] == 'FTR-24'][0]
    f24 = copy.deepcopy(f24_old)
    gate_all = bool(audit['audit_clean'])''')

# ---- P9 closing branch ----
rep("""    if gate_all:
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
        disposition = 'v1_22_to_v1_23_ftr23_ce_not_predictive'""",
"""    if gate_all:
        f24['scope_limits'] = f24_old['scope_limits'] + SCOPE_APPEND
        reg24['features'] = [f24 if f['id'] == 'FTR-24' else f
                             for f in reg23['features']]
        disposition = 'v1_23_to_v1_24_eline_closed'
    else:
        reg24 = copy.deepcopy(reg23)
        disposition = 'v1_23_unchanged_audit_dirty_report_only'""")

# ---- P10 preservation asserts ----
rep(r"""    # preservation asserts
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
        json.dumps(reg23['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'""",
r"""    # preservation asserts
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
        assert json.dumps(reg23.get(k), ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg24.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    assert json.dumps(reg23['failures'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg24['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'""")

# ---- P11 registry write + meta ----
rep('''    rp = os.path.join(OUTDIR, 'atlas_registry_v1_23.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg23, ensure_ascii=False, indent=1))
    reg23_sha = sha8_file(rp)
    log('registry v1.23 written sha8 %s (%s)' % (reg23_sha, disposition))

    meta_sha = {'reg23': reg23_sha}''',
'''    rp = os.path.join(OUTDIR, 'atlas_registry_v1_24.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg24, ensure_ascii=False, indent=1))
    reg24_sha = sha8_file(rp)
    log('registry v1.24 written sha8 %s (%s)' % (reg24_sha, disposition))

    meta_sha = {'reg24': reg24_sha}''')

# ---- P12 HTML headers ----
rep("""    H.append('<title>Atlas v1.25 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
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
                 span('meta.created', CREATED_STR)))""",
"""    H.append('<title>Atlas v1.26 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.26 &mdash; 语义机制图谱（registry v1.24 渲染）</h1>')
    H.append('<div class="sub">Phase 3201 G5-E3 &middot; E 线收官审计臂（%s） &middot; '
             'registry v1.24 sha8=%s &middot; v1.23 sha8=%s &middot; v1.22 sha8=%s &middot; v1.21 sha8=%s &middot; gap v1.6 sha8=%s'
             ' &middot; 源 p3178 %s + p3183 %s &middot; 渲染于 %s</div>' % (
                 '审计 CLEAN + E_LINE_CLOSED（FTR-24 scope_limits 定论 append）' if gate_all else '审计矛盾登记，report-only',
                 span('meta.registry_v124_sha8', reg24_sha),
                 span('meta.registry_v123_sha8', '5f5b9e73'),
                 span('meta.registry_v122_sha8', '9fd87433'),
                 span('meta.registry_v121_sha8', '47cf01e0'),
                 span('meta.gap_v16_sha8', '0f6423e0'),
                 span('meta.p3178_result', SHA_ANCHOR['p3178_result']),
                 span('meta.p3183_result', SHA_ANCHOR['p3183_result']),
                 span('meta.created', CREATED_STR)))""")

rep("""    H.append('<h2>跨模型稳定特征登记表（registry v1.23，%d 条）</h2>' % len(reg23['features']))""",
"""    H.append('<h2>跨模型稳定特征登记表（registry v1.24，%d 条）</h2>' % len(reg24['features']))""")

rep("""    H.append('<div class="footer">RDC/LPF atlas v1.25 &middot; 渲染自 atlas_registry_v1_23.json (%s)'
             ' + gap ledger v1.5 (4fb3f37d) &middot; 增量来源: Phase 3200 G5-E2 FTR-23 held_out 升格臂'
             '（数据源 p3178 %s + p3183 %s）'
             '&middot; 本文件为 Phase 3200 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg23_sha, SHA_ANCHOR['p3178_result'], SHA_ANCHOR['p3183_result']))""",
"""    H.append('<div class="footer">RDC/LPF atlas v1.26 &middot; 渲染自 atlas_registry_v1_24.json (%s)'
             ' + gap ledger v1.5 (4fb3f37d) &middot; 增量来源: Phase 3201 G5-E3 E 线收官审计臂'
             '（数据源 p3178 %s + p3183 %s）'
             '&middot; 本文件为 Phase 3201 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg24_sha, SHA_ANCHOR['p3178_result'], SHA_ANCHOR['p3183_result']))""")

rep("""    hp = os.path.join(OUTDIR, 'atlas_v1_25.html')""",
"""    hp = os.path.join(OUTDIR, 'atlas_v1_26.html')""")

rep("""    log('atlas_v1_25.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (""",
"""    log('atlas_v1_26.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (""")

# ---- P13 final dict ----
rep("""    final = {
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
        'html_file': 'atlas_v1_25.html', 'html_sha8': html_sha,""",
"""    final = {
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
        'html_file': 'atlas_v1_26.html', 'html_sha8': html_sha,""")

# ---- P14 flatten meta ----
rep("""    d['meta.registry_v123_sha8'] = meta_sha['reg23']
    d['meta.registry_v122_sha8'] = '9fd87433'""",
"""    d['meta.registry_v124_sha8'] = meta_sha['reg24']
    d['meta.registry_v123_sha8'] = '5f5b9e73'
    d['meta.registry_v122_sha8'] = '9fd87433'""")

# ---- P15 main: audit call + verdict ----
rep("""    control = run_control(sealed_3178)
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
                   'E1_repeatable, registry v1.22 -> v1.23)' % n_units)""",
"""    control = run_audit(reg_v123)
    gate_all = bool(control['audit_clean'])
    if gate_all:
        verdict = ('eline_closed (audit A1-A3 clean 0 contradictions; FTR-24 '
                   'scope_limits methodology verdict append; registry v1.23 -> '
                   'v1.24, atlas_v1_26)')
    else:
        verdict = ('eline_audit_dirty (%d contradictions registered; report-only, '
                   'registry v1.23 unchanged)' % len(control['contradictions']))""")

rep("""    sealed_3178 = json.load(open(SRC['p3178_result'], encoding='utf-8'))
    assert sealed_3178['res_sha8'] == CONTENT_3178[0] and sealed_3178['seal_sha8'] == CONTENT_3178[1]
    log('sealed results double-locked: 3183/3178')
    for m in MODELS:
        build_dvu(m, sealed_3183)
    log('device anchors: 3157 tc + 3183 dv bitwise x3')""",
"""    sealed_3178 = json.load(open(SRC['p3178_result'], encoding='utf-8'))
    assert sealed_3178['res_sha8'] == CONTENT_3178[0] and sealed_3178['seal_sha8'] == CONTENT_3178[1]
    log('sealed results double-locked: 3183/3178')
    for m in MODELS:
        build_dvu(m, sealed_3183)
    log('device anchors: 3157 tc + 3183 dv bitwise x3')
    reg_v123 = json.load(open(SRC['registry_v123'], encoding='utf-8'))
    assert reg_v123['version'] == '1.23' and len(reg_v123['features']) == 26
    log('sealed registry v1.23 loaded (%s)' % SRC['registry_v123'])""")

rep("""    summary = {
        'phase': 3200, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',""",
"""    summary = {
        'phase': 3201, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',""")

rep("""    log('== Phase 3200 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))""",
"""    log('== Phase 3201 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))""")

# ---- P16 gates in summary ----
rep("""        'gates': dict(GATE, upgraded=bool(gate_all), n_units_pass=n_units),
        'control': control,""",
"""        'gates': dict(GATE, eline_closed=bool(gate_all),
                      n_contradictions=len(control['contradictions'])),
        'audit': control,""")

# ---- P16b drop G_MED constant ----
rep('G_MED = 0.5\n', '')

# ---- P17 leftovers: rename reg23->reg24 in remaining HTML body code ----
n23 = t.count('reg23')
t = t.replace('reg23', 'reg24')
print('global reg23->reg24: %d' % n23)
for bad in ("'phase': 3200", 'Phase 3199 tail G5-E2', 'v1_22_to_v1_23',
            'atlas_registry_v1_22', 'run_control', 'G_MED', 'ftr23_heldout_predictive'):
    assert bad not in t, ('leftover', bad)
assert 'reg22' not in t and 'f23' not in t and 'FTR-23 E1' not in t

io.open(DST, 'w', encoding='utf-8', newline='').write(t)
back = io.open(DST, encoding='utf-8').read()
assert back == t
print('written %s sha8 %s' % (DST, hashlib.sha256(t.encode()).hexdigest()[:8]))
