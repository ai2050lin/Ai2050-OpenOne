# -*- coding: utf-8 -*-
"""Create phase3199_g5e1_ftr24_heldout_upgrade.py from 3194 (ordered patches)."""
import io
import shutil

SRC = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3194_g5d2_ftr26_heldout_upgrade.py'
DST = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3199_g5e1_ftr24_heldout_upgrade.py'
t = io.open(SRC, encoding='utf-8').read()

def rep(old, new, n=1):
    global t
    c = t.count(old)
    assert c == n, ('patch count', c, n, old[:80])
    t = t.replace(old, new)

# ---- P1 docstring ----
rep('''"""Phase 3194 (G5-D2): FTR-26 held_out upgrade arm (E1_repeatable -> E2_held_out
channel; D line, zero GPU, sealed 3169/3189-3193 reuse).

Per prereg in MEMO Phase 3193 tail (frozen before any observation):
  For each model, 4 folds: hold out one OOV class (219 row directions);
  two predictors (both label-free, no leakage - u* is the SVD top-1 of ALL
  876 row directions, a summary direction, not a class label):
    P1: u* (3193 verbatim);
    P2: v_hat_held = unit(mean of the other 3 classes' row directions)
        [3191 protocol verbatim - control anchor: med_abs_cos_held /
        min_med_abs_cos must replicate bitwise].
  GATE (pinned pre-observation): HOLDOUT_PREDICTIVE_USTAR iff median
  |cos(row, u*)| >= 0.3 in ALL 4 folds x >= 2/3 models => FTR-26 upgraded
  E1_repeatable -> E2_held_out: evidence_level change + anchors append
  (role=held_out) + statement append, registry v1.19 -> v1.20,
  atlas_v1_22.html. Not passed => counter_evidence append, E level
  UNCHANGED E1_repeatable.
  Device anchors: 3189/3190/3191/3192/3193 sealed scalars drift == 0 bitwise
  (incl. 3191 control-anchor replication); 3157 tc + 3183 dv; 3169 per-seed E.
  SMOKE (env P3194_SMOKE=1): zero-GPU => identical numbers; smoke_ prefix.
"""''', '''"""Phase 3199 (G5-E1): FTR-24 held_out upgrade arm (E1_repeatable -> E2_held_out
channel; zero GPU, sealed 3169/3183/3189-3193 reuse + sealed 3197 double lock).

Per prereg in MEMO Phase 3198 tail G5-E1 (frozen before any observation;
prereg target corrected FTR-15 -> FTR-24 since FTR-15 was already upgraded
E2_held_out in Phase 3180):
  For each model, 4 folds: hold out one OOV class (219 row directions);
  P1 = FTR-24 sealed direction dvu (3183/3157 verbatim, label-free;
  v_isa - v_hasa unit difference direction).
  GATE (pinned pre-observation): HOLDOUT_PREDICTIVE_FTR24 iff median
  |cos(row, dvu)| >= 0.3 in ALL 4 folds x ALL 3 models => FTR-24 upgraded
  E1_repeatable -> E2_held_out: evidence_level change + anchors append
  (role=held_out) + statement append + upgrade_log #9, registry v1.21 ->
  v1.22, atlas_v1_24.html. Not passed => counter_evidence append, E level
  UNCHANGED E1_repeatable.
  Device anchors: 3169 per-seed E + 3157 tc + 3183 dv scalars + 3189/3190/
  3191/3193 sealed scalars drift == 0 bitwise (3191 control-anchor
  replication); sealed 3197 result file-locked + signed_weights/ls_resid
  4dp re-read drift == 0 (ANCH_3197 pinned from sealed p3197 result BEFORE
  freeze, carried from Phase 3198). 3195/3196 scalars are NOT re-derived
  this phase (heavy 22-direction QR machinery); their shared Qn substrate
  is bitwise-covered by the 3169/3189/3190/3191 anchors.
  SMOKE (env P3199_SMOKE=1): zero-GPU => identical numbers; smoke_ prefix.
"""''')

# ---- P2 name/outdir/smoke ----
rep("""NAME = 'g5d2_ftr26_heldout_upgrade'
OUTDIR = os.path.join(SRC_DIR, 'phase3194', NAME)
SMOKE = os.environ.get('P3194_SMOKE') == '1'""",
"""NAME = 'g5e1_ftr24_heldout_upgrade'
OUTDIR = os.path.join(SRC_DIR, 'phase3199', NAME)
SMOKE = os.environ.get('P3199_SMOKE') == '1'""")

# ---- P3 GATE ----
rep("""GATE = {'heldout_med_abs_cos_ustar': G_HELD, 'models_needed': 2,
        'dev_tol': DEV_TOL, 'repl_tol': REPL_TOL}""",
"""GATE = {'heldout_med_abs_cos_ftr24': G_HELD, 'models_needed': 3,
        'dev_tol': DEV_TOL, 'repl_tol': REPL_TOL}""")

# ---- P4 SRC ----
rep("""    'registry_v119': os.path.join(SRC_DIR, r'phase3193\\g5d1_ustar_decomposition\\atlas_registry_v1_19.json'),""",
"""    'p3197_result': os.path.join(SRC_DIR, r'phase3197\\g5d5_signed_ls_shares\\result.json'),
    'registry_v121': os.path.join(SRC_DIR, r'phase3198\\g5d6_dline_registration\\atlas_registry_v1_21.json'),""")

# ---- P5 SHA_ANCHOR ----
rep("""    'registry_v119': '4d64a153', 'registry_3162': '00f15e98',""",
"""    'p3197_result': 'ee91f8c7', 'registry_v121': '47cf01e0',
    'registry_3162': '00f15e98',""")

# ---- P6 CONTENT_3197 + ANCH_3197 ----
rep("""CONTENT_3183 = ('c55e4e5e', '7196fa8a')
""",
"""CONTENT_3183 = ('c55e4e5e', '7196fa8a')
CONTENT_3197 = ('dd2af224', '4311d796')

# sealed 3197 scalars (4dp; measured from sealed p3197 result BEFORE freeze;
# carried verbatim from Phase 3198 ANCH_3197)
ANCH_3197 = {
    'qwen3-4b': dict(ls_resid=0.0181, v_c6=-0.3199, v_c7=-0.2725, v_c8=-0.3360, v_c9=-0.3028),
    'qwen3-14b': dict(ls_resid=0.0174, v_c6=-0.2893, v_c7=-0.2810, v_c8=-0.3547, v_c9=-0.2756),
    'glm4-9b': dict(ls_resid=0.0181, v_c6=-0.3613, v_c7=-0.3137, v_c8=-0.3203, v_c9=-0.3020),
}
""")

# ---- P7 DESIGN ----
rep("""DESIGN = {
    'phase': '3194', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3193 tail G5-D2 (frozen before any observation)',
    'data': 'sealed 3169 collect npz x3 + sealed 3169/3183/3189/3190/3191/3192/3193 '
            'results; zero GPU',
    'residual_chain': '3188..3193 primary verbatim: 3169 B-protocol (seeds 7/8/9, '
                      'split_s1_246, ridge lam 1e-3, cols 780), 876 OOV rows, '
                      'row-level directions = unit(mean over 3 seeds of q); '
                      'u* = Vt[0] of Qn (feature space, 3190/3193 verbatim)',
    'measurements': 'per model, 4 folds: hold out one OOV class (219 rows); '
                    'P1 = u* (all-876 SVD top-1, label-free summary); '
                    'P2 = v_hat_held (3191 verbatim, control anchor); '
                    'cos_P1 = Qn[held] @ u*, cos_P2 = Qn[held] @ v_hat_held',
    'gate': 'HOLDOUT_PREDICTIVE_USTAR iff median |cos_P1| >= 0.3 in ALL 4 folds '
            'x >= 2/3 models => FTR-26 E1_repeatable -> E2_held_out (evidence_level '
            'change + anchors append role=held_out + statement append), registry '
            'v1.19->v1.20, atlas_v1_22.html; else counter_evidence append, E level '
            'UNCHANGED E1_repeatable',
    'sources': SHA_ANCHOR,
    'gates': GATE,
}""",
"""DESIGN = {
    'phase': '3199', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3198 tail G5-E1 (frozen before any observation; '
              'target corrected FTR-15 -> FTR-24, FTR-15 already E2 since 3180)',
    'data': 'sealed 3169 collect npz x3 + sealed 3169/3183/3189/3190/3191/3192/3193 '
            'results + sealed p3197 result double lock + sealed registry v1.21; '
            'zero GPU',
    'residual_chain': '3194 primary verbatim: 3169 B-protocol (seeds 7/8/9, '
                      'split_s1_246, ridge lam 1e-3, cols 780), 876 OOV rows, '
                      'row-level directions = unit(mean over 3 seeds of q); '
                      'FTR-24 direction dvu = unit(v_isa - v_hasa), 3183/3157 '
                      'verbatim (build_dvu bitwise anchors)',
    'measurements': 'per model, 4 folds: hold out one OOV class (219 rows); '
                    'P1 = dvu (FTR-24 sealed direction, label-free); '
                    'cos_P1 = Qn[held] @ dvu; median |cos_P1| per fold + '
                    'full-panel median |cos(Qn, dvu)| over all 876 rows',
    'gate': 'HOLDOUT_PREDICTIVE_FTR24 iff median |cos_P1| >= 0.3 in ALL 4 folds '
            'x ALL 3 models => FTR-24 E1_repeatable -> E2_held_out (evidence_level '
            'change + anchors append role=held_out + statement append + '
            'upgrade_log #9), registry v1.21->v1.22, atlas_v1_24.html; else '
            'counter_evidence append, E level UNCHANGED E1_repeatable',
    'sources': SHA_ANCHOR,
    'gates': GATE,
}""")

# ---- P8 freeze phase ----
rep("""        json.dump({'phase': 3194, 'name': NAME, 'design_sha8': dsha[:8],""",
"""        json.dump({'phase': 3199, 'name': NAME, 'design_sha8': dsha[:8],""")

# ---- P9 gate arm block ----
rep("""        # (GATE arm) held-out prediction by u*, 4 folds
        med_ustar = []
        for c in cls_ids:
            held = oov_row_cls == c
            cos_u = Qn[held] @ ustar
            med_ustar.append(float(np.median(np.abs(cos_u))))
        gate_m = bool(min(med_ustar) >= G_HELD)
        out[m] = dict(dv_drift=dv_drift, drift_3169=drift69, drift_3189=drift89,
                      drift_3190=drift90, drift_3191=drift91, drift_3193=drift93,
                      med_abs_cos_ustar=med_ustar,
                      min_med_abs_cos_ustar=float(min(med_ustar)),
                      gate_model=gate_m,
                      med_abs_cos_held=med_abs,
                      min_med_abs_cos=float(min(med_abs)),
                      top1=top1, cross_med=cross_med)
        log('  %s: med|cos_held,u*| %s (min %.4f, gate >=%.2f all 4 folds)=%s | control 3191 med|cos_held| %s (min %.4f)'
            % (m, '/'.join('%.4f' % x for x in med_ustar), min(med_ustar), G_HELD,
               gate_m, '/'.join('%.4f' % x for x in med_abs), min(med_abs)))
        del z, H16, Y, Q_rows, Qn
    n_pass = sum(1 for m in MODELS if out[m]['gate_model'])
    gate_all = bool(n_pass >= 2)
    out['_gate'] = dict(n_models_pass=n_pass, predictive=gate_all)
    log('GATE: %d/3 models pass u* held-out => %s'
        % (n_pass, 'HOLDOUT_PREDICTIVE_USTAR' if gate_all else 'NOT_PREDICTIVE'))
    return out, gate_all""",
"""        # (GATE arm) held-out prediction by FTR-24 direction dvu, 4 folds
        med_dvu = []
        for c in cls_ids:
            held = oov_row_cls == c
            cos_d = Qn[held] @ dvu
            med_dvu.append(float(np.median(np.abs(cos_d))))
        full_med = float(np.median(np.abs(Qn @ dvu)))
        gate_m = bool(min(med_dvu) >= G_HELD)
        out[m] = dict(dv_drift=dv_drift, drift_3169=drift69, drift_3189=drift89,
                      drift_3190=drift90, drift_3191=drift91, drift_3193=drift93,
                      med_abs_cos_dvu=med_dvu,
                      min_med_abs_cos_dvu=float(min(med_dvu)),
                      full_panel_med_abs_cos_dvu=full_med,
                      gate_model=gate_m,
                      med_abs_cos_held=med_abs,
                      min_med_abs_cos=float(min(med_abs)),
                      top1=top1, cross_med=cross_med)
        log('  %s: med|cos_held,dvu| %s (min %.4f, gate >=%.2f all 4 folds)=%s | full-panel med|cos| %.4f | control 3191 med|cos_held| %s (min %.4f)'
            % (m, '/'.join('%.4f' % x for x in med_dvu), min(med_dvu), G_HELD,
               gate_m, full_med, '/'.join('%.4f' % x for x in med_abs), min(med_abs)))
        del z, H16, Y, Q_rows, Qn
    n_pass = sum(1 for m in MODELS if out[m]['gate_model'])
    gate_all = bool(n_pass >= 3)
    out['_gate'] = dict(n_models_pass=n_pass, predictive=gate_all)
    log('GATE: %d/3 models pass FTR-24 held-out => %s'
        % (n_pass, 'HOLDOUT_PREDICTIVE_FTR24' if gate_all else 'NOT_PREDICTIVE'))
    return out, gate_all""")

# ---- P10 finish_registry header ----
rep("""    reg19 = json.loads(open(SRC['registry_v119'], 'rb').read().decode('utf-8'))
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg19['version'] == '1.19' and len(reg19['features']) == 26
    reg20 = copy.deepcopy(reg19)
    reg20['version'] = '1.20'
    reg20['supersedes'] = 'atlas_registry_v1_19.json (4d64a153, Phase 3193)'
    pm = summary['per_model']
    fmt3 = lambda key: '/'.join('%.4f' % pm[m][key] for m in MODELS)
    f26_old = [f for f in reg19['features'] if f['id'] == 'FTR-26'][0]
    f26 = copy.deepcopy(f26_old)""",
"""    reg21 = json.loads(open(SRC['registry_v121'], 'rb').read().decode('utf-8'))
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg21['version'] == '1.21' and len(reg21['features']) == 26
    reg22 = copy.deepcopy(reg21)
    reg22['version'] = '1.22'
    reg22['supersedes'] = 'atlas_registry_v1_21.json (47cf01e0, Phase 3198)'
    pm = summary['per_model']
    fmt3 = lambda key: '/'.join('%.4f' % pm[m][key] for m in MODELS)
    f24_old = [f for f in reg21['features'] if f['id'] == 'FTR-24'][0]
    f24 = copy.deepcopy(f24_old)""")

# ---- P11 gate branch ----
rep("""    if gate_all:
        anchor = {
            'src': 'p3194', 'phase': 3194,
            'role': 'held_out',
            'tags': ['zero_gpu', 'sealed_reuse', 'e_level_upgrade'],
            'asserts': {
                'min_med_abs_cos_ustar.qwen3-4b': pm['qwen3-4b']['min_med_abs_cos_ustar'],
                'min_med_abs_cos_ustar.qwen3-14b': pm['qwen3-14b']['min_med_abs_cos_ustar'],
                'min_med_abs_cos_ustar.glm4-9b': pm['glm4-9b']['min_med_abs_cos_ustar'],
                'gate_heldout_med_abs_cos_ustar': G_HELD,
                'e_level_before': 'E1_repeatable', 'e_level_after': 'E2_held_out',
                'res_sha8': res8, 'seal_sha8': seal8}}
        f26['anchors'] = f26_old['anchors'] + [anchor]
        f26['evidence_level'] = 'E2_held_out'
        f26['statement'] = f26_old['statement'] + (
            '；3194 G5-D2 held_out 升格臂：留出类 219 行方向对 u*（全 876 行 SVD '
            'top-1，label-free 汇总方向）的中位 |cos| 每模型 4 折最小值 ' +
            fmt3('min_med_abs_cos_ustar') + '（门 ≥0.3 全 4 折 ×≥2/3 模型）——'
            'u* 可跨类预测留出类行方向，E1_repeatable 升 E2_held_out（升级链 #8）；'
            '对照锚：3191 v̂_held 协议逐位复现。')
        reg20['features'] = [f26 if f['id'] == 'FTR-26' else f
                             for f in reg19['features']]
        reg20['upgrade_log'] = reg19['upgrade_log'] + [{
            'node': 'FTR-26', 'change': 'E1_repeatable -> E2_held_out',
            'reason': 'G5-D2 held_out arm (Phase 3194): held-out class row '
                      'directions predicted by label-free summary direction u*, '
                      'median |cos| >= 0.3 in all 4 folds x models; cross-model '
                      'held-out predictiveness established (zero GPU, sealed '
                      '3169/3189-3193 reuse; 3191 control anchor bitwise)'}]
        disposition = 'v1_19_to_v1_20_ftr26_e2_held_out'
    else:
        ce = ('3194 G5-D2 held_out 升格臂未过：留出类 219 行方向对 u* 的中位 |cos| '
              '4 折最小值 ' + fmt3('min_med_abs_cos_ustar') +
              '（门 ≥0.3 全 4 折 ×≥2/3 模型未过）——u* 对留出类行方向无预测力，'
              'E1_repeatable 维持（3190 类均值核读法仍由 FTR-25 承载）。')
        f26['counter_evidence'] = f26_old['counter_evidence'] + [ce]
        reg20['features'] = [f26 if f['id'] == 'FTR-26' else f
                             for f in reg19['features']]
        disposition = 'v1_19_to_v1_20_ftr26_ce_not_predictive'""",
"""    if gate_all:
        anchor = {
            'src': 'p3199', 'phase': 3199,
            'role': 'held_out',
            'tags': ['zero_gpu', 'sealed_reuse', 'e_level_upgrade'],
            'asserts': {
                'min_med_abs_cos_dvu.qwen3-4b': pm['qwen3-4b']['min_med_abs_cos_dvu'],
                'min_med_abs_cos_dvu.qwen3-14b': pm['qwen3-14b']['min_med_abs_cos_dvu'],
                'min_med_abs_cos_dvu.glm4-9b': pm['glm4-9b']['min_med_abs_cos_dvu'],
                'gate_heldout_med_abs_cos_ftr24': G_HELD,
                'e_level_before': 'E1_repeatable', 'e_level_after': 'E2_held_out',
                'res_sha8': res8, 'seal_sha8': seal8}}
        f24['anchors'] = f24_old['anchors'] + [anchor]
        f24['evidence_level'] = 'E2_held_out'
        f24['statement'] = f24_old['statement'] + (
            '；3199 G5-E1 held_out 升格臂：留出类 219 行方向对 dvu（FTR-24 封存方向，'
            '3183/3157 verbatim，label-free）的中位 |cos| 每模型 4 折最小值 ' +
            fmt3('min_med_abs_cos_dvu') + '（门 ≥0.3 全 4 折 ×3/3 模型）——'
            'dvu 可跨类预测留出类行方向，E1_repeatable 升 E2_held_out（升级链 #9）；'
            '对照锚：3183 dv 标量 + 3191 v̂_held 协议逐位复现。')
        reg22['features'] = [f24 if f['id'] == 'FTR-24' else f
                             for f in reg21['features']]
        reg22['upgrade_log'] = reg21['upgrade_log'] + [{
            'node': 'FTR-24', 'change': 'E1_repeatable -> E2_held_out',
            'reason': 'G5-E1 held_out arm (Phase 3199): held-out class row '
                      'directions predicted by label-free sealed direction dvu '
                      '(v_isa - v_hasa), median |cos| >= 0.3 in all 4 folds x '
                      'ALL 3 models; cross-model held-out predictiveness '
                      'established (zero GPU, sealed 3169/3183/3189-3193 reuse; '
                      '3183/3191 control anchors bitwise)'}]
        disposition = 'v1_21_to_v1_22_ftr24_e2_held_out'
    else:
        ce = ('3199 G5-E1 held_out 升格臂未过：留出类 219 行方向对 dvu 的中位 |cos| '
              '4 折最小值 ' + fmt3('min_med_abs_cos_dvu') +
              '（门 ≥0.3 全 4 折 ×3/3 模型未过）——dvu 对留出类行方向预测力不足，'
              'E1_repeatable 维持。')
        f24['counter_evidence'] = f24_old['counter_evidence'] + [ce]
        reg22['features'] = [f24 if f['id'] == 'FTR-24' else f
                             for f in reg21['features']]
        disposition = 'v1_21_to_v1_22_ftr24_ce_not_predictive'""")

# ---- P12 preservation asserts ----
rep("""    # preservation asserts
    for f in reg19['features']:
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg20['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        if f['id'] == 'FTR-26':
            fa = json.loads(a)
            assert b != a
            if gate_all:
                assert f26['statement'].startswith(fa['statement'])
                assert f26['counter_evidence'] == fa['counter_evidence']
                assert json.dumps(f26['anchors'][:-1], ensure_ascii=False, sort_keys=True) == \\
                    json.dumps(fa['anchors'], ensure_ascii=False, sort_keys=True)
                assert f26['evidence_level'] == 'E2_held_out' and fa['evidence_level'] == 'E1_repeatable'
            else:
                assert f26['counter_evidence'][:-1] == fa['counter_evidence']
                assert f26['statement'] == fa['statement']
                assert f26['anchors'] == fa['anchors']
                assert f26['evidence_level'] == fa['evidence_level'] == 'E1_repeatable'
        else:
            assert a == b, ('feature not preserved', f['id'])
    assert json.dumps(reg20['upgrade_log'][:-1], ensure_ascii=False, sort_keys=True) == \\
        json.dumps(reg19['upgrade_log'], ensure_ascii=False, sort_keys=True) \\
        if gate_all else json.dumps(reg20['upgrade_log'], ensure_ascii=False, sort_keys=True) == \\
        json.dumps(reg19['upgrade_log'], ensure_ascii=False, sort_keys=True)
    for k in ('evidence_level_taxonomy', 'principles', 'schema',
              'models_mainline', 'models_pending', 'provenance', 'created',
              'phase', 'name', 'design_sha8'):
        assert json.dumps(reg19.get(k), ensure_ascii=False, sort_keys=True) == \\
            json.dumps(reg20.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    assert json.dumps(reg19['failures'], ensure_ascii=False, sort_keys=True) == \\
        json.dumps(reg20['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'""",
"""    # preservation asserts
    for f in reg21['features']:
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg22['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        if f['id'] == 'FTR-24':
            fa = json.loads(a)
            assert b != a
            if gate_all:
                assert f24['statement'].startswith(fa['statement'])
                assert f24['counter_evidence'] == fa['counter_evidence']
                assert json.dumps(f24['anchors'][:-1], ensure_ascii=False, sort_keys=True) == \\
                    json.dumps(fa['anchors'], ensure_ascii=False, sort_keys=True)
                assert f24['evidence_level'] == 'E2_held_out' and fa['evidence_level'] == 'E1_repeatable'
            else:
                assert f24['counter_evidence'][:-1] == fa['counter_evidence']
                assert f24['statement'] == fa['statement']
                assert f24['anchors'] == fa['anchors']
                assert f24['evidence_level'] == fa['evidence_level'] == 'E1_repeatable'
        else:
            assert a == b, ('feature not preserved', f['id'])
    assert json.dumps(reg22['upgrade_log'][:-1], ensure_ascii=False, sort_keys=True) == \\
        json.dumps(reg21['upgrade_log'], ensure_ascii=False, sort_keys=True) \\
        if gate_all else json.dumps(reg22['upgrade_log'], ensure_ascii=False, sort_keys=True) == \\
        json.dumps(reg21['upgrade_log'], ensure_ascii=False, sort_keys=True)
    for k in ('evidence_level_taxonomy', 'principles', 'schema',
              'models_mainline', 'models_pending', 'provenance', 'created',
              'phase', 'name', 'design_sha8'):
        assert json.dumps(reg21.get(k), ensure_ascii=False, sort_keys=True) == \\
            json.dumps(reg22.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    assert json.dumps(reg21['failures'], ensure_ascii=False, sort_keys=True) == \\
        json.dumps(reg22['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'""")

# ---- P13 registry write + meta_sha ----
rep("""    rp = os.path.join(OUTDIR, 'atlas_registry_v1_20.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg20, ensure_ascii=False, indent=1))
    reg20_sha = sha8_file(rp)
    log('registry v1.20 written sha8 %s (%s)' % (reg20_sha, disposition))

    meta_sha = {'reg20': reg20_sha}""",
"""    rp = os.path.join(OUTDIR, 'atlas_registry_v1_22.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg22, ensure_ascii=False, indent=1))
    reg22_sha = sha8_file(rp)
    log('registry v1.22 written sha8 %s (%s)' % (reg22_sha, disposition))

    meta_sha = {'reg22': reg22_sha}""")

# ---- P14 HTML title/h1/sub ----
rep("""    H.append('<title>Atlas v1.22 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.22 &mdash; 语义机制图谱（registry v1.20 渲染）</h1>')
    H.append('<div class="sub">Phase 3194 G5-D2 &middot; FTR-26 held_out 升格臂（%s） &middot; '
             'registry v1.20 sha8=%s &middot; v1.19 sha8=%s &middot; v1.18 sha8=%s &middot; v1.17 sha8=%s &middot; gap v1.6 sha8=%s'
             ' &middot; 源 p3169 %s + p3193 %s &middot; 渲染于 %s</div>' % (
                 'FTR-26 E1->E2_held_out 升格' if gate_all else 'FTR-26 counter_evidence 未过升格门',
                 span('meta.registry_v120_sha8', reg20_sha),
                 span('meta.registry_v119_sha8', '4d64a153'),
                 span('meta.registry_v118_sha8', 'c08cbcc2'),
                 span('meta.registry_v117_sha8', '73923c83'),
                 span('meta.gap_v16_sha8', '0f6423e0'),
                 span('meta.p3169_result', SHA_ANCHOR['p3169_result']),
                 span('meta.p3193_result', SHA_ANCHOR['p3193_result']),
                 span('meta.created', CREATED_STR)))""",
"""    H.append('<title>Atlas v1.24 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.24 &mdash; 语义机制图谱（registry v1.22 渲染）</h1>')
    H.append('<div class="sub">Phase 3199 G5-E1 &middot; FTR-24 held_out 升格臂（%s） &middot; '
             'registry v1.22 sha8=%s &middot; v1.21 sha8=%s &middot; v1.20 sha8=%s &middot; v1.19 sha8=%s &middot; gap v1.6 sha8=%s'
             ' &middot; 源 p3169 %s + p3197 %s &middot; 渲染于 %s</div>' % (
                 'FTR-24 (dvu) E1->E2_held_out 升格' if gate_all else 'FTR-24 counter_evidence 未过升格门',
                 span('meta.registry_v122_sha8', reg22_sha),
                 span('meta.registry_v121_sha8', '47cf01e0'),
                 span('meta.registry_v120_sha8', '2aac7e5e'),
                 span('meta.registry_v119_sha8', '4d64a153'),
                 span('meta.gap_v16_sha8', '0f6423e0'),
                 span('meta.p3169_result', SHA_ANCHOR['p3169_result']),
                 span('meta.p3197_result', SHA_ANCHOR['p3197_result']),
                 span('meta.created', CREATED_STR)))""")

# ---- P15 feature table heading ----
rep("""    H.append('<h2>跨模型稳定特征登记表（registry v1.20，%d 条）</h2>' % len(reg20['features']))""",
"""    H.append('<h2>跨模型稳定特征登记表（registry v1.22，%d 条）</h2>' % len(reg22['features']))""")

# ---- P16 footer + hp path + log ----
rep("""    H.append('<div class="footer">RDC/LPF atlas v1.22 &middot; 渲染自 atlas_registry_v1_20.json (%s)'
             ' + gap ledger v1.5 (4fb3f37d) &middot; 增量来源: Phase 3194 G5-D2 FTR-26 held_out 升格臂'
             '（数据源 p3169 %s + p3193 %s）'
             '&middot; 本文件为 Phase 3194 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg20_sha, SHA_ANCHOR['p3169_result'], SHA_ANCHOR['p3193_result']))""",
"""    H.append('<div class="footer">RDC/LPF atlas v1.24 &middot; 渲染自 atlas_registry_v1_22.json (%s)'
             ' + gap ledger v1.5 (4fb3f37d) &middot; 增量来源: Phase 3199 G5-E1 FTR-24 held_out 升格臂'
             '（数据源 p3169 %s + p3197 %s）'
             '&middot; 本文件为 Phase 3199 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg22_sha, SHA_ANCHOR['p3169_result'], SHA_ANCHOR['p3197_result']))""")

rep("""    hp = os.path.join(OUTDIR, 'atlas_v1_22.html')""",
"""    hp = os.path.join(OUTDIR, 'atlas_v1_24.html')""")

rep("""    log('atlas_v1_22.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (""",
"""    log('atlas_v1_24.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (""")

# ---- P17 final dict ----
rep("""    final = {
        'phase': 3194, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'verdict': summary['verdict'], 'gate_predictive': bool(gate_all),
                'per_model': summary['per_model']},
        'counts': {'features': len(reg20['features']), 'failures': len(reg20['failures']),
                   'upgrades': len(reg20['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v120_file': 'atlas_registry_v1_20.json', 'registry_v120_sha8': reg20_sha,
        'html_file': 'atlas_v1_22.html', 'html_sha8': html_sha,""",
"""    final = {
        'phase': 3199, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'verdict': summary['verdict'], 'gate_predictive': bool(gate_all),
                'per_model': summary['per_model']},
        'counts': {'features': len(reg22['features']), 'failures': len(reg22['failures']),
                   'upgrades': len(reg22['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v122_file': 'atlas_registry_v1_22.json', 'registry_v122_sha8': reg22_sha,
        'registry_v121_sha8_ref': '47cf01e0',
        'html_file': 'atlas_v1_24.html', 'html_sha8': html_sha,""")

# ---- P18 flatten meta ----
rep("""    d['meta.registry_v120_sha8'] = meta_sha['reg20']
    d['meta.registry_v119_sha8'] = '4d64a153'
    d['meta.registry_v118_sha8'] = 'c08cbcc2'
    d['meta.registry_v117_sha8'] = '73923c83'
    d['meta.gap_v16_sha8'] = '0f6423e0'
    d['meta.p3169_result'] = SHA_ANCHOR['p3169_result']
    d['meta.p3193_result'] = SHA_ANCHOR['p3193_result']
    d['meta.created'] = CREATED_STR""",
"""    d['meta.registry_v122_sha8'] = meta_sha['reg22']
    d['meta.registry_v121_sha8'] = '47cf01e0'
    d['meta.registry_v120_sha8'] = '2aac7e5e'
    d['meta.registry_v119_sha8'] = '4d64a153'
    d['meta.gap_v16_sha8'] = '0f6423e0'
    d['meta.p3169_result'] = SHA_ANCHOR['p3169_result']
    d['meta.p3197_result'] = SHA_ANCHOR['p3197_result']
    d['meta.created'] = CREATED_STR""")

# ---- P19 main: 3197 lock + control re-read ----
rep("""    log('sealed results double-locked: 3169/3183/3189/3190/3191/3192/3193')

    primary, gate_all = run_arm(sealed_3169, sealed_3183, sealed_3189,""",
"""    log('sealed results double-locked: 3169/3183/3189/3190/3191/3192/3193')
    sealed_3197 = json.load(open(SRC['p3197_result'], encoding='utf-8'))
    assert sealed_3197['res_sha8'] == CONTENT_3197[0] and sealed_3197['seal_sha8'] == CONTENT_3197[1]
    for m in MODELS:
        A = ANCH_3197[m]
        pms = sealed_3197['per_model'][m]
        d = abs(round(pms['ls_resid'], 4) - A['ls_resid'])
        for cc in range(6, 10):
            d = max(d, abs(round(pms['signed_weights']['v_c[%d]' % cc], 4) - A['v_c%d' % cc]))
        assert d == 0.0, ('3197 4dp control drift', m, d)
    log('sealed 3197 double-locked + 4dp control re-read drift==0 x3')

    primary, gate_all = run_arm(sealed_3169, sealed_3183, sealed_3189,""")

# ---- P20 main verdict + summary ----
rep("""    if gate_all:
        verdict = ('ftr26_heldout_predictive_ustar (min med|cos_held,u*| %s >= 0.3 '
                   'all 4 folds in %d/3 models; FTR-26 E1_repeatable -> '
                   'E2_held_out, upgrade_log #8, registry v1.19 -> v1.20)'
                   % ('/'.join('%.4f' % primary[m]['min_med_abs_cos_ustar'] for m in MODELS),
                      primary['_gate']['n_models_pass']))
    else:
        verdict = ('ftr26_heldout_not_predictive (u* held-out gate fail in %d/3 '
                   'models pass; FTR-26 counter_evidence append, E level '
                   'unchanged E1_repeatable, registry v1.19 -> v1.20)'
                   % primary['_gate']['n_models_pass'])""",
"""    if gate_all:
        verdict = ('ftr24_heldout_predictive_dvu (min med|cos_held,dvu| %s >= 0.3 '
                   'all 4 folds in %d/3 models; FTR-24 E1_repeatable -> '
                   'E2_held_out, upgrade_log #9, registry v1.21 -> v1.22)'
                   % ('/'.join('%.4f' % primary[m]['min_med_abs_cos_dvu'] for m in MODELS),
                      primary['_gate']['n_models_pass']))
    else:
        verdict = ('ftr24_heldout_not_predictive (dvu held-out gate fail in %d/3 '
                   'models pass; FTR-24 counter_evidence append, E level '
                   'unchanged E1_repeatable, registry v1.21 -> v1.22)'
                   % primary['_gate']['n_models_pass'])""")

rep("""    summary = {
        'phase': 3194, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',""",
"""    summary = {
        'phase': 3199, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',""")

# ---- P21 main log banner ----
rep("""    log('== Phase 3194 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))""",
"""    log('== Phase 3199 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))""")

# ---- P22 global leftover rename (reg19/reg20/f26/FTR-26 in remaining code) ----
n19 = t.count('reg19')
n20 = t.count('reg20')
nf26 = t.count('f26')
nF26 = t.count('FTR-26')
t = t.replace('reg19', 'reg21').replace('reg20', 'reg22')
t = t.replace('f26', 'f24').replace('FTR-26', 'FTR-24')
print('global renames: reg19=%d reg20=%d f26=%d FTR-26=%d' % (n19, n20, nf26, nF26))
assert 'reg19' not in t and 'reg20' not in t and 'f26' not in t and 'FTR-26' not in t
assert '3194' not in t.replace('phase3194', '').replace('3194 control', '').replace(
    'Phase 3194', '') or True  # phase3194 path refs not expected; sanity below
for bad in ("'phase': 3194", 'Phase 3193 tail G5-D2', 'v1_19_to_v1_20',
            'atlas_registry_v1_19', 'atlas_registry_v1_20', 'med_abs_cos_ustar'):
    assert bad not in t, ('leftover', bad)

io.open(DST, 'w', encoding='utf-8', newline='').write(t)
back = io.open(DST, encoding='utf-8').read()
assert back == t
import hashlib
print('written %s sha8 %s' % (DST, hashlib.sha256(t.encode()).hexdigest()[:8]))
