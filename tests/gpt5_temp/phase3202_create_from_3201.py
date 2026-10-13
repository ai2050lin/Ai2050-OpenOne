# -*- coding: utf-8 -*-
"""Create phase3202_g5f1_ftr04_recheck.py from 3201 (marker-splice patches)."""
import io
import hashlib

SRC = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/phase3201_g5e3_eline_audit.py'
DST = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/phase3202_g5f1_ftr04_recheck.py'
t = io.open(SRC, encoding='utf-8').read()

def splice(start_marker, end_marker, new_text, keep_end=True):
    global t
    i = t.index(start_marker)
    j = t.index(end_marker, i)
    assert t.count(start_marker) == 1
    t = t[:i] + new_text + (t[j:] if keep_end else '')

def rep(old, new):
    global t
    c = t.count(old)
    assert c == 1, ('count', c, old[:80])
    t = t.replace(old, new)

# ---- P1 docstring ----
splice('"""Phase 3201', 'import copy', '''"""Phase 3202 (G5-F1): FTR-04 glm4 weak-specificity recheck arm (registry
v1.24 -> v1.25, zero GPU, sealed 3181 scalars re-read + unit decomposition).

Per prereg in MEMO Phase 3201 tail G5-F1 (frozen before any observation;
default arm (a) confirmed viable by pre-freeze probe: sealed 3181 result
carries unit-level per_model fields; probe registered honestly):
  RECHECK (pinned): R1 - sealed 3181 per_model scalars re-read vs pinned 4dp
  constants (loeo_acc / mean_margin / kappa / pr / proj_ratio, e90==8)
  drift == 0 x3 models; R2 - FTR-04 p3181 anchor asserts (kappa/margin/
  proj_ratio glm4) bitwise-consistent with sealed scalars; R3 - unit-level
  decomposition bars (pinned post-probe, conservative): loeo_gap =
  min(qwen loeo) - glm4 loeo >= 0.10 AND margin_ratio = glm4 margin /
  max(qwen margin) >= 1.5 AND geometry band healthy (|kappa ratio - 1| <
  0.25, |pr ratio - 1| < 0.2, e90 equal 8).
  Registration (R1-R3 clean): FTR-04 scope_limits append ONLY (unit-level
  decomposition verdict: glm4 weak specificity = readout inconsistency -
  highest margin with lowest LOEO accuracy - NOT a geometry deficit;
  supports 3181 noise_level verdict), registry v1.24 -> v1.25,
  atlas_v1_27.html; upgrade_log / failures / all other features UNCHANGED.
  Dirty => contradictions registered, report-only (v1.24 unchanged).
  Control anchors: sealed 3181 content double lock (res 1e95a19d /
  seal 462fd221) + 3178/3183 double locks + 3157 tc + 3183 dv bitwise x3
  (device anchors carried) + sealed registry v1.24 sha 0d74b628.
  SMOKE (env P3202_SMOKE=1): registry/html writes skipped; checks only.
"""

''')

# ---- P2 name/outdir/smoke ----
rep("""NAME = 'g5e3_eline_audit'
OUTDIR = os.path.join(SRC_DIR, 'phase3201', NAME)
SMOKE = os.environ.get('P3201_SMOKE') == '1'""",
"""NAME = 'g5f1_ftr04_recheck'
OUTDIR = os.path.join(SRC_DIR, 'phase3202', NAME)
SMOKE = os.environ.get('P3202_SMOKE') == '1'""")

# ---- P3 GATE ----
rep("""GATE = {'audit_clean': True, 'closing_append': True,
        'dev_tol': DEV_TOL, 'repl_tol': REPL_TOL}""",
"""GATE = {'reread_drift_zero': True, 'anchor_consistent': True,
        'decomposition_bars': True, 'dev_tol': DEV_TOL, 'repl_tol': REPL_TOL}""")

# ---- P4 SRC + SHA_ANCHOR ----
rep("""    'registry_v123': os.path.join(SRC_DIR, r'phase3200\\g5e2_ftr23_heldout_upgrade\\atlas_registry_v1_23.json'),""",
"""    'registry_v124': os.path.join(SRC_DIR, r'phase3201\\g5e3_eline_audit\\atlas_registry_v1_24.json'),
    'p3181_result': os.path.join(SRC_DIR, r'phase3181\\g5b6_ftr04_glm4_attribution\\result.json'),""")
rep("""    'registry_v123': '5f5b9e73', 'registry_3162': '00f15e98',""",
"""    'registry_v124': '0d74b628', 'p3181_result': '5ab51de6',
    'registry_3162': '00f15e98',""")

# ---- P5 constants block (splice ANCH_3178..SCOPE_APPEND -> 3181 pins) ----
splice('# sealed 3178 scalars', 'DESIGN = {', '''# sealed 3181 scalars (4dp probed from sealed p3181 result BEFORE freeze;
# measured-anchor discipline; content locks res/seal are field-level)
CONTENT_3181 = ('1e95a19d', '462fd221')
PIN_3181 = {
    'qwen3-4b': dict(loeo=0.6019, margin=2.6614, kappa=48118411.9811,
                     pr=8.7763, proj=0.0763, e90=8),
    'qwen3-14b': dict(loeo=0.5981, margin=1.4164, kappa=46016538.1628,
                      pr=8.9016, proj=0.0328, e90=8),
    'glm4-9b': dict(loeo=0.4800, margin=5.2326, kappa=44953766.4873,
                    pr=8.9220, proj=0.0294, e90=8),
}
DECOMP_BARS = {'loeo_gap_min': 0.10, 'margin_ratio_min': 1.5}
SCOPE_APPEND_F04 = ('；3202 G5-F1 FTR-04 glm4 弱特异性单元级分解复读（封存 3181 读数复用，'
                    '零 GPU）：glm4-9b 弱特异性 = 读出不一致性而非几何缺陷——几何带健康'
                    '（kappa 44953766.4873 vs 4b 48118411.9811 比差 <25%、pr 8.9220 vs '
                    '8.7763 比差 <20%、e90 同为 8），但 LOEO 留出读出准确率最低 '
                    '0.4800 vs 0.6019/0.5981（gap 0.1181 ≥0.10 钉定 bar）且 mean_margin '
                    '最高 5.2326（为 qwen 最大值 2.6614 的 1.966 倍 ≥1.5 bar）——margin '
                    '高而留出准确率低 ⇒ 类对间隔分布散、读出不稳定；支撑 3181 '
                    'noise_level 定判（弱特异性不可归因于几何或读出机制缺失，E 级维持 '
                    'E1_repeatable）。对照锚：3181 标量 4dp 回读 drift==0 ×3 + FTR-04 '
                    'p3181 锚 asserts 与封存标量逐位一致 + 3181 内容双锁。')

''')

# ---- P6 DESIGN block (splice) ----
splice('DESIGN = {', 'def log(s):', '''DESIGN = {
    'phase': '3202', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3201 tail G5-F1 (frozen before any observation; '
              'default arm (a) FTR-04 glm4 weak-specificity recheck; pre-freeze '
              'probe confirmed sealed 3181 unit-level fields available)',
    'data': 'sealed p3181 result (per_model scalars 4dp re-read + content double '
            'lock) + sealed registry v1.24 (0d74b628) + sealed p3178/p3183 results '
            '+ sealed 3157 npz x3; zero GPU',
    'control_anchor': 'sealed 3181 scalars re-read vs pinned 4dp constants drift '
                      '== 0 x3 (probed BEFORE freeze) + FTR-04 p3181 anchor asserts '
                      'bitwise-consistent + 3181 content double lock (1e95a19d/'
                      '462fd221) + 3178/3183 double locks + 3157 tc + 3183 dv '
                      'bitwise x3 (device anchors carried)',
    'measurements': 'R1 4dp re-read x3 (loeo/margin/kappa/pr/proj/e90); R2 anchor '
                    'bitwise consistency; R3 decomposition loeo_gap (min qwen loeo '
                    '- glm4 loeo) + margin_ratio (glm4 margin / max qwen margin) + '
                    'geometry band (kappa/pr ratios, e90)',
    'gate': 'REGISTERED iff R1 drift==0 x3 AND R2 anchor bitwise AND R3 bars '
            '(loeo_gap >= 0.10, margin_ratio >= 1.5, geometry band) => FTR-04 '
            'scope_limits append ONLY, registry v1.24->v1.25, atlas_v1_27.html; '
            'else contradictions registered, report-only',
    'sources': SHA_ANCHOR,
    'gates': GATE,
}


''', keep_end=True)

# ---- P7 freeze phase ----
rep("        json.dump({'phase': 3201, 'name': NAME, 'design_sha8': dsha[:8],",
    "        json.dump({'phase': 3202, 'name': NAME, 'design_sha8': dsha[:8],")

# ---- P8 run_audit -> run_recheck (splice) ----
splice('def run_audit(reg):', '# ---------------- registry', '''def run_recheck(sealed_3181, reg_v124):
    """Pinned recheck R1-R3 over sealed 3181 scalars + registry v1.24 FTR-04."""
    out = {'contradictions': []}
    pm81 = sealed_3181['per_model']
    for m in MODELS:
        E = pm81[m]
        P = PIN_3181[m]
        drift = max(abs(round(E['loeo_acc'], 4) - P['loeo']),
                    abs(round(E['mean_margin'], 4) - P['margin']),
                    abs(round(E['kappa'], 4) - P['kappa']),
                    abs(round(E['pr'], 4) - P['pr']),
                    abs(round(E['proj_ratio'], 4) - P['proj']),
                    abs(int(E['e90']) - P['e90']))
        assert drift == 0.0, ('3181 reread drift', m, drift)
    log('R1: sealed 3181 scalars == pinned 4dp constants x3 (drift==0)')
    f04 = [f for f in reg_v124['features'] if f['id'] == 'FTR-04'][0]
    a81 = [a for a in f04['anchors'] if a['src'] == 'p3181'][0]
    A = a81['asserts']
    bit_ok = (A['kappa.glm4-9b'] == pm81['glm4-9b']['kappa']
              and A['mean_margin.glm4-9b'] == pm81['glm4-9b']['mean_margin']
              and A['proj_ratio.glm4-9b'] == pm81['glm4-9b']['proj_ratio']
              and A['res_sha8'] == CONTENT_3181[0]
              and A['seal_sha8'] == CONTENT_3181[1])
    if not bit_ok:
        out['contradictions'].append(('R2_anchor_bitwise', 'FTR-04/p3181'))
    log('R2: FTR-04 p3181 anchor asserts bitwise-consistent: %s'
        % ('OK' if bit_ok else 'FAIL'))
    loeo_g = pm81['glm4-9b']['loeo_acc']
    loeo_q = min(pm81['qwen3-4b']['loeo_acc'], pm81['qwen3-14b']['loeo_acc'])
    gap = loeo_q - loeo_g
    mratio = (pm81['glm4-9b']['mean_margin']
              / max(pm81['qwen3-4b']['mean_margin'], pm81['qwen3-14b']['mean_margin']))
    geo_ok = (abs(pm81['glm4-9b']['kappa'] / pm81['qwen3-4b']['kappa'] - 1) < 0.25
              and abs(pm81['glm4-9b']['pr'] / pm81['qwen3-4b']['pr'] - 1) < 0.2
              and int(pm81['glm4-9b']['e90']) == int(pm81['qwen3-4b']['e90']) == 8)
    out['decomposition'] = {'loeo_gap': gap, 'margin_ratio': mratio,
                            'geometry_band_ok': bool(geo_ok)}
    if not gap >= DECOMP_BARS['loeo_gap_min']:
        out['contradictions'].append(('R3_loeo_gap', gap))
    if not mratio >= DECOMP_BARS['margin_ratio_min']:
        out['contradictions'].append(('R3_margin_ratio', mratio))
    if not geo_ok:
        out['contradictions'].append(('R3_geometry_band', False))
    log('R3: loeo_gap %.4f (bar >=0.10) | margin_ratio %.4f (bar >=1.5) | '
        'geometry band: %s' % (gap, mratio, 'OK' if geo_ok else 'FAIL'))
    out['recheck_clean'] = not out['contradictions']
    log('RECHECK: %s (%d contradictions)'
        % ('CLEAN' if out['recheck_clean'] else 'DIRTY', len(out['contradictions'])))
    return out


''')

# ---- P9 finish_registry (splice whole function) ----
splice('def finish_registry(control, res8, seal8, design_sha):', 'def flatten_v122(reg, nodes, gl, meta_sha):', '''def finish_registry(control, res8, seal8, design_sha):
    global CREATED_STR
    CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    reg24 = json.loads(open(SRC['registry_v124'], 'rb').read().decode('utf-8'))
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg24['version'] == '1.24' and len(reg24['features']) == 26
    reg25 = copy.deepcopy(reg24)
    reg25['version'] = '1.25'
    reg25['supersedes'] = 'atlas_registry_v1_24.json (0d74b628, Phase 3201)'
    audit = control
    f04_old = [f for f in reg24['features'] if f['id'] == 'FTR-04'][0]
    f04 = copy.deepcopy(f04_old)
    gate_all = bool(audit['recheck_clean'])
    if gate_all:
        f04['scope_limits'] = f04_old['scope_limits'] + SCOPE_APPEND_F04
        reg25['features'] = [f04 if f['id'] == 'FTR-04' else f
                             for f in reg24['features']]
        disposition = 'v1_24_to_v1_25_ftr04_recheck_registered'
    else:
        reg25 = copy.deepcopy(reg24)
        disposition = 'v1_24_unchanged_recheck_dirty_report_only'
    # preservation asserts
    for f in reg24['features']:
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg25['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        if f['id'] == 'FTR-04':
            fa = json.loads(a)
            if gate_all:
                assert b != a
                assert f04['scope_limits'].startswith(fa['scope_limits'])
                assert f04['scope_limits'][:-len(SCOPE_APPEND_F04)] == fa['scope_limits']
                assert f04['evidence_level'] == fa['evidence_level'] == 'E1_repeatable'
                assert f04['statement'] == fa['statement']
                assert f04['anchors'] == fa['anchors']
                assert f04['counter_evidence'] == fa['counter_evidence']
            else:
                assert a == b
        else:
            assert a == b, ('feature not preserved', f['id'])
    assert json.dumps(reg25['upgrade_log'], ensure_ascii=False, sort_keys=True) == \\
        json.dumps(reg24['upgrade_log'], ensure_ascii=False, sort_keys=True)
    for k in ('evidence_level_taxonomy', 'principles', 'schema',
              'models_mainline', 'models_pending', 'provenance', 'created',
              'phase', 'name', 'design_sha8'):
        assert json.dumps(reg25.get(k), ensure_ascii=False, sort_keys=True) == \\
            json.dumps(reg24.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    assert json.dumps(reg25['failures'], ensure_ascii=False, sort_keys=True) == \\
        json.dumps(reg24['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'
    rp = os.path.join(OUTDIR, 'atlas_registry_v1_25.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg25, ensure_ascii=False, indent=1))
    reg25_sha = sha8_file(rp)
    log('registry v1.25 written sha8 %s (%s)' % (reg25_sha, disposition))

    meta_sha = {'reg25': reg25_sha}
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.27 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.27 &mdash; 语义机制图谱（registry v1.25 渲染）</h1>')
    H.append('<div class="sub">Phase 3202 G5-F1 &middot; FTR-04 glm4 弱特异性复读臂（%s） &middot; '
             'registry v1.25 sha8=%s &middot; v1.24 sha8=%s &middot; v1.23 sha8=%s &middot; v1.22 sha8=%s &middot; v1.21 sha8=%s &middot; gap v1.6 sha8=%s'
             ' &middot; 源 p3181 %s + p3178 %s + p3183 %s &middot; 渲染于 %s</div>' % (
                 '复读 CLEAN + REGISTERED（FTR-04 scope_limits 单元级分解 append）' if gate_all else '复读矛盾登记，report-only',
                 span('meta.registry_v125_sha8', reg25_sha),
                 span('meta.registry_v124_sha8', '0d74b628'),
                 span('meta.registry_v123_sha8', '5f5b9e73'),
                 span('meta.registry_v122_sha8', '9fd87433'),
                 span('meta.registry_v121_sha8', '47cf01e0'),
                 span('meta.gap_v16_sha8', '0f6423e0'),
                 span('meta.p3181_result', SHA_ANCHOR['p3181_result']),
                 span('meta.p3178_result', SHA_ANCHOR['p3178_result']),
                 span('meta.p3183_result', SHA_ANCHOR['p3183_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg25['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg25['principles']):
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
    for f in reg25['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.25，%d 条）</h2>' % len(reg25['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg25['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg25['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg25['failures']:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td class="mono">%s</td><td>%s</td><td class="mono">%s</td></tr>' % (
            fl['id'], span(fl['id'] + '.kind', fl['kind']), span(fl['id'] + '.phase', fl['phase']),
            span(fl['id'] + '.text', fl['text']), span(fl['id'] + '.evidence', fl['evidence'])))
    H.append('</table>')
    gl15 = json.loads(open(os.path.join(SRC_DIR, r'phase3175\\g5a12_residual_arms\\gap_ledger_v1_5.json'), encoding='utf-8').read())
    H.append('<h2>缺口账本 v1.5（GAP-4 quantified_collapse 定稿；v1.6 mechanism_note 见 3192 封存）</h2>')
    for g in gl15['gaps']:
        closed_at = str(g['closed_at_phase']) if g.get('closed_at_phase') is not None else '-'
        badge_closed = '<span class="badge">closed@%s</span>' % span(g['id'] + '.closed_at', closed_at) \\
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
    H.append('<div class="footer">RDC/LPF atlas v1.27 &middot; 渲染自 atlas_registry_v1_25.json (%s)'
             ' + gap ledger v1.5 (4fb3f37d) &middot; 增量来源: Phase 3202 G5-F1 FTR-04 glm4 弱特异性复读臂'
             '（数据源 p3181 %s）'
             '&middot; 本文件为 Phase 3202 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg25_sha, SHA_ANCHOR['p3181_result']))
    H.append('</body></html>')
    html_text = '\\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_27.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v122(reg25, nodes, gl15, meta_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_27.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3202, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'recheck_clean': bool(gate_all), 'recheck': audit,
                'decomposition': audit.get('decomposition')},
        'counts': {'features': len(reg25['features']), 'failures': len(reg25['failures']),
                   'upgrades': len(reg25['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v125_file': 'atlas_registry_v1_25.json', 'registry_v125_sha8': reg25_sha,
        'registry_v124_sha8_ref': '0d74b628',
        'html_file': 'atlas_v1_27.html', 'html_sha8': html_sha,
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


''')

# ---- P10 flatten meta keys ----
rep("""    d['meta.registry_v124_sha8'] = meta_sha['reg24']
    d['meta.registry_v123_sha8'] = '5f5b9e73'
    d['meta.registry_v122_sha8'] = '9fd87433'
    d['meta.registry_v121_sha8'] = '47cf01e0'
    d['meta.registry_v120_sha8'] = '2aac7e5e'""",
"""    d['meta.registry_v125_sha8'] = meta_sha['reg25']
    d['meta.registry_v124_sha8'] = '0d74b628'
    d['meta.registry_v123_sha8'] = '5f5b9e73'
    d['meta.registry_v122_sha8'] = '9fd87433'
    d['meta.registry_v121_sha8'] = '47cf01e0'
    d['meta.p3181_result'] = SHA_ANCHOR['p3181_result']""")

# ---- P11 main ----
rep("""    reg_v123 = json.load(open(SRC['registry_v123'], encoding='utf-8'))
    assert reg_v123['version'] == '1.23' and len(reg_v123['features']) == 26
    log('sealed registry v1.23 loaded (%s)' % SRC['registry_v123'])""",
"""    sealed_3181 = json.load(open(SRC['p3181_result'], encoding='utf-8'))
    assert sealed_3181['res_sha8'] == CONTENT_3181[0] and sealed_3181['seal_sha8'] == CONTENT_3181[1]
    log('sealed 3181 double-locked (res 1e95a19d / seal 462fd221)')
    reg_v124 = json.load(open(SRC['registry_v124'], encoding='utf-8'))
    assert reg_v124['version'] == '1.24' and len(reg_v124['features']) == 26
    log('sealed registry v1.24 loaded (%s)' % SRC['registry_v124'])""")

rep("""    control = run_audit(reg_v123)
    gate_all = bool(control['audit_clean'])
    if gate_all:
        verdict = ('eline_closed (audit A1-A3 clean 0 contradictions; FTR-24 '
                   'scope_limits methodology verdict append; registry v1.23 -> '
                   'v1.24, atlas_v1_26)')
    else:
        verdict = ('eline_audit_dirty (%d contradictions registered; report-only, '
                   'registry v1.23 unchanged)' % len(control['contradictions']))""",
"""    control = run_recheck(sealed_3181, reg_v124)
    gate_all = bool(control['recheck_clean'])
    if gate_all:
        verdict = ('ftr04_recheck_registered (3181 scalars 4dp drift==0 x3; FTR-04 '
                   'p3181 anchor bitwise consistent; decomposition loeo_gap %.4f / '
                   'margin_ratio %.4f / geometry band OK; FTR-04 scope_limits '
                   'append; registry v1.24 -> v1.25, atlas_v1_27)'
                   % (control['decomposition']['loeo_gap'],
                      control['decomposition']['margin_ratio']))
    else:
        verdict = ('ftr04_recheck_dirty (%d contradictions; report-only, registry '
                   'v1.24 unchanged)' % len(control['contradictions']))""")

rep("""    log('== Phase 3201 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))""",
    """    log('== Phase 3202 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))""")

rep("""    summary = {
        'phase': 3201, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',""",
    """    summary = {
        'phase': 3202, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',""")

rep("""        'gates': dict(GATE, eline_closed=bool(gate_all),
                      n_contradictions=len(control['contradictions'])),
        'audit': control,""",
    """        'gates': dict(GATE, recheck_clean=bool(gate_all),
                      n_contradictions=len(control['contradictions'])),
        'recheck': control,""")

rep("# ---------------- registry v1.22 -> v1.23 + atlas_v1_25 ----------------",
    "# ---------------- registry v1.24 -> v1.25 + atlas_v1_27 ----------------")

# ---- P12 leftover assertions ----
for bad in ('run_audit', 'reg_v123', 'eline_closed', 'atlas_v1_26',
            'ANCH_3178', 'UPG_SEQ_PINNED',
            'FTR-24', 'audit_clean', 'v1_23_', 'v1_22_to'):
    assert bad not in t, ('leftover', bad)
assert t.count('Phase 3201') == 3  # docstring prereg + DESIGN prereg + supersedes provenance
assert t.count("'phase': 3201") == 0
assert 'reg25' in t and 'run_recheck' in t and 'PIN_3181' in t

io.open(DST, 'w', encoding='utf-8', newline='').write(t)
back = io.open(DST, encoding='utf-8').read()
assert back == t
print('written %s sha8 %s' % (DST, hashlib.sha256(t.encode()).hexdigest()[:8]))
