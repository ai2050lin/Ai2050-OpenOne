# -*- coding: utf-8 -*-
"""Phase 3212 (G5-G12): FTR-04 glm4 weak-specificity attribution closed-loop
confirmation arm (report-only, zero GPU).

Per prereg in MEMO Phase 3211 tail G5-G12 (frozen before any observation;
default arm (a)): re-read the FTR-04 glm4 attribution evidence chain and
confirm closure. Probe finding (pre-freeze): the attribution is ALREADY
closed on the books — 3181 G5-B6 dual-arm verdict noise_level (registry
v1.7->v1.8) + 3202 G5-F1 unit-level decomposition recheck (registry
v1.24->v1.25, FTR-04 scope_limits append). This arm therefore runs as a
closure-confirmation re-read:
  A1 - 3181 sealed product on disk (res 1e95a19d / seal 462fd221 / design
       7c567972, verdict noise_level) with per_model scalars BITWISE equal
       to the FTR-04 p3181 anchor asserts (kappa x3, proj_ratio.glm4-9b,
       mean_margin.glm4-9b, gates Ga/Gb false).
  A2 - 3202 sealed product on disk (res af81e9f4 / seal 480bd271 / design
       822996d4): decomposition loeo_gap 0.1180952380952382 / margin_ratio
       1.9661111513569065 re-derived from 3181 per_model within 1e-12;
       registry v1.28 FTR-04 scope_limits 3202 append + counter_evidence
       acc triple 0.6019/0.5981/0.4800 verbatim (4dp == per_model).
  A3 - verdict chain complete: E1_repeatable maintained; MEMO 3181 seg
       'noise_level_not_attributable'; MEMO 3202 seg anchors (drift==0 /
       1.9661 / 0.1181 / 读出不一致 / 单元级分解); counter_evidence 3176
       FAILED gate + 3181 归因定判 entries verbatim.
  A4 - ledger untouched (n=363, d637e53f before==after).
  Verdict (report-only): ftr04_glm4_attribution_closed (closed@3202;
  E1_repeatable maintained; no registry upgrade needed — closure text
  already in registry v1.28 scope_limits).
"""
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

ROOT = r'D://AI2050//Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
OUTDIR = os.path.join(SRC_DIR, 'phase3212', 'g5g12_ftr04_attribution_closed')
SMOKE = os.environ.get('P3212_SMOKE') == '1'
LOG = []

SRC = {
    'r3181': os.path.join(SRC_DIR, r'phase3181\g5b6_ftr04_glm4_attribution\result.json'),
    'r3202': os.path.join(SRC_DIR, r'phase3202\g5f1_ftr04_recheck\result.json'),
    'reg128': os.path.join(SRC_DIR, r'phase3206\g5g5_ftr22_chain_recheck\atlas_registry_v1_28.json'),
    'ledger': os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'),
    'memo': os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md'),
}
SHA_ANCHOR = {'r3181': '5ab51de6', 'r3202': 'af7d8a70', 'reg128': None}

DESIGN = {
    'phase': '3212', 'name': 'g5g12_ftr04_attribution_closed',
    'prereg': 'AGI_GPT5_MEMO Phase 3211 tail G5-G12 (frozen before any '
              'observation; default arm (a) FTR-04 glm4 attribution '
              're-read; probe found attribution already closed at 3181+3202 '
              '=> runs as closure-confirmation arm)',
    'data': 'sealed 3181/3202 results (read-only) + registry v1.28 (read-'
            'only) + MEMO (read-only); zero GPU, report-only',
    'gate': 'registered iff A1 3181 anchor bitwise + A2 3202 decomposition '
            're-derived + A3 chain/MEMO anchors + A4 ledger untouched; '
            'disposition: closed@3202, E1_repeatable maintained',
    'sources': {'r3181': '5ab51de6 (file bytes)', 'r3202': 'af7d8a70 '
                '(file bytes)', 'reg_content': '1e95a19d/462fd221 + '
                'af81e9f4/480bd271 + reg v1.28 5995bbe9', 'memo': 'read-only'},
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
        json.dump({'phase': 3212, 'name': 'g5g12_ftr04_attribution_closed',
                   'design_sha8': dsha[:8], 'design': DESIGN,
                   'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def run_audit(r3181, r3202, reg, memo, led):
    out = {'contradictions': [], 'dispositions': {}}
    f04 = [f for f in reg['features'] if f['id'] == 'FTR-04'][0]
    a3181 = [a for a in f04['anchors'] if a['src'] == 'p3181'][0]
    pm = r3181['per_model']

    # A1: 3181 sealed product + p3181 anchor bitwise
    a1_bad = []
    if not (r3181['res_sha8'] == '1e95a19d' and r3181['seal_sha8'] == '462fd221'
            and r3181['design_sha8'] == '7c567972'):
        a1_bad.append(('3181 locks', r3181['res_sha8'], r3181['seal_sha8']))
    if not r3181['verdict'].startswith('noise_level'):
        a1_bad.append(('3181 verdict', r3181['verdict']))
    pairs = [('kappa.qwen3-4b', pm['qwen3-4b']['kappa']),
             ('kappa.qwen3-14b', pm['qwen3-14b']['kappa']),
             ('kappa.glm4-9b', pm['glm4-9b']['kappa']),
             ('proj_ratio.glm4-9b', pm['glm4-9b']['proj_ratio']),
             ('mean_margin.glm4-9b', pm['glm4-9b']['mean_margin'])]
    for k, v in pairs:
        if a3181['asserts'].get(k) != v:
            a1_bad.append((k, a3181['asserts'].get(k), v))
    if a3181['asserts'].get('gate_Ga_geometry') is not False or \
       a3181['asserts'].get('gate_Gb_readout') is not False:
        a1_bad.append(('3181 gates', a3181['asserts'].get('gate_Ga_geometry'),
                       a3181['asserts'].get('gate_Gb_readout')))
    if r3181['gates']['Ga_geometry'] is not False or r3181['gates']['Gb_readout'] is not False:
        a1_bad.append(('3181 result gates',))
    if a1_bad:
        out['contradictions'].append(('A1_3181', a1_bad[:5]))
    log('A1: 3181 sealed product + FTR-04 p3181 anchor bitwise: %s'
        % ('OK' if not a1_bad else 'FAIL %s' % a1_bad[:3]))

    # A2: 3202 sealed product + decomposition re-derivation + registry text
    a2_bad = []
    if not (r3202['res_sha8'] == 'af81e9f4' and r3202['seal_sha8'] == '480bd271'
            and r3202['design_sha8'] == '822996d4'):
        a2_bad.append(('3202 locks', r3202['res_sha8'], r3202['seal_sha8']))
    if not r3202['verdict'].startswith('ftr04_recheck_registered'):
        a2_bad.append(('3202 verdict', r3202['verdict']))
    dec = r3202['recheck']['decomposition']
    gap_derived = pm['qwen3-14b']['loeo_acc'] - pm['glm4-9b']['loeo_acc']
    ratio_derived = pm['glm4-9b']['mean_margin'] / pm['qwen3-4b']['mean_margin']
    if abs(dec['loeo_gap'] - gap_derived) > 1e-12 or abs(dec['loeo_gap'] - 0.1180952380952382) > 1e-15:
        a2_bad.append(('loeo_gap', dec['loeo_gap'], gap_derived))
    if abs(dec['margin_ratio'] - ratio_derived) > 1e-12 or abs(dec['margin_ratio'] - 1.9661111513569065) > 1e-15:
        a2_bad.append(('margin_ratio', dec['margin_ratio'], ratio_derived))
    if dec['geometry_band_ok'] is not True or not r3202['recheck']['recheck_clean']:
        a2_bad.append(('3202 decomposition flags',))
    sl = f04['scope_limits']
    ce = ' | '.join(f04['counter_evidence'])
    for frag in ['3202 G5-F1 FTR-04 glm4 弱特异性单元级分解复读',
                 'LOEO 留出读出准确率最低 0.4800 vs 0.6019/0.5981（gap 0.1181 ≥0.10 钉定 bar）',
                 'mean_margin 最高 5.2326', '支撑 3181 noise_level 定判']:
        if frag not in sl:
            a2_bad.append(('scope_limits', frag[:30]))
    acc_triple = '0.6019/0.5981/0.4800'
    if acc_triple not in ce:
        a2_bad.append(('counter_evidence acc triple',))
    r4 = lambda x: '%.4f' % x
    if (r4(pm['qwen3-4b']['loeo_acc']), r4(pm['qwen3-14b']['loeo_acc']),
            r4(pm['glm4-9b']['loeo_acc'])) != ('0.6019', '0.5981', '0.4800'):
        a2_bad.append(('4dp acc mismatch',))
    if a2_bad:
        out['contradictions'].append(('A2_3202', a2_bad[:5]))
    log('A2: 3202 product + decomposition re-derived + registry append '
        'verbatim: %s' % ('OK' if not a2_bad else 'FAIL %s' % a2_bad[:3]))

    # A3: verdict chain + MEMO verbatim anchors
    a3_bad = []
    if f04['evidence_level'] != 'E1_repeatable':
        a3_bad.append(('E-level changed', f04['evidence_level']))
    i = memo.find('## Phase 3181')
    j = memo.find('\n## ', i + 1)
    seg1 = memo[i:j]
    if 'noise_level_not_attributable' not in seg1:
        a3_bad.append(('memo 3181', 'noise_level_not_attributable'))
    i = memo.find('## Phase 3202')
    j = memo.find('\n## ', i + 1)
    seg2 = memo[i:j]
    for kw in ['drift==0', '1.9661', '0.1181', '读出不一致', '单元级分解', 'noise_level']:
        if kw not in seg2:
            a3_bad.append(('memo 3202', kw))
    if not any('3176 arm (a) FAILED gate' in c for c in f04['counter_evidence']):
        a3_bad.append(('counter_evidence 3176',))
    if not any(c.startswith('3181 归因定判') for c in f04['counter_evidence']):
        a3_bad.append(('counter_evidence 3181',))
    if a3_bad:
        out['contradictions'].append(('A3_chain', a3_bad[:5]))
    log('A3: verdict chain (3176 FAIL -> 3181 noise_level -> 3202 decompose) '
        '+ MEMO anchors: %s' % ('OK' if not a3_bad else 'FAIL %s' % a3_bad[:3]))

    out['dispositions']['FTR-04 glm4 attribution'] = {
        'provenance': '3181 G5-B6 dual-arm noise_level verdict (registry '
                      'v1.7->v1.8) + 3202 G5-F1 unit-level decomposition '
                      'recheck (registry v1.24->v1.25, scope_limits append)',
        'status': 'closed_at_3202',
        'conclusion': 'glm4-9b weak specificity = readout instability '
                      '(margin 5.2326 = 1.966x qwen max, LOEO acc 0.4800 '
                      'lowest) NOT geometry deficit (kappa 44953766.4873 '
                      'vs 48118411.9811 <25%, pr 8.9220 vs 8.7763 <20%, '
                      'e90 8==8); E1_repeatable maintained, no upgrade path',
        'recommendation': 'no registry action needed (closure text already '
                          'in v1.28 scope_limits); drop from candidate '
                          'follow-up list',
    }
    out['recheck_clean'] = not out['contradictions']
    log('RECHECK: %s (%d contradictions)'
        % ('CLEAN' if out['recheck_clean'] else 'DIRTY', len(out['contradictions'])))
    return out


def main():
    design_sha = freeze()
    log('== Phase 3212 g5g12_ftr04_attribution_closed %s =='
        % ('SMOKE' if SMOKE else 'FULL'))
    for k in ('r3181', 'r3202'):
        got = sha8_file(SRC[k])
        assert got == SHA_ANCHOR[k], ('source sha', k, got)
    log('sealed sources sha-asserted: r3181 5ab51de6, r3202 af7d8a70')
    r3181 = json.load(open(SRC['r3181'], encoding='utf-8'))
    r3202 = json.load(open(SRC['r3202'], encoding='utf-8'))
    reg = json.load(open(SRC['reg128'], encoding='utf-8'))
    led = json.load(open(SRC['ledger'], encoding='utf-8'))
    memo = open(SRC['memo'], encoding='utf-8').read()
    assert reg['version'] == '1.28' and len(reg['features']) == 26
    assert len(led['measurements']) == 363 and led['ledger_sha256_8'] == 'd637e53f'
    log('inputs loaded: 3181+3202 sealed + registry v1.28 (26) + ledger (363, d637e53f) + MEMO')

    control = run_audit(r3181, r3202, reg, memo, led)
    gate_all = bool(control['recheck_clean'])
    if gate_all:
        verdict = ('ftr04_glm4_attribution_closed (A1 3181 noise_level '
                   'sealed, p3181 anchor bitwise; A2 3202 decomposition '
                   're-derived 0.1180952380952382/1.9661111513569065 + '
                   'registry append verbatim; A3 chain 3176 FAIL -> 3181 '
                   'noise_level -> 3202 decompose, E1_repeatable '
                   'maintained; disposition closed@3202, no registry '
                   'action, report-only)')
    else:
        verdict = ('ftr04_glm4_attribution_dirty (%d contradictions; '
                   'report-only)' % len(control['contradictions']))
    log('verdict: %s' % verdict)
    led2 = json.load(open(SRC['ledger'], encoding='utf-8'))
    assert len(led2['measurements']) == 363 and led2['ledger_sha256_8'] == 'd637e53f'
    log('A4: ledger untouched re-assertion OK (n=363, d637e53f)')

    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3212, 'name': 'g5g12_ftr04_attribution_closed',
        'mode': 'SMOKE' if SMOKE else 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True, 'report_only': True,
        'gates': {'A1_3181': not any(c[0] == 'A1_3181' for c in control['contradictions']),
                  'A2_3202': not any(c[0] == 'A2_3202' for c in control['contradictions']),
                  'A3_chain': not any(c[0] == 'A3_chain' for c in control['contradictions']),
                  'recheck_clean': bool(gate_all),
                  'n_contradictions': len(control['contradictions'])},
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
    with open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
