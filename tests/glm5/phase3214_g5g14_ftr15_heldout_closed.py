# -*- coding: utf-8 -*-
"""Phase 3214 (G5-G14): FTR-15 held_out closed-loop confirmation arm
(report-only, zero GPU).

Per prereg in MEMO Phase 3213 tail G5-G14 (frozen before any observation;
default arm (a)): re-read the FTR-15 held_out evidence chain. Probe finding
(pre-freeze): the held_out reinforcement recheck is ALREADY closed — 3204
G5-G1 (registry v1.25->v1.26, FTR-15 scope_limits append) re-read the
sealed 3158/3180 chain. This arm therefore runs as a closure-confirmation:
  H1 - 3158 sealed per-model products on disk (file sha c49614fc /
       1f532016 / 90e734fd): /ratio bitwise == FTR-15 p3158 anchor asserts
       (1.0283477199188433 / 1.3316768638966026 / 1.2472595295618603).
  H2 - 3180 sealed product (res f1689e94 / seal a3c1b53e): per_model
       ratio_full_recompute bitwise == 3158 ratios x3 == p3180 asserts;
       drift == 0.0 x3; max_band bitwise == p3180 asserts (0.030024603607
       133406 / 0.027780384524118094 / 0.018213681402158396 < 0.25);
       all_same_class true x3; n_c_gt1 10/16/15; 16 folds x3 all
       same_class; gates G1_device/G2_heldout True.
  H3 - 3204 sealed product (res 263934c8 / seal 7bb722b1 / design
       588dcf87, verdict ftr15_heldout_recheck_registered, recheck clean)
       + registry v1.28 FTR-15 scope_limits 3204 append verbatim +
       E2_held_out caliber note on disk + MEMO 3204 seg anchors.
  H4 - ledger untouched (n=365, 3bd36b2a before==after).
  Verdict (report-only): ftr15_heldout_closed (closed@3204; E2_predictive
  with E2_held_out caliber note maintained; no registry action).
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
OUTDIR = os.path.join(SRC_DIR, 'phase3214', 'g5g14_ftr15_heldout_closed')
SMOKE = os.environ.get('P3214_SMOKE') == '1'
LOG = []

SRC = {
    'r3158_4b': os.path.join(SRC_DIR, r'phase3158\g4p1_output_equivalence_class\qwen3-4b\result.json'),
    'r3158_14b': os.path.join(SRC_DIR, r'phase3158\g4p1_output_equivalence_class\qwen3-14b\result.json'),
    'r3158_glm': os.path.join(SRC_DIR, r'phase3158\g4p1_output_equivalence_class\glm4\result.json'),
    'r3180': os.path.join(SRC_DIR, r'phase3180\g5b5_ftr15_heldout\result.json'),
    'r3204': os.path.join(SRC_DIR, r'phase3204\g5g1_ftr15_heldout_recheck\result.json'),
    'reg128': os.path.join(SRC_DIR, r'phase3206\g5g5_ftr22_chain_recheck\atlas_registry_v1_28.json'),
    'ledger': os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'),
    'memo': os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md'),
}
SHA_ANCHOR = {'r3158_4b': '90e734fd', 'r3158_14b': '1f532016', 'r3158_glm': 'c49614fc',
              'r3180': 'cb811310', 'r3204': 'b1021387'}

DESIGN = {
    'phase': '3214', 'name': 'g5g14_ftr15_heldout_closed',
    'prereg': 'AGI_GPT5_MEMO Phase 3213 tail G5-G14 (frozen before any '
              'observation; default arm (a) FTR-15 held_out re-read; probe '
              'found the recheck already closed at 3204 G5-G1 => runs as '
              'closure-confirmation arm)',
    'data': 'sealed 3158/3180/3204 products (read-only) + registry v1.28 '
            '(read-only) + MEMO (read-only); zero GPU, report-only',
    'gate': 'registered iff H1 3158 ratios bitwise + H2 3180 held-out chain '
            'bitwise + H3 3204 product/append/MEMO + H4 ledger untouched; '
            'disposition: closed@3204, E2 caliber unchanged',
    'sources': {'3158': '90e734fd/1f532016/c49614fc (file bytes)',
                '3180': 'cb811310 (file bytes; res f1689e94/seal a3c1b53e)',
                '3204': 'b1021387 (file bytes; res 263934c8/seal 7bb722b1)',
                'registry': 'v1.28 5995bbe9', 'memo': 'read-only'},
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
        json.dump({'phase': 3214, 'name': 'g5g14_ftr15_heldout_closed',
                   'design_sha8': dsha[:8], 'design': DESIGN,
                   'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def run_audit(r4b, r14b, rglm, r3180, r3204, reg, memo, led):
    out = {'contradictions': [], 'dispositions': {}}
    f15 = [f for f in reg['features'] if f['id'] == 'FTR-15'][0]
    a3158 = [a for a in f15['anchors'] if a['src'] == 'p3158'][0]
    a3180 = [a for a in f15['anchors'] if a['src'] == 'p3180'][0]
    ratios = {'qwen3-4b': r4b['ratio'], 'qwen3-14b': r14b['ratio'], 'glm4': rglm['ratio']}
    mkey = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4': 'glm4'}

    # H1: 3158 ratios bitwise == p3158 anchor asserts
    h1_bad = []
    for mk, val in ratios.items():
        k = 'ratios.' + mk
        if a3158['asserts'].get(k) != val:
            h1_bad.append((k, a3158['asserts'].get(k), val))
    if h1_bad:
        out['contradictions'].append(('H1_3158', h1_bad))
    log('H1: 3158 sealed ratios bitwise == FTR-15 p3158 anchor x3: %s'
        % ('OK' if not h1_bad else 'FAIL %s' % h1_bad[:2]))

    # H2: 3180 held-out chain bitwise
    h2_bad = []
    if not (r3180['res_sha8'] == 'f1689e94' and r3180['seal_sha8'] == 'a3c1b53e'):
        h2_bad.append(('3180 locks', r3180['res_sha8'], r3180['seal_sha8']))
    if not r3180['verdict'].startswith('ftr15_upgraded_E2_predictive'):
        h2_bad.append(('3180 verdict', r3180['verdict']))
    for mk, val in ratios.items():
        pm = r3180['per_model'][mkey[mk]]
        if pm['ratio_full_recompute'] != val:
            h2_bad.append(('ratio_full_recompute.' + mk, pm['ratio_full_recompute'], val))
        if a3180['asserts'].get('ratio_full_recompute.' + mk) != val:
            h2_bad.append(('anchor ratio.' + mk,))
        if pm['drift'] != 0.0:
            h2_bad.append(('drift.' + mk, pm['drift']))
        if a3180['asserts'].get('max_log_band.' + mk) != pm['max_band']:
            h2_bad.append(('max_band.' + mk, a3180['asserts'].get('max_log_band.' + mk), pm['max_band']))
        if not (pm['max_band'] < 0.25):
            h2_bad.append(('max_band gate.' + mk, pm['max_band']))
        if pm['all_same_class'] is not True or a3180['asserts'].get('all_same_class') is not True:
            h2_bad.append(('all_same_class.' + mk,))
        if len(r3180['folds_full'][mkey[mk]]) != 16:
            h2_bad.append(('n_folds.' + mk, len(r3180['folds_full'][mkey[mk]])))
        if not all(fd['same_class'] for fd in r3180['folds_full'][mkey[mk]]):
            h2_bad.append(('fold same_class.' + mk,))
    nc = [r3180['per_model']['qwen3-4b']['n_c_gt1'],
          r3180['per_model']['qwen3-14b']['n_c_gt1'],
          r3180['per_model']['glm4']['n_c_gt1']]
    if nc != [10, 16, 15]:
        h2_bad.append(('n_c_gt1', nc))
    if r3180['gates'].get('G1_device') is not True or r3180['gates'].get('G2_heldout') is not True:
        h2_bad.append(('gates', r3180['gates']))
    if h2_bad:
        out['contradictions'].append(('H2_3180', h2_bad[:5]))
    log('H2: 3180 sealed chain (ratio bitwise x3, drift 0, band <0.25, '
        '16 folds same_class, n_c_gt1 10/16/15, gates): %s'
        % ('OK' if not h2_bad else 'FAIL %s' % h2_bad[:3]))

    # H3: 3204 product + registry append + MEMO anchors
    h3_bad = []
    if not (r3204['res_sha8'] == '263934c8' and r3204['seal_sha8'] == '7bb722b1'
            and r3204['design_sha8'] == '588dcf87'):
        h3_bad.append(('3204 locks', r3204['res_sha8'], r3204['seal_sha8']))
    if not r3204['verdict'].startswith('ftr15_heldout_recheck_registered'):
        h3_bad.append(('3204 verdict', r3204['verdict']))
    if not r3204['recheck']['recheck_clean']:
        h3_bad.append(('3204 recheck dirty',))
    sl = f15['scope_limits']
    for frag in ['3204 G5-G1 FTR-15 held_out 补强复读',
                 'ratio_full_recompute 逐位 == 3158 ×3（device drift 0 ×3）',
                 'max_log_band 0.0300/0.0278/0.0182 <0.25',
                 'E2_held_out 为 E2_predictive 强化档注记，不新增独立 evidence_level']:
        if frag not in sl:
            h3_bad.append(('scope_limits', frag[:30]))
    i = memo.find('## Phase 3204')
    j = memo.find('\n## ', i + 1)
    seg = memo[i:j]
    for kw in ['## Phase 3204 — G5-G1 FTR-15 held_out 补强复读臂',
               'E2_held_out', 'held_out']:
        if kw not in seg:
            h3_bad.append(('memo 3204', kw[:25]))
    if h3_bad:
        out['contradictions'].append(('H3_3204', h3_bad[:5]))
    log('H3: 3204 sealed product + registry append verbatim + MEMO anchors: %s'
        % ('OK' if not h3_bad else 'FAIL %s' % h3_bad[:3]))

    out['dispositions']['FTR-15 held_out'] = {
        'provenance': '3158 primary (quotient budget ratios) + 3180 '
                      'entity-LOO held_out arm (16 folds x3, E2 upgrade) + '
                      '3204 G5-G1 reinforcement recheck (registry '
                      'v1.25->v1.26, scope_limits append)',
        'status': 'closed_at_3204',
        'conclusion': 'held_out evidence = entity LOO fold (16 anchors = 16 '
                      'entities) caliber; ratio_full_recompute bitwise == '
                      '3158 x3 with device drift 0; max log band '
                      '0.0300/0.0278/0.0182 < 0.25; E2_predictive with '
                      'E2_held_out caliber note (no separate evidence_level)',
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
    log('== Phase 3214 g5g14_ftr15_heldout_closed %s =='
        % ('SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sealed sources sha-asserted: %s' % sorted(SHA_ANCHOR))
    r4b = json.load(open(SRC['r3158_4b'], encoding='utf-8'))
    r14b = json.load(open(SRC['r3158_14b'], encoding='utf-8'))
    rglm = json.load(open(SRC['r3158_glm'], encoding='utf-8'))
    r3180 = json.load(open(SRC['r3180'], encoding='utf-8'))
    r3204 = json.load(open(SRC['r3204'], encoding='utf-8'))
    reg = json.load(open(SRC['reg128'], encoding='utf-8'))
    led = json.load(open(SRC['ledger'], encoding='utf-8'))
    memo = open(SRC['memo'], encoding='utf-8').read()
    assert reg['version'] == '1.28' and len(reg['features']) == 26
    assert len(led['measurements']) == 365 and led['ledger_sha256_8'] == '3bd36b2a'
    log('inputs loaded: 3158 x3 + 3180 + 3204 sealed + registry v1.28 (26) '
        '+ ledger (365, 3bd36b2a) + MEMO')

    control = run_audit(r4b, r14b, rglm, r3180, r3204, reg, memo, led)
    gate_all = bool(control['recheck_clean'])
    if gate_all:
        verdict = ('ftr15_heldout_closed (H1 3158 ratios bitwise x3; H2 '
                   '3180 held-out chain clean — ratio_full_recompute '
                   'bitwise == 3158 x3, drift 0 x3, band '
                   '0.0300/0.0278/0.0182 <0.25, 16 folds same_class x3, '
                   'n_c_gt1 10/16/15; H3 3204 sealed + registry append '
                   'verbatim + MEMO anchors, E2_predictive with E2_held_out '
                   'caliber note maintained; disposition closed@3204, no '
                   'registry action, report-only)')
    else:
        verdict = ('ftr15_heldout_dirty (%d contradictions; report-only)'
                   % len(control['contradictions']))
    log('verdict: %s' % verdict)
    led2 = json.load(open(SRC['ledger'], encoding='utf-8'))
    assert len(led2['measurements']) == 365 and led2['ledger_sha256_8'] == '3bd36b2a'
    log('H4: ledger untouched re-assertion OK (n=365, 3bd36b2a)')

    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3214, 'name': 'g5g14_ftr15_heldout_closed',
        'mode': 'SMOKE' if SMOKE else 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True, 'report_only': True,
        'gates': {'H1_3158': not any(c[0] == 'H1_3158' for c in control['contradictions']),
                  'H2_3180': not any(c[0] == 'H2_3180' for c in control['contradictions']),
                  'H3_3204': not any(c[0] == 'H3_3204' for c in control['contradictions']),
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
