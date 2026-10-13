# -*- coding: utf-8 -*-
"""Phase 3220 (G5-G22): 5-appendix on-disk anchor re-read (report-only,
zero GPU).

Per prereg in MEMO Phase 3219 tail G5-G22 (frozen before any observation;
default arm (a)): re-read the on-disk artifacts behind APPX-1..5 anchors
and re-affirm the "cannot be closed zero-GPU" conclusion — NO status
change, NO registry/gap action:
  A1 - GPU-type anchors APPX-1/2/3 re-read with file-sha lock + key
       scalars: p3166 result.json 448af595 (census mixed, cls_agree
       false), p3165 result.json 511d9b13 (cross_model_K present,
       tokenizer-side), p3155 summary/result_summary.json 46a5034d
       (k2_int_share_kout 0.1519/0.1825/0.1641);
  A2 - zero-GPU non-closable re-affirmation: each of APPX-1..3 requires
       new GPU measurement (14b eff>=0.5 runs / per-model vocab
       rebuild / K2-K3 direct measurement) => stays open;
  A3 - crossline anchors APPX-4/5 (agate 24c60160,
       a_gate_closure_v1.json) re-read; N-line draft still
       draft_pending_cross_line, lcorr still SEALED_BUT_NOT_APPLIED;
  A4 - ledger untouched (n=371, 5cff05f6).
Verdict: appendix_anchors_reread_registered iff A1-A4 clean.
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
OUTDIR = os.path.join(SRC_DIR, 'phase3220', 'g5g22_appendix_anchors_reread')
SMOKE = os.environ.get('P3220_SMOKE') == '1'
LOG = []

SRC = {
    'gl15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'p3166': os.path.join(SRC_DIR, r'phase3166\g5a3b_logic_direction\result.json'),
    'p3165': os.path.join(SRC_DIR, r'phase3165\g5a3_family_alignment\result.json'),
    'p3155': os.path.join(SRC_DIR, r'phase3155\g2p1_relation_family_operator_separability\summary\result_summary.json'),
    'agate': os.path.join(ROOT, r'research\deepseek\atlas\a_gate_closure_v1.json'),
    'draft': os.path.join(SRC_DIR, r'phase3208\g5g7_nline_backfill_draft\n_line_backfill_draft_v1.json'),
    'lcorr': os.path.join(ROOT, r'research\deepseek\atlas\ledger_corrections_v1.json'),
    'ledger': os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'),
}
SHA_ANCHOR = {'gl15': '4fb3f37d', 'p3166': '448af595', 'p3165': '511d9b13',
              'p3155': '46a5034d', 'agate': '24c60160',
              'draft': 'f29da2de', 'lcorr': 'beb775bd'}
LED_N, LED_POST = 371, '5cff05f6'

DESIGN = {
    'phase': '3220', 'name': 'g5g22_appendix_anchors_reread',
    'prereg': 'AGI_GPT5_MEMO Phase 3219 tail G5-G22 (frozen before any '
              'observation; default arm (a) 5-appendix on-disk anchor re-read)',
    'data': 'gl15 (4fb3f37d) + p3166/p3165/p3155 anchors (448af595/511d9b13/'
            '46a5034d) + agate (24c60160) + draft/lcorr + ledger (read-only, '
            'n=371, 5cff05f6); zero GPU, report-only, no status change',
    'gate': 'registered iff A1 three GPU-type anchors re-read with sha lock '
            '+ key scalars + A2 zero-GPU non-closable re-affirmed + A3 '
            'crossline anchors unchanged + A4 ledger untouched (n=371, '
            '5cff05f6)',
    'sources': SHA_ANCHOR,
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
        json.dump({'phase': 3220, 'name': 'g5g22_appendix_anchors_reread',
                   'design_sha8': dsha[:8], 'design': DESIGN,
                   'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def main():
    design_sha = freeze()
    log('== Phase 3220 g5g22_appendix_anchors_reread %s ==' % ('SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources sha-asserted: %s' % sorted(SHA_ANCHOR))
    gl15 = json.load(open(SRC['gl15'], encoding='utf-8'))
    r1 = json.load(open(SRC['p3166'], encoding='utf-8'))
    r2 = json.load(open(SRC['p3165'], encoding='utf-8'))
    r3 = json.load(open(SRC['p3155'], encoding='utf-8'))
    agate = json.load(open(SRC['agate'], encoding='utf-8'))
    draft = json.load(open(SRC['draft'], encoding='utf-8'))
    lcorr = json.load(open(SRC['lcorr'], encoding='utf-8'))
    led = json.load(open(SRC['ledger'], encoding='utf-8'))
    assert len(gl15['appendix']) == 5
    assert len(led['measurements']) == LED_N and led['ledger_sha256_8'] == LED_POST
    log('inputs loaded: gl15 (appendix 5) + p3166/p3165/p3155 + agate + '
        'draft/lcorr + ledger (%d, %s)' % (LED_N, LED_POST))

    contradictions = []

    # A1: GPU-type anchors re-read with key scalars
    a1_bad = []
    ov = r1.get('overall', {})
    if not (ov.get('census_cls') == 'mixed' and ov.get('cls_agree') is False):
        a1_bad.append(('p3166 overall', ov.get('census_cls'), ov.get('cls_agree')))
    if 'cross_model_K' not in r2:
        a1_bad.append(('p3165 cross_model_K missing',))
    ks = r3.get('k2_int_share_kout', {})
    kexp = {'qwen3-4b': 0.15187579769212284, 'qwen3-14b': 0.1824533831239199,
            'glm4': 0.16411745881744746}
    for m, v in kexp.items():
        if ks.get(m) != v:
            a1_bad.append(('p3155 k2_int_share', m, ks.get(m)))
    c1 = not a1_bad
    if not c1:
        contradictions.append(('A1_anchors', a1_bad))
    log('A1: p3166 (census mixed, cls_agree False) + p3165 (cross_model_K) '
        '+ p3155 (k2_int_share 0.1519/0.1825/0.1641) re-read: %s'
        % ('OK' if c1 else 'FAIL %s' % a1_bad[:2]))

    # A2: zero-GPU non-closable re-affirmation
    appx = {a['id']: a for a in gl15['appendix']}
    a2_bad = []
    for aid in ('APPX-1', 'APPX-2', 'APPX-3'):
        if appx[aid]['status'] != 'appendix_open':
            a2_bad.append((aid, appx[aid]['status']))
    c2 = not a2_bad
    if not c2:
        contradictions.append(('A2_nonclosable', a2_bad))
    log('A2: APPX-1..3 statuses appendix_open unchanged (GPU-dependent, '
        'cannot close zero-GPU): %s' % ('OK' if c2 else 'FAIL'))

    # A3: crossline anchors unchanged
    a3_bad = []
    if agate.get('schema') != 'rdc_cross_line_ledger_patch_v1' and 'schema' not in agate:
        a3_bad.append(('agate schema', str(agate.get('schema'))[:40]))
    if draft['status'] != 'draft_pending_cross_line' or len(draft['entries']) != 5:
        a3_bad.append(('draft', draft['status'], len(draft['entries'])))
    if lcorr['status'] != 'SEALED_BUT_NOT_APPLIED' or len(lcorr['operations']) != 4:
        a3_bad.append(('lcorr', lcorr['status'], len(lcorr['operations'])))
    c3 = not a3_bad
    if not c3:
        contradictions.append(('A3_crossline', a3_bad))
    log('A3: agate (24c60160) + N-line draft pending (5) + lcorr '
        'SEALED_BUT_NOT_APPLIED (4 ops) unchanged: %s'
        % ('OK' if c3 else 'FAIL %s' % a3_bad[:2]))

    # A4: ledger untouched
    led2 = json.load(open(SRC['ledger'], encoding='utf-8'))
    c4 = (len(led2['measurements']) == LED_N and led2['ledger_sha256_8'] == LED_POST)
    if not c4:
        contradictions.append(('A4_ledger',))
    log('A4: ledger untouched (n=%d, %s): %s' % (LED_N, LED_POST, 'OK' if c4 else 'FAIL'))

    clean = not contradictions
    verdict = ('appendix_anchors_reread_registered' if clean
               else 'appendix_anchors_reread_dirty (%d contradictions)' % len(contradictions))
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3220, 'name': 'g5g22_appendix_anchors_reread',
        'mode': 'SMOKE' if SMOKE else 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True, 'report_only': True, 'no_status_change': True,
        'gates': {'recheck_clean': bool(clean), 'n_contradictions': len(contradictions)},
        'anchors': {'APPX-1': '448af595', 'APPX-2': '511d9b13',
                    'APPX-3': '46a5034d', 'APPX-4/5': '24c60160'},
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
