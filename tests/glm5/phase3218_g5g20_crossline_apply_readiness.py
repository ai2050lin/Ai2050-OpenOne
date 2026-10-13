# -*- coding: utf-8 -*-
"""Phase 3218 (G5-G20): cross-line apply readiness inventory (report-only,
zero GPU, non-mutating).

Per prereg in MEMO Phase 3217 tail G5-G20 (frozen before any observation;
default arm (a)): inventory + one-shot application checklist for the two
pending cross-line items — NO actual application:
  - N-line backfill draft (f29da2de, draft_pending_cross_line): 5 entries
    (N-P3..N-P7) itemized entry-wise; plan = single append batch of 5 to
    atlas_ledger measurements, one cross-line confirmation, bak + rollback;
  - APPX-5 ledger_corrections_v1 (beb775bd, SEALED_BUT_NOT_APPLIED):
    4 operations itemized (set_field ledger_sha256_8, unify_schema,
    enumerate_field, append_measurement); CRITICAL FINDING: patch
    precondition target_sha256_before = bbda63df vs current ledger
    7978e4f7 => baseline STALE, re-baseline required before apply.
  Checks: R1 draft itemized (5, sources keys per entry); R2 ops itemized
  (4) + stale-baseline finding recorded; R3 checklist sealed
  (crossline_apply_checklist_v1.json, content sha); R4 ledger untouched
  (n=369, 7978e4f7). Verdict: crossline_apply_readiness_registered iff
  R1-R4 clean.
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
OUTDIR = os.path.join(SRC_DIR, 'phase3218', 'g5g20_crossline_apply_readiness')
SMOKE = os.environ.get('P3218_SMOKE') == '1'
LOG = []

SRC = {
    'draft': os.path.join(SRC_DIR, r'phase3208\g5g7_nline_backfill_draft\n_line_backfill_draft_v1.json'),
    'lcorr': os.path.join(ROOT, r'research\deepseek\atlas\ledger_corrections_v1.json'),
    'ledger': os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'),
}
SHA_ANCHOR = {'draft': 'f29da2de', 'lcorr': 'beb775bd'}
LED_N, LED_POST = 369, '7978e4f7'

DESIGN = {
    'phase': '3218', 'name': 'g5g20_crossline_apply_readiness',
    'prereg': 'AGI_GPT5_MEMO Phase 3217 tail G5-G20 (frozen before any '
              'observation; default arm (a) cross-line apply readiness '
              'inventory)',
    'data': 'N-line draft (f29da2de) + ledger_corrections_v1 (beb775bd) + '
            'atlas ledger (read-only, n=369, 7978e4f7); zero GPU, '
            'report-only, no application performed',
    'gate': 'registered iff R1 N-line 5 entries itemized + R2 lcorr 4 ops '
            'itemized with stale-baseline finding + R3 checklist sealed '
            '(crossline_apply_checklist_v1.json) + R4 ledger untouched '
            '(n=369, 7978e4f7)',
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
        json.dump({'phase': 3218, 'name': 'g5g20_crossline_apply_readiness',
                   'design_sha8': dsha[:8], 'design': DESIGN,
                   'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def main():
    design_sha = freeze()
    log('== Phase 3218 g5g20_crossline_apply_readiness %s ==' % ('SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources sha-asserted: %s' % sorted(SHA_ANCHOR))
    draft = json.load(open(SRC['draft'], encoding='utf-8'))
    lcorr = json.load(open(SRC['lcorr'], encoding='utf-8'))
    led = json.load(open(SRC['ledger'], encoding='utf-8'))
    assert draft['status'] == 'draft_pending_cross_line' and len(draft['entries']) == 5
    assert lcorr['status'] == 'SEALED_BUT_NOT_APPLIED' and len(lcorr['operations']) == 4
    assert len(led['measurements']) == LED_N and led['ledger_sha256_8'] == LED_POST
    log('inputs loaded: draft (5, pending) + lcorr (SEALED_BUT_NOT_APPLIED, 4 ops) '
        '+ ledger (%d, %s)' % (LED_N, LED_POST))

    contradictions = []

    # R1: N-line entries itemized
    r1_bad = []
    n_items = []
    for e in draft['entries']:
        item = {'phase': e['phase'], 'name': e['name'], 'status': e['status'],
                'sources_keys': sorted(e['sources'].keys())}
        n_items.append(item)
        if not (item['phase'] and item['name'] and item['sources_keys']):
            r1_bad.append(('entry incomplete', e.get('phase')))
    c1 = draft['target'].endswith('atlas_ledger.json#measurements (append 5)') and not r1_bad
    if not c1:
        contradictions.append(('R1_nline', r1_bad or [draft['target']]))
    log('R1: N-line 5 entries itemized (append-batch target ok): %s'
        % ('OK' if c1 else 'FAIL %s' % (r1_bad[:2],)))

    # R2: lcorr ops itemized + stale-baseline finding
    ops = [{'idx': i, 'op': op.get('op'), 'note': op.get('note', '')[:60]}
           for i, op in enumerate(lcorr['operations'])]
    stale = (lcorr['target_sha256_before'] != led['ledger_sha256_8'])
    finding = ('APPX-5 patch precondition target_sha256_before=%s vs current '
               'ledger %s => baseline STALE; re-baseline (recompute ops '
               'against current ledger state) required before any apply'
               % (lcorr['target_sha256_before'][:8], led['ledger_sha256_8']))
    c2 = (len(ops) == 4 and all(o['op'] for o in ops)
          and lcorr['target'].endswith('atlas_ledger.json') and stale)
    if not c2:
        contradictions.append(('R2_lcorr', [len(ops), lcorr['target_sha256_before'][:8]]))
    log('R2: lcorr 4 ops itemized + stale-baseline finding (bbda63df != %s): %s'
        % (led['ledger_sha256_8'], 'OK (finding recorded)' if c2 else 'FAIL'))

    # R3: checklist sealed
    checklist = {
        'schema': 'crossline_apply_checklist_v1',
        'created': time.strftime('%Y-%m-%d %H:%M'),
        'by_phase': 3218,
        'governance': 'inventory only; NO application performed this phase; '
                      'application requires explicit cross-line confirmation '
                      '(APPX-4/5 同型：只补不改)',
        'nline': {
            'source': 'n_line_backfill_draft_v1.json (f29da2de)',
            'status': draft['status'], 'target': draft['target'],
            'n_entries': len(n_items), 'entries': n_items,
            'apply_plan': ['one cross-line confirmation', 'bak ledger file',
                           'single append batch of 5 to measurements',
                           'recompute ledger_sha256_8 (post-sha convention)',
                           'rollback = restore bak + assert equality'],
        },
        'appx5': {
            'source': 'ledger_corrections_v1.json (beb775bd)',
            'status': lcorr['status'], 'target': lcorr['target'],
            'target_sha256_before': lcorr['target_sha256_before'],
            'recorded_correct_value': lcorr['recorded_correct_value'],
            'n_operations': len(ops), 'operations': ops,
            'finding_stale_baseline': finding,
            'apply_plan': ['cross-line confirmation', 'RE-BASELINE ops to '
                           'current ledger sha', 'apply 4 ops in order',
                           'verify final sha against recomputed convention',
                           'bak + rollback'],
        },
        'provenance': {'draft': 'f29da2de', 'lcorr': 'beb775bd',
                       'ledger_at_inventory': '%s (n=%d)' % (LED_POST, LED_N)},
    }
    cblob = json.dumps(checklist, ensure_ascii=False, indent=1, sort_keys=True)
    c_sha = sha8(cblob.encode('utf-8'))
    cblob = json.dumps(dict(checklist, sha8=c_sha), ensure_ascii=False, indent=1, sort_keys=True)
    c3 = (c_sha and len(cblob) > 1000)
    if not c3:
        contradictions.append(('R3_checklist',))
    log('R3: crossline_apply_checklist_v1 content sha %s: %s'
        % (c_sha, 'OK' if c3 else 'FAIL'))
    cp = os.path.join(OUTDIR, 'crossline_apply_checklist_v1.json')
    if not SMOKE:
        with open(cp, 'w', encoding='utf-8', newline='') as f:
            f.write(cblob)
        log('checklist written, file sha8 %s' % sha8_file(cp))

    # R4: ledger untouched
    led2 = json.load(open(SRC['ledger'], encoding='utf-8'))
    c4 = (len(led2['measurements']) == LED_N and led2['ledger_sha256_8'] == LED_POST)
    if not c4:
        contradictions.append(('R4_ledger',))
    log('R4: ledger untouched (n=%d, %s): %s' % (LED_N, LED_POST, 'OK' if c4 else 'FAIL'))

    clean = not contradictions
    verdict = ('crossline_apply_readiness_registered' if clean
               else 'crossline_apply_readiness_dirty (%d contradictions)' % len(contradictions))
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3218, 'name': 'g5g20_crossline_apply_readiness',
        'mode': 'SMOKE' if SMOKE else 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True, 'report_only': True, 'no_application': True,
        'gates': {'recheck_clean': bool(clean), 'n_contradictions': len(contradictions)},
        'checklist_content_sha8': c_sha,
        'nline_entries': len(n_items), 'appx5_operations': len(ops),
        'stale_baseline_finding': bool(stale),
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
