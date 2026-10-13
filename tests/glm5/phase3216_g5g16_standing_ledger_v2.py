# -*- coding: utf-8 -*-
"""Phase 3216 (G5-G16): standing-ledger v2 final audit arm (report-only,
zero GPU).

Per prereg in MEMO Phase 3215 tail G5-G16 (frozen before any observation;
default arm (a)): consolidate the 3210-3215 closure results + 5 appendix +
cross-line items into a standing-ledger v2 disposition table.
  M1 - closure sweep: ledger governance entries 3210..3215 (six entries,
       verdict prefixes per phase) on disk; each confirmed closure
       (3210 audit / 3211 UI v1.7 / 3212 FTR-04 glm4 closed@3202 /
       3213 UI v1.8 / 3214 FTR-15 held_out closed@3204 / 3215 GAP-4
       chain closed).
  M2 - 5 appendix (gl15 4fb3f37d) item-by-item status: all open, GPU or
       cross-line dependent => remain standing (cannot be closed zero-GPU);
       table recorded.
  M3 - cross-line items: N-line backfill draft (f29da2de,
       draft_pending_cross_line, NOT applied) + APPX-5 cross-line ledger
       corrections (ledger_corrections_v1 beb775bd on disk, not applied)
       => remain standing.
  M4 - standing-list v2 built and sealed (standing_ledger_v2.json):
       closed = FTR-04 glm4 attribution / FTR-15 held_out / GAP-4 chain /
       水果类 (recommend close, user-confirmed edit pending) / K4
       (recommend close, user-confirmed edit pending); open = 5 appendix
       + N-line apply + APPX-5 apply + UI iteration line. 水果类/K4 still
       verbatim in MEMO 3177 tail (user confirmation pending).
  M5 - ledger untouched (n=367, bb1474e6 before==after).
  Verdict (report-only): standing_ledger_v2_registered.
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
OUTDIR = os.path.join(SRC_DIR, 'phase3216', 'g5g16_standing_ledger_v2')
SMOKE = os.environ.get('P3216_SMOKE') == '1'
LOG = []

SRC = {
    'ledger': os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'),
    'gl15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'gl16': os.path.join(SRC_DIR, r'phase3192\g5c5_gap4_amplitude_attribution\gap_ledger_v1_6.json'),
    'draft': os.path.join(SRC_DIR, r'phase3208\g5g7_nline_backfill_draft\n_line_backfill_draft_v1.json'),
    'lcorr': os.path.join(ROOT, r'research\deepseek\atlas\ledger_corrections_v1.json'),
    'memo': os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md'),
}
SHA_ANCHOR = {'gl15': '4fb3f37d', 'gl16': '0f6423e0', 'draft': 'f29da2de',
              'lcorr': 'beb775bd'}

DESIGN = {
    'phase': '3216', 'name': 'g5g16_standing_ledger_v2',
    'prereg': 'AGI_GPT5_MEMO Phase 3215 tail G5-G16 (frozen before any '
              'observation; default arm (a) standing-ledger v2 final audit)',
    'data': 'atlas ledger (read-only) + gl15/gl16 + N-line draft + '
            'ledger_corrections (read-only) + MEMO (read-only); zero GPU, '
            'report-only',
    'gate': 'registered iff M1 closure sweep 3210-3215 + M2 appendix x5 '
            'table + M3 cross-line items + M4 standing_ledger_v2 sealed + '
            'M5 ledger untouched',
    'sources': {'gl15': '4fb3f37d', 'gl16': '0f6423e0', 'draft': 'f29da2de',
                'lcorr': 'beb775bd', 'memo': 'read-only'},
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
        json.dump({'phase': 3216, 'name': 'g5g16_standing_ledger_v2',
                   'design_sha8': dsha[:8], 'design': DESIGN,
                   'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


V2 = {
    'schema': 'standing_ledger_v2',
    'phase': 3216,
    'created': None,
    'closed': [],
    'open': [],
    'pending_user_confirmation': [],
}


def run_audit(led, gl15, gl16, draft, lcorr, memo):
    out = {'contradictions': [], 'dispositions': {}}

    # M1: closure sweep 3210..3215
    m1_bad = []
    expect = {3210: 'g5g9_standing_items_audit|REGISTERED',
              3211: 'g5g10_ui_v17_anchor_values|REGISTERED',
              3212: 'g5g12_ftr04_attribution_closed|REGISTERED',
              3213: 'g5g13_ui_v18_block_anchor_details|REGISTERED',
              3214: 'g5g14_ftr15_heldout_closed|REGISTERED',
              3215: 'g5g15_gap4_closed|REGISTERED'}
    by_ph = {}
    for m in led['measurements']:
        ph = m.get('phase')
        if ph in expect:
            by_ph[ph] = m
    for ph, vp in sorted(expect.items()):
        m = by_ph.get(ph)
        if m is None:
            m1_bad.append((ph, 'governance entry missing'))
        elif not m['verdict'].startswith(vp):
            m1_bad.append((ph, m['verdict'][:40]))
    if m1_bad:
        out['contradictions'].append(('M1_closure_sweep', m1_bad))
    log('M1: closure sweep 3210-3215 governance entries x6: %s'
        % ('OK' if not m1_bad else 'FAIL %s' % m1_bad[:3]))

    # M2: appendix x5 status table
    m2_bad = []
    appx_rows = []
    for a in gl15['appendix']:
        st = a['status']
        appx_rows.append({'id': a['id'], 'title': a['title'], 'status': st,
                          'anchor': a['anchor'], 'anchor_sha8': a['anchor_sha8']})
        if not st.startswith('appendix_open'):
            m2_bad.append((a['id'], st))
    if len(appx_rows) != 5:
        m2_bad.append(('n_appx', len(appx_rows)))
    if m2_bad:
        out['contradictions'].append(('M2_appendix', m2_bad))
    log('M2: appendix x5 all open (GPU/cross-line dependent): %s'
        % ('OK' if not m2_bad else 'FAIL %s' % m2_bad[:3]))

    # M3: cross-line items
    m3_bad = []
    if draft['status'] != 'draft_pending_cross_line':
        m3_bad.append(('draft status', draft['status']))
    if len(draft['entries']) != 5:
        m3_bad.append(('draft entries', len(draft['entries'])))
    # lcorr schema: single correction record, top-level 'operations' list
    n_lcorr = (len(lcorr.get('operations', [])) if isinstance(lcorr, dict) else 0)
    if n_lcorr < 1:
        m3_bad.append(('ledger_corrections empty', n_lcorr))
    if m3_bad:
        out['contradictions'].append(('M3_crossline', m3_bad))
    log('M3: N-line draft pending (5) + ledger_corrections on disk (%d): %s'
        % (n_lcorr, 'OK' if not m3_bad else 'FAIL %s' % m3_bad[:3]))

    # M4: standing-list v2
    m4_bad = []
    first = memo.find('水果类、K4')
    if first < 0:
        m4_bad.append(('水果类、K4 no longer in MEMO (unexpected before user confirm)',))
    else:
        ph_i = memo.rfind('## Phase', 0, first)
        hdr = memo[ph_i:ph_i + 24].splitlines()[0]
        if '3177' not in hdr:
            m4_bad.append(('first list occurrence moved', hdr))
    if m4_bad:
        out['contradictions'].append(('M4_list', m4_bad))
    log('M4: 水果类/K4 still verbatim in 3177 tail (user confirm pending): %s'
        % ('OK' if not m4_bad else 'FAIL %s' % m4_bad[:3]))

    out['dispositions']['appendix_rows'] = appx_rows
    out['dispositions']['n_ledger_corrections'] = n_lcorr
    out['recheck_clean'] = not out['contradictions']
    log('RECHECK: %s (%d contradictions)'
        % ('CLEAN' if out['recheck_clean'] else 'DIRTY', len(out['contradictions'])))
    return out


def main():
    design_sha = freeze()
    log('== Phase 3216 g5g16_standing_ledger_v2 %s =='
        % ('SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources sha-asserted: %s' % sorted(SHA_ANCHOR))
    led = json.load(open(SRC['ledger'], encoding='utf-8'))
    gl15 = json.load(open(SRC['gl15'], encoding='utf-8'))
    gl16 = json.load(open(SRC['gl16'], encoding='utf-8'))
    draft = json.load(open(SRC['draft'], encoding='utf-8'))
    lcorr = json.load(open(SRC['lcorr'], encoding='utf-8'))
    memo = open(SRC['memo'], encoding='utf-8').read()
    assert len(led['measurements']) == 367 and led['ledger_sha256_8'] == 'bb1474e6'
    log('inputs loaded: ledger (367, bb1474e6) + gl15/gl16 + draft + '
        'ledger_corrections + MEMO')

    control = run_audit(led, gl15, gl16, draft, lcorr, memo)
    gate_all = bool(control['recheck_clean'])
    if gate_all:
        verdict = ('standing_ledger_v2_registered (M1 closures 3210-3215 '
                   'x6 on ledger; M2 appendix x5 all open GPU/cross-line '
                   'dependent; M3 N-line draft + APPX-5 corrections '
                   'pending cross-line; M4 水果类/K4 user-confirm pending; '
                   'standing_ledger_v2 sealed; report-only)')
    else:
        verdict = ('standing_ledger_v2_dirty (%d contradictions; '
                   'report-only)' % len(control['contradictions']))
    log('verdict: %s' % verdict)
    led2 = json.load(open(SRC['ledger'], encoding='utf-8'))
    assert len(led2['measurements']) == 367 and led2['ledger_sha256_8'] == 'bb1474e6'
    log('M5: ledger untouched re-assertion OK (n=367, bb1474e6)')

    # build + seal standing_ledger_v2.json (FULL only)
    suffix = 'smoke_' if SMOKE else ''
    v2p = os.path.join(OUTDIR, suffix + 'standing_ledger_v2.json')
    if gate_all and not SMOKE:
        V2['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
        V2['closed'] = [
            {'item': 'FTR-04 glm4 弱特异性归因', 'closed_at': 3202,
             'evidence': '3181 noise_level + 3202 unit-level decomposition; ledger #364'},
            {'item': 'FTR-15 held_out 补强', 'closed_at': 3204,
             'evidence': '3180 entity-LOO + 3204 recheck; ledger #366'},
            {'item': 'GAP-4 机制链', 'closed_at': None,
             'evidence': '3169->3171->3172->3175->3192 chain closed on the books '
                         '(quantified_collapse by design); ledger #367'},
            {'item': '水果类', 'closed_at': None,
             'evidence': '3210 audit: LPF-line 2767/2800/2802 historical theme, '
                         'recommend close (ledger #362)'},
            {'item': 'K4', 'closed_at': None,
             'evidence': '3210 audit: P-K4 FALSE@2810 + double-falsified@2833, '
                         'recommend close (ledger #362)'},
        ]
        V2['open'] = [
            {'item': 'APPX-1..5', 'why': 'GPU or cross-line dependent (M2 table)'},
            {'item': 'N 线补录施加 (N-P3..N-P7)', 'why': 'draft f29da2de sealed, '
                        'needs cross-line confirmation (APPX-4 governance)'},
            {'item': 'APPX-5 跨线账本补丁施加', 'why': 'ledger_corrections_v1 '
                        'beb775bd on disk, not applied'},
            {'item': 'UI 迭代线', 'why': 'v1.6->v1.8 delivered; v1.9 optional'},
        ]
        V2['pending_user_confirmation'] = [
            {'item': 'MEMO 挂账列表删除「水果类、K4」', 'why': 'doc edit needs '
                        'explicit user confirmation (3210 disposition)'},
        ]
        V2['provenance'] = {
            'ledger': 'bb1474e6 (n=367)', 'gl15': '4fb3f37d', 'gl16': '0f6423e0',
            'draft': 'f29da2de', 'lcorr': 'beb775bd', 'memo': 'read-only',
            'closure_ledger_entries': [362, 363, 364, 365, 366, 367],
        }
        blob = json.dumps(V2, ensure_ascii=False, indent=1, sort_keys=True)
        v2_sha = sha8(blob.encode('utf-8'))
        blob = json.dumps(dict(V2, sha8=v2_sha), ensure_ascii=False, indent=1, sort_keys=True)
        with open(v2p, 'w', encoding='utf-8', newline='') as f:
            f.write(blob)
        log('standing_ledger_v2.json written sha8 %s' % v2_sha)

    summary = {
        'phase': 3216, 'name': 'g5g16_standing_ledger_v2',
        'mode': 'SMOKE' if SMOKE else 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True, 'report_only': True,
        'gates': {'M1_closure_sweep': not any(c[0] == 'M1_closure_sweep' for c in control['contradictions']),
                  'M2_appendix': not any(c[0] == 'M2_appendix' for c in control['contradictions']),
                  'M3_crossline': not any(c[0] == 'M3_crossline' for c in control['contradictions']),
                  'M4_list': not any(c[0] == 'M4_list' for c in control['contradictions']),
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
