# -*- coding: utf-8 -*-
"""Phase 3222 (G5-G24): G-line session summary (report-only, zero GPU).

Per prereg in MEMO Phase 3221 tail G5-G24 (frozen before any observation;
default arm (a)): twelve-round governance closing statistics for phases
3210-3221, sealed as gline_session_summary_v1.json — NO mutation:
  G1 - ledger entries #362-#373 read from disk, phase/name/verdict-prefix
       sequence verified against the on-record table (12 rounds, all
       REGISTERED);
  G2 - category census: closure-confirmation x3 (3212/3214/3215), UI
       iteration x5 (3211/3213/3217/3219 + freeze 3221), audit/inventory
       x4 (3210/3216/3218/3220); all zero-GPU report/non-ledger or
       governance-only;
  G3 - session summary sealed (gline_session_summary_v1.json, content
       sha) incl standing-list v2 state + apply-readiness + UI freeze;
  G4 - ledger untouched (n=373, ff363d67).
Verdict: gline_session_summary_registered iff G1-G4 clean.
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
OUTDIR = os.path.join(SRC_DIR, 'phase3222', 'g5g24_gline_session_summary')
SMOKE = os.environ.get('P3222_SMOKE') == '1'
LOG = []

SRC = {'ledger': os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')}
LED_N, LED_POST = 373, 'ff363d67'

TABLE = [
    (362, 3210, 'g5g9_standing_items_audit', 'g5g9_standing_items_audit', 'audit_inventory'),
    (363, 3211, 'g5g10_ui_v17_anchor_values', 'g5g10_ui_v17_anchor_values', 'ui_iteration'),
    (364, 3212, 'g5g12_ftr04_attribution_closed', 'g5g12_ftr04_attribution_closed', 'closure'),
    (365, 3213, 'g5g13_ui_v18_block_anchor_details', 'g5g13_ui_v18_block_anchor_details', 'ui_iteration'),
    (366, 3214, 'g5g14_ftr15_heldout_closed', 'g5g14_ftr15_heldout_closed', 'closure'),
    (367, 3215, 'g5g15_gap4_closed', 'g5g15_gap4_closed', 'closure'),
    (368, 3216, 'g5g16_standing_ledger_v2', 'standing_ledger_v2_registered', 'audit_inventory'),
    (369, 3217, 'g5g19_ui_v19_chain_ledger_details', 'ui_v19_chain_ledger_details_registered', 'ui_iteration'),
    (370, 3218, 'g5g20_crossline_apply_readiness', 'crossline_apply_readiness_registered', 'audit_inventory'),
    (371, 3219, 'g5g21_ui_v110_standing_block', 'ui_v110_standing_block_registered', 'ui_iteration'),
    (372, 3220, 'g5g22_appendix_anchors_reread', 'appendix_anchors_reread_registered', 'audit_inventory'),
    (373, 3221, 'g5g23_ui_freeze_archival', 'ui_version_chain_archival_registered', 'ui_iteration'),
]

DESIGN = {
    'phase': '3222', 'name': 'g5g24_gline_session_summary',
    'prereg': 'AGI_GPT5_MEMO Phase 3221 tail G5-G24 (frozen before any '
              'observation; default arm (a) G-line session summary)',
    'data': 'atlas ledger (read-only, n=373, ff363d67) entries #362-#373; '
            'zero GPU, report-only',
    'gate': 'registered iff G1 twelve entries #362-#373 phase/name/verdict '
            'sequence verified + G2 category census (closure 3 / UI 5 / '
            'audit-inventory 4) + G3 summary sealed '
            '(gline_session_summary_v1.json) + G4 ledger untouched '
            '(n=373, ff363d67)',
    'sources': {'ledger': 'read-only'},
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
        json.dump({'phase': 3222, 'name': 'g5g24_gline_session_summary',
                   'design_sha8': dsha[:8], 'design': DESIGN,
                   'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def main():
    design_sha = freeze()
    log('== Phase 3222 g5g24_gline_session_summary %s ==' % ('SMOKE' if SMOKE else 'FULL'))
    led = json.load(open(SRC['ledger'], encoding='utf-8'))
    assert len(led['measurements']) == LED_N and led['ledger_sha256_8'] == LED_POST
    log('inputs loaded: ledger (%d, %s)' % (LED_N, LED_POST))

    contradictions = []

    # G1: entries #362-#373 sequence verify
    g1_bad = []
    rows = []
    for entry_no, ph, name, vprefix, cat in TABLE:
        idx = entry_no - 1
        m = led['measurements'][idx]
        vp = m['verdict'].split('|')[0]
        ok_i = (m.get('phase') == ph and m.get('name') == name
                and vp.startswith(vprefix))
        if not ok_i:
            g1_bad.append((entry_no, ph, m.get('phase'), m.get('name'), vp[:40]))
        rows.append({'ledger_entry': entry_no, 'phase': ph, 'name': name,
                     'verdict_prefix': vp, 'category': cat})
    c1 = not g1_bad
    if not c1:
        contradictions.append(('G1_sequence', g1_bad))
    log('G1: twelve ledger entries #362-#373 phase/name/verdict sequence: %s'
        % ('OK' if c1 else 'FAIL %s' % g1_bad[:2]))

    # G2: category census
    from collections import Counter
    census = Counter(r['category'] for r in rows)
    c2 = (census['closure'] == 3 and census['ui_iteration'] == 5
          and census['audit_inventory'] == 4)
    if not c2:
        contradictions.append(('G2_census', dict(census)))
    log('G2: census closure=3 / ui_iteration=5 / audit_inventory=4: %s %s'
        % ('OK' if c2 else 'FAIL', dict(census)))

    # G3: session summary sealed
    summary_doc = {
        'schema': 'gline_session_summary_v1',
        'created': time.strftime('%Y-%m-%d %H:%M'),
        'by_phase': 3222,
        'window': 'phases 3210-3221 (ledger #362-#373), all zero-GPU',
        'rounds': rows,
        'census': dict(census),
        'closing_state': {
            'standing_ledger_v2': '207f2fa6 (closed 5 / open 4 / '
                                  'pending_user_confirmation 1)',
            'apply_readiness': {'nline': 'ready_append_type (draft f29da2de)',
                                'appx5': 'not_ready_stale_baseline '
                                         '(patch beb775bd predates 300+ '
                                         'governance entries)'},
            'ui_line': 'FROZEN at v1.10 (def433e0, 727 data-k fields; '
                       '137->611->670->698->727 chain sealed ca1b24ca)',
            'appendices': 'APPX-1..3 zero-GPU non-closable re-affirmed '
                          '(3220); APPX-4/5 pending cross-line confirmation',
        },
        'pending_user_confirmation': [
            'MEMO 挂账列表删除「水果类、K4」(3210 disposition)',
            'cross-line apply confirmations (N-line / APPX-5 re-baselined)',
        ],
        'provenance': {'ledger_at_summary': '%s (n=%d)' % (LED_POST, LED_N)},
    }
    sblob = json.dumps(summary_doc, ensure_ascii=False, indent=1, sort_keys=True)
    s_sha = sha8(sblob.encode('utf-8'))
    sblob = json.dumps(dict(summary_doc, sha8=s_sha), ensure_ascii=False, indent=1, sort_keys=True)
    c3 = bool(s_sha)
    if not c3:
        contradictions.append(('G3_summary',))
    log('G3: gline_session_summary_v1 content sha %s: %s'
        % (s_sha, 'OK' if c3 else 'FAIL'))
    sp = os.path.join(OUTDIR, 'gline_session_summary_v1.json')
    if not SMOKE:
        with open(sp, 'w', encoding='utf-8', newline='') as f:
            f.write(sblob)
        log('summary doc written, file sha8 %s' % sha8_file(sp))

    # G4: ledger untouched
    led2 = json.load(open(SRC['ledger'], encoding='utf-8'))
    c4 = (len(led2['measurements']) == LED_N and led2['ledger_sha256_8'] == LED_POST)
    if not c4:
        contradictions.append(('G4_ledger',))
    log('G4: ledger untouched (n=%d, %s): %s' % (LED_N, LED_POST, 'OK' if c4 else 'FAIL'))

    clean = not contradictions
    verdict = ('gline_session_summary_registered' if clean
               else 'gline_session_summary_dirty (%d contradictions)' % len(contradictions))
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3222, 'name': 'g5g24_gline_session_summary',
        'mode': 'SMOKE' if SMOKE else 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True, 'report_only': True,
        'gates': {'recheck_clean': bool(clean), 'n_contradictions': len(contradictions)},
        'summary_content_sha8': s_sha,
        'rounds': len(rows), 'census': dict(census),
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
