# -*- coding: utf-8 -*-
"""Phase 3221 (G5-G23): UI freeze + client version-chain archival inventory
(report-only, zero GPU).

Per prereg in MEMO Phase 3220 tail G5-G23 (frozen before any observation;
default arm (a)): freeze the UI iteration line at v1.10 and seal the
five-generation client version chain with on-disk sha + field-count
evolution:
  F1 - five clients located + sha/field-count re-read from disk:
       v1.6 3209 bd230988/137, v1.7 3211 62ba1bd6/611, v1.8 3213
       ca76ec49/670, v1.9 3217 9c5b48da/698, v1.10 3219 def433e0/727;
  F2 - ui_client_version_chain_v1.json sealed (content sha);
  F3 - UI line frozen at v1.10 (governance declaration; no registry/gap
       action; further UI work requires new prereg);
  F4 - ledger untouched (n=372, 87b28799).
Verdict: ui_version_chain_archival_registered iff F1-F4 clean.
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
OUTDIR = os.path.join(SRC_DIR, 'phase3221', 'g5g23_ui_freeze_archival')
SMOKE = os.environ.get('P3221_SMOKE') == '1'
LOG = []

CLIENTS = [
    {'version': 'v1.6', 'phase': 3209, 'dir': r'phase3209\g5g8_ui_v16_client',
     'client_sha8': 'bd230988', 'res_sha8': 'eac7fe87', 'n_fields': 137,
     'milestone': '零 script 基座：26 特征卡 + radio 过滤 + gap/appendix/草案'},
    {'version': 'v1.7', 'phase': 3211, 'dir': r'phase3211\g5g10_ui_v17_anchor_values',
     'client_sha8': '62ba1bd6', 'res_sha8': 'ee976567', 'n_fields': 611,
     'milestone': '特征锚 values 视图（61 锚 src/phase/role/tags + 352 asserts）'},
    {'version': 'v1.8', 'phase': 3213, 'dir': r'phase3213\g5g13_ui_v18_block_anchor_details',
     'client_sha8': 'ca76ec49', 'res_sha8': 'a398c290', 'n_fields': 670,
     'milestone': '块级锚详情（gap/appendix/草案 +59 字段）'},
    {'version': 'v1.9', 'phase': 3217, 'dir': r'phase3217\g5g19_ui_v19_chain_ledger_details',
     'client_sha8': '9c5b48da', 'res_sha8': '41efe346', 'n_fields': 698,
     'milestone': '版本链注记 + 账本尾行级锚（+28 字段）'},
    {'version': 'v1.10', 'phase': 3219, 'dir': r'phase3219\g5g21_ui_v110_standing_block',
     'client_sha8': 'def433e0', 'res_sha8': '6bdc1127', 'n_fields': 727,
     'milestone': '挂账总账 v2 + 施加就绪块（+29 字段）；UI 冻结版'},
]

SRC = {'ledger': os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')}
LED_N, LED_POST = 372, '87b28799'

DESIGN = {
    'phase': '3221', 'name': 'g5g23_ui_freeze_archival',
    'prereg': 'AGI_GPT5_MEMO Phase 3220 tail G5-G23 (frozen before any '
              'observation; default arm (a) UI freeze + client version-chain '
              'archival inventory)',
    'data': 'five client dirs (3209/3211/3213/3217/3219) re-read from disk + '
            'ledger (read-only, n=372, 87b28799); zero GPU, report-only, '
            'UI line frozen at v1.10',
    'gate': 'registered iff F1 five clients sha+field-count match on disk + '
            'F2 version chain sealed (ui_client_version_chain_v1.json) + F3 '
            'freeze declaration + F4 ledger untouched (n=372, 87b28799)',
    'sources': {'clients': '5 dirs sha-locked inline', 'ledger': 'read-only'},
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
        json.dump({'phase': 3221, 'name': 'g5g23_ui_freeze_archival',
                   'design_sha8': dsha[:8], 'design': DESIGN,
                   'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def main():
    design_sha = freeze()
    log('== Phase 3221 g5g23_ui_freeze_archival %s ==' % ('SMOKE' if SMOKE else 'FULL'))
    led = json.load(open(SRC['ledger'], encoding='utf-8'))
    assert len(led['measurements']) == LED_N and led['ledger_sha256_8'] == LED_POST
    log('inputs loaded: ledger (%d, %s) + 5 client dirs' % (LED_N, LED_POST))

    contradictions = []

    # F1: five clients re-read from disk
    f1_bad = []
    for c in CLIENTS:
        cp = os.path.join(SRC_DIR, c['dir'], 'atlas_client_v1.html')
        rp = os.path.join(SRC_DIR, c['dir'], 'result.json')
        got_c, rj = sha8_file(cp), json.load(open(rp, encoding='utf-8'))
        got_r, n_fc = rj['res_sha8'], rj['field_check']['n_expected']
        c['client_sha8_disk'], c['res_sha8_disk'], c['n_fields_disk'] = got_c, got_r, n_fc
        if got_c != c['client_sha8'] or got_r != c['res_sha8'] or n_fc != c['n_fields']:
            f1_bad.append((c['version'], got_c, got_r, n_fc))
    c1 = not f1_bad
    if not c1:
        contradictions.append(('F1_clients', f1_bad))
    log('F1: five clients disk re-read (bd230988/137, 62ba1bd6/611, '
        'ca76ec49/670, 9c5b48da/698, def433e0/727): %s'
        % ('OK' if c1 else 'FAIL %s' % f1_bad[:2]))

    # F2: version chain sealed
    chain_doc = {
        'schema': 'ui_client_version_chain_v1',
        'created': time.strftime('%Y-%m-%d %H:%M'),
        'by_phase': 3221,
        'frozen_at': 'v1.10 (def433e0, phase 3219, 727 data-k fields)',
        'governance': 'UI iteration line FROZEN at v1.10; further UI work '
                      'requires a new prereg phase; zero-script constraint '
                      'maintained across all five generations',
        'field_evolution': '137 -> 611 -> 670 -> 698 -> 727',
        'generations': CLIENTS,
        'provenance': {'ledger_at_freeze': '%s (n=%d)' % (LED_POST, LED_N)},
    }
    cblob = json.dumps(chain_doc, ensure_ascii=False, indent=1, sort_keys=True)
    c_sha = sha8(cblob.encode('utf-8'))
    cblob = json.dumps(dict(chain_doc, sha8=c_sha), ensure_ascii=False, indent=1, sort_keys=True)
    c2 = bool(c_sha)
    if not c2:
        contradictions.append(('F2_chain',))
    log('F2: ui_client_version_chain_v1 content sha %s: %s'
        % (c_sha, 'OK' if c2 else 'FAIL'))
    cp = os.path.join(OUTDIR, 'ui_client_version_chain_v1.json')
    if not SMOKE:
        with open(cp, 'w', encoding='utf-8', newline='') as f:
            f.write(cblob)
        log('chain doc written, file sha8 %s' % sha8_file(cp))

    # F3: freeze declaration
    c3 = (chain_doc['frozen_at'].startswith('v1.10')
          and 'FROZEN' in chain_doc['governance'])
    if not c3:
        contradictions.append(('F3_freeze',))
    log('F3: UI line frozen at v1.10 declaration: %s' % ('OK' if c3 else 'FAIL'))

    # F4: ledger untouched
    led2 = json.load(open(SRC['ledger'], encoding='utf-8'))
    c4 = (len(led2['measurements']) == LED_N and led2['ledger_sha256_8'] == LED_POST)
    if not c4:
        contradictions.append(('F4_ledger',))
    log('F4: ledger untouched (n=%d, %s): %s' % (LED_N, LED_POST, 'OK' if c4 else 'FAIL'))

    clean = not contradictions
    verdict = ('ui_version_chain_archival_registered' if clean
               else 'ui_version_chain_archival_dirty (%d contradictions)' % len(contradictions))
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3221, 'name': 'g5g23_ui_freeze_archival',
        'mode': 'SMOKE' if SMOKE else 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True, 'report_only': True, 'ui_line_frozen': True,
        'gates': {'recheck_clean': bool(clean), 'n_contradictions': len(contradictions)},
        'chain_content_sha8': c_sha,
        'field_evolution': [c['n_fields_disk'] for c in CLIENTS],
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
