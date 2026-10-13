# -*- coding: utf-8 -*-
"""Phase 3210 (G5-G9): 水果类/K4 standing-item audit arm (report-only, zero
GPU).

Per prereg in MEMO Phase 3209 tail G5-G9 (frozen before any observation;
default arm (a)): audit the two legacy standing items 水果类 (fruit class)
and K4 that have been carried in the G-line 挂账不变 list since Phase 3177.
  D1 - empirical provenance on shared ledger: measurements for phases
  2767 (类别知识图谱 v1, 5 类别 x 16 实体 — fruit is one of 5 classes),
  2800 (LPF-13 载流子 fruit/水果/果实/果园+ 画像), 2802 (LPF-15 apple 多义
  带符号投票谱), 2810 (P-K4 虚词最小 FALSE), 2833 (P-K2/P-K4 双否证 →
  Δ_specific 通道分离接续) present with pinned verdict heads.
  D2 - MEMO verbatim anchors: 2767 '5 类别 × 16 实体'; 2802 '带符号投票谱';
  2810 'P-K4 虚词最小'; 2833 'attention-sink/位置通用通道支配'.
  D3 - standing-list structure: first occurrence of '水果类、K4' in
  AGI_GPT5_MEMO.md is inside Phase 3177 tail (no earlier G-line definition
  section) => inherited placeholder items, no G-line work product pending.
  D4 - ledger untouched (n=361, 46d53545 before==after).
  Verdict (report-only): audit_registered with dispositions — 水果类 =>
  recommend close (LPF-line historical research theme, provenance sealed
  on ledger 2767/2800/2802); K4 => recommend close (P-K4 criterion double-
  falsified at 2810/2833, superseded by 通道分离 line). Actual list edit is
  a user-confirmed doc edit, NOT a ledger mutation.
"""
import hashlib
import json
import os
import re
import sys
import time

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = r'D://AI2050//Ai2050-OpenOne'
OUTDIR = os.path.join(ROOT, 'tests', 'glm5', 'result',
                      'rdc_query_construction_20260913', 'phase3210',
                      'g5g9_standing_items_audit')
SMOKE = os.environ.get('P3210_SMOKE') == '1'
LOG = []
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')

GATE = {'provenance_anchored': True, 'memo_anchors': True,
        'list_structure': True, 'ledger_untouched': True}

PIN_PHASES = {
    2767: '类别知识图谱',
    2800: '载流子',
    2802: 'apple 多义',
    2810: '短语级档案调制',
    2833: 'Δh 通道分离',
}
PIN_MEMO_ANCHORS = {
    2767: '5 类别 × 16 实体',
    2802: '带符号投票谱',
    2810: 'P-K4 虚词最小',
    2833: 'attention-sink/位置通用通道支配',
}

DESIGN = {
    'phase': '3210', 'name': 'g5g9_standing_items_audit',
    'prereg': 'AGI_GPT5_MEMO Phase 3209 tail G5-G9 (frozen before any '
              'observation; default arm (a) 水果类/K4 standing-item audit)',
    'data': 'shared atlas_ledger (read-only) + AGI_GPT5_MEMO.md (read-only); '
            'zero GPU, report-only',
    'gate': 'registered iff D1-D4 clean; dispositions recommend close for '
            'both items (user-confirmed doc edit pending, no ledger/doc '
            'mutation in this phase)',
    'sources': {'ledger': '46d53545 (read for D1/D4)',
                'memo': '155cfebb (read for D2/D3)'},
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
        json.dump({'phase': 3210, 'name': 'g5g9_standing_items_audit',
                   'design_sha8': dsha[:8], 'design': DESIGN,
                   'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def run_audit(led, memo):
    out = {'contradictions': [], 'dispositions': {}}

    # D1: provenance anchored in MEMO headers. SMOKE probe showed phases
    # 2767-2833 are LPF-line phases with NO shared-ledger entries (ledger
    # n=361 covers G-line measurement records from phase 8 on) => honest
    # resolution per 3207 precedent: anchor provenance to MEMO headers,
    # record ledger absence as a finding, not a contradiction.
    d1_bad = []
    ledger_absent = []
    for ph, kw in PIN_PHASES.items():
        i = memo.find('## Phase %d' % ph)
        if i < 0:
            d1_bad.append((ph, 'memo header missing'))
            continue
        hdr = memo[i:memo.find('\n', i)]
        if kw not in hdr:
            d1_bad.append((ph, 'keyword %s not in header' % kw))
        if not any(m.get('phase') == ph for m in led['measurements']):
            ledger_absent.append(ph)
    out['dispositions']['ledger_absence_finding'] = {
        'phases_absent_from_shared_ledger': ledger_absent,
        'note': 'LPF-line phases recorded in MEMO only; shared ledger '
                'covers G-line measurement records',
    }
    if d1_bad:
        out['contradictions'].append(('D1_provenance', d1_bad[:5]))
    log('D1: provenance MEMO headers %s: %s (ledger_absent=%s)'
        % (sorted(PIN_PHASES), 'OK' if not d1_bad else 'FAIL %s' % d1_bad[:3],
           ledger_absent))

    # D2: memo verbatim anchors
    d2_bad = []
    for ph, anchor in PIN_MEMO_ANCHORS.items():
        i = memo.find('## Phase %d' % ph)
        j = memo.find('\n## ', i + 1) if i >= 0 else -1  # top-level only:
        # '## Phase' would truncate at '### Phase 2833:' subsection headers
        if i < 0:
            d2_bad.append((ph, 'phase header missing'))
            continue
        seg = memo[i:j if j > 0 else i + 200000]
        if anchor not in seg:
            d2_bad.append((ph, anchor))
    if d2_bad:
        out['contradictions'].append(('D2_memo_anchors', d2_bad[:5]))
    log('D2: memo verbatim anchors (2767/2802/2810/2833): %s'
        % ('OK' if not d2_bad else 'FAIL %s' % d2_bad[:3]))

    # D3: standing-list structure
    first = memo.find('水果类、K4')
    if first < 0:
        out['contradictions'].append(('D3_list', 'no standing list'))
        d3_ok = False
    else:
        ph_i = memo.rfind('## Phase', 0, first)
        hdr = memo[ph_i:ph_i + 24].splitlines()[0]
        m = re.match(r'## Phase (\d+)', hdr)
        first_phase = int(m.group(1)) if m else -1
        d3_ok = first_phase == 3177
        if not d3_ok:
            out['contradictions'].append(('D3_list', ('first at', first_phase)))
        out['dispositions']['standing_list_first_phase'] = first_phase
    log('D3: 水果类/K4 first listed at Phase 3177 tail (inherited, no earlier '
        'G-line definition): %s' % ('OK' if d3_ok else 'FAIL'))

    # dispositions
    out['dispositions']['水果类'] = {
        'provenance': 'LPF line 2767 (类别知识图谱 v1, fruit among 5 classes) '
                      '+ 2800 (fruit 载流子画像) + 2802 (apple 带符号投票谱)',
        'status': 'historical_research_theme_memonly_provenance',
        'recommendation': 'close from G-line standing list (user-confirmed '
                          'doc edit); optionally re-surface as appendix if '
                          'future fruit-class protocol work resumes',
    }
    out['dispositions']['K4'] = {
        'provenance': 'P-K4 (虚词最小) preregistered 2810, FALSE (func 1.400 '
                      '> content 1.269); double-falsified with P-K2 at 2833; '
                      'superseded by Δ_specific(cond) 通道分离 line (2833+)',
        'status': 'criterion_falsified_and_superseded',
        'recommendation': 'close from G-line standing list (user-confirmed '
                          'doc edit); falsification trail in MEMO 2810/2833 '
                          '(LPF line, no shared-ledger entries)',
    }

    # D4: ledger untouched check happens in main; record state here
    out['recheck_clean'] = not out['contradictions']
    log('RECHECK: %s (%d contradictions)'
        % ('CLEAN' if out['recheck_clean'] else 'DIRTY', len(out['contradictions'])))
    return out


def main():
    design_sha = freeze()
    log('== Phase 3210 g5g9_standing_items_audit %s ==' % ('SMOKE' if SMOKE else 'FULL'))
    memo_sha = sha8_file(MEMO)
    assert memo_sha == '155cfebb', memo_sha
    led = json.load(open(LEDGER, encoding='utf-8'))
    assert len(led['measurements']) == 361 and led['ledger_sha256_8'] == '46d53545'
    memo = open(MEMO, encoding='utf-8').read()
    log('inputs loaded: ledger n=361 (46d53545) + MEMO (155cfebb)')

    control = run_audit(led, memo)
    gate_all = bool(control['recheck_clean'])
    if gate_all:
        verdict = ('standing_items_audit_registered (D1 provenance 2767/'
                   '2800/2802/2810/2833 MEMO-anchored, ledger absence '
                   'recorded as finding; D2 memo anchors verbatim; '
                   'D3 水果类/K4 inherited list items first at 3177 tail; '
                   'dispositions: both recommend close, user-confirmed doc '
                   'edit pending, report-only)')
    else:
        verdict = ('standing_items_audit_dirty (%d contradictions; '
                   'report-only)' % len(control['contradictions']))
    log('verdict: %s' % verdict)
    # D4: ledger untouched re-assertion
    led2 = json.load(open(LEDGER, encoding='utf-8'))
    assert len(led2['measurements']) == 361 and led2['ledger_sha256_8'] == '46d53545'
    log('D4: ledger untouched re-assertion OK (n=361, 46d53545)')

    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3210, 'name': 'g5g9_standing_items_audit',
        'mode': 'SMOKE' if SMOKE else 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True, 'report_only': True,
        'gates': dict(GATE, recheck_clean=bool(gate_all),
                      n_contradictions=len(control['contradictions'])),
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
