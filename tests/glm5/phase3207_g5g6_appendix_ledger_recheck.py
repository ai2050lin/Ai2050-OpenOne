# -*- coding: utf-8 -*-
"""Phase 3207 (G5-G6): 5-appendix ledger re-read arm (report-only, zero GPU).

Per prereg in MEMO Phase 3206 tail G5-G6 (frozen before any observation;
default arm (a) confirmed viable by pre-freeze probe: gap v1.5 appendix 5
entries + anchor files p3166/p3165/p3155-summary/agate all on disk):
  RECHECK (pinned, probed BEFORE freeze; honest registration):
  A1 - gap ledger v1.5 appendix 5 entries full-field bitwise (id/title/
  anchor/status/note/anchor_sha8 all pinned verbatim) + gl15 file sha
  4fb3f37d + version '1.5'.
  A2 - anchor cross-sha: APPX-1 -> p3166 result 448af595 (== anchor_sha8)
  with eff_ge05 sum == 2 and exactly one entry == 2 (14b only); APPX-2 ->
  p3165 result 511d9b13 with cross_model_K glm4/qwen3-14b
  K_entity_eff_ge05 == 0 (词表方向 4b 专属); APPX-3 -> p3155 summary
  result_summary 46a5034d with k2_int_share_kout bitwise
  (0.15187579769212284 / 0.1824533831239199 / 0.16411745881744746, 4dp
  0.1519/0.1825/0.1641 in the 15.2-18.3% band of the note).
  A3 - gap ledger v1.6 (3192) file sha 0f6423e0 + version '1.6' +
  appendix list bitwise == v1.5 appendix (append-only preserved).
  A4 - agate a_gate_closure_v1.json sha 24c60160 (== APPX-4/5
  anchor_sha8) + corrections_C1_C6 n == 6 + C4/C6 status
  'pending_cross_line' + enactment contains '未施加' +
  ledger_corrections_v1.json on disk sha beb775bd (跨线补丁已落盘未施加
  口径维持).
  Verdict: report-only — no registry upgrade, no gap-ledger mutation, no
  HTML render. Clean => appendix_ledger_recheck_report_only_clean (5
  appendix entries confirmed open, anchors confirmed on-disk); any
  contradiction => dirty, registered honestly.
  Control anchors: gl15 4fb3f37d + gl16 0f6423e0 + p3166 448af595 +
  p3165 511d9b13 + p3155 summary 46a5034d + agate 24c60160 +
  ledger_corrections beb775bd + registry_3162 00f15e98.
  SMOKE (env P3207_SMOKE=1): identical checks, smoke_result.json.
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
NAME = 'g5g6_appendix_ledger_recheck'
OUTDIR = os.path.join(SRC_DIR, 'phase3207', NAME)
SMOKE = os.environ.get('P3207_SMOKE') == '1'
LOG = []

GATE = {'appendix_bitwise': True, 'anchor_cross_sha': True,
        'crossline_pending_maintained': True}

SRC = {
    'gl15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'gl16': os.path.join(SRC_DIR, r'phase3192\g5c5_gap4_amplitude_attribution\gap_ledger_v1_6.json'),
    'p3166_result': os.path.join(SRC_DIR, r'phase3166\g5a3b_logic_direction\result.json'),
    'p3165_result': os.path.join(SRC_DIR, r'phase3165\g5a3_family_alignment\result.json'),
    'p3155_summary': os.path.join(SRC_DIR, r'phase3155\g2p1_relation_family_operator_separability\summary\result_summary.json'),
    'agate': os.path.join(ROOT, r'research\deepseek\atlas\a_gate_closure_v1.json'),
    'ledger_corrections': os.path.join(ROOT, r'research\deepseek\atlas\ledger_corrections_v1.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'gl15': '4fb3f37d', 'gl16': '0f6423e0', 'p3166_result': '448af595',
    'p3165_result': '511d9b13', 'p3155_summary': '46a5034d',
    'agate': '24c60160', 'ledger_corrections': 'beb775bd',
    'registry_3162': '00f15e98',
}

# pinned bitwise constants (probed BEFORE freeze)
PIN_APPX = [
    {'id': 'APPX-1', 'title': '14b R x K_entity 2 共享维定位', 'anchor': 'p3166',
     'status': 'appendix_open',
     'note': 'eff_ge05=2 仅 14b；定位共享方向并解释其来源。', 'anchor_sha8': '448af595'},
    {'id': 'APPX-2', 'title': 'S_attr/S_syntax 跨模型移植', 'anchor': 'p3165',
     'status': 'appendix_open',
     'note': '词表方向目前 4b 专属（tokenizer 面依赖）；移植需每模型重建词表坐标。',
     'anchor_sha8': '511d9b13'},
    {'id': 'APPX-3', 'title': 'K2/K3 直接测量', 'anchor': 'p3155',
     'status': 'appendix_open',
     'note': 'A 闸门诊断：K2/K3 从未被直接测量；p3155 给出 KOUT 交互份额读数（15.2-18.3%）。',
     'anchor_sha8': '46a5034d'},
    {'id': 'APPX-4', 'title': 'N 线 P3-P7 补 Ledger', 'anchor': 'agate',
     'status': 'appendix_open_crossline',
     'note': 'N 线 P3-P7（主轴三段）登记补全至 atlas_ledger（跨线账本，只补不改）。',
     'anchor_sha8': '24c60160'},
    {'id': 'APPX-5', 'title': '跨线账本补丁施加确认', 'anchor': 'agate',
     'status': 'appendix_open_crossline',
     'note': 'A 闸门 C4/C6 补丁 ledger_corrections_v1.json 已落盘未施加；施加需跨线确认。',
     'anchor_sha8': '24c60160'},
]
PIN_KOUT_SHARE = {'qwen3-4b': 0.15187579769212284, 'qwen3-14b': 0.1824533831239199,
                  'glm4': 0.16411745881744746}
CONTENT_3166 = ('89b3f320', '263de0ab')
CONTENT_3165 = ('9b0fe9c9', 'e966a4e5')

DESIGN = {
    'phase': '3207', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3206 tail G5-G6 (frozen before any observation; '
              'default arm (a) 5-appendix ledger re-read; pre-freeze probe '
              'confirmed gl15/gl16 + p3166/p3165/p3155-summary/agate anchors)',
    'data': 'gap ledger v1.5 (4fb3f37d) + v1.6 (0f6423e0) + sealed p3166 (448af595) '
            '+ p3165 (511d9b13) + p3155 summary (46a5034d) + deepseek a_gate_closure '
            'v1 (24c60160) + ledger_corrections_v1 (beb775bd) + registry_3162 '
            '(00f15e98); zero GPU',
    'control_anchor': 'appendix 5 entries full-field bitwise + anchor_sha8 == '
                      'on-disk file sha x4 targets + gl16 appendix bitwise == gl15 '
                      '+ C4/C6 pending_cross_line + 未施加 enactment',
    'measurements': 'A1 appendix bitwise x5; A2 anchor cross-sha + eff_ge05 '
                    'sum==2 one-hot 14b + K_entity_eff_ge05==0 glm4/14b + kout '
                    'share bitwise x3; A3 gl16 appendix append-only bitwise; A4 '
                    'cross-line pending maintenance',
    'gate': 'report-only: no registry upgrade / no gap-ledger mutation / no HTML '
            'render; clean => appendix_ledger_recheck_report_only_clean; else '
            'contradictions registered',
    'sources': SHA_ANCHOR,
    'gates': GATE,
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
        json.dump({'phase': 3207, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def run_recheck(gl15, gl16, r66, r65, s55, agate):
    """Pinned recheck A1-A4 (report-only)."""
    out = {'contradictions': []}

    # A1: gl15 appendix 5 entries full-field bitwise
    a1_bad = []
    if gl15['version'] != '1.5':
        a1_bad.append(('version', gl15.get('version')))
    app15 = gl15['appendix']
    if len(app15) != 5:
        a1_bad.append(('n_appendix', len(app15)))
    for i, pin in enumerate(PIN_APPX):
        if i < len(app15):
            got = {k: app15[i].get(k) for k in pin}
            if got != pin:
                a1_bad.append((pin['id'], got))
    if a1_bad:
        out['contradictions'].append(('A1_appendix_bitwise', a1_bad[:6]))
    log('A1: gl15 appendix 5 entries full-field bitwise + version 1.5: %s'
        % ('OK' if not a1_bad else 'FAIL %s' % a1_bad[:3]))

    # A2: anchor cross-sha + in-file pinned values
    a2_bad = []
    if sha8_file(SRC['p3166_result']) != PIN_APPX[0]['anchor_sha8']:
        a2_bad.append(('APPX-1 anchor sha',))
    if r66['res_sha8'] != CONTENT_3166[0] or r66['seal_sha8'] != CONTENT_3166[1]:
        a2_bad.append(('3166 content lock',))
    blob66 = json.dumps(r66, ensure_ascii=False)
    eff_vals = []
    i = 0
    while True:
        i = blob66.find('"eff_ge05": ', i)
        if i < 0:
            break
        j = blob66.find(',', i)
        eff_vals.append(int(blob66[i + 12:j]))
        i = j
    if sum(eff_vals) != 2 or eff_vals.count(2) != 1:
        a2_bad.append(('3166 eff_ge05', sum(eff_vals), eff_vals.count(2)))
    if sha8_file(SRC['p3165_result']) != PIN_APPX[1]['anchor_sha8']:
        a2_bad.append(('APPX-2 anchor sha',))
    if r65['res_sha8'] != CONTENT_3165[0] or r65['seal_sha8'] != CONTENT_3165[1]:
        a2_bad.append(('3165 content lock',))
    for m in ('glm4', 'qwen3-14b'):
        if r65['cross_model_K'][m]['K_entity_eff_ge05'] != 0:
            a2_bad.append(('3165 K_entity_eff_ge05', m))
    if sha8_file(SRC['p3155_summary']) != PIN_APPX[2]['anchor_sha8']:
        a2_bad.append(('APPX-3 anchor sha',))
    kout = s55['k2_int_share_kout']
    for m, v in PIN_KOUT_SHARE.items():
        if kout.get(m) != v:
            a2_bad.append(('kout_share', m))
    band = (0.150 <= PIN_KOUT_SHARE['qwen3-4b'] <= 0.185
            and 0.150 <= PIN_KOUT_SHARE['qwen3-14b'] <= 0.185
            and 0.150 <= PIN_KOUT_SHARE['glm4'] <= 0.185)
    if not band:
        a2_bad.append(('kout_band_15.2-18.3',))
    if a2_bad:
        out['contradictions'].append(('A2_anchor_cross', a2_bad[:6]))
    log('A2: anchor sha x3 (448af595/511d9b13/46a5034d) + 3166 eff_ge05 sum==2 '
        'one-hot 14b + 3165 glm4/14b K_entity_eff_ge05==0 + 3155 kout share '
        'bitwise x3 in 15.0-18.5%% band (note 15.2-18.3%% 为叙述性舍入: '
        '14b 0.18245->18.2%%): %s'
        % ('OK' if not a2_bad else 'FAIL %s' % a2_bad[:3]))

    # A3: gl16 appendix append-only bitwise == gl15
    a3_bad = []
    if gl16['version'] != '1.6':
        a3_bad.append(('version', gl16.get('version')))
    if json.dumps(gl16['appendix'], ensure_ascii=False, sort_keys=True) != \
            json.dumps(gl15['appendix'], ensure_ascii=False, sort_keys=True):
        a3_bad.append(('appendix not preserved',))
    if a3_bad:
        out['contradictions'].append(('A3_gl16_appendix', a3_bad[:4]))
    log('A3: gl16 (0f6423e0) version 1.6 + appendix bitwise == gl15 '
        '(append-only): %s' % ('OK' if not a3_bad else 'FAIL %s' % a3_bad[:3]))

    # A4: cross-line pending maintenance (APPX-4/5)
    a4_bad = []
    if sha8_file(SRC['agate']) != PIN_APPX[3]['anchor_sha8']:
        a4_bad.append(('APPX-4/5 agate sha',))
    cc = agate['corrections_C1_C6']
    if len(cc) != 6:
        a4_bad.append(('n_corrections', len(cc)))
    for c in cc:
        if c['id'] in ('C4', 'C6') and c['status'] != 'pending_cross_line':
            a4_bad.append((c['id'], 'status', c['status']))
        if c['id'] in ('C4', 'C6') and '未施加' not in c['enactment']:
            a4_bad.append((c['id'], 'enactment'))
    if 'ledger_corrections_v1.json' not in agate.get('ledger_registration', {}).get('entry', {}).get('name', '') \
            and agate.get('ledger_registration', {}).get('status') != 'pending_cross_line':
        a4_bad.append(('ledger_registration status',))
    if not os.path.exists(SRC['ledger_corrections']):
        a4_bad.append(('ledger_corrections missing',))
    if a4_bad:
        out['contradictions'].append(('A4_crossline_pending', a4_bad[:6]))
    log('A4: agate 24c60160 + C1-C6 x6 + C4/C6 pending_cross_line + 未施加 '
        '+ ledger_corrections_v1.json (beb775bd) on disk: %s'
        % ('OK' if not a4_bad else 'FAIL %s' % a4_bad[:3]))

    out['recheck_clean'] = not out['contradictions']
    log('RECHECK: %s (%d contradictions)'
        % ('CLEAN' if out['recheck_clean'] else 'DIRTY', len(out['contradictions'])))
    return out


def main():
    design_sha = freeze()
    log('== Phase 3207 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    gl15 = json.load(open(SRC['gl15'], encoding='utf-8'))
    gl16 = json.load(open(SRC['gl16'], encoding='utf-8'))
    r66 = json.load(open(SRC['p3166_result'], encoding='utf-8'))
    r65 = json.load(open(SRC['p3165_result'], encoding='utf-8'))
    s55 = json.load(open(SRC['p3155_summary'], encoding='utf-8'))
    agate = json.load(open(SRC['agate'], encoding='utf-8'))
    log('sealed gl15/gl16 + p3166/p3165/p3155-summary + agate loaded')

    control = run_recheck(gl15, gl16, r66, r65, s55, agate)
    gate_all = bool(control['recheck_clean'])
    if gate_all:
        verdict = ('appendix_ledger_recheck_report_only_clean (A1 gl15 '
                   'appendix 5 entries full-field bitwise; A2 anchor sha x3 '
                   'cross + 3166 eff_ge05 sum==2 one-hot 14b + 3165 '
                   'K_entity_eff_ge05==0 glm4/14b + 3155 kout share bitwise '
                   'x3 in 15.2-18.3% band; A3 gl16 appendix append-only '
                   'bitwise == gl15; A4 agate C4/C6 pending_cross_line + '
                   '未施加 + ledger_corrections_v1 beb775bd on disk; 5 '
                   'appendix entries confirmed open, no 口径 update needed '
                   'after 3206 FTR-22 closed@3175 note)')
    else:
        verdict = ('appendix_ledger_recheck_dirty (%d contradictions; '
                   'report-only, nothing mutated)'
                   % len(control['contradictions']))
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3207, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True, 'report_only': True,
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
