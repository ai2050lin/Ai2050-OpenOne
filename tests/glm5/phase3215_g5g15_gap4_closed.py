# -*- coding: utf-8 -*-
"""Phase 3215 (G5-G15): GAP-4 closed-loop confirmation arm (report-only, zero
GPU).

Per prereg in MEMO Phase 3214 tail G5-G15 (frozen before any observation;
default arm (a)): re-read the GAP-4 attribution chain. Probe finding
(pre-freeze): GAP-4 is ALREADY closed on the books — mechanism chain
complete in gap ledger v1.6 mechanism_note (3169 quantified collapse ->
3171 encoding_missing negated -> 3172 port partial recovery ->
3175 residual port-intrinsic -> 3191 held_out -> 3192 amplitude carrier)
with status quantified_collapse and closure text on file. This arm runs as
a closure confirmation:
  K1 - sealed products on disk + key scalars bitwise vs gap-ledger
       evidence: 3169 per-model ratio_B (2.8291123613715263 / 2.12575372
       3276223 / 2.7142111478689683, all > 2); 3172 pooled curve (2.5388
       114997805062 / 2.031459140015479 / 1.9429485491548533 / 1.8766676
       450829491 / 1.800176217907229, k=0 bitwise == 3169 caliber); 3175
       delta 0.06603206261005456 (|delta| <= 0.15 port_residual_dominant);
       3192 amp_ratio 10.598382949829102 / 7.110019207000732 / 8.890898704
       528809 (>= 1.3 x3).
  K2 - gap ledger v1.6 GAP-4: status quantified_collapse; mechanism_note
       six-stage chain phrases; anchor_sha8 six keys (q03/p3151/p3169/
       p3171/p3172/p3175).
  K3 - MEMO verbatim anchors (3169/3175/3192 segs) + 3172 verdict
       borderline_partial_recovery.
  K4 - ledger untouched (n=366, f30394dc before==after).
  Verdict (report-only): gap4_closed_loop_confirmed (mechanism chain
  closed on the books; GAP-4 remains quantified_collapse by design —
  collapse quantified + mechanism localized, full recovery not achieved
  and not pending).
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
OUTDIR = os.path.join(SRC_DIR, 'phase3215', 'g5g15_gap4_closed')
SMOKE = os.environ.get('P3215_SMOKE') == '1'
LOG = []

SRC = {
    'r3169': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\result.json'),
    'r3172': os.path.join(SRC_DIR, r'phase3172\g5a9_port_calibration\result.json'),
    'r3175': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\result.json'),
    'r3192': os.path.join(SRC_DIR, r'phase3192\g5c5_gap4_amplitude_attribution\result.json'),
    'gl16': os.path.join(SRC_DIR, r'phase3192\g5c5_gap4_amplitude_attribution\gap_ledger_v1_6.json'),
    'ledger': os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'),
    'memo': os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md'),
}
SHA_ANCHOR = {'r3169': '5b51c2c1', 'r3172': None, 'r3175': None, 'r3192': None,
              'gl16': '0f6423e0'}
RES_SEAL = {'r3169': ('49430a39', '018c6024'), 'r3172': ('463f42c8', 'cdb85525'),
            'r3175': ('cb76a177', '9e1a2f08'), 'r3192': ('6221fe00', '58c4f183')}

DESIGN = {
    'phase': '3215', 'name': 'g5g15_gap4_closed',
    'prereg': 'AGI_GPT5_MEMO Phase 3214 tail G5-G15 (frozen before any '
              'observation; default arm (a) GAP-4 chain re-read; probe '
              'found the mechanism chain already closed on the books => '
              'runs as closure-confirmation arm)',
    'data': 'sealed 3169/3172/3175/3192 products (read-only) + gap ledger '
            'v1.6 (read-only) + MEMO (read-only); zero GPU, report-only',
    'gate': 'registered iff K1 sealed scalars bitwise vs ledger evidence + '
            'K2 GAP-4 status/mechanism_note/anchors + K3 MEMO anchors + K4 '
            'ledger untouched; disposition: mechanism chain closed on the '
            'books, GAP-4 remains quantified_collapse by design',
    'sources': {'r3169': '5b51c2c1 (file bytes; res 49430a39/seal 018c6024)',
                'r3172': 'res 463f42c8/seal cdb85525',
                'r3175': 'res cb76a177/seal 9e1a2f08',
                'r3192': 'res 6221fe00/seal 58c4f183',
                'gl16': '0f6423e0', 'memo': 'read-only'},
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
        json.dump({'phase': 3215, 'name': 'g5g15_gap4_closed',
                   'design_sha8': dsha[:8], 'design': DESIGN,
                   'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def run_audit(r3169, r3172, r3175, r3192, gl16, memo, led):
    out = {'contradictions': [], 'dispositions': {}}
    g4 = [g for g in gl16['gaps'] if g['id'] == 'GAP-4'][0]

    # K1: sealed scalars bitwise vs gap-ledger evidence
    k1_bad = []
    for key, (res8, seal8) in RES_SEAL.items():
        r = {'r3169': r3169, 'r3172': r3172, 'r3175': r3175, 'r3192': r3192}[key]
        if r['res_sha8'] != res8 or r['seal_sha8'] != seal8:
            k1_bad.append((key, r['res_sha8'], r['seal_sha8']))
    ratios = {'qwen3-4b': r3169['per_model']['qwen3-4b']['B']['ratio'],
              'qwen3-14b': r3169['per_model']['qwen3-14b']['B']['ratio'],
              'glm4-9b': r3169['per_model']['glm4-9b']['B']['ratio']}
    if ratios != {'qwen3-4b': 2.8291123613715263, 'qwen3-14b': 2.125753723276223,
                  'glm4-9b': 2.7142111478689683}:
        k1_bad.append(('3169 per-model ratios', ratios))
    if not all(v > 2 for v in ratios.values()):
        k1_bad.append(('3169 ratio>2 gate',))
    curve = {int(k): v['ratio'] for k, v in r3172['pooled'].items()}
    if curve != {0: 2.5388114997805062, 1: 2.031459140015479, 2: 1.9429485491548533,
                 4: 1.8766676450829491, 8: 1.800176217907229}:
        k1_bad.append(('3172 pooled curve', curve))
    if curve[8] >= 2.0 or curve[8] < 1.5:
        k1_bad.append(('3172 k8 band [1.5,2)', curve[8]))
    d = r3175['delta_gate']['delta']
    if d != 0.06603206261005456 or abs(d) > 0.15:
        k1_bad.append(('3175 delta', d))
    amps = {'qwen3-4b': r3192['per_model']['qwen3-4b']['amp_ratio'],
            'qwen3-14b': r3192['per_model']['qwen3-14b']['amp_ratio'],
            'glm4-9b': r3192['per_model']['glm4-9b']['amp_ratio']}
    if amps != {'qwen3-4b': 10.598382949829102, 'qwen3-14b': 7.110019207000732,
                'glm4-9b': 8.890898704528809}:
        k1_bad.append(('3192 amp_ratio', amps))
    if not all(v >= 1.3 for v in amps.values()):
        k1_bad.append(('3192 amp>=1.3 gate',))
    if k1_bad:
        out['contradictions'].append(('K1_scalars', k1_bad[:5]))
    log('K1: sealed 3169/3172/3175/3192 scalars bitwise vs ledger evidence: %s'
        % ('OK' if not k1_bad else 'FAIL %s' % k1_bad[:3]))

    # K2: gap ledger GAP-4 status + mechanism chain + anchors
    k2_bad = []
    if g4['status'] != 'quantified_collapse':
        k2_bad.append(('status', g4['status']))
    mn = g4['mechanism_note']
    for frag in ['3171 机制定位', '判决 mixed_across_models',
                 '3172 端口校准干预', 'borderline_partial_recovery',
                 '3175 交叉臂定稿', 'port_residual_dominant',
                 '机制链闭环：3169 量化 -> 3171 encoding_missing 否定 -> '
                 '3172 端口部分恢复 -> 3175 残留=端口固有',
                 '3192 G5-C5 幅度归因', 'u*_amplitude_carrier']:
        if frag not in mn:
            k2_bad.append(('mechanism_note', frag[:30]))
    if set(g4['anchor_sha8']) != {'q03', 'p3151', 'p3169', 'p3171', 'p3172', 'p3175'}:
        k2_bad.append(('anchor_sha8 keys', sorted(g4['anchor_sha8'])))
    if g4['anchor_sha8'].get('p3169') != '5b51c2c1':
        k2_bad.append(('anchor p3169 sha', g4['anchor_sha8'].get('p3169')))
    if k2_bad:
        out['contradictions'].append(('K2_gap4', k2_bad[:5]))
    log('K2: gap v1.6 GAP-4 status quantified_collapse + six-stage chain + '
        'anchors: %s' % ('OK' if not k2_bad else 'FAIL %s' % k2_bad[:3]))

    # K3: MEMO verbatim anchors
    k3_bad = []
    for ph, kws in [(3169, ['ratio_B=2.5388', 'collapse_confirmed']),
                    (3175, ['port_residual_dominant', '0.0660']),
                    (3192, ['10.5984', 'u*'])]:
        i = memo.find('## Phase %d' % ph)
        j = memo.find('\n## ', i + 1) if i >= 0 else -1
        if i < 0:
            k3_bad.append((ph, 'header missing'))
            continue
        seg = memo[i:j]
        for kw in kws:
            if kw not in seg:
                k3_bad.append((ph, kw))
    if k3_bad:
        out['contradictions'].append(('K3_memo', k3_bad[:5]))
    log('K3: MEMO verbatim anchors (3169/3175/3192): %s'
        % ('OK' if not k3_bad else 'FAIL %s' % k3_bad[:3]))

    out['dispositions']['GAP-4'] = {
        'provenance': '3169 quantified collapse (pooled ratio_B 2.5388) + '
                      '3171 encoding_missing negated + 3172 port calibration '
                      '(29.1% recovery, borderline_partial_recovery) + 3175 '
                      'cross-arm (delta +0.0660, port_residual_dominant) + '
                      '3192 amplitude carrier (u*)',
        'status': 'mechanism_chain_closed_on_the_books',
        'conclusion': 'collapse quantified (2.54x) + mechanism localized '
                      '(one-hot port ~29% + port-intrinsic residual ~1.8x, '
                      'u* amplitude carrier); GAP-4 stays quantified_collapse '
                      'by design — full recovery not achieved and not pending',
        'recommendation': 'no registry/gap action needed (closure text '
                          'already in gap v1.6 mechanism_note); drop from '
                          'candidate follow-up list',
    }
    out['recheck_clean'] = not out['contradictions']
    log('RECHECK: %s (%d contradictions)'
        % ('CLEAN' if out['recheck_clean'] else 'DIRTY', len(out['contradictions'])))
    return out


def main():
    design_sha = freeze()
    log('== Phase 3215 g5g15_gap4_closed %s ==' % ('SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        if v is None:
            continue
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sealed sources sha-asserted: r3169 5b51c2c1, gl16 0f6423e0 '
        '(3172/3175/3192 locked via res/seal fields)')
    r3169 = json.load(open(SRC['r3169'], encoding='utf-8'))
    r3172 = json.load(open(SRC['r3172'], encoding='utf-8'))
    r3175 = json.load(open(SRC['r3175'], encoding='utf-8'))
    r3192 = json.load(open(SRC['r3192'], encoding='utf-8'))
    gl16 = json.load(open(SRC['gl16'], encoding='utf-8'))
    led = json.load(open(SRC['ledger'], encoding='utf-8'))
    memo = open(SRC['memo'], encoding='utf-8').read()
    assert len(gl16['gaps']) == 4
    assert len(led['measurements']) == 366 and led['ledger_sha256_8'] == 'f30394dc'
    log('inputs loaded: 3169/3172/3175/3192 sealed + gap v1.6 (4) + ledger '
        '(366, f30394dc) + MEMO')

    control = run_audit(r3169, r3172, r3175, r3192, gl16, memo, led)
    gate_all = bool(control['recheck_clean'])
    if gate_all:
        verdict = ('gap4_closed_loop_confirmed (K1 sealed scalars bitwise — '
                   '3169 ratios 2.8291/2.1258/2.7142 all>2, 3172 pooled '
                   'curve k0 2.5388 -> k8 1.8002, 3175 delta +0.0660, 3192 '
                   'amp 10.5984/7.1100/8.8909; K2 gap v1.6 quantified_'
                   'collapse + six-stage mechanism chain + anchors; K3 MEMO '
                   'anchors; disposition: mechanism chain closed on the '
                   'books, no registry/gap action, report-only)')
    else:
        verdict = ('gap4_closed_dirty (%d contradictions; report-only)'
                   % len(control['contradictions']))
    log('verdict: %s' % verdict)
    led2 = json.load(open(SRC['ledger'], encoding='utf-8'))
    assert len(led2['measurements']) == 366 and led2['ledger_sha256_8'] == 'f30394dc'
    log('K4: ledger untouched re-assertion OK (n=366, f30394dc)')

    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3215, 'name': 'g5g15_gap4_closed',
        'mode': 'SMOKE' if SMOKE else 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True, 'report_only': True,
        'gates': {'K1_scalars': not any(c[0] == 'K1_scalars' for c in control['contradictions']),
                  'K2_gap4': not any(c[0] == 'K2_gap4' for c in control['contradictions']),
                  'K3_memo': not any(c[0] == 'K3_memo' for c in control['contradictions']),
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
