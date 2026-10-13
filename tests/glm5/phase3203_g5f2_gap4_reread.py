# -*- coding: utf-8 -*-
"""Phase 3203 (G5-F2): GAP-4 residual mechanism re-read arm (report-only,
zero GPU, sealed 3169/3172/3175/3188-3192 scalars re-read; no registry bump).

Per prereg in MEMO Phase 3202 tail G5-F2 (frozen before any observation;
default arm (a); pre-freeze probe confirmed all 8 sealed results + gap
v1.5/v1.6 + FTR-25 anchors available):
  RE-READ (pinned, 4dp constants probed BEFORE freeze; honest registration):
  G1 quantification: 3169 pooled ratio_B 2.5388 (E_oov_B 0.9479 / E_seen_B
  0.3734); 3172 ratio_kmax 1.8002 (borderline_partial_recovery); 3175 delta
  0.0660, ratio_in_kmax 1.8002, ratio_out_kmax 1.8662
  (port_residual_dominant).
  G2 C-line: 3188 coupled=False / n_models_pass=2 (DECOUPLED);
  3189 shared=True / n=3 + top1 0.3065/0.2936/0.3008 + cross_med
  0.5441/0.5663/0.4449; 3190 robust=True + loo_max_delta
  0.0342/0.0214/0.0332 + ks_seen_ustar 0.9932/0.9761/0.9715;
  3191 predictive=True + min_med_abs_cos 0.3906/0.4174/0.3717;
  3192 carrier=True + amp_ratio 10.5984/7.1100/8.8909 + ratio_B_anchor
  bitwise == 3169 ratio_B.
  G3 cross-consistency: FTR-25 anchors in sealed registry v1.25 bitwise
  match sealed 3189/3190/3191 scalars; gap v1.6 mechanism_note carries
  u*_amplitude_carrier + amp markers.
  G4 proposal (report-only): gap ledger v1.6 UNCHANGED (authoritative);
  proposal text only in result/MEMO/ledger.
  GATE: REGISTERED iff G1-G3 all clean (0 contradictions); else dirty,
  contradictions registered, no proposal.
  SMOKE (env P3203_SMOKE=1): checks only, result prefixed smoke_.
"""
import datetime
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

ROOT = r'D:/AI2050/Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g5f2_gap4_reread'
OUTDIR = os.path.join(SRC_DIR, 'phase3203', NAME)
SMOKE = os.environ.get('P3203_SMOKE') == '1'
LOG = []
MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']

SRC = {
    'p3169': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\result.json'),
    'p3172': os.path.join(SRC_DIR, r'phase3172\g5a9_port_calibration\result.json'),
    'p3175': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\result.json'),
    'p3188': os.path.join(SRC_DIR, r'phase3188\g5c1_gap4_dv_attribution\result.json'),
    'p3189': os.path.join(SRC_DIR, r'phase3189\g5c2_residual_structure\result.json'),
    'p3190': os.path.join(SRC_DIR, r'phase3190\g5c3_shared_direction_robustness\result.json'),
    'p3191': os.path.join(SRC_DIR, r'phase3191\g5c4_ftr25_heldout_upgrade\result.json'),
    'p3192': os.path.join(SRC_DIR, r'phase3192\g5c5_gap4_amplitude_attribution\result.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'gap_v16': os.path.join(SRC_DIR, r'phase3192\g5c5_gap4_amplitude_attribution\gap_ledger_v1_6.json'),
    'registry_v125': os.path.join(SRC_DIR, r'phase3202\g5f1_ftr04_recheck\atlas_registry_v1_25.json'),
}
SHA_ANCHOR = {
    'p3169': '5b51c2c1', 'p3172': 'd8ddc481', 'p3175': 'e0d2d459',
    'p3188': '2dd53368', 'p3189': '5bd804d0', 'p3190': '6a8893ae',
    'p3191': 'a218ead4', 'p3192': '5d8d4858',
    'gap_v15': '4fb3f37d', 'gap_v16': '0f6423e0', 'registry_v125': '3bb33a24',
}
CONTENT = {
    'p3169': ('49430a39', '018c6024'), 'p3172': ('463f42c8', 'cdb85525'),
    'p3175': ('cb76a177', '9e1a2f08'), 'p3188': ('2d4f0184', '09d7d7c4'),
    'p3189': ('3a26f541', 'fb8c1f65'), 'p3190': ('24461856', '0c975f5b'),
    'p3191': ('836fe933', 'a66c0b3d'), 'p3192': ('6221fe00', '58c4f183'),
}

# pinned 4dp scalars (probed BEFORE freeze; measured-anchor discipline)
PIN = {
    'p3169': dict(ratio_B=2.5388, e_oov_B=0.9479, e_seen_B=0.3734),
    'p3172': dict(ratio_kmax=1.8002, main_cls='borderline_partial_recovery'),
    'p3175': dict(delta=0.0660, ratio_in=1.8002, ratio_out=1.8662,
                  main_cls='port_residual_dominant'),
    'p3189': dict(top1={'qwen3-4b': 0.3065, 'qwen3-14b': 0.2936, 'glm4-9b': 0.3008},
                  cross_med={'qwen3-4b': 0.5441, 'qwen3-14b': 0.5663, 'glm4-9b': 0.4449}),
    'p3190': dict(loo={'qwen3-4b': 0.0342, 'qwen3-14b': 0.0214, 'glm4-9b': 0.0332},
                  ks={'qwen3-4b': 0.9932, 'qwen3-14b': 0.9761, 'glm4-9b': 0.9715}),
    'p3191': dict(min_med={'qwen3-4b': 0.3906, 'qwen3-14b': 0.4174, 'glm4-9b': 0.3717}),
    'p3192': dict(amp={'qwen3-4b': 10.5984, 'qwen3-14b': 7.1100, 'glm4-9b': 8.8909}),
}

DESIGN = {
    'phase': '3203', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3202 tail G5-F2 (frozen before any observation; '
              'default arm (a) GAP-4 residual mechanism re-read, report-only; '
              'pre-freeze probe confirmed 8 sealed results + gap v1.6 + FTR-25 '
              'anchors)',
    'data': 'sealed p3169/p3172/p3175/p3188/p3189/p3190/p3191/p3192 results + gap '
            'ledger v1.5 (4fb3f37d) + v1.6 (0f6423e0) + sealed registry v1.25 '
            '(3bb33a24); zero GPU',
    'control_anchor': 'content double locks x8 (res/seal fields) + 4dp scalars '
                      'probed BEFORE freeze + FTR-25 anchor bitwise consistency + '
                      'gap v1.6 mechanism_note text markers',
    'measurements': 'G1 quantification chain (ratio_B / ratio_kmax / delta / '
                    'ratio_in / ratio_out); G2 C-line five arms (coupled/shared/'
                    'robust/predictive/carrier + per-model scalars); G3 '
                    'cross-consistency (FTR-25 anchors bitwise + mechanism_note '
                    'markers); G4 proposal (report-only, gap ledger unchanged)',
    'gate': 'REGISTERED iff G1-G3 clean (0 contradictions) => ledger #355 + MEMO '
            'proposal, registry v1.25 UNCHANGED (report-only); else dirty, '
            'contradictions registered, no proposal',
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
        json.dump({'phase': 3203, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def d4(x):
    return round(float(x), 4)


def run_reread(sealed, reg):
    """Pinned re-read G1-G3 over the GAP-4 chain."""
    out = {'contradictions': []}

    def bad(tag, info):
        out['contradictions'].append((tag, info))

    # G1 quantification chain
    g = sealed['p3169']['gate']
    if d4(g['ratio_B']) != PIN['p3169']['ratio_B'] \
            or d4(g['pooled_E_oov_B']) != PIN['p3169']['e_oov_B'] \
            or d4(g['pooled_E_seen_B']) != PIN['p3169']['e_seen_B']:
        bad('G1_3169_ratio_B', dict(ratio_B=d4(g['ratio_B'])))
    if d4(sealed['p3172']['overall']['ratio_kmax']) != PIN['p3172']['ratio_kmax'] \
            or sealed['p3172']['overall']['main_cls'] != PIN['p3172']['main_cls']:
        bad('G1_3172', d4(sealed['p3172']['overall']['ratio_kmax']))
    dg = sealed['p3175']['delta_gate']
    if d4(dg['delta']) != PIN['p3175']['delta'] \
            or d4(dg['ratio_in_kmax']) != PIN['p3175']['ratio_in'] \
            or d4(dg['ratio_out_kmax']) != PIN['p3175']['ratio_out'] \
            or dg['main_cls'] != PIN['p3175']['main_cls']:
        bad('G1_3175', dict(delta=d4(dg['delta'])))
    log('G1 quantification: ratio_B %.4f -> kmax %.4f (delta %.4f, in %.4f / out %.4f): %s'
        % (d4(g['ratio_B']), PIN['p3172']['ratio_kmax'], d4(dg['delta']),
           d4(dg['ratio_in_kmax']), d4(dg['ratio_out_kmax']),
           'OK' if not any(c[0].startswith('G1') for c in out['contradictions']) else 'FAIL'))

    # G2 C-line five arms
    g88 = sealed['p3188']['gates']
    if g88['coupled'] is not False or g88['n_models_pass'] != 2:
        bad('G2_3188', dict(coupled=g88['coupled'], n=g88['n_models_pass']))
    g89 = sealed['p3189']['gates']
    pm89 = sealed['p3189']['per_model']
    if g89['shared'] is not True or g89['n_models_pass'] != 3:
        bad('G2_3189_gates', dict(shared=g89['shared'], n=g89['n_models_pass']))
    for m in MODELS:
        if d4(pm89[m]['top1']) != PIN['p3189']['top1'][m] \
                or d4(pm89[m]['cross_med']) != PIN['p3189']['cross_med'][m]:
            bad('G2_3189_scalars', m)
    g90 = sealed['p3190']['gates']
    pm90 = sealed['p3190']['per_model']
    if g90['robust'] is not True or g90['n_models_pass'] != 3:
        bad('G2_3190_gates', dict(robust=g90['robust']))
    for m in MODELS:
        if d4(pm90[m]['loo_max_delta']) != PIN['p3190']['loo'][m] \
                or d4(pm90[m]['ks_seen_ustar']) != PIN['p3190']['ks'][m]:
            bad('G2_3190_scalars', m)
    g91 = sealed['p3191']['gates']
    pm91 = sealed['p3191']['per_model']
    if g91['predictive'] is not True or g91['n_models_pass'] != 3:
        bad('G2_3191_gates', dict(predictive=g91['predictive']))
    for m in MODELS:
        if d4(pm91[m]['min_med_abs_cos']) != PIN['p3191']['min_med'][m]:
            bad('G2_3191_scalars', m)
    g92 = sealed['p3192']['gates']
    pm92 = sealed['p3192']['per_model']
    if g92['carrier'] is not True or g92['n_models_pass'] != 3:
        bad('G2_3192_gates', dict(carrier=g92['carrier']))
    for m in MODELS:
        if d4(pm92[m]['amp_ratio']) != PIN['p3192']['amp'][m]:
            bad('G2_3192_scalars', m)
    if sealed['p3192']['gates']['ratio_B_anchor'] != sealed['p3169']['gate']['ratio_B']:
        bad('G2_3192_ratioB_anchor', 'bitwise mismatch vs 3169')
    log('G2 C-line: 3188 DECOUPLED(2/3) -> 3189 SHARED(3/3) -> 3190 LOO robust(3/3) '
        '-> 3191 held-out predictive(3/3) -> 3192 amplitude carrier(3/3): %s'
        % ('OK' if not any(c[0].startswith('G2') for c in out['contradictions']) else 'FAIL'))

    # G3 cross-consistency: FTR-25 anchors vs sealed scalars (bitwise)
    f25 = [f for f in reg['features'] if f['id'] == 'FTR-25'][0]
    A = {a['src']: a['asserts'] for a in f25['anchors']}
    ok89 = all(A['p3189']['top1.%s' % m] == pm89[m]['top1'] for m in MODELS) \
        and A['p3189']['res_sha8'] == CONTENT['p3189'][0]
    ok90 = all(A['p3190']['loo_max_delta.%s' % m] == pm90[m]['loo_max_delta'] for m in MODELS) \
        and A['p3190']['res_sha8'] == CONTENT['p3190'][0]
    ok91 = all(A['p3191']['min_med_abs_cos.%s' % m] == pm91[m]['min_med_abs_cos'] for m in MODELS) \
        and A['p3191']['res_sha8'] == CONTENT['p3191'][0]
    if not (ok89 and ok90 and ok91):
        bad('G3_ftr25_anchor_bitwise', dict(a89=ok89, a90=ok90, a91=ok91))
    mech = [g for g in json.load(open(SRC['gap_v16'], encoding='utf-8'))['gaps']
            if g['id'] == 'GAP-4'][0]['mechanism_note']
    mech_ok = ('u*_amplitude_carrier' in mech and '10.5984' in mech
               and '7.1100' in mech and '8.8909' in mech
               and 'ratio_B 2.5388' in mech)
    if not mech_ok:
        bad('G3_mechanism_note_markers', False)
    log('G3 cross-consistency: FTR-25 anchors bitwise (89/90/91: %s/%s/%s) + '
        'gap v1.6 mechanism_note markers: %s'
        % (ok89, ok90, ok91, 'OK' if mech_ok else 'FAIL'))

    out['recheck_clean'] = not out['contradictions']
    log('RE-READ: %s (%d contradictions)'
        % ('CLEAN' if out['recheck_clean'] else 'DIRTY', len(out['contradictions'])))
    return out


def main():
    design_sha = freeze()
    log('== Phase 3203 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    sealed = {}
    for k in CONTENT:
        r = json.load(open(SRC[k], encoding='utf-8'))
        assert r['res_sha8'] == CONTENT[k][0] and r['seal_sha8'] == CONTENT[k][1], \
            ('content lock', k)
        sealed[k] = r
    log('sealed results double-locked x8 (content res/seal fields)')
    reg = json.load(open(SRC['registry_v125'], encoding='utf-8'))
    assert reg['version'] == '1.25' and len(reg['features']) == 26
    log('sealed registry v1.25 loaded')

    control = run_reread(sealed, reg)
    gate_all = bool(control['recheck_clean'])
    if gate_all:
        proposal = ('GAP-4 状态复读 CLEAN：量化链（ratio_B 2.5388 -> 端口校准 1.8002 '
                    'borderline_partial_recovery -> 残留 1.8662 port_residual_dominant, '
                    'delta 0.0660）与 C 线五连（DECOUPLED 2/3 -> SHARED 3/3 -> LOO '
                    'robust 3/3 -> held-out predictive 3/3 -> u*_amplitude_carrier '
                    '3/3）封存标量全数复现、FTR-25 锚逐位一致、gap v1.6 mechanism_note '
                    '标记齐备。提案（report-only）：gap ledger v1.6 维持权威定稿不动，'
                    'GAP-4 状态保持 quantified_collapse_closed（机制链 3169->3175 + '
                    '3188->3192 双链闭环复读成立）；后续若需 v1.7 仅在新增证据时触发。')
        verdict = ('gap4_reread_clean (G1-G3 0 contradictions; proposal: gap '
                   'ledger v1.6 authoritative unchanged; report-only, registry '
                   'v1.25 untouched)')
    else:
        proposal = None
        verdict = ('gap4_reread_dirty (%d contradictions; no proposal, report-only)'
                   % len(control['contradictions']))
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3203, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'gates': {'reread_clean': bool(gate_all),
                  'n_contradictions': len(control['contradictions'])},
        'recheck': control,
        'proposal': proposal,
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
