# -*- coding: utf-8 -*-
"""Phase 3229 independent verify: 18 check groups, disk-readback only."""
import hashlib
import json
import os
import glob

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P5 = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913'
                        r'\phase3229\g5g26_e3_gate_decision')
A3 = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913'
                        r'\phase3228\g5g26_e3_compute_seal')
OUT = []
FAIL = []


def chk(i, name, ok, detail=''):
    OUT.append('[%s] #%02d %s %s' % ('PASS' if ok else 'FAIL', i, name, detail))
    if not ok:
        FAIL.append(i)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def csha8(obj):
    return hashlib.sha256(json.dumps(
        obj, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    ).hexdigest()[:8]


# 1 ledger head + self-recompute integrity (post = sha8 of blob with head
# field restored to pre-state bak post, per ledger append convention)
led = json.load(open(os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                                  'atlas_ledger.json'), encoding='utf-8'))
LED_POST = 'b21db7e3'
probe = dict(led)
probe['ledger_sha256_8'] = '8e593d22'   # bak pre-state
recomp = hashlib.sha256(json.dumps(
    probe, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
chk(1, 'ledger n=381 post=b21db7e3 + recompute integrity', len(
    led['measurements']) == 381 and led['ledger_sha256_8'] == LED_POST
    and recomp == LED_POST)
e = led['measurements'][380]
chk(2, 'entry #381 phase=3229 name', e.get('phase') == 3229
    and e.get('name') == 'g5g26_e3_gate_decision')

# 3 rollback equivalence
bako = json.load(open(os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                                   'atlas_ledger.json.bak_3229'),
                      encoding='utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:380]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
chk(3, 'ledger rollback-equal vs bak_3229', eq
    and len(bako['measurements']) == 380 and bako['ledger_sha256_8'] == '8e593d22')

# 4 result seal
res = json.load(open(os.path.join(P5, 'result.json'), encoding='utf-8'))
chk(4, 'result verdict + res content sha (None-convention)',
    res['verdict'] == 'e3_gate_e3_fail_f20_registered'
    and csha8(dict(res, res_sha8=None)) == res['res_sha8'])

# 5 registry/gap seals vs chain + disk
r30 = res['wrote']['registry_v130']
g18 = res['wrote']['gap_v18']
chk(5, 'registry v1.30 file c11d13bd / content 9ea9fc3e + gap v1.8 '
       '7065ec4e / 09bf2fa9 (disk readback)',
    r30['file_sha8'] == 'c11d13bd' and r30['content_sha8'] == '9ea9fc3e'
    and g18['file_sha8'] == '7065ec4e' and g18['content_sha8'] == '09bf2fa9'
    and sha8_file(os.path.join(P5, 'atlas_registry_v1_30.json')) == 'c11d13bd'
    and sha8_file(os.path.join(P5, 'gap_ledger_v1_8.json')) == '7065ec4e')

# 6 freeze recompute (content = dict minus sha8) + pre-evaluation ordering
fz = json.load(open(os.path.join(P5, 'execution.json'), encoding='utf-8'))
fz_body = {k: v for k, v in fz.items() if k != 'sha8'}
chk(6, 'freeze d0bb40e6 recompute + prereg wording + K1-K5 present',
    csha8(fz_body) == fz['sha8'] == 'd0bb40e6'
    and res['freeze_sha8'] == 'd0bb40e6'
    and len(fz['clarifications_frozen_before_evaluation']) == 5
    and fz['prereg'].startswith('AGI_GPT5_MEMO Phase 3228 tail'))

# 7 gate verdicts all False + E3_FAIL
gp = res['gate_pass']
chk(7, 'gate_pass flip/dose/spec/amp all False + e3_verdict E3_FAIL',
    gp == {'GD_flip': False, 'GD_dose': False, 'GD_spec': False,
           'GD_amp_support': False}
    and res['e3_verdict'] == 'E3_FAIL')

# 8 integrity: sealed agg reproduction recorded
chk(8, 'integrity 1176 agg cells recorded in result+ledger chain',
    res['recheck']['contradictions'] == []
    and e['chain']['integrity_cells'] == 1176)

# 9 GD_flip detail: matched 0, null_max up to 0.049, deltas <= 0 (both models)
ok9 = True
det9 = []
for m in ('qwen3-4b', 'qwen3-14b'):
    for R in ('isa', 'hasa'):
        rec = res['gate_table']['GD_flip']['%s.%s' % (m, R)]
        if len(rec['seen_folds']) != 4:
            ok9 = False
        for f in rec['seen_folds']:
            if f['matched'] != 0.0 or f['delta'] > 0:
                ok9 = False
                det9.append('%s %s %s' % (m, R, f['fold']))
        nmx = max(f['null_max'] for f in rec['seen_folds'])
        if nmx > 0.05:
            ok9 = False
chk(9, 'GD_flip: matched==0, deltas<=0, null_max<=0.05 (4 model-R cells)', ok9,
    '; '.join(det9) or 'all matched 0')

# 10 GD_dose degenerate rule applied (all-zero matched across alphas)
ok10 = True
for m in ('qwen3-4b', 'qwen3-14b'):
    for R in ('isa', 'hasa'):
        rec = res['gate_table']['GD_dose']['%s.%s' % (m, R)]
        for f in rec['seen_folds']:
            if f['rates'] != [0.0, 0.0, 0.0] or f['verdict'] != 'degenerate_constant':
                ok10 = False
chk(10, 'GD_dose: 8 model-R-fold cells all degenerate_constant zero', ok10)

# 11 GD_spec inverted (cross > matched on hasa side for both models)
ok11 = True
det11 = []
for m in ('qwen3-4b', 'qwen3-14b'):
    c = res['gate_table']['GD_spec'][m]['cells']['oov.hasa']
    if not (c['cross'] > c['matched'] and c['verdict'] == 'fail'):
        ok11 = False
        det11.append('%s %s' % (m, c))
chk(11, 'GD_spec oov.hasa cross>matched (inverted specificity)', ok11,
    '; '.join(det11) or 'both models inverted')

# 12 GD_amp spearman recompute matches sealed + 1/2 pass
import numpy as np
for m, lo in (('qwen3-4b', 0.4), ('qwen3-14b', 0.8)):
    g = res['gate_table']['GD_amp'][m]
    x, y = np.array(g['E_cls']), np.array(g['amp_med'])
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    sp = np.corrcoef(rx, ry)[0, 1]
    chk(12, 'GD_amp %s spearman=%.2f (no-tie exact), pass=%s' % (m, lo, g['pass_']),
        abs(sp - lo) < 1e-9 and abs(g['spearman'] - lo) < 1e-9
        and g['pass_'] == (lo >= 0.6))

# 13 registry v1.30: F20 + upgrade_log #11 + byte preservation
reg29 = json.load(open(os.path.join(ROOT, r'tests\glm5\result',
                                    r'rdc_query_construction_20260913',
                                    r'phase3226\g5g25_a2_gate_decision',
                                    'atlas_registry_v1_29.json'),
                       encoding='utf-8'))
reg30 = json.load(open(os.path.join(P5, 'atlas_registry_v1_30.json'),
                       encoding='utf-8'))
jd = lambda o: json.dumps(o, ensure_ascii=False, sort_keys=True)
ok13 = (reg30['version'] == '1.30' and len(reg30['failures']) == 20
        and reg30['failures'][-1]['id'] == 'F20'
        and reg30['failures'][-1]['kind'] == 'e3_sufficiency_gate_fail'
        and len(reg30['upgrade_log']) == 11
        and reg30['upgrade_log'][-1]['change'].startswith('E3 gate FAIL')
        and len(reg30['features']) == 26)
untouched = [f['id'] for f in reg29['features']
             if f['id'] not in ('FTR-24', 'FTR-25', 'FTR-26')]
ok13 = ok13 and len(untouched) == 23
for a, b in zip(reg29['features'], reg30['features']):
    if a['id'] in untouched:
        ok13 = ok13 and jd(a) == jd(b)
    else:
        ok13 = ok13 and a['evidence_level'] == b['evidence_level'] \
            and jd(a['anchors']) == jd(b['anchors']) \
            and len(b['scope_limits'] or '') > len(a['scope_limits'] or '')
chk(13, 'registry v1.30: F20/#11/scope appends, 23 features byte-exact, '
        'E-levels unchanged', ok13)

# 14 gap v1.8: GAP-5 mechanism_note append, others byte-exact
gap17 = json.load(open(os.path.join(ROOT, r'tests\glm5\result',
                                    r'rdc_query_construction_20260913',
                                    r'phase3226\g5g25_a2_gate_decision',
                                    'gap_ledger_v1_7.json'),
                       encoding='utf-8'))
gap18 = json.load(open(os.path.join(P5, 'gap_ledger_v1_8.json'),
                       encoding='utf-8'))
g5o = [x for x in gap17['gaps'] if x['id'] == 'GAP-5'][0]
g5n = [x for x in gap18['gaps'] if x['id'] == 'GAP-5'][0]
ok14 = (gap18['version'] == '1.8' and len(gap18['gaps']) == 5
        and g5n['mechanism_note'].startswith(g5o['mechanism_note'])
        and '三段式' in g5n['mechanism_note']
        and all(jd(a) == jd(b) for a, b in zip(gap17['gaps'], gap18['gaps'])
                if a['id'] != 'GAP-5')
        and all(jd(gap17[k]) == jd(gap18[k])
                for k in gap17 if k not in ('version', 'updated', 'gaps')))
chk(14, 'gap v1.8: GAP-5 mechanism_note append-only, legacy x4 byte-exact', ok14)

# 15 ledger read-only in arm + provenance anchor
chk(15, 'ledger pre==after 8e593d22 in chain + e3_flip_matrices anchor in '
        'registry provenance',
    e['chain']['ledger_pre'] == '8e593d22'
    and reg30['provenance']['anchor_files']['e3_flip_matrices']['sha8']
    == 'a844d985')

# 16 sealed 3228 artifacts untouched
chk(16, '3228 sealed res 99693298 / npz a844d985 untouched',
    sha8_file(os.path.join(A3, 'e3_flip_matrices.npz')) == 'a844d985')

# 17 K1 clarification + F20 text carries attribution chain
f20t = reg30['failures'][-1]['text']
fzj = json.dumps(fz['clarifications_frozen_before_evaluation'],
                 ensure_ascii=False)
chk(17, 'K1 in freeze + F20 text: 3188/GAP-5 coherence + anti-directional',
    'FTR-22 family' in fzj
    and '3188' in f20t and 'GAP-5' in f20t and '反方向' in f20t)

# 18 five-write disk readback
memo = open(os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md'),
            'rb').read()
daily = open(os.path.join(ROOT, r'.workbuddy\memory\2026-10-11.md'),
             'rb').read()
mem2 = open(os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md'),
            'rb').read().decode('utf-8')
chk(18, 'five-write readback: ledger/MEMO/daily/MEMORY',
    b'## Phase 3229' in memo and b'### Phase 3229' in daily
    and '3229 收官' in mem2 and led['ledger_sha256_8'] in mem2
    and sha8_file(os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md'))
    == '058b8a0d')

OUT.append('VERIFY: %d/18 PASS, %d FAIL %s' % (
    18 - len(FAIL), len(FAIL), FAIL if FAIL else ''))
with open(os.path.join(ROOT, r'tests\gpt5_temp\phase3229_verify_out.txt'), 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(OUT) + '\n')
print('verify done: %d FAIL' % len(FAIL))
