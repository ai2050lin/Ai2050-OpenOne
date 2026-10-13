# -*- coding: utf-8 -*-
"""Phase 3230 independent verify: 18 check groups, disk-readback basis.

Verify-side only: on FAIL, fix the verify (or its calibers), never the
sealed artifacts."""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
sys.path.insert(0, os.path.join(ROOT, 'tests', 'glm5'))
import phase3194_g5d2_ftr26_heldout_upgrade as p94

P5 = os.path.join(ROOT, 'tests', 'glm5', 'result',
                  'rdc_query_construction_20260913', 'phase3230',
                  'g5g27_relcontrast_efficacy')
RES = []
NFAIL = [0]


def chk(i, desc, ok, extra=''):
    tag = 'PASS' if ok else 'FAIL'
    if not ok:
        NFAIL[0] += 1
    RES.append('#%02d %s %s %s' % (i, tag, desc, extra))


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def csha(o):
    return hashlib.sha256(json.dumps(
        o, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    ).hexdigest()[:8]


MODELS = ['qwen3-4b', 'qwen3-14b']

# 1 ledger head + chaining recompute
LP = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
led = json.load(open(LP, encoding='utf-8'))
stored = led['ledger_sha256_8']
led['ledger_sha256_8'] = 'b21db7e3'
rec = hashlib.sha256(json.dumps(
    led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
chk(1, 'ledger n=382 post=3cba5317 chain-recompute',
    len(led['measurements']) == 382 and stored == '3cba5317' and rec == '3cba5317')

# 2 ledger entry 3230
e = [x for x in led['measurements'] if x.get('phase') == 3230]
chk(2, 'ledger entry 3230 unique, prereg G5-G27, verdict dR_lever_fail',
    len(e) == 1 and e[0]['prereg_id'] == 'G5-G27'
    and 'dR_lever_fail' in e[0]['verdict'] and e[0]['superseded_by'] is None
    and e[0]['line'] == 'G')

# 3 freeze content sha (exclude sha8 key)
fz = json.load(open(os.path.join(P5, 'execution.json'), encoding='utf-8'))
chk(3, 'freeze 1d02dd97 recompute + phase 3230',
    csha({k: v for k, v in fz.items() if k != 'sha8'}) == fz['sha8']
    == '1d02dd97' and fz['phase'] == '3230')

# 4 freeze gates verbatim-pin
g = fz['gates']
chk(4, 'freeze GA_flip (0.15, alpha=2.0) + GB_dir (strict monotone) + verdict rule',
    '0.15' in g['GA_flip'] and 'alpha=2.0' in g['GA_flip']
    and 'strictly increasing' in g['GB_dir'] and 'dR_lever_pass iff GA_flip '
    'and GB_dir' in g['verdict'])

# 5 clarifications K1..K5 (sealed in result.json)
res = json.load(open(os.path.join(P5, 'result.json'), encoding='utf-8'))
cl = res['clarifications']
chk(5, 'clarifications K1..K5 (5 items; K1 NH-1; K4 gates-before-observation)',
    len(cl) == 5 and 'NH-1' in cl[0] and cl[0].startswith('K1')
    and cl[3].startswith('K4') and 'BEFORE any dR number' in cl[3])

# 6 result seal (None convention)
chk(6, 'res 1b63b063 None-convention + verdict + 0 contradictions',
    csha(dict(res, res_sha8=None)) == res['res_sha8'] == '1b63b063'
    and res['verdict'] == 'dR_lever_fail' and res['contradictions'] == [])

# 7 npz seal + keys
npzp = os.path.join(P5, 'e3_relcontrast_flip.npz')
z = np.load(npzp)
need = [m + sfx for m in MODELS
        for sfx in ('_dR_unit', '_dR_raw_unit', '_seen_margins', '_oov_margins',
                    '_seen_J0', '_oov_J0', '_seen_m0', '_oov_m0')]
chk(7, 'npz d430d29e + payload keys x16',
    fsha(npzp) == res['sealed_npz']['file_sha8'] == 'd430d29e'
    and all(k in z.files for k in need))

# 8/9 dR bitwise recompute from sealed 3157 cache
ok8 = ok9 = True
for m in MODELS:
    D = int(json.load(open(os.path.join(
        os.path.expanduser('~'), '.cache', 'huggingface', 'hub',
        'models--Qwen--%s' % ('Qwen3-4B-Base' if m == 'qwen3-4b' else 'Qwen3-14B'),
        'snapshots', sorted(os.listdir(os.path.join(
            os.path.expanduser('~'), '.cache', 'huggingface', 'hub',
            'models--Qwen--%s' % ('Qwen3-4B-Base' if m == 'qwen3-4b' else 'Qwen3-14B'),
            'snapshots')))[0], 'config.json'), encoding='utf-8'))['num_hidden_layers'])
    z57 = np.load(p94.SRC[p94.NPZ57_KEY[m]])
    H = z57['H'].astype(np.float32)
    RL = z57['rel'].astype(int)
    dpost = H[:, D, :][RL == 0].mean(0) - H[:, D, :][RL == 1].mean(0)
    draw = H[:, D - 1, :][RL == 0].mean(0) - H[:, D - 1, :][RL == 1].mean(0)
    dpost = dpost / (np.linalg.norm(dpost) + 1e-18)
    draw = draw / (np.linalg.norm(draw) + 1e-18)
    ok8 &= float(np.abs(dpost - z[m + '_dR_unit']).max()) == 0.0
    ok9 &= float(np.abs(draw - z[m + '_dR_raw_unit']).max()) == 0.0
chk(8, 'dR_post unit bitwise recompute x2', ok8)
chk(9, 'dR_raw unit bitwise recompute x2', ok9)

# 10 RMS scale device gates
ok10 = all(res['models'][m]['device']['rms_scale']['med_fin'] < 5.0
           < res['models'][m]['device']['rms_scale']['med_prev']
           for m in MODELS)
chk(10, 'RMS scale sanity recorded (med_fin<5<med_prev) x2', ok10)

# 11 anchors drift 0
ok11 = all(res['models'][m]['u_R_anchors']['tc_drift'] == 0.0
           and res['models'][m]['u_R_anchors']['dv_drift'] == 0.0
           for m in MODELS)
chk(11, '3157 tc + 3183 dv anchors drift==0 x2', ok11)

# 12 C6v3 device gate
ok12 = True
for m, ref in (('qwen3-4b', 0.9963), ('qwen3-14b', 0.9941)):
    md = res['models'][m]['oov']['marg_diag']
    ok12 &= md['agree_all'] >= 0.98 and abs(md['agree_all'] - ref) < 5e-5
    ok12 &= md['mean_centered_cos'] >= 0.9999
chk(12, 'C6v3 marg agreement >=0.98, == arbitrated 0.9963/0.9941, cos>=0.9999',
    ok12)

# 13 seen baseline == sealed 3228
ok13 = True
for m, d0 in (('qwen3-4b', -3.966), ('qwen3-14b', -4.693)):
    s = res['models'][m]['seen']
    ok13 &= s['n_rows'] == 208 and s['J0_isa_share'] == 0.0
    ok13 &= abs(s['d0_mean'] - d0) < 5e-4
chk(13, 'seen baseline matches sealed 3228 (d0 4dp, J0isa=0, n=208)', ok13)

# 14 GB_dir recompute + mirror symmetry
ok14 = True
for m in MODELS:
    for panel in ('seen', 'oov'):
        d = res['models'][m][panel]['dose']
        v = (d['dR|a0.5'], d['dR|a1.0'], d['dR|a2.0'])
        ok14 &= v[0] < v[1] < v[2] and v[1] > 0
        for a in ('0.5', '1.0', '2.0'):
            ok14 &= abs(d['dR_neg|a%s' % a] + d['dR|a%s' % a]) < 1e-6
chk(14, 'GB_dir strict monotone + positive; dR_neg perfect mirror (6 cells x2)',
    ok14)

# 15 GA recompute from agg
ok15 = True
for m in MODELS:
    agg = res['models'][m]['seen']['agg']
    for f in ('cut0', 'cut1', 'cut2', 'cut3'):
        ok15 &= agg['isa|dR|%s|a2.0' % f]['rate'] == 0.0
        ok15 &= max(vv['rate'] for kk, vv in agg.items()
                    if kk.endswith('|%s|a2.0' % f)
                    and kk.split('|')[1].startswith(('rand_', 'orth_'))) == 0.0
    ok15 &= res['models'][m]['gates']['ga_folds'] == 0
    ok15 &= res['models'][m]['gates']['gb_pass'] is True
chk(15, 'GA: matched+null rates all 0 at a2.0, folds 0/4; gb_pass x2', ok15)

# 16 gates_global + verdict routing + decoupled diag branch
ok16 = (res['gates_global']['GA_flip'] is False
        and res['gates_global']['GB_dir'] is True
        and res['verdict'] == 'dR_lever_fail')
cvisa = [res['models'][m]['dR']['diag']['cos_dR_visa'] for m in MODELS]
ok16 &= all(abs(c) < 0.5 for c in cvisa)
ok16 &= abs(cvisa[0] - (-0.2463)) < 5e-5 and abs(cvisa[1] - (-0.0547)) < 5e-5
chk(16, 'verdict routing (GA F, GB T -> fail) + decoupled diag |cos|<0.5 x2',
    ok16)

# 17 ledger read-only in-arm + no registry writes
chk(17, 'freeze declares ledger read-only n=381; result carries no registry/gap',
    'read-only' in fz['ledger'] and '381' in fz['ledger']
    and not any(k.startswith('registry') or k.startswith('gap')
                for k in res.keys()))

# 18 five-write readback
MP = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
DP = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-11.md')
MEMP = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
memo = open(MP, encoding='utf-8').read()
daily = open(DP, encoding='utf-8').read()
mem = open(MEMP, encoding='utf-8').read()
chk(18, 'five-write readback: MEMO/daily/MEMORY + sha8s',
    '## Phase 3230 —' in memo and 'dR_lever_fail' in memo
    and fsha(MP) == '71dfa8e9'
    and '### Phase 3230' in daily and '3cba5317' in daily
    and fsha(DP) == '9ea9811c'
    and '3230 收官' in mem and '3cba5317' in mem and '3231 已预注册' in mem
    and fsha(MEMP) == 'fc92b200')

with open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'phase3230_verify_out.txt'),
          'w', encoding='utf-8') as f:
    f.write('\n'.join(RES) + '\nVERIFY: %d/18 PASS, %d FAIL\n'
            % (18 - NFAIL[0], NFAIL[0]))
print('VERIFY DONE: %d fail' % NFAIL[0])
