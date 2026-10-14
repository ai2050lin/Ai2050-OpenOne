# -*- coding: utf-8 -*-
"""Independent verification of Phase 3231 (G5-G28) sealed artifacts.

READ-ONLY over sealed products. Recomputes everything from primary
sources (sealed 3230 npz/result, registry v1.30/gap v1.8, sha-pinned
W_U weight slices) and asserts consistency with registry v1.31 /
gap v1.9 / result.json. On FAIL: fix THIS script, never the sealed
artifacts (discipline).
"""

import hashlib
import json
import os
import glob

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RD = os.path.join(ROOT, 'tests', 'glm5', 'result',
                  'rdc_query_construction_20260913')
A4DIR = os.path.join(RD, 'phase3230', 'g5g27_relcontrast_efficacy')
OUT31 = os.path.join(RD, 'phase3231', 'g5g28_registry_decision')
REG30 = os.path.join(RD, 'phase3229', 'g5g26_e3_gate_decision',
                     'atlas_registry_v1_30.json')
GAP18 = os.path.join(RD, 'phase3229', 'g5g26_e3_gate_decision',
                     'gap_ledger_v1_8.json')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')

EXP = {'freeze': '45a130ea', 'res': '54e55c7a',
       'reg31_file': 'ef5189e7', 'reg31_content': '4b6648c2',
       'gap19_file': '9fca29bd', 'gap19_content': '83dac915',
       'a4_res': '1b63b063', 'a4_npz': 'd430d29e',
       'reg30_sha': 'c11d13bd', 'gap18_sha': '7065ec4e',
       'led_n': 382, 'led_post': '3cba5317'}
MODELS = ('qwen3-4b', 'qwen3-14b')
ALPHAS = (0.5, 1.0, 2.0)
SMOKE_PREFIXED = [p for p in os.listdir(OUT31) if p.startswith('smoke_')]

G = []


def gate(name, ok, detail=''):
    G.append({'gate': name, 'pass': bool(ok), 'detail': detail})
    print(('PASS ' if ok else 'FAIL ') + name + (' :: ' + detail if detail
                                                 else ''), flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def csha8(obj):
    return hashlib.sha256(json.dumps(
        obj, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    ).hexdigest()[:8]


def jd(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True)


# ---------- V1: file identity ----------
res = json.load(open(os.path.join(OUT31, 'result.json'), encoding='utf-8'))
ok = (res['res_sha8'] == EXP['res'] and res['verdict'] ==
      'f20_refinement_registered' and res['freeze_sha8'] == EXP['freeze'])
gate('V1 result identity (res %s / freeze %s / verdict)' %
     (EXP['res'], EXP['freeze']), ok,
     'res=%s freeze=%s verdict=%s' % (res['res_sha8'], res['freeze_sha8'],
                                      res['verdict']))

p31 = os.path.join(OUT31, 'atlas_registry_v1_31.json')
p19 = os.path.join(OUT31, 'gap_ledger_v1_9.json')
gate('V2 sealed file shas', sha8_file(p31) == EXP['reg31_file']
     and sha8_file(p19) == EXP['gap19_file'],
     'reg31 %s / gap19 %s' % (sha8_file(p31), sha8_file(p19)))

reg31 = json.load(open(p31, encoding='utf-8'))
v31 = dict(reg31)
v31.pop('sha8', None)
gap19 = json.load(open(p19, encoding='utf-8'))
gate('V3 content shas', csha8(v31) == EXP['reg31_content']
     and csha8(gap19) == EXP['gap19_content'],
     'reg31 %s / gap19 %s' % (csha8(v31), csha8(gap19)))

# smoke artifacts present but segregated
gate('V4 smoke segregation',
     all(not f.startswith('atlas_registry_v1_31')
         or f.startswith('smoke_') for f in SMOKE_PREFIXED)
     and res['arm'].startswith('F20 refinement'),
     'smoke files: %d (all prefixed)' % len(SMOKE_PREFIXED))

# ---------- V5: primary-source anchors ----------
a4 = json.load(open(os.path.join(A4DIR, 'result.json'), encoding='utf-8'))
gate('V5 3230 anchors', a4['res_sha8'] == EXP['a4_res']
     and a4['sealed_npz']['file_sha8'] == EXP['a4_npz']
     and a4['verdict'] == 'dR_lever_fail',
     'res %s npz %s verdict %s' % (a4['res_sha8'],
                                   a4['sealed_npz']['file_sha8'],
                                   a4['verdict']))
gate('V6 v1.30/v1.8 source shas unchanged',
     sha8_file(REG30) == EXP['reg30_sha']
     and sha8_file(GAP18) == EXP['gap18_sha'],
     'reg30 %s gap18 %s' % (sha8_file(REG30), sha8_file(GAP18)))

# ---------- V7: independent recompute from sealed npz ----------
NPZP = os.path.join(A4DIR, 'e3_relcontrast_flip.npz')
assert sha8_file(NPZP) == EXP['a4_npz']
z = np.load(NPZP)
npz_ok, notes = True, []
for m in MODELS:
    mg, m0 = z[m + '_seen_margins'], z[m + '_seen_m0']
    d0 = m0[:, 0] - m0[:, 1]
    if abs(float(d0.mean()) - a4['models'][m]['seen']['d0_mean']) >= 1e-9:
        npz_ok = False
        notes.append('d0 drift ' + m)
    for ai, a in enumerate(ALPHAS):
        dd = float(((mg[0, ai, :, 0] - mg[0, ai, :, 1]) - d0).mean())
        if abs(dd - a4['models'][m]['seen']['dose']['dR|a%s' % a]) >= 1e-9:
            npz_ok = False
            notes.append('dD drift %s a%s' % (m, a))
    jud = (mg[0, 1, :, 0] > mg[0, 1, :, 1]).astype(np.int8)
    zs = np.load(os.path.join(RD, 'phase3154',
                              'g1p4_mfd_multifactor_disentangle', m,
                              'collect.npz'))
    ci, ho = zs['ci'].astype(int), zs['ho'].astype(int)
    del zs
    J0 = z[m + '_seen_J0']
    tot = 0
    for c in range(4):
        sl = np.where((ci == c) & (ho == 0))[0]
        for Rv in (0, 1):
            tot += int(((J0[sl] == (1 - Rv)) & (jud[sl] == Rv)).sum())
    if tot != 0:
        npz_ok = False
        notes.append('GA flips nonzero %s=%d' % (m, tot))
    crr = float(z[m + '_dR_unit'] @ z[m + '_dR_raw_unit'])
    if abs(crr - a4['models'][m]['dR']['diag']['cos_dR_dRraw']) >= 1e-6:
        npz_ok = False
        notes.append('cos dR dRraw drift ' + m)
gate('V7 npz recompute (d0/dD x3/GA flips/cos dR dRraw) x2 models',
     npz_ok, '; '.join(notes) or 'bit-exact; GA 0 both')

# ---------- V8: cos(dR,dw) from sha-pinned W_U slices ----------
from safetensors import safe_open
HUB = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
wdw_ok, wdw_notes = True, []
wdw_vals = {}
for m, (p, k) in {
        'qwen3-4b': (os.path.join(ROOT, 'models', 'hf', 'qwen3-4b',
                                  'model-00001-of-00003.safetensors'),
                     'model.embed_tokens.weight'),
        'qwen3-14b': (glob.glob(os.path.join(
            HUB, 'models--Qwen--Qwen3-14B', 'snapshots', '*'))[0]
            + os.sep + 'model-00008-of-00008.safetensors', 'lm_head.weight')
}.items():
    with safe_open(p, framework='pt') as f:
        WT = f.get_slice(k)[[101158, 18830]].float().numpy().astype(np.float32)
    dw = WT[0] - WT[1]
    dwu = dw / np.linalg.norm(dw)
    c = float(z[m + '_dR_unit'] @ dwu)
    wdw_vals[m] = c
    sv = a4['models'][m]['dR']['diag']['cos_dR_wdw']
    if abs(c - sv) >= 1e-6:
        wdw_ok = False
        wdw_notes.append('%s recompute %.6f vs sealed %.6f' % (m, c, sv))
    del WT
gate('V8 cos(dR,dw) independent from W_U slices x2', wdw_ok,
     '; '.join(wdw_notes) or ('%.6f / %.6f' % (wdw_vals['qwen3-4b'],
                                               wdw_vals['qwen3-14b'])))

# ---------- V9: registry v1.31 derivation ----------
reg30 = json.load(open(REG30, encoding='utf-8'))
ok, notes = True, []
if reg31['version'] != '1.31':
    ok = False
    notes.append('version')
if len(reg31['features']) != 26 or len(reg31['upgrade_log']) != 11:
    ok = False
    notes.append('counts')
if jd(reg30['upgrade_log']) != jd(reg31['upgrade_log']):
    ok = False
    notes.append('upgrade_log drift')
if jd(reg30['failures'][:19]) != jd(reg31['failures'][:19]):
    ok = False
    notes.append('failures[:19] drift')
f20o, f20n = reg30['failures'][-1], reg31['failures'][-1]
if f20o['id'] != 'F20':
    ok = False
    notes.append('F20 id')
for k in f20o:
    if jd(f20o[k]) != jd(f20n.get(k)):
        ok = False
        notes.append('F20 field drift ' + k)
if list(f20n.keys()) != list(f20o.keys()) + ['refinements'] or \
        len(f20n['refinements']) != 1:
    ok = False
    notes.append('F20 refinements shape')
touched = {'FTR-24', 'FTR-25', 'FTR-26'}
for a, b in zip(reg30['features'], reg31['features']):
    if a['id'] != b['id']:
        ok = False
        notes.append('feature order ' + a['id'])
    elif a['id'] not in touched:
        if jd(a) != jd(b):
            ok = False
            notes.append('feature drift ' + a['id'])
    else:
        for kk in ('statement', 'evidence_level', 'anchors'):
            if jd(a[kk]) != jd(b[kk]):
                ok = False
                notes.append('%s %s drift' % (a['id'], kk))
        if len(b['scope_limits']) <= len(a.get('scope_limits') or ''):
            ok = False
            notes.append('%s scope not appended' % a['id'])
gate('V9 registry v1.31 byte-preservation + append shape', ok,
     '; '.join(notes) or '23 features byte-exact + 3 scope appends + '
     'F20 pre-existing fields exact')

# F20 refinement text numbers vs sealed values
txt = f20n['refinements'][0]['text']
GB = {m: a4['models'][m]['seen']['dose'] for m in MODELS}
DG = {m: a4['models'][m]['dR']['diag'] for m in MODELS}
NRM = {m: a4['models'][m]['dR']['norm_post'] for m in MODELS}
D0 = {m: a4['models'][m]['seen']['d0_mean'] for m in MODELS}
exp_nums = {
    '|dR| 7-9x norms': (NRM['qwen3-4b'], NRM['qwen3-14b']),
    'dD 4b a2': GB['qwen3-4b']['dR|a2.0'],
    'dD 14b a2': GB['qwen3-14b']['dR|a2.0'],
    'cos dR wdw 4b': DG['qwen3-4b']['cos_dR_wdw'],
    'cos dR wdw 14b': DG['qwen3-14b']['cos_dR_wdw'],
    'd0 4b': abs(D0['qwen3-4b']),
    'd0 14b': abs(D0['qwen3-14b']),
    'cos dR visa 4b': DG['qwen3-4b']['cos_dR_visa'],
    'cos dR visa 14b': DG['qwen3-14b']['cos_dR_visa'],
    'cos dR vhasa 4b': DG['qwen3-4b']['cos_dR_vhasa'],
    'cos dR vhasa 14b': DG['qwen3-14b']['cos_dR_vhasa'],
}
frag_ok = all(s in txt for s in ('16.40', '20.72', '0.0997', '0.0625',
                                 '3.97', '4.69', '-0.2463', '-0.0547',
                                 '0.71', '0.65', '7\u20139'))
gate('V10 F20 refinement text numbers vs sealed', frag_ok,
     '|dR| %.2f/%.2f dD@2 %.2f/%.2f coswdw %.4f/%.4f |d0| %.2f/%.2f'
     % (NRM['qwen3-4b'], NRM['qwen3-14b'],
        GB['qwen3-4b']['dR|a2.0'], GB['qwen3-14b']['dR|a2.0'],
        DG['qwen3-4b']['cos_dR_wdw'], DG['qwen3-14b']['cos_dR_wdw'],
        abs(D0['qwen3-4b']), abs(D0['qwen3-14b'])))

# ---------- V11: gap v1.9 derivation ----------
gap18 = json.load(open(GAP18, encoding='utf-8'))
ok, notes = True, []
if gap19['version'] != '1.9' or len(gap19['gaps']) != 5:
    ok = False
    notes.append('version/count')
for a, b in zip(gap18['gaps'], gap19['gaps']):
    if a['id'] != b['id']:
        ok = False
        notes.append('order ' + a['id'])
    elif a['id'] != 'GAP-5':
        if jd(a) != jd(b):
            ok = False
            notes.append('drift ' + a['id'])
    else:
        for k in a:
            if k != 'mechanism_note' and jd(a[k]) != jd(b[k]):
                ok = False
                notes.append('gap5 field ' + k)
        if len(b['mechanism_note']) <= len(a['mechanism_note']):
            ok = False
            notes.append('gap5 note not appended')
for k in gap18:
    if k not in ('version', 'updated', 'gaps') and jd(gap18[k]) != jd(gap19[k]):
        ok = False
        notes.append('top ' + k)
gate('V11 gap v1.9 byte-preservation + GAP-5 append', ok,
     '; '.join(notes) or 'legacy x4 byte-exact + GAP-5 note appended')

# ---------- V12: ledger untouched ----------
led = json.load(open(LEDGER, encoding='utf-8'))
gate('V12 ledger untouched (n=%d post %s)' % (EXP['led_n'], EXP['led_post']),
     len(led['measurements']) == EXP['led_n']
     and led['ledger_sha256_8'] == EXP['led_post'],
     'observed n=%d post=%s' % (len(led['measurements']),
                                led['ledger_sha256_8']))

n_fail = sum(1 for g in G if not g['pass'])
verdict = 'VERIFY_PASS' if n_fail == 0 else 'VERIFY_FAIL(%d)' % n_fail
print('==', verdict, '==', flush=True)
out = {'phase': 3231, 'verify': G, 'verdict': verdict,
       'wdw_independent': wdw_vals}
with open(os.path.join(OUT31, 'verify.json'), 'w', encoding='utf-8',
          newline='') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
