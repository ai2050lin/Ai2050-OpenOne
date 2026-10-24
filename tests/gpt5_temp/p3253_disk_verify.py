# -*- coding: utf-8 -*-
"""Phase 3253 independent disk verification (fresh eyes)."""
import hashlib
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
OK = []
FAIL = []


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def sha8_file(p):
    return sha8(open(p, 'rb').read())


def check(name, cond, detail=''):
    (OK if cond else FAIL).append('%s %s' % (name, detail))


# 1. sealed result
RD = os.path.join(SRC_DIR, r'phase3253\g5h16_ustar_subspace')
R = json.load(open(os.path.join(RD, 'result.json'), encoding='utf-8'))
check('1a result.json', R['phase'] == 3253 and R['mode'] == 'full'
      and R['verdict'] == 'ustar_subspace_registered')
check('1b hashes', R['design_sha8'] == '8b10eba4' and R['res_sha8'] == 'e19e2844',
      '%s/%s' % (R['design_sha8'], R['res_sha8']))
check('1c gate_all', R['gate_all'] is True)

# 2. twelve drifts + gate + k_mid
PM = R['per_model']
EXP = {'qwen3-4b': 23, 'qwen3-14b': 28, 'glm4-9b': 15}
n_ok = 0
for m in PM:
    pm = PM[m]
    if (pm['drift_3169'] == 0.0 and pm['drift_3189'] == 0.0
            and pm['drift_3190'] == 0.0 and pm['drift_3252'] == 0.0
            and pm['k_mid'] == EXP[m] and pm['cont_ok'] and pm['aff_ok']
            and pm['gate_model'] and len(pm['spectra']['mid_out']) == 8
            and pm['sigma1']['mid_out'] >= 0.3
            and pm['sigma1']['mid_out'] > pm['sigma1']['mid_emb']):
        n_ok += 1
check('2 drifts x12 + gate 3/3 + k_mid', n_ok == 3, '%d/3' % n_ok)

# 3. res sha recompute (file/insertion order; writer-side int layer keys)
res2 = dict(R)
declared = res2.pop('res_sha8')
check('3 res_sha8 recompute (file order)',
      sha8(json.dumps(res2, ensure_ascii=False, indent=1).encode('utf-8')) == declared)

# 4. npz
import numpy as np
z = np.load(os.path.join(RD, 'subspace.npz'))
need = set()
for m in PM:
    for suf in ('ov_out', 'ov_emb', 'spec_mid_out', 'spec_mid_emb'):
        need.add('%s_%s' % (m, suf))
ok = set(z.files) == need
if ok:
    for m in PM:
        pm = PM[m]
        ov = z['%s_ov_out' % m]
        if abs(float(ov[pm['kout']]) - pm['ov_out'][str(pm['kout'])]) > 1e-9:
            ok = False
check('4 npz subspace', ok, ','.join(sorted(z.files)))

# 5. smoke artifact
sm = json.load(open(os.path.join(RD, 'result_smoke.json'), encoding='utf-8'))
check('5 smoke artifact', sm['mode'] == 'smoke' and sm['design_sha8'] == '8b10eba4')

# 6. ledger #405 chain + backup
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
led = json.load(open(LED, encoding='utf-8'))
check('6a ledger n=405', len(led['measurements']) == 405)
e = led['measurements'][-1]
check('6b entry 3253', e.get('phase') == 3253
      and e.get('verdict', '').startswith('ustar_subspace_registered'))
chk = dict(led)
chk['ledger_sha256_8'] = 'fdca5c5c'
rec = sha8(json.dumps(chk, ensure_ascii=False, indent=1).encode('utf-8'))
check('6c chain post recompute', rec == led['ledger_sha256_8'],
      '%s' % led['ledger_sha256_8'])
bak = LED + '.bak_3253'
tail = e['detail'].split('Pre-append ledger file sha8 ')[-1].rstrip('."')
check('6d backup pre-file sha', os.path.exists(bak) and sha8(open(bak, 'rb').read()) == tail,
      '%s' % tail)

# 7. MEMO + append preview
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
mb = open(MEMO, 'rb').read()
check('7a memo 3253 once', mb.count(b'## Phase 3253:') == 1)
check('7b append trigger in memo', 'append_trigger'.encode('utf-8') in mb
      or 'MET'.encode('utf-8') in mb)
PV = json.load(open(os.path.join(RD, 'registry_append_preview_v1_34.json'),
                    encoding='utf-8'))
check('7c preview report_only + based_on', PV['report_only'] is True
      and PV['based_on']['registry'] == 'v1.33/a30215fd'
      and PV['based_on']['gap_ledger'] == 'v1.11/5934631b'
      and PV['trigger_status'] == 'append_trigger_MET')
gap = json.load(open(os.path.join(SRC_DIR, r'phase3237\g5g31_registry_landing'
                                  r'\gap_ledger_v1_11.json'), encoding='utf-8'))
check('7d sealed gap v1.11 untouched', len(gap['appendix']) == 5
      and all(a['id'] != 'APPX-6' for a in gap['appendix']))

# 8. dual wlogs
for p, tag in ((r'research\gpt5\docs\WORKLOG_gpt5.md', 'ws'),
               (r'research\deepseek\docs\WORKLOG_gpt5_mirror.md', 'u')):
    txt = open(os.path.join(ROOT, p), encoding='utf-8').read()
    check('8 wlog %s' % tag, '3253 u* 子空间机制臂' in txt)

# 9. registry + MEMORY
REG = os.path.join(SRC_DIR, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json')
check('9a registry byte sha8', sha8_file(REG) == 'a30215fd', sha8_file(REG))
mem = open(os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md'), encoding='utf-8').read()
check('9b memory synced', '3253 收官（u* 子空间机制臂' in mem and len(mem) <= 3000,
      '%d chars' % len(mem))

# 10. freeze marker
exe = json.load(open(os.path.join(RD, 'execution.json'), encoding='utf-8'))
check('10 freeze before observation', exe['frozen_before'] == 'any observation'
      and exe['design_sha8'] == '8b10eba4')

print('VERIFY 3253: %d OK / %d FAIL' % (len(OK), len(FAIL)))
for line in OK:
    print('  OK', line)
for line in FAIL:
    print('  FAIL', line)
if FAIL:
    raise SystemExit(1)
print('ALL OK')
