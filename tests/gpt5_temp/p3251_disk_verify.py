# -*- coding: utf-8 -*-
"""Phase 3251 independent disk verification (fresh eyes, no shared state with
the closeout): re-read everything from disk and recompute all anchors."""
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


# 1. sealed result exists, full mode, correct verdict/hashes
RD = os.path.join(SRC_DIR, r'phase3251\g5h14_ustar_lexical_attribution')
R = json.load(open(os.path.join(RD, 'result.json'), encoding='utf-8'))
check('1a result.json', R['phase'] == 3251 and R['mode'] == 'full'
      and R['verdict'] == 'ustar_lexical_partial')
check('1b hashes', R['design_sha8'] == '48169586' and R['res_sha8'] == '0601eb1b',
      '%s/%s' % (R['design_sha8'], R['res_sha8']))
check('1c gate_all False', R['gate_all'] is False)

# 2. nine bit-level drifts == 0.0 (recomputed from the sealed per_model)
PM = R['per_model']
n_zero = 0
for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
    pm = PM[m]
    if pm['drift_3169'] == 0.0 and pm['drift_3189'] == 0.0 and pm['drift_3190'] == 0.0:
        n_zero += 1
check('2 nine drifts 0.0', n_zero == 3, '%d/3 models' % n_zero)

# 3. res sha8 recompute from the payload (excluding res_sha8 key)
res2 = dict(R)
declared = res2.pop('res_sha8')
blob = json.dumps(res2, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
check('3 res_sha8 recompute', sha8(blob) == declared,
      '%s vs %s' % (sha8(blob), declared))

# 4. npz on disk, cosine spectra match the json
import numpy as np
z = np.load(os.path.join(RD, 'ustar_lex.npz'))
ok_npz = True
for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
    arr = z['%s_cos_lex' % m]
    if not np.allclose(arr, np.array(PM[m]['cos_lex']), atol=0, rtol=0):
        ok_npz = False
check('4 npz cos_lex bitwise', ok_npz)

# 5. smoke artifacts present and marked smoke
sm = json.load(open(os.path.join(RD, 'result_smoke.json'), encoding='utf-8'))
check('5 smoke artifact', sm['mode'] == 'smoke' and sm['design_sha8'] == '48169586')

# 6. ledger #403 chain
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
led = json.load(open(LED, encoding='utf-8'))
check('6a ledger n=403', len(led['measurements']) == 403)
e = led['measurements'][-1]
check('6b entry phase 3251', e.get('phase') == 3251)
chk = dict(led)
chk['ledger_sha256_8'] = 'a79a1e4f'
rec = sha8(json.dumps(chk, ensure_ascii=False, indent=1).encode('utf-8'))
check('6c chain post recompute', rec == led['ledger_sha256_8'],
      '%s vs %s' % (rec, led['ledger_sha256_8']))
bak = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json.bak_3251')
check('6d backup + pre-file sha', os.path.exists(bak)
      and sha8(open(bak, 'rb').read()) == e['detail'].split('Pre-append ledger file sha8 ')[-1].rstrip('."'))

# 7. MEMO section + doc arm present exactly once
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
mb = open(MEMO, 'rb').read()
check('7a memo 3251 once', mb.count(b'## Phase 3251:') == 1)
check('7b doc-arm closure summary', 'G5 保真收线小结（3239–3250'.encode('utf-8') in mb)
check('7c gap appendix candidate', 'gap_v1_12_appendix_fidelity_closure'.encode('utf-8') in mb)

# 8. dual wlogs
for p, tag in ((r'research\gpt5\docs\WORKLOG_gpt5.md', 'ws'),
               (r'research\deepseek\docs\WORKLOG_gpt5_mirror.md', 'u')):
    txt = open(os.path.join(ROOT, p), encoding='utf-8').read()
    check('8 wlog %s' % tag, '3251 u* 类级词汇归因' in txt)

# 9. registry untouched + MEMORY synced
REG = os.path.join(SRC_DIR, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json')
check('9a registry byte sha8', sha8_file(REG) == 'a30215fd', sha8_file(REG))
mem = open(os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md'), encoding='utf-8').read()
check('9b memory synced', '3251 收官（u* 词汇归因' in mem and len(mem) <= 3000,
      '%d chars' % len(mem))

# 10. execution.json freeze marker
exe = json.load(open(os.path.join(RD, 'execution.json'), encoding='utf-8'))
check('10 freeze before observation', exe['frozen_before'] == 'any observation'
      and exe['design_sha8'] == '48169586')

print('VERIFY 3251: %d OK / %d FAIL' % (len(OK), len(FAIL)))
for line in OK:
    print('  OK', line)
for line in FAIL:
    print('  FAIL', line)
if FAIL:
    raise SystemExit(1)
print('ALL OK')
