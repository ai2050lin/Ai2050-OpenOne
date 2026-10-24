# -*- coding: utf-8 -*-
"""Phase 3252 independent disk verification (fresh eyes)."""
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
RD = os.path.join(SRC_DIR, r'phase3252\g5h15_ustar_layer_trace')
R = json.load(open(os.path.join(RD, 'result.json'), encoding='utf-8'))
check('1a result.json', R['phase'] == 3252 and R['mode'] == 'full'
      and R['verdict'] == 'ustar_layer_registered')
check('1b hashes', R['design_sha8'] == 'fa13399b' and R['res_sha8'] == 'a4435eb3',
      '%s/%s' % (R['design_sha8'], R['res_sha8']))
check('1c gate_all', R['gate_all'] is True)

# 2. nine drifts + embedding rank-1 + full curves
PM = R['per_model']
n_ok = 0
for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
    pm = PM[m]
    if (pm['drift_3169'] == 0.0 and pm['drift_3189'] == 0.0
            and pm['drift_3190'] == 0.0 and len(pm['top1_curve']) == pm['NH']
            and abs(float(list(pm['top1_curve'].values())[0]) - 1.0) < 1e-9
            and pm['dom_ok'] and pm['stab_ok']):
        n_ok += 1
check('2 drifts+rank1+full curves', n_ok == 3, '%d/3' % n_ok)

# 3. res sha recompute (writer-side keys were INT layer ids serialized in
# numeric order; after a JSON round-trip they are strings, so a sort_keys
# recompute would reorder "10" < "2" -- recompute in file/insertion order,
# which reproduces the writer-side blob for this payload)
res2 = dict(R)
declared = res2.pop('res_sha8')
check('3 res_sha8 recompute (file order)',
      sha8(json.dumps(res2, ensure_ascii=False, indent=1).encode('utf-8')) == declared)

# 4. npz
import numpy as np
z = np.load(os.path.join(RD, 'layer_trace.npz'))
ok = all(abs(z['%s_top1' % m][-1] - list(PM[m]['top1_curve'].values())[-1]) == 0
         for m in PM)
check('4 npz layer_trace', ok and set(z.files) == {'%s_top1' % m for m in PM})

# 5. smoke artifact
sm = json.load(open(os.path.join(RD, 'result_smoke.json'), encoding='utf-8'))
check('5 smoke artifact', sm['mode'] == 'smoke')

# 6. ledger #404 chain + backup
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
led = json.load(open(LED, encoding='utf-8'))
check('6a ledger n=404', len(led['measurements']) == 404)
e = led['measurements'][-1]
check('6b entry 3252', e.get('phase') == 3252)
chk = dict(led)
chk['ledger_sha256_8'] = 'be50a5c5'
rec = sha8(json.dumps(chk, ensure_ascii=False, indent=1).encode('utf-8'))
check('6c chain post recompute', rec == led['ledger_sha256_8'],
      '%s' % led['ledger_sha256_8'])
bak = LED + '.bak_3252'
check('6d backup pre-file sha', os.path.exists(bak)
      and sha8(open(bak, 'rb').read())
      == e['detail'].split('Pre-append ledger file sha8 ')[-1].rstrip('."'))

# 7. MEMO + APPX-6 draft
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
mb = open(MEMO, 'rb').read()
check('7a memo 3252 once', mb.count(b'## Phase 3252:') == 1)
check('7b APPX-6 in memo', 'APPX-6'.encode('utf-8') in mb)
DRAFT = json.load(open(os.path.join(RD, 'gap_appendix_draft_v1_12.json'),
                       encoding='utf-8'))
check('7c draft entry + based_on', DRAFT['entry']['id'] == 'APPX-6'
      and DRAFT['based_on_file_sha8'] == '5934631b'
      and DRAFT['entry']['status'] == 'appendix_recorded')
gap = json.load(open(os.path.join(SRC_DIR, r'phase3237\g5g31_registry_landing'
                                  r'\gap_ledger_v1_11.json'), encoding='utf-8'))
check('7d sealed gap v1.11 untouched', len(gap['appendix']) == 5
      and all(a['id'] != 'APPX-6' for a in gap['appendix']))

# 8. dual wlogs
for p, tag in ((r'research\gpt5\docs\WORKLOG_gpt5.md', 'ws'),
               (r'research\deepseek\docs\WORKLOG_gpt5_mirror.md', 'u')):
    txt = open(os.path.join(ROOT, p), encoding='utf-8').read()
    check('8 wlog %s' % tag, '3252 u* 层轨迹稳定性' in txt)

# 9. registry + MEMORY
REG = os.path.join(SRC_DIR, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json')
check('9a registry byte sha8', sha8_file(REG) == 'a30215fd', sha8_file(REG))
mem = open(os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md'), encoding='utf-8').read()
check('9b memory synced', '3252 收官（u* 层轨迹' in mem and len(mem) <= 3000,
      '%d chars' % len(mem))

# 10. freeze marker
exe = json.load(open(os.path.join(RD, 'execution.json'), encoding='utf-8'))
check('10 freeze before observation', exe['frozen_before'] == 'any observation'
      and exe['design_sha8'] == 'fa13399b')

print('VERIFY 3252: %d OK / %d FAIL' % (len(OK), len(FAIL)))
for line in OK:
    print('  OK', line)
for line in FAIL:
    print('  FAIL', line)
if FAIL:
    raise SystemExit(1)
print('ALL OK')
