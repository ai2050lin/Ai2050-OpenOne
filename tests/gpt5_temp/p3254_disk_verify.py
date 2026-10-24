# -*- coding: utf-8 -*-
"""Phase 3254 independent disk verification (fresh eyes)."""
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


RD = os.path.join(SRC_DIR, r'phase3254\g5h17_registry_landing')

# 1. sealed result
R = json.load(open(os.path.join(RD, 'result.json'), encoding='utf-8'))
check('1a result.json', R['phase'] == 3254 and R['mode'] == 'full'
      and R['verdict'] == 'registry_landed')
check('1b hashes', R['design_sha8'] == '8be6ccad' and R['res_sha8'] == '48ced87e',
      '%s/%s' % (R['design_sha8'], R['res_sha8']))
check('1c revert_check', R['revert_check'] == {'gap': True, 'registry': True})
check('1d file shas', R['gap_v112']['file_sha8'] == 'aa8cff21'
      and R['registry_v134']['file_sha8'] == '3deabf25')

# 2. landed gap v1.12 on disk
g12 = json.load(open(os.path.join(RD, 'gap_ledger_v1_12.json'), encoding='utf-8'))
check('2a version+appendix', g12['version'] == '1.12' and len(g12['appendix']) == 6)
a6 = g12['appendix'][-1]
check('2b APPX-6 entry', a6['id'] == 'APPX-6' and a6['anchor'] == 'p3250'
      and a6['anchor_sha8'] == '9d0f1825' and a6['status'] == 'appendix_recorded'
      and sorted(a6.keys()) == ['anchor', 'anchor_sha8', 'id', 'note', 'status', 'title'])
g11 = json.load(open(os.path.join(SRC_DIR, r'phase3237\g5g31_registry_landing'
                                  r'\gap_ledger_v1_11.json'), encoding='utf-8'))
legacy_ok = g12['gaps'] == g11['gaps'] and g12['appendix'][:5] == g11['appendix'] \
    and g12['schema'] == g11['schema'] and g12['created'] == g11['created'] \
    and g12['provenance'] == g11['provenance']
check('2c legacy gap bytes preserved', legacy_ok)

# 3. landed registry v1.34 on disk
r34 = json.load(open(os.path.join(RD, 'atlas_registry_v1_34.json'), encoding='utf-8'))
check('3a version+anchors', r34['version'] == '1.34'
      and len(r34['provenance']['anchor_files']) == 28
      and len(r34['upgrade_log']) == 13)
r33 = json.load(open(os.path.join(SRC_DIR, r'phase3238\g5h1_census_rank'
                                  r'\atlas_registry_v1_33.json'), encoding='utf-8'))
legacy_ok = r34['features'] == r33['features'] and r34['failures'] == r33['failures'] \
    and r34['evidence_level_taxonomy'] == r33['evidence_level_taxonomy'] \
    and r34['principles'] == r33['principles'] \
    and r34['census_3238'] == r33['census_3238'] \
    and r34['models_mainline'] == r33['models_mainline'] \
    and r34['provenance']['anchor_files'] == {
        **r33['provenance']['anchor_files'],
        **r34['provenance']['anchor_files']}
check('3b legacy registry bytes preserved', legacy_ok)
anch_ok = all(
    r34['provenance']['anchor_files'][k]['sha8'] == sha8_file(
        os.path.join(ROOT, r34['provenance']['anchor_files'][k]['path']))
    for k in ('p3246', 'p3250', 'p3252', 'p3253'))
check('3c new anchors sha8 match files on disk', anch_ok)
check('3d supersedes', 'v1_33' in r34['supersedes'] and 'a30215fd' in r34['supersedes']
      and 'v1_11' in g12['supersedes'] and '5934631b' in g12['supersedes'])

# 4. sealed predecessors byte-unchanged after landing
check('4a gap v1.11 sealed', sha8_file(os.path.join(
    SRC_DIR, r'phase3237\g5g31_registry_landing\gap_ledger_v1_11.json')) == '5934631b')
check('4b registry v1.33 sealed', sha8_file(os.path.join(
    SRC_DIR, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json')) == 'a30215fd')

# 5. smoke artifact
sm = json.load(open(os.path.join(RD, 'result_smoke.json'), encoding='utf-8'))
check('5 smoke artifact', sm['mode'] == 'smoke' and sm['design_sha8'] == '8be6ccad')

# 6. ledger #406 chain + backup
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
led = json.load(open(LED, encoding='utf-8'))
check('6a ledger n=406', len(led['measurements']) == 406)
e = led['measurements'][-1]
check('6b entry 3254', e.get('phase') == 3254
      and e.get('verdict', '').startswith('registry_landed'))
chk = dict(led)
chk['ledger_sha256_8'] = 'a13359da'
rec = sha8(json.dumps(chk, ensure_ascii=False, indent=1).encode('utf-8'))
check('6c chain post recompute', rec == led['ledger_sha256_8'],
      '%s' % led['ledger_sha256_8'])
bak = LED + '.bak_3254'
tail = e['detail'].split('Pre-append ledger file sha8 ')[-1].rstrip('."')
check('6d backup pre-file sha', os.path.exists(bak)
      and sha8(open(bak, 'rb').read()) == tail, '%s' % tail)

# 7. MEMO
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
mb = open(MEMO, 'rb').read()
check('7a memo 3254 once', mb.count(b'## Phase 3254:') == 1)
check('7b v1.34 shas in memo', 'aa8cff21'.encode() in mb and '3deabf25'.encode() in mb)

# 8. dual wlogs
for p, tag in ((r'research\gpt5\docs\WORKLOG_gpt5.md', 'ws'),
               (r'research\deepseek\docs\WORKLOG_gpt5_mirror.md', 'u')):
    txt = open(os.path.join(ROOT, p), encoding='utf-8').read()
    check('8 wlog %s' % tag, '3254 registry/gap append' in txt)

# 9. MEMORY
mem = open(os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md'), encoding='utf-8').read()
check('9a memory synced', '3254 落册（registry_landed）' in mem
      and 'registry v1.34 3deabf25' in mem and len(mem) <= 3000,
      '%d chars' % len(mem))

# 10. freeze marker
exe = json.load(open(os.path.join(RD, 'execution.json'), encoding='utf-8'))
check('10 freeze before observation', exe['frozen_before'] == 'any observation'
      and exe['design_sha8'] == '8be6ccad')

print('VERIFY 3254: %d OK / %d FAIL' % (len(OK), len(FAIL)))
for line in OK:
    print('  OK', line)
for line in FAIL:
    print('  FAIL', line)
if FAIL:
    raise SystemExit(1)
print('ALL OK')
