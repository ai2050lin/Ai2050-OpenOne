# -*- coding: utf-8 -*-
"""Phase 3255 independent disk verification (fresh eyes)."""
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


RD = os.path.join(SRC_DIR, r'phase3255\g5h18_client_synthesis')

# 1. sealed result
R = json.load(open(os.path.join(RD, 'result.json'), encoding='utf-8'))
check('1a result.json', R['phase'] == 3255 and R['mode'] == 'full'
      and R['verdict'] == 'client_synthesis_registered')
check('1b hashes', R['design_sha8'] == '1d19fc3d' and R['res_sha8'] == 'ac6bd1d9')
check('1c engineering readings', R['engineering']['endpoint_version'] == '1.12'
      and R['engineering']['appendix_n'] == 6
      and R['engineering']['ui_check'] == '17/17')
check('1d doc arm appended', R['doc_arm_status'] == 'appended')

# 2. engineering artifacts on disk
svc = open(os.path.join(ROOT, 'server', 'ai_rnd_service.py'), encoding='utf-8').read()
check('2a endpoint appendix patch', '"appendix": data.get("appendix", [])' in svc)
panel = open(os.path.join(ROOT, r'frontend\src\components\app\rdc_fusion'
                          r'\GapLedgerPanel.jsx'), encoding='utf-8').read()
check('2b panel file', 'data-k="gap_ledger_panel"' in panel
      and '/api/ai-rnd/atlas/gaps' in panel and 'anchor_sha8' in panel)
lens = open(os.path.join(ROOT, r'frontend\src\components\app\rdc_fusion'
                         r'\AtlasLens.jsx'), encoding='utf-8').read()
check('2c panel mounted', "GapLedgerPanel from './GapLedgerPanel.jsx'" in lens
      and '<GapLedgerPanel />' in lens and 'v1.28' not in lens)
css = open(os.path.join(ROOT, r'frontend\src\components\app\rdc_fusion'
                        r'\rdc_fusion.css'), encoding='utf-8').read()
check('2d css classes', '.glp-gaps' in css and '.glp-appx.glp-newest' in css)

# 3. check outputs
ep = open(os.path.join(ROOT, r'tests\gpt5_temp\p3255_endpoint_out.txt'),
          encoding='utf-8').read()
check('3a endpoint check', 'ENDPOINT CHECK OK' in ep and '9d0f1825' in ep)
ui = open(os.path.join(ROOT, r'tests\gpt5_temp\p3255_ui_out.txt'),
          encoding='utf-8').read()
check('3b ui check 17/17', 'UI CHECK: 17/17' in ui and 'FAIL' not in ui
      and 'console errors: none' in ui)

# 4. MEMO: synthesis section once + phase section once
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
mb = open(MEMO, 'rb').read()
check('4a memo synthesis once', mb.count('## u* 几何图谱综论'.encode('utf-8')) == 1)
check('4b memo 3255 once', mb.count(b'## Phase 3255:') == 1)

# 5. sealed atlas files untouched
check('5a registry v1.34 sealed', sha8_file(os.path.join(
    SRC_DIR, r'phase3254\g5h17_registry_landing\atlas_registry_v1_34.json')) == '3deabf25')
check('5b gap v1.12 sealed', sha8_file(os.path.join(
    SRC_DIR, r'phase3254\g5h17_registry_landing\gap_ledger_v1_12.json')) == 'aa8cff21')

# 6. ledger #407 chain + backup
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
led = json.load(open(LED, encoding='utf-8'))
check('6a ledger n=407', len(led['measurements']) == 407)
e = led['measurements'][-1]
check('6b entry 3255', e.get('phase') == 3255
      and e.get('verdict', '').startswith('client_synthesis_registered'))
chk = dict(led)
chk['ledger_sha256_8'] = '8fb0bce0'
rec = sha8(json.dumps(chk, ensure_ascii=False, indent=1).encode('utf-8'))
check('6c chain post recompute', rec == led['ledger_sha256_8'],
      '%s' % led['ledger_sha256_8'])
bak = LED + '.bak_3255'
tail = e['detail'].split('Pre-append ledger file sha8 ')[-1].rstrip('."')
check('6d backup pre-file sha', os.path.exists(bak)
      and sha8(open(bak, 'rb').read()) == tail, '%s' % tail)

# 7. dual wlogs
for p, tag in ((r'research\gpt5\docs\WORKLOG_gpt5.md', 'ws'),
               (r'research\deepseek\docs\WORKLOG_gpt5_mirror.md', 'u')):
    txt = open(os.path.join(ROOT, p), encoding='utf-8').read()
    check('7 wlog %s' % tag, '3255 registry v1.34 客户端化' in txt)

# 8. MEMORY
mem = open(os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md'), encoding='utf-8').read()
check('8 memory synced', '3255' in mem and '双轨道几何' in mem and len(mem) <= 3000,
      '%d chars' % len(mem))

# 9. freeze marker
exe = json.load(open(os.path.join(RD, 'execution.json'), encoding='utf-8'))
check('9 freeze before observation', exe['frozen_before'] == 'any observation'
      and exe['design_sha8'] == '1d19fc3d')

print('VERIFY 3255: %d OK / %d FAIL' % (len(OK), len(FAIL)))
for line in OK:
    print('  OK', line)
for line in FAIL:
    print('  FAIL', line)
if FAIL:
    raise SystemExit(1)
print('ALL OK')
