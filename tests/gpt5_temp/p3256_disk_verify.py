# -*- coding: utf-8 -*-
"""Phase 3256 independent disk verification (fresh eyes)."""
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


RD = os.path.join(SRC_DIR, r'phase3256\g5h19_closure_seal')
R134 = os.path.join(SRC_DIR, r'phase3254\g5h17_registry_landing\atlas_registry_v1_34.json')
G112 = os.path.join(SRC_DIR, r'phase3254\g5h17_registry_landing\gap_ledger_v1_12.json')
R135 = os.path.join(RD, 'atlas_registry_v1_35.json')

# 1. sealed result
R = json.load(open(os.path.join(RD, 'result.json'), encoding='utf-8'))
check('1a result.json', R['phase'] == 3256 and R['mode'] == 'formal'
      and R['verdict'] == 'closure_sealed_registered' and all(R['gates'].values()))
check('1b hashes', R['design_sha8'] == '5061e1e8' and R['res_sha8'] == '01393c60')
check('1c landing', R['registry_landing']['to'] == '1.35'
      and R['registry_landing']['file_sha8'] == 'e8335331'
      and R['registry_landing']['revert_bitwise'] is True)
check('1d facts', R['facts']['repair_vs_zero'] == 0.7436
      and R['facts']['G0_SCLS'] == -1.751489
      and R['facts']['ORC_ls_full_SCLS'] == 0.329005
      and R['facts']['incumbent_headline_minSCLSENT'] == -0.449039)

# 2. landed registry v1.35 (independent structural + revert check)
raw135 = open(R135, 'rb').read()
check('2a file sha', sha8(raw135) == 'e8335331')
r35 = json.loads(raw135.decode('utf-8'))
raw134 = open(R134, 'rb').read()
r34 = json.loads(raw134.decode('utf-8'))
check('2b version chain', r35['version'] == '1.35'
      and r35['supersedes'].startswith('atlas_registry_v1_34.json (3deabf25')
      and len(r35['upgrade_log']) == len(r34['upgrade_log']) + 1)
seal = r35['upgrade_log'][-1]
check('2c seal entry', 'closure seal' in seal['change']
      and 'XC_p5_b8' in seal['change'] and 'F20' in seal['change']
      and '3256' in seal['reason'])
# independent revert: undo planned ptrs -> must equal v1.34 raw bytes
import copy
o = copy.deepcopy(r35)
o['version'] = r34['version']
o['supersedes'] = r34['supersedes']
o['upgrade_log'] = o['upgrade_log'][:-1]
check('2d revert bitwise (fresh)', json.dumps(o, ensure_ascii=False, indent=1).encode('utf-8') == raw134)
# independent pointer diff (only these three change)
o2 = copy.deepcopy(r35)
diffs = set()
if o2['version'] != r34['version']:
    diffs.add('version')
    o2['version'] = r34['version']
if o2['supersedes'] != r34['supersedes']:
    diffs.add('supersedes')
    o2['supersedes'] = r34['supersedes']
o2['upgrade_log'] = o2['upgrade_log'][:-1]
if json.dumps(o2, sort_keys=True) != json.dumps(r34, sort_keys=True):
    diffs.add('OTHER-CONTENT-CHANGED')
check('2e ptr diff exact', diffs == {'version', 'supersedes'}, str(diffs))

# 3. closure summary numbers measured
sm = open(os.path.join(RD, 'closure_summary_3239_3255.md'), encoding='utf-8').read()
for token in ('-1.751489', '+0.329005', '-0.449039', '74.36%', 'XC_p5_b8', 'F20',
              'INFORMATION_GAP', 'v1.35'):
    check('3 summary has %s' % token, token in sm)

# 4. handoff manifest consistency with disk
man = json.load(open(os.path.join(RD, 'nline_handoff_manifest.json'), encoding='utf-8'))
check('4a manifest schema', man['schema'] == 'nline_handoff_v1'
      and man['closure_facts']['repair_vs_zero'] == 0.7436)
sa = man['sealed_artifacts']
check('4b manifest shas match disk',
      sa['registry_prev']['sha8_file'] == sha8_file(R134) == '3deabf25'
      and sa['gap_ledger_current']['sha8_file'] == sha8_file(G112) == 'aa8cff21'
      and sa['registry_current']['sha8_file'] == sha8_file(R135) == 'e8335331'
      and sa['ledger']['post_sha8'] == '5601b3f5' and sa['ledger']['entries'] == 408)
tri = man['u_star_trilogy']
check('4c trilogy res sha', all(os.path.isfile(p) for p in (
    os.path.join(SRC_DIR, r'phase3251\g5h14_ustar_lexical_attribution\result.json'),
    os.path.join(SRC_DIR, r'phase3252\g5h15_ustar_layer_trace\result.json'),
    os.path.join(SRC_DIR, r'phase3253\g5h16_ustar_subspace\result.json')))
    and tri['3251_lexical']['res_sha8'] == sha8_file(os.path.join(
        SRC_DIR, r'phase3251\g5h14_ustar_lexical_attribution\result.json'))
    and tri['3252_layer']['res_sha8'] == sha8_file(os.path.join(
        SRC_DIR, r'phase3252\g5h15_ustar_layer_trace\result.json'))
    and tri['3253_subspace']['res_sha8'] == sha8_file(os.path.join(
        SRC_DIR, r'phase3253\g5h16_ustar_subspace\result.json')))
check('4d gpu backlog', len(man['gpu_backlog']) == 2
      and 'checkpoint' in man['gpu_backlog'][0])

# 5. sealed predecessors untouched
check('5a v1.34 sealed', sha8_file(R134) == '3deabf25')
check('5b v1.12 sealed', sha8_file(G112) == 'aa8cff21')

# 6. freeze marker (design.json + design_sha8.txt, hash excludes created)
D = json.load(open(os.path.join(RD, 'design.json'), encoding='utf-8'))
dh = sha8(json.dumps({k: v for k, v in D.items() if k != 'created'},
                     sort_keys=True, ensure_ascii=False).encode('utf-8'))
check('6 freeze marker', open(os.path.join(RD, 'design_sha8.txt'), encoding='utf-8').read() == dh
      and dh == R['design_sha8'] == '5061e1e8' and D['mode'] == 'formal'
      and len(D['planned_registry_ptrs']) == 3)

# 7. MEMO phase section once
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
mb = open(MEMO, 'rb').read()
check('7 memo 3256 once', mb.count(b'## Phase 3256:') == 1
      and 'closure_sealed_registered'.encode('utf-8') in mb
      and 'e8335331'.encode('utf-8') in mb)

# 8. ledger #408 chain + backup
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
led = json.load(open(LED, encoding='utf-8'))
check('8a ledger n=408', len(led['measurements']) == 408)
e = led['measurements'][-1]
check('8b entry 3256', e.get('phase') == 3256
      and e.get('verdict', '').startswith('closure_sealed_registered'))
chk = dict(led)
chk['ledger_sha256_8'] = 'b411f7ef'
rec = sha8(json.dumps(chk, ensure_ascii=False, indent=1).encode('utf-8'))
check('8c chain post recompute', rec == led['ledger_sha256_8'],
      '%s' % led['ledger_sha256_8'])
bak = LED + '.bak_3256'
tail = e['detail'].split('Pre-append ledger file sha8 ')[-1].rstrip('."')
check('8d backup pre-file sha', os.path.exists(bak)
      and sha8(open(bak, 'rb').read()) == tail, '%s' % tail)

# 9. dual wlogs
for p, tag in ((r'research\gpt5\docs\WORKLOG_gpt5.md', 'ws'),
               (r'research\deepseek\docs\WORKLOG_gpt5_mirror.md', 'u')):
    txt = open(os.path.join(ROOT, p), encoding='utf-8').read()
    check('9 wlog %s' % tag, '3256 收线终点封章' in txt)

# 10. MEMORY
mem = open(os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md'), encoding='utf-8').read()
check('10 memory synced', '3256 封章' in mem and '3257 已预注册' in mem
      and len(mem) <= 3000, '%d chars' % len(mem))

print('VERIFY 3256: %d OK / %d FAIL' % (len(OK), len(FAIL)))
for line in OK:
    print('  OK', line)
for line in FAIL:
    print('  FAIL', line)
if FAIL:
    raise SystemExit(1)
print('ALL OK')
