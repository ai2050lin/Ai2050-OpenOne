# -*- coding: utf-8 -*-
"""Phase 3257 independent disk verification (fresh eyes)."""
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


RD = os.path.join(SRC_DIR, r'phase3257\g5h20_final_audit')

# 1. sealed result
R = json.load(open(os.path.join(RD, 'result.json'), encoding='utf-8'))
check('1a result.json', R['phase'] == 3257 and R['mode'] == 'formal'
      and R['verdict'] == 'final_audit_passed' and all(R['gates'].values()))
check('1b hashes', R['design_sha8'] == 'c2390c7f' and R['res_sha8'] == '961a9e97')
check('1c audit numbers', R['three_account_audit']['anchor_files']['match'] == 28
      and R['three_account_audit']['anchor_files']['total'] == 28
      and R['three_account_audit']['gap_anchor_bad' if 'gap_anchor_bad' in R['three_account_audit'] else 'ledger']['post'] == '5601b3f5')

# 2. live check outputs on disk (fresh read, independent)
ep = open(os.path.join(ROOT, r'tests\gpt5_temp\p3257_endpoint_out.txt'), encoding='utf-8').read()
check('2a endpoint output', 'ENDPOINT CHECK OK' in ep
      and ep.count('PASS') == 14 and 'FAIL' not in ep)
ui = open(os.path.join(ROOT, r'tests\gpt5_temp\p3257_ui_out.txt'), encoding='utf-8').read()
check('2b ui output', 'UI CHECK: 11/11' in ui and 'FAIL' not in ui
      and 'console errors: none' in ui)

# 3. disk audit re-executed fresh (anchors)
reg35 = json.load(open(os.path.join(SRC_DIR, r'phase3256\g5h19_closure_seal'
                                    r'\atlas_registry_v1_35.json'), encoding='utf-8'))
bad = [k for k, ent in reg35['provenance']['anchor_files'].items()
       if not (os.path.isfile(os.path.join(ROOT, ent['path']))
               and sha8_file(os.path.join(ROOT, ent['path'])) == ent['sha8'])]
check('3a anchor_files 28/28 fresh', len(reg35['provenance']['anchor_files']) == 28
      and not bad, str(bad[:3]))

# 4. gap anchor conventions re-verified fresh
gap = json.load(open(os.path.join(SRC_DIR, r'phase3254\g5h17_registry_landing'
                                  r'\gap_ledger_v1_12.json'), encoding='utf-8'))
CONV = {
    'APPX-1': ('file', os.path.join(SRC_DIR, r'phase3166\g5a3b_logic_direction\result.json')),
    'APPX-2': ('file', os.path.join(SRC_DIR, r'phase3165\g5a3_family_alignment\result.json')),
    'APPX-3': ('file', os.path.join(SRC_DIR, r'phase3155\g2p1_relation_family_operator_separability'
                                    r'\summary\result_summary.json')),
    'APPX-4': ('file', os.path.join(ROOT, r'research\deepseek\atlas\a_gate_closure_v1.json')),
    'APPX-5': ('file', os.path.join(ROOT, r'research\deepseek\atlas\a_gate_closure_v1.json')),
    'APPX-6': ('res_sha8', os.path.join(SRC_DIR, r'phase3250\g5h13_kernel_boundary\result.json')),
}
gbad = []
for a in gap['appendix']:
    kind, p = CONV[a['id']]
    if kind == 'file':
        ok_a = sha8_file(p) == a['anchor_sha8']
    else:
        ok_a = json.load(open(p, encoding='utf-8'))['res_sha8'] == a['anchor_sha8']
    if not ok_a:
        gbad.append(a['id'])
check('4 gap anchors 6/6 fresh', not gbad, str(gbad))

# 5. apply_preview content + read-only guarantee
ap = json.load(open(os.path.join(RD, 'apply_preview.json'), encoding='utf-8'))
check('5a apply preview', ap['schema'] == 'nline_apply_preview_v1'
      and ap['read_only'] is True and ap['writes_to_gline'] is False
      and len(ap['steps_if_applied']) == 5
      and ap['manifest_sha8_file'] == sha8_file(os.path.join(
          SRC_DIR, r'phase3256\g5h19_closure_seal\nline_handoff_manifest.json')))
check('5b manifest reverify', ap['manifest_reverify']['bad'] == []
      and ap['manifest_reverify']['ledger_snapshot_ok'] is True)

# 6. sealed files untouched
check('6a v1.34 sealed', sha8_file(os.path.join(
    SRC_DIR, r'phase3254\g5h17_registry_landing\atlas_registry_v1_34.json')) == '3deabf25')
check('6b v1.12 sealed', sha8_file(os.path.join(
    SRC_DIR, r'phase3254\g5h17_registry_landing\gap_ledger_v1_12.json')) == 'aa8cff21')
check('6c v1.35 sealed', sha8_file(os.path.join(
    SRC_DIR, r'phase3256\g5h19_closure_seal\atlas_registry_v1_35.json')) == 'e8335331')

# 7. freeze marker
D = json.load(open(os.path.join(RD, 'design.json'), encoding='utf-8'))
dh = sha8(json.dumps({k: v for k, v in D.items() if k != 'created'},
                     sort_keys=True, ensure_ascii=False).encode('utf-8'))
check('7 freeze marker', open(os.path.join(RD, 'design_sha8.txt'), encoding='utf-8').read() == dh
      and dh == R['design_sha8'] == 'c2390c7f' and D['mode'] == 'formal'
      and D['read_only_arm_b'] is True)

# 8. MEMO phase section once
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
mb = open(MEMO, 'rb').read()
check('8 memo 3257 once', mb.count(b'## Phase 3257:') == 1
      and 'final_audit_passed'.encode('utf-8') in mb
      and '961a9e97'.encode('utf-8') in mb)
check('8b erratum applied', '14 断言全过'.encode('utf-8') in mb
      and '15 断言全过'.encode('utf-8') not in mb
      and '15/15'.encode('utf-8') in mb)  # erratum note quotes the stale label once

# 9. ledger #409 chain + backup
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
led = json.load(open(LED, encoding='utf-8'))
check('9a ledger n=409', len(led['measurements']) == 409)
e = led['measurements'][-1]
check('9b entry 3257', e.get('phase') == 3257
      and e.get('verdict', '').startswith('final_audit_passed'))
chk = dict(led)
chk['ledger_sha256_8'] = '5601b3f5'
rec = sha8(json.dumps(chk, ensure_ascii=False, indent=1).encode('utf-8'))
check('9c chain post recompute', rec == led['ledger_sha256_8'],
      '%s' % led['ledger_sha256_8'])
bak = LED + '.bak_3257'
tail = e['detail'].split('Pre-append ledger file sha8 ')[-1].rstrip('."')
check('9d backup pre-file sha', os.path.exists(bak)
      and sha8(open(bak, 'rb').read()) == tail, '%s' % tail)

# 10. dual wlogs + MEMORY
for p, tag in ((r'research\gpt5\docs\WORKLOG_gpt5.md', 'ws'),
               (r'research\deepseek\docs\WORKLOG_gpt5_mirror.md', 'u')):
    txt = open(os.path.join(ROOT, p), encoding='utf-8').read()
    check('10 wlog %s' % tag, '3257 G 线收官核验' in txt)
mem = open(os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md'), encoding='utf-8').read()
check('11 memory synced', '3257 终审' in mem and '可交付' in mem
      and len(mem) <= 3000, '%d chars' % len(mem))

print('VERIFY 3257: %d OK / %d FAIL' % (len(OK), len(FAIL)))
for line in OK:
    print('  OK', line)
for line in FAIL:
    print('  FAIL', line)
if FAIL:
    raise SystemExit(1)
print('ALL OK')
