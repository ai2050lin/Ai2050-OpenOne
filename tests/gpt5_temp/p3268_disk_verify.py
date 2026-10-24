# -*- coding: utf-8 -*-
# Phase 3268 independent disk verification (read-only)
import json, hashlib, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                   'phase3268', 'g5h31_closure_standby')
LED = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
GAP113 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                      'phase3265', 'g5h28_appx7_landing', 'gap_ledger_v1_13.json')
REG136 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                      'phase3265', 'g5h28_appx7_landing', 'atlas_registry_v1_36.json')
GAP112 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                      'phase3254', 'g5h17_registry_landing', 'gap_ledger_v1_12.json')
REG135 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                      'phase3256', 'g5h19_closure_seal', 'atlas_registry_v1_35.json')
MF_V1 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                     'phase3256', 'g5h19_closure_seal', 'nline_handoff_manifest.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')

def s8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

fails = []
def chk(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail)
    if not cond:
        fails.append(name)

R = json.load(open(os.path.join(OUT, 'result.json'), encoding='utf-8'))

# 1 freeze
dfile = open(os.path.join(OUT, 'design_sha8.txt'), encoding='utf-8').read().strip()
chk("1 freeze consistent", dfile == R['design_sha8'] == 'bf5a0dfb', dfile)

# 2 res_sha8 recompute
r2 = {k: v for k, v in R.items() if k != 'res_sha8'}
rec = hashlib.sha256(json.dumps(r2, ensure_ascii=False, sort_keys=True,
                                indent=1).encode('utf-8')).hexdigest()[:8]
chk("2 res_sha8 recompute", rec == R['res_sha8'] == '1f6d94b0', rec)

# 3 artifact
A = json.load(open(os.path.join(OUT, 'standby_snapshot_v1.json'), encoding='utf-8'))
chk("3 artifact sha + status", s8(os.path.join(OUT, 'standby_snapshot_v1.json'))
    == '48308482' and A['status'].startswith('STANDBY'))

# 4 ledger #420 head
L = json.load(open(LED, encoding='utf-8'))
ms = L['measurements']
last = ms[-1]
chk("4 ledger head", len(ms) == 425 and last['phase'] == '3268'
    and last['name'] == 'closure_standby_registered', 'n=%d' % len(ms))

# 5 chain recompute (rec included, field = prev post)
Lc = dict(L)
Lc['measurements'] = ms
Lc['ledger_sha256_8'] = last['prev_post']
recompute = hashlib.sha256(json.dumps(Lc, ensure_ascii=False, sort_keys=True,
                                      indent=1).encode('utf-8')).hexdigest()[:8]
chk("5 chain recompute", recompute == L['ledger_sha256_8'] == 'b7407287', recompute)

# 6 gates
chk("6 gates 6/6", all(R['gates'][k] is True for k in R['gates'])
    and len(R['gates']) == 6)

# 7 incumbent untouched by this phase
chk("7 incumbent bytes", s8(GAP113) == '814ce808' and s8(REG136) == '78d37211')

# 8 sealed untouched
chk("8 sealed bytes", s8(GAP112) == 'aa8cff21' and s8(REG135) == 'e8335331'
    and s8(MF_V1) == 'bc212d52')

# 9 APPX matrix in artifact matches appendix
chk("9 appx matrix", A['appx_status_matrix'] == {
    'APPX-1': 'appendix_open', 'APPX-2': 'appendix_open', 'APPX-3': 'appendix_open',
    'APPX-4': 'appendix_open_crossline', 'APPX-5': 'appendix_open_crossline',
    'APPX-6': 'appendix_recorded', 'APPX-7': 'appendix_recorded'})

# 10 standby declaration: 5 owner items, no auto-resume
chk("10 standby declaration", len(A['standby_conditions']['owner_approval_required']) == 5
    and 'none' in A['standby_conditions']['auto_resume'])

# 11 MEMO section + post
mtxt = open(MEMO, encoding='utf-8').read()
chk("11 MEMO section", '## Phase 3268 (G5-H31):' in mtxt
    and 'b7407287' in mtxt and mtxt.count('## Phase 3268 (G5-H31):') == 1)

# 12 wlogs
w1 = open(os.path.join(ROOT, 'research', 'gpt5', 'docs', 'WORKLOG_gpt5.md'),
          encoding='utf-8').read()
w2 = open(os.path.join(ROOT, 'research', 'deepseek', 'docs', 'WORKLOG_gpt5_mirror.md'),
          encoding='utf-8').read()
chk("12 wlogs", w1.count('3268 G5-H31') == 1 and w2.count('3268 G5-H31') == 1)

# 13 MEMORY
mtxt2 = open(WMEM, encoding='utf-8').read()
chk("13 MEMORY", len(mtxt2) <= 3000 and '**3268' in mtxt2, 'len=%d' % len(mtxt2))

# 14 zero-write assertion in artifact
chk("14 zero-write flags", A['zero_ledger_write'] is True and A['zero_gpu'] is True
    and R['arm_c_summary']['pending_status_changed'] is False)

print('VERIFY %s: %d fails' % ('OK' if not fails else 'FAIL', len(fails)))
