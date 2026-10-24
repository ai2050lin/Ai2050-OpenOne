# -*- coding: utf-8 -*-
# Phase 3267 independent disk verification (read-only)
import json, hashlib, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                   'phase3267', 'g5h30_appx23_protocol')
LED = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
GAP113 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                      'phase3265', 'g5h28_appx7_landing', 'gap_ledger_v1_13.json')
REG136 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                      'phase3265', 'g5h28_appx7_landing', 'atlas_registry_v1_36.json')
GAP112 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                      'phase3254', 'g5h17_registry_landing', 'gap_ledger_v1_12.json')
REG135 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                      'phase3256', 'g5h19_closure_seal', 'atlas_registry_v1_35.json')
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
chk("1 freeze consistent", dfile == R['design_sha8'] == '05235373', dfile)

# 2 res_sha8 recompute
r2 = {k: v for k, v in R.items() if k != 'res_sha8'}
rec = hashlib.sha256(json.dumps(r2, ensure_ascii=False, sort_keys=True,
                                indent=1).encode('utf-8')).hexdigest()[:8]
chk("2 res_sha8 recompute", rec == R['res_sha8'] == '5519514c', rec)

# 3 artifact
A = json.load(open(os.path.join(OUT, 'appx23_protocol_v2.json'), encoding='utf-8'))
chk("3 artifact sha + status", s8(os.path.join(OUT, 'appx23_protocol_v2.json'))
    == 'ce73984d' and A['status'].startswith('PREREGISTERED_DRAFT_V2'))

# 4 ledger #419 + chain recompute
L = json.load(open(LED, encoding='utf-8'))
ms = L['measurements']
last = ms[-1]
chk("4 ledger head", len(ms) == 424 and last['phase'] == '3267'
    and last['name'] == 'appx23_protocol_v2_registered', 'n=%d' % len(ms))
Lc = dict(L)
Lc['measurements'] = ms  # chain rule: new rec INCLUDED, field = prev post
Lc['ledger_sha256_8'] = last['prev_post']
recompute = hashlib.sha256(json.dumps(Lc, ensure_ascii=False, sort_keys=True,
                                      indent=1).encode('utf-8')).hexdigest()[:8]
chk("5 chain recompute", recompute == L['ledger_sha256_8'] == '6f2a9170', recompute)

# 6 gates
chk("6 gates 6/6", all(R['gates'][k] is True for k in R['gates'])
    and len(R['gates']) == 6)

# 7 arm-c pending untouched
chk("7 pending unchanged", R['arm_c_summary']['pending_count'] == 3
    and R['arm_c_summary']['pending_status_changed'] is False)

# 8 sealed v1.12/v1.35 byte-unchanged
chk("8 sealed bytes", s8(GAP112) == 'aa8cff21' and s8(REG135) == 'e8335331')

# 9 incumbent v1.13/v1.36 untouched by this phase
chk("9 incumbent bytes", s8(GAP113) == '814ce808' and s8(REG136) == '78d37211')

# 10 MEMO section + post
mtxt = open(MEMO, encoding='utf-8').read()
chk("10 MEMO section", '## Phase 3267 (G5-H30):' in mtxt
    and '6f2a9170' in mtxt and mtxt.count('## Phase 3267 (G5-H30):') == 1)

# 11 wlogs
w1 = open(os.path.join(ROOT, 'research', 'gpt5', 'docs', 'WORKLOG_gpt5.md'),
          encoding='utf-8').read()
w2 = open(os.path.join(ROOT, 'research', 'deepseek', 'docs', 'WORKLOG_gpt5_mirror.md'),
          encoding='utf-8').read()
chk("11 wlogs", '3267 G5-H30' in w1 and '3267 G5-H30' in w2
    and w1.count('3267 G5-H30') == 1 and w2.count('3267 G5-H30') == 1)

# 12 MEMORY
mtxt2 = open(WMEM, encoding='utf-8').read()
chk("12 MEMORY", len(mtxt2) <= 3000 and '**3267' in mtxt2, 'len=%d' % len(mtxt2))

# 13 artifact content spot: deathline verbatim
chk("13 K2/K3 verbatim in artifact",
    'cos 降 >50%' in json.dumps(A['arm_b_appx3']['deathlines'], ensure_ascii=False)
    and 'top-50 覆盖率均 <30%' in json.dumps(A['arm_b_appx3']['deathlines'],
                                             ensure_ascii=False))

# 14 APPX-2 steps/gates counts
chk("14 protocol structure", len(A['arm_a_appx2']['steps']) == 4
    and len(A['arm_a_appx2']['gates']) == 5
    and len(A['arm_b_appx3']['steps']) == 4
    and len(A['arm_b_appx3']['gates']) == 5)

print('VERIFY %s: %d fails' % ('OK' if not fails else 'FAIL', len(fails)))
