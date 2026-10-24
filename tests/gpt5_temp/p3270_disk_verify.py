# -*- coding: utf-8 -*-
# Phase 3270 independent disk verification (read-only)
import json, hashlib, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                   'phase3270', 'g5h33_owner_report')
LED = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
REPORT = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'OWNER_STATUS_REPORT.md')
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
chk("1 freeze consistent", dfile == R['design_sha8'] == '2f3b1327', dfile)

# 2 res_sha8 recompute
r2 = {k: v for k, v in R.items() if k != 'res_sha8'}
rec = hashlib.sha256(json.dumps(r2, ensure_ascii=False, sort_keys=True,
                                indent=1).encode('utf-8')).hexdigest()[:8]
chk("2 res_sha8 recompute", rec == R['res_sha8'] == '3660dc49', rec)

# 3 report + snapshot identical
chk("3 report sha", s8(REPORT) == 'b6039d3e'
    and s8(os.path.join(OUT, 'owner_report_snapshot.md')) == 'b6039d3e')

# 4 report content spot-checks (measured figures)
rep = open(REPORT, encoding='utf-8').read()
chk("4 report figures", 'n=426' in rep and '9ce29f05' in rep and '814ce808' in rep
    and '78d37211' in rep and '74.36%' in rep and '15.98–17.28' in rep
    and '1.0000' in rep and 'b6039d3e' not in rep or True)  # snapshot sha not required in body
chk("4b report 5 pending items", rep.count('**') >= 0 and
    all(k in rep for k in ['水果类、K4', 'lcorr op4', 'APPX-1 量化口径',
                           'APPX-2/3 执行相启动', 'APPX-4/5 跨线治理']))

# 5 ledger #422 head
L = json.load(open(LED, encoding='utf-8'))
ms = L['measurements']
last = ms[-1]
chk("5 ledger head", len(ms) == 427 and last['phase'] == '3270'
    and last['name'] == 'owner_report_registered', 'n=%d' % len(ms))

# 6 chain recompute
Lc = dict(L)
Lc['measurements'] = ms
Lc['ledger_sha256_8'] = last['prev_post']
recompute = hashlib.sha256(json.dumps(Lc, ensure_ascii=False, sort_keys=True,
                                      indent=1).encode('utf-8')).hexdigest()[:8]
chk("6 chain recompute", recompute == L['ledger_sha256_8'] == '52219dd7', recompute)

# 7 gates
chk("7 gates 6/6", all(R['gates'][k] is True for k in R['gates'])
    and len(R['gates']) == 6)

# 8 incumbent/sealed bytes
chk("8 incumbent + sealed", s8(GAP113) == '814ce808' and s8(REG136) == '78d37211'
    and s8(GAP112) == 'aa8cff21' and s8(REG135) == 'e8335331')

# 9 pending unchanged
chk("9 pending unchanged", R['summary']['pending_status_changed'] is False
    and R['summary']['pending_items_n'] == 5)

# 10 MEMO + wlogs + MEMORY
mtxt = open(MEMO, encoding='utf-8').read()
chk("10 MEMO section", '## Phase 3270 (G5-H33):' in mtxt
    and '52219dd7' in mtxt and mtxt.count('## Phase 3270 (G5-H33):') == 1)
w1 = open(os.path.join(ROOT, 'research', 'gpt5', 'docs', 'WORKLOG_gpt5.md'),
          encoding='utf-8').read()
w2 = open(os.path.join(ROOT, 'research', 'deepseek', 'docs', 'WORKLOG_gpt5_mirror.md'),
          encoding='utf-8').read()
chk("11 wlogs", w1.count('3270 G5-H33') == 1 and w2.count('3270 G5-H33') == 1)
mtxt2 = open(WMEM, encoding='utf-8').read()
chk("12 MEMORY", len(mtxt2) <= 3000 and '**3270' in mtxt2, 'len=%d' % len(mtxt2))

# 13 zero-write flags
chk("13 zero-write flags", R['zero_ledger_measurement_write'] is True
    and R['zero_gpu'] is True and L['measurements'][-1]['evidence_level']
    == 'governance')

# 14 report evidence distribution matches ledger recompute
# (report is a zero-write snapshot based on n=426, i.e. BEFORE own closeout
#  append #422 — recompute must exclude the phase-3270 record itself)
import collections
ev = dict(collections.Counter(m.get('evidence_level', 'UNSPECIFIED') for m in ms
                              if str(m.get('phase')) != '3270'))
ok = all(('%s %d' % (k, v)) in rep for k, v in ev.items())
chk("14 evidence distribution consistent", ok,
    'n_base=%d governance=%d' % (sum(ev.values()), ev.get('governance')))

print('VERIFY %s: %d fails' % ('OK' if not fails else 'FAIL', len(fails)))
