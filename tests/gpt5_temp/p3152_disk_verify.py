# -*- coding: utf-8 -*-
"""Phase 3152 disk verify: fix 2 typos in MEMO + independent verification."""
import hashlib
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LED = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-01.md')
MEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
BASE = os.path.join(ROOT, 'tests', 'glm5', 'result',
                    'rdc_query_construction_20260913',
                    'phase3152', 'g1p2_tri_model_k1')

rep = []

# 0) typo fixes in MEMO (same-closeout transcription slips, pre-close)
memo = open(MEMO, encoding='utf-8').read()
fixes = [
    (u'k7 单层跳变 0.006→0.197', u'k7 单层跳变 0.008→0.197'),
    (u'（读出层 margin +0.75/+1.08/+1.08）',
     u'（读出层 margin +0.75/+1.08/+0.90）'),
]
nfix = 0
for a, b in fixes:
    if a in memo:
        memo = memo.replace(a, b)
        nfix += 1
if nfix:
    with open(MEMO, 'w', encoding='utf-8', newline='') as f:
        f.write(memo)
rep.append('typo fixes applied: %d' % nfix)

# 1) MEMO re-check after fix
memo = open(MEMO, encoding='utf-8').read()
assert '## Phase 3152:' in memo, 'MEMO 3152 section missing!'
assert 'k1_not_triggered_b4_additive_at_kstar_operator_line_kept' in memo
assert '0.008→0.197' in memo
assert '预注册 Phase 3153' in memo
rep.append('MEMO: Phase 3152 section present, 3153 pre-reg present')

# 2) ledger
led = json.load(open(LED, encoding='utf-8'))
n = len(led['measurements'])
e3152 = [m for m in led['measurements'] if m.get('phase') == 3152]
assert n == 289 and len(e3152) == 1, (n, len(e3152))
rep.append('ledger n=%d, 3152 entry verdict=%s seal=%s' %
           (n, e3152[0]['verdict'][:50], e3152[0]['seal_sha8']))

# 3) daily
daily = open(DAILY, encoding='utf-8').read()
assert 'Phase 3152 (gpt5 线)' in daily
rep.append('daily: Phase 3152 note present')

# 4) MEMORY.md
mem = open(MEM, encoding='utf-8').read()
assert 'n=289 @3152' in mem and '3152：G1-P2' in mem
rep.append('MEMORY.md: updated')

# 5) artifacts on disk
for tag, fn in [('qwen3-4b', 'result.json'), ('qwen3-14b', 'result.json'),
                ('glm4k1', 'result.json'), ('summary', 'result_summary.json')]:
    p = os.path.join(BASE, tag, fn)
    raw = open(p, 'rb').read()
    d = hashlib.sha256(raw).hexdigest()[:8]
    r = json.loads(raw.decode('utf-8'))
    rep.append('%s/%s disk=%s res=%s seal=%s verdict=%s' %
               (tag, fn, d, r.get('res_sha8'), r.get('seal_sha8'),
                r['verdict'][:70]))
for tag in ['qwen3-4b', 'qwen3-14b']:
    p = os.path.join(BASE, tag, 'collect.npz')
    sz = os.path.getsize(p)
    rep.append('%s collect.npz size=%d' % (tag, sz))
for tag in ['qwen3-4b', 'qwen3-14b', 'glm4k1', 'summary']:
    p = os.path.join(BASE, tag, 'execution.json')
    json.load(open(p, encoding='utf-8'))
    rep.append('%s execution.json ok' % tag)

# 6) numeric cross-check vs extracted reports
r14b = json.load(open(os.path.join(BASE, 'qwen3-14b', 'result.json'),
                      encoding='utf-8'))
kr = r14b['k1_model_report']
assert abs(kr['b4_rel_kstar_mean3seed'] - 0.0807) < 5e-4
assert kr['m1_kstar']['pass_all3'] is True
r4b = json.load(open(os.path.join(BASE, 'qwen3-4b', 'result.json'),
                     encoding='utf-8'))
assert abs(r4b['k1_model_report']['b4_rel_kstar_mean3seed'] - 0.0079) < 5e-4
rg4 = json.load(open(os.path.join(BASE, 'glm4k1', 'result.json'),
                     encoding='utf-8'))
assert rg4['k1_model_report']['anchor_b4_k39_drift'] == 0.0
rsum = json.load(open(os.path.join(BASE, 'summary',
                                   'result_summary.json'), encoding='utf-8'))
assert rsum['k1_triggered'] is False
assert rsum['k1_verdict'] == ('k1_not_triggered_b4_additive_at_kstar_'
                              'operator_line_kept')
rep.append('numeric cross-check: all consistent')

txt = '\n'.join(rep)
open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3152_verify_report.txt'),
     'w', encoding='utf-8').write(txt + '\n')
print(txt)
print('VERIFY OK')
