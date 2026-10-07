# -*- coding: utf-8 -*-
"""Phase 14 收尾链终局清点：列出两个 Phase14 目录全部产物 + 关键文件 sha8/体积。"""
import io
import os
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
S14 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase14')
T14 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase14')
OUT = os.path.join(T14, 'final_inventory_phase14.txt')

KEY = [
    (os.path.join(T14, 'result_phase14.json'), 'result'),
    (os.path.join(T14, 'judgement_phase14.json'), 'judgement'),
    (os.path.join(T14, 'n2h1a7_report_qwen3-4b.txt'), 'report'),
    (os.path.join(T14, 'memo_append_phase14.md'), 'memo_append'),
    (os.path.join(T14, 'N2h1a7_design_seal.json'), 'seal'),
    (os.path.join(T14, 'N2h1a7_design_seal_amend1.json'), 'amend1'),
    (os.path.join(T14, 'N2h1a7_design_seal_amend2.json'), 'amend2'),
    (os.path.join(T14, 'execution_phase14.json'), 'exec'),
    (os.path.join(S14, 'disk_verify_phase14.txt'), 'disk_verify'),
    (os.path.join(S14, 'verify_append_phase14.txt'), 'verify_append'),
    (os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md'), 'MEMO'),
    (os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json'), 'Ledger'),
    (os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md'), 'MEMORY.md'),
    (os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md'), 'wlog'),
    (os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'memo_baseline.json'), 'baseline'),
    (r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md', 'skill:probe'),
    (r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md', 'skill:closeout'),
]


def info(p):
    if not os.path.exists(p):
        return 'MISSING'
    b = open(p, 'rb').read()
    return 'bytes=%-8d sha8=%s' % (len(b), hashlib.sha256(b).hexdigest()[:8])


L = ['=== Phase 14 终局清点 ===', '',
     '--- 关键产物 ---']
for p, tag in KEY:
    L.append('  %-16s %s' % (tag, info(p)))
    L.append('  %-16s %s' % ('', p.replace(ROOT + '\\', '')))
L.append('')
for d, nm in [(S14, 'tests/deepseek/Phase14'), (T14, 'tests/deepseek_temp/Phase14')]:
    fs = sorted(x for x in os.listdir(d) if os.path.isfile(os.path.join(d, x)))
    L.append('--- %s（%d 个文件）---' % (nm, len(fs)))
    for x in fs:
        p = os.path.join(d, x)
        L.append('  %-46s %6d B' % (x, os.path.getsize(p)))
    L.append('')
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L))
