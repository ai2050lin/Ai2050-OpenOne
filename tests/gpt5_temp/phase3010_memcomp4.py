# -*- coding: utf-8 -*-
"""Phase 3010 MEMORY micro compression round 4."""
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()

pairs = [
    ('脚本 tests\\glm5\\phase{N}_*.py；产物 tests\\glm5\\result\\rdc_query_construction_20260913\\phase{N}\\{arm}\\；临时 tests\\gpt5_temp\\。',
     '脚本 tests\\glm5\\phase{N}_*.py；产物 ...\\rdc_query_construction_20260913\\phase{N}\\{arm}\\；临时 tests\\gpt5_temp\\。'),
    ('重跑先删旧 execution.json/result.json/npz；负结果如实登记；verdict 须在判据分支内显式赋值。',
     '重跑先删旧产物；负结果如实登记；verdict 须在判据分支内显式赋值。'),
    ('MEMO 模板占位符一律 %(key)s 风格（{key} 是字面花括号不替换——2971/2972 两次产物行残留教训，收尾后必 Grep 复核）。',
     'MEMO 占位符一律 %(key)s 风格（{key} 是字面花括号不替换——教训在册，收尾后必 Grep 复核）。'),
]
miss = []
for a, b in pairs:
    if a in t:
        t = t.replace(a, b, 1)
    else:
        miss.append(a[:20])

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem10d.txt', 'w',
        encoding='utf-8').write(
    'len=%d miss=%s' % (len(t2), miss))
print('ok')
