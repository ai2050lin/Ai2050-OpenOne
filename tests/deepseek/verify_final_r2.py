import os, hashlib

root = r'D:\AI2050\Ai2050-OpenOne'
out = []

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

# 1. memo
p = os.path.join(root, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
b = open(p, 'rb').read()
T = b.decode('utf-8-sig')
L = T.splitlines()
out.append('[1] deepseek MEMO bytes=%d lines=%d sha8=%s' % (len(b), len(L), sha8(p)))
for h in ['## Phase 7', '## 复核 R1:', '## 复核 R1 确认 + 登记约定变更']:
    idx = [i + 1 for i, l in enumerate(L) if l.startswith(h)]
    out.append('    hdr %-34s count=%d lines=%s' % (h, len(idx), idx))
out.append('    has 137 项: %s ; has P-N3b: %s ; has tests/deepseek_temp: %s' % ('137 项' in T, 'P-N3b' in T, 'tests/deepseek_temp' in T))

# 2. MEMORY.md
p = os.path.join(root, '.workbuddy', 'memory', 'MEMORY.md')
T2 = open(p, encoding='utf-8').read()
out.append('[2] MEMORY.md bytes=%d' % os.path.getsize(p))
for k in ['产物落点（2026-10-01 21:10 变更', '5/5 适用 + 1 排除', 'R1-P1 纠错', 'R1/R2 复核', '23 条坑', 'tests\\deepseek_temp\\']:
    out.append('    has %-38s -> %d' % (k, T2.count(k)))

# 3. daily log
p = os.path.join(root, '.workbuddy', 'memory', '2026-10-01.md')
T3 = open(p, encoding='utf-8').read()
out.append('[3] daily 2026-10-01.md bytes=%d ; has R2 section %s' % (os.path.getsize(p), '## 复核 R2' in T3))

# 4. skill
p = r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md'
T4 = open(p, encoding='utf-8').read()
out.append('[4] SKILL bytes=%d lines=%d sha8=%s' % (len(T4.encode('utf-8')), T4.count('\n') + 1, sha8(p)))
for k in ['23 条', 'tests/deepseek_temp/', 'seal 漂移纪律', '21. **', '22. **', '23. **', '5/5 适用 + 1 排除']:
    out.append('    has %-24s -> %d' % (k, T4.count(k)))

# 5. dest dirs
for d in ['tests/deepseek', 'tests/deepseek_temp']:
    dd = os.path.join(root, d)
    fs = [f for f in os.listdir(dd) if os.path.isfile(os.path.join(dd, f))]
    sub = [x for x in os.listdir(dd) if os.path.isdir(os.path.join(dd, x))]
    py = [f for f in fs if f.endswith('.py')]
    out.append('[5] %-20s files=%d (.py=%d) dirs=%s' % (d, len(fs), len(py), sub))

# 6. leftovers
import re
for tag, d in [('tests/gpt5_temp', os.path.join(root, 'tests', 'gpt5_temp')), ('root gpt5_temp', os.path.join(root, 'gpt5_temp'))]:
    left = [f for f in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, f))
            and re.search(r'^(e[123]b?_|n[123]|do_(append|memory|wlog)_|memo_append_|wlog_|probe_n2|probe_e4)', f)]
    out.append('[6] %-16s leftover=%d %s' % (tag, len(left), left))

# 7. review dir
rv = os.path.join(root, 'tests', 'deepseek', 'result', 'memo_review_20261001')
out.append('[7] review dir exists=%s files=%s' % (os.path.isdir(rv), len(os.listdir(rv)) if os.path.isdir(rv) else 0))

open(os.path.join(root, 'tests', 'deepseek_temp', 'verify_final_r2.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out))
