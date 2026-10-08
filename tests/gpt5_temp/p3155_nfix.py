# -*- coding: utf-8 -*-
# p3155_nfix.py: 修正 3155 文本中 ledger n=306 -> n=307（他线并发追加 P40 后的真实值）
import io, os
ROOT = r'D:\AI2050\Ai2050-OpenOne'
jobs = [
    (os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md'),
     'collect npz：`bc24f615`/`bc9e3297`/`6dd3c77f`。ledger n=**306**。',
     'collect npz：`bc24f615`/`bc9e3297`/`6dd3c77f`。ledger n=**307**（含他线 deepseek P40 并发条目，本线未动）。'),
    (os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md'),
     'KSTAR 指纹不稳（0.24–0.91，判定层=KOUT）；ledger n=**306**。',
     'KSTAR 指纹不稳（0.24–0.91，判定层=KOUT）；ledger n=**307**（他线 P40 并发未动）。'),
    (os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-08.md'),
     'IC、T 变换探索）。ledger n=306。',
     'IC、T 变换探索）。ledger n=307（他线 P40 并发未动）。'),
]
out = []
for path, old, new in jobs:
    txt = io.open(path, encoding='utf-8').read()
    c = txt.count(old)
    assert c == 1, ('count', path, c)
    io.open(path, 'w', encoding='utf-8').write(txt.replace(old, new))
    out.append('fixed %s' % os.path.basename(path))
# 复核
for path, old, new in jobs:
    assert new in io.open(path, encoding='utf-8').read()
    assert old not in io.open(path, encoding='utf-8').read()
out.append('recheck OK: all three files show n=307')
io.open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3155_nfix_out.txt'), 'w', encoding='utf-8').write(chr(10).join(out))
print('NFIX OK')
