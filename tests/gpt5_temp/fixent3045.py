# -*- coding: utf-8 -*-
import io

FILES = [
    (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
     r'\AGI_GPT5_MEMO.md', 'memo'),
    (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
     r'\hdmcc_knowledge_map_review_20260921.md',
     'audit'),
    (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md', 'memory'),
    (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\2026-09-21.md', 'wlog'),
]
o = []
for path, tag in FILES:
    s = io.open(path, encoding='utf-8').read()
    n = (s.count('&lt;') + s.count('&gt;')
         + s.count('&#123;') + s.count('&#125;'))
    if n == 0:
        o.append('%s: clean' % tag)
        continue
    s = (s.replace('&lt;', '<').replace('&gt;', '>')
         .replace('&#123;', '{').replace('&#125;', '}'))
    io.open(path, 'w', encoding='utf-8').write(s)
    s2 = io.open(path, encoding='utf-8').read()
    assert (s2.count('&lt;') + s2.count('&gt;')
            + s2.count('&#123;')
            + s2.count('&#125;')) == 0, tag
    o.append('%s: fixed %d' % (tag, n))
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\fixent3045_result.txt', 'w',
        encoding='utf-8').write('\n'.join(o) + '\n')
print('fixent ok')
