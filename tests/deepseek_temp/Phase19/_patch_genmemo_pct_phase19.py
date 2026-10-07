# -*- coding: utf-8 -*-
"""补丁：gen_memo 里两处 %% 出现在**无 % 操作符**的字符串中 => 会原样输出 %% 。改为单 %。"""
import io

p = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase19\gen_memo_phase19.py'
s = io.open(p, encoding='utf-8').read()

for old, new in [('加载 19%% 时 segfault**', '加载 19% 时 segfault**'),
                 ('（~19%% 权重）', '（~19% 权重）')]:
    assert s.count(old) == 1, 'count(%r)=%d' % (old, s.count(old))
    s = s.replace(old, new)

io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched OK')
print('has 19%% left:', '19%%' in s)
