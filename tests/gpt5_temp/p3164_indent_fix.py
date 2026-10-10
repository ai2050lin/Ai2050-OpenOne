# -*- coding: utf-8 -*-
import io
P = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3164_closeout.py'
lines = io.open(P, encoding='utf-8').read().split('\n')
# 定位：从 "sect = '''" 行到 "log('2. MEMO" 行（含），整体缩进 4 格
i_start = next(i for i, l in enumerate(lines) if l.startswith('sect = '))
i_end = next(i for i, l in enumerate(lines) if l.startswith("log('2. MEMO"))
for i in range(i_start, i_end + 1):
    if lines[i].strip():
        lines[i] = '    ' + lines[i]
io.open(P, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
import py_compile
py_compile.compile(P, doraise=True)
print('INDENT PATCH OK (lines %d-%d)' % (i_start + 1, i_end + 1))
