# -*- coding: utf-8 -*-
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3042_omega_p39_style_field_probe_qwen.py')
s = io.open(P, encoding='utf-8').read()

old1 = ("        log('cond=%d body=%d pos=%d len=%d %r'\n"
        "            % (ci, bi, ids.index(t), len(ids)))")
new1 = ("        log('cond=%d body=%d pos=%d len=%d %r'\n"
        "            % (ci, bi, ids.index(t), len(ids),\n"
        "               s))")
assert s.count(old1) == 1, ('old1', s.count(old1))
s = s.replace(old1, new1)

old2 = "    'corrections': 'none (first run)',"
new2 = ("    'corrections': 'run1 crashed PRE-verdict '\n"
        "                   '(assembly logging only, '\n"
        "                   'no test statistic '\n"
        "                   'observed): log format '\n"
        "                   'string had 5 placeholders '\n"
        "                   'but 4 arguments (missing '\n"
        "                   'arg for the '\n"
        "                   'assembled-prompt %r); '\n"
        "                   'corrected to pass the '\n"
        "                   'assembled string; verdict '\n"
        "                   'tree unchanged',")
assert s.count(old2) == 1, ('old2', s.count(old2))
s = s.replace(old2, new2)

io.open(P, 'w', encoding='utf-8').write(s)

import py_compile
py_compile.compile(P, doraise=True)
with open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
          r'\patch3042_result.txt', 'w',
          encoding='utf-8') as f:
    f.write('patch ok, compile ok')
print('done')
