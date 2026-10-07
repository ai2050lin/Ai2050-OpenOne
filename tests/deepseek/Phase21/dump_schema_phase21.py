# -*- coding: utf-8 -*-
"""dump result_phase21.json / execution_phase21.json 的结构（键路径 + 类型 + 标量样例），供 closeout_docs 渲染参考。"""
import io
import os
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P21T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase21')
OUT = os.path.join(P21T, '_schema_dump.txt')
o = []


def w(s=''):
    o.append(str(s))


def walk(cur, path, depth=0, maxdepth=4):
    if depth > maxdepth:
        w('  ' * depth + path + ' ... (truncated)')
        return
    if isinstance(cur, dict):
        for k, v in cur.items():
            walk(v, path + '.' + str(k), depth + 1, maxdepth)
    elif isinstance(cur, list):
        w('  ' * depth + path + ' : list(len=%d)' % len(cur))
        if cur:
            walk(cur[0], path + '[0]', depth + 1, maxdepth)
    else:
        s = repr(cur)
        if len(s) > 70:
            s = s[:70] + '...'
        w('  ' * depth + path + ' = ' + s)


for name in ['result_phase21.json', 'execution_phase21.json', 'N2h1a14_design_seal.json']:
    p = os.path.join(P21T, name)
    w('=' * 20 + ' ' + name + ' (bytes=%d)' % os.path.getsize(p))
    try:
        d = json.load(io.open(p, encoding='utf-8'))
        walk(d, '$', 0, 4)
    except Exception:
        e = None
        raise
    w('')

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('DUMP OK ->', OUT)
