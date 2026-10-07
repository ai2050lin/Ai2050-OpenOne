# -*- coding: utf-8 -*-
"""Dump the key tree of 3126 result.json for 3127 interface alignment."""
import json
import io

P = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase3126\omega_p124_glm4_anchoredlast_regen_writechain\result.json'
OUT = r'D:\AI2050\Ai2050-OpenOne\gpt5_temp\p3126_result_tree.txt'

r = json.load(io.open(P, encoding='utf-8'))
out = []


def walk(d, p='', depth=0):
    if depth > 4:
        out.append(p + '  ...')
        return
    if isinstance(d, dict):
        for k in d:
            walk(d[k], p + '/' + str(k), depth + 1)
    elif isinstance(d, list):
        out.append(p + '  [list len=%d]' % len(d))
    else:
        s = repr(d)
        if len(s) > 80:
            s = s[:80] + '...'
        out.append(p + '  = ' + s)


walk(r)
f = io.open(OUT, 'w', encoding='utf-8')
f.write('\n'.join(out))
f.close()
print('TREE_OK', len(out))
