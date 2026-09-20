# -*- coding: utf-8 -*-
# Phase 2964 patch: func_tid taken directly from tokenizer.
import ast
import io
import shutil
import os

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase2964_carrier_anatomy.py')
t = io.open(P, encoding='utf-8').read()

old = ("    func_tid = tid_map['the']\n")
new = ("    ids_the = tok(' the', add_special_tokens=False)[\n"
       "        'input_ids']\n"
       "    assert len(ids_the) == 1\n"
       "    func_tid = int(ids_the[0])\n")
assert old in t, 'anchor not found'
t = t.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(t)

t2 = io.open(P, encoding='utf-8').read()
try:
    ast.parse(t2)
    r = 'syntax OK'
except SyntaxError as e:
    r = 'SYNTAX ERROR line %s: %s' % (e.lineno, e.msg)
d = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase2964')
if os.path.exists(d):
    shutil.rmtree(d)
clean = not os.path.exists(d)
out = io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
              r'\p2964_patch.txt', 'w', encoding='utf-8')
out.write('%s\ncleaned: %s\n' % (r, clean))
out.close()
print('patched')
