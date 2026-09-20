# -*- coding: utf-8 -*-
# Phase 2963 patch 2: fix T2 permutation loop syntax.
import ast
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase2963_frequency_controlled_band.py')
t = io.open(P, encoding='utf-8').read()

old = (
    "        for _ in range(N_PERM):\n"
    "            if abs(spearman(\n"
    "                    B_all[ix_c],\n"
    "                    rng2.permutation(tid_test[ix_c])))\n"
    "                    >= abs(rho2) - 1e-12:\n"
    "                cnt2 += 1\n")
new = (
    "        for _ in range(N_PERM):\n"
    "            v2 = abs(spearman(\n"
    "                B_all[ix_c],\n"
    "                rng2.permutation(tid_test[ix_c])))\n"
    "            if v2 >= abs(rho2) - 1e-12:\n"
    "                cnt2 += 1\n")

assert old in t, 'T2 loop not found'
t = t.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(t)

t2 = io.open(P, encoding='utf-8').read()
try:
    ast.parse(t2)
    r = 'syntax OK'
except SyntaxError as e:
    r = 'SYNTAX ERROR line %s col %s: %s' % (
        e.lineno, e.offset, e.msg)
out = io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
              r'\p2963_syntax2.txt', 'w', encoding='utf-8')
out.write(r + '\n')
out.close()
print('patched')
