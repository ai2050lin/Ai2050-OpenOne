# -*- coding: utf-8 -*-
"""Patch2: set(None) TypeError in ABL layers."""
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3122_omega_p120_write_content_readout_'
     r'sentence_causal_dist_recon.py')
src = io.open(P, encoding='utf-8').read()

old = "    ABL['layers'] = set(layers_abl)"
new = ("    ABL['layers'] = set(layers_abl) "
       "if layers_abl else set()")
c = src.count(old)
assert c == 2, 'expect 2 occurrences, got %d' % c
src = src.replace(old, new)

py_compile.compile(P, doraise=True)
io.open(P, 'w', encoding='utf-8').write(src)
msg = 'patch2 OK: %d replaced; COMPILE_OK' % c
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\p3122_patch2_out.txt', 'w',
        encoding='utf-8').write(msg + chr(10))
print(msg)
