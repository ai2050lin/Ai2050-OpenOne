# -*- coding: utf-8 -*-
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase2995_omega_f1_glm4_panel.py')
s = io.open(P, encoding='utf-8').read()
anchor = ("    g1 = g2 = g3 = None\n"
          "    if not anchor_ok:")
assert anchor in s, 'anchor missing'
assert 'proj_top = np.zeros(n_cells)' not in s, \
    'already patched'
s = s.replace(anchor,
              "    g1 = g2 = g3 = None\n"
              "    proj_top = np.zeros(n_cells)\n"
              "    if not anchor_ok:", 1)
io.open(P, 'w', encoding='utf-8').write(s)
print('patched')
