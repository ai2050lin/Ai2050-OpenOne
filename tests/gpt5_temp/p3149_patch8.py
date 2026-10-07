# -*- coding: utf-8 -*-
"""p3149 patch8: kmin via fstep_store
(gens not persisted in cut trials).
Idempotent."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\glm5\phase3149_omega_p147_'
      r'carrier_dlogit_poslate_kdose_'
      r'v3amp.py')
s = io.open(FP, encoding='utf-8').read()

old = ("    L_cuts = {}\n"
       "    for jj, j in "
       "enumerate(flp_rows):\n"
       "        kmin = -1\n"
       "        for cut in L_CUTS:\n"
       "            g = E_res['l_cut%d'\n"
       "                      % cut]"
       "['gens'][jj]\n"
       "            if pad12(g) != "
       "base12_A1[j]:\n"
       "                kmin = cut\n"
       "                break\n"
       "        L_cuts[str(j)] = kmin")
n = s.count(old)
assert n == 1, ('kmin', n)
new = ("    L_cuts = {}\n"
       "    for jj, j in "
       "enumerate(flp_rows):\n"
       "        kmin = -1\n"
       "        for cut in L_CUTS:\n"
       "            fs_c = fstep_store[\n"
       "                'l_cut%d' % cut]\n"
       "            if fs_c[jj] >= 0:\n"
       "                kmin = cut\n"
       "                break\n"
       "        L_cuts[str(j)] = kmin")
s = s.replace(old, new)
assert 'rev-3149e' not in s
s = s.rstrip() + ('\n# rev-3149e patch5: '
                  'kmin via fstep_store\n')
io.open(FP, 'w', encoding='utf-8',
        newline='\n').write(s)
print('KMIN_FIXED')
