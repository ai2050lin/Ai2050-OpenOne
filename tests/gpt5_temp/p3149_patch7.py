# -*- coding: utf-8 -*-
"""p3149 patch7: L cut gen pass row
contents (not row indices). Idempotent."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\glm5\phase3149_omega_p147_'
      r'carrier_dlogit_poslate_kdose_'
      r'v3amp.py')
s = io.open(FP, encoding='utf-8').read()

old = ("    for cut in L_CUTS:\n"
       "        _run_coord_trial(\n"
       "            'l_cut%d' % cut, "
       "_tail_p, 1,\n"
       "            1.0, flp_rows, "
       "base12_A1,\n"
       "            mode='cut%d' % cut)")
n = s.count(old)
assert n == 1, ('lcut', n)
new = ("    for cut in L_CUTS:\n"
       "        _run_coord_trial(\n"
       "            'l_cut%d' % cut, "
       "_tail_p, 1,\n"
       "            1.0,\n"
       "            [rows_A1[j]\n"
       "             for j in flp_rows],\n"
       "            [base12_A1[j]\n"
       "             for j in flp_rows],\n"
       "            mode='cut%d' % cut)")
s = s.replace(old, new)
assert 'rev-3149d' not in s
s = s.rstrip() + ('\n# rev-3149d patch4: L '
                  'cut gen passes row '
                  'contents not row '
                  'indices\n')
io.open(FP, 'w', encoding='utf-8',
        newline='\n').write(s)
print('LCUT_FIXED')
