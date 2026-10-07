# -*- coding: utf-8 -*-
"""p3149 patch6: L d131 scalar dot fix.
Idempotent."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\glm5\phase3149_omega_p147_'
      r'carrier_dlogit_poslate_kdose_'
      r'v3amp.py')
s = io.open(FP, encoding='utf-8').read()

old = ('            d131 = float(\n'
       '                ((hi - hb)\n'
       '                 @ w131)[0,\n'
       '                            '
       'SPEC_TOKS[0]]\n'
       '                .cpu())')
n = s.count(old)
assert n == 1, ('d131', n)
new = ('            d131 = float(\n'
       '                ((hi - hb) @ w131)\n'
       '                .cpu())')
s = s.replace(old, new)
assert 'rev-3149c' not in s
s = s.rstrip() + ('\n# rev-3149c patch3: L '
                  'd131 scalar dot (1D inner '
                  'product -> 0-dim tensor)\n')
io.open(FP, 'w', encoding='utf-8',
        newline='\n').write(s)
print('D131_FIXED')
