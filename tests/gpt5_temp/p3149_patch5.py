# -*- coding: utf-8 -*-
"""p3149 patch5: _injp tuple guard.
Idempotent."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\glm5\phase3149_omega_p147_'
      r'carrier_dlogit_poslate_kdose_'
      r'v3amp.py')
s = io.open(FP, encoding='utf-8').read()

old = ('                    def _injp(mod, '
       'inp,\n'
       '                              out,\n'
       '                              '
       '_tp=_tail_p):\n'
       '                        o2 = out[0]\n'
       '                        o2[0, -1, '
       '_tp] += \\\n'
       '                            1.0 * '
       'DELTA_L17\n'
       '                        return None')
n = s.count(old)
assert n == 1, ('injp', n)
new = ('                    def _injp(mod, '
       'inp,\n'
       '                              out,\n'
       '                              '
       '_tp=_tail_p):\n'
       '                        o2 = out[0] '
       '\\\n'
       '                            if '
       'isinstance(\n'
       '                                '
       'out,\n'
       '                                '
       'tuple) \\\n'
       '                            else '
       'out\n'
       '                        o2[0, -1, '
       '_tp] += \\\n'
       '                            1.0 * '
       'DELTA_L17\n'
       '                        return None')
s = s.replace(old, new)
assert 'rev-3149b' not in s
s = s.rstrip() + ('\n# rev-3149b patch2: '
                  '_injp tuple guard '
                  '(L17 out is tensor -> '
                  'out[0] sliced dim0)\n')
io.open(FP, 'w', encoding='utf-8',
        newline='\n').write(s)
print('INJP_FIXED')
