# -*- coding: utf-8 -*-
import numpy as np

NPZ = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2953'
       r'\a11_s_response\a11_s_response.npz')
OUTTXT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2953_curves_probe.txt')

z = np.load(NPZ, allow_pickle=True)
g17 = z['grid17']
g16 = z['grid16']
keep17 = z['keep_L17']
keep16 = z['keep_L16']
out = []
out.append('keep17 %s' % keep17.tolist())
out.append('keep16 %s' % keep16.tolist())
for hh in [1, 20, 21, 22, 19, 0]:
    if hh in keep17.tolist():
        cur = [float(np.median(z['A11_L17|%.4f' % s][:, hh]))
               for s in g17]
        out.append('L17 h%d: base %.4f -> %s'
                   % (hh, float(np.median(z['A11b_L17'][:, hh])),
                      [round(v, 4) for v in cur]))
    else:
        out.append('L17 h%d NOT in keep' % hh)
for hh in [27, 17, 13, 31, 19, 26]:
    if hh in keep16.tolist():
        cur = [float(np.median(z['A11_L16|%.4f' % s][:, hh]))
               for s in g16]
        out.append('L16 h%d: base %.4f -> %s'
                   % (hh, float(np.median(z['A11b_L16'][:, hh])),
                      [round(v, 4) for v in cur]))
    else:
        out.append('L16 h%d NOT in keep' % hh)
med17 = ([float(np.median(z['A11b_L17'][:, keep17]))]
         + [float(np.median(z['A11_L17|%.4f' % s][:, keep17]))
            for s in g17])
med16 = ([float(np.median(z['A11b_L16'][:, keep16]))]
         + [float(np.median(z['A11_L16|%.4f' % s][:, keep16]))
            for s in g16])
out.append('L17 median-keep (s=0..2): %s'
           % [round(v, 4) for v in med17])
out.append('L16 median-keep (s=0..2): %s'
           % [round(v, 4) for v in med16])
with open(OUTTXT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
print('ok')
