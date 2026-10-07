# -*- coding: utf-8 -*-
import json, io
D38 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase3138'
       r'\omega_p136_statebank_bicdecomp_probecontrast')
r38 = json.load(io.open(D38 + r'\result.json', encoding='utf-8'))
ish17 = r38['part_c']['ish']['17']
out = ['ish17 = %r' % ish17]
out.append('csh17 = %r' % r38['part_c']['csh']['17'])
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3139_probeish.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('OK')
