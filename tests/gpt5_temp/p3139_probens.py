# -*- coding: utf-8 -*-
import json, io
D38 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase3138'
       r'\omega_p136_statebank_bicdecomp_probecontrast')
r38 = json.load(io.open(D38 + r'\result.json', encoding='utf-8'))
ns17 = r38['part_c']['norm_share']['17']
out = ['ns17 = %r' % ns17]
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3139_probens.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('OK')
