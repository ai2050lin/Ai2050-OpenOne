# -*- coding: utf-8 -*-
import json, io
D34 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase3134'
       r'\omega_p132_carrier_matrix_forkcoord_stepscan')
r34 = json.load(io.open(D34 + r'\result.json', encoding='utf-8'))
pb = r34['part_b']
out = []
out.append('b_dose = %r' % (pb.get('b_dose'),))
out.append('type = %s' % type(pb.get('b_dose')).__name__)
out.append('chg_matrix keys = %s' % sorted(pb['chg_matrix'].keys()))
out.append('chg_matrix[38] = %r' % (pb['chg_matrix']['38'],))
# b_carrier / trials structure peek
bc = pb.get('b_carrier')
out.append('b_carrier type=%s' % type(bc).__name__)
if isinstance(bc, dict):
    out.append('b_carrier keys = %s' % sorted(bc.keys())[:12])
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3137_check34dose.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('OK')
