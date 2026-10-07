# -*- coding: utf-8 -*-
import json, io
D37 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase3137'
       r'\omega_p135_modecoop_k0anat_coorddecomp_l35recheck')
r37 = json.load(io.open(D37 + r'\result.json', encoding='utf-8'))
pe = r37['part_e']
out = []
out.append('xphase = %r' % pe['xphase_base_match_P'])
out.append('scales = %s' % json.dumps(pe['scales']))
out.append('e_mode = %s' % json.dumps(pe['e_mode'], sort_keys=True))
out.append('e_po = %s' % json.dumps(pe['e_po'], sort_keys=True))
out.append('mode_matrix = %s' % json.dumps(pe['mode_matrix'], sort_keys=True))
out.append('a672 = %r' % pe['a672_l38_po_s2'])
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3137_pe.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('OK')
