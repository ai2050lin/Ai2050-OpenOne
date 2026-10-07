# -*- coding: utf-8 -*-
import numpy as np
import io

P37 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase3037'
       r'\omega_p34_kv_situational_specificity_qwen'
       r'\omega_p34_kv_situational_specificity_qwen.npz')
z = np.load(P37, allow_pickle=True)
out = ['=== z37 keys ===']
for k in sorted(z.files):
    out.append('%s : %s %s' % (k, z[k].shape, z[k].dtype))
out.append('')
for ph, nm in ((3043, 'omega_p40_field_variance_qwen'),
               (3044, 'omega_p41_field_axis_injection_qwen'),
               (3045, 'omega_p42_l20_axis_anatomy_qwen')):
    p = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
         r'\rdc_query_construction_20260913\phase%d' % ph
         + '\\' + nm + '\\' + nm + '.npz')
    zz = np.load(p, allow_pickle=True)
    out.append('=== z%d keys ===' % ph)
    for k in sorted(zz.files):
        out.append('%s : %s %s' % (k, zz[k].shape,
                                   zz[k].dtype))
    out.append('')
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp\probe3046_keys.txt',
        'w', encoding='utf-8').write('\n'.join(out))
print('ok')
