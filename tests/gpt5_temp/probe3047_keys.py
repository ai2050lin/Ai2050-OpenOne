# -*- coding: utf-8 -*-
import io
import numpy as np

P = r'D:\AI2050\Ai2050-OpenOne\gpt5_temp\probe3047_keys.txt'
io.open(P, 'w', encoding='utf-8').write('')

def w(s):
    with io.open(P, 'a', encoding='utf-8') as f:
        f.write(s + '\n')

base = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
for tag, rel in [
    ('z46', base + r'\phase3046\omega_p43_kfield_injection_qwen'
             r'\omega_p43_kfield_injection_qwen.npz'),
    ('z45', base + r'\phase3045\omega_p42_l20_axis_anatomy_qwen'
             r'\omega_p42_l20_axis_anatomy_qwen.npz'),
]:
    try:
        z = np.load(rel, allow_pickle=True)
        w('=== %s ===' % tag)
        for k in z.files:
            a = z[k]
            w('  %-28s %s %s' % (k, a.shape, a.dtype))
    except Exception as e:
        w('%s ERROR %r' % (tag, e))
w('done')
