import io, os
import numpy as np

RDIR = (r'D:\AI2050\Ai2050-OpenOne'
        r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
BASE = RDIR
out = []


def find_in(ph, fname):
    p = os.path.join(BASE, ph)
    if not os.path.isdir(p):
        return None
    for root, dirs, files in os.walk(p):
        if fname in files:
            return os.path.join(root, fname)
    return None


# w sources: 3110/3111/3112 refit readout w
for ph in ('phase3110', 'phase3111',
           'phase3112'):
    p = os.path.join(BASE, ph)
    if not os.path.isdir(p):
        out.append('!! missing dir ' + ph)
        continue
    for root, dirs, files in os.walk(p):
        for fn in files:
            if fn.endswith('.npz'):
                fp = os.path.join(root, fn)
                z = np.load(fp,
                            allow_pickle=False)
                wkeys = [k for k in z.files
                         if ('w' in k.lower()
                             or 'coef' in k.lower()
                             or 'probe' in k.lower()
                             or 'readout' in k.lower())]
                out.append('%s / %s: %s'
                           % (ph, fn,
                              ['%s%s'
                               % (k, z[k].shape)
                               for k in wkeys][:14]))
                out.append('   ALL: %s'
                           % sorted(z.files)[:18])
# 3127 npz quick shape recap
z7 = np.load(
    RDIR + (r'\phase3127'
            r'\omega_p125_writechain_port_'
            'crossmodel_a1closure_fullregen'
            r'\p125_readout.npz'),
    allow_pickle=False)
out.append('p125 regen keys: %s'
           % [k for k in sorted(z7.files)
              if k.startswith('regen')]
           [:10])
out.append('p125 regen shape: %s %s'
           % (z7['regen_s1_P'].shape,
              z7['regenidx_s1_P'].shape))
# p123/p124 spans for make_materials
z25 = np.load(
    find_in('phase3125', 'p123_readout.npz'),
    allow_pickle=False)
out.append('p123 span_idx shape: %s'
           % (z25['span_idx_P'].shape,))
io.open(r'D:\AI2050\Ai2050-OpenOne'
        r'\gpt5_temp\p3128_probe.txt',
        'w', encoding='utf-8').write(
    chr(10).join(out))
print('PROBE_OK')
