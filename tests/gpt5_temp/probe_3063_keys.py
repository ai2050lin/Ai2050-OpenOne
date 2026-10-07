import numpy as np
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')

BANKS = [
    ('z48', 'phase3048', 'omega_p45_kvpos_full_replay_qwen',
     'omega_p45_kvpos_full_replay_qwen.npz'),
    ('z51', 'phase3051', 'omega_p48_l35_anatomy_qwen',
     'omega_p48_l35_anatomy_qwen.npz'),
    ('z53', 'phase3053', 'omega_p50_gate_source_qwen',
     'omega_p50_gate_source_qwen.npz'),
    ('z54', 'phase3054', 'omega_p51_norm_projection_qwen',
     'omega_p51_norm_projection_qwen.npz'),
    ('z59', 'phase3059', 'omega_p56_payload_subspace_qwen',
     'omega_p56_payload_subspace_qwen.npz'),
    ('z60', 'phase3060', 'omega_p57_pc1_identity_qwen',
     'omega_p57_pc1_identity_qwen.npz'),
    ('z61', 'phase3061', 'omega_p58_write_highdim_qwen',
     'omega_p58_write_highdim_qwen.npz'),
]

out = []
for tag, ph, nm, fn in BANKS:
    p = os.path.join(BASE, ph, nm, fn)
    if not os.path.isfile(p):
        out.append('%s MISSING %s' % (tag, p))
        continue
    z = np.load(p, allow_pickle=True)
    out.append('===== %s (%s) =====' % (tag, fn))
    for k in z.files:
        try:
            a = z[k]
            out.append('  %-24s %-12s %s'
                       % (k, str(a.shape), a.dtype))
        except Exception as e:
            out.append('  %-24s ERR %s' % (k, e))
    z.close()

with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\probe_3063_keys.txt',
          'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('OK')
