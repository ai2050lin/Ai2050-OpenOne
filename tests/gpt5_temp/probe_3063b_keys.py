# 3063 pre-run key probe v2: correct BASE
import numpy as np
import os

OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\probe_3063c_keys.txt'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
lines = []


def probe(tag, sub, fn, keys):
    p = os.path.join(BASE, sub, fn)
    lines.append('===== %s =====' % tag)
    if not os.path.exists(p):
        lines.append('  MISSING: ' + p)
        return
    z = np.load(p, allow_pickle=True)
    for k in keys:
        if k in z.files:
            a = z[k]
            lines.append('  %-24s %-14s %s' %
                         (k, str(a.shape), a.dtype))
        else:
            lines.append('  %-24s ABSENT' % k)


probe('z55', r'phase3055\omega_p52_gamma_prealign_qwen',
      'omega_p52_gamma_prealign_qwen.npz', ['GAMMA_STATS'])
probe('z58', r'phase3058\omega_p55_payload_channel_identity_qwen',
      'omega_p55_payload_channel_identity_qwen.npz',
      ['TOP64', 'FLAT64', 'COAL_TOP64'])
probe('z22', r'phase3022\omega_p2p_l3_relay_neurons_qwen',
      'omega_p2p_l3_relay_neurons_qwen.npz', ['s_relay'])
probe('z2802', r'phase2802\qwen4_polysemy_spectrum',
      'polysemy.npz', ['sig', 'votes'])

with open(OUT, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('WROTE', OUT)
