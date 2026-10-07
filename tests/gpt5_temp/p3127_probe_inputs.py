# -*- coding: utf-8 -*-
# p3127_probe_inputs.py: inventory frozen npz
# keys/shapes/dtypes for Phase 3127 design.
import io
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
VF = (ROOT + r'\tests\gpt5_temp'
      r'\p3127_probe_inputs_out.txt')
o = []
TGTS = {
    'capture_b(3113)': os.path.join(
        RDIR, 'phase3113',
        'omega_p111_artifact_writein',
        'capture_b.npz'),
    'p120(3122)': os.path.join(
        RDIR, 'phase3122',
        'omega_p120_write_content_readout_'
        'sentence_causal_dist_recon',
        'p120_readout.npz'),
    'traj(3118)': os.path.join(
        RDIR, 'phase3118',
        'omega_p116_autoregressive_margin_'
        'trajectory', 'traj_readout.npz'),
    'p118(3120)': os.path.join(
        RDIR, 'phase3120',
        'omega_p118_content_attr_amplifier_'
        'behavior_opshape', 'p118_readout.npz'),
    'p122(3124)': os.path.join(
        RDIR, 'phase3124',
        'omega_p122_resid_cm_kalman_l35rel_'
        'glm4x', 'p122_readout.npz'),
    'p123(3125)': os.path.join(
        RDIR, 'phase3125',
        'omega_p123_third_comp_qwen_'
        'inputstream', 'p123_readout.npz'),
    'p124(3126)': os.path.join(
        RDIR, 'phase3126',
        'omega_p124_glm4_anchoredlast_'
        'regen_writechain', 'p124_readout.npz')}
for tag, pth in TGTS.items():
    o.append('==== %s ====' % tag)
    try:
        z = np.load(pth, allow_pickle=False)
        for k in sorted(z.files):
            a = z[k]
            o.append('  %-42s %-12s %s'
                     % (k, str(a.dtype),
                        str(a.shape)))
    except Exception as e:
        o.append('  LOAD FAIL: %r' % e)
io.open(VF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('probe done')
