# -*- coding: utf-8 -*-
"""Probe: list keys/shapes of p122_readout.npz + upstream npz."""
import io
import os

import numpy as np

RDIR = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
out = io.StringIO()
w = out.write

targets = [
    RDIR + r'\phase3124\omega_p122_resid_cm_kalman_'
    r'l35rel_glm4x\p122_readout.npz',
    RDIR + r'\phase3118\omega_p116_autoregressive_'
    r'margin_trajectory\traj_readout.npz',
    RDIR + r'\phase3120\omega_p118_content_attr_'
    r'amplifier_behavior_opshape\p118_readout.npz',
    RDIR + r'\phase3122\omega_p120_write_content_'
    r'readout_sentence_causal_dist_recon'
    r'\p120_readout.npz',
    RDIR + r'\phase3123\omega_p121_dirfit_anchor_'
    r'l35loc_syntax_trace\p121_readout.npz',
]
for tp in targets:
    w('=== %s ===\n' % os.path.basename(tp))
    if not os.path.exists(tp):
        w('MISSING\n\n')
        continue
    z = np.load(tp, allow_pickle=False)
    for k in z.files:
        a = z[k]
        w('  %s : %s %s\n' % (k, a.shape, a.dtype))
    w('\n')

with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\p3124_npz_keys.txt', 'w',
          encoding='utf-8') as f:
    f.write(out.getvalue())
print('WROTE_OK')
