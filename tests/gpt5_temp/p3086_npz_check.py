# -*- coding: utf-8 -*-
"""p3086 npz check: verify authoritative npz keys and values."""
import numpy as np
import io

R = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913'
     r'\phase3086\omega_p83_continuum_test')
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3086_npz_check.txt'

z = np.load(R + r'\omega_p83_continuum_test.npz',
            allow_pickle=False)
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('VERDICT=%s SMOKE=%s N_PERM=%d '
            'SETUP_OK=%s N_UNITS=%d\n'
            % (str(z['VERDICT']), bool(z['SMOKE']),
               int(z['N_PERM']), bool(z['SETUP_OK']),
               int(z['N_UNITS'])))
    f.write('ANCH A1=%s A2=%s A3=%s\n'
            % (bool(z['ANCH_A1']), bool(z['ANCH_A2']),
               bool(z['ANCH_A3'])))
    f.write('rho_T=%.6f p_T=%.6f | rho_U=%.6f '
            'p_U=%.6f | rho_MIG=%.6f p_MIG=%.6f\n'
            % (float(z['RHO_T']), float(z['P_T']),
               float(z['RHO_U']), float(z['P_U']),
               float(z['RHO_MIG']),
               float(z['P_MIG'])))
    f.write('sp_mean T=%.4f U=%.4f MIG=%.4f\n'
            % (float(z['SP_MEAN_T']),
               float(z['SP_MEAN_U']),
               float(z['SP_MEAN_MIG'])))
    f.write('unit keys:\n')
    for k in sorted(z.files):
        if k.startswith(('S_LO_', 'S_MEAN_',
                         'T_MED_', 'U_MED_',
                         'MIG_M')):
            f.write('  %s = %.6f\n'
                    % (k, float(z[k])))
