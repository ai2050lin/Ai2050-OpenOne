# -*- coding: utf-8 -*-
"""Phase 3134 SMOKE launcher."""
import os
import runpy
import sys

os.environ['P3134_SMOKE'] = '1'
sys.argv = [r'D:\AI2050\Ai2050-OpenOne'
            r'\tests\glm5'
            r'\phase3134_omega_p132_'
            r'carrier_matrix_forkcoord_'
            r'stepscan.py']
runpy.run_path(sys.argv[0],
               run_name='__main__')
