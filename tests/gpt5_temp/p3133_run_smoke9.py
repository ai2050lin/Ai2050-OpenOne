# -*- coding: utf-8 -*-
"""Phase 3133 SMOKE launcher (R9: keep
ckpt for the R10 full-RESUME test)."""
import os
import runpy
import sys

os.environ['P3133_SMOKE'] = '1'
os.environ['P3133_CKPT_KEEP'] = '1'
sys.argv = [r'D:\AI2050\Ai2050-OpenOne'
            r'\tests\glm5'
            r'\phase3133_omega_p131_'
            r'transplant_a1fork_migrate.py']
runpy.run_path(sys.argv[0],
               run_name='__main__')
