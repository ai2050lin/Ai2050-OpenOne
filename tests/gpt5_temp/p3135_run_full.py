# -*- coding: utf-8 -*-
"""FULL launcher for Phase 3135
(no P3135_SMOKE -> SMOKE=False)."""
import os
import runpy
import sys

os.environ.pop('P3135_SMOKE', None)
sys.argv = ['phase3135']
runpy.run_path(
    r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
    r'\phase3135_omega_p133_conduction_'
    r'co36ablation_window.py',
    run_name='__main__')
