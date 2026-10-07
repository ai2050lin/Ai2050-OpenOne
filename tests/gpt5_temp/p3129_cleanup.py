# -*- coding: utf-8 -*-
"""Clean smoke dir + leftover check + recompile."""
import io
import os
import shutil
import psutil
import py_compile

root = r'D:\AI2050\Ai2050-OpenOne'
smk = os.path.join(
    root, r'tests\glm5\result'
    r'\rdc_query_construction_20260913'
    r'\phase3129'
    r'\omega_p127_dose_sweep_gen_'
    r'decouple_s0full_symbolfield\smoke')
if os.path.isdir(smk):
    shutil.rmtree(smk)
left = []
for p in psutil.process_iter(
        ['pid', 'name', 'cmdline']):
    try:
        j = ' '.join(str(x) for x in
                     (p.info['cmdline'] or []))
        if 'python' in (p.info['name']
                        or '').lower() \
                and '3129' in j:
            left.append('pid=%s'
                        % p.info['pid'])
    except Exception:
        pass
py_compile.compile(
    os.path.join(root, r'tests\glm5'
                 r'\phase3129_omega_p127_dose_'
                 r'sweep_gen_decouple_s0full_'
                 r'symbolfield.py'), doraise=True)
gpu = os.popen(
    r'"C:\Windows\System32\nvidia-smi.exe"'
    ' --query-gpu=memory.used'
    ' --format=csv,noheader').read().strip()
with io.open(os.path.join(
        root, r'tests\gpt5_temp'
        r'\p3129_cleanup.txt'), 'w') as f:
    f.write('smoke_dir_removed=%s\n'
            'leftover=%s\ngpu=%s\n'
            'COMPILE_OK'
            % (not os.path.isdir(smk),
               left or 'none', gpu))
print('CLEANUP_OK')
