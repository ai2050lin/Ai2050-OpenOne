# -*- coding: utf-8 -*-
"""Find phase3129 json/npz artifacts."""
import os
import io

root = r'D:\AI2050\Ai2050-OpenOne'
hits = []
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in
             ('.git', 'node_modules', '__pycache__', '.venv', 'models')]
    for f in fn:
        if '3129' in f and (f.endswith('.json') or f.endswith('.npz')):
            hits.append(os.path.join(dp, f))
out = root + r'\tests\gpt5_temp\p3129_files.txt'
with io.open(out, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(hits))
print('found', len(hits))
