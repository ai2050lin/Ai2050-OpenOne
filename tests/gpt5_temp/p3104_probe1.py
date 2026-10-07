# -*- coding: utf-8 -*-
import os, json
root = r'D:\AI2050\Ai2050-OpenOne\models\hf'
out = []
for name in sorted(os.listdir(root)):
    p = os.path.join(root, name)
    is_dir = os.path.isdir(p)
    n_files = 0
    if is_dir:
        try:
            n_files = len(os.listdir(p))
        except OSError as e:
            n_files = -1
    out.append({'name': name, 'isdir': is_dir, 'n_entries': n_files})
q4 = os.path.join(root, 'qwen3-4b')
detail = []
if os.path.isdir(q4):
    for f in sorted(os.listdir(q4)):
        fp = os.path.join(q4, f)
        detail.append({'f': f, 'size': os.path.getsize(fp) if os.path.isfile(fp) else -1})
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3104_probe1_out.txt', 'w', encoding='utf-8') as fh:
    fh.write(json.dumps({'dirs': out, 'qwen3_4b_files': detail}, indent=1))
print('ok')
