# -*- coding: utf-8 -*-
import os, json
root = r'D:\AI2050\Ai2050-OpenOne'
out = []
def w(s): out.append(str(s))
h = os.path.join(root, 'models', 'hf')
for d in sorted(os.listdir(h)):
    dd = os.path.join(h, d)
    if not os.path.isdir(dd): continue
    fs = []
    for f in sorted(os.listdir(dd)):
        p = os.path.join(dd, f)
        fs.append('%s %.2fGB' % (f, os.path.getsize(p) / 1e9) if os.path.isfile(p) else f + '/')
    w('== %s' % d)
    for f in fs: w('   ' + f)
open(os.path.join(root, 'gpt5_temp', 'probe_models2.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('ok')
