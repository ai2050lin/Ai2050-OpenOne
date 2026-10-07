# -*- coding: utf-8 -*-
"""Find the deepseek-r1-distill-qwen-7b model dir (2 shards,
15.231 GB, config hidden=3584 layers=28 vocab=152064)."""
import os
import json

OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp' \
      '\p3081_find_ds7b.txt'
lines = []


def add(s):
    lines.append(str(s))


roots = [
    r'D:\AI2050',
    r'D:\models',
    r'D:\hf',
    r'C:\models',
    r'C:\Users\Admin\.cache\huggingface\hub',
]
seen = set()
for root in roots:
    if not os.path.isdir(root):
        add('MISSING root: ' + root)
        continue
    add('== root: ' + root)
    try:
        for dirpath, dirnames, filenames in \
                os.walk(root):
            depth = dirpath[len(root):] \
                .count(os.sep)
            if depth >= 4:
                dirnames[:] = []
                continue
            low = dirpath.lower()
            if 'deepseek' in low \
                    and 'seen_ds7b' not in seen:
                n = len(filenames)
                tot = sum(
                    os.path.getsize(os.path.join(
                        dirpath, f))
                    for f in filenames
                    if f.endswith('.safetensors'))
                add('DEEPSEEK HIT: ' + dirpath
                    + ' files=%d st_total=%.3f GB'
                    % (n, tot / 1e9))
                seen.add('seen_ds7b')
            for d in list(dirnames):
                if d.lower() == 'hf' or \
                        'hf' == d.lower():
                    hd = os.path.join(dirpath, d)
                    try:
                        add('HF DIR: ' + hd
                            + ' -> '
                            + repr(os.listdir(hd)))
                    except OSError as e:
                        add('HF DIR err: ' + str(e))
    except OSError as e:
        add('walk err ' + root + ': ' + str(e))

add('')
add('== models/hf full listing ==')
mh = r'D:\AI2050\Ai2050-OpenOne\models\hf'
for d in sorted(os.listdir(mh)):
    p = os.path.join(mh, d)
    if os.path.isdir(p):
        fs = os.listdir(p)
        tot = 0.0
        for f in fs:
            fp = os.path.join(p, f)
            if os.path.isfile(fp) and \
                    f.endswith('.safetensors'):
                tot += os.path.getsize(fp) / 1e9
        add('%s  entries=%d  st=%.3f GB'
            % (d, len(fs), tot))

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('done')
