# -*- coding: utf-8 -*-
"""Dump 3125 result tree + npz key inventories for 3127 interface check."""
import json
import io
import os
import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = (r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
       r'\p3127_frozen_keys.txt')

f = io.open(OUT, 'w', encoding='utf-8')


def w(s):
    f.write(s + '\n')


# ---- 3125 result tree ----
def find_in(phase_dir, fname):
    """Locate fname inside phase_dir (recursing one level,
    skipping 'smoke' subdirs)."""
    direct = os.path.join(phase_dir, fname)
    if os.path.exists(direct):
        return direct
    for nm in sorted(os.listdir(phase_dir)):
        if nm == 'smoke':
            continue
        cand = os.path.join(phase_dir, nm, fname)
        if os.path.exists(cand):
            return cand
    return None


p25 = os.path.join(BASE, 'phase3125')
d25 = find_in(p25, 'result.json')
r = json.load(io.open(d25, encoding='utf-8'))
w('=== 3125 result tree (%s) ===' % d25)


def walk(d, p='', depth=0):
    if depth > 4:
        w(p + '  ...')
        return
    if isinstance(d, dict):
        for k in d:
            walk(d[k], p + '/' + str(k), depth + 1)
    elif isinstance(d, list):
        w(p + '  [list len=%d]' % len(d))
    else:
        s = repr(d)
        if len(s) > 70:
            s = s[:70] + '...'
        w(p + '  = ' + s)


walk(r)

# ---- npz key inventories ----
for tag, ph, fname in (
        ('z18-p116', 'phase3118', 'traj_readout.npz'),
        ('z20-p118', 'phase3120', 'p118_readout.npz'),
        ('z22-p120', 'phase3122', 'p120_readout.npz'),
        ('z24-p122', 'phase3124', 'p122_readout.npz'),
        ('z25-p123', 'phase3125', 'p123_readout.npz'),
        ('z26-p124', 'phase3126', 'p124_readout.npz'),
        ('capb', 'phase3113', 'capture_b.npz')):
    path = find_in(os.path.join(BASE, ph), fname)
    if path is None:
        w('!! missing npz: %s/%s' % (ph, fname))
        continue
    z = np.load(path, allow_pickle=False)
    w('')
    w('=== %s (%s) ===' % (tag, path))
    for k in sorted(z.files):
        w('  %-24s %s %s' % (k, z[k].shape, z[k].dtype))

# ---- mat5 keys ----
m5 = os.path.join(BASE, 'phase3105',
                  'omega_p103_incontext_truth_consistency',
                  'material.json')
if os.path.exists(m5):
    mm = json.load(io.open(m5, encoding='utf-8'))
    w('')
    w('=== mat5 (%s) ===' % m5)
    for k in sorted(mm.keys()):
        v = mm[k]
        try:
            w('  %-16s %s len=%d' % (k, type(v).__name__,
                                     len(v)))
        except TypeError:
            w('  %-16s %s = %r' % (k, type(v).__name__, v))
else:
    w('!! missing material.json: ' + m5)
f.close()
print('KEYS_OK')
