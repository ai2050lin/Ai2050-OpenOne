"""Phase 2860 R1-failure diagnosis: word-level npz cross-compare."""
import glob
import json
import os

import numpy as np

B55 = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result' \
      r'\rdc_query_construction_20260913\phase2855\window_anatomy'
B60 = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result' \
      r'\rdc_query_construction_20260913\phase2860\g_direct'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_p2860_diag_report.txt'

lines = []


def w(s):
    lines.append(s)


for tag, d in (('2855', B55), ('2860', B60)):
    w('[%s] files: %s' % (tag, sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(d, '*')))))

z55 = np.load(glob.glob(os.path.join(B55, '*.npz'))[0])
z60 = np.load(glob.glob(os.path.join(B60, '*.npz'))[0])
w('[2855] npz keys: %s' % sorted(z55.files))
w('[2860] npz keys: %s' % sorted(z60.files))
for k in sorted(z55.files):
    w('[2855] %-14s %s' % (k, z55[k].shape))
for k in sorted(z60.files):
    w('[2860] %-14s %s' % (k, z60[k].shape))

g55 = z55['g_comp'].astype(np.float64)     # (80, 10)
gl55 = z55['g_ln'].astype(np.float64)
gm55 = z55['g_mlp'].astype(np.float64)
g60 = z60['g_comp'].astype(np.float64)     # (80, 10)
gd60 = z60['g_direct'].astype(np.float64)
dn60 = z60['dln_norm'].astype(np.float64)
dc60 = z60['dln_cos'].astype(np.float64)

# sanity: word count / n_win match
w('shapes: g55 %s g60 %s' % (g55.shape, g60.shape))

# A) identity check within 2855: g_mlp vs g_comp
dA = np.abs(gm55 - g55)
w('A) 2855 |g_mlp - g_comp|: max %.6f mean %.6f' % (dA.max(), dA.mean()))

# B) g_ln reproduction: 2855 g_ln vs 2860 dln_norm*dln_cos
rec = dn60 * dc60
dB = np.abs(gl55 - rec)
w('B) |g_ln55 - dlnnorm*cos| : max %.6f mean %.6f  -> LN2 path '
  'identical? %s' % (dB.max(), dB.mean(), bool(dB.max() < 0.02)))

# C) g_comp cross: per-word, per-layer
dC = g55 - g60
w('C) g_comp55 - g_comp60 : max %.4f mean %.4f' % (np.abs(dC).max(),
                                                   dC.mean()))
w('   per-layer mean diff : %s' % np.round(dC.mean(0), 4).tolist())
w('   per-layer max  diff : %s' % np.round(np.abs(dC).max(0), 4).tolist())

# D) is the diff systematic across words? per-word mean over L32-34 (q6..8)
seg = dC[:, 6:9]
w('D) diff on L32-34 per word: mean %.4f std %.4f min %.4f max %.4f'
  % (seg.mean(), seg.std(), seg.min(), seg.max()))
n_pos = int((seg.mean(1) > 0.5).sum())
n_neg = int((seg.mean(1) < -0.5).sum())
w('   words with mean diff > +0.5: %d ; < -0.5: %d  (of %d)'
  % (n_pos, n_neg, seg.shape[0]))

# E) 2860 internal: g_comp vs g_direct relation
w('E) 2860 per-layer mean g_comp : %s' % np.round(g60.mean(0), 4).tolist())
w('   2860 per-layer mean g_direct: %s' % np.round(gd60.mean(0), 4).tolist())

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('WROTE', OUT)
