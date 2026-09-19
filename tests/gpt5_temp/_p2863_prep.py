"""2863 prep: per-class slices of the 2846 census tensor (zero forward).

drops_all (80, 36, 32) is word-major in CAT_WORDS order x 8 words.
For each class c: slice mean spectrum S_c (1152,) and compare against
the all-class spectrum and against other classes:
  A) Spearman(S_c, S_all) per class            (within-tensor fidelity)
  B) 10x10 pairwise Spearman(S_c, S_c')        (class similarity)
  C) top-64 heads per class vs global top-64   (front-edge overlap)
  D) formatter-set overlap: global formatters (drop>=p75 & direct<=p50)
     vs per-class formatters
"""
import json

import numpy as np

R = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result' \
    r'\rdc_query_construction_20260913'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_p2863_prep_report.txt'

exec2806 = json.load(open(
    R + r'\phase2806\qwen4_hierarchy\execution.json', encoding='utf-8'))
CAT_WORDS = list(exec2806['cats'].keys())

z = np.load(R + r'\phase2846\fullhead_census\census_full.npz')
drops = z['drops_all'].astype(np.float64)          # (80, 36, 32)
mean_s0 = z['mean_s0'].astype(np.float64)
mean_s1 = z['mean_s1'].astype(np.float64)
NL, NH = 36, 32
WPC = drops.shape[0] // len(CAT_WORDS)
assert WPC * len(CAT_WORDS) == drops.shape[0]
direct = mean_s0 + mean_s1

lines = []


def w(s):
    lines.append(str(s))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


S_all = drops.mean(0).reshape(-1)
S_cls = {}
for ci, cat in enumerate(CAT_WORDS):
    S_cls[cat] = drops[ci * WPC:(ci + 1) * WPC].mean(0).reshape(-1)

w('=== A) per-class spectrum vs all-class (Spearman, 1152) ===')
for cat in CAT_WORDS:
    w('  %-10s %.4f' % (cat, spearman(S_cls[cat], S_all)))

w('')
w('=== B) pairwise class similarity (Spearman, 1152) ===')
w('        ' + ' '.join('%7s' % c[:7] for c in CAT_WORDS))
for ca in CAT_WORDS:
    row = [' '.join(['%7.3f' % spearman(S_cls[ca], S_cls[cb])
                     if ci <= cj else '      .' for ci, cb in
                     enumerate(CAT_WORDS)]) for cj, ca0 in [(0, ca)]]
    vals = [spearman(S_cls[ca], S_cls[cb]) for cb in CAT_WORDS]
    w('  %-7s %s' % (ca[:7], ' '.join('%7.3f' % v for v in vals)))
vals_sym = [spearman(S_cls[CAT_WORDS[i]], S_cls[CAT_WORDS[j]])
            for i in range(len(CAT_WORDS)) for j in range(i + 1, len(CAT_WORDS))]
w('  offdiag: mean %.4f min %.4f max %.4f'
  % (float(np.mean(vals_sym)), float(np.min(vals_sym)),
     float(np.max(vals_sym))))

w('')
w('=== C) per-class top-64 overlap with global top-64 ===')
g_order = np.argsort(S_all)[::-1]
g_top64 = set(g_order[:64].tolist())
for cat in CAT_WORDS:
    c_order = np.argsort(S_cls[cat])[::-1]
    ov = len(g_top64 & set(c_order[:64].tolist()))
    w('  %-10s %2d/64' % (cat, ov))

w('')
w('=== D) formatter-set overlap (drop>=p75_all & direct<=p50_all) ===')
p75 = float(np.percentile(S_all, 75))
p50d = float(np.percentile(direct, 50))
glob_form = set(np.where((S_all >= p75) & (direct <= p50d))[0].tolist())
w('  global formatters: %d heads' % len(glob_form))
for cat in CAT_WORDS:
    sc = S_cls[cat]
    dc = direct  # direct writes are class-agg (s0/s1 saved aggregated)
    cf = set(np.where((sc >= float(np.percentile(sc, 75)))
                      & (dc <= p50d))[0].tolist())
    w('  %-10s formatters %3d, overlap with global %3d'
      % (cat, len(cf), len(glob_form & cf)))

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('WROTE', OUT)
