# -*- coding: utf-8 -*-
"""phase2920 probe: design-feasibility checks BEFORE freezing.
1) 2887 npz keys + labels_concept / ck class structure
2) 2917 npz sig-set margins (anchor risk: min sig vs max nonsig)
3) 2919 npz keys + dirs_all shape + axis_names
4) tokenization feasibility of candidate attribute pools
Output: tests/gpt5_temp/phase2920_probe.txt
"""
import numpy as np
from collections import Counter

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\phase2920_probe.txt'
L = []


def w(s):
    L.append(str(s))


# ---------- 1) 2887 ----------
z87 = np.load(BASE + r'\phase2887\language_axis_mlp'
              r'\language_axis_mlp.npz', allow_pickle=True)
w('== 2887 keys: %s' % sorted(z87.files))
for k in sorted(z87.files):
    a = z87[k]
    w('  %s shape=%s dtype=%s' % (k, a.shape, a.dtype))
words = [tuple(str(x).split(':')) for x in z87['words']]
lab_lang = np.asarray(z87['labels_lang']).astype(int)
w('words n=%d lang counts=%s'
  % (len(words), Counter(x[0] for x in words)))
ck = [int(x[1]) for x in words]
cnt_ck = Counter(ck)
w('ck unique=%d sizes=%s'
  % (len(cnt_ck), sorted(cnt_ck.values(), reverse=True)))
if 'labels_concept' in z87.files:
    lc = np.asarray(z87['labels_concept']).tolist()
    w('labels_concept n=%d unique=%d sizes=%s'
      % (len(lc), len(set(lc)),
         sorted(Counter(lc).values(), reverse=True)))
    w('lc == ck? %s' % (lc == ck))
ex = {}
for (lang, c, wd) in words:
    ex.setdefault(c, []).append('%s:%s' % (lang, wd))
for c in sorted(ex, key=lambda x: -len(ex[x]))[:8]:
    w('  ck %s (n=%d): %s' % (c, len(ex[c]), ex[c]))

# ---------- 2) 2917 ----------
z17 = np.load(BASE + r'\phase2917\event_atlas\event_atlas.npz',
              allow_pickle=True)
w('== 2917 keys: %s' % sorted(z17.files))
pm = z17['p_maxT'].astype(np.float64)
sm = z17['sign_M'].astype(np.float64)
sig = pm <= 0.05
w('n_sig(p_maxT<=0.05)=%d margins min=%.6f max=%.6f'
  % (int(sig.sum()), float(sm[sig].min()), float(sm[sig].max())))
w('sig cells: %s'
  % sorted([(int(h), int(li), round(float(sm[h, li]), 5))
            for h, li in zip(*np.where(sig))],
           key=lambda t: -t[2]))
nonsig = sm[~sig]
w('max nonsig margin=%.6f (gap to min sig = %.6f)'
  % (float(nonsig.max()), float(sm[sig].min() - nonsig.max())))
mp = z17['max_perm'].astype(np.float64)
w('max_perm: min=%.6f p95=%.6f max=%.6f'
  % (float(mp.min()), float(np.percentile(mp, 95)),
     float(mp.max())))
bh = z17['B_heads']
w('B_heads shape=%s dtype=%s absmax=%.4f'
  % (bh.shape, bh.dtype, float(np.abs(bh).max())))

# ---------- 3) 2919 ----------
z19 = np.load(BASE + r'\phase2919\multiaxis_direction_families'
              r'\multiaxis_families.npz', allow_pickle=True)
w('== 2919 keys: %s' % sorted(z19.files))
w('dirs_all %s axis_names=%s'
  % (z19['dirs_all'].shape,
     [str(x) for x in z19['axis_names']]))

# ---------- 4) tokenization ----------
from transformers import AutoTokenizer
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
tok = AutoTokenizer.from_pretrained(
    MD, local_files_only=True, trust_remote_code=True,
    use_fast=True)
POOLS = {
    'speed_hi': ['fast', 'quick', 'rapid', 'swift', 'speedy',
                 'brisk', 'hasty', 'fleet'],
    'speed_lo': ['slow', 'sluggish', 'leisurely', 'plodding',
                 'unhurried', 'creeping', 'crawling', 'stagnant'],
    'size_hi': ['huge', 'enormous', 'giant', 'massive', 'immense',
                'colossal', 'gigantic', 'vast'],
    'size_lo': ['tiny', 'small', 'little', 'miniature', 'minute',
                'petite', 'microscopic', 'diminutive'],
    'moist_hi': ['wet', 'damp', 'humid', 'moist', 'soggy',
                 'soaked', 'drenched', 'dank'],
    'moist_lo': ['dry', 'arid', 'parched', 'dehydrated',
                 'withered', 'dusty', 'waterless', 'baked'],
}
w('== tokenization (spaced / bare) ==')
bad = []
for pool, ws in POOLS.items():
    row = []
    for t in ws:
        ids = tok(' ' + t, add_special_tokens=False)['input_ids']
        ok1 = len(ids) == 1
        if not ok1:
            ids = tok(t, add_special_tokens=False)['input_ids']
        ok2 = len(ids) == 1
        row.append('%s:%d/%d' % (t, len(tok(
            ' ' + t, add_special_tokens=False)['input_ids']),
            len(ids)))
        if not (ok1 or ok2):
            bad.append((pool, t))
    w('  %s: %s' % (pool, ', '.join(row)))
w('FAILS: %s' % (bad if bad else 'none'))
# lang words quick recheck
fails87 = []
tc = {}
for (lang, c, t) in words:
    ids = tok(' ' + t, add_special_tokens=False)['input_ids']
    if len(ids) != 1:
        ids = tok(t, add_special_tokens=False)['input_ids']
    if len(ids) != 1:
        fails87.append(t)
w('2887 words multi-token fails: %s'
  % (fails87 if fails87 else 'none'))

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('OK probe')
