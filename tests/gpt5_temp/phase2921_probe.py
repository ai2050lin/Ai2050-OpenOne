# -*- coding: utf-8 -*-
"""phase2921 probe: tokenizer screening of expanded attribute
vocabularies (target n ~ 40-50 per axis, poles >= 15 each).
2920 survivors are FIXED members; NEW candidates screened here
(spaced form single token per the 2917 tid() convention, bare
fallback allowed).
Output: tests/gpt5_temp/phase2921_probe.txt
"""
from transformers import AutoTokenizer

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\phase2921_probe.txt'
tok = AutoTokenizer.from_pretrained(
    MD, local_files_only=True, trust_remote_code=True,
    use_fast=True)

FIXED = {
    'speed_hi': ['fast', 'quick', 'rapid', 'swift', 'speedy',
                 'brisk', 'fleet', 'speeding', 'flying',
                 'racing'],
    'speed_lo': ['slow', 'sluggish', 'creeping', 'crawling',
                 'stagnant', 'idle', 'static', 'sleepy',
                 'passive', 'frozen', 'drifting'],
    'size_hi': ['huge', 'enormous', 'giant', 'massive', 'immense',
                'colossal', 'gigantic', 'vast', 'towering'],
    'size_lo': ['tiny', 'small', 'little', 'miniature', 'minute',
                'petite', 'microscopic', 'mini'],
    'moist_hi': ['wet', 'damp', 'humid', 'moist', 'soaked',
                 'dank', 'rainy', 'dripping'],
    'moist_lo': ['dry', 'dusty', 'baked', 'dried', 'barren',
                 'crisp', 'thirsty'],
}
CANDS = {
    'speed_hi': ['rushing', 'dashing', 'lively', 'agile', 'nimble',
                 'dynamic', 'accelerated', 'galloping', 'bolting',
                 'streaking', 'whizzing', 'blazing', 'ripping',
                 'charging', 'soaring', 'darting', 'humming',
                 'quickened', 'hasty', 'instant'],
    'speed_lo': ['drowsy', 'lethargic', 'listless', 'dormant',
                 'inactive', 'motionless', 'immobile', 'still',
                 'stalled', 'halted', 'parked', 'anchored',
                 'languid', 'tardy', 'dawdling', 'trailing',
                 'slowing', 'decelerating', 'stopped', 'sleeping',
                 'torpid', 'unhurried', 'leisurely', 'plodding',
                 'lagging'],
    'size_hi': ['mammoth', 'monumental', 'gargantuan', 'jumbo',
                'bulky', 'broad', 'wide', 'grand', 'great',
                'large', 'big', 'hefty', 'weighty', 'mighty',
                'prodigious', 'boundless', 'expansive',
                'extensive', 'spacious', 'oversized', 'titanic',
                'mountainous', 'swollen', 'inflated'],
    'size_lo': ['micro', 'wee', 'slim', 'slender', 'narrow',
                'slight', 'compact', 'smallish', 'pocket', 'toy',
                'baby', 'puny', 'dainty', 'minor', 'lesser',
                'shrunken', 'condensed', 'downsized', 'skimpy',
                'squat', 'short', 'undersized', 'teeny', 'weeny',
                'midget'],
    'moist_hi': ['soggy', 'drenched', 'sopping', 'marshy',
                 'misty', 'watery', 'drizzly', 'showery', 'dewy',
                 'clammy', 'sticky', 'slimy', 'muggy',
                 'saturated', 'soaking', 'wetted', 'sodden',
                 'moistened', 'flooded', 'swampy', 'steamy',
                 'juicy', 'briny', 'waterlogged'],
    'moist_lo': ['arid', 'parched', 'dehydrated', 'withered',
                 'waterless', 'rainless', 'wilted', 'powdery',
                 'sunbaked', 'shriveled', 'scorched', 'toasted',
                 'burnt', 'stale', 'brittle', 'flaky', 'crumbly',
                 'sandy', 'desiccated', 'absorbent', 'drier',
                 'driest', 'dehumidified', 'wrung', 'droughty',
                 'sere'],
}

L = ['== 2921 vocab screening ==']
final = {}
seen = {}
for pool in FIXED:
    seen[pool] = set()
allw = []
for pool, ws in FIXED.items():
    for t in ws:
        allw.append(t)
dups = [t for t, c in __import__('collections').Counter(
    allw).items() if c > 1]
L.append('duplicate words across FIXED pools: %s'
         % (dups if dups else 'none'))

for pool in FIXED:
    final[pool] = list(FIXED[pool])
for pool, ws in CANDS.items():
    ok, bad = [], []
    for t in ws:
        n1 = len(tok(' ' + t, add_special_tokens=False)
                 ['input_ids'])
        n2 = len(tok(t, add_special_tokens=False)['input_ids'])
        if n1 == 1 or n2 == 1:
            ok.append(t)
        else:
            bad.append(t)
    final[pool] = final[pool] + ok
    L.append('%s NEW ok(%d): %s' % (pool, len(ok), ok))
    L.append('%s NEW rejected(%d): %s'
             % (pool, len(bad), bad))

L.append('== final pools ==')
tot = 0
allw2 = []
for pool in ['speed_hi', 'speed_lo', 'size_hi', 'size_lo',
             'moist_hi', 'moist_lo']:
    ws = final[pool]
    dd = [t for t, c in __import__('collections').Counter(
        ws).items() if c > 1]
    allw2 += ws
    L.append('%s: n=%d %s %s'
             % (pool, len(ws), ws,
                ('DUP=%s' % dd) if dd else ''))
ax_n = {a: len(final[a + '_hi']) + len(final[a + '_lo'])
        for a in ('speed', 'size', 'moist')}
L.append('axis totals: %s (grand %d)'
         % (ax_n, len(allw2)))
L.append('cross-pool duplicates: %s'
         % ([t for t, c in __import__('collections').Counter(
             allw2).items() if c > 1] or 'none'))
L.append('short axes (<40 total or pole <15): %s'
         % ([a for a in ax_n
             if ax_n[a] < 40
             or len(final[a + '_hi']) < 15
             or len(final[a + '_lo']) < 15] or 'none'))

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('OK probe 2921')
