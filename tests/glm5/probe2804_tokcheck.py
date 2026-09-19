# -*- coding: utf-8 -*-
"""Phase 2804 tokenizer precheck: candidate k>=3 targets + new anchor sets.
Run BEFORE freezing the word list (discipline: prereg after precheck)."""
from pathlib import Path
import sys

from transformers import AutoTokenizer

ROOT = Path(r"D:\AI2050\Ai2050-OpenOne")
tok = AutoTokenizer.from_pretrained(
    str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
    trust_remote_code=True, use_fast=True)

# 2803 existing domains (reused): fruit food plant company animal vehicle
# metal tool furniture clothing color music sport computer  (14)
# NEW domains for 2804:
NEW_SETS = {
    'country': ['China', 'America', 'France', 'Germany', 'Japan',
                'Italy', 'Spain', 'Russia', 'Brazil', 'India'],
    'anatomy': ['heart', 'hand', 'foot', 'eye', 'ear', 'nose',
                'mouth', 'arm', 'leg', 'bone'],
    'money': ['money', 'cash', 'coin', 'dollar', 'loan', 'debt',
              'tax', 'fund', 'profit', 'salary'],
    'weapon': ['gun', 'rifle', 'pistol', 'cannon', 'bullet', 'bomb',
               'sword', 'arrow', 'spear', 'shield'],
    'container': ['box', 'bottle', 'jar', 'barrel', 'basket',
                  'bucket', 'tube', 'crate', 'kettle', 'tray'],
    'card': ['poker', 'bridge', 'king', 'queen', 'jack', 'joker',
             'spade', 'heart', 'deal', 'bet'],
    'geometry': ['square', 'circle', 'triangle', 'cube', 'sphere',
                 'cone', 'angle', 'curve', 'oval', 'prism'],
}
CAND_TARGETS = {
    'turkey': ['animal', 'food', 'country'],
    'kiwi': ['fruit', 'animal', 'country'],
    'shell': ['animal', 'company', 'computer'],
    'blackberry': ['fruit', 'plant', 'company'],
    'squash': ['food', 'sport', 'plant'],
    'mint': ['plant', 'money', 'color'],
    'gold': ['metal', 'color', 'sport'],
    'silver': ['metal', 'color', 'sport'],
    'bronze': ['metal', 'color', 'sport'],
    'salmon': ['animal', 'food', 'color'],
    'olive': ['fruit', 'food', 'color'],
    'bow': ['weapon', 'clothing', 'music'],
    'drum': ['music', 'container', 'tool'],
    'port': ['vehicle', 'computer', 'food'],
    'bench': ['furniture', 'tool', 'sport'],
    'trunk': ['plant', 'animal', 'vehicle'],
    'boot': ['clothing', 'vehicle', 'computer'],
    'horn': ['music', 'anatomy', 'vehicle'],
    'diamond': ['metal', 'card', 'geometry'],
    'club': ['weapon', 'card', 'sport'],
    'tank': ['vehicle', 'weapon', 'container'],
    'polish': ['country', 'tool', 'clothing'],
}
# also reuse 2803 domains for target senses: check every sense domain exists
OLD_SET_NAMES = ['fruit', 'food', 'plant', 'company', 'animal', 'vehicle',
                 'metal', 'tool', 'furniture', 'clothing', 'color', 'music',
                 'sport', 'computer']

out_lines = []
tc = {}


def tid(t):
    ids = tok(' ' + t, add_special_tokens=False)['input_ids']
    ok_sp = len(ids) == 1
    if not ok_sp:
        ids = tok(t, add_special_tokens=False)['input_ids']
    ok_bare = len(ids) == 1
    tc[t] = (ok_sp, ok_bare, len(tok(' ' + t, add_special_tokens=False)['input_ids']), len(tok(t, add_special_tokens=False)['input_ids']))
    return ok_sp or ok_bare


bad = []
for dom, ws in NEW_SETS.items():
    for w in ws:
        if not tid(w):
            bad.append(('SET', dom, w))
for w in CAND_TARGETS:
    if not tid(w):
        bad.append(('TARGET', w, CAND_TARGETS[w]))
# self-reference check: target word must not appear in its own anchor sets
selfref = []
all_sets = dict(NEW_SETS)
# reload 2803 sets from its file for self-ref check
import json
src2803 = Path(r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\sources")
# simpler: hardcode the 14 old sets (same as phase2803 script)
OLD = {
    'fruit': ['apple', 'banana', 'orange', 'grape', 'lemon', 'peach', 'pear', 'mango', 'cherry', 'berry'],
    'food': ['bread', 'rice', 'cheese', 'egg', 'meat', 'soup', 'pasta', 'pizza', 'honey', 'butter'],
    'plant': ['tree', 'flower', 'rose', 'leaf', 'root', 'grass', 'oak', 'pine', 'maple', 'fern'],
    'company': ['Google', 'Microsoft', 'Amazon', 'Meta', 'Tesla', 'Nvidia', 'Samsung', 'Sony', 'IBM', 'Intel'],
    'animal': ['dog', 'cat', 'horse', 'cow', 'lion', 'tiger', 'wolf', 'rabbit', 'bird', 'fish'],
    'vehicle': ['car', 'bus', 'truck', 'train', 'ship', 'boat', 'plane', 'bicycle', 'taxi', 'tram'],
    'metal': ['gold', 'silver', 'iron', 'copper', 'steel', 'bronze', 'brass', 'tin', 'aluminum', 'nickel'],
    'tool': ['hammer', 'knife', 'file', 'wrench', 'drill', 'saw', 'axe', 'nail', 'rope', 'shovel'],
    'furniture': ['chair', 'table', 'bed', 'desk', 'sofa', 'shelf', 'cabinet', 'bench', 'stool', 'wardrobe'],
    'clothing': ['shirt', 'pants', 'dress', 'coat', 'shoe', 'sock', 'hat', 'glove', 'scarf', 'jacket'],
    'color': ['red', 'blue', 'green', 'yellow', 'purple', 'pink', 'black', 'white', 'brown', 'gray'],
    'music': ['piano', 'violin', 'guitar', 'drums', 'flute', 'trumpet', 'opera', 'jazz', 'melody', 'rhythm'],
    'sport': ['soccer', 'tennis', 'golf', 'boxing', 'rugby', 'hockey', 'baseball', 'cricket', 'cycling', 'skiing'],
    'computer': ['computer', 'keyboard', 'screen', 'laptop', 'server', 'software', 'code', 'data', 'app', 'internet'],
}
all_sets.update(OLD)
for w, senses in CAND_TARGETS.items():
    for s in senses:
        for aw in all_sets[s]:
            if aw.lower() == w.lower():
                selfref.append((w, s, aw))
            elif w.lower() in aw.lower() and len(w) >= 4:
                selfref.append((w, s, aw + ' (substring)'))

out_lines.append("BAD TOKENS: %d" % len(bad))
for b in bad:
    out_lines.append("  %s" % str(b))
out_lines.append("SELFREF ISSUES: %d" % len(selfref))
for s in selfref:
    out_lines.append("  %s" % str(s))
out_lines.append("CAND n=%d" % len(CAND_TARGETS))
detail = {w: tc[w] for w in tc}
out_lines.append("TOKEN DETAIL (spaced_ok, bare_ok, n_sp, n_bare):")
for k, v in detail.items():
    out_lines.append("  %-12s %s" % (k, v))
p = Path(r"D:\AI2050\Ai2050-OpenOne\tests\glm5\probe2804_tokcheck.txt")
p.write_text("\n".join(out_lines), encoding='utf-8')
print("WROTE", p)
