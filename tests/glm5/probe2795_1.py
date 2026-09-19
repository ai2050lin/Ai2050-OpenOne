"""Probe 2795-1: tokenizer check for the 100-word atlas vocabulary.

Verify that ' ' + word is a single token for every candidate word,
all 10 category words, and the near-pair property words.  Words that
fail are replaced in the frozen list.  Writes report to
probe2795_1.txt.
"""
import sys
from pathlib import Path

REPORT = Path('C:/Users/Admin/WorkBuddy/2026-09-15-08-09-16/probe2795_1.txt')
lines = []


def out(s=''):
    lines.append(str(s))


CANDIDATES = {
    'fruit': ['apple', 'banana', 'orange', 'grape', 'lemon', 'peach',
              'pear', 'mango', 'cherry', 'melon'],
    'animal': ['dog', 'cat', 'horse', 'cow', 'lion', 'tiger', 'wolf',
               'rabbit', 'bird', 'fish'],
    'metal': ['gold', 'silver', 'iron', 'copper', 'steel', 'bronze',
              'brass', 'tin', 'aluminum', 'nickel'],
    'vehicle': ['car', 'bus', 'truck', 'train', 'ship', 'boat',
                'plane', 'bicycle', 'taxi', 'tram'],
    'country': ['Japan', 'China', 'France', 'Germany', 'Brazil',
                'India', 'Canada', 'Russia', 'Italy', 'Egypt'],
    'food': ['bread', 'rice', 'cheese', 'egg', 'meat', 'soup',
             'pasta', 'pizza', 'honey', 'butter'],
    'nature': ['ocean', 'river', 'mountain', 'forest', 'desert',
               'lake', 'rain', 'snow', 'cloud', 'wind'],
    'furniture': ['chair', 'table', 'bed', 'desk', 'sofa', 'shelf',
                  'cabinet', 'bench', 'stool', 'wardrobe'],
    'tool': ['hammer', 'knife', 'screwdriver', 'wrench', 'drill',
             'saw', 'axe', 'chisel', 'pliers', 'shovel'],
    'clothing': ['shirt', 'pants', 'dress', 'coat', 'shoe', 'sock',
                 'hat', 'glove', 'scarf', 'jacket'],
}
EXTRA = ['fruit', 'animal', 'metal', 'vehicle', 'country', 'food',
         'nature', 'furniture', 'tool', 'clothing',
         'red', 'yellow', 'bark', 'meow', 'driver', 'passenger',
         'item', 'plant', 'water', 'sea', 'pet', 'mammal']


def main():
    sys.path.insert(0, 'D:/AI2050/Ai2050-OpenOne/tests/glm5')
    import rdc_construction_common as cc
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(cc.ROOT / 'models' / 'hf' / 'qwen3-4b'),
        local_files_only=True, trust_remote_code=True, use_fast=True)

    def check(t):
        ids = tok(' ' + t, add_special_tokens=False)['input_ids']
        if len(ids) == 1:
            return 'OK1', ids
        ids2 = tok(t, add_special_tokens=False)['input_ids']
        if len(ids2) == 1:
            return 'OK0', ids2
        return 'MULTI%d' % len(ids), ids

    for cat, words in CANDIDATES.items():
        bad = []
        for w in words:
            st, ids = check(w)
            if st != 'OK1':
                bad.append((w, st))
        out('%s: %s' % (cat, 'ALL_OK' if not bad else bad))
    out('--- extras ---')
    for t in EXTRA:
        st, ids = check(t)
        out('%s: %s id=%s' % (t, st, ids))
    out('total words=%d' % sum(len(v) for v in CANDIDATES.values()))
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    print('OK')


if __name__ == '__main__':
    main()
