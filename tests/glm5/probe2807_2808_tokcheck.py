# -*- coding: utf-8 -*-
"""Tokenizer + tensor-availability precheck for Phase 2807/2808.

For each model: read config (architectures, tie, rms eps), check
safetensors for lm_head / embed_tokens / norm weights, and precheck
single-token status of 2806 atlas words + backup pools.

qwen4: precheck the HELD-OUT candidate pool (2807).
ds7 / glm4 / qwen17 / qwen25: precheck the 2806 atlas pool (2808).
"""
import io
import json
import os

ROOT = r"D:\AI2050\Ai2050-OpenOne"
MH = os.path.join(ROOT, "models", "hf")
OUT = os.path.join(ROOT, "tests", "glm5", "probe2807_2808_tokcheck.txt")

MODELS = {
    'qwen4': 'qwen3-4b',
    'ds7': 'deepseek-r1-distill-qwen-7b',
    'glm4': 'glm4-9b-chat-hf',
    'qwen17': 'qwen3-1.7b',
    'qwen25': 'qwen2.5-3b-instruct',
}

ATLAS = {
    'fruit': ['apple', 'banana', 'orange', 'grape', 'lemon', 'peach',
              'pear', 'mango', 'cherry', 'berry'],
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
    'tool': ['hammer', 'knife', 'file', 'wrench', 'drill', 'saw',
             'axe', 'nail', 'rope', 'shovel'],
    'clothing': ['shirt', 'pants', 'dress', 'coat', 'shoe', 'sock',
                 'hat', 'glove', 'scarf', 'jacket'],
}
BACKUP = {
    'fruit': ['melon', 'date', 'kiwi', 'almond'],
    'animal': ['owl', 'seal', 'crab', 'ant'],
    'metal': ['lithium', 'sodium', 'barium', 'cesium'],
    'vehicle': ['sleigh', 'raft', 'glider', 'buggy'],
    'country': ['Qatar', 'Nepal', 'Ghana', 'Wales'],
    'food': ['stew', 'oats', 'yogurt', 'onion'],
    'nature': ['cliff', 'meadow', 'reef', 'breeze'],
    'furniture': ['crib', 'buffet', 'mat', 'hutch'],
    'tool': ['trowel', 'auger', 'mallet', 'vice'],
    'clothing': ['cap', 'gown', 'kimono', 'sneaker'],
}
HELD_MAIN = {
    'fruit': ['plum', 'apricot', 'papaya', 'guava', 'fig', 'lychee',
              'coconut', 'olive', 'raisin', 'prune'],
    'animal': ['deer', 'bear', 'monkey', 'elephant', 'dolphin', 'whale',
               'snake', 'frog', 'duck', 'goat'],
    'metal': ['lead', 'zinc', 'platinum', 'tungsten', 'titanium',
              'cobalt', 'chrome', 'magnesium', 'calcium', 'radium'],
    'vehicle': ['scooter', 'tractor', 'subway', 'ferry', 'canoe',
                'kayak', 'trailer', 'wagon', 'sedan', 'limo'],
    'country': ['Norway', 'Sweden', 'Finland', 'Poland', 'Portugal',
                'Greece', 'Mexico', 'Kenya', 'Chile', 'Peru'],
    'food': ['noodle', 'tofu', 'sausage', 'bacon', 'ham', 'salad',
             'curry', 'pie', 'cake', 'candy'],
    'nature': ['beach', 'island', 'valley', 'storm', 'thunder', 'fog',
               'frost', 'star', 'moon', 'cave'],
    'furniture': ['couch', 'drawer', 'cradle', 'mattress', 'pillow',
                  'rug', 'dresser', 'bookcase', 'ottoman', 'cot'],
    'tool': ['chisel', 'router', 'lathe', 'sander', 'grinder', 'spanner',
             'crowbar', 'anvil', 'hatchet', 'mop'],
    'clothing': ['vest', 'jeans', 'skirt', 'blouse', 'sweater', 'robe',
                 'uniform', 'apron', 'scarf', 'boot'],
}

atlas_words = set(w for v in ATLAS.values() for w in v)
lines = []

from transformers import AutoTokenizer

for key, dirname in MODELS.items():
    d = os.path.join(MH, dirname)
    lines.append("=== %s (%s) ===" % (key, dirname))
    try:
        cfg = json.load(io.open(os.path.join(d, 'config.json'),
                                encoding='utf-8'))
        lines.append("arch=%s tie=%s rms_eps=%s vocab=%s"
                     % (cfg.get('architectures'),
                        cfg.get('tie_word_embeddings'),
                        cfg.get('rms_norm_eps'), cfg.get('vocab_size')))
    except Exception as e:
        lines.append("config ERR %s" % e)
    idxp = os.path.join(d, 'model.safetensors.index.json')
    tensors = {}
    if os.path.exists(idxp):
        wm = json.load(io.open(idxp, encoding='utf-8'))['weight_map']
        for t in ('lm_head.weight', 'model.embed_tokens.weight',
                  'model.norm.weight'):
            tensors[t] = wm.get(t, 'MISSING')
    else:
        st = os.path.join(d, 'model.safetensors')
        tensors['single_file'] = os.path.exists(st)
    lines.append("tensors: %s" % tensors)
    try:
        tok = AutoTokenizer.from_pretrained(d, local_files_only=True,
                                            trust_remote_code=True,
                                            use_fast=True)
    except Exception as e:
        lines.append("tokenizer ERR %s" % e)
        continue

    def ntok(t):
        ids = tok(' ' + t, add_special_tokens=False)['input_ids']
        if len(ids) == 1:
            return 1
        ids2 = tok(t, add_special_tokens=False)['input_ids']
        return len(ids2)

    if key == 'qwen4':
        pool = {}
        for c, ws in HELD_MAIN.items():
            pool[c] = list(ws) + BACKUP[c]
        bad = []
        sel = {}
        for c, ws in pool.items():
            got = []
            for w in ws:
                if w in atlas_words:
                    bad.append((c, w, 'in-atlas'))
                    continue
                n = ntok(w)
                if n == 1 and len(got) < 10:
                    got.append(w)
                elif n != 1:
                    bad.append((c, w, '%d-tok' % n))
            sel[c] = got
        lines.append("HELD selection per class: %s"
                     % {c: len(v) for c, v in sel.items()})
        for c, v in sel.items():
            if len(v) != 10:
                lines.append("  SHORT %s: %s" % (c, v))
        lines.append("HELD rejected: %s" % bad)
        lines.append("HELD final: %s" % json.dumps(sel))
    else:
        bad = []
        for c, ws in ATLAS.items():
            for w in ws:
                n = ntok(w)
                if n != 1:
                    bad.append((c, w, '%d-tok' % n))
        lines.append("ATLAS non-single-token: %s" % bad)
        bsel = {}
        for c, ws in BACKUP.items():
            bsel[c] = [(w, ntok(w)) for w in ws]
        lines.append("BACKUP ntok: %s" % bsel)
    lines.append("")

with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines) + "\n")
print("done")
