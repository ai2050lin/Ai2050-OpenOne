"""P2821 probe: tokenization for entity pool + adjective pairs."""
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path('D:/AI2050/Ai2050-OpenOne')
mdir = ROOT / 'models' / 'hf' / 'qwen3-4b'
tok = AutoTokenizer.from_pretrained(str(mdir), local_files_only=True,
                                    trust_remote_code=True, use_fast=True)

ENTS_CAND = ['apple', 'cherry', 'lemon', 'banana', 'grape', 'dog', 'cat',
             'mouse', 'elephant', 'lion', 'rock', 'feather', 'pillow',
             'sponge', 'steel', 'car', 'plane', 'rocket', 'sun', 'moon',
             'ice', 'cloud', 'streetlight', 'skyscraper', 'tower', 'cup',
             'tree', 'mountain', 'river', 'stone', 'bee', 'snail', 'ox',
             'horse', 'kitten', 'puppy', 'trumpet', 'piano', 'anvil',
             'brick', 'balloon', 'bubble', 'bread', 'hammer', 'nail',
             'anchovy', 'whale', 'shark', 'sparrow', 'crumb']
ADJS = ['big', 'small', 'large', 'tiny', 'huge', 'heavy', 'light',
        'hot', 'cold', 'warm', 'fast', 'slow', 'bright', 'dark',
        'hard', 'soft', 'dry', 'wet', 'loud', 'quiet']
FUNCT = ['The', ' is', ' a', ' very']

rep = {'entities': {}, 'adjectives': {}, 'funct': {}}
for w in ENTS_CAND:
    a = tok(' ' + w, add_special_tokens=False)['input_ids']
    b = tok(w, add_special_tokens=False)['input_ids']
    rep['entities'][w] = {'sp': len(a), 'nosp': len(b)}
for w in ADJS:
    a = tok(' ' + w, add_special_tokens=False)['input_ids']
    b = tok(w, add_special_tokens=False)['input_ids']
    rep['adjectives'][w] = {'sp': len(a), 'nosp': len(b)}
for w in FUNCT:
    a = tok(' ' + w, add_special_tokens=False)['input_ids']
    b = tok(w, add_special_tokens=False)['input_ids']
    rep['funct'][w] = {'sp': len(a), 'nosp': len(b)}

out = ROOT / 'tests' / 'gpt5_temp' / 'probe_2821_tok.json'
out.write_text(json.dumps(rep, indent=0), encoding='utf-8')
multi_e = [w for w, v in rep['entities'].items() if v['sp'] > 1]
multi_a = [w for w, v in rep['adjectives'].items() if v['sp'] > 1]
print('WROTE', out)
print('MULTI_TOKEN_ENTITIES', multi_e)
print('MULTI_TOKEN_ADJ', multi_a)
