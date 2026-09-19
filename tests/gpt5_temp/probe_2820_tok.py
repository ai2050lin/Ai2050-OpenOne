"""P2820 probe: tokenization check for all words used in tid()."""
import json
from pathlib import Path

ROOT = Path('D:/AI2050/Ai2050-OpenOne')
mdir = ROOT / 'models' / 'hf' / 'qwen3-4b'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(mdir), local_files_only=True,
                                    trust_remote_code=True, use_fast=True)

COLOURS = ['red', 'black', 'purple', 'blue', 'green', 'yellow']
COLOR_WORDS = ['red', 'green', 'blue', 'black', 'white', 'yellow',
               'brown', 'pink', 'gray', 'purple', 'orange', 'crimson']
ENTS = ['apple', 'cherry', 'strawberry', 'tomato', 'blood',
        'streetlight', 'sky', 'grass', 'coal', 'banana']

rep = {}
for w in sorted(set(COLOURS + COLOR_WORDS)):
    a = tok(' ' + w, add_special_tokens=False)['input_ids']
    b = tok(w, add_special_tokens=False)['input_ids']
    rep[w] = {'sp': len(a), 'nosp': len(b), 'sp_ids': a}
for s in ENTS:
    a = tok(' ' + s, add_special_tokens=False)['input_ids']
    b = tok(s, add_special_tokens=False)['input_ids']
    rep[s] = {'sp': len(a), 'nosp': len(b), 'sp_ids': a}
for w in ['The', ' is']:
    a = tok(' ' + w, add_special_tokens=False)['input_ids']
    b = tok(w, add_special_tokens=False)['input_ids']
    rep[w] = {'sp': len(a), 'nosp': len(b), 'sp_ids': a}

out = ROOT / 'tests' / 'gpt5_temp' / 'probe_2820_tok.json'
out.write_text(json.dumps(rep, indent=1), encoding='utf-8')
print('WROTE', out)
