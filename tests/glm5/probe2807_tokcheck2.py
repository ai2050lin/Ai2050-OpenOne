# -*- coding: utf-8 -*-
"""Second-round token precheck for held-out pool gaps (qwen4)."""
import io

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    r"D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b", local_files_only=True,
    trust_remote_code=True, use_fast=True)

CAND = {
    'tool': ['screw', 'bolt', 'nut', 'clamp', 'wedge', 'punch', 'ruler',
             'compass', 'bit', 'adze', 'file', 'shovel'],
    'fruit': ['lime', 'quince', 'durian', 'pomelo', 'pawpaw', 'pear'],
    'metal': ['mercury', 'osmium', 'iridium', 'rhodium', 'gallium',
              'arsenic', 'selenium', 'indium'],
}
atlas = {'apple', 'banana', 'orange', 'grape', 'lemon', 'peach', 'pear',
         'mango', 'cherry', 'berry', 'hammer', 'knife', 'file', 'wrench',
         'drill', 'saw', 'axe', 'nail', 'rope', 'shovel', 'gold', 'silver',
         'iron', 'copper', 'steel', 'bronze', 'brass', 'tin', 'aluminum',
         'nickel'}

lines = []
for c, ws in CAND.items():
    for w in ws:
        n1 = len(tok(' ' + w, add_special_tokens=False)['input_ids'])
        n2 = len(tok(w, add_special_tokens=False)['input_ids'])
        n = 1 if n1 == 1 else n2
        lines.append("%s %s ntok=%d%s" % (c, w, n,
                     ' IN-ATLAS' if w in atlas else ''))

with io.open(r"D:\AI2050\Ai2050-OpenOne\tests\glm5\probe2807_tokcheck2.txt",
             'w', encoding='utf-8') as f:
    f.write("\n".join(lines) + "\n")
print("done")
