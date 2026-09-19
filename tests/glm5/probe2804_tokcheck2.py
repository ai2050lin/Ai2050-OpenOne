# -*- coding: utf-8 -*-
from pathlib import Path
from transformers import AutoTokenizer
ROOT = Path(r"D:\AI2050\Ai2050-OpenOne")
tok = AutoTokenizer.from_pretrained(str(ROOT / 'models' / 'hf' / 'qwen3-4b'),
                                    local_files_only=True, trust_remote_code=True, use_fast=True)
out = []
for t in ['chip', 'suit', 'ace', 'flush', 'whist', 'canasta']:
    ns = len(tok(' ' + t, add_special_tokens=False)['input_ids'])
    nb = len(tok(t, add_special_tokens=False)['input_ids'])
    out.append("%-8s spaced=%d bare=%d ok=%s" % (t, ns, nb, min(ns, nb) == 1))
p = Path(r"D:\AI2050\Ai2050-OpenOne\tests\glm5\probe2804_tokcheck2.txt")
p.write_text("\n".join(out), encoding='utf-8')
print("WROTE", p)
