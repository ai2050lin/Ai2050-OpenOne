# -*- coding: utf-8 -*-
"""Phase 2995 probe part 2: tokenizer + determinism (model reload)."""
import io
import json
import os
import traceback

import torch as _t

MODEL = r'D:\AI2050\Ai2050-OpenOne\models\hf\glm4-9b-chat-hf'
BASE72 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
          r'\rdc_query_construction_20260913\phase2972')
OUT = r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05\.workbuddy\tmp_probe2995b.txt'

o = []
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    e72 = None
    for s in os.listdir(BASE72):
        p = os.path.join(BASE72, s, 'execution.json')
        if os.path.isfile(p):
            e72 = json.load(io.open(p, encoding='utf-8'))
            break
    cells = e72['cells']
    L_CAND = ["because", "therefore", "although", "unless",
              "however", "thus", "moreover", "since", "whereas",
              "despite", "hence", "nevertheless", "consequently",
              "furthermore", "otherwise", "instead", "while",
              "accordingly", "likewise", "meanwhile", "nonetheless",
              "thereafter", "whereby", "albeit"]
    allw = {k: cells[k] for k in ['F_en', 'F_fr', 'C_en', 'C_fr']}
    allw['L_cand'] = L_CAND
    for k, ws in allw.items():
        single, multi = [], []
        for w in ws:
            ii = tok(w, add_special_tokens=False)['input_ids']
            (single if len(ii) == 1 else multi).append(w)
        o.append('%s n=%d single=%d multi=%s' % (
            k, len(ws), len(single), multi))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=_t.bfloat16).cuda().eval()
    ids = tok('the', add_special_tokens=False)['input_ids']
    seq = ids + tok('apple', add_special_tokens=False)['input_ids']

    def fwd_last():
        with _t.no_grad():
            oo = model(_t.tensor([seq], device='cuda'),
                       output_hidden_states=True)
        return oo.hidden_states[-1][0, -1].detach().float().cpu().numpy()

    a = fwd_last()
    b = fwd_last()
    o.append('determinism max|a-b|=%.3e rel=%.3e' % (
        abs(a - b).max(),
        (abs(a - b) / max(abs(a).max(), 1e-30)).max()))
    o.append('mem=%.1f GB' % (_t.cuda.memory_allocated() / 2**30))
except Exception:
    o.append(traceback.format_exc())

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('probe2 done')
