# -*- coding: utf-8 -*-
"""p3087 bf16 timing: is full bf16 viable on 16GB?
Measure memory peak + per-forward speed at two
seq lengths; compare with int8 numbers."""
import gc
import io
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MDIR = ROOT + r'\models\hf\glm4-9b-chat-hf'
OUT = ROOT + r'\tests\gpt5_temp\p3087_bf16_timing.txt'
o = []


def w(msg):
    o.append(str(msg))
    with io.open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(o) + '\n')


import torch
from transformers import AutoModelForCausalLM, \
    AutoTokenizer

tok = AutoTokenizer.from_pretrained(MDIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MDIR, dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
w('bf16 loaded %.1fs alloc=%.2fGB peak=%.2fGB'
  % (time.time() - t0,
     torch.cuda.memory_allocated() / 1e9,
     torch.cuda.max_memory_allocated() / 1e9))
w('total VRAM=%.2fGB reserved=%.2fGB'
  % (torch.cuda.get_device_properties(0)
     .total_memory / 1e9,
     torch.cuda.memory_reserved() / 1e9))

for sl in (11, 30, 60):
    ids = tok(' '.join(['The apple is a fruit.'
                        ] * (sl // 5)),
              return_tensors='pt')
    n = ids['input_ids'].shape[1]
    try:
        with torch.no_grad():
            model(input_ids=ids['input_ids'].to(
                'cuda'),
                attention_mask=ids[
                    'attention_mask'].to('cuda'),
                output_hidden_states=True)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(15):
                model(input_ids=ids[
                    'input_ids'].to('cuda'),
                    attention_mask=ids[
                        'attention_mask'].to(
                            'cuda'))
        per = (time.time() - t0) / 15.0
        w('seq=%d per-fwd %.3fs peak=%.2fGB'
          % (n, per,
             torch.cuda.max_memory_allocated()
             / 1e9))
    except torch.cuda.OutOfMemoryError:
        w('seq=%d OOM' % n)
        break
    except Exception:
        import traceback
        w('seq=%d FAIL %s'
          % (n, traceback.format_exc()[-400:]))
        break
w('BF16_TIMING_DONE')
