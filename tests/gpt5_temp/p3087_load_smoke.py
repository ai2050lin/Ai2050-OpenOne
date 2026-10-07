# -*- coding: utf-8 -*-
"""p3087 GLM4 load smoke (decision experiment):
try bf16 full (expect OOM) then int8; single
forward + hooks + timing -> write report file."""
import gc
import io
import time
import traceback

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MDIR = ROOT + r'\models\hf\glm4-9b-chat-hf'
OUT = ROOT + r'\tests\gpt5_temp\p3087_load_smoke.txt'
o = []


def w(msg):
    o.append(str(msg))
    with io.open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(o) + '\n')


import torch
from transformers import AutoModelForCausalLM, \
    AutoTokenizer

w('torch %s cuda=%s' % (
    torch.__version__,
    torch.cuda.is_available()))

tok = AutoTokenizer.from_pretrained(MDIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
w('tokenizer ok vocab=%d pad=%r eos=%r'
  % (len(tok), tok.pad_token, tok.eos_token))

s = 'The apple is a fruit. The banana is'
ids = tok(s, return_tensors='pt')
w('encode: len=%d ids[:8]=%s'
  % (ids['input_ids'].shape[1],
     ids['input_ids'][0][:8].tolist()))

# ---- attempt 1: bf16 full ----
t0 = time.time()
try:
    m1 = AutoModelForCausalLM.from_pretrained(
        MDIR, torch_dtype=torch.bfloat16,
        attn_implementation='eager')
    m1 = m1.to('cuda').eval()
    w('bf16 FULL loaded %.1fs (unexpected)'
      % (time.time() - t0))
    del m1
    gc.collect()
    torch.cuda.empty_cache()
except torch.cuda.OutOfMemoryError:
    w('bf16 OOM as expected (%.1fs)'
      % (time.time() - t0))
    gc.collect()
    torch.cuda.empty_cache()
except Exception:
    w('bf16 FAIL other:\n%s'
      % traceback.format_exc()[-1500:])
    gc.collect()
    torch.cuda.empty_cache()

# ---- attempt 2: int8 ----
from transformers import BitsAndBytesConfig
t0 = time.time()
bnb = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, quantization_config=bnb,
    device_map='auto').eval()
w('int8 loaded %.1fs gpu=%.2fGB layers=%d'
  % (time.time() - t0,
     torch.cuda.memory_allocated() / 1e9,
     len(model.model.layers)))

# ---- single forward + hooks ----
ids = ids.to(model.device)
cap = {}


def hk():
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) \
            else out
        cap['x'] = t.detach().float().cpu()
    return hook


h = model.model.layers[20].register_forward_hook(
    hk())
t0 = time.time()
with torch.no_grad():
    out = model(
        input_ids=ids['input_ids'],
        attention_mask=ids['attention_mask'],
        output_hidden_states=True)
t1 = time.time() - t0
h.remove()
hs = out.hidden_states
w('forward %.2fs hidden_states=%d '
  'L20 hook shape=%s norm=%.2f '
  'last_norm=%.2f logits=%s'
  % (t1, len(hs),
     tuple(cap['x'].shape),
     float(cap['x'].norm()),
     float(hs[-1].float().norm()),
     tuple(out.logits.shape)))

# ---- greedy continuation sanity ----
with torch.no_grad():
    gen = model.generate(
        input_ids=ids['input_ids'],
        attention_mask=ids['attention_mask'],
        max_new_tokens=12, do_sample=False)
w('gen: %r'
  % tok.decode(gen[0], skip_special_tokens=True))

# ---- timing loop (20 forwards) ----
t0 = time.time()
with torch.no_grad():
    for _ in range(20):
        model(input_ids=ids['input_ids'],
              attention_mask=ids['attention_mask'])
per = (time.time() - t0) / 20.0
w('per-forward (short seq) %.3fs -> 20k '
  'forwards ~ %.0fs seq-dependent'
  % (per, per * 20000))
w('LOAD_SMOKE_DONE')
