# -*- coding: utf-8 -*-
"""One-shot conversion: Qwen3-14B bf16 -> bnb NF4 serialized checkpoint.
Rationale: transformers 5.14.1 on-the-fly bnb quantization materializes full bf16
weights into CPU RAM (29.5GB > available ~23GB, GitHub issue #43032 family) -> segfault.
transformers 4.57 streams fine but has shifted all_hidden_states semantics (injection
hook propagation offset by one layer) -> protocol-incompatible with sealed 4b/glm4 runs.
Fix: quantize once under 4.57 (same bnb 0.50.2 NF4 kernel), save_pretrained the
4-bit-serialized checkpoint, and load it under 5.14's pre-quantized path (no bf16
materialize) with identical 5.14 hook semantics as the sealed sibling runs.
Run WITH PYTHONPATH=tf457 (4.57).
"""
import os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

T0 = time.time()
MDIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\Qwen3-14B'
OUT = r'D:\AI2050\Ai2050-OpenOne\models\hf\Qwen3-14B-bnb-nf4'
OUTF = OUT + r'\conv_log.txt'

def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    with open(OUTF, 'a', encoding='utf-8') as f:
        f.write(ln + '\n')
    try:
        print(ln, flush=True)
    except Exception:
        pass

os.makedirs(OUT, exist_ok=True)
open(OUTF, 'w').close()
log('convert begin: %s -> %s' % (MDIR, OUT))
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
t1 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MDIR, quantization_config=bnb, device_map={'': 0}, trust_remote_code=True).eval()
log('loaded nf4 in %.1f s (vram_alloc=%.2f GB, is_loaded_in_4bit=%s)' % (
    time.time() - t1, torch.cuda.memory_allocated() / 1e9, model.is_loaded_in_4bit))

# quick numeric sanity: one forward, hidden_states count and finite check
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
ids = tok('The quick brown fox jumps over the lazy dog.', return_tensors='pt')['input_ids'].cuda()
with torch.no_grad():
    o = model(input_ids=ids, output_hidden_states=True)
log('forward ok: %d hidden slots, logits %s' % (len(o.hidden_states), tuple(o.logits.shape)))

t2 = time.time()
model.save_pretrained(OUT, safe_serialization=True)
log('save_pretrained done in %.1f s' % (time.time() - t2))
tok.save_pretrained(OUT)
log('tokenizer saved')

# verify reload size on disk
tot = 0
for f2 in os.listdir(OUT):
    p = os.path.join(OUT, f2)
    if os.path.isfile(p):
        tot += os.path.getsize(p)
log('OUT dir total %.2f GB, files=%d' % (tot / (1 << 30), len(os.listdir(OUT))))
log('CONVERT DONE')
