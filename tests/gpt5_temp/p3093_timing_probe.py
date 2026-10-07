# -*- coding: utf-8 -*-
"""p3093 timing probe: Qwen3-14B bf16 eager
full-load viability + per-forward cost, BEFORE
the phase3093 A1 authoritative run (~12210
forwards).  Protocol mirrors the A1 load path
exactly (from_pretrained bf16 eager -> .to cuda).
Writes report to p3093_timing_report.txt."""
import gc
import io
import time

import numpy as np
import torch
from transformers import AutoTokenizer, \
    AutoModelForCausalLM

MDIR = r'D:\AI2050\Ai2050-OpenOne\models\hf' \
       r'\Qwen3-14B'
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\p3093_timing_report.txt')
o = []


def log(m):
    o.append(str(m))


t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR)
log('tokenizer ok %.1fs' % (time.time() - t0))

t1 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda') \
    .eval()
log('model loaded %.1fs' % (time.time() - t1))
log('config: layers=%d hidden=%d heads=%d kv=%d '
    'head_dim=%s vocab=%d inter=%d tied=%s'
    % (model.config.num_hidden_layers,
       model.config.hidden_size,
       model.config.num_attention_heads,
       model.config.num_key_value_heads,
       getattr(model.config, 'head_dim', None),
       model.config.vocab_size,
       model.config.intermediate_size,
       getattr(model.config,
               'tie_word_embeddings', None)))
free_b, total_b = torch.cuda.mem_get_info()
log('cuda allocated=%.2f GB reserved=%.2f GB '
    'device total=%.2f GB device free=%.2f GB'
    % (torch.cuda.memory_allocated() / 1e9,
       torch.cuda.memory_reserved() / 1e9,
       total_b / 1e9, free_b / 1e9))

TEXTS = [
    'The weather was cold, so',
    'In a formal style, He studied every '
    'night because',
    'Regarding the experiment, The solution '
    'turned acidic, so',
]
ids_list = [[int(x) for x in tok(
    s, add_special_tokens=False)['input_ids']]
    for s in TEXTS]
log('lens=%s' % [len(x) for x in ids_list])


def fwd(ids):
    with torch.no_grad():
        out = model(torch.tensor(
            [ids], device='cuda'),
            use_cache=False)
    return out.logits[0, -1].detach() \
        .double().cpu().numpy()


# warmup (3)
for ids in ids_list:
    fwd(ids)
log('warmup done %.1fs' % (time.time() - t0))

# determinism check
lg1 = fwd(ids_list[0])
lg2 = fwd(ids_list[0])
dd = float(np.max(np.abs(lg1 - lg2)))
log('determinism diff=%.6e ok=%s'
    % (dd, dd == 0.0))

# timed block: 10 forwards
ts = []
for i in range(10):
    ids = ids_list[i % 3]
    ta = time.time()
    fwd(ids)
    ts.append(time.time() - ta)
ts = np.array(ts)
log('timed 10 fw: mean=%.3fs median=%.3fs '
    'min=%.3fs max=%.3fs'
    % (ts.mean(), float(np.median(ts)),
       ts.min(), ts.max()))

free_b2, _ = torch.cuda.mem_get_info()
log('after forwards: allocated=%.2f GB '
    'device free=%.2f GB'
    % (torch.cuda.memory_allocated() / 1e9,
       free_b2 / 1e9))

mean_fw = float(ts.mean())
# A1 authoritative forward count:
# 3 fam x (32 banks + 4 b0 + 1 b1 + 1 b7a)
#   = 114
# + 4 layers x 3 fam x (24 ladder
#   + 40 heads x 24 scan + 24 R_ALL) = 12096
N_FW = 12210
log('projection: A1 authoritative %d fw '
    '~ %.2f h (at mean %.3fs/fw); smoke '
    '(1 layer, K3=4, NP_USE=8) ~630 fw '
    '~ %.2f h'
    % (N_FW, N_FW * mean_fw / 3600.0,
       mean_fw, 630 * mean_fw / 3600.0))

del model, tok
gc.collect()
torch.cuda.empty_cache()
log('cleaned %.1fs total' % (time.time() - t0))
io.open(OUT, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('PROBE_OK')
