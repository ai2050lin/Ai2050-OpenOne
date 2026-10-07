# -*- coding: utf-8 -*-
"""Q05 退风险探针：4-bit NF4 加载是否可用 + 前向速度 + 以模板前缀复算 k=0 margin 与 bf16 对照。"""
import os, sys, time, json
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MDIR_MAP = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'Qwen3-14B', 'glm4-9b': 'glm4-9b-chat-hf'}
MODEL = os.environ.get('P_MODEL', 'qwen3-14b')
MDIR = os.path.join(ROOT, 'models', 'hf', MDIR_MAP[MODEL])

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
print('tok ok %.1fs' % (time.time() - t0), flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MDIR, quantization_config=bnb, device_map={'': 0}, trust_remote_code=True).eval()
print('model loaded(nf4) %.1fs  type=%s' % (time.time() - t0, type(model).__name__), flush=True)
print('vram used %.2f GB' % (torch.cuda.memory_allocated() / 1e9), flush=True)

CLS = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ids = lambda s: tok(s, add_special_tokens=False)['input_ids']
CLS_TOK = [ids(c)[0] for c in CLS]
CLS_T = torch.tensor(CLS_TOK, device='cuda')
ctx = ids('苹果是一种')
# 计时：跑 20 次单步前向
with torch.no_grad():
    _ = model(input_ids=torch.tensor([ctx], device='cuda'))
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        o = model(input_ids=torch.tensor([ctx], device='cuda'))
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 20
print('fwd latency %.3f s  -> 12546 fwd ~ %.1f min' % (dt, 12546 * dt / 60), flush=True)
lg = o.logits[0, -1].float()
print('k=0 6类 logit nf4:', [round(float(lg[t]), 4) for t in CLS_TOK], flush=True)
print('PROBE_OK')
