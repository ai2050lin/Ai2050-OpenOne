# -*- coding: utf-8 -*-
"""Phase 3228 C6 arbitration: is the 14B marg disagreement bf16 arithmetic or a bug?"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3194_g5d2_ftr26_heldout_upgrade as p94

SRC_DIR = p94.SRC_DIR
OUT = []
m = 'qwen3-14b'
D, NL = 5120, 40
fin = NL
eps_cfg = json.load(open(os.path.join(
    r'C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen3-14B',
    'snapshots', os.listdir(os.path.join(
        r'C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen3-14B',
        'snapshots')[0], 'config.json')), encoding='utf-8')) \
    if False else None
import glob
SNAP = glob.glob(os.path.join(os.path.expanduser('~'), '.cache', 'huggingface',
                              'hub', 'models--Qwen--Qwen3-14B', 'snapshots', '*'))[0]
cfg = json.load(open(os.path.join(SNAP, 'config.json'), encoding='utf-8'))
OUT.append('14B cfg: eps=%s tie=%s hidden=%s vocab=%s'
           % (cfg.get('rms_norm_eps'), cfg.get('tie_word_embeddings'),
              cfg['hidden_size'], cfg['vocab_size']))
eps = float(cfg.get('rms_norm_eps', 1e-6))

from safetensors import safe_open
from tokenizers import Tokenizer
tk = Tokenizer.from_file(os.path.join(SNAP, 'tokenizer.json'))
CLS = ['水果', '动物', '交通工具', '家具', '金属', '颜色', '乐器', '天气', '运动', '电器']
cls_tok = [tk.encode(c, add_special_tokens=False).ids[0] for c in CLS]

with safe_open(os.path.join(SNAP, 'model-00008-of-00008.safetensors'), framework='pt') as f:
    W_t = f.get_tensor('lm_head.weight')          # torch bf16
    g_t = f.get_tensor('model.norm.weight')
W_fp32 = W_t.float().numpy()
gamma = g_t.float().numpy().reshape(-1)

z69 = np.load(p94.SRC['p3169_14b'])
MARG = z69['marg'].reshape(2190, 10)
Hf = z69['H'][:, :, fin, :].reshape(2190, D).astype(np.float32)
del z69

import torch
torch.manual_seed(0)
H_t = torch.from_numpy(Hf)                        # fp32
W_torch = W_t.clone()

def lnf_np(X):
    return (X / np.sqrt((X ** 2).mean(axis=-1, keepdims=True) + eps)) * gamma

rows = np.arange(0, 2190, 8)                      # 274-row sample
Hs = Hf[rows]
Mg = MARG[rows]

# path 1: my fp32 exact
B = lnf_np(Hs)
c_fp32 = B @ W_fp32[cls_tok].T

# path 2: bf16-emulated head (x bf16, W bf16, fp32 accum via torch)
xt = torch.from_numpy(Hs)
xg = (xt / torch.sqrt((xt ** 2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16) * g_t.to(torch.bfloat16)
c_bf16 = (xg.to(torch.bfloat16) @ W_t[cls_tok].to(torch.bfloat16).T).float().numpy()

am_c = Mg.argmax(1)
am_32 = c_fp32.argmax(1)
am_bf = c_bf16.argmax(1)
OUT.append('sample rows=%d' % len(rows))
OUT.append('agree(fp32 vs cached) = %.4f' % float((am_32 == am_c).mean()))
OUT.append('agree(bf16-emul vs cached) = %.4f' % float((am_bf == am_c).mean()))
OUT.append('agree(fp32 vs bf16-emul) = %.4f' % float((am_32 == am_bf).mean()))
for i in range(len(rows)):
    if i >= 6:
        break
    OUT.append('row %d cached %s | fp32 %s | bf16 %s' % (
        rows[i], np.round(Mg[i] - Mg[i].mean(), 2).tolist(),
        np.round(c_fp32[i] - c_fp32[i].mean(), 2).tolist(),
        np.round(c_bf16[i] - c_bf16[i].mean(), 2).tolist()))
# magnitude of the fp32-vs-bf16 logit deltas
dlt = c_fp32 - c_bf16
OUT.append('fp32-vs-bf16 logit delta: mean=%.4f p95abs=%.4f maxabs=%.4f'
           % (float(np.abs(dlt).mean()), float(np.percentile(np.abs(dlt), 95)),
              float(np.abs(dlt).max())))
dlt2 = c_bf16 - (Mg - Mg.mean(1, keepdims=True))
OUT.append('bf16-emul-vs-cached delta: mean=%.4f p95abs=%.4f maxabs=%.4f'
           % (float(np.abs(dlt2).mean()), float(np.percentile(np.abs(dlt2), 95)),
              float(np.abs(dlt2).max())))
dlt3 = c_fp32 - (Mg - Mg.mean(1, keepdims=True))
OUT.append('fp32-vs-cached delta: mean=%.4f p95abs=%.4f maxabs=%.4f'
           % (float(np.abs(dlt3).mean()), float(np.percentile(np.abs(dlt3), 95)),
              float(np.abs(dlt3).max())))

with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\phase3228_c6arb_out.txt', 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(OUT) + '\n')
print('arb done')
