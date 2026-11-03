# -*- coding: utf-8 -*-
"""Q8-D 地基探针：inputs_embeds 前向 / tied 解绑 / GLM 显存边界。

这一步不是实验，只回答"能不能做"：
  P1 Qwen3-4B 能否接 inputs_embeds 前向，且与 embed_tokens 路径逐位一致
  P2 Qwen 的 lm_head 是否与 embed_tokens 共享存储，能否解绑（copy 成独立 Parameter）
  P3 GLM-4-9B（18 GB 权重 / 17.1 GB 显存）在 max_memory offload 下能否加载并前向
  P4 两个模型的"可定义频率 token 掩码"能否与词表对齐（U+FFFD 残片要排除）
"""
import os
import sys
import json
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch

OUT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


QWEN = 'D:/develop/model/Qwen/Qwen3-4B'
GLM = 'D:/develop/model/ZhipuAI/glm-4-9b-chat-hf'


def cpu_ok():
    say('torch %s cuda %s' % (torch.__version__, torch.cuda.is_available()))
    if torch.cuda.is_available():
        f, t = torch.cuda.mem_get_info()
        say('GPU %s free %.2f / total %.2f GB'
            % (torch.cuda.get_device_name(0), f / 1e9, t / 1e9))


def probe_qwen():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN)
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.bfloat16, device_map='cuda:0')
    m.eval()
    say('[P1] Qwen 加载 %.1f s  显存 %.2f GB'
        % (time.time() - t0, torch.cuda.memory_allocated() / 1e9))
    emb = m.get_input_embeddings()
    head = m.get_output_embeddings()
    say('[P2] tie_word_embeddings=%s' % getattr(m.config, 'tie_word_embeddings', None))
    say('[P2] emb is head: %s' % (emb.weight is head.weight))
    say('[P2] data_ptr 相同: %s' % (emb.weight.data_ptr() == head.weight.data_ptr()))
    say('[P2] lm_head 是 Linear: %s  bias=%s'
        % (isinstance(head, torch.nn.Linear), getattr(head, 'bias', None) is not None))

    txt = ['苹果、香蕉、橘子、', '今天天气很']
    ids = tok(txt, return_tensors='pt', padding=True).to('cuda:0')
    with torch.no_grad():
        o1 = m(**ids).logits[:, -1, :].float()
        E = emb(ids['input_ids'])
        o2 = m(inputs_embeds=E, attention_mask=ids['attention_mask']).logits[:, -1, :].float()
    dmax = float((o1 - o2).abs().max())
    say('[P1] embed 路径 vs inputs_embeds 路径 logits max|Δ|=%.3e  %s'
        % (dmax, 'PASS' if dmax < 1e-2 else 'FAIL'))

    # 解绑测试
    w0 = head.weight.detach().clone()
    newh = torch.nn.Linear(w0.shape[1], w0.shape[0], bias=False,
                           dtype=w0.dtype, device=w0.device)
    newh.weight.data.copy_(w0)
    m.lm_head = newh
    say('[P2] 解绑后 emb is head: %s' % (m.get_input_embeddings().weight is
                                        m.get_output_embeddings().weight))
    with torch.no_grad():
        o3 = m(**ids).logits[:, -1, :].float()
    say('[P2] 解绑后 logits 与原始 max|Δ|=%.3e  %s'
        % (float((o1 - o3).abs().max()),
           'PASS(解绑无副作用)' if float((o1 - o3).abs().max()) < 1e-2 else 'FAIL'))
    del m, o1, o2, o3
    torch.cuda.empty_cache()


def probe_glm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GLM)
    say('[P3] GLM 权重目录 %.1f GB'
        % (sum(f.stat().st_size for f in Path(GLM).glob('*.safetensors')) / 1e9))
    t0 = time.time()
    try:
        m = AutoModelForCausalLM.from_pretrained(
            GLM, dtype=torch.bfloat16, device_map='auto',
            max_memory={0: '13GiB', 'cpu': '40GiB'},
            low_cpu_mem_usage=True)
        m.eval()
        say('[P3] GLM 加载 %.1f s  显存 %.2f GB'
            % (time.time() - t0, torch.cuda.memory_allocated() / 1e9))
        hd = getattr(m, 'hf_device_map', None)
        if hd:
            from collections import Counter
            say('[P3] device_map 分布 %s' % dict(Counter(str(v) for v in hd.values())))
        txt = ['苹果、香蕉、橘子、']
        ids = tok(txt, return_tensors='pt').to('cuda:0')
        t1 = time.time()
        with torch.no_grad():
            o = m(**ids).logits[:, -1, :].float()
        say('[P3] GLM 前向 %.2f s  logits %s' % (time.time() - t1, tuple(o.shape)))
        head = m.get_output_embeddings()
        say('[P3] GLM tie=%s  emb is head: %s'
            % (getattr(m.config, 'tie_word_embeddings', None),
               m.get_input_embeddings().weight is head.weight))
        del m, o
        torch.cuda.empty_cache()
    except Exception as e:
        say('[P3] GLM 加载失败: %s: %s' % (type(e).__name__, str(e)[:300]))


def probe_mask():
    from transformers import AutoTokenizer
    from wordfreq import zipf_frequency
    for name, p in (('Qwen', QWEN), ('GLM', GLM)):
        tok = AutoTokenizer.from_pretrained(p)
        V = len(tok)
        t0 = time.time()
        dec = [tok.decode([i]) for i in range(V)]
        n_ok = 0
        for s in dec:
            if not s or '\ufffd' in s:
                continue
            if zipf_frequency(s, 'zh') > 0:
                n_ok += 1
        say('[P4] %s V=%d 有频率定义 %d (%.1f%%) 解码耗时 %.1f s'
            % (name, V, n_ok, 100.0 * n_ok / V, time.time() - t0))


if __name__ == '__main__':
    cpu_ok()
    probe_mask()
    probe_qwen()
    probe_glm()
    p = Path('tests/gpt5_temp/q8d_probe.txt')
    p.write_text('\n'.join(OUT) + '\n', encoding='utf-8', newline='')
    say('WROTE', p)
