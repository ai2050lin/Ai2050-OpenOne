# -*- coding: utf-8 -*-
"""Q8-D 地基探针 2：GLM 的显存切分与批大小对前向耗时的影响。

探针 1 结果：max_memory={0:13GiB} 时 29 个模块在 GPU、15 个在 CPU，单条 8 token 前向 8.83 s。
8.83 s 主要不是算力，是**每个 forward 都要把 CPU 侧的权重搬过 PCIe**。
这一步只找"能跑网格"的配置：(1) 把更多层塞进 GPU；(2) 批量化摊薄。
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch

GLM = 'D:/develop/model/ZhipuAI/glm-4-9b-chat-hf'
OUT = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def bench(m, tok, bs, sl, reps=2):
    txt = ['苹果、香蕉、橘子、今天天气很'[:sl]] * bs
    ids = tok(txt, return_tensors='pt', padding=True).to('cuda:0')
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(reps):
            o = m(**ids).logits[:, -1, :].float()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    return dt, tuple(ids['input_ids'].shape), float(o.abs().mean())


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from collections import Counter
    tok = AutoTokenizer.from_pretrained(GLM)
    for mm in ('15GiB', '14GiB'):
        say('=' * 60)
        say('max_memory 0:%s' % mm)
        try:
            t0 = time.time()
            m = AutoModelForCausalLM.from_pretrained(
                GLM, dtype=torch.bfloat16, device_map='auto',
                max_memory={0: mm, 'cpu': '48GiB'}, low_cpu_mem_usage=True)
            m.eval()
            say('加载 %.1f s 已分配 %.2f GB' % (time.time() - t0,
                                          torch.cuda.memory_allocated() / 1e9))
            say('device_map %s' % dict(Counter(str(v) for v in m.hf_device_map.values())))
            for bs, sl in ((1, 8), (16, 12), (48, 12), (96, 12)):
                try:
                    dt, shp, mag = bench(m, tok, bs, sl)
                    say('bs=%-3d len=%-2d -> %.3f s   shape %s  |logit|%.2f'
                        % (bs, sl, dt, shp, mag))
                except torch.cuda.OutOfMemoryError as e:
                    say('bs=%-3d OOM' % bs)
                    torch.cuda.empty_cache()
            del m
            torch.cuda.empty_cache()
        except Exception as e:
            say('FAIL %s: %s' % (type(e).__name__, str(e)[:240]))
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
    Path('tests/gpt5_temp/q8d_probe2.txt').write_text('\n'.join(OUT) + '\n',
                                                      encoding='utf-8', newline='')
    say('WROTE')
