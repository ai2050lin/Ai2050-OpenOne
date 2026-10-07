# -*- coding: utf-8 -*-
"""
Phase 15 可行性探针 v2：禁用 disk offload，实测前向墙钟。
A. 量化后端可用性（bitsandbytes / torchao）
B. bf16 + max_memory(禁 disk) 的层放置与单前向耗时
C. bf16 + max_memory 下 hook 写入的 dtype/device
用法：python n2h1a8_feas_probe2.py [glm4|qwen3|bnb]
"""
import os
import sys
import io
import json
import time
import gc
import traceback
import shutil

import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
OUT = os.path.join(PT, 'feas_probe2_phase15.txt')
_L = []


def w(s=''):
    _L.append(str(s))
    print(s)
    sys.stdout.flush()


def flush():
    io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(_L) + '\n')


def ram():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return '%.1f/%.1f GB' % (vm.available / 1e9, vm.total / 1e9)
    except Exception:
        return 'NA'


def vram():
    try:
        f, t = torch.cuda.mem_get_info(0)
        return '%.1f/%.1f GB' % (f / 1e9, t / 1e9)
    except Exception:
        return 'NA'


def part_bnb():
    w('=== A. quantization backends ===')
    for m in ['bitsandbytes', 'torchao', 'optimum', 'accelerate', 'transformers']:
        try:
            mod = __import__(m)
            w('  %-14s %s' % (m, getattr(mod, '__version__', '?')))
        except Exception as e:
            w('  %-14s MISSING (%s)' % (m, type(e).__name__))
    try:
        import torch
        w('  torch.cuda.is_bf16_supported = %s' % torch.cuda.is_bf16_supported())
    except Exception:
        pass
    t, u, fr = shutil.disk_usage('D:/')
    w('  D: free %.1f GB / %.1f GB' % (fr / 1e9, t / 1e9))
    flush()


def part_bf16(dirname, tag, gpu_gib, cpu_gib):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    MDIR = os.path.join(ROOT, 'models', 'hf', dirname)
    w('')
    w('=== B. %s bf16 + max_memory(gpu=%sGiB, cpu=%sGiB) ===' % (tag, gpu_gib, cpu_gib))
    w('  before: RAM %s ; VRAM %s' % (ram(), vram()))
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
    t0 = time.time()
    kw = dict(dtype=torch.bfloat16, trust_remote_code=True,
              attn_implementation='eager', device_map='auto',
              low_cpu_mem_usage=True,
              max_memory={0: '%dGiB' % int(gpu_gib), 'cpu': '%dGiB' % int(cpu_gib)})
    model = AutoModelForCausalLM.from_pretrained(MDIR, **kw)
    model.eval()
    w('  loaded %.1fs ; RAM %s ; VRAM %s' % (time.time() - t0, ram(), vram()))
    dc = {}
    for k, v in model.hf_device_map.items():
        dc[str(v)] = dc.get(str(v), 0) + 1
    w('  device_map summary: %s' % dc)
    layers = model.model.layers
    L = len(layers)
    place = {}
    for i, ly in enumerate(layers):
        d = str(next(ly.parameters()).device)
        place.setdefault(d, []).append(i)
    w('  L=%d ; layer placement: %s' %
      (L, {k: 'L%d..L%d (n=%d)' % (v[0], v[-1], len(v)) for k, v in place.items()}))
    ii = torch.tensor([tok.encode('%s是一种' % '苹果', add_special_tokens=False)])
    with torch.no_grad():
        o = model(input_ids=ii)
        lg1 = o.logits[0, -1].float().cpu().numpy()
        ts = []
        for _ in range(3):
            t1 = time.time()
            o = model(input_ids=ii)
            _ = o.logits[0, -1].float().cpu().numpy()
            ts.append(time.time() - t1)
        lg2 = _
    w('  forward latencies (s) = %s ; median %.3f' %
      (['%.3f' % x for x in ts], float(np.median(ts))))
    w('  determinism max|d| = %.3e' % float(np.max(np.abs(lg1 - lg2))))
    # hook 冒烟（在 GPU 层与 CPU 层各一处）
    for idx in sorted(set([L // 4, L // 2, (3 * L) // 4])):
        dev = str(next(layers[idx].parameters()).device)
        try:
            vec = (np.random.default_rng(7).standard_normal(model.config.hidden_size) * 0.1).astype(np.float32)
            box = {}

            def hk(mod, inp, out, _vec=vec, _box=box):
                t = out[0] if isinstance(out, tuple) else out
                h = t.clone()
                h[0, -1, :] = torch.tensor(_vec, dtype=h.dtype, device=h.device)
                _box['dev'] = str(h.device)
                _box['dt'] = str(h.dtype)
                return h if not isinstance(out, tuple) else (h,) + tuple(out[1:])
            hh = layers[idx].register_forward_hook(hk)
            try:
                with torch.no_grad():
                    o3 = model(input_ids=ii)
            finally:
                hh.remove()
            lg3 = o3.logits[0, -1].float().cpu().numpy()
            w('  hook @L%-2d (layer device %-7s) OK write dev=%s dt=%s ; effect max|d|=%.3e' %
              (idx, dev, box.get('dev'), box.get('dt'), float(np.max(np.abs(lg3 - lg2)))))
        except Exception as e:
            w('  hook @L%-2d (layer device %-7s) FAILED: %r' % (idx, dev, e))
    lat = float(np.median(ts))
    for ns, na, npair in [(18, 14, 24), (12, 9, 24), (9, 9, 24), (9, 7, 24), (6, 7, 24)]:
        w('  budget sites=%d alphas=%d pairs=%d -> %d fw -> %.1f min @%.2fs' %
          (ns, na, npair, ns * na * npair, ns * na * npair * lat / 60, lat))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    w('  after unload: RAM %s ; VRAM %s' % (ram(), vram()))
    flush()


def main():
    global OUT
    sel = sys.argv[1] if len(sys.argv) > 1 else 'all'
    OUT = os.path.join(PT, 'feas_probe2_%s.txt' % sel)
    w('Phase 15 feas probe v2 ; clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    w('python %s ; torch %s' % (sys.version.split()[0], torch.__version__))
    flush()
    if sel in ('all', 'bnb'):
        part_bnb()
    if sel in ('all', 'glm4'):
        try:
            part_bf16('glm4-9b-chat-hf', 'glm4-9b', 14, 24)
        except Exception:
            w('!! glm4 FAILED'); w(traceback.format_exc()); flush()
    if sel in ('all', 'qwen3'):
        try:
            part_bf16('Qwen3-14B', 'qwen3-14b', 14, 24)
        except Exception:
            w('!! qwen3-14b FAILED'); w(traceback.format_exc()); flush()


if __name__ == '__main__':
    main()
