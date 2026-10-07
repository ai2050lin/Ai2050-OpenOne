# -*- coding: utf-8 -*-
"""
Phase 15 探针 v4：专攻 qwen3-14b 的 nf4 加载（前两次在加载阶段被硬杀，无 Python 栈 -> RAM/CUDA 峰值）。
逐个配置在独立进程中试：
  c1 : nf4  device_map={'':0}                       (gpu only, double_quant=True)
  c2 : nf4  device_map='auto' max_memory gpu14/cpu24
  c3 : nf4  device_map={'':0} double_quant=False
  c4 : nf4  device_map='auto' max_memory gpu13/cpu22, double_quant=True
用法：python n2h1a8_feas_probe4.py <c1|c2|c3|c4>
"""
import os
import sys
import io
import time
import json
import gc
import traceback

import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
_L = []


def w(s=''):
    _L.append(str(s)); print(s, flush=True)


def mem():
    try:
        import psutil
        vm = psutil.virtual_memory(); r = '%.1f/%.1fGB' % (vm.available / 1e9, vm.total / 1e9)
    except Exception:
        r = 'NA'
    try:
        f, t = torch.cuda.mem_get_info(0); v = '%.1f/%.1fGB' % (f / 1e9, t / 1e9)
    except Exception:
        v = 'NA'
    return 'RAM %s VRAM %s' % (r, v)


def main():
    cfg = sys.argv[1] if len(sys.argv) > 1 else 'c1'
    OUT = os.path.join(PT, 'feas_probe4_%s.txt' % cfg)
    tag = cfg
    w('probe4 cfg=%s clock %s ; %s' % (cfg, time.strftime('%H:%M:%S'), mem()))
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    MDIR = os.path.join(ROOT, 'models', 'hf', 'Qwen3-14B')
    dq = (cfg != 'c3')
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=dq)
    kw = dict(quantization_config=bnb, trust_remote_code=True,
              attn_implementation='eager', low_cpu_mem_usage=True)
    if cfg in ('c1', 'c3'):
        kw['device_map'] = {'': 0}
    elif cfg == 'c2':
        kw['device_map'] = 'auto'
        kw['max_memory'] = {0: '14GiB', 'cpu': '24GiB'}
    else:
        kw['device_map'] = 'auto'
        kw['max_memory'] = {0: '13GiB', 'cpu': '22GiB'}
    w('  kwargs device_map=%r max_memory=%r double_quant=%s' %
      (kw.get('device_map'), kw.get('max_memory'), dq))
    sys.stdout.flush()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MDIR, **kw)
    model.eval()
    w('  LOADED %.1fs ; %s' % (time.time() - t0, mem()))
    dc = {}
    for k, v in getattr(model, 'hf_device_map', {}).items():
        dc[str(v)] = dc.get(str(v), 0) + 1
    w('  hf_device_map summary: %s' % dc)
    dev = {}
    for nm, p in model.named_parameters():
        dev[str(p.device)] = dev.get(str(p.device), 0) + 1
    w('  param devices: %s' % dev)
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
    ii = torch.tensor([tok.encode('%s是一种' % '苹果', add_special_tokens=False)], device='cuda')
    with torch.no_grad():
        o = model(input_ids=ii)
        lg1 = o.logits[0, -1].float().cpu().numpy()
        ts = []
        for _ in range(3):
            t1 = time.time(); o = model(input_ids=ii)
            lg2 = o.logits[0, -1].float().cpu().numpy(); ts.append(time.time() - t1)
    w('  latencies %s median %.3f ; determinism %.3e' %
      (['%.3f' % x for x in ts], float(np.median(ts)), float(np.max(np.abs(lg1 - lg2)))))
    L = len(model.model.layers)
    hid = model.config.hidden_size
    for idx in sorted(set([L // 4, L // 2, (3 * L) // 4, L - 1])):
        try:
            vec = (np.random.default_rng(7).standard_normal(hid) * 0.1).astype(np.float32)
            box = {}

            def hk(mod, inp, out, _v=vec, _b=box):
                t = out[0] if isinstance(out, tuple) else out
                h = t.clone(); h[0, -1, :] = torch.tensor(_v, dtype=h.dtype, device=h.device)
                _b['dt'] = str(t.dtype); _b['dev'] = str(h.device)
                return h if not isinstance(out, tuple) else (h,) + tuple(out[1:])
            hh = model.model.layers[idx].register_forward_hook(hk)
            try:
                with torch.no_grad():
                    o3 = model(input_ids=ii)
            finally:
                hh.remove()
            lg3 = o3.logits[0, -1].float().cpu().numpy()
            w('  hook @L%-2d OK out_dtype=%s write dev=%s ; effect max|d|=%.3e' %
              (idx, box.get('dt'), box.get('dev'), float(np.max(np.abs(lg3 - lg2)))))
        except Exception as e:
            w('  hook @L%-2d FAILED %r' % (idx, e))
    lat = float(np.median(ts))
    for ns, na, npair in [(18, 14, 24), (18, 9, 24), (12, 9, 24)]:
        w('  budget sites=%d alphas=%d pairs=%d -> %d fw -> %.1f min @%.2fs' %
          (ns, na, npair, ns * na * npair, ns * na * npair * lat / 60, lat))
    w('  SUCCESS cfg=%s ; median %.3fs' % (cfg, lat))
    io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(_L) + '\n')
    io.open(os.path.join(PT, 'feas_probe4_%s.json' % cfg), 'w', encoding='utf-8').write(
        json.dumps({'cfg': cfg, 'ok': True, 'median_s': lat, 'param_devices': dev,
                    'hf_device_map': dc}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        _L.append('EXC:'); _L.append(traceback.format_exc())
        io.open(os.path.join(PT, 'feas_probe4_%s.txt' % (sys.argv[1] if len(sys.argv) > 1 else 'cX')),
                'w', encoding='utf-8', newline='\n').write('\n'.join(_L) + '\n')
        raise
