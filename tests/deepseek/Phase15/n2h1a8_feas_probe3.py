# -*- coding: utf-8 -*-
"""
Phase 15 可行性探针 v3：bitsandbytes nf4（4bit 权重 + bf16 计算）统一口径。
动机：bf16 下 qwen3-14b(29.5GB) 需要 CPU 侧 ~15.5GB 而本机 RAM 仅 ~17GB 可用 -> 被 OOM 杀死；
      glm4-9b bf16+max_memory 可行但口径将与 14B 不一致。为「两臂口径一致」改用 nf4。
本探针回答：nf4 下三模型能否全载 GPU、前向墙钟、双前向确定性、hook 写入兼容性。
用法：python n2h1a8_feas_probe3.py [4b|glm4|qwen3|all]
"""
import os
import sys
import io
import json
import time
import gc
import traceback

import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
_L = []
OUT = os.path.join(PT, 'feas_probe3_all.txt')


def w(s=''):
    _L.append(str(s)); print(s); sys.stdout.flush()


def flush():
    io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(_L) + '\n')


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


def probe(dirname, tag, gpu_only=True):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    MDIR = os.path.join(ROOT, 'models', 'hf', dirname)
    w('')
    w('=' * 74)
    w('MODEL %s  nf4(4bit, compute=bf16, double_quant=True)  before: %s' % (tag, mem()))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    kw = dict(quantization_config=bnb, trust_remote_code=True,
              attn_implementation='eager', low_cpu_mem_usage=True)
    kw['device_map'] = ({'': 0} if gpu_only else 'auto')
    try:
        model = AutoModelForCausalLM.from_pretrained(MDIR, **kw)
    except Exception:
        w('  !! load failed (gpu_only=%s)' % gpu_only); w(traceback.format_exc()); flush(); return None
    model.eval()
    w('  loaded %.1fs ; %s' % (time.time() - t0, mem()))
    L = len(model.model.layers)
    hid = model.config.hidden_size
    # 权重驻留设备统计
    devs = {}
    for nm, p in model.named_parameters():
        d = str(p.device); devs[d] = devs.get(d, 0) + 1
    w('  L=%d hid=%d ; param devices: %s' % (L, hid, devs))
    dtypes = {}
    for nm, p in model.named_parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + 1
    w('  param dtypes: %s' % dtypes)
    ii = torch.tensor([tok.encode('%s是一种' % '苹果', add_special_tokens=False)], device='cuda')
    with torch.no_grad():
        o = model(input_ids=ii)
        lg1 = o.logits[0, -1].float().cpu().numpy()
        ts = []
        for _ in range(3):
            t1 = time.time(); o = model(input_ids=ii)
            lg2 = o.logits[0, -1].float().cpu().numpy(); ts.append(time.time() - t1)
    w('  forward latencies (s) = %s ; median %.3f' % (['%.3f' % x for x in ts], float(np.median(ts))))
    w('  determinism max|d| = %.3e' % float(np.max(np.abs(lg1 - lg2))))
    # hook 冒烟（浅层/中层/深层）
    for idx in sorted(set([L // 4, L // 2, (3 * L) // 4, L - 1])):
        try:
            vec = (np.random.default_rng(7).standard_normal(hid) * 0.1).astype(np.float32)
            box = {}

            def hk(mod, inp, out, _v=vec, _b=box):
                t = out[0] if isinstance(out, tuple) else out
                h = t.clone(); h[0, -1, :] = torch.tensor(_v, dtype=h.dtype, device=h.device)
                _b['dev'] = str(h.device); _b['dt'] = str(h.dtype); _b['outdt'] = str(t.dtype)
                return h if not isinstance(out, tuple) else (h,) + tuple(out[1:])
            hh = model.model.layers[idx].register_forward_hook(hk)
            try:
                with torch.no_grad():
                    o3 = model(input_ids=ii)
            finally:
                hh.remove()
            lg3 = o3.logits[0, -1].float().cpu().numpy()
            w('  hook @L%-2d OK out_dtype=%s write dev=%s dt=%s ; effect max|d|=%.3e' %
              (idx, box.get('outdt'), box.get('dev'), box.get('dt'), float(np.max(np.abs(lg3 - lg2)))))
        except Exception as e:
            w('  hook @L%-2d FAILED: %r' % (idx, e))
    lat = float(np.median(ts))
    for ns, na, npair in [(18, 14, 24), (18, 9, 24), (12, 9, 24)]:
        w('  budget sites=%d alphas=%d pairs=%d -> %d fw -> %.1f min @%.2fs' %
          (ns, na, npair, ns * na * npair, ns * na * npair * lat / 60, lat))
    # 与 bf16 的一致性抽查（仅 4b：与已知 bf16 量级对照不做在此，仅报输出）
    del model; gc.collect(); torch.cuda.empty_cache()
    w('  after unload: %s' % mem())
    flush()
    return dict(tag=tag, dir=dirname, L=L, hid=hid, fw_median_s=lat,
                determinism=float(np.max(np.abs(lg1 - lg2))), param_devices=devs, param_dtypes=dtypes)


def main():
    global OUT
    sel = sys.argv[1] if len(sys.argv) > 1 else 'all'
    OUT = os.path.join(PT, 'feas_probe3_%s.txt' % sel)
    w('Phase 15 feas probe v3 (nf4) ; clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    w('torch %s ; cuda %s ; %s' % (torch.__version__, torch.cuda.is_available(), mem()))
    flush()
    jobs = [('qwen3-4b', 'qwen3-4b'), ('glm4-9b-chat-hf', 'glm4-9b'), ('Qwen3-14B', 'qwen3-14b')]
    if sel != 'all':
        jobs = [j for j in jobs if sel in j[0] or sel in j[1]]
    recs = []
    for d, t in jobs:
        try:
            r = probe(d, t)
            if r:
                recs.append(r)
        except Exception:
            w('!! %s FAILED' % t); w(traceback.format_exc()); flush()
    io.open(os.path.join(PT, 'feas_probe3_%s.json' % sel), 'w', encoding='utf-8', newline='\n').write(
        json.dumps(recs, ensure_ascii=False, indent=1))
    w('REPORT -> %s' % OUT)


if __name__ == '__main__':
    main()
