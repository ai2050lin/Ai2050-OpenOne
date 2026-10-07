# -*- coding: utf-8 -*-
"""Q05 资源前置探针：模型体量 / 主机内存 / 量化后端可用性 / offload 可行性。只读，不写产物。"""
import os, sys, json, importlib.util
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = r'D:\AI2050\Ai2050-OpenOne'
out = []
def P(s):
    out.append(str(s)); print(s, flush=True)

# 1) 模型体量
P('=== models ===')
for name in ['qwen3-4b', 'Qwen3-14B', 'glm4-9b-chat-hf']:
    d = os.path.join(ROOT, 'models', 'hf', name)
    if not os.path.isdir(d):
        P('  %-16s MISSING' % name); continue
    tot = 0; nst = 0; cfgp = os.path.join(d, 'config.json')
    for f in os.listdir(d):
        fp = os.path.join(d, f)
        if os.path.isfile(fp):
            tot += os.path.getsize(fp)
            if f.endswith('.safetensors'): nst += 1
    nl = dh = None; dt = None
    if os.path.exists(cfgp):
        cfg = json.load(open(cfgp, encoding='utf-8'))
        nl = cfg.get('num_hidden_layers'); dh = cfg.get('hidden_size')
        dt = cfg.get('torch_dtype') or cfg.get('dtype')
    P('  %-16s %7.2f GB  layers=%s  hidden=%s  dtype=%s  safetensors=%d'
      % (name, tot / 1e9, nl, dh, dt, nst))

# 2) 主机内存
P('=== host RAM ===')
try:
    import ctypes
    class MS(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
    ms = MS(); ms.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    P('  total=%.1f GB  avail=%.1f GB  load=%d%%'
      % (ms.ullTotalPhys / 1e9, ms.ullAvailPhys / 1e9, ms.dwMemoryLoad))
except Exception as e:
    P('  RAM probe failed: %s' % e)

# 3) 量化/offload 后端
P('=== backends ===')
for mod in ['torch', 'transformers', 'accelerate', 'bitsandbytes']:
    spec = importlib.util.find_spec(mod)
    if spec is None:
        P('  %-14s NOT INSTALLED' % mod); continue
    try:
        m = __import__(mod)
        P('  %-14s %s' % (mod, getattr(m, '__version__', '?')))
    except Exception as e:
        P('  %-14s import-fail: %s' % (mod, e))

# 4) torch cuda 细节
P('=== cuda ===')
try:
    import torch
    P('  cuda_available=%s  bf16_supported=%s' % (torch.cuda.is_available(), torch.cuda.is_bf16_supported()))
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        P('  dev=%s  vram=%.1f GB  cc=%d.%d' % (p.name, p.total_memory / 1e9, p.major, p.minor))
except Exception as e:
    P('  cuda probe failed: %s' % e)

open(os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q05_resource_probe.txt'),
     'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('PROBE_DONE')
