# -*- coding: utf-8 -*-
"""p3161 14b NF4 load diagnostic: resource-monitored minimal repro.
EXP-A: all-GPU NF4 (current free_vram 13.9GB > 11GB need) - reproduce segfault?
A monitor thread samples VRAM/RAM every 2s and flushes to file, so the last
lines before a segfault show the resource state at crash time.
"""
import os, sys, time, ctypes, threading
# EXP-B: force sync loading (transformers 5.14 core_model_loading.py L1567) to stop
# 4-worker async futures from materializing full bf16 weights into RAM (crash at ram_free~0.4GB)
os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

T0 = time.time()
MDIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\Qwen3-14B'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3161_diag3.txt'
LINES = []
MON_STOP = {'v': False}


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def flush():
    try:
        with open(OUT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(LINES[-300:]) + '\n')
    except Exception:
        pass


def log(s):
    LINES.append('[%7.1f] %s' % (time.time() - T0, s))
    flush()
    try:
        print(LINES[-1], flush=True)
    except Exception:
        pass


def mon():
    while not MON_STOP['v']:
        try:
            fv, tv = torch.cuda.mem_get_info()
            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            LINES.append('MON t=%7.1f vram_free=%5.2f ram_free=%5.2f load=%d%%' % (
                time.time() - T0, fv / (1 << 30), m.ullAvailPhys / (1 << 30), m.dwMemoryLoad))
            flush()
        except Exception as e:
            LINES.append('MON err %r' % (e,))
            flush()
        time.sleep(2)


log('diag3 begin torch=%s' % torch.__version__)
fv, tv = torch.cuda.mem_get_info()
log('pre-load vram_free=%.2f / %.2f GB' % (fv / (1 << 30), tv / (1 << 30)))

th = threading.Thread(target=mon, daemon=True)
th.start()

log('EXP-A: all-GPU NF4 from_pretrained begin')
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
t1 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MDIR, quantization_config=bnb, device_map={'': 0}, trust_remote_code=True).eval()
log('EXP-A model loaded in %.1f s (vram_alloc=%.2f GB)' % (
    time.time() - t1, torch.cuda.memory_allocated() / 1e9))

tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
ids = tok('The quick brown fox jumps over the lazy dog. ' * 3, return_tensors='pt')['input_ids'].cuda()
with torch.no_grad():
    o = model(input_ids=ids, output_hidden_states=True)
log('forward OK hidden_states=%d logits=%s' % (len(o.hidden_states), tuple(o.logits.shape)))
MON_STOP['v'] = True
log('DIAG3 PASS (all-GPU NF4 works now)')
flush()
