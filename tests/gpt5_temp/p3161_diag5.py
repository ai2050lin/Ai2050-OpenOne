# -*- coding: utf-8 -*-
"""diag5 (under stock 5.14, NO PYTHONPATH): load the pre-quantized NF4 checkpoint,
verify (a) RAM stays flat during load (no bf16 materialize), (b) injection hook at
layers[L_MID-1] propagates to hidden_states slot L_MID (5.14 slot semantics restored),
(c) 41 hidden slots."""
import os, time, ctypes, threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

T0 = time.time()
MDIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\Qwen3-14B-bnb-nf4'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3161_diag5.txt'
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


log('diag5 begin (stock transformers %s)' % __import__('transformers').__version__)
th = threading.Thread(target=mon, daemon=True)
th.start()

# load as PRE-QUANTIZED: pass the same bnb config; transformers detects 4bit checkpoint
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
t1 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MDIR, quantization_config=bnb, device_map={'': 0}, trust_remote_code=True).eval()
log('loaded pre-quantized nf4 in %.1f s (vram_alloc=%.2f GB, in4bit=%s)' % (
    time.time() - t1, torch.cuda.memory_allocated() / 1e9, model.is_loaded_in_4bit))

tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
NL = model.config.num_hidden_layers
L_MID = NL // 2
D = model.config.hidden_size

# injection-slot semantics check: post-hook at layers[L_MID-1] must appear at slot L_MID
cap = {}
def _inj(module, args, output):
    if cap.get('on'):
        out0 = output[0] if isinstance(output, tuple) else output
        new0 = out0.clone()
        new0[:, -1, :] = new0[:, -1, :] + 50.0
        if isinstance(output, tuple):
            return (new0,) + tuple(output[1:])
        return new0
    return None

h = model.model.layers[L_MID - 1].register_forward_hook(_inj)
ids = tok('The quick brown fox jumps over the lazy dog. ' * 3, return_tensors='pt')['input_ids'].cuda()
cap['on'] = False
with torch.no_grad():
    o0 = model(input_ids=ids, output_hidden_states=True)
cap['on'] = True
with torch.no_grad():
    o1 = model(input_ids=ids, output_hidden_states=True)
h.remove()
n_slots = len(o0.hidden_states)
d_mid = (o1.hidden_states[L_MID][0, -1] - o0.hidden_states[L_MID][0, -1]).abs().max().item()
d_next = (o1.hidden_states[L_MID + 1][0, -1] - o0.hidden_states[L_MID + 1][0, -1]).abs().max().item()
d_pre = (o1.hidden_states[L_MID - 1][0, -1] - o0.hidden_states[L_MID - 1][0, -1]).abs().max().item()
log('slots=%d (expect %d) d[L_MID-1]=%.3f d[L_MID]=%.3f d[L_MID+1]=%.3f' % (
    n_slots, NL + 1, d_pre, d_mid, d_next))
ok = (n_slots == NL + 1) and d_mid > 10 and d_pre == 0
MON_STOP['v'] = True
log('DIAG5 %s' % ('PASS' if ok else 'FAIL'))
flush()
