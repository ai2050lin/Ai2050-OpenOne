# -*- coding: utf-8 -*-
import json, io, os, sys
import numpy as np
ROOT = r'D:\AI2050\Ai2050-OpenOne'
B = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
mdir = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
sys.path.insert(0, os.path.join(ROOT, 'tests', 'glm5'))
os.environ['P3165_SMOKE'] = '1'
import importlib.util
spec = importlib.util.spec_from_file_location(
    'm3165', os.path.join(ROOT, 'tests', 'glm5', 'phase3165_g5a3_family_alignment.py'))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)   # module-level only; main() not called

W4, cfg4 = M.load_WU(M.DIR4 if hasattr(M, 'DIR4') else M.MDIR4)
dW_main, class_words, single_main = M.build_S_class(W4)

# verify-path rebuild
from transformers import AutoTokenizer
e2806 = json.load(io.open(os.path.join(B, 'phase2806', 'qwen4_hierarchy', 'execution.json'), encoding='utf-8'))
CATS = e2806['cats']
tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True, trust_remote_code=True, use_fast=True)
tc = {}
def tid(t):
    if t not in tc:
        ids = tok(' ' + t, add_special_tokens=False)['input_ids']
        if len(ids) != 1:
            ids = tok(t, add_special_tokens=False)['input_ids']
        assert len(ids) == 1, t
        tc[t] = int(ids[0])
    return tc[t]
single_v = []
for w in [x for v in CATS.values() for x in v]:
    try:
        tid(w); single_v.append(w)
    except AssertionError:
        pass
out = []
out.append('single_tok count main=%d verify=%d equal=%s seq_equal=%s' % (
    len(single_main), len(single_v), set(single_main) == set(single_v), single_main == single_v))
for i, cat in enumerate(CATS.keys()):
    wm = [w for w in CATS[cat] if w in single_main]
    wv = [w for w in CATS[cat] if w in single_v]
    if wm != wv:
        out.append('cat %s wordlist differs: main=%s verify=%s' % (cat, wm, wv))
cents_m = []
for cat in CATS.keys():
    ws = [w for w in CATS[cat] if w in single_main]
    cents_m.append(np.stack([W4[tid(w)] for w in ws]).mean(0))
Cm_m = np.stack(cents_m)
dW_c_m = Cm_m - (Cm_m.sum(0, keepdims=True) - Cm_m) / 9.0
dW_v = dW_c_m / np.linalg.norm(dW_c_m, axis=1, keepdims=True)
out.append('dW bitwise equal main vs verify-path: %s' % np.array_equal(dW_main, dW_v.astype(np.float32)))
out.append('max abs diff: %.3e' % float(np.abs(dW_main.astype(np.float64) - dW_v).max()))
open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3165_diag_sc.txt', 'w').write('\n'.join(out))
print('ok')
