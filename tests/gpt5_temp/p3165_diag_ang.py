# -*- coding: utf-8 -*-
import json, io, os, sys
import numpy as np
ROOT = r'D:\AI2050\Ai2050-OpenOne'
B = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(B, 'phase3165', 'g5a3_family_alignment')
sys.path.insert(0, os.path.join(ROOT, 'tests', 'glm5'))
os.environ['P3165_SMOKE'] = '1'
import importlib.util
spec = importlib.util.spec_from_file_location(
    'm3165', os.path.join(ROOT, 'tests', 'glm5', 'phase3165_g5a3_family_alignment.py'))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

W4, cfg4 = M.load_WU(M.MDIR4)
dW_main, _, _ = M.build_S_class(W4)
S_class = dW_main.astype(np.float64)

z58 = np.load(os.path.join(B, r'phase3158\g4p1_output_equivalence_class\qwen3-4b\collect.npz'))
top64 = z58['top64'].astype(np.float64)
z57 = np.load(os.path.join(B, r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz'))
H = z57['H'].astype(np.float64)
NL = H.shape[1] - 1
X = H[:, NL, :]
Xc = X - X.mean(0, keepdims=True)
_, _, Vh = np.linalg.svd(Xc, full_matrices=False)
K_ent = Vh[:8]

s1, t1 = M.kross(top64.T, S_class)
s2, t2 = M.kross(K_ent, S_class)
fp = lambda a: __import__('hashlib').md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:10]
out = ['FP S_class=%s top64T=%s K_ent=%s' % (fp(S_class), fp(top64.T), fp(K_ent)),
       'main-path recomputed K_entity__S_class = %.3f (res says 73.405)' % t2,
       'top64 dtype=%s shape=%s C-contig=%s' % (top64.dtype, top64.shape, top64.flags['C_CONTIGUOUS']),
       'H dtype=%s shape=%s' % (H.dtype, H.shape)]
# also try float32 round-trip of S_class (as in main: astype32 -> astype64)
S32 = S_class.astype(np.float32).astype(np.float64)
_, t3 = M.kross(top64.T, S32)
out.append('with float32 roundtrip S_class: %.3f' % t3)
# and top64 float32 roundtrip
_, t4 = M.kross(top64.astype(np.float32).astype(np.float64).T, S_class)
out.append('with float32 roundtrip top64: %.3f' % t4)
open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3165_diag_ang.txt', 'w').write('\n'.join(out))
print('ok')
