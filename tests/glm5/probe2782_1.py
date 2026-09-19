import numpy as np
BASE = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
z61 = np.load(BASE + r'\phase2761\qwen4_kc_fault\fault_scores.npz', allow_pickle=False)
cw = np.array(sorted(int(i) for i in z61['wrong_idx']))
zbd = np.load(BASE + r'\phase2763\qwen4_debias_repair\bias_dirs.npz', allow_pickle=False)
print('keys', list(zbd.keys()))
vc = zbd['v_rows'][cw]
print('shape', vc.shape, 'row norms[:5]', np.round(np.linalg.norm(vc, axis=1)[:5], 6).tolist())
zp = np.load(BASE + r'\phase2774\qwen4_pull_validation\pull_stats.npz', allow_pickle=False)
print('pull shape', zp['pull'].shape, 'wrong align', bool((zp['wrong_idx'] == cw).all()))
print('v_norms range', float(zbd['v_norms'][cw].min()), float(zbd['v_norms'][cw].max()))
# verify pull reproduces: need W_U; instead cross-check pull sign distribution
print('pull<0 count', int((zp['pull'] < 0).sum()), '/', len(cw))
