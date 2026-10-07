# -*- coding: utf-8 -*-
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3002_omega_g2_robustness_source_qwen.py')
t = io.open(P, encoding='utf-8').read()

a1 = """    xdir_t = torch.tensor(xdir, device='cuda',
                          dtype=torch.bfloat16)
    log('inj vec armed n=%d' % xdir.shape[0], lines)"""
b1 = """    xdir_t = torch.tensor(xdir, device='cuda',
                          dtype=torch.bfloat16)
    inj['vec'] = xdir_t
    log('inj vec armed n=%d' % xdir.shape[0], lines)"""
assert a1 in t, 'a1 not found'
t = t.replace(a1, b1, 1)

a2 = """    def arm(vec_t, layer, k, tag, spreads):
        projs = []
        ratios = []
        coef = {layer: 1.0}
        for _ in range(k):"""
b2 = """    def arm(vec_t, layer, k, tag, spreads):
        projs = []
        ratios = []
        coef = {layer: 1.0}
        inj['vec'] = vec_t
        for _ in range(k):"""
assert a2 in t, 'a2 not found'
t = t.replace(a2, b2, 1)

a3 = """            ratios.append(float(np.median(
                np.linalg.norm(cs, axis=1))))
        P = np.stack(projs)"""
b3 = """            ratios.append(float(np.median(
                np.linalg.norm(cs, axis=1))))
        inj['vec'] = xdir_t
        P = np.stack(projs)"""
assert a3 in t, 'a3 not found'
t = t.replace(a3, b3, 1)

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
assert "inj['vec'] = xdir_t" in t2
assert t2.count("inj['vec'] = vec_t") == 1
assert "inj['vec'] = xdir_t\n    log" in t2
print('patched ok')
