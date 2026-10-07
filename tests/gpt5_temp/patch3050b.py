# -*- coding: utf-8 -*-
"""Patch 3050b: fix the T5 lens block. forward_lens
keeps last-position logits only (NL, NVOC); a143
compares per prompt and discards; COS_LENS loop
re-captures base/pref per pair on the fly. No full
lens tensor is ever stored."""
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3050_omega_p47_kvdeep_dissection_'
     r'qwen.py')
s = io.open(P, encoding='utf-8').read()

old1 = """    n = len(ids)
    lens = np.zeros((NL, n, NVOC), dtype=np.float64)
    with torch.no_grad():
        for li in range(NL):
            hH = capH[li]['orig'][0]
            ln = model.model.norm(hH)
            lg = model.lm_head(ln)
            lens[li] = lg.detach().double() \\
                .cpu().numpy()
    reset_all()
    return lens"""
new1 = """    lens = np.zeros((NL, NVOC), dtype=np.float64)
    with torch.no_grad():
        for li in range(NL):
            hH = capH[li]['orig'][0]
            ln = model.model.norm(hH)
            lg = model.lm_head(ln)
            lens[li] = lg[0, -1].detach() \\
                .double().cpu().numpy()
    reset_all()
    return lens"""
assert s.count(old1) == 1, s.count(old1)
s = s.replace(old1, new1)

old2 = """a143_diff = 0.0
for i in range(n_pr):
    ll = forward_lens(assembled[i]['ids'])
    LENSLOG[i] = ll
    a143_diff = max(a143_diff, float(np.max(
        np.abs(ll[NL - 1, -1, :]
               - LG[i].astype(np.float64)))))"""
new2 = """a143_diff = 0.0
for i in range(n_pr):
    ll = forward_lens(assembled[i]['ids'])
    a143_diff = max(a143_diff, float(np.max(
        np.abs(ll[NL - 1, :]
               - LG[i].astype(np.float64)))))"""
assert s.count(old2) == 1, s.count(old2)
s = s.replace(old2, new2)

old3 = """log('=== T5 lens capture (32 prompts) ===')
LENSLOG = np.zeros((n_pr, NL), dtype=object)
"""
new3 = """log('=== T5 lens capture (32 prompts) ===')
"""
assert s.count(old3) == 1, s.count(old3)
s = s.replace(old3, new3)

old4 = """    t = t_targets[(b, c)].astype(np.float64)
    nt = float(np.linalg.norm(t))
    for li in range(NL):
        d = LENSLOG[pref_i][li, -1, :] \\
            - LENSLOG[base_i][li, -1, :]
        nr = float(np.linalg.norm(d))"""
new4 = """    t = t_targets[(b, c)].astype(np.float64)
    nt = float(np.linalg.norm(t))
    lb = forward_lens(assembled[base_i]['ids'])
    lp = forward_lens(assembled[pref_i]['ids'])
    for li in range(NL):
        d = lp[li] - lb[li]
        nr = float(np.linalg.norm(d))"""
assert s.count(old4) == 1, s.count(old4)
s = s.replace(old4, new4)

assert 'LENSLOG' not in s, 'LENSLOG remnant'
with io.open(P, 'w', encoding='utf-8') as f:
    f.write(s)
print('patched ok')
