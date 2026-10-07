# -*- coding: utf-8 -*-
"""Patch 3050d: dim guard in forward_lens (the
decoder-layer hook stores out[0]; the layer returns
a bare tensor so orig is already (n,2560) without
the batch dim) + register the run2 correction,
mark run3."""
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3050_omega_p47_kvdeep_dissection_'
     r'qwen.py')
s = io.open(P, encoding='utf-8').read()

old1 = """        for li in range(NL):
            hH = capH[li]['orig'][0]
            ln = model.model.norm(hH)"""
new1 = """        for li in range(NL):
            hH = capH[li]['orig']
            if hH.dim() == 3:
                hH = hH[0]
            ln = model.model.norm(hH)"""
assert s.count(old1) == 1, s.count(old1)
s = s.replace(old1, new1)

old2 = """                   'base/pref per pair on the fly; '
                   'run2 authoritative',"""
new2 = """                   'base/pref per pair on the fly; '
                   'run2 crashed pre-anchor inside '
                   'forward_lens: the Qwen3 decoder '
                   'layer returns a bare tensor so '
                   'the hook out[0] already stripped '
                   'the batch dim and the extra [0] '
                   'left a 1-D hidden (lg 1-D -> '
                   'IndexError); dim guard added '
                   '(hH.dim()==3 -> hH[0]); '
                   'run3 authoritative',"""
assert s.count(old2) == 1, s.count(old2)
s = s.replace(old2, new2)

old3 = """          'run': 'run2 authoritative (fp32; run1 '
                 'crashed pre-anchor at the T5 '
                 'lens capture, see corrections; '"""
new3 = """          'run': 'run3 authoritative (fp32; run1-2 '
                 'crashed pre-anchor at the T5 '
                 'lens capture, see corrections; '"""
assert s.count(old3) == 1, s.count(old3)
s = s.replace(old3, new3)

with io.open(P, 'w', encoding='utf-8') as f:
    f.write(s)
print('patched ok')
