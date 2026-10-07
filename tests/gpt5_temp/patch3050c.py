# -*- coding: utf-8 -*-
"""Patch 3050c: register the run1 correction in
PREREG and mark the next run as run2."""
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3050_omega_p47_kvdeep_dissection_'
     r'qwen.py')
s = io.open(P, encoding='utf-8').read()

old1 = """                             'one branch',
}"""
new1 = """                             'one branch',
    'corrections': 'run1 crashed pre-anchor at '
                   'the T5 lens capture: LENSLOG '
                   'was declared (n_pr, NL) while '
                   'forward_lens returned (NL, n, '
                   'NVOC), and full per-layer lens '
                   'storage is infeasible anyway '
                   '(~1.4TB for 32 prompts); '
                   'forward_lens now keeps last-'
                   'position logits only (NL, '
                   'NVOC), a143 compares per '
                   'prompt and discards, and the '
                   'COS_LENS loop re-captures '
                   'base/pref per pair on the fly; '
                   'run2 authoritative',
}"""
assert s.count(old1) == 1, s.count(old1)
s = s.replace(old1, new1)

old2 = """          'run': 'run1 (fp32; capture bank loaded '
                 'from the phase3048 npz; per-pair '
                 'chain anchors vs the phase3049 '
                 'npz)',"""
new2 = """          'run': 'run2 authoritative (fp32; run1 '
                 'crashed pre-anchor at the T5 '
                 'lens capture, see corrections; '
                 'capture bank loaded from the '
                 'phase3048 npz; per-pair chain '
                 'anchors vs the phase3049 npz)',"""
assert s.count(old2) == 1, s.count(old2)
s = s.replace(old2, new2)

with io.open(P, 'w', encoding='utf-8') as f:
    f.write(s)
print('patched ok')
