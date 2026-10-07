# -*- coding: utf-8 -*-
"""Patch 3050e: the real a143 root cause. In
forward_lens the lm_head input is already batch-
stripped (n,2560), so the output is 2-D (n,NVOC)
and lg[0, -1] is a SCALAR (token 0, last vocab
entry), not the last-token vector - lens rows were
constants (hence the blocky med profile and the
~22 a143 diff). probe2's full-matrix comparison
(lm_head(norm(cap35)) vs logits, all positions
diff 0.0) proves the capture+composition is
correct. Fix: take lg[-1] (last token). Also
register the run3 anchor failure, mark run4."""
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3050_omega_p47_kvdeep_dissection_'
     r'qwen.py')
s = io.open(P, encoding='utf-8').read()

old1 = """            lg = model.lm_head(ln)
            lens[li] = lg[0, -1].detach() \\
                .double().cpu().numpy()"""
new1 = """            lg = model.lm_head(ln)
            lens[li] = lg[-1].detach() \\
                .double().cpu().numpy()"""
assert s.count(old1) == 1, s.count(old1)
s = s.replace(old1, new1)

old2 = """                   '(hH.dim()==3 -> hH[0]); '
                   'run3 authoritative',"""
new2 = """                   '(hH.dim()==3 -> hH[0]); '
                   'run3 COMPLETED BUT FAILED the '
                   'a143 anchor (lens final layer '
                   'vs LG diff 22.4, lens profile '
                   'degenerate): lg[0,-1] on the '
                   '2-D lm_head output (n,NVOC) is '
                   'a SCALAR (token 0, last vocab '
                   'entry), not the last-token '
                   'vector - lens rows were '
                   'constant vectors; fixed to '
                   'lg[-1] (a full-matrix probe '
                   'lm_head(norm(cap35)) vs logits '
                   'diff 0.0 at every position '
                   'proves the capture+composition '
                   'is otherwise exact); run4 '
                   'authoritative',"""
assert s.count(old2) == 1, s.count(old2)
s = s.replace(old2, new2)

old3 = """          'run': 'run3 authoritative (fp32; run1-2 '
                 'crashed pre-anchor at the T5 '
                 'lens capture, see corrections; '"""
new3 = """          'run': 'run4 authoritative (fp32; run1-2 '
                 'crashed pre-anchor at the T5 '
                 'lens capture and run3 failed the '
                 'a143 anchor, see corrections; '"""
assert s.count(old3) == 1, s.count(old3)
s = s.replace(old3, new3)

with io.open(P, 'w', encoding='utf-8') as f:
    f.write(s)
print('patched ok')
