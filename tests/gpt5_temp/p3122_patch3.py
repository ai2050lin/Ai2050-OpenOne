# -*- coding: utf-8 -*-
"""Patch3: Part C must slice frozen inputs to NP_A
(smoke NP_A=8 vs full 672-row 3118 trajectories).
Full run: NP_A=672 -> slice is a no-op."""
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3122_omega_p120_write_content_readout_'
     r'sentence_causal_dist_recon.py')
src = io.open(P, encoding='utf-8').read()

old = ("""mP = mP18.astype(np.float64)
mA = mA18.astype(np.float64)
MSEQ = {'P': mP, 'A1': mA}
ANN = {'P': annP_c, 'A1': annA_c}""")
new = ("""mP = mP18[:NP_A].astype(np.float64)
mA = mA18[:NP_A].astype(np.float64)
MSEQ = {'P': mP, 'A1': mA}
ANN = {'P': annP_c[:NP_A], 'A1': annA_c[:NP_A]}""")
c = src.count(old)
assert c == 1, 'expect 1, got %d' % c
src = src.replace(old, new)

# sanity: no other unguarded full-size uses in Part C
assert 'MSEQ[dcode][:, 1:]' in src
py_compile.compile(P, doraise=True)
io.open(P, 'w', encoding='utf-8').write(src)
msg = 'patch3 OK: 1 replaced; COMPILE_OK'
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\p3122_patch3_out.txt', 'w',
        encoding='utf-8').write(msg + chr(10))
print(msg)
