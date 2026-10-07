# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3049_omega_p46_kvload_localization_'
     r'qwen.py')
R = (r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
     r'\patch3049e_result.txt')
s = io.open(P, encoding='utf-8').read()

# band_rep assignment side: rot_apply returns
# (NL,8,128); repK[:, j, :] is (NL,1024) so the
# reshape must be (NL, HDIM*8), not (HDIM*8,)
old1 = ("        repK[:, j, :] = rot_apply(src, off) \\\n"
        "            .reshape(HDIM * 8)")
new1 = ("        repK[:, j, :] = rot_apply(src, off) \\\n"
        "            .reshape(NL, HDIM * 8)")
assert s.count(old1) == 1, ('a1', s.count(old1))
s = s.replace(old1, new1)

# T3 layer-band block: reshape to (NL, HDIM*8)
# then slice the layer band
old2 = ("            repK[l0:l1, j, :] = rot_apply(\n"
        "                src, off).reshape(HDIM * 8)")
new2 = ("            repK[l0:l1, j, :] = rot_apply(\n"
        "                src, off).reshape(\n"
        "                NL, HDIM * 8)[l0:l1]")
assert s.count(old2) == 1, ('a2', s.count(old2))
s = s.replace(old2, new2)

# corrections: register the run4 crash
old3 = "T3 layer-band block); run4 authoritative',"
new3 = ("T3 layer-band block); run4 crashed pre-"
        "anchor on the assignment side of the same "
        "slice: rot_apply returns (NL,8,128) and "
        "repK[:, j, :] is (NL,1024), so the reshape "
        "must be (NL,HDIM*8) (T3 block reshapes to "
        "(NL,HDIM*8) then slices [l0:l1]); run5 "
        "authoritative',")
assert s.count(old3) == 1, ('a3', s.count(old3))
s = s.replace(old3, new3)

# run label
old4 = "          'run': 'run4 authoritative (fp32; "
new4 = "          'run': 'run5 authoritative (fp32; "
assert s.count(old4) == 1, ('a4', s.count(old4))
s = s.replace(old4, new4)

io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)
io.open(R, 'w', encoding='utf-8').write('ok\n')
print('ok')
