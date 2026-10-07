# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3049_omega_p46_kvload_localization_'
     r'qwen.py')
R = (r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
     r'\patch3049d_result.txt')
s = io.open(P, encoding='utf-8').read()

# band_rep: source slice KPpost[pref_i, :, off+j, :]
# has shape (NL, 1024) - reshape must be (NL, 8, 128),
# not (8, 128). Two occurrences inside band_rep
# (front band build) plus one in T3 (same slice).
old1 = ("        src = KPpost[pref_i, :, off + j, :] \\\n"
        "            .reshape(8, HDIM)")
new1 = ("        src = KPpost[pref_i, :, off + j, :] \\\n"
        "            .reshape(NL, 8, HDIM)")
assert s.count(old1) == 1, ('a1', s.count(old1))
s = s.replace(old1, new1)
old2 = ("            src = KPpost[pref_i, :, off + j, :] \\\n"
        "                .reshape(8, HDIM)")
new2 = ("            src = KPpost[pref_i, :, off + j, :] \\\n"
        "                .reshape(NL, 8, HDIM)")
assert s.count(old2) == 1, ('a2', s.count(old2))
s = s.replace(old2, new2)

# corrections: register the run3 crash
old3 = "reshape removed, K side unchanged; run3 authoritative',"
new3 = ("reshape removed, K side unchanged; run3 "
        "crashed pre-anchor inside band_rep on the "
        "K source slice reshape: KPpost[pref_i, :, "
        "off+j, :] is (NL,1024) and must reshape to "
        "(NL,8,128), not (8,128) (same fix in the "
        "T3 layer-band block); run4 authoritative',")
assert s.count(old3) == 1, ('a3', s.count(old3))
s = s.replace(old3, new3)

# run label
old4 = "          'run': 'run3 authoritative (fp32; "
new4 = "          'run': 'run4 authoritative (fp32; "
assert s.count(old4) == 1, ('a4', s.count(old4))
s = s.replace(old4, new4)

io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)
io.open(R, 'w', encoding='utf-8').write('ok\n')
print('ok')
