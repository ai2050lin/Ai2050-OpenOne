# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3049_omega_p46_kvload_localization_'
     r'qwen.py')
R = (r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
     r'\patch3049f_result.txt')
s = io.open(P, encoding='utf-8').read()

# band_rep returns the FULL-length repl field
# (non-band positions self-replaced from the
# base capture), so the mask must be all-True;
# a partial mask made repl rows (nb) != index
# rows (mask.sum()).
old = ("    mask = np.zeros(nb, dtype=bool)\n"
       "    mask[sel] = True\n"
       "    return repK, repV, mask")
new = ("    # full-length repl + all-True mask:\n"
       "    # non-band rows already hold the base\n"
       "    # self-replacement values\n"
       "    mask = np.ones(nb, dtype=bool)\n"
       "    return repK, repV, mask")
assert s.count(old) == 1, ('a1', s.count(old))
s = s.replace(old, new)

# corrections: register the run5 crash
old3 = ("(NL,HDIM*8) then slices [l0:l1]); run5 "
        "authoritative',")
new3 = ("(NL,HDIM*8) then slices [l0:l1]); run5 "
        "crashed pre-anchor on a K-hook row-count "
        "mismatch: band_rep returns the full-length "
        "repl field (nb rows) paired with a partial "
        "band mask (mask.sum()<nb), but the hook "
        "requires repl rows == mask.sum(); fixed by "
        "an all-True mask (non-band rows already "
        "hold the base self-replacement values; "
        "the FRONT-band null MC reuses the same "
        "front_mask and follows automatically); "
        "run6 authoritative',")
assert s.count(old3) == 1, ('a3', s.count(old3))
s = s.replace(old3, new3)

# run label
old4 = "          'run': 'run5 authoritative (fp32; "
new4 = "          'run': 'run6 authoritative (fp32; "
assert s.count(old4) == 1, ('a4', s.count(old4))
s = s.replace(old4, new4)

io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)
io.open(R, 'w', encoding='utf-8').write('ok\n')
print('ok')
