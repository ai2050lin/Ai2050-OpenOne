# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3049_omega_p46_kvload_localization_'
     r'qwen.py')
R = (r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
     r'\patch3049c_result.txt')
s = io.open(P, encoding='utf-8').read()

# V replacement: v_proj output is (1,s,1024);
# the 3048 protocol passes replV flat (no
# reshape). Remove the spurious reshape that
# only the K side (k_norm output (1,s,8,128))
# needs.
old = ("        rv = torch.tensor(np.ascontiguousarray(\n"
       "            replV.reshape(NL, -1, 8, HDIM)),\n"
       "            dtype=torch.float32, device='cuda')")
new = ("        rv = torch.tensor(np.ascontiguousarray(\n"
       "            replV),\n"
       "            dtype=torch.float32, device='cuda')")
assert s.count(old) == 1, ('a1', s.count(old))
s = s.replace(old, new)

# corrections: register the run2 crash
old3 = "rows; run2 authoritative',"
new3 = ("rows; run2 crashed at the first V-replacement "
        "forward (a135 sham): the copied forward_run "
        "applied the K-side reshape (NL,-1,8,128) to "
        "replV as well, but the v_proj output is "
        "(1,s,1024) and the 3048 protocol passes "
        "replV flat - reshape removed, K side "
        "unchanged; run3 authoritative',")
assert s.count(old3) == 1, ('a3', s.count(old3))
s = s.replace(old3, new3)

# run label
old4 = "          'run': 'run2 authoritative (fp32; "
new4 = "          'run': 'run3 authoritative (fp32; "
assert s.count(old4) == 1, ('a4', s.count(old4))
s = s.replace(old4, new4)

io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)
io.open(R, 'w', encoding='utf-8').write('ok\n')
print('ok')
