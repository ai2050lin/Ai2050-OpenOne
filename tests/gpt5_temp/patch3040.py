# -*- coding: utf-8 -*-
import io
import os
import py_compile
import traceback

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3040_omega_p37_situational_component_qwen.py')
OUTP = (r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3040_result.txt')
try:
    src = io.open(P, encoding='utf-8').read()
    eol = '\r\n' if '\r\n' in src else '\n'

    old1 = ('    V20o[oi] = KVs[3 if False else 20]'
            '[pi][1][pos]')
    new1 = '    V20o[oi] = KVs[20][pi][1][pos]'

    old2 = eol.join([
        "obs_d1 = float(np.median(cos3_arr[sd_mask])",
        "               - np.median(cos3_arr[cd_mask])) \\",
        "    if sd_mask.any() else float('nan')",
        "obs_d1 = float(np.median(cos3_arr[sd_mask])) \\",
        "    - float(np.median(cos3_arr[cd_mask]))"])
    new2 = eol.join([
        "obs_d1 = float(np.median(cos3_arr[sd_mask])) \\",
        "    - float(np.median(cos3_arr[cd_mask]))"])

    c1 = src.count(old1)
    c2 = src.count(old2)
    assert c1 == 1, 'old1 count=%d' % c1
    assert c2 == 1, 'old2 count=%d' % c2
    src = src.replace(old1, new1)
    src = src.replace(old2, new2)
    io.open(P, 'w', encoding='utf-8', newline='').write(
        src)

    src2 = io.open(P, encoding='utf-8').read()
    assert old1 not in src2, 'old1 still present'
    assert old2 not in src2, 'old2 still present'
    assert src2.count(new1) == 1
    py_compile.compile(P, doraise=True)
    msg = 'PATCH_OK'
except Exception:
    msg = 'PATCH_FAIL\n' + traceback.format_exc()
with io.open(OUTP, 'w', encoding='utf-8') as f:
    f.write(msg)
print(msg)
