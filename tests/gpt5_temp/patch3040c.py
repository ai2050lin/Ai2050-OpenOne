# -*- coding: utf-8 -*-
import io
import py_compile
import traceback

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3040_omega_p37_situational_component_qwen.py')
OUTP = (r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3040c_result.txt')
try:
    src = io.open(P, encoding='utf-8').read()
    eol = '\r\n' if '\r\n' in src else '\n'

    old1 = eol.join([
        "dp = np.abs(pos_r_of(R3) - 0) if False else \\",
        "    np.abs(R3['pos_r'][iu3] - R3['pos_r'][ju3])[cdx]"])
    new1 = "dp = np.abs(R3['pos_r'][iu3] - R3['pos_r'][ju3])[cdx]"

    old2 = "            Mu = M / counts0[:, None].astype(\n" \
           "                np.float64)"
    new2 = "            Mu = M / np.maximum(\n" \
           "                counts0, 1)[:, None].astype(\n" \
           "                np.float64)"

    c1 = src.count(old1)
    c2 = src.count(old2)
    assert c1 == 1, 'old1 count=%d' % c1
    assert c2 == 1, 'old2 count=%d' % c2
    src = src.replace(old1, new1)
    src = src.replace(old2, new2)
    io.open(P, 'w', encoding='utf-8', newline='').write(
        src)
    src2 = io.open(P, encoding='utf-8').read()
    assert 'pos_r_of' not in src2
    assert new1 in src2
    assert 'np.maximum(' in src2 and old2 not in src2
    py_compile.compile(P, doraise=True)
    msg = 'PATCH_OK'
except Exception:
    msg = 'PATCH_FAIL\n' + traceback.format_exc()
with io.open(OUTP, 'w', encoding='utf-8') as f:
    f.write(msg)
print(msg)
