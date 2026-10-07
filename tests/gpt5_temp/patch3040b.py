# -*- coding: utf-8 -*-
import io
import py_compile
import traceback

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3040_omega_p37_situational_component_qwen.py')
OUTP = (r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3040b_result.txt')
try:
    src = io.open(P, encoding='utf-8').read()
    eol = '\r\n' if '\r\n' in src else '\n'

    old = eol.join([
        "    U, S, Vt = np.linalg.svd(WM,",
        "                             full_matrices=False)",
        "    r = int(np.sum(S > 1e-8 * S[0]))",
        "    B = U[:, :r].copy()"])
    new = eol.join([
        "    U, S, Vt = np.linalg.svd(WM,",
        "                             full_matrices=False)",
        "    r = int(np.sum(S > 1e-8 * S[0]))",
        "    # basis of the ROW space of WM lives in",
        "    # R^HDIM: WM = U S Vt, Vt is (n_types x",
        "    # HDIM) so the orthonormal basis columns",
        "    # are the first r rows of Vt transposed",
        "    B = Vt[:r].T.copy()"])
    c = src.count(old)
    assert c == 1, 'old count=%d' % c
    src = src.replace(old, new)

    anchor = ("                             'assignment "
              "(no sequential '\n"
              "                             "
              "'re-judging)',\n"
              "}")
    corr = ("                             'assignment "
            "(no sequential '\n"
            "                             "
            "'re-judging)',\n"
            "    'corrections': 'run1 crashed at the "
            "subspace build '\n"
            "                   '(make_sub) BEFORE any "
            "verdict '\n"
            "                   'statistic was "
            "observed: the word-mean '\n"
            "                   'SVD basis was taken "
            "from U (n_types x '\n"
            "                   'n_types, token-type "
            "space) instead of Vt '\n"
            "                   '(n_types x head_dim, "
            "residual-stream '\n"
            "                   'row space); corrected "
            "to B = Vt[:r].T; '\n"
            "                   'verdict tree, tests "
            "and anchors '\n"
            "                   'unchanged; run1 had "
            "completed anchors '\n"
            "                   'a73/a74 (0.0), a75 "
            "(0.0453) and a78 '\n"
            "                   '(25/25 bit 0.0) only',"
            "\n"
            "}")
    c2 = src.count(anchor)
    assert c2 == 1, 'anchor count=%d' % c2
    src = src.replace(anchor, corr)

    io.open(P, 'w', encoding='utf-8', newline='').write(
        src)
    src2 = io.open(P, encoding='utf-8').read()
    assert old not in src2
    assert src2.count('B = Vt[:r].T.copy()') == 1
    assert src2.count('B = U[:, :r]') == 0
    assert src2.count("'corrections': 'run1 crashed") == 1
    py_compile.compile(P, doraise=True)
    msg = 'PATCH_OK'
except Exception:
    msg = 'PATCH_FAIL\n' + traceback.format_exc()
with io.open(OUTP, 'w', encoding='utf-8') as f:
    f.write(msg)
print(msg)
