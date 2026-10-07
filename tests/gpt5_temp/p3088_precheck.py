# -*- coding: utf-8 -*-
"""p3088 static pre-check: compile + bit-identity
of frozen statistics block vs 3086 + bad/want
tokens.  Output -> tests/gpt5_temp/p3088_check_out.txt"""
import io
import os
import py_compile
import re

F86 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase3086_omega_p83_continuum_test.py')
F88 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase3088_omega_p86_continuum_n12.py')
OUTF = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\p3088_check_out.txt')
o = []
ok = True

# 1. compile
try:
    py_compile.compile(F88, doraise=True)
    o.append('py_compile OK')
except Exception as e:
    ok = False
    o.append('py_compile FAIL: %r' % e)

s86 = io.open(F86, encoding='utf-8').read()
s88 = io.open(F88, encoding='utf-8').read()

# 2. bit-identity of frozen stats block
def block(src):
    i = src.index('def spearman(a, b):')
    j = src.index('def sha8(path):')
    return src[i:j]

b86, b88 = block(s86), block(s88)
same = (b86 == b88)
ok = ok and same
o.append('frozen stats block (spearman+perm_p) '
         'bit-identical to 3086: %s (len86=%d '
         'len88=%d)' % (same, len(b86), len(b88)))

# 3. BAD tokens (must be absent)
BAD = ["'phase3086', NAME)",
       "os.path.join(BASE, 'phase3086'",
       'continuum_confirmed',
       'omega_p83_continuum_test',
       'SEED = 3086',
       'N_PERM = 1000 if SMOKE else 20000\nSEED']
for tok in BAD:
    c = s88.count(tok)
    good = (c == 0)
    ok = ok and good
    o.append('BAD %-40r count=%d %s'
             % (tok, c,
                'OK' if good else 'VIOLATION'))

# 4. WANT tokens (exact counts)
WANT = {"'phase3088', NAME": 1,
        'phase3088': 1,
        '32d3bf08': 4,
        'cross_architecture_confirmed': 5,
        'cross_architecture_positive_ns': 2,
        'qwen_lineage_internal': 5,
        'continuum_setup_failed': 3,
        "'GLM4', S87, S87, '', 'MIG'": 1,
        "'GLM4': (0.69, 0.79)": 1,
        "'GLM4': z87": 2,
        "int(z87['FORWARDS']) > 20000": 1,
        "sha8(S87) == '32d3bf08'": 1,
        ".replace('GLM4', 'MGLM4')": 1,
        "'TOP3_GLM4'": 1,
        "'SHA87'": 1,
        "'S87': sha8(S87)": 1,
        "('4B', 'DS7B', '3B', 'GLM4')": 1,
        'z87 = np.load(S87': 1,
        "k.startswith('GLM4')": 1,
        "not k.startswith('GLM4')": 1}
for tok, want in WANT.items():
    c = s88.count(tok)
    if want is None:
        good = c >= 2
    else:
        good = (c == want)
    ok = ok and good
    o.append('WANT %-45r count=%d want=%s %s'
             % (tok, c, want,
                'OK' if good else 'MISMATCH'))

# 5. structural: OUT dir, verdict mapping order
for pat, label in [
        (r"OUT = os\.path\.join\(BASE, 'phase3088', NAME\)",
         'OUT phase3088'),
        (r"if rho_T > 0 and p_T < 0\.05:\s*\n\s*verdict = \\\s*\n\s*'cross_architecture_confirmed'",
         'verdict branch 1'),
        (r"elif rho_T > 0:\s*\n\s*verdict = \\\s*\n\s*'cross_architecture_positive_ns'",
         'verdict branch 2'),
        (r"else:\s*\n\s*verdict = 'qwen_lineage_internal'",
         'verdict branch 3')]:
    m = re.search(pat, s88)
    good = m is not None
    ok = ok and good
    o.append('STRUCT %-18s %s'
             % (label,
                'OK' if good else 'MISSING'))

o.append('PRECHECK_%s' % ('PASS' if ok else 'FAIL'))
io.open(OUTF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
