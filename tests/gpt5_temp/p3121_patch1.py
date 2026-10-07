# -*- coding: utf-8 -*-
"""p3121_patch1: three self-review fixes to the
3121 main script before first run."""
import compileall
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3121_omega_p119_repl_causality_'
     r'erase_polarity_recon.py')
LOG = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\gpt5_temp\p3121_patch1_log.txt')
o = []
src = io.open(P, encoding='utf-8').read()


def rep(old, new, tag, expect=1):
    global src
    c = src.count(old)
    if c == expect:
        src = src.replace(old, new)
        o.append('%s: applied' % tag)
    else:
        o.append('%s: SKIP count=%d' % (tag, c))


# P1: remove dead conditional-expression hack
rep(
    "for ys in seal_unused() if False else \\\n"
    "        ['yes', 'Yes', ' yes', ' Yes', "
    "'YES']:",
    "for ys in ['yes', 'Yes', ' yes', ' Yes',\n"
    "           'YES']:",
    'P1-seal-unused')

# P2: E_10 must exclude spans with len<2
rep(
    "        n_span += 1\n"
    "        k1, k2 = si['k1'], si['k2']",
    "        n_span += 1\n"
    "        lens.append(si['len'])\n"
    "        k1, k2 = si['k1'], si['k2']",
    'P2a-lens-collect')
rep(
    "    Dm = {c: float(np.mean(Ds[c])) "
    "for c in CONDS}\n"
    "    e10_all = [a - b for (a, b) in\n"
    "               zip(Ds['c1'], Ds['c2'])]",
    "    Dm = {c: float(np.mean(Ds[c])) "
    "for c in CONDS}\n"
    "    e10_all = [a - b\n"
    "               for (a, b, ln) in\n"
    "               zip(Ds['c1'], Ds['c2'], lens)\n"
    "               if ln >= 2]\n"
    "    assert len(e10_all) == n_len2",
    'P2b-e10-filter')

# lens list init: add next to Ds init
rep(
    "    Ds = {c: [] for c in CONDS}\n"
    "    n_span = 0",
    "    Ds = {c: [] for c in CONDS}\n"
    "    lens = []\n"
    "    n_span = 0",
    'P2c-lens-init')

# P3: assert auc18 length
rep(
    "auc18 = z18['auc_curve']\n"
    "assert abs(float(auc18[0])",
    "auc18 = z18['auc_curve']\n"
    "assert len(auc18) == N_NEW + 1\n"
    "assert abs(float(auc18[0])",
    'P3-auc-len')

with io.open(P, 'w', encoding='utf-8') as f:
    f.write(src)
back = io.open(P, encoding='utf-8').read()
o.append('disk check: p1=%s p2b=%s p3=%s'
         % ("for ys in ['yes'" in back,
            "if ln >= 2" in back,
            "len(auc18) == N_NEW + 1" in back))
compileall.compile_file(P, force=True, quiet=2)
o.append('compile done')
io.open(LOG, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
