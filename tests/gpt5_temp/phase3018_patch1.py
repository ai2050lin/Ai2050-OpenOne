# -*- coding: utf-8 -*-
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3018_omega_p2l_dilution_decomposition_'
     r'qwen.py')
t = io.open(P, encoding='utf-8').read()

# 1) remove the dead med_*_l block referencing _rcol
dead = (
    "            # per-layer med profiles\n"
    "            Ls = list(range(L_START, NL - 1))\n"
    "            med_R_l = {}\n"
    "            med_cr_l = {}\n"
    "            med_cA_l = {}\n"
    "            med_cM_l = {}\n"
    "            if nL:\n"
    "                for li in Ls:\n"
    "                    med_R_l[str(li)] = round(float(\n"
    "                        np.median(A['R'])), 6) \\\n"
    "                        if False else round(float(\n"
    "                            np.median(\n"
    "                                _rcol(A, 'R', li))),\n"
    "                            6)\n"
    "            # NOTE: per-layer R/credit need column\n"
    "            # extraction; handled via helper below.\n")
assert dead in t, 'dead block anchor miss'
t = t.replace(dead, '', 1)

# 2) remove the _rcol helper entirely
i0 = t.index('def _rcol(')
i1 = t.index('if __name__')
t = t[:i0].rstrip() + '\n\n\n' + t[i1:]

io.open(P, 'w', encoding='utf-8').write(t)

t2 = io.open(P, encoding='utf-8').read()
checks = {
    'no_rcol': '_rcol' not in t2,
    'no_medRl': 'med_R_l' not in t2,
    'has_T2a': "'cancel_share_med'" in t2,
    'has_a21': 'a21_3017' in t2,
    'has_verdict': 'decomp_dilution_dominant_qwen'
    in t2,
}
io.open(r'C:\Users\Admin\WorkBuddy'
        r'\2026-09-17-01-30-05\.workbuddy\tmp_p18.txt',
        'w', encoding='utf-8').write(
    ' '.join('%s=%s' % kv for kv in checks.items()))
print('ok')
