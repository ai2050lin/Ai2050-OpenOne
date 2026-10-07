# -*- coding: utf-8 -*-
"""Phase 3014 run2 crash fix: med_rj built with float
keys but read with str keys. Unify to str keys."""
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3014_omega_p2h_reverse_dose_law_qwen.py')
t = io.open(P, encoding='utf-8').read()
miss = []


def rep(old, new):
    global t
    if old in t:
        t = t.replace(old, new, 1)
    else:
        miss.append(old[:60])


rep("""            med_rj = {s: float(np.median(
                retain['JOINT'][str(s)]))
                for s in SCALE_GRID[1:]}""",
    """            med_rj = {str(s): float(np.median(
                retain['JOINT'][str(s)]))
                for s in SCALE_GRID[1:]}""")

rep("""            'run2: '
            'authoritative',""",
    """            'run2: crashed at med_retain_05 - '
            'med_rj was built with float keys but '
            'read with str keys (KeyError 0.5); '
            'unified to str keys; no design '
            'change; run3: authoritative',""")

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
chk = {
    'str_key_build':
        "med_rj = {str(s): float(np.median(" in t2,
    'run3_note': 'run3: authoritative' in t2,
}
out = 'patched ok=%s miss=%s' % (
    all(chk.values()),
    {k: v for k, v in chk.items() if not v} or miss)
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_p14c.txt', 'w',
        encoding='utf-8').write(out + '\n' + str(chk))
print(out)
