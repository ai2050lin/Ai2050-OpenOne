# -*- coding: utf-8 -*-
"""Phase 3014 run1 crash fix: js[arm] is a dict keyed
by scale, np.array(dict) is 0-dimensional. Use the
s=0.0 list as denominator; stack scales for npz."""
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


rep("""            retain = {}
            for arm, jA in (('JOINT',
                             np.array(js['JOINT'])),
                            ('KONLY',
                             np.array(js['KONLY'])),
                            ('VONLY',
                             np.array(js['VONLY']))):
                retain[arm] = {}
                for s in SCALE_GRID[1:]:
                    r = np.array(js[arm][s]) / jA[0]
                    retain[arm][str(s)] = r""",
    """            retain = {}
            for arm in ('JOINT', 'KONLY', 'VONLY'):
                j0 = np.array(js[arm][0.0])
                retain[arm] = {}
                for s in SCALE_GRID[1:]:
                    r = np.array(js[arm][s]) / j0
                    retain[arm][str(s)] = r""")

rep("""            'js_joint': np.array(js['JOINT'])
            if js.get('JOINT') else np.array([]),
            'js_konly': np.array(js['KONLY'])
            if js.get('KONLY') else np.array([]),
            'js_vonly': np.array(js['VONLY'])
            if js.get('VONLY') else np.array([])}""",
    """            'js_joint': np.array(
                [js['JOINT'][s] for s in SCALE_GRID])
            if js.get('JOINT') else np.array([]),
            'js_konly': np.array(
                [js['KONLY'][s] for s in SCALE_GRID])
            if js.get('KONLY') else np.array([]),
            'js_vonly': np.array(
                [js['VONLY'][s] for s in SCALE_GRID])
            if js.get('VONLY') else np.array([])}""")

rep("""        'correction_note':
            'first run (3013 machine verbatim; T2a '
            'dose grid is a new T2 with no reuse of '
            '3013 refill code; known pitfalls from '
            '3013 pre-fixed: tag parse strips P '
            'prefix, BFloat16 never converted to '
            'numpy, no phantom-edit-prone refactor)',""",
    """        'correction_note':
            'run1: crashed at the retain computation '
            '- js[arm] is a dict keyed by scale and '
            'np.array(dict) is 0-dimensional (same '
            'bug in the npz save block); fix uses '
            'the s=0.0 list as denominator and '
            'stacks scales for npz; no design '
            'change, crash was before any retain '
            'was computed; no verdict-bearing '
            'deviation from PREREG; run2: '
            'authoritative',""")

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
chk = {
    'denom_fix': "j0 = np.array(js[arm][0.0])" in t2,
    'no_dict_array':
        "np.array(js['JOINT'])" not in t2
        and "np.array(js['KONLY'])" not in t2,
    'npz_stack':
        "[js['JOINT'][s] for s in SCALE_GRID]" in t2,
    'note_run2': 'run2: ' in t2
        and 'authoritative' in t2,
}
out = 'patched ok=%s miss=%s' % (
    all(chk.values()),
    {k: v for k, v in chk.items() if not v} or miss)
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_p14b.txt', 'w',
        encoding='utf-8').write(out + '\n' + str(chk))
print(out)
