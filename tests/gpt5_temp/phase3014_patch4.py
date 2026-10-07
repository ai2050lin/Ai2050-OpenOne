# -*- coding: utf-8 -*-
"""Phase 3014 run3 fatal fix: case bug in
kv_scale_arm - arm tuple had lowercase 'joint' but
callers pass 'JOINT', so the JOINT arm silently did
nothing (JS all 0, retain all NaN, verdict was a
degenerate artifact). Fix case + full correction
note; run3 void."""
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


rep("""    def kv_scale_arm(past, p, s, arm, li=L3_GATED):
        L = past.layers[li]
        if arm in ('joint', 'KONLY'):
            L.keys[:, :, p, :] *= s
        if arm in ('joint', 'VONLY'):
            L.values[:, :, p, :] *= s""",
    """    def kv_scale_arm(past, p, s, arm, li=L3_GATED):
        L = past.layers[li]
        if arm in ('JOINT', 'KONLY'):
            L.keys[:, :, p, :] *= s
        if arm in ('JOINT', 'VONLY'):
            L.values[:, :, p, :] *= s""")

rep("""        'correction_note':
            'run1: crashed at the retain computation '
            '- js[arm] is a dict keyed by scale and '
            'np.array(dict) is 0-dimensional (same '
            'bug in the npz save block); fix uses '
            'the s=0.0 list as denominator and '
            'stacks scales for npz; no design '
            'change, crash was before any retain '
            'was computed; no verdict-bearing '
            'deviation from PREREG; run2: crashed at med_retain_05 - '
            'med_rj was built with float keys but '
            'read with str keys (KeyError 0.5); '
            'unified to str keys; no design '
            'change; run3: authoritative',""",
    """        'correction_note':
            'run1: crashed at the retain computation '
            '- js[arm] is a dict keyed by scale and '
            'np.array(dict) is 0-dimensional (same '
            'bug in the npz save block); fix uses '
            'the s=0.0 list as denominator and '
            'stacks scales for npz; no design '
            'change, crash was before any retain '
            'was computed; run2: crashed at '
            'med_retain_05 - med_rj was built with '
            'float keys but read with str keys '
            '(KeyError 0.5); unified to str keys; '
            'no design change; run3: ran to '
            'completion but the JOINT arm was '
            'silently INERT - kv_scale_arm tested '
            "lowercase 'joint' while callers pass "
            "'JOINT', so joint K,V scaling never "
            'applied: med_js_joint0 = 0.0000, all '
            'retains NaN, sham JS exactly 0, and '
            'the NaN fell into the else branch '
            'producing a DEGENERATE verdict '
            'gate_destruction_graded_qwen - run3 '
            'verdict VOID by protocol (degenerate '
            'statistic); arm case fixed; run4: '
            'authoritative',""")

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
chk = {
    'case_fix':
        "if arm in ('JOINT', 'KONLY'):" in t2
        and "if arm in ('JOINT', 'VONLY'):" in t2,
    'no_lowercase_left':
        "in ('joint'" not in t2,
    'note_run4': 'run4: ' in t2
        and 'authoritative' in t2,
    'void_registered': 'VOID by protocol' in t2,
}
out = 'patched ok=%s miss=%s' % (
    all(chk.values()),
    {k: v for k, v in chk.items() if not v} or miss)
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_p14d.txt', 'w',
        encoding='utf-8').write(out + '\n' + str(chk))
print(out)
