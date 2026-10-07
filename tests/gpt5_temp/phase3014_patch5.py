# -*- coding: utf-8 -*-
"""Phase 3014 note completion: the run2/run3 crash
registrations never landed on disk (anchor mismatch
twice). Fix with the actual on-disk anchor; then
rerun as run5 authoritative."""
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


rep("""            'was computed; no verdict-bearing '
            'deviation from PREREG; run2: '
            'authoritative',""",
    """            'was computed; run2: crashed at '
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
            'statistic); arm case fixed; run4: full '
            'pass with valid data but its '
            'correction_note missed the run2/run3 '
            'registrations (anchor mismatch) - '
            'note completed; run5: authoritative',""")

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
chk = {
    'note_complete':
        'run5: authoritative' in t2
        and 'VOID by protocol' in t2
        and 'run2: crashed at ' in t2,
    'case_fix_intact':
        "if arm in ('JOINT', 'KONLY'):" in t2,
}
out = 'patched ok=%s miss=%s' % (
    all(chk.values()),
    {k: v for k, v in chk.items() if not v} or miss)
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_p14e.txt', 'w',
        encoding='utf-8').write(out + '\n' + str(chk))
print(out)
