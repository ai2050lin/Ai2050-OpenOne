# -*- coding: utf-8 -*-
"""Phase 3014 pre-run semantic fix patch.

Unify retain = JS(s)/JS(0) (destruction-retained
share) everywhere; rename verdicts fragile/robust/
graded; fix PREREG text and code to match.
"""
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


# 1) module docstring verdict block
rep("""Verdict (frozen):
  anchor fail                              => anchor_fail_
                                              all_void
  med retain_joint(0.5) >= 0.75            => gate_
                                              destruction_
                                              resistant_qwen
  med retain_joint(0.5) <= 0.25            => gate_
                                              destruction_
                                              steep_qwen
  else                                     => gate_
                                              destruction_
                                              graded_qwen""",
    """Verdict (frozen; retain = JS(s)/JS(0) is the
DESTRUCTION-RETAINED share - high retain at s=0.5
means half-erasure already loses most of the gate,
i.e. the gate is fragile to partial erasure):
  anchor fail                              => anchor_fail_
                                              all_void
  med retain_joint(0.5) >= 0.75            => gate_
                                              destruction_
                                              fragile_qwen
  med retain_joint(0.5) <= 0.25            => gate_
                                              destruction_
                                              robust_qwen
  else                                     => gate_
                                              destruction_
                                              graded_qwen""")

# 2) PREREG T2a definition
rep("""retain(arm,s) = JS(arm,s)/JS(arm,0) per '
           'position (JS(0)>0 gate); PRIMARY stat = med '
           'retain_joint(0.5) over logic positions; '
           'gates nL>=8""",
    """retain(arm,s) = JS(arm,s)/JS(arm,0) per '
           'position = DESTRUCTION-RETAINED share '
           '(JS(0)>0 gate; high retain = half-erasure '
           'already destroys most of the gate = '
           'fragile); PRIMARY stat = med '
           'retain_joint(0.5) over logic positions; '
           'gates nL>=8""")

# 3) PREREG verdict text
rep("""'verdict': 'anchor fail => anchor_fail_all_void; med '
               'retain_joint(0.5) >= 0.75 => '
               'gate_destruction_resistant_qwen; med '
               'retain_joint(0.5) <= 0.25 => '
               'gate_destruction_steep_qwen; else => '
               'gate_destruction_graded_qwen',""",
    """'verdict': 'anchor fail => anchor_fail_all_void; med '
               'retain_joint(0.5) >= 0.75 => '
               'gate_destruction_fragile_qwen; med '
               'retain_joint(0.5) <= 0.25 => '
               'gate_destruction_robust_qwen; else => '
               'gate_destruction_graded_qwen',""")

# 4) T2b docstring line
rep("""      against (1-s) [linear information+capacity] and
      (1-s)**2 [quadratic, attention-logits mixing];""",
    """      against (1-s) [linear information+capacity] and
      (1-s)**2 [quadratic, attention-logits mixing]
      (retain(s) decreasing in s);""")

# 5) code: retain computation
rep("""                for s in SCALE_GRID[1:]:
                    r = 1.0 - np.array(
                        js[arm][s]) / jA[0]
                    retain[arm][str(s)] = r""",
    """                for s in SCALE_GRID[1:]:
                    r = np.array(js[arm][s]) / jA[0]
                    retain[arm][str(s)] = r""")

# 6) code: verdict branch names
rep("""            elif med_retain_05 >= RETAIN_RESIST:
                verdict = 'gate_destruction_' \\
                          'resistant_qwen'
            elif med_retain_05 <= RETAIN_STEEP:
                verdict = 'gate_destruction_' \\
                          'steep_qwen'""",
    """            elif med_retain_05 >= RETAIN_RESIST:
                verdict = 'gate_destruction_' \\
                          'fragile_qwen'
            elif med_retain_05 <= RETAIN_STEEP:
                verdict = 'gate_destruction_' \\
                          'robust_qwen'""")

# 7) shape-fit comment: med_curve now decreasing
rep("""                x = 1.0 - grid
                for name, pred in (
                        ('linear', x),
                        ('quadratic', x ** 2)):""",
    """                x = 1.0 - grid  # retain(s)
                # decreasing in s; linear pred = (1-s),
                # quadratic pred = (1-s)^2 captures
                # attention-mixing steepness
                for name, pred in (
                        ('linear', x),
                        ('quadratic', x ** 2)):""")

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
chk = {
    'fragile_in_prereg':
        'gate_destruction_fragile_qwen' in t2,
    'robust_in_prereg':
        'gate_destruction_robust_qwen' in t2,
    'no_resistant_left':
        'resistant_qwen' not in t2,
    'no_steep_verdict_left':
        "steep_qwen'" not in t2,
    'retain_ratio_code':
        'r = np.array(js[arm][s]) / jA[0]' in t2,
    'no_inverted_code':
        'r = 1.0 - np.array(' not in t2,
    'const_names_kept':
        'RETAIN_STEEP = 0.25' in t2
        and 'RETAIN_RESIST = 0.75' in t2,
}
out = 'patched ok=%s miss=%s' % (
    all(chk.values()),
    {k: v for k, v in chk.items() if not v} or miss)
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_p14.txt', 'w',
        encoding='utf-8').write(out + '\n' + str(chk))
print(out)
