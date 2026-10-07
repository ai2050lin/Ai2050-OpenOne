# -*- coding: utf-8 -*-
"""rev-3135a patch: apply 3 fixes that
phantom-edited in the last Edit batch.
Each replacement asserts count==1, then
the file is re-read and re-verified."""
import io
import sys

SRC = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\glm5'
       r'\phase3135_omega_p133_conduction'
       r'_co36ablation_window.py')

REPS = [
    # (old, new)
    ("CKPTF = os.path.join(OUT, "
     "'p132_ckpt.pkl')",
     "CKPTF = os.path.join(OUT, "
     "'p133_ckpt.pkl')"),
    ("- base_states[r]\n"
     "              .astype(np.float64))",
     "- base_states[r][:n_cap]\n"
     "              .astype(np.float64))"),
    ("inj=(17, coords, delta_l17,",
     "inj=(17, coords, DELTA_L17,"),
]

with io.open(SRC, 'r', encoding='utf-8') \
        as fh:
    txt = fh.read()

report = []
for i, (old, new) in enumerate(REPS):
    n = txt.count(old)
    if n != 1:
        report.append(
            'REP %d: ABORT count=%d for '
            '%r' % (i, n, old[:40]))
        continue
    txt = txt.replace(old, new)
    report.append('REP %d: applied '
                  '(count was 1)' % i)

with io.open(SRC, 'w', encoding='utf-8') \
        as fh:
    fh.write(txt)

# re-read verify
with io.open(SRC, 'r', encoding='utf-8') \
        as fh:
    txt2 = fh.read()
report.append('verify: p132_ckpt left=%d '
              'p133_ckpt=%d'
              % (txt2.count('p132_ckpt.pkl'),
                 txt2.count("'p133_ckpt.pkl'")))
report.append('verify: base_states[r] '
              'plain=%d, [:n_cap]=%d'
              % (txt2.count(
                  '- base_states[r]\n'),
                 txt2.count(
                     'base_states[r][:n_cap]')))
report.append('verify: delta_l17 (lower, '
              'non-seal)=%d, DELTA_L17=%d'
              % (txt2.count(
                  'coords, delta_l17,'),
                 txt2.count('DELTA_L17')))

out = (r'D:\AI2050\Ai2050-OpenOne'
       r'\gpt5_temp\p3135_patch_report.txt')
with io.open(out, 'w',
             encoding='utf-8') as fh:
    fh.write('\n'.join(report))
print('WROTE', out)
