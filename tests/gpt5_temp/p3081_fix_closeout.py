# -*- coding: utf-8 -*-
"""Fix phase3081_closeout.py: remove the duplicated
first 'claim = (...)' block (L58-133) which calls
np_med before its definition; keep the second block
(after 'def np_med').  Writes confirmation to file
(bash shim: python -c stdout unreliable)."""
import io

SRC = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase3081_closeout.py')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\p3081_fix_closeout.txt')

src = io.open(SRC, encoding='utf-8').read()
lines = src.split('\n')

# locate first 'claim = (' and 'def np_med'
i_first = None
i_def = None
for idx, ln in enumerate(lines):
    if ln.startswith('claim = (') and i_first is None:
        i_first = idx
    if ln.startswith('def np_med'):
        i_def = idx
        break
assert i_first is not None and i_def is not None, \
    (i_first, i_def)
# sanity: between them there must be a
# 'float(np_med(' call (the broken variant)
seg = '\n'.join(lines[i_first:i_def])
assert 'float(np_med(' in seg, 'first block OK?'
assert seg.count('claim = (') == 1, 'ambiguity'

# delete lines [i_first, i_def) -> keeps def np_med
new_lines = lines[:i_first] + lines[i_def:]
new_src = '\n'.join(new_lines)
# exactly one claim block remains
assert new_src.count('claim = (') == 1
assert new_src.index('claim = (') > \
    new_src.index('def np_med')
io.open(SRC, 'w', encoding='utf-8').write(new_src)

# verify on disk
chk = io.open(SRC, encoding='utf-8').read()
n_claim = chk.count('claim = (')
pos_claim = chk.index('claim = (')
pos_def = chk.index('def np_med')
io.open(OUT, 'w', encoding='utf-8').write(
    'removed lines %d..%d (0-based)\n'
    'on-disk: claims=%d, claim_pos=%d, '
    'def_np_med_pos=%d, claim_after_def=%s\n'
    'total_lines=%d\n'
    % (i_first, i_def, n_claim, pos_claim, pos_def,
       str(pos_claim > pos_def),
       len(chk.split('\n'))))
