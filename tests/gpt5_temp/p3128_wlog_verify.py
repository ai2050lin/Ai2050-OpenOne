# -*- coding: utf-8 -*-
"""Append 3128 verify record to both wlogs
(idempotent), output to file."""
import io

WLOGS = [
    (r'D:\AI2050\Ai2050-OpenOne'
     r'\.workbuddy\memory'
     r'\2026-09-25.md'),
    (r'C:\Users\Admin\WorkBuddy'
     r'\2026-09-17-01-30-05'
     r'\.workbuddy\memory\2026-09-25.md')]
LINE = ('- Phase 3128 Omega-P126 disk verify: '
        '19 partitions all green (TOTAL FAILS '
        '0); result sha8 ee779d8b; ledger sha8 '
        '1eed79d9; A0 gates 1e-12, joint npz '
        'medians, rho_cap recompute 3e-8, '
        '48 rho_prof 0.0, flip labels '
        'bit-exact, ledger sha recomp match.')

out = []
for wl in WLOGS:
    try:
        prev = io.open(wl,
                       encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3128 Omega-P126 disk verify' \
            not in prev:
        with io.open(wl, 'a',
                     encoding='utf-8') as f:
            f.write(LINE + '\n')
        out.append('appended: ' + wl)
    else:
        out.append('exists: ' + wl)

with io.open(
        r'D:\AI2050\Ai2050-OpenOne\tests'
        r'\gpt5_temp\p3128_wlog_verify_out.txt',
        'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('WLOG_VERIFY_OK')
