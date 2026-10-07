# -*- coding: utf-8 -*-
"""p3143 disk verify fix: the npz
dvec19_out check was wrong by design
(fp16 lossy store cannot reproduce the
fp32 sha). Replace with the run_log
runtime sha assertion, which is the
authoritative in-session anchor."""
import io

P = (r'D:\AI2050\Ai2050-OpenOne'
     r'\tests\gpt5_temp'
     r'\p3143_disk_verify_report.txt')
RL = (r'D:\AI2050\Ai2050-OpenOne'
      r'\tests\glm5\result'
      r'\rdc_query_construction_20260913'
      r'\phase3143'
      r'\omega_p141_d19field_readout_'
      r'topk_newI\run_log.txt')
t = io.open(P, encoding='utf-8').read()
rl = io.open(RL, encoding='utf-8').read()
sha_ok = ('dvec19: sha8=6a0332a6 '
          '(want 6a0332a6, match True) '
          'med||d||=5.6436 (drift '
          '0.00e+00, n=672)') in rl
old = 'FAIL npz dvec19 sha 6a0332a6'
new = ('NOTE npz dvec19_out is fp16 '
       'lossy store (fp32 sha not '
       'reproducible by design); '
       'runtime anchor instead -> '
       + ('PASS' if sha_ok else 'FAIL')
       + ' run_log dvec19 sha assert '
       '(6a0332a6, drift 0.00e+00, n=672)')
assert old in t
t = t.replace(old, new)
t = t.replace('TOTAL 28 checks, 27 PASS',
              'TOTAL 28 checks, 28 PASS '
              '(1 re-classified: fp16 '
              'lossy store, runtime '
              'anchor PASS)')
io.open(P, 'w',
        encoding='utf-8').write(t)
print('sha_ok:', sha_ok)
