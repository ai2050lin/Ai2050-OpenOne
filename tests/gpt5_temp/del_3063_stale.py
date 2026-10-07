# delete stale run1 artifacts (3063)
import os

D = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3063'
     r'\omega_p60_adversarial_trace_qwen')
for fn in ('execution.json', 'run_log.txt'):
    p = os.path.join(D, fn)
    if os.path.exists(p):
        os.remove(p)
        print('removed', fn)
    else:
        print('absent', fn)
rest = os.listdir(D)
print('remaining:', rest)
assert not any(f in rest for f in
               ('execution.json', 'run_log.txt'))
print('CLEAN_OK')
