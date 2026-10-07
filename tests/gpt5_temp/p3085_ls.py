import os
d = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913'
     r'\phase3085'
     r'\omega_p82_l34_full_arbitration')
o = []
for f in sorted(os.listdir(d)):
    o.append('%s %d' % (f, os.path.getsize(
        os.path.join(d, f))))
with open(r'D:\AI2050\Ai2050-OpenOne\tests'
          r'\gpt5_temp\p3085_ls.txt', 'w') as fh:
    fh.write('\n'.join(o) + '\n')
print('LS_DONE')
