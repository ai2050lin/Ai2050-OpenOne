# -*- coding: utf-8 -*-
P2 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
      r'\rdc_query_construction_20260913\phase2953'
      r'\a11_s_response.npz')
OUTTXT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2953_diag2.txt')
with open(OUTTXT, 'w', encoding='utf-8') as f:
    f.write('repr=%r\n' % P2)
    f.write('bytes=%r\n' % P2.encode('utf-8'))
    f.write('len=%d\n' % len(P2))
print('ok')
