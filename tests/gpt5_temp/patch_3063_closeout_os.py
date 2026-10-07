# patch: ensure import os in closeout script
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
     r'\phase3063_closeout.py')
s = io.open(P, encoding='utf-8').read()
if 'import os' in s:
    print('ALREADY_PRESENT')
else:
    old = 'import hashlib\nimport io\nimport json\n'
    cnt = s.count(old)
    assert cnt == 1, 'count=%d' % cnt
    s = s.replace(
        old,
        'import hashlib\nimport io\nimport json\n'
        'import os\n')
    io.open(P, 'w', encoding='utf-8').write(s)
    print('PATCH_OK')
