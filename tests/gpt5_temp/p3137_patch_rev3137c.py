# -*- coding: utf-8 -*-
import io
F = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3137_omega_p135_modecoop_'
     r'k0anat_coorddecomp_l35recheck.py')
src = io.open(F, encoding='utf-8').read()
old = "assert len(ORDER) == 36"
new = "assert len(ORDER) == 50"
assert src.count(old) == 1, \
    'patch count=%d' % src.count(old)
src = src.replace(old, new)
old2 = """log('G co36_rank semantics=%s '
    '(top25=%s... bot25=%s...)'
    % (ordtag, top25[:5].tolist(),
       bot25[:5].tolist()))"""
new2 = """log('G co36_rank semantics=%s '
    'len=%d (top25=%s... bot25=%s...)'
    % (ordtag, len(ORDER),
       top25[:5].tolist(),
       bot25[:5].tolist()))"""
assert src.count(old2) == 1, \
    'patch2 count=%d' % src.count(old2)
src = src.replace(old2, new2)
io.open(F, 'w', encoding='utf-8').write(src)
chk = io.open(F, encoding='utf-8').read()
assert chk.count('len(ORDER) == 50') == 1
assert chk.count('len=%d (top25') == 1
print('PATCH OK rev3137c')
