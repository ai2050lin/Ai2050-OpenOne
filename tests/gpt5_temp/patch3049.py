# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3049_omega_p46_kvload_localization_'
     r'qwen.py')
R = (r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
     r'\patch3049_result.txt')
s = io.open(P, encoding='utf-8').read()

old = "                'seed %d + mc, stat = 24-pair median '\n                'cos, p = P(null >= obs)',\n                % (R_NULL, SEED_NULL),"
new = "                'seed %d + mc, stat = 24-pair median '\n                'cos, p = P(null >= obs)'\n                % (R_NULL, SEED_NULL),"
assert s.count(old) == 1, ('anchor', s.count(old))
s = s.replace(old, new)

# dead-code / placeholder scan
assert 'if False' not in s, 'dead-code if False'
assert 'TODO' not in s and 'XXX' not in s, 'placeholder'
io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)
io.open(R, 'w', encoding='utf-8').write('ok\n')
print('ok')
