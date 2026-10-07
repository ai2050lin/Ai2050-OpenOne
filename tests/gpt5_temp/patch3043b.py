# -*- coding: utf-8 -*-
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3043_omega_p40_field_variance_qwen.py')
s = io.open(P, encoding='utf-8').read()

# 1) T2 prereg text
old_a = ("'T2': 'content-interaction outlier: obs = "
         "norm of '\n"
         "          'the (topic-prefix, weather-body) "
         "cell; '\n"
         "          'null = prefix-label permutation "
         "(bodies '\n"
         "          'fixed), stat = norm of the cell "
         "carrying '\n"
         "          'label 3 at body 0; N_PERM=%d seed "
         "%d; '")
new_a = ("'T2': 'content-interaction outlier: obs = "
         "norm of '\n"
         "          'the (topic-prefix, weather-body) "
         "cell; '\n"
         "          'null = WITHIN-BLOCK body-label "
         "permuta- '\n"
         "          'tion (the 8 rows of the "
         "topic-prefix '\n"
         "          'block; exactly one row carries body "
         "0 '\n"
         "          'per draw, so the cell always "
         "exists); '\n"
         "          'N_PERM=%d seed %d; '")
assert s.count(old_a) == 1, ('a', s.count(old_a))
s = s.replace(old_a, new_a)

# 2) corrections text
old_b = ("    'corrections': 'run1 crashed PRE-verdict "
         "'\n"
         "                   '(anchor stage only, no '")
new_b = ("    'corrections': 'run2 crashed mid-T2 '\n"
         "                   '(T1 statistics were "
         "observed and '\n"
         "                   'unchanged: f_prefix/f_body "
         "printed; '\n"
         "                   'T3/T4/T5 not yet observed): "
         "the '\n"
         "                   'prefix-label permutation "
         "leaves '\n"
         "                   'the (label-3, body-0) cell "
         "empty '\n"
         "                   'with probability (2/3)^3 "
         "-> IndexError; '\n"
         "                   'T2 null redefined as "
         "within-block '\n"
         "                   'body-label permutation "
         "(exact, cell '\n"
         "                   'always exists); run1 "
         "crashed '\n"
         "                   'PRE-verdict (anchor stage "
         "only, no '")
assert s.count(old_b) == 1, ('b', s.count(old_b))
s = s.replace(old_b, new_b)

# 3) T2 permutation code
old_c = ("rng2 = np.random.default_rng(SEED_T2)\n"
         "null2 = np.zeros(N_PERM)\n"
         "for it in range(N_PERM):\n"
         "    pc = rng2.permutation(c_old)\n"
         "    k = int(np.where((pc == WEATHER_PAIR[0])\n"
         "                     & (b_old == "
         "WEATHER_PAIR[1]))\n"
         "            [0][0])\n"
         "    null2[it] = float(nrmD[k])")
new_c = ("rng2 = np.random.default_rng(SEED_T2)\n"
         "blk = np.where(c_old == WEATHER_PAIR[0])[0]\n"
         "assert len(blk) == 8\n"
         "null2 = np.zeros(N_PERM)\n"
         "for it in range(N_PERM):\n"
         "    pb = rng2.permutation(b_old[blk])\n"
         "    j = int(np.where(\n"
         "        pb == WEATHER_PAIR[1])[0][0])\n"
         "    null2[it] = float(nrmD[blk[j]])")
assert s.count(old_c) == 1, ('c', s.count(old_c))
s = s.replace(old_c, new_c)

# 4) run label: run3 authoritative
old_d = "    'run': 'run1 authoritative',"
new_d = ("    'run': 'run3 authoritative (run1 crash "
         "pre-'\n"
         "           'verdict a98 npz-key bug; run2 "
         "crash '\n"
         "           'mid-T2 empty-cell under prefix "
         "label '\n"
         "           'permutation; T2 null redefined to "
         "the '\n"
         "           'within-block body-label "
         "permutation; '\n"
         "           'T1 observed in run2, design "
         "unchanged)',")
assert s.count(old_d) == 1, ('d', s.count(old_d))
s = s.replace(old_d, new_d)

io.open(P, 'w', encoding='utf-8').write(s)

import py_compile
py_compile.compile(P, doraise=True)
with open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
          r'\patch3043b_result.txt', 'w',
          encoding='utf-8') as f:
    f.write('patch ok, compile ok')
print('done')
