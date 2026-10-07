# -*- coding: utf-8 -*-
import io
import py_compile

P = r'D:\AI2050\Ai2050-OpenOne\tests\glm5' \
    r'\phase3047_omega_p44_kv_joint_replay_qwen.py'
s = io.open(P, encoding='utf-8').read()

old = """    'T4_removal': {'med_rem_frac': float(
        np.median(FRAC['RM'])),
        'med_rem_cos': float(np.median(COS['RM']))},"""
new = """    'T4_removal': {'med_rem_frac': float(
        np.median(FRAC['RM'])),
        # rem_cos = cos(lgRM - lgA, -t_bc);
        # +1 = full erasure (lgRM ~ lgB)
        'med_rem_cos': float(
            np.median(-COS['RM']))},"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)
s2 = io.open(P, encoding='utf-8').read()
ok = "np.median(-COS['RM'])" in s2
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3047b_result.txt', 'w',
        encoding='utf-8').write('rem_cos fixed=%s\n'
                                'compile ok\n' % ok)
print('ok')
