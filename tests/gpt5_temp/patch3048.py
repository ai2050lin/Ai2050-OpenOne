# -*- coding: utf-8 -*-
import io
import py_compile

P = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3048_omega_p43_kfield_injection_qwen.py'
P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3048_omega_p45_kvpos_full_replay_qwen.py')
s = io.open(P, encoding='utf-8').read()

old1 = """    kp = np.stack([capKpre[li]['orig'][n] \\
                   .reshape(n, NL * 0 + HDIM * 8)
                   .cpu().numpy() if False else
                   capKpre[li]['orig'].cpu().numpy()
                   for li in range(NL)])"""
new1 = """    kp = np.stack([capKpre[li]['orig']
                   .cpu().numpy()
                   for li in range(NL)])"""
assert s.count(old1) == 1, ('dead-code', s.count(old1))
s = s.replace(old1, new1)

old2 = """    'T4_SCR': 'on the prefix prompt: replace '
              'k_norm/v_proj outputs at the %d '
              'prefix-token positions with '
              'norm-matched random vectors (seed '
              '%d); resp = lg - LG[pref]; kill_frac '
              '= ||resp||/||t_bc||, kill_cos = '
              'cos(resp, -t_bc); descriptive '
              'removal complement'
              % (0, SEED_SCR),"""
new2 = """    'T4_SCR': 'on the prefix prompt: replace '
              'k_norm/v_proj outputs at the '
              'prefix-token positions (0..off-1, '
              'pair-dependent) with norm-matched '
              'random vectors (seed %d); resp = lg '
              '- LG[pref]; kill_frac = '
              '||resp||/||t_bc||, kill_cos = '
              'cos(resp, -t_bc); descriptive '
              'removal complement' % SEED_SCR,"""
assert s.count(old2) == 1, ('t4scr', s.count(old2))
s = s.replace(old2, new2)

io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)
out = 'patched ok; compile ok\n'
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3048_result.txt', 'w',
        encoding='utf-8').write(out)
print(out)
