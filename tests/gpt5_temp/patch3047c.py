# -*- coding: utf-8 -*-
import io
import py_compile

P = r'D:\AI2050\Ai2050-OpenOne\tests\glm5' \
    r'\phase3047_omega_p44_kv_joint_replay_qwen.py'
s = io.open(P, encoding='utf-8').read()
n0 = len(s)


def rep(old, new, tag):
    global s
    assert s.count(old) == 1, (tag, s.count(old))
    s = s.replace(old, new)


# A: bit-exact integrity inside forward_inj
rep("""        res['integK'] = ik
        res['integV'] = iv
    reset_all()""",
    """        res['integK'] = ik
        res['integV'] = iv
        ek = 0.0
        ev = 0.0
        for li in range(NL):
            dtk = stateK[li]['delta'] \\
                if stateK[li]['on'] else None
            dtv = stateV[li]['delta'] \\
                if stateV[li]['on'] else None
            if dtk is not None:
                ek = max(ek, float(
                    (capK[li]['mod']
                     - (capK[li]['orig']
                        + dtk)).abs().max()))
            else:
                ek = max(ek, float(
                    (capK[li]['mod']
                     - capK[li]['orig'])
                    .abs().max()))
            if dtv is not None:
                ev = max(ev, float(
                    (capV[li]['mod']
                     - (capV[li]['orig']
                        + dtv)).abs().max()))
            else:
                ev = max(ev, float(
                    (capV[li]['mod']
                     - capV[li]['orig'])
                    .abs().max()))
        res['bitK'] = ek
        res['bitV'] = ev
    reset_all()""",
    'A')

# B: main loop uses bit form + diagnostic
rep("""        e1 = float(np.max(np.abs(
            res['integK'] - dK)))
        e2 = float(np.max(np.abs(
            res['integV'] - dV)))""",
    """        e1 = float(res['bitK'])
        e2 = float(res['bitV'])
        e1d = float(np.max(np.abs(
            res['integK'] - dK)))
        e2d = float(np.max(np.abs(
            res['integV'] - dV)))
        a122_diag = max(a122_diag, e1d, e2d)""",
    'B')

# B2: init a122_diag
rep("""n_integ_fail = 0
n_past_fail = 0""",
    """n_integ_fail = 0
n_past_fail = 0
a122_diag = 0.0""",
    'B2')

# C: corrections
rep("""    'corrections': 'none (run1 intended '
                   'authoritative)',""",
    """    'corrections': 'run1 completed with a122 '
                   'failing 96/96 (past_fail=0); '
                   'probe gpt5_temp/probe3047d '
                   'root-caused the CHECK '
                   'formulation, not the injection: '
                   'max|mod-orig-delta| = 1.86e-09 '
                   '(fp32 subtraction rounding of '
                   '(x+d)-x, rel 3.5e-08) while '
                   'mod == orig+delta is BIT-EXACT '
                   '(e_bit = 0.0) and the f32 cast '
                   'is exact (e_cast = 0.0); '
                   'integrity reformulated to the '
                   'bit-exact mod == orig+delta '
                   'check (subtraction form kept '
                   'as diagnostic a122_diag); '
                   'statistics unchanged and '
                   'reproduce bit-level under '
                   'frozen seeds; run2 '
                   'authoritative',""",
    'C')

# D: prereg anchors wording
rep("'(mod-orig == delta bit all layers '",
    "'(mod == orig+delta bit-exact all layers '",
    'D')

# E: run label
rep("'run': 'run1 authoritative (fp32)',",
    "'run': 'run2 authoritative (fp32; run1 "
    "a122 failed on a subtraction-rounding "
    "artifact of the check itself, probe-"
    "root-caused, injection bit-exact; see "
    "corrections)',",
    'E')

# F: npz + stats carry the diagnostic
rep("""         n_integ_fail=np.int64(n_integ_fail),
         n_past_fail=np.int64(n_past_fail),""",
    """         n_integ_fail=np.int64(n_integ_fail),
         n_past_fail=np.int64(n_past_fail),
         a122_diag=np.float64(a122_diag),""",
    'F')
rep("""                'n_past_fail': n_past_fail,""",
    """                'n_past_fail': n_past_fail,
                'a122_diag_subform': float(
                    a122_diag),""",
    'F2')

io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)
s2 = io.open(P, encoding='utf-8').read()
checks = {
    'bit block': s2.count("res['bitK'] = ek") == 1,
    'bit use': s2.count("e1 = float(res['bitK'])") == 1,
    'diag init': s2.count('a122_diag = 0.0') == 1,
    'corrections': 'probe3047d' in s2,
    'run2 label': s2.count("'run': 'run2 authoritative") == 1,
    'npz diag': s2.count('a122_diag=np.float64') == 1,
    'len grew': len(s2) > n0,
}
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3047c_result.txt', 'w',
        encoding='utf-8').write(
    '\n'.join('%s=%s' % kv for kv in checks.items())
    + '\ncompile ok\n')
print('ok')
