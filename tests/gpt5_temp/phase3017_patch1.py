# -*- coding: utf-8 -*-
"""Phase 3017 patch1: fix T2a block (indexing bug +
dead code removal) + absorb_stack returns res_b."""
import io
import re

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3017_omega_p2k_deep_absorption_qwen.py')
t = io.open(P, encoding='utf-8').read()
miss = []

# ---- 1. absorb_stack: collect res_b ----
a = """                e_cols = []
                js_e_l = []
                ca_cols = []
                cm_cols = []
                idf_l = []"""
b = """                e_cols = []
                rb_cols = []
                js_e_l = []
                ca_cols = []
                cm_cols = []
                idf_l = []"""
if a in t:
    t = t.replace(a, b, 1)
else:
    miss.append('rb-init')

a = """                    e_all = res_e - res_b
                    e_cols.append(e_all)"""
b = """                    e_all = res_e - res_b
                    e_cols.append(e_all)
                    rb_cols.append(res_b)"""
if a in t:
    t = t.replace(a, b, 1)
else:
    miss.append('rb-append')

a = """                out_d = {
                    'e': np.stack(e_cols)
                    if e_cols else np.array([]),"""
b = """                out_d = {
                    'e': np.stack(e_cols)
                    if e_cols else np.array([]),
                    'res_b': np.stack(rb_cols)
                    if rb_cols else np.array([]),"""
if a in t:
    t = t.replace(a, b, 1)
else:
    miss.append('rb-out')

# ---- 2. replace maxT + sig + shrink/rel block ----
start = t.index('            # maxT permutation null')
end = t.index('            gates_ok = bool(')
new_blk = """            # maxT permutation null (circular shift,
            # shared shift per permutation)
            # family: layers l = 4..34; for each (tag,
            # layer): D = e[l+1]-e[l], stat = cos(D, e)
            p_layer = np.ones(NL)
            med_c_of = {}
            if nL:
                Ls = list(range(L_START, NL - 1))
                D_sub = e_stack[:, 1:, :] \\
                    - e_stack[:, :-1, :]
                D_sub = D_sub[:, L_START:NL - 1, :]
                E_sub = e_stack[:, L_START:NL - 1, :]
                dnorm = np.linalg.norm(D_sub, axis=2)
                enorm = np.linalg.norm(E_sub, axis=2)
                corr = np.fft.irfft(
                    np.fft.rfft(D_sub, axis=-1)
                    * np.conj(np.fft.rfft(
                        E_sub, axis=-1)),
                    n=HID, axis=-1)
                # nc[layer, tag, shift] normalized cos
                nc = (corr / np.maximum(
                    dnorm[:, :, None]
                    * enorm[:, :, None], 1e-30)
                ).transpose(1, 0, 2)
                obs_stat = np.array(
                    [float(med_c[li - L_START])
                     for li in Ls])
                for li in Ls:
                    med_c_of[li] = float(
                        med_c[li - L_START])
                rng_p = np.random.default_rng(
                    SEED_NULL + 3017)
                cnt = np.zeros(len(Ls))
                for _ in range(N_PERM):
                    s = int(rng_p.integers(0, HID))
                    st = np.median(nc[:, :, s],
                                   axis=1)
                    m = float(np.min(st))
                    cnt += (st <= obs_stat) \\
                        .astype(float)
                p_layer[Ls] = cnt / N_PERM
            sig_layers = [li for li in
                          range(L_START, NL - 1)
                          if nL
                          and p_layer[li] < P_GATE
                          and med_c_of[li] < 0]
            sig_late = [li for li in sig_layers
                        if li >= L_LATE]
            shrink = None
            reldecay = None
            rel_prof = None
            if nL:
                r_e = np.linalg.norm(
                    e_stack[:, 35, :], axis=1) \\
                    / np.maximum(np.linalg.norm(
                        e_stack[:, 4, :], axis=1),
                        1e-30)
                shrink = float(np.median(r_e))
                rel_prof = np.linalg.norm(
                    e_stack, axis=2) / np.maximum(
                    np.linalg.norm(A['res_b'],
                                   axis=2), 1e-30)
                reldecay = float(
                    np.median(rel_prof[:, 35]
                              / np.maximum(
                                  rel_prof[:, 4],
                                  1e-30)))

"""
t = t[:start] + new_blk + t[end:]

io.open(P, 'w', encoding='utf-8').write(t)

# verify
t2 = io.open(P, encoding='utf-8').read()
chk = {
    'rb-cols': "rb_cols.append(res_b)" in t2,
    'fft-fixed': "D_sub[:, L_START:NL - 1, :]" in t2,
    'no-dead-F': "if False else None" not in t2,
    'no-dead-rel': "rel = np.linalg.norm" not in t2,
    'med_c_of': "med_c_of[li] = float(" in t2,
    'sig-clean': "and med_c_of[li] < 0]" in t2,
    'relprof-resb': "A['res_b']" in t2,
}
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_p17.txt', 'w',
        encoding='utf-8').write(
    'miss=%s chk=%s' % (miss, chk))
print('ok')
