# -*- coding: utf-8 -*-
"""Phase 3017 patch3: identity gate -> single-chain
form (denominator = write norm, same scale as bf16
noise); cross-chain identity follows linearly."""
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3017_omega_p2k_deep_absorption_qwen.py')
t = io.open(P, encoding='utf-8').read()
miss = []

# 1. absorb_stack: replace identity computation
a = """                    ca = np.zeros(NL)
                    cm = np.zeros(NL)
                    idmax = 0.0
                    dnorm_med = []
                    for li in range(L_START, NL - 1):
                        dA = a_e[li] - a_b[li]
                        dM = m_e[li] - m_b[li]
                        dT = e_all[li + 1] \\
                            - e_all[li]
                        dnorm_med.append(
                            float(np.linalg.norm(dT)))
                        idmax = max(idmax, float(
                            np.max(np.abs(
                                dA + dM - dT))))
                        na = float(np.linalg.norm(dA))"""
b = """                    ca = np.zeros(NL)
                    cm = np.zeros(NL)
                    idmax = 0.0
                    wnorm = []
                    dnorm_med = []
                    for li in range(L_START, NL - 1):
                        dA = a_e[li] - a_b[li]
                        dM = m_e[li] - m_b[li]
                        dT = e_all[li + 1] \\
                            - e_all[li]
                        dnorm_med.append(
                            float(np.linalg.norm(dT)))
                        na = float(np.linalg.norm(dA))"""
if a in t:
    t = t.replace(a, b, 1)
else:
    miss.append('id-block')

# 2. insert single-chain identity loop before the
#    ca_cols.append
a = """                    ca_cols.append(ca)
                    cm_cols.append(cm)
                    idf_l.append(
                        idmax / max(float(np.median(
                            dnorm_med)), 1e-30)
                        if dnorm_med else np.nan)"""
b = """                    for cr, caa, cmm in (
                            (res_b, a_b, m_b),
                            (res_e, a_e, m_e)):
                        for li2 in range(L_START,
                                         NL - 1):
                            w = caa[li2] + cmm[li2]
                            wnorm.append(float(
                                np.linalg.norm(w)))
                            idmax = max(idmax, float(
                                np.max(np.abs(
                                    cr[li2 + 1]
                                    - cr[li2] - w))))
                    ca_cols.append(ca)
                    cm_cols.append(cm)
                    idf_l.append(
                        idmax / max(float(np.median(
                            wnorm)), 1e-30)
                        if wnorm else np.nan)"""
if a in t:
    t = t.replace(a, b, 1)
else:
    miss.append('single-chain-loop')

# 3. correction_note: register run2
a = "'run2: authoritative',"
b = ("'run2: verdict again absorption_"
     "undetermined_void (idf=0.152) - the cross-"
     "chain identity gate divides by med||dT||, a "
     "SMALL DIFFERENCE, so bf16 quantization noise "
     "(rel ~2^-9, amplified in differencing) "
     "dominates the gate (run1 structural error "
     "was 0.79, run2 bf16 noise 0.152); fix: the "
     "identity gate now checks the SINGLE-CHAIN "
     "recursion res[l+1]-res[l]-attn-mlp against "
     "the write norm ||a+m|| (same scale as the "
     "noise); the cross-chain identity D_l = "
     "e_{l+1}-e_l follows linearly from the two "
     "single-chain identities; run3: "
     "authoritative',")
if a in t:
    t = t.replace(a, b, 1)
else:
    miss.append('note-run2')

io.open(P, 'w', encoding='utf-8').write(t)

t2 = io.open(P, encoding='utf-8').read()
chk = {
    'wnorm': 'wnorm.append(float(' in t2,
    'single': 'cr[li2 + 1]\n                                    - cr[li2] - w' in t2,
    'idf-wnorm': 'max(float(np.median(\n                            wnorm)), 1e-30)' in t2,
    'note-run2': 'run3: ' in t2,
    'no-old-idf': 'idf_l.append(\n                        idmax / max(float(np.median(\n                            dnorm_med)), 1e-30)' not in t2,
}
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_p17d.txt', 'w',
        encoding='utf-8').write(
    'miss=%s chk=%s' % (miss, chk))
print('ok')
