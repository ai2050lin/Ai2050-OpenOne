# -*- coding: utf-8 -*-
"""3153 patch2: M1 门段 Mte/Yte/e_m1 移出 t 循环（3152 同构），删 per-tpl 伪 e_m1。"""
import io

P = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3153_g1p3_failure_mode_anatomy.py"
s = io.open(P, encoding="utf-8").read()

old = """    Rm1 = np.zeros((NE_KEEP, NC, D), np.float32)
    subj = []
    for t in range(NT):
        Rg = np.zeros((NE_KEEP, NC, D), np.float32)
        mask = np.zeros((NE_KEEP, NC), bool)
        for pi, (i, c) in enumerate(PAIRS):
            if (i, c) in tr_set:
                Rg[keep_e.index(i), c] = \\
                    H16[t, pi, KSTAR, :].astype(np.float32) - \\
                    f7k_r['B4tr'][f7k_r['tr_rows'].index(t * NP_ + pi)]
                mask[keep_e.index(i), c] = True
        Rhat = als_complete(Rg, mask, RANK_M1, ALS_ITERS, ALS_RIDGE,
                            ALS_SEED + 1000 * SEEDS_S1[0] + KSTAR + t)
        if Rhat is None:
            Rhat = np.zeros_like(Rg)
        Rm1 += Rhat / NT
        Mte = np.zeros((len(f7k_r['te_rows']), D), np.float32)
        for j, r in enumerate(f7k_r['te_rows']):
            if r // NP_ == t:
                i2, c2 = PAIRS[r % NP_]
                Mte[j] = f7k_r['B4te'][j] + Rhat[keep_e.index(i2), c2]
        Yte = Y[f7k_r['te_rows']]
        e_m1 = float((((Mte - Yte) ** 2).sum(1) / f7k_r['Dk']).mean())
        subj.append(e_m1)
        log('M1@k* tpl%d mean=%.4f' % (t, e_m1))
    e_m1_mean = float(np.mean(subj))
"""
new = """    Rm1 = np.zeros((NE_KEEP, NC, D), np.float32)
    Mte = np.zeros((len(f7k_r['te_rows']), D), np.float32)
    for t in range(NT):
        Rg = np.zeros((NE_KEEP, NC, D), np.float32)
        mask = np.zeros((NE_KEEP, NC), bool)
        for pi, (i, c) in enumerate(PAIRS):
            if (i, c) in tr_set:
                Rg[keep_e.index(i), c] = \\
                    H16[t, pi, KSTAR, :].astype(np.float32) - \\
                    f7k_r['B4tr'][f7k_r['tr_rows'].index(t * NP_ + pi)]
                mask[keep_e.index(i), c] = True
        Rhat = als_complete(Rg, mask, RANK_M1, ALS_ITERS, ALS_RIDGE,
                            ALS_SEED + 1000 * SEEDS_S1[0] + KSTAR + t)
        if Rhat is None:
            Rhat = np.zeros_like(Rg)
        Rm1 += Rhat / NT
        for j, r in enumerate(f7k_r['te_rows']):
            if r // NP_ == t:
                i2, c2 = PAIRS[r % NP_]
                Mte[j] = f7k_r['B4te'][j] + Rhat[keep_e.index(i2), c2]
        log('M1@k* tpl%d done' % t)
    Yte = Y[f7k_r['te_rows']]
    e_m1_mean = float((((Mte - Yte) ** 2).sum(1) / f7k_r['Dk']).mean())
"""
assert s.count(old) == 1, ("m1 gate block count", s.count(old))
s = s.replace(old, new)

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("patched m1 gate block")
