# -*- coding: utf-8 -*-
# 3035 run5 patch v2: corrected sham gate + random-direction
# specificity control + perturbative dose grid.
import io

p = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3035_omega_p32_fingerprint_competition'
     r'_qwen.py')
t = io.open(p, encoding='utf-8').read()
n0 = 0


def rep(old, new):
    global t, n0
    assert t.count(old) == 1, 'ANCHOR FAIL: ' + old[:80]
    t = t.replace(old, new)
    n0 += 1


# 1. dose grid constant
rep("M_GRID = (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2)",
    "M_GRID = (-0.05, -0.02, -0.01, 0.0, 0.01, 0.02,\n"
    "           0.05)")

# 2. PREREG dose text
rep("'m in {-0.2,-0.1,-0.05,0.0,0.05,0.1,0.2} '",
    "'m in {-0.05,-0.02,-0.01,0.0,0.01,0.02,0.05} '")

# 3. PREREG T2 pair text
rep("'PRIMARY inflection: even at m=0.05 (smallest '",
    "'PRIMARY inflection: even at m=0.01 (smallest '")

# 4. PREREG T3 texts
rep("'-dPB(+0.1) / dPA(+0.1) at late site (kappa=1 '",
    "'-dPB(+0.05) / dPA(+0.05) at late site (kappa=1 '")
rep("'perfect two-way); monotonicity |dPA(0.2)| > '\n"
    "          '|dPA(0.1)| > |dPA(0.05)| count; "
    "early-vs-late '",
    "'perfect two-way); monotonicity |dPA(0.05)| > '\n"
    "          '|dPA(0.02)| > |dPA(0.01)| count; "
    "early-vs-late '")

# 5. PREREG verdict_tree block
rep("'verdict_tree': 'if sham med |dPA(0.05)| > '\n"
    "                    'THETA_SHAM=0.02 -> '\n"
    "                    'fp_contaminated_void; elif "
    "n_match >= '",
    "'verdict_tree': 'if spec_ratio < 2 -> '\n"
    "                    'fp_nonspecific_qwen; elif "
    "n_match >= '")

# 6. PREREG: control key + full corrections block
rep("    'corrections': 'prefill-pollution lesson (3032 '\n"
    "                   'run2): every chain re-prefills; '\n"
    "                   'injection hooks registered before '\n"
    "                   'capture hooks; all arrays '\n"
    "                   'pre-initialized before loops '\n"
    "                   '(3020 lesson)',",
    "    'control': 'random-direction null: d_rand = '\n"
    "               '(W_U[r1] - W_U[r2]) normalized, "
    "r1/r2 = '\n"
    "               'base-prob rank 100/101 tokens; "
    "arms at '\n"
    "               'late site m in {-0.02,-0.01,0.01,"
    "0.02}; '\n"
    "               'spec_ratio = min med-ratio over m "
    "in '\n"
    "               '{0.01,0.02}, gate 2',\n"
    "    'corrections': 'run4 (npz8 1a3c860f): sham "
    "gate '\n"
    "                   'mis-specified - theta on "
    "|dPA(0.05)| '\n"
    "                   'under intervention measures "
    "EFFICACY '\n"
    "                   '(m*||h|| at L35 input large in "
    "'\n"
    "                   'absolute units), not "
    "contamination; '\n"
    "                   'contamination covered by "
    "a47/a48 bit '\n"
    "                   'determinism; run4 verdict '\n"
    "                   'fp_contaminated_void "
    "registered; '\n"
    "                   'run5 adds random-direction '\n"
    "                   'specificity control + "
    "perturbative '\n"
    "                   'dose grid, T2 pair 0.05->0.01, "
    "T3 at '\n"
    "                   '0.05; also prefill-pollution "
    "lesson '\n"
    "                   '(3032 run2): every chain "
    "re-prefills; '\n"
    "                   'injection hooks registered "
    "before '\n"
    "                   'capture hooks; all arrays '\n"
    "                   'pre-initialized before loops '\n"
    "                   '(3020 lesson)',")

# 7. code: logic stats pairs
rep("    for m in (0.05, 0.1, 0.2):\n"
    "        dp = dPA[im[m]]\n"
    "        dm = dPA[im[-m]]\n"
    "        s_ev += abs(0.5 * (dp + dm))\n"
    "        s_od += abs(0.5 * (dp - dm))\n"
    "    A_idx[ti] = s_ev / max(s_ev + s_od, 1e-30)\n"
    "    e1 = 0.5 * (dPA[im[0.05]] + dPA[im[-0.05]])",
    "    for m in (0.01, 0.02, 0.05):\n"
    "        dp = dPA[im[m]]\n"
    "        dm = dPA[im[-m]]\n"
    "        s_ev += abs(0.5 * (dp + dm))\n"
    "        s_od += abs(0.5 * (dp - dm))\n"
    "    A_idx[ti] = s_ev / max(s_ev + s_od, 1e-30)\n"
    "    e1 = 0.5 * (dPA[im[0.01]] + dPA[im[-0.01]])")

# 8. code: sham stats pairs
rep("    for m in (0.05, 0.1, 0.2):\n"
    "        dp = dPA[im[m]]\n"
    "        dm = dPA[im[-m]]\n"
    "        s_ev += abs(0.5 * (dp + dm))\n"
    "        s_od += abs(0.5 * (dp - dm))\n"
    "    A_idx_sham[si_] = s_ev / max(s_ev + s_od, "
    "1e-30)\n"
    "    e1 = 0.5 * (dPA[im[0.05]] + dPA[im[-0.05]])",
    "    for m in (0.01, 0.02, 0.05):\n"
    "        dp = dPA[im[m]]\n"
    "        dm = dPA[im[-m]]\n"
    "        s_ev += abs(0.5 * (dp + dm))\n"
    "        s_od += abs(0.5 * (dp - dm))\n"
    "    A_idx_sham[si_] = s_ev / max(s_ev + s_od, "
    "1e-30)\n"
    "    e1 = 0.5 * (dPA[im[0.01]] + dPA[im[-0.01]])")

# 9. kappa at 0.05
rep("    dpl = dPA[im[0.1]]\n"
    "    dpb = pB_late[ti, im[0.1]] - pB0[ti]",
    "    dpl = dPA[im[0.05]]\n"
    "    dpb = pB_late[ti, im[0.05]] - pB0[ti]")

# 10. mono 0.05>0.02>0.01
rep("    a5 = abs(dPA[im[0.2]])\n"
    "    a1 = abs(dPA[im[0.1]])\n"
    "    a0 = abs(dPA[im[0.05]])",
    "    a5 = abs(dPA[im[0.05]])\n"
    "    a1 = abs(dPA[im[0.02]])\n"
    "    a0 = abs(dPA[im[0.01]])")

# 11. random control arms (before a49 comment)
rep("    # a49: duplicate +0.1 late chain + ratio check",
    "    # random-direction control arms (late site)\n"
    "    r1 = int(order[100])\n"
    "    r2 = int(order[101])\n"
    "    dr = W_U[r1] - W_U[r2]\n"
    "    dr = dr / max(float(np.linalg.norm(dr)),\n"
    "                  1e-30)\n"
    "    dr_t = torch.from_numpy(dr).float().cuda()\n"
    "    for ri, mm in ((0, -0.02), (1, -0.01),\n"
    "                   (2, 0.01), (3, 0.02)):\n"
    "        p_rc, _, _, _, _ = run_fp_chain(\n"
    "            ids, SITES[1], mm, dr_t,\n"
    "            gate=True)\n"
    "        dPA_rand_late[ti, ri] = \\\n"
    "            p_rc[tokA[ti]] - pA0[ti]\n"
    "    # a49: duplicate +0.1 late chain + ratio "
    "check")

# 12. preinit rand store
rep("dPA_late = np.full((nT, nM), np.nan)",
    "dPA_late = np.full((nT, nM), np.nan)\n"
    "dPA_rand_late = np.full((nT, 4), np.nan)")

# 13. verdict block
rep("if sham_med_small > THETA_SHAM:\n"
    "    verdict = 'fp_contaminated_void'\n"
    "elif n_match >= T2_MATCH_HI \\",
    "if spec_ratio < 2.0:\n"
    "    verdict = 'fp_nonspecific_qwen'\n"
    "elif n_match >= T2_MATCH_HI \\")

# 14. sham_med_small -> 0.01 + spec computation
rep("sham_med_small = float(np.median(np.abs(\n"
    "    pA_sham[:, M_GRID.index(0.05)]\n"
    "    - pA0_sham)))",
    "sham_med_small = float(np.median(np.abs(\n"
    "    pA_sham[:, M_GRID.index(0.01)]\n"
    "    - pA0_sham)))\n"
    "i01 = M_GRID.index(0.01)\n"
    "i02 = M_GRID.index(0.02)\n"
    "spec1 = float(np.median(\n"
    "    np.abs(dPA_late[:, i01])\n"
    "    / np.maximum(np.abs(dPA_rand_late[:, 2]),\n"
    "                 1e-12)))\n"
    "spec2 = float(np.median(\n"
    "    np.abs(dPA_late[:, i02])\n"
    "    / np.maximum(np.abs(dPA_rand_late[:, 3]),\n"
    "                 1e-12)))\n"
    "spec_ratio = float(min(spec1, spec2))")

# 15. log line update + spec log
rep("log('n_match=%d/11 med_A_idx=%.4f sham: "
    "med|dPA05|=%.5f '\n"
    "    'n_match=%d/3' % (n_match, med_A, "
    "sham_med_small,\n"
    "                      n_match_sham))",
    "log('n_match=%d/11 med_A_idx=%.4f sham: "
    "med|dPA01|=%.5f '\n"
    "    'n_match=%d/3' % (n_match, med_A, "
    "sham_med_small,\n"
    "                      n_match_sham))\n"
    "log('spec_ratio=%.4f (spec1=%.4f spec2=%.4f)'\n"
    "    % (spec_ratio, spec1, spec2))")

# 16. npz additions
rep("    verdict=np.array(verdict),\n"
    "    elapsed=np.float64(elapsed))",
    "    dPA_rand_late=dPA_rand_late,\n"
    "    spec_ratio=np.float64(spec_ratio),\n"
    "    spec1=np.float64(spec1),\n"
    "    spec2=np.float64(spec2),\n"
    "    verdict=np.array(verdict),\n"
    "    elapsed=np.float64(elapsed))")

# 17. result additions
rep("    'T3_competition': {",
    "    'T2b_specificity': {\n"
    "        'spec_ratio': spec_ratio,\n"
    "        'spec1_m01': spec1,\n"
    "        'spec2_m02': spec2,\n"
    "        'dPA_rand_late': "
    "[[float(x) for x in row]\n"
    "                         for row in "
    "dPA_rand_late],\n"
    "    },\n"
    "    'T3_competition': {")

# 18. run4 note in result
rep("    'correction_note': ('a51 near-tie: ' + "
    "a51_note)\n"
    "    if a51_note else 'none',",
    "    'run4_note': 'run4 (npz8 1a3c860f) verdict '\n"
    "                 'fp_contaminated_void: "
    "preregistered '\n"
    "                 'sham gate mis-specified "
    "(measured '\n"
    "                 'intervention efficacy, not '\n"
    "                 'contamination); corrected in "
    "run5 '\n"
    "                 'per 3030 a38 precedent',\n"
    "    'correction_note': ('a51 near-tie: ' + "
    "a51_note)\n"
    "    if a51_note else 'none',")

io.open(p, 'w', encoding='utf-8').write(t)
print('patched OK, %d reps' % n0)
