# -*- coding: utf-8 -*-
"""Patch phase3032 -> phase3034 (head-set identity).
Surgical, assert count==1 on every anchor."""
import io
import os
import shutil

SRC = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase3032_omega_p2z_deep_peak_anatomy'
       r'_qwen.py')
DST = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase3034_omega_p31_headset_identity'
       r'_qwen.py')

shutil.copyfile(SRC, DST)
t = io.open(DST, encoding='utf-8').read()
n0 = len(t)


def rep(old, new, cnt=1):
    global t
    c = t.count(old)
    assert c == cnt, (c, old[:80])
    t = t.replace(old, new)


# ---- 1. identity ----
rep("PHASE = 3032", "PHASE = 3034")
rep("NAME = 'omega_p2z_deep_peak_anatomy_qwen'",
    "NAME = 'omega_p31_headset_identity_qwen'")
rep("OUT = os.path.join(BASE, 'phase3032', NAME)",
    "OUT = os.path.join(BASE, 'phase3034', NAME)")

# ---- 2. docstring ----
i0 = t.index('"""')
i1 = t.index('"""', i0 + 3) + 3
DOC = '"""Phase 3034: Omega-P31 - deep-peak\n\
head-SET identity (pathway vs contextual).\n\
\n\
Question (3032 follow-up A): the deep-peak\n\
alpha=2 arm response is carried by top-8/32\n\
heads (med 89.3% of per-head d-o energy).\n\
Is that head SET the same across tags (a\n\
pathway property, like a dedicated circuit)\n\
or tag-specific (a third contextual property,\n\
after 3027 consumption and 3032 carrier\n\
maps)?  PRIMARY T1: mean pairwise Jaccard of\n\
the per-tag top-8 head sets (d-d energy at\n\
each tag deep-peak layer ldp, verbatim 3032\n\
machine) vs exact hypergeometric null of\n\
random 8-of-32 subsets (200k, seed 30341).\n\
T2: within-tag deep-vs-mid-peak set Jaccard\n\
(seed 30342).  T3: GQA group-7 (q28-31)\n\
enrichment of pooled top-8 picks (seed\n\
30343).  T4: per-tag Spearman(dd, p_h) vs\n\
3027 consumption head profile (seed 30344).\n\
\n\
Verdict map (frozen):\n\
  gates or a42-a46 fail => headset_undetermined_void\n\
  p_T1<.05 and J>null med => headset_pathway_qwen\n\
  p_T1<.05 and J<null med => headset_anti_aligned_qwen\n\
  else                    => headset_relational_qwen\n\
\n\
Anchors: full 3032 suite (a0-a41 incl a28/\n\
a30/a32/a39/a40 bit-level 0.0, a41 vs 3022)\n\
plus a42 head_top8 share recompute from stashed\n\
dd vs 3032 npz (bit-level), a43 E matrix\n\
bit-level vs 3032, a44 ldp recompute from E\n\
vs 3032, a45 3027 p_h row-sum + tags, a46\n\
source seal integrity (3032/3027 npz).\n\
"""'
t = t[:i0] + DOC + t[i1:]

# ---- 3. PREREG ----
p0 = t.index('PREREG = {')
p1 = t.index('\n}\n', p0) + 3
PR = '''PREREG = {
    'mode': 'GPU rerun of the verbatim 3032 '
            'machine (chains/lens/attribution '
            'identical, anchors a0-a41 re-checked '
            'in-run) with dd per-head d-o energy '
            'stashed at ldp and mid-peak layers; '
            'head-set statistics are new',
    'question': 'is the 3032 deep-peak top-8 head '
                'set a pathway property (same heads '
                'across tags, Jaccard above the '
                'random 8-of-32 null) or a third '
                'contextual property (Jaccard at '
                'null level)?',
    'T1': 'PRIMARY: per-tag top-8 head sets by dd '
          '(per-head squared d-o norm at ldp, '
          'stable argsort); mean pairwise Jaccard '
          'over the 45 pairs of valid tags; null '
          '= hypergeometric(32,8,8) overlap, '
          'NPERM=200000 seed 30341, one-sided '
          'p = P(null >= J_obs)',
    'T2': 'within-tag deep-vs-mid-peak top-8 '
          'Jaccard mean; null = hypergeometric '
          'mean over the same number of pairs, '
          'seed 30342',
    'T3': 'pooled count of top-8 members in GQA '
          'group-7 (q28-31) across valid tags; '
          'null = hypergeometric(32,4,8) summed '
          'over tags, seed 30343',
    'T4': 'per-tag Spearman(dd, p_h_3027) median; '
          'null = within-tag rank permutation, '
          'seed 30344; exploratory (3027 showed '
          'consumption is not head-specific)',
    'T3m': 'margin caveat: 10 valid tags, '
           'head sets of size 8 from 32; '
           'P7 has no deep peak (nan, excluded); '
           'set statistics exact via '
           'hypergeometric, no asymptotics',
    'verdict': 'gates or a42-a46 fail => '
               'headset_undetermined_void; '
               'p_T1<.05 and J>null med => '
               'headset_pathway_qwen; p_T1<.05 '
               'and J<null med => '
               'headset_anti_aligned_qwen; '
               'else => '
               'headset_relational_qwen',
    'tags': 'Omega-P31 / deep-peak head-set '
            'identity / verbatim 3032 machine '
            'with stash / exact hypergeometric '
            'nulls / no hallucination naming',
}
'''
t = t[:p0] + PR + t[p1:]

# ---- 4. pre-init ----
rep("    js_alpha0_all = []\n",
    "    js_alpha0_all = []\n"
    "    dd_deep_list = []\n"
    "    dd_mid_list = []\n"
    "    DD = np.zeros((0, NHQ))\n"
    "    DDM = np.zeros((0, NHQ))\n"
    "    a42_diff = None\n"
    "    a42_ok = False\n"
    "    a43_diff = None\n"
    "    a43_ok = False\n"
    "    a44_diff = None\n"
    "    a44_ok = False\n"
    "    a45_max = None\n"
    "    a45_ok = False\n"
    "    a46_ok = False\n"
    "    J_obs = float('nan')\n"
    "    p_T1 = float('nan')\n"
    "    K_obs = -1\n"
    "    p_T3 = float('nan')\n"
    "    rho4 = float('nan')\n"
    "    p_T4 = float('nan')\n")

# ---- 5. stash in deep_share_head ----
rep("            def deep_share_head(o_2, o_b, li):\n"
    "                dd = np.linalg.norm(\n"
    "                    o_2[li].reshape(NHQ, HDIM)\n"
    "                    - o_b[li].reshape(NHQ, HDIM),\n"
    "                    axis=1) ** 2\n"
    "                tot = float(dd.sum())",
    "            DD_STASH = {}\n"
    "\n"
    "            def deep_share_head(o_2, o_b, li):\n"
    "                dd = np.linalg.norm(\n"
    "                    o_2[li].reshape(NHQ, HDIM)\n"
    "                    - o_b[li].reshape(NHQ, HDIM),\n"
    "                    axis=1) ** 2\n"
    "                DD_STASH[li] = dd.copy()\n"
    "                tot = float(dd.sum())")

# ---- 6. capture deep dd ----
rep("                    head8_list.append(\n"
    "                        deep_share_head(o_2, o_b,\n"
    "                                        li_l))",
    "                    head8_list.append(\n"
    "                        deep_share_head(o_2, o_b,\n"
    "                                        li_l))\n"
    "                    dd_deep_list.append(\n"
    "                        DD_STASH[li_l].copy())")

# ---- 7. capture mid dd ----
rep("                        head8_mid.append(\n"
    "                            deep_share_head(\n"
    "                                o_2, o_b, li_m))",
    "                        head8_mid.append(\n"
    "                            deep_share_head(\n"
    "                                o_2, o_b, li_m))\n"
    "                        dd_mid_list.append(\n"
    "                            DD_STASH[li_m]"
    ".copy())")

# ---- 8. no-deep-peak else branch ----
rep("                else:\n"
    "                    head8_list.append(np.nan)\n"
    "                    mlp32_list.append(np.nan)",
    "                else:\n"
    "                    head8_list.append(np.nan)\n"
    "                    dd_deep_list.append(\n"
    "                        np.full(NHQ, np.nan))\n"
    "                    mlp32_list.append(np.nan)")

# ---- 9. outer-else mid nan ----
rep("                    ldp_layer_list.append(-1)\n"
    "                    late_list.append(np.nan)\n"
    "                    head8_mid.append(np.nan)\n"
    "                    mlp32_mid.append(np.nan)",
    "                    ldp_layer_list.append(-1)\n"
    "                    late_list.append(np.nan)\n"
    "                    head8_mid.append(np.nan)\n"
    "                    dd_mid_list.append(\n"
    "                        np.full(NHQ, np.nan))\n"
    "                    mlp32_mid.append(np.nan)")

# ---- 10. no-mid-peak else branch ----
rep("                    else:\n"
    "                        head8_mid.append(np.nan)\n"
    "                        mlp32_mid.append(np.nan)",
    "                    else:\n"
    "                        head8_mid.append(np.nan)\n"
    "                        dd_mid_list.append(\n"
    "                            np.full(NHQ, np.nan))\n"
    "                        mlp32_mid.append(np.nan)")

# ---- 11. stats splice ----
s0 = t.index("            # ---------- T2a verdict ----------")
s1 = t.index("    if verdict is None:")
NEW = '''            # ---------- a42-a46 ----------
            z32n = np.load(os.path.join(
                BASE, 'phase3032',
                'omega_p2z_deep_peak_anatomy_qwen',
                'omega_p2z_deep_peak_anatomy_'
                'qwen.npz'), allow_pickle=True)
            D27 = os.path.join(
                BASE, 'phase3027',
                'omega_p2u_consumer_heads_qwen')
            zc27 = np.load(os.path.join(
                D27,
                'omega_p2u_consumer_heads_qwen'
                '.npz'), allow_pickle=True)
            ph = zc27['p_h'].astype(np.float64)
            s32 = json.load(io.open(
                os.path.join(
                    BASE, 'phase3032',
                    'omega_p2z_deep_peak_anatomy_'
                    'qwen', 'seal.json'),
                encoding='utf-8'))
            s27 = json.load(io.open(
                os.path.join(D27, 'seal.json'),
                encoding='utf-8'))
            a46_ok = bool(
                sha8(os.path.join(
                    BASE, 'phase3032',
                    'omega_p2z_deep_peak_anatomy_'
                    'qwen',
                    'omega_p2z_deep_peak_'
                    'anatomy_qwen.npz'))
                == s32['npz_sha256_8']
                and sha8(os.path.join(
                    D27,
                    'omega_p2u_consumer_heads_'
                    'qwen.npz'))
                == s27['npz_sha256_8'])
            a45_max = float(np.max(np.abs(
                ph.sum(axis=1) - 1.0)))
            a45_ok = bool(
                a45_max <= 1e-6
                and [str(x)
                     for x in zc27['tags']]
                == list(tags))

            DD = np.stack(dd_deep_list)
            DDM = np.stack(dd_mid_list)
            ht8 = z32n['head_top8']

            d42 = []
            for i in range(DD.shape[0]):
                if not np.isfinite(DD[i]).all():
                    continue
                if not np.isfinite(ht8[i]):
                    continue
                sh = float(
                    np.sort(DD[i])[::-1][:TOPH]
                    .sum()) / max(float(
                    DD[i].sum()), 1e-30)
                d42.append(abs(sh
                               - float(ht8[i])))
            a42_diff = float(max(d42)) \\
                if d42 else float('nan')
            a42_ok = bool(d42
                          and a42_diff == 0.0)

            a43_diff = float(np.max(np.abs(
                E_all - z32n['E'])))
            a43_ok = bool(a43_diff == 0.0)

            ldp_rec = []
            for i in range(E_all.shape[0]):
                ev = E_all[i]
                pdv = float(ev[DEEP_LO:DEEP_HI]
                            .sum())
                if pdv > 0:
                    ldp_rec.append(DEEP_LO
                                   + int(np.argmax(
                                       ev[DEEP_LO:
                                          DEEP_HI])))
                else:
                    ldp_rec.append(-1)
            a44_diff = float(np.max(np.abs(
                np.array(ldp_rec)
                - z32n['ldp_idx'])))
            a44_ok = bool(a44_diff == 0.0)
            log('a42 %.2e a43 %.2e a44 %.2e '
                'a45 %.2e a46 %s'
                % (a42_diff, a43_diff, a44_diff,
                   a45_max, a46_ok), lines)

            # ---------- T1 sets ----------
            valid_idx = [i for i in
                         range(len(tags))
                         if np.isfinite(ht8[i])]
            sets_d = {}
            for i in valid_idx:
                order = np.argsort(-DD[i],
                                   kind='stable')
                sets_d[i] = set(int(h)
                                for h in
                                order[:TOPH])
            pairs = [(a, b)
                     for ai, a in
                     enumerate(valid_idx)
                     for b in
                     valid_idx[ai + 1:]]

            def jac(sa, sb):
                u = len(sa | sb)
                if u == 0:
                    return float('nan')
                return float(len(sa & sb)) / u

            J_obs = float(np.mean(
                [jac(sets_d[a], sets_d[b])
                 for a, b in pairs]))
            NPERM = 200000
            rng1 = np.random.default_rng(
                30341)
            ov1 = rng1.hypergeometric(
                TOPH, NHQ - TOPH, TOPH,
                size=(NPERM, len(pairs)))
            null1 = (ov1.astype(float)
                     / (2.0 * TOPH
                        - ov1)).mean(axis=1)
            p_T1 = float(
                (np.sum(null1 >= J_obs) + 1)
                / (NPERM + 1))
            log('T1 J_obs=%.4f nullmed=%.4f '
                'p=%.6f'
                % (J_obs,
                   float(np.median(null1)),
                   p_T1), lines)

            # ---------- T2 deep vs mid ---
            pairs2 = []
            for i in valid_idx:
                if np.isfinite(DDM[i]).all() \\
                        and float(DDM[i].sum()) > 0:
                    order = np.argsort(
                        -DDM[i], kind='stable')
                    sets_m = set(int(h)
                                 for h in
                                 order[:TOPH])
                    pairs2.append(jac(
                        sets_d[i], sets_m))
            J2_obs = float(np.mean(pairs2)) \\
                if pairs2 else float('nan')
            rng2 = np.random.default_rng(
                30342)
            ov2 = rng2.hypergeometric(
                TOPH, NHQ - TOPH, TOPH,
                size=(NPERM,
                      max(len(pairs2), 1)))
            null2 = (ov2.astype(float)
                     / (2.0 * TOPH
                        - ov2)).mean(axis=1)
            p_T2 = float(
                (np.sum(null2 >= J2_obs) + 1)
                / (NPERM + 1)) if pairs2 \\
                else float('nan')
            log('T2 J2=%.4f p=%.6f n=%d'
                % (J2_obs, p_T2,
                   len(pairs2)), lines)

            # ---------- T3 GQA group7 ----
            GH = {28, 29, 30, 31}
            K_obs = int(sum(
                len(sets_d[i] & GH)
                for i in valid_idx))
            rng3 = np.random.default_rng(
                30343)
            nul3 = rng3.hypergeometric(
                4, NHQ - 4, TOPH,
                size=(NPERM,
                      len(valid_idx))).sum(
                axis=1)
            p_T3 = float(
                (np.sum(nul3 >= K_obs) + 1)
                / (NPERM + 1))
            log('T3 K_obs=%d p=%.6f'
                % (K_obs, p_T3), lines)

            # ---------- T4 dd vs p_h -----
            def rank_avg1(v):
                v = np.asarray(v, dtype=float)
                order = np.argsort(v,
                                   kind='stable')
                ranks = np.empty(v.size,
                                 dtype=float)
                sv = v[order]
                i = 0
                while i < v.size:
                    j = i
                    while j + 1 < v.size \\
                            and sv[j + 1] == sv[i]:
                        j += 1
                    avg = 0.5 * (i + j) + 1.0
                    ranks[order[i:j + 1]] = avg
                    i = j + 1
                return ranks

            Rxc = []
            Ryc = []
            sp_list = []
            for i in valid_idx:
                rx = rank_avg1(DD[i])
                ry = rank_avg1(ph[i])
                rc = rx - rx.mean()
                yc = ry - ry.mean()
                nrx = float(np.sqrt(
                    (rc * rc).sum()))
                nry = float(np.sqrt(
                    (yc * yc).sum()))
                if nrx <= 0 or nry <= 0:
                    sp_list.append(
                        float('nan'))
                    Rxc.append(np.zeros(NHQ))
                    Ryc.append(np.zeros(NHQ))
                    continue
                Rxc.append(rc / nrx)
                Ryc.append(yc / nry)
                sp_list.append(float(
                    (rc * yc).sum())
                    / (nrx * nry))
            rho4 = float(np.nanmedian(
                sp_list)) if sp_list \\
                else float('nan')
            rng4 = np.random.default_rng(
                30344)
            perms4 = np.argsort(
                rng4.random((NPERM, NHQ)),
                axis=1)
            Rm = np.stack(Rxc)
            Ym = np.stack(Ryc)
            null4 = np.empty(NPERM)
            CH = 10000
            for c0 in range(0, NPERM, CH):
                pc = perms4[c0:c0 + CH]
                vals = np.einsum(
                    'ij,ikj->ik', Rm,
                    Ym[:, pc])
                null4[c0:c0 + CH] = \\
                    np.median(vals, axis=0)
            p_T4 = float(
                (np.sum(np.abs(null4)
                        >= abs(rho4)) + 1)
                / (NPERM + 1))
            log('T4 rho=%.4f p=%.6f'
                % (rho4, p_T4), lines)

            # ---------- verdict ----------
            head_arr = np.array(head8_list,
                                dtype=float)
            mlp_arr = np.array(mlp32_list,
                               dtype=float)
            gates_ok = bool(nL >= 8
                            and min(js_er) > 0
                            and len(valid_idx) >= 6
                            and a42_ok and a43_ok
                            and a44_ok and a45_ok
                            and a46_ok)
            med_null1 = float(
                np.median(null1))
            if not gates_ok:
                verdict = \\
                    'headset_undetermined_void'
            elif p_T1 < 0.05 \\
                    and J_obs > med_null1:
                verdict = \\
                    'headset_pathway_qwen'
            elif p_T1 < 0.05 \\
                    and J_obs < med_null1:
                verdict = \\
                    'headset_anti_aligned_qwen'
            else:
                verdict = \\
                    'headset_relational_qwen'
            T2a = {
                'n_logic': nL,
                'n_sham': nS,
                'n_valid': len(valid_idx),
                'J_obs': round(J_obs, 6),
                'p_T1': round(p_T1, 6),
                'null1_med':
                    round(med_null1, 6),
                'null_expect':
                    round(2.0 / 14.0, 6),
                'sets_per_tag': {
                    tags[i]:
                        sorted(sets_d[i])
                    for i in valid_idx},
                'head_top8_per_tag':
                    [None if not np.isfinite(v)
                     else round(float(v), 4)
                     for v in head_arr],
                'mlp_top32_per_tag':
                    [None if not np.isfinite(v)
                     else round(float(v), 4)
                     for v in mlp_arr],
                'nperm': NPERM,
                'gates_ok': gates_ok}

            T2b = {
                'J_deep_mid':
                    round(J2_obs, 6)
                    if np.isfinite(J2_obs)
                    else None,
                'p_T2': round(p_T2, 6)
                if np.isfinite(p_T2)
                else None,
                'n_pairs2': len(pairs2),
                'group7_count_obs': K_obs,
                'group7_expect': round(
                    4.0 * TOPH / NHQ
                    * len(valid_idx), 3),
                'p_T3': round(p_T3, 6),
                'spearman_dd_ph': [
                    round(float(s), 4)
                    if np.isfinite(s)
                    else None
                    for s in sp_list],
                'med_spearman_dd_ph':
                    round(rho4, 6),
                'p_T4': round(p_T4, 6),
                'ldp_layer_per_tag':
                    ldp_layer_list,
                'note': 'T2 deep-vs-mid same-tag '
                        'set Jaccard; T3 GQA '
                        'q28-31 enrichment; T4 '
                        'dd vs 3027 consumption '
                        'profile (exploratory)'}

            T2c = {
                'med_js_sham': round(
                    float(np.median(js_sham_all)),
                    6) if js_sham_all else None,
                'a41_ident_med': round(
                    float(np.median(ident_list)),
                    6) if ident_list else None,
                'note': 'sham chain self-js '
                        'calibration; ident = '
                        'attribution identity '
                        'metric (3022 machine)'}

'''
t = t[:s0] + NEW + t[s1:]

# ---- 12. anchor_all_ok conjunction ----
rep("            and a41_ok),",
    "            and a41_ok\n"
    "            and a42_ok and a43_ok\n"
    "            and a44_ok and a45_ok\n"
    "            and a46_ok),")

# ---- 13. anchors dict ----
rep("        'a41_s_relay_diff': a41_diff,",
    "        'a41_s_relay_diff': a41_diff,\n"
    "        'a42_headshare_recompute': a42_diff,\n"
    "        'a43_E_matrix_3032': a43_diff,\n"
    "        'a44_ldp_recompute': a44_diff,\n"
    "        'a45_p3027_sanity': a45_max,\n"
    "        'a46_source_seals': a46_ok,")

# ---- 14. npz save additions ----
rep("        'decode': np.array(\n"
    "            [d if d is not None else ''\n"
    "             for d in dec_list], dtype=object),\n"
    "    }",
    "        'decode': np.array(\n"
    "            [d if d is not None else ''\n"
    "             for d in dec_list], dtype=object),\n"
    "        'dd_deep': DD,\n"
    "        'dd_mid': DDM,\n"
    "        'J_obs': np.float64(J_obs),\n"
    "        'p_T1': np.float64(p_T1),\n"
    "        'group7_count_obs': np.int64(K_obs),\n"
    "        'p_T3': np.float64(p_T3),\n"
    "        'med_spearman_dd_ph': np.float64(rho4),\n"
    "        'p_T4': np.float64(p_T4),\n"
    "    }")

io.open(DST, 'w', encoding='utf-8').write(t)
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\phase3034_patch.txt', 'w',
        encoding='utf-8').write(
    'patched ok %d -> %d chars'
    % (n0, len(t)))
print('patched ok')
