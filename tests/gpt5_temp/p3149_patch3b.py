# -*- coding: utf-8 -*-
"""p3149 patch3b: verdict tags + result
parts + npz rewrite. Idempotent."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\glm5\phase3149_omega_p147_'
      r'carrier_dlogit_poslate_kdose_'
      r'v3amp.py')
s = io.open(FP, encoding='utf-8').read()


def rep(old, new, tag):
    global s
    n = s.count(old)
    assert n == 1, (tag, n)
    s = s.replace(old, new)
    print('OK', tag)


# 1) verdict tags
rep("""tags = ['a_3147_ok']""",
    """tags = ['a_3148_ok']""",
    'verdict head')
rep("""tags.append(xcross_tag)
tags.append(fstep_tag)
tags.append(h_pair)
tags.append(h_head)
tags.append(h_overlap)
tags.append(u_tag)
tags.append(v_sym)
tags.append('xphase_ok' if xphase_ok""",
    """tags.append(t2_tag)
tags.append(l_tag)
tags.append(kx_tag)
tags.append(v3_tag)
tags.append('xphase_ok' if xphase_ok""",
    'verdict tags')

# 2) result parts (slice replace)
iR1 = s.index("    'part_x2': {")
iR2 = s.index("RF = os.path.join(OUT, "
              "'result.json')")
NEW_PARTS = """    'part_t2': {
        'stats': t2_stats,
        'common_neg': common_tok['neg'],
        'common_pos': common_tok['pos'],
        'shared': [int(t)
                   for t in shared],
        't2_tag': t2_tag},
    'part_l': {
        'flp_rows': [int(j)
                     for j in flp_rows],
        'fsteps': [fs_j[j]
                   for j in flp_rows],
        'kmin_cuts': [int(L_cuts[str(j)])
                      for j in flp_rows],
        'wdn_early': wdn_early,
        'wdn_late': wdn_late,
        'd131_early': d131_early,
        'd131_late': d131_late,
        'frac_fixed': frac_fixed,
        'l_tag': l_tag},
    'part_k': {
        'kx': {'top%d_d%g' % (kk, dd):
               float(v)
               for (kk, dd), v
               in sorted(kx.items())},
        'best_pair': [int(best_pair[0]),
                      float(best_pair[1]),
                      float(best_pair[2])],
        'best_gap': float(best_gap),
        'kx_tag': kx_tag},
    'part_v3': {
        'chg_ampan005': _chg(
            'v3_ampan_a005'),
        'chg_ampan010': _chg(
            'v3_ampan_a010'),
        'chg_ampan015': _chg(
            'v3_ampan_a015'),
        'sym_curve': {k: float(v)
                      for k, v
                      in sym_curve.items()},
        'v3_tag': v3_tag},
    }
"""
s = s[:iR1] + NEW_PARTS + s[iR2:]
print('OK result parts')

# 3) npz rewrite
rep("""npz_out = {
    'dvec19_sha': np.array([dvec19_sha]),
    'head_contrib': head_contrib,
    'order_head':
        order_head.astype(np.int64),
    'tail_doses': np.array(TAIL_DOSES),
    'pos_curve': np.array(
        [posc[d] for d in TAIL_DOSES]),
    'neg_curve': np.array(
        [negc[d] for d in TAIL_DOSES]),
    'sym_ratios': np.array([sym25,
                            sym50]),
    'pc1_res': pc1_res.astype(
        np.float32)}
np.savez(os.path.join(OUT,
                      'p146_readout.npz'),
         **npz_out)""",
    """npz_out = {
    'dvec19_sha': np.array([dvec19_sha]),
    't2_common_neg': np.array(
        [int(t) for t in
         common_tok['neg']],
        dtype=np.int64),
    't2_common_pos': np.array(
        [int(t) for t in
         common_tok['pos']],
        dtype=np.int64),
    'l_traj_wdn': np.array(
        [[st['wdn']
          for st in L_traj[str(j)]]
         for j in flp_rows],
        dtype=np.float64),
    'l_traj_d131': np.array(
        [[st['d131']
          for st in L_traj[str(j)]]
         for j in flp_rows],
        dtype=np.float64),
    'l_kmin': np.array(
        [int(L_cuts[str(j)])
         for j in flp_rows],
        dtype=np.int64),
    'kx_curve': np.array(
        [float(kx[(kk, dd)])
         for kk in K_SUBS
         for dd in K_DOSES],
        dtype=np.float64),
    'sym_curve': np.array(
        [float(sym_curve[k]) for k in
         ('0.05', '0.1', '0.15', '0.25',
          '0.5')],
        dtype=np.float64),
    'pc1_res': pc1_res.astype(
        np.float32)}
np.savez(os.path.join(OUT,
                      'p147_readout.npz'),
         **npz_out)""",
    'npz')

io.open(FP, 'w', encoding='utf-8',
        newline='\n').write(s)
print('PATCH3B_DONE')
