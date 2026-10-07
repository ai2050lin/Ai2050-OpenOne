# -*- coding: utf-8 -*-
"""p3149 patch2b: N trim + X2 rewrite +
drop H/U + V2 bit loop. Idempotent."""
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


# 1) N header + pcres trials drop
rep("""# ================================================================
# PART N: pcres d1 replays + neg mechanism
# ================================================================
log('== PART N: pcres replay + neg ==')
dv100 = (pc1_res
         / max(np.linalg.norm(pc1_res),
               1e-12) * pnorm38 * 1.0)     .astype(np.float32)
dv100_tile = np.tile(dv100[None, :],
                     (NCAP, 1))
_run_vec_trial('e_pcres38_d1.0_pos', 38,
               dv100_tile, 1.0, 'allstep',
               rows_scan, base12_P)
_run_vec_trial('e_pcres38_d1.0_neg', 38,
               -dv100_tile, 1.0, 'allstep',
               rows_scan, base12_P)
if not SMOKE:
    for tn in ('e_pcres38_d1.0_pos',
               'e_pcres38_d1.0_neg'):
        got = E_res[tn]['chg']
        m = abs(got - BIT11[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': BIT11[tn],
            'match': bool(m)}
    log('N replay: +2 bit anchors (total '
        '%d/%d)' % (n_bit_all,
                    N_BIT_TOT))
# N1: neg_d0.5 replay (soft anchor
# 0.2421875 from 3146)""",
    """# ================================================================
# PART N: neg_d0.5 bit + flip rows
# ================================================================
log('== PART N: neg_d0.5 + flips ==')
# N1: neg_d0.5 replay (bit anchor
# 0.2421875 from 3146)""",
    'N header')

# 2) N bit record
rep("""n_repro_ok = abs(chg_n - 0.2421875) < 1e-9
log('N1 soft anchor vs 3146: %s'
    % n_repro_ok)""",
    """n_repro_ok = abs(chg_n - 0.2421875) < 1e-9
if not SMOKE:
    got = chg_n
    m = abs(got - BIT8['n_neg_d0.5']) \\
        < 1e-9
    n_bit_all += int(m)
    bit_anchors['n_neg_d0.5'] = {
        'got': got,
        'want': BIT8['n_neg_d0.5'],
        'match': bool(m)}
    log('N replay: +1 bit anchor (total '
        '%d/%d)' % (n_bit_all,
                    N_BIT_TOT))
log('N1 anchor vs 3146: %s'
    % n_repro_ok)""",
    'N bit')

# 3) N2 frozen list rename
rep("""if not SMOKE:
    assert flip_rows == N47['flip_rows'], \\
        flip_rows
    assert [int(fstep_store['n_neg_d0.5'][j])
            for j in flip_rows] == \\
        N47['fsteps'], 'fsteps drift'
    log('N2 frozen-list match ok (3147)')""",
    """if not SMOKE:
    assert flip_rows == \\
        N47FROZEN['rows'], flip_rows
    assert [int(fstep_store['n_neg_d0.5'][j])
            for j in flip_rows] == \\
        N47FROZEN['fs'], 'fsteps drift'
    log('N2 frozen-list match ok (3147)')""",
    'N2 frozen')

io.open(FP, 'w', encoding='utf-8',
        newline='\n').write(s)
print('PATCH2B1_DONE')
