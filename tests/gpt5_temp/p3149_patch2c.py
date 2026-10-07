# -*- coding: utf-8 -*-
"""p3149 patch2c: X2 rewrite + drop H/U +
V2 bit loop. Idempotent."""
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


NEW_X2 = """# ================================================================
# PART X2: tail d1 pair (bit + T2/L rows)
# ================================================================
log('== PART X2: tail d1 pair ==')
TAIL_DOSES = (1.0,)
for dd in TAIL_DOSES:
    _run_coord_trial('s2_tailpos_d%g' % dd,
                     TAIL25, 1, dd, rows_A1,
                     base12_A1,
                     save_gens=True)
    _run_coord_trial('s2_tailneg_d%g' % dd,
                     TAIL25, -1, dd, rows_A1,
                     base12_A1)
if not SMOKE:
    for tn in ('s2_tailpos_d1',
               's2_tailneg_d1'):
        got = E_res[tn]['chg']
        m = abs(got - BIT8[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': BIT8[tn],
            'match': bool(m)}
    log('X2 replay: +2 bit anchors (total '
        '%d/%d)' % (n_bit_all,
                    N_BIT_TOT))
# flip-row extraction (T2/L inputs)
fs_p1 = fstep_store['s2_tailpos_d1']
fs_n1 = fstep_store['s2_tailneg_d1']
flp_rows = [j for j in range(NCAP)
            if fs_p1[j] >= 0]
fln_rows = [j for j in range(NCAP)
            if fs_n1[j] >= 0]
log('X2: pos_d1 flips n=%d (fs %s); '
    'neg_d1 flips n=%d'
    % (len(flp_rows),
       [int(fs_p1[j]) for j in flp_rows],
       len(fln_rows)))
if not SMOKE:
    assert len(flp_rows) == \\
        FLIP_P1_48['n'], len(flp_rows)
    assert float(np.median(
        [int(fs_p1[j])
         for j in flp_rows])) == \\
        FLIP_P1_48['med']
"""

# 1) X2 block replace (slice)
iX1 = s.index(
    '# ================================================================\n'
    '# PART X2: tail sign-crossing mid-dose fill')
iX2 = s.index(
    '# ================================================================\n'
    '# PART H: top-2 source write-side identity')
s = s[:iX1] + NEW_X2 + '\n\n' + s[iX2:]
print('OK X2 block')

# 2) drop PART H (slice to PART U head)
iH1 = s.index(
    '# ================================================================\n'
    '# PART H: top-2 source write-side identity')
iH2 = s.index(
    '# ================================================================\n'
    '# PART U: unembed-cancel causality')
s = s[:iH1] + s[iH2:]
print('OK H drop')

# 3) drop PART U (slice to PART V2 head)
iU1 = s.index(
    '# ================================================================\n'
    '# PART U: unembed-cancel causality')
iU2 = s.index(
    '# ================================================================\n'
    '# PART V2: v1 perturbation symmetry')
s = s[:iU1] + s[iU2:]
print('OK U drop')

# 4) V2 bit loop
rep("""if not SMOKE:
    for tn, wv in WANT_V2.items():
        got = E_res[tn]['chg']
        m = abs(got - wv) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': wv,
            'match': bool(m)}
    log('V2 replay: +2 bit anchors (total '
        '%d/%d)' % (n_bit_all, N_BIT_TOT))""",
    """if not SMOKE:
    tn = 'v2_clip_a025'
    got = E_res[tn]['chg']
    m = abs(got - BIT8[tn]) < 1e-9
    n_bit_all += int(m)
    bit_anchors[tn] = {
        'got': got, 'want': BIT8[tn],
        'match': bool(m)}
    log('V2 replay: +1 bit anchor (total '
        '%d/%d)' % (n_bit_all, N_BIT_TOT))""",
    'V2 bit')

io.open(FP, 'w', encoding='utf-8',
        newline='\n').write(s)
print('PATCH2C_DONE')
