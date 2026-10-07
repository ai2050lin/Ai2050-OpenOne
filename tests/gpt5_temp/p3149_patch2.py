# -*- coding: utf-8 -*-
"""p3149 patch2: PART A rewrite + helper
extensions + trim S/X/T/N/X2 + drop H/U +
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


NEW_PARTA = """# ================================================================
# PART A: 3148 link asserts
# ================================================================
log('== PART A: link asserts ==')
res48 = json.load(io.open(
    D48 + r'\\result.json',
    encoding='utf-8'))
assert res48['smoke'] is False
V48g = res48['verdict']
assert V48g == EXP_V48, V48g
raw48 = io.open(
    D48 + r'\\result.json', 'rb').read()
sha48 = hashlib.sha256(
    raw48).hexdigest()[:8]
assert sha48 == RES48_SHA, sha48
assert str(res48['seal_sha8']) == SEAL48
pa48 = res48['part_a']
assert abs(float(pa48['xphase_P'])
           - XPHASE48) < 1e-12
assert abs(float(pa48['xphase_A1'])
           - XPHASE48) < 1e-12
px248 = res48['part_x2']
for k, v in (('1', 'pos_1'),
             ('2', 'pos_2'),
             ('2.5', 'pos_25'),
             ('3', 'pos_3'),
             ('3.5', 'pos_35'),
             ('4', 'pos_4')):
    assert abs(float(
        px248['pos_curve'][k])
        - X248[v]) < 1e-9, v
for k, v in (('1', 'neg_1'),
             ('2', 'neg_2'),
             ('2.5', 'neg_25'),
             ('3', 'neg_3'),
             ('3.5', 'neg_35'),
             ('4', 'neg_4')):
    assert abs(float(
        px248['neg_curve'][k])
        - X248[v]) < 1e-9, v
assert px248['xcross_tag'] == \\
    'tail_xcross_located'
assert [float(v) for v
        in px248['xcross_interval']] == \\
    XCROSS48
fp148 = px248['flip_pos_d1']
assert int(fp148['n']) == \\
    FLIP_P1_48['n']
assert abs(float(fp148['med_fstep'])
           - FLIP_P1_48['med']) < 1e-9
fn448 = px248['flip_neg_d4']
assert int(fn448['n']) == \\
    FLIP_N4_48['n']
assert abs(float(fn448['med_fstep'])
           - FLIP_N4_48['med']) < 1e-9
ph48 = res48['part_h']
assert abs(float(ph48['solo1_d2'])
           - H48['solo1']) < 1e-9
assert abs(float(ph48['solo2_d2'])
           - H48['solo2']) < 1e-9
assert abs(float(ph48['pair_d2'])
           - H48['pair']) < 1e-9
assert abs(float(ph48['resid_pair'])
           - H48['resid']) < 1e-9
assert abs(float(ph48['frac_top2'])
           - H48['frac_top2']) < 1e-9
assert int(ph48['rank_dv19']['2530']) == \\
    H48['rank19_2530']
pu48 = res48['part_u']
for k in ('only', 'cancel', 'rand',
          'wdn'):
    assert abs(float(pu48['chg_' + k])
               - U48[k]) < 1e-9, k
pv248 = res48['part_v2']
for k in ('amp025', 'amp050',
          'clip025', 'clip050'):
    assert abs(float(pv248['chg_' + k])
               - V248[k]) < 1e-9, k
log('A hard asserts ok (3148 sha8 %s '
    'seal %s: xphase/part_x2 curves/'
    'xcross/flip rows/part_h/part_u/'
    'part_v2 all verified)'
    % (sha48, SEAL48))
"""

# 1) PART A block replace (slice)
i1 = s.index(
    '# ================================================================\n'
    '# PART A: 3147 link asserts')
i2 = s.index('\n\n\n# p3148 patch1 applied')
s = s[:i1] + NEW_PARTA + s[i2:]
print('OK PART A block')

# 2) _run_coord_trial extension
rep("""def _run_coord_trial(tname, coords, sgn,
                     dsc, rows, base12,
                     il=D17_L):""",
    """def _run_coord_trial(tname, coords, sgn,
                     dsc, rows, base12,
                     il=D17_L, mode=0,
                     save_gens=False):""",
    'coord sig')
rep("""            inj=[(il, coords,
                  dsc * DELTA_L17, sgn,
                  0)]))""",
    """            inj=[(il, coords,
                  dsc * DELTA_L17, sgn,
                  mode)]))""",
    'coord mode')
rep("""    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {
        'res': E_res[tname],
        'fstep': fs.astype(np.int8)
        .tolist()})""",
    """    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    _rec = {'res': E_res[tname],
            'fstep': fs.astype(np.int8)
            .tolist()}
    if save_gens:
        _rec['res']['gens'] = gen_l
    ck_save(tname, _rec)""",
    'coord save')

# 3) _inj closure cut branch
rep("""                if _m == 'allstep' \\
                        or _st['step'] == 0 \\
                        or _st['step'] == _m:
                    o2[:, -1, _co] += _dv""",
    """                if _m == 'allstep' \\
                        or _st['step'] == 0 \\
                        or _st['step'] == _m \\
                        or (isinstance(_m, str)
                            and _m.startswith(
                                'cut')
                            and _st['step']
                            <= int(_m[3:])):
                    o2[:, -1, _co] += _dv""",
    'inj cut')

# 4) S trim
rep("""# 2 bit replays @d4 sgn-1 (3146, 2nd)
_run_coord_trial('d_tbot15_d4.0', TBOT15,
                 -1, 4.0, rows_A1,
                 base12_A1)
_run_coord_trial('d_ttop10_d4.0', TTOP10,
                 -1, 4.0, rows_A1,
                 base12_A1)
if not SMOKE:
    for tn in ('d_tbot15_d4.0',
               'd_ttop10_d4.0'):
        got = E_res[tn]['chg']
        m = abs(got - BIT11[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': BIT11[tn],
            'match': bool(m)}
    log('S replay: +2 bit anchors (total '
        '%d/%d)' % (n_bit_all,
                    N_BIT_TOT))
# 3148: tail sign curve moved to
# PART X2 (s2_* namespace, full
# d{1,2,2.5,3,3.5,4} both signs)""",
    """# 3149: tail d2..d4 curve dropped
# (xcross located in 3148); d1 pair
# retained in PART X2 (s2_* namespace)""",
    'S trim')

# 5) X trim
rep("""# 3 bit replays + low-dose fill
_run_coord_trial('d_co50ex_d1.0', co50ex,
                 -1, 1.0, rows_A1, base12_A1)
_run_coord_trial('d_co50ex_d2.0', co50ex,
                 -1, 2.0, rows_A1, base12_A1)
_run_coord_trial('d_co50ex_d4.0', co50ex,
                 -1, 4.0, rows_A1, base12_A1)
if not SMOKE:
    for tn in ('d_co50ex_d1.0',
               'd_co50ex_d2.0',
               'd_co50ex_d4.0'):
        got = E_res[tn]['chg']
        m = abs(got - BIT11[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got,
            'want': BIT11[tn],
            'match': bool(m)}
    log('X replay: +3 bit anchors '
        '(total %d/%d)'
        % (n_bit_all, N_BIT_TOT))
assert [int(c) for c in order_ex[:2]] == ORDER_EX2, list(order_ex[:2])
log('order_ex top2 == 3147 %s (enrichment desc)' % ORDER_EX2)""",
    """# 1 bit replay (K baseline)
_run_coord_trial('d_co50ex_d2.0', co50ex,
                 -1, 2.0, rows_A1, base12_A1)
if not SMOKE:
    tn = 'd_co50ex_d2.0'
    got = E_res[tn]['chg']
    m = abs(got - BIT8[tn]) < 1e-9
    n_bit_all += int(m)
    bit_anchors[tn] = {
        'got': got, 'want': BIT8[tn],
        'match': bool(m)}
    log('X replay: +1 bit anchor '
        '(total %d/%d)'
        % (n_bit_all, N_BIT_TOT))
assert [int(c) for c in order_ex[:2]] == ORDER_EX2, list(order_ex[:2])
log('order_ex top2 == 3148 link %s (enrichment desc)' % ORDER_EX2)""",
    'X trim')

# 6) T block drop
iT1 = s.index(
    '# ================================================================\n'
    '# PART T: pc1 instant replay (1 bit)')
iT2 = s.index(
    '# ================================================================\n'
    '# PART N: pcres d1 replays + neg mechanism')
s = s[:iT1] + s[iT2:]
print('OK T drop')

io.open(FP, 'w', encoding='utf-8',
        newline='\n').write(s)
print('PATCH2A_DONE')
