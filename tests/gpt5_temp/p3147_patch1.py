# -*- coding: utf-8 -*-
"""p3147 patch1: fix dead code, N1 ckpt,
missing pcres replays, double pad_batch."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\glm5\phase3147_omega_p145_'
      r'tbsym_co50exk_negmech_v1micro.py')
t = io.open(FP, encoding='utf-8').read()

# --- fix 1: remove b15p dead code ------
old1 = """b15n = BIT11['d_tbot15_d4.0']
b15p = _chg('s_tb15_pos_d2.0') \\
    if False else _chg('s_tb15pos_d4') \\
    if False else None
# bot15 pos @d4 not in grid; use neg d2
# vs pos d2 for sign, and full d4 curve
bp2 = _chg('s_tb15_pos_d2.0')"""
new1 = """b15n = BIT11['d_tbot15_d4.0']
# bot15 pos @d4 not in grid; use neg d2
# vs pos d2 for sign, and full d4 curve
bp2 = _chg('s_tb15_pos_d2.0')"""
c1 = t.count(old1)
assert c1 == 1, ('fix1', c1)
t = t.replace(old1, new1)

# --- fix 2: tailpos_d2 soft anchor ----
old2 = """log('S-GATE: TAIL25 sign curve pos %s vs '
    'neg %s -> %s'
    % (json.dumps({'1': round(p1, 4),
                   '2': round(p2, 4),
                   '4': round(p4t, 4)}),
       json.dumps({'1': round(n1t, 4),
                   '2': round(n2t, 4),
                   '4': round(n4t, 4)}),
       tail_sign))"""
new2 = """log('S-GATE: TAIL25 sign curve pos %s vs '
    'neg %s -> %s'
    % (json.dumps({'1': round(p1, 4),
                   '2': round(p2, 4),
                   '4': round(p4t, 4)}),
       json.dumps({'1': round(n1t, 4),
                   '2': round(n2t, 4),
                   '4': round(n4t, 4)}),
       tail_sign))
p2_want = 0.2578125
tailpos2_repro = abs(p2 - p2_want) < 1e-9
log('S-SOFT: s_tailpos_d2 %.4f vs 3145 '
    '0.2578 repro %s' % (p2,
                         tailpos2_repro))"""
c2 = t.count(old2)
assert c2 == 1, ('fix2', c2)
t = t.replace(old2, new2)

# --- fix 3: N1 via _run_vec_trial -----
old3 = """dv_n = (pc1_res
        / max(np.linalg.norm(pc1_res),
              1e-12) * pnorm38 * 0.5) \\
    .astype(np.float32)
dv_n_tile = np.tile(dv_n[None, :],
                    (NCAP, 1))
_gen_n = []
for b0 in range(0, NCAP, GEN_BATCH):
    batch = rows_scan[b0:b0 + GEN_BATCH]
    _gen_n.extend(gen_batch_g2(
        batch,
        inj_vec=[(38,
                  -dv_n_tile[b0:b0
                             + len(batch)],
                  1.0, 'allstep')]))
chg_n, first_n, fs_n = trial_metrics(
    _gen_n, base12_P)
E_res['n_neg_d0.5'] = {'chg': chg_n,
                       'first': int(first_n),
                       'gens': _gen_n}
fstep_store['n_neg_d0.5'] = fs_n
log('N1 n_neg_d0.5: chg=%.4f first=%d '
    '(3146 want 0.2422)'
    % (chg_n, first_n))"""
new3 = """dv_n = (pc1_res
        / max(np.linalg.norm(pc1_res),
              1e-12) * pnorm38 * 0.5) \\
    .astype(np.float32)
dv_n_tile = np.tile(dv_n[None, :],
                    (NCAP, 1))
_run_vec_trial('n_neg_d0.5', 38,
               -dv_n_tile, 1.0, 'allstep',
               rows_scan, base12_P,
               save_gens=True)
chg_n = E_res['n_neg_d0.5']['chg']
_gen_n = E_res['n_neg_d0.5']['gens']
fs_n = fstep_store['n_neg_d0.5']
log('N1 n_neg_d0.5 done (3146 want '
    '0.2422)')"""
c3 = t.count(old3)
assert c3 == 1, ('fix3', c3)
t = t.replace(old3, new3)

# --- fix 4: pcres d1.0 replays --------
old4 = """# ================================================================
# PART N: neg late-flip mechanism
# ================================================================
log('== PART N: neg late-flip ==')"""
new4 = """# ================================================================
# PART N: pcres d1 replays + neg mechanism
# ================================================================
log('== PART N: pcres replay + neg ==')
dv100 = (pc1_res
         / max(np.linalg.norm(pc1_res),
               1e-12) * pnorm38 * 1.0) \
    .astype(np.float32)
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
        '%d/11)' % n_bit_all)"""
c4 = t.count(old4)
assert c4 == 1, ('fix4', c4)
t = t.replace(old4, new4)

# --- fix 5: double pad_batch ----------
old5 = """        _proj_store.clear()
        _pad_batch(batch)
        ids_p, mask_p = _pad_batch(batch)"""
new5 = """        _proj_store.clear()
        ids_p, mask_p = _pad_batch(batch)"""
c5 = t.count(old5)
assert c5 == 1, ('fix5', c5)
t = t.replace(old5, new5)

# --- fix 6: neg_repro tag includes
# tailpos2 soft anchor in result -------
old6 = """        'tail_sub': tail_sub},"""
new6 = """        'tail_sub': tail_sub,
        'tailpos_d2_got': p2,
        'tailpos_d2_repro':
            bool(tailpos2_repro)},"""
c6 = t.count(old6)
assert c6 == 1, ('fix6', c6)
t = t.replace(old6, new6)

io.open(FP, 'w', encoding='utf-8').write(t)
chk = io.open(FP, encoding='utf-8').read()
assert "if False else" not in chk
assert 'e_pcres38_d1.0_pos' in chk
assert '_proj_store.clear()' in chk
assert chk.count('_pad_batch(batch)') >= 1
import py_compile
py_compile.compile(FP, doraise=True)
print('PATCH1 OK: 6 fixes applied')
