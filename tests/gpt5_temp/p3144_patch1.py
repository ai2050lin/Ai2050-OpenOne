# -*- coding: utf-8 -*-
"""Patch p3144 script: reorder B1/B2,
fix chunked logit, fix PART E argmax,
fix PART A bit count, drop dead lines."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\glm5\phase3144_omega_p142_'
      r'readouttraj_co36sign_unembed_'
      r'd19resid.py')
txt = io.open(FP, encoding='utf-8').read()

# ---- locate segment boundaries -------
M1 = '# ================================================================\n# PART B1: readout trajectory captures'
M2 = '# ================================================================\n# PART B2: generation bit anchors + token'
M3 = '# ================================================================\n# PART B3: teacher-forced logit'
i1 = txt.index(M1)
i2 = txt.index(M2)
i3 = txt.index(M3)
seg_before = txt[:i1]
seg_b1 = txt[i1:i2]
seg_b2 = txt[i2:i3]
seg_after = txt[i3:]
assert 'PART B1' in seg_b1 and 'PART B2' in seg_b2

# ---- NEW B1: gen anchors + sign + tokcls
NEW_B1 = '''# ================================================================
# PART B1: generation bit anchors + pc1
# sign resolution + interfering-token
# classification
# ================================================================
log('== PART B1: gen anchors + sign ==')
v1 = PC[29]['V'][0]
mn29 = PC[29]['mnorm']
dv_pc1_pos = np.tile(
    (v1 * mn29 * 2.0)[None, :],
    (NCAP, 1)).astype(np.float32)
dv_pc1_neg = -dv_pc1_pos
dv_pc1 = dv_pc1_pos
dv_dv29 = (dvec[29][:NCAP] * 2.0) \\
    .astype(np.float32)
E_res = {}


def _run_vec_trial(tname, il, dv_batch,
                   scale, mode, rows,
                   base12):
    _K = CK['data'].get(tname)
    if _K is not None:
        E_res[tname] = _K['res']
        log('%s RESUMED chg=%.4f'
            % (tname, E_res[tname]['chg']))
        return
    gen_l = []
    for b0 in range(0, len(rows),
                    GEN_BATCH):
        batch = rows[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj_vec=[(il,
                      dv_batch[b0:b0
                               + len(batch)],
                      scale, mode)]))
    chg_l, first_l, _fs = trial_metrics(
        gen_l, base12)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l),
                    'gens': gen_l}
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': E_res[tname]})


_run_vec_trial('b_pc1_l29_d2.0', 29,
               dv_pc1_pos, 1.0, 'allstep',
               rows_scan, base12_P)
pc1_sign = None
if not SMOKE and abs(
        E_res['b_pc1_l29_d2.0']['chg']
        - PC1_D2) >= 1e-9:
    _run_vec_trial('b_pc1neg_l29_d2.0',
                   29, dv_pc1_neg, 1.0,
                   'allstep', rows_scan,
                   base12_P)
    if abs(E_res['b_pc1neg_l29_d2.0']
           ['chg'] - PC1_D2) < 1e-9:
        pc1_sign = -1
    else:
        raise AssertionError(
            'pc1 sign unresolved: pos '
            '%.6f neg %.6f want %.6f'
            % (E_res['b_pc1_l29_d2.0']
               ['chg'],
               E_res['b_pc1neg_l29_d2.0']
               ['chg'], PC1_D2))
else:
    if not SMOKE:
        pc1_sign = 1
if pc1_sign == -1:
    dv_pc1 = dv_pc1_neg
    log('B1 pc1 sign resolved: -1')
elif pc1_sign == 1:
    log('B1 pc1 sign resolved: +1')
_run_vec_trial('b_dvec29_l29_d2.0', 29,
               dv_dv29, 1.0, 'allstep',
               rows_scan, base12_P)
_KJ = CK['data'].get('b_joint_l29_d2.0')
if _KJ is None:
    gen_l = []
    for b0 in range(0, NCAP, GEN_BATCH):
        batch = rows_scan[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj_vec=[(29, dv_pc1[b0:b0
                                + len(batch)],
                      1.0, 'allstep'),
                     (29, dv_dv29[b0:b0
                                  + len(batch)],
                      1.0, 'allstep')]))
    chg_l, first_l, _fs = trial_metrics(
        gen_l, base12_P)
    E_res['b_joint_l29_d2.0'] = {
        'chg': chg_l, 'first': int(first_l),
        'gens': gen_l}
    log('b_joint_l29_d2.0: chg=%.4f '
        'first=%d' % (chg_l, first_l))
    ck_save('b_joint_l29_d2.0', {
        'res': E_res['b_joint_l29_d2.0']})
else:
    E_res['b_joint_l29_d2.0'] = _KJ['res']
    log('b_joint_l29_d2.0 RESUMED '
        'chg=%.4f'
        % E_res['b_joint_l29_d2.0']['chg'])
n_bitB = 0
bit_anchors_B = {}
if not SMOKE:
    for tn, v in (('b_pc1_l29_d2.0',
                   PC1_D2),
                  ('b_dvec29_l29_d2.0',
                   DVEC29_D2),
                  ('b_joint_l29_d2.0',
                   JOINT_D2)):
        got = E_res[tn]['chg']
        m = abs(got - v) < 1e-9
        n_bitB += int(m)
        bit_anchors_B[tn] = {
            'got': got, 'want': v,
            'match': bool(m)}
    log('B1 3142 repro: %d/3 bit-match'
        % n_bitB)
else:
    log('B1 3142 repro SKIPPED (smoke)')


def tok_class(tid):
    if tid in (YES_G, NO_G):
        return 'answer'
    if tid == DOT_G:
        return 'format'
    return 'other'


tokcls = {}
for tn in ('b_pc1_l29_d2.0',
           'b_dvec29_l29_d2.0',
           'b_joint_l29_d2.0'):
    gens = E_res[tn]['gens']
    counts = {'answer': 0, 'format': 0,
              'other': 0, 'none': 0}
    tstars = []
    for j in range(len(gens)):
        g12 = pad12(gens[j])
        b12 = base12_P[j]
        tstar = -1
        for t in range(N_NEW):
            if g12[t] != b12[t]:
                tstar = t
                break
        if tstar < 0:
            counts['none'] += 1
            continue
        tstars.append(tstar)
        counts[tok_class(g12[tstar])] += 1
    nint = sum(counts[k] for k in
               ('answer', 'format',
                'other'))
    tokcls[tn] = {
        'counts': counts,
        'answer_frac':
            counts['answer'] / max(nint, 1),
        'format_frac':
            counts['format'] / max(nint, 1),
        'med_tstar':
            float(np.median(tstars))
            if tstars else -1.0}
    log('B1 tokcls %s: %s answer %.3f '
        'format %.3f med_t* %s'
        % (tn, json.dumps(counts),
           tokcls[tn]['answer_frac'],
           tokcls[tn]['format_frac'],
           tokcls[tn]['med_tstar']))
_af = tokcls['b_pc1_l29_d2.0']['answer_frac']
_ff = tokcls['b_pc1_l29_d2.0']['format_frac']
if _af > FORMAT_GATE:
    pc1_channel = 'pc1_channel_answer'
elif _ff > FORMAT_GATE:
    pc1_channel = 'pc1_channel_format'
else:
    pc1_channel = 'pc1_channel_mixed'
log('B1-GATE: pc1 interfering-token '
    'channel -> %s' % pc1_channel)

'''

# ---- NEW B2: trajectory (from old B1
# with vector defs removed) ------------
NEW_B2 = '''# ================================================================
# PART B2: readout trajectory captures
# (pc1/dv29/joint @L29, layers 29-39)
# ================================================================
log('== PART B2: readout trajectory ==')
_traj_specs = {
    'pc1': [(29, dv_pc1, 1.0)],
    'dv29': [(29, dv_dv29, 1.0)],
    'joint': [(29, dv_pc1, 1.0),
              (29, dv_dv29, 1.0)]}
HT = {}
for cn, spec in _traj_specs.items():
    _K = CK['data'].get('traj_%s' % cn)
    if _K is not None:
        HT[cn] = {int(k):
                  v.astype(np.float32)
                  for k, v in
                  _K['h'].items()}
        log('traj %s RESUMED' % cn)
        continue
    HT[cn] = capture_states_inject2(
        rows_cap, list(TRAJ_L), spec)
    log('traj %s done (%d layers)'
        % (cn, len(TRAJ_L)))
    ck_save('traj_%s' % cn, {
        'h': {str(l): HT[cn][l]
              .astype(np.float16)
              for l in TRAJ_L}})
traj = {}
for cn in ('pc1', 'dv29', 'joint'):
    traj[cn] = {}
    for r in TRAJ_L:
        dh = (HT[cn][r].astype(np.float64)
              - base_states[r]
              .astype(np.float64))
        traj[cn][r] = {
            'wdn': float(np.median(
                dh @ w_dn_g.astype(
                    np.float64))),
            'v1': float(np.median(
                dh @ v1.astype(
                    np.float64))),
            'norm': float(np.median(
                np.linalg.norm(dh,
                               axis=1)))}
gap = {r: traj['dv29'][r]['wdn']
       - traj['joint'][r]['wdn']
       for r in TRAJ_L}
max_gap = max(gap.values())
blk_layer = max(gap, key=lambda r: gap[r])
traj_tag = ('traj_located_l%02d'
            % blk_layer
            if max_gap > GAP_TOL
            else 'traj_gradual')
log('B2-SOFT: w_dn traj pc1 %s' %
    json.dumps({str(r): round(
        traj['pc1'][r]['wdn'], 3)
        for r in TRAJ_L}))
log('B2-SOFT: w_dn traj dv29 %s' %
    json.dumps({str(r): round(
        traj['dv29'][r]['wdn'], 3)
        for r in TRAJ_L}))
log('B2-SOFT: w_dn traj joint %s' %
    json.dumps({str(r): round(
        traj['joint'][r]['wdn'], 3)
        for r in TRAJ_L}))
log('B2-SOFT: v1 traj joint %s' %
    json.dumps({str(r): round(
        traj['joint'][r]['v1'], 3)
        for r in TRAJ_L}))
log('B2-GATE: gap(dv29-joint) max %.4f at '
    'L%02d (tol %.2f) -> %s'
    % (max_gap, blk_layer, GAP_TOL,
       traj_tag))

'''

txt = seg_before + NEW_B1 + NEW_B2 + seg_after

# ---- fix chunked_toplogit ------------
OLD_FN = '''def chunked_toplogit(DH, topk=5):
    """DH (n,4096) @ WUG.T in chunks ->
    (n,V) argmax + topk ids per row."""
    n = DH.shape[0]
    top_ids = np.zeros((n, topk),
                       dtype=np.int64)
    top_vals = np.zeros((n, topk),
                        dtype=np.float64)
    am = np.zeros(n, dtype=np.int64)
    W = WUG.detach()
    CH = 16384
    for s0 in range(0, VOCAB, CH):
        s1 = min(s0 + CH, VOCAB)
        Wc = W[s0:s1].to(torch.float32) \\
            .cpu().numpy() \\
            .astype(np.float64)
        LG = DH @ Wc.T
        am_part = np.argmax(LG, axis=1)
        am = np.where(
            (LG[np.arange(n), am_part]
             > LG[np.arange(n), am])
            | (s0 == 0),
            s0 + am_part, am)
        for t in range(topk):
            if t == 0:
                cand = LG
            ids_t = np.argsort(-cand,
                               axis=1)[:, :topk]
            vals_t = np.take_along_axis(
                cand, ids_t, axis=1)
            if s0 == 0:
                top_ids[:, :ids_t.shape[1]] \\
                    = ids_t + s0
                top_vals[:, :ids_t.shape[1]] \\
                    = vals_t
            else:
                allv = np.concatenate(
                    [top_vals, vals_t],
                    axis=1)
                alli = np.concatenate(
                    [top_ids,
                     ids_t + s0], axis=1)
                sel = np.argsort(
                    -allv, axis=1)[:, :topk]
                top_vals = np.take_along_axis(
                    allv, sel, axis=1)
                top_ids = np.take_along_axis(
                    alli, sel, axis=1)
    return am, top_ids, top_vals'''
NEW_FN = '''def chunked_logit(DH, topk=5):
    """DH (n,4096) @ WUG.T in chunks ->
    per-row global argmax id + topk
    (ids, logits)."""
    n = DH.shape[0]
    W = WUG.detach()
    best_val = np.full(n, -np.inf)
    am = np.zeros(n, dtype=np.int64)
    top_ids = np.zeros((n, topk),
                       dtype=np.int64)
    top_vals = np.full((n, topk),
                       -np.inf)
    CH = 16384
    for s0 in range(0, VOCAB, CH):
        s1 = min(s0 + CH, VOCAB)
        Wc = W[s0:s1].to(torch.float32) \\
            .cpu().numpy() \\
            .astype(np.float64)
        LG = DH @ Wc.T
        mx = LG.max(axis=1)
        upd = mx > best_val
        am[upd] = s0 + np.argmax(
            LG, axis=1)[upd]
        best_val[upd] = mx[upd]
        pids = np.argsort(
            -LG, axis=1)[:, :topk]
        pvals = np.take_along_axis(
            LG, pids, axis=1)
        allv = np.concatenate(
            [top_vals, pvals], axis=1)
        alli = np.concatenate(
            [top_ids, pids + s0], axis=1)
        sel = np.argsort(
            -allv, axis=1)[:, :topk]
        top_vals = np.take_along_axis(
            allv, sel, axis=1)
        top_ids = np.take_along_axis(
            alli, sel, axis=1)
    return am, top_ids, top_vals'''
assert txt.count(OLD_FN) == 1
txt = txt.replace(OLD_FN, NEW_FN)
assert txt.count('chunked_toplogit(') == 1
txt = txt.replace('am39, top5_ids, '
                  'top5_vals = \\\n    '
                  'chunked_toplogit(dh39, 5)',
                  'am39, top5_ids, '
                  'top5_vals = \\\n    '
                  'chunked_logit(dh39, 5)')
assert 'chunked_toplogit' not in txt

# ---- PART A bit count: include newI --
OLD_A = '''nb43 = 0
for v in pr43['bit_anchors_3142'].values():
    nb43 += int(v['match'])
for v in pt43['bit_anchors'].values():
    nb43 += int(v['match'])
assert 'repro_bit_5' in V43 and nb43 == 4'''
NEW_A = '''nb43 = 0
for v in pr43['bit_anchors_3142'].values():
    nb43 += int(v['match'])
for v in pt43['bit_anchors'].values():
    nb43 += int(v['match'])
for v in pn43['bit_anchors'].values():
    nb43 += int(v['match'])
assert 'repro_bit_5' in V43 and nb43 == 5'''
assert txt.count(OLD_A) == 1
txt = txt.replace(OLD_A, NEW_A)

# ---- PART E argmax via chunked_logit --
OLD_E = '''hit_top1 = 0
for i, pk in enumerate(pks):
    (s, o) = (int(v)
              for v in pk.split('_'))
    self_cos[i] = float(ip17[i] @ E_vec[s])
    obj_cos[i] = float(ip17[i] @ E_vec[o])
    cands = [x for x in range(ENTS_N)
             if x != s]
    pick = [cands[rng_e.randrange(
        len(cands))]
        for _ in range(N_RAND_ENT)]
    rand_cos[i] = float(np.median(
        [ip17[i] @ E_vec[x]
         for x in pick]))
    top1 = int(np.argmax(ip17[i] @ (
        WUG.to(torch.float32).cpu().numpy()
        .astype(np.float64).T)))
    if top1 in ent_id_set:
        hit_top1 += 1'''
NEW_E = '''ip17_top1, _, _ = chunked_logit(
    ip17, 1)
hit_top1 = 0
for i, pk in enumerate(pks):
    (s, o) = (int(v)
              for v in pk.split('_'))
    self_cos[i] = float(ip17[i] @ E_vec[s])
    obj_cos[i] = float(ip17[i] @ E_vec[o])
    cands = [x for x in range(ENTS_N)
             if x != s]
    pick = [cands[rng_e.randrange(
        len(cands))]
        for _ in range(N_RAND_ENT)]
    rand_cos[i] = float(np.median(
        [ip17[i] @ E_vec[x]
         for x in pick]))
    top1 = int(ip17_top1[i])
    if top1 in ent_id_set:
        hit_top1 += 1'''
assert txt.count(OLD_E) == 1
txt = txt.replace(OLD_E, NEW_E)

# ---- co36/co36_rank set assert -------
OLD_F = '''assert len(CO_SETS['co36']) == 50
led = json.load(io.open('''
NEW_F = '''assert len(CO_SETS['co36']) == 50
assert set(G_ORDER.tolist()) == \\
    set(CO_SETS['co36'].tolist()), \\
    'co36_rank set != co36 set'
led = json.load(io.open('''
assert txt.count(OLD_F) == 1
txt = txt.replace(OLD_F, NEW_F)

io.open(FP, 'w', encoding='utf-8').write(txt)
# verify on disk
chk = io.open(FP, encoding='utf-8').read()
assert 'PART B1: generation bit anchors' \
    in chk
assert 'PART B2: readout trajectory' in chk
assert 'PART B3: teacher-forced' in chk
assert 'if False else None' not in chk
assert 'chunked_logit' in chk
assert txt == chk
print('PATCH OK %d chars' % len(chk))
