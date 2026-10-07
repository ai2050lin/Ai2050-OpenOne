# -*- coding: utf-8 -*-
"""p3149 patch3a: insert new PART T2/L/K/V3
after V-GATE2 log. Idempotent."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\glm5\phase3149_omega_p147_'
      r'carrier_dlogit_poslate_kdose_'
      r'v3amp.py')
s = io.open(FP, encoding='utf-8').read()

ANCHOR = """log('V-GATE2: amp025 %.4f vs clip025 '
    '%.4f (sym %.2f); amp050 %.4f vs '
    'clip050 %.4f (sym %.2f) -> %s'
    % (_chg('v2_amp_a025'),
       _chg('v2_clip_a025'), sym25,
       _chg('v2_amp_a050'),
       _chg('v2_clip_a050'), sym50,
       v_sym))
"""
assert s.count(ANCHOR) == 1, \
    s.count(ANCHOR)

NEW_PARTS = '''
# ================================================================
# PART T2: flip-carrier dlogit decomposition
# ================================================================
log('== PART T2: carrier dlogit ==')
WUG_T = WUG.detach().float()


def _dlogit_steps(row_ids, base_tok,
                  inj_tok, dv_row,
                  inj_coord, n_steps):
    """Per-step dlogit (inj - base) for
    steps 0..n_steps-1. dv_row: L38 vec
    injection or None. inj_coord:
    (coords, delta) L17 injection or
    None. Returns list of vocab np
    vectors (fp64)."""
    outs = []
    for side in range(2):
        toks = base_tok if side == 0 \\
            else inj_tok
        step_h = []
        for k in range(n_steps):
            feats = {39: None}
            hooks = []

            def _mk39(mod, inp, out,
                      _f=feats):
                o2 = out[0] \\
                    if isinstance(out,
                                  tuple) \\
                    else out
                _f[39] = o2[0, -1, :] \\
                    .detach()
                return None

            hooks.append(
                model_g.model.layers[39]
                .register_forward_hook(
                    _mk39))
            if side == 1:
                if dv_row is not None:
                    dv_t = torch.as_tensor(
                        dv_row,
                        dtype=torch.float32,
                        device='cuda') \\
                        .to(torch.bfloat16)

                    def _injv2(mod, inp,
                               out,
                               _dv=dv_t):
                        o2 = out[0] \\
                            if isinstance(
                                out,
                                tuple) \\
                            else out
                        o2[0, -1, :] += _dv
                        return None

                    hooks.append(
                        model_g.model
                        .layers[38]
                        .register_forward_hook(
                            _injv2))
                if inj_coord is not None:
                    (co_l, dl_v) = inj_coord
                    co_t = torch.as_tensor(
                        np.asarray(
                            co_l,
                            dtype=np.int64),
                        device='cuda')

                    def _injc2(mod, inp,
                               out,
                               _co=co_t,
                               _dl=dl_v):
                        o2 = out[0] \\
                            if isinstance(
                                out,
                                tuple) \\
                            else out
                        o2[0, -1, _co] += \\
                            float(_dl)
                        return None

                    hooks.append(
                        model_g.model
                        .layers[17]
                        .register_forward_hook(
                            _injc2))
            try:
                with torch.inference_mode():
                    model_g(
                        torch.tensor(
                            [list(row_ids)
                             + [int(t)
                                for t in
                                toks[:k]]],
                            device='cuda'),
                        use_cache=False)
            finally:
                for hk in hooks:
                    hk.remove()
            step_h.append(feats[39]
                          .float())
        outs.append(step_h)
    res_dlog = []
    for k in range(n_steps):
        hb = norm_g(outs[0][k].to(
            torch.bfloat16)
            .unsqueeze(0))[0].float()
        hi = norm_g(outs[1][k].to(
            torch.bfloat16)
            .unsqueeze(0))[0].float()
        dl = ((hi - hb) @ WUG_T.T) \\
            .cpu().numpy()
        res_dlog.append(
            dl.astype(np.float64))
    return res_dlog


_T2CK = CK['data'].get('t2')
if _T2CK is not None:
    T2_DATA = _T2CK['t2']
    log('T2 RESUMED')
else:
    T2_DATA = {'neg': {}, 'pos': {}}
    _tail_p = [int(c) for c in TAIL25]
    for j in flip_rows:
        _bt = base12_P[j]
        _it = pad12(_gen_n[j])
        dls = _dlogit_steps(
            rows_scan[j], _bt, _it,
            dv_n[j], None, T2_STEPS)
        _rec = {}
        for k in range(T2_STEPS):
            dl = dls[k]
            top = np.argsort(
                -np.abs(dl))[:T2_TOPK]
            r131 = int(np.sum(
                dl > dl[SPEC_TOKS[0]])) + 1
            _rec['s%d' % k] = {
                'top50': [[int(t),
                           float(dl[t])]
                          for t in top],
                'd131': float(
                    dl[SPEC_TOKS[0]]),
                'r131': r131}
        T2_DATA['neg'][str(j)] = _rec
        log('T2 neg row %d done' % j)
    for j in flp_rows:
        _bt = base12_A1[j]
        _it = pad12(
            E_res['s2_tailpos_d1']
            ['gens'][j])
        dls = _dlogit_steps(
            rows_A1[j], _bt, _it, None,
            (_tail_p, 1.0 * DELTA_L17),
            T2_STEPS)
        _rec = {}
        for k in range(T2_STEPS):
            dl = dls[k]
            top = np.argsort(
                -np.abs(dl))[:T2_TOPK]
            r131 = int(np.sum(
                dl > dl[SPEC_TOKS[0]])) + 1
            _rec['s%d' % k] = {
                'top50': [[int(t),
                           float(dl[t])]
                          for t in top],
                'd131': float(
                    dl[SPEC_TOKS[0]]),
                'r131': r131}
        T2_DATA['pos'][str(j)] = _rec
        log('T2 pos row %d done' % j)
    ck_save('t2', {'t2': T2_DATA})
common_tok = {}
t2_stats = {}
for grp in ('neg', 'pos'):
    rows_l = flip_rows if grp == 'neg' \\
        else flp_rows
    all_t = {}
    d131_all = []
    r131_all = []
    for j in rows_l:
        d = T2_DATA[grp][str(j)]
        for k in range(T2_STEPS):
            r = d['s%d' % k]
            d131_all.append(r['d131'])
            r131_all.append(r['r131'])
            for (tk, vl) in r['top50']:
                if vl > 0:
                    all_t[int(tk)] = \\
                        all_t.get(int(tk),
                                  0) + 1
    thr = int(np.ceil(len(rows_l)
                      * T2_SHARE))
    common = sorted(
        [t for t, c in all_t.items()
         if c >= thr
         and t != SPEC_TOKS[0]])
    common_tok[grp] = common
    t2_stats[grp] = {
        'n_rows': len(rows_l),
        'thr': thr,
        'n_common': len(common),
        'common': common[:30],
        'd131_med': float(np.median(
            d131_all)),
        'r131_med': float(np.median(
            r131_all))}
shared = sorted(set(common_tok['neg'])
                & set(common_tok['pos']))
if len(common_tok['neg']) >= 3 \\
        or len(common_tok['pos']) >= 3:
    t2_tag = 'carrier_common_found'
elif len(common_tok['neg']) >= 1 \\
        or len(common_tok['pos']) >= 1:
    t2_tag = 'carrier_common_sparse'
else:
    t2_tag = 'carrier_131401_only'
log('T2-GATE: neg common %s | pos '
    'common %s | shared %s | d131 med '
    'neg %.3f (r %.0f) pos %.3f '
    '(r %.0f) -> %s'
    % (json.dumps(common_tok['neg']
                  [:15]),
       json.dumps(common_tok['pos']
                  [:15]),
       json.dumps(shared[:10]),
       t2_stats['neg']['d131_med'],
       t2_stats['neg']['r131_med'],
       t2_stats['pos']['d131_med'],
       t2_stats['pos']['r131_med'],
       t2_tag))

# ================================================================
# PART L: pos_d1 late-onset mechanism
# ================================================================
log('== PART L: pos_d1 late mech ==')
_LCK = CK['data'].get('late')
if _LCK is not None:
    L_traj = _LCK['traj']
    L_cuts = _LCK['cuts']
    log('L RESUMED')
else:
    L_traj = {}
    wdn_np = w_dn_g.astype(np.float64)
    _tail_p = [int(c) for c in TAIL25]
    for j in flp_rows:
        _bt = base12_A1[j]
        _it = pad12(
            E_res['s2_tailpos_d1']
            ['gens'][j])
        outs = []
        for side in range(2):
            toks = _bt if side == 0 \\
                else _it
            step_h = []
            for k in range(L_STEPS):
                feats = {39: None}
                hooks = []

                def _mk39(mod, inp, out,
                          _f=feats):
                    o2 = out[0] \\
                        if isinstance(out,
                                      tuple) \\
                        else out
                    _f[39] = o2[0, -1, :] \\
                        .detach()
                    return None

                hooks.append(
                    model_g.model
                    .layers[39]
                    .register_forward_hook(
                        _mk39))
                if side == 1:

                    def _injp(mod, inp,
                              out,
                              _tp=_tail_p):
                        o2 = out[0]
                        o2[0, -1, _tp] += \\
                            1.0 * DELTA_L17
                        return None

                    hooks.append(
                        model_g.model
                        .layers[17]
                        .register_forward_hook(
                            _injp))
                try:
                    with torch.inference_mode():
                        model_g(
                            torch.tensor(
                                [list(rows_A1[j])
                                 + [int(t)
                                    for t in
                                    toks[:k]]],
                                device='cuda'),
                            use_cache=False)
                finally:
                    for hk in hooks:
                        hk.remove()
                step_h.append(
                    feats[39].float())
            outs.append(step_h)
        steps = []
        for k in range(L_STEPS):
            hb = norm_g(outs[0][k].to(
                torch.bfloat16)
                .unsqueeze(0))[0].float()
            hi = norm_g(outs[1][k].to(
                torch.bfloat16)
                .unsqueeze(0))[0].float()
            dh = ((hi - hb).cpu().numpy()
                  .astype(np.float64))
            d131 = float(
                ((hi - hb)
                 @ WUG_T.T)[0,
                            SPEC_TOKS[0]]
                .cpu())
            steps.append({
                'wdn': float(dh @ wdn_np),
                'dh': float(np.linalg
                            .norm(dh)),
                'd131': d131})
        L_traj[str(j)] = steps
        log('L traj row %d done' % j)
    for cut in L_CUTS:
        _run_coord_trial(
            'l_cut%d' % cut, _tail_p, 1,
            1.0, flp_rows, base12_A1,
            mode='cut%d' % cut)
    L_cuts = {}
    for jj, j in enumerate(flp_rows):
        kmin = -1
        for cut in L_CUTS:
            g = E_res['l_cut%d'
                      % cut]['gens'][jj]
            if pad12(g) != base12_A1[j]:
                kmin = cut
                break
        L_cuts[str(j)] = kmin
    ck_save('late', {'traj': L_traj,
                     'cuts': L_cuts})
fs_j = {j: int(fs_p1[j])
        for j in flp_rows}
km_list = [int(L_cuts[str(j)])
           for j in flp_rows]
fs_list = [fs_j[j] for j in flp_rows]
frac_fixed = float(np.mean(
    [0 <= L_cuts[str(j)] <= 1
     for j in flp_rows]))
wdn_early = float(np.median(
    [L_traj[str(j)][0]['wdn']
     for j in flp_rows]))
wdn_late = float(np.median(
    [L_traj[str(j)][min(L_STEPS - 1,
                        fs_j[j])]
     ['wdn'] for j in flp_rows]))
d131_early = float(np.median(
    [L_traj[str(j)][0]['d131']
     for j in flp_rows]))
d131_late = float(np.median(
    [L_traj[str(j)][min(L_STEPS - 1,
                        fs_j[j])]
     ['d131'] for j in flp_rows]))
log('L-SOFT: kmin %s vs fstep %s | wdn '
    'early %.3f late %.3f | d131 early '
    '%.3f late %.3f | frac_fixed %.2f'
    % (json.dumps(km_list),
       json.dumps(fs_list), wdn_early,
       wdn_late, d131_early, d131_late,
       frac_fixed))
if frac_fixed >= 0.5:
    l_tag = 'poslate_fixed_early'
elif wdn_early > 0.3 \\
        and wdn_late > wdn_early:
    l_tag = 'poslate_readout_comp'
elif d131_early > 0.3 \\
        and d131_late > d131_early:
    l_tag = 'poslate_format_grow'
else:
    l_tag = 'poslate_inj_delay'
log('L-GATE: %s' % l_tag)

# ================================================================
# PART K: coordinate-dose interchange
# ================================================================
log('== PART K: coord-dose interchange ==')
for kk in K_SUBS:
    for dd in K_DOSES:
        _run_coord_trial(
            'k_top%d_d%g' % (kk, dd),
            [int(c) for c in
             order_ex[:kk]], -1, dd,
            rows_A1, base12_A1)
kx = {(kk, dd): _chg('k_top%d_d%g'
                     % (kk, dd))
      for kk in K_SUBS
      for dd in K_DOSES}
full_curve = {0.5: X47K['full_05'],
              1.0: X47K['full_1'],
              2.0: X47K['full_2']}
best_gap = None
best_pair = None
for kk in K_SUBS:
    for dd in K_DOSES:
        for df, vf in full_curve.items():
            g = abs(kx[(kk, dd)] - vf)
            if best_gap is None \\
                    or g < best_gap:
                best_gap = g
                best_pair = (kk, dd, df)
if best_gap < 0.05:
    kx_tag = 'kx_interchangeable'
elif best_gap < 0.15:
    kx_tag = 'kx_partial'
else:
    kx_tag = 'kx_breadth_required'
log('K-GATE: kx %s | best match '
    'top%d@d%g~full@d%g gap %.4f -> %s'
    % (json.dumps({'top%d_d%g'
                   % (kk, dd):
                   round(float(v), 4)
                   for (kk, dd), v
                   in sorted(kx.items())}),
        best_pair[0], best_pair[1],
        best_pair[2], best_gap, kx_tag))

# ================================================================
# PART V3: v1 micro-amp fill (bias dose)
# ================================================================
log('== PART V3: v1 micro-amp fill ==')
v1f = v1.astype(np.float32)
for (a, tname) in V3_ALPHA:
    _run_vec_trial(tname, 29, None, 1.0,
                   'allstep', rows_scan,
                   base12_P,
                   clip=[(38, v1f, a,
                          'all')])
sym_curve = {
    '0.05': _chg('v3_ampan_a005')
    / max(V47MICRO['005'], 1e-9),
    '0.1': _chg('v3_ampan_a010')
    / max(V47MICRO['010'], 1e-9),
    '0.15': _chg('v3_ampan_a015')
    / max(V47MICRO['015'], 1e-9),
    '0.25': _chg('v2_amp_a025')
    / max(_chg('v2_clip_a025'), 1e-9),
    '0.5': _chg('v2_amp_a050')
    / max(_chg('v2_clip_a050'), 1e-9)}
sv = [sym_curve[k] for k in
      ('0.05', '0.1', '0.15', '0.25',
       '0.5')]
if max(sv) - min(sv) < 0.1:
    v3_tag = 'v3_bias_constant'
elif sv[0] > sv[-1] + 0.1:
    v3_tag = 'v3_bias_growing'
elif sv[-1] > sv[0] + 0.1:
    v3_tag = 'v3_bias_shrinking'
else:
    v3_tag = 'v3_bias_irregular'
log('V3-GATE: sym curve %s -> %s'
    % (json.dumps({k: round(float(v), 3)
                   for k, v
                   in sym_curve.items()}),
       v3_tag))
'''

s = s.replace(ANCHOR, ANCHOR + NEW_PARTS)
print('OK new parts inserted')

io.open(FP, 'w', encoding='utf-8',
        newline='\n').write(s)
print('PATCH3A_DONE')
