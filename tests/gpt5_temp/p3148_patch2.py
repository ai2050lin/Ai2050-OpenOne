# -*- coding: utf-8 -*-
"""p3148 patch2: replace PART V + verdict
tail with new PART X2/H/U/V2 + verdict/
result/npz. Run AFTER patch1."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
      r'\phase3148_omega_p146_xcross_'
      r'headsrc_uncancel_v1sym.py')

s = io.open(FP, encoding='utf-8').read()
if 'p3148 patch2 applied' in s:
    print('ALREADY_APPLIED')
    raise SystemExit
if 'p3148 patch1 applied' not in s:
    raise SystemExit('PATCH1_MISSING')

MARK = ('# ================================================================\n'
        '# PART V: v1 micro clip + amplitude traj')
idx = s.index(MARK)
head = s[:idx]

NEW_TAIL = r'''# ================================================================
# PART X2: tail sign-crossing mid-dose fill
# ================================================================
log('== PART X2: sign-cross fill ==')
TAIL_DOSES = (1.0, 2.0) + tuple(XDOSE_MID) \
    + (4.0,)
for dd in TAIL_DOSES:
    _run_coord_trial('s2_tailpos_d%g' % dd,
                     TAIL25, 1, dd, rows_A1,
                     base12_A1)
    _run_coord_trial('s2_tailneg_d%g' % dd,
                     TAIL25, -1, dd, rows_A1,
                     base12_A1)
if not SMOKE:
    for tn, wv in WANT_TAIL.items():
        got = E_res[tn]['chg']
        m = abs(got - wv) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': wv,
            'match': bool(m)}
    log('X2 replay: +5 bit anchors (total '
        '%d/%d)' % (n_bit_all, N_BIT_TOT))
posc = {dd: _chg('s2_tailpos_d%g' % dd)
        for dd in TAIL_DOSES}
negc = {dd: _chg('s2_tailneg_d%g' % dd)
        for dd in TAIL_DOSES}
diffc = {dd: posc[dd] - negc[dd]
         for dd in TAIL_DOSES}
tp2_repro = abs(posc[2.0] - 0.2578125) \
    < 1e-9
log('X2-SOFT: pos_d2 %.4f vs 3147 '
    '0.2578 repro %s' % (posc[2.0],
                         tp2_repro))
d_lo = None
d_hi = None
for dd in TAIL_DOSES:
    if diffc[dd] > SIGN_TOL:
        d_lo = dd
    elif d_lo is not None \
            and d_hi is None \
            and diffc[dd] < -SIGN_TOL:
        d_hi = dd
        break
if d_lo is not None and d_hi is not None:
    xcross_tag = 'tail_xcross_located'
    xcross_int = [float(d_lo),
                  float(d_hi)]
else:
    xcross_tag = 'tail_xcross_gradual'
    xcross_int = None
log('X-GATE2: pos %s'
    % json.dumps({'%g' % k: round(v, 4)
                  for k, v
                  in sorted(posc.items())}))
log('X-GATE2: neg %s'
    % json.dumps({'%g' % k: round(v, 4)
                  for k, v
                  in sorted(negc.items())}))
log('X-GATE2: diff %s -> %s (interval %s)'
    % json.dumps({'%g' % k: round(v, 4)
                  for k, v
                  in sorted(diffc.items())}),
    xcross_tag, xcross_int))
# fstep structure: pos_d1 flips vs
# neg_d4 flips (format vs answer-side)
fs_p1 = fstep_store['s2_tailpos_d1']
fs_n4 = fstep_store['s2_tailneg_d4']
fl_p = [int(fs_p1[j]) for j in range(NCAP)
        if fs_p1[j] >= 0]
fl_n = [int(fs_n4[j]) for j in range(NCAP)
        if fs_n4[j] >= 0]
med_fp = float(np.median(fl_p)) \
    if fl_p else -1.0
med_fn = float(np.median(fl_n)) \
    if fl_n else -1.0
early_p = float(np.mean([v <= 1
                         for v in fl_p])) \
    if fl_p else 0.0
early_n = float(np.mean([v <= 1
                         for v in fl_n])) \
    if fl_n else 0.0
if med_fn > med_fp:
    fstep_tag = 'xcross_fstep_neglate'
elif med_fp > med_fn:
    fstep_tag = 'xcross_fstep_poslate'
else:
    fstep_tag = 'xcross_fstep_equal'
log('X-SOFT2: pos_d1 flips n=%d med_fs='
    '%.1f early=%.2f | neg_d4 flips n=%d '
    'med_fs=%.1f early=%.2f -> %s'
    % (len(fl_p), med_fp, early_p,
       len(fl_n), med_fn, early_n,
       fstep_tag))

# ================================================================
# PART H: top-2 source write-side identity
# ================================================================
log('== PART H: top-2 identity ==')
for dd in H_DOSES:
    _run_coord_trial('x_h1_d%g' % dd,
                     [HEAD_COORDS[0]], -1, dd,
                     rows_A1, base12_A1)
    _run_coord_trial('x_h2_d%g' % dd,
                     [HEAD_COORDS[1]], -1, dd,
                     rows_A1, base12_A1)
    _run_coord_trial('x_h12_d%g' % dd,
                     HEAD_COORDS, -1, dd,
                     rows_A1, base12_A1)
c1_2 = _chg('x_h1_d2')
c2_2 = _chg('x_h2_d2')
c12_2 = _chg('x_h12_d2')
resid_pair = c12_2 - (c1_2 + c2_2)
if abs(resid_pair) < SIGN_TOL:
    h_pair = 'h_pair_additive'
elif resid_pair > SIGN_TOL:
    h_pair = 'h_pair_super'
else:
    h_pair = 'h_pair_sub'
share_top2 = c12_2 / max(
    DVALS46['d_co50ex_d2.0'], 1e-9)
log('H-GATE1: solo1 %.4f solo2 %.4f pair '
    '%.4f resid %+.4f -> %s (share of '
    'co50ex_d2 %.3f)'
    % (c1_2, c2_2, c12_2, resid_pair,
       h_pair, share_top2))


def _mk_head_abl(head_h):
    def pre(mod, args):
        x = args[0]
        xb = x.view(x.shape[0], x.shape[1],
                    NHEADS, -1).clone()
        xb[:, :, head_h, :] = 0
        return (xb.view(x.shape),)
    return pre


def _cap_swap_head(rows_all, cap_layers,
                   head_h):
    """Single-sample capture with swap4
    layer-skip + o_proj-input head
    ablation (R55-legal)."""
    OUTS = {l: np.zeros((len(rows_all),
                         HIDG),
                        dtype=np.float32)
            for l in cap_layers}
    lyr17 = model_g.model.layers[17]
    for j in range(len(rows_all)):
        feats = {l: None
                 for l in cap_layers}
        hooks = []
        hooks.append(
            lyr17.self_attn.o_proj
            .register_forward_pre_hook(
                _mk_head_abl(head_h)))
        for l in SWAP_L:
            lyr = model_g.model.layers[l]

            def _swap(mod, inp, out):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                i2 = inp[0] \
                    if isinstance(inp,
                                  tuple) \
                    else inp
                o2.copy_(
                    i2.to(o2.dtype))

            hooks.append(
                lyr.register_forward_hook(
                    _swap))

        def _mk(_l):
            def hook(mod, inp, out):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                feats[_l] = \
                    o2[0, -1, :].detach()
            return hook

        for l in cap_layers:
            hooks.append(
                model_g.model.layers[l]
                .register_forward_hook(
                    _mk(l)))
        try:
            with torch.inference_mode():
                model_g(
                    torch.tensor(
                        [list(rows_all[j])],
                        device='cuda'),
                    use_cache=False)
        finally:
            for hk in hooks:
                hk.remove()
        for l in cap_layers:
            OUTS[l][j] = feats[l].float() \
                .cpu().numpy()
        del feats
    return OUTS


rows_head = rows_cap[:HEAD_ROWS]
_KH = CK['data'].get('headab')
if _KH is not None:
    head_contrib = np.asarray(
        _KH['contrib'],
        dtype=np.float32)
    log('headab RESUMED (%d heads)'
        % head_contrib.shape[0])
else:
    head_contrib = np.zeros(
        (NHEADS, HEAD_ROWS, 2),
        dtype=np.float32)
    for h in range(NHEADS):
        sw = _cap_swap_head(
            rows_head, [19], h)[19]
        fh = (sw - base_states[19]
              [:HEAD_ROWS]) \
            .astype(np.float32)
        for k, c in enumerate(
                HEAD_COORDS):
            head_contrib[h, :, k] = (
                np.abs(field[19]
                       [:HEAD_ROWS, c]
                       .astype(np.float64))
                - np.abs(fh[:, c]
                         .astype(
                             np.float64))
                ).astype(np.float32)
        log('headab h=%02d done' % h)
    ck_save('headab', {
        'contrib': head_contrib})
contrib_head = np.median(head_contrib,
                         axis=1).sum(axis=1)
order_head = np.argsort(-contrib_head)
pos_sum = float(np.sum(
    np.maximum(contrib_head, 0.0)))
if pos_sum > 1e-9:
    frac_top2 = float(
        contrib_head[order_head[:2]]
        .sum() / pos_sum)
else:
    frac_top2 = 0.0
if frac_top2 > 0.4:
    h_head = 'head_conc_top2'
elif frac_top2 > 0.25:
    h_head = 'head_conc_moderate'
else:
    h_head = 'head_conc_diffuse'
log('H-GATE2: per-head contrib top6 %s '
    'frac_top2 %.3f -> %s'
    % (json.dumps([round(
        float(contrib_head[h]), 4)
        for h in order_head[:6]]),
       frac_top2, h_head))
# H3: top-50 overlap with frozen dvecs
dv29_mean = np.abs(
    dvec[29].astype(np.float64)
    .mean(axis=0))
dv19_mean = np.abs(
    dvec19.astype(np.float64)
    .mean(axis=0))
top29 = np.argsort(-dv29_mean)[:50]
top19 = np.argsort(-dv19_mean)[:50]
s29 = set(top29.tolist())
s19 = set(top19.tolist())
in29 = [int(c) in s29
        for c in HEAD_COORDS]
in19 = [int(c) in s19
        for c in HEAD_COORDS]
rank29 = {}
rank19 = {}
for c in HEAD_COORDS:
    c = int(c)
    rank29[c] = (int(np.where(
        top29 == c)[0][0]) + 1
        if c in s29 else -1)
    rank19[c] = (int(np.where(
        top19 == c)[0][0]) + 1
        if c in s19 else -1)
n_in = int(sum(in29) + sum(in19))
if n_in >= 2:
    h_overlap = 'src_top_overlap'
elif n_in == 1:
    h_overlap = 'src_top_partial'
else:
    h_overlap = 'src_top_absent'
log('H-SOFT3: ranks dv29 %s dv19 %s -> '
    '%s' % (json.dumps(rank29),
            json.dumps(rank19),
            h_overlap))

# ================================================================
# PART U: unembed-cancel causality
# ================================================================
log('== PART U: unembed cancel ==')
w_tok = WUG[SPEC_TOKS[0]] \
    .to(torch.float32).cpu().numpy() \
    .astype(np.float64)
w_tok_u = (w_tok / max(np.linalg.norm(
    w_tok), 1e-12)).astype(np.float32)
_rng_u = np.random.RandomState(RNG_SEED)
_r_u = _rng_u.randn(4096)
r_u = (_r_u / np.linalg.norm(_r_u)) \
    .astype(np.float32)
wdn_u = (w_dn_g.astype(np.float64)
         / max(np.linalg.norm(w_dn_g),
               1e-12)).astype(np.float32)
chg_only = chg_n
_run_vec_trial('u_neg_cancel', 38,
               -dv_n_tile, 1.0, 'allstep',
               rows_scan, base12_P,
               clip=[(39, w_tok_u, 1.0,
                      'all')])
_run_vec_trial('u_neg_rand', 38,
               -dv_n_tile, 1.0, 'allstep',
               rows_scan, base12_P,
               clip=[(39, r_u, 1.0, 'all')])
_run_vec_trial('u_neg_wdn', 38,
               -dv_n_tile, 1.0, 'allstep',
               rows_scan, base12_P,
               clip=[(39, wdn_u, 1.0,
                      'all')])
d_cancel = chg_only - _chg('u_neg_cancel')
d_rand = chg_only - _chg('u_neg_rand')
d_wdn = chg_only - _chg('u_neg_wdn')
if d_cancel > UCANCEL_TOL \
        and d_cancel > 2.0 * max(d_rand,
                                 0.01) \
        and d_cancel > d_wdn + 0.02:
    u_tag = 'uncancel_format_sufficient'
elif d_cancel > UCANCEL_TOL \
        and d_wdn > UCANCEL_TOL:
    u_tag = 'uncancel_nonspecific'
elif d_cancel <= UCANCEL_TOL:
    u_tag = 'uncancel_insufficient'
else:
    u_tag = 'uncancel_partial'
fs_can = fstep_store['u_neg_cancel']
fs_rnd = fstep_store['u_neg_rand']
rec_can = float(np.mean(
    [fs_can[j] == -1
     for j in flip_rows])) \
    if flip_rows else 0.0
rec_rnd = float(np.mean(
    [fs_rnd[j] == -1
     for j in flip_rows])) \
    if flip_rows else 0.0
log('U-GATE: only %.4f cancel %.4f rand '
    '%.4f wdn %.4f (d %+.4f/%+.4f/'
    '%+.4f) flip-rec can %.2f rnd %.2f '
    '-> %s'
    % (chg_only, _chg('u_neg_cancel'),
       _chg('u_neg_rand'),
       _chg('u_neg_wdn'), d_cancel,
       d_rand, d_wdn, rec_can, rec_rnd,
       u_tag))

# ================================================================
# PART V2: v1 perturbation symmetry
# ================================================================
log('== PART V2: v1 symmetry ==')
v1f = v1.astype(np.float32)
for (a, tname) in V2_ALPHA:
    _run_vec_trial(tname, 29, None, 1.0,
                   'allstep', rows_scan,
                   base12_P,
                   clip=[(38, v1f, a,
                          'all')])
if not SMOKE:
    for tn, wv in WANT_V2.items():
        got = E_res[tn]['chg']
        m = abs(got - wv) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': wv,
            'match': bool(m)}
    log('V2 replay: +2 bit anchors (total '
        '%d/%d)' % (n_bit_all, N_BIT_TOT))
sym25 = _chg('v2_amp_a025') / max(
    _chg('v2_clip_a025'), 1e-9)
sym50 = _chg('v2_amp_a050') / max(
    _chg('v2_clip_a050'), 1e-9)
if sym25 > 0.8 and sym50 > 0.8:
    v_sym = 'v1_sym_amplitude'
elif sym25 < 0.5 and sym50 < 0.5:
    v_sym = 'v1_sym_directional'
else:
    v_sym = 'v1_sym_mixed'
log('V-GATE2: amp025 %.4f vs clip025 '
    '%.4f (sym %.2f); amp050 %.4f vs '
    'clip050 %.4f (sym %.2f) -> %s'
    % (_chg('v2_amp_a025'),
       _chg('v2_clip_a025'), sym25,
       _chg('v2_amp_a050'),
       _chg('v2_clip_a050'), sym50,
       v_sym))

# ================================================================
# verdict + result.json + npz
# ================================================================
tags = ['a_3147_ok']
if not SMOKE:
    tags.append('repro_bit_%d' % n_bit_all)
    tags.append('repro_bit_ok'
                if n_bit_all == N_BIT_TOT
                else 'repro_bit_drift')
else:
    tags.append('repro_smoke_skipped')
tags.append('dvec19_repro_%s'
            % dvec19_sha[:6])
tags.append('field_self_ok'
            if field_self_ok
            else 'field_self_drift')
tags.append('z35_cos17_ok' if z35_ok
            else 'z35_cos17_drift')
tags.append('resid_anchor_ok'
            if resid_anchor_ok
            else 'resid_anchor_drift')
tags.append(xcross_tag)
tags.append(fstep_tag)
tags.append(h_pair)
tags.append(h_head)
tags.append(h_overlap)
tags.append(u_tag)
tags.append(v_sym)
tags.append('xphase_ok' if xphase_ok
            else 'xphase_drift')
tags.append('coverage_full')
verdict = '|'.join(tags)
log('VERDICT: %s' % verdict)

result = {
    'phase': 3148,
    'name': NAME,
    'smoke': SMOKE,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'runtime_s': time.time() - T0,
    'seal_sha8': hashlib.sha256(
        json.dumps(SEAL, sort_keys=True,
                   ensure_ascii=False)
        .encode('utf-8')).hexdigest()[:8],
    'verdict': verdict,
    'constants': SEAL['constants'],
    'part_a': {
        'res47_sha8': sha47,
        'seal47': SEAL47,
        'xphase_P': xphase_P,
        'xphase_A1': xphase_A1},
    'part_field': {
        'dvec19_sha8': dvec19_sha,
        'dvec19_sha_ok':
            bool(d19_sha_ok),
        'dvec19_medn': dvec19_medn,
        'field17_vs_dvec17_cos': fs17,
        'field19_vs_dvec19_cos': fs19,
        'field_self_ok':
            bool(field_self_ok),
        'z35_c17_maxdiff': z35_diff,
        'z35_ok': bool(z35_ok)},
    'part_resid': {
        'share38': share38,
        'wdn_p38': wdn_p38,
        'wdn_p39': wdn_p39,
        'dh19_wdn39': dh19_wdn39,
        'pnorm38': pnorm38,
        'resid_align': ra_all,
        'anchor_ok':
            bool(resid_anchor_ok)},
    'part_bits': {'bit_anchors':
                  bit_anchors},
    'part_x2': {
        'pos_curve': {'%g' % k:
                      float(v) for k, v
                      in sorted(
                          posc.items())},
        'neg_curve': {'%g' % k:
                      float(v) for k, v
                      in sorted(
                          negc.items())},
        'diff_curve': {'%g' % k:
                       float(v) for k, v
                       in sorted(
                           diffc.items())},
        'xcross_interval': xcross_int,
        'xcross_tag': xcross_tag,
        'tailpos_d2_repro':
            bool(tp2_repro),
        'flip_pos_d1': {
            'n': len(fl_p),
            'med_fstep': med_fp,
            'early_frac': early_p},
        'flip_neg_d4': {
            'n': len(fl_n),
            'med_fstep': med_fn,
            'early_frac': early_n},
        'fstep_tag': fstep_tag},
    'part_h': {
        'solo1_d2': c1_2,
        'solo2_d2': c2_2,
        'pair_d2': c12_2,
        'resid_pair': resid_pair,
        'h_pair': h_pair,
        'share_top2_vs_co50ex_d2':
            share_top2,
        'dose_solo1': {'%g' % dd:
                       _chg('x_h1_d%g' % dd)
                       for dd in H_DOSES},
        'dose_solo2': {'%g' % dd:
                       _chg('x_h2_d%g' % dd)
                       for dd in H_DOSES},
        'dose_pair': {'%g' % dd:
                      _chg('x_h12_d%g' % dd)
                      for dd in H_DOSES},
        'head_order_top6': [int(h) for h
                            in order_head
                            [:6]],
        'head_contrib_top6': [
            float(contrib_head[h])
            for h in order_head[:6]],
        'frac_top2': frac_top2,
        'h_head': h_head,
        'rank_dv29': rank29,
        'rank_dv19': rank19,
        'in_top50': {'dv29':
                     [bool(v) for v
                      in in29],
                     'dv19':
                     [bool(v) for v
                      in in19]},
        'h_overlap': h_overlap},
    'part_u': {
        'chg_only': chg_only,
        'chg_cancel': _chg('u_neg_cancel'),
        'chg_rand': _chg('u_neg_rand'),
        'chg_wdn': _chg('u_neg_wdn'),
        'd_cancel': d_cancel,
        'd_rand': d_rand,
        'd_wdn': d_wdn,
        'rec_cancel': rec_can,
        'rec_rand': rec_rnd,
        'u_tag': u_tag},
    'part_v2': {
        'chg_amp025': _chg('v2_amp_a025'),
        'chg_amp050': _chg('v2_amp_a050'),
        'chg_clip025': _chg(
            'v2_clip_a025'),
        'chg_clip050': _chg(
            'v2_clip_a050'),
        'sym25': sym25,
        'sym50': sym50,
        'v_sym': v_sym},
    }
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False,
              indent=1)
npz_out = {
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
         **npz_out)
if not SMOKE:
    try:
        os.remove(CKPTF)
        log('ckpt cleaned (final)')
    except OSError:
        pass
log('result.json + npz written. DONE '
    '(verdict %s)' % verdict)
'''

# fix: the log call with embedded % json
# needs parens balanced; patch it here
NEW_TAIL = NEW_TAIL.replace(
    "log('X-GATE2: diff %s -> %s (interval %s)'\n"
    "    % json.dumps({'%g' % k: round(v, 4)\n"
    "                  for k, v\n"
    "                  in sorted(diffc.items())}),\n"
    "    xcross_tag, xcross_int))",
    "log('X-GATE2: diff %s -> %s (interval %s)'\n"
    "    % (json.dumps({'%g' % k: round(v, 4)\n"
    "                     for k, v\n"
    "                     in sorted(diffc.items())}),\n"
    "       xcross_tag, xcross_int))")

out = head + NEW_TAIL + '\n# p3148 patch2 applied\n'
io.open(FP, 'w', encoding='utf-8',
        newline='\n').write(out)
print('PATCH2_OK tail_len=%d total=%d'
      % (len(NEW_TAIL), len(out)))
