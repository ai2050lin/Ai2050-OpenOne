# -*- coding: utf-8 -*-
"""Phase 2965: functional identity of L34/h15, the unique
maxT-significant class-effect carrier head (2964).

Why: 2964 localized the function-vs-content load-band gap
to layer L34 (gap -4.0) and head L34/h15 (gap -1.66,
unique maxT-significant head). Open question: is h15 a
CAUSAL class readout head (its per-head write into the
residual stream drives the B class gap; ablating it
abolishes the effect) or a correlational bystander (its
chunk tracks class because of rebalancing / shared input)?

Design (frozen before any observation):
  Word list VERBATIM from the sealed 2964 npz (30 words,
  labels from npz). 2937 pass1 protocol verbatim single
  forwards [the, w]; captures o_proj input pos1 all 36
  layers; B = prof[6:13].mean() - prof[28:36].mean()
  (2964 spec). INACT run (30 forwards) = cross-phase
  anchor. ABLATION sweep: for each head h in 0..31, zero
  the o_proj INPUT slice [1, h*128:(h+1)*128] at L34 only
  (2932 proven hook pattern, in-place mutation) and
  re-forward all 30 words (960 forwards).

Anchors (frozen):
  a1 o_proj shape L34 == (2560, 4096)
  a2 dirs_word reload vs 2927 npz < 1e-6
  a3 intact B (30 words) vs 2964 npz B rel < 1e-4
     (cross-phase protocol identity, batch composition
     bit-identical: same [the, w] single forwards)
  a4 single-token 30/30 + word list == 2964 npz words
  a5 gap_heads_L34[15] vs 2964 result.json round gate
     5.01e-3 (npz authoritative)
  a6 chunk-vs-direct < 1e-9 at L34 (intact)
  a7 ablation efficacy: |C_abl[34,w,15]| < 1e-6 for all
     30 words (implementation gate, 2929 lesson)
  a8 non-degeneracy: intact C[34] per-head std > 0 (32/32)

Tests (frozen):
  T1 descriptive (NO gate): per-head write identity at
     L34. out_w[h] = Wo_L34[:, h] @ x_h(w); class contrast
     c_h = mean_F(out) - mean_N(out); cos(c_h, u35) for
     all 32 heads; rank of |cos| for h15; projection of
     c15 on Vt8 rows. h15 population rank is descriptive.
  T2 carrier correlation: rho(C[34,w,15], B_w) over 30
     words, two-sided label permutation (rng 2971, 10000);
     gate p <= 0.01.
  T3 functional ablation (main):
     T3a Freedman-Lane on B_abl[15] (rng 2972, 10000),
         same regression as 2964 T1 (B ~ rank(tid)+group);
         gate: p_fl_abl > 0.05 => class effect abolished
     T3b descriptive: R[h] = gap_intact - gap_abl[h] for
         all 32 heads; rank of R[15]; mean|dB| per head
         (global-damage check); class-selectivity of h15
         = R[15] vs mean|dB| of h15.

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T2 pass (p<=0.01) & T3a p>0.05 => h15_necessary_class_carrier
  T2 pass & T3a p<=0.05 => h15_shared_carrier_effect_survives
  T2 fail & T3a p>0.05 => h15_causal_correlation_mismatch
  else => h15_correlational_weak
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2964 = os.path.join(BASE, 'phase2964', 'carrier_anatomy',
                        'carrier_anatomy.npz')
SRC_2964_R = os.path.join(BASE, 'phase2964',
                          'carrier_anatomy', 'result.json')
OUT = os.path.join(BASE, 'phase2965', 'h15_functional_identity')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2965_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
LI_ABL = 34
H_TGT = 15
N_PERM = 10000
RNG_T2, RNG_T3A = 2971, 2972

PREREG = {
    'mode': '990 single forwards (30 intact + 32 heads x 30 '
            'ablated), 2937 pass1 protocol verbatim; word '
            'list verbatim from sealed 2964 npz; ablation = '
            'zero o_proj input slice [1, h*128:(h+1)*128] '
            'at L34 only (2932 pattern)',
    'question': 'is L34/h15 a causal class readout head '
                '(ablation abolishes the function-vs-content '
                'B gap) or a correlational bystander?',
    'anchors': {
        'a1': 'o_proj shape L34 == (2560, 4096)',
        'a2': 'dirs_word reload vs 2927 npz < 1e-6',
        'a3': 'intact B vs 2964 npz rel < 1e-4',
        'a4': 'single-token 30/30 + list == 2964 npz',
        'a5': 'gap_heads_L34[15] vs 2964 round gate 5.01e-3',
        'a6': 'chunk-vs-direct L34 < 1e-9',
        'a7': 'ablation efficacy |C_abl[34,w,15]| < 1e-6',
        'a8': 'non-degeneracy C[34] std > 0 (32/32)',
    },
    'T1': 'descriptive (no gate): cos(c_h, u35) population '
          'over 32 heads, rank of h15; c15 on Vt8 rows',
    'T2': 'rho(C[34,w,15], B_w), two-sided perm rng 2971 '
          'x10000; gate p <= 0.01',
    'T3': 'T3a FL on B_abl[15] (rng 2972 x10000), gate '
          'p > 0.05 => abolished; T3b descriptive R[h] '
          'profile + global damage',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T2 pass & T3a>0.05 => '
               'h15_necessary_class_carrier; T2 pass & '
               'T3a<=0.05 => '
               'h15_shared_carrier_effect_survives; T2 '
               'fail & T3a>0.05 => '
               'h15_causal_correlation_mismatch; else => '
               'h15_correlational_weak',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def log(msg, lines):
    lines.append(msg)
    print(msg, flush=True)


def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra = rankdata(np.asarray(a, dtype=np.float64))
    rb = rankdata(np.asarray(b, dtype=np.float64))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if den < 1e-30:
        return 0.0
    return float((ra * rb).sum() / den)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2965,
                   'name': 'h15_functional_identity',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2927': sha8(SRC_2927),
                               's2939': sha8(SRC_2939),
                               's2964': sha8(SRC_2964),
                               's2964r': sha8(SRC_2964_R)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'abl_layer': LI_ABL, 'target_head': H_TGT,
                   'n_forwards': 990, 'n_perm': N_PERM,
                   'rng': {'T2': RNG_T2, 'T3a': RNG_T3A},
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
    u35 = dirs27[NL - 1]
    z39 = np.load(SRC_2939, allow_pickle=True)
    Vt8 = z39['Vt8'].astype(np.float64)
    z64 = np.load(SRC_2964, allow_pickle=True)
    B64 = z64['B'].astype(np.float64)
    C64 = z64['C'].astype(np.float64)          # (36,30,32)
    lab = z64['labels'].astype(int)
    wlist = [str(w).split(':')[1]
             for w in z64['words']]
    r64 = json.load(open(SRC_2964_R, encoding='utf-8'))
    gap_h64 = np.load(SRC_2964, allow_pickle=True)[
        'gap_heads_L34'].astype(np.float64)

    a2_diff = float(np.abs(dirs27
                           - z27['dirs_word']
                           .astype(np.float64)).max())
    a2_ok = bool(a2_diff < 1e-6)
    log('a2 dirs_word reload %.2e ok=%s'
        % (a2_diff, a2_ok), lines)
    top5_64 = dict((int(a), float(b))
                   for a, b in r64['T3']['top5_heads'])
    a5_diff = abs(float(gap_h64[H_TGT])
                  - top5_64[H_TGT])
    a5_ok = bool(a5_diff < 5.01e-3)
    log('a5 gap_heads_L34[15] npz %.4f vs json %.4f '
        'diff %.2e ok=%s'
        % (gap_h64[H_TGT], top5_64[H_TGT],
           a5_diff, a5_ok), lines)

    mF = lab == 0
    mN = lab == 1

    # ---------- model ----------
    import torch
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tid_map = {}
    n_single = 0
    for w in wlist:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)[
                'input_ids']
        assert len(ids) == 1, '%s -> %s' % (w, ids)
        tid_map[w] = int(ids[0])
        n_single += 1
    a4_ok = bool(n_single == 30
                 and wlist == [str(w).split(':')[1]
                               for w in z64['words']])
    log('a4 single-token %d/30, list==2964 ok=%s'
        % (n_single, a4_ok), lines)
    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    Wo34 = layers[LI_ABL].self_attn.o_proj.weight.detach() \
        .float().cpu().numpy()
    a1_ok = bool(Wo34.shape == (2560, NH * HD))
    log('a1 o_proj shape L34 %s ok=%s'
        % (Wo34.shape, a1_ok), lines)

    cap_op = {li: [] for li in range(NL)}
    abl = {'h': None}
    handles = []

    def hook_op(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            if li == LI_ABL and abl['h'] is not None:
                hh = abl['h']
                x[:, 1, hh * HD:(hh + 1) * HD] = 0
            cap_op[li].append(
                x[:, 1, :].detach().float().cpu().numpy())
            return None
        return h

    for li in range(NL):
        handles.append(
            layers[li].self_attn.o_proj
            .register_forward_pre_hook(
                hook_op(li), with_kwargs=True))

    def clear_cap():
        for li in cap_op:
            del cap_op[li][:]

    def forward1(toks, abl_head=None):
        clear_cap()
        abl['h'] = abl_head
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        abl['h'] = None
        return {li: cap_op[li][0].astype(np.float64)
                for li in range(NL)}

    M = np.zeros((NL, NH * HD))
    for li in range(NL):
        Wl = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wl

    def contributions(op):
        C = np.zeros((NL, NH))
        prof = np.zeros(NL)
        for li in range(NL):
            x = op[li].reshape(-1)
            xm = (x * M[li]).reshape(NH, HD)
            C[li] = xm.sum(axis=1)
            prof[li] = float(C[li].sum())
        return C, prof

    def band_of(prof):
        return (float(prof[6:13].mean())
                - float(prof[28:36].mean()))

    # ---------- intact run ----------
    B_int = np.zeros(30)
    op34 = np.zeros((30, NH * HD))
    C34_int = np.zeros((30, NH))
    for i, w in enumerate(wlist):
        op = forward1([func_tid, tid_map[w]])
        C, prof = contributions(op)
        B_int[i] = band_of(prof)
        op34[i] = op[LI_ABL].reshape(-1)
        C34_int[i] = C[LI_ABL]
        if (i + 1) % 10 == 0:
            log('intact [%d/30]' % (i + 1), lines)

    a3_rel = float(np.abs(B_int - B64).max()
                   / max(float(np.abs(B64).max()), 1e-30))
    a3_ok = bool(a3_rel < 1e-4)
    log('a3 intact B vs 2964 npz rel %.2e ok=%s'
        % (a3_rel, a3_ok), lines)

    a6_rel = 0.0
    x34 = op34[0].reshape(-1)
    direct = float(np.dot(x34, M[LI_ABL]))
    xm = (x34 * M[LI_ABL]).reshape(NH, HD)
    a6_rel = abs(direct - float(xm.sum())) \
        / max(abs(direct), 1e-30)
    a6_ok = bool(a6_rel < 1e-9)
    log('a6 chunk-vs-direct L34 rel %.2e ok=%s'
        % (a6_rel, a6_ok), lines)

    a8_ok = bool(C34_int.std(axis=0).min() > 0)
    log('a8 non-degeneracy C34 std min %.3e ok=%s'
        % (C34_int.std(axis=0).min(), a8_ok), lines)

    gap_intact = float(np.median(B_int[mF])
                       - np.median(B_int[mN]))

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a8_ok)
    verdict = None
    t1 = t2 = t3 = None
    save = {}
    a7_max = 0.0

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- T1 descriptive write identity ------
        W15 = Wo34[:, H_TGT * HD:(H_TGT + 1) * HD]
        out15 = op34[:, H_TGT * HD:(H_TGT + 1) * HD] @ W15.T
        c15 = out15[mF].mean(0) - out15[mN].mean(0)
        cos_c15_u35 = float(
            c15 @ u35 / max(float(np.linalg.norm(c15))
                            * float(np.linalg.norm(u35)),
                            1e-30))
        # population over 32 heads
        cos_pop = np.zeros(NH)
        for h in range(NH):
            Wh = Wo34[:, h * HD:(h + 1) * HD]
            outh = op34[:, h * HD:(h + 1) * HD] @ Wh.T
            ch = outh[mF].mean(0) - outh[mN].mean(0)
            cos_pop[h] = ch @ u35 / max(
                float(np.linalg.norm(ch))
                * float(np.linalg.norm(u35)), 1e-30)
        rank15 = int(np.argsort(-np.abs(cos_pop))
                     .tolist().index(H_TGT)) + 1
        c15_vt = [round(float(c15 @ Vt8[k]
                           / max(float(np.linalg.norm(c15))
                                 * float(np.linalg.norm(
                                     Vt8[k])), 1e-30)), 4)
                  for k in range(4)]
        t1 = {'cos_c15_u35': round(cos_c15_u35, 4),
              'abs_cos_rank_of_h15': rank15,
              'cos_pop_top3': [
                  (int(h), round(float(cos_pop[h]), 4))
                  for h in np.argsort(
                      -np.abs(cos_pop))[:3]],
              'c15_on_Vt8_top4': c15_vt,
              'note': 'descriptive, no gate (frozen)'}
        log('T1 cos(c15,u35) %+.4f rank |cos| %d/32 | '
            'Vt8 %s' % (cos_c15_u35, rank15, c15_vt),
            lines)

        # ---------- T2 carrier correlation ----------
        rho2 = spearman(C34_int[:, H_TGT], B_int)
        rng2 = np.random.default_rng(RNG_T2)
        cnt2 = 0
        for _ in range(N_PERM):
            if abs(spearman(C34_int[:, H_TGT],
                            rng2.permutation(B_int))) \
                    >= abs(rho2) - 1e-12:
                cnt2 += 1
        p2 = (cnt2 + 1) / (N_PERM + 1)
        t2_pass = bool(p2 <= 0.01)
        t2 = {'rho': round(float(rho2), 4),
              'perm_p': float('%.3e' % p2),
              'pass': t2_pass}
        log('T2 rho(C15,B) %.4f perm-p %.3e pass=%s'
            % (rho2, p2, t2_pass), lines)

        # ---------- T3 ablation sweep ----------
        B_abl = np.zeros((NH, 30))
        gap_abl = np.zeros(NH)
        a7_max = 0.0
        for h in range(NH):
            for i, w in enumerate(wlist):
                op = forward1([func_tid, tid_map[w]],
                              abl_head=h)
                C, prof = contributions(op)
                B_abl[h, i] = band_of(prof)
                if h == H_TGT:
                    a7_max = max(a7_max,
                                 abs(float(C[LI_ABL, H_TGT])))
            gap_abl[h] = float(np.median(B_abl[h][mF])
                               - np.median(B_abl[h][mN]))
            if (h + 1) % 8 == 0:
                log('ablation sweep [%d/32]' % (h + 1),
                    lines)
        a7_ok = bool(a7_max < 1e-6)
        log('a7 ablation efficacy max|C_abl34,15| %.2e '
            'ok=%s' % (a7_max, a7_ok), lines)

        R = gap_intact - gap_abl
        rank_R15 = int(np.argsort(-R).tolist()
                       .index(H_TGT)) + 1
        mean_dB = np.abs(B_abl - B_int[None, :]).mean(
            axis=1)

        # T3a: FL on B_abl[15]
        tid_test = np.array([tid_map[w] for w in wlist])
        x_fc = rankdata(tid_test.astype(np.float64))
        x_fc = (x_fc - x_fc.mean()) / x_fc.std()
        g_fc = mN.astype(np.float64)
        Xr = np.stack([np.ones(30), x_fc], axis=1)
        bb = B_abl[H_TGT]
        beta_r, *_ = np.linalg.lstsq(Xr, bb, rcond=None)
        resid = bb - Xr @ beta_r
        Xf = np.stack([np.ones(30), x_fc, g_fc], axis=1)

        def full_coef(v):
            beta, *_ = np.linalg.lstsq(Xf, v, rcond=None)
            return float(beta[2])

        obs_a = full_coef(bb)
        rng3 = np.random.default_rng(RNG_T3A)
        cnt3 = 0
        for _ in range(N_PERM):
            ep = rng3.permutation(resid)
            if abs(full_coef(Xr @ beta_r + ep)) \
                    >= abs(obs_a) - 1e-12:
                cnt3 += 1
        p3a = (cnt3 + 1) / (N_PERM + 1)
        abolished = bool(p3a > 0.05)
        t3 = {'gap_intact': round(gap_intact, 4),
              'gap_abl_h15': round(float(gap_abl[H_TGT]), 4),
              'R_h15': round(float(R[H_TGT]), 4),
              'R_rank_of_h15': rank_R15,
              'FL_coef_abl15': round(obs_a, 4),
              'p_fl_abl': float('%.3e' % p3a),
              'abolished': abolished,
              'mean_dB_h15': round(
                  float(mean_dB[H_TGT]), 4),
              'mean_dB_median_all': round(
                  float(np.median(mean_dB)), 4),
              'top3_R_heads': [
                  (int(h), round(float(R[h]), 4))
                  for h in np.argsort(-R)[:3]]}
        log('T3a FL coef %+.4f p %.3e abolished=%s | '
            'R15 %+.4f rank %d | mean|dB| %.4f vs med %.4f'
            % (obs_a, p3a, abolished, R[H_TGT], rank_R15,
               mean_dB[H_TGT], np.median(mean_dB)), lines)

        # ---------- verdict ----------
        if t2_pass and abolished:
            verdict = 'h15_necessary_class_carrier'
        elif t2_pass:
            verdict = 'h15_shared_carrier_effect_survives'
        elif abolished:
            verdict = 'h15_causal_correlation_mismatch'
        else:
            verdict = 'h15_correlational_weak'
        save = {'C34_intact': C34_int.astype(np.float32),
                'B_intact': B_int,
                'B_abl': B_abl.astype(np.float32),
                'gap_abl': gap_abl,
                'R': R, 'cos_pop': cos_pop,
                'tids': tid_test, 'labels': lab,
                'words': np.array(
                    ['%d:%s' % (lab[i], w)
                     for i, w in enumerate(wlist)])}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2965, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_ok': a1_ok,
                       'a2_diff': float('%.3e' % a2_diff),
                       'a2_ok': a2_ok,
                       'a3_rel': float('%.3e' % a3_rel),
                       'a3_ok': a3_ok,
                       'a4_ok': a4_ok,
                       'a5_diff': float('%.3e' % a5_diff),
                       'a5_ok': a5_ok,
                       'a6_rel': float('%.3e' % a6_rel),
                       'a6_ok': a6_ok,
                       'a7_max': float('%.3e' % a7_max),
                       'a7_ok': a7_ok,
                       'a8_ok': a8_ok,
                       'ok': bool(anchor_ok and a7_ok)},
           'T1': t1, 'T2': t2, 'T3': t3,
           'gap_intact': round(gap_intact, 4),
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void':
        np.savez_compressed(os.path.join(
            OUT, 'h15_identity.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2965 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
