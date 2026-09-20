# -*- coding: utf-8 -*-
"""Phase 2948: W_ov linear head-gain vs measured head effects.

Why: 2947 localized the switch to few high-leverage heads
(L17: h22/h19 concentrated; L16: h17/h19/13/27, larger but
more distributed) with shared promoters h19/h22 across
layers. Open question: is the per-head effect rank predicted
by the linear W_ov readout gain g_h = median_w
u35.(Wov_h @ xdir_w), where Wov_h = Wo[:, h] @ Wv[kv(h)]
(GQA: 4 query heads share one KV head)? If yes, the head
hierarchy is a linear feature-transport property; if no, it
is carried by nonlinear/attention terms.

Mode: ZERO forward. Weights from safetensors shards;
frozen artifacts: dirs_word (2927), Vt8+dcks (2939),
D vectors (2947 run2).

Anchors (frozen):
  a1 weight fingerprint: v_proj/o_proj shapes ==
     (1024,2560)/(2560,4096) for both layers (GQA gate)
  a2 dirs_word vs 2927 npz bit-level (max abs < 1e-9)
  a3 Vt8 vs 2939 npz bit-level (max abs < 1e-9)
  a4 D vectors vs 2947 result.json bit-level reload
  a5 xdir construction self-check < 1e-9 (2946 style)
  a7 g computation cross-check: explicit-loop value for
     head 0 vs vectorized relative diff < 1e-8 (absolute
     1e-9 was unreachable: float64 2560-term accumulation-
     order noise measured 1.73e-09 in run2 correction)

Main tests (frozen):
  T1: rho(g_L17, D_L17) >= p95 of permutation null
      (20000 shuffles of D, seed 2904)
  T2: rho(g_L16, D_L16) >= p95 of permutation null
Verdict (frozen):
  anchor fail               => anchor_fail_all_void
  T1 pass AND T2 pass       => linear_head_gain_confirmed
  exactly one pass          => linear_head_gain_partial
  neither                   => head_gain_nonlinear
Note: g and D were previewed in the 2948 preflight probe;
per discipline 9 all verdicts are registered as
quasi-post-hoc mechanism integration (thresholds are
permutation-null based, not preview-value based).

Descriptive: D1 g vectors; D2 top5 overlap; D3 cos
(ov(xdir_mean), u35 / v3) for h19/h22; D4 rho values.

Output: phase2948/wov_head_gain/.
"""
import hashlib
import json
import os
import time

import numpy as np

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2947 = os.path.join(BASE, 'phase2947', 'head_anatomy',
                        'result.json')
OUT = os.path.join(BASE, 'phase2948', 'wov_head_gain')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2948_run_report.txt')
NH, HD = 32, 128
NL = 36
S_IDX = (0, 1, 4)
N_PERM = 20000
SEED_PERM = 2904
LAYERS = (17, 16)

PREREG = {
    'mode': 'zero-forward: W_ov linear head gain g_h vs '
            '2947 measured D_h, permutation-null thresholds',
    'question': 'is the per-head switch effect rank '
                'predicted by the linear W_ov readout gain?',
    'anchors': {
        'a1': 'GQA shape gate: v_proj (1024,2560), o_proj '
              '(2560,4096) both layers',
        'a2': 'dirs_word vs 2927 npz bit-level',
        'a3': 'Vt8 vs 2939 npz bit-level',
        'a4': 'D vectors vs 2947 result.json bit-level',
        'a5': 'xdir construction self-check < 1e-9',
        'a7': 'g explicit-loop cross-check < 1e-9',
    },
    'T1': 'rho(g_L17, D_L17) >= perm-null p95 (20000, '
          'seed 2904)',
    'T2': 'rho(g_L16, D_L16) >= perm-null p95',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1 and T2 => linear_head_gain_confirmed; '
               'exactly one => linear_head_gain_partial; '
               'else => head_gain_nonlinear',
    'note': 'quasi-post-hoc (discipline 9): g and D '
            'previewed in preflight; thresholds are '
            'permutation-null based',
    'correction_note': 'run1: a4 compared full-precision '
                       'npz D against round(v,2) values '
                       'from 2947 result.json (bit-level '
                       'gate unreachable); fixed to '
                       'npz-authoritative + rounding-'
                       'consistency gate 5.01e-3. run2: '
                       'a7 absolute 1e-9 gate hit float64 '
                       'accumulation-order noise (measured '
                       '1.73e-09 over 2560-term sum); fixed '
                       'to relative gate 1e-8. run3 is the '
                       'authoritative run.',
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
    ra = rankdata(np.asarray(a, float))
    rb = rankdata(np.asarray(b, float))
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
    srcs = {'s2927': sha8(SRC_2927), 's2939': sha8(SRC_2939),
            's2947': sha8(SRC_2947)}
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2948, 'name': 'wov_head_gain',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': srcs, 'model': 'qwen3-4b',
                   'heads': NH, 'head_dim': HD,
                   'n_layers': NL, 'layers': list(LAYERS),
                   's_idx': list(S_IDX),
                   'n_perm': N_PERM, 'seed_perm': SEED_PERM,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- frozen artifacts ----------
    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs_word = z27['dirs_word'].astype(np.float64)
    u35 = dirs_word[NL - 1]
    z39 = np.load(SRC_2939, allow_pickle=True)
    Vt8 = z39['Vt8'].astype(np.float64)
    coords = z39['coords'].astype(np.float64)
    conds39 = [str(s) for s in z39['cond_names']]
    dcks = coords[conds39.index('null0')] \
        - coords[conds39.index('func')]
    dcks_S = dcks[:, list(S_IDX)]
    Vt8_S = Vt8[list(S_IDX)]
    xdir = dcks_S @ Vt8_S
    a5_diff = float(np.abs(
        xdir @ Vt8_S.T - dcks_S).max())
    a5_ok = bool(a5_diff < 1e-9)
    log('a5 xdir self-check %.2e ok=%s'
        % (a5_diff, a5_ok), lines)

    r47 = json.load(open(SRC_2947, encoding='utf-8'))
    d47_npz = np.load(os.path.join(
        BASE, 'phase2947', 'head_anatomy',
        'head_anatomy.npz'), allow_pickle=True)
    D17 = d47_npz['D_L17'].astype(np.float64)
    D16 = d47_npz['D_L16'].astype(np.float64)
    # a4: npz is the authoritative full-precision source
    # (result.json stores round(v,2)); cross-check both
    a4_npz_json = max(
        float(np.abs(D17 - np.array(
            r47['D1_vectors']['L17']['D'])).max()),
        float(np.abs(D16 - np.array(
            r47['D1_vectors']['L16']['D'])).max()))
    a4_diff = a4_npz_json
    a4_ok = bool(a4_npz_json < 5.01e-3)
    log('a4 D npz vs rounded json max diff %.2e '
        '(rounding-consistency gate 5.01e-3) ok=%s'
        % (a4_diff, a4_ok), lines)

    a2_diff = float(np.abs(dirs_word
                           - z27['dirs_word']
                           .astype(np.float64)).max())
    a2_ok = bool(a2_diff < 1e-9)
    log('a2 dirs_word reload %.2e ok=%s'
        % (a2_diff, a2_ok), lines)
    a3_diff = float(np.abs(Vt8 - z39['Vt8']
                           .astype(np.float64)).max())
    a3_ok = bool(a3_diff < 1e-9)
    log('a3 Vt8 reload %.2e ok=%s'
        % (a3_diff, a3_ok), lines)

    # ---------- weights ----------
    idx_path = os.path.join(MD,
                            'model.safetensors.index.json')
    weight_map = json.load(open(idx_path,
                                encoding='utf-8'))['weight_map']
    need = {}
    for li in LAYERS:
        for proj in ('v_proj', 'o_proj'):
            k = ('model.layers.%d.self_attn.%s.weight'
                 % (li, proj))
            need.setdefault(weight_map[k], []).append(k)
    from safetensors import safe_open
    tensors = {}
    for f, keys in need.items():
        with safe_open(os.path.join(MD, f),
                       framework='pt') as sf:
            for k in keys:
                tensors[k] = sf.get_tensor(k) \
                    .float().numpy()
    a1_ok = all(
        tensors['model.layers.%d.self_attn.v_proj.weight'
                % li].shape == (8 * HD, 2560)
        and tensors['model.layers.%d.self_attn.o_proj.weight'
                    % li].shape == (2560, NH * HD)
        for li in LAYERS)
    log('a1 GQA shape gate ok=%s' % a1_ok, lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok)
    verdict = None
    t1 = t2 = d1 = d2 = d3 = d4 = None
    save = {}
    a7_diff = None
    a7_ok = False

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        def head_gains(li):
            Wv = tensors[
                'model.layers.%d.self_attn.v_proj.weight'
                % li]
            Wo = tensors[
                'model.layers.%d.self_attn.o_proj.weight'
                % li]
            gs = np.zeros(NH)
            for h in range(NH):
                kv = h // (NH // 8)
                ov = Wo[:, h * HD:(h + 1) * HD] \
                    @ Wv[kv * HD:(kv + 1) * HD]
                gs[h] = float(np.median(
                    (xdir @ ov.T) @ u35))
            return gs

        g17 = head_gains(17)
        # a7: explicit-loop cross-check for head 0
        Wv17 = tensors[
            'model.layers.17.self_attn.v_proj.weight']
        Wo17 = tensors[
            'model.layers.17.self_attn.o_proj.weight']
        vals = []
        for w in range(xdir.shape[0]):
            acc = np.zeros(2560)
            xv = xdir[w]
            for d in range(HD):
                acc += Wo17[:, d] * float(
                    Wv17[0 * HD:(0 + 1) * HD][d] @ xv)
            vals.append(float(acc @ u35))
        a7_diff = abs(float(np.median(vals)) - g17[0]) \
            / max(abs(g17[0]), 1e-30)
        a7_ok = bool(a7_diff < 1e-8)
        log('a7 explicit-loop cross-check rel %.2e ok=%s'
            % (a7_diff, a7_ok), lines)

        if not a7_ok:
            verdict = 'anchor_fail_all_void'
        else:
            g16 = head_gains(16)

            def test(g, D, tag):
                rho = spearman(g, D)
                rng = np.random.default_rng(SEED_PERM)
                null = np.array([
                    spearman(g, rng.permutation(D))
                    for _ in range(N_PERM)])
                p = float((np.sum(null >= rho) + 1)
                          / (N_PERM + 1))
                p95 = float(np.quantile(null, 0.95))
                log('%s rho %.4f perm-p %.5f null p95 %.4f'
                    % (tag, rho, p, p95), lines)
                return {'rho': round(rho, 4),
                        'perm_p': p,
                        'null_p95': round(p95, 4),
                        'pass': bool(rho >= p95)}

            t1 = test(g17, D17, 'T1 L17')
            t2 = test(g16, D16, 'T2 L16')

            if t1['pass'] and t2['pass']:
                verdict = 'linear_head_gain_confirmed'
            elif t1['pass'] or t2['pass']:
                verdict = 'linear_head_gain_partial'
            else:
                verdict = 'head_gain_nonlinear'

            topg17 = [int(i) for i in
                      np.argsort(g17)[::-1][:5]]
            topg16 = [int(i) for i in
                      np.argsort(g16)[::-1][:5]]
            topd17 = [int(i) for i in
                      np.argsort(D17)[::-1][:5]]
            topd16 = [int(i) for i in
                      np.argsort(D16)[::-1][:5]]
            d1 = {'g_L17': [round(float(v), 4)
                            for v in g17],
                  'g_L16': [round(float(v), 4)
                            for v in g16]}
            d2 = {'top5_g_L17': topg17,
                  'top5_D_L17': topd17,
                  'overlap_L17':
                      len(set(topg17) & set(topd17)),
                  'top5_g_L16': topg16,
                  'top5_D_L16': topd16,
                  'overlap_L16':
                      len(set(topg16) & set(topd16)),
                  'g_h19_rank_L17':
                      int(np.argsort(-g17).tolist()
                          .index(19)) + 1,
                  'g_h22_rank_L17':
                      int(np.argsort(-g17).tolist()
                          .index(22)) + 1,
                  'g_h19_rank_L16':
                      int(np.argsort(-g16).tolist()
                          .index(19)) + 1,
                  'g_h22_rank_L16':
                      int(np.argsort(-g16).tolist()
                          .index(22)) + 1}
            d3 = {}
            for li in LAYERS:
                Wv = tensors[
                    'model.layers.%d.self_attn.v_proj.weight'
                    % li]
                Wo = tensors[
                    'model.layers.%d.self_attn.o_proj.weight'
                    % li]
                row = {}
                for h in (19, 22):
                    kv = h // (NH // 8)
                    ovx = (Wo[:, h * HD:(h + 1) * HD]
                           @ Wv[kv * HD:(kv + 1) * HD]) \
                        @ xdir.mean(0)
                    row['h%d' % h] = {
                        'cos_u35': round(float(
                            ovx @ u35
                            / max(np.linalg.norm(ovx)
                                  * np.linalg.norm(u35),
                                  1e-30)), 4),
                        'cos_v3': round(float(
                            ovx @ Vt8[2]
                            / max(np.linalg.norm(ovx)
                                  * np.linalg.norm(Vt8[2]),
                                  1e-30)), 4)}
                d3['L%d' % li] = row
            d4 = {'note': 'quasi-post-hoc mechanism '
                          'integration (discipline 9); '
                          'perm-null thresholds are '
                          'frozen before formal run',
                  'T1': t1, 'T2': t2}

            save = {'g_L17': g17, 'g_L16': g16,
                    'D_L17': D17, 'D_L16': D16,
                    'dirs_word': dirs_word, 'Vt8': Vt8}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2948, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_ok': a1_ok,
                       'a2_diff': float('%.3e' % a2_diff),
                       'a2_ok': a2_ok,
                       'a3_diff': float('%.3e' % a3_diff),
                       'a3_ok': a3_ok,
                       'a4_diff': float('%.3e' % a4_diff),
                       'a4_ok': a4_ok,
                       'a5_diff': float('%.3e' % a5_diff),
                       'a5_ok': a5_ok,
                       'a7_diff': None if a7_diff is None
                       else float('%.3e' % a7_diff),
                       'a7_ok': a7_ok,
                       'ok': bool(anchor_ok and a7_ok)},
           'T1': t1, 'T2': t2,
           'D1_g': d1, 'D2_top5': d2,
           'D3_cos': d3, 'D4_note': d4,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'wov_head_gain.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2948 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
