# -*- coding: utf-8 -*-
"""Phase 2970 patch: T1 grouped-vectorized maxT + T3 dead code fix."""
import ast
import io

P = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase2970_delay_carrier_localization.py'
t = io.open(P, encoding='utf-8').read()

old1 = """        nvh = len(valid_h)
        t1_sig = False
        t1 = {'n_valid_heads': nvh,
              'validity_ok': bool(nvh >= NVH_MIN)}
        if nvh >= NVH_MIN:
            rng1 = np.random.default_rng(RNG_T1)
            # block-wise maxT
            obs_max = max(abs(d_head[k]) for k in valid_h)
            cnt_max = 0
            B_SZ = 200
            done = 0
            while done < N_PERM:
                bsz = min(B_SZ, N_PERM - done)
                for b in range(bsz):
                    sg = rng1.choice([-1.0, 1.0],
                                     size=(1, n_p2))[0]
                    worst = 0.0
                    for k in valid_h:
                        pk = pk_cache_h[k]
                        d = np.array([pk[fr] - pk[en]
                                      for en, fr in pairs_all
                                      if pk[en] is not None
                                      and pk[fr] is not None])
                        worst = max(worst, abs(
                            float((sg[:len(d)] * d)
                                  .mean())))
                    if worst >= obs_max:
                        cnt_max += 1
                done += bsz
            q_maxt = (cnt_max + 1) / (N_PERM + 1)
            t1_sig = bool(q_maxt < 0.05)
            t1.update({'q_maxT': float('%.4g' % q_maxt),
                       'obs_max_abs_d': round(float(obs_max),
                                              4),
                       'pass': t1_sig})"""
new1 = """        nvh = len(valid_h)
        t1_sig = False
        t1 = {'n_valid_heads': nvh,
              'validity_ok': bool(nvh >= NVH_MIN)}
        if nvh >= NVH_MIN:
            # precompute per-head paired d arrays, group by
            # pair count for vectorized maxT
            from collections import defaultdict
            d_arr = {}
            groups = defaultdict(list)
            for k in valid_h:
                pk = pk_cache_h[k]
                d = np.array([pk[fr] - pk[en]
                              for en, fr in pairs_all
                              if pk[en] is not None
                              and pk[fr] is not None])
                d_arr[k] = d
                groups[len(d)].append(k)
            rng1 = np.random.default_rng(RNG_T1)
            obs_max = max(abs(d_head[k]) for k in valid_h)
            joint = np.empty(N_PERM)
            B_SZ = 250
            done = 0
            while done < N_PERM:
                bsz = min(B_SZ, N_PERM - done)
                worst = np.zeros(bsz)
                sg = rng1.choice([-1.0, 1.0],
                                 size=(bsz, n_p2))
                for L, ks in groups.items():
                    D = np.stack([d_arr[k] for k in ks])
                    means = np.abs(
                        (sg[:, :L] * D[None, :, :])
                        .mean(axis=2))
                    worst = np.maximum(worst,
                                       means.max(axis=1))
                joint[done:done + bsz] = worst
                done += bsz
            q_maxt = (int((joint >= obs_max).sum()) + 1) \\
                / (N_PERM + 1)
            t1_sig = bool(q_maxt < 0.05)
            t1.update({'q_maxT': float('%.4g' % q_maxt),
                       'obs_max_abs_d': round(float(obs_max),
                                              4),
                       'pass': t1_sig})"""
assert old1 in t, 'T1 block not found'
t = t.replace(old1, new1, 1)

old2 = """        d3415 = d_head[LI_TGT * NH + H_TGT]
        d3415_diff = abs(float(d3415)
                         - float(r69['T2_paired_lang']
                                 ['mean_d_L_minus_en'])) \\
            if False else None
        # 2969 result not loaded above; load now
        r69 = json.load(open(os.path.join(
            BASE, 'phase2969', 'peak_word_attributes',
            'result.json'), encoding='utf-8'))
        d3415_diff = abs(float(d3415) - float(
            r69['T2_paired_lang']['mean_d_L_minus_en']))"""
new2 = """        d3415 = d_head[LI_TGT * NH + H_TGT]
        r69 = json.load(open(os.path.join(
            BASE, 'phase2969', 'peak_word_attributes',
            'result.json'), encoding='utf-8'))
        d3415_diff = abs(float(d3415) - float(
            r69['T2_paired_lang']['mean_d_L_minus_en']))"""
assert old2 in t, 'T3 block not found'
t = t.replace(old2, new2, 1)

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
res = []
try:
    ast.parse(t2)
    res.append('syntax OK')
except SyntaxError as e:
    res.append('SYNTAX ERROR line %s col %s: %s'
               % (e.lineno, e.offset, e.msg))
res.append('grouped maxT landed: %d'
           % t2.count('groups = defaultdict'))
res.append('dead code gone: %d'
           % (1 - t2.count('if False else None')))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\p2970_syntax.txt', 'w',
        encoding='utf-8').write('\n'.join(res) + '\n')
print('patched')
