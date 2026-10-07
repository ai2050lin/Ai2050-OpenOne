# -*- coding: utf-8 -*-
import io
import numpy as np
import traceback

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
NPZ = BASE + (r'\phase3040'
              r'\omega_p37_situational_component_qwen'
              r'\omega_p37_situational_component_qwen.npz')
NPZ37 = BASE + (r'\phase3037'
                r'\omega_p34_kv_situational_specificity_qwen'
                r'\omega_p34_kv_situational_specificity_qwen.npz')
OUTP = (r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\probe3040_result.txt')
L = []


def log(m):
    L.append(str(m))


try:
    z = np.load(NPZ, allow_pickle=True)
    V3 = z['V3']
    SIT3 = z['SIT3']
    cos = z['cos_sit3']
    sd = z['pair_same_prompt'].astype(bool)
    dw = z['pair_diff_word'].astype(bool)
    occ_w = z['occ_w']
    occ_tid = z['occ_tid']
    occ_pr = z['occ_prompt']
    n = len(V3)
    log('n_occ=%d n_pairs=%d' % (n, len(cos)))

    nrm_sit = np.linalg.norm(SIT3, axis=1)
    nrm_v = np.linalg.norm(V3, axis=1)
    log('SIT norms: min=%.3e med=%.3e max=%.3e '
        'n<1e-6: %d'
        % (nrm_sit.min(), float(np.median(nrm_sit)),
           nrm_sit.max(), int((nrm_sit < 1e-6).sum())))
    log('V norms: med=%.3f min=%.3f max=%.3f'
        % (float(np.median(nrm_v)), nrm_v.min(),
           nrm_v.max()))
    log('rel residual norm: med=%.4f min=%.4f max=%.4f'
        % (float(np.median(nrm_sit / nrm_v)),
           float((nrm_sit / nrm_v).min()),
           float((nrm_sit / nrm_v).max())))

    log('cos_sit3: n_exact_zero=%d '
        'quantiles[0,1,5,25,50,75,95,99,100]=%s'
        % (int((cos == 0.0).sum()),
           np.round(np.percentile(
               cos, [0, 1, 5, 25, 50, 75, 95, 99,
                     100]), 5).tolist()))
    log('|cos| med=%.5f  cos std=%.5f'
        % (float(np.median(np.abs(cos))),
           float(cos.std())))

    sdv = cos[sd]
    cdv = cos[dw]
    log('same-prompt diff-word cos: n=%d '
        'quantiles=%s med=%.6f'
        % (len(sdv),
           np.round(np.percentile(
               sdv, [0, 25, 50, 75, 100]),
               6).tolist(),
           float(np.median(sdv))))
    log('cross-prompt diff-word cos: n=%d med=%.6f'
        % (len(cdv), float(np.median(cdv))))

    # orthogonality of SIT to word-mean span
    n_types = int(occ_w.max()) + 1
    WM = np.zeros((n_types, V3.shape[1]))
    for wi in range(n_types):
        WM[wi] = V3[occ_w == wi].mean(axis=0)
    U, S, Vt = np.linalg.svd(WM,
                             full_matrices=False)
    r = int(np.sum(S > 1e-8 * S[0]))
    B = Vt[:r].T
    projerr = np.abs(SIT3 @ B).max()
    log('SIT@B max=%.3e (should be ~1e-6 fp32)'
        % float(projerr))

    # permutation null shape (2000 draws)
    rng = np.random.default_rng(777)
    pi_arr = z['pair_i']
    pj_arr = z['pair_j']
    obs = float(np.median(sdv)) - float(np.median(cdv))
    st = np.zeros(2000)
    for it in range(2000):
        pl = rng.permutation(occ_pr)
        sp = pl[pi_arr] == pl[pj_arr]
        ms = dw & sp
        st[it] = (float(np.median(cos[ms]))
                  - float(np.median(cos[dw & ~sp])))
    log('perm null (2000): mean=%.6f std=%.6f '
        'min=%.6f max=%.6f  obs=%.6f  P(>=obs)=%.4f'
        % (st.mean(), st.std(), st.min(), st.max(),
           obs, float(np.mean(st >= obs))))

    # sanity: FULL-V same-word cross-prompt cos for
    # the 8 logic targets vs 3037 (med ~0.85-0.98)
    z37 = np.load(NPZ37, allow_pickle=True)
    w37 = [str(x) for x in z37['occ_word']]
    log('3037 npz occ_word sample=%s' % w37[:5])
    tid2w = {}
    for oi in range(n):
        t = int(occ_tid[oi])
        if t not in tid2w:
            tid2w[t] = 'w%d' % t
    # use 3037 npz V3 directly (bit-same per a78)
    V37 = z37['V3']
    pr37 = np.array([int(x)
                     for x in z37['occ_prompt']])
    w37a = np.array(w37)
    same = []
    diff = []
    for i in range(len(w37)):
        for j in range(i + 1, len(w37)):
            if pr37[i] == pr37[j]:
                continue
            c = float(abs(np.dot(V37[i], V37[j]))
                      / max(np.linalg.norm(V37[i])
                            * np.linalg.norm(V37[j]),
                            1e-30))
            if w37[i] == w37[j]:
                same.append(c)
            else:
                diff.append(c)
    log('3037 targets FULL-V: same-word cross med=%.4f '
        '(n=%d) diff-word cross med=%.4f (n=%d) '
        'ratio=%.3f'
        % (float(np.median(same)), len(same),
           float(np.median(diff)), len(diff),
           float(np.median(same))
           / max(float(np.median(diff)), 1e-30)))

    # residual cos for the SAME target-word pairs
    # (SIT from 3040 npz restricted to target rows):
    # map target occ (word,prompt) -> 3040 row
    po37 = [int(x) for x in z37['occ_pos']]
    key2row = {}
    for oi in range(n):
        key2row[(int(occ_tid[oi]), int(occ_pr[oi]),
                 int(z['occ_pos'][oi]))] = oi
    tid_of_word = {}
    for t, w in tid2w.items():
        pass
    # rebuild tid->word via 3037 word_tok values
    wt = {'because': 1576, 'however': 4764,
          'while': 1393, 'although': 7892,
          'therefore': 8916, 'yet': 3602,
          'thus': 8450, 'so': 773}
    rows = []
    rw = []
    for i in range(len(w37)):
        k = (wt[w37[i]], pr37[i], po37[i])
        rows.append(key2row[k])
        rw.append(w37[i])
    rows = np.array(rows)
    rw = np.array(rw)
    SITt = SIT3[rows]
    prr = pr37
    s2 = []
    d2 = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if prr[i] == prr[j]:
                continue
            c = float(np.dot(SITt[i], SITt[j])
                      / max(np.linalg.norm(SITt[i])
                            * np.linalg.norm(SITt[j]),
                            1e-30))
            if rw[i] == rw[j]:
                s2.append(c)
            else:
                d2.append(c)
    log('targets SIT: same-word cross med=%.6f (n=%d) '
        'diff-word cross med=%.6f (n=%d)'
        % (float(np.median(s2)) if s2 else float('nan'),
           len(s2), float(np.median(d2)), len(d2)))
    msg = 'PROBE_OK'
except Exception:
    msg = 'PROBE_FAIL\n' + traceback.format_exc()
with io.open(OUTP, 'w', encoding='utf-8') as f:
    f.write(msg + '\n' + '\n'.join(L))
print(msg)
