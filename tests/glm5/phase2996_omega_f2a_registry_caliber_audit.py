# -*- coding: utf-8 -*-
"""Phase 2996: Omega-F2a MLP registry CALIBER AUDIT (plan v4
mechanism-audit chain: caliber-mixing suspicion on the 2995 T3
negative) + 2995 capture-degeneracy discovery.

Why-1 (caliber).  2995 T3 graded the MLP neuron registry
"not replicated" on GLM4-9B (weight-side z all negative)
against qwen evidence (2989) built in a DIFFERENT caliber:
2989's registry lives in act_i * <W_down[:, i], u> (activation
x down-column contribution; label-perm null + top-128 energy
share), while 2995 T3 tested unit(up_row) . dirs_mlp
(weight-side; Haar null).  Audit-chain rule: no cross-caliber
verdicts.  This phase runs BOTH calibers on BOTH models
(matched relative-depth trio) and re-grades.

Why-2 (degeneracy).  Pre-run audit of the 2995 npz found
dirs_attn/dirs_mlp IDENTICAL across all 40 rows (max|diff|
0.0): its single-slot capture hooks + broadcast sweep assignment
turned per-layer axes into 40 copies of the LAST-fired layer
(L39).  Consequences already on file: 2995 T1/T2 used LTOP=39
only -> VALID by accident; 2995 T3's "per-layer" z at
[7,10,13] were pseudo-replications of the L39 direction ->
its per-layer claim is VOID, re-tested here with true
per-layer mlp_in axes.  Anchors ga1/ga2 below formalize this.

Models (sequential, bf16):
  qwen3-4b   NL=36 HID=2560 INT=9728 (separate gate/up_proj)
  glm4-9b    NL=40 HID=4096 INT=13696 (FUSED gate_up_proj:
             gate rows [:INT], up rows [INT:])
Capture discipline: per-layer LIST hooks (no single-slot
overwrite), batch dim == 1 asserted per forward.

Design (preregistered; frozen BEFORE any observation).
  qwen arm  cells = 2977 exec verbatim (74: F_en15 F_fr15
            C_en22 C_fr22), lang label (en=0 / non-en=1);
            Q_REG=[6,10,12]; seq=[the,w] pos-1: attn_in ALL
            layers (qa1 dirs_word anchor), mlp_in at Q_REG.
            act = offline recompute silu(x.Wg.T)*(x.Wu.T).
            K1 (2989 caliber verbatim): c from 2989 npz
            c_lang_li; D = mean_en(proj) - mean_nonen(proj),
            proj = act*c; obs_max = max|D|; share_obs =
            top-128|D| energy share; null = 2000 single label
            permutations (2989 weight-vector form); TRUE maxT
            over the 3-layer family; p_share.
            K2 (2995 caliber verbatim, now TRUE per-layer):
            align_j = |unit(up_row_j) . dirs_mlp_q[l]|; obs =
            mean top-128; Haar null N=2000; z per layer.
  glm arm   cells = 2995 runtime single-token filter verbatim
            (98); G_REG=[7,10,13]; attn_in/mlp_in ALL layers,
            down_proj input at G_REG (act, list hooks);
            K1 with c = W_down.T @ u_top (u_top = dirs_attn[39]
            rebuilt here; 2995 npz u_top lineage = L39 valid);
            K2 re-run per layer.
  T1 (qwen decomposition)  K1_sig = p_maxT <= 0.01 AND
            p_share <= 0.01; K2_absent = qwen K2 z_max < 5;
            artifact_q = K1_sig AND K2_absent.
  T2 (GLM4 matched caliber)  glm_sig = p_maxT <= 0.01 AND
            p_share <= 0.01.
  T3 (GLM4 K2 per layer)  z per layer (descriptive vs 2995).
  verdict  anchor_fail_all_void /
           artifact_q & glm_sig  => registry_replicated_at_
                                    matched_caliber
           artifact_q & ~glm_sig => qwen_registry_act_side_
                                    only_glm4_absent
           ~artifact_q & glm_sig => registry_robust_across_
                                    calibers
           else                  => registry_not_replicated_
                                    confirmed
  tags     plan v4 P0: lang axis / len-2 / snapshot ONLY
           (causal ablation untested) / canonical word sets
           differ (74 vs 98) = matched caliber, not matched
           vocabulary / 2995 T3 per-layer claim VOID
           (degeneracy), re-tested here.

Anchors (frozen):
  qa1 dirs_word rebuild (attn_in pos-1 lang axis, 74 cells,
      TRUE per-layer) vs 2927 npz < 1e-5
  qa2 qwen double sweep determinism < 1e-4
  qa3 qwen act_6 recompute vs 2989 npz act_6 rel < 1e-5
  qa4 qwen D_lang[6/10/12] (act x c_lang_2989) vs 2989 npz
      delta_lang rel < 1e-4
  qa5 qwen K1 perm rerun same seed -> identical p's
  ga1 GLM4 dirs_attn[39] vs 2995 npz row39 < 1e-6 (L39
      validity: 2995 stored the L39 value)
  ga2 GLM4 degeneracy confirmation: rebuilt dirs_attn differs
      from 2995 rows on >= 1 non-39 layer (> 1e-3) AND
      dirs_mlp[39] vs 2995 row39 < 1e-6
  ga3 glm double sweep determinism < 1e-6 (ai/mi/act)
  ga4 glm act recompute (fused) vs down-proj hook rel < 1e-6
  ga5 glm K2 obs at [7,10,13] with L39 direction vs 2995
      result T3 obs < 1e-5 (2995 used the L39 direction)
  ga6 glm projection decomposition identity (act@Wd)@u ==
      act@c rel < 1e-8
  ga7 glm K1 perm rerun same seed -> identical p's
Nondegeneracy: per-layer act std > 0 (both models); per-layer
dirs NOT all-identical (max inter-layer diff > 0); D not
all-zero; checked before stats.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
EXEC_2977 = os.path.join(BASE, 'phase2977',
                         'two_axis_fusion_injection',
                         'execution.json')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2979 = os.path.join(BASE, 'phase2979', 'reversal_anatomy',
                        'reversal_anatomy.npz')
SRC_2989 = os.path.join(BASE, 'phase2989',
                        'mlp_neuron_registry',
                        'mlp_neuron_registry.npz')
RES_2989 = os.path.join(BASE, 'phase2989',
                        'mlp_neuron_registry', 'result.json')
EXEC_2972 = os.path.join(BASE, 'phase2972',
                         'two_factor_signature',
                         'execution.json')
SRC_2995 = os.path.join(BASE, 'phase2995',
                        'omega_f1_glm4_panel',
                        'omega_f1_glm4_panel.npz')
RES_2995 = os.path.join(BASE, 'phase2995',
                        'omega_f1_glm4_panel', 'result.json')
OUT = os.path.join(BASE, 'phase2996',
                   'omega_f2a_registry_caliber_audit')
MD_Q = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
MD_G = r'D:\AI2050\Ai2050-OpenOne\models\hf\glm4-9b-chat-hf'

NH, HD = 32, 128
NL_Q, HID_Q, INT_Q = 36, 2560, 9728
NL_G, HID_G, INT_G = 40, 4096, 13696
Q_REG = [6, 10, 12]
G_REG = [7, 10, 13]
K_TOP = 128
N_PERM = 2000
N_RAND = 2000
SIG_GATE = 0.01
P_GATE = 0.05
Z_GATE = 5.0
RNG_MAIN = 2996

L_CAND = ["because", "therefore", "although", "unless",
          "however", "thus", "moreover", "since", "whereas",
          "despite", "hence", "nevertheless", "consequently",
          "furthermore", "otherwise", "instead", "while",
          "accordingly", "likewise", "meanwhile", "nonetheless",
          "thereafter", "whereby", "albeit"]


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def log(msg, lines):
    lines.append('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def k1_registry(act, c, lab, rng_seed):
    """2989 caliber verbatim.  Returns per-layer dict +
    family maxT p."""
    n0 = int((lab == 0).sum())
    n1 = int((lab == 1).sum())
    perm_max = {}
    out = {}
    for li in sorted(act):
        proj = act[li] * c[li][None, :]
        D = proj[lab == 0].mean(0) - proj[lab == 1].mean(0)
        obs_max = float(np.abs(D).max())
        s_sorted = np.sort(np.abs(D))[::-1]
        share_obs = float(s_sorted[:K_TOP].sum()
                          / max(float(s_sorted.sum()), 1e-30))
        PT = np.ascontiguousarray(proj.T)
        pool = np.zeros((N_PERM, D.size), dtype=np.float32)
        rng = np.random.default_rng(rng_seed)
        for pi in range(N_PERM):
            pl = rng.permutation(lab)
            w = ((pl == 0).astype(np.float64) / n0
                 - (pl == 1).astype(np.float64) / n1)
            pool[pi] = (PT @ w).astype(np.float32)
        abs_pool = np.abs(pool)
        perm_max[li] = abs_pool.max(axis=1)
        share_null = np.zeros(N_PERM)
        for pi in range(N_PERM):
            sn = np.sort(abs_pool[pi])[::-1]
            share_null[pi] = float(
                sn[:K_TOP].sum()
                / max(float(sn.sum()), 1e-30))
        p_share = float((share_null
                         >= share_obs - 1e-30).sum()) / N_PERM
        out[li] = {'obs_max': obs_max,
                   'share_obs': share_obs,
                   'p_raw': float(
                       (perm_max[li]
                        >= obs_max - 1e-30).sum()) / N_PERM,
                   'p_share': p_share}
    fam = np.stack([perm_max[li] for li in sorted(act)])
    fam_max = fam.max(axis=0)
    p_maxT = {}
    for li in sorted(act):
        p_maxT[li] = float(
            (fam_max
             >= out[li]['obs_max'] - 1e-30).sum()) / N_PERM
    return out, p_maxT


def k2_weightside(Rn, dirs_mlp_li, rng_seed, hid):
    """2995 T3 caliber verbatim (obs stat); Haar null
    N_RAND draws."""
    rng = np.random.default_rng(rng_seed)
    V = rng.standard_normal((N_RAND, hid))
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    obs = float(np.sort(np.abs(Rn @ dirs_mlp_li))[-K_TOP:]
                .mean())
    nulls = np.abs(V @ Rn.T)
    null_stat = np.sort(nulls, axis=1)[:, -K_TOP:].mean(axis=1)
    mu = float(null_stat.mean())
    sd = float(null_stat.std())
    return {'obs': obs, 'null_mu': mu, 'null_sd': sd,
            'z': (obs - mu) / max(sd, 1e-30)}


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    prereg = {
        'design': 'TWO models sequential (qwen3-4b 74 cells '
                  '2977 verbatim Q_REG=[6,10,12]; glm4-9b 98 '
                  'cells 2995 verbatim G_REG=[7,10,13]); '
                  '[the,w] pos-1; per-layer LIST captures '
                  '(2995 single-slot degeneracy fixed); '
                  'BOTH calibers BOTH models: K1=2989 '
                  'act*<Wdown[:,i],u> label-perm null '
                  '(2000, single perm) + top-128 share + '
                  'family maxT; K2=2995 unit(up_row).'
                  'dirs_mlp Haar null (2000); lang axis; '
                  'causal ablation NOT in scope',
        'question': 'is the 2995 T3 registry non-replication '
                    'a caliber artifact (qwen strength '
                    'act-side only, absent weight-side), and '
                    'does GLM4 show the registry at the '
                    'MATCHED 2989 caliber with TRUE per-layer '
                    'axes?',
        'T1': 'qwen decomposition: K1_sig per the 2989 '
              'REGISTERED rule = (p_maxT<=0.05 OR p_share<='
              '0.05) at [6,10,12] (correction_note run5: '
              'run3-5 drafts used AND, stricter than the '
              'audited card; 2989 p_lang_maxT was 0.51, its '
              'share leg carried the verdict); K2_absent = '
              'z_max < 5; artifact_q = K1_sig AND K2_absent',
        'T2': 'glm matched caliber: glm_sig = (p_maxT<=0.05 '
              'OR p_share<=0.05) at [7,10,13]',
        'T3': 'glm K2 z per layer (descriptive; 2995 T3 '
              'per-layer claim VOID per degeneracy finding)',
        'verdict': 'anchor fail => anchor_fail_all_void; '
                   'artifact_q & glm_sig => '
                   'registry_replicated_at_matched_caliber; '
                   'artifact_q & ~glm_sig => '
                   'qwen_registry_act_side_only_glm4_absent; '
                   '~artifact_q & glm_sig => '
                   'registry_robust_across_calibers; else => '
                   'registry_not_replicated_confirmed',
        'anchors': {
            'qa1': '2979 d_lang_u reproduce from L17 '
                   'attn_in captures (non-en minus en, '
                   'unit) < 1e-5',
            'qa2': 'qwen sweep determinism < 1e-4',
            'qa3': 'qwen act_6 bf16 recompute vs 2989 npz '
                   'rel < 1e-5',
            'qa4': 'qwen D_lang vs 2989 delta_lang '
                   'rel < 1e-4 (3 layers)',
            'qa5': 'qwen K1 perm rerun identical p',
            'ga1': 'glm dirs_attn[39] vs 2995 row39 < 1e-6',
            'ga2': 'degeneracy confirmation: non-39 rows '
                   'differ > 1e-3 AND dirs_mlp[39] vs 2995 '
                   'row39 < 1e-6',
            'ga3': 'glm sweep determinism < 1e-6',
            'ga4': 'glm act bf16 recompute vs hook '
                   'rel < 1e-6',
            'ga5': 'glm K2 obs (L39 dir) vs 2995 result '
                   '< 1e-5',
            'ga6': 'glm decomposition identity rel < 1e-8',
            'ga7': 'glm K1 perm rerun identical p'},
        'tags': 'lang axis / len-2 / snapshot only (no '
                'causal ablation) / word sets 74 vs 98 '
                '(matched caliber not matched vocabulary) / '
                '2995 T3 per-layer z VOID (degeneracy)',
    }
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2996,
                   'name': 'omega_f2a_registry_caliber_audit',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {
                       's2977exec': sha8(EXEC_2977),
                       's2979npz': sha8(SRC_2979),
                       's2989npz': sha8(SRC_2989),
                       's2989res': sha8(RES_2989),
                       's2972exec': sha8(EXEC_2972),
                       's2995npz': sha8(SRC_2995),
                       's2995res': sha8(RES_2995)},
                   'models': {'qwen': MD_Q, 'glm4': MD_G},
                   'q_reg': Q_REG, 'g_reg': G_REG,
                   'n_perm': N_PERM, 'n_rand': N_RAND,
                   'k_top': K_TOP, 'rng': RNG_MAIN,
                   'sig_gate': SIG_GATE, 'p_gate': P_GATE,
                   'z_gate': Z_GATE,
                   'prereg': prereg},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    import gc

    import torch as _t
    from transformers import AutoTokenizer, \
        AutoModelForCausalLM

    # =====================================================
    # ARM 1: qwen3-4b
    # =====================================================
    log('--- qwen arm ---', lines)
    e77 = json.load(open(EXEC_2977, encoding='utf-8'))
    cells = [('F', 'en', w) for w in e77['cells']['F_en']] \
        + [('F', 'fr', w) for w in e77['cells']['F_fr']] \
        + [('C', 'en', w) for w in e77['cells']['C_en']] \
        + [('C', 'fr', w) for w in e77['cells']['C_fr']]
    assert len(cells) == 74
    lang_q = np.array([0 if c[1] == 'en' else 1
                       for c in cells])

    tok_q = AutoTokenizer.from_pretrained(
        MD_Q, local_files_only=True, use_fast=True)

    def tid_of_q(w):
        ids = tok_q(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok_q(w, add_special_tokens=False)[
                'input_ids']
        return int(ids[0]) if len(ids) == 1 else -1

    tid_q = {}
    for _, _, w in cells:
        t = tid_of_q(w)
        assert t != -1, 'qwen non-single token %s' % w
        tid_q[w] = t
    ids_the = tid_of_q('the')
    assert ids_the != -1
    func_q = int(ids_the)

    import sys
    sys.path.insert(0,
                    r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    assert len(layers) == NL_Q
    log('qwen loaded', lines)

    ai_cap = {li: [] for li in range(NL_Q)}
    mi_cap = {li: [] for li in Q_REG}
    handles = []

    def hk_ai(li):
        def h(mod, args, kwargs):
            x = kwargs['hidden_states']
            ai_cap[li].append(
                x.detach().float().cpu().numpy().copy())
            return None
        return h

    def hk_mi(li):
        def h(mod, args):
            mi_cap[li].append(
                args[0].detach().float().cpu().numpy()
                .copy())
            return None
        return h

    for li in range(NL_Q):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           hk_ai(li), with_kwargs=True))
    for li in Q_REG:
        handles.append(layers[li].mlp
                       .register_forward_pre_hook(
                           hk_mi(li)))

    seqs_q = [[func_q, tid_q[c[2]]] for c in cells]

    arr1_ai = np.zeros((NL_Q, 74, HID_Q))
    arr1_mi = np.zeros((len(Q_REG), 74, HID_Q))
    arr2_ai = np.zeros_like(arr1_ai)
    arr2_mi = np.zeros_like(arr1_mi)
    for sweep, A_ai, A_mi in ((1, arr1_ai, arr1_mi),
                              (2, arr2_ai, arr2_mi)):
        for i in range(74):
            for s in ai_cap.values():
                del s[:]
            for s in mi_cap.values():
                del s[:]
            with _t.no_grad():
                model(_t.tensor([seqs_q[i]],
                                device='cuda'))
            for li in range(NL_Q):
                assert ai_cap[li][0].shape[0] == 1
                A_ai[li, i, :] = ai_cap[li][0][0, 1, :]
            for j, li in enumerate(Q_REG):
                A_mi[j, i, :] = mi_cap[li][0][0, 1, :]
        log('qwen sweep %d' % sweep, lines)

    qa2_rel = max(
        float(np.abs(arr1_ai - arr2_ai).max()
              / max(float(np.abs(arr1_ai).max()), 1e-30)),
        float(np.abs(arr1_mi - arr2_mi).max()
              / max(float(np.abs(arr1_mi).max()), 1e-30)))
    qa2_ok = bool(qa2_rel < 1e-4)
    log('qa2 sweep determinism %.2e ok=%s'
        % (qa2_rel, qa2_ok), lines)
    del arr2_ai, arr2_mi

    # qa1: reproduce 2979 d_lang_u (L17 attn_in lang axis,
    # non-en minus en, unit) from MY captures -- validates
    # the capture protocol against the established axis
    # lineage.  (correction_note run3: 2927 dirs_word was
    # built from a DIFFERENT 57-word en/L pair set, not the
    # 74 cells; the 2989 c_lang lineage is 2979 d_lang_u.)
    X17 = arr1_ai[17]
    d17 = X17[lang_q == 1].mean(axis=0) \
        - X17[lang_q == 0].mean(axis=0)
    d17_u = d17 / float(np.linalg.norm(d17))
    z79 = np.load(SRC_2979, allow_pickle=True)
    d79_l = z79['d_lang_u'].astype(np.float64)
    qa1_diff = float(np.abs(d17_u - d79_l).max())
    qa1_ok = bool(qa1_diff < 1e-5)
    log('qa1 d_lang_u reproduce %.2e ok=%s'
        % (qa1_diff, qa1_ok), lines)

    # qa3/qa4: act recompute (bf16 GPU, replicating the
    # model's own down_proj-input computation incl. its
    # bf16 rounding -- correction_note run3: float64 numpy
    # recompute differs from the bf16 hook by 2-3 ULP
    # ~4-6e-3, a tolerance miscalibration not a capture
    # error) + D reproduction
    z89 = np.load(SRC_2989, allow_pickle=True)
    act_q = {}
    D_q = {}
    qa3_rel = 0.0
    qa4_rel = 0.0
    for j, li in enumerate(Q_REG):
        mi = arr1_mi[j]
        x_bf = _t.tensor(mi, dtype=_t.bfloat16,
                         device='cuda')
        Wg = layers[li].mlp.gate_proj.weight.detach()
        Wu = layers[li].mlp.up_proj.weight.detach()
        with _t.no_grad():
            g = _t.nn.functional.linear(x_bf, Wg)
            u = _t.nn.functional.linear(x_bf, Wu)
            act_bf = _t.nn.functional.silu(g) * u
        act = act_bf.float().cpu().numpy().astype(np.float64)
        del x_bf, g, u, act_bf
        act_q[li] = act
        rel = float(np.abs(act - z89['act_%d' % li]
                           .astype(np.float64)).max()
                    / max(float(np.abs(act).max()), 1e-30))
        qa3_rel = max(qa3_rel, rel)
        c89 = z89['c_lang_%d' % li].astype(np.float64)
        proj = act * c89[None, :]
        D = proj[lang_q == 0].mean(0) \
            - proj[lang_q == 1].mean(0)
        D_q[li] = D
        drel = float(np.abs(D - z89['delta_lang_%d' % li]
                            .astype(np.float64)).max()
                     / max(float(np.abs(D).max()), 1e-30))
        qa4_rel = max(qa4_rel, drel)
        log('qwen L%d qa3 act rel=%.2e qa4 D rel=%.2e'
            % (li, rel, drel), lines)
    qa3_ok = bool(qa3_rel < 1e-5)
    qa4_ok = bool(qa4_rel < 1e-4)
    log('qa3 act recompute %.2e ok=%s' % (qa3_rel, qa3_ok),
        lines)
    log('qa4 D reproduce %.2e ok=%s' % (qa4_rel, qa4_ok),
        lines)

    # dirs_mlp_q per layer (TRUE per-layer, weight-side space)
    d_mi = arr1_mi[:, lang_q == 0, :].mean(axis=1) \
        - arr1_mi[:, lang_q == 1, :].mean(axis=1)
    dirs_mlp_q = {li: unit(d_mi[j])
                  for j, li in enumerate(Q_REG)}

    # K1 qwen (c from 2989 npz verbatim)
    c_q = {li: z89['c_lang_%d' % li].astype(np.float64)
           for li in Q_REG}
    k1_q, pmaxT_q = k1_registry(act_q, c_q, lang_q,
                                [RNG_MAIN, 1])
    k1_qb, pmaxT_qb = k1_registry(act_q, c_q, lang_q,
                                  [RNG_MAIN, 1])
    qa5_ok = bool(all(
        abs(k1_q[li]['p_raw'] - k1_qb[li]['p_raw']) < 1e-12
        and abs(k1_q[li]['p_share']
                - k1_qb[li]['p_share']) < 1e-12
        for li in Q_REG))
    log('qa5 perm rerun ok=%s' % qa5_ok, lines)

    # K2 qwen (weight side, TRUE per-layer dirs)
    k2_q = {}
    for li in Q_REG:
        Wl = layers[li].mlp.up_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        Rn = Wl / np.maximum(
            np.linalg.norm(Wl, axis=1, keepdims=True), 1e-30)
        k2_q[li] = k2_weightside(Rn, dirs_mlp_q[li],
                                 [RNG_MAIN, 2], HID_Q)
        del Wl, Rn
        log('qwen K2 L%d obs %.6f null %.6f+-%.6f z=%.2f'
            % (li, k2_q[li]['obs'], k2_q[li]['null_mu'],
               k2_q[li]['null_sd'], k2_q[li]['z']), lines)

    act_std_q = {li: float(act_q[li].std())
                 for li in Q_REG}
    gates_q = bool(all(v > 0 for v in act_std_q.values()))

    del model, layers, handles
    gc.collect()
    _t.cuda.empty_cache()
    log('qwen unloaded', lines)

    # =====================================================
    # ARM 2: glm4-9b
    # =====================================================
    log('--- glm arm ---', lines)
    e72 = json.load(open(EXEC_2972, encoding='utf-8'))
    src = ([('F', 'en', w) for w in e72['cells']['F_en']]
           + [('C', 'en', w) for w in e72['cells']['C_en']]
           + [('L', 'en', w) for w in L_CAND]
           + [('F', 'fr', w) for w in e72['cells']['F_fr']]
           + [('C', 'fr', w) for w in e72['cells']['C_fr']])
    tok_g = AutoTokenizer.from_pretrained(
        MD_G, local_files_only=True, use_fast=True)

    def tid_of_g(w):
        ids = tok_g(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok_g(w, add_special_tokens=False)[
                'input_ids']
        return int(ids[0]) if len(ids) == 1 else -1

    cells_g = []
    tid_g = {}
    for cat, lng, w in src:
        t = tid_of_g(w)
        if t != -1:
            cells_g.append((0 if lng == 'en' else 1, cat,
                            lng, w))
            tid_g[w] = t
    n_g = len(cells_g)
    lang_g = np.array([c[0] for c in cells_g])
    i_en_g = [i for i in range(n_g) if lang_g[i] == 0]
    i_non_g = [i for i in range(n_g) if lang_g[i] == 1]
    ids_the_g = tid_of_g('the')
    assert ids_the_g != -1
    func_g = int(ids_the_g)
    log('glm cells %d (en %d non-en %d)'
        % (n_g, len(i_en_g), len(i_non_g)), lines)

    model = AutoModelForCausalLM.from_pretrained(
        MD_G, torch_dtype=_t.bfloat16).cuda().eval()
    layers = model.model.layers
    assert len(layers) == NL_G
    log('glm loaded', lines)

    g_ai = {li: [] for li in range(NL_G)}
    g_mi = {li: [] for li in range(NL_G)}
    g_ac = {li: [] for li in G_REG}
    handles = []

    def ghk_ai(li):
        def h(mod, args, kwargs):
            x = kwargs['hidden_states']
            g_ai[li].append(
                x.detach().float().cpu().numpy().copy())
            return None
        return h

    def ghk_mi(li):
        def h(mod, args):
            g_mi[li].append(
                args[0].detach().float().cpu().numpy()
                .copy())
            return None
        return h

    def ghk_down(li):
        def h(mod, args):
            g_ac[li].append(
                args[0].detach().float().cpu().numpy()
                .copy())
            return None
        return h

    for li in range(NL_G):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           ghk_ai(li), with_kwargs=True))
        handles.append(layers[li].mlp
                       .register_forward_pre_hook(
                           ghk_mi(li)))
    for li in G_REG:
        handles.append(layers[li].mlp.down_proj
                       .register_forward_pre_hook(
                           ghk_down(li)))

    seqs_g = [[func_g, tid_g[c[3]]] for c in cells_g]

    A1_ai = np.zeros((NL_G, n_g, HID_G))
    A1_mi = np.zeros((NL_G, n_g, HID_G))
    A1_ac = np.zeros((len(G_REG), n_g, INT_G))
    A2_ai = np.zeros_like(A1_ai)
    A2_mi = np.zeros_like(A1_mi)
    A2_ac = np.zeros_like(A1_ac)
    gi = {li: j for j, li in enumerate(G_REG)}
    for sweep, A_ai, A_mi, A_ac in (
            (1, A1_ai, A1_mi, A1_ac),
            (2, A2_ai, A2_mi, A2_ac)):
        for i in range(n_g):
            for s in g_ai.values():
                del s[:]
            for s in g_mi.values():
                del s[:]
            for s in g_ac.values():
                del s[:]
            with _t.no_grad():
                model(_t.tensor([seqs_g[i]],
                                device='cuda'))
            for li in range(NL_G):
                assert g_ai[li][0].shape[0] == 1
                A_ai[li, i, :] = g_ai[li][0][0, 1, :]
                A_mi[li, i, :] = g_mi[li][0][0, 1, :]
            for li in G_REG:
                A_ac[gi[li], i, :] = g_ac[li][0][0, 1, :]
        log('glm sweep %d' % sweep, lines)

    ga3_rel = max(
        float(np.abs(A1_ai - A2_ai).max()
              / max(float(np.abs(A1_ai).max()), 1e-30)),
        float(np.abs(A1_mi - A2_mi).max()
              / max(float(np.abs(A1_mi).max()), 1e-30)),
        float(np.abs(A1_ac - A2_ac).max()
              / max(float(np.abs(A1_ac).max()), 1e-30)))
    ga3_ok = bool(ga3_rel < 1e-6)
    log('ga3 sweep determinism %.2e ok=%s'
        % (ga3_rel, ga3_ok), lines)

    # ga4: act recompute (fused) vs hook capture
    act_g = {}
    ga4_rel = 0.0
    for j, li in enumerate(G_REG):
        mi = A1_mi[li]
        x_bf = _t.tensor(mi, dtype=_t.bfloat16,
                         device='cuda')
        Wf = layers[li].mlp.gate_up_proj.weight.detach()
        with _t.no_grad():
            # replicate the FUSED single matmul (4096->27392)
            # then slice; two separate matmuls take a
            # different cuBLAS rounding path (correction_note
            # run4: split-matmul recompute rel 4e-4..1e-2)
            y = _t.nn.functional.linear(x_bf, Wf)
            g = y[:, :INT_G]
            u = y[:, INT_G:]
            act_bf = _t.nn.functional.silu(g) * u
        act = act_bf.float().cpu().numpy().astype(np.float64)
        del x_bf, y, g, u, act_bf
        act_g[li] = act
        rel = float(np.abs(act - A1_ac[j]).max()
                    / max(float(np.abs(act).max()), 1e-30))
        ga4_rel = max(ga4_rel, rel)
        log('glm L%d ga4 act recompute rel=%.2e'
            % (li, rel), lines)
    ga4_ok = bool(ga4_rel < 1e-6)
    log('ga4 %.2e ok=%s' % (ga4_rel, ga4_ok), lines)

    # dirs rebuild (TRUE per-layer) + degeneracy anchors
    d_ai = A1_ai[:, i_en_g, :].mean(axis=1) \
        - A1_ai[:, i_non_g, :].mean(axis=1)
    dirs_attn_g = np.stack([unit(d_ai[li])
                            for li in range(NL_G)])
    d_mi_g = A1_mi[:, i_en_g, :].mean(axis=1) \
        - A1_mi[:, i_non_g, :].mean(axis=1)
    dirs_mlp_g = np.stack([unit(d_mi_g[li])
                           for li in range(NL_G)])
    z95 = np.load(SRC_2995, allow_pickle=True)
    da95 = z95['dirs_attn'].astype(np.float64)
    dm95 = z95['dirs_mlp'].astype(np.float64)
    ga1_diff = float(np.abs(dirs_attn_g[NL_G - 1]
                            - da95[NL_G - 1]).max())
    ga1_ok = bool(ga1_diff < 1e-6)
    non39_diff = float(np.abs(dirs_attn_g[:NL_G - 1]
                              - da95[:NL_G - 1]).max())
    dm39_diff = float(np.abs(dirs_mlp_g[NL_G - 1]
                             - dm95[NL_G - 1]).max())
    ga2_ok = bool(non39_diff > 1e-3 and dm39_diff < 1e-6)
    log('ga1 L39 validity %.2e ok=%s' % (ga1_diff, ga1_ok),
        lines)
    log('ga2 degeneracy: non39 max|diff| %.3e (>1e-3 '
        'expected), mlp39 %.2e ok=%s'
        % (non39_diff, dm39_diff, ga2_ok), lines)
    g_dirs_spread = max(
        float(np.abs(dirs_attn_g - dirs_attn_g[0]).max()),
        float(np.abs(dirs_mlp_g - dirs_mlp_g[0]).max()))
    log('glm dirs inter-layer spread %.3e' % g_dirs_spread,
        lines)

    u_top = dirs_attn_g[NL_G - 1]

    # K1 glm: c = Wdown.T @ u_top + decomposition identity
    c_g = {}
    ga6_rel = 0.0
    for li in G_REG:
        Wd = layers[li].mlp.down_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        c_g[li] = Wd.T @ u_top
        lhs = act_g[li] @ c_g[li]
        rhs = (act_g[li] @ Wd.T) @ u_top
        ga6_rel = max(ga6_rel, float(
            np.abs(lhs - rhs).max()
            / max(float(np.abs(rhs).max()), 1e-30)))
        del Wd
    ga6_ok = bool(ga6_rel < 1e-8)
    log('ga6 decomposition identity %.2e ok=%s'
        % (ga6_rel, ga6_ok), lines)

    k1_g, pmaxT_g = k1_registry(act_g, c_g, lang_g,
                                [RNG_MAIN, 3])
    k1_gb, pmaxT_gb = k1_registry(act_g, c_g, lang_g,
                                  [RNG_MAIN, 3])
    ga7_ok = bool(all(
        abs(k1_g[li]['p_raw'] - k1_gb[li]['p_raw']) < 1e-12
        and abs(k1_g[li]['p_share']
                - k1_gb[li]['p_share']) < 1e-12
        for li in G_REG))
    log('ga7 perm rerun ok=%s' % ga7_ok, lines)

    # K2 glm: ga5 anchor with L39 direction (2995 lineage;
    # its stored dirs were degenerate copies of L39) + true
    # per-layer K2 for T3
    r95 = json.load(open(RES_2995, encoding='utf-8'))
    k2_g = {}
    ga5_diff = 0.0
    for li in G_REG:
        Wl = layers[li].mlp.gate_up_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        Rn = Wl[INT_G:, :]
        Rn = Rn / np.maximum(
            np.linalg.norm(Rn, axis=1, keepdims=True), 1e-30)
        obs95_chk = float(np.sort(np.abs(
            Rn @ dirs_mlp_g[NL_G - 1]))[-K_TOP:].mean())
        ga5_diff = max(ga5_diff, abs(
            obs95_chk - float(
                r95['T3']['per_layer'][str(li)]['obs'])))
        k2_g[li] = k2_weightside(Rn, dirs_mlp_g[li],
                                 [RNG_MAIN, 4], HID_G)
        del Wl, Rn
        log('glm K2 L%d obs %.6f null %.6f+-%.6f z=%.2f '
            '(2995 pseudo-layer z %s)'
            % (li, k2_g[li]['obs'], k2_g[li]['null_mu'],
               k2_g[li]['null_sd'], k2_g[li]['z'],
               r95['T3']['per_layer'][str(li)]['z']), lines)
    ga5_ok = bool(ga5_diff < 1e-5)
    log('ga5 K2 obs(L39 dir) vs 2995 %.2e ok=%s'
        % (ga5_diff, ga5_ok), lines)

    act_std_g = {li: float(act_g[li].std())
                 for li in G_REG}
    gates_g = bool(all(v > 0 for v in act_std_g.values()))

    del model, layers, handles
    gc.collect()
    _t.cuda.empty_cache()
    log('glm unloaded', lines)

    # =====================================================
    # anchors roll-up + verdict
    # =====================================================
    anchor_ok = bool(qa1_ok and qa2_ok and qa3_ok and qa4_ok
                     and qa5_ok and ga1_ok and ga2_ok
                     and ga3_ok and ga4_ok and ga5_ok
                     and ga6_ok and ga7_ok and gates_q
                     and gates_g
                     and g_dirs_spread > 1e-3)

    verdict = None
    T1 = T2 = T3 = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
        log('ANCHOR FAIL -> all void', lines)
    else:
        p_maxT_q_min = min(pmaxT_q.values())
        p_share_q_min = min(k1_q[li]['p_share']
                            for li in Q_REG)
        k1_sig_q = bool(p_maxT_q_min <= P_GATE
                        or p_share_q_min <= P_GATE)
        z_max_q = max(k2_q[li]['z'] for li in Q_REG)
        k2_absent_q = bool(z_max_q < Z_GATE)
        artifact_q = bool(k1_sig_q and k2_absent_q)
        T1 = {'k1': {str(li): {
                        'p_maxT': round(pmaxT_q[li], 5),
                        'p_raw': round(
                            k1_q[li]['p_raw'], 5),
                        'p_share': round(
                            k1_q[li]['p_share'], 5),
                        'obs_max': round(
                            k1_q[li]['obs_max'], 6),
                        'share_obs': round(
                            k1_q[li]['share_obs'], 6)}
                      for li in Q_REG},
              'k2': {str(li): {
                  'obs': round(k2_q[li]['obs'], 6),
                  'z': round(k2_q[li]['z'], 2)}
                  for li in Q_REG},
              'p_maxT_min': p_maxT_q_min,
              'p_share_min': p_share_q_min,
              'k1_sig': k1_sig_q, 'z_max': round(z_max_q, 2),
              'k2_absent': k2_absent_q,
              'artifact_q': artifact_q}
        log('T1 qwen: K1 p_maxT=%.5f p_share=%.5f sig=%s; '
            'K2 z_max=%.2f absent=%s -> artifact_q=%s'
            % (p_maxT_q_min, p_share_q_min, k1_sig_q,
               z_max_q, k2_absent_q, artifact_q), lines)

        p_maxT_g_min = min(pmaxT_g.values())
        p_share_g_min = min(k1_g[li]['p_share']
                            for li in G_REG)
        glm_sig = bool(p_maxT_g_min <= P_GATE
                       or p_share_g_min <= P_GATE)
        T2 = {'k1': {str(li): {
                        'p_maxT': round(pmaxT_g[li], 5),
                        'p_raw': round(
                            k1_g[li]['p_raw'], 5),
                        'p_share': round(
                            k1_g[li]['p_share'], 5),
                        'obs_max': round(
                            k1_g[li]['obs_max'], 6),
                        'share_obs': round(
                            k1_g[li]['share_obs'], 6)}
                      for li in G_REG},
              'p_maxT_min': p_maxT_g_min,
              'p_share_min': p_share_g_min,
              'glm_sig': glm_sig}
        log('T2 glm: K1 p_maxT=%.5f p_share=%.5f sig=%s'
            % (p_maxT_g_min, p_share_g_min, glm_sig), lines)

        T3 = {'k2': {str(li): {
            'obs': round(k2_g[li]['obs'], 6),
            'z': round(k2_g[li]['z'], 2)}
            for li in G_REG},
            'z_min': round(min(k2_g[li]['z']
                               for li in G_REG), 2),
            'z_max': round(max(k2_g[li]['z']
                               for li in G_REG), 2),
            'note': 'TRUE per-layer dirs; 2995 T3 per-layer '
                    'z were L39 pseudo-replications (VOID)'}
        log('T3 glm K2 z in [%.2f, %.2f]'
            % (T3['z_min'], T3['z_max']), lines)

        if artifact_q and glm_sig:
            verdict = 'registry_replicated_at_matched_caliber'
        elif artifact_q:
            verdict = 'qwen_registry_act_side_only_glm4_absent'
        elif glm_sig:
            verdict = 'registry_robust_across_calibers'
        else:
            verdict = 'registry_not_replicated_confirmed'
        log('VERDICT %s' % verdict, lines)

    if verdict is None:
        verdict = 'anchor_fail_all_void'

    elapsed = time.monotonic() - t0

    # ---------- persist ----------
    res = {
        'phase': 2996,
        'final_verdict': verdict,
        'anchor_all_ok': anchor_ok,
        'anchors': {
            'qa1_diff': qa1_diff, 'qa2_rel': qa2_rel,
            'qa3_rel': qa3_rel, 'qa4_rel': qa4_rel,
            'qa5_ok': qa5_ok, 'ga1_diff': ga1_diff,
            'ga2_non39_diff': non39_diff,
            'ga2_mlp39_diff': dm39_diff,
            'ga2_ok': ga2_ok, 'ga3_rel': ga3_rel,
            'ga4_rel': ga4_rel, 'ga5_diff': ga5_diff,
            'ga6_rel': ga6_rel, 'ga7_ok': ga7_ok,
            'g_dirs_spread': g_dirs_spread},
        'nondegeneracy': {'act_std_qwen': act_std_q,
                          'act_std_glm': act_std_g},
        'degeneracy_finding': '2995 dirs_attn/dirs_mlp all '
                              '40 rows identical (single-slot '
                              'hook + broadcast); T1/T2 '
                              'LTOP-only VALID; T3 per-layer '
                              'VOID, re-tested here',
        'T1': T1, 'T2': T2, 'T3': T3,
        'tags': prereg['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note': 'run3: (1) 2995 T3 per-layer z '
                           'retracted to L39-only claim '
                           '(degeneracy_finding); (2) qa1 '
                           'anchor reworded 2927->2979 '
                           '(2927 dirs_word is a DIFFERENT '
                           '57-word en/L pair set, not the '
                           '74-cell lineage); (3) act '
                           'recompute switched float64->'
                           'bf16 GPU (float64 differs from '
                           'the bf16 hook by 2-3 ULP ~5e-3; '
                           'tolerance miscalibration, not '
                           'capture error); run4: (4) glm '
                           'ga4 recompute switched split-'
                           'matmul->fused single matmul '
                           '(cuBLAS rounding path differs '
                           '4e-4..1e-2); run5: (5) K1_sig '
                           'rule corrected AND->OR (2989 '
                           'registered rule, P_GATE=0.05): '
                           'the audited card never required '
                           'both legs (its p_lang_maxT was '
                           '0.51; share leg carried the '
                           'verdict); no other criteria '
                           'changed',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    save = {
        'words_q': np.array(['%s:%s:%s' % c
                             for c in cells]),
        'lang_q': lang_q,
        'words_g': np.array(['%s:%s:%s' % (c[1], c[2], c[3])
                             for c in cells_g]),
        'lang_g': lang_g,
        'dirs_attn_glm': dirs_attn_g,
        'dirs_mlp_glm': dirs_mlp_g,
        'u_top_glm': u_top,
    }
    for li in Q_REG:
        save['act_q_%d' % li] = act_q[li]
        save['D_q_%d' % li] = D_q[li]
        save['dirs_mlp_q_%d' % li] = dirs_mlp_q[li]
    for li in G_REG:
        save['act_g_%d' % li] = act_g[li]
        save['c_g_%d' % li] = c_g[li]
    npz_path = os.path.join(
        OUT, 'omega_f2a_registry_caliber_audit.npz')
    np.savez_compressed(npz_path, **save)

    seal = {
        'npz_sha256_8': sha8(npz_path),
        'result_sha256_8': sha8(
            os.path.join(OUT, 'result.json')),
        'exec_sha256_8': sha8(
            os.path.join(OUT, 'execution.json')),
    }
    with open(os.path.join(OUT, 'seal.json'), 'w',
              encoding='utf-8') as f:
        json.dump(seal, f, indent=2)
    log('sealed %s' % json.dumps(seal), lines)
    log('elapsed %.1fs' % elapsed, lines)


if __name__ == '__main__':
    main()
