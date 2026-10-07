# -*- coding: utf-8 -*-
"""Phase 3065: Omega-P62 V-arm sign orchestration tracing.

Three models bf16 native, sequential (qwen3-1.7b ->
qwen3-4b -> deepseek-r1-distill-qwen-7b). Question
(3065 A, menu of 3064): WHERE is the V-arm sign
introduced - at injection-layer V geometry or in
downstream propagation?

E1 ladder: per layer l x 24 pairs V-only FRONT-row
  replacement (repV = base VB; rows 0..3 = srcV from
  the prefixed counterpart); metric cos(lg - LG0,
  TT[k]) med over pairs.
E2 geometry (banks only): per layer med over pairs
  of med over FRONT rows of cos(srcV, VB_base) and
  ||srcV - VB_base|| / ||VB_base||.
E3 trajectory (free capture from E1 forwards):
  d_l' at the last position from the layer-l
  injection; cos(d_l', W_U[t_k]) for all l' >= l;
  sign-flip counts.
E4 readout: last-layer injection d_final -> Y =
  D @ W_U^T (GPU bf16 matmul on the resident lm_head);
  top-256 |Y| mass share; Y at the target token;
  top-16 signed strings; cross-model top-256 string
  Jaccard (descriptive, null 2000 perms seed 3022+).
STAT track: n_sel >= 5 AND |pearson(med_c, med_cV)
  over layers with med_rho >= 0.1| >= 0.6 AND perm
  p <= 0.01 (5000 layer shuffles, seed 3021).
verdict: setup anchors fail -> setup_failed_v_sign;
  n_track 3 -> sign_geometry_all3; 0 ->
  sign_decoupled_all3; else sign_geometry_{n}of3_
  <nonttracking tags joined by _>_orchestrated
  (single branch).
anchors: per model b0 recapture bit 0.0 (LG+VB+PB);
  b1 full-field V sham self-replacement bit identity;
  b3 finite; a4 = DS7B ladder med_c(27) vs 3064 S1
  0.9134400687623977 tol 1e-6 (in setup_ok);
  a5 = qwen3-4b med_c(35) vs 3051/3063 fp32 -0.3409
  tol 0.10 RECORDED ONLY (precision change).
memory discipline: no fp64 W_U copies (GPU bf16
  lm_head reused for Y and t_k rows); banks fp64 CPU;
  models del + empty_cache between loads (user rule:
  one model fully done before the next).
"""
import gc
import hashlib
import io
import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, \
    AutoModelForCausalLM

PHASE = 3065
NAME = 'omega_p62_v_sign_orchestration'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3065', NAME)
LOG = os.path.join(OUT, 'run_log.txt')

MODELS = (
    ('qwen3_1p7b',
     os.path.join(ROOT, 'models', 'hf',
                  'qwen3-1.7b'),
     dict(NL=28, HID=2048, KV_HEAD=8,
          NQ_HEAD=16)),
    ('qwen3_4b',
     os.path.join(ROOT, 'models', 'hf',
                  'qwen3-4b'),
     dict(NL=36, HID=2560, KV_HEAD=8,
          NQ_HEAD=32)),
    ('ds7b',
     os.path.join(ROOT, 'models', 'hf',
                  'deepseek-r1-distill-qwen-7b'),
     dict(NL=28, HID=3584, KV_HEAD=4,
          NQ_HEAD=28)),
)
HDIM = 128
FRONT = 4
SEED_MAIN = 3020
N_PERM_R = 5000
N_PERM_J = 2000
BODIES = (
    'The weather was cold, so',
    'He studied every night because',
    'The experiment failed, therefore',
    'He missed the train, however',
    'The garden grows quickly while',
    'The price was high, yet',
    'She speaks French, although',
    'The road was closed, thus',)
TARGETS = ('so', 'because', 'therefore',
           'however', 'while', 'yet',
           'although', 'thus')
PREFIXES = ('', 'In a formal style,',
            'In Shakespearean style,',
            'Regarding the weather,')

PREREG = {
    'mode': 'three models bf16 native sequential '
            '(qwen3-1.7b -> qwen3-4b -> ds7b), one '
            'model fully done before the next '
            '(GPU OOM discipline); same-precision '
            'bit anchors internal; a4 DS7B ladder '
            'consistency vs 3064 S1 tol 1e-6 '
            'hard; a5 qwen3-4b vs 3051/3063 fp32 '
            '-0.3409 tol 0.10 recorded-only '
            '(precision change); eager attention, '
            'seed 3020',
    'question': '3065 A (menu of 3064): is the '
                'V-arm sign (qwen3-4b -0.34 vs '
                'DS7B +0.913 at the last layer) '
                'introduced at injection-layer V '
                'geometry or in downstream '
                'propagation? Three-model sign '
                'matrix + per-layer geometry-'
                'tracking test.',
    'E1_ladder': 'per layer l (0..NL-1) x 24 '
                 'c-major pairs: V-only FRONT-row '
                 'replacement (repV = base VB; '
                 'rows 0..3 = srcV of the same '
                 'rows in the prefixed '
                 'counterpart); metric cos(lg - '
                 'LG0, TT[k]) med over pairs',
    'E2_geometry': 'banks only: per layer med '
                   'over pairs of med over FRONT '
                   'rows of cos(srcV, VB_base) '
                   'and rho = ||srcV - VB_base|| '
                   '/ ||VB_base||',
    'E3_trajectory': 'free capture from E1 '
                     'forwards: d_l\' at the last '
                     'position from the layer-l '
                     'injection; cos(d_l\', '
                     'W_U[t_k]) for all l\' >= l; '
                     'sign flips over l\'>l counted '
                     '(recorded diagnostic)',
    'E4_readout': 'last-layer injection d_final -> '
                  'Y = D @ W_U^T via GPU bf16 '
                  'matmul on resident lm_head; '
                  'top-256 |Y| mass share; Y at '
                  'target; top-16 signed strings; '
                  'cross-model top-256 string '
                  'Jaccard descriptive (null 2000 '
                  'perms seed 3022+pair)',
    'STAT_track': 'select layers with med_rho >= '
                  '0.1 (n_sel >= 5 required); r = '
                  'pearson(med_c, med_cV); perm p '
                  'from 5000 layer shuffles seed '
                  '3021; track = |r| >= 0.6 AND '
                  'p <= 0.01 AND n_sel >= 5',
    'verdict': 'setup anchors fail -> '
               'setup_failed_v_sign; n_track 3 -> '
               'sign_geometry_all3; 0 -> '
               'sign_decoupled_all3; else '
               'sign_geometry_{n}of3_<nonttracking '
               'tags>_orchestrated; single branch',
    'anchors': 'per model: b0 recapture 4 prompts '
               'bit 0.0 (LG+VB+PB); b1 full-field '
               'V sham self-replacement bit '
               'identity; b3 finite; a4 ds7b '
               'hard; a5 qwen3-4b recorded-only',
    'statistics_discipline': 'same-precision '
                             'permutation nulls on '
                             'frozen seeds; PC1/sign '
                             'none needed; no '
                             'cross-space cosine '
                             'except t_k rows '
                             '(hidden space) and Y '
                             '(logit space); '
                             'cross-model comparisons '
                             'only in logit/string '
                             'space, marked '
                             'descriptive',
    'memory_discipline': 'no fp64 W_U copies '
                         '(GPU bf16 lm_head reused); '
                         'banks fp64 CPU; models del '
                         '+ gc + empty_cache between '
                         'loads; allocations per '
                         'forward are transient',
}

os.makedirs(OUT, exist_ok=True)
for fn in (NAME + '.npz', 'run_log.txt',
           'execution.json', 'result.json',
           'seal.json'):
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        os.remove(p)
t0 = time.time()
created = time.strftime('%Y-%m-%d %H:%M:%S')
execution = {'phase': PHASE, 'name': NAME,
             'created': created, 'prereg': PREREG}
with open(os.path.join(OUT, 'execution.json'),
          'w', encoding='utf-8') as f:
    json.dump(execution, f, ensure_ascii=False,
              indent=1)

lines = []


def log(msg):
    lines.append(str(msg))
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


log('execution.json written (prereg frozen) %s'
    % created)
torch.manual_seed(SEED_MAIN)
np.random.seed(SEED_MAIN)


def cosv(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b) / (na * nb)


def pearson_perm(x, y, n_perm, seed):
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        rp = float(np.corrcoef(
            x, rng.permutation(y))[0, 1])
        if abs(rp) >= abs(r):
            cnt += 1
    return r, (float(cnt) + 1.0) \
        / (n_perm + 1.0)


def run_model(tag, mdir, cfg):
    """Full per-model pipeline; returns dict of
    arrays and stats."""
    NL = cfg['NL']
    HID = cfg['HID']
    KV_HEAD = cfg['KV_HEAD']
    NQ = cfg['NQ_HEAD']
    KVW = KV_HEAD * HDIM
    log('=== model %s: load (%dL hid=%d kv=%d) '
        'gpu=%.2f GB ==='
        % (tag, NL, HID, KV_HEAD,
           torch.cuda.memory_allocated() / 1e9))
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForCausalLM.from_pretrained(
        mdir, torch_dtype=torch.bfloat16,
        attn_implementation='eager'
        ).to('cuda').eval()
    layers = model.model.layers
    assert len(layers) == NL
    assert int(model.config.num_key_value_heads) \
        == KV_HEAD
    assert int(model.config.num_attention_heads) \
        == NQ
    assert int(model.config.hidden_size) == HID
    NVOC = int(model.config.vocab_size)
    Wemb = model.get_output_embeddings().weight
    assert int(Wemb.shape[0]) == NVOC
    assert int(Wemb.shape[1]) == HID
    log('%s loaded bf16 (vocab=%d)' % (tag, NVOC))

    stateV = {li: {'repl': None, 'mask': None}
              for li in range(NL)}
    capV = {li: {'rec': False, 'orig': None}
            for li in range(NL)}
    capP = {li: {'rec': False, 'v': None}
            for li in range(NL)}

    def hook_v(st, cp):
        def h(module, inp, out):
            if cp['rec']:
                cp['orig'] = out[0].detach() \
                    .clone()
            if st['repl'] is not None:
                out[0][st['mask']] = st['repl']
            return out
        return h

    def hook_post(cp):
        def h(module, inp, out):
            if cp['rec']:
                t = out[0] \
                    if isinstance(out, tuple) \
                    else out
                cp['v'] = t[0].detach().clone()
            return out
        return h

    for li in range(NL):
        layers[li].self_attn.v_proj \
            .register_forward_hook(hook_v(
                stateV[li], capV[li]))
        layers[li].register_forward_hook(
            hook_post(capP[li]))

    def reset_all():
        for li in range(NL):
            stateV[li]['repl'] = None
            stateV[li]['mask'] = None
            capV[li]['rec'] = False
            capV[li]['orig'] = None
            capP[li]['rec'] = False
            capP[li]['v'] = None

    def forward_gen(ids, repl=None):
        """repl: None or (NL, n, KVW) fp64; V-only
        replacement field with all-True mask.
        Returns lg (NVOC,), vb (NL, n, KVW),
        posts (NL, n, HID) - all fp64 cpu."""
        reset_all()
        m_all = torch.ones(
            len(ids), dtype=torch.bool,
            device='cuda')
        if repl is not None:
            rt = torch.tensor(
                np.ascontiguousarray(repl),
                dtype=torch.bfloat16,
                device='cuda')
            for li in range(NL):
                stateV[li]['repl'] = rt[li]
                stateV[li]['mask'] = m_all
        for li in range(NL):
            capV[li]['rec'] = True
            capP[li]['rec'] = True
        with torch.no_grad():
            out = model(torch.tensor(
                [ids], device='cuda'),
                use_cache=False)
        lg = out.logits[0, -1].detach() \
            .double().cpu().numpy()
        n = len(ids)
        vb = np.stack([capV[li]['orig'].double()
                       .cpu().numpy()
                       for li in range(NL)])
        po = np.stack([capP[li]['v'].double()
                       .cpu().numpy()
                       for li in range(NL)])
        reset_all()
        return lg, vb, po

    # ---- assembly (tokenizer-driven, 3064-identical) ----
    word_tok = {}
    for w in TARGETS:
        wi = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        assert len(wi) == 1, (tag, w, wi)
        word_tok[w] = int(wi[0])
    assembled = []
    for bi in range(len(BODIES)):
        for ci in range(len(PREFIXES)):
            s = (PREFIXES[ci] + ' ' + BODIES[bi]) \
                if PREFIXES[ci] else BODIES[bi]
            ids = [int(x) for x in tok(
                s, add_special_tokens=False)[
                'input_ids']]
            t = word_tok[TARGETS[bi]]
            assert ids.count(t) == 1, (tag, bi, ci)
            assembled.append({'ids': ids,
                              'cond': ci,
                              'body': bi})
    n_pr = len(assembled)
    assert n_pr == 32
    idx_of = {}
    for i in range(n_pr):
        idx_of[(assembled[i]['cond'],
                assembled[i]['body'])] = i
    for i in range(n_pr):
        ci = assembled[i]['cond']
        if ci == 0:
            assembled[i]['off'] = 0
        else:
            bid = assembled[idx_of[(0,
                assembled[i]['body'])]]['ids']
            pid = assembled[i]['ids']
            off = len(pid) - len(bid)
            assert off > 0, (tag, i)
            assert list(pid[off + 1:]) \
                == list(bid[1:]), (tag, i)
            w0b = tok.decode([bid[0]]).strip()
            w0p = tok.decode([pid[off]]).strip()
            assert w0b == w0p, (tag, i, w0b, w0p)
            assembled[i]['off'] = off
    LENS = np.array([len(assembled[i]['ids'])
                     for i in range(n_pr)])
    NMAX = int(LENS.max())
    log('%s assembled 32 prompts (lens %d-%d)'
        % (tag, int(LENS.min()), NMAX))

    cidx = []
    bidx = []
    for ci in (1, 2, 3):
        for bi in range(len(BODIES)):
            cidx.append(ci)
            bidx.append(bi)
    cidx = np.array(cidx)
    bidx = np.array(bidx)
    NP_ = 24

    # ---- banks ----
    LG = np.zeros((n_pr, NVOC))
    VB = np.zeros((n_pr, NL, NMAX, KVW))
    PB = np.zeros((n_pr, NL, NMAX, HID))
    for i in range(n_pr):
        lg, vb, po = forward_gen(
            assembled[i]['ids'])
        n = int(LENS[i])
        LG[i] = lg
        VB[i, :, :n, :] = vb
        PB[i, :, :n, :] = po
    log('%s banks: LG%s VB%s PB%s'
        % (tag, LG.shape, VB.shape, PB.shape))

    b0_diff = 0.0
    for si in (0, 9, 17, 31):
        lg2, vb2, po2 = forward_gen(
            assembled[si]['ids'])
        n2 = int(LENS[si])
        b0_diff = max(b0_diff, float(np.max(
            np.abs(LG[si] - lg2))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(VB[si, :, :n2, :] - vb2))))
        b0_diff = max(b0_diff, float(np.max(
            np.abs(PB[si, :, :n2, :] - po2))))
    b0_diff = float(b0_diff)
    b0_ok = bool(b0_diff == 0.0)
    log('%s b0 recapture diff=%.3e ok=%s'
        % (tag, b0_diff, b0_ok))

    TT = np.stack([
        LG[idx_of[(int(cidx[k]), int(bidx[k]))]]
        - LG[idx_of[(0, int(bidx[k]))]]
        for k in range(NP_)])

    # b1 sham: full-field V self-replacement
    base0 = idx_of[(0, int(bidx[0]))]
    n0 = int(LENS[base0])
    selfV = VB[base0, :, :n0, :].copy()
    lg_s, _, _ = forward_gen(
        assembled[base0]['ids'], repl=selfV)
    b1_diff = float(np.max(np.abs(lg_s
                                  - LG[base0])))
    b1_ok = bool(b1_diff == 0.0)
    log('%s b1 sham self-replacement diff=%.3e '
        'ok=%s' % (tag, b1_diff, b1_ok))

    b3_ok = bool(np.isfinite(LG).all()
                 and np.isfinite(VB).all()
                 and np.isfinite(PB).all()
                 and np.isfinite(TT).all())
    log('%s b3 finite=%s' % (tag, b3_ok))

    # target unembed rows (8, HID) fp64 exact
    tgt_ids = [word_tok[w] for w in TARGETS]
    tk_rows = Wemb[torch.tensor(
        tgt_ids, device='cuda')].detach().double() \
        .cpu().numpy()
    assert np.isfinite(tk_rows).all()

    ridx = np.arange(FRONT)

    def pair_idx(k):
        b = int(bidx[k])
        c = int(cidx[k])
        base_i = idx_of[(0, b)]
        pref_i = idx_of[(c, b)]
        off = assembled[pref_i]['off']
        return b, base_i, pref_i, off

    # ---- E1 ladder (+ E3 traj free, + E4 d_fin) ----
    COS_LAD = np.zeros((NL, NP_))
    TRAJ_C = np.full((NL, NP_, NL), np.nan)
    TRAJ_N = np.full((NL, NP_, NL), np.nan)
    D_FIN = np.zeros((NP_, HID))
    for l in range(NL):
        for k in range(NP_):
            b, base_i, pref_i, off = pair_idx(k)
            nb = int(LENS[base_i])
            repV = VB[base_i, :, :nb, :].copy()
            repV[l, ridx, :] = VB[pref_i][
                l, off + ridx, :]
            lg, _, po = forward_gen(
                assembled[base_i]['ids'],
                repl=repV)
            dlg = lg - LG[base_i]
            COS_LAD[l, k] = cosv(dlg, TT[k])
            for lp in range(l, NL):
                d = po[lp, nb - 1, :] \
                    - PB[base_i][lp, nb - 1, :]
                TRAJ_C[l, k, lp] = cosv(
                    d, tk_rows[b])
                TRAJ_N[l, k, lp] = float(
                    np.linalg.norm(d))
                if l == NL - 1:
                    D_FIN[k] = d
        if l % 6 == 0 or l == NL - 1:
            log('%s E1 l=%d med_c=%.4f'
                % (tag, l,
                   float(np.median(COS_LAD[l]))))
    med_c = np.median(COS_LAD, axis=1)

    # ---- E2 geometry (banks only) ----
    med_cV = np.zeros(NL)
    med_rho = np.zeros(NL)
    for l in range(NL):
        cs = []
        rh = []
        for k in range(NP_):
            b, base_i, pref_i, off = pair_idx(k)
            src = VB[pref_i][l, off + ridx, :]
            bas = VB[base_i][l, ridx, :]
            cs.append(float(np.median(
                [cosv(src[j], bas[j])
                 for j in range(FRONT)])))
            dv = src - bas
            rh.append(float(np.median(
                [float(np.linalg.norm(dv[j]))
                 / max(float(np.linalg.norm(
                     bas[j])), 1e-12)
                 for j in range(FRONT)])))
        med_cV[l] = float(np.median(cs))
        med_rho[l] = float(np.median(rh))

    # ---- E3 flips (recorded diagnostic) ----
    flips_late = []
    end_flip = 0
    for l in range(NL):
        fl = []
        for k in range(NP_):
            seq = TRAJ_C[l, k, l:]
            seq = seq[np.isfinite(seq)]
            sgn = np.sign(seq[seq != 0.0])
            fl.append(int(np.sum(sgn[1:]
                                  != sgn[:-1])))
            if l < NL - 1 and np.isfinite(
                    TRAJ_C[l, k, l]) and np.isfinite(
                    TRAJ_C[l, k, NL - 1]):
                if np.sign(TRAJ_C[l, k, l]) \
                        != np.sign(
                            TRAJ_C[l, k, NL - 1]):
                    end_flip += 1
        flips_late.append(float(np.median(fl)))
    late_med_flip = float(np.median(
        flips_late[NL - 7:]))
    log('%s E3 flips: med per inj-layer (last 7) '
        '=%s endpoint_flip=%d/%d late_med=%.1f'
        % (tag, np.round(
            flips_late[-7:], 1).tolist(),
           end_flip, NL * NP_, late_med_flip))

    # ---- STAT track ----
    sel = med_rho >= 0.1
    n_sel = int(sel.sum())
    if n_sel >= 5:
        r_g, p_g = pearson_perm(
            med_cV[sel], med_c[sel], N_PERM_R, 3021)
        track = bool(abs(r_g) >= 0.6
                     and p_g <= 0.01)
    else:
        r_g, p_g = float('nan'), float('nan')
        track = False
    log('%s STAT: n_sel=%d r=%.4f p=%.4f '
        'track=%s' % (tag, n_sel, r_g, p_g,
                      track))

    # ---- E4 readout (GPU bf16 matmul) ----
    Dt = torch.tensor(D_FIN,
                      dtype=torch.bfloat16,
                      device='cuda')
    with torch.no_grad():
        Yt = (Dt @ Wemb.t()).float() \
            .cpu().numpy()
    assert Yt.shape == (NP_, NVOC)
    Yabs = np.abs(Yt)
    mass = np.sort(Yabs, axis=1)[:, ::-1]
    mass256 = (mass[:, :256].sum(axis=1)
               / np.maximum(
                   Yabs.sum(axis=1), 1e-12))
    y_tgt = np.array([Yt[k, tgt_ids[int(bidx[k])]]
                      for k in range(NP_)])
    meanY = Yt.mean(axis=0)
    top256_ids = np.argsort(
        np.abs(meanY))[::-1][:256]
    top_str = [tok.decode([int(i)]).strip()
               for i in top256_ids]
    order = np.argsort(meanY)
    bot8 = [tok.decode([int(i)]).strip()
            for i in order[:8]]
    top8 = [tok.decode([int(i)]).strip()
            for i in order[::-1][:8]]
    log('%s E4: mass256 med=%.4f y_tgt med=%.3e '
        'top8=%s bot8=%s'
        % (tag, float(np.median(mass256)),
           float(np.median(y_tgt)),
           top8[:4], bot8[:4]))

    str_of = [tok.decode([i]).strip()
              for i in range(NVOC)]
    out = dict(
        tag=tag, NL=NL, HID=HID, NVOC=NVOC,
        str_of=str_of,
        top8=top8, bot8=bot8,
        LG=LG.astype(np.float32),
        VB=VB, TT=TT.astype(np.float32),
        COS_LAD=COS_LAD, med_c=med_c,
        med_cV=med_cV, med_rho=med_rho,
        TRAJ_C=TRAJ_C, TRAJ_N=TRAJ_N,
        flips=np.array(flips_late),
        late_med_flip=late_med_flip,
        end_flip=end_flip,
        D_FIN=D_FIN, Y=Yt.astype(np.float32),
        mass256=mass256, y_tgt=y_tgt,
        top256_ids=top256_ids,
        top256_str=top_str,
        b0_ok=b0_ok, b1_ok=b1_ok,
        b3_ok=b3_ok,
        r=r_g, p_g=p_g, n_sel=n_sel,
        track=track)
    del model, layers, stateV, capV, capP
    del tok, Wemb, Dt
    gc.collect()
    torch.cuda.empty_cache()
    log('%s unloaded; gpu=%.2f GB'
        % (tag,
           torch.cuda.memory_allocated() / 1e9))
    return out


results = {}
for tag, mdir, cfg in MODELS:
    results[tag] = run_model(tag, mdir, cfg)

# ---- a4 / a5 consistency ----
a4_diff = abs(float(results['ds7b']['med_c'][27])
              - 0.9134400687623977)
a4_ok = bool(a4_diff <= 1e-6)
log('a4 ds7b ladder consistency: med_c(27)=%.10f '
    'diff=%.3e ok=%s'
    % (float(results['ds7b']['med_c'][27]),
       a4_diff, a4_ok))
a5_val = float(results['qwen3_4b']['med_c'][35])
a5_diff = abs(a5_val - (-0.3409))
a5_ok = bool(a5_diff <= 0.10)
log('a5 qwen3-4b consistency (recorded-only): '
    'med_c(35)=%.4f vs -0.3409 diff=%.3f ok=%s'
    % (a5_val, a5_diff, a5_ok))

# ---- cross-model E4 string jaccard ----
jac = {}
for i in range(len(MODELS)):
    for j in range(i + 1, len(MODELS)):
        ta = MODELS[i][0]
        tb = MODELS[j][0]
        sa = set(results[ta]['top256_str'])
        sb = set(results[tb]['top256_str'])
        obs = float(len(sa & sb)) \
            / float(len(sa | sb))
        rng = np.random.default_rng(
            3022 + i * 10 + j)
        nv_a = results[ta]['NVOC']
        nv_b = results[tb]['NVOC']
        cnt = 0
        for _ in range(N_PERM_J):
            ia = rng.choice(nv_a, 256,
                            replace=False)
            ib = rng.choice(nv_b, 256,
                            replace=False)
            ja = set(results[ta]['str_of'][int(x)]
                     for x in ia)
            jb = set(results[tb]['str_of'][int(x)]
                     for x in ib)
            if float(len(ja & jb)) \
                    / float(len(ja | jb)) >= obs:
                cnt += 1
        p_desc = (float(cnt) + 1.0) \
            / (N_PERM_J + 1.0)
        jac['%s|%s' % (ta, tb)] = {
            'obs': obs, 'p_desc': p_desc}
        log('XJ %s|%s: jaccard=%.4f (null p_desc'
            '=%.4f)' % (ta, tb, obs, p_desc))

# ---- verdict ----
setup_ok = bool(all(
    results[t]['b0_ok'] and results[t]['b1_ok']
    and results[t]['b3_ok']
    for t, _, _ in MODELS) and a4_ok)
tracked = {t: bool(results[t]['track'])
           for t, _, _ in MODELS}
n_tr = int(sum(tracked.values()))
non_tr = [t for t in tracked
          if not tracked[t]]
if not setup_ok:
    verdict = 'setup_failed_v_sign'
elif n_tr == 3:
    verdict = 'sign_geometry_all3'
elif n_tr == 0:
    verdict = 'sign_decoupled_all3'
else:
    verdict = ('sign_geometry_%dof3_%s_'
               'orchestrated'
               % (n_tr, '_'.join(non_tr)))
log('VERDICT: %s (tracked=%s setup_ok=%s)'
    % (verdict, tracked, setup_ok))

# ---- npz ----
npz_path = os.path.join(OUT, NAME + '.npz')
save = {
    'A4_DIFF': np.float64(a4_diff),
    'A5_VAL': np.float64(a5_val),
    'A5_DIFF': np.float64(a5_diff),
    'SETUP_OK': np.bool_(setup_ok),
    'VERDICT': np.array(verdict),
    'ELAPSED': np.float64(time.time() - t0)}
for tag, _, _ in MODELS:
    rr = results[tag]
    pre = tag + '_'
    save[pre + 'LG'] = rr['LG']
    save[pre + 'VB'] = rr['VB']
    save[pre + 'TT'] = rr['TT']
    save[pre + 'COS_LAD'] = rr['COS_LAD']
    save[pre + 'MED_C'] = rr['med_c']
    save[pre + 'MED_CV'] = rr['med_cV']
    save[pre + 'MED_RHO'] = rr['med_rho']
    save[pre + 'TRAJ_C'] = rr['TRAJ_C']
    save[pre + 'TRAJ_N'] = rr['TRAJ_N']
    save[pre + 'FLIPS'] = rr['flips']
    save[pre + 'D_FIN'] = rr['D_FIN']
    save[pre + 'Y'] = rr['Y']
    save[pre + 'MASS256'] = rr['mass256']
    save[pre + 'Y_TGT'] = rr['y_tgt']
    save[pre + 'TOP256_IDS'] = rr[
        'top256_ids'].astype(np.int64)
    save[pre + 'R'] = np.float64(rr['r'])
    save[pre + 'P_G'] = np.float64(rr['p_g'])
    save[pre + 'N_SEL'] = np.int64(rr['n_sel'])
    save[pre + 'TRACK'] = np.bool_(rr['track'])
    save[pre + 'LATE_MED_FLIP'] = np.float64(
        rr['late_med_flip'])
np.savez(npz_path, **save)

# ---- result.json ----
stats = {}
for tag, _, _ in MODELS:
    rr = results[tag]
    stats[tag] = {
        'nl': rr['NL'], 'hid': rr['HID'],
        'nvoc': rr['NVOC'],
        'med_c_last': float(rr['med_c'][-1]),
        'med_c_late_mean': float(
            rr['med_c'][-6:].mean()),
        'med_cV_last': float(rr['med_cV'][-1]),
        'med_rho_last': float(
            rr['med_rho'][-1]),
        'r': float(rr['r']) if np.isfinite(
            rr['r']) else None,
        'p_r': float(rr['p_g']) if np.isfinite(
            rr['p_g']) else None,
        'n_sel': rr['n_sel'],
        'track': rr['track'],
        'late_med_flip': rr['late_med_flip'],
        'end_flip': rr['end_flip'],
        'mass256_med': float(np.median(
            rr['mass256'])),
        'y_tgt_med': float(np.median(
            rr['y_tgt'])),
        'top8_pos': rr['top8'],
        'top8_neg': rr['bot8'],
        'b0_ok': rr['b0_ok'],
        'b1_ok': rr['b1_ok'],
        'b3_ok': rr['b3_ok'],
    }
# cross-model jaccard null uses per-model str_of
result = {
    'phase': PHASE, 'name': NAME,
    'created': created,
    'elapsed': time.time() - t0,
    'run': 'run1 authoritative (three models '
           'bf16 sequential: qwen3-1.7b -> '
           'qwen3-4b -> ds7b)',
    'prereg': PREREG,
    'stats': stats,
    'cross_jaccard': jac,
    'anchors': {
        'a4_diff': a4_diff, 'a4_ok': a4_ok,
        'a5_val': a5_val, 'a5_diff': a5_diff,
        'a5_ok_recorded_only': a5_ok,
        'setup_ok': setup_ok},
    'verdict': verdict,
}
with open(os.path.join(OUT, 'result.json'),
          'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False,
              indent=1)

seal = {
    'phase': PHASE, 'name': NAME,
    'created': created,
    'npz_sha256_8': sha8(npz_path),
    'result_sha256_8': sha8(
        os.path.join(OUT, 'result.json')),
    'exec_sha256_8': sha8(os.path.join(
        OUT, 'execution.json')),
    'script_sha256_8': sha8(os.path.abspath(
        __file__)),
    'verdict': verdict,
    'setup_ok': setup_ok,
}
with open(os.path.join(OUT, 'seal.json'), 'w',
          encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False,
              indent=1)
log('sealed npz8=%s result8=%s exec8=%s '
    'script8=%s elapsed=%.1fs'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8'],
       seal['exec_sha256_8'],
       seal['script_sha256_8'],
       time.time() - t0))
log('sealed')
print('RUN_COMPLETE %s' % verdict)
