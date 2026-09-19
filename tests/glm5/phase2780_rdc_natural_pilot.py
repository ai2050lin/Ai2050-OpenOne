"""Phase 2780 (Gamma): natural-language migration pilot + oracle-boost
ceiling control.

Part 1 (natural pilot).  48 hand-authored natural factual questions
(frozen list below; prefix + fact span + suffix + single-token answer).
v_row computed with the 2763 recipe adapted to free text: h_final(with
span) - h_final(without span) at the final position, token concat rule
tok(prefix)+tok(span)+tok(suffix).  Pipeline arms identical to 2778
(combo for pull<0, sign-flip bsub for pull>=0 -- NOTE: the flip arm is
ORACLE-GUIDED, u_ro = unit(W_U[target]-W_U[rival]); flagged in 2778).

Preregistered (frozen before any forward):
  G-N0: decode roundtrip on first 8 items.
  G-N1: all 48 v_norms > 0 (non-degenerate).
  G-N2: multi-token answers recorded and excluded from repair stats.
  P-G1: n_wrong >= 8 else 'insufficient_errors_pilot' (report only).
  P-G2: pipeline flips on natural wrong rows >= 3 AND random-direction
        ctrl flips = 0 on the same rows.
  P-G3: dictated-arm collateral on sampled natural correct rows <= 2/20.
  verdict: natural_migration_pilot iff P-G1 and P-G2 and P-G3.

Part 2 (exploratory, post-hoc origin acknowledged: motivated by the 2779
M-R4 finding that sign-flip bsub repairs 5/8 neg pull>0 rows).  Control
C-X1: plain bsub with the ORACLE direction u_ro =
unit(W_U[target]-W_U[rival]) at alpha=0.3 at L35:
  (a) on the 8 controlled neg pull>0 rows: if flips ~= 5/8, the flip-arm
      repair reduces to generic oracle target amplification;
  (b) on all 65 controlled wrong rows: the oracle-boost repair ceiling.
No preregistered criterion; descriptive.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

BASE = cc.BASE
OUT = BASE / 'phase2780' / 'qwen4_natural_pilot'
P2761 = BASE / 'phase2761' / 'qwen4_kc_fault'
P2763 = BASE / 'phase2763' / 'qwen4_debias_repair'
P2774 = BASE / 'phase2774' / 'qwen4_pull_validation'

M_LAYERS = [31, 32, 33, 34, 35]
TOP_HEADS = 4
ALPHA = 0.3
N_COLL_ROWS = 20
COLL_SEED = 27800
CTRL_SEED = 27801

# (prefix, span, suffix, answer) -- frozen 2026-09-16 before any forward
ITEMS = [
    ("The author of the play Hamlet is", "Hamlet", "", " Shakespeare"),
    ("The tallest mountain in the world is", "tallest mountain", "",
     " Everest"),
    ("The capital city of Japan is", "Japan", "", " Tokyo"),
    ("The largest planet in the solar system is", "largest planet", "",
     " Jupiter"),
    ("The chemical symbol for gold is", "gold", "", " Au"),
    ("The painter of the Mona Lisa is", "Mona Lisa", "", " Leonardo"),
    ("The largest desert in the world is the", "largest desert", "",
     " Sahara"),
    ("The currency of the United Kingdom is the", "United Kingdom", "",
     " pound"),
    ("The author of Romeo and Juliet is", "Romeo and Juliet", "",
     " Shakespeare"),
    ("The longest river in the world is the", "longest river", "",
     " Nile"),
    ("The smallest planet in the solar system is", "smallest planet", "",
     " Mercury"),
    ("The boiling point of water is", "boiling point", " degrees Celsius",
     " 100"),
    ("The composer of the Ninth Symphony was", "Ninth Symphony", "",
     " Beethoven"),
    ("The largest country by land area is", "largest country", "",
     " Russia"),
    ("The main language spoken in Brazil is", "Brazil", "",
     " Portuguese"),
    ("The powerhouse organelle of the cell is the", "powerhouse organelle",
     "", " mitochondria"),
    ("The freezing point of water in Fahrenheit is", "freezing point",
     " degrees", " 32"),
    ("The inventor of the telephone was", "telephone", "", " Bell"),
    ("The largest mammal on Earth is the", "largest mammal", "", " blue"),
    ("The capital of Australia is", "Australia", "", " Canberra"),
    ("The study of earthquakes is called", "earthquakes", "",
     " seismology"),
    ("The hardest natural substance is", "hardest natural substance", "",
     " diamond"),
    ("The first President of the United States was", "United States", "",
     " Washington"),
    ("The largest bone in the human body is the", "largest bone", "",
     " femur"),
    ("The process by which plants make food is called",
     "plants make food", "", " photosynthesis"),
    ("The closest star to Earth is the", "closest star", "", " Sun"),
    ("The author of the theory of relativity was", "theory of relativity",
     "", " Einstein"),
    ("The longest wall in the world is in", "longest wall", "", " China"),
    ("The metal that is liquid at room temperature is",
     "liquid at room temperature", "", " mercury"),
    ("The national animal of China is the", "China", "", " panda"),
    ("The largest island in the world is", "largest island", "",
     " Greenland"),
    ("The primary gas in Earth's atmosphere is", "Earth's atmosphere", "",
     " nitrogen"),
    ("The city known as the Big Apple is", "Big Apple", "", " New"),
    ("The founder of Microsoft is", "Microsoft", "", " Bill"),
    ("The fastest land animal is the", "fastest land animal", "",
     " cheetah"),
    ("The number of continents on Earth is", "continents", "", " seven"),
    ("The largest source of vitamin C is", "vitamin C", "", " oranges"),
    ("The capital of Canada is", "Canada", "", " Ottawa"),
    ("The nearest planet to the Sun is", "nearest planet", "", " Mercury"),
    ("The author of Pride and Prejudice was", "Pride and Prejudice", "",
     " Jane"),
    ("The largest moon of Saturn is", "largest moon", "", " Titan"),
    ("The wizard school in Harry Potter is called", "Harry Potter", "",
     " Hogwarts"),
    ("The tallest animal in the world is the", "tallest animal", "",
     " giraffe"),
    ("The Great Barrier Reef is located off the coast of",
     "Great Barrier Reef", "", " Australia"),
    ("The first man to walk on the moon was", "walk on the moon", "",
     " Neil"),
    ("The currency of the United States is the", "United States", "",
     " dollar"),
    ("The bird that cannot fly but swims is the", "cannot fly but swims",
     "", " penguin"),
    ("The largest artery in the human body is the", "largest artery", "",
     " aorta"),
]

PREREG = {
    'G-N0': 'decode roundtrip on first 8 items',
    'G-N1': 'all v_norms > 0',
    'G-N2': 'multi-token answers recorded, excluded from repair stats',
    'P-G1': 'n_wrong >= 8 else insufficient_errors_pilot',
    'P-G2': 'pipeline flips on natural wrong rows >= 3 AND ctrl random '
            'direction flips = 0 on the same rows',
    'P-G3': 'dictated-arm collateral on sampled natural correct rows '
            '<= 2/20',
    'C-X1': 'EXPLORATORY (post-hoc): plain oracle u_ro bsub ceiling on 65 '
            'controlled wrong rows + on 8 neg pull>0 rows',
    'verdict': 'natural_migration_pilot iff P-G1 and P-G2 and P-G3',
}


def main():
    import torch
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        str(cc.ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'n_items': len(ITEMS)}
    fc.save(OUT / 'execution.json', execution)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok2 = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    n_heads = model.config.num_attention_heads
    qdim = model.model.layers[0].self_attn.q_proj.out_features
    hd = qdim // n_heads
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    W_O = {l: model.model.layers[l].self_attn.o_proj.weight.detach().
           float().cpu().numpy() for l in M_LAYERS}

    state = {'bsub': None, 'abl': frozenset()}
    cap = {'on': False, 'oproj': {}, 'hfin': None}
    handles = []

    def make_bsub_hook():
        def hook(module, args, output):
            if state['bsub'] is None:
                return None
            out = output[0] if isinstance(output, tuple) else output
            h = out[0, -1]
            out[0, -1] = h - state['bsub'][0] * h.norm() * state['bsub'][1]
            return None
        return hook

    def make_abl_pre(l):
        def pre(module, args):
            hs_ = [h for (ll, h) in state['abl'] if ll == l]
            if not hs_:
                return None
            x = args[0].clone()
            for h in hs_:
                x[0, -1, h * hd:(h + 1) * hd] = 0
            return (x,) + tuple(args[1:])
        return pre

    def make_cap_pre(l):
        def pre(module, args):
            if cap['on']:
                cap['oproj'][l] = args[0].detach()
            return None
        return pre

    def make_fin_hook():
        def hook(module, args, output):
            cap['hfin'] = output.detach()
            return None
        return hook

    handles.append(model.model.layers[35].register_forward_hook(
        make_bsub_hook()))
    handles.append(model.model.layers[35].register_forward_hook(
        make_fin_hook()))
    for l in M_LAYERS:
        handles.append(model.model.layers[l].self_attn.o_proj.
                       register_forward_pre_hook(make_abl_pre(l)))
        handles.append(model.model.layers[l].self_attn.o_proj.
                       register_forward_pre_hook(make_cap_pre(l)))

    def fwd(ids):
        with torch.inference_mode():
            return model(torch.tensor([ids], device=device))

    def arg_of(o):
        return int(o.logits[0, -1].float().argmax())

    def h_final(ids):
        cap['hfin'] = None
        try:
            fwd(ids)
        finally:
            h = cap['hfin'][0, -1].float().cpu().numpy().copy()
            cap['hfin'] = None
        return h

    # ================= Part 1: natural items =================
    ids_full, ids_wo, tgt_n, multi = [], [], [], []
    for (pre, span, suf, ans) in ITEMS:
        a = pre + span + suf
        ids_f = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(span, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        ids_w = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        ids_full.append(ids_f)
        ids_wo.append(ids_w)
        tt = tok(ans, add_special_tokens=False)['input_ids']
        multi.append(len(tt) > 1)
        tgt_n.append(tt[0])
    g0 = all(tok.decode(ids_full[i]) ==
             ITEMS[i][0] + ITEMS[i][1] + ITEMS[i][2] for i in range(8))
    assert g0, 'G-N0 decode roundtrip'

    a0 = [arg_of(fwd(ids_full[i])) for i in range(4)]
    a1 = [arg_of(fwd(ids_full[i])) for i in range(4)]
    assert a0 == a1, 'G1 determinism'

    arg_nat = np.array([arg_of(fwd(ids_full[i]))
                        for i in range(len(ITEMS))])
    tgt_arr = np.array(tgt_n, dtype=np.int64)
    wrong_mask = (arg_nat != tgt_arr) & ~np.array(multi)
    n_multi = int(sum(multi))
    print('P2780 NATIVE n_wrong=%d n_multi_answer=%d'
          % (int(wrong_mask.sum()), n_multi), flush=True)

    # v_row via span deletion
    v_rows, v_norms = [], []
    for i in range(len(ITEMS)):
        v = h_final(ids_full[i]) - h_final(ids_wo[i])
        nrm = float(np.linalg.norm(v))
        assert nrm > 0, ('G-N1 degenerate', i)
        v_rows.append(v / nrm)
        v_norms.append(nrm)

    def rival_of(i):
        zi = fwd(ids_full[i]).logits[0, -1].float()
        tmp = zi.clone()
        tmp[tgt_arr[i]] = -1e30
        return int(tmp.argmax()), zi

    def top4_heads(ids, u):
        cap['on'] = True
        cap['oproj'].clear()
        try:
            fwd(ids)
        finally:
            cap['on'] = False
        gains = {}
        for l in M_LAYERS:
            X = cap['oproj'][l][0, -1].float().cpu().numpy()
            Wl = W_O[l]
            for h in range(n_heads):
                sl = slice(h * hd, (h + 1) * hd)
                gains[(l, h)] = float(u @ (Wl[:, sl] @ X[sl]))
        return [(l, h) for (l, h), g in
                sorted(gains.items(), key=lambda x: -x[1])
                if g > 0][:TOP_HEADS]

    wrong_idx = [int(i) for i in np.where(wrong_mask)[0]]
    per_row = []
    n_flips = 0
    rng_ctrl = np.random.default_rng(CTRL_SEED)
    for i in wrong_idx:
        riv, zi = rival_of(i)
        u = (W_U[tgt_arr[i]] - W_U[riv])
        u = u / np.linalg.norm(u)
        pl = float(v_rows[i] @ (W_U[tgt_arr[i]] - W_U[riv]))
        rec = {'row': i, 'pull': round(pl, 4), 'rival': riv}
        if pl < 0:
            rec['arm'] = 'combo'
            heads = top4_heads(ids_full[i], (W_U[riv] - W_U[tgt_arr[i]]) /
                               np.linalg.norm(W_U[riv] - W_U[tgt_arr[i]]))
            state['abl'] = frozenset(heads)
            state['bsub'] = (ALPHA, torch.tensor(v_rows[i], device=device))
        else:
            rec['arm'] = 'flip'
            v_flip = v_rows[i] - 2.0 * float(v_rows[i] @ u) * u
            v_flip = v_flip / np.linalg.norm(v_flip)
            state['bsub'] = (ALPHA, torch.tensor(v_flip, device=device))
        try:
            m = arg_of(fwd(ids_full[i]))
        finally:
            state['abl'] = frozenset()
            state['bsub'] = None
        rec['flipped'] = bool(m == tgt_arr[i])
        n_flips += int(rec['flipped'])
        # random-direction ctrl on the same row
        r = rng_ctrl.normal(size=len(v_rows[i]))
        r = r / np.linalg.norm(r)
        state['bsub'] = (ALPHA, torch.tensor(r.astype(np.float32),
                                             device=device))
        try:
            mc = arg_of(fwd(ids_full[i]))
        finally:
            state['bsub'] = None
        rec['ctrl_flipped'] = bool(mc == tgt_arr[i])
        per_row.append(rec)

    ctrl_flips = int(sum(r['ctrl_flipped'] for r in per_row))

    # collateral on native-correct natural rows
    correct_idx = [int(i) for i in np.where(~wrong_mask)[0]]
    rng_coll = np.random.default_rng(COLL_SEED)
    coll_rows = list(rng_coll.choice(correct_idx,
                                     size=min(N_COLL_ROWS,
                                              len(correct_idx)),
                                     replace=False))
    coll_breaks = 0
    coll_detail = []
    for i0 in coll_rows:
        i = int(i0)
        riv, zi = rival_of(i)
        u = (W_U[tgt_arr[i]] - W_U[riv])
        u = u / np.linalg.norm(u)
        pl = float(v_rows[i] @ (W_U[tgt_arr[i]] - W_U[riv]))
        if pl < 0:
            heads = top4_heads(ids_full[i], (W_U[riv] - W_U[tgt_arr[i]]) /
                               np.linalg.norm(W_U[riv] - W_U[tgt_arr[i]]))
            state['abl'] = frozenset(heads)
            state['bsub'] = (ALPHA, torch.tensor(v_rows[i], device=device))
        else:
            v_flip = v_rows[i] - 2.0 * float(v_rows[i] @ u) * u
            v_flip = v_flip / np.linalg.norm(v_flip)
            state['bsub'] = (ALPHA, torch.tensor(v_flip, device=device))
        try:
            m = arg_of(fwd(ids_full[i]))
        finally:
            state['abl'] = frozenset()
            state['bsub'] = None
        coll_breaks += int(m != arg_nat[i])
        coll_detail.append({'row': i, 'pull': round(pl, 4),
                            'broke': bool(m != arg_nat[i])})

    n_wrong = int(wrong_mask.sum())
    pg1 = bool(n_wrong >= 8)
    pg2 = bool(n_flips >= 3 and ctrl_flips == 0) if pg1 else None
    pg3 = bool(coll_breaks <= 2)
    p1 = {'prereg': 'natural_migration_pilot',
          'P-G1': {'pass': pg1, 'n_wrong': n_wrong},
          'P-G2': {'pass': pg2, 'pipeline_flips': int(n_flips),
                   'ctrl_flips': ctrl_flips},
          'P-G3': {'pass': pg3, 'coll_breaks': int(coll_breaks),
                   'n': len(coll_rows)},
          'n_multi_answer': n_multi}
    p1['verdict'] = ('natural_migration_pilot'
                     if (pg1 and pg2 and pg3) else
                     ('insufficient_errors_pilot' if not pg1 else
                      'natural_migration_pilot_failed'))
    print('P2780 PART1 %s' % json.dumps(p1), flush=True)

    # ================= Part 2: C-X1 oracle ceiling (controlled) =========
    z61 = np.load(P2761 / 'fault_scores.npz', allow_pickle=False)
    c_wrong = np.array(sorted(int(i) for i in z61['wrong_idx']))
    zp = np.load(P2774 / 'pull_stats.npz', allow_pickle=False)
    pull_c = zp['pull']
    zbd = np.load(P2763 / 'bias_dirs.npz', allow_pickle=False)
    v_c = zbd['v_rows']
    vn_c = zbd['v_norms']
    import phase2747_rdc_material as mat2747
    material, data = mat2747.freeze()
    c_rows = [r for r in data['diagnostic']
              if r['kind'] == 'controlled_relation']
    c_tgt = np.array([r['target'] for r in c_rows], dtype=np.int64)
    c_ids = [r['prompt_ids'] for r in c_rows]

    def u_ro_c(i, riv):
        u = W_U[c_tgt[i]] - W_U[riv]
        return u / np.linalg.norm(u)

    cx1_neg = {'flips': 0, 'n': 0, 'rows': []}
    for k, i0 in enumerate(c_wrong):
        i = int(i0)
        if pull_c[k] <= 0:
            continue
        zi = fwd(c_ids[i]).logits[0, -1].float()
        tmp = zi.clone()
        tmp[c_tgt[i]] = -1e30
        riv = int(tmp.argmax())
        u = u_ro_c(i, riv)
        state['bsub'] = (ALPHA, torch.tensor(u.astype(np.float32),
                                             device=device))
        try:
            m = arg_of(fwd(c_ids[i]))
        finally:
            state['bsub'] = None
        fam_i = str(mat_fam(c_rows, i))
        if fam_i == 'negation_scope':
            cx1_neg['n'] += 1
            cx1_neg['flips'] += int(m == c_tgt[i])
        cx1_neg['rows'].append({'row': i, 'fam': fam_i,
                                'flipped': bool(m == c_tgt[i])})
    cx1_all = 0
    for k, i0 in enumerate(c_wrong):
        i = int(i0)
        zi = fwd(c_ids[i]).logits[0, -1].float()
        tmp = zi.clone()
        tmp[c_tgt[i]] = -1e30
        riv = int(tmp.argmax())
        u = u_ro_c(i, riv)
        state['bsub'] = (ALPHA, torch.tensor(u.astype(np.float32),
                                             device=device))
        try:
            m = arg_of(fwd(c_ids[i]))
        finally:
            state['bsub'] = None
        cx1_all += int(m == c_tgt[i])
    print('P2780 CX1 neg=%d/%d all=%d/65'
          % (cx1_neg['flips'], cx1_neg['n'], cx1_all), flush=True)

    result = {'phase': 2780, 'prereg': PREREG, 'part1': p1,
              'per_row': per_row, 'collateral': coll_detail,
              'cx1_neg': cx1_neg, 'cx1_all_flips': int(cx1_all),
              'items': [{'prefix': p, 'span': s, 'suffix': x, 'ans': a,
                         'multi': bool(m)}
                        for (p, s, x, a), m in zip(ITEMS, multi)]}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'natural_stats.npz',
           arg_native=arg_nat, tgt=tgt_arr,
           wrong=np.where(wrong_mask)[0],
           v_norms=np.array(v_norms))
    for h in handles:
        h.remove()
    print('P2780 DONE', flush=True)


def mat_fam(rows, i):
    return rows[i]['family']


if __name__ == '__main__':
    main()
