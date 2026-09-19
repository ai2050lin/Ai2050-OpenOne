"""Phase 2787: does the e0 anchor-axis recipe bridge the 2782 gap?

Background.  2786 established the repair carrier as a SINGLE shared
anchor direction e0 (top PC of the 21 pull<0 bsub-fixed controlled
v_rows; nearly orthogonal to token embeddings; LOO 21/21; recipe 25/65
> archival 21/65).  Open questions now attacked together:
  Part 1 (natural): does projecting natural v_rows onto the FROZEN
      controlled e0 and bsub-ing that 1-D component repair natural
      errors?  2782's 65-D subspace projection failed (0/35, energy 27%)
      -- but e0 alone was never tried.  Collateral on the 13 native
      correct rows recorded (Pareto check vs 7/13 full-v, 2/13
      heads-only).
  Part 2 (controlled): does an e0-axis SIGN-FLIP (v' = v - 2(v.e0)e0)
      repair the 24 pull>0 rows?  Comparison points: oracle axis 12/24,
      rival axis 4/24 (2785).
e0 is recomputed deterministically from the archival 21 rows (no
natural leakage) and frozen before any forward.

Prereg (frozen before any forward):
  G-C  harness gate: e0 recipe reproduces 21/21 on the controlled
       bsub-fixed rows (and 25/65 total expected, descriptive).
  P1   e0_natural_works iff e0-component bsub flips >= 3/35 natural
       wrong rows AND random-direction ctrl = 0 (10 draws/row).
  P2   descriptive: e0-recipe collateral on the 13 native-correct
       natural rows (compare 7/13 full-v, 2/13 heads-only).
  P3   e0_flip_works iff e0-axis flip flips >= 4/24 AND > random-axis
       flip null (10 draws/row).
  D    descriptive: energy share |v.e0| of natural v_rows on e0;
       row-set overlaps for Part 2 vs 2785 arms.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

BASE = cc.BASE
OUT = BASE / 'phase2787' / 'qwen4_e0_natural_flip'
P2761 = BASE / 'phase2761' / 'qwen4_kc_fault'
P2763 = BASE / 'phase2763' / 'qwen4_debias_repair'
P2774 = BASE / 'phase2774' / 'qwen4_pull_validation'

ALPHA = 0.3
BSUB_LAYER = 35
N_NULL = 10
NULL_SEED = 27870

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
    ("The nearest planet to the Sun is", "nearest planet", "",
     " Mercury"),
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
    'G-C': 'e0 recipe reproduces 21/21 on the controlled bsub-fixed rows',
    'P1': 'e0_natural_works iff e0-component bsub flips >= 3/35 AND '
          'random-direction ctrl = 0 (10 draws/row)',
    'P2': 'descriptive collateral of e0 recipe on native-correct rows',
    'P3': 'e0_flip_works iff e0-axis flip >= 4/24 AND > random-axis '
          'flip null (10 draws/row)',
}


def main():
    import torch
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        str(cc.ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    # ---- freeze e0 from the 21 archival rows (deterministic) ----
    z61 = np.load(P2761 / 'fault_scores.npz', allow_pickle=False)
    c_wrong = np.array(sorted(int(i) for i in z61['wrong_idx']))
    zbd = np.load(P2763 / 'bias_dirs.npz', allow_pickle=False)
    v_c, vn_c = zbd['v_rows'], zbd['v_norms']
    zp = np.load(P2774 / 'pull_stats.npz', allow_pickle=False)
    pull_c, bsub_c = zp['pull'], zp['bsub'].astype(bool)
    assert (zp['wrong_idx'] == c_wrong).all()
    sel21 = [k for k in range(len(c_wrong))
             if pull_c[k] < 0 and bool(bsub_c[k])]
    assert len(sel21) == 21
    ids21 = [int(c_wrong[k]) for k in sel21]
    V21 = np.stack([v_c[i] / vn_c[i] for i in ids21])
    _, _, Vh21 = np.linalg.svd(V21.astype(np.float64), full_matrices=False)
    e0 = Vh21[0] / np.linalg.norm(Vh21[0])

    from phase2662_symmetric_mapping_contract import load_native
    model, tok2 = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device

    state = {'bsub': None}
    cap = {'hfin': None}

    def make_bsub_hook():
        def hook(module, args, output):
            if state['bsub'] is None:
                return None
            out = output[0] if isinstance(output, tuple) else output
            h = out[0, -1]
            out[0, -1] = h - state['bsub'][0] * h.norm() * state['bsub'][1]
            return None
        return hook

    def make_fin_hook():
        def hook(module, args, output):
            cap['hfin'] = output.detach()
            return None
        return hook

    h1 = model.model.layers[BSUB_LAYER].register_forward_hook(
        make_bsub_hook())
    h2 = model.model.layers[BSUB_LAYER].register_forward_hook(
        make_fin_hook())

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

    def run_e0_bsub(ids, v):
        """v: unit natural v_row; bsub its e0 component."""
        vf = float(v @ e0) * e0
        n = np.linalg.norm(vf)
        assert n > 1e-8
        vf = vf / n
        state['bsub'] = (ALPHA, torch.tensor(vf.astype(np.float32),
                                             device=device))
        try:
            return arg_of(fwd(ids))
        finally:
            state['bsub'] = None

    # ================= G-C: controlled harness gate =================
    import phase2747_rdc_material as mat2747
    material, data = mat2747.freeze()
    c_rows = [r for r in data['diagnostic']
              if r['kind'] == 'controlled_relation']
    c_tgt = np.array([r['target'] for r in c_rows], dtype=np.int64)
    c_ids = [r['prompt_ids'] for r in c_rows]
    fam_arr = np.array([r['family'] for r in c_rows], dtype=np.str_)

    kept21 = 0
    for i in ids21:
        vf = float((v_c[i] / vn_c[i]) @ e0) * e0
        vf = vf / np.linalg.norm(vf)
        state['bsub'] = (ALPHA, torch.tensor(vf.astype(np.float32),
                                             device=device))
        try:
            m = arg_of(fwd(c_ids[i]))
        finally:
            state['bsub'] = None
        kept21 += int(m == c_tgt[i])
    assert kept21 == 21, ('G-C failed', kept21)
    print('P2787 GC_OK e0 recipe 21/21 on controlled', flush=True)

    # ================= Part 1: natural e0 recipe =================
    ids_full, ids_wo, tgt_n, multi = [], [], [], []
    for (pre, span, suf, ans) in ITEMS:
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
    tgt_arr = np.array(tgt_n, dtype=np.int64)

    arg_nat = np.array([arg_of(fwd(ids_full[i]))
                        for i in range(len(ITEMS))])
    wrong_mask = (arg_nat != tgt_arr) & ~np.array(multi)
    wrong_idx = [int(i) for i in np.where(wrong_mask)[0]]
    correct_idx = [int(i) for i in np.where(~wrong_mask)[0]]
    print('P2787 NATIVE n_wrong=%d n_correct=%d'
          % (len(wrong_idx), len(correct_idx)), flush=True)
    assert len(wrong_idx) == 35 and len(correct_idx) == 13

    v_nat = {}
    for i in range(len(ITEMS)):
        v = h_final(ids_full[i]) - h_final(ids_wo[i])
        nrm = float(np.linalg.norm(v))
        assert nrm > 0
        v_nat[i] = v / nrm

    # energy share of natural v_rows on e0
    e_nat = [abs(float(v_nat[i] @ e0)) for i in wrong_idx]
    print('P2787 natural |v.e0| mean=%.4f min=%.4f max=%.4f'
          % (float(np.mean(e_nat)), float(np.min(e_nat)),
             float(np.max(e_nat))), flush=True)

    flips = 0
    nat_rows = {}
    for i in wrong_idx:
        m = run_e0_bsub(ids_full[i], v_nat[i])
        ok = bool(m == tgt_arr[i])
        nat_rows[str(i)] = ok
        flips += int(ok)
    print('P2787 P1 e0-natural flips=%d/35' % flips, flush=True)

    rng = np.random.default_rng(NULL_SEED)
    ctrl_flips = 0
    for i in wrong_idx:
        for _ in range(N_NULL):
            r = rng.standard_normal(len(e0))
            r /= np.linalg.norm(r)
            state['bsub'] = (ALPHA, torch.tensor(r.astype(np.float32),
                                                 device=device))
            try:
                m = arg_of(fwd(ids_full[i]))
            finally:
                state['bsub'] = None
            ctrl_flips += int(m == tgt_arr[i])
    print('P2787 P1 ctrl=%d/%d' % (ctrl_flips, len(wrong_idx) * N_NULL),
          flush=True)

    coll = {}
    for i in correct_idx:
        m = run_e0_bsub(ids_full[i], v_nat[i])
        coll[str(i)] = bool(m != arg_nat[i])
    coll_breaks = int(sum(coll.values()))
    print('P2787 P2 collateral=%d/13' % coll_breaks, flush=True)

    p1_pass = bool(flips >= 3 and ctrl_flips == 0)

    # ================= Part 2: e0-axis flip on 24 pull>0 rows ==========
    pos_k = [k for k in range(len(c_wrong)) if pull_c[k] > 0]
    assert len(pos_k) == 24
    f_flips = 0
    flip_rows = {}
    for k in pos_k:
        i = int(c_wrong[k])
        v = v_c[i] / vn_c[i]
        vflip = v - 2.0 * float(v @ e0) * e0
        vflip = vflip / np.linalg.norm(vflip)
        state['bsub'] = (ALPHA, torch.tensor(vflip.astype(np.float32),
                                             device=device))
        try:
            m = arg_of(fwd(c_ids[i]))
        finally:
            state['bsub'] = None
        ok = bool(m == c_tgt[i])
        flip_rows[str(i)] = ok
        f_flips += int(ok)
    print('P2787 P3 e0-flip flips=%d/24' % f_flips, flush=True)

    null_flip = 0
    for k in pos_k:
        i = int(c_wrong[k])
        v = v_c[i] / vn_c[i]
        for _ in range(N_NULL):
            u = rng.standard_normal(len(e0))
            u /= np.linalg.norm(u)
            vf2 = v - 2.0 * float(v @ u) * u
            vf2 /= np.linalg.norm(vf2)
            state['bsub'] = (ALPHA, torch.tensor(vf2.astype(np.float32),
                                                 device=device))
            try:
                m = arg_of(fwd(c_ids[i]))
            finally:
                state['bsub'] = None
            null_flip += int(m == c_tgt[i])
    print('P2787 P3 null=%d/%d' % (null_flip, len(pos_k) * N_NULL),
          flush=True)

    p3_pass = bool(f_flips >= 4 and
                   f_flips > np.percentile(
                       np.full(N_NULL, null_flip / N_NULL), 95))

    # family decomposition of e0-flip rows
    fam_dec = {}
    ORACLE_SET = {71, 79, 95, 111, 119, 273, 277, 281, 285, 298, 313, 314}
    RIVAL_SET = {298, 302, 314, 318}
    e0_set = set(int(i) for i, ok in flip_rows.items() if ok)
    for k in pos_k:
        i = int(c_wrong[k])
        if not flip_rows[str(i)]:
            continue
        f = str(fam_arr[i])
        fam_dec.setdefault(f, [0, 0])
        fam_dec[f][0] += 1
    for f in fam_dec:
        pass
    counts = {}
    for k in pos_k:
        i = int(c_wrong[k])
        f = str(fam_arr[i])
        counts.setdefault(f, [0, 0])
        counts[f][1] += 1
        counts[f][0] += int(flip_rows[str(i)])

    verdict = {
        'e0_natural_works': p1_pass,
        'natural_flips': flips, 'natural_ctrl_flips': ctrl_flips,
        'collateral_breaks': coll_breaks,
        'natural_energy_on_e0_mean': float(np.mean(e_nat)),
        'e0_flip_works': p3_pass,
        'e0_flip_flips': f_flips, 'flip_null': null_flip,
        'e0_flip_fam': counts,
        'overlap_with_oracle12': len(e0_set & ORACLE_SET),
        'overlap_with_rival4': len(e0_set & RIVAL_SET),
    }
    result = {'phase': 2787, 'prereg': PREREG, 'verdict': verdict,
              'natural_flip_rows': nat_rows, 'collateral': coll,
              'e0_flip_rows': flip_rows}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'e0_natural_stats.npz',
           wrong=np.array(wrong_idx, dtype=np.int64),
           flips=np.array([int(nat_rows[str(i)])
                           for i in wrong_idx], dtype=np.int64),
           energy_e0=np.array(e_nat))
    for h in (h1, h2):
        h.remove()
    print('P2787 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
