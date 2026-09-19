"""Phase 2772 (auto-continuation): combined non-oracle repair pipeline and
family-generality of the template-token readout carrier.

Background.  2769 localised the kc non-bias readout competition carrier:
per-row top-4 carrier heads at modules 31-35 whose attention writeback
projects positively on u = unit(W_U[rival] - W_U[target]); targeted head
ablation flipped 7/14 kc rows (null q95 = 2) with 0.5% collateral, repaired
5/5 deep-knows rows, and the rival mass concentrates on TEMPLATE answer-code
tokens ("Yes"/"No" in the instruction line), NOT the question fact span
(E1 failed).  2763: bsub (question-fact-span suppression at module 35)
repairs 21/65 but is family-biased (ws 13, kc 2).

Design (frozen before any intervention forward):
  Rows: all 65 native-wrong controlled_relation rows (2761/2763 ordering).
  C001 per-row carrier identification (same procedure as 2769 C002, extended
    to all 65 wrong rows): rival token, u direction, per-head rival gain at
    modules 31-35, top-4 carrier heads, source composition (fact/content/
    other) per family.
  C002 interventions per wrong row:
    (a) head-abl: zero o_proj input at the final position for the 4 carrier
        heads jointly;
    (b) bsub35: alpha=0.3 v_row subtraction at module 35 (2763 direction);
    (c) combo: (a) and (b) simultaneously.
  C003 matched random-head control on the 51 non-kc rows, 30 draws.
  C004 collateral: ws-row carrier heads applied to the 64 attribute_binding
    correct rows.
Preregistered criteria (frozen):
  F1 (combined repair): combo total flips >= 24/65 AND combo total > bsub
      alone (2763: 21).
  F2 (specificity): random-head control total flips <= 1/3 of head-abl
      total flips.
  F3 (descriptive): per-family flip table + carrier source composition.
  verdict: combined_repair_confirmed iff F1 and F2.
Integrity gates: G3 wrong counts == 2761 record; G1 determinism (first 8).
Status: descriptive + preregistered; NOT mechanism closure.
"""
import time

import numpy as np

import rdc_construction_common as cc
import rdc_feature_common as fc
import phase2747_rdc_material as mat2747
import phase2759_rdc_question_span_deletion as p2759

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2772' / 'qwen4_combined_repair'
P2761 = BASE / 'phase2761' / 'qwen4_kc_fault'
P2763 = BASE / 'phase2763' / 'qwen4_debias_repair'
M_LAYERS = [31, 32, 33, 34, 35]
TOP_HEADS = 4
N_CTRL_DRAWS = 30
SEED_CTRL = 2772002
ALPHA_BSUB = 0.3
G3_EXPECTED = {'attribute_binding': 0, 'knowledge_chain': 14,
               'long_distance_role': 10, 'negation_scope': 16,
               'word_sense': 25}

PREREG = {
    'phase': 2772,
    'question': 'Does combining template-carrier head ablation (2769) with '
                'readout-end bias suppression (2763) yield a stronger '
                'oracle-free repair, and is the template carrier '
                'family-general?',
    'criteria': {'F1': 'combo flips >= 24/65 AND combo > bsub alone (21)',
                 'F2': 'random control flips <= 1/3 of head-abl flips',
                 'F3': 'descriptive per-family table and source composition'},
    'verdict': 'combined_repair_confirmed iff F1 and F2',
    'frozen_before_any_intervention_forward': True,
}


def main():
    t0 = time.time()
    cc.guard(0)
    assert not (OUT / 'result.json').exists(), 'immutable; delete before rerun'
    OUT.mkdir(parents=True, exist_ok=True)
    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    import torch
    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic'] if r['kind'] == 'controlled_relation']
    assert len(rows) == 320, len(rows)
    tgt_ids = np.array([r['target'] for r in rows], dtype=np.int64)
    fam_arr = np.array([r['family'] for r in rows], dtype=np.str_)
    id_list = [r['prompt_ids'] for r in rows]

    z61 = np.load(P2761 / 'fault_scores.npz', allow_pickle=False)
    wrong_idx = z61['wrong_idx']
    tgt61 = z61['target']
    assert (tgt61[wrong_idx] == tgt_ids[wrong_idx]).all()
    assert len(wrong_idx) == 65

    from phase2662_symmetric_mapping_contract import load_native
    model, tok = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    n_heads = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    qdim = model.model.layers[0].self_attn.q_proj.out_features
    hd = qdim // n_heads
    group = n_heads // n_kv

    cap = {}
    capture_handles = []

    def oproj_pre(l):
        def hook(module, args):
            cap.setdefault('oproj', {})[l] = args[0].detach()
        return hook

    def vproj_post(l):
        def hook(module, args, output):
            cap.setdefault('v', {})[l] = output.detach()
        return hook

    abl_state = {'pairs': frozenset()}
    bsub_state = {'on': False, 'vec': None}

    def abl_pre(l):
        def hook(module, args):
            hs_ = [h for (ll, h) in abl_state['pairs'] if ll == l]
            if not hs_:
                return None
            x = args[0].clone()
            for h in hs_:
                x[0, -1, h * hd:(h + 1) * hd] = 0
            return (x,) + tuple(args[1:])
        return hook

    def bsub_hook(module, args, output):
        if not bsub_state['on']:
            return None
        out = output[0] if isinstance(output, tuple) else output
        h = out[0, -1]
        out[0, -1] = h - ALPHA_BSUB * h.norm() * bsub_state['vec']
        return None

    for l in M_LAYERS:
        capture_handles.append(
            model.model.layers[l].self_attn.o_proj.
            register_forward_pre_hook(oproj_pre(l)))
        capture_handles.append(
            model.model.layers[l].self_attn.v_proj.
            register_forward_hook(vproj_post(l)))
        capture_handles.append(
            model.model.layers[l].self_attn.o_proj.
            register_forward_pre_hook(abl_pre(l)))
    bsub_handles = [model.model.layers[35].register_forward_hook(bsub_hook)]

    def base_forward(ids, want_att=False):
        with torch.inference_mode():
            t = torch.tensor([ids], device=device)
            o = model(t, output_attentions=want_att,
                      output_hidden_states=True)
            return (o.logits[0, -1].float(),
                    o.hidden_states[-1][0, -1].float(),
                    o.attentions if want_att else None)

    arg_native = np.empty(len(rows), dtype=np.int64)
    h_last0 = None
    for i, ids in enumerate(id_list):
        logits, h_last, _ = base_forward(ids)
        arg_native[i] = int(logits.argmax())
        if i == 0:
            h_last0 = h_last.cpu().numpy().copy()
    g1 = float(np.abs(base_forward(id_list[0])[1].cpu().numpy() -
                      h_last0).max())
    assert g1 == 0.0, ('G1', g1)
    wrong_native = arg_native != tgt_ids
    fam_wrong = {f: int(wrong_native[fam_arr == f].sum()) for f in G3_EXPECTED}
    assert fam_wrong == G3_EXPECTED, ('G3', fam_wrong)

    # ---------- C001 carrier identification for all 65 wrong rows ---------
    carrier = {}
    src_stats = {}
    rival_info = {}
    for k, i in enumerate(wrong_idx):
        i = int(i)
        cap.clear()
        logits, _, att = base_forward(id_list[i], want_att=True)
        zi = logits.cpu().numpy()
        tmp = zi.copy()
        tmp[tgt_ids[i]] = -1e30
        rival = int(tmp.argmax())
        W_U = model.lm_head.weight.detach().float()
        u = (W_U[rival] - W_U[tgt_ids[i]]).detach()
        u = (u / u.norm()).cpu().numpy()
        gains = {}
        for l in M_LAYERS:
            X = cap['oproj'][l][0, -1].float().cpu().numpy()
            Wl = model.model.layers[l].self_attn.o_proj.weight.detach().\
                float().cpu().numpy()
            for h in range(n_heads):
                sl = slice(h * hd, (h + 1) * hd)
                gains[(l, h)] = float(u @ (Wl[:, sl] @ X[sl]))
        ch = [(l, h) for (l, h), g in
              sorted(gains.items(), key=lambda x: -x[1]) if g > 0][:TOP_HEADS]
        carrier[i] = ch
        rival_info[i] = {'rival_id': rival,
                         'rival_tok': tok.decode([rival]).strip()}
        _, fact, _, _ = p2759.fact_tokens(rows[i], tok)
        fact_set = set(int(x) for x in fact)
        ids = id_list[i]
        per_head_src = []
        for (l, h) in ch:
            A = att[l][0, h, -1].float().cpu().numpy()
            V = cap['v'][l][0].float().cpu().numpy()
            v_s = V[:, (h // group) * hd:((h // group) + 1) * hd]
            Wl = model.model.layers[l].self_attn.o_proj.weight.detach().\
                float().cpu().numpy()
            o_s = (Wl[:, h * hd:(h + 1) * hd] @ v_s.T).T
            g_s = (o_s @ u) * A
            pos_mass = float(np.maximum(g_s, 0).sum())
            cls_mass = {'fact': 0.0, 'content': 0.0, 'other': 0.0}
            for s in range(len(ids)):
                if g_s[s] <= 0:
                    continue
                if s in fact_set:
                    c_ = 'fact'
                else:
                    txt = tok.decode([ids[s]])
                    n_al = sum(ch2.isalnum() for ch2 in txt)
                    c_ = 'content' if n_al >= 2 else 'other'
                cls_mass[c_] += float(g_s[s])
            per_head_src.append({
                'layer': l, 'head': h,
                'class_share': {kk: (v / pos_mass if pos_mass > 0 else 0.0)
                                for kk, v in cls_mass.items()}})
        src_stats[i] = per_head_src

    fam_src = {}
    for f in G3_EXPECTED:
        rows_f = [int(i) for i in wrong_idx if fam_arr[i] == f]
        if rows_f and src_stats.get(rows_f[0]):
            fam_src[f] = {
                'fact': float(np.median([np.mean([hh['class_share']['fact']
                                                  for hh in src_stats[i]])
                                         for i in rows_f])),
                'content': float(np.median([np.mean(
                    [hh['class_share']['content'] for hh in src_stats[i]])
                    for i in rows_f])),
                'other': float(np.median([np.mean(
                    [hh['class_share']['other'] for hh in src_stats[i]])
                    for i in rows_f])),
                'rival_toks': dict(
                    (rival_info[i]['rival_tok'],
                     1) for i in rows_f[:8])}

    # ---------- C002 interventions -----------------------------------------
    def run_intervention(i, pairs=None, bsub_vec=None):
        abl_state['pairs'] = frozenset(pairs or [])
        bsub_state['on'] = bsub_vec is not None
        bsub_state['vec'] = bsub_vec
        try:
            with torch.inference_mode():
                t = torch.tensor([id_list[i]], device=device)
                o = model(t)
                return int(o.logits[0, -1].float().argmax())
        finally:
            abl_state['pairs'] = frozenset()
            bsub_state['on'] = False
            bsub_state['vec'] = None

    z63 = np.load(P2763 / 'bias_dirs.npz', allow_pickle=False)
    v_rows = z63['v_rows']

    flips = {'abl': {}, 'bsub': {}, 'combo': {}}
    for i in map(int, wrong_idx):
        v = v_rows[i] / np.linalg.norm(v_rows[i])
        v_t = torch.tensor(v.astype(np.float32), device=device)
        a1 = run_intervention(i, pairs=carrier[i])
        a2 = run_intervention(i, bsub_vec=v_t)
        a3 = run_intervention(i, pairs=carrier[i], bsub_vec=v_t)
        flips['abl'][i] = bool(a1 == tgt_ids[i])
        flips['bsub'][i] = bool(a2 == tgt_ids[i])
        flips['combo'][i] = bool(a3 == tgt_ids[i])
    n_abl = int(sum(flips['abl'].values()))
    n_bsub = int(sum(flips['bsub'].values()))
    n_combo = int(sum(flips['combo'].values()))

    # ---------- C003 random control (51 non-kc rows) ------------------------
    nonkc = [int(i) for i in wrong_idx if fam_arr[i] != 'knowledge_chain']
    rng = np.random.default_rng(SEED_CTRL)
    null_totals = []
    for _ in range(N_CTRL_DRAWS):
        tot = 0
        for i in nonkc:
            ls = rng.choice(M_LAYERS, TOP_HEADS)
            hs_ = rng.integers(0, n_heads, TOP_HEADS)
            pairs = [(int(l), int(h)) for l, h in zip(ls, hs_)]
            tot += int(run_intervention(i, pairs=pairs) == tgt_ids[i])
        null_totals.append(tot)
    null_totals = np.array(null_totals)
    n_abl_nonkc = int(sum(flips['abl'][i] for i in nonkc))
    f2_pass = bool(float(null_totals.mean()) <= n_abl_nonkc / 3.0)

    # ---------- C004 collateral (ws carriers x ab-correct rows) -------------
    ab_correct = [i for i in range(len(rows))
                  if fam_arr[i] == 'attribute_binding'
                  and arg_native[i] == tgt_ids[i]]
    ws_rows = [int(i) for i in wrong_idx if fam_arr[i] == 'word_sense']
    union_ws = sorted({p for i in ws_rows for p in carrier[i]})
    breaks = 0
    for (l, h) in union_ws:
        for i in ab_correct:
            breaks += int(run_intervention(i, pairs=[(l, h)]) != tgt_ids[i])
    e_collat = breaks / max(len(union_ws) * len(ab_correct), 1)

    fam_tab = {}
    for f in G3_EXPECTED:
        rows_f = [int(i) for i in wrong_idx if fam_arr[i] == f]
        fam_tab[f] = {
            'abl': int(sum(flips['abl'][i] for i in rows_f)),
            'bsub': int(sum(flips['bsub'][i] for i in rows_f)),
            'combo': int(sum(flips['combo'][i] for i in rows_f)),
            'n': len(rows_f)}

    f1_pass = bool(n_combo >= 24 and n_combo > n_bsub)
    verdict = 'combined_repair_confirmed' if (f1_pass and f2_pass) else \
        'not_confirmed'
    results = {
        'phase': 2772, 'G3_fam_wrong': fam_wrong,
        'C001': {'fam_src_composition': fam_src,
                 'carrier_heads': {str(i): [list(p) for p in carrier[i]]
                                   for i in map(int, wrong_idx)},
                 'rival_info': {str(i): rival_info[int(i)]
                                for i in map(int, wrong_idx)}},
        'C002': {'n_flips_abl': n_abl, 'n_flips_bsub': n_bsub,
                 'n_flips_combo': n_combo, 'fam_table': fam_tab,
                 'per_row': {str(i): {m: flips[m][int(i)]
                                      for m in ('abl', 'bsub', 'combo')}
                             for i in map(int, wrong_idx)},
                 'F1_pass': f1_pass},
        'C003': {'null_mean': float(null_totals.mean()),
                 'null_max': int(null_totals.max()),
                 'n_abl_nonkc': n_abl_nonkc, 'F2_pass': f2_pass},
        'C004': {'collateral_rate_ws_carriers_on_ab': e_collat,
                 'n_union_ws': len(union_ws)},
        'verdict': verdict, 'seconds': time.time() - t0}
    fc.save(OUT / 'result.json', results)
    fc.npz(OUT / 'repair_stats.npz',
           wrong_idx=wrong_idx,
           abl=np.array([flips['abl'][int(i)] for i in wrong_idx]),
           bsub=np.array([flips['bsub'][int(i)] for i in wrong_idx]),
           combo=np.array([flips['combo'][int(i)] for i in wrong_idx]),
           null_totals=null_totals)
    print('PHASE2772_DONE verdict=%s F1=%s F2=%s abl=%d bsub=%d combo=%d '
          'null_mean=%.2f collat=%.4f' %
          (verdict, f1_pass, f2_pass, n_abl, n_bsub, n_combo,
           float(null_totals.mean()), e_collat), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        (OUT / 'crash.txt').write_text(traceback.format_exc(),
                                       encoding='utf-8')
        raise
