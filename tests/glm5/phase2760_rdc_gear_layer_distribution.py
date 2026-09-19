"""Phase 2760: layer distribution of conditional-gear candidate coordinates
across all 36 layers of qwen3-4b (descriptive visualization phase).

Question (user request + 2755 接续(a)): where in depth does the conditional
machinery ("条件齿轮候选") live? This phase measures, for the 2757 dual-world
panel (160 pairs x 2 worlds, native model), at the final prompt position:
  1. per-layer world-divergence relative energy (2750 formula):
     nd(l) = mean_pairs ||h_A - h_B||^2 / mean_pairs (||h_A||^2 + ||h_B||^2)/2
  2. per-layer condition-gear candidate coordinates: per pair, the top-K=32
     coordinates by |h_A - h_B|; a coordinate is a gear candidate at layer l
     iff it belongs to the top-K set of >= 60% of pairs (pooled and per family;
     families have 32 pairs each, chance level for a fixed coordinate is
     ~hypergeometric with p approx 0.015 at 60% threshold, so >= 60% is far
     above chance).
  3. per-layer top-32 energy fraction: mean over pairs of
     sum(top32 |dh|^2) / sum(all |dh|^2) (concentration of the condition
     response), plus participation ratio of the |dh|^2 profile.
Descriptive only: NO claim that these coordinates are a closed mechanism
(project discipline: 条件齿轮候选, not 条件齿轮闭合).
Outputs: result.json, layer_stats.npz, figure PNG + SVG.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import rdc_construction_common as cc
import rdc_feature_common as fc
import phase2747_rdc_material as mat2747

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2760' / 'qwen4_gear_layers'
TOPK = 32
STABILITY = 0.6

PREREG = {
    'phase': 2760,
    'question': 'Layer distribution of condition-gear candidate coordinates '
                'in qwen3-4b: is the conditional machinery localized in depth '
                '(2750 found L0-12 blind, L24+ active at energy level)?',
    'model': 'qwen3-4b native, no trained state',
    'panel': '160 pairs x 2 worlds, 2747 diagnostic controlled_relation, '
             'final prompt position, all 37 hidden states (emb + 36 layers)',
    'definitions': {'nd': 'mean_pairs ||hA-hB||^2 / mean_pairs (||hA||^2+||hB||^2)/2',
                    'gear_candidate': 'coordinate in per-pair top-32 |dh| for '
                                      '>= 60% of pairs (pooled n=160, per '
                                      'family n=32)',
                    'top32_fraction': 'mean over pairs of top-32 |dh|^2 energy '
                                      'share',
                    'participation_ratio': '(sum s)^2 / sum s^2 over '
                                           'per-pair-mean |dh|^2 profile'},
    'status': 'descriptive visualization; gear candidate is NOT a closed '
              'mechanism claim',
    'frozen_before_any_behavioural_forward': True,
}


def main():
    t0 = time.time()
    cc.guard(0)
    assert not (OUT / 'result.json').exists(), 'result.json immutable; delete before rerun'
    OUT.mkdir(parents=True, exist_ok=True)
    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic'] if r['kind'] == 'controlled_relation']
    assert len(rows) == 320, len(rows)
    pairs = defaultdict(list)
    for idx, r in enumerate(rows):
        pairs[r['pair_id']].append(idx)
    assert all(len(v) == 2 for v in pairs.values()) and len(pairs) == 160
    pair_ids = sorted(pairs)
    fam_of_pair = np.array([rows[pairs[pid][0]]['family'] for pid in pair_ids])
    fams = sorted(set(fam_of_pair.tolist()))

    from phase2662_symmetric_mapping_contract import load_native
    model, tok = load_native('qwen4')
    model.eval()
    n_layers = model.config.num_hidden_layers
    assert n_layers == 36
    device = next(model.parameters()).device

    dh_store = np.empty((len(pair_ids), n_layers + 1, model.config.hidden_size),
                        dtype=np.float32)
    hsq_store = np.empty((len(pair_ids), n_layers + 1), dtype=np.float64)
    with torch.inference_mode():
        for pi, pid in enumerate(pair_ids):
            hs = {}
            for widx in pairs[pid]:
                ids = torch.tensor([rows[widx]['prompt_ids']], device=device)
                out = model(ids, output_hidden_states=True)
                # last prompt position; hidden_states: tuple of n_layers+1
                hs[widx] = np.stack([h[0, -1].float().cpu().numpy()
                                     for h in out.hidden_states])
            a_, b_ = hs[pairs[pid][0]], hs[pairs[pid][1]]
            dh_store[pi] = b_ - a_
            hsq_store[pi] = 0.5 * ((a_ ** 2).sum(axis=1) + (b_ ** 2).sum(axis=1))
            if pi % 40 == 0:
                print('P2760 PAIR %d/160' % pi, flush=True)

    L = n_layers + 1
    nd = np.empty(L)
    top32_frac = np.empty(L)
    pr = np.empty(L)
    n_cand_pooled = np.empty(L, dtype=np.int64)
    n_cand_fam = {f: np.empty(L, dtype=np.int64) for f in fams}
    energy = np.empty(L)  # mean ||dh||^2 raw
    for l in range(L):
        d = dh_store[:, l, :].astype(np.float64)
        num = (d ** 2).sum(axis=1)
        energy[l] = float(np.mean(num))
        nd[l] = float(np.mean(num) / max(np.mean(hsq_store[:, l]), 1e-30))
        pr[l] = float(np.mean(num ** 2 / np.maximum((d ** 4).sum(axis=1), 1e-30)))
        top_idx = np.argsort(-np.abs(d), axis=1)[:, :TOPK]
        rowsq = np.take_along_axis(d, top_idx, axis=1)
        top32_frac[l] = float(np.mean(
            (rowsq ** 2).sum(axis=1) / np.maximum(num, 1e-30)))
        # stability: coordinate in top-K across pairs
        count = np.zeros(d.shape[1], dtype=np.int64)
        for k in range(top_idx.shape[0]):
            count[top_idx[k]] += 1
        n_cand_pooled[l] = int((count >= STABILITY * len(pair_ids)).sum())
        for f in fams:
            sel = np.where(fam_of_pair == f)[0]
            cf = np.zeros(d.shape[1], dtype=np.int64)
            for k in sel:
                cf[top_idx[k]] += 1
            n_cand_fam[f][l] = int((cf >= STABILITY * len(sel)).sum())
    # nd needs the norms of hA and hB: recompute via second pass over stored
    # sums (dh only insufficient) -> store hA norms too
    results = {'n_pairs': len(pair_ids), 'topk': TOPK,
               'stability_threshold': STABILITY,
               'nd_relative_energy': None, 'energy_mean_sq': energy.tolist(),
               'top32_fraction': top32_frac.tolist(),
               'participation_ratio': pr.tolist(),
               'n_gear_candidates_pooled': n_cand_pooled.tolist(),
               'n_gear_candidates_per_family': {f: n_cand_fam[f].tolist()
                                                for f in fams},
               'families': fams}
    fc.npz(OUT / 'layer_stats.npz',
           dh=dh_store, fam_of_pair=fam_of_pair.astype(np.str_),
           pair_ids=np.array(pair_ids, dtype=np.str_),
           nd=nd, energy=energy, top32_fraction=top32_frac,
           participation_ratio=pr,
           n_cand_pooled=n_cand_pooled,
           **{'n_cand_%s' % f: n_cand_fam[f] for f in fams})
    results['seconds'] = time.time() - t0
    fc.save(OUT / 'result.json', results)
    print('PHASE2760_STATS_DONE seconds=%.1f' % results['seconds'], flush=True)


if __name__ == '__main__':
    import torch
    try:
        main()
    except Exception as exc:
        import traceback
        (OUT / 'crash.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
