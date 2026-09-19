"""Phase 2761 addendum A: matched-magnitude random-direction control.

Phase 2761 K3 failed only on the control bound (28.6% random flip rate, but
that pooled alpha up to 1.0 = 100% of hidden norm). This addendum reruns the
random-direction control at alpha = 0.1 ONLY (the magnitude at which the
targeted direction flips 71% of kc (row, layer) cells), across all 36
residual positions, on the same 65 wrong rows. Preregistered comparison:
targeted per-cell flip share (from fault_scores.npz alpha_res == 0.1) vs
random per-cell flip share at the same magnitude; direction-specificity =
targeted > random with per-row paired bootstrap (1000, seed 2761011).
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
OUT = BASE / 'phase2761' / 'qwen4_kc_fault'
SEED = 2761001
BOOT_SEED = 2761011


def main():
    t0 = time.time()
    assert not (OUT / 'addendum_result.json').exists()
    import torch
    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic'] if r['kind'] == 'controlled_relation']
    z = np.load(OUT / 'fault_scores.npz', allow_pickle=False)
    wrong_idx = z['wrong_idx']
    fam = z['fam'][wrong_idx]
    alpha_res = z['alpha_res']
    tgt_ids = np.array([rows[i]['target'] for i in wrong_idx], dtype=np.int64)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    W = model.lm_head.weight.detach().float()

    state = {'layer': None, 'alpha': None, 'vec': None}

    def hook(module, args, output):
        if state['layer'] is None:
            return None
        out = output[0] if isinstance(output, tuple) else output
        h = out[0, -1]
        out[0, -1] = h + state['alpha'] * h.norm() * state['vec']
        return None

    handles = {i: model.model.layers[i].register_forward_hook(hook)
               for i in range(n_layers)}
    rng = np.random.default_rng(SEED)
    rand_flip = np.zeros((len(wrong_idx), n_layers), dtype=bool)
    try:
        with torch.inference_mode():
            for j, i in enumerate(wrong_idx):
                row = rows[i]
                ids = torch.tensor([row['prompt_ids']], device=device)
                rv = torch.tensor(rng.normal(size=W.shape[1]), dtype=torch.float32,
                                  device=device)
                rv = rv / rv.norm()
                for ridx in range(1, n_layers + 1):
                    state['layer'] = ridx - 1
                    state['alpha'] = 0.1
                    state['vec'] = rv
                    out = model(ids).logits[0, -1]
                    rand_flip[j, ridx - 1] = int(out.argmax()) == int(tgt_ids[j])
                state['layer'] = None
                if j % 20 == 0:
                    print('P2761A %d/%d' % (j, len(wrong_idx)), flush=True)
    finally:
        for h in handles.values():
            h.remove()

    tgt_share_all = (alpha_res == 0.1)
    results = {}
    boot_lo = {}
    for f in ['knowledge_chain', 'long_distance_role', 'negation_scope', 'word_sense']:
        sel = fam == f
        t_share = float(tgt_share_all[sel].mean())
        r_share = float(rand_flip[sel].mean())
        # per-row paired diff of per-layer flip indicators
        diff = tgt_share_all[sel].astype(float) - rand_flip[sel].astype(float)
        rngb = np.random.default_rng(BOOT_SEED)
        meds = np.empty(1000)
        for b in range(1000):
            idx = rngb.integers(0, diff.shape[0], diff.shape[0])
            meds[b] = diff[idx].mean()
        lo, hi = np.percentile(meds, [2.5, 97.5])
        results[f] = {'n_wrong': int(sel.sum()),
                      'targeted_flip_share_at_0.1': t_share,
                      'random_flip_share_at_0.1': r_share,
                      'paired_diff_mean': float(diff.mean()),
                      'boot_ci95': [float(lo), float(hi)],
                      'direction_specific': bool(diff.mean() > 0 and lo > 0)}
    value = {'timestamp': fc.stamp(), 'alpha': 0.1, 'seed': SEED,
             'results': results, 'seconds': time.time() - t0,
             'note': 'addendum to phase2761; matched-magnitude random control'}
    fc.save(OUT / 'addendum_result.json', value)
    fc.npz(OUT / 'addendum_scores.npz', rand_flip=rand_flip,
           wrong_idx=wrong_idx, fam=fam)
    print('P2761A_DONE ' + json.dumps(results), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        (OUT / 'addendum_crash.txt').write_text(traceback.format_exc(),
                                                encoding='utf-8')
        raise
