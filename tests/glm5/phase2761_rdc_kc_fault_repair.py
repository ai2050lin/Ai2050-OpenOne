"""Phase 2761: localize and repair the knowledge-chain "knows internally but
cannot say it" fault in qwen3-4b (native model).

Question: knowledge_chain is the weakest relation family behaviourally while
Phase 2750 showed deep internal states respond strongly to history-value
perturbations without changing behaviour (internal-behaviour decoupling). This
phase measures, on all 320 diagnostic controlled rows (native model):
  Pass 1 (behaviour + logit lens): forward each row once; behaviour = greedy
    first answer token vs target; logit lens z_l = lm_head(final_norm(h_l))
    for l = 0..36 (hidden_states tuple); target rank and margin per layer.
    knows_internally(l in [8,35]) := min rank == 1.
  Pass 2 (resistance profile + repair, wrong rows only): inject the target
    unembed direction w = lm_head.weight[target] (unit-normalised) at residual
    index r in [1,36] (hook on layers[r-1] output, last position):
    h := h + alpha * ||h|| * w_hat, alpha in {0.1, 0.3, 1.0}.
    alpha_res(r) = min alpha flipping greedy first token to target.
  Control: per-row random unit direction (seed 2761001), same sweep at
    r in {16,24,28,31,34}.
Preregistered criteria (frozen in execution.json):
  K1: fraction of wrong knowledge_chain rows with knows_internally >= 0.5.
  K2 (fault localisation): paired per-row comparison, median alpha_res over
      r in [30,35] > median over r in [12,19] on wrong kc rows (bootstrap
      1000, seed 2761010, CI excludes 0) - late layers actively suppress the
      internally-present answer direction.
  K3 (repair): per-row best injection (argmin alpha over r in [1,36]) flips
      >= 40% of wrong kc rows; random-direction control flips <= 15%.
  K4: per-family profiles reported descriptively.
Behaviour protocol: greedy first token at final prompt position (B1-analog);
no free generation. Descriptive: the injected direction is the model's own
output row for the target - repair demonstrates bypassability of the fault,
not autonomous recovery.
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
ALPHAS = [0.1, 0.3, 1.0]
CTRL_LAYERS = [16, 24, 28, 31, 34]
CTRL_SEED = 2761001
BOOT_SEED = 2761010
BOOT_N = 1000

PREREG = {
    'phase': 2761,
    'question': 'Where does the knowledge-chain "internally knows but cannot '
                'say" fault live, and can bypassing it repair behaviour?',
    'model': 'qwen3-4b native',
    'rows': 'all 320 diagnostic controlled_relation rows (5 families x 64)',
    'behaviour': 'greedy first token at final prompt position (B1-analog)',
    'knows_internally': 'target rank 1 in logit lens z_l = '
                        'lm_head(final_norm(h_l)) for some l in [8,35]',
    'injection': 'h := h + alpha*||h||*unit(lm_head.weight[target]) at '
                 'residual index r in [1,36], alpha in [0.1, 0.3, 1.0]',
    'criteria': {'K1': 'kc wrong rows with knows_internally >= 0.5',
                 'K2': 'median alpha_res(r in [30,35]) > median(r in [12,19]) '
                       'paired per row, bootstrap CI excludes 0',
                 'K3': 'best-injection flip rate >= 0.4 on wrong kc rows; '
                       'random-direction control <= 0.15',
                 'K4': 'descriptive per-family profiles'},
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

    import torch
    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic'] if r['kind'] == 'controlled_relation']
    assert len(rows) == 320, len(rows)
    fams = sorted({r['family'] for r in rows})
    assert fams == ['attribute_binding', 'knowledge_chain', 'long_distance_role',
                    'negation_scope', 'word_sense'], fams

    from phase2662_symmetric_mapping_contract import load_native
    model, tok = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    final_norm = model.model.norm
    lm_head = model.lm_head
    W = lm_head.weight.detach().float()  # vocab x d

    # ---------- Pass 1: behaviour + logit lens ----------
    beh = np.empty(len(rows), dtype=np.int64)
    hs_all = []
    with torch.inference_mode():
        for i, r in enumerate(rows):
            ids = torch.tensor([r['prompt_ids']], device=device)
            out = model(ids, output_hidden_states=True)
            hs = torch.stack([h[0, -1].float() for h in out.hidden_states])  # 37 x d
            hs_all.append(hs.cpu())
            beh[i] = int(out.logits[0, -1].argmax())
            if i % 80 == 0:
                print('P2761 PASS1 %d/320' % i, flush=True)
    hs_all = torch.stack(hs_all)  # 320 x 37 x d
    # chunked lens to bound memory
    lens_rank = np.empty((len(rows), n_layers + 1), dtype=np.int64)
    lens_margin = np.empty((len(rows), n_layers + 1), dtype=np.float64)
    tgt_ids = np.array([r['target'] for r in rows], dtype=np.int64)
    with torch.inference_mode():
        for s in range(0, len(rows), 32):
            hs = hs_all[s:s + 32].to(device)
            z = final_norm(hs) @ W.T  # b x 37 x V
            b = z.shape[0]
            nl = z.shape[1]
            tgt = torch.tensor(tgt_ids[s:s + b], device=device)
            tix = tgt.view(b, 1, 1).expand(b, nl, 1)
            zt = z.gather(2, tix).squeeze(2)  # b x nl
            zc = z.scatter(2, tix, -1e30).max(-1).values
            lens_rank[s:s + b] = (z > zt.unsqueeze(-1)).sum(-1).cpu().numpy()
            lens_margin[s:s + b] = (zt - zc).cpu().numpy()
            del z, zt, zc, hs

    fam_arr = np.array([r['family'] for r in rows], dtype=np.str_)
    wrong = beh != tgt_ids
    summary_fam = {}
    for f in fams:
        sel = fam_arr == f
        summary_fam[f] = {'n': int(sel.sum()), 'n_wrong': int(wrong[sel].sum()),
                          'behaviour_correct_rate': float((~wrong[sel]).mean())}
    knows = (lens_rank[:, 8:36] == 1).any(axis=1)
    for f in fams:
        sel = (fam_arr == f) & wrong
        summary_fam[f]['knows_rate_wrong'] = (float(knows[sel].mean())
                                              if sel.sum() else None)
    k1_kc = summary_fam['knowledge_chain']['knows_rate_wrong']
    print('P2761 PASS1_DONE wrong per fam: %s' %
          {f: summary_fam[f]['n_wrong'] for f in fams}, flush=True)

    # ---------- Pass 2: resistance profile + repair on wrong rows ----------
    wrong_idx = np.where(wrong)[0]
    inject_hook = {'layer': None, 'alpha': None, 'vec': None}

    def make_hook():
        def hook(module, args, output):
            if inject_hook['layer'] is None:
                return None
            out = output[0] if isinstance(output, tuple) else output
            h = out[0, -1]
            v = inject_hook['vec']
            out[0, -1] = h + inject_hook['alpha'] * h.norm() * v
            return None
        return hook

    handle = None
    alpha_res = np.full((len(wrong_idx), n_layers), np.inf)  # r-1 in [0,35]
    best_flip = np.zeros(len(wrong_idx), dtype=bool)
    best_alpha = np.full(len(wrong_idx), np.inf)
    best_layer = np.full(len(wrong_idx), -1, dtype=np.int64)
    ctrl_flip = np.zeros(len(wrong_idx), dtype=bool)
    handles = {i: model.model.layers[i].register_forward_hook(make_hook())
               for i in range(n_layers)}
    rng = np.random.default_rng(CTRL_SEED)
    try:
        with torch.inference_mode():
            for j, i in enumerate(wrong_idx):
                r = rows[i]
                ids = torch.tensor([r['prompt_ids']], device=device)
                tgt = int(tgt_ids[i])
                wv = W[tgt]
                wv = (wv / wv.norm()).to(device)
                hs_row = hs_all[i]  # 37 x d on CPU
                flipped_any = False
                for ridx in range(1, n_layers + 1):
                    h = hs_row[ridx].to(device)
                    inject_hook['vec'] = wv
                    for a in ALPHAS:
                        inject_hook['layer'] = ridx - 1
                        inject_hook['alpha'] = a
                        out = model(ids).logits[0, -1]
                        if int(out.argmax()) == tgt:
                            alpha_res[j, ridx - 1] = a
                            if not flipped_any or a < best_alpha[j]:
                                pass
                            break
                    inject_hook['layer'] = None
                if np.isfinite(alpha_res[j]).any():
                    best_flip[j] = True
                    la = int(np.argmin(alpha_res[j]))
                    best_layer[j] = la + 1
                    best_alpha[j] = float(alpha_res[j].min())
                # random-direction control
                rv = torch.tensor(rng.normal(size=W.shape[1]), dtype=torch.float32,
                                  device=device)
                rv = rv / rv.norm()
                cflip = False
                for ridx in CTRL_LAYERS:
                    inject_hook['vec'] = rv
                    for a in ALPHAS:
                        inject_hook['layer'] = ridx - 1
                        inject_hook['alpha'] = a
                        out = model(ids).logits[0, -1]
                        if int(out.argmax()) == tgt:
                            cflip = True
                            break
                    if cflip:
                        break
                inject_hook['layer'] = None
                ctrl_flip[j] = cflip
                if j % 20 == 0:
                    print('P2761 PASS2 %d/%d' % (j, len(wrong_idx)), flush=True)
    finally:
        for h in handles.values():
            h.remove()

    fam_wrong = fam_arr[wrong_idx]
    kc_sel = fam_wrong == 'knowledge_chain'
    kc_wrong_n = int(kc_sel.sum())
    results = {'per_family': summary_fam,
               'n_wrong_total': int(wrong.sum()),
               'K1_kc_knows_rate': k1_kc,
               'K1_pass': bool(k1_kc is not None and k1_kc >= 0.5)}
    # K2: paired comparison on kc wrong rows
    if kc_wrong_n:
        early = np.nanmin(np.where(alpha_res[kc_sel][:, 12:20] == np.inf,
                                   np.nan, alpha_res[kc_sel][:, 12:20]), axis=1)
        late = np.nanmin(np.where(alpha_res[kc_sel][:, 30:36] == np.inf,
                                  np.nan, alpha_res[kc_sel][:, 30:36]), axis=1)
        early = np.where(np.isfinite(early), early, 10.0)
        late = np.where(np.isfinite(late), late, 10.0)
        diff = late - early
        rngb = np.random.default_rng(BOOT_SEED)
        boots = np.empty(BOOT_N)
        for b in range(BOOT_N):
            idx = rngb.integers(0, len(diff), len(diff))
            boots[b] = np.median(diff[idx])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        results['K2'] = {'median_alpha_early_r12_19': float(np.median(early)),
                         'median_alpha_late_r30_35': float(np.median(late)),
                         'diff_median': float(np.median(diff)),
                         'boot_ci95': [float(lo), float(hi)],
                         'pass': bool(np.median(diff) > 0 and lo > 0)}
    else:
        results['K2'] = None
    # K3
    flip_rate = float(best_flip[kc_sel].mean()) if kc_wrong_n else None
    ctrl_rate = float(ctrl_flip[kc_sel].mean()) if kc_wrong_n else None
    results['K3'] = {'kc_wrong_n': kc_wrong_n,
                     'best_injection_flip_rate': flip_rate,
                     'random_control_flip_rate': ctrl_rate,
                     'pass': bool(flip_rate is not None and flip_rate >= 0.4
                                  and ctrl_rate is not None and ctrl_rate <= 0.15)}
    # K4 profiles
    prof = {}
    for f in fams:
        sel = fam_wrong == f
        if sel.sum():
            prof[f] = {'n_wrong': int(sel.sum()),
                       'alpha_res_median_by_r': [
                           (None if not np.isfinite(alpha_res[sel][:, rr]).any()
                            else float(np.median(alpha_res[sel][:, rr][np.isfinite(alpha_res[sel][:, rr])])))
                           for rr in range(n_layers)],
                       'flip_rate': float(best_flip[sel].mean()),
                       'ctrl_flip_rate': float(ctrl_flip[sel].mean())}
    results['K4_profiles'] = prof
    results['verdict'] = ('fault_localised_and_repaired'
                          if results['K1_pass'] and results.get('K2', {}).get('pass')
                          and results['K3']['pass'] else
                          'partially_confirmed' if results['K1_pass'] or
                          (results.get('K2') or {}).get('pass') or
                          results['K3']['pass'] else 'not_confirmed')
    results['seconds'] = time.time() - t0
    fc.npz(OUT / 'fault_scores.npz',
           beh=beh, target=tgt_ids, fam=fam_arr,
           lens_rank=lens_rank, lens_margin=lens_margin,
           wrong_idx=wrong_idx, alpha_res=alpha_res,
           best_flip=best_flip, best_alpha=best_alpha,
           best_layer=best_layer, ctrl_flip=ctrl_flip,
           knows=knows)
    fc.save(OUT / 'fault_scores_meta.json',
            {'schema': 'alpha_res: rows=wrong rows in wrong_idx order, cols=r-1 '
                       '(residual index r = 1..36, inf = no flip); '
                       'lens_rank/margin: 320 x 37 hidden_states layers'})
    fc.save(OUT / 'result.json', results)
    print('PHASE2761_DONE ' + json.dumps(
        {'verdict': results['verdict'], 'K1': results['K1_pass'],
         'K2': results.get('K2'), 'K3': results['K3'],
         'seconds': results['seconds']}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        (OUT / 'crash.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
