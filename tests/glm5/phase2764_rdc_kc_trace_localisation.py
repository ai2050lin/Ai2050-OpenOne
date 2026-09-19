"""Phase 2764: layer-localisation of the kc readout competition (zero-GPU
analysis of the 2761 fault_scores.npz lens traces, cross-referenced with 2760
gear layers and 2763 bsub repair sets).

Preregistered descriptive predictions (frozen before analysis):
  D1: for kc native-wrong rows with knows=True, the recover layer (first
      logit-lens layer where target reaches rank 1, searched in [8,35]) has
      median >= 20 (aligned with the 2760 gear plateau L20+).
  D2: the lose layer (last lens layer where target still rank 1) is >= 33 for
      a majority of kc knows rows (end-stage collapse, not mid-chain loss).
  D3: ws native-wrong rows that were repaired by 2763 bsub (alpha=0.3) show a
      "deep-positive then collapse" trace shape (peak margin >= 0 exists),
      distinguishing them from unrepaired ws rows (no rank-1 anywhere).
  D4: the 10 kc rows repaired by perm_2747/true_2747 trained states have
      recover layers and peak margins compatible with the unrepaired ones
      (repair is a readout rewrite, not a knowledge change) -- descriptive.

Outputs: result.json, trace figure (png/svg), layer_stats.npz.
"""
import json
import time

import numpy as np

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2764' / 'qwen4_kc_trace'
SRC = BASE / 'phase2761' / 'qwen4_kc_fault' / 'fault_scores.npz'
SRC2763 = BASE / 'phase2763' / 'qwen4_debias_repair' / 'behaviour_scores.npz'

PREREG = {
    'phase': 2764,
    'question': 'At which layers does the kc internally-known answer surface '
                'and where is it lost; do the 2763 bsub-repaired ws rows and '
                'trained-state-repaired kc rows show distinct trace shapes?',
    'source': 'phase2761 fault_scores.npz (lens_margin 320x37, lens_rank, '
              'knows, wrong_idx) + phase2763 behaviour_scores.npz (bsub '
              'flips, trained-state base correctness)',
    'definitions': {'recover_layer': 'first l in [8,35] with lens_margin>=0',
                    'lose_layer': 'last l in [8,36] with lens_margin>=0 '
                                  '(= recover if single crossing)',
                    'peak_layer': 'argmax lens_margin over [8,36]'},
    'predictions': {'D1': 'kc knows rows recover_layer median >= 20',
                    'D2': 'majority of kc knows rows lose_layer >= 33',
                    'D3': 'bsub-repaired ws rows: peak_margin >= 0 exists '
                          '(deep-positive collapse shape) vs unrepaired ws: '
                          'no rank-1 anywhere',
                    'D4': 'trained-state-repaired kc rows trace shape '
                          'compatible with unrepaired (descriptive)'},
    'frozen_before_analysis': True,
}


def main():
    t0 = time.time()
    cc.guard(0)
    assert not (OUT / 'result.json').exists(), 'immutable; delete before rerun'
    OUT.mkdir(parents=True, exist_ok=True)
    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    z = np.load(SRC, allow_pickle=False)
    fam = z['fam']
    wrong_idx = z['wrong_idx']
    lm = z['lens_margin']  # 320 x 37
    tgt = z['target']

    # 2763 linkage
    z63 = np.load(SRC2763, allow_pickle=False)
    bsub_flip = {}
    for i in wrong_idx:
        bsub_flip[i] = bool(z63['native__bsub_a0.3_%d' % i] == tgt[i])
    kc_repaired_by_state = {}
    for i in wrong_idx:
        kc_repaired_by_state[i] = {
            rk: bool(z63['%s__base_correct' % rk][i])
            for rk in ['perm_2747', 'true_2747']}

    rows_out = []
    for i in wrong_idx:
        m = lm[i]
        know_l = np.where(m[8:36] >= 0)[0]
        rec = int(know_l[0]) + 8 if len(know_l) else -1
        lose = int(know_l[-1]) + 8 if len(know_l) else -1
        pk = int(np.argmax(m[8:37])) + 8
        rows_out.append({
            'row': int(i), 'family': str(fam[i]),
            'knows': bool(len(know_l)),
            'recover_layer': rec, 'lose_layer': lose,
            'peak_layer': pk, 'peak_margin': float(m[pk]),
            'final_margin': float(m[36]),
            'bsub_flip': bsub_flip[i],
            'perm_repair': kc_repaired_by_state[i]['perm_2747'],
            'true_repair': kc_repaired_by_state[i]['true_2747'],
        })

    fams = ['knowledge_chain', 'long_distance_role', 'negation_scope',
            'word_sense']
    summary = {}
    for f in fams:
        sel = [r for r in rows_out if r['family'] == f]
        knows_rows = [r for r in sel if r['knows']]
        rec_layers = [r['recover_layer'] for r in knows_rows]
        lose_layers = [r['lose_layer'] for r in knows_rows]
        summary[f] = {
            'n_wrong': len(sel), 'n_knows': len(knows_rows),
            'recover_median': float(np.median(rec_layers)) if rec_layers else None,
            'lose_median': float(np.median(lose_layers)) if lose_layers else None,
            'lose_ge33_frac': (float(np.mean([l >= 33 for l in lose_layers]))
                               if lose_layers else None),
            'peak_margin_median': float(np.median([r['peak_margin']
                                                   for r in knows_rows]))
            if knows_rows else None,
        }
    # D1/D2 on kc
    kc = summary['knowledge_chain']
    d1 = {'recover_median': kc['recover_median'],
          'pass': bool(kc['recover_median'] is not None and
                       kc['recover_median'] >= 20)}
    d2 = {'lose_ge33_frac': kc['lose_ge33_frac'],
          'pass': bool(kc['lose_ge33_frac'] is not None and
                       kc['lose_ge33_frac'] > 0.5)}
    # D3 ws split by bsub repair
    ws = [r for r in rows_out if r['family'] == 'word_sense']
    ws_flip = [r for r in ws if r['bsub_flip']]
    ws_noflip = [r for r in ws if not r['bsub_flip']]
    d3 = {'n_flip': len(ws_flip), 'n_noflip': len(ws_noflip),
          'flip_knows_frac': (float(np.mean([r['knows'] for r in ws_flip]))
                              if ws_flip else None),
          'noflip_knows_frac': (float(np.mean([r['knows'] for r in ws_noflip]))
                                if ws_noflip else None),
          'flip_peak_margin_median':
              float(np.median([r['peak_margin'] for r in ws_flip]))
              if ws_flip else None,
          'pass': bool(ws_flip and ws_noflip and
                       np.mean([r['knows'] for r in ws_flip]) >
                       np.mean([r['knows'] for r in ws_noflip]))}
    # D4 kc split by trained-state repair
    kcr = [r for r in rows_out if r['family'] == 'knowledge_chain']
    rep = [r for r in kcr if r['perm_repair'] or r['true_repair']]
    norep = [r for r in kcr if not (r['perm_repair'] or r['true_repair'])]
    d4 = {'n_repaired': len(rep), 'n_not': len(norep),
          'repaired_recover_median':
              float(np.median([r['recover_layer'] for r in rep]))
              if rep else None,
          'not_repaired_recover_median':
              float(np.median([r['recover_layer'] for r in norep]))
              if norep else None,
          'repaired_peak_median':
              float(np.median([r['peak_margin'] for r in rep])) if rep else None,
          'not_peak_median':
              float(np.median([r['peak_margin'] for r in norep]))
              if norep else None}

    results = {'summary_by_family': summary, 'D1': d1, 'D2': d2, 'D3': d3,
               'D4': d4, 'rows': rows_out,
               'seconds': time.time() - t0}
    fc.save(OUT / 'result.json', results)
    fc.npz(OUT / 'trace_stats.npz',
           wrong_idx=wrong_idx, fam=fam[wrong_idx],
           lens_margin_wrong=lm[wrong_idx],
           bsub_flip=np.array([r['bsub_flip'] for r in rows_out]),
           perm_repair=np.array([r['perm_repair'] for r in rows_out]),
           recover=np.array([r['recover_layer'] for r in rows_out]),
           lose=np.array([r['lose_layer'] for r in rows_out]),
           peak=np.array([r['peak_layer'] for r in rows_out]))

    # ---------------- figure -------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {'knowledge_chain': '#c0392b', 'long_distance_role': '#27ae60',
              'negation_scope': '#e67e22', 'word_sense': '#8e44ad'}
    ax = axes[0]
    for f in fams:
        sel = [r for r in rows_out if r['family'] == f and r['knows']]
        if not sel:
            continue
        traces = np.stack([lm[r['row']] for r in sel])
        ax.plot(range(8, 37), traces[:, 8:37].mean(axis=0), lw=1.8,
                color=colors[f], label='%s (knows %d/%d)' % (f[:12],
                                                             len(sel),
                                                             sum(1 for r in rows_out if r['family'] == f)))
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.set_xlabel('layer (logit lens)'); ax.set_ylabel('mean lens margin (target - best rival)')
    ax.set_title('(a) Mean margin traces of "knows" wrong rows')
    ax.legend(fontsize=7)
    ax = axes[1]
    for f in fams:
        vals = [r['lose_layer'] for r in rows_out
                if r['family'] == f and r['knows']]
        if not vals:
            continue
        ax.hist(vals, bins=np.arange(8, 38, 2), alpha=0.55, color=colors[f],
                label=f[:12])
    ax.set_xlabel('lose layer (last rank-1 layer)'); ax.set_ylabel('# rows')
    ax.set_title('(b) Where the internally-known answer is lost')
    ax.legend(fontsize=7)
    ax = axes[2]
    ws_flip = [r for r in rows_out if r['family'] == 'word_sense' and r['bsub_flip']]
    ws_keep = [r for r in rows_out if r['family'] == 'word_sense' and not r['bsub_flip']]
    if ws_flip:
        ax.plot(range(8, 37), np.stack([lm[r['row']] for r in ws_flip])[:, 8:37].mean(axis=0),
                lw=1.8, color='#8e44ad', label='ws bsub-repaired (n=%d)' % len(ws_flip))
    if ws_keep:
        ax.plot(range(8, 37), np.stack([lm[r['row']] for r in ws_keep])[:, 8:37].mean(axis=0),
                lw=1.8, color='#bdc3c7', label='ws not repaired (n=%d)' % len(ws_keep))
    kcr_knows = [r for r in rows_out if r['family'] == 'knowledge_chain' and r['knows']]
    ax.plot(range(8, 37), np.stack([lm[r['row']] for r in kcr_knows])[:, 8:37].mean(axis=0),
            lw=1.8, color='#c0392b', label='kc knows (n=%d)' % len(kcr_knows))
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.set_xlabel('layer'); ax.set_ylabel('mean lens margin')
    ax.set_title('(c) bsub-repaired ws vs kc: collapse shape')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'kc_trace_localisation.png', dpi=150)
    fig.savefig(OUT / 'kc_trace_localisation.svg')
    print('PHASE2764_DONE ' + json.dumps(
        {'D1': d1, 'D2': d2, 'D3': {k: v for k, v in d3.items()},
         'D4': d4, 'seconds': results['seconds']}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        (OUT / 'crash.txt').write_text(traceback.format_exc(),
                                       encoding='utf-8')
        raise
