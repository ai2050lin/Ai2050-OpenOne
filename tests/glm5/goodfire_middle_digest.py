"""Descriptive closeout totals without refitting or changing frozen screens."""
import argparse
import json
import numpy as np
import goodfire_middle as M
C=M.C


def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=M.SPECS);a=p.parse_args();M.configure(a.model)
    report=C.read(M.OUT/'report.json');layers=report['prediction']['layers'];regimes=report['regimes']
    pools={k:0. for k in ['direct','composed','mean','exact_token']};rows=[];joint=0
    for row in layers:
        b=row['block'];r=next(x for x in regimes if x['block']==b)
        with np.load(M.OUT/f'B{b}-prediction.npz') as z:
            direct=z['A_holdout_conditional_mse'];comp=z['A_holdout_composed_mse'];mean=z['A_holdout_mean_mse'];exact=z['A_holdout_exact_token_background_mse']
            for k,v in [('direct',direct),('composed',comp),('mean',mean),('exact_token',exact)]:pools[k]+=float(v.astype(np.float64).sum())
            joint+=int((z['A_composed_candidate']&z['A_candidate']).sum())
        rows.append(dict(block=b,DN_mass=r['holdout_DN_mass_mean'],negative_derivative=r['holdout_negative_silu_derivative_mean'],
            G=row['counts']['G'],V=row['counts']['V'],A=row['counts']['A'],composed=row['composed_candidates'],
            directA_over_mean=row['holdout']['A']['total_MSE_ratio']))
    natural=report['intervention']['natural'];cap=report['intervention']['capability']
    natural_zero=[{k:r[k] for k in ['block','unit','n_total','n_matched','argmax_changes_all','mean_nll_change_all','mean_kl_all']}
                  for r in natural['rows'] if r['condition']=='native' and r['dose']==0]
    cap_native=[r for r in cap['rows'] if r['condition']=='native']
    data=dict(units=report['n_units'],layers=rows,counts={f:sum(x['counts'][f] for x in layers) for f in ['G','V','A']},
        composed_count=sum(x['composed_candidates'] for x in layers),composed_and_direct_screen=joint,
        pooled_A={k+'_over_mean':v/pools['mean'] for k,v in pools.items()},
        composed_over_direct=pools['composed']/pools['direct'],natural_zero=natural_zero,
        capability_native_passed=sum(r['capability_gate'] for r in cap_native),capability_native_total=len(cap_native),
        capability_baseline_correct=sorted(set(r['baseline_correct'] for r in cap_native)),
        capability_losses=[dict(block=r['block'],unit=r['unit'],dose=r['dose'],lost=r['lost'],gained=r['gained'],upper=r['lost_fraction_upper95']) for r in cap_native],
        replay={s:{k:report['intervention'][s][k] for k in ['replay_passed','restore_all','max_replay_kl','norm_controls_valid','norm_controls_total']} for s in ['natural','capability']},
        one_step=report['neighbor_summary']['one_step'],rollout_final=report['neighbor_summary']['rollout'][-1],
        scope='Post-analysis descriptive summaries only. No new selection, fitting, independent confirmation or pooled cross-model estimate.')
    M.write(M.OUT/'descriptive-digest.json',data);print(json.dumps(data,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
