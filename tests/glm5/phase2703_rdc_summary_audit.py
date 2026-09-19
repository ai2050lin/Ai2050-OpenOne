"""Exact summary counts and full-native-coordinate scientific figures."""
from rdc_conditional_common import *
OUT=CAMPAIGN/'i_factorial'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    rows=read(OUT/'material.json');tr,va,te=splits(rows);result=read(OUT/'result.json')
    y=np.array([[r['fact_truth'],r['expected_yes']] for r in rows],int);old=[]
    for l in (12,24,36):
      for mode in ('C','UVC'):
        with np.load(OUT/f'predictions/old_H{l}_{mode}.npz') as z:p=z['prediction'][te]
        old.append({'model':f'old_H{l}_{mode}','test_n':len(te),'correct':np.sum((p>.5)==y[te],0).tolist(),'mse':float(np.mean((p-y[te])**2))})
    behavior=[read(OUT/f'behavior/{r["sample_id"]}.json') for r in rows]
    sums={'first_correct':sum(b['first_token_correct'] for b in behavior),'pair_correct':sum(b['pair_correct'] for b in behavior),
      'heldout_first_correct':sum(behavior[i]['first_token_correct'] for i in te),'heldout_pair_correct':sum(behavior[i]['pair_correct'] for i in te)}
    errors=[]
    with np.load(OUT/'predictions/H24_shared.npz') as z:p=z['prediction']
    for j,i in enumerate(te):
        if ((p[j]>.5)!=y[i]).any():errors.append({'sample_id':rows[i]['sample_id'],'record':rows[i]['record'],'claim':rows[i]['positive_statement'],'question':rows[i]['question'],'target_support_answer':y[i].tolist(),'predicted_scores':p[j].tolist(),'native_first_token':behavior[i]['argmax_text']})
    stats={'timestamp':stamp(),'frozen_old_on_same_test':old,'native_behavior':sums,'shared_H24_error_cases':errors,
      'token_instances':sum(len(r['prompt_ids']) for r in rows),'scanned_H_scalars':sum(len(r['prompt_ids'])*37*2560 for r in rows),
      'saved_panel_H_scalars':sum(len(r['prompt_ids'])*37*2560 for r in rows if r['full_token_panel']),
      'role_scalars':4096*37*3*2560,'native_scalars':4096*3*(3*9728+3*2560),
      'bytes':{name:sum(p.stat().st_size for p in (OUT/name).rglob('*') if p.is_file()) for name in ('fields','moments','features','models','predictions')},
      'main_comparisons':len(result['results']),'capacity_sensitivity_comparisons':len(read(OUT/'capacity_audit.json')['results'])}
    save(OUT/'summary_audit.json',stats)
    # All2560 coordinates in fixed index order, never a Top-K subset.
    figout=OUT/'figures';figout.mkdir(exist_ok=True)
    with np.load(OUT/'features/interactions.npz') as z:interaction=z['support_query_interaction_rms']
    fig,axes=plt.subplots(2,4,figsize=(28,8),constrained_layout=True,dpi=450)
    scale=max(float(np.max(interaction)),1e-12)
    for i,ax in enumerate(axes.flat):
        im=ax.imshow(interaction[i],aspect='auto',origin='lower',interpolation='nearest',norm=SymLogNorm(linthresh=.001,vmin=0,vmax=scale),cmap='magma')
        ax.set_title(rows[i*512]['family']);ax.set_xlabel('Native coordinate 0..2559');ax.set_ylabel('H checkpoint 0..36')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='RMS support-query contrast; symlog color, raw arrays retained')
    fig.canvas.draw();interaction_panel_pixel_width=[float(ax.get_window_extent().width) for ax in axes.flat]
    fig.savefig(figout/'all_coordinate_interactions.png');plt.close(fig)
    with np.load(OUT/'features/all_token_moments.npz') as z:means=z['mean'];std=z['std']
    names=read(OUT/'features/all_token_moments.json')['groups'];idx=names.index('taxonomy_chain_en')
    fig,axes=plt.subplots(1,2,figsize=(20,5),constrained_layout=True,dpi=350)
    maximum=max(float(np.max(np.abs(means[idx]))),1e-12)
    im=axes[0].imshow(means[idx],origin='lower',aspect='auto',interpolation='nearest',cmap='RdBu_r',vmin=-maximum,vmax=maximum)
    fig.colorbar(im,ax=axes[0],label='Raw token-weighted mean');axes[0].set_title('taxonomy_chain_en: raw mean, all tokens')
    normalized=means[idx]/np.maximum(std[idx],1e-12)
    im=axes[1].imshow(normalized,origin='lower',aspect='auto',interpolation='nearest',cmap='RdBu_r',vmin=-3,vmax=3)
    fig.colorbar(im,ax=axes[1],label='Mean / tokenwise population SD; display clipped at +/-3');axes[1].set_title('Scale-normalized companion; not a semantic map')
    for ax in axes:ax.set_xlabel('Native coordinate 0..2559');ax.set_ylabel('H checkpoint 0..36')
    fig.canvas.draw();moment_panel_pixel_width=[float(ax.get_window_extent().width) for ax in axes]
    fig.savefig(figout/'raw_and_normalized_full_coordinate_field.png');plt.close(fig)
    save(figout/'display_contract.json',{'source_moment_sha':sha(OUT/'features/all_token_moments.npz'),'source_interaction_sha':sha(OUT/'features/interactions.npz'),
      'coordinate_order':'0..2559 unchanged, H0..H36, no ranking or PCA','interaction_normalization':'Symlog color only; no values dropped; full raw arrays downloadable',
      'moment_normalization':'raw tokenweightedmean side-by-side mean/populationSD; all4096cases descriptive, not training-only fit; standardized colorclip+-3 explicit','standardized_clipped_cells':int(np.sum(np.abs(normalized)>3)),
      'interaction_panel_pixel_width':interaction_panel_pixel_width,'moment_panel_pixel_width':moment_panel_pixel_width,'matplotlib_version':matplotlib.__version__})
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
