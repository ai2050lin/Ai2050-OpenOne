"""All-coordinate figures and numeric extraction for the append-only Phase2719 record."""
from rdc_joint_common import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = BASE/'figures'
    out.mkdir(parents=True,exist_ok=True)
    with np.load(BASE/'layer_atlas/full_coordinate_moments.npz') as z:
        mu,sd,mean,second,count = z['train_mean'],z['train_std'],z['mean'],z['second_moment'],z['counts']
    fig,axes = plt.subplots(2,1,figsize=(20,8),constrained_layout=True)
    rawmax = float(np.max(np.abs(mu[[0,2]])))
    for ax,role,label in zip(axes,[0,2],['First position','Ordinary anchors']):
        im=ax.imshow(mu[role],aspect='auto',origin='lower',interpolation='nearest',cmap='coolwarm',vmin=-rawmax,vmax=rawmax)
        ax.set(xlabel='Native residual coordinate 0..2559 (fixed order)',ylabel='Embedding / completed blocks 0..36',title=label+' — training mean, raw common color scale; no coordinate subset')
        fig.colorbar(im,ax=ax,label='Raw BF16-derived activation')
    fig.savefig(out/'all_layer_native_mean.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(20,8),constrained_layout=True)
    transformed=np.arcsinh(mu)
    vmax=float(np.abs(transformed[[0,2]]).max())
    for ax,role,label in zip(axes,[0,2],['First position','Ordinary anchors']):
        im=ax.imshow(transformed[role],aspect='auto',origin='lower',interpolation='nearest',cmap='coolwarm',vmin=-vmax,vmax=vmax)
        ax.set(xlabel='All2560 native coordinates, unchanged order',ylabel='Completed blocks',title=label+' — asinh(raw training mean / 1), same monotone scale, no clipping')
        fig.colorbar(im,ax=ax,label='asinh(raw mean / 1); low-amplitude background retained')
    fig.savefig(out/'all_layer_native_mean_asinh.png',dpi=160);plt.close(fig)
    testmean = np.sum(mean[4:]*count[4:,:,None,None],0)/count[4:].sum(0)[:,None,None]
    score = (testmean-mu)/np.maximum(sd,1e-6)
    limit = float(np.max(np.abs(score[[0,2]])))
    fig,axes = plt.subplots(2,1,figsize=(20,8),constrained_layout=True)
    for ax,role,label in zip(axes,[0,2],['First position','Ordinary anchors']):
        im=ax.imshow(score[role],aspect='auto',origin='lower',interpolation='nearest',cmap='coolwarm',vmin=-limit,vmax=limit)
        ax.set(xlabel='Native coordinate, unchanged order',ylabel='Completed blocks',title=label+' — test mean shift / TRAIN std, common un-clipped color scale')
        fig.colorbar(im,ax=ax,label='Training standard deviations')
    fig.savefig(out/'all_layer_heldout_standardized.png',dpi=160);plt.close(fig)
    layer = read(BASE/'layer_atlas/result.json')
    energy = np.asarray(layer['training_energy_by_role_layer'])
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    for i,name in enumerate(('First','Second','Ordinary')):ax.plot(np.arange(37),energy[i],label=name)
    ax.axvline(layer['training_selected_amplification_block_zero_index']+1,color='gray',ls='--',label='After train-selected block')
    ax.set(yscale='log',xlabel='Completed blocks (0 = embedding)',ylabel='Mean squared activation, ALL2560 coordinates',title='Position-conditioned numerical amplification; not an attention-sink diagnosis')
    ax.legend();fig.savefig(out/'position_layer_energy.png',dpi=150);plt.close(fig)
    relation = read(BASE/'relation_atlas/result.json')
    selected=[r for r in relation['entries'] if r['status']=='computed' and r['view']=='train_z']
    matrix,labels=[],[]
    for r in selected:
        with np.load(BASE/'relation_atlas'/r['profile']) as z:
            matrix.append(z['train_diagonal']);matrix.append(z['test_diagonal'])
        short = 'band' if r['control'].startswith('distance_band') else 'exact'
        labels.extend([r['relation']+' '+short+' train',r['relation']+' '+short+' test'])
    matrix=np.stack(matrix)
    fig,ax=plt.subplots(figsize=(20,12),constrained_layout=True)
    vmax=float(np.abs(matrix).max())
    im=ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap='coolwarm',vmin=-vmax,vmax=vmax)
    ax.set_yticks(np.arange(len(labels)),labels,fontsize=7)
    ax.set(xlabel='All2560 diagonal coordinate pairs, native order',title='H12 -> H23 typed contrasts: diagonal VIEW of exact full2560x2560 computation')
    fig.colorbar(im,ax=ax,label='Contrast of train-z products');fig.savefig(out/'typed_relation_all_coordinate_diagonals.png',dpi=160);plt.close(fig)
    first=json.loads(gzip.decompress((BASE/'prior_confirmation/first_rows.json.gz').read_bytes()).decode('utf-8'))
    first_sorted=sorted(first,key=lambda r:r['methods']['common_increment']['MSE'],reverse=True)
    material={r['sample_id']:r for r in rows()}
    prior=read(BASE/'prior_confirmation/result.json')
    summary={'timestamp':stamp(),'source':snapshot(Path(__file__)),'old_relation':read(BASE/'relation_atlas/prior_confirmation.json'),
        'new_relation_rows':[{'relation':r['relation'],'control':r['control'],'view':r['view'],'groups':r['groups'],'cosine':r['all_coordinate_train_test_cosine'],'projection':r['frozen_train_projection']} for r in relation['entries'] if r['status']=='computed'],
        'first_common_MSE_quantiles':np.quantile([r['methods']['common_increment']['MSE'] for r in first],[0,.5,.9,.99,1]).tolist(),
        'first_largest_five_diagnostic_only':[{**r,'text':material[r['sample_id']]['text'],'first_token_text':material[r['sample_id']]['tokens'][0]} for r in first_sorted[:5]],
        'representative_selection':'Largest observed error is an explicitly posthoc diagnostic; all samples remain in primary means/intervals, no removal or rule retuning.',
        'position_energy':layer['training_energy_by_role_layer'],'first_seen':prior['first_seen_strata'],
        'figures':[{'path':p.name,'sha256':sha(p),'view':'Complete requested coordinates; native order, no rank reduction. Diagonal figure explicitly only a view of full matrix, not replacement.'} for p in sorted(out.glob('*.png'))]}
    save(BASE/'observation/summary.json',summary)
    save(out/'index.json',{'timestamp':stamp(),'figures':summary['figures']})
    print('PHASE2719_SUMMARY',json.dumps({'first_quantiles':summary['first_common_MSE_quantiles'],'outliers':[(r['sample_id'],r['first_token_text'],r['methods']['common_increment']['MSE']) for r in summary['first_largest_five_diagnostic_only']],
         'train_z_relations':[(r['relation'],r['control'],r['cosine'],r['projection']['interval95']) for r in summary['new_relation_rows'] if r['view']=='train_z']},ensure_ascii=True),flush=True)


if __name__=='__main__':
    import gzip
    main()
