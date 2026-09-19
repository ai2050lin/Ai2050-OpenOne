"""Static scientific full-coordinate plots, explicitly separate raw/standardized quantities."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from rdc_prefix_common import *
OUT=CAMPAIGN/'figures'


def main():
    OUT.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    index=[];contract={'timestamp':stamp(),'source_sha':sha(Path(__file__)),
      'coordinates':'Every native residual coordinate retained in original index order. No PCA, Top-K selection, coordinate clustering or sorting.',
      'panels':'Two predeclared full panels, one English and one Chinese, shown at H0/H12/H24/H36 for every observed token. Not all512 sentences have a saved full panel.',
      'normalization':'Raw values and training-all-token per-layer per-coordinate z-scores are separate figures. SymLog color scales use a common range across all panels in each figure; no clipping. Figure titles give linthresh.',
      'conditional_profiles':'All18 declared cue types, H36 full2560 coordinates. Gray row means nonestimable, not zero. Residual nuisance fit is training-only; this does not establish pure semantics.',
      'correlations':'All2560x2560 covariance entries standardized by stored source/target coordinate std, epsilon1e-12. Same [-1,1] color limits. nsubj comes from only16 full panels, not all512 sentences.',
      'forecasts':'Full2560 coordinate MSE, original index order, shared log y-range across main/confirmation. Corrected hash controls labeled software revision, not a new confirmation.',
      'reuse':'Figures derive directly from saved field/metric files. Their source paths and SHA values are registered here. Zoomed browser display can resample pixels; query/download exact arrays.',
      'sources':{}}
    def source(p):contract['sources'][str(p.relative_to(CAMPAIGN))]=sha(p)
    def finish(fig,key,title):
        fig.savefig(OUT/f'{key}.png',dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
        index.append({'id':key,'file':key+'.png','title':title})
    rows=read(CAMPAIGN/'material_stratified.json');chosen=[next(r for r in rows if r['full_panel'] and r['language']==lang) for lang in ('en','zh')]
    with np.load(CAMPAIGN/'qwen4/all_token_moments.npz') as z:
        n=z['counts'][:2].sum();mu=z['sums'][:2].sum(0)/n;std=np.sqrt(np.maximum(z['squares'][:2].sum(0)/n-mu*mu,1e-12))
    values=[]
    for r in chosen:
        p=CAMPAIGN/f'qwen4/full_panels/{r["sample_id"]}.npz';source(p)
        with np.load(p) as z:values.append(unbits(z['h']))
    contract['panel_samples']=[{'sample_id':r['sample_id'],'text':r['text'],'language':r['language'],'split':r['split'],'tokens':len(r['prompt_ids'])} for r in chosen]
    for mode in ('raw','zscore'):
        data=values if mode=='raw' else [(v-mu[:,None])/std[:,None] for v in values]
        limit=max(float(np.max(np.abs(v[[0,12,24,36]]))) for v in data);linthresh=.05 if mode=='raw' else .25
        fig,ax=plt.subplots(4,2,figsize=(19,12),layout='constrained')
        norm=SymLogNorm(linthresh=linthresh,vmin=-limit,vmax=limit)
        for j,v in enumerate(data):
            for i,l in enumerate((0,12,24,36)):
                im=ax[i,j].imshow(v[l],aspect='auto',interpolation='nearest',cmap='RdBu_r',norm=norm,extent=(-.5,2559.5,len(v[l])-.5,-.5))
                ax[i,j].set_title(f'{chosen[j]["sample_id"]} | H{l} | all {v.shape[1]} tokens');ax[i,j].set_ylabel('Observed token index')
                if i==3:ax[i,j].set_xlabel('All native coordinate indices (0..2559)')
        fig.colorbar(im,ax=ax.ravel().tolist(),shrink=.75,label=f'{mode}; SymLog linthresh={linthresh}; common full range')
        fig.suptitle(f'Natural-prefix full-token panels | {mode} | all coordinates, no selection')
        finish(fig,'full_panels_'+mode,'自然语句全token×全坐标面板 · '+mode)
    p=CAMPAIGN/'atlas/nuisance_profiles.npz';source(p)
    with np.load(p) as z:
        names=GRAPH_NAMES[:18];raw=np.stack([z[k+'_raw_test'][3] if k+'_raw_test' in z else np.full(2560,np.nan) for k in names]);res=np.stack([z[k+'_nuisance_residual_test'][3] if k+'_nuisance_residual_test' in z else np.full(2560,np.nan) for k in names])
    limit=max(float(np.nanmax(np.abs(raw))),float(np.nanmax(np.abs(res))));fig,ax=plt.subplots(2,1,figsize=(21,10),layout='constrained');cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#bbbbbb')
    for a,v,title in zip(ax,(raw,res),('Raw held-out present-minus-absent profiles','Nuisance-residual held-out profiles (not pure semantics)')):
        im=a.imshow(v,aspect='auto',interpolation='nearest',cmap=cmap,norm=SymLogNorm(.05,vmin=-limit,vmax=limit));a.set_yticks(range(18),names);a.set_title(title);a.set_xlabel('All native H36 coordinates 0..2559')
    fig.colorbar(im,ax=ax.tolist(),shrink=.8,label='Common raw response units; SymLog linthresh=.05; no clipping')
    finish(fig,'condition_profiles','18类前缀线索：原始与混杂残差的全部坐标纹理')
    fig,ax=plt.subplots(1,2,figsize=(30,14),layout='constrained')
    for a,key in zip(ax,('adjacent_H12','ud_nsubj_H12')):
        p=CAMPAIGN/f'atlas/matrices/{key}.npz';source(p)
        with np.load(p) as z:c=z['covariance']/np.maximum(z['source_std'][:,None]*z['target_std'][None,:],1e-12)
        n=next(r['n'] for r in read(CAMPAIGN/'atlas/matrix_index.json') if r['id']==key)
        im=a.imshow(c,cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest');a.set_title(f'{key} | n={n} paired positions | all2560 x2560');a.set_xlabel('Target native coordinate');a.set_ylabel('Source native coordinate')
    fig.colorbar(im,ax=ax.tolist(),shrink=.7,label='Correlation (not causal connectivity)');finish(fig,'coordinate_pair_relations','相邻token与事后主语依存：完整2560×2560关系矩阵')
    fig,ax=plt.subplots(2,1,figsize=(22,9),sharex=True,sharey=True,layout='constrained')
    for a,sub,scope in zip(ax,('shared_rules','confirmation'),('test','confirmation')):
        for name,color in [('early_linear','#1f6799'),('full_linear','#328062'),('graph_interaction','#d78620'),('hash_interaction','#9967ac')]:
            p=CAMPAIGN/'causal_hash_control'/f'predictions/{scope}_{name}.npz' if 'hash' in name else CAMPAIGN/sub/f'predictions/{name}.npz';source(p)
            with np.load(p) as z:key='coordinate_mse' if 'hash' in name else ('H36_coordinate_mse' if scope=='test' else 'h36_coordinate_mse');v=z[key]
            a.plot(np.arange(2560),v,lw=.6,alpha=.85,label=name+(' (corrected control)' if 'hash' in name else ''),color=color)
        a.set_yscale('log');a.set_ylabel('Coordinate MSE (log)');a.set_title(scope+' | complete coordinate errors; fixed native order');a.legend(ncol=4,fontsize=9);a.grid(alpha=.2)
    ax[-1].set_xlabel('All native coordinate indices 0..2559');finish(fig,'all_coordinate_forecasts','统一规则：主测试与冻结确认的全部坐标误差')
    p=CAMPAIGN/'layer_operators/result.json';source(p);r=read(p);fig,ax=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    for x in r['single_layer_reports']:ax[0].plot(np.arange(1,37),x['by_transition_mse'],label=x['model'])
    for x in r['rollout_reports']:ax[1].plot(np.arange(13,37),x['trajectory_mse'],label=x['model'])
    for a in ax:a.set_yscale('log');a.set_xlabel('Target H checkpoint');a.set_ylabel('All-coordinate MSE (log)');a.legend(fontsize=8);a.grid(alpha=.2)
    ax[0].set_title('Observed current layer: one-step forecast');ax[1].set_title('Start H12: every later input is own prediction')
    finish(fig,'layer_rollout','单步接续与连续自预测：误差如何累积')
    save(OUT/'index.json',index);save(OUT/'display_contract.json',contract)
    guard();print('PREFIX_FIGURES_COMPLETE',len(index),usage(),CEILING-usage(),flush=True)


if __name__=='__main__':main()
