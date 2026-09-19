"""Complete coordinate figures for source organization and retrospective relation profiles."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from phase2714_rdc_full_source_history import *


def main():
    figdir=CAMPAIGN/'figures';index=read(figdir/'index.json');index=[r for r in index if not r['id'].startswith('source_')]
    contract={'timestamp':stamp(),'source_sha':sha(Path(__file__)),'full_coordinates':2560,'native_index_order':True,
      'forecast':'All2560 coordinate errors, same log-y axis for main test and fresh64. No Top-K/coordinate sorting. Current rule was selected before fresh capture.',
      'relations':'All14 retrospective UD relation profiles after same-sentence exact signed-token-distance control, full2560 diagonal coordinate products. NOT full covariance. Common unclipped SymLog linthresh0.01 color scale.',
      'fresh_scope':'Source-kernel forecast is frozen-before-fresh; new relation profile analysis was designed after fresh prediction results and is exploratory reuse, not independent new confirmation.',
      'sources':{}}
    def source(p):contract['sources'][str(p.relative_to(CAMPAIGN))]=sha(p)
    def savefig(fig,key,title):
        fig.savefig(figdir/f'{key}.png',dpi=160,bbox_inches='tight',facecolor='white');plt.close(fig)
        index.append({'id':key,'file':key+'.png','title':title,'contract':'display_contract_2714.json'})
    fig,ax=plt.subplots(2,1,figsize=(20,8),sharex=True,sharey=True,layout='constrained')
    for a,scope in zip(ax,('test','fresh')):
        for rule in RULES:
            p=OUT/f'predictions/{scope}_{rule}.npz';source(p)
            with np.load(p) as z:v=z['coordinate_mse']
            a.plot(np.arange(2560),v,lw=.6,label=rule,alpha=.8)
        a.set_yscale('log');a.set_title(scope+' | all source-position inputs vs current H12');a.set_ylabel('Coordinate MSE (log)');a.legend(ncol=4);a.grid(alpha=.2)
    ax[-1].set_xlabel('All2560 native coordinate indices');savefig(fig,'source_forecast_errors','全来源位置核：主测试与新64条冻结确认的全部坐标误差')
    p=OUT/'relations/all_coordinate_profiles.npz';source(p);r=read(OUT/'relations/result.json');names=[x['relation'] for x in r['reports']]
    values=[];labels=[]
    with np.load(p) as z:
        for split in ('test','fresh'):
            values.append(np.stack([z[k+'_'+split+'_z_delta'] for k in names]))
            labels.append([f'{x["relation"]} (n={x["splits"][split]["pairs"]}, G={x["splits"][split]["source_units"]})' for x in r['reports']])
    limit=max(float(np.max(np.abs(v))) for v in values);fig,ax=plt.subplots(2,1,figsize=(21,10),layout='constrained')
    for i,(a,v) in enumerate(zip(ax,values)):
        im=a.imshow(v,aspect='auto',interpolation='nearest',cmap='RdBu_r',norm=SymLogNorm(.01,vmin=-limit,vmax=limit));a.set_yticks(range(len(names)),labels[i],fontsize=9)
        a.set_xlabel('All2560 same-coordinate source x target products, standardized before multiplication')
        a.set_title(('Main test' if i==0 else 'Reused fresh material: exploratory relation analysis')+' | relation minus matched signed-distance control')
    fig.colorbar(im,ax=ax.tolist(),label='Mean product difference; common SymLog color; linthresh0.01; no clipping',shrink=.85)
    savefig(fig,'source_relation_profiles','14类真实依存关系：同句同距离对照后的全部坐标纹理（探索性）')
    save(figdir/'display_contract_2714.json',contract);save(figdir/'index.json',index)
    print('SOURCE_FIGURES_COMPLETE',usage(),CEILING-usage(),flush=True);guard(12*1024**2)


if __name__=='__main__':main()
