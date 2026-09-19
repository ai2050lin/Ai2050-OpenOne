"""Native-order full-coordinate atlas and actual local-training forecasts."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdc_law_common import *


def main():
    start=time.monotonic();out=BASE/'figures';out.mkdir(exist_ok=True);index=[]
    plt.rcParams.update({'font.size':9,'figure.dpi':130})
    def keep(fig,name,title,scope):
        fig.tight_layout();fig.savefig(out/name);plt.close(fig)
        index.append({'path':name,'title':title,'scope':scope,'sha256':sha(out/name)})
    atlas=read(BASE/'atlas/result.json');cohorts=('gum','ewt','cmrc','squad_qa','cmrc_qa','hotpot_qa')
    for view in ('raw_mean','train_z_mean','raw_asinh'):
        values=[]
        for c in cohorts:
            item=next(r for r in atlas['profiles'] if r['condition']=='cohort:'+c)
            with np.load(BASE/item['path']) as z:values.append(np.arcsinh(z['raw_mean']/1.) if view=='raw_asinh' else z[view])
        limit=max(float(np.max(abs(a))) for a in values);fig,axes=plt.subplots(3,2,figsize=(18,11))
        for ax,c,a in zip(axes.flat,cohorts,values):
            im=ax.imshow(a,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
            ax.set(title=c+' : '+view,xlabel='All native coordinates 0..2559',ylabel='H0..H36');fig.colorbar(im,ax=ax,fraction=.02)
        keep(fig,'full_conditions_'+view+'.png','All native coordinates across six cohorts: '+view,
            'Every37x2560 mean value supplied in original order; finite display pixels may combine visibility but no coordinates removed from source. Identical symmetric color limits across six panels, stated on colorbars. Z uses training-only per-coordinate standard deviation with floor1e-6. raw_asinh uses asinh(raw_mean/1), with transformed-value colorbars; raw and Z views retained. Observational overlapping conditions, not semantic effects.')
    fig,axes=plt.subplots(3,2,figsize=(13,15))
    for ax,item in zip(axes.flat,atlas['coordinate_covariances']):
        with np.load(BASE/item['path']) as z:a=z['correlation']
        im=ax.imshow(a,origin='lower',cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest')
        ax.set(title=f"{item['cohort']}: H12 to MLP{item['block']}",xlabel='All output coordinates',ylabel='All early coordinates')
        fig.colorbar(im,ax=ax,fraction=.02)
    keep(fig,'all_coordinate_correlations.png','Six complete2560x2560 cross-coordinate correlation matrices',
        'All matrix entries retained in downloadable arrays; display resampling at finite resolution is declared. Train sources only, no coordinate sorting/threshold/Top-K. Empirical correlation is not a computational edge.')
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for ax,item in zip(axes,atlas['joint_ledgers']):
        for r in item['reports']:
            ax.plot(range(4),np.diag(r['mean_normalized_Gram']),marker='o',label=r['cohort'])
        ax.set(title=f"block{item['block']}: nonorthogonal terms",xticks=range(4),xticklabels=[r'$\bar\phi\bar u$',r'$\bar\phi\Delta u$',r'$\Delta\phi\bar u$',r'$\Delta\phi\Delta u$'],ylabel='Squared term / total output squared norm')
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7)
    keep(fig,'signed_joint_term_context.png','Complete product terms differ by layer and context',
        'These diagonals are center-dependent and do not sum to one; signed off-diagonal interference is retained in all16entry Gram arrays. Values are not percentages of explained language or conserved physical energy.')
    result=read(BASE/'formation/initial_stable/result.json');fig,axes=plt.subplots(1,3,figsize=(16,5))
    panel=read(BASE/'formation/protocol.json')['panel'];test=np.array([i for i,r in enumerate(panel) if r['split']=='test'])
    for ax,step in zip(axes,(1e-5,1e-4,1e-3)):
        for r in result['actual_updates']:
            if r['relative_step_requested']!=step:continue
            with np.load(BASE/'formation/initial_stable/actual_updates'/f"{r['id']}.npz") as z:
                ax.scatter(z['predicted_loss_delta'][test],z['measured_loss_delta'][test],s=5,alpha=.3)
        lo,hi=ax.get_xlim();ax.plot([lo,hi],[lo,hi],'k--',lw=.8)
        ax.set(title=f'Actual parameter relative step {step:g}',xlabel='Prospective full-gradient loss delta',ylabel='Measured CE loss delta');ax.grid(alpha=.2)
    keep(fig,'actual_parameter_update_prediction.png','12 training examples x3 actual update sizes:96 held-out queries each',
        'All1152 donor-query points per panel, dependent comparisons. Full74711040native parameters, true single-example SGD. Original checkpoint read-only; FP32 smooth replica of original BF16 values. Local training influence prediction, not future answer prediction.')
    old=read(out/'index.json') if (out/'index.json').exists() else {}
    names={r['path'] for r in index};other=[r for r in old.get('figures',[]) if r['path'] not in names]
    save(out/'index.json',dict(old,timestamp=stamp(),source=snapshot(Path(__file__)),figures=index+other))
    ledger('phase2728_full_coordinate_figures',time.monotonic()-start,figures=len(index))
    print('LAW_ATLAS_FIGURES',len(index),flush=True)


if __name__=='__main__':main()
