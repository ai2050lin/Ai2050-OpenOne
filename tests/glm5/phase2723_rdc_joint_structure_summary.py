"""Full-coordinate conditional output geometry and complete-unit cross-term bookkeeping."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdc_joint_common import *
from rdc_relation_native_parameters import parameter,decode

OUT=BASE/'extension/native_regimes'

def main():
    guard(4*1024**2);result=read(OUT/'result.json');records=json.loads(gzip.decompress((OUT/'rows_corrected.json.gz').read_bytes()))
    # Complete native activations and all original down columns; no column selection.
    exact={};precision_profiles={}
    for b in (6,16,34):
        activations=[];native=[]
        for r in records:
            with np.load(OUT/'factors'/f'{r["sample_id"]}.npz') as z:
                j=z['positions'].tolist().index(r['position']);activations.append(unbits(z[f'L{b}_activation'][j]));native.append(unbits(z[f'L{b}_mlp'][j]))
        aa=np.array(activations,dtype=float);nn=np.array(native,dtype=float);w=decode(parameter(ROOT,f'model.layers.{b}.mlp.down_proj.weight')).astype(float)
        reconstructed=aa@w.T;diagonal=aa*aa@(np.sum(w*w,axis=0)/2560);recon_energy=np.mean(reconstructed*reconstructed,1)
        for condition in result['condition_summaries']:
            stage,role=condition.split('_',1);ix=[i for i,r in enumerate(records) if r['stage']==stage and r['role']==role]
            error=reconstructed[ix]-nn[ix]
            exact[condition,b]={'FP64_reconstructed_MLP_energy':float(recon_energy[ix].mean()),'FP64_sum_unit_diagonal_energy':float(diagonal[ix].mean()),
                'FP64_cross_unit_energy':float((recon_energy[ix]-diagonal[ix]).mean()),
                'FP64_energy_to_diagonal_ratio':float(recon_energy[ix].mean()/diagonal[ix].mean()),
                'FP64_vs_native_BF16_relative_MSE':float(np.mean(error*error)/np.mean(nn[ix]*nn[ix]))}
            precision_profiles[condition+f'_L{b}']=np.stack([error.mean(0),np.mean(error*error,0)]).astype(np.float32)
        del aa,nn,w,reconstructed
    npz(OUT/'FP64_native_factor_reconstruction_profiles.npz',**precision_profiles)
    summaries={}
    for condition,s in result['condition_summaries'].items():
        stage,role=condition.split('_',1);rr=[r for r in records if r['stage']==stage and r['role']==role]
        summaries[condition]={}
        for b in (6,16,34):
            e=s['blocks'][str(b)]['MLP_energy'];diag=s['blocks'][str(b)]['sum_unit_diagonal_energies']
            summaries[condition][str(b)]={'ratio_of_mean_native_output_to_sum_diagonal_energy':e/diag,
                'approx_cross_unit_fraction_of_native_energy':1-diag/e,
                'caveat':'Native output contains BF16 GEMM rounding; exact FP64 equality requires reconstructed output. This is aggregate interference, NOT number of necessary units.',
                'MLP_energy':e,'diagonal_energy':diag,**exact[condition,b]}
    with np.load(OUT/'corrected_training_signatures.npz') as z:signatures=np.stack([z[k] for k in ('L6_first','L16_event','L34_event')])
    with np.load(OUT/'all_native_unit_conditional_profiles.npz') as z:
        keys=[f'new_{role}_L{b}' for role in ('first','event','same_ID_non_event') for b in (6,16,34)]
        projected=np.stack([z[k][7] for k in keys]);activated=np.stack([z[k][2] for k in keys])
    figures=[]
    def finish(fig,name,description):
        path=BASE/'figures'/name;fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)
        figures.append({'path':name,'description':description,'sha256':sha(path),'scope':'All native indices, uncompressed arrays; raster overview can merge pixels.'})
    fig,axes=plt.subplots(2,1,figsize=(14,7.5),gridspec_kw={'height_ratios':[1,1.1]},layout='constrained')
    color=axes[0].imshow(np.arcsinh(signatures),aspect='auto',cmap='RdBu_r',vmin=-np.max(np.abs(np.arcsinh(signatures))),vmax=np.max(np.abs(np.arcsinh(signatures))));axes[0].set_yticks(range(3),['L6 first (corrected mean)','L16 event TRAIN mean','L34 event TRAIN mean']);axes[0].set_xlabel('Native residual coordinate 0..2559');axes[0].set_title('Complete native output prototypes; shared index is not shared function');fig.colorbar(color,ax=axes[0],label='asinh(value), scale 1')
    for role,color in [('first','#8056a5'),('event','#c65326'),('same_ID_non_event','#187e9f')]:
        rr=[r for r in records if r['stage']=='new' and r['role']==role]
        for r in rr:axes[1].plot(range(37),r['energy_by_layer'],color=color,alpha=.12,lw=.7)
        axes[1].plot(range(37),np.mean([r['energy_by_layer'] for r in rr],0),color=color,label=f'{role}: {len(rr)} probes',lw=2)
    axes[1].set_yscale('log');axes[1].set_xlabel('Actual hidden-state boundary H0..H36');axes[1].set_ylabel('Full-coordinate mean square');axes[1].legend(ncol=3,fontsize=8);axes[1].grid(alpha=.2)
    finish(fig,'natural_regimes_full_coordinate_shapes_and_paths.png','3 training full-coordinate output prototypes and every new selected natural trajectory. L6 normalization repaired transparently; new first positions not a random population sample.')
    fig,axes=plt.subplots(2,1,figsize=(16,8),layout='constrained')
    for ax,a,title in zip(axes,[activated,projected],['Native activation conditional mean: every MLP unit','Signed down-column projection to own block prototype: every MLP unit']):
        transformed=np.arcsinh(a);limit=np.max(np.abs(transformed));im=ax.imshow(transformed,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set_yticks(range(len(keys)),[k.replace('new_','') for k in keys],fontsize=8);ax.set_xlabel('Native MLP unit index 0..9727 (different layers are not identical units)');ax.set_title(title);fig.colorbar(im,ax=ax,label='asinh(value), scale 1')
    finish(fig,'natural_regimes_ALL9728_unit_activation_and_signed_projection.png','All9728 units, all3 sampled blocks and all3 new conditions. Native unit order; no Top-K, no cross-layer identity claim.')
    index=read(BASE/'figures/index.json');known={f['path'] for f in index['figures']};index['figures'] += [f for f in figures if f['path'] not in known];save(BASE/'figures/index.json',index)
    save(OUT/'structure_summary.json',{'timestamp':stamp(),'conditions':summaries,'figures':figures,'source':snapshot(Path(__file__)),
        'limits':'Near-collinear condition means plus cross-term accounting describe natural responses. Neither output prototypes nor exact parameter sums alone identify a language algorithm or causal necessity.'})
    print('STRUCTURE_SUMMARY_COMPLETE',[(b,summaries['new_event'][str(b)]['ratio_of_mean_native_output_to_sum_diagonal_energy']) for b in (6,16,34)],flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
