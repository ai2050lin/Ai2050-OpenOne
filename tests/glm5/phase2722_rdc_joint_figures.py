"""Full-coordinate, uncompressed scientific views of event and readout discrimination."""
from rdc_joint_common import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    guard(8*1024**2);out=BASE/'figures';extension=BASE/'extension';added=[]
    def done(fig,name,description):
        fig.savefig(out/name,dpi=160,bbox_inches='tight');plt.close(fig)
        added.append({'path':name,'sha256':sha(out/name),'view':description})
    with np.load(BASE/'layer_atlas/full_coordinate_moments.npz') as z:mu=z['train_mean']
    fig,axes=plt.subplots(2,1,figsize=(20,8),constrained_layout=True);transformed=np.arcsinh(mu);vmax=float(np.abs(transformed[[0,2]]).max())
    for ax,role,label in zip(axes,[0,2],['First position','Ordinary anchors']):
        im=ax.imshow(transformed[role],aspect='auto',origin='lower',interpolation='nearest',cmap='coolwarm',vmin=-vmax,vmax=vmax)
        ax.set(xlabel='All 2560 native coordinates, unchanged order',ylabel='Completed blocks',title=label+' - asinh(raw training mean / 1), shared scale; low values retained, no clipping')
        fig.colorbar(im,ax=ax,label='asinh(raw mean / 1)')
    done(fig,'all_layer_native_mean_asinh_v2.png','Presentation-only shortened colorbar label; original array, shared scale and every native coordinate unchanged. Previous image retained on disk.')
    trace=json.loads(gzip.decompress((extension/'event_trace/rows.json.gz').read_bytes()))
    fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True,constrained_layout=True)
    for ax,split in zip(axes,['train','confirmation']):
        for role,color in [('event','#ae371b'),('same_ID_non_event','#147ca8')]:
            rr=[r for r in trace if r['split']==split and r['role']==role]
            for j,r in enumerate(rr):ax.plot(range(37),r['energy_by_layer'],color=color,alpha=.55,lw=1.1,label=f'{role}, n={len(rr)}' if j==0 else None)
        ax.set(yscale='log',xlabel='Layer boundary (0 = embedding)',ylabel='Full-coordinate mean square',title=split+' - all selected event/control trajectories')
        ax.axvspan(12,23,color='gray',alpha=.07);ax.legend(fontsize=8)
    done(fig,'event_all_layer_full_coordinate_energy.png','All traced event/control tokens, all37 boundaries; full-coordinate energies, no state-coordinate selection. Same-ID controls may repeat across pairs.')
    with np.load(extension/'all_token_event_coordinate_moments.npz') as z:old=z['train_event']
    with np.load(extension/'tail_confirmation/full_coordinate_moments.npz') as z:new=z['pooled_event'];normal=z['pooled_non_event']
    value=np.vstack([np.sqrt(np.maximum(x[1],0)) for x in (old,new,normal)]);labels=[f'{name} H{layer}' for name in ('train event','new event','new non-event') for layer in (12,23,36)]
    fig,ax=plt.subplots(figsize=(20,5),constrained_layout=True);view=np.arcsinh(value);im=ax.imshow(view,aspect='auto',interpolation='nearest',cmap='viridis',vmin=0,vmax=float(view.max()))
    ax.set_yticks(range(len(labels)),labels);ax.set(xlabel='All 2560 native coordinates, unchanged order',title='Per-coordinate RMS on original tokens - event conditioning is numerical, not semantic')
    fig.colorbar(im,ax=ax,label='asinh(coordinate RMS / 1)')
    done(fig,'event_and_confirmation_all_native_coordinate_RMS.png','Every native coordinate, RMS aggregation separately for training events, new events and new non-events. Same un-clipped asinh scale; primary all-sample denominator unchanged.')
    temp=read(extension/'temperature/result.json')['routes'];fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    methods=['MSE','MSE_train_temperature','MSE_validation_temperature','KL_fit'];short=['MSE','train temp','val temp','KL-fit']
    for ax,split in zip(axes,['test','confirmation']):
        for key,item in temp.items():ax.plot(range(4),[item['splits'][split][m]['KL'] for m in methods],marker='o',label=key)
        ax.set_xticks(range(4),short);ax.set(ylabel='Mean KL over ALL 151936 vocabulary IDs',title=split+' - state objective vs scalar confidence vs KL objective')
        ax.legend(fontsize=7)
    done(fig,'full_vocabulary_temperature_vs_KL_fit.png','Frozen routes, positive temperature cannot change argmax. Validation-only scalar calibration explicitly distinguished from train fit; no test/fresh tuning.')
    index=read(out/'index.json');prior=index['figures'];replace={'all_layer_native_mean_asinh.png',*[r['path'] for r in added]}
    save(out/'index.json',{'timestamp':stamp(),'figures':[r for r in prior if r['path'] not in replace]+added,'source':snapshot(Path(__file__)),
        'presentation_revision':'Original asinh PNG retained; index points to shorter unclipped colorbar label. Scientific numbers unchanged.'})
    print('EXTENSION_FIGURES',added,flush=True)


if __name__=='__main__':main()
