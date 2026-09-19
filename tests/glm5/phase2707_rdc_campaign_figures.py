"""Native-index scientific figures; every coordinate included, no sorted or latent axes."""
from rdc_conditional_common import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    outputs=[]
    j=CAMPAIGN/'j_predictive_gates';out=j/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(3,1,figsize=(35,7),dpi=320,constrained_layout=True);stored={}
    for ax,l in zip(axes,(11,23,35)):
        with np.load(j/f'unit_errors/L{l}_weighted_global.npz') as z:g=z['coordinate_mse']
        with np.load(j/f'unit_errors/L{l}_weighted_family_language.npz') as z:f=z['coordinate_mse']
        delta=f-g;stored[f'L{l}_family_minus_global_mse']=delta
        ax.plot(np.arange(len(g)),delta,lw=.35,color='#126a88');ax.axhline(0,color='#333',lw=.4)
        ax.set_yscale('symlog',linthresh=1e-6);ax.set_title(f'L{l}: all native unit errors, weighted family-language minus weighted global')
        ax.set_xlabel('Native MLP unit 0..9727 (fixed order)');ax.set_ylabel('MSE difference (symlog)')
    fig.canvas.draw();pixelwidth=[float(ax.get_window_extent().width) for ax in axes]
    path=out/'all_unit_conditional_error_difference.png';fig.savefig(path);plt.close(fig);npz(out/'plotted_full_values.npz',**stored)
    save(out/'display_contract.json',{'source_sha':sha(Path(__file__)),'all_units':9728,'coordinate_order':'Unchanged native unit indices','normalization':'Signed MSE difference, symlog Y axis linthresh1e-6, no clipping','axis_pixel_width':pixelwidth,'meaning':'Descriptive same-block reconstruction error, not advanceprediction or causal importance'})
    outputs.append(str(path.relative_to(CAMPAIGN)))
    k=CAMPAIGN/'k_long';out=k/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,1,figsize=(18,7),dpi=220,constrained_layout=True);arr={}
    for name,color in [('H12_quadratic','#bd5538'),('H12_previousH36_quadratic','#246782')]:
        with np.load(k/f'predictions/{name}.npz') as z:e=z['coordinate_mse']
        arr[name]=e;axes[0].plot(e,lw=.55,label=name,color=color)
    axes[0].set_yscale('symlog',linthresh=.01);axes[0].legend();axes[0].set_xlabel('Every native H36 coordinate 0..2559');axes[0].set_ylabel('Held-out MSE; symlog');axes[0].set_title('K: full-state error; unequal main-fit effective capacity is a limitation')
    for name,color in [('H12_quadratic','#bd5538'),('H12_previousH36_quadratic','#246782')]:
        z=read(k/f'vocabulary/{name}.json');v=np.array([r['kl_native_to_prediction'] for r in z['states']]);arr[name+'_KL']=v
        axes[1].plot(v,lw=.7,label=name,color=color)
    axes[1].legend();axes[1].set_xlabel('All331 selected test states in frozen material order');axes[1].set_ylabel('Full-vocabulary KL');axes[1].set_title('States are correlated within32 prefixes; argmax agreement is not task correctness')
    fig.canvas.draw();pixelwidth=[float(ax.get_window_extent().width) for ax in axes]
    path=out/'full_coordinate_and_vocabulary_error.png';fig.savefig(path);plt.close(fig);npz(out/'plotted_full_values.npz',**arr)
    save(out/'display_contract.json',{'source_sha':sha(Path(__file__)),'H_coordinates':2560,'test_states':331,'coordinate_order':'Native0..2559; teststates frozen material order, not ranked byerror','normalization':'Coordinate MSE symlog0.01, full-vocabKL linear, no clipping','axis_pixel_width':pixelwidth,'scope':'Main-fit contrasts, not matched-capacity causal estimates'})
    outputs.append(str(path.relative_to(CAMPAIGN)))
    l=CAMPAIGN/'l_aligned'
    if (l/'result.json').exists():
        out=l/'figures';out.mkdir(exist_ok=True);result=read(l/'result.json');fig,axes=plt.subplots(1,2,figsize=(14,5),dpi=220,constrained_layout=True)
        for key,color in [('qwen4','#126a88'),('qwen14','#a34d2a'),('glm4','#634c99')]:
            for stage,style in [('prefill','--'),('content','-')]:
                rr=[r for r in result['models'][key]['readers'] if r['stage']==stage];maximum=max(r['H'] for r in rr)
                for target,ax in enumerate(axes):ax.plot([r['H']/maximum for r in rr],[r['accuracy'][target] for r in rr],style,color=color,label=key+' '+stage)
        for ax,target in zip(axes,('record support t','requested answer y')):
            ax.set_ylim(0,1.02);ax.set_xlabel('Fraction of native layer count; NOT coordinate alignment');ax.set_ylabel('Held-out128 reader accuracy');ax.set_title(target);ax.legend(fontsize=7)
        path=out/'prefill_content_all_layer_readers.png';fig.savefig(path);plt.close(fig)
        save(out/'display_contract.json',{'source_sha':sha(Path(__file__)),'source_result_sha':sha(l/'result.json'),'axes':'Layer fraction is display normalization; all37 or41 native checkpoints included, not assumed equivalent geometry.','limits':'External readers and2testentitygroups; prefill/content may be identical inQwen, causal mechanisms not established.'});outputs.append(str(path.relative_to(CAMPAIGN)))
    m=CAMPAIGN/'m_order'
    if (m/'result.json').exists():
        out=m/'figures';out.mkdir(exist_ok=True);arrays={};fig,axes=plt.subplots(3,1,figsize=(18,11),dpi=220,constrained_layout=True)
        for ax,family in zip(axes,('handover','category_chain','quantity_update')):
            group=[];names=[];missing=[]
            for order in ('result_first','trace_first','neutral_first'):
                bid=f'm-{family}-6-0-en-{order}';b=read(m/f'behavior_scored/{bid}.json');sid=b['result_boundary_state']
                if sid:
                    with np.load(m/f'ledgers/{sid}.npz') as z:v=z['L23_source_vectors']
                    group.append(v);names.extend([order+'/'+g for g in ('record','prompt','trace','neutral','result','other')])
                else:missing.append(order)
            if not group:ax.text(.1,.5,'No valid Result boundary for declared examples');continue
            matrix=np.concatenate(group);arrays[family]=matrix;maximum=max(float(np.abs(matrix).max()),1e-12)
            im=ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap='RdBu_r',norm=SymLogNorm(linthresh=.001,vmin=-maximum,vmax=maximum))
            ax.set_yticks(np.arange(len(names)),names,fontsize=6);ax.set_xlabel('Every native attention-output coordinate 0..2559');ax.set_title(f'M {family}, entity6 state0 English, L23'+('; missing field: '+','.join(missing) if missing else '; all three orders'))
            fig.colorbar(im,ax=ax,label='Source write (symlog)')
        fig.canvas.draw();pixelwidth=[float(ax.get_window_extent().width) for ax in axes]
        path=out/'all_coordinate_source_group_vectors.png';fig.savefig(path);plt.close(fig);npz(out/'plotted_full_values.npz',**arrays)
        save(out/'display_contract.json',{'source_sha':sha(Path(__file__)),'example_selection':'All3families entity6 state0 en, all3orders; specified in figure code before M pilot','coordinate_order':'0..2559; six sourcegroups include every actual source; no ranking','color':'Symmetric per-family maximum, symlog1e-3; no clipping','axis_pixel_width':pixelwidth,'scope':'Three declared illustrative paired-record sets, not a comprehensive statistical test; full summaries separate'});outputs.append(str(path.relative_to(CAMPAIGN)))
    save(CAMPAIGN/'figure_index.json',{'timestamp':stamp(),'source_sha':sha(Path(__file__)),'figures':outputs,'purpose':'Full-index scientific comparisons; raw values and display contracts retained; no latent semantic geometry.'})
    print('CAMPAIGN_FIGURES',len(outputs),flush=True)


if __name__=='__main__':main()
