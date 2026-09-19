"""Original-axis scientific figures; no image generation, PCA or amplitude cuts."""
from rdc_construction_common import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start=time.monotonic();out=BASE/'phase2746/figures';out.mkdir(parents=True,exist_ok=True);records=[]
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.titlesize':15})
    def finish(fig,name,caption,sources):
        path=out/(name+'.png');fig.savefig(path,dpi=160,bbox_inches='tight',facecolor='white');plt.close(fig)
        records.append({'path':path.relative_to(BASE).as_posix(),'sha256':sha(path),'caption':caption,'sources':sources,
            'coordinate_policy':'All original axes retained in numerical data. Overview pixels cannot resolve each native coordinate; no amplitude/top-k selection or index resorting.'})
    reuse=read(BASE/'phase2746/runtime_reuse/result.json')
    for view,label in [(0,'Raw'),(1,'Whole-vector RMS')]:
        fig,axes=plt.subplots(4,2,figsize=(17,12),layout='constrained',sharex=True,sharey=True)
        for ax,r in zip(axes.flat,reuse['reports']):
            with np.load(BASE/r['field_path']) as z:a=z['within_step_correlation'][view,:,2];valid=z['within_step_valid'][view,:,2]
            im=ax.imshow(np.ma.array(a,mask=~valid),origin='upper',aspect='auto',cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest')
            ax.set_title(r['family']+f" | {r['source_groups']} source groups");ax.set_xlabel('Native MLP unit index (all 9,728)');ax.set_ylabel('Native block (0..35)')
            ax.set_xticks([0,2500,5000,7500,9727]);ax.set_yticks([0,12,24,35])
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label='Within-output-phase adjacent-step correlation')
        fig.suptitle(label+' | complete-unit conditional time correlation\nSame first 3 steps; phase means removed; not a semantic-reuse or cross-layer alignment claim')
        finish(fig,'all_unit_reuse_'+str(view),'Eight families; raw and vector-RMS statistics separately. Common scale[-1,1], all36by9728product units; undefined correlations masked, never set to zero.',
            [{'path':r['field_path'],'sha256':r['field_sha256']} for r in reuse['reports']])
    differential=read(BASE/'phase2746/differential/analysis/result.json')
    fig,axes=plt.subplots(1,3,figsize=(16,5.3),layout='constrained',sharey=True)
    colors=['#235d86','#b45c27','#4c8652'];labels=['Dense fixed direction','Previous MLP write','Fixed down column']
    for ax,entry in zip(axes,[12,24,35]):
        for direction in range(3):
            rr=sorted([r for r in differential['reports'] if r['family']=='all' and r['start']==entry and r['direction']==direction],key=lambda r:r['epsilon'])
            for key,style in [('postnorm_central','-'),('BF16_postnorm_central','--')]:
                ax.plot([r['epsilon'] for r in rr],[r['measures'][key]['source_group_mean']['mean'] for r in rr],style,marker='o',color=colors[direction],label=labels[direction]+(' FP32' if style=='-' else ' BF16'))
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Finite epsilon (direction RMS matched to entry H)');ax.set_title('Full remaining network from H'+str(entry));ax.grid(alpha=.2)
    axes[0].set_ylabel('Relative L2 error of postnorm central difference');axes[-1].legend(fontsize=8)
    fig.suptitle('Same-valued smooth derivative versus original BF16 finite response\nMeans over51sourcegroups; same192endpoints / three directions / three scales')
    finish(fig,'native_finite_scales','Solid=FP32 smooth reference; dashed=BF16 native finite differences. Each tail propagates through every remaining coordinate/module; three directions do not span the whole Jacobian.',
        [{'path':'phase2746/differential/analysis/result.json','sha256':sha(BASE/'phase2746/differential/analysis/result.json')}])
    pred=read(BASE/'phase2746/history_prediction/confirmation/analysis/result.json');own=read(BASE/'phase2746/history_prediction/confirmation/autonomous/analysis.json')
    frozen=read(BASE/'phase2746/history_prediction/frozen.json');names=[frozen['autonomous_direct_baseline_route'],frozen['autonomous_primary_native_constrained_route']]
    fig,axes=plt.subplots(1,3,figsize=(15,4.9),layout='constrained')
    rr=next(r for r in pred['reports'] if r['subset']=='natural' and r['horizon']=='first3_available_positions')
    for ax,key,title in [(axes[0],'postnorm_MSE','New natural same-history state error'),(axes[1],'KL_reference_prediction','New natural same-history output KL')]:
        points=[next(r for r in rr['results'] if r['route']==name)['metrics'][key] for name in names]
        vals=[p['mean'] for p in points];err=np.array([[p['mean']-p['interval95'][0] for p in points],[p['interval95'][1]-p['mean'] for p in points]])
        ax.bar([0,1],vals,yerr=err,capsize=5,color=['#608bb0','#bc7743']);ax.set_xticks([0,1],['Direct H36','General Q + native']);ax.set_title(title);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.15)
    rr=next(r for r in own['reports'] if r['subset']=='controlled')
    axes[2].bar(range(3),[r['metrics']['correct_and_stopped']['mean'] for r in rr['routes']],color=['#5a8268','#608bb0','#bc7743'])
    axes[2].set_xticks(range(3),['Native B1','Direct','General Q']);axes[2].set_ylim(0,1);axes[2].set_ylabel('Complete correct answer and EOS');axes[2].set_title('New controlled self-fed behavior')
    fig.suptitle('State fidelity, readout fidelity and self-fed correctness are distinct\nFixed old rules; new192documents and320expressions; intervals clustered by source, not coordinate')
    finish(fig,'prediction_behavior_boundary','Natural KL improves while full-state MSE worsens; controlled self-fed accuracy remains near chance. Natural repetition is displayed separately, not treated as a unique gold metric.',
        [{'path':'phase2746/history_prediction/confirmation/analysis/result.json','sha256':sha(BASE/'phase2746/history_prediction/confirmation/analysis/result.json')},
         {'path':'phase2746/history_prediction/confirmation/autonomous/analysis.json','sha256':sha(BASE/'phase2746/history_prediction/confirmation/autonomous/analysis.json')}])
    examples=gzread(BASE/'phase2746/history_prediction/confirmation/autonomous/examples.json.gz');example=examples[0]
    fig,axes=plt.subplots(3,1,figsize=(16,10),layout='constrained',sharex=True,sharey=True);values=[]
    for r in example['records']:
        with np.load(BASE/r['field_path']) as z:a=unbits(z['postnorm']) if r['route']=='native_B1_cache' else z['compiled_postnorm']
        values.append(a)
    vmax=max(np.abs(np.arcsinh(a)).max() for a in values)
    for ax,r,a in zip(axes,example['records'],values):
        im=ax.imshow(np.arcsinh(a),origin='upper',aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest')
        ax.set_title(r['route']+' | first frozen new EWT source, not outcome-selected');ax.set_ylabel('Actual emitted step (0..31)');ax.set_xlabel('All2,560original postnorm coordinates')
    fig.colorbar(im,ax=axes,label='asinh(raw postnorm), common scale, no clipping',shrink=.8)
    fig.suptitle('Separate self-fed histories: complete-coordinate trajectories\nNative-prefix initialized; no teacher refresh; same-shape initial early H and KV verified bitwise')
    finish(fig,'self_history_all_coordinates','Common asinh scale; all32stepsby2560coordinates. Different-histories state differences are not a same-input prediction-error measurement.',
        [{'path':r['field_path'],'sha256':r['field_sha256']} for r in example['records']])
    save(out/'index.json',{'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'figures':records,
        'visual_review':'Pending actual image inspection; successful rendering is not visual QA.','seconds':time.monotonic()-start})
    ledger('phase2746_scientific_figures',time.monotonic()-start);print('PHASE2746_FIGURES',len(records),flush=True)


if __name__=='__main__':main()
