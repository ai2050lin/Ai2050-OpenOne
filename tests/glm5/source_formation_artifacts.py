"""Post-execution figures, native-coordinate feature cards and typed evidence graph."""
import gc,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D
os.environ['MPLCONFIGDIR']=str(D.ROOT/'tests/glm5_temp/source_formation_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from source_binding_native_v2 import unbits
from source_formation_analysis import close

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
DEST=D.OUT/'figures'

def savefig(fig,name,dpi=180):
    fig.savefig(DEST/(name+'.png'),dpi=dpi,bbox_inches='tight');fig.savefig(DEST/(name+'.svg'),dpi=dpi,bbox_inches='tight');plt.close(fig)

def overview(model,panels):
    chosen=[(s,p) for s,p in panels.items() if s!='discovery' and 'formation' in p]
    if not chosen:return
    fig,axes=plt.subplots(2,len(chosen),figsize=(7*len(chosen),8),squeeze=False)
    labels=['hs20 / 128 selected','hs20 / 128 random','Embedding / 128','True Q + predicted K','Predicted Q + true K']
    methods=['native','random','embedding','true_Q_pred_K','pred_Q_true_K']
    for col,(split,p) in enumerate(chosen):
        cs=p['formation']['results']['candidates'];errors={x['id']:x for x in p['error-split']['results']}
        a=np.array([[c['methods'][m]['improvement'] if m in c['methods'] else errors[c['id']]['methods'][m]['improvement'] for m in methods] for c in cs])
        ax=axes[0,col];im=ax.imshow(a,cmap='RdBu',vmin=-1,vmax=1,aspect='auto')
        for (i,j),v in np.ndenumerate(a):ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if abs(v)>.55 else '#151515')
        ax.set_xticks(range(5),labels,rotation=30,ha='right');ax.set_yticks(range(4),[c['id'] for c in cs]);ax.set_title(f'{model} / {split}: local write error reduction R')
        ax.set_xlabel('Same true V and original WO for all methods; last two columns use oracle inputs')
        fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='R = 1 - SSE / discovery-mean SSE\nColor clipped at [-1, 1]; numbers untruncated')
        out=p['update-output']['summary'];names=list(out);x=np.arange(len(names))
        ax=axes[1,col];ax.bar(x,[out[n]['forced_accuracy'] for n in names],color=['#999999','#477ba7','#267f77','#bb8e62'])
        ax.axhline(1/6,color='#666666',linestyle='--',label='Six-choice chance')
        ax.set_xticks(x,[n.replace('_','\n') for n in names]);ax.set_ylim(0,1);ax.set_ylabel('Six-choice accuracy');ax.set_title('Conditional rollout from real hs20; future prefix V retained')
        for i,n in enumerate(names):ax.text(i,out[n]['forced_accuracy']+.025,f"{out[n]['forced_accuracy']:.1%}\nKL {out[n]['KL_mean']:.2f}\nTop-1 match {out[n]['top1_agreement']:.1%}",ha='center',fontsize=8)
        ax.legend(loc='upper right',fontsize=8)
    fig.suptitle('Local Q/K forecast improves some writes; full-output recurrence remains inaccurate',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.95]);savefig(fig,model+'-forecast-and-rollout')

def causal(model,panels):
    chosen=[(s,p) for s,p in panels.items() if 'causal' in p]
    if not chosen:return
    fig,axes=plt.subplots(1,len(chosen),figsize=(7*len(chosen),4.8),squeeze=False)
    for col,(split,p) in enumerate(chosen):
        c=p['causal'];ax=axes[0,col];arms=['head_correct','head_other','mlp_selected','mlp_random']
        means=np.array([c['summary'][a]['margin_change'] for a in arms]);cis=np.array([c['summary'][a]['margin_change_ci_family'] for a in arms])
        for i,arm in enumerate(arms):
            admitted=c['matching']['head' if i<2 else 'mlp']['admitted']
            ax.errorbar(i,means[i],yerr=np.maximum(np.array([[means[i]-cis[i,0]],[cis[i,1]-means[i]]]),0),fmt='o',capsize=5,color='#267f77' if admitted else '#9b9b9b')
        ax.axhline(0,color='#555555',linewidth=1);ax.set_xticks(range(4),[a.replace('_','\n') for a in arms]);ax.set_ylabel('Gold minus strongest competitor: change in logit margin')
        flags='; '.join(k+(' PASS' if c['matching'][k]['admitted'] else ' FAILED') for k in ['head','mlp'])
        ax.set_title(f'{split}\nNorm matching: {flags}');ax.margins(y=.15)
    fig.text(.5,.01,'95% family bootstrap. Gray = failed norm matching; descriptive only. Small target dose does not test full necessity.',ha='center',fontsize=9)
    fig.suptitle(model+': original-model group interventions',fontsize=14);fig.tight_layout(rect=[0,.07,1,.9]);savefig(fig,model+'-matched-interventions')

def native_fields(model):
    root=D.OUT/model;rows=D.read(D.OUT/f'{model}-materials.json')['splits']['discovery'];h=np.load(root/'discovery-native/hidden.npy',mmap_mode='r')
    # No coordinate sorting or omission. Scale is common to every layer at each coordinate.
    dim=h.shape[-1];total=np.zeros((41,dim),np.float64);squares=np.zeros_like(total)
    for lo in range(0,len(rows),32):
        x=unbits(h[lo:lo+32,:,-1]);total+=x.sum(0,dtype=np.float64);squares+=np.square(x,dtype=np.float64).sum(0)
    close(h);mean=(total/len(rows)).astype('f');rms=np.sqrt(squares/len(rows));scale=np.maximum(np.sqrt(squares.sum(0)/(41*len(rows))),1e-8)
    scaled=(mean/scale).astype('f');np.savez(DEST/(model+'-native-field-data.npz'),mean=mean,rms=rms.astype('f'),coordinate_scale=scale.astype('f'),scaled_mean=scaled)
    fig,axes=plt.subplots(2,1,figsize=(15,7),constrained_layout=True)
    for ax,data,title in zip(axes,[mean,scaled],['Raw mean activation, BF16 decoded to FP32','Mean / discovery RMS per coordinate (one fixed scale across layers)']):
        bound=float(np.max(np.abs(data)));im=ax.imshow(data,aspect='auto',origin='lower',cmap='RdBu_r',vmin=-bound,vmax=bound,interpolation='nearest',extent=[-.5,dim-.5,-.5,40.5])
        ax.set_title(title);ax.set_ylabel('Raw residual boundary hs[l]');ax.set_xlabel(f'Native coordinate index: all {dim}, original order');fig.colorbar(im,ax=ax,fraction=.025,pad=.02)
    fig.suptitle(model+' discovery: final query position, all 41 boundaries including embedding; no Top-K or PCA',fontsize=12)
    # At450dpi the axes exceed5120px, retaining at least one output pixel per native coordinate.
    savefig(fig,model+'-native-field',dpi=450)

def cards(model,panels):
    root=D.OUT/model;fit=D.read(root/'formation-fit.json');rows=D.read(D.OUT/f'{model}-materials.json')['splits']['discovery'];snapshot=D.read(D.OUT/f'{model}-identity.json')['model_snapshot']
    q=np.load(root/'discovery-native/qpre.npy',mmap_mode='r');mass=np.load(root/'discovery-native/mass.npy',mmap_mode='r');units=np.load(root/'discovery-native/units.npy',mmap_mode='r')
    gate=np.load(root/'discovery-native/gate.npy',mmap_mode='r');value=np.load(root/'discovery-native/value.npy',mmap_mode='r')
    lineage=D.read(root/'native-parameter-fit.json')['lineage'];result=[]
    common={'model':model,'model_snapshot':snapshot,'method_version':D.RUN,'snapshot_scope':'Exact local BF16 snapshot; same model indices are not cross-model correspondences.',
        'natural_corpus_statistics':{'status':'not measured; controlled ledger templates only'},'unembedding_readout':{'status':'No isolated feature unembedding measurement; group downstream effects and conditional-rollout full outputs separately reported.'},
        'individual_causal_test':'Not performed. Group effect must not be assigned to each member.',
        'formal_evidence_promotion':False,'discovery_manifest_sha256':D.sha(root/'discovery-native-manifest.json'),'fit_sha256':fit['fit_sha256']}
    for bi,c in enumerate(fit['heads']):
        tag=f'B{c["block"]}H{c["head"]}';a=unbits(q[:,bi,c['head']]);m=np.array(mass[:,bi,c['head']]);strength=np.array([m[i,r['target_record']] for i,r in enumerate(rows)])
        examples=[{'id':rows[i]['id'],'text':rows[i]['content'],'record_mass':float(strength[i])} for i in np.argsort(strength)[-3:][::-1]]
        observed={}
        for split,p in panels.items():
            if 'formation' not in p:continue
            f=next(x for x in p['formation']['results']['candidates'] if x['id']==tag)
            observed[split]={'prediction':f,'scope':'in-sample discovery' if split=='discovery' else 'frozen heldout prediction'}
            if 'causal' in p:observed[split]['group_intervention']={'group':'all four selected heads','summary':p['causal']['summary'],'matching':p['causal']['matching']['head'],'pair':p['causal']['paired_comparisons']['head']}
        result.append({**common,'feature_id':model+'/'+tag,'native_index':c,'boundary':'Block index is zero based; post Q/K normalization, pre RoPE; final query Q. Head writes to raw hs[block+1].',
            'controlled_activation_statistics':{'n':len(a),'query_coordinate_mean':a.mean(0).tolist(),'query_coordinate_std':a.std(0).tolist(),'mean_correct_record_mass':float(strength.mean())},
            'typical_controlled_examples':examples,'parameter_lineage':[x for x in lineage if x['id'].startswith(tag+'_')],
            'early_native_coordinates':fit['selection'],'evidence':observed,
            'counterexamples':{'failed_prediction_panels':[s for s,x in observed.items() if not x['prediction']['passes']], 'behavior_failure_list':f'report.json#/models/{model}/panels/*/behavior/failures'},
            'upstream':'Fitted hs20->Q/K response prediction; original same-layer scalar-weight lineage separately recorded.',
            'downstream':'Original WO algebra and scope-limited group interventions; no inferred head-to-head causal graph.'})
    for bi,c in enumerate(fit['mlps']):
        idx=np.array(c['selected']);a=unbits(units[:,bi,idx]);g=unbits(gate[:,bi,idx]);v=unbits(value[:,bi,idx]);assert a.shape==(len(rows),256)
        result.append({**common,'feature_id':model+f'/B{c["block"]}MLP256','native_index':c,
            'boundary':'SwiGLU intermediate units A=silu(G)*V, down_proj input; distinct from residual coordinates.',
            'controlled_activation_statistics':{'n':len(rows),'mean':a.mean(0).tolist(),'std':a.std(0).tolist(),'positive_fraction':(a>0).mean(0).tolist(),'gate_positive_fraction':(g>0).mean(0).tolist(),'value_positive_fraction':(v>0).mean(0).tolist()},
            'parameter_lineage':{'weight':f'model.layers.{c["block"]}.mlp.down_proj.weight','columns':c['selected'],'row_range':'all residual output coordinates'},
            'typical_controlled_examples':[{'id':rows[i]['id'],'text':rows[i]['content']} for i in np.argsort(np.square(a).sum(1))[-3:][::-1]],
            'group_interventions':{s:{'matching':p['causal']['matching']['mlp'],'comparison':p['causal']['paired_comparisons']['mlp']} for s,p in panels.items() if 'causal' in p},
            'counterexamples':'Matching failures are retained. Membership selected by variance*Down-column norm does not establish semantic selectivity.',
            'upstream':'Original gate/value computation recorded; source-selectivity and individual-unit upstream causal edges untested.',
            'downstream':'Original down_proj columns; only the two-block 512-unit group was intervened on.'})
        if c['block']==39 and (root/'confirm-native39-matching-numerics.json').exists():
            result[-1]['posthoc_numerical_diagnostic']={'arithmetic':D.read(root/'confirm-native39-matching-numerics.json'),
                'coordinate_lineage':D.read(root/'confirm-matching-coordinate-lineage.json'),'scope':'Natural-state arithmetic diagnosis, not new causal evidence or a semantic coordinate label.'}
    close(q,mass,units,gate,value);return result

def main():
    report=D.read(D.OUT/'report.json');DEST.mkdir(exist_ok=True);D.check_write(DEST,64*2**20,operation='Research figures and feature cards')
    allcards=[];nodes=[];edges=[];timer=time.monotonic()
    for model,data in report['models'].items():
        panels=data['panels']
        if 'formation' not in panels.get('discovery',{}):continue
        overview(model,panels);causal(model,panels);native_fields(model);allcards+=cards(model,panels)
        nodes.append({'id':model+'/source-entity','type':'external_relation','label':'Queried source AND entity in three ledgers'})
        nodes.append({'id':model+'/hs20','type':'observed_residual_field','native_boundary':20})
        fit=D.read(D.OUT/model/'formation-fit.json')
        for c in fit['heads']:
            tag=model+f'/B{c["block"]}H{c["head"]}';nodes.append({'id':tag,'type':'head_feature_card'})
            edges.append({'from':model+'/source-entity','to':tag,'type':'observational_template_association','counterexamples':'See feature card and behavior failures'})
            edges.append({'from':model+'/hs20','to':tag,'type':'heldout_conditional_prediction','condition':'Original same-layer V retained for write prediction',
                'passes_by_panel':{s:next(x['passes'] for x in p['formation']['results']['candidates'] if tag.endswith(x['id'])) for s,p in panels.items() if s!='discovery' and 'formation' in p}})
        nodes.append({'id':model+'/four-head-group','type':'intervention_group','members':[model+f'/B{c["block"]}H{c["head"]}' for c in fit['heads']]})
        nodes.append({'id':model+'/output','type':'model_output'})
        nodes.append({'id':model+'/two-mlp-group','type':'intervention_group','members':[model+f'/B{c["block"]}MLP256' for c in fit['mlps']]})
        for c in fit['mlps']:nodes.append({'id':model+f'/B{c["block"]}MLP256','type':'native_mlp_unit_set_card','unit_count':256})
        for split,p in panels.items():
            if 'causal' not in p:continue
            for kind,group in [('head','four-head-group'),('mlp','two-mlp-group')]:
                matching=p['causal']['matching'][kind]
                edges.append({'from':model+'/'+group,'to':model+'/output','type':'scope_limited_original_model_intervention' if matching['admitted'] else 'execution_only_failed_matching',
                    'panel':split,'matching':matching,'dose':'0.5*common minimum realized branch norm; not full necessity','individual_member_effects':'not identified'})
        D.log(model+'/artifacts',1,1,timer);gc.collect()
    D.save('feature-cards.json',{'cards':allcards,'source':'Controlled corpus only; native scalar parameters, forecasts and group causality kept separate.'})
    D.save('typed-graph.json',{'nodes':nodes,'edges':edges,'formal_evidence_promotion':0,'layout_is_not_geometry':True})
    D.save('figure-manifest.json',{'files':{str(p.relative_to(D.OUT)):{'sha256':D.sha(p),'bytes':p.stat().st_size} for p in DEST.glob('*')},
        'native_display':'No coordinate removal, sorting, PCA or Top-K. Mean at final query over discovery; raw symmetric full range; scaled by per-coordinate RMS across all discovery samples and41 layers. All arrays exported.450dpi native-field export retains at least one pixel per coordinate; reduced app previews may hide thin columns.',
        'scientific_scope':'Descriptive displays; all primary and secondary scientific scripts remain frozen.'})

if __name__=='__main__':main()
