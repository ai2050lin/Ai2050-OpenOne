"""Scientific figures and typed evidence cards derived from locked results."""
import argparse,os,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_relay2_design as D
from source_relay2_report import reporting_log
os.environ['MPLCONFIGDIR']=str(D.OUT/'mpl-cache')
from source_binding_native_v2 import unbits

def build():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    r=D.read(D.OUT/'report.json');out=D.OUT/'figures';D.check_write(out,128<<20,operation='来源接力科研图表');out.mkdir(exist_ok=True)
    timer=time.monotonic();panels=[s for s in ['confirm','lockbox_frame'] if 'forecast' in r['panels'][s]]
    methods=['native','aware_native','random','pca','aware_pca','embedding'];labels=['native ridge','native query-aware','random ridge','PCA ridge','PCA query-aware','embedding ridge']
    colors=['#64748b','#137f79','#b6bec8','#7969ae','#c18c3e','#c27f87']
    fig,axes=plt.subplots(2,len(panels),figsize=(7*len(panels),8),squeeze=False,constrained_layout=True)
    for ci,split in enumerate(panels):
        for bi,c in enumerate(D.HEADS):
            ax=axes[bi,ci];entry=r['panels'][split]['forecast']['heads'][bi];vals=[entry['methods'][m]['write_R'] for m in methods]
            bars=ax.bar(np.arange(len(vals)),vals,color=colors);ax.axhline(0,color='#334155',lw=.8);ax.set_xticks(np.arange(len(vals)),labels,rotation=25,ha='right')
            ax.set_title(f'{split}: B{c["block"]} H{c["head"]}');ax.set_ylabel('Write MSE reduction vs discovery mean (R)')
            ax.bar_label(bars,labels=[f'{v:.3f}' for v in vals],padding=4,fontsize=9);ax.margins(y=.18)
    fig.suptitle('Exploratory local write forecasts\nAll methods here use true future V',fontsize=13)
    for ext in ['png','svg']:fig.savefig(out/f'local-forecast.{ext}',dpi=180)
    plt.close(fig);reporting_log('artifacts',1,7,timer)
    fig,axes=plt.subplots(2,len(panels),figsize=(7*len(panels),8),squeeze=False,constrained_layout=True)
    for ci,split in enumerate(panels):
        for bi,c in enumerate(D.HEADS):
            ax=axes[bi,ci];entry=r['panels'][split]['forecast']['heads'][bi];x=np.arange(2);methods2=['native','aware_native']
            for j,(kind,label,color) in enumerate([('trueV','true future V','#bd8a4b'),('predV','predicted V','#137f79'),('past_predV','predicted past V','#7c90b0')]):
                values=[entry['methods'][m]['write_R'] if kind=='trueV' else entry['methods'][m]['content_variants'][kind]['write_R'] for m in methods2]
                bars=ax.bar(x+(j-1)*.25,values,.25,label=label,color=color);ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],fontsize=8,padding=3)
            ax.set_xticks(x,['native ridge','native query-aware']);ax.axhline(0,color='#334155',lw=.8);ax.margins(y=.2)
            ax.set_title(f'{split}: B{c["block"]} H{c["head"]}');ax.set_ylabel('Write MSE reduction vs discovery mean');ax.legend(fontsize=8)
    fig.suptitle('Exploratory content prediction\nTrue future V is privileged; predicted V uses hs20',fontsize=12)
    for ext in ['png','svg']:fig.savefig(out/f'content-information.{ext}',dpi=180)
    plt.close(fig);reporting_log('artifacts',2,7,timer)
    fig,axes=plt.subplots(1,len(panels),figsize=(8*len(panels),7),squeeze=False,constrained_layout=True)
    for ci,split in enumerate(panels):
        ax=axes[0,ci];summary=r['panels'][split]['causal']['summary'];arms=list(summary);values=[summary[a]['forced_accuracy']*100 for a in arms]
        y=np.arange(len(arms));agree=[summary[a]['top1_agreement']*100 for a in arms]
        bars=ax.barh(y-.18,values,.36,color='#137f79',label='six-choice accuracy')
        second=ax.barh(y+.18,agree,.36,color='#c58c42',label='full-vocabulary top1 agreement')
        ax.set_yticks(y,arms);ax.invert_yaxis();ax.set_xlim(0,112)
        ax.set_xlabel('Next-token metrics (%); no free generation under intervention');ax.set_title(split)
        ax.bar_label(bars,labels=[f'{v:.1f}' for v in values],padding=3,fontsize=8)
        ax.bar_label(second,labels=[f'{v:.1f}' for v in agree],padding=3,fontsize=8)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),fontsize=8,ncol=1)
    fig.suptitle('Exploratory original-model interventions\nTotal rerouting; doses are not matched',fontsize=12)
    for ext in ['png','svg']:fig.savefig(out/f'path-interventions.{ext}',dpi=180)
    plt.close(fig);reporting_log('artifacts',3,7,timer)
    fig,axes=plt.subplots(2,len(panels),figsize=(7*len(panels),8),squeeze=False,constrained_layout=True)
    for ci,split in enumerate(panels):
        summary=r['panels'][split]['update']['summary'];ms=list(summary);x=np.arange(len(ms));acc=[summary[m]['forced_accuracy']*100 for m in ms];agree=[summary[m]['top1_agreement']*100 for m in ms]
        ax=axes[0,ci];ax.bar(x-.18,acc,.36,label='six-choice accuracy',color='#137f79');ax.bar(x+.18,agree,.36,label='full-vocabulary top1 agreement',color='#c58c42');ax.axhline(100/6,color='gray',ls='--',lw=1,label='six-choice chance');ax.set_ylim(0,108);ax.set_xticks(x,ms,rotation=30,ha='right');ax.set_title(split);ax.set_ylabel('%');ax.legend(fontsize=8)
        ax=axes[1,ci];values=[summary[m]['KL_mean'] for m in ms];bars=ax.bar(x,values,color='#7184a4');ax.set_xticks(x,ms,rotation=30,ha='right');ax.set_ylabel('KL(original || forecast)');ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=3);ax.margins(y=.15)
    fig.suptitle('Exploratory recursive output fidelity\nPredicted future V versus privileged past V',fontsize=13)
    for ext in ['png','svg']:fig.savefig(out/f'recurrent-output.{ext}',dpi=180)
    plt.close(fig);reporting_log('artifacts',4,7,timer)
    # Full-coordinate discovery display, with no PCA, Top-K, sorting or percentile clipping.
    h=np.load(D.OUT/D.MODEL/'discovery-native/hidden.npy',mmap_mode='r');raw=np.zeros((41,h.shape[-1]),np.float64);rms=np.zeros(h.shape[-1],np.float64)
    for lo in range(0,len(h),32):
        v=unbits(h[lo:lo+32,:,-1]).astype(np.float64);raw+=v.sum(0);rms+=np.square(v).sum((0,1))
    raw/=len(h);rms=np.sqrt(rms/(len(h)*41));h._mmap.close();scaled=raw/np.maximum(rms[None],1e-8)
    np.savez(out/'native-field-data.npz',raw=raw,coordinate_RMS=rms,scaled=scaled)
    fig,axes=plt.subplots(2,1,figsize=(15,7),constrained_layout=True)
    for ax,data,title in zip(axes,[raw,scaled],['Raw mean, final query position','Mean / per-coordinate RMS across discovery examples and all layers']):
        limit=np.max(np.abs(data));im=ax.imshow(data,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest',origin='lower');ax.set_title(title);ax.set_ylabel('Residual boundary (hs0 = embedding)');fig.colorbar(im,ax=ax,pad=.015,shrink=.9)
    axes[-1].set_xlabel('All 5120 native residual coordinates, original order');fig.suptitle('Discovery full field — visualization is descriptive, not a mechanism gate')
    for ext in ['png','svg']:fig.savefig(out/f'native-field.{ext}',dpi=450)
    plt.close(fig);reporting_log('artifacts',5,7,timer)
    ledger=np.load(D.OUT/D.MODEL/'discovery-parameter-ledger.npz');descs=r['panels']['discovery']['parameters']['heads']
    fig,axes=plt.subplots(2,2,figsize=(17,9),constrained_layout=True)
    for bi,c in enumerate(D.HEADS):
        b=c['block'];info=descs[bi];values=ledger[f'B{b}_score_components'].mean(0);limit=max(np.abs(values).max(),1e-10)
        ax=axes[bi,0];im=ax.imshow(values,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
        ax.set_xticks(np.arange(values.shape[1]),info['labels'],rotation=60,ha='right',fontsize=8);ax.set_yticks(np.arange(5),info['roles'],fontsize=8)
        ax.set_title(f'B{b} H{c["head"]}: signed score ledger');fig.colorbar(im,ax=ax,pad=.01)
        values=ledger[f'B{b}_coordinate_absolute_mean'];ax=axes[bi,1];im=ax.imshow(values,aspect='auto',cmap='viridis',vmin=0,vmax=values.max(),interpolation='nearest')
        ax.set_yticks(np.arange(5),info['roles'],fontsize=8);ax.set_xlabel('All 5120 residual coordinates, original order');ax.set_title('Mean absolute scalar-coordinate contribution');fig.colorbar(im,ax=ax,pad=.01)
    fig.suptitle('Discovery native K-score ledger — true Q and norm scales; algebra, not causal attribution',fontsize=14)
    for ext in ['png','svg']:fig.savefig(out/f'native-score-ledger.{ext}',dpi=450)
    plt.close(fig);ledger.close();reporting_log('artifacts',6,7,timer)
    fit=np.load(D.OUT/D.MODEL/'fit.npz');cards=[];edges=[]
    for bi,c in enumerate(D.HEADS):
        identity=f'qwen14.B{c["block"]}.H{c["head"]}.source-relay-v2';card={'id':identity,**c,'model_snapshot':D.read(D.OUT/'plan.json')['model_snapshot'],
            'input_boundary':'hs20 full prefix; Q final position, K/V each token','q_native_input_coordinates':fit['q_transform_native_idx'].tolist(),
            'kv_native_input_coordinates':fit['k_transform_native_idx'].tolist(),'fit_sha256':D.sha(D.OUT/D.MODEL/'fit.npz'),
            'panels':{s:r['panels'][s]['forecast']['heads'][bi] for s in panels},'primary_dual_panel_pass':c in r['dual_panel_primary_pass_heads'],
            'evidence_grade':'Frozen dual-panel local forecast passed; formal promotion remains0' if c in r['dual_panel_primary_pass_heads'] else 'Exploratory single-panel measurements; prospective power preflight missing; dual-panel primary claim not established',
            'source':'controlled-corpus learned linear forecasts; not an SAE or native-weight substitute','counterexamples':'See every failed primary/secondary measure and failed behavior item in report.json.',
            'natural_corpus_statistics':None,'unembedding_feature_readout':None,'individual_parameter_intervention':None,
            'missing_evidence':'Natural-corpus statistics, feature-specific unembedding interpretation and individual-parameter causality were not measured.',
            'causal_scope':'Two-head joint replacement in the original model, not individual-head necessity or all-model closure.'}
        cards.append(card);edges.append({'from':'hs20-prefix','to':identity,'type':'heldout_forecast','run':D.RUN,'artifact':'report.json','limits':card['causal_scope']})
        edges.append({'from':identity,'to':'native-head-write','type':'native_WO_calculation','artifact':f'qwen14/fit.npz:wo_{bi}','limits':'local operator; source of predicted Q/K/V is a fitted model'})
        edges.append({'from':f'native-K-parameters.B{c["block"]}','to':identity,'type':'fixed_state_algebraic_source','artifact':'qwen14/discovery-parameter-ledger.npz',
            'limits':'Full scalar parameter rows and all input coordinates; true late Q and norm scales held fixed; rounding residual explicit. No individual-parameter intervention.'})
    fit.close();cards.append({'id':'source-relay-path-masks','type':'bounded_original_model_path_intervention','panels':{s:r['panels'][s]['causal']['summary'] for s in panels},
        'limits':'Final position only, from block20; not equal norm, no free generation, other-task side effects unmeasured.'})
    cards.append({'id':'source-relay-recursive-model','type':'heldout_output_forecast','panels':{s:r['panels'][s]['update']['summary'] for s in panels},
        'limits':'Observed hs20 prefix; fitted20-block transitions. Full autoregressive continuation not measured.'})
    external=[{'id':'external.source','type':'source-role','levels':3},{'id':'external.entity','type':'entity-role','levels':2},
              {'id':'external.presentation-order','type':'surface-order','levels':2}]
    for node in external:edges.append({'from':node['id'],'to':'source-relay-path-masks','type':'controlled_material_factor','artifact':'qwen14-materials.json',
        'limits':'Experimental factors, not established native concepts or independent causal modules.'})
    if not (D.OUT/'feature-cards.json').exists():D.save('feature-cards.json',{'cards':cards,'scientific_promotion':0})
    if not (D.OUT/'typed-graph.json').exists():D.save('typed-graph.json',{'external_nodes':external,'internal_feature_ids':[c['id'] for c in cards],
        'edges':edges,'layout_semantics':'No graph distance or line denotes an untested causal connection.'})
    D.save('figure-manifest.json',{'files':{str(p.relative_to(D.OUT)):{'bytes':p.stat().st_size,'sha256':D.sha(p)} for p in out.iterdir()},
        'generator_sha256':D.sha(Path(__file__)),
        'native_display':'No coordinate pruning/sorting/PCA; raw symmetric max-absolute color scale, separate full-sample coordinate-RMS scaling. Original arrays retained. 450dpi master may be downsampled in UI.'})
    reporting_log('artifacts',7,7,timer)

if __name__=='__main__':argparse.ArgumentParser().parse_args();build()
