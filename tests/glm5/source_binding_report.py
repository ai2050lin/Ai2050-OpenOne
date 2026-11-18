"""Descriptive reports, honest baseline comparisons and full-coordinate plots."""
import argparse,os,re,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D

def text_baselines(rows):
    tally={k:[] for k in ['text_parser','latest_record','source_only','entity_only']}
    for r in rows:
        content=r['content'];q=content.split('Question: ',1)[1].split('\n')[0]
        src=re.search(r'\b(?:Orion|Vega)\b',q).group();name=re.search(r'\b[A-Z][a-z]{5}\b',q).group()
        archives=re.findall(r'Archive (Orion|Vega):\n(.*?)(?=Archive |Question:)',content,flags=re.S)
        parsed={};flat=[]
        for source,body in archives:
            parsed[source]={}
            for line in body.strip().splitlines():
                entity=re.search(r'\b[A-Z][a-z]{5}\b',line).group()
                value=re.search(r'\b(?:'+('|'.join(D.WORDS))+r')\b',line).group()
                parsed[source][entity]=value;flat.append(value)
        pred={'text_parser':parsed[src][name],'latest_record':flat[-1],
              'source_only':next(iter(parsed[src].values())),'entity_only':parsed[archives[-1][0]][name]}
        for key,value in pred.items():tally[key].append(value==r['gold'])
    return {key:float(np.mean(v)) for key,v in tally.items()}

def report():
    result={'models':{},'formal_evidence_promotion':0,'limits':['controlled four-code source/entity binding, not open knowledge or general reasoning',
        'QK and MLP predictors receive same-layer actual internal states; not autonomous text-to-output reconstruction',
        'candidate discovery and confirmation have disjoint artificial entity names; only four paraphrases per panel; source labels remain Orion/Vega',
        'causal edge controls match record role, not intervention norm; matched MLP controls must pass their own calibration',
        'no SAE dictionary, natural multi-step reasoning, unrelated-task capability panel or general AGI mechanism claim']}
    for model in D.MODELS:
        root=D.OUT/model;entry={'splits':{}};materials=D.read(D.OUT/f'{model}-materials.json')['splits']
        if (root/'pilot-gate.json').exists():
            entry['pilot_gate']=D.read(root/'pilot-gate.json')
            if not entry['pilot_gate']['passed']:entry['status']='pilot_behavior_failed; internal replication not run'
        if (root/'pilot-output-diagnostic.json').exists():
            diag=D.read(root/'pilot-output-diagnostic.json');entry['failed_pilot_output_diagnostic']=diag
            lookup={x['id']:x for x in materials['discovery']};categories={}
            for row in diag['rows']:
                original=lookup[row['id']];name=original['names'][original['entity']];text=row['text'];gold=row['gold']
                category='exact' if row['exact_code_only'] else 'case_only' if text.lower()==gold.lower() else 'entity_only' if text==name else 'entity_label_and_period' if text==name+': '+gold+'.' else 'wrong_code_or_other'
                categories.setdefault(category,[]).append(row['id'])
            entry['failed_pilot_posthoc_output_categories']=categories
        for split,rows in materials.items():
            x={'planned_material_rows':len(rows),'families':len({r['family'] for r in rows}),'observed_behavior_rows':0,'baselines':text_baselines(rows)}
            for suffix,key in [('gate','behavior'),('prediction','prediction'),('generation','generation'),('causal','causal')]:
                p=root/f'{split}-{suffix}.json'
                if p.exists():
                    obj=D.read(p);x[key]=obj.get('summary',obj.get('candidates',obj))
                    if suffix=='gate':x['observed_behavior_rows']=obj['n']
                    if suffix=='causal':
                        x['causal_controls']={'skipped_invalid_controls':obj.get('skipped_invalid_controls',[]),
                            'mean_local_head_write_change_norm':{arm:float(np.mean([v for item in obj['local_removed_write_norms'] if item['arm']==arm and item['kind']=='head' for v in item['write_norm']]))
                                for arm in ['correct_edge','wrong_edge','sparse_qk','random_qk']}}
            for suffix in ['conditional-update','conditional-output','parameter-replay']:
                p=root/f'{split}-{suffix}.json'
                if p.exists():
                    obj=D.read(p);x[suffix]=obj.get('summary',obj)
            entry['splits'][split]=x
        p=root/'candidates.json'
        if p.exists():
            c=D.read(p);entry['candidates']={'heads':[{k:v for k,v in x.items() if k not in ['energy','mean_context']} for x in c['heads']],
                'mlps':[{k:v for k,v in x.items() if k not in ['mean_units','mean_output','selected','random','matched']} for x in c['mlps']]}
        seen=set()
        for path in root.glob('*-behavior.json'):seen.update(r['id'] for r in D.read(path)['rows'])
        entry['unique_behavior_rows_including_pilot']=len(seen)
        entry['native_rows']=sum(D.read(path)['n'] for path in root.glob('*-native-manifest.json'))
        result['models'][model]=entry
    p=D.OUT/'qwen14/replication-decision.json'
    if p.exists():result['replication_decision']=D.read(p)
    logs=[__import__('json').loads(line) for line in (D.OUT/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
    process_seconds={}
    for x in logs:
        if x.get('cuda_peak',0)>0:process_seconds[x['pid']]=max(process_seconds.get(x['pid'],0),x['elapsed_seconds'])
    result['resources']={'max_cuda_peak_GiB':max(x.get('cuda_peak',0) for x in logs)/2**30,
        'max_shared_GPU_MiB':max(x.get('shared_gpu_bytes',0) or 0 for x in logs)/2**20,
        'minimum_RAM_available_GiB':min(x['ram_available_bytes'] for x in logs)/2**30,
        'all_progress_rows':len(logs),'GPU_process_logged_wall_minutes':sum(process_seconds.values())/60,
        'output_bytes':sum(p.stat().st_size for p in D.OUT.rglob('*') if p.is_file())}
    D.save('report.json',result)

def plots(model):
    from source_binding_native import unbits
    os.environ['MPLCONFIGDIR']=str(D.ROOT/'tests/glm5_temp/source_binding_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'savefig.dpi':150})
    root=D.OUT/model;manifest=D.read(root/'discovery-native-manifest.json');c=D.read(root/'candidates.json')
    rows=D.read(D.OUT/f'{model}-materials.json')['splits']['discovery'];targets=np.array([r['target_record'] for r in rows])
    mass=np.load(root/'discovery-native/source_mass.npy',mmap_mode='r');correct=mass[np.arange(len(rows)),:,:,targets].mean(0)
    hidden=np.load(root/'discovery-native/hidden.npy',mmap_mode='r');raw=np.zeros((41,hidden.shape[-1]));source=np.zeros_like(raw);std=np.zeros_like(raw)
    start=time.monotonic()
    for b in range(41):
        x=unbits(hidden[:,b,-1]);raw[b]=np.sqrt((x*x).mean(0));std[b]=x.std(0);xf=x.reshape(-1,2,2,2,x.shape[-1]);source[b]=np.sqrt(np.square(xf[:,1]-xf[:,0]).mean((0,1,2)))
        if b%10==0:D.log(model+'/plots',b+1,41,start)
    standardized=np.divide(source,std,out=np.zeros_like(source),where=std>0)
    np.savez(root/'full-coordinate-statistics.npz',hidden_RMS=raw,hidden_std=std,source_paired_RMS=source,standardized_source_paired_RMS=standardized,correct_record_attention=correct)
    linear,la=plt.subplots(2,1,figsize=(15,6),layout='constrained')
    for ax,data,title in [(la[0],raw,'Native hidden RMS: untransformed linear color scale'),(la[1],source,'Source-switch RMS: untransformed linear color scale')]:
        im=ax.imshow(data,origin='lower',aspect='auto',interpolation='nearest',cmap='magma');ax.set_title(title);ax.set_xlabel('Native residual coordinate');ax.set_ylabel('Raw boundary');linear.colorbar(im,ax=ax)
    linear.savefig(root/'natural-linear-full-coordinate.png',dpi=400);plt.close(linear)
    fig,axes=plt.subplots(4,1,figsize=(15,13),layout='constrained')
    for ax,data,title in [(axes[0],np.log1p(raw),'Natural hidden RMS: log(1+RMS) display; all native coordinates'),(axes[1],np.log1p(source),'Paired source-switch RMS: log(1+RMS) display; all native coordinates'),
                          (axes[2],standardized,'Source-switch RMS / discovery coordinate SD (zero SD displayed as0; not covariance whitening)')]:
        im=ax.imshow(data,aspect='auto',origin='lower',interpolation='nearest',cmap='magma');ax.set_title(title);ax.set_ylabel('Raw boundary (0=embedding)');ax.set_xlabel('Native residual coordinate');fig.colorbar(im,ax=ax,shrink=.8)
    im=axes[3].imshow(correct,aspect='auto',origin='lower',vmin=0,vmax=1,cmap='viridis');axes[3].set_title('Attention mass on requested source/entity record; all heads');axes[3].set_xlabel('Native head');axes[3].set_ylabel('Block')
    for x in c['heads']:axes[3].scatter(x['head'],x['block'],facecolors='none',edgecolors='red',s=80,linewidths=1.5)
    fig.colorbar(im,ax=axes[3],shrink=.8);fig.savefig(root/'natural-full-coordinate.png',dpi=400);plt.close(fig)
    panels=[]
    for split in ['confirm','lockbox_frame']:
        p=root/f'{split}-prediction.json'
        if p.exists():panels.append((split,D.read(p)))
    if panels:
        fig,axes=plt.subplots(1,len(panels),figsize=(8*len(panels),5),layout='constrained',sharey=True);axes=np.atleast_1d(axes)
        for ax,(split,data) in zip(axes,panels):
            names=[];selected=[];random=[]
            for x in data['candidates']:
                names.append(x['id']);ishead=x['kind']=='head';selected.append(x['methods']['selected32' if ishead else 'selected256']['improvement']);random.append(x['methods']['random32' if ishead else 'random256']['improvement'])
            loc=np.arange(len(names));ax.bar(loc-.18,selected,.36,label='Selected native subset');ax.bar(loc+.18,random,.36,label='Equal-count random subset');ax.axhline(.1,color='black',ls='--',lw=1)
            ax.set_xticks(loc,names,rotation=35);ax.set_ylabel('MSE improvement vs discovery mean');ax.set_title(split);ax.legend()
        fig.savefig(root/'heldout-component-prediction.png');plt.close(fig)
    panels=[]
    for split in ['confirm','lockbox_frame']:
        p=root/f'{split}-causal.json'
        if p.exists():panels.append((split,D.read(p)))
    if panels:
        fig,axes=plt.subplots(1,len(panels),figsize=(8*len(panels),5),layout='constrained',sharey=True);axes=np.atleast_1d(axes)
        for ax,(split,data) in zip(axes,panels):
            x=data['summary'];names=[k for k in x if k not in ['base','noop']];values=[x[k]['gold_margin_change'] for k in names];ci=np.array([x[k]['margin_change_ci_family'] for k in names])
            ax.bar(np.arange(len(names)),values,yerr=np.stack([np.array(values)-ci[:,0],ci[:,1]-np.array(values)]),capsize=4);ax.axhline(0,color='black',lw=1);ax.set_xticks(np.arange(len(names)),names,rotation=35);ax.set_title(split);ax.set_ylabel('Change in correct-answer margin')
        fig.savefig(root/'native-intervention-effects.png');plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['report','plots']);ap.add_argument('--model',default='qwen14');a=ap.parse_args()
    if a.stage=='report':report()
    else:plots(a.model)
