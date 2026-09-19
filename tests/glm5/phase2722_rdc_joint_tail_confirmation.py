"""Frozen rare-event forecast on new natural material; all coordinates, bounded retention."""
from collections import Counter,defaultdict
import gc,sys
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score
from rdc_joint_common import *
from rdc_joint_capture import ledger,array_identity
from phase2722_rdc_joint_amplification import token_kind

OUT=BASE/'extension/tail_confirmation'


class ThreeLayers:
    def __init__(self,model):
        self.data={};self.handles=[]
        for layer in (12,23,36):
            def hook(m,a,o,layer=layer):
                v=o[0] if isinstance(o,tuple) else o
                self.data['h'+str(layer)]=bits(v[0])
            self.handles.append(model.model.layers[layer-1].register_forward_hook(hook))
    def close(self):
        for handle in self.handles:handle.remove()


def main():
    import torch,psutil
    from phase2662_symmetric_mapping_contract import load_native
    if (OUT/'result.json').exists():return
    start=time.monotonic();guard(40*1024**2);protocol=read(OUT/'protocol.json')
    assert sha(OUT/'material.json.gz')==protocol['material_sha']
    for path,digest in protocol['frozen_files'].items():assert sha(BASE/path)==digest
    material=json.loads(gzip.decompress((OUT/'material.json.gz').read_bytes()))
    with np.load(BASE/'extension/event_forecast.npz') as z:fit={k:z[k] for k in z.files}
    threshold=read(BASE/'extension/event_threshold.json')
    old=json.loads(gzip.decompress((BASE/'extension/all_token_amplitudes.json.gz').read_bytes()))
    training=[r for r in old if r['split']=='train' and r['position']>0]
    base=np.mean([r['event'] for r in training]);idtable=defaultdict(lambda:[0,0]);postable=defaultdict(lambda:[0,0])
    for r in training:
        for table,key in ((idtable,r['token_id']),(postable,(r['language'],min(r['position']//8,7)))):
            table[key][0]+=1;table[key][1]+=int(r['event'])
    def control(table,key):
        n,k=table.get(key,(0,0));return (k+10*base)/(n+10)
    del old,training
    keeps={r['sample_id'] for key in {r['source_key'] for r in material} for r in [x for x in material if x['source_key']==key][:2]}
    moments=defaultdict(lambda:np.zeros((2,3,2560),float));counts=Counter();records=[];events=[];identities=[];sourcecounts=Counter()
    model,tok=load_native('qwen4');device=model.get_input_embeddings().weight.device;observer=ThreeLayers(model)
    assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    runtime={'timestamp':stamp(),'source':snapshot(Path(__file__)),'torch':torch.__version__,'dtype':str(model.dtype),'quantized':False,
        'model_source_sha':sha(Path(sys.modules[model.model.__class__.__module__].__file__)),'tokenizer_sha':sha(ROOT/'models/hf/qwen3-4b/tokenizer.json'),
        'config':model.config.to_dict(),'execution':'CUDA BF16 eager, unpadded untruncated natural batch1, no chat template, use_cache=False; all native token H12/H23/H36.',
        'retention':'First2 per corpus full fields; EVERY threshold event3layer row; all-source per-array identities and full-coordinate stratified moments. Nonfixture raw buffers released only AFTER complete event/moment/hash analysis; replay via frozen materials and exact config. No old field cleanup.'}
    save(OUT/'runtime.json',runtime)
    try:
      with torch.inference_mode():
        for i,r in enumerate(material):
            assert tok(r['text'],add_special_tokens=False)['input_ids']==r['prompt_ids']
            ids=torch.tensor([r['prompt_ids']],device=device);observer.data={}
            result=model.model(input_ids=ids,use_cache=False);packet=observer.data
            h=np.stack([unbits(packet[k]).astype(float) for k in ('h12','h23','h36')],1)
            assert np.isfinite(h).all();energy=np.mean(h*h,-1);nn=np.arange(len(h))>0
            event=nn&(energy[:,1]>threshold['H23_energy_threshold'])&(energy[:,1]/np.maximum(energy[:,0],1e-20)>=threshold['energy_ratio_min'])
            standardized=((h[:,0].astype(np.float32)-fit['mean'])/fit['standard_deviation']).astype(float)
            p=expit(standardized@fit['coefficient']+fit['intercept'][0])
            identity={k:array_identity(v) for k,v in packet.items()}
            identities.append({'sample_id':r['sample_id'],'arrays':identity,'full_fixture_retained':r['sample_id'] in keeps,'event_positions':np.flatnonzero(event).tolist()})
            if r['sample_id'] in keeps:
                npz(OUT/'fields'/f'{r["sample_id"]}.npz',**packet)
                with np.load(OUT/'fields'/f'{r["sample_id"]}.npz') as z:
                    assert all(array_identity(z[k])==identity[k] for k in z.files)
            if event.any():
                npz(OUT/'events'/f'{r["sample_id"]}.npz',**{k:v[event] for k,v in packet.items()},positions=np.flatnonzero(event))
            for label,mask in [('all_noninitial',nn),('event',event),('non_event',nn&~event),('first',~nn)]:
                for key in ('pooled_'+label,r['source_key']+'_'+label):
                    value=h[mask];moments[key][0]+=value.sum(0);moments[key][1]+=(value*value).sum(0);counts[key]+=len(value)
            for position in range(len(h)):
                record={'sample_id':r['sample_id'],'source_group':r['source_group'],'source_key':r['source_key'],'language':r['language'],
                    'position':position,'token_id':r['prompt_ids'][position],'token':r['tokens'][position],'token_kind':token_kind(r['tokens'][position]),
                    'energy_H12_H23_H36':energy[position].tolist(),'event':bool(event[position]),
                    'prediction':{'full_H12_linear_logistic':float(p[position]),'training_constant':float(base),
                        'token_ID_shrink10':control(idtable,r['prompt_ids'][position]),'language_position_shrink10':control(postable,(r['language'],min(position//8,7)))}}
                records.append(record)
                if event[position]:events.append(record)
            sourcecounts[r['source_key']]+=1
            del ids,result,h,packet,energy,p,standardized
            observer.data={}
            if (i+1)%128==0:
                assert psutil.virtual_memory().available>read(BASE/'resource_allocation.json')['host_runtime_floor_bytes']
                assert time.monotonic()-start<read(BASE/'resource_allocation.json')['per_process_max_seconds']
                guard();print('TAIL_CONFIRMATION',i+1,len(material),'events',len(events),flush=True)
    finally:
        observer.close();del observer,model;gc.collect();torch.cuda.empty_cache()
    compressed_json(OUT/'array_identities.json.gz',identities);compressed_json(OUT/'all_token_results.json.gz',records);compressed_json(OUT/'event_tokens.json.gz',events)
    npz(OUT/'full_coordinate_moments.npz',**{key:(v/max(counts[key],1)).astype(np.float32) for key,v in moments.items()})
    evaluation={}
    for scope in ('pooled',*sorted(sourcecounts)):
        rr=[r for r in records if r['position']>0 and (scope=='pooled' or r['source_key']==scope)]
        yy=np.array([r['event'] for r in rr],int);groups=[r['source_group'] for r in rr];energy=np.array([r['energy_H12_H23_H36'] for r in rr]);reports={}
        for name in rr[0]['prediction']:
            pp=np.array([r['prediction'][name] for r in rr]);loss=-yy*np.log(np.maximum(pp,1e-15))-(1-yy)*np.log(np.maximum(1-pp,1e-15));decision=pp>=.5
            reports[name]={'average_precision':float(average_precision_score(yy,pp)) if yy.sum() else None,'ROC_AUC':float(roc_auc_score(yy,pp)) if 0<yy.sum()<len(yy) else None,
                'logloss':float(loss.mean()),'Brier':float(np.mean((pp-yy)**2)),'group_logloss':paired_summary(loss,groups),
                'threshold0.5':{'TP':int((decision&(yy==1)).sum()),'FP':int((decision&(yy==0)).sum()),'FN':int((~decision&(yy==1)).sum())}}
        eventloss=-yy*np.log(np.maximum([r['prediction']['full_H12_linear_logistic'] for r in rr],1e-15))-(1-yy)*np.log(np.maximum(1-np.array([r['prediction']['full_H12_linear_logistic'] for r in rr]),1e-15))
        constantloss=-yy*np.log(base)-(1-yy)*np.log(1-base)
        evaluation[scope]={'sources':len({r['sample_id'] for r in rr}),'noninitial_tokens':len(rr),'events':int(yy.sum()),'groups_with_events':len({r['source_group'] for r in rr if r['event']}),
            'event_fraction':float(yy.mean()),'event_share_H23_energy':float(energy[yy==1,1].sum()/energy[:,1].sum()),'event_token_kind':dict(Counter(r['token_kind'] for r in rr if r['event'])),
            'reports':reports,'paired_group_logloss_gain_vs_constant':paired_summary(constantloss-eventloss,groups)}
    result={'timestamp':stamp(),'passed_collection_checks':True,'sources':len(material),'tokens':len(records),'event_count':len(events),'source_counts':dict(sourcecounts),
        'evaluation':evaluation,'moment_counts':dict(counts),'full_fields_retained':sorted(keeps),'event_sources_retained':len({r['sample_id'] for r in events}),
        'frozen_files_unchanged':all(sha(BASE/p)==v for p,v in protocol['frozen_files'].items()),
        'limits':'New outcome-blind numerical-event confirmation, not semantics or native causal closure. Original fit has7 positive training tokens/1 validation positive. Corpus/length coverage restricted; PUD translations and unknown articles/GSD article clustering limit independence. All primary noninitial tokens retained in metric denominator. No retraining or threshold adjustment.'}
    save(OUT/'result.json',result);ledger('joint_frozen_tail_confirmation',time.monotonic()-start,sources=len(material),tokens=len(records));guard()
    print('TAIL_CONFIRMATION_COMPLETE',json.dumps(result,ensure_ascii=True),flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
