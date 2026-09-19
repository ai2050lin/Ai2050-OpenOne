"""Repair only the random control affected by unfinished UTF-8 character text, preserving frozen primary rules."""
from tokenizers import Tokenizer
from rdc_prefix_estimators import *
OUT=CAMPAIGN/'causal_hash_control'


def safe_data(run):
    path=CAMPAIGN/'shared_rules'/run
    rows=read(path/'rows.json')
    with np.load(path/'features.npz') as z:data={k:z[k] for k in z.files}
    tok=Tokenizer.from_file(str(ROOT/'models/hf/qwen3-4b/tokenizer.json'));changes=[]
    for i,r in enumerate(rows):
        raw=read(CAMPAIGN/run/f'rows/{r["sample_id"]}.json');ids=raw['prompt_ids'];p=r['position']
        visible=tok.decode(ids[:p+1],skip_special_tokens=False)
        graph=prefix_graph(visible,p,r['language']);assert np.array_equal(descriptor(graph),data['graph'][i])
        ng=prefix_graph(tok.decode(ids[:p+2],skip_special_tokens=False),p+1,r['language']);assert np.array_equal(descriptor(ng),data['next_graph'][i])
        seed=int(hashlib.sha256(visible.encode()).hexdigest()[:16],16)
        noise=np.random.default_rng(seed).standard_normal(len(data['hash_graph'][i])).astype(np.float32)
        if not np.array_equal(noise,data['hash_graph'][i]):changes.append(i)
        data['hash_graph'][i]=noise;r['prefix']=visible
    return rows,data,changes


def main():
    audit=read(CAMPAIGN/'causal_prefix_audit.json');assert audit['descriptor_changed_positions']==0
    frozen=read(CAMPAIGN/'shared_rules/frozen_models.json')
    for rel,digest in frozen['files'].items():assert sha(CAMPAIGN/'shared_rules'/rel)==digest
    rows,data,changes=safe_data('qwen4');newrows,new,nchanges=safe_data('qwen4_confirmation');tr,va,te=splits(rows)
    reports=[]
    for temporal in (False,True):
        bank=KernelBank(data,tr,temporal=temporal);newbank=KernelBank(new,[],bank.serial_scales(),temporal=temporal)
        gram=bank.gram('hash_interaction',np.arange(len(rows)),np.arange(len(rows)))
        cross=newbank.gram('hash_interaction',np.arange(len(newrows)),tr,bank)
        labels=('next_h12','next_h36') if temporal else ('h23','h24','h36');y=np.concatenate([data[k] for k in labels],1)
        for df in ([None] if temporal else [None,128]):
            name=('temporal_' if temporal else '')+'hash_interaction'+('_df128' if df else '')
            mp=OUT/f'models/{name}.npz'
            pred,val,meta=fit(gram,tr,va,te,y,[(i*2560,(i+1)*2560) for i in range(len(labels))],mp,df)
            with np.load(mp) as z:confirmation=(cross@z['alpha'])*z['target_scales']+z['means']
            for split,pp,rr,target,train in [('test',pred,[rows[i] for i in te],data[labels[-1]][te],data[labels[-1]][tr]),
              ('confirmation',confirmation,newrows,new[labels[-1]],data[labels[-1]][tr])]:
                report,arr=errors(target,pp[:,-2560:],train,rr)
                reports.append({'model':name,'evaluation':split,**meta,'final_target':labels[-1],**report})
                npz(OUT/f'predictions/{split}_{name}.npz',prediction=pp.astype(np.float32),**arr)
                print('CAUSAL_HASH',name,split,report['mse'],flush=True)
    save(OUT/'result.json',{'timestamp':stamp(),'phase':2713,'main_changed_anchor_indices':changes,'confirmation_changed_anchor_indices':nchanges,
      'primary_numeric_descriptors_all_equal':True,'primary_frozen_models_unchanged':True,'reports':reports,
      'reason':'Repair a token-byte prefix bug in the deterministic random-prefix-hash nuisance control. The actual primary graph descriptors remain bitwise identical at all3840 positions.',
      'evidence_status':'Software-corrected control rerun after earlier confirmation scores were visible; not a new independent confirmation of a newly selected model. Primary frozen non-hash rules and their original confirmation remain unchanged.',
      'old_versions':'Original hashes, models and scores retained. New figures/client comparisons should select this corrected control and show revision label.'})
    guard();print('CAUSAL_HASH_REPAIR_COMPLETE',len(changes),len(nchanges),flush=True)


if __name__=='__main__':main()
