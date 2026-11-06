"""Fresh independently frozen fact x question polarity disambiguation, no refitting."""
import sys,time,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
import native_mlp_long_materials as OLD
from native_mlp_function_run_v4 import Recorder,status
from shared.dist_protocol.mechanism_trace import GpuLease,TraceStore,digest,file_sha,unbits
from shared.dist_protocol.natural_register_runner import snapshot
from tools.power_analysis import check_phase

def prepare():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True);raw=np.load(OLD.OUT/'input-tokens.npy',mmap_mode='r');rng=np.random.default_rng(P.SEED);indices=rng.permutation(2048)[544:672];items=[]
    prefix=tok.encode('<|im_start|>user\n',add_special_tokens=False);end=tok.encode('\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False)
    for k,di in enumerate(indices):
        name='navilo'+str(k);key='K' if (k//4)%2==0 else 'M';cue=k%2;frame=(k//2)%2
        pos=[f'All containers in set {name}_amber have a lid, with no exceptions.',f'No container in set {name}_amber is without a lid.'][cue]
        neg=[f'The set {name}_birch is named "all", but at least one container has no lid.',f'At least one container in set {name}_birch lacks a lid.'][cue]
        records={'K':pos if key=='K' else neg,'M':pos if key=='M' else neg};common='\n\nUse the following invented definitions only. Each set contains at least two containers.\nEntry K: '+records['K']+'\nEntry M: '+records['M']+'\n'
        variants={}
        for fact in [0,1]:
            selected=key if fact else ('M' if key=='K' else 'K')
            for question in [0,1]:
                # Exactly equal token length, constant background andfinaltoken across all4variants.
                query=f'Does {"each" if question else "some"} member of Entry {selected}\'s set {"have" if question else "lack"} a lid?'
                text=common+query+' Reply only Yes or No.\nAnswer:';variants[(fact,question)]=tok.encode(text,add_special_tokens=False)
        assert len(set(len(x) for x in variants.values()))==1
        bg=1024-len(prefix)-len(next(iter(variants.values())));assert bg>=512
        for (fact,question),tail in variants.items():
            ids=prefix+raw[di,:bg].tolist()+tail;expected='Yes' if fact==question else 'No'
            items.append({'id':f'polarity:{k}:{fact}:{question}','pair_id':f'polarity:{k}:{question}','quad_id':f'polarity:{k}','fact':fact,'question_polarity':question,'cue_form':cue,'frame':frame,'split':'fresh_polarity','background_doc_index':int(di),
                'expected':expected,'contrast':'No' if expected=='Yes' else 'Yes','token_ids':ids+end,'readout_positions':[1023,1023+len(end)],'text':tok.decode(ids),'prompt':tok.decode(ids+end),'entity':name,'scope':'事实属性×提问极性正交对照'})
    selection=P.read(P.OUT/'coordinate-selection.json');contract={'schema':'research_polarity_prereg.v1','kind':'real','run_id':P.RUN,'model_snapshot':P.read(P.OUT/'corpus.json')['model_snapshot'],
        'n_independent_background_quads':128,'n_inputs':512,'hypotheses':selection['selected'],'source_selection_sha256':file_sha(P.OUT/'coordinate-selection.json'),
        'primary':'Previous frozen candidate sign, no refit. Fact-order accuracy separately underpositive andnegative question. Answer-order reverses fact sign undernegative question. Bothstages tested;6units x2stages alpha .05/12.',
        'criteria':{'min_accuracy':.75,'alpha':.05/12,'min_unequal_fraction':.8,'min_behavior_cell_accuracy':.9,'min_behavior_cell_n':32,'power':check_phase(.30,128,p_d=.5,alpha=.05/12)},
        'counterexample':'same fact changes desired Yes/No answer when question changes; facts, chosen entity, background, token length andfinaltoken otherwise controlled; question Each/Some andhave/lack are equal-length token substitutions',
        'scope':'one controlled universal-set task, fresh previously unused background documents andnonce stems; not all semantic roles, not proof of a universal answer neuron',
        'script_sha256':file_sha(__file__),'declared_before_new_outputs':True}
    assert contract['criteria']['power']['pass'];P.save(P.PUBLIC/'polarity-prereg.research.json',contract);P.save(P.OUT/'polarity-corpus.json',{'kind':'real','run_id':P.RUN,'items':items,'model_snapshot':contract['model_snapshot']})
    print(json.dumps({'polarity_inputs':len(items),'power':contract['criteria']['power']}),flush=True)

def metric(x,sign):
    x=np.asarray(x)*sign;n=int(np.count_nonzero(x));wins=int(np.sum(x>0));acc=float(np.mean(x>0)+.5*np.mean(x==0));p=float(binomtest(wins,n,.5,alternative='greater').pvalue) if n else 1.
    return {'accuracy':acc,'p':p,'unequal_fraction':n/len(x),'passed':bool(acc>=.75 and p<.05/12 and n/len(x)>=.8)}
def run():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    c=P.read(P.OUT/'polarity-corpus.json');units=P.read(P.OUT/'coordinate-selection.json')['selected'];prov=P.read(P.OUT/'observations.json')['provenance'];lease=GpuLease()
    for attempt in range(13):
        try:lease.acquire();break
        except RuntimeError:
            if attempt==12:raise
            time.sleep(5)
    try:
        assert snapshot(P.MODEL)==c['model_snapshot'];torch.set_num_threads(4);tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True)
        m=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False);rows=[];values=[];store=TraceStore()
        choices={x:[tok.encode(s,add_special_tokens=False)[0] for s in [x,' '+x] if len(tok.encode(s,add_special_tokens=False))==1] for x in ['Yes','No']}
        for i,item in enumerate(c['items']):
            with Recorder(m,item['readout_positions'],P.OUT/'polarity-native'/digest(item['id'])[:20],deep=False) as rec:
                with torch.inference_mode():o=m(input_ids=torch.tensor([item['token_ids']],device='cuda'),use_cache=False,output_hidden_states=True,logits_to_keep=1)
            d,path=rec.finish(o,item,tok,c['model_snapshot'],P.RUN,prov);store.register(path);lp=o.logits[0,-1].float().log_softmax(-1);gap=float(lp[choices[item['expected']]].max()-lp[choices[item['contrast']]].max());v=[]
            for u in units:
                z=store.block(d['trace_id'],u['layer']);v.append(unbits(z['activation'][:,u['unit']]))
            values.append(np.stack(v,axis=-1));rows.append({k:item[k] for k in ['id','quad_id','fact','question_polarity','cue_form','frame','expected','contrast']}|{'correct':gap>0,'choice_gap':gap,'trace_id':d['trace_id']})
            if (i+1)%32==0:status('polarity','running',done=i+1,total=512)
        assert snapshot(P.MODEL)==c['model_snapshot']
    finally:lease.release()
    v=np.stack(values);np.save(P.OUT/'polarity-A.npy',v);by={r['id']:i for i,r in enumerate(rows)};findings=[];cells=[]
    for cue in [0,1]:
        for frame in [0,1]:
            for fact in [0,1]:
                for q in [0,1]:
                    rr=[r for r in rows if (r['cue_form'],r['frame'],r['fact'],r['question_polarity'])==(cue,frame,fact,q)];assert len(rr)==32
                    cells.append({'cue':cue,'frame':frame,'fact':fact,'question':q,'n':32,'accuracy':float(np.mean([r['correct'] for r in rr]))})
    gate=all(c['accuracy']>=.9 for c in cells)
    for j,u in enumerate(units):
        for stage in [0,1]:
            results={}
            for q in [0,1]:
                diff=np.asarray([v[by[f'polarity:{k}:1:{q}'],stage,j]-v[by[f'polarity:{k}:0:{q}'],stage,j] for k in range(128)])
                results[str(q)]={'fact_order':metric(diff,u['sign']),'answer_order':metric(diff,u['sign']*(1 if q else -1))}
            fact_supported=gate and all(results[str(q)]['fact_order']['passed'] for q in [0,1]);answer_supported=gate and all(results[str(q)]['answer_order']['passed'] for q in [0,1])
            findings.append({'family':u['family'],'layer':u['layer'],'unit':u['unit'],'stage':stage,'results':results,'fact_supported':fact_supported,'answer_supported':answer_supported,
                'interpretation':'answer-aligned_in_this_scope' if answer_supported else 'fact-aligned_in_this_scope' if fact_supported else 'not_disambiguated'})
    result={'kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'rows':rows,'cells':cells,'behavior_gate':gate,'findings':findings,'provenance':prov}
    P.save(P.OUT/'polarity-result.json',result);status('polarity','done' if gate else 'behavior_failed',behavior_gate=gate,findings=findings)
if __name__=='__main__':
    prepare()
    while not (P.OUT/'audit-traces.json').exists():time.sleep(5)
    run()
