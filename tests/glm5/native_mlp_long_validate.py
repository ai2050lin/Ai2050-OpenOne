"""Prospective matched natural-query doses, unchanged-weight audit and NI gate."""
import sys,json,time,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_long_materials as P
from native_mlp_long_run import status
from shared.dist_protocol.mechanism_trace import GpuLease,Recorder as BaseRecorder,TraceStore,digest,file_sha,unbits,save_json
from shared.dist_protocol.natural_register_runner import snapshot
from phase2773_w6_loop import paired_interval

class Recorder(BaseRecorder):
    def finish(self,*args,**kwargs):
        d,p=super().finish(*args,**kwargs)
        d['causal_visibility']='Plain natural prefix; query is its last position. No future token, source label or expected answer is supplied to the model.'
        d.pop('trace_id');d['trace_id']=digest(d);save_json(p,d);return d,p

def install(model,group,alpha,position=-1):
    handles=[]
    for l,u in group:
        def hook(_m,args,u=u):
            x=args[0].clone();x[:,position,u]*=1-alpha;return (x,)
        handles.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook))
    return handles

def interval(values):
    values=np.asarray(values);rng=np.random.default_rng(P.SEED)
    means=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return {'mean':float(values.mean()),'ci95':np.quantile(means,[.025,.975]).tolist()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['dose','capability','trace'],required=True);part=parser.parse_args().part
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    selection=P.read(P.OUT/'selection.json');data=P.read(P.OUT/'documents.json');prov=selection['provenance'];raw=np.load(P.OUT/'input-tokens.npy',mmap_mode='r')
    inputs=P.ROOT/'tests/glm5/result/phase2773_w6_v2/capability-inputs.json'
    contract={'kind':'real','run_id':P.RUN,'part':part,'script_sha256':file_sha(__file__),'selection_sha256':file_sha(P.OUT/'selection.json'),
        'units':selection['units'],'random_units':selection['random_units'],'query_operation':'Only the evaluated query position; past positions unchanged',
        'conditions':['base','target_half','target_full','random_full'],'n':256,'primary_effect':'natural actual-next-token top1 loss at alpha1 >= .10 and target-minus-random loss >= .05; paired bootstrap lower95 bounds >0',
        'secondary':'mean full-vocabulary KL and NLL; descriptive, no unique-neuron mechanism claim',
        'capability_input_sha256':file_sha(inputs),'capability_margin':.02,'provenance':prov}
    P.save(P.OUT/f'{part}-contract.json',contract)
    with GpuLease():
        assert snapshot(P.MODEL)==prov['model_revision'];torch.set_num_threads(4);torch.manual_seed(P.SEED);started=time.time()
        tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True);tok.padding_side='left'
        model=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval().requires_grad_(False)
        def forward(ids,**kw):
            with torch.inference_mode():return model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False,**kw)
        if part=='dose':
            rows=[];logits=[];instrumentation=[];pairs=data['pairs']['lockbox'];status('intervention','running',done=0,total=256)
            for i,pair in enumerate(pairs):
                di,q=pair['a'],pair['position'];ids=np.asarray(raw[di,:1024]);truth=int(raw[di,q+1]);by={};captured={};hh=[]
                # Control capture at the exact query, then bitwise compare with an uninstrumented forward.
                for l,u in selection['units']:
                    def cap(_m,args,l=l,u=u):captured[(l,u)]=float(args[0][0,q,u])
                    hh.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(cap))
                o=forward(ids,logits_to_keep=torch.tensor([q],device='cuda'));audit=o.logits[0,-1].detach().clone()
                for h in hh:h.remove()
                maxerr=0.
                for l,u in selection['units']:
                    ix=[r['unit'] for r in data['units'] if r['layer']==l].index(u);a=np.load(P.OUT/f'L{l:02d}-A.npy',mmap_mode='r')
                    maxerr=max(maxerr,abs(captured[(l,u)]-float(unbits(a[di,q,ix]))))
                for name,alpha,group in [('base',0,[]),('target_half',.5,selection['units']),('target_full',1,selection['units']),('random_full',1,selection['random_units'])]:
                    handles=install(model,group,alpha,q)
                    try:o=forward(ids,logits_to_keep=torch.tensor([q],device='cuda'))
                    finally:
                        for h in handles:h.remove()
                    v=o.logits[0,-1];lp=v.float().log_softmax(-1);by[name]=lp.detach().clone()
                    logits.append(v.contiguous().view(torch.uint16).cpu().numpy())
                    row={'pair_index':i,'doc_index':di,'position':q,'condition':name,'expected_token_id':truth,'top1':int(lp.argmax()),'nll':-float(lp[truth])}
                    row['correct']=row['top1']==truth;rows.append(row)
                    if name=='base':instrumentation.append({'pair_index':i,'hook_logits_bitwise_equal':bool(torch.equal(v,audit)),'batch2_to_batch1_selected_A_max_absolute':maxerr})
                for row in rows[-4:]:row['KL_from_base']=float((by['base'].exp()*(by['base']-by[row['condition']])).sum())
                if (i+1)%32==0:status('intervention','running',done=i+1,total=256)
            np.save(P.OUT/'dose-logits.npy',np.stack(logits));by={c:[r for r in rows if r['condition']==c] for c in contract['conditions']};base=by['base']
            effects={c:{'accuracy':float(np.mean([r['correct'] for r in rr])),
                'loss':interval([b['correct']-r['correct'] for b,r in zip(base,rr)]),
                'KL':interval([r['KL_from_base'] for r in rr]),'NLL_shift':interval([r['nll']-b['nll'] for b,r in zip(base,rr)])} for c,rr in by.items()}
            specificity=interval([r['correct']-t['correct'] for r,t in zip(by['random_full'],by['target_full'])]);loss=effects['target_full']['loss']
            result={'kind':'real','run_id':P.RUN,'rows':rows,'effects':effects,'specificity':specificity,
                'supported':bool(loss['mean']>=.1 and loss['ci95'][0]>0 and specificity['mean']>=.05 and specificity['ci95'][0]>0),
                'instrumentation':instrumentation,'seconds':time.time()-started,'provenance':prov}
            P.save(P.OUT/'dose-result.json',result);status('intervention',supported=result['supported'],effects=effects,specificity=specificity)
        elif part=='capability':
            qs=P.read(inputs);ids=[tok.encode(x,add_special_tokens=False)[0] for x in 'ABCDE'];rows={};status('capability','running',done=0,total=2*len(qs))
            for condition in ['base','selected_units']:
                handles=install(model,selection['units'],1.) if condition=='selected_units' else [];records=[]
                try:
                    for start in range(0,len(qs),4):
                        chunk=qs[start:start+4];tokens=[]
                        for q in chunk:
                            text=q['question']+'\n'+'\n'.join(f'{s}. {choice}' for s,choice in zip('ABCDE',q['choices']))+'\n请只回答正确选项字母。'
                            prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False);tokens.append(tok.encode(prompt,add_special_tokens=False))
                        batch={k:v.cuda() for k,v in tok.pad({'input_ids':tokens},padding=True,return_tensors='pt').items()}
                        with torch.inference_mode():o=model(**batch,use_cache=False,logits_to_keep=1)
                        for j,q in enumerate(chunk):
                            scores=o.logits[j,-1,ids[:len(q['choices'])]].float().cpu();pred='ABCDE'[int(scores.argmax())]
                            records.append({'id':q['id'],'prediction':pred,'answer':q['answer'],'correct':pred==q['answer'],'source':q['source'],'gate_eligible':q['gate_eligible'],'choice_logits':scores.tolist(),'input_tokens':len(tokens[j])})
                        if start%400==0:status('capability','running',condition=condition,done=start+len(chunk),total=len(qs))
                finally:
                    for h in handles:h.remove()
                rows[condition]=records;P.save(P.OUT/f'capability-{condition}.json',records)
            result=paired_interval([x['correct'] for x in rows['base']],[x['correct'] for x in rows['selected_units']],np.asarray([q['gate_eligible'] for q in qs]))
            result.update(kind='real',run_id=P.RUN,scope='Repeated old keyed pool, new16unit current-query operation; 140 reference-only excluded',seconds=time.time()-started,provenance=prov)
            P.save(P.OUT/'capability-result.json',result);status('capability',result=result)
        else:
            # Text-independent fixed representative: top confirm-selected unit and first four lockbox pairs, both sides.
            l,u=selection['units'][0];records=[];store=TraceStore()
            for i,pair in enumerate(data['pairs']['lockbox'][:4]):
                for side in ['a','b']:
                    di=pair[side];doc=data['docs'][di];q=pair['position'];ids=raw[di,:q+1].tolist();text=tok.decode(ids)
                    item={'id':f'{P.RUN}:pair{i}:{side}','entity':doc.get('title') or doc['doc_id'],'scope':'自然长文 · 同token/同位置配对','split':'lockbox','token_ids':ids,'text':text,'prompt':text}
                    with Recorder(model,[q],P.OUT/'native'/f'pair{i}-{side}',deep=True) as rec:o=forward(ids,logits_to_keep=1,output_hidden_states=True)
                    d,p=rec.finish(o,item,tok,prov['model_revision'],P.RUN,prov);store.register(p)
                    records.append({'trace_id':d['trace_id'],'pair_index':i,'side':side,'doc_index':di,'position':q,'layer':l,'unit':u,'n_tokens':len(ids),'max_attention_relative_l2':max(v['attention_relative_l2'] for v in d['reconstruction'].values())})
                    del o;status('validation','running',done=len(records),total=8)
            assert max(r['max_attention_relative_l2'] for r in records)<.05
            P.save(P.OUT/'trace-examples.json',{'kind':'real','run_id':P.RUN,'rows':records,'provenance':prov});status('validation',traces=8)
        assert snapshot(P.MODEL)==prov['model_revision']

if __name__=='__main__':main()
