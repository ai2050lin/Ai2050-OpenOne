"""Compact, resumable BF16 native bank for every MLP unit; no uncaptured zero rows."""
import sys,json,time,sqlite3,subprocess,shutil,hashlib,os
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_atlas_code_materials_v2 as P
from mlp_atlas_code_pilot_v2 import status
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha,digest
from shared.dist_protocol.natural_register_runner import snapshot

def atomic(path,d):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.swap');tmp.write_text(json.dumps(d,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)

def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    c=P.read(P.OUT/'corpus.json');pilot=P.read(P.OUT/'pilot-result.json');assert pilot['passed'];items=c['items'];n=len(items);assert n==9984
    prov=pilot['provenance'];folder=P.OUT/'bank';folder.mkdir(parents=True,exist_ok=True);cfg=P.read(P.MODEL/'config.json');D=cfg['hidden_size'];F=cfg['intermediate_size'];L=cfg['num_hidden_layers'];H=cfg['num_attention_heads']*cfg['head_dim'];V=cfg['vocab_size']
    assert (D,F,L,H)==(2560,9728,36,4096)
    shapes={f'B{b:02d}-{k}.npy':(n,2,d) for b in range(L) for k,d in [('G',F),('V',F),('A',F),('input',D),('output',D),('attention_output',D),('head_z',H)]}
    shapes.update({'hidden.npy':(n,2,L+1,D),'logits.npy':(n,V)});predicted_bytes=sum(np.prod(s,dtype=np.int64)*2+256 for s in shapes.values())
    contract={'schema':'mlp_native_bank_plan.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'provenance':prov,'n_items':n,'batch':2,'readout_roles':['question_end','assistant_boundary'],'full_native_GVA_units':L*F,
        'files':{name:list(s) for name,s in shapes.items()},'estimated_native_bank_bytes':int(predicted_bytes),'budget_GiB':65,'free_GiB_before_allocation':round(shutil.disk_usage(P.OUT).free/2**30,3),
        'absence':'No per-token QKV or attention rows in the compact bank. They must never be filled with zeros or substituted from another query. Separate deep audit traces may be added.',
        'negative_control':'all37hidden states atbefore-selector andhs[0] atquestion-end are saved as exact full-vector hashes; raw37hidden retained atbothdeclared analysis readouts',
        'source_sha256':file_sha(__file__),'pilot_sha256':file_sha(P.OUT/'pilot-result.json'),'corpus_sha256':digest(c),'prereg_sha256':file_sha(P.PUBLIC/'prereg.research.json'),'resume_chunk':32}
    freeze=P.OUT/'bank-execution-freeze.json'
    if freeze.exists():
        old=P.read(freeze)
        for key in ['source_sha256','pilot_sha256','corpus_sha256','prereg_sha256','model_snapshot']:assert old[key]==contract[key],('resume source drift',key)
        contract=old
    else:
        assert predicted_bytes<62*2**30 and shutil.disk_usage(P.OUT).free>85*2**30
        P.save(freeze,contract);P.save(P.PUBLIC/'bank-plan.research.json',contract)
    db=sqlite3.connect(folder/'samples.sqlite3');db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE IF NOT EXISTS samples(idx INTEGER PRIMARY KEY,id TEXT UNIQUE,body TEXT NOT NULL)');db.commit()
    completed=db.execute('SELECT COUNT(*) FROM samples').fetchone()[0];assert completed==0 or db.execute('SELECT MAX(idx)+1 FROM samples').fetchone()[0]==completed
    manifest={'schema':'mlp_native_bank.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'provenance':prov,'n_items':n,'completed':completed,'state':'collecting','n_layers':L,'d_ff':F,'d_model':D,'n_heads':cfg['num_attention_heads'],'n_kv_heads':cfg['num_key_value_heads'],'head_dim':cfg['head_dim'],'readout_roles':contract['readout_roles'],
        'file_shapes':contract['files'],'boundary_names':{'G':'gate_proj.output','V':'up_proj.output','A':'down_proj.input','input':'mlp.input','output':'mlp.output','attention_output':'self_attn.o_proj.output','head_z':'self_attn.o_proj.input'},'capture_contract_sha256':file_sha(freeze),'full_vocabulary_logits':True,'attention_rows_captured':False,'all37hidden':True,'native_dtype':'bfloat16_uint16_bits','last_hidden_after_final_norm':True}
    maps={name:np.lib.format.open_memmap(folder/name,mode='r+' if (folder/name).exists() else 'w+',dtype=np.uint16,shape=shape) for name,shape in shapes.items()};atomic(folder/'manifest.json',manifest)
    if completed==n and P.read(folder/'manifest.json').get('state')=='complete':return
    start=time.time();context={};pending=[];hooks=[];formula_total=0;formula_equal=0
    with GpuLease():
        assert snapshot(P.MODEL)==c['model_snapshot'];torch.set_num_threads(4);torch.manual_seed(P.SEED);tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
        choices={s:list({tok.encode(t,add_special_tokens=False)[0] for t in [s,' '+s] if len(tok.encode(t,add_special_tokens=False))==1}) for labels in c['codes'].values() for s in labels}
        def bits(x):return x.detach().contiguous().view(torch.uint16).cpu().numpy()
        def rows(x):return torch.stack([x[j,r['readout_positions'][2:]] for j,r in enumerate(context['items'])])
        def savefield(b,k,x):maps[f'B{b:02d}-{k}.npy'][context['start']:context['stop']]=bits(rows(x))
        for b,layer in enumerate(model.model.layers):
            def gate(_m,args,out,b=b):context[(b,'G')]=rows(out);savefield(b,'G',out)
            def value(_m,args,out,b=b):context[(b,'V')]=rows(out);savefield(b,'V',out)
            def activation(_m,args,b=b):
                nonlocal formula_total,formula_equal
                a=rows(args[0]);expected=torch.nn.functional.silu(context.pop((b,'G')))*context.pop((b,'V'));equal=torch.equal(a,expected);assert equal,('native formula differs',b)
                formula_equal+=a.numel();formula_total+=a.numel();savefield(b,'A',args[0])
            hooks.extend([layer.mlp.gate_proj.register_forward_hook(gate),layer.mlp.up_proj.register_forward_hook(value),layer.mlp.down_proj.register_forward_pre_hook(activation),
                layer.mlp.register_forward_pre_hook(lambda _m,args,b=b:savefield(b,'input',args[0])),layer.mlp.register_forward_hook(lambda _m,args,out,b=b:savefield(b,'output',out)),
                layer.self_attn.o_proj.register_forward_pre_hook(lambda _m,args,b=b:savefield(b,'head_z',args[0])),layer.self_attn.o_proj.register_forward_hook(lambda _m,args,out,b=b:savefield(b,'attention_output',out))])
        status('collection','running',done=completed,total=n,estimated_GiB=predicted_bytes/2**30)
        def commit():
            nonlocal completed,pending
            for x in maps.values():x.flush()
            with db:
                for index,row in pending:db.execute('INSERT INTO samples VALUES(?,?,?)',(index,row['id'],json.dumps(row,ensure_ascii=False,separators=(',',':'))))
            completed+=len(pending);pending=[];manifest['completed']=completed;manifest['seconds_this_session']=time.time()-start;atomic(folder/'manifest.json',manifest)
            status('collection','running',done=completed,total=n,seconds=round(time.time()-start,1),formula_exact=True)
        for i in range(completed,n,2):
            chunk=items[i:i+2];context.update(start=i,stop=i+len(chunk),items=chunk)
            with torch.inference_mode():o=model(input_ids=torch.tensor([r['token_ids'] for r in chunk],device='cuda'),use_cache=False,output_hidden_states=True,logits_to_keep=1)
            native_hidden=np.stack([bits(rows(h)) for h in o.hidden_states],axis=2);maps['hidden.npy'][i:i+len(chunk)]=native_hidden;maps['logits.npy'][i:i+len(chunk)]=bits(o.logits[:,-1]);lp=o.logits[:,-1].float().log_softmax(-1)
            for j,r in enumerate(chunk):
                before=np.stack([bits(h[j,r['readout_positions'][0]]) for h in o.hidden_states]);gap=float(lp[j,choices[r['expected']]].max()-lp[j,choices[r['contrast']]].max());top=int(lp[j].argmax());probs,inds=lp[j].exp().topk(12)
                row={**r,'kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'bank_index':i+j,'choice_gap':gap,'correct':gap>0,'top1_id':top,'top1_text':tok.decode([top]),'first_top1':top in choices[r['expected']],
                    'before_selector_hidden_sha256':hashlib.sha256(before.tobytes()).hexdigest(),'question_embedding_sha256':hashlib.sha256(native_hidden[j,0,0].tobytes()).hexdigest(),'output_top':[{'token_id':int(t),'text':tok.decode([int(t)]),'probability':float(p)} for t,p in zip(inds,probs)]}
                pending.append((i+j,row))
            del o,lp,native_hidden
            if len(pending)>=32 or i+len(chunk)==n:commit()
        for hook in hooks:hook.remove()
        assert snapshot(P.MODEL)==c['model_snapshot'];manifest.update(state='complete',formula_exact_fraction=formula_equal/max(1,formula_total),peak_cuda_bytes=torch.cuda.max_memory_allocated(),snapshot_unchanged=True)
    manifest['files']=[{'path':name,'shape':list(shapes[name]),'bytes':(folder/name).stat().st_size,'sha256':file_sha(folder/name)} for name in sorted(shapes)]
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();manifest['sample_index_sha256']=file_sha(folder/'samples.sqlite3');atomic(folder/'manifest.json',manifest)
    status('collection','done',done=n,total=n,bank_bytes=sum(f['bytes'] for f in manifest['files']),formula_exact_fraction=manifest['formula_exact_fraction']);print(json.dumps({'completed':n,'bank_bytes':sum(f['bytes'] for f in manifest['files'])}),flush=True)
if __name__=='__main__':main()
