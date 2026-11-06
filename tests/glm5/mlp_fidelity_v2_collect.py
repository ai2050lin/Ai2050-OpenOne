"""Sequential real Qwen3-4B: discovery pilot, fresh natural fields, controlled native bank."""
import sys,json,time,sqlite3,os,shutil,hashlib,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_plan as P
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha,digest
from shared.dist_protocol.natural_register_runner import snapshot
from mlp_atlas_code_collect import atomic

def main(part):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    plan=P.read(P.OUT/'plan.json');corpus=P.read(P.OUT/'controlled-corpus.json');prov=plan['provenance'];folder=P.OUT/('controlled-bank' if part=='controlled' else 'natural-bank');batch=4
    if part=='controlled':assert P.read(P.OUT/'pilot-result.json')['passed']
    if part=='pilot':items=[r for r in corpus['items'] if r['background_index']<32];assert len(items)==512
    elif part=='controlled':items=corpus['items']
    else:items=P.read(P.OUT/'fresh-documents.json')['natural']
    if part!='pilot' and (folder/'manifest.json').exists() and P.read(folder/'manifest.json')['state']=='complete':print('Completed immutable collection exists; skipped.');return
    positions=np.sort(np.random.default_rng(P.SEED+55).choice(np.arange(512,1024),8,replace=False));shapes={};maps={};hooks=[];ctx={};pending=[];db=None
    if part=='natural':shapes={f'B{b:02d}-{k}.npy':(512,8,2560) for b in range(36) for k in ['input','output']}
    elif part=='controlled':
        shapes={f'B{b:02d}-{k}.npy':(8192,2,d) for b in [8,24,35] for k,d in [('G',9728),('V',9728),('A',9728),('input',2560),('output',2560),('head_z',4096),('attention_output',2560)]};shapes['hidden.npy']=(8192,2,37,2560)
    freeze={'kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':prov,'part':part,'source_sha256':file_sha(__file__),'corpus_sha256':digest(corpus),'plan_sha256':file_sha(P.OUT/'plan.json'),'batch':batch,'shapes':{k:list(v) for k,v in shapes.items()}}
    P.save(P.OUT/(part+'-execution-freeze.json'),freeze);completed=0
    if part!='pilot':
        folder.mkdir(exist_ok=True);predicted=sum(np.prod(v)*2+256 for v in shapes.values());existing=sum((folder/k).stat().st_size for k in shapes if (folder/k).exists());assert shutil.disk_usage(P.OUT).free>predicted-existing+4*2**30
        maps={k:np.lib.format.open_memmap(folder/k,mode='r+' if (folder/k).exists() else 'w+',dtype=np.uint16,shape=v) for k,v in shapes.items()}
    if part=='controlled':
        db=sqlite3.connect(folder/'samples.sqlite3');db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE IF NOT EXISTS samples(idx INTEGER PRIMARY KEY,id TEXT UNIQUE,body TEXT NOT NULL)');db.commit();completed=db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
    elif part=='natural' and (folder/'progress.json').exists():completed=P.read(folder/'progress.json')['completed']
    manifest={'schema':'mlp_native_bank.v1' if part=='controlled' else 'natural_mlp_dictionary_bank.v1','kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':prov,'state':'collecting','n_items':len(items),'completed':completed,'n_layers':36,'d_ff':9728,'d_model':2560,'n_heads':32,'n_kv_heads':8,'head_dim':128,'captured_blocks':[8,24,35] if part=='controlled' else list(range(36)),'positions':positions.tolist() if part=='natural' else None,'readout_roles':['question_end','assistant_boundary'],'all37hidden':part=='controlled','full_vocabulary_logits':False,'attention_rows_captured':False,'last_hidden_after_final_norm':True,'native_dtype':'bfloat16_uint16_bits','file_shapes':freeze['shapes'],'capture_contract_sha256':file_sha(P.OUT/(part+'-execution-freeze.json'))}
    if part!='pilot':atomic(folder/'manifest.json',manifest)
    rows=[];checks=[];start=time.time();torch.set_num_threads(4)
    with GpuLease():
        assert snapshot(P.MODEL)==plan['model_snapshot'];tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True);m=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
        choices={s:list({tok.encode(v,add_special_tokens=False)[0] for v in [s,' '+s] if len(tok.encode(v,add_special_tokens=False))==1}) for pair in corpus['codes'].values() for s in pair}
        def bits(t):return t.detach().contiguous().view(torch.uint16).cpu().numpy()
        def field(t):return t[:,positions] if part=='natural' else torch.stack([t[j,r['readout_positions'][2:]] for j,r in enumerate(ctx['items'])])
        def write(b,k,t):maps[f'B{b:02d}-{k}.npy'][ctx['lo']:ctx['hi']]=bits(field(t))
        if part=='natural':
            for b,l in enumerate(m.model.layers):hooks.extend([l.mlp.register_forward_pre_hook(lambda _m,args,b=b:write(b,'input',args[0])),l.mlp.register_forward_hook(lambda _m,args,o,b=b:write(b,'output',o))])
        elif part=='controlled':
            for b in [8,24,35]:
                l=m.model.layers[b]
                def gate(_m,args,o,b=b):ctx[b,'G']=field(o);write(b,'G',o)
                def value(_m,args,o,b=b):ctx[b,'V']=field(o);write(b,'V',o)
                def activation(_m,args,b=b):assert torch.equal(field(args[0]),torch.nn.functional.silu(ctx.pop((b,'G')))*ctx.pop((b,'V')));write(b,'A',args[0])
                hooks.extend([l.mlp.gate_proj.register_forward_hook(gate),l.mlp.up_proj.register_forward_hook(value),l.mlp.down_proj.register_forward_pre_hook(activation),l.mlp.register_forward_pre_hook(lambda _m,args,b=b:write(b,'input',args[0])),l.mlp.register_forward_hook(lambda _m,args,o,b=b:write(b,'output',o)),l.self_attn.o_proj.register_forward_pre_hook(lambda _m,args,b=b:write(b,'head_z',args[0])),l.self_attn.o_proj.register_forward_hook(lambda _m,args,o,b=b:write(b,'attention_output',o))])
        P.status(part,'running',done=completed,total=len(items))
        canary=None
        for lo in range(completed,len(items),batch):
            chunk=items[lo:lo+batch];ctx.update(lo=lo,hi=lo+len(chunk),items=chunk);ids=torch.tensor([r['tokens'][:1024] if part=='natural' else r['token_ids'] for r in chunk],device='cuda')
            with torch.inference_mode():o=m(input_ids=ids,use_cache=False,output_hidden_states=part=='controlled',logits_to_keep=1)
            if lo==0:canary=(ids.clone(),o.logits.detach().clone())
            if part=='controlled':maps['hidden.npy'][lo:lo+len(chunk)]=np.stack([bits(field(h)) for h in o.hidden_states],axis=2)
            if part!='natural':
                lp=o.logits[:,-1].float().log_softmax(-1)
                for j,r in enumerate(chunk):
                    top=int(lp[j].argmax());row={**r,'kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'bank_index':lo+j,'choice_gap':float(lp[j,choices[r['expected']]].max()-lp[j,choices[r['contrast']]].max()),'first_top1':top in choices[r['expected']],'top1_id':top,'top1_text':tok.decode([top])};row['correct']=row['choice_gap']>0
                    if part=='controlled':row['before_selector_hidden_sha256']=hashlib.sha256(np.stack([bits(h[j,r['readout_positions'][0]]) for h in o.hidden_states]).tobytes()).hexdigest();row['question_embedding_sha256']=hashlib.sha256(bits(o.hidden_states[0][j,1023]).tobytes()).hexdigest();pending.append((lo+j,row))
                    else:rows.append(row)
            del o
            if (lo+len(chunk))%32==0:
                if maps:
                    for v in maps.values():v.flush()
                if db:
                    with db:
                        for idx,r in pending:db.execute('INSERT INTO samples VALUES(?,?,?)',(idx,r['id'],json.dumps(r,ensure_ascii=False,separators=(',',':'))))
                    pending=[]
                completed=lo+len(chunk);manifest['completed']=completed
                if part!='pilot':atomic(folder/'manifest.json',manifest)
                if part=='natural':atomic(folder/'progress.json',{'completed':completed})
                P.status(part,'running',done=completed,total=len(items),seconds=round(time.time()-start,1))
        for h in hooks:h.remove()
        if canary:
            with torch.inference_mode():clean=m(input_ids=canary[0],use_cache=False,logits_to_keep=1)
            checks.append({'observer_vs_clean_logits_bitwise_equal':torch.equal(clean.logits,canary[1])});assert checks[-1]['observer_vs_clean_logits_bitwise_equal']
        assert snapshot(P.MODEL)==plan['model_snapshot']
    if part=='pilot':
        cells=[]
        for key in sorted({tuple(r[k] for k in ['code','mapping','fact','question','frame']) for r in rows}):
            rr=[r for r in rows if tuple(r[k] for k in ['code','mapping','fact','question','frame'])==key];assert len(rr)==16;cells.append(dict(zip(['code','mapping','fact','question','frame'],key))|{'n':16,'correct':sum(r['correct'] for r in rr),'first_top1':sum(r['first_top1'] for r in rr),'passed':sum(r['correct'] for r in rr)>=15})
        result={'kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':prov,'rows':rows,'cells':cells,'passed':all(c['passed'] for c in cells),'checks':checks,'seconds':time.time()-start};P.save(P.OUT/'pilot-result.json',result);P.status('pilot','done' if result['passed'] else 'behavior_failed',failed_cells=sum(not c['passed'] for c in cells),total_cells=len(cells))
    else:
        manifest.update(state='complete',completed=len(items),checks=checks,seconds=time.time()-start,snapshot_unchanged=True)
        for v in maps.values():v.flush()
        manifest['files']=[{'path':k,'bytes':(folder/k).stat().st_size,'shape':list(shapes[k]),'sha256':file_sha(folder/k)} for k in sorted(shapes)]
        if db:db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();manifest['sample_index_sha256']=file_sha(folder/'samples.sqlite3')
        atomic(folder/'manifest.json',manifest);P.status(part,'done',inputs=len(items),native_bytes=sum(r['bytes'] for r in manifest['files']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['pilot','natural','controlled']);main(p.parse_args().part)
