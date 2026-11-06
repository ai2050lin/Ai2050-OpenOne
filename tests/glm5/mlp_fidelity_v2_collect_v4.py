"""Natural generation gate and selected native fields; no coded-choice gate."""
import sys,time,json,sqlite3,hashlib,shutil,argparse,re
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_plan as P
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha,digest
from shared.dist_protocol.natural_register_runner import snapshot
from mlp_atlas_code_collect import atomic

def first_word(text):
    m=re.match(r'^\s*[\"\x27`]*(\w+)',text.lower());return m.group(1) if m else ''

def greedy(model,cache,logits,eos_ids,limit=8):
    import torch
    rows=[[] for _ in range(len(logits))];done=[False]*len(rows);next_id=logits.argmax(-1)
    for step in range(limit):
        for j,tok in enumerate(next_id.tolist()):
            if not done[j]:
                if tok in eos_ids:done[j]=True
                else:rows[j].append(tok)
        if all(done) or step==limit-1:break
        with torch.inference_mode():o=model(input_ids=next_id[:,None],past_key_values=cache,use_cache=True,logits_to_keep=1)
        cache=o.past_key_values;next_id=o.logits[:,-1].argmax(-1)
    return rows

def main(part):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    c=P.read(P.OUT/'controlled-v4-corpus.json');pr=P.read(P.PUBLIC/'controlled-v4-prereg.research.json');assert digest(c)==pr['corpus_sha256'];plan=P.read(P.OUT/'plan.json');items=[r for r in c['items'] if part=='controlled' or r['background_index']<32];n=len(items);batch=4;folder=P.OUT/'controlled-bank';rows=[];pending=[];hooks=[];ctx={'capture':False};maps={};db=None;completed=0
    if part=='controlled':
        assert P.read(P.OUT/'pilot-v4-result.json')['passed']
        assert P.read(P.OUT/'material-preflight.json')['passed']
    if part=='controlled' and (folder/'manifest.json').exists() and P.read(folder/'manifest.json')['state']=='complete':print('Immutable collection complete; skip.');return
    shapes={f'B{b:02d}-{k}.npy':(n,2,d) for b in [8,24,35] for k,d in [('G',9728),('V',9728),('A',9728),('input',2560),('output',2560),('head_z',4096),('attention_output',2560)]} if part=='controlled' else {}
    if shapes:shapes['hidden.npy']=(n,2,37,2560)
    freeze={'kind':'real','run_id':P.RUN,'part':part,'design_version':4,'source_sha256':file_sha(__file__),'corpus_sha256':digest(c),'prereg_sha256':file_sha(P.PUBLIC/'controlled-v4-prereg.research.json'),'batch':batch,'shapes':{k:list(v) for k,v in shapes.items()}}
    P.save(P.OUT/f'v4-{part}-execution-freeze.json',freeze)
    if shapes:
        folder.mkdir(exist_ok=True);required=sum(np.prod(s)*2+256 for s in shapes.values());existing=sum((folder/k).stat().st_size for k in shapes if (folder/k).exists());assert shutil.disk_usage(P.OUT).free>required-existing+4*2**30;maps={k:np.lib.format.open_memmap(folder/k,mode='r+' if (folder/k).exists() else 'w+',dtype=np.uint16,shape=s) for k,s in shapes.items()};db=sqlite3.connect(folder/'samples.sqlite3');db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE IF NOT EXISTS samples(idx INTEGER PRIMARY KEY,id TEXT UNIQUE,body TEXT NOT NULL)');db.commit();completed=db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
    manifest={'schema':'mlp_native_bank.v1','kind':'real','run_id':P.RUN,'design_version':4,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance'],'state':'collecting','completed':completed,'n_items':n,'n_layers':36,'d_model':2560,'d_ff':9728,'n_heads':32,'n_kv_heads':8,'head_dim':128,'captured_blocks':[8,24,35],'readout_roles':['selected_record_end','assistant_boundary'],'full_vocabulary_logits':False,'attention_rows_captured':False,'all37hidden':True,'last_hidden_after_final_norm':True,'file_shapes':freeze['shapes'],'capture_contract_sha256':file_sha(P.OUT/f'v4-{part}-execution-freeze.json')}
    if db:atomic(folder/'manifest.json',manifest)
    start=time.time();checks=[];canary=None;torch.set_num_threads(4)
    with GpuLease():
        assert snapshot(P.MODEL)==plan['model_snapshot'];tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True);m=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False);eos=m.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos);choices={s:list({tok.encode(v,add_special_tokens=False)[0] for v in [s,s.capitalize(),' '+s,' '+s.capitalize()]}) for s in ['coastal','inland','fishing','mining']}
        def bits(x):return x.detach().contiguous().view(torch.uint16).cpu().numpy()
        def field(x):return torch.stack([x[j,r['readout_positions'][2:]] for j,r in enumerate(ctx['items'])])
        def write(b,k,x):
            if ctx['capture']:maps[f'B{b:02d}-{k}.npy'][ctx['lo']:ctx['hi']]=bits(field(x))
        if db:
            for b in [8,24,35]:
                l=m.model.layers[b]
                def gate(_m,args,o,b=b):
                    if ctx['capture']:ctx[b,'G']=field(o);write(b,'G',o)
                def value(_m,args,o,b=b):
                    if ctx['capture']:ctx[b,'V']=field(o);write(b,'V',o)
                def activation(_m,args,b=b):
                    if ctx['capture']:assert torch.equal(field(args[0]),torch.nn.functional.silu(ctx.pop((b,'G')))*ctx.pop((b,'V')));write(b,'A',args[0])
                hooks.extend([l.mlp.gate_proj.register_forward_hook(gate),l.mlp.up_proj.register_forward_hook(value),l.mlp.down_proj.register_forward_pre_hook(activation),l.mlp.register_forward_pre_hook(lambda _m,args,b=b:write(b,'input',args[0])),l.mlp.register_forward_hook(lambda _m,args,o,b=b:write(b,'output',o)),l.self_attn.o_proj.register_forward_pre_hook(lambda _m,args,b=b:write(b,'head_z',args[0])),l.self_attn.o_proj.register_forward_hook(lambda _m,args,o,b=b:write(b,'attention_output',o))])
        P.status('v4_'+part,'running',done=completed,total=n)
        for lo in range(completed,n,batch):
            chunk=items[lo:lo+batch];ctx.update(capture=bool(db),lo=lo,hi=lo+len(chunk),items=chunk);ids=torch.tensor([r['token_ids'] for r in chunk],device='cuda')
            with torch.inference_mode():o=m(input_ids=ids,use_cache=True,output_hidden_states=bool(db),logits_to_keep=1)
            logits=o.logits[:,-1].float();lp=logits.log_softmax(-1);before=[];qhash=[]
            if db:
                native=np.stack([bits(field(h)) for h in o.hidden_states],axis=2);maps['hidden.npy'][lo:lo+len(chunk)]=native
                for j,r in enumerate(chunk):before.append(hashlib.sha256(np.stack([bits(h[j,r['readout_positions'][0]]) for h in o.hidden_states]).tobytes()).hexdigest());qhash.append(hashlib.sha256(native[j,0].tobytes()).hexdigest())
            if canary is None:canary=(ids.clone(),o.logits.detach().clone())
            ctx['capture']=False;generated=greedy(m,o.past_key_values,logits,eos)
            for j,r in enumerate(chunk):
                text=tok.decode(generated[j],skip_special_tokens=True);word=first_word(text);top=int(logits[j].argmax());row={**r,'kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'bank_index':lo+j,'generated_ids':generated[j],'generated_text':text,'first_word':word,'correct':word==r['expected'],'strict_format':text.strip().lower().strip(' .\n\"\x27`')==r['expected'],'choice_gap':float(lp[j,choices[r['expected']]].max()-lp[j,choices[r['contrast']]].max()),'choice_gap_scope':'first-token prefix diagnostic; generation decidesbehavior','first_top1':top in choices[r['expected']],'top1_id':top,'top1_text':tok.decode([top])}
                if db:row.update(before_selector_hidden_sha256=before[j],selected_record_hidden_sha256=qhash[j],question_embedding_sha256=hashlib.sha256(native[j,0,0].tobytes()).hexdigest());pending.append((lo+j,row))
                else:rows.append(row)
            del o,logits,lp
            if (lo+len(chunk))%32==0:
                if db:
                    for a in maps.values():a.flush()
                    with db:
                        for idx,row in pending:db.execute('INSERT INTO samples VALUES(?,?,?)',(idx,row['id'],json.dumps(row,ensure_ascii=False,separators=(',',':'))))
                    pending=[];manifest['completed']=lo+len(chunk);atomic(folder/'manifest.json',manifest)
                P.status('v4_'+part,'running',done=lo+len(chunk),total=n,seconds=round(time.time()-start,1))
        for h in hooks:h.remove()
        with torch.inference_mode():clean=m(input_ids=canary[0],use_cache=True,logits_to_keep=1)
        assert torch.equal(clean.logits,canary[1]);checks=[{'observer_vs_clean_logits_bitwise_equal':True}];assert snapshot(P.MODEL)==plan['model_snapshot']
    if not db:
        cells=[]
        for key in sorted({tuple(r[k] for k in ['mapping','fact','question','frame']) for r in rows}):
            rr=[r for r in rows if tuple(r[k] for k in ['mapping','fact','question','frame'])==key];assert len(rr)==16;cells.append(dict(zip(['mapping','fact','question','frame'],key))|{'n':16,'correct':sum(r['correct'] for r in rr),'strict_format':sum(r['strict_format'] for r in rr),'passed':sum(r['correct'] for r in rr)>=15})
        result={'kind':'real','run_id':P.RUN,'design_version':4,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance'],'rows':rows,'cells':cells,'passed':all(c['passed'] for c in cells),'checks':checks};P.save(P.OUT/'pilot-v4-result.json',result);P.status('pilot_v4','done' if result['passed'] else 'behavior_failed',failed_cells=sum(not c['passed'] for c in cells),cells=len(cells))
    else:
        for a in maps.values():a.flush()
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();manifest.update(state='complete',completed=n,checks=checks,snapshot_unchanged=True,sample_index_sha256=file_sha(folder/'samples.sqlite3'),files=[{'path':k,'shape':list(shapes[k]),'bytes':(folder/k).stat().st_size,'sha256':file_sha(folder/k)} for k in shapes]);atomic(folder/'manifest.json',manifest);P.status('controlled','done',inputs=n,native_bytes=sum(f['bytes'] for f in manifest['files']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['pilot','controlled']);main(p.parse_args().part)
