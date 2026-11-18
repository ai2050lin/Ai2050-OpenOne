"""Unmodified HF generation semantics, original BF16 CUDA layer streaming."""
import time,gc
import b36_transition14 as M
S=M.S

def admit(plan):
    for n,h in plan['source_hashes'].items():assert S.file_sha(S.ROOT/'tests/glm5'/n)==h,n

def history_rows(plan,tok):
    source=S.read(S.OUT/'full.json');by={r['id']:r for r in plan['rows'] if r['arm']=='full'};out={a:[] for a in M.HISTORY};missing=[]
    for result in source['results']:
        row=by[result['id']];prefix=M.history_prefix(result['text'],result['tokens'],row['depth'],tok)
        if prefix is None:missing.append(row['id']);continue
        gold=M.replace_steps(prefix['text'],row['truth']);fv=row['truth'][:-1]+[row['foil']];foil=M.replace_steps(prefix['text'],fv)
        assert len(tok.encode(gold,add_special_tokens=False))==len(tok.encode(foil,add_special_tokens=False))
        for arm in M.HISTORY:
            pref=prefix['text'] if arm in ['replay','boundary'] else gold if arm=='gold' else foil
            prefids=prefix['tokens'] if arm in ['replay','boundary'] else tok.encode(pref,add_special_tokens=False)
            if arm!='replay':pref+='\nValue: ';prefids=prefids+tok.encode('\nValue: ',add_special_tokens=False)
            assert tok.decode(prefids,skip_special_tokens=False)==pref
            out[arm].append(dict(row,id=row['family']+':'+arm,arm=arm,tokens=row['tokens']+prefids,prefix=pref,prefix_tokens=prefids,source_id=row['id'],source_prefix=prefix,original_tail=result['tokens'][prefix['cut']:],history_values=prefix['values'] if arm in ['replay','boundary'] else row['truth'] if arm=='gold' else fv))
    S.save('history-plan.json',dict(rows=out,missing=missing,source_sha256=S.file_sha(S.OUT/'full.json'),plan_sha256=S.file_sha(S.OUT/'plan.json'),protocol=plan['history_protocol']))
    return out

def run():
    S.configure(500*2**20);plan=S.read(S.OUT/'plan.json');admit(plan);start=time.monotonic()
    for i,f in enumerate(plan['model_files']):assert S.file_sha(S.MODEL/f['path'])==f['sha256'];S.log('identity',i+1,len(plan['model_files']),start)
    import torch
    from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True);tok.padding_side='left';cfg=AutoConfig.from_pretrained(S.MODEL,local_files_only=True);R=S.bind()
    with S.GpuLease(),torch.inference_mode():
        with torch.device('meta'):model=AutoModelForCausalLM.from_config(cfg,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
        for obj,key in [(model.model.embed_tokens,'model.embed_tokens.weight'),(model.model.norm,'model.norm.weight'),(model.lm_head,'lm_head.weight')]:obj.weight=torch.nn.Parameter(R.C.weight(key).cuda(),requires_grad=False)
        model.model.rotary_emb=Qwen3RotaryEmbedding(cfg,device='cpu').cuda();hooks=[];state=dict(step=0,arm='',cap=64,batch_start=0.)
        for b,layer in enumerate(model.model.layers):
            def load(m,a,b=b):m.load_state_dict(R.C.load_block(b,cfg).state_dict(),assign=True,strict=True)
            def release(m,a,y,b=b):
                m.to_empty(device='meta')
                if b==39:
                    state['step']+=1;S.log(state['arm'],state['step'],state['cap'],state['batch_start'],global_elapsed=round(time.monotonic()-start,2))
                    if time.monotonic()-start>plan['budget']['hard_runtime_minutes']*60:raise RuntimeError('Frozen180 minute GPU runtime cap reached; preserve completed batches.')
            hooks.extend([layer.register_forward_pre_hook(load),layer.register_forward_hook(release)])
        def generate(arm,rows,cap,derived=None):
            dest=S.OUT/(arm+'.json')
            if dest.exists():
                saved=S.read(dest);assert saved['plan_sha256']==S.file_sha(S.OUT/'plan.json')
                if derived:assert saved['history_plan_sha256']==derived
                return
            if not rows:S.save(dest.name,dict(results=[],plan_sha256=S.file_sha(S.OUT/'plan.json'),history_plan_sha256=derived,arm=arm,seconds=0.,reason='No extractable source prefixes.'));return
            state.update(step=0,arm=arm,cap=cap,batch_start=time.monotonic());S.log('batch_start',0,len(rows),start,arm=arm)
            inputs=tok.pad(dict(input_ids=[r['tokens'] for r in rows]),padding=True,return_tensors='pt').to('cuda')
            out=model.generate(**inputs,max_new_tokens=cap,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
            results=[]
            for row,seq in zip(rows,out):
                ids=seq[inputs['input_ids'].shape[1]:].tolist();eos=tok.eos_token_id in ids
                if eos:ids=ids[:ids.index(tok.eos_token_id)+1]
                tail=tok.decode(ids,skip_special_tokens=False);text=row.get('prefix','')+tail
                item=dict(id=row['id'],tokens=ids,tail_text=tail,text=text,parsed=M.evaluate(text,row,eos))
                if derived:item.update(prefix=row['prefix'],replay_exact=ids==row['original_tail'] if arm=='replay' else None)
                results.append(item)
            admit(plan);S.save(dest.name,dict(results=results,plan_sha256=S.file_sha(S.OUT/'plan.json'),history_plan_sha256=derived,arm=arm,seconds=time.monotonic()-state['batch_start']))
            del inputs,out;gc.collect();torch.cuda.empty_cache();S.log('batch_complete',len(rows),len(rows),start,arm=arm)
        for arm in M.ARMS:generate(arm,[r for r in plan['rows'] if r['arm']==arm],M.CAPS[arm])
        if (S.OUT/'history-plan.json').exists():
            hp=S.read(S.OUT/'history-plan.json');assert hp['source_sha256']==S.file_sha(S.OUT/'full.json') and hp['plan_sha256']==S.file_sha(S.OUT/'plan.json');derived=hp['rows']
        else:derived=history_rows(plan,tok)
        hp_sha=S.file_sha(S.OUT/'history-plan.json')
        for arm in M.HISTORY:generate(arm,derived[arm],plan['generation']['history_caps'][arm],hp_sha)
        for h in hooks:h.remove()
    print('All frozen natural and eligible history batches complete',flush=True)

if __name__=='__main__':run()
