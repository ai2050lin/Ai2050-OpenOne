"""Natural cached decode, unchanged native attention implementation, all source K/V/P."""
import argparse,gc,sys,shutil
from rdc_mechanism_common import *
OUT=CAMPAIGN/'c_generation';LAYERS=(11,23,35)

class Capture:
    def __init__(self,model):
        self.enabled=False;self.data={};self.hooks=[]
        self.module=sys.modules[model.model.layers[0].self_attn.__class__.__module__]
        self.original=self.module.eager_attention_forward
        def attention(module,q,k,v,*args,**kwargs):
            result=self.original(module,q,k,v,*args,**kwargs)
            if self.enabled and module.layer_idx in LAYERS:
                l=module.layer_idx
                for name,x in [('q',q[0,:,-1]),('k',k[0]),('v',v[0]),('p',result[1][0,:,-1])]:self.data[f'L{l}_{name}']=bits(x)
            return result
        self.module.eager_attention_forward=attention
        def register(module,key,pre=False):
            fn=(lambda m,a:self.put(key,a[0])) if pre else (lambda m,a,o:self.put(key,o))
            self.hooks.append(module.register_forward_pre_hook(fn) if pre else module.register_forward_hook(fn))
        register(model.model.embed_tokens,'H0');register(model.model.norm,'postnorm')
        for l,block in enumerate(model.model.layers):
            register(block,f'H{l+1}')
            if l in LAYERS:
                for k,m in [('gate',block.mlp.gate_proj),('up',block.mlp.up_proj),('down',block.mlp.down_proj),('attention_out',block.self_attn.o_proj),('mlp_x',block.post_attention_layernorm),('attention_x',block.input_layernorm)]:register(m,f'L{l}_{k}')
                register(block.mlp.down_proj,f'L{l}_a',True);register(block.self_attn.o_proj,f'L{l}_head_output',True)
    def put(self,key,tensor):
        if self.enabled:self.data[key]=bits(tensor[0,-1:])
    def close(self):
        for h in self.hooks:h.remove()
        self.module.eager_attention_forward=self.original

def prepare():
    rows=[r for r in read(CAMPAIGN/'b_relations/material.json') if r['unit'] in (0,6)]
    assert len(rows)==128
    immutable(OUT/'prefixes.json',rows)
    immutable(OUT/'protocol.json',{'source_sha':sha(Path(__file__)),'cases':128,'max_steps':4,'native_layers':list(LAYERS),
        'selection':'All8 families, base0 training andbase6 heldout, all4 fact/query cells andboth languages; selected without model outcome filtering.',
        'capture':'All37 last-query H checkpoints full2560, native MLP full9728; Q lastquery, K/V every actual cache source, P every head/source, actual O-proj input/output.',
        'execution':'BF16 Qwen3-4B eager, batch1, no padding, original forward function return unchanged; natural cache generation until EOS or4steps.',
        'noops':'first16 prefixes compare generated tokens to model.generate same max4 settings',
        'scope':'Subset of B: not another independent relation test. New information is cached continuation and actual native source paths.',
        'budget':'max128x4 steps, floor8GiB; same-shape pilot16 before fullrun.'})
    return rows

def main(limit):
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    rows=prepare()[:limit];assert shutil.disk_usage(OUT).free>8*1024**3
    if all((OUT/f'prefix_commits/{r["sample_id"]}.json').exists() for r in rows):print('ALREADY_COMPLETE');return
    announce('c_generation',state='loading',completed=len(list((OUT/'commits').glob('*.json'))),total=512)
    model,tok=load_native('qwen4');assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    cap=Capture(model);allrows=read(OUT/'material.json') if (OUT/'material.json').exists() else []
    runtime={'timestamp':stamp(),'model_code_sha':sha(Path(cap.module.__file__)),'dtype':str(model.dtype),'cache':True,
        'actual_devices':sorted({str(p.device) for p in model.parameters()}),'torch':torch.__version__}
    save(OUT/'runtime.json',runtime);noops=[]
    try:
        with torch.inference_mode():
            for ri,r in enumerate(rows):
                if (OUT/f'prefix_commits/{r["sample_id"]}.json').exists():continue
                start=time.monotonic();prefix=r['prompt_ids'].copy();generated=[];cache=None;step_ids=[]
                for step in range(4):
                    cap.data={};cap.enabled=True
                    ids=torch.tensor([prefix if step==0 else prefix[-1:]],device='cuda:0')
                    state=model.model(input_ids=ids,use_cache=True,past_key_values=cache);cache=state.past_key_values
                    logits=model.lm_head(state.last_hidden_state[:,-1]).float()[0];cap.enabled=False
                    nextid=int(logits.argmax());h=np.stack([cap.data.pop(f'H{l}') for l in range(37)])
                    assert h.shape==(37,1,2560)
                    assert all(np.isfinite(unbits(a)).all() for a in cap.data.values())
                    for l in LAYERS:assert cap.data[f'L{l}_p'].shape==(32,len(prefix)) and cap.data[f'L{l}_v'].shape==(8,len(prefix),128)
                    sid=f'c-{r["sample_id"]}-s{step}';step_ids.append(sid)
                    answer_ids=[tok.encode(w,add_special_tokens=False)[0] for w in (('Yes','No') if r['language']=='en' else ('是','否'))]
                    row=dict(r,sample_id=sid,prefix_id=r['sample_id'],generation_step=step,query_position=len(prefix)-1,
                        prompt_ids=prefix.copy(),tokens=tok.convert_ids_to_tokens(prefix),prompt=tok.decode(prefix),next_token_id=nextid,next_token=tok.decode([nextid]))
                    behavior={'generated':tok.decode(generated+[nextid],skip_special_tokens=True),'expected_target':r['target'],
                        'correct':tok.decode(generated+[nextid],skip_special_tokens=True).strip().casefold()==r['target'].casefold(),
                        'eos':nextid==tok.eos_token_id,'next_id':nextid,'answer_ids':answer_ids,'yes_no_logits':logits[answer_ids].cpu().tolist(),
                        'first_pair_correct':bool(logits[answer_ids[0]]>logits[answer_ids[1]])==r['expected_yes'] if step==0 else None}
                    fp=OUT/f'fields/{sid}.npz';npz(fp,h=h,logits=logits.cpu().numpy(),**cap.data)
                    bp=OUT/f'behavior/{sid}.json';save(bp,behavior)
                    save(OUT/f'commits/{sid}.json',{'sample_id':sid,'protocol_sha':sha(OUT/'protocol.json'),
                        'files':{str(p.relative_to(OUT)):sha(p) for p in (fp,bp)}})
                    allrows=[x for x in allrows if x['sample_id']!=sid]+[row];save(OUT/'material.json',allrows)
                    announce('c_generation',state='running',completed=len(allrows),total=512,current_sample=sid,completed_prefixes=ri)
                    events('c_generation','generation_step_committed',sample_id=sid,step=step,prefix_tokens=len(prefix))
                    generated.append(nextid);prefix.append(nextid)
                    if nextid==tok.eos_token_id:break
                if ri<16:
                    ids=torch.tensor([r['prompt_ids']],device='cuda:0')
                    plain=model.generate(input_ids=ids,do_sample=False,max_new_tokens=4,use_cache=True,pad_token_id=tok.eos_token_id)[0,len(r['prompt_ids']):].tolist()
                    assert plain==generated,(plain,generated)
                    noops.append({'prefix_id':r['sample_id'],'generated_ids_equal':True})
                save(OUT/f'prefix_commits/{r["sample_id"]}.json',{'prefix_id':r['sample_id'],'steps':step_ids,'generated_ids':generated,
                    'eos':tok.eos_token_id in generated,'elapsed_seconds':time.monotonic()-start})
                print('GEN',ri+1,len(rows),len(generated),round(time.monotonic()-start,2),flush=True)
                del state,cache,logits;cap.data={};gc.collect()
                assert shutil.disk_usage(OUT).free>8*1024**3
                if limit==16:assert time.monotonic()-start<60
    except BaseException as e:
        announce('c_generation',state='failed',error=repr(e),completed=len(allrows),total=512);raise
    finally:cap.close()
    if noops:save(OUT/f'noop_{limit}.json',{'records':noops,'all_passed':True})
    announce('c_generation',state='captured' if limit==128 else 'pilot_complete',completed=len(allrows),total=len(allrows),prefixes=len(list((OUT/'prefix_commits').glob('*.json'))))
    del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,choices=(16,128),default=16);p.add_argument('--prepare',action='store_true');a=p.parse_args()
    prepare() if a.prepare else main(a.limit)
