"""Original CUDA/HF greedy generation with resumable fixed-arm batches."""
import time,gc,sys
import b36_map14 as M
S=M.S

def admit(p):
    for name,h in p['source_hashes'].items():assert S.file_sha(S.ROOT/'tests/glm5'/name)==h,name

def run():
    S.configure(300*2**20);p=S.read(S.OUT/'plan.json');admit(p);start=time.monotonic()
    for i,f in enumerate(p['model_files']):assert S.file_sha(S.MODEL/f['path'])==f['sha256'];S.log('identity',i+1,len(p['model_files']),start)
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
                if b==39:state['step']+=1;S.log(state['arm'],state['step'],state['cap'],state['batch_start'],global_elapsed=round(time.monotonic()-start,2))
            hooks.extend([layer.register_forward_pre_hook(load),layer.register_forward_hook(release)])
        for ai,arm in enumerate(M.ARMS):
            dest=S.OUT/(arm+'.json')
            if dest.exists():assert S.read(dest)['plan_sha256']==S.file_sha(S.OUT/'plan.json');continue
            rows=[r for r in p['rows'] if r['arm']==arm];cap=p['generation']['max_tokens'][arm.rstrip('02')];state.update(step=0,arm=arm,cap=cap,batch_start=time.monotonic())
            S.log('batch_start',ai,len(M.ARMS),start,arm=arm)
            inputs=tok([r['prompt'] for r in rows],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
            out=model.generate(**inputs,max_new_tokens=cap,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
            results=[]
            for row,seq in zip(rows,out):
                ids=seq[inputs['input_ids'].shape[1]:].tolist();eos=tok.eos_token_id in ids
                if eos:ids=ids[:ids.index(tok.eos_token_id)+1]
                text=tok.decode(ids,skip_special_tokens=False);results.append(dict(id=row['id'],tokens=ids,text=text,parsed=M.evaluate(text,row,eos)))
            admit(p);S.save(dest.name,dict(results=results,plan_sha256=S.file_sha(S.OUT/'plan.json'),arm=arm,seconds=time.monotonic()-state['batch_start']))
            del inputs,out;gc.collect();torch.cuda.empty_cache();S.log('batch_complete',ai+1,len(M.ARMS),start,arm=arm)
        for h in hooks:h.remove()
    print('All672 generated records complete',flush=True)

if __name__=='__main__':run()
