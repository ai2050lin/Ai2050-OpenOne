"""Serial nonquantized scale check; actual model coordinate widths, no cross-model index alignment."""
import argparse,gc,os,sys,shutil
from rdc_continuity_common import *
from rdc_continuity_material import build
MODELS={'qwen14':'Qwen3-14B','glm4':'glm4-9b-chat-hf'}

def load(key,out):
    import torch,psutil
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import transformers.modeling_utils as loading
    torch.set_num_threads(4);os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
    path=(ROOT/'models/hf'/MODELS[key]).resolve();assert psutil.virtual_memory().available>13*1024**3
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True,trust_remote_code=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    old=loading.safe_open
    def pread(*a,**kw):kw['backend']='pread';return old(*a,**kw)
    loading.safe_open=pread
    try:model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,device_map='auto',max_memory={0:'12GiB','cpu':'10GiB'},offload_folder=str(out/'checkpoint_offload_index'),offload_state_dict=True,offload_buffers=True,local_files_only=True,trust_remote_code=True,low_cpu_mem_usage=True,attn_implementation='eager').eval()
    finally:loading.safe_open=old
    assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    save(out/'runtime.json',{'timestamp':stamp(),'torch':torch.__version__,'dtype':str(model.dtype),'quantized':False,'device_map':model.hf_device_map,'host_available_after_load':psutil.virtual_memory().available,'source_sha':sha(Path(sys.modules[model.model.__class__.__module__].__file__))})
    return model,tok

class Capture:
    def __init__(self,model):
        self.hooks=[];self.on=False;self.data={};self.n=0;self.native=sorted({0,round(len(model.model.layers)/3)-1,round(2*len(model.model.layers)/3)-1,len(model.model.layers)-1})
        def register(module,k,last=False,pre=False):
            fn=(lambda m,a:self.put(k,a[0],last)) if pre else (lambda m,a,o:self.put(k,o,last))
            self.hooks.append(module.register_forward_pre_hook(fn) if pre else module.register_forward_hook(fn))
        register(model.model.embed_tokens,'H0');register(model.model.norm,'postnorm')
        for l,block in enumerate(model.model.layers):
            register(block,f'H{l+1}')
            if l in self.native:
                if hasattr(block.mlp,'gate_proj'):
                    register(block.mlp.gate_proj,f'L{l}_gate',True);register(block.mlp.up_proj,f'L{l}_up',True)
                else:register(block.mlp.gate_up_proj,f'L{l}_gate_up',True)
                register(block.mlp.down_proj,f'L{l}_a',True,True);register(block.mlp.down_proj,f'L{l}_down',True)
    def put(self,k,t,last):
        if self.on:self.data[k]=bits(t[0,self.n-1:self.n] if last else t[0,:self.n])
    def close(self):
        for h in self.hooks:h.remove()

def main(key,limit):
    import torch
    from transformers import AutoTokenizer
    out=CAMPAIGN/'h_scale'/key;path=ROOT/'models/hf'/MODELS[key]
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True,trust_remote_code=True)
    rows=[r for r in build(tok) if r['unit'] in (0,8,12,13)]
    assert len(rows)==256;maximum=max(len(r['prompt_ids']) for r in rows)
    immutable(out/'material.json',rows)
    immutable(out/'protocol.json',{'phase':2702,'source_sha':sha(Path(__file__)),'model':key,'maximum_cases':256,'pilot':4,'material_sha':sha(out/'material.json'),
      'execution':'nonquantized BF16 device_map auto12GiBGPU+10GiBhost+checkpoint disk as needed; onephysicalmodel. Batch1 globalrightpadding correctmask; allactualtoken H; original checkpoint native MLP factors atlastquery.',
      'max_length':maximum,'noops':'Pilot first4 same-shape with/without hooks must be bitwise identical.',
      'reader':'model-specific native-coordinate fullC at everylayer; no direct4B weights/index transfer; trainunit0 validation8 test12/13; comparesame4B subset.',
      'resource_policy':'Pilot timing before expanding; target <=3600seconds/model and8GiB disk floor. If256 projection exceedsbudget, complete128-case reducedtrain/test with fixedridge, explicitly exploratory; if128 still exceedsbudget, retainpilot and reportresourcebound, notcompletedfullscale.',
      'limits':['Only4 baseunits per family, limited lexical diversity for scale check.','Different modelchat templates and tokenizers are native and recorded.','Same training content is not an independent confirmation dataset acrossmodels.','No claim of functionally isomorphic gears from similar readout accuracy.']})
    if all((out/f'commits/{r["sample_id"]}.json').exists() for r in rows[:limit]):return
    assert shutil.disk_usage(out).free>8*1024**3+sum(len(r['prompt_ids'])*41*(5120 if key=='qwen14' else 4096)*2 for r in rows[:limit])
    announce('h_scale_'+key,state='loading',completed=len(list((out/'commits').glob('*.json'))),total=limit)
    model,tok=load(key,out);cap=Capture(model);device=model.get_input_embeddings().weight.device;times=[]
    try:
      with torch.inference_mode():
       for i,r in enumerate(rows[:limit]):
        if (out/f'commits/{r["sample_id"]}.json').exists():continue
        start=time.monotonic();n=len(r['prompt_ids']);cap.n=n
        ids=torch.tensor([r['prompt_ids']+[tok.pad_token_id]*(maximum-n)],device=device);mask=(torch.arange(maximum,device=device)[None]<n).long()
        cap.on=True;cap.data={};s=model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
        logits=model.lm_head(s[:,n-1:n]).float()[0,0];cap.on=False
        if i<4:
            plain=model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state;assert torch.equal(s,plain);del plain
        h=np.stack([cap.data.pop(f'H{l}') for l in range(len(model.model.layers)+1)])
        assert np.isfinite(unbits(h)).all() and all(np.isfinite(unbits(a)).all() for a in cap.data.values())
        fp=out/f'fields/{r["sample_id"]}.npz';npz(fp,h=h,**cap.data)
        answerids=[tok.encode(w,add_special_tokens=False) for w in (('Yes','No') if r['language']=='en' else ('是','否'))]
        assert all(len(x)==1 for x in answerids);answerids=[x[0] for x in answerids];choice=int(logits.argmax())
        behavior={'first_argmax':choice,'actual_token':tok.decode([choice]),'answer_logits':logits[answerids].cpu().tolist(),'argmax_answer_correct':choice==answerids[0 if r['expected_yes'] else 1],'pair_correct':int(logits[answerids].argmax())==(0 if r['expected_yes'] else 1),'same_shape_noop':True if i<4 else None,'generation_not_performed':True}
        bp=out/f'behavior/{r["sample_id"]}.json';save(bp,behavior);elapsed=time.monotonic()-start
        save(out/f'commits/{r["sample_id"]}.json',{'sample_id':r['sample_id'],'files':{str(p.relative_to(out)):sha(p) for p in (fp,bp)},'elapsed_seconds':elapsed,'fullH_shape':list(h.shape)})
        times.append(elapsed);print('SCALE',key,i+1,limit,round(elapsed,2),flush=True)
        announce('h_scale_'+key,state='running',completed=i+1,total=limit,last_case_seconds=elapsed)
        del h,s,logits;cap.data={};gc.collect()
    finally:cap.close()
    save(out/f'capture_{limit}.json',{'new_case_seconds':times,'mean_seconds':float(np.mean(times)) if times else None,'max_cases':limit})
    announce('h_scale_'+key,state='pilot_complete' if limit==4 else 'captured',completed=limit,total=limit)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=MODELS);p.add_argument('--limit',type=int,choices=(4,128,256),default=4);a=p.parse_args();main(a.model,a.limit)
