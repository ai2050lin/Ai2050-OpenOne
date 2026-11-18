"""Original HF generation with CUDA layer streaming and immutable cache copies."""
import gc,time,hashlib
import numpy as np
import b36_switch14 as M
S=M.S


def bits(t):
    import torch
    return t.detach().contiguous().view(torch.uint16).cpu().numpy().copy()


def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def copy_cache(cache):
    return [(layer.keys.detach().cpu().clone(),layer.values.detach().cpu().clone()) for layer in cache.layers]


def cache_description(data):
    return [dict(layer=i,keys_shape=list(k.shape),values_shape=list(v.shape),dtype=str(k.dtype),keys_sha256=tensor_hash(k),values_sha256=tensor_hash(v)) for i,(k,v) in enumerate(data)]


def restore_cache(data,cfg):
    from transformers import DynamicCache
    return DynamicCache(ddp_cache_data=[(k.cuda(),v.cuda()) for k,v in data],config=cfg)


class Stream:
    def __init__(self,start,hard_minutes=180):
        import torch
        from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        self.tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True);self.tok.padding_side='left'
        self.cfg=AutoConfig.from_pretrained(S.MODEL,local_files_only=True)
        assert self.cfg.num_hidden_layers==40 and self.cfg.hidden_size==5120
        assert all(x=='full_attention' for x in self.cfg.layer_types)
        self.R=S.bind();self.start=start;self.hard_minutes=hard_minutes
        with torch.device('meta'):self.model=AutoModelForCausalLM.from_config(self.cfg,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
        for obj,key in [(self.model.model.embed_tokens,'model.embed_tokens.weight'),(self.model.model.norm,'model.norm.weight'),(self.model.lm_head,'lm_head.weight')]:obj.weight=torch.nn.Parameter(self.R.C.weight(key).cuda(),requires_grad=False)
        self.model.model.rotary_emb=Qwen3RotaryEmbedding(self.cfg,device='cpu').cuda()
        self.hooks=[];self.step=0;self.stage='';self.stage_start=start;self.cap=64
        for b,layer in enumerate(self.model.model.layers):
            def load(m,a,b=b):m.load_state_dict(self.R.C.load_block(b,self.cfg).state_dict(),assign=True,strict=True)
            def release(m,a,y,b=b):
                m.to_empty(device='meta')
                if b==39:
                    self.step+=1;S.log(self.stage,self.step,self.cap,self.stage_start,global_elapsed=time.monotonic()-self.start)
                    if time.monotonic()-self.start>self.hard_minutes*60:raise RuntimeError('Frozen GPU runtime cap exceeded')
            self.hooks.extend([layer.register_forward_pre_hook(load),layer.register_forward_hook(release)])

    def inputs(self,rows):
        return self.tok.pad(dict(input_ids=[r['tokens'] for r in rows]),padding=True,return_tensors='pt').to('cuda')

    def generate(self,stage,inputs,cap=64,**kw):
        self.step=0;self.stage=stage;self.cap=cap;self.stage_start=time.monotonic()
        return self.model.generate(**inputs,**kw,max_new_tokens=cap,do_sample=False,use_cache=True,pad_token_id=self.tok.pad_token_id,eos_token_id=self.tok.eos_token_id)

    def records(self,rows,out,prompt_width):
        records=[]
        for row,seq in zip(rows,out):
            ids=seq[prompt_width:].tolist();eos=self.tok.eos_token_id in ids
            if eos:ids=ids[:ids.index(self.tok.eos_token_id)+1]
            text=self.tok.decode(ids,skip_special_tokens=False)
            records.append(dict(id=row['id'],tokens=ids,text=text,parsed=M.T.evaluate(text,row,eos)))
        return records

    def close(self):
        import torch
        for h in self.hooks:h.remove()
        del self.model;gc.collect();torch.cuda.empty_cache()
