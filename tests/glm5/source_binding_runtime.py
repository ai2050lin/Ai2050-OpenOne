"""Identical original BF16 blocks, CPU byte reads, CUDA layer streaming."""
import gc,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D

class Runtime:
    def __init__(self,model):
        import torch
        from transformers import AutoConfig,AutoTokenizer
        self.t=torch;self.key=model;self.path=D.MODELS[model];self.cfg=AutoConfig.from_pretrained(self.path,local_files_only=True)
        self.cfg._attn_implementation='sdpa';self.tok=AutoTokenizer.from_pretrained(self.path,local_files_only=True)
        if self.cfg.model_type=='qwen3':from transformers.models.qwen3 import modeling_qwen3 as module
        else:from transformers.models.glm import modeling_glm as module
        self.mod=module;prefix='Qwen3' if self.cfg.model_type=='qwen3' else 'Glm'
        self.Layer=getattr(module,prefix+'DecoderLayer');self.Norm=getattr(module,prefix+'RMSNorm');self.Rotary=getattr(module,prefix+'RotaryEmbedding')
        self.index=D.read(self.path/'model.safetensors.index.json')['weight_map'];self.headers={}
        for f in set(self.index.values()):
            with (self.path/f).open('rb') as stream:n=int.from_bytes(stream.read(8),'little');self.headers[f]=(8+n,json.loads(stream.read(n)))
        self.dim=self.cfg.hidden_size;self.width=self.cfg.intermediate_size;self.heads=self.cfg.num_attention_heads;self.kv=self.cfg.num_key_value_heads;self.hd=self.cfg.head_dim
        torch.set_num_threads(4);self.audit=[]

    def weight(self,key,device='cuda'):
        t=self.t;shard=self.index[key];offset,hdr=self.headers[shard];spec=hdr[key];assert spec['dtype']=='BF16'
        a,b=spec['data_offsets']
        with (self.path/shard).open('rb') as f:f.seek(offset+a);raw=bytearray(f.read(b-a));assert len(raw)==b-a
        cpu=t.frombuffer(raw,dtype=t.bfloat16).reshape(spec['shape']);out=cpu.to(device)
        return out

    def block(self,b):
        t=self.t
        with t.device('meta'):layer=self.Layer(self.cfg,b)
        prefix=f'model.layers.{b}.';state={k[len(prefix):]:self.weight(k) for k in self.index if k.startswith(prefix)}
        layer.load_state_dict(state,assign=True,strict=True);layer.eval().requires_grad_(False)
        assert all(p.dtype==t.bfloat16 and p.device.type=='cuda' for p in layer.parameters())
        return layer

    def context(self,rows):
        t=self.t;length=max(len(r['tokens']) for r in rows);n=len(rows);pad=self.tok.pad_token_id or self.tok.eos_token_id
        if isinstance(pad,list):pad=pad[0]
        ids=t.full((n,length),pad,device='cuda',dtype=t.long);valid=t.zeros((n,length),device='cuda',dtype=t.bool)
        for i,r in enumerate(rows):ids[i,-len(r['tokens']):]=t.tensor(r['tokens'],device='cuda');valid[i,-len(r['tokens']):]=True
        pos=(valid.long().cumsum(1)-1).clamp_min(0)
        rotary=self.Rotary(self.cfg,device='cuda');pe=rotary(t.zeros((n,length,1),dtype=t.bfloat16,device='cuda'),pos)
        causal=t.ones((length,length),dtype=t.bool,device='cuda').tril()[None,None]&valid[:,None,None,:]
        return ids,pos,pe,causal

    def embed(self,ids):
        w=self.weight('model.embed_tokens.weight');h=self.t.nn.functional.embedding(ids,w);del w;self.t.cuda.empty_cache();return h

    def finish(self,h):
        norm=self.Norm(self.dim,eps=self.cfg.rms_norm_eps).to(device='cuda',dtype=self.t.bfloat16)
        norm.weight=self.t.nn.Parameter(self.weight('model.norm.weight'),requires_grad=False)
        final=norm(h);del norm
        key='lm_head.weight' if 'lm_head.weight' in self.index else 'model.embed_tokens.weight'
        weight=self.weight(key);logits=(final@weight.T).float();del weight;self.t.cuda.empty_cache()
        return final,logits

    def score(self,logits,rows):
        result=[]
        for i,r in enumerate(rows):
            values=logits[i,r['choice_ids']];pred=int(values.argmax());top=int(logits[i].argmax());gold=r['target_record']
            others=[j for j in range(4) if j!=gold]
            result.append({'id':r['id'],'pred_record':pred,'top1':top,'forced_correct':pred==gold,'top1_correct':top==r['gold_id'],
                'scores':values.cpu().tolist(),'gold_margin':float(values[gold]-values[others].max())})
        return result

    def forward(self,rows,callback=None):
        t=self.t;ids,pos,pe,mask=self.context(rows);h=self.embed(ids);length=h.shape[1]
        if callback:callback(-1,None,h,None,rows,length)
        for b in range(self.cfg.num_hidden_layers):
            layer=self.block(b)
            for lo in range(0,len(rows),8):
                hi=min(lo+8,len(rows));x=h[lo:hi];kwargs={'attention_mask':mask[lo:hi],'position_ids':pos[lo:hi],
                    'position_embeddings':tuple(p[lo:hi] for p in pe),'use_cache':False}
                y=layer(x,**kwargs)
                if callback:callback(b,layer,x,y,rows[lo:hi],length,offset=lo,context=kwargs)
                h[lo:hi]=y
            del layer;gc.collect();t.cuda.empty_cache()
        final,logits=self.finish(h[:,-1]);return final,logits
