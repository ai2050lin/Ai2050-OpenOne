"""Read-only pre/post QKV normalization and RoPE at all real token coordinates.

Extends the proven read-only source wrapper without replacing model outputs.
Must pass native model no-op equivalence before scientific field collection.
"""
import numpy as np
from phase2679_native_source_capture import NativeSourceCapture

class NativeQKVCapture(NativeSourceCapture):
    def __init__(self,model,selected):
        self.upstream={};self.real_tokens=0
        super().__init__(model,selected)
        try:
            for l in self.selected:
                att=model.model.layers[l].self_attn
                assert hasattr(att,'q_norm') and hasattr(att,'k_norm'),'This capture contract is Qwen3-specific; never silently apply to GLM/DS'
                self.hooks.append(att.register_forward_pre_hook(lambda m,a,kw,l=l:self.entry(l,a,kw),with_kwargs=True))
                for kind in ('q','k','v'):
                    self.hooks.append(getattr(att,kind+'_proj').register_forward_hook(lambda m,a,out,l=l,kind=kind:self.full(l,'linear_'+kind,out)))
                for kind in ('q','k'):
                    self.hooks.append(getattr(att,kind+'_norm').register_forward_hook(lambda m,a,out,l=l,kind=kind:self.full(l,'normalized_'+kind,out)))
        except BaseException:self.close();raise

    def reset(self,body_token,task_token):
        super().reset(body_token,task_token);self.real_tokens=task_token+1;self.upstream={}

    def full(self,l,key,t):
        if self.enabled:
            assert t.shape[0]==1 and t.shape[1]>=self.real_tokens
            self.upstream.setdefault(l,{})[key]=self.array(t[0,:self.real_tokens])

    def entry(self,l,args,kwargs):
        if not self.enabled:return
        self.full(l,'attention_x',args[0] if args else kwargs['hidden_states'])
        cos,sin=kwargs['position_embeddings']
        self.full(l,'rope_cos',cos);self.full(l,'rope_sin',sin)

    def make_wrapper(self,original):
        parent=super().make_wrapper(original)
        def wrapped(module,query,key,value,attention_mask,scaling,*args,**kwargs):
            if self.enabled and id(module) in self.modules:
                l=self.modules[id(module)]
                self.upstream.setdefault(l,{})['query_post_rope_full']=self.array(query[0,:,:self.real_tokens].transpose(0,1))
            return parent(module,query,key,value,attention_mask,scaling,*args,**kwargs)
        return wrapped

    def upstream_pack(self):
        expected={'attention_x','rope_cos','rope_sin','linear_q','linear_k','linear_v','normalized_q','normalized_k','query_post_rope_full'}
        assert set(self.upstream)==set(self.selected)
        for l,data in self.upstream.items():
            assert set(data)==expected and all(np.isfinite(v).all() for v in data.values())
            assert all(v.shape[0]==self.real_tokens for v in data.values())
        return self.upstream
