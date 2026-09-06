"""Read-only native eager-attention inputs and MLP boundaries for two queries.

Prepared instrumentation, not completed Phase2679. Instrumentation must pass a
same-shape no-op forward comparison before any research interpretation. The
temporary Python call wrapper returns the original function result unchanged;
no model weights, masks, Q/K/V, attention probabilities, or residuals are edited.
"""
import inspect
import numpy as np


class NativeSourceCapture:
    def __init__(self,model,selected):
        assert not model.training
        assert model.config._attn_implementation=='eager'
        self.model=model;self.selected=tuple(selected);self.enabled=False
        self.hooks=[];self.wrappers=[];self.modules={};self.reset(0,0)
        try:
            for l in self.selected:
                b=model.model.layers[l];att=b.self_attn
                assert hasattr(att,'o_proj') and hasattr(b,'post_attention_layernorm')
                self.modules[id(att)]=l
                module=inspect.getmodule(type(att));assert module is not None
                if not any(m is module for m,_,_ in self.wrappers):
                    original=getattr(module,'eager_attention_forward')
                    wrapper=self.make_wrapper(original)
                    setattr(module,'eager_attention_forward',wrapper)
                    self.wrappers.append((module,original,wrapper))
                self.hooks.append(b.register_forward_pre_hook(lambda m,a,kw,l=l:self.block_input(l,a,kw),with_kwargs=True))
                self.hooks.append(att.o_proj.register_forward_pre_hook(lambda m,a,l=l:self.take(l,'native_head_concat',a[0])))
                self.hooks.append(att.register_forward_hook(lambda m,a,out,l=l:self.take(l,'attention_output',out[0])))
                norm=b.post_attention_layernorm
                self.hooks.append(norm.register_forward_pre_hook(lambda m,a,l=l:self.take(l,'pre_mlp_norm',a[0])))
                self.hooks.append(norm.register_forward_hook(lambda m,a,out,l=l:self.take(l,'mlp_x',out)))
                mlp=b.mlp
                if hasattr(mlp,'gate_proj'):
                    self.hooks.append(mlp.gate_proj.register_forward_hook(lambda m,a,out,l=l:self.take(l,'gate',out)))
                    self.hooks.append(mlp.up_proj.register_forward_hook(lambda m,a,out,l=l:self.take(l,'up',out)))
                else:
                    assert hasattr(mlp,'gate_up_proj')
                    self.hooks.append(mlp.gate_up_proj.register_forward_hook(lambda m,a,out,l=l:self.merged(l,out)))
                self.hooks.append(mlp.down_proj.register_forward_pre_hook(lambda m,a,l=l:self.take(l,'mlp_a',a[0])))
                self.hooks.append(mlp.down_proj.register_forward_hook(lambda m,a,out,l=l:self.take(l,'mlp_down',out)))
        except BaseException:
            self.close();raise

    def reset(self,body_token,task_token):
        assert 0<=body_token<=task_token
        self.query_positions=(int(body_token),int(task_token));self.data={};self.calls={}

    @staticmethod
    def array(t):
        # BF16 and FP32 both convert exactly to FP64, but this does NOT make
        # the model execution FP64. Persist actual execution dtype separately.
        return t.detach().double().cpu().numpy().copy()

    def take(self,l,key,t):
        if self.enabled:
            assert t.ndim==3 and t.shape[0]==1
            self.data.setdefault(l,{})[key]=self.array(t[0,list(self.query_positions)])

    def block_input(self,l,args,kwargs):
        if self.enabled:self.take(l,'residual_before_attention',args[0] if args else kwargs['hidden_states'])

    def merged(self,l,t):
        if self.enabled:
            g,u=t.chunk(2,dim=-1);self.take(l,'gate',g);self.take(l,'up',u)

    def make_wrapper(self,original):
        def observed(module,query,key,value,attention_mask,scaling,*args,**kwargs):
            active=self.enabled and id(module) in self.modules
            if active:
                l=self.modules[id(module)];q=list(self.query_positions)
                assert query.shape[0]==key.shape[0]==value.shape[0]==1
                assert key.shape==value.shape and query.shape[-1]==value.shape[-1]
                target=self.data.setdefault(l,{})
                target['actual_query_post_rope']=self.array(query[0,:,q,:].transpose(0,1))
                target['actual_key_post_rope']=self.array(key[0].transpose(0,1))
                target['actual_value']=self.array(value[0].transpose(0,1))
                target['scaling']=float(scaling)
                target['execution_dtype']=str(query.dtype)
                # Preserve infinities in the additive mask; finite64 applies to
                # activations, not the allowed -inf causal-mask representation.
                target['actual_mask']=None if attention_mask is None else attention_mask.detach()[...,q,:].double().cpu().numpy().copy()
            result=original(module,query,key,value,attention_mask,scaling,*args,**kwargs)
            if active:
                output,probabilities=result;assert probabilities is not None
                target['actual_probability']=self.array(probabilities[0,:,q,:].transpose(0,1))
                self.calls[l]=self.calls.get(l,0)+1
            return result
        return observed

    def pack(self):
        assert set(self.data)==set(self.selected) and all(self.calls.get(l)==1 for l in self.selected)
        required={'residual_before_attention','native_head_concat','attention_output','pre_mlp_norm','mlp_x','gate','up','mlp_a','mlp_down',
                  'actual_query_post_rope','actual_key_post_rope','actual_value','actual_probability','actual_mask','scaling','execution_dtype'}
        for l,row in self.data.items():
            assert set(row)==required,(l,set(row)^required)
            assert all(np.isfinite(a).all() for k,a in row.items() if isinstance(a,np.ndarray) and k!='actual_mask')
        return self.data

    def close(self):
        self.enabled=False
        for hook in self.hooks:hook.remove()
        self.hooks=[]
        for module,original,wrapper in reversed(self.wrappers):
            assert getattr(module,'eager_attention_forward') is wrapper,'Another component changed the instrumentation; refuse to overwrite it.'
            setattr(module,'eager_attention_forward',original)
        self.wrappers=[]

    def __enter__(self):return self

    def __exit__(self,*exc):self.close()
