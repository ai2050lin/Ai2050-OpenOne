"""Passive original-module hooks, preserving all units and all source positions."""
from rdc_construction_common import *

COORDINATE_FIELDS=['attention_input','attention_write','pre_MLP','MLP_input','MLP_write']
UNIT_FIELDS=['gate','up','product']


class Observer:
    def __init__(self,model):
        self.model=model;self.depth=len(model.model.layers);self.width=model.config.hidden_size
        self.heads=model.config.num_attention_heads;self.head_dim=model.config.head_dim
        self.state={};self.handles=[];self.full=True;self.active=True;self.pending={}
        self.product_checks=0;self.product_max_error=0.0
        def hidden(index,value):
            if self.active:self.state['H'+str(index)]=bits(value[:,-1])
        self.handles.append(model.model.embed_tokens.register_forward_hook(lambda m,a,o:hidden(0,o)))
        def put(index,name,value):
            if self.active and self.full:self.state[(index,name)]=bits(value[:,-1])
        for block,layer in enumerate(model.model.layers):
            self.handles.append(layer.register_forward_hook(lambda m,a,o,i=block:hidden(i+1,o)))
            self.handles.append(layer.input_layernorm.register_forward_hook(lambda m,a,o,i=block:put(i,'attention_input',o)))
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda m,a,i=block:put(i,'pre_MLP',a[0])))
            self.handles.append(layer.post_attention_layernorm.register_forward_hook(lambda m,a,o,i=block:put(i,'MLP_input',o)))
            mlp=layer.mlp
            assert hasattr(mlp,'gate_proj') and hasattr(mlp,'up_proj'),'This runtime qualification currently covers native Qwen, not a guessed GLM MLP API'
            def gate(m,a,o,i=block):
                if self.active and self.full:
                    self.pending[(i,'gate')]=o[:,-1].detach().clone()
                    put(i,'gate',o)
            def up(m,a,o,i=block):
                if self.active and self.full:
                    self.pending[(i,'up')]=o[:,-1].detach().clone()
                    put(i,'up',o)
            def product(m,a,i=block,activation=mlp.act_fn):
                if self.active and self.full:
                    import torch
                    expected=activation(self.pending.pop((i,'gate')))*self.pending.pop((i,'up'))
                    actual=a[0][:,-1]
                    err=float((expected.float()-actual.float()).abs().max())
                    assert torch.equal(expected,actual),('Native gate/up product mismatch',i,err)
                    self.product_checks+=actual.numel();self.product_max_error=max(self.product_max_error,err)
                    put(i,'product',a[0])
            self.handles += [mlp.gate_proj.register_forward_hook(gate),mlp.up_proj.register_forward_hook(up),
                mlp.down_proj.register_forward_pre_hook(product),mlp.register_forward_hook(lambda m,a,o,i=block:put(i,'MLP_write',o))]
            att=layer.self_attn
            def attention(m,a,o,i=block):
                if self.active and self.full:
                    assert o[1] is not None,'Native eager attention weights required'
                    put(i,'attention_write',o[0])
                    self.state[(i,'attention')]=bits(o[1][:,:,-1])
            self.handles.append(att.register_forward_hook(attention))
            def query(m,a,o,i=block):
                if self.active and self.full:
                    value=o[:,-1].reshape(-1,self.heads,self.head_dim)
                    self.state[(i,'Q_before_RoPE')]=bits(value)
            self.handles.append((att.q_norm if hasattr(att,'q_norm') else att.q_proj).register_forward_hook(query))

    def collect(self):
        result={'hidden':np.stack([self.state['H'+str(i)] for i in range(self.depth+1)],axis=1)}
        if self.full:
            for group,names in [('coordinates',COORDINATE_FIELDS),('units',UNIT_FIELDS)]:
                result[group]=np.stack([np.stack([self.state[(block,name)] for name in names],axis=1) for block in range(self.depth)],axis=1)
            result['Q_before_RoPE']=np.stack([self.state[(block,'Q_before_RoPE')] for block in range(self.depth)],axis=1)
            result['attention']=np.stack([self.state[(block,'attention')] for block in range(self.depth)],axis=1)
        assert not self.pending
        self.state.clear()
        return result

    def reset(self,full=True):
        self.state.clear();self.pending.clear();self.active=True;self.full=full

    def close(self):
        for h in self.handles:h.remove()
        self.handles.clear();self.state.clear();self.pending.clear()
