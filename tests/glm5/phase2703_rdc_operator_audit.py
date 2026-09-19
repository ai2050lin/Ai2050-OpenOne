"""Bounded first-observed divergent operator audit, exact source-order eager replay."""
import sys,gc,importlib
from collections import Counter
from rdc_conditional_common import *
OUT=CAMPAIGN/'i_factorial/operator_audit'


class Trace:
    def __init__(self,model):
        import torch
        self.torch=torch;self.enabled=False;self.values=[];self.n=0;self.hooks=[]
        self.module=importlib.import_module(model.model.__class__.__module__)
        self.original=self.module.eager_attention_forward
        self.hooks.append(model.model.embed_tokens.register_forward_hook(lambda m,a,o:self.put('embedding',o)))
        for l,layer in enumerate(model.model.layers):
            for name,module in [('inputnorm',layer.input_layernorm),('q_proj',layer.self_attn.q_proj),('q_norm',layer.self_attn.q_norm),('k_proj',layer.self_attn.k_proj),('k_norm',layer.self_attn.k_norm),('v_proj',layer.self_attn.v_proj),('o_proj',layer.self_attn.o_proj),('mlpnorm',layer.post_attention_layernorm),('gate_proj',layer.mlp.gate_proj),('up_proj',layer.mlp.up_proj),('down_proj',layer.mlp.down_proj)]:
                self.hooks.append(module.register_forward_hook(lambda m,a,o,l=l,name=name:self.put(f'L{l}_{name}',o)))
            self.hooks.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m,a,l=l:self.put(f'L{l}_a',a[0])))
            self.hooks.append(layer.register_forward_hook(lambda m,a,o,l=l:self.put(f'L{l}_postblock',o)))
        self.hooks.append(model.model.norm.register_forward_hook(lambda m,a,o:self.put('finalnorm',o)))
        self.module.eager_attention_forward=self.attention

    def put(self,name,value,kind='token'):
        if not self.enabled:return
        if kind=='matrix':value=value[0,:,:self.n,:self.n]
        elif kind=='heads':value=value[0,:,:self.n]
        else:value=value[0,:self.n]
        self.values.append((name,value.detach().float().cpu().numpy().copy()))

    def attention(self,module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
        t=self.torch;l=module.layer_idx
        self.put(f'L{l}_rope_q',query,'heads');self.put(f'L{l}_rope_k',key,'heads')
        ks=self.module.repeat_kv(key,module.num_key_value_groups);vs=self.module.repeat_kv(value,module.num_key_value_groups)
        score=t.matmul(query,ks.transpose(2,3))*scaling;self.put(f'L{l}_qk_scaled',score,'matrix')
        if attention_mask is not None:score=score+attention_mask
        # Masked -inf values are expected; masked score not subjected to finite statistics.
        probability=t.nn.functional.softmax(score,dim=-1,dtype=t.float32).to(query.dtype)
        self.put(f'L{l}_softmax',probability,'matrix')
        probability=t.nn.functional.dropout(probability,p=dropout,training=module.training)
        output=t.matmul(probability,vs);self.put(f'L{l}_weighted_value',output,'heads')
        return output.transpose(1,2).contiguous(),probability

    def close(self):
        self.module.eager_attention_forward=self.original
        for h in self.hooks:h.remove()


def main():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    rows=[r for r in read(CAMPAIGN/'i_factorial/material.json') if r['unit']==0 and not r['negative_query'] and r['form']==r['style']==0]
    assert len(rows)==32
    immutable(OUT/'protocol.json',{'phase':2703,'code_sha':sha(Path(__file__)),'samples':[r['sample_id'] for r in rows],
      'comparison':'Natural n versus same realtoken IDs rightpadded to fullmaterialmaxlen with explicit causal/padmask; compare only valid positions. All36 layers all declared module/attention operator outputs, fullcoordinates and allheads/source entries.',
      'order':'Exact installed eager code arithmetic order; wrapper adds observations. Validate complete outputs against unwrapped forward bitwise in both shapes for all32 cases.',
      'limits':['First observed divergent operation is not CUDA kernel instruction localization.','RMSNorm remains pertoken hidden-dimension normalization; future padded sources are masked.','Shape discrepancy is numerical, not a linguistic function.']})
    model,tok=load_native('qwen4');maxlen=max(len(r['prompt_ids']) for r in read(CAMPAIGN/'i_factorial/material.json'))
    tracer=Trace(model);results=[]
    try:
      with torch.inference_mode():
       for r in rows:
        n=len(r['prompt_ids']);tracer.n=n;ids=torch.tensor([r['prompt_ids']],device='cuda');track=[];checks=[]
        for matched in (False,True):
            inp=torch.nn.functional.pad(ids,(0,maxlen-n),value=tok.pad_token_id or tok.eos_token_id) if matched else ids
            mask=(torch.arange(inp.shape[1],device='cuda')[None]<n).long()
            tracer.values=[];tracer.enabled=True
            actual=model.model(input_ids=inp,attention_mask=mask,use_cache=False).last_hidden_state;tracer.enabled=False
            track.append(tracer.values);tracer.values=[]
            tracer.module.eager_attention_forward=tracer.original
            plain=model.model(input_ids=inp,attention_mask=mask,use_cache=False).last_hidden_state
            equal=torch.equal(actual,plain);assert equal
            tracer.module.eager_attention_forward=tracer.attention;checks.append(equal)
        assert [k for k,v in track[0]]==[k for k,v in track[1]]
        comparisons=[];first=None
        for (name,a),(_,b) in zip(*track):
            assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            err=b.astype(np.float64)-a
            entry={'operator':name,'shape':list(a.shape),'different_scalars':int(np.count_nonzero(err)),'max_abs':float(np.abs(err).max())}
            comparisons.append(entry)
            if first is None and entry['different_scalars']:
                first=entry
                npz(OUT/f'first_divergence/{r["sample_id"]}.npz',natural=a,matched=b,difference=err)
        results.append({'sample_id':r['sample_id'],'length':n,'matched_length':maxlen,'noop_bitwise':checks,'first_divergence':first,'operators':comparisons})
        save(OUT/'progress.json',{'completed':len(results),'total':32});print('OPERATOR',len(results),None if first is None else first['operator'],flush=True)
        del track,actual,plain;gc.collect()
    finally:tracer.close()
    save(OUT/'result.json',{'phase':2703,'timestamp':stamp(),'cases':results,'first_operator_counts':dict(Counter('none' if r['first_divergence'] is None else r['first_divergence']['operator'] for r in results)),'limits':read(OUT/'protocol.json')['limits']})


if __name__=='__main__':main()
