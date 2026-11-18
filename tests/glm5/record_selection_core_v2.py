"""Record selection: native head / SwiGLU measurements, no surrogate forward."""
import contextlib
import numpy as np
import state_formation_core as C
C.RUN='RECORD-SELECTION-20261109-v2';C.SEED=202611090834
C.LOGICAL_OUT=C.ROOT/'tests/glm5/result/record_selection_v2';C.OUT=C.LOGICAL_OUT.resolve()
HEAD_LAYERS=list(range(18,27));UNIT_LAYERS=[19,23,25]
SOURCES=['record1','record2','query_instruction','other']

class Recorder:
    def __init__(self,model):
        self.model=model;self.base=C.Capture(model);self.cache={};self.handles=[]
        for b,layer in enumerate(model.model.layers):
            att=layer.self_attn
            self.handles.append(att.register_forward_pre_hook(self.rope_hook(b),with_kwargs=True))
            for key,module in [('q',att.q_norm),('k',att.k_norm),('v',att.v_proj)]:self.handles.append(module.register_forward_hook(self.out_hook(b,key)))
            self.handles.append(att.o_proj.register_forward_pre_hook(self.in_hook(b,'head')))
            if b in UNIT_LAYERS:
                self.handles.append(layer.mlp.gate_proj.register_forward_hook(self.out_hook(b,'gate',last=True)))
                self.handles.append(layer.mlp.up_proj.register_forward_hook(self.out_hook(b,'value',last=True)))
                self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self.in_hook(b,'units')))
    def rope_hook(self,b):
        def hook(mod,args,kwargs):self.cache[b]={'rope':kwargs['position_embeddings']}
        return hook
    def out_hook(self,b,key,last=False):
        def hook(mod,args,out):self.cache[b][key]=(out[:,-1:] if last else out).detach()
        return hook
    def in_hook(self,b,key):
        def hook(mod,args):self.cache[b][key]=args[0][:,-1:].detach().clone()
        return hook
    def run(self,row):
        import torch
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb,repeat_kv
        logits,base=self.base.forward(row);arrays=self.base.arrays(base)
        result={'hidden':arrays['hidden'],'final_norm':arrays['final_norm']};heads=[];probabilities=[];mass=[];source_context=[];g=[];v=[];a=[];errors=[]
        n=len(row['tokens']);source_indices=[row['spans']['record1'],row['spans']['record2'],row['spans']['query_instruction']]
        used=set(sum(source_indices,[]));source_indices.append([j for j in range(n) if j not in used])
        assert len(used)==sum(map(len,source_indices[:3])) and sum(map(len,source_indices))==n
        for b in range(36):
            x=self.cache[b];att=self.model.model.layers[b].self_attn
            q,k=apply_rotary_pos_emb(x['q'].transpose(1,2),x['k'].transpose(1,2),*x['rope'])
            k=repeat_kv(k,att.num_key_value_groups);value=repeat_kv(x['v'].view(1,n,-1,att.head_dim).transpose(1,2),att.num_key_value_groups)
            p=torch.softmax(torch.matmul(q[:,:,-1:].float(),k.float().transpose(-1,-2))*att.scaling,-1)
            actual=x['head'].reshape(32,128);replay=(p@value.float())[0,:,0]
            err=float((replay-actual.float()).norm()/actual.float().norm().clamp_min(1e-8));errors.append(err)
            # FP32 QK softmax is a diagnostic decomposition of BF16 SDPA, not
            # claimed bitwise SDPA arithmetic. Both actual vectors and error stay.
            assert err<.02,('QK/V replay tolerance',b,err)
            heads.append(C.bits(actual));probabilities.append(p[0,:,0].cpu().numpy())
            mass.append(torch.stack([p[0,:,0,indices].sum(-1) for indices in source_indices],-1).cpu().numpy())
            if b in HEAD_LAYERS:
                parts=torch.stack([(p[0,:,0,indices,None]*value[0,:,indices].float()).sum(1) for indices in source_indices],1)
                assert torch.allclose(parts.sum(1),replay,atol=2e-5,rtol=2e-4)
                source_context.append(parts.cpu().numpy())
            if b in UNIT_LAYERS:
                gate=x['gate'][0,0];val=x['value'][0,0];unit=x['units'][0,0]
                assert torch.equal(torch.nn.functional.silu(gate)*val,unit),'Native SwiGLU identity failed'
                g.append(C.bits(gate));v.append(C.bits(val));a.append(C.bits(unit))
        result.update(head_context=np.stack(heads),attention_prob=np.stack(probabilities),source_mass=np.stack(mass),
            source_context=np.stack(source_context),gate=np.stack(g),value=np.stack(v),units=np.stack(a),qkv_relative_error=np.array(errors))
        self.cache.clear();return logits,result
    def close(self):
        self.base.close()
        for h in self.handles:h.remove()

@contextlib.contextmanager
def group_edit(model,candidate,dose):
    layer=model.model.layers[candidate['block']];module=layer.self_attn.o_proj if candidate['kind']=='heads' else layer.mlp.down_proj
    ids=candidate['indices']
    def hook(mod,args):
        value=args[0].clone()
        if candidate['kind']=='heads':
            shaped=value[:,-1].view(value.shape[0],32,128);shaped[:,ids]=(shaped[:,ids].float()*(1-dose)).to(value.dtype)
        else:value[:,-1,ids]=(value[:,-1,ids].float()*(1-dose)).to(value.dtype)
        return (value,*args[1:])
    handle=module.register_forward_pre_hook(hook)
    try:yield
    finally:handle.remove()

@contextlib.contextmanager
def group_source_exclusion(model,candidate,positions):
    att=model.model.layers[candidate['block']].self_attn
    def hook(mod,args,kwargs):
        mask=kwargs['attention_mask'];assert mask.dtype==__import__('torch').bool
        mask=mask.expand(mask.shape[0],32,*mask.shape[2:]).clone()
        for head in candidate['indices']:mask[:,head,-1,positions]=False
        return args,{**kwargs,'attention_mask':mask}
    handle=att.register_forward_pre_hook(hook,with_kwargs=True)
    try:yield
    finally:handle.remove()

def score(logits,row,p):
    result=C.scores(logits,row,p);offset=0 if row['language']==1 else 2
    result['red_blue_margin']=result['four_logits'][offset]-result['four_logits'][offset+1]
    result['color_tie']=result['red_blue_margin']==0
    # Tie is half-correct for reporting, strict positive for the frozen gate.
    result['strict_color_correct']=row['color']*result['red_blue_margin']>0
    return result

def behavior(scores,p):
    groups={}
    for r in scores:groups.setdefault(f"{r['frame']}:{r['valid_position']}:{r['conflict']}:{r['language']}",[]).append(r)
    cells={k:{'n':len(v),'strict_accuracy':float(np.mean([r['strict_color_correct'] for r in v])),'top1':float(np.mean([r['first_top1'] for r in v]))} for k,v in groups.items()}
    return {'schema':'record_selection_behavior.v1','cells':cells,'n':len(scores),'passed':all(v['strict_accuracy']>=p['behavior_min'] for v in cells.values()),
        'top1':float(np.mean([r['first_top1'] for r in scores])),'strict_accuracy':float(np.mean([r['strict_color_correct'] for r in scores])),
        'ties':int(sum(r['color_tie'] for r in scores)),'scope':'strict two-color first-token margin, conditioned on requested output language; top1 separately reported'}
