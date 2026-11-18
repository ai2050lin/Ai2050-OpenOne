"""Fixed native candidates; source choice versus answer-code identity."""
import numpy as np
import record_selection_core_v3 as R
from record_selection_core_v3 import C
C.RUN='SOURCE-OUTPUT-20261109-v1';C.SEED=202611091003
C.LOGICAL_OUT=C.ROOT/'tests/glm5/result/source_output_v1';C.OUT=C.LOGICAL_OUT.resolve()
OLD=C.ROOT/'tests/glm5/result/record_selection_v3'
TASKS=['status','time','explicit'];ALPHABETS=['digits','letters']

def branch(row):return row['task']+':'+row['alphabet']

def score(logits,row,p):
    ids=p['answer_ids'][row['alphabet']];values=[float(logits[i]) for i in ids];margin=values[0]-values[1]
    sign=row['output_sign'];top=int(logits.argmax());target=sign*margin
    return {'pair_logits':values,'pair_margin':margin,'target_margin':target,'strict_correct':bool(target>0),
        'first_top1':top==ids[0 if sign==1 else 1],'tie':target==0,'argmax':top,
        'conditional_correct_probability':float(1/(1+np.exp(-target)))}

def behavior(rows,scores,p):
    assert len(rows)==len(scores);result={}
    for task in TASKS:
        for alphabet in ALPHABETS:
            ids=[i for i,r in enumerate(rows) if r['task']==task and r['alphabet']==alphabet];cells={}
            for i in ids:
                r=rows[i];key=f"position={r['target_sign']};map={r['code_sign']}";cells.setdefault(key,[]).append(i)
                cells.setdefault(f"language={r['language']}",[]).append(i)
            records={key:{'n':len(v),'accuracy':float(np.mean([scores[i]['strict_correct'] for i in v])),
                'top1':float(np.mean([scores[i]['first_top1'] for i in v]))} for key,v in cells.items()}
            result[task+':'+alphabet]={'n':len(ids),'accuracy':float(np.mean([scores[i]['strict_correct'] for i in ids])),
                'top1':float(np.mean([scores[i]['first_top1'] for i in ids])),'errors':int(sum(not scores[i]['strict_correct'] for i in ids)),
                'ties':int(sum(scores[i]['tie'] for i in ids)),'cells':records,
                'passed':bool(records and all(v['n']>=p['minimum_cell_n'] and v['accuracy']>=.9 and v['top1']>=.9 for v in records.values()))}
    return {'schema':'source_output_behavior.v1','n':len(rows),'branches':result,'passed':all(v['passed'] for v in result.values()),
        'rule':'Each entire task x alphabet branch must pass every source-position x mapping and language cell; no row or family filtering.'}

class Capture:
    def __init__(self,model):
        self.model=model;self.base=C.Capture(model);self.handles=[];self.cache={};self.positions=[]
        att=model.model.layers[29].self_attn
        self.handles.append(att.register_forward_pre_hook(self.rope,with_kwargs=True))
        for key,module in [('q',att.q_norm),('k',att.k_norm),('v',att.v_proj)]:self.handles.append(module.register_forward_hook(self.out(key)))
        self.handles.append(att.o_proj.register_forward_pre_hook(self.ins('head')))
        mlp=model.model.layers[26].mlp
        self.handles.append(mlp.gate_proj.register_forward_hook(self.out('gate',True)));self.handles.append(mlp.up_proj.register_forward_hook(self.out('value',True)))
        self.handles.append(mlp.down_proj.register_forward_pre_hook(self.ins('units')))
    def rope(self,mod,args,kwargs):self.cache={'rope':kwargs['position_embeddings'],**{k:v for k,v in self.cache.items() if k in ['gate','value','units']}}
    def out(self,key,selected=False):
        def hook(mod,args,out):self.cache[key]=(out[:,self.positions] if selected else out).detach().clone()
        return hook
    def ins(self,key):
        def hook(mod,args):self.cache[key]=args[0][:,self.positions].detach().clone()
        return hook
    def run(self,row):
        import torch
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb,repeat_kv
        self.positions=row['positions'];self.cache={};logits,base=self.base.forward(row);arrays=self.base.arrays(base)
        x=self.cache;att=self.model.model.layers[29].self_attn;n=len(row['tokens']);cos,sin=x['rope']
        q,k=apply_rotary_pos_emb(x['q'].transpose(1,2),x['k'].transpose(1,2),cos,sin)
        k=repeat_kv(k,att.num_key_value_groups);v=repeat_kv(x['v'].view(1,n,8,128).transpose(1,2),4)
        scores=torch.matmul(q[:,:,self.positions].float(),k.float().transpose(-1,-2))*att.scaling
        for i,position in enumerate(self.positions):scores[:,:,i,position+1:]=float('-inf')
        prob=torch.softmax(scores,-1);replay=(prob@v.float())[0].transpose(0,1);actual=x['head'][0].reshape(2,32,128)
        error=(replay-actual.float()).flatten(1).norm(dim=1)/actual.float().flatten(1).norm(dim=1).clamp_min(1e-8)
        assert float(error.max())<.02
        sources=[row['spans']['record1'],row['spans']['record2'],row['spans']['instruction']];used=set(sum(sources,[]))
        assert len(used)==sum(map(len,sources));sources.append([i for i in range(n) if i not in used])
        mass=torch.stack([prob[0,:,:,ids].sum(-1).transpose(0,1) for ids in sources],-1)
        parts=torch.stack([torch.einsum('hpt,htd->phd',prob[0,[11,15]][:,:,ids],v[0,[11,15]][:,ids].float()) for ids in sources],2)
        assert torch.allclose(parts.sum(2),replay[:,[11,15]],atol=2e-5,rtol=2e-4)
        gate=x['gate'][0];value=x['value'][0];units=x['units'][0]
        assert torch.equal(torch.nn.functional.silu(gate)*value,units)
        data={'hidden':arrays['hidden'],'final_norm':arrays['final_norm'],'head_context':C.bits(actual),
            'attention_prob':prob[0].transpose(0,1).cpu().numpy(),'source_mass':mass.cpu().numpy(),
            'selected_source_context':parts.cpu().numpy(),'gate':C.bits(gate),'value':C.bits(value),'units':C.bits(units),
            'qkv_relative_error':error.cpu().numpy()}
        self.cache.clear();return logits,data
    def close(self):
        self.base.close()
        for handle in self.handles:handle.remove()
