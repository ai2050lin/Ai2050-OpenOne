"""Complete free-generation gate; ephemeral KV cache is only used to score behavior."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D
from source_binding_runtime import Runtime as Base

class Runtime(Base):
    def score(self,logits,rows):
        result=[]
        for i,r in enumerate(rows):
            values=logits[i,r['choice_ids']];pred=int(values.argmax());top=int(logits[i].argmax());gold=r['target_record']
            others=[j for j in range(len(r['choice_ids'])) if j!=gold]
            result.append({'id':r['id'],'pred_record':pred,'top1':top,'forced_correct':pred==gold,'top1_correct':top==r['gold_id'],
                'scores':values.cpu().tolist(),'gold_margin':float(values[gold]-values[others].max())})
        return result

def behavior(model,split):
    import torch
    from transformers import DynamicCache
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    pre=D.verify();root=D.OUT/model;root.mkdir(exist_ok=True)
    if model=='glm9':assert D.read(D.OUT/'qwen14/replication-decision.json')['glm_eligible']
    if split!='discovery':assert D.read(root/'discovery-gate.json')['passed']
    assert not (root/f'{split}-behavior-started.json').exists()
    rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split]
    assert D.sha(D.OUT/f'{model}-materials.json')==pre['materials'][model]
    D.save(f'{model}/{split}-behavior-started.json',{'stage':'behavior only','n':len(rows),'persistent_internal_arrays':False},model)
    timer=time.monotonic();D.log(model+'/'+split+'/behavior',0,len(rows),timer)
    assert snapshot(D.MODELS[model])==pre['models'][model]
    saved=[];calibration=[]
    with GpuLease(),torch.inference_mode():
        rt=Runtime(model);eos=rt.cfg.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        for chunk in range(0,len(rows),64):
            rs=rows[chunk:chunk+64];ids,pos,pe,mask=rt.context(rs);h=rt.embed(ids);L=h.shape[1]
            caches=[DynamicCache(config=rt.cfg) for _ in range(0,len(rs),8)]
            done=[False]*len(rs);generated=[[] for _ in rs];items=None
            for step in range(4):
                for b in range(rt.cfg.num_hidden_layers):
                    layer=rt.block(b)
                    for lo in range(0,len(rs),8):
                        hi=min(lo+8,len(rs));x=h[lo:hi];kw={'attention_mask':mask[lo:hi],'position_ids':pos[lo:hi],
                            'position_embeddings':tuple(p[lo:hi] for p in pe),'use_cache':True,'past_key_values':caches[lo//8]}
                        y=layer(x,**kw)
                        if chunk==0 and step==0 and b in [0,20,39] and lo==0:
                            plain=layer(x,attention_mask=mask[lo:hi],position_ids=pos[lo:hi],position_embeddings=tuple(p[lo:hi] for p in pe),use_cache=False)
                            assert torch.equal(plain,y),'Caching changed baseline';calibration.append({'block':b,'cache_no_cache_equal':True})
                        h[lo:hi]=y
                    del layer;gc.collect();torch.cuda.empty_cache()
                    if (b+1)%10==0:D.log(model+'/'+split+'/behavior-step',chunk*160+step*40*len(rs)+(b+1)*len(rs),len(rows)*160,timer,token_step=step,block=b)
                _,lg=rt.finish(h[:,-1]);token=lg.argmax(-1).cpu().tolist()
                if step==0:items=rt.score(lg,rs)
                for i,v in enumerate(token):
                    if not done[i]:generated[i].append(v);done[i]=v in eos
                del lg
                if all(done):break
                ids=torch.tensor(token,device='cuda')[:,None];h=rt.embed(ids)
                lengths=torch.tensor([len(r['tokens']) for r in rs],device='cuda')
                pos=(lengths+step)[:,None]
                rotary=rt.Rotary(rt.cfg,device='cuda');pe=rotary(torch.zeros((len(rs),1,1),dtype=torch.bfloat16,device='cuda'),pos)
                total=L+step+1;valid=torch.arange(total,device='cuda')[None]>=L-lengths[:,None]
                mask=valid[:,None,None,:]
            for i,r in enumerate(rs):
                text=rt.tok.decode(generated[i],skip_special_tokens=True).strip()
                items[i].update(generated_ids=generated[i],generated_text=text,terminated=done[i],strict_correct=text==r['gold'],
                    normalized_correct=D.normalized(text)==D.normalized(r['gold']))
            saved+=items
            del h,ids,pos,pe,mask,caches;gc.collect();torch.cuda.empty_cache()
            D.log(model+'/'+split+'/behavior',chunk+len(rs),len(rows),timer)
    stat=D.gate(rows,saved)
    D.save(f'{model}/{split}-behavior.json',{'complete':True,'rows':saved,'calibration':calibration,'elapsed_seconds':time.monotonic()-timer},model)
    D.save(f'{model}/{split}-gate.json',{**stat,'behavior_sha256':D.sha(root/f'{split}-behavior.json'),
        'plan_sha256':D.sha(D.OUT/'plan.json'),'model_snapshot':pre['models'][model]},model)
    D.log(model+'/'+split+'/gate',len(rows),len(rows),timer,passed=stat['passed'],forced=stat['forced_accuracy'],generation=stat['normalized_generation_accuracy'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,choices=['qwen14','glm9']);ap.add_argument('--split',required=True,choices=list(D.COUNTS));a=ap.parse_args();behavior(a.model,a.split)
