"""Canonical fixed-width right padding: future suffix lengths cannot move prefix tokens."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import source_relay2_design as D
from source_binding_runtime import Runtime as Base

def generation_key_mask(lengths,prompt_width,step):
    """At step1, cached original tokens plus the first newly appended token are visible."""
    import torch
    assert step>=1
    positions=torch.arange(prompt_width+step,device=lengths.device)[None]
    return (positions<lengths[:,None])|(positions>=prompt_width)

class Runtime(Base):
    def __init__(self,model):
        super().__init__(model);self.fixed_length=D.read(D.OUT/'plan.json')['fixed_length']

    def context(self,rows):
        t=self.t;n=len(rows);L=self.fixed_length;pad=self.tok.pad_token_id or self.tok.eos_token_id
        if isinstance(pad,list):pad=pad[0]
        assert max(len(r['tokens']) for r in rows)<=L
        ids=t.full((n,L),pad,device='cuda',dtype=t.long);valid=t.zeros((n,L),device='cuda',dtype=t.bool)
        for i,r in enumerate(rows):ids[i,:len(r['tokens'])]=t.tensor(r['tokens'],device='cuda');valid[i,:len(r['tokens'])]=True
        pos=t.arange(L,device='cuda')[None].expand(n,-1).clone();pos[~valid]=0
        rot=self.Rotary(self.cfg,device='cuda');pe=rot(t.zeros((n,L,1),dtype=t.bfloat16,device='cuda'),pos)
        causal=t.ones((L,L),dtype=t.bool,device='cuda').tril()[None,None]&valid[:,None,None,:]
        return ids,pos,pe,causal

    def score(self,logits,rows):
        result=[]
        for i,r in enumerate(rows):
            values=logits[i,r['choice_ids']];pred=int(values.argmax());top=int(logits[i].argmax());gold=r['target_record']
            others=[j for j in range(len(r['choice_ids'])) if j!=gold]
            result.append({'id':r['id'],'pred_record':pred,'top1':top,'forced_correct':pred==gold,'top1_correct':top==r['gold_id'],
                'scores':values.cpu().tolist(),'gold_margin':float(values[gold]-values[others].max())})
        return result
