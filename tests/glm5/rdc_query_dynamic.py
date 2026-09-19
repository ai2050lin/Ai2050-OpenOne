"""100 finite response tests of an already observed native cached history."""
from collections import defaultdict
from rdc_query_common import *

class QueryEngine:
    def __init__(self,model):
        import torch
        self.model=model;self.probes=read(BASE/'probes/protocol.json')['probes'];self.groups=defaultdict(list)
        for i,p in enumerate(self.probes):self.groups[len(p['token_ids'])].append(i)
        with np.load(BASE/'prototypes/qwen4.npz') as z:self.reference=torch.tensor(z['logprobs'],device='cuda')
    def run(self,cache,verify=False):
        import torch
        before=cache_id(cache) if verify else None;length=cache.get_seq_length();post=np.zeros((100,2560),np.uint16);stats=np.zeros((100,4))
        for _,indices in sorted(self.groups.items()):
          for j0 in range(0,len(indices),16):
            batch=indices[j0:j0+16];own=clone_cache(cache,self.model.config,len(batch))
            o=self.model.model(input_ids=torch.tensor([self.probes[q]['token_ids'] for q in batch],device='cuda'),past_key_values=own,use_cache=True)
            h=o.last_hidden_state[:,-1];lp=self.model.lm_head(h).float().double().log_softmax(-1);p=lp.exp();ref=self.reference[batch]
            stats[batch]=torch.stack([-(p*lp).sum(-1),(p*(lp-ref)).sum(-1),(ref.exp()*(ref-lp)).sum(-1),lp.argmax(-1).double()],-1).cpu().numpy();post[batch]=bits(h)
            del own,o,h,lp,p,ref
        assert all(l.keys.shape[-2]==length for l in cache.layers)
        if verify:assert before==cache_id(cache)
        return post,stats
