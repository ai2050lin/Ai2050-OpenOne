"""Additional implementation audit; frozen campaign source and thresholds unchanged."""
import json,inspect,hashlib
from datetime import datetime
import torch
import mlp_puzzle_campaign as P

def main():
    # Keep the scientific protocol frozen. Correct only the precision of the
    # already-realized BF16 write difference; rounding this difference again
    # changes the control. Preserve the old source and first failure separately.
    source=inspect.getsource(P.causal)
    assert source.count('actual.to(h.dtype)')==1
    revised=source.replace('actual.to(h.dtype)','actual')
    amendment={'schema':'native_puzzle_control_amendment.v1','created_at':datetime.now().astimezone().isoformat(),
        'reason':'First-family engineering check rejected secondary BF16 rounding of an already realized control delta; no completed causal records existed.',
        'scientific_changes':None,'completed_records_before_amendment':len(list(P.OUT.glob('causal-[0-9]*.json'))),
        'frozen_source_sha256':P.sha(P.__file__),'runner_sha256':P.sha(__file__),
        'executed_function_sha256':hashlib.sha256(revised.encode()).hexdigest(),'executed_function_source':revised,
        'frozen_sha256':P.sha(P.OUT/'frozen.json'),'effect_frozen_sha256':P.sha(P.OUT/'effect-frozen.json'),
        'failed_log_sha256':P.sha(P.OUT/'causal-bf16-failed.log'),
        'scope':'holdout response already observed; candidates, doses, random seed, prediction and scientific gates unchanged; same physical node'}
    path=P.OUT/'control-amendment.json'
    if path.exists():assert P.read(path)['runner_sha256']==amendment['runner_sha256']
    else:
        assert amendment['completed_records_before_amendment']==0
        P.save(path.name,amendment)
    exec(compile(revised,str(__file__)+'::causal_v11','exec'),P.__dict__)
    original=P.Capture.suffix
    forward=P.Capture.forward
    chosen={g['id']:g['units'] for g in P.read(P.OUT/'frozen.json')['causal_groups']}
    checked=set()
    state={'controls':0,'max_actual_delta_error':0.,'full_forward_checks':0}
    def retain_batch(self,*args,**kwargs):
        result=forward(self,*args,**kwargs)
        self.audit_batch=result[2]
        return result
    P.Capture.forward=retain_batch
    def audited(self,base,group,dose,random_delta=None):
        logits,delta=original(self,base,group,dose,random_delta)
        if random_delta is not None:
            error=float((delta-random_delta.float()).abs().max())
            state['max_actual_delta_error']=max(state['max_actual_delta_error'],error)
            state['controls']+=1
            assert error==0, 'BF16 random control realized delta differs from recorded delta'
        elif dose==1 and group['id'] not in checked and group['units']==chosen[group['id']]:
            def ablate(module,args):
                value=args[0].clone();value[:,-1,group['units']]=0
                return (value,)
            hook=self.m.model.layers[group['block']].mlp.down_proj.register_forward_pre_hook(ablate)
            try:
                full=self.m(**self.audit_batch,use_cache=False,logits_to_keep=1).logits[:,0]
                assert torch.equal(full,logits), 'Suffix edit differs from full original model ablation'
            finally:hook.remove()
            checked.add(group['id']);state['full_forward_checks']+=1
        return logits,delta
    P.Capture.suffix=audited
    P.main()
    P.save('causal-implementation-audit.json',state)
    print(json.dumps(state),flush=True)

if __name__=='__main__':main()
