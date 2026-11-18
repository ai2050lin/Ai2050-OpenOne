"""Independent engineering checks before opening the new holdout."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
import mlp_trigger_design as D
import mlp_trigger_run as R
from shared.dist_protocol.mechanism_trace import GpuLease

def main():
    p=D.checked();R.configure(p);assert not list(D.OUT.glob('holdout-[0-9]*.npz'))
    torch.set_num_threads(4);torch.manual_seed(D.SEED);torch.use_deterministic_algorithms(True)
    sel=D.READ(D.OUT/'selection.json');rows=D.materials(p,'discovery');checks=[]
    with GpuLease(),torch.inference_mode():
        model=R.LOADER.load(D.MODEL,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval();a=R.Assay(model)
        try:
            for f in range(16):
                row=rows[8*f+(f%8)];logits,base=a.base(row);pos=a.selected(base,row)
                for key,(b,u,t,v) in D.PAIRS.items():
                    h,r,value,delta=a.edited(base,b,u,pos[key],sel['coordinates'][key],0)
                    assert torch.equal(h,base[f'H{b}']),('zero_state',row['id'],key)
                    assert torch.equal(a.suffix(h,b+1),logits),('zero_logits',row['id'],key)
                    _,_,clamped,_=a.edited(base,b,u,pos[key],sel['coordinates'][key],1,clamp='unit')
                    assert torch.equal(clamped,base[f'A{b}'][0,pos[key],u]),('clamp',row['id'],key)
                checks.append(row['id'])
        finally:a.close()
    D.save('implementation-audit.json',{'audit_code_sha256':D.HASH(__file__),'checked':checks,
         'zero_input_edit_native_state_bitwise':True,'zero_logits_bitwise':True,'source_clamp_exact':True,
         'heldout_unopened':True},p)
    print({'native_checks':len(checks),'passed':True},flush=True)
if __name__=='__main__':main()
