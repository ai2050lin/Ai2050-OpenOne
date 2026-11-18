"""Numerical matched-control calibration on already-open discovery material only."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
import mlp_condition_v2 as V
V.configure()
import mlp_condition_design as D
import mlp_condition_run as R
import mlp_condition_stream_load as L
from time import monotonic

def main():
    plan=D.checked();selection=D.READ(D.OUT/'selection.json');checks=[]
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    with R.GpuLease(),torch.inference_mode():
        model=L.load(D.MODEL,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
        assay=R.Assay(model);start=monotonic();families=D.families(plan,'discovery')[:64:5]
        try:
            for f in families:
                row=f['rows'][(f['family']//4)%16];original,base=assay.observe([row])
                for label,src,u,dst,v in D.PAIRS:
                    for dose in D.DOSES:
                        state,delta=assay.source_edit(base,src,u,dose*selection[label]['scale'])
                        null,n,cn,valid=R.norm_match(base[f'H{src}'][0,-1],state[0,-1],torch.Generator(device='cuda').manual_seed(D.SEED+f['family']))
                        checks.append({'id':row['id'],'pair':label,'dose':dose,'norm':n,'control_norm':cn,'valid':valid})
                    assert torch.equal(assay.suffix(base[f'H{src}'],src+1),original)
                R.log('matched_control_calibration',len(checks)//6,len(families),start)
        finally:assay.close()
    result={'checks':checks,'valid':sum(r['valid'] for r in checks),'total':len(checks),
            'scope':'numerical calibration only on opened discovery; no effect-based selection or threshold changes',
            'before_holdout':not list(D.OUT.glob('holdout-[0-9]*.npz'))}
    R.save('matched-control-calibration.json',result,plan)
    print({'valid':result['valid'],'total':result['total']},flush=True)
    assert result['valid']==result['total']

if __name__=='__main__':main()
