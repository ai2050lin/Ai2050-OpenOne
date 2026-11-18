"""Posthoc adversarial baseline, trained only on discovery; no new blind claim.
The full factorial model is an eight-cell lookup. Compare a four-state semantic
color × requested-language lookup and final-token-embedding constant mean.
"""
import time
from datetime import datetime
import numpy as np
import torch
from transformers import AutoTokenizer
import mlp_puzzle_campaign as P

def main():
    plan=P.checked();coef=np.load(P.OUT/'predictors.npz');rows=P.read(P.OUT/'holdout-materials.json');fr=P.read(P.OUT/'frozen.json')
    cols=[0,3,6,7];source=P.sha(__file__)
    P.save('baseline-audit-plan.json',{'created_at':datetime.now().astimezone().isoformat(),'code_sha256':source,
       'predictors_sha256':P.sha(P.OUT/'predictors.npz'),'columns':cols,'status':'posthoc challenge after primary report; coefficients from discovery only; not independent confirmation',
       'question':'Does eight-cell response prediction beat the simpler four-state output-class lookup?','constant':'same final token embedding predicts discovery mean'})
    torch.manual_seed(P.SEED);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True);tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token_id=tok.eos_token_id
    kl=[];groupstats={g['id']:[] for g in fr['groups']};start=time.monotonic()
    with P.GpuLease(),torch.inference_mode():
        m=P.load_model(plan);P.log('baseline_replay',0,512,start)
        for f in range(512):
            rs=rows[f*8:f*8+8];X=np.stack([P.features(r) for r in rs]);batch=tok.pad({'input_ids':[r['tokens'] for r in rs]},return_tensors='pt').to('cuda')
            lo=m(**batch,use_cache=False,logits_to_keep=1).logits[:,0];lp=lo.double().log_softmax(-1);weights=coef['logits'];values={}
            for name,pred in [('full',X@weights),('four_state',X[:,cols]@weights[cols]),('constant',np.broadcast_to(weights[0],(8,151936)))]:
                lq=torch.tensor(pred,device='cuda',dtype=torch.float64).log_softmax(-1);values[name]=float((lp.exp()*(lp-lq)).sum(-1).mean())
            original=np.mean([s['distribution_KL'] for s in P.read(P.OUT/f'holdout-{f:04d}.json')['scores']]);assert abs(values['full']-original)<1e-10
            kl.append({'family':f,**values})
            with np.load(P.OUT/f'holdout-{f:04d}.npz') as z:
                for g in fr['groups']:
                    Y=P.unbits(z[f'A{g["block"]}'])[:,g['units']];c=coef[f'A{g["block"]}'][:,g['units']]
                    groupstats[g['id']].append([float(((Y-X@c)**2).mean()),float(((Y-X[:,cols]@c[cols])**2).mean()),float(((Y-c[0])**2).mean())])
            if (f+1)%32==0:P.log('baseline_replay',f+1,512,start)
    stats={key:{'eight_cell_mse':float(np.mean(v,0)[0]),'four_state_mse':float(np.mean(v,0)[1]),'constant_mse':float(np.mean(v,0)[2]),
                   'eight_over_four':float(np.mean(v,0)[0]/np.mean(v,0)[1])} for key,v in groupstats.items()}
    P.save('baseline-audit.json',{'schema':'native_puzzle_baselines.v1','output_KL':{k:float(np.mean([r[k] for r in kl])) for k in ['full','four_state','constant']},
        'groups':stats,'rows':kl,'plan_sha256':P.sha(P.OUT/'baseline-audit-plan.json'),
        'interpretation':'Four-state baseline is given the authored binding rule qr; same legal prefix information as the full predictor. Neither is a discovered circuit. Posthoc diagnostic.'})

if __name__=='__main__':main()
