"""Exposed-confirm diagnostic: map autonomous state predictions to full vocabulary.

No original transformer forward. Real hs[start] is an input, so this measures
partial-state surrogate fidelity, never text-to-output mechanism closure.
"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import common_bench_b1_repair as R
import common_bench_b1_stage_c as C

def main():
    assert not (R.OUT/'rollout-output.json').exists()
    R.save('rollout-output-plan.json',{'scope':'post-hoc exposed confirm diagnostic; real hs14/27/32 inputs, no text-only claim',
        'code_sha256':R.sha(__file__),'source_state_analysis_sha256':R.sha(R.OUT/'state-analysis.json'),
        'final_boundary':'Original rollout stopped at hs35. Fit hs35 -> normalized hs36 directly on discovery; no false residual identity across final norm.',
        'methods':['constant','decoded_A2','linear15'],'readout':'BF16 saved original lm_head, full vocabulary; original normalized state baseline recomputed with identical weights',
        'validation':'Recomputed true-state top1 and forced-class decisions must agree with archived original forward at >=99.9%; maximum accepted-class logit difference <=0.5',
        'metrics':['KL(true||surrogate)','top1_agreement','gold_forced_accuracy','gold_full_vocab_top1_accuracy'],
        'warning':'Final boundary and output metrics added after observing continuous-state failures; not a new frozen confirm experiment'},True)
    start=time.monotonic();a=C.labels('discovery');b=C.labels('confirm');fit=a['screened']&a['correct']&~a['reserved']
    primary=b['screened']&b['correct'];z=np.load(R.OLD/'stagec-estimators.npz');m=np.load(R.OUT/'state-linear15-models.npz')
    P=np.load(R.OUT/'state-linear15-projection.npy');preds={}
    for initial in (14,27,32):
        h=C.layer('confirm',initial);states={name:h.copy() for name in ['constant','decoded_A2','linear15']}
        for l in range(initial,35):
            states['constant']+=z['mean_increment'][l]
            cell=C.nearest(states['decoded_A2'],C.CELL_X[True]@z['coef_A2'][l],np.maximum(z['residual_variance'][l],1e-6))
            states['decoded_A2']+=C.CELL_X[True][cell]@z['increment_A2'][l]
            features=(states['linear15']-m[f'mean_{l}'])/m[f'scale_{l}']@P
            states['linear15']+=np.column_stack([np.ones(len(h)),features])@m[f'W_{l}']
        for name,s in states.items():preds[f'hs{initial}/{name}']=s
        R.log('rollout-output-cpu',initial,32,start)
    h=C.layer('discovery',35)[fit];y=C.layer('discovery',36)[fit];mean=h.mean(0);scale=np.sqrt(np.maximum(h.var(0),1e-6))
    X=np.column_stack([np.ones(len(h)),(h-mean)/scale@P]);penalty=np.eye(16)*.01;penalty[0,0]=0
    W=np.linalg.solve(X.T@X+penalty,X.T@(y-h));inc=z['coef_A2'][36]-z['coef_A2'][35]
    for name,p in preds.items():
        if name.endswith('/constant'):p=p+(y-h).mean(0)
        elif name.endswith('/decoded_A2'):
            cell=C.nearest(p,C.CELL_X[True]@z['coef_A2'][35],np.maximum(z['residual_variance'][35],1e-6));p=p+C.CELL_X[True][cell]@inc
        else:p=p+np.column_stack([np.ones(len(p)),(p-mean)/scale@P])@W
        preds[name]=p.astype(np.float32)
    preds['mean_normalized_state']=np.tile(y.mean(0),(len(primary),1))
    np.savez(R.OUT/'rollout-output-states.npz',**preds)
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    plan=R.read(R.OLD/'plan.json');ids=[plan['choice_ids'][k] for k in R.B.CLASSES]
    true=C.layer('confirm',36);beh=R.read(R.OLD/'stagec-behaviour-confirm.json')
    gold=np.array([v['gold'] for v in beh]);allrows=R.read(R.OLD/'confirm-materials.json')
    metrics={name:[] for name in preds};validation=[]
    with GpuLease(),torch.inference_mode():
        weight=R.native_weight('model.embed_tokens.weight').to('cuda')
        for lo in range(0,len(primary),32):
            end=min(lo+32,len(primary));t=torch.tensor(true[lo:end],device='cuda',dtype=torch.bfloat16)
            logits=(t@weight.T).float();lp=logits.log_softmax(-1);probs=lp.exp();top=logits.argmax(-1)
            sc=torch.stack([logits[:,j].max(1).values for j in ids],1);forced=sc.argmax(1)
            for j in range(end-lo):
                old=beh[lo+j];validation.append({'id':old['id'],'top1_equal':int(top[j])==old['top1'],
                    'forced_equal':int(forced[j])==old['pred'],'max_score_difference':float(np.max(np.abs(sc[j].cpu().numpy()-np.array(old['scores']))))})
            for name,p in preds.items():
                lg=(torch.tensor(p[lo:end],device='cuda',dtype=torch.bfloat16)@weight.T).float()
                pp=torch.stack([lg[:,j].max(1).values for j in ids],1).argmax(1);pt=lg.argmax(1)
                kl=(probs*(lp-lg.log_softmax(-1))).sum(-1)
                for j in range(end-lo):metrics[name].append({'id':beh[lo+j]['id'],'kl':float(kl[j]),
                    'top1_agreement':bool(pt[j]==top[j]),'gold_forced_correct':int(pp[j])==gold[lo+j],
                    'gold_top1_correct':int(pt[j]) in ids[gold[lo+j]]})
            if lo%512==0 or end==len(primary):R.log('rollout-output-cuda',end,len(primary),start)
    valid={'top1_agreement':float(np.mean([v['top1_equal'] for v in validation])),
           'forced_agreement':float(np.mean([v['forced_equal'] for v in validation])),
           'max_score_difference':max(v['max_score_difference'] for v in validation)}
    valid['passed']=valid['top1_agreement']>=.999 and valid['forced_agreement']>=.999 and valid['max_score_difference']<=.5
    summary={}
    for name,rows in metrics.items():
        summary[name]={scope:{k:float(np.mean([r[k] for r,yes in zip(rows,sel) if yes])) for k in ['kl','top1_agreement','gold_forced_correct','gold_top1_correct']} for scope,sel in [('all',np.ones(len(primary),bool)),('primary',primary)]}
    R.save('rollout-output-items.json',{'methods':metrics,'validation':validation},True)
    R.save('rollout-output.json',{'validation':valid,'summary':summary,'primary_n':int(primary.sum()),
        'elapsed_seconds':time.monotonic()-start,'scope':'exposed confirm, partial internal-state surrogate','formal_evidence_promotion':0},True)
    assert valid['passed'],'Readout replay calibration failed: diagnostic results not admissible'
    print(summary,flush=True)

if __name__=='__main__':main()
