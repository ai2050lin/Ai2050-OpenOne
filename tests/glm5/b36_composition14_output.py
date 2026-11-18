"""Conditional full-vocabulary fidelity of frozen late-state predictors."""
import argparse,gc,time
import numpy as np
from threadpoolctl import threadpool_limits
import b36_composition14 as B
import b36_composition14_analysis as A
from branch_reuse14_analysis import predict


def run(split='confirm'):
    B.configure(40*2**20)
    from b36_composition14_tail import admit
    admit();rows,_=A.eligible(split);raw=A.state(split,40);h=A.state(split,37);xx=A.inputs(split,1)
    with threadpool_limits(limits=4):
        preds={name:(h+predict(A.get('update-'+name),x)).astype(np.float32) for name,x in xx.items()}
    preds.update(identity=h.astype(np.float32),mean=(h+A.get('update-selectedA')['ym']).astype(np.float32))
    R=B.bind();start=time.monotonic();output=[]
    with B.GpuLease():
      import torch
      from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
      with torch.inference_mode():
        cfg=B.read(B.MODEL/'config.json');norm=Qwen3RMSNorm(B.DIM,eps=cfg['rms_norm_eps']).cuda()
        norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False)
        head=R.C.weight('lm_head.weight').cuda()
        for lo in range(0,len(rows),B.BATCH):
            rr=rows[lo:lo+B.BATCH];teacher=torch.nn.functional.linear(norm(torch.tensor(raw[lo:lo+len(rr)],device='cuda',dtype=torch.bfloat16)),head).float()
            lp=teacher.log_softmax(-1);prob=lp.exp()
            for name,pred in preds.items():
                logits=torch.nn.functional.linear(norm(torch.tensor(pred[lo:lo+len(rr)],device='cuda',dtype=torch.bfloat16)),head).float()
                kl=(prob*(lp-logits.log_softmax(-1))).sum(-1)
                for j,r in enumerate(rr):
                    s=logits[j,r['choice_ids']].cpu().tolist();t=teacher[j,r['choice_ids']].cpu().tolist();g=r['gold']
                    output.append(dict(id=r['id'],family=r['family'],branch=r['branch'],depth=r['depth'],mode=r['mode'],title=r['title'],method=name,KL=float(kl[j]),
                        strict_correct=s[g]>max(v for k,v in enumerate(s) if k!=g),teacher_strict_correct=t[g]>max(v for k,v in enumerate(t) if k!=g),scores=s,teacher_scores=t,gold=g))
            if lo%64==0 or lo+len(rr)==len(rows):B.log('conditional_output_'+split,lo+len(rr),len(rows),start)
        del head,norm;gc.collect();torch.cuda.empty_cache()
    summary=[]
    for key in sorted({(r['branch'],r['depth'],r['mode'],r['method']) for r in output}):
        rr=[r for r in output if (r['branch'],r['depth'],r['mode'],r['method'])==key]
        summary.append(dict(zip(['branch','depth','mode','method'],key),n=len(rr),KL=float(np.mean([r['KL'] for r in rr])),strict_accuracy=float(np.mean([r['strict_correct'] for r in rr]))))
    B.save(f'{split}-output-report.json',dict(kind='real',rows=output,summary=summary,source_sha256=B.file_sha(__file__),
        scope='All predictions carry the same actually measured raw h37; this is conditional late-state output fidelity. No text-only mechanism claim.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='confirm',choices=['confirm','format']);a=p.parse_args();run(a.split)
