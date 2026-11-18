"""Late-state conditional full-vocabulary fidelity, without re-running the backbone."""
import time
import gc
import numpy as np
from threadpoolctl import threadpool_limits
import branch_reuse14 as B
import branch_reuse14_collect as C
import branch_reuse14_analysis as A
import branch_reuse14_execution as E
import recursive_branch14 as R


def run():
    B.configure();E.admit();selection=A.admit();rows,meta=A.load('confirm');assert meta['behavior']['passed']
    assert not (B.OUT/'output-fidelity.json').exists()
    start=time.monotonic();raw=A.state('confirm',40,1);roots=selection['graphs']['correlation']['roots']
    reader=A.get('output-root');predictions={}
    with threadpool_limits(limits=4):
      with np.load(B.OUT/'confirm-predictions.npz') as data:
       for method in A.METHODS:
        for mode in ['direct','composed']:
          root=data[f'{method}-r1-{mode}'].reshape(len(rows),-1,3)[:,:,2]
          predictions[f'{method}-{mode}']=A.predict(reader,root)
      for key,x in [('oracle',A.field('confirm',39,'A',1,roots)),('embedding',A.pooled('confirm')),
                    ('lexical',A.lexical(rows,1)),('residual',A.state('confirm',37,1))]:
        predictions[key]=A.predict(reader if key=='oracle' else A.get('output-'+key),x)
      with np.load(B.OUT/'output-mean.npz') as data:predictions['mean']=np.broadcast_to(data['value'],raw.shape)
    results={key:[] for key in predictions};replay=[]
    with B.GpuLease():
      import torch
      from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
      with torch.inference_mode():
        norm=Qwen3RMSNorm(C.DIM,eps=B.read(B.MODEL/'config.json')['rms_norm_eps']).cuda();norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False)
        head=R.C.weight('lm_head.weight').cuda()
        for lo in range(0,len(rows),C.BATCH):
          rr=rows[lo:lo+C.BATCH]
          truth=norm(torch.from_numpy(raw[lo:lo+len(rr)]).to('cuda',torch.bfloat16))
          logits=torch.nn.functional.linear(truth,head).float();lp=logits.log_softmax(-1);prob=lp.exp()
          for j,r in enumerate(rr):replay.append(float(np.max(np.abs(logits[j,r['choice_ids']].cpu().numpy()-np.array(meta['scores'][lo+j])))))
          for key,states in predictions.items():
            pred=norm(torch.tensor(states[lo:lo+len(rr)],dtype=torch.bfloat16,device='cuda'))
            plogits=torch.nn.functional.linear(pred,head).float();plp=plogits.log_softmax(-1)
            kl=(prob*(lp-plp)).sum(-1)
            for j,r in enumerate(rr):
                scores=plogits[j,r['choice_ids']].cpu().tolist()
                results[key].append(dict(KL=float(kl[j]),scores=scores,correct=int(np.argmax(scores))==r['gold'],top1=int(plogits[j].argmax())))
          if lo%64==0:B.log('conditional_output_fidelity',lo+len(rr),len(rows),start)
        assert max(replay)==0.,'Native final norm/readout differs from collected scores'
        del head,norm,truth,logits,lp,prob,pred,plogits,plp,kl;gc.collect();torch.cuda.empty_cache()
    summary=[]
    for branch in A.BRANCHES:
      ix=np.array([i for i,r in enumerate(rows) if r['branch']==branch]);rr=[rows[i] for i in ix]
      for key,values in results.items():
        loss=np.array([values[i]['KL'] for i in ix]);ratios={base:A.ratio_summary(rr,loss,np.array([results[base][i]['KL'] for i in ix])) for base in ['mean','embedding','lexical','residual','oracle'] if base!=key}
        summary.append(dict(branch=branch,method=key,mean_KL=float(loss.mean()),median_KL=float(np.median(loss)),
            accuracy=float(np.mean([values[i]['correct'] for i in ix])),ratios=ratios,
            passed=key.startswith(('correlation','parameter')) and all(ratios[base]['ratio']<=.9 for base in ['mean','embedding'])))
    B.save('output-fidelity.json',dict(kind='real',rows={key:val for key,val in results.items()},row_ids=[r['id'] for r in rows],summary=summary,
        replay_choice_max_abs=max(replay),source_sha256=B.file_sha(__file__),selection_sha256=B.file_sha(B.OUT/'frozen-selection.json'),
        scope='Conditional prediction seeded with actual late-layer state, not text-to-output decoding. Oracle root A also measured. Native RMSNorm/lm_head retained; costs and inputs unequal across strong baselines.'))


if __name__=='__main__':run()
