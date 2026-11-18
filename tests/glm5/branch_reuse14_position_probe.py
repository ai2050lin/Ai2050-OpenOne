"""Pilot-only option-order diagnostic; no fitting or confirmation access."""
import gc
import time
import numpy as np
import branch_reuse14 as B
import branch_reuse14_collect as C
import recursive_branch14 as R


def run():
    B.configure();C.admit();C.bind()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(B.MODEL,local_files_only=True)
    source=[r for r in B.read(B.OUT/'materials.json')['rows'] if r['split']=='pilot' and r['branch']=='sequence']
    rows=[]
    for shift in [0,1,2]:
      for r in source:
        choices=r['choices'][shift:]+r['choices'][:shift]
        prefix=r['prompt'].split('\nA. ')[0]
        prompt=prefix+'\n'+''.join(f'{"ABC"[j]}. {c}\n' for j,c in enumerate(choices))+'Give only the option letter.\nAnswer:'
        ids=tok.encode(prompt,add_special_tokens=False)
        if shift==0:assert ids==r['tokens']
        assert len(ids)<=C.LENGTH
        rows.append(dict(r,id=r['id']+f':rotate{shift}',base_id=r['id'],shift=shift,choices=choices,
                         gold=(r['gold']-shift)%3,prompt=prompt,tokens=ids,answer_pos=len(ids)-1))
    B.save('option-position-plan.json',dict(rows=rows,source_sha256=B.file_sha(__file__),
        scope='Exact failed sequence pilot families, three cyclic option orders. Calibration only, not independent evidence.'))
    scores=[];start=time.monotonic()
    with B.GpuLease():
      import torch
      from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
      with torch.inference_mode():
        cfg,pos,pe,causal=R.context()
        for lo in range(0,len(rows),64):
          rr=rows[lo:lo+64];size=len(rr);emb=R.C.weight('model.embed_tokens.weight').cuda()
          stage=torch.nn.functional.embedding(torch.from_numpy(R.padded(rr)).cuda(),emb)
          del emb;torch.cuda.empty_cache();mask=R.mask_for(rr,causal)
          for block in range(40):
            layer=R.C.load_block(block,cfg)
            for i in range(0,size,C.BATCH):stage[i:i+C.BATCH]=layer(stage[i:i+C.BATCH],attention_mask=mask[i:i+C.BATCH],position_ids=pos,position_embeddings=pe,use_cache=False)
            del layer;gc.collect();torch.cuda.empty_cache()
            if block%5==4:B.log('option_position_pilot',lo*40+size*(block+1),len(rows)*40,start)
          norm=Qwen3RMSNorm(C.DIM,eps=cfg.rms_norm_eps).cuda();norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False)
          final=norm(R.pick(stage,rr)[:,1]);del stage,norm,mask;torch.cuda.empty_cache()
          head=R.C.weight('lm_head.weight').cuda()
          for i in range(0,size,C.BATCH):
            logits=torch.nn.functional.linear(final[i:i+C.BATCH],head).float()
            for j,r in enumerate(rr[i:i+C.BATCH]):scores.append(logits[j,r['choice_ids']].cpu().tolist())
          del head,final,logits;gc.collect();torch.cuda.empty_cache()
    cells=[]
    for role in ['rule','record']:
      for gold in range(3):
        ix=[i for i,r in enumerate(rows) if r['role']==role and r['gold']==gold]
        cells.append(dict(role=role,gold='ABC'[gold],n=len(ix),accuracy=float(np.mean([np.argmax(scores[i])==gold for i in ix]))))
    prior=B.read(B.OUT/'pilot/collection.json');lookup=dict(zip(prior['row_ids'],prior['scores']))
    replay=max(abs(np.array(scores[i])-lookup[r['base_id']]).max() for i,r in enumerate(rows) if r['shift']==0)
    B.save('option-position-result.json',dict(scores=scores,cells=cells,replay_max_abs=float(replay),seconds=time.monotonic()-start,
        source_sha256=B.file_sha(__file__),plan_sha256=B.file_sha(B.OUT/'option-position-plan.json')))
    print(cells,flush=True)


if __name__=='__main__':run()
