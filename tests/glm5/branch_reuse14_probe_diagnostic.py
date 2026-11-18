"""Discovery-pilot-only question calibration after an applicability wording failure."""
import gc
import time
import numpy as np
import branch_reuse14 as B
import branch_reuse14_collect as C
import recursive_branch14 as R

QUESTIONS={
 'apply':'Apply the document\'s general rule to input {query}, if a general rule is given. A general rule also applies to inputs absent from its examples. If the document gives only individual records and no general rule, answer "Not specified". Which option is correct?',
 'compute':'Determine the output for the new input {query}. Follow any instruction stated to apply to all inputs. Otherwise, if only particular records are given and this input is unlisted, the answer is "Not specified". Do not extend a list of records into a rule. Which option follows?',
 'conditional':'If the document supplies a general instruction, execute it on {query} and choose the result. If it supplies only particular input-output records, choose "Not specified" for this unlisted input. What is the correct option?'
}


def run():
    B.configure(100*2**20);C.admit();C.bind()
    assert not B.read(B.OUT/'pilot/collection.json')['behavior']['passed']
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(B.MODEL,local_files_only=True)
    source=[r for r in B.read(B.OUT/'materials.json')['rows'] if r['split']=='pilot']
    rows=[]
    for variant,question in QUESTIONS.items():
      for r in source:
        tail=question.format(query=r['query'])+'\n'+''.join(f'{"ABC"[j]}. {c}\n' for j,c in enumerate(r['choices']))+'Give only the option letter.\nAnswer:'
        prompt=r['body']+'\n'+tail;ids=tok.encode(prompt,add_special_tokens=False)
        assert len(ids)<=C.LENGTH and ids[:r['content_end']+1]==r['tokens'][:r['content_end']+1]
        for char,cid in zip('ABC',r['choice_ids']):assert tok.encode(prompt+' '+char,add_special_tokens=False)==ids+[cid]
        rows.append(dict(r,id=r['id']+':probe-'+variant,base_id=r['id'],variant=variant,prompt=prompt,tokens=ids,answer_pos=len(ids)-1))
    B.save('probe-diagnostic-plan.json',dict(kind='discovery_pilot_only',rows=rows,
        source_sha256=B.file_sha(__file__),original_materials_sha256=B.file_sha(B.OUT/'materials.json'),
        restriction='Only the exact pilot families; no confirmation input/output. Question differs after the frozen content readout. Best wording may be chosen for a new frozen run, never promoted as independent confirmation.'))
    scores=[];start=time.monotonic();B.log('probe_calibration',0,len(rows)*40,start)
    with B.GpuLease():
      import torch
      from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
      with torch.inference_mode():
        cfg,pos,pe,causal=R.context()
        for lo in range(0,len(rows),64):
          chunk=rows[lo:lo+64];size=len(chunk);embedding=R.C.weight('model.embed_tokens.weight').cuda()
          stage=torch.nn.functional.embedding(torch.from_numpy(R.padded(chunk)).cuda(),embedding)
          del embedding;torch.cuda.empty_cache();mask=R.mask_for(chunk,causal)
          for block in range(40):
            layer=R.C.load_block(block,cfg)
            for i in range(0,size,C.BATCH):
                stage[i:i+C.BATCH]=layer(stage[i:i+C.BATCH],attention_mask=mask[i:i+C.BATCH],position_ids=pos,position_embeddings=pe,use_cache=False)
            del layer;gc.collect();torch.cuda.empty_cache()
            if block%5==4:B.log('probe_calibration',lo*40+size*(block+1),len(rows)*40,start,block=block)
          norm=Qwen3RMSNorm(C.DIM,eps=cfg.rms_norm_eps).cuda();norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False)
          final=norm(R.pick(stage,chunk)[:,1]);del stage,norm,mask;torch.cuda.empty_cache()
          head=R.C.weight('lm_head.weight').cuda()
          for i in range(0,size,C.BATCH):
            logits=torch.nn.functional.linear(final[i:i+C.BATCH],head).float()
            for j,r in enumerate(chunk[i:i+C.BATCH]):scores.append(logits[j,r['choice_ids']].cpu().tolist())
            del logits
          del final,head;gc.collect();torch.cuda.empty_cache()
    reports={}
    for variant in QUESTIONS:
        ix=[i for i,r in enumerate(rows) if r['variant']==variant]
        reports[variant]=C.behavior([rows[i] for i in ix],[scores[i] for i in ix],pilot=True)
    B.save('probe-diagnostic-result.json',dict(kind='real',source_sha256=B.file_sha(__file__),scores=scores,reports=reports,
        plan_sha256=B.file_sha(B.OUT/'probe-diagnostic-plan.json'),seconds=time.monotonic()-start,
        restriction='Exploratory calibration on reused pilot families; no independent sample increase.'))
    print({k:(v['overall_accuracy'],v['passed']) for k,v in reports.items()},flush=True)


if __name__=='__main__':run()
