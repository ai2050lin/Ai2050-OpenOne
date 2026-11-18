"""Final discovery calibration: described relation without adoption reasoning."""
import gc
import time
import numpy as np
import recursive_branch14 as R
from recursive_branch14_diagnose import variant as old_variant


def variant(row,tok):
    r=old_variant(row,'counterfactual_explicit',tok)
    if r['role']=='quoted':
        opening=['An editor quotes the following instruction for proofreading only: "','A rejected proposal states: "','An example printed for discussion reads: "'][r['frame']]
        closing=['" It has not been adopted. No other rule for the current case is provided.',
                 '" This proposal is not in force, and the current specification supplies no replacement rule.',
                 '" This example does not govern the current case; no applicable instruction has been given.'][r['frame']]
        new_open=['A reference book describes this rule: "','A technical note records the following relationship: "','An explanatory example states: "'][r['frame']]
        body=r['body'].replace(opening,new_open).replace(closing,'"')
        assert body!=r['body']
        r['prompt']=body+r['prompt'][len(r['body']):];r['body']=body
    r['tokens']=tok.encode(r['prompt'],add_special_tokens=False);r['answer_pos']=len(r['tokens'])-1
    r['id']=row['id']+'|neutral-reference';r['diagnostic_variant']='neutral_reference'
    assert len(r['tokens'])<=R.LENGTH
    for s,k in zip([' A',' B',' C'],r['choice_ids']):assert tok.encode(r['prompt']+s,add_special_tokens=False)==r['tokens']+[k]
    return r


def run():
    import torch
    from transformers import AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    R.configure();R.admission();assert not (R.OUT/'neutral/report.json').exists()
    parent=R.read(R.OUT/'diagnosis/report.json');old=R.read(R.OUT/'pilot/collection.json');allrows=R.materials()
    base=[allrows[i] for i in old['row_indices']];assert all(r['split']=='discovery' for r in base)
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True);rows=[variant(r,tok) for r in base]
    cache={R.digest(m['row']['prompt']):m['scores'] for m in parent['metrics']}
    todo={R.digest(r['prompt']):r for r in rows if R.digest(r['prompt']) not in cache}
    query=list(todo.values())+base[:4]
    assert len(query)%4==0
    R.save('neutral/design.json',dict(kind='discovery_protocol_calibration',source_sha256=R.file_sha(__file__),
        earlier_diagnosis_sha256=R.file_sha(R.OUT/'diagnosis/report.json'),parent_plan_sha256=R.file_sha(R.OUT/'plan.json'),
        change='Replace rejected/unadopted quotations by neutral reference-book descriptions; retain counterfactual question and explicit rotation.',
        scope='Common described relation, not applicability reasoning. Old failure remains a negative result. No internal activations analyzed.',
        stop_rule='If any hypothesis/frame/role/title cell below .8, do not further tune this calibration.',
        all_rows=rows,query_rows=query,new_unique_prompts=len(todo),inherited_rows=len(rows)-len(todo)))
    start=time.monotonic();scores=[]
    with R.GpuLease(),torch.inference_mode():
      cfg,pos,pe,causal=R.context()
      for lo in range(0,len(query),R.CHUNK):
        rr=query[lo:lo+R.CHUNK];size=len(rr);mask=R.mask_for(rr,causal)
        emb=R.C.weight('model.embed_tokens.weight').cuda();ids=torch.from_numpy(R.padded(rr)).cuda()
        hidden=torch.nn.functional.embedding(ids,emb);del emb;torch.cuda.empty_cache()
        for b in range(40):
            layer=R.C.load_block(b,cfg)
            for i in range(0,size,R.BATCH):hidden[i:i+R.BATCH]=layer(hidden[i:i+R.BATCH],attention_mask=mask[i:i+R.BATCH],position_ids=pos,position_embeddings=pe,use_cache=False)
            del layer;gc.collect();torch.cuda.empty_cache()
            if b%5==0 or b==39:R.log('neutral_reference_calibration',lo*40+(b+1)*size,len(query)*40,start,block=b,chunk=lo)
        norm=Qwen3RMSNorm(R.DIM,eps=cfg.rms_norm_eps).cuda();norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False)
        at=torch.tensor([r['answer_pos'] for r in rr],device='cuda');final=norm(hidden[torch.arange(size,device='cuda'),at]);wu=R.C.weight('lm_head.weight').cuda()
        for i in range(0,size,R.BATCH):
            logits=torch.nn.functional.linear(final[i:i+R.BATCH],wu).float()
            for j,r in enumerate(rr[i:i+R.BATCH]):scores.append(logits[j,r['choice_ids']].cpu().tolist())
        del hidden,mask,ids,norm,at,final,wu,logits;gc.collect();torch.cuda.empty_cache()
    numeric=[]
    original={r['id']:s for r,s in zip(base,old['scores'])}
    for r,s in zip(query,scores):
        if r['id'] in original:numeric.append(float(np.max(np.abs(np.array(s)-original[r['id']]))))
        cache[R.digest(r['prompt'])]=s
    assert all(x==0 for x in numeric)
    ss=[cache[R.digest(r['prompt'])] for r in rows];cells=R.behavior(rows,ss,by_frame=True)
    R.save('neutral/report.json',dict(kind='real_discovery_calibration',passed=all(c['passed'] for c in cells),cells=cells,
        metrics=[dict(row=r,scores=s,correct=int(np.argmax(s))==r['gold']) for r,s in zip(rows,ss)],
        replay_max_score_errors=numeric,new_unique_prompts=len(todo),source_sha256=R.file_sha(__file__),seconds=time.monotonic()-start,
        limitation='Protocol selected using discovery responses; confirmation still required. Inherited rows are not independent new samples.'))
    R.log('neutral_calibration_complete',len(query),len(query),start)


if __name__=='__main__':run()
