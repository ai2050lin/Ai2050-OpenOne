"""Discovery-only behavior diagnosis; never analyzes failed-cell activations."""
import gc
import time
import numpy as np
import recursive_branch14 as R


def variant(row,kind,tok):
    r=dict(row);body=r['body'];prompt=r['prompt'];counter='counterfactual' in kind;clear='explicit' in kind
    if clear and r['hypothesis']=='procedure' and r['frame']==0 and r['metadata']['operation']=='rotate':
        body=body.replace('Move the first entry to the end, leaving the other two in their original order.',
            'The output consists of the second entry, then the third entry, then the first entry.')
        prompt=body+prompt[len(r['body']):]
    if counter:
        tail=prompt[len(body):];options=tail[tail.index('\nA. '):]
        if r['hypothesis']=='structure':
            question=f'If the pairing described in the passage were used, which item would be paired with {r["metadata"]["lexical_words"][0]}?'
            unknown='No pairing is specified'
        else:
            question='If the ordering procedure described in the passage were carried out, what would its output be?'
            unknown='No output procedure is specified'
        options=options.replace('Not specified by an applicable rule',unknown)
        prompt=body+'\n'+question+options
        r['gold']=r['metadata']['option_order'] if r['role']!='inventory' else (r['metadata']['option_order']+2)%3
    r.update(body=body,prompt=prompt,tokens=tok.encode(prompt,add_special_tokens=False),diagnostic_variant=kind,original_id=r['id'])
    r['id']=r['id']+'|diagnosis:'+kind;r['answer_pos']=len(r['tokens'])-1
    assert len(r['tokens'])<=R.LENGTH
    for s,k in zip([' A',' B',' C'],r['choice_ids']):assert tok.encode(prompt+s,add_special_tokens=False)==r['tokens']+[k]
    return r


def run():
    import torch
    from transformers import AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    R.configure();R.admission();assert not (R.OUT/'diagnosis/report.json').exists()
    old=R.read(R.OUT/'pilot/collection.json');allrows=R.materials();base=[allrows[i] for i in old['row_indices']]
    assert all(r['split']=='discovery' for r in base)
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    rows=[variant(r,k,tok) for k in ['original','counterfactual','explicit','counterfactual_explicit'] for r in base]
    cache={R.digest(r['prompt']):s for r,s in zip(base,old['scores'])}
    todo={}
    for r in rows:
        key=R.digest(r['prompt'])
        if key not in cache:todo.setdefault(key,r)
    # Same-shape numerical check, separate from behavioral success.
    checks=base[:4];query=list(todo.values())+checks
    if len(query)%4:
        padding=4-len(query)%4;query+=checks[:padding]
    R.save('diagnosis/design.json',dict(kind='discovery_only_behavior_design',source_sha256=R.file_sha(__file__),
        parent_plan_sha256=R.file_sha(R.OUT/'plan.json'),pilot_collection_sha256=R.file_sha(R.OUT/'pilot/collection.json'),
        variants=['original','counterfactual','explicit','counterfactual_explicit'],logical_rows=len(rows),new_unique_prompts=len(todo),
        purpose='Separate applicability-status failure from reading a stated relation; explicit-order arm changes only rotation wording. A counterfactual answer does NOT validate applicability reasoning.',
        threshold='Same .8 per hypothesis/frame/role/title; this is protocol development on discovery, never independent confirmation',
        observed_failure='Original frame0 quoted cells 0/16; frame0 procedure-operative12/16. Numerical full-HF comparison was bitwise.',
        query_rows=query,all_variant_rows=rows))
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
            if b%5==0 or b==39:R.log('discovery_behavior_diagnosis',lo*40+(b+1)*size,len(query)*40,start,block=b,chunk=lo)
        norm=Qwen3RMSNorm(R.DIM,eps=cfg.rms_norm_eps).cuda();norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False)
        at=torch.tensor([r['answer_pos'] for r in rr],device='cuda');final=norm(hidden[torch.arange(size,device='cuda'),at])
        wu=R.C.weight('lm_head.weight').cuda()
        for i in range(0,size,R.BATCH):
            logits=torch.nn.functional.linear(final[i:i+R.BATCH],wu).float()
            for j,r in enumerate(rr[i:i+R.BATCH]):scores.append(logits[j,r['choice_ids']].cpu().tolist())
        del hidden,mask,ids,norm,at,final,wu,logits;gc.collect();torch.cuda.empty_cache()
    numerical=[]
    for r,s in zip(query,scores):
        key=R.digest(r['prompt'])
        if key in cache:numerical.append(dict(id=r['id'],max_choice_score_error=float(np.max(np.abs(np.array(s)-cache[key])))))
        cache[key]=s
    assert all(c['max_choice_score_error']==0 for c in numerical),'Behavior-only engine differs from original same-shape replay'
    cells={};metrics=[]
    for kind in ['original','counterfactual','explicit','counterfactual_explicit']:
        rs=[r for r in rows if r['diagnostic_variant']==kind];ss=[cache[R.digest(r['prompt'])] for r in rs]
        cells[kind]=R.behavior(rs,ss,by_frame=True)
        metrics.extend(dict(row=r,scores=s,correct=int(np.argmax(s))==r['gold']) for r,s in zip(rs,ss))
    R.save('diagnosis/report.json',dict(kind='real_discovery_behavior_diagnosis',cells=cells,metrics=metrics,numerical_replay=numerical,
        seconds=time.monotonic()-start,source_sha256=R.file_sha(__file__),new_unique_prompts=len(todo),
        passed={k:all(c['passed'] for c in cc) for k,cc in cells.items()},
        scope='Discovery only; repeated and inherited rows are not additional independent samples. No G/V/A interpretation of failed cells.'))
    R.log('diagnosis_complete',len(query),len(query),start)


if __name__=='__main__':run()
