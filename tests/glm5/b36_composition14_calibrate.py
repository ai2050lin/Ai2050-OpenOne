"""Discovery-pilot-only source-selection interface calibration."""
import gc
import time
import numpy as np
import b36_composition14 as B


def explicit_prompt(row):
    question=('Ignore the stored output. Calculate by executing the operations in the stated order on the input. Select that calculated value.' if row['mode']=='program' else
              'Ignore the operations. Copy the stored output listed for this input in the archive. Select that recorded value.')
    return row['body']+'\n'+question+'\n'+''.join(f'{"ABC"[i]}. {c}\n' for i,c in enumerate(row['choices']))+'Reply with only the option symbol.\nAnswer:'


def prepare():
    B.configure();assert not (B.OUT/'calibration-plan.json').exists()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(B.MODEL,local_files_only=True);allrows=B.read(B.OUT/'materials.json')['rows']
    selected=[r for r in allrows if r['split']=='pilot' and r['title']==0 and r['rank']<3];assert len(selected)==72
    output=[]
    for r in selected:
        for variant in ['chat','chat_explicit','plain_explicit']:
            prompt=r['prompt'] if variant=='chat' else explicit_prompt(r)
            if variant.startswith('chat'):prompt=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.encode(prompt,add_special_tokens=False);choices=[]
            for c in 'ABC':
                full=tok.encode(prompt+' '+c,add_special_tokens=False);assert full[:-1]==ids;choices.append(full[-1])
            assert len(ids)<=256
            output.append(dict(r,id=r['id']+':'+variant,variant=variant,prompt=prompt,tokens=ids,answer_pos=len(ids)-1,choice_ids=choices))
    B.save('calibration-plan.json',dict(rows=output,n=len(output),source_sha256=B.file_sha(__file__),
        exposure='Only72 already exposed failed-pilot rows, no discovery-fit or confirmation family. Native single-next-token choice ranking, no claim of free generation.',
        metric='Strict correct score > all alternatives; report each branch/depth/source and title0 only.',
        preserve_previous=True,variants=['chat','chat_explicit','plain_explicit'],length=256,batch=4))


def run():
    B.configure();p=B.read(B.OUT/'calibration-plan.json');assert p['source_sha256']==B.file_sha(__file__)
    rows=p['rows'];R=B.bind();R.LENGTH=256;start=time.monotonic();results=[]
    assert not (B.OUT/'calibration-results.json').exists()
    with B.GpuLease():
      import torch
      from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
      with torch.inference_mode():
        cfg,pos,pe,causal=R.context()
        for lo in range(0,len(rows),64):
            rr=rows[lo:lo+64];emb=R.C.weight('model.embed_tokens.weight').cuda();hidden=torch.nn.functional.embedding(torch.from_numpy(R.padded(rr)).cuda(),emb)
            del emb;torch.cuda.empty_cache();mask=R.mask_for(rr,causal)
            for block in range(40):
                layer=R.C.load_block(block,cfg)
                for i in range(0,len(rr),4):hidden[i:i+4]=layer(hidden[i:i+4],attention_mask=mask[i:i+4],position_ids=pos,position_embeddings=pe,use_cache=False)
                del layer;gc.collect();torch.cuda.empty_cache()
                if block%10==9:B.log('pilot_source_calibration',lo*40+len(rr)*(block+1),len(rows)*40,start,block=block)
            selected=hidden[torch.arange(len(rr),device='cuda'),torch.tensor([r['answer_pos'] for r in rr],device='cuda')]
            norm=Qwen3RMSNorm(B.DIM,eps=cfg.rms_norm_eps).cuda();norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False)
            final=norm(selected);del hidden,mask,selected,norm;torch.cuda.empty_cache();head=R.C.weight('lm_head.weight').cuda()
            for i in range(0,len(rr),4):
                logits=torch.nn.functional.linear(final[i:i+4],head).float()
                for j,r in enumerate(rr[i:i+4]):
                    scores=logits[j,r['choice_ids']].cpu().tolist()
                    results.append(dict(id=r['id'],original_id=r['id'].rsplit(':',1)[0],variant=r['variant'],branch=r['branch'],depth=r['depth'],sequence=r['sequence'],mode=r['mode'],family=r['family'],scores=scores,gold=r['gold'],
                        correct=scores[r['gold']]>max(v for k,v in enumerate(scores) if k!=r['gold']),top1=int(logits[j].argmax())))
            del head,logits,final;gc.collect();torch.cuda.empty_cache()
        summary=[]
        for key in sorted({(r['variant'],r['branch'],r['depth'],r['mode']) for r in results}):
            rr=[r for r in results if (r['variant'],r['branch'],r['depth'],r['mode'])==key]
            summary.append(dict(zip(['variant','branch','depth','mode'],key),n=len(rr),correct=sum(r['correct'] for r in rr),accuracy=np.mean([r['correct'] for r in rr]).item()))
        B.save('calibration-results.json',dict(kind='real',rows=results,summary=summary,seconds=time.monotonic()-start,source_sha256=B.file_sha(__file__)))
        B.log('pilot_calibration_complete',len(rows),len(rows),start)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args();globals()[a.stage]()
