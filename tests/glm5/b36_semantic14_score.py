"""Resumable frozen one-token behavioral measurement on original CUDA blocks."""
import argparse,gc,time
import numpy as np
import b36_semantic14 as S

def identity():
    previous=S.read(S.ROOT/'tests/glm5/result/b36_composition14_20261113_v3/model-identity.json');start=time.monotonic()
    for i,f in enumerate(previous['files']):
        p=S.MODEL/f['path'];assert p.stat().st_size==f['bytes'] and S.file_sha(p)==f['sha256']
        S.log('identity',i+1,len(previous['files']),start)
    S.save('model-identity.json',dict(files=previous['files'],config=previous['config'],source_sha256=S.file_sha(__file__)))

def admit(plan):
    for name,h in plan['source_hashes'].items():assert S.file_sha(S.ROOT/'tests/glm5'/name)==h,name
    if 'graph_sha256' in plan:assert S.file_sha(S.OUT/'graph-families.json')==plan['graph_sha256']

def summarize(rows,results):
    by={r['id']:r for r in rows};cells=[]
    for key in sorted({(r['frame'],r['branch'],r['depth'],r['format']) for r in rows}):
        rr=[r for r in results if tuple(by[r['id']][k] for k in ['frame','branch','depth','format'])==key]
        n=len(rr);strict=sum(r['strict_correct'] for r in rr);top=sum(r['top1_correct'] for r in rr)
        cells.append(dict(zip(['frame','branch','depth','format'],key),n=n,strict_correct=strict,top1_correct=top,strict_accuracy=strict/n,top1_accuracy=top/n,passed=min(strict,top)/n>=.9))
    frames=[]
    for frame in sorted({r['frame'] for r in rows}):
        cc=[c for c in cells if c['frame']==frame]
        frames.append(dict(frame=frame,passed=all(c['passed'] for c in cc),worst_accuracy=min(min(c['strict_accuracy'],c['top1_accuracy']) for c in cc)))
    return dict(cells=cells,frames=frames,passed=any(f['passed'] for f in frames))

def run(plan_name='pilot-plan.json',destination='pilot-score'):
    S.configure(60*2**20);plan=S.read(S.OUT/plan_name);admit(plan)
    if not (S.OUT/'model-identity.json').exists():identity()
    rows=plan['rows'];dest=S.OUT/destination;dest.mkdir(exist_ok=True);assert not (dest/'summary.json').exists()
    R=S.bind();start=time.monotonic();results=[]
    with S.GpuLease():
      import torch
      from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
      with torch.inference_mode():
        cfg,pos,pe,causal=R.context()
        for lo in range(0,len(rows),S.CHUNK):
            rr=rows[lo:lo+S.CHUNK];path=dest/f'chunk-{lo:05d}.json'
            if path.exists():
                c=S.read(path);assert c['row_ids']==[r['id'] for r in rr] and c['plan_sha256']==S.file_sha(S.OUT/plan_name);results+=c['results'];continue
            S.log(destination,lo*40,len(rows)*40,start,chunk=lo)
            emb=R.C.weight('model.embed_tokens.weight').cuda();hidden=torch.nn.functional.embedding(torch.from_numpy(R.padded(rr)).cuda(),emb);del emb;torch.cuda.empty_cache();mask=R.mask_for(rr,causal)
            for block in range(40):
                layer=R.C.load_block(block,cfg)
                for i in range(0,len(rr),S.BATCH):hidden[i:i+S.BATCH]=layer(hidden[i:i+S.BATCH],attention_mask=mask[i:i+S.BATCH],position_ids=pos,position_embeddings=pe,use_cache=False)
                del layer;gc.collect();torch.cuda.empty_cache()
                if block%10==9:S.log(destination,lo*40+len(rr)*(block+1),len(rows)*40,start,block=block)
            chosen=hidden[torch.arange(len(rr),device='cuda'),torch.tensor([r['answer_pos'] for r in rr],device='cuda')]
            norm=Qwen3RMSNorm(S.DIM,eps=cfg.rms_norm_eps).cuda();norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False)
            final=norm(chosen);del hidden,mask,chosen,norm;torch.cuda.empty_cache();head=R.C.weight('lm_head.weight').cuda();local=[]
            for i in range(0,len(rr),S.BATCH):
                logits=torch.nn.functional.linear(final[i:i+S.BATCH],head).float()
                for j,r in enumerate(rr[i:i+S.BATCH]):
                    z=logits[j];scores=z[r['choice_ids']].cpu().tolist();values,ids=z.topk(10)
                    local.append(dict(id=r['id'],scores=scores,top1=int(ids[0]),top10_ids=ids.cpu().tolist(),top10_logits=values.cpu().tolist(),
                        strict_correct=scores[r['gold']]>max(v for k,v in enumerate(scores) if k!=r['gold']),top1_correct=int(ids[0])==r['choice_ids'][r['gold']]))
            del head,logits,final;gc.collect();torch.cuda.empty_cache();admit(plan)
            S.save(f'{destination}/{path.name}',dict(row_ids=[r['id'] for r in rr],results=local,plan_sha256=S.file_sha(S.OUT/plan_name)));results+=local
        S.save(f'{destination}/summary.json',dict(kind='real',results=results,report=summarize(rows,results),plan_sha256=S.file_sha(S.OUT/plan_name),source_sha256=S.file_sha(__file__),seconds=time.monotonic()-start))
        S.log(destination+'_complete',len(rows),len(rows),start)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',default='pilot-plan.json');p.add_argument('--destination',default='pilot-score');a=p.parse_args();run(a.plan,a.destination)
