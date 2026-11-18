"""All-layer hidden states plus complete local B36..39 native fields."""
import argparse
import gc
import importlib.metadata
import time
import numpy as np
import b36_composition14 as B


def sources():
    names=['b36_composition14.py','b36_composition14_materials.py','b36_composition14_calibrate_simple.py','b36_composition14_collect.py',
           'b36_composition14_scope_v3.py','recursive_branch14.py','qwen14_mlp_atlas.py','source_binding_resources.py']
    return {n:B.file_sha(B.ROOT/'tests/glm5'/n) for n in names}


def freeze():
    B.configure(int(3.8*2**30));assert not (B.OUT/'trial-plan.json').exists()
    previous=B.read(B.PREVIOUS/'model-identity.json');start=time.monotonic()
    for i,item in enumerate(previous['files']):
        path=B.MODEL/item['path'];assert path.stat().st_size==item['bytes'] and B.file_sha(path)==item['sha256']
        B.log('model_identity',i+1,len(previous['files']),start,file=item['path'])
    B.save('model-identity.json',dict(files=previous['files'],config=previous['config'],snapshot_rehashed=True))
    scope=B.read(B.OUT/'scope-plan.json')
    B.save('trial-plan.json',dict(kind='prospective_behavior_and_collection',source_hashes=sources(),materials_sha256=B.file_sha(B.OUT/'materials.json'),
        scope_plan_sha256=B.file_sha(B.OUT/'scope-plan.json'),eligible_strata=scope['eligible_strata'],planned_native_counts=scope['planned_native_counts'],
        runtime={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','safetensors','scipy','scikit-learn']},
        model='Qwen3-14B original BF16, CPU-initialized rotary, CUDA native SDPA',batch=B.BATCH,length=B.LENGTH,chunk=B.CHUNK,
        target_group=B.group(),target_group_origin_sha256=B.file_sha(B.PREVIOUS/'frozen-selection.json'),
        observation='All41 full residual boundaries at body end and answer end; all G/V/A units B36 and B39 at answer end. B37/B38 answer and all four blocks body use tracked union only; attention/MLP outputs at both positions. B36 group trajectory at all valid tokens; prefix means at embedding/h37. Redundant MLP input/residual omitted; reconstruct from hidden, attention and normalization weights.',
        tracked_body_units=B.tracked(),
        scope_reason='Focused study of B36 group and its last three transitions. All hidden layers retained; earlier complete MLP fields omitted to fit 3.8GiB peak with 20GiB reserve.',
        behavior=dict(metric='gold score strictly exceeds every competitor; ties fail',pilot_cell_min=.85,pilot_overall_min=.9,
                      formal_cell_min=.9,cells=['branch','depth','mode','title'],format_cells=['branch','depth','mode'],
                      action='Joint gate cannot pass with any failed cell. Preregistered partial admission: a branch/depth/mode requires BOTH title cells pass pilot and formal gates. Discovery needs all primitive cells. Failed combination cells remain behavior-only and cannot support internal mechanism claims; successful primitive/source cells may proceed. Format/order also require the corresponding main-confirm cell.'),
        resources=dict(new_peak_GiB=3.8,reserve_GiB=20,shared_GPU_GiB_max=1,serial_models=True),
        hypotheses=['The historical fixed group contributes beyond technical-title, choice-symbol and source-word alternatives.',
                    'Completing residual and context state improves unseen-composition predictions over selected-A-only models.',
                    'Parameter-based local propagation predicts native counterfactual effects; clamped paths separate bypass, attention and MLP mediation.',
                    'Higher composition depth has an effect beyond source format; same information alone does not prove a new complex feature.'],
        further_freeze='Prediction/intervention methods and code freeze before any confirmation collection. Pilot activations excluded from all selection/fitting.',
        numerical=dict(reference_relative_L2_max=.02,restore_bitwise=True,original_BF16_product_bitwise=True),
        retention=dict(review_date='2026-12-13',purpose='Focused native response, state sufficiency and causal-path audit'),
        limits=['One checkpoint; no training-time formation claim.','New sequences absent from fitting but present in prospective behavior calibration.',
                'Format panel reuses confirmation families, not independent N.','No automatic global conclusion if a subset passes.']))


def admit():
    plan=B.read(B.OUT/'trial-plan.json');assert B.file_sha(B.OUT/'materials.json')==plan['materials_sha256']
    assert B.file_sha(B.OUT/'scope-plan.json')==plan['scope_plan_sha256']
    for name,sha in plan['source_hashes'].items():assert B.file_sha(B.ROOT/'tests/glm5'/name)==sha,name
    for name,version in plan['runtime'].items():assert importlib.metadata.version(name)==version
    return plan


def behavior(rows,scores,pilot=False):
    fields=['branch','depth','mode']+([] if rows[0]['split']=='format' else ['title'])
    cells=[]
    for key in sorted({tuple(r[k] for k in fields) for r in rows}):
        ix=[i for i,r in enumerate(rows) if tuple(r[k] for k in fields)==key]
        correct=sum(scores[i][rows[i]['gold']]>max(v for j,v in enumerate(scores[i]) if j!=rows[i]['gold']) for i in ix)
        cells.append(dict(zip(fields,key),n=len(ix),correct=correct,accuracy=correct/len(ix),passed=correct/len(ix)>=(.85 if pilot else .9)))
    overall=sum(s[r['gold']]>max(v for j,v in enumerate(s) if j!=r['gold']) for r,s in zip(rows,scores))/len(rows)
    return dict(cells=cells,overall_accuracy=overall,passed=all(c['passed'] for c in cells) and overall>=.9)


def collect(split):
    B.configure();plan=admit();R=B.bind()
    if split!='pilot':
        for c in B.read(B.OUT/'pilot/collection.json')['behavior']['cells']:
            if [c['branch'],c['depth'],c['mode']] in plan['eligible_strata']:assert c['passed']
    if split in ['confirm','format','order']:
        from b36_composition14_tail import admit as admit_execution
        admit_execution()
    allrows=B.read(B.OUT/'materials.json')['rows'];indices=[i for i,r in enumerate(allrows) if r['split']==split and (split=='pilot' or [r['branch'],r['depth'],r['mode']] in plan['eligible_strata'])];rows=[allrows[i] for i in indices]
    assert rows,'No native collection admitted for this split'
    n=len(rows);dest=B.OUT/split;B.check_write(dest);dest.mkdir(exist_ok=True);assert not (dest/'collection.json').exists()
    start=time.monotonic();chunks=[]
    with B.GpuLease():
      import torch
      from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
      with torch.inference_mode():
        cfg,pos,pe,causal=R.context()
        for lo in range(0,n,B.CHUNK):
          rr=rows[lo:lo+B.CHUNK];size=len(rr);folder=dest/f'chunk-{lo:05d}';folder.mkdir(exist_ok=True)
          if (folder/'chunk.json').exists():
            saved=B.read(folder/'chunk.json');assert saved['row_ids']==[r['id'] for r in rr]
            for item in saved['files']:assert B.file_sha(folder/item['path'])==item['sha256']
            chunks.append(saved);continue
          B.check_write(folder,180*2**20,operation='Focused native chunk plus tail output staging')
          B.log('collect_'+split,lo*40,n*40,start,chunk=lo)
          embedding=R.C.weight('model.embed_tokens.weight').cuda();ids=torch.from_numpy(R.padded(rr)).cuda()
          stage=torch.nn.functional.embedding(ids,embedding);hidden=np.empty((size,41,2,B.DIM),np.uint16);hidden[:,0]=R.bits(R.pick(stage,rr))
          pooled=np.stack([stage[i,:r['content_end']+1].float().mean(0).cpu().numpy() for i,r in enumerate(rr)])
          del embedding,ids;torch.cuda.empty_cache();mask=R.mask_for(rr,causal);observed={};trajectory=None;prefix37=None
          for block in range(40):
            layer=R.C.load_block(block,cfg);cap={};hooks=[]
            needed=block>=36 or (split=='pilot' and lo==0 and block in [0,19,26])
            if needed:
              hooks=[layer.mlp.gate_proj.register_forward_hook(lambda m,a,y:cap.update(G=y)),
                  layer.mlp.up_proj.register_forward_hook(lambda m,a,y:cap.update(V=y)),
                  layer.mlp.down_proj.register_forward_pre_hook(lambda m,a:cap.update(A=a[0])),
                  layer.self_attn.o_proj.register_forward_hook(lambda m,a,y:cap.update(attention=y)),
                  layer.mlp.down_proj.register_forward_hook(lambda m,a,y:cap.update(mlp=y)),
                  layer.post_attention_layernorm.register_forward_pre_hook(lambda m,a:cap.update(mlp_residual=a[0])),
                  layer.post_attention_layernorm.register_forward_hook(lambda m,a,y:cap.update(mlp_input=y))]
              arrays={f:np.empty((size,B.WIDTH if block in [36,39] else len(B.tracked())),np.uint16) for f in 'GVA'}
              arrays.update({f+'_content':np.empty((size,len(B.tracked())),np.uint16) for f in 'GVA'})
              arrays.update({f:np.empty((size,2,B.DIM),np.uint16) for f in ['attention','mlp']})
            if block==36:trajectory=np.empty((size,B.LENGTH,len(B.group()),3),np.uint16)
            for i in range(0,size,B.BATCH):
              local=rr[i:i+B.BATCH];batch=stage[i:i+len(local)]
              stage[i:i+len(local)]=layer(batch,attention_mask=mask[i:i+len(local)],position_ids=pos,position_embeddings=pe,use_cache=False)
              if needed:
                assert torch.equal(cap['A'],torch.nn.functional.silu(cap['G'])*cap['V'])
                for key in 'GVA':
                    picked=R.pick(cap[key],local)
                    arrays[key][i:i+len(local)]=R.bits(picked[:,1] if block in [36,39] else picked[:,1,B.tracked()])
                    arrays[key+'_content'][i:i+len(local)]=R.bits(picked[:,0,B.tracked()])
                    if split=='pilot' and lo==0 and i==0 and block in [0,19,26,39]:observed[block,key]=R.bits(picked).copy()
                for key in ['attention','mlp']:arrays[key][i:i+len(local)]=R.bits(R.pick(cap[key],local))
                if block==36:
                    trajectory[i:i+len(local)]=R.bits(torch.stack([cap[f][:,:,B.group()] for f in 'GVA'],-1))
                    for j,r in enumerate(local):trajectory[i+j,r['answer_pos']+1:]=0
                cap.clear()
            hidden[:,block+1]=R.bits(R.pick(stage,rr))
            if block==36:prefix37=np.stack([stage[i,:r['answer_pos']].float().mean(0).cpu().numpy() for i,r in enumerate(rr)])
            for h in hooks:h.remove()
            if needed:
              if block>=36:B.arrays(folder/f'B{block}.npz',arrays)
              del arrays
            del layer,cap;gc.collect();torch.cuda.empty_cache()
            if block%5==4 or block>=36:B.log('collect_'+split,lo*40+size*(block+1),n*40,start,block=block,chunk=lo)
          norm=Qwen3RMSNorm(B.DIM,eps=cfg.rms_norm_eps).cuda();norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False)
          raw=R.pick(stage,rr);final=norm(raw);finalbits=R.bits(final)
          del stage,mask;torch.cuda.empty_cache();head=R.C.weight('lm_head.weight').cuda();scores=[];top1=[]
          for i in range(0,size,B.BATCH):
            logits=torch.nn.functional.linear(final[i:i+B.BATCH,1],head).float()
            for j,r in enumerate(rr[i:i+B.BATCH]):scores.append(logits[j,r['choice_ids']].cpu().tolist());top1.append(int(logits[j].argmax()))
          del logits,head,norm,raw,final;gc.collect();torch.cuda.empty_cache()
          B.arrays(folder/'states.npz',dict(hidden=hidden,final_hidden=finalbits,prefix_embedding=pooled,prefix_h37=prefix37,B36_group_trajectory=trajectory))
          reference=[]
          if split=='pilot' and lo==0:
            observed['final']=finalbits[:4];reference=R.reference(rr[:4],observed);assert all(r['relative_L2']<=.02 for r in reference)
          files=[dict(path=p.name,bytes=p.stat().st_size,sha256=B.file_sha(p)) for p in sorted(folder.glob('*.npz'))]
          info=dict(row_indices=indices[lo:lo+size],row_ids=[r['id'] for r in rr],scores=scores,top1=top1,files=files,numeric_reference=reference,
                    trial_plan_sha256=B.file_sha(B.OUT/'trial-plan.json'))
          B.save(f'{split}/{folder.name}/chunk.json',info);chunks.append(info)
          del hidden,finalbits,pooled,prefix37,trajectory,observed;gc.collect()
        scores=sum((c['scores'] for c in chunks),[]);gate=behavior(rows,scores,split=='pilot')
        B.save(f'{split}/collection.json',dict(kind='real',row_indices=indices,row_ids=[r['id'] for r in rows],scores=scores,behavior=gate,
            chunks=[dict(name=f'chunk-{lo:05d}',sha256=B.file_sha(dest/f'chunk-{lo:05d}'/'chunk.json')) for lo in range(0,n,B.CHUNK)],
            native_bytes=sum(x['bytes'] for c in chunks for x in c['files']),seconds=time.monotonic()-start,source_sha256=B.file_sha(__file__)))
        B.log('completed_'+split,n,n,start,behavior=gate)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','collect']);p.add_argument('--split',default='pilot',choices=['pilot','discovery','confirm','format','order']);a=p.parse_args()
    freeze() if a.stage=='freeze' else collect(a.split)
