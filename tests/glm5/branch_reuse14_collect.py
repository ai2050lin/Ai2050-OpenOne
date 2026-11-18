"""Sequential original-BF16 CUDA collection, losslessly compressed by chunk/block."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import branch_reuse14 as B
import recursive_branch14 as R

LENGTH=256
BATCH=4
CHUNK=64
DEPTH=40
WIDTH=17408
DIM=5120
CANDIDATES={'B19U64':(19,64),'B26U13099':(26,13099)}


def sources():
    names=['branch_reuse14.py','branch_reuse14_materials.py','branch_reuse14_collect.py',
        'recursive_branch14.py','qwen14_mlp_atlas.py','source_binding_resources.py']
    return {str((B.ROOT/'tests/glm5'/name).relative_to(B.ROOT)):B.file_sha(B.ROOT/'tests/glm5'/name) for name in names}


def freeze():
    B.configure(13*2**30);assert not (B.OUT/'trial-plan.json').exists()
    from shared.dist_protocol.natural_register_runner import PATTERNS
    old=B.read(B.PREVIOUS/'model-identity.json');identity=[];start=time.monotonic()
    paths=sorted({p for pattern in PATTERNS for p in B.MODEL.glob(pattern) if p.is_file()})
    assert {p.name for p in paths}=={r['path'] for r in old['files']}
    for i,p in enumerate(paths):
        sha=B.file_sha(p);expected=next(x for x in old['files'] if x['path']==p.name)
        assert sha==expected['sha256'] and p.stat().st_size==expected['bytes']
        identity.append(expected);B.log('trial_model_identity',i+1,len(paths),start,file=p.name)
    B.save('model-identity.json',dict(files=identity,config=old['config'],full_sha_matches_previous=True))
    from scipy.stats import norm
    power=float(norm.cdf(.5*np.sqrt(80)-norm.ppf(1-(.05/6)/2)))
    B.save('trial-plan.json',dict(kind='prospective_pilot_protocol',created_local=time.strftime('%Y-%m-%d %H:%M'),
        materials_sha256=B.file_sha(B.OUT/'materials.json'),source_hashes=sources(),runtime={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','safetensors','scipy','scikit-learn']},
        model='Qwen3-14B',precision='original BF16',length=LENGTH,batch=BATCH,chunk=CHUNK,
        observation='All 40 native G/V/A fields and hs[0..40] at content end before query/options and final answer prompt; two historical candidates also every token; raw BF16 losslessly compressed.',
        behavior=dict(pilot='Each branch/role/title cell has 16 rows; require accuracy >=.85 in every cell and overall >=.9. Only pilot material is opened before fitting.',
            formal='Each branch/role/title cell >=.9; all cells must pass for a global claim; no result-dependent threshold adjustment.'),
        primary_response=dict(candidate='B19/U64',fields=['V','A'],contrast='rule minus finite record, within family and title averaged',
            expected='positive in all three eligible branches',minimum_dz=.5,alpha=.05/6,confirm_families_per_branch=80,approximate_power=power,
            baseline='Relationship metadata must predict beyond pooled prefix embedding/position/title on new material, not just beyond weak hashed words.'),
        additional='B26/U13099 is a prespecified secondary historical candidate. New group and graph selection only uses the 384-row number/coordinate discovery set; label is unseen by predictor fitting. Sequence is ineligible after v2 calibration and option-order bias; all old material is exposed development data.',
        graph='Freeze full method and interventions after discovery, before opening any confirmation activations. Compare direct-A and composed-A recurrence, parameter-informed/correlation/random parent sets, strong residual and embedding baselines; intervention-predicted effects tested separately.',
        acceptance='Positive finite-scope evidence, contradiction within specified effect range, and insufficient evidence are distinct outcomes. No finite dataset decides every possible graph or training-time origin.',
        numeric=dict(full_HF_relative_L2_max=.02,native_product_bitwise=True,replay_KL_max=.001),
        resources=dict(new_peak_GiB=13,reserve_GiB=20,shared_GPU_GiB_max=1,model_serial=True),
        retention=dict(purpose='Prospective branch comparison and reusable native-field audit',review_date='2026-12-13'),
        limits=['Rules/records differ in wording and length; controls are not concept-absent.',
            'The same numerical values can occur across partitions; no unseen-number claim.',
            'Pilot activations are excluded from all fitting/feature selection; number and coordinate use separate discovery families, label is an unseen fitting branch.',
            'No training checkpoints; fixed-snapshot results cannot establish temporal formation of branches.']))


def admit():
    p=B.read(B.OUT/'trial-plan.json');assert p['materials_sha256']==B.file_sha(B.OUT/'materials.json')
    for name,sha in p['source_hashes'].items():assert B.file_sha(B.ROOT/name)==sha,name
    for name,version in p['runtime'].items():assert importlib.metadata.version(name)==version
    return p


def bind():
    R.LENGTH=LENGTH;R.BATCH=BATCH;R.MODEL=B.MODEL;R.C.MODEL=B.MODEL


def write_arrays(path,arrays):
    B.check_write(path)
    if path.exists():
        with np.load(path) as old:
            assert set(old.files)==set(arrays)
            for key,value in arrays.items():assert np.array_equal(old[key],value),(path,key)
        return
    temp=path.with_suffix('.partial')
    assert not temp.exists(),'Incomplete compressed file retained; inspect before recovery'
    with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temp.replace(path)


def behavior(rows,scores,pilot=False):
    cells=[]
    for branch,role,title in sorted({(r['branch'],r['role'],r['title']) for r in rows}):
        ix=[i for i,r in enumerate(rows) if (r['branch'],r['role'],r['title'])==(branch,role,title)]
        correct=sum(int(np.argmax(scores[i]))==rows[i]['gold'] for i in ix)
        cells.append(dict(branch=branch,role=role,title=title,n=len(ix),correct=correct,accuracy=correct/len(ix),passed=correct/len(ix)>=(.85 if pilot else .9)))
    overall=sum(int(np.argmax(s))==r['gold'] for r,s in zip(rows,scores))/len(rows)
    return dict(cells=cells,overall_accuracy=overall,passed=all(c['passed'] for c in cells) and overall>=.9)


def collect(split):
    B.configure();plan=admit();bind()
    if split!='pilot':assert B.read(B.OUT/'pilot/collection.json')['behavior']['passed']
    if split=='confirm':assert (B.OUT/'frozen-selection.json').exists() and (B.OUT/'analysis-plan.json').exists()
    allrows=B.read(B.OUT/'materials.json')['rows'];indices=[i for i,r in enumerate(allrows) if r['split']==split]
    rows=[allrows[i] for i in indices];n=len(rows);dest=B.OUT/split;B.check_write(dest);dest.mkdir(exist_ok=True)
    assert not (dest/'collection.json').exists(),'Completed native collection is immutable'
    start=time.monotonic();chunk_metadata=[]
    with B.GpuLease():
      import torch
      from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
      with torch.inference_mode():
        cfg,pos,pe,causal=R.context()
        for lo in range(0,n,CHUNK):
          rr=rows[lo:lo+CHUNK];size=len(rr);chunk_dir=dest/f'chunk-{lo:05d}';chunk_dir.mkdir(exist_ok=True)
          committed=chunk_dir/'chunk.json'
          if committed.exists():
            info=B.read(committed);assert info['row_ids']==[r['id'] for r in rr]
            assert info['trial_plan_sha256']==B.file_sha(B.OUT/'trial-plan.json')
            for item in info['files']:
                p=chunk_dir/item['path'];assert p.stat().st_size==item['bytes'] and B.file_sha(p)==item['sha256']
            chunk_metadata.append(info);continue
          B.check_write(dest,256*2**20,operation='Compressed native chunk and analysis reserve')
          B.log('native_'+split,lo*DEPTH,n*DEPTH,start,chunk=lo)
          stage=torch.empty((size,LENGTH,DIM),device='cuda',dtype=torch.bfloat16)
          embedding=R.C.weight('model.embed_tokens.weight').cuda();ids=torch.from_numpy(R.padded(rr)).cuda()
          for i in range(0,size,BATCH):stage[i:i+BATCH]=torch.nn.functional.embedding(ids[i:i+BATCH],embedding)
          hidden=np.empty((size,41,2,DIM),np.uint16);hidden[:,0]=R.bits(R.pick(stage,rr))
          pooled=np.stack([stage[i,:r['content_end']+1].float().mean(0).cpu().numpy() for i,r in enumerate(rr)])
          trajectories={name:np.zeros((size,LENGTH,3),np.uint16) for name in CANDIDATES}
          del embedding,ids;torch.cuda.empty_cache();mask=R.mask_for(rr,causal);observed={}
          for block in range(DEPTH):
            layer=R.C.load_block(block,cfg);capture={}
            hooks=[layer.mlp.gate_proj.register_forward_hook(lambda m,a,y:capture.update(G=y)),
                   layer.mlp.up_proj.register_forward_hook(lambda m,a,y:capture.update(V=y)),
                   layer.mlp.down_proj.register_forward_pre_hook(lambda m,a:capture.update(A=a[0]))]
            arrays={f:np.empty((size,2,WIDTH),np.uint16) for f in 'GVA'}
            for i in range(0,size,BATCH):
                local=rr[i:i+BATCH]
                stage[i:i+len(local)]=layer(stage[i:i+len(local)],attention_mask=mask[i:i+len(local)],position_ids=pos,position_embeddings=pe,use_cache=False)
                assert torch.equal(capture['A'],torch.nn.functional.silu(capture['G'])*capture['V'])
                for f in 'GVA':arrays[f][i:i+len(local)]=R.bits(R.pick(capture[f],local))
                for name,(b,u) in CANDIDATES.items():
                    if block==b:trajectories[name][i:i+len(local)]=R.bits(torch.stack([capture[f][:,:,u] for f in 'GVA'],-1))
                capture.clear()
            hidden[:,block+1]=R.bits(R.pick(stage,rr))
            if split=='pilot' and lo==0 and block in [0,19,26,39]:
                for f in 'GVA':observed[block,f]=arrays[f][:4].copy()
            for handle in hooks:handle.remove()
            del layer,capture;gc.collect();torch.cuda.empty_cache()
            write_arrays(chunk_dir/f'B{block}.npz',arrays);del arrays
            B.log('native_'+split,lo*DEPTH+size*(block+1),n*DEPTH,start,block=block,chunk=lo)
          norm=Qwen3RMSNorm(DIM,eps=cfg.rms_norm_eps).cuda();norm.weight=torch.nn.Parameter(R.C.weight('model.norm.weight').cuda(),requires_grad=False)
          final=norm(R.pick(stage,rr));finalbits=R.bits(final);del stage,norm,mask;torch.cuda.empty_cache()
          head=R.C.weight('lm_head.weight').cuda();scores=[];top1=[]
          for i in range(0,size,BATCH):
            logits=torch.nn.functional.linear(final[i:i+BATCH,1],head).float()
            for j,r in enumerate(rr[i:i+BATCH]):scores.append(logits[j,r['choice_ids']].cpu().tolist());top1.append(int(logits[j].argmax()))
            del logits
          del head,final;gc.collect();torch.cuda.empty_cache()
          write_arrays(chunk_dir/'states.npz',dict(hidden=hidden,final_hidden=finalbits,prefix_embedding=pooled,**trajectories))
          reference=[]
          if split=='pilot' and lo==0:
              observed['final']=finalbits[:4].copy();reference=R.reference(rr[:4],observed)
              assert all(x['relative_L2']<=.02 for x in reference),'Independent full-HF numerical gate failed'
          files=[dict(path=p.name,bytes=p.stat().st_size,sha256=B.file_sha(p)) for p in sorted(chunk_dir.glob('*.npz'))]
          info=dict(row_indices=indices[lo:lo+size],row_ids=[r['id'] for r in rr],scores=scores,top1=top1,files=files,
              numeric_reference=reference,trial_plan_sha256=B.file_sha(B.OUT/'trial-plan.json'))
          B.save(f'{split}/{chunk_dir.name}/chunk.json',info);chunk_metadata.append(info)
          del hidden,pooled,trajectories,finalbits,observed;gc.collect()
        scores=sum((c['scores'] for c in chunk_metadata),[])
        gate=behavior(rows,scores,split=='pilot')
        B.save(f'{split}/collection.json',dict(kind='real',row_indices=indices,row_ids=[r['id'] for r in rows],
            scores=scores,behavior=gate,chunks=[dict(name=f'chunk-{lo:05d}',sha256=B.file_sha(dest/f'chunk-{lo:05d}'/'chunk.json')) for lo in range(0,n,CHUNK)],
            native_bytes=sum(x['bytes'] for c in chunk_metadata for x in c['files']),seconds=time.monotonic()-start,
            trial_plan_sha256=B.file_sha(B.OUT/'trial-plan.json'),source_sha256=B.file_sha(__file__)))
        B.log('collection_complete_'+split,n,n,start,behavior=gate)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','collect']);parser.add_argument('--split',choices=['pilot','discovery','confirm'],default='pilot');args=parser.parse_args()
    freeze() if args.stage=='freeze' else collect(args.split)
