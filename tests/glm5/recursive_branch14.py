"""Frozen prospective branch study of two *14B* native candidates.

Layer-streamed, original BF16 Qwen3 blocks on CUDA; CPU-initialized RoPE.
Every stored G/V/A value is the native value, not a float32 reconstruction.
"""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
from deploy.storage_policy import check_write
from shared.dist_protocol.mechanism_trace import GpuLease, file_sha, digest, save_json, unbits
from source_binding_resources import resources
import qwen14_mlp_atlas as C
from recursive_branch14_materials import OUT, SEED

TEMP=ROOT/'tests/glm5_temp/recursive_branch14_20261113_v6'
MODEL=ROOT/'models/hf/Qwen3-14B'
SNAPSHOT='1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce'
RUN='RECURSIVE-BRANCH14-20261113-v6'
DIM=5120; WIDTH=17408; DEPTH=40; LENGTH=192; BATCH=4; CHUNK=128
CANDIDATES={'structure':(26,13099),'procedure':(19,64)}
SPLITS=['discovery','new_words','new_frames','new_domains']


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def configure():
    for p in (OUT,TEMP):check_write(p);p.mkdir(parents=True,exist_ok=True)
    for k in ['TEMP','TMP','TMPDIR','HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR']:
        os.environ[k]=str(TEMP)
    os.environ.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',GIT_CONFIG_COUNT='1',
        GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0=str(ROOT))
    C.MODEL=MODEL


def save(name,d):
    from tools import provenance
    path=check_write(OUT/name);provenance.RESULTS=OUT/'provenance';check_write(provenance.RESULTS)
    helper=provenance.write_result('recursive_branch14_'+str(name).replace('/','_').replace('\\','_'),
        dict(run_id=RUN,payload_sha256=digest(d),result_path=str(path)),model_path=MODEL,seed=SEED,tool='recursive_branch14_native')
    prov=read(helper)['provenance'];assert not provenance.check_provenance({'provenance':prov})
    save_json(path,dict(**d,run_id=RUN,model_snapshot=SNAPSHOT,provenance=prov,
        provenance_helper=str(helper.relative_to(ROOT))))


def log(stage,done=0,total=0,start=None,**kw):
    elapsed=time.monotonic()-start if start else 0
    r=dict(stage=stage,done=done,total=total,seconds=round(elapsed,2),
           eta_seconds=round(elapsed/done*(total-done),1) if done else None,**resources(),**kw)
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        t=sys.modules['torch'];r.update(cuda_allocated=t.cuda.memory_allocated(),cuda_peak=t.cuda.max_memory_allocated())
    print(json.dumps(r,ensure_ascii=False),flush=True)
    with (OUT/'progress.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    assert r.get('shared_gpu_bytes') is not None,'Required shared GPU telemetry unavailable'
    assert r['shared_gpu_bytes']<2**30,'Shared GPU residency above frozen 1 GiB limit'
    check_write(OUT)


def materials():return read(OUT/'materials.json')['rows']


def source_hashes():
    names=['recursive_branch14.py','recursive_branch14_materials.py','recursive_branch14_analysis.py',
           'recursive_branch14_intervene.py','recursive_branch14_revise.py','recursive_branch14_neutral.py',
           'recursive_branch14_diagnose.py','qwen14_mlp_atlas.py','source_binding_resources.py']
    return {str((Path(__file__).parent/n).relative_to(ROOT)):file_sha(Path(__file__).parent/n) for n in names}


def power(d,n,alpha):
    from scipy.stats import norm
    z=norm.ppf(1-alpha/2);need=int(np.ceil(((z+norm.ppf(.8))/d)**2))
    return dict(paired_standardized_effect=d,n_families=n,alpha=alpha,n_required=need,
                approximate_power=float(norm.cdf(d*np.sqrt(n)-z)))


def freeze():
    assert not (OUT/'plan.json').exists(),'Immutable plan exists'
    rows=materials();assert len(rows)==6144
    # Check current bytes, not just a remembered snapshot string.
    from shared.dist_protocol.natural_register_runner import snapshot, PATTERNS
    start=time.monotonic();log('model_identity',start=start)
    snap=snapshot(MODEL);assert snap==SNAPSHOT
    identity=[]
    for p in sorted({p for pat in PATTERNS for p in MODEL.glob(pat) if p.is_file()}):
        identity.append(dict(path=p.name,bytes=p.stat().st_size,sha256=file_sha(p)))
    save('model-identity.json',dict(files=identity,config=read(MODEL/'config.json')))
    # 57.4 GiB native fields plus <=6 GiB targeted replay input, <=2 GiB
    # analysis products, and staging. All independent tasks must fit separately.
    check_write(OUT,72*2**30,operation='Qwen14 branch campaign total new peak')
    prim=power(.35,128,.05/6);inter=power(.35,192,.05/4)
    from tools.power_analysis import check_phase
    binary=check_phase(.08,192,alpha=.05/4)
    assert binary['pass'],'Paired binary planning gate failed'
    assert prim['approximate_power']>=.8 and inter['approximate_power']>=.8
    specs={h:dict(block=b,unit=u,field='G',expected_direction='positive',
        claim=('Descriptions of explicit source/counterpart relations' if h=='structure' else
               'Descriptions of executable ordering transformations'),
        primary='mean(operative,quoted)-inventory at the content-end common token',
        secondary='instruction-reference wording response; does not test whether a rejected rule applies',
        alternatives=['technical title','word identity and repeated phrases','domain/style','length/position','general instruction processing'])
        for h,(b,u) in CANDIDATES.items()}
    # Select intervention material without looking at model output; include all
    # six correlated views, never count them as independent families.
    intervention={}
    for h in CANDIDATES:
        own=[];other=[]
        for split in SPLITS[1:]:
            for domain in sorted({r['domain'] for r in rows if r['split']==split and r['hypothesis']==h}):
                fam=sorted({r['family'] for r in rows if r['split']==split and r['domain']==domain and r['hypothesis']==h},key=digest)
                own.extend(fam[:32])
            for domain in sorted({r['domain'] for r in rows if r['split']==split and r['hypothesis']!=h}):
                fam=sorted({r['family'] for r in rows if r['split']==split and r['domain']==domain and r['hypothesis']!=h},key=digest)
                other.extend(fam[:8])
        intervention[h]=dict(target_families=own,other_task_families=other,
            row_indices=[i for i,r in enumerate(rows) if r['family'] in own+other])
        assert len(own)==192 and len(other)==48
    plan=dict(schema='recursive_branch14_plan.v6',kind='prospective_preregistration',
        created_local=time.strftime('%Y-%m-%d %H:%M'),material_sha256=file_sha(OUT/'materials.json'),
        material_audit_sha256=file_sha(OUT/'materials-audit.json'),source_hashes=source_hashes(),
        runtime={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','safetensors','scipy','scikit-learn']},
        model_name='Qwen3-14B',precision='BF16',rotary='CPU float32 initialized, transferred to CUDA',
        batch=BATCH,padded_length=LENGTH,chunk=CHUNK,model_streaming='one original decoder block at a time',
        hypotheses=specs,power=dict(primary=prim,intervention=inter,binary_operational=binary,
            binary_scope='One operative neutral-title variant per target family; 8-point paired accuracy change, planned discordance .10; not powered for every small effect or other-task subset'),
        pilot='Eight discovery families per hypothesis/domain/wording; all 6 conditions = 384 rows; 16 families per hypothesis/frame/role/title cell',
        behavior_gate=dict(min_cell_acc=.8,pilot_cells='hypothesis/frame/role/title; small discovery pilot is descriptive; detailed domain cells also retained',
            confirm_cells='hypothesis/split/role/title; every cell >= .8; keep failures, do not retune'),
        extraction='hs[0] plus all 40 residual outputs; G/V/A every unit of all 40 blocks at content end and answer; candidate G/V/A also every valid token',
        readouts=['content_end_before_probe','answer_prompt_end'],
        primary_decision='positive G relation effect at content readout; paired family t p<.05/6 and dz>=.35 in EACH of three held-out partitions, and instruction/reference each have positive mean G versus inventory; technical title separately reported; role-adjusted model must beat lexical baseline MSE by 10% in all partitions',
        predictions=dict(features='prefix only; 512-d fixed signed hashed token uni/bigrams + length vs semantic role + title + domain + operation; and combination; frozen ridge=1',
            input_warning='Semantic role is experimenter-defined available stimulus metadata, not a discovered parser or an autonomous mechanism.',
            fit='discovery only; no post-holdout tuning',joint='direct G/V/A and composed SiLU(predG)*predV compared',
            embedding_baseline='same current token at content end => exactly equal hs[0]; pooled prefix embedding ridge also compared'),
        group_selection='at each original candidate layer: original unit plus seven highest positive discovery G relation effects after residualizing length/title; random same-layer units matched by discovery std(A)*norm(down), disjoint; all selections frozen before any heldout response',
        backward=dict(root_block=39,root_units_per_hypothesis=8,boundaries=[(38,39),(37,38),(36,37)],parents=32,
            selection='discovery-only positive G relation roots; A-parent group selected by joint correlation to G/V/A children; ridge direct G/V/A and composed A; random matched parents',
            success='all 3 holdouts: MSE <=.9 mean and <=.9 lexical baseline and <=.9 random-parent model, also free B36-to-B39 rollout retains advantage',
            scope='pilot of predictive response graph, not training-time recursive growth; no graph expansion before this gate'),
        intervention=dict(material=intervention,doses=[0.,2.],unit_group_size=8,
            operation='scale actual down_proj input selected units in original forward, all rule-body positions through content-end; titles, question and options untouched',
            arms=['baseline','restore','single_0','single_2','group_0','group_2','single_norm_control','group_norm_control','matched_group_0','matched_group_2'],
            norm_control='same realized local BF16 write-norm per token; random frozen direction; report <=2% mismatch qualification and all failures',
            primary='paired family change of correct-choice logit minus strongest competing choice at dose0, single and group, each hypothesis; 4 tests alpha .05/4; pointwise other doses/exploratory effects labelled',
            causal_specificity='compare operative vs quoted/inventory, target vs other hypothesis, matched random units and norm direction; confidence intervals by family',
            restoration='dose=1 unit group must reproduce unedited final hidden bitwise; original collection choice and final hidden numerical check',
            scope='original-unit intervention at selected context positions; not scalar-parameter editing, no proof of unique/complete mechanism'),
        gates=dict(full_HF_relative_L2_max=.02,replay_KL_max=.001,native_product_bitwise=True,restore_bitwise=True),
        resource=dict(peak_new_GiB=72,reserve_GiB=20,shared_GPU_GiB_max=1,collection_minutes_max=180,intervention_minutes_max=180),
        stopping='Failed behavior cells excluded from mechanism-positive claims. Failed response prediction still permits frozen interventions as negative/falsification tests. Failed numerical/resource gate stops affected stage. No posthoc threshold relaxation.',
        material_provenance='Constructed original language stimuli. v5 discovery-only applicability failures and diagnostics selected this narrower neutral-reference protocol; all family partitions retained, heldout model responses never read before group freeze. No globally independent human lockbox.',
        limitations=['Only two of six previous candidates belong to Qwen14; four 4B IDs not mapped by index.',
            'Fixed-template rule descriptions operationalize a narrow slice of earlier broad natural-language hypotheses; applicability reasoning failed in v5 and is outside this protocol.',
            'New-frame and new-domain partitions share a held-out word pool; not independent replications.',
            'Family tests do not remove all reused-word/template dependence. Only two held-out domains per hypothesis.',
            'Final static checkpoint cannot establish training-time recursive feature growth.',
            'No matched SAE/transcoder or free natural generation in this campaign; native mechanism remains partial.'],
        retention=dict(purpose='Reuse all-layer native field and frozen holdout evidence',review_date='2026-12-13'))
    save('plan.json',plan);log('freeze_complete',1,1,start)


def admission():
    p=read(OUT/'plan.json')
    assert p['model_snapshot']==SNAPSHOT
    assert file_sha(OUT/'materials.json')==p['material_sha256']
    for name,h in p['source_hashes'].items():assert file_sha(ROOT/name)==h,name
    check_write(OUT)
    return p


def context():
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    torch.set_num_threads(4)
    cfg=AutoConfig.from_pretrained(MODEL,local_files_only=True);cfg._attn_implementation='sdpa'
    pos=torch.arange(LENGTH,device='cuda')[None]
    rotary=Qwen3RotaryEmbedding(cfg,device='cpu').cuda()
    pe=rotary(torch.zeros((1,LENGTH,DIM),device='cuda',dtype=torch.bfloat16),pos)
    causal=torch.arange(LENGTH,device='cuda')[None,:]<=torch.arange(LENGTH,device='cuda')[:,None]
    return cfg,pos,pe,causal


def padded(rows):
    ids=np.full((len(rows),LENGTH),151643,dtype=np.int64)
    for i,r in enumerate(rows):ids[i,:len(r['tokens'])]=r['tokens']
    return ids


def mask_for(rows,causal):
    import torch
    valid=torch.arange(LENGTH,device='cuda')[None,:]<torch.tensor([len(r['tokens']) for r in rows],device='cuda')[:,None]
    return torch.zeros((len(rows),1,LENGTH,LENGTH),device='cuda',dtype=torch.bfloat16).masked_fill(~(causal[None,None]&valid[:,None,None,:]),float('-inf'))


def pick(x,rows):
    import torch
    at=torch.tensor([[r['content_end'],r['answer_pos']] for r in rows],device='cuda')
    return x[torch.arange(len(rows),device='cuda')[:,None],at]


def bits(x):return x.contiguous().view(sys.modules['torch'].uint16).cpu().numpy()


def pilot_rows(rows):
    chosen=[]
    for h in CANDIDATES:
      for domain in sorted({r['domain'] for r in rows if r['hypothesis']==h and r['split']=='discovery'}):
       for frame in (0,1):
        fam=sorted({r['family'] for r in rows if r['domain']==domain and r['frame']==frame and r['split']=='discovery' and r['hypothesis']==h},key=digest)
        chosen+=fam[:8]
    return [i for i,r in enumerate(rows) if r['family'] in chosen]


def behavior(rows,scores,by_domain=False,by_frame=False):
    keys=['hypothesis','split','role','lexical']+(['domain','frame'] if by_domain else ['frame'] if by_frame else [])
    result=[]
    for key in sorted({tuple(r[k] for k in keys) for r in rows}):
        ix=[i for i,r in enumerate(rows) if tuple(r[k] for k in keys)==key]
        correct=[int(np.argmax(scores[i]))==rows[i]['gold'] for i in ix]
        result.append(dict(zip(keys,key),n=len(ix),correct=sum(correct),accuracy=float(np.mean(correct)),passed=float(np.mean(correct))>=.8))
    return result


def reference(rows,observed):
    """Independent full HF orchestration, same original weights/shape/mask."""
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    cfg,pos,pe,causal=context();capture={};hooks=[]
    with torch.device('meta'):
        model=AutoModelForCausalLM.from_config(cfg,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    model.model.embed_tokens.weight=torch.nn.Parameter(C.weight('model.embed_tokens.weight').cuda(),requires_grad=False)
    model.model.norm.weight=torch.nn.Parameter(C.weight('model.norm.weight').cuda(),requires_grad=False)
    model.model.rotary_emb=Qwen3RotaryEmbedding(cfg,device='cpu').cuda()
    for b,layer in enumerate(model.model.layers):
        def load(m,a,b=b):m.load_state_dict(C.load_block(b,cfg).state_dict(),assign=True,strict=True)
        def release(m,a,y):m.to_empty(device='meta')
        hooks.extend([layer.register_forward_pre_hook(load),layer.register_forward_hook(release)])
        if b in [0,19,26,39]:
            hooks.extend([layer.mlp.gate_proj.register_forward_hook(lambda m,a,y,b=b:capture.update({(b,'G'):pick(y,rows).clone()})),
                          layer.mlp.up_proj.register_forward_hook(lambda m,a,y,b=b:capture.update({(b,'V'):pick(y,rows).clone()})),
                          layer.mlp.down_proj.register_forward_pre_hook(lambda m,a,b=b:capture.update({(b,'A'):pick(a[0],rows).clone()}))])
    out=model.model(input_ids=torch.from_numpy(padded(rows)).cuda(),attention_mask=mask_for(rows,causal),position_ids=pos,use_cache=False)
    checks=[]
    for key,value in capture.items():
        a=bits(value);b=observed[key]
        rel=float(np.linalg.norm(unbits(a)-unbits(b))/max(np.linalg.norm(unbits(b)),1e-20))
        checks.append(dict(block=key[0],field=key[1],relative_L2=rel,bitwise=bool(np.array_equal(a,b))))
    a=bits(pick(out.last_hidden_state,rows));b=observed['final']
    checks.append(dict(boundary='final_norm',relative_L2=float(np.linalg.norm(unbits(a)-unbits(b))/max(np.linalg.norm(unbits(b)),1e-20)),bitwise=bool(np.array_equal(a,b))))
    for h in hooks:h.remove()
    del model,out,capture;gc.collect();torch.cuda.empty_cache()
    return checks


def collect(split):
    # Acquire before creating any raw field: a busy GPU never leaves misleading
    # all-zero files that resemble a partially completed acquisition.
    with GpuLease():return _collect(split)


def _collect(split):
    import torch
    p=admission();allrows=materials();pilot=split=='pilot'
    if pilot:indices=pilot_rows(allrows)
    else:
        assert read(OUT/'pilot.json')['numerical_passed'],'Run numerical pilot first'
        assert read(OUT/'pilot.json')['behavior_passed'],'Pilot behavior failed; affected material must not enter internal analysis'
        if split!='discovery':assert (OUT/'selection.json').exists(),'Freeze groups before heldout responses'
        indices=[i for i,r in enumerate(allrows) if r['split']==split]
    rows=[allrows[i] for i in indices];n=len(rows);dest=OUT/split
    assert all(len(r['tokens'])<=LENGTH for r in rows)
    check_write(dest,n*(DEPTH*2*WIDTH*3*2+(DEPTH+1)*2*DIM*2)+8*2**30)
    assert not (dest/'collection.json').exists(),'Collection already complete'
    dest.mkdir(exist_ok=True)
    assert not (dest/'B0-G.npy').exists(),'Partial fields retained; use a new run for restart'
    fields={}
    for b in range(DEPTH):
        for f in ['G','V','A']:fields[b,f]=np.lib.format.open_memmap(dest/f'B{b}-{f}.npy',mode='w+',dtype=np.uint16,shape=(n,2,WIDTH))
    hs=np.lib.format.open_memmap(dest/'hidden-all.npy',mode='w+',dtype=np.uint16,shape=(n,DEPTH+1,2,DIM))
    final=np.lib.format.open_memmap(dest/'final-hidden.npy',mode='w+',dtype=np.uint16,shape=(n,2,DIM))
    trajectories={h:np.lib.format.open_memmap(dest/f'{h}-all-token-GVA.npy',mode='w+',dtype=np.uint16,shape=(n,LENGTH,3)) for h in CANDIDATES}
    # Save only predeclared intervention rows, full prefix at candidate entrance.
    entrances={};entrance_map={}
    if not pilot:
        for h,(block,unit) in CANDIDATES.items():
            selected=p['intervention']['material'][h]['row_indices']
            ix=[i for i,g in enumerate(indices) if g in selected]
            if ix:
                entrances[h]=np.lib.format.open_memmap(dest/f'{h}-replay-input.npy',mode='w+',dtype=np.uint16,shape=(len(ix),LENGTH,DIM))
                entrance_map[h]={local:j for j,local in enumerate(ix)}
    start=time.monotonic();replay=[];times=[];scores=np.zeros((n,3),np.float32);top=[];observed={}
    log('collection_start_'+split,0,n*DEPTH,start)
    with torch.inference_mode():
      cfg,pos,pe,causal=context()
      for lo in range(0,n,CHUNK):
        chunk=rows[lo:lo+CHUNK];size=len(chunk)
        stage=torch.empty((size,LENGTH,DIM),device='cuda',dtype=torch.bfloat16)
        emb=C.weight('model.embed_tokens.weight').cuda()
        ids=torch.from_numpy(padded(chunk)).cuda()
        for i in range(0,size,BATCH):stage[i:i+BATCH]=torch.nn.functional.embedding(ids[i:i+BATCH],emb)
        # Pooled full-prefix embedding baseline: only tokens up through content.
        if lo==0:pool=np.lib.format.open_memmap(dest/'prefix-embedding.npy',mode='w+',dtype=np.float32,shape=(n,DIM))
        for i,r in enumerate(chunk):pool[lo+i]=stage[i,:r['content_end']+1].float().mean(0).cpu().numpy()
        hs[lo:lo+size,0]=bits(pick(stage,chunk));del emb;torch.cuda.empty_cache()
        mask=mask_for(chunk,causal)
        for b in range(DEPTH):
          t=time.monotonic();layer=C.load_block(b,cfg);capture={}
          hooks=[layer.mlp.gate_proj.register_forward_hook(lambda m,a,y:capture.update(G=y)),
                 layer.mlp.up_proj.register_forward_hook(lambda m,a,y:capture.update(V=y)),
                 layer.mlp.down_proj.register_forward_pre_hook(lambda m,a:capture.update(A=a[0]))]
          for h,(block,unit) in CANDIDATES.items():
            if b==block and h in entrances:
                for j in range(size):
                    if lo+j in entrance_map[h]:entrances[h][entrance_map[h][lo+j]]=bits(stage[j])
          for i in range(0,size,BATCH):
            rr=chunk[i:i+BATCH];y=layer(stage[i:i+BATCH],attention_mask=mask[i:i+BATCH],position_ids=pos,position_embeddings=pe,use_cache=False)
            stage[i:i+BATCH]=y;hs[lo+i:lo+i+len(rr),b+1]=bits(pick(y,rr))
            for f in ['G','V','A']:
                got=bits(pick(capture[f],rr));fields[b,f][lo+i:lo+i+len(rr)]=got
                if pilot and lo==0 and i==0 and b in [0,19,26,39]:observed[b,f]=got.copy()
            for h,(block,unit) in CANDIDATES.items():
                if b==block:
                    z=torch.stack([capture[f][:,:,unit] for f in ['G','V','A']],-1)
                    trajectories[h][lo+i:lo+i+len(rr)]=bits(z)
            if i==0:
                equal=torch.equal(torch.nn.functional.silu(capture['G'])*capture['V'],capture['A'])
                replay.append(dict(chunk=lo,block=b,native_product_bitwise=equal));assert equal
            capture.clear();del y
          for hook in hooks:hook.remove()
          del layer;gc.collect();torch.cuda.empty_cache()
          times.append(dict(block=b,chunk=lo,seconds=time.monotonic()-t))
          log('collect_'+split,lo*DEPTH+(b+1)*size,n*DEPTH,start,block=b,chunk=lo)
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        norm=Qwen3RMSNorm(DIM,eps=cfg.rms_norm_eps).cuda();norm.weight=torch.nn.Parameter(C.weight('model.norm.weight').cuda(),requires_grad=False)
        v=norm(pick(stage,chunk));final[lo:lo+size]=bits(v)
        if pilot and lo==0:observed['final']=bits(v[:BATCH])
        wu=C.weight('lm_head.weight').cuda()
        for i in range(0,size,BATCH):
            logits=torch.nn.functional.linear(v[i:i+BATCH,1],wu).float()
            for j,r in enumerate(chunk[i:i+BATCH]):
                scores[lo+i+j]=logits[j,r['choice_ids']].cpu().numpy();top.append(int(logits[j].argmax()))
        for a in list(fields.values())+[hs,final,pool]+list(trajectories.values())+list(entrances.values()):a.flush()
        del stage,ids,mask,norm,v,wu,logits;gc.collect();torch.cuda.empty_cache()
      checks=reference(rows[:BATCH],observed) if pilot else []
    cells=behavior(rows,scores,by_frame=pilot);passed=all(c['passed'] for c in cells)
    native=dict(kind='real',row_indices=indices,row_ids=[r['id'] for r in rows],rows=n,seconds=time.monotonic()-start,
        plan_sha256=file_sha(OUT/'plan.json'),source_hashes=source_hashes(),timings=times,native_replay=replay,
        behavior_cells=cells,behavior_passed=passed,scores=scores.tolist(),top1=top,
        entrance_local_indices={h:list(mapping) for h,mapping in entrance_map.items()},
        field_layout='rows, [content_end, answer_pos], native coordinate; uint16 BF16 bit patterns',
        all_token_layout=f'rows, padded {LENGTH} positions, [G,V,A]; valid positions given by prompt token length')
    save(f'{split}/collection.json',native)
    if pilot:
        numeric=all(c['relative_L2']<=.02 for c in checks)
        estimate=sum(x['seconds'] for x in times)*6144/n
        save('pilot.json',dict(numerical_passed=numeric,full_HF_checks=checks,behavior_passed=passed,
            cells=cells,detailed_cells=behavior(rows,scores,True),estimated_collection_seconds=estimate,
            note='Only discovery responses read; four/small-cell estimates are not precise accuracy estimates'))
        assert numeric,'Full HF reference numerical gate failed'
        assert estimate<p['resource']['collection_minutes_max']*60,'Collection budget failed'
    log('collection_complete_'+split,n,n,start)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','collect']);ap.add_argument('--split',choices=['pilot']+SPLITS,default='pilot')
    a=ap.parse_args();configure()
    if a.stage=='freeze':freeze()
    else:collect(a.split)


if __name__=='__main__':main()
