"""Prospective, serial Qwen3 native B18--B28 study (inclusive, zero based).

Keeps historical runners and artifacts immutable. All new files stay on the
verified project drive. No model work is admitted before the storage gate.
"""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
import qwen14_mlp_atlas as C
from deploy.storage_policy import check_write
from shared.dist_protocol.mechanism_trace import save_json as raw_save
from source_binding_resources import resources
from tools import provenance

BASE = ROOT / 'tests/glm5/result/goodfire_middle_20261113_v2'
TEMP = ROOT / 'tests/glm5_temp/goodfire_middle_20261113_v2'
HISTORY = ROOT / 'tests/glm5/result/qwen14_mlp_atlas_v1'
BLOCKS = list(range(18, 29))
SEED = 20261113
PEAK_GIB = 34
SPECS = {
    'qwen4': dict(folder='qwen3-4b', name='Qwen3-4B', width=9728, dim=2560, layers=36,
                  snapshot='6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad'),
    'qwen14': dict(folder='Qwen3-14B', name='Qwen3-14B', width=17408, dim=5120, layers=40,
                   snapshot='1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce'),
}
ORIGINAL_WEIGHT = C.weight
MODEL_KEY = None
SPEC = None
OUT = None
RUN = None


def write(path, body):
    path = check_write(path)
    if path.parent != OUT.resolve():
        raise ValueError('Unexpected result destination')
    provenance.RESULTS = BASE / 'provenance'
    check_write(provenance.RESULTS)
    helper = provenance.write_result(MODEL_KEY + '_' + path.stem,
        {'run_id': RUN, 'payload_sha256': C.digest(body), 'result_path': str(path)},
        model_path=C.MODEL, seed=SEED, tool='goodfire_middle_native')
    prov = C.read(helper)['provenance']
    assert not provenance.check_provenance({'provenance': prov})
    raw_save(path, {**body, 'kind': 'real', 'run_id': RUN,
        'model_snapshot': SPEC['snapshot'], 'provenance': prov,
        'provenance_helper': str(helper.relative_to(ROOT))})


def log(stage, done=0, total=0, start=None, **kw):
    elapsed = time.monotonic() - start if start else 0
    if stage == 'profiles':
        done += 10*C.WIDTH; total = len(BLOCKS)*C.WIDTH
    if stage == 'parameter_readout':
        done += 10*C.WIDTH; total = len(BLOCKS)*C.WIDTH
    row = dict(model=MODEL_KEY, stage=stage, done=done, total=total,
        seconds=round(elapsed, 2),
        eta_seconds=round(elapsed/done*(total-done), 1) if done else None,
        **resources(), **kw)
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        t = sys.modules['torch']
        row.update(cuda_allocated=t.cuda.memory_allocated(), cuda_peak=t.cuda.max_memory_allocated())
    check_write(OUT)
    if row.get('shared_gpu_bytes') is not None:
        assert row['shared_gpu_bytes'] < 2**30, 'Shared GPU spill exceeds 1 GiB'
    print(json.dumps(row, ensure_ascii=False), flush=True)


def weight(key):
    if key == 'lm_head.weight' and SPEC['layers'] == 36:
        key = 'model.embed_tokens.weight'  # Qwen3-4B ties input/output weights.
    return ORIGINAL_WEIGHT(key)


def context():
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    torch.set_num_threads(4)
    config=AutoConfig.from_pretrained(C.MODEL,local_files_only=True);config._attn_implementation='sdpa'
    rotary=Qwen3RotaryEmbedding(config,device='cpu').cuda()
    pos=torch.arange(C.LENGTH,device='cuda')[None]
    x=torch.zeros((1,C.LENGTH,C.DIM),device='cuda',dtype=torch.bfloat16)
    pe=rotary(x,pos)
    mask=torch.full((C.LENGTH,C.LENGTH),float('-inf'),device='cuda',dtype=torch.bfloat16).triu(1)[None,None]
    return config,pos,pe,mask


def configure(key):
    global MODEL_KEY, SPEC, OUT, RUN
    MODEL_KEY = key; SPEC = SPECS[key]
    OUT = BASE/key; RUN = f'{key.upper()}-GOODFIRE-B18-B28-20261113-v2'
    for p in [BASE, OUT, TEMP]:
        check_write(p); p.mkdir(parents=True, exist_ok=True)
    for k in ['TEMP','TMP','TMPDIR','HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE',
              'TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR']:
        os.environ[k] = str(TEMP)
    os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', GIT_CONFIG_COUNT='1',
        GIT_CONFIG_KEY_0='safe.directory', GIT_CONFIG_VALUE_0=str(ROOT))
    C.MODEL = ROOT/'models/hf'/SPEC['folder']; C.OUT = OUT; C.STORE = OUT
    C.RUN = RUN; C.WIDTH = SPEC['width']; C.DIM = SPEC['dim']; C.BLOCKS = BLOCKS
    C.save_json = write; C.log = log; C.weight = weight; C.context=context
    return SPEC


def source_files():
    names = ['goodfire_middle.py', 'goodfire_middle_analysis.py', 'goodfire_middle_intervene.py', 'goodfire_middle_audit.py', 'qwen14_mlp_atlas.py',
             'qwen14_mlp_analysis.py', 'qwen14_goodfire_analysis.py',
             'qwen14_goodfire_scaled_prediction.py', 'qwen14_mlp_intervene.py',
             'source_binding_resources.py']
    return {str((Path(__file__).parent/n).relative_to(ROOT)):C.file_sha(Path(__file__).parent/n)
            for n in names}


def prepare():
    """Read-only model hashing and small plans are allowed while storage is blocked."""
    from shared.dist_protocol.natural_register_runner import snapshot, PATTERNS
    from tools.power_analysis import check_phase
    assert not (OUT/'plan.json').exists(), 'Frozen plan exists'
    check_write(BASE, 200*2**20)
    start = time.monotonic(); log('preflight', start=start)
    snap = snapshot(C.MODEL); assert snap == SPEC['snapshot']
    identity=[]
    for p in sorted({p for pat in PATTERNS for p in C.MODEL.glob(pat) if p.is_file()}):
        identity.append(dict(path=p.name, bytes=p.stat().st_size, sha256=C.file_sha(p)))
    docs=C.read(HISTORY/'documents.json')['docs']; token=np.load(HISTORY/'tokens.npy')
    assert len(docs)==len(token)==1024 and len({d['doc_id'] for d in docs})==1024
    assert C.file_sha(C.MODEL/'tokenizer.json')==C.file_sha(ROOT/'models/hf/Qwen3-14B/tokenizer.json')
    assert np.array_equal(token,np.asarray([d['tokens'] for d in docs]))
    for split,(lo,hi) in C.SPLITS.items():
        assert all(d['split']==split and d.get('license') and d.get('revision') for d in docs[lo:hi])
    np.save(OUT/'tokens.npy',token)
    write(OUT/'documents.json',dict(docs=docs,positions=C.POSITIONS,
        source_sha256=C.file_sha(HISTORY/'documents.json'),
        exposure='Previously used licensed natural documents. Family holdouts for new fits, not new globally blind material.'))
    write(OUT/'model-identity.json',dict(files=identity,config=C.read(C.MODEL/'config.json')))
    binary_power=check_phase(.08,256,alpha=.05/6)
    assert binary_power['pass']
    raw_bytes=11*(8192*2*(3*C.WIDTH+2*C.DIM)+3*C.WIDTH*C.DIM*2)
    hidden_bytes=8192*2*C.DIM*(SPEC['layers']+1)
    plan=dict(schema='goodfire_middle_plan.v2',model_name=SPEC['name'],precision='BF16',
        rotary_initialization='CPU float32 inverse frequencies moved to CUDA, matching full HF loader; v1 CUDA-initialized frequencies superseded before analysis.',
        blocks=BLOCKS,width=C.WIDTH,hidden_size=C.DIM,model_layers=SPEC['layers'],
        units=11*C.WIDTH,documents=1024,context_tokens=1024,positions=C.POSITIONS,splits=C.SPLITS,
        tokens_sha256=C.file_sha(OUT/'tokens.npy'),documents_sha256=C.file_sha(OUT/'documents.json'),
        source_hashes=source_files(),
        runtime={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','safetensors','scipy']},
        hypotheses=[
            'Native G/V/A and down-column arithmetic reproduce recorded MLP values.',
            'Discovery-selected context predictors generalize beyond current-token and background controls.',
            'Predicting both gate and value then composing SiLU(G)*V improves product prediction.',
            'Frozen adjacent-unit predictors generalize and retain useful information during free rollout.'],
        thresholds=dict(replay_relative_max=.02,predictive_MSE_ratio_max=.9,control_norm_error_max=.02,
                        replay_KL_max=.001,restore_bitwise=True),
        evidence='Observation and predictive screening; no semantic confirmation from words, parameter cosine or correlations.',
        resources=dict(campaign_new_peak_GiB=PEAK_GIB,raw_bytes=raw_bytes,all_hidden_bytes=hidden_bytes,
            free_disk_GiB_min=20,forward_batch=2,document_chunk=256,collection_minutes_max=120,
            shared_GPU_GiB_max=1,models_serial=True),
        acquisition='Native G/V/A, normalized MLP input/output at eight positions; hs[0] and all residual block outputs at same positions. Both G/V/A poles scanned at all positions 512..1023.',
        prediction='Discovery-only standardization and fixed ridge; lexical hash, matched-current, exact-token shrinkage and shifted-document controls; both document holdouts. Product-composed prediction vs direct A ridge.',
        intervention=dict(blocks=[18,23,28],units_per_block=2,doses=[0.,.5,2.],natural_n=256,
            selection='Discovery std(A)*norm(down) and distinct largest double-negative abs(A)*norm(down); no heldout selection.',
            binary_power=binary_power,capability_n=512,
            scope='Finite final-token write interventions and norm-matched random controls; not semantic steering.'),
        reuse='Same frozen corpus. Historical B28 drift descriptive because legacy RoPE initialization differs; full HF reference at same batch/mask supplies numerical admission.',
        retention=dict(purpose='All-unit client query, later condition experiments and reproducibility',review_date='2026-12-13'),
        limitations=['Corpus previously exposed','Only eight saved positions per document',
            'No globally independent semantic lockbox','No SAE/transcoder comparison in this native atlas phase'],
        references=['https://already-clean.goodfire.com/','https://www.goodfire.com/research/interpreting-lm-parameters'])
    write(OUT/'plan.json',plan)
    write(OUT/'analysis-plan.json',dict(schema='goodfire_middle_analysis.v1',plan_sha256=C.file_sha(OUT/'plan.json'),
        fit='Discovery rows only',feature_budget=262,ridge_per_sample=.01,token_mean_shrinkage=10.,
        selection='Both holdouts: context <=0.9 hashed lexical; beats mean, matched current, exact-token+background and shifted targets.',
        scope='Descriptive census and predictive screening, not FWER-controlled semantic confirmation'))
    write(OUT/'preflight.json',dict(storage_admitted=False,required_additional_peak_GiB=PEAK_GIB,
        free_GiB=shutil.disk_usage(BASE).free/2**30,model_hashes_verified=True,source_frozen=True))
    log('preparation_complete',1,1,start)


def admit():
    check_write(BASE, PEAK_GIB*2**30, operation='Both-model B18-B28 campaign peak')
    for key in SPECS:
        p=BASE/key/'plan.json'; assert p.exists(), 'Prepare both models before admission'
        for name,digest in C.read(p)['source_hashes'].items():
            assert C.file_sha(ROOT/name)==digest, name
    write(OUT/'storage-admission.json',dict(passed=True,peak_GiB=PEAK_GIB,
        free_GiB=shutil.disk_usage(BASE).free/2**30,plan_sha256=C.file_sha(OUT/'plan.json')))


def require_admission():
    admission=BASE/'qwen4/storage-admission.json'
    assert admission.exists() and C.read(admission)['passed'], 'Storage not admitted; no GPU work allowed'
    p=C.read(OUT/'plan.json')
    for name,digest in p['source_hashes'].items():assert C.file_sha(ROOT/name)==digest,name
    assert C.file_sha(OUT/'tokens.npy')==p['tokens_sha256']
    check_write(OUT)
    return p


def pilot():
    import torch
    p=require_admission();start=time.monotonic();checks=[];times=[]
    token=np.load(OUT/'tokens.npy')[:4]
    with C.GpuLease(),torch.inference_mode():
        cfg,pos,pe,mask=C.context();emb=weight('model.embed_tokens.weight').cuda()
        hidden=torch.nn.functional.embedding(torch.from_numpy(token.astype(np.int64)).cuda(),emb)
        del emb;torch.cuda.empty_cache()
        for b in range(SPEC['layers']):
            t=time.monotonic();layer=C.load_block(b,cfg);loaded=time.monotonic()-t;t=time.monotonic()
            a=layer(hidden[:2],attention_mask=mask,position_ids=pos,position_embeddings=pe,use_cache=False)
            c=layer(hidden[2:],attention_mask=mask,position_ids=pos,position_embeddings=pe,use_cache=False)
            torch.cuda.synchronize();duration=time.monotonic()-t
            if b in [0,18,23,28,SPEC['layers']-1]:
                one=layer(hidden[:1],attention_mask=mask,position_ids=pos,position_embeddings=pe,use_cache=False)
                rel=float((one.float()-a[:1].float()).norm()/a[:1].float().norm().clamp_min(1e-12))
                assert rel<=p['thresholds']['replay_relative_max']
                checks.append(dict(block=b,relative_L2=rel,bitwise=bool(torch.equal(one,a[:1]))))
                del one
            hidden=torch.cat([a,c]);del layer,a,c;gc.collect();torch.cuda.empty_cache()
            times.append(dict(block=b,load_seconds=loaded,four_document_seconds=duration))
            log('pilot',b+1,SPEC['layers'],start)
    estimate=sum(x['four_document_seconds'] for x in times)*256+sum(x['load_seconds'] for x in times)*4
    assert estimate<120*60
    write(OUT/'pilot.json',dict(checks=checks,timings=times,estimated_collection_seconds=estimate))


def collect():
    import torch
    p=require_admission();assert (OUT/'pilot.json').exists()
    assert not (OUT/'B18-G.npy').exists(), 'Never overwrite partial native fields'
    check_write(OUT,p['resources']['raw_bytes']+p['resources']['all_hidden_bytes']+2*2**30)
    start=time.monotonic();token=np.load(OUT/'tokens.npy');maps={};extrema={};replay=[]
    for b in BLOCKS:
        for f,w in [('G',C.WIDTH),('V',C.WIDTH),('A',C.WIDTH),('input',C.DIM),('output',C.DIM)]:
            maps[b,f]=np.lib.format.open_memmap(OUT/f'B{b}-{f}.npy',mode='w+',dtype=np.uint16,shape=(1024,8,w))
    hs=np.lib.format.open_memmap(OUT/'hidden-all.npy',mode='w+',dtype=np.uint16,
        shape=(1024,SPEC['layers']+1,8,C.DIM))
    final=np.lib.format.open_memmap(OUT/'final-hidden.npy',mode='w+',dtype=np.uint16,shape=(1024,8,C.DIM))
    log('collection_started',0,1024*SPEC['layers'],start)
    with C.GpuLease(),torch.inference_mode():
        cfg,pos,pe,mask=C.context()
        for chunk in range(0,1024,256):
            stage=torch.empty((256,1024,C.DIM),device='cuda',dtype=torch.bfloat16)
            emb=weight('model.embed_tokens.weight').cuda()
            for i in range(0,256,2):
                stage[i:i+2]=torch.nn.functional.embedding(torch.from_numpy(token[chunk+i:chunk+i+2].astype(np.int64)).cuda(),emb)
            del emb;torch.cuda.empty_cache()
            hs[chunk:chunk+256,0]=stage[:,C.POSITIONS].contiguous().view(torch.uint16).cpu().numpy()
            for b in range(SPEC['layers']):
                layer=C.load_block(b,cfg);capture={};hooks=[]
                if b in BLOCKS:
                    hooks=[layer.mlp.gate_proj.register_forward_hook(lambda m,a,y:capture.update(G=y)),
                        layer.mlp.up_proj.register_forward_hook(lambda m,a,y:capture.update(V=y)),
                        layer.mlp.down_proj.register_forward_pre_hook(lambda m,a:capture.update(A=a[0])),
                        layer.mlp.register_forward_pre_hook(lambda m,a:capture.update(input=a[0])),
                        layer.mlp.register_forward_hook(lambda m,a,y:capture.update(output=y))]
                    if chunk==0:
                        for f,mod in [('gate',layer.mlp.gate_proj),('up',layer.mlp.up_proj),('down',layer.mlp.down_proj)]:
                            np.save(OUT/f'B{b}-W-{f}.npy',mod.weight.contiguous().view(torch.uint16).cpu().numpy())
                for i in range(0,256,2):
                    y=layer(stage[i:i+2],attention_mask=mask,position_ids=pos,position_embeddings=pe,use_cache=False)
                    stage[i:i+2]=y
                    hs[chunk+i:chunk+i+2,b+1]=y[:,C.POSITIONS].contiguous().view(torch.uint16).cpu().numpy()
                    if b in BLOCKS:
                        for f in ['G','V','A','input','output']:
                            maps[b,f][chunk+i:chunk+i+2]=capture[f][:,C.POSITIONS].contiguous().view(torch.uint16).cpu().numpy()
                        split=next(s for s,(lo,hi) in C.SPLITS.items() if lo<=chunk+i<hi)
                        for f in ['G','V','A']:
                            z=capture[f][:,512:].reshape(-1,C.WIDTH).float()
                            for pole in ['high','low']:
                                val,ix=z.max(0) if pole=='high' else z.min(0)
                                ind=(chunk+i+ix//512)*1024+512+ix%512
                                val,ind=val.cpu().numpy(),ind.cpu().numpy();key=(b,f,split,pole)
                                if key not in extrema:extrema[key]=[val,ind]
                                else:
                                    old=extrema[key];sel=val>old[0] if pole=='high' else val<old[0]
                                    old[0][sel]=val[sel];old[1][sel]=ind[sel]
                        if i==0:
                            a=torch.nn.functional.silu(capture['G'])*capture['V']
                            out=torch.nn.functional.linear(a,layer.mlp.down_proj.weight)
                            rel=float((out.float()-capture['output'].float()).norm()/capture['output'].float().norm().clamp_min(1e-12))
                            assert torch.equal(a,capture['A']) and rel<=.02
                            replay.append(dict(chunk=chunk,block=b,product_bitwise=True,output_relative_L2=rel))
                            del a,out
                        capture.clear()
                    del y
                for h in hooks:h.remove()
                del layer;gc.collect();torch.cuda.empty_cache()
                log('collection',chunk*SPEC['layers']+(b+1)*256,1024*SPEC['layers'],start,block=b,chunk=chunk)
                assert time.monotonic()-start<120*60, 'Collection time budget exceeded'
            norm=weight('model.norm.weight').cuda()
            x=stage[:,C.POSITIONS];x=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+cfg.rms_norm_eps)).to(torch.bfloat16)*norm
            final[chunk:chunk+256]=x.contiguous().view(torch.uint16).cpu().numpy()
            del stage,norm,x
            for a in maps.values():a.flush()
            hs.flush();final.flush()
            write(OUT/'collection-progress.json',dict(completed_documents=chunk+256,replay=replay))
        np.savez(OUT/'prefix-extrema.npz',**{f'B{b}_{f}_{s}_{pole}_{k}':v[j]
            for (b,f,s,pole),v in extrema.items() for j,k in enumerate(['value','index'])})
    write(OUT/'collection.json',dict(documents=1024,blocks=BLOCKS,selected_positions=8,
        full_prefix_extrema_positions=[512,1023],hidden_layers=SPEC['layers']+1,
        hidden_boundary='hs[0] embedding; hs[b+1] residual block b output before final norm',
        seconds=time.monotonic()-start,replay=replay,plan_sha256=C.file_sha(OUT/'plan.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=SPECS);p.add_argument('stage',choices=['prepare','admit','pilot','collect'])
    a=p.parse_args();configure(a.model);globals()[a.stage]()


if __name__=='__main__':main()
