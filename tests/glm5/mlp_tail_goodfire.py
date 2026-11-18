"""Native B32--B35 atlas: long-context poles, signed weights and frozen layer prediction.

No dictionary labels, semantic claims or causal edges are inferred from correlation.
Reuses verified original BF16 normalized MLP inputs; GPU replays actual native MLPs.
"""
import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from shared.dist_protocol.mechanism_trace import GpuLease, digest, file_sha, save_json, unbits

RUN = 'MLP-TAIL-GOODFIRE-20261108-v1'
OUT = ROOT / 'tests/glm5/result/mlp_tail_goodfire_v1'
PUBLIC = ROOT / 'research/mlp_tail/v1'
BANK = ROOT / 'tests/glm5/result/mlp_atlas_dictionary_v1'
TOKENS = ROOT / 'tests/glm5/result/native_mlp_long_v1/input-tokens.npy'
MODEL = ROOT / 'models/hf/qwen3-4b'
BLOCKS = [32, 33, 34, 35]
WIDTH, DIM, N = 9728, 2560, 32768
SNAPSHOT = '6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad'
SEED = 20261108


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def log(stage, done=0, total=0, started=None, **fields):
    elapsed = time.monotonic() - started if started else 0
    fields.update(stage=stage, done=done, total=total, seconds=round(elapsed, 2),
                  eta_seconds=round(elapsed / done * (total-done), 1) if done and total else None)
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():
        t = sys.modules['torch']
        fields.update(cuda_allocated_bytes=t.cuda.memory_allocated(), cuda_reserved_bytes=t.cuda.memory_reserved(),
                      shared_gpu_memory_bytes=None, shared_memory_note='WDDM共享显存计数未取得；计算张量显式CUDA，不做CPU卸载。')
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def weights(block):
    from safetensors import safe_open
    index = read(MODEL / 'model.safetensors.index.json')['weight_map']
    result = {}
    for field, suffix in [('gate','gate_proj'),('up','up_proj'),('down','down_proj')]:
        key = f'model.layers.{block}.mlp.{suffix}.weight'
        with safe_open(MODEL / index[key], framework='pt', device='cpu') as f:
            result[field] = f.get_tensor(key)
    return result


def data(block, field, split=None):
    x = np.load(OUT / f'B{block}-{field}.npy', mmap_mode='r')
    if split is not None:
        a,b = {'discovery':(0,16384),'confirm':(16384,24576),'holdout':(24576,32768)}[split]
        x = x[a:b]
    return unbits(x)


def preflight():
    OUT.mkdir(parents=True, exist_ok=True); PUBLIC.mkdir(parents=True, exist_ok=True)
    if (OUT/'plan.json').exists():
        raise RuntimeError('frozen plan exists; use the recorded run, not a new freeze')
    import torch
    from shared.dist_protocol.natural_register_runner import snapshot
    m = read(BANK / 'bank/manifest.json')
    assert m['state']=='complete' and m['model_snapshot']==SNAPSHOT
    started=time.monotonic();log('preflight',started=started)
    assert snapshot(MODEL)==SNAPSHOT
    assert file_sha(TOKENS)==read(ROOT/'research/mlp_atlas/dictionary_v1/prereg.research.json')['source_tokens_sha256']
    required=[]
    for block in BLOCKS:
        for field in ['input','output']:
            meta=next(f for f in m['files'] if f['path']==f'B{block:02d}-{field}.npy')
            path=BANK/'bank'/meta['path'];assert file_sha(path)==meta['sha256']
            required.append(meta)
    assert torch.cuda.is_available()
    checks=[]
    with GpuLease(), torch.inference_mode():
        for block in BLOCKS:
            w={k:v.cuda() for k,v in weights(block).items()}
            native=np.load(BANK/f'bank/B{block:02d}-input.npy',mmap_mode='r')[:128].reshape(-1,DIM)
            x=torch.from_numpy(np.asarray(native).copy()).view(torch.bfloat16).cuda()
            a=torch.nn.functional.silu(torch.nn.functional.linear(x,w['gate']))*torch.nn.functional.linear(x,w['up'])
            y=torch.nn.functional.linear(a,w['down']).float().cpu().numpy()
            expected=unbits(np.load(BANK/f'bank/B{block:02d}-output.npy',mmap_mode='r')[:128]).reshape(-1,DIM)
            relative=float(np.linalg.norm(y-expected)/max(np.linalg.norm(expected),1e-12))
            checks.append({'block':block,'n_positions':len(x),'relative_output_error':relative,
                           'bitwise_equal':bool(np.array_equal(y,expected))})
            assert relative<=.02, 'replay calibration failed; do not derive features'
            del x,a,y,w;torch.cuda.empty_cache()
    estimated=N*WIDTH*2*3*4 + 3*3*WIDTH**2*4 + 3*2*WIDTH**2*4 +4*3*WIDTH*DIM*2
    assert estimated<16*2**30 and shutil.disk_usage(OUT).free>estimated+8*2**30
    docs=read(BANK/'source-documents.json')['docs']
    plan={'schema':'mlp_tail_plan.v1','kind':'real','run_id':RUN,'model_snapshot':SNAPSHOT,
          'blocks':BLOCKS,'n_units':len(BLOCKS)*WIDTH,'n_documents':2048,'positions':m['positions'],
          'context_tokens':1024,'n_positions_per_layer':N,'dtype':'BF16 native bits; derived statistics FP32/FP64',
          'sources':dict(Counter(r['source'] for r in docs)), 'source_manifest_sha256':file_sha(BANK/'bank/manifest.json'),
          'source_tokens_sha256':file_sha(TOKENS),'source_files':required,
          'split':{'discovery_docs':[0,1024],'confirm_docs':[1024,1536],'holdout_docs':[1536,2048]},
          'exposure':'Existing long-context bank used previously. Heldout means not fitted by this new estimator; not globally blind or new materials.',
          'method':'Native gate/up/product poles and four sign regimes; all-unit weight alignment and context associations; frozen sparse adjacent-layer predictor plus three-step rollout.',
          'thresholds':{'replay_relative_output_error_max':.02,'association_abs_min':.2,'ridge_lambda_per_sample':.01,
                        'parents_per_unit':8,'top_contexts_per_pole_per_split':8,'frozen_before_new_associations':True},
          'prediction_inputs':'Observed B32 native A for rollout; observed previous-layer A for one-step. Target states, future tokens and heldout labels unavailable to fitting.',
          'baselines':['discovery mean','same-index previous unit','eight seeded random previous units','token identity + language/position summaries'],
          'evidence':'Correlations and weight compatibility are not causal paths; semantic labels remain candidate/unknown until tested.',
          'resource':{'new_disk_budget_GiB':16,'estimated_new_bytes':int(estimated),'GPU_minutes_max':30,'seed':SEED},
          'preflight':checks,'source_code_sha256':file_sha(__file__),
          'references':['https://already-clean.goodfire.com/','https://www.goodfire.com/research/interpreting-lm-parameters']}
    save_json(OUT/'plan.json',plan);save_json(PUBLIC/'prereg.research.json',plan)
    save_json(OUT/'preflight.json',{'checks':checks,'seconds':time.monotonic()-started,'snapshot':SNAPSHOT})
    log('preflight_complete',4,4,started,estimated_GiB=estimated/2**30)


def derive():
    import torch
    plan=read(OUT/'plan.json'); assert plan['model_snapshot']==SNAPSHOT
    torch.set_num_threads(4);started=time.monotonic();result=[]
    with GpuLease(),torch.inference_mode():
        for bi,block in enumerate(BLOCKS):
            assert not (OUT/f'B{block}-A.npy').exists(), 'do not overwrite native derived data'
            w={k:v.cuda() for k,v in weights(block).items()}
            for key,value in w.items():
                np.save(OUT/f'B{block}-W-{key}.npy',value.detach().cpu().contiguous().view(torch.uint16).numpy())
            maps={f:np.lib.format.open_memmap(OUT/f'B{block}-{f}.npy',mode='w+',dtype=np.uint16,shape=(N,WIDTH)) for f in ['G','V','A']}
            xmap=np.load(BANK/f'bank/B{block:02d}-input.npy',mmap_mode='r').reshape(N,DIM)
            ymap=np.load(BANK/f'bank/B{block:02d}-output.npy',mmap_mode='r').reshape(N,DIM)
            squared_error=0.;energy=0.;equal=0
            for i in range(0,N,2048):
                x=torch.from_numpy(np.asarray(xmap[i:i+2048]).copy()).view(torch.bfloat16).cuda()
                g=torch.nn.functional.linear(x,w['gate']);v=torch.nn.functional.linear(x,w['up']);a=torch.nn.functional.silu(g)*v
                y=torch.nn.functional.linear(a,w['down']).float().cpu().numpy();expected=unbits(ymap[i:i+2048])
                squared_error+=float(np.square(y-expected,dtype=np.float64).sum());energy+=float(np.square(expected,dtype=np.float64).sum())
                equal+=int(np.count_nonzero(y==expected))
                for field,tensor in [('G',g),('V',v),('A',a)]:maps[field][i:i+len(x)]=tensor.contiguous().view(torch.uint16).cpu().numpy()
                log('native_replay',bi*N+i+len(x),len(BLOCKS)*N,started,block=block)
                del x,g,v,a,y
            for a in maps.values():a.flush()
            relative=math.sqrt(squared_error/max(energy,1e-12));assert relative<=plan['thresholds']['replay_relative_output_error_max']
            result.append({'block':block,'relative_output_error':relative,'exact_output_fraction':equal/(N*DIM)})
            del maps,xmap,ymap,w;gc.collect();torch.cuda.empty_cache()
    save_json(OUT/'replay.json',{'schema':'mlp_tail_replay.v1','kind':'real','run_id':RUN,'checks':result,'seconds':time.monotonic()-started})


def profiles():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    docs=read(BANK/'source-documents.json')['docs'];positions=read(OUT/'plan.json')['positions']
    tokens=np.load(TOKENS,mmap_mode='r');ids=np.asarray(tokens[:,positions]).reshape(-1)
    classes=[]
    for id in ids:
        text=tok.decode([int(id)])
        classes.append([int(any('\u4e00'<=c<='\u9fff' for c in text)),int(any(c.isdigit() for c in text)),
                        int(any(c in '，。！？；：,.!?;:' for c in text)),int('\n' in text),int(any(c in '{}[]();=<>`' for c in text))])
    classes=np.asarray(classes,np.float32)
    np.save(OUT/'token_ids.npy',ids);np.save(OUT/'token_classes.npy',classes)
    # One copy of long causal prefixes, shared by all unit cards; future continuation explicitly separate.
    contexts=[]
    for i,doc in enumerate(docs):
        contexts.append({**doc,'tokens':tokens[i,:1024].tolist()})
    save_json(OUT/'documents.json',{'docs':contexts,'positions':positions,'license_scope':'Original source metadata retained; original token bank hash in prereg.'})
    started=time.monotonic()
    for bi,block in enumerate(BLOCKS):
        arrays={f:data(block,f) for f in ['G','V','A']}
        fields={}
        for field,x in arrays.items():
            mean=x.mean(0,dtype=np.float64);std=x.std(0,dtype=np.float64)
            stats={'mean':mean.tolist(),'std':std.tolist(),'min':x.min(0).tolist(),'max':x.max(0).tolist()}
            for split,(lo,hi) in {'discovery':(0,16384),'confirm':(16384,24576),'holdout':(24576,32768)}.items():
                z=x[lo:hi]
                high=np.argpartition(z,-8,axis=0)[-8:];low=np.argpartition(z,8,axis=0)[:8]
                columns=np.arange(WIDTH)[None,:]
                high=np.take_along_axis(high,np.argsort(z[high,columns],axis=0)[::-1],axis=0)
                low=np.take_along_axis(low,np.argsort(z[low,columns],axis=0),axis=0)
                stats[split]={'positive_indices':(high+lo).T.tolist(),'negative_indices':(low+lo).T.tolist()}
            fields[field]=stats
        g,v,a=(arrays[f] for f in ['G','V','A'])
        regimes=np.stack([((g>=0)&(v>=0)).mean(0),((g>=0)&(v<0)).mean(0),((g<0)&(v>=0)).mean(0),((g<0)&(v<0)).mean(0)],1)
        cues={}
        for split,(lo,hi) in {'discovery':(0,16384),'confirm':(16384,24576),'holdout':(24576,32768)}.items():
            centered=classes[lo:hi]-classes[lo:hi].mean(0);z=g[lo:hi]-g[lo:hi].mean(0)
            den=np.sqrt(np.square(centered,dtype=np.float64).sum(0)[:,None]*np.square(z,dtype=np.float64).sum(0)[None,:])
            corr=np.divide(centered.T@z,den,out=np.zeros((5,WIDTH)),where=den>1e-12)
            cues[split]=corr.T.tolist()
        profile={'schema':'mlp_tail_layer.v1','kind':'real','run_id':RUN,'model_snapshot':SNAPSHOT,'block':block,'width':WIDTH,
                 'fields':fields,'sign_regime_fractions':regimes.tolist(),'cue_correlations':cues,
                 'cue_names':['当前token汉字','数字字符','标点字符','换行','代码符号'],
                 'label_status':'candidate_token_cues_only_not_semantic_explanations','scope':'长上下文实际输入，词身份相关可构成解释替代；非上下文语义选择性或因果证明。'}
        save_json(OUT/f'B{block}-profile.json',profile)
        log('profiles',bi+1,4,started,block=block)
        del arrays,g,v,a;gc.collect()


def associations():
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    started=time.monotonic();audit=[]
    with GpuLease(),torch.inference_mode():
        for pair,(source,target) in enumerate(zip(BLOCKS,BLOCKS[1:])):
            for split in ['discovery','confirm','holdout']:
                x=torch.from_numpy(data(source,'A',split)).cuda();y=torch.from_numpy(data(target,'A',split)).cuda()
                x-=x.mean(0);y-=y.mean(0)
                den=x.square().sum(0).sqrt()[:,None]*y.square().sum(0).sqrt()[None,:]
                matrix=(x.T@y)/den.clamp_min(1e-20)
                assert torch.isfinite(matrix).all()
                np.save(OUT/f'B{source}-B{target}-rho-{split}.npy',matrix.cpu().numpy())
                audit.append({'source':source,'target':target,'split':split,'shape':[WIDTH,WIDTH]})
                del x,y,den,matrix;torch.cuda.empty_cache()
                log('full_associations',pair*3+['discovery','confirm','holdout'].index(split)+1,9,started)
            down=torch.from_numpy(unbits(np.load(OUT/f'B{source}-W-down.npy'))).cuda().T
            down=down/down.norm(dim=1,keepdim=True).clamp_min(1e-20)
            for field in ['gate','up']:
                readw=torch.from_numpy(unbits(np.load(OUT/f'B{target}-W-{field}.npy'))).cuda()
                readw=readw/readw.norm(dim=1,keepdim=True).clamp_min(1e-20)
                matrix=down@readw.T
                np.save(OUT/f'B{source}-B{target}-weight-{field}.npy',matrix.cpu().numpy())
                del readw,matrix
            del down;torch.cuda.empty_cache()
    save_json(OUT/'associations.json',{'schema':'mlp_tail_associations.v1','kind':'real','run_id':RUN,'matrices':audit,
              'scope':'All native unit pairs retained. Pearson associations and raw write/read cosine exclude causal proof; weight cosine omits attention and RMSNorm.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','derive','profiles','associations'])
    globals()[parser.parse_args().stage]()
