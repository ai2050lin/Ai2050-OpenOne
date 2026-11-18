"""BF16, layer-streamed Qwen3-14B native B28--B35 census.

Original HF decoder blocks, explicit causal masks, no quantization. A census
does not establish semantic monosemanticity or a causal circuit.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from shared.dist_protocol.mechanism_trace import GpuLease, digest, file_sha, save_json, unbits

MODEL = ROOT / 'models/hf/Qwen3-14B'
OUT = (ROOT / 'tests/glm5/result/qwen14_mlp_atlas_v1').resolve()
STORE = Path(r'C:\Users\Admin\.codex\visualizations\2026\11\03\01a1a045-c160-7150-acad-c557907e4e8a\research-storage\qwen14_mlp_atlas_v1')
RUN = 'QWEN14-MLP-B28-B35-20261109-v1'
BLOCKS = list(range(28, 36))
WIDTH, DIM, LENGTH = 17408, 5120, 1024
POSITIONS = [533, 598, 687, 787, 813, 840, 945, 1006]
SPLITS = {'discovery': (0, 512), 'confirm': (512, 768), 'holdout': (768, 1024)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def log(stage, done=0, total=0, start=None, **kw):
    import psutil
    elapsed = time.monotonic() - start if start else 0
    row = dict(stage=stage, done=done, total=total, seconds=round(elapsed, 2),
               eta_seconds=round(elapsed / done * (total-done), 1) if done else None,
               available_ram_bytes=psutil.virtual_memory().available, **kw)
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        t = sys.modules['torch']
        row.update(cuda_allocated=t.cuda.memory_allocated(), cuda_reserved=t.cuda.memory_reserved(),
                   cuda_peak=t.cuda.max_memory_allocated(), shared_gpu_bytes=None,
                   shared_note='WDDM shared-memory counter not measured; only each current block is resident on CUDA.')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def weight(key):
    from safetensors import safe_open
    index = read(MODEL/'model.safetensors.index.json')['weight_map']
    with safe_open(MODEL/index[key], framework='pt', device='cpu') as f:
        return f.get_tensor(key)


def load_block(block, config):
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    from safetensors import safe_open
    index = read(MODEL/'model.safetensors.index.json')['weight_map']
    prefix = f'model.layers.{block}.'
    with torch.device('meta'):
        layer = Qwen3DecoderLayer(config, block)
    state = {}
    for shard in sorted({v for k, v in index.items() if k.startswith(prefix)}):
        with safe_open(MODEL/shard, framework='pt', device='cpu') as f:
            for key in index:
                if key.startswith(prefix) and index[key] == shard:
                    state[key[len(prefix):]] = f.get_tensor(key).to(device='cuda', dtype=torch.bfloat16)
    layer.load_state_dict(state, assign=True, strict=True)
    return layer.eval()


def prepare():
    from transformers import AutoTokenizer
    from shared.dist_protocol.natural_register_runner import snapshot, PATTERNS
    assert OUT.resolve() == STORE.resolve(), 'Create the verified result junction before preparation'
    if (OUT/'plan.json').exists():
        raise RuntimeError('Immutable plan exists')
    start = time.monotonic(); log('preflight_hashing', start=start)
    sources = ROOT/'tests/glm5/result/mlp_tail_goodfire_v1/documents.json'
    docs = read(sources)['docs']
    chosen = []
    for split, count in [('discovery',512), ('confirm',256), ('holdout',256)]:
        source_split = 'lockbox' if split == 'holdout' else split
        pool = [d for d in docs if d['split'] == source_split]
        # Preserve document-family partitions inherited from the licensed corpus.
        selected = sorted(pool, key=lambda d: digest(d['doc_id']))[:count]
        chosen += [{**d, 'source_split':source_split, 'split':split,
                    'source_doc_index': d['doc_index'], 'doc_index':len(chosen)+i}
                   for i,d in enumerate(selected)]
    assert len(chosen)==1024 and len({d['prefix_sha256'] for d in chosen})==1024
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok4 = AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b', local_files_only=True)
    assert file_sha(MODEL/'tokenizer.json') == file_sha(ROOT/'models/hf/qwen3-4b/tokenizer.json')
    nonroundtrip=[]
    for d in chosen:
        text = tok4.decode(d['tokens'])
        encoded=tok.encode(text, add_special_tokens=False)
        assert encoded==tok4.encode(text,add_special_tokens=False), 'tokenizer mappings differ'
        if encoded!=d['tokens']: nonroundtrip.append(d['doc_id'])
    save_json(OUT/'tokenizer-audit.json', {'identical_tokenizer_file':True,
        'input_policy':'Preserve licensed original 1024-token prefixes, using identical token mappings. Decoding a cut byte-token prefix need not roundtrip.',
        'decode_encode_nonroundtrip_document_ids':nonroundtrip})
    np.save(OUT/'tokens.npy', np.asarray([d['tokens'] for d in chosen], dtype=np.int32))
    save_json(OUT/'documents.json', {'docs':chosen, 'positions':POSITIONS,
        'scope':'Licensed existing long documents; inherited document partitions; not globally unexposed material.'})
    snap = snapshot(MODEL)
    files = []
    for p in sorted({p for pattern in PATTERNS for p in MODEL.glob(pattern) if p.is_file()}):
        files.append({'path':p.name,'bytes':p.stat().st_size,'sha256':file_sha(p)})
    save_json(OUT/'model-identity.json', {'model_snapshot':snap,'files':files,'config':read(MODEL/'config.json')})
    plan = {'schema':'qwen14_mlp_plan.v1','kind':'real','run_id':RUN,'model_snapshot':snap,
        'model_name':'Qwen3-14B','precision':'BF16','blocks':BLOCKS,'width':WIDTH,'hidden_size':DIM,
        'model_layers':40,'units':len(BLOCKS)*WIDTH,'documents':1024,'context_tokens':LENGTH,
        'positions':POSITIONS,'splits':SPLITS,'source_documents_sha256':file_sha(sources),
        'tokens_sha256':file_sha(OUT/'tokens.npy'),'documents_sha256':file_sha(OUT/'documents.json'),
        'collector_sha256':file_sha(__file__),
        'method':'Native G/V/A both poles, four sign quadrants, full 512..1023-context extrema; original gate/up/down parameters. Discovery-only candidate fitting with inherited document holdouts.',
        'hypotheses':['Native arithmetic reproduces recorded MLP output within 2% relative error.',
            'Discovery conditional-response predictors improve unseen-document error over discovery-mean and lexical/position controls.',
            'Adjacent-layer response association is stable across document splits; this alone is not causality.'],
        'thresholds':{'replay_relative_max':.02,'holdout_mse_ratio_max':.9,'min_effective_documents':32,
                      'association_abs_min':.2,'sign_consistency_required':True},
        'scope':'Observational native-neuron atlas. No semantic/causal promotion from labels, correlations or weight alignment.',
        'exposure':'Corpus already used in Qwen3-4B research; Qwen3-14B responses new. Confirm/holdout excluded from feature fitting, not universally blind.',
        'resources':{'new_disk_GiB_max':26,'free_disk_GiB_min':5,'gpu_wall_minutes_max':120,
                     'document_chunk':256,'forward_batch':2,'pilot_discovery_documents':4},
        'validation':'HF native blocks with explicit causal mask; block arithmetic replay; batch sensitivity pilot; model/tokenizer hashes; complete dimensions and finite arrays.',
        'references':['https://already-clean.goodfire.com/','https://www.goodfire.com/research/interpreting-lm-parameters']}
    save_json(OUT/'plan.json',plan); log('preflight_complete',1,1,start,snapshot=snap)


def context():
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    torch.set_num_threads(4)
    config=AutoConfig.from_pretrained(MODEL,local_files_only=True);config._attn_implementation='sdpa'
    rotary=Qwen3RotaryEmbedding(config,device='cuda')
    pos=torch.arange(LENGTH,device='cuda')[None]
    x=torch.zeros((1,LENGTH,DIM),device='cuda',dtype=torch.bfloat16)
    pe=rotary(x,pos)
    mask=torch.full((LENGTH,LENGTH),float('-inf'),device='cuda',dtype=torch.bfloat16).triu(1)[None,None]
    return config,pos,pe,mask


def pilot():
    import torch
    start=time.monotonic();plan=read(OUT/'plan.json');token=np.load(OUT/'tokens.npy')[:4]
    times=[];checks=[]
    with GpuLease(),torch.inference_mode():
        cfg,pos,pe,mask=context();emb=weight('model.embed_tokens.weight').cuda().to(torch.bfloat16)
        hidden=torch.nn.functional.embedding(torch.from_numpy(token.astype(np.int64)).cuda(),emb)
        del emb;torch.cuda.empty_cache()
        for b in range(40):
            t=time.monotonic();layer=load_block(b,cfg);loaded=time.monotonic()-t
            t=time.monotonic();a=layer(hidden[:2],attention_mask=mask,position_ids=pos,position_embeddings=pe,use_cache=False)
            c=layer(hidden[2:],attention_mask=mask,position_ids=pos,position_embeddings=pe,use_cache=False)
            torch.cuda.synchronize();seconds=time.monotonic()-t
            if b in (0,28,35,39):
                one=layer(hidden[:1],attention_mask=mask,position_ids=pos,position_embeddings=pe,use_cache=False)
                error=float((one.float()-a[:1].float()).norm()/a[:1].float().norm().clamp_min(1e-12))
                checks.append({'block':b,'batch1_vs2_relative_error':error,'bitwise_equal':bool(torch.equal(one,a[:1]))})
                assert error<.02
            hidden=torch.cat([a,c]);del layer,a,c;gc.collect();torch.cuda.empty_cache()
            times.append({'block':b,'load_seconds':loaded,'four_doc_seconds':seconds});log('pilot',b+1,40,start,**times[-1])
        estimated=sum(r['four_doc_seconds'] for r in times)*256+sum(r['load_seconds'] for r in times)*4
        result={'kind':'real','run_id':RUN,'model_snapshot':plan['model_snapshot'],'checks':checks,
            'timings':times,'estimated_collection_minutes':estimated/60,'cuda_peak_bytes':torch.cuda.max_memory_allocated(),
            'scope':'4 discovery documents only; non-identical batch arithmetic is reported, not treated as a new model.'}
        save_json(OUT/'pilot.json',result)
    log('pilot_complete',1,1,start,estimate_minutes=estimated/60)


def collect():
    import torch
    plan=read(OUT/'plan.json');pilot_result=read(OUT/'pilot.json')
    assert pilot_result['estimated_collection_minutes']<plan['resources']['gpu_wall_minutes_max']
    assert file_sha(OUT/'tokens.npy')==plan['tokens_sha256']
    assert shutil.disk_usage(OUT).free>20*2**30
    assert not (OUT/'collection.json').exists()
    start=time.monotonic();tokens=np.load(OUT/'tokens.npy');n=len(tokens);p=len(POSITIONS)
    maps={}
    for b in BLOCKS:
        for f,w in [('G',WIDTH),('V',WIDTH),('A',WIDTH),('input',DIM),('output',DIM)]:
            path=OUT/f'B{b}-{f}.npy'
            assert not path.exists(), 'Use a new explicit recovery run rather than overwrite raw data'
            maps[b,f]=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint16,shape=(n,p,w))
    final=np.lib.format.open_memmap(OUT/'final-hidden.npy',mode='w+',dtype=np.uint16,shape=(n,p,DIM))
    extrema={};replay=[];finished=[]
    with GpuLease(),torch.inference_mode():
        cfg,pos,pe,mask=context()
        for chunk in range(0,n,256):
            size=min(256,n-chunk)
            # Resident states fit comfortably on this GPU; only weights stream.
            # Keep the identical two-document native forwards, without disk ping-pong.
            stage=torch.empty((size,LENGTH,DIM),dtype=torch.bfloat16,device='cuda')
            emb=weight('model.embed_tokens.weight').cuda().to(torch.bfloat16)
            for i in range(0,size,2):
                ids=torch.from_numpy(tokens[chunk+i:chunk+i+2].astype(np.int64)).cuda()
                stage[i:i+2]=torch.nn.functional.embedding(ids,emb)
            del emb;torch.cuda.empty_cache()
            for b in range(40):
                layer=load_block(b,cfg);capture={}
                hooks=[]
                if b in BLOCKS:
                    def gate_hook(m,a,y):capture['G']=y
                    def up_hook(m,a,y):capture['V']=y
                    def down_pre(m,a):capture['A']=a[0]
                    def mlp_pre(m,a):capture['input']=a[0]
                    def mlp_post(m,a,y):capture['output']=y
                    hooks=[layer.mlp.gate_proj.register_forward_hook(gate_hook),layer.mlp.up_proj.register_forward_hook(up_hook),
                        layer.mlp.down_proj.register_forward_pre_hook(down_pre),layer.mlp.register_forward_pre_hook(mlp_pre),layer.mlp.register_forward_hook(mlp_post)]
                    if chunk==0:
                        for field,mod in [('gate',layer.mlp.gate_proj),('up',layer.mlp.up_proj),('down',layer.mlp.down_proj)]:
                            np.save(OUT/f'B{b}-W-{field}.npy',mod.weight.contiguous().view(torch.uint16).cpu().numpy())
                for i in range(0,size,2):
                    x=stage[i:i+2]
                    y=layer(x,attention_mask=mask,position_ids=pos,position_embeddings=pe,use_cache=False)
                    stage[i:i+2]=y
                    if b in BLOCKS:
                        for f in ['G','V','A','input','output']:
                            maps[b,f][chunk+i:chunk+i+2]=capture[f][:,POSITIONS].contiguous().view(torch.uint16).cpu().numpy()
                        # Observe both poles over every token after a long 512-token prefix.
                        split=next(s for s,(lo,hi) in SPLITS.items() if lo<=chunk+i<hi)
                        for f in ['G','V','A']:
                            z=capture[f][:,512:].reshape(-1,WIDTH).float()
                            for pole in ['high','low']:
                                values,ix=(z.max(0) if pole=='high' else z.min(0))
                                indices=(chunk+i+ix//512)*LENGTH+512+ix%512
                                val=values.cpu().numpy();ind=indices.cpu().numpy()
                                key=(b,f,split,pole)
                                if key not in extrema:extrema[key]=[val,ind]
                                else:
                                    old=extrema[key];choose=val>old[0] if pole=='high' else val<old[0]
                                    old[0][choose]=val[choose];old[1][choose]=ind[choose]
                        if i==0:
                            a=torch.nn.functional.silu(capture['G'])*capture['V']
                            product_equal=bool(torch.equal(a,capture['A']))
                            yy=torch.nn.functional.linear(a,layer.mlp.down_proj.weight)
                            rel=float((yy.float()-capture['output'].float()).norm()/capture['output'].float().norm().clamp_min(1e-12))
                            assert product_equal and rel<=.02
                            replay.append({'chunk':chunk,'block':b,'product_bitwise_equal':product_equal,'output_relative_error':rel,
                                           'output_bitwise_equal':bool(torch.equal(yy,capture['output']))})
                            del a,yy
                        capture.clear()
                    del x,y
                    if i%32==0:log('collect',(chunk*40+b*size+i+2),n*40,start,chunk=chunk,block=b,document=i+2)
                for h in hooks:h.remove()
                del layer;gc.collect();torch.cuda.empty_cache()
                if time.monotonic()-start>plan['resources']['gpu_wall_minutes_max']*60:
                    raise RuntimeError('Frozen resource budget reached; retain raw data and stage for recovery')
            norm=weight('model.norm.weight').cuda().to(torch.bfloat16)
            for i in range(0,size,2):
                x=stage[i:i+2,POSITIONS]
                xx=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+cfg.rms_norm_eps)).to(torch.bfloat16)*norm
                final[chunk+i:chunk+i+2]=xx.contiguous().view(torch.uint16).cpu().numpy()
            del norm,x,xx
            for m in maps.values():m.flush()
            final.flush()
            del stage
            finished.append(chunk)
            save_json(OUT/'collection-progress.json',{'completed_chunks':finished,'replay':replay,'plan_sha256':file_sha(OUT/'plan.json')})
            log('chunk_complete',chunk+size,n,start)
        np.savez(OUT/'prefix-extrema.npz',**{f'B{b}_{f}_{s}_{pole}_{kind}':arr[j] for (b,f,s,pole),arr in extrema.items() for j,kind in enumerate(['value','index'])})
    save_json(OUT/'collection.json',{'schema':'qwen14_collection.v1','kind':'real','run_id':RUN,'model_snapshot':plan['model_snapshot'],
        'documents':n,'selected_positions':p,'full_prefix_extrema_positions':[512,1023],'replay':replay,'seconds':time.monotonic()-start,
        'collector_sha256':file_sha(__file__),'plan_sha256':file_sha(OUT/'plan.json'),'completed_chunks':finished})
    # Only these explicitly named, verified, reconstructible scratch files are removed.
    released=0
    for name in ['scratch-state-0.npy','scratch-state-1.npy']:
        path=OUT/name
        assert path.resolve().parent==STORE.resolve()
        if path.exists():released+=path.stat().st_size;path.unlink()
    save_json(OUT/'scratch-cleanup.json',{'bytes_released':released,'paths':['scratch-state-0.npy','scratch-state-1.npy'],
                                      'retained':'all native readouts, parameters, documents, extrema, replay and provenance'})
    log('collection_complete',n,n,start,released_bytes=released)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','pilot','collect'])
    globals()[parser.parse_args().stage]()
