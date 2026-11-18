"""Identity, storage, BF16 collection and native interventions for state formation."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import contextlib,hashlib,json,shutil,subprocess,time,zipfile
from pathlib import Path
from datetime import datetime
import numpy as np
from functools import lru_cache
from tools import provenance

ROOT=Path(__file__).resolve().parents[2];LOGICAL_OUT=ROOT/'tests/glm5/result/state_formation_v1';OUT=LOGICAL_OUT.resolve()
MODEL=ROOT/'models/hf/qwen3-4b';RUN='STATE-FORMATION-20261109-v1';SEED=202611090612
SNAP='6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad'
READOUTS=['record_end','query_end','instruction_end','answer_prefix']
TERMS=['constant','query','record','language','query_language','record_language','color','color_language']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
@lru_cache(None)
def git_value(args):return subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}',*args],cwd=ROOT,capture_output=True,text=True).stdout.strip()
def setup_provenance():
    provenance.RESULTS=OUT/'provenance'
    provenance._git=lambda *args:git_value(args)
    provenance.model_revision=lambda _:SNAP
def save(name,body):
    OUT.mkdir(exist_ok=True);setup_provenance()
    helper=provenance.write_result(Path(name).stem,body,model_path=MODEL,seed=SEED,tool=RUN)
    prov=read(helper)['provenance'];assert not provenance.check_provenance({'provenance':prov})
    doc={**body,'kind':'real','run_id':RUN,'model_snapshot':SNAP,'provenance':prov,'helper_artifact':str(helper.relative_to(OUT))}
    target=OUT/name;tmp=target.with_suffix(target.suffix+'.tmp')
    tmp.write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x),encoding='utf-8');tmp.replace(target)
    return doc
def save_npz(name,**arrays):
    target=OUT/name;tmp=target.with_suffix('.tmp');prov=read(OUT/'plan.json')['provenance']
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for key,value in {**arrays,'provenance':json.dumps(prov)}.items():
            with archive.open(key+'.npy','w',force_zip64=True) as stream:np.lib.format.write_array(stream,np.asarray(value),allow_pickle=False)
    tmp.replace(target);return sha(target)
def checked():
    p=read(OUT/'plan.json')
    for name,digest in p['dependencies'].items():assert sha(ROOT/name)==digest,f'Frozen source changed: {name}'
    for name,digest in p['materials'].items():assert sha(OUT/name)==digest,name
    assert OUT.resolve()==Path(p['resources']['physical_output'])
    return p
def reserve():
    free=shutil.disk_usage(OUT.resolve()).free
    assert free>30*2**30,'Actual C volume reserve below 30 GiB; completed evidence retained'
    return free/2**30
def log(stage,n,total,start,**kwargs):
    elapsed=time.monotonic()-start
    row={'time':datetime.now().astimezone().isoformat(),'stage':stage,'done':n,'total':total,'seconds':round(elapsed,1),
         'eta_seconds':round(elapsed*(total-n)/n,1) if n else None,'free_GiB':round(shutil.disk_usage(OUT.resolve()).free/2**30,2),**kwargs}
    import sys
    torch=sys.modules.get('torch')
    if torch and torch.cuda.is_initialized():
        row.update(cuda_allocated_MiB=round(torch.cuda.memory_allocated()/2**20),cuda_peak_MiB=round(torch.cuda.max_memory_allocated()/2**20))
        try:
            command="(Get-Counter '\\GPU Process Memory(*)\\Shared Usage' -ErrorAction Stop).CounterSamples | Where-Object {$_.InstanceName -like 'pid_"+str(os.getpid())+"_*'} | Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum"
            proc=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,timeout=8)
            row['shared_gpu_MiB']=float(proc.stdout.strip())/2**20 if proc.returncode==0 and proc.stdout.strip() else None
        except Exception:row['shared_gpu_MiB']=None
    text=json.dumps(row,ensure_ascii=False);print(text,flush=True)
    with (OUT/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(text+'\n')
def unbits(value):return (np.asarray(value,dtype=np.uint32)<<16).view(np.float32)
def bits(value):
    import torch
    return value.detach().contiguous().view(torch.uint16).cpu().numpy().copy()
def design(rows):
    return np.array([[1,r['query'],r['record'],r['language'],r['query']*r['language'],r['record']*r['language'],r['color'],r['color']*r['language']] for r in rows],float)

def load_model(stage):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig,AutoModelForCausalLM
    from shared.dist_protocol.natural_register_runner import snapshot
    assert snapshot(MODEL)==SNAP
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    config=AutoConfig.from_pretrained(MODEL,local_files_only=True)
    with init_empty_weights():model=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16,attn_implementation='sdpa')
    index=read(MODEL/'model.safetensors.index.json')['weight_map'];files={};headers={};offsets={};records=[];start=time.monotonic()
    try:
        for name in sorted(set(index.values())):
            f=(MODEL/name).open('rb');size=int.from_bytes(f.read(8),'little');headers[name]=json.loads(f.read(size));files[name]=f;offsets[name]=8+size
        for i,(name,shard) in enumerate(sorted(index.items())):
            info=headers[shard][name];assert info['dtype']=='BF16';lo,hi=info['data_offsets'];f=files[shard];f.seek(offsets[shard]+lo);raw=bytearray(f.read(hi-lo))
            assert len(raw)==hi-lo;value=torch.frombuffer(raw,dtype=torch.bfloat16).reshape(info['shape']).to('cuda')
            expected=hashlib.sha256(raw).hexdigest();assert hashlib.sha256(bits(value).tobytes()).hexdigest()==expected
            parent,leaf=name.rsplit('.',1);module=model.get_submodule(parent);assert getattr(module,leaf).shape==value.shape
            setattr(module,leaf,torch.nn.Parameter(value,requires_grad=False));records.append({'name':name,'bytes':len(raw),'sha256':expected})
            del value,raw
            if (i+1)%64==0:log('load',i+1,len(index),start)
        model.tie_weights()
        for name,buffer in list(model.named_buffers()):
            assert not buffer.is_meta;parent,leaf=name.rsplit('.',1);model.get_submodule(parent)._buffers[leaf]=buffer.to('cuda')
        assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in model.parameters())
        save(f'loader-{stage}.json',{'schema':'state_formation_loader.v1','checks':records,'torch':torch.__version__,
            'plan_sha256':sha(OUT/'plan.json'),'precision':'native BF16, no quantization','device':'cuda:0','attention':'SDPA explicit bool causal mask','batch_size':1})
        return model.eval().requires_grad_(False)
    finally:
        for f in files.values():f.close()

class Capture:
    def __init__(self,model):
        self.model=model;self.on=False;self.values={};self.handles=[];self.positions=None
        self.handles.append(model.model.embed_tokens.register_forward_hook(self.hook('embedding')))
        self.handles.append(model.model.norm.register_forward_hook(self.hook('final_norm')))
        self.handles.append(model.model.norm.register_forward_pre_hook(self.pre('hidden35')))
        for block,layer in enumerate(model.model.layers):
            if block:self.handles.append(layer.register_forward_pre_hook(self.pre(f'hidden{block-1}')))
            self.handles.append(layer.self_attn.register_forward_hook(self.hook(f'attn{block}',tuple_output=True)))
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(self.pre(f'after_attn{block}')))
            self.handles.append(layer.mlp.register_forward_hook(self.hook(f'mlp{block}')))
            self.handles.append(layer.register_forward_hook(self.hook(f'hidden{block}')))
    def hook(self,name,tuple_output=False):
        def callback(module,args,out):
            if self.on:self.values[name]=(out[0] if tuple_output else out)[:,self.positions].detach().clone()
        return callback
    def pre(self,name):
        def callback(module,args):
            if self.on:self.values[name]=args[0][:,self.positions].detach().clone()
        return callback
    def forward(self,row,full=True):
        import torch
        self.positions=row['positions'] if full else [-1];self.on=True;self.values={}
        ids=torch.tensor([row['tokens']],device='cuda');n=ids.shape[1]
        mask=torch.ones((1,1,n,n),device='cuda',dtype=torch.bool).tril()
        with torch.inference_mode():out=self.model(input_ids=ids,attention_mask={'full_attention':mask},use_cache=False,logits_to_keep=1)
        self.on=False;return out.logits[0,0],self.values.copy()
    def arrays(self,values):
        import torch
        h=torch.stack([values['embedding'],*[values[f'hidden{b}'] for b in range(36)]],1)
        a=torch.stack([values[f'attn{b}'] for b in range(36)],1)
        r=torch.stack([values[f'after_attn{b}'] for b in range(36)],1)
        m=torch.stack([values[f'mlp{b}'] for b in range(36)],1)
        assert torch.equal(h[:,:36]+a,r),'Attention residual sum differs'
        assert torch.equal(r+m,h[:,1:]),'MLP residual sum differs'
        return {'hidden':bits(h[0]),'post_attention':bits(r[0]),'attn_write':bits(a[0]),'mlp_write':bits(m[0]),'final_norm':bits(values['final_norm'][0])}
    def close(self):
        for handle in self.handles:handle.remove()

@contextlib.contextmanager
def ablated(model,candidate,dose):
    block=candidate['block'];kind=candidate['module'];module=getattr(model.model.layers[block],'self_attn' if kind=='attention' else 'mlp')
    def callback(mod,args,out):
        value=out[0] if kind=='attention' else out;edited=value.clone();edited[:,-1]=(value[:,-1].float()*(1-dose)).to(value.dtype)
        return (edited,*out[1:]) if kind=='attention' else edited
    handle=module.register_forward_hook(callback)
    try:yield
    finally:handle.remove()

def boundary(values,candidate):return values[('after_attn' if candidate['module']=='attention' else 'hidden')+str(candidate['block'])][:,-1]

def matched_state(base,target_norm,seed):
    import torch
    generator=torch.Generator(device=base.device).manual_seed(seed)
    direction=torch.randn(base.shape,device=base.device,generator=generator);direction/=direction.norm();amount=target_norm;best=None
    if target_norm==0:return base.clone(),0.,0.
    for _ in range(24):
        state=(base.float()+direction*amount).to(base.dtype);actual=float((state.float()-base.float()).norm());error=abs(actual-target_norm)/target_norm
        if best is None or error<best[1]:best=(state,error,actual)
        if error<=.02:break
        amount*=target_norm/max(actual,1e-12)
    return best

@contextlib.contextmanager
def replaced_state(model,candidate,state):
    layer=model.model.layers[candidate['block']]
    if candidate['module']=='attention':
        raise RuntimeError('Attention residual replacement requires explicit block replay')
    else:
        def hook(mod,args,out):
            value=out.clone();value[:,-1]=state;return value
        handle=layer.register_forward_hook(hook)
    try:yield
    finally:handle.remove()

@contextlib.contextmanager
def residual_replay(model,candidate,state):
    """Replace the exact post-attention residual including its bypass and MLP.

    Other positions are unchanged. Executed inside the original decoder call;
    recomputation is verified by identity replay before causal measurements.
    """
    layer=model.model.layers[candidate['block']]
    if candidate['module']=='mlp':
        with replaced_state(model,candidate,state):yield
        return
    captured={}
    def read_residual(mod,args):captured['residual']=args[0]
    def rewrite(mod,args,out):
        value=captured['residual'].clone();value[:,-1]=state
        with __import__('torch').inference_mode():return value+layer.mlp(layer.post_attention_layernorm(value))
    first=layer.post_attention_layernorm.register_forward_pre_hook(read_residual);second=layer.register_forward_hook(rewrite)
    try:yield
    finally:first.remove();second.remove()

@contextlib.contextmanager
def exclude_source(model,candidate,positions):
    module=model.model.layers[candidate['block']].self_attn
    def hook(mod,args,kwargs):
        mask=kwargs['attention_mask'];assert mask is not None and mask.dtype==__import__('torch').bool
        mask=mask.clone();mask[:,:,-1,positions]=False
        return args,{**kwargs,'attention_mask':mask}
    handle=module.register_forward_pre_hook(hook,with_kwargs=True)
    try:yield
    finally:handle.remove()

def scores(logits,row,plan):
    keys=['en_red','en_blue','zh_red','zh_blue'];d=plan['choice_ids'];v=[float(logits[d[key]].max()) for key in keys]
    gold=(0 if row['color']==1 else 1)+(0 if row['language']==1 else 2);other=gold^1;other_lang=gold^2;top=int(logits.argmax())
    return {'four_logits':v,'color_margin':v[gold]-v[other],'language_margin':v[gold]-v[other_lang],
            'four_correct':int(np.argmax(v))==gold,'color_correct':v[gold]>v[other],'language_correct':v[gold]>v[other_lang],
            'first_top1':top in d[keys[gold]],'argmax':top}
