"""Native, content-addressed Qwen3 computation traces. No semantic auto-labels.

All coordinates/units are retained at declared readouts. Deep traces additionally
save full-prefix hidden states and normalized, post-RoPE Q/K plus grouped V.
Attention rows are recomputed, explicitly labelled, never treated as causality.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_RAW=ROOT/'tests/glm5/result'
DEFAULT_INDEX=ROOT/'distributed_data/mechanism_traces'


def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def unbits(x):return (np.asarray(x,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
def save_json(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_bytes(canonical(x));tmp.replace(path)


class GpuLease:
    """Cross-process advisory lock; the OS releases it if its owner exits."""
    def __init__(self,path=None):self.path=Path(path or ROOT/'tests/glm5_temp/mechanism-gpu.lock');self.handle=None
    def acquire(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);f=self.path.open('a+b')
        if not self.path.stat().st_size:f.write(b'0');f.flush()
        f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt;msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl;fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except (OSError,BlockingIOError):f.close();raise RuntimeError('GPU is leased by another research/live process')
        self.handle=f;return self
    def release(self):
        if self.handle:self.handle.close();self.handle=None
    def __enter__(self):return self.acquire()
    def __exit__(self,*_):self.release()


class TraceStore:
    def __init__(self,index=DEFAULT_INDEX,raw_root=DEFAULT_RAW):
        self.index=Path(index);self.index.mkdir(parents=True,exist_ok=True);self.raw_root=Path(raw_root).resolve();self.verified={}
        with self.db() as db:db.execute('CREATE TABLE IF NOT EXISTS traces(id TEXT PRIMARY KEY,path TEXT,body TEXT,created REAL)')
    @contextmanager
    def db(self):
        db=sqlite3.connect(self.index/'index.sqlite3',timeout=30);db.row_factory=sqlite3.Row
        try:
            with db:yield db
        finally:db.close()
    def register(self,path):
        path=Path(path).resolve()
        if not path.is_relative_to(self.raw_root):raise ValueError('trace must stay inside configured result root')
        d=json.loads(path.read_text(encoding='utf-8'))
        if d.get('schema')!='mechanism_trace.v1' or d.get('kind')!='real':raise ValueError('real trace required')
        identity=d.get('trace_id');body={k:v for k,v in d.items() if k!='trace_id'}
        if identity!=digest(body):raise ValueError('trace manifest fingerprint differs')
        if len(d.get('model_snapshot',''))!=64:raise ValueError('full snapshot required')
        for f in d['files']:
            p=(path.parent/f['path']).resolve()
            if not p.is_relative_to(path.parent) or not p.is_file():raise ValueError('invalid native trace path')
            if p.stat().st_size!=f['bytes']:raise ValueError('native trace byte length differs')
        with self.db() as db:db.execute('INSERT OR IGNORE INTO traces VALUES(?,?,?,?)',(identity,str(path),canonical(d).decode(),time.time()))
        return d
    def catalog(self,run_id=''):
        with self.db() as db:rows=db.execute('SELECT id,body FROM traces ORDER BY created DESC').fetchall()
        keys=['trace_id','run_id','sample_id','entity','scope','split','n_tokens','model_snapshot','deep','readout_positions','n_layers','d_model','d_ff','n_heads','n_kv_heads','head_dim']
        return {'schema':'mechanism_trace_catalog.v1','traces':[{k:d.get(k) for k in keys} for r in rows for d in [json.loads(r['body'])] if not run_id or d['run_id']==run_id]}
    def manifest(self,identity):
        with self.db() as db:row=db.execute('SELECT path,body FROM traces WHERE id=?',(identity,)).fetchone()
        if not row:raise KeyError('trace not registered')
        d=json.loads(row['body']);path=Path(row['path']).resolve()
        if not path.is_relative_to(self.raw_root):raise ValueError('index path outside result root')
        if digest({k:v for k,v in d.items() if k!='trace_id'})!=identity:raise ValueError('index manifest corrupted')
        return d,path.parent
    def native_path(self,identity,name):
        d,root=self.manifest(identity);meta=next((x for x in d['files'] if x['path']==name),None)
        if not meta:raise KeyError('native field was not captured')
        p=(root/name).resolve()
        if not p.is_relative_to(root):raise ValueError('native path outside trace')
        stamp=(p.stat().st_size,p.stat().st_mtime_ns)
        if self.verified.get(str(p))!=stamp:
            if stamp[0]!=meta['bytes'] or file_sha(p)!=meta['sha256']:raise ValueError('native field corrupted')
            self.verified[str(p)]=stamp
        return p
    def block(self,identity,block):
        d,_=self.manifest(identity)
        if type(block)is not int or not 0<=block<d['n_layers']:raise ValueError('block outside trace')
        with np.load(self.native_path(identity,f'block_{block:02d}.npz')) as z:return {k:z[k] for k in z.files}
    def field(self,identity,block,position,boundary):
        d,_=self.manifest(identity);z=self.block(identity,block)
        allowed={'mlp_gate':'gate','mlp_value':'value','mlp_activation':'activation','mlp_output':'mlp_output',
            'attention_output':'attention_output','attention_input':'attention_input','mlp_input':'mlp_input','head_z':'head_z','resid_in':'resid_in'}
        if boundary not in allowed:raise ValueError('unsupported field boundary')
        if position not in d['readout_positions']:raise KeyError('this position was not captured at module boundaries')
        ri=d['readout_positions'].index(position);v=unbits(z[allowed[boundary]][ri])
        return {'schema':'mechanism_field.v1','kind':'real','trace_id':identity,'block':block,'position':position,'boundary':boundary,
            'shape':list(v.shape),'values':v.reshape(-1).tolist(),'source':'native_BF16_bits','model_snapshot':d['model_snapshot']}
    def attention_row(self,identity,block,head,query):
        d,_=self.manifest(identity)
        if type(head)is not int or not 0<=head<d['n_heads'] or type(query)is not int or not 0<=query<d['n_tokens']:raise ValueError('attention index outside trace')
        z=self.block(identity,block)
        if query in d['readout_positions']:
            r=d['readout_positions'].index(query);v=unbits(z['attention_rows'][r,head])
        elif d['deep']:
            q=unbits(z['q'][head,query]);k=unbits(z['k'][head//(d['n_heads']//d['n_kv_heads'])]);scores=k@q/np.sqrt(d['head_dim'])
            scores[np.arange(len(scores))>query]=-np.inf;scores-=scores.max();v=np.exp(scores);v/=v.sum()
        else:raise KeyError('query position not captured; select a recorded readout or a deep trace')
        return {'schema':'mechanism_attention_row.v1','trace_id':identity,'block':block,'head':head,'query_position':query,
            'kv_group':head//(d['n_heads']//d['n_kv_heads']),'weights':v.tolist(),'source':'recomputed_from_actual_postRoPE_QK',
            'not_causal_importance':True,'validation':d['reconstruction'][str(block)]['attention_relative_l2']}
    def hidden(self,identity,layer,position):
        d,_=self.manifest(identity)
        if not 0<=layer<=d['n_layers'] or not 0<=position<d['n_tokens']:raise ValueError('hidden index outside trace')
        if d['deep']:
            a=np.load(self.native_path(identity,'hidden_full.npy'),mmap_mode='r');v=unbits(a[layer,position])
        else:
            if position not in d['readout_positions']:raise KeyError('hidden position not captured')
            a=np.load(self.native_path(identity,'hidden_readouts.npy'),mmap_mode='r');v=unbits(a[layer,d['readout_positions'].index(position)])
        return {'schema':'mechanism_field.v1','trace_id':identity,'layer':layer,'position':position,'boundary':'embedding' if layer==0 else 'hidden_state',
            'shape':[len(v)],'values':v.tolist(),'source':'native_BF16_bits'}


class Recorder:
    """Captures the actual SDPA boundary and native module inputs/outputs."""
    def __init__(self,model,positions,folder,deep=False,position_offset=0):
        self.model=model;self.positions=list(positions);self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.deep=deep;self.offset=position_offset;self.data={};self.reconstruction={};self.handles=[];self.files=[]
    def bits(self,t):return t.detach().contiguous().view(self.torch.uint16).cpu().numpy().copy()
    def rows(self,t):return t[0,self.positions]
    def addfile(self,path):self.files.append({'path':path.name,'sha256':file_sha(path),'bytes':path.stat().st_size})
    def __enter__(self):
        import torch
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.torch=torch;self.registry=ALL_ATTENTION_FUNCTIONS;self.original=ALL_ATTENTION_FUNCTIONS['sdpa']
        def attention(module,q,k,v,mask,**kwargs):
            actual=self.original(module,q,k,v,mask,**kwargs);i=module.layer_idx;d=self.data[i]
            # Compute complete rows only; never materialize the full T-by-T matrix.
            qr=q[0,:,self.positions,:].transpose(0,1).float();kg=k[0].repeat_interleave(module.num_key_value_groups,0).float()
            scores=torch.einsum('rhd,htd->rht',qr,kg)*module.scaling
            query_positions=torch.tensor(self.positions,device=q.device)+self.offset
            scores=scores.masked_fill(torch.arange(k.shape[-2],device=q.device)[None,None,:]>query_positions[:,None,None],float('-inf'))
            if mask is not None:scores=scores+mask[0,0,self.positions,:][:,None,:]
            weights=torch.softmax(scores,-1);d['attention_rows']=self.bits(weights.to(q.dtype))
            reconstructed=torch.einsum('rht,htd->rhd',weights,v[0].repeat_interleave(module.num_key_value_groups,0).float())
            z=actual[0][0,self.positions];err=float((reconstructed-z.float()).norm()/z.float().norm().clamp_min(1e-8))
            self.reconstruction[str(i)]={'attention_relative_l2':err};d['head_z']=self.bits(z)
            if self.deep:d.update(q=self.bits(q[0]),k=self.bits(k[0]),v=self.bits(v[0]))
            return actual
        self.registry.register('sdpa',attention)
        for i,layer in enumerate(self.model.model.layers):
            self.data[i]={}
            def pre(_m,args,i=i):self.data[i]['resid_in']=self.bits(self.rows(args[0]))
            def ain(_m,args,out,i=i):self.data[i]['attention_input']=self.bits(self.rows(out))
            def aout(_m,args,out,i=i):self.data[i]['attention_output']=self.bits(self.rows(out))
            def min_(_m,args,i=i):self.data[i]['mlp_input']=self.bits(self.rows(args[0]))
            def gate(_m,args,out,i=i):self.data[i]['gate']=self.bits(self.rows(out))
            def value(_m,args,out,i=i):self.data[i]['value']=self.bits(self.rows(out))
            def act(_m,args,i=i):self.data[i]['activation']=self.bits(self.rows(args[0]))
            def mout(_m,args,out,i=i):
                self.data[i]['mlp_output']=self.bits(self.rows(out))
                # Flush each layer to bound host RAM. Original BF16 bits retained.
                p=self.folder/f'block_{i:02d}.npz'
                with p.open('wb') as f:np.savez(f,**self.data[i])
                self.addfile(p);self.data[i].clear()
            for module,func,prehook in [(layer,pre,True),(layer.input_layernorm,ain,False),(layer.self_attn.o_proj,aout,False),
                (layer.mlp,min_,True),(layer.mlp.gate_proj,gate,False),(layer.mlp.up_proj,value,False),
                (layer.mlp.down_proj,act,True),(layer.mlp,mout,False)]:
                self.handles.append(module.register_forward_pre_hook(func) if prehook else module.register_forward_hook(func))
        def final(_m,args):self.final_pre=self.bits(self.rows(args[0]))
        self.handles.append(self.model.model.norm.register_forward_pre_hook(final));return self
    def __exit__(self,*_):
        self.registry.register('sdpa',self.original)
        for h in self.handles:h.remove()
    def finish(self,output,item,tok,model_snapshot,run_id,provenance):
        hs=output.hidden_states
        hidden=np.stack([self.bits(x[0,self.positions]) for x in hs]);p=self.folder/'hidden_readouts.npy';np.save(p,hidden);self.addfile(p)
        if self.deep:
            p=self.folder/'hidden_full.npy';fp=np.lib.format.open_memmap(p,mode='w+',dtype=np.uint16,shape=(len(hs),hs[0].shape[1],hs[0].shape[-1]))
            for l,x in enumerate(hs):fp[l]=self.bits(x[0])
            fp.flush();del fp;self.addfile(p)
        logits=output.logits[0,-1].float();top=logits.softmax(-1).topk(12)
        d={'schema':'mechanism_trace.v1','kind':'real','run_id':run_id,'sample_id':item['id'],'entity':item.get('entity','用户输入'),
            'scope':item.get('scope','自定义观察'),'split':item.get('split','unregistered_live'),'model_snapshot':model_snapshot,'precision':'bf16',
            'n_layers':len(hs)-1,'d_model':hs[0].shape[-1],'d_ff':self.model.config.intermediate_size,
            'n_heads':self.model.config.num_attention_heads,'n_kv_heads':self.model.config.num_key_value_heads,'head_dim':self.model.config.head_dim,
            'n_tokens':len(item['token_ids']),'token_ids':item['token_ids'],'tokens':[tok.decode([t]) for t in item['token_ids']],
            'text':item['text'],'prompt':item['prompt'],'deep':self.deep,'readout_positions':[p+self.offset for p in self.positions],
            'entity_position':item.get('entity_position'),'document_colour_position':item.get('document_colour_position'),
            'expected':item.get('expected'),'prior':item.get('prior'),'document':item.get('document'),
            'files':self.files,'reconstruction':self.reconstruction,'final_pre_bits':self.final_pre.tolist(),
            'output_top':[{'token_id':int(i),'text':tok.decode([int(i)]),'probability':float(v)} for v,i in zip(top.values,top.indices)],
            'boundaries':'hidden[0]=actual embedding;hidden[-1]=after final norm; head_z=actual SDPA output before O; Q/K postnorm and postRoPE; V grouped; G/V/A native MLP values',
            'causal_visibility':'earlier entity/cue readouts do not encode the later scope question by construction',
            'attention_source':'rows recomputed from actual Q/K, causal mask and scale; validation compared with actual SDPA z; not a causal effect',
            'generation_step':item.get('generation_step',0),'parent_trace_id':item.get('parent_trace_id'),
            'provenance':provenance}
        d['trace_id']=digest(d);p=self.folder/'manifest.json';save_json(p,d);return d,p
