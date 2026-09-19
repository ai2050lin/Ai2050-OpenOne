"""Native microbatch arithmetic, reordered across independent requests by layer.

No larger arithmetic batch, merged KV, parameter approximation or new model law.
The only shared object is a temporary copy of the same original layer weights.
"""
import json
import math
import struct
import sys
from rdc_formation_common import *


class TensorSliceReader:
    """Read only one immutable BF16 checkpoint tensor, without mapping its shard.

    Format: https://github.com/safetensors/safetensors#format . This is a
    process-local I/O implementation, not a new parameter format or precision.
    Only local audited BF16/C-order checkpoints are admitted.
    """
    def __init__(self):
        self.headers={};self.reads=0;self.total_bytes=0;self.maximum_tensor_bytes=0

    def metadata(self,path):
        path=Path(path).resolve();stat=path.stat();identity=(stat.st_size,stat.st_mtime_ns)
        if path in self.headers:
            header,offset,old=self.headers[path]
            assert old==identity,('Checkpoint file changed during run',str(path))
            return header,offset
        assert sys.byteorder=='little'
        def unique(pairs):
            result={}
            for key,value in pairs:
                assert key not in result,('Duplicate checkpoint header key',key)
                result[key]=value
            return result
        with path.open('rb') as stream:
            prefix=stream.read(8);assert len(prefix)==8
            n=struct.unpack('<Q',prefix)[0];assert 2<=n<=16*1024**2 and n+8<=stat.st_size
            raw=stream.read(n);assert len(raw)==n and raw[:1]==b'{'
        header=json.loads(raw,object_pairs_hook=unique);offset=8+n;ranges=[]
        for key,entry in header.items():
            if key=='__metadata__':continue
            assert entry['dtype']=='BF16',('Only native BF16 checkpoint tensors admitted',key,entry['dtype'])
            shape=entry['shape'];assert all(isinstance(d,int) and d>=0 for d in shape)
            begin,end=entry['data_offsets']
            assert isinstance(begin,int) and isinstance(end,int) and 0<=begin<=end<=stat.st_size-offset
            assert end-begin==2*math.prod(shape),(key,shape,begin,end)
            ranges.append((begin,end))
        cursor=0
        for begin,end in sorted(ranges):
            assert begin==cursor,('Non-contiguous or overlapping checkpoint data',str(path),begin,cursor)
            cursor=end
        assert cursor==stat.st_size-offset
        self.headers[path]=(header,offset,identity)
        return header,offset

    def tensor(self,path,key):
        import torch
        header,offset=self.metadata(path);entry=header[key];begin,end=entry['data_offsets'];n=end-begin
        buffer=bytearray(n)
        with Path(path).open('rb') as stream:
            stream.seek(offset+begin);view=memoryview(buffer);done=0
            while done<n:
                count=stream.readinto(view[done:]);assert count,('Truncated checkpoint tensor',str(path),key)
                done+=count
        value=torch.frombuffer(buffer,dtype=torch.bfloat16).reshape(entry['shape']) if n else torch.empty(entry['shape'],dtype=torch.bfloat16)
        self.reads+=1;self.total_bytes+=n;self.maximum_tensor_bytes=max(self.maximum_tensor_bytes,n)
        return value


class LayerWeights:
    def __init__(self,device='cuda'):
        from accelerate.utils.offload import OffloadedWeightsLoader
        self.owner=OffloadedWeightsLoader;self.original=OffloadedWeightsLoader.__getitem__
        self.device=device;self.cache={};self.active=False;self.loads=0;self.hits=0;self.bytes=0;self.max_bytes=0
        self.reader=TensorSliceReader()
        def get(owner,key):
            if not self.active:return self.original(owner,key)
            identity=(id(owner),key)
            if identity in self.cache:
                self.hits+=1;return self.cache[identity]
            if key in owner.state_dict:value=owner.state_dict[key]
            elif owner.index[key].get('safetensors_file') is not None:
                info=owner.index[key]
                assert info.get('dtype','bfloat16')=='bfloat16'
                value=self.reader.tensor(info['safetensors_file'],info.get('weight_name',key))
            else:value=self.original(owner,key)
            if value is None:return value
            tensor=value.to(self.device)
            assert tensor.dtype==value.dtype and tensor.shape==value.shape
            self.cache[identity]=tensor;self.loads+=1;self.bytes+=tensor.numel()*tensor.element_size()
            self.max_bytes=max(self.max_bytes,self.bytes)
            assert self.bytes<1024**3,'Only one original decoder layer is retained at a time'
            return tensor
        self.installed=get;self.owner.__getitem__=get
    def begin(self):
        self.cache.clear();self.bytes=0;self.active=True
    def end(self):
        self.cache.clear();self.bytes=0;self.active=False
    def close(self):
        self.end();assert self.owner.__getitem__ is self.installed
        self.owner.__getitem__=self.original


class NativeWave:
    def __init__(self,model,device='cuda'):
        self.model=model;self.core=model.model;self.device=device
        self.config=model.config;self.depth=len(self.core.layers)
        assert self.config.model_type in {'qwen3','glm4'}
        assert not getattr(self.core,'has_sliding_layers',False)
        self.module=sys.modules[self.core.__class__.__module__]
        self.weights=LayerWeights(device);self.forward_waves=0;self.layer_calls=0

    def forward(self,requests):
        """Each request has native B1/B8 shape and its own complete DynamicCache."""
        import torch
        from transformers.cache_utils import DynamicCache
        states=[];self.forward_waves+=1
        with torch.inference_mode():
            for r in requests:
                ids=r['input_ids'];h=self.core.embed_tokens(ids)
                cache=r.get('cache')
                if r['use_cache'] and cache is None:cache=DynamicCache(config=self.config)
                positions=r.get('position_ids')
                if positions is None:
                    seen=cache.get_seq_length() if cache is not None else 0
                    positions=(torch.arange(h.shape[1],device=h.device)+seen).unsqueeze(0)
                mask=self.module.create_causal_mask(config=self.config,inputs_embeds=h,
                    attention_mask=r.get('attention_mask'),past_key_values=cache,position_ids=positions)
                rotary=self.core.rotary_emb(h,position_ids=positions)
                states.append({'h':h,'cache':cache,'position_ids':positions,'mask':mask,'rotary':rotary,
                    'hidden':[bits(h[:,-1])] if r.get('collect_hidden') else None})
            for li,layer in enumerate(self.core.layers):
                self.weights.begin()
                try:
                    for r,s in zip(requests,states):
                        s['h']=layer(s['h'],attention_mask=s['mask'],position_embeddings=s['rotary'],
                            position_ids=s['position_ids'],past_key_values=s['cache'],use_cache=r['use_cache'])
                        if s['hidden'] is not None:s['hidden'].append(bits(s['h'][:,-1]))
                        self.layer_calls+=1
                finally:self.weights.end()
            answer=[]
            for r,s in zip(requests,states):
                h=self.core.norm(s['h'])[:,-1]
                r['cache']=s['cache']
                answer.append({'postnorm':h,'hidden':np.stack(s['hidden'],axis=1) if s['hidden'] is not None else None})
            del states
        return answer

    def close(self):self.weights.close()


def generate_wave(model,tok,batches,engine,progress=None):
    """Reuse original B8 grouping and exact first-B1 operations, full frozen caps."""
    import torch
    device=engine.device
    stop=model.generation_config.eos_token_id or tok.eos_token_id
    stop=set(stop if isinstance(stop,list) else [stop])
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    flat=[r for batch in batches for r in batch]
    b1_requests=[{'input_ids':torch.tensor([r['model_prompt_ids']],device=device),'use_cache':False} for r in flat]
    tick=time.monotonic();b1_outputs=engine.forward(b1_requests);b1={};b1stats={}
    with torch.inference_mode():
        for row,value in zip(flat,b1_outputs):
            h=value['postnorm'];lp=model.lm_head(h).float()[0].double().log_softmax(-1)
            sid=row['sample_id'];b1[sid]=bits(h[0]);b1stats[sid]=[int(lp.argmax()),float(-(lp.exp()*lp).sum()),
                float(-lp[row['model_target_ids'][0]]) if row['model_target_ids'] else 0.]
        del b1_outputs,b1_requests,h,lp
        if progress:progress('first_B1',0,len(flat),time.monotonic()-tick)
        groups=[]
        for rows in batches:
            n,maxlen=len(rows),max(len(r['model_prompt_ids']) for r in rows)
            ids=torch.full((n,maxlen),pad,device=device,dtype=torch.long);mask=torch.zeros_like(ids)
            for bi,row in enumerate(rows):
                ids[bi,-len(row['model_prompt_ids']):]=torch.tensor(row['model_prompt_ids'],device=device)
                mask[bi,-len(row['model_prompt_ids']):]=1
            groups.append({'rows':rows,'done':[False]*n,'finished':False,'tokens':[[] for _ in rows],
                'posts':[[] for _ in rows],'stats':[[] for _ in rows],'fields':[[] for _ in rows],
                'first_nll':[0.]*n,'request':{'input_ids':ids,'attention_mask':mask,
                    'position_ids':(mask.cumsum(-1)-1).clamp_min(0),'use_cache':True,'cache':None,
                    'collect_hidden':any(r['collect_all_hidden'] for r in rows)}})
        for step in range(max(r['max_new_tokens'] for r in flat)):
            active_groups=[g for g in groups if not g['finished']]
            if not active_groups:break
            outputs=engine.forward([g['request'] for g in active_groups])
            for g,value in zip(active_groups,outputs):
                h=value['postnorm'];lp=model.lm_head(h).float().double().log_softmax(-1);choice=lp.argmax(-1)
                post=bits(h)
                for bi,row in enumerate(g['rows']):
                    if g['done'][bi]:continue
                    chosen=int(choice[bi]);g['tokens'][bi].append(chosen);g['posts'][bi].append(post[bi])
                    g['stats'][bi].append([float(-(lp[bi].exp()*lp[bi]).sum()),float(lp[bi,chosen])])
                    if step==0 and row['model_target_ids']:g['first_nll'][bi]=float(-lp[bi,row['model_target_ids'][0]])
                    # Retaining a row view would retain every uncollected B8
                    # row underneath it. Copy the exact selected bytes only.
                    if row['collect_all_hidden']:g['fields'][bi].append(value['hidden'][bi].copy())
                    g['done'][bi]=chosen in stop or step+1==row['max_new_tokens']
                g['finished']=all(g['done']);request=g['request']
                if g['finished']:
                    request['cache']=None
                else:
                    mask=request['attention_mask'];active=torch.tensor([not d for d in g['done']],device=device,dtype=torch.long)
                    mask=torch.cat([mask,active[:,None]],-1);ids=choice[:,None];ids[active==0]=pad
                    request.update(input_ids=ids,attention_mask=mask,position_ids=(mask.sum(-1)-1).clamp_min(0)[:,None])
                del lp,h
            del outputs
            if progress:progress('own_step',step+1,sum(len(v) for g in groups for v in g['tokens']),time.monotonic()-tick)
        answers=[]
        for g in groups:
            packets=[]
            for bi,row in enumerate(g['rows']):
                sid=row['sample_id'];packet={'generated_ids':np.array(g['tokens'][bi],np.int64),
                    'postnorm_BF16':np.stack(g['posts'][bi]),'statistics':np.array(g['stats'][bi]),
                    'first_B1_postnorm_BF16':b1[sid],'first_B1_statistics':np.array(b1stats[sid]),
                    'first_B8_target_NLL':np.array(g['first_nll'][bi]),'first_target_available':np.array(bool(row['model_target_ids']))}
                if g['fields'][bi]:packet['all_hidden_BF16']=np.stack(g['fields'][bi])
                packets.append(packet)
            answers.append(packets)
    return answers,stop
