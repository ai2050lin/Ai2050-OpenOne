"""Avoid Windows mapped-storage slicing; load identical BF16 bytes one tensor at a time."""
import hashlib
import json
import os
os.environ['PYTHONUTF8']='1'
os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
from pathlib import Path
from time import monotonic

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig,AutoModelForCausalLM
import mlp_condition_design as D
import mlp_condition_run as R

def load(path,**kwargs):
    assert kwargs['dtype']==torch.bfloat16 and kwargs['device_map']=='cuda:0' and kwargs['attn_implementation']=='sdpa'
    config=AutoConfig.from_pretrained(path,local_files_only=True)
    with init_empty_weights():
        model=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16,attn_implementation='sdpa')
    index=D.READ(Path(path)/'model.safetensors.index.json')['weight_map']
    files={};headers={};offsets={};checks=[];start=monotonic()
    try:
        for name in sorted(set(index.values())):
            stream=(Path(path)/name).open('rb');n=int.from_bytes(stream.read(8),'little')
            headers[name]=json.loads(stream.read(n));offsets[name]=8+n;files[name]=stream
        for i,(name,shard) in enumerate(sorted(index.items())):
            entry=headers[shard][name];assert entry['dtype']=='BF16',name
            lo,hi=entry['data_offsets'];f=files[shard];f.seek(offsets[shard]+lo)
            raw=bytearray(f.read(hi-lo));assert len(raw)==hi-lo
            value=torch.frombuffer(raw,dtype=torch.bfloat16).reshape(entry['shape']).to('cuda')
            # Verify every loaded parameter, not a sampled subset.
            native=value.view(torch.uint16).cpu().numpy().tobytes()
            expected=hashlib.sha256(raw).hexdigest();actual=hashlib.sha256(native).hexdigest()
            assert actual==expected,(name,'native bytes differ')
            parent,leaf=name.rsplit('.',1);module=model.get_submodule(parent)
            assert getattr(module,leaf).shape==value.shape,name
            setattr(module,leaf,torch.nn.Parameter(value,requires_grad=False))
            checks.append({'name':name,'bytes':len(raw),'sha256':actual})
            del native,raw,value
            if (i+1)%32==0:print(json.dumps({'stage':'stream_load','done':i+1,'total':len(index),'seconds':round(monotonic()-start,1),'allocated_MiB':round(torch.cuda.memory_allocated()/2**20)}),flush=True)
        model.tie_weights()
        for name,buffer in list(model.named_buffers()):
            assert not buffer.is_meta,name
            parent,leaf=name.rsplit('.',1);model.get_submodule(parent)._buffers[leaf]=buffer.to('cuda')
        assert all(not p.is_meta and p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in model.parameters())
        plan=D.checked()
        R.save('loader-audit.json',{'schema':'mlp_stream_load_audit.v1','loader_sha256':D.HASH(Path(__file__)),
            'checks':checks,'config_sha256':D.HASH(Path(path)/'config.json'),
            'reason':'Windows mapped-storage access violation reproduced with default and sequential HF loading; read raw file bytes and verify each CUDA parameter bitwise',
            'forward_implementation':'unchanged Transformers Qwen3 SDPA BF16; identical weights, config and tokenizer'},plan)
        return model
    finally:
        for stream in files.values():stream.close()

if __name__=='__main__':
    R.AutoModelForCausalLM.from_pretrained=staticmethod(load)
    R.main()
