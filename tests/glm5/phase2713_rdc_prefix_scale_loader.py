"""Bounded nonquantized sequential loader with explicit host reserve and original-shard offload."""
import os
import sys
from rdc_prefix_common import *
MODELS={'qwen14':'Qwen3-14B','glm4':'glm4-9b-chat-hf'}


def load(key,out):
    import torch
    import psutil
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import transformers.modeling_utils as loading
    torch.set_num_threads(4);os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
    available=psutil.virtual_memory().available
    assert available>11*1024**3,('Need8GiB host allocation plus3GiB reserve',available)
    path=ROOT/'models/hf'/MODELS[key];before=usage()
    save(out/'load_policy.json',{'timestamp':stamp(),'source_sha':sha(Path(__file__)),'host_available_before':available,
      'GPU_allocation':'12GiB','host_allocation':'8GiB','minimum_preload_host':'11GiB',
      'reason':'The prior 10GiB CPU/13GiB available guard was not met after imports. Reduce CPU residency by2GiB and keep3GiB pre-load reserve, retaining BF16 and automatic device map.',
      'offload':'Installed Transformers safetensors disk-offload uses existing original checkpoint shards, not duplicated weights. Verify newly generated offload directory stays small.',
      'other_apps':'Only this task owned API42392/68580 was temporarily stopped; user frontend and browser/editor applications were not stopped.'})
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True,trust_remote_code=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    previous=loading.safe_open
    def pread(*a,**kw):kw['backend']='pread';return previous(*a,**kw)
    loading.safe_open=pread
    try:
        model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,device_map='auto',max_memory={0:'12GiB','cpu':'8GiB'},
          offload_folder=str(out/'checkpoint_offload_index'),offload_state_dict=True,offload_buffers=True,
          local_files_only=True,trust_remote_code=True,low_cpu_mem_usage=True,attn_implementation='eager').eval()
    finally:loading.safe_open=previous
    assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    disk_bytes=sum(p.stat().st_size for p in (out/'checkpoint_offload_index').rglob('*') if p.is_file())
    assert disk_bytes<20*1024**2,(disk_bytes,'Unexpected weight duplication')
    save(out/'load_audit.json',{'timestamp':stamp(),'host_available_after':psutil.virtual_memory().available,
      'device_map':getattr(model,'hf_device_map',{}),'dtype':str(model.dtype),'quantized':False,
      'new_offload_directory_bytes':disk_bytes,'campaign_growth_during_load':usage()-before,
      'native_source_sha':sha(Path(sys.modules[model.model.__class__.__module__].__file__))})
    guard();return model,tok
