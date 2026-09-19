"""Original-BF16 model dispatch with Windows commit headroom and bounded GPU cache."""
import os
import sys
import ctypes
from rdc_operator_common import *
from phase2721_rdc_joint_scale import MODELS


def memory():
    import psutil
    result={'host_available_bytes':psutil.virtual_memory().available,'process_private_bytes':psutil.Process().memory_info().private if os.name=='nt' else psutil.Process().memory_info().rss}
    if os.name=='nt':
        class Status(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('totalphys','availphys','totalpage','availpage','totalvirtual','availvirtual','availext')]
        s=Status();s.length=ctypes.sizeof(s)
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        result.update(system_commit_limit=s.totalpage,system_commit_headroom=s.availpage)
    return result


def load(key,out,cpu_gib=6,gpu_limit=None):
    if key=='qwen4':
        from phase2721_rdc_joint_scale import load as previous
        return previous(key,out)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import transformers.modeling_utils as loading
    from safetensors import safe_open
    torch.set_num_threads(2);os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
    assert cpu_gib in (6,11)
    before=memory();requested=cpu_gib;gpu_gib=13 if requested==11 else 12
    if gpu_limit is not None:
        assert requested==6 and (gpu_limit==6 or (key=='glm4' and gpu_limit==11)), \
            'Only registered6GPU/6CPU or GLM11GPU/6CPU layer-wave profiles are admitted; renewed native replay is required'
        gpu_gib=gpu_limit
    if requested==11:
        eligible=[v for v in (11,9,6) if before['host_available_bytes']>(v+4)*1024**3 and before.get('system_commit_headroom',100*1024**3)>(v+22)*1024**3]
        assert eligible,('No commit-safe QA residency fits',before)
        cpu_gib=eligible[0]
    assert before['host_available_bytes']>10*1024**3, before
    assert before.get('system_commit_headroom',100*1024**3)>24*1024**3, before
    save(out/'resource_check.json',{'timestamp':stamp(),**before,'GPU_dispatch':f'{gpu_gib}GiB','CPU_dispatch':f'{cpu_gib}GiB','requested_max_CPU_GiB':requested,'quantization':False,
        'layer_wave_GPU_reserve':gpu_limit is not None,
        'reason':f'Explicit{gpu_gib}GPU/6CPU reserves device memory for independent native-B8 KV caches; altered request order and changed residency require complete historical-fixture/original-pilot replay. Original parameters and kernels are unchanged; the memory profile alone does not certify numerical equivalence.' if gpu_limit is not None else 'Original13GPU/9CPU capture encountered commit exhaustion with old API cached. Conservative12GPU/6CPU plus cache release/API pause passed128sources;12GPU/9CPU QA passed exact replay but is disk-bound. QA v3 requests13GPU/up-to11CPU with measured host>(CPU+4)GiB and commit>(CPU+22)GiB. Native replays required; no user process or system settings changed.'})
    path=ROOT/'models/hf'/MODELS[key]
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True,trust_remote_code=True)
    original=loading.safe_open
    def pread(*a,**kw):kw['backend']='pread';return original(*a,**kw)
    loading.safe_open=pread
    try:
        model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,device_map='auto',max_memory={0:f'{gpu_gib}GiB','cpu':f'{cpu_gib}GiB'},
            offload_folder=str(out/'checkpoint_offload_index'),offload_state_dict=True,offload_buffers=True,
            local_files_only=True,trust_remote_code=True,low_cpu_mem_usage=True,attn_implementation='eager').eval()
    finally:loading.safe_open=original
    disk=sum(p.stat().st_size for p in (out/'checkpoint_offload_index').rglob('*') if p.is_file())
    assert disk<20*1024**2
    head_resident=False
    if model.hf_device_map.get('lm_head')=='disk':
        from accelerate.hooks import remove_hook_from_module
        from accelerate.utils import set_module_tensor_to_device
        index=read(path/'model.safetensors.index.json')['weight_map'];headbytes=model.config.vocab_size*model.config.hidden_size*2
        free,total=torch.cuda.mem_get_info();assert free>headbytes+1024**3
        with safe_open(str(path/index['lm_head.weight']),framework='pt',device='cpu',backend='pread') as f:head=f.get_tensor('lm_head.weight')
        remove_hook_from_module(model.lm_head);set_module_tensor_to_device(model.lm_head,'weight','cuda:0',value=head)
        del head;head_resident=True
    assert not getattr(model,'is_quantized',False) and model.dtype==torch.bfloat16
    torch.cuda.empty_cache();after=memory();assert after['host_available_bytes']>2*1024**3
    model._rdc_load_profile=f'GPU{gpu_gib}_CPU{cpu_gib}'
    save(out/'load_audit.json',{'timestamp':stamp(),'model':key,'dtype':str(model.dtype),'quantized':False,'device_map':model.hf_device_map,
        'memory':after,'offload_index_bytes':disk,'head_original_BF16_GPU_resident':head_resident,
        'GPU_allocated':torch.cuda.memory_allocated(),'GPU_reserved':torch.cuda.memory_reserved(),
        'method':f'Original checkpoint references;{gpu_gib}GiB GPU/{cpu_gib}GiB CPU dispatch with actual host/commit preflight. Every block remains BF16; no new checkpoint copies.',
        'model_source_sha':sha(Path(sys.modules[model.model.__class__.__module__].__file__)),'loader':snapshot(Path(__file__))})
    return model,tok
