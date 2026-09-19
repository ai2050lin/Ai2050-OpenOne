"""Lower CPU residency, unchanged BF16 checkpoint, with new numerical admission."""
from rdc_construction_common import *


def load_q14(out):
    import torch
    import transformers.modeling_utils as loading
    import accelerate.utils.offload as runtime
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open
    from accelerate.hooks import remove_hook_from_module
    from accelerate.utils import set_module_tensor_to_device
    from rdc_operator_model import memory
    os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    before=memory()
    assert before['host_available_bytes']>8*1024**3, before
    assert before['system_commit_headroom']>22*1024**3, before
    path=ROOT/'models/hf'/MODELS['qwen14']
    original=loading.safe_open
    def pread(*args,**kwargs):
        kwargs['backend']='pread'
        return safe_open(*args,**kwargs)
    pread._rdc_pread=True
    loading.safe_open=pread;runtime.safe_open=pread
    try:
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True,trust_remote_code=True)
        model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,device_map='auto',
            max_memory={0:'12GiB','cpu':'4GiB'},offload_folder=str(out/'checkpoint_offload_index_4GiB'),
            offload_state_dict=True,offload_buffers=True,local_files_only=True,trust_remote_code=True,
            low_cpu_mem_usage=True,attn_implementation='eager').eval()
    finally:
        loading.safe_open=original
    index=read(path/'model.safetensors.index.json')['weight_map']
    if any(p.device.type!='cuda' for p in model.lm_head.parameters()):
        required=model.config.vocab_size*model.config.hidden_size*2
        assert torch.cuda.mem_get_info()[0]>required+1024**3
        with safe_open(str(path/index['lm_head.weight']),framework='pt',device='cpu',backend='pread') as file:
            value=file.get_tensor('lm_head.weight')
        remove_hook_from_module(model.lm_head)
        set_module_tensor_to_device(model.lm_head,'weight','cuda:0',value=value)
        del value
    # Avoid repeated tiny disk storage reads at final normalization.
    if any(p.device.type!='cuda' for p in model.model.norm.parameters()):
        with safe_open(str(path/index['model.norm.weight']),framework='pt',device='cpu',backend='pread') as file:
            value=file.get_tensor('model.norm.weight')
        remove_hook_from_module(model.model.norm)
        set_module_tensor_to_device(model.model.norm,'weight','cuda:0',value=value)
        del value
    assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    torch.cuda.empty_cache();after=memory()
    assert after['host_available_bytes']>2*1024**3,after
    index_bytes=sum(p.stat().st_size for p in (out/'checkpoint_offload_index_4GiB').rglob('*') if p.is_file())
    assert index_bytes<20*1024**2
    save(out/'low_memory_load_audit.json',{'timestamp':stamp(),'source':snapshot(__file__),
        'model':'qwen14','dtype':str(model.dtype),'quantized':False,'before':before,'after':after,
        'device_map':model.hf_device_map,'GPU_dispatch_GiB':12,'CPU_dispatch_GiB':4,
        'index_bytes':index_bytes,'checkpoint_modified':False,
        'numerical_admission':'Requires all six B1 fixtures at every hidden boundary and postnorm to match original captures before new B8generation.',
        'reason':'Previous6GiB CPU residency failed its10GiB preflight. Reduce actual CPU weight residency by2GiB and retain4GiB overhead reserve; slower disk reads, not quantization or a relaxed same-residency check.'})
    return model,tok
