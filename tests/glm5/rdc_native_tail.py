"""Original Qwen modules with immutable same-history prefix KV, layer streaming."""
from rdc_construction_common import *


class FrozenPrefix:
    def __init__(self,key,value,block):
        self.key,self.value,self.block=key,value,block

    def update(self,key,value,layer_idx,*args,**kwargs):
        import torch
        assert layer_idx==self.block
        # No state mutation. The current key and value still depend on the
        # current input, including in the JVP/VJP. Only prior tokens are fixed.
        return torch.cat([self.key,key],dim=-2),torch.cat([self.value,value],dim=-2)


def block_call(layer,x,key,value,cos,sin,block):
    return layer(hidden_states=x,attention_mask=None,
        past_key_values=FrozenPrefix(key,value,block),
        position_embeddings=(cos,sin),use_cache=True)


def config():
    from transformers import AutoConfig
    c=AutoConfig.from_pretrained(ROOT/'models/hf'/MODELS['qwen4'],local_files_only=True)
    c._attn_implementation='eager';return c


def checkpoint_tensor(name,dtype,device='cuda'):
    from safetensors import safe_open
    folder=ROOT/'models/hf'/MODELS['qwen4']
    index=read(folder/'model.safetensors.index.json')['weight_map']
    with safe_open(str(folder/index[name]),framework='pt',device='cpu',backend='pread') as f:
        return f.get_tensor(name).to(device=device,dtype=dtype)


def loaded_block(index,dtype):
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    with torch.device('meta'):layer=Qwen3DecoderLayer(config(),index)
    state={name:checkpoint_tensor(f'model.layers.{index}.'+name,dtype) for name in layer.state_dict()}
    layer.load_state_dict(state,assign=True);layer.eval();layer.requires_grad_(False)
    return layer


def final_norm(x,weight,epsilon):
    # Exactly the native Qwen3RMSNorm source operation order. In the smooth
    # reference x is FP32 and there is no BF16 intermediate rounding.
    dtype=x.dtype;y=x.float();y=y*(__import__('torch').rsqrt(y.pow(2).mean(-1,keepdim=True)+epsilon))
    return weight*y.to(dtype)


def cuda_singleton(script_names):
    import psutil
    ancestors={os.getpid(),*(p.pid for p in psutil.Process().parents())}
    for process in psutil.process_iter(['pid','cmdline']):
        if process.info['pid'] not in ancestors and any(Path(a).name in script_names for a in process.info['cmdline'] or []):
            raise RuntimeError('Another scoped CUDA process is running: '+str(process.info['pid']))


CUDA_TASKS={'phase2746_rdc_runtime_capture.py','phase2746_rdc_tail_fixtures.py','phase2746_rdc_tail_differential.py',
    'phase2745_rdc_construction_language.py','phase2745_rdc_construction_capture.py',
    'phase2745_rdc_construction_compile.py','phase2745_rdc_construction_fit.py'}
