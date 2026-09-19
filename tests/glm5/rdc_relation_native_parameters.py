"""Read-only bit-preserving BF16 checkpoint views; no model or GPU allocation."""
import json,struct
import numpy as np
from pathlib import Path


def parameter(root,key,model_name='qwen3-4b'):
    if model_name not in ('qwen3-4b','Qwen3-14B','glm4-9b-chat-hf'):raise ValueError('Unregistered local model')
    folder=Path(root)/'models/hf'/model_name
    index=json.loads((folder/'model.safetensors.index.json').read_text(encoding='utf-8'))['weight_map']
    if key not in index:raise KeyError(key)
    path=folder/index[key]
    with path.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size));item=header[key]
    if item['dtype']!='BF16':raise ValueError(('Expected original BF16',key,item['dtype']))
    return np.memmap(path,mode='r',dtype='<u2',offset=8+size+item['data_offsets'][0],shape=tuple(item['shape']))


def decode(value):return (np.asarray(value).astype(np.uint32)<<16).view(np.float32)
