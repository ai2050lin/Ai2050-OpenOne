"""CPU-only bounded safetensors reads. No global connected-model state or inference."""
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

router=APIRouter(prefix='/api/parameters',tags=['parameters'])


def roots():
    repo=Path(__file__).resolve().parents[1]
    paths=[repo/'models',repo/'models'/'hf',Path(os.environ.get('AI2050_NATIVE_DIR','D:/ai2050_native'))/'models'/'hf',Path('D:/develop/model')]
    if os.environ.get('AI2050_MODELS_DIR'):paths.insert(0,Path(os.environ['AI2050_MODELS_DIR']).expanduser())
    return list(dict.fromkeys(paths))


from .parameter_reader import catalog as read_catalog, header, read_slice, model_snapshot

def catalog():
    return read_catalog(roots())

def lookup(model):
    found=next((m for m in catalog() if m['id']==model),None)
    if not found:raise HTTPException(404,'模型不在已配置的本机模型目录中')
    return found


@router.get('/models')
def models():
    return {'models':[{k:v for k,v in m.items() if k!='_path'} for m in catalog()],
            'source':'local_config_and_safetensors','inference':False}


@router.get('/identity')
def identity(model:str):
    m=lookup(model)
    try:return {'model':model,**model_snapshot(m['_path'])}
    except (ValueError,OSError) as exc:raise HTTPException(422,str(exc)) from exc


@router.get('/tensors')
def tensors(model:str,block:int=Query(0,ge=0),embedding:bool=False):
    m=lookup(model);out=[]
    for path in sorted(m['_path'].glob('*.safetensors')):
        h,_,fingerprint=header(path)
        for name,meta in h.items():
            if (embedding and 'embed_tokens' in name) or (not embedding and f'layers.{block}.' in name):
                out.append({'name':name,'shape':meta['shape'],'dtype':meta['dtype'],'header_sha256':fingerprint})
    return {'model':model,'block':block,'tensors':out,'source':'safetensors_header','model_snapshot':None,
            'note':'文件头指纹不是完整权重快照哈希；数值切片在CPU直接读原文件。'}


@router.get('/slice')
def tensor_slice(model:str,name:str,row:int=Query(0,ge=0),col:int=Query(0,ge=0),rows:int=Query(8,ge=1,le=64),cols:int=Query(8,ge=1,le=64)):
    m=lookup(model)
    try:return {'model':model,**read_slice(m['_path'],name,row,col,rows,cols)}
    except FileNotFoundError as exc:raise HTTPException(404,'找不到参数') from exc
    except (ValueError,OSError) as exc:raise HTTPException(422,str(exc)) from exc
