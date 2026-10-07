"""Read-only evidence and numerical summaries. Never starts inference or imports torch."""
import json
from pathlib import Path
import numpy as np
from fastapi import APIRouter,HTTPException

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'tests/glm5/result/rdc_trusted_rebuild_20260923'
router=APIRouter(prefix='/trusted-rebuild',tags=['RDC evidence repair'])


def read_json(path):
    if not path.is_file():raise HTTPException(409,'This artifact has not been completed')
    return json.loads(path.read_text(encoding='utf-8'))


def model_dir(model):
    if model not in ('4B','14B'):raise HTTPException(422,'Model must be 4B or 14B')
    return DATA/model


@router.get('')
def index():
    return read_json(DATA/'index.json')


@router.get('/claims')
def claims():
    return read_json(DATA/'evidence/claims.json')


@router.get('/claims/{claim_id}')
def claim(claim_id:str):
    data=claims()
    for row in data['claims']:
        if row['id']==claim_id:return row
    raise HTTPException(404,'Unknown registered claim ID')


@router.get('/models/{model}')
def summary(model:str):
    return read_json(model_dir(model)/'summary.json')


@router.get('/models/{model}/field')
def field(model:str,kind:str='interaction_mean',layer:int=0):
    directory=model_dir(model)
    if kind not in ('raw_mean','interaction_mean'):raise HTTPException(422,'Unknown field')
    path=directory/'heatmap_values.npz'
    if not path.is_file():raise HTTPException(409,'Field not completed')
    with np.load(path,allow_pickle=False) as z:
        data=z[kind]
        if layer<0 or layer>=len(data):raise HTTPException(422,'Layer index out of range')
        values=data[layer].tolist();shape=list(data.shape)
    return dict(model=model,kind=kind,layer=layer,shape=shape,values=values,
        metadata=read_json(directory/'heatmap_metadata.json'))
