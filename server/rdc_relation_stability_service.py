"""Read-only evidence for approximate relational coding, Phase2754."""
import json
from pathlib import Path
import numpy as np
from fastapi import APIRouter,HTTPException

DATA=Path(__file__).resolve().parents[1]/'tests/glm5/result/rdc_relation_stability_20260923'
router=APIRouter(prefix='/relation-stability',tags=['RDC relational regularities'])

def read(name):
    p=DATA/name
    if not p.is_file():raise HTTPException(409,'Evidence not completed')
    return json.loads(p.read_text(encoding='utf-8'))

@router.get('')
def index():return read('index.json')

@router.get('/summary/{model}')
def summary(model:str):
    if model not in ('4B','14B'):raise HTTPException(422,'Unknown model')
    return read(model+'/regularity_summary.json')

@router.get('/probes')
def probes():return read('probe_confirmation.json')

@router.get('/mechanism')
def mechanism():return read('mechanism_summary.json')

@router.get('/order-challenge')
def order_challenge():return read('fact_order_challenge/summary.json')

@router.get('/field/{model}')
def field(model:str,row:int=0,factor:str='truth',layer:int=0):
    if model not in ('4B','14B'):raise HTTPException(422,'Unknown model')
    meta=read(model+'/native_factor_metadata.json')
    if factor not in meta['factors']:raise HTTPException(422,'Unknown factor')
    path=DATA/model/'native_factor_means.npz'
    if not path.is_file():raise HTTPException(409,'Evidence not completed')
    with np.load(path,allow_pickle=False) as z:
        x=z['means']
        if row<0 or row>=len(x) or layer<0 or layer>=x.shape[2]:raise HTTPException(422,'Index outside captured range')
        return dict(model=model,row=meta['rows'][row],factor=factor,layer=layer,shape=list(x.shape),values=x[row,meta['factors'].index(factor),layer].tolist(),scope=meta['coordinates'])

@router.get('/units/{layer}')
def units(layer:int,row:int=0):
    meta=read('mechanism_summary.json');key=str(layer)
    if key not in meta['units']:raise HTTPException(422,'Layer was not collected at unit level')
    path=DATA/'all_unit_truth_contributions.npz'
    if not path.is_file():raise HTTPException(409,'Evidence not completed')
    with np.load(path,allow_pickle=False) as z:
        values=z[key]
        if row<0 or row>=len(values):raise HTTPException(422,'Row outside captured range')
        return dict(layer_zero_based=layer,row=meta['unit_rows'][row],values=values[row].tolist(),unit_count=values.shape[1],
            scope='All actual SwiGLU units projected through real down-projection weights onto fixed readout direction; observational attribution, not a unique causal circuit.')

@router.get('/worlds/{world_id}')
def world(world_id:str):
    mat=read('material.json');w=next((w for w in mat['worlds'] if w['id']==world_id),None)
    if w is None:raise HTTPException(404,'Unknown world')
    return dict(world=w,rows=[r for r in mat['rows'] if r['world']==world_id],sampling_unit='world;16 repeated condition/template observations')
