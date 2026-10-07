"""Read-only, scoped Phase2753 evidence queries."""
import json
from pathlib import Path
import numpy as np
from fastapi import APIRouter,HTTPException

DATA=Path(__file__).resolve().parents[1]/'tests/glm5/result/rdc_early_interaction_20260923'
router=APIRouter(prefix='/early-interaction',tags=['RDC early forecasts'])

def read(name):
    path=DATA/name
    if not path.is_file():raise HTTPException(409,'Evidence has not completed')
    return json.loads(path.read_text(encoding='utf-8'))

@router.get('')
def index():return read('index.json')

@router.get('/summary')
def summary():return read('confirmation_summary.json')

@router.get('/readout')
def readout():return read('readout_summary.json')

@router.get('/field')
def field(kind:str='observed',row:int=0):
    if kind not in ('observed','predicted','residual'):raise HTTPException(422,'Unknown field kind')
    path=DATA/'full_coordinate_fields.npz'
    if not path.is_file():raise HTTPException(409,'Evidence has not completed')
    with np.load(path,allow_pickle=False) as z:
        values=z[kind]
        if row<0 or row>=len(values):raise HTTPException(422,'Row outside captured range')
        return dict(kind=kind,row=row,shape=list(values.shape),values=values[row].tolist(),metadata=read('field_metadata.json'))

@router.get('/worlds/{world_id}')
def world(world_id:str):
    mat=read('material.json'); item=next((w for w in mat['worlds'] if w['id']==world_id),None)
    if item is None:raise HTTPException(404,'Unknown world')
    rows=[r for r in mat['rows'] if r['world']==world_id]
    return dict(world=item,rows=rows,annotations=[a for a in read('confirmation_annotations.json') if a['group'] in {r['group'] for r in rows}],
        sampling_unit='world, not8 independent prompts',scope='Supplied graph annotations; fixed synthetic relation families')
