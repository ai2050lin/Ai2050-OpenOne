"""Read-only Phase2752 evidence API; no model loading or job execution."""
import json
from pathlib import Path
import numpy as np
from fastapi import APIRouter, HTTPException

DATA=Path(__file__).resolve().parents[1]/'tests/glm5/result/rdc_context_interaction_20260923'
router=APIRouter(prefix='/context-interaction',tags=['RDC conditional interaction'])


def read(path):
    if not path.is_file():
        raise HTTPException(409,'Artifact has not completed')
    return json.loads(path.read_text(encoding='utf-8'))


def directory(model,control):
    if model not in ('4B','14B') or control not in ('primary','assertion'):
        raise HTTPException(422,'Unknown model or control')
    if control=='assertion' and model!='4B':
        raise HTTPException(422,'Assertion control was only allocated to4B')
    return (DATA if control=='primary' else DATA/'assertion_control')/model


@router.get('')
def index():
    return read(DATA/'index.json')


@router.get('/claims')
def claims():
    return read(DATA/'reference_claims.json')


@router.get('/models/{model}')
def summary(model:str,control:str='primary'):
    return read(directory(model,control)/'prediction_summary.json')


@router.get('/models/{model}/field')
def field(model:str,control:str='primary',kind:str='observed',layer:int=0):
    dest=directory(model,control)
    allowed={'observed','zero','family_mean','surface','graph_only','graph_surface','base_coordinate','source_product','hybrid'}
    if kind not in allowed:
        raise HTTPException(422,'Unknown field kind')
    path=dest/'full_coordinate_fields.npz'
    if not path.is_file():
        raise HTTPException(409,'Field has not completed')
    with np.load(path,allow_pickle=False) as z:
        x=z[kind]
        if layer<0 or layer>=len(x):
            raise HTTPException(422,'Layer outside captured range')
        return dict(model=model,control=control,kind=kind,layer=layer,shape=list(x.shape),values=x[layer].tolist(),
                    metadata=read(dest/'field_metadata.json'))


@router.get('/worlds/{world_id}')
def world(world_id:str,control:str='primary'):
    if control not in ('primary','assertion'):
        raise HTTPException(422,'Unknown control')
    data=read((DATA if control=='primary' else DATA/'assertion_control')/'material.json')
    found=next((w for w in data['worlds'] if w['id']==world_id),None)
    if found is None:
        raise HTTPException(404,'Unknown world identity')
    rows=[r for r in data['rows'] if r['world']==world_id]
    graph=next((g for g in read(DATA/'external_graphs.json')['graphs'] if g['world']==world_id),None)
    return dict(world=found,graph=graph,rows=rows,scope='Generator annotations and actual inputs, not inferred internal concepts',
                sampling_unit='world; wordings and conditions are repeated measurements')
