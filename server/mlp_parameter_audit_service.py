"""Verified whole-unit parameter arithmetic, separate from functional interpretation."""
import json
from pathlib import Path
import numpy as np
from fastapi import APIRouter,HTTPException
from shared.dist_protocol.mechanism_trace import file_sha
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tests/glm5/result/mlp_atlas_parameter_audit_v1'
router=APIRouter(prefix='/mlp-parameter-audit',tags=['native-parameter-audit']);_verified={}
def source(block):
    if not 0<=block<36:raise HTTPException(422,'块越界')
    p=OUT/'audit-result.json'
    if not p.exists():raise HTTPException(409,'全层参数审计尚未完成')
    d=json.loads(p.read_text(encoding='utf-8'));p=OUT/f'B{block:02d}.npz';stamp=(p.stat().st_size,p.stat().st_mtime_ns)
    if _verified.get(str(p))!=stamp:
        if file_sha(p)!=d['layers'][block]['sha256']:raise HTTPException(409,'参数审计数组哈希不符')
        _verified[str(p)]=stamp
    return d,p

@router.get('/layers/{block}')
def layer(block:int):
    d,p=source(block)
    with np.load(p) as z:values={k:z[k].tolist() for k in ['Gate_term_RMS','Value_term_RMS','rounding_residual_RMS','G_relative_RMS','V_relative_RMS','A_relative_RMS']}
    return {'kind':'real','model_snapshot':d['model_snapshot'],'block':block,'d_ff':9728,'source_sha256':d['layers'][block]['sha256'],**values}

@router.get('/units/{block}/{unit}')
def unit(block:int,unit:int):
    if not 0<=unit<9728:raise HTTPException(422,'单元越界')
    d,p=source(block)
    with np.load(p) as z:values={k:z[k][...,unit].tolist() for k in z.files if not k.startswith('output_') or k=='output_weight_axis_norm'}
    return {'kind':'real','model_snapshot':d['model_snapshot'],'block':block,'unit':unit,'source_sha256':d['layers'][block]['sha256'],'scope':d['plan']['interpretation'],**values}
