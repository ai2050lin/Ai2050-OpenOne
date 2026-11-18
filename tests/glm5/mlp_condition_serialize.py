"""Output-only NumPy scalar conversion; frozen analysis and thresholds unchanged."""
from datetime import datetime
import numpy as np
import mlp_condition_v2 as V
import mlp_condition_design as D
import mlp_condition_analysis as A

V.configure();plan=D.checked();save=D.SAVE;converted=[]
def normalize(value,path='root'):
    if isinstance(value,np.generic):
        converted.append({'path':path,'dtype':str(value.dtype)})
        return value.item()
    if isinstance(value,dict):return {k:normalize(v,path+'.'+k) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [normalize(v,path+f'[{i}]') for i,v in enumerate(value)]
    return value
def store(path,doc):
    cleaned=normalize(doc)
    if path.name=='report.json':
        audit={'schema':'mlp_condition_serialization.v1','provenance':plan['provenance'],
            'time':datetime.now().astimezone().isoformat(),'source_sha256':D.HASH(__file__),
            'analysis_sha256':D.HASH(D.ROOT/'tests/glm5/mlp_condition_analysis.py'),
            'converted':converted,'scope':'Only JSON serialization of NumPy scalars; no metric, threshold, input or fitting change'}
        save(D.OUT/'serialization-audit.json',audit)
        cleaned['serialization_audit_sha256']=D.HASH(D.OUT/'serialization-audit.json')
    save(path,cleaned)
D.SAVE=store
A.report(plan)
