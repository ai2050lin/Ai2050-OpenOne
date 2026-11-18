"""List all errors of the frozen routing diagnostic, without reselection."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D
OUT=ROOT/'tests/glm5/result/conditional_lookup_native_v1'
seal=D.read(OUT/'candidate.json');rows=D.read(OUT/'confirm-materials.json');head=seal['head'];errors=[];offset=0
for file in sorted(OUT.glob('confirm-??.npz')):
    part=D.read(file.with_suffix('.json'))
    with np.load(file) as z:mass=z['mass'+str(head['layer'])][:,-1,head['head']]
    for j,m in enumerate(mass):
        r=rows[offset+j];pred=int(m[:4].argmax())
        if pred!=r['target_record']:
            errors.append(dict(id=r['id'],family=r['family'],r0=r['r0'],r1=r['r1'],b=r['b'],late=r['late'],content=r['content'],gold_slot=r['target_record'],attention_argmax_slot=pred,mass=m.tolist(),model_score=part['scores'][j]))
    offset+=len(mass)
result=dict(provenance=seal['provenance'],candidate_sha=D.sha(OUT/'candidate.json'),head=head,n_errors=len(errors),examples=errors,scope='all frozen-head source argmax errors; diagnostic errors are not necessarily head computation errors or evidence of compensation')
p=OUT/'routing-counterexamples.json';assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'n_errors':len(errors),'ids':[r['id'] for r in errors]}),flush=True)
