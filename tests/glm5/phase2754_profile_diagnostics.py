"""Descriptive replication of all-layer native source profiles; noTopK."""
import json
from pathlib import Path
import numpy as np
from phase2754_relation_stability import OUT,FAMILIES,write,sha,now,snapshot
from phase2754_analysis import metadata,source,bootstrap

def run():
    src=source();mm=metadata('4B','discovery')+metadata('4B','confirmation');groups={}
    for m in mm:groups.setdefault(m['group'],[]).append(m)
    records=[];vectors=[]
    for gid,rr in groups.items():
        a=src[rr[0]['id']];v=np.mean([src[m['id']]['truth_sign']*np.stack([m['source_numerator']['attention'],m['source_numerator']['mlp']])/m['rms'] for m in rr],axis=0)
        records.append(dict(group=gid,world=a['world'],family=a['family'],split=a['split']));vectors.append(v)
    x=np.stack(vectors);train={f:x[[i for i,r in enumerate(records) if r['split']=='train' and r['family']==f]].mean(0) for f in FAMILIES}
    output=[]
    for r,v in zip(records,x):
        if r['split'] in ('train','validation'):continue
        mu=train[r['family']];cos=float(np.sum(v*mu)/(np.linalg.norm(v)*np.linalg.norm(mu)))
        output.append(dict(**r,cos=cos,attention_total=float(v[0].sum()),mlp_total=float(v[1].sum()),relative_error=float(np.linalg.norm(v-mu)/np.linalg.norm(v))))
    summary={s:{k:bootstrap([r for r in output if r['split']==s],k) for k in ('cos','relative_error','attention_total','mlp_total')} for s in ('entity','surface','depth')}
    fg=json.loads((OUT/'4B/field_generalization.json').read_text(encoding='utf-8'));geometry={}
    for s in ('entity','surface','depth'):
        geometry[s]={k:float(np.mean([r['cos'][36] for r in fg if r['split']==s and r['factor']==k and r['nonzero'][36]])) for k in ('predicate','truth')}
    write(OUT/'source_profile_diagnostic.json',dict(created_utc=now(),source=snapshot(Path(__file__)),status='Post-confirmation descriptive profile analysis, not a newly preregistered heldout predictor',
        source_scope='All36attention plus36MLP source contributions, native model4B and fixed output readout; train-family mean prototypes.',splits=summary,final_field_cos=geometry,
        limits='Shared yes/no readout may itself induce shared source patterns. This is not a universal language circuit or cross-model coordinate alignment.'))
    np.savez(OUT/'source_profile_prototypes.npz',**train)
    print(summary,flush=True)

if __name__=='__main__':run()
