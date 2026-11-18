"""Sequential CUDA behavior calibration and native capture."""
import argparse,time
import numpy as np
import record_selection_core_v2 as R
from record_selection_core_v2 import C
from shared.dist_protocol.mechanism_trace import GpuLease

def collect(stage):
    p=C.checked();split='discovery' if stage=='pilot' else stage
    if stage=='discovery':
        assert C.read(C.OUT/'pilot-behavior.json')['passed'],'Discovery pilot behavior gate failed'
        check_analysis()
    if stage=='confirm':
        assert C.read(C.OUT/'discovery-behavior.json')['passed'];check_analysis()
        select=C.read(C.OUT/'selection.json');assert select['confirm_unopened']
    material=C.read(C.OUT/(split+'-materials.json'))['rows'];families=p['pilot_families'] if stage=='pilot' else list(range(len(material)//8))
    with GpuLease():
        model=C.load_model(stage);capture=C.Capture(model) if stage=='pilot' else R.Recorder(model)
        scores=[];start=time.monotonic();C.log(stage,0,len(families),start)
        try:
            for n,f in enumerate(families):
                path=C.OUT/f'{stage}-{f:04d}.json'
                if path.exists():
                    saved=C.read(path);assert saved['plan_sha256']==C.sha(C.OUT/'plan.json');scores+=saved['scores'];continue
                records=[];native=[];prefix=None
                for i,row in enumerate(material[f*8:(f+1)*8]):
                    if stage=='pilot':logits,_=capture.forward(row,full=False)
                    else:
                        logits,arrays=capture.run(row)
                        if prefix is None:prefix=arrays['hidden'][:,0].copy()
                        else:assert np.array_equal(prefix,arrays['hidden'][:,0]),'Future factor prefix control failed'
                        name=f'{stage}-{f:04d}-{i}.npz';native.append({'path':name,'sha256':C.save_npz(name,**arrays),'sample_id':row['id'],
                            'qkv_error_max':float(arrays['qkv_relative_error'].max())})
                    records.append({**R.score(logits,row,p),**{k:row[k] for k in ['id','family','frame','valid_position','conflict','language','color','mapping','query']}})
                C.save(path.name,{'plan_sha256':C.sha(C.OUT/'plan.json'),'scores':records,'native':native,'prefix_control_bitwise':stage!='pilot'})
                scores+=records;C.reserve()
                if (n+1)%4==0 or n+1==len(families):C.log(stage,n+1,len(families),start)
                assert time.monotonic()-start<90*60
            gate=R.behavior(scores,p);C.save(stage+'-behavior.json',gate);C.log(stage+'_complete',len(families),len(families),start,passed=gate['passed'],accuracy=gate['strict_accuracy'])
        finally:capture.close()

def check_analysis():
    plan=C.read(C.OUT/'analysis-plan.json')
    assert plan['base_plan_sha256']==C.sha(C.OUT/'plan.json')
    for name,digest in plan['sources'].items():assert C.sha(C.ROOT/name)==digest,name
    return plan

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['pilot','discovery','confirm']);args=parser.parse_args();collect(args.stage)
