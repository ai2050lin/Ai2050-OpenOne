"""Explicit restricted-scope amendment after failed discovery calibration.

Raw fields are archived for every condition; failed behavior cells and their
whole paired families are excluded from candidate fitting and positive claims.
The original failed gates are immutable, and this is not a full-scope pass.
"""
import argparse,time
import numpy as np
import record_selection_core_v3 as R
from record_selection_core_v3 import C
from record_selection_runner_v3 import check_analysis
from shared.dist_protocol.mechanism_trace import GpuLease

def eligible(rows,gate):
    bad={k for k,v in gate['cells'].items() if v['strict_accuracy']<.9}
    def key(r):return ':'.join(str(r[k]) for k in ['frame','valid_position','conflict','language'])
    return [all(key(r) not in bad for r in rows[f:f+8]) for f in range(0,len(rows),8)]

def collect(split):
    p=C.checked();check_analysis();rows=C.read(C.OUT/(split+'-materials.json'))['rows'];n=len(rows)//8
    if split=='confirm':assert C.read(C.OUT/'selection.json')['confirm_unopened']
    start=time.monotonic();scores=[]
    with GpuLease():
        model=C.load_model('restricted_'+split);capture=R.Recorder(model);C.log(split,0,n,start)
        try:
            for f in range(n):
                path=C.OUT/f'{split}-{f:04d}.json'
                if path.exists():
                    saved=C.read(path);assert saved['plan_sha256']==C.sha(C.OUT/'plan.json');scores+=saved['scores'];continue
                records=[];native=[];prefix=None
                for i,row in enumerate(rows[f*8:(f+1)*8]):
                    logits,arrays=capture.run(row)
                    if prefix is None:prefix=arrays['hidden'][:,0].copy()
                    else:assert np.array_equal(prefix,arrays['hidden'][:,0])
                    name=f'{split}-{f:04d}-{i}.npz';native.append({'path':name,'sha256':C.save_npz(name,**arrays),'sample_id':row['id'],
                        'qkv_error_max':float(arrays['qkv_relative_error'].max())})
                    records.append({**R.score(logits,row,p),**{k:row[k] for k in ['id','family','frame','valid_position','conflict','language','color','mapping','query']}})
                C.save(path.name,{'plan_sha256':C.sha(C.OUT/'plan.json'),'analysis_plan_sha256':C.sha(C.OUT/'analysis-plan.json'),
                    'scores':records,'native':native,'prefix_control_bitwise':True,'scope':'raw archive; behavior eligibility applied before feature analysis'})
                scores+=records;C.reserve()
                if (f+1)%4==0:C.log(split,f+1,n,start)
                assert time.monotonic()-start<90*60
            gate=R.behavior(scores,p);C.save(split+'-behavior.json',gate)
            keep=eligible(rows,gate);C.save(split+'-admission.json',{'eligible_families':[i for i,k in enumerate(keep) if k],
                'excluded_families':[i for i,k in enumerate(keep) if not k],'whole_scope_passed':gate['passed'],
                'rule':'exclude whole eight-condition family if any cell failed unchanged .90 behavior threshold',
                'selection_bias':'conditional subset is not an independent demonstration of full-scope capability',
                'behavior_sha256':C.sha(C.OUT/(split+'-behavior.json'))})
            C.log(split+'_complete',n,n,start,passed=gate['passed'],accuracy=gate['strict_accuracy'])
        finally:capture.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('split',choices=['discovery','confirm']);collect(parser.parse_args().split)
