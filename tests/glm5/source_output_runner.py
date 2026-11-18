"""Sequential behavior gates and strictly branch-scoped native collection."""
import argparse,time
import numpy as np
import source_output_core as S
from source_output_core import C
from shared.dist_protocol.mechanism_trace import GpuLease

def check_analysis():
    a=C.read(C.OUT/'analysis-plan.json');assert a['base_plan_sha256']==C.sha(C.OUT/'plan.json')
    for name,digest in a['sources'].items():assert C.sha(C.ROOT/name)==digest,name
    return a

def admitted(split):
    cal=C.read(C.OUT/'behavior-calibration.json')['branches'];other=C.read(C.OUT/f'behavior-{split}.json')['branches']
    return [k for k in cal if cal[k]['passed'] and other[k]['passed']]

def behavior(split):
    p=C.checked();rows=C.read(C.OUT/(split+'-materials.json'))['rows'];nf=len(rows)//24;scores=[];start=time.monotonic()
    if split=='confirm':
        check_analysis();pred=C.read(C.OUT/'predictor-freeze.json');assert pred['confirm_unopened']
        for name,digest in pred['artifacts'].items():assert C.sha(C.OUT/name)==digest
    with GpuLease():
        model=C.load_model('behavior_'+split);cap=C.Capture(model);C.log('behavior_'+split,0,nf,start)
        try:
            for f in range(nf):
                path=C.OUT/f'behavior-{split}-{f:04d}.json'
                if path.exists():
                    meta=C.read(path);assert meta['plan_sha256']==C.sha(C.OUT/'plan.json');scores+=meta['scores'];continue
                group=[]
                for row in rows[f*24:(f+1)*24]:
                    logits,_=cap.forward(row,full=False);group.append({'id':row['id'],**S.score(logits,row,p)})
                C.save(path.name,{'scores':group,'plan_sha256':C.sha(C.OUT/'plan.json')});scores+=group
                if (f+1)%4==0:C.log('behavior_'+split,f+1,nf,start)
                assert time.monotonic()-start<60*60
        finally:cap.close()
    gate=S.behavior(rows,scores,p);C.save('behavior-'+split+'.json',gate)
    C.log('behavior_'+split+'_complete',nf,nf,start,branches={k:v['passed'] for k,v in gate['branches'].items()})

def collect(split):
    p=C.checked();check_analysis();active=admitted(split);rows=C.read(C.OUT/(split+'-materials.json'))['rows'];nf=len(rows)//24;start=time.monotonic()
    C.save('admission-'+split+'.json',{'active_branches':active,'stopped_branches':[k for k in C.read(C.OUT/'behavior-'+split+'.json')['branches'] if k not in active],
        'rule':'whole predeclared task x alphabet branches; never discard individual families or conditions','family_count_per_active_branch':nf,
        'calibration_behavior_sha256':C.sha(C.OUT/'behavior-calibration.json'),'split_behavior_sha256':C.sha(C.OUT/f'behavior-{split}.json')})
    if not active:C.save('native-'+split+'-complete.json',{'n':0,'reason':'All branches failed their behavior gates'});return
    with GpuLease():
        model=C.load_model('native_'+split);cap=S.Capture(model);C.log('native_'+split,0,nf,start)
        try:
            for f in range(nf):
                path=C.OUT/f'native-{split}-{f:04d}.json'
                if path.exists():assert C.read(path)['plan_sha256']==C.sha(C.OUT/'plan.json');continue
                original=C.read(C.OUT/f'behavior-{split}-{f:04d}.json')['scores'];records=[];prefixes={}
                for i,row in enumerate(rows[f*24:(f+1)*24]):
                    if S.branch(row) not in active:continue
                    logits,arrays=cap.run(row);score=S.score(logits,row,p)
                    for key,value in score.items():assert original[i][key]==value,('behavior replay',split,f,i,key)
                    key=(row['task'],row['status_position']);prefix=arrays['hidden'][:,0]
                    if key in prefixes:np.testing.assert_array_equal(prefix,prefixes[key])
                    else:prefixes[key]=prefix.copy()
                    name=f'native-{split}-{f:04d}-{i:02d}.npz';digest=C.save_npz(name,**arrays)
                    records.append({'id':row['id'],'variant':i,'path':name,'sha256':digest,'qkv_error_max':float(arrays['qkv_relative_error'].max())})
                C.save(path.name,{'plan_sha256':C.sha(C.OUT/'plan.json'),'analysis_plan_sha256':C.sha(C.OUT/'analysis-plan.json'),
                    'records':records,'prefix_control_bitwise':True,'behavior_scores_bitwise':True})
                C.reserve()
                if (f+1)%4==0:C.log('native_'+split,f+1,nf,start)
                assert time.monotonic()-start<60*60
        finally:cap.close()
    C.save('native-'+split+'-complete.json',{'n':nf*len(active)*4,'families':nf,'active_branches':active})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['behavior','collect']);parser.add_argument('split',choices=['calibration','confirm']);args=parser.parse_args();globals()[args.stage](args.split)
