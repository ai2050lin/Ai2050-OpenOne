"""Prespecified supplemental native-direction amplitude control analysis."""
import sys
import numpy as np
import record_selection_core_v3 as R
from record_selection_core_v3 import C
from record_selection_analysis import admission
from state_formation_math import bootstrap

def freeze():
    assert not list(C.OUT.glob('causal-[0-9]*.json'))
    C.save('controls-analysis-freeze.json',{'source_sha256':C.sha(__file__),'amendment_sha256':C.sha(C.OUT/'controls-amendment.json'),
        'scope':'supplemental diagnostics, no primary candidate or threshold change; fixed mean over four controls'})

def report():
    frozen=C.read(C.OUT/'controls-analysis-freeze.json');assert frozen['source_sha256']==C.sha(__file__)
    assert frozen['amendment_sha256']==C.sha(C.OUT/'controls-amendment.json')
    keep=admission('confirm');rows=C.read(C.OUT/'confirm-materials.json')['rows'];frames=np.array([rows[f*8]['frame'] for f in range(384)])[keep];result=[]
    for candidate in C.read(C.OUT/'selection.json')['candidates']:
        loss=[];controls=[];scales=[];errors=[];individual=[];actual_ratios=[]
        for f in np.where(keep)[0]:
            item=C.read(C.OUT/f'causal-{f:04d}.json');rec=next(z for z in item['records'] if z['candidate']==candidate['id'] and z['method']=='ablation' and z['dose']==1)
            base=item['baseline']['target_margin'];loss.append(base-rec['edited']['target_margin'])
            values=[base-z['matched']['target_margin'] for z in rec['group_controls']];controls.append(float(np.mean(values)));individual.append(values)
            scales.extend(z['scale'] for z in rec['group_controls']);errors.extend(z['matched_norm_relative_error'] for z in rec['group_controls'])
            actual_ratios.extend(z['realized_norm']/rec['realized_norm'] for z in rec['group_controls'])
        gain=np.array(loss)-controls
        result.append({'candidate':candidate['id'],'n':len(loss),'candidate_loss':float(np.mean(loss)),'control_mean_loss':float(np.mean(controls)),
            'individual_control_mean_loss':np.mean(individual,axis=0).tolist(),'excess':bootstrap(gain,frames,alpha=.0125),
            'positive_fraction':float(np.mean(gain>0)),'scale_min_median_max':np.quantile(scales,[0,.5,1]).tolist(),
            'unscaled_realized_norm_ratio_min_median_max':np.quantile(actual_ratios,[0,.5,1]).tolist(),'maximum_norm_error':max(errors),
            'scope':'conditional descriptive intervention; residual direction matching is not native group ablation; no semantic confirmation'})
    C.save('matched-control-report.json',{'schema':'record_selection_matched_controls.v1','results':result,
        'analysis_freeze_sha256':C.sha(C.OUT/'controls-analysis-freeze.json'),'formal_promotion':False})

if __name__=='__main__':{'freeze':freeze,'report':report}[sys.argv[-1]]()
