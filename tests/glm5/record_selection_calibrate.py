"""Behavior-only expansion of the prespecified discovery material, no features."""
import time
from datetime import datetime
import record_selection_core_v3 as R
from record_selection_core_v3 import C
from shared.dist_protocol.mechanism_trace import GpuLease

def main():
    p=C.checked();assert not list(C.OUT.glob('confirm-[0-9]*.npz'))
    C.save('calibration-expansion.json',{'schema':'record_selection_calibration_expansion.v1','created_at':datetime.now().astimezone().isoformat(),
        'base_plan_sha256':C.sha(C.OUT/'plan.json'),'source_sha256':C.sha(__file__),'pilot_sha256':C.sha(C.OUT/'pilot-behavior.json'),
        'reason':'pilot has only four rows per frozen gate cell; estimate behavior on all128 discovery families before any internal selection',
        'threshold_unchanged':.9,'new_confirm_exposure':0,'native_features_collected':False,'reuse_scope':'old32 pilot families are included, not counted as new independent cases'})
    material=C.read(C.OUT/'discovery-materials.json')['rows'];scores=[];start=time.monotonic()
    with GpuLease():
        model=C.load_model('calibration_full');cap=C.Capture(model)
        try:
            for f in range(128):
                rows=[]
                for row in material[f*8:(f+1)*8]:
                    logits,_=cap.forward(row,full=False)
                    rows.append({**R.score(logits,row,p),**{k:row[k] for k in ['id','family','frame','valid_position','conflict','language','color','mapping','query']}})
                C.save(f'calibration-full-{f:04d}.json',{'scores':rows,'calibration_sha256':C.sha(C.OUT/'calibration-expansion.json')});scores+=rows
                if (f+1)%8==0:C.log('calibration_full',f+1,128,start)
        finally:cap.close()
    gate=R.behavior(scores,p);C.save('calibration-full-behavior.json',gate);print({'passed':gate['passed'],'accuracy':gate['strict_accuracy']},flush=True)

if __name__=='__main__':main()
