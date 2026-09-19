"""Measured QA-only residency refinement after successful conservative larger-model capture."""
from rdc_operator_common import *

def main():
    out=BASE/'qa/qwen14/main';path=out/'residency_refinement.json'
    if path.exists():return
    commits=[read(p) for p in (out/'commits').glob('*.json')]
    save(path,{'timestamp':stamp(),'source':snapshot(Path(__file__)),'original_CPU_GiB':6,'new_CPU_GiB':11,'GPU_GiB':12,'BF16_unchanged':True,
        'completed_questions':len(commits),'first_two_progress_seconds':[162.6,261.4],
        'prestop_measured':{'system_committed_bytes':56761143296,'commit_limit':70173745152,'host_available_bytes':10914512896},
        'action':'Stopped only verified own Q14QA child75760/parent40852. All completed commits and pending native trace files kept. Reuse same64cases and scoring; one completed full trace and generated sequence must replay exactly on new residency before continuing.',
        'accounting':'Last flushed261.4seconds is lower-bound elapsed; subsequent partly completed question time is separately bounded by wall timestamps. This is a resource/performance refinement, not a scientific rerun selection.'})
    ledger('Q14_QA6CPU_first_attempt_lower_bound',261.4,completed_questions=len(commits))

if __name__=='__main__':main()
