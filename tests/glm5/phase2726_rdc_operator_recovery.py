"""Record the measured first14B attempt failure without discarding committed data."""
from rdc_operator_common import *

def main():
    out=BASE/'scale/qwen14';target=out/'commit_recovery.json'
    if target.exists():return
    save(target,{'timestamp':stamp(),'source':snapshot(Path(__file__)),
        'first_attempt_start':read(out/'main/resource_check.json')['timestamp'],'committed_sources_before_interrupt':len(list((out/'rows').glob('*.json'))),
        'unified_exit_code':1,'last_progress_seconds':316.5,
        'Windows_evidence':{'log':'System','event_id':2004,'event_times_local':['2026-09-11 10:03:06','2026-09-11 10:03:07'],
            'diagnosis':'Virtual memory exhaustion','process':'python.exe','pid':53052,'reported_private_commit_bytes':24779997184},
        'actions':'Stopped only own prior uvicorn127.0.0.1:5001 child51664/parent15028 after command-line verification; did not stop user browsers or language-server. Changed new loader dispatch13GPU/9CPU to12GPU/6CPU, retained originalBF16. Empty CUDA allocator cache between sources. Replay already committed32sources and verify raw fields bitwise before aggregating; do not silently lose their all-token moments.',
        'retention':'No result file deletion; original load_audit under main remains. Reduced-residency audit uses main_commit_safe. Not a scientific failure or evidence of model-scale differences.'})
    ledger('qwen14_first_attempt_measured_lower_bound',316.5,completed_sources=32,scope='Lower bound from last flushed progress; process ended shortly after; exact uninstrumented tail unknown.')

if __name__=='__main__':main()
