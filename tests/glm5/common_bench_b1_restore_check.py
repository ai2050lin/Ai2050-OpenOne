"""Real CUDA engineering calibration after nonzero edits, discovery only."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import common_bench_b1_repair as R
from common_bench_b1_repair_run import Runner

def main():
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    assert not (R.OUT/'restore-check.json').exists()
    rows=R.read(R.OLD/'discovery-materials.json')[:8];targets=R.target_map(rows)
    groups={k:{int(b):v for b,v in g.items()} for k,g in R.read(R.OUT/'path-frozen.json')['groups'].items()}
    start=time.monotonic();checks=[]
    with GpuLease(),torch.inference_mode():
        rr=Runner('restore-check');batch=rr.batch(rows);base,_,_=rr.run(batch,capture=True)
        saved=rr.grab['raw',31].clone();ds,_=rr.vectors(rows,targets,saved)
        for arm in R.BASE_ARMS+list(groups):
            if arm in groups:rr.run(batch,head_group=groups[arm])
            elif arm.startswith('donor_'):rr.run(batch,mode='set',block=31,delta=saved[[i^(2 if arm=='donor_q' else 1) for i in range(8)]])
            else:rr.run(batch,mode='add',block=35 if arm.startswith('direct_') else 31,delta=ds[arm])
            restored,_,_=rr.run(batch)
            checks.append({'arm':arm,'full_vocab_bitwise_equal':bool(torch.equal(restored,base)),'state_cleared':rr.s=={}})
    R.save('restore-check.json',{'scope':'engineering calibration on one exposed discovery family, no new scientific samples',
        'checks':checks,'all_passed':all(all(v for k,v in c.items() if k!='arm') for c in checks),
        'code_sha256':R.sha(__file__),'elapsed_seconds':time.monotonic()-start},True)
    assert all(c['full_vocab_bitwise_equal'] and c['state_cleared'] for c in checks)

if __name__=='__main__':main()
