"""Conservative supplements for interrupted wall-time tails, never invented measured timings."""
from rdc_operator_common import *


def main():
    out=BASE/'verification/recovery_resource_accounting.json'
    if out.exists():return
    entries=read(BASE/'compute_ledger.json');by={r['kind']:r for r in entries}
    first=by['qwen14_first_attempt_measured_lower_bound']['seconds'];qa=by['Q14_QA6CPU_first_attempt_lower_bound']['seconds']
    additions=[
        {'kind':'qwen14_initial_failure_unmeasured_tail_allowance','seconds':360.-first,
         'scope':'Conservative360s total allowance minus booked316.5s flushed lower bound. Preflight09:57:48.676, System2004at10:03:06/07, exited by the10:03:15 status query. This supplement is not measured execution time.'},
        {'kind':'qwen14_QA6CPU_unmeasured_tail_allowance','seconds':810.-qa,
         'scope':'Conservative810s total allowance minus booked261.4s. Original preflight10:45:45.371, own-process stop command10:58:43; includes margin for initialization/stop. Five commits retained. Not an additional full run or an exact timing.'},
        {'kind':'operator_failed_preflight_startup_allowance','seconds':180.,
         'scope':'Conservative extra allowance for uninstrumented failed pre-forward initialization/protocol/residency attempts and shell startup. Explicit allowance, not fabricated measured GPU work.'}]
    for r in additions:
        assert r['seconds']>=0
        if r['kind'] not in by:ledger(r['kind'],r['seconds'],scope=r['scope'],estimated_allowance_not_measurement=True)
    save(out,{'timestamp':stamp(),'source':snapshot(Path(__file__)),'allowances':additions,
        'load_profile_correction':'QA refinement planned max11GiB CPU, actual preflight selected9GiB. Completed old5questions used6GiB; resumed query fields and full generated IDs passed exact replay underGPU12_CPU9.',
        'limits':'Script timers plus explicit conservative interrupted-run allowances are a budget account, not total human/UI wall-clock or GPU kernel time. Running jobs are booked when they finish; no claim budget is exhausted.'})
    print('RECOVERY_RESOURCE_ALLOWANCES_BOOKED',sum(r['seconds'] for r in additions),flush=True)


if __name__=='__main__':main()
