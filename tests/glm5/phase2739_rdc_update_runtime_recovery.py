"""Preserve implementation-only failure evidence before deterministic reruns."""
from rdc_update_common import *


def main():
    out=BASE/'runtime_recovery';path=out/'scale_pilot_and_figure_serialization.json'
    if path.exists():return
    folder=BASE/'scale/qwen4';commits=[read(p) for p in sorted((folder/'commits').glob('*.json'))]
    sources0=[snapshot(ROOT/'tests/glm5'/name) for name in ('phase2738_rdc_update_scale.py','phase2739_rdc_update_figures.py')]
    arrays={p.relative_to(BASE).as_posix():sha(p) for p in (folder/'fields').glob('*.npz')}
    save(path,{'timestamp':stamp(),'source':snapshot(__file__),'old_sources':sources0,'commits_before_rerun':commits,
      'array_sha256_before_rerun':arrays,'scale_failure_files':[p.relative_to(BASE).as_posix() for p in folder.glob('failure_*.json')],
      'scale_error':'Pilot passed is numpy.bool_, not JSON serializable; convert to Python bool.6completed prefills have not changed any model/data/config.',
      'figure_error':'Research common read accepts one argument; use explicit existence check for final-vs-preliminary scoring. No data or plotted values selected/changed.',
      'timing':'Scale failure is already ledgered by its exception handler; no duplicate charge. Figure failed duration was not recorded and is not invented.',
      'recovery':'Repeat unchanged Q4 scientific work, require all6preexisting arrays and generated IDs identical. Larger models have not started; same-history completed and will skip.'})
    print('IMPLEMENTATION_FAILURES_PRESERVED',len(commits),len(arrays),flush=True)

if __name__=='__main__':main()
