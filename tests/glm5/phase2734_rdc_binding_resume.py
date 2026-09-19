"""Preserve and verify a capped scale attempt without changing any material."""
import argparse
from rdc_binding_common import *

def checkpoint():
    out=BASE/'scale/qwen14';dest=out/'recovery_before_resume.json'
    if dest.exists():return
    start=time.monotonic();commits={}
    for path in sorted((out/'commits').glob('*.json')):
        r=read(path);field=out/'fields'/f'{r["sample_id"]}.npz'
        assert sha(field)==r['array_sha']
        commits[r['sample_id']]={'commit_sha256':sha(path),'field_sha256':r['array_sha']}
    profile={name:read(out/name) for name in ('runtime.json','residency/load_audit.json','residency/resource_check.json')}
    immutable(dest,{'timestamp':stamp(),'source':snapshot(Path(__file__)),'committed_rows':len(commits),
      'unchanged_evidence':commits,'first_attempt_profile':profile,'suite_progress':read(BASE/'scale/suite_progress.json'),
      'model_failures':[read(p) for p in sorted(out.glob('failure_*.json'))],
      'protocol_sha256':sha(BASE/'scale/protocol.json'),'seconds':time.monotonic()-start,
      'scope':'Capture the completed commit boundary and original residency before restarting the same serial suite. No native output, fit or parameter is altered.'})
    ledger('binding_qwen14_recovery_checkpoint_audit',time.monotonic()-start)
    print('QWEN14_RECOVERY_CHECKPOINT',len(commits),flush=True)

def verify():
    start=time.monotonic();out=BASE/'scale/qwen14';old=read(out/'recovery_before_resume.json');current=read(out/'result.json')
    assert sha(BASE/'scale/protocol.json')==old['protocol_sha256']
    for sid,r in old['unchanged_evidence'].items():
        assert sha(out/'commits'/f'{sid}.json')==r['commit_sha256']
        assert sha(out/'fields'/f'{sid}.npz')==r['field_sha256']
    before=old['first_attempt_profile']['runtime.json'];after=current['runtime']
    comparison={k:before[k]==after[k] for k in ('width','units','depth','early','last_block','dtype','quantized','device_map')}
    assert all(v for k,v in comparison.items() if k!='device_map')
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'all_passed':True,
      'prior_commits_and_arrays_unchanged':old['committed_rows'],'total_committed_rows':len(list((out/'commits').glob('*.json'))),
      'profile_comparison':comparison,'first_attempt_profile':old['first_attempt_profile'],
      'resumed_profile':{'runtime':after,'load_audit':read(out/'residency/load_audit.json')},
      'failed_model_attempt_seconds':[r['seconds'] for r in old['model_failures']],
      'final_model_attempt_seconds':current['seconds'],
      'sum_recorded_model_attempt_seconds':sum(r['seconds'] for r in old['model_failures'])+current['seconds'],
      'seconds':time.monotonic()-start,
      'scope':'Exact bytes for all first-attempt committed outputs, same frozen material; final model attempt timer alone omits the earlier capped attempt. Compare residency separately from model precision.'}
    assert result['total_committed_rows']==128
    save(out/'recovery_verification.json',result);ledger('binding_qwen14_recovery_verify',result['seconds'])
    print('QWEN14_RECOVERY_VERIFIED',result['prior_commits_and_arrays_unchanged'],comparison,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',action='store_true');parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    assert args.checkpoint!=args.verify
    checkpoint() if args.checkpoint else verify()
