"""Checkpoint safe native QA residency refinement; no model or result mutation."""
from datetime import datetime
from rdc_operator_common import *


def main():
    out=BASE/'qa/qwen14/main/residency_v3';now=stamp();prior=BASE/'qa/qwen14/main/adaptive_residency'
    resource=read(prior/'resource_check.json');elapsed=(datetime.fromisoformat(now)-datetime.fromisoformat(resource['timestamp'])).total_seconds()
    commits=list((BASE/'qa/qwen14/main/commits').glob('*.json'))
    immutable(out/'refinement.json',{'timestamp':now,'source':snapshot(Path(__file__)),'old_load_audit':read(prior/'load_audit.json'),
        'old_replay_check':read(BASE/'qa/qwen14/main/residency_replay_check.json'),
        'completed_commits':len(commits),'commits_sha256':{p.name:sha(p) for p in commits},
        'old_QA_source':snapshot(ROOT/'tests/glm5/rdc_operator_qa.py'),'old_loader_source':snapshot(ROOT/'tests/glm5/rdc_operator_model.py'),
        'pre_stop_resource_observation':'11:35:33 local: hostfree8164532KiB, system virtual/commitfree10665228KiB; no old API running. Model child20024,parent86604 verified; user language-server58888 explicitly not a target.',
        'new_request':'13GiB GPU max dispatch, adaptive up to11GiB CPU with host>(CPU+4)GiB and commit>(CPU+22)GiB preflight. OriginalBF16 and identical questions/token budget. Head checkpoint remains native BF16.',
        'required_replay':'Exact full-layer/all-coordinate/MLP-factor/all-source-attention and complete generated token sequence against first committed case before any new answer.',
        'scientific_selection_change':False,'seconds_pre_stop':elapsed,
        'allowance':'Book preflight-to-checkpoint wall interval plus60s conservative initialization/stop margin. This run was interrupted for resource efficiency, not answer quality.'})
    kind='Q14_QA9CPU_interrupted_walltime_with_margin'
    if not any(r['kind']==kind for r in read(BASE/'compute_ledger.json')):
        ledger(kind,elapsed+60,estimated_allowance_not_exact=True,completed_questions=len(commits))
    print('QA_RESIDENCY_V3_CHECKPOINT',len(commits),'seconds_booked',elapsed+60,flush=True)


if __name__=='__main__':main()
