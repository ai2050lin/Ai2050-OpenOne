"""Post-run admissibility audit. Preserve immutable freezes; never backdate power."""
from datetime import datetime
import subprocess
import mlp_tail_goodfire as C
from tools.power_analysis import check_phase,n_required


def main():
    git=['git','-c','safe.directory='+str(C.ROOT).replace('\\','/'),'-C',str(C.ROOT)]
    commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()
    dirty=bool(subprocess.check_output(git+['status','--porcelain','-uno'],text=True))
    provenance={'git_sha':commit,'git_dirty':dirty,'model_path':str(C.MODEL.resolve()),'model_revision':C.SNAPSHOT,'seed':C.SEED}
    audit={'schema':'mlp_tail_admissibility_audit.v1','kind':'real','run_id':C.RUN,'model_snapshot':C.SNAPSHOT,
        'provenance':provenance,'capture_time':datetime.now().astimezone().isoformat(),
        'capture_scope':'Retrospective provenance audit, not a claim these five fields or power were frozen before execution. Exact executed source hashes and material/model hashes remain in original unmodified plans.',
        'scientific_admissibility':'exploratory_observation_only','formal_mechanism_confirmed':False,
        'gaps':['No prospective expected binary effect / power calculation in path-plan; 96 documents cannot confirm small behavioral effects.',
                'Original run metadata lacked the five-field provenance envelope. This sidecar records the deficiency; it does not retroactively authorize or confirm the run.',
                'Old bank exposure is broader than this estimator; no independent new semantic confirmation.'],
        'allowed_claims':['Native numerical observations and exact replay','Descriptive matched-subset intervention/restore outcomes','Predictor failures and coverage'],
        'forbidden_claims':['All native units semantically decoded','Confirmed new behavioral mechanism','General capability non-inferiority'],
        'future_design_example_not_preregistration':{'expected_delta':.05,'p_d':.1,'comparisons':48,'alpha_per_comparison':.05/48,
            'required_n':n_required(.05,alpha=.05/48),'current_96_diagnostic':check_phase(.05,96,alpha=.05/48)},
        'raw_manifest_before_audit_sha256':C.file_sha(C.OUT/'manifest.json')}
    C.save_json(C.OUT/'admissibility-audit.json',audit);C.save_json(C.PUBLIC/'admissibility-audit.research.json',audit)
    report=C.read(C.OUT/'report.json');report['provenance']=provenance;report['admissibility']=audit
    C.save_json(C.OUT/'report.json',report);C.save_json(C.PUBLIC/'report.research.json',report)
    m=C.read(C.OUT/'manifest.json');m['provenance']=provenance;m['admissibility']='exploratory_observation_only'
    byname={r['path']:r for r in m['files']}
    for n in ['report.json','admissibility-audit.json']:
        p=C.OUT/n;byname[n]={'path':n,'bytes':p.stat().st_size,'sha256':C.file_sha(p),'display_required':True}
    m['files']=sorted(byname.values(),key=lambda r:r['path']);C.save_json(C.OUT/'manifest.json',m)
    print(audit['future_design_example_not_preregistration'],flush=True)


if __name__=='__main__':main()
