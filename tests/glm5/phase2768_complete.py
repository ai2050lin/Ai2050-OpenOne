"""Seal the observed client acceptance separately from the negative research verdict."""
import json
from pathlib import Path
import sys
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import phase2768_knowledge_loop as L
from deploy.research_library import Library


def main():
    audit=L.read(L.OUT/'execution-audit.json');package=L.read(L.OUT/'package-verification.json')
    assert audit['passed'] and package['passed'] and L.sha(package['installer'])==package['sha256']
    for name in ['native-fragment.jpg','native-response.jpg','native-array.jpg']:
        assert (L.OUT/'screenshots'/name).is_file()
    assert L.read(L.OUT/'validation.json')['passed'] is False
    L.status('client_acceptance','done',browser_verified=True,desktop_version='1.0.2',
        offline_extra_verified=True,exact_model_identity_verified=True,complete_response_array_verified=True,
        research_gate='not_supported',software_tests_passed=342)
    p=L.PUBLIC/'00-progress.research.json';lib=Library(ROOT/'distributed_data/research_library')
    ref=lib.ingest(L.read(p),str(p.relative_to(ROOT)),kind='local_report')['sha256']
    acceptance={'schema':'research_loop_acceptance.v1','completed_at':datetime.now().astimezone().isoformat(),
        'run_id':L.RUN,'closed_loop_completed':True,'scientific_preregistered_gate_passed':False,
        'scientific_outcome':'single-unit generalization not supported across unseen entities and frame',
        'software_tests':{'dist_protocol':261,'backend':33,'client_components':23,'existing_frontend':14,'electron':11,'total':342,'passed':True},
        'integrity_audit':audit,'package':package,'workpackage_revision_sha256':ref,
        'observed_ui_checks':['browser fragment history','full-file snapshot before native indexing',
        'original long-text and activation','complete 9728-unit sample array','packaged file deep link',
        'desktop main and research-library extra download','desktop local node identity bridge'],
        'screenshots':['screenshots/native-fragment.jpg','screenshots/native-response.jpg','screenshots/native-array.jpg'],
        'boundaries':{'installer_wizard_executed':False,'remote_center_deployed':False,'second_node_replication':False,
            'causal_intervention':False,'graph_only_output_prediction':False,'model_tasks_created_by_client_acceptance':0}}
    L.save(L.OUT/'acceptance.json',acceptance,True)
    print(json.dumps({'closed_loop_completed':True,'research_gate_passed':False,'software_tests':342,'client':'1.0.2'},ensure_ascii=False))


if __name__=='__main__':main()
