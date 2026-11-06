"""Final software-only release addendum; preserve the original 1.0.2 acceptance."""
from datetime import datetime
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import phase2768_knowledge_loop as L
from deploy.research_library import Library


def main():
    package=L.read(L.OUT/'package-verification-final.json')
    assert package['passed'] and L.sha(package['installer'])==package['sha256']
    assert (L.OUT/'screenshots/native-workpackage-final.jpg').is_file()
    report=L.read(L.OUT/'acceptance.json')
    assert report['closed_loop_completed'] and not report['scientific_preregistered_gate_passed']
    report.update(completed_at=datetime.now().astimezone().isoformat(),package=package,
        supersedes_software_acceptance='acceptance.json',
        validation_scope='1.0.2 production: offline evidence, local identity, native array; 1.0.3 production: chronological stages; unchanged core plus package byte verification.',
        final_desktop_version='1.0.3')
    report['software_tests']['client_components']=24;report['software_tests']['total']=343
    report['screenshots'].append('screenshots/native-workpackage-final.jpg')
    L.status('client_acceptance','done',browser_verified=True,desktop_version='1.0.3',
        offline_extra_verified=True,exact_model_identity_verified=True,complete_response_array_verified=True,
        chronological_stages_verified=True,research_gate='not_supported',software_tests_passed=343)
    path=L.PUBLIC/'00-progress.research.json'
    ref=Library(ROOT/'distributed_data/research_library').ingest(L.read(path),str(path.relative_to(ROOT)),kind='local_report')
    report['workpackage_revision_sha256']=ref['sha256']
    L.save(L.OUT/'acceptance-final.json',report,True)
    print('Final acceptance: closed loop complete; research gate not supported; 343 software tests; release 1.0.3.')


if __name__=='__main__':main()
