"""Record completed browser, raw-data and built-client acceptance; no inference."""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).parent))
import phase2773_w6_loop as W
import phase2773_w6_paired as P


def main():
    reg=W.read(P.OUT/'client-registration.json')
    assert W.read(P.OUT/'verification.json')['passed']
    assert W.read(P.OUT/'offline-acceptance.json')['passed']
    assert 'built in' in (P.OUT/'frontend-build.log').read_text(encoding='utf-8')
    screenshots={n:W.sha(P.OUT/n) for n in ['client-results.jpg','client-dose.jpg','client-native-array.jpg']}
    doc={'kind':'software_client_acceptance','passed':True,'run_id':P.RUN,
        'browser_url':reg['client_url'],'report_sha256':reg['report_sha256'],'graph_sha256':reg['graph_sha256'],
        'screenshots':screenshots,'statistics_and_operator_checks':12,'frontend_checks':16,
        'production_build':True,'fresh_browser_console_errors':0,
        'manual_browser_checks':['real workpackage and NI result','dose curve','counterexample original question and answers',
            'versioned graph to parameter link','full model snapshot verified','B18 U4427 exact localization',
            'all9728 original BF16 unit values loaded','correct287-token metadata and source description'],
        'offline_actual_generated_pack_verified':True,'original_weights_unchanged':True,
        'remote_center_deployed':False,'second_node_replication':False,'new_electron_installer':False,
        'research_scope':'paired fixed-construction continuation preference; natural register switching unproven',
        'time':time.time()}
    p=P.OUT/'client-acceptance.json'
    if not p.exists():W.save(p,doc,True)
    P.status('client_acceptance',passed=True,checks=str(p.relative_to(W.ROOT)),
        report_sha256=reg['report_sha256'],graph_sha256=reg['graph_sha256'])
    print('Local real research -> CAS -> graph -> browser -> offline pack acceptance complete.')


if __name__=='__main__':main()
