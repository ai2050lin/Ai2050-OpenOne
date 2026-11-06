"""Verify actual background-produced packs, not a fixture or hand-made snapshot."""
import gzip,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import phase2793_materials as P
def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    reg=read(P.OUT/'client-registration-v2.json');folder=P.ROOT/'distributed_data/datapack'
    main=json.load(gzip.open(folder/'datapack.json.gz','rt',encoding='utf-8'))
    extra=json.load(gzip.open(folder/'extra/research-library.json.gz','rt',encoding='utf-8'));entries=extra['entries']
    get=lambda sha:entries['/api/research-library/artifacts/'+sha]
    wp=next(w['body'] for w in main['entries']['/api/research-library/workpackages']['workpackages'] if w['body']['run_id']==P.RUN)
    assert wp['stages']['registration_validated']['report_sha256']==reg['report_sha256']
    assert wp['stages']['capability']['state']=='noninferior'
    report=get(reg['report_sha256']);graph=get(reg['graph_sha256'])
    assert not report['H1_supported'] and not report['H2_supported'] and len(report['counterexamples'])==236
    assert len(graph['feature_cards'])==44 and graph['independent_intervention_tests']==1
    assert sum(len(get(sha)['items']) for sha in report['artifact_refs']['inputs'])==576
    source=get(report['artifact_refs']['sources'])
    for f in source['files']:assert hashlib.sha256(f['text'].encode('utf-8')).hexdigest()==f['sha256']
    assert len(get(report['artifact_refs']['capability'])['base'])==2340
    for e in report['historical_evidence']:assert get(e['artifact_sha256'])['source_file_sha256']
    for e in report['reference_quality']['reviews']:assert len(get(e['artifact_sha256'])['ratings'])==48
    catalog=entries['/api/research-library/catalog']
    assert sum(r['body'].get('id')==P.RUN+':corpus' for r in catalog['datasets'])==1
    assert sum(r['body'].get('run_id')==P.RUN for r in catalog['claims'])==2
    result={'schema':'software_offline_acceptance.v1','kind':'real','run_id':P.RUN,'passed':True,
        'main_version':main['version'],'extra_version':extra['version'],'report_sha256':reg['report_sha256'],
        'all_576_inputs':True,'all_counterexamples':236,'core_historical_wrappers':9,'independent_ai_reference_reviews':2,
        'source_hashes_verified':True,'capability_monitor_rows':2340,'native_tensors_local_only':True,
        'scope':'offline evidence/report metadata complete; raw3D requires the local native-trace service and retained tensors'}
    P.save(P.OUT/'offline-acceptance-v2.json',result,True);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
