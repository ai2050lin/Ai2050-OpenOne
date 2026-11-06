"""Validate actual automatically generated client evidence packs."""
import gzip,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import phase2794_materials as P
def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'));reg=read(P.OUT/'client-registration.json');folder=P.ROOT/'distributed_data/datapack'
    main=json.load(gzip.open(folder/'datapack.json.gz','rt',encoding='utf-8'));extra=json.load(gzip.open(folder/'extra/research-library.json.gz','rt',encoding='utf-8'))
    get=lambda sha:extra['entries']['/api/research-library/artifacts/'+sha]
    wp=next(w['body'] for w in main['entries']['/api/research-library/workpackages']['workpackages'] if w['body']['run_id']==P.RUN)
    assert wp['stages']['registration_validated']['report_sha256']==reg['report_sha256']
    r=get(reg['report_sha256']);g=get(reg['graph_sha256']);assert r['analysis']['behavior_gate'] and not r['analysis']['H2_supported']
    assert len(r['counterexamples'])==6 and len(g['feature_cards'])==44 and len(g['graph_edges'])==64
    assert sum(len(get(sha)['items']) for sha in r['artifact_refs']['inputs'])==704
    assert sum(len(get(sha)['manifests']) for sha in r['artifact_refs']['manifests'])==719
    for card in g['feature_cards']:
        if card['index']['space']=='mlp':
            c=get(card['scope']['parameter_connection_sha256']);assert c['verified_full_snapshot'] and len(c['mlp_output_terms'])==9728 and len(c['gate_terms'])==2560
    for f in get(r['artifact_refs']['sources'])['files']:assert hashlib.sha256(f['text'].encode('utf-8')).hexdigest()==f['sha256']
    assert len(get(r['artifact_refs']['capability'])['base'])==2340
    assert len(get(r['artifact_refs']['numeric_control'])['result']['rows'])==384
    assert len(r['calibration_versions'])==4 and all(not v['passed'] for v in r['calibration_versions'][:3])
    result={'kind':'real','run_id':P.RUN,'passed':True,'report_sha256':reg['report_sha256'],'main_version':main['version'],'extra_version':extra['version'],
        'inputs':704,'native_manifests':719,'complete_parameter_contribution_artifacts':32,'counterexamples':6,'calibration_versions':4,
        'source_text_hashes_verified':True,'native_tensor_files_local_only':True,'scope':'offline reports, inputs, model identities and contribution arrays; live3D reads retained tensors through local service'}
    P.save(P.OUT/'offline-acceptance.json',result,True);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
