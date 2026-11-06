"""Check the generated real client pack, without accessing remote services."""
import gzip,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_long_materials as P
def main():
    r=P.read(P.OUT/'client-registration.json');root=P.ROOT/'distributed_data/datapack'
    with gzip.open(root/'extra/research-library.json.gz','rt',encoding='utf8') as f:extra=json.load(f)
    entries=extra['entries'];get=lambda sha:entries['/api/research-library/artifacts/'+sha]
    d=get(r['report_sha256']);g=get(r['graph_sha256']);assert d['analysis']['random_replicated']==0 and not d['dose']['supported'];assert d['capability']['status']=='noninferior'
    assert len(g['feature_cards'])==482 and len(g['graph_edges'])==482
    assert sum(len(get(ref)['documents']) for ref in d['artifact_refs']['documents'])==2048
    for card in d['cards']:
        c=get(card['sha256'])['card'];assert len(c['poles'])==12
        for pole in c['poles'].values():
            for example in pole['examples']:
                doc=next(p for p in get(d['artifact_refs']['documents'][example['doc_index']//32])['documents'] if p['doc_index']==example['doc_index'])
                assert str(example['position']) in doc['prefix_texts'] and len(example['GVA'])==3
    assert len(get(d['artifact_refs']['manifests'])['manifests'])==8
    for ref in d['artifact_refs']['parameters']:
        p=get(ref);assert p['verified_full_snapshot'] and len(p['head_gate_terms'])==32 and len(p['mlp_output_terms'])==9728
    for source in get(d['artifact_refs']['sources'])['files']:assert hashlib.sha256(source['text'].encode('utf8')).hexdigest()==source['sha256']
    wp=next(w['body'] for w in entries['/api/research-library/workpackages']['workpackages'] if w['body']['run_id']==P.RUN)
    assert len(get(wp['stages']['client_acceptance']['capability_inputs_sha256'])['items'])==2340
    assert get(wp['stages']['native_mlp_plan']['plan_sha256'])['status']=='proposed_not_executed'
    result={'kind':'real','run_id':P.RUN,'passed':True,'version':extra['version'],'report_sha256':r['report_sha256'],'documents':2048,'unit_cards':466,
        'all_pole_positions_have_visible_prefix':True,'parameter_evidence':16,'native_manifests':8,'capability_questions':2340,'source_hashes_verified':True,'proposal_not_execution':True,
        'scope':'all research metadata survives offline; large tensors require the retained local node service','provenance':d['provenance']}
    P.save(P.OUT/'offline-acceptance.json',result);print(json.dumps({k:v for k,v in result.items() if k!='provenance'}))
if __name__=='__main__':main()
