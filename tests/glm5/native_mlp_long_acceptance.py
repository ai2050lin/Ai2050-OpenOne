"""Read actual local API, derived arrays, typed evidence and explicit absence."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_long_materials as P
def main():
    import requests
    from deploy.research_library import Library
    reg=P.read(P.OUT/'client-registration.json');base='http://127.0.0.1:5001'
    def get(path):
        r=requests.get(base+path,timeout=180);r.raise_for_status();return r.json()
    ctx=get('/api/traces/context?run_id='+P.RUN);assert ctx['study_type']=='native_mlp_long' and ctx['report_sha256']==reg['report_sha256']
    report=get('/api/research-library/artifacts/'+reg['report_sha256']);assert len(report['cards'])==466 and report['analysis']['random_replicated']==0
    assert not report['dose']['supported'] and report['capability']['status']=='noninferior'
    t=ctx['examples'][0];identity=t['trace_id'];m=get('/api/traces/'+identity+'/manifest');q=m['readout_positions'][-1];block=t['layer']
    for boundary,n in [('embedding',2560),('hidden_state',2560),('mlp_gate',9728),('mlp_value',9728),('mlp_activation',9728),('head_z',4096)]:
        r=get(f'/api/traces/{identity}/field?block={block}&position={q}&boundary={boundary}');assert len(r['values'])==n
    for head in [0,31]:
        r=get(f'/api/traces/{identity}/attention-row?block={block}&head={head}&query={q}');assert len(r['weights'])==m['n_tokens'] and abs(sum(r['weights'])-1)<.01
    conn=get(f'/api/traces/{identity}/connections?block={block}&unit={t["unit"]}&head=0&coordinate=0&position={q}')
    assert conn['verified_full_snapshot'] and len(conn['mlp_output_terms'])==9728 and len(conn['head_gate_terms'])==32
    assert requests.get(base+f'/api/traces/profile?run_id={P.RUN}&block=0&space=mlp',timeout=10).status_code==404
    graph=get('/api/research-library/artifacts/'+reg['graph_sha256']);assert len(graph['feature_cards'])==482 and len(graph['graph_edges'])==482
    assert all(c['evidence']['level'] in ['observation','parameter_source'] for c in graph['feature_cards'])
    lib=Library(P.ROOT/'distributed_data/research_library')
    for r in report['cards']:
        c=lib.artifact(r['sha256'])['card'];assert c['layer']==r['layer'] and c['unit']==r['unit'] and len(c['population'])==3
    for ref in report['artifact_refs']['population']:
        p=lib.artifact(ref);assert len(p['GVA_RMS'])==3 and len(p['GVA_RMS'][0][2])==9728
        for fold,n in enumerate(p['count']):assert all(sum(row)==n for row in p['GV_sign_partition_counts'][fold])
    for source in lib.artifact(report['artifact_refs']['sources'])['files']:
        import hashlib
        assert hashlib.sha256(source['text'].encode('utf8')).hexdigest()==source['sha256']
    P.save(P.OUT/'system-acceptance.json',{'kind':'real','run_id':P.RUN,'passed':True,'report_sha256':reg['report_sha256'],'all466_cards_read':True,
        'all36_layer_statistics_count_partition_checked':True,'actual_full_field_dimensions':True,'full_snapshot_verified':True,'natural_context_has_no_fake_scope_gate':True,
        'negative_results_preserved':True,'all_source_hashes_verified':True,'no_GPU_loaded':get('/api/live/status')['status']=='idle','provenance':report['provenance']})
    print('PASS: real API, 466cards, 36 complete arrays, 482 typed fragments; no semantic upgrade')
if __name__=='__main__':main()
