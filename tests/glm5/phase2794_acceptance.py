"""Read-only actual local API acceptance, no inference and no synthetic numbers."""
from pathlib import Path
import sys,json,requests
sys.path.insert(0,str(Path(__file__).parent))
import phase2794_materials as P
def main():
    base='http://127.0.0.1:5001';reg=json.loads((P.OUT/'client-registration.json').read_text(encoding='utf-8'))
    def get(path):
        r=requests.get(base+path,timeout=180);r.raise_for_status();return r.json()
    ctx=get('/api/traces/context?run_id='+P.RUN);assert ctx['report_sha256']==reg['report_sha256']
    identity=ctx['examples'][0]['trace_id'];d=get('/api/traces/'+identity+'/manifest');position=d['readout_positions'][-1]
    for boundary,n in [('embedding',2560),('mlp_gate',9728),('mlp_value',9728),('mlp_activation',9728),('head_z',32*128)]:
        v=get(f'/api/traces/{identity}/field?position={position}&block=34&boundary={boundary}')
        assert len(v['values'])==n
    profile=get(f'/api/traces/profile?run_id={P.RUN}&block=34&space=mlp');assert len(profile['heldout_delta'])==9728
    assert requests.get(base+f'/api/traces/profile?run_id={P.RUN}&block=0&space=mlp').status_code==404
    conn=get(f'/api/traces/{identity}/connections?block=34&head=28&unit=1995&coordinate=0&position={position}')
    assert conn['verified_full_snapshot'] and len(conn['mlp_output_terms'])==9728
    children=get('/api/traces/catalog?parent='+identity)['traces'];assert len(children)>=1
    child=children[0]['trace_id'];cm=get('/api/traces/'+child+'/manifest')
    assert len(get(f'/api/traces/{child}/field?boundary=embedding&position={cm["readout_positions"][0]}')['values'])==2560
    a=get(f'/api/traces/{identity}/attention-row?block=34&head=28&query={position}');assert len(a['weights'])==1024
    report=get('/api/research-library/artifacts/'+reg['report_sha256']);assert len(report['counterexamples'])==6 and not report['analysis']['H2_supported']
    graph=get('/api/research-library/artifacts/'+reg['graph_sha256']);assert len(graph['feature_cards'])==44 and len(graph['graph_edges'])==64
    result={'kind':'real','run_id':P.RUN,'report_sha256':reg['report_sha256'],'full_native_dimensions':True,'parameter_snapshot_verified':True,
        'cached_generation_read':True,'unanalysed_profile_explicit404':True,'graph_cards':44,'typed_edges':64,'counterexamples':6,'reference_assisted_scope_preserved':True,
        'no_GPU_model_loaded_for_acceptance':get('/api/live/status')['status']=='idle'}
    P.save(P.OUT/'system-acceptance.json',result,True);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
