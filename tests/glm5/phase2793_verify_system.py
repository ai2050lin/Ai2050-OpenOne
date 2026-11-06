"""Actual local-service integration acceptance; never synthesizes research data."""
from pathlib import Path
import json,sys,time
import requests
sys.path.insert(0,str(Path(__file__).parent))
import phase2793_materials as P

def main():
    base='http://127.0.0.1:5001'
    def get(path):
        r=requests.get(base+path,timeout=180);r.raise_for_status();return r.json()
    reg=json.loads((P.OUT/'client-registration-v2.json').read_text(encoding='utf-8'))
    context=get('/api/traces/context?run_id='+P.RUN);assert context['report_sha256']==reg['report_sha256']
    identity=context['examples'][-1]['trace_id'];m=get(f'/api/traces/{identity}/manifest');pos=m['readout_positions'][-1]
    arrays={b:get(f'/api/traces/{identity}/field?block=32&position={pos}&boundary={b}') for b in ['embedding','mlp_gate','mlp_value','mlp_activation','head_z']}
    assert len(arrays['embedding']['values'])==2560 and all(len(arrays[b]['values'])==9728 for b in ['mlp_gate','mlp_value','mlp_activation'])
    a=get(f'/api/traces/{identity}/attention-row?block=32&head=11&query={pos}')
    early=get(f'/api/traces/{identity}/attention-row?block=32&head=11&query={m["entity_position"]}')
    assert a['query_position']!=early['query_position'] and len(a['weights'])==m['n_tokens']
    unit=context['profiles']['32']['unit'];conn=get(f'/api/traces/{identity}/connections?block=32&position={pos}&head=11&unit={unit}&coordinate=0')
    assert conn['verified_full_snapshot'] and len(conn['mlp_output_terms'])==9728 and len(conn['head_gate_terms'])==32
    profile=get(f'/api/traces/profile?run_id={P.RUN}&block=32&space=mlp');assert len(profile['heldout_delta'])==9728
    children=get('/api/traces/catalog?parent='+identity)['traces'];assert children
    child=get('/api/traces/'+children[0]['trace_id']+'/manifest');assert child['parent_trace_id']==identity
    # Exercise the public capture adapter on the actual local CUDA model, then unload.
    r=requests.post(base+'/api/live/load',timeout=15);r.raise_for_status()
    try:
        for _ in range(120):
            st=get('/api/live/status')
            if st['status']=='ready':break
            if st['status']=='error':raise RuntimeError(st['error'])
            time.sleep(1)
        else:raise RuntimeError('local model load timed out')
        assert st['device']=='cuda'
        c=json.loads((P.OUT/'corpus.json').read_text(encoding='utf-8'));item=next(x for x in c['items'] if x['length']==1024)
        r=requests.post(base+'/api/live/run',json={'text':item['text'],'max_tokens':1024},timeout=180);r.raise_for_status();live=r.json()
        assert live['n_tokens']==1024 and not live['truncated'] and live['deep'] and live['model_snapshot']==m['model_snapshot']
        assert len(live['files'])>=38 and get('/api/traces/'+live['trace_id']+'/manifest')['trace_id']==live['trace_id']
        r=requests.post(base+'/api/live/run',json={'text':item['text'],'max_tokens':512},timeout=20)
        assert r.status_code==422 and '未截断' in r.json()['detail']
    finally:
        requests.post(base+'/api/live/unload',timeout=30).raise_for_status()
    result={'kind':'real','run_id':P.RUN,'stored_trace_id':identity,'live_trace_id':live['trace_id'],'live_tokens':live['n_tokens'],
        'n_heads':m['n_heads'],'all_mlp_units':9728,'parameter_snapshot_verified':True,'early_and_final_attention_separate':True,
        'oversize_explicitly_rejected':True,'post_unload':get('/api/live/status')['status'],'report_sha256':reg['report_sha256'],
        'generation_step':child['generation_step'],'local_node_only':True,'no_new_research_claim':True}
    (P.OUT/'system-acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
