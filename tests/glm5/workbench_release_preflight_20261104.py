"""Read-only live API checks. These never count as browser visual acceptance."""
from pathlib import Path
import json
import urllib.request
import sys
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tests/glm5/result/workbench_release_acceptance_20261104'
BASE='http://127.0.0.1:5001'

def get(path):
    with urllib.request.urlopen(BASE+path,timeout=90) as r:return json.load(r)

def main():
    cfg=get('/api/platform/config')
    source=json.loads((ROOT/'tests/glm5/result/phase2794_arbitration_v4/client-registration.json').read_text(encoding='utf-8'))
    import urllib.parse
    trace=urllib.parse.parse_qs(urllib.parse.urlparse(source['trace_url']).query)['trace'][0]
    m=get(f'/api/traces/{trace}/manifest');p=m['readout_positions'][-1]
    embedding=get(f'/api/traces/{trace}/field?block=0&position={p}&boundary=embedding')
    activation=get(f'/api/traces/{trace}/field?block=34&position={p}&boundary=mlp_activation')
    row=get(f'/api/traces/{trace}/attention-row?block=34&head=10&query={p}')
    connection=get(f'/api/traces/{trace}/connections?block=34&head=10&unit=1995&position={p}&coordinate=0')
    assert len(embedding['values'])==m['d_model']
    assert len(activation['values'])==m['d_ff']
    assert len(row['weights'])==m['n_tokens']
    assert connection['verified_full_snapshot'] and connection['model_snapshot']==m['model_snapshot']
    assert len(connection['mlp_output_terms'])==m['d_ff'] and len(connection['head_output_terms'])==m['n_heads']
    space=get('/api/traces/research-space');models=get('/api/parameters/models')
    d={'schema':'workbench_release_preflight.v1','kind':'engineering','passed':True,'browser_visual_acceptance':False,
        'model_inference_executed':False,'backend':BASE,'trace_id':trace,'model_snapshot':m['model_snapshot'],
        'tokens':m['n_tokens'],'embedding_coordinates':len(embedding['values']),'mlp_units':len(activation['values']),
        'query_position':p,'attention_sources':len(row['weights']),'complete_snapshot_matches':True,
        'all_head_contributions':len(connection['head_output_terms']),'all_mlp_contributions':len(connection['mlp_output_terms']),
        'local_model_id':connection['local_model_id'],'configured_models':len(models['models']),
        'research_space_counts':space.get('counts'),
        'data_sync':{'published':bool(cfg.get('datapack',{}).get('version')),'metadata':{k:cfg.get('datapack',{}).get(k) for k in ['version','bytes','created_at']}},
        'contribution':{'local_tools':cfg.get('local_tools'),'note':'API readiness only; interactive node binding/authorization was not exercised.'},
        'package_allowed':False,'package_gate':'actual browser visual acceptance still required'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'api-preflight.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:d[k] for k in ['passed','embedding_coordinates','mlp_units','complete_snapshot_matches','browser_visual_acceptance','package_allowed']}))

if __name__=='__main__':main()
