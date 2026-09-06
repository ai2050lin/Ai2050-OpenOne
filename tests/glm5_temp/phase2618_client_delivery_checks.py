"""Isolated asset router test: no model services or GPU loading."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/glm5'))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.research_asset_service import router,resolve_research_asset
import phase2605_c676097_c692480_singleprompt_source_patch as io
path='research_kernel/c42641_output_conditioned_crossmodel_field.json'
app=FastAPI();app.include_router(router)
with TestClient(app) as client:
    health=client.get('/api/research-assets/health')
    response=client.get('/api/research-assets/file/'+path,headers={'Range':'bytes=0-1023'})
payload=io.load_json(resolve_research_asset(path))
panels=[p for p in payload['models'] if p['key'].startswith(('phase2614_','phase2615_','phase2617_','phase2619_'))]
checks={'health':health.status_code==200,'range_response':response.status_code==206 and len(response.content)==1024,
        'all11new_panels':len(panels)==11,'latest_phase2619':payload['phase']==2619,'all_dimensions':all(len(r['values'])==p['coordinate_count'] for p in panels for r in p['rows']),
        'physical_dimensions_include_all_models':{2560,5120,4096,3584}.issubset({p['coordinate_count'] for p in panels})}
result={'checks':checks,'all_checks_passed':all(checks.values()),'asset_sha256':io.sha256(resolve_research_asset(path)),'range_status':response.status_code,
        'panels':[{'key':p['key'],'dimensions':p['coordinate_count'],'rows':len(p['rows'])} for p in panels]}
io.save_json(ROOT/'tests/glm5/result/phase2618_delivery_storage_and_continuation_audit/analysis/client_api_checks.json',result)
print(json.dumps(result))
assert result['all_checks_passed']
