"""Read-only API/asset/build contracts, with no model service startup."""
import json,sys,subprocess,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.research_asset_service import router,resolve_research_asset
from server.native_parameter_query import SOURCES
import numpy as np

def run_checks(build=True,expected_phase=2629):
    app=FastAPI();app.include_router(router);responses={};checks={}
    with TestClient(app) as client:
        for model,directory in SOURCES.items():
            r=client.get('/api/research-assets/native-parameter',params={'model':model,'case':0,'j':853,'k':3242});responses[model]=r.json()
            W=np.load(RESULT/directory/'field/final_down_weights.float32.npy',mmap_mode='r')
            checks[model+'_scalar_matches_full_matrix']=r.status_code==200 and r.json()['values']['actual_down_weight_jk']==float(W[853,3242])
            dims=r.json()['counts']
            edge=client.get('/api/research-assets/native-parameter',params={'model':model,'case':dims['cases']-1,'j':dims['coordinates']-1,'k':dims['neurons']-1})
            checks[model+'_last_indices_accessible']=edge.status_code==200
        for key,params in [('unknown_model',{'model':'../../models'}),('negative',{'j':-1}),('oversize',{'k':1000000})]:
            checks[key+'_rejected']=client.get('/api/research-assets/native-parameter',params=params).status_code==400
        checks['invalid_integer_rejected']=client.get('/api/research-assets/native-parameter',params={'j':'word'}).status_code==422
        full=client.get('/api/research-assets/native-parameter',params={'model':'qwen4','case':0,'j':5,'checkpoint':0,'token':0})
        expected=float(np.load(RESULT/SOURCES['qwen4']/'field/fulltoken/case_0000.float32.npy',mmap_mode='r')[0,0,5])
        checks['fulltoken_exact_scalar']=full.status_code==200 and full.json()['full_token_coordinate']==expected
        response=client.get('/api/research-assets/file/research_kernel/c42641_output_conditioned_crossmodel_field.json',headers={'Range':'bytes=0-1023'})
        checks['asset_range206']=response.status_code==206 and len(response.content)==1024
    asset=resolve_research_asset('research_kernel/c42641_output_conditioned_crossmodel_field.json');payload=read(asset)
    new=[p for p in payload['models'] if p['key'].startswith(('phase2628_','phase2629_'))]
    checks['all14_new_panels']=len(new)==14 if expected_phase==2629 else len(new)==12
    checks['all_rows_full_physical_width']=all(len(r['values'])==p['coordinate_count'] for p in new for r in p['rows'])
    checks['latest_phase']=payload['phase']==expected_phase
    checks['all_four_weight_panels']=sum(p['key'].startswith('phase2628_weights_') for p in new)==4
    checks['native_and_target_objectives_separated']=any(p['key']=='phase2629_native_vs_task' for p in new) if expected_phase==2629 else True
    builds=None
    if build:
        node=Path('C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
        process=subprocess.run([str(node),'node_modules/vite/bin/vite.js','build'],cwd=ROOT/'frontend',capture_output=True,text=True,encoding='utf-8',errors='replace')
        builds={'exit_code':process.returncode,'stdout':process.stdout,'stderr':process.stderr};checks['frontend_build']=process.returncode==0
    result={'checks':checks,'all_checks_passed':all(checks.values()),'asset_sha256':sha(asset),'panels':[{'key':p['key'],'width':p['coordinate_count'],'rows':len(p['rows'])} for p in new],
        'api_scalar_samples':responses,'build':builds,'not_done':'browser visual interaction not exercised; API and production bundle checked'}
    filename='delivery_checks.json' if build else 'post_cleanup_api_checks.json'
    save(RESULT/'phase2629_expanded_native_confirmation/analysis'/filename,result)
    return result

if __name__=='__main__':
    result=run_checks();print(json.dumps({'checks':result['checks'],'all_checks_passed':result['all_checks_passed']}));assert result['all_checks_passed']
