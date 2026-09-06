"""Read-only parameter-path and legacy API QA; no model/backend process startup."""
import argparse
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2635_native_path_delivery import OUT, SOURCE, ASSET
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.research_asset_service import router
from server.native_path_parameter_query import LAYERS, INPUT_KEY
from server.native_parameter_query import SOURCES

def run(build=True):
    checks={};samples=[];app=FastAPI();app.include_router(router)
    with TestClient(app) as client:
        options=client.get('/api/research-assets/native-path-frames');frames=options.json()['frames']
        checks['all34_published_frames']=options.status_code==200 and len(frames)==34
        frame_ids=sorted({frames[0]['frame_id'],frames[-1]['frame_id'],next(f['frame_id'] for f in frames if f['eos'])})
        weights=read(SOURCE/'protocol/weights.json')
        for frame in frame_ids:
            with np.load(SOURCE/f'field/factors/frame_{frame:04d}.npz',allow_pickle=False) as pack:
                for l in LAYERS:
                    for name,inputkey in INPUT_KEY.items():
                        key=f'L{l}_{name}';shape=weights[key]['shape']
                        x=pack[f'L{l}_{inputkey}'];g=pack[key+'__g'];value=pack[key+'__value']
                        for j,k in ((0,0),(shape[0]-1,shape[1]-1)):
                            response=client.get('/api/research-assets/native-path-parameter',params={'frame':frame,'layer':l,'module':name,'j':j,'k':k})
                            r=response.json();tag=f'frame{frame}/{key}/{j},{k}'
                            terms=x[:,k].astype('float64')*g[:,j].astype('float64');W=np.load(SOURCE/f'field/weights/{key}.float32.npy',mmap_mode='r')
                            checks[tag]=response.status_code==200 and r['values']['actual_weight_jk']==float(W[j,k]) and math.isclose(r['values']['all_token_derivative'],math.fsum(terms),abs_tol=1e-12,rel_tol=1e-12) and r['values']['last_token_only_derivative']==float(terms[-1]) and len(r['tokens'])==len(x) and all(t['product']==float(terms[q]) and t['projection_output_j']==float(value[q,j]) for q,t in enumerate(r['tokens']))
                            if l==35 and name=='v_proj' and j==0:samples.append({k:v for k,v in r.items() if k!='tokens'})
        for name,params,code in [('bad_layer',{'layer':36},400),('bad_module',{'module':'../../models'},400),('negative',{'j':-1},400),('large',{'k':10000000},400),('float_index',{'k':'2.5'},422),('unpublished',{'frame':2},404),('missing_frame',{'frame':9999},404)]:
            checks[name]=client.get('/api/research-assets/native-path-parameter',params=params).status_code==code
        for model,directory in SOURCES.items():
            r=client.get('/api/research-assets/native-parameter',params={'model':model,'case':0,'j':853,'k':3242})
            W=np.load(RESULT/directory/'field/final_down_weights.float32.npy',mmap_mode='r')
            checks['legacy_'+model]=r.status_code==200 and r.json()['values']['actual_down_weight_jk']==float(W[853,3242])
        r=client.get('/api/research-assets/file/research_kernel/c42641_output_conditioned_crossmodel_field.json',headers={'Range':'bytes=0-1023'})
        checks['asset_range206']=r.status_code==206 and len(r.content)==1024
    payload=read(ASSET);new=[p for p in payload['models'] if p['key'].startswith('phase2635_')]
    checks['all8_new_panels']=len(new)==8
    checks['prior14_panels_preserved']=len([p for p in payload['models'] if p['key'].startswith(('phase2628_','phase2629_'))])==14
    checks['all_new_rows_full_width']=all(len(r['values'])==p['coordinate_count'] for p in new for r in p['rows'])
    checks['finite_values']=all(np.isfinite(r['values']).all() for p in new for r in p['rows'])
    checks['latest_asset_phase']=payload['phase']==2635
    checks['asset_matches_publication_hash']=sha(ASSET)==read(OUT/'analysis/publication.json')['asset_sha256']
    proof=None
    if build:
        node='C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
        completed=subprocess.run([node,'node_modules/vite/bin/vite.js','build'],cwd=ROOT/'frontend',capture_output=True,text=True,encoding='utf-8',errors='replace')
        proof={'exit_code':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr};checks['frontend_build']=completed.returncode==0
    result={'checks':checks,'all_checks_passed':all(checks.values()),'build':proof,'api_samples':samples,
            'asset_sha256':sha(ASSET),'not_done':'browser visual interaction; API behavior, exact values and production bundle verified'}
    save(OUT/'analysis'/('delivery_checks.json' if build else 'post_cleanup_checks.json'),result)
    print(json.dumps({'check_count':len(checks),'failed':[k for k,v in checks.items() if not v],'all_checks_passed':result['all_checks_passed']}),flush=True)
    assert result['all_checks_passed']
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--no-build',action='store_true');args=parser.parse_args();run(not args.no_build)
