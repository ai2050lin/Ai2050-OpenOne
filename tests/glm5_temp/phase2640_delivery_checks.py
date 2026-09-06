"""Exact paired-precision scalar API, published field, prior API and production build QA."""
import argparse
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2640_paired_precision_atlas import OUT,INITIAL,ASSET
from server.native_path_parameter_query import LAYERS,INPUT_KEY,SOURCE
from server.research_asset_service import router
from fastapi import FastAPI
from fastapi.testclient import TestClient

def run(build=True):
    checks={};samples=[];app=FastAPI();app.include_router(router)
    with TestClient(app) as client:
        r=client.get('/api/research-assets/native-precision-frames');frames=r.json()['frames']
        checks['all16_published_frames']=r.status_code==200 and len(frames)==16
        for frame in (frames[0]['frame_id'],frames[-1]['frame_id']):
            with np.load(INITIAL['bf16']/f'field/frame_{frame:04d}.npz') as a,np.load(INITIAL['fp32']/f'field/frame_{frame:04d}.npz') as b:
                for l in LAYERS:
                    for module,inputkey in INPUT_KEY.items():
                        key=f'L{l}_{module}';W=np.load(SOURCE/f'field/weights/{key}.float32.npy',mmap_mode='r')
                        for j,k in ((0,0),(W.shape[0]-1,W.shape[1]-1)):
                            r=client.get('/api/research-assets/native-precision-parameter',params={'frame':frame,'layer':l,'module':module,'j':j,'k':k,'hj':2559,'ak':9727})
                            p=r.json();okay=r.status_code==200 and p['actual_weight_jk_same_both_precisions']==float(W[j,k])
                            for precision,pack in [('bf16',a),('fp32',b)]:
                                x=pack[f'L{l}_{inputkey}'][:,k].astype('float64');g=pack[key+'__g'][:,j].astype('float64');terms=x*g
                                v=p['values'][precision]
                                okay=okay and len(p['tokens'])==len(terms) and math.isclose(v['all_token_derivative'],math.fsum(terms),rel_tol=1e-12,abs_tol=1e-12) and v['last_token_derivative']==float(terms[-1])
                                okay=okay and v['hidden_block_output_hj']==float(pack['hidden_boundary'][l+1,2559]) and v['mlp_neuron_ak']==float(pack['mlp_boundary'][l,9727])
                                okay=okay and all(t[precision]['input_k']==float(x[i]) and t[precision]['output_adjoint_j']==float(g[i]) and t[precision]['product']==float(terms[i]) for i,t in enumerate(p['tokens']))
                            checks[f'frame{frame}/{key}/{j},{k}']=okay
                            if l==0 and module=='v_proj' and j==0:samples.append({k:v for k,v in p.items() if k!='tokens'})
        for name,params,code in [('bad_layer',{'layer':3},400),('bad_module',{'module':'../weights'},400),('negative',{'j':-1},400),('oversize',{'k':10000000},400),('hidden_oversize',{'hj':2560},400),('neuron_oversize',{'ak':9728},400),('noninteger',{'ak':'word'},422),('unpublished',{'frame':0},404)]:
            checks[name]=client.get('/api/research-assets/native-precision-parameter',params=params).status_code==code
        checks['old_native_path_works']=client.get('/api/research-assets/native-path-parameter').status_code==200
        for model in ('qwen4','qwen14','glm4','ds7'):
            checks['legacy_'+model]=client.get('/api/research-assets/native-parameter',params={'model':model}).status_code==200
        r=client.get('/api/research-assets/file/research_kernel/c42641_output_conditioned_crossmodel_field.json',headers={'Range':'bytes=0-1023'})
        checks['range206']=r.status_code==206 and len(r.content)==1024
    payload=read(ASSET);new=[p for p in payload['models'] if p['key'].startswith('phase2640_')]
    checks['four_new_panels']=len(new)==4
    checks['prior22_panels_preserved']=len([p for p in payload['models'] if p['key'].startswith(('phase2628_','phase2629_','phase2635_'))])==22
    checks['full_coordinate_width']=all(len(r['values'])==p['coordinate_count'] for p in new for r in p['rows'])
    checks['all_values_finite']=all(np.isfinite(r['values']).all() for p in new for r in p['rows'])
    checks['latest_phase']=payload['phase']==2640
    checks['asset_hash']=sha(ASSET)==read(OUT/'analysis/publication.json')['asset_sha256']
    proof=None
    if build:
        process=subprocess.run(['C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe','node_modules/vite/bin/vite.js','build'],cwd=ROOT/'frontend',capture_output=True,text=True,encoding='utf-8',errors='replace')
        proof={'exit_code':process.returncode,'stdout':process.stdout,'stderr':process.stderr};checks['frontend_build']=process.returncode==0
    report={'checks':checks,'all_checks_passed':all(checks.values()),'build':proof,'api_samples':samples,
            'not_done':'Browser visual interaction not exercised; API exact values and production bundle verified.'}
    save(OUT/'analysis'/('delivery_checks.json' if build else 'post_cleanup_checks.json'),report)
    print(json.dumps({'checks':len(checks),'failures':[k for k,v in checks.items() if not v],'all_checks_passed':report['all_checks_passed']}),flush=True)
    assert report['all_checks_passed'];return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--no-build',action='store_true');args=p.parse_args();run(not args.no_build)
