"""Matched-operation scalar accuracy, range/validation, previous routes, production build."""
import argparse,math,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2647_matched_operation_delivery import OUT,ASSET,BF,INITIAL,CONFIRM,LAYERS
from server.native_path_parameter_query import SOURCE
from server.research_asset_service import router
from fastapi import FastAPI
from fastapi.testclient import TestClient

def run(build=True):
    app=FastAPI();app.include_router(router);checks={};samples=[]
    cases=read(OUT/'material/published_cases.json')
    with TestClient(app) as client:
        r=client.get('/api/research-assets/native-operation-cases');checks['32_case_options']=r.status_code==200 and len(r.json()['cases'])==32
        for c in cases:
            ci=c['case_index'];source=INITIAL if c['field_set']=='initial' else CONFIRM
            with np.load(source/f'field/case_{ci:04d}.npz') as f,np.load(BF/f'field/case_{ci:04d}.npz') as b:
                for l in LAYERS:
                    W=np.load(SOURCE/f'field/weights/L{l}_v_proj.float32.npy',mmap_mode='r')
                    for j,k in ((0,0),(1023,2559)):
                        params={'case':ci,'layer':l,'j':j,'k':k,'hj':2559,'ak':9727,'checkpoint':36,'token':0}
                        response=client.get('/api/research-assets/native-operation-parameter',params=params);p=response.json();ok=response.status_code==200
                        ok &= p['actual_weight']==float(W[j,k]) and p['generated']==c['generated'] and p['target']==c['target']
                        for name,z in [('bf16',b),('fp32',f)]:
                            h=z['hidden'];aa=z['mlp_positions'];v=p['fields'][name]
                            ok &= v['embedding_token_hj']==float(h[0,0,2559]) and v['hidden_checkpoint_token_hj']==float(h[36,0,2559]) and v['mlp_neuron_boundary_ak']==float(aa[l,2,9727])
                        for obj in ('native','common'):
                            x=f[f'L{l}_v_x'][:,k].astype('float64');g=f[f'{obj}__L{l}_v_g'][:,j].astype('float64');terms=x*g;value=p['objectives'][obj]
                            ok &= value['available'] and value['output_ids']==c[obj+'_ids'] and math.isclose(value['all_token_parameter_derivative'],math.fsum(terms),rel_tol=1e-12,abs_tol=1e-12)
                            ok &= len(p['tokens'])==len(x) and all(t['V_input_k']==float(x[i]) and t[obj]['adjoint_j']==float(g[i]) and t[obj]['parameter_product']==float(terms[i]) for i,t in enumerate(p['tokens']))
                        checks[f'case{ci}/L{l}/{j},{k}']=bool(ok)
                        if ci==0 and l==0 and j==0:samples.append({k:v for k,v in p.items() if k!='tokens'})
        for name,params,code in [('bad_layer',{'layer':1},400),('negative',{'j':-1},400),('large_j',{'j':1024},400),('large_k',{'k':2560},400),('large_h',{'hj':2560},400),('large_a',{'ak':9728},400),('large_checkpoint',{'checkpoint':37},400),('bad_token',{'token':-1},400),('oversize_token',{'token':1000},400),('noninteger',{'case':'oops'},422),('unpublished',{'case':1},404)]:
            checks[name]=client.get('/api/research-assets/native-operation-parameter',params=params).status_code==code
        checks['default_boundary']=client.get('/api/research-assets/native-operation-parameter').status_code==200
        for name in ('native-path-parameter','native-precision-parameter'):checks['old_'+name]=client.get('/api/research-assets/'+name).status_code==200
        for model in ('qwen4','qwen14','glm4','ds7'):checks['legacy_'+model]=client.get('/api/research-assets/native-parameter',params={'model':model}).status_code==200
        response=client.get('/api/research-assets/file/research_kernel/c42641_output_conditioned_crossmodel_field.json',headers={'Range':'bytes=0-1023'})
        checks['range206']=response.status_code==206 and len(response.content)==1024
    p=read(ASSET);pub=read(OUT/'analysis/publication.json');new=[m for m in p['models'] if m['key'].startswith('phase2647_')]
    checks.update(eight_new_panels=len(new)==8,all_prior_panels=set(pub['prior_panel_keys']).issubset({m['key'] for m in p['models']}),
        latest_phase=p['phase']==2647,all_coordinate_widths=all(len(r['values'])==m['coordinate_count'] for m in new for r in m['rows']),
        finite_values=all(np.isfinite(r['values']).all() for m in new for r in m['rows']),asset_hash=sha(ASSET)==pub['asset_sha256'])
    proof=None
    if build:
        cp=subprocess.run(['C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe','node_modules/vite/bin/vite.js','build'],cwd=ROOT/'frontend',capture_output=True,text=True,encoding='utf-8',errors='replace')
        proof={'exit_code':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr};checks['frontend_build']=cp.returncode==0
    report={'checks':checks,'all_checks_passed':all(checks.values()),'build':proof,'samples':samples,'not_done':'Browser visual interaction; numeric API and production bundle verified.'}
    save(OUT/'analysis'/('delivery_checks.json' if build else 'post_cleanup_checks.json'),report)
    print(json.dumps({'checks':len(checks),'failures':[k for k,v in checks.items() if not v],'all_checks_passed':report['all_checks_passed']}),flush=True)
    assert report['all_checks_passed']

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--no-build',action='store_true');args=parser.parse_args();run(not args.no_build)
