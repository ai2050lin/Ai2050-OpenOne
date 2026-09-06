"""Exact live-shape sequence API, allpublished raw values, older routes and frontend build."""
import argparse,math,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2669_symmetric_multitoken_delivery import OUT,ASSET,BF,FP,Q14,LAYERS
from server.native_path_parameter_query import SOURCE
from phase2666_multitoken_parameter_engine import PARTS
from server.research_asset_service import router
from fastapi import FastAPI
from fastapi.testclient import TestClient


def run(build=True,exact_only=False):
    app=FastAPI();app.include_router(router);checks={};samples=[];cases=read(OUT/'material/published_cases.json')
    with TestClient(app) as client:
        r=client.get('/api/research-assets/native-multitoken-cases');checks['64_options']=r.status_code==200 and len(r.json()['cases'])==64
        for c in cases:
            ci=c['case_index'];T=len(c['prompt_ids'])
            with np.load(FP/f'field/case_{ci:04d}.npz') as fpack,np.load(BF/f'field/case_{ci:04d}.npz') as bpack:
                f={key:fpack[key] for key in fpack.files};b={key:bpack[key] for key in bpack.files}
                for l in LAYERS:
                    W=np.load(SOURCE/f'field/weights/L{l}_v_proj.float32.npy',mmap_mode='r',allow_pickle=False)
                    for j,k in ((0,0),(1023,2559)):
                        response=client.get('/api/research-assets/native-multitoken-parameter',params={'case':ci,'layer':l,'j':j,'k':k,'hj':2559,'ak':9727,'checkpoint':36,'token':0});v=response.json();ok=response.status_code==200
                        ok &= v['actual_weight']==float(W[j,k]) and v['generated']==c['generated'] and v['sequence_contrast']==c['contrast']
                        for name,z in [('bf16',b),('fp32',f)]:
                            fields=v['fields'][name];ok &= fields['embedding']==float(z['hidden_fulltoken'][0,0,2559]) and fields['hidden']==float(z['hidden_fulltoken'][36,0,2559]) and fields['mlp']==float(z['mlp_boundary'][l,9727])
                            if 'decision_hidden_fulltoken' in z:ok &= fields['decision_hidden_boundary']==float(z['decision_hidden_fulltoken'][36,-1,2559]) and fields['decision_mlp_boundary']==float(z['decision_mlp_boundary'][l,9727])
                        sums=[]
                        for bi,label in enumerate(('Y','N')):
                            x=f[f'{label}__L{l}_v_x'][:,k].astype('float64');g=f[f'{label}__L{l}_v_g'][:,j].astype('float64');products=x*g;obj=v['branches'][bi];total=math.fsum(products);sums.append(total)
                            ok &= obj['categories']==c['branches'][bi]['categories'] and obj['part_logprobs']==c['branches'][bi]['part_logprobs']
                            for part in PARTS:
                                gp=f[f'{label}__L{l}_v_g_{part}'][:,j].astype('float64');pp=x*gp
                                ok &= obj['part_derivatives'][part]==math.fsum(pp)
                                ok &= all(t['parts'][part]['adjoint']==float(gp[i]) and t['parts'][part]['product']==float(pp[i]) for i,t in enumerate(obj['tokens']))
                            ok &= obj['target_ids']==c['branches'][bi]['target_ids'] and obj['logprobs']==c['branches'][bi]['logprobs'] and obj['derivative']==total
                            ok &= obj['prompt_last_only']==float(products[T-1]) and obj['branch_last_only']==float(products[-1])
                            ok &= all(t['x']==float(x[i]) and t['adjoint']==float(g[i]) and t['product']==float(products[i]) for i,t in enumerate(obj['tokens']))
                            ok &= obj['hidden_prompt_adjoint']==float(f[f'{label}__hidden_adjoint_prompt_boundary'][l+1,2559]) and obj['mlp_prompt_adjoint']==float(f[f'{label}__mlp_adjoint_prompt_boundary'][l,9727])
                        ok &= v['parameter_derivative']==sums[0]-sums[1] and all(v['part_parameter_derivatives'][p]==v['branches'][0]['part_derivatives'][p]-v['branches'][1]['part_derivatives'][p] for p in PARTS);checks[f'case{ci}/L{l}/{j},{k}']=bool(ok)
                        if ci==256 and l==0 and j==0:samples.append({k:v for k,v in v.items() if k!='branches'})
        for name,params,code in [('badlayer',{'layer':1},400),('negative',{'j':-1},400),('j',{'j':1024},400),('k',{'k':2560},400),('hj',{'hj':2560},400),('ak',{'ak':9728},400),('checkpoint',{'checkpoint':37},400),('tokenneg',{'token':-1},400),('tokenmax',{'token':1000},400),('noninteger',{'case':'oops'},422),('unpublished',{'case':0},404)]:
            checks[name]=client.get('/api/research-assets/native-multitoken-parameter',params=params).status_code==code
        checks['default_case256']=client.get('/api/research-assets/native-multitoken-parameter').status_code==200
        for name in ('native-path-parameter','native-precision-parameter','native-operation-parameter','native-output-parameter','native-sequence-parameter'):checks['old_'+name]=client.get('/api/research-assets/'+name).status_code==200
        for model in ('qwen4','qwen14','glm4','ds7'):checks['legacy_'+model]=client.get('/api/research-assets/native-parameter',params={'model':model}).status_code==200
        response=client.get('/api/research-assets/file/research_kernel/c42641_output_conditioned_crossmodel_field.json',headers={'Range':'bytes=0-1023'});checks['range206']=response.status_code==206 and len(response.content)==1024
    if exact_only:
        report={'checks':checks,'all_checks_passed':all(checks.values()),'scope':'Pre-publication exact API values only; no bundle publication/build/visual QA claimed.'}
        save(OUT/'analysis/api_exact_preflight.json',report);print(json.dumps({'checks':len(checks),'failures':[k for k,v in checks.items() if not v],'all_checks_passed':report['all_checks_passed']}),flush=True);assert report['all_checks_passed'];return
    asset=read(ASSET);pub=read(OUT/'analysis/publication.json');new=[m for m in asset['models'] if m['key'].startswith('phase2669_')]
    checks.update(ten_panels=len(new)==10,all_prior_panels=set(pub['prior_panel_keys']).issubset({m['key'] for m in asset['models']}),latest_phase=asset['phase']==2669,
        all_coordinate_widths=all(len(r['values'])==m['coordinate_count'] for m in new for r in m['rows']),finite_values=all(np.isfinite(r['values']).all() for m in new for r in m['rows']),asset_hash=sha(ASSET)==pub['asset_sha256'])
    qinfo=read(Q14/'protocol/model.json');checks['separate_Qwen14_hidden_width']=next(m for m in new if m['key']=='phase2669_q14_h')['coordinate_count']==qinfo['dimensions']['hidden']
    checks['separate_Qwen14_mlp_width']=next(m for m in new if m['key']=='phase2669_q14_mlp')['coordinate_count']==qinfo['dimensions']['mlp']
    catalog=read(OUT/'material/client_panel_catalog.json')
    with TestClient(app) as client:
        result=client.get('/api/research-assets/native-atlas-panels');checks['ten_live_panel_types']=result.status_code==200 and len(result.json()['panels'])==10
        for m in new:
            for ri in (0,len(m['rows'])-1):
                response=client.get('/api/research-assets/native-atlas-rows',params={'panel':m['key'],'start':ri,'count':1});v=response.json()
                checks[f"exact_full_panel/{m['key']}/{ri}"]=response.status_code==200 and v['coordinate_count']==m['coordinate_count'] and v['rows'][0]['values']==m['rows'][ri]['values'] and v['rows'][0]['label']==m['rows'][ri]['label']
        for name,params,code in [('path',{'panel':'../../model'},404),('negative',{'panel':new[0]['key'],'start':-1},400),('last',{'panel':new[0]['key'],'start':len(new[0]['rows'])},400),('count0',{'panel':new[0]['key'],'count':0},400),('count9',{'panel':new[0]['key'],'count':9},400)]:
            checks['panel_'+name]=client.get('/api/research-assets/native-atlas-rows',params=params).status_code==code
    checks['all10_panel_matrices_hashes']=len(catalog['panels'])==10 and all(sha(OUT/'maps/client_panels'/(p['key']+'.npz'))==p['matrix_sha256'] for p in catalog['panels'])
    proof=None
    if build:
        cp=subprocess.run(['C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe','node_modules/vite/bin/vite.js','build'],cwd=ROOT/'frontend',capture_output=True,text=True,encoding='utf-8',errors='replace')
        proof={'exit_code':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr};checks['frontend_build']=cp.returncode==0
    report={'checks':checks,'all_checks_passed':all(checks.values()),'build':proof,'samples':samples,'scope':'This script verifies numeric API and production build only. Actual browser observations, if completed, are recorded separately in browser_checks.json.'}
    save(OUT/'analysis'/('delivery_checks.json' if build else 'post_cleanup_checks.json'),report);print(json.dumps({'checks':len(checks),'failures':[k for k,v in checks.items() if not v],'all_checks_passed':report['all_checks_passed']}),flush=True);assert report['all_checks_passed']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--no-build',action='store_true');p.add_argument('--exact-only',action='store_true');a=p.parse_args();run(not a.no_build,a.exact_only)
