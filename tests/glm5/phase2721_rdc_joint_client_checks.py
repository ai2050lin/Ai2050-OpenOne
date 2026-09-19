"""Read-only API contracts against actual persisted arrays, without the model server."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server import rdc_joint_service as api
from rdc_joint_common import *


def main():
    app=FastAPI();app.include_router(api.router);client=TestClient(app);checks=[]
    def get(path,params=None,status=200):
        r=client.get('/api/rdc-joint/'+path,params=params)
        assert r.status_code==status,(path,params,r.status_code,r.text[:500])
        checks.append({'endpoint':path,'params':params,'status':status})
        return r.json() if status==200 and 'application/json' in r.headers.get('content-type','') else r
    overview=get('overview');assert overview['material']['main_units']==512
    material=rows(True);r=material[0];sid=r['sample_id'];z=field(r,True)
    assert len(get('samples',{'scope':'fresh'}))==256
    detail=get('sample',{'scope':'fresh','sample':sid});assert detail['archive_status']
    for part in ('h12','h23','h36','postnorm'):
        a=get('field',{'sample':sid,'layer':part});assert np.array_equal(a['values'],unbits(z[part]))
    a=get('field',{'sample':sid,'layer':'all_layers','position_index':5});assert np.array_equal(a['values'],unbits(z['layers'][:,5]))
    a=get('field',{'sample':sid,'layer':'h12','view':'source_RMS','start':2557,'count':3})
    raw=unbits(z['h12']).astype(float);expected=raw/np.sqrt(np.mean(raw**2,1,keepdims=True));assert np.max(abs(np.array(a['values'])-expected[:,2557:]))<1e-6
    for choice,name in overview['choices'].items():
        a=get('prediction',{'sample':sid,'choice':choice,'anchor':1})
        scope='temporal' if choice.startswith('temporal') else 'current'
        with np.load(BASE/'confirmation/predictions'/f'{scope}_{name}.npz') as zz:expected=zz['prediction'][1,-2560:]
        assert np.array_equal(a['values'][1],expected)
    for block in (6,12,23):
        a=get('factors',{'sample':sid,'block':block,'part':'activation'})
        with np.load(BASE/'native_factors/fields'/f'{sid}.npz') as zz:assert np.array_equal(a['values'],unbits(zz[f'L{block}_activation']))
    a=get('scalar',{'sample':sid,'block':6,'unit':9727,'input_coordinate':2559,'output_coordinate':2559})
    assert np.allclose(np.array(a['input_terms']['values']).sum(1),a['chain']['all_input_sums'],rtol=0,atol=1e-10)
    assert abs(np.array(a['unit_terms']['values']).sum()-a['chain']['all_unit_sum'])<1e-10
    for view in ('raw','train_z','source_RMS'):
        a=get('matrix',{'relation':'ud:nmod','control':'exact_distance_POS_noninitial','view':view,'row_start':2552,'column_start':2552,'count':8})
        p=BASE/'relation_atlas/profiles'/f'ud_nmod_exact_distance_POS_noninitial_{view}.npz'
        with np.load(p) as zz:expected=zz['test_diagonal'][2552:]
        assert np.allclose(np.diag(a['values']),expected,rtol=5e-5,atol=1e-5),(view,np.max(abs(np.diag(a['values'])-expected)))
    for branch in ('MSE_embedding','KL_embedding','KL_history'):
        a=get('generation',{'sample':sid,'branch':branch});assert len(a['field']['values'])==64
    entries=get('analysis-index');item=next(e for e in entries if e['array']=='mean' and e['file'].startswith('layer_atlas'))
    a=get('analysis-field',{'id':item['id'],'row_start':128,'row_count':3,'start':2558,'count':2})
    with np.load(BASE/item['file']) as zz:assert np.array_equal(a['values'],zz[item['array']].reshape(-1,2560)[128:131,2558:])
    for endpoint,params,status in [('field',{'sample':sid,'layer':'h36','view':'train_z'},422),('field',{'sample':sid,'start':2560},422),
        ('sample',{'sample':'../../resource_allocation'},404),('samples',{'scope':'../../main'},422),
        ('prediction',{'sample':sid,'choice':'not-a-choice'},422),('scale-field',{'sample':'../../x'},404),
        ('scalar',{'sample':sid,'unit':9728},422),('analysis-field',{'id':'../../x'},404),
        ('matrix',{'relation':'not-recorded'},422)]:get(endpoint,params,status)
    empty=next(e for e in api.compressed(BASE/'relation_atlas/pair_index.json.gz') if not e['controls']['same_dependent_ID'])
    # Find a genuinely absent matched relation/split set; never equate one empty edge with no data.
    pairs=api.compressed(BASE/'relation_atlas/pair_index.json.gz')
    absent=next((rel for rel in sorted({e['relation'] for e in pairs}) if not any(e['relation']==rel and e['split']=='test' and e['controls']['same_dependent_ID'] for e in pairs)),None)
    if absent:get('matrix',{'relation':absent,'control':'same_dependent_ID'},409)
    get('download',{'sample':sid});get('figure/'+overview['figures'][0]['path'])
    models=[]
    for model in ('qwen4','qwen14','glm4'):
        rr=get('scale-samples',{'model':model})
        if not rr:continue
        a=get('scale-field',{'model':model,'sample':rr[0]['sample_id']})
        with np.load(BASE/'scale'/model/'fields'/f'{rr[0]["sample_id"]}.npz') as zz:assert np.array_equal(a['values'],unbits(zz['late']))
        models.append({'model':model,'currently_committed_sources':len(rr),'width':a['native_width']})
    save(BASE/'verification/client_api.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'checks':checks,'scale':models,
        'all_passed':True,'scope':'Numerical and HTTP validation in isolated TestClient; live browser is a separate acceptance check.'})
    print('JOINT_CLIENT_API_PASS',len(checks),models,flush=True)


if __name__=='__main__':main()
