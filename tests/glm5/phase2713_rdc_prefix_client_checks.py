"""Read-only API verification against saved arrays; results belong to the new campaign."""
import requests
from rdc_prefix_common import *
API='http://127.0.0.1:5001/api/rdc-prefix';checks=[]


def get(path,code=200):
    response=requests.get(API+path,timeout=60);assert response.status_code==code,(path,response.status_code,response.text[:300])
    checks.append({'path':path,'status':code});return response.json() if code==200 else None


def main():
    o=get('/overview');assert len(o['runs'])==4 and len(o['figures'])==8
    assert len(o['metrics'])==34 and all(r['H36_mse'] is not None for r in o['metrics'])
    for r in o['metrics']:
        assert r['KL'] is not None
        if 'hash' in r['model']:assert 'not fresh' in r['status']
    for run,n,d in [('qwen4',512,2560),('qwen4_confirmation',128,2560),('qwen14',64,5120),('glm4',64,4096)]:
        rows=get(f'/runs/{run}/samples');assert len(rows)==n
        for r in (rows[0],next(x for x in rows if x['language']=='zh'),rows[-1]):
            sid=r['sample_id'];detail=get(f'/runs/{run}/sample/{sid}?anchor=3');p=r['positions'][3]
            from tokenizers import Tokenizer
            md={'qwen4':'qwen3-4b','qwen4_confirmation':'qwen3-4b','qwen14':'Qwen3-14B','glm4':'glm4-9b-chat-hf'}[run]
            tok=Tokenizer.from_file(str(ROOT/'models/hf'/md/'tokenizer.json'))
            assert detail['graph']['observed_prefix']==tok.decode(detail['prompt_ids'][:p+1],skip_special_tokens=False)
            a=get(f'/runs/{run}/field/{sid}?anchor=3&count={d}');assert a['native_width']==d
            with np.load(CAMPAIGN/run/f'fields/{sid}.npz') as z:actual=unbits(z['h'][:,3])
            assert np.array_equal(np.asarray(a['values']),actual)
            b=get(f'/runs/{run}/field/{sid}?anchor=3&start={d-1}&count=1');assert np.array_equal(np.array(b['values'])[:,0],actual[:,-1])
            b=get(f'/runs/{run}/field/{sid}?anchor=3&start={d-1}&count=1&normalized=true');assert np.isfinite(np.array(b['values'])).all()
            get(f'/runs/{run}/field/{sid}?start={d}',422)
        sid=rows[0]['sample_id'];get(f'/runs/{run}/field/{sid}?view=bogus',422)
        if run.startswith('qwen4'):
            for k in (0,5):
                a=get(f'/runs/{run}/field/{sid}?view=mlp&anchor={k}&count=9728');assert len(a['values'][0])==9728
            a=get(f'/native/{sid}?run={run}&unit=9727&input_coordinate=2559&output_coordinate=2559');assert len(a['all_unit_contributions'])==9728
            assert abs(sum(a['all_unit_contributions'])-a['all_9728_unit_write_sum'])<1e-9
            test=next(x for x in rows if x['split'] in ('test','confirmation'))['sample_id']
            for model in ('early_linear','full_linear','graph_interaction','temporal_full_linear'):
                a=get(f'/prediction/{test}?run={run}&model={model}');assert np.array(a['values']).shape==(3,2560)
                assert np.allclose(np.array(a['values'])[1]-np.array(a['values'])[0],np.array(a['values'])[2],rtol=1e-5,atol=1e-5)
        else:get(f'/runs/{run}/field/{sid}?view=mlp',409)
    panel=next(r for r in get('/runs/qwen4/samples') if r['full_panel'])['sample_id']
    a=get(f'/runs/qwen4/field/{panel}?view=tokens&layer=36');assert np.array(a['values']).shape[1]==2560
    for r in read(CAMPAIGN/'atlas/matrix_index.json'):
        a=get(f'/matrix?matrix_id={r["id"]}&row=2559&column=2559');assert np.array(a['values']).shape==(1,1)
        response=requests.get(API+f'/matrix-file/{r["id"]}',timeout=60);assert response.status_code==200
        import hashlib
        assert hashlib.sha256(response.content).hexdigest()==sha(CAMPAIGN/r['path'])
    for f in o['figures']:
        response=requests.get(API+'/figures/'+f['id'],timeout=60);assert response.status_code==200 and response.content[:8]==b'\x89PNG\r\n\x1a\n'
    get('/runs/unknown/samples',404);get('/matrix?matrix_id=unknown',404)
    h=get('/history/overview');assert h['selected_before_fresh']=='current' and len(h['metrics'])==8
    for scope,n in [('main',512),('fresh',64)]:
        rr=get('/history/samples?scope='+scope);assert len(rr)==n
        for r in (rr[0],rr[-1]):
            sid=r['sample_id'];a=get(f'/history/field?scope={scope}&sample={sid}');assert len(a['values'])==r['tokens'] and a['native_width']==2560
            from phase2714_rdc_full_source_history import source_field
            raw=next(x for x in (read(CAMPAIGN/'material_stratified.json') if scope=='main' else read(CAMPAIGN/'full_source_history/fresh_material.json')) if x['sample_id']==sid)
            assert np.array_equal(np.asarray(a['values']),source_field(raw,scope=='fresh'))
            b=get(f'/history/field?scope={scope}&sample={sid}&start=2559&count=1');assert np.array_equal(np.asarray(b['values'])[:,0],np.asarray(a['values'])[:,-1])
            b=get(f'/history/field?scope={scope}&sample={sid}&normalized=true');assert np.isfinite(np.asarray(b['values'])).all()
        sid=next(r for r in rr if r['split'] in ('test','confirmation'))['sample_id']
        for rule in ('current','mean_history','absolute_history','relative_history'):
            for anchor in (0,1):
                a=get(f'/history/prediction?scope={scope}&sample={sid}&rule={rule}&anchor={anchor}');v=np.asarray(a['values'])
                assert v.shape==(3,2560) and np.allclose(v[1]-v[0],v[2],rtol=1e-5,atol=1e-5)
        a=get('/history/relations?scope='+scope);assert np.asarray(a['values']).shape==(14,2560)
        b=get('/history/relations?scope='+scope+'&start=2559&count=1');assert np.array_equal(np.asarray(a['values'])[:,-1],np.asarray(b['values'])[:,0])
    get('/history/samples?scope=unknown',422);get('/history/field?sample=unknown',404)
    # Legacy endpoints remain read-only; do not overwrite prior campaign QA files.
    legacy=requests.get('http://127.0.0.1:5001/api/rdc/runs',timeout=60);assert legacy.status_code==200
    save(CAMPAIGN/'verification/api_checks.json',{'passed':True,'timestamp':stamp(),'checks':checks,'legacy_runs_available':True,
      'scope':'Native full matrices/end coordinates, decoder prefix, normalized fields, prediction differences, all9728 scalar ledger, all registered matrices and figures; not all repository APIs.'})
    print('PREFIX_API_CHECKS_PASS',len(checks),flush=True)


if __name__=='__main__':main()
