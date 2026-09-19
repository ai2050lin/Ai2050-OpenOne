"""Read-only API contract checks against original array values, with no model loading."""
from rdc_operator_common import *


def main():
    import sys
    sys.path.insert(0,str(ROOT))
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from server.rdc_operator_service import router
    app=FastAPI();app.include_router(router);client=TestClient(app);results=[];start=time.monotonic()
    def get(route,**params):
        response=client.get('/api/rdc-operator'+route,params=params)
        assert response.status_code==200,(route,params,response.status_code,response.text[:300])
        results.append({'route':route,'params':params,'status':response.status_code})
        return response.json()
    overview=get('/overview');assert overview['material']['sources']==2048
    ss=get('/samples');assert len(ss)==2048 and all(r['committed'] for r in ss)
    fixtures=[r for r in ss if r['full_field']];assert len(fixtures)==16
    for row in [fixtures[0],fixtures[-1]]:
        sid=row['sample_id'];scope='confirmation' if row['split']=='confirmation' else 'main'
        source=get('/sample',sample=sid);assert source['prompt_ids']
        raw=get('/field',sample=sid,mode='all_layers',anchor=1,view='raw')
        with np.load(BASE/'capture'/scope/'fields'/f'{sid}.npz') as z:expected=unbits(z['H'][:,1])
        assert np.array_equal(np.asarray(raw['values']),expected) and len(raw['values'])==37 and raw['end']==2560
        normalized=get('/field',sample=sid,mode='all_layers',anchor=1,view='RMS')
        assert np.max(np.abs(np.mean(np.asarray(normalized['values'])**2,1)-1))<1e-5
        standardized=get('/field',sample=sid,mode='all_layers',anchor=1,view='train_z')
        with np.load(BASE/'observation/training_scales.npz') as z:expected_z=(expected-z['mean'])/z['standard_deviation']
        assert np.allclose(standardized['values'],expected_z,rtol=1e-6,atol=1e-7)
        alltoken=get('/field',sample=sid,mode='all_tokens',layer=12)
        assert len(alltoken['values'])==len(source['prompt_ids']) and alltoken['end']==2560
    for b in (6,16,34):
        scalar=get('/scalar',sample=fixtures[0]['sample_id'],block=b,anchor=1,unit=9727,input_coordinate=2559,output_coordinate=2559)
        assert np.shape(scalar['input_terms']['values'])==(2,2560)
        assert np.shape(scalar['unit_terms']['values'])==(1,9728)
        assert abs(sum(scalar['unit_terms']['values'][0])-scalar['chain']['all_unit_sum'])<1e-8
    areas=get('/areas');array_files=0;arrays_total=0;registered_paths=set()
    for area in areas:
        files=get('/files',area=area['area']);assert len(files)==area['files'];array_files+=len(files)
        registered_paths.update((BASE/area['area']/item['file']).resolve() for item in files)
        if not files:continue
        headers=get('/arrays',area=area['area'],file=files[-1]['file']);assert headers
        arrays_total+=len(headers)
    actual_paths={p.resolve() for p in BASE.rglob('*.npz')}
    assert registered_paths==actual_paths,{'unregistered':[str(p) for p in actual_paths-registered_paths],
        'missing':[str(p) for p in registered_paths-actual_paths]}
    headers=get('/arrays',area='calculus',file='L6_complete_native_operators.npz')
    name=next(r['array'] for r in headers if r['shape']==[2560,2560])
    end=get('/array',area='calculus',file='L6_complete_native_operators.npz',name=name,row_start=2559,row_count=1,count=2560)
    assert end['row_end']==2560 and len(end['values'][0])==2560
    vocabulary=get('/array',area='compiled/confirmation/full_vocab',file='joint_selected.npz',name='native_logits',row_start=0,row_count=1,start=151935,count=1)
    assert vocabulary['end']==151936 and len(vocabulary['values'][0])==1
    for model in ('qwen4','qwen14','glm4'):
        for scope in ('main','confirmation'):
            qas=get('/qa-index',model=model,scope=scope)
            if qas:
                qa=get('/qa',model=model,scope=scope,id=qas[0]['question_id'])
                assert qa['prompt_ids'] and qa['generated_ids'] and qa['raw_query_field']
    qa=get('/qa',model='qwen4',scope='main',id=read(BASE/'figures/example_ids.json')['QA'])
    assert qa['typed_hyperedge']['hyperedge']['support_char_spans']
    gen=get('/generation-index');assert len(gen)==64
    g=get('/generation',sample=gen[0]['sample_id']);assert {'native','joint_global','joint_selected'}.issubset(g['branches'])
    if (BASE/'metric_followup/result.json').exists():
        assert 'output_selected_hybrid' in g['branches']
        assert g['branches']['output_selected_hybrid']['steps']
        geometry_files=get('/files',area='metric_followup/readout_geometry');assert len(geometry_files)==2
        for item in geometry_files:
            corner=get('/array',area='metric_followup/readout_geometry',file=item['file'],
                name='G_full_native_coordinates',row_start=2559,row_count=1,start=2559,count=1)
            with np.load(BASE/'metric_followup/readout_geometry'/item['file']) as z:
                assert corner['values'][0][0]==float(z['G_full_native_coordinates'][-1,-1])
    if (BASE/'operations/result.json').exists():
        oi=get('/operation-index');assert len(oi)==64
        op=get('/operation',id=oi[0]['question_id']);assert op['context_first'] and op['generated_ids']
    if (BASE/'metric_followup/population_geometry/result.json').exists():
        assert overview['metric_population']['queries']==640
        final=get('/array',area='metric_followup/population_geometry',file='pooled_complete_geometry.npz',
            name='G_average_full_native',row_start=2559,row_count=1,start=2559,count=1)
        with np.load(BASE/'metric_followup/population_geometry/pooled_complete_geometry.npz') as z:
            assert final['values'][0][0]==float(z['G_average_full_native'][-1,-1])
    for f in overview['figures']:
        response=client.get('/api/rdc-operator/figure/'+f['path']);assert response.status_code==200 and response.headers['content-type']=='image/png'
    for route,params,status in [('/download',{'area':'../','file':'x'},404),('/field',{'sample':fixtures[0]['sample_id'],'view':'wrong'},422),
        ('/field',{'sample':fixtures[0]['sample_id'],'layer':37},422),('/scalar',{'sample':fixtures[0]['sample_id'],'unit':9728},422),
        ('/array',{'area':'calculus','file':'../L6_complete_native_operators.npz','name':'K'},404)]:
        response=client.get('/api/rdc-operator'+route,params=params);assert response.status_code==status
        results.append({'route':route,'params':params,'status':status,'expected_rejection':True})
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'passed':True,'checks':results,'registered_array_files':array_files,
        'sampled_header_array_entries':arrays_total,'all_stored_npz_paths_registered':True,
        'native_full_fixture_count':len(fixtures),'figure_count':len(overview['figures']),
        'server_source':snapshot(ROOT/'server/rdc_operator_service.py'),'frontend_source':snapshot(ROOT/'frontend/src/components/app/RdcOperatorAtlas.jsx'),
        'scope':'In-process HTTP contract and original-number checks; no native model loaded. Browser canvas/controls are a separate check. Missing in-progress larger-model scopes are reported empty, never filled with demo results.'}
    save(BASE/'verification/client_contract.json',result);ledger('operator_read_only_client_contract',time.monotonic()-start)
    print('OPERATOR_CLIENT_PASS',len(results),array_files,flush=True)


if __name__=='__main__':main()
