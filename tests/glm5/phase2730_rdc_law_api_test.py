"""Read-only client contract and native-coordinate boundary checks; no model load."""
from rdc_law_common import *


def main():
    import sys
    sys.path.insert(0,str(ROOT))
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from server.rdc_law_service import router,arrays_index
    app=FastAPI();app.include_router(router);client=TestClient(app);checks=[];start=time.monotonic()
    def check(path,params=None,status=200):
        r=client.get('/api/rdc-law'+path,params=params);assert r.status_code==status,(path,params,r.status_code,r.text[:500])
        checks.append({'path':path,'params':params,'status':status});return r.json() if 'json' in r.headers.get('content-type','') else r
    overview=check('/overview');assert overview['material']['main_rows']==960
    if (BASE/'deployment/paired_analysis.json').exists():
        assert len(overview['deployment_paired']['scale_combination_visibility'])==3
        assert len(overview['deployment_paired']['strata'])==56
    samples=check('/samples');assert len(samples)==1152
    for cohort in ('gum','ewt','cmrc','squad_qa','cmrc_qa','hotpot_qa'):
        row=next(r for r in samples if r['cohort']==cohort and r['split']=='train')
        source=check('/sample',{'sample':row['sample_id']});assert source['prompt_ids']
        for view in ('raw','RMS','train_z'):
            r=check('/field',{'sample':row['sample_id'],'anchor':len(row['anchors'])-1,'view':view});assert np.asarray(r['values']).shape==(37,2560)
        s=check('/scalar',{'sample':row['sample_id'],'block':35,'anchor':0,'unit':9727,'input_coordinate':2559,'output_coordinate':2559})
        assert len(s['input_terms']['values'][0])==2560 and len(s['unit_terms']['values'][0])==9728
    fixture=next(r for r in samples if r['full_field']);r=check('/field',{'sample':fixture['sample_id'],'mode':'all_tokens','layer':12});assert len(r['values'])==fixture['tokens']
    missing=next(r for r in samples if not r['full_field']);check('/field',{'sample':missing['sample_id'],'mode':'all_tokens'},409)
    qa=next(r for r in samples if r['kind']=='QA');check('/field',{'sample':qa['sample_id'],'anchor':2},422)
    check('/field',{'sample':'../../private'},404)
    for panel in (0,287):
        r=check('/gradient',{'panel_index':panel,'unit':9727,'input_coordinate':2559,'output_coordinate':2559})
        assert len(r['output_gradient_terms']['values'][0])==9728
        assert len(r['all_query_gradient_inner_products']['values'][0])==288
    areas=check('/areas');assert areas
    profiles=check('/files',{'area':'atlas/condition_profiles'});assert len(profiles)==70 and all(r['label']!=r['file'] for r in profiles)
    inventory=arrays_index()
    selected=['atlas/joint_products/L35_complete_unit_ledger.npz','formation/trajectories/coherent_seed2728/final_native_parameter_deltas.npz',
              'prediction/training_features.npz','confirmation/training/initial_full_panel.npz','scale/qwen4/feature_rulers.npz']
    for key in selected:
        area,file=key.rsplit('/',1);check('/files',{'area':area});headers=check('/arrays',{'area':area,'file':file})
        h=next(r for r in headers if r['shape'] and np.dtype(r['dtype']).kind in 'buif')
        shape=h['shape'];n=int(np.prod(shape[:-1])) if len(shape)>1 else 1
        result=check('/array',{'area':area,'file':file,'name':h['array'],'row_start':n-1,'row_count':1,'start':shape[-1]-1,'count':1})
        with np.load(inventory[key]) as z:a=z[h['array']].reshape(-1,shape[-1])
        actual=unbits(a) if a.dtype==np.uint16 else a
        assert result['values']==[[float(actual[-1,-1])]]
        check('/array',{'area':area,'file':file,'name':h['array'],'row_start':n},422)
        check('/array',{'area':area,'file':file,'name':h['array'],'start':shape[-1]},422)
    check('/arrays',{'area':'../../models','file':'model.safetensors'},404)
    check('/download',{'area':'capture/main/fields','file':'../resources.json'},404)
    for model in ('live','own_history','qwen4','qwen14','glm4'):
        index=check('/behavior-index',{'model':model})
        if index:
            r=index[0];data=check('/behavior',{'model':model,'sample':r['sample_id'],'branch':r['branch']});assert data['generated_ids']
    check('/behavior',{'model':'qwen4','sample':'not-a-case'},404)
    for f in overview['figures']:check('/figure/'+f['path'])
    on_disk={p.relative_to(BASE).as_posix() for p in BASE.rglob('*.npz') if p.is_file()}
    assert set(arrays_index())==on_disk
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'server_source':snapshot(ROOT/'server/rdc_law_service.py'),
        'passed':True,'checks':checks,'checks_count':len(checks),'registered_npz_at_start':len(inventory),'seconds':time.monotonic()-start,
        'all_stored_npz_paths_registered':True,'registered_array_files':len(on_disk),
        'scope':'In-process read-only API contract. Array finalscalar values checked against exactNPZ, noCUDA model loaded. Complete registry coverage matches the files at this audit time; later scientific/finalchecks must agree.'}
    save(BASE/'verification/client_api.json',result);ledger('read_only_law_client_API',result['seconds'])
    print('LAW_API_CHECKS_PASS',len(checks),flush=True)


if __name__=='__main__':main()
