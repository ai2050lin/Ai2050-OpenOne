"""In-process read-only API regression; no model load or CUDA context."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['AI2050_SKIP_MODEL_LOAD']='1'
from rdc_conditional_common import *
sys.path.insert(0,str(ROOT))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.rdc_feature_service import router


def main():
    app=FastAPI();app.include_router(router);client=TestClient(app);checks=[]
    def get(path,code=200):
        r=client.get('/api/rdc'+path);assert r.status_code==code,(path,r.status_code,r.text[:300]);checks.append({'path':path,'status':r.status_code});return r.json()
    for run in ('i_factorial','k_long','m_order','o_generalization','aligned_qwen4','aligned_qwen14','aligned_glm4','e_confirmation','g_generation','scale_qwen14','scale_glm4'):
        get('/runs/'+run+'/status');get('/runs/'+run+'/results');rows=get('/runs/'+run+'/material')['samples']
        if not rows:continue
        row=next(r for r in rows if r['committed']);sid=row['sample_id']
        if run=='o_generalization':row=next(r for r in rows if r['analysis_selected']);sid=row['sample_id']
        if run in ('i_factorial','k_long','m_order','o_generalization') or run.startswith('aligned_'):
            d=5120 if run=='aligned_qwen14' else 4096 if run=='aligned_glm4' else 2560;j=17408 if d==5120 else 13696 if d==4096 else 9728;l=39 if d>2560 else 35
            if run=='o_generalization':l=23
            h=get(f'/runs/{run}/field?sample={sid}&layer={40 if d>2560 else 36}&coordinate={d-1}&width=1');assert h['shown_values']==1
            p=get(f'/conditional/parameter_path?run={run}&sample={sid}&layer={l}&unit={j-1}&input_coordinate={d-1}&output_coordinate={d-1}&coordinate={d-1}&width=1');assert p['shown_values']==5 and p['scalar_chain']['input_coordinate']==d-1
            get(f'/conditional/parameter_path?run={run}&sample={sid}&layer={l}&unit={j}',422)
        if run=='i_factorial':
            get(f'/runs/{run}/field?sample={sid}&field=h_full');get(f'/runs/{run}/field?sample={sid}&field=u_mean')
            get(f'/conditional/inspect?run={run}&sample={sid}&view=gate&layer=23&coordinate=9727&width=1')
            test=next(r for r in rows if r['word_split']=='test')['sample_id']
            for view in ('forecast_a','forecast_down'):get(f'/conditional/inspect?run={run}&sample={test}&view={view}')
        if run=='k_long':
            b=get(f'/runs/{run}/field?sample={sid}');assert b['behavior']['scores']['score_version']==2
            test=next(r for r in rows if r['word_split']=='test' and r['analysis_selected'])['sample_id']
            get(f'/conditional/inspect?run={run}&sample={test}&view=forecast_h')
            get(f'/runs/{run}/field?sample={sid}&field=logits&coordinate=151935&width=1')
        if run=='m_order':
            b=get(f'/runs/{run}/field?sample={sid}');assert b['behavior']['scores']['score_version']==2
            boundary=next((r['sample_id'] for r in rows if r.get('result_field_onset')),None)
            if boundary:
                for field in ('v','p'):get(f'/runs/{run}/field?sample={boundary}&field={field}&layer=23&token=0')
                if (CAMPAIGN/f'm_order/ledgers/{boundary}.npz').exists():
                    z=get(f'/conditional/source_groups?sample={boundary}&layer=23&coordinate=2559&width=1');assert z['shown_values']==9
            if (CAMPAIGN/'m_order/result.json').exists():
                assert boundary and len({r['prefix_id'] for r in rows})==288
                zh=next(r for r in rows if r.get('result_field_onset') and r['language']=='zh')
                get(f'/conditional/source_groups?sample={zh["sample_id"]}&layer=11&coordinate=2559&width=1')
                get(f'/runs/{run}/field?sample={boundary}&field=logits&coordinate=151935&width=1')
                get(f'/conditional/source_groups?sample={boundary}&layer=12',422)
                test=next(r for r in rows if r.get('result_field_onset') and r['word_split']=='test')['sample_id']
                get(f'/conditional/inspect?run={run}&sample={test}&view=forecast_attention&coordinate=2559&width=1')
        if run=='o_generalization' and (CAMPAIGN/'o_generalization/result.json').exists():
            assert len({r['prefix_id'] for r in rows})==512
            assert all(r['target']=='L23 attention output' and r['split']=='frozen_new_material' for r in get(f'/runs/{run}/results')['result']['results'])
            get(f'/conditional/inspect?run={run}&sample={sid}&view=forecast_attention&coordinate=2559&width=1')
            get(f'/runs/{run}/field?sample={sid}&field=p&layer=23&coordinate=31&width=1&token=0')
            get(f'/runs/{run}/field?sample={sid}&field=v&layer=23&coordinate=1023&width=1&token={row["query_position"]}')
            z=get(f'/runs/{run}/field?sample={sid}&field=logits&coordinate=151935&width=1');assert z['original_input']['source']
            test=next(r for r in rows if r['analysis_selected'] and r['unit']>=12)['sample_id']
            z=get(f'/conditional/inspect?run={run}&sample={test}&view=forecast_token_conditioned&coordinate=2559&width=1');assert z['shown_values']==6 and 'NOT_independent' in z['source_mode']
            get(f'/conditional/inspect?run={run}&sample={sid}&view=forecast_token_conditioned',422)
    for name in ('i_interactions','i_field','j_units','k_errors','l_readers','m_sources','n_attention','o_attention','p_attention'):
        if name=='m_sources' and not (CAMPAIGN/'m_order/result.json').exists():continue
        if name in ('n_attention','o_attention') and not (CAMPAIGN/'n_cached_attention/figures/display_contract.json').exists():continue
        get('/figures/'+name+'?asset=contract')
        r=client.get('/api/rdc/figures/'+name);assert r.status_code==200 and r.headers['content-type']=='image/png'
        checks.append({'path':'/figures/'+name,'status':200,'image_bytes':len(r.content)})
    get('/figures/unlisted',404)
    save(CAMPAIGN/'client_api_audit.json',{'timestamp':stamp(),'passed':True,'checks':checks,'source_sha':sha(Path(__file__))});print('API_CHECKS',len(checks),'PASS',flush=True)


if __name__=='__main__':main()
