"""Read-only API regression against actual committed native arrays and checkpoint bits."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from rdc_continuity_common import *
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.rdc_feature_service import router

def main():
    import torch
    from safetensors import safe_open
    app=FastAPI();app.include_router(router);client=TestClient(app);checks=[]
    def get(path,expected=200,**params):
        r=client.get('/api/rdc'+path,params=params)
        assert r.status_code==expected,(path,params,r.status_code,r.text[:300])
        checks.append({'path':path,'params':params,'status':r.status_code})
        return r.json()
    e=read(CAMPAIGN/'e_confirmation/material.json');r=e[0];sid=r['sample_id'];n=len(r['prompt_ids'])
    for view,layer,end in [('shape',36,2559),('ruler',24,2559),('gate',23,9727),('forecast',24,2559)]:
        sample=next(x['sample_id'] for x in e if x['word_split']=='test') if view=='forecast' else sid
        d=get('/continuity/inspect',sample=sample,view=view,layer=layer,coordinate=end,width=128)
        assert d['coordinate_count']==1 and d['no_topk'] and np.isfinite(d['values']).all()
        assert d['source_mode']!='actual_parameter_composition_NOT_causal_necessity'
    get('/continuity/inspect',422,sample=sid,view='forecast')
    get('/continuity/inspect',422,sample=sid,view='shape',coordinate=2560)
    d=get('/mechanism/parameter_path',run='e_confirmation',sample=sid,layer=23,token=n-1,unit=9727,coordinate=2559,input_coordinate=2559,output_coordinate=2559)
    assert d['scalar_chain']['input_coordinate']==2559 and d['scalar_chain']['output_coordinate']==2559
    get('/mechanism/parameter_path',422,run='e_confirmation',sample=sid,layer=23,token=n-1,unit=9728)
    for key,width,units in [('qwen14',5120,17408),('glm4',4096,13696)]:
        out=CAMPAIGN/'h_scale'/key;row=read(out/'material.json')[0];sample=row['sample_id'];pos=len(row['prompt_ids'])-1
        with np.load(out/f'fields/{sample}.npz') as z:
            h=unbits(z['h']);a=unbits(z['L39_a'])
        d=get(f'/runs/scale_{key}/field',sample=sample,field='h',layer=40,token=pos,coordinate=width-1,width=128)
        assert d['values'][0][0][0]==float(h[40,pos,width-1]) and d['coordinate_count']==1
        d=get(f'/runs/scale_{key}/field',sample=sample,field='a',layer=39,layers=1,coordinate=units-1,width=128)
        assert d['values'][0][0][0]==float(a[0,-1]) and d['coordinate_count']==1
        get(f'/runs/scale_{key}/field',422,sample=sample,field='h',layer=41)
        get(f'/runs/scale_{key}/field',422,sample=sample,field='h',coordinate=width)
        get(f'/runs/scale_{key}/field',422,sample=sample,field='q',layer=39)
        model='Qwen3-14B' if key=='qwen14' else 'glm4-9b-chat-hf'
        for component in ('gate','up','down'):
            rowindex=width-1 if component=='down' else units-1
            start=units-1 if component=='down' else width-1
            d=get('/parameter',model_name=model,layer=39,component=component,row=rowindex,start=start,count=16)
            index=read(ROOT/f'models/hf/{model}/model.safetensors.index.json')['weight_map']
            with safe_open(str(ROOT/f'models/hf/{model}'/index[d['key']]),framework='pt',device='cpu') as f:
                a=f.get_slice(d['key'])[d['physical_row']:d['physical_row']+1,start:start+1].clone()
            assert d['native_bits']==a.view(torch.uint16)[0].tolist()
            if key=='glm4' and component=='up':assert d['physical_row']==2*units-1
    grows=read(CAMPAIGN/'g_generation/material.json')
    for step in (0,1,2):
        row=next(r for r in grows if r['generation_step']==step);sample=row['sample_id']
        d=get('/mechanism/output_ledger',run='g_generation',sample=sample,coordinate=2559)
        assert d['coordinate_count']==1 and d['account']
        d=get('/mechanism/output_units',run='g_generation',sample=sample,coordinate=9727)
        assert d['coordinate_count']==1
        d=get('/mechanism/source_ledger',run='g_generation',sample=sample,token=row['query_position'],coordinate=31,width=1)
        assert d['token_ids']==[row['query_position']] and d['coordinate_count']==1
        get('/mechanism/source_ledger',422,run='g_generation',sample=sample,token=row['query_position']+1)
    get('/runs/unknown/material',404)
    get('/parameter',422,model_name='glm4-9b-chat-hf',component='up',row=13696)
    save(CAMPAIGN/'client_api_audit.json',{'timestamp':stamp(),'passed':True,'checks':checks,'count':len(checks),'mode':'CPU TestClient, real artifact/weight slices, no model load'})
    print('API_AUDIT',len(checks),'passed',flush=True)

if __name__=='__main__':main()
