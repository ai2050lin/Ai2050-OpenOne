"""Actual local HTTP publication checks, including legacy readonly API preservation."""
import json,sys,urllib.request,urllib.error,urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
OUT=RESULT/'phase2676_native_mlp_delivery';BASE='http://127.0.0.1:5001/api/research-assets'

def get(path):
    with urllib.request.urlopen(BASE+path,timeout=60) as r:return json.load(r)

def main():
    checks={};meta=get('/native-mlp-cases');checks['actual_mlp_cases_16']=len(meta['cases'])==16
    obj=get('/native-mlp-parameter?'+urllib.parse.urlencode({'case':meta['cases'][0]['case'],'layer':23,'unit':6197,'coordinate':0,'checkpoint':24,'token':0}));checks['actual_weight_and_hidden_fields']=all(k in obj['values'] for k in ('actual_Wgate_jk','actual_Wup_jk','actual_Wdown_kj','embedding_coordinate','hidden_coordinate'))
    checks['allthree_fourpart_derivatives']=all(set(v)=={'all','content','format','eos'} for v in obj['scalar_sequence_derivatives'].values())
    try:get('/native-mlp-parameter?coordinate=-1');checks['negative_bounds400']=False
    except urllib.error.HTTPError as e:checks['negative_bounds400']=e.code==400
    cat=get('/native-atlas-panels');checks['all34panels']=len(cat['panels'])==34
    for p in cat['panels']:
        r=get('/native-atlas-rows?'+urllib.parse.urlencode({'panel':p['key'],'start':len(p['rows'])-1,'count':8}));checks['HTTP_'+p['key']]=r['coordinate_count']==len(r['rows'][0]['values'])==p['coordinate_count']
    for path in ('/native-parameter?model=qwen4&case=0&j=0&k=0','/native-parameter?model=qwen14&case=0&j=0&k=0','/native-parameter?model=glm4&case=0&j=0&k=0','/native-parameter?model=ds7&case=0&j=0&k=0','/native-multitoken-cases','/native-sequence-cases'):
        checks['legacy'+path]=bool(get(path))
    req=urllib.request.Request(BASE+'/file/research_kernel/c42641_output_conditioned_crossmodel_field.json',headers={'Range':'bytes=0-1023'})
    with urllib.request.urlopen(req,timeout=60) as r:checks['legacy_asset_range206']=r.status==206 and len(r.read())==1024
    result={'checks':checks,'all_checks_passed':all(checks.values()),'count':len(checks),'exact_first_query':obj,'scope':'RealHTTP localCPU artifactbackend, not mockFastAPI andnot GPUmodel requests.'};save(OUT/'analysis/live_api_checks.json',result);print(json.dumps({'checks':len(checks),'all_passed':result['all_checks_passed']}));assert result['all_checks_passed']

if __name__=='__main__':main()
