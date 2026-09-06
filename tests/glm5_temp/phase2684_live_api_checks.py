"""Real HTTP evidence, native scalar and full-width row values; no model inference."""
import argparse,json,sys,urllib.request,urllib.error,urllib.parse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2684_source_campaign_delivery import OUT,SCALAR

BASE='http://127.0.0.1:5001/api/research-assets'


def get(path):
    with urllib.request.urlopen(BASE+path,timeout=60) as r:return json.load(r)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--post-cleanup',action='store_true');a=ap.parse_args();checks={}
    meta=get('/native-source-cases');checks['HTTP128published_source_cases']=len(meta['cases'])==128;checks['HTTP30native_scalar_controls']=len(meta['scalar_controls'])==30
    results=[];records={r['case_index']:r for r in read(SCALAR/'analysis/records.json')}
    for dataset in ('original','fresh'):
        subset=[r for r in meta['cases'] if r['dataset']==dataset]
        for r in (subset[0],subset[-1]):
            for s in (meta['scalar_controls'][0],meta['scalar_controls'][1],meta['scalar_controls'][-1]):
                params=dict(dataset=dataset,case=r['case'],layer=s['layer'],unit=s['unit'],coordinate=s['coordinate'],checkpoint=0,source_token=r['tokens']-1,head=31,head_coordinate=127,query_position=1,hidden_token=0)
                obj=get('/native-source-parameter?'+urllib.parse.urlencode(params));results.append(obj)
                checks[f'{dataset}_{r["case"]}_{s["key"]}_{s["control"]}_EequalsH0']=obj['values']['embedding_coordinate']==obj['values']['hidden_coordinate']
                checks[f'{dataset}_{r["case"]}_{s["key"]}_{s["control"]}_learned_weight']=obj['values'][{'gate':'actual_Wgate_jk','up':'actual_Wup_jk','down':'actual_Wdown_kj'}[s['kind']]]==s['original_weight']
                if dataset=='fresh':
                    expected=[c for c in records[r['case']]['conditions'] if (c['layer'],c['unit'],c['coordinate'])==(s['layer'],s['unit'],s['coordinate'])]
                    checks[f'fresh_{r["case"]}_{s["key"]}_{s["control"]}_actual_conditions']=obj['numeric_scalar_validation']['effects']==expected and len(expected)>=4
    cat=get('/native-atlas-panels');new=[p for p in cat['panels'] if p['key'].startswith('phase2684_')]
    compact=get('/native-atlas-panels?compact=true')
    checks['real_HTTP_compact_preserves_panel_axes']=len(compact['panels'])==len(cat['panels']) and all(
        'rows' not in small and small['key']==large['key'] and small['row_count']==len(large['rows'])
        and small['coordinate_count']==large['coordinate_count'] for small,large in zip(compact['panels'],cat['panels']))
    checks['34legacy_types_preserved']=len(cat['panels'])-len(new)==34
    catalog=read(OUT/'material/client_panel_catalog.json');checks['all_new_types_HTTP']=len(new)==len(catalog['panels'])
    for p in cat['panels']:
        start=len(p['rows'])-1;obj=get('/native-atlas-rows?'+urllib.parse.urlencode(dict(panel=p['key'],start=start,count=8)))
        checks['allcols_'+p['key']]=len(obj['rows'])==1 and len(obj['rows'][0]['values'])==p['coordinate_count']
        if p['key'].startswith('phase2684_'):
            descriptor=next(v for v in catalog['panels'] if v['key']==p['key'])['rows'][start]
            with np.load(RESULT/descriptor['file']) as z:raw=z[descriptor['array']][tuple(descriptor['index'])]
            if descriptor.get('encoding')=='native_bf16':raw=(raw.astype(np.uint32)<<16).view(np.float32)
            checks['exact_values_'+p['key']]=np.array_equal(np.array(obj['rows'][0]['values']),raw)
    first=meta['cases'][0]
    for tail in ('coordinate=-1','head=32','checkpoint=37','source_token=-1','dataset=unknown'):
        try:get(f'/native-source-parameter?case={first["case"]}&dataset={first["dataset"]}&'+tail);checks['reject_'+tail]=False
        except urllib.error.HTTPError as e:checks['reject_'+tail]=e.code in (400,404)
    for path in ('/native-parameter?model=qwen4&case=0&j=0&k=0','/native-parameter?model=qwen14&case=0&j=0&k=0','/native-parameter?model=glm4&case=0&j=0&k=0','/native-parameter?model=ds7&case=0&j=0&k=0','/native-multitoken-cases','/native-sequence-cases','/native-mlp-cases'):
        checks['legacy_'+path]=bool(get(path))
    req=urllib.request.Request(BASE+'/file/research_kernel/c42641_output_conditioned_crossmodel_field.json',headers={'Range':'bytes=0-1023'})
    with urllib.request.urlopen(req,timeout=60) as r:checks['legacy_asset_range206']=r.status==206 and len(r.read())==1024
    result={'all_checks_passed':all(checks.values()),'checks':checks,'real_HTTP':True,'post_cleanup':a.post_cleanup,'new_panels':len(new),'source_queries':len(results),
            'scope':'Actual5001HTTP and exact values against original arrays; not mockHTTP or modeltests. Not visualbrowserQA.'}
    save(OUT/f'analysis/{"post_cleanup_checks" if a.post_cleanup else "live_api_checks"}.json',result)
    save(OUT/'analysis/HTTP_source_examples.json',results);print({k:v for k,v in result.items() if k!='checks'});assert all(checks.values()),[k for k,v in checks.items() if not v]


if __name__=='__main__':main()
