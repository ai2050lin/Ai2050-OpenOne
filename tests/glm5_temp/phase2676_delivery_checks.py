"""Direct readonly API checks against persisted raw coordinates and independent scalar reductions."""
import argparse,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/glm5'))
from fastapi import HTTPException
from phase2620_native_coordinate_contract import *
from phase2670_native_mlp_contract import FIELD,SITES,LAYERS
from phase2674_native_mlp_scalar import OUT as FP
from phase2676_native_mlp_delivery import OUT
from server.native_mlp_parameter_query import options,query,native,precise,metadata
from server.native_atlas_heatmap_query import options as panels,rows as panel_rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--post-cleanup',action='store_true');p.add_argument('--build',action='store_true');args=p.parse_args();checks={};meta=options();checks['16MLPcases']=len(meta['cases'])==16
    ref={r['case_index']:r for r in read(FP/'analysis/records.json')};sites=read(FP/'protocol/frozen.json')['sites'];n=0;maxderiv=0.
    for case in meta['cases']:
        ci=case['case'];raw=native(ci);r=next(r for r in metadata() if r['case_index']==ci)
        for si,s in enumerate(sites):
            k=s['coordinate'];l=s['layer'];j=s['unit'];li=LAYERS.index(l);obj=query(ci,l,j,k,0,0);n+=1
            checks[f'{ci}_{si}_actual_embedding']=obj['values']['embedding_coordinate']==float(raw['full__h'][0,0,k])
            checks[f'{ci}_{si}_actual_gate']=obj['values']['gate_unit']==float(raw['full__gate'][li,0,j])
            weight_key={'gate':'actual_Wgate_jk','up':'actual_Wup_jk','down':'actual_Wdown_kj'}[s['kind']]
            checks[f'{ci}_{si}_actual_weight']=obj['values'][weight_key]==s['original']
            validation=next(v for v in obj['actual_scalar_validation'] if v['site_index']==si)
            checks[f'{ci}_{si}_actual_finite_effects']=validation['effects']==[e for e in ref[ci]['effects'] if e['indices']==[si]] and validation['original_weight']==s['original']
            for part in ('all','content','format','eos'):maxderiv=max(maxderiv,abs(obj['scalar_sequence_derivatives'][s['kind']][part]-ref[ci]['gradients'][part][si]))
        # Physically last coordinate and last token at finalcheckpoint, independently grounded in raw tensor.
        obj=query(ci,28,8513,2559,36,None);checks[f'{ci}_last_physical_H']=obj['values']['hidden_coordinate']==float(raw['full__h'][36,-1,2559]);checks[f'{ci}_all_token_terms']=len(obj['all_token_scalar_terms']['gate']['Y'])==len(ref[ci]['gradient_info']['branches'][0]['input_ids'])
    checks['480scalar_queries']=n==480;checks['1920_derivatives_match_saved']=maxderiv<1e-8
    for label,kwargs in [('unpublished',dict(case=0)),('bad_unit',dict(unit=6198)),('bad_coordinate',dict(coordinate=2560)),('negative_token',dict(token=-1)),('bad_checkpoint',dict(checkpoint=37))]:
        params=dict(case=meta['cases'][0]['case'],layer=23,unit=6197,coordinate=0,checkpoint=0,token=0);params.update(kwargs)
        try:query(**params);checks['reject_'+label]=False
        except HTTPException as e:checks['reject_'+label]=e.status_code in (400,404)
    cat=panels();new=[p for p in cat['panels'] if p['key'].startswith('phase2676_')];old=[p for p in cat['panels'] if not p['key'].startswith('phase2676_')]
    checks['10_old_panels_retained']=len(old)==10;checks['24_new_panels']=len(new)==24
    for info in cat['panels']:
        source=OUT if info['key'].startswith('phase2676_') else RESULT/'phase2669_symmetric_multitoken_delivery'
        with np.load(source/'maps/client_panels'/(info['key']+'.npz')) as z:
            for start in (0,len(info['rows'])-1):
                got=panel_rows(info['key'],start,8);checks[info['key']+f'_row{start}']=got['coordinate_count']==z['values'].shape[1] and np.array_equal(np.asarray(got['rows'][0]['values']),z['values'][start])
    checks['native_api_imports_no_model']=not any(s in (ROOT/'server/native_mlp_parameter_query.py').read_text() for s in ('import torch','transformers','load_model('))
    if args.build:
        node='C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe';run=subprocess.run([node,'node_modules/vite/bin/vite.js','build'],cwd=ROOT/'frontend',capture_output=True,text=True,encoding='utf-8');checks['frontend_production_build']=run.returncode==0
        save(OUT/'analysis/build_output.json',{'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr})
    result={'checks':checks,'all_checks_passed':all(checks.values()),'count':len(checks),'queries':n,'maximum_scalar_difference':maxderiv,'scope':'Published numeric API +allold/newpanel edges; not browservisual QA.'}
    save(OUT/f'analysis/{"post_cleanup_checks" if args.post_cleanup else "delivery_checks"}.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}));assert result['all_checks_passed'],[k for k,v in checks.items() if not v]

if __name__=='__main__':main()
