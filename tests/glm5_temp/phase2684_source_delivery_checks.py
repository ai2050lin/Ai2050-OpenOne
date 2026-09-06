"""Independent published-array checks; preview mode does not claim live delivery."""
import argparse,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/glm5'))
from fastapi import HTTPException
from phase2620_native_coordinate_contract import *
from phase2684_source_campaign_delivery import OUT,FIELD,SOURCE
import server.native_source_parameter_query as source
import server.native_atlas_heatmap_query as atlas


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');ap.add_argument('--build',action='store_true');a=ap.parse_args();prefix='staged_' if a.preview else '';checks={}
    data=read(OUT/f'material/{prefix}published_source_cases.json');cat=read(OUT/f'material/{prefix}client_panel_catalog.json')
    original=atlas.catalog()
    if a.preview:
        # Only metadata route selection is redirected; values are actual measured arrays.
        source.metadata=lambda:data;atlas.catalog=lambda:{**original,'panels':original['panels']+cat['panels']}
    checks['128_actual_published_sources']=len(source.options()['cases'])==128
    compact=atlas.options(include_rows=False);full=atlas.options()
    checks['compact_catalog_same_panels_and_full_row_counts']=len(compact['panels'])==len(full['panels']) and all(
        'rows' not in small and small['key']==large['key'] and small['row_count']==len(large['rows'])
        and small['coordinate_count']==large['coordinate_count'] for small,large in zip(compact['panels'],full['panels']))
    with np.load(SOURCE/'weights/native_source_weights.npz') as z:w={k:source.decode(z[k]).astype(np.float64) for k in z.files}
    maxerror=0.;n=0
    for r in data:
        with np.load(RESULT/r['source_path']) as z:sp={k:source.decode(z[k]).astype(np.float64) for k in z.files}
        with np.load(RESULT/r['native_path']) as z:hs=z['full__h']
        for ii,(l,j) in enumerate(source.SITES):
            k=(0,84,533,1217,2559)[ii];q=ii%2;s=min(len(r['token_strings'])-1,ii*7);h=(ii*7)%32;d=127-ii
            got=source.query(r['dataset'],r['case_index'],l,j,k,36,s,h,d,q,0);v=got['values'];p=sp[f'L{l}__actual_probability'];vv=sp[f'L{l}__actual_value'];wo=w[f'L{l}__Wo']
            expected=float(p[q,h,s]*sum(float(vv[s,h//4,t])*float(wo[k,h*128+t]) for t in range(128)))
            maxerror=max(maxerror,abs(v['single_head_source_output_coordinate']-expected));assert abs(v['single_head_source_output_coordinate']-expected)<1e-10
            assert v['embedding_coordinate']==float(source.decode(hs[0,0,k])) and v['hidden_coordinate']==float(source.decode(hs[36,0,k]))
            assert v['actual_Wgate_jk']==float(w[f'L{l}__J{j}_gate'][k]) and v['actual_probability']==float(p[q,h,s])
            assert len(got['source_trace'])==len(r['token_strings']) and len(got['head_coordinate_trace']['Q'])==128
            n+=1
        checks[r['dataset']+str(r['case_index'])+'_all5sites_native_values']=True
    checks['640queries']=n==640;checks['source_term_independent_FP64']=maxerror<1e-10
    for panel in cat['panels']:
        for start in sorted({0,len(panel['rows'])//2,len(panel['rows'])-1}):
            got=atlas.rows(panel['key'],start,8);spec=panel['rows'][start]
            with np.load(RESULT/spec['file']) as z:expected=z[spec['array']][tuple(spec['index'])]
            if spec.get('encoding')=='native_bf16':expected=source.decode(expected)
            checks[panel['key']+f'_row{start}']=np.array_equal(np.array(got['rows'][0]['values']),expected) and len(expected)==panel['coordinate_count']
    checks['34_legacy_panels_retained']=len([p for p in atlas.catalog()['panels'] if not p['key'].startswith('phase2684_')])==34
    for change in ({'coordinate':2560},{'head':32},{'head_coordinate':128},{'source_token':-1},{'hidden_token':-1},{'checkpoint':37},{'unit':0},{'dataset':'unpublished'}):
        r=data[0];args=dict(dataset=r['dataset'],case=r['case_index'],layer=23,unit=6197,coordinate=0,checkpoint=0,source_token=0,head=0,head_coordinate=0,query_position=1);args.update(change)
        try:source.query(**args);checks['reject_'+str(change)]=False
        except HTTPException as e:checks['reject_'+str(change)]=e.status_code in (400,404)
    if a.build:
        node='C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
        run=subprocess.run([node,'node_modules/vite/bin/vite.js','build'],cwd=ROOT/'frontend',capture_output=True,text=True,encoding='utf-8')
        checks['frontend_build']=run.returncode==0;save(OUT/'analysis/build_output.json',{'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr})
    result={'preview_only':a.preview,'checks':checks,'all_checks_passed':all(checks.values()),'queries':n,'max_independent_source_error':maxerror,'live_HTTP_or_browser':False}
    save(OUT/f'analysis/{prefix}delivery_checks.json',result);print({k:v for k,v in result.items() if k!='checks'});assert all(checks.values()),[k for k,v in checks.items() if not v]


if __name__=='__main__':main()
