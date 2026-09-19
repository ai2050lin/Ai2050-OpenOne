"""Actual CPU regression tests of staged lazy native panels and scalar queries."""
import sys,hashlib,json,math,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
import numpy as np
from fastapi import HTTPException
from phase2620_native_coordinate_contract import read,save,sha,RESULT
from server import native_atlas_heatmap_query as atlas
from server import native_qkv_parameter_query as qkv

OUT=RESULT/'phase2693_qkv_campaign_delivery'

def decode(x):return (x.astype(np.uint32)<<16).view(np.float32) if x.dtype==np.uint16 else x

def expected_row(panel,row):
    for b in panel['blocks']:
        if row>=b['row_count']:row-=b['row_count'];continue
        idx=[]
        for size in reversed(b['shape'][:-1]):row,remainder=divmod(row,size);idx.append(remainder)
        assert row==0;idx=tuple(reversed(idx))
        with np.load(RESULT/b['file'],allow_pickle=False) as z:a=z[b['array']][idx]
        if b.get('encoding')=='native_bf16':a=decode(a)
        return np.asarray(a)
    raise AssertionError('Missing row')

def evidence(published):
    path=OUT/('material/client_panel_catalog.json' if published else 'material/staged_client_panel_catalog.json')
    return {'catalog_sha256':sha(path),'source_sha256':{str(p.relative_to(RESULT.parents[2])):sha(p) for p in (
        Path(atlas.__file__),Path(qkv.__file__))}}

def panel_test(published=False):
    staged=read(OUT/('material/client_panel_catalog.json' if published else 'material/staged_client_panel_catalog.json'))
    assert staged['preview_only'] is (not published)
    legacy=atlas.catalog();old=[p for p in legacy['panels'] if not p['key'].startswith('phase2693_')]
    assert len(old)==75
    if published:
        assert [p for p in legacy['panels'] if p['key'].startswith('phase2693_')]==staged['panels']
        merged=legacy
    else:
        assert len(legacy['panels'])==75
        merged={**legacy,'panels':old+staged['panels']}
        atlas.catalog=lambda:merged
    compact=atlas.options(include_rows=False)
    assert len(compact['panels'])==75+len(staged['panels']) and all('blocks' not in r and 'rows' not in r for r in compact['panels'])
    reports=[]
    for p in staged['panels']:
        assert p['row_count']==sum(math.prod(b['shape'][:-1]) for b in p['blocks'])
        indices={0,p['row_count']-1}
        if len(p['blocks'])>1:indices.add(max(0,p['blocks'][0]['row_count']-2))
        records=[]
        for start in sorted(indices):
            actual=atlas.rows(p['key'],start,8)
            assert actual['phase']==2693 and actual['coordinate_count']==p['coordinate_count']
            for r in actual['rows']:
                expected=expected_row(p,r['row_index']);assert len(r['values'])==len(expected)==p['coordinate_count']
                assert np.array_equal(r['values'],expected) and np.isfinite(expected).all()
            records.append({'start':start,'count':len(actual['rows']),'last_row':actual['rows'][-1]['row_index'],
                'last_coordinate':p['coordinate_count']-1,'last_value':actual['rows'][-1]['values'][-1]})
        reports.append({'key':p['key'],'logical_rows':p['row_count'],'columns':p['coordinate_count'],'actual_checks':records})
    for start,count in ((-1,8),(p['row_count'],8),(0,9),(0,0)):
        try:atlas.rows(p['key'],start,count)
        except HTTPException as e:assert e.status_code==400
        else:raise AssertionError('Bad request accepted')
    atlas.descriptor_array.cache_clear()
    prefix='' if published else 'staged_'
    save(OUT/f'analysis/{prefix}panel_direct_audit.json',{'all_checks_passed':True,'preview_only':not published,'legacy_types_preserved':75,
        'new_types':len(reports),'full_column_rows_and_block_transitions_checked':reports,'code_sha256':sha(Path(__file__)),
        'not_live_HTTP_or_browser':True,**evidence(published)})
    print('2693 ACTUAL LAZY ROWS PASS',len(reports),'plus75legacy',flush=True)

def parameter_test(published=False):
    meta=qkv.options();controls=meta['controls'];results=[]
    weights=RESULT/'phase2685_native_attention_contract/weights/native_qkv_windows.npz'
    with np.load(weights) as z:W={c['vector']:decode(z[c['vector']]).astype(np.float64) for c in controls}
    for ci,r in enumerate(meta['cases']):
        case=r['case'];n=r['tokens'];sourcepath=RESULT/f'phase2687_role_qkv_field/source/case_{case:04d}.npz'
        with np.load(sourcepath) as raw:
            for i,c in enumerate(controls):
                l=c['layer'];k=c['input_coordinate'];kind=c['kind'];outrow=c['output_row'];token=n-1 if i%2 else 0
                j=(ci*607+i*157)%9728;out=2559 if i%2 else 0;head=31 if i%2 else 0;d=127 if i%2 else 0
                data=qkv.query(case,l,kind,outrow,k,token,i%2,n-1,head,d,36,j,out)
                x=decode(raw[f'L{l}__upstream_attention_x'][token]).astype(np.float64)
                expected=W[c['vector']]*x;vals=data['values']
                assert vals['actual_W_kind_r_k']==c['original_weight']
                assert np.array_equal(data['traces']['projection_input']['Wx'],expected)
                assert np.array_equal(data['traces']['projection_input']['W'],W[c['vector']])
                assert len(data['scalar_effects'])==len(data['changed_weight_natural'])==4 and not data['percase_changed_P_available']
                assert len(data['traces']['all_source_tokens'])==n and len(data['traces']['headnorm']['linear'])==128
                p=decode(raw[f'L{l}__actual_probability'][i%2,head]).astype(np.float64)
                assert [t['P'] for t in data['traces']['all_source_tokens']]==p.tolist()
                for e in data['scalar_effects']:assert e['input_coordinate']==k and e['output_row']==outrow and e['layer']==l and e['projection_kind']==kind
                results.append({'case':case,'control_index':i,'token':token,'MLPunit':j,'outcoord':out,'actual_weight':vals['actual_W_kind_r_k'],
                    'last_product':data['traces']['projection_input']['Wx'][-1],'all2560products_exact':True})
        print('2693 ACTUAL PARAMETER CASE',ci+1,16,flush=True)
    for kwargs in ({'case':-1},{'input_coordinate':2560},{'output_row':4096},{'unit':9728},{'layer':34},{'kind':'invalid'},{'head_coordinate':128}):
        try:qkv.query(**kwargs)
        except HTTPException as e:assert e.status_code in (400,404)
        else:raise AssertionError('Bad physical address accepted')
    prefix='' if published else 'staged_'
    save(OUT/f'analysis/{prefix}parameter_direct_audit.json',{'all_checks_passed':True,'preview_only':not published,'cases':16,'actual_queries':len(results),
        'all48controls_all16cases':results,'code_sha256':sha(Path(__file__)),'not_live_HTTP_or_browser':True,
        **evidence(published),
        'boundary':'Independent frozen QKV weight vectors and rawx/P reference. MLP parameters addressed but require additional independent checkpoint-byte SHA audit before final delivery.'})
    assert len(results)==768

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--published',action='store_true');args=parser.parse_args()
    panel_test(args.published);parameter_test(args.published)
