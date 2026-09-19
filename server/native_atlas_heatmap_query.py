"""Small exact full-coordinate pages from lossless published panels, no model load."""
import json
from functools import lru_cache
import numpy as np
from fastapi import HTTPException
from .native_path_parameter_query import RESULT
OUT=RESULT/'phase2669_symmetric_multitoken_delivery'
NEW=RESULT/'phase2676_native_mlp_delivery'
LATEST=RESULT/'phase2684_source_campaign_delivery'
QKV=RESULT/'phase2693_qkv_campaign_delivery'


@lru_cache(maxsize=1)
def catalog():
    try:
        obj=json.loads((OUT/'material/client_panel_catalog.json').read_text(encoding='utf-8'))
        path=NEW/'material/client_panel_catalog.json'
        if path.exists():
            added=json.loads(path.read_text(encoding='utf-8'));obj['panels']+=added['panels'];obj['phase']=added['phase'];obj['boundary']+=' '+added['boundary']
        path=LATEST/'material/client_panel_catalog.json'
        if path.exists():
            added=json.loads(path.read_text(encoding='utf-8'));obj['panels']+=added['panels'];obj['phase']=added['phase'];obj['boundary']+=' '+added['boundary']
        path=QKV/'material/client_panel_catalog.json'
        if path.exists():
            added=json.loads(path.read_text(encoding='utf-8'))
            if not added.get('preview_only',True):obj['panels']+=added['panels'];obj['phase']=added['phase'];obj['boundary']+=' '+added['boundary']
        return obj
    except FileNotFoundError as e:raise HTTPException(status_code=404,detail='Full-coordinate panel publication not completed yet') from e


def options(include_rows=True):
    obj=catalog();panels=[]
    for p in obj['panels']:
        info={k:v for k,v in p.items() if k not in ('matrix_sha256','rows','blocks')}
        info['row_count']=p['row_count'] if p.get('storage')=='native_block_descriptor' else len(p['rows'])
        if include_rows and p.get('storage')!='native_block_descriptor':
            info['rows']=[{k:v for k,v in r.items() if k in ('label','source')} for r in p['rows']]
        panels.append(info)
    return {'phase':obj['phase'],'boundary':obj['boundary'],'display':obj['display'],'panels':panels}


@lru_cache(maxsize=4)
def descriptor_array(relative,key):
    path=(RESULT/relative).resolve()
    if not path.is_relative_to(RESULT.resolve()) or path.suffix!='.npz':raise HTTPException(400,'Invalid published artifact')
    with np.load(path,allow_pickle=False) as z:value=z[key]
    value.flags.writeable=False;return value


def descriptor_row(row):
    a=descriptor_array(row['file'],row['array'])[tuple(row['index'])]
    if row.get('encoding')=='native_bf16':a=(a.astype(np.uint32)<<16).view(np.float32)
    assert a.ndim==1 and np.isfinite(a).all()
    return a


@lru_cache(maxsize=2)
def matrix(key):
    if key not in {p['key'] for p in catalog()['panels']}:raise HTTPException(status_code=404,detail='Unknown published panel')
    source=NEW if key.startswith('phase2676_') else OUT
    with np.load(source/'maps/client_panels'/(key+'.npz'),allow_pickle=False) as z:value=z['values']
    value.flags.writeable=False;return value


def rows(panel,start,count):
    info=next((p for p in catalog()['panels'] if p['key']==panel),None)
    if info is None:raise HTTPException(status_code=404,detail='Unknown published panel')
    total=info['row_count'] if info.get('storage')=='native_block_descriptor' else len(info['rows'])
    if start<0 or start>=total or count<1 or count>8:raise HTTPException(status_code=400,detail='Rows must start within this panel; count1..8')
    if info.get('storage')=='native_block_descriptor':
        rr=[];offset=0;end=min(start+count,total)
        for block in info['blocks']:
            stop=offset+block['row_count']
            for i in range(max(start,offset),min(end,stop)):
                prefix=tuple(block['shape'][:-1]);idx=tuple(int(v) for v in np.unravel_index(i-offset,prefix)) if prefix else ()
                row={**block,'index':idx};value=descriptor_row(row)
                assert len(value)==info['coordinate_count']
                specific=block['row_labels'][i-offset] if 'row_labels' in block else ''
                rr.append({'row_index':i,'label':block['label']+'/index'+str(idx)+('/'+specific if specific else ''),'values':value.tolist()})
            offset=stop
            if offset>=end:break
        assert len(rr)==end-start
        return {'phase':info.get('phase',2693),'key':panel,'title':info['title'],'coordinate_count':info['coordinate_count'],
            'total_rows':total,'start':start,'rows':rr,'display':info['boundary']+' Exact full native last axis. Prefix indices in row labels; rows paged only.'}
    if info.get('storage')=='native_descriptor':
        end=min(start+count,len(info['rows']))
        rr=[{'label':info['rows'][i]['label'],'row_index':i,'values':descriptor_row(info['rows'][i]).tolist()} for i in range(start,end)]
        assert all(len(r['values'])==info['coordinate_count'] for r in rr)
        return {'phase':2684,'key':panel,'title':info['title'],'coordinate_count':info['coordinate_count'],'total_rows':len(info['rows']),
                'start':start,'rows':rr,'display':info['boundary']+' All physical columns, exact native-data views. Only rows are paged; no TopK or average bins.'}
    a=matrix(panel);end=min(start+count,len(a))
    return {'phase':2676 if panel.startswith('phase2676_') else 2669,'key':panel,'title':info['title'],'coordinate_count':info['coordinate_count'],'total_rows':len(a),'start':start,
        'rows':[{**info['rows'][i],'row_index':i,'values':a[i].tolist()} for i in range(start,end)],
        'display':'All original physical columns included, noTopK or averaged bins. Rows are paged only for display. Use scalar query to distinguish a real weight from a derivative or activation.'}
