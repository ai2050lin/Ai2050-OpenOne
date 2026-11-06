"""Read-only compact native banks. Committed rows only; uncaptured fields stay absent."""
import json,re,sqlite3
from contextlib import closing
from pathlib import Path
import numpy as np
from fastapi import APIRouter,HTTPException,Query
from shared.dist_protocol.mechanism_trace import unbits,file_sha,digest
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'tests/glm5/result'
router=APIRouter(prefix='/mlp-bank',tags=['whole-native-mlp-bank'])
_verified={}

def discover():
    out=[]
    paths=list(RESULTS.glob('mlp_atlas_code_v*/bank/manifest.json'))+list(RESULTS.glob('mlp_fidelity_v*/controlled-bank/manifest.json'))
    for p in sorted(paths):
        try:d=json.loads(p.read_text(encoding='utf-8'))
        except (OSError,ValueError):continue
        if d.get('schema')=='mlp_native_bank.v1' and d.get('kind')=='real':out.append((d,p.parent))
    return out

def bank(run):
    if not re.fullmatch(r'MLP-(?:ATLAS-CODE|FIDELITY)-\d{8}-v\d+',run):raise HTTPException(422,'非法运行ID')
    matches=[(d,p) for d,p in discover() if d['run_id']==run]
    if len(matches)!=1:raise HTTPException(404,'本节点没有唯一匹配的原生MLP库；不使用替代数据。')
    return matches[0]

def verify_index(d,p):
    if d['state']!='complete':return
    path=p/'samples.sqlite3';stamp=(path.stat().st_size,path.stat().st_mtime_ns)
    if _verified.get(str(path))!=stamp:
        if file_sha(path)!=d.get('sample_index_sha256'):raise HTTPException(409,'原始样本索引哈希不符')
        _verified[str(path)]=stamp

def sample(run,index):
    d,p=bank(run)
    if not 0<=index<d['n_items']:raise HTTPException(422,'样本索引越界')
    if index>=d['completed']:raise HTTPException(409,'该样本尚未提交采集，不显示预分配数组中的空行。')
    verify_index(d,p)
    with closing(sqlite3.connect(f'file:{(p/"samples.sqlite3").as_posix()}?mode=ro',uri=True)) as db:r=db.execute('SELECT body FROM samples WHERE idx=?',(index,)).fetchone()
    if not r:raise HTTPException(409,'该样本尚未提交采集')
    s=json.loads(r[0]);s['bank_sample_id']=digest({'run_id':run,'model_snapshot':d['model_snapshot'],'bank_contract':d['capture_contract_sha256'],'sample_id':s['id'],'index':index})
    return d,p,s

def array(d,p,name,index,stage):
    if name not in d['file_shapes']:raise HTTPException(404,'该原生字段未采集')
    path=p/name
    if d['state']=='complete':
        meta=next((x for x in d.get('files',[]) if x['path']==name),None)
        if not meta:raise HTTPException(409,'文件哈希核对尚未完成')
        stamp=(path.stat().st_size,path.stat().st_mtime_ns)
        if _verified.get(str(path))!=stamp:
            if stamp[0]!=meta['bytes'] or file_sha(path)!=meta['sha256']:raise HTTPException(409,'原生数据文件哈希不符')
            _verified[str(path)]=stamp
    a=np.load(path,mmap_mode='r')
    if list(a.shape)!=d['file_shapes'][name] or a.dtype!=np.uint16:raise HTTPException(409,'原生数组格式不符')
    return unbits(a[index,stage])

def native_field(d,p,index,block,stage,boundary):
    if boundary in ['embedding','hidden_state']:
        layer=0 if boundary=='embedding' else block
        if not 0<=layer<=d['n_layers']:raise HTTPException(422,'隐态层越界')
        return array(d,p,'hidden.npy',index,stage)[layer]
    if not 0<=block<d['n_layers']:raise HTTPException(422,'块越界')
    kinds={'mlp_gate':'G','mlp_value':'V','mlp_activation':'A','mlp_input':'input','mlp_output':'output','attention_output':'attention_output','head_z':'head_z'}
    if boundary not in kinds:raise HTTPException(404,'该字段未采集，不能以其他位置替代')
    value=array(d,p,f'B{block:02d}-{kinds[boundary]}.npy',index,stage)
    return value.reshape(d['n_heads'],d['head_dim']) if boundary=='head_z' else value

@router.get('/catalog')
def catalog():return {'schema':'mlp_native_bank_catalog.v1','banks':[{k:d[k] for k in ['run_id','model_snapshot','state','completed','n_items','n_layers','d_ff']} for d,p in discover()]}

@router.get('/{run}/manifest')
def manifest(run:str):return bank(run)[0]

@router.get('/{run}/samples')
def samples(run:str,offset:int=Query(0,ge=0),limit:int=Query(30,ge=1,le=200),split:str='',code:str=''):
    d,p=bank(run);verify_index(d,p);where=['idx<?'];args=[d['completed']]
    for key,value in [('split',split),('code',code)]:
        if value:where.append(f"json_extract(body,'$.{key}')=?");args.append(value)
    with closing(sqlite3.connect(p/'samples.sqlite3')) as db:
        total=db.execute('SELECT COUNT(*) FROM samples WHERE '+' AND '.join(where),args).fetchone()[0]
        rows=db.execute('SELECT idx,body FROM samples WHERE '+' AND '.join(where)+' ORDER BY idx LIMIT ? OFFSET ?',args+[limit,offset]).fetchall()
    keys=['id','bank_index','split','code','mapping','fact','question','decision','expected','correct','choice_gap','input_tokens']
    return {'total':total,'completed':d['completed'],'n_items':d['n_items'],'samples':[{k:s[k] for k in keys} for i,r in rows for s in [json.loads(r)]]}

@router.get('/{run}/samples/{index}')
def sample_manifest(run:str,index:int):
    d,p,s=sample(run,index);return {**s,'dimensions':{k:d[k] for k in ['n_layers','d_ff','d_model','n_heads','n_kv_heads','head_dim']},'analysis_readouts':s['readout_positions'][2:],
        'capture_state':d['state'],'raw_format':'mlp_native_bank.v1','attention_rows_captured':False,'source_contract_sha256':d['capture_contract_sha256']}

@router.get('/{run}/samples/{index}/field')
def field(run:str,index:int,block:int=Query(0,ge=0),stage:int=Query(0,ge=0,le=1),boundary:str='mlp_activation'):
    d,p,s=sample(run,index);v=native_field(d,p,index,block,stage,boundary)
    return {'schema':'mechanism_field.v1','kind':'real','model_snapshot':d['model_snapshot'],'run_id':run,'bank_sample_id':s['bank_sample_id'],'block':block,'stage':stage,'position':s['readout_positions'][stage+2],
        'boundary':boundary,'shape':list(v.shape),'values':v.reshape(-1).tolist(),'source':'native_BF16_bits','capture_state':d['state'],'source_contract_sha256':d['capture_contract_sha256']}

@router.get('/{run}/samples/{index}/connections')
def connections(run:str,index:int,block:int=Query(0,ge=0),stage:int=Query(0,ge=0,le=1),head:int=Query(0,ge=0),unit:int=Query(0,ge=0),coordinate:int=Query(0,ge=0)):
    d,p,s=sample(run,index);values={k:native_field(d,p,index,block,stage,b) for k,b in [('gate','mlp_gate'),('value','mlp_value'),('activation','mlp_activation'),('mlp_input','mlp_input'),('head_z','head_z'),('attention_output','attention_output'),('mlp_output','mlp_output')]}
    values['resid_in']=native_field(d,p,index,block,stage,'hidden_state')
    native={**d,'trace_id':s['bank_sample_id'],'readout_positions':s['readout_positions'][2:],'deep':False};pos=native['readout_positions'][stage]
    from server.mechanism_trace_service import contributions,checked
    out=checked(contributions,s['bank_sample_id'],block,pos,head,unit,coordinate,native=(native,values))
    return {**out,'source_format':'mlp_native_bank.v1','bank_index':index,'capture_state':d['state'],'attention_rows_captured':False}

def analysis_data(run,block):
    d,p=bank(run)
    if not 0<=block<d['n_layers']:raise HTTPException(422,'块越界')
    path=p.parent/'analysis-result.json'
    if not path.exists():raise HTTPException(409,'全部预注册分析尚未完成')
    report=json.loads(path.read_text(encoding='utf-8'));meta=report['layers'][block];path=p.parent/'analysis'/f'B{block:02d}.npz'
    stamp=(path.stat().st_size,path.stat().st_mtime_ns)
    if _verified.get(str(path))!=stamp:
        if file_sha(path)!=meta['sha256']:raise HTTPException(409,'分析数组哈希不符')
        _verified[str(path)]=stamp
    return report,path

@router.get('/{run}/analysis/layers/{block}')
def analysis_layer(run:str,block:int):
    report,p=analysis_data(run,block)
    with np.load(p) as z:data={k:z[k].tolist() for k in ['response_replicated','eligible_supported','shuffled_replicated','confirm_fvu','lockbox_fvu']}
    return {'schema':'mlp_atlas_layer_analysis.v1','model_snapshot':report['model_snapshot'],'block':block,'d_ff':9728,'behavior_gate':report['behavior_gate'],'readout_roles':report['readout_roles'],'source_sha256':report['layers'][block]['sha256'],**data}

@router.get('/{run}/analysis/units/{block}/{unit}')
def analysis_unit(run:str,block:int,unit:int):
    if not 0<=unit<9728:raise HTTPException(422,'单元越界')
    report,p=analysis_data(run,block)
    with np.load(p) as z:data={k:(z[k].tolist() if k=='factorial_names' else z[k][...,unit].tolist()) for k in z.files}
    return {'schema':'mlp_atlas_unit_analysis.v1','model_snapshot':report['model_snapshot'],'block':block,'unit':unit,'behavior_gate':report['behavior_gate'],'scope':report['scope'],'source_sha256':report['layers'][block]['sha256'],**data}
