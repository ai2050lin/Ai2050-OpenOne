"""All archived studies in a typed 3D index; no inferred mechanism edges."""
import json
from pathlib import Path
from fastapi import APIRouter,HTTPException
router=APIRouter()

def library():
    from deploy.research_library import library as get
    return get()

def bindings(run_id=None):
    lib=library()
    with lib.db() as db:
        rows=db.execute("SELECT sha FROM artifacts WHERE kind='space_binding' ORDER BY created DESC,sha DESC").fetchall()
    result=[];seen=set()
    for row in rows:
        d=lib.artifact(row['sha'])
        key=(d['run_id'],d['model_snapshot'])
        if (run_id is None or d['run_id']==run_id) and key not in seen:result.append({**d,'binding_sha256':row['sha']});seen.add(key)
    return result

def binding(run_id,model_snapshot=None):
    matches=[b for b in bindings(run_id) if model_snapshot is None or b['model_snapshot']==model_snapshot]
    if len(matches)>1:raise ValueError('该运行有多个模型快照，请指定完整model_snapshot。')
    return matches[0] if matches else None

@router.get('/research-space')
def research_space():
    lib=library()
    with lib.db() as db:
        features=db.execute('''SELECT * FROM (SELECT g.*,a.created,ROW_NUMBER() OVER(PARTITION BY g.model,g.id,g.type ORDER BY a.created DESC,g.sha DESC) AS rn
          FROM graph_objects g JOIN artifacts a ON a.sha=g.sha) WHERE rn=1 ORDER BY model,id''').fetchall()
        artifacts=[dict(r) for r in db.execute('SELECT sha,source,model,kind,created,bytes FROM artifacts ORDER BY created DESC,sha DESC')]
    studies=[];all_bindings=bindings()
    for w in lib.workpackages()['workpackages']:
        d=w['body'];bb=[b for b in all_bindings if b['run_id']==d['run_id']];report=d.get('stages',{}).get('registration_validated',{}).get('report_sha256')
        for b in bb or [None]:
          studies.append({'run_id':d['run_id'],'study_key':d['run_id']+'|'+(b['model_snapshot'] if b else ''),'title':d.get('title',d['run_id']),'workpackage_sha256':w['sha'],'report_sha256':b['report_sha256'] if b else report,
              'stages':d['stages'],'model_snapshot':b.get('model_snapshot') if b else None,'binding_sha256':b.get('binding_sha256') if b else None,
              'traces':b.get('traces',[]) if b else [],'localization':'native_binding' if b and b.get('traces') else 'native_bank' if b and b.get('context',{}).get('native_bank') else 'native_aggregate' if b and b.get('context',{}).get('study_type')=='mlp_reverse_inventory' else 'metadata_only'})
    from shared.dist_protocol.mechanism_trace import TraceStore
    models={}
    trace_index=lib.root.parent/'mechanism_traces';traces=[]
    if (trace_index/'index.sqlite3').is_file():
      from server.trace_catalog import dimensions
      traces=dimensions(TraceStore(index=trace_index))
    for t in traces:models[t['model_snapshot']]={**{k:t[k] for k in ['n_layers','d_ff','d_model','n_heads']},'snapshot':t['model_snapshot'],'reference_trace_id':t['trace_id']}
    for b in all_bindings:
        snapshot=b['model_snapshot']
        dims=b.get('context',{}).get('model_dimensions',{})
        limits={'n_layers':256,'d_ff':262144,'d_model':65536,'n_heads':512}
        if snapshot not in models and isinstance(dims,dict) and all(type(dims.get(k)) is int and 0<dims[k]<=v for k,v in limits.items()):
            models[snapshot]={'snapshot':snapshot,**{k:dims[k] for k in limits},'dimension_binding_sha256':b['binding_sha256']}
    objects=[{'sha':r['sha'],'id':r['id'],'type':r['type'],'model_snapshot':r['model'],'body':json.loads(r['body'])} for r in features]
    for o in objects:
        if o['model_snapshot'] not in models:models[o['model_snapshot']]={'snapshot':o['model_snapshot'],'localization':'no_native_dimensions'}
    return {'schema':'research_space_index.v1','kind':'real','fingerprint':lib.fingerprint(),'studies':studies,'objects':objects,'artifacts':artifacts,'models':list(models.values()),
        'counts':{'studies':len(studies),'objects':len(objects),'artifacts':len(artifacts)},
        'interpretation':'Native indices locate observations. Unlocated reports remain on an explicitly schematic artifact rack. Layout distance is not model geometry; no evidence promotion.'}
