"""Bounded-memory metadata queries; immutable native manifests stay untouched."""
import json

KEYS=['trace_id','run_id','sample_id','entity','scope','split','n_tokens','model_snapshot','deep','readout_positions','n_layers','d_model','d_ff','n_heads','n_kv_heads','head_dim','generation_step']

def refresh(db):
    db.execute('CREATE TABLE IF NOT EXISTS trace_catalog_cache(id TEXT PRIMARY KEY,run TEXT,model TEXT,sample TEXT,parent TEXT,created REAL,summary TEXT)')
    db.execute('CREATE INDEX IF NOT EXISTS trace_catalog_latest ON trace_catalog_cache(run,model,sample,created DESC,id DESC)')
    db.execute('CREATE INDEX IF NOT EXISTS trace_catalog_parent ON trace_catalog_cache(parent)')
    args=','.join("'"+k+"',json_extract(t.body,'$."+k+"')" for k in KEYS)
    db.execute("INSERT OR IGNORE INTO trace_catalog_cache SELECT t.id,json_extract(t.body,'$.run_id'),json_extract(t.body,'$.model_snapshot'),json_extract(t.body,'$.sample_id'),json_extract(t.body,'$.parent_trace_id'),t.created,json_object("+args+") FROM traces t WHERE NOT EXISTS(SELECT 1 FROM trace_catalog_cache c WHERE c.id=t.id)")

def catalog(store,run_id='',parent=''):
    with store.db() as db:
        refresh(db)
        if parent:
            rows=db.execute('SELECT summary FROM trace_catalog_cache WHERE parent=? ORDER BY created,id',(parent,))
        else:
            where='WHERE run=?' if run_id else ''
            rows=db.execute('SELECT summary FROM (SELECT summary,sample,created,id,ROW_NUMBER() OVER(PARTITION BY run,model,sample ORDER BY created DESC,id DESC) rn FROM trace_catalog_cache '+where+') WHERE rn=1 ORDER BY created DESC,id DESC',(run_id,) if run_id else ())
        traces=[]
        for r in rows:
            d=json.loads(r['summary'])
            # sqlite json_object embeds json arrays as arrays, never their string form.
            if isinstance(d['readout_positions'],str):d['readout_positions']=json.loads(d['readout_positions'])
            if parent or ':G' not in d['sample_id']:traces.append(d)
    return {'schema':'mechanism_trace_catalog.v1','traces':traces}

def dimensions(store):
    with store.db() as db:
        refresh(db)
        rows=db.execute('SELECT summary FROM (SELECT summary,ROW_NUMBER() OVER(PARTITION BY model ORDER BY created DESC,id DESC) rn FROM trace_catalog_cache) WHERE rn=1')
        return [json.loads(r['summary']) for r in rows]
