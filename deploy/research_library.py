"""Durable, content-addressed research index. Execution != mechanism evidence.

The existing center result store remains authoritative for uploads/audits. This
index retains revisions of local reports and explicit feature/edge declarations;
it never infers a scientific graph from filenames or model-generated prose.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Query

router = APIRouter(prefix='/api/research-library', tags=['research-library'])
_lock = threading.RLock()


def canonical(doc):
    return json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def synthetic(doc):
    return str(doc.get('kind','')).lower()=='synthetic' or str(doc.get('mode','')).lower()=='synthetic' or doc.get('synthetic') is True or doc.get('sandbox_synthetic') is True


class Library:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS artifacts(sha TEXT PRIMARY KEY,source TEXT NOT NULL,
                    model TEXT NOT NULL,created REAL NOT NULL,kind TEXT NOT NULL,bytes INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS observations(sha TEXT NOT NULL,source TEXT NOT NULL,
                    PRIMARY KEY(sha,source));
                CREATE TABLE IF NOT EXISTS graph_objects(sha TEXT NOT NULL,id TEXT NOT NULL,
                    type TEXT NOT NULL,model TEXT NOT NULL,body TEXT NOT NULL,PRIMARY KEY(sha,id,type));
                CREATE INDEX IF NOT EXISTS objects_model ON graph_objects(model,type);
                CREATE TABLE IF NOT EXISTS file_cache(path TEXT PRIMARY KEY,stamp TEXT NOT NULL,sha TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS workpackage_revisions(sha TEXT PRIMARY KEY,run_id TEXT NOT NULL);
            ''')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.root / 'library.sqlite3', timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:yield db
        finally:db.close()

    def ingest(self, doc, source, *, kind='report'):
        with _lock:return self._ingest(doc,source,kind=kind)

    def _ingest(self, doc, source, *, kind='report'):
        if not isinstance(doc, dict) or synthetic(doc):
            raise ValueError('Only non-synthetic structured research artifacts can be indexed')
        raw = canonical(doc)
        if len(raw)>16*1024*1024:
            raise ValueError('Research metadata must not exceed 16MB')
        sha = hashlib.sha256(raw).hexdigest()
        if doc.get('schema')=='research_space_binding.v1':
            if doc.get('kind')!='real' or not isinstance(doc.get('run_id'),str) or not doc['run_id'].strip():raise ValueError('real named spatial binding required')
            snapshot=doc.get('model_snapshot','')
            if len(snapshot)!=64 or any(c not in '0123456789abcdef' for c in snapshot):raise ValueError('full binding model snapshot required')
            for key in ['report_sha256','graph_sha256']:
                linked=self.artifact(doc.get(key,''))
                if linked.get('model_snapshot')!=snapshot:raise ValueError('binding proof snapshot differs')
                if key=='graph_sha256' and linked.get('schema')!='research_graph.v1':raise ValueError('binding graph must be an explicit research_graph.v1')
            if not isinstance(doc.get('traces',[]),list) or not isinstance(doc.get('context',{}),dict):raise ValueError('binding trace/context collections required')
            for t in doc.get('traces',[]):
                identity=t.get('trace_id','') if isinstance(t,dict) else ''
                if len(identity)!=64 or any(c not in '0123456789abcdef' for c in identity):raise ValueError('invalid binding trace identity')
                if t.get('manifest_sha256'):
                    manifest=self.artifact(t['manifest_sha256'])
                    if manifest.get('trace_id')!=identity or manifest.get('model_snapshot')!=snapshot:raise ValueError('binding trace manifest differs')
            kind='space_binding'
        identity = doc.get('model_snapshot') or doc.get('model_rev') or doc.get('model') or doc.get('model_id') or ''
        workpackage=doc.get('schema')=='research_workpackage.v1'
        if workpackage and (not isinstance(doc.get('run_id'),str) or not doc['run_id'].strip() or not isinstance(doc.get('stages'),dict)):
            raise ValueError('Workpackage requires run_id and stages')
        model = json.dumps(identity, sort_keys=True, ensure_ascii=False) if isinstance(identity, dict) else str(identity)
        cards = doc.get('feature_cards', [])
        edges = doc.get('graph_edges', [])
        if not isinstance(cards, list) or not isinstance(edges, list) or len(cards)+len(edges)>10000:
            raise ValueError('feature_cards / graph_edges must be bounded lists')
        if cards or edges:
            if doc.get('schema') != 'research_graph.v1' or doc.get('kind') != 'real':
                raise ValueError('Graph declarations require research_graph.v1 and kind=real')
            snapshot = doc.get('model_snapshot')
            if not isinstance(snapshot, str) or len(snapshot)!=64 or any(c not in '0123456789abcdef' for c in snapshot):
                raise ValueError('Graph declarations require an exact model_snapshot sha256')
            ids=set()
            for card in cards:
                if not isinstance(card, dict) or not all(card.get(k) is not None for k in ('id','layer','boundary','index','method','run_id','evidence','counterexamples')):
                    raise ValueError('Feature card lacks identity, boundary, method, run, evidence or counterexamples')
                if not isinstance(card['id'], str) or not card['id'] or card['id'] in ids:
                    raise ValueError('Feature ids must be unique nonempty strings')
                if type(card['layer']) is not int or card['layer']<0 or not isinstance(card['counterexamples'],list):
                    raise ValueError('Feature layer must be a nonnegative integer; counterexamples must be a list')
                if not all(isinstance(card[k],str) and card[k].strip() for k in ('boundary','method','run_id')):
                    raise ValueError('Feature boundary, method and run must be nonempty strings')
                if not isinstance(card['index'],dict) or card['index'].get('space') not in ('residual','mlp','attention','dictionary','parameter'):
                    raise ValueError('Feature index requires an explicit native space')
                ids.add(card['id'])
            card_ids=set(ids)
            for edge in edges:
                if not isinstance(edge,dict) or not all(edge.get(k) for k in ('id','source','target','relation','run_id','evidence')):
                    raise ValueError('Graph edge lacks source, target, relation, run or evidence')
                if edge['target'] not in card_ids or (edge['source'] not in card_ids and not str(edge['source']).startswith('language:')):
                    raise ValueError('Graph endpoints must resolve within this revision or to a typed language node')
                if edge['id'] in ids:
                    raise ValueError('Graph ids must be unique')
                ids.add(edge['id'])
            allowed={'hypothesis','observation','heldout','intervention','parameter_source','negative'}
            for obj in cards+edges:
                if not isinstance(obj['evidence'],dict) or obj['evidence'].get('level') not in allowed or not obj['evidence'].get('artifact_sha256'):
                    raise ValueError('Evidence requires an explicit level and an artifact sha256')
                ref=obj['evidence']['artifact_sha256']
                if not isinstance(ref,str) or len(ref)!=64 or any(c not in '0123456789abcdef' for c in ref):
                    raise ValueError('Invalid evidence artifact sha256')
                with self.db() as db:
                    if not db.execute('SELECT 1 FROM artifacts WHERE sha=?',(ref,)).fetchone():
                        raise ValueError('Evidence artifact must be archived before graph registration')
        dest=self.root/'objects'/sha[:2]/(sha+'.json')
        with self.db() as db:
            if db.execute('SELECT 1 FROM artifacts WHERE sha=?',(sha,)).fetchone():
                if not dest.is_file() or hashlib.sha256(dest.read_bytes()).hexdigest()!=sha:raise ValueError('Archived object missing or corrupted')
                db.execute('INSERT OR IGNORE INTO observations VALUES(?,?)',(sha,str(source)))
                if workpackage:db.execute('INSERT OR IGNORE INTO workpackage_revisions VALUES(?,?)',(sha,doc['run_id']))
                return {'sha256':sha,'features':len(cards),'edges':len(edges)}
        dest.parent.mkdir(parents=True,exist_ok=True)
        # Publish a complete blob atomically without overwriting an existing revision.
        fd,tmp=tempfile.mkstemp(dir=dest.parent,suffix='.tmp')
        try:
            with os.fdopen(fd,'wb') as f:
                f.write(raw);f.flush();os.fsync(f.fileno())
            if dest.exists():
                if dest.read_bytes()!=raw:raise ValueError('Content addressed object mismatch')
            else:
                # Concurrent publishers of this sha have identical canonical bytes.
                try:os.replace(tmp,dest)
                except PermissionError:
                    if not dest.is_file() or dest.read_bytes()!=raw:raise
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
        with self.db() as db:
            db.execute('INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?,?,?)',(sha,str(source),model,time.time(),kind,len(raw)))
            db.execute('INSERT OR IGNORE INTO observations VALUES(?,?)',(sha,str(source)))
            if workpackage:db.execute('INSERT OR IGNORE INTO workpackage_revisions VALUES(?,?)',(sha,doc['run_id']))
            for typ,objects in [('feature',cards),('edge',edges)]:
                for obj in objects:
                    db.execute('INSERT OR IGNORE INTO graph_objects VALUES(?,?,?,?,?)',(sha,obj['id'],typ,model,json.dumps(obj,ensure_ascii=False)))
        return {'sha256':sha,'features':len(cards),'edges':len(edges)}

    def scan(self, root, *, extra_paths=(), source_prefix='', kind='local_report'):
        """New/modified report revisions are archived; missing source files never delete evidence."""
        count=0; errors=[]
        root=Path(root)
        paths=set(root.rglob('report.json')) | set(root.rglob('*.research.json'))
        paths.update(p for p in extra_paths if p.is_file())
        for name in ('neurons','interventions'):
            paths.update((root/name).rglob('*.json'))
        for path in sorted(paths):
            if any('synthetic' in part.lower() for part in path.parts): continue
            st=path.stat()
            if st.st_size>16*1024*1024: continue
            stamp=f'{st.st_mtime_ns}:{st.st_size}'
            with self.db() as db: row=db.execute('SELECT stamp FROM file_cache WHERE path=?',(str(path.resolve()),)).fetchone()
            if row and row['stamp']==stamp:continue
            try:
                raw=path.read_bytes();nonfinite=[]
                def mark(value):
                    nonfinite.append(value)
                    return {'$nonfinite':value}
                doc=json.loads(raw.decode('utf-8-sig'),parse_constant=mark)
                if isinstance(doc,dict) and synthetic(doc):continue
                if nonfinite:
                    if not isinstance(doc,dict) or doc.get('schema')=='research_graph.v1':raise ValueError('Graph declarations require finite JSON values')
                    # Preserve legacy failed/undefined statistics, including exact original UTF-8 bytes.
                    doc={'schema':'legacy_report_import.v1','kind':'legacy_report',
                         'model_snapshot':doc.get('model_snapshot') or doc.get('model_rev') or doc.get('model') or doc.get('model_id'),
                         'source_sha256':hashlib.sha256(raw).hexdigest(),'source_text':raw.decode('utf-8'),
                         'normalization':{'nonfinite_literals':sorted(set(nonfinite)),'note':'$nonfinite marks undefined statistics, never zero or valid evidence'},'report':doc}
                result=self.ingest(doc,source_prefix+str(path.relative_to(root)),kind=kind)
                with self.db() as db:db.execute('INSERT OR REPLACE INTO file_cache VALUES(?,?,?)',(str(path.resolve()),stamp,result['sha256']))
                count+=1
            except (ValueError,OSError,TypeError,AttributeError) as exc:errors.append({'source':str(path.relative_to(root)),'error':str(exc)[:240]})
        return {'new_revisions':count,'errors':errors}

    def summary(self, model=''):
        where=' WHERE model=?' if model else '';args=(model,) if model else ()
        with self.db() as db:
            models=[dict(r) for r in db.execute('SELECT model,COUNT(*) AS artifacts FROM artifacts GROUP BY model')]
            recent=[dict(r) for r in db.execute('SELECT * FROM artifacts'+where+' ORDER BY created DESC LIMIT 100',args)]
            counts={t:db.execute('SELECT COUNT(*) FROM graph_objects WHERE type=?'+(' AND model=?' if model else ''),(t,)+args).fetchone()[0] for t in ('feature','edge')}
            total=db.execute('SELECT COUNT(*) FROM artifacts'+where,args).fetchone()[0]
            status=db.execute("SELECT value FROM metadata WHERE key='sync'").fetchone()
        return {'schema':'research_library.v1','artifacts':total,'graph_revisions':counts,'models':models,'recent':recent,
                'sync':json.loads(status[0]) if status else None,
                'note':'产物归档不表示证据晋级；特征及边必须显式登记并引用已归档证据。'}

    def fingerprint(self):
        with self.db() as db:
            rows=list(db.execute('SELECT sha FROM artifacts ORDER BY sha'))
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='blind_labels'").fetchone():
                rows+=list(db.execute('SELECT sha FROM blind_labels ORDER BY sha'))
        return hashlib.sha256(''.join(r[0] for r in rows).encode()).hexdigest()[:20]

    def graph(self, model='', offset=0, limit=100):
        with self.db() as db:
            rows=db.execute('SELECT * FROM graph_objects'+(' WHERE model=?' if model else '')+' ORDER BY sha,id LIMIT ? OFFSET ?',((model,) if model else ())+(limit,offset)).fetchall()
        return {'objects':[{**dict(r),'body':json.loads(r['body'])} for r in rows],'offset':offset,'limit':limit}

    def artifacts(self,offset=0,limit=50):
        with self.db() as db:
            rows=[dict(r) for r in db.execute('SELECT * FROM artifacts ORDER BY created DESC,sha LIMIT ? OFFSET ?',(limit,offset))]
            total=db.execute('SELECT COUNT(*) FROM artifacts').fetchone()[0]
        return {'artifacts':rows,'offset':offset,'limit':limit,'total':total}

    def fragments(self,offset=0,limit=50):
        """Latest declaration per model + stable feature id, with immutable history.

        Latest is a revision selection, never evidence promotion. Negative revisions
        replace the current view while earlier observations remain accessible.
        """
        with self.db() as db:
            base='''SELECT g.*,a.created,ROW_NUMBER() OVER
                (PARTITION BY g.model,g.id ORDER BY a.created DESC,g.sha DESC) AS revision_rank
                FROM graph_objects g JOIN artifacts a ON a.sha=g.sha WHERE g.type='feature' '''
            total=db.execute('SELECT COUNT(*) FROM ('+base+') WHERE revision_rank=1').fetchone()[0]
            rows=db.execute('SELECT * FROM ('+base+') WHERE revision_rank=1 ORDER BY created DESC,id LIMIT ? OFFSET ?',(limit,offset)).fetchall()
            features=[]
            for r in rows:
                history=[{'sha':h['sha'],'level':json.loads(h['body'])['evidence']['level'],'created':h['created']}
                    for h in db.execute('''SELECT g.sha,g.body,a.created FROM graph_objects g JOIN artifacts a ON a.sha=g.sha
                        WHERE g.type='feature' AND g.model=? AND g.id=? ORDER BY a.created DESC,g.sha DESC''',(r['model'],r['id']))]
                features.append({'sha':r['sha'],'id':r['id'],'model_snapshot':r['model'],'body':json.loads(r['body']),'history':history})
        return {'schema':'research_fragments.v1','features':features,'total':total,'offset':offset,'limit':limit}

    def workpackages(self):
        with self.db() as db:
            rows=db.execute('''SELECT * FROM (SELECT w.*,a.created,ROW_NUMBER() OVER
                (PARTITION BY w.run_id ORDER BY a.created DESC,w.sha DESC) AS revision_rank
                FROM workpackage_revisions w JOIN artifacts a ON a.sha=w.sha)
                WHERE revision_rank=1 ORDER BY created DESC''').fetchall()
        return {'schema':'research_workpackages.v1','workpackages':[{'sha':r['sha'],'body':self.artifact(r['sha'])} for r in rows]}

    def artifact(self, sha):
        if len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):raise ValueError('Invalid sha256')
        raw=(self.root/'objects'/sha[:2]/(sha+'.json')).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Artifact checksum mismatch')
        return json.loads(raw)


def services():
    if 'distributed_service' in sys.modules:
        return sys.modules['distributed_service']
    from . import distributed_service as D
    return D


def library():
    D=services()
    return Library(Path(D._DATA_DIR)/'research_library')


def sync():
    D=services()
    lib=library()
    result=lib.scan(Path(os.environ.get('AI2050_RESEARCH_DIR',str(Path(__file__).resolve().parents[1]/'research'))))
    # Retain computed graph revisions too; a future rebuild may change the displayed atlas.
    snapshots=Path(D._DATA_DIR)/'snapshots'
    if os.environ.get('AI2050_DIST_ALLOW_SYNTHETIC')!='1':
        saved=lib.scan(snapshots,extra_paths=[snapshots/(name+'.json') for name in ('atlas','lang_atlas','coverage','scoreboard')],source_prefix='snapshot:',kind='derived_snapshot')
        result['new_revisions']+=saved['new_revisions'];result['errors']+=saved['errors']
    # Uploads are already immutable in the authoritative center CAS; retain their metadata and pointer.
    rows=D._db().execute('''SELECT r.sha,r.tm_id,r.model_id,r.model_rev,r.kind,r.summary_json,r.design_sha,
        a.status AS audit_status,a.reviewer AS audit_reviewer,a.note AS audit_note,a.checks_json AS audit_checks,a.created_at AS audit_at
        FROM results r LEFT JOIN audits a ON r.sha=a.result_sha WHERE r.kind=?''',('real',)).fetchall()
    for row in rows:
        try:
            doc=dict(row);doc['summary']=json.loads(doc.pop('summary_json') or '{}')
            lib.ingest(doc,'center:'+doc['sha'],kind='distributed_result')
        except (ValueError,TypeError) as exc:result['errors'].append({'source':'center:'+row['sha'],'error':str(exc)[:240]})
    db=D._db()
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='census_results'").fetchone():
        for row in db.execute('''SELECT r.*,j.snapshot_sha256 AS model_snapshot,j.model_key
                FROM census_results r LEFT JOIN census_jobs j ON r.job_id=j.job_id WHERE r.kind='real' '''):
            try:
                doc=dict(row)
                for key in ('summary_json','validation_json'):doc[key.removesuffix('_json')]=json.loads(doc.pop(key) or '{}')
                lib.ingest(doc,'census:'+doc['lease_id'],kind='census_result')
            except (ValueError,TypeError) as exc:result['errors'].append({'source':'census:'+row['lease_id'],'error':str(exc)[:240]})
    with lib.db() as db:db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)',('sync',json.dumps({**result,'completed_at':time.time()})))
    return result


@router.get('/overview')
def overview(model: str=''):
    return library().summary(model)


@router.get('/progress')
def progress():
    D=services();db=D._db();templates={}
    mapping={'KN':'知识网络','ENT':'知识网络','RS':'分析推理','SQ':'分析推理','TK':'分析推理','GR':'语法体系'}
    for tpl in db.execute('SELECT tm_id,design_sha,bundle_dir FROM templates'):
        if tpl['tm_id'].endswith('-P'):continue
        try:
            corpus=json.loads((Path(tpl['bundle_dir'])/'corpus.json').read_text(encoding='utf-8'))
            family=str(corpus.get('family_id') or tpl['tm_id'].removeprefix('TG-'))
            group=mapping.get(family.split('-')[0])
            n=int(corpus['shards'])
            if group and n>0:templates[tpl['tm_id']]={'design':tpl['design_sha'],'family':family,'name':corpus.get('name',family),'group':group,'planned':n}
        except (OSError,ValueError,KeyError,TypeError):continue
    buckets={};seen={}
    for r in db.execute("SELECT r.*,a.status AS audit_status FROM results r LEFT JOIN audits a ON r.sha=a.result_sha WHERE r.kind='real'"):
        tpl=templates.get(r['tm_id'])
        if not tpl or r['design_sha']!=tpl['design']:continue
        s=json.loads(r['summary_json'] or '{}');precision=(s.get('_upload_meta') or {}).get('precision') or s.get('precision') or 'unknown'
        key=json.dumps([r['model_id'],r['model_rev'],precision])
        snapshot_verified=isinstance(r['model_rev'],str) and len(r['model_rev'])==64 and all(c in '0123456789abcdef' for c in r['model_rev'])
        if key not in buckets:
            buckets[key]={'id':key,'model':r['model_id'],'snapshot':r['model_rev'],'snapshot_verified':snapshot_verified,'precision':precision,'directions':{g:{'done':0,'planned':sum(t['planned'] for t in templates.values() if t['group']==g)} for g in mapping.values()},'templates':[{**t,'tm_id':tm,'done':0} for tm,t in templates.items()]}
            seen[key]=set()
        shard=s.get('shard');canary=(s.get('_canary') or {}).get('status')
        if not snapshot_verified or precision=='unknown' or r['audit_status']!='passed' or canary not in ('ok','reference'):continue
        try:
            if isinstance(shard,str) and '/' in shard:
                i,n=map(int,shard.split('/'))
                if n!=tpl['planned']:continue
            elif tpl['planned']==1 and shard in ('',None):i=0
            elif type(shard) is int:i=shard
            else:continue
        except (TypeError,ValueError):continue
        identity=(r['tm_id'],i)
        if 0<=i<tpl['planned'] and identity not in seen[key]:
            seen[key].add(identity);buckets[key]['directions'][tpl['group']]['done']+=1
            next(t for t in buckets[key]['templates'] if t['tm_id']==r['tm_id'])['done']+=1
    return {'buckets':list(buckets.values()),'definition':'正式冻结模板（排除-P试跑）、完整快照SHA256与已知精度桶，金丝雀通过且审计passed的唯一分片 / 计划分片；旧cfg修订标识待核验，不计完成；不代表破解比例'}


@router.get('/fragments')
def fragments(offset:int=Query(0,ge=0),limit:int=Query(50,ge=1,le=1000)):
    return library().fragments(offset,limit)


@router.get('/workpackages')
def workpackages():
    return library().workpackages()


@router.get('/graph')
def graph(model: str='',offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=1000)):
    return library().graph(model,offset,limit)


@router.get('/artifacts/{sha}')
def artifact(sha:str):
    try:return library().artifact(sha)
    except (ValueError,FileNotFoundError) as exc:raise HTTPException(404,str(exc)) from exc


@router.get('/artifacts')
def artifacts(offset:int=Query(0,ge=0),limit:int=Query(50,ge=1,le=1000)):
    return library().artifacts(offset,limit)


@router.post('/ingest')
def ingest(doc:dict,x_admin_token:str=Header(default=''),authorization:str=Header(default='')):
    D=services()
    if x_admin_token:D._auth_admin(x_admin_token)
    else:D._auth_call(D._accounts.require_admin,authorization)
    try:return library().ingest(doc,'registered',kind='registered')
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@router.post('/submissions')
def submit(doc:dict,authorization:str=Header(default='')):
    D=services();user=D._auth_call(D._accounts.require_user,authorization)
    snapshot=doc.get('model_snapshot','')
    if doc.get('schema')!='research_submission.v1' or doc.get('kind')!='real' or not isinstance(doc.get('report'),dict) or not doc.get('run_id'):
        raise HTTPException(422,'提交须包含 research_submission.v1、真实运行、报告和运行ID')
    if not isinstance(snapshot,str) or len(snapshot)!=64 or any(c not in '0123456789abcdef' for c in snapshot):raise HTTPException(422,'提交须包含完整模型快照哈希')
    if doc.get('feature_cards') or doc.get('graph_edges'):raise HTTPException(422,'贡献报告不直接登记图谱对象；需研究审核后单独登记')
    try:
        result=library().ingest(doc,'account:'+user['user_id'],kind='contributed_unreviewed')
        return {**result,'verification':'unreviewed'}
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@router.post('/sync')
def refresh(x_admin_token:str=Header(default=''),authorization:str=Header(default='')):
    D=services()
    if x_admin_token:D._auth_admin(x_admin_token)
    else:D._auth_call(D._accounts.require_admin,authorization)
    with _lock:return sync()


def registry():
    from shared.dist_protocol.research_registry import Registry
    return Registry(library())


@router.get('/catalog')
def catalog():
    return registry().catalog()


def _registry_admin(x_admin_token,authorization):
    D=services()
    if x_admin_token:D._auth_admin(x_admin_token)
    else:D._auth_call(D._accounts.require_admin,authorization)


@router.post('/catalog/register')
def catalog_register(doc:dict,x_admin_token:str=Header(default=''),authorization:str=Header(default='')):
    _registry_admin(x_admin_token,authorization)
    try:return registry().register(doc)
    except (ValueError,FileNotFoundError) as exc:raise HTTPException(422,str(exc)) from exc


@router.post('/reviews')
def review_create(doc:dict,x_admin_token:str=Header(default=''),authorization:str=Header(default='')):
    _registry_admin(x_admin_token,authorization)
    try:return registry().create_review(doc)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@router.post('/reviews/{campaign_id}/next')
def review_next(campaign_id:str,authorization:str=Header(default='')):
    D=services();actor=D._auth_call(D._accounts.require_user,authorization)
    try:return {'item':registry().next_review(campaign_id,actor['user_id'])}
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@router.post('/reviews/{campaign_id}/{item_id}')
def review_submit(campaign_id:str,item_id:str,doc:dict,authorization:str=Header(default='')):
    D=services();actor=D._auth_call(D._accounts.require_user,authorization)
    try:return registry().submit_review(campaign_id,item_id,actor['user_id'],doc)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@router.post('/ai-reviews')
def ai_review_create(doc:dict,x_admin_token:str=Header(default=''),authorization:str=Header(default='')):
    _registry_admin(x_admin_token,authorization)
    try:return registry().create_ai_review(doc)
    except (ValueError,FileNotFoundError) as exc:raise HTTPException(422,str(exc)) from exc


@router.post('/ai-reviews/{campaign_id}')
def ai_review_submit(campaign_id:str,doc:dict,x_admin_token:str=Header(default=''),authorization:str=Header(default='')):
    _registry_admin(x_admin_token,authorization)
    try:return registry().submit_ai_review(campaign_id,doc)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
