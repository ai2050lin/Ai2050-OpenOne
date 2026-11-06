"""Dataset lineage, scoped claims and blinded review, backed by immutable CAS.

Registry counts describe records, never solved-mechanism percentages. A human
label requires explicit attestation; two accounts are not proof of independence.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import secrets
import sqlite3
import time


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def bounded(value,name,n=200):
    if not isinstance(value,str) or not value.strip() or len(value)>n:raise ValueError(f'invalid {name}')
    return value


class Registry:
    def __init__(self,library):
        self.lib=library
        with self.lib.db() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS research_registry(kind TEXT,id TEXT,sha TEXT,created REAL,
                PRIMARY KEY(kind,id,sha));
              CREATE TABLE IF NOT EXISTS blind_campaigns(id TEXT PRIMARY KEY,sha TEXT,salt TEXT,created REAL,body TEXT);
              CREATE TABLE IF NOT EXISTS blind_items(campaign TEXT,item_id TEXT,body TEXT,
                PRIMARY KEY(campaign,item_id));
              CREATE TABLE IF NOT EXISTS blind_assignments(campaign TEXT,item_id TEXT,reviewer TEXT,expires REAL,
                PRIMARY KEY(campaign,item_id,reviewer));
              CREATE TABLE IF NOT EXISTS blind_labels(campaign TEXT,item_id TEXT,reviewer TEXT,sha TEXT,source_type TEXT,
                created REAL,body TEXT,PRIMARY KEY(campaign,item_id,reviewer));
              CREATE TABLE IF NOT EXISTS external_ai_campaigns(id TEXT PRIMARY KEY,sha TEXT,body TEXT);
              CREATE TABLE IF NOT EXISTS external_ai_ratings(campaign TEXT,rater TEXT,sha TEXT,body TEXT,
                PRIMARY KEY(campaign,rater));
            ''')

    def register(self,doc):
        schema=doc.get('schema');kind={'dataset_version.v1':'dataset','research_claim.v1':'claim',
            'research_feature_alias.v1':'alias'}.get(schema)
        if not kind or doc.get('kind')!='real':raise ValueError('unsupported real registry schema')
        identity=bounded(doc.get('id'),'id',300)
        if kind=='dataset':
            for key in ['dataset_id','version','source','license','label_status','role']:bounded(doc.get(key),key,2000)
            if not sha(doc.get('content_sha256')):raise ValueError('content_sha256 required')
            groups=doc.get('source_groups')
            if not isinstance(groups,list) or not groups or any(not isinstance(x,str) or not x for x in groups) or len(groups)!=len(set(groups)):raise ValueError('unique source groups required')
            if type(doc.get('sample_count')) is not int or doc['sample_count']<len(groups):raise ValueError('invalid sample count')
            if doc['role'] not in ['natural_discovery','controlled_mechanism','capability_monitor','counterexample']:raise ValueError('invalid data role')
        else:
            if not sha(doc.get('model_snapshot')):raise ValueError('full model snapshot required')
            proof=self.lib.artifact(doc.get('artifact_sha256',''))
            if proof.get('model_snapshot')!=doc['model_snapshot']:raise ValueError('proof model differs')
            bounded(doc.get('canonical_feature_id'),'canonical_feature_id',300)
            if kind=='claim':
                for k in ['external_pattern_id','outcome_metric','scope','status']:bounded(doc.get(k),k,4000)
                if doc['status'] not in ['supported_in_scope','not_supported','inconclusive','not_measured']:raise ValueError('invalid claim status')
                datasets=doc.get('dataset_versions')
                if not isinstance(datasets,list) or not datasets:raise ValueError('claim dataset versions required')
                with self.lib.db() as db:
                    for d in datasets:
                        if not db.execute("SELECT 1 FROM research_registry WHERE kind='dataset' AND id=?",(d,)).fetchone():raise ValueError('unregistered dataset')
            else:
                if doc.get('relation') not in ['same_extracted_feature','member_locator']:raise ValueError('explicit alias relation required')
                card=next((c for c in proof.get('feature_cards',[]) if c['id']==doc.get('legacy_feature_id')),None)
                if not card:raise ValueError('legacy feature missing in proof')
                if doc.get('relation')=='member_locator' and not card.get('scope',{}).get('representative_unit_only'):raise ValueError('member alias must name a declared representative')
        ref=self.lib.ingest(doc,'registry:'+identity,kind='registered')['sha256']
        with self.lib.db() as db:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT sha FROM research_registry WHERE kind=? AND id=? ORDER BY created DESC LIMIT 1',(kind,identity)).fetchone()
            if old:
                previous=self.lib.artifact(old['sha'])
                fixed={'dataset':['dataset_id','version','content_sha256'],
                    'claim':['canonical_feature_id','model_snapshot','external_pattern_id','outcome_metric','scope'],
                    'alias':['canonical_feature_id','model_snapshot','legacy_feature_id','relation']}[kind]
                if any(previous.get(k)!=doc.get(k) for k in fixed):raise ValueError('identity collision: create a new version or scoped claim')
            db.execute('INSERT OR IGNORE INTO research_registry VALUES(?,?,?,?)',(kind,identity,ref,time.time()))
        return {'id':identity,'sha256':ref,'type':kind}

    def catalog(self):
        with self.lib.db() as db:rows=db.execute('SELECT * FROM research_registry ORDER BY created,id').fetchall()
        all_rows=[{**dict(r),'body':self.lib.artifact(r['sha'])} for r in rows]
        latest={}
        for row in all_rows:latest[(row['kind'],row['id'])]=row
        datasets=[r for (k,_),r in latest.items() if k=='dataset']
        claims=[r for (k,_),r in latest.items() if k=='claim']
        aliases=[r for (k,_),r in latest.items() if k=='alias']
        return {'schema':'research_catalog.v1','datasets':datasets,'claims':claims,'aliases':aliases,
            'versions':len(all_rows),'reviews':self.review_summary(),
            'note':'主张按指标和适用范围区分；版本、来源数与定位卡数量不是机制破解比例。'}

    def create_review(self,doc):
        if doc.get('schema')!='blind_review_campaign.v1' or doc.get('kind')!='real':raise ValueError('invalid review campaign')
        cid=bounded(doc.get('id'),'campaign id')
        if doc.get('required_human_reviews')!=2:raise ValueError('two human reviews required')
        for k in ['rubric_version','purpose']:bounded(doc.get(k),k,2000)
        items=doc.get('items')
        if not isinstance(items,list) or not 1<=len(items)<=10000:raise ValueError('invalid review items')
        keys=set()
        for item in items:
            key=bounded(item.get('key'),'private item key',300)
            if key in keys:raise ValueError('duplicate review item')
            keys.add(key);bounded(item.get('text'),'review text',30000)
            if not isinstance(item.get('facts'),dict):raise ValueError('structured facts required for content review')
        ref=digest(doc)
        manifest={'schema':'blind_review_manifest.v1','kind':'real','campaign_id':cid,'campaign_sha256':ref,
            'items':len(items),'rubric_version':doc['rubric_version'],'required_human_reviews':2,
            'note':'Assignment conditions and submitted labels stay private until explicit unblinding; review progress is public.'}
        self.lib.ingest(manifest,'review-campaign:'+cid,kind='registered')
        with self.lib.db() as db:
            db.execute('BEGIN IMMEDIATE')
            previous=db.execute('SELECT sha FROM blind_campaigns WHERE id=?',(cid,)).fetchone()
            if previous:
                if previous['sha']!=ref:raise ValueError('campaign immutable; create a new version')
                return {'id':cid,'sha256':ref,'items':len(items)}
            salt=secrets.token_hex(32)
            db.execute('INSERT INTO blind_campaigns VALUES(?,?,?,?,?)',(cid,ref,salt,time.time(),json.dumps(doc,ensure_ascii=False)))
            for item in items:
                opaque=hmac.new(bytes.fromhex(salt),item['key'].encode(),'sha256').hexdigest()[:24]
                # Conditions, generator, run ids, labels and source identities stay private.
                public={'item_id':opaque,'text':item['text'],'facts':item['facts'],'rubric_version':doc['rubric_version']}
                db.execute('INSERT INTO blind_items VALUES(?,?,?)',(cid,opaque,json.dumps(public,ensure_ascii=False)))
        return {'id':cid,'sha256':ref,'items':len(items)}

    def next_review(self,cid,reviewer):
        bounded(reviewer,'reviewer')
        with self.lib.db() as db:
            db.execute('BEGIN IMMEDIATE')
            campaign=db.execute('SELECT * FROM blind_campaigns WHERE id=?',(cid,)).fetchone()
            if not campaign:raise ValueError('unknown campaign')
            doc=json.loads(campaign['body'])
            if reviewer in doc.get('excluded_reviewer_accounts',[]):raise ValueError('reviewer excluded by campaign')
            rows=db.execute('SELECT * FROM blind_items WHERE campaign=? ORDER BY item_id',(cid,)).fetchall()
            for row in rows:
                labels=db.execute('SELECT reviewer FROM blind_labels WHERE campaign=? AND item_id=? AND source_type=?',(cid,row['item_id'],'human_attested')).fetchall()
                if reviewer in {x['reviewer'] for x in labels} or len(labels)>=2:continue
                # Active reservations count toward two slots, preventing accidental oversubscription.
                active=db.execute('SELECT reviewer FROM blind_assignments WHERE campaign=? AND item_id=? AND expires>?',(cid,row['item_id'],time.time())).fetchall()
                owners={x['reviewer'] for x in active}|{x['reviewer'] for x in labels}
                if len(owners)>=2 and reviewer not in owners:continue
                db.execute('INSERT OR REPLACE INTO blind_assignments VALUES(?,?,?,?)',(cid,row['item_id'],reviewer,time.time()+3600))
                return {**json.loads(row['body']),'campaign_id':cid,'instructions':doc.get('instructions','')}
        return None

    def submit_review(self,cid,item_id,reviewer,body):
        if body.get('style') not in ['formal','spoken','mixed','uncertain']:raise ValueError('invalid style')
        if body.get('meaning') not in ['preserved','changed','uncertain']:raise ValueError('invalid meaning')
        if type(body.get('naturalness')) is not int or not 1<=body['naturalness']<=5:raise ValueError('naturalness 1..5 required')
        if body.get('human_attestation') is not True:raise ValueError('AI outputs cannot count as human review')
        note=body.get('note','')
        if not isinstance(note,str) or len(note)>3000:raise ValueError('invalid note')
        with self.lib.db() as db:
            db.execute('BEGIN IMMEDIATE')
            assignment=db.execute('SELECT * FROM blind_assignments WHERE campaign=? AND item_id=? AND reviewer=?',(cid,item_id,reviewer)).fetchone()
            if not assignment:raise ValueError('no assignment for this reviewer')
            label={'schema':'blind_human_label.v1','kind':'real','campaign_id':cid,'item_id':item_id,'reviewer_account':reviewer,
                'source_type':'human_attested',**{k:body[k] for k in ['style','meaning','naturalness']},'note':note}
            ref=digest(label)
            existing=db.execute('SELECT sha FROM blind_labels WHERE campaign=? AND item_id=? AND reviewer=?',(cid,item_id,reviewer)).fetchone()
            if existing:
                if existing['sha']!=ref:raise ValueError('label immutable; adjudication is separate')
                return {'sha256':ref,'replay':True}
            if assignment['expires']<time.time():raise ValueError('review assignment expired')
            db.execute('INSERT INTO blind_labels VALUES(?,?,?,?,?,?,?)',(cid,item_id,reviewer,ref,'human_attested',time.time(),json.dumps(label,ensure_ascii=False)))
        return {'sha256':ref,'replay':False,'mechanism_evidence_promoted':False}

    def review_summary(self):
        summaries=[]
        with self.lib.db() as db:
            campaigns=db.execute('SELECT * FROM blind_campaigns ORDER BY created DESC').fetchall()
            for campaign in campaigns:
                cid=campaign['id']
                items=db.execute('SELECT item_id FROM blind_items WHERE campaign=?',(cid,)).fetchall()
                paired=disagreements=0
                for item in items:
                    labels=db.execute('SELECT body FROM blind_labels WHERE campaign=? AND item_id=? AND source_type=?',(cid,item['item_id'],'human_attested')).fetchall()
                    if len(labels)>=2:
                        paired+=1;values=[json.loads(x['body']) for x in labels]
                        disagreements+=int(len({(x['style'],x['meaning']) for x in values})>1)
                summaries.append({'id':cid,'items':len(items),'double_reviewed':paired,'disagreements':disagreements,
                    'status':'awaiting_adjudication' if paired==len(items) and disagreements else 'double_review_complete' if paired==len(items) else 'awaiting_human_reviews',
                    'independence':'two distinct authenticated accounts with human attestation; independence not automatically established',
                    'evidence_promoted':False})
        with self.lib.db() as db:
            for campaign in db.execute('SELECT * FROM external_ai_campaigns ORDER BY id'):
                c=json.loads(campaign['body']);ratings=db.execute('SELECT * FROM external_ai_ratings WHERE campaign=?',(campaign['id'],)).fetchall()
                disagree=None
                if len(ratings)==2:
                    a,b=[{x['id']:x for x in json.loads(r['body'])['ratings']} for r in ratings]
                    disagree=sum((a[k]['style'],a[k]['meaning'])!=(b[k]['style'],b[k]['meaning']) for k in a)
                summaries.append({'id':campaign['id'],'source_type':'external_ai','items':c['n_items'],
                    'completed_raters':len(ratings),'rater_models':c['rater_models'],'disagreements':disagree,
                    'rating_artifacts':[{'rater_model':r['rater'],'sha256':r['sha']} for r in ratings],
                    'status':'ai_double_review_complete' if len(ratings)==2 else 'awaiting_ai_reviews',
                    'evidence_promoted':False,'note':'两种AI独立评分；不是人类标注，也不是机制证据自动晋级。'})
        return summaries

    def create_ai_review(self,doc):
        if doc.get('schema')!='external_ai_review_campaign.v1' or doc.get('kind')!='real':raise ValueError('invalid AI campaign')
        cid=bounded(doc.get('id'),'AI campaign id');raters=doc.get('rater_models',[])
        if len(raters)!=2 or len(set(raters))!=2 or any(not isinstance(x,str) or not x for x in raters):raise ValueError('two distinct named AI models required')
        if not sha(doc.get('input_sha256')):raise ValueError('frozen input file hash required')
        blind=self.lib.artifact(doc.get('blind_artifact_sha256',''));items=blind.get('records',[])
        if not items or len(items)!=doc.get('n_items'):raise ValueError('blind item count differs')
        ref=self.lib.ingest(doc,'AI-campaign:'+cid,kind='registered')['sha256']
        with self.lib.db() as db:
            old=db.execute('SELECT sha FROM external_ai_campaigns WHERE id=?',(cid,)).fetchone()
            if old and old['sha']!=ref:raise ValueError('AI campaign immutable')
            db.execute('INSERT OR IGNORE INTO external_ai_campaigns VALUES(?,?,?)',(cid,ref,json.dumps(doc,ensure_ascii=False)))
        return {'id':cid,'sha256':ref}

    def submit_ai_review(self,cid,doc):
        with self.lib.db() as db:row=db.execute('SELECT body FROM external_ai_campaigns WHERE id=?',(cid,)).fetchone()
        if not row:raise ValueError('unknown AI campaign')
        campaign=json.loads(row['body']);rater=doc.get('rater_model')
        if doc.get('schema')!='external_ai_ratings.v1' or rater not in campaign['rater_models'] or doc.get('input_sha256')!=campaign['input_sha256']:raise ValueError('AI rater or input hash differs')
        blind=self.lib.artifact(campaign['blind_artifact_sha256']);expected={x['id'] for x in blind['records']}
        ratings=doc.get('ratings',[])
        if len(ratings)!=len(expected) or {x.get('id') for x in ratings}!=expected:raise ValueError('incomplete or duplicate AI ratings')
        for x in ratings:
            if x.get('style') not in ['formal','spoken','mixed','uncertain'] or x.get('meaning') not in ['preserved','changed','uncertain','not_applicable']:raise ValueError('invalid AI label')
            if type(x.get('naturalness')) is not int or not 1<=x['naturalness']<=5:raise ValueError('invalid naturalness')
            bounded(x.get('reason'),'AI reason',3000)
        ref=self.lib.ingest(doc,'AI-review:'+cid+':'+rater,kind='contributed_unreviewed')['sha256']
        with self.lib.db() as db:
            db.execute('BEGIN IMMEDIATE');old=db.execute('SELECT sha FROM external_ai_ratings WHERE campaign=? AND rater=?',(cid,rater)).fetchone()
            if old and old['sha']!=ref:raise ValueError('AI labels immutable')
            db.execute('INSERT OR IGNORE INTO external_ai_ratings VALUES(?,?,?,?)',(cid,rater,ref,json.dumps(doc,ensure_ascii=False)))
        return {'sha256':ref,'source_type':'external_ai','human_labels':False,'evidence_promoted':False}
