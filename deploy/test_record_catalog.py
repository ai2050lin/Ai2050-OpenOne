"""Read-only study descriptions, independent of live run identity and evidence admission."""
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = {'goodfire': ROOT / 'shared/dist_protocol/test_records/pilot.json',
        'history': ROOT / 'deploy/test_record_history.json'}


@lru_cache(maxsize=6)
def _document(path, mtime_ns, size):
    return json.loads(path.read_bytes())


def catalog_version(scope):
    from deploy.workspace_service import published_root
    local = published_root() / f'{scope}-records.json'
    path = local if local.is_file() else DATA[scope]
    stat = path.stat()
    return path,stat.st_mtime_ns,stat.st_size


def document(scope):
    version = catalog_version(scope)
    return _goodfire_document(version, annotation_version()) if scope == 'goodfire' else _document(*version)


def catalog(scope):
    return document(scope)['records']


def annotation_version():
    from deploy.workspace_service import published_root
    local = published_root() / 'goodfire-evidence.json'
    path = local if local.is_file() else Path(__file__).with_name('workspace_data') / local.name
    stat = path.stat()
    return path, stat.st_mtime_ns, stat.st_size


def annotations(version=None):
    doc = _document(*(version or annotation_version()))
    if doc.get('schema') != 'goodfire_record_annotations.v1':
        raise ValueError('unsupported Goodfire annotation schema')
    return doc


@lru_cache(maxsize=4)
def _goodfire_document(version, revision):
    doc = _document(*version)
    items = {row['id']:row for row in annotations(revision)['records']}
    rows = []
    for row in doc['records']:
        item = items.get(row['id'])
        if item and item.get('phase_sha256') and item['phase_sha256'] == row.get('phase_sha256'):
            row = {**row, **{key:item[key] for key in ('earlier','novelty','case_state','case_note','proofs') if key in item}}
        rows.append(row)
    return {**doc,'records':rows}


def terms(value):
    """Overlapping Chinese pairs and whole model/head IDs, without an external service."""
    tokens = set()
    for word in re.findall(r'[a-z0-9][a-z0-9_./-]*|[\u3400-\u9fff]+', value.casefold()):
        if '\u3400' <= word[0] <= '\u9fff' and len(word)>1:
            tokens.update(word[i:i+2] for i in range(len(word)-1))
        else:
            tokens.add(word)
    return tokens


@lru_cache(maxsize=4)
def _index(version, revision):
    entries = []
    freq = Counter()
    doc = _goodfire_document(version, revision) if revision else _document(*version)
    for row in doc['records']:
        label = ' '.join(str(row.get(k) or '') for k in ('id','phase','title','question','type','objective','model'))
        text = label+' '+' '.join(str(row.get(k) or '') for k in ('principle','result','counter','gate'))
        words = terms(text)
        freq.update(words)
        entries.append((row, label.casefold(), words))
    return entries, freq


def search(q='', scope='goodfire', offset=0, limit=30):
    entries, freq = _index(catalog_version(scope), annotation_version() if scope == 'goodfire' else None)
    query = q.strip().casefold()
    if not query:
        found = [(row,[]) for row,_,_ in entries]
    else:
        tokens = terms(query)
        scored = []
        for row,label,words in entries:
            common = words & tokens
            exact = query in label
            if not common and not exact:
                continue
            weight = sum(1/(1+freq[t])**0.5 for t in common)
            scored.append((int(exact),weight,row['id'],row,sorted(common,key=lambda t:(freq[t],t))[:6]))
        scored.sort(key=lambda item:(-item[0],-item[1],item[2]))
        found = [(row,why) for _,_,_,row,why in scored]
    return {'total':len(found),'items':[dict(row, match_terms=why) for row,why in found[offset:offset+limit]],
            'scope':scope,'coverage':'indexed_records_only'}


def similar(plan):
    q = ' '.join([plan['title'],plan['objective'],*plan['family_ids']])[:1000]
    scopes = ['goodfire'] + (['history'] if DATA['history'].exists() else [])
    return [dict(id=r['id'],title=r['title'],type=r['type'],phase=r.get('phase'),terms=r['match_terms'],
                 new_validation=r.get('novelty'))
            for scope in scopes for r in search(q,scope,limit=7)['items']]


def detail(key):
    for scope in DATA:
        if DATA[scope].exists():
            for row in catalog(scope):
                if row['id']==key:
                    return row
    raise FileNotFoundError('historical record not found')

